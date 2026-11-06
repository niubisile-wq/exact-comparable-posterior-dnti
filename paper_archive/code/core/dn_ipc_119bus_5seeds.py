# -*- coding: utf-8 -*-
"""IP-C 119-bus: 5种子从头训练 → 补充验证（报告为supplemental）"""
import os, re, copy, time, warnings
import numpy as np
import torch
import torch.nn as nn
import pandapower as pp
import networkx as nx
import scipy.stats as stats
warnings.filterwarnings('ignore')

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = r"<LOCAL_WORKSPACE>"
K_FIXED  = 25; N_STEPS = 100000; BATCH = 256; LR = 3e-4
SIGMA    = 0.009; LF_MIN, LF_MAX = 0.8, 1.2; N_LF = 101
MISSING_RATES = [0.0, 0.05, 0.10, 0.20, 0.30]
ALL_SEEDS = [42, 123, 456, 789, 2024]
DATA_PATH = f"{SAVE_DIR}\\ElectricalSystemsDataForReconfiguration\\SystemData_119.txt"
print(f"Device: {DEVICE}  IP-C-119bus 5seeds")

def build_ieee119():
    with open(DATA_PATH, encoding='utf-8') as f:
        content = f.read()
    vnom = float(re.search(r"Vnominal\s*=\s*([\d.]+)", content).group(1))
    bus_data = {}
    in_bus = False
    for line in content.split("\n"):
        s = line.strip()
        if re.match(r"Bus\s+PD", s): in_bus = True; continue
        if in_bus:
            nums = re.findall(r"[-+]?\d*\.?\d+", s)
            if len(nums) >= 3: bus_data[int(nums[0])] = (float(nums[1]), float(nums[2]))
            elif s == "": in_bus = False
    branch_lines = []
    for line in content.split("\n"):
        s = line.strip()
        if re.match(r"^\d+\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+", s):
            nums = re.findall(r"[-+]?\d*\.?\d+", s)
            if len(nums) >= 5:
                branch_lines.append((int(nums[0]),int(nums[1]),int(nums[2]),float(nums[3]),float(nums[4])))
    sorted_br = sorted(branch_lines, key=lambda x: x[2])
    normal_br = [b for b in sorted_br if b[2] <= 118]
    tie_br    = [b for b in sorted_br if b[2] > 118]
    all_nodes = sorted(set(n for b in normal_br+tie_br for n in (b[0],b[1])))
    node2idx  = {n: i for i, n in enumerate(all_nodes)}
    N_BUS_    = len(all_nodes)
    net = pp.create_empty_network()
    for n in all_nodes: pp.create_bus(net, vn_kv=vnom)
    for b in normal_br:
        pp.create_line_from_parameters(net, node2idx[b[0]], node2idx[b[1]], 1,
                                       max(b[3],0.0001), max(b[4],0.0001), 0, 9999, in_service=True)
    for b in tie_br:
        pp.create_line_from_parameters(net, node2idx[b[0]], node2idx[b[1]], 1,
                                       b[3], b[4], 0, 9999, in_service=False)
    for bus_net, (p, q) in bus_data.items():
        if (p > 0 or q > 0) and bus_net in node2idx:
            pp.create_load(net, node2idx[bus_net], p/1000, q/1000)
    pp.create_ext_grid(net, node2idx[0], vm_pu=1.0)
    ne = [(node2idx[b[0]], node2idx[b[1]]) for b in normal_br]
    te = [(node2idx[b[0]], node2idx[b[1]]) for b in tie_br]
    return net, ne, te, N_BUS_

def enum_topos(ne, te, n):
    G = nx.Graph(); G.add_edges_from(ne)
    topos = [list(range(len(ne)))]; seen = {frozenset(range(len(ne)))}
    for ti2, tie in enumerate(te):
        try: path = nx.shortest_path(G, tie[0], tie[1])
        except: continue
        for i in range(len(path)-1):
            oe = frozenset([path[i], path[i+1]])
            ni = [j for j,e in enumerate(ne) if frozenset(e) != oe]
            key = frozenset(ni)
            if key in seen: continue
            edges = [ne[j] for j in ni] + [tie]
            Gt = nx.Graph(); Gt.add_nodes_from(range(n)); Gt.add_edges_from(edges)
            if nx.is_connected(Gt) and nx.is_tree(Gt):
                seen.add(key); topos.append(ni + [len(ne)+ti2])
    return topos

def run_pf_scaled(net_base, t_raw, ne, te, lf=1.0):
    net = copy.deepcopy(net_base)
    if len(net.load) > 0:
        net.load['p_mw'] *= lf; net.load['q_mvar'] *= lf
    n_ne = len(ne)
    active_ne = {x for x in t_raw if x < n_ne}
    active_te = {x-n_ne for x in t_raw if x >= n_ne}
    for li in range(n_ne):
        net.line.at[net.line.index[li], 'in_service'] = (li in active_ne)
    for li in range(len(te)):
        net.line.at[net.line.index[n_ne+li], 'in_service'] = (li in active_te)
    try:
        pp.runpp(net, algorithm='bfsw', numba=False, max_iteration=100, tolerance_mva=1e-6)
        if net.converged: return net.res_bus.vm_pu.values.copy()
    except: pass
    return None

class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,d),nn.LayerNorm(d),nn.GELU(),nn.Linear(d,d),nn.LayerNorm(d))
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.net(x))

class LoadAwareNRE(nn.Module):
    def __init__(self, n_topo, n_bus):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_bus*3,512),nn.LayerNorm(512),nn.GELU())
        self.res1 = ResBlock(512); self.res2 = ResBlock(512); self.res3 = ResBlock(512)
        self.head = nn.Sequential(nn.Linear(512,256),nn.LayerNorm(256),nn.GELU(),nn.Linear(256,n_topo))
    def forward(self, x):
        h = self.embed(x); h = self.res1(h); h = self.res2(h); h = self.res3(h); return self.head(h)

# 加载 V_library
print("Loading V_library_119bus.npz...")
dat = np.load(f"{SAVE_DIR}\\v_library_119bus.npz")
V_library = dat['V_library']; base_P_norm = dat['base_P_norm']; lf_grid = dat['lf_grid']
N_TOPOS = V_library.shape[0]; N_BUS = V_library.shape[2]
print(f"V_library: {V_library.shape}  N_TOPOS={N_TOPOS}  N_BUS={N_BUS}")

net119, ne119, te119, _ = build_ieee119()
topos_raw = enum_topos(ne119, te119, N_BUS)

# 加载朴素NRE (119-bus IP1 seed=42)
print("Loading naive NRE (119-bus seed=42)...")
nai_ckpt = torch.load(f"{SAVE_DIR}\\nre_119bus_ip1_seed42.pt", map_location=DEVICE, weights_only=False)
nai_model = LoadAwareNRE(N_TOPOS, N_BUS).to(DEVICE)
nai_model.load_state_dict(nai_ckpt['model_state']); nai_model.eval()

def infer(model, reported, obs_v, lf):
    x = np.zeros(N_BUS*3, dtype=np.float32)
    if len(reported) > 0:
        x[reported]=obs_v; x[N_BUS+reported]=1.0; x[2*N_BUS+reported]=base_P_norm[reported]*lf
    with torch.no_grad():
        logits=model(torch.tensor(x).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
    p=np.exp(logits-logits.max()); p/=p.sum(); return p

def evaluate_ipc(rob_model):
    rob_model.eval()
    rows = {}
    for miss_rate in MISSING_RATES:
        n_miss = int(K_FIXED*miss_rate)
        acc_rob, acc_nai = [], []
        rng = np.random.RandomState(77)
        for _ in range(1000):
            ti=rng.randint(0,N_TOPOS); lf_idx=rng.randint(0,N_LF); lf=lf_grid[lf_idx]
            installed=np.sort(rng.choice(range(1,N_BUS),K_FIXED,replace=False))
            if n_miss > 0:
                miss_idx=rng.choice(len(installed),n_miss,replace=False)
                reported=np.delete(installed,miss_idx)
            else:
                reported=installed
            obs_v=V_library[ti,lf_idx,reported]+rng.normal(0,SIGMA,len(reported))
            acc_rob.append(int(np.argmax(infer(rob_model,reported,obs_v,lf))==ti))
            acc_nai.append(int(np.argmax(infer(nai_model,reported,obs_v,lf))==ti))
        rows[miss_rate] = {'rob': float(np.mean(acc_rob)), 'nai': float(np.mean(acc_nai))}
    return rows

def gen_batch_robust(rng, n):
    xs, ys = [], []
    for _ in range(n):
        ti=rng.randint(0,N_TOPOS); lf_idx=rng.randint(0,N_LF); lf=lf_grid[lf_idx]
        installed=np.sort(rng.choice(range(1,N_BUS),K_FIXED,replace=False))
        miss_rate_tr=rng.uniform(0.0,0.3); n_miss=int(K_FIXED*miss_rate_tr)
        if n_miss>0:
            miss_idx=rng.choice(len(installed),n_miss,replace=False)
            reported=np.delete(installed,miss_idx)
        else:
            reported=installed
        obs_v=V_library[ti,lf_idx,reported]+rng.normal(0,SIGMA,len(reported))
        x=np.zeros(N_BUS*3,dtype=np.float32)
        x[reported]=obs_v; x[N_BUS+reported]=1.0; x[2*N_BUS+reported]=base_P_norm[reported]*lf
        xs.append(x); ys.append(ti)
    return (torch.tensor(np.array(xs),dtype=torch.float32).to(DEVICE),
            torch.tensor(ys,dtype=torch.long).to(DEVICE))

# 确认naive基准
print("Evaluating naive baseline (119-bus)...")
nai_rows = evaluate_ipc(nai_model)
nai10 = nai_rows[0.10]['nai']; nai30 = nai_rows[0.30]['nai']
print(f"Naive @10%={nai10:.3f}  @30%={nai30:.3f}")

# 训练5个种子
all_results = {}
for seed in ALL_SEEDS:
    save_path = f"{SAVE_DIR}\\nre_119bus_ipc_seed{seed}.pt"
    if os.path.exists(save_path):
        print(f"\nSeed {seed}: loading existing model...")
        ckpt = torch.load(save_path, map_location=DEVICE, weights_only=False)
        rob_model = LoadAwareNRE(N_TOPOS, N_BUS).to(DEVICE)
        rob_model.load_state_dict(ckpt['model_state'])
    else:
        print(f"\n{'='*55}\nTraining IP-C 119-bus seed={seed}...")
        torch.manual_seed(seed); np.random.seed(seed)
        rob_model = LoadAwareNRE(N_TOPOS, N_BUS).to(DEVICE)
        opt = torch.optim.AdamW(rob_model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS, eta_min=1e-5)
        loss_fn = nn.CrossEntropyLoss()
        rng_tr = np.random.RandomState(seed)
        rob_model.train(); t0 = time.time()
        for step in range(1, N_STEPS+1):
            xb,yb = gen_batch_robust(rng_tr, BATCH)
            loss = loss_fn(rob_model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step(); scheduler.step()
            if step % 25000 == 0:
                print(f"  step {step:6d}  loss={loss.item():.4f}  {time.time()-t0:.0f}s", flush=True)
        torch.save({'model_state': rob_model.state_dict(), 'seed': seed, 'N_TOPOS': N_TOPOS}, save_path)
    rows = evaluate_ipc(rob_model)
    all_results[seed] = {'rob10': rows[0.10]['rob'], 'rob30': rows[0.30]['rob']}
    print(f"Seed {seed}: rob10={rows[0.10]['rob']:.3f}(d{rows[0.10]['rob']-nai10:+.3f})  "
          f"rob30={rows[0.30]['rob']:.3f}(d{rows[0.30]['rob']-nai30:+.3f})", flush=True)

# 汇总 + Wilcoxon
all_rob10 = [all_results[s]['rob10'] for s in ALL_SEEDS]
all_rob30 = [all_results[s]['rob30'] for s in ALL_SEEDS]
all_d10 = [r-nai10 for r in all_rob10]
all_d30 = [r-nai30 for r in all_rob30]

print(f"\n{'='*60}")
print(f"IP-C-119bus 5-seed SUMMARY  nai10={nai10:.3f}  nai30={nai30:.3f}")
for s,r10,r30,d10,d30 in zip(ALL_SEEDS,all_rob10,all_rob30,all_d10,all_d30):
    print(f"  seed={s:>4}  rob10={r10:.3f}(d{d10:+.3f})  rob30={r30:.3f}(d{d30:+.3f})")
print(f"  delta @10%: mean={np.mean(all_d10):+.3f}±{np.std(all_d10):.3f}  min={np.min(all_d10):+.3f}")
print(f"  delta @30%: mean={np.mean(all_d30):+.3f}±{np.std(all_d30):.3f}  min={np.min(all_d30):+.3f}")

# Wilcoxon (n=5, supplemental)
for label, rob_arr, nai_val in [('@10%', all_rob10, nai10), ('@30%', all_rob30, nai30)]:
    nai_arr = [nai_val]*len(rob_arr)
    stat, p = stats.wilcoxon(rob_arr, nai_arr, alternative='greater')
    verdict = '*' if p < 0.05 else 'ns'
    print(f"  Wilcoxon 119-bus {label}: p={p:.4f} {verdict}  n={len(rob_arr)}")

gate_10 = all(d > 0.05 for d in all_d10)
gate_30 = all(d > 0.05 for d in all_d30)
print(f"\n  Gate @10% all-seed delta>5pp: {'PASS' if gate_10 else 'FAIL'}")
print(f"  Gate @30% all-seed delta>5pp: {'PASS' if gate_30 else 'FAIL'}")

out = ["seed,rob10,nai10,rob30,nai30"]
for s,r10,r30 in zip(ALL_SEEDS,all_rob10,all_rob30):
    out.append(f"{s},{r10:.4f},{nai10:.4f},{r30:.4f},{nai30:.4f}")
with open(f"{SAVE_DIR}\\ipc_119bus_5seed_result.txt", 'w', encoding='utf-8') as f:
    f.write('\n'.join(out) + '\n')
print(f"Saved: ipc_119bus_5seed_result.txt")
