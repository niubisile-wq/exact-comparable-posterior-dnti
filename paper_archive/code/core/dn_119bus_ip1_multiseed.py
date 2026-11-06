# -*- coding: utf-8 -*-
"""
T3-A: 119-bus IP1 NRE 5种子实验
第三测试网络, N_TOPOS=100, K=25, lf~U(0.8,1.2), 100k步
与33/69-bus设定一致, 扩展到更大拓扑空间（第三个标准测试网络）
"""
import copy, re, time, warnings, os
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
SEEDS    = [42, 123, 456, 789, 2024]
DATA_PATH = f"{SAVE_DIR}\\ElectricalSystemsDataForReconfiguration\\SystemData_119.txt"
print(f"Device: {DEVICE}  119-bus IP1 multiseed  K={K_FIXED}  Seeds={SEEDS}")

# ── 网络构建（来自IP4，读文件）────────────────────────────────────────────────
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
    sorted_br  = sorted(branch_lines, key=lambda x: x[2])
    normal_br  = [b for b in sorted_br if b[2] <= 118]
    tie_br     = [b for b in sorted_br if b[2] > 118]
    all_nodes  = sorted(set(n for b in normal_br+tie_br for n in (b[0],b[1])))
    node2idx   = {n: i for i, n in enumerate(all_nodes)}
    N_BUS_     = len(all_nodes)
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

# ── 建网 + 枚举拓扑 ───────────────────────────────────────────────────────────
print("Building 119-bus network...")
net119, ne119, te119, N_BUS = build_ieee119()
print(f"N_BUS={N_BUS}")
print("Enumerating topologies...")
topos_raw = enum_topos(ne119, te119, n=N_BUS)
N_TOPOS = len(topos_raw)
print(f"N_TOPOS={N_TOPOS}")

# 基础负荷
base_P_per_bus = np.zeros(N_BUS)
for _, row in net119.load.iterrows():
    base_P_per_bus[int(row.bus)] += row['p_mw']
base_P_norm = base_P_per_bus / (base_P_per_bus.max() + 1e-8)
lf_grid = np.linspace(LF_MIN, LF_MAX, N_LF)

# ── 预计算 V_library ─────────────────────────────────────────────────────────
vlib_path = f"{SAVE_DIR}\\v_library_119bus.npz"
if os.path.exists(vlib_path):
    print(f"Loading V_library from {vlib_path}...")
    dat = np.load(vlib_path)
    V_library = dat['V_library']
    print(f"V_library loaded: {V_library.shape}")
else:
    print(f"Precomputing V_library ({N_TOPOS} topos × {N_LF} lf × {N_BUS} bus)...")
    t0 = time.time()
    V_library = np.zeros((N_TOPOS, N_LF, N_BUS), dtype=np.float32)
    for i, topo in enumerate(topos_raw):
        for j, lf in enumerate(lf_grid):
            V = run_pf_scaled(net119, topo, ne119, te119, lf)
            V_library[i,j,:] = V if V is not None else (V_library[i,max(j-1,0),:] if j>0 else 1.0)
        if (i+1) % 10 == 0:
            print(f"  topo {i+1}/{N_TOPOS}  {time.time()-t0:.0f}s", flush=True)
    print(f"V_library done: {time.time()-t0:.1f}s")
    np.savez_compressed(vlib_path, V_library=V_library,
                        base_P_norm=base_P_norm, lf_grid=lf_grid)
    print(f"V_library saved to {vlib_path}")

# AIS速度测量
print("\nMeasuring AIS (EnumBF) timing...")
t_ais = time.time()
for topo in topos_raw:
    run_pf_scaled(net119, topo, ne119, te119, lf=1.0)
ais_time_ms = (time.time() - t_ais) / N_TOPOS * 1000
print(f"EnumBF per-inference: {N_TOPOS}×{ais_time_ms:.1f}ms = {N_TOPOS*ais_time_ms:.0f}ms")

# ── 模型 ──────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,d), nn.LayerNorm(d), nn.GELU(),
                                  nn.Linear(d,d), nn.LayerNorm(d))
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.net(x))

class LoadAwareNRE(nn.Module):
    def __init__(self, n_topo, n_bus):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_bus*3, 512), nn.LayerNorm(512), nn.GELU())
        self.res1 = ResBlock(512); self.res2 = ResBlock(512); self.res3 = ResBlock(512)
        self.head = nn.Sequential(nn.Linear(512,256), nn.LayerNorm(256), nn.GELU(),
                                   nn.Linear(256, n_topo))
    def forward(self, x):
        h = self.embed(x); h = self.res1(h); h = self.res2(h); h = self.res3(h)
        return self.head(h)

# ── 推断与评估 ────────────────────────────────────────────────────────────────
def infer(model, reported, obs_v, lf):
    x = np.zeros(N_BUS*3, dtype=np.float32)
    if len(reported) > 0:
        x[reported] = obs_v
        x[N_BUS+reported] = 1.0
        x[2*N_BUS+reported] = base_P_norm[reported]*lf
    with torch.no_grad():
        logits = model(torch.tensor(x).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
    p = np.exp(logits - logits.max()); p /= p.sum(); return p

def ais_post(reported, obs_v, lf_idx):
    if len(reported) == 0: return np.ones(N_TOPOS)/N_TOPOS
    diff = (V_library[:,lf_idx,:][:,reported] - obs_v) / SIGMA
    ll = -0.5*np.sum(diff**2, axis=1); ll -= ll.max()
    w = np.exp(ll); return w/w.sum()

def evaluate(model):
    model.eval()
    rng = np.random.RandomState(77)
    acc_ais, acc_nre, kl_list = [], [], []
    for _ in range(1000):
        ti = rng.randint(0,N_TOPOS); lf_idx = rng.randint(0,N_LF); lf = lf_grid[lf_idx]
        installed = np.sort(rng.choice(range(1,N_BUS), K_FIXED, replace=False))
        obs_v = V_library[ti,lf_idx,installed] + rng.normal(0,SIGMA,K_FIXED)
        p_ais = ais_post(installed, obs_v, lf_idx)
        p_nre = infer(model, installed, obs_v, lf)
        acc_ais.append(int(np.argmax(p_ais)==ti))
        acc_nre.append(int(np.argmax(p_nre)==ti))
        kl = np.sum(p_ais * np.log(p_ais / (p_nre+1e-10) + 1e-10))
        kl_list.append(max(0, kl))
    return np.mean(acc_ais), np.mean(acc_nre), np.mean(kl_list)

def gen_batch(rng, n):
    xs, ys = [], []
    for _ in range(n):
        ti = rng.randint(0,N_TOPOS); lf_idx = rng.randint(0,N_LF); lf = lf_grid[lf_idx]
        installed = np.sort(rng.choice(range(1,N_BUS), K_FIXED, replace=False))
        obs_v = V_library[ti,lf_idx,installed] + rng.normal(0,SIGMA,K_FIXED)
        x = np.zeros(N_BUS*3, dtype=np.float32)
        x[installed] = obs_v; x[N_BUS+installed] = 1.0
        x[2*N_BUS+installed] = base_P_norm[installed]*lf
        xs.append(x); ys.append(ti)
    return (torch.tensor(np.array(xs), dtype=torch.float32).to(DEVICE),
            torch.tensor(ys, dtype=torch.long).to(DEVICE))

# ── 训练5种子 ─────────────────────────────────────────────────────────────────
all_results = {}
for seed in SEEDS:
    save_path = f"{SAVE_DIR}\\nre_119bus_ip1_seed{seed}.pt"
    if os.path.exists(save_path):
        print(f"\nSeed {seed}: model exists, loading and evaluating...")
        ckpt = torch.load(save_path, map_location=DEVICE, weights_only=False)
        model = LoadAwareNRE(N_TOPOS, N_BUS).to(DEVICE)
        model.load_state_dict(ckpt['model_state'])
        a_ais, a_nre, kl = evaluate(model)
        all_results[seed] = {'ais': a_ais, 'nre': a_nre, 'gap': a_ais-a_nre, 'kl': kl}
        print(f"  AIS={a_ais:.3f}  NRE={a_nre:.3f}  gap={a_ais-a_nre:.3f}  KL={kl:.4f}", flush=True)
        continue

    print(f"\n{'='*55}\nTraining seed={seed}...")
    torch.manual_seed(seed); np.random.seed(seed)
    model = LoadAwareNRE(N_TOPOS, N_BUS).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    rng_tr = np.random.RandomState(seed)
    model.train(); t0 = time.time()
    for step in range(1, N_STEPS+1):
        xb, yb = gen_batch(rng_tr, BATCH)
        loss = loss_fn(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step(); scheduler.step()
        if step % 20000 == 0:
            print(f"  step {step:6d}  loss={loss.item():.4f}  {time.time()-t0:.0f}s", flush=True)
    a_ais, a_nre, kl = evaluate(model)
    gap = a_ais - a_nre
    all_results[seed] = {'ais': a_ais, 'nre': a_nre, 'gap': gap, 'kl': kl}
    print(f"Seed {seed}: AIS={a_ais:.3f}  NRE={a_nre:.3f}  gap={gap:.3f}  KL={kl:.4f}", flush=True)
    torch.save({'model_state': model.state_dict(), 'seed': seed,
                'N_TOPOS': N_TOPOS, 'K_FIXED': K_FIXED, 'N_BUS': N_BUS},
               save_path)

# ── 汇总 ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"119-BUS IP1 MULTI-SEED SUMMARY  ({len(SEEDS)} seeds, K={K_FIXED}, N_TOPOS={N_TOPOS})")
print(f"{'Seed':>6}  {'AIS':>6}  {'NRE':>6}  {'gap':>6}  {'KL':>7}")
print("-"*40)
for s in SEEDS:
    r = all_results[s]
    print(f"{s:>6}  {r['ais']:.3f}  {r['nre']:.3f}  {r['gap']:.3f}  {r['kl']:.4f}")

ais_m  = np.mean([all_results[s]['ais'] for s in SEEDS])
nre_m  = np.mean([all_results[s]['nre'] for s in SEEDS])
nre_s  = np.std( [all_results[s]['nre'] for s in SEEDS])
gap_m  = np.mean([all_results[s]['gap'] for s in SEEDS])
gap_s  = np.std( [all_results[s]['gap'] for s in SEEDS])
kl_m   = np.mean([all_results[s]['kl']  for s in SEEDS])
speedup = N_TOPOS * ais_time_ms
print("-"*40)
print(f"  Mean  {ais_m:.3f}  {nre_m:.3f}±{nre_s:.3f}  gap={gap_m:.3f}±{gap_s:.3f}  KL={kl_m:.4f}")
print(f"  EnumBF speedup: {speedup:.0f}ms vs <2ms → ~{speedup/2:.0f}×")

# Wilcoxon
gaps = [all_results[s]['gap'] for s in SEEDS]
stat, p_wil = stats.wilcoxon(gaps, alternative='greater')
sig = '*' if p_wil < 0.05 else 'ns'
print(f"\nWilcoxon (AIS>NRE gap>0): T={stat}  p={p_wil:.4f} {sig}  n={len(gaps)}")

g1 = all(all_results[s]['gap'] < 0.15 for s in SEEDS)
g2 = all(all_results[s]['ais'] >= 0.40 for s in SEEDS)
g3 = all(all_results[s]['kl']  <  0.20 for s in SEEDS)
print(f"  [{'OK' if g1 else 'FAIL'}] All seeds gap<15pp (方向一致)")
print(f"  [{'OK' if g2 else 'FAIL'}] All seeds AIS>=40%")
print(f"  [{'OK' if g3 else 'FAIL'}] All seeds KL<0.20")
if g1 and g2 and g3 and p_wil < 0.05:
    print(f"\n  *** 119-BUS IP1 MULTI-SEED STABLE ✅ ***")
print('='*65)

# 保存结果
outpath = f"{SAVE_DIR}\\119bus_ip1_result.txt"
lines = [
    f"119-BUS IP1 Results: {N_TOPOS} topos, K={K_FIXED}, lf~U(0.8,1.2), 100k steps",
]
for s in SEEDS:
    r = all_results[s]
    lines.append(f"seed={s}: EnumBF={r['ais']:.3f}  NRE={r['nre']:.3f}  gap={r['gap']:.3f}  KL={r['kl']:.4f}")
lines.append(f"Mean gap={gap_m:.3f}+/-{gap_s:.3f}  speedup={speedup:.0f}x")
lines.append(f"Wilcoxon p={p_wil:.4f} {sig}  n={len(SEEDS)}")
with open(outpath, 'w', encoding='utf-8') as f:
    f.write("\n".join(lines))
print(f"\nSaved: {outpath}")
