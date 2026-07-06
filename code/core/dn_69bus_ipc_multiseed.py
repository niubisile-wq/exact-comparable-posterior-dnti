# -*- coding: utf-8 -*-
"""
Step3 IP-C: 69-bus 鲁棒NRE 3种子（加载IP1保存的V_library）
设定与33-bus一致: K=20, lf~U(0.8,1.2), missing~U(0,0.3), 60k步
闸门: all-seed delta>10pp at miss=10% and miss=30%
"""
import copy, time, warnings
import numpy as np
import torch
import torch.nn as nn
import pandapower as pp
import networkx as nx
warnings.filterwarnings('ignore')

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = r"<LOCAL_WORKSPACE>"
K_FIXED  = 20; N_STEPS = 100000; BATCH = 512; LR = 3e-4
SIGMA    = 0.009; LF_MIN, LF_MAX = 0.8, 1.2; N_LF = 101
SEEDS    = [42, 123, 456]
MISSING_RATES = [0.0, 0.05, 0.10, 0.20, 0.30]
N_BUS    = 69
print(f"Device: {DEVICE}  69-bus IP-C multiseed  Seeds={SEEDS}")

# ── 网络构建（同IP1）────────────────────────────────────────────────────────
def build_ieee69():
    net = pp.create_empty_network()
    for i in range(69): pp.create_bus(net, vn_kv=12.66)
    branch_data = [
        (1,2,0.0005,0.0012,0,0),(2,3,0.0005,0.0012,0,0),(3,4,0.0015,0.0036,0,0),(4,5,0.0251,0.0294,0,0),
        (5,6,0.3660,0.1864,2.6,2.2),(6,7,0.3811,0.1941,40.4,30.0),(7,8,0.0922,0.0470,75.0,54.0),
        (8,9,0.0493,0.0251,30.0,22.0),(9,10,0.8190,0.2707,28.0,19.0),(10,11,0.1872,0.0619,145.0,104.0),
        (11,12,0.7114,0.2351,145.0,104.0),(12,13,1.0300,0.3400,8.0,5.5),(13,14,1.0440,0.3450,8.0,5.5),
        (14,15,1.0580,0.3496,0.0,0.0),(15,16,0.1966,0.0650,45.5,30.0),(16,17,0.3744,0.1238,60.0,35.0),
        (17,18,0.0047,0.0016,60.0,35.0),(18,19,0.3276,0.1083,0.0,0.0),(19,20,0.2106,0.0690,1.0,0.6),
        (20,21,0.3416,0.1129,114.0,81.0),(21,22,0.0140,0.0046,5.3,3.5),(22,23,0.1591,0.0526,0.0,0.0),
        (23,24,0.3463,0.1145,28.0,20.0),(24,25,0.7488,0.2475,0.0,0.0),(25,26,0.3089,0.1021,14.0,10.0),
        (26,27,0.1732,0.0572,14.0,10.0),(3,28,0.0044,0.0108,26.0,18.6),(28,29,0.0640,0.1565,26.0,18.6),
        (29,30,0.3978,0.1315,0.0,0.0),(30,31,0.0702,0.0232,0.0,0.0),(31,32,0.3510,0.1160,0.0,0.0),
        (32,33,0.8390,0.2816,14.0,10.0),(33,34,1.7080,0.5646,19.5,14.0),(34,35,1.4740,0.4873,6.0,4.0),
        (35,36,0.0044,0.0108,26.0,18.6),(36,37,0.0640,0.1565,26.0,18.6),(37,38,0.1053,0.1230,0.0,0.0),
        (38,39,0.0304,0.0355,24.0,17.0),(39,40,0.0018,0.0021,24.0,17.0),(40,41,0.7283,0.8509,1.2,1.0),
        (41,42,0.3100,0.3623,0.0,0.0),(42,43,0.0410,0.0478,6.0,4.3),(43,44,0.0092,0.0116,0.0,0.0),
        (44,45,0.1089,0.1373,39.2,26.3),(45,46,0.0009,0.0012,39.2,26.3),(4,47,0.0034,0.0084,0.0,0.0),
        (47,48,0.0851,0.2083,79.0,56.4),(48,49,0.2898,0.7091,384.7,274.5),(49,50,0.0822,0.2011,384.7,274.5),
        (8,51,0.0928,0.0473,40.5,28.3),(51,52,0.3319,0.1114,3.6,2.7),(9,53,0.1740,0.0886,4.35,3.5),
        (53,54,0.2030,0.1034,26.4,19.0),(54,55,0.2842,0.1447,24.0,17.2),(55,56,0.2813,0.1433,0.0,0.0),
        (56,57,1.5900,0.5337,0.0,0.0),(57,58,0.7837,0.2630,0.0,0.0),(58,59,0.3042,0.1006,100.0,72.0),
        (59,60,0.3861,0.1172,0.0,0.0),(60,61,0.5075,0.2585,1244.0,888.0),(61,62,0.0974,0.0496,32.0,23.0),
        (62,63,0.1450,0.0738,0.0,0.0),(63,64,0.7105,0.3619,227.0,162.0),(64,65,1.0410,0.5302,59.0,42.0),
        (11,66,0.2012,0.0611,18.0,13.0),(66,67,0.0047,0.0014,18.0,13.0),
        (12,68,0.7394,0.2444,28.0,20.0),(68,69,0.0047,0.0014,28.0,20.0),
    ]
    tie_data = [(11,43,0.5,0.5),(13,21,0.5,0.5),(15,46,0.5,0.5),(50,59,0.5,0.5),(27,65,0.5,0.5)]
    ne, te = [], []
    for f1,t1,r,x,p,q in branch_data:
        f0,t0=f1-1,t1-1
        pp.create_line_from_parameters(net,f0,t0,1,r,x,0,9999,in_service=True)
        if p>0 or q>0: pp.create_load(net,t0,p/1000,q/1000)
        ne.append((f0,t0))
    for f1,t1,r,x in tie_data:
        f0,t0=f1-1,t1-1
        pp.create_line_from_parameters(net,f0,t0,1,r,x,0,9999,in_service=False)
        te.append((f0,t0))
    pp.create_ext_grid(net,0,vm_pu=1.0)
    return net, ne, te

def enum_topos(ne, te, n=69):
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

# ── 加载V_library ─────────────────────────────────────────────────────────────
print("Loading V_library from disk...")
import os
vlib_path = f"{SAVE_DIR}\\v_library_69bus.npz"
if os.path.exists(vlib_path):
    dat = np.load(vlib_path)
    V_library = dat['V_library']
    base_P_norm = dat['base_P_norm']
    lf_grid = dat['lf_grid']
    print(f"V_library loaded: {V_library.shape}")
else:
    print("ERROR: V_library not found. Run dn_69bus_ip1_multiseed.py first!")
    exit(1)

print("Building network for topology enumeration...")
net69, ne69, te69 = build_ieee69()
topos_raw = enum_topos(ne69, te69, n=69)
N_TOPOS = len(topos_raw)
print(f"N_TOPOS={N_TOPOS}")

# ── 模型 ──────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,d), nn.LayerNorm(d), nn.GELU(),
                                  nn.Linear(d,d), nn.LayerNorm(d))
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.net(x))

class LoadAwareNRE(nn.Module):
    def __init__(self, n_topo, n_bus=69):
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
        x[reported] = obs_v; x[N_BUS+reported] = 1.0
        x[2*N_BUS+reported] = base_P_norm[reported]*lf
    with torch.no_grad():
        logits = model(torch.tensor(x).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
    p = np.exp(logits - logits.max()); p /= p.sum(); return p

def evaluate_ipc(rob_model, nai_model):
    rob_model.eval(); nai_model.eval()
    rows = []
    for miss_rate in MISSING_RATES:
        n_miss = int(K_FIXED * miss_rate)
        acc_rob, acc_nai = [], []
        rng = np.random.RandomState(77)
        for _ in range(1000):
            ti = rng.randint(0,N_TOPOS); lf_idx = rng.randint(0,N_LF); lf = lf_grid[lf_idx]
            installed = np.sort(rng.choice(range(1,N_BUS), K_FIXED, replace=False))
            if n_miss > 0:
                miss_idx = rng.choice(len(installed), n_miss, replace=False)
                reported = np.delete(installed, miss_idx)
            else:
                reported = installed
            obs_v = V_library[ti, lf_idx, reported] + rng.normal(0, SIGMA, len(reported))
            acc_rob.append(int(np.argmax(infer(rob_model, reported, obs_v, lf)) == ti))
            acc_nai.append(int(np.argmax(infer(nai_model, reported, obs_v, lf)) == ti))
        rows.append({'miss': miss_rate, 'rob': np.mean(acc_rob), 'nai': np.mean(acc_nai),
                     'delta': np.mean(acc_rob) - np.mean(acc_nai)})
    return rows

def gen_batch_robust(rng, n):
    xs, ys = [], []
    for _ in range(n):
        ti = rng.randint(0,N_TOPOS); lf_idx = rng.randint(0,N_LF); lf = lf_grid[lf_idx]
        installed = np.sort(rng.choice(range(1,N_BUS), K_FIXED, replace=False))
        miss_rate = rng.uniform(0.0, 0.3); n_miss = int(K_FIXED*miss_rate)
        if n_miss > 0:
            miss_idx = rng.choice(len(installed), n_miss, replace=False)
            reported = np.delete(installed, miss_idx)
        else:
            reported = installed
        obs_v = V_library[ti, lf_idx, reported] + rng.normal(0, SIGMA, len(reported))
        x = np.zeros(N_BUS*3, dtype=np.float32)
        x[reported] = obs_v; x[N_BUS+reported] = 1.0
        x[2*N_BUS+reported] = base_P_norm[reported]*lf
        xs.append(x); ys.append(ti)
    return (torch.tensor(np.array(xs), dtype=torch.float32).to(DEVICE),
            torch.tensor(ys, dtype=torch.long).to(DEVICE))

# 加载朴素NRE（seed=42）
nai_ckpt = torch.load(f"{SAVE_DIR}\\nre_69bus_ip1_seed42.pt", map_location=DEVICE, weights_only=False)
nai_model = LoadAwareNRE(N_TOPOS, N_BUS).to(DEVICE)
nai_model.load_state_dict(nai_ckpt['model_state']); nai_model.eval()
print("Naive NRE (seed=42) loaded.")

# ── 训练3个seed ───────────────────────────────────────────────────────────────
all_results = {}
for seed in SEEDS:
    print(f"\n{'='*55}\nTraining robust NRE seed={seed}...")
    torch.manual_seed(seed); np.random.seed(seed)
    rob_model = LoadAwareNRE(N_TOPOS, N_BUS).to(DEVICE)
    opt = torch.optim.AdamW(rob_model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    rng_tr = np.random.RandomState(seed)
    rob_model.train(); t0 = time.time()
    for step in range(1, N_STEPS+1):
        xb, yb = gen_batch_robust(rng_tr, BATCH)
        loss = loss_fn(rob_model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step(); scheduler.step()
        if step % 25000 == 0:
            print(f"  step {step:6d}  loss={loss.item():.4f}  {time.time()-t0:.0f}s", flush=True)
    rows = evaluate_ipc(rob_model, nai_model)
    all_results[seed] = rows
    for r in rows:
        print(f"  miss={r['miss']*100:.0f}%: rob={r['rob']:.3f}  nai={r['nai']:.3f}  delta={r['delta']:+.3f}", flush=True)
    torch.save({'model_state': rob_model.state_dict(), 'seed': seed, 'N_TOPOS': N_TOPOS},
               f"{SAVE_DIR}\\nre_69bus_ipc_seed{seed}.pt")

# ── 汇总 ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"69-BUS IP-C MULTI-SEED SUMMARY (3 seeds: 42,123,456)")
print(f"{'Miss%':>6}  {'Rob_mean':>9}  {'Rob_std':>8}  {'Nai_mean':>9}  {'Delta_mean':>11}  {'Delta_min':>10}")
print("-"*60)
for i, miss_rate in enumerate(MISSING_RATES):
    robs   = [all_results[s][i]['rob']   for s in SEEDS]
    nais   = [all_results[s][i]['nai']   for s in SEEDS]
    deltas = [all_results[s][i]['delta'] for s in SEEDS]
    print(f"  {miss_rate*100:>4.0f}%  {np.mean(robs):>9.3f}  {np.std(robs):>8.3f}  "
          f"{np.mean(nais):>9.3f}  {np.mean(deltas):>+11.3f}  {np.min(deltas):>+10.3f}", flush=True)
gate_10 = all(all_results[s][2]['delta'] > 0.10 for s in SEEDS)
gate_30 = all(all_results[s][4]['delta'] > 0.10 for s in SEEDS)
print(f"\n  Gate miss=10% all-seed delta>10pp: {'PASS' if gate_10 else 'FAIL'}")
print(f"  Gate miss=30% all-seed delta>10pp: {'PASS' if gate_30 else 'FAIL'}")
if gate_10 and gate_30:
    print(f"  *** 69-BUS IP-C MULTI-SEED STABLE ✅ ***")
print('='*60)
