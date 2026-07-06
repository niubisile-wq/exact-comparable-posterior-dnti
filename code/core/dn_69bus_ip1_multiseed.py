# -*- coding: utf-8 -*-
"""
Step3 IP1: 69-bus NRE 5种子多seed验证
设定与33-bus完全一致: K=20, lf~U(0.8,1.2), sigma=0.009, 100k步
V_library预计算后保存到磁盘，供IP-C/IP-A复用
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
SEEDS    = [42, 123, 456, 789, 2024]
N_BUS    = 69
print(f"Device: {DEVICE}  69-bus IP1 multiseed  Seeds={SEEDS}")

# ── 网络构建 ──────────────────────────────────────────────────────────────────
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

def run_pf_scaled(net_base, t_raw, ne, te, lf=1.0):
    net = copy.deepcopy(net_base)
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

# ── 预计算 ────────────────────────────────────────────────────────────────────
print("Building 69-bus network and enumerating topologies...")
net69, ne69, te69 = build_ieee69()
topos_raw = enum_topos(ne69, te69, n=69)
N_TOPOS = len(topos_raw)
print(f"N_TOPOS={N_TOPOS}")

# 基础负荷（用于NRE输入第三维）
base_P_per_bus = np.zeros(N_BUS)
for _, row in net69.load.iterrows():
    base_P_per_bus[int(row.bus)] += row['p_mw']
base_P_norm = base_P_per_bus / (base_P_per_bus.max() + 1e-8)

lf_grid = np.linspace(LF_MIN, LF_MAX, N_LF)

# 检查是否有已保存的V_library
import os
vlib_path = f"{SAVE_DIR}\\v_library_69bus.npz"
if os.path.exists(vlib_path):
    print(f"Loading V_library from {vlib_path}...")
    dat = np.load(vlib_path)
    V_library = dat['V_library']
    print(f"V_library loaded: {V_library.shape}")
else:
    print(f"Precomputing V_library ({N_TOPOS} topos × {N_LF} lf)...")
    t0 = time.time()
    V_library = np.zeros((N_TOPOS, N_LF, N_BUS), dtype=np.float32)
    for i, topo in enumerate(topos_raw):
        for j, lf in enumerate(lf_grid):
            V = run_pf_scaled(net69, topo, ne69, te69, lf)
            V_library[i,j,:] = V if V is not None else (V_library[i,max(j-1,0),:] if j>0 else 1.0)
        if (i+1) % 10 == 0:
            print(f"  topo {i+1}/{N_TOPOS}  {time.time()-t0:.0f}s", flush=True)
    print(f"V_library done: {time.time()-t0:.1f}s")
    np.savez_compressed(vlib_path, V_library=V_library,
                        base_P_norm=base_P_norm, lf_grid=lf_grid)
    print(f"V_library saved to {vlib_path}")

# AIS基线速度测量
print("\nMeasuring AIS timing (60 topos × PF)...")
t_ais = time.time()
for topo in topos_raw:
    run_pf_scaled(net69, topo, ne69, te69, lf=1.0)
ais_time_ms = (time.time() - t_ais) / N_TOPOS * 1000
print(f"AIS per-inference: {N_TOPOS} × {ais_time_ms:.1f}ms = {N_TOPOS*ais_time_ms:.0f}ms")

# ── 推断函数 ──────────────────────────────────────────────────────────────────
def infer(model, reported, obs_v, lf):
    x = np.zeros(N_BUS*3, dtype=np.float32)
    if len(reported) > 0:
        x[reported] = obs_v
        x[N_BUS+reported] = 1.0
        x[2*N_BUS+reported] = base_P_norm[reported] * lf
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

# ── NRE批量生成 ───────────────────────────────────────────────────────────────
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

# ── 训练多seed ────────────────────────────────────────────────────────────────
all_results = {}
for seed in SEEDS:
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
    # NRE推断时间
    t1 = time.time()
    for _ in range(1000):
        infer(model, np.arange(1,21), np.ones(20), 1.0)
    nre_ms = (time.time()-t1)
    all_results[seed] = {'ais': a_ais, 'nre': a_nre, 'gap': gap, 'kl': kl}
    print(f"Seed {seed}: AIS={a_ais:.3f}  NRE={a_nre:.3f}  gap={gap:.3f}  KL={kl:.4f}  NRE_time={nre_ms:.3f}s/1k", flush=True)
    torch.save({'model_state': model.state_dict(), 'seed': seed,
                'N_TOPOS': N_TOPOS, 'K_FIXED': K_FIXED, 'SIGMA': SIGMA,
                'N_BUS': N_BUS},
               f"{SAVE_DIR}\\nre_69bus_ip1_seed{seed}.pt")

# ── 汇总 ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"69-BUS IP1 MULTI-SEED SUMMARY  ({len(SEEDS)} seeds, K={K_FIXED}, lf~U(0.8,1.2), 100k steps)")
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
print("-"*40)
print(f"  Mean  {ais_m:.3f}  {nre_m:.3f}±{nre_s:.3f}  gap={gap_m:.3f}±{gap_s:.3f}  KL={kl_m:.4f}")
print()
g1 = all(all_results[s]['gap'] < 0.10 for s in SEEDS)
g2 = all(all_results[s]['ais'] >= 0.60 for s in SEEDS)
g3 = all(all_results[s]['kl']  <  0.10 for s in SEEDS)
print(f"  [{'OK' if g1 else 'FAIL'}] All seeds gap<10pp (方向一致闸门)")
print(f"  [{'OK' if g2 else 'FAIL'}] All seeds AIS>=60%")
print(f"  [{'OK' if g3 else 'FAIL'}] All seeds KL<0.10")
print(f"  AIS speedup vs NRE: ~{N_TOPOS*ais_time_ms:.0f}ms vs <2ms")
if g1 and g2 and g3:
    print(f"\n  *** 69-BUS IP1 MULTI-SEED STABLE ✅ ***")
else:
    print(f"\n  *** 69-BUS IP1 NEEDS INVESTIGATION ***")
print('='*60)
