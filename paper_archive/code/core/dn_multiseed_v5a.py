# -*- coding: utf-8 -*-
"""
v5A 多seed验证：5个训练seed顺序运行，报告均值±标准差
配置：K=20, lf~U(0.8,1.2), 100k步（与v5A完全一致）
Seeds: 42(已跑), 123, 456, 789, 2024
"""
import copy, time, warnings
import numpy as np
import torch
import torch.nn as nn
import pandapower as pp
import networkx as nx
warnings.filterwarnings('ignore')

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR  = r"<LOCAL_WORKSPACE>"
K_FIXED   = 20
N_STEPS   = 100000
BATCH     = 512
LR        = 3e-4
SIGMA     = 0.009
LF_MIN, LF_MAX = 0.8, 1.2
N_LF      = 101
LOG_EVERY = 20000   # 减少日志输出
SEEDS     = [42, 123, 456, 789, 2024]
TEST_SEED = 77      # 固定测试seed，评估一致性
print(f"Device: {DEVICE}")
print(f"Multi-seed v5A: K={K_FIXED}, lf~U({LF_MIN},{LF_MAX}), {N_STEPS} steps")
print(f"Seeds: {SEEDS}\n")

# ── 网络构建（只建一次）──────────────────────────────────────────────────────
def build_ieee33():
    net = pp.create_empty_network()
    for i in range(33): pp.create_bus(net, vn_kv=12.66)
    br = [(0,1,0.0922,0.0470),(1,2,0.4930,0.2511),(2,3,0.3660,0.1864),
          (3,4,0.3811,0.1941),(4,5,0.8190,0.7070),(5,6,0.1872,0.6188),
          (6,7,0.7114,0.2351),(7,8,1.0300,0.7400),(8,9,1.0440,0.7400),
          (9,10,0.1966,0.0650),(10,11,0.3744,0.1238),(11,12,1.4680,1.1550),
          (12,13,0.5416,0.7129),(13,14,0.5910,0.5260),(14,15,0.7463,0.5450),
          (15,16,1.2890,1.7210),(16,17,0.7320,0.5740),(1,18,0.1640,0.1565),
          (18,19,1.5042,1.3554),(19,20,0.4095,0.4784),(20,21,0.7089,0.9373),
          (2,22,0.4512,0.3083),(22,23,0.8980,0.7091),(23,24,0.8960,0.7011),
          (5,25,0.2030,0.1034),(25,26,0.2842,0.1447),(26,27,1.0590,0.9337),
          (27,28,0.8042,0.7006),(28,29,0.5075,0.2585),(29,30,0.9744,0.9630),
          (30,31,0.3105,0.3619),(31,32,0.3410,0.5302)]
    ti = [(7,20,0.089,0.089),(8,14,0.059,0.059),
          (11,21,0.089,0.089),(17,32,0.038,0.085),(24,28,0.056,0.065)]
    for f,t,r,x in br: pp.create_line_from_parameters(net,f,t,1,r,x,0,1,in_service=True)
    for f,t,r,x in ti: pp.create_line_from_parameters(net,f,t,1,r,x,0,1,in_service=False)
    ld = [(1,100,60),(2,90,40),(3,120,80),(4,60,30),(5,60,20),(6,200,100),
          (7,200,100),(8,60,20),(9,60,20),(10,45,30),(11,60,35),(12,60,35),
          (13,120,80),(14,60,10),(15,60,20),(16,60,20),(17,90,40),(18,90,40),
          (19,90,40),(20,90,40),(21,90,40),(22,90,50),(23,420,200),(24,420,200),
          (25,60,25),(26,60,25),(27,60,20),(28,120,70),(29,200,600),
          (30,150,70),(31,210,100),(32,60,40)]
    for b,p,q in ld: pp.create_load(net, b, p/1000, q/1000)
    pp.create_ext_grid(net, 0, vm_pu=1.0)
    return net, [(int(f),int(t)) for f,t,r,x in br], [(int(f),int(t)) for f,t,r,x in ti]

def enum_topos(ne, te, n=33):
    G = nx.Graph(); G.add_edges_from(ne)
    topos = [list(range(32))]; seen = {frozenset(range(32))}
    for ti2, tie in enumerate(te):
        path = nx.shortest_path(G, tie[0], tie[1])
        for i in range(len(path)-1):
            oe = frozenset([path[i], path[i+1]])
            ni = [j for j,e in enumerate(ne) if frozenset(e) != oe]
            key = frozenset(ni)
            if key in seen: continue
            edges = [ne[j] for j in ni] + [tie]
            Gt = nx.Graph(); Gt.add_nodes_from(range(n)); Gt.add_edges_from(edges)
            if nx.is_connected(Gt) and nx.is_tree(Gt):
                seen.add(key); topos.append(ni + [32+ti2])
    return topos

def run_pf_scaled(net_base, t_raw, lf=1.0):
    net = copy.deepcopy(net_base)
    net.load['p_mw']   = net.load['p_mw']   * lf
    net.load['q_mvar'] = net.load['q_mvar'] * lf
    ns = {x for x in t_raw if x < 32}; ts = {x-32 for x in t_raw if x >= 32}
    for li in range(37):
        active = (li in ns) if li < 32 else ((li-32) in ts)
        net.line.at[net.line.index[li], 'in_service'] = active
    try:
        pp.runpp(net, algorithm='bfsw', numba=False, max_iteration=50, tolerance_mva=1e-8)
        if net.converged: return net.res_bus.vm_pu.values.copy()
    except: pass
    return None

class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,d),nn.LayerNorm(d),nn.GELU(),
                                 nn.Linear(d,d),nn.LayerNorm(d))
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.net(x))

class LoadAwareNRE(nn.Module):
    def __init__(self, n_topo, n_bus=33):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_bus*3,512),nn.LayerNorm(512),nn.GELU())
        self.res1 = ResBlock(512); self.res2 = ResBlock(512); self.res3 = ResBlock(512)
        self.head = nn.Sequential(nn.Linear(512,256),nn.LayerNorm(256),nn.GELU(),
                                  nn.Linear(256,n_topo))
    def forward(self, x):
        h = self.embed(x)
        h = self.res1(h); h = self.res2(h); h = self.res3(h)
        return self.head(h)

# ── 预计算（所有seed共享同一个库）────────────────────────────────────────────
print("Building network and precomputing library...")
net33, ne33, te33 = build_ieee33()
topos_raw = enum_topos(ne33, te33)
N_TOPOS = len(topos_raw); N_BUS = 33
base_P_per_bus = np.zeros(N_BUS)
for _, row in net33.load.iterrows():
    base_P_per_bus[int(row.bus)] += row['p_mw']
base_P_norm = base_P_per_bus / (base_P_per_bus.max() + 1e-8)
lf_grid = np.linspace(LF_MIN, LF_MAX, N_LF)

t0 = time.time()
V_library = np.zeros((N_TOPOS, N_LF, N_BUS), dtype=np.float32)
for i, topo in enumerate(topos_raw):
    for j, lf in enumerate(lf_grid):
        V = run_pf_scaled(net33, topo, lf)
        V_library[i,j,:] = V if V is not None else (V_library[i,max(j-1,0),:] if j>0 else 1.0)
print(f"Library done: {time.time()-t0:.1f}s  ({N_TOPOS}x{N_LF} PF runs)\n")

def gen_batch(rng, n):
    xs, ys = [], []
    for _ in range(n):
        ti=rng.randint(0,N_TOPOS); lf_idx=rng.randint(0,N_LF); lf=lf_grid[lf_idx]
        V=V_library[ti,lf_idx,:]
        obs=np.sort(rng.choice(range(1,N_BUS),K_FIXED,replace=False))
        obs_v=V[obs]+rng.normal(0,SIGMA,K_FIXED)
        x=np.zeros(N_BUS*3,dtype=np.float32)
        x[obs]=obs_v; x[N_BUS+obs]=1.0; x[2*N_BUS+obs]=base_P_norm[obs]*lf
        xs.append(x); ys.append(ti)
    return (torch.tensor(np.array(xs),dtype=torch.float32).to(DEVICE),
            torch.tensor(ys,dtype=torch.long).to(DEVICE))

def evaluate(model, test_seed=TEST_SEED, n_test=1000):
    model.eval()
    rng_te = np.random.RandomState(test_seed)
    acc_ais, acc_nre, kl_vals, ci90 = [], [], [], []
    for _ in range(n_test):
        ti=rng_te.randint(0,N_TOPOS); lf_idx=rng_te.randint(0,N_LF); lf=lf_grid[lf_idx]
        V=V_library[ti,lf_idx,:]
        obs=np.sort(rng_te.choice(range(1,N_BUS),K_FIXED,replace=False))
        obs_v=V[obs]+rng_te.normal(0,SIGMA,K_FIXED)
        diff=(V_library[:,lf_idx,:][:,obs]-obs_v)/SIGMA
        ll=-0.5*np.sum(diff**2,axis=1); ll-=ll.max(); w=np.exp(ll); p_ais=w/w.sum()
        acc_ais.append(int(np.argmax(p_ais)==ti))
        x=np.zeros(N_BUS*3,dtype=np.float32)
        x[obs]=obs_v; x[N_BUS+obs]=1.0; x[2*N_BUS+obs]=base_P_norm[obs]*lf
        with torch.no_grad():
            logits=model(torch.tensor(x).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
        p_nre=np.exp(logits-logits.max()); p_nre/=p_nre.sum()
        acc_nre.append(int(np.argmax(p_nre)==ti))
        kl_vals.append(np.sum(p_ais*np.log((p_ais+1e-10)/(p_nre+1e-10))))
        si=np.argsort(p_nre)[::-1]; cs=np.cumsum(p_nre[si])
        ci90.append(int(ti in si[:np.searchsorted(cs,0.90)+1]))
    return (np.mean(acc_ais), np.mean(acc_nre),
            np.mean(acc_ais)-np.mean(acc_nre),
            np.mean(kl_vals), np.mean(ci90))

# ── 主循环：5个seed顺序训练 ───────────────────────────────────────────────────
results = []
# seed=42已跑过v5A，直接记录结果
results.append({'seed': 42, 'ais': 0.721, 'nre': 0.670,
                'gap': 0.051, 'kl': 0.0452, 'ci90': 0.982})
print(f"Seed 42 (pre-computed from v5A): AIS=72.1%, NRE=67.0%, gap=5.1pp, KL=0.0452")

for seed in SEEDS[1:]:   # [123, 456, 789, 2024]
    print(f"\n{'='*55}")
    print(f"Training seed={seed}...")
    torch.manual_seed(seed); np.random.seed(seed)
    model = LoadAwareNRE(N_TOPOS, N_BUS).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    rng_tr = np.random.RandomState(seed)
    model.train(); t0 = time.time()
    for step in range(1, N_STEPS+1):
        xb,yb = gen_batch(rng_tr, BATCH)
        loss = loss_fn(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step(); scheduler.step()
        if step % LOG_EVERY == 0:
            print(f"  step {step:6d}  loss={loss.item():.4f}  {time.time()-t0:.0f}s", flush=True)
    ais, nre, gap, kl, ci = evaluate(model)
    results.append({'seed': seed, 'ais': ais, 'nre': nre,
                    'gap': gap, 'kl': kl, 'ci90': ci})
    print(f"Seed {seed}: AIS={ais:.3f}, NRE={nre:.3f}, gap={gap:.3f}, KL={kl:.4f}, CI90={ci:.3f}", flush=True)
    torch.save({'model_state': model.state_dict(), 'seed': seed,
                'N_TOPOS': N_TOPOS, 'K_FIXED': K_FIXED, 'SIGMA': SIGMA,
                'N_BUS': N_BUS, 'lf_grid': lf_grid, 'V_library': V_library,
                'base_P_norm': base_P_norm, 'topos_raw': topos_raw},
               f"{SAVE_DIR}\\nre_ip1_v5a_seed{seed}.pt")

# ── 汇总 ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print(f"MULTI-SEED SUMMARY  (5 seeds, K=20, lf~U(0.8,1.2), 100k steps)", flush=True)
print(f"{'Seed':>6}  {'AIS':>6}  {'NRE':>6}  {'gap':>6}  {'KL':>7}  {'CI90':>6}", flush=True)
print("-"*50, flush=True)
for r in results:
    print(f"  {r['seed']:>4}  {r['ais']:.3f}  {r['nre']:.3f}  "
          f"{r['gap']:.3f}  {r['kl']:.4f}  {r['ci90']:.3f}", flush=True)

ais_arr  = np.array([r['ais']  for r in results])
nre_arr  = np.array([r['nre']  for r in results])
gap_arr  = np.array([r['gap']  for r in results])
kl_arr   = np.array([r['kl']   for r in results])
ci90_arr = np.array([r['ci90'] for r in results])

print("-"*50, flush=True)
print(f"  Mean  {ais_arr.mean():.3f}  {nre_arr.mean():.3f}  "
      f"{gap_arr.mean():.3f}  {kl_arr.mean():.4f}  {ci90_arr.mean():.3f}", flush=True)
print(f"  Std   {ais_arr.std():.3f}  {nre_arr.std():.3f}  "
      f"{gap_arr.std():.3f}  {kl_arr.std():.4f}  {ci90_arr.std():.3f}", flush=True)

all_gap_ok = all(gap_arr < 0.06)
all_kl_ok  = all(kl_arr  < 0.05)
all_ais_ok = all(ais_arr >= 0.70)
print(f"\n  [{'OK' if all_gap_ok else 'FAIL'}] All seeds gap<6pp")
print(f"  [{'OK' if all_kl_ok  else 'FAIL'}] All seeds KL<0.05")
print(f"  [{'OK' if all_ais_ok else 'FAIL'}] All seeds AIS>=70%")
stable = all([all_gap_ok, all_kl_ok, all_ais_ok])
print(f"\n  {'*** MULTI-SEED STABLE — Step 2 CONFIRMED ***' if stable else '--- Not stable across seeds ---'}", flush=True)
print(f"{'='*60}", flush=True)
