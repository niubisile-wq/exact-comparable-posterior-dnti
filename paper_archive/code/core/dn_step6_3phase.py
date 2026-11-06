# -*- coding: utf-8 -*-
"""
Step6: 三相不平衡配电网拓扑辨识（初步扩展验证）
网络：自定义10-bus三相不平衡馈线（2个联络开关）
目的：证明NRE框架可扩展到三相不平衡场景（堵审稿人攻击点）
对标：Xu 2021（IEEE TPS）用三相不平衡作为主卖点；本文做初步验证
方法：同IP1（NRE摊还推断），输入改为三相电压[Va,Vb,Vc,mask]
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
K_FIXED  = 6     # 10-bus网络观测6个节点（非松弛节点共9个）
N_STEPS  = 60000; BATCH = 512; LR = 3e-4
SIGMA    = 0.005; LF_MIN, LF_MAX = 0.8, 1.2; N_LF = 51  # 变负荷，51个负荷水平
SEEDS    = [42, 123, 456]
N_BUS    = 10
print(f"Device: {DEVICE}  Step6 三相不平衡  Seeds={SEEDS}")

# ── 网络构建 ──────────────────────────────────────────────────────────────────
def build_3ph_net():
    net = pp.create_empty_network()
    for i in range(N_BUS): pp.create_bus(net, vn_kv=12.66)
    pp.create_ext_grid(net, bus=0, vm_pu=1.0,
        s_sc_max_mva=1000, rx_max=0.1, rx_min=0.1,
        x0x_max=0.5, x0x_min=0.5, r0x0_max=0.1, r0x0_min=0.1)
    # 正常支路（9条，树形）
    br = [(0,1,0.12,0.08),(1,2,0.20,0.12),(2,3,0.15,0.10),
          (3,4,0.18,0.11),(4,5,0.25,0.15),(1,6,0.10,0.07),
          (6,7,0.22,0.13),(7,8,0.17,0.10),(8,9,0.19,0.12)]
    for f,t,r,x in br:
        pp.create_line_from_parameters(net,f,t,1,r,x,0,999,
            r0_ohm_per_km=r*3, x0_ohm_per_km=x*3, c0_nf_per_km=0, in_service=True)
    # 联络开关（2条，正常断开）
    ti = [(5,9,0.05,0.05),(3,7,0.05,0.05)]
    for f,t,r,x in ti:
        pp.create_line_from_parameters(net,f,t,1,r,x,0,999,
            r0_ohm_per_km=r*3, x0_ohm_per_km=x*3, c0_nf_per_km=0, in_service=False)
    # 三相不平衡负荷（各相功率不同）
    base_loads = [
        (1,0.040,0.050,0.030,0.015,0.018,0.012),
        (2,0.060,0.040,0.070,0.020,0.015,0.025),
        (3,0.030,0.060,0.040,0.010,0.022,0.015),
        (4,0.080,0.050,0.060,0.028,0.018,0.022),
        (5,0.050,0.070,0.040,0.018,0.025,0.014),
        (6,0.040,0.030,0.050,0.014,0.011,0.018),
        (7,0.070,0.060,0.050,0.024,0.021,0.018),
        (8,0.030,0.050,0.060,0.011,0.018,0.022),
        (9,0.050,0.040,0.030,0.018,0.014,0.011),
    ]
    for bus,pa,pb,pc,qa,qb,qc in base_loads:
        pp.create_asymmetric_load(net, bus=bus,
            p_a_mw=pa, p_b_mw=pb, p_c_mw=pc,
            q_a_mvar=qa, q_b_mvar=qb, q_c_mvar=qc)
    ne = [(int(f),int(t)) for f,t,r,x in br]
    te = [(int(f),int(t)) for f,t,r,x in ti]
    return net, ne, te, base_loads

def enum_topos(ne, te, n=N_BUS):
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

def run_pf_3ph(net_base, t_raw, ne, te, lf=1.0):
    """三相潮流，返回(N_BUS, 3)的电压幅值矩阵"""
    net = copy.deepcopy(net_base)
    net.asymmetric_load['p_a_mw'] *= lf
    net.asymmetric_load['p_b_mw'] *= lf
    net.asymmetric_load['p_c_mw'] *= lf
    net.asymmetric_load['q_a_mvar'] *= lf
    net.asymmetric_load['q_b_mvar'] *= lf
    net.asymmetric_load['q_c_mvar'] *= lf
    n_ne = len(ne)
    active_ne = {x for x in t_raw if x < n_ne}
    active_te = {x-n_ne for x in t_raw if x >= n_ne}
    for li in range(n_ne):
        net.line.at[net.line.index[li], 'in_service'] = (li in active_ne)
    for li in range(len(te)):
        net.line.at[net.line.index[n_ne+li], 'in_service'] = (li in active_te)
    try:
        pp.runpp_3ph(net, numba=False, max_iteration=30, tolerance_va_degree=1e-5)
        if net.converged:
            Va = net.res_bus_3ph.vm_a_pu.values
            Vb = net.res_bus_3ph.vm_b_pu.values
            Vc = net.res_bus_3ph.vm_c_pu.values
            return np.stack([Va, Vb, Vc], axis=1)  # (N_BUS, 3)
    except: pass
    return None

# ── 预计算 ────────────────────────────────────────────────────────────────────
print("Building 3-phase network and enumerating topologies...")
net3, ne3, te3, base_loads = build_3ph_net()
topos_raw = enum_topos(ne3, te3)
N_TOPOS = len(topos_raw)
print(f"N_TOPOS={N_TOPOS}  (2 tie switches)")

lf_grid = np.linspace(LF_MIN, LF_MAX, N_LF)
print(f"Precomputing V_library_3ph ({N_TOPOS} topos × {N_LF} lf)...")
t0 = time.time()
V_lib3 = np.zeros((N_TOPOS, N_LF, N_BUS, 3), dtype=np.float32)
ok_count = 0
for i, topo in enumerate(topos_raw):
    for j, lf in enumerate(lf_grid):
        V = run_pf_3ph(net3, topo, ne3, te3, lf)
        if V is not None:
            V_lib3[i,j,:,:] = V; ok_count += 1
        elif j > 0:
            V_lib3[i,j,:,:] = V_lib3[i,j-1,:,:]
        else:
            V_lib3[i,j,:,:] = 1.0
print(f"Done: {time.time()-t0:.1f}s  PF convergence: {ok_count}/{N_TOPOS*N_LF}")

# AIS速度测量（三相版：每次潮流更慢）
print("Measuring EnumBF timing...")
t1 = time.time()
for topo in topos_raw:
    run_pf_3ph(net3, topo, ne3, te3, 1.0)
enum_time_ms = (time.time()-t1)/N_TOPOS*1000
print(f"EnumBF per-inference: {N_TOPOS}×{enum_time_ms:.1f}ms = {N_TOPOS*enum_time_ms:.0f}ms")

# ── 模型（输入：N_BUS×4维: Va_obs, Vb_obs, Vc_obs, mask）──────────────────
IN_DIM = N_BUS * 4  # 3相电压 + 掩码

class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,d), nn.LayerNorm(d), nn.GELU(),
                                  nn.Linear(d,d), nn.LayerNorm(d))
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.net(x))

class NRE3ph(nn.Module):
    def __init__(self, n_topo):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(IN_DIM, 256), nn.LayerNorm(256), nn.GELU())
        self.res1 = ResBlock(256); self.res2 = ResBlock(256)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(),
                                   nn.Linear(128, n_topo))
    def forward(self, x):
        h = self.embed(x); h = self.res1(h); h = self.res2(h); return self.head(h)

def infer(model, installed, obs_V3):
    """obs_V3: (K, 3) 三相电压幅值"""
    x = np.zeros(IN_DIM, dtype=np.float32)
    x[installed]            = obs_V3[:, 0]  # Va
    x[N_BUS+installed]      = obs_V3[:, 1]  # Vb
    x[2*N_BUS+installed]    = obs_V3[:, 2]  # Vc
    x[3*N_BUS+installed]    = 1.0            # mask
    with torch.no_grad():
        logits = model(torch.tensor(x).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
    p = np.exp(logits - logits.max()); p /= p.sum(); return p

def enum_post(installed, obs_V3, lf_idx):
    """精确后验（穷举枚举，变负荷版）"""
    if len(installed) == 0: return np.ones(N_TOPOS)/N_TOPOS
    diff_a = (V_lib3[:, lf_idx, :, 0][:, installed] - obs_V3[:,0]) / SIGMA
    diff_b = (V_lib3[:, lf_idx, :, 1][:, installed] - obs_V3[:,1]) / SIGMA
    diff_c = (V_lib3[:, lf_idx, :, 2][:, installed] - obs_V3[:,2]) / SIGMA
    ll = -0.5*(np.sum(diff_a**2,axis=1)+np.sum(diff_b**2,axis=1)+np.sum(diff_c**2,axis=1))
    ll -= ll.max(); w = np.exp(ll); return w/w.sum()

def gen_batch(rng, n):
    xs, ys = [], []
    for _ in range(n):
        ti = rng.randint(0, N_TOPOS); lf_idx = rng.randint(0, N_LF)
        installed = np.sort(rng.choice(range(1,N_BUS), K_FIXED, replace=False))
        obs_V3 = V_lib3[ti, lf_idx, installed, :] + rng.normal(0, SIGMA, (K_FIXED, 3))
        x = np.zeros(IN_DIM, dtype=np.float32)
        x[installed] = obs_V3[:,0]; x[N_BUS+installed] = obs_V3[:,1]
        x[2*N_BUS+installed] = obs_V3[:,2]; x[3*N_BUS+installed] = 1.0
        xs.append(x); ys.append(ti)
    return (torch.tensor(np.array(xs), dtype=torch.float32).to(DEVICE),
            torch.tensor(ys, dtype=torch.long).to(DEVICE))

def evaluate(model):
    model.eval()
    rng = np.random.RandomState(77); acc_enum, acc_nre, kl_list = [], [], []
    for _ in range(500):
        ti = rng.randint(0,N_TOPOS); lf_idx = rng.randint(0,N_LF)
        installed = np.sort(rng.choice(range(1,N_BUS), K_FIXED, replace=False))
        obs_V3 = V_lib3[ti, lf_idx, installed, :] + rng.normal(0, SIGMA, (K_FIXED, 3))
        p_enum = enum_post(installed, obs_V3, lf_idx)
        p_nre  = infer(model, installed, obs_V3)
        acc_enum.append(int(np.argmax(p_enum)==ti))
        acc_nre.append(int(np.argmax(p_nre)==ti))
        kl = np.sum(p_enum * np.log(p_enum/(p_nre+1e-10)+1e-10))
        kl_list.append(max(0,kl))
    return np.mean(acc_enum), np.mean(acc_nre), np.mean(kl_list)

# ── 训练 ─────────────────────────────────────────────────────────────────────
all_results = {}
for seed in SEEDS:
    print(f"\n{'='*50}\nTraining seed={seed}...")
    torch.manual_seed(seed); np.random.seed(seed)
    model = NRE3ph(N_TOPOS).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    rng_tr = np.random.RandomState(seed); model.train(); t0 = time.time()
    for step in range(1, N_STEPS+1):
        xb, yb = gen_batch(rng_tr, BATCH)
        loss = loss_fn(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step(); scheduler.step()
        if step % 20000 == 0:
            print(f"  step {step:6d}  loss={loss.item():.4f}  {time.time()-t0:.0f}s", flush=True)
    a_enum, a_nre, kl = evaluate(model)
    # NRE速度
    t2 = time.time()
    for _ in range(1000): infer(model, np.arange(1,K_FIXED+1), np.ones((K_FIXED,3)))
    nre_ms = (time.time()-t2)
    speedup = N_TOPOS*enum_time_ms / (nre_ms/1000)
    all_results[seed] = {'enum': a_enum, 'nre': a_nre, 'gap': a_enum-a_nre,
                          'kl': kl, 'speedup': speedup}
    print(f"seed={seed}: EnumBF={a_enum:.3f}  NRE={a_nre:.3f}  gap={a_enum-a_nre:.3f}  "
          f"KL={kl:.4f}  speedup={speedup:.0f}x", flush=True)
    torch.save({'model_state': model.state_dict(), 'seed': seed, 'N_TOPOS': N_TOPOS},
               f"{SAVE_DIR}\\nre_3ph_seed{seed}.pt")

# ── 汇总 ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("STEP6 三相不平衡 SUMMARY")
print(f"  网络: 10-bus 三相不平衡, {N_TOPOS}种拓扑, K={K_FIXED}观测节点")
print(f"  {'Seed':>6}  {'EnumBF':>8}  {'NRE':>8}  {'gap':>6}  {'KL':>7}  {'Speedup':>9}")
print("  " + "-"*50)
for s in SEEDS:
    r = all_results[s]
    print(f"  {s:>6}  {r['enum']:>8.3f}  {r['nre']:>8.3f}  {r['gap']:>6.3f}  "
          f"{r['kl']:>7.4f}  {r['speedup']:>7.0f}x")
gaps = [all_results[s]['gap'] for s in SEEDS]
sps  = [all_results[s]['speedup'] for s in SEEDS]
print(f"  {'Mean':>6}  "
      f"  {np.mean([all_results[s]['enum'] for s in SEEDS]):.3f}  "
      f"  {np.mean([all_results[s]['nre'] for s in SEEDS]):.3f}  "
      f"{np.mean(gaps):.3f}  "
      f"  {np.mean([all_results[s]['kl'] for s in SEEDS]):.4f}  "
      f"{np.mean(sps):>7.0f}x")
g1 = all(all_results[s]['gap'] < 0.15 for s in SEEDS)
g2 = all(all_results[s]['speedup'] > 100 for s in SEEDS)
print(f"\n  [{'OK' if g1 else 'FAIL'}] gap<15pp (三相方向一致闸门)")
print(f"  [{'OK' if g2 else 'FAIL'}] speedup>100x")
if g1 and g2:
    print("  *** 三相不平衡NRE STABLE - 框架可扩展性初步验证 ***")
print('='*60)

out = [f"Step6 3-phase Results: {N_TOPOS} topos, 10-bus, K={K_FIXED}"]
for s in SEEDS:
    r=all_results[s]
    out.append(f"seed={s}: EnumBF={r['enum']:.3f} NRE={r['nre']:.3f} "
               f"gap={r['gap']:.3f} KL={r['kl']:.4f} speedup={r['speedup']:.0f}x")
out.append(f"Mean gap={np.mean(gaps):.3f}+/-{np.std(gaps):.3f}  speedup={np.mean(sps):.0f}x")
with open(f"{SAVE_DIR}\\step6_3ph_result.txt", 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print(f"Saved: {SAVE_DIR}\\step6_3ph_result.txt")
