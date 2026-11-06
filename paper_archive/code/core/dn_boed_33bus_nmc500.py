# -*- coding: utf-8 -*-
"""
T3-B: BOED 33-bus N_MC=500 重跑 + MVG直接对比
目标：在同一次实验中对比 BOED(N_MC=500) vs MVG vs Random vs GreedyLoop
验证：BOED在小K(K=4)是否在更精确EIG估计下反超MVG
N_TEST=500 (与原实验一致), 同一RNG seed(2024)
"""
import copy, time, warnings
import numpy as np
import torch
import pandapower as pp
import networkx as nx
warnings.filterwarnings('ignore')

SIGMA   = 0.009
LF_MIN  = 0.8; LF_MAX = 1.2; N_LF = 101; N_LF_C = 11
N_TEST  = 500
N_MC    = 500   # ← 从100提升到500
K_MAX   = 12
SAVE_DIR = r"<LOCAL_WORKSPACE>"
print(f"BOED 33-bus N_MC={N_MC} N_TEST={N_TEST}")

# ── 33-bus 建网 ───────────────────────────────────────────────────────────────
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
    for f,t,r,x in br:
        pp.create_line_from_parameters(net,f,t,1,r,x,0,1,in_service=True)
    for f,t,r,x in ti:
        pp.create_line_from_parameters(net,f,t,1,r,x,0,1,in_service=False)
    ld = [(1,100,60),(2,90,40),(3,120,80),(4,60,30),(5,60,20),(6,200,100),
          (7,200,100),(8,60,20),(9,60,20),(10,45,30),(11,60,35),(12,60,35),
          (13,120,80),(14,60,10),(15,60,20),(16,60,20),(17,90,40),(18,90,40),
          (19,90,40),(20,90,40),(21,90,40),(22,90,50),(23,420,200),(24,420,200),
          (25,60,25),(26,60,25),(27,60,20),(28,120,70),(29,200,600),
          (30,150,70),(31,210,100),(32,60,40)]
    for b,p,q in ld: pp.create_load(net,b,p/1000,q/1000)
    pp.create_ext_grid(net,0,vm_pu=1.0)
    return net

def enum_topos_33(ne, te):
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
            Gt = nx.Graph(); Gt.add_nodes_from(range(33)); Gt.add_edges_from(edges)
            if nx.is_connected(Gt) and nx.is_tree(Gt):
                seen.add(key); topos.append(ni + [32+ti2])
    return topos

def run_pf(net_base, t_raw, lf=1.0):
    net = copy.deepcopy(net_base)
    net.load['p_mw'] *= lf; net.load['q_mvar'] *= lf
    ns = {x for x in t_raw if x < 32}; ts = {x-32 for x in t_raw if x >= 32}
    for li in range(37):
        active = (li in ns) if li < 32 else ((li-32) in ts)
        net.line.at[net.line.index[li], 'in_service'] = active
    try:
        pp.runpp(net, algorithm='bfsw', numba=False, max_iteration=50, tolerance_mva=1e-8)
        if net.converged: return net.res_bus.vm_pu.values.copy()
    except: pass
    return None

# ── 建网、枚举拓扑 ─────────────────────────────────────────────────────────────
print("Building 33-bus network...")
net33 = build_ieee33()
ne33  = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),
         (12,13),(13,14),(14,15),(15,16),(16,17),(1,18),(18,19),(19,20),(20,21),(2,22),
         (22,23),(23,24),(5,25),(25,26),(26,27),(27,28),(28,29),(29,30),(30,31),(31,32)]
te33  = [(7,20),(8,14),(11,21),(17,32),(24,28)]
topos_raw = enum_topos_33(ne33, te33)
N_TOPOS = len(topos_raw); N_BUS = 33
CANDIDATES = list(range(1, N_BUS))
print(f"N_TOPOS={N_TOPOS}")

# ── 预计算电压库 ──────────────────────────────────────────────────────────────
import os
# 先用33-bus NRE checkpoint里的V_library
V_LIB_PATH = f"{SAVE_DIR}\\v_library_33bus_boed.npz"
if os.path.exists(V_LIB_PATH):
    dat = np.load(V_LIB_PATH)
    V_lib_fine = dat['V_lib_fine']; V_lib_c = dat['V_lib_c']
    lf_grid_fine = dat['lf_grid_fine']; lf_grid_c = dat['lf_grid_c']
    print(f"V_library loaded from cache: fine={V_lib_fine.shape} coarse={V_lib_c.shape}")
else:
    lf_grid_fine = np.linspace(LF_MIN, LF_MAX, N_LF)
    lf_idx_c     = np.round(np.linspace(0, N_LF-1, N_LF_C)).astype(int)
    lf_grid_c    = lf_grid_fine[lf_idx_c]
    print("Precomputing V_lib_fine and V_lib_c for 33-bus...")
    t0 = time.time()
    V_lib_fine = np.zeros((N_TOPOS, N_LF, N_BUS), dtype=np.float32)
    for i, topo in enumerate(topos_raw):
        for j, lf in enumerate(lf_grid_fine):
            V = run_pf(net33, topo, lf)
            V_lib_fine[i,j,:] = V if V is not None else (V_lib_fine[i,max(j-1,0),:] if j>0 else 1.0)
    print(f"V_lib_fine done: {time.time()-t0:.1f}s")
    V_lib_c = V_lib_fine[:, lf_idx_c, :]
    np.savez_compressed(V_LIB_PATH, V_lib_fine=V_lib_fine, V_lib_c=V_lib_c,
                        lf_grid_fine=lf_grid_fine, lf_grid_c=lf_grid_c)
    print(f"Saved to {V_LIB_PATH}")

# ── 后验函数 ──────────────────────────────────────────────────────────────────
def ais_posterior(obs_nodes, obs_vals):
    if len(obs_nodes) == 0: return np.ones(N_TOPOS)/N_TOPOS
    diff = (V_lib_c[:, :, obs_nodes] - np.array(obs_vals)) / SIGMA
    ll = -0.5*np.sum(diff**2, axis=2); ll -= ll.max()
    w = np.exp(ll).sum(axis=1); return w/w.sum()

def ais_posterior_joint(obs_nodes, obs_vals):
    if len(obs_nodes) == 0: return np.ones(N_TOPOS*N_LF_C)/(N_TOPOS*N_LF_C)
    diff = (V_lib_c[:, :, obs_nodes] - np.array(obs_vals)) / SIGMA
    ll = -0.5*np.sum(diff**2, axis=2); ll -= ll.max()
    w = np.exp(ll).flatten(); return w/w.sum()

def entropy(p): p = np.clip(p,1e-10,1); return -np.sum(p*np.log(p))

# ── EIG（BOED核心，N_MC=500）─────────────────────────────────────────────────
def eig_all_candidates(obs_nodes, obs_vals, candidates, rng, n_mc=N_MC):
    joint_w = ais_posterior_joint(obs_nodes, obs_vals)
    p_cur   = joint_w.reshape(N_TOPOS, N_LF_C).sum(axis=1); p_cur /= p_cur.sum()
    H_cur   = entropy(p_cur)
    mc_flat = rng.choice(N_TOPOS*N_LF_C, size=n_mc, p=joint_w)
    mc_tau  = mc_flat // N_LF_C; mc_lf = mc_flat % N_LF_C
    n_cand  = len(candidates)
    x_j_base = V_lib_c[mc_tau, mc_lf, :][:, candidates]
    noise    = rng.normal(0, SIGMA, (n_mc, n_cand))
    x_j_all  = x_j_base + noise
    obs_nodes_ext = list(obs_nodes)
    obs_vals_arr  = np.array(obs_vals)
    H_after = np.zeros(n_cand)
    for ci, cand in enumerate(candidates):
        new_nodes = obs_nodes_ext + [cand]
        h_sum = 0.0
        for m in range(n_mc):
            new_vals = np.append(obs_vals_arr, x_j_all[m, ci])
            p_new = ais_posterior(new_nodes, new_vals)
            h_sum += entropy(p_new)
        H_after[ci] = h_sum / n_mc
    return H_cur - H_after, H_cur

# ── MVG（最大后验方差贪婪）───────────────────────────────────────────────────
def mvg_select(candidates_arr, p_cur):
    V_mean_lf = V_lib_c[:, :, candidates_arr].mean(axis=1)   # (N_TOPOS, n_cand)
    E_V  = (p_cur[:, None] * V_mean_lf).sum(axis=0)
    E_V2 = (p_cur[:, None] * V_mean_lf**2).sum(axis=0)
    var  = E_V2 - E_V**2
    return candidates_arr[int(np.argmax(var))]

# ── GreedyLoop ────────────────────────────────────────────────────────────────
G33 = nx.Graph(); G33.add_edges_from(ne33)
LOOPS = [set(nx.shortest_path(G33, t[0], t[1])) for t in te33]

def greedy_loop_select(current_nodes, candidates):
    covered = [any(n in loop for n in current_nodes) for loop in LOOPS]
    uncov = [l for l,c in zip(LOOPS,covered) if not c]
    if not uncov:
        return max(candidates, key=lambda c: min((abs(c-n) for n in current_nodes), default=0)) if current_nodes else candidates[0]
    best, bc = candidates[0], -1
    for c in candidates:
        cnt = sum(1 for l in uncov if c in l)
        if cnt > bc: bc, best = cnt, c
    return best

# ── 生成测试场景（同原实验RNG=2024）─────────────────────────────────────────
rng_main = np.random.RandomState(2024)
test_cases = []
for _ in range(N_TEST):
    ti     = rng_main.randint(0, N_TOPOS)
    lf_idx = rng_main.randint(0, N_LF)
    true_v = V_lib_fine[ti, lf_idx, :]
    full_v = true_v + rng_main.normal(0, SIGMA, N_BUS)
    test_cases.append((ti, lf_idx, full_v))

strategies = ['BOED_NMC500', 'MVG', 'Random', 'GreedyLoop']
results = {s: {'acc': np.zeros(K_MAX), 'entropy': np.zeros(K_MAX)} for s in strategies}
boed_selections = [[] for _ in range(K_MAX)]

print(f"\nRunning experiment... N_TEST={N_TEST} N_MC={N_MC} K_MAX={K_MAX}")
t0 = time.time()
for case_idx, (true_ti, lf_idx, full_v) in enumerate(test_cases):
    if case_idx % 50 == 0:
        elapsed = time.time()-t0; eta = elapsed/max(case_idx,1)*(N_TEST-case_idx)
        print(f"  Case {case_idx}/{N_TEST}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)
    rng_case = np.random.RandomState(case_idx*31+7)
    rng_rand = np.random.RandomState(case_idx*17+3)

    # BOED (N_MC=500)
    sel_nodes, sel_vals = [], []
    remain = list(CANDIDATES)
    for k in range(K_MAX):
        eig_scores, _ = eig_all_candidates(sel_nodes, sel_vals, remain, rng_case)
        best_idx = int(np.argmax(eig_scores))
        node = remain[best_idx]
        boed_selections[k].append(node)
        sel_nodes.append(node); sel_vals.append(full_v[node]); remain.remove(node)
        p = ais_posterior(sel_nodes, sel_vals)
        results['BOED_NMC500']['acc'][k]     += int(np.argmax(p) == true_ti)
        results['BOED_NMC500']['entropy'][k] += entropy(p)

    # MVG
    sel_nodes, sel_vals = [], []
    remain = list(CANDIDATES)
    for k in range(K_MAX):
        p_cur = ais_posterior(sel_nodes, sel_vals)
        node = mvg_select(np.array(remain), p_cur)
        sel_nodes.append(node); sel_vals.append(full_v[node]); remain.remove(node)
        p = ais_posterior(sel_nodes, sel_vals)
        results['MVG']['acc'][k]     += int(np.argmax(p) == true_ti)
        results['MVG']['entropy'][k] += entropy(p)

    # Random
    perm = list(CANDIDATES); rng_rand.shuffle(perm)
    sel_nodes, sel_vals = [], []
    for k in range(K_MAX):
        sel_nodes.append(perm[k]); sel_vals.append(full_v[perm[k]])
        p = ais_posterior(sel_nodes, sel_vals)
        results['Random']['acc'][k]     += int(np.argmax(p) == true_ti)
        results['Random']['entropy'][k] += entropy(p)

    # GreedyLoop
    remain = list(CANDIDATES); sel_nodes, sel_vals = [], []
    for k in range(K_MAX):
        node = greedy_loop_select(sel_nodes, remain)
        sel_nodes.append(node); sel_vals.append(full_v[node]); remain.remove(node)
        p = ais_posterior(sel_nodes, sel_vals)
        results['GreedyLoop']['acc'][k]     += int(np.argmax(p) == true_ti)
        results['GreedyLoop']['entropy'][k] += entropy(p)

for s in strategies:
    results[s]['acc']     /= N_TEST
    results[s]['entropy'] /= N_TEST

# ── 输出结果 ──────────────────────────────────────────────────────────────────
print(f"\n{'='*75}")
print(f"BOED(N_MC=500) vs MVG vs Random vs GreedyLoop — 33-bus  N_TEST={N_TEST}")
print(f"{'K':>3}  {'BOED_500':>10}  {'MVG':>7}  {'Random':>8}  {'Greedy':>8}  {'BOED-MVG':>10}")
print("-"*75)
for k in range(K_MAX):
    b = results['BOED_NMC500']['acc'][k]; m = results['MVG']['acc'][k]
    r = results['Random']['acc'][k];      g = results['GreedyLoop']['acc'][k]
    print(f"{k+1:>3}  {b:>10.3f}  {m:>7.3f}  {r:>8.3f}  {g:>8.3f}  {b-m:>+10.3f}")

k4_boed = results['BOED_NMC500']['acc'][3]; k4_mvg = results['MVG']['acc'][3]
k7_boed = results['BOED_NMC500']['acc'][6]; k7_mvg = results['MVG']['acc'][6]
print(f"\nKey comparison:")
print(f"  K=4: BOED={k4_boed:.3f}  MVG={k4_mvg:.3f}  delta={k4_boed-k4_mvg:+.3f}"
      f"  {'BOED>MVG ✅' if k4_boed>k4_mvg else 'MVG>BOED (BOED loses at small K)'}")
print(f"  K=7: BOED={k7_boed:.3f}  MVG={k7_mvg:.3f}  delta={k7_boed-k7_mvg:+.3f}"
      f"  {'BOED>MVG ✅' if k7_boed>k7_mvg else 'MVG>BOED'}")
print(f"  Reference (N_MC=100): K=4 BOED=0.574 MVG=0.642  K=7 BOED=0.754 MVG=0.752")
print('='*75)

# 保存
outpath = f"{SAVE_DIR}\\boed_33bus_nmc500_result.txt"
lines = [f"BOED 33-bus N_MC=500 (vs N_MC=100 reference)",
         f"N_TEST={N_TEST}  K_MAX={K_MAX}"]
for k in range(K_MAX):
    b = results['BOED_NMC500']['acc'][k]; m = results['MVG']['acc'][k]
    r = results['Random']['acc'][k];      g = results['GreedyLoop']['acc'][k]
    lines.append(f"K={k+1:2d}: BOED={b:.3f}  MVG={m:.3f}  Random={r:.3f}  Greedy={g:.3f}  delta={b-m:+.3f}")
lines.append(f"K=4: BOED={k4_boed:.3f} MVG={k4_mvg:.3f} delta={k4_boed-k4_mvg:+.3f}")
lines.append(f"K=7: BOED={k7_boed:.3f} MVG={k7_mvg:.3f} delta={k7_boed-k7_mvg:+.3f}")
with open(outpath, 'w', encoding='utf-8') as f:
    f.write("\n".join(lines))
print(f"\nSaved: {outpath}")
