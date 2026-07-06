# -*- coding: utf-8 -*-
"""
IP-A 正式实验 v2：精确AIS-BOED + 变负荷（与IP1/IP-C设定一致）
核心：序贯传感器选址，最大化EIG，AIS后验边缘化负荷因子
N_TEST=500, N_MC=100, N_LF_C=11（粗网格EIG近似，N_MC提高到100以减少MC方差）
"""
import copy, time, warnings
import numpy as np
import pandapower as pp
import networkx as nx
warnings.filterwarnings('ignore')

SIGMA   = 0.009
LF_MIN  = 0.8; LF_MAX = 1.2; N_LF = 101; N_LF_C = 11  # 粗粒度供EIG用
N_TEST  = 500
N_MC    = 100
K_MAX   = 12
SAVE_DIR = r"<LOCAL_WORKSPACE>"

print(f"IP-A v2 load-aware: N_TEST={N_TEST} N_MC={N_MC} N_LF_C={N_LF_C}")

# ── 网络构建 ──────────────────────────────────────────────────────────────────
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

# ── 预计算电压库 ──────────────────────────────────────────────────────────────
print("Building network...")
net33, ne33, te33 = build_ieee33()
topos_raw = enum_topos(ne33, te33)
N_TOPOS = len(topos_raw); N_BUS = 33
CANDIDATES = list(range(1, N_BUS))

lf_grid_fine = np.linspace(LF_MIN, LF_MAX, N_LF)
lf_grid_c    = np.linspace(LF_MIN, LF_MAX, N_LF_C)  # 粗网格供EIG用

print(f"N_TOPOS={N_TOPOS}, N_BUS={N_BUS}")
print("Precomputing V_library (fine, for test generation)...")
t0 = time.time()
V_lib_fine = np.zeros((N_TOPOS, N_LF, N_BUS), dtype=np.float32)
for i, topo in enumerate(topos_raw):
    for j, lf in enumerate(lf_grid_fine):
        V = run_pf(net33, topo, lf)
        V_lib_fine[i,j,:] = V if V is not None else (V_lib_fine[i,max(j-1,0),:] if j>0 else 1.0)
print(f"V_lib_fine done: {time.time()-t0:.1f}s")

print("Precomputing V_library_c (coarse, for EIG)...")
t1 = time.time()
V_lib_c = np.zeros((N_TOPOS, N_LF_C, N_BUS), dtype=np.float32)
for i, topo in enumerate(topos_raw):
    for j, lf in enumerate(lf_grid_c):
        V = run_pf(net33, topo, lf)
        V_lib_c[i,j,:] = V if V is not None else (V_lib_c[i,max(j-1,0),:] if j>0 else 1.0)
print(f"V_lib_c done: {time.time()-t1:.1f}s")

# ── AIS后验（边缘化负荷因子，粗网格）────────────────────────────────────────
def ais_posterior(obs_nodes, obs_vals):
    """p(τ|x_obs)，边缘化lf，返回shape=(N_TOPOS,)"""
    if len(obs_nodes) == 0:
        return np.ones(N_TOPOS) / N_TOPOS
    obs_nodes = list(obs_nodes)
    diff = (V_lib_c[:, :, obs_nodes] - np.array(obs_vals)) / SIGMA  # (N_TOPOS, N_LF_C, K)
    ll = -0.5 * np.sum(diff**2, axis=2)  # (N_TOPOS, N_LF_C)
    ll -= ll.max()
    w = np.exp(ll)  # (N_TOPOS, N_LF_C)
    p = w.sum(axis=1)  # 边缘化lf → (N_TOPOS,)
    return p / p.sum()

def ais_posterior_joint(obs_nodes, obs_vals):
    """联合权重 w[τ,lf]，用于MC采样，返回(N_TOPOS×N_LF_C,)的归一化概率"""
    if len(obs_nodes) == 0:
        return np.ones(N_TOPOS * N_LF_C) / (N_TOPOS * N_LF_C)
    obs_nodes = list(obs_nodes)
    diff = (V_lib_c[:, :, obs_nodes] - np.array(obs_vals)) / SIGMA  # (N_TOPOS, N_LF_C, K)
    ll = -0.5 * np.sum(diff**2, axis=2)
    ll -= ll.max()
    w = np.exp(ll).flatten()
    return w / w.sum()

def entropy(p):
    p = np.clip(p, 1e-10, 1.0)
    return -np.sum(p * np.log(p))

# ── 精确EIG（边缘化lf，MC采样）──────────────────────────────────────────────
def eig_all_candidates(obs_nodes, obs_vals, candidates, rng, n_mc=N_MC):
    """
    对所有候选节点计算EIG(j) = H(τ|x_S) - E_{x_j|x_S}[H(τ|x_S,x_j)]
    MC：从联合后验p(τ,lf|x_S)采样，再从预测分布采x_j
    """
    joint_w = ais_posterior_joint(obs_nodes, obs_vals)
    p_cur   = joint_w.reshape(N_TOPOS, N_LF_C).sum(axis=1)
    p_cur  /= p_cur.sum()
    H_cur   = entropy(p_cur)

    # 从联合后验采样(τ,lf)对
    mc_flat = rng.choice(N_TOPOS * N_LF_C, size=n_mc, p=joint_w)
    mc_tau  = mc_flat // N_LF_C
    mc_lf   = mc_flat % N_LF_C

    n_cand = len(candidates)
    H_after = np.zeros(n_cand)

    # 预先生成所有MC×候选的噪声
    x_j_base = V_lib_c[mc_tau, mc_lf, :][:, candidates]  # (N_MC, N_CAND)
    noise     = rng.normal(0, SIGMA, (n_mc, n_cand))
    x_j_all   = x_j_base + noise  # (N_MC, N_CAND)

    obs_nodes_ext = list(obs_nodes)
    obs_vals_arr  = np.array(obs_vals)

    for ci, cand in enumerate(candidates):
        new_nodes = obs_nodes_ext + [cand]
        h_sum = 0.0
        for m in range(n_mc):
            new_vals = np.append(obs_vals_arr, x_j_all[m, ci])
            p_new = ais_posterior(new_nodes, new_vals)
            h_sum += entropy(p_new)
        H_after[ci] = h_sum / n_mc

    return H_cur - H_after, H_cur

# ── 贪婪回路覆盖 ──────────────────────────────────────────────────────────────
def get_fundamental_loops(ne, te):
    G = nx.Graph(); G.add_edges_from(ne)
    return [set(nx.shortest_path(G, tie[0], tie[1])) for tie in te]

LOOPS = get_fundamental_loops(ne33, te33)

def greedy_loop_select(current_nodes, candidates):
    covered = [any(n in loop for n in current_nodes) for loop in LOOPS]
    uncov = [l for l, c in zip(LOOPS, covered) if not c]
    if not uncov:
        return max(candidates, key=lambda c: min((abs(c-n) for n in current_nodes), default=0)) if current_nodes else candidates[0]
    best, best_cnt = candidates[0], -1
    for c in candidates:
        cnt = sum(1 for l in uncov if c in l)
        if cnt > best_cnt: best_cnt, best = cnt, c
    return best

# ── 主实验 ────────────────────────────────────────────────────────────────────
rng_main = np.random.RandomState(2024)
print(f"\nGenerating {N_TEST} test scenarios (variable load lf~U(0.8,1.2))...")
test_cases = []
for _ in range(N_TEST):
    ti     = rng_main.randint(0, N_TOPOS)
    lf_idx = rng_main.randint(0, N_LF)
    true_v = V_lib_fine[ti, lf_idx, :]
    full_v = true_v + rng_main.normal(0, SIGMA, N_BUS)
    test_cases.append((ti, lf_idx, full_v))

strategies = ['BOED_AIS', 'Random', 'GreedyLoop']
results = {s: {'acc': np.zeros(K_MAX), 'entropy': np.zeros(K_MAX)} for s in strategies}
boed_selections = [[] for _ in range(K_MAX)]

print(f"Running BOED experiment... N_TEST={N_TEST} N_MC={N_MC} K_MAX={K_MAX}")
t0 = time.time()

for case_idx, (true_ti, lf_idx, full_v) in enumerate(test_cases):
    if case_idx % 50 == 0:
        elapsed = time.time() - t0
        eta = elapsed / max(case_idx, 1) * (N_TEST - case_idx)
        print(f"  Case {case_idx}/{N_TEST}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    rng_case = np.random.RandomState(case_idx * 31 + 7)

    # ── BOED_AIS ──
    sel_nodes, sel_vals = [], []
    remain = list(CANDIDATES)
    for k in range(K_MAX):
        eig_scores, _ = eig_all_candidates(sel_nodes, sel_vals, remain, rng_case)
        best_idx = int(np.argmax(eig_scores))
        node = remain[best_idx]
        boed_selections[k].append(node)
        sel_nodes.append(node)
        sel_vals.append(full_v[node])
        remain.remove(node)
        p = ais_posterior(sel_nodes, sel_vals)
        results['BOED_AIS']['acc'][k]     += int(np.argmax(p) == true_ti)
        results['BOED_AIS']['entropy'][k] += entropy(p)

    # ── Random ──
    rng_rand = np.random.RandomState(case_idx * 17 + 3)
    perm = list(CANDIDATES); rng_rand.shuffle(perm)
    sel_nodes, sel_vals = [], []
    for k in range(K_MAX):
        node = perm[k]
        sel_nodes.append(node); sel_vals.append(full_v[node])
        p = ais_posterior(sel_nodes, sel_vals)
        results['Random']['acc'][k]     += int(np.argmax(p) == true_ti)
        results['Random']['entropy'][k] += entropy(p)

    # ── GreedyLoop ──
    remain = list(CANDIDATES)
    sel_nodes, sel_vals = [], []
    for k in range(K_MAX):
        node = greedy_loop_select(sel_nodes, remain)
        sel_nodes.append(node); sel_vals.append(full_v[node])
        remain.remove(node)
        p = ais_posterior(sel_nodes, sel_vals)
        results['GreedyLoop']['acc'][k]     += int(np.argmax(p) == true_ti)
        results['GreedyLoop']['entropy'][k] += entropy(p)

for s in strategies:
    results[s]['acc']     /= N_TEST
    results[s]['entropy'] /= N_TEST

total_time = time.time() - t0

# ── 输出 ──────────────────────────────────────────────────────────────────────
lines = []
lines.append("=" * 72)
lines.append("IP-A AIS-BOED v2 RESULTS (Variable Load lf~U(0.8,1.2), Exact EIG)")
lines.append(f"N_TEST={N_TEST}, N_MC={N_MC}, N_LF_C={N_LF_C}, K_MAX={K_MAX}, time={total_time:.0f}s")
lines.append("=" * 72)
lines.append("")
lines.append("Top-1 Accuracy (AIS posterior marginalizing lf):")
hdr = f"{'K':>4}  {'BOED_AIS':>10}  {'Random':>10}  {'GreedyLoop':>12}"
lines.append(hdr); lines.append("-" * len(hdr))
for k in range(K_MAX):
    lines.append(f"{k+1:>4}  {results['BOED_AIS']['acc'][k]:>10.3f}  "
                 f"{results['Random']['acc'][k]:>10.3f}  "
                 f"{results['GreedyLoop']['acc'][k]:>12.3f}")

lines.append("")
lines.append("Posterior Entropy H(K) [lower = more informative]:")
lines.append(hdr); lines.append("-" * len(hdr))
for k in range(K_MAX):
    lines.append(f"{k+1:>4}  {results['BOED_AIS']['entropy'][k]:>10.3f}  "
                 f"{results['Random']['entropy'][k]:>10.3f}  "
                 f"{results['GreedyLoop']['entropy'][k]:>12.3f}")

lines.append("")
lines.append("BOED sensor savings:")
for thresh in [0.60, 0.70, 0.80]:
    boed_k   = next((k+1 for k in range(K_MAX) if results['BOED_AIS']['acc'][k] >= thresh), None)
    rand_k   = next((k+1 for k in range(K_MAX) if results['Random']['acc'][k]   >= thresh), None)
    greedy_k = next((k+1 for k in range(K_MAX) if results['GreedyLoop']['acc'][k] >= thresh), None)
    b = f"K={boed_k}" if boed_k else "never"
    r = f"K={rand_k}" if rand_k else "never"
    g = f"K={greedy_k}" if greedy_k else "never"
    savings = (rand_k - boed_k) if (boed_k and rand_k) else None
    lines.append(f"  {thresh*100:.0f}% acc: BOED@{b}  Random@{r}  Greedy@{g}"
                 + (f"  → saves {savings} sensors" if savings else ""))

lines.append("")
lines.append("Gate: BOED_AIS > Random at K=4 and K=7?")
g4 = results['BOED_AIS']['acc'][3] > results['Random']['acc'][3]
g7 = results['BOED_AIS']['acc'][6] > results['Random']['acc'][6]
lines.append(f"  K=4: BOED={results['BOED_AIS']['acc'][3]:.3f} vs Random={results['Random']['acc'][3]:.3f} → {'PASS' if g4 else 'FAIL'}")
lines.append(f"  K=7: BOED={results['BOED_AIS']['acc'][6]:.3f} vs Random={results['Random']['acc'][6]:.3f} → {'PASS' if g7 else 'FAIL'}")
lines.append(f"  Overall: {'PASS' if g4 and g7 else 'FAIL'}")

lines.append("")
lines.append("Most frequently selected nodes at each step (BOED):")
for k in range(min(6, K_MAX)):
    counts = {}
    for n in boed_selections[k]: counts[n] = counts.get(n,0)+1
    top3 = sorted(counts.items(), key=lambda x: -x[1])[:3]
    lines.append(f"  Step {k+1}: " + ", ".join(f"node{n}({c})" for n,c in top3))
lines.append("=" * 72)

output = "\n".join(lines)
print(output)
out_path = f"{SAVE_DIR}\\boed_v2_result.txt"
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(output)
print(f"\nSaved: {out_path}")
