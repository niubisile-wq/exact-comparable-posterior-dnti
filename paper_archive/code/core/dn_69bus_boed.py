# -*- coding: utf-8 -*-
"""
Step3 IP-A: 69-bus AIS-BOED 变负荷正式实验
加载已有V_library，N_TEST=200, N_MC=100, N_LF_C=11
"""
import copy, time, warnings, os
import numpy as np
import pandapower as pp
import networkx as nx
warnings.filterwarnings('ignore')

SIGMA    = 0.009
LF_MIN   = 0.8; LF_MAX = 1.2; N_LF = 101; N_LF_C = 11
N_TEST   = 200; N_MC = 100; K_MAX = 12
N_BUS    = 69
SAVE_DIR = r"<LOCAL_WORKSPACE>"
print(f"IP-A 69-bus: N_TEST={N_TEST} N_MC={N_MC} N_LF_C={N_LF_C}")

# ── 网络构建（仅用于拓扑枚举和回路结构）────────────────────────────────────
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
dat = np.load(f"{SAVE_DIR}\\v_library_69bus.npz")
V_lib_fine = dat['V_library']           # (N_TOPOS, N_LF, N_BUS)
base_P_norm = dat['base_P_norm']
lf_grid_fine = dat['lf_grid']
N_TOPOS = V_lib_fine.shape[0]
print(f"V_lib_fine: {V_lib_fine.shape}  N_TOPOS={N_TOPOS}")

# 粗网格（EIG用）
lf_grid_c = np.linspace(LF_MIN, LF_MAX, N_LF_C)
lf_idx_c  = np.round(np.linspace(0, N_LF-1, N_LF_C)).astype(int)
V_lib_c   = V_lib_fine[:, lf_idx_c, :]   # (N_TOPOS, N_LF_C, N_BUS)
print(f"V_lib_c: {V_lib_c.shape}")

# 拓扑枚举（用于贪婪回路覆盖）
print("Building network for loop structure...")
net69, ne69, te69 = build_ieee69()
CANDIDATES = list(range(1, N_BUS))       # 68个候选节点

# ── AIS后验 ───────────────────────────────────────────────────────────────────
def ais_posterior(obs_nodes, obs_vals):
    if not obs_nodes: return np.ones(N_TOPOS)/N_TOPOS
    diff = (V_lib_c[:, :, obs_nodes] - np.array(obs_vals)) / SIGMA
    ll = -0.5*np.sum(diff**2, axis=2); ll -= ll.max()
    w = np.exp(ll).sum(axis=1); return w/w.sum()

def ais_posterior_joint(obs_nodes, obs_vals):
    if not obs_nodes: return np.ones(N_TOPOS*N_LF_C)/(N_TOPOS*N_LF_C)
    diff = (V_lib_c[:, :, obs_nodes] - np.array(obs_vals)) / SIGMA
    ll = -0.5*np.sum(diff**2, axis=2); ll -= ll.max()
    w = np.exp(ll).flatten(); return w/w.sum()

def entropy(p): p=np.clip(p,1e-10,1.0); return -np.sum(p*np.log(p))

# ── 精确EIG ───────────────────────────────────────────────────────────────────
def eig_all_candidates(obs_nodes, obs_vals, candidates, rng):
    joint_w = ais_posterior_joint(obs_nodes, obs_vals)
    p_cur   = joint_w.reshape(N_TOPOS, N_LF_C).sum(axis=1); p_cur /= p_cur.sum()
    H_cur   = entropy(p_cur)
    mc_flat = rng.choice(N_TOPOS*N_LF_C, size=N_MC, p=joint_w)
    mc_tau  = mc_flat // N_LF_C; mc_lf = mc_flat % N_LF_C
    n_cand  = len(candidates)
    x_j_base = V_lib_c[mc_tau, mc_lf, :][:, candidates]
    x_j_all  = x_j_base + rng.normal(0, SIGMA, (N_MC, n_cand))
    H_after  = np.zeros(n_cand)
    obs_nodes_l = list(obs_nodes); obs_vals_a = np.array(obs_vals)
    for ci, cand in enumerate(candidates):
        new_nodes = obs_nodes_l + [cand]; h = 0.0
        for m in range(N_MC):
            p_new = ais_posterior(new_nodes, np.append(obs_vals_a, x_j_all[m,ci]))
            h += entropy(p_new)
        H_after[ci] = h/N_MC
    return H_cur - H_after, H_cur

# ── 贪婪回路覆盖 ──────────────────────────────────────────────────────────────
def get_loops(ne, te):
    G = nx.Graph(); G.add_edges_from(ne)
    return [set(nx.shortest_path(G, tie[0], tie[1])) for tie in te]

LOOPS = get_loops(ne69, te69)

def greedy_loop_select(current_nodes, candidates):
    covered = [any(n in loop for n in current_nodes) for loop in LOOPS]
    uncov = [l for l,c in zip(LOOPS, covered) if not c]
    if not uncov:
        return max(candidates, key=lambda c: min((abs(c-n) for n in current_nodes), default=0)) if current_nodes else candidates[0]
    best, best_cnt = candidates[0], -1
    for c in candidates:
        cnt = sum(1 for l in uncov if c in l)
        if cnt > best_cnt: best_cnt, best = cnt, c
    return best

# ── 主实验 ────────────────────────────────────────────────────────────────────
rng_main = np.random.RandomState(2024)
print(f"\nGenerating {N_TEST} test scenarios...")
test_cases = []
for _ in range(N_TEST):
    ti     = rng_main.randint(0, N_TOPOS)
    lf_idx = rng_main.randint(0, N_LF)
    full_v = V_lib_fine[ti, lf_idx, :] + rng_main.normal(0, SIGMA, N_BUS)
    test_cases.append((ti, lf_idx, full_v))

strategies = ['BOED_AIS', 'Random', 'GreedyLoop']
results = {s: {'acc': np.zeros(K_MAX), 'entropy': np.zeros(K_MAX)} for s in strategies}
boed_selections = [[] for _ in range(K_MAX)]

print(f"Running... N_TEST={N_TEST} N_MC={N_MC} K_MAX={K_MAX} candidates={len(CANDIDATES)}")
t0 = time.time()

for case_idx, (true_ti, lf_idx, full_v) in enumerate(test_cases):
    if case_idx % 20 == 0:
        elapsed = time.time()-t0
        eta = elapsed/max(case_idx,1)*(N_TEST-case_idx)
        print(f"  Case {case_idx}/{N_TEST}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    rng_case = np.random.RandomState(case_idx*31+7)

    # BOED_AIS
    sel_nodes, sel_vals = [], []
    remain = list(CANDIDATES)
    for k in range(K_MAX):
        eig_scores, _ = eig_all_candidates(sel_nodes, sel_vals, remain, rng_case)
        node = remain[int(np.argmax(eig_scores))]
        boed_selections[k].append(node)
        sel_nodes.append(node); sel_vals.append(full_v[node]); remain.remove(node)
        p = ais_posterior(sel_nodes, sel_vals)
        results['BOED_AIS']['acc'][k]     += int(np.argmax(p)==true_ti)
        results['BOED_AIS']['entropy'][k] += entropy(p)

    # Random
    rng_rand = np.random.RandomState(case_idx*17+3)
    perm = list(CANDIDATES); rng_rand.shuffle(perm)
    sel_nodes, sel_vals = [], []
    for k in range(K_MAX):
        sel_nodes.append(perm[k]); sel_vals.append(full_v[perm[k]])
        p = ais_posterior(sel_nodes, sel_vals)
        results['Random']['acc'][k]     += int(np.argmax(p)==true_ti)
        results['Random']['entropy'][k] += entropy(p)

    # GreedyLoop
    remain = list(CANDIDATES)
    sel_nodes, sel_vals = [], []
    for k in range(K_MAX):
        node = greedy_loop_select(sel_nodes, remain)
        sel_nodes.append(node); sel_vals.append(full_v[node]); remain.remove(node)
        p = ais_posterior(sel_nodes, sel_vals)
        results['GreedyLoop']['acc'][k]     += int(np.argmax(p)==true_ti)
        results['GreedyLoop']['entropy'][k] += entropy(p)

for s in strategies:
    results[s]['acc'] /= N_TEST; results[s]['entropy'] /= N_TEST

total_time = time.time()-t0

# ── 输出 ──────────────────────────────────────────────────────────────────────
lines = ["="*72,
         "IP-A AIS-BOED 69-bus RESULTS (Variable Load lf~U(0.8,1.2))",
         f"N_TEST={N_TEST}, N_MC={N_MC}, N_LF_C={N_LF_C}, K_MAX={K_MAX}, time={total_time:.0f}s",
         "="*72, "",
         "Top-1 Accuracy:"]
hdr = f"{'K':>4}  {'BOED_AIS':>10}  {'Random':>10}  {'GreedyLoop':>12}"
lines += [hdr, "-"*len(hdr)]
for k in range(K_MAX):
    lines.append(f"{k+1:>4}  {results['BOED_AIS']['acc'][k]:>10.3f}  "
                 f"{results['Random']['acc'][k]:>10.3f}  "
                 f"{results['GreedyLoop']['acc'][k]:>12.3f}")
lines += ["", "Posterior Entropy H(K) [lower=more informative]:", hdr, "-"*len(hdr)]
for k in range(K_MAX):
    lines.append(f"{k+1:>4}  {results['BOED_AIS']['entropy'][k]:>10.3f}  "
                 f"{results['Random']['entropy'][k]:>10.3f}  "
                 f"{results['GreedyLoop']['entropy'][k]:>12.3f}")
lines.append("\nBOED sensor savings:")
for thresh in [0.20, 0.30, 0.40]:
    bk = next((k+1 for k in range(K_MAX) if results['BOED_AIS']['acc'][k]>=thresh), None)
    rk = next((k+1 for k in range(K_MAX) if results['Random']['acc'][k]>=thresh), None)
    gk = next((k+1 for k in range(K_MAX) if results['GreedyLoop']['acc'][k]>=thresh), None)
    b=f"K={bk}" if bk else "never"; r=f"K={rk}" if rk else "never"; g=f"K={gk}" if gk else "never"
    sv=(rk-bk) if (bk and rk) else None
    lines.append(f"  {thresh*100:.0f}% acc: BOED@{b}  Random@{r}  Greedy@{g}"
                 + (f"  → saves {sv} sensors" if sv else ""))
g4 = results['BOED_AIS']['acc'][3] > results['Random']['acc'][3]
g7 = results['BOED_AIS']['acc'][6] > results['Random']['acc'][6]
lines += ["", "Gate: BOED_AIS > Random at K=4 and K=7?",
          f"  K=4: BOED={results['BOED_AIS']['acc'][3]:.3f} vs Random={results['Random']['acc'][3]:.3f} → {'PASS' if g4 else 'FAIL'}",
          f"  K=7: BOED={results['BOED_AIS']['acc'][6]:.3f} vs Random={results['Random']['acc'][6]:.3f} → {'PASS' if g7 else 'FAIL'}",
          f"  Overall: {'PASS' if g4 and g7 else 'FAIL'}", "", "Most frequently selected nodes (BOED):"]
for k in range(min(6,K_MAX)):
    counts={}
    for n in boed_selections[k]: counts[n]=counts.get(n,0)+1
    top3=sorted(counts.items(),key=lambda x:-x[1])[:3]
    lines.append(f"  Step {k+1}: "+", ".join(f"node{n}({c})" for n,c in top3))
lines.append("="*72)

output = "\n".join(lines)
print(output)
out_path = f"{SAVE_DIR}\\boed_69bus_result.txt"
with open(out_path,'w',encoding='utf-8') as f: f.write(output)
print(f"\nSaved: {out_path}")
