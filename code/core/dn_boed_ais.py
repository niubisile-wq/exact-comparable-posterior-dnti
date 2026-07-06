# -*- coding: utf-8 -*-
"""
IP-A 正式实验 v2：精确AIS-BOED序贯传感器选择
核心修正：用精确AIS计算EIG（穷举32种拓扑），不依赖NRE质量
故事线：BOED(AIS)离线规划最优安装位置 → NRE在线实时推断
"""
import copy, time, warnings
import numpy as np
import torch
import pandapower as pp
import networkx as nx
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = r"<LOCAL_DESKTOP>\nre_model.pt"
OUT_PATH   = r"<LOCAL_DESKTOP>\boed_ais_result.txt"
N_MC   = 100   # 每候选节点的x_j蒙特卡洛样本（纯向量运算，快）
N_TEST = 100   # 测试场景数
K_MAX  = 12    # 最多选12个节点
SIGMA  = 0.009
print(f"Device: {DEVICE}")

# ── 网络构建 ─────────────────────────────────────────────────────────────

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

# ── 加载预计算电压 ────────────────────────────────────────────────────────
print("Loading precomputed voltages from nre_model.pt...")
ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
V_all = ckpt['voltages']  # (N_TOPOS, 33)
N_TOPOS = V_all.shape[0]
print(f"N_TOPOS={N_TOPOS}, V_all={V_all.shape}")
CANDIDATES = list(range(1, 33))

# ── 精确AIS后验（向量化，极快）────────────────────────────────────────────

def ais_posterior(obs_nodes, obs_vals, sigma=SIGMA):
    """精确后验：shape=(N_TOPOS,)"""
    if len(obs_nodes) == 0:
        return np.ones(N_TOPOS) / N_TOPOS
    diff = (V_all[:, obs_nodes] - obs_vals) / sigma  # (N_TOPOS, K)
    ll = -0.5 * np.sum(diff**2, axis=1)
    ll -= ll.max()
    w = np.exp(ll)
    return w / w.sum()

def entropy(p):
    p = np.clip(p, 1e-10, 1.0)
    return -np.sum(p * np.log(p))

# ── 精确AIS EIG（核心）───────────────────────────────────────────────────

def ais_eig_all_candidates(obs_nodes, obs_vals, candidates, rng, n_mc=N_MC):
    """
    对所有候选节点并行计算精确EIG：
    EIG(j) = H(τ|x_S) - E_{x_j~p(x_j|x_S)}[H(τ|x_S, x_j)]

    E_{x_j|x_S}[H(τ|x_S,x_j)] ≈ sum_τ' p(τ'|x_S) * (1/n_mc) * sum_m H(τ|x_S, x_j^{τ',m})
    其中 x_j^{τ',m} ~ N(V_all[τ',j], σ)

    全部向量化：O(N_TOPOS × n_mc × |candidates|) 次标量运算
    """
    p_cur = ais_posterior(obs_nodes, obs_vals)  # 当前后验
    H_cur = entropy(p_cur)                       # 当前熵

    n_cand = len(candidates)
    H_after = np.zeros(n_cand)  # E[H(τ|S∪{j})]

    # 对每个拓扑τ'，采样n_mc个x_j值
    # 然后对每个(τ', m, j)三元组计算新后验的熵
    for ti_idx in range(N_TOPOS):
        w_ti = p_cur[ti_idx]
        if w_ti < 1e-8:
            continue  # 当前后验概率极小的拓扑跳过（贡献忽略）

        # 为所有候选节点同时采样：shape=(n_mc, n_cand)
        noise = rng.normal(0, SIGMA, (n_mc, n_cand))
        # x_j^m = V_all[τ', j] + noise
        x_aug = V_all[ti_idx, candidates] + noise  # (n_mc, n_cand)

        for ci, cand in enumerate(candidates):
            h_sum = 0.0
            new_nodes = list(obs_nodes) + [cand]
            for m in range(n_mc):
                new_vals = np.append(obs_vals, x_aug[m, ci])
                p_new = ais_posterior(new_nodes, new_vals)
                h_sum += entropy(p_new)
            H_after[ci] += w_ti * (h_sum / n_mc)

    eig = H_cur - H_after  # 信息增益 = 当前熵 - 期望后熵
    return eig, H_cur

# ── 贪婪回路覆盖策略 ──────────────────────────────────────────────────────

def get_fundamental_loops(ne, te):
    G = nx.Graph(); G.add_edges_from(ne)
    return [set(nx.shortest_path(G, tie[0], tie[1])) for tie in te]

print("Building network for loop structure...")
net33, ne33, te33 = build_ieee33()
LOOPS = get_fundamental_loops(ne33, te33)

def greedy_loop_select(current_nodes, candidates):
    covered = [any(n in loop for n in current_nodes) for loop in LOOPS]
    uncov = [l for l, c in zip(LOOPS, covered) if not c]
    if not uncov:
        if current_nodes:
            return max(candidates, key=lambda c: min(abs(c - n) for n in current_nodes))
        return candidates[0]
    best_node, best_count = candidates[0], -1
    for c in candidates:
        cnt = sum(1 for l in uncov if c in l)
        if cnt > best_count:
            best_count, best_node = cnt, c
    return best_node

# ── 主实验 ───────────────────────────────────────────────────────────────
rng_main = np.random.RandomState(2024)
print(f"\nGenerating {N_TEST} test scenarios...")
test_cases = []
for _ in range(N_TEST):
    ti = rng_main.randint(0, N_TOPOS)
    full_v = V_all[ti] + rng_main.normal(0, SIGMA, 33)
    test_cases.append((ti, full_v))

strategies = ['BOED_AIS', 'Random', 'GreedyLoop']
results = {s: {'acc': np.zeros(K_MAX), 'entropy': np.zeros(K_MAX),
               'eig': np.zeros(K_MAX)} for s in strategies}
# 额外记录BOED每步选的节点分布（用于论文可视化）
boed_selections = [[] for _ in range(K_MAX)]

print(f"Running {N_TEST} cases × {K_MAX} steps (AIS exact EIG, N_MC={N_MC})...")
t0 = time.time()

for case_idx, (true_ti, full_v) in enumerate(test_cases):
    if case_idx % 10 == 0:
        elapsed = time.time() - t0
        eta = elapsed / max(case_idx, 1) * (N_TEST - case_idx)
        print(f"  Case {case_idx}/{N_TEST}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    rng_case = np.random.RandomState(case_idx * 31 + 7)

    # ── BOED_AIS：精确EIG序贯选择 ──
    sel_nodes, sel_vals = [], np.array([])
    remain = list(CANDIDATES)
    for k in range(K_MAX):
        eig_scores, _ = ais_eig_all_candidates(
            sel_nodes, sel_vals, remain, rng_case)
        best_idx = np.argmax(eig_scores)
        node = remain[best_idx]
        boed_selections[k].append(node)
        sel_nodes.append(node)
        sel_vals = np.append(sel_vals, full_v[node])
        remain.remove(node)
        p = ais_posterior(sel_nodes, sel_vals)
        results['BOED_AIS']['acc'][k] += int(np.argmax(p) == true_ti)
        results['BOED_AIS']['entropy'][k] += entropy(p)
        results['BOED_AIS']['eig'][k] += eig_scores[best_idx]

    # ── Random ──
    rng_rand = np.random.RandomState(case_idx * 17 + 3)
    perm = list(CANDIDATES); rng_rand.shuffle(perm)
    sel_nodes, sel_vals = [], np.array([])
    for k in range(K_MAX):
        node = perm[k]
        sel_nodes.append(node)
        sel_vals = np.append(sel_vals, full_v[node])
        p = ais_posterior(sel_nodes, sel_vals)
        results['Random']['acc'][k] += int(np.argmax(p) == true_ti)
        results['Random']['entropy'][k] += entropy(p)

    # ── GreedyLoop ──
    remain = list(CANDIDATES)
    sel_nodes, sel_vals = [], np.array([])
    for k in range(K_MAX):
        node = greedy_loop_select(sel_nodes, remain)
        sel_nodes.append(node)
        sel_vals = np.append(sel_vals, full_v[node])
        remain.remove(node)
        p = ais_posterior(sel_nodes, sel_vals)
        results['GreedyLoop']['acc'][k] += int(np.argmax(p) == true_ti)
        results['GreedyLoop']['entropy'][k] += entropy(p)

for s in strategies:
    results[s]['acc'] /= N_TEST
    results[s]['entropy'] /= N_TEST
    results[s]['eig'] /= N_TEST

# ── 输出 ─────────────────────────────────────────────────────────────────
total_time = time.time() - t0
lines = []
lines.append("=" * 72)
lines.append("IP-A AIS-BOED EXPERIMENT RESULTS (Exact EIG)")
lines.append(f"N_TEST={N_TEST}, N_MC={N_MC}, K_MAX={K_MAX}, time={total_time:.0f}s")
lines.append("=" * 72)
lines.append("")
lines.append("Top-1 Accuracy (AIS posterior, ground truth):")
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
lines.append("BOED sensor savings (K to reach accuracy threshold):")
for thresh in [0.60, 0.70, 0.80]:
    boed_k = next((k+1 for k in range(K_MAX) if results['BOED_AIS']['acc'][k] >= thresh), None)
    rand_k  = next((k+1 for k in range(K_MAX) if results['Random']['acc'][k] >= thresh), None)
    greedy_k = next((k+1 for k in range(K_MAX) if results['GreedyLoop']['acc'][k] >= thresh), None)
    bstr = f"K={boed_k}" if boed_k else "never"
    rstr = f"K={rand_k}" if rand_k else "never"
    gstr = f"K={greedy_k}" if greedy_k else "never"
    savings_r = (rand_k - boed_k) if (boed_k and rand_k) else None
    lines.append(f"  {thresh*100:.0f}% acc: BOED@{bstr}  Random@{rstr}  Greedy@{gstr}"
                 + (f"  → saves {savings_r} sensors vs Random" if savings_r else ""))

lines.append("")
lines.append("Most frequently selected nodes at each step (BOED):")
for k in range(min(6, K_MAX)):
    counts = {}
    for n in boed_selections[k]:
        counts[n] = counts.get(n, 0) + 1
    top3 = sorted(counts.items(), key=lambda x: -x[1])[:3]
    lines.append(f"  Step {k+1}: " + ", ".join(f"node{n}({c})" for n,c in top3))

lines.append("=" * 72)

output = "\n".join(lines)
print(output)
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    f.write(output)
print(f"\nSaved: {OUT_PATH}")
