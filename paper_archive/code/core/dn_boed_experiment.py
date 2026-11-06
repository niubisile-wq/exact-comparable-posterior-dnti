# -*- coding: utf-8 -*-
"""
IP-A 正式实验：BOED序贯传感器选择（批量化版）
对比：BOED(BA bound) vs 随机 vs 贪婪回路覆盖 vs Uniform
批量化MC评估：32候选节点 × N_MC样本 一次前向
"""
import copy, time, warnings
import numpy as np
import torch
import torch.nn as nn
import pandapower as pp
import networkx as nx
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = r"<LOCAL_DESKTOP>\nre_model.pt"
OUT_PATH   = r"<LOCAL_DESKTOP>\boed_result.txt"
N_MC   = 200   # 每候选节点的EIG蒙特卡洛样本数
N_TEST = 50    # 测试拓扑数（批量化后50个足够稳定）
K_MAX  = 12    # 最多选12个节点
SIGMA  = 0.009
print(f"Device: {DEVICE}")

# ── 网络构建 ────────────────────────────────────────────────────────────

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

class MaskedNRE(nn.Module):
    def __init__(self, n_topos):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(66, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, n_topos))
    def forward(self, x): return self.net(x)

# ── 加载 ────────────────────────────────────────────────────────────────
print("Loading network and model...")
net33, ne33, te33 = build_ieee33()
topos_raw = enum_topos(ne33, te33)
N_TOPOS = len(topos_raw)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
V_all = ckpt['voltages']  # (N_TOPOS, 33)
model = MaskedNRE(N_TOPOS).to(DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"N_TOPOS={N_TOPOS}, V_all={V_all.shape}")

CANDIDATES = list(range(1, 33))  # 节点1-32

# ── 批量EIG计算（核心加速）─────────────────────────────────────────────

def batch_eig(current_nodes, current_vals, candidates, rng):
    """
    批量计算所有候选节点的EIG（BA bound）
    一次前向完成所有计算: (|candidates| × N_MC) 个样本
    返回 eig_scores: shape=(|candidates|,)
    """
    n_cand = len(candidates)
    # 预采样 N_MC 个拓扑（所有候选节点共享同一组采样）
    mc_topos = rng.randint(0, N_TOPOS, N_MC)

    # 构建输入矩阵: (n_cand * N_MC, 66)
    X = np.zeros((n_cand * N_MC, 66), dtype=np.float32)

    for ci, cand in enumerate(candidates):
        for mi, ti_mc in enumerate(mc_topos):
            row = ci * N_MC + mi
            # 已选节点
            if len(current_nodes) > 0:
                X[row, current_nodes] = current_vals
                X[row, 33 + np.array(current_nodes)] = 1.0
            # 候选节点的测量（加噪）
            X[row, cand] = V_all[ti_mc, cand] + rng.normal(0, SIGMA)
            X[row, 33 + cand] = 1.0

    # 批量前向
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
        log_probs = torch.log_softmax(logits, dim=1).cpu().numpy()
        # (n_cand * N_MC, N_TOPOS)

    # 提取每个(候选,MC样本)对应的真实拓扑的对数概率
    eig_scores = np.zeros(n_cand)
    for ci in range(n_cand):
        for mi, ti_mc in enumerate(mc_topos):
            row = ci * N_MC + mi
            eig_scores[ci] += log_probs[row, ti_mc]
        eig_scores[ci] /= N_MC

    return eig_scores

def nre_posterior_batch(obs_nodes_list, obs_vals_list):
    """批量计算一组观测的后验，返回(N,N_TOPOS)"""
    X = np.zeros((len(obs_nodes_list), 66), dtype=np.float32)
    for i, (nodes, vals) in enumerate(zip(obs_nodes_list, obs_vals_list)):
        if len(nodes) > 0:
            X[i, nodes] = vals
            X[i, 33 + np.array(nodes)] = 1.0
    with torch.no_grad():
        probs = torch.softmax(
            model(torch.tensor(X, dtype=torch.float32).to(DEVICE)), dim=1
        ).cpu().numpy()
    return probs

def posterior_entropy(p):
    p = np.clip(p, 1e-10, 1.0)
    return -np.sum(p * np.log(p))

# ── 基本回路覆盖策略 ────────────────────────────────────────────────────

def get_fundamental_loops(ne, te):
    G = nx.Graph(); G.add_edges_from(ne)
    loops = []
    for tie in te:
        path = nx.shortest_path(G, tie[0], tie[1])
        loops.append(set(path))
    return loops

LOOPS = get_fundamental_loops(ne33, te33)

def greedy_loop_select(current_nodes, candidates):
    covered = [any(n in loop for n in current_nodes) for loop in LOOPS]
    uncov_loops = [l for l, c in zip(LOOPS, covered) if not c]
    if not uncov_loops:
        # 回路全覆盖后，选距已选节点最远的节点（增加多样性）
        if current_nodes:
            best = max(candidates, key=lambda c: min(abs(c - n) for n in current_nodes))
        else:
            best = candidates[0]
        return best
    best_node, best_count = candidates[0], -1
    for c in candidates:
        count = sum(1 for l in uncov_loops if c in l)
        if count > best_count:
            best_count, best_node = count, c
    return best_node

# ── 主实验 ──────────────────────────────────────────────────────────────
rng_main = np.random.RandomState(2024)

# 预生成测试场景
print(f"\nGenerating {N_TEST} test scenarios...")
test_cases = []
for _ in range(N_TEST):
    ti = rng_main.randint(0, N_TOPOS)
    full_v = V_all[ti] + rng_main.normal(0, SIGMA, 33)
    test_cases.append((ti, full_v))

strategies = ['BOED', 'Random', 'GreedyLoop', 'Uniform']
results = {s: {'acc': np.zeros(K_MAX), 'entropy': np.zeros(K_MAX)} for s in strategies}

print(f"Running {N_TEST} test cases x {K_MAX} steps x 4 strategies (batched EIG)...")
t0 = time.time()

for case_idx, (true_ti, full_v) in enumerate(test_cases):
    elapsed = time.time() - t0
    if case_idx % 5 == 0 and case_idx > 0:
        eta = elapsed / case_idx * (N_TEST - case_idx)
        print(f"  Case {case_idx}/{N_TEST}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
    elif case_idx == 0:
        print(f"  Case 0/{N_TEST}  starting...")

    rng_case = np.random.RandomState(case_idx * 7 + 13)

    # ── BOED ──
    sel_nodes, sel_vals = [], []
    remain = list(CANDIDATES)
    for k in range(K_MAX):
        eig_scores = batch_eig(sel_nodes, np.array(sel_vals, dtype=np.float32) if sel_vals else np.array([]),
                               remain, rng_case)
        best_idx = np.argmax(eig_scores)
        node = remain[best_idx]
        sel_nodes.append(node); sel_vals.append(full_v[node])
        remain.remove(node)
        p = nre_posterior_batch([sel_nodes], [np.array(sel_vals, dtype=np.float32)])[0]
        results['BOED']['acc'][k] += int(np.argmax(p) == true_ti)
        results['BOED']['entropy'][k] += posterior_entropy(p)

    # ── Random ──
    rng_rand = np.random.RandomState(case_idx * 17 + 5)
    perm = list(CANDIDATES); rng_rand.shuffle(perm)
    sel_nodes, sel_vals = [], []
    for k in range(K_MAX):
        node = perm[k]
        sel_nodes.append(node); sel_vals.append(full_v[node])
        p = nre_posterior_batch([sel_nodes], [np.array(sel_vals, dtype=np.float32)])[0]
        results['Random']['acc'][k] += int(np.argmax(p) == true_ti)
        results['Random']['entropy'][k] += posterior_entropy(p)

    # ── GreedyLoop ──
    remain = list(CANDIDATES)
    sel_nodes, sel_vals = [], []
    for k in range(K_MAX):
        node = greedy_loop_select(sel_nodes, remain)
        sel_nodes.append(node); sel_vals.append(full_v[node])
        remain.remove(node)
        p = nre_posterior_batch([sel_nodes], [np.array(sel_vals, dtype=np.float32)])[0]
        results['GreedyLoop']['acc'][k] += int(np.argmax(p) == true_ti)
        results['GreedyLoop']['entropy'][k] += posterior_entropy(p)

    # ── Uniform（均匀间隔）──
    step_size = max(1, len(CANDIDATES) // K_MAX)
    uniform_order = CANDIDATES[::step_size][:K_MAX]
    if len(uniform_order) < K_MAX:
        rest = [n for n in CANDIDATES if n not in uniform_order]
        uniform_order += rest[:K_MAX - len(uniform_order)]
    sel_nodes, sel_vals = [], []
    for k in range(K_MAX):
        node = uniform_order[k]
        sel_nodes.append(node); sel_vals.append(full_v[node])
        p = nre_posterior_batch([sel_nodes], [np.array(sel_vals, dtype=np.float32)])[0]
        results['Uniform']['acc'][k] += int(np.argmax(p) == true_ti)
        results['Uniform']['entropy'][k] += posterior_entropy(p)

# 平均
for s in strategies:
    results[s]['acc'] /= N_TEST
    results[s]['entropy'] /= N_TEST

# ── 输出 ────────────────────────────────────────────────────────────────
total_time = time.time() - t0
lines = []
lines.append("=" * 72)
lines.append("IP-A BOED EXPERIMENT RESULTS (Batched MC)")
lines.append(f"N_TEST={N_TEST}, N_MC={N_MC}, K_MAX={K_MAX}, time={total_time:.0f}s")
lines.append("=" * 72)
lines.append("")
lines.append("Top-1 Accuracy vs K:")
header = f"{'K':>4}  {'BOED':>10}  {'Random':>10}  {'GreedyLoop':>12}  {'Uniform':>10}"
lines.append(header); lines.append("-" * len(header))
for k in range(K_MAX):
    lines.append(f"{k+1:>4}  {results['BOED']['acc'][k]:>10.3f}  "
                 f"{results['Random']['acc'][k]:>10.3f}  "
                 f"{results['GreedyLoop']['acc'][k]:>12.3f}  "
                 f"{results['Uniform']['acc'][k]:>10.3f}")

lines.append("")
lines.append("Posterior Entropy H(K) [lower = more certain]:")
lines.append(header); lines.append("-" * len(header))
for k in range(K_MAX):
    lines.append(f"{k+1:>4}  {results['BOED']['entropy'][k]:>10.3f}  "
                 f"{results['Random']['entropy'][k]:>10.3f}  "
                 f"{results['GreedyLoop']['entropy'][k]:>12.3f}  "
                 f"{results['Uniform']['entropy'][k]:>10.3f}")

lines.append("")
lines.append("BOED sensor savings (steps to reach threshold vs best baseline):")
for thresh in [0.70, 0.80]:
    boed_k = next((k+1 for k in range(K_MAX) if results['BOED']['acc'][k] >= thresh), None)
    for s in ['Random', 'GreedyLoop']:
        base_k = next((k+1 for k in range(K_MAX) if results[s]['acc'][k] >= thresh), None)
        if boed_k and base_k:
            lines.append(f"  {thresh*100:.0f}% acc: BOED@K={boed_k} vs {s}@K={base_k}"
                         f"  (saves {base_k-boed_k} sensors)")
        elif boed_k and not base_k:
            lines.append(f"  {thresh*100:.0f}% acc: BOED@K={boed_k} vs {s}=never reached")

lines.append("=" * 72)

output = "\n".join(lines)
print(output)
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    f.write(output)
print(f"\nSaved: {OUT_PATH}")
