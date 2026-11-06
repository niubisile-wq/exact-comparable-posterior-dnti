# -*- coding: utf-8 -*-
"""
Step5-B修订版：最大方差贪婪（MVG）传感器选址基线
替代错误的Fisher-OED（Fisher信息对离散拓扑参数无效，且产生次于随机的结果）

MVG：序贯贪婪，每步选在当前后验下电压方差最大的候选节点
  score(j) = Var_{τ~p(τ|x_S)}[V_τ[j]]  ← 加权方差，权重为当前后验
当前后验 = 精确枚举（EnumBF）

与BOED的区别：
  MVG  → 近似信息增益（仅用方差，忽略测量噪声和后验形状）
  BOED → 精确期望信息增益（EIG），考虑完整后验分布
"""
import warnings, time
import numpy as np
warnings.filterwarnings('ignore')

SAVE_DIR = r"<LOCAL_WORKSPACE>"
SIGMA    = 0.009; N_LF = 101; K_MAX = 12

def run_mvg(net_name, N_BUS, boed_ref_k4, boed_ref_k7):
    print(f"\n{'='*60}")
    print(f"MVG vs BOED vs Random vs GreedyLoop: {net_name}")

    if net_name == '33bus':
        import torch
        ckpt = torch.load(f"{SAVE_DIR}\\nre_ipc_loadaware.pt",
                          map_location='cpu', weights_only=False)
        V_library   = ckpt['V_library']
        lf_grid     = ckpt['lf_grid']
        N_TOPOS     = ckpt['N_TOPOS']
    else:
        dat = np.load(f"{SAVE_DIR}\\v_library_69bus.npz")
        V_library   = dat['V_library']
        lf_grid     = dat['lf_grid']
        N_TOPOS     = V_library.shape[0]

    CANDIDATES = list(range(1, N_BUS))
    N_LF_C = 11
    lf_idx_c = np.round(np.linspace(0, N_LF-1, N_LF_C)).astype(int)
    V_lib_c  = V_library[:, lf_idx_c, :]

    N_TEST = 500 if net_name == '33bus' else 200

    def ais_posterior(obs_nodes, obs_vals):
        if not obs_nodes: return np.ones(N_TOPOS)/N_TOPOS
        diff = (V_lib_c[:, :, obs_nodes] - np.array(obs_vals)) / SIGMA
        ll = -0.5*np.sum(diff**2, axis=2); ll -= ll.max()
        w = np.exp(ll).sum(axis=1); return w/w.sum()

    def entropy(p): p=np.clip(p,1e-10,1); return -np.sum(p*np.log(p))

    def mvg_select(obs_nodes, candidates, p_cur):
        """选后验加权电压方差最大的节点"""
        # E[V_j] = sum_τ p(τ) * mean_lf(V_τ[j])
        V_mean_lf = V_library[:, :, candidates].mean(axis=1)  # (N_TOPOS, N_CAND)
        E_V  = (p_cur[:, None] * V_mean_lf).sum(axis=0)       # (N_CAND,)
        E_V2 = (p_cur[:, None] * V_mean_lf**2).sum(axis=0)    # (N_CAND,)
        var  = E_V2 - E_V**2                                    # (N_CAND,)
        return candidates[int(np.argmax(var))]

    import networkx as nx
    import pandapower as pp
    import copy

    def get_loops(ne, te):
        G = nx.Graph(); G.add_edges_from(ne)
        return [set(nx.shortest_path(G, t[0], t[1])) for t in te]

    if net_name == '33bus':
        net = pp.create_empty_network()
        for i in range(33): pp.create_bus(net, vn_kv=12.66)
        br=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(1,18),(18,19),(19,20),(20,21),(2,22),(22,23),(23,24),(5,25),(25,26),(26,27),(27,28),(28,29),(29,30),(30,31),(31,32)]
        ti=[(7,20),(8,14),(11,21),(17,32),(24,28)]
        LOOPS = get_loops(br, ti)
    else:
        br_raw=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),(18,19),(19,20),(20,21),(21,22),(22,23),(23,24),(24,25),(25,26),(2,27),(27,28),(28,29),(29,30),(30,31),(31,32),(32,33),(33,34),(34,35),(35,36),(36,37),(37,38),(38,39),(39,40),(40,41),(41,42),(42,43),(43,44),(44,45),(3,46),(46,47),(47,48),(48,49),(7,50),(50,51),(8,52),(52,53),(53,54),(54,55),(55,56),(56,57),(57,58),(58,59),(59,60),(60,61),(61,62),(62,63),(63,64),(10,65),(65,66),(11,67),(67,68)]
        ti_raw=[(10,42),(12,20),(14,45),(49,58),(26,64)]
        LOOPS = get_loops(br_raw, ti_raw)

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

    # 生成测试场景
    rng_main = np.random.RandomState(2024)
    test_cases = []
    for _ in range(N_TEST):
        ti     = rng_main.randint(0, N_TOPOS)
        lf_idx = rng_main.randint(0, N_LF)
        full_v = V_library[ti, lf_idx, :] + rng_main.normal(0, SIGMA, N_BUS)
        test_cases.append((ti, lf_idx, full_v))

    strategies = ['MVG', 'Random', 'GreedyLoop']
    acc = {s: np.zeros(K_MAX) for s in strategies}
    H   = {s: np.zeros(K_MAX) for s in strategies}

    t0 = time.time()
    for ci, (true_ti, lf_idx, full_v) in enumerate(test_cases):
        if ci % (N_TEST//5) == 0:
            print(f"  [{net_name}] {ci}/{N_TEST}  {time.time()-t0:.0f}s", flush=True)
        rng_rand = np.random.RandomState(ci*17+3)

        # MVG（后验自适应方差贪婪）
        sel, sv, remain = [], [], list(CANDIDATES)
        for k in range(K_MAX):
            p = ais_posterior(sel, sv)
            node = mvg_select(sel, remain, p)
            sel.append(node); sv.append(full_v[node]); remain.remove(node)
            p2 = ais_posterior(sel, sv)
            acc['MVG'][k] += int(np.argmax(p2)==true_ti)
            H['MVG'][k]   += entropy(p2)

        # Random
        perm = list(CANDIDATES); rng_rand.shuffle(perm)
        sel, sv = [], []
        for k in range(K_MAX):
            sel.append(perm[k]); sv.append(full_v[perm[k]])
            p = ais_posterior(sel, sv)
            acc['Random'][k] += int(np.argmax(p)==true_ti)
            H['Random'][k]   += entropy(p)

        # GreedyLoop
        remain = list(CANDIDATES); sel, sv = [], []
        for k in range(K_MAX):
            node = greedy_loop_select(sel, remain)
            sel.append(node); sv.append(full_v[node]); remain.remove(node)
            p = ais_posterior(sel, sv)
            acc['GreedyLoop'][k] += int(np.argmax(p)==true_ti)
            H['GreedyLoop'][k]   += entropy(p)

    for s in strategies:
        acc[s] /= N_TEST; H[s] /= N_TEST

    print(f"\n  {net_name} 传感器选址对比（top-1准确率）：")
    print(f"  {'K':>3}  {'MVG':>7}  {'Random':>8}  {'GreedyLoop':>11}  {'BOED(ref)':>10}")
    boed_ref = boed_ref_k4  # full array
    for k in range(K_MAX):
        boed_v = boed_ref[k] if k < len(boed_ref) else float('nan')
        print(f"  {k+1:>3}  {acc['MVG'][k]:>7.3f}  {acc['Random'][k]:>8.3f}  "
              f"{acc['GreedyLoop'][k]:>11.3f}  {boed_v:>10.3f}")

    k4_mvg, k7_mvg = acc['MVG'][3], acc['MVG'][6]
    k4_boed = boed_ref_k4[3] if len(boed_ref_k4)>3 else 0
    k7_boed = boed_ref_k7[6] if len(boed_ref_k7)>6 else 0
    print(f"\n  K=4: MVG={k4_mvg:.3f}  BOED={k4_boed:.3f}  delta_BOED-MVG={k4_boed-k4_mvg:+.3f}")
    print(f"  K=7: MVG={k7_mvg:.3f}  BOED={k7_boed:.3f}  delta_BOED-MVG={k7_boed-k7_mvg:+.3f}")

    return acc

# BOED完整曲线
boed_33 = [0.150,0.278,0.450,0.574,0.666,0.718,0.754,0.782,0.790,0.796,0.804,0.812]
boed_69 = [0.095,0.170,0.315,0.395,0.500,0.595,0.670,0.705,0.755,0.780,0.790,0.785]

all_acc = {}
for net_name, N_BUS, boed_ref in [('33bus', 33, boed_33), ('69bus', 69, boed_69)]:
    all_acc[net_name] = run_mvg(net_name, N_BUS, boed_ref, boed_ref)

print(f"\n{'='*60}")
print("MVG BASELINE SUMMARY")
for net_name in ['33bus', '69bus']:
    a = all_acc[net_name]
    br = boed_33 if net_name=='33bus' else boed_69
    print(f"\n  {net_name}:")
    print(f"    K=4: MVG={a['MVG'][3]:.3f}  Random={a['Random'][3]:.3f}  "
          f"GreedyLoop={a['GreedyLoop'][3]:.3f}  BOED={br[3]:.3f}")
    print(f"    K=7: MVG={a['MVG'][6]:.3f}  Random={a['Random'][6]:.3f}  "
          f"GreedyLoop={a['GreedyLoop'][6]:.3f}  BOED={br[6]:.3f}")
    g4 = a['MVG'][3] > a['Random'][3]
    g7 = a['MVG'][6] > a['Random'][6]
    print(f"    MVG>Random: K=4={'OK' if g4 else 'FAIL'}  K=7={'OK' if g7 else 'FAIL'}")
    print(f"    BOED>MVG:   K=4={'OK' if br[3]>a['MVG'][3] else 'FAIL'}  K=7={'OK' if br[6]>a['MVG'][6] else 'FAIL'}")

with open(f"{SAVE_DIR}\\step5_mvg_result.txt", 'w', encoding='utf-8') as f:
    for net_name in ['33bus', '69bus']:
        a = all_acc[net_name]
        br = boed_33 if net_name=='33bus' else boed_69
        f.write(f"{net_name}: MVG K4={a['MVG'][3]:.3f} K7={a['MVG'][6]:.3f}  "
                f"Random K4={a['Random'][3]:.3f} K7={a['Random'][6]:.3f}  "
                f"BOED K4={br[3]:.3f} K7={br[6]:.3f}\n")
print(f"\nSaved: {SAVE_DIR}\\step5_mvg_result.txt")
