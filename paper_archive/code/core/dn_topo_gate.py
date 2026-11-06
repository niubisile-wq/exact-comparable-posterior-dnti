# -*- coding: utf-8 -*-
"""
关门实验：IEEE 33-bus 配电网拓扑辨识后验分布多峰性验证
"""
import copy, warnings
import numpy as np
import pandapower as pp
import networkx as nx
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product
warnings.filterwarnings('ignore')
np.random.seed(42)

# ── 1. 构建 IEEE 33-bus（含5条联络开关线路，初始断开）───────────────
def build_ieee33():
    net = pp.create_empty_network()
    for i in range(33):
        pp.create_bus(net, vn_kv=12.66, name=f"Bus{i}")

    # 32条正常支路 (from, to, R_ohm, X_ohm) —— Baran&Wu 1989
    branches = [
        (0,1,0.0922,0.0470),(1,2,0.4930,0.2511),(2,3,0.3660,0.1864),
        (3,4,0.3811,0.1941),(4,5,0.8190,0.7070),(5,6,0.1872,0.6188),
        (6,7,0.7114,0.2351),(7,8,1.0300,0.7400),(8,9,1.0440,0.7400),
        (9,10,0.1966,0.0650),(10,11,0.3744,0.1238),(11,12,1.4680,1.1550),
        (12,13,0.5416,0.7129),(13,14,0.5910,0.5260),(14,15,0.7463,0.5450),
        (15,16,1.2890,1.7210),(16,17,0.7320,0.5740),(1,18,0.1640,0.1565),
        (18,19,1.5042,1.3554),(19,20,0.4095,0.4784),(20,21,0.7089,0.9373),
        (2,22,0.4512,0.3083),(22,23,0.8980,0.7091),(23,24,0.8960,0.7011),
        (5,25,0.2030,0.1034),(25,26,0.2842,0.1447),(26,27,1.0590,0.9337),
        (27,28,0.8042,0.7006),(28,29,0.5075,0.2585),(29,30,0.9744,0.9630),
        (30,31,0.3105,0.3619),(31,32,0.3410,0.5302),
    ]
    # 5条联络开关 (from, to, R_ohm, X_ohm) —— 标准IEEE 33-bus tie-line数据
    tie_data = [
        (7, 20, 0.0890, 0.0890),
        (8, 14, 0.0590, 0.0590),
        (11,21, 0.0890, 0.0890),
        (17,32, 0.0380, 0.0850),
        (24,28, 0.0560, 0.0650),
    ]

    # 添加32条正常支路（默认in_service=True）
    for (f,t,r,x) in branches:
        pp.create_line_from_parameters(net, from_bus=f, to_bus=t, length_km=1,
            r_ohm_per_km=r, x_ohm_per_km=x, c_nf_per_km=0, max_i_ka=1,
            in_service=True)

    # 添加5条联络开关（默认in_service=False，断开状态）
    for (f,t,r,x) in tie_data:
        pp.create_line_from_parameters(net, from_bus=f, to_bus=t, length_km=1,
            r_ohm_per_km=r, x_ohm_per_km=x, c_nf_per_km=0, max_i_ka=1,
            in_service=False)

    loads = [
        (1,100,60),(2,90,40),(3,120,80),(4,60,30),(5,60,20),(6,200,100),
        (7,200,100),(8,60,20),(9,60,20),(10,45,30),(11,60,35),(12,60,35),
        (13,120,80),(14,60,10),(15,60,20),(16,60,20),(17,90,40),(18,90,40),
        (19,90,40),(20,90,40),(21,90,40),(22,90,50),(23,420,200),(24,420,200),
        (25,60,25),(26,60,25),(27,60,20),(28,120,70),(29,200,600),
        (30,150,70),(31,210,100),(32,60,40),
    ]
    for (bus,p,q) in loads:
        pp.create_load(net, bus=bus, p_mw=p/1000, q_mvar=q/1000)
    pp.create_ext_grid(net, bus=0, vm_pu=1.0)

    # 记录支路信息：索引0-31=正常支路，32-36=联络开关
    normal_edges = [(int(f),int(t)) for (f,t,r,x) in branches]
    tie_edges    = [(int(f),int(t)) for (f,t,r,x) in tie_data]
    return net, normal_edges, tie_edges

# ── 2. 枚举有效辐射状拓扑 ─────────────────────────────────────────────
def enumerate_topologies(normal_edges, tie_edges, n_buses=33):
    """
    每个联络开关闭合 → 产生环路 → 断开环路中某一正常支路 → 新有效拓扑
    返回：每个拓扑用 (normal_closed, tie_closed) 表示
    normal_closed: list of normal edge indices (0-31) 活跃
    tie_closed:    list of tie edge indices (0-4) 活跃
    """
    G_base = nx.Graph()
    G_base.add_edges_from(normal_edges)

    valid_topos = []
    # 基础拓扑：所有正常支路闭合，所有联络开关断开
    valid_topos.append((list(range(32)), []))

    seen = {frozenset(range(32))}

    for ti, tie in enumerate(tie_edges):
        try:
            path = nx.shortest_path(G_base, tie[0], tie[1])
        except nx.NetworkXNoPath:
            continue
        path_edges_set = [frozenset([path[i], path[i+1]])
                          for i in range(len(path)-1)]
        # 找到这条path对应哪些normal_edges的索引
        path_ne_idx = []
        for pe in path_edges_set:
            for ni, ne in enumerate(normal_edges):
                if frozenset(ne) == pe:
                    path_ne_idx.append(ni); break

        for open_ni in path_ne_idx:
            new_normal = [ni for ni in range(32) if ni != open_ni]
            key = frozenset(new_normal)
            if key in seen:
                continue
            # 验证：n_buses节点，新normal+这个tie共32条，是否是树
            edge_list = [normal_edges[ni] for ni in new_normal] + [tie]
            G = nx.Graph(); G.add_nodes_from(range(n_buses))
            G.add_edges_from(edge_list)
            if nx.is_connected(G) and nx.is_tree(G):
                seen.add(key)
                valid_topos.append((new_normal, [ti]))

    return valid_topos

# ── 3. 潮流计算 ───────────────────────────────────────────────────────
def run_pf(net_base, normal_closed_idx, tie_closed_idx):
    """
    net_base 中：line索引0-31=正常支路，32-36=联络开关
    根据 normal_closed_idx 和 tie_closed_idx 设置 in_service
    """
    net = copy.deepcopy(net_base)
    normal_set = set(normal_closed_idx)
    tie_set    = set(tie_closed_idx)
    for li in range(37):
        if li < 32:
            active = li in normal_set
        else:
            active = (li - 32) in tie_set
        net.line.at[net.line.index[li], 'in_service'] = active
    try:
        pp.runpp(net, algorithm='bfsw', numba=False,
                 max_iteration=50, tolerance_mva=1e-8)
        if net.converged:
            return net.res_bus.vm_pu.values.copy()
    except Exception:
        pass
    return None

# ── 4. 计算后验 ───────────────────────────────────────────────────────
def compute_posterior(voltages, obs_nodes, obs_v, sigma):
    log_lls, valid_ids = [], []
    for i, V in enumerate(voltages):
        if V is None: continue
        diff = (V[obs_nodes] - obs_v) / sigma
        log_lls.append(-0.5 * float(np.sum(diff**2)))
        valid_ids.append(i)
    if not log_lls:
        return [], np.array([])
    log_lls = np.array(log_lls)
    log_lls -= log_lls.max()
    lls = np.exp(log_lls)
    s = lls.sum()
    if s == 0 or np.isnan(s):
        return valid_ids, np.full(len(valid_ids), np.nan)
    return valid_ids, lls / s

# ── 5. 主实验 ─────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("IEEE 33-bus  Topology Posterior Multi-Modality Gate Test")
    print("=" * 62)

    net, normal_edges, tie_edges = build_ieee33()
    print(f"Network: {len(net.bus)} buses, {len(net.line)} lines "
          f"(32 normal + 5 tie)")

    print("\n[1] Enumerating valid radial topologies...")
    topos = enumerate_topologies(normal_edges, tie_edges)
    print(f"    Found {len(topos)} valid topologies")

    print("\n[2] Running power flow for each topology...")
    voltages = []
    for (norm_idx, tie_idx) in topos:
        V = run_pf(net, norm_idx, tie_idx)
        voltages.append(V)
    n_ok = sum(1 for V in voltages if V is not None)
    print(f"    Converged: {n_ok}/{len(topos)}")

    # 真实拓扑 = 基础拓扑（index=0）
    true_idx = 0
    V_true = voltages[true_idx]
    print(f"\n[3] True topology  index={true_idx}  "
          f"V range: {V_true.min():.4f}~{V_true.max():.4f} pu")

    # 不同拓扑间最大电压偏差
    v_max_diffs = []
    for V in voltages:
        if V is not None:
            v_max_diffs.append(float(np.max(np.abs(V - V_true))))
    v_max_diffs = np.array(v_max_diffs)
    nonzero = v_max_diffs[v_max_diffs > 1e-6]
    print(f"    Inter-topology max-voltage diff (excl. self): "
          f"min={nonzero.min():.5f}  median={np.median(nonzero):.5f}  "
          f"max={nonzero.max():.5f} pu")

    # 选sigma：让"中等偏差"的拓扑有明显但非无穷大的对数似然差
    sigma_ref = float(np.percentile(nonzero, 25)) / 2
    sigma = max(0.001, sigma_ref)
    print(f"    Chosen sigma = {sigma:.5f} pu  "
          f"(measurement noise ~ {sigma*100:.3f}%)")

    # ── 实验循环 ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("IEEE 33-bus: Topology Posterior vs Measurement Count",
                 fontsize=13)
    results = []
    K_list  = [3, 5, 8, 12, 18, 25]

    for ax_idx, K in enumerate(K_list):
        ax = axes[ax_idx // 3][ax_idx % 3]
        rng = np.random.RandomState(0 + K)
        obs_nodes = np.sort(rng.choice(range(1, 33), K, replace=False))
        obs_v = V_true[obs_nodes] + rng.normal(0, sigma * 0.3, K)

        valid_ids, post = compute_posterior(voltages, obs_nodes, obs_v, sigma)

        if len(post) == 0 or np.any(np.isnan(post)):
            print(f"  K={K}: NaN posterior"); continue

        sorted_p = np.sort(post)[::-1]
        max_p    = float(sorted_p[0])
        second_p = float(sorted_p[1]) if len(sorted_p) > 1 else 0.0
        n_sig    = int(np.sum(post > 0.05))
        entropy  = float(-np.sum(post * np.log(post + 1e-15)))
        # 真实拓扑排名
        try:
            tp = valid_ids.index(true_idx)
            true_rank = int(np.where(np.argsort(post)[::-1] == tp)[0][0]) + 1
        except Exception:
            true_rank = -1

        is_mm = second_p > 0.10   # 第二高>10% = 多峰
        results.append(dict(K=K, max_p=max_p, second_p=second_p,
                            n_sig=n_sig, entropy=entropy,
                            true_rank=true_rank, multimodal=is_mm))

        # 绘图
        n_show = min(20, len(sorted_p))
        ax.bar(range(n_show), sorted_p[:n_show],
               color=['tomato']+['steelblue']*(n_show-1))
        label = "MULTI-MODAL" if is_mm else \
                ("weak" if n_sig > 1 else "single-peak")
        color = 'red' if is_mm else ('orange' if n_sig > 1 else 'green')
        ax.text(0.97, 0.95, label, transform=ax.transAxes,
                ha='right', va='top', color=color,
                fontsize=10, fontweight='bold')
        ax.set_title(
            f"K={K}  top={max_p:.3f}  2nd={second_p:.3f}  "
            f"n_sig(>5%)={n_sig}", fontsize=9)
        ax.set_xlabel("Topology rank"); ax.set_ylabel("Posterior prob.")
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    out = r"<LOCAL_DESKTOP>\dn_topo_posterior.png"
    plt.savefig(out, dpi=150)
    print(f"\n[4] Figure saved: {out}")

    # ── 汇总 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"{'K':>4}  {'max_p':>7}  {'2nd_p':>7}  "
          f"{'n_sig':>6}  {'entropy':>8}  {'true_rank':>10}  verdict")
    print("-" * 62)
    for r in results:
        v = ("MULTI-MODAL" if r['multimodal'] else
             ("weak-multi" if r['n_sig'] > 1 else "single-peak"))
        print(f"{r['K']:>4}  {r['max_p']:>7.3f}  {r['second_p']:>7.3f}  "
              f"{r['n_sig']:>6d}  {r['entropy']:>8.3f}  "
              f"{r['true_rank']:>10d}  {v}")

    n_mm = sum(1 for r in results if r['multimodal'])
    n_wk = sum(1 for r in results if r['n_sig'] > 1)
    print("\n" + "=" * 62)
    if n_mm >= 2:
        print("[PASS]  Multi-modal posterior CONFIRMED.")
        print("        IP1 core claim is VALID. All 4 IPs can proceed.")
    elif n_wk >= 2:
        print("[PARTIAL]  Weak multi-modality detected at low K.")
        print("           IP1 holds but frame as 'limited-measurement' regime.")
    else:
        print("[FAIL]  No multi-modal posterior. IP1 needs reframing.")

if __name__ == "__main__":
    main()
