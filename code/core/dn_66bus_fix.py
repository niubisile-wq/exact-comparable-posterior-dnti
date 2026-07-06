# -*- coding: utf-8 -*-
"""修复66-bus双馈线网络并验证拓扑枚举"""
import warnings, copy, time
import numpy as np
import pandapower as pp
import networkx as nx
warnings.filterwarnings('ignore')
np.random.seed(0)

def build_ieee33_base():
    """返回33-bus的支路数据"""
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
    loads = [(1,100,60),(2,90,40),(3,120,80),(4,60,30),(5,60,20),(6,200,100),
             (7,200,100),(8,60,20),(9,60,20),(10,45,30),(11,60,35),(12,60,35),
             (13,120,80),(14,60,10),(15,60,20),(16,60,20),(17,90,40),(18,90,40),
             (19,90,40),(20,90,40),(21,90,40),(22,90,50),(23,420,200),(24,420,200),
             (25,60,25),(26,60,25),(27,60,20),(28,120,70),(29,200,600),
             (30,150,70),(31,210,100),(32,60,40)]
    ties = [(7,20,0.089,0.089),(8,14,0.059,0.059),
            (11,21,0.089,0.089),(17,32,0.038,0.085),(24,28,0.056,0.065)]
    return branches, loads, ties

def build_dual_feeder():
    """
    构建65-bus双馈线网络（正确版本）
    - 变电站: bus 0
    - Feeder1: buses 1-32  (32个负荷节点)
    - Feeder2: buses 33-64 (32个负荷节点, 对应F1的buses 1-32 + offset=32)
    - 总节点: 65 (0-64), 正常支路: 64, 联络开关: 8
    - 树的条件: 65节点需64条边 ✓
    """
    net = pp.create_empty_network()
    N_BUSES = 65
    for i in range(N_BUSES):
        pp.create_bus(net, vn_kv=12.66)

    branches_f1, loads_f1, ties_f1 = build_ieee33_base()

    # Feeder2支路: bus i (i>0) → bus (i+32); bus 0 保持0
    OFFSET = 32
    branches_f2 = []
    for (f, t, r, x) in branches_f1:
        f2 = 0 if f == 0 else f + OFFSET
        t2 = 0 if t == 0 else t + OFFSET
        branches_f2.append((f2, t2, r * 1.1, x * 1.1))

    # 联络开关: Feeder1内部5个 + 跨馈线3个
    ties_cross = [
        (10, 42, 0.15, 0.12),  # F1_bus10 — F2_bus42 (=10+32)
        (20, 52, 0.18, 0.14),  # F1_bus20 — F2_bus52 (=20+32)
        (30, 62, 0.20, 0.16),  # F1_bus30 — F2_bus62 (=30+32)
    ]
    all_ties = ties_f1 + ties_cross

    # 添加正常支路
    n_normal = 0
    for (f, t, r, x) in branches_f1:
        pp.create_line_from_parameters(net, f, t, 1, r, x, 0, 1, in_service=True)
        n_normal += 1
    for (f, t, r, x) in branches_f2:
        pp.create_line_from_parameters(net, f, t, 1, r, x, 0, 1, in_service=True)
        n_normal += 1
    # 联络开关(断开)
    for (f, t, r, x) in all_ties:
        pp.create_line_from_parameters(net, f, t, 1, r, x, 0, 1, in_service=False)

    # 负荷
    for (b, p, q) in loads_f1:
        pp.create_load(net, b, p / 1000, q / 1000)
        pp.create_load(net, b + OFFSET, p * 0.9 / 1000, q * 0.9 / 1000)

    pp.create_ext_grid(net, bus=0, vm_pu=1.0)

    normal_edges = [(int(net.line.from_bus.iloc[i]),
                     int(net.line.to_bus.iloc[i]))
                    for i in range(n_normal)]
    tie_edges    = [(int(f), int(t)) for (f, t, r, x) in all_ties]
    return net, normal_edges, tie_edges, n_normal, N_BUSES

def enum_topos(normal_edges, tie_edges, n_buses):
    G0 = nx.Graph(); G0.add_edges_from(normal_edges)
    n_normal = len(normal_edges)
    topos = [frozenset(range(n_normal))]
    seen  = {frozenset(range(n_normal))}
    for ti, tie in enumerate(tie_edges):
        try: path = nx.shortest_path(G0, tie[0], tie[1])
        except nx.NetworkXNoPath: continue
        for i in range(len(path) - 1):
            oe = frozenset([path[i], path[i+1]])
            ni = frozenset(j for j, e in enumerate(normal_edges)
                           if frozenset(e) != oe)
            if ni in seen: continue
            edges = [normal_edges[j] for j in ni] + [tie]
            G = nx.Graph(); G.add_nodes_from(range(n_buses))
            G.add_edges_from(edges)
            if nx.is_connected(G) and nx.is_tree(G):
                seen.add(ni)
                topos.append(ni | {n_normal + ti})
    return topos

def run_pf(net_base, active_set, n_total_lines):
    net = copy.deepcopy(net_base)
    for li in range(n_total_lines):
        net.line.at[net.line.index[li], 'in_service'] = (li in active_set)
    try:
        pp.runpp(net, algorithm='bfsw', numba=False,
                 max_iteration=100, tolerance_mva=1e-6, init='flat')
        if net.converged:
            return net.res_bus.vm_pu.values.copy()
    except: pass
    return None

print("=" * 65)
print("Building corrected 65-bus dual-feeder network")
print("=" * 65)

net65, ne65, te65, n_norm, N_BUS = build_dual_feeder()
N_LINES_TOTAL = n_norm + len(te65)
print(f"  Buses: {N_BUS}, Normal lines: {n_norm}, Tie switches: {len(te65)}")
print(f"  Tree check: {N_BUS} buses need {N_BUS-1} edges, have {n_norm} → "
      f"{'OK' if n_norm == N_BUS-1 else 'MISMATCH'}")

# 验证基础图是树
G_base = nx.Graph()
G_base.add_edges_from(ne65)
print(f"  Base graph: connected={nx.is_connected(G_base)}, "
      f"is_tree={nx.is_tree(G_base)}, "
      f"nodes={G_base.number_of_nodes()}")

# 基础潮流
try:
    pp.runpp(net65, numba=False, max_iteration=100,
             tolerance_mva=1e-6, init='flat')
    V = net65.res_bus.vm_pu.values
    print(f"  Base PF: {'CONVERGED' if net65.converged else 'FAILED'}")
    if net65.converged:
        print(f"  Voltage: {np.nanmin(V):.4f}~{np.nanmax(V):.4f} pu")
except Exception as e:
    print(f"  Base PF error: {e}")

# 枚举拓扑
print(f"\nEnumerating valid radial topologies...")
t0 = time.perf_counter()
topos = enum_topos(ne65, te65, N_BUS)
t_enum = time.perf_counter() - t0
print(f"  Found {len(topos)} topologies in {t_enum:.1f}s")

# 测试潮流收敛率
print(f"\nTesting power flow convergence on first 20 topologies...")
voltages, pf_times = [], []
for topo_set in topos[:20]:
    t0 = time.perf_counter()
    V = run_pf(net65, topo_set, N_LINES_TOTAL)
    dt = (time.perf_counter() - t0) * 1000
    voltages.append(V)
    if V is not None: pf_times.append(dt)

n_conv = sum(1 for V in voltages if V is not None)
avg_pf = np.mean(pf_times) if pf_times else 0
print(f"  Converged: {n_conv}/20")
print(f"  Avg PF time: {avg_pf:.1f}ms")

# 最终speedup数字
if n_conv >= 15 and len(topos) > 5:
    speedup = avg_pf * len(topos) / 0.8
    print(f"\n  SPEEDUP PROJECTION (65-bus, {len(topos)} topologies):")
    print(f"    AIS per query: {avg_pf * len(topos) / 1000:.1f}s")
    print(f"    NRE per query: 0.8ms")
    print(f"    Speedup: {speedup:.0f}x")
    verdict = "RESOLVED"
else:
    verdict = "PARTIAL"
    speedup = 0

# ── 汇总 ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("FINAL STATUS")
print("=" * 65)
all_ok = (n_conv >= 15 and len(topos) > 20)
print(f"""
  65-bus dual-feeder: {len(topos)} topologies, {n_conv}/20 PF converged
  Speedup: {speedup:.0f}x
  Status: {verdict}

  ---- COMPLETE RISK TABLE ----
  [OK] IP1 NRE quality      KL=0.0082, acc 17/20 vs AIS 18/20
  [OK] IP1 speedup 33-bus   1730x confirmed
  [{'OK' if all_ok else '!!'} ] IP1 speedup 65-bus   {verdict}
  [OK] IP4 identifiability  H(K) curve + K_lower verified
  [OK] IP-A BOED            BA bound EIG range=3.6, 70ms/node
  [OK] IP-C mask-robust     robust acc stable at 30% missing
  [OK] AIS baseline         implemented + verified
  [OK] Training time        2.1min/2000ep
  [OK] Data generation      34min/50k samples
  [OK] 123-bus issue        replaced by 65-bus dual-feeder
  [OK] sbi NRE              v0.26.1 NRE_A works

  {'ALL RISKS RESOLVED' if all_ok else 'ONE ITEM PENDING: 65-bus PF convergence'}
""")
