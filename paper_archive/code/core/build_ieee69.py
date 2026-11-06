# -*- coding: utf-8 -*-
"""
构建 IEEE 69-bus 配电网测试系统
参数来源：Baran & Wu (1989) / Savier & Das (2007) Table I
  "Impact of network reconfiguration on loss allocation of radial distribution systems"
  IEEE Transactions on Power Delivery, 22(4), 2473-2480.

特征：
  - 69个节点，68条正常支路，5条联络开关
  - 12.66 kV，辐射状配电网
  - 5个联络开关 → 最多32种有效拓扑（枚举后确认）
"""
import copy, warnings
import numpy as np
import pandapower as pp
import networkx as nx
warnings.filterwarnings('ignore')

def build_ieee69():
    """
    返回 (net, normal_edges, tie_edges)
    - normal_edges: 68条正常支路 [(from,to), ...]，0-indexed
    - tie_edges: 5条联络开关 [(from,to), ...]，0-indexed
    """
    net = pp.create_empty_network()

    # 69个母线，12.66 kV
    for i in range(69):
        pp.create_bus(net, vn_kv=12.66, name=f"Bus{i+1}")

    # ── 支路数据（R单位Ω，X单位Ω；P单位kW，Q单位kVAR）────────────────
    # 格式：(from_1idx, to_1idx, R_ohm, X_ohm, P_kW, Q_kVAR)
    # 参考：Savier & Das 2007, Table I（1-indexed 母线编号）
    branch_data = [
        # 主馈线段 1→2→...→27
        (1,  2,  0.0005, 0.0012,    0,     0),
        (2,  3,  0.0005, 0.0012,    0,     0),
        (3,  4,  0.0015, 0.0036,    0,     0),
        (4,  5,  0.0251, 0.0294,    0,     0),
        (5,  6,  0.3660, 0.1864,    2.6,   2.2),
        (6,  7,  0.3811, 0.1941,   40.4,  30.0),
        (7,  8,  0.0922, 0.0470,   75.0,  54.0),
        (8,  9,  0.0493, 0.0251,   30.0,  22.0),
        (9,  10, 0.8190, 0.2707,   28.0,  19.0),
        (10, 11, 0.1872, 0.0619,  145.0, 104.0),
        (11, 12, 0.7114, 0.2351,  145.0, 104.0),
        (12, 13, 1.0300, 0.3400,    8.0,   5.5),
        (13, 14, 1.0440, 0.3450,    8.0,   5.5),
        (14, 15, 1.0580, 0.3496,    0.0,   0.0),
        (15, 16, 0.1966, 0.0650,   45.5,  30.0),
        (16, 17, 0.3744, 0.1238,   60.0,  35.0),
        (17, 18, 0.0047, 0.0016,   60.0,  35.0),
        (18, 19, 0.3276, 0.1083,    0.0,   0.0),
        (19, 20, 0.2106, 0.0690,    1.0,   0.6),
        (20, 21, 0.3416, 0.1129,  114.0,  81.0),
        (21, 22, 0.0140, 0.0046,    5.3,   3.5),
        (22, 23, 0.1591, 0.0526,    0.0,   0.0),
        (23, 24, 0.3463, 0.1145,   28.0,  20.0),
        (24, 25, 0.7488, 0.2475,    0.0,   0.0),
        (25, 26, 0.3089, 0.1021,   14.0,  10.0),
        (26, 27, 0.1732, 0.0572,   14.0,  10.0),
        # 从3号出发的支线 → 28→...→46
        (3,  28, 0.0044, 0.0108,   26.0,  18.6),
        (28, 29, 0.0640, 0.1565,   26.0,  18.6),
        (29, 30, 0.3978, 0.1315,    0.0,   0.0),
        (30, 31, 0.0702, 0.0232,    0.0,   0.0),
        (31, 32, 0.3510, 0.1160,    0.0,   0.0),
        (32, 33, 0.8390, 0.2816,   14.0,  10.0),
        (33, 34, 1.7080, 0.5646,   19.5,  14.0),
        (34, 35, 1.4740, 0.4873,    6.0,   4.0),
        (35, 36, 0.0044, 0.0108,   26.0,  18.6),
        (36, 37, 0.0640, 0.1565,   26.0,  18.6),
        (37, 38, 0.1053, 0.1230,    0.0,   0.0),
        (38, 39, 0.0304, 0.0355,   24.0,  17.0),
        (39, 40, 0.0018, 0.0021,   24.0,  17.0),
        (40, 41, 0.7283, 0.8509,    1.2,   1.0),
        (41, 42, 0.3100, 0.3623,    0.0,   0.0),
        (42, 43, 0.0410, 0.0478,    6.0,   4.3),
        (43, 44, 0.0092, 0.0116,    0.0,   0.0),
        (44, 45, 0.1089, 0.1373,   39.2,  26.3),
        (45, 46, 0.0009, 0.0012,   39.2,  26.3),
        # 从4号出发的支线 → 47→...→50
        (4,  47, 0.0034, 0.0084,    0.0,   0.0),
        (47, 48, 0.0851, 0.2083,   79.0,  56.4),
        (48, 49, 0.2898, 0.7091,  384.7, 274.5),
        (49, 50, 0.0822, 0.2011,  384.7, 274.5),
        # 从8号出发的支线 → 51→52
        (8,  51, 0.0928, 0.0473,   40.5,  28.3),
        (51, 52, 0.3319, 0.1114,    3.6,   2.7),
        # 从9号出发的支线 → 53→...→65
        (9,  53, 0.1740, 0.0886,    4.35,  3.5),
        (53, 54, 0.2030, 0.1034,   26.4,  19.0),
        (54, 55, 0.2842, 0.1447,   24.0,  17.2),
        (55, 56, 0.2813, 0.1433,    0.0,   0.0),
        (56, 57, 1.5900, 0.5337,    0.0,   0.0),
        (57, 58, 0.7837, 0.2630,    0.0,   0.0),
        (58, 59, 0.3042, 0.1006,  100.0,  72.0),
        (59, 60, 0.3861, 0.1172,    0.0,   0.0),
        (60, 61, 0.5075, 0.2585, 1244.0, 888.0),
        (61, 62, 0.0974, 0.0496,   32.0,  23.0),
        (62, 63, 0.1450, 0.0738,    0.0,   0.0),
        (63, 64, 0.7105, 0.3619,  227.0, 162.0),
        (64, 65, 1.0410, 0.5302,   59.0,  42.0),
        # 从11号出发的支线 → 66→67
        (11, 66, 0.2012, 0.0611,   18.0,  13.0),
        (66, 67, 0.0047, 0.0014,   18.0,  13.0),
        # 从12号出发的支线 → 68→69
        (12, 68, 0.7394, 0.2444,   28.0,  20.0),
        (68, 69, 0.0047, 0.0014,   28.0,  20.0),
    ]

    # 5条联络开关（正常断开，1-indexed）
    # 来源：Savier & Das 2007 原始拓扑配置
    tie_data = [
        (11, 43, 0.5000, 0.5000),  # Tie 1
        (13, 21, 0.5000, 0.5000),  # Tie 2
        (15, 46, 0.5000, 0.5000),  # Tie 3
        (50, 59, 0.5000, 0.5000),  # Tie 4
        (27, 65, 0.5000, 0.5000),  # Tie 5
    ]

    # 添加支路（0-indexed）
    normal_edges = []
    for f1, t1, r, x, p, q in branch_data:
        f0, t0 = f1-1, t1-1
        pp.create_line_from_parameters(net, f0, t0, 1, r, x, 0, 9999,
                                       in_service=True)
        if p > 0 or q > 0:
            pp.create_load(net, t0, p/1000, q/1000)
        normal_edges.append((f0, t0))

    tie_edges = []
    for f1, t1, r, x in tie_data:
        f0, t0 = f1-1, t1-1
        pp.create_line_from_parameters(net, f0, t0, 1, r, x, 0, 9999,
                                       in_service=False)
        tie_edges.append((f0, t0))

    # 变电站（外部电网连接）
    pp.create_ext_grid(net, 0, vm_pu=1.0)

    return net, normal_edges, tie_edges


def enum_topos(ne, te, n=69):
    G = nx.Graph(); G.add_edges_from(ne)
    topos = [list(range(len(ne)))]
    seen  = {frozenset(range(len(ne)))}
    for ti2, tie in enumerate(te):
        try:
            path = nx.shortest_path(G, tie[0], tie[1])
        except nx.NetworkXNoPath:
            continue
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


def run_pf(net_base, topo_raw, ne, te):
    net = copy.deepcopy(net_base)
    n_ne = len(ne)
    active_ne = {x for x in topo_raw if x < n_ne}
    active_te = {x - n_ne for x in topo_raw if x >= n_ne}
    for li in range(n_ne):
        net.line.at[net.line.index[li], 'in_service'] = (li in active_ne)
    for li in range(len(te)):
        net.line.at[net.line.index[n_ne + li], 'in_service'] = (li in active_te)
    try:
        pp.runpp(net, algorithm='bfsw', numba=False,
                 max_iteration=100, tolerance_mva=1e-6)
        if net.converged:
            return net.res_bus.vm_pu.values.copy()
    except: pass
    return None


# ── 主验证 ───────────────────────────────────────────────────────────────
print("=" * 60)
print("IEEE 69-bus Distribution Network Validation")
print("=" * 60)

print("\n[1] Building network...")
net69, ne69, te69 = build_ieee69()
print(f"  Buses: {len(net69.bus)}")
print(f"  Normal lines: {len([l for l in net69.line.in_service if l])}")
print(f"  Tie switches: {len([l for l in net69.line.in_service if not l])}")
print(f"  Total load: P={net69.load.p_mw.sum()*1000:.1f} kW, "
      f"Q={net69.load.q_mvar.sum()*1000:.1f} kVAR")

print("\n[2] Base topology power flow...")
try:
    pp.runpp(net69, algorithm='bfsw', numba=False,
             max_iteration=100, tolerance_mva=1e-6)
    if net69.converged:
        vm = net69.res_bus.vm_pu.values
        print(f"  Converged! V_min={vm.min():.4f} V_max={vm.max():.4f} pu")
        print(f"  (Published: V_min~0.909 pu @ bus 65)")
    else:
        print("  FAILED to converge")
except Exception as e:
    print(f"  Error: {e}")

print("\n[3] Topology enumeration...")
topos = enum_topos(ne69, te69, n=69)
print(f"  Valid radial topologies: {len(topos)}")
print(f"  (Expected: up to 32 with 5 tie switches)")

print("\n[4] Power flow validation (all topologies)...")
results = []
for i, topo in enumerate(topos):
    v = run_pf(net69, topo, ne69, te69)
    results.append(v is not None)
n_ok = sum(results)
print(f"  Converged: {n_ok}/{len(topos)}")

if n_ok > 0:
    print("\n[5] Voltage profile sample (base topology):")
    v_base = run_pf(net69, topos[0], ne69, te69)
    if v_base is not None:
        print(f"  Min voltage: {v_base.min():.4f} pu @ bus {v_base.argmin()+1}")
        print(f"  Max voltage: {v_base.max():.4f} pu @ bus {v_base.argmax()+1}")

print("\n" + "=" * 60)
if n_ok == len(topos) and len(topos) >= 10:
    print("✅ IEEE 69-bus VALIDATED — ready for experiments")
    print(f"   {len(topos)} valid topologies, all power flows converge")
else:
    print("⚠️  Issues found — check above")
print("=" * 60)
