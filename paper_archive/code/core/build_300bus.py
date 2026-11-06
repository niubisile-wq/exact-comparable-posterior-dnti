# -*- coding: utf-8 -*-
"""
Build a synthetic 300-bus radial distribution feeder and voltage library.

Scope for Step 5:
  - synthetic mid-scale scalability evidence, not a real utility feeder claim
  - 150-250 valid radial topology candidates
  - pandapower power-flow voltage library for later IP4/IP1/IP-C experiments
"""
import copy
import os
import time
import warnings

import networkx as nx
import numpy as np
import pandapower as pp

warnings.filterwarnings("ignore")

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
N_BUS = 300
N_LF = 31
LF_MIN, LF_MAX = 0.80, 1.20
TARGET_MIN, TARGET_MAX = 150, 250
MAX_TOPOLOGIES = 220
SEED = 3007


def make_base_edges():
    edges = []
    # Main feeder.
    for i in range(0, 120):
        edges.append((i, i + 1))
    next_bus = 121
    branch_roots = [8, 16, 24, 33, 43, 54, 66, 79, 93, 108]
    branch_lengths = [16, 18, 14, 20, 15, 19, 17, 18, 16, 26]
    for root, length in zip(branch_roots, branch_lengths):
        prev = root
        for _ in range(length):
            edges.append((prev, next_bus))
            prev = next_bus
            next_bus += 1
    assert next_bus == N_BUS, next_bus
    return edges


def make_tie_edges():
    # Tie switches connecting distant points/laterals. They are normally open.
    return [
        (30, 145), (48, 170), (64, 190), (82, 210), (100, 235),
        (118, 260), (135, 165), (155, 185), (175, 205), (195, 225),
        (215, 245), (240, 285),
    ]


def make_network(base_edges, tie_edges):
    rng = np.random.RandomState(SEED)
    net = pp.create_empty_network()
    for _ in range(N_BUS):
        pp.create_bus(net, vn_kv=12.66)
    for f, t in base_edges:
        dist = abs(t - f)
        length = 1.0
        r = 0.010 + 0.0006 * (dist % 7) + rng.uniform(0.0, 0.002)
        x = 0.008 + 0.0005 * (dist % 5) + rng.uniform(0.0, 0.0015)
        pp.create_line_from_parameters(net, f, t, length, r, x, 0.0, 1.0, in_service=True)
    for f, t in tie_edges:
        pp.create_line_from_parameters(net, f, t, 1.0, 0.009, 0.007, 0.0, 1.0, in_service=False)
    # Smooth diversified loads; bus 0 is slack.
    for b in range(1, N_BUS):
        p_kw = 5.0 + 10.0 * rng.rand()
        # Larger laterals/end sections carry some heavier customers.
        if b % 17 == 0 or b > 240:
            p_kw += 8.0 * rng.rand()
        q_kw = p_kw * (0.35 + 0.20 * rng.rand())
        pp.create_load(net, b, p_mw=p_kw / 1000.0, q_mvar=q_kw / 1000.0)
    pp.create_ext_grid(net, 0, vm_pu=1.0)
    return net


def enumerate_topologies(base_edges, tie_edges):
    base_graph = nx.Graph()
    base_graph.add_nodes_from(range(N_BUS))
    base_graph.add_edges_from(base_edges)
    assert nx.is_tree(base_graph)
    edge_index = {frozenset(e): i for i, e in enumerate(base_edges)}
    topologies = [list(range(len(base_edges)))]
    seen = {frozenset(topologies[0])}
    for tie_i, tie in enumerate(tie_edges):
        path = nx.shortest_path(base_graph, tie[0], tie[1])
        # Keep every second edge on long paths to avoid too many near-duplicates.
        for j in range(0, len(path) - 1, 2):
            old = frozenset((path[j], path[j + 1]))
            if old not in edge_index:
                continue
            active = [idx for idx in range(len(base_edges)) if idx != edge_index[old]]
            topo = active + [len(base_edges) + tie_i]
            key = frozenset(topo)
            if key in seen:
                continue
            edges = [base_edges[idx] for idx in active] + [tie]
            g = nx.Graph()
            g.add_nodes_from(range(N_BUS))
            g.add_edges_from(edges)
            if nx.is_tree(g):
                seen.add(key)
                topologies.append(topo)
            if len(topologies) >= MAX_TOPOLOGIES:
                return topologies
    return topologies


def run_pf(net_base, topo, lf):
    net = copy.deepcopy(net_base)
    net.load["p_mw"] *= lf
    net.load["q_mvar"] *= lf
    active = set(topo)
    for li in range(len(net.line)):
        net.line.at[net.line.index[li], "in_service"] = li in active
    try:
        pp.runpp(net, algorithm="bfsw", numba=False, max_iteration=80, tolerance_mva=1e-8)
        if net.converged:
            return net.res_bus.vm_pu.values.astype(np.float32)
    except Exception:
        return None
    return None


def main():
    t0 = time.time()
    base_edges = make_base_edges()
    tie_edges = make_tie_edges()
    topologies = enumerate_topologies(base_edges, tie_edges)
    if not (TARGET_MIN <= len(topologies) <= TARGET_MAX):
        raise RuntimeError(f"Topology count {len(topologies)} outside target [{TARGET_MIN}, {TARGET_MAX}]")
    net = make_network(base_edges, tie_edges)
    lf_grid = np.linspace(LF_MIN, LF_MAX, N_LF).astype(np.float32)
    v_library = np.zeros((len(topologies), N_LF, N_BUS), dtype=np.float32)
    failed = 0
    for i, topo in enumerate(topologies):
        last = None
        for j, lf in enumerate(lf_grid):
            v = run_pf(net, topo, float(lf))
            if v is None:
                failed += 1
                v = last if last is not None else np.ones(N_BUS, dtype=np.float32)
            v_library[i, j] = v
            last = v
        if (i + 1) % 25 == 0:
            print(f"  topology {i + 1}/{len(topologies)} elapsed={time.time() - t0:.1f}s", flush=True)

    base_p = np.zeros(N_BUS, dtype=np.float32)
    for _, row in net.load.iterrows():
        base_p[int(row.bus)] += float(row.p_mw)
    base_p_norm = base_p / max(float(base_p.max()), 1e-8)

    out_path = os.path.join(SAVE_DIR, "v_library_300bus.npz")
    np.savez_compressed(
        out_path,
        V_library=v_library,
        lf_grid=lf_grid,
        base_P_norm=base_p_norm,
        base_edges=np.array(base_edges, dtype=np.int32),
        tie_edges=np.array(tie_edges, dtype=np.int32),
        topologies=np.array([np.array(t, dtype=np.int32) for t in topologies], dtype=object),
        n_bus=N_BUS,
        n_topologies=len(topologies),
        failed_pf=failed,
    )
    result = [
        "Step 5 synthetic 300-bus build result",
        f"n_bus={N_BUS}",
        f"n_base_edges={len(base_edges)}",
        f"n_tie_edges={len(tie_edges)}",
        f"n_topologies={len(topologies)}",
        f"n_lf={N_LF}",
        f"failed_power_flows={failed}",
        f"v_library_shape={v_library.shape}",
        f"saved={out_path}",
        "Boundary: synthetic mid-scale radial feeder for scalability stress evidence, not a real utility feeder.",
        f"elapsed_sec={time.time() - t0:.1f}",
    ]
    with open(os.path.join(SAVE_DIR, "build_300bus_result.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(result) + "\n")
    print("\n".join(result))


if __name__ == "__main__":
    main()
