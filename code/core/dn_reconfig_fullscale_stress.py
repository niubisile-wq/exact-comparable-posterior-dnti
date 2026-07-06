import copy
import re
import time
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandapower as pp

warnings.filterwarnings("ignore")


ROOT = Path(r"<LOCAL_WORKSPACE>\ElectricalSystemsDataForReconfiguration")
OUT = Path(r"<LOCAL_WORKSPACE>\reconfig_fullscale_stress_result.txt")
SYSTEMS = ["SystemData_119", "SystemData_202", "SystemData_417"]
LF_GRID = [0.9, 1.0, 1.1]
MAX_TOPOLOGIES = 220


BUS_ROW = re.compile(r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$")
BRANCH_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s*$")
VNOM_ROW = re.compile(r"Vnominal\s*=?\s*([0-9.]+)")
SLACK_ROW = re.compile(r"(BusSE|Barra_SE)\s*[:=]?\s*(\d+)")


def parse_system(path: Path):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    buses = []
    branches = []
    vnom = None
    slack = None
    in_branch = False

    for line in lines:
        if vnom is None:
            m = VNOM_ROW.search(line)
            if m:
                vnom = float(m.group(1))
        if slack is None:
            m = SLACK_ROW.search(line)
            if m:
                slack = int(m.group(2))

        lower = line.lower()
        if (("send" in lower or "env" in lower) and ("recv" in lower or "rec" in lower) and "line" in lower):
            in_branch = True
            continue

        if in_branch:
            m = BRANCH_ROW.match(line)
            if m:
                send, recv, line_id, r, x = m.groups()
                branches.append((int(send), int(recv), int(line_id), float(r), float(x)))
            continue

        m = BUS_ROW.match(line)
        if m:
            bus, pd, qd, qc = m.groups()
            buses.append((int(bus), float(pd), float(qd), float(qc)))

    bus_ids = sorted({row[0] for row in buses})
    node2idx = {bus: i for i, bus in enumerate(bus_ids)}
    n_bus = len(bus_ids)
    sorted_branches = sorted(branches, key=lambda row: row[2])
    n_tree = max(0, n_bus - 1)
    base_branches = sorted_branches[:n_tree]
    tie_branches = sorted_branches[n_tree:]

    return {
        "name": path.stem,
        "vnom": vnom or 12.66,
        "slack_bus_raw": slack if slack in node2idx else bus_ids[0],
        "buses": buses,
        "node2idx": node2idx,
        "base_branches": base_branches,
        "tie_branches": tie_branches,
        "n_bus": n_bus,
    }


def build_net(spec, lf=1.0):
    net = pp.create_empty_network()
    for _ in range(spec["n_bus"]):
        pp.create_bus(net, vn_kv=spec["vnom"])
    for f_raw, t_raw, _, r, x in spec["base_branches"]:
        pp.create_line_from_parameters(
            net,
            spec["node2idx"][f_raw],
            spec["node2idx"][t_raw],
            1.0,
            max(r, 1e-4),
            max(x, 1e-4),
            0.0,
            9999.0,
            in_service=True,
        )
    for f_raw, t_raw, _, r, x in spec["tie_branches"]:
        pp.create_line_from_parameters(
            net,
            spec["node2idx"][f_raw],
            spec["node2idx"][t_raw],
            1.0,
            max(r, 1e-4),
            max(x, 1e-4),
            0.0,
            9999.0,
            in_service=False,
        )
    for bus_raw, pd, qd, _ in spec["buses"]:
        if pd > 0 or qd > 0:
            pp.create_load(net, spec["node2idx"][bus_raw], p_mw=(pd * lf) / 1000.0, q_mvar=(qd * lf) / 1000.0)
    pp.create_ext_grid(net, spec["node2idx"][spec["slack_bus_raw"]], vm_pu=1.0)
    return net


def enumerate_topologies(spec):
    base_edges = [(spec["node2idx"][f], spec["node2idx"][t]) for f, t, _, _, _ in spec["base_branches"]]
    tie_edges = [(spec["node2idx"][f], spec["node2idx"][t]) for f, t, _, _, _ in spec["tie_branches"]]
    base_graph = nx.Graph()
    base_graph.add_nodes_from(range(spec["n_bus"]))
    base_graph.add_edges_from(base_edges)
    topologies = [list(range(len(base_edges)))]
    seen = {frozenset(topologies[0])}

    for tie_i, tie in enumerate(tie_edges):
        try:
            path = nx.shortest_path(base_graph, tie[0], tie[1])
        except Exception:
            continue
        step = 1 if len(path) <= 16 else 2 if len(path) <= 32 else 3
        for j in range(0, len(path) - 1, step):
            old = frozenset((path[j], path[j + 1]))
            active = []
            removed = None
            for idx, edge in enumerate(base_edges):
                if frozenset(edge) == old and removed is None:
                    removed = idx
                    continue
                active.append(idx)
            if removed is None:
                continue
            topo = active + [len(base_edges) + tie_i]
            key = frozenset(topo)
            if key in seen:
                continue
            g = nx.Graph()
            g.add_nodes_from(range(spec["n_bus"]))
            g.add_edges_from([base_edges[idx] for idx in active] + [tie])
            if nx.is_tree(g):
                seen.add(key)
                topologies.append(topo)
            if len(topologies) >= MAX_TOPOLOGIES:
                return topologies
    return topologies


def run_pf(net_base, n_tree, n_tie, topo):
    net = copy.deepcopy(net_base)
    active_tree = {idx for idx in topo if idx < n_tree}
    active_tie = {idx - n_tree for idx in topo if idx >= n_tree}
    for li in range(n_tree):
        net.line.at[net.line.index[li], "in_service"] = li in active_tree
    for li in range(n_tie):
        net.line.at[net.line.index[n_tree + li], "in_service"] = li in active_tie
    try:
        pp.runpp(net, algorithm="bfsw", numba=False, max_iteration=100, tolerance_mva=1e-6)
        if net.converged:
            vm = net.res_bus.vm_pu.values
            return float(vm.min()), float(vm.max()), float(net.res_line.pl_mw.sum() * 1000.0)
    except Exception:
        return None
    return None


def eval_system(spec):
    topologies = enumerate_topologies(spec)
    n_tree = len(spec["base_branches"])
    n_tie = len(spec["tie_branches"])
    result = {
        "name": spec["name"],
        "n_bus": spec["n_bus"],
        "n_tree": n_tree,
        "n_tie": n_tie,
        "n_topologies": len(topologies),
        "lf_rows": [],
    }
    for lf in LF_GRID:
        net = build_net(spec, lf=lf)
        base = run_pf(net, n_tree, n_tie, topologies[0])
        t0 = time.time()
        vals = [run_pf(net, n_tree, n_tie, topo) for topo in topologies]
        elapsed = time.time() - t0
        good = [v for v in vals if v is not None]
        result["lf_rows"].append(
            {
                "lf": lf,
                "base_ok": base is not None,
                "base_vmin": None if base is None else base[0],
                "base_vmax": None if base is None else base[1],
                "conv": len(good),
                "conv_rate": len(good) / len(topologies),
                "worst_vmin": None if not good else min(v[0] for v in good),
                "best_vmax": None if not good else max(v[1] for v in good),
                "mean_loss_kw": None if not good else float(np.mean([v[2] for v in good])),
                "elapsed_sec": elapsed,
            }
        )
    return result


def main():
    rows = []
    for name in SYSTEMS:
        rows.append(eval_system(parse_system(ROOT / f"{name}.txt")))

    lines = [
        "Reconfiguration full-scale stress result",
        f"root={ROOT}",
        f"systems={','.join(SYSTEMS)}",
        f"lf_grid={LF_GRID}",
        f"max_topologies={MAX_TOPOLOGIES}",
        "",
    ]
    for row in rows:
        lines.append(
            f"{row['name']}: n_bus={row['n_bus']} n_tree={row['n_tree']} n_tie={row['n_tie']} sampled_topologies={row['n_topologies']}"
        )
        for lf_row in row["lf_rows"]:
            lines.append(
                f"  lf={lf_row['lf']:.1f} base_ok={lf_row['base_ok']} conv={lf_row['conv']}/{row['n_topologies']} "
                f"conv_rate={lf_row['conv_rate']:.3f} "
                f"base_vmin={lf_row['base_vmin'] if lf_row['base_vmin'] is not None else 'NA'} "
                f"worst_vmin={lf_row['worst_vmin'] if lf_row['worst_vmin'] is not None else 'NA'} "
                f"best_vmax={lf_row['best_vmax'] if lf_row['best_vmax'] is not None else 'NA'} "
                f"mean_loss_kw={lf_row['mean_loss_kw'] if lf_row['mean_loss_kw'] is not None else 'NA'} "
                f"elapsed_sec={lf_row['elapsed_sec']:.1f}"
            )
        lines.append("")

    lines.extend(
        [
            "Boundary",
            "- This is a balanced full-scale stress track based on local reconfiguration assets.",
            "- It strengthens scale and topology-library coverage beyond the frozen synthetic 300-bus point.",
            "- It does not replace the remaining unbalanced-three-phase benchmark gap.",
        ]
    )
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
