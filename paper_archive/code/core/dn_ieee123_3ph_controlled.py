import copy
import time
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


ROOT = Path.home() / "Desktop" / "配电网实验_临时" / "feeder123" / "feeder123"
OUT = Path.home() / "Desktop" / "配电网实验_临时" / "ieee123_3ph_controlled_result.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LF_GRID = np.array([0.9, 1.0, 1.1], dtype=np.float32)
SIGMA = 0.0035
SEEDS = [42, 123, 456]
TRAIN_STEPS = 7000
BATCH = 192
LR = 3e-4
N_EVAL = 500
K_FIXED = 30
LOOP_TIES = [("54", "94"), ("151", "300")]
SOURCE_BUS = "150"


CONFIG_RX = {
    1: (0.36, 0.32),
    2: (0.36, 0.32),
    3: (0.36, 0.32),
    4: (0.36, 0.32),
    5: (0.36, 0.32),
    6: (0.36, 0.32),
    7: (0.45, 0.34),
    8: (0.45, 0.34),
    9: (0.52, 0.38),
    10: (0.60, 0.42),
    11: (0.60, 0.42),
    12: (0.24, 0.18),
}


class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class Controlled3PhNRE(nn.Module):
    def __init__(self, n_topos, n_bus):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_bus * 5, 384), nn.LayerNorm(384), nn.GELU())
        self.res1 = ResBlock(384)
        self.res2 = ResBlock(384)
        self.head = nn.Sequential(nn.Linear(384, 192), nn.LayerNorm(192), nn.GELU(), nn.Linear(192, n_topos))

    def forward(self, x):
        h = self.embed(x)
        h = self.res1(h)
        h = self.res2(h)
        return self.head(h)


def read_assets():
    line_df = pd.read_excel(ROOT / "line data.xls", header=None).iloc[3:].copy()
    switch_df = pd.read_excel(ROOT / "switch data.xls", header=None).iloc[3:].copy()
    load_df = pd.read_excel(ROOT / "spot loads data.xls", header=None).iloc[4:].copy()
    return line_df, switch_df, load_df


def build_spec():
    line_df, switch_df, load_df = read_assets()
    buses = set()
    base_branches = []
    closed_switches = []

    for _, row in line_df.iterrows():
        if pd.isna(row[0]) or pd.isna(row[1]):
            continue
        a = str(int(row[0]))
        b = str(int(row[1]))
        length_ft = float(row[2])
        config = int(row[3])
        buses.update([a, b])
        base_branches.append((a, b, length_ft, config))

    for _, row in switch_df.iterrows():
        if pd.isna(row[0]) or pd.isna(row[1]):
            continue
        a = str(int(row[0]))
        b = str(int(row[1]))
        status = str(row[2]).strip().lower()
        buses.update([a, b])
        if status == "closed":
            closed_switches.append((a, b))

    load_map = {}
    for _, row in load_df.iterrows():
        if pd.isna(row[0]):
            continue
        token = str(row[0]).strip()
        if not token.replace(".", "", 1).isdigit():
            continue
        bus = str(int(float(row[0])))
        vals = []
        for col in [2, 3, 4, 5, 6, 7]:
            vals.append(0.0 if pd.isna(row[col]) else float(row[col]))
        p_kw = vals[0] + vals[2] + vals[4]
        q_kvar = vals[1] + vals[3] + vals[5]
        load_map[bus] = (p_kw, q_kvar)
        buses.add(bus)

    graph = nx.Graph()
    graph.add_nodes_from(buses)
    for a, b, _, _ in base_branches:
        graph.add_edge(a, b)
    for a, b in closed_switches:
        graph.add_edge(a, b)

    main_cc = max(nx.connected_components(graph), key=lambda c: (SOURCE_BUS in c, len(c)))
    bus_ids = sorted(main_cc, key=lambda x: int(x))
    node2idx = {bus: i for i, bus in enumerate(bus_ids)}
    load_map = {bus: pq for bus, pq in load_map.items() if bus in main_cc}
    base_branches = [(a, b, length_ft, config) for a, b, length_ft, config in base_branches if a in main_cc and b in main_cc]
    closed_switches = [(a, b) for a, b in closed_switches if a in main_cc and b in main_cc]

    tie_branches = []
    for a, b in LOOP_TIES:
        if a in main_cc and b in main_cc and nx.has_path(graph, a, b):
            tie_branches.append((a, b))

    return {
        "bus_ids": bus_ids,
        "node2idx": node2idx,
        "base_branches": base_branches,
        "closed_switches": closed_switches,
        "tie_branches": tie_branches,
        "load_map": load_map,
    }


def branch_impedance(length_ft, config):
    r_pm, x_pm = CONFIG_RX.get(int(config), (0.50, 0.36))
    miles = max(length_ft / 5280.0, 0.001)
    return r_pm * miles, x_pm * miles


def phase_split(bus_raw):
    bus_num = int(bus_raw)
    base = np.array(
        [
            0.28 + 0.02 * (bus_num % 5),
            0.33 + 0.01 * ((bus_num + 1) % 4),
            0.39 + 0.015 * ((bus_num + 2) % 3),
        ],
        dtype=np.float32,
    )
    base /= base.sum()
    return base


def build_net(spec, lf):
    net = pp.create_empty_network()
    for _ in spec["bus_ids"]:
        pp.create_bus(net, vn_kv=4.16)
    pp.create_ext_grid(
        net,
        spec["node2idx"][SOURCE_BUS],
        vm_pu=1.0,
        s_sc_max_mva=1000,
        rx_max=0.1,
        rx_min=0.1,
        x0x_max=0.5,
        x0x_min=0.5,
        r0x0_max=0.1,
        r0x0_min=0.1,
    )
    for a, b, length_ft, config in spec["base_branches"]:
        r, x = branch_impedance(length_ft, config)
        pp.create_line_from_parameters(
            net,
            spec["node2idx"][a],
            spec["node2idx"][b],
            1.0,
            r,
            x,
            0.0,
            9999.0,
            r0_ohm_per_km=r * 3.0,
            x0_ohm_per_km=x * 3.0,
            c0_nf_per_km=0.0,
            in_service=True,
        )
    for a, b in spec["closed_switches"]:
        pp.create_line_from_parameters(
            net,
            spec["node2idx"][a],
            spec["node2idx"][b],
            1.0,
            0.003,
            0.002,
            0.0,
            9999.0,
            r0_ohm_per_km=0.009,
            x0_ohm_per_km=0.006,
            c0_nf_per_km=0.0,
            in_service=True,
        )
    for a, b in spec["tie_branches"]:
        pp.create_line_from_parameters(
            net,
            spec["node2idx"][a],
            spec["node2idx"][b],
            1.0,
            0.003,
            0.002,
            0.0,
            9999.0,
            r0_ohm_per_km=0.009,
            x0_ohm_per_km=0.006,
            c0_nf_per_km=0.0,
            in_service=False,
        )
    for bus_raw, (p_kw, q_kvar) in spec["load_map"].items():
        split = phase_split(bus_raw)
        pp.create_asymmetric_load(
            net,
            spec["node2idx"][bus_raw],
            p_a_mw=(p_kw * lf * split[0]) / 1000.0,
            p_b_mw=(p_kw * lf * split[1]) / 1000.0,
            p_c_mw=(p_kw * lf * split[2]) / 1000.0,
            q_a_mvar=(q_kvar * lf * split[0]) / 1000.0,
            q_b_mvar=(q_kvar * lf * split[1]) / 1000.0,
            q_c_mvar=(q_kvar * lf * split[2]) / 1000.0,
        )
    return net


def enumerate_topologies(spec):
    base_edges = [(spec["node2idx"][a], spec["node2idx"][b]) for a, b, _, _ in spec["base_branches"]]
    closed_edges = [(spec["node2idx"][a], spec["node2idx"][b]) for a, b in spec["closed_switches"]]
    tree_edges = base_edges + closed_edges
    tie_edges = [(spec["node2idx"][a], spec["node2idx"][b]) for a, b in spec["tie_branches"]]
    g0 = nx.Graph()
    g0.add_nodes_from(range(len(spec["bus_ids"])))
    g0.add_edges_from(tree_edges)
    topologies = [list(range(len(tree_edges)))]
    seen = {frozenset(topologies[0])}
    for tie_i, tie in enumerate(tie_edges):
        path = nx.shortest_path(g0, tie[0], tie[1])
        for j in range(len(path) - 1):
            old = frozenset((path[j], path[j + 1]))
            active = []
            removed = None
            for idx, edge in enumerate(tree_edges):
                if frozenset(edge) == old and removed is None:
                    removed = idx
                    continue
                active.append(idx)
            if removed is None:
                continue
            topo = active + [len(tree_edges) + tie_i]
            key = frozenset(topo)
            if key in seen:
                continue
            g = nx.Graph()
            g.add_nodes_from(range(len(spec["bus_ids"])))
            g.add_edges_from([tree_edges[idx] for idx in active] + [tie])
            if nx.is_tree(g):
                seen.add(key)
                topologies.append(topo)
    return topologies, len(tree_edges), len(tie_edges)


def run_pf_3ph(net_base, n_tree, n_tie, topo):
    net = copy.deepcopy(net_base)
    active_tree = {idx for idx in topo if idx < n_tree}
    active_tie = {idx - n_tree for idx in topo if idx >= n_tree}
    for li in range(n_tree):
        net.line.at[net.line.index[li], "in_service"] = li in active_tree
    for li in range(n_tie):
        net.line.at[net.line.index[n_tree + li], "in_service"] = li in active_tie
    try:
        pp.runpp_3ph(net, numba=False, max_iteration=30, tolerance_va_degree=1e-5)
        if net.converged:
            return np.stack(
                [net.res_bus_3ph.vm_a_pu.values, net.res_bus_3ph.vm_b_pu.values, net.res_bus_3ph.vm_c_pu.values],
                axis=1,
            ).astype(np.float32)
    except Exception:
        return None
    return None


def build_library(spec, topologies, n_tree, n_tie):
    n_bus = len(spec["bus_ids"])
    v_library = np.zeros((len(topologies), len(LF_GRID), n_bus, 3), dtype=np.float32)
    keep = np.ones(len(topologies), dtype=bool)
    failed = 0
    for i, topo in enumerate(topologies):
        ok = True
        for j, lf in enumerate(LF_GRID):
            net = build_net(spec, float(lf))
            v = run_pf_3ph(net, n_tree, n_tie, topo)
            if v is None:
                ok = False
                failed += 1
                break
            v_library[i, j] = v
        keep[i] = ok
    kept_topos = [topologies[i] for i in range(len(topologies)) if keep[i]]
    return kept_topos, v_library[keep], failed


def deployment(n_bus, k):
    return np.unique(np.rint(np.linspace(1, n_bus - 1, k)).astype(int))


def build_features(obs_v3, buses, n_bus, base_p_norm, lf):
    x = np.zeros(n_bus * 5, dtype=np.float32)
    x[buses] = (obs_v3[:, 0] - 1.0) / SIGMA
    x[n_bus + buses] = (obs_v3[:, 1] - 1.0) / SIGMA
    x[2 * n_bus + buses] = (obs_v3[:, 2] - 1.0) / SIGMA
    x[3 * n_bus + buses] = 1.0
    x[4 * n_bus : 5 * n_bus] = base_p_norm * lf
    return x


def exact_posterior(v_library, buses, obs_v3, lf_idx):
    da = (v_library[:, lf_idx, :, 0][:, buses] - obs_v3[:, 0]) / SIGMA
    db = (v_library[:, lf_idx, :, 1][:, buses] - obs_v3[:, 1]) / SIGMA
    dc = (v_library[:, lf_idx, :, 2][:, buses] - obs_v3[:, 2]) / SIGMA
    ll = -0.5 * (np.sum(da * da, axis=1) + np.sum(db * db, axis=1) + np.sum(dc * dc, axis=1))
    q = np.exp(ll - np.max(ll))
    q /= np.sum(q)
    return q.astype(np.float32)


def gen_batch(rng, v_library, base_p_norm, n_bus, n_topos, buses):
    xs = np.zeros((BATCH, n_bus * 5), dtype=np.float32)
    ys = np.zeros(BATCH, dtype=np.int64)
    q_batch = np.zeros((BATCH, n_topos), dtype=np.float32)
    for i in range(BATCH):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, len(LF_GRID))
        lf = float(LF_GRID[lf_idx])
        obs_v3 = v_library[ti, lf_idx, buses, :] + rng.normal(0.0, SIGMA, size=(len(buses), 3))
        xs[i] = build_features(obs_v3, buses, n_bus, base_p_norm, lf)
        ys[i] = ti
        q_batch[i] = exact_posterior(v_library, buses, obs_v3, lf_idx)
    return (
        torch.tensor(xs, dtype=torch.float32, device=DEVICE),
        torch.tensor(ys, dtype=torch.long, device=DEVICE),
        torch.tensor(q_batch, dtype=torch.float32, device=DEVICE),
    )


def infer(model, x_np):
    with torch.no_grad():
        logits = model(torch.tensor(x_np, dtype=torch.float32, device=DEVICE))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


def main():
    spec = build_spec()
    topologies, n_tree, n_tie = enumerate_topologies(spec)
    kept_topos, v_library, failed = build_library(spec, topologies, n_tree, n_tie)
    n_bus = len(spec["bus_ids"])
    buses = deployment(n_bus, K_FIXED)
    base_p = np.zeros(n_bus, dtype=np.float32)
    for bus_raw, (p_kw, _) in spec["load_map"].items():
        base_p[spec["node2idx"][bus_raw]] += p_kw / 1000.0
    base_p_norm = base_p / max(float(base_p.max()), 1e-8)

    rows = []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        model = Controlled3PhNRE(v_library.shape[0], n_bus).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_STEPS, eta_min=1e-5)
        kl_fn = nn.KLDivLoss(reduction="batchmean")
        ce_fn = nn.CrossEntropyLoss()

        model.train()
        t0 = time.time()
        for _ in range(TRAIN_STEPS):
            xb, yb, qb = gen_batch(rng, v_library, base_p_norm, n_bus, v_library.shape[0], buses)
            logits = model(xb)
            loss = kl_fn(torch.log_softmax(logits, dim=1), qb) + 0.15 * ce_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()

        x_eval = np.zeros((N_EVAL, n_bus * 5), dtype=np.float32)
        y_eval = np.zeros(N_EVAL, dtype=np.int64)
        q_eval = np.zeros((N_EVAL, v_library.shape[0]), dtype=np.float32)
        lf_eval = np.zeros(N_EVAL, dtype=np.int64)
        for i in range(N_EVAL):
            ti = rng.randint(0, v_library.shape[0])
            lf_idx = rng.randint(0, len(LF_GRID))
            lf = float(LF_GRID[lf_idx])
            obs_v3 = v_library[ti, lf_idx, buses, :] + rng.normal(0.0, SIGMA, size=(len(buses), 3))
            x_eval[i] = build_features(obs_v3, buses, n_bus, base_p_norm, lf)
            y_eval[i] = ti
            q_eval[i] = exact_posterior(v_library, buses, obs_v3, lf_idx)
            lf_eval[i] = lf_idx

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        probs = infer(model, x_eval)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        nre_sec = time.perf_counter() - t1
        nre_pred = np.argmax(probs, axis=1)

        t2 = time.perf_counter()
        exact_pred = np.argmax(q_eval, axis=1)
        exact_sec = time.perf_counter() - t2

        rows.append(
            {
                "seed": seed,
                "exact_top1": float(np.mean(exact_pred == y_eval)),
                "nre_top1": float(np.mean(nre_pred == y_eval)),
                "kl_ref_nre": float(
                    np.mean(np.sum(q_eval * (np.log(np.clip(q_eval, 1e-12, 1.0)) - np.log(np.clip(probs, 1e-12, 1.0))), axis=1))
                ),
                "nre_ms": nre_sec / N_EVAL * 1000.0,
                "exact_ms": exact_sec / N_EVAL * 1000.0,
                "train_sec": time.time() - t0,
            }
        )

    lines = [
        "IEEE123 controlled three-phase benchmark result",
        f"device={DEVICE}",
        f"n_bus={n_bus}",
        f"loop_ties={'; '.join([a + '-' + b for a, b in LOOP_TIES])}",
        f"raw_topologies={len(topologies)}",
        f"kept_topologies={len(kept_topos)}",
        f"failed_power_flows={failed}",
        f"K={len(buses)}",
        f"lf_grid={LF_GRID.tolist()}",
        f"sigma={SIGMA}",
        "seed,exact_top1,nre_top1,gap,kl_ref_nre,nre_ms,exact_ms,speedup,train_sec",
    ]
    for row in rows:
        gap = row["exact_top1"] - row["nre_top1"]
        speedup = row["exact_ms"] / max(row["nre_ms"], 1e-9)
        lines.append(
            f"{row['seed']},{row['exact_top1']:.4f},{row['nre_top1']:.4f},{gap:.4f},{row['kl_ref_nre']:.4f},"
            f"{row['nre_ms']:.6f},{row['exact_ms']:.6f},{speedup:.1f},{row['train_sec']:.1f}"
        )
    lines.append(f"mean_exact_top1={np.mean([r['exact_top1'] for r in rows]):.4f}")
    lines.append(f"mean_nre_top1={np.mean([r['nre_top1'] for r in rows]):.4f}")
    lines.append(f"mean_gap={np.mean([r['exact_top1'] - r['nre_top1'] for r in rows]):.4f}")
    lines.append(f"mean_kl={np.mean([r['kl_ref_nre'] for r in rows]):.4f}")
    lines.append(f"mean_speedup={np.mean([r['exact_ms'] / max(r['nre_ms'], 1e-9) for r in rows]):.1f}")
    lines.extend(
        [
            "Boundary",
            "- This is a controlled IEEE123 raw-asset-derived unbalanced three-phase benchmark using the natural loop-forming tie set.",
            "- It materially strengthens the large three-phase story beyond the 10/13/37-bus synthetic suite.",
            "- It remains a controlled raw-asset benchmark rather than a full utility-grade field feeder validation.",
        ]
    )
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
