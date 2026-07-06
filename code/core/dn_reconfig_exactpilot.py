import copy
import re
import time
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandapower as pp
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


ROOT = Path(r"<LOCAL_WORKSPACE>\ElectricalSystemsDataForReconfiguration")
SAVE_DIR = Path(r"<LOCAL_WORKSPACE>")
SYSTEMS = [
    {"name": "SystemData_202", "k": 90, "train_n": 18000, "epochs": 6, "seed": 2026},
    {"name": "SystemData_417", "k": 150, "train_n": 12000, "epochs": 6, "seed": 4176},
]
LF_GRID = np.array([0.9, 1.0, 1.1], dtype=np.float32)
MAX_TOPOLOGIES = 220
SIGMA = 0.002
BATCH = 256
LR = 4e-4
N_EVAL = 600
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BUS_ROW = re.compile(r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$")
BRANCH_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s*$")
VNOM_ROW = re.compile(r"Vnominal\s*=?\s*([0-9.]+)")
SLACK_ROW = re.compile(r"(BusSE|Barra_SE)\s*[:=]?\s*(\d+)")


class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class StressNRE(nn.Module):
    def __init__(self, n_topos, n_bus):
        super().__init__()
        width = 512 if n_bus <= 250 else 640
        self.embed = nn.Sequential(nn.Linear(n_bus * 3, width), nn.LayerNorm(width), nn.GELU())
        self.res1 = ResBlock(width)
        self.res2 = ResBlock(width)
        self.head = nn.Sequential(nn.Linear(width, width // 2), nn.LayerNorm(width // 2), nn.GELU(), nn.Linear(width // 2, n_topos))

    def forward(self, x):
        h = self.embed(x)
        h = self.res1(h)
        h = self.res2(h)
        return self.head(h)


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
        pp.create_line_from_parameters(net, spec["node2idx"][f_raw], spec["node2idx"][t_raw], 1.0, max(r, 1e-4), max(x, 1e-4), 0.0, 9999.0, in_service=True)
    for f_raw, t_raw, _, r, x in spec["tie_branches"]:
        pp.create_line_from_parameters(net, spec["node2idx"][f_raw], spec["node2idx"][t_raw], 1.0, max(r, 1e-4), max(x, 1e-4), 0.0, 9999.0, in_service=False)
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
            return net.res_bus.vm_pu.values.astype(np.float32)
    except Exception:
        return None
    return None


def build_library(spec, topologies):
    n_tree = len(spec["base_branches"])
    n_tie = len(spec["tie_branches"])
    v_library = np.zeros((len(topologies), len(LF_GRID), spec["n_bus"]), dtype=np.float32)
    failed = 0
    for j, lf in enumerate(LF_GRID):
        net = build_net(spec, float(lf))
        for i, topo in enumerate(topologies):
            v = run_pf(net, n_tree, n_tie, topo)
            if v is None:
                failed += 1
                v = np.ones(spec["n_bus"], dtype=np.float32)
            v_library[i, j] = v
    base_p = np.zeros(spec["n_bus"], dtype=np.float32)
    for bus_raw, pd, _, _ in spec["buses"]:
        base_p[spec["node2idx"][bus_raw]] += float(pd) / 1000.0
    base_p_norm = base_p / max(float(base_p.max()), 1e-8)
    return {"V": v_library, "lf": LF_GRID, "base_p": base_p_norm, "failed": failed}


def deployment(n_bus, k):
    return np.unique(np.rint(np.linspace(1, n_bus - 1, k)).astype(int))


def exact_posterior(lib, buses, obs, lf_idx):
    pred_v = lib["V"][:, lf_idx, :][:, buses]
    ll = -0.5 * np.sum(((pred_v - obs[None, :]) / SIGMA) ** 2, axis=1)
    q = np.exp(ll - np.max(ll))
    q /= np.sum(q)
    return q.astype(np.float32)


def make_dataset(lib, rng, n, k):
    n_topos, n_lf, n_bus = lib["V"].shape
    xs = np.zeros((n, n_bus * 3), dtype=np.float32)
    ys = np.zeros(n, dtype=np.int64)
    qs = np.zeros((n, n_topos), dtype=np.float32)
    buses = deployment(n_bus, k)
    for i in range(n):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        lf = lib["lf"][lf_idx]
        obs = lib["V"][ti, lf_idx, buses] + rng.normal(0.0, SIGMA, size=len(buses))
        xs[i, buses] = (obs - 1.0) / SIGMA
        xs[i, n_bus + buses] = 1.0
        xs[i, 2 * n_bus + buses] = lib["base_p"][buses] * lf
        ys[i] = ti
        qs[i] = exact_posterior(lib, buses, obs, lf_idx)
    return xs, ys, qs


def train_and_eval(system_cfg):
    spec = parse_system(ROOT / f"{system_cfg['name']}.txt")
    topologies = enumerate_topologies(spec)
    lib = build_library(spec, topologies)
    rng = np.random.RandomState(system_cfg["seed"])
    xs, ys, qs = make_dataset(lib, rng, system_cfg["train_n"], system_cfg["k"])

    model = StressNRE(len(topologies), spec["n_bus"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(np.ceil(len(xs) / BATCH)) * system_cfg["epochs"], eta_min=1e-5)
    kl_fn = nn.KLDivLoss(reduction="batchmean")
    ce_fn = nn.CrossEntropyLoss()

    model.train()
    t0 = time.time()
    for epoch in range(system_cfg["epochs"]):
        order = rng.permutation(len(xs))
        for start in range(0, len(xs), BATCH):
            idx = order[start:start + BATCH]
            xb = torch.tensor(xs[idx], dtype=torch.float32).to(DEVICE)
            yb = torch.tensor(ys[idx], dtype=torch.long).to(DEVICE)
            qb = torch.tensor(qs[idx], dtype=torch.float32).to(DEVICE)
            logits = model(xb)
            loss = kl_fn(torch.log_softmax(logits, dim=1), qb) + 0.15 * ce_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            scheduler.step()
        print(f"{system_cfg['name']} epoch={epoch+1}/{system_cfg['epochs']} loss={loss.item():.4f} elapsed={time.time()-t0:.0f}s", flush=True)

    eval_rng = np.random.RandomState(system_cfg["seed"] + 99)
    buses = deployment(spec["n_bus"], system_cfg["k"])
    x_eval = np.zeros((N_EVAL, spec["n_bus"] * 3), dtype=np.float32)
    y_eval = np.zeros(N_EVAL, dtype=np.int64)
    q_eval = np.zeros((N_EVAL, len(topologies)), dtype=np.float32)
    lf_eval = np.zeros(N_EVAL, dtype=np.int64)
    for i in range(N_EVAL):
        ti = eval_rng.randint(0, len(topologies))
        lf_idx = eval_rng.randint(0, len(LF_GRID))
        lf = lib["lf"][lf_idx]
        obs = lib["V"][ti, lf_idx, buses] + eval_rng.normal(0.0, SIGMA, size=len(buses))
        x_eval[i, buses] = (obs - 1.0) / SIGMA
        x_eval[i, spec["n_bus"] + buses] = 1.0
        x_eval[i, 2 * spec["n_bus"] + buses] = lib["base_p"][buses] * lf
        y_eval[i] = ti
        lf_eval[i] = lf_idx
        q_eval[i] = exact_posterior(lib, buses, obs, lf_idx)

    xb = torch.tensor(x_eval, dtype=torch.float32)
    with torch.no_grad():
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        logits = model(xb.to(DEVICE))
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        nre_sec = time.perf_counter() - t1
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    nre_pred = np.argmax(probs, axis=1)

    t2 = time.perf_counter()
    exact_pred = np.zeros(N_EVAL, dtype=np.int64)
    for i in range(N_EVAL):
        mask = x_eval[i, spec["n_bus"]:2 * spec["n_bus"]] > 0.5
        obs = x_eval[i, :spec["n_bus"]][mask] * SIGMA + 1.0
        q = exact_posterior(lib, np.where(mask)[0], obs, int(lf_eval[i]))
        exact_pred[i] = int(np.argmax(q))
    exact_sec = time.perf_counter() - t2

    return {
        "name": system_cfg["name"],
        "n_bus": spec["n_bus"],
        "k": system_cfg["k"],
        "n_topologies": len(topologies),
        "failed_pf": lib["failed"],
        "train_sec": time.time() - t0,
        "exact_top1": float(np.mean(exact_pred == y_eval)),
        "nre_top1": float(np.mean(nre_pred == y_eval)),
        "kl_ref_nre": float(np.mean(np.sum(q_eval * (np.log(np.clip(q_eval, 1e-12, 1.0)) - np.log(np.clip(probs, 1e-12, 1.0))), axis=1))),
        "nre_ms": nre_sec / N_EVAL * 1000.0,
        "exact_ms": exact_sec / N_EVAL * 1000.0,
    }


def main():
    print(f"Device: {DEVICE}", flush=True)
    rows = [train_and_eval(cfg) for cfg in SYSTEMS]
    lines = [
        "Reconfiguration exact-comparable pilot result",
        f"systems={','.join(cfg['name'] for cfg in SYSTEMS)}",
        f"lf_grid={LF_GRID.tolist()}",
        f"sigma={SIGMA}",
        "system,n_bus,K,n_topologies,failed_pf,exact_top1,nre_top1,gap,kl_ref_nre,nre_ms,exact_ms,speedup,train_sec",
    ]
    for row in rows:
        speedup = row["exact_ms"] / max(row["nre_ms"], 1e-9)
        gap = row["exact_top1"] - row["nre_top1"]
        lines.append(
            f"{row['name']},{row['n_bus']},{row['k']},{row['n_topologies']},{row['failed_pf']},"
            f"{row['exact_top1']:.4f},{row['nre_top1']:.4f},{gap:.4f},{row['kl_ref_nre']:.4f},"
            f"{row['nre_ms']:.6f},{row['exact_ms']:.6f},{speedup:.1f},{row['train_sec']:.1f}"
        )
    lines.extend(
        [
            "Boundary",
            "- This is a balanced exact-comparable pilot on larger local reconfiguration assets.",
            "- It strengthens the full-scale learned-posterior story but does not replace the missing unbalanced-three-phase benchmark.",
        ]
    )
    out = SAVE_DIR / "reconfig_exactpilot_result.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
