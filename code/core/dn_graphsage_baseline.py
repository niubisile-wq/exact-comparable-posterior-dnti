# -*- coding: utf-8 -*-
"""
Step 2 GraphSAGE topology-classification baseline.

Scope:
  Strong point-estimate GNN baseline for 33-bus and 69-bus under the same
  variable-load observation protocol used by IP1. Pure PyTorch implementation;
  no torch_geometric dependency.
"""
import copy
import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import pandapower as pp

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGMA = 0.009
N_LF = 101
K_FIXED = 20
N_STEPS = 30000
BATCH = 256
LR = 3e-4
SEEDS = [42, 123, 456, 789, 2024]


def build_33bus_edges():
    # Standard IEEE 33-bus radial backbone, 0-indexed.
    edges_1 = [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
        (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
        (16, 17), (17, 18), (2, 19), (19, 20), (20, 21), (21, 22), (3, 23),
        (23, 24), (24, 25), (6, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (30, 31), (31, 32), (32, 33),
    ]
    return [(a - 1, b - 1) for a, b in edges_1]


def build_ieee69_edges_and_loads():
    # Same feeder definition as the existing 69-bus BOED script.
    branch_data = [
        (1,2,0.0005,0.0012,0,0),(2,3,0.0005,0.0012,0,0),(3,4,0.0015,0.0036,0,0),(4,5,0.0251,0.0294,0,0),
        (5,6,0.3660,0.1864,2.6,2.2),(6,7,0.3811,0.1941,40.4,30.0),(7,8,0.0922,0.0470,75.0,54.0),
        (8,9,0.0493,0.0251,30.0,22.0),(9,10,0.8190,0.2707,28.0,19.0),(10,11,0.1872,0.0619,145.0,104.0),
        (11,12,0.7114,0.2351,145.0,104.0),(12,13,1.0300,0.3400,8.0,5.5),(13,14,1.0440,0.3450,8.0,5.5),
        (14,15,1.0580,0.3496,0.0,0.0),(15,16,0.1966,0.0650,45.5,30.0),(16,17,0.3744,0.1238,60.0,35.0),
        (17,18,0.0047,0.0016,60.0,35.0),(18,19,0.3276,0.1083,0.0,0.0),(19,20,0.2106,0.0690,1.0,0.6),
        (20,21,0.3416,0.1129,114.0,81.0),(21,22,0.0140,0.0046,5.3,3.5),(22,23,0.1591,0.0526,0.0,0.0),
        (23,24,0.3463,0.1145,28.0,20.0),(24,25,0.7488,0.2475,0.0,0.0),(25,26,0.3089,0.1021,14.0,10.0),
        (26,27,0.1732,0.0572,14.0,10.0),(3,28,0.0044,0.0108,26.0,18.6),(28,29,0.0640,0.1565,26.0,18.6),
        (29,30,0.3978,0.1315,0.0,0.0),(30,31,0.0702,0.0232,0.0,0.0),(31,32,0.3510,0.1160,0.0,0.0),
        (32,33,0.8390,0.2816,14.0,10.0),(33,34,1.7080,0.5646,19.5,14.0),(34,35,1.4740,0.4873,6.0,4.0),
        (35,36,0.0044,0.0108,26.0,18.6),(36,37,0.0640,0.1565,26.0,18.6),(37,38,0.1053,0.1230,0.0,0.0),
        (38,39,0.0304,0.0355,24.0,17.0),(39,40,0.0018,0.0021,24.0,17.0),(40,41,0.7283,0.8509,1.2,1.0),
        (41,42,0.3100,0.3623,0.0,0.0),(42,43,0.0410,0.0478,6.0,4.3),(43,44,0.0092,0.0116,0.0,0.0),
        (44,45,0.1089,0.1373,39.2,26.3),(45,46,0.0009,0.0012,39.2,26.3),(4,47,0.0034,0.0084,0.0,0.0),
        (47,48,0.0851,0.2083,79.0,56.4),(48,49,0.2898,0.7091,384.7,274.5),(49,50,0.0822,0.2011,384.7,274.5),
        (8,51,0.0928,0.0473,40.5,28.3),(51,52,0.3319,0.1114,3.6,2.7),(9,53,0.1740,0.0886,4.35,3.5),
        (53,54,0.2030,0.1034,26.4,19.0),(54,55,0.2842,0.1447,24.0,17.2),(55,56,0.2813,0.1433,0.0,0.0),
        (56,57,1.5900,0.5337,0.0,0.0),(57,58,0.7837,0.2630,0.0,0.0),(58,59,0.3042,0.1006,100.0,72.0),
        (59,60,0.3861,0.1172,0.0,0.0),(60,61,0.5075,0.2585,1244.0,888.0),(61,62,0.0974,0.0496,32.0,23.0),
        (62,63,0.1450,0.0738,0.0,0.0),(63,64,0.7105,0.3619,227.0,162.0),(64,65,1.0410,0.5302,59.0,42.0),
        (11,66,0.2012,0.0611,18.0,13.0),(66,67,0.0047,0.0014,18.0,13.0),
        (12,68,0.7394,0.2444,28.0,20.0),(68,69,0.0047,0.0014,28.0,20.0),
    ]
    edges = [(a - 1, b - 1) for a, b, *_ in branch_data]
    loads = np.zeros(69, dtype=np.float32)
    for a, b, r, x, p, q in branch_data:
        loads[b - 1] += p
    mx = max(float(loads.max()), 1.0)
    return edges, loads / mx


def make_adj(n_bus, edges):
    adj = torch.zeros(n_bus, n_bus, dtype=torch.float32)
    for a, b in edges:
        adj[a, b] = 1.0
        adj[b, a] = 1.0
    deg = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (adj / deg).to(DEVICE)


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, adj):
        neigh = torch.einsum("ij,bjf->bif", adj, x)
        h = torch.cat([x, neigh], dim=-1)
        return torch.relu(self.norm(self.lin(h)))


class GraphSAGEClassifier(nn.Module):
    def __init__(self, n_bus, n_topos, in_dim=3, hidden=96):
        super().__init__()
        self.s1 = GraphSAGELayer(in_dim, hidden)
        self.s2 = GraphSAGELayer(hidden, hidden)
        self.s3 = GraphSAGELayer(hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 192), nn.GELU(), nn.LayerNorm(192),
            nn.Linear(192, n_topos)
        )

    def forward(self, x, adj):
        h = self.s1(x, adj)
        h = self.s2(h, adj)
        h = self.s3(h, adj)
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        return self.head(pooled)


def load_network_data(net_name):
    if net_name == "33bus":
        ckpt = torch.load(f"{SAVE_DIR}\\nre_ipc_loadaware.pt", map_location="cpu", weights_only=False)
        v_library = ckpt["V_library"]
        lf_grid = ckpt["lf_grid"]
        base_p = ckpt["base_P_norm"].astype(np.float32)
        edges = build_33bus_edges()
    elif net_name == "69bus":
        dat = np.load(f"{SAVE_DIR}\\v_library_69bus.npz")
        v_library = dat["V_library"]
        lf_grid = dat["lf_grid"]
        base_p = dat["base_P_norm"].astype(np.float32)
        edges, fallback_base = build_ieee69_edges_and_loads()
        if base_p.shape[0] != 69 or np.max(np.abs(base_p)) == 0:
            base_p = fallback_base
    else:
        raise ValueError(net_name)
    return v_library, lf_grid, base_p, edges


def make_features(v_library, lf_grid, base_p, rng, n_samples, k_fixed):
    n_topos, n_lf, n_bus = v_library.shape
    xs = np.zeros((n_samples, n_bus, 3), dtype=np.float32)
    ys = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        lf = lf_grid[lf_idx]
        installed = np.sort(rng.choice(np.arange(1, n_bus), k_fixed, replace=False))
        obs = v_library[ti, lf_idx, installed] + rng.normal(0, SIGMA, size=k_fixed)
        xs[i, installed, 0] = obs
        xs[i, installed, 1] = 1.0
        xs[i, installed, 2] = base_p[installed] * lf
        ys[i] = ti
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long)


def evaluate(model, adj, v_library, lf_grid, base_p, n_eval=1200):
    model.eval()
    rng = np.random.RandomState(77)
    correct = 0
    n_topos = v_library.shape[0]
    with torch.no_grad():
        for start in range(0, n_eval, BATCH):
            bs = min(BATCH, n_eval - start)
            xb, yb = make_features(v_library, lf_grid, base_p, rng, bs, K_FIXED)
            logits = model(xb.to(DEVICE), adj)
            pred = logits.argmax(dim=1).cpu()
            correct += int((pred == yb).sum().item())
    return correct / n_eval


def train_one(net_name, seed, v_library, lf_grid, base_p, adj):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_bus = v_library.shape[2]
    n_topos = v_library.shape[0]
    model = GraphSAGEClassifier(n_bus, n_topos).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.RandomState(seed)
    t0 = time.time()
    model.train()
    for step in range(1, N_STEPS + 1):
        xb, yb = make_features(v_library, lf_grid, base_p, rng, BATCH, K_FIXED)
        logits = model(xb.to(DEVICE), adj)
        loss = loss_fn(logits, yb.to(DEVICE))
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        if step % 10000 == 0:
            print(f"[{net_name} seed={seed}] step={step} loss={loss.item():.4f} elapsed={time.time() - t0:.0f}s", flush=True)
    acc = evaluate(model, adj, v_library, lf_grid, base_p)
    torch.save(
        {"model_state": model.state_dict(), "seed": seed, "net": net_name, "n_topos": v_library.shape[0]},
        f"{SAVE_DIR}\\graphsage_{net_name}_seed{seed}.pt",
    )
    return acc


def main():
    print(f"Device: {DEVICE}")
    all_results = {}
    refs = {
        "33bus": {"nre": 0.682, "enum": 0.721},
        "69bus": {"nre": 0.359, "enum": 0.402},
    }
    for net_name in ["33bus", "69bus"]:
        print("\n" + "=" * 80)
        print(f"GraphSAGE baseline: {net_name}")
        v_library, lf_grid, base_p, edges = load_network_data(net_name)
        adj = make_adj(v_library.shape[2], edges)
        print(f"N_TOPOS={v_library.shape[0]} N_BUS={v_library.shape[2]} K={K_FIXED} edges={len(edges)}")
        accs = []
        for seed in SEEDS:
            acc = train_one(net_name, seed, v_library, lf_grid, base_p, adj)
            accs.append(acc)
            print(f"[{net_name} seed={seed}] GraphSAGE top-1={acc:.3f}  NRE={refs[net_name]['nre']:.3f}  EnumBF={refs[net_name]['enum']:.3f}", flush=True)
        all_results[net_name] = accs

        out_path = f"{SAVE_DIR}\\graphsage_{net_name}_result.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"GraphSAGE baseline result: {net_name}\n")
            f.write(f"seeds={SEEDS}\n")
            f.write("accs=" + ",".join(f"{x:.4f}" for x in accs) + "\n")
            f.write(f"mean={np.mean(accs):.4f}\n")
            f.write(f"std={np.std(accs):.4f}\n")
            f.write(f"NRE_ref={refs[net_name]['nre']:.4f}\n")
            f.write(f"EnumBF_ref={refs[net_name]['enum']:.4f}\n")
            f.write("Boundary: GraphSAGE is a point-estimate baseline; it does not provide exact-comparable posterior quality.\n")
        print(f"Saved: {out_path}")

    summary = []
    summary.append("=" * 80)
    summary.append("GraphSAGE baseline summary")
    summary.append("Network   GraphSAGE mean/std   NRE ref   EnumBF ref")
    summary.append("-" * 80)
    for net_name in ["33bus", "69bus"]:
        accs = all_results[net_name]
        summary.append(f"{net_name:<8}  {np.mean(accs):.3f}+/-{np.std(accs):.3f}        {refs[net_name]['nre']:.3f}     {refs[net_name]['enum']:.3f}")
    summary.append("")
    summary.append("Q1-standard interpretation:")
    summary.append("  GraphSAGE is included as a strong shared-graph GNN point-estimate baseline.")
    summary.append("  It uses a common feeder backbone graph under the same observation protocol,")
    summary.append("  not a different dynamic graph for each candidate topology.")
    summary.append("  NRE's claim must not be only top-1 accuracy; the differentiator is")
    summary.append("  exact-comparable posterior output for uncertainty-aware decisions.")
    summary.append("=" * 80)
    out = "\n".join(summary)
    print(out)
    with open(f"{SAVE_DIR}\\graphsage_baseline_summary.txt", "w", encoding="utf-8") as f:
        f.write(out)


if __name__ == "__main__":
    main()
