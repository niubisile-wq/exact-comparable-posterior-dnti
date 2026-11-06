# -*- coding: utf-8 -*-
"""
Step 3 robustness-boundary audit for 33-bus IP-C.

Loads existing 33-bus robust NRE checkpoints and the fixed naive NRE baseline.
Produces the required Step 3 artifacts:
  - missing_curve_33bus_result.txt
  - outage_33bus_result.txt
  - noise_sensitivity_33bus_result.txt

Boundary: the outage experiment is outage-induced structured measurement loss,
not feeder-fault diagnosis or topology-change localization.
"""
import os
import time
import warnings
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
N_BUS = 33
K_FIXED = 20
BATCH = 512
N_EVAL = 2000
SEEDS = [42, 123, 456, 789, 2024, 500, 1000, 1500]
MISSING_RATES = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
NOISE_SIGMAS = [0.003, 0.006, 0.009, 0.015, 0.020, 0.030]
BASE_SIGMA = 0.009
ROBUST_MODEL_FILES = {
    42: "nre_ipc_loadaware.pt",
    123: "nre_ipc_seed123.pt",
    456: "nre_ipc_seed456.pt",
    789: "nre_ipc_seed789.pt",
    2024: "nre_ipc_seed2024.pt",
    500: "nre_ipc_seed500.pt",
    1000: "nre_ipc_seed1000.pt",
    1500: "nre_ipc_seed1500.pt",
}


class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class LoadAwareNRE(nn.Module):
    def __init__(self, n_topo, n_bus=33):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_bus * 3, 512), nn.LayerNorm(512), nn.GELU())
        self.res1 = ResBlock(512)
        self.res2 = ResBlock(512)
        self.res3 = ResBlock(512)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, n_topo),
        )

    def forward(self, x):
        h = self.embed(x)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        return self.head(h)


def load_library():
    ckpt = torch.load(os.path.join(SAVE_DIR, "nre_ipc_loadaware.pt"), map_location="cpu", weights_only=False)
    return {
        "v_library": ckpt["V_library"].astype(np.float32),
        "lf_grid": ckpt["lf_grid"].astype(np.float32),
        "base_p": ckpt["base_P_norm"].astype(np.float32),
        "n_topos": int(ckpt["N_TOPOS"]),
    }


def load_model(path, n_topos):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = LoadAwareNRE(n_topos, N_BUS).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def make_features(v_library, lf_grid, base_p, rng, n_eval, miss_rate, sigma, outage_mode=False):
    n_topos, n_lf, n_bus = v_library.shape
    xs = np.zeros((n_eval, n_bus * 3), dtype=np.float32)
    ys = np.zeros(n_eval, dtype=np.int64)
    meta_missing = np.zeros(n_eval, dtype=np.int64)
    outage_edges = build_outage_scenarios() if outage_mode else None
    scenario_hits = defaultdict(int)

    for i in range(n_eval):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        lf = lf_grid[lf_idx]
        installed = np.sort(rng.choice(np.arange(1, n_bus), K_FIXED, replace=False))

        if outage_mode:
            name, affected = outage_edges[rng.randint(0, len(outage_edges))]
            affected = np.array(sorted(set(affected) - {0}), dtype=np.int64)
            reported = np.array([b for b in installed if b not in affected], dtype=np.int64)
            scenario_hits[name] += 1
        else:
            n_miss = int(K_FIXED * miss_rate)
            if n_miss > 0:
                miss_idx = rng.choice(len(installed), n_miss, replace=False)
                reported = np.delete(installed, miss_idx)
            else:
                reported = installed

        obs_v = v_library[ti, lf_idx, reported] + rng.normal(0, sigma, len(reported))
        xs[i, reported] = obs_v
        xs[i, n_bus + reported] = 1.0
        xs[i, 2 * n_bus + reported] = base_p[reported] * lf
        ys[i] = ti
        meta_missing[i] = K_FIXED - len(reported)

    return xs, ys, meta_missing, dict(scenario_hits)


def batched_accuracy(model, xs, ys):
    correct = 0
    with torch.no_grad():
        for start in range(0, len(ys), BATCH):
            xb = torch.tensor(xs[start:start + BATCH], dtype=torch.float32, device=DEVICE)
            pred = model(xb).argmax(dim=1).cpu().numpy()
            correct += int(np.sum(pred == ys[start:start + BATCH]))
    return correct / len(ys)


def eval_models(models, naive_model, lib, rng_seed, miss_rate=0.0, sigma=BASE_SIGMA, outage_mode=False):
    rng = np.random.RandomState(rng_seed)
    xs, ys, meta_missing, scenario_hits = make_features(
        lib["v_library"], lib["lf_grid"], lib["base_p"], rng, N_EVAL, miss_rate, sigma, outage_mode
    )
    naive_acc = batched_accuracy(naive_model, xs, ys)
    rows = []
    for seed, model in models.items():
        rob_acc = batched_accuracy(model, xs, ys)
        rows.append(
            {
                "seed": seed,
                "rob": rob_acc,
                "naive": naive_acc,
                "delta": rob_acc - naive_acc,
                "mean_missing": float(np.mean(meta_missing)),
                "max_missing": int(np.max(meta_missing)),
            }
        )
    return rows, scenario_hits


def summarize(rows):
    robs = np.array([r["rob"] for r in rows], dtype=float)
    naive = float(rows[0]["naive"])
    deltas = np.array([r["delta"] for r in rows], dtype=float)
    return {
        "rob_mean": float(np.mean(robs)),
        "rob_std": float(np.std(robs)),
        "naive": naive,
        "delta_mean": float(np.mean(deltas)),
        "delta_min": float(np.min(deltas)),
        "all_delta_positive": bool(np.all(deltas > 0.0)),
        "mean_missing": float(np.mean([r["mean_missing"] for r in rows])),
        "max_missing": int(np.max([r["max_missing"] for r in rows])),
    }


def build_outage_scenarios():
    # Standard 33-bus radial backbone, 0-indexed. Each scenario drops measurements
    # from the downstream island of one branch; this is structured sensor loss.
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
        (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),
        (15, 16), (16, 17), (1, 18), (18, 19), (19, 20), (20, 21), (2, 22),
        (22, 23), (23, 24), (5, 25), (25, 26), (26, 27), (27, 28), (28, 29),
        (29, 30), (30, 31), (31, 32),
    ]
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    scenarios = []
    # Chosen to span lateral/terminal outages without wiping out most sensors.
    chosen_edges = [(1, 18), (2, 22), (5, 25), (8, 9), (14, 15), (29, 30)]
    for a, b in chosen_edges:
        seen = {a}
        q = deque([b])
        affected = []
        while q:
            u = q.popleft()
            if u in seen:
                continue
            seen.add(u)
            affected.append(u)
            for v in adj[u]:
                if v not in seen:
                    q.append(v)
        scenarios.append((f"line_{a+1}_{b+1}", affected))
    return scenarios


def write_missing(models, naive_model, lib):
    lines = []
    lines.append("Step 3 missing-rate curve: 33-bus IP-C")
    lines.append(f"seeds={SEEDS}")
    lines.append(f"n_eval={N_EVAL} per rate, K={K_FIXED}, sigma={BASE_SIGMA}")
    lines.append("miss_rate,rob_mean,rob_std,naive,delta_mean,delta_min,all_delta_positive")
    all_gate_10_to_30 = True
    for miss in MISSING_RATES:
        rows, _ = eval_models(models, naive_model, lib, rng_seed=7700 + int(miss * 1000), miss_rate=miss)
        s = summarize(rows)
        if 0.10 <= miss <= 0.30 and not s["all_delta_positive"]:
            all_gate_10_to_30 = False
        lines.append(
            f"{miss:.2f},{s['rob_mean']:.4f},{s['rob_std']:.4f},{s['naive']:.4f},"
            f"{s['delta_mean']:.4f},{s['delta_min']:.4f},{s['all_delta_positive']}"
        )
    lines.append(f"gate_all_seed_positive_delta_10_to_30pct={all_gate_10_to_30}")
    lines.append("Boundary: at 0% missing, the robust model may trade a small clean-data accuracy loss for missing-data robustness.")
    lines.append("Boundary: 40-50% missing are stress-test boundary points, not the trained operating range.")
    out = "\n".join(lines) + "\n"
    with open(os.path.join(SAVE_DIR, "missing_curve_33bus_result.txt"), "w", encoding="utf-8") as f:
        f.write(out)
    print(out)


def write_noise(models, naive_model, lib):
    lines = []
    lines.append("Step 3 noise sensitivity: 33-bus IP-C at 30% missing")
    lines.append(f"seeds={SEEDS}")
    lines.append(f"n_eval={N_EVAL} per sigma, K={K_FIXED}, miss_rate=0.30")
    lines.append("sigma,rob_mean,rob_std,naive,delta_mean,delta_min,all_delta_positive")
    gate = True
    for sigma in NOISE_SIGMAS:
        rows, _ = eval_models(models, naive_model, lib, rng_seed=8800 + int(sigma * 100000), miss_rate=0.30, sigma=sigma)
        s = summarize(rows)
        if not s["all_delta_positive"]:
            gate = False
        lines.append(
            f"{sigma:.3f},{s['rob_mean']:.4f},{s['rob_std']:.4f},{s['naive']:.4f},"
            f"{s['delta_mean']:.4f},{s['delta_min']:.4f},{s['all_delta_positive']}"
        )
    lines.append(f"gate_all_seed_positive_delta_all_sigmas={gate}")
    lines.append("Boundary: this is voltage-measurement noise sensitivity under missing data, not PMU/AMI hardware modeling.")
    out = "\n".join(lines) + "\n"
    with open(os.path.join(SAVE_DIR, "noise_sensitivity_33bus_result.txt"), "w", encoding="utf-8") as f:
        f.write(out)
    print(out)


def write_outage(models, naive_model, lib):
    rows, scenario_hits = eval_models(models, naive_model, lib, rng_seed=9900, sigma=BASE_SIGMA, outage_mode=True)
    s = summarize(rows)
    lines = []
    lines.append("Step 3 outage-induced missing experiment: 33-bus IP-C")
    lines.append(f"seeds={SEEDS}")
    lines.append(f"n_eval={N_EVAL}, K={K_FIXED}, sigma={BASE_SIGMA}")
    lines.append("Scenario definition: randomly choose one predefined feeder branch and drop all installed sensors on its downstream island.")
    lines.append("This is structured measurement loss, not feeder-fault diagnosis or topology-change localization.")
    lines.append("scenario_hits=" + ",".join(f"{k}:{v}" for k, v in sorted(scenario_hits.items())))
    lines.append("seed,rob,naive,delta,mean_missing,max_missing")
    for r in rows:
        lines.append(
            f"{r['seed']},{r['rob']:.4f},{r['naive']:.4f},{r['delta']:.4f},"
            f"{r['mean_missing']:.2f},{r['max_missing']}"
        )
    lines.append(
        "summary,"
        f"{s['rob_mean']:.4f},{s['naive']:.4f},{s['delta_mean']:.4f},"
        f"delta_min={s['delta_min']:.4f},all_delta_positive={s['all_delta_positive']},"
        f"mean_missing={s['mean_missing']:.2f},max_missing={s['max_missing']}"
    )
    lines.append(f"gate_outage_direction_supports_ipc={s['all_delta_positive']}")
    lines.append("Boundary: claim only unified handling of outage-induced missing measurements.")
    out = "\n".join(lines) + "\n"
    with open(os.path.join(SAVE_DIR, "outage_33bus_result.txt"), "w", encoding="utf-8") as f:
        f.write(out)
    print(out)


def main():
    print(f"Device: {DEVICE}")
    t0 = time.time()
    lib = load_library()
    print(f"Loaded library: N_TOPOS={lib['n_topos']} V={lib['v_library'].shape}")
    naive_model = load_model(os.path.join(SAVE_DIR, "nre_ip1_v5a.pt"), lib["n_topos"])
    models = {}
    for seed in SEEDS:
        models[seed] = load_model(os.path.join(SAVE_DIR, ROBUST_MODEL_FILES[seed]), lib["n_topos"])
    print(f"Loaded {len(models)} robust models")
    write_missing(models, naive_model, lib)
    write_outage(models, naive_model, lib)
    write_noise(models, naive_model, lib)
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
