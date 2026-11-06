# -*- coding: utf-8 -*-
"""
Step 2 GraphSAGE inference-time audit and unified baseline table.

This script loads the trained GraphSAGE checkpoints from dn_graphsage_baseline.py
and measures forward-pass latency on the same synthetic observation protocol.
"""
import os
import time
import warnings
import numpy as np
import torch

from dn_graphsage_baseline import (
    BATCH,
    DEVICE,
    K_FIXED,
    SEEDS,
    GraphSAGEClassifier,
    load_network_data,
    make_adj,
    make_features,
)

warnings.filterwarnings("ignore")

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
REFS = {
    "33bus": {"nre": 0.682, "enum": 0.721, "graphsage_mean": 0.6813, "graphsage_std": 0.0080},
    "69bus": {"nre": 0.359, "enum": 0.402, "graphsage_mean": 0.2927, "graphsage_std": 0.0093},
}


def load_model(net_name, seed, n_bus, n_topos):
    model = GraphSAGEClassifier(n_bus, n_topos).to(DEVICE)
    ckpt_path = os.path.join(SAVE_DIR, f"graphsage_{net_name}_seed{seed}.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def measure_latency(model, adj, v_library, lf_grid, base_p, seed, n_eval=4096, warmup=8):
    rng = np.random.RandomState(seed + 10000)
    xb, _ = make_features(v_library, lf_grid, base_p, rng, n_eval, K_FIXED)
    xb = xb.to(DEVICE)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(xb[:BATCH], adj)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for start in range(0, n_eval, BATCH):
            _ = model(xb[start:start + BATCH], adj)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return elapsed / n_eval * 1000.0


def main():
    lines = []
    lines.append("=" * 96)
    lines.append("Unified baseline table after Step 2 GraphSAGE audit")
    lines.append("Device: " + str(DEVICE))
    lines.append("Protocol: K=20 random sensors, variable load, sigma=0.009, 5 seeds.")
    lines.append("-" * 96)
    lines.append("Network  Method      top-1 mean/std       latency ms/sample   posterior output")
    lines.append("-" * 96)

    for net_name in ["33bus", "69bus"]:
        v_library, lf_grid, base_p, edges = load_network_data(net_name)
        adj = make_adj(v_library.shape[2], edges)
        latencies = []
        for seed in SEEDS:
            model = load_model(net_name, seed, v_library.shape[2], v_library.shape[0])
            latencies.append(measure_latency(model, adj, v_library, lf_grid, base_p, seed))
        lat_mean = float(np.mean(latencies))
        lat_std = float(np.std(latencies))
        r = REFS[net_name]
        lines.append(
            f"{net_name:<8} GraphSAGE  {r['graphsage_mean']:.4f}+/-{r['graphsage_std']:.4f}"
            f"       {lat_mean:.4f}+/-{lat_std:.4f}        no, shared feeder graph"
        )
        lines.append(
            f"{net_name:<8} NRE        {r['nre']:.4f} ref          not remeasured      yes, learned posterior"
        )
        lines.append(
            f"{net_name:<8} EnumBF     {r['enum']:.4f} ref         not remeasured      yes, exact enumeration"
        )
        lines.append("")

    lines.append("Q1-standard interpretation:")
    lines.append("- GraphSAGE is now present as a strong shared-graph GNN point-estimate")
    lines.append("  baseline for 33/69-bus under the same observation protocol.")
    lines.append("- 33-bus GraphSAGE is essentially tied with NRE in top-1 accuracy; do not claim a")
    lines.append("  top-1-only NRE win there.")
    lines.append("- 69-bus GraphSAGE is materially below NRE and EnumBF under the same K=20 protocol.")
    lines.append("- The defensible NRE contribution is exact-comparable posterior inference for")
    lines.append("  uncertainty-aware decisions, not merely classification accuracy.")
    lines.append("=" * 96)

    out = "\n".join(lines)
    with open(os.path.join(SAVE_DIR, "baseline_unified_table.txt"), "w", encoding="utf-8") as f:
        f.write(out + "\n")
    with open(os.path.join(SAVE_DIR, "graphsage_inference_time.txt"), "w", encoding="utf-8") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
