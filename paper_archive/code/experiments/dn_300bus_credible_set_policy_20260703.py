# -*- coding: utf-8 -*-
"""
300-bus credible-set policy audit.

Evaluates whether calibrated posterior mass sets provide a more principled
operating policy than a fixed top-20 candidate list under severe missing data.
"""

from pathlib import Path
import sys
import numpy as np
import torch

TEMP_ROOT = Path.home() / "Desktop" / "配电网实验_临时"
sys.path.insert(0, str(TEMP_ROOT))

import dn_300bus_redundant_sensing_attack_20260702 as m

PKG_ROOT = Path.home() / "Desktop" / "配电网论文一区投稿成果包_20260702"
PKG_STATS = PKG_ROOT / "03_frozen_tables_stats"
OUT_NAME = "300bus_credible_set_policy_20260703.txt"

SEEDS = [42, 123, 456]
N_EVAL = 12000
K_VALUES = [280, 299]
MASS_TARGETS = [0.80, 0.90, 0.95, 0.99]


def load_model(lib, k, seed):
    n_topos, _, n_bus = lib["V"].shape
    model = m.base.NRE300(n_topos, n_bus).to(m.DEVICE)
    path = TEMP_ROOT / f"nre_300bus_ipc_K{k}_miss30_redundant_seed{seed}_20260702.pt"
    obj = torch.load(path, map_location=m.DEVICE, weights_only=False)
    model.load_state_dict(obj["model_state"])
    return model


def predict_prob(models, xs):
    probs = []
    for model in models:
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(xs), m.BATCH):
                xb = torch.tensor(xs[start:start + m.BATCH], dtype=torch.float32).to(m.DEVICE)
                chunks.append(torch.softmax(model(xb), dim=1).cpu().numpy())
        probs.append(np.vstack(chunks))
    return np.mean(np.stack(probs, axis=0), axis=0)


def credible_set_indices(p_row, mass):
    order = np.argsort(-p_row)
    csum = np.cumsum(p_row[order])
    k = int(np.searchsorted(csum, mass, side="left")) + 1
    return order[:k]


def eval_k(lib, k):
    old_k = m.K_ATTACK
    m.K_ATTACK = k
    xs, y_true, y_map, ll = m.make_dataset(lib, N_EVAL, 940000 + k, k, return_ll=True)
    models = [load_model(lib, k, seed) for seed in SEEDS]
    p = predict_prob(models, xs)
    direct = np.argmax(p, axis=1)
    top20 = np.argsort(-p, axis=1)[:, :20]
    top20_rerank = np.array([top20[i, int(np.argmax(ll[i, top20[i]]))] for i in range(len(y_true))], dtype=np.int64)

    rows = []
    for mass in MASS_TARGETS:
        sizes = np.empty(len(y_true), dtype=np.int64)
        truth_cov = np.zeros(len(y_true), dtype=bool)
        map_cov = np.zeros(len(y_true), dtype=bool)
        rerank_pred = np.empty(len(y_true), dtype=np.int64)
        for i in range(len(y_true)):
            cs = credible_set_indices(p[i], mass)
            sizes[i] = len(cs)
            truth_cov[i] = y_true[i] in cs
            map_cov[i] = y_map[i] in cs
            rerank_pred[i] = cs[int(np.argmax(ll[i, cs]))]
        rows.append(
            {
                "mass": mass,
                "mean_size": float(np.mean(sizes)),
                "median_size": float(np.median(sizes)),
                "p90_size": float(np.quantile(sizes, 0.90)),
                "truth_coverage": float(np.mean(truth_cov)),
                "map_coverage": float(np.mean(map_cov)),
                "credible_rerank_true_acc": float(np.mean(rerank_pred == y_true)),
                "credible_rerank_map_agree": float(np.mean(rerank_pred == y_map)),
                "mean_size_vs_top20": float(np.mean(sizes) / 20.0),
            }
        )

    summary = {
        "K": k,
        "observed_after_missing": int(k - int(k * m.MISS_RATE)),
        "same_draw_exact_map_top1": float(np.mean(y_map == y_true)),
        "direct_true_acc": float(np.mean(direct == y_true)),
        "direct_map_agree": float(np.mean(direct == y_map)),
        "truth_in_top20": float(np.mean([y_true[i] in top20[i] for i in range(len(y_true))])),
        "full_rerank20_true_acc": float(np.mean(top20_rerank == y_true)),
        "full_rerank20_map_agree": float(np.mean(top20_rerank == y_map)),
    }
    m.K_ATTACK = old_k
    return summary, rows


def main():
    lib = m.base.load_300_lib()
    lines = []
    lines.append("300-bus credible-set policy audit")
    lines.append("date=2026-07-03")
    lines.append("role=posterior credible-set calibration and exact-rerank policy audit under severe 30% missing")
    lines.append("not_claimed=direct top-1 recovery; no training; no checkpoint writes")
    lines.append(f"device={m.DEVICE}")
    lines.append(f"eval_samples_per_K={N_EVAL}")
    lines.append(f"mass_targets={' '.join(f'{x:.2f}' for x in MASS_TARGETS)}")
    for k in K_VALUES:
        summary, rows = eval_k(lib, k)
        lines.append(f"K={k}")
        for key, val in summary.items():
            lines.append(f"{key}={val:.4f}" if isinstance(val, float) else f"{key}={val}")
        lines.append("mass,mean_size,median_size,p90_size,truth_coverage,map_coverage,credible_rerank_true_acc,credible_rerank_map_agree,mean_size_vs_top20")
        for r in rows:
            lines.append(
                f"{r['mass']:.2f},{r['mean_size']:.4f},{r['median_size']:.4f},{r['p90_size']:.4f},{r['truth_coverage']:.4f},"
                f"{r['map_coverage']:.4f},{r['credible_rerank_true_acc']:.4f},{r['credible_rerank_map_agree']:.4f},{r['mean_size_vs_top20']:.4f}"
            )
    text = "\n".join(lines) + "\n"
    out = PKG_STATS / OUT_NAME
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
