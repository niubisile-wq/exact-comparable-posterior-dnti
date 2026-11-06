# -*- coding: utf-8 -*-
"""
300-bus confidence-stratified direct/rerank policy audit.

Shows whether high-confidence NRE predictions are reliable and whether
low-confidence samples should be routed to top-20 exact residual reranking.
No training and no checkpoint writes are performed.
"""

from pathlib import Path
import numpy as np
import torch
import dn_300bus_redundant_sensing_attack_20260702 as m

ROOT = Path(__file__).resolve().parent
PKG_ROOT = Path.home() / "Desktop" / "配电网论文一区投稿成果包_20260702"
PKG_STATS = PKG_ROOT / "03_frozen_tables_stats"
PKG_CODE = PKG_ROOT / "02_code"
OUT_NAME = "300bus_confidence_stratified_policy_20260702.txt"
SEEDS = [42, 123, 456]
N_EVAL = 12000
KEEP_DIRECT_FRACS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 1.00]


def softmax_np(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    p = np.exp(z)
    return p / np.sum(p, axis=1, keepdims=True)


def load_models(lib, k):
    n_topos, _, n_bus = lib["V"].shape
    models = []
    for seed in SEEDS:
        model = m.base.NRE300(n_topos, n_bus).to(m.DEVICE)
        path = ROOT / f"nre_300bus_ipc_K{k}_miss30_redundant_seed{seed}_20260702.pt"
        obj = torch.load(path, map_location=m.DEVICE, weights_only=False)
        model.load_state_dict(obj["model_state"])
        models.append(model)
    return models


def predict_prob(models, xs):
    probs = []
    for model in models:
        model.eval(); chunks = []
        with torch.no_grad():
            for start in range(0, len(xs), m.BATCH):
                xb = torch.tensor(xs[start:start + m.BATCH], dtype=torch.float32).to(m.DEVICE)
                chunks.append(torch.softmax(model(xb), dim=1).cpu().numpy())
        probs.append(np.vstack(chunks))
    return np.mean(np.stack(probs, axis=0), axis=0)


def eval_case(lib, k):
    xs, y_true, y_map, ll = m.make_dataset(lib, N_EVAL, 940000 + k, k, return_ll=True)
    p = predict_prob(load_models(lib, k), xs)
    direct = np.argmax(p, axis=1)
    conf = np.max(p, axis=1)
    cand = np.argsort(-p, axis=1)[:, :20]
    rerank = np.array([cand[i, int(np.argmax(ll[i, cand[i]]))] for i in range(len(y_true))], dtype=np.int64)
    truth_in = np.array([y_true[i] in cand[i] for i in range(len(y_true))], dtype=bool)
    order_high = np.argsort(-conf)
    rows = []
    for frac in KEEP_DIRECT_FRACS:
        n_direct = int(round(frac * len(y_true)))
        direct_mask = np.zeros(len(y_true), dtype=bool)
        if n_direct > 0:
            direct_mask[order_high[:n_direct]] = True
        pred = rerank.copy()
        pred[direct_mask] = direct[direct_mask]
        low_mask = ~direct_mask
        rows.append({
            "direct_frac": frac,
            "direct_n": int(np.sum(direct_mask)),
            "rerank_n": int(np.sum(low_mask)),
            "direct_subset_acc": float(np.mean(direct[direct_mask] == y_true[direct_mask])) if np.any(direct_mask) else float("nan"),
            "direct_subset_map_agree": float(np.mean(direct[direct_mask] == y_map[direct_mask])) if np.any(direct_mask) else float("nan"),
            "direct_subset_mean_conf": float(np.mean(conf[direct_mask])) if np.any(direct_mask) else float("nan"),
            "low_subset_rerank_acc": float(np.mean(rerank[low_mask] == y_true[low_mask])) if np.any(low_mask) else float("nan"),
            "low_subset_truth_top20": float(np.mean(truth_in[low_mask])) if np.any(low_mask) else float("nan"),
            "combined_acc": float(np.mean(pred == y_true)),
            "combined_map_agree": float(np.mean(pred == y_map)),
        })
    summary = {
        "same_draw_exact_map_top1": float(np.mean(y_map == y_true)),
        "direct_all_acc": float(np.mean(direct == y_true)),
        "direct_all_map_agree": float(np.mean(direct == y_map)),
        "full_rerank20_acc": float(np.mean(rerank == y_true)),
        "full_rerank20_map_agree": float(np.mean(rerank == y_map)),
        "truth_in_top20": float(np.mean(truth_in)),
        "mean_conf": float(np.mean(conf)),
        "p90_conf": float(np.quantile(conf, 0.90)),
        "p75_conf": float(np.quantile(conf, 0.75)),
        "p50_conf": float(np.quantile(conf, 0.50)),
    }
    return summary, rows


def main():
    lib = m.base.load_300_lib()
    lines = []
    lines.append("300-bus confidence-stratified direct/rerank policy audit")
    lines.append("date=2026-07-02")
    lines.append("role=online operating-policy audit; high-confidence direct NRE, low-confidence top-20 exact rerank; no training")
    lines.append(f"device={m.DEVICE}")
    lines.append(f"eval_samples_per_K={N_EVAL}")
    for k in [280, 299]:
        summary, rows = eval_case(lib, k)
        lines.append(f"K={k}")
        lines.append(f"observed_after_missing={k - int(k * m.MISS_RATE)}")
        for key, val in summary.items():
            lines.append(f"{key}={val:.4f}")
        lines.append("direct_frac,direct_n,rerank_n,direct_subset_acc,direct_subset_map_agree,direct_subset_mean_conf,low_subset_rerank_acc,low_subset_truth_top20,combined_acc,combined_map_agree")
        for r in rows:
            lines.append(
                f"{r['direct_frac']:.2f},{r['direct_n']},{r['rerank_n']},{r['direct_subset_acc']:.4f},"
                f"{r['direct_subset_map_agree']:.4f},{r['direct_subset_mean_conf']:.6f},{r['low_subset_rerank_acc']:.4f},"
                f"{r['low_subset_truth_top20']:.4f},{r['combined_acc']:.4f},{r['combined_map_agree']:.4f}"
            )
    text = "\n".join(lines) + "\n"
    out = ROOT / OUT_NAME
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved: {out}")
    if PKG_STATS.exists():
        (PKG_STATS / OUT_NAME).write_text(text, encoding="utf-8")
        print(f"Saved: {PKG_STATS / OUT_NAME}")
    if PKG_CODE.exists():
        (PKG_CODE / Path(__file__).name).write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Saved: {PKG_CODE / Path(__file__).name}")


if __name__ == "__main__":
    main()
