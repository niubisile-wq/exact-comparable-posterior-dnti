# -*- coding: utf-8 -*-
"""
300-bus selective rerank curve.

Evaluates whether exact/high-fidelity reranking can be reserved for low-confidence
NRE samples instead of being applied to every sample. No training and no
checkpoint writes are performed.
"""

import os
from pathlib import Path
import numpy as np
import torch

import dn_300bus_redundant_sensing_attack_20260702 as m

ROOT = Path(__file__).resolve().parent
PKG_ROOT = Path.home() / "Desktop" / "配电网论文一区投稿成果包_20260702"
PKG_STATS = PKG_ROOT / "03_frozen_tables_stats"
PKG_CODE = PKG_ROOT / "02_code"
OUT_NAME = "300bus_selective_rerank_curve_20260702.txt"
SEEDS = [42, 123, 456]
N_EVAL = 10000
CALL_RATES = [0.0, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]


def load_model(lib, k, seed):
    n_topos, _, n_bus = lib["V"].shape
    model = m.base.NRE300(n_topos, n_bus).to(m.DEVICE)
    path = ROOT / f"nre_300bus_ipc_K{k}_miss30_redundant_seed{seed}_20260702.pt"
    obj = torch.load(path, map_location=m.DEVICE, weights_only=False)
    model.load_state_dict(obj["model_state"])
    return model


def softmax_np(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    p = np.exp(z)
    return p / np.sum(p, axis=1, keepdims=True)


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


def top20_rerank_pred(p, ll):
    cand = np.argsort(-p, axis=1)[:, :20]
    pred = np.empty(len(p), dtype=np.int64)
    for i in range(len(p)):
        ci = cand[i]
        pred[i] = ci[int(np.argmax(ll[i, ci]))]
    return pred, cand


def eval_k(lib, k):
    old_k = m.K_ATTACK
    m.K_ATTACK = k
    xs, y_true, y_map, ll = m.make_dataset(lib, N_EVAL, 930000 + k, k, return_ll=True)
    models = [load_model(lib, k, seed) for seed in SEEDS]
    p = predict_prob(models, xs)
    direct = np.argmax(p, axis=1)
    conf = np.max(p, axis=1)
    rr20, cand = top20_rerank_pred(p, ll)
    truth_in_top20 = np.array([y_true[i] in cand[i] for i in range(len(y_true))])
    order_low_conf = np.argsort(conf)
    rows = []
    for rate in CALL_RATES:
        n_call = int(round(rate * len(y_true)))
        use_rerank = np.zeros(len(y_true), dtype=bool)
        if n_call > 0:
            use_rerank[order_low_conf[:n_call]] = True
        pred = direct.copy()
        pred[use_rerank] = rr20[use_rerank]
        rows.append({
            "call_rate": rate,
            "called": int(n_call),
            "true_acc": float(np.mean(pred == y_true)),
            "map_agree": float(np.mean(pred == y_map)),
            "called_truth_top20": float(np.mean(truth_in_top20[use_rerank])) if n_call > 0 else float("nan"),
            "avg_conf_called": float(np.mean(conf[use_rerank])) if n_call > 0 else float("nan"),
            "avg_conf_not_called": float(np.mean(conf[~use_rerank])) if n_call < len(y_true) else float("nan"),
        })
    summary = {
        "K": k,
        "observed_after_missing": int(k - int(k * m.MISS_RATE)),
        "same_draw_exact_map_top1": float(np.mean(y_map == y_true)),
        "direct_true_acc": float(np.mean(direct == y_true)),
        "direct_map_agree": float(np.mean(direct == y_map)),
        "truth_in_top20": float(np.mean(truth_in_top20)),
        "full_rerank20_true_acc": float(np.mean(rr20 == y_true)),
        "full_rerank20_map_agree": float(np.mean(rr20 == y_map)),
    }
    m.K_ATTACK = old_k
    return summary, rows


def main():
    lib = m.base.load_300_lib()
    lines = []
    lines.append("300-bus selective low-confidence exact-rerank curve")
    lines.append("date=2026-07-02")
    lines.append("role=operational fallback audit; no training; no checkpoint writes")
    lines.append(f"device={m.DEVICE}")
    lines.append(f"eval_samples_per_K={N_EVAL}")
    lines.append("policy=rank samples by ensemble max posterior confidence; apply top-20 exact rerank only to lowest-confidence fraction")
    for k in [280, 299]:
        summary, rows = eval_k(lib, k)
        lines.append(f"K={k}")
        for key, val in summary.items():
            lines.append(f"{key}={val:.4f}" if isinstance(val, float) else f"{key}={val}")
        lines.append("call_rate,called,true_acc,map_agree,called_truth_top20,avg_conf_called,avg_conf_not_called")
        for r in rows:
            lines.append(
                f"{r['call_rate']:.2f},{r['called']},{r['true_acc']:.4f},{r['map_agree']:.4f},"
                f"{r['called_truth_top20']:.4f},{r['avg_conf_called']:.6f},{r['avg_conf_not_called']:.6f}"
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
