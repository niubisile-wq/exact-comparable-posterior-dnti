# -*- coding: utf-8 -*-
"""
Bootstrap confidence intervals for the new counterattack experiments.

Covers 119-bus K60, 300-bus K299, IEEE123 K75, and SoCal enhanced real-
measurement posterior audit. This script recomputes per-sample predictions and
bootstraps accuracy/coverage statistics. No model training except the SoCal
blocked-CV lightweight classifier used by that audit.
"""

from pathlib import Path
import numpy as np
import torch

import dn_119bus_sensor_policy_v2_train_20260702 as m119
import dn_300bus_redundant_sensing_attack_20260702 as m300
import dn_ieee123_3ph_K75_warmstart_20260702 as m123
import dn_socal_measurement_conditioned_enhanced_posterior_20260702 as msocal

ROOT = Path(__file__).resolve().parent
PKG_ROOT = Path.home() / "Desktop" / "配电网论文一区投稿成果包_20260702"
PKG_STATS = PKG_ROOT / "03_frozen_tables_stats"
PKG_CODE = PKG_ROOT / "02_code"
OUT_NAME = "counterattack_bootstrap_ci_20260702.txt"
BOOT = 1200
RNG = np.random.RandomState(20260702)
SEEDS_119 = [42, 123, 456, 789, 2024]
SEEDS_300 = [42, 123, 456]
SEEDS_123 = [42, 123, 456]
SENSORS_119_K60 = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118], dtype=np.int64)


def softmax_np(z):
    z = z - np.max(z, axis=1, keepdims=True)
    p = np.exp(z)
    return p / np.sum(p, axis=1, keepdims=True)


def ci_binary(arr):
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    vals = np.empty(BOOT, dtype=np.float64)
    for b in range(BOOT):
        idx = RNG.randint(0, n, size=n)
        vals[b] = np.mean(arr[idx])
    return float(np.mean(arr)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def emit_ci(lines, case, metric, arr):
    mean, lo, hi = ci_binary(arr)
    lines.append(f"{case},{metric},{len(arr)},{mean:.6f},{lo:.6f},{hi:.6f}")


def eval_119(lines):
    lib = m119.base.load_119_lib()
    m119.SENSORS = SENSORS_119_K60
    xs, y, y_map, ll = m119.make_dataset(lib, 9000, 119982, return_ll=True)
    logits = []
    for seed in SEEDS_119:
        model = m119.load_old_model(seed, lib)
        ckpt = ROOT / f"nre_119bus_ip1_K60_sensor_v2_seed{seed}_20260702.pt"
        obj = torch.load(ckpt, map_location=m119.DEVICE, weights_only=False)
        model.load_state_dict(obj["model_state"])
        logits.append(m119.predict_logits(model, xs))
    p = np.mean(np.stack([softmax_np(z) for z in logits], axis=0), axis=0)
    direct = np.argmax(p, axis=1)
    cand = np.argsort(-p, axis=1)[:, :20]
    rerank = np.array([cand[i, int(np.argmax(ll[i, cand[i]]))] for i in range(len(y))], dtype=np.int64)
    emit_ci(lines, "119bus_K60", "direct_top1", direct == y)
    emit_ci(lines, "119bus_K60", "exact_map_top1", y_map == y)
    emit_ci(lines, "119bus_K60", "truth_in_top20", np.array([y[i] in cand[i] for i in range(len(y))]))
    emit_ci(lines, "119bus_K60", "rerank20_top1", rerank == y)
    emit_ci(lines, "119bus_K60", "direct_minus_old25_nre_pp", (direct == y).astype(float) - 0.5345)


def eval_300(lines):
    lib = m300.base.load_300_lib()
    xs, y, y_map, ll = m300.make_dataset(lib, 10000, 300982, 299, return_ll=True)
    n_topos, _, n_bus = lib["V"].shape
    logits = []
    for seed in SEEDS_300:
        model = m300.base.NRE300(n_topos, n_bus).to(m300.DEVICE)
        ckpt = ROOT / f"nre_300bus_ipc_K299_miss30_redundant_seed{seed}_20260702.pt"
        obj = torch.load(ckpt, map_location=m300.DEVICE, weights_only=False)
        model.load_state_dict(obj["model_state"])
        logits.append(m300.predict_logits(model, xs))
    p = np.mean(np.stack([softmax_np(z) for z in logits], axis=0), axis=0)
    direct = np.argmax(p, axis=1)
    conf = np.max(p, axis=1)
    cand = np.argsort(-p, axis=1)[:, :20]
    rerank = np.array([cand[i, int(np.argmax(ll[i, cand[i]]))] for i in range(len(y))], dtype=np.int64)
    order_high = np.argsort(-conf)
    high10 = np.zeros(len(y), dtype=bool)
    high10[order_high[: int(round(0.10 * len(y)))]] = True
    pred_policy = rerank.copy()
    pred_policy[high10] = direct[high10]
    emit_ci(lines, "300bus_K299", "direct_top1", direct == y)
    emit_ci(lines, "300bus_K299", "exact_map_top1", y_map == y)
    emit_ci(lines, "300bus_K299", "truth_in_top20", np.array([y[i] in cand[i] for i in range(len(y))]))
    emit_ci(lines, "300bus_K299", "rerank20_top1", rerank == y)
    emit_ci(lines, "300bus_K299", "highconf10_direct_top1", (direct == y)[high10])
    emit_ci(lines, "300bus_K299", "highconf10_direct_lowconf_rerank_top1", pred_policy == y)


def eval_ieee123(lines):
    lib = m123.load_library()
    xs, y, q = m123.make_dataset(lib, 1500, 975987, 75)
    probs = []
    for seed in SEEDS_123:
        model = m123.Controlled3PhNRE(lib["n_topos"], lib["n_bus"]).to(m123.DEVICE)
        ckpt = ROOT / f"nre_ieee123_3ph_K75_warmstart_seed{seed}_20260702.pt"
        obj = torch.load(ckpt, map_location=m123.DEVICE, weights_only=False)
        model.load_state_dict(obj["model_state"])
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(xs), m123.BATCH):
                xb = torch.tensor(xs[start:start + m123.BATCH], dtype=torch.float32, device=m123.DEVICE)
                chunks.append(torch.softmax(model(xb), dim=1).cpu().numpy())
        probs.append(np.vstack(chunks))
    p = np.mean(np.stack(probs, axis=0), axis=0)
    direct = np.argmax(p, axis=1)
    exact = np.argmax(q, axis=1)
    cand = np.argsort(-p, axis=1)[:, :3]
    rerank = np.array([cand[i, int(np.argmax(q[i, cand[i]]))] for i in range(len(y))], dtype=np.int64)
    emit_ci(lines, "ieee123_K75", "direct_top1", direct == y)
    emit_ci(lines, "ieee123_K75", "exact_top1", exact == y)
    emit_ci(lines, "ieee123_K75", "truth_in_top3", np.array([y[i] in cand[i] for i in range(len(y))]))
    emit_ci(lines, "ieee123_K75", "rerank3_top1", rerank == y)


def eval_socal(lines):
    meas = msocal.load_measurements()
    tables = msocal.load_status_tables()
    df0, uniq = msocal.attach_states(meas, tables)
    df, feature_cols = msocal.add_online_features(df0)
    counts = df["state_id"].value_counts().sort_index().to_dict()
    class_states = [int(k) for k, v in counts.items() if v >= msocal.FOLDS * 3]
    dft = df[df["state_id"].isin(class_states)].copy().reset_index(drop=True)
    probs_by_idx, y_by_idx = {}, {}
    for fold in range(msocal.FOLDS):
        train_idx, test_idx = msocal.blocked_fold_indices(dft, class_states, fold)
        seed_probs, yy = [], None
        for seed in msocal.SEEDS:
            p, y = msocal.train_eval(dft, feature_cols, train_idx, test_idx, class_states, seed + fold * 1000)
            seed_probs.append(p)
            yy = y
        pmean = np.mean(np.stack(seed_probs, axis=0), axis=0)
        for pos, idx in enumerate(test_idx):
            probs_by_idx[int(idx)] = pmean[pos]
            y_by_idx[int(idx)] = int(yy[pos])
    all_idx = sorted(probs_by_idx.keys())
    all_probs = np.stack([probs_by_idx[i] for i in all_idx], axis=0)
    all_y = np.array([y_by_idx[i] for i in all_idx], dtype=np.int64)
    pred = np.argmax(all_probs, axis=1)
    majority = np.zeros_like(all_y)
    emit_ci(lines, "socal_enhanced", "blocked_cv_top1", pred == all_y)
    emit_ci(lines, "socal_enhanced", "majority_baseline_top1", majority == all_y)
    emit_ci(lines, "socal_enhanced", "gain_over_majority_pp", (pred == all_y).astype(float) - (majority == all_y).astype(float))


def main():
    lines = []
    lines.append("Counterattack bootstrap confidence interval audit")
    lines.append("date=2026-07-02")
    lines.append("role=bootstrap CI over recomputed per-sample predictions for new counterattack experiments")
    lines.append(f"bootstrap_replicates={BOOT}")
    lines.append("case,metric,n,mean,ci95_low,ci95_high")
    eval_119(lines)
    eval_300(lines)
    eval_ieee123(lines)
    eval_socal(lines)
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
