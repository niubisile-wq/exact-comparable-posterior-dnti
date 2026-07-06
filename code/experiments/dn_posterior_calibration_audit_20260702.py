# -*- coding: utf-8 -*-
"""
Posterior calibration audit for new counterattack models.

Post-hoc temperature/uniform-mix calibration on independent validation draws,
then reports posterior quality on independent test draws for 119-bus K60 and
300-bus K299. No model training and no checkpoint writes are performed.
"""

from pathlib import Path
import numpy as np
import torch

import dn_119bus_sensor_policy_v2_train_20260702 as m119
import dn_300bus_redundant_sensing_attack_20260702 as m300

ROOT = Path(__file__).resolve().parent
PKG_ROOT = Path.home() / "Desktop" / "配电网论文一区投稿成果包_20260702"
PKG_STATS = PKG_ROOT / "03_frozen_tables_stats"
PKG_CODE = PKG_ROOT / "02_code"
OUT_NAME = "posterior_calibration_audit_20260702.txt"
SEEDS = [42, 123, 456]
N_VAL = 5000
N_TEST = 7000
SENSORS_119_K60 = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118], dtype=np.int64)


def softmax_np(z):
    z = z - np.max(z, axis=1, keepdims=True)
    p = np.exp(z)
    return p / np.sum(p, axis=1, keepdims=True)


def ll_to_q(ll):
    return softmax_np(ll.astype(np.float64)).astype(np.float64)


def mean_logits_119(xs):
    lib = m119.base.load_119_lib()
    outs = []
    for seed in SEEDS:
        model = m119.load_old_model(seed, lib)
        ckpt = ROOT / f"nre_119bus_ip1_K60_sensor_v2_seed{seed}_20260702.pt"
        obj = torch.load(ckpt, map_location=m119.DEVICE, weights_only=False)
        model.load_state_dict(obj["model_state"])
        outs.append(m119.predict_logits(model, xs))
    return np.mean(np.stack(outs, axis=0), axis=0)


def mean_logits_300(xs):
    lib = m300.base.load_300_lib()
    n_topos, _, n_bus = lib["V"].shape
    outs = []
    for seed in SEEDS:
        model = m300.base.NRE300(n_topos, n_bus).to(m300.DEVICE)
        ckpt = ROOT / f"nre_300bus_ipc_K299_miss30_redundant_seed{seed}_20260702.pt"
        obj = torch.load(ckpt, map_location=m300.DEVICE, weights_only=False)
        model.load_state_dict(obj["model_state"])
        outs.append(m300.predict_logits(model, xs))
    return np.mean(np.stack(outs, axis=0), axis=0)


def apply_cal(logits, T, eps):
    p = softmax_np(logits / T)
    c = p.shape[1]
    return (1.0 - eps) * p + eps / c


def posterior_metrics(p, q, y):
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    q = q / np.sum(q, axis=1, keepdims=True)
    pred = np.argmax(p, axis=1)
    conf = np.max(p, axis=1)
    correct = (pred == y).astype(float)
    kl = float(np.mean(np.sum(q * (np.log(q) - np.log(p)), axis=1)))
    ce_ref = float(np.mean(-np.sum(q * np.log(p), axis=1)))
    nll_true = float(np.mean(-np.log(p[np.arange(len(y)), y])))
    brier = float(np.mean(np.sum((p - np.eye(p.shape[1])[y]) ** 2, axis=1)))
    ece = 0.0
    for lo in np.linspace(0, 0.9, 10):
        hi = lo + 0.1
        mask = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if np.any(mask):
            ece += np.mean(mask) * abs(float(np.mean(conf[mask])) - float(np.mean(correct[mask])))
    out = {
        "top1": float(np.mean(pred == y)),
        "kl_ref_to_p": kl,
        "cross_entropy_ref": ce_ref,
        "nll_true": nll_true,
        "brier": brier,
        "ece": float(ece),
        "mean_conf": float(np.mean(conf)),
    }
    order = np.argsort(-p, axis=1)
    csum = np.take_along_axis(p, order, axis=1).cumsum(axis=1)
    for level in [0.50, 0.80, 0.90, 0.95]:
        sizes = np.argmax(csum >= level, axis=1) + 1
        hit = []
        for i, s in enumerate(sizes):
            hit.append(y[i] in order[i, :s])
        out[f"cred{int(level*100)}_coverage"] = float(np.mean(hit))
        out[f"cred{int(level*100)}_avg_size"] = float(np.mean(sizes))
    return out


def tune_calibration(logits, q, y):
    best = None
    for T in np.concatenate([np.linspace(0.35, 1.5, 24), np.linspace(1.6, 6.0, 23)]):
        for eps in [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15]:
            p = apply_cal(logits, float(T), float(eps))
            met = posterior_metrics(p, q, y)
            score = met["kl_ref_to_p"]
            if best is None or score < best["score"]:
                best = {"T": float(T), "eps": float(eps), "score": score, "metrics": met}
    return best


def run_119():
    lib = m119.base.load_119_lib()
    m119.SENSORS = SENSORS_119_K60
    xs_val, y_val, _ym_val, ll_val = m119.make_dataset(lib, N_VAL, 119970, return_ll=True)
    xs_test, y_test, _ym_test, ll_test = m119.make_dataset(lib, N_TEST, 119971, return_ll=True)
    q_val = ll_to_q(ll_val)
    q_test = ll_to_q(ll_test)
    val_logits = mean_logits_119(xs_val)
    test_logits = mean_logits_119(xs_test)
    raw_val = posterior_metrics(softmax_np(val_logits), q_val, y_val)
    best = tune_calibration(val_logits, q_val, y_val)
    raw_test = posterior_metrics(softmax_np(test_logits), q_test, y_test)
    cal_test = posterior_metrics(apply_cal(test_logits, best["T"], best["eps"]), q_test, y_test)
    return "119bus_K60", raw_val, best, raw_test, cal_test


def run_300():
    lib = m300.base.load_300_lib()
    xs_val, y_val, _ym_val, ll_val = m300.make_dataset(lib, N_VAL, 300970, 299, return_ll=True)
    xs_test, y_test, _ym_test, ll_test = m300.make_dataset(lib, N_TEST, 300971, 299, return_ll=True)
    q_val = ll_to_q(ll_val)
    q_test = ll_to_q(ll_test)
    val_logits = mean_logits_300(xs_val)
    test_logits = mean_logits_300(xs_test)
    raw_val = posterior_metrics(softmax_np(val_logits), q_val, y_val)
    best = tune_calibration(val_logits, q_val, y_val)
    raw_test = posterior_metrics(softmax_np(test_logits), q_test, y_test)
    cal_test = posterior_metrics(apply_cal(test_logits, best["T"], best["eps"]), q_test, y_test)
    return "300bus_K299_miss30", raw_val, best, raw_test, cal_test


def emit_case(lines, name, raw_val, best, raw_test, cal_test):
    lines.append(f"case={name}")
    lines.append(f"validation_selected_T={best['T']:.4f}")
    lines.append(f"validation_selected_uniform_eps={best['eps']:.4f}")
    lines.append(f"validation_raw_kl={raw_val['kl_ref_to_p']:.6g}")
    lines.append(f"validation_calibrated_kl={best['metrics']['kl_ref_to_p']:.6g}")
    lines.append("metric,raw_test,calibrated_test,delta_cal_minus_raw")
    keys = ["top1", "kl_ref_to_p", "cross_entropy_ref", "nll_true", "brier", "ece", "mean_conf", "cred50_coverage", "cred50_avg_size", "cred80_coverage", "cred80_avg_size", "cred90_coverage", "cred90_avg_size", "cred95_coverage", "cred95_avg_size"]
    for key in keys:
        raw = raw_test[key]
        cal = cal_test[key]
        lines.append(f"{key},{raw:.6g},{cal:.6g},{cal-raw:.6g}")


def main():
    lines = []
    lines.append("Posterior calibration audit for counterattack models")
    lines.append("date=2026-07-02")
    lines.append("role=post-hoc temperature/uniform-mix calibration against exact posterior references; no retraining")
    lines.append(f"validation_samples={N_VAL}")
    lines.append(f"test_samples={N_TEST}")
    for result in [run_119(), run_300()]:
        emit_case(lines, *result)
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
