# -*- coding: utf-8 -*-
"""
Topology-conditioned measurement residual audit.

Reports state-estimation-style residual consistency for direct NRE predictions,
NRE top-20 exact residual reranking, exact MAP, and the true sampled topology.
No training and no checkpoint writes are performed.
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
OUT_NAME = "topology_conditioned_residual_audit_20260702.txt"
SEEDS = [42, 123, 456]
N_EVAL_119 = 9000
N_EVAL_300 = 9000

SENSORS_119_K60 = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118], dtype=np.int64)


def softmax_np(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    p = np.exp(z)
    return p / np.sum(p, axis=1, keepdims=True)


def ll_to_rmse(ll, sigma, k_obs):
    return sigma * np.sqrt(np.maximum(-2.0 * ll / max(k_obs, 1), 0.0))


def residual_summary(name, y_true, y_map, ll, p, sigma, k_obs):
    direct = np.argmax(p, axis=1)
    cand = np.argsort(-p, axis=1)[:, :20]
    rerank = np.array([cand[i, int(np.argmax(ll[i, cand[i]]))] for i in range(len(y_true))], dtype=np.int64)
    true_rmse = ll_to_rmse(ll[np.arange(len(y_true)), y_true], sigma, k_obs)
    direct_rmse = ll_to_rmse(ll[np.arange(len(y_true)), direct], sigma, k_obs)
    rerank_rmse = ll_to_rmse(ll[np.arange(len(y_true)), rerank], sigma, k_obs)
    map_rmse = ll_to_rmse(ll[np.arange(len(y_true)), y_map], sigma, k_obs)
    truth_in_top20 = np.array([y_true[i] in cand[i] for i in range(len(y_true))])
    map_in_top20 = np.array([y_map[i] in cand[i] for i in range(len(y_true))])
    out = []
    out.append(f"case={name}")
    out.append(f"samples={len(y_true)}")
    out.append(f"k_observed={k_obs}")
    out.append(f"direct_top1={np.mean(direct == y_true):.4f}")
    out.append(f"direct_map_agree={np.mean(direct == y_map):.4f}")
    out.append(f"truth_in_top20={np.mean(truth_in_top20):.4f}")
    out.append(f"map_in_top20={np.mean(map_in_top20):.4f}")
    out.append(f"rerank20_top1={np.mean(rerank == y_true):.4f}")
    out.append(f"rerank20_map_agree={np.mean(rerank == y_map):.4f}")
    out.append(f"exact_map_top1={np.mean(y_map == y_true):.4f}")
    out.append("residual_rmse_pu_mean_median_p90")
    for label, arr in [("true_topology", true_rmse), ("direct_nre", direct_rmse), ("rerank20", rerank_rmse), ("exact_map", map_rmse)]:
        out.append(f"{label},{np.mean(arr):.6g},{np.median(arr):.6g},{np.quantile(arr, 0.90):.6g}")
    out.append(f"direct_to_rerank_mean_residual_reduction={(np.mean(direct_rmse)-np.mean(rerank_rmse)):.6g}")
    out.append(f"direct_to_exact_map_mean_residual_reduction={(np.mean(direct_rmse)-np.mean(map_rmse)):.6g}")
    out.append(f"rerank_to_exact_map_mean_residual_gap={(np.mean(rerank_rmse)-np.mean(map_rmse)):.6g}")
    return out


def eval_119_k60():
    lib = m119.base.load_119_lib()
    m119.SENSORS = SENSORS_119_K60
    xs, y_true, y_map, ll = m119.make_dataset(lib, N_EVAL_119, 119960, return_ll=True)
    logits = []
    for seed in SEEDS:
        model = m119.load_old_model(seed, lib)
        ckpt = ROOT / f"nre_119bus_ip1_K60_sensor_v2_seed{seed}_20260702.pt"
        obj = torch.load(ckpt, map_location=m119.DEVICE, weights_only=False)
        model.load_state_dict(obj["model_state"])
        logits.append(m119.predict_logits(model, xs))
    p = np.mean(np.stack([softmax_np(z) for z in logits], axis=0), axis=0)
    return residual_summary("119bus_K60_sensor_policy", y_true, y_map, ll, p, m119.SIGMA, len(SENSORS_119_K60))


def eval_300_k299():
    lib = m300.base.load_300_lib()
    old_k = m300.K_ATTACK
    m300.K_ATTACK = 299
    xs, y_true, y_map, ll = m300.make_dataset(lib, N_EVAL_300, 300960, 299, return_ll=True)
    logits = []
    n_topos, _, n_bus = lib["V"].shape
    for seed in SEEDS:
        model = m300.base.NRE300(n_topos, n_bus).to(m300.DEVICE)
        ckpt = ROOT / f"nre_300bus_ipc_K299_miss30_redundant_seed{seed}_20260702.pt"
        obj = torch.load(ckpt, map_location=m300.DEVICE, weights_only=False)
        model.load_state_dict(obj["model_state"])
        logits.append(m300.predict_logits(model, xs))
    p = np.mean(np.stack([softmax_np(z) for z in logits], axis=0), axis=0)
    k_obs = 299 - int(299 * m300.MISS_RATE)
    m300.K_ATTACK = old_k
    return residual_summary("300bus_K299_miss30", y_true, y_map, ll, p, m300.SIGMA, k_obs)


def main():
    lines = []
    lines.append("Topology-conditioned measurement residual audit")
    lines.append("date=2026-07-02")
    lines.append("role=state-estimation-style measurement consistency audit for posterior candidates; no training")
    lines.extend(eval_119_k60())
    lines.extend(eval_300_k299())
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
