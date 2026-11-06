# -*- coding: utf-8 -*-
"""
Posterior-weighted residual consistency audit.

Extends the residual line by quantifying:
1. posterior-expected residual vs direct / rerank / exact MAP
2. whether uncertainty identifies samples where reranking matters most
"""

from pathlib import Path
import sys
import numpy as np
import torch

TEMP_ROOT = Path.home() / "Desktop" / "配电网实验_临时"
sys.path.insert(0, str(TEMP_ROOT))

import dn_119bus_sensor_policy_v2_train_20260702 as m119
import dn_300bus_redundant_sensing_attack_20260702 as m300

ROOT = TEMP_ROOT
PKG_ROOT = Path.home() / "Desktop" / "配电网论文一区投稿成果包_20260702"
PKG_STATS = PKG_ROOT / "03_frozen_tables_stats"
OUT_NAME = "posterior_weighted_residual_consistency_20260703.txt"

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


def entropy(p):
    return -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1)


def summarize_case(name, y_true, y_map, ll, p, sigma, k_obs):
    direct = np.argmax(p, axis=1)
    cand = np.argsort(-p, axis=1)[:, :20]
    rerank = np.array([cand[i, int(np.argmax(ll[i, cand[i]]))] for i in range(len(y_true))], dtype=np.int64)
    rmse_all = ll_to_rmse(ll, sigma, k_obs)
    true_rmse = rmse_all[np.arange(len(y_true)), y_true]
    direct_rmse = rmse_all[np.arange(len(y_true)), direct]
    rerank_rmse = rmse_all[np.arange(len(y_true)), rerank]
    map_rmse = rmse_all[np.arange(len(y_true)), y_map]
    posterior_expected_rmse = np.sum(p * rmse_all, axis=1)
    posterior_ent = entropy(p)
    direct_gap = direct_rmse - rerank_rmse

    q75 = np.quantile(posterior_ent, 0.75)
    hi = posterior_ent >= q75
    lo = posterior_ent < q75

    lines = []
    lines.append(f"case={name}")
    lines.append(f"samples={len(y_true)}")
    lines.append(f"k_observed={k_obs}")
    lines.append(f"direct_top1={np.mean(direct == y_true):.4f}")
    lines.append(f"rerank20_top1={np.mean(rerank == y_true):.4f}")
    lines.append(f"exact_map_top1={np.mean(y_map == y_true):.4f}")
    lines.append("mean_rmse_pu")
    lines.append(f"true_topology={np.mean(true_rmse):.8f}")
    lines.append(f"direct_nre={np.mean(direct_rmse):.8f}")
    lines.append(f"rerank20={np.mean(rerank_rmse):.8f}")
    lines.append(f"exact_map={np.mean(map_rmse):.8f}")
    lines.append(f"posterior_expected={np.mean(posterior_expected_rmse):.8f}")
    lines.append("uncertainty_stratified")
    lines.append(f"entropy_q75={q75:.6f}")
    lines.append(f"high_entropy_frac={np.mean(hi):.4f}")
    lines.append(f"high_entropy_direct_top1={np.mean(direct[hi] == y_true[hi]):.4f}")
    lines.append(f"high_entropy_rerank20_top1={np.mean(rerank[hi] == y_true[hi]):.4f}")
    lines.append(f"high_entropy_mean_direct_rmse={np.mean(direct_rmse[hi]):.8f}")
    lines.append(f"high_entropy_mean_rerank_rmse={np.mean(rerank_rmse[hi]):.8f}")
    lines.append(f"high_entropy_mean_residual_reduction={np.mean(direct_gap[hi]):.8f}")
    lines.append(f"low_entropy_direct_top1={np.mean(direct[lo] == y_true[lo]):.4f}")
    lines.append(f"low_entropy_rerank20_top1={np.mean(rerank[lo] == y_true[lo]):.4f}")
    lines.append(f"low_entropy_mean_direct_rmse={np.mean(direct_rmse[lo]):.8f}")
    lines.append(f"low_entropy_mean_rerank_rmse={np.mean(rerank_rmse[lo]):.8f}")
    lines.append(f"low_entropy_mean_residual_reduction={np.mean(direct_gap[lo]):.8f}")
    lines.append(f"entropy_direct_residual_gap_corr={np.corrcoef(posterior_ent, direct_gap)[0,1]:.6f}")
    lines.append(f"posterior_expected_to_exact_map_gap={np.mean(posterior_expected_rmse) - np.mean(map_rmse):.8f}")
    return lines


def eval_119():
    lib = m119.base.load_119_lib()
    m119.SENSORS = SENSORS_119_K60
    xs, y_true, y_map, ll = m119.make_dataset(lib, N_EVAL_119, 119970, return_ll=True)
    logits = []
    for seed in SEEDS:
        model = m119.load_old_model(seed, lib)
        ckpt = ROOT / f"nre_119bus_ip1_K60_sensor_v2_seed{seed}_20260702.pt"
        obj = torch.load(ckpt, map_location=m119.DEVICE, weights_only=False)
        model.load_state_dict(obj["model_state"])
        logits.append(m119.predict_logits(model, xs))
    p = np.mean(np.stack([softmax_np(z) for z in logits], axis=0), axis=0)
    return summarize_case("119bus_K60_sensor_policy", y_true, y_map, ll, p, m119.SIGMA, len(SENSORS_119_K60))


def eval_300():
    lib = m300.base.load_300_lib()
    old_k = m300.K_ATTACK
    m300.K_ATTACK = 299
    xs, y_true, y_map, ll = m300.make_dataset(lib, N_EVAL_300, 300970, 299, return_ll=True)
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
    return summarize_case("300bus_K299_miss30", y_true, y_map, ll, p, m300.SIGMA, k_obs)


def main():
    lines = []
    lines.append("Posterior-weighted residual consistency audit")
    lines.append("date=2026-07-03")
    lines.append("role=state-estimation-style consistency extension for posterior uncertainty and rerank operating value; no training")
    lines.extend(eval_119())
    lines.extend(eval_300())
    text = "\n".join(lines) + "\n"
    out = PKG_STATS / OUT_NAME
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
