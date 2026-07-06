# -*- coding: utf-8 -*-
"""
Large-system sensor-policy scan.

Purpose:
  Test whether the 119/300 large-system weakness can be repaired by improving
  the measurement design rather than repeatedly fine-tuning the same weak
  deployment.

Protocol:
  - no checkpoint is overwritten;
  - the topology libraries, noise levels, K values, and 300-bus 30% missing
    condition are unchanged;
  - sensor policy is chosen from voltage-library topology variance only;
  - final evaluation uses fresh simulated draws.
"""

import os
import time

import numpy as np
import torch

import dn_large_system_candidate_rerank_20260702 as base


ROOT = os.path.dirname(os.path.abspath(__file__))
PKG_STATS = r"<REPOSITORY_ROOT>\03_frozen_tables_stats"
PKG_CODE = r"<REPOSITORY_ROOT>\02_code"
DEVICE = base.DEVICE
OUT_NAME = "large_system_sensor_policy_scan_20260702.txt"


def discriminative_sensors(V, k, exclude_slack=True):
    # Score buses by expected topology variance over the load-factor grid.
    score = np.mean(np.var(V, axis=0), axis=0)
    if exclude_slack and len(score) > 0:
        score[0] = -np.inf
    return np.sort(np.argsort(-score)[:k]).astype(np.int64), score


def softmax_np(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    p = np.exp(z)
    return p / np.sum(p, axis=1, keepdims=True)


def exact_ll(V, sigma, buses_list, obs_list, lf_idx_list):
    n = len(obs_list)
    ll = np.empty((n, V.shape[0]), dtype=np.float32)
    for i in range(n):
        buses = buses_list[i]
        obs = obs_list[i]
        pred = V[:, lf_idx_list[i], :][:, buses]
        ll[i] = (-0.5 * np.sum(((pred - obs[None, :]) / sigma) ** 2, axis=1)).astype(np.float32)
    return ll


def rerank_metrics(p, y, ll, m=20):
    exact_pred = np.argmax(ll, axis=1)
    cand = np.argsort(-p, axis=1)[:, :m]
    truth_in = np.array([y[i] in cand[i] for i in range(len(y))], dtype=bool)
    exact_in = np.array([exact_pred[i] in cand[i] for i in range(len(y))], dtype=bool)
    pred = np.empty(len(y), dtype=np.int64)
    for i in range(len(y)):
        ci = cand[i]
        pred[i] = ci[int(np.argmax(ll[i, ci]))]
    return {
        "truth_in_top20": float(np.mean(truth_in)),
        "exact_in_top20": float(np.mean(exact_in)),
        "rerank20": float(np.mean(pred == y)),
        "rerank20_vs_exact": float(np.mean(pred == exact_pred)),
    }


def load_119_logits(xs):
    lib = base.load_119_lib()
    n_topos, _, n_bus = lib["V"].shape
    seeds = [42, 123, 456, 789, 2024]
    x_t = torch.tensor(xs, dtype=torch.float32).to(DEVICE)
    logits = []
    for seed in seeds:
        model = base.LoadAwareNRE119(n_topos, n_bus).to(DEVICE)
        ckpt = torch.load(os.path.join(ROOT, f"nre_119bus_ip1_seed{seed}.pt"), map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            logits.append(model(x_t).detach().cpu().numpy())
    return logits


def load_300_logits(xs):
    lib = base.load_300_lib()
    n_topos, _, n_bus = lib["V"].shape
    seeds = [42, 123, 456]
    x_t = torch.tensor(xs, dtype=torch.float32).to(DEVICE)
    logits = []
    for seed in seeds:
        model = base.NRE300(n_topos, n_bus).to(DEVICE)
        ckpt = torch.load(os.path.join(ROOT, f"nre_300bus_ipc_seed{seed}.pt"), map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            logits.append(model(x_t).detach().cpu().numpy())
    return logits


def ensemble_prob(logits_list):
    return np.mean(np.stack([softmax_np(z) for z in logits_list], axis=0), axis=0)


def make_119(lib, n, seed, policy, sensors=None):
    rng = np.random.RandomState(seed)
    V, lf_grid, base_p = lib["V"], lib["lf"], lib["base_p"]
    n_topos, n_lf, n_bus = V.shape
    sigma = 0.009
    xs = np.zeros((n, n_bus * 3), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    buses_list, obs_list, lf_idx_list = [], [], []
    for i in range(n):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        lf = float(lf_grid[lf_idx])
        if policy == "random":
            buses = np.sort(rng.choice(range(1, n_bus), 25, replace=False))
        elif policy == "discriminative_fixed":
            buses = sensors
        else:
            raise ValueError(policy)
        obs = V[ti, lf_idx, buses] + rng.normal(0.0, sigma, size=len(buses))
        xs[i, buses] = obs
        xs[i, n_bus + buses] = 1.0
        xs[i, 2 * n_bus + buses] = base_p[buses] * lf
        y[i] = ti
        buses_list.append(buses)
        obs_list.append(obs.astype(np.float32))
        lf_idx_list.append(lf_idx)
    ll = exact_ll(V, sigma, buses_list, obs_list, lf_idx_list)
    return xs, y, ll


def make_300(lib, n, seed, policy, sensors=None):
    rng = np.random.RandomState(seed)
    V, lf_grid, base_p = lib["V"], lib["lf"], lib["base_p"]
    n_topos, n_lf, n_bus = V.shape
    sigma = 0.0015
    k = 150
    n_miss = int(k * 0.30)
    if policy == "uniform_fixed":
        full = base.deployment_sensors(n_bus, k)
    elif policy == "discriminative_fixed":
        full = sensors
    else:
        raise ValueError(policy)
    xs = np.zeros((n, n_bus * 3), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    buses_list, obs_list, lf_idx_list = [], [], []
    for i in range(n):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        lf = float(lf_grid[lf_idx])
        buses = np.delete(full, rng.choice(len(full), n_miss, replace=False))
        obs = V[ti, lf_idx, buses] + rng.normal(0.0, sigma, size=len(buses))
        xs[i, buses] = (obs - 1.0) / sigma
        xs[i, n_bus + buses] = 1.0
        xs[i, 2 * n_bus + buses] = base_p[buses] * lf
        y[i] = ti
        buses_list.append(buses)
        obs_list.append(obs.astype(np.float32))
        lf_idx_list.append(lf_idx)
    ll = exact_ll(V, sigma, buses_list, obs_list, lf_idx_list)
    return xs, y, ll


def summarize(system, policy, xs, y, ll, logits_list):
    p = ensemble_prob(logits_list)
    exact_pred = np.argmax(ll, axis=1)
    nre_pred = np.argmax(p, axis=1)
    rr = rerank_metrics(p, y, ll, m=20)
    return {
        "system": system,
        "policy": policy,
        "n": len(y),
        "exact_top1": float(np.mean(exact_pred == y)),
        "nre_ensemble_top1": float(np.mean(nre_pred == y)),
        "nre_exact_agree": float(np.mean(nre_pred == exact_pred)),
        **rr,
    }


def main():
    t0 = time.time()
    lines = []
    lines.append("Large-system sensor-policy scan")
    lines.append("date=2026-07-02")
    lines.append(f"device={DEVICE}")
    lines.append("sensor_policy=topology-variance discriminative sensors selected from voltage library only")
    lines.append("")

    lib119 = base.load_119_lib()
    s119, score119 = discriminative_sensors(lib119["V"], 25)
    lines.append("119_discriminative_sensors=" + " ".join(map(str, s119.tolist())))
    xs, y, ll = make_119(lib119, 3000, 619119, "random")
    rows = [summarize("119bus_IP1", "random25", xs, y, ll, load_119_logits(xs))]
    xs, y, ll = make_119(lib119, 3000, 619120, "discriminative_fixed", s119)
    rows.append(summarize("119bus_IP1", "variance_top25_fixed", xs, y, ll, load_119_logits(xs)))

    lib300 = base.load_300_lib()
    s300, score300 = discriminative_sensors(lib300["V"], 150)
    lines.append("300_discriminative_sensors=" + " ".join(map(str, s300.tolist())))
    xs, y, ll = make_300(lib300, 4000, 630300, "uniform_fixed")
    rows.append(summarize("300bus_IPC_miss30", "uniform150_miss30", xs, y, ll, load_300_logits(xs)))
    xs, y, ll = make_300(lib300, 4000, 630301, "discriminative_fixed", s300)
    rows.append(summarize("300bus_IPC_miss30", "variance_top150_miss30", xs, y, ll, load_300_logits(xs)))

    lines.append("")
    lines.append("system,policy,n,exact_top1,nre_ensemble_top1,nre_exact_agree,truth_in_top20,exact_in_top20,rerank20,rerank20_vs_exact")
    for r in rows:
        lines.append(
            f"{r['system']},{r['policy']},{r['n']},{r['exact_top1']:.4f},"
            f"{r['nre_ensemble_top1']:.4f},{r['nre_exact_agree']:.4f},"
            f"{r['truth_in_top20']:.4f},{r['exact_in_top20']:.4f},"
            f"{r['rerank20']:.4f},{r['rerank20_vs_exact']:.4f}"
        )

    by_key = {(r["system"], r["policy"]): r for r in rows}
    lines.append("")
    lines.append("deltas")
    lines.append(
        "119_exact_delta={:.4f}".format(
            by_key[("119bus_IP1", "variance_top25_fixed")]["exact_top1"]
            - by_key[("119bus_IP1", "random25")]["exact_top1"]
        )
    )
    lines.append(
        "119_nre_delta={:.4f}".format(
            by_key[("119bus_IP1", "variance_top25_fixed")]["nre_ensemble_top1"]
            - by_key[("119bus_IP1", "random25")]["nre_ensemble_top1"]
        )
    )
    lines.append(
        "300_exact_delta={:.4f}".format(
            by_key[("300bus_IPC_miss30", "variance_top150_miss30")]["exact_top1"]
            - by_key[("300bus_IPC_miss30", "uniform150_miss30")]["exact_top1"]
        )
    )
    lines.append(
        "300_nre_delta={:.4f}".format(
            by_key[("300bus_IPC_miss30", "variance_top150_miss30")]["nre_ensemble_top1"]
            - by_key[("300bus_IPC_miss30", "uniform150_miss30")]["nre_ensemble_top1"]
        )
    )
    lines.append(f"elapsed_sec={time.time() - t0:.1f}")
    text = "\n".join(lines) + "\n"
    local_out = os.path.join(ROOT, OUT_NAME)
    with open(local_out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"Saved: {local_out}")
    if os.path.isdir(PKG_STATS):
        pkg_out = os.path.join(PKG_STATS, OUT_NAME)
        with open(pkg_out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved: {pkg_out}")
    if os.path.isdir(PKG_CODE):
        code_out = os.path.join(PKG_CODE, os.path.basename(__file__))
        with open(__file__, "r", encoding="utf-8") as f:
            code = f.read()
        with open(code_out, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"Saved: {code_out}")


if __name__ == "__main__":
    main()
