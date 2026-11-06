# -*- coding: utf-8 -*-
"""
Large-system diversity-constrained sensor-policy search.

This is a corrective follow-up to the naive variance scan. The naive top-score
rule clusters sensors and can reduce identifiability. Here, sensor candidates
are selected on a validation draw using exact-reference accuracy, then reported
on an independent final draw.
"""

import os
import time

import numpy as np

import dn_large_system_candidate_rerank_20260702 as base
import dn_large_system_sensor_policy_scan_20260702 as scan


ROOT = os.path.dirname(os.path.abspath(__file__))
PKG_STATS = r"<REPOSITORY_ROOT>\03_frozen_tables_stats"
PKG_CODE = r"<REPOSITORY_ROOT>\02_code"
OUT_NAME = "large_system_sensor_policy_search_20260702.txt"


def bus_scores(V):
    score = np.mean(np.var(V, axis=0), axis=0).astype(np.float64)
    score[0] = 0.0
    score -= score.min()
    score += 1e-12
    return score


def stratified_choice(score, n_bus, k, rng, power=1.0):
    buses = np.arange(1, n_bus)
    bins = np.array_split(buses, k)
    out = []
    for b in bins:
        w = score[b] ** power
        w = w / np.sum(w)
        out.append(int(rng.choice(b, p=w)))
    return np.sort(np.unique(out)).astype(np.int64)


def fill_to_k(s, score, n_bus, k):
    selected = set(int(x) for x in s)
    if len(selected) < k:
        for b in np.argsort(-score):
            if b == 0:
                continue
            selected.add(int(b))
            if len(selected) == k:
                break
    if len(selected) > k:
        ordered = sorted(selected, key=lambda x: score[x], reverse=True)[:k]
        selected = set(ordered)
    return np.sort(np.array(list(selected), dtype=np.int64))


def jitter_uniform_300(score, rng, width=2):
    base_set = base.deployment_sensors(300, 150)
    out = []
    used = set()
    for b in base_set:
        lo = max(1, int(b) - width)
        hi = min(299, int(b) + width)
        cand = np.array([x for x in range(lo, hi + 1) if x not in used], dtype=np.int64)
        if len(cand) == 0:
            cand = np.array([int(b)], dtype=np.int64)
        w = score[cand]
        w = w / np.sum(w)
        x = int(rng.choice(cand, p=w))
        out.append(x)
        used.add(x)
    return fill_to_k(np.array(out, dtype=np.int64), score, 300, 150)


def make_val_119(lib, n, seed):
    rng = np.random.RandomState(seed)
    V = lib["V"]
    n_topos, n_lf, _ = V.shape
    return {
        "ti": rng.randint(0, n_topos, size=n),
        "lf": rng.randint(0, n_lf, size=n),
        "noise": rng.normal(0.0, 0.009, size=(n, 25)).astype(np.float32),
    }


def make_val_300(lib, n, seed):
    rng = np.random.RandomState(seed)
    V = lib["V"]
    n_topos, n_lf, _ = V.shape
    k = 150
    obs_k = k - int(k * 0.30)
    return {
        "ti": rng.randint(0, n_topos, size=n),
        "lf": rng.randint(0, n_lf, size=n),
        "keep": np.array([np.sort(rng.choice(k, obs_k, replace=False)) for _ in range(n)], dtype=np.int64),
        "noise": rng.normal(0.0, 0.0015, size=(n, obs_k)).astype(np.float32),
    }


def exact_acc_119(lib, sensors, val):
    V = lib["V"]
    y = val["ti"]
    pred = []
    for i, ti in enumerate(y):
        lf_idx = val["lf"][i]
        obs = V[ti, lf_idx, sensors] + val["noise"][i]
        ll = -0.5 * np.sum(((V[:, lf_idx, :][:, sensors] - obs[None, :]) / 0.009) ** 2, axis=1)
        pred.append(int(np.argmax(ll)))
    return float(np.mean(np.array(pred, dtype=np.int64) == y))


def exact_acc_300(lib, sensors, val):
    V = lib["V"]
    y = val["ti"]
    pred = []
    sensors = np.asarray(sensors, dtype=np.int64)
    for i, ti in enumerate(y):
        lf_idx = val["lf"][i]
        obs_buses = sensors[val["keep"][i]]
        obs = V[ti, lf_idx, obs_buses] + val["noise"][i]
        ll = -0.5 * np.sum(((V[:, lf_idx, :][:, obs_buses] - obs[None, :]) / 0.0015) ** 2, axis=1)
        pred.append(int(np.argmax(ll)))
    return float(np.mean(np.array(pred, dtype=np.int64) == y))


def search_119(lib):
    rng = np.random.RandomState(119909)
    score = bus_scores(lib["V"])
    val = make_val_119(lib, 1000, 119910)
    candidates = []
    names = []
    for p in [0.5, 1.0, 2.0, 4.0]:
        for _ in range(60):
            candidates.append(fill_to_k(stratified_choice(score, 119, 25, rng, power=p), score, 119, 25))
            names.append(f"strat_p{p}")
    for _ in range(80):
        w = score[1:] / np.sum(score[1:])
        candidates.append(np.sort(rng.choice(np.arange(1, 119), 25, replace=False, p=w)).astype(np.int64))
        names.append("global_weighted")
    best = None
    for name, s in zip(names, candidates):
        acc = exact_acc_119(lib, s, val)
        if best is None or acc > best["val_exact"]:
            best = {"name": name, "sensors": s, "val_exact": acc}
    return best


def search_300(lib):
    rng = np.random.RandomState(300909)
    score = bus_scores(lib["V"])
    val = make_val_300(lib, 800, 300910)
    candidates = [base.deployment_sensors(300, 150)]
    names = ["uniform"]
    for p in [0.5, 1.0, 2.0, 4.0]:
        for _ in range(35):
            candidates.append(fill_to_k(stratified_choice(score, 300, 150, rng, power=p), score, 300, 150))
            names.append(f"strat_p{p}")
    for width in [1, 2, 3, 5]:
        for _ in range(25):
            candidates.append(jitter_uniform_300(score, rng, width=width))
            names.append(f"jitter_w{width}")
    best = None
    for name, s in zip(names, candidates):
        acc = exact_acc_300(lib, s, val)
        if best is None or acc > best["val_exact"]:
            best = {"name": name, "sensors": s, "val_exact": acc}
    return best


def main():
    t0 = time.time()
    lines = []
    lines.append("Large-system diversity-constrained sensor-policy search")
    lines.append("date=2026-07-02")
    lines.append("selection=exact-reference validation draw; reporting=fresh final draw")
    lines.append("")

    lib119 = base.load_119_lib()
    best119 = search_119(lib119)
    lines.append(f"119_selected_policy={best119['name']}")
    lines.append(f"119_validation_exact={best119['val_exact']:.4f}")
    lines.append("119_selected_sensors=" + " ".join(map(str, best119["sensors"].tolist())))
    rows = []
    xs, y, ll = scan.make_119(lib119, 4000, 919119, "random")
    rows.append(scan.summarize("119bus_IP1", "random25", xs, y, ll, scan.load_119_logits(xs)))
    xs, y, ll = scan.make_119(lib119, 4000, 919120, "discriminative_fixed", best119["sensors"])
    rows.append(scan.summarize("119bus_IP1", "searched_fixed25", xs, y, ll, scan.load_119_logits(xs)))

    lib300 = base.load_300_lib()
    best300 = search_300(lib300)
    lines.append(f"300_selected_policy={best300['name']}")
    lines.append(f"300_validation_exact={best300['val_exact']:.4f}")
    lines.append("300_selected_sensors=" + " ".join(map(str, best300["sensors"].tolist())))
    xs, y, ll = scan.make_300(lib300, 5000, 930300, "uniform_fixed")
    rows.append(scan.summarize("300bus_IPC_miss30", "uniform150_miss30", xs, y, ll, scan.load_300_logits(xs)))
    xs, y, ll = scan.make_300(lib300, 5000, 930301, "discriminative_fixed", best300["sensors"])
    rows.append(scan.summarize("300bus_IPC_miss30", "searched150_miss30", xs, y, ll, scan.load_300_logits(xs)))

    lines.append("")
    lines.append("system,policy,n,exact_top1,nre_ensemble_top1,nre_exact_agree,truth_in_top20,exact_in_top20,rerank20,rerank20_vs_exact")
    for r in rows:
        lines.append(
            f"{r['system']},{r['policy']},{r['n']},{r['exact_top1']:.4f},"
            f"{r['nre_ensemble_top1']:.4f},{r['nre_exact_agree']:.4f},"
            f"{r['truth_in_top20']:.4f},{r['exact_in_top20']:.4f},"
            f"{r['rerank20']:.4f},{r['rerank20_vs_exact']:.4f}"
        )
    d = {(r["system"], r["policy"]): r for r in rows}
    lines.append("")
    lines.append("deltas")
    lines.append(f"119_exact_delta={d[('119bus_IP1','searched_fixed25')]['exact_top1'] - d[('119bus_IP1','random25')]['exact_top1']:.4f}")
    lines.append(f"119_nre_delta={d[('119bus_IP1','searched_fixed25')]['nre_ensemble_top1'] - d[('119bus_IP1','random25')]['nre_ensemble_top1']:.4f}")
    lines.append(f"300_exact_delta={d[('300bus_IPC_miss30','searched150_miss30')]['exact_top1'] - d[('300bus_IPC_miss30','uniform150_miss30')]['exact_top1']:.4f}")
    lines.append(f"300_nre_delta={d[('300bus_IPC_miss30','searched150_miss30')]['nre_ensemble_top1'] - d[('300bus_IPC_miss30','uniform150_miss30')]['nre_ensemble_top1']:.4f}")
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
