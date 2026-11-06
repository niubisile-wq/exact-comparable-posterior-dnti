# -*- coding: utf-8 -*-
"""
119-bus sensor-budget attack wrapper.

Selects stronger K=35/K=40 fixed sensor policies by exact-reference validation
and trains the existing sensor-specific v2 NRE on the selected K=40 policy.
No previous 25-sensor result or checkpoint is overwritten.
"""

import os
import time
import numpy as np
import dn_119bus_sensor_policy_v2_train_20260702 as m

BASE25 = np.array([5, 8, 11, 19, 23, 27, 34, 36, 44, 49, 51, 57, 62, 67, 74, 76, 84, 88, 93, 95, 102, 106, 109, 111, 118], dtype=np.int64)
OUT_NAME = "119bus_K40_sensor_budget_attack_20260702.txt"
SEL_NAME = "119bus_sensor_budget_selection_20260702.txt"


def eval_exact_sensors(lib, sensors, n=4500, seed=400123):
    rng = np.random.RandomState(seed)
    V = lib["V"]
    lf_grid = lib["lf"]
    n_topos, n_lf, _ = V.shape
    correct = 0
    for _ in range(n):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        obs = V[ti, lf_idx, sensors] + rng.normal(0.0, m.SIGMA, size=len(sensors))
        pred = V[:, lf_idx, :][:, sensors]
        ll = -0.5 * np.sum(((pred - obs[None, :]) / m.SIGMA) ** 2, axis=1)
        correct += int(np.argmax(ll) == ti)
    return correct / float(n)


def diversity_fill(lib, base, target_k, min_gap=2):
    V = lib["V"]
    n_bus = V.shape[2]
    flat = V.reshape(V.shape[0] * V.shape[1], n_bus)
    # Topology/load separability plus load-aware tie-breaker.
    scores = flat.std(axis=0) + 0.35 * lib["base_p"] / max(float(np.max(lib["base_p"])), 1e-9)
    selected = list(dict.fromkeys(int(x) for x in base if 0 < int(x) < n_bus))
    order = [int(i) for i in np.argsort(-scores) if i not in selected and i > 0]
    for idx in order:
        if len(selected) >= target_k:
            break
        if all(abs(idx - s) >= min_gap for s in selected):
            selected.append(idx)
    if len(selected) < target_k:
        for idx in order:
            if len(selected) >= target_k:
                break
            if idx not in selected:
                selected.append(idx)
    return np.array(sorted(selected[:target_k]), dtype=np.int64)


def lin_policy(n_bus, k):
    return np.unique(np.rint(np.linspace(1, n_bus - 1, k)).astype(int))


def main():
    t0 = time.time()
    lib = m.base.load_119_lib()
    n_bus = lib["V"].shape[2]
    policies = []
    policies.append(("old25", BASE25))
    for k in [35, 40]:
        policies.append((f"lin{k}", lin_policy(n_bus, k)))
        policies.append((f"hybrid{k}_gap2", diversity_fill(lib, BASE25, k, min_gap=2)))
        policies.append((f"hybrid{k}_gap4", diversity_fill(lib, BASE25, k, min_gap=4)))
    rows = []
    for name, sensors in policies:
        acc = eval_exact_sensors(lib, sensors, n=4500, seed=401000 + len(sensors) * 17 + len(name))
        rows.append((name, len(sensors), acc, sensors))
        print(f"policy={name} K={len(sensors)} exact={acc:.4f} sensors={' '.join(map(str, sensors))}", flush=True)
    best40 = max([r for r in rows if r[1] == 40], key=lambda r: r[2])
    best35 = max([r for r in rows if r[1] == 35], key=lambda r: r[2])
    selected = best40[3]

    sel_lines = []
    sel_lines.append("119-bus sensor-budget exact-reference selection")
    sel_lines.append("date=2026-07-02")
    sel_lines.append("selection_rule=exact_reference_validation_over_fixed_sensor_policies")
    sel_lines.append("policy,K,exact_top1,sensors")
    for name, k, acc, sensors in rows:
        sel_lines.append(f"{name},{k},{acc:.4f},{' '.join(map(str, sensors))}")
    sel_lines.append(f"best35={best35[0]},{best35[2]:.4f}")
    sel_lines.append(f"best40={best40[0]},{best40[2]:.4f}")
    sel_text = "\n".join(sel_lines) + "\n"
    with open(os.path.join(m.ROOT, SEL_NAME), "w", encoding="utf-8") as f:
        f.write(sel_text)
    print(sel_text)

    # Train selected K=40 specialized NRE.
    m.SENSORS = selected
    m.OUT_NAME = OUT_NAME
    m.N_TRAIN = 140000
    m.N_VAL = 5000
    m.N_FINAL = 8000
    m.EPOCHS = 10
    m.LR = 5e-5
    m.main()

    pkg = os.path.join(os.path.expanduser("~"), "Desktop", "配电网论文一区投稿成果包_20260702", "03_frozen_tables_stats")
    code = os.path.join(os.path.expanduser("~"), "Desktop", "配电网论文一区投稿成果包_20260702", "02_code")
    if os.path.isdir(pkg):
        for name in [SEL_NAME, OUT_NAME]:
            src = os.path.join(m.ROOT, name)
            if os.path.exists(src):
                with open(src, "r", encoding="utf-8") as f:
                    txt = f.read()
                with open(os.path.join(pkg, name), "w", encoding="utf-8") as f:
                    f.write(txt)
                print("Copied:", os.path.join(pkg, name))
    if os.path.isdir(code):
        with open(__file__, "r", encoding="utf-8") as f:
            txt = f.read()
        with open(os.path.join(code, os.path.basename(__file__)), "w", encoding="utf-8") as f:
            f.write(txt)
    print(f"elapsed_total_sec={time.time()-t0:.1f}")


if __name__ == "__main__":
    main()
