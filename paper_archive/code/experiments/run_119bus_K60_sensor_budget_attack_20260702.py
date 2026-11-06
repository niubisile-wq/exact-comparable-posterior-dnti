# -*- coding: utf-8 -*-
"""
119-bus high-budget sensor attack wrapper.

Screens K45/K50/K60 fixed policies by exact-reference validation and trains one
selected high-budget policy with explicit checkpoint names. No K25/K40
checkpoint is overwritten.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import dn_119bus_sensor_policy_v2_train_20260702 as m

BASE25 = np.array([5, 8, 11, 19, 23, 27, 34, 36, 44, 49, 51, 57, 62, 67, 74, 76, 84, 88, 93, 95, 102, 106, 109, 111, 118], dtype=np.int64)
BASE40 = np.array([5, 8, 11, 14, 19, 21, 23, 27, 31, 34, 36, 40, 42, 44, 49, 51, 54, 57, 62, 67, 71, 74, 76, 78, 80, 82, 84, 86, 88, 93, 95, 97, 102, 104, 106, 109, 111, 114, 116, 118], dtype=np.int64)
SEL_NAME = "119bus_highK_sensor_budget_selection_20260702.txt"
OUT_NAME = "119bus_K60_sensor_budget_attack_20260702.txt"


def eval_exact_sensors(lib, sensors, n=5000, seed=600123):
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


def lin_policy(n_bus, k):
    return np.unique(np.rint(np.linspace(1, n_bus - 1, k)).astype(int))


def diversity_fill(lib, base, target_k, min_gap=2):
    V = lib["V"]
    n_bus = V.shape[2]
    flat = V.reshape(V.shape[0] * V.shape[1], n_bus)
    topo_sep = flat.std(axis=0)
    load = lib["base_p"] / max(float(np.max(lib["base_p"])), 1e-9)
    scores = topo_sep + 0.30 * load
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


def custom_train_one(seed, lib, xs_train, y_true_train, y_map_train, xs_val, y_true_val, y_map_val):
    model = m.load_old_model(seed, lib)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=m.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=m.EPOCHS * int(np.ceil(m.N_TRAIN / m.BATCH)), eta_min=8e-6)
    x_train_t = torch.tensor(xs_train, dtype=torch.float32)
    y_true_t = torch.tensor(y_true_train, dtype=torch.long)
    y_map_t = torch.tensor(y_map_train, dtype=torch.long)
    rng = np.random.RandomState(619000 + seed)
    best_score = -1e9
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    best_val = None
    for epoch in range(1, m.EPOCHS + 1):
        order = rng.permutation(m.N_TRAIN)
        model.train()
        for start in range(0, m.N_TRAIN, m.BATCH):
            idx = order[start:start + m.BATCH]
            xb = x_train_t[idx].to(m.DEVICE)
            yb_true = y_true_t[idx].to(m.DEVICE)
            yb_map = y_map_t[idx].to(m.DEVICE)
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb_true) + 0.20 * ce(logits, yb_map)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            scheduler.step()
        val = m.eval_logits(m.predict_logits(model, xs_val), y_true_val, y_map_val)
        score = val["true_acc"] + 0.05 * val["map_agree"]
        print(f"seed={seed} K={len(m.SENSORS)} epoch={epoch}/{m.EPOCHS} val_true={val['true_acc']:.4f} val_map_agree={val['map_agree']:.4f} exact={val['exact_map_true_acc']:.4f}", flush=True)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_val = val
    model.load_state_dict(best_state)
    ckpt_name = f"nre_119bus_ip1_K{len(m.SENSORS)}_sensor_v2_seed{seed}_20260702.pt"
    torch.save({
        "model_state": model.state_dict(),
        "seed": seed,
        "sensors": m.SENSORS,
        "input": "original raw voltage + mask + load feature",
        "loss": "CE(true_topology)+0.20*CE(exact_MAP)",
        "best_epoch": best_epoch,
        "validation": best_val,
    }, os.path.join(m.ROOT, ckpt_name))
    return model, ckpt_name, best_epoch, best_val


def main():
    t0 = time.time()
    lib = m.base.load_119_lib()
    n_bus = lib["V"].shape[2]
    policies = [("old25", BASE25), ("hybrid40_gap2", BASE40)]
    for k in [45, 50, 60]:
        policies.append((f"lin{k}", lin_policy(n_bus, k)))
        policies.append((f"hybrid{k}_from40_gap2", diversity_fill(lib, BASE40, k, min_gap=2)))
        policies.append((f"hybrid{k}_from40_gap3", diversity_fill(lib, BASE40, k, min_gap=3)))
    rows = []
    for name, sensors in policies:
        acc = eval_exact_sensors(lib, sensors, n=5000, seed=602000 + len(name) * 13 + len(sensors))
        rows.append((name, len(sensors), acc, sensors))
        print(f"policy={name} K={len(sensors)} exact={acc:.4f} sensors={' '.join(map(str, sensors))}", flush=True)
    candidates = [r for r in rows if r[1] in (50, 60)]
    best = max(candidates, key=lambda r: r[2])
    sel_lines = [
        "119-bus high-K sensor-budget exact-reference selection",
        "date=2026-07-02",
        "selection_rule=exact_reference_validation_over_fixed_sensor_policies; train best among K50/K60 candidates",
        "policy,K,exact_top1,sensors",
    ]
    for name, k, acc, sensors in rows:
        sel_lines.append(f"{name},{k},{acc:.4f},{' '.join(map(str, sensors))}")
    sel_lines.append(f"selected={best[0]},{best[1]},{best[2]:.4f}")
    sel_text = "\n".join(sel_lines) + "\n"
    with open(os.path.join(m.ROOT, SEL_NAME), "w", encoding="utf-8") as f:
        f.write(sel_text)
    print(sel_text)

    m.SENSORS = best[3]
    m.OUT_NAME = OUT_NAME.replace("K60", f"K{len(m.SENSORS)}")
    m.N_TRAIN = 150000
    m.N_VAL = 5000
    m.N_FINAL = 9000
    m.EPOCHS = 10
    m.LR = 5e-5
    m.train_one = custom_train_one
    m.main()

    pkg = os.path.join(os.path.expanduser("~"), "Desktop", "配电网论文一区投稿成果包_20260702", "03_frozen_tables_stats")
    code = os.path.join(os.path.expanduser("~"), "Desktop", "配电网论文一区投稿成果包_20260702", "02_code")
    if os.path.isdir(pkg):
        for name in [SEL_NAME, m.OUT_NAME]:
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
