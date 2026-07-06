# -*- coding: utf-8 -*-
"""
119-bus IP1 v2 retraining: exact-MAP distillation.

This is an isolated method-upgrade run. It does not overwrite old checkpoints.

Why this v2 differs from the failed short pilots:
  - start from the successful frozen 119-bus raw-input checkpoints;
  - keep the original raw-input convention;
  - train to imitate the exact-reference MAP topology, not the smoothed exact
    posterior alone;
  - retain a small true-topology CE term to preserve the original task metric;
  - select epochs only on a validation draw and report an independent test draw.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn

import dn_large_system_candidate_rerank_20260702 as base


ROOT = os.path.dirname(os.path.abspath(__file__))
PKG_STATS = r"<REPOSITORY_ROOT>\03_frozen_tables_stats"
PKG_CODE = r"<REPOSITORY_ROOT>\02_code"
DEVICE = base.DEVICE

SIGMA = 0.009
K_FIXED = 25
BATCH = 1024
N_TRAIN = 120000
N_VAL = 3000
N_TEST = 5000
EPOCHS = 8
LR = 8e-5
SEEDS = [42, 123, 456, 789, 2024]
OUT_NAME = "119bus_v2_mapdistill_20260702.txt"


def make_dataset(lib, n, seed):
    rng = np.random.RandomState(seed)
    V = lib["V"]
    lf_grid = lib["lf"]
    base_p = lib["base_p"]
    n_topos, n_lf, n_bus = V.shape
    xs = np.zeros((n, n_bus * 3), dtype=np.float32)
    y_true = np.zeros(n, dtype=np.int64)
    y_map = np.zeros(n, dtype=np.int64)
    for i in range(n):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        lf = float(lf_grid[lf_idx])
        buses = np.sort(rng.choice(range(1, n_bus), K_FIXED, replace=False))
        obs = V[ti, lf_idx, buses] + rng.normal(0.0, SIGMA, size=K_FIXED)
        pred = V[:, lf_idx, :][:, buses]
        ll = -0.5 * np.sum(((pred - obs[None, :]) / SIGMA) ** 2, axis=1)
        xs[i, buses] = obs
        xs[i, n_bus + buses] = 1.0
        xs[i, 2 * n_bus + buses] = base_p[buses] * lf
        y_true[i] = ti
        y_map[i] = int(np.argmax(ll))
    return xs, y_true, y_map


def predict_logits(model, xs):
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(xs), BATCH):
            xb = torch.tensor(xs[start:start + BATCH], dtype=torch.float32).to(DEVICE)
            outs.append(model(xb).detach().cpu().numpy())
    return np.vstack(outs)


def eval_logits(logits, y_true, y_map):
    pred = np.argmax(logits, axis=1)
    return {
        "true_acc": float(np.mean(pred == y_true)),
        "map_agree": float(np.mean(pred == y_map)),
        "exact_map_true_acc": float(np.mean(y_map == y_true)),
    }


def softmax_np(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    p = np.exp(z)
    return p / np.sum(p, axis=1, keepdims=True)


def ensemble_eval(logits_list, y_true, y_map):
    p = np.mean(np.stack([softmax_np(z) for z in logits_list], axis=0), axis=0)
    pred = np.argmax(p, axis=1)
    return {
        "true_acc": float(np.mean(pred == y_true)),
        "map_agree": float(np.mean(pred == y_map)),
        "exact_map_true_acc": float(np.mean(y_map == y_true)),
    }


def train_one(seed, lib, xs_train, y_true_train, y_map_train, xs_val, y_true_val, y_map_val, xs_test, y_true_test, y_map_test):
    n_topos, _, n_bus = lib["V"].shape
    ckpt = torch.load(os.path.join(ROOT, f"nre_119bus_ip1_seed{seed}.pt"), map_location=DEVICE, weights_only=False)
    model = base.LoadAwareNRE119(n_topos, n_bus).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    before = eval_logits(predict_logits(model, xs_test), y_true_test, y_map_test)

    torch.manual_seed(seed + 9000)
    np.random.seed(seed + 9000)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=EPOCHS * int(np.ceil(N_TRAIN / BATCH)),
        eta_min=1e-5,
    )
    ce = nn.CrossEntropyLoss()
    rng = np.random.RandomState(seed + 9100)
    best_val = None
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0

    x_train_t = torch.tensor(xs_train, dtype=torch.float32)
    y_true_t = torch.tensor(y_true_train, dtype=torch.long)
    y_map_t = torch.tensor(y_map_train, dtype=torch.long)

    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(N_TRAIN)
        model.train()
        for start in range(0, N_TRAIN, BATCH):
            idx = order[start:start + BATCH]
            xb = x_train_t[idx].to(DEVICE)
            yb_true = y_true_t[idx].to(DEVICE)
            yb_map = y_map_t[idx].to(DEVICE)
            logits = model(xb)
            loss = ce(logits, yb_map) + 0.20 * ce(logits, yb_true)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            scheduler.step()
        val = eval_logits(predict_logits(model, xs_val), y_true_val, y_map_val)
        score = val["true_acc"] + 0.15 * val["map_agree"]
        print(
            f"seed={seed} epoch={epoch}/{EPOCHS} val_true={val['true_acc']:.4f} "
            f"val_map_agree={val['map_agree']:.4f} exact={val['exact_map_true_acc']:.4f}",
            flush=True,
        )
        if best_val is None or score > best_val["score"]:
            best_val = {**val, "score": score}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    model.load_state_dict(best_state)
    after_logits = predict_logits(model, xs_test)
    after = eval_logits(after_logits, y_true_test, y_map_test)
    ckpt_name = f"nre_119bus_ip1_v2_mapdistill_seed{seed}_20260702.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "seed": seed,
            "n_topos": n_topos,
            "n_bus": n_bus,
            "input": "original raw voltage + mask + load feature",
            "loss": "CE(exact_MAP)+0.20*CE(true_topology)",
            "best_epoch": best_epoch,
            "validation": best_val,
        },
        os.path.join(ROOT, ckpt_name),
    )
    return {
        "seed": seed,
        "before": before,
        "after": after,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "checkpoint": ckpt_name,
        "after_logits": after_logits,
    }


def main():
    t0 = time.time()
    lib = base.load_119_lib()
    print(f"Device: {DEVICE}")
    print(f"119-bus v2 MAP distillation: train={N_TRAIN} val={N_VAL} test={N_TEST} seeds={SEEDS}")
    xs_train, y_true_train, y_map_train = make_dataset(lib, N_TRAIN, seed=119202)
    xs_val, y_true_val, y_map_val = make_dataset(lib, N_VAL, seed=119303)
    xs_test, y_true_test, y_map_test = make_dataset(lib, N_TEST, seed=119404)

    rows = []
    logits_after = []
    for seed in SEEDS:
        row = train_one(
            seed,
            lib,
            xs_train,
            y_true_train,
            y_map_train,
            xs_val,
            y_true_val,
            y_map_val,
            xs_test,
            y_true_test,
            y_map_test,
        )
        rows.append(row)
        logits_after.append(row["after_logits"])

    ens = ensemble_eval(logits_after, y_true_test, y_map_test)
    before_mean = np.mean([r["before"]["true_acc"] for r in rows])
    after_mean = np.mean([r["after"]["true_acc"] for r in rows])
    after_std = np.std([r["after"]["true_acc"] for r in rows])

    lines = []
    lines.append("119-bus IP1 v2 exact-MAP distillation retraining")
    lines.append("date=2026-07-02")
    lines.append("status=isolated_v2_does_not_overwrite_original_checkpoints")
    lines.append(f"device={DEVICE}")
    lines.append(f"train_samples={N_TRAIN}")
    lines.append(f"validation_samples={N_VAL}")
    lines.append(f"test_samples={N_TEST}")
    lines.append("input=original raw voltage + mask + load feature")
    lines.append("loss=CE(exact_MAP)+0.20*CE(true_topology)")
    lines.append(f"same_draw_exact_MAP_top1={rows[0]['after']['exact_map_true_acc']:.4f}")
    lines.append("seed,before_true_acc,after_true_acc,before_map_agree,after_map_agree,best_epoch,best_val_true,best_val_map_agree,checkpoint")
    for r in rows:
        lines.append(
            f"{r['seed']},{r['before']['true_acc']:.4f},{r['after']['true_acc']:.4f},"
            f"{r['before']['map_agree']:.4f},{r['after']['map_agree']:.4f},"
            f"{r['best_epoch']},{r['best_val']['true_acc']:.4f},{r['best_val']['map_agree']:.4f},"
            f"{r['checkpoint']}"
        )
    lines.append(f"before_seed_mean_true_acc={before_mean:.4f}")
    lines.append(f"after_seed_mean_true_acc={after_mean:.4f}")
    lines.append(f"after_seed_std_true_acc={after_std:.4f}")
    lines.append(f"after_probability_ensemble_true_acc={ens['true_acc']:.4f}")
    lines.append(f"after_probability_ensemble_map_agree={ens['map_agree']:.4f}")
    lines.append(f"delta_seed_mean_true_acc={after_mean - before_mean:.4f}")
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
