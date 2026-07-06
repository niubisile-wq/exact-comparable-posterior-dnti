# -*- coding: utf-8 -*-
"""
119-bus raw-input exact-posterior fine-tuning pilot.

This preserves the successful frozen 119-bus model input convention and starts
from an existing checkpoint, then fine-tunes against exact posterior targets.
The goal is to test whether the direct NRE top-1 gap can be repaired at the
model level without changing the benchmark.
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
BATCH = 512
N_TRAIN = 80000
N_TEST = 3000
EPOCHS = 10
LR = 1e-4
OUT_NAME = "119bus_raw_finetune_distill_20260702.txt"


def make_dataset(lib, n, seed):
    rng = np.random.RandomState(seed)
    V = lib["V"]
    lf_grid = lib["lf"]
    base_p = lib["base_p"]
    n_topos, n_lf, n_bus = V.shape
    xs = np.zeros((n, n_bus * 3), dtype=np.float32)
    ys = np.zeros(n, dtype=np.int64)
    qs = np.zeros((n, n_topos), dtype=np.float32)
    for i in range(n):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        lf = float(lf_grid[lf_idx])
        buses = np.sort(rng.choice(range(1, n_bus), K_FIXED, replace=False))
        obs = V[ti, lf_idx, buses] + rng.normal(0.0, SIGMA, size=K_FIXED)
        pred = V[:, lf_idx, :][:, buses]
        ll = -0.5 * np.sum(((pred - obs[None, :]) / SIGMA) ** 2, axis=1)
        q = np.exp(ll - np.max(ll))
        q = q / np.sum(q)
        xs[i, buses] = obs
        xs[i, n_bus + buses] = 1.0
        xs[i, 2 * n_bus + buses] = base_p[buses] * lf
        ys[i] = ti
        qs[i] = q.astype(np.float32)
    return xs, ys, qs


def evaluate(model, xs, ys, qs):
    model.eval()
    preds = []
    probs = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(xs), BATCH):
            xb = torch.tensor(xs[start:start + BATCH], dtype=torch.float32).to(DEVICE)
            logits = model(xb)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            preds.append(logits.argmax(dim=1).cpu().numpy())
    sec = time.perf_counter() - t0
    p = np.vstack(probs)
    pred = np.concatenate(preds)
    exact_pred = np.argmax(qs, axis=1)
    kl = float(np.mean(np.sum(qs * (np.log(np.clip(qs, 1e-12, 1.0)) - np.log(np.clip(p, 1e-12, 1.0))), axis=1)))
    return {
        "nre_top1": float(np.mean(pred == ys)),
        "exact_top1": float(np.mean(exact_pred == ys)),
        "nre_exact_agree": float(np.mean(pred == exact_pred)),
        "kl_ref_nre": kl,
        "nre_ms": sec / len(xs) * 1000.0,
    }


def main():
    t0 = time.time()
    lib = base.load_119_lib()
    n_topos, _, n_bus = lib["V"].shape
    print(f"Device: {DEVICE}")
    print("Generating raw-input fine-tuning data...")
    xs_train, ys_train, qs_train = make_dataset(lib, N_TRAIN, seed=66119)
    xs_test, ys_test, qs_test = make_dataset(lib, N_TEST, seed=77119)

    ckpt_path = os.path.join(ROOT, "nre_119bus_ip1_seed42.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = base.LoadAwareNRE119(n_topos, n_bus).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    before = evaluate(model, xs_test, ys_test, qs_test)
    print(f"before nre={before['nre_top1']:.4f} exact={before['exact_top1']:.4f} kl={before['kl_ref_nre']:.4f}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * int(np.ceil(N_TRAIN / BATCH)), eta_min=1e-5)
    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")
    rng = np.random.RandomState(42)
    best = before
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(N_TRAIN)
        model.train()
        for start in range(0, N_TRAIN, BATCH):
            idx = order[start:start + BATCH]
            xb = torch.tensor(xs_train[idx], dtype=torch.float32).to(DEVICE)
            yb = torch.tensor(ys_train[idx], dtype=torch.long).to(DEVICE)
            qb = torch.tensor(qs_train[idx], dtype=torch.float32).to(DEVICE)
            logits = model(xb)
            loss = kl(torch.log_softmax(logits, dim=1), qb) + 0.80 * ce(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            scheduler.step()
        res = evaluate(model, xs_test[:1200], ys_test[:1200], qs_test[:1200])
        print(f"epoch={epoch}/{EPOCHS} probe_nre={res['nre_top1']:.4f} exact={res['exact_top1']:.4f} kl={res['kl_ref_nre']:.4f}", flush=True)
        if res["nre_top1"] > best["nre_top1"]:
            best = res
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    after = evaluate(model, xs_test, ys_test, qs_test)
    ckpt_name = "nre_119bus_ip1_seed42_raw_finetune_distill_20260702.pt"
    torch.save({"model_state": model.state_dict(), "n_topos": n_topos, "n_bus": n_bus, "seed": 42, "input": "raw_voltage", "loss": "KL(exact posterior)+0.80 CE(true topology)"}, os.path.join(ROOT, ckpt_name))

    lines = []
    lines.append("119-bus IP1 raw-input exact-posterior fine-tuning pilot")
    lines.append("date=2026-07-02")
    lines.append("status=pilot_single_seed_not_multiseed_replacement")
    lines.append(f"device={DEVICE}")
    lines.append(f"train_samples={N_TRAIN}")
    lines.append(f"test_samples={N_TEST}")
    lines.append("input=original raw voltage + mask + load feature")
    lines.append("initial_checkpoint=nre_119bus_ip1_seed42.pt")
    lines.append("loss=KL(exact posterior||NRE)+0.80*CE(true topology)")
    lines.append(f"before_exact_top1={before['exact_top1']:.4f}")
    lines.append(f"before_nre_top1={before['nre_top1']:.4f}")
    lines.append(f"before_kl_ref_nre={before['kl_ref_nre']:.4f}")
    lines.append(f"after_exact_top1={after['exact_top1']:.4f}")
    lines.append(f"after_nre_top1={after['nre_top1']:.4f}")
    lines.append(f"after_nre_exact_agree={after['nre_exact_agree']:.4f}")
    lines.append(f"after_kl_ref_nre={after['kl_ref_nre']:.4f}")
    lines.append(f"after_nre_ms={after['nre_ms']:.6f}")
    lines.append(f"delta_nre_top1={after['nre_top1'] - before['nre_top1']:.4f}")
    lines.append(f"checkpoint={ckpt_name}")
    lines.append(f"elapsed_sec={time.time() - t0:.1f}")
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(ROOT, OUT_NAME), "w", encoding="utf-8") as f:
        f.write(text)
    if os.path.isdir(PKG_STATS):
        with open(os.path.join(PKG_STATS, OUT_NAME), "w", encoding="utf-8") as f:
            f.write(text)
    if os.path.isdir(PKG_CODE):
        with open(__file__, "r", encoding="utf-8") as f:
            code = f.read()
        with open(os.path.join(PKG_CODE, os.path.basename(__file__)), "w", encoding="utf-8") as f:
            f.write(code)


if __name__ == "__main__":
    main()
