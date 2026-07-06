# -*- coding: utf-8 -*-
"""
300-bus severe-missing attack experiment: redundant sensing.

Goal:
  Move beyond defending the weak K=150, 30% missing result by testing an
  operational repair: deploy more sensors, keep the same 30% missing rate, and
  train a sensor-specific NRE for the repaired deployment.

This does not overwrite any old checkpoints.
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

SIGMA = 0.0015
MISS_RATE = 0.30
BATCH = 1024
N_TRAIN = 220000
N_VAL = 6000
N_FINAL = 9000
EPOCHS = 14
LR = 4e-5
SEEDS = [42, 123, 456]
K_ATTACK = 220
OUT_NAME = "300bus_redundant_sensing_attack_20260702.txt"


def make_dataset(lib, n, seed, k, return_ll=False):
    rng = np.random.RandomState(seed)
    V = lib["V"]
    lf_grid = lib["lf"]
    base_p = lib["base_p"]
    n_topos, n_lf, n_bus = V.shape
    deploy = base.deployment_sensors(n_bus, k)
    n_miss = int(k * MISS_RATE)
    n_obs = k - n_miss
    xs = np.zeros((n, n_bus * 3), dtype=np.float32)
    y_true = np.zeros(n, dtype=np.int64)
    y_map = np.zeros(n, dtype=np.int64)
    ll_all = np.zeros((n, n_topos), dtype=np.float32) if return_ll else None
    for i in range(n):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        lf = float(lf_grid[lf_idx])
        obs_idx = np.sort(rng.choice(k, n_obs, replace=False))
        buses = deploy[obs_idx]
        obs = V[ti, lf_idx, buses] + rng.normal(0.0, SIGMA, size=len(buses))
        pred = V[:, lf_idx, :][:, buses]
        ll = -0.5 * np.sum(((pred - obs[None, :]) / SIGMA) ** 2, axis=1)
        xs[i, buses] = (obs - 1.0) / SIGMA
        xs[i, n_bus + buses] = 1.0
        xs[i, 2 * n_bus + buses] = base_p[buses] * lf
        y_true[i] = ti
        y_map[i] = int(np.argmax(ll))
        if return_ll:
            ll_all[i] = ll.astype(np.float32)
    if return_ll:
        return xs, y_true, y_map, ll_all
    return xs, y_true, y_map


def softmax_np(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    p = np.exp(z)
    return p / np.sum(p, axis=1, keepdims=True)


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


def avg_prob(logits_list):
    return np.mean(np.stack([softmax_np(z) for z in logits_list], axis=0), axis=0)


def eval_prob(p, y_true, y_map):
    pred = np.argmax(p, axis=1)
    return {
        "true_acc": float(np.mean(pred == y_true)),
        "map_agree": float(np.mean(pred == y_map)),
        "exact_map_true_acc": float(np.mean(y_map == y_true)),
    }


def rerank_metrics(p, y_true, y_map, ll, m=20):
    cand = np.argsort(-p, axis=1)[:, :m]
    truth_in = np.array([y_true[i] in cand[i] for i in range(len(y_true))], dtype=bool)
    map_in = np.array([y_map[i] in cand[i] for i in range(len(y_true))], dtype=bool)
    pred = np.empty(len(y_true), dtype=np.int64)
    for i in range(len(y_true)):
        ci = cand[i]
        pred[i] = ci[int(np.argmax(ll[i, ci]))]
    return {
        "truth_in_top20": float(np.mean(truth_in)),
        "map_in_top20": float(np.mean(map_in)),
        "rerank20_true_acc": float(np.mean(pred == y_true)),
        "rerank20_map_agree": float(np.mean(pred == y_map)),
    }


def load_old_model(seed, lib):
    n_topos, _, n_bus = lib["V"].shape
    model = base.NRE300(n_topos, n_bus).to(DEVICE)
    ckpt = torch.load(os.path.join(ROOT, f"nre_300bus_ipc_seed{seed}.pt"), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model


def train_one(seed, lib, xs_train, y_true_train, y_map_train, xs_val, y_true_val, y_map_val):
    model = load_old_model(seed, lib)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=EPOCHS * int(np.ceil(N_TRAIN / BATCH)),
        eta_min=6e-6,
    )
    rng = np.random.RandomState(220000 + seed)
    x_train_t = torch.tensor(xs_train, dtype=torch.float32)
    y_true_t = torch.tensor(y_true_train, dtype=torch.long)
    y_map_t = torch.tensor(y_map_train, dtype=torch.long)
    best_score = -1e9
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    best_val = None
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(N_TRAIN)
        model.train()
        for start in range(0, N_TRAIN, BATCH):
            idx = order[start:start + BATCH]
            xb = x_train_t[idx].to(DEVICE)
            yb_true = y_true_t[idx].to(DEVICE)
            yb_map = y_map_t[idx].to(DEVICE)
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb_true) + 0.15 * nn.functional.cross_entropy(logits, yb_map)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            scheduler.step()
        val = eval_logits(predict_logits(model, xs_val), y_true_val, y_map_val)
        score = val["true_acc"] + 0.05 * val["map_agree"]
        print(
            f"seed={seed} K={K_ATTACK} epoch={epoch}/{EPOCHS} val_true={val['true_acc']:.4f} "
            f"val_map_agree={val['map_agree']:.4f} exact={val['exact_map_true_acc']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_val = val
    model.load_state_dict(best_state)
    ckpt_name = f"nre_300bus_ipc_K{K_ATTACK}_miss30_redundant_seed{seed}_20260702.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "seed": seed,
            "K": K_ATTACK,
            "missing_rate": MISS_RATE,
            "loss": "CE(true_topology)+0.15*CE(exact_MAP)",
            "best_epoch": best_epoch,
            "validation": best_val,
        },
        os.path.join(ROOT, ckpt_name),
    )
    return model, ckpt_name, best_epoch, best_val


def exact_curve(lib):
    rows = []
    for k, seed in [(150, 150404), (180, 180404), (220, 220404)]:
        _, y, y_map, _ = make_dataset(lib, 6000, seed, k, return_ll=True)
        rows.append((k, int(k - int(k * MISS_RATE)), float(np.mean(y_map == y))))
    return rows


def main():
    t0 = time.time()
    lib = base.load_300_lib()
    print(f"Device: {DEVICE}")
    print(f"300-bus redundant sensing attack: K={K_ATTACK} miss={MISS_RATE} train={N_TRAIN} val={N_VAL} final={N_FINAL}")
    curve = exact_curve(lib)
    xs_train, y_true_train, y_map_train = make_dataset(lib, N_TRAIN, 220101, K_ATTACK)
    xs_val, y_true_val, y_map_val, ll_val = make_dataset(lib, N_VAL, 220202, K_ATTACK, return_ll=True)
    xs_final, y_true_final, y_map_final, ll_final = make_dataset(lib, N_FINAL, 220303, K_ATTACK, return_ll=True)

    old_val_logits = []
    old_final_logits = []
    new_val_logits = []
    new_final_logits = []
    rows = []
    for seed in SEEDS:
        old = load_old_model(seed, lib)
        old_val_logits.append(predict_logits(old, xs_val))
        old_final_logits.append(predict_logits(old, xs_final))
        model, ckpt_name, best_epoch, best_val = train_one(
            seed,
            lib,
            xs_train,
            y_true_train,
            y_map_train,
            xs_val,
            y_true_val,
            y_map_val,
        )
        new_val_logits.append(predict_logits(model, xs_val))
        new_final_logits.append(predict_logits(model, xs_final))
        final = eval_logits(new_final_logits[-1], y_true_final, y_map_final)
        rows.append({"seed": seed, "checkpoint": ckpt_name, "best_epoch": best_epoch, "best_val": best_val, "final": final})

    p_old_final = avg_prob(old_final_logits)
    p_new_final = avg_prob(new_final_logits)
    old_final = eval_prob(p_old_final, y_true_final, y_map_final)
    new_final = eval_prob(p_new_final, y_true_final, y_map_final)
    old_rr = rerank_metrics(p_old_final, y_true_final, y_map_final, ll_final, m=20)
    new_rr = rerank_metrics(p_new_final, y_true_final, y_map_final, ll_final, m=20)

    lines = []
    lines.append("300-bus IP-C severe-missing redundant-sensing attack")
    lines.append("date=2026-07-02")
    lines.append("status=isolated_redundant_sensing_v2_does_not_overwrite_original_checkpoints")
    lines.append(f"device={DEVICE}")
    lines.append(f"missing_rate={MISS_RATE:.2f}")
    lines.append("exact_reference_curve")
    lines.append("K,observed_after_30pct_missing,exact_top1")
    for k, obs, acc in curve:
        lines.append(f"{k},{obs},{acc:.4f}")
    lines.append(f"attack_K={K_ATTACK}")
    lines.append(f"train_samples={N_TRAIN}")
    lines.append(f"validation_samples={N_VAL}")
    lines.append(f"final_test_samples={N_FINAL}")
    lines.append("loss=CE(true_topology)+0.15*CE(exact_MAP)")
    lines.append(f"final_same_draw_exact_MAP_top1={old_final['exact_map_true_acc']:.4f}")
    lines.append("seed,v2_final_true_acc,v2_final_map_agree,best_epoch,best_val_true,best_val_map_agree,checkpoint")
    for r in rows:
        lines.append(
            f"{r['seed']},{r['final']['true_acc']:.4f},{r['final']['map_agree']:.4f},"
            f"{r['best_epoch']},{r['best_val']['true_acc']:.4f},{r['best_val']['map_agree']:.4f},"
            f"{r['checkpoint']}"
        )
    lines.append(f"old_K150_checkpoint_ensemble_on_K{K_ATTACK}_final_true_acc={old_final['true_acc']:.4f}")
    lines.append(f"old_K150_checkpoint_ensemble_on_K{K_ATTACK}_map_agree={old_final['map_agree']:.4f}")
    lines.append(f"old_K150_checkpoint_ensemble_on_K{K_ATTACK}_truth_in_top20={old_rr['truth_in_top20']:.4f}")
    lines.append(f"old_K150_checkpoint_ensemble_on_K{K_ATTACK}_rerank20_true_acc={old_rr['rerank20_true_acc']:.4f}")
    lines.append(f"v2_K{K_ATTACK}_ensemble_final_true_acc={new_final['true_acc']:.4f}")
    lines.append(f"v2_K{K_ATTACK}_ensemble_map_agree={new_final['map_agree']:.4f}")
    lines.append(f"v2_K{K_ATTACK}_ensemble_truth_in_top20={new_rr['truth_in_top20']:.4f}")
    lines.append(f"v2_K{K_ATTACK}_ensemble_map_in_top20={new_rr['map_in_top20']:.4f}")
    lines.append(f"v2_K{K_ATTACK}_ensemble_rerank20_true_acc={new_rr['rerank20_true_acc']:.4f}")
    lines.append(f"delta_v2_vs_old_checkpoint_on_K{K_ATTACK}={new_final['true_acc'] - old_final['true_acc']:.4f}")
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
