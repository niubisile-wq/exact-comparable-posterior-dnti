# -*- coding: utf-8 -*-
"""
300-bus IP-C 30% missing v2 retraining: exact-MAP distillation.

Isolated upgrade run:
  - starts from frozen nre_300bus_ipc_seed*.pt checkpoints;
  - keeps the same 300-bus library, sensor deployment, sigma, and 30% missing
    setting;
  - writes new v2 checkpoints and a separate frozen result file;
  - compares old ensemble, v2 ensemble, and validation-selected mix on a fresh
    final-test draw.
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
K_FIXED = 150
MISS_RATE = 0.30
BATCH = 1024
N_TRAIN = 140000
N_VAL = 4000
N_FINAL = 6000
EPOCHS = 10
LR = 5e-5
SEEDS = [42, 123, 456]
OUT_NAME = "300bus_v2_miss30_mapdistill_20260702.txt"


def make_dataset(lib, n, seed, return_ll=False):
    rng = np.random.RandomState(seed)
    V = lib["V"]
    lf_grid = lib["lf"]
    base_p = lib["base_p"]
    n_topos, n_lf, n_bus = V.shape
    xs = np.zeros((n, n_bus * 3), dtype=np.float32)
    y_true = np.zeros(n, dtype=np.int64)
    y_map = np.zeros(n, dtype=np.int64)
    ll_all = np.zeros((n, n_topos), dtype=np.float32) if return_ll else None
    installed_full = base.deployment_sensors(n_bus, K_FIXED)
    n_miss = int(K_FIXED * MISS_RATE)
    for i in range(n):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        lf = float(lf_grid[lf_idx])
        buses = np.delete(installed_full, rng.choice(len(installed_full), n_miss, replace=False))
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


def choose_mix(p_old_val, p_new_val, y_true_val, y_map_val):
    best = None
    for lam in np.linspace(0.0, 1.0, 101):
        p = lam * p_old_val + (1.0 - lam) * p_new_val
        res = eval_prob(p, y_true_val, y_map_val)
        score = res["true_acc"] + 0.05 * res["map_agree"]
        if best is None or score > best["score"]:
            best = {"lambda_old": float(lam), "score": score, **res}
    return best


def load_model(seed, lib, variant):
    n_topos, _, n_bus = lib["V"].shape
    model = base.NRE300(n_topos, n_bus).to(DEVICE)
    if variant == "old":
        path = os.path.join(ROOT, f"nre_300bus_ipc_seed{seed}.pt")
    elif variant == "v2":
        path = os.path.join(ROOT, f"nre_300bus_ipc_seed{seed}.pt")
    else:
        raise ValueError(variant)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model


def train_one(seed, lib, xs_train, y_true_train, y_map_train, xs_val, y_true_val, y_map_val):
    model = load_model(seed, lib, "v2")
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=EPOCHS * int(np.ceil(N_TRAIN / BATCH)),
        eta_min=8e-6,
    )
    rng = np.random.RandomState(300000 + seed)
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
            loss = ce(logits, yb_map) + 0.20 * ce(logits, yb_true)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            scheduler.step()
        val = eval_logits(predict_logits(model, xs_val), y_true_val, y_map_val)
        score = val["true_acc"] + 0.05 * val["map_agree"]
        print(
            f"seed={seed} epoch={epoch}/{EPOCHS} val_true={val['true_acc']:.4f} "
            f"val_map_agree={val['map_agree']:.4f} exact={val['exact_map_true_acc']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_val = val
    model.load_state_dict(best_state)
    ckpt_name = f"nre_300bus_ipc_miss30_v2_mapdistill_seed{seed}_20260702.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "seed": seed,
            "input": "300-bus IPC normalized voltage + mask + load feature",
            "loss": "CE(exact_MAP)+0.20*CE(true_topology)",
            "missing_rate": MISS_RATE,
            "best_epoch": best_epoch,
            "validation": best_val,
        },
        os.path.join(ROOT, ckpt_name),
    )
    return model, ckpt_name, best_epoch, best_val


def main():
    t0 = time.time()
    lib = base.load_300_lib()
    print(f"Device: {DEVICE}")
    print(f"300-bus v2 miss30 MAP distillation: train={N_TRAIN} val={N_VAL} final={N_FINAL}")
    xs_train, y_true_train, y_map_train = make_dataset(lib, N_TRAIN, seed=300202)
    xs_val, y_true_val, y_map_val, ll_val = make_dataset(lib, N_VAL, seed=300303, return_ll=True)
    xs_final, y_true_final, y_map_final, ll_final = make_dataset(lib, N_FINAL, seed=300404, return_ll=True)

    old_val_logits = []
    old_final_logits = []
    new_val_logits = []
    new_final_logits = []
    seed_rows = []
    for seed in SEEDS:
        old_model = load_model(seed, lib, "old")
        old_val_logits.append(predict_logits(old_model, xs_val))
        old_final_logits.append(predict_logits(old_model, xs_final))
        new_model, ckpt_name, best_epoch, best_val = train_one(
            seed,
            lib,
            xs_train,
            y_true_train,
            y_map_train,
            xs_val,
            y_true_val,
            y_map_val,
        )
        new_val_logits.append(predict_logits(new_model, xs_val))
        new_final_logits.append(predict_logits(new_model, xs_final))
        final_res = eval_logits(new_final_logits[-1], y_true_final, y_map_final)
        seed_rows.append(
            {
                "seed": seed,
                "checkpoint": ckpt_name,
                "best_epoch": best_epoch,
                "best_val": best_val,
                "final": final_res,
            }
        )

    p_old_val = avg_prob(old_val_logits)
    p_old_final = avg_prob(old_final_logits)
    p_new_val = avg_prob(new_val_logits)
    p_new_final = avg_prob(new_final_logits)
    old_final = eval_prob(p_old_final, y_true_final, y_map_final)
    new_final = eval_prob(p_new_final, y_true_final, y_map_final)
    mix_rule = choose_mix(p_old_val, p_new_val, y_true_val, y_map_val)
    p_mix_final = mix_rule["lambda_old"] * p_old_final + (1.0 - mix_rule["lambda_old"]) * p_new_final
    mix_final = eval_prob(p_mix_final, y_true_final, y_map_final)
    old_rerank = rerank_metrics(p_old_final, y_true_final, y_map_final, ll_final, m=20)
    new_rerank = rerank_metrics(p_new_final, y_true_final, y_map_final, ll_final, m=20)
    mix_rerank = rerank_metrics(p_mix_final, y_true_final, y_map_final, ll_final, m=20)

    lines = []
    lines.append("300-bus IP-C 30% missing v2 exact-MAP distillation retraining")
    lines.append("date=2026-07-02")
    lines.append("status=isolated_v2_does_not_overwrite_original_checkpoints")
    lines.append(f"device={DEVICE}")
    lines.append(f"train_samples={N_TRAIN}")
    lines.append(f"validation_samples={N_VAL}")
    lines.append(f"final_test_samples={N_FINAL}")
    lines.append(f"missing_rate={MISS_RATE:.2f}")
    lines.append("loss=CE(exact_MAP)+0.20*CE(true_topology)")
    lines.append(f"final_same_draw_exact_MAP_top1={old_final['exact_map_true_acc']:.4f}")
    lines.append("seed,v2_final_true_acc,v2_final_map_agree,best_epoch,best_val_true,best_val_map_agree,checkpoint")
    for r in seed_rows:
        lines.append(
            f"{r['seed']},{r['final']['true_acc']:.4f},{r['final']['map_agree']:.4f},"
            f"{r['best_epoch']},{r['best_val']['true_acc']:.4f},{r['best_val']['map_agree']:.4f},"
            f"{r['checkpoint']}"
        )
    lines.append(f"old_probability_ensemble_final_true_acc={old_final['true_acc']:.4f}")
    lines.append(f"old_probability_ensemble_final_map_agree={old_final['map_agree']:.4f}")
    lines.append(f"old_probability_ensemble_truth_in_top20={old_rerank['truth_in_top20']:.4f}")
    lines.append(f"old_probability_ensemble_map_in_top20={old_rerank['map_in_top20']:.4f}")
    lines.append(f"old_probability_ensemble_rerank20_true_acc={old_rerank['rerank20_true_acc']:.4f}")
    lines.append(f"v2_probability_ensemble_final_true_acc={new_final['true_acc']:.4f}")
    lines.append(f"v2_probability_ensemble_final_map_agree={new_final['map_agree']:.4f}")
    lines.append(f"v2_probability_ensemble_truth_in_top20={new_rerank['truth_in_top20']:.4f}")
    lines.append(f"v2_probability_ensemble_map_in_top20={new_rerank['map_in_top20']:.4f}")
    lines.append(f"v2_probability_ensemble_rerank20_true_acc={new_rerank['rerank20_true_acc']:.4f}")
    lines.append(f"validation_selected_lambda_old={mix_rule['lambda_old']:.2f}")
    lines.append(f"validation_selected_mix_val_true_acc={mix_rule['true_acc']:.4f}")
    lines.append(f"validation_selected_mix_val_map_agree={mix_rule['map_agree']:.4f}")
    lines.append(f"validation_selected_mix_final_true_acc={mix_final['true_acc']:.4f}")
    lines.append(f"validation_selected_mix_final_map_agree={mix_final['map_agree']:.4f}")
    lines.append(f"validation_selected_mix_truth_in_top20={mix_rerank['truth_in_top20']:.4f}")
    lines.append(f"validation_selected_mix_map_in_top20={mix_rerank['map_in_top20']:.4f}")
    lines.append(f"validation_selected_mix_rerank20_true_acc={mix_rerank['rerank20_true_acc']:.4f}")
    lines.append(f"delta_mix_vs_old_final={mix_final['true_acc'] - old_final['true_acc']:.4f}")
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
