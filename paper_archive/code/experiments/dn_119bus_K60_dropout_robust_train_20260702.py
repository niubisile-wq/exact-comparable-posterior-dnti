# -*- coding: utf-8 -*-
"""
119-bus K60 dropout-robust NRE training.

Repairs the random sensor-dropout weakness exposed by the no-retraining dropout
audit. Trains sensor-specific K60 models with random deployment-sensor dropout
augmentation and evaluates against fixed K60 checkpoints on identical dropout
regimes. No old checkpoints are overwritten.
"""

from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn

import dn_119bus_sensor_policy_v2_train_20260702 as m119

ROOT = Path.home() / "Desktop" / "配电网实验_临时"
PKG_ROOT = Path.home() / "Desktop" / "配电网论文一区投稿成果包_20260702"
PKG_STATS = PKG_ROOT / "03_frozen_tables_stats"
PKG_CODE = PKG_ROOT / "02_code"
OUT_NAME = "119bus_K60_dropout_robust_train_20260702.txt"
DEVICE = m119.DEVICE

SENSORS = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118], dtype=np.int64)
DROP_TRAIN = [0.0, 0.10, 0.20, 0.40]
DROP_EVAL = [0.0, 0.10, 0.20, 0.40]
SEEDS = [42, 123, 456]
BATCH = 1024
N_TRAIN = 120000
N_VAL = 4500
N_TEST = 5000
EPOCHS = 8
LR = 5e-5
SIGMA = m119.SIGMA


def make_dataset(lib, n, seed, drop_rates, return_ll=False):
    rng = np.random.RandomState(seed)
    V = lib["V"]
    lf_grid = lib["lf"]
    base_p = lib["base_p"]
    n_topos, n_lf, n_bus = V.shape
    xs = np.zeros((n, n_bus * 3), dtype=np.float32)
    y_true = np.zeros(n, dtype=np.int64)
    y_map = np.zeros(n, dtype=np.int64)
    ll_all = np.zeros((n, n_topos), dtype=np.float32) if return_ll else None
    retained = np.zeros(n, dtype=np.int64)
    for i in range(n):
        dr = float(drop_rates[rng.randint(0, len(drop_rates))])
        keep_n = max(1, int(round(len(SENSORS) * (1.0 - dr))))
        buses = np.sort(rng.choice(SENSORS, keep_n, replace=False))
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        lf = float(lf_grid[lf_idx])
        obs = V[ti, lf_idx, buses] + rng.normal(0.0, SIGMA, size=len(buses))
        pred = V[:, lf_idx, :][:, buses]
        ll = -0.5 * np.sum(((pred - obs[None, :]) / SIGMA) ** 2, axis=1)
        xs[i, buses] = obs
        xs[i, n_bus + buses] = 1.0
        xs[i, 2 * n_bus + buses] = base_p[buses] * lf
        y_true[i] = ti
        y_map[i] = int(np.argmax(ll))
        retained[i] = keep_n
        if return_ll:
            ll_all[i] = ll.astype(np.float32)
    if return_ll:
        return xs, y_true, y_map, ll_all, retained
    return xs, y_true, y_map, retained


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


def avg_prob(logits_list):
    return np.mean(np.stack([softmax_np(z) for z in logits_list], axis=0), axis=0)


def metrics_prob(p, y_true, y_map, ll):
    pred = np.argmax(p, axis=1)
    cand = np.argsort(-p, axis=1)[:, :20]
    truth_in = np.array([y_true[i] in cand[i] for i in range(len(y_true))], dtype=bool)
    map_in = np.array([y_map[i] in cand[i] for i in range(len(y_true))], dtype=bool)
    rerank = np.array([cand[i, int(np.argmax(ll[i, cand[i]]))] for i in range(len(y_true))], dtype=np.int64)
    return {
        "direct_top1": float(np.mean(pred == y_true)),
        "map_agree": float(np.mean(pred == y_map)),
        "exact_map_top1": float(np.mean(y_map == y_true)),
        "truth_top20": float(np.mean(truth_in)),
        "map_top20": float(np.mean(map_in)),
        "rerank20": float(np.mean(rerank == y_true)),
        "rerank20_map_agree": float(np.mean(rerank == y_map)),
    }


def load_fixed_model(seed, lib):
    model = m119.load_old_model(seed, lib)
    ckpt = ROOT / f"nre_119bus_ip1_K60_sensor_v2_seed{seed}_20260702.pt"
    obj = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(obj["model_state"])
    return model


def train_one(seed, lib, xs_train, y_true_train, y_map_train, xs_val, y_true_val, y_map_val, ll_val):
    torch.manual_seed(seed + 660000)
    np.random.seed(seed + 660000)
    model = load_fixed_model(seed, lib)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * int(np.ceil(N_TRAIN / BATCH)), eta_min=8e-6)
    x_train_t = torch.tensor(xs_train, dtype=torch.float32)
    y_true_t = torch.tensor(y_true_train, dtype=torch.long)
    y_map_t = torch.tensor(y_map_train, dtype=torch.long)
    rng = np.random.RandomState(seed + 661000)
    best_score = -1e9
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    best_val = None
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(N_TRAIN)
        model.train()
        for start in range(0, N_TRAIN, BATCH):
            idx = order[start:start + BATCH]
            xb = x_train_t[idx].to(DEVICE)
            yb_true = y_true_t[idx].to(DEVICE)
            yb_map = y_map_t[idx].to(DEVICE)
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb_true) + 0.20 * nn.functional.cross_entropy(logits, yb_map)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
        p_val = softmax_np(predict_logits(model, xs_val))
        val = metrics_prob(p_val, y_true_val, y_map_val, ll_val)
        score = val["direct_top1"] + 0.03 * val["truth_top20"] + 0.02 * val["map_agree"]
        print(f"seed={seed} epoch={epoch}/{EPOCHS} val_direct={val['direct_top1']:.4f} top20={val['truth_top20']:.4f} rerank={val['rerank20']:.4f}", flush=True)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_val = val
    model.load_state_dict(best_state)
    ckpt_name = f"nre_119bus_ip1_K60_dropout_robust_seed{seed}_20260702.pt"
    torch.save({"model_state": model.state_dict(), "seed": seed, "sensors": SENSORS, "drop_train": DROP_TRAIN, "best_epoch": best_epoch, "validation": best_val}, ROOT / ckpt_name)
    return model, ckpt_name, best_epoch, best_val, time.time() - t0


def main():
    t0 = time.time()
    lib = m119.base.load_119_lib()
    xs_train, y_true_train, y_map_train, _ret_train = make_dataset(lib, N_TRAIN, 661001, DROP_TRAIN)
    xs_val, y_true_val, y_map_val, ll_val, _ret_val = make_dataset(lib, N_VAL, 661002, DROP_TRAIN, return_ll=True)
    robust_models = []
    train_rows = []
    for seed in SEEDS:
        model, ckpt, best_epoch, best_val, train_sec = train_one(seed, lib, xs_train, y_true_train, y_map_train, xs_val, y_true_val, y_map_val, ll_val)
        robust_models.append(model)
        train_rows.append((seed, ckpt, best_epoch, best_val, train_sec))
    fixed_models = [load_fixed_model(seed, lib) for seed in SEEDS]

    lines = []
    lines.append("119-bus K60 dropout-robust NRE training")
    lines.append("date=2026-07-02")
    lines.append("role=direct-NRE repair for random sensor dropout; exact/rerank evaluated on retained measurements")
    lines.append(f"device={DEVICE}")
    lines.append(f"sensors={' '.join(map(str, SENSORS))}")
    lines.append(f"drop_train={' '.join(str(x) for x in DROP_TRAIN)}")
    lines.append(f"train_samples={N_TRAIN}")
    lines.append(f"validation_samples={N_VAL}")
    lines.append("seed,best_epoch,val_direct,val_top20,val_rerank,train_sec,checkpoint")
    for seed, ckpt, best_epoch, best_val, train_sec in train_rows:
        lines.append(f"{seed},{best_epoch},{best_val['direct_top1']:.4f},{best_val['truth_top20']:.4f},{best_val['rerank20']:.4f},{train_sec:.1f},{ckpt}")
    lines.append("dropout_eval")
    lines.append("drop_rate,retained_K,model,exact_map_top1,direct_top1,map_agree,truth_top20,map_top20,rerank20,rerank20_map_agree")
    for dr in DROP_EVAL:
        xs_test, y_true_test, y_map_test, ll_test, retained = make_dataset(lib, N_TEST, 662000 + int(dr * 1000), [dr], return_ll=True)
        p_fixed = avg_prob([predict_logits(model, xs_test) for model in fixed_models])
        p_robust = avg_prob([predict_logits(model, xs_test) for model in robust_models])
        for label, p in [("fixed_K60", p_fixed), ("dropout_robust", p_robust)]:
            met = metrics_prob(p, y_true_test, y_map_test, ll_test)
            lines.append(f"{dr:.2f},{int(np.mean(retained))},{label},{met['exact_map_top1']:.4f},{met['direct_top1']:.4f},{met['map_agree']:.4f},{met['truth_top20']:.4f},{met['map_top20']:.4f},{met['rerank20']:.4f},{met['rerank20_map_agree']:.4f}")
    lines.append(f"elapsed_sec={time.time() - t0:.1f}")
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
