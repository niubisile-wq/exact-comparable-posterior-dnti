# -*- coding: utf-8 -*-
"""
IEEE123 K75 dropout-robust three-phase NRE training.

Repairs the direct-NRE random sensor-dropout weakness for the controlled
unbalanced IEEE123 benchmark. Warm-starts from K75 checkpoints and trains with
random retained subsets of the K75 deployment. No old checkpoints are overwritten.
"""

from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn

import dn_ieee123_3ph_K75_warmstart_20260702 as m123

ROOT = Path.home() / "Desktop" / "配电网实验_临时"
PKG_ROOT = Path.home() / "Desktop" / "配电网论文一区投稿成果包_20260702"
PKG_STATS = PKG_ROOT / "03_frozen_tables_stats"
PKG_CODE = PKG_ROOT / "02_code"
OUT_NAME = "ieee123_K75_dropout_robust_train_20260702.txt"
DEVICE = m123.DEVICE

DROP_TRAIN = [0.0, 0.10, 0.20, 0.40]
DROP_EVAL = [0.0, 0.10, 0.20, 0.40]
SEEDS = [42, 123, 456]
K_BASE = 75
BATCH = 256
TRAIN_STEPS = 4200
LR = 2.5e-4
N_VAL = 900
N_TEST = 1200
SIGMA = m123.SIGMA


def make_dataset(lib, n, seed, drop_rates):
    rng = np.random.RandomState(seed)
    V = lib["v"]
    n_topos, n_lf, n_bus, _ = V.shape
    deploy = m123.deployment(n_bus, K_BASE)
    xs = np.zeros((n, n_bus * 5), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    q = np.zeros((n, n_topos), dtype=np.float32)
    retained = np.zeros(n, dtype=np.int64)
    for i in range(n):
        dr = float(drop_rates[rng.randint(0, len(drop_rates))])
        keep_n = max(1, int(round(len(deploy) * (1.0 - dr))))
        buses = np.sort(rng.choice(deploy, keep_n, replace=False))
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        lf = float(m123.LF_GRID[lf_idx])
        obs = V[ti, lf_idx, buses, :] + rng.normal(0.0, SIGMA, size=(len(buses), 3))
        xs[i] = m123.build_features(obs, buses, n_bus, lib["base_p_norm"], lf)
        y[i] = ti
        q[i] = m123.exact_posterior(V, buses, obs, lf_idx)
        retained[i] = keep_n
    return xs, y, q, retained


def predict_probs(model, xs):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(xs), BATCH):
            xb = torch.tensor(xs[start:start+BATCH], dtype=torch.float32, device=DEVICE)
            chunks.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.vstack(chunks)


def metrics(p, y, q):
    exact_pred = np.argmax(q, axis=1)
    direct = np.argmax(p, axis=1)
    top = np.argsort(-p, axis=1)
    out = {
        "exact_top1": float(np.mean(exact_pred == y)),
        "direct_top1": float(np.mean(direct == y)),
        "exact_agree": float(np.mean(direct == exact_pred)),
        "kl_ref_nre": float(np.mean(np.sum(q * (np.log(np.clip(q, 1e-12, 1.0)) - np.log(np.clip(p, 1e-12, 1.0))), axis=1))),
    }
    for k in [3, 5, 10, 20]:
        cand = top[:, :k]
        out[f"truth_top{k}"] = float(np.mean([y[i] in cand[i] for i in range(len(y))]))
        out[f"exact_top{k}"] = float(np.mean([exact_pred[i] in cand[i] for i in range(len(y))]))
        rerank = np.array([cand[i, int(np.argmax(q[i, cand[i]]))] for i in range(len(y))], dtype=np.int64)
        out[f"rerank{k}"] = float(np.mean(rerank == y))
    return out


def load_fixed_model(lib, seed):
    model = m123.Controlled3PhNRE(lib["n_topos"], lib["n_bus"]).to(DEVICE)
    ckpt = ROOT / f"nre_ieee123_3ph_K75_warmstart_seed{seed}_20260702.pt"
    obj = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(obj["model_state"])
    return model


def train_one(lib, seed, xs_val, y_val, q_val):
    torch.manual_seed(seed + 750000)
    np.random.seed(seed + 750000)
    rng = np.random.RandomState(seed + 751000)
    n_topos = lib["n_topos"]
    n_bus = lib["n_bus"]
    deploy = m123.deployment(n_bus, K_BASE)
    model = load_fixed_model(lib, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_STEPS, eta_min=1e-5)
    kl_fn = nn.KLDivLoss(reduction="batchmean")
    ce_fn = nn.CrossEntropyLoss()
    best_score = -1e9
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_step = 0
    best_val = None
    t0 = time.time()
    for step in range(1, TRAIN_STEPS + 1):
        xs = np.zeros((BATCH, n_bus * 5), dtype=np.float32)
        ys = np.zeros(BATCH, dtype=np.int64)
        qs = np.zeros((BATCH, n_topos), dtype=np.float32)
        for i in range(BATCH):
            dr = float(DROP_TRAIN[rng.randint(0, len(DROP_TRAIN))])
            keep_n = max(1, int(round(len(deploy) * (1.0 - dr))))
            buses = np.sort(rng.choice(deploy, keep_n, replace=False))
            ti = rng.randint(0, n_topos)
            lf_idx = rng.randint(0, len(m123.LF_GRID))
            lf = float(m123.LF_GRID[lf_idx])
            obs = lib["v"][ti, lf_idx, buses, :] + rng.normal(0.0, SIGMA, size=(len(buses), 3))
            xs[i] = m123.build_features(obs, buses, n_bus, lib["base_p_norm"], lf)
            ys[i] = ti
            qs[i] = m123.exact_posterior(lib["v"], buses, obs, lf_idx)
        xb = torch.tensor(xs, dtype=torch.float32, device=DEVICE)
        yb = torch.tensor(ys, dtype=torch.long, device=DEVICE)
        qb = torch.tensor(qs, dtype=torch.float32, device=DEVICE)
        logits = model(xb)
        loss = kl_fn(torch.log_softmax(logits, dim=1), qb) + 0.20 * ce_fn(logits, yb)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
        if step % 700 == 0 or step == TRAIN_STEPS:
            p_val = predict_probs(model, xs_val)
            val = metrics(p_val, y_val, q_val)
            score = val["direct_top1"] + 0.03 * val["truth_top20"] - 0.01 * val["kl_ref_nre"]
            print(f"seed={seed} step={step}/{TRAIN_STEPS} val_direct={val['direct_top1']:.4f} top20={val['truth_top20']:.4f} rerank20={val['rerank20']:.4f}", flush=True)
            if score > best_score:
                best_score = score
                best_step = step
                best_val = val
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    ckpt_name = f"nre_ieee123_3ph_K75_dropout_robust_seed{seed}_20260702.pt"
    torch.save({"model_state": model.state_dict(), "seed": seed, "K_base": K_BASE, "drop_train": DROP_TRAIN, "best_step": best_step, "validation": best_val}, ROOT / ckpt_name)
    return model, ckpt_name, best_step, best_val, time.time() - t0


def avg_probs(models, xs):
    return np.mean(np.stack([predict_probs(model, xs) for model in models], axis=0), axis=0)


def main():
    t0 = time.time()
    lib = m123.load_library()
    xs_val, y_val, q_val, _ret = make_dataset(lib, N_VAL, 752001, DROP_TRAIN)
    robust_models = []
    train_rows = []
    for seed in SEEDS:
        model, ckpt, best_step, best_val, train_sec = train_one(lib, seed, xs_val, y_val, q_val)
        robust_models.append(model)
        train_rows.append((seed, ckpt, best_step, best_val, train_sec))
    fixed_models = [load_fixed_model(lib, seed) for seed in SEEDS]

    lines = []
    lines.append("IEEE123 K75 dropout-robust three-phase NRE training")
    lines.append("date=2026-07-02")
    lines.append("role=direct-NRE repair for random sensor dropout on controlled unbalanced IEEE123 benchmark")
    lines.append(f"device={DEVICE}")
    lines.append(f"base_K={K_BASE}")
    lines.append(f"drop_train={' '.join(str(x) for x in DROP_TRAIN)}")
    lines.append(f"validation_samples={N_VAL}")
    lines.append("seed,best_step,val_direct,val_top20,val_rerank20,val_kl,train_sec,checkpoint")
    for seed, ckpt, best_step, best_val, train_sec in train_rows:
        lines.append(f"{seed},{best_step},{best_val['direct_top1']:.4f},{best_val['truth_top20']:.4f},{best_val['rerank20']:.4f},{best_val['kl_ref_nre']:.4f},{train_sec:.1f},{ckpt}")
    lines.append("dropout_eval")
    lines.append("drop_rate,retained_K,model,exact_top1,direct_top1,exact_agree,kl,truth_top3,truth_top5,truth_top10,truth_top20,rerank3,rerank5,rerank10,rerank20")
    for dr in DROP_EVAL:
        xs_test, y_test, q_test, retained = make_dataset(lib, N_TEST, 753000 + int(dr * 1000), [dr])
        p_fixed = avg_probs(fixed_models, xs_test)
        p_robust = avg_probs(robust_models, xs_test)
        for label, p in [("fixed_K75", p_fixed), ("dropout_robust", p_robust)]:
            met = metrics(p, y_test, q_test)
            lines.append(f"{dr:.2f},{int(np.mean(retained))},{label},{met['exact_top1']:.4f},{met['direct_top1']:.4f},{met['exact_agree']:.4f},{met['kl_ref_nre']:.4f},{met['truth_top3']:.4f},{met['truth_top5']:.4f},{met['truth_top10']:.4f},{met['truth_top20']:.4f},{met['rerank3']:.4f},{met['rerank5']:.4f},{met['rerank10']:.4f},{met['rerank20']:.4f}")
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
