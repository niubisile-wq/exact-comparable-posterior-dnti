# -*- coding: utf-8 -*-
"""
300-bus K299 MAP-teacher/posterior-distillation attack.

The previous K299 severe-missing repair achieved very high top-20 coverage but
weak direct top-1. This experiment changes the training target instead of the
wording: warm-start from the frozen K299 checkpoints and train the NRE to match
the exact MAP/posterior teacher under the same K299, 30% missing regime.
Old checkpoints are not overwritten.
"""

from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn as nn

ROOT = Path.home() / "Desktop" / "配电网实验_临时"
sys.path.insert(0, str(ROOT))

import dn_300bus_redundant_sensing_attack_20260702 as m

PKG_ROOT = Path.home() / "Desktop" / "配电网论文一区投稿成果包_20260702"
PKG_STATS = PKG_ROOT / "03_frozen_tables_stats"
PKG_CODE = PKG_ROOT / "02_code"
OUT_NAME = "300bus_K299_mapteacher_attack_20260702.txt"

DEVICE = m.DEVICE
K_ATTACK = 299
MISS_RATE = m.MISS_RATE
BATCH = 1024
N_TRAIN = 90000
N_VAL = 4000
N_TEST = 9000
EPOCHS = 8
LR = 2.5e-5
SEEDS = [42, 123, 456]


def softmax_np(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    p = np.exp(z)
    return p / np.sum(p, axis=1, keepdims=True)


def make_q(ll):
    z = ll - np.max(ll, axis=1, keepdims=True)
    q = np.exp(z)
    q /= np.sum(q, axis=1, keepdims=True)
    return q.astype(np.float32)


def make_dataset(lib, n, seed):
    xs, y_true, y_map, ll = m.make_dataset(lib, n, seed, K_ATTACK, return_ll=True)
    return xs, y_true, y_map, ll, make_q(ll)


def predict_logits(model, xs):
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(xs), BATCH):
            xb = torch.tensor(xs[start:start + BATCH], dtype=torch.float32).to(DEVICE)
            outs.append(model(xb).detach().cpu().numpy())
    return np.vstack(outs)


def avg_prob(models, xs):
    return np.mean(np.stack([softmax_np(predict_logits(model, xs)) for model in models], axis=0), axis=0)


def metrics_prob(p, y_true, y_map, q):
    pred = np.argmax(p, axis=1)
    cand = np.argsort(-p, axis=1)
    out = {
        "exact_map_top1": float(np.mean(y_map == y_true)),
        "direct_true_acc": float(np.mean(pred == y_true)),
        "direct_map_agree": float(np.mean(pred == y_map)),
        "kl_ref_nre": float(np.mean(np.sum(q * (np.log(np.clip(q, 1e-12, 1.0)) - np.log(np.clip(p, 1e-12, 1.0))), axis=1))),
    }
    arrays = {
        "direct_correct": pred == y_true,
        "map_agree": pred == y_map,
    }
    for k in [5, 10, 20, 50]:
        topk = cand[:, :k]
        truth_in = np.array([y_true[i] in topk[i] for i in range(len(y_true))], dtype=bool)
        map_in = np.array([y_map[i] in topk[i] for i in range(len(y_true))], dtype=bool)
        rerank = np.array([topk[i, int(np.argmax(q[i, topk[i]]))] for i in range(len(y_true))], dtype=np.int64)
        out[f"truth_top{k}"] = float(np.mean(truth_in))
        out[f"map_top{k}"] = float(np.mean(map_in))
        out[f"rerank{k}"] = float(np.mean(rerank == y_true))
        arrays[f"truth_top{k}"] = truth_in
        arrays[f"map_top{k}"] = map_in
        arrays[f"rerank{k}"] = rerank == y_true
    return out, arrays


def boot_ci_diff(a, b, boot=1000):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rng = np.random.RandomState(299771)
    vals = np.empty(boot, dtype=np.float64)
    for j in range(boot):
        idx = rng.randint(0, len(a), size=len(a))
        vals[j] = np.mean(a[idx] - b[idx])
    return float(np.mean(a - b)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def load_k299_model(seed, lib):
    n_topos, _, n_bus = lib["V"].shape
    model = m.base.NRE300(n_topos, n_bus).to(DEVICE)
    ckpt = ROOT / f"nre_300bus_ipc_K299_miss30_redundant_seed{seed}_20260702.pt"
    obj = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(obj["model_state"])
    return model


def train_one(seed, lib, xs_train, y_true_train, y_map_train, q_train, xs_val, y_true_val, y_map_val, q_val):
    torch.manual_seed(seed + 299000)
    np.random.seed(seed + 299000)
    model = load_k299_model(seed, lib)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * int(np.ceil(N_TRAIN / BATCH)), eta_min=5e-6)
    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")
    xt = torch.tensor(xs_train, dtype=torch.float32)
    yt = torch.tensor(y_true_train, dtype=torch.long)
    ym = torch.tensor(y_map_train, dtype=torch.long)
    qt = torch.tensor(q_train, dtype=torch.float32)
    rng = np.random.RandomState(seed + 299100)
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
            xb = xt[idx].to(DEVICE)
            ytb = yt[idx].to(DEVICE)
            ymb = ym[idx].to(DEVICE)
            qb = qt[idx].to(DEVICE)
            logits = model(xb)
            loss = 0.85 * ce(logits, ymb) + 0.15 * ce(logits, ytb) + 0.20 * kl(torch.log_softmax(logits, dim=1), qb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
        val_p = softmax_np(predict_logits(model, xs_val))
        val, _ = metrics_prob(val_p, y_true_val, y_map_val, q_val)
        score = val["direct_true_acc"] + 0.05 * val["direct_map_agree"] + 0.02 * val["truth_top20"] - 0.01 * val["kl_ref_nre"]
        print(
            f"seed={seed} epoch={epoch}/{EPOCHS} val_direct={val['direct_true_acc']:.4f} "
            f"map_agree={val['direct_map_agree']:.4f} top20={val['truth_top20']:.4f} kl={val['kl_ref_nre']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    ckpt_name = f"nre_300bus_ipc_K299_miss30_mapteacher_seed{seed}_20260702.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "seed": seed,
            "K": K_ATTACK,
            "missing_rate": MISS_RATE,
            "loss": "0.85*CE(exact_MAP)+0.15*CE(true)+0.20*KL(exact_posterior)",
            "best_epoch": best_epoch,
            "validation": best_val,
        },
        ROOT / ckpt_name,
    )
    return model, ckpt_name, best_epoch, best_val, time.time() - t0


def main():
    t0 = time.time()
    lib = m.base.load_300_lib()
    xs_train, y_true_train, y_map_train, _ll_train, q_train = make_dataset(lib, N_TRAIN, 299501)
    xs_val, y_true_val, y_map_val, _ll_val, q_val = make_dataset(lib, N_VAL, 299502)
    xs_test, y_true_test, y_map_test, _ll_test, q_test = make_dataset(lib, N_TEST, 299503)

    old_models = [load_k299_model(seed, lib) for seed in SEEDS]
    old_p = avg_prob(old_models, xs_test)
    old_met, old_arr = metrics_prob(old_p, y_true_test, y_map_test, q_test)

    new_models = []
    rows = []
    for seed in SEEDS:
        model, ckpt, best_epoch, best_val, train_sec = train_one(
            seed,
            lib,
            xs_train,
            y_true_train,
            y_map_train,
            q_train,
            xs_val,
            y_true_val,
            y_map_val,
            q_val,
        )
        new_models.append(model)
        rows.append((seed, ckpt, best_epoch, best_val, train_sec))
    new_p = avg_prob(new_models, xs_test)
    new_met, new_arr = metrics_prob(new_p, y_true_test, y_map_test, q_test)

    gain_direct = boot_ci_diff(new_arr["direct_correct"], old_arr["direct_correct"])
    gain_map = boot_ci_diff(new_arr["map_agree"], old_arr["map_agree"])
    gain_top20 = boot_ci_diff(new_arr["truth_top20"], old_arr["truth_top20"])
    gain_rerank20 = boot_ci_diff(new_arr["rerank20"], old_arr["rerank20"])

    lines = []
    lines.append("300-bus K299 MAP-teacher/posterior-distillation attack")
    lines.append("date=2026-07-02")
    lines.append("role=experiment-level attempt to repair severe-missing direct NRE by training toward exact MAP/posterior teacher")
    lines.append("comparison=frozen K299 redundant checkpoints vs new MAP-teacher warm-start checkpoints on identical final draw")
    lines.append(f"device={DEVICE}")
    lines.append(f"K={K_ATTACK}")
    lines.append(f"missing_rate={MISS_RATE:.2f}")
    lines.append(f"observed_after_missing={K_ATTACK - int(K_ATTACK * MISS_RATE)}")
    lines.append(f"train_samples={N_TRAIN}")
    lines.append(f"validation_samples={N_VAL}")
    lines.append(f"test_samples={N_TEST}")
    lines.append("loss=0.85*CE(exact_MAP)+0.15*CE(true)+0.20*KL(exact_posterior)")
    lines.append("seed,best_epoch,val_exact_map_top1,val_direct_true,val_direct_map_agree,val_top20,val_rerank20,val_kl,train_sec,checkpoint")
    for seed, ckpt, best_epoch, best_val, train_sec in rows:
        lines.append(
            f"{seed},{best_epoch},{best_val['exact_map_top1']:.4f},{best_val['direct_true_acc']:.4f},"
            f"{best_val['direct_map_agree']:.4f},{best_val['truth_top20']:.4f},{best_val['rerank20']:.4f},"
            f"{best_val['kl_ref_nre']:.4f},{train_sec:.1f},{ckpt}"
        )
    lines.append("test_comparison")
    lines.append("model,exact_map_top1,direct_true_acc,direct_map_agree,kl,truth_top5,truth_top10,truth_top20,truth_top50,map_top20,rerank5,rerank10,rerank20,rerank50")
    for label, met in [("frozen_K299", old_met), ("mapteacher_K299", new_met)]:
        lines.append(
            f"{label},{met['exact_map_top1']:.4f},{met['direct_true_acc']:.4f},{met['direct_map_agree']:.4f},"
            f"{met['kl_ref_nre']:.4f},{met['truth_top5']:.4f},{met['truth_top10']:.4f},{met['truth_top20']:.4f},"
            f"{met['truth_top50']:.4f},{met['map_top20']:.4f},{met['rerank5']:.4f},{met['rerank10']:.4f},"
            f"{met['rerank20']:.4f},{met['rerank50']:.4f}"
        )
    lines.append("bootstrap_gain_mapteacher_minus_frozen")
    lines.append("metric,mean,ci95_low,ci95_high")
    for key, val in [
        ("direct_true_acc", gain_direct),
        ("direct_map_agree", gain_map),
        ("truth_top20", gain_top20),
        ("rerank20", gain_rerank20),
    ]:
        lines.append(f"{key},{val[0]:.4f},{val[1]:.4f},{val[2]:.4f}")
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
