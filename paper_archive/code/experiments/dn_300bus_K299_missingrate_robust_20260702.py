# -*- coding: utf-8 -*-
"""
300-bus K299 missing-rate-robust NRE training.

The severe 30% missing K299 case is information-limited. This experiment
strengthens the experimental story by training one K299 model across a missing
rate curriculum (0/10/20/30/40%) and reporting a full operating curve against
the frozen K299 model. Old checkpoints are not overwritten.
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
OUT_NAME = "300bus_K299_missingrate_robust_20260702.txt"

DEVICE = m.DEVICE
K_ATTACK = 299
BATCH = 1024
MISS_TRAIN = [0.0, 0.10, 0.20, 0.30, 0.40]
MISS_EVAL = [0.0, 0.10, 0.20, 0.30, 0.40]
N_TRAIN = 100000
N_VAL = 4500
N_TEST = 6500
EPOCHS = 7
LR = 3e-5
SIGMA = m.SIGMA
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


def make_dataset(lib, n, seed, miss_rates):
    rng = np.random.RandomState(seed)
    V = lib["V"]
    lf_grid = lib["lf"]
    base_p = lib["base_p"]
    n_topos, n_lf, n_bus = V.shape
    deploy = m.base.deployment_sensors(n_bus, K_ATTACK)
    xs = np.zeros((n, n_bus * 3), dtype=np.float32)
    y_true = np.zeros(n, dtype=np.int64)
    y_map = np.zeros(n, dtype=np.int64)
    ll_all = np.zeros((n, n_topos), dtype=np.float32)
    retained = np.zeros(n, dtype=np.int64)
    used_miss = np.zeros(n, dtype=np.float32)
    for i in range(n):
        mr = float(miss_rates[rng.randint(0, len(miss_rates))])
        n_miss = int(K_ATTACK * mr)
        n_obs = K_ATTACK - n_miss
        obs_idx = np.sort(rng.choice(K_ATTACK, n_obs, replace=False))
        buses = deploy[obs_idx]
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        lf = float(lf_grid[lf_idx])
        obs = V[ti, lf_idx, buses] + rng.normal(0.0, SIGMA, size=len(buses))
        pred = V[:, lf_idx, :][:, buses]
        ll = -0.5 * np.sum(((pred - obs[None, :]) / SIGMA) ** 2, axis=1)
        xs[i, buses] = (obs - 1.0) / SIGMA
        xs[i, n_bus + buses] = 1.0
        xs[i, 2 * n_bus + buses] = base_p[buses] * lf
        y_true[i] = ti
        y_map[i] = int(np.argmax(ll))
        ll_all[i] = ll.astype(np.float32)
        retained[i] = n_obs
        used_miss[i] = mr
    return xs, y_true, y_map, ll_all, make_q(ll_all), retained, used_miss


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
    rng = np.random.RandomState(299884)
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
    torch.manual_seed(seed + 299800)
    np.random.seed(seed + 299800)
    model = load_k299_model(seed, lib)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * int(np.ceil(N_TRAIN / BATCH)), eta_min=5e-6)
    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")
    xt = torch.tensor(xs_train, dtype=torch.float32)
    yt = torch.tensor(y_true_train, dtype=torch.long)
    ym = torch.tensor(y_map_train, dtype=torch.long)
    qt = torch.tensor(q_train, dtype=torch.float32)
    rng = np.random.RandomState(seed + 299900)
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
            loss = 0.75 * ce(logits, ytb) + 0.25 * ce(logits, ymb) + 0.15 * kl(torch.log_softmax(logits, dim=1), qb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
        val_p = softmax_np(predict_logits(model, xs_val))
        val, _ = metrics_prob(val_p, y_true_val, y_map_val, q_val)
        score = val["direct_true_acc"] + 0.03 * val["truth_top20"] + 0.02 * val["direct_map_agree"] - 0.01 * val["kl_ref_nre"]
        print(
            f"seed={seed} epoch={epoch}/{EPOCHS} val_direct={val['direct_true_acc']:.4f} "
            f"top20={val['truth_top20']:.4f} rerank20={val['rerank20']:.4f} kl={val['kl_ref_nre']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    ckpt_name = f"nre_300bus_ipc_K299_missingrate_robust_seed{seed}_20260702.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "seed": seed,
            "K": K_ATTACK,
            "missing_train": MISS_TRAIN,
            "loss": "0.75*CE(true)+0.25*CE(exact_MAP)+0.15*KL(exact_posterior)",
            "best_epoch": best_epoch,
            "validation": best_val,
        },
        ROOT / ckpt_name,
    )
    return model, ckpt_name, best_epoch, best_val, time.time() - t0


def main():
    t0 = time.time()
    lib = m.base.load_300_lib()
    xs_train, y_true_train, y_map_train, _ll_train, q_train, _ret_train, _mr_train = make_dataset(lib, N_TRAIN, 299801, MISS_TRAIN)
    xs_val, y_true_val, y_map_val, _ll_val, q_val, _ret_val, _mr_val = make_dataset(lib, N_VAL, 299802, MISS_TRAIN)
    fixed_models = [load_k299_model(seed, lib) for seed in SEEDS]
    robust_models = []
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
        robust_models.append(model)
        rows.append((seed, ckpt, best_epoch, best_val, train_sec))

    eval_rows = []
    for mr in MISS_EVAL:
        xs_test, y_true_test, y_map_test, _ll_test, q_test, retained, _mr = make_dataset(lib, N_TEST, 299900 + int(mr * 1000), [mr])
        p_fixed = avg_prob(fixed_models, xs_test)
        p_robust = avg_prob(robust_models, xs_test)
        fixed_met, fixed_arr = metrics_prob(p_fixed, y_true_test, y_map_test, q_test)
        robust_met, robust_arr = metrics_prob(p_robust, y_true_test, y_map_test, q_test)
        direct_gain = boot_ci_diff(robust_arr["direct_correct"], fixed_arr["direct_correct"])
        top20_gain = boot_ci_diff(robust_arr["truth_top20"], fixed_arr["truth_top20"])
        rerank20_gain = boot_ci_diff(robust_arr["rerank20"], fixed_arr["rerank20"])
        eval_rows.append((mr, int(np.mean(retained)), "fixed_K299", fixed_met, None, None, None))
        eval_rows.append((mr, int(np.mean(retained)), "missingrate_robust", robust_met, direct_gain, top20_gain, rerank20_gain))

    lines = []
    lines.append("300-bus K299 missing-rate-robust NRE training")
    lines.append("date=2026-07-02")
    lines.append("role=experiment-level operating-curve attack for large-system missing-measurement degradation")
    lines.append("comparison=frozen 30%-missing K299 checkpoints vs missing-rate-robust warm-start checkpoints")
    lines.append(f"device={DEVICE}")
    lines.append(f"K={K_ATTACK}")
    lines.append(f"missing_train={' '.join(str(x) for x in MISS_TRAIN)}")
    lines.append(f"train_samples={N_TRAIN}")
    lines.append(f"validation_samples={N_VAL}")
    lines.append(f"test_samples_per_missing_rate={N_TEST}")
    lines.append("loss=0.75*CE(true)+0.25*CE(exact_MAP)+0.15*KL(exact_posterior)")
    lines.append("seed,best_epoch,val_exact_map_top1,val_direct_true,val_direct_map_agree,val_top20,val_rerank20,val_kl,train_sec,checkpoint")
    for seed, ckpt, best_epoch, best_val, train_sec in rows:
        lines.append(
            f"{seed},{best_epoch},{best_val['exact_map_top1']:.4f},{best_val['direct_true_acc']:.4f},"
            f"{best_val['direct_map_agree']:.4f},{best_val['truth_top20']:.4f},{best_val['rerank20']:.4f},"
            f"{best_val['kl_ref_nre']:.4f},{train_sec:.1f},{ckpt}"
        )
    lines.append("missing_rate_eval")
    lines.append("missing_rate,retained_K,model,exact_map_top1,direct_true_acc,direct_map_agree,kl,truth_top5,truth_top10,truth_top20,truth_top50,map_top20,rerank5,rerank10,rerank20,rerank50,direct_gain_ci,top20_gain_ci,rerank20_gain_ci")
    for mr, retained, label, met, direct_gain, top20_gain, rerank20_gain in eval_rows:
        if direct_gain is None:
            dg = tg = rg = ""
        else:
            dg = f"{direct_gain[0]:.4f}[{direct_gain[1]:.4f},{direct_gain[2]:.4f}]"
            tg = f"{top20_gain[0]:.4f}[{top20_gain[1]:.4f},{top20_gain[2]:.4f}]"
            rg = f"{rerank20_gain[0]:.4f}[{rerank20_gain[1]:.4f},{rerank20_gain[2]:.4f}]"
        lines.append(
            f"{mr:.2f},{retained},{label},{met['exact_map_top1']:.4f},{met['direct_true_acc']:.4f},"
            f"{met['direct_map_agree']:.4f},{met['kl_ref_nre']:.4f},{met['truth_top5']:.4f},{met['truth_top10']:.4f},"
            f"{met['truth_top20']:.4f},{met['truth_top50']:.4f},{met['map_top20']:.4f},{met['rerank5']:.4f},"
            f"{met['rerank10']:.4f},{met['rerank20']:.4f},{met['rerank50']:.4f},{dg},{tg},{rg}"
        )
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
