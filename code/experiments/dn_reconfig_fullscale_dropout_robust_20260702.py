# -*- coding: utf-8 -*-
"""
Full-scale reconfiguration dropout-robust NRE training.

This repairs the no-retraining random sensor-dropout weakness exposed for the
balanced 202/417 full-scale stress track. It warm-starts from the frozen
K160/K220 full-scale checkpoints and trains on random retained-sensor subsets.
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

import dn_reconfig_fullscale_attack_20260702 as base

PKG_ROOT = Path.home() / "Desktop" / "配电网论文一区投稿成果包_20260702"
PKG_STATS = PKG_ROOT / "03_frozen_tables_stats"
PKG_CODE = PKG_ROOT / "02_code"
OUT_NAME = "reconfig_fullscale_dropout_robust_20260702.txt"

DEVICE = base.DEVICE
SIGMA = base.SIGMA
BATCH = 512
DROP_TRAIN = [0.0, 0.10, 0.20, 0.40]
DROP_EVAL = [0.0, 0.10, 0.20, 0.40]

SYSTEMS = [
    {
        "name": "SystemData_202",
        "K": 160,
        "seeds": [202, 1202],
        "train_n": 36000,
        "val_n": 1800,
        "test_n": 2600,
        "epochs": 6,
        "lr": 4e-5,
    },
    {
        "name": "SystemData_417",
        "K": 220,
        "seeds": [417, 1417],
        "train_n": 34000,
        "val_n": 1800,
        "test_n": 2600,
        "epochs": 6,
        "lr": 4e-5,
    },
]


def softmax_np(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    p = np.exp(z)
    return p / np.sum(p, axis=1, keepdims=True)


def make_dropout_dataset(lib, k, n, seed, drop_rates):
    rng = np.random.RandomState(seed)
    V = lib["V"]
    n_topos, n_lf, n_bus = V.shape
    full_buses = base.deployment(n_bus, k)
    xs = np.zeros((n, n_bus * 3), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    q = np.zeros((n, n_topos), dtype=np.float32)
    retained = np.zeros(n, dtype=np.int64)
    for i in range(n):
        dr = float(drop_rates[rng.randint(0, len(drop_rates))])
        keep_n = max(1, int(round(k * (1.0 - dr))))
        buses = np.sort(rng.choice(full_buses, keep_n, replace=False))
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        lf = lib["lf"][lf_idx]
        obs = V[ti, lf_idx, buses] + rng.normal(0.0, SIGMA, size=len(buses))
        xs[i, buses] = (obs - 1.0) / SIGMA
        xs[i, n_bus + buses] = 1.0
        xs[i, 2 * n_bus + buses] = lib["base_p"][buses] * lf
        y[i] = ti
        q[i] = base.exact_posterior(lib, buses, obs, lf_idx)
        retained[i] = keep_n
    return xs, y, q, retained


def predict_probs(model, xs):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(xs), BATCH):
            xb = torch.tensor(xs[start:start + BATCH], dtype=torch.float32, device=DEVICE)
            chunks.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.vstack(chunks)


def predict_logits(model, xs):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(xs), BATCH):
            xb = torch.tensor(xs[start:start + BATCH], dtype=torch.float32, device=DEVICE)
            chunks.append(model(xb).detach().cpu().numpy())
    return np.vstack(chunks)


def ensemble_probs(models, xs):
    return np.mean(np.stack([predict_probs(m, xs) for m in models], axis=0), axis=0)


def metrics_from_probs(p, y, q):
    exact_pred = np.argmax(q, axis=1)
    direct = np.argmax(p, axis=1)
    top = np.argsort(-p, axis=1)
    out = {
        "exact_top1": float(np.mean(exact_pred == y)),
        "direct_top1": float(np.mean(direct == y)),
        "exact_agree": float(np.mean(direct == exact_pred)),
        "kl_ref_nre": float(np.mean(np.sum(q * (np.log(np.clip(q, 1e-12, 1.0)) - np.log(np.clip(p, 1e-12, 1.0))), axis=1))),
    }
    arrays = {
        "direct_correct": direct == y,
        "exact_correct": exact_pred == y,
    }
    for m in [5, 10, 20]:
        cand = top[:, :m]
        truth_in = np.array([y[i] in cand[i] for i in range(len(y))], dtype=bool)
        exact_in = np.array([exact_pred[i] in cand[i] for i in range(len(y))], dtype=bool)
        rerank = np.array([cand[i, int(np.argmax(q[i, cand[i]]))] for i in range(len(y))], dtype=np.int64)
        out[f"truth_top{m}"] = float(np.mean(truth_in))
        out[f"exact_top{m}"] = float(np.mean(exact_in))
        out[f"rerank{m}"] = float(np.mean(rerank == y))
        arrays[f"truth_top{m}"] = truth_in
        arrays[f"exact_top{m}"] = exact_in
        arrays[f"rerank{m}"] = rerank == y
    return out, arrays


def boot_ci_diff(a, b, boot=1000):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rng = np.random.RandomState(77031)
    vals = np.empty(boot, dtype=np.float64)
    for j in range(boot):
        idx = rng.randint(0, len(a), size=len(a))
        vals[j] = np.mean(a[idx] - b[idx])
    return float(np.mean(a - b)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def load_fixed_model(lib, cfg, seed):
    n_topos, _, n_bus = lib["V"].shape
    model = base.src.StressNRE(n_topos, n_bus).to(DEVICE)
    ckpt = ROOT / f"nre_reconfig_{cfg['name']}_K{cfg['K']}_seed{seed}_20260702.pt"
    obj = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(obj["model_state"])
    return model


def train_one(lib, cfg, seed, xs_train, y_train, q_train, xs_val, y_val, q_val):
    torch.manual_seed(seed + 990000)
    np.random.seed(seed + 990000)
    model = load_fixed_model(lib, cfg, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    steps = cfg["epochs"] * int(np.ceil(len(xs_train) / BATCH))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=8e-6)
    kl_fn = nn.KLDivLoss(reduction="batchmean")
    ce_fn = nn.CrossEntropyLoss()
    xt = torch.tensor(xs_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    qt = torch.tensor(q_train, dtype=torch.float32)
    rng = np.random.RandomState(seed + 991000)
    best_score = -1e9
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    best_val = None
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        order = rng.permutation(len(xs_train))
        model.train()
        for start in range(0, len(xs_train), BATCH):
            idx = order[start:start + BATCH]
            xb = xt[idx].to(DEVICE)
            yb = yt[idx].to(DEVICE)
            qb = qt[idx].to(DEVICE)
            logits = model(xb)
            loss = kl_fn(torch.log_softmax(logits, dim=1), qb) + 0.15 * ce_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
        p_val = predict_probs(model, xs_val)
        val, _ = metrics_from_probs(p_val, y_val, q_val)
        score = val["direct_top1"] + 0.03 * val["truth_top20"] + 0.02 * val["exact_agree"] - 0.01 * val["kl_ref_nre"]
        print(
            f"{cfg['name']} seed={seed} epoch={epoch}/{cfg['epochs']} "
            f"val_direct={val['direct_top1']:.4f} top20={val['truth_top20']:.4f} rerank20={val['rerank20']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    ckpt_name = f"nre_reconfig_{cfg['name']}_K{cfg['K']}_dropout_robust_seed{seed}_20260702.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "system": cfg["name"],
            "K": cfg["K"],
            "seed": seed,
            "drop_train": DROP_TRAIN,
            "best_epoch": best_epoch,
            "validation": best_val,
        },
        ROOT / ckpt_name,
    )
    return model, ckpt_name, best_epoch, best_val, time.time() - t0


def run_system(cfg):
    lib = base.load_or_build(cfg["name"])
    xs_train, y_train, q_train, _ = make_dropout_dataset(lib, cfg["K"], cfg["train_n"], cfg["K"] * 9000 + 1, DROP_TRAIN)
    xs_val, y_val, q_val, _ = make_dropout_dataset(lib, cfg["K"], cfg["val_n"], cfg["K"] * 9000 + 2, DROP_TRAIN)
    robust_models = []
    train_rows = []
    for seed in cfg["seeds"]:
        model, ckpt, best_epoch, best_val, train_sec = train_one(lib, cfg, seed, xs_train, y_train, q_train, xs_val, y_val, q_val)
        robust_models.append(model)
        train_rows.append((seed, ckpt, best_epoch, best_val, train_sec))
    fixed_models = [load_fixed_model(lib, cfg, seed) for seed in cfg["seeds"]]
    eval_rows = []
    for dr in DROP_EVAL:
        xs_test, y_test, q_test, retained = make_dropout_dataset(lib, cfg["K"], cfg["test_n"], cfg["K"] * 9100 + int(dr * 1000), [dr])
        p_fixed = ensemble_probs(fixed_models, xs_test)
        p_robust = ensemble_probs(robust_models, xs_test)
        fixed_met, fixed_arr = metrics_from_probs(p_fixed, y_test, q_test)
        robust_met, robust_arr = metrics_from_probs(p_robust, y_test, q_test)
        direct_gain = boot_ci_diff(robust_arr["direct_correct"], fixed_arr["direct_correct"])
        top20_gain = boot_ci_diff(robust_arr["truth_top20"], fixed_arr["truth_top20"])
        rerank20_gain = boot_ci_diff(robust_arr["rerank20"], fixed_arr["rerank20"])
        eval_rows.append((dr, int(np.mean(retained)), "fixed", fixed_met, None, None, None))
        eval_rows.append((dr, int(np.mean(retained)), "dropout_robust", robust_met, direct_gain, top20_gain, rerank20_gain))
    return lib, train_rows, eval_rows


def main():
    t0 = time.time()
    lines = []
    lines.append("Full-scale reconfiguration dropout-robust NRE training")
    lines.append("date=2026-07-02")
    lines.append("role=experimental repair for random sensor dropout on balanced full-scale 202/417 stress benchmarks")
    lines.append("comparison=frozen fixed K160/K220 full-scale checkpoints vs dropout-robust warm-start models")
    lines.append("not_claimed=three-phase utility deployment or private field validation")
    lines.append(f"device={DEVICE}")
    lines.append(f"drop_train={' '.join(str(x) for x in DROP_TRAIN)}")
    for cfg in SYSTEMS:
        lib, train_rows, eval_rows = run_system(cfg)
        lines.append(f"system={cfg['name']}")
        lines.append(f"n_bus={lib['n_bus']}")
        lines.append(f"n_topologies={lib['n_topologies']}")
        lines.append(f"attack_K={cfg['K']}")
        lines.append(f"train_samples={cfg['train_n']}")
        lines.append(f"validation_samples={cfg['val_n']}")
        lines.append(f"test_samples_per_dropout={cfg['test_n']}")
        lines.append("seed,best_epoch,val_exact,val_direct,val_top20,val_rerank20,val_kl,train_sec,checkpoint")
        for seed, ckpt, best_epoch, best_val, train_sec in train_rows:
            lines.append(
                f"{seed},{best_epoch},{best_val['exact_top1']:.4f},{best_val['direct_top1']:.4f},"
                f"{best_val['truth_top20']:.4f},{best_val['rerank20']:.4f},{best_val['kl_ref_nre']:.4f},"
                f"{train_sec:.1f},{ckpt}"
            )
        lines.append("dropout_eval")
        lines.append("drop_rate,retained_K,model,exact_top1,direct_top1,exact_agree,kl,truth_top5,truth_top10,truth_top20,rerank5,rerank10,rerank20,direct_gain_ci,top20_gain_ci,rerank20_gain_ci")
        for dr, retained, label, met, direct_gain, top20_gain, rerank20_gain in eval_rows:
            if direct_gain is None:
                dg = tg = rg = ""
            else:
                dg = f"{direct_gain[0]:.4f}[{direct_gain[1]:.4f},{direct_gain[2]:.4f}]"
                tg = f"{top20_gain[0]:.4f}[{top20_gain[1]:.4f},{top20_gain[2]:.4f}]"
                rg = f"{rerank20_gain[0]:.4f}[{rerank20_gain[1]:.4f},{rerank20_gain[2]:.4f}]"
            lines.append(
                f"{dr:.2f},{retained},{label},{met['exact_top1']:.4f},{met['direct_top1']:.4f},"
                f"{met['exact_agree']:.4f},{met['kl_ref_nre']:.4f},{met['truth_top5']:.4f},"
                f"{met['truth_top10']:.4f},{met['truth_top20']:.4f},{met['rerank5']:.4f},"
                f"{met['rerank10']:.4f},{met['rerank20']:.4f},{dg},{tg},{rg}"
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
