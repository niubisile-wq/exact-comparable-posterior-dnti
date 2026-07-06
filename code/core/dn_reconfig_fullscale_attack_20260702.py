# -*- coding: utf-8 -*-
"""
Full-scale reconfiguration exact-comparable attack experiment.

Strengthens the 202/417 balanced full-scale stress track with higher sensor
budgets, NRE top-k candidate coverage, exact reranking, and bootstrap CIs.
Uses local reconfiguration assets and caches voltage libraries. This is still a
balanced stress track, not an unbalanced utility field deployment.
"""

from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn

import dn_reconfig_exactpilot as src

ROOT = Path.home() / "Desktop" / "配电网实验_临时"
PKG_ROOT = Path.home() / "Desktop" / "配电网论文一区投稿成果包_20260702"
PKG_STATS = PKG_ROOT / "03_frozen_tables_stats"
PKG_CODE = PKG_ROOT / "02_code"
OUT_NAME = "reconfig_fullscale_attack_20260702.txt"
DEVICE = src.DEVICE
SIGMA = src.SIGMA
BATCH = 256
BOOT = 1000

SYSTEMS = [
    {"name": "SystemData_202", "K": 160, "curve": [90, 120, 160], "train_n": 26000, "val_n": 1000, "test_n": 1800, "epochs": 8, "seeds": [202, 1202]},
    {"name": "SystemData_417", "K": 220, "curve": [150, 180, 220], "train_n": 24000, "val_n": 1000, "test_n": 1800, "epochs": 8, "seeds": [417, 1417]},
]


def cache_path(name):
    return ROOT / f"reconfig_{name}_library_20260702.npz"


def load_or_build(name):
    path = cache_path(name)
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {
            "V": z["V"].astype(np.float32),
            "lf": z["lf"].astype(np.float32),
            "base_p": z["base_p"].astype(np.float32),
            "failed": int(z["failed"]),
            "n_bus": int(z["n_bus"]),
            "n_topologies": int(z["n_topologies"]),
            "source": "cache",
        }
    spec = src.parse_system(src.ROOT / f"{name}.txt")
    topologies = src.enumerate_topologies(spec)
    lib = src.build_library(spec, topologies)
    np.savez_compressed(
        path,
        V=lib["V"].astype(np.float32),
        lf=lib["lf"].astype(np.float32),
        base_p=lib["base_p"].astype(np.float32),
        failed=np.array(lib["failed"], dtype=np.int64),
        n_bus=np.array(spec["n_bus"], dtype=np.int64),
        n_topologies=np.array(len(topologies), dtype=np.int64),
    )
    lib["n_bus"] = spec["n_bus"]
    lib["n_topologies"] = len(topologies)
    lib["source"] = "built"
    return lib


def deployment(n_bus, k):
    return src.deployment(n_bus, k)


def exact_posterior(lib, buses, obs, lf_idx):
    pred = lib["V"][:, lf_idx, :][:, buses]
    ll = -0.5 * np.sum(((pred - obs[None, :]) / SIGMA) ** 2, axis=1)
    q = np.exp(ll - np.max(ll))
    q /= np.sum(q)
    return q.astype(np.float32)


def make_dataset(lib, n, seed, k):
    rng = np.random.RandomState(seed)
    V = lib["V"]
    n_topos, n_lf, n_bus = V.shape
    buses = deployment(n_bus, k)
    xs = np.zeros((n, n_bus * 3), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    q = np.zeros((n, n_topos), dtype=np.float32)
    for i in range(n):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        lf = lib["lf"][lf_idx]
        obs = V[ti, lf_idx, buses] + rng.normal(0.0, SIGMA, size=len(buses))
        xs[i, buses] = (obs - 1.0) / SIGMA
        xs[i, n_bus + buses] = 1.0
        xs[i, 2 * n_bus + buses] = lib["base_p"][buses] * lf
        y[i] = ti
        q[i] = exact_posterior(lib, buses, obs, lf_idx)
    return xs, y, q


def predict_probs(model, xs):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(xs), BATCH):
            xb = torch.tensor(xs[start:start+BATCH], dtype=torch.float32, device=DEVICE)
            chunks.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.vstack(chunks)


def metrics_from_probs(p, y, q):
    exact_pred = np.argmax(q, axis=1)
    direct = np.argmax(p, axis=1)
    top = np.argsort(-p, axis=1)
    out = {
        "exact_top1": float(np.mean(exact_pred == y)),
        "nre_top1": float(np.mean(direct == y)),
        "exact_agree": float(np.mean(direct == exact_pred)),
        "kl_ref_nre": float(np.mean(np.sum(q * (np.log(np.clip(q, 1e-12, 1.0)) - np.log(np.clip(p, 1e-12, 1.0))), axis=1))),
    }
    for m in [5, 10, 20]:
        cand = top[:, :m]
        out[f"truth_top{m}"] = float(np.mean([y[i] in cand[i] for i in range(len(y))]))
        out[f"exact_top{m}"] = float(np.mean([exact_pred[i] in cand[i] for i in range(len(y))]))
        rerank = np.array([cand[i, int(np.argmax(q[i, cand[i]]))] for i in range(len(y))], dtype=np.int64)
        out[f"rerank{m}"] = float(np.mean(rerank == y))
    return out, direct, exact_pred, top


def train_one(lib, K, cfg, seed, xs_train, y_train, q_train, xs_val, y_val, q_val):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_topos, _, n_bus = lib["V"].shape
    model = src.StressNRE(n_topos, n_bus).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=src.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"] * int(np.ceil(len(xs_train) / BATCH)), eta_min=1e-5)
    kl_fn = nn.KLDivLoss(reduction="batchmean")
    ce_fn = nn.CrossEntropyLoss()
    xt = torch.tensor(xs_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    qt = torch.tensor(q_train, dtype=torch.float32)
    rng = np.random.RandomState(seed + 991)
    best_score = -1e9
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    best_val = None
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        order = rng.permutation(len(xs_train))
        model.train()
        for start in range(0, len(xs_train), BATCH):
            idx = order[start:start+BATCH]
            xb = xt[idx].to(DEVICE)
            yb = yt[idx].to(DEVICE)
            qb = qt[idx].to(DEVICE)
            logits = model(xb)
            loss = kl_fn(torch.log_softmax(logits, dim=1), qb) + 0.15 * ce_fn(logits, yb)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
        val_p = predict_probs(model, xs_val)
        val, _, _, _ = metrics_from_probs(val_p, y_val, q_val)
        score = val["nre_top1"] + 0.02 * val["exact_agree"] - 0.01 * val["kl_ref_nre"]
        print(f"{cfg['name']} K={K} seed={seed} epoch={epoch}/{cfg['epochs']} val_nre={val['nre_top1']:.4f} val_exact={val['exact_top1']:.4f} top20={val['truth_top20']:.4f}", flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    ckpt = ROOT / f"nre_reconfig_{cfg['name']}_K{K}_seed{seed}_20260702.pt"
    torch.save({"model_state": model.state_dict(), "system": cfg["name"], "K": K, "seed": seed, "best_epoch": best_epoch, "validation": best_val}, ckpt)
    return model, ckpt.name, best_epoch, best_val, time.time() - t0


def boot_ci(arr):
    arr = np.asarray(arr, dtype=np.float64)
    rng = np.random.RandomState(777)
    vals = np.empty(BOOT, dtype=np.float64)
    for b in range(BOOT):
        idx = rng.randint(0, len(arr), size=len(arr))
        vals[b] = np.mean(arr[idx])
    return float(np.mean(arr)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def run_system(cfg):
    lib = load_or_build(cfg["name"])
    curve = []
    for k in cfg["curve"]:
        _, y_c, q_c = make_dataset(lib, 1500, cfg["K"] * 1000 + k, k)
        curve.append((k, float(np.mean(np.argmax(q_c, axis=1) == y_c))))
    xs_train, y_train, q_train = make_dataset(lib, cfg["train_n"], cfg["K"] * 100 + 1, cfg["K"])
    xs_val, y_val, q_val = make_dataset(lib, cfg["val_n"], cfg["K"] * 100 + 2, cfg["K"])
    xs_test, y_test, q_test = make_dataset(lib, cfg["test_n"], cfg["K"] * 100 + 3, cfg["K"])
    probs = []
    rows = []
    for seed in cfg["seeds"]:
        model, ckpt, best_epoch, best_val, train_sec = train_one(lib, cfg["K"], cfg, seed, xs_train, y_train, q_train, xs_val, y_val, q_val)
        p = predict_probs(model, xs_test)
        met, _, _, _ = metrics_from_probs(p, y_test, q_test)
        probs.append(p)
        rows.append((seed, ckpt, best_epoch, best_val, met, train_sec))
    p_ens = np.mean(np.stack(probs, axis=0), axis=0)
    ens, direct, exact_pred, top = metrics_from_probs(p_ens, y_test, q_test)
    cand20 = top[:, :20]
    rerank20 = np.array([cand20[i, int(np.argmax(q_test[i, cand20[i]]))] for i in range(len(y_test))], dtype=np.int64)
    ci = {
        "direct": boot_ci(direct == y_test),
        "exact": boot_ci(exact_pred == y_test),
        "truth_top20": boot_ci(np.array([y_test[i] in cand20[i] for i in range(len(y_test))])),
        "rerank20": boot_ci(rerank20 == y_test),
    }
    return lib, curve, rows, ens, ci


def main():
    t0 = time.time()
    lines = []
    lines.append("Full-scale reconfiguration exact-comparable attack experiment")
    lines.append("date=2026-07-02")
    lines.append("role=balanced full-scale stress enhancement with higher sensor budgets, top-k rerank, and bootstrap CI")
    lines.append("not_claimed=unbalanced three-phase or utility field deployment")
    lines.append(f"device={DEVICE}")
    for cfg in SYSTEMS:
        lib, curve, rows, ens, ci = run_system(cfg)
        lines.append(f"system={cfg['name']}")
        lines.append(f"n_bus={lib['n_bus']}")
        lines.append(f"n_topologies={lib['n_topologies']}")
        lines.append(f"failed_pf={lib['failed']}")
        lines.append(f"library_source={lib['source']}")
        lines.append("exact_curve")
        lines.append("K,exact_top1")
        for k, v in curve:
            lines.append(f"{k},{v:.4f}")
        lines.append(f"attack_K={cfg['K']}")
        lines.append(f"train_samples={cfg['train_n']}")
        lines.append(f"validation_samples={cfg['val_n']}")
        lines.append(f"test_samples={cfg['test_n']}")
        lines.append("seed,best_epoch,exact_top1,nre_top1,gap,exact_agree,kl,truth_top5,truth_top10,truth_top20,rerank5,rerank10,rerank20,train_sec,checkpoint")
        for seed, ckpt, best_epoch, best_val, met, train_sec in rows:
            lines.append(f"{seed},{best_epoch},{met['exact_top1']:.4f},{met['nre_top1']:.4f},{met['exact_top1']-met['nre_top1']:.4f},{met['exact_agree']:.4f},{met['kl_ref_nre']:.4f},{met['truth_top5']:.4f},{met['truth_top10']:.4f},{met['truth_top20']:.4f},{met['rerank5']:.4f},{met['rerank10']:.4f},{met['rerank20']:.4f},{train_sec:.1f},{ckpt}")
        lines.append("ensemble")
        for key in ["exact_top1", "nre_top1", "exact_agree", "kl_ref_nre", "truth_top5", "truth_top10", "truth_top20", "rerank5", "rerank10", "rerank20"]:
            lines.append(f"ensemble_{key}={ens[key]:.4f}")
        lines.append("bootstrap_ci")
        lines.append("metric,mean,ci95_low,ci95_high")
        for key, (mean, lo, hi) in ci.items():
            lines.append(f"{key},{mean:.4f},{lo:.4f},{hi:.4f}")
    lines.append(f"elapsed_sec={time.time()-t0:.1f}")
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
