# -*- coding: utf-8 -*-
"""
Sensor-dropout robustness audit for counterattack models.

Evaluates already-trained high-sensor-budget models under random sensor dropout.
No retraining and no checkpoint writes are performed. Exact references are
computed on the same retained measurement subset.
"""

from pathlib import Path
import numpy as np
import torch

import dn_119bus_sensor_policy_v2_train_20260702 as m119
import dn_300bus_redundant_sensing_attack_20260702 as m300
import dn_ieee123_3ph_K75_warmstart_20260702 as m123
import dn_reconfig_fullscale_attack_20260702 as mfs

ROOT = Path.home() / "Desktop" / "配电网实验_临时"
PKG_ROOT = Path.home() / "Desktop" / "配电网论文一区投稿成果包_20260702"
PKG_STATS = PKG_ROOT / "03_frozen_tables_stats"
PKG_CODE = PKG_ROOT / "02_code"
OUT_NAME = "counterattack_sensor_dropout_20260702.txt"
DROP_RATES = [0.0, 0.10, 0.20, 0.40]

SENSORS_119_K60 = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118], dtype=np.int64)


def softmax_np(z):
    z = z - np.max(z, axis=1, keepdims=True)
    p = np.exp(z)
    return p / np.sum(p, axis=1, keepdims=True)


def summarize(p, y, q):
    exact_pred = np.argmax(q, axis=1)
    direct = np.argmax(p, axis=1)
    top = np.argsort(-p, axis=1)
    row = {
        "exact_top1": float(np.mean(exact_pred == y)),
        "direct_top1": float(np.mean(direct == y)),
        "exact_agree": float(np.mean(direct == exact_pred)),
        "kl_ref_nre": float(np.mean(np.sum(q * (np.log(np.clip(q, 1e-12, 1.0)) - np.log(np.clip(p, 1e-12, 1.0))), axis=1))),
    }
    for k in [5, 10, 20]:
        cand = top[:, :k]
        row[f"truth_top{k}"] = float(np.mean([y[i] in cand[i] for i in range(len(y))]))
        rerank = np.array([cand[i, int(np.argmax(q[i, cand[i]]))] for i in range(len(y))], dtype=np.int64)
        row[f"rerank{k}"] = float(np.mean(rerank == y))
    return row


def predict_119(xs):
    lib = m119.base.load_119_lib()
    probs = []
    for seed in [42, 123, 456, 789, 2024]:
        model = m119.load_old_model(seed, lib)
        ckpt = ROOT / f"nre_119bus_ip1_K60_sensor_v2_seed{seed}_20260702.pt"
        obj = torch.load(ckpt, map_location=m119.DEVICE, weights_only=False)
        model.load_state_dict(obj["model_state"])
        probs.append(softmax_np(m119.predict_logits(model, xs)))
    return np.mean(np.stack(probs, axis=0), axis=0)


def make_119(drop_rate, n=3500):
    lib = m119.base.load_119_lib()
    rng = np.random.RandomState(119880 + int(drop_rate * 1000))
    V = lib["V"]
    n_topos, n_lf, n_bus = V.shape
    k0 = len(SENSORS_119_K60)
    keep_n = max(1, int(round(k0 * (1.0 - drop_rate))))
    xs = np.zeros((n, n_bus * 3), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    q = np.zeros((n, n_topos), dtype=np.float32)
    for i in range(n):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        lf = float(lib["lf"][lf_idx])
        buses = np.sort(rng.choice(SENSORS_119_K60, keep_n, replace=False))
        obs = V[ti, lf_idx, buses] + rng.normal(0.0, m119.SIGMA, size=len(buses))
        pred = V[:, lf_idx, :][:, buses]
        ll = -0.5 * np.sum(((pred - obs[None, :]) / m119.SIGMA) ** 2, axis=1)
        qq = np.exp(ll - np.max(ll)); qq /= np.sum(qq)
        xs[i, buses] = obs
        xs[i, n_bus + buses] = 1.0
        xs[i, 2 * n_bus + buses] = lib["base_p"][buses] * lf
        y[i] = ti
        q[i] = qq.astype(np.float32)
    return keep_n, xs, y, q


def predict_123(xs, lib):
    probs = []
    for seed in [42, 123, 456]:
        model = m123.Controlled3PhNRE(lib["n_topos"], lib["n_bus"]).to(m123.DEVICE)
        ckpt = ROOT / f"nre_ieee123_3ph_K75_warmstart_seed{seed}_20260702.pt"
        obj = torch.load(ckpt, map_location=m123.DEVICE, weights_only=False)
        model.load_state_dict(obj["model_state"])
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(xs), m123.BATCH):
                xb = torch.tensor(xs[start:start+m123.BATCH], dtype=torch.float32, device=m123.DEVICE)
                chunks.append(torch.softmax(model(xb), dim=1).cpu().numpy())
        probs.append(np.vstack(chunks))
    return np.mean(np.stack(probs, axis=0), axis=0)


def make_123(drop_rate, n=1000):
    lib = m123.load_library()
    rng = np.random.RandomState(123880 + int(drop_rate * 1000))
    V = lib["v"]
    n_topos, n_lf, n_bus, _ = V.shape
    deploy = m123.deployment(n_bus, 75)
    keep_n = max(1, int(round(len(deploy) * (1.0 - drop_rate))))
    xs = np.zeros((n, n_bus * 5), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    q = np.zeros((n, n_topos), dtype=np.float32)
    for i in range(n):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        lf = float(m123.LF_GRID[lf_idx])
        buses = np.sort(rng.choice(deploy, keep_n, replace=False))
        obs = V[ti, lf_idx, buses, :] + rng.normal(0.0, m123.SIGMA, size=(len(buses), 3))
        xs[i] = m123.build_features(obs, buses, n_bus, lib["base_p_norm"], lf)
        y[i] = ti
        q[i] = m123.exact_posterior(V, buses, obs, lf_idx)
    return lib, keep_n, xs, y, q


def predict_fs(xs, lib, system_name, K):
    probs = []
    seeds = [202, 1202] if system_name == "SystemData_202" else [417, 1417]
    for seed in seeds:
        model = mfs.src.StressNRE(lib["n_topologies"], lib["n_bus"]).to(mfs.DEVICE)
        ckpt = ROOT / f"nre_reconfig_{system_name}_K{K}_seed{seed}_20260702.pt"
        obj = torch.load(ckpt, map_location=mfs.DEVICE, weights_only=False)
        model.load_state_dict(obj["model_state"])
        probs.append(mfs.predict_probs(model, xs))
    return np.mean(np.stack(probs, axis=0), axis=0)


def make_fs(system_name, K, drop_rate, n=1300):
    lib = mfs.load_or_build(system_name)
    rng = np.random.RandomState(K * 100 + int(drop_rate * 1000))
    V = lib["V"]
    n_topos, n_lf, n_bus = V.shape
    deploy = mfs.deployment(n_bus, K)
    keep_n = max(1, int(round(len(deploy) * (1.0 - drop_rate))))
    xs = np.zeros((n, n_bus * 3), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    q = np.zeros((n, n_topos), dtype=np.float32)
    for i in range(n):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        lf = float(lib["lf"][lf_idx])
        buses = np.sort(rng.choice(deploy, keep_n, replace=False))
        obs = V[ti, lf_idx, buses] + rng.normal(0.0, mfs.SIGMA, size=len(buses))
        pred = V[:, lf_idx, :][:, buses]
        ll = -0.5 * np.sum(((pred - obs[None, :]) / mfs.SIGMA) ** 2, axis=1)
        qq = np.exp(ll - np.max(ll)); qq /= np.sum(qq)
        xs[i, buses] = (obs - 1.0) / mfs.SIGMA
        xs[i, n_bus + buses] = 1.0
        xs[i, 2 * n_bus + buses] = lib["base_p"][buses] * lf
        y[i] = ti
        q[i] = qq.astype(np.float32)
    return lib, keep_n, xs, y, q


def main():
    lines = []
    lines.append("Counterattack sensor-dropout robustness audit")
    lines.append("date=2026-07-02")
    lines.append("role=no-retraining random sensor dropout stress; exact reference uses retained measurements")
    lines.append("case,drop_rate,retained_K,exact_top1,direct_top1,exact_agree,kl_ref_nre,truth_top5,truth_top10,truth_top20,rerank5,rerank10,rerank20")
    for dr in DROP_RATES:
        keep_n, xs, y, q = make_119(dr)
        p = predict_119(xs)
        r = summarize(p, y, q)
        lines.append(f"119bus_K60,{dr:.2f},{keep_n},{r['exact_top1']:.4f},{r['direct_top1']:.4f},{r['exact_agree']:.4f},{r['kl_ref_nre']:.4f},{r['truth_top5']:.4f},{r['truth_top10']:.4f},{r['truth_top20']:.4f},{r['rerank5']:.4f},{r['rerank10']:.4f},{r['rerank20']:.4f}")
    for dr in DROP_RATES:
        lib, keep_n, xs, y, q = make_123(dr)
        p = predict_123(xs, lib)
        r = summarize(p, y, q)
        lines.append(f"ieee123_K75,{dr:.2f},{keep_n},{r['exact_top1']:.4f},{r['direct_top1']:.4f},{r['exact_agree']:.4f},{r['kl_ref_nre']:.4f},{r['truth_top5']:.4f},{r['truth_top10']:.4f},{r['truth_top20']:.4f},{r['rerank5']:.4f},{r['rerank10']:.4f},{r['rerank20']:.4f}")
    for system_name, K in [("SystemData_202", 160), ("SystemData_417", 220)]:
        for dr in DROP_RATES:
            lib, keep_n, xs, y, q = make_fs(system_name, K, dr)
            p = predict_fs(xs, lib, system_name, K)
            r = summarize(p, y, q)
            lines.append(f"{system_name}_K{K},{dr:.2f},{keep_n},{r['exact_top1']:.4f},{r['direct_top1']:.4f},{r['exact_agree']:.4f},{r['kl_ref_nre']:.4f},{r['truth_top5']:.4f},{r['truth_top10']:.4f},{r['truth_top20']:.4f},{r['rerank5']:.4f},{r['rerank10']:.4f},{r['rerank20']:.4f}")
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
