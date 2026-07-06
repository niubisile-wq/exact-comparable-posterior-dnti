# -*- coding: utf-8 -*-
"""
IEEE123 controlled three-phase attack experiment.

Purpose:
  Turn the three-phase evidence from a small extension chain into a stronger
  raw-asset-derived controlled benchmark by adding a redundant-measurement
  recovery curve and a K=60 sensor-specific NRE run.

Controls:
  - Uses the same raw feeder assets, natural loop-forming ties, LF grid, and
    three-phase exact-reference library as dn_ieee123_3ph_controlled.py.
  - Does not claim utility field validation or speedup.
  - Does not overwrite the original result file.
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import dn_ieee123_3ph_controlled as src


ROOT = Path(__file__).resolve().parent
PKG_STATS = Path.home() / "Desktop" / "配电网论文一区投稿成果包_20260702" / "03_frozen_tables_stats"
PKG_CODE = Path.home() / "Desktop" / "配电网论文一区投稿成果包_20260702" / "02_code"
OUT_NAME = "ieee123_3ph_redundant_attack_20260702.txt"
LIB_NAME = "ieee123_3ph_library_20260702.npz"
DEVICE = src.DEVICE

SIGMA = src.SIGMA
SEEDS = [42, 123, 456]
K_ATTACK = 60
BATCH = 256
TRAIN_STEPS = 8500
LR = 3e-4
N_CURVE = 2500
N_FINAL = 1200


def load_or_build_library():
    path = ROOT / LIB_NAME
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {
            "v": z["v"].astype(np.float32),
            "base_p_norm": z["base_p_norm"].astype(np.float32),
            "n_bus": int(z["n_bus"]),
            "n_topos": int(z["n_topos"]),
            "failed": int(z["failed"]),
            "raw_topos": int(z["raw_topos"]),
        }

    spec = src.build_spec()
    topologies, n_tree, n_tie = src.enumerate_topologies(spec)
    kept_topos, v_library, failed = src.build_library(spec, topologies, n_tree, n_tie)
    n_bus = len(spec["bus_ids"])
    base_p = np.zeros(n_bus, dtype=np.float32)
    for bus_raw, (p_kw, _) in spec["load_map"].items():
        base_p[spec["node2idx"][bus_raw]] += p_kw / 1000.0
    base_p_norm = base_p / max(float(base_p.max()), 1e-8)
    np.savez_compressed(
        path,
        v=v_library.astype(np.float32),
        base_p_norm=base_p_norm.astype(np.float32),
        n_bus=n_bus,
        n_topos=v_library.shape[0],
        failed=failed,
        raw_topos=len(topologies),
    )
    return {
        "v": v_library.astype(np.float32),
        "base_p_norm": base_p_norm.astype(np.float32),
        "n_bus": n_bus,
        "n_topos": int(v_library.shape[0]),
        "failed": int(failed),
        "raw_topos": int(len(topologies)),
    }


def deployment(n_bus, k):
    return src.deployment(n_bus, k)


def make_dataset(lib, n, seed, k):
    rng = np.random.RandomState(seed)
    v = lib["v"]
    n_topos, n_lf, n_bus, _ = v.shape
    buses = deployment(n_bus, k)
    xs = np.zeros((n, n_bus * 5), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    q = np.zeros((n, n_topos), dtype=np.float32)
    for i in range(n):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        lf = float(src.LF_GRID[lf_idx])
        obs = v[ti, lf_idx, buses, :] + rng.normal(0.0, SIGMA, size=(len(buses), 3))
        xs[i] = src.build_features(obs, buses, n_bus, lib["base_p_norm"], lf)
        y[i] = ti
        q[i] = src.exact_posterior(v, buses, obs, lf_idx)
    return xs, y, q


def eval_exact_curve(lib):
    rows = []
    for k in [30, 45, 60, 75]:
        _, y, q = make_dataset(lib, N_CURVE, 710000 + k, k)
        exact_pred = np.argmax(q, axis=1)
        rows.append((k, float(np.mean(exact_pred == y))))
    return rows


def train_one(lib, seed, xs_val, y_val, q_val, xs_final, y_final, q_final):
    torch.manual_seed(seed + 123000)
    np.random.seed(seed + 123000)
    rng = np.random.RandomState(seed + 456000)
    n_bus = lib["n_bus"]
    n_topos = lib["n_topos"]
    buses = deployment(n_bus, K_ATTACK)
    model = src.Controlled3PhNRE(n_topos, n_bus).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_STEPS, eta_min=1e-5)
    kl_fn = nn.KLDivLoss(reduction="batchmean")
    ce_fn = nn.CrossEntropyLoss()
    best_score = -1e9
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_step = 0
    best_val = None
    t0 = time.time()
    model.train()
    for step in range(1, TRAIN_STEPS + 1):
        xs = np.zeros((BATCH, n_bus * 5), dtype=np.float32)
        ys = np.zeros(BATCH, dtype=np.int64)
        qs = np.zeros((BATCH, n_topos), dtype=np.float32)
        for i in range(BATCH):
            ti = rng.randint(0, n_topos)
            lf_idx = rng.randint(0, len(src.LF_GRID))
            lf = float(src.LF_GRID[lf_idx])
            obs = lib["v"][ti, lf_idx, buses, :] + rng.normal(0.0, SIGMA, size=(len(buses), 3))
            xs[i] = src.build_features(obs, buses, n_bus, lib["base_p_norm"], lf)
            ys[i] = ti
            qs[i] = src.exact_posterior(lib["v"], buses, obs, lf_idx)
        xb = torch.tensor(xs, dtype=torch.float32, device=DEVICE)
        yb = torch.tensor(ys, dtype=torch.long, device=DEVICE)
        qb = torch.tensor(qs, dtype=torch.float32, device=DEVICE)
        logits = model(xb)
        loss = kl_fn(torch.log_softmax(logits, dim=1), qb) + 0.20 * ce_fn(logits, yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == TRAIN_STEPS:
            val = eval_model(model, xs_val, y_val, q_val)
            score = val["nre_top1"] + 0.05 * val["exact_agree"]
            print(
                f"seed={seed} step={step}/{TRAIN_STEPS} val_nre={val['nre_top1']:.4f} "
                f"val_exact={val['exact_top1']:.4f} top5={val['truth_in_top5']:.4f}",
                flush=True,
            )
            if score > best_score:
                best_score = score
                best_val = val
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    final = eval_model(model, xs_final, y_final, q_final)
    ckpt_name = f"nre_ieee123_3ph_K{K_ATTACK}_redundant_seed{seed}_20260702.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "seed": seed,
            "K": K_ATTACK,
            "n_topos": n_topos,
            "n_bus": n_bus,
            "best_step": best_step,
            "validation": best_val,
        },
        ROOT / ckpt_name,
    )
    return {
        "seed": seed,
        "best_step": best_step,
        "best_val": best_val,
        "final": final,
        "train_sec": time.time() - t0,
        "checkpoint": ckpt_name,
    }


def eval_model(model, xs, y, q):
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(xs), BATCH):
            xb = torch.tensor(xs[start:start + BATCH], dtype=torch.float32, device=DEVICE)
            probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    p = np.vstack(probs)
    return metrics_from_probs(p, y, q)


def metrics_from_probs(p, y, q):
    exact_pred = np.argmax(q, axis=1)
    nre_pred = np.argmax(p, axis=1)
    out = {
        "exact_top1": float(np.mean(exact_pred == y)),
        "nre_top1": float(np.mean(nre_pred == y)),
        "exact_agree": float(np.mean(nre_pred == exact_pred)),
    }
    top = np.argsort(-p, axis=1)
    for m in [3, 5, 10, 20]:
        cand = top[:, :m]
        out[f"truth_in_top{m}"] = float(np.mean([y[i] in cand[i] for i in range(len(y))]))
        out[f"exact_in_top{m}"] = float(np.mean([exact_pred[i] in cand[i] for i in range(len(y))]))
        rerank = np.array([cand[i, int(np.argmax(q[i, cand[i]]))] for i in range(len(y))], dtype=np.int64)
        out[f"rerank{m}_top1"] = float(np.mean(rerank == y))
    return out


def main():
    t0 = time.time()
    lib = load_or_build_library()
    print(f"Device: {DEVICE}")
    print(f"IEEE123 3ph redundant attack: n_bus={lib['n_bus']} n_topos={lib['n_topos']} K={K_ATTACK}")
    curve = eval_exact_curve(lib)
    xs_val, y_val, q_val = make_dataset(lib, 900, 880123, K_ATTACK)
    xs_final, y_final, q_final = make_dataset(lib, N_FINAL, 990123, K_ATTACK)
    rows = [train_one(lib, seed, xs_val, y_val, q_val, xs_final, y_final, q_final) for seed in SEEDS]

    lines = []
    lines.append("IEEE123 controlled three-phase redundant-measurement attack")
    lines.append("date=2026-07-02")
    lines.append(f"device={DEVICE}")
    lines.append(f"n_bus={lib['n_bus']}")
    lines.append(f"n_topologies={lib['n_topos']}")
    lines.append(f"raw_topologies={lib['raw_topos']}")
    lines.append(f"failed_power_flows={lib['failed']}")
    lines.append("role=raw-asset-derived controlled unbalanced exact-comparability benchmark")
    lines.append("not_claimed=utility field validation or NRE speedup")
    lines.append("exact_curve")
    lines.append("K,exact_top1")
    for k, acc in curve:
        lines.append(f"{k},{acc:.4f}")
    lines.append(f"attack_K={K_ATTACK}")
    lines.append(f"final_samples={N_FINAL}")
    lines.append("seed,best_step,exact_top1,nre_top1,gap,exact_agree,truth_top5,truth_top10,truth_top20,rerank5,rerank10,rerank20,train_sec,checkpoint")
    for r in rows:
        f = r["final"]
        lines.append(
            f"{r['seed']},{r['best_step']},{f['exact_top1']:.4f},{f['nre_top1']:.4f},"
            f"{f['exact_top1'] - f['nre_top1']:.4f},{f['exact_agree']:.4f},"
            f"{f['truth_in_top5']:.4f},{f['truth_in_top10']:.4f},{f['truth_in_top20']:.4f},"
            f"{f['rerank5_top1']:.4f},{f['rerank10_top1']:.4f},{f['rerank20_top1']:.4f},"
            f"{r['train_sec']:.1f},{r['checkpoint']}"
        )
    for key in ["exact_top1", "nre_top1", "exact_agree", "truth_in_top5", "truth_in_top10", "truth_in_top20", "rerank5_top1", "rerank10_top1", "rerank20_top1"]:
        lines.append(f"mean_{key}={np.mean([r['final'][key] for r in rows]):.4f}")
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
