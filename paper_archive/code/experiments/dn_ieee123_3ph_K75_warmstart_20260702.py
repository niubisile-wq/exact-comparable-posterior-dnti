# -*- coding: utf-8 -*-
"""
IEEE123 controlled three-phase K=75 warm-start attack experiment.

This script avoids pandapower dependency by using the frozen
ieee123_3ph_library_20260702.npz produced earlier. It warm-starts from the
K=60 redundant NRE checkpoints and fine-tunes for the K=75 deployment.
No previous result or checkpoint is overwritten.
"""

from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
PKG_ROOT = Path.home() / "Desktop" / "配电网论文一区投稿成果包_20260702"
PKG_STATS = PKG_ROOT / "03_frozen_tables_stats"
PKG_CODE = PKG_ROOT / "02_code"
LIB_PATH = ROOT / "ieee123_3ph_library_20260702.npz"
OUT_NAME = "ieee123_3ph_K75_warmstart_attack_20260702.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LF_GRID = np.array([0.9, 1.0, 1.1], dtype=np.float32)
SIGMA = 0.0035
SEEDS = [42, 123, 456]
K_ATTACK = 75
BATCH = 256
TRAIN_STEPS = 5000
LR = 2.5e-4
N_CURVE = 2500
N_VAL = 900
N_FINAL = 1500


class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class Controlled3PhNRE(nn.Module):
    def __init__(self, n_topos, n_bus):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_bus * 5, 384), nn.LayerNorm(384), nn.GELU())
        self.res1 = ResBlock(384)
        self.res2 = ResBlock(384)
        self.head = nn.Sequential(nn.Linear(384, 192), nn.LayerNorm(192), nn.GELU(), nn.Linear(192, n_topos))

    def forward(self, x):
        h = self.embed(x)
        h = self.res1(h)
        h = self.res2(h)
        return self.head(h)


def deployment(n_bus, k):
    return np.unique(np.rint(np.linspace(1, n_bus - 1, k)).astype(int))


def build_features(obs_v3, buses, n_bus, base_p_norm, lf):
    x = np.zeros(n_bus * 5, dtype=np.float32)
    x[buses] = (obs_v3[:, 0] - 1.0) / SIGMA
    x[n_bus + buses] = (obs_v3[:, 1] - 1.0) / SIGMA
    x[2 * n_bus + buses] = (obs_v3[:, 2] - 1.0) / SIGMA
    x[3 * n_bus + buses] = 1.0
    x[4 * n_bus : 5 * n_bus] = base_p_norm * lf
    return x


def exact_posterior(v_library, buses, obs_v3, lf_idx):
    da = (v_library[:, lf_idx, :, 0][:, buses] - obs_v3[:, 0]) / SIGMA
    db = (v_library[:, lf_idx, :, 1][:, buses] - obs_v3[:, 1]) / SIGMA
    dc = (v_library[:, lf_idx, :, 2][:, buses] - obs_v3[:, 2]) / SIGMA
    ll = -0.5 * (np.sum(da * da, axis=1) + np.sum(db * db, axis=1) + np.sum(dc * dc, axis=1))
    q = np.exp(ll - np.max(ll))
    q /= np.sum(q)
    return q.astype(np.float32)


def load_library():
    z = np.load(LIB_PATH, allow_pickle=True)
    return {
        "v": z["v"].astype(np.float32),
        "base_p_norm": z["base_p_norm"].astype(np.float32),
        "n_bus": int(z["n_bus"]),
        "n_topos": int(z["n_topos"]),
        "failed": int(z["failed"]),
        "raw_topos": int(z["raw_topos"]),
    }


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
        lf = float(LF_GRID[lf_idx])
        obs = v[ti, lf_idx, buses, :] + rng.normal(0.0, SIGMA, size=(len(buses), 3))
        xs[i] = build_features(obs, buses, n_bus, lib["base_p_norm"], lf)
        y[i] = ti
        q[i] = exact_posterior(v, buses, obs, lf_idx)
    return xs, y, q


def eval_exact_curve(lib):
    rows = []
    for k in [30, 45, 60, 75]:
        _, y, q = make_dataset(lib, N_CURVE, 770000 + k, k)
        rows.append((k, float(np.mean(np.argmax(q, axis=1) == y))))
    return rows


def eval_model(model, xs, y, q):
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(xs), BATCH):
            xb = torch.tensor(xs[start:start + BATCH], dtype=torch.float32, device=DEVICE)
            probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    p = np.vstack(probs)
    exact_pred = np.argmax(q, axis=1)
    nre_pred = np.argmax(p, axis=1)
    top = np.argsort(-p, axis=1)
    out = {
        "exact_top1": float(np.mean(exact_pred == y)),
        "nre_top1": float(np.mean(nre_pred == y)),
        "exact_agree": float(np.mean(nre_pred == exact_pred)),
    }
    for m in [3, 5, 10, 20]:
        cand = top[:, :m]
        out[f"truth_in_top{m}"] = float(np.mean([y[i] in cand[i] for i in range(len(y))]))
        out[f"exact_in_top{m}"] = float(np.mean([exact_pred[i] in cand[i] for i in range(len(y))]))
        rerank = np.array([cand[i, int(np.argmax(q[i, cand[i]]))] for i in range(len(y))], dtype=np.int64)
        out[f"rerank{m}_top1"] = float(np.mean(rerank == y))
    return out


def load_warmstart(model, seed):
    ckpt = ROOT / f"nre_ieee123_3ph_K60_redundant_seed{seed}_20260702.pt"
    if not ckpt.exists():
        return "scratch"
    obj = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(obj["model_state"])
    return ckpt.name


def train_one(lib, seed, xs_val, y_val, q_val, xs_final, y_final, q_final):
    torch.manual_seed(seed + 75000)
    np.random.seed(seed + 75000)
    rng = np.random.RandomState(seed + 751000)
    n_bus = lib["n_bus"]
    n_topos = lib["n_topos"]
    buses = deployment(n_bus, K_ATTACK)
    model = Controlled3PhNRE(n_topos, n_bus).to(DEVICE)
    warmstart = load_warmstart(model, seed)
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
            ti = rng.randint(0, n_topos)
            lf_idx = rng.randint(0, len(LF_GRID))
            lf = float(LF_GRID[lf_idx])
            obs = lib["v"][ti, lf_idx, buses, :] + rng.normal(0.0, SIGMA, size=(len(buses), 3))
            xs[i] = build_features(obs, buses, n_bus, lib["base_p_norm"], lf)
            ys[i] = ti
            qs[i] = exact_posterior(lib["v"], buses, obs, lf_idx)
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
            print(f"seed={seed} step={step}/{TRAIN_STEPS} val_nre={val['nre_top1']:.4f} val_exact={val['exact_top1']:.4f} top3={val['truth_in_top3']:.4f}", flush=True)
            if score > best_score:
                best_score = score
                best_val = val
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    final = eval_model(model, xs_final, y_final, q_final)
    ckpt_name = f"nre_ieee123_3ph_K75_warmstart_seed{seed}_20260702.pt"
    torch.save({"model_state": model.state_dict(), "seed": seed, "K": K_ATTACK, "warmstart": warmstart, "best_step": best_step, "validation": best_val}, ROOT / ckpt_name)
    return {"seed": seed, "warmstart": warmstart, "best_step": best_step, "best_val": best_val, "final": final, "train_sec": time.time() - t0, "checkpoint": ckpt_name}


def main():
    t0 = time.time()
    lib = load_library()
    print(f"Device: {DEVICE}")
    print(f"IEEE123 K75 warm-start attack: n_bus={lib['n_bus']} n_topos={lib['n_topos']} K={K_ATTACK}")
    curve = eval_exact_curve(lib)
    xs_val, y_val, q_val = make_dataset(lib, N_VAL, 875123, K_ATTACK)
    xs_final, y_final, q_final = make_dataset(lib, N_FINAL, 975123, K_ATTACK)
    rows = [train_one(lib, seed, xs_val, y_val, q_val, xs_final, y_final, q_final) for seed in SEEDS]

    lines = []
    lines.append("IEEE123 controlled three-phase K75 warm-start redundant-measurement attack")
    lines.append("date=2026-07-02")
    lines.append(f"device={DEVICE}")
    lines.append(f"n_bus={lib['n_bus']}")
    lines.append(f"n_topologies={lib['n_topos']}")
    lines.append(f"raw_topologies={lib['raw_topos']}")
    lines.append(f"failed_power_flows={lib['failed']}")
    lines.append("role=raw-asset-derived controlled unbalanced exact-comparability benchmark")
    lines.append("method=K60_checkpoint_warmstart_then_K75_finetune")
    lines.append("not_claimed=utility field validation or NRE speedup")
    lines.append("exact_curve")
    lines.append("K,exact_top1")
    for k, acc in curve:
        lines.append(f"{k},{acc:.4f}")
    lines.append(f"attack_K={K_ATTACK}")
    lines.append(f"validation_samples={N_VAL}")
    lines.append(f"final_samples={N_FINAL}")
    lines.append("seed,warmstart,best_step,exact_top1,nre_top1,gap,exact_agree,truth_top3,truth_top5,truth_top10,truth_top20,rerank3,rerank5,rerank10,rerank20,train_sec,checkpoint")
    for r in rows:
        f = r["final"]
        lines.append(
            f"{r['seed']},{r['warmstart']},{r['best_step']},{f['exact_top1']:.4f},{f['nre_top1']:.4f},"
            f"{f['exact_top1'] - f['nre_top1']:.4f},{f['exact_agree']:.4f},"
            f"{f['truth_in_top3']:.4f},{f['truth_in_top5']:.4f},{f['truth_in_top10']:.4f},{f['truth_in_top20']:.4f},"
            f"{f['rerank3_top1']:.4f},{f['rerank5_top1']:.4f},{f['rerank10_top1']:.4f},{f['rerank20_top1']:.4f},"
            f"{r['train_sec']:.1f},{r['checkpoint']}"
        )
    for key in ["exact_top1", "nre_top1", "exact_agree", "truth_in_top3", "truth_in_top5", "truth_in_top10", "truth_in_top20", "rerank3_top1", "rerank5_top1", "rerank10_top1", "rerank20_top1"]:
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
