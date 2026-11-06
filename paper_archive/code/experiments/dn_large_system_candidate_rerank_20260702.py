# -*- coding: utf-8 -*-
"""
Focused large-system fallback experiment for the Q1 manuscript.

Purpose:
  Quantify whether NRE remains useful under 119/300-bus stress cases by acting
  as a candidate-screening stage before exact/high-fidelity reranking.

This script does not retrain models. It loads the frozen libraries and NRE
checkpoints, then reports:
  - NRE top-1 accuracy
  - full exact-reference top-1 accuracy
  - truth captured by NRE top-M candidate sets
  - exact-reference winner captured by NRE top-M candidate sets
  - exact rerank@M accuracy restricted to the NRE top-M candidates
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn


ROOT = os.path.dirname(os.path.abspath(__file__))
PKG_STATS = r"<REPOSITORY_ROOT>\03_frozen_tables_stats"
OUT_NAME = "large_system_candidate_rerank_20260702.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class LoadAwareNRE119(nn.Module):
    def __init__(self, n_topo, n_bus):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_bus * 3, 512), nn.LayerNorm(512), nn.GELU())
        self.res1 = ResBlock(512)
        self.res2 = ResBlock(512)
        self.res3 = ResBlock(512)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, n_topo),
        )

    def forward(self, x):
        h = self.embed(x)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        return self.head(h)


class NRE300(nn.Module):
    def __init__(self, n_topos, n_bus):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_bus * 3, 768), nn.LayerNorm(768), nn.GELU())
        self.res1 = ResBlock(768)
        self.res2 = ResBlock(768)
        self.res3 = ResBlock(768)
        self.head = nn.Sequential(
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Linear(384, n_topos),
        )

    def forward(self, x):
        h = self.embed(x)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        return self.head(h)


def softmax_np(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    p = np.exp(z)
    return p / np.sum(p, axis=1, keepdims=True)


def top_indices_desc(p, max_m):
    return np.argsort(-p, axis=1)[:, :max_m]


def summarize_candidate_metrics(system, seed, y_true, p_nre, ll_exact, m_list, n_candidates):
    n = len(y_true)
    exact_pred = np.argmax(ll_exact, axis=1)
    nre_pred = np.argmax(p_nre, axis=1)
    max_m = max(m_list)
    top = top_indices_desc(p_nre, max_m)

    rows = []
    base = {
        "system": system,
        "seed": seed,
        "n": n,
        "n_candidates": n_candidates,
        "nre_top1": float(np.mean(nre_pred == y_true)),
        "exact_full_top1": float(np.mean(exact_pred == y_true)),
        "nre_exact_agree": float(np.mean(nre_pred == exact_pred)),
    }

    for m in m_list:
        cand = top[:, :m]
        truth_in = np.array([y_true[i] in cand[i] for i in range(n)], dtype=bool)
        exact_winner_in = np.array([exact_pred[i] in cand[i] for i in range(n)], dtype=bool)
        rerank_pred = np.empty(n, dtype=np.int64)
        for i in range(n):
            ci = cand[i]
            rerank_pred[i] = ci[int(np.argmax(ll_exact[i, ci]))]
        rows.append(
            {
                **base,
                "M": m,
                "candidate_fraction": float(m / n_candidates),
                "truth_in_nre_topM": float(np.mean(truth_in)),
                "exact_winner_in_nre_topM": float(np.mean(exact_winner_in)),
                "rerank_top1": float(np.mean(rerank_pred == y_true)),
                "rerank_vs_exact_full": float(np.mean(rerank_pred == exact_pred)),
                "nre_mass_topM": float(np.mean(np.sum(np.take_along_axis(p_nre, cand, axis=1), axis=1))),
            }
        )
    return rows


def exact_ll_119(V, sigma, buses_list, obs_list, lf_idx_list):
    n = len(obs_list)
    n_topos = V.shape[0]
    ll = np.empty((n, n_topos), dtype=np.float32)
    t0 = time.perf_counter()
    for i in range(n):
        buses = buses_list[i]
        obs = obs_list[i]
        pred = V[:, lf_idx_list[i], :][:, buses]
        vals = -0.5 * np.sum(((pred - obs[None, :]) / sigma) ** 2, axis=1)
        ll[i] = vals.astype(np.float32)
    return ll, time.perf_counter() - t0


def build_119_samples(lib, n_eval=1000, seed=770119):
    V = lib["V"]
    lf_grid = lib["lf"]
    base_p = lib["base_p"]
    n_topos, n_lf, n_bus = V.shape
    sigma = 0.009
    k_fixed = 25
    rng = np.random.RandomState(seed)
    xs = np.zeros((n_eval, n_bus * 3), dtype=np.float32)
    ys = np.zeros(n_eval, dtype=np.int64)
    buses_list = []
    obs_list = []
    lf_idx_list = np.zeros(n_eval, dtype=np.int64)
    for i in range(n_eval):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        lf = float(lf_grid[lf_idx])
        installed = np.sort(rng.choice(range(1, n_bus), k_fixed, replace=False))
        obs = V[ti, lf_idx, installed] + rng.normal(0.0, sigma, size=k_fixed)
        xs[i, installed] = obs
        xs[i, n_bus + installed] = 1.0
        xs[i, 2 * n_bus + installed] = base_p[installed] * lf
        ys[i] = ti
        buses_list.append(installed)
        obs_list.append(obs.astype(np.float32))
        lf_idx_list[i] = lf_idx
    return xs, ys, buses_list, obs_list, lf_idx_list


def load_119_lib():
    z = np.load(os.path.join(ROOT, "v_library_119bus.npz"))
    return {
        "V": z["V_library"].astype(np.float32),
        "base_p": z["base_P_norm"].astype(np.float32),
        "lf": z["lf_grid"].astype(np.float32),
    }


def evaluate_119():
    lib = load_119_lib()
    V = lib["V"]
    n_topos, _, n_bus = V.shape
    xs, ys, buses_list, obs_list, lf_idx_list = build_119_samples(lib)
    ll, exact_sec = exact_ll_119(V, 0.009, buses_list, obs_list, lf_idx_list)
    rows = []
    seeds = [42, 123, 456, 789, 2024]
    x_t = torch.tensor(xs, dtype=torch.float32).to(DEVICE)
    for seed in seeds:
        ckpt_path = os.path.join(ROOT, f"nre_119bus_ip1_seed{seed}.pt")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = LoadAwareNRE119(n_topos, n_bus).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(x_t).detach().cpu().numpy()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            nre_sec = time.perf_counter() - t0
        p = softmax_np(logits)
        seed_rows = summarize_candidate_metrics("119bus_IP1", seed, ys, p, ll, [5, 10, 20, 50], n_topos)
        for r in seed_rows:
            r["exact_full_ms"] = exact_sec / len(ys) * 1000.0
            r["nre_ms"] = nre_sec / len(ys) * 1000.0
        rows.extend(seed_rows)
    return rows


def load_300_lib():
    z = np.load(os.path.join(ROOT, "v_library_300bus.npz"), allow_pickle=True)
    return {
        "V": z["V_library"].astype(np.float32),
        "lf": z["lf_grid"].astype(np.float32),
        "base_p": z["base_P_norm"].astype(np.float32),
        "n_bus": int(z["n_bus"]),
        "n_topos": int(z["n_topologies"]),
    }


def deployment_sensors(n_bus, k):
    return np.unique(np.rint(np.linspace(1, n_bus - 1, k)).astype(int))


def build_300_samples(lib, n_eval=1500, seed=1177):
    V = lib["V"]
    lf_grid = lib["lf"]
    base_p = lib["base_p"]
    n_topos, n_lf, n_bus = V.shape
    k_fixed = 150
    miss_rate = 0.30
    sigma = 0.0015
    rng = np.random.RandomState(seed)
    xs = np.zeros((n_eval, n_bus * 3), dtype=np.float32)
    ys = np.zeros(n_eval, dtype=np.int64)
    buses_list = []
    obs_list = []
    lf_idx_list = np.zeros(n_eval, dtype=np.int64)
    installed_full = deployment_sensors(n_bus, k_fixed)
    for i in range(n_eval):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        lf = float(lf_grid[lf_idx])
        installed = installed_full.copy()
        n_miss = int(k_fixed * miss_rate)
        installed = np.delete(installed, rng.choice(len(installed), n_miss, replace=False))
        obs = V[ti, lf_idx, installed] + rng.normal(0.0, sigma, size=len(installed))
        xs[i, installed] = (obs - 1.0) / sigma
        xs[i, n_bus + installed] = 1.0
        xs[i, 2 * n_bus + installed] = base_p[installed] * lf
        ys[i] = ti
        buses_list.append(installed)
        obs_list.append(obs.astype(np.float32))
        lf_idx_list[i] = lf_idx
    return xs, ys, buses_list, obs_list, lf_idx_list


def exact_ll_300(V, sigma, buses_list, obs_list, lf_idx_list):
    n = len(obs_list)
    n_topos = V.shape[0]
    ll = np.empty((n, n_topos), dtype=np.float32)
    t0 = time.perf_counter()
    for i in range(n):
        buses = buses_list[i]
        obs = obs_list[i]
        pred = V[:, lf_idx_list[i], :][:, buses]
        vals = -0.5 * np.sum(((pred - obs[None, :]) / sigma) ** 2, axis=1)
        ll[i] = vals.astype(np.float32)
    return ll, time.perf_counter() - t0


def evaluate_300():
    lib = load_300_lib()
    V = lib["V"]
    n_topos, _, n_bus = V.shape
    xs, ys, buses_list, obs_list, lf_idx_list = build_300_samples(lib)
    ll, exact_sec = exact_ll_300(V, 0.0015, buses_list, obs_list, lf_idx_list)
    rows = []
    seeds = [42, 123, 456]
    x_t = torch.tensor(xs, dtype=torch.float32).to(DEVICE)
    for seed in seeds:
        ckpt_path = os.path.join(ROOT, f"nre_300bus_ipc_seed{seed}.pt")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = NRE300(n_topos, n_bus).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(x_t).detach().cpu().numpy()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            nre_sec = time.perf_counter() - t0
        p = softmax_np(logits)
        seed_rows = summarize_candidate_metrics("300bus_IPC_miss30", seed, ys, p, ll, [10, 20, 50, 100], n_topos)
        for r in seed_rows:
            r["exact_full_ms"] = exact_sec / len(ys) * 1000.0
            r["nre_ms"] = nre_sec / len(ys) * 1000.0
        rows.extend(seed_rows)
    return rows


def mean_std(vals):
    arr = np.array(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def format_summary(rows):
    out = []
    out.append("Large-system NRE candidate-screening + exact-rerank experiment")
    out.append("date=2026-07-02")
    out.append(f"device={DEVICE}")
    out.append("No retraining; frozen checkpoints and frozen voltage libraries are used.")
    out.append("Exact rerank@M means: rank all candidates by NRE, keep top M, then apply the same exact-reference voltage likelihood only inside that candidate set.")
    out.append("")

    header = (
        "system,seed,M,n,n_candidates,candidate_fraction,nre_top1,exact_full_top1,"
        "truth_in_nre_topM,exact_winner_in_nre_topM,rerank_top1,rerank_vs_exact_full,"
        "nre_exact_agree,nre_mass_topM,nre_ms,exact_full_ms"
    )
    out.append(header)
    for r in rows:
        out.append(
            f"{r['system']},{r['seed']},{r['M']},{r['n']},{r['n_candidates']},"
            f"{r['candidate_fraction']:.4f},{r['nre_top1']:.4f},{r['exact_full_top1']:.4f},"
            f"{r['truth_in_nre_topM']:.4f},{r['exact_winner_in_nre_topM']:.4f},"
            f"{r['rerank_top1']:.4f},{r['rerank_vs_exact_full']:.4f},"
            f"{r['nre_exact_agree']:.4f},{r['nre_mass_topM']:.4f},"
            f"{r['nre_ms']:.6f},{r['exact_full_ms']:.6f}"
        )

    out.append("")
    out.append("Aggregated by system and M")
    out.append(
        "system,M,n_seeds,nre_top1_mean,nre_top1_std,exact_full_top1_mean,"
        "truth_in_topM_mean,exact_winner_in_topM_mean,rerank_top1_mean,rerank_top1_std,"
        "rerank_vs_exact_full_mean,candidate_fraction"
    )
    for system in sorted(set(r["system"] for r in rows)):
        ms = sorted(set(r["M"] for r in rows if r["system"] == system))
        for m in ms:
            sub = [r for r in rows if r["system"] == system and r["M"] == m]
            nre_m, nre_s = mean_std([r["nre_top1"] for r in sub])
            rer_m, rer_s = mean_std([r["rerank_top1"] for r in sub])
            exact_m, _ = mean_std([r["exact_full_top1"] for r in sub])
            truth_m, _ = mean_std([r["truth_in_nre_topM"] for r in sub])
            exactwin_m, _ = mean_std([r["exact_winner_in_nre_topM"] for r in sub])
            rer_vs_exact_m, _ = mean_std([r["rerank_vs_exact_full"] for r in sub])
            frac = sub[0]["candidate_fraction"]
            out.append(
                f"{system},{m},{len(sub)},{nre_m:.4f},{nre_s:.4f},{exact_m:.4f},"
                f"{truth_m:.4f},{exactwin_m:.4f},{rer_m:.4f},{rer_s:.4f},"
                f"{rer_vs_exact_m:.4f},{frac:.4f}"
            )

    out.append("")
    out.append("Interpretation rule")
    out.append("For ordinary or moderate missingness, use NRE as the fast posterior engine.")
    out.append("For severe missingness or large-system stress, use NRE as the fast candidate-screening stage and apply exact/high-fidelity likelihood reranking to the retained credible set.")
    out.append("This preserves the fast amortized interface while making the severe-missing boundary evidence-backed rather than text-only.")
    return "\n".join(out) + "\n"


def main():
    t0 = time.time()
    rows = []
    print(f"Device: {DEVICE}")
    print("Evaluating 119-bus IP1 candidate rerank...")
    rows.extend(evaluate_119())
    print("Evaluating 300-bus IP-C 30% missing candidate rerank...")
    rows.extend(evaluate_300())
    text = format_summary(rows)
    text += f"elapsed_sec={time.time() - t0:.1f}\n"

    local_out = os.path.join(ROOT, OUT_NAME)
    with open(local_out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved: {local_out}")

    if os.path.isdir(PKG_STATS):
        pkg_out = os.path.join(PKG_STATS, OUT_NAME)
        with open(pkg_out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved: {pkg_out}")


if __name__ == "__main__":
    main()
