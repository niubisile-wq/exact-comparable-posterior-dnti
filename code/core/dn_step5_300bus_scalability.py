# -*- coding: utf-8 -*-
"""
Step 5 synthetic 300-bus scalability experiment.

Uses v_library_300bus.npz from build_300bus.py.
Outputs:
  - ip4_300bus_result.txt
  - ip1_300bus_result.txt
  - ipc_300bus_result.txt
  - scalability_result.txt
"""
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 123, 456]
K_FIXED = 150
SIGMA = 0.0015
BATCH = 512
CLEAN_TRAIN_SAMPLES = 120000
ROBUST_TRAIN_SAMPLES = 180000
CLEAN_EPOCHS = 10
ROBUST_EPOCHS = 24
LR = 4e-4
N_EVAL = 1500
HK_LIST = [40, 80, 120, 150, 180, 220]


class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class NRE300(nn.Module):
    def __init__(self, n_topos, n_bus):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_bus * 3, 768), nn.LayerNorm(768), nn.GELU())
        self.res1 = ResBlock(768)
        self.res2 = ResBlock(768)
        self.res3 = ResBlock(768)
        self.head = nn.Sequential(nn.Linear(768, 384), nn.LayerNorm(384), nn.GELU(), nn.Linear(384, n_topos))

    def forward(self, x):
        h = self.embed(x)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        return self.head(h)


def load_lib():
    z = np.load(os.path.join(SAVE_DIR, "v_library_300bus.npz"), allow_pickle=True)
    return {
        "V": z["V_library"].astype(np.float32),
        "lf": z["lf_grid"].astype(np.float32),
        "base_p": z["base_P_norm"].astype(np.float32),
        "n_bus": int(z["n_bus"]),
        "n_topos": int(z["n_topologies"]),
    }


def deployment_sensors(n_bus, k):
    return np.unique(np.rint(np.linspace(1, n_bus - 1, k)).astype(int))


def make_batch(lib, rng, n, robust=False, miss_rate=None, sigma=SIGMA):
    V, lf_grid, base_p = lib["V"], lib["lf"], lib["base_p"]
    n_topos, n_lf, n_bus = V.shape
    xs = np.zeros((n, n_bus * 3), dtype=np.float32)
    ys = np.zeros(n, dtype=np.int64)
    for i in range(n):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        lf = lf_grid[lf_idx]
        installed = deployment_sensors(n_bus, K_FIXED)
        if robust:
            if miss_rate is None:
                # Bias training toward the hardest reported condition while still
                # retaining lower-missing examples for the 10/20/30% curve.
                mr = rng.choice([0.10, 0.20, 0.30], p=[0.20, 0.25, 0.55])
            else:
                mr = miss_rate
            n_miss = int(K_FIXED * mr)
            if n_miss > 0:
                installed = np.delete(installed, rng.choice(len(installed), n_miss, replace=False))
        obs = V[ti, lf_idx, installed] + rng.normal(0.0, sigma, size=len(installed))
        xs[i, installed] = (obs - 1.0) / sigma
        xs[i, n_bus + installed] = 1.0
        xs[i, 2 * n_bus + installed] = base_p[installed] * lf
        ys[i] = ti
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long)


def exact_posterior_from_sample(lib, buses, obs, lf_idx):
    pred_v = lib["V"][:, lf_idx, :][:, buses]
    ll = -0.5 * np.sum(((pred_v - obs[None, :]) / SIGMA) ** 2, axis=1)
    q = np.exp(ll - np.max(ll))
    q = q / np.sum(q)
    return q.astype(np.float32)


def make_distill_dataset(lib, rng, n, robust=False, miss_rate=None):
    V, lf_grid, base_p = lib["V"], lib["lf"], lib["base_p"]
    n_topos, n_lf, n_bus = V.shape
    xs = np.zeros((n, n_bus * 3), dtype=np.float32)
    ys = np.zeros(n, dtype=np.int64)
    qs = np.zeros((n, n_topos), dtype=np.float32)
    for i in range(n):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        lf = lf_grid[lf_idx]
        installed = deployment_sensors(n_bus, K_FIXED)
        if robust:
            if miss_rate is None:
                mr = rng.choice([0.10, 0.20, 0.30], p=[0.20, 0.25, 0.55])
            else:
                mr = miss_rate
            n_miss = int(K_FIXED * mr)
            if n_miss > 0:
                installed = np.delete(installed, rng.choice(len(installed), n_miss, replace=False))
        obs = V[ti, lf_idx, installed] + rng.normal(0.0, SIGMA, size=len(installed))
        xs[i, installed] = (obs - 1.0) / SIGMA
        xs[i, n_bus + installed] = 1.0
        xs[i, 2 * n_bus + installed] = base_p[installed] * lf
        ys[i] = ti
        qs[i] = exact_posterior_from_sample(lib, installed, obs, lf_idx)
        if (i + 1) % 30000 == 0:
            print(f"  generated {i + 1}/{n} distillation samples", flush=True)
    return xs, ys, qs


def exact_posterior_predict(lib, xs_np, return_q=False):
    V, lf_grid, base_p = lib["V"], lib["lf"], lib["base_p"]
    n_topos, n_lf, n_bus = V.shape
    preds = []
    qs = []
    t0 = time.perf_counter()
    for x in xs_np:
        mask = x[n_bus:2 * n_bus] > 0.5
        buses = np.where(mask)[0]
        obs = x[buses] * SIGMA + 1.0
        # Load factor is estimated from load feature / base_p at observed buses.
        valid = base_p[buses] > 1e-8
        if np.any(valid):
            lf_est = float(np.median(x[2 * n_bus + buses[valid]] / base_p[buses[valid]]))
        else:
            lf_est = 1.0
        lf_idx = int(np.argmin(np.abs(lf_grid - lf_est)))
        pred_v = V[:, lf_idx, :][:, buses]
        ll = -0.5 * np.sum(((pred_v - obs[None, :]) / SIGMA) ** 2, axis=1)
        q = np.exp(ll - np.max(ll))
        q = q / np.sum(q)
        preds.append(int(np.argmax(q)))
        if return_q:
            qs.append(q.astype(np.float32))
    if return_q:
        return np.array(preds, dtype=np.int64), np.vstack(qs), time.perf_counter() - t0
    return np.array(preds, dtype=np.int64), time.perf_counter() - t0


def entropy_exact(lib, k, n_eval=500):
    rng = np.random.RandomState(9000 + k)
    V, lf_grid = lib["V"], lib["lf"]
    n_topos, n_lf, n_bus = V.shape
    ents = []
    for _ in range(n_eval):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        buses = deployment_sensors(n_bus, k)
        obs = V[ti, lf_idx, buses] + rng.normal(0.0, SIGMA, size=k)
        pred = V[:, lf_idx, :][:, buses]
        ll = -0.5 * np.sum(((pred - obs[None, :]) / SIGMA) ** 2, axis=1)
        p = np.exp(ll - np.max(ll))
        p = p / np.sum(p)
        ents.append(float(-np.sum(p * np.log(np.clip(p, 1e-12, 1.0)))))
    return float(np.mean(ents)), float(np.std(ents))


def train_model(lib, seed, robust=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    model = NRE300(lib["n_topos"], lib["n_bus"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    n_train = ROBUST_TRAIN_SAMPLES if robust else CLEAN_TRAIN_SAMPLES
    epochs = ROBUST_EPOCHS if robust else CLEAN_EPOCHS
    print(f"Building {'IPC' if robust else 'IP1'} distillation set seed={seed} n={n_train} K={K_FIXED}", flush=True)
    xs, ys, qs = make_distill_dataset(lib, rng, n_train, robust=robust)
    steps = int(np.ceil(n_train / BATCH)) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    kl_fn = nn.KLDivLoss(reduction="batchmean")
    ce_weight = 0.30 if robust else 0.10
    t0 = time.time()
    model.train()
    step = 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n_train)
        for start in range(0, n_train, BATCH):
            idx = order[start:start + BATCH]
            xb = torch.tensor(xs[idx], dtype=torch.float32)
            yb = torch.tensor(ys[idx], dtype=torch.long)
            qb = torch.tensor(qs[idx], dtype=torch.float32)
            step += 1
            logits = model(xb.to(DEVICE))
            logp = torch.log_softmax(logits, dim=1)
            loss = kl_fn(logp, qb.to(DEVICE)) + ce_weight * loss_fn(logits, yb.to(DEVICE))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            scheduler.step()
        with torch.no_grad():
            pred = model(torch.tensor(xs[:2000], dtype=torch.float32).to(DEVICE)).argmax(dim=1).cpu().numpy()
        acc_probe = float(np.mean(pred == ys[:2000]))
        print(f"{'IPC' if robust else 'IP1'} seed={seed} epoch={epoch}/{epochs} loss={loss.item():.4f} probe_acc={acc_probe:.4f} elapsed={time.time()-t0:.0f}s", flush=True)
    train_sec = time.time() - t0
    return model, train_sec


def eval_model(model, lib, robust_eval=False, miss_rate=0.0):
    rng = np.random.RandomState(777 + int(miss_rate * 1000) + (100 if robust_eval else 0))
    xb, yb = make_batch(lib, rng, N_EVAL, robust=robust_eval, miss_rate=miss_rate)
    xs_np = xb.numpy()
    with torch.no_grad():
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(xb.to(DEVICE))
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        nre_sec = time.perf_counter() - t0
        pred = logits.argmax(dim=1).cpu().numpy()
    exact_pred, exact_q, exact_sec = exact_posterior_predict(lib, xs_np[:500], return_q=True)
    exact_acc = float(np.mean(exact_pred == yb.numpy()[:500]))
    nre_q = torch.softmax(logits[:500], dim=1).cpu().numpy()
    nre_pred_500 = pred[:500]
    exact_agree = float(np.mean(nre_pred_500 == exact_pred))
    kl_ref_nre = float(np.mean(np.sum(exact_q * (np.log(np.clip(exact_q, 1e-12, 1.0)) - np.log(np.clip(nre_q, 1e-12, 1.0))), axis=1)))
    return {
        "acc": float(np.mean(pred == yb.numpy())),
        "nre_ms": nre_sec / N_EVAL * 1000.0,
        "exact_acc_500": exact_acc,
        "exact_ms": exact_sec / 500 * 1000.0,
        "speedup": (exact_sec / 500) / (nre_sec / N_EVAL),
        "exact_agree": exact_agree,
        "kl_ref_nre": kl_ref_nre,
    }


def write_ip4(lib):
    lines = ["Step 5 300-bus IP4 H(K)", f"n_topologies={lib['n_topos']}", f"sigma={SIGMA}", "sensor_policy=fixed uniform feeder coverage", "K,H_mean,H_std"]
    vals = []
    for k in HK_LIST:
        h, s = entropy_exact(lib, k)
        vals.append((k, h, s))
        lines.append(f"{k},{h:.4f},{s:.4f}")
    monotone = all(vals[i][1] >= vals[i + 1][1] for i in range(len(vals) - 1))
    lines.append(f"monotone_nonincreasing={monotone}")
    lines.append("Boundary: synthetic 300-bus fixed-deployment identifiability stress test, not a real utility feeder.")
    with open(os.path.join(SAVE_DIR, "ip4_300bus_result.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return vals


def main():
    print(f"Device: {DEVICE}")
    lib = load_lib()
    print(f"Loaded 300-bus library: V={lib['V'].shape} n_topos={lib['n_topos']}")
    t_all = time.time()
    h_vals = write_ip4(lib)

    ip1_rows = []
    ipc_rows = []
    for seed in SEEDS:
        model, train_sec = train_model(lib, seed, robust=False)
        res = eval_model(model, lib, robust_eval=False, miss_rate=0.0)
        res.update({"seed": seed, "train_sec": train_sec})
        ip1_rows.append(res)
        torch.save({"model_state": model.state_dict(), "seed": seed, "n_topos": lib["n_topos"], "n_bus": lib["n_bus"]}, os.path.join(SAVE_DIR, f"nre_300bus_ip1_seed{seed}.pt"))

    for seed in SEEDS:
        model, train_sec = train_model(lib, seed, robust=True)
        row = {"seed": seed, "train_sec": train_sec}
        for mr in [0.10, 0.20, 0.30]:
            res = eval_model(model, lib, robust_eval=True, miss_rate=mr)
            row[f"acc_miss{int(mr*100)}"] = res["acc"]
            row[f"nre_ms_miss{int(mr*100)}"] = res["nre_ms"]
            row[f"exact_acc500_miss{int(mr*100)}"] = res["exact_acc_500"]
            row[f"exact_ms_miss{int(mr*100)}"] = res["exact_ms"]
            row[f"exact_agree_miss{int(mr*100)}"] = res["exact_agree"]
            row[f"kl_ref_nre_miss{int(mr*100)}"] = res["kl_ref_nre"]
            row[f"speedup_miss{int(mr*100)}"] = res["speedup"]
        ipc_rows.append(row)
        torch.save({"model_state": model.state_dict(), "seed": seed, "n_topos": lib["n_topos"], "n_bus": lib["n_bus"]}, os.path.join(SAVE_DIR, f"nre_300bus_ipc_seed{seed}.pt"))

    ip1_lines = ["Step 5 300-bus IP1 NRE", f"n_topologies={lib['n_topos']}", f"K={K_FIXED}", f"sigma={SIGMA}", "sensor_policy=fixed uniform feeder coverage", "training=exact-posterior distillation + 0.10 hard-label CE", "seed,acc,train_sec,nre_ms,exact_acc_500,exact_ms,exact_agree,kl_ref_nre,speedup"]
    for r in ip1_rows:
        ip1_lines.append(f"{r['seed']},{r['acc']:.4f},{r['train_sec']:.1f},{r['nre_ms']:.6f},{r['exact_acc_500']:.4f},{r['exact_ms']:.6f},{r['exact_agree']:.4f},{r['kl_ref_nre']:.4f},{r['speedup']:.1f}")
    ip1_lines.append(f"mean_acc={np.mean([r['acc'] for r in ip1_rows]):.4f}")
    ip1_lines.append(f"std_acc={np.std([r['acc'] for r in ip1_rows]):.4f}")
    with open(os.path.join(SAVE_DIR, "ip1_300bus_result.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ip1_lines) + "\n")
    print("\n".join(ip1_lines))

    ipc_lines = ["Step 5 300-bus IP-C robust NRE", f"n_topologies={lib['n_topos']}", f"K={K_FIXED}", f"sigma={SIGMA}", "sensor_policy=fixed uniform feeder coverage; missing sensors sampled from deployment set", "training=exact-posterior distillation + 0.30 hard-label CE", "seed,train_sec,acc10,acc20,acc30,nre_ms30,exact_acc500_30,exact_ms30,exact_agree30,kl_ref_nre30,speedup30"]
    for r in ipc_rows:
        ipc_lines.append(f"{r['seed']},{r['train_sec']:.1f},{r['acc_miss10']:.4f},{r['acc_miss20']:.4f},{r['acc_miss30']:.4f},{r['nre_ms_miss30']:.6f},{r['exact_acc500_miss30']:.4f},{r['exact_ms_miss30']:.6f},{r['exact_agree_miss30']:.4f},{r['kl_ref_nre_miss30']:.4f},{r['speedup_miss30']:.1f}")
    ipc_lines.append(f"mean_acc30={np.mean([r['acc_miss30'] for r in ipc_rows]):.4f}")
    ipc_lines.append(f"std_acc30={np.std([r['acc_miss30'] for r in ipc_rows]):.4f}")
    with open(os.path.join(SAVE_DIR, "ipc_300bus_result.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ipc_lines) + "\n")
    print("\n".join(ipc_lines))

    scal = []
    scal.append("Step 5 scalability summary")
    scal.append("system,n_bus,n_topologies,IP1_acc_mean,IPC_acc30_mean,NRE_ms_mean,Exact_ms_mean,Speedup_mean")
    scal.append(
        f"synthetic_300bus,{lib['n_bus']},{lib['n_topos']},"
        f"{np.mean([r['acc'] for r in ip1_rows]):.4f},"
        f"{np.mean([r['acc_miss30'] for r in ipc_rows]):.4f},"
        f"{np.mean([r['nre_ms'] for r in ip1_rows]):.6f},"
        f"{np.mean([r['exact_ms'] for r in ip1_rows]):.6f},"
        f"{np.mean([r['speedup'] for r in ip1_rows]):.1f}"
    )
    scal.append(f"total_elapsed_sec={time.time() - t_all:.1f}")
    scal.append("Boundary: synthetic mid-scale fixed-deployment scalability evidence only; do not claim real utility full-scale validation.")
    with open(os.path.join(SAVE_DIR, "scalability_result.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(scal) + "\n")
    print("\n".join(scal))


if __name__ == "__main__":
    main()
