# -*- coding: utf-8 -*-
"""
IEEE123 controlled three-phase composite stress audit.

Evaluates the frozen nominal K75 model and the frozen dropout-robust K75 model
under simultaneous random sensor dropout and measurement-noise inflation. This
is an experiment-level robustness audit, not a wording change, and it does not
overwrite any checkpoints.
"""

from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path.home() / "Desktop" / "配电网实验_临时"
sys.path.insert(0, str(ROOT))

import dn_ieee123_3ph_K75_warmstart_20260702 as m123

PKG_ROOT = Path.home() / "Desktop" / "配电网论文一区投稿成果包_20260702"
PKG_STATS = PKG_ROOT / "03_frozen_tables_stats"
PKG_CODE = PKG_ROOT / "02_code"
OUT_NAME = "ieee123_3ph_composite_stress_20260703.txt"

DEVICE = m123.DEVICE
K_BASE = 75
SEEDS = [42, 123, 456]
BATCH = 256
N_TEST = 1800

SCENARIOS = [
    ("nominal", 0.00, 1.0),
    ("noise2x", 0.00, 2.0),
    ("noise3x", 0.00, 3.0),
    ("drop10_noise1x", 0.10, 1.0),
    ("drop20_noise1x", 0.20, 1.0),
    ("drop40_noise1x", 0.40, 1.0),
    ("drop10_noise2x", 0.10, 2.0),
    ("drop20_noise2x", 0.20, 2.0),
    ("drop20_noise3x", 0.20, 3.0),
    ("drop40_noise2x", 0.40, 2.0),
]


def exact_posterior_sigma(v_library, buses, obs_v3, lf_idx, sigma_eff):
    da = (v_library[:, lf_idx, :, 0][:, buses] - obs_v3[:, 0]) / sigma_eff
    db = (v_library[:, lf_idx, :, 1][:, buses] - obs_v3[:, 1]) / sigma_eff
    dc = (v_library[:, lf_idx, :, 2][:, buses] - obs_v3[:, 2]) / sigma_eff
    ll = -0.5 * (np.sum(da * da, axis=1) + np.sum(db * db, axis=1) + np.sum(dc * dc, axis=1))
    q = np.exp(ll - np.max(ll))
    q /= np.sum(q)
    return q.astype(np.float32)


def make_dataset(lib, n, seed, drop_rate, noise_mult):
    rng = np.random.RandomState(seed)
    v = lib["v"]
    n_topos, n_lf, n_bus, _ = v.shape
    deploy = m123.deployment(n_bus, K_BASE)
    keep_n = max(1, int(round(len(deploy) * (1.0 - drop_rate))))
    sigma_eff = m123.SIGMA * noise_mult
    xs = np.zeros((n, n_bus * 5), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    q = np.zeros((n, n_topos), dtype=np.float32)
    retained = np.zeros(n, dtype=np.int64)
    for i in range(n):
        buses = np.sort(rng.choice(deploy, keep_n, replace=False))
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        lf = float(m123.LF_GRID[lf_idx])
        obs = v[ti, lf_idx, buses, :] + rng.normal(0.0, sigma_eff, size=(len(buses), 3))
        xs[i] = m123.build_features(obs, buses, n_bus, lib["base_p_norm"], lf)
        y[i] = ti
        q[i] = exact_posterior_sigma(v, buses, obs, lf_idx, sigma_eff)
        retained[i] = len(buses)
    return xs, y, q, retained


def predict_probs(model, xs):
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(xs), BATCH):
            xb = torch.tensor(xs[start:start + BATCH], dtype=torch.float32, device=DEVICE)
            probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.vstack(probs)


def avg_probs(models, xs):
    return np.mean(np.stack([predict_probs(model, xs) for model in models], axis=0), axis=0)


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
    arrays = {
        "direct_correct": direct == y,
        "exact_agree": direct == exact_pred,
    }
    for k in [3, 5, 10, 20]:
        cand = top[:, :k]
        truth_in = np.array([y[i] in cand[i] for i in range(len(y))], dtype=bool)
        exact_in = np.array([exact_pred[i] in cand[i] for i in range(len(y))], dtype=bool)
        rerank = np.array([cand[i, int(np.argmax(q[i, cand[i]]))] for i in range(len(y))], dtype=np.int64)
        out[f"truth_top{k}"] = float(np.mean(truth_in))
        out[f"exact_top{k}"] = float(np.mean(exact_in))
        out[f"rerank{k}"] = float(np.mean(rerank == y))
        arrays[f"truth_top{k}"] = truth_in
        arrays[f"exact_top{k}"] = exact_in
        arrays[f"rerank{k}"] = rerank == y
    return out, arrays


def boot_ci_diff(a, b, boot=1000):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rng = np.random.RandomState(12375)
    vals = np.empty(boot, dtype=np.float64)
    for j in range(boot):
        idx = rng.randint(0, len(a), size=len(a))
        vals[j] = np.mean(a[idx] - b[idx])
    return float(np.mean(a - b)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def load_model(lib, seed, tag):
    model = m123.Controlled3PhNRE(lib["n_topos"], lib["n_bus"]).to(DEVICE)
    if tag == "fixed":
        ckpt = ROOT / f"nre_ieee123_3ph_K75_warmstart_seed{seed}_20260702.pt"
    elif tag == "dropout_robust":
        ckpt = ROOT / f"nre_ieee123_3ph_K75_dropout_robust_seed{seed}_20260702.pt"
    else:
        raise ValueError(tag)
    obj = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(obj["model_state"])
    return model


def main():
    t0 = time.time()
    lib = m123.load_library()
    fixed_models = [load_model(lib, seed, "fixed") for seed in SEEDS]
    robust_models = [load_model(lib, seed, "dropout_robust") for seed in SEEDS]
    lines = []
    lines.append("IEEE123 controlled three-phase composite stress audit")
    lines.append("date=2026-07-03")
    lines.append("role=simultaneous random sensor dropout and noise-inflation robustness for controlled unbalanced IEEE123 posterior inference")
    lines.append("not_claimed=utility field deployment; evaluates frozen K75 checkpoints without retraining")
    lines.append(f"device={DEVICE}")
    lines.append(f"n_bus={lib['n_bus']}")
    lines.append(f"n_topologies={lib['n_topos']}")
    lines.append(f"base_K={K_BASE}")
    lines.append(f"test_samples_per_scenario={N_TEST}")
    lines.append("scenario_eval")
    lines.append("scenario,drop_rate,noise_mult,retained_K,model,exact_top1,direct_top1,exact_agree,kl,truth_top3,truth_top5,truth_top10,truth_top20,rerank3,rerank5,rerank10,rerank20,direct_gain_ci_vs_fixed,top20_gain_ci_vs_fixed,rerank20_gain_ci_vs_fixed")
    for si, (name, drop_rate, noise_mult) in enumerate(SCENARIOS):
        xs, y, q, retained = make_dataset(lib, N_TEST, 880000 + si * 97, drop_rate, noise_mult)
        p_fixed = avg_probs(fixed_models, xs)
        p_robust = avg_probs(robust_models, xs)
        p_dual = 0.5 * p_fixed + 0.5 * p_robust
        fixed_met, fixed_arr = metrics(p_fixed, y, q)
        results = [("fixed_K75", fixed_met, fixed_arr)]
        for label, p in [("dropout_robust", p_robust), ("dual_average", p_dual)]:
            met, arr = metrics(p, y, q)
            results.append((label, met, arr))
        for label, met, arr in results:
            if label == "fixed_K75":
                dg = tg = rg = ""
            else:
                direct_gain = boot_ci_diff(arr["direct_correct"], fixed_arr["direct_correct"])
                top20_gain = boot_ci_diff(arr["truth_top20"], fixed_arr["truth_top20"])
                rerank20_gain = boot_ci_diff(arr["rerank20"], fixed_arr["rerank20"])
                dg = f"{direct_gain[0]:.4f}[{direct_gain[1]:.4f},{direct_gain[2]:.4f}]"
                tg = f"{top20_gain[0]:.4f}[{top20_gain[1]:.4f},{top20_gain[2]:.4f}]"
                rg = f"{rerank20_gain[0]:.4f}[{rerank20_gain[1]:.4f},{rerank20_gain[2]:.4f}]"
            lines.append(
                f"{name},{drop_rate:.2f},{noise_mult:.1f},{int(np.mean(retained))},{label},"
                f"{met['exact_top1']:.4f},{met['direct_top1']:.4f},{met['exact_agree']:.4f},"
                f"{met['kl_ref_nre']:.4f},{met['truth_top3']:.4f},{met['truth_top5']:.4f},"
                f"{met['truth_top10']:.4f},{met['truth_top20']:.4f},{met['rerank3']:.4f},"
                f"{met['rerank5']:.4f},{met['rerank10']:.4f},{met['rerank20']:.4f},{dg},{tg},{rg}"
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
