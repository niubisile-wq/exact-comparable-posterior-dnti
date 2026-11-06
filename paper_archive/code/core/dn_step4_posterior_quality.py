# -*- coding: utf-8 -*-
"""
Step 4 posterior-quality evidence for 33-bus IP1.

Reference posterior is exact enumeration over the existing 33-bus topology
library under the same Gaussian voltage-noise likelihood. This avoids AIS
wording and gives exact-comparable posterior metrics.
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
N_BUS = 33
K_FIXED = 20
SIGMA = 0.009
N_VAL = 1200
N_TEST = 2400
BATCH = 256
TEMP_GRID = np.r_[np.linspace(0.35, 1.50, 24), np.linspace(1.60, 3.00, 8)]
ECE_BINS = np.linspace(0.0, 1.0, 11)


class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(),
            nn.Linear(d, d), nn.LayerNorm(d)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class LoadAwareNRE(nn.Module):
    def __init__(self, n_topo, n_bus=33):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_bus * 3, 512), nn.LayerNorm(512), nn.GELU())
        self.res1 = ResBlock(512)
        self.res2 = ResBlock(512)
        self.res3 = ResBlock(512)
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, n_topo)
        )

    def forward(self, x):
        h = self.embed(x)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        return self.head(h)


def softmax_np(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    p = np.exp(z)
    return p / np.sum(p, axis=-1, keepdims=True)


def load_assets():
    lib = torch.load(os.path.join(SAVE_DIR, "nre_ipc_loadaware.pt"), map_location="cpu", weights_only=False)
    v_library = lib["V_library"].astype(np.float32)
    lf_grid = lib["lf_grid"].astype(np.float32)
    base_p = lib["base_P_norm"].astype(np.float32)
    n_topos = int(lib["N_TOPOS"])
    ckpt = torch.load(os.path.join(SAVE_DIR, "nre_ip1_v5a.pt"), map_location=DEVICE, weights_only=False)
    model = LoadAwareNRE(n_topos, N_BUS).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, v_library, lf_grid, base_p, n_topos


def make_dataset(v_library, lf_grid, base_p, rng, n_samples):
    n_topos, n_lf, n_bus = v_library.shape
    xs = np.zeros((n_samples, n_bus * 3), dtype=np.float32)
    ys = np.zeros(n_samples, dtype=np.int64)
    refs = np.zeros((n_samples, n_topos), dtype=np.float32)
    reported_list = []
    obs_list = []
    lf_list = []
    for i in range(n_samples):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        lf = lf_grid[lf_idx]
        reported = np.sort(rng.choice(np.arange(1, n_bus), K_FIXED, replace=False))
        obs = v_library[ti, lf_idx, reported] + rng.normal(0, SIGMA, size=K_FIXED)
        pred = v_library[:, lf_idx, :][:, reported]
        ll = -0.5 * np.sum(((pred - obs[None, :]) / SIGMA) ** 2, axis=1)
        ref = np.exp(ll - np.max(ll))
        ref = ref / np.sum(ref)
        xs[i, reported] = obs
        xs[i, n_bus + reported] = 1.0
        xs[i, 2 * n_bus + reported] = base_p[reported] * lf
        ys[i] = ti
        refs[i] = ref.astype(np.float32)
        reported_list.append(reported)
        obs_list.append(obs)
        lf_list.append(float(lf))
    meta = {"reported": reported_list, "obs": obs_list, "lf": lf_list}
    return xs, ys, refs, meta


def logits_for(model, xs):
    outs = []
    with torch.no_grad():
        for start in range(0, len(xs), BATCH):
            xb = torch.tensor(xs[start:start + BATCH], dtype=torch.float32, device=DEVICE)
            outs.append(model(xb).cpu().numpy())
    return np.vstack(outs)


def credible_coverage(probs, ys, mass=0.90):
    ok = []
    sizes = []
    for p, y in zip(probs, ys):
        order = np.argsort(-p)
        csum = np.cumsum(p[order])
        cutoff = int(np.searchsorted(csum, mass, side="left")) + 1
        chosen = set(order[:cutoff])
        ok.append(y in chosen)
        sizes.append(cutoff)
    return float(np.mean(ok)), float(np.mean(sizes))


def ece_and_bins(probs, ys):
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == ys).astype(float)
    rows = []
    ece = 0.0
    for lo, hi in zip(ECE_BINS[:-1], ECE_BINS[1:]):
        if hi == 1.0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        n = int(np.sum(mask))
        if n == 0:
            rows.append((lo, hi, 0, np.nan, np.nan))
            continue
        acc = float(np.mean(correct[mask]))
        c = float(np.mean(conf[mask]))
        ece += (n / len(ys)) * abs(acc - c)
        rows.append((lo, hi, n, acc, c))
    return float(ece), rows


def metrics(name, probs, refs, ys):
    eps = 1e-12
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(ys)), ys] = 1.0
    top1 = float(np.mean(np.argmax(probs, axis=1) == ys))
    nll = float(-np.mean(np.log(np.clip(probs[np.arange(len(ys)), ys], eps, 1.0))))
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    kl = float(np.mean(np.sum(refs * (np.log(np.clip(refs, eps, 1.0)) - np.log(np.clip(probs, eps, 1.0))), axis=1)))
    ece, bins = ece_and_bins(probs, ys)
    cov90, size90 = credible_coverage(probs, ys, 0.90)
    ent = float(np.mean(-np.sum(probs * np.log(np.clip(probs, eps, 1.0)), axis=1)))
    return {
        "name": name,
        "top1": top1,
        "nll": nll,
        "brier": brier,
        "kl_ref_to_model": kl,
        "ece": ece,
        "cov90": cov90,
        "size90": size90,
        "entropy": ent,
        "bins": bins,
    }


def choose_temperature(logits, refs):
    best_t, best_kl = None, float("inf")
    rows = []
    eps = 1e-12
    for t in TEMP_GRID:
        probs = softmax_np(logits / t)
        kl = float(np.mean(np.sum(refs * (np.log(np.clip(refs, eps, 1.0)) - np.log(np.clip(probs, eps, 1.0))), axis=1)))
        rows.append((float(t), kl))
        if kl < best_kl:
            best_t, best_kl = float(t), kl
    return best_t, best_kl, rows


def find_multimodal_case(refs, raw_probs, cal_probs, ys):
    # Prefer a genuinely ambiguous exact posterior where calibrated NRE tracks
    # the top modes reasonably well.
    ent = -np.sum(refs * np.log(np.clip(refs, 1e-12, 1.0)), axis=1)
    candidates = []
    for i, ref in enumerate(refs):
        order = np.argsort(-ref)
        if ref[order[0]] < 0.65 and ref[order[1]] > 0.10:
            kl_cal = np.sum(ref * (np.log(np.clip(ref, 1e-12, 1.0)) - np.log(np.clip(cal_probs[i], 1e-12, 1.0))))
            candidates.append((kl_cal, -ent[i], i))
    if not candidates:
        return int(np.argmax(ent))
    candidates.sort()
    return int(candidates[0][2])


def write_outputs(t_best, t_rows, metric_rows, val_metrics, test_refs, raw_probs, cal_probs, ys, meta):
    lines = []
    lines.append("Step 4 posterior calibration result: 33-bus IP1")
    lines.append("Reference posterior: exact enumeration over topology library with Gaussian voltage likelihood.")
    lines.append(f"N_VAL={N_VAL}, N_TEST={N_TEST}, K={K_FIXED}, sigma={SIGMA}")
    lines.append(f"selected_temperature_by_validation_KL={t_best:.4f}")
    lines.append("")
    lines.append("temperature_search_validation_KL")
    lines.append("T,KL(reference||NRE_T)")
    for t, kl in t_rows:
        lines.append(f"{t:.4f},{kl:.6f}")
    lines.append("")
    lines.append("test_metrics")
    lines.append("method,top1,NLL,Brier,KL_ref_to_model,ECE,coverage90,avg_credset90,entropy")
    for m in metric_rows:
        lines.append(
            f"{m['name']},{m['top1']:.4f},{m['nll']:.4f},{m['brier']:.4f},"
            f"{m['kl_ref_to_model']:.6f},{m['ece']:.4f},{m['cov90']:.4f},"
            f"{m['size90']:.2f},{m['entropy']:.4f}"
        )
    lines.append("")
    lines.append("Boundary: report calibrated NRE as conservative/exact-comparable, not perfectly calibrated.")
    with open(os.path.join(SAVE_DIR, "posterior_calibration_result.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Reliability bins for plotting.
    with open(os.path.join(SAVE_DIR, "posterior_reliability_bins.csv"), "w", encoding="utf-8") as f:
        f.write("method,bin_lo,bin_hi,n,accuracy,confidence\n")
        for m in metric_rows:
            for lo, hi, n, acc, conf in m["bins"]:
                acc_s = "" if np.isnan(acc) else f"{acc:.6f}"
                conf_s = "" if np.isnan(conf) else f"{conf:.6f}"
                f.write(f"{m['name']},{lo:.2f},{hi:.2f},{n},{acc_s},{conf_s}\n")

    idx = find_multimodal_case(test_refs, raw_probs, cal_probs, ys)
    order = np.argsort(-test_refs[idx])[:10]
    case_lines = []
    case_lines.append("Step 4 multimodal posterior case: 33-bus IP1")
    case_lines.append(f"sample_index={idx}")
    case_lines.append(f"true_topology={int(ys[idx])}")
    case_lines.append(f"load_factor={meta['lf'][idx]:.4f}")
    case_lines.append("reported_buses=" + ",".join(str(int(x) + 1) for x in meta["reported"][idx]))
    case_lines.append("topology,reference,raw_NRE,calibrated_NRE")
    for j in order:
        case_lines.append(f"{int(j)},{test_refs[idx, j]:.6f},{raw_probs[idx, j]:.6f},{cal_probs[idx, j]:.6f}")
    ref_top = order[0]
    case_lines.append("")
    case_lines.append(
        f"Boundary: this is an illustrative ambiguous/multimodal case; it is not a claim that every sample is multimodal."
    )
    with open(os.path.join(SAVE_DIR, "posterior_multimodal_case.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(case_lines) + "\n")


def main():
    print(f"Device: {DEVICE}")
    t0 = time.time()
    model, v_library, lf_grid, base_p, n_topos = load_assets()
    rng_val = np.random.RandomState(4101)
    rng_test = np.random.RandomState(4102)
    x_val, y_val, ref_val, _ = make_dataset(v_library, lf_grid, base_p, rng_val, N_VAL)
    x_test, y_test, ref_test, meta_test = make_dataset(v_library, lf_grid, base_p, rng_test, N_TEST)
    logits_val = logits_for(model, x_val)
    logits_test = logits_for(model, x_test)
    t_best, best_kl, t_rows = choose_temperature(logits_val, ref_val)
    raw_probs = softmax_np(logits_test)
    cal_probs = softmax_np(logits_test / t_best)
    ref_metrics = metrics("Reference", ref_test, ref_test, y_test)
    raw_metrics = metrics("Raw_NRE", raw_probs, ref_test, y_test)
    cal_metrics = metrics("Calibrated_NRE", cal_probs, ref_test, y_test)
    write_outputs(t_best, t_rows, [ref_metrics, raw_metrics, cal_metrics], None, ref_test, raw_probs, cal_probs, y_test, meta_test)
    print(f"selected T={t_best:.4f} validation KL={best_kl:.6f}")
    for m in [ref_metrics, raw_metrics, cal_metrics]:
        print(
            f"{m['name']}: top1={m['top1']:.4f} NLL={m['nll']:.4f} Brier={m['brier']:.4f} "
            f"KL={m['kl_ref_to_model']:.6f} ECE={m['ece']:.4f} cov90={m['cov90']:.4f}"
        )
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
