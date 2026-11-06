# -*- coding: utf-8 -*-
"""
69-bus BOED sensor-efficiency bootstrap audit.

Focused per-sample comparison of AdaptiveBOED against AdaptiveMVG and Random at
K=4 and K=7 using the frozen 69-bus voltage library. No pandapower dependency.
"""

from pathlib import Path
import time
import numpy as np

ROOT = Path.home() / "Desktop" / "配电网实验_临时"
PKG_ROOT = Path.home() / "Desktop" / "配电网论文一区投稿成果包_20260702"
PKG_STATS = PKG_ROOT / "03_frozen_tables_stats"
PKG_CODE = PKG_ROOT / "02_code"
OUT_NAME = "69bus_boed_bootstrap_ci_20260702.txt"

SIGMA = 0.009
N_LF_C = 11
K_REPORT = [4, 7]
K_MAX = 7
N_TEST = 100
N_MC = 30
BOOT = 1200


def softmax_stable(logw):
    z = logw - np.max(logw)
    w = np.exp(z)
    return w / np.sum(w)


def topo_posterior_from_joint(joint_w, n_topos, n_lf_c):
    p = joint_w.reshape(n_topos, n_lf_c).sum(axis=1)
    return p / p.sum()


def posterior_joint(v_flat, obs_nodes, obs_vals):
    if len(obs_nodes) == 0:
        return np.ones(v_flat.shape[0]) / v_flat.shape[0]
    diff = (v_flat[:, obs_nodes] - np.asarray(obs_vals)[None, :]) / SIGMA
    logw = -0.5 * np.sum(diff * diff, axis=1)
    return softmax_stable(logw)


def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def select_adaptive_mvg(v_flat_c, candidates, full_v):
    selected, remaining = [], list(candidates)
    obs_vals = []
    for _ in range(K_MAX):
        joint = posterior_joint(v_flat_c, selected, obs_vals)
        rem = np.array(remaining, dtype=np.int64)
        mean = joint @ v_flat_c[:, rem]
        second = joint @ (v_flat_c[:, rem] ** 2)
        var = second - mean * mean
        node = int(rem[int(np.argmax(var))])
        selected.append(node)
        obs_vals.append(float(full_v[node]))
        remaining.remove(node)
    return selected


def select_adaptive_boed(v_flat_c, candidates, n_topos, n_lf_c, full_v, rng):
    selected, remaining = [], list(candidates)
    obs_vals = []
    for _ in range(K_MAX):
        current_joint = posterior_joint(v_flat_c, selected, obs_vals)
        current_h = entropy(topo_posterior_from_joint(current_joint, n_topos, n_lf_c))
        sample_idx = rng.choice(v_flat_c.shape[0], size=N_MC, p=current_joint)
        best_node, best_eig = None, -np.inf
        for node in remaining:
            h_after = 0.0
            y_samples = v_flat_c[sample_idx, node] + rng.normal(0.0, SIGMA, size=N_MC)
            for y in y_samples:
                joint_new = posterior_joint(v_flat_c, selected + [node], obs_vals + [float(y)])
                h_after += entropy(topo_posterior_from_joint(joint_new, n_topos, n_lf_c))
            eig = current_h - h_after / N_MC
            if eig > best_eig:
                best_eig, best_node = eig, node
        selected.append(best_node)
        obs_vals.append(float(full_v[best_node]))
        remaining.remove(best_node)
    return selected


def evaluate(v_flat_c, order, full_v, true_ti, n_topos, n_lf_c, k):
    nodes = order[:k]
    obs = full_v[nodes]
    joint = posterior_joint(v_flat_c, nodes, obs)
    p = topo_posterior_from_joint(joint, n_topos, n_lf_c)
    return int(np.argmax(p) == true_ti), entropy(p)


def boot_mean(arr):
    arr = np.asarray(arr, dtype=np.float64)
    rng = np.random.RandomState(20260704)
    vals = np.empty(BOOT, dtype=np.float64)
    for b in range(BOOT):
        idx = rng.randint(0, len(arr), size=len(arr))
        vals[b] = np.mean(arr[idx])
    return float(np.mean(arr)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def main():
    t0 = time.time()
    dat = np.load(ROOT / "v_library_69bus.npz")
    v_library = dat["V_library"]
    n_topos, n_lf, n_bus = v_library.shape
    lf_idx_c = np.round(np.linspace(0, n_lf - 1, N_LF_C)).astype(int)
    v_c = v_library[:, lf_idx_c, :]
    v_flat_c = v_c.reshape(n_topos * N_LF_C, n_bus)
    candidates = list(range(1, n_bus))
    rng = np.random.RandomState(20260704)
    test_cases = []
    for _ in range(N_TEST):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        full_v = v_library[ti, lf_idx, :] + rng.normal(0.0, SIGMA, size=n_bus)
        test_cases.append((ti, lf_idx, full_v))

    methods = ["Random", "AdaptiveMVG", "AdaptiveBOED"]
    acc = {m: {k: [] for k in K_REPORT} for m in methods}
    ent = {m: {k: [] for k in K_REPORT} for m in methods}

    for ci, (true_ti, _lf_idx, full_v) in enumerate(test_cases):
        if ci % 20 == 0:
            print(f"case {ci}/{N_TEST} elapsed={time.time()-t0:.1f}s", flush=True)
        rng_rand = np.random.RandomState(17000 + ci)
        rand_order = list(candidates)
        rng_rand.shuffle(rand_order)
        orders = {
            "Random": rand_order[:K_MAX],
            "AdaptiveMVG": select_adaptive_mvg(v_flat_c, candidates, full_v),
            "AdaptiveBOED": select_adaptive_boed(v_flat_c, candidates, n_topos, N_LF_C, full_v, np.random.RandomState(91000 + ci)),
        }
        for name, order in orders.items():
            for k in K_REPORT:
                ok, h = evaluate(v_flat_c, order, full_v, true_ti, n_topos, N_LF_C, k)
                acc[name][k].append(ok)
                ent[name][k].append(h)

    lines = []
    lines.append("69-bus BOED sensor-efficiency bootstrap audit")
    lines.append("date=2026-07-02")
    lines.append("role=focused per-sample BOED/MVG/Random comparison with bootstrap CI")
    lines.append(f"N_TEST={N_TEST}")
    lines.append(f"N_MC={N_MC}")
    lines.append(f"N_LF_C={N_LF_C}")
    lines.append(f"bootstrap_replicates={BOOT}")
    lines.append("metric,method_or_delta,K,mean,ci95_low,ci95_high")
    for k in K_REPORT:
        for name in methods:
            mean, lo, hi = boot_mean(acc[name][k])
            lines.append(f"accuracy,{name},{k},{mean:.6f},{lo:.6f},{hi:.6f}")
        for name in methods:
            mean, lo, hi = boot_mean(ent[name][k])
            lines.append(f"entropy,{name},{k},{mean:.6f},{lo:.6f},{hi:.6f}")
        for other in ["Random", "AdaptiveMVG"]:
            diff_acc = np.asarray(acc["AdaptiveBOED"][k], dtype=float) - np.asarray(acc[other][k], dtype=float)
            mean, lo, hi = boot_mean(diff_acc)
            lines.append(f"accuracy_delta_BOED_minus_{other},AdaptiveBOED-{other},{k},{mean:.6f},{lo:.6f},{hi:.6f}")
            diff_ent = np.asarray(ent[other][k], dtype=float) - np.asarray(ent["AdaptiveBOED"][k], dtype=float)
            mean, lo, hi = boot_mean(diff_ent)
            lines.append(f"entropy_reduction_BOED_vs_{other},{other}-AdaptiveBOED,{k},{mean:.6f},{lo:.6f},{hi:.6f}")
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
