# -*- coding: utf-8 -*-
"""
119-bus BOED dropout-stress audit.

Evaluates whether sensor-selection strategies retain decision value when a
fraction of the selected measurements are dropped after selection.
"""

from pathlib import Path
import time
import numpy as np

ROOT = Path.home() / "Desktop" / "配电网实验_临时"
PKG_ROOT = Path.home() / "Desktop" / "配电网论文一区投稿成果包_20260702"
PKG_STATS = PKG_ROOT / "03_frozen_tables_stats"

OUT_NAME = "boed_119bus_dropout_stress_20260703.txt"
SIGMA = 0.009
N_LF_C = 11
K_REPORT = [5, 7, 10]
K_MAX = 10
N_TEST = 90
N_MC = 32
BOOT = 800
DROPOUTS = [0.0, 0.2, 0.4]


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


def select_adaptive_fisher(v_flat_c, candidates, full_v):
    selected, remaining = [], list(candidates)
    obs_vals = []
    for _ in range(K_MAX):
        joint = posterior_joint(v_flat_c, selected, obs_vals)
        best_node, best_score = None, -np.inf
        for node in remaining:
            nodes = selected + [node]
            x = v_flat_c[:, nodes]
            mu = joint @ x
            xc = x - mu[None, :]
            fim = (xc * joint[:, None]).T @ xc / (SIGMA * SIGMA)
            sign, logdet = np.linalg.slogdet(fim + 1e-6 * np.eye(len(nodes)))
            score = logdet if sign > 0 else -np.inf
            if score > best_score:
                best_score, best_node = score, node
        selected.append(best_node)
        obs_vals.append(float(full_v[best_node]))
        remaining.remove(best_node)
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
            y_samples = v_flat_c[sample_idx, node] + rng.normal(0.0, SIGMA, size=N_MC)
            h_after = 0.0
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


def evaluate_order(v_flat_c, order, full_v, true_ti, n_topos, n_lf_c, k, drop_rate, rng):
    nodes = order[:k]
    keep_mask = rng.rand(len(nodes)) >= drop_rate
    if not np.any(keep_mask):
        kept_nodes = []
        obs = []
    else:
        kept_nodes = list(np.asarray(nodes, dtype=np.int64)[keep_mask])
        obs = full_v[kept_nodes]
    joint = posterior_joint(v_flat_c, kept_nodes, obs)
    p = topo_posterior_from_joint(joint, n_topos, n_lf_c)
    return int(np.argmax(p) == true_ti), entropy(p), len(kept_nodes)


def boot_mean(arr):
    arr = np.asarray(arr, dtype=np.float64)
    rng = np.random.RandomState(20260706)
    vals = np.empty(BOOT, dtype=np.float64)
    for b in range(BOOT):
        idx = rng.randint(0, len(arr), size=len(arr))
        vals[b] = np.mean(arr[idx])
    return float(np.mean(arr)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def main():
    t0 = time.time()
    dat = np.load(ROOT / "v_library_119bus.npz")
    v_library = dat["V_library"]
    n_topos, n_lf, n_bus = v_library.shape
    lf_idx_c = np.round(np.linspace(0, n_lf - 1, N_LF_C)).astype(int)
    v_c = v_library[:, lf_idx_c, :]
    v_flat_c = v_c.reshape(n_topos * N_LF_C, n_bus)
    candidates = list(range(1, n_bus))

    rng = np.random.RandomState(20260706)
    test_cases = []
    for _ in range(N_TEST):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        noisy_full_v = v_library[ti, lf_idx, :] + rng.normal(0.0, SIGMA, size=n_bus)
        test_cases.append((ti, noisy_full_v))

    methods = ["Random", "AdaptiveMVG", "AdaptiveFisher", "AdaptiveBOED"]
    acc = {d: {m: {k: [] for k in K_REPORT} for m in methods} for d in DROPOUTS}
    ent = {d: {m: {k: [] for k in K_REPORT} for m in methods} for d in DROPOUTS}
    kept = {d: {m: {k: [] for k in K_REPORT} for m in methods} for d in DROPOUTS}

    for ci, (true_ti, full_v) in enumerate(test_cases):
        if ci % 10 == 0:
            print(f"case {ci}/{N_TEST} elapsed={time.time()-t0:.1f}s", flush=True)
        rng_rand = np.random.RandomState(62000 + ci)
        rand_order = list(candidates)
        rng_rand.shuffle(rand_order)
        orders = {
            "Random": rand_order[:K_MAX],
            "AdaptiveMVG": select_adaptive_mvg(v_flat_c, candidates, full_v),
            "AdaptiveFisher": select_adaptive_fisher(v_flat_c, candidates, full_v),
            "AdaptiveBOED": select_adaptive_boed(v_flat_c, candidates, n_topos, N_LF_C, full_v, np.random.RandomState(92000 + ci)),
        }
        for d in DROPOUTS:
            for name, order in orders.items():
                for k in K_REPORT:
                    ok, h, kept_n = evaluate_order(
                        v_flat_c,
                        order,
                        full_v,
                        true_ti,
                        n_topos,
                        N_LF_C,
                        k,
                        d,
                        np.random.RandomState(150000 + ci * 37 + int(100 * d) * 7 + k * 3 + len(name)),
                    )
                    acc[d][name][k].append(ok)
                    ent[d][name][k].append(h)
                    kept[d][name][k].append(kept_n)

    lines = []
    lines.append("119-bus BOED dropout-stress audit")
    lines.append("date=2026-07-03")
    lines.append("role=adaptive sensing decision-value under post-selection sensor dropout")
    lines.append("not_claimed=installation cost modeling")
    lines.append(f"N_TEST={N_TEST}")
    lines.append(f"N_MC={N_MC}")
    lines.append(f"K_REPORT={' '.join(map(str, K_REPORT))}")
    lines.append(f"drop_rates={' '.join(map(str, DROPOUTS))}")
    lines.append("per_setting_metrics")
    lines.append("drop_rate,metric,method_or_delta,K,mean,ci95_low,ci95_high")
    for d in DROPOUTS:
        for k in K_REPORT:
            for name in methods:
                mean, lo, hi = boot_mean(acc[d][name][k])
                lines.append(f"{d:.1f},accuracy,{name},{k},{mean:.6f},{lo:.6f},{hi:.6f}")
            for name in methods:
                mean, lo, hi = boot_mean(ent[d][name][k])
                lines.append(f"{d:.1f},entropy,{name},{k},{mean:.6f},{lo:.6f},{hi:.6f}")
            for name in methods:
                mean, lo, hi = boot_mean(kept[d][name][k])
                lines.append(f"{d:.1f},kept_sensors,{name},{k},{mean:.6f},{lo:.6f},{hi:.6f}")
            for other in ["Random", "AdaptiveMVG", "AdaptiveFisher"]:
                diff_acc = np.asarray(acc[d]["AdaptiveBOED"][k], dtype=float) - np.asarray(acc[d][other][k], dtype=float)
                mean, lo, hi = boot_mean(diff_acc)
                lines.append(f"{d:.1f},accuracy_delta_BOED_minus_{other},AdaptiveBOED-{other},{k},{mean:.6f},{lo:.6f},{hi:.6f}")
        boed_auc = np.mean([np.mean(acc[d]["AdaptiveBOED"][k]) for k in K_REPORT])
        mvg_auc = np.mean([np.mean(acc[d]["AdaptiveMVG"][k]) for k in K_REPORT])
        fisher_auc = np.mean([np.mean(acc[d]["AdaptiveFisher"][k]) for k in K_REPORT])
        random_auc = np.mean([np.mean(acc[d]["Random"][k]) for k in K_REPORT])
        lines.append(f"drop_rate_summary={d:.1f}")
        lines.append(f"accuracy_auc_BOED={boed_auc:.6f}")
        lines.append(f"accuracy_auc_MVG={mvg_auc:.6f}")
        lines.append(f"accuracy_auc_Fisher={fisher_auc:.6f}")
        lines.append(f"accuracy_auc_Random={random_auc:.6f}")
    lines.append(f"elapsed_sec={time.time()-t0:.1f}")

    text = "\n".join(lines) + "\n"
    out = PKG_STATS / OUT_NAME
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
