# -*- coding: utf-8 -*-
"""
Step 1 minimal IP-A result for 119-bus.

Purpose:
  Close the 119-bus chain with a traceable sensor-placement result at K=4 and K=7.

Scope:
  This is a minimum validation result, not the final full BOED curve. It uses the
  existing 119-bus voltage library and exact enumerable posterior evaluation.
"""
import time
import numpy as np

SAVE_DIR = r"<LOCAL_WORKSPACE>"
VLIB_PATH = f"{SAVE_DIR}\\v_library_119bus.npz"
OUT_PATH = f"{SAVE_DIR}\\boed_119bus_minimal_result.txt"

SIGMA = 0.009
N_LF_C = 11
K_MAX = 7
K_REPORT = [4, 7]
N_TEST = 120
N_MC = 40


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


def evaluate_fixed_set(v_fine, v_flat_c, lf_idx_c, sensor_order, test_cases, n_topos, n_lf_c):
    acc = {k: 0 for k in K_REPORT}
    ent = {k: 0.0 for k in K_REPORT}
    for true_ti, lf_idx, noisy_full_v in test_cases:
        for k in K_REPORT:
            nodes = sensor_order[:k]
            obs = noisy_full_v[nodes]
            joint = posterior_joint(v_flat_c, nodes, obs)
            p_topo = topo_posterior_from_joint(joint, n_topos, n_lf_c)
            acc[k] += int(np.argmax(p_topo) == true_ti)
            ent[k] += entropy(p_topo)
    return {k: acc[k] / len(test_cases) for k in K_REPORT}, {k: ent[k] / len(test_cases) for k in K_REPORT}


def select_mvg(v_library, candidates):
    # Maximum voltage variance across topology and load-factor scenarios.
    var = np.var(v_library.reshape(-1, v_library.shape[2]), axis=0)
    order = sorted(candidates, key=lambda n: -var[n])
    return order[:K_MAX]


def select_fisher(v_library, candidates):
    # D-optimal proxy using centered voltage differences at lf closest to 1.0.
    lf_mid = v_library.shape[1] // 2
    v_fixed = v_library[:, lf_mid, :]
    j_full = (v_fixed - v_fixed.mean(axis=0, keepdims=True)) / SIGMA
    selected, remaining = [], list(candidates)
    for _ in range(K_MAX):
        best_node, best_score = None, -np.inf
        for node in remaining:
            nodes = selected + [node]
            fim = j_full[:, nodes].T @ j_full[:, nodes]
            sign, logdet = np.linalg.slogdet(fim + 1e-6 * np.eye(len(nodes)))
            score = logdet if sign > 0 else -np.inf
            if score > best_score:
                best_score, best_node = score, node
        selected.append(best_node)
        remaining.remove(best_node)
    return selected


def select_boed_prior(v_flat_c, candidates, n_topos, n_lf_c, rng):
    # Sequential prior BOED. This avoids test-label leakage and keeps the 119-bus
    # minimum result computationally bounded while still using exact posterior updates.
    selected, remaining = [], list(candidates)
    obs_vals = []
    for step in range(K_MAX):
        current_joint = posterior_joint(v_flat_c, selected, obs_vals)
        current_h = entropy(topo_posterior_from_joint(current_joint, n_topos, n_lf_c))
        sample_idx = rng.choice(v_flat_c.shape[0], size=N_MC, p=current_joint)
        best_node, best_eig = None, -np.inf
        for node in remaining:
            h_after = 0.0
            # Synthetic predictive samples for this node under current posterior.
            y_samples = v_flat_c[sample_idx, node] + rng.normal(0.0, SIGMA, size=N_MC)
            for y in y_samples:
                joint_new = posterior_joint(v_flat_c, selected + [node], obs_vals + [y])
                h_after += entropy(topo_posterior_from_joint(joint_new, n_topos, n_lf_c))
            eig = current_h - h_after / N_MC
            if eig > best_eig:
                best_eig, best_node = eig, node
        selected.append(best_node)
        obs_vals.append(float(np.mean(v_flat_c[:, best_node])))
        remaining.remove(best_node)
    return selected


def select_boed_adaptive(v_flat_c, candidates, n_topos, n_lf_c, full_v, rng):
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
                joint_new = posterior_joint(v_flat_c, selected + [node], obs_vals + [y])
                h_after += entropy(topo_posterior_from_joint(joint_new, n_topos, n_lf_c))
            eig = current_h - h_after / N_MC
            if eig > best_eig:
                best_eig, best_node = eig, node
        selected.append(best_node)
        obs_vals.append(float(full_v[best_node]))
        remaining.remove(best_node)
    return selected


def select_adaptive_mvg(v_flat_c, candidates, full_v):
    selected, remaining = [], list(candidates)
    obs_vals = []
    for _ in range(K_MAX):
        joint = posterior_joint(v_flat_c, selected, obs_vals)
        mean = joint @ v_flat_c[:, remaining]
        second = joint @ (v_flat_c[:, remaining] ** 2)
        var = second - mean * mean
        node = remaining[int(np.argmax(var))]
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


def main():
    t0 = time.time()
    dat = np.load(VLIB_PATH)
    v_library = dat["V_library"]
    lf_grid = dat["lf_grid"]
    n_topos, n_lf, n_bus = v_library.shape
    lf_idx_c = np.round(np.linspace(0, n_lf - 1, N_LF_C)).astype(int)
    v_c = v_library[:, lf_idx_c, :]
    v_flat_c = v_c.reshape(n_topos * N_LF_C, n_bus)
    candidates = list(range(1, n_bus))

    rng = np.random.RandomState(20260702)
    test_cases = []
    for _ in range(N_TEST):
        ti = rng.randint(0, n_topos)
        lf_idx = rng.randint(0, n_lf)
        noisy_full_v = v_library[ti, lf_idx, :] + rng.normal(0.0, SIGMA, size=n_bus)
        test_cases.append((ti, lf_idx, noisy_full_v))

    orders = {}
    orders["Random"] = list(candidates)
    np.random.RandomState(123).shuffle(orders["Random"])
    orders["MVG"] = select_mvg(v_library, candidates)
    orders["Fisher"] = select_fisher(v_library, candidates)
    orders["BOED"] = select_boed_prior(v_flat_c, candidates, n_topos, N_LF_C, np.random.RandomState(456))

    lines = []
    lines.append("=" * 80)
    lines.append("119-bus minimal IP-A sensor placement result")
    lines.append(f"N_TOPOS={n_topos}, N_BUS={n_bus}, N_TEST={N_TEST}, N_MC={N_MC}, N_LF_C={N_LF_C}")
    lines.append("Scope: minimum Step-1 closure result at K=4 and K=7; full curve remains future work.")
    lines.append("=" * 80)
    lines.append("")

    metrics = {}
    for name, order in orders.items():
        acc, ent = evaluate_fixed_set(v_library, v_flat_c, lf_idx_c, order, test_cases, n_topos, N_LF_C)
        metrics[name] = (acc, ent)
        lines.append(f"{name} selected nodes K<=7: {order[:K_MAX]}")
        for k in K_REPORT:
            lines.append(f"  K={k}: acc={acc[k]:.3f}, H={ent[k]:.3f}")
        lines.append("")

    # Adaptive policies: the next sensor is selected after observing previous
    # sensor values. These are fairer comparisons for AdaptiveBOED than static
    # fixed-set baselines.
    adaptive_metrics = {
        "AdaptiveMVG": ({k: 0 for k in K_REPORT}, {k: 0.0 for k in K_REPORT}),
        "AdaptiveFisher": ({k: 0 for k in K_REPORT}, {k: 0.0 for k in K_REPORT}),
        "AdaptiveBOED": ({k: 0 for k in K_REPORT}, {k: 0.0 for k in K_REPORT}),
    }
    t_ad = time.time()
    for ci, (true_ti, lf_idx, noisy_full_v) in enumerate(test_cases):
        if ci % 20 == 0:
            print(f"Adaptive policies case {ci}/{len(test_cases)}  elapsed={time.time() - t_ad:.1f}s", flush=True)
        adaptive_orders = {
            "AdaptiveMVG": select_adaptive_mvg(v_flat_c, candidates, noisy_full_v),
            "AdaptiveFisher": select_adaptive_fisher(v_flat_c, candidates, noisy_full_v),
            "AdaptiveBOED": select_boed_adaptive(v_flat_c, candidates, n_topos, N_LF_C, noisy_full_v, np.random.RandomState(9000 + ci)),
        }
        for name, order in adaptive_orders.items():
            acc_dict, ent_dict = adaptive_metrics[name]
            for k in K_REPORT:
                nodes = order[:k]
                obs = noisy_full_v[nodes]
                joint = posterior_joint(v_flat_c, nodes, obs)
                p_topo = topo_posterior_from_joint(joint, n_topos, N_LF_C)
                acc_dict[k] += int(np.argmax(p_topo) == true_ti)
                ent_dict[k] += entropy(p_topo)
    for name, (acc_dict, ent_dict) in adaptive_metrics.items():
        acc = {k: acc_dict[k] / len(test_cases) for k in K_REPORT}
        ent = {k: ent_dict[k] / len(test_cases) for k in K_REPORT}
        metrics[name] = (acc, ent)
        lines.append(f"{name} selected nodes vary by test case.")
        for k in K_REPORT:
            lines.append(f"  K={k}: acc={acc[k]:.3f}, H={ent[k]:.3f}")
        lines.append("")

    lines.append("Summary table")
    lines.append("Method       K=4 acc   K=7 acc   K=4 H     K=7 H")
    lines.append("-" * 58)
    for name in ["Random", "MVG", "Fisher", "BOED", "AdaptiveMVG", "AdaptiveFisher", "AdaptiveBOED"]:
        acc, ent = metrics[name]
        lines.append(f"{name:<10}  {acc[4]:>7.3f}   {acc[7]:>7.3f}   {ent[4]:>7.3f}   {ent[7]:>7.3f}")

    boed_acc, boed_ent = metrics["BOED"]
    ad_mvg_acc, _ = metrics["AdaptiveMVG"]
    ad_fisher_acc, _ = metrics["AdaptiveFisher"]
    ad_boed_acc, ad_boed_ent = metrics["AdaptiveBOED"]
    random_acc, _ = metrics["Random"]
    fisher_acc, _ = metrics["Fisher"]
    mvg_acc, _ = metrics["MVG"]
    lines.append("")
    lines.append("Gates")
    lines.append(f"  BOED > Random at K=4: {'PASS' if boed_acc[4] > random_acc[4] else 'FAIL'}")
    lines.append(f"  BOED > Random at K=7: {'PASS' if boed_acc[7] > random_acc[7] else 'FAIL'}")
    lines.append(f"  BOED vs Fisher at K=7: delta={boed_acc[7] - fisher_acc[7]:+.3f}")
    lines.append(f"  BOED vs MVG at K=7: delta={boed_acc[7] - mvg_acc[7]:+.3f}")
    lines.append(f"  AdaptiveBOED > Random at K=4: {'PASS' if ad_boed_acc[4] > random_acc[4] else 'FAIL'}")
    lines.append(f"  AdaptiveBOED > Random at K=7: {'PASS' if ad_boed_acc[7] > random_acc[7] else 'FAIL'}")
    lines.append(f"  AdaptiveBOED vs AdaptiveMVG at K=7: delta={ad_boed_acc[7] - ad_mvg_acc[7]:+.3f}")
    lines.append(f"  AdaptiveBOED vs AdaptiveFisher at K=7: delta={ad_boed_acc[7] - ad_fisher_acc[7]:+.3f}")
    lines.append(f"  AdaptiveBOED vs Fisher at K=7: delta={ad_boed_acc[7] - fisher_acc[7]:+.3f}")
    lines.append(f"  AdaptiveBOED vs MVG at K=7: delta={ad_boed_acc[7] - mvg_acc[7]:+.3f}")
    lines.append("")
    lines.append("Boundary")
    lines.append("  This result only closes the 119-bus minimum IP-A evidence for Step 1.")
    lines.append("  Do not use it to claim full BOED scalability; Step 5 still requires 300-bus.")
    lines.append(f"Runtime: {time.time() - t0:.1f}s")
    lines.append("=" * 80)

    out = "\n".join(lines)
    print(out)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
