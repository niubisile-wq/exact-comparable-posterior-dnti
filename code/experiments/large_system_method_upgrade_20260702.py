# -*- coding: utf-8 -*-
"""
Method-upgrade audit for large-system stress cases.

This script answers a stricter reviewer concern than the rerank-only audit:
can the large-system numbers be improved by a stronger, legitimate inference
procedure instead of merely explaining a weak direct NRE top-1 result?

No retraining is performed. The upgrade uses frozen checkpoints and separate
validation/test draws:
  1) validation-selected single checkpoint,
  2) probability/logit ensembles over frozen NRE seeds,
  3) validation-selected confidence-gated exact reranking over NRE top-20.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn

import dn_large_system_candidate_rerank_20260702 as base


ROOT = os.path.dirname(os.path.abspath(__file__))
PKG_STATS = r"<REPOSITORY_ROOT>\03_frozen_tables_stats"
PKG_CODE = r"<REPOSITORY_ROOT>\02_code"
OUT_NAME = "large_system_method_upgrade_20260702.txt"
DEVICE = base.DEVICE


def nll(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))))


def acc_from_p(p, y):
    return float(np.mean(np.argmax(p, axis=1) == y))


def topm_rerank_pred(p, ll, m):
    cand = np.argsort(-p, axis=1)[:, :m]
    out = np.empty(len(p), dtype=np.int64)
    for i in range(len(p)):
        ci = cand[i]
        out[i] = ci[int(np.argmax(ll[i, ci]))]
    return out


def evaluate_distribution(name, p, y, ll, m=20):
    exact_pred = np.argmax(ll, axis=1)
    cand = np.argsort(-p, axis=1)[:, :m]
    truth_in = np.array([y[i] in cand[i] for i in range(len(y))], dtype=bool)
    exact_in = np.array([exact_pred[i] in cand[i] for i in range(len(y))], dtype=bool)
    rerank_pred = topm_rerank_pred(p, ll, m)
    return {
        "method": name,
        "top1": acc_from_p(p, y),
        "nll": nll(p, y),
        "truth_in_topm": float(np.mean(truth_in)),
        "exact_winner_in_topm": float(np.mean(exact_in)),
        "rerank_topm": float(np.mean(rerank_pred == y)),
        "rerank_vs_exact": float(np.mean(rerank_pred == exact_pred)),
        "exact_full": float(np.mean(exact_pred == y)),
    }


def choose_confidence_gate(p_val, y_val, ll_val, m=20):
    exact_pred = np.argmax(ll_val, axis=1)
    full_exact_acc = float(np.mean(exact_pred == y_val))
    nre_pred = np.argmax(p_val, axis=1)
    rerank_pred = topm_rerank_pred(p_val, ll_val, m)
    conf = np.max(p_val, axis=1)

    best = None
    for tau in np.linspace(0.05, 0.95, 91):
        use_exact = conf < tau
        pred = np.where(use_exact, rerank_pred, nre_pred)
        acc = float(np.mean(pred == y_val))
        call_rate = float(np.mean(use_exact))
        candidate = {
            "tau": float(tau),
            "acc": acc,
            "call_rate": call_rate,
            "full_exact_acc": full_exact_acc,
        }
        if acc >= full_exact_acc - 0.005:
            if best is None or call_rate < best["call_rate"]:
                best = candidate
    if best is not None:
        best["rule"] = "min_call_rate_within_0.5pp_of_full_exact"
        return best

    for tau in np.linspace(0.05, 0.95, 91):
        use_exact = conf < tau
        pred = np.where(use_exact, rerank_pred, nre_pred)
        acc = float(np.mean(pred == y_val))
        call_rate = float(np.mean(use_exact))
        candidate = {
            "tau": float(tau),
            "acc": acc,
            "call_rate": call_rate,
            "full_exact_acc": full_exact_acc,
        }
        if best is None or (acc > best["acc"] + 1e-12) or (abs(acc - best["acc"]) <= 1e-12 and call_rate < best["call_rate"]):
            best = candidate
    best["rule"] = "max_validation_accuracy"
    return best


def apply_confidence_gate(p, y, ll, tau, m=20):
    exact_pred = np.argmax(ll, axis=1)
    nre_pred = np.argmax(p, axis=1)
    rerank_pred = topm_rerank_pred(p, ll, m)
    conf = np.max(p, axis=1)
    use_exact = conf < tau
    pred = np.where(use_exact, rerank_pred, nre_pred)
    return {
        "top1": float(np.mean(pred == y)),
        "call_rate": float(np.mean(use_exact)),
        "vs_exact": float(np.mean(pred == exact_pred)),
        "exact_full": float(np.mean(exact_pred == y)),
    }


def logits_to_distributions(logits_list):
    probs = [base.softmax_np(z) for z in logits_list]
    prob_ens = np.mean(np.stack(probs, axis=0), axis=0)
    logit_ens = base.softmax_np(np.mean(np.stack(logits_list, axis=0), axis=0))
    return probs, prob_ens, logit_ens


def prior_correct_from_validation(p_val, p_test):
    target = 1.0 / p_val.shape[1]
    avg = np.mean(p_val, axis=0)
    weights = target / np.clip(avg, 1e-6, None)
    out = p_test * weights[None, :]
    out /= np.sum(out, axis=1, keepdims=True)
    return out, weights


def weighted_probability_ensemble(val_probs, y_val, test_probs, n_trials=3000, seed=20260702):
    rng = np.random.RandomState(seed)
    n_models = len(val_probs)
    val_stack = np.stack(val_probs, axis=0)
    test_stack = np.stack(test_probs, axis=0)
    candidates = []
    candidates.append(np.ones(n_models, dtype=np.float64) / n_models)
    for i in range(n_models):
        w = np.zeros(n_models, dtype=np.float64)
        w[i] = 1.0
        candidates.append(w)
    for _ in range(n_trials):
        candidates.append(rng.dirichlet(np.ones(n_models)))

    best_w = None
    best_nll = None
    for w in candidates:
        p = np.tensordot(w, val_stack, axes=(0, 0))
        score = nll(p, y_val)
        if best_nll is None or score < best_nll:
            best_nll = score
            best_w = w
    p_test = np.tensordot(best_w, test_stack, axes=(0, 0))
    return p_test, best_w, float(best_nll)


def log_softmax_np(z):
    z0 = z - np.max(z, axis=1, keepdims=True)
    lse = np.log(np.sum(np.exp(z0), axis=1, keepdims=True))
    return z0 - lse


def meta_features(logits_list):
    return np.concatenate([log_softmax_np(z) for z in logits_list], axis=1).astype(np.float32)


def train_stacked_logit_calibrator(train_logits, y_train, test_logits, n_topos, seed=7):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = meta_features(train_logits)
    Xt = meta_features(test_logits)
    y = np.asarray(y_train, dtype=np.int64)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(y))
    n_fit = int(0.8 * len(y))
    fit_idx = order[:n_fit]
    val_idx = order[n_fit:]
    mu = X[fit_idx].mean(axis=0, keepdims=True)
    sd = X[fit_idx].std(axis=0, keepdims=True) + 1e-6
    Xn = (X - mu) / sd
    Xtn = (Xt - mu) / sd

    model = nn.Linear(Xn.shape[1], n_topos).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=5e-3)
    loss_fn = nn.CrossEntropyLoss()
    X_fit = torch.tensor(Xn[fit_idx], dtype=torch.float32)
    y_fit = torch.tensor(y[fit_idx], dtype=torch.long)
    X_val = torch.tensor(Xn[val_idx], dtype=torch.float32).to(DEVICE)
    y_val = torch.tensor(y[val_idx], dtype=torch.long).to(DEVICE)
    best_state = None
    best_val = -1.0
    batch = 512
    for _ in range(80):
        perm = rng.permutation(len(fit_idx))
        model.train()
        for start in range(0, len(fit_idx), batch):
            idx = perm[start:start + batch]
            xb = X_fit[idx].to(DEVICE)
            yb = y_fit[idx].to(DEVICE)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(X_val).argmax(dim=1)
            val_acc = float((pred == y_val).float().mean().item())
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xtn, dtype=torch.float32).to(DEVICE)).detach().cpu().numpy()
    return base.softmax_np(logits), best_val


def load_logits_119(xs, seeds):
    lib = base.load_119_lib()
    n_topos, _, n_bus = lib["V"].shape
    x_t = torch.tensor(xs, dtype=torch.float32).to(DEVICE)
    logits = []
    for seed in seeds:
        ckpt = torch.load(os.path.join(ROOT, f"nre_119bus_ip1_seed{seed}.pt"), map_location=DEVICE, weights_only=False)
        model = base.LoadAwareNRE119(n_topos, n_bus).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            logits.append(model(x_t).detach().cpu().numpy())
    return logits


def load_logits_300(xs, seeds):
    lib = base.load_300_lib()
    n_topos, _, n_bus = lib["V"].shape
    x_t = torch.tensor(xs, dtype=torch.float32).to(DEVICE)
    logits = []
    for seed in seeds:
        ckpt = torch.load(os.path.join(ROOT, f"nre_300bus_ipc_seed{seed}.pt"), map_location=DEVICE, weights_only=False)
        model = base.NRE300(n_topos, n_bus).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            logits.append(model(x_t).detach().cpu().numpy())
    return logits


def evaluate_system_119():
    lib = base.load_119_lib()
    seeds = [42, 123, 456, 789, 2024]

    xs_val, y_val, b_val, o_val, lf_val = base.build_119_samples(lib, n_eval=1000, seed=880119)
    ll_val, _ = base.exact_ll_119(lib["V"], 0.009, b_val, o_val, lf_val)
    xs_stack, y_stack, _, _, _ = base.build_119_samples(lib, n_eval=8000, seed=770880)
    xs_test, y_test, b_test, o_test, lf_test = base.build_119_samples(lib, n_eval=2000, seed=990119)
    ll_test, _ = base.exact_ll_119(lib["V"], 0.009, b_test, o_test, lf_test)

    val_logits = load_logits_119(xs_val, seeds)
    stack_logits = load_logits_119(xs_stack, seeds)
    test_logits = load_logits_119(xs_test, seeds)
    val_probs, p_val_probens, p_val_logitens = logits_to_distributions(val_logits)
    test_probs, p_test_probens, p_test_logitens = logits_to_distributions(test_logits)

    val_accs = [acc_from_p(p, y_val) for p in val_probs]
    best_idx = int(np.argmax(val_accs))
    best_seed = seeds[best_idx]
    gate = choose_confidence_gate(p_val_probens, y_val, ll_val, m=20)
    gated = apply_confidence_gate(p_test_probens, y_test, ll_test, gate["tau"], m=20)

    rows = []
    indiv_test_accs = [acc_from_p(p, y_test) for p in test_probs]
    rows.append({"system": "119bus_IP1", "method": "individual_seed_mean", "top1": float(np.mean(indiv_test_accs)), "extra": f"std={np.std(indiv_test_accs):.4f}"})
    rows.append({"system": "119bus_IP1", "method": f"validation_selected_seed{best_seed}", **evaluate_distribution("validation_selected", test_probs[best_idx], y_test, ll_test, 20), "extra": f"val_acc={val_accs[best_idx]:.4f}"})
    rows.append({"system": "119bus_IP1", **evaluate_distribution("probability_ensemble", p_test_probens, y_test, ll_test, 20), "extra": "all frozen seeds"})
    rows.append({"system": "119bus_IP1", **evaluate_distribution("logit_ensemble", p_test_logitens, y_test, ll_test, 20), "extra": "all frozen seeds"})
    p_weighted, w_best, val_nll = weighted_probability_ensemble(val_probs, y_val, test_probs, n_trials=3000, seed=119)
    rows.append({"system": "119bus_IP1", **evaluate_distribution("validation_weighted_probability_ensemble", p_weighted, y_test, ll_test, 20), "extra": "val_nll={:.4f};weights={}".format(val_nll, "/".join(f"{x:.2f}" for x in w_best))})
    p_prior, prior_w = prior_correct_from_validation(p_val_probens, p_test_probens)
    rows.append({"system": "119bus_IP1", **evaluate_distribution("uniform_prior_corrected_probability_ensemble", p_prior, y_test, ll_test, 20), "extra": "label_free_validation_prior_correction"})
    p_stack, stack_val = train_stacked_logit_calibrator(stack_logits, y_stack, test_logits, n_topos=lib["V"].shape[0], seed=119)
    rows.append({"system": "119bus_IP1", **evaluate_distribution("stacked_logit_calibrator", p_stack, y_test, ll_test, 20), "extra": f"offline_meta_train_n=8000;internal_val_acc={stack_val:.4f}"})
    rows.append({"system": "119bus_IP1", "method": "confidence_gated_prob_ensemble_rerank20", "top1": gated["top1"], "exact_full": gated["exact_full"], "rerank_vs_exact": gated["vs_exact"], "extra": f"tau={gate['tau']:.2f};call_rate={gated['call_rate']:.4f};val_rule={gate['rule']};val_acc={gate['acc']:.4f};val_call={gate['call_rate']:.4f}"})
    return rows


def evaluate_system_300():
    lib = base.load_300_lib()
    seeds = [42, 123, 456]

    xs_val, y_val, b_val, o_val, lf_val = base.build_300_samples(lib, n_eval=1000, seed=330088)
    ll_val, _ = base.exact_ll_300(lib["V"], 0.0015, b_val, o_val, lf_val)
    xs_stack, y_stack, _, _, _ = base.build_300_samples(lib, n_eval=8000, seed=330077)
    xs_test, y_test, b_test, o_test, lf_test = base.build_300_samples(lib, n_eval=2000, seed=330099)
    ll_test, _ = base.exact_ll_300(lib["V"], 0.0015, b_test, o_test, lf_test)

    val_logits = load_logits_300(xs_val, seeds)
    stack_logits = load_logits_300(xs_stack, seeds)
    test_logits = load_logits_300(xs_test, seeds)
    val_probs, p_val_probens, p_val_logitens = logits_to_distributions(val_logits)
    test_probs, p_test_probens, p_test_logitens = logits_to_distributions(test_logits)

    val_accs = [acc_from_p(p, y_val) for p in val_probs]
    best_idx = int(np.argmax(val_accs))
    best_seed = seeds[best_idx]
    gate = choose_confidence_gate(p_val_probens, y_val, ll_val, m=20)
    gated = apply_confidence_gate(p_test_probens, y_test, ll_test, gate["tau"], m=20)

    rows = []
    indiv_test_accs = [acc_from_p(p, y_test) for p in test_probs]
    rows.append({"system": "300bus_IPC_miss30", "method": "individual_seed_mean", "top1": float(np.mean(indiv_test_accs)), "extra": f"std={np.std(indiv_test_accs):.4f}"})
    rows.append({"system": "300bus_IPC_miss30", "method": f"validation_selected_seed{best_seed}", **evaluate_distribution("validation_selected", test_probs[best_idx], y_test, ll_test, 20), "extra": f"val_acc={val_accs[best_idx]:.4f}"})
    rows.append({"system": "300bus_IPC_miss30", **evaluate_distribution("probability_ensemble", p_test_probens, y_test, ll_test, 20), "extra": "all frozen seeds"})
    rows.append({"system": "300bus_IPC_miss30", **evaluate_distribution("logit_ensemble", p_test_logitens, y_test, ll_test, 20), "extra": "all frozen seeds"})
    p_weighted, w_best, val_nll = weighted_probability_ensemble(val_probs, y_val, test_probs, n_trials=3000, seed=300)
    rows.append({"system": "300bus_IPC_miss30", **evaluate_distribution("validation_weighted_probability_ensemble", p_weighted, y_test, ll_test, 20), "extra": "val_nll={:.4f};weights={}".format(val_nll, "/".join(f"{x:.2f}" for x in w_best))})
    p_prior, prior_w = prior_correct_from_validation(p_val_probens, p_test_probens)
    rows.append({"system": "300bus_IPC_miss30", **evaluate_distribution("uniform_prior_corrected_probability_ensemble", p_prior, y_test, ll_test, 20), "extra": "label_free_validation_prior_correction"})
    p_stack, stack_val = train_stacked_logit_calibrator(stack_logits, y_stack, test_logits, n_topos=lib["V"].shape[0], seed=300)
    rows.append({"system": "300bus_IPC_miss30", **evaluate_distribution("stacked_logit_calibrator", p_stack, y_test, ll_test, 20), "extra": f"offline_meta_train_n=8000;internal_val_acc={stack_val:.4f}"})
    rows.append({"system": "300bus_IPC_miss30", "method": "confidence_gated_prob_ensemble_rerank20", "top1": gated["top1"], "exact_full": gated["exact_full"], "rerank_vs_exact": gated["vs_exact"], "extra": f"tau={gate['tau']:.2f};call_rate={gated['call_rate']:.4f};val_rule={gate['rule']};val_acc={gate['acc']:.4f};val_call={gate['call_rate']:.4f}"})
    return rows


def format_rows(rows, elapsed):
    out = []
    out.append("Large-system method-upgrade audit")
    out.append("date=2026-07-02")
    out.append(f"device={DEVICE}")
    out.append("No retraining. Validation and test draws are separate from each other.")
    out.append("Methods: validation-selected seed, probability/logit ensemble, validation-weighted ensemble, uniform-prior correction, stacked logit calibration, and validation-selected confidence-gated exact rerank@20.")
    out.append("")
    out.append("system,method,top1,exact_full,truth_in_top20,exact_winner_in_top20,rerank20,rerank_vs_exact,nll,extra")
    for r in rows:
        out.append(
            f"{r.get('system','')},{r.get('method','')},"
            f"{r.get('top1', float('nan')):.4f},"
            f"{r.get('exact_full', float('nan')):.4f},"
            f"{r.get('truth_in_topm', float('nan')):.4f},"
            f"{r.get('exact_winner_in_topm', float('nan')):.4f},"
            f"{r.get('rerank_topm', float('nan')):.4f},"
            f"{r.get('rerank_vs_exact', float('nan')):.4f},"
            f"{r.get('nll', float('nan')):.4f},"
            f"{r.get('extra','')}"
        )
    out.append("")
    out.append("Interpretation")
    out.append("If ensemble top-1 improves direct NRE, report it as a stronger frozen inference procedure.")
    out.append("If confidence-gated reranking recovers exact-level accuracy at a partial call rate, report it as a deployable high-fidelity fallback rather than a post-hoc excuse.")
    out.append(f"elapsed_sec={elapsed:.1f}")
    return "\n".join(out) + "\n"


def main():
    t0 = time.time()
    print(f"Device: {DEVICE}")
    print("Evaluating 119-bus method upgrades...")
    rows = evaluate_system_119()
    print("Evaluating 300-bus method upgrades...")
    rows.extend(evaluate_system_300())
    text = format_rows(rows, time.time() - t0)
    local_out = os.path.join(ROOT, OUT_NAME)
    with open(local_out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved: {local_out}")
    if os.path.isdir(PKG_STATS):
        with open(os.path.join(PKG_STATS, OUT_NAME), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved: {os.path.join(PKG_STATS, OUT_NAME)}")
    if os.path.isdir(PKG_CODE):
        src = os.path.abspath(__file__)
        dst = os.path.join(PKG_CODE, OUT_NAME.replace(".txt", ".py"))
        with open(src, "r", encoding="utf-8") as f:
            code = f.read()
        with open(dst, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"Saved: {dst}")


if __name__ == "__main__":
    main()
