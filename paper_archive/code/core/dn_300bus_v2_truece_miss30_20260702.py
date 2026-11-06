# -*- coding: utf-8 -*-
"""
300-bus IP-C 30% missing v2 retraining: true-label fixed-missing fine-tuning.

This run tests whether the low direct top-1 result is mainly caused by the old
robust model being trained across mixed missing rates and posterior-distillation
targets. It keeps the benchmark fixed and fine-tunes only on the exact 30%
missing deployment distribution, optimizing true topology CE directly.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn

import dn_large_system_candidate_rerank_20260702 as base
import dn_300bus_v2_miss30_mapdistill_20260702 as mapv2


ROOT = os.path.dirname(os.path.abspath(__file__))
PKG_STATS = r"<REPOSITORY_ROOT>\03_frozen_tables_stats"
PKG_CODE = r"<REPOSITORY_ROOT>\02_code"
DEVICE = base.DEVICE

BATCH = 1024
N_TRAIN = 160000
N_VAL = 5000
N_FINAL = 8000
EPOCHS = 12
LR = 3e-5
SEEDS = [42, 123, 456]
OUT_NAME = "300bus_v2_truece_miss30_20260702.txt"


def load_old_model(seed, lib):
    n_topos, _, n_bus = lib["V"].shape
    model = base.NRE300(n_topos, n_bus).to(DEVICE)
    ckpt = torch.load(os.path.join(ROOT, f"nre_300bus_ipc_seed{seed}.pt"), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model


def train_one(seed, lib, xs_train, y_true_train, xs_val, y_true_val, y_map_val):
    model = load_old_model(seed, lib)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=EPOCHS * int(np.ceil(N_TRAIN / BATCH)),
        eta_min=6e-6,
    )
    x_train_t = torch.tensor(xs_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_true_train, dtype=torch.long)
    rng = np.random.RandomState(400000 + seed)
    best_score = -1e9
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    best_val = None
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(N_TRAIN)
        model.train()
        for start in range(0, N_TRAIN, BATCH):
            idx = order[start:start + BATCH]
            xb = x_train_t[idx].to(DEVICE)
            yb = y_train_t[idx].to(DEVICE)
            logits = model(xb)
            loss = ce(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            scheduler.step()
        val = mapv2.eval_logits(mapv2.predict_logits(model, xs_val), y_true_val, y_map_val)
        score = val["true_acc"]
        print(
            f"seed={seed} trueCE epoch={epoch}/{EPOCHS} val_true={val['true_acc']:.4f} "
            f"val_map_agree={val['map_agree']:.4f} exact={val['exact_map_true_acc']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_val = val
    model.load_state_dict(best_state)
    ckpt_name = f"nre_300bus_ipc_miss30_v2_truece_seed{seed}_20260702.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "seed": seed,
            "input": "300-bus IPC normalized voltage + mask + load feature",
            "loss": "CE(true_topology) on fixed 30% missing distribution",
            "best_epoch": best_epoch,
            "validation": best_val,
        },
        os.path.join(ROOT, ckpt_name),
    )
    return model, ckpt_name, best_epoch, best_val


def main():
    t0 = time.time()
    lib = base.load_300_lib()
    print(f"Device: {DEVICE}")
    print(f"300-bus v2 trueCE miss30: train={N_TRAIN} val={N_VAL} final={N_FINAL}")
    xs_train, y_true_train, y_map_train = mapv2.make_dataset(lib, N_TRAIN, seed=300505)
    xs_val, y_true_val, y_map_val, ll_val = mapv2.make_dataset(lib, N_VAL, seed=300606, return_ll=True)
    xs_final, y_true_final, y_map_final, ll_final = mapv2.make_dataset(lib, N_FINAL, seed=300707, return_ll=True)

    old_val_logits = []
    old_final_logits = []
    new_val_logits = []
    new_final_logits = []
    rows = []
    for seed in SEEDS:
        old_model = load_old_model(seed, lib)
        old_val_logits.append(mapv2.predict_logits(old_model, xs_val))
        old_final_logits.append(mapv2.predict_logits(old_model, xs_final))
        model, ckpt_name, best_epoch, best_val = train_one(seed, lib, xs_train, y_true_train, xs_val, y_true_val, y_map_val)
        new_val_logits.append(mapv2.predict_logits(model, xs_val))
        new_final_logits.append(mapv2.predict_logits(model, xs_final))
        final = mapv2.eval_logits(new_final_logits[-1], y_true_final, y_map_final)
        rows.append({"seed": seed, "checkpoint": ckpt_name, "best_epoch": best_epoch, "best_val": best_val, "final": final})

    p_old_val = mapv2.avg_prob(old_val_logits)
    p_old_final = mapv2.avg_prob(old_final_logits)
    p_new_val = mapv2.avg_prob(new_val_logits)
    p_new_final = mapv2.avg_prob(new_final_logits)
    old_final = mapv2.eval_prob(p_old_final, y_true_final, y_map_final)
    new_final = mapv2.eval_prob(p_new_final, y_true_final, y_map_final)
    mix_rule = mapv2.choose_mix(p_old_val, p_new_val, y_true_val, y_map_val)
    p_mix_final = mix_rule["lambda_old"] * p_old_final + (1.0 - mix_rule["lambda_old"]) * p_new_final
    mix_final = mapv2.eval_prob(p_mix_final, y_true_final, y_map_final)
    old_rerank = mapv2.rerank_metrics(p_old_final, y_true_final, y_map_final, ll_final, m=20)
    new_rerank = mapv2.rerank_metrics(p_new_final, y_true_final, y_map_final, ll_final, m=20)
    mix_rerank = mapv2.rerank_metrics(p_mix_final, y_true_final, y_map_final, ll_final, m=20)

    lines = []
    lines.append("300-bus IP-C 30% missing v2 true-label fixed-missing fine-tuning")
    lines.append("date=2026-07-02")
    lines.append("status=isolated_v2_does_not_overwrite_original_checkpoints")
    lines.append(f"device={DEVICE}")
    lines.append(f"train_samples={N_TRAIN}")
    lines.append(f"validation_samples={N_VAL}")
    lines.append(f"final_test_samples={N_FINAL}")
    lines.append("loss=CE(true_topology) on fixed 30% missing distribution")
    lines.append(f"final_same_draw_exact_MAP_top1={old_final['exact_map_true_acc']:.4f}")
    lines.append("seed,v2_final_true_acc,v2_final_map_agree,best_epoch,best_val_true,best_val_map_agree,checkpoint")
    for r in rows:
        lines.append(
            f"{r['seed']},{r['final']['true_acc']:.4f},{r['final']['map_agree']:.4f},"
            f"{r['best_epoch']},{r['best_val']['true_acc']:.4f},{r['best_val']['map_agree']:.4f},"
            f"{r['checkpoint']}"
        )
    lines.append(f"old_probability_ensemble_final_true_acc={old_final['true_acc']:.4f}")
    lines.append(f"old_probability_ensemble_final_map_agree={old_final['map_agree']:.4f}")
    lines.append(f"old_probability_ensemble_truth_in_top20={old_rerank['truth_in_top20']:.4f}")
    lines.append(f"old_probability_ensemble_map_in_top20={old_rerank['map_in_top20']:.4f}")
    lines.append(f"old_probability_ensemble_rerank20_true_acc={old_rerank['rerank20_true_acc']:.4f}")
    lines.append(f"v2_probability_ensemble_final_true_acc={new_final['true_acc']:.4f}")
    lines.append(f"v2_probability_ensemble_final_map_agree={new_final['map_agree']:.4f}")
    lines.append(f"v2_probability_ensemble_truth_in_top20={new_rerank['truth_in_top20']:.4f}")
    lines.append(f"v2_probability_ensemble_map_in_top20={new_rerank['map_in_top20']:.4f}")
    lines.append(f"v2_probability_ensemble_rerank20_true_acc={new_rerank['rerank20_true_acc']:.4f}")
    lines.append(f"validation_selected_lambda_old={mix_rule['lambda_old']:.2f}")
    lines.append(f"validation_selected_mix_val_true_acc={mix_rule['true_acc']:.4f}")
    lines.append(f"validation_selected_mix_val_map_agree={mix_rule['map_agree']:.4f}")
    lines.append(f"validation_selected_mix_final_true_acc={mix_final['true_acc']:.4f}")
    lines.append(f"validation_selected_mix_final_map_agree={mix_final['map_agree']:.4f}")
    lines.append(f"validation_selected_mix_truth_in_top20={mix_rerank['truth_in_top20']:.4f}")
    lines.append(f"validation_selected_mix_map_in_top20={mix_rerank['map_in_top20']:.4f}")
    lines.append(f"validation_selected_mix_rerank20_true_acc={mix_rerank['rerank20_true_acc']:.4f}")
    lines.append(f"delta_mix_vs_old_final={mix_final['true_acc'] - old_final['true_acc']:.4f}")
    lines.append(f"elapsed_sec={time.time() - t0:.1f}")
    text = "\n".join(lines) + "\n"
    local_out = os.path.join(ROOT, OUT_NAME)
    with open(local_out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"Saved: {local_out}")
    if os.path.isdir(PKG_STATS):
        pkg_out = os.path.join(PKG_STATS, OUT_NAME)
        with open(pkg_out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved: {pkg_out}")
    if os.path.isdir(PKG_CODE):
        code_out = os.path.join(PKG_CODE, os.path.basename(__file__))
        with open(__file__, "r", encoding="utf-8") as f:
            code = f.read()
        with open(code_out, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"Saved: {code_out}")


if __name__ == "__main__":
    main()
