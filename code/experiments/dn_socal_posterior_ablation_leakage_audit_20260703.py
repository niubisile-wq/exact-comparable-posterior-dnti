# -*- coding: utf-8 -*-
"""
SoCal synchronized posterior ablation and leakage audit.

This script strengthens the real-data line by testing:
1. feature-family value: magnitude-only vs phasor-only vs fusion
2. temporal feature-bank value: raw features vs causal temporal bank
3. stricter evaluation: blocked CV vs within-state forward chronological split
"""

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler

warnings.simplefilter("ignore", PerformanceWarning)

TEMP_ROOT = Path.home() / "Desktop" / "配电网论文实验总汇_20260703" / "配电网实验_临时"
MEAS = TEMP_ROOT / "real_data_preview_samples"
STATUS = TEMP_ROOT / "digital-twin-dataset" / "sample_dataset" / "topology" / "parameter_timeseries"
PKG_ROOT = Path.home() / "Desktop" / "配电网论文实验总汇_20260703" / "配电网论文一区投稿成果包_20260702"
PKG_STATS = PKG_ROOT / "03_frozen_tables_stats"

OUT_NAME = "socal_posterior_ablation_leakage_audit_20260703.txt"
WINDOW_START = pd.Timestamp("2024-11-14T07:00:00")
WINDOW_END = pd.Timestamp("2024-11-14T07:30:42")
DEVICES = ["cb_121", "cb_123", "cb_128", "swmp_17-2"]
FILES = {
    "cb_121": STATUS / "cb_121-tbus_status.csv",
    "cb_123": STATUS / "cb_123-tbus_status.csv",
    "cb_128": STATUS / "cb_128-tbus_status.csv",
    "swmp_17-2": STATUS / "swmp_17-2-tbus_status.csv",
}

MAG_FEATURES = ["mains_v"]
PHASOR_FEATURES = ["frequency", "rms", "magnitude_harmonic_0", "magnitude_harmonic_1", "magnitude_harmonic_2"]
FUSION_FEATURES = MAG_FEATURES + PHASOR_FEATURES
FOLDS = 5
SEEDS = [42]
FORWARD_TRAIN_RATIO = 0.75


def load_measurements():
    mains = pd.read_csv(MEAS / "egauge_9-Mains_Power.csv")
    mains["t"] = pd.to_datetime(mains["t"])
    mains = mains.rename(columns={"v": "mains_v"})[["t", "mains_v"]]

    s3 = pd.read_csv(MEAS / "egauge_9-S3.csv")
    s3["t"] = pd.to_datetime(s3["t"])
    s3 = s3[["t", "frequency", "rms", "magnitude_harmonic_0", "magnitude_harmonic_1", "magnitude_harmonic_2"]]

    merged = pd.merge_asof(
        mains.sort_values("t"),
        s3.sort_values("t"),
        on="t",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=10),
    )
    merged = merged[(merged["t"] >= WINDOW_START) & (merged["t"] <= WINDOW_END)].copy()
    return merged.dropna(subset=FUSION_FEATURES).reset_index(drop=True)


def load_status_tables():
    tables = {}
    for dev, path in FILES.items():
        df = pd.read_csv(path)
        df["t"] = pd.to_datetime(df["t"]) + pd.Timedelta(hours=8)
        tables[dev] = df.sort_values("t").reset_index(drop=True)
    return tables


def state_at(tables, t):
    vals = []
    for dev in DEVICES:
        df = tables[dev]
        mask = df["t"] <= t
        vals.append(str(df.loc[mask, "str"].iloc[-1]) if mask.any() else str(df["str"].iloc[0]))
    return tuple(vals)


def attach_states(meas, tables):
    states = [state_at(tables, t) for t in meas["t"]]
    uniq, sid = [], []
    for st in states:
        if st not in uniq:
            uniq.append(st)
        sid.append(uniq.index(st))
    out = meas.copy()
    out["state_id"] = sid
    return out, uniq


def add_feature_bank(df, base_features, use_temporal_bank):
    out = df[["t", "state_id"] + base_features].copy()
    if not use_temporal_bank:
        return out, list(base_features)

    eps = 1e-9
    for col in base_features:
        s = out[col].astype(float)
        for lag in [1, 2, 3, 5, 10, 20, 40, 80, 160]:
            lagged = s.shift(lag)
            out[f"{col}_lag{lag}"] = lagged.fillna(s)
            out[f"{col}_d{lag}"] = (s - lagged).fillna(0.0)
            out[f"{col}_reld{lag}"] = ((s - lagged) / (np.abs(lagged) + eps)).fillna(0.0)
        for w in [8, 15, 30, 60, 120, 240]:
            roll = s.rolling(window=w, min_periods=2)
            mean = roll.mean().fillna(s)
            std = roll.std().fillna(0.0)
            out[f"{col}_mean{w}"] = mean
            out[f"{col}_std{w}"] = std
            out[f"{col}_min{w}"] = roll.min().fillna(s)
            out[f"{col}_max{w}"] = roll.max().fillna(s)
            out[f"{col}_devmean{w}"] = s - mean
            out[f"{col}_z{w}"] = ((s - mean) / (std + eps)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
            out[f"{col}_range{w}"] = (out[f"{col}_max{w}"] - out[f"{col}_min{w}"]).fillna(0.0)
        for span in [10, 30, 90]:
            ewm = s.ewm(span=span, adjust=False, min_periods=2).mean().fillna(s)
            out[f"{col}_ewm{span}"] = ewm
            out[f"{col}_dewewm{span}"] = s - ewm

    if set(MAG_FEATURES).issubset(base_features) and set(PHASOR_FEATURES).issubset(base_features):
        out["v_over_rms"] = out["mains_v"] / (np.abs(out["rms"]) + eps)
        out["harmonic12_sum"] = out["magnitude_harmonic_1"] + out["magnitude_harmonic_2"]
        out["harmonic12_ratio"] = out["harmonic12_sum"] / (np.abs(out["magnitude_harmonic_0"]) + eps)
        out["freq_rms_product"] = out["frequency"] * out["rms"]

    feature_cols = [c for c in out.columns if c not in {"t", "state_id"}]
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, feature_cols


def select_drift_robust_features(feature_cols):
    keep = []
    suffix_markers = ("_d", "_reld", "_z", "_range", "_dewewm", "_std")
    named_keep = {"v_over_rms", "harmonic12_sum", "harmonic12_ratio", "freq_rms_product"}
    for col in feature_cols:
        if col in named_keep:
            keep.append(col)
            continue
        if any(marker in col for marker in suffix_markers):
            keep.append(col)
    return keep


def blocked_fold_indices(df, class_states, fold):
    train_idx, test_idx = [], []
    for sid in class_states:
        idx = df.index[df["state_id"] == sid].to_numpy()
        chunks = np.array_split(idx, FOLDS)
        for j, ch in enumerate(chunks):
            if len(ch) == 0:
                continue
            if j == fold:
                test_idx.extend(ch.tolist())
            else:
                train_idx.extend(ch.tolist())
    return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)


def forward_split_indices(df, class_states):
    train_idx, test_idx = [], []
    for sid in class_states:
        idx = df.index[df["state_id"] == sid].to_numpy()
        cut = max(3, int(len(idx) * FORWARD_TRAIN_RATIO))
        cut = min(cut, len(idx) - 1)
        train_idx.extend(idx[:cut].tolist())
        test_idx.extend(idx[cut:].tolist())
    return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)


def normalize_probs(p, n_classes):
    p = np.asarray(p, dtype=np.float64)
    if p.shape[1] == n_classes:
        out = p
    else:
        out = np.zeros((p.shape[0], n_classes), dtype=np.float64)
        out[:, : p.shape[1]] = p
    out = np.clip(out, 1e-9, 1.0)
    out /= out.sum(axis=1, keepdims=True)
    return out


def metrics(probs, y, n_classes):
    pred = np.argmax(probs, axis=1)
    acc = float(np.mean(pred == y))
    macro_f1 = float(f1_score(y, pred, labels=list(range(n_classes)), average="macro", zero_division=0))
    conf = np.max(probs, axis=1)
    corr = (pred == y).astype(float)
    ece = 0.0
    for lo in np.linspace(0, 0.9, 10):
        hi = lo + 0.1
        mask = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if np.any(mask):
            ece += np.mean(mask) * abs(float(np.mean(conf[mask])) - float(np.mean(corr[mask])))
    onehot = np.eye(n_classes)[y]
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    return acc, macro_f1, ece, brier


def boot_ci_acc(probs, y, boot=600):
    pred = np.argmax(probs, axis=1)
    correct = (pred == y).astype(float)
    rng = np.random.RandomState(703)
    vals = np.empty(boot, dtype=np.float64)
    for i in range(boot):
        idx = rng.randint(0, len(correct), len(correct))
        vals[i] = np.mean(correct[idx])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def make_models(seed):
    return [
        ExtraTreesClassifier(
            n_estimators=220,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    ]


def train_eval(df, feature_cols, class_states, train_idx, test_idx, seed):
    x = df[feature_cols].to_numpy(dtype=np.float32)
    y_raw = df["state_id"].to_numpy(dtype=np.int64)
    label_map = {sid: i for i, sid in enumerate(class_states)}
    y = np.array([label_map[int(v)] for v in y_raw], dtype=np.int64)

    scaler = RobustScaler(quantile_range=(5, 95))
    x_train = scaler.fit_transform(x[train_idx])
    x_test = scaler.transform(x[test_idx])

    probs = []
    for model in make_models(seed):
        model.fit(x_train, y[train_idx])
        probs.append(normalize_probs(model.predict_proba(x_test), len(class_states)))
    return np.mean(np.stack(probs, axis=0), axis=0), y[test_idx]


def evaluate_protocol(df, feature_cols, class_states, split_mode):
    if split_mode == "blocked_cv":
        probs_by_idx, y_by_idx = {}, {}
        for fold in range(FOLDS):
            train_idx, test_idx = blocked_fold_indices(df, class_states, fold)
            seed_probs = []
            y_fold = None
            for seed in SEEDS:
                p, y = train_eval(df, feature_cols, class_states, train_idx, test_idx, seed + fold * 1000)
                seed_probs.append(p)
                y_fold = y
            pmean = np.mean(np.stack(seed_probs, axis=0), axis=0)
            for pos, idx in enumerate(test_idx):
                probs_by_idx[int(idx)] = pmean[pos]
                y_by_idx[int(idx)] = int(y_fold[pos])
        order = sorted(probs_by_idx)
        probs = np.stack([probs_by_idx[i] for i in order], axis=0)
        y = np.array([y_by_idx[i] for i in order], dtype=np.int64)
    else:
        train_idx, test_idx = forward_split_indices(df, class_states)
        seed_probs = []
        y = None
        for seed in SEEDS:
            p, y = train_eval(df, feature_cols, class_states, train_idx, test_idx, seed + 7000)
            seed_probs.append(p)
        probs = np.mean(np.stack(seed_probs, axis=0), axis=0)

    acc, macro_f1, ece, brier = metrics(probs, y, len(class_states))
    ci_lo, ci_hi = boot_ci_acc(probs, y)
    return {
        "test_n": len(y),
        "acc": acc,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "macro_f1": macro_f1,
        "ece": ece,
        "brier": brier,
    }


def main():
    t0 = time.time()
    meas = load_measurements()
    tables = load_status_tables()
    base_df, uniq = attach_states(meas, tables)
    counts = base_df["state_id"].value_counts().sort_index().to_dict()
    class_states = [int(k) for k, v in counts.items() if v >= FOLDS * 3]
    base_df = base_df[base_df["state_id"].isin(class_states)].copy().reset_index(drop=True)

    protocols = [
        ("fusion_raw_blocked", FUSION_FEATURES, False, "blocked_cv"),
        ("fusion_temporal_blocked", FUSION_FEATURES, True, "blocked_cv"),
        ("magnitude_temporal_blocked", MAG_FEATURES, True, "blocked_cv"),
        ("phasor_temporal_blocked", PHASOR_FEATURES, True, "blocked_cv"),
        ("fusion_temporal_forward", FUSION_FEATURES, True, "forward_split"),
        ("fusion_driftrobust_forward", FUSION_FEATURES, True, "forward_split"),
    ]

    results = []
    for name, base_features, use_temporal_bank, split_mode in protocols:
        feat_df, feature_cols = add_feature_bank(base_df, base_features, use_temporal_bank)
        if name == "fusion_driftrobust_forward":
            feature_cols = select_drift_robust_features(feature_cols)
        metrics_dict = evaluate_protocol(feat_df, feature_cols, class_states, split_mode)
        results.append(
            {
                "protocol": name,
                "base_family": "+".join(base_features),
                "temporal_bank": int(use_temporal_bank),
                "split_mode": split_mode,
                "feature_dim": len(feature_cols),
                **metrics_dict,
            }
        )

    by_name = {row["protocol"]: row for row in results}
    blocked_gain = by_name["fusion_temporal_blocked"]["acc"] - by_name["fusion_raw_blocked"]["acc"]
    fusion_vs_mag = by_name["fusion_temporal_blocked"]["acc"] - by_name["magnitude_temporal_blocked"]["acc"]
    fusion_vs_phasor = by_name["fusion_temporal_blocked"]["acc"] - by_name["phasor_temporal_blocked"]["acc"]
    forward_gap = by_name["fusion_temporal_blocked"]["acc"] - by_name["fusion_temporal_forward"]["acc"]
    drift_forward_gain = by_name["fusion_driftrobust_forward"]["acc"] - by_name["fusion_temporal_forward"]["acc"]

    lines = []
    lines.append("SoCal synchronized posterior ablation and leakage audit")
    lines.append("date=2026-07-03")
    lines.append("role=real synchronized posterior credibility upgrade via feature-family ablation and stricter time-split audit")
    lines.append("not_claimed=utility deployment; independent field label audit; timestamp feature usage")
    lines.append(f"window_utc={WINDOW_START.isoformat()} to {WINDOW_END.isoformat()}")
    lines.append(f"measurement_rows={len(base_df)}")
    lines.append(f"trainable_states={' '.join(map(str, class_states))}")
    lines.append(f"excluded_states_due_to_low_support={' '.join(str(i) for i in range(len(uniq)) if i not in class_states) or 'none'}")
    lines.append(f"forward_train_ratio={FORWARD_TRAIN_RATIO:.2f}")
    lines.append("protocol_metrics")
    lines.append("protocol,base_family,temporal_bank,split_mode,feature_dim,test_n,acc,acc_ci95_lo,acc_ci95_hi,macro_f1,ece,brier")
    for row in results:
        lines.append(
            f"{row['protocol']},{row['base_family']},{row['temporal_bank']},{row['split_mode']},"
            f"{row['feature_dim']},{row['test_n']},{row['acc']:.4f},{row['ci_lo']:.4f},{row['ci_hi']:.4f},"
            f"{row['macro_f1']:.4f},{row['ece']:.4f},{row['brier']:.4f}"
        )
    lines.append("headline_deltas")
    lines.append(f"fusion_temporal_vs_fusion_raw_acc_gain={blocked_gain:.4f}")
    lines.append(f"fusion_temporal_vs_magnitude_temporal_acc_gain={fusion_vs_mag:.4f}")
    lines.append(f"fusion_temporal_vs_phasor_temporal_acc_gain={fusion_vs_phasor:.4f}")
    lines.append(f"fusion_blocked_vs_fusion_forward_acc_gap={forward_gap:.4f}")
    lines.append(f"fusion_driftrobust_forward_vs_fusion_temporal_forward_acc_gain={drift_forward_gain:.4f}")
    lines.append(f"elapsed_sec={time.time() - t0:.1f}")

    text = "\n".join(lines) + "\n"
    out = PKG_STATS / OUT_NAME
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()


