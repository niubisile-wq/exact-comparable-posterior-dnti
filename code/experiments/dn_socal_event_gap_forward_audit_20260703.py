# -*- coding: utf-8 -*-
"""
SoCal event-gap-aware forward audit.

This audit separates two deployment questions:
1. stable-state forward generalization away from switching boundaries
2. transition tracking across real switching events
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

OUT_NAME = "socal_event_gap_forward_audit_20260703.txt"
WINDOW_START = pd.Timestamp("2024-11-14T07:00:00")
WINDOW_END = pd.Timestamp("2024-11-14T07:30:42")
DEVICES = ["cb_121", "cb_123", "cb_128", "swmp_17-2"]
FILES = {
    "cb_121": STATUS / "cb_121-tbus_status.csv",
    "cb_123": STATUS / "cb_123-tbus_status.csv",
    "cb_128": STATUS / "cb_128-tbus_status.csv",
    "swmp_17-2": STATUS / "swmp_17-2-tbus_status.csv",
}
BASE_FEATURES = ["mains_v", "frequency", "rms", "magnitude_harmonic_0", "magnitude_harmonic_1", "magnitude_harmonic_2"]
EVENTS = [
    ("cb_121", pd.Timestamp("2024-11-14T07:09:00"), 0, 1),
    ("cb_123", pd.Timestamp("2024-11-14T07:20:05"), 1, 2),
    ("cb_128", pd.Timestamp("2024-11-14T07:20:21"), 2, 3),
]
BUFFER_SWEEP = [10, 15, 30, 60]
FORWARD_TRAIN_RATIO = 0.75
SEEDS = [42, 123]


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
    return merged.dropna(subset=BASE_FEATURES).reset_index(drop=True)


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


def add_feature_bank(df):
    out = df.copy()
    eps = 1e-9
    for col in BASE_FEATURES:
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
    out["v_over_rms"] = out["mains_v"] / (np.abs(out["rms"]) + eps)
    out["harmonic12_sum"] = out["magnitude_harmonic_1"] + out["magnitude_harmonic_2"]
    out["harmonic12_ratio"] = out["harmonic12_sum"] / (np.abs(out["magnitude_harmonic_0"]) + eps)
    out["freq_rms_product"] = out["frequency"] * out["rms"]
    feature_cols = [c for c in out.columns if c not in {"t", "state_id"}]
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, feature_cols


def stable_mask(df, buffer_sec):
    mask = np.ones(len(df), dtype=bool)
    for _, event_time, _, _ in EVENTS:
        lo = event_time - pd.Timedelta(seconds=buffer_sec)
        hi = event_time + pd.Timedelta(seconds=buffer_sec)
        mask &= ~((df["t"] >= lo) & (df["t"] <= hi)).to_numpy()
    return mask


def forward_split_indices(df, states):
    train_idx, test_idx = [], []
    for sid in states:
        idx = df.index[df["state_id"] == sid].to_numpy()
        if len(idx) < 8:
            continue
        cut = max(4, int(len(idx) * FORWARD_TRAIN_RATIO))
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


def boot_ci_acc(probs, y, boot=600):
    pred = np.argmax(probs, axis=1)
    correct = (pred == y).astype(float)
    rng = np.random.RandomState(704)
    vals = np.empty(boot, dtype=np.float64)
    for i in range(boot):
        idx = rng.randint(0, len(correct), len(correct))
        vals[i] = np.mean(correct[idx])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def make_model(seed):
    return ExtraTreesClassifier(
        n_estimators=220,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def evaluate_forward(df, feature_cols, states):
    train_idx, test_idx = forward_split_indices(df, states)
    label_map = {sid: i for i, sid in enumerate(states)}
    x = df[feature_cols].to_numpy(dtype=np.float32)
    y_raw = df["state_id"].to_numpy(dtype=np.int64)
    y = np.array([label_map[int(v)] for v in y_raw], dtype=np.int64)
    scaler = RobustScaler(quantile_range=(5, 95))
    x_train = scaler.fit_transform(x[train_idx])
    x_test = scaler.transform(x[test_idx])
    probs = []
    for seed in SEEDS:
        model = make_model(seed)
        model.fit(x_train, y[train_idx])
        probs.append(normalize_probs(model.predict_proba(x_test), len(states)))
    pmean = np.mean(np.stack(probs, axis=0), axis=0)
    pred = np.argmax(pmean, axis=1)
    acc = float(np.mean(pred == y[test_idx]))
    macro_f1 = float(f1_score(y[test_idx], pred, labels=list(range(len(states))), average="macro", zero_division=0))
    ci_lo, ci_hi = boot_ci_acc(pmean, y[test_idx])
    majority = max(np.bincount(y[test_idx], minlength=len(states))) / len(test_idx)
    return {
        "train_n": len(train_idx),
        "test_n": len(test_idx),
        "acc": acc,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "macro_f1": macro_f1,
        "majority": majority,
        "scaler": scaler,
        "models": [make_model(seed).fit(x_train, y[train_idx]) for seed in SEEDS],
        "label_map": label_map,
    }


def predict_full(df, feature_cols, scaler, models, states):
    x = df[feature_cols].to_numpy(dtype=np.float32)
    x_scaled = scaler.transform(x)
    probs = []
    for model in models:
        probs.append(normalize_probs(model.predict_proba(x_scaled), len(states)))
    return np.mean(np.stack(probs, axis=0), axis=0)


def event_metrics(df, probs, states):
    pred = np.argmax(probs, axis=1)
    label_index = {sid: i for i, sid in enumerate(states)}
    lines = []
    for device, event_time, pre_state, post_state in EVENTS:
        if pre_state not in label_index or post_state not in label_index:
            continue
        pre_idx = label_index[pre_state]
        post_idx = label_index[post_state]
        for win_sec in [30, 60, 180]:
            pre_mask = (df["t"] >= event_time - pd.Timedelta(seconds=win_sec)) & (df["t"] < event_time)
            post_mask = (df["t"] >= event_time) & (df["t"] <= event_time + pd.Timedelta(seconds=win_sec))
            pre_mass = float(np.mean(probs[pre_mask, pre_idx])) if np.any(pre_mask) else np.nan
            post_mass = float(np.mean(probs[post_mask, post_idx])) if np.any(post_mask) else np.nan
            pre_top1 = float(np.mean(pred[pre_mask] == pre_idx)) if np.any(pre_mask) else np.nan
            post_top1 = float(np.mean(pred[post_mask] == post_idx)) if np.any(post_mask) else np.nan
            lines.append((device, event_time, pre_state, post_state, win_sec, int(pre_mask.sum()), int(post_mask.sum()), pre_mass, post_mass, pre_top1, post_top1))

    pair_start = EVENTS[1][1]
    pair_end = EVENTS[2][1]
    if 1 in label_index and 3 in label_index:
        state1_idx = label_index[1]
        state3_idx = label_index[3]
        pre_pair = (df["t"] >= pair_start - pd.Timedelta(seconds=60)) & (df["t"] < pair_start)
        bridge = (df["t"] >= pair_start) & (df["t"] <= pair_end)
        post_pair = (df["t"] > pair_end) & (df["t"] <= pair_end + pd.Timedelta(seconds=60))
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-9, 1.0)), axis=1)
        pair_summary = {
            "pre_state1_mass": float(np.mean(probs[pre_pair, state1_idx])) if np.any(pre_pair) else np.nan,
            "bridge_mean_entropy": float(np.mean(entropy[bridge])) if np.any(bridge) else np.nan,
            "bridge_max_entropy": float(np.max(entropy[bridge])) if np.any(bridge) else np.nan,
            "post_state3_mass": float(np.mean(probs[post_pair, state3_idx])) if np.any(post_pair) else np.nan,
            "bridge_n": int(bridge.sum()),
        }
    else:
        pair_summary = None
    return lines, pair_summary


def main():
    t0 = time.time()
    meas = load_measurements()
    tables = load_status_tables()
    df0, uniq = attach_states(meas, tables)
    df, feature_cols = add_feature_bank(df0)

    results = []
    best = None
    for buffer_sec in BUFFER_SWEEP:
        mask = stable_mask(df, buffer_sec)
        dfs = df[mask].copy().reset_index(drop=True)
        counts = dfs["state_id"].value_counts().sort_index().to_dict()
        states = [int(k) for k, v in counts.items() if v >= 40]
        dfs = dfs[dfs["state_id"].isin(states)].copy().reset_index(drop=True)
        if len(states) < 2 or dfs.empty:
            continue
        res = evaluate_forward(dfs, feature_cols, states)
        row = {
            "buffer_sec": buffer_sec,
            "stable_rows": len(dfs),
            "states": states,
            **{k: v for k, v in res.items() if k not in {"scaler", "models", "label_map"}},
        }
        results.append(row)
        if best is None or row["acc"] > best["acc"]:
            best = {
                "buffer_sec": buffer_sec,
                "stable_rows": len(dfs),
                "states": states,
                "df": dfs,
                "scaler": res["scaler"],
                "models": res["models"],
                "acc": row["acc"],
                "majority": row["majority"],
                "macro_f1": row["macro_f1"],
                "ci_lo": row["ci_lo"],
                "ci_hi": row["ci_hi"],
            }

    lines = []
    lines.append("SoCal event-gap-aware forward audit")
    lines.append("date=2026-07-03")
    lines.append("role=stable-state forward generalization plus transition tracking under synchronized real measurements")
    lines.append("not_claimed=full window forward benchmark; state_2 treated as transient if removed by support rule")
    lines.append(f"window_utc={WINDOW_START.isoformat()} to {WINDOW_END.isoformat()}")
    lines.append(f"feature_dim={len(feature_cols)}")
    lines.append(f"forward_train_ratio={FORWARD_TRAIN_RATIO:.2f}")
    lines.append("stable_forward_metrics")
    lines.append("buffer_sec,stable_rows,states,train_n,test_n,acc,acc_ci95_lo,acc_ci95_hi,macro_f1,majority")
    for row in results:
        state_text = " ".join(map(str, row["states"]))
        lines.append(
            f"{row['buffer_sec']},{row['stable_rows']},{state_text},{row['train_n']},{row['test_n']},"
            f"{row['acc']:.4f},{row['ci_lo']:.4f},{row['ci_hi']:.4f},{row['macro_f1']:.4f},{row['majority']:.4f}"
        )

    if best is not None:
        lines.append("best_stable_forward")
        lines.append(f"buffer_sec={best['buffer_sec']}")
        lines.append(f"stable_rows={best['stable_rows']}")
        lines.append(f"states={' '.join(map(str, best['states']))}")
        lines.append(f"acc={best['acc']:.4f}")
        lines.append(f"acc_ci95={best['ci_lo']:.4f},{best['ci_hi']:.4f}")
        lines.append(f"macro_f1={best['macro_f1']:.4f}")
        lines.append(f"majority={best['majority']:.4f}")
        probs = predict_full(df, feature_cols, best["scaler"], best["models"], best["states"])
        event_rows, pair_summary = event_metrics(df, probs, best["states"])
        lines.append("transition_metrics")
        lines.append("device,event_utc,pre_state,post_state,window_sec,pre_n,post_n,pre_true_state_mass,post_true_state_mass,pre_top1_match,post_top1_match")
        for row in event_rows:
            lines.append(
                f"{row[0]},{row[1].isoformat()},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]},"
                f"{row[7]:.4f},{row[8]:.4f},{row[9]:.4f},{row[10]:.4f}"
            )
        if pair_summary is not None:
            lines.append("compressed_pair_transition")
            lines.append("pair=cb_123_to_cb_128")
            lines.append(f"bridge_n={pair_summary['bridge_n']}")
            lines.append(f"pre_state1_mass={pair_summary['pre_state1_mass']:.4f}")
            lines.append(f"bridge_mean_entropy={pair_summary['bridge_mean_entropy']:.4f}")
            lines.append(f"bridge_max_entropy={pair_summary['bridge_max_entropy']:.4f}")
            lines.append(f"post_state3_mass={pair_summary['post_state3_mass']:.4f}")

    lines.append(f"elapsed_sec={time.time() - t0:.1f}")
    text = "\n".join(lines) + "\n"
    out = PKG_STATS / OUT_NAME
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()


