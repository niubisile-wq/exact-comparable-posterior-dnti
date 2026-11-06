# -*- coding: utf-8 -*-
"""
SoCal measurement-conditioned event posterior trajectory audit.

Fits the synchronized real-measurement posterior model on the full event window
and inspects how posterior mass moves around real switch events.
"""

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import RobustScaler

warnings.simplefilter("ignore", PerformanceWarning)

TEMP_ROOT = Path.home() / "Desktop" / "配电网论文实验总汇_20260703" / "配电网实验_临时"
MEAS = TEMP_ROOT / "real_data_preview_samples"
STATUS = TEMP_ROOT / "digital-twin-dataset" / "sample_dataset" / "topology" / "parameter_timeseries"
PKG_ROOT = Path.home() / "Desktop" / "配电网论文实验总汇_20260703" / "配电网论文一区投稿成果包_20260702"
PKG_STATS = PKG_ROOT / "03_frozen_tables_stats"

OUT_NAME = "socal_event_posterior_trajectory_20260703.txt"
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


def make_models(seed):
    return [
        ExtraTreesClassifier(
            n_estimators=550,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        RandomForestClassifier(
            n_estimators=420,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed + 11,
            n_jobs=-1,
        ),
        HistGradientBoostingClassifier(
            max_iter=260,
            learning_rate=0.045,
            max_leaf_nodes=18,
            l2_regularization=0.04,
            class_weight="balanced",
            random_state=seed + 23,
        ),
    ]


def fit_probs(df, feature_cols, class_states):
    x = df[feature_cols].to_numpy(dtype=np.float32)
    y_raw = df["state_id"].to_numpy(dtype=np.int64)
    label_map = {sid: i for i, sid in enumerate(class_states)}
    y = np.array([label_map[int(v)] for v in y_raw], dtype=np.int64)

    scaler = RobustScaler(quantile_range=(5, 95))
    x_scaled = scaler.fit_transform(x)

    probs = []
    for seed in SEEDS:
        for model in make_models(seed):
            model.fit(x_scaled, y)
            probs.append(normalize_probs(model.predict_proba(x_scaled), len(class_states)))
    pmean = np.mean(np.stack(probs, axis=0), axis=0)
    return pmean


def event_summary(df, probs, event_time, pre_state, post_state):
    out = []
    pred = np.argmax(probs, axis=1)
    post_label_idx = post_state
    pre_label_idx = pre_state
    for win_sec in [30, 60, 180]:
        pre_mask = (df["t"] >= event_time - pd.Timedelta(seconds=win_sec)) & (df["t"] < event_time)
        post_mask = (df["t"] >= event_time) & (df["t"] <= event_time + pd.Timedelta(seconds=win_sec))
        pre_mass = float(np.mean(probs[pre_mask, pre_label_idx])) if np.any(pre_mask) else np.nan
        post_mass = float(np.mean(probs[post_mask, post_label_idx])) if np.any(post_mask) else np.nan
        pre_top1 = float(np.mean(pred[pre_mask] == pre_label_idx)) if np.any(pre_mask) else np.nan
        post_top1 = float(np.mean(pred[post_mask] == post_label_idx)) if np.any(post_mask) else np.nan
        out.append((win_sec, pre_mask.sum(), post_mask.sum(), pre_mass, post_mass, pre_top1, post_top1))

    after = df[df["t"] >= event_time].copy()
    delay_sec = np.nan
    if not after.empty:
        after_pred = pred[after.index.to_numpy()]
        hit = np.where(after_pred == post_label_idx)[0]
        if len(hit) > 0:
            delay_sec = float((after["t"].iloc[int(hit[0])] - event_time).total_seconds())
    return out, delay_sec


def main():
    t0 = time.time()
    meas = load_measurements()
    tables = load_status_tables()
    df0, uniq = attach_states(meas, tables)
    counts = df0["state_id"].value_counts().sort_index().to_dict()
    class_states = [int(k) for k, v in counts.items() if v >= 15]
    df = df0[df0["state_id"].isin(class_states)].copy().reset_index(drop=True)
    df, feature_cols = add_feature_bank(df)
    probs = fit_probs(df, feature_cols, class_states)

    lines = []
    lines.append("SoCal measurement-conditioned event posterior trajectory audit")
    lines.append("date=2026-07-03")
    lines.append("role=real switch-event trajectory evidence over synchronized measurement posterior")
    lines.append("not_claimed=independent held-out benchmark; state_4 excluded due to single synchronized sample")
    lines.append(f"window_utc={WINDOW_START.isoformat()} to {WINDOW_END.isoformat()}")
    lines.append(f"measurement_rows={len(df)}")
    lines.append(f"feature_dim={len(feature_cols)}")
    lines.append(f"trainable_states={' '.join(map(str, class_states))}")
    lines.append(f"ensemble_models_per_seed=3")
    lines.append(f"seeds={' '.join(map(str, SEEDS))}")
    lines.append("event_window_metrics")
    lines.append("device,event_utc,pre_state,post_state,window_sec,pre_n,post_n,pre_true_state_mass,post_true_state_mass,pre_top1_match,post_top1_match")
    for device, event_time, pre_state, post_state in EVENTS:
        rows, delay_sec = event_summary(df, probs, event_time, pre_state, post_state)
        for row in rows:
            lines.append(
                f"{device},{event_time.isoformat()},{pre_state},{post_state},{row[0]},{row[1]},{row[2]},"
                f"{row[3]:.4f},{row[4]:.4f},{row[5]:.4f},{row[6]:.4f}"
            )
        lines.append(f"event_switch_delay_sec,{device},{event_time.isoformat()},{delay_sec:.3f}")
    lines.append(f"elapsed_sec={time.time() - t0:.1f}")

    text = "\n".join(lines) + "\n"
    out = PKG_STATS / OUT_NAME
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()


