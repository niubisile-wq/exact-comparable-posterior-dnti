# -*- coding: utf-8 -*-
"""
SoCal long-preview event-rank sweep.

Searches short-window topology-shift scores over the 24-hour public preview and
evaluates whether the three labeled switch events are ranked unusually high
among all local peaks, even when unlabeled regime shifts exist elsewhere.
"""

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler

warnings.simplefilter("ignore", PerformanceWarning)

TEMP_ROOT = Path.home() / "Desktop" / "配电网论文实验总汇_20260703" / "配电网实验_临时"
MEAS = TEMP_ROOT / "real_data_preview_samples"
STATUS = TEMP_ROOT / "digital-twin-dataset" / "sample_dataset" / "topology" / "parameter_timeseries"
PKG_ROOT = Path.home() / "Desktop" / "配电网论文实验总汇_20260703" / "配电网论文一区投稿成果包_20260702"
PKG_STATS = PKG_ROOT / "03_frozen_tables_stats"

OUT_NAME = "socal_long_preview_event_rank_sweep_20260703.txt"
TRAIN_WINDOW_START = pd.Timestamp("2024-11-14T07:00:00")
TRAIN_WINDOW_END = pd.Timestamp("2024-11-14T07:30:42")
LONG_WINDOW_START = pd.Timestamp("2024-11-14T07:00:00")
LONG_WINDOW_END = pd.Timestamp("2024-11-15T06:59:50.039133")
DEVICES = ["cb_121", "cb_123", "cb_128", "swmp_17-2"]
FILES = {
    "cb_121": STATUS / "cb_121-tbus_status.csv",
    "cb_123": STATUS / "cb_123-tbus_status.csv",
    "cb_128": STATUS / "cb_128-tbus_status.csv",
    "swmp_17-2": STATUS / "swmp_17-2-tbus_status.csv",
}
BASE_FEATURES = ["mains_v", "frequency", "rms", "magnitude_harmonic_0", "magnitude_harmonic_1", "magnitude_harmonic_2"]
EVENTS = [
    ("cb_121", pd.Timestamp("2024-11-14T07:09:00")),
    ("cb_123", pd.Timestamp("2024-11-14T07:20:05")),
    ("cb_128", pd.Timestamp("2024-11-14T07:20:21")),
]
TRAIN_BUFFER_SEC = 60
WINDOW_OPTIONS = [15, 30, 60, 120]
PEAK_SEP_SEC = 60
SEEDS = [42, 123]


def load_measurements():
    mains = pd.read_csv(MEAS / "egauge_9-Mains_Power.csv")
    mains["t"] = pd.to_datetime(mains["t"])
    mains = mains.rename(columns={"v": "mains_v"})[["t", "mains_v"]]
    s3 = pd.read_csv(MEAS / "egauge_9-S3.csv")
    s3["t"] = pd.to_datetime(s3["t"])
    s3 = s3[["t", "frequency", "rms", "magnitude_harmonic_0", "magnitude_harmonic_1", "magnitude_harmonic_2"]]
    merged = pd.merge_asof(
        s3.sort_values("t"),
        mains.sort_values("t"),
        on="t",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=10),
    )
    merged = merged[(merged["t"] >= LONG_WINDOW_START) & (merged["t"] <= LONG_WINDOW_END)].copy()
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


def stable_mask(df):
    mask = np.ones(len(df), dtype=bool)
    for _, event_time in EVENTS:
        lo = event_time - pd.Timedelta(seconds=TRAIN_BUFFER_SEC)
        hi = event_time + pd.Timedelta(seconds=TRAIN_BUFFER_SEC)
        mask &= ~((df["t"] >= lo) & (df["t"] <= hi)).to_numpy()
    return mask


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


def js_divergence(p, q):
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(np.clip(p, 1e-12, 1.0)) - np.log(np.clip(m, 1e-12, 1.0))))
    kl_qm = np.sum(q * (np.log(np.clip(q, 1e-12, 1.0)) - np.log(np.clip(m, 1e-12, 1.0))))
    return float(0.5 * (kl_pm + kl_qm))


def percentile_rank(x):
    return pd.Series(np.asarray(x, dtype=float)).rank(method="average", pct=True).to_numpy(dtype=float)


def measurement_shift_absz(df, t, window_sec):
    vals = []
    pre = df[(df["t"] >= t - pd.Timedelta(seconds=window_sec)) & (df["t"] < t)]
    post = df[(df["t"] >= t) & (df["t"] <= t + pd.Timedelta(seconds=window_sec))]
    for col in BASE_FEATURES:
        pre_vals = pd.to_numeric(pre[col], errors="coerce").dropna().to_numpy(dtype=float)
        post_vals = pd.to_numeric(post[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(pre_vals) < 2 or len(post_vals) < 2:
            continue
        pre_mean = float(np.mean(pre_vals))
        post_mean = float(np.mean(post_vals))
        pre_sd = float(np.std(pre_vals, ddof=1))
        post_sd = float(np.std(post_vals, ddof=1))
        pooled = float(np.sqrt(np.nanmean([pre_sd ** 2, post_sd ** 2])))
        if np.isfinite(pooled) and pooled > 0:
            vals.append(abs((post_mean - pre_mean) / pooled))
    return float(np.nanmax(vals)) if len(vals) else np.nan


def fit_stable_model(df, feature_cols, stable_states):
    dft = df[df["state_id"].isin(stable_states)].copy().reset_index(drop=True)
    x = dft[feature_cols].to_numpy(dtype=np.float32)
    y = np.array([stable_states.index(int(v)) for v in dft["state_id"].to_numpy(dtype=np.int64)], dtype=np.int64)
    scaler = RobustScaler(quantile_range=(5, 95))
    x_scaled = scaler.fit_transform(x)
    models = []
    for seed in SEEDS:
        model = ExtraTreesClassifier(
            n_estimators=220,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(x_scaled, y)
        models.append(model)
    raw_train = dft[BASE_FEATURES].to_numpy(dtype=np.float64)
    raw_mu = raw_train.mean(axis=0, keepdims=True)
    raw_sd = raw_train.std(axis=0, keepdims=True) + 1e-9
    raw_z = (raw_train - raw_mu) / raw_sd
    centroids = []
    for sid in stable_states:
        mask = dft["state_id"].to_numpy(dtype=np.int64) == sid
        centroids.append(raw_z[mask].mean(axis=0))
    centroids = np.stack(centroids, axis=0)
    return scaler, models, raw_mu, raw_sd, centroids


def predict_full(df, feature_cols, scaler, models):
    x = df[feature_cols].to_numpy(dtype=np.float32)
    x_scaled = scaler.transform(x)
    probs = []
    for model in models:
        probs.append(normalize_probs(model.predict_proba(x_scaled), models[0].n_classes_))
    return np.mean(np.stack(probs, axis=0), axis=0)


def build_scored(df, probs, raw_mu, raw_sd, centroids, window_sec):
    raw_z = (df[BASE_FEATURES].to_numpy(dtype=np.float64) - raw_mu) / raw_sd
    sqdist = np.sum((raw_z[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    posterior_expected_resid = np.sum(probs * sqdist, axis=1)
    rows = []
    for idx in range(len(df)):
        t = df.loc[idx, "t"]
        if t - pd.Timedelta(seconds=window_sec) < LONG_WINDOW_START or t + pd.Timedelta(seconds=window_sec) > LONG_WINDOW_END:
            continue
        pre_mask = (df["t"] >= t - pd.Timedelta(seconds=window_sec)) & (df["t"] < t)
        post_mask = (df["t"] >= t) & (df["t"] <= t + pd.Timedelta(seconds=window_sec))
        pre_idx = np.flatnonzero(pre_mask.to_numpy())
        post_idx = np.flatnonzero(post_mask.to_numpy())
        if len(pre_idx) < 3 or len(post_idx) < 3:
            continue
        pre_p = probs[pre_idx].mean(axis=0)
        post_p = probs[post_idx].mean(axis=0)
        tv = float(0.5 * np.sum(np.abs(post_p - pre_p)))
        js = js_divergence(pre_p, post_p)
        resid_jump = float(abs(np.mean(posterior_expected_resid[post_idx]) - np.mean(posterior_expected_resid[pre_idx])))
        shift_absz = measurement_shift_absz(df, t, window_sec)
        rows.append(
            {
                "t": t,
                "posterior_tv": tv,
                "posterior_js": js,
                "residual_jump": resid_jump,
                "measurement_shift_absz": shift_absz,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for col in ["posterior_tv", "posterior_js", "residual_jump", "measurement_shift_absz"]:
        out[f"{col}_pct"] = percentile_rank(out[col].to_numpy(dtype=float))
    out["score"] = (
        0.40 * out["posterior_tv_pct"]
        + 0.20 * out["posterior_js_pct"]
        + 0.20 * out["residual_jump_pct"]
        + 0.20 * out["measurement_shift_absz_pct"]
    )
    return out


def local_peaks(scored):
    scores = scored["score"].to_numpy(dtype=float)
    keep = np.ones(len(scored), dtype=bool)
    radius = PEAK_SEP_SEC
    times = scored["t"].to_numpy()
    for i in range(len(scored)):
        if not keep[i]:
            continue
        left = max(0, i - radius)
        right = min(len(scored), i + radius + 1)
        if np.any(scores[i] < scores[left:right]):
            keep[i] = False
    return scored[keep].sort_values("score", ascending=False).reset_index(drop=True)


def nearest_peak_rank(peaks, t):
    dt = np.abs((peaks["t"] - t).dt.total_seconds())
    if len(dt) == 0:
        return np.nan, np.nan, np.nan
    idx = int(np.argmin(dt.to_numpy()))
    return idx + 1, float(peaks.loc[idx, "score"]), float(dt.iloc[idx])


def main():
    t0 = time.time()
    meas = load_measurements()
    tables = load_status_tables()
    df0, _uniq = attach_states(meas, tables)
    df, feature_cols = add_feature_bank(df0)

    train_df = df[(df["t"] >= TRAIN_WINDOW_START) & (df["t"] <= TRAIN_WINDOW_END)].copy()
    mask = stable_mask(train_df)
    stable_states = sorted(train_df.loc[mask, "state_id"].value_counts()[lambda s: s >= 40].index.astype(int).tolist())
    stable_train = train_df[mask & train_df["state_id"].isin(stable_states)].copy()
    scaler, models, raw_mu, raw_sd, centroids = fit_stable_model(stable_train, feature_cols, stable_states)
    probs = predict_full(df, feature_cols, scaler, models)

    best = None
    rows = []
    for window_sec in WINDOW_OPTIONS:
        scored = build_scored(df, probs, raw_mu, raw_sd, centroids, window_sec)
        if scored.empty:
            continue
        peaks = local_peaks(scored)
        event_ranks = []
        hits_top50 = 0
        for dev, t in EVENTS:
            rank, score, dt = nearest_peak_rank(peaks, t)
            event_ranks.append((dev, rank, score, dt))
            if np.isfinite(rank) and rank <= 50 and dt <= 60:
                hits_top50 += 1
        mean_rank = float(np.nanmean([r[1] for r in event_ranks]))
        mean_dt = float(np.nanmean([r[3] for r in event_ranks]))
        top50_rate = hits_top50 / len(EVENTS)
        rows.append((window_sec, len(peaks), mean_rank, mean_dt, top50_rate))
        if best is None or (top50_rate > best["top50_rate"]) or (top50_rate == best["top50_rate"] and mean_rank < best["mean_rank"]):
            best = {"window_sec": window_sec, "peaks": peaks, "event_ranks": event_ranks, "mean_rank": mean_rank, "mean_dt": mean_dt, "top50_rate": top50_rate}

    lines = []
    lines.append("SoCal long-preview event-rank sweep")
    lines.append("date=2026-07-03")
    lines.append("role=24-hour local-peak ranking audit for labeled real switch events under a topology-shift score")
    lines.append("not_claimed=24-hour topology labels; field deployment")
    lines.append(f"long_window_utc={LONG_WINDOW_START.isoformat()} to {LONG_WINDOW_END.isoformat()}")
    lines.append(f"merged_measurement_rows={len(df)}")
    lines.append(f"stable_states={' '.join(map(str, stable_states))}")
    lines.append(f"stable_train_rows={len(stable_train)}")
    lines.append(f"peak_separation_sec={PEAK_SEP_SEC}")
    lines.append("window_sweep")
    lines.append("window_sec,peak_count,event_mean_rank,event_mean_abs_time_error_sec,event_top50_hit_rate")
    for row in rows:
        lines.append(f"{row[0]},{row[1]},{row[2]:.4f},{row[3]:.4f},{row[4]:.4f}")
    lines.append("best_setting")
    lines.append(f"window_sec={best['window_sec']}")
    lines.append(f"peak_count={len(best['peaks'])}")
    lines.append(f"event_mean_rank={best['mean_rank']:.4f}")
    lines.append(f"event_mean_abs_time_error_sec={best['mean_dt']:.4f}")
    lines.append(f"event_top50_hit_rate={best['top50_rate']:.4f}")
    lines.append("event_peak_ranks")
    lines.append("device,event_utc,nearest_peak_rank,nearest_peak_score,nearest_peak_abs_time_error_sec")
    for dev, rank, score, dt in best["event_ranks"]:
        lines.append(f"{dev},{dict(EVENTS)[dev].isoformat()},{int(rank) if np.isfinite(rank) else 'nan'},{score:.6f},{dt:.4f}")
    lines.append("top20_peaks")
    lines.append("rank,peak_utc,score")
    for i, r in best["peaks"].head(20).iterrows():
        lines.append(f"{i+1},{r['t'].isoformat()},{r['score']:.6f}")
    lines.append(f"elapsed_sec={time.time() - t0:.1f}")

    text = "\n".join(lines) + "\n"
    out = PKG_STATS / OUT_NAME
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()


