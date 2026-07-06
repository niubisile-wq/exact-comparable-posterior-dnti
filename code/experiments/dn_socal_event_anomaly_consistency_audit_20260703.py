# -*- coding: utf-8 -*-
"""
SoCal real-event anomaly and consistency audit.

Train a stable-state posterior model away from switching boundaries, replay it
over the full synchronized real window, and quantify whether real switch events
produce detectable posterior/consistency jumps beyond non-event baselines.
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

OUT_NAME = "socal_event_anomaly_consistency_audit_20260703.txt"
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
    ("cb_121", pd.Timestamp("2024-11-14T07:09:00")),
    ("cb_123", pd.Timestamp("2024-11-14T07:20:05")),
    ("cb_128", pd.Timestamp("2024-11-14T07:20:21")),
    ("swmp_17-2", pd.Timestamp("2024-11-14T07:30:42")),
]
TRAIN_BUFFER_SEC = 60
WINDOW_SEC = 30
NON_EVENT_EXCLUSION_SEC = 90
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


def empirical_auc(pos, neg):
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    wins = 0.0
    for x in pos:
        wins += np.sum(x > neg) + 0.5 * np.sum(x == neg)
    return float(wins / (len(pos) * len(neg)))


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


def candidate_score(df, probs, raw_mu, raw_sd, centroids):
    ent = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
    raw_z = (df[BASE_FEATURES].to_numpy(dtype=np.float64) - raw_mu) / raw_sd
    sqdist = np.sum((raw_z[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    posterior_expected_resid = np.sum(probs * sqdist, axis=1)

    rows = []
    for idx in range(len(df)):
        t = df.loc[idx, "t"]
        if t - pd.Timedelta(seconds=WINDOW_SEC) < WINDOW_START or t + pd.Timedelta(seconds=WINDOW_SEC) > WINDOW_END:
            continue
        pre_mask = (df["t"] >= t - pd.Timedelta(seconds=WINDOW_SEC)) & (df["t"] < t)
        post_mask = (df["t"] >= t) & (df["t"] <= t + pd.Timedelta(seconds=WINDOW_SEC))
        pre_idx = np.flatnonzero(pre_mask.to_numpy())
        post_idx = np.flatnonzero(post_mask.to_numpy())
        if len(pre_idx) < 5 or len(post_idx) < 5:
            continue

        pre_p = probs[pre_idx].mean(axis=0)
        post_p = probs[post_idx].mean(axis=0)
        tv = float(0.5 * np.sum(np.abs(post_p - pre_p)))
        js = js_divergence(pre_p, post_p)
        ent_jump = float(np.mean(ent[post_idx]) - np.mean(ent[pre_idx]))
        resid_jump = float(abs(np.mean(posterior_expected_resid[post_idx]) - np.mean(posterior_expected_resid[pre_idx])))

        pre_raw = df.iloc[pre_idx][BASE_FEATURES].to_numpy(dtype=np.float64)
        post_raw = df.iloc[post_idx][BASE_FEATURES].to_numpy(dtype=np.float64)
        pre_mean = pre_raw.mean(axis=0)
        post_mean = post_raw.mean(axis=0)
        pre_sd = pre_raw.std(axis=0, ddof=1) if len(pre_raw) > 1 else np.zeros(len(BASE_FEATURES))
        post_sd = post_raw.std(axis=0, ddof=1) if len(post_raw) > 1 else np.zeros(len(BASE_FEATURES))
        pooled = np.sqrt(np.maximum((pre_sd ** 2 + post_sd ** 2) / 2.0, 1e-9))
        shift_absz = float(np.max(np.abs((post_mean - pre_mean) / pooled)))

        rows.append(
            {
                "idx": idx,
                "t": t,
                "posterior_tv": tv,
                "posterior_js": js,
                "entropy_jump": ent_jump,
                "residual_jump": resid_jump,
                "measurement_shift_absz": shift_absz,
            }
        )
    return pd.DataFrame(rows)


def add_composite_score(scored):
    out = scored.copy()
    metric_cols = ["posterior_tv", "posterior_js", "residual_jump", "measurement_shift_absz"]
    for col in metric_cols:
        lo = out[col].quantile(0.05)
        hi = out[col].quantile(0.95)
        scale = max(hi - lo, 1e-9)
        out[f"{col}_norm"] = np.clip((out[col] - lo) / scale, 0.0, 1.0)
    out["event_score"] = out[[f"{c}_norm" for c in metric_cols]].mean(axis=1)
    out["topology_event_score"] = (
        0.60 * out["posterior_tv_norm"]
        + 0.30 * out["posterior_js_norm"]
        + 0.10 * out["residual_jump_norm"]
    )
    return out


def label_event_rows(scored):
    out = scored.copy()
    out["is_event"] = 0
    out["event_device"] = ""
    for dev, t in EVENTS:
        mask = out["t"] == t
        out.loc[mask, "is_event"] = 1
        out.loc[mask, "event_device"] = dev
    far = np.ones(len(out), dtype=bool)
    for _, t in EVENTS:
        far &= (np.abs((out["t"] - t).dt.total_seconds()) > NON_EVENT_EXCLUSION_SEC).to_numpy()
    out["is_far_non_event"] = far.astype(int)
    return out


def main():
    t0 = time.time()
    meas = load_measurements()
    tables = load_status_tables()
    df0, uniq = attach_states(meas, tables)
    df, feature_cols = add_feature_bank(df0)

    mask = stable_mask(df)
    stable_states = sorted(df.loc[mask, "state_id"].value_counts()[lambda s: s >= 40].index.astype(int).tolist())
    stable_train = df[mask & df["state_id"].isin(stable_states)].copy()

    scaler, models, raw_mu, raw_sd, centroids = fit_stable_model(stable_train, feature_cols, stable_states)
    probs = predict_full(df, feature_cols, scaler, models)
    scored = candidate_score(df, probs, raw_mu, raw_sd, centroids)
    scored = add_composite_score(scored)
    scored = label_event_rows(scored)

    event_rows = scored[scored["is_event"] == 1].sort_values("t").copy()
    baseline = scored[scored["is_far_non_event"] == 1].copy()

    if len(baseline):
        base_scores = baseline["event_score"].to_numpy(dtype=float)
        base_topo_scores = baseline["topology_event_score"].to_numpy(dtype=float)
        p95 = float(np.quantile(base_scores, 0.95))
        p99 = float(np.quantile(base_scores, 0.99))
        topo_p95 = float(np.quantile(base_topo_scores, 0.95))
        topo_p99 = float(np.quantile(base_topo_scores, 0.99))
    else:
        p95 = np.nan
        p99 = np.nan
        topo_p95 = np.nan
        topo_p99 = np.nan

    event_scores = event_rows["event_score"].to_numpy(dtype=float)
    topo_event_scores = event_rows["topology_event_score"].to_numpy(dtype=float)
    auc = empirical_auc(event_scores, baseline["event_score"].to_numpy(dtype=float)) if len(event_rows) and len(baseline) else np.nan
    topo_auc = empirical_auc(topo_event_scores, baseline["topology_event_score"].to_numpy(dtype=float)) if len(event_rows) and len(baseline) else np.nan

    lines = []
    lines.append("SoCal real-event anomaly and consistency audit")
    lines.append("date=2026-07-03")
    lines.append("role=stable-state posterior replay over the synchronized real window with event-level anomaly and consistency scoring")
    lines.append("not_claimed=field deployment; independent utility alarm labels")
    lines.append(f"window_utc={WINDOW_START.isoformat()} to {WINDOW_END.isoformat()}")
    lines.append(f"measurement_rows={len(df)}")
    lines.append(f"feature_dim={len(feature_cols)}")
    lines.append(f"stable_training_buffer_sec={TRAIN_BUFFER_SEC}")
    lines.append(f"evaluation_half_window_sec={WINDOW_SEC}")
    lines.append(f"stable_states={' '.join(map(str, stable_states))}")
    lines.append(f"stable_train_rows={len(stable_train)}")
    lines.append(f"candidate_rows={len(scored)}")
    lines.append(f"far_non_event_rows={len(baseline)}")
    lines.append("event_score_baseline")
    lines.append(f"baseline_event_score_p95={p95:.6f}")
    lines.append(f"baseline_event_score_p99={p99:.6f}")
    lines.append(f"event_vs_far_non_event_auc={auc:.4f}")
    lines.append(f"events_above_p95={int(np.sum(event_scores > p95)) if np.isfinite(p95) else 0}/{len(event_scores)}")
    lines.append(f"events_above_p99={int(np.sum(event_scores > p99)) if np.isfinite(p99) else 0}/{len(event_scores)}")
    lines.append("topology_event_score_baseline")
    lines.append(f"baseline_topology_event_score_p95={topo_p95:.6f}")
    lines.append(f"baseline_topology_event_score_p99={topo_p99:.6f}")
    lines.append(f"topology_event_vs_far_non_event_auc={topo_auc:.4f}")
    lines.append(f"topology_events_above_p95={int(np.sum(topo_event_scores > topo_p95)) if np.isfinite(topo_p95) else 0}/{len(topo_event_scores)}")
    lines.append(f"topology_events_above_p99={int(np.sum(topo_event_scores > topo_p99)) if np.isfinite(topo_p99) else 0}/{len(topo_event_scores)}")
    lines.append("event_rows")
    lines.append("device,event_utc,event_score,topology_event_score,posterior_tv,posterior_js,entropy_jump,residual_jump,measurement_shift_absz,score_percentile_vs_far_non_event,topology_score_percentile_vs_far_non_event")
    for _, r in event_rows.iterrows():
        if len(baseline):
            pct = float(np.mean(baseline["event_score"].to_numpy(dtype=float) <= float(r["event_score"])))
            topo_pct = float(np.mean(baseline["topology_event_score"].to_numpy(dtype=float) <= float(r["topology_event_score"])))
        else:
            pct = np.nan
            topo_pct = np.nan
        lines.append(
            f"{r['event_device']},{r['t'].isoformat()},{r['event_score']:.6f},{r['topology_event_score']:.6f},{r['posterior_tv']:.6f},{r['posterior_js']:.6f},"
            f"{r['entropy_jump']:.6f},{r['residual_jump']:.6f},{r['measurement_shift_absz']:.6f},{pct:.4f},{topo_pct:.4f}"
        )
    lines.append("baseline_means")
    for col in ["event_score", "topology_event_score", "posterior_tv", "posterior_js", "entropy_jump", "residual_jump", "measurement_shift_absz"]:
        lines.append(f"{col}_far_non_event_mean={baseline[col].mean():.6f}")
        lines.append(f"{col}_event_mean={event_rows[col].mean():.6f}")
    lines.append(f"elapsed_sec={time.time() - t0:.1f}")

    text = "\n".join(lines) + "\n"
    out = PKG_STATS / OUT_NAME
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()


