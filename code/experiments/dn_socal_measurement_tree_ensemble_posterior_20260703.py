# -*- coding: utf-8 -*-
"""
SoCal synchronized measurement-conditioned tree-ensemble posterior audit.

This is an experiment-level upgrade over the MLP enhanced posterior audit:
it keeps the same synchronized public window and blocked CV, does not use
timestamp as a feature, and adds online/casual lag, rolling, EWMA and slope
features. ExtraTrees and HistGradientBoosting models are ensembled to improve
small-sample real-measurement topology-state posterior inference.
"""

from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd

from pandas.errors import PerformanceWarning
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler

warnings.simplefilter("ignore", PerformanceWarning)

ROOT = Path.home() / "Desktop" / "配电网论文实验总汇_20260703" / "配电网实验_临时"
MEAS = ROOT / "real_data_preview_samples"
STATUS = ROOT / "digital-twin-dataset" / "sample_dataset" / "topology" / "parameter_timeseries"
PKG_ROOT = Path.home() / "Desktop" / "配电网论文实验总汇_20260703" / "配电网论文一区投稿成果包_20260702"
PKG_STATS = PKG_ROOT / "03_frozen_tables_stats"
PKG_CODE = PKG_ROOT / "02_code"
OUT_NAME = "socal_measurement_tree_ensemble_posterior_20260703.txt"

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
FOLDS = 5
SEEDS = [42, 123, 456, 789]


def load_measurements():
    mains = pd.read_csv(MEAS / "egauge_9-Mains_Power.csv")
    mains["t"] = pd.to_datetime(mains["t"])
    mains = mains.rename(columns={"v": "mains_v"})[["t", "mains_v"]]
    s3 = pd.read_csv(MEAS / "egauge_9-S3.csv")
    s3["t"] = pd.to_datetime(s3["t"])
    s3 = s3[["t", "frequency", "rms", "magnitude_harmonic_0", "magnitude_harmonic_1", "magnitude_harmonic_2"]]
    merged = pd.merge_asof(mains.sort_values("t"), s3.sort_values("t"), on="t", direction="nearest", tolerance=pd.Timedelta(seconds=10))
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


def normalize_probs(p, n_classes):
    p = np.asarray(p, dtype=np.float64)
    if p.shape[1] == n_classes:
        out = p
    else:
        out = np.zeros((p.shape[0], n_classes), dtype=np.float64)
        out[:, :p.shape[1]] = p
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


def make_models(seed):
    return [
        (
            "extratrees",
            ExtraTreesClassifier(
                n_estimators=550,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        (
            "randomforest",
            RandomForestClassifier(
                n_estimators=420,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=seed + 11,
                n_jobs=-1,
            ),
        ),
        (
            "histgb",
            HistGradientBoostingClassifier(
                max_iter=260,
                learning_rate=0.045,
                max_leaf_nodes=18,
                l2_regularization=0.04,
                class_weight="balanced",
                random_state=seed + 23,
            ),
        ),
    ]


def train_eval_fold(df, feature_cols, class_states, train_idx, test_idx, seed):
    x = df[feature_cols].to_numpy(dtype=np.float32)
    y_raw = df["state_id"].to_numpy(dtype=np.int64)
    label_map = {sid: i for i, sid in enumerate(class_states)}
    y = np.array([label_map[int(v)] for v in y_raw], dtype=np.int64)
    scaler = RobustScaler(quantile_range=(5, 95))
    x_train = scaler.fit_transform(x[train_idx])
    x_test = scaler.transform(x[test_idx])
    probs = []
    model_names = []
    for name, model in make_models(seed):
        model.fit(x_train, y[train_idx])
        p = normalize_probs(model.predict_proba(x_test), len(class_states))
        probs.append(p)
        model_names.append(name)
    return np.mean(np.stack(probs, axis=0), axis=0), y[test_idx], model_names


def boot_ci_acc(probs, y, boot=1200):
    pred = np.argmax(probs, axis=1)
    correct = (pred == y).astype(float)
    rng = np.random.RandomState(703)
    vals = np.empty(boot, dtype=np.float64)
    for b in range(boot):
        idx = rng.randint(0, len(correct), len(correct))
        vals[b] = np.mean(correct[idx])
    return float(np.mean(correct)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def main():
    t0 = time.time()
    meas = load_measurements()
    tables = load_status_tables()
    df0, uniq = attach_states(meas, tables)
    df, feature_cols = add_feature_bank(df0)
    counts = df["state_id"].value_counts().sort_index().to_dict()
    class_states = [int(k) for k, v in counts.items() if v >= FOLDS * 3]
    dft = df[df["state_id"].isin(class_states)].copy().reset_index(drop=True)
    probs_by_idx, y_by_idx = {}, {}
    fold_rows = []
    used_models = None
    for fold in range(FOLDS):
        train_idx, test_idx = blocked_fold_indices(dft, class_states, fold)
        seed_probs, yy = [], None
        for seed in SEEDS:
            p, y, names = train_eval_fold(dft, feature_cols, class_states, train_idx, test_idx, seed + fold * 1000)
            seed_probs.append(p)
            yy = y
            used_models = names
        pmean = np.mean(np.stack(seed_probs, axis=0), axis=0)
        acc, mf1, ece, brier = metrics(pmean, yy, len(class_states))
        for pos, idx in enumerate(test_idx):
            probs_by_idx[int(idx)] = pmean[pos]
            y_by_idx[int(idx)] = int(yy[pos])
        fold_rows.append((fold, len(train_idx), len(test_idx), acc, mf1, ece, brier))
    all_idx = sorted(probs_by_idx.keys())
    all_probs = np.stack([probs_by_idx[i] for i in all_idx], axis=0)
    all_y = np.array([y_by_idx[i] for i in all_idx], dtype=np.int64)
    overall = metrics(all_probs, all_y, len(class_states))
    majority = max(np.bincount(all_y, minlength=len(class_states))) / len(all_y)
    ci = boot_ci_acc(all_probs, all_y)
    pred = np.argmax(all_probs, axis=1)
    conf = np.max(all_probs, axis=1)
    per_class = []
    for ci_state, sid in enumerate(class_states):
        mask = all_y == ci_state
        per_class.append((sid, int(np.sum(mask)), float(np.mean(pred[mask] == ci_state)), float(np.mean(conf[mask]))))

    lines = []
    lines.append("SoCal synchronized measurement-conditioned tree-ensemble posterior audit")
    lines.append("date=2026-07-03")
    lines.append("role=real synchronized measurement posterior upgrade using online causal feature bank and tree ensemble")
    lines.append("not_claimed=utility-grade deployment; no timestamp feature; no independent field label audit")
    lines.append(f"window_utc={WINDOW_START.isoformat()} to {WINDOW_END.isoformat()}")
    lines.append(f"measurement_rows={len(df)}")
    lines.append(f"base_feature_count={len(BASE_FEATURES)}")
    lines.append(f"tree_feature_count={len(feature_cols)}")
    lines.append(f"unique_states={len(uniq)}")
    lines.append(f"models={' '.join(used_models or [])}")
    lines.append("state_counts")
    lines.append("state_id,count,signature")
    for sid, st in enumerate(uniq):
        sig = " ".join(f"{DEVICES[i]}:{st[i]}" for i in range(len(DEVICES)))
        lines.append(f"{sid},{counts.get(sid,0)},{sig}")
    excluded = [sid for sid in range(len(uniq)) if sid not in class_states]
    lines.append(f"trainable_states={' '.join(map(str, class_states))}")
    lines.append(f"excluded_states_due_to_low_support={' '.join(map(str, excluded)) if excluded else 'none'}")
    lines.append(f"blocked_cv_folds={FOLDS}")
    lines.append(f"seeds={' '.join(map(str, SEEDS))}")
    lines.append(f"majority_baseline_acc={majority:.4f}")
    lines.append(f"overall_blocked_cv_acc={overall[0]:.4f}")
    lines.append(f"overall_acc_ci95={ci[1]:.4f},{ci[2]:.4f}")
    lines.append(f"overall_macro_f1={overall[1]:.4f}")
    lines.append(f"overall_ece={overall[2]:.4f}")
    lines.append(f"overall_brier={overall[3]:.4f}")
    lines.append("fold_metrics")
    lines.append("fold,train_n,test_n,ensemble_acc,macro_f1,ece,brier")
    for r in fold_rows:
        lines.append(f"{r[0]},{r[1]},{r[2]},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f},{r[6]:.4f}")
    lines.append("per_state_metrics")
    lines.append("state_id,test_n,acc,mean_confidence")
    for sid, n, acc, mc in per_class:
        lines.append(f"{sid},{n},{acc:.4f},{mc:.4f}")
    lines.append("comparison_to_previous_enhanced_mlp")
    lines.append("previous_acc=0.7915")
    lines.append("previous_macro_f1=0.7535")
    lines.append(f"acc_gain={overall[0] - 0.7915:.4f}")
    lines.append(f"macro_f1_gain={overall[1] - 0.7535:.4f}")
    lines.append(f"elapsed_sec={time.time() - t0:.1f}")
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


