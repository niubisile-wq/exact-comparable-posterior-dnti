# -*- coding: utf-8 -*-
"""
Enhanced SoCal measurement-conditioned topology-state posterior audit.

Adds online-available trailing rolling statistics and finite differences to the
public synchronized measurement features. It does not use timestamp as a feature
and does not claim private utility-grade field validation.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path.home() / "Desktop" / "配电网论文实验总汇_20260703" / "配电网实验_临时"
MEAS = ROOT / "real_data_preview_samples"
STATUS = ROOT / "digital-twin-dataset" / "sample_dataset" / "topology" / "parameter_timeseries"
PKG_ROOT = Path.home() / "Desktop" / "配电网论文实验总汇_20260703" / "配电网论文一区投稿成果包_20260702"
PKG_STATS = PKG_ROOT / "03_frozen_tables_stats"
PKG_CODE = PKG_ROOT / "02_code"
OUT_NAME = "socal_measurement_conditioned_enhanced_posterior_20260702.txt"

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
SEEDS = [42, 123, 456]
FOLDS = 5
EPOCHS = 520
LR = 1.7e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EnhancedPosterior(nn.Module):
    def __init__(self, d, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(0.08),
            nn.Linear(96, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(96, 48), nn.GELU(), nn.Linear(48, c),
        )
    def forward(self, x):
        return self.net(x)


def load_measurements():
    mains = pd.read_csv(MEAS / "egauge_9-Mains_Power.csv")
    mains["t"] = pd.to_datetime(mains["t"])
    mains = mains.rename(columns={"v": "mains_v"})[["t", "mains_v"]]
    s3 = pd.read_csv(MEAS / "egauge_9-S3.csv")
    s3["t"] = pd.to_datetime(s3["t"])
    s3 = s3[["t", "frequency", "rms", "magnitude_harmonic_0", "magnitude_harmonic_1", "magnitude_harmonic_2"]]
    merged = pd.merge_asof(mains.sort_values("t"), s3.sort_values("t"), on="t", direction="nearest", tolerance=pd.Timedelta(seconds=10))
    merged = merged[(merged["t"] >= WINDOW_START) & (merged["t"] <= WINDOW_END)].copy()
    merged = merged.dropna(subset=BASE_FEATURES).reset_index(drop=True)
    return merged


def add_online_features(df):
    out = df.copy()
    for col in BASE_FEATURES:
        out[f"{col}_d1"] = out[col].diff(1).fillna(0.0)
        out[f"{col}_d10"] = out[col].diff(10).fillna(0.0)
        for w in [15, 60, 180]:
            roll = out[col].rolling(window=w, min_periods=2)
            out[f"{col}_mean{w}"] = roll.mean().fillna(out[col])
            out[f"{col}_std{w}"] = roll.std().fillna(0.0)
            out[f"{col}_devmean{w}"] = out[col] - out[f"{col}_mean{w}"]
    feature_cols = [c for c in out.columns if c != "t" and c not in {"state_id"}]
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, feature_cols


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
    meas = meas.copy()
    meas["state_id"] = sid
    return meas, uniq


def blocked_fold_indices(df, states, fold):
    train_idx, test_idx = [], []
    for sid in states:
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


def metrics(probs, y, c):
    pred = np.argmax(probs, axis=1)
    acc = float(np.mean(pred == y))
    f1s = []
    for k in range(c):
        tp = np.sum((pred == k) & (y == k)); fp = np.sum((pred == k) & (y != k)); fn = np.sum((pred != k) & (y == k))
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1s.append(float(2 * prec * rec / max(prec + rec, 1e-12)))
    conf = np.max(probs, axis=1); corr = (pred == y).astype(float)
    ece = 0.0
    for lo in np.linspace(0, 0.9, 10):
        hi = lo + 0.1
        mask = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if np.any(mask):
            ece += np.mean(mask) * abs(float(np.mean(conf[mask])) - float(np.mean(corr[mask])))
    onehot = np.eye(c)[y]
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    return acc, float(np.mean(f1s)), ece, brier


def train_eval(df, feature_cols, train_idx, test_idx, class_states, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    x = df[feature_cols].to_numpy(dtype=np.float32)
    y_raw = df["state_id"].to_numpy(dtype=np.int64)
    mp = {sid: i for i, sid in enumerate(class_states)}
    y = np.array([mp[int(v)] for v in y_raw], dtype=np.int64)
    mu = x[train_idx].mean(axis=0, keepdims=True); sd = x[train_idx].std(axis=0, keepdims=True) + 1e-6
    xz = np.clip((x - mu) / sd, -8.0, 8.0)
    c = len(class_states)
    counts = np.bincount(y[train_idx], minlength=c).astype(np.float32)
    weights = (np.sum(counts) / np.maximum(counts, 1.0)); weights = weights / np.mean(weights)
    model = EnhancedPosterior(xz.shape[1], c).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-3)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))
    xt = torch.tensor(xz[train_idx], dtype=torch.float32, device=DEVICE); yt = torch.tensor(y[train_idx], dtype=torch.long, device=DEVICE)
    xv = torch.tensor(xz[test_idx], dtype=torch.float32, device=DEVICE)
    best_state, best_loss = None, float("inf")
    for epoch in range(EPOCHS):
        model.train(); logits = model(xt); loss = ce(logits, yt)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step()
        if float(loss.detach().cpu()) < best_loss:
            best_loss = float(loss.detach().cpu())
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(xv), dim=1).cpu().numpy()
    return probs, y[test_idx]


def main():
    meas = load_measurements()
    tables = load_status_tables()
    df0, uniq = attach_states(meas, tables)
    df, feature_cols = add_online_features(df0)
    counts = df["state_id"].value_counts().sort_index().to_dict()
    class_states = [int(k) for k, v in counts.items() if v >= FOLDS * 3]
    dft = df[df["state_id"].isin(class_states)].copy().reset_index(drop=True)
    probs_by_idx, y_by_idx = {}, {}
    fold_rows = []
    for fold in range(FOLDS):
        train_idx, test_idx = blocked_fold_indices(dft, class_states, fold)
        seed_probs, yy = [], None
        for seed in SEEDS:
            p, y = train_eval(dft, feature_cols, train_idx, test_idx, class_states, seed + fold * 1000)
            seed_probs.append(p); yy = y
        pmean = np.mean(np.stack(seed_probs, axis=0), axis=0)
        acc, mf1, ece, brier = metrics(pmean, yy, len(class_states))
        for pos, idx in enumerate(test_idx):
            probs_by_idx[int(idx)] = pmean[pos]; y_by_idx[int(idx)] = int(yy[pos])
        fold_rows.append((fold, len(train_idx), len(test_idx), acc, mf1, ece, brier))
    all_idx = sorted(probs_by_idx.keys())
    all_probs = np.stack([probs_by_idx[i] for i in all_idx], axis=0)
    all_y = np.array([y_by_idx[i] for i in all_idx], dtype=np.int64)
    overall = metrics(all_probs, all_y, len(class_states))
    majority = max(np.bincount(all_y, minlength=len(class_states))) / len(all_y)
    pred = np.argmax(all_probs, axis=1)
    conf = np.max(all_probs, axis=1)
    per_class = []
    for ci, sid in enumerate(class_states):
        mask = all_y == ci
        per_class.append((sid, int(np.sum(mask)), float(np.mean(pred[mask] == ci)), float(np.mean(conf[mask]))))
    lines = []
    lines.append("Enhanced SoCal measurement-conditioned topology-state posterior audit")
    lines.append("date=2026-07-02")
    lines.append("role=real synchronized measurement posterior audit using online rolling/difference features")
    lines.append("not_claimed=utility-grade deployment; no timestamp feature; no independent field label audit")
    lines.append(f"device={DEVICE}")
    lines.append(f"window_utc={WINDOW_START.isoformat()} to {WINDOW_END.isoformat()}")
    lines.append(f"measurement_rows={len(df)}")
    lines.append(f"base_feature_count={len(BASE_FEATURES)}")
    lines.append(f"enhanced_feature_count={len(feature_cols)}")
    lines.append(f"unique_states={len(uniq)}")
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

