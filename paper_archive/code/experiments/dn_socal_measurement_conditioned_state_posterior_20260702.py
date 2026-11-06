# -*- coding: utf-8 -*-
"""
SoCal measurement-conditioned topology-state posterior audit.

Uses synchronized public real measurements and forward-filled real topology
status labels inside the 2024-11-14T07:00:00--07:30:42 UTC event window.
This is a measurement-conditioned state audit, not a utility-grade posterior
benchmark with independently verified field labels.
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
OUT_NAME = "socal_measurement_conditioned_state_posterior_20260702.txt"

WINDOW_START = pd.Timestamp("2024-11-14T07:00:00")
WINDOW_END = pd.Timestamp("2024-11-14T07:30:42")
DEVICES = ["cb_121", "cb_123", "cb_128", "swmp_17-2"]
FILES = {
    "cb_121": STATUS / "cb_121-tbus_status.csv",
    "cb_123": STATUS / "cb_123-tbus_status.csv",
    "cb_128": STATUS / "cb_128-tbus_status.csv",
    "swmp_17-2": STATUS / "swmp_17-2-tbus_status.csv",
}
FEATURES = ["mains_v", "frequency", "rms", "magnitude_harmonic_0", "magnitude_harmonic_1", "magnitude_harmonic_2"]
SEEDS = [42, 123, 456]
FOLDS = 5
EPOCHS = 450
LR = 2e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TinyPosterior(nn.Module):
    def __init__(self, d, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(48, 48),
            nn.GELU(),
            nn.Linear(48, c),
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
    merged = pd.merge_asof(
        mains.sort_values("t"),
        s3.sort_values("t"),
        on="t",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=10),
    )
    merged = merged[(merged["t"] >= WINDOW_START) & (merged["t"] <= WINDOW_END)].copy()
    merged = merged.dropna(subset=FEATURES).reset_index(drop=True)
    return merged


def load_status_tables():
    tables = {}
    for dev, path in FILES.items():
        df = pd.read_csv(path)
        df["t"] = pd.to_datetime(df["t"]) + pd.Timedelta(hours=8)
        df = df.sort_values("t").reset_index(drop=True)
        tables[dev] = df
    return tables


def state_at(tables, t):
    out = []
    for dev in DEVICES:
        df = tables[dev]
        mask = df["t"] <= t
        out.append(str(df.loc[mask, "str"].iloc[-1]) if mask.any() else str(df["str"].iloc[0]))
    return tuple(out)


def attach_states(meas, tables):
    states = [state_at(tables, t) for t in meas["t"]]
    uniq = []
    sid = []
    for st in states:
        if st not in uniq:
            uniq.append(st)
        sid.append(uniq.index(st))
    meas = meas.copy()
    meas["state_id"] = sid
    return meas, uniq


def blocked_fold_indices(df, states, fold):
    train_idx = []
    test_idx = []
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
    acc = float(np.mean(pred == y)) if len(y) else float("nan")
    f1s = []
    for k in range(c):
        tp = np.sum((pred == k) & (y == k))
        fp = np.sum((pred == k) & (y != k))
        fn = np.sum((pred != k) & (y == k))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        f1s.append(float(f1))
    conf = np.max(probs, axis=1)
    correct = (pred == y).astype(float)
    ece = 0.0
    for lo in np.linspace(0, 0.9, 10):
        hi = lo + 0.1
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if np.any(mask):
            ece += np.mean(mask) * abs(float(np.mean(conf[mask])) - float(np.mean(correct[mask])))
    onehot = np.eye(c)[y]
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    return acc, float(np.mean(f1s)), ece, brier


def train_eval_fold(df, train_idx, test_idx, class_states, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = df[FEATURES].to_numpy(dtype=np.float32)
    y_raw = df["state_id"].to_numpy(dtype=np.int64)
    state_to_class = {sid: i for i, sid in enumerate(class_states)}
    y = np.array([state_to_class[int(v)] for v in y_raw], dtype=np.int64)

    mu = x[train_idx].mean(axis=0, keepdims=True)
    sd = x[train_idx].std(axis=0, keepdims=True) + 1e-6
    xz = (x - mu) / sd
    c = len(class_states)
    counts = np.bincount(y[train_idx], minlength=c).astype(np.float32)
    weights = (np.sum(counts) / np.maximum(counts, 1.0))
    weights = weights / np.mean(weights)

    model = TinyPosterior(x.shape[1], c).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))
    xt = torch.tensor(xz[train_idx], dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y[train_idx], dtype=torch.long, device=DEVICE)
    xv = torch.tensor(xz[test_idx], dtype=torch.float32, device=DEVICE)
    for _ in range(EPOCHS):
        model.train()
        logits = model(xt)
        loss = ce(logits, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(xv), dim=1).cpu().numpy()
    acc, macro_f1, ece, brier = metrics(probs, y[test_idx], c)
    return probs, y[test_idx], acc, macro_f1, ece, brier


def event_posterior_shift(df, probs_all, y_all, class_states):
    # probs_all is indexed to full df for trainable rows only; NaN rows ignored.
    events = [
        ("cb_121", pd.Timestamp("2024-11-14T07:09:00")),
        ("cb_123", pd.Timestamp("2024-11-14T07:20:05")),
        ("cb_128", pd.Timestamp("2024-11-14T07:20:21")),
    ]
    rows = []
    for dev, t in events:
        pre = df.index[(df["t"] >= t - pd.Timedelta(seconds=120)) & (df["t"] < t)].to_numpy()
        post = df.index[(df["t"] >= t) & (df["t"] <= t + pd.Timedelta(seconds=120))].to_numpy()
        pre = np.array([i for i in pre if i in probs_all], dtype=np.int64)
        post = np.array([i for i in post if i in probs_all], dtype=np.int64)
        if len(pre) == 0 or len(post) == 0:
            continue
        pre_state = int(df.loc[pre[-1], "state_id"])
        post_state = int(df.loc[post[0], "state_id"])
        if pre_state not in class_states or post_state not in class_states:
            rows.append((dev, t.isoformat(), pre_state, post_state, len(pre), len(post), math.nan, math.nan, math.nan))
            continue
        pre_class = class_states.index(pre_state)
        post_class = class_states.index(post_state)
        pre_prob = np.mean([probs_all[int(i)][pre_class] for i in pre])
        post_prob = np.mean([probs_all[int(i)][post_class] for i in post])
        post_gain = np.mean([probs_all[int(i)][post_class] for i in post]) - np.mean([probs_all[int(i)][post_class] for i in pre])
        rows.append((dev, t.isoformat(), pre_state, post_state, len(pre), len(post), pre_prob, post_prob, post_gain))
    return rows


def main():
    meas = load_measurements()
    tables = load_status_tables()
    df, uniq = attach_states(meas, tables)
    counts = df["state_id"].value_counts().sort_index().to_dict()
    trainable_states = [int(k) for k, v in counts.items() if v >= FOLDS * 3]
    # Exclude one-sample terminal state from train/test; retain 16-sample state if available.
    dft = df[df["state_id"].isin(trainable_states)].copy().reset_index(drop=True)
    class_states = sorted(trainable_states)

    fold_rows = []
    probs_by_index = {}
    y_by_index = {}
    for fold in range(FOLDS):
        train_idx, test_idx = blocked_fold_indices(dft, class_states, fold)
        fold_probs = []
        fold_y = None
        seed_metrics = []
        for seed in SEEDS:
            probs, y, acc, macro_f1, ece, brier = train_eval_fold(dft, train_idx, test_idx, class_states, seed + fold * 1000)
            fold_probs.append(probs)
            fold_y = y
            seed_metrics.append((acc, macro_f1, ece, brier))
        p = np.mean(np.stack(fold_probs, axis=0), axis=0)
        acc, macro_f1, ece, brier = metrics(p, fold_y, len(class_states))
        for pos, idx in enumerate(test_idx):
            probs_by_index[int(idx)] = p[pos]
            y_by_index[int(idx)] = int(fold_y[pos])
        fold_rows.append((fold, len(train_idx), len(test_idx), acc, macro_f1, ece, brier, seed_metrics))

    all_idx = sorted(probs_by_index.keys())
    all_probs = np.stack([probs_by_index[i] for i in all_idx], axis=0)
    all_y = np.array([y_by_index[i] for i in all_idx], dtype=np.int64)
    overall = metrics(all_probs, all_y, len(class_states))
    majority = max(np.bincount(all_y, minlength=len(class_states))) / len(all_y)
    event_rows = event_posterior_shift(dft, probs_by_index, y_by_index, class_states)

    lines = []
    lines.append("SoCal measurement-conditioned topology-state posterior audit")
    lines.append("date=2026-07-02")
    lines.append("role=synchronized real-measurement posterior audit with forward-filled public topology-state labels")
    lines.append("not_claimed=utility-grade field posterior benchmark or independent ground-truth deployment")
    lines.append(f"device={DEVICE}")
    lines.append(f"window_utc={WINDOW_START.isoformat()} to {WINDOW_END.isoformat()}")
    lines.append(f"measurement_rows={len(df)}")
    lines.append(f"feature_columns={' '.join(FEATURES)}")
    lines.append(f"unique_states={len(uniq)}")
    lines.append("state_counts")
    lines.append("state_id,count,signature")
    for sid, st in enumerate(uniq):
        sig = " ".join(f"{DEVICES[i]}:{st[i]}" for i in range(len(DEVICES)))
        lines.append(f"{sid},{counts.get(sid,0)},{sig}")
    lines.append(f"trainable_states={' '.join(map(str, class_states))}")
    excluded = [sid for sid in range(len(uniq)) if sid not in class_states]
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
    lines.append("event_posterior_shift_120s")
    lines.append("device,event_utc,pre_state,post_state,pre_n,post_n,mean_pre_state_prob,mean_post_state_prob,post_state_prob_gain")
    for r in event_rows:
        vals = []
        for x in r:
            if isinstance(x, float):
                vals.append("nan" if math.isnan(x) else f"{x:.4f}")
            else:
                vals.append(str(x))
        lines.append(",".join(vals))

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

