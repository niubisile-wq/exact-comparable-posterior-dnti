# -*- coding: utf-8 -*-
"""
SoCal synchronized event-window measurement shift audit.

This is a real-measurement experiment on the public preview files. It does not
claim ground-truth posterior labels. It checks whether topology-event times that
fall inside the synchronized window have measurable local changes in the
available magnitude/phasor streams.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home() / "Desktop" / "配电网论文实验总汇_20260703" / "配电网实验_临时"
MEAS = ROOT / "real_data_preview_samples"
STATUS = ROOT / "digital-twin-dataset" / "sample_dataset" / "topology" / "parameter_timeseries"
PKG_ROOT = Path.home() / "Desktop" / "配电网论文实验总汇_20260703" / "配电网论文一区投稿成果包_20260702"
PKG_STATS = PKG_ROOT / "03_frozen_tables_stats"
PKG_CODE = PKG_ROOT / "02_code"
OUT_NAME = "socal_event_measurement_shift_20260702.txt"

EVENT_FILES = {
    "cb_121": STATUS / "cb_121-tbus_status.csv",
    "cb_123": STATUS / "cb_123-tbus_status.csv",
    "cb_128": STATUS / "cb_128-tbus_status.csv",
    "swmp_17-2": STATUS / "swmp_17-2-tbus_status.csv",
}
WINDOW_START = pd.Timestamp("2024-11-14T07:00:00")
WINDOW_END = pd.Timestamp("2024-11-14T07:30:42")
PRE_POST_SECONDS = [60, 180, 300]


def load_measurements():
    mains = pd.read_csv(MEAS / "egauge_9-Mains_Power.csv")
    mains["t"] = pd.to_datetime(mains["t"])
    mains = mains.rename(columns={"v": "mains_v"})[["t", "mains_v"]]

    s3 = pd.read_csv(MEAS / "egauge_9-S3.csv")
    s3["t"] = pd.to_datetime(s3["t"])
    keep = ["t", "frequency", "rms", "magnitude_harmonic_0", "magnitude_harmonic_1", "magnitude_harmonic_2"]
    s3 = s3[keep]

    merged = pd.merge_asof(
        mains.sort_values("t"),
        s3.sort_values("t"),
        on="t",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=10),
    )
    merged = merged[(merged["t"] >= WINDOW_START) & (merged["t"] <= WINDOW_END)].copy()
    return merged


def load_events():
    rows = []
    for dev, path in EVENT_FILES.items():
        df = pd.read_csv(path)
        df["t"] = pd.to_datetime(df["t"])
        # Source timestamps are local SoCal time in the public topology files.
        # Convert local 2024-11-13 23:xx:xx to UTC 2024-11-14 07:xx:xx by +8 h.
        df["t_utc"] = df["t"] + pd.Timedelta(hours=8)
        prev = df["str"].shift(1)
        changed = df[(prev.notna()) & (df["str"] != prev)].copy()
        for _, r in changed.iterrows():
            if WINDOW_START <= r["t_utc"] <= WINDOW_END:
                rows.append({"device": dev, "t_utc": r["t_utc"], "state": str(r["str"])})
    rows.sort(key=lambda x: (x["t_utc"], x["device"]))
    return rows


def feature_stats(df, cols):
    out = {}
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(dtype=float)
        if len(x) == 0:
            out[c] = (np.nan, np.nan, 0)
        else:
            out[c] = (float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0, int(len(x)))
    return out


def audit_event(meas, event, seconds):
    t = event["t_utc"]
    pre = meas[(meas["t"] >= t - pd.Timedelta(seconds=seconds)) & (meas["t"] < t)]
    post = meas[(meas["t"] >= t) & (meas["t"] <= t + pd.Timedelta(seconds=seconds))]
    cols = ["mains_v", "frequency", "rms", "magnitude_harmonic_0", "magnitude_harmonic_1", "magnitude_harmonic_2"]
    ps = feature_stats(pre, cols)
    qs = feature_stats(post, cols)
    rows = []
    for c in cols:
        pre_mean, pre_sd, pre_n = ps[c]
        post_mean, post_sd, post_n = qs[c]
        pooled = np.sqrt(np.nanmean([pre_sd ** 2, post_sd ** 2])) if pre_n > 1 and post_n > 1 else np.nan
        delta = post_mean - pre_mean if np.isfinite(pre_mean) and np.isfinite(post_mean) else np.nan
        z = delta / pooled if pooled and np.isfinite(pooled) and pooled > 0 else np.nan
        rows.append({
            "device": event["device"],
            "event_utc": t.isoformat(),
            "state": event["state"],
            "window_sec": seconds,
            "feature": c,
            "pre_n": pre_n,
            "post_n": post_n,
            "pre_mean": pre_mean,
            "post_mean": post_mean,
            "delta": delta,
            "std_delta": z,
        })
    return rows


def main():
    meas = load_measurements()
    events = load_events()
    rows = []
    for ev in events:
        for sec in PRE_POST_SECONDS:
            rows.extend(audit_event(meas, ev, sec))
    res = pd.DataFrame(rows)

    lines = []
    lines.append("SoCal synchronized event-window measurement shift audit")
    lines.append("date=2026-07-02")
    lines.append("role=real-measurement synchronized event audit; not a ground-truth posterior benchmark")
    lines.append(f"measurement_window={WINDOW_START.isoformat()} to {WINDOW_END.isoformat()} UTC")
    lines.append(f"measurement_rows_after_asof_merge={len(meas)}")
    lines.append(f"event_count={len(events)}")
    lines.append("events")
    lines.append("device,event_utc,state")
    for ev in events:
        lines.append(f"{ev['device']},{ev['t_utc'].isoformat()},{ev['state']}")
    lines.append("feature_shift_rows")
    lines.append("device,event_utc,state,window_sec,feature,pre_n,post_n,pre_mean,post_mean,delta,std_delta")
    for _, r in res.iterrows():
        lines.append(
            f"{r['device']},{r['event_utc']},{r['state']},{int(r['window_sec'])},{r['feature']},"
            f"{int(r['pre_n'])},{int(r['post_n'])},{r['pre_mean']:.6g},{r['post_mean']:.6g},{r['delta']:.6g},{r['std_delta']:.6g}"
        )

    # Compact event-level score: maximum absolute standardized shift over features/window sizes.
    finite = res[np.isfinite(res["std_delta"])]
    lines.append("event_level_max_abs_standardized_shift")
    lines.append("device,event_utc,max_abs_std_delta,feature,window_sec")
    if len(finite):
        idx = finite.assign(absz=np.abs(finite["std_delta"])).groupby(["device", "event_utc"])["absz"].idxmax()
        for _, r in finite.loc[idx].sort_values(["event_utc", "device"]).iterrows():
            lines.append(f"{r['device']},{r['event_utc']},{abs(float(r['std_delta'])):.4f},{r['feature']},{int(r['window_sec'])}")

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

