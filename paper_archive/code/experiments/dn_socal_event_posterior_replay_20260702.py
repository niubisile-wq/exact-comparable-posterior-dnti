# -*- coding: utf-8 -*-
"""
SoCal event-conditioned posterior replay.

Builds a topology-state library from the four real switching events inside the
synchronized public measurement window and evaluates exact posterior recovery
for each real pre/post event transition under controlled replay observations.
This is not a ground-truth real-measurement posterior benchmark.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch

import dn_socal_replay_benchmark as src

ROOT = Path.home() / "Desktop" / "配电网论文实验总汇_20260703" / "配电网实验_临时"
STATUS = ROOT / "digital-twin-dataset" / "sample_dataset" / "topology" / "parameter_timeseries"
PKG_ROOT = Path.home() / "Desktop" / "配电网论文实验总汇_20260703" / "配电网论文一区投稿成果包_20260702"
PKG_STATS = PKG_ROOT / "03_frozen_tables_stats"
PKG_CODE = PKG_ROOT / "02_code"
OUT_NAME = "socal_event_posterior_replay_20260702.txt"

DEVICES = ["cb_121", "cb_123", "cb_128", "swmp_17-2"]
FILES = {
    "cb_121": STATUS / "cb_121-tbus_status.csv",
    "cb_123": STATUS / "cb_123-tbus_status.csv",
    "cb_128": STATUS / "cb_128-tbus_status.csv",
    "swmp_17-2": STATUS / "swmp_17-2-tbus_status.csv",
}
WINDOW_START = pd.Timestamp("2024-11-14T07:00:00")
WINDOW_END = pd.Timestamp("2024-11-14T07:30:42")
LOAD_REGIMES = [0.9, 1.0, 1.1]
SIGMA = 0.0015
K_OBS = 40
N_PER_STATE = 400


def read_tables():
    tables = {}
    for dev, path in FILES.items():
        df = pd.read_csv(path)
        df["t"] = pd.to_datetime(df["t"]) + pd.Timedelta(hours=8)
        df = df.sort_values("t").reset_index(drop=True)
        tables[dev] = df
    return tables


def state_at(tables, t):
    st = {}
    for dev, df in tables.items():
        mask = df["t"] <= t
        if mask.any():
            st[dev] = str(df.loc[mask, "str"].iloc[-1])
        else:
            st[dev] = str(df["str"].iloc[0])
    return st


def event_times(tables):
    rows = []
    for dev, df in tables.items():
        prev = df["str"].shift(1)
        changed = df[(prev.notna()) & (df["str"] != prev)].copy()
        for _, r in changed.iterrows():
            if WINDOW_START <= r["t"] <= WINDOW_END:
                rows.append({"device": dev, "t": r["t"], "state": str(r["str"])})
    rows.sort(key=lambda x: (x["t"], x["device"]))
    return rows


def unique_states_from_events(tables, events):
    times = [WINDOW_START] + [e["t"] for e in events]
    states = []
    labels = []
    seen = {}
    sequence = []
    for i, t in enumerate(times):
        st = state_at(tables, t)
        key = tuple(st[d] for d in DEVICES)
        if key not in seen:
            seen[key] = len(states)
            states.append(st)
            labels.append("initial" if i == 0 else events[i-1]["device"])
        sequence.append(seen[key])
    return states, labels, sequence


def posterior_from_obs(library, obs):
    # library shape: states x load_regimes x K
    arr = library.numpy()
    ll = -0.5 * np.sum(((arr - obs[None, None, :]) / SIGMA) ** 2, axis=2)
    # uniform load-regime prior: logsumexp over load regimes, then softmax over states
    mx_l = np.max(ll, axis=1, keepdims=True)
    state_ll = np.squeeze(mx_l, axis=1) + np.log(np.mean(np.exp(ll - mx_l), axis=1) + 1e-300)
    q = np.exp(state_ll - np.max(state_ll))
    q /= np.sum(q)
    return q.astype(np.float64)


def eval_state(library, state_idx, seed):
    rng = np.random.RandomState(seed)
    n_states, n_lf, k = library.shape
    arr = library.numpy()
    correct = 0
    mass = []
    entropy = []
    top2_gap = []
    for _ in range(N_PER_STATE):
        lf_idx = rng.randint(0, n_lf)
        obs = arr[state_idx, lf_idx, :] + rng.normal(0.0, SIGMA, size=k)
        q = posterior_from_obs(library, obs)
        order = np.argsort(-q)
        correct += int(order[0] == state_idx)
        mass.append(float(q[state_idx]))
        entropy.append(float(-np.sum(q * np.log(q + 1e-12))))
        top2_gap.append(float(q[order[0]] - q[order[1]]) if n_states > 1 else 1.0)
    return {
        "acc": correct / float(N_PER_STATE),
        "mean_mass": float(np.mean(mass)),
        "mean_entropy": float(np.mean(entropy)),
        "mean_top2_gap": float(np.mean(top2_gap)),
    }


def main():
    # Force a controlled load-regime grid for this replay audit.
    src.LOAD_REGIMES = LOAD_REGIMES
    src.K_OBS = K_OBS
    tables = read_tables()
    events = event_times(tables)
    states, labels, sequence = unique_states_from_events(tables, events)
    bus_kv, edges, sources, load_buses = src.load_network()
    common_buses, graphs = src.source_reachable_buses(states, edges, sources)
    full_library = src.build_library(states, graphs, common_buses, common_buses, sources, bus_kv, load_buses)
    obs_buses = src.choose_obs_buses(common_buses, full_library, load_buses)
    library = src.build_library(states, graphs, common_buses, obs_buses, sources, bus_kv, load_buses)

    state_metrics = []
    for i in range(len(states)):
        state_metrics.append(eval_state(library, i, 720000 + i))

    lines = []
    lines.append("SoCal synchronized event-conditioned posterior replay")
    lines.append("date=2026-07-02")
    lines.append("role=exact posterior replay over real event-derived topology-state sequence; not real-measurement posterior label benchmark")
    lines.append(f"window_utc={WINDOW_START.isoformat()} to {WINDOW_END.isoformat()}")
    lines.append(f"devices={' '.join(DEVICES)}")
    lines.append(f"event_count={len(events)}")
    lines.append(f"unique_topology_states={len(states)}")
    lines.append(f"sequence_length={len(sequence)}")
    lines.append(f"reachable_union_buses={len(common_buses)}")
    lines.append(f"observed_buses_K={len(obs_buses)}")
    lines.append(f"load_regimes={' '.join(str(x) for x in LOAD_REGIMES)}")
    lines.append("events")
    lines.append("idx,device,event_utc,state,pre_state,post_state")
    for i, ev in enumerate(events):
        lines.append(f"{i},{ev['device']},{ev['t'].isoformat()},{ev['state']},{sequence[i]},{sequence[i+1]}")
    lines.append("state_metrics")
    lines.append("state_idx,label,posterior_top1,mean_true_mass,mean_entropy,mean_top2_gap,state_signature")
    for i, st in enumerate(states):
        sig = " ".join(f"{d}:{st[d]}" for d in DEVICES)
        r = state_metrics[i]
        lines.append(f"{i},{labels[i]},{r['acc']:.4f},{r['mean_mass']:.4f},{r['mean_entropy']:.4f},{r['mean_top2_gap']:.4f},{sig}")
    lines.append("transition_metrics")
    lines.append("event_idx,device,pre_state,post_state,pre_acc,post_acc,pre_mass,post_mass,post_entropy")
    for i, ev in enumerate(events):
        pre = sequence[i]
        post = sequence[i + 1]
        pr = state_metrics[pre]
        po = state_metrics[post]
        lines.append(f"{i},{ev['device']},{pre},{post},{pr['acc']:.4f},{po['acc']:.4f},{pr['mean_mass']:.4f},{po['mean_mass']:.4f},{po['mean_entropy']:.4f}")
    lines.append(f"mean_state_top1={np.mean([r['acc'] for r in state_metrics]):.4f}")
    lines.append(f"min_state_top1={np.min([r['acc'] for r in state_metrics]):.4f}")
    lines.append(f"mean_true_posterior_mass={np.mean([r['mean_mass'] for r in state_metrics]):.4f}")

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

