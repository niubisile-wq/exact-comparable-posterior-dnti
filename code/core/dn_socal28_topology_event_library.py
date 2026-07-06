import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path


STATUS_DIR = Path.home() / "Desktop" / "配电网实验_临时" / "digital-twin-dataset" / "sample_dataset" / "topology" / "parameter_timeseries"
OUT_FILE = Path.home() / "Desktop" / "配电网实验_临时" / "socal28_topology_event_library.txt"


def read_status_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_ts(ts: str):
    return datetime.fromisoformat(ts)


def main():
    device_rows = {}
    timeline = defaultdict(dict)
    for status_file in sorted(STATUS_DIR.glob("*_status.csv")):
        rows = read_status_rows(status_file)
        device = status_file.stem.replace("_status", "")
        device_rows[device] = rows
        for row in rows:
            timeline[row["t"]][device] = row["str"]

    devices = sorted(device_rows)
    current = {}
    snapshots = []
    unique_states = {}
    for timestamp in sorted(timeline):
        for device, state in timeline[timestamp].items():
            current[device] = state
        snapshot = tuple(current.get(device, "UNK") for device in devices)
        snapshots.append((timestamp, snapshot))
        unique_states.setdefault(snapshot, []).append(timestamp)

    transition_count = 0
    transition_pairs = defaultdict(int)
    device_flip_counts = defaultdict(int)
    dwell_seconds = defaultdict(float)
    for i, ((prev_ts, prev), (curr_ts, curr)) in enumerate(zip(snapshots, snapshots[1:])):
        if prev != curr:
            transition_count += 1
            transition_pairs[(prev, curr)] += 1
            for device, p, c in zip(devices, prev, curr):
                if p != c:
                    device_flip_counts[device] += 1
        dwell_seconds[prev] += (parse_ts(curr_ts) - parse_ts(prev_ts)).total_seconds()

    device_row_counts = {device: len(rows) for device, rows in device_rows.items()}

    lines = [
        "SoCal 28-bus topology event library summary",
        f"status_device_count={len(devices)}",
        f"timeline_snapshots={len(snapshots)}",
        f"unique_joint_topology_states={len(unique_states)}",
        f"state_transition_steps={transition_count}",
    ]

    if snapshots:
        lines.append(f"first_snapshot={snapshots[0][0]}")
        lines.append(f"last_snapshot={snapshots[-1][0]}")

    for idx, (state, timestamps) in enumerate(sorted(unique_states.items(), key=lambda item: len(item[1]), reverse=True)[:5], start=1):
        nc = sum(1 for x in state if x == "NC")
        no = sum(1 for x in state if x == "NO")
        unk = sum(1 for x in state if x == "UNK")
        lines.append(
            f"top_state_{idx}=count:{len(timestamps)} first:{timestamps[0]} last:{timestamps[-1]} NC:{nc} NO:{no} UNK:{unk}"
        )

    if dwell_seconds:
        lines.append("longest_state_dwells_hours")
        top_dwells = sorted(dwell_seconds.items(), key=lambda item: item[1], reverse=True)[:5]
        for idx, (state, secs) in enumerate(top_dwells, start=1):
            nc = sum(1 for x in state if x == "NC")
            no = sum(1 for x in state if x == "NO")
            lines.append(f"dwell_{idx}=hours:{secs / 3600.0:.2f} NC:{nc} NO:{no}")

    if transition_pairs:
        lines.append("top_transition_edges")
        ranked = sorted(transition_pairs.items(), key=lambda item: item[1], reverse=True)[:5]
        for idx, ((prev, curr), count) in enumerate(ranked, start=1):
            prev_nc = sum(1 for x in prev if x == "NC")
            curr_nc = sum(1 for x in curr if x == "NC")
            diff = sum(1 for p, c in zip(prev, curr) if p != c)
            lines.append(
                f"transition_{idx}=count:{count} changed_devices:{diff} prev_NC:{prev_nc} curr_NC:{curr_nc}"
            )

    if device_flip_counts:
        lines.append("device_flip_frequency")
        ranked_devices = sorted(device_flip_counts.items(), key=lambda item: item[1], reverse=True)
        for idx, (device, count) in enumerate(ranked_devices[:8], start=1):
            rows = device_row_counts.get(device, 0)
            lines.append(f"flip_{idx}=device:{device} changes:{count} rows:{rows}")

    lines.append("boundary=This library is derived only from the local topology status sample and is not yet paired with synchronized measurement streams.")
    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_FILE)


if __name__ == "__main__":
    main()
