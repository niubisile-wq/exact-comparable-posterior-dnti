import csv
import json
from pathlib import Path


ROOT = Path(r"<LOCAL_WORKSPACE>\digital-twin-dataset")
TOPO_ROOT = ROOT / "sample_dataset" / "topology"
NETWORK_FILE = TOPO_ROOT / "network_files" / "circuit3" / "2023-08-01T00h00m00.000000s.json"
NETFILE_CHANGES = TOPO_ROOT / "metadata" / "netfile_changes.csv"
PARAMETER_CHANGES = TOPO_ROOT / "metadata" / "parameter_changes.csv"
STATUS_DIR = TOPO_ROOT / "parameter_timeseries"
OUT_FILE = Path(r"<LOCAL_WORKSPACE>\socal28_qualitative_validation.txt")


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    network = json.loads(NETWORK_FILE.read_text(encoding="utf-8"))
    netfile_rows = read_csv_rows(NETFILE_CHANGES)
    parameter_rows = read_csv_rows(PARAMETER_CHANGES)
    status_files = sorted(STATUS_DIR.glob("*_status.csv"))

    tie_candidates = []
    for swmp in network.get("SwitchMultiPosition", []):
        for terminal in swmp.get("tbus", []):
            if terminal.get("status") == "NO" or terminal.get("nominal_status") == "NO":
                tie_candidates.append(f"{swmp['fbus']}->{terminal['name']}")
                break

    status_event_rows = 0
    change_rows = 0
    distinct_status_timestamps = set()
    changed_devices = 0
    for status_file in status_files:
        rows = read_csv_rows(status_file)
        status_event_rows += len(rows)
        distinct_status_timestamps.update(row.get("t", "") for row in rows if row.get("t"))
        if len({row.get("str", "") for row in rows}) > 1:
            changed_devices += 1
            for prev, curr in zip(rows, rows[1:]):
                if prev.get("str") != curr.get("str"):
                    change_rows += 1

    lines = [
        "SoCal 28-bus digital-twin qualitative validation",
        "dataset_source=SoCal 28-bus digital-twin sample_dataset",
        f"sample_network_file={NETWORK_FILE.name}",
        f"PhysicalAssetBus={len(network.get('Bus', []))}",
        f"Line={len(network.get('Line', []))}",
        f"Switch={len(network.get('Switch', []))}",
        f"SwitchMultiPosition={len(network.get('SwitchMultiPosition', []))}",
        f"CB={len(network.get('CB', []))}",
        f"Transformer={len(network.get('Transformer', []))}",
        f"TieCandidates={len(tie_candidates)}",
        f"SampleTies={'; '.join(tie_candidates[:8])}",
        f"netfile_change_events={len(netfile_rows)}",
        f"parameter_change_events={len(parameter_rows)}",
        f"status_csv_files={len(status_files)}",
        f"status_event_rows={status_event_rows}",
        f"distinct_status_timestamps={len(distinct_status_timestamps)}",
        f"devices_with_status_changes={changed_devices}",
        f"status_transition_events={change_rows}",
    ]

    if netfile_rows:
        lines.append(f"first_netfile_timestamp={netfile_rows[0].get('t', '')}")
        lines.append(f"last_netfile_timestamp={netfile_rows[-1].get('t', '')}")
    if parameter_rows:
        lines.append(f"first_parameter_timestamp={parameter_rows[0].get('t', '')}")
        lines.append(f"last_parameter_timestamp={parameter_rows[-1].get('t', '')}")

    lines.extend(
        [
            "boundary=Current local sample supports qualitative real-network topology/metadata validation only.",
            "boundary=The present workspace sample does not provide a ready-made exact-comparable posterior benchmark on this real feeder, so no new top-1/KL/ECE claim is made here.",
        ]
    )

    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_FILE)


if __name__ == "__main__":
    main()
