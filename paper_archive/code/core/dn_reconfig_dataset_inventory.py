import re
from pathlib import Path


ROOT = Path(r"<LOCAL_WORKSPACE>\ElectricalSystemsDataForReconfiguration")
OUT = Path(r"<LOCAL_WORKSPACE>\reconfig_dataset_inventory.txt")


BUS_ROW = re.compile(r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$")
BRANCH_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s*$")
VNOM_ROW = re.compile(r"Vnominal\s*=?\s*([0-9.]+)")
SLACK_ROW = re.compile(r"(BusSE|Barra_SE)\s*[:=]?\s*(\d+)")


def parse_system(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    buses = []
    branches = []
    vnom = None
    slack = None
    in_branch = False

    for line in text:
        if vnom is None:
            m = VNOM_ROW.search(line)
            if m:
                vnom = float(m.group(1))
        if slack is None:
            m = SLACK_ROW.search(line)
            if m:
                slack = int(m.group(2))

        lower = line.lower()
        if (("send" in lower or "env" in lower) and ("recv" in lower or "rec" in lower) and "line" in lower):
            in_branch = True
            continue

        if in_branch:
            m = BRANCH_ROW.match(line)
            if m:
                send, recv, line_id, r, x = m.groups()
                branches.append((int(send), int(recv), int(line_id), float(r), float(x)))
            continue

        m = BUS_ROW.match(line)
        if m:
            bus, pd, qd, qc = m.groups()
            buses.append((int(bus), float(pd), float(qd), float(qc)))

    bus_ids = {row[0] for row in buses}
    n_bus = len(bus_ids)
    n_branch = len(branches)
    n_load_bus = sum(1 for _, pd, qd, _ in buses if pd > 0.1001 or qd > 0.0001)
    tie_like = max(0, n_branch - max(0, n_bus - 1))
    total_pd = sum(pd for _, pd, _, _ in buses)
    total_qd = sum(qd for _, _, qd, _ in buses)
    max_branch_id = max((row[2] for row in branches), default=0)

    return {
        "system": path.stem,
        "vnom_kv": vnom,
        "slack_bus": slack,
        "n_bus": n_bus,
        "n_load_bus": n_load_bus,
        "n_branch": n_branch,
        "estimated_tie_like_branches": tie_like,
        "max_branch_id": max_branch_id,
        "total_pd_kw": total_pd,
        "total_qd_kvar": total_qd,
    }


def main():
    rows = []
    for path in sorted(ROOT.glob("SystemData_*.txt")):
        rows.append(parse_system(path))

    lines = [
        "Reconfiguration-dataset inventory",
        f"root={ROOT}",
        "",
        "system,vnom_kv,slack_bus,n_bus,n_load_bus,n_branch,estimated_tie_like_branches,max_branch_id,total_pd_kw,total_qd_kvar",
    ]
    for row in rows:
        lines.append(
            f"{row['system']},{row['vnom_kv']},{row['slack_bus']},{row['n_bus']},"
            f"{row['n_load_bus']},{row['n_branch']},{row['estimated_tie_like_branches']},"
            f"{row['max_branch_id']},{row['total_pd_kw']:.3f},{row['total_qd_kvar']:.3f}"
        )

    lines.extend(
        [
            "",
            "Interpretation",
            "- estimated_tie_like_branches = n_branch - (n_bus - 1)",
            "- this is a structural upper-bound proxy for reconfiguration candidates, not a full feasibility certification",
            "- these text assets are balanced reconfiguration benchmarks, so they support full-scale stress coverage rather than the unbalanced-three-phase main gap",
        ]
    )
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
