# -*- coding: utf-8 -*-
"""Step 7 main-figure and supplementary-material freeze.

Consumes Step 6 frozen tables plus raw figure data. It does not rerun models.
"""
import csv
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SAVE_DIR
PALETTE = {
    "navy": "#355C7D",
    "blue": "#4F81BD",
    "green": "#5B8E2D",
    "gold": "#C48A1D",
    "red": "#C65A46",
    "gray": "#7A7A7A",
}


def pjoin(name):
    return os.path.join(SAVE_DIR, name)


def pct_to_float(s):
    return float(s.replace("%", "").split("+/-")[0]) / 100.0


def parse_ip4_v5():
    txt = open(pjoin("ip4_hk_result_v5.txt"), "r", encoding="utf-8", errors="replace").read().splitlines()
    out = {}
    current = None
    for line in txt:
        m = re.match(r"IEEE\s+(33|69|119)-bus", line)
        if m:
            current = f"{m.group(1)}-bus"
            out[current] = {"K": [], "H": [], "top1": []}
            continue
        if current:
            row = re.match(r"\s*(\d+)\s+([0-9.]+)\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)", line)
            if row:
                out[current]["K"].append(int(row.group(1)))
                out[current]["H"].append(float(row.group(2)))
                out[current]["top1"].append(float(row.group(3)))
    return out


def parse_ip4_300():
    vals = {"K": [], "H": []}
    with open(pjoin("ip4_300bus_result.txt"), "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) == 3 and p[0].isdigit():
                vals["K"].append(int(p[0]))
                vals["H"].append(float(p[1]))
    return vals


def parse_table_section(section_name):
    lines = open(pjoin("main_tables_final.txt"), "r", encoding="utf-8").read().splitlines()
    out = []
    capture = False
    header = None
    for line in lines:
        if line.strip() == section_name:
            capture = True
            header = None
            continue
        if capture and line.startswith("Table ") and line.strip() != section_name:
            break
        if capture and "," in line:
            if header is None:
                header = line.split(",")
            else:
                vals = line.split(",")
                if len(vals) == len(header):
                    out.append(dict(zip(header, vals)))
    return out


def read_missing_curve():
    rows = []
    with open(pjoin("missing_curve_33bus_result.txt"), "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) == 7 and p[0].replace(".", "", 1).isdigit():
                rows.append({
                    "miss": float(p[0]),
                    "rob": float(p[1]),
                    "rob_std": float(p[2]),
                    "naive": float(p[3]),
                    "delta": float(p[4]),
                })
    return rows


def read_outage_summary():
    with open(pjoin("outage_33bus_result.txt"), "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("summary,"):
                p = line.strip().split(",")
                return {"rob": float(p[1]), "naive": float(p[2]), "delta": float(p[3])}
    raise RuntimeError("No outage summary found")


def read_noise_sensitivity():
    rows = []
    with open(pjoin("noise_sensitivity_33bus_result.txt"), "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) == 7 and p[0].replace(".", "", 1).isdigit():
                rows.append({"sigma": float(p[0]), "rob": float(p[1]), "naive": float(p[3]), "delta": float(p[4])})
    return rows


def read_reliability():
    rows = []
    with open(pjoin("posterior_reliability_bins.csv"), "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["n"] and int(r["n"]) > 0:
                rows.append(r)
    return rows


def read_multimodal():
    rows = []
    with open(pjoin("posterior_multimodal_case.txt"), "r", encoding="utf-8") as f:
        capture = False
        for line in f:
            if line.startswith("topology,"):
                capture = True
                continue
            if capture and "," in line:
                p = line.strip().split(",")
                if len(p) == 4:
                    rows.append({"topology": p[0], "reference": float(p[1]), "raw": float(p[2]), "cal": float(p[3])})
    return rows[:2]


def setup_style():
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 240,
        "axes.facecolor": "#fbfbfa",
        "figure.facecolor": "white",
    })


def fig1_identifiability():
    data = parse_ip4_v5()
    d300 = parse_ip4_300()
    fig, ax = plt.subplots(figsize=(5.8, 3.6), constrained_layout=True)
    colors = {
        "33-bus": PALETTE["navy"],
        "69-bus": PALETTE["green"],
        "119-bus": PALETTE["gold"],
        "300-bus": PALETTE["red"],
    }
    for name, d in data.items():
        ax.plot(d["K"], d["H"], marker="o", linewidth=1.8, markersize=3.5, label=name, color=colors[name])
    ax.plot(d300["K"], d300["H"], marker="s", linewidth=1.8, markersize=3.5, label="300-bus synthetic", color=colors["300-bus"])
    ax.set_xlabel("Installed sensors K")
    ax.set_ylabel("Posterior entropy H(K)")
    ax.set_title("Identifiability improves with measurements")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    out = pjoin("fig1_identifiability.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig2_ip1_speed():
    rows = parse_table_section("Table A. IP1 exact-comparable posterior inference")
    labels = [r["system"] for r in rows]
    exact = [pct_to_float(r["exact_top1_mean"]) for r in rows]
    nre = [pct_to_float(r["nre_top1_mean"]) for r in rows]
    speed = [float(r["speedup"].replace("x", "")) for r in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
    width = 0.36
    axes[0].bar(x - width / 2, exact, width, label="Exact", color=PALETTE["gray"])
    axes[0].bar(x + width / 2, nre, width, label="NRE", color=PALETTE["navy"])
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 0.8)
    axes[0].set_ylabel("Top-1 accuracy")
    axes[0].set_title("Posterior inference accuracy")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].bar(labels, speed, color=PALETTE["green"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Speedup vs exact")
    axes[1].set_title("Amortized online inference")
    axes[1].grid(axis="y", alpha=0.25)
    out = pjoin("fig2_ip1_accuracy_speed.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig3_robustness():
    miss = read_missing_curve()
    outage = read_outage_summary()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
    xs = [r["miss"] * 100 for r in miss]
    rob = [r["rob"] for r in miss]
    naive = [r["naive"] for r in miss]
    err = [r["rob_std"] for r in miss]
    axes[0].errorbar(xs, rob, yerr=err, marker="o", linewidth=1.8, capsize=3, label="IP-C robust", color=PALETTE["navy"])
    axes[0].plot(xs, naive, marker="s", linewidth=1.8, label="Naive", color=PALETTE["red"])
    axes[0].set_xlabel("Missing measurements (%)")
    axes[0].set_ylabel("Top-1 accuracy")
    axes[0].set_ylim(0, 0.78)
    axes[0].set_title("Missing-rate boundary")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].bar(["Naive", "IP-C"], [outage["naive"], outage["rob"]], color=[PALETTE["red"], PALETTE["navy"]])
    axes[1].set_ylim(0, 0.75)
    axes[1].set_ylabel("Top-1 accuracy")
    axes[1].set_title("Outage-induced missing")
    axes[1].grid(axis="y", alpha=0.25)
    out = pjoin("fig3_robustness.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig4_boed():
    rows = parse_table_section("Table D. IP-A / BOED sensor placement")
    systems = ["33-bus", "69-bus", "119-bus"]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1), constrained_layout=True, sharey=True)
    palette = {
        "Random": PALETTE["gray"],
        "GreedyLoop": PALETTE["gold"],
        "MVG": PALETTE["green"],
        "BOED": PALETTE["navy"],
        "AdaptiveMVG": PALETTE["green"],
        "AdaptiveFisher": PALETTE["gold"],
        "AdaptiveBOED": PALETTE["navy"],
    }
    for ax, system in zip(axes, systems):
        sub = [r for r in rows if r["system"] == system]
        methods = [r["policy"] for r in sub]
        vals = [pct_to_float(r["K7_top1"]) for r in sub]
        ax.bar(np.arange(len(methods)), vals, color=[palette.get(m, "#7f7f7f") for m in methods])
        ax.set_title(f"{system} K=7")
        ax.set_xticks(np.arange(len(methods)), methods, rotation=25, ha="right")
        ax.set_ylim(0, 0.8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Top-1 accuracy")
    out = pjoin("fig4_boed_sensor_placement.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig5_scalability_copy():
    # Regenerate through the existing script to keep its source of truth.
    import fig_scalability
    fig_scalability.main()
    src = pjoin("fig_scalability.png")
    dst = pjoin("fig5_scalability.png")
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        fdst.write(fsrc.read())
    return dst


def fig6_posterior_quality():
    rel = read_reliability()
    multi = read_multimodal()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color=PALETTE["gray"], linewidth=1.0)
    for method, color in [("Reference", PALETTE["gray"]), ("Raw_NRE", PALETTE["gold"]), ("Calibrated_NRE", PALETTE["navy"])]:
        sub = [r for r in rel if r["method"] == method]
        conf = [float(r["confidence"]) for r in sub]
        acc = [float(r["accuracy"]) for r in sub]
        axes[0].plot(conf, acc, marker="o", linewidth=1.5, markersize=3, label=method.replace("_", " "), color=color)
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Empirical accuracy")
    axes[0].set_title("Reliability")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    labels = [r["topology"] for r in multi]
    x = np.arange(len(labels))
    width = 0.36
    axes[1].bar(x - width / 2, [r["reference"] for r in multi], width, label="Exact", color=PALETTE["gray"])
    axes[1].bar(x + width / 2, [r["cal"] for r in multi], width, label="Calibrated NRE", color=PALETTE["navy"])
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("Topology")
    axes[1].set_ylabel("Posterior probability")
    axes[1].set_title("Ambiguous case")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    out = pjoin("fig6_posterior_quality.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def write_manifest(outputs):
    lines = []
    lines.append("Step 7 figure freeze manifest")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Status: artifacts generated; advisor review pending.")
    lines.append("")
    lines.append("Main figures, max 6")
    entries = [
        ("Fig. 1", "fig1_identifiability.png", "ip4_hk_result_v5.txt; ip4_300bus_result.txt", "H(K) identifiability across 33/69/119/300-bus."),
        ("Fig. 2", "fig2_ip1_accuracy_speed.png", "main_tables_final.txt Table A", "IP1 exact-vs-NRE accuracy and online speedup."),
        ("Fig. 3", "fig3_robustness.png", "missing_curve_33bus_result.txt; outage_33bus_result.txt", "Missing-rate and outage-induced missing robustness."),
        ("Fig. 4", "fig4_boed_sensor_placement.png", "main_tables_final.txt Table D", "BOED/OED sensor placement comparison."),
        ("Fig. 5", "fig5_scalability.png", "main_tables_final.txt Table F; fig_scalability.py; scalability_result.txt; ip4_300bus_result.txt; ip1_300bus_result.txt; ipc_300bus_result.txt", "Synthetic 300-bus scalability evidence."),
        ("Fig. 6", "fig6_posterior_quality.png", "posterior_reliability_bins.csv; posterior_multimodal_case.txt", "Posterior reliability and ambiguous-case posterior."),
    ]
    for label, fname, source, note in entries:
        if fname not in outputs:
            raise RuntimeError(f"Missing output {fname}")
        size = os.path.getsize(pjoin(fname))
        if size < 1000:
            raise RuntimeError(f"Output too small: {fname}")
        lines.append(f"{label}: {fname}")
        lines.append(f"  source: {source}")
        lines.append(f"  note: {note}")
    lines.append("")
    lines.append("Supplementary figure candidates")
    lines.append("  fig_noise_sensitivity.png / fig_noise_sensitivity.py: noise sensitivity at 30% missing.")
    lines.append("  fig_reliability_diagram.png and fig_posterior_case.png: component views supporting Fig. 6.")
    lines.append("")
    lines.append("Boundary reminders")
    lines.append("  300-bus is synthetic fixed-deployment mid-scale scalability only.")
    lines.append("  Outage plot means outage-induced missing measurements, not full fault diagnosis.")
    lines.append("  BOED is not claimed to universally dominate MVG.")
    open(pjoin("figure_freeze_manifest.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    supp = []
    supp.append("Step 7 supplementary-material manifest")
    supp.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    supp.append("Status: synchronized with main figure freeze; advisor review pending.")
    supp.append("")
    supp.append("Supplementary figures")
    supp.append("  S1: fig_noise_sensitivity.png")
    supp.append("      source: noise_sensitivity_33bus_result.txt")
    supp.append("      role: voltage-noise sensitivity under 30% missing measurements.")
    supp.append("  S2: fig_reliability_diagram.png")
    supp.append("      source: posterior_reliability_bins.csv")
    supp.append("      role: detailed reliability diagram supporting main Fig. 6.")
    supp.append("  S3: fig_posterior_case.png")
    supp.append("      source: posterior_multimodal_case.txt")
    supp.append("      role: detailed multimodal posterior case supporting main Fig. 6.")
    supp.append("")
    supp.append("Supplementary tables")
    supp.append("  ST1: stats_final.txt")
    supp.append("      role: full Wilcoxon/BH-FDR/Holm/Bonferroni details, including supplemental 119-bus IP-C tests.")
    supp.append("  ST2: main_tables_final.txt")
    supp.append("      role: frozen numeric source for main tables and figure panels.")
    supp.append("")
    supp.append("Boundary")
    supp.append("  Supplementary materials do not add new claims beyond the frozen Step 6/7 evidence.")
    open(pjoin("supplementary_materials_manifest.txt"), "w", encoding="utf-8").write("\n".join(supp) + "\n")


def main():
    setup_style()
    paths = [
        fig1_identifiability(),
        fig2_ip1_speed(),
        fig3_robustness(),
        fig4_boed(),
        fig5_scalability_copy(),
        fig6_posterior_quality(),
    ]
    outputs = {os.path.basename(p) for p in paths}
    write_manifest(outputs)
    print("Generated Step 7 figures:")
    for p in paths:
        print(f"  {os.path.basename(p)} {os.path.getsize(p)} bytes")
    print("Generated figure_freeze_manifest.txt")
    print("Generated supplementary_materials_manifest.txt")


if __name__ == "__main__":
    main()
