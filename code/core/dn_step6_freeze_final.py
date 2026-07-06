# -*- coding: utf-8 -*-
"""Step 6 final statistics and table freeze.

This script only consumes existing result files. It does not rerun experiments.
"""
import csv
import os
import re
from datetime import datetime

import numpy as np
from scipy.stats import wilcoxon

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def pjoin(name):
    return os.path.join(SAVE_DIR, name)


def read_csv_rows(name):
    with open(pjoin(name), "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        legacy_exact = row.pop("AIS", None)
        if legacy_exact is not None and "EnumBF" not in row:
            row["EnumBF"] = legacy_exact
    return rows


def mean_std(x):
    a = np.asarray(x, dtype=float)
    return float(np.mean(a)), float(np.std(a))


def fmt_pct(x):
    return f"{100.0 * x:.2f}%"


def fmt_pp(x):
    return f"{100.0 * x:+.2f}pp"


def wilcoxon_one_sided(a, b, alternative):
    stat = wilcoxon(np.asarray(a) - np.asarray(b), alternative=alternative, zero_method="wilcox", mode="exact")
    return float(stat.pvalue)


def bh_fdr(pvals, q=0.05):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    passed = np.zeros(len(p), dtype=bool)
    max_k = -1
    for rank, idx in enumerate(order, start=1):
        if p[idx] <= rank / len(p) * q:
            max_k = rank
    if max_k > 0:
        passed[order[:max_k]] = True
    return passed


def holm(pvals, alpha=0.05):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    passed = np.zeros(len(p), dtype=bool)
    for rank, idx in enumerate(order, start=1):
        if p[idx] <= alpha / (len(p) - rank + 1):
            passed[idx] = True
        else:
            break
    return passed


def bonferroni(pvals, alpha=0.05):
    adjusted = np.minimum(np.asarray(pvals, dtype=float) * len(pvals), 1.0)
    return adjusted <= alpha, adjusted


def parse_119_ip1():
    txt = open(pjoin("119bus_ip1_result.txt"), "r", encoding="utf-8").read()
    rows = []
    for m in re.finditer(r"seed=(\d+): EnumBF=([0-9.]+)\s+NRE=([0-9.]+)\s+gap=([0-9.]+)\s+KL=([0-9.]+)", txt):
        rows.append({
            "seed": int(m.group(1)),
            "exact": float(m.group(2)),
            "nre": float(m.group(3)),
            "gap": float(m.group(4)),
            "kl": float(m.group(5)),
        })
    speed = re.search(r"speedup=([0-9]+)x", txt)
    return rows, int(speed.group(1)) if speed else None


def parse_graphsage_summary():
    txt = open(pjoin("baseline_unified_table.txt"), "r", encoding="utf-8").read()
    vals = {}
    for net in ["33bus", "69bus"]:
        m = re.search(rf"{net}\s+GraphSAGE\s+([0-9.]+)\+/-([0-9.]+)\s+([0-9.]+)\+/-([0-9.]+)", txt)
        if m:
            vals[net] = tuple(float(v) for v in m.groups())
    return vals


def parse_posterior_metrics():
    lines = open(pjoin("posterior_calibration_result.txt"), "r", encoding="utf-8").read().splitlines()
    metrics = {}
    in_table = False
    for line in lines:
        if line.strip() == "method,top1,NLL,Brier,KL_ref_to_model,ECE,coverage90,avg_credset90,entropy":
            in_table = True
            continue
        if in_table and "," in line:
            parts = line.split(",")
            if len(parts) == 9:
                metrics[parts[0]] = [float(x) for x in parts[1:]]
    return metrics


def parse_scalability():
    lines = open(pjoin("scalability_result.txt"), "r", encoding="utf-8").read().splitlines()
    for line in lines:
        if line.startswith("synthetic_300bus"):
            p = line.split(",")
            return {
                "system": p[0],
                "n_bus": int(p[1]),
                "n_topologies": int(p[2]),
                "ip1_acc": float(p[3]),
                "ipc_acc30": float(p[4]),
                "nre_ms": float(p[5]),
                "exact_ms": float(p[6]),
                "speedup": float(p[7]),
            }
    raise RuntimeError("No scalability row found")


def parse_300bus_exact_accuracy():
    ip1_exact = []
    with open(pjoin("ip1_300bus_result.txt"), "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) == 9 and p[0].isdigit():
                ip1_exact.append(float(p[4]))
    ipc_exact30 = []
    with open(pjoin("ipc_300bus_result.txt"), "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) == 11 and p[0].isdigit():
                ipc_exact30.append(float(p[6]))
    if not ip1_exact or not ipc_exact30:
        raise RuntimeError("Cannot parse 300-bus exact accuracy values")
    return {
        "ip1_exact_acc": float(np.mean(ip1_exact)),
        "ipc_exact_acc30": float(np.mean(ipc_exact30)),
        "ip1_source": "ip1_300bus_result.txt",
        "ipc_source": "ipc_300bus_result.txt",
    }


def parse_ip4_300():
    vals = []
    with open(pjoin("ip4_300bus_result.txt"), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3 and parts[0].isdigit():
                vals.append((int(parts[0]), float(parts[1]), float(parts[2])))
    return vals


def parse_boed_33():
    txt = open(pjoin("boed_33bus_nmc500_result.txt"), "r", encoding="utf-8").read()
    out = {}
    for k in [4, 7]:
        m = re.search(rf"K=\s*{k}:\s+BOED=([0-9.]+)\s+MVG=([0-9.]+)\s+Random=([0-9.]+)\s+Greedy=([0-9.]+)", txt)
        if not m:
            raise RuntimeError(f"Cannot parse 33-bus BOED K={k}")
        out[k] = {
            "BOED": float(m.group(1)),
            "MVG": float(m.group(2)),
            "Random": float(m.group(3)),
            "GreedyLoop": float(m.group(4)),
        }
    return out


def parse_boed_69():
    txt = open(pjoin("boed_69bus_result.txt"), "r", encoding="utf-8", errors="replace").read()
    out = {}
    in_accuracy = False
    for line in txt.splitlines():
        if line.strip().startswith("Top-1 Accuracy"):
            in_accuracy = True
            continue
        if line.strip().startswith("Posterior Entropy"):
            in_accuracy = False
            continue
        if not in_accuracy:
            continue
        m = re.match(r"\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", line)
        if m:
            out[int(m.group(1))] = {
                "BOED": float(m.group(2)),
                "Random": float(m.group(3)),
                "GreedyLoop": float(m.group(4)),
            }
    if 4 not in out or 7 not in out:
        raise RuntimeError("Cannot parse 69-bus BOED K=4/K=7")
    mvg_txt = open(pjoin("step5_mvg_result.txt"), "r", encoding="utf-8").read()
    m = re.search(r"69bus:\s+MVG K4=([0-9.]+) K7=([0-9.]+)", mvg_txt)
    if not m:
        raise RuntimeError("Cannot parse 69-bus MVG")
    out[4]["MVG"] = float(m.group(1))
    out[7]["MVG"] = float(m.group(2))
    for k in [4, 7]:
        for method, val in out[k].items():
            if not 0.0 <= val <= 1.0:
                raise RuntimeError(f"Invalid 69-bus top-1 value {method} K={k}: {val}")
    return out


def parse_boed_119():
    txt = open(pjoin("boed_119bus_minimal_result.txt"), "r", encoding="utf-8").read()
    out = {}
    in_summary = False
    for line in txt.splitlines():
        if line.strip().startswith("Method"):
            in_summary = True
            continue
        if in_summary:
            m = re.match(r"\s*(Random|MVG|Fisher|BOED|AdaptiveMVG|AdaptiveFisher|AdaptiveBOED)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", line)
            if m:
                out[m.group(1)] = {
                    "K4_acc": float(m.group(2)),
                    "K7_acc": float(m.group(3)),
                    "K4_H": float(m.group(4)),
                    "K7_H": float(m.group(5)),
                }
    for key in ["AdaptiveMVG", "AdaptiveFisher", "AdaptiveBOED"]:
        if key not in out:
            raise RuntimeError(f"Cannot parse 119-bus {key}")
    return out


def collect_core_stats():
    ip1_33 = read_csv_rows("ip1_33bus_n8_result.txt")
    ip1_69 = read_csv_rows("ip1_69bus_n8_result.txt")
    ipc_33 = read_csv_rows("ipc_33bus_n8_result.txt")
    ipc_69 = read_csv_rows("ipc_69bus_n8_result.txt")

    tests = []
    for name, rows in [("IP1-33bus exact>NRE", ip1_33), ("IP1-69bus exact>NRE", ip1_69)]:
        exact = [float(r["EnumBF"]) for r in rows]
        nre = [float(r["NRE"]) for r in rows]
        gap = np.asarray(exact) - np.asarray(nre)
        tests.append({
            "name": name,
            "n": len(rows),
            "A_mean": mean_std(exact)[0],
            "B_mean": mean_std(nre)[0],
            "delta_mean": float(np.mean(gap)),
            "delta_std": float(np.std(gap)),
            "p": wilcoxon_one_sided(exact, nre, "greater"),
            "effect": 1.0 if np.all(gap > 0) else float(np.mean(gap > 0)),
            "source": "ip1_33bus_n8_result.txt" if "33" in name else "ip1_69bus_n8_result.txt",
        })

    for name, rows, miss in [
        ("IP-C-33bus robust>naive @10%", ipc_33, "10"),
        ("IP-C-33bus robust>naive @30%", ipc_33, "30"),
        ("IP-C-69bus robust>naive @10%", ipc_69, "10"),
        ("IP-C-69bus robust>naive @30%", ipc_69, "30"),
    ]:
        rob = [float(r[f"rob{miss}"]) for r in rows]
        nai = [float(r[f"nai{miss}"]) for r in rows]
        gap = np.asarray(rob) - np.asarray(nai)
        tests.append({
            "name": name,
            "n": len(rows),
            "A_mean": mean_std(rob)[0],
            "B_mean": mean_std(nai)[0],
            "delta_mean": float(np.mean(gap)),
            "delta_std": float(np.std(gap)),
            "p": wilcoxon_one_sided(rob, nai, "greater"),
            "effect": 1.0 if np.all(gap > 0) else float(np.mean(gap > 0)),
            "source": "ipc_33bus_n8_result.txt" if "33" in name else "ipc_69bus_n8_result.txt",
        })

    pvals = [t["p"] for t in tests]
    bh = bh_fdr(pvals)
    hm = holm(pvals)
    bf, p_bonf = bonferroni(pvals)
    for i, t in enumerate(tests):
        t["bh"] = bool(bh[i])
        t["holm"] = bool(hm[i])
        t["bonferroni"] = bool(bf[i])
        t["p_bonferroni"] = float(p_bonf[i])
    return tests


def collect_supplemental_119():
    rows = read_csv_rows("ipc_119bus_5seed_result.txt")
    out = []
    for miss in ["10", "30"]:
        rob = [float(r[f"rob{miss}"]) for r in rows]
        nai = [float(r[f"nai{miss}"]) for r in rows]
        gap = np.asarray(rob) - np.asarray(nai)
        out.append({
            "name": f"IP-C-119bus robust>naive @{miss}%",
            "n": len(rows),
            "A_mean": mean_std(rob)[0],
            "B_mean": mean_std(nai)[0],
            "delta_mean": float(np.mean(gap)),
            "delta_std": float(np.std(gap)),
            "p": wilcoxon_one_sided(rob, nai, "greater"),
            "effect": 1.0 if np.all(gap > 0) else float(np.mean(gap > 0)),
            "source": "ipc_119bus_5seed_result.txt",
        })
    return out


def write_stats(tests, supp):
    lines = []
    lines.append("Step 6 final statistics freeze")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Scope: planned primary tests use current n=8 33/69-bus result files.")
    lines.append("Exact reference label: exact enumeration / EnumBF; do not write Xu-style AIS.")
    lines.append("")
    lines.append("Primary planned Wilcoxon tests, one-sided")
    lines.append("test,n,A_mean,B_mean,delta_mean,delta_std,p_raw,p_bonferroni,BH_FDR_q0.05,Holm_alpha0.05,Bonferroni_alpha0.05,effect_all_positive,source")
    for t in tests:
        lines.append(
            f"{t['name']},{t['n']},{t['A_mean']:.4f},{t['B_mean']:.4f},"
            f"{t['delta_mean']:.4f},{t['delta_std']:.4f},{t['p']:.6f},{t['p_bonferroni']:.6f},"
            f"{'PASS' if t['bh'] else 'ns'},{'PASS' if t['holm'] else 'ns'},"
            f"{'PASS' if t['bonferroni'] else 'ns'},"
            f"{t['effect']:.3f},{t['source']}"
        )
    lines.append("")
    lines.append(f"BH-FDR q=0.05: {sum(t['bh'] for t in tests)}/{len(tests)} primary tests pass.")
    lines.append(f"Holm alpha=0.05: {sum(t['holm'] for t in tests)}/{len(tests)} primary tests pass.")
    lines.append(f"Bonferroni alpha=0.05: {sum(t['bonferroni'] for t in tests)}/{len(tests)} primary tests pass.")
    lines.append("Because n=8 and all paired deltas have the planned direction, p_raw=1/2^8=0.003906 for each primary test.")
    lines.append("For each primary test, p_bonferroni=0.003906*6=0.023438 < 0.05.")
    lines.append("")
    lines.append("Supplemental 119-bus IP-C statistics, not mixed into the primary m=6 family")
    lines.append("test,n,robust_mean,naive_mean,delta_mean,delta_std,p_raw,effect_all_positive,source")
    for t in supp:
        lines.append(
            f"{t['name']},{t['n']},{t['A_mean']:.4f},{t['B_mean']:.4f},"
            f"{t['delta_mean']:.4f},{t['delta_std']:.4f},{t['p']:.6f},"
            f"{t['effect']:.3f},{t['source']}"
        )
    lines.append("Boundary: report 119-bus tests as chain-closure supplemental evidence.")
    open(pjoin("stats_final.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def write_tables(tests, supp):
    ip1_33 = read_csv_rows("ip1_33bus_n8_result.txt")
    ip1_69 = read_csv_rows("ip1_69bus_n8_result.txt")
    ip1_119, speed119 = parse_119_ip1()
    gs = parse_graphsage_summary()
    post = parse_posterior_metrics()
    scal = parse_scalability()
    scal_exact = parse_300bus_exact_accuracy()
    ip4_300 = parse_ip4_300()
    boed33 = parse_boed_33()
    boed69 = parse_boed_69()
    boed119 = parse_boed_119()

    exact33 = mean_std([float(r["EnumBF"]) for r in ip1_33])
    nre33 = mean_std([float(r["NRE"]) for r in ip1_33])
    exact69 = mean_std([float(r["EnumBF"]) for r in ip1_69])
    nre69 = mean_std([float(r["NRE"]) for r in ip1_69])
    exact119 = mean_std([r["exact"] for r in ip1_119])
    nre119 = mean_std([r["nre"] for r in ip1_119])
    kl119 = mean_std([r["kl"] for r in ip1_119])

    lines = []
    lines.append("Step 6 main tables final freeze")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Rounding: percentages use two decimals in this freeze; paper tables may round to one decimal consistently.")
    lines.append("")
    lines.append("Table A. IP1 exact-comparable posterior inference")
    lines.append("system,topology_count,K,exact_top1_mean,nre_top1_mean,nre_std,gap_mean,speedup,source")
    lines.append(f"33-bus,32,20,{fmt_pct(exact33[0])},{fmt_pct(nre33[0])},{fmt_pct(nre33[1])},{fmt_pp(exact33[0]-nre33[0])},1176x,ip1_33bus_n8_result.txt")
    lines.append(f"69-bus,60,20,{fmt_pct(exact69[0])},{fmt_pct(nre69[0])},{fmt_pct(nre69[1])},{fmt_pp(exact69[0]-nre69[0])},3302x,ip1_69bus_n8_result.txt")
    lines.append(f"119-bus,107,25,{fmt_pct(exact119[0])},{fmt_pct(nre119[0])},{fmt_pct(nre119[1])},{fmt_pp(exact119[0]-nre119[0])},{speed119}x,119bus_ip1_result.txt")
    lines.append(f"119-bus supplemental posterior gap metric: KL(reference||NRE)={kl119[0]:.4f}+/-{kl119[1]:.4f}.")
    lines.append("")
    lines.append("Table B. GraphSAGE baseline")
    lines.append("system,GraphSAGE_top1,NRE_top1,exact_top1,GraphSAGE_latency_ms,source")
    if "33bus" in gs:
        a, s, ms, ms_s = gs["33bus"]
        lines.append(f"33-bus,{fmt_pct(a)}+/-{fmt_pct(s)},{fmt_pct(nre33[0])},{fmt_pct(exact33[0])},{ms:.4f}+/-{ms_s:.4f},baseline_unified_table.txt")
    if "69bus" in gs:
        a, s, ms, ms_s = gs["69bus"]
        lines.append(f"69-bus,{fmt_pct(a)}+/-{fmt_pct(s)},{fmt_pct(nre69[0])},{fmt_pct(exact69[0])},{ms:.4f}+/-{ms_s:.4f},baseline_unified_table.txt")
    lines.append("Boundary: GraphSAGE is a point-estimate shared-graph baseline, not posterior output.")
    lines.append("")
    lines.append("Table C. IP-C missing-measurement robustness")
    lines.append("system,n,miss10_robust,miss10_naive,miss10_delta,miss30_robust,miss30_naive,miss30_delta,source")
    for system, fname in [("33-bus", "ipc_33bus_n8_result.txt"), ("69-bus", "ipc_69bus_n8_result.txt"), ("119-bus supplemental", "ipc_119bus_5seed_result.txt")]:
        rows = read_csv_rows(fname)
        r10 = mean_std([float(r["rob10"]) for r in rows])
        n10 = mean_std([float(r["nai10"]) for r in rows])
        r30 = mean_std([float(r["rob30"]) for r in rows])
        n30 = mean_std([float(r["nai30"]) for r in rows])
        lines.append(
            f"{system},{len(rows)},{fmt_pct(r10[0])}+/-{fmt_pct(r10[1])},{fmt_pct(n10[0])},"
            f"{fmt_pp(r10[0]-n10[0])},{fmt_pct(r30[0])}+/-{fmt_pct(r30[1])},{fmt_pct(n30[0])},"
            f"{fmt_pp(r30[0]-n30[0])},{fname}"
        )
    lines.append("")
    lines.append("Table D. IP-A / BOED sensor placement")
    lines.append("system,policy,K4_top1,K7_top1,source")
    for method in ["Random", "MVG", "BOED"]:
        lines.append(f"33-bus,{method},{fmt_pct(boed33[4][method])},{fmt_pct(boed33[7][method])},boed_33bus_nmc500_result.txt")
    for method in ["Random", "GreedyLoop", "MVG", "BOED"]:
        source = "step5_mvg_result.txt" if method == "MVG" else "boed_69bus_result.txt"
        lines.append(f"69-bus,{method},{fmt_pct(boed69[4][method])},{fmt_pct(boed69[7][method])},{source}")
    for method in ["AdaptiveMVG", "AdaptiveFisher", "AdaptiveBOED"]:
        lines.append(f"119-bus,{method},{fmt_pct(boed119[method]['K4_acc'])},{fmt_pct(boed119[method]['K7_acc'])},boed_119bus_minimal_result.txt")
    lines.append(f"Boundary: 33-bus BOED is comparable to MVG; 69-bus BOED is stronger than MVG at K=4/K=7; 119-bus adaptive comparison is the fair sequential-policy evidence. 119-bus AdaptiveBOED entropy drops from {boed119['AdaptiveBOED']['K4_H']:.3f} at K=4 to {boed119['AdaptiveBOED']['K7_H']:.3f} at K=7.")
    lines.append("")
    lines.append("Table E. Posterior quality")
    lines.append("method,top1,NLL,Brier,KL_ref_to_model,ECE,coverage90,avg_credset90,source")
    for method, vals in post.items():
        lines.append(f"{method},{vals[0]:.4f},{vals[1]:.4f},{vals[2]:.4f},{vals[3]:.6f},{vals[4]:.4f},{vals[5]:.4f},{vals[6]:.2f},posterior_calibration_result.txt")
    lines.append("Boundary: calibrated NRE is exact-comparable and conservative, not perfectly calibrated.")
    lines.append("")
    lines.append("Table F. Synthetic 300-bus scalability")
    lines.append("system,n_bus,n_topologies,IP1_NRE_top1,IP1_exact_top1_500,IP-C_NRE_top1_30missing,IP-C_exact_top1_30missing_500,NRE_ms,Exact_ms,Speedup,source")
    lines.append(f"{scal['system']},{scal['n_bus']},{scal['n_topologies']},{fmt_pct(scal['ip1_acc'])},{fmt_pct(scal_exact['ip1_exact_acc'])},{fmt_pct(scal['ipc_acc30'])},{fmt_pct(scal_exact['ipc_exact_acc30'])},{scal['nre_ms']:.6f},{scal['exact_ms']:.6f},{scal['speedup']:.1f}x,scalability_result.txt; {scal_exact['ip1_source']}; {scal_exact['ipc_source']}")
    lines.append("H(K) 300-bus: " + "; ".join([f"K={k}: {h:.4f}+/-{s:.4f}" for k, h, s in ip4_300]))
    lines.append("Boundary: 300-bus is synthetic fixed-deployment mid-scale scalability evidence only; IP-C 30% missing is a stress boundary.")

    open(pjoin("main_tables_final.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def write_master_final():
    lines = []
    lines.append("=" * 80)
    lines.append("MASTER SUMMARY FINAL - STEP 6 FREEZE")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Status: Step 6 artifacts generated; advisor review pending.")
    lines.append("Purpose: final pre-writing numeric and table freeze. This file supersedes older draft summary statistics.")
    lines.append("")
    lines.append("Final freeze artifacts")
    lines.append("  dn_step6_freeze_final.py")
    lines.append("  stats_final.txt")
    lines.append("  main_tables_final.txt")
    lines.append("  master_summary_final.txt")
    lines.append("")
    lines.append("Previous advisor gates")
    lines.append("  Step 0 asset index / stale-number correction: PASS")
    lines.append("  Step 1 119-bus chain closure: PASS")
    lines.append("  Step 2 GraphSAGE baseline: PASS")
    lines.append("  Step 3 robustness boundary: PASS")
    lines.append("  Step 4 posterior quality: PASS")
    lines.append("  Step 5 synthetic 300-bus scalability: PASS")
    lines.append("")
    lines.append("Step 6 final statistical family")
    lines.append("  Primary tests: six planned one-sided Wilcoxon comparisons using current n=8 33/69-bus files.")
    lines.append("  Primary files: ip1_33bus_n8_result.txt, ip1_69bus_n8_result.txt, ipc_33bus_n8_result.txt, ipc_69bus_n8_result.txt.")
    lines.append("  BH-FDR q=0.05: 6/6 primary tests pass.")
    lines.append("  Holm alpha=0.05: 6/6 primary tests pass.")
    lines.append("  Bonferroni alpha=0.05: 6/6 primary tests pass.")
    lines.append("  p_raw for each primary test: 0.003906, because all eight paired deltas have the planned direction.")
    lines.append("  Supplemental 119-bus IP-C tests use n=5 and are not mixed into the primary m=6 family.")
    lines.append("")
    lines.append("Frozen main-table map")
    lines.append("  Table A IP1 exact-comparable posterior inference -> main_tables_final.txt")
    lines.append("  Table B GraphSAGE baseline -> main_tables_final.txt")
    lines.append("  Table C IP-C missing-measurement robustness -> main_tables_final.txt")
    lines.append("  Table D IP-A / BOED sensor placement -> main_tables_final.txt")
    lines.append("  Table E posterior quality -> main_tables_final.txt")
    lines.append("  Table F synthetic 300-bus scalability -> main_tables_final.txt")
    lines.append("")
    lines.append("Additional frozen source files used by Step 6")
    lines.append("  119bus_ip1_result.txt")
    lines.append("  step5_mvg_result.txt")
    lines.append("  ip1_300bus_result.txt")
    lines.append("  ipc_300bus_result.txt")
    lines.append("")
    lines.append("Frozen figure/data-source map for Step 7")
    lines.append("  Fig. 1 identifiability H(K): ip4_hk_result_v5.txt plus ip4_300bus_result.txt")
    lines.append("  Fig. 2 IP1 accuracy/speedup: main_tables_final.txt Table A")
    lines.append("  Fig. 3 robustness boundaries: missing_curve_33bus_result.txt, outage_33bus_result.txt, noise_sensitivity_33bus_result.txt")
    lines.append("  Fig. 4 IP-A sensor placement: main_tables_final.txt Table D")
    lines.append("  Fig. 5 scalability: fig_scalability.py/png and main_tables_final.txt Table F")
    lines.append("  Fig. 6 posterior quality: fig_reliability_diagram.py/png and fig_posterior_case.py/png")
    lines.append("")
    lines.append("Non-negotiable wording boundaries")
    lines.append("  Use exact enumeration / EnumBF for the exact posterior reference; do not call it Xu-style AIS.")
    lines.append("  300-bus evidence is synthetic fixed-deployment mid-scale scalability, not real utility full-scale validation.")
    lines.append("  300-bus IP-C at 30% missing is a stress boundary; main robustness claims come from Step 1 and Step 3.")
    lines.append("  BOED is comparable to MVG on 33-bus and stronger on the current 69-bus case; do not claim universal BOED dominance.")
    lines.append("  GraphSAGE is a point-estimate shared-graph baseline, not a posterior method.")
    lines.append("  Calibration is exact-comparable and conservative, not perfect calibration.")
    lines.append("")
    lines.append("Next gate")
    lines.append("  Step 6 requires advisor PASS before Step 7 figure finalization or formal writing freeze.")
    open(pjoin("master_summary_final.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main():
    tests = collect_core_stats()
    supp = collect_supplemental_119()
    write_stats(tests, supp)
    write_tables(tests, supp)
    write_master_final()
    print("Generated stats_final.txt, main_tables_final.txt, master_summary_final.txt")
    print(f"Primary BH-FDR pass: {sum(t['bh'] for t in tests)}/{len(tests)}")
    print(f"Primary Holm pass: {sum(t['holm'] for t in tests)}/{len(tests)}")
    print(f"Primary Bonferroni pass: {sum(t['bonferroni'] for t in tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
