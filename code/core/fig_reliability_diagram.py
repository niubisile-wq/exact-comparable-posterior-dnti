# -*- coding: utf-8 -*-
"""Generate Step 4 reliability diagram from posterior_reliability_bins.csv."""
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(BASE, "posterior_reliability_bins.csv")
out = os.path.join(BASE, "fig_reliability_diagram.png")
plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9})

data = defaultdict(list)
with open(src, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if not row["accuracy"]:
            continue
        mid = (float(row["bin_lo"]) + float(row["bin_hi"])) / 2
        data[row["method"]].append((mid, float(row["accuracy"]), float(row["confidence"]), int(row["n"])))

styles = {
    "Reference": ("#333333", "o"),
    "Raw_NRE": ("#d95f02", "s"),
    "Calibrated_NRE": ("#1b9e77", "^"),
}

plt.figure(figsize=(4.2, 3.4))
plt.plot([0, 1], [0, 1], color="0.65", linestyle="--", linewidth=1, label="ideal")
for method in ["Reference", "Raw_NRE", "Calibrated_NRE"]:
    vals = data.get(method, [])
    if not vals:
        continue
    conf = [v[2] for v in vals]
    acc = [v[1] for v in vals]
    color, marker = styles[method]
    label = method.replace("_", " ")
    plt.plot(conf, acc, marker=marker, color=color, linewidth=1.6, label=label)
plt.xlabel("Mean confidence")
plt.ylabel("Empirical accuracy")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, alpha=0.25)
plt.legend(frameon=False, loc="lower right")
plt.tight_layout()
plt.savefig(out, dpi=300)
print(out)
