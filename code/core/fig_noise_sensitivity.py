# -*- coding: utf-8 -*-
"""Generate Step 3 noise sensitivity curve from result text."""
import csv
import os

import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(BASE, "noise_sensitivity_33bus_result.txt")
out = os.path.join(BASE, "fig_noise_sensitivity.png")
plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9})

rows = []
with open(src, "r", encoding="utf-8") as f:
    reader = csv.DictReader(line for line in f if line and (line[0].isdigit() or line.startswith("sigma,")))
    for row in reader:
        rows.append(row)

sigma = [float(r["sigma"]) for r in rows]
rob = [float(r["rob_mean"]) * 100 for r in rows]
rob_std = [float(r["rob_std"]) * 100 for r in rows]
naive = [float(r["naive"]) * 100 for r in rows]

plt.figure(figsize=(4.8, 3.1))
plt.errorbar(sigma, rob, yerr=rob_std, marker="o", linewidth=2, capsize=3, label="Robust NRE")
plt.plot(sigma, naive, marker="s", linewidth=2, label="Naive NRE")
plt.xlabel("Voltage noise sigma")
plt.ylabel("Top-1 accuracy at 30% missing (%)")
plt.grid(True, alpha=0.25)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(out, dpi=300)
print(out)
