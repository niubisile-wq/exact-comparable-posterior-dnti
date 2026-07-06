# -*- coding: utf-8 -*-
"""Generate Step 3 outage-induced missing bar chart from result text."""
import os

import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(BASE, "outage_33bus_result.txt")
out = os.path.join(BASE, "fig_outage_bar.png")
plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9})

seed_rows = []
with open(src, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) >= 4 and parts[0].isdigit():
            seed_rows.append({"seed": int(parts[0]), "rob": float(parts[1]), "naive": float(parts[2])})

rob_mean = sum(r["rob"] for r in seed_rows) / len(seed_rows)
naive = seed_rows[0]["naive"]
rob_std = (sum((r["rob"] - rob_mean) ** 2 for r in seed_rows) / len(seed_rows)) ** 0.5

plt.figure(figsize=(3.6, 3.0))
plt.bar([0, 1], [naive * 100, rob_mean * 100], yerr=[0, rob_std * 100], capsize=4, color=["#8a8f98", "#2f6f9f"])
plt.xticks([0, 1], ["Naive NRE", "Robust NRE"])
plt.ylabel("Top-1 accuracy (%)")
plt.title("Structured outage-induced missing")
plt.ylim(0, max(naive, rob_mean) * 100 + 12)
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(out, dpi=300)
print(out)
