# -*- coding: utf-8 -*-
"""Generate Step 4 multimodal posterior top-k comparison figure."""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(BASE, "posterior_multimodal_case.txt")
out = os.path.join(BASE, "fig_posterior_case.png")
plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 9})

rows = []
with open(src, "r", encoding="utf-8") as f:
    lines = f.readlines()
start = None
for i, line in enumerate(lines):
    if line.startswith("topology,reference"):
        start = i
        break
if start is None:
    raise RuntimeError("No posterior table found")

reader = csv.DictReader(lines[start:])
for row in reader:
    if not row.get("topology") or not row["topology"].isdigit():
        continue
    rows.append(row)

# Keep visible modes; retain at least top-2 for the multimodal comparison.
visible = [
    r for r in rows
    if max(float(r["reference"]), float(r["raw_NRE"]), float(r["calibrated_NRE"])) > 1e-3
]
rows = visible if len(visible) >= 2 else rows[:2]
labels = [f"T{int(r['topology'])}" for r in rows]
ref = np.array([float(r["reference"]) for r in rows])
raw = np.array([float(r["raw_NRE"]) for r in rows])
cal = np.array([float(r["calibrated_NRE"]) for r in rows])
x = np.arange(len(rows))
w = 0.24

plt.figure(figsize=(4.8, 3.1))
plt.bar(x - w, ref, width=w, label="Reference", color="#333333")
plt.bar(x, raw, width=w, label="Raw NRE", color="#d95f02")
plt.bar(x + w, cal, width=w, label="Calibrated NRE", color="#1b9e77")
plt.xticks(x, labels)
plt.ylabel("Posterior probability")
plt.ylim(0, max(ref.max(), raw.max(), cal.max()) * 1.18)
plt.grid(axis="y", alpha=0.25)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(out, dpi=300)
print(out)
