"""Task 0003 comparison: regenerated figures vs committed figures. Raw numbers only.

Usage: python -I compare.py <regen_dir> <reference_dir> <out_csv>
"""
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
from matplotlib.image import imread

regen_dir = Path(sys.argv[1])
ref_dir = Path(sys.argv[2])
out_csv = Path(sys.argv[3])


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


rows = []
for regen_png in sorted(regen_dir.glob("*.png")):
    stem = regen_png.stem
    ref_png = ref_dir / f"{stem}.png"
    row = {"stem": stem, "has_reference": ref_png.exists()}
    if not ref_png.exists():
        rows.append(row)
        continue
    a = imread(regen_png)
    b = imread(ref_png)
    row["regen_shape"] = "x".join(map(str, a.shape))
    row["ref_shape"] = "x".join(map(str, b.shape))
    if a.shape == b.shape:
        d = np.abs(a.astype(float) - b.astype(float))
        row["max_abs_diff"] = float(d.max())
        row["mean_abs_diff"] = float(d.mean())
        row["pixels_differing"] = int(np.count_nonzero(d.max(axis=-1) > 0)) if d.ndim == 3 else int(np.count_nonzero(d))
    else:
        row["max_abs_diff"] = ""
        row["mean_abs_diff"] = ""
        row["pixels_differing"] = "shape_mismatch"
    regen_pdf = regen_dir / f"{stem}.pdf"
    ref_pdf = ref_dir / f"{stem}.pdf"
    row["pdf_sha256_equal"] = (
        sha256(regen_pdf) == sha256(ref_pdf) if regen_pdf.exists() and ref_pdf.exists() else "missing"
    )
    rows.append(row)

# Reference figures that the scripts did not regenerate
regen_stems = {p.stem for p in regen_dir.glob("*.png")}
for ref_png in sorted(ref_dir.glob("*.png")):
    if ref_png.stem not in regen_stems:
        rows.append({"stem": ref_png.stem, "has_reference": True, "pixels_differing": "not_regenerated"})

fields = [
    "stem", "has_reference", "regen_shape", "ref_shape",
    "max_abs_diff", "mean_abs_diff", "pixels_differing", "pdf_sha256_equal",
]
with out_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"rows={len(rows)}")
