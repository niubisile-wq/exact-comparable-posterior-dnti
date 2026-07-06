# Reproduction Guide

## Environment

Use Python 3.10 or later where possible. Install the draft dependencies with:

```powershell
py -3 -m pip install -r requirements.txt
```

For manuscript compilation, use a LaTeX distribution with `latexmk` and the Elsevier `elsarticle` class.

## Route 1: Manuscript and Figure Reproduction

1. Copy the manuscript `.tex`, figure files, and plotting scripts into `manuscript/`.
2. Regenerate figures by running the `plot_fig*.py` scripts.
3. Compile the manuscript:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

Expected assets include figures 1 to 9, the noise-sensitivity figure, and the schematic PNG files.

## Route 2: Frozen-Result Audit

Use the files under `data/frozen_tables_stats/` to verify each manuscript number without retraining:

- IP1 exact-comparable posterior results
- IP-C missing-measurement robustness
- GraphSAGE and stronger graph-baseline checks
- BOED and sensor-budget results
- posterior-quality calibration and credible-set diagnostics
- 202-/417-bus dropout stress
- 300-bus severe-missing rerank and credible-set policy audits
- SoCal public-measurement replay
- IEEE123 and 37-bus three-phase stress
- statistical testing manifests

Each audit should compare the frozen text manifest with the manuscript table or figure that cites it.

## Route 3: Full Experiment Rerun

Full reruns use scripts from `code/core/` and `code/experiments/`. This route can be slow because it rebuilds candidate libraries, trains neural posterior models, and evaluates exact or high-fidelity references.

Before rerunning:

- replace local absolute paths with repository-relative paths or environment variables;
- confirm whether model checkpoints should be regenerated or reused;
- confirm GPU availability for training-heavy scripts;
- keep random seeds and frozen split files unchanged when reproducing manuscript numbers.

## Result Integrity Checks

Before public release, run:

```powershell
Get-ChildItem -Recurse -File | Select-String -Pattern "C:/Users","/Users/","Desktop","TODO","PLACEHOLDER","TBD"
```

Remove local paths from public scripts or document them as reviewer-only provenance notes.
