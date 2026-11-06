# Exact-Comparable Posterior DNTI

Code, figure source data, frozen result manifests, and audit notes for the manuscript:

**An Exact-Comparable Posterior Contract for Decision-Ready Distribution-Network Topology Identification**

This repository supports editor and reviewer inspection of the numerical results, figures, and reproducibility claims in the manuscript. It is intended as the public archival package for GitHub/Zenodo release.

## Repository Contents

```text
code/
  core/                  Core experiment and audit scripts
  experiments/           Late-stage stress-test and extension scripts
scripts/                 Final plotting scripts used for manuscript figures
figures/                 Final PDF, PNG, and SVG figure assets
data/
  frozen_tables_stats/   Frozen result manifests, statistical summaries, and source CSVs
  external_data_access/  Notes for public/third-party data sources not redistributed here
docs/                    Reproduction guide, data dictionary, traceability map, release checklist
```

The repository deliberately excludes private raw utility deployments, large model checkpoints, binary candidate-library caches, LaTeX build intermediates, browser/session files, and unpublished third-party raw datasets that the authors are not allowed to redistribute.

## Quick Reproduction

Install the Python dependencies:

```powershell
py -3 -m pip install -r requirements.txt
```

Regenerate the final figures from the repository root:

```powershell
python scripts/plot_fig1_identifiability.py
python scripts/plot_fig2_ip1.py
python scripts/plot_fig3_robustness.py
python scripts/plot_fig_noise_sensitivity.py
python scripts/plot_fig4_boed_sensor_placement.py
python scripts/plot_fig5_scalability.py
python scripts/plot_fig6_posterior_quality.py
python scripts/plot_fig7_evidence_board.py
python scripts/plot_fig8_experiment_landscape.py
python scripts/plot_fig9_ablation_controls.py
python scripts/plot_submission_schematics.py
```

The scripts save figure assets next to the scripts by default. The published figure copies are kept in `figures/`.

## Frozen-Result Audit

Use `docs/RESULT_TRACEABILITY.md` to map manuscript tables and figures to the relevant frozen result files under `data/frozen_tables_stats/`.

Recommended audit route:

1. Open `docs/RESULT_TRACEABILITY.md`.
2. Locate the manuscript figure or table.
3. Inspect the listed result manifest in `data/frozen_tables_stats/`.
4. Compare the frozen numerical value with the manuscript table, caption, or plotted value.

## Full Experiment Reruns

The scripts under `code/` document the training, exact-audit, robustness, baseline, sensor-placement, three-phase, SoCal replay, and large-system experiments. Full reruns can be computationally expensive and may require external public datasets, local candidate-library caches, and GPU acceleration.

Direct local paths in the public copy have been sanitized. If a rerun requires local data, configure paths according to `docs/REPRODUCTION_GUIDE.md` and `data/external_data_access/`.

## Data And Code Availability

The public release contains:

- final plotting scripts and figure assets
- frozen result manifests and statistical summaries
- traceability map from manuscript claims to repository artifacts
- public/third-party data access notes
- experiment and audit scripts needed to inspect the computational workflow

Raw third-party datasets are not redistributed unless redistribution is permitted. Access instructions are provided instead.

## Citation

If you use this repository, cite the archived Zenodo release DOI after it is generated. Before DOI assignment, cite the GitHub repository URL:

```text
Liu, Z. and Xiong, W. (2026). Exact-comparable posterior contract for decision-ready distribution-network topology identification: code, figure source data, and result manifests. GitHub. https://github.com/niubisile-wq/exact-comparable-posterior-dnti
```

## License

Code is released under the MIT License. Derived result manifests, documentation, and figures are released under CC BY 4.0 unless a file or external-data note states otherwise. Third-party raw data remain subject to their original providers' terms.
