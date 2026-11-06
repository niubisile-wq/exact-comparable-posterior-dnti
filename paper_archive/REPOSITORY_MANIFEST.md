# Repository Manifest

This manifest summarizes the public release package prepared on 2026-07-06.

## Public Contents

- `code/core/`: core Python scripts for exact audits, posterior training/evaluation, baselines, stress tests, and data-source checks.
- `code/experiments/`: late-stage extension scripts used for strengthened robustness, scale, and boundary experiments.
- `scripts/`: manuscript plotting scripts for the final figures and schematic figures.
- `figures/`: final PDF, PNG, and SVG figure assets.
- `data/frozen_tables_stats/`: frozen result manifests, tables, statistical summaries, and source CSVs used to audit manuscript numbers.
- `data/external_data_access/`: access notes for third-party or public datasets that are not redistributed here.
- `docs/`: reproduction guide, result-traceability map, data dictionary, release checklist, and submission availability wording.

## Excluded From Public Release

- private raw utility data or field deployments
- third-party raw datasets without redistribution permission
- model checkpoints and large binary candidate-library caches
- LaTeX build intermediates and temporary visual checks
- browser cookies, tokens, account files, or machine-specific environment files
- internal manuscript drafts that are not part of the code/data archive

## Integrity Notes

The public copy sanitizes direct local absolute paths. Some experiment scripts still require users to configure repository-relative data paths or external data locations before full reruns. Frozen-result auditing and figure regeneration do not require private raw data.
