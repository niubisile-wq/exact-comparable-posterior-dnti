# Data Dictionary

## `frozen_tables_stats`

Text manifests that freeze headline numerical results and audit logs. These files are the primary evidence source for manuscript tables, figures, and statistical claims.

Key result families:

- `33bus_*`, `69bus_*`, `119bus_*`: exact-comparable posterior, missing-data, sensor-budget, and robustness results.
- `300bus_*`: synthetic 300-bus scalability, severe-missing rerank, confidence-stratified policy, and credible-set audits.
- `boed_*`: Bayesian optimal experimental design and sensor-placement comparisons.
- `graphsage_*` or graph-baseline manifests: point-estimate graph neural network comparator checks.
- `ieee123_*` and `37bus_*`: controlled three-phase and unbalanced stress evidence.
- `socal_*`: public synchronized-measurement replay and leakage/blocking audit.
- `statistical_*`, `wilcoxon_*`, or similar files: planned statistical-test outputs.

## `figure_source_data`

Derived values used by plotting scripts. Public release should include either the exact data files used by each plot or a script that extracts those values from `frozen_tables_stats`.

## `external_data_access`

Access notes for third-party or public datasets that should not be redistributed as raw data unless their source licenses permit it. This folder should include access instructions, citations, and request templates where applicable.

## `code/core`

Candidate-library construction, exact-reference evaluation, posterior training/evaluation, calibration, and utility scripts.

## `code/experiments`

Experiment drivers for manuscript-specific evidence lines, including robustness, BOED, graph baselines, reranking, large-system stress, and public-replay audits.

## `manuscript`

LaTeX source, generated PDF, figure scripts, and final figure files.
