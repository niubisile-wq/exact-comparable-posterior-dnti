# Result Traceability Map

This map links manuscript display items to the repository artifacts that should support them. It is intended for editor/reviewer audit and should be kept synchronized with the manuscript.

## Figures

| Manuscript item | Asset or script | Evidence source |
|---|---|---|
| Fig. 1, workflow schematic | `fig_workflow_3d.*`, `plot_submission_schematics.py` | manuscript method definition |
| Fig. 2, exact-comparable principle | `fig_principle_3d.*`, `plot_submission_schematics.py` | method definition and exact-posterior equations |
| Fig. 3, experiment landscape | `fig8_experiment_landscape.*`, `plot_fig8_experiment_landscape.py` | `figure_freeze_manifest.txt`, `main_tables_final.txt`, `stats_final.txt`, headline frozen manifests |
| Fig. 4, benchmark schematic | `fig_benchmark_3d.*`, `plot_submission_schematics.py` | benchmark-role definitions in manuscript |
| Fig. 5, identifiability | `fig1_identifiability.*`, `plot_fig1_identifiability.py` | `check_entropy.py`, entropy/sensor-budget manifests, `build_300bus_result.txt` |
| Fig. 6, IP1 accuracy and speed | `fig2_ip1_accuracy_speed.*`, `plot_fig2_ip1.py` | `ip1_33bus_n8_result.txt`, `ip1_69bus_n8_result.txt`, `119bus_ip1_result.txt`, `ip1_300bus_result.txt` |
| Fig. 7, missing-measurement robustness | `fig3_robustness.*`, `plot_fig3_robustness.py` | `ipc_33bus_n8_result.txt`, `ipc_69bus_n8_result.txt`, `ipc_119bus_5seed_result.txt`, `outage_33bus_result.txt` |
| Fig. 8, noise sensitivity | `fig_noise_sensitivity.*`, `plot_fig_noise_sensitivity.py` | `noise_sensitivity_33bus_result.txt`, `counterattack_noise_robustness_20260702.txt` |
| Fig. 9, BOED sensor placement | `fig4_boed_sensor_placement.*`, `plot_fig4_boed_sensor_placement.py` | `boed_33bus_nmc500_result.txt`, `boed_69bus_result.txt`, `boed_119bus_budget_curve_20260703.txt` |
| Fig. 10, 300-bus scalability | `fig5_scalability.*`, `plot_fig5_scalability.py` | `ip1_300bus_result.txt`, `ipc_300bus_result.txt`, `300bus_selective_rerank_curve_20260702.txt`, `300bus_credible_set_policy_20260703.txt` |
| Fig. 11, posterior quality | `fig6_posterior_quality.*`, `plot_fig6_posterior_quality.py` | `posterior_calibration_audit_20260702.txt`, `posterior_reliability_bins.csv`, `posterior_multimodal_case.txt` |
| Fig. 12, evidence board | `fig7_evidence_board.*`, `plot_fig7_evidence_board.py` | `main_tables_final.txt`, `stats_final.txt`, `q1_main_text_result_table_candidates_20260703.txt` |
| Fig. 13, ablation dashboard | `fig9_ablation_controls.*`, `plot_fig9_ablation_controls.py` | `baseline_unified_table.txt`, `step5_ablation_tables.txt`, `socal_posterior_ablation_leakage_audit_20260703.txt` |

## Tables

| Manuscript item | Evidence source |
|---|---|
| Table 1, operational decision contract | method definition and decision-rule text |
| Table 2, IP1 exact-comparable posterior inference | `ip1_33bus_n8_result.txt`, `ip1_69bus_n8_result.txt`, `119bus_ip1_result.txt` |
| Table 3, GraphSAGE point-estimate baseline | `graphsage_33bus_result.txt`, `graphsage_69bus_result.txt`, `graphsage_baseline_summary.txt`, `graphsage_inference_time.txt` |
| Table 4, IP-C missing-measurement robustness | `ipc_33bus_n8_result.txt`, `ipc_69bus_n8_result.txt`, `ipc_119bus_5seed_result.txt` |
| Table 5, IP-A sensor-placement comparison | `boed_33bus_nmc500_result.txt`, `boed_69bus_result.txt`, `boed_119bus_minimal_result.txt`, `boed_119bus_budget_curve_20260703.txt` |
| Table 6, posterior-quality metrics | `posterior_calibration_audit_20260702.txt`, `posterior_calibration_result.txt` |
| Table 7, realistic-measurement and three-phase stress | `socal_posterior_ablation_leakage_audit_20260703.txt`, `ieee123_3ph_composite_stress_20260703.txt`, `ieee123_3ph_phase_ambiguity_audit_20260703.txt`, `ieee123_3ph_unbalance_severity_audit_20260703.txt`, `ieee123_native_pilot_5seed_result_20260703.txt`, `threephase_37bus_fast_result.txt` |
| Table 8, scale, decision value, severe-missing boundary | `reconfig_fullscale_stress_result.txt`, `boed_119bus_budget_curve_20260703.txt`, `300bus_credible_set_policy_20260703.txt`, `300bus_confidence_stratified_policy_20260702.txt`, `large_system_candidate_rerank_20260702.txt`, `large_system_method_upgrade_20260702.txt` |
| Table 9, field-readiness audit | `socal_*`, `ieee123_*`, `reconfig_*`, `300bus_*`, repository release notes |
| Table 10, comparative and negative-control evidence | `boed_ais_result.txt`, `baseline_unified_table.txt`, `posterior_weighted_residual_consistency_20260703.txt`, `step5_ablation_tables.txt` |

## Rule

Every manuscript number should be traceable to one frozen manifest or one plotting script output. If a number is changed in the manuscript, update this map and the corresponding frozen manifest first.
