# JorJor-Inspect Audit Summary

## Executive Summary

| item | status |
|---|---|
| Poker verdict | `composition_shortcut_supported` |
| Synthetic validation | `passed` |
| Toy oracle validation | `passed` |
| Learned synthetic validation | `passed` |
| Explicit belief-state validation | `passed` |
| Non-poker protocol transfer | `passed` |
| Frozen-LM latent-order audit | `order_information_supported` |
| Range-supervised strict gate | `composition_shortcut_supported` |
| MLP capacity gate | `mlp_inconclusive` |
| CBPS claim ladder | `cbps_supported` |
| CBPS behavior link | `composition_bounded_behavior_link` |
| CBPS claim scope | `diagnostic_interpretation_not_ordered_belief` |
| Order-intervention stress | `order_sensitive_stress_supported` |
| Order-intervention scope | `input_level_order_stress_not_legal_poker_causality` |
| Range-subspace stress | `range_subspace_behavior_footprint_not_supported` |
| Range-subspace scope | `range_probe_subspace_behavior_footprint_stress_not_causal_mediation` |
| Range-rank sensitivity | `range_subspace_rank_sensitivity_not_supported` |
| Range-rank scope | `rank_sensitivity_stress_not_causal_mediation` |
| Reproduction manifest | `present` |
| Gaps | `0` |

## Verdict

`composition_shortcut_supported`

## Sources

| role | required | present | path |
|---|---|---|---|
| belief_shortcut_report.json | True | True | outputs/inspect_v119_smoke/belief_shortcut_report.json |
| belief_shortcut_report.md | True | True | outputs/inspect_v119_smoke/belief_shortcut_report.md |
| matched_composition_report.json | True | True | outputs/inspect_v119_smoke/matched_composition_report.json |
| matched_composition_report.md | True | True | outputs/inspect_v119_smoke/matched_composition_report.md |
| claim_gate_matrix.json | True | True | outputs/inspect_v119_smoke/claim_gate_matrix.json |
| claim_gate_matrix.md | True | True | outputs/inspect_v119_smoke/claim_gate_matrix.md |
| order_counterfactual_report.json | False | True | outputs/experiments/v1.5/seed_k_smoke/order_counterfactual_report.json |
| order_counterfactual_report.md | False | True | outputs/experiments/v1.5/seed_k_smoke/order_counterfactual_report.md |
| prefix_trajectory_report.json | False | True | outputs/experiments/v1.8/seed_k_smoke/prefix_trajectory_report.json |
| prefix_trajectory_report.md | False | True | outputs/experiments/v1.8/seed_k_smoke/prefix_trajectory_report.md |
| prefix_trajectory_report.json | False | True | outputs/experiments/v1.8/seed_l_smoke/prefix_trajectory_report.json |
| prefix_trajectory_report.md | False | True | outputs/experiments/v1.8/seed_l_smoke/prefix_trajectory_report.md |
| prefix_trajectory_report.json | False | True | outputs/experiments/v1.8/seed_m_smoke/prefix_trajectory_report.json |
| prefix_trajectory_report.md | False | True | outputs/experiments/v1.8/seed_m_smoke/prefix_trajectory_report.md |
| prefix_composition_baseline_report.json | False | True | outputs/experiments/v1.9/seed_k_smoke/prefix_composition_baseline_report.json |
| prefix_composition_baseline_report.md | False | True | outputs/experiments/v1.9/seed_k_smoke/prefix_composition_baseline_report.md |
| prefix_composition_baseline_report.json | False | True | outputs/experiments/v1.9/seed_l_smoke/prefix_composition_baseline_report.json |
| prefix_composition_baseline_report.md | False | True | outputs/experiments/v1.9/seed_l_smoke/prefix_composition_baseline_report.md |
| prefix_composition_baseline_report.json | False | True | outputs/experiments/v1.9/seed_m_smoke/prefix_composition_baseline_report.json |
| prefix_composition_baseline_report.md | False | True | outputs/experiments/v1.9/seed_m_smoke/prefix_composition_baseline_report.md |
| synthetic_oracle_belief_report.json | False | True | outputs/experiments/v1.7/smoke/synthetic_oracle_belief_report.json |
| synthetic_oracle_belief_report.md | False | True | outputs/experiments/v1.7/smoke/synthetic_oracle_belief_report.md |
| toy_imperfect_info_oracle_report.json | False | True | outputs/experiments/v1.15/smoke/toy_imperfect_info_oracle_report.json |
| toy_imperfect_info_oracle_report.md | False | True | outputs/experiments/v1.15/smoke/toy_imperfect_info_oracle_report.md |
| learned_synthetic_belief_report.json | False | True | outputs/experiments/v1.10/smoke/learned_synthetic_belief_report.json |
| learned_synthetic_belief_report.md | False | True | outputs/experiments/v1.10/smoke/learned_synthetic_belief_report.md |
| explicit_belief_state_report.json | False | True | outputs/experiments/v1.20/smoke/explicit_belief_state_report.json |
| explicit_belief_state_report.md | False | True | outputs/experiments/v1.20/smoke/explicit_belief_state_report.md |
| k.1_latent_type_protocol_transfer_report.json | False | True | outputs/k/paper/k.1_full_ci_fixed/k.1_latent_type_protocol_transfer_report.json |
| k.1_latent_type_protocol_transfer_report.md | False | True | outputs/k/paper/k.1_full_ci_fixed/k.1_latent_type_protocol_transfer_report.md |
| k.1_protocol_transfer_manifest.json | False | True | outputs/k/paper/k.1_full_ci_fixed/k.1_protocol_transfer_manifest.json |
| k.2.6_llm_full_sweep_report.json | False | True | outputs/k/k.2.6.3_llm_full_sweep_remote_qwen05_paper_table/k.2.6_llm_full_sweep_report.json |
| k.2.6_llm_full_sweep_report.md | False | True | outputs/k/k.2.6.3_llm_full_sweep_remote_qwen05_paper_table/k.2.6_llm_full_sweep_report.md |
| k.2.6_llm_full_sweep_manifest.json | False | True | outputs/k/k.2.6.3_llm_full_sweep_remote_qwen05_paper_table/k.2.6_llm_full_sweep_manifest.json |
| cbps_claim_ladder | False | True | outputs/k/k.5.6_cbps_claim_ladder/k.5.3_cbps_claim_ladder.json |
| cbps_claim_ladder_md | False | True | outputs/k/k.5.6_cbps_claim_ladder/k.5.3_cbps_claim_ladder.md |
| cbps_claim_ladder_manifest | False | True | outputs/k/k.5.6_cbps_claim_ladder/k.5.3_cbps_manifest.json |
| cbps_claim_ladder_paper_table | False | True | outputs/k/k.5.6_cbps_claim_ladder/paper_table_cbps_claim_ladder.md |
| order_intervention_report | False | True | outputs/k/k.6.0_order_intervention/k.6.0_order_intervention_report.json |
| order_intervention_report_md | False | True | outputs/k/k.6.0_order_intervention/k.6.0_order_intervention_report.md |
| order_intervention_manifest | False | True | outputs/k/k.6.0_order_intervention/k.6.0_order_intervention_manifest.json |
| range_subspace_report | False | True | outputs/k/k.6.3_range_subspace_ablation/k.6.3_range_subspace_ablation_report.json |
| range_subspace_report_md | False | True | outputs/k/k.6.3_range_subspace_ablation/k.6.3_range_subspace_ablation_report.md |
| range_subspace_manifest | False | True | outputs/k/k.6.3_range_subspace_ablation/k.6.3_range_subspace_ablation_manifest.json |
| range_rank_sensitivity_report | False | True | outputs/k/k.6.5_range_subspace_rank_sensitivity/k.6.5_range_subspace_rank_sensitivity_report.json |
| range_rank_sensitivity_report_md | False | True | outputs/k/k.6.5_range_subspace_rank_sensitivity/k.6.5_range_subspace_rank_sensitivity_report.md |
| range_rank_sensitivity_manifest | False | True | outputs/k/k.6.5_range_subspace_rank_sensitivity/k.6.5_range_subspace_rank_sensitivity_manifest.json |
| one_command_reproduction_manifest | False | True | outputs/paper/v1.13/reproduction_manifest.json |
| one_command_reproduction_manifest_md | False | True | outputs/paper/v1.13/reproduction_manifest.md |
| range_supervised_strict_gate | False | True | outputs/k/k.2.3_strict_tables/limit_ar_range_supervised_positive_control_pack.json |
| probe_capacity_s6_s7 | False | True | outputs/k/k.2.3_strict_tables/probe_capacity_s6_s7_report.json |

## Gaps

- none

## CBPS Claim Ladder

Status: `cbps_supported`

Behavior link: `composition_bounded_behavior_link`

Scope: `diagnostic_interpretation_not_ordered_belief`

Positive hidden-state range probes are insufficient evidence for ordered belief tracking. For the no-range poker AR row, the best-supported diagnostic interpretation is composition-bounded predictive support: behavior prediction is positive, while strict matched-composition gates block an ordered-belief claim.

## Order-Intervention Stress

Status: `order_sensitive_stress_supported`

Scope: `input_level_order_stress_not_legal_poker_causality`

Pairs: `17337` across `3` seeds.

This input-level stress evidence bounds pure bag-of-actions invariance but does not upgrade the poker verdict to legal posterior tracking.

## Range-Subspace Stress

Status: `range_subspace_behavior_footprint_not_supported`

Scope: `range_probe_subspace_behavior_footprint_stress_not_causal_mediation`

Probe top10 mean: `0.12538886070251465`

Policy delta mean: `-0.004071830240916046`

Value delta mean: `-0.16412875887104836`

This linear subspace stress evidence cautions against interpreting range decodability as behavior mediation.

## Range-Rank Sensitivity

Status: `range_subspace_rank_sensitivity_not_supported`

Scope: `rank_sensitivity_stress_not_causal_mediation`

Ranks: `6`

Mean policy delta across ranks: `-0.004126965157469471`

Mean value delta across ranks: `-0.10528861085259278`

This stress evidence checks whether the range-subspace result is stable across predeclared ranks.

## Methodology Validation

`passed`

Synthetic oracle-belief controls validate the diagnostic protocol only; they do not override the poker model verdict.
Non-poker protocol-transfer validation checks whether the audit maps beyond betting histories; it does not upgrade the poker verdict.

## Claim Rule

Positive probe results are insufficient evidence for ordered belief tracking unless they survive shortcut and order-counterfactual gates.
