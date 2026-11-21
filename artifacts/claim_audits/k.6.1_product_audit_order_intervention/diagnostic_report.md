# JorJor-Inspect Audit Summary

## Executive Summary

| item | status |
|---|---|
| Poker verdict | `composition_shortcut_supported` |
| Synthetic validation | `missing` |
| Toy oracle validation | `missing` |
| Learned synthetic validation | `missing` |
| Explicit belief-state validation | `missing` |
| Non-poker protocol transfer | `missing` |
| Frozen-LM latent-order audit | `missing` |
| Range-supervised strict gate | `composition_shortcut_supported` |
| MLP capacity gate | `mlp_inconclusive` |
| CBPS claim ladder | `cbps_supported` |
| CBPS behavior link | `composition_bounded_behavior_link` |
| CBPS claim scope | `diagnostic_interpretation_not_ordered_belief` |
| Order-intervention stress | `order_sensitive_stress_supported` |
| Order-intervention scope | `input_level_order_stress_not_legal_poker_causality` |
| Reproduction manifest | `missing` |
| Gaps | `6` |

## Verdict

`composition_shortcut_supported`

## Sources

| role | required | present | path |
|---|---|---|---|
| belief_shortcut_report.json | True | True | outputs/inspect/belief_shortcut_report.json |
| belief_shortcut_report.md | True | True | outputs/inspect/belief_shortcut_report.md |
| matched_composition_report.json | True | True | outputs/inspect/matched_composition_report.json |
| matched_composition_report.md | True | True | outputs/inspect/matched_composition_report.md |
| claim_gate_matrix.json | True | True | outputs/inspect/claim_gate_matrix.json |
| claim_gate_matrix.md | True | True | outputs/inspect/claim_gate_matrix.md |
| cbps_claim_ladder | False | True | outputs/k/k.5.6_cbps_claim_ladder/k.5.3_cbps_claim_ladder.json |
| cbps_claim_ladder_md | False | True | outputs/k/k.5.6_cbps_claim_ladder/k.5.3_cbps_claim_ladder.md |
| cbps_claim_ladder_manifest | False | True | outputs/k/k.5.6_cbps_claim_ladder/k.5.3_cbps_manifest.json |
| cbps_claim_ladder_paper_table | False | True | outputs/k/k.5.6_cbps_claim_ladder/paper_table_cbps_claim_ladder.md |
| order_intervention_report | False | True | outputs/k/k.6.0_order_intervention/k.6.0_order_intervention_report.json |
| order_intervention_report_md | False | True | outputs/k/k.6.0_order_intervention/k.6.0_order_intervention_report.md |
| order_intervention_manifest | False | True | outputs/k/k.6.0_order_intervention/k.6.0_order_intervention_manifest.json |
| range_supervised_strict_gate | False | True | outputs/k/k.2.3_strict_tables/limit_ar_range_supervised_positive_control_pack.json |
| probe_capacity_s6_s7 | False | True | outputs/k/k.2.3_strict_tables/probe_capacity_s6_s7_report.json |

## Gaps

- optional order-counterfactual report is missing
- optional prefix trajectory report is missing
- optional prefix composition baseline report is missing
- optional synthetic oracle-belief validation report is missing
- optional learned synthetic belief validation report is missing
- optional explicit belief-state validation report is missing

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

## Methodology Validation

`missing`

Synthetic oracle-belief controls validate the diagnostic protocol only; they do not override the poker model verdict.
Non-poker protocol-transfer validation checks whether the audit maps beyond betting histories; it does not upgrade the poker verdict.

## Claim Rule

Positive probe results are insufficient evidence for ordered belief tracking unless they survive shortcut and order-counterfactual gates.
