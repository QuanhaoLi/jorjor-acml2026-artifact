
# Table and Figure Provenance

## Main Paper Tables

| Manuscript item | Artifact source |
|---|---|
| Claim preview / evidence-to-claim summary | `artifacts/claim_audits/k.5.6_cbps_claim_ladder/paper_table_cbps_claim_ladder.md` |
| Claim-gate matrix | `artifacts/claim_audits/k.6.6_current_product_audit_pack/claim_gate_matrix.md` |
| Known-control CBPS validation | `artifacts/claim_audits/k.5.0_synthetic_cbps_control/k.5.0_synthetic_cbps_report.md` |
| Full-test predictive link | `artifacts/claim_audits/k.5.1_full_poker_predictive_link/k.5.1_poker_predictive_link_report.md` |
| Weak range probes and uncertainty packs | `artifacts/checkpoints_tables/limit_ar_no_range_uncertainty_pack.md` |
| Composition-boundedness and matched-composition rows | `artifacts/claim_audits/k.5.2_composition_boundedness/k.5.2_composition_boundedness_report.md` |
| Fresh five-run G5/G6/S7 summary | `artifacts/camera_ready_rebuttal/five_run_and_exact_posterior_data.json`; narrative source `supplement/camera_ready_rebuttal/source_reports/2026-08-31-k.8.0.md` |
| Exact-posterior five-run summary | `artifacts/camera_ready_rebuttal/five_run_and_exact_posterior_data.json`; narrative source `supplement/camera_ready_rebuttal/source_reports/2026-09-01-k.9.0.md` |
| Composition-feature sensitivity | `artifacts/claim_audits/k.3.1_composition_feature_sensitivity/k.3.1_composition_feature_sensitivity_report.md` |
| Posterior and learned-positive validations | explicit and latent-order report data in `artifacts/claim_audits/k.6.6_current_product_audit_pack/tables.json`; related boundary reports in `artifacts/claim_audits/k.4.0_behavior_required_pomdp/` and `artifacts/claim_audits/k.3.0_poker_oracle_positive_control/` |

## Main Paper Figures

| Manuscript item | Artifact source |
|---|---|
| Five-run realized-hand gates | `artifacts/camera_ready_rebuttal/figure_five_run_realized_hand_gates.*`; verified data in `five_run_and_exact_posterior_data.json` |
| Exact teacher-induced posterior | `artifacts/camera_ready_rebuttal/figure_exact_teacher_posterior_evidence.*`; verified data in `five_run_and_exact_posterior_data.json` |
| Composition gate per seed | `artifacts/figures/figure_3_composition_gate_per_seed_paper_flat.png`; source JSON/SVG in `artifacts/paper_outputs/v1.16/` |
| Matched-composition decisive figure | `artifacts/figures/figure_4_matched_composition_decisive_paper_flat.png`; source JSON/SVG in `artifacts/paper_outputs/v1.16/` |
| Oracle validation ordered vs shortcut | `artifacts/figures/figure_5_oracle_validation_ordered_vs_shortcut_paper_flat.png`; source JSON/SVG in `artifacts/paper_outputs/v1.16/` |

The composition and matched-composition figures above are retained as historical submission provenance. The camera-ready main text replaces them with the five-run figure. The oracle-validation figure is supplementary in the camera-ready version.

## Submission-Era Product Audit Packs

The most complete pre-addendum static audit packs are:

- `artifacts/claim_audits/k.6.6_current_product_audit_pack/`
- `artifacts/claim_audits/k.5.7_product_audit_cbps_behavior/`

These contain `artifact_index.json`, `source_manifest.json`, `tables.json`, `diagnostic_report.*`, and `claim_gate_matrix.md`. They preserve submission-era provenance and do not integrate the later five-run or exact-posterior camera-ready additions.
