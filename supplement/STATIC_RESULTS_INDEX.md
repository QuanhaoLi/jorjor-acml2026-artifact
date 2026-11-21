
# Static Results Index

## Primary Evidence Packs

- `artifacts/camera_ready_rebuttal/`
- `supplement/CAMERA_READY_REBUTTAL_ADDENDUM.md`
- `supplement/camera_ready_rebuttal/source_reports/`

- `artifacts/checkpoints_tables/limit_ar_paper_evidence_pack.*`
- `artifacts/checkpoints_tables/limit_ar_no_range_uncertainty_pack.*`
- `artifacts/claim_audits/k.5.1_full_poker_predictive_link/`
- `artifacts/claim_audits/k.5.2_composition_boundedness/`
- `artifacts/claim_audits/k.5.5_representation_behavior_link/`
- `artifacts/claim_audits/k.6.6_current_product_audit_pack/`

## Validation and Boundary Packs

- `artifacts/claim_audits/k.3.0_poker_oracle_positive_control/`
- `artifacts/claim_audits/k.3.1_composition_feature_sensitivity/`
- `artifacts/claim_audits/k.4.0_behavior_required_pomdp/`
- `artifacts/claim_audits/k.5.0_synthetic_cbps_control/`
- `artifacts/claim_audits/k.6.0_order_intervention/`
- `artifacts/claim_audits/k.6.3_range_subspace_ablation/`
- `artifacts/claim_audits/k.6.5_range_subspace_rank_sensitivity/`

## Figure Packs

- `artifacts/paper_outputs/v1.16/figure_pack.*`
- `artifacts/figures/*.png`
- `artifacts/camera_ready_rebuttal/figure_*`

## Checksum Manifest

Run:

```bash
python tools/check_artifact.py
```

This writes `MANIFEST_CHECKSUMS.sha256` with SHA-256 checksums for included files.
