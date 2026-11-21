
# Reviewer Guide

This guide gives a short path through the supplementary artifact for reviewers who want to verify the evidence boundary without rerunning long training jobs.

## Suggested Inspection Order

1. Read `CODE_OF_ARTIFACT.md` to understand artifact scope and excluded materials.
2. Run `python tools/check_artifact.py`.
3. Inspect `supplement/CLAIM_LEDGER.md` to verify allowed and forbidden claims.
4. Inspect `supplement/TABLE_FIGURE_PROVENANCE.md` to map manuscript results to artifact files.
5. Inspect `supplement/CAMERA_READY_REBUTTAL_ADDENDUM.md` and `artifacts/camera_ready_rebuttal/` for the camera-ready five-run and exact-posterior additions.
6. Use `artifacts/claim_audits/k.6.6_current_product_audit_pack/` as submission-era audit provenance; it does not integrate the later camera-ready addendum.
7. Run `PYTHONPATH=src python -m unittest discover tests` if local dependencies are available.

## What To Check First

- `claim_gate_matrix.md` for strict and refined case labels.
- `diagnostic_report.md` for the audit summary.
- `tables.json` for machine-readable table rows.
- `source_manifest.json` for source provenance.

## Expected Reviewer Conclusion

The artifact should make clear that the poker row is not upgraded to belief tracking. The submitted panel supplies scoped predictive/CBPS evidence, while the fresh five-run panel separately supplies the composition bound. S7 is target-stratified sensitivity evidence, and the exact-posterior pair analysis measures cross-hand geometry rather than legal temporal updates. Validation rows test protocol behavior; they are not evidence that the audited poker model itself implements posterior tracking.
