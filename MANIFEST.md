
# Artifact Manifest

This manifest summarizes the submitted supplementary artifact. The package is organized so that reviewers can inspect the evidence-to-claim mapping without downloading raw checkpoints or rerunning long training jobs.

## Core Source

| Path | Purpose |
|---|---|
| `src/jorjor/` | Python package implementing data, environments, models, probes, inspection utilities, and training helpers. |
| `scripts/` | Report builders, paper table reproduction helpers, smoke scripts, and audit entry points. |
| `configs/` | Data, training, and probe configuration files. |
| `tests/` | Unit and smoke tests that can run locally. |

## Static Evidence

| Path | Purpose |
|---|---|
| `artifacts/paper_outputs/v1.13/` | Final diagnostic table packs and reproduction manifest from the paper-generation line. |
| `artifacts/paper_outputs/v1.16/` | Figure source JSON and SVG files for the paper figure pack. |
| `artifacts/paper_outputs/v1.17/` | Range-supervised positive-control smoke pack. |
| `artifacts/camera_ready_rebuttal/` | Verified run-level JSON and figure exports for the fresh five-run and exact-posterior camera-ready addendum. |
| `artifacts/checkpoints_tables/` | Summary evidence and uncertainty packs copied from checkpoint table outputs. |
| `artifacts/claim_audits/k.5.1*` | Full-test poker predictive-link reports by seed and aggregate. |
| `artifacts/claim_audits/k.5.2*` | Composition-boundedness reports. |
| `artifacts/claim_audits/k.5.5*` | Representation-to-behavior link reports. |
| `artifacts/claim_audits/k.5.7*` | Product audit with CBPS behavior evidence. |
| `artifacts/claim_audits/k.6.2*`, `k.6.6*` | Submission-era product audit packs and claim-gate outputs retained for provenance. |
| `artifacts/figures/` | Historical submission PNG figure exports retained for provenance. |

## Supplement Files

| Path | Purpose |
|---|---|
| `supplement/CLAIM_LEDGER.md` | Allowed claims, forbidden claims, and evidence boundaries. |
| `supplement/TABLE_FIGURE_PROVENANCE.md` | Mapping from paper tables/figures to artifact files. |
| `supplement/STATIC_RESULTS_INDEX.md` | Index of included JSON/MD static results. |
| `supplement/REVIEWER_GUIDE.md` | Suggested reviewer workflow for inspecting the artifact. |
| `supplement/LIMITATIONS_AND_EXTERNAL_ARCHIVE.md` | What requires an external archive root. |
| `supplement/CAMERA_READY_REBUTTAL_ADDENDUM.md` | Camera-ready five-run and exact-posterior evidence boundary, support repair, hashes, and source map. |
| `supplement/camera_ready_rebuttal/` | Frozen narrative reports and inspection-only code snapshot for the camera-ready addendum. |

## Generated Files

`tools/check_artifact.py` generates:

- `MANIFEST_CHECKSUMS.sha256`
- `artifact_check_report.json`

These files are safe to regenerate after any package change. The report uses artifact-relative paths only.

## Validation Status

See `VALIDATION_STATUS.md` for the latest local check and full test-suite results.
