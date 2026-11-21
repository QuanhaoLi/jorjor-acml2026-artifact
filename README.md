# JorJor ACML 2026 Supplementary Artifact

This package accompanies the ACML 2026 conference-track submission on auditing hidden-state belief claims in imperfect-information autoregressive models. It is designed for reviewer inspection: the package includes the source code, configurations, static evidence files, table and figure provenance, validation notes, and lightweight checks needed to verify the paper's reported claim-gate evidence.

The artifact is intentionally lightweight. It does not bundle long-run checkpoints, virtual environments, raw probe caches, or model weights.

## Quick Checks

From the artifact root:

```bash
python tools/check_artifact.py
python tools/smoke_static_artifacts.py
```

The first command checks package hygiene, required files, checksums, README coverage, and banned local-development traces. The second command loads key JSON evidence files and verifies that the included figure exports and claim-ledger text are present.

For a full local source sanity check, use a Python environment with PyTorch installed:

```bash
PYTHONPATH=src python -m unittest discover tests
```

## Package Layout

| Path | Role |
|---|---|
| `src/jorjor/` | Source package for environments, models, training helpers, probes, and audit utilities. |
| `scripts/` | Reproduction and report-building entry points. |
| `configs/` | Data, training, and probe configurations. |
| `tests/` | Unit and smoke tests for source code and static paper outputs. |
| `artifacts/` | Static paper evidence, claim-audit outputs, figure exports, and provenance packs. |
| `supplement/` | Reviewer-facing claim ledger, result index, table/figure provenance, and archive-boundary notes. |
| `tools/` | Dependency-light integrity and static-artifact checks. |

## Main Evidence Pointers

- Camera-ready five-run and exact-posterior addendum: `supplement/CAMERA_READY_REBUTTAL_ADDENDUM.md`
- Camera-ready verified figure data and exports: `artifacts/camera_ready_rebuttal/`
- Claim gate matrix: `artifacts/claim_audits/k.6.6_current_product_audit_pack/claim_gate_matrix.md`
- Product audit report: `artifacts/claim_audits/k.6.6_current_product_audit_pack/diagnostic_report.md`
- Poker predictive link: `artifacts/claim_audits/k.5.1_full_poker_predictive_link/`
- Composition-boundedness: `artifacts/claim_audits/k.5.2_composition_boundedness/`
- Known-control CBPS validation: `artifacts/claim_audits/k.5.0_synthetic_cbps_control/`
- Historical submission figure exports: `artifacts/figures/`

## Claim Boundary

The included evidence supports a bounded interpretation. The submitted three-run audit supplies the predictive and early-gate evidence used for scoped CBPS, while the fresh five-run panel separately shows a stable conditional-composition bound; it does not create a new joint five-run CBPS label. S7 is a post-decision target-stratified sensitivity, and the exact-posterior addendum shows a small pointwise gain with weak cross-hand pairwise geometry. None of these rows establishes faithful posterior tracking or a causal belief mechanism.
