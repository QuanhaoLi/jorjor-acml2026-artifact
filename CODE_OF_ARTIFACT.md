
# Code of Artifact

## What This Artifact Supports

This artifact supports the ACML 2026 conference-track paper on interpreting positive hidden-state range probes in imperfect-information autoregressive models.

The package is intended to let reviewers inspect:

- the implementation of the JorJor audit and probe utilities;
- experiment and probe configurations;
- table/figure source packs used by the paper;
- static claim-audit outputs, diagnostic reports, and source manifests;
- local smoke/unit tests and package hygiene checks.

## Claims Supported by Included Static Artifacts

- Positive but bounded poker range-probe evidence.
- Action/value-controlled range signal in selected seeds.
- A full-composition control using pre-decision history features conditional on target-side A/V nuisances, plus a post-decision target-stratified matched sensitivity that agrees within fixed action/value target strata.
- Known-control and oracle rows demonstrating that the audit can accept posterior-sensitive states.
- CBPS as a bounded interpretation below ordered posterior tracking.
- Claim ledgers that separate strict verdicts, refined interpretations, and forbidden stronger claims.

Some frozen static outputs retain the historical label `belief_like_residual_supported`. In the camera-ready claim contract, that legacy name means only `realized_label_controls_passed`; it does not establish posterior fidelity and cannot yield integrated `belief_like_supported` without independent pointwise and predeclared pairwise-posterior fidelity gates.

## Claims Not Made

- The poker AR model learns opponent-range belief.
- Probe accuracy proves a causal belief mechanism.
- Synthetic or oracle controls upgrade the poker case label.
- CBPS is equivalent to posterior tracking.
- Projection-level or readout evidence is activation-patching evidence.

## Excluded From This Package

- `.git` history and local development metadata.
- `.venv` and local package installations.
- Python caches and `.DS_Store` files.
- Model checkpoints, `.pt`, `.pth`, `.ckpt`, and `.safetensors` files.
- Full Hugging Face model directories.
- Large ad hoc smoke outputs and remote-machine logs not needed for paper evidence.

## External Data Boundary

Long-run regeneration may require an external archive root containing raw checkpoints, extracted probe caches, and training outputs. This package documents the expected commands and static outputs, but it intentionally avoids bundling large training artifacts.

## Package Hygiene

The submitted package is intended to be reviewer-facing. It excludes local operating-system metadata, Python caches, local virtual environments, raw model checkpoints, and development-only files. Run `python tools/check_artifact.py` to verify the included hygiene checks.
