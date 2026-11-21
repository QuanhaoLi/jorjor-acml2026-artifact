# Camera-Ready Rebuttal Addendum

## Scope

This addendum records the two evidence packages incorporated after review while preserving the paper's original CBPS thesis and claim-gated structure:

1. the fresh five-run robustness audit of realized-hand gates G5/G6 and the target-stratified S7 sensitivity; and
2. the exact 169-way teacher-induced-posterior audit.

These packages sharpen the evidence boundary. They do not upgrade the audited transformer to posterior tracking and do not establish a causal belief mechanism.

## Five-Run Robustness

The homogeneous panel uses runs `k`--`o`, with distinct model, teacher-data, and split seeds. G5/G6 use 10,000 hand-cluster bootstrap draws; S7 uses 10,000 matched-bucket draws. G6 uses pre-decision composition features conditional on target-side A/V nuisances. S7 additionally fixes current action and terminal return as matching strata; neither variable is a model input. Thus G6 is conditional and S7 is post-decision target-stratified, rather than either being a pure decision-time estimand. Between-run intervals are equal-weight Student-t intervals over five run estimates.

| Gate | Positive | Negative | Inconclusive | Mean | 95% run-level CI |
|---|---:|---:|---:|---:|---|
| G5 A/V residual | 3 | 0 | 2 | 0.008004 | [-0.001675, 0.017683] |
| G6 conditional composition | 0 | 5 | 0 | -0.060525 | [-0.061127, -0.059923] |
| S7 target-stratified match | 0 | 5 | 0 | -0.074461 | [-0.080476, -0.068446] |

Five-run result checksum: `00236dc927bb4ae24e13c2e3cdca72a5e3d5f1a080e45455a38c184fd80acf8c`.

## Exact Teacher-Induced Posterior

The oracle enumerates all 1,326 physical opponent combinations, applies card removal and the frozen deterministic teacher likelihood, and aggregates to 169 canonical classes. Both readouts share the exact card-removal log-prior. The augmented readout adds only hidden features residualized against standardized A/V+composition features.

| Evidence | Mean | 95% run-level CI |
|---|---:|---|
| KL/CE improvement | 0.008431 | [0.006687, 0.010175] |
| JS improvement | 0.002374 | [0.001848, 0.002901] |
| Brier improvement | 0.000040 | [0.000033, 0.000047] |
| Pair-JS Pearson | 0.128961 | [0.089732, 0.168191] |
| Pair-JS slope | 0.006107 | [0.004291, 0.007923] |
| Pair-JS MAE improvement | 0.000152 | [-0.000159, 0.000463] |

Because the Pair-JS MAE interval crosses zero, this analysis does not establish a reliable pairwise-distance calibration improvement; it does not establish that an improvement is absent. These are fixed cross-hand pairs, not legal temporal transitions, so the result is pairwise posterior geometry rather than a direct update-fidelity test.

The original support rule required at least 100 buckets and 2,000 pairs, which was structurally unattainable with 30 buckets and the fixed 64-pair cap. The k.9.0.1 feasibility repair changed only the support rule to at least 20 buckets, 1,500 nonzero pairs, and 1,000 hand nodes; no posterior, pair, score, or interval changed. The pairwise result is therefore reported as a support-repaired sensitivity.

Support-repaired result checksum: `6cafa8302023c5760103c340c2f1db97c488c01853269b2b8f7723b6e031e3f5`.

## Figure and Source Map

- Main Figure 2 data and plot: `artifacts/camera_ready_rebuttal/five_run_and_exact_posterior_data.json` and `figure_five_run_realized_hand_gates.*`.
- Main Figure 3 data and plot: the same JSON and `figure_exact_teacher_posterior_evidence.*`.
- Former main posterior-control visualization, now supplementary: `artifacts/figures/figure_5_oracle_validation_ordered_vs_shortcut_paper_flat.png`.
- Frozen narrative reports: `supplement/camera_ready_rebuttal/source_reports/`.
- Minimal code snapshot: `supplement/camera_ready_rebuttal/code/`.

The submitted lightweight archive does not bundle the multi-gigabyte raw checkpoint archives or matched-pair arrays. The figure script therefore plots only verified run-level outputs; it does not reconstruct pair observations from aggregate statistics.

Deterministic PDF hashes:

- Figure 2: `476db4c464977b759d73d0d194b550cdadbaca5232b48653cb0f0a6d97101ee8`.
- Figure 3: `195a5813edfbc972595f26b2cafa2dd6ec9ba8664dde0984273f6b83f40d5105`.
