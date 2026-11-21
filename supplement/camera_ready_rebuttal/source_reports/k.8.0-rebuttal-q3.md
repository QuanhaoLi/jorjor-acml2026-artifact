
# k.8.0 Rebuttal - Question 3

## Reviewer Concern

```text
The main results are based on only three training seeds, and one seed gives an
inconclusive result, suggesting some variability across training runs.
```

## Recommended Response

Thank you for highlighting training-run variability. We agree that the
original three-run panel was too small to characterize it. We therefore ran a
fresh panel of five predeclared independent end-to-end runs in one frozen
environment. Each run uses a distinct model, teacher-data, and hand-split seed,
while keeping the architecture, 50,000-hand budget, optimization schedule, and
locked `encoder_layer_2` unchanged. We report every run, within-run clustered
bootstrap intervals, and an equal-weight 95% Student-t interval over the five
run-level estimates.

The added runs confirm variability at the weaker action/value-controlled gate:
3/5 runs have positive within-run intervals and 2/5 are inconclusive; the
across-run mean delta is 0.0080 (SD 0.0078, 95% t interval
[-0.0017, 0.0177]). We therefore revised the manuscript to avoid implying that
this weaker residual signal is stable across runs. Crucially, the
claim-determining composition results are consistent in all five runs. G6 uses
pre-decision composition features conditional on target-side A/V nuisances;
its residual delta is negative in 5/5 runs, with across-run mean -0.0605 and
95% t interval [-0.0611, -0.0599]. The post-decision target-stratified S7
sensitivity is also negative in 5/5 runs, with mean -0.0745 and interval
[-0.0805, -0.0684]. Thus the additional runs validate the reviewer's
variability concern for the weaker probe evidence while strengthening the
central diagnostic conclusion: the no-range poker evidence supports a robust
composition-shortcut diagnosis, not ordered belief tracking.

## Required Manuscript Changes

- Replace `two of three seeds and one inconclusive seed` with the five-run G5
  counts, mean, SD, and seed-level interval.
- Do not describe G5 as stable or uniformly positive.
- Expand the G6 and target-stratified S7 main rows/figure from three to five runs.
- State that a run changes model, generated teacher data, and split seeds; use
  `independent end-to-end runs` rather than initialization-only seeds.
- Report the training run as the between-run statistical unit and retain the
  hand/bucket bootstrap only as within-run uncertainty.
- Preserve the strict claim boundary: five-run robustness does not license a
  belief-tracking or causal-mechanism claim.

## Compact Result Table

| Gate | Positive | Negative | Inconclusive | Across-run mean | 95% t interval |
|---|---:|---:|---:|---:|---|
| G5 A/V residual | 3 | 0 | 2 | 0.008004 | [-0.001675, 0.017683] |
| G6 conditional composition | 0 | 5 | 0 | -0.060525 | [-0.061127, -0.059923] |
| S7 target-stratified matched sensitivity | 0 | 5 | 0 | -0.074461 | [-0.080476, -0.068446] |

Source product checksum:

```text
00236dc927bb4ae24e13c2e3cdca72a5e3d5f1a080e45455a38c184fd80acf8c
```

