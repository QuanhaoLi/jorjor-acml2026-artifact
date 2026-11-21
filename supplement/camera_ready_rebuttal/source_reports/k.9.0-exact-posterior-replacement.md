
# k.9.0 Exact-Posterior Paper Replacement

## Safe ACML Position

The exact known-teacher posterior audit replaces the binary range-label proxy
with proper-distribution evidence. It supports a small metric-specific hidden
increment under pointwise proper scores, but calibrated matched-composition
posterior tracking remains unsupported.

## Abstract Replacement Sentence

```text
An exact 169-way posterior audit under the known teacher finds a small,
five-run-stable proper-score increment from residual hidden state beyond
action/value and composition controls; however, matched-composition posterior
changes are severely compressed and do not establish a reliable improvement in pairwise JS calibration,
so the evidence does not support faithful ordered posterior tracking.
```

## Main Results Replacement

Replace the range-label proxy paragraph with:

```text
We replace the binary same/different range-label proxy with the exact 169-way
opponent posterior under the known deterministic teacher. All readouts use the
same exact card-removal prior as a fixed offset. Adding the locked residual
hidden state to the action/value+composition readout improves held-out KL by
0.00843 nats on average across five independent runs (95% run-level t interval
[0.00669, 0.01018]); JS and Brier improve consistently as well. This establishes
a small linear proper-score increment beyond the specified controls. The
stronger matched-composition test is not passed, however. Predicted and exact
pairwise posterior JS have mean correlation 0.129, but the mean slope is only
0.0061, indicating severe compression, and the pair-JS MAE improvement over
action/value+composition is 0.00015 with interval [-0.00016, 0.00046]. We
therefore classify the exact-posterior evidence as mixed: metric-specific
posterior information is recoverable, but calibrated matched-pair posterior
tracking is not supported.
```

## Main Table

| Evidence | Five-run result | Allowed interpretation |
|---|---|---|
| KL/CE increment over A/V+C | 0.008431 [0.006687, 0.010175] | proper-score increment supported |
| JS increment over A/V+C | 0.002374 [0.001848, 0.002901] | proper-score increment supported |
| Brier increment over A/V+C | 0.000040 [0.000033, 0.000047] | proper-score increment supported |
| Matched pair Pearson | 0.128961 [0.089732, 0.168191] | weak positive association |
| Matched pair slope | 0.006107 [0.004291, 0.007923] | severely compressed changes |
| Pair-JS MAE improvement | 0.000152 [-0.000159, 0.000463] | calibration improvement inconclusive |
| Final label | `mixed_exact_posterior_evidence` | no faithful posterior-tracking claim |

## Oracle Replacement

The old poker `oracle_top10=1` row is a one-hot class-identity gate check, not a
posterior calculation. Replace its main-text role with the k.9 identity oracle:

```text
On the same exact-posterior targets and matched pair list, q=b yields zero KL,
JS, Brier, and pair-JS error with unit correlation and slope, verifying that the
audit can accept a perfectly calibrated posterior representation.
```

The old one-hot row may remain appendix provenance but should not be called an
exact posterior oracle.

## Limitation Replacement

Replace “the poker boundedness proxy uses range-label gaps rather than exact
posterior JS” with:

```text
The poker posterior is exact only relative to the known uniform-deck,
deterministic-teacher generator and the declared player information set. It is
not an oracle for unknown or strategic opponents. Moreover, the observed
proper-score increment does not establish calibrated posterior dynamics or a
causal behavior mechanism.
```

## Gate Consequence

Do not silently upgrade or erase the existing categorical composition result.
The combined evidence is:

```text
weak probes are variable;
categorical composition gates are robustly negative;
exact posterior proper scores show a small conditional increment;
exact matched-pair calibration remains unsupported.
```

The correct label is mixed exact-posterior evidence, not `belief_like_supported`
and not proof that no posterior information exists.

