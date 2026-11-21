# Limit AR Paper Evidence Pack

## Core Claim

Across three 10k independently trained 4-layer Limit Hold'em autoregressive models and one 50k same-architecture robustness run, hidden states linearly expose opponent-hand range information from betting history, and this range signal remains after linearly controlling for action and value labels.

## Core 4-Layer Evidence

| group | history delta mean | history delta min | full delta mean | full delta min |
|---|---:|---:|---:|---:|
| v0.8.4 seed-a/b/c | 0.012520 | 0.009489 | 0.015050 | 0.009792 |

## 50k Scale Robustness

| group | seed count | history delta mean | history delta min | full delta mean | full delta min |
|---|---:|---:|---:|---:|---:|
| 50k 4-layer seed-d/f/g | 3 | 0.011963 | 0.008666 | 0.010596 | 0.009689 |

## Robustness Rows

| row | policy acc | range top10 | history delta | full delta | interpretation |
|---|---:|---:|---:|---:|---|
| 50k 4-layer seed-d | 0.822993 | 0.158393 | 0.017758 | 0.012335 | positive |
| 50k 4-layer seed-f | 0.838175 | 0.151471 | 0.008666 | 0.009764 | positive |
| 50k 4-layer seed-g | 0.840634 | 0.150999 | 0.009465 | 0.009689 | positive |
| 10k 8-layer default seed-e | 0.598698 | 0.126590 | -0.001703 | -0.006209 | boundary |
| 10k 8-layer stabilized seed-e | 0.840060 | 0.143615 | -0.007211 | 0.002404 | boundary |

## Boundary

Depth robustness is not established: the stabilized 8-layer history-only residual range signal does not beat the action+value-only nuisance baseline under the tested recipe.

## Limitations

- This is linear nuisance-control evidence, not causal independence from the training objective.
- Value probe evidence is weak and should not be a main positive claim.
- Depth robustness is not established by v0.9.1 or v0.9.2.
