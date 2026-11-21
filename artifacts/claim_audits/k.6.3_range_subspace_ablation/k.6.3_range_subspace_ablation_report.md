# k.6.3 Range-Subspace Ablation Stress Report

This is a representation-edit stress test. It is not a causal mediation proof.

final_diagnostic_label: `range_subspace_behavior_footprint_not_supported`

| seed | label | probe top10 | rank | range policy JS | random policy JS | policy delta | range value delta | random value delta | value delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed_k | `range_subspace_behavior_footprint_not_supported` | 0.123545 | 16 | 0.00016428 | 0.00373479 | -0.00357051 | 0.04468262 | 0.18751094 | -0.14282832 |
| seed_l | `range_subspace_behavior_footprint_not_supported` | 0.128487 | 16 | 0.00030981 | 0.00608813 | -0.00577831 | 0.02528101 | 0.17278224 | -0.14750123 |
| seed_m | `range_subspace_behavior_footprint_not_supported` | 0.124135 | 16 | 0.00016735 | 0.00303402 | -0.00286667 | 0.01698201 | 0.21903873 | -0.20205672 |

## Aggregate

| metric | mean | lower | upper |
|---|---:|---:|---:|
| probe_top10_accuracy | 0.12538886 | 0.12354486 | 0.12848674 |
| range_ablation_policy_js | 0.00021382 | 0.00016428 | 0.00030981 |
| range_ablation_value_abs_delta | 0.02898188 | 0.01698201 | 0.04468262 |
| random_ablation_policy_js_mean | 0.00428565 | 0.00303402 | 0.00608813 |
| random_ablation_value_abs_delta_mean | 0.19311064 | 0.17278224 | 0.21903873 |
| range_minus_random_policy_js_delta | -0.00407183 | -0.00577831 | -0.00286667 |
| range_minus_random_value_delta | -0.16412876 | -0.20205672 | -0.14282832 |

## Claim Discipline

Top range-probe subspace ablation can be compared with same-rank random subspace ablation as a behavior-footprint stress test. It is not a causal mediation proof.

Forbidden claim: This subspace ablation proves the range-correlated hidden directions causally drive policy.
