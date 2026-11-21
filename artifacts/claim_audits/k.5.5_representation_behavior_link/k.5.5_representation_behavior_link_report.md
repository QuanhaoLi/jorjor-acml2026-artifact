# k.5.5 Representation-To-Behavior Link Report

This report tests whether frozen hidden representations predict behavior targets beyond input-only public-composition features. It is not a causal mediation test.

final_diagnostic_label: `composition_bounded_behavior_link`

| seed | label | hidden acc | C_full linear acc | hidden delta | hidden lower | residual acc | residual delta | residual lower | hidden value delta | residual value delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed_k | `composition_bounded_behavior_link` | 0.846120 | 0.835631 | 0.010489 | 0.008569 | 0.597918 | -0.237712 | -0.242057 | 0.225321 | -0.310140 |
| seed_l | `composition_bounded_behavior_link` | 0.847822 | 0.838077 | 0.009745 | 0.008181 | 0.610378 | -0.227699 | -0.231759 | 0.333298 | -0.534923 |
| seed_m | `composition_bounded_behavior_link` | 0.838008 | 0.838415 | -0.000407 | -0.002157 | 0.637365 | -0.201050 | -0.204612 | 0.395974 | -0.607927 |

## Aggregate

| comparison | mean action delta | action lower | action upper | mean value delta | value lower | value upper |
|---|---:|---:|---:|---:|---:|---:|
| hidden_vs_C_full_linear | 0.006609 | -0.000407 | 0.010489 | 0.318198 | 0.225321 | 0.395974 |
| hidden_vs_C_full_lookup | -0.032636 | -0.038000 | -0.028117 | 3.713865 | 3.546632 | 3.933730 |
| hidden_vs_action_prior | 0.364062 | 0.359908 | 0.366653 | 0.805804 | 0.533196 | 0.975358 |
| residual_hidden_vs_C_full_linear | -0.222154 | -0.237712 | -0.209933 | -0.484330 | -0.607927 | -0.310140 |
| residual_hidden_vs_C_full_lookup | -0.261399 | -0.279992 | -0.238643 | 2.911337 | 2.678411 | 3.125772 |

## Claim Discipline

Frozen no-range poker AR hidden states are evaluated for action/value predictiveness against input-only public-composition baselines; this can support predictive CBPS language but cannot establish causal mediation or ordered belief tracking.

Forbidden claim: This report proves that range-correlated hidden dimensions causally drive policy.
