# Range-Supervised Positive-Control Audit Pack

## Summary

- final label: `belief_like_real_poker_positive_control`
- model family: range-supervised 50k 4-layer Limit AR
- main layer: locked `encoder_layer_2`

## Rows

| seed | gate | unit | real top10 | baseline top10 | mean delta | lower | upper |
|---|---|---|---:|---:|---:|---:|---:|
| seed_d | S5_action_value_residual | hand_id | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |
| seed_d | S6_composition_residual | hand_id | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |
| seed_d | S7_matched_composition | matched_composition_bucket | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |
| seed_f | S5_action_value_residual | hand_id | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |
| seed_f | S6_composition_residual | hand_id | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |
| seed_f | S7_matched_composition | matched_composition_bucket | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |
| seed_g | S5_action_value_residual | hand_id | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |
| seed_g | S6_composition_residual | hand_id | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |
| seed_g | S7_matched_composition | matched_composition_bucket | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |

## Gate Minimums

| gate | min lower |
|---|---:|
| S5 action/value residual | 1.000000 |
| S6 composition residual | 1.000000 |
| S7 matched_composition | 1.000000 |
