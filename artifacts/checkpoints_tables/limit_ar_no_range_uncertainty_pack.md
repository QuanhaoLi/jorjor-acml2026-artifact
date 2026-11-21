# No-Range Probe Uncertainty

## Residual Rows

| seed | feature control | layer | count | mean top10 | std top10 | min | max |
|---|---|---|---:|---:|---:|---:|---:|
| seed_h | residualize_action_value | embedding_final | 4 | 0.128291 | 0.003546 | 0.123673 | 0.131224 |
| seed_h | residualize_action_value | encoder_layer_0 | 4 | 0.141403 | 0.003864 | 0.136429 | 0.145816 |
| seed_h | residualize_action_value | encoder_layer_1 | 4 | 0.143393 | 0.002938 | 0.139184 | 0.146020 |
| seed_h | residualize_action_value | encoder_layer_2 | 4 | 0.142041 | 0.003114 | 0.138469 | 0.144898 |
| seed_h | residualize_action_value | encoder_layer_3 | 4 | 0.141020 | 0.005621 | 0.135306 | 0.148163 |
| seed_h | residualize_action_value | sequence_hidden | 4 | 0.141020 | 0.005621 | 0.135306 | 0.148163 |
| seed_i | residualize_action_value | embedding_final | 4 | 0.130725 | 0.004215 | 0.126594 | 0.135469 |
| seed_i | residualize_action_value | encoder_layer_0 | 4 | 0.137101 | 0.004410 | 0.131490 | 0.142099 |
| seed_i | residualize_action_value | encoder_layer_1 | 4 | 0.139753 | 0.004849 | 0.133531 | 0.145058 |
| seed_i | residualize_action_value | encoder_layer_2 | 4 | 0.141130 | 0.003855 | 0.135673 | 0.144140 |
| seed_i | residualize_action_value | encoder_layer_3 | 4 | 0.132536 | 0.003335 | 0.128328 | 0.136489 |
| seed_i | residualize_action_value | sequence_hidden | 4 | 0.132536 | 0.003335 | 0.128328 | 0.136489 |
| seed_j | residualize_action_value | embedding_final | 4 | 0.127662 | 0.005325 | 0.121166 | 0.134108 |
| seed_j | residualize_action_value | encoder_layer_0 | 4 | 0.138694 | 0.002710 | 0.135738 | 0.141241 |
| seed_j | residualize_action_value | encoder_layer_1 | 4 | 0.141470 | 0.002921 | 0.138388 | 0.144400 |
| seed_j | residualize_action_value | encoder_layer_2 | 4 | 0.143279 | 0.002664 | 0.140630 | 0.145623 |
| seed_j | residualize_action_value | encoder_layer_3 | 4 | 0.143177 | 0.002060 | 0.141547 | 0.145929 |
| seed_j | residualize_action_value | sequence_hidden | 4 | 0.143177 | 0.002060 | 0.141547 | 0.145929 |

## Baseline Rows

| seed | feature control | layer | count | mean top10 | std top10 | min | max |
|---|---|---|---:|---:|---:|---:|---:|
| seed_h | action_value_only | nuisance | 4 | 0.136684 | 0.001410 | 0.134898 | 0.138265 |
| seed_i | action_value_only | nuisance | 4 | 0.129323 | 0.005033 | 0.122616 | 0.133939 |
| seed_j | action_value_only | nuisance | 4 | 0.135993 | 0.000536 | 0.135534 | 0.136554 |

## Paired Deltas

| seed | layer | count | mean delta | std delta | min | max |
|---|---|---:|---:|---:|---:|---:|
| seed_h | embedding_final | 4 | -0.008393 | 0.003100 | -0.011224 | -0.005510 |
| seed_h | encoder_layer_0 | 4 | 0.004719 | 0.004502 | -0.000714 | 0.009388 |
| seed_h | encoder_layer_1 | 4 | 0.006709 | 0.003498 | 0.002041 | 0.009592 |
| seed_h | encoder_layer_2 | 4 | 0.005357 | 0.002307 | 0.003265 | 0.007959 |
| seed_h | encoder_layer_3 | 4 | 0.004337 | 0.005223 | 0.000408 | 0.011735 |
| seed_h | sequence_hidden | 4 | 0.004337 | 0.005223 | 0.000408 | 0.011735 |
| seed_i | embedding_final | 4 | 0.001403 | 0.007983 | -0.005713 | 0.012853 |
| seed_i | encoder_layer_0 | 4 | 0.007778 | 0.006209 | -0.000816 | 0.013873 |
| seed_i | encoder_layer_1 | 4 | 0.010430 | 0.007442 | 0.001224 | 0.016628 |
| seed_i | encoder_layer_2 | 4 | 0.011808 | 0.001568 | 0.009589 | 0.013057 |
| seed_i | encoder_layer_3 | 4 | 0.003213 | 0.007136 | -0.001224 | 0.013873 |
| seed_i | sequence_hidden | 4 | 0.003213 | 0.007136 | -0.001224 | 0.013873 |
| seed_j | embedding_final | 4 | -0.008331 | 0.004880 | -0.014369 | -0.002446 |
| seed_j | encoder_layer_0 | 4 | 0.002701 | 0.002763 | 0.000204 | 0.005707 |
| seed_j | encoder_layer_1 | 4 | 0.005477 | 0.002421 | 0.002853 | 0.008051 |
| seed_j | encoder_layer_2 | 4 | 0.007286 | 0.002142 | 0.005095 | 0.009273 |
| seed_j | encoder_layer_3 | 4 | 0.007184 | 0.001661 | 0.006012 | 0.009579 |
| seed_j | sequence_hidden | 4 | 0.007184 | 0.001661 | 0.006012 | 0.009579 |

## Decision Inputs

- Use paired deltas, not raw residual scores, for the no-range claim gate.
- Do not use this table alone as a claim; it measures probe-seed stability for rows selected by v1.0.2-v1.0.4.
