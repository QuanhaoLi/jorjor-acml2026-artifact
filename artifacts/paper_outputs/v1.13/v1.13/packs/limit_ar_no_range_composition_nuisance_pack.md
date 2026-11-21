# No-Range Composition Nuisance Pack

## Paired Deltas

| seed | comparison | layer | real top10 | baseline top10 | mean delta | lower | upper |
|---|---|---|---:|---:|---:|---:|---:|
| seed_k | ordered_residual_vs_action_value | encoder_layer_2 | 0.146403 | 0.134600 | 0.013216 | 0.008764 | 0.017700 |
| seed_k | composition_only_vs_action_value | nuisance | 0.149070 | 0.134600 | 0.011597 | 0.007053 | 0.016758 |
| seed_k | action_value_composition_only_vs_action_value | nuisance | 0.166390 | 0.134600 | 0.028161 | 0.022744 | 0.033565 |
| seed_k | composition_residual_vs_action_value | encoder_layer_2 | 0.122413 | 0.134600 | -0.030659 | -0.035011 | -0.025019 |
| seed_k | composition_residual_vs_action_value_composition | encoder_layer_2 | 0.122413 | 0.166390 | -0.058820 | -0.064841 | -0.054209 |
| seed_l | ordered_residual_vs_action_value | encoder_layer_2 | 0.140546 | 0.134049 | 0.001844 | -0.003296 | 0.006614 |
| seed_l | composition_only_vs_action_value | nuisance | 0.152726 | 0.134049 | 0.014013 | 0.010078 | 0.017791 |
| seed_l | action_value_composition_only_vs_action_value | nuisance | 0.165455 | 0.134049 | 0.027356 | 0.022687 | 0.031527 |
| seed_l | composition_residual_vs_action_value | encoder_layer_2 | 0.121666 | 0.134049 | -0.033518 | -0.038860 | -0.028316 |
| seed_l | composition_residual_vs_action_value_composition | encoder_layer_2 | 0.121666 | 0.165455 | -0.060874 | -0.066681 | -0.055651 |
| seed_m | ordered_residual_vs_action_value | encoder_layer_2 | 0.142453 | 0.135859 | 0.006417 | 0.001253 | 0.011241 |
| seed_m | composition_only_vs_action_value | nuisance | 0.152284 | 0.135859 | 0.014728 | 0.009193 | 0.018909 |
| seed_m | action_value_composition_only_vs_action_value | nuisance | 0.166511 | 0.135859 | 0.026229 | 0.020862 | 0.030928 |
| seed_m | composition_residual_vs_action_value | encoder_layer_2 | 0.114141 | 0.135859 | -0.040589 | -0.045635 | -0.036109 |
| seed_m | composition_residual_vs_action_value_composition | encoder_layer_2 | 0.114141 | 0.166511 | -0.066818 | -0.072105 | -0.061806 |

## Summary

| metric | value |
|---|---:|
| ordered residual min lower | -0.003296 |
| composition-only min lower | 0.007053 |
| action/value+composition min lower | 0.020862 |
| composition-residual vs action/value min lower | -0.045635 |
| composition-residual vs action/value+composition min lower | -0.072105 |
| composition-residual all lower above zero | False |
