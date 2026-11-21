# Final Diagnostic Tables

## Rule

- Main rows require paired/bucket bootstrap CI.
- `encoder_layer_2` is the locked candidate layer.
- Best-layer selection is appendix-only and absent from these main rows.

## Rows

| role | gate | seed | layer | layer role | delta | CI lower | CI upper | unit | label |
|---|---|---|---|---|---:|---:|---:|---|---|
| boundary | no_range_action_value_residual_ci | seed_k | sequence_hidden | primary_final_boundary | 0.010542 | 0.006139 | 0.014913 | hand_id | positive_ci_lower_above_zero |
| main | no_range_action_value_residual_ci | seed_k | encoder_layer_2 | locked_candidate | 0.013216 | 0.008425 | 0.017571 | hand_id | positive_ci_lower_above_zero |
| boundary | no_range_action_value_residual_ci | seed_l | sequence_hidden | primary_final_boundary | 0.004033 | -0.001366 | 0.008349 | hand_id | ci_overlaps_zero |
| main | no_range_action_value_residual_ci | seed_l | encoder_layer_2 | locked_candidate | 0.001844 | -0.002686 | 0.007261 | hand_id | ci_overlaps_zero |
| boundary | no_range_action_value_residual_ci | seed_m | sequence_hidden | primary_final_boundary | 0.010398 | 0.006155 | 0.014868 | hand_id | positive_ci_lower_above_zero |
| main | no_range_action_value_residual_ci | seed_m | encoder_layer_2 | locked_candidate | 0.006417 | 0.002138 | 0.011823 | hand_id | positive_ci_lower_above_zero |
| main | composition_nuisance:ordered_residual_vs_action_value | seed_k | encoder_layer_2 | locked_candidate | 0.013216 | 0.008764 | 0.017700 | hand_id | positive_ci_lower_above_zero |
| main | composition_nuisance:composition_only_vs_action_value | seed_k | nuisance | nuisance_baseline | 0.011597 | 0.007053 | 0.016758 | hand_id | positive_ci_lower_above_zero |
| main | composition_nuisance:action_value_composition_only_vs_action_value | seed_k | nuisance | nuisance_baseline | 0.028161 | 0.022744 | 0.033565 | hand_id | positive_ci_lower_above_zero |
| main | composition_nuisance:composition_residual_vs_action_value | seed_k | encoder_layer_2 | locked_candidate | -0.030659 | -0.035011 | -0.025019 | hand_id | negative_ci_upper_below_zero |
| main | composition_nuisance:composition_residual_vs_action_value_composition | seed_k | encoder_layer_2 | locked_candidate | -0.058820 | -0.064841 | -0.054209 | hand_id | negative_ci_upper_below_zero |
| main | composition_nuisance:ordered_residual_vs_action_value | seed_l | encoder_layer_2 | locked_candidate | 0.001844 | -0.003296 | 0.006614 | hand_id | ci_overlaps_zero |
| main | composition_nuisance:composition_only_vs_action_value | seed_l | nuisance | nuisance_baseline | 0.014013 | 0.010078 | 0.017791 | hand_id | positive_ci_lower_above_zero |
| main | composition_nuisance:action_value_composition_only_vs_action_value | seed_l | nuisance | nuisance_baseline | 0.027356 | 0.022687 | 0.031527 | hand_id | positive_ci_lower_above_zero |
| main | composition_nuisance:composition_residual_vs_action_value | seed_l | encoder_layer_2 | locked_candidate | -0.033518 | -0.038860 | -0.028316 | hand_id | negative_ci_upper_below_zero |
| main | composition_nuisance:composition_residual_vs_action_value_composition | seed_l | encoder_layer_2 | locked_candidate | -0.060874 | -0.066681 | -0.055651 | hand_id | negative_ci_upper_below_zero |
| main | composition_nuisance:ordered_residual_vs_action_value | seed_m | encoder_layer_2 | locked_candidate | 0.006417 | 0.001253 | 0.011241 | hand_id | positive_ci_lower_above_zero |
| main | composition_nuisance:composition_only_vs_action_value | seed_m | nuisance | nuisance_baseline | 0.014728 | 0.009193 | 0.018909 | hand_id | positive_ci_lower_above_zero |
| main | composition_nuisance:action_value_composition_only_vs_action_value | seed_m | nuisance | nuisance_baseline | 0.026229 | 0.020862 | 0.030928 | hand_id | positive_ci_lower_above_zero |
| main | composition_nuisance:composition_residual_vs_action_value | seed_m | encoder_layer_2 | locked_candidate | -0.040589 | -0.045635 | -0.036109 | hand_id | negative_ci_upper_below_zero |
| main | composition_nuisance:composition_residual_vs_action_value_composition | seed_m | encoder_layer_2 | locked_candidate | -0.066818 | -0.072105 | -0.061806 | hand_id | negative_ci_upper_below_zero |

## Gate Summary

| item | value |
|---|---|
| all main rows have CI | `True` |
| uses best-layer selection | `False` |
| locked candidate layer | `encoder_layer_2` |
| final hidden boundary layer | `sequence_hidden` |
