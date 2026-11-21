# k.4.0 Behavior-Required POMDP Report

This is a non-poker behavior-only learned positive-control task. It does not upgrade the poker AR claim.

behavior_only_belief_supported: `False`
behavior_only_partial_support: `True`

| mode | target | verdict | action acc | trajectory corr | matched buckets | matched pairs | rep JS | lower | upper | matched corr |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_posterior_state | none_oracle_state | behavior_only_belief_like_supported | 0.000000 | 1.0 | 25 | 4096 | 0.452339 | 0.411144 | 0.484963 | 0.9999999999999999 |
| composition_shortcut_state | none_shortcut_state | composition_shortcut_supported | 0.000000 | 0.013998885826896667 | 25 | 4096 | 0.000000 | 0.000000 | 0.000000 | None |
| behavior_only_gru_q_head | full_action_value_vector_behavior_only | inconclusive | 1.000000 | -0.0032758559214904645 | 25 | 4096 | 0.400116 | 0.362362 | 0.430213 | 0.9166663894232818 |
| behavior_only_gru_hidden_probe | action_value_only | inconclusive | 1.000000 | 0.40547629667072077 | 25 | 4096 | 0.512732 | 0.460899 | 0.556440 | 0.9237290183829265 |
| behavior_only_gru_q_probe | action_value_only | inconclusive | 1.000000 | -0.010097236079414856 | 25 | 4096 | 0.387302 | 0.354560 | 0.417374 | 0.9117872078088539 |
