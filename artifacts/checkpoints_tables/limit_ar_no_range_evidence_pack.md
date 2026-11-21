# No-Range-Supervision Evidence Pack

## Main Rule

Primary pass/fail uses `sequence_hidden`. `best_over_layers` is descriptive only.

## Core Rows

| seed | policy acc | range supervised | diagnostic range top10 | history seq delta | history best delta | full seq delta | full best delta |
|---|---:|---|---:|---:|---:|---:|---:|
| seed_h | 0.833061 | False | 0.041020 | 0.000918 | 0.003265 | -0.002041 | 0.003673 |
| seed_i | 0.837295 | False | 0.056921 | -0.001224 | 0.009589 | -0.003264 | 0.010813 |
| seed_j | 0.821461 | False | 0.051768 | 0.007031 | 0.008968 | -0.004484 | 0.000000 |

## Label Controls

| row | sequence_hidden top10 | best top10 | best layer |
|---|---:|---:|---|
| real | 0.140714 | 0.140714 | encoder_layer_3 |
| shuffle_global | 0.089592 | 0.096735 | encoder_layer_2 |
| shuffle_within_bucket | 0.097347 | 0.097347 | encoder_layer_3 |

## History Controls

| row | input control | feature control | sequence_hidden top10 | best top10 | best layer |
|---|---|---|---:|---:|---|
| drop_history | drop_history | none | 0.123061 | 0.123061 | encoder_layer_3 |
| shuffle_history_order | shuffle_history_order | none | 0.128776 | 0.135408 | encoder_layer_0 |
| bag_of_actions | bag_of_actions | none | 0.123571 | 0.128878 | embedding_final |
| no_observation | no_observation | none | 0.093061 | 0.096429 | encoder_layer_0 |
| round_position_action_prior | none | round_position_action_only | n/a | 0.122143 | nuisance |
| history_only_action_only | history_only | action_only | n/a | 0.127551 | nuisance |
