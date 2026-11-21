# No-Range Locked-Layer Bootstrap CI

## Rule

- Primary bootstrap rows resample hand clusters using paired `top10_correct` deltas.
- Rows fail construction unless residual and nuisance correctness artifacts align by `point_id`.

## CI Rows

| seed | layer | hands | points | residual top10 | baseline top10 | cluster mean delta | cluster lower | cluster upper |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| seed_k | sequence_hidden | 10000 | 49480 | 0.144543 | 0.134600 | 0.010542 | 0.006139 | 0.014913 |
| seed_k | encoder_layer_2 | 10000 | 49480 | 0.146403 | 0.134600 | 0.013216 | 0.008425 | 0.017571 |
| seed_l | sequence_hidden | 10000 | 49258 | 0.141946 | 0.134049 | 0.004033 | -0.001366 | 0.008349 |
| seed_l | encoder_layer_2 | 10000 | 49258 | 0.140546 | 0.134049 | 0.001844 | -0.002686 | 0.007261 |
| seed_m | sequence_hidden | 10000 | 49132 | 0.142188 | 0.135859 | 0.010398 | 0.006155 | 0.014868 |
| seed_m | encoder_layer_2 | 10000 | 49132 | 0.142453 | 0.135859 | 0.006417 | 0.002138 | 0.011823 |

## Summary

| layer | mean delta | min lower | all lower above zero |
|---|---:|---:|---|
| sequence_hidden | 0.008324 | -0.001366 | False |
| encoder_layer_2 | 0.007159 | -0.002686 | False |
