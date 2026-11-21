# k.3.3 G1-G4 Threshold Manifest

| gate | name | statistic | pass rule | CI/permutation rule | source |
|---|---|---|---|---|---|
| G1 | naive probe positivity | positive intermediate-layer range probe | Pass if locked-layer or predeclared layer probe is above the stated prior baseline. | Use paired/bucket bootstrap CI when archived; otherwise mark descriptive. | checkpoints/remote_archive/v100s/881534f/outputs/tables/limit_ar_no_range_fresh50_control_pack.json |
| G2 | label permutation control | real labels minus shuffled labels | Pass if real-label delta is positive against global and within-bucket shuffled labels. | Permutation label controls plus positive minimum delta. | checkpoints/remote_archive/v100s/881534f/outputs/tables/limit_ar_no_range_fresh50_control_pack.json |
| G3 | no-observation/prior control | ordered history signal above prior/action-position baselines | Pass if ordered-history row beats no-observation or round-position-action prior with lower CI above zero. | Paired bootstrap CI over hands or matched buckets. | checkpoints/remote_archive/v100s/1d9574f/outputs/tables/limit_ar_no_range_history_composition_pack.json |
| G4 | history destruction control | normal history minus destroyed history | Pass if normal history beats drop/shuffle/bagged controls with lower CI above zero. | Paired bootstrap CI over hand ids. | checkpoints/remote_archive/v100s/881534f/outputs/tables/limit_ar_no_range_fresh50_control_pack.json |
