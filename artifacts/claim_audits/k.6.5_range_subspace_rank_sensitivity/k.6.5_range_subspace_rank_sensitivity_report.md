# k.6.5 Range-Subspace Rank-Sensitivity Report

This is a rank-sensitivity stress test for k.6.3. It is not a causal mediation proof.

final_diagnostic_label: `range_subspace_rank_sensitivity_not_supported`

## Per-Rank Aggregate

| rank | label | policy delta | value delta | probe top10 |
|---:|---|---:|---:|---:|
| 1 | `range_subspace_behavior_footprint_not_supported` | -0.00016060 | -0.02132972 | 0.125395 |
| 2 | `range_subspace_behavior_footprint_not_supported` | -0.00034360 | -0.04301355 | 0.125395 |
| 4 | `range_subspace_behavior_footprint_not_supported` | -0.00077286 | -0.05734651 | 0.125395 |
| 8 | `range_subspace_behavior_footprint_not_supported` | -0.00205350 | -0.08890461 | 0.125395 |
| 16 | `range_subspace_behavior_footprint_not_supported` | -0.00702155 | -0.17646536 | 0.125395 |
| 32 | `range_subspace_behavior_footprint_not_supported` | -0.01440968 | -0.24467192 | 0.125395 |

## Per-Seed Per-Rank

| seed | rank | label | policy delta | value delta | range policy JS | random policy JS |
|---|---:|---|---:|---:|---:|---:|
| seed_k | 1 | `range_subspace_behavior_footprint_not_supported` | -0.00015259 | -0.01456282 | 0.00000070 | 0.00015328 |
| seed_k | 2 | `range_subspace_behavior_footprint_not_supported` | -0.00018551 | -0.05109026 | 0.00000312 | 0.00018863 |
| seed_k | 4 | `range_subspace_behavior_footprint_not_supported` | -0.00107012 | -0.04285526 | 0.00000704 | 0.00107716 |
| seed_k | 8 | `range_subspace_behavior_footprint_not_supported` | -0.00251857 | -0.08242441 | 0.00004040 | 0.00255898 |
| seed_k | 16 | `range_subspace_behavior_footprint_not_supported` | -0.00479581 | -0.10796605 | 0.00016428 | 0.00496010 |
| seed_k | 32 | `range_subspace_behavior_footprint_not_supported` | -0.01471073 | -0.23559209 | 0.00086439 | 0.01557513 |
| seed_l | 1 | `range_subspace_behavior_footprint_not_supported` | -0.00013138 | -0.03854867 | 0.00000374 | 0.00013512 |
| seed_l | 2 | `range_subspace_behavior_footprint_not_supported` | -0.00029032 | -0.02909982 | 0.00000590 | 0.00029622 |
| seed_l | 4 | `range_subspace_behavior_footprint_not_supported` | -0.00073718 | -0.07534666 | 0.00000728 | 0.00074446 |
| seed_l | 8 | `range_subspace_behavior_footprint_not_supported` | -0.00177806 | -0.09802310 | 0.00009803 | 0.00187609 |
| seed_l | 16 | `range_subspace_behavior_footprint_not_supported` | -0.00821569 | -0.21638079 | 0.00031060 | 0.00852629 |
| seed_l | 32 | `range_subspace_behavior_footprint_not_supported` | -0.01231108 | -0.19060609 | 0.00116880 | 0.01347989 |
| seed_m | 1 | `range_subspace_behavior_footprint_not_supported` | -0.00019782 | -0.01087767 | 0.00001768 | 0.00021550 |
| seed_m | 2 | `range_subspace_behavior_footprint_not_supported` | -0.00055499 | -0.04885058 | 0.00001069 | 0.00056568 |
| seed_m | 4 | `range_subspace_behavior_footprint_not_supported` | -0.00051127 | -0.05383761 | 0.00000928 | 0.00052055 |
| seed_m | 8 | `range_subspace_behavior_footprint_not_supported` | -0.00186387 | -0.08626631 | 0.00002867 | 0.00189254 |
| seed_m | 16 | `range_subspace_behavior_footprint_not_supported` | -0.00805314 | -0.20504924 | 0.00009164 | 0.00814479 |
| seed_m | 32 | `range_subspace_behavior_footprint_not_supported` | -0.01620723 | -0.30781757 | 0.00258836 | 0.01879558 |

## Claim Discipline

Range-probe behavior-footprint conclusions can be reported as stable or rank-sensitive across the predeclared rank grid. This is a sensitivity stress test, not causal mediation.

Forbidden claim: Rank sensitivity proves the presence or absence of belief tracking or a causal circuit.
