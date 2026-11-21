# k.6.0 Order-Intervention Stress Report

This is an input-level order permutation stress test. It is not a legal poker causal proof.

final_diagnostic_label: `order_sensitive_stress_supported`

| seed | label | pairs | hidden cosine | policy JS | range JS | value delta | policy changed |
|---|---|---:|---:|---:|---:|---:|---:|
| seed_k | `order_sensitive_stress_supported` | 5906 | 0.03147946 | 0.01840322 | 0.00234968 | 0.19593712 | 0.024382 |
| seed_l | `order_sensitive_stress_supported` | 5661 | 0.01968243 | 0.00365922 | 0.00213277 | 0.21020367 | 0.003886 |
| seed_m | `order_sensitive_stress_supported` | 5770 | 0.03077172 | 0.01141433 | 0.00299132 | 0.27857727 | 0.014385 |

## Aggregate

| metric | mean | lower | upper |
|---|---:|---:|---:|
| mean_hidden_cosine_distance | 0.02731120 | 0.01968243 | 0.03147946 |
| mean_policy_js | 0.01115892 | 0.00365922 | 0.01840322 |
| mean_range_head_js | 0.00249126 | 0.00213277 | 0.00299132 |
| mean_value_abs_delta | 0.22823935 | 0.19593712 | 0.27857727 |
| policy_argmax_changed_rate | 0.01421766 | 0.00388624 | 0.02438198 |
| range_argmax_changed_rate | 0.21816920 | 0.18422877 | 0.27945593 |

## Claim Discipline

Under input-level same-composition order permutations, the frozen poker AR can be audited for order sensitivity. This is a stress test, not a legal poker causal proof.

Forbidden claim: This input permutation proves or disproves ordered opponent-belief tracking.
