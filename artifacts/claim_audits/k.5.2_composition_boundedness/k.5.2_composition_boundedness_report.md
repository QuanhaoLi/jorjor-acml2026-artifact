# k.5.2 Composition-Boundedness Report

This is a poker composition-boundedness audit. The pair analysis uses opponent 169-class range-label proxy gaps, not exact oracle posterior JS.

final_diagnostic_label: `composition_bounded_supported`

## Composition Ladder

| variant | lower delta | mean delta | upper delta | label | estimand note |
|---|---:|---:|---:|---|---|
| C_min | 0.002114 | 0.002114 | NA | `visible_shortcut_proxy_positive` | heterogeneous_evidence_ladder |
| C_mid | 0.021482 | 0.021482 | NA | `visible_shortcut_proxy_positive` | heterogeneous_evidence_ladder |
| C_full | -0.093948 | -0.052369 | -0.045442 | `composition_shortcut_supported` | locked_matched_composition_bucket_estimand |

## Seed Rows

| seed | label | S7 delta | S7 lower | proxy pairs | proxy buckets | same hidden dist | diff hidden dist | gap slope | slope lower | slope upper |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed_k | `composition_bounded_supported` | -0.043977 | -0.080361 | 9252 | 146 | 0.148663 | 0.145328 | -0.003335 | -0.017343 | 0.009236 |
| seed_l | `composition_bounded_supported` | -0.043790 | -0.081007 | 8880 | 146 | 0.203590 | 0.191208 | -0.012382 | -0.026485 | 0.000776 |
| seed_m | `composition_bounded_supported` | -0.052369 | -0.093948 | 8664 | 146 | 0.207717 | 0.215036 | 0.007319 | -0.008733 | 0.023498 |

## Aggregate

| metric | value |
|---|---:|
| composition ladder label | `full_composition_only_blocks` |
| matched delta mean | -0.046712 |
| proxy slope mean | -0.002799 |
| proxy slope seed lower | -0.012382 |
| proxy slope seed upper | 0.007319 |
