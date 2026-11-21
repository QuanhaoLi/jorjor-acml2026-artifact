# k.5.3 CBPS Claim Ladder

final_label: `cbps_supported`

## Claim-Safe Sentence

Positive hidden-state range probes are insufficient evidence for ordered belief tracking. For the no-range poker AR row, the best-supported diagnostic interpretation is composition-bounded predictive support: behavior prediction is positive, while strict matched-composition gates block an ordered-belief claim.

## Rows

| domain | model/state | predictive | composition bounded | matched proxy | capacity | final label | claim scope |
|---|---|---|---|---|---|---|---|
| synthetic_latent_order | posterior/composition/CBPS known states | `passed` | `cbps_supported` | `not_applicable_exact_synthetic_posterior_available` | `not_applicable` | `synthetic_cbps_control_passed` | `diagnostic_label_validation` |
| limit_holdem | no-range 4-layer AR, seeds k/l/m | `predictive_supported` | `composition_bounded_supported` | `range_label_proxy_tracking_not_supported` | `mlp_inconclusive` | `cbps_supported` | `diagnostic_interpretation_not_ordered_belief` |


## Gaps

- none
