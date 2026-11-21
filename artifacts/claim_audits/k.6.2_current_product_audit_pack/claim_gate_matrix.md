# Claim Gate Matrix

| case | probe positive | label artifact ruled out | history prior ruled out | action/value nuisance ruled out | composition nuisance ruled out | matched-composition residual supported | final diagnostic label |
|---|---|---|---|---|---|---|---|
| range_supervised_ar | fail | direct_range_supervision | not_primary_gate | fail | fail | fail | composition_shortcut_supported |
| action_value_only_no_range_ar | pass | passes | passes | pass | fail | fail | composition_shortcut_supported |
| action_value_only_no_range_ar_mlp_appendix | pass | inherits_main_linear_gate | inherits_main_linear_gate | pass | inconclusive | inconclusive | mlp_inconclusive |

## Sources

| key | path | present |
|---|---|---|
| belief_shortcut_report | outputs/inspect_v119_smoke/belief_shortcut_report.json | True |
| matched_composition_report | outputs/inspect_v119_smoke/matched_composition_report.json | True |
| paper_evidence_pack | checkpoints/tables/limit_ar_paper_evidence_pack.json | True |
| range_supervised_strict_gate | checkpoints/remote_archive/v100s/64a2220/outputs/tables/limit_ar_range_supervised_positive_control_pack.json | True |
| probe_capacity_s6_s7 | checkpoints/remote_archive/v100s/64a2220/outputs/tables/probe_capacity_s6_s7_report.json | True |
