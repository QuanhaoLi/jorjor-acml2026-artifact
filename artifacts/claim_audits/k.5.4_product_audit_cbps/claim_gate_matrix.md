# Claim Gate Matrix

| case | probe positive | label artifact ruled out | history prior ruled out | action/value nuisance ruled out | composition nuisance ruled out | matched-composition residual supported | final diagnostic label |
|---|---|---|---|---|---|---|---|
| range_supervised_ar | pass | not_primary_gate | not_primary_gate | pass | not_tested | not_tested | probe_positive_action_value_controlled |
| action_value_only_no_range_ar | pass | passes | passes | pass | fail | fail | composition_shortcut_supported |

## Sources

| key | path | present |
|---|---|---|
| belief_shortcut_report | outputs/inspect/belief_shortcut_report.json | True |
| matched_composition_report | outputs/inspect/matched_composition_report.json | True |
| paper_evidence_pack | checkpoints/tables/limit_ar_paper_evidence_pack.json | True |
