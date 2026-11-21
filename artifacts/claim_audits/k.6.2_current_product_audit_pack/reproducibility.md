# Reproducibility

## Command

```text
python scripts/jorjor_inspect_audit.py --inspect-dir outputs/inspect_v119_smoke --output-dir outputs/k/k.6.2_current_product_audit_pack --counterfactual-dir outputs/experiments/v1.5 --trajectory-dir outputs/experiments/v1.8 --baseline-dir outputs/experiments/v1.9 --synthetic-dir outputs/experiments/v1.7 --toy-oracle-dir outputs/experiments/v1.15 --learned-synthetic-dir outputs/experiments/v1.10 --explicit-belief-state-dir outputs/experiments/v1.20 --latent-protocol-dir outputs/k/paper/k.1_full_ci_fixed --llm-full-sweep-dir outputs/k/k.2.6.3_llm_full_sweep_remote_qwen05_paper_table --cbps-claim-ladder-dir outputs/k/k.5.6_cbps_claim_ladder --order-intervention-dir outputs/k/k.6.0_order_intervention --reproduction-manifest outputs/paper/v1.13/reproduction_manifest.json --strict-table-dir outputs/k/k.2.3_strict_tables
```

## Inputs

- inspect_dir: `outputs/inspect_v119_smoke`
- counterfactual_dir: `outputs/experiments/v1.5`
- trajectory_dir: `outputs/experiments/v1.8`
- baseline_dir: `outputs/experiments/v1.9`
- synthetic_dir: `outputs/experiments/v1.7`
- toy_oracle_dir: `outputs/experiments/v1.15`
- learned_synthetic_dir: `outputs/experiments/v1.10`
- explicit_belief_state_dir: `outputs/experiments/v1.20`
- latent_protocol_dir: `outputs/k/paper/k.1_full_ci_fixed`
- llm_latent_order_dir: `None`
- llm_full_sweep_dir: `outputs/k/k.2.6.3_llm_full_sweep_remote_qwen05_paper_table`
- cbps_claim_ladder_dir: `outputs/k/k.5.6_cbps_claim_ladder`
- order_intervention_dir: `outputs/k/k.6.0_order_intervention`
