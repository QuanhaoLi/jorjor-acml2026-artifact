from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.inspect import build_audit_report


def main() -> int:
    """Run the product-facing JorJor-Inspect audit assembler."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--counterfactual-dir", default=None)
    parser.add_argument("--trajectory-dir", default=None)
    parser.add_argument("--baseline-dir", default=None)
    parser.add_argument("--synthetic-dir", default=None)
    parser.add_argument("--toy-oracle-dir", default=None)
    parser.add_argument("--learned-synthetic-dir", default=None)
    parser.add_argument("--explicit-belief-state-dir", default=None)
    parser.add_argument("--latent-protocol-dir", default=None)
    parser.add_argument("--llm-latent-order-dir", default=None)
    parser.add_argument("--llm-full-sweep-dir", default=None)
    parser.add_argument("--cbps-claim-ladder-dir", default=None)
    parser.add_argument("--order-intervention-dir", default=None)
    parser.add_argument("--range-subspace-dir", default=None)
    parser.add_argument("--range-rank-sensitivity-dir", default=None)
    parser.add_argument("--reproduction-manifest", default=None)
    parser.add_argument("--strict-table-dir", default=None)
    args = parser.parse_args()

    command = "python scripts/jorjor_inspect_audit.py " + " ".join(sys.argv[1:])
    payload = build_audit_report(
        inspect_dir=Path(args.inspect_dir),
        output_dir=Path(args.output_dir),
        counterfactual_dir=Path(args.counterfactual_dir) if args.counterfactual_dir else None,
        trajectory_dir=Path(args.trajectory_dir) if args.trajectory_dir else None,
        baseline_dir=Path(args.baseline_dir) if args.baseline_dir else None,
        synthetic_dir=Path(args.synthetic_dir) if args.synthetic_dir else None,
        toy_oracle_dir=Path(args.toy_oracle_dir) if args.toy_oracle_dir else None,
        learned_synthetic_dir=Path(args.learned_synthetic_dir) if args.learned_synthetic_dir else None,
        explicit_belief_state_dir=Path(args.explicit_belief_state_dir) if args.explicit_belief_state_dir else None,
        latent_protocol_dir=Path(args.latent_protocol_dir) if args.latent_protocol_dir else None,
        llm_latent_order_dir=Path(args.llm_latent_order_dir) if args.llm_latent_order_dir else None,
        llm_full_sweep_dir=Path(args.llm_full_sweep_dir) if args.llm_full_sweep_dir else None,
        cbps_claim_ladder_dir=Path(args.cbps_claim_ladder_dir) if args.cbps_claim_ladder_dir else None,
        order_intervention_dir=Path(args.order_intervention_dir) if args.order_intervention_dir else None,
        range_subspace_dir=Path(args.range_subspace_dir) if args.range_subspace_dir else None,
        range_rank_sensitivity_dir=Path(args.range_rank_sensitivity_dir) if args.range_rank_sensitivity_dir else None,
        reproduction_manifest_path=Path(args.reproduction_manifest) if args.reproduction_manifest else None,
        strict_table_dir=Path(args.strict_table_dir) if args.strict_table_dir else None,
        command=command,
    )
    print(
        "wrote JorJor-Inspect audit "
        f"final_verdict={payload['final_verdict']} "
        f"methodology_validation={payload['methodology_validation_status']} "
        f"learned_methodology_validation={payload['learned_methodology_validation_status']} "
        f"latent_protocol_transfer={payload['latent_protocol_transfer_status']} "
        f"cbps_claim_ladder={payload['cbps_claim_ladder_status']} "
        f"cbps_behavior_link={payload['cbps_behavior_link_gate']} "
        f"order_intervention={payload['order_intervention_status']} "
        f"range_subspace={payload['range_subspace_status']} "
        f"range_rank_sensitivity={payload['range_rank_sensitivity_status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
