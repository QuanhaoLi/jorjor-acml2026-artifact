from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.teacher_artifact import build_limit_abstract_cfr_teacher_artifact


def main() -> int:
    """Build the k.0.4 Limit abstract CFR teacher artifact smoke."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iteration-count", type=int, default=10)
    parser.add_argument("--deal-sample-count", type=int, default=20)
    parser.add_argument("--hand-count", type=int, default=100)
    parser.add_argument("--evaluation-hand-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260603)
    args = parser.parse_args()

    payload = build_limit_abstract_cfr_teacher_artifact(
        output_dir=Path(args.output_dir),
        iteration_count=args.iteration_count,
        deal_sample_count=args.deal_sample_count,
        hand_count=args.hand_count,
        evaluation_hand_count=args.evaluation_hand_count,
        seed=args.seed,
    )
    print(
        "wrote k.0.4 Limit abstract CFR teacher artifact "
        f"contract_passed={payload['teacher_artifact_contract_passed']} "
        f"rows={payload['decision_point_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
