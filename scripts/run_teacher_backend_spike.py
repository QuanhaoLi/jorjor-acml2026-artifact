from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.teacher_backend_spike import run_teacher_backend_spike


def main() -> int:
    """Run the k.0.1 teacher backend spike."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--leduc-iteration-count", type=int, default=20)
    parser.add_argument("--leduc-hand-count", type=int, default=20)
    parser.add_argument("--limit-iteration-count", type=int, default=4)
    parser.add_argument("--limit-deal-sample-count", type=int, default=8)
    parser.add_argument("--limit-hand-count", type=int, default=8)
    args = parser.parse_args()

    payload = run_teacher_backend_spike(
        output_path=Path(args.output_path),
        leduc_iteration_count=args.leduc_iteration_count,
        leduc_hand_count=args.leduc_hand_count,
        limit_iteration_count=args.limit_iteration_count,
        limit_deal_sample_count=args.limit_deal_sample_count,
        limit_hand_count=args.limit_hand_count,
        seed=args.seed,
    )
    print(
        "wrote k.0.1 teacher backend spike "
        f"recommended_backend={payload['recommended_backend']} "
        f"local_backend_pass_count={payload['local_backend_pass_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
