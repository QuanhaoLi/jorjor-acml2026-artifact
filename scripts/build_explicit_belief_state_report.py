from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.train.synthetic_belief import build_explicit_belief_state_report


def main() -> int:
    """Build the v1.20 explicit belief-state positive-control report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-count", type=int, default=512)
    parser.add_argument("--length", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--max-order-pairs", type=int, default=4096)
    parser.add_argument("--bootstrap-count", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260603)
    args = parser.parse_args()

    payload = build_explicit_belief_state_report(
        output_dir=Path(args.output_dir),
        sequence_count=args.sequence_count,
        length=args.length,
        seed=args.seed,
        max_order_pairs=args.max_order_pairs,
        bootstrap_count=args.bootstrap_count,
        bootstrap_seed=args.bootstrap_seed,
    )
    labels = ",".join(case["final_diagnostic_label"] for case in payload["cases"])
    print(f"wrote explicit belief-state report labels={labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
