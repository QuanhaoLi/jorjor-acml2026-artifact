from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.train.synthetic_belief import build_learned_synthetic_belief_report


def main() -> int:
    """Build the v1.10 learned synthetic belief diagnostic report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-count", type=int, default=512)
    parser.add_argument("--length", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-size", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-order-pairs", type=int, default=4096)
    args = parser.parse_args()

    payload = build_learned_synthetic_belief_report(
        output_dir=Path(args.output_dir),
        sequence_count=args.sequence_count,
        length=args.length,
        seed=args.seed,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        max_order_pairs=args.max_order_pairs,
    )
    labels = ",".join(case["final_diagnostic_label"] for case in payload["cases"])
    print(f"wrote learned synthetic belief report labels={labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
