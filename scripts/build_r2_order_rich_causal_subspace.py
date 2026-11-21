from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.order_rich_causal import build_order_rich_causal_subspace_report


def main() -> int:
    """Build the r.2 causal subspace report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--length", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--signal-probability", type=float, default=0.70)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-pairs", type=int, default=30000)
    parser.add_argument("--subspace-rank", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=20260611)
    args = parser.parse_args()

    payload = build_order_rich_causal_subspace_report(
        output_dir=Path(args.output_dir),
        length=args.length,
        seed=args.seed,
        signal_probability=args.signal_probability,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_pairs=args.max_pairs,
        subspace_rank=args.subspace_rank,
        random_seed=args.random_seed,
    )
    print(
        "wrote r.2 order-rich causal subspace report "
        f"final_diagnostic_label={payload['final_diagnostic_label']} "
        f"checksum={payload['checksum']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
