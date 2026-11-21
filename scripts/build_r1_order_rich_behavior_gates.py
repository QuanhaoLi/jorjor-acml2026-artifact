from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.train.order_rich_poker import build_order_rich_behavior_gate_report


def main() -> int:
    """Build the r.1 learned behavior-model gate report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--length", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--signal-probability", type=float, default=0.70)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--posterior-loss-weight", type=float, default=2.0)
    parser.add_argument("--max-pairs", type=int, default=30000)
    parser.add_argument("--bootstrap-count", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260610)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--correlation-threshold", type=float, default=0.80)
    parser.add_argument("--min-pair-count", type=int, default=2000)
    parser.add_argument("--min-bucket-count", type=int, default=100)
    parser.add_argument("--min-policy-accuracy", type=float, default=0.80)
    args = parser.parse_args()

    payload = build_order_rich_behavior_gate_report(
        output_dir=Path(args.output_dir),
        length=args.length,
        seed=args.seed,
        signal_probability=args.signal_probability,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        posterior_loss_weight=args.posterior_loss_weight,
        max_pairs=args.max_pairs,
        bootstrap_count=args.bootstrap_count,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
        correlation_threshold=args.correlation_threshold,
        min_pair_count=args.min_pair_count,
        min_bucket_count=args.min_bucket_count,
        min_policy_accuracy=args.min_policy_accuracy,
    )
    print(
        "wrote r.1 order-rich behavior gate report "
        f"positive_control={payload['protocol_positive_control_supported']} "
        f"behavior_only={payload['behavior_only_supported']} "
        f"checksum={payload['checksum']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
