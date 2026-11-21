from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.envs.order_rich_poker import build_order_rich_exact_posterior_report


def main() -> int:
    """Build the r.0.2 order-rich exact-posterior feasibility report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-count", type=int, default=20000)
    parser.add_argument("--length", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--signal-probability", type=float, default=0.70)
    parser.add_argument("--max-pairs", type=int, default=30000)
    parser.add_argument("--bootstrap-count", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260610)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--min-bucket-count", type=int, default=100)
    parser.add_argument("--min-pair-count", type=int, default=2000)
    args = parser.parse_args()

    payload = build_order_rich_exact_posterior_report(
        output_dir=Path(args.output_dir),
        sequence_count=args.sequence_count,
        length=args.length,
        seed=args.seed,
        signal_probability=args.signal_probability,
        max_pairs=args.max_pairs,
        bootstrap_count=args.bootstrap_count,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
        min_bucket_count=args.min_bucket_count,
        min_pair_count=args.min_pair_count,
    )
    print(
        "wrote r.0.2 order-rich exact-posterior report "
        f"final_diagnostic_label={payload['final_diagnostic_label']} "
        f"checksum={payload['checksum']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
