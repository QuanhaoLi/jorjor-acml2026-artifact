from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.order_rich_causal import build_order_rich_robustness_pack


def main() -> int:
    """Build the r.3 multi-seed robustness pack."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--length", type=int, default=6)
    parser.add_argument("--signal-probability", type=float, default=0.70)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-pairs", type=int, default=30000)
    parser.add_argument("--subspace-rank", type=int, default=3)
    args = parser.parse_args()

    seeds = tuple(int(seed.strip()) for seed in args.seeds.split(",") if seed.strip())
    payload = build_order_rich_robustness_pack(
        output_dir=Path(args.output_dir),
        seeds=seeds,
        length=args.length,
        signal_probability=args.signal_probability,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_pairs=args.max_pairs,
        subspace_rank=args.subspace_rank,
    )
    print(
        "wrote r.3 order-rich robustness pack "
        f"final_diagnostic_label={payload['final_diagnostic_label']} "
        f"checksum={payload['checksum']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
