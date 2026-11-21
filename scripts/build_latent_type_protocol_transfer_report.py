from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.train.latent_type_sequence import build_latent_type_protocol_transfer_report


def main() -> int:
    """Build the k.1 latent-type protocol-transfer report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-count", type=int, default=512)
    parser.add_argument("--length", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--signal-probability", type=float, default=0.72)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-order-pairs", type=int, default=4096)
    parser.add_argument("--bootstrap-count", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260603)
    parser.add_argument("--correlation-threshold", type=float, default=0.80)
    args = parser.parse_args()

    payload = build_latent_type_protocol_transfer_report(
        output_dir=Path(args.output_dir),
        sequence_count=args.sequence_count,
        length=args.length,
        seed=args.seed,
        signal_probability=args.signal_probability,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_order_pairs=args.max_order_pairs,
        bootstrap_count=args.bootstrap_count,
        bootstrap_seed=args.bootstrap_seed,
        correlation_threshold=args.correlation_threshold,
    )
    labels = ",".join(case["final_diagnostic_label"] for case in payload["cases"])
    print(
        "wrote k.1 latent-type protocol-transfer report "
        f"protocol_transfer_supported={payload['protocol_transfer_supported']} labels={labels}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
