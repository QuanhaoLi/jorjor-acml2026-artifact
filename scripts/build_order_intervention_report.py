from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.order_intervention import DEFAULT_SEEDS, build_order_intervention_report


def main() -> int:
    """Build the k.6.0 input-level order-intervention stress report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-names", nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--train-config-dir", default="configs/train")
    parser.add_argument("--bootstrap-count", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260610)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-test-points", type=int, default=None)
    parser.add_argument("--max-pairs", type=int, default=10000)
    parser.add_argument("--locked-layer", default="encoder_layer_2")
    parser.add_argument("--min-pairs-for-label", type=int, default=100)
    parser.add_argument("--movement-epsilon", type=float, default=1e-6)
    args = parser.parse_args()

    payload = build_order_intervention_report(
        archive_root=Path(args.archive_root),
        output_dir=Path(args.output_dir),
        seed_names=tuple(args.seed_names),
        train_config_dir=Path(args.train_config_dir),
        bootstrap_count=args.bootstrap_count,
        bootstrap_seed=args.bootstrap_seed,
        device=args.device,
        batch_size=args.batch_size,
        max_test_points=args.max_test_points,
        max_pairs=args.max_pairs,
        locked_layer=args.locked_layer,
        min_pairs_for_label=args.min_pairs_for_label,
        movement_epsilon=args.movement_epsilon,
    )
    print(
        "wrote k.6.0 order-intervention report "
        f"final_diagnostic_label={payload['final_diagnostic_label']} "
        f"checksum={payload['checksum']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
