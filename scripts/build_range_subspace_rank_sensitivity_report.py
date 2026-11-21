from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.range_subspace_rank_sensitivity import (
    DEFAULT_RANK_GRID,
    DEFAULT_SEEDS,
    build_range_subspace_rank_sensitivity_report,
)


def main() -> int:
    """Build the k.6.5 range-subspace rank-sensitivity report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-names", nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--rank-grid", nargs="+", type=int, default=list(DEFAULT_RANK_GRID))
    parser.add_argument("--train-config-dir", default="configs/train")
    parser.add_argument("--bootstrap-count", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260610)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-train-points", type=int, default=30000)
    parser.add_argument("--max-test-points", type=int, default=None)
    parser.add_argument("--locked-layer", default="sequence_hidden")
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--probe-learning-rate", type=float, default=0.01)
    parser.add_argument("--random-subspace-count", type=int, default=8)
    parser.add_argument("--probe-top10-floor", type=float, default=0.08)
    args = parser.parse_args()

    payload = build_range_subspace_rank_sensitivity_report(
        archive_root=Path(args.archive_root),
        output_dir=Path(args.output_dir),
        seed_names=tuple(args.seed_names),
        rank_grid=tuple(args.rank_grid),
        train_config_dir=Path(args.train_config_dir),
        bootstrap_count=args.bootstrap_count,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
        device=args.device,
        batch_size=args.batch_size,
        max_train_points=args.max_train_points,
        max_test_points=args.max_test_points,
        locked_layer=args.locked_layer,
        probe_epochs=args.probe_epochs,
        probe_learning_rate=args.probe_learning_rate,
        random_subspace_count=args.random_subspace_count,
        probe_top10_floor=args.probe_top10_floor,
    )
    print(
        "wrote k.6.5 range-subspace rank-sensitivity report "
        f"final_diagnostic_label={payload['final_diagnostic_label']} "
        f"checksum={payload['checksum']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
