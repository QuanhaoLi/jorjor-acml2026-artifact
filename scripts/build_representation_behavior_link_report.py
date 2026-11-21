from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.cbps import DEFAULT_SEEDS, build_representation_behavior_link_report


def main() -> int:
    """Build the k.5.5 representation-to-behavior link report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-names", nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--train-config-dir", default="configs/train")
    parser.add_argument("--bootstrap-count", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260610)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-train-points", type=int, default=30000)
    parser.add_argument("--max-test-points", type=int, default=None)
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    args = parser.parse_args()

    payload = build_representation_behavior_link_report(
        archive_root=Path(args.archive_root),
        output_dir=Path(args.output_dir),
        seed_names=tuple(args.seed_names),
        train_config_dir=Path(args.train_config_dir),
        bootstrap_count=args.bootstrap_count,
        bootstrap_seed=args.bootstrap_seed,
        device=args.device,
        batch_size=args.batch_size,
        max_train_points=args.max_train_points,
        max_test_points=args.max_test_points,
        probe_epochs=args.probe_epochs,
        probe_batch_size=args.probe_batch_size,
        learning_rate=args.learning_rate,
    )
    print(
        "wrote k.5.5 representation-behavior link report "
        f"final_diagnostic_label={payload['final_diagnostic_label']} "
        f"checksum={payload['checksum']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
