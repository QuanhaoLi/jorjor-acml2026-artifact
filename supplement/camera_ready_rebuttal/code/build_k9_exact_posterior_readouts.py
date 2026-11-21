"""CLI for exact known-teacher 169-way posterior readouts and five-run aggregation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.limit_posterior import (
    FIVE_RUN_SEEDS,
    build_five_run_exact_posterior_report,
    build_single_seed_exact_posterior_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--seed-names", nargs="+", default=list(FIVE_RUN_SEEDS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--extraction-batch-size", type=int, default=512)
    parser.add_argument("--readout-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--residual-learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bootstrap-count", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--max-pairs-per-bucket", type=int, default=64)
    parser.add_argument("--max-train-hands", type=int, default=None)
    parser.add_argument("--max-validation-hands", type=int, default=None)
    parser.add_argument("--max-test-hands", type=int, default=None)
    parser.add_argument("--no-reuse-complete", action="store_true")
    args = parser.parse_args()

    common = {
        "archive_root": Path(args.archive_root),
        "source_revision": args.source_revision,
        "device": args.device,
        "extraction_batch_size": args.extraction_batch_size,
        "readout_batch_size": args.readout_batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "residual_learning_rate": args.residual_learning_rate,
        "weight_decay": args.weight_decay,
        "bootstrap_count": args.bootstrap_count,
        "bootstrap_seed": args.bootstrap_seed,
        "confidence_level": args.confidence_level,
        "max_pairs_per_bucket": args.max_pairs_per_bucket,
        "max_train_hands": args.max_train_hands,
        "max_validation_hands": args.max_validation_hands,
        "max_test_hands": args.max_test_hands,
    }
    seed_names = tuple(args.seed_names)
    if len(seed_names) == 1:
        payload = build_single_seed_exact_posterior_report(
            seed_name=seed_names[0],
            output_dir=Path(args.output_dir),
            **common,
        )
    else:
        payload = build_five_run_exact_posterior_report(
            output_dir=Path(args.output_dir),
            seed_names=seed_names,
            reuse_complete=not args.no_reuse_complete,
            **common,
        )
    print(
        "wrote k.9.0 exact posterior product "
        f"final_diagnostic_label={payload['final_diagnostic_label']} "
        f"checksum={payload['checksum']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

