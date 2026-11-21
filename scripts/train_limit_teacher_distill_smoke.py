from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.train.teacher_distill import run_limit_teacher_distill_smoke


def main() -> int:
    """Run the k.0.5 Limit teacher distillation smoke."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-rows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--value-loss-weight", type=float, default=0.3)
    parser.add_argument("--split-seed", type=int, default=20260603)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-save-model", action="store_true")
    args = parser.parse_args()

    result = run_limit_teacher_distill_smoke(
        artifact_rows_path=Path(args.artifact_rows),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        value_loss_weight=args.value_loss_weight,
        split_seed=args.split_seed,
        seed=args.seed,
        device=args.device,
        save_model=not args.no_save_model,
    )
    report = result.report
    print(
        "wrote k.0.5 Limit teacher distillation smoke "
        f"rows={report['decision_point_count']} "
        f"kl={report['test_teacher_distribution_kl']:.6f} "
        f"ce={report['test_teacher_distribution_cross_entropy']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
