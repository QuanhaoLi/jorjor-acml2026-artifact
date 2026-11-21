from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.teacher_distill_gates import build_strong_teacher_gate_report


def main() -> int:
    """Build a k.0.6 strong-teacher strict gate report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-rows", required=True)
    parser.add_argument("--student-metrics", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--locked-layer", default="encoder_layer_2")
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    parser.add_argument("--probe-learning-rate", type=float, default=0.01)
    parser.add_argument("--bootstrap-count", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260603)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--minimum-matched-points", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    payload = build_strong_teacher_gate_report(
        artifact_rows_path=Path(args.artifact_rows),
        student_metrics_path=Path(args.student_metrics),
        model_path=Path(args.model_path),
        output_dir=Path(args.output_dir),
        locked_layer=args.locked_layer,
        probe_epochs=args.probe_epochs,
        probe_batch_size=args.probe_batch_size,
        probe_learning_rate=args.probe_learning_rate,
        bootstrap_count=args.bootstrap_count,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
        minimum_matched_points=args.minimum_matched_points,
        seed=args.seed,
        device=args.device,
    )
    print(
        "wrote k.0.6 strong-teacher gate report "
        f"label={payload['final_diagnostic_label']} "
        f"s6={payload['gates']['s6_hidden_vs_action_value_composition']['delta']:.6f} "
        f"s7={payload['gates']['s7_matched_hidden_vs_composition_prior']['delta']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
