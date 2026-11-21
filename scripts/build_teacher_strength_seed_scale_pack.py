from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.inspect.teacher_strength_sweep import build_teacher_strength_seed_scale_pack


def main() -> int:
    """Build the k.0.8 teacher-strength seed/scale pack."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--teacher-iterations", type=int, default=4)
    parser.add_argument("--teacher-deal-sample-count", type=int, default=8)
    parser.add_argument("--teacher-hand-count", type=int, default=24)
    parser.add_argument("--teacher-evaluation-hand-count", type=int, default=24)
    parser.add_argument("--student-epochs", type=int, default=3)
    parser.add_argument("--student-batch-size", type=int, default=16)
    parser.add_argument("--student-hidden-size", type=int, default=32)
    parser.add_argument("--student-num-heads", type=int, default=4)
    parser.add_argument("--student-num-layers", type=int, default=2)
    parser.add_argument("--probe-epochs", type=int, default=3)
    parser.add_argument("--probe-batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-count", type=int, default=64)
    parser.add_argument("--minimum-matched-points", type=int, default=2)
    parser.add_argument("--locked-layer", default="encoder_layer_2")
    parser.add_argument("--local-smoke-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    payload = build_teacher_strength_seed_scale_pack(
        output_dir=Path(args.output_dir),
        seeds=tuple(args.seeds),
        teacher_iteration_count=args.teacher_iterations,
        teacher_deal_sample_count=args.teacher_deal_sample_count,
        teacher_hand_count=args.teacher_hand_count,
        teacher_evaluation_hand_count=args.teacher_evaluation_hand_count,
        student_epochs=args.student_epochs,
        student_batch_size=args.student_batch_size,
        student_hidden_size=args.student_hidden_size,
        student_num_heads=args.student_num_heads,
        student_num_layers=args.student_num_layers,
        probe_epochs=args.probe_epochs,
        probe_batch_size=args.probe_batch_size,
        bootstrap_count=args.bootstrap_count,
        minimum_matched_points=args.minimum_matched_points,
        locked_layer=args.locked_layer,
        local_smoke_only=args.local_smoke_only,
        device=args.device,
    )
    print(
        "wrote k.0.8 teacher-strength seed/scale pack "
        f"seeds={payload['seed_count']} "
        f"label={payload['robustness_label']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
