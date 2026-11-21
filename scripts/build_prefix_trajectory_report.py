from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.counterfactual import build_prefix_trajectory_report


def main() -> int:
    """Build the v1.8 prefix trajectory report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["encoder_layer_2", "sequence_hidden"],
    )
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--min-non-final-transitions", type=int, default=100)
    parser.add_argument("--alignment-threshold", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    payload = build_prefix_trajectory_report(
        train_config_path=Path(args.train_config),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        layers=tuple(args.layers),
        max_points=args.max_points,
        min_non_final_transitions=args.min_non_final_transitions,
        alignment_threshold=args.alignment_threshold,
        device=args.device,
    )
    print(
        "wrote prefix trajectory report "
        f"transitions={payload['summary']['transition_count']} "
        f"final_diagnostic_label={payload['final_diagnostic_label']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
