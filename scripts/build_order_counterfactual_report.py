from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.counterfactual import build_order_counterfactual_report


def main() -> int:
    """Build the v1.5 order-counterfactual causal report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["embedding_final", "encoder_layer_0", "encoder_layer_1", "encoder_layer_2", "encoder_layer_3", "sequence_hidden"],
    )
    parser.add_argument("--min-group-size", type=int, default=2)
    parser.add_argument("--min-posterior-js", type=float, default=0.01)
    parser.add_argument("--min-pairs", type=int, default=100)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--separation-threshold", type=float, default=0.001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--write-pairs", action="store_true")
    args = parser.parse_args()

    payload = build_order_counterfactual_report(
        train_config_path=Path(args.train_config),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        layers=tuple(args.layers),
        min_group_size=args.min_group_size,
        min_posterior_js=args.min_posterior_js,
        min_pairs=args.min_pairs,
        max_pairs=args.max_pairs,
        separation_threshold=args.separation_threshold,
        device=args.device,
        write_pairs=args.write_pairs,
    )
    print(
        "wrote order-counterfactual report "
        f"pairs={payload['pair_count']} final_diagnostic_label={payload['final_diagnostic_label']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
