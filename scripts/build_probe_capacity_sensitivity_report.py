from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.capacity import build_probe_capacity_sensitivity_report


def main() -> int:
    """Build v1.14 linear-vs-MLP probe capacity sensitivity report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--linear-metrics", required=True)
    parser.add_argument("--mlp-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--locked-layer", default="encoder_layer_2")
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()

    payload = build_probe_capacity_sensitivity_report(
        linear_metrics_path=Path(args.linear_metrics),
        mlp_metrics_path=Path(args.mlp_metrics),
        output_dir=Path(args.output_dir),
        locked_layer=args.locked_layer,
        tolerance=args.tolerance,
    )
    print(
        "wrote probe capacity sensitivity report "
        f"verdict_agreement={payload['verdict_agreement']} "
        f"score_delta={payload['score_delta_mlp_minus_linear']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
