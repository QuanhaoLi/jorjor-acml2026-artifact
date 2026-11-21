from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.counterfactual import build_prefix_composition_baseline_report


def main() -> int:
    """Build the v1.9 prefix composition baseline report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--min-non-final-transitions", type=int, default=100)
    args = parser.parse_args()

    payload = build_prefix_composition_baseline_report(
        train_config_path=Path(args.train_config),
        output_dir=Path(args.output_dir),
        max_points=args.max_points,
        min_non_final_transitions=args.min_non_final_transitions,
    )
    print(
        "wrote prefix composition baseline report "
        f"transitions={payload['summary']['transition_count']} "
        f"final_diagnostic_label={payload['final_diagnostic_label']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
