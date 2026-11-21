from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.inspect.cbps_report import build_cbps_claim_ladder


def main() -> int:
    """Build the k.5.3 CBPS claim ladder."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-cbps-report", required=True)
    parser.add_argument("--predictive-link-report", required=True)
    parser.add_argument("--composition-boundedness-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--poker-oracle-report", default=None)
    parser.add_argument("--composition-sensitivity-report", default=None)
    parser.add_argument("--threshold-manifest", default=None)
    parser.add_argument("--capacity-report", default=None)
    parser.add_argument("--behavior-link-report", default=None)
    parser.add_argument("--git-commit", default=None)
    args = parser.parse_args()

    payload = build_cbps_claim_ladder(
        synthetic_cbps_report=Path(args.synthetic_cbps_report),
        predictive_link_report=Path(args.predictive_link_report),
        composition_boundedness_report=Path(args.composition_boundedness_report),
        output_dir=Path(args.output_dir),
        poker_oracle_report=Path(args.poker_oracle_report) if args.poker_oracle_report is not None else None,
        composition_sensitivity_report=(
            Path(args.composition_sensitivity_report) if args.composition_sensitivity_report is not None else None
        ),
        threshold_manifest=Path(args.threshold_manifest) if args.threshold_manifest is not None else None,
        capacity_report=Path(args.capacity_report) if args.capacity_report is not None else None,
        behavior_link_report=Path(args.behavior_link_report) if args.behavior_link_report is not None else None,
        git_commit=args.git_commit,
    )
    print(
        f"wrote {payload['version']} CBPS claim ladder "
        f"final_label={payload['final_label']} checksum={payload['checksum']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
