from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.k3_diagnostics import build_composition_feature_sensitivity_report


def main() -> int:
    """Build the k.3.1 composition feature sensitivity report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--belief-report", required=True)
    parser.add_argument("--matched-report", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    payload = build_composition_feature_sensitivity_report(
        belief_report_path=Path(args.belief_report),
        matched_report_path=Path(args.matched_report),
        output_dir=Path(args.output_dir),
    )
    print(
        "wrote k.3.1 composition feature sensitivity "
        f"label={payload['final_diagnostic_label']} variants={payload['variant_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
