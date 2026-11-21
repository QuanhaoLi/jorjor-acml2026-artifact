from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.k3_diagnostics import build_poker_oracle_positive_control_report


def main() -> int:
    """Build the k.3.0 poker oracle positive-control report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-count", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260605)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    args = parser.parse_args()

    payload = build_poker_oracle_positive_control_report(
        matched_report_path=Path(args.matched_report),
        output_dir=Path(args.output_dir),
        bootstrap_count=args.bootstrap_count,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
    )
    print(
        "wrote k.3.0 poker oracle positive control "
        f"label={payload['final_diagnostic_label']} "
        f"min_lower={payload['min_bucket_bootstrap_lower_delta']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
