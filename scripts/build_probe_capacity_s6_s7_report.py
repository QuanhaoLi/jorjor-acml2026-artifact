from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.capacity import build_probe_capacity_s6_s7_report


def main() -> int:
    """Build v1.18 real no-range MLP capacity S6/S7 report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mlp-commit", required=True)
    parser.add_argument("--linear-commit", required=True)
    parser.add_argument("--bootstrap-count", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260603)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    args = parser.parse_args()

    payload = build_probe_capacity_s6_s7_report(
        archive_root=Path(args.archive_root),
        output_dir=Path(args.output_dir),
        mlp_commit=args.mlp_commit,
        linear_commit=args.linear_commit,
        bootstrap_count=args.bootstrap_count,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
    )
    print(
        "wrote probe capacity S6/S7 report "
        f"rows={payload['row_count']} verdict={payload['final_verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
