from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.tables import build_range_supervised_positive_control_pack


def main() -> int:
    """Build the v1.17 range-supervised positive-control audit pack."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--bootstrap-count", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260603)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    args = parser.parse_args()

    payload = build_range_supervised_positive_control_pack(
        archive_root=Path(args.archive_root),
        output_dir=Path(args.output_dir),
        commit=args.commit,
        bootstrap_count=args.bootstrap_count,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
    )
    print(
        "wrote range-supervised positive-control pack "
        f"rows={payload['row_count']} label={payload['final_diagnostic_label']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
