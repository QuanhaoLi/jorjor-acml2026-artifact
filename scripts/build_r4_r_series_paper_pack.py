from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.order_rich_causal import build_r_series_paper_pack


def main() -> int:
    """Build the r.4 r-series paper pack."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--r001", required=True)
    parser.add_argument("--r002", required=True)
    parser.add_argument("--r1", required=True)
    parser.add_argument("--r2", required=True)
    parser.add_argument("--r3", required=True)
    args = parser.parse_args()

    payload = build_r_series_paper_pack(
        output_dir=Path(args.output_dir),
        r001_path=Path(args.r001),
        r002_path=Path(args.r002),
        r1_path=Path(args.r1),
        r2_path=Path(args.r2),
        r3_path=Path(args.r3),
    )
    print(
        "wrote r.4 r-series paper pack "
        f"final_diagnostic_label={payload['final_diagnostic_label']} "
        f"checksum={payload['checksum']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
