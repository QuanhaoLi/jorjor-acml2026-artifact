"""CLI for the immutable-source k.9.0.1 matched-support contract repair.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.limit_posterior import build_k901_support_repair


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    payload = build_k901_support_repair(
        source_report_path=Path(args.source_report),
        output_dir=Path(args.output_dir),
    )
    print(
        "wrote k.9.0.1 support repair "
        f"final_diagnostic_label={payload['final_diagnostic_label']} "
        f"checksum={payload['checksum']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

