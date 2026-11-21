from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.inspect.paper_section_pack import build_k2_paper_section_pack


def main() -> int:
    """Build the k.2.5 paper-section pack."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-summary-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    payload = build_k2_paper_section_pack(
        paper_summary_json=Path(args.paper_summary_json),
        output_dir=Path(args.output_dir),
    )
    print(
        "wrote k.2.5 paper section pack "
        f"gap_count={payload['gap_count']} poker={payload['rows']['poker_verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
