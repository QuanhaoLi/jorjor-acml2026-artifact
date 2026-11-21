from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.inspect import build_claim_gate_matrix


def main() -> int:
    """Build the v1.3 ACML claim-gate matrix."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-dir", default="outputs/inspect")
    parser.add_argument("--paper-table-dir", default="checkpoints/tables")
    parser.add_argument("--strict-table-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/inspect")
    args = parser.parse_args()

    payload = build_claim_gate_matrix(
        inspect_dir=Path(args.inspect_dir),
        paper_table_dir=Path(args.paper_table_dir),
        output_dir=Path(args.output_dir),
        strict_table_dir=Path(args.strict_table_dir) if args.strict_table_dir else None,
    )
    print(f"wrote claim gate matrix rows={len(payload['rows'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
