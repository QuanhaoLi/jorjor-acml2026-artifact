from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.tables import build_paper_evidence_pack


def main() -> int:
    """Build a paper evidence pack from archived Limit AR summaries."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", default="checkpoints/remote_archive/3090")
    parser.add_argument("--output-dir", default="checkpoints/tables")
    args = parser.parse_args()

    payload = build_paper_evidence_pack(
        archive_root=Path(args.archive_root),
        output_dir=Path(args.output_dir),
    )
    print(f"wrote paper evidence pack recommendation={payload['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
