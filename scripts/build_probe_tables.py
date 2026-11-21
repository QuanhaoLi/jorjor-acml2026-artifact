from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.tables import build_probe_tables


def main() -> int:
    """Build paper-ready probe summary tables from archived metrics."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", default="checkpoints/remote_archive/3090")
    parser.add_argument("--output-dir", default="checkpoints/tables")
    parser.add_argument("--full-control-commit", required=True)
    args = parser.parse_args()

    payload = build_probe_tables(
        archive_root=Path(args.archive_root),
        output_dir=Path(args.output_dir),
        full_control_commit=args.full_control_commit,
    )
    print(f"wrote rows={len(payload['rows'])} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
