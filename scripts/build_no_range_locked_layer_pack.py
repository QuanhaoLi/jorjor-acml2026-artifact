from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.tables import build_no_range_locked_layer_pack


def main() -> int:
    """Build the fresh no-range locked-layer verification pack."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", default="checkpoints/remote_archive/3090")
    parser.add_argument("--output-dir", default="checkpoints/tables")
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()

    payload = build_no_range_locked_layer_pack(
        archive_root=Path(args.archive_root),
        output_dir=Path(args.output_dir),
        commit=args.commit,
    )
    print(
        "wrote no-range locked-layer pack "
        f"encoder_layer_2_delta_mean={payload['encoder_layer_2_delta_mean']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
