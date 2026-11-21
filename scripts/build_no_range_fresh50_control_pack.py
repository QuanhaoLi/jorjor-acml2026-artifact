from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.tables import build_no_range_fresh50_control_pack


def main() -> int:
    """Build the fresh 50k no-range supplemental control pack."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", default="checkpoints/remote_archive/3090")
    parser.add_argument("--output-dir", default="checkpoints/tables")
    parser.add_argument("--commit-50k", required=True)
    parser.add_argument("--commit-10k", default="9a368b2")
    parser.add_argument("--bootstrap-count", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260602)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    args = parser.parse_args()

    payload = build_no_range_fresh50_control_pack(
        archive_root=Path(args.archive_root),
        output_dir=Path(args.output_dir),
        commit_50k=args.commit_50k,
        commit_10k=args.commit_10k,
        bootstrap_count=args.bootstrap_count,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
    )
    print(
        "wrote no-range fresh50 control pack "
        f"encoder_layer_2_repeat_delta_mean={payload['encoder_layer_2_repeat_delta_mean']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
