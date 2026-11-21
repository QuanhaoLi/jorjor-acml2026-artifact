from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.tables import build_no_range_bootstrap_pack


def main() -> int:
    """Build no-range paired bootstrap confidence intervals from archived correctness rows."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", default="checkpoints/remote_archive/3090")
    parser.add_argument("--output-dir", default="checkpoints/tables")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--bootstrap-count", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260602)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    args = parser.parse_args()

    payload = build_no_range_bootstrap_pack(
        archive_root=Path(args.archive_root),
        output_dir=Path(args.output_dir),
        commit=args.commit,
        bootstrap_count=args.bootstrap_count,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
    )
    print(
        "wrote no-range bootstrap pack "
        f"encoder_layer_2_mean_delta={payload['encoder_layer_2_mean_delta']:.6f} "
        f"encoder_layer_2_min_lower_delta={payload['encoder_layer_2_min_lower_delta']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
