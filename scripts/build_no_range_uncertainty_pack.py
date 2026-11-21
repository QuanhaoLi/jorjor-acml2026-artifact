from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.tables import build_no_range_uncertainty_pack


def main() -> int:
    """Build no-range probe uncertainty summaries from archived remote artifacts."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", default="checkpoints/remote_archive/3090")
    parser.add_argument("--output-dir", default="checkpoints/tables")
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()

    payload = build_no_range_uncertainty_pack(
        archive_root=Path(args.archive_root),
        output_dir=Path(args.output_dir),
        commit=args.commit,
    )
    print(
        "wrote no-range uncertainty pack "
        f"sequence_hidden_mean_std={payload['sequence_hidden_mean_std']:.6f} "
        f"sequence_hidden_delta_mean={payload['sequence_hidden_delta_mean']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
