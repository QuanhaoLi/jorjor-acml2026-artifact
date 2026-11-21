from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.inspect import build_matched_composition_report


def main() -> int:
    """Build the legacy-key target-stratified matched sensitivity report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", default="checkpoints/remote_archive/v100s")
    parser.add_argument("--output-dir", default="outputs/inspect")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--bootstrap-count", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260602)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    args = parser.parse_args()

    payload = build_matched_composition_report(
        archive_root=Path(args.archive_root),
        output_dir=Path(args.output_dir),
        commit=args.commit,
        bootstrap_count=args.bootstrap_count,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
    )
    print(
        "wrote target-stratified matched sensitivity report "
        f"final_diagnostic_label={payload['final_diagnostic_label']} "
        f"min_delta={payload['min_delta_residual_minus_prior']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
