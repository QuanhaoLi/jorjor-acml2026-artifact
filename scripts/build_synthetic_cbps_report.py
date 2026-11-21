from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.train.synthetic_cbps import build_synthetic_cbps_report


def main() -> int:
    """Build the k.5.0 synthetic CBPS known-control report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-count", type=int, default=4096)
    parser.add_argument("--length", type=int, default=6)
    parser.add_argument("--signal-probability", type=float, default=0.72)
    parser.add_argument("--max-pairs", type=int, default=4096)
    parser.add_argument("--bootstrap-count", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260610)
    parser.add_argument("--matched-pair-threshold", type=int, default=512)
    parser.add_argument("--mean-posterior-js-threshold", type=float, default=0.05)
    args = parser.parse_args()

    payload = build_synthetic_cbps_report(
        output_dir=Path(args.output_dir),
        sequence_count=args.sequence_count,
        length=args.length,
        signal_probability=args.signal_probability,
        max_pairs=args.max_pairs,
        bootstrap_count=args.bootstrap_count,
        bootstrap_seed=args.bootstrap_seed,
        matched_pair_threshold=args.matched_pair_threshold,
        mean_posterior_js_threshold=args.mean_posterior_js_threshold,
    )
    print(
        "wrote k.5.0 synthetic CBPS report "
        f"synthetic_cbps_control_passed={payload['synthetic_cbps_control_passed']} "
        f"checksum={payload['checksum']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
