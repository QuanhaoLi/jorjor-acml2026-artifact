from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.envs.leduc_exact import build_exact_leduc_feasibility_report


def main() -> int:
    """Build the r.0.1 exact-posterior Leduc feasibility report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hand-count", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--max-pairs", type=int, default=20000)
    parser.add_argument("--bootstrap-count", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260610)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--min-bucket-count", type=int, default=100)
    parser.add_argument("--min-pair-multiplicity", type=int, default=2000)
    args = parser.parse_args()

    payload = build_exact_leduc_feasibility_report(
        output_dir=Path(args.output_dir),
        hand_count=args.hand_count,
        seed=args.seed,
        max_pairs=args.max_pairs,
        bootstrap_count=args.bootstrap_count,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
        min_bucket_count=args.min_bucket_count,
        min_pair_multiplicity=args.min_pair_multiplicity,
    )
    print(
        "wrote r.0.1 exact Leduc feasibility report "
        f"final_diagnostic_label={payload['final_diagnostic_label']} "
        f"checksum={payload['checksum']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
