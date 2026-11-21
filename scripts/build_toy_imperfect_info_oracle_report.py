from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.envs.toy_imperfect_info import build_toy_imperfect_info_oracle_report


def main() -> int:
    """Build v1.15 toy imperfect-information oracle validation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-count", type=int, default=512)
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    parser.add_argument("--max-order-pairs", type=int, default=1024)
    args = parser.parse_args()

    payload = build_toy_imperfect_info_oracle_report(
        output_dir=Path(args.output_dir),
        sequence_count=args.sequence_count,
        length=args.length,
        seed=args.seed,
        correlation_threshold=args.correlation_threshold,
        max_order_pairs=args.max_order_pairs,
    )
    labels = ",".join(case["final_diagnostic_label"] for case in payload["cases"])
    print(f"wrote toy imperfect-info oracle report labels={labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
