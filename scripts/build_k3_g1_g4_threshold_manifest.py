from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.k3_diagnostics import build_g1_g4_threshold_manifest


def main() -> int:
    """Build the k.3.3 G1-G4 threshold/config manifest."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--belief-report", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    payload = build_g1_g4_threshold_manifest(
        belief_report_path=Path(args.belief_report),
        output_dir=Path(args.output_dir),
    )
    print(f"wrote k.3.3 G1-G4 threshold manifest gates={payload['gate_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
