from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.figures import build_paper_figure_pack


def main() -> int:
    """Build v1.16 paper figures from diagnostic artifacts."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--belief-shortcut-report", required=True)
    parser.add_argument("--final-tables", required=True)
    parser.add_argument("--matched-composition-report", required=True)
    parser.add_argument("--synthetic-oracle-report", required=True)
    parser.add_argument("--learned-synthetic-report", required=True)
    parser.add_argument("--toy-oracle-report", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    payload = build_paper_figure_pack(
        belief_shortcut_report_path=Path(args.belief_shortcut_report),
        final_tables_path=Path(args.final_tables),
        matched_composition_report_path=Path(args.matched_composition_report),
        synthetic_oracle_report_path=Path(args.synthetic_oracle_report),
        learned_synthetic_report_path=Path(args.learned_synthetic_report),
        toy_oracle_report_path=Path(args.toy_oracle_report),
        output_dir=Path(args.output_dir),
    )
    print(f"wrote paper figure pack figures={len(payload['figures'])} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
