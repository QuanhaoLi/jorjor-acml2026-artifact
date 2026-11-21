from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.inspect.teacher_strength import build_teacher_strength_comparison_pack


def main() -> int:
    """Build the k.0.7 teacher-strength comparison pack."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-report", required=True)
    parser.add_argument("--student-report", required=True)
    parser.add_argument("--gate-report", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    payload = build_teacher_strength_comparison_pack(
        teacher_report_path=Path(args.teacher_report),
        student_report_path=Path(args.student_report),
        gate_report_path=Path(args.gate_report),
        output_dir=Path(args.output_dir),
    )
    row = payload["rows"][0]
    print(
        "wrote k.0.7 teacher-strength comparison pack "
        f"teacher={row['teacher_family']} "
        f"label={row['final_diagnostic_label']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
