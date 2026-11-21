from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.k3_diagnostics import build_llm_template_robustness_report


def main() -> int:
    """Build the k.3.2 LLM prompt-template robustness report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-reports", required=True, help="Comma-separated k.2.6 sweep report paths.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    payload = build_llm_template_robustness_report(
        sweep_report_paths=tuple(Path(part.strip()) for part in args.sweep_reports.split(",") if part.strip()),
        output_dir=Path(args.output_dir),
    )
    print(
        "wrote k.3.2 LLM template robustness "
        f"label={payload['final_diagnostic_label']} models={payload['model_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
