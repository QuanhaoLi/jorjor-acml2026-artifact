from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.inspect import build_belief_shortcut_report


def main() -> int:
    """Build the v1.1 belief-vs-shortcut diagnostic report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", default="checkpoints/remote_archive/v100s")
    parser.add_argument("--output-dir", default="outputs/inspect")
    parser.add_argument("--commit", required=True, help="composition nuisance commit, normally v1.0.16")
    parser.add_argument("--fresh50-commit", default="881534f")
    parser.add_argument("--history-composition-commit", default="1d9574f")
    args = parser.parse_args()

    payload = build_belief_shortcut_report(
        archive_root=Path(args.archive_root),
        output_dir=Path(args.output_dir),
        composition_commit=args.commit,
        fresh50_commit=args.fresh50_commit,
        history_composition_commit=args.history_composition_commit,
    )
    case = payload["cases"][0]
    print(
        "wrote belief shortcut report "
        f"case={case['case']} final_diagnostic_label={case['final_diagnostic_label']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
