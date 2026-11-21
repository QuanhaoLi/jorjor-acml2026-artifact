from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.tables import build_paper_table_reproduction


def _current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    """Rebuild final paper tables and a one-command reproduction manifest."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fresh50-commit", required=True)
    parser.add_argument("--bootstrap-count", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260602)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--composition-commit", default=None)
    parser.add_argument("--history-composition-commit", default=None)
    parser.add_argument("--composition-pack", default=None)
    parser.add_argument("--history-composition-pack", default=None)
    args = parser.parse_args()

    command = "python scripts/reproduce_paper_tables.py " + " ".join(sys.argv[1:])
    payload = build_paper_table_reproduction(
        archive_root=Path(args.archive_root),
        output_dir=Path(args.output_dir),
        fresh50_commit=args.fresh50_commit,
        bootstrap_count=args.bootstrap_count,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
        composition_commit=args.composition_commit,
        history_composition_commit=args.history_composition_commit,
        composition_pack_path=Path(args.composition_pack) if args.composition_pack else None,
        history_composition_pack_path=Path(args.history_composition_pack) if args.history_composition_pack else None,
        command=command,
        code_commit=_current_commit(),
    )
    print(
        "wrote paper table reproduction manifest "
        f"outputs={len(payload['outputs'])} "
        f"missing_optional={len(payload['missing_optional_packs'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
