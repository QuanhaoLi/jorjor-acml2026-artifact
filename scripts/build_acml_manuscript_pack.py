from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.inspect import build_acml_manuscript_pack


def main() -> int:
    """Build the v1.4 ACML manuscript pack."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-dir", default="outputs/inspect")
    parser.add_argument("--output-dir", default="outputs/inspect")
    args = parser.parse_args()

    payload = build_acml_manuscript_pack(
        inspect_dir=Path(args.inspect_dir),
        output_dir=Path(args.output_dir),
    )
    print(f"wrote ACML manuscript pack {payload['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
