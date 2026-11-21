from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.train.latent_type_sequence import build_latent_type_multiseed_pack


def main() -> int:
    """Build the k.1.5 multi-seed latent-type protocol-transfer pack."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-glob", default="**/k.1_latent_type_protocol_transfer_report.json")
    args = parser.parse_args()

    payload = build_latent_type_multiseed_pack(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        report_glob=args.report_glob,
    )
    print(
        "wrote k.1 multiseed protocol-transfer pack "
        f"reports={payload['report_count']} "
        f"excluded={payload['excluded_report_count']} "
        f"robust_protocol_transfer_supported={payload['robust_protocol_transfer_supported']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
