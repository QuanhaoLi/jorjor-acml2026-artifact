from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.inspect.paper_summary import build_k2_paper_summary


def main() -> int:
    """Build the k.2.4 paper-ready product summary."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--product-pack-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    payload = build_k2_paper_summary(
        product_pack_dir=Path(args.product_pack_dir),
        output_dir=Path(args.output_dir),
    )
    print(
        "wrote k.2.4 paper summary "
        f"gap_count={payload['gap_count']} k2={payload['rows']['frozen_lm_latent_order_audit']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
