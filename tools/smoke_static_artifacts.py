"""Dependency-light smoke test for included static paper artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

JSON_FILES = [
    "artifacts/claim_audits/k.6.6_current_product_audit_pack/diagnostic_report.json",
    "artifacts/claim_audits/k.6.6_current_product_audit_pack/tables.json",
    "artifacts/claim_audits/k.5.1_full_poker_predictive_link/k.5.1_poker_predictive_link_report.json",
    "artifacts/claim_audits/k.5.2_composition_boundedness/k.5.2_composition_boundedness_report.json",
    "artifacts/claim_audits/k.5.0_synthetic_cbps_control/k.5.0_synthetic_cbps_report.json",
    "artifacts/paper_outputs/v1.16/v1.16/figure_pack.json",
]

TEXT_EXPECTATIONS = {
    "supplement/CLAIM_LEDGER.md": [
        "not exact posterior tracking",
        "Validation rows evaluate the diagnostic behavior",
    ],
    "artifacts/claim_audits/k.6.6_current_product_audit_pack/claim_gate_matrix.md": [
        "composition",
    ],
}


def main() -> int:
    loaded = {}
    for rel in JSON_FILES:
        path = ROOT / rel
        if not path.exists():
            raise FileNotFoundError(rel)
        loaded[rel] = json.loads(path.read_text(encoding="utf-8"))

    for rel, needles in TEXT_EXPECTATIONS.items():
        text = (ROOT / rel).read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                raise AssertionError(f"{needle!r} not found in {rel}")

    figure_dir = ROOT / "artifacts" / "figures"
    png_count = len(list(figure_dir.glob("*.png")))
    if png_count < 3:
        raise AssertionError(f"expected at least 3 PNG figures, found {png_count}")

    print(
        json.dumps(
            {
                "loaded_json_files": len(loaded),
                "figure_png_count": png_count,
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
