from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jorjor.inspect.paper_summary import build_k2_paper_summary


class K2PaperSummaryTest(unittest.TestCase):
    """Test k.2.4 paper-ready summary extraction from product packs."""

    def test_build_k2_paper_summary_preserves_claim_safe_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            product_pack = root / "pack"
            output_dir = root / "summary"
            _write_product_pack(product_pack)

            payload = build_k2_paper_summary(product_pack_dir=product_pack, output_dir=output_dir)
            saved = json.loads((output_dir / "k.2.4_paper_summary.json").read_text(encoding="utf-8"))
            markdown = (output_dir / "k.2.4_paper_summary.md").read_text(encoding="utf-8")

        self.assertEqual(payload["gap_count"], 0)
        self.assertEqual(saved["rows"]["frozen_lm_latent_order_audit"], "order_information_supported")
        self.assertEqual(saved["rows"]["poker_verdict"], "composition_shortcut_supported")
        self.assertIn("does not license a belief-like LLM claim", markdown)
        self.assertIn("composition_shortcut_supported", markdown)

    def test_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            product_pack = root / "pack"
            output_dir = root / "summary"
            _write_product_pack(product_pack)

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_k2_paper_summary.py",
                    "--product-pack-dir",
                    str(product_pack),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote k.2.4 paper summary", result.stdout)


def _write_product_pack(product_pack: Path) -> None:
    product_pack.mkdir(parents=True)
    diagnostic = {
        "checksum": "pack-checksum",
        "final_verdict": "composition_shortcut_supported",
        "methodology_validation_status": "passed",
        "toy_oracle_validation_status": "passed",
        "learned_methodology_validation_status": "passed",
        "explicit_methodology_validation_status": "passed",
        "latent_protocol_transfer_status": "passed",
        "llm_latent_order_status": "order_information_supported",
        "range_supervised_strict_gate_status": "composition_shortcut_supported",
        "probe_capacity_s6_s7_status": "mlp_inconclusive",
        "reproduction_manifest_status": "present",
        "gaps": [],
    }
    tables = {
        "llm_latent_order": [
            {
                "cases": [
                    {
                        "backend": "hf_causal_lm",
                        "final_diagnostic_label": "order_information_supported",
                    }
                ]
            }
        ]
    }
    (product_pack / "diagnostic_report.json").write_text(json.dumps(diagnostic, sort_keys=True), encoding="utf-8")
    (product_pack / "tables.json").write_text(json.dumps(tables, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
