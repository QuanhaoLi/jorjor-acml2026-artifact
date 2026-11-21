from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jorjor.inspect.paper_section_pack import build_k2_paper_section_pack


class K2PaperSectionPackTest(unittest.TestCase):
    """Test paper-section snippets generated from the k.2.4 summary."""

    def test_build_k2_paper_section_pack_preserves_conservative_claims(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary_json = root / "k.2.4_paper_summary.json"
            output_dir = root / "section_pack"
            _write_summary(summary_json)

            payload = build_k2_paper_section_pack(paper_summary_json=summary_json, output_dir=output_dir)
            markdown = (output_dir / "k.2.5_paper_section_pack.md").read_text(encoding="utf-8")

        self.assertEqual(payload["gap_count"], 0)
        self.assertEqual(payload["rows"]["poker_verdict"], "composition_shortcut_supported")
        self.assertEqual(payload["rows"]["frozen_lm_latent_order_audit"], "order_information_supported")
        self.assertIn("does not license a belief-like LLM claim", markdown)
        self.assertIn("composition_shortcut_supported", markdown)
        self.assertIn("order_information_supported", markdown)

    def test_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary_json = root / "k.2.4_paper_summary.json"
            output_dir = root / "section_pack"
            _write_summary(summary_json)

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_k2_paper_section_pack.py",
                    "--paper-summary-json",
                    str(summary_json),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote k.2.5 paper section pack", result.stdout)


def _write_summary(path: Path) -> None:
    payload = {
        "version": "k.2.4",
        "report_type": "k.2.4_paper_summary",
        "product_pack_dir": "outputs/k/k.2.3_full_product_audit_pack_gap0",
        "product_pack_checksum": "pack-checksum",
        "gap_count": 0,
        "gaps": [],
        "rows": {
            "poker_verdict": "composition_shortcut_supported",
            "synthetic_validation": "passed",
            "toy_oracle_validation": "passed",
            "learned_synthetic_validation": "passed",
            "explicit_belief_state_validation": "passed",
            "non_poker_protocol_transfer": "passed",
            "frozen_lm_latent_order_audit": "order_information_supported",
            "range_supervised_strict_gate": "composition_shortcut_supported",
            "mlp_capacity_gate": "mlp_inconclusive",
            "reproduction_manifest": "present",
        },
        "llm_case_summary": {
            "backend": "hf_causal_lm",
            "final_diagnostic_label": "order_information_supported",
        },
        "paper_safe_claim": "claim-safe fixture sentence",
        "checksum": "summary-checksum",
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
