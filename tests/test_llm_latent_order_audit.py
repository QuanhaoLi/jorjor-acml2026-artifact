from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from jorjor.probe.llm_latent_order import (
    BELIEF_LIKE_TRANSFER_SUPPORTED,
    COMPOSITION_SHORTCUT_SUPPORTED,
    ORDER_INFORMATION_SUPPORTED,
    build_llm_latent_order_report,
)


class LLMLatentOrderAuditTest(unittest.TestCase):
    """Test the k.2 frozen-LM latent-order audit report contract."""

    def test_fake_backends_label_positive_shortcut_and_order_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "k2"
            payload = build_llm_latent_order_report(
                output_dir=output_dir,
                length=5,
                max_pairs=128,
                bootstrap_count=100,
                bootstrap_seed=20260604,
                backends=("oracle_posterior", "composition", "order_only", "surface_noise"),
                surface_template_count=2,
            )
            saved = json.loads((output_dir / "k.2_llm_latent_order_report.json").read_text(encoding="utf-8"))
            manifest = json.loads((output_dir / "k.2_llm_latent_order_manifest.json").read_text(encoding="utf-8"))
            table_exists = (output_dir / "k.2_llm_latent_order_table.md").exists()

        labels = {case["backend"]: case["final_diagnostic_label"] for case in payload["cases"]}
        self.assertEqual(labels["oracle_posterior"], BELIEF_LIKE_TRANSFER_SUPPORTED)
        self.assertEqual(labels["composition"], COMPOSITION_SHORTCUT_SUPPORTED)
        self.assertEqual(labels["order_only"], ORDER_INFORMATION_SUPPORTED)
        self.assertNotEqual(labels["surface_noise"], BELIEF_LIKE_TRANSFER_SUPPORTED)
        self.assertEqual(saved["report_type"], "k.2_llm_latent_order_audit")
        self.assertIn("frozen_lm_transfer_stress_test", saved["claim_scope"])
        self.assertEqual(manifest["report_checksum"], saved["checksum"])
        self.assertTrue(table_exists)

    def test_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_llm_latent_order_report.py",
                    "--output-dir",
                    str(Path(temp_dir) / "k2"),
                    "--length",
                    "5",
                    "--max-pairs",
                    "128",
                    "--bootstrap-count",
                    "100",
                    "--backends",
                    "oracle_posterior,composition,order_only,surface_noise",
                    "--hf-batch-size",
                    "4",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote k.2 LLM latent-order audit", result.stdout)

    def test_capped_matched_pairs_cover_buckets_broadly(self) -> None:
        started = time.monotonic()
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = build_llm_latent_order_report(
                output_dir=Path(temp_dir) / "k2",
                length=6,
                max_pairs=8192,
                bootstrap_count=20,
                bootstrap_seed=20260604,
                backends=("oracle_posterior",),
                surface_template_count=2,
            )

        oracle = payload["cases"][0]
        matched = oracle["matched_composition_summary"]
        self.assertEqual(matched["pair_count"], 8192)
        self.assertGreaterEqual(matched["bucket_count"], 180)
        self.assertLess(time.monotonic() - started, 20.0)


if __name__ == "__main__":
    unittest.main()
