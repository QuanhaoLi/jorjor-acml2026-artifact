from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jorjor.envs.order_rich_poker import (
    COMPOSITION_COLLAPSED,
    EXACT_POSTERIOR_SIGNAL_SUPPORTED,
    ORDER_RICH_EXACT_POSTERIOR_SUPPORTED,
    build_order_rich_exact_posterior_report,
    composition_key,
    composition_state,
    exact_posterior,
    matched_order_rows,
    matched_order_summary,
)


class OrderRichPokerTest(unittest.TestCase):
    """Test r.0.2 order-rich exact-posterior benchmark."""

    def test_exact_posterior_is_order_sensitive_under_same_composition(self) -> None:
        left = ("p0_probe", "p1_raise", "p0_call", "p1_check", "p0_check", "p1_call")
        right = ("p0_check", "p1_raise", "p0_call", "p1_check", "p0_probe", "p1_call")

        self.assertEqual(composition_key(left), composition_key(right))
        self.assertAlmostEqual(sum(exact_posterior(left)), 1.0)
        self.assertNotEqual(exact_posterior(left), exact_posterior(right))

    def test_oracle_rows_pass_while_composition_rows_collapse(self) -> None:
        oracle_rows = matched_order_rows(
            length=5,
            representation_fn=exact_posterior,
            max_pairs=1024,
        )
        composition_rows = matched_order_rows(
            length=5,
            representation_fn=composition_state,
            max_pairs=1024,
        )

        self.assertGreater(len(oracle_rows), 0)
        self.assertGreater(sum(float(row["representation_js"]) for row in oracle_rows), 0.0)
        self.assertEqual(sum(float(row["representation_js"]) for row in composition_rows), 0.0)

    def test_matched_summary_uses_bucket_estimand_for_point_and_ci(self) -> None:
        rows = [
            {"composition_key": "a", "posterior_js": 1.0, "representation_js": 1.0},
            {"composition_key": "a", "posterior_js": 1.0, "representation_js": 1.0},
            {"composition_key": "b", "posterior_js": 0.0, "representation_js": 0.0},
        ]

        summary = matched_order_summary(
            rows,
            bootstrap_count=20,
            bootstrap_seed=11,
        )

        self.assertEqual(summary["estimand"], "unweighted_composition_bucket_mean")
        self.assertAlmostEqual(summary["mean_representation_js"], 0.5)
        self.assertAlmostEqual(summary["representation_js_ci"]["mean"], summary["mean_representation_js"])
        self.assertAlmostEqual(summary["pair_table_mean_representation_js"], 2.0 / 3.0)

    def test_report_builder_writes_supported_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "r002"
            payload = build_order_rich_exact_posterior_report(
                output_dir=output_dir,
                sequence_count=512,
                length=5,
                seed=41,
                max_pairs=2048,
                bootstrap_count=100,
                min_bucket_count=5,
                min_pair_count=256,
            )
            saved = json.loads(
                (output_dir / "r.0.2_order_rich_exact_posterior_report.json").read_text(encoding="utf-8")
            )
            manifest_exists = (output_dir / "r.0.2_order_rich_exact_posterior_manifest.json").exists()

        self.assertEqual(payload["report_type"], "r.0.2_order_rich_exact_posterior")
        self.assertEqual(saved["version"], "r.0.2")
        self.assertEqual(payload["oracle_detection_label"], EXACT_POSTERIOR_SIGNAL_SUPPORTED)
        self.assertEqual(payload["composition_detection_label"], COMPOSITION_COLLAPSED)
        self.assertEqual(payload["final_diagnostic_label"], ORDER_RICH_EXACT_POSTERIOR_SUPPORTED)
        self.assertTrue(manifest_exists)

    def test_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_r002_order_rich_exact_posterior.py",
                    "--output-dir",
                    str(Path(temp_dir) / "r002"),
                    "--sequence-count",
                    "512",
                    "--length",
                    "5",
                    "--max-pairs",
                    "2048",
                    "--bootstrap-count",
                    "100",
                    "--min-bucket-count",
                    "5",
                    "--min-pair-count",
                    "256",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote r.0.2 order-rich exact-posterior report", result.stdout)


if __name__ == "__main__":
    unittest.main()
