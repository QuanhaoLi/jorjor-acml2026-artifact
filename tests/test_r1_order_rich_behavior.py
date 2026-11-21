from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jorjor.envs.order_rich_poker import EXACT_POSTERIOR_SIGNAL_SUPPORTED
from jorjor.train.order_rich_poker import (
    COMPOSITION_SHORTCUT_STATE,
    COMPOSITION_SHORTCUT_SUPPORTED,
    INCONCLUSIVE,
    LEARNED_POSTERIOR_SUPPORTED,
    ORACLE_POSTERIOR_STATE,
    POSTERIOR_SUPERVISED_HEAD,
    build_order_rich_behavior_gate_report,
)
from jorjor.models.tiny import require_torch


class OrderRichBehaviorGateTest(unittest.TestCase):
    """Test r.1 learned behavior-model gate report."""

    def setUp(self) -> None:
        try:
            require_torch()
        except RuntimeError as exc:
            self.skipTest(str(exc))

    def test_report_builder_writes_expected_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "r1"
            payload = build_order_rich_behavior_gate_report(
                output_dir=output_dir,
                length=4,
                seed=51,
                epochs=8,
                hidden_size=24,
                batch_size=128,
                max_pairs=512,
                bootstrap_count=80,
                min_bucket_count=5,
                min_pair_count=128,
                min_policy_accuracy=0.0,
            )
            saved = json.loads((output_dir / "r.1_order_rich_behavior_gates_report.json").read_text(encoding="utf-8"))
            manifest_exists = (output_dir / "r.1_order_rich_behavior_gates_manifest.json").exists()

        labels = {case["mode"]: case["final_diagnostic_label"] for case in payload["cases"]}
        self.assertEqual(saved["report_type"], "r.1_order_rich_behavior_gates")
        self.assertEqual(labels[ORACLE_POSTERIOR_STATE], EXACT_POSTERIOR_SIGNAL_SUPPORTED)
        self.assertEqual(labels[COMPOSITION_SHORTCUT_STATE], COMPOSITION_SHORTCUT_SUPPORTED)
        self.assertIn(labels[POSTERIOR_SUPERVISED_HEAD], {LEARNED_POSTERIOR_SUPPORTED, INCONCLUSIVE})
        self.assertTrue(manifest_exists)
        for case in payload["cases"]:
            matched = case["matched_composition_summary"]
            self.assertEqual(matched["estimand"], "unweighted_composition_bucket_mean")
            self.assertAlmostEqual(matched["representation_js_ci"]["mean"], matched["mean_representation_js"])

    def test_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_r1_order_rich_behavior_gates.py",
                    "--output-dir",
                    str(Path(temp_dir) / "r1"),
                    "--length",
                    "4",
                    "--epochs",
                    "8",
                    "--hidden-size",
                    "24",
                    "--batch-size",
                    "128",
                    "--max-pairs",
                    "512",
                    "--bootstrap-count",
                    "80",
                    "--min-bucket-count",
                    "5",
                    "--min-pair-count",
                    "128",
                    "--min-policy-accuracy",
                    "0.0",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote r.1 order-rich behavior gate report", result.stdout)


if __name__ == "__main__":
    unittest.main()
