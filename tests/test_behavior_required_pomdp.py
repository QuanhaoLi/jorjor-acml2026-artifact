from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from jorjor.envs.belief_required_pomdp import (
    BEHAVIOR_ONLY_BELIEF_SUPPORTED,
    COMPOSITION_SHORTCUT_SUPPORTED,
    action_value_targets,
    build_preflight_report,
    composition_state,
    exact_posterior,
    matched_order_rows,
)
from jorjor.train.behavior_required_pomdp import build_behavior_required_pomdp_report


class BehaviorRequiredPOMDPTest(unittest.TestCase):
    """Test k.4 behavior-only belief-required POMDP diagnostics."""

    def test_preflight_finds_behavior_disagreement_inside_matched_composition(self) -> None:
        left = ("o0", "o1", "o2", "o3")
        right = ("o3", "o2", "o1", "o0")

        self.assertNotEqual(exact_posterior(left), exact_posterior(right))
        self.assertNotEqual(action_value_targets(left)["action_label"], action_value_targets(right)["action_label"])

        report = build_preflight_report(length=5, max_order_pairs=512)

        self.assertGreater(report["matched_pair_count"], 0)
        self.assertGreater(report["mean_posterior_js"], 0.0)
        self.assertGreaterEqual(report["behavior_disagreement_fraction"], 0.25)
        self.assertLess(report["composition_action_accuracy_upper_bound"], 0.85)
        self.assertTrue(report["belief_required"])

    def test_oracle_and_composition_rows_are_separated(self) -> None:
        oracle_rows = matched_order_rows(length=5, representation_fn=exact_posterior, max_order_pairs=256)
        shortcut_rows = matched_order_rows(length=5, representation_fn=composition_state, max_order_pairs=256)

        self.assertGreater(sum(row["representation_js"] for row in oracle_rows), 0.0)
        self.assertEqual(sum(row["representation_js"] for row in shortcut_rows), 0.0)

    def test_behavior_only_gru_report_records_behavior_only_partial_support(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "k4"

            payload = build_behavior_required_pomdp_report(
                output_dir=output_dir,
                length=5,
                sequence_count=128,
                epochs=30,
                hidden_size=32,
                batch_size=128,
                max_order_pairs=512,
                bootstrap_count=100,
                correlation_threshold=0.70,
                action_accuracy_threshold=0.80,
            )
            saved = json.loads(
                (output_dir / "k.4.0_behavior_required_pomdp_report.json").read_text(encoding="utf-8")
            )

        labels = {case["mode"]: case["final_diagnostic_label"] for case in payload["cases"]}
        self.assertEqual(labels["oracle_posterior_state"], BEHAVIOR_ONLY_BELIEF_SUPPORTED)
        self.assertEqual(labels["composition_shortcut_state"], COMPOSITION_SHORTCUT_SUPPORTED)
        self.assertIn(labels["behavior_only_gru_q_probe"], {BEHAVIOR_ONLY_BELIEF_SUPPORTED, "inconclusive"})
        self.assertIn("behavior_only_gru_q_head", labels)
        self.assertIn("behavior_only_gru_hidden_probe", labels)
        q_probe = next(case for case in payload["cases"] if case["mode"] == "behavior_only_gru_q_probe")
        self.assertGreater(q_probe["matched_composition_summary"]["representation_js_ci"]["bootstrap_lower"], 0.0)
        self.assertTrue(payload["preflight"]["belief_required"])
        self.assertEqual(saved["report_type"], "k.4.0_behavior_required_pomdp")

    def test_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_behavior_required_pomdp_report.py",
                    "--output-dir",
                    str(Path(temp_dir) / "k4"),
                    "--length",
                    "5",
                    "--sequence-count",
                    "128",
                    "--epochs",
                    "30",
                    "--hidden-size",
                    "32",
                    "--batch-size",
                    "128",
                    "--max-order-pairs",
                    "512",
                    "--bootstrap-count",
                    "100",
                    "--correlation-threshold",
                    "0.70",
                    "--action-accuracy-threshold",
                    "0.80",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote k.4.0 behavior-required POMDP report", result.stdout)


if __name__ == "__main__":
    unittest.main()
