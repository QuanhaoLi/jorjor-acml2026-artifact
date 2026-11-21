from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jorjor.envs.synthetic_belief import (
    BELIEF_LIKE_ORACLE_SUPPORTED,
    build_synthetic_oracle_belief_report,
    exact_posterior,
)
from jorjor.envs.toy_imperfect_info import build_toy_imperfect_info_oracle_report
from jorjor.train.synthetic_belief import build_explicit_belief_state_report, build_learned_synthetic_belief_report
from jorjor.probe.inspect import COMPOSITION_SHORTCUT_SUPPORTED


class SyntheticBeliefTest(unittest.TestCase):
    """Test synthetic oracle-belief positive and negative controls."""

    def test_exact_posterior_sums_to_one(self) -> None:
        posterior = exact_posterior(("x", "y", "x"))

        self.assertAlmostEqual(sum(posterior), 1.0)
        self.assertEqual(len(posterior), 3)

    def test_report_separates_oracle_belief_from_shortcut(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "synthetic"
            payload = build_synthetic_oracle_belief_report(
                output_dir=output_dir,
                sequence_count=128,
                length=6,
                seed=7,
            )
            saved = json.loads((output_dir / "synthetic_oracle_belief_report.json").read_text(encoding="utf-8"))

        labels = {case["mode"]: case["final_diagnostic_label"] for case in payload["cases"]}
        order_pairs = {
            case["mode"]: case["order_counterfactual_summary"]["pair_count"]
            for case in payload["cases"]
        }
        order_js = {
            case["mode"]: case["order_counterfactual_summary"]["mean_representation_js"]
            for case in payload["cases"]
        }
        self.assertEqual(labels["oracle_belief"], BELIEF_LIKE_ORACLE_SUPPORTED)
        self.assertEqual(labels["composition_shortcut"], COMPOSITION_SHORTCUT_SUPPORTED)
        self.assertGreater(order_pairs["oracle_belief"], 0)
        self.assertGreater(order_js["oracle_belief"], 0.0)
        self.assertEqual(order_js["composition_shortcut"], 0.0)
        self.assertEqual(saved["cases"][0]["mode"], "oracle_belief")

    def test_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_synthetic_oracle_belief_report.py",
                    "--output-dir",
                    str(Path(temp_dir) / "synthetic"),
                    "--sequence-count",
                    "64",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote synthetic oracle-belief report", result.stdout)

    def test_learned_report_separates_ordered_belief_from_shortcut(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "learned"
            payload = build_learned_synthetic_belief_report(
                output_dir=output_dir,
                sequence_count=96,
                length=5,
                seed=11,
                epochs=20,
                hidden_size=16,
                batch_size=64,
                max_order_pairs=256,
            )
            saved = json.loads((output_dir / "learned_synthetic_belief_report.json").read_text(encoding="utf-8"))

        labels = {case["mode"]: case["final_diagnostic_label"] for case in payload["cases"]}
        order_js = {
            case["mode"]: case["order_counterfactual_summary"]["mean_representation_js"]
            for case in payload["cases"]
        }
        self.assertEqual(labels["ordered_belief_ar"], BELIEF_LIKE_ORACLE_SUPPORTED)
        self.assertEqual(labels["composition_biased_ar"], COMPOSITION_SHORTCUT_SUPPORTED)
        self.assertGreater(order_js["ordered_belief_ar"], 0.0)
        self.assertEqual(order_js["composition_biased_ar"], 0.0)
        self.assertEqual(saved["report_type"], "learned_synthetic_belief")

    def test_learned_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_learned_synthetic_belief_report.py",
                    "--output-dir",
                    str(Path(temp_dir) / "learned"),
                    "--sequence-count",
                    "64",
                    "--length",
                    "5",
                    "--epochs",
                    "10",
                    "--hidden-size",
                    "12",
                    "--max-order-pairs",
                    "128",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote learned synthetic belief report", result.stdout)

    def test_toy_imperfect_info_oracle_report_separates_belief_from_shortcut(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "toy"
            payload = build_toy_imperfect_info_oracle_report(
                output_dir=output_dir,
                sequence_count=96,
                length=5,
                seed=17,
                max_order_pairs=256,
            )
            saved = json.loads((output_dir / "toy_imperfect_info_oracle_report.json").read_text(encoding="utf-8"))

        labels = {case["mode"]: case["final_diagnostic_label"] for case in payload["cases"]}
        order_js = {
            case["mode"]: case["order_counterfactual_summary"]["mean_representation_js"]
            for case in payload["cases"]
        }
        self.assertEqual(labels["oracle_belief"], BELIEF_LIKE_ORACLE_SUPPORTED)
        self.assertEqual(labels["composition_shortcut"], COMPOSITION_SHORTCUT_SUPPORTED)
        self.assertGreater(order_js["oracle_belief"], 0.0)
        self.assertEqual(order_js["composition_shortcut"], 0.0)
        self.assertEqual(saved["report_type"], "toy_imperfect_info_oracle")

    def test_toy_imperfect_info_oracle_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_toy_imperfect_info_oracle_report.py",
                    "--output-dir",
                    str(Path(temp_dir) / "toy"),
                    "--sequence-count",
                    "64",
                    "--length",
                    "5",
                    "--max-order-pairs",
                    "128",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote toy imperfect-info oracle report", result.stdout)

    def test_explicit_belief_state_report_passes_strict_positive_control(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "explicit"
            payload = build_explicit_belief_state_report(
                output_dir=output_dir,
                sequence_count=96,
                length=5,
                seed=23,
                max_order_pairs=256,
                bootstrap_count=200,
                bootstrap_seed=31,
            )
            saved = json.loads((output_dir / "explicit_belief_state_report.json").read_text(encoding="utf-8"))

        labels = {case["mode"]: case["final_diagnostic_label"] for case in payload["cases"]}
        intervals = {
            case["mode"]: case["order_counterfactual_summary"]["representation_js_ci"]
            for case in payload["cases"]
        }
        self.assertEqual(labels["explicit_belief_state_rnn"], BELIEF_LIKE_ORACLE_SUPPORTED)
        self.assertEqual(labels["composition_state_rnn"], COMPOSITION_SHORTCUT_SUPPORTED)
        self.assertGreater(intervals["explicit_belief_state_rnn"]["bootstrap_lower"], 0.0)
        self.assertEqual(intervals["composition_state_rnn"]["bootstrap_upper"], 0.0)
        self.assertEqual(saved["report_type"], "explicit_belief_state_positive_control")

    def test_explicit_belief_state_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_explicit_belief_state_report.py",
                    "--output-dir",
                    str(Path(temp_dir) / "explicit"),
                    "--sequence-count",
                    "64",
                    "--length",
                    "5",
                    "--max-order-pairs",
                    "128",
                    "--bootstrap-count",
                    "100",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote explicit belief-state report", result.stdout)


if __name__ == "__main__":
    unittest.main()
