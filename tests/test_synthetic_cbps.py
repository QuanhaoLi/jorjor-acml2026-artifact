from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from jorjor.train.synthetic_cbps import (
    CBPS_SUPPORTED,
    CBPS_STATE,
    COMPOSITION_SHORTCUT_SUPPORTED,
    COMPOSITION_STATE,
    POSTERIOR_STATE,
    POSTERIOR_TRACKING_SUPPORTED,
    build_synthetic_cbps_report,
    cbps_behavior_targets,
    cbps_state,
)


class SyntheticCBPSTest(unittest.TestCase):
    """Test k.5.0 synthetic CBPS known-control diagnostics."""

    def test_cbps_state_is_public_predictive_but_not_raw_composition(self) -> None:
        left = ("a", "b", "c", "d", "a", "b")
        right = ("b", "a", "d", "c", "b", "a")

        self.assertNotEqual(cbps_state(left), cbps_state(right))
        self.assertEqual(
            cbps_behavior_targets(left)["action_label"],
            max(range(3), key=lambda index: cbps_behavior_targets(left)["action_scores"][index]),
        )

    def test_report_separates_posterior_composition_and_cbps_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "k5"
            payload = build_synthetic_cbps_report(
                output_dir=output_dir,
                sequence_count=512,
                length=5,
                max_pairs=1024,
                bootstrap_count=100,
                matched_pair_threshold=256,
            )
            saved = json.loads((output_dir / "k.5.0_synthetic_cbps_report.json").read_text(encoding="utf-8"))

        labels = {case["mode"]: case["final_diagnostic_label"] for case in payload["cases"]}
        self.assertEqual(labels[POSTERIOR_STATE], POSTERIOR_TRACKING_SUPPORTED)
        self.assertEqual(labels[COMPOSITION_STATE], COMPOSITION_SHORTCUT_SUPPORTED)
        self.assertEqual(labels[CBPS_STATE], CBPS_SUPPORTED)
        self.assertTrue(payload["synthetic_cbps_control_passed"])
        self.assertEqual(saved["report_type"], "k.5.0_synthetic_cbps_control")
        cbps_case = next(case for case in payload["cases"] if case["mode"] == CBPS_STATE)
        self.assertGreater(cbps_case["predictive_metrics"]["predictive_action_delta_vs_C_min"], 0.0)
        self.assertLessEqual(
            cbps_case["composition_boundedness"]["matched_posterior_gap_slope_ci"]["lower"],
            0.0,
        )
        self.assertTrue(cbps_case["composition_boundedness"]["posterior_gap_tracking_failed"])

    def test_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_synthetic_cbps_report.py",
                    "--output-dir",
                    str(Path(temp_dir) / "k5"),
                    "--sequence-count",
                    "512",
                    "--length",
                    "5",
                    "--max-pairs",
                    "1024",
                    "--bootstrap-count",
                    "100",
                    "--matched-pair-threshold",
                    "256",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote k.5.0 synthetic CBPS report", result.stdout)


if __name__ == "__main__":
    unittest.main()
