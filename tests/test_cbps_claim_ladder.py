from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from jorjor.inspect.cbps_report import build_cbps_claim_ladder


class CBPSClaimLadderTest(unittest.TestCase):
    """Test k.5.3 CBPS claim-ladder assembly."""

    def test_builder_outputs_claim_safe_cbps_row(self) -> None:
        paths = _input_paths()
        if not all(path.exists() for path in paths.values()):
            self.skipTest("k.5 input reports are unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "k53"
            payload = build_cbps_claim_ladder(
                synthetic_cbps_report=paths["synthetic"],
                predictive_link_report=paths["predictive"],
                composition_boundedness_report=paths["bounded"],
                poker_oracle_report=paths["oracle"],
                composition_sensitivity_report=paths["sensitivity"],
                threshold_manifest=paths["threshold"],
                capacity_report=paths["capacity"],
                behavior_link_report=paths["behavior"],
                output_dir=output_dir,
                git_commit="test-commit",
            )
            saved = json.loads((output_dir / "k.5.3_cbps_claim_ladder.json").read_text(encoding="utf-8"))
            table = (output_dir / "paper_table_cbps_claim_ladder.md").read_text(encoding="utf-8")

        self.assertEqual(payload["report_type"], "k.5.3_cbps_claim_ladder")
        self.assertEqual(saved["final_label"], "cbps_supported")
        poker = [row for row in payload["rows"] if row["domain"] == "limit_holdem"][0]
        self.assertEqual(poker["claim_scope"], "diagnostic_interpretation_not_ordered_belief")
        self.assertEqual(poker["matched_range_label_proxy_gate"], "range_label_proxy_tracking_not_supported")
        self.assertEqual(poker["representation_behavior_link_gate"], "composition_bounded_behavior_link")
        self.assertEqual(poker["capacity_sensitivity"], "mlp_inconclusive")
        self.assertEqual(payload["gaps"], [])
        self.assertIn("cbps_supported", table)
        self.assertIn("composition_bounded_behavior_link", table)

    def test_script_runs_directly(self) -> None:
        paths = _input_paths()
        if not all(path.exists() for path in paths.values()):
            self.skipTest("k.5 input reports are unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_cbps_claim_ladder.py",
                    "--synthetic-cbps-report",
                    str(paths["synthetic"]),
                    "--predictive-link-report",
                    str(paths["predictive"]),
                    "--composition-boundedness-report",
                    str(paths["bounded"]),
                    "--poker-oracle-report",
                    str(paths["oracle"]),
                    "--composition-sensitivity-report",
                    str(paths["sensitivity"]),
                    "--threshold-manifest",
                    str(paths["threshold"]),
                    "--capacity-report",
                    str(paths["capacity"]),
                    "--behavior-link-report",
                    str(paths["behavior"]),
                    "--output-dir",
                    str(Path(temp_dir) / "k53"),
                    "--git-commit",
                    "test-commit",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote k.5.6 CBPS claim ladder", result.stdout)


def _input_paths() -> dict[str, Path]:
    return {
        "synthetic": Path("outputs/k/k.5.0_synthetic_cbps_control/k.5.0_synthetic_cbps_report.json"),
        "predictive": Path("outputs/k/k.5.1_full_poker_predictive_link/k.5.1_poker_predictive_link_report.json"),
        "bounded": Path("outputs/k/k.5.2_composition_boundedness/k.5.2_composition_boundedness_report.json"),
        "oracle": Path("outputs/k/k.3.0_poker_oracle_positive_control/k.3.0_poker_oracle_positive_control_report.json"),
        "sensitivity": Path(
            "outputs/k/k.3.1_composition_feature_sensitivity/k.3.1_composition_feature_sensitivity_report.json"
        ),
        "threshold": Path("outputs/k/k.3.3_g1_g4_threshold_manifest/k.3.3_g1_g4_threshold_manifest.json"),
        "capacity": Path("outputs/k/k.2.3_strict_tables/probe_capacity_s6_s7_report.json"),
        "behavior": Path("outputs/k/k.5.5_representation_behavior_link/k.5.5_representation_behavior_link_report.json"),
    }


if __name__ == "__main__":
    unittest.main()
