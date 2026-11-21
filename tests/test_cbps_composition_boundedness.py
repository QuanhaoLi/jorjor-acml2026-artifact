from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from jorjor.probe.cbps import build_composition_boundedness_report


class CBPSCompositionBoundednessTest(unittest.TestCase):
    """Test k.5.2 composition-boundedness report assembly."""

    def test_report_builder_uses_proxy_scope_and_writes_manifest(self) -> None:
        archive_root = Path("checkpoints/remote_archive/v100s/2dfcb42")
        matched_report = Path("outputs/inspect/matched_composition_report.json")
        composition_report = Path("outputs/k/k.3.1_composition_feature_sensitivity/k.3.1_composition_feature_sensitivity_report.json")
        if not (archive_root / "outputs/limit_ar_action_value_only_seed_k_50k_4layer/model.pt").exists():
            self.skipTest("local archived no-range 50k checkpoint is unavailable")
        if not matched_report.exists() or not composition_report.exists():
            self.skipTest("required composition artifacts are unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "k52"
            payload = build_composition_boundedness_report(
                matched_report_path=matched_report,
                composition_report_path=composition_report,
                archive_root=archive_root,
                output_dir=output_dir,
                seed_names=("seed_k",),
                bootstrap_count=100,
                max_test_points=2048,
                max_pairs_per_bucket=16,
            )
            saved = json.loads((output_dir / "k.5.2_composition_boundedness_report.json").read_text(encoding="utf-8"))
            manifest = json.loads((output_dir / "k.5.2_composition_boundedness_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["report_type"], "k.5.2_composition_boundedness")
        self.assertEqual(saved["claim_scope"], "poker_composition_boundedness_with_range_label_proxy_not_exact_oracle_posterior")
        self.assertIn("one_command_reproduction", manifest)
        self.assertEqual({row["variant"] for row in payload["composition_ladder_rows"]}, {"C_min", "C_mid", "C_full"})
        row = payload["seed_rows"][0]
        self.assertEqual(row["seed"], "seed_k")
        self.assertGreaterEqual(row["range_label_proxy"]["pair_count"], 0)
        slope = row["range_label_proxy"]["range_label_gap_slope"]
        self.assertEqual(set(slope), {"point", "lower", "upper"})

    def test_script_runs_directly_for_one_seed(self) -> None:
        archive_root = Path("checkpoints/remote_archive/v100s/2dfcb42")
        matched_report = Path("outputs/inspect/matched_composition_report.json")
        composition_report = Path("outputs/k/k.3.1_composition_feature_sensitivity/k.3.1_composition_feature_sensitivity_report.json")
        if not (archive_root / "outputs/limit_ar_action_value_only_seed_k_50k_4layer/model.pt").exists():
            self.skipTest("local archived no-range 50k checkpoint is unavailable")
        if not matched_report.exists() or not composition_report.exists():
            self.skipTest("required composition artifacts are unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_composition_boundedness_report.py",
                    "--matched-report",
                    str(matched_report),
                    "--composition-report",
                    str(composition_report),
                    "--archive-root",
                    str(archive_root),
                    "--output-dir",
                    str(Path(temp_dir) / "k52"),
                    "--seed-names",
                    "seed_k",
                    "--bootstrap-count",
                    "100",
                    "--max-test-points",
                    "2048",
                    "--max-pairs-per-bucket",
                    "16",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote k.5.2 composition-boundedness report", result.stdout)


if __name__ == "__main__":
    unittest.main()
