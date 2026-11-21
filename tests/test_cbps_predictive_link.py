from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from jorjor.probe.cbps import build_poker_predictive_link_report


class CBPSPredictiveLinkTest(unittest.TestCase):
    """Test k.5.1 poker predictive-link report assembly."""

    def test_report_builder_uses_archived_smoke_checkpoint(self) -> None:
        archive_root = Path("checkpoints/remote_archive/v100s/2dfcb42")
        if not (archive_root / "outputs/limit_ar_action_value_only_seed_k_50k_4layer/model.pt").exists():
            self.skipTest("local archived no-range 50k checkpoint is unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "k51"
            payload = build_poker_predictive_link_report(
                archive_root=archive_root,
                output_dir=output_dir,
                seed_names=("seed_k",),
                bootstrap_count=100,
                max_test_points=1024,
                cache_dir=output_dir / "cache",
            )
            saved = json.loads((output_dir / "k.5.1_poker_predictive_link_report.json").read_text(encoding="utf-8"))
            cached = build_poker_predictive_link_report(
                archive_root=archive_root,
                output_dir=output_dir,
                seed_names=("seed_k",),
                bootstrap_count=100,
                max_test_points=1024,
                cache_dir=output_dir / "cache",
            )

        self.assertEqual(payload["report_type"], "k.5.1_poker_predictive_link")
        self.assertEqual(saved["report_type"], "k.5.1_poker_predictive_link")
        self.assertEqual(len(payload["rows"]), 1)
        row = payload["rows"][0]
        self.assertEqual(row["seed"], "seed_k")
        self.assertEqual(row["cache_status"], "miss")
        self.assertGreater(row["behavior_metrics"]["model_action_accuracy"], 0.0)
        self.assertIn("C_min", row["behavior_metrics"]["baseline_rows"])
        self.assertEqual(cached["rows"][0]["cache_status"], "hit")

    def test_script_runs_directly_for_one_seed(self) -> None:
        archive_root = Path("checkpoints/remote_archive/v100s/2dfcb42")
        if not (archive_root / "outputs/limit_ar_action_value_only_seed_k_50k_4layer/model.pt").exists():
            self.skipTest("local archived no-range 50k checkpoint is unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_poker_predictive_link_report.py",
                    "--archive-root",
                    str(archive_root),
                    "--output-dir",
                    str(Path(temp_dir) / "k51"),
                    "--seed-names",
                    "seed_k",
                    "--bootstrap-count",
                    "100",
                    "--max-test-points",
                    "1024",
                    "--cache-dir",
                    str(Path(temp_dir) / "cache"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote k.5.1 poker predictive-link report", result.stdout)


if __name__ == "__main__":
    unittest.main()
