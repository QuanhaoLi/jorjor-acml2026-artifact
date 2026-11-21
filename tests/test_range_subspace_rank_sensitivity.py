from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from jorjor.probe.range_subspace_rank_sensitivity import (
    RANK_SENSITIVITY_NOT_SUPPORTED,
    RANK_SENSITIVE_INCONCLUSIVE,
    RANK_SENSITIVITY_SUPPORTED,
    build_range_subspace_rank_sensitivity_report,
)


ARCHIVE_ROOT = Path("checkpoints/remote_archive/v100s/2dfcb42")
SEED_K_CHECKPOINT = ARCHIVE_ROOT / "outputs/limit_ar_action_value_only_seed_k_50k_4layer/model.pt"


class RangeSubspaceRankSensitivityTest(unittest.TestCase):
    """Test k.6.5 range-subspace rank-sensitivity report."""

    def test_duplicate_rank_grid_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "duplicate"):
                build_range_subspace_rank_sensitivity_report(
                    archive_root=ARCHIVE_ROOT,
                    output_dir=Path(temp_dir) / "k65",
                    seed_names=("seed_k",),
                    rank_grid=(1, 1),
                    bootstrap_count=10,
                    max_train_points=64,
                    max_test_points=32,
                    probe_epochs=1,
                    random_subspace_count=1,
                    probe_top10_floor=0.0,
                )

    def test_report_builder_uses_one_probe_for_rank_grid(self) -> None:
        if not SEED_K_CHECKPOINT.exists():
            self.skipTest("local archived no-range 50k checkpoint is unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "k65"
            payload = build_range_subspace_rank_sensitivity_report(
                archive_root=ARCHIVE_ROOT,
                output_dir=output_dir,
                seed_names=("seed_k",),
                rank_grid=(1, 2, 4),
                bootstrap_count=50,
                max_train_points=1024,
                max_test_points=512,
                probe_epochs=2,
                random_subspace_count=2,
                probe_top10_floor=0.0,
            )
            saved = json.loads(
                (output_dir / "k.6.5_range_subspace_rank_sensitivity_report.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (output_dir / "k.6.5_range_subspace_rank_sensitivity_manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(payload["report_type"], "k.6.5_range_subspace_rank_sensitivity")
        self.assertEqual(saved["version"], "k.6.5")
        self.assertEqual(payload["rank_grid"], [1, 2, 4])
        self.assertIn("one_command_reproduction", manifest)
        self.assertEqual(len(payload["rows"]), 1)
        self.assertEqual(len(payload["rows"][0]["rank_rows"]), 3)
        self.assertEqual(payload["aggregate"]["rank_count"], 3)
        self.assertIn(
            payload["final_diagnostic_label"],
            {RANK_SENSITIVITY_NOT_SUPPORTED, RANK_SENSITIVE_INCONCLUSIVE, RANK_SENSITIVITY_SUPPORTED},
        )
        for rank_row in payload["rows"][0]["rank_rows"]:
            self.assertIn("range_minus_random_policy_js_delta_ci", rank_row["summary"])

    def test_script_runs_directly_for_one_seed(self) -> None:
        if not SEED_K_CHECKPOINT.exists():
            self.skipTest("local archived no-range 50k checkpoint is unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_range_subspace_rank_sensitivity_report.py",
                    "--archive-root",
                    str(ARCHIVE_ROOT),
                    "--output-dir",
                    str(Path(temp_dir) / "k65"),
                    "--seed-names",
                    "seed_k",
                    "--rank-grid",
                    "1",
                    "2",
                    "4",
                    "--bootstrap-count",
                    "50",
                    "--max-train-points",
                    "1024",
                    "--max-test-points",
                    "512",
                    "--probe-epochs",
                    "2",
                    "--random-subspace-count",
                    "2",
                    "--probe-top10-floor",
                    "0.0",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote k.6.5 range-subspace rank-sensitivity report", result.stdout)


if __name__ == "__main__":
    unittest.main()
