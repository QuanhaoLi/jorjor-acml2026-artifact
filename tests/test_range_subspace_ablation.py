from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from jorjor.models.tiny import require_torch
from jorjor.probe.range_subspace_ablation import build_range_subspace_ablation_report, remove_subspace


ARCHIVE_ROOT = Path("checkpoints/remote_archive/v100s/2dfcb42")
SEED_K_CHECKPOINT = ARCHIVE_ROOT / "outputs/limit_ar_action_value_only_seed_k_50k_4layer/model.pt"


class RangeSubspaceAblationTest(unittest.TestCase):
    """Test k.6.3 range-subspace behavior-footprint stress report."""

    def test_remove_subspace_removes_basis_component(self) -> None:
        torch, _ = require_torch()
        basis = torch.eye(3)[:2]
        hidden = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 4.0, 0.5]])

        edited = remove_subspace(hidden, basis)

        self.assertTrue(torch.allclose(edited[:, :2], torch.zeros_like(edited[:, :2]), atol=1e-6))
        self.assertTrue(torch.allclose(edited[:, 2], hidden[:, 2], atol=1e-6))

    def test_report_builder_uses_archived_smoke_checkpoint(self) -> None:
        if not SEED_K_CHECKPOINT.exists():
            self.skipTest("local archived no-range 50k checkpoint is unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "k63"
            payload = build_range_subspace_ablation_report(
                archive_root=ARCHIVE_ROOT,
                output_dir=output_dir,
                seed_names=("seed_k",),
                bootstrap_count=50,
                max_train_points=1024,
                max_test_points=512,
                probe_epochs=2,
                random_subspace_count=2,
                max_subspace_rank=4,
                probe_top10_floor=0.0,
            )
            saved = json.loads((output_dir / "k.6.3_range_subspace_ablation_report.json").read_text(encoding="utf-8"))
            manifest = json.loads(
                (output_dir / "k.6.3_range_subspace_ablation_manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(payload["report_type"], "k.6.3_range_subspace_ablation")
        self.assertEqual(saved["version"], "k.6.3")
        self.assertIn("one_command_reproduction", manifest)
        row = payload["rows"][0]
        self.assertEqual(row["seed"], "seed_k")
        self.assertEqual(row["locked_layer"], "sequence_hidden")
        self.assertLessEqual(row["subspace_rank"], 4)
        self.assertIn("range_ablation_policy_js", row["summary"])
        self.assertIn("range_minus_random_policy_js_delta_ci", row["summary"])

    def test_non_sequence_hidden_layer_is_rejected(self) -> None:
        if not SEED_K_CHECKPOINT.exists():
            self.skipTest("local archived no-range 50k checkpoint is unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "sequence_hidden"):
                build_range_subspace_ablation_report(
                    archive_root=ARCHIVE_ROOT,
                    output_dir=Path(temp_dir) / "k63",
                    seed_names=("seed_k",),
                    bootstrap_count=10,
                    max_train_points=64,
                    max_test_points=32,
                    probe_epochs=1,
                    random_subspace_count=1,
                    max_subspace_rank=2,
                    locked_layer="encoder_layer_2",
                    probe_top10_floor=0.0,
                )

    def test_script_runs_directly_for_one_seed(self) -> None:
        if not SEED_K_CHECKPOINT.exists():
            self.skipTest("local archived no-range 50k checkpoint is unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_range_subspace_ablation_report.py",
                    "--archive-root",
                    str(ARCHIVE_ROOT),
                    "--output-dir",
                    str(Path(temp_dir) / "k63"),
                    "--seed-names",
                    "seed_k",
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
                    "--max-subspace-rank",
                    "4",
                    "--probe-top10-floor",
                    "0.0",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote k.6.3 range-subspace ablation report", result.stdout)


if __name__ == "__main__":
    unittest.main()
