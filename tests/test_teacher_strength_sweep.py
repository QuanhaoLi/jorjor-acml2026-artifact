from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jorjor.inspect.teacher_strength_sweep import build_teacher_strength_seed_scale_pack


class TeacherStrengthSweepTest(unittest.TestCase):
    """Test k.0.8 strong-teacher seed and scale robustness pack."""

    def test_seed_scale_pack_runs_full_pipeline_per_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "sweep"
            payload = build_teacher_strength_seed_scale_pack(
                output_dir=output_dir,
                seeds=(101, 103),
                teacher_iteration_count=4,
                teacher_deal_sample_count=8,
                teacher_hand_count=10,
                teacher_evaluation_hand_count=12,
                student_epochs=2,
                student_batch_size=8,
                student_hidden_size=16,
                student_num_heads=4,
                student_num_layers=1,
                probe_epochs=2,
                bootstrap_count=16,
                minimum_matched_points=2,
                local_smoke_only=True,
            )
            report = json.loads((output_dir / "seed_scale_pack.json").read_text(encoding="utf-8"))
            markdown_exists = (output_dir / "seed_scale_pack.md").exists()

        self.assertEqual(payload["version"], "k.0.8")
        self.assertEqual(report["checksum"], payload["checksum"])
        self.assertTrue(markdown_exists)
        self.assertTrue(report["local_smoke_only"])
        self.assertEqual(report["seed_count"], 2)
        self.assertEqual(len(report["rows"]), 2)
        self.assertEqual(
            report["label_counts"]["realized_label_controls_passed"]
            + report["label_counts"]["composition_shortcut_supported"]
            + report["label_counts"]["inconclusive"],
            2,
        )
        for row in report["rows"]:
            self.assertIn("seed", row)
            self.assertIn("teacher_report_path", row)
            self.assertIn("student_report_path", row)
            self.assertIn("gate_report_path", row)
            self.assertIn(row["final_diagnostic_label"], report["label_counts"])
        self.assertIn(report["robustness_label"], {"robust_belief_like", "robust_shortcut", "mixed_or_inconclusive"})

    def test_seed_scale_pack_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "sweep"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_teacher_strength_seed_scale_pack.py",
                    "--output-dir",
                    str(output_dir),
                    "--seeds",
                    "107",
                    "109",
                    "--teacher-iterations",
                    "4",
                    "--teacher-deal-sample-count",
                    "8",
                    "--teacher-hand-count",
                    "10",
                    "--teacher-evaluation-hand-count",
                    "12",
                    "--student-epochs",
                    "2",
                    "--student-batch-size",
                    "8",
                    "--student-hidden-size",
                    "16",
                    "--student-num-heads",
                    "4",
                    "--student-num-layers",
                    "1",
                    "--probe-epochs",
                    "2",
                    "--bootstrap-count",
                    "16",
                    "--minimum-matched-points",
                    "2",
                    "--local-smoke-only",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote k.0.8 teacher-strength seed/scale pack", result.stdout)
        self.assertIn("seeds=2", result.stdout)


if __name__ == "__main__":
    unittest.main()
