from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jorjor.teacher_artifact import build_limit_abstract_cfr_teacher_artifact
from jorjor.train.teacher_distill import run_limit_teacher_distill_smoke


class TeacherDistillTest(unittest.TestCase):
    """Test k.0.5 strong-teacher AR distillation smoke."""

    def test_limit_teacher_distill_smoke_writes_policy_distribution_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            teacher_dir = root / "teacher"
            build_limit_abstract_cfr_teacher_artifact(
                output_dir=teacher_dir,
                iteration_count=4,
                deal_sample_count=8,
                hand_count=12,
                evaluation_hand_count=16,
                seed=29,
            )

            output_dir = root / "student"
            result = run_limit_teacher_distill_smoke(
                artifact_rows_path=teacher_dir / "teacher_artifact_rows.jsonl",
                output_dir=output_dir,
                epochs=3,
                batch_size=8,
                hidden_size=16,
                num_heads=4,
                num_layers=1,
                seed=31,
                save_model=True,
            )
            report = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
            train_log_exists = (output_dir / "train_log.jsonl").exists()
            model_exists = (output_dir / "model.pt").exists()

        self.assertEqual(result.report["version"], "k.0.5")
        self.assertEqual(report["checksum"], result.report["checksum"])
        self.assertEqual(report["claim_boundary"], "student_distillation_smoke_only; does_not_prove_belief_tracking")
        self.assertFalse(report["opponent_hand_supervised"])
        self.assertFalse(report["probe_only_hidden_labels_used_for_training"])
        self.assertEqual(report["tokenizer_hidden_label_policy"], "dummy_valid_labels_for_unused_tokenizer_fields")
        self.assertTrue(report["teacher_artifact_validation"]["passed"])
        self.assertEqual(report["teacher_artifact_validation"]["student_input_forbidden_field_count"], 0)
        self.assertGreater(report["train_points"], 0)
        self.assertGreater(report["test_points"], 0)
        self.assertGreaterEqual(report["last_loss"], 0.0)
        self.assertLessEqual(report["last_loss"], report["first_loss"] + 0.5)
        self.assertGreaterEqual(report["test_teacher_distribution_kl"], 0.0)
        self.assertGreater(report["test_teacher_distribution_cross_entropy"], 0.0)
        self.assertGreaterEqual(report["test_sampled_action_accuracy"], 0.0)
        self.assertLessEqual(report["test_sampled_action_accuracy"], 1.0)
        self.assertGreaterEqual(report["test_teacher_argmax_accuracy"], 0.0)
        self.assertLessEqual(report["test_teacher_argmax_accuracy"], 1.0)
        self.assertTrue(train_log_exists)
        self.assertTrue(model_exists)

    def test_limit_teacher_distill_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            teacher_dir = root / "teacher"
            build_limit_abstract_cfr_teacher_artifact(
                output_dir=teacher_dir,
                iteration_count=4,
                deal_sample_count=8,
                hand_count=10,
                evaluation_hand_count=16,
                seed=37,
            )
            output_dir = root / "student"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/train_limit_teacher_distill_smoke.py",
                    "--artifact-rows",
                    str(teacher_dir / "teacher_artifact_rows.jsonl"),
                    "--output-dir",
                    str(output_dir),
                    "--epochs",
                    "2",
                    "--batch-size",
                    "8",
                    "--hidden-size",
                    "16",
                    "--num-heads",
                    "4",
                    "--num-layers",
                    "1",
                    "--seed",
                    "41",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote k.0.5 Limit teacher distillation smoke", result.stdout)
        self.assertIn("kl=", result.stdout)


if __name__ == "__main__":
    unittest.main()
