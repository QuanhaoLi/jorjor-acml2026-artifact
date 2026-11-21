from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jorjor.probe.teacher_distill_gates import build_strong_teacher_gate_report
from jorjor.teacher_artifact import build_limit_abstract_cfr_teacher_artifact
from jorjor.train.teacher_distill import run_limit_teacher_distill_smoke


class TeacherDistillGatesTest(unittest.TestCase):
    """Test k.0.6 strict gates for the strong-teacher distilled student."""

    def test_strong_teacher_gate_report_writes_strict_gate_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            teacher_dir, student_dir = _build_smoke_inputs(root)
            output_dir = root / "gates"

            payload = build_strong_teacher_gate_report(
                artifact_rows_path=teacher_dir / "teacher_artifact_rows.jsonl",
                student_metrics_path=student_dir / "metrics.json",
                model_path=student_dir / "model.pt",
                output_dir=output_dir,
                probe_epochs=3,
                bootstrap_count=32,
                seed=43,
                locked_layer="encoder_layer_0",
                minimum_matched_points=2,
            )
            report = json.loads((output_dir / "strong_teacher_gate_report.json").read_text(encoding="utf-8"))
            markdown_exists = (output_dir / "strong_teacher_gate_report.md").exists()

        self.assertEqual(payload["version"], "k.0.6")
        self.assertEqual(report["checksum"], payload["checksum"])
        self.assertTrue(markdown_exists)
        self.assertEqual(report["student_contract"]["opponent_hand_supervised"], False)
        self.assertEqual(report["student_contract"]["probe_only_hidden_labels_used_for_training"], False)
        self.assertEqual(report["locked_layer_requested"], "encoder_layer_0")
        self.assertEqual(report["locked_layer_used"], "encoder_layer_0")
        self.assertIn(report["final_diagnostic_label"], {"realized_label_controls_passed", "composition_shortcut_supported", "inconclusive"})
        for gate in ("s5_hidden_vs_action_value", "s6_hidden_vs_action_value_composition", "s7_matched_hidden_vs_composition_prior"):
            self.assertIn(gate, report["gates"])
            self.assertIn("delta_ci", report["gates"][gate])
            self.assertEqual(len(report["gates"][gate]["delta_ci"]), 2)
        self.assertIn("hidden_probe_top10", report["metrics"])
        self.assertIn("label_permutation_hidden_top10", report["metrics"])

    def test_strong_teacher_gate_report_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            teacher_dir, student_dir = _build_smoke_inputs(root)
            output_dir = root / "gates"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_strong_teacher_gate_report.py",
                    "--artifact-rows",
                    str(teacher_dir / "teacher_artifact_rows.jsonl"),
                    "--student-metrics",
                    str(student_dir / "metrics.json"),
                    "--model-path",
                    str(student_dir / "model.pt"),
                    "--output-dir",
                    str(output_dir),
                    "--probe-epochs",
                    "2",
                    "--bootstrap-count",
                    "16",
                    "--locked-layer",
                    "encoder_layer_0",
                    "--minimum-matched-points",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote k.0.6 strong-teacher gate report", result.stdout)
        self.assertIn("label=", result.stdout)


def _build_smoke_inputs(root: Path) -> tuple[Path, Path]:
    teacher_dir = root / "teacher"
    build_limit_abstract_cfr_teacher_artifact(
        output_dir=teacher_dir,
        iteration_count=4,
        deal_sample_count=8,
        hand_count=14,
        evaluation_hand_count=16,
        seed=47,
    )
    student_dir = root / "student"
    run_limit_teacher_distill_smoke(
        artifact_rows_path=teacher_dir / "teacher_artifact_rows.jsonl",
        output_dir=student_dir,
        epochs=3,
        batch_size=8,
        hidden_size=16,
        num_heads=4,
        num_layers=1,
        seed=53,
        split_seed=59,
        save_model=True,
    )
    return teacher_dir, student_dir


if __name__ == "__main__":
    unittest.main()
