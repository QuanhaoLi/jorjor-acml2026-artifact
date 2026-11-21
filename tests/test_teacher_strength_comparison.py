from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jorjor.inspect.teacher_strength import build_teacher_strength_comparison_pack


class TeacherStrengthComparisonTest(unittest.TestCase):
    """Test k.0.7 teacher-strength comparison pack."""

    def test_teacher_strength_pack_preserves_strict_gate_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            teacher_path, student_path, gate_path = _write_fixture_reports(root)
            output_dir = root / "pack"

            payload = build_teacher_strength_comparison_pack(
                teacher_report_path=teacher_path,
                student_report_path=student_path,
                gate_report_path=gate_path,
                output_dir=output_dir,
            )
            report = json.loads((output_dir / "teacher_strength_comparison_pack.json").read_text(encoding="utf-8"))
            markdown = (output_dir / "teacher_strength_comparison_pack.md").read_text(encoding="utf-8")

        self.assertEqual(payload["version"], "k.0.7")
        self.assertEqual(report["checksum"], payload["checksum"])
        self.assertEqual(len(report["rows"]), 1)
        row = report["rows"][0]
        self.assertEqual(row["teacher_family"], "local_limit_abstract_cfr")
        self.assertEqual(row["teacher_strength_win_rate_vs_random"], 0.65)
        self.assertEqual(row["teacher_strength_average_return_vs_random"], 1.22)
        self.assertEqual(row["final_diagnostic_label"], "inconclusive")
        self.assertEqual(row["s6_lower_delta"], -0.1)
        self.assertEqual(row["s7_lower_delta"], -0.05)
        self.assertEqual(report["recommended_next_version"], "k.0.8")
        self.assertIn("| local_limit_abstract_cfr |", markdown)
        self.assertIn("inconclusive", markdown)

    def test_teacher_strength_pack_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            teacher_path, student_path, gate_path = _write_fixture_reports(root)
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_teacher_strength_comparison_pack.py",
                    "--teacher-report",
                    str(teacher_path),
                    "--student-report",
                    str(student_path),
                    "--gate-report",
                    str(gate_path),
                    "--output-dir",
                    str(root / "pack"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote k.0.7 teacher-strength comparison pack", result.stdout)
        self.assertIn("label=inconclusive", result.stdout)


def _write_fixture_reports(root: Path) -> tuple[Path, Path, Path]:
    teacher_path = root / "teacher.json"
    student_path = root / "student.json"
    gate_path = root / "gate.json"
    teacher_path.write_text(
        json.dumps(
            {
                "version": "k.0.4",
                "teacher_family": "local_limit_abstract_cfr",
                "teacher_strength_proxy": {
                    "teacher_win_rate": 0.65,
                    "teacher_average_return": 1.22,
                },
                "policy_entropy": 0.44,
                "decision_point_count": 573,
                "checksum": "teacher-checksum",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    student_path.write_text(
        json.dumps(
            {
                "version": "k.0.5",
                "test_teacher_distribution_kl": 0.28,
                "test_teacher_distribution_cross_entropy": 0.75,
                "test_sampled_action_accuracy": 0.57,
                "test_teacher_argmax_accuracy": 0.71,
                "opponent_hand_supervised": False,
                "probe_only_hidden_labels_used_for_training": False,
                "checksum": "student-checksum",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    gate_path.write_text(
        json.dumps(
            {
                "version": "k.0.6",
                "locked_layer_used": "encoder_layer_2",
                "metrics": {
                    "hidden_probe_top10": 0.25,
                    "label_permutation_hidden_top10": 0.12,
                    "action_value_nuisance_top10": 0.27,
                    "action_value_composition_nuisance_top10": 0.21,
                },
                "gates": {
                    "s5_hidden_vs_action_value": {"delta": -0.02, "delta_ci": [-0.14, 0.09]},
                    "s6_hidden_vs_action_value_composition": {"delta": 0.01, "delta_ci": [-0.1, 0.14]},
                    "s7_matched_hidden_vs_composition_prior": {
                        "delta": 0.09,
                        "delta_ci": [-0.05, 0.29],
                        "matched_point_count": 30,
                    },
                },
                "final_diagnostic_label": "inconclusive",
                "checksum": "gate-checksum",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return teacher_path, student_path, gate_path


if __name__ == "__main__":
    unittest.main()
