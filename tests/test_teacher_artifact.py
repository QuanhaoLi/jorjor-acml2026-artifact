from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jorjor.teacher_artifact import (
    FIXED_LIMIT_ACTION_SPACE,
    build_limit_abstract_cfr_teacher_artifact,
    validate_teacher_artifact_rows,
)


class TeacherArtifactTest(unittest.TestCase):
    """Test k.0.4 Limit abstract CFR teacher artifact rows."""

    def test_limit_abstract_cfr_teacher_artifact_writes_contract_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "teacher_artifact"
            payload = build_limit_abstract_cfr_teacher_artifact(
                output_dir=output_dir,
                iteration_count=4,
                deal_sample_count=8,
                hand_count=8,
                evaluation_hand_count=16,
                seed=17,
            )
            report = json.loads((output_dir / "teacher_artifact_report.json").read_text(encoding="utf-8"))
            rows = [
                json.loads(line)
                for line in (output_dir / "teacher_artifact_rows.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertTrue(payload["teacher_artifact_contract_passed"])
        self.assertEqual(report["checksum"], payload["checksum"])
        self.assertGreater(len(rows), 0)
        row = rows[0]
        self.assertEqual(tuple(row["action_space"]), FIXED_LIMIT_ACTION_SPACE)
        self.assertEqual(sorted(row["teacher_action_distribution"]), sorted(FIXED_LIMIT_ACTION_SPACE))
        self.assertAlmostEqual(sum(row["teacher_action_distribution"].values()), 1.0)
        self.assertIn(row["sampled_action"], row["legal_actions"])
        self.assertIn("teacher_abstraction_bucket", row)
        self.assertIn("hidden_label_for_probe_only", row)
        self.assertNotIn("hidden_label_for_probe_only", row["student_input"])
        self.assertNotIn("teacher_abstraction_bucket", row["student_input"])
        self.assertEqual(payload["validation"]["student_input_forbidden_field_count"], 0)

    def test_validate_teacher_artifact_rows_rejects_hidden_student_input(self) -> None:
        bad_row = {
            "student_input": {"opponent_private_card": "As Ah"},
            "legal_actions": ["check", "bet"],
            "sampled_action": "check",
            "action_space": list(FIXED_LIMIT_ACTION_SPACE),
            "teacher_action_distribution": {action: 0.25 for action in FIXED_LIMIT_ACTION_SPACE},
            "hidden_label_for_probe_only": {"opponent_private_card": "As Ah"},
            "teacher_abstraction_bucket": "bucket",
        }

        report = validate_teacher_artifact_rows([bad_row])

        self.assertEqual(report["student_input_forbidden_field_count"], 1)
        self.assertFalse(report["passed"])

    def test_limit_abstract_cfr_teacher_artifact_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_limit_abstract_cfr_teacher_artifact.py",
                    "--output-dir",
                    str(Path(temp_dir) / "teacher_artifact"),
                    "--iteration-count",
                    "4",
                    "--deal-sample-count",
                    "8",
                    "--hand-count",
                    "8",
                    "--evaluation-hand-count",
                    "16",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote k.0.4 Limit abstract CFR teacher artifact", result.stdout)


if __name__ == "__main__":
    unittest.main()
