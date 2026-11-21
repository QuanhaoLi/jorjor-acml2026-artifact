from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from jorjor.config import load_leduc_teacher_config
from jorjor.data.writer import read_decision_points_jsonl
from jorjor.leduc_teacher import (
    evaluate_teacher_against_random,
    generate_teacher_points,
    run_leduc_teacher_smoke,
)


class LeducTeacherTest(unittest.TestCase):
    """Test the v0.2.0 heuristic teacher smoke path."""

    def test_load_leduc_teacher_config(self) -> None:
        config = load_leduc_teacher_config(Path("configs/data/leduc_teacher_smoke.json"))

        self.assertEqual(config.run_version, "0.2.0")
        self.assertEqual(config.seed, 41)
        self.assertEqual(config.hand_count, 100)
        self.assertEqual(config.artifact_dir, Path("outputs/leduc_teacher_smoke"))

    def test_teacher_points_are_deterministic(self) -> None:
        left = generate_teacher_points(hand_count=25, seed=41)
        right = generate_teacher_points(hand_count=25, seed=41)

        self.assertEqual(left, right)
        self.assertGreater(len(left), 50)

    def test_teacher_beats_random_smoke(self) -> None:
        evaluation = evaluate_teacher_against_random(seed=41, hand_count=2000)

        self.assertGreater(evaluation.teacher_win_rate, 0.55)
        self.assertGreater(evaluation.teacher_average_return, 0.0)
        self.assertEqual(evaluation.teacher_player0_hands, 1000)
        self.assertEqual(evaluation.teacher_player1_hands, 1000)

    def test_run_leduc_teacher_smoke_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_leduc_teacher_smoke(
                config_path=Path("configs/data/leduc_teacher_smoke.json"),
                artifact_dir=Path(temp_dir),
            )
            report = json.loads((result.artifact_dir / "teacher_report.json").read_text(encoding="utf-8"))
            points = read_decision_points_jsonl(result.artifact_dir / "decision_points.jsonl")

        self.assertEqual(result.hand_count, 100)
        self.assertEqual(result.illegal_action_count, 0)
        self.assertEqual(result.leakage_count, 0)
        self.assertEqual(report["version"], "0.2.0")
        self.assertEqual(report["read_back_count"], len(points))
        self.assertTrue(report["read_back_matches"])
        self.assertEqual(report["checksum"], result.checksum)


if __name__ == "__main__":
    unittest.main()
