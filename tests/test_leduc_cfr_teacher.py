from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from jorjor.config import load_leduc_cfr_teacher_config
from jorjor.data.writer import read_decision_points_jsonl
from jorjor.leduc_cfr_teacher import (
    LeducCfrPolicy,
    run_leduc_cfr_teacher_smoke,
    train_leduc_cfr,
)


class LeducCfrTeacherTest(unittest.TestCase):
    """Test v0.5.0 Leduc tabular CFR teacher smoke."""

    def test_load_leduc_cfr_teacher_config(self) -> None:
        config = load_leduc_cfr_teacher_config(Path("configs/data/leduc_cfr_teacher_smoke.json"))

        self.assertEqual(config.run_version, "0.5.0")
        self.assertEqual(config.seed, 149)
        self.assertEqual(config.iteration_count, 200)
        self.assertEqual(config.hand_count, 100)
        self.assertEqual(config.evaluation_hand_count, 500)

    def test_train_leduc_cfr_is_deterministic_and_normalized(self) -> None:
        left = train_leduc_cfr(iteration_count=50, seed=11)
        right = train_leduc_cfr(iteration_count=50, seed=11)

        self.assertEqual(left.policy_table, right.policy_table)
        self.assertGreater(left.information_set_count, 0)
        for action_probs in left.policy_table.values():
            self.assertAlmostEqual(sum(action_probs.values()), 1.0)
            self.assertTrue(all(probability >= 0.0 for probability in action_probs.values()))

    def test_cfr_policy_selects_legal_actions(self) -> None:
        result = train_leduc_cfr(iteration_count=50, seed=11)
        policy = LeducCfrPolicy(result.policy_table, seed=13)

        points = policy.generate_points(hand_count=25, seed=17)

        self.assertGreater(len(points), 50)
        for point in points:
            self.assertIn(point.action, point.legal_actions)

    def test_run_leduc_cfr_teacher_smoke_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_leduc_cfr_teacher_smoke(
                config_path=Path("configs/data/leduc_cfr_teacher_smoke.json"),
                artifact_dir=Path(temp_dir),
            )
            report = json.loads((result.artifact_dir / "cfr_teacher_report.json").read_text(encoding="utf-8"))
            policy = json.loads((result.artifact_dir / "average_policy.json").read_text(encoding="utf-8"))
            points = read_decision_points_jsonl(result.artifact_dir / "decision_points.jsonl")

        self.assertEqual(result.hand_count, 100)
        self.assertGreater(result.information_set_count, 0)
        self.assertEqual(result.illegal_action_count, 0)
        self.assertEqual(result.leakage_count, 0)
        self.assertGreater(result.teacher_average_return, 0.0)
        self.assertEqual(report["checksum"], result.checksum)
        self.assertEqual(report["read_back_count"], len(points))
        self.assertGreater(len(policy["policy_table"]), 0)


if __name__ == "__main__":
    unittest.main()
