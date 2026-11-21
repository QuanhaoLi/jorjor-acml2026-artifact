from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from jorjor.config import load_limit_abstract_cfr_teacher_config
from jorjor.data.writer import read_decision_points_jsonl
from jorjor.limit_abstract_cfr_teacher import (
    LimitAbstractCfrPolicy,
    run_limit_abstract_cfr_teacher_smoke,
    train_limit_abstract_cfr,
)


class LimitAbstractCfrTeacherTest(unittest.TestCase):
    """Test v0.6.1 Limit abstract CFR teacher smoke."""

    def test_load_limit_abstract_cfr_teacher_config(self) -> None:
        config = load_limit_abstract_cfr_teacher_config(Path("configs/data/limit_abstract_cfr_teacher_smoke.json"))

        self.assertEqual(config.run_version, "0.6.1")
        self.assertEqual(config.iteration_count, 10)
        self.assertEqual(config.deal_sample_count, 20)
        self.assertEqual(config.hand_count, 100)
        self.assertEqual(config.evaluation_hand_count, 100)

    def test_load_limit_abstract_cfr_teacher_scale_config(self) -> None:
        config = load_limit_abstract_cfr_teacher_config(Path("configs/data/limit_abstract_cfr_teacher_scale.json"))

        self.assertEqual(config.run_version, "0.6.2")
        self.assertEqual(config.iteration_count, 80)
        self.assertEqual(config.deal_sample_count, 120)
        self.assertEqual(config.hand_count, 500)
        self.assertEqual(config.evaluation_hand_count, 2000)

    def test_train_limit_abstract_cfr_is_normalized(self) -> None:
        result = train_limit_abstract_cfr(iteration_count=10, deal_sample_count=20, seed=19)

        self.assertGreater(result.information_set_count, 0)
        for action_probs in result.policy_table.values():
            self.assertAlmostEqual(sum(action_probs.values()), 1.0)
            self.assertTrue(all(probability >= 0.0 for probability in action_probs.values()))

    def test_policy_generates_legal_points(self) -> None:
        result = train_limit_abstract_cfr(iteration_count=10, deal_sample_count=20, seed=19)
        policy = LimitAbstractCfrPolicy(result.policy_table, seed=23)
        points = policy.generate_points(hand_count=10, seed=29)

        self.assertGreater(len(points), 10)
        for point in points:
            self.assertIn(point.action, point.legal_actions)

    def test_run_limit_abstract_cfr_teacher_smoke_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_limit_abstract_cfr_teacher_smoke(
                config_path=Path("configs/data/limit_abstract_cfr_teacher_smoke.json"),
                artifact_dir=Path(temp_dir),
            )
            report = json.loads((result.artifact_dir / "abstract_cfr_teacher_report.json").read_text())
            policy = json.loads((result.artifact_dir / "average_policy.json").read_text())
            points = read_decision_points_jsonl(result.artifact_dir / "decision_points.jsonl")

        self.assertEqual(result.hand_count, 100)
        self.assertEqual(result.illegal_action_count, 0)
        self.assertEqual(result.leakage_count, 0)
        self.assertGreater(result.information_set_count, 0)
        self.assertEqual(report["checksum"], result.checksum)
        self.assertEqual(report["read_back_count"], len(points))
        self.assertGreater(len(policy["policy_table"]), 0)


if __name__ == "__main__":
    unittest.main()
