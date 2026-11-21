from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from jorjor.config import load_limit_teacher_config
from jorjor.data.writer import read_decision_points_jsonl
from jorjor.envs.limit_holdem import HeuristicLimitHoldemTeacherPolicy, LimitHoldemPolicyState
from jorjor.limit_teacher import evaluate_teacher_against_random, generate_teacher_points, run_limit_teacher_smoke


class LimitTeacherTest(unittest.TestCase):
    """Test v0.4.2 Limit heuristic teacher data path."""

    def test_teacher_policy_chooses_legal_actions(self) -> None:
        teacher = HeuristicLimitHoldemTeacherPolicy()
        raise_state = LimitHoldemPolicyState(
            hand_id="teacher",
            acting_player=0,
            private_cards=("As", "Ah"),
            public_cards=(),
            round_name="preflop",
            pot=2,
            contributions=(1, 1),
            betting_history=(),
            legal_actions=("check", "bet"),
        )
        call_state = LimitHoldemPolicyState(
            hand_id="teacher",
            acting_player=0,
            private_cards=("2c", "7d"),
            public_cards=("As", "Kd", "Qh"),
            round_name="flop",
            pot=4,
            contributions=(1, 3),
            betting_history=(),
            legal_actions=("fold", "call"),
        )

        self.assertEqual(teacher.choose_action(raise_state), "bet")
        self.assertEqual(teacher.choose_action(call_state), "fold")

    def test_load_limit_teacher_config(self) -> None:
        config = load_limit_teacher_config(Path("configs/data/limit_teacher_smoke.json"))

        self.assertEqual(config.run_version, "0.4.2")
        self.assertEqual(config.seed, 67)
        self.assertEqual(config.hand_count, 100)
        self.assertEqual(config.evaluation_hand_count, 400)

    def test_teacher_points_are_deterministic(self) -> None:
        left = generate_teacher_points(hand_count=25, seed=67)
        right = generate_teacher_points(hand_count=25, seed=67)

        self.assertEqual(left, right)
        self.assertGreater(len(left), 50)

    def test_teacher_beats_random_smoke(self) -> None:
        evaluation = evaluate_teacher_against_random(seed=67, hand_count=400)

        self.assertGreater(evaluation.teacher_average_return, 0.0)
        self.assertEqual(evaluation.teacher_player0_hands, 200)
        self.assertEqual(evaluation.teacher_player1_hands, 200)

    def test_run_limit_teacher_smoke_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_limit_teacher_smoke(
                config_path=Path("configs/data/limit_teacher_smoke.json"),
                artifact_dir=Path(temp_dir),
            )
            report = json.loads((result.artifact_dir / "teacher_report.json").read_text(encoding="utf-8"))
            points = read_decision_points_jsonl(result.artifact_dir / "decision_points.jsonl")

        self.assertEqual(result.hand_count, 100)
        self.assertEqual(result.illegal_action_count, 0)
        self.assertEqual(result.leakage_count, 0)
        self.assertGreater(result.teacher_average_return, 0.0)
        self.assertEqual(report["version"], "0.4.2")
        self.assertEqual(report["read_back_count"], len(points))
        self.assertTrue(report["read_back_matches"])
        self.assertEqual(report["checksum"], result.checksum)


if __name__ == "__main__":
    unittest.main()
