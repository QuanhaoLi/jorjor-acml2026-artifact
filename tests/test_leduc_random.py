from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from jorjor.config import load_leduc_random_config
from jorjor.data.validation import validate_decision_points
from jorjor.envs.leduc import HeuristicLeducTeacherPolicy, LeducPolicyState, LeducRandomGenerator, showdown_winner
from jorjor.leduc_random import flatten_decision_points, run_leduc_random_smoke


class LeducRandomTest(unittest.TestCase):
    """Test the deterministic v0.1.0 Leduc random-data path."""

    def test_showdown_winner(self) -> None:
        self.assertEqual(showdown_winner(("K0", "Q0"), "K1"), 0)
        self.assertEqual(showdown_winner(("J0", "Q0"), "Q1"), 1)
        self.assertIsNone(showdown_winner(("J0", "J1"), "Q0"))

    def test_generation_validates_no_illegal_actions_or_leakage(self) -> None:
        generator = LeducRandomGenerator(seed=11)
        hands = generator.generate_hands(25)
        points = flatten_decision_points(hands)
        report = validate_decision_points(points)

        self.assertEqual(report.hand_count, 25)
        self.assertEqual(report.illegal_action_count, 0)
        self.assertEqual(report.leakage_count, 0)
        self.assertGreater(report.decision_point_count, 50)

    def test_teacher_policy_chooses_legal_actions(self) -> None:
        teacher = HeuristicLeducTeacherPolicy()
        call_state = LeducPolicyState(
            hand_id="test",
            acting_player=0,
            private_card="K0",
            public_card=None,
            round_name="preflop",
            pot=4,
            contributions=(3, 1),
            betting_history=(),
            legal_actions=("fold", "call"),
        )
        bet_state = LeducPolicyState(
            hand_id="test",
            acting_player=0,
            private_card="J0",
            public_card="Q0",
            round_name="public",
            pot=2,
            contributions=(1, 1),
            betting_history=(),
            legal_actions=("check", "bet"),
        )

        self.assertEqual(teacher.choose_action(call_state), "call")
        self.assertEqual(teacher.choose_action(bet_state), "check")

    def test_load_leduc_random_config(self) -> None:
        config = load_leduc_random_config(Path("configs/leduc_random_smoke.json"))

        self.assertEqual(config.seed, 11)
        self.assertEqual(config.hand_count, 25)
        self.assertEqual(config.artifact_dir, Path("outputs/leduc_random_smoke"))

    def test_run_leduc_random_smoke_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_leduc_random_smoke(
                config_path=Path("configs/leduc_random_smoke.json"),
                artifact_dir=Path(temp_dir),
            )
            report_path = result.artifact_dir / "validation_report.json"
            dataset_path = result.artifact_dir / "decision_points.jsonl"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            first_line = dataset_path.read_text(encoding="utf-8").splitlines()[0]
            first_point = json.loads(first_line)

        self.assertEqual(result.hand_count, 25)
        self.assertEqual(result.illegal_action_count, 0)
        self.assertEqual(result.leakage_count, 0)
        self.assertEqual(report["checksum"], result.checksum)
        self.assertIn(first_point["labels"]["action"], first_point["legal_actions"])
        self.assertNotEqual(
            first_point["input"]["private_card"],
            first_point["labels"]["opponent_private_card"],
        )


if __name__ == "__main__":
    unittest.main()
