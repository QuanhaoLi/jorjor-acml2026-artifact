from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from jorjor.config import load_limit_random_config
from jorjor.data.validation import validate_decision_points
from jorjor.envs.limit_holdem import (
    HAND_169_LABELS,
    LimitHoldemRandomGenerator,
    hand_169_label,
    showdown_winner,
)
from jorjor.limit_random import count_invalid_opponent_labels, flatten_decision_points, run_limit_random_smoke


class LimitHoldemTest(unittest.TestCase):
    """Test the deterministic v0.3.1 Limit Hold'em random adapter."""

    def test_hand_169_labeler_handles_pairs_suited_and_offsuit(self) -> None:
        self.assertEqual(len(HAND_169_LABELS), 169)
        self.assertEqual(len(set(HAND_169_LABELS)), 169)
        self.assertEqual(hand_169_label(("As", "Ah")), "AA")
        self.assertEqual(hand_169_label(("Ad", "Kd")), "AKs")
        self.assertEqual(hand_169_label(("Kc", "Ah")), "AKo")

    def test_showdown_winner_uses_best_five_of_seven(self) -> None:
        self.assertEqual(
            showdown_winner(
                private_cards=(("As", "Ah"), ("Kd", "Qd")),
                public_cards=("2c", "7d", "9h", "Jc", "3s"),
            ),
            0,
        )
        self.assertEqual(
            showdown_winner(
                private_cards=(("2s", "3h"), ("Ad", "Kd")),
                public_cards=("4c", "5d", "6h", "9c", "Ts"),
            ),
            0,
        )

    def test_generation_validates_no_illegal_actions_leakage_or_bad_labels(self) -> None:
        generator = LimitHoldemRandomGenerator(seed=31)
        hands = generator.generate_hands(25)
        points = flatten_decision_points(hands)
        report = validate_decision_points(points)

        self.assertEqual(report.hand_count, 25)
        self.assertEqual(report.illegal_action_count, 0)
        self.assertEqual(report.leakage_count, 0)
        self.assertEqual(count_invalid_opponent_labels(points), 0)
        self.assertGreater(report.decision_point_count, 50)

    def test_load_limit_random_config(self) -> None:
        config = load_limit_random_config(Path("configs/data/limit_random_smoke.json"))

        self.assertEqual(config.run_version, "0.3.1")
        self.assertEqual(config.seed, 31)
        self.assertEqual(config.hand_count, 25)
        self.assertEqual(config.artifact_dir, Path("outputs/limit_random_smoke"))

    def test_run_limit_random_smoke_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_limit_random_smoke(
                config_path=Path("configs/data/limit_random_smoke.json"),
                artifact_dir=Path(temp_dir),
            )
            report_path = result.artifact_dir / "validation_report.json"
            dataset_path = result.artifact_dir / "decision_points.jsonl"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            first_point = json.loads(dataset_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(result.hand_count, 25)
        self.assertEqual(result.illegal_action_count, 0)
        self.assertEqual(result.leakage_count, 0)
        self.assertEqual(result.invalid_opponent_label_count, 0)
        self.assertEqual(result.hand_label_class_count, 169)
        self.assertEqual(report["checksum"], result.checksum)
        self.assertTrue(report["read_back_matches"])
        self.assertIn(first_point["labels"]["action"], first_point["legal_actions"])
        for hidden_card in first_point["labels"]["opponent_private_card"].split():
            self.assertNotIn(hidden_card, first_point["input"]["private_cards"])


if __name__ == "__main__":
    unittest.main()
