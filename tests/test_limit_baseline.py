from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from jorjor.config import load_limit_baseline_config
from jorjor.envs.limit_holdem import LimitHoldemRandomGenerator
from jorjor.limit_baseline import (
    evaluate_limit_card_removal_baseline,
    possible_opponent_hands,
    run_limit_baseline_smoke,
)
from jorjor.limit_random import flatten_decision_points


class LimitBaselineTest(unittest.TestCase):
    """Test v0.4.4 Limit card-removal baseline behavior."""

    def test_possible_opponent_hands_exclude_visible_cards(self) -> None:
        point = flatten_decision_points(LimitHoldemRandomGenerator(seed=31).generate_hands(1))[0]
        candidates = possible_opponent_hands(point)
        visible_cards = set(point.visible_state["private_cards"]) | set(point.visible_state["public_cards"])
        hidden_cards = set(point.opponent_private_card.split())

        self.assertGreater(len(candidates), 0)
        self.assertIn(frozenset(hidden_cards), candidates)
        for candidate in candidates:
            self.assertFalse(set(candidate) & visible_cards)

    def test_evaluate_limit_card_removal_baseline(self) -> None:
        points = flatten_decision_points(LimitHoldemRandomGenerator(seed=31).generate_hands(25))
        metrics = evaluate_limit_card_removal_baseline(points)

        self.assertEqual(metrics.count, len(points))
        self.assertGreaterEqual(metrics.top1_accuracy, 0.0)
        self.assertLessEqual(metrics.top1_accuracy, 1.0)
        self.assertGreaterEqual(metrics.top5_accuracy, metrics.top1_accuracy)
        self.assertGreaterEqual(metrics.top10_accuracy, metrics.top5_accuracy)
        self.assertGreater(metrics.cross_entropy, 0.0)

    def test_load_limit_baseline_config(self) -> None:
        config = load_limit_baseline_config(Path("configs/data/limit_baseline_smoke.json"))

        self.assertEqual(config.run_version, "0.4.4")
        self.assertEqual(config.data_source, "teacher")
        self.assertEqual(config.seed, 67)
        self.assertEqual(config.hand_count, 100)

    def test_load_limit_baseline_10k_config(self) -> None:
        config = load_limit_baseline_config(Path("configs/data/limit_baseline_teacher_10k.json"))

        self.assertEqual(config.run_version, "0.4.8")
        self.assertEqual(config.data_source, "teacher")
        self.assertEqual(config.seed, 67)
        self.assertEqual(config.hand_count, 10000)

    def test_run_limit_baseline_smoke_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_limit_baseline_smoke(
                config_path=Path("configs/data/limit_baseline_smoke.json"),
                artifact_dir=Path(temp_dir),
            )
            report = json.loads((result.artifact_dir / "baseline_report.json").read_text(encoding="utf-8"))

        self.assertEqual(result.hand_count, 100)
        self.assertEqual(result.illegal_action_count, 0)
        self.assertEqual(result.leakage_count, 0)
        self.assertGreaterEqual(result.top10_accuracy, result.top5_accuracy)
        self.assertEqual(report["checksum"], result.checksum)


if __name__ == "__main__":
    unittest.main()
