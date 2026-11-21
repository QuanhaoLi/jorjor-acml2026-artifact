from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from jorjor.config import load_leduc_random_baseline_config
from jorjor.data.baseline import (
    card_removal_probability,
    evaluate_card_removal_baseline,
    possible_opponent_cards,
)
from jorjor.data.split import (
    SplitFractions,
    apply_split,
    build_split_manifest,
    has_split_overlap,
)
from jorjor.data.writer import read_decision_points_jsonl, write_decision_points_jsonl
from jorjor.envs.leduc import LeducRandomGenerator
from jorjor.leduc_random import flatten_decision_points
from jorjor.leduc_random_baseline import run_leduc_random_baseline


class LeducRandomBaselineTest(unittest.TestCase):
    """Test the v0.1.1 reader, split, and visible-information baseline."""

    def _points(self):
        generator = LeducRandomGenerator(seed=11)
        return flatten_decision_points(generator.generate_hands(25))

    def test_jsonl_reader_round_trip(self) -> None:
        points = self._points()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "points.jsonl"
            write_decision_points_jsonl(path, points)
            read_points = read_decision_points_jsonl(path)

        self.assertEqual(points, read_points)

    def test_split_is_deterministic_and_has_no_overlap(self) -> None:
        points = self._points()
        fractions = SplitFractions(train=0.8, validation=0.1, test=0.1)
        left = build_split_manifest(points, fractions=fractions, seed=19)
        right = build_split_manifest(points, fractions=fractions, seed=19)
        splits = apply_split(points, left)

        self.assertEqual(left, right)
        self.assertFalse(has_split_overlap(left))
        self.assertEqual(len(left.train_hand_ids), 20)
        self.assertEqual(len(left.validation_hand_ids), 2)
        self.assertEqual(len(left.test_hand_ids), 3)
        self.assertEqual(len(splits.train) + len(splits.validation) + len(splits.test), len(points))

    def test_card_removal_baseline_uses_visible_cards(self) -> None:
        points = self._points()
        pre_public = points[0]
        post_public = next(point for point in points if point.visible_state["public_card"] is not None)

        self.assertEqual(len(possible_opponent_cards(pre_public)), 5)
        self.assertEqual(card_removal_probability(pre_public), 0.2)
        self.assertEqual(len(possible_opponent_cards(post_public)), 4)
        self.assertEqual(card_removal_probability(post_public), 0.25)

    def test_evaluate_card_removal_baseline(self) -> None:
        metrics = evaluate_card_removal_baseline(self._points())

        self.assertEqual(metrics.count, 104)
        self.assertGreaterEqual(metrics.accuracy, 0.0)
        self.assertLessEqual(metrics.accuracy, 1.0)
        self.assertGreater(metrics.cross_entropy, 0.0)

    def test_load_leduc_random_baseline_config(self) -> None:
        config = load_leduc_random_baseline_config(Path("configs/leduc_random_baseline.json"))

        self.assertEqual(config.seed, 11)
        self.assertEqual(config.hand_count, 25)
        self.assertEqual(config.split_seed, 19)
        self.assertEqual(config.artifact_dir, Path("outputs/leduc_random_baseline"))

    def test_run_leduc_random_baseline_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_leduc_random_baseline(
                config_path=Path("configs/leduc_random_baseline.json"),
                artifact_dir=Path(temp_dir),
            )
            report = json.loads((result.artifact_dir / "baseline_report.json").read_text(encoding="utf-8"))
            manifest = json.loads((result.artifact_dir / "split_manifest.json").read_text(encoding="utf-8"))
            points = read_decision_points_jsonl(result.artifact_dir / "decision_points.jsonl")

        self.assertEqual(result.hand_count, 25)
        self.assertEqual(result.decision_point_count, 104)
        self.assertEqual(result.illegal_action_count, 0)
        self.assertEqual(result.leakage_count, 0)
        self.assertFalse(result.split_overlap)
        self.assertEqual(report["checksum"], result.checksum)
        self.assertEqual(manifest["counts"]["train_hands"], 20)
        self.assertEqual(len(points), 104)


if __name__ == "__main__":
    unittest.main()
