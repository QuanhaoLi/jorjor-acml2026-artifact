from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from jorjor.config import load_limit_prior_baseline_config
from jorjor.limit_prior_baseline import evaluate_grouped_prior, run_limit_prior_baseline
from jorjor.limit_teacher import generate_teacher_points


class LimitPriorBaselineTest(unittest.TestCase):
    """Test v0.4.12 Limit empirical prior baselines."""

    def test_load_limit_prior_baseline_config(self) -> None:
        config = load_limit_prior_baseline_config(Path("configs/data/limit_prior_teacher_10k.json"))

        self.assertEqual(config.run_version, "0.4.12")
        self.assertEqual(config.data_source, "teacher")
        self.assertEqual(config.hand_count, 10000)
        self.assertEqual(config.split_seed, 83)

    def test_evaluate_grouped_prior_metrics_are_monotonic(self) -> None:
        points = generate_teacher_points(hand_count=25, seed=67)
        train = points[: len(points) // 2]
        test = points[len(points) // 2 :]
        metrics = evaluate_grouped_prior(train, test, group_name="global")

        self.assertGreater(metrics.count, 0)
        self.assertGreaterEqual(metrics.top5_accuracy, metrics.top1_accuracy)
        self.assertGreaterEqual(metrics.top10_accuracy, metrics.top5_accuracy)
        self.assertGreater(metrics.cross_entropy, 0.0)

    def test_run_limit_prior_baseline_writes_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_limit_prior_baseline(
                config_path=Path("configs/data/limit_prior_teacher_10k.json"),
                artifact_dir=Path(temp_dir),
                hand_count_override=100,
            )
            report = json.loads((result.artifact_dir / "prior_report.json").read_text(encoding="utf-8"))

        self.assertEqual(result.hand_count, 100)
        self.assertEqual(result.illegal_action_count, 0)
        self.assertEqual(result.leakage_count, 0)
        self.assertIn("global", result.baselines)
        self.assertIn("action_round_position", result.baselines)
        self.assertEqual(report["checksum"], result.checksum)


if __name__ == "__main__":
    unittest.main()
