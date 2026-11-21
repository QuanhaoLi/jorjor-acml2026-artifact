from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
import unittest

from jorjor.config import load_limit_tiny_train_config
from jorjor.train.limit_tiny import run_limit_tiny_train


class LimitTinyTrainTest(unittest.TestCase):
    """Test v0.4.1 Limit tiny random training smoke."""

    def test_load_limit_tiny_train_config(self) -> None:
        config = load_limit_tiny_train_config(Path("configs/train/limit_tiny_random_smoke.json"))

        self.assertEqual(config.run_version, "0.4.1")
        self.assertEqual(config.data_source, "random")
        self.assertEqual(config.seed, 53)
        self.assertEqual(config.data_seed, 31)
        self.assertEqual(config.hand_count, 60)
        self.assertEqual(config.epochs, 3)
        self.assertEqual(config.device, "cpu")
        self.assertFalse(config.save_model)

    def test_load_limit_tiny_teacher_train_config(self) -> None:
        config = load_limit_tiny_train_config(Path("configs/train/limit_tiny_teacher_smoke.json"))

        self.assertEqual(config.run_version, "0.4.2")
        self.assertEqual(config.data_source, "teacher")
        self.assertEqual(config.seed, 71)
        self.assertEqual(config.data_seed, 67)
        self.assertEqual(config.hand_count, 60)

    def test_load_limit_tiny_10k_configs(self) -> None:
        random_config = load_limit_tiny_train_config(Path("configs/train/limit_tiny_random_10k.json"))
        teacher_config = load_limit_tiny_train_config(Path("configs/train/limit_tiny_teacher_10k.json"))

        self.assertEqual(random_config.run_version, "0.4.3-random")
        self.assertEqual(random_config.data_source, "random")
        self.assertEqual(random_config.hand_count, 10000)
        self.assertEqual(teacher_config.run_version, "0.4.3-teacher")
        self.assertEqual(teacher_config.data_source, "teacher")
        self.assertEqual(teacher_config.hand_count, 10000)
        self.assertEqual(random_config.split_seed, teacher_config.split_seed)
        self.assertEqual(random_config.hidden_size, teacher_config.hidden_size)
        self.assertEqual(random_config.epochs, teacher_config.epochs)

    def test_run_limit_tiny_train_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_limit_tiny_train(
                config_path=Path("configs/train/limit_tiny_random_smoke.json"),
                artifact_dir=Path(temp_dir),
            )
            metrics = json.loads((result.artifact_dir / "metrics.json").read_text(encoding="utf-8"))

        self.assertGreater(result.train_points, 0)
        self.assertGreater(result.validation_points, 0)
        self.assertGreater(result.test_points, 0)
        self.assertTrue(math.isfinite(result.first_loss))
        self.assertTrue(math.isfinite(result.last_loss))
        self.assertTrue(math.isfinite(result.value_mse))
        self.assertGreaterEqual(result.policy_accuracy, 0.0)
        self.assertLessEqual(result.policy_accuracy, 1.0)
        self.assertGreaterEqual(result.opponent_hand_accuracy, 0.0)
        self.assertLessEqual(result.opponent_hand_accuracy, 1.0)
        self.assertGreaterEqual(result.opponent_hand_top5_accuracy, result.opponent_hand_accuracy)
        self.assertGreaterEqual(result.opponent_hand_top10_accuracy, result.opponent_hand_top5_accuracy)
        self.assertEqual(metrics["version"], "0.4.1")
        self.assertEqual(metrics["illegal_action_count"], 0)
        self.assertEqual(metrics["leakage_count"], 0)
        self.assertEqual(metrics["opponent_hand_classes"], 169)
        self.assertIn("opponent_hand_top5_accuracy", metrics)
        self.assertIn("opponent_hand_top10_accuracy", metrics)
        self.assertEqual(metrics["checksum"], result.checksum)


if __name__ == "__main__":
    unittest.main()
