from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.config import load_limit_seed_sweep_config
from jorjor.limit_prior_baseline import run_limit_prior_baseline
from jorjor.limit_seed_sweep import run_limit_seed_sweep


class LimitSeedSweepTest(unittest.TestCase):
    """Test v0.4.13 Limit AR-vs-prior seed sweep reports."""

    def test_load_limit_seed_sweep_config(self) -> None:
        config = load_limit_seed_sweep_config(Path("configs/limit_seed_sweep_10k.json"))

        self.assertEqual(config.run_version, "0.4.13")
        self.assertEqual(len(config.seeds), 3)
        self.assertTrue(config.run_missing_priors)
        self.assertEqual(config.seeds[0].name, "seed_a")

    def test_run_limit_seed_sweep_reads_completed_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ar_dir = root / "ar"
            prior_dir = root / "prior"
            report_dir = root / "sweep"
            config_path = root / "sweep.json"
            prior_config_path = root / "prior.json"
            ar_config_path = root / "ar.json"
            ar_dir.mkdir(parents=True)

            ar_record = {
                "version": "0.4.13-test",
                "hand_count": 100,
                "decision_point_count": 493,
                "illegal_action_count": 0,
                "leakage_count": 0,
                "opponent_hand_accuracy": 0.02,
                "opponent_hand_top5_accuracy": 0.08,
                "opponent_hand_top10_accuracy": 0.16,
                "policy_accuracy": 0.8,
                "value_mse": 1.0,
            }
            write_json_artifact(ar_dir / "metrics.json", {**ar_record, "checksum": checksum_payload(ar_record)})
            prior_result = run_limit_prior_baseline(
                config_path=Path("configs/data/limit_prior_teacher_seed_a_10k.json"),
                artifact_dir=prior_dir,
                hand_count_override=100,
            )
            prior_config_path.write_text(
                json.dumps(
                    {
                        "artifact_dir": str(prior_dir),
                        "data_source": "teacher",
                        "hand_count": 100,
                        "run_version": "0.4.13-test",
                        "seed": 67,
                        "split_seed": 83,
                        "test_fraction": 0.2,
                        "train_fraction": 0.6,
                        "validation_fraction": 0.2,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            ar_config_path.write_text(
                json.dumps(
                    {
                        "artifact_dir": str(ar_dir),
                        "batch_size": 512,
                        "data_seed": 67,
                        "data_source": "teacher",
                        "device": "cpu",
                        "epochs": 6,
                        "hand_count": 100,
                        "hidden_size": 64,
                        "learning_rate": 0.003,
                        "num_heads": 4,
                        "num_layers": 1,
                        "run_version": "0.4.13-test",
                        "save_model": False,
                        "seed": 101,
                        "split_seed": 83,
                        "test_fraction": 0.2,
                        "train_fraction": 0.6,
                        "validation_fraction": 0.2,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            config_path.write_text(
                json.dumps(
                    {
                        "artifact_dir": str(report_dir),
                        "run_missing_priors": False,
                        "run_version": "0.4.13-test",
                        "seeds": [
                            {
                                "name": "seed_a_test",
                                "ar_config": str(ar_config_path),
                                "prior_config": str(prior_config_path),
                            }
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_limit_seed_sweep(config_path=config_path)
            report = json.loads((result.artifact_dir / "sweep_report.json").read_text(encoding="utf-8"))

        expected_best = max(metric.top10_accuracy for metric in prior_result.baselines.values())
        self.assertEqual(result.completed_count, 1)
        self.assertEqual(result.missing_count, 0)
        self.assertAlmostEqual(report["rows"][0]["ar"]["top10"], 0.16)
        self.assertAlmostEqual(report["rows"][0]["best_prior"]["top10"], expected_best)
        self.assertAlmostEqual(report["rows"][0]["delta_vs_best_prior"]["top10"], 0.16 - expected_best)
        self.assertIn("top10_delta_mean", report["aggregate"])

    def test_run_limit_seed_sweep_reports_missing_ar(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "sweep.json"
            config_path.write_text(
                json.dumps(
                    {
                        "artifact_dir": str(root / "report"),
                        "run_missing_priors": False,
                        "run_version": "0.4.13-test",
                        "seeds": [
                            {
                                "name": "missing",
                                "ar_config": "configs/train/limit_ar_teacher_seed_a_10k.json",
                                "prior_config": "configs/data/limit_prior_teacher_seed_a_10k.json",
                            }
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_limit_seed_sweep(config_path=config_path, artifact_dir=root / "report")

        self.assertEqual(result.completed_count, 0)
        self.assertEqual(result.missing_count, 1)


if __name__ == "__main__":
    unittest.main()
