from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from jorjor.config import load_limit_abstraction_audit_config
from jorjor.envs.limit_holdem import LimitHoldemRandomGenerator
from jorjor.limit_abstraction import (
    abstract_information_set_key,
    private_hand_bucket,
    public_board_bucket,
    run_limit_abstraction_audit,
)
from jorjor.limit_random import flatten_decision_points


class LimitAbstractionTest(unittest.TestCase):
    """Test v0.6.0 Limit abstraction audit."""

    def test_load_limit_abstraction_audit_config(self) -> None:
        config = load_limit_abstraction_audit_config(Path("configs/data/limit_abstraction_audit_smoke.json"))

        self.assertEqual(config.run_version, "0.6.0")
        self.assertEqual(config.data_source, "teacher")
        self.assertEqual(config.hand_count, 1000)
        self.assertEqual(config.seed, 167)

    def test_buckets_are_deterministic(self) -> None:
        self.assertEqual(private_hand_bucket(("Ah", "Ad")), "premium_pair")
        self.assertEqual(private_hand_bucket(("Ah", "Kh")), "broadway_suited")
        self.assertEqual(private_hand_bucket(("7h", "2c")), "trash")
        self.assertEqual(public_board_bucket(()), "preflop")
        self.assertEqual(public_board_bucket(("Ah", "Kd", "Qc")), "broadway_flop")
        self.assertEqual(public_board_bucket(("2h", "7d", "9c", "Ts")), "turn")
        self.assertEqual(public_board_bucket(("2h", "7d", "9c", "Ts", "Jd")), "river")

    def test_abstract_information_set_key_is_stable(self) -> None:
        point = flatten_decision_points(LimitHoldemRandomGenerator(seed=41).generate_hands(1))[0]
        left = abstract_information_set_key(point)
        right = abstract_information_set_key(point)

        self.assertEqual(left, right)
        self.assertIn("round=", left)
        self.assertIn("private=", left)
        self.assertIn("legal=", left)

    def test_run_limit_abstraction_audit_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_limit_abstraction_audit(
                config_path=Path("configs/data/limit_abstraction_audit_smoke.json"),
                artifact_dir=Path(temp_dir),
                hand_count_override=100,
            )
            report = json.loads((result.artifact_dir / "abstraction_report.json").read_text(encoding="utf-8"))

        self.assertEqual(result.hand_count, 100)
        self.assertEqual(result.illegal_action_count, 0)
        self.assertEqual(result.leakage_count, 0)
        self.assertGreater(result.information_set_count, 0)
        self.assertIn(result.recommendation, {"prototype_abstract_cfr", "use_external_solver"})
        self.assertEqual(report["checksum"], result.checksum)


if __name__ == "__main__":
    unittest.main()
