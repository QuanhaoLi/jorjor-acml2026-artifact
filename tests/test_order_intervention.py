from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from jorjor.data.schema import DecisionPoint
from jorjor.probe.order_intervention import INSUFFICIENT_PAIRS, build_order_intervention_report, order_intervention_point


class OrderInterventionTest(unittest.TestCase):
    """Test k.6.0 input-level order-intervention report."""

    def test_order_intervention_preserves_counts_and_last_action(self) -> None:
        point = DecisionPoint(
            hand_id="h0",
            step_index=3,
            acting_player=0,
            visible_state={
                "acting_player": 0,
                "round": "turn",
                "private_cards": ["As", "Kd"],
                "public_cards": ["2c", "7d", "Jh", "4s"],
                "legal_actions": ["bet", "check"],
                "pot": 6,
                "betting_history": [
                    {"type": "action", "action": "bet"},
                    {"type": "action", "action": "call"},
                    {"type": "action", "action": "check"},
                    {"type": "action", "action": "bet"},
                ],
            },
            legal_actions=("bet", "check"),
            action="check",
            opponent_private_card="Qs Qh",
            opponent_rank="QQ",
            terminal_return=1,
        )

        changed = order_intervention_point(point, seed=7)
        self.assertIsNotNone(changed)
        original_actions = [event["action"] for event in point.visible_state["betting_history"]]
        changed_actions = [event["action"] for event in changed.visible_state["betting_history"]]
        self.assertEqual(original_actions[-1], changed_actions[-1])
        self.assertEqual(sorted(original_actions), sorted(changed_actions))
        self.assertNotEqual(original_actions, changed_actions)
        self.assertEqual(point.action, changed.action)
        self.assertEqual(point.terminal_return, changed.terminal_return)

    def test_report_builder_uses_archived_smoke_checkpoint(self) -> None:
        archive_root = Path("checkpoints/remote_archive/v100s/2dfcb42")
        if not (archive_root / "outputs/limit_ar_action_value_only_seed_k_50k_4layer/model.pt").exists():
            self.skipTest("local archived no-range 50k checkpoint is unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "k60"
            payload = build_order_intervention_report(
                archive_root=archive_root,
                output_dir=output_dir,
                seed_names=("seed_k",),
                bootstrap_count=50,
                max_test_points=1024,
                max_pairs=128,
                min_pairs_for_label=32,
            )
            saved = json.loads((output_dir / "k.6.0_order_intervention_report.json").read_text(encoding="utf-8"))
            manifest = json.loads((output_dir / "k.6.0_order_intervention_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["report_type"], "k.6.0_order_intervention")
        self.assertEqual(saved["claim_scope"], "input_level_order_stress_not_legal_poker_causality")
        self.assertIn("one_command_reproduction", manifest)
        self.assertEqual(len(payload["rows"]), 1)
        self.assertIn("mean_policy_js", payload["rows"][0]["summary"])
        self.assertIn("lower", payload["rows"][0]["summary"]["policy_js_ci"])

    def test_low_pair_count_is_not_labeled_as_strong_evidence(self) -> None:
        archive_root = Path("checkpoints/remote_archive/v100s/2dfcb42")
        if not (archive_root / "outputs/limit_ar_action_value_only_seed_k_50k_4layer/model.pt").exists():
            self.skipTest("local archived no-range 50k checkpoint is unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = build_order_intervention_report(
                archive_root=archive_root,
                output_dir=Path(temp_dir) / "k60",
                seed_names=("seed_k",),
                bootstrap_count=20,
                max_test_points=64,
                max_pairs=4,
                min_pairs_for_label=100,
            )

        self.assertEqual(payload["rows"][0]["final_diagnostic_label"], INSUFFICIENT_PAIRS)

    def test_script_runs_directly_for_one_seed(self) -> None:
        archive_root = Path("checkpoints/remote_archive/v100s/2dfcb42")
        if not (archive_root / "outputs/limit_ar_action_value_only_seed_k_50k_4layer/model.pt").exists():
            self.skipTest("local archived no-range 50k checkpoint is unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_order_intervention_report.py",
                    "--archive-root",
                    str(archive_root),
                    "--output-dir",
                    str(Path(temp_dir) / "k60"),
                    "--seed-names",
                    "seed_k",
                    "--bootstrap-count",
                    "50",
                    "--max-test-points",
                    "1024",
                    "--max-pairs",
                    "128",
                    "--min-pairs-for-label",
                    "32",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote k.6.0 order-intervention report", result.stdout)


if __name__ == "__main__":
    unittest.main()
