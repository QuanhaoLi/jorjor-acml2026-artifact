from __future__ import annotations

from dataclasses import replace
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from jorjor.data.schema import DecisionPoint
from jorjor.probe.cbps import _input_only_c_full_key, build_representation_behavior_link_report


class CBPSBehaviorLinkTest(unittest.TestCase):
    """Test k.5.5 representation-to-behavior link report."""

    def test_input_only_c_full_key_excludes_current_targets(self) -> None:
        base = DecisionPoint(
            hand_id="h0",
            step_index=1,
            acting_player=0,
            visible_state={
                "acting_player": 0,
                "round": "flop",
                "private_cards": ["As", "Kd"],
                "public_cards": ["2c", "7d", "Jh"],
                "legal_actions": ["bet", "check"],
                "betting_history": [{"type": "action", "action": "call"}],
            },
            legal_actions=("bet", "check"),
            action="bet",
            opponent_private_card="Qs Qh",
            opponent_rank="QQ",
            terminal_return=1,
        )
        changed_targets = replace(base, action="check", terminal_return=-1)

        self.assertEqual(_input_only_c_full_key(base), _input_only_c_full_key(changed_targets))

    def test_report_builder_uses_archived_smoke_checkpoint(self) -> None:
        archive_root = Path("checkpoints/remote_archive/v100s/2dfcb42")
        if not (archive_root / "outputs/limit_ar_action_value_only_seed_k_50k_4layer/model.pt").exists():
            self.skipTest("local archived no-range 50k checkpoint is unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "k55"
            payload = build_representation_behavior_link_report(
                archive_root=archive_root,
                output_dir=output_dir,
                seed_names=("seed_k",),
                bootstrap_count=50,
                max_train_points=1024,
                max_test_points=512,
                probe_epochs=2,
            )
            saved = json.loads(
                (output_dir / "k.5.5_representation_behavior_link_report.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (output_dir / "k.5.5_representation_behavior_link_manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(payload["report_type"], "k.5.5_representation_behavior_link")
        self.assertEqual(saved["target_leakage_guard"], "composition baselines exclude current action label and terminal value target")
        self.assertIn("one_command_reproduction", manifest)
        row = payload["rows"][0]
        self.assertEqual(row["seed"], "seed_k")
        self.assertIn("hidden_vs_C_full_linear", row["metrics"]["comparisons"])
        self.assertIn("residual_hidden_vs_C_full_linear", row["metrics"]["comparisons"])
        self.assertNotIn("per_point_action_hits", row["metrics"]["probe_rows"]["hidden"])

    def test_script_runs_directly_for_one_seed(self) -> None:
        archive_root = Path("checkpoints/remote_archive/v100s/2dfcb42")
        if not (archive_root / "outputs/limit_ar_action_value_only_seed_k_50k_4layer/model.pt").exists():
            self.skipTest("local archived no-range 50k checkpoint is unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_representation_behavior_link_report.py",
                    "--archive-root",
                    str(archive_root),
                    "--output-dir",
                    str(Path(temp_dir) / "k55"),
                    "--seed-names",
                    "seed_k",
                    "--bootstrap-count",
                    "50",
                    "--max-train-points",
                    "1024",
                    "--max-test-points",
                    "512",
                    "--probe-epochs",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote k.5.5 representation-behavior link report", result.stdout)


if __name__ == "__main__":
    unittest.main()
