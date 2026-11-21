from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jorjor.data.schema import DecisionPoint
from jorjor.probe.counterfactual import (
    COMPOSITION_SHORTCUT_SUPPORTED,
    INSUFFICIENT_PAIRS,
    ORDER_SENSITIVE_SUPPORTED,
    build_counterfactual_pairs,
    build_order_counterfactual_report,
    build_order_groups,
    build_prefix_composition_baseline_report,
    composition_key,
    composition_coverage,
    truncate_point_history,
    jensen_shannon,
    order_key,
)


def _point(index: int, actions: list[str], label: str = "AA") -> DecisionPoint:
    history = [
        {"type": "action", "round": "flop", "player": step % 2, "action": action}
        for step, action in enumerate(actions)
    ]
    return DecisionPoint(
        hand_id=f"hand-{index}",
        step_index=index,
        acting_player=0,
        visible_state={
            "game": "limit_holdem_hu",
            "hand_id": f"hand-{index}",
            "acting_player": 0,
            "private_cards": ["As", "Ah"],
            "public_cards": ["2c", "3d", "4h"],
            "round": "flop",
            "pot": 6,
            "contributions": {"0": 3, "1": 3},
            "betting_history": history,
            "legal_actions": ["check", "bet"],
        },
        legal_actions=("check", "bet"),
        action="check",
        opponent_private_card="Ac Ad",
        opponent_rank=label,
        terminal_return=1,
    )


class ProbeCounterfactualTest(unittest.TestCase):
    """Test order-counterfactual pair construction and reporting."""

    def test_composition_key_ignores_order_but_order_key_keeps_it(self) -> None:
        left = _point(0, ["bet", "call", "check"])
        right = _point(1, ["call", "bet", "check"])

        self.assertEqual(composition_key(left), composition_key(right))
        self.assertNotEqual(order_key(left), order_key(right))

    def test_build_counterfactual_pairs_requires_posterior_difference(self) -> None:
        points = [
            _point(0, ["bet", "call", "check"], "AA"),
            _point(1, ["bet", "call", "check"], "AA"),
            _point(2, ["call", "bet", "check"], "KK"),
            _point(3, ["call", "bet", "check"], "KK"),
        ]
        labels = [0, 0, 1, 1]
        groups = build_order_groups(points, labels, class_count=2, min_group_size=2)
        pairs = build_counterfactual_pairs(groups, min_posterior_js=0.01, max_pairs=None)

        self.assertEqual(len(groups), 2)
        self.assertEqual(len(pairs), 1)
        self.assertGreater(pairs[0].posterior_js, 0.0)

    def test_jensen_shannon_zero_for_identical_distributions(self) -> None:
        self.assertEqual(jensen_shannon((0.5, 0.5), (0.5, 0.5)), 0.0)

    def test_composition_coverage_reports_order_varying_support(self) -> None:
        points = [
            _point(0, ["bet", "call", "check"], "AA"),
            _point(1, ["call", "bet", "check"], "KK"),
        ]

        rows = composition_coverage(points)

        strict = rows[0]
        self.assertEqual(strict["profile"], "strict")
        self.assertEqual(strict["order_varying_bucket_count"], 1)
        self.assertEqual(strict["points_in_order_varying_buckets"], 2)

    def test_truncate_point_history_removes_future_action_events(self) -> None:
        point = _point(0, ["bet", "call", "check"], "AA")

        truncated = truncate_point_history(point, 1)

        actions = [
            event["action"]
            for event in truncated.visible_state["betting_history"]
            if event["type"] == "action"
        ]
        self.assertEqual(actions, ["bet"])
        self.assertEqual(order_key(truncated), "bet")

    def test_report_labels_insufficient_pairs_without_model_forward(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_config = root / "train.json"
            checkpoint = root / "model.pt"
            train_config.write_text("{}", encoding="utf-8")
            checkpoint.write_text("not a torch checkpoint", encoding="utf-8")
            output_dir = root / "out"
            points = [
                _point(0, ["bet", "call", "check"], "AA"),
                _point(1, ["bet", "call", "check"], "AA"),
            ]

            with patch("jorjor.probe.counterfactual.load_limit_ar_train_config", return_value=_TrainConfig()):
                with patch("jorjor.probe.counterfactual._load_points", return_value=points):
                    payload = build_order_counterfactual_report(
                        train_config_path=train_config,
                        checkpoint_path=checkpoint,
                        output_dir=output_dir,
                        layers=("sequence_hidden",),
                        min_group_size=2,
                        min_pairs=2,
                        device="cpu",
                    )
            saved = json.loads((output_dir / "order_counterfactual_report.json").read_text(encoding="utf-8"))
            self.assertTrue((output_dir / "order_counterfactual_report.md").exists())

        self.assertEqual(payload["final_diagnostic_label"], INSUFFICIENT_PAIRS)
        self.assertEqual(saved["final_diagnostic_label"], INSUFFICIENT_PAIRS)
        self.assertIn("coverage_profiles", saved)

    def test_prefix_composition_baseline_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_config = root / "train.json"
            train_config.write_text("{}", encoding="utf-8")
            output_dir = root / "out"
            points = [
                _point(0, [], "AA"),
                _point(1, ["bet"], "AA"),
                _point(2, [], "KK"),
                _point(3, ["check"], "KK"),
            ]

            with patch("jorjor.probe.counterfactual.load_limit_ar_train_config", return_value=_TrainConfig()):
                with patch("jorjor.probe.counterfactual._load_points", return_value=points):
                    payload = build_prefix_composition_baseline_report(
                        train_config_path=train_config,
                        output_dir=output_dir,
                        max_points=None,
                        min_non_final_transitions=1,
                    )
            report_exists = (output_dir / "prefix_composition_baseline_report.json").exists()

        self.assertIn("summary", payload)
        self.assertTrue(report_exists)


class _TrainConfig:
    run_version = "test"
    data_source = "teacher"
    seed = 1
    data_seed = 1
    split_seed = 1
    hand_count = 1
    artifact_dir = Path("outputs/test")
    train_fraction = 0.0
    validation_fraction = 0.0
    test_fraction = 1.0
    epochs = 1
    learning_rate = 0.001
    hidden_size = 8
    num_heads = 2
    num_layers = 1
    input_control = "none"
    range_label_control = "none"
    range_label_seed = 0
    policy_loss_weight = 1.0
    value_loss_weight = 0.0
    opponent_hand_loss_weight = 0.0
    abstract_cfr_iteration_count = 1
    abstract_cfr_deal_sample_count = 1
    abstract_cfr_policy_path = None
    batch_size = 2
    device = "cpu"
    save_model = False


if __name__ == "__main__":
    unittest.main()
