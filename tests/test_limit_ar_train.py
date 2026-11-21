from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
import unittest

from jorjor.config import load_limit_ar_train_config
from jorjor.data.schema import DecisionPoint
from jorjor.envs.limit_holdem import LimitHoldemRandomGenerator
from jorjor.limit_abstract_cfr_teacher import run_limit_abstract_cfr_teacher_smoke
from jorjor.limit_random import flatten_decision_points
from jorjor.models.ar import build_tiny_ar_model, require_torch
from jorjor.tokenizer import LimitTokenizer
from jorjor.train.limit_controls import controlled_limit_points, controlled_range_labels
from jorjor.train.limit_ar import run_limit_ar_train


class LimitArTrainTest(unittest.TestCase):
    """Test v0.4.6 tiny AR Limit training smoke."""

    def test_tiny_ar_model_forward_returns_all_heads(self) -> None:
        torch, _ = require_torch()
        model = build_tiny_ar_model(
            vocab_size=32,
            action_size=4,
            opponent_card_size=169,
            hidden_size=16,
            num_heads=4,
            num_layers=1,
        )
        outputs = model(torch.tensor([[1, 2, 3, -1], [4, 5, -1, -1]], dtype=torch.long))

        self.assertEqual(outputs["policy_logits"].shape, (2, 4))
        self.assertEqual(outputs["value"].shape, (2,))
        self.assertEqual(outputs["opponent_card_logits"].shape, (2, 169))

    def test_load_limit_ar_train_config(self) -> None:
        config = load_limit_ar_train_config(Path("configs/train/limit_ar_teacher_smoke.json"))

        self.assertEqual(config.run_version, "0.4.6")
        self.assertEqual(config.data_source, "teacher")
        self.assertEqual(config.num_heads, 4)
        self.assertEqual(config.num_layers, 1)
        self.assertEqual(config.input_control, "none")
        self.assertEqual(config.range_label_control, "none")
        self.assertEqual(config.range_label_seed, 0)
        self.assertEqual(config.policy_loss_weight, 1.0)
        self.assertEqual(config.value_loss_weight, 0.3)
        self.assertEqual(config.opponent_hand_loss_weight, 0.7)

    def test_load_limit_ar_action_value_only_smoke_config(self) -> None:
        config = load_limit_ar_train_config(Path("configs/train/limit_ar_action_value_only_smoke.json"))

        self.assertEqual(config.run_version, "1.0.2-smoke")
        self.assertEqual(config.policy_loss_weight, 1.0)
        self.assertEqual(config.value_loss_weight, 0.3)
        self.assertEqual(config.opponent_hand_loss_weight, 0.0)
        self.assertEqual(config.device, "cpu")

    def test_load_limit_ar_combined_control_config(self) -> None:
        config = load_limit_ar_train_config(Path("configs/train/limit_ar_teacher_combined_mask_smoke.json"))

        self.assertEqual(config.run_version, "0.4.9")
        self.assertEqual(config.input_control, "drop_history_mask_cards")

    def test_load_limit_ar_shuffle_label_control_config(self) -> None:
        config = load_limit_ar_train_config(Path("configs/train/limit_ar_teacher_shuffle_labels_smoke.json"))

        self.assertEqual(config.run_version, "0.4.14-smoke")
        self.assertEqual(config.range_label_control, "shuffle_within_split")
        self.assertEqual(config.range_label_seed, 211)

    def test_load_limit_ar_abstract_cfr_teacher_config(self) -> None:
        config = load_limit_ar_train_config(Path("configs/train/limit_ar_abstract_cfr_teacher_smoke.json"))

        self.assertEqual(config.run_version, "0.6.3-smoke")
        self.assertEqual(config.data_source, "abstract_cfr_teacher")
        self.assertEqual(config.abstract_cfr_iteration_count, 10)
        self.assertEqual(config.abstract_cfr_deal_sample_count, 20)
        self.assertIsNone(config.abstract_cfr_policy_path)

    def test_load_limit_ar_fixed_abstract_cfr_policy_config(self) -> None:
        config = load_limit_ar_train_config(Path("configs/train/limit_ar_abstract_cfr_fixed_policy_smoke.json"))

        self.assertEqual(config.run_version, "0.6.4-smoke")
        self.assertEqual(config.data_source, "abstract_cfr_teacher")
        self.assertEqual(
            config.abstract_cfr_policy_path,
            Path("outputs/limit_abstract_cfr_teacher_smoke/average_policy.json"),
        )

    def test_combined_control_drops_history_and_masks_visible_cards(self) -> None:
        point = flatten_decision_points(LimitHoldemRandomGenerator(seed=31).generate_hands(1))[0]
        controlled = controlled_limit_points((point,), "drop_history_mask_cards")[0]

        self.assertEqual(controlled.visible_state["betting_history"], [])
        self.assertEqual(controlled.visible_state["public_cards"], [])
        self.assertEqual(controlled.visible_state["private_cards"], ["mask"])
        self.assertEqual(controlled.action, point.action)
        self.assertEqual(controlled.opponent_private_card, point.opponent_private_card)
        self.assertGreater(len(LimitTokenizer().encode(controlled).input_ids), 0)

    def test_strict_controls_mask_expected_visible_fields_only(self) -> None:
        point = flatten_decision_points(LimitHoldemRandomGenerator(seed=41).generate_hands(1))[0]
        tokenizer = LimitTokenizer()

        legal_mask = controlled_limit_points((point,), "mask_legal_actions")[0]
        self.assertEqual(legal_mask.visible_state["legal_actions"], [])
        self.assertEqual(legal_mask.action, point.action)
        self.assertEqual(legal_mask.opponent_rank, point.opponent_rank)
        self.assertGreater(len(tokenizer.encode(legal_mask).input_ids), 0)

        round_position_mask = controlled_limit_points((point,), "mask_round_position")[0]
        self.assertEqual(round_position_mask.visible_state["acting_player"], "mask")
        self.assertEqual(round_position_mask.visible_state["round"], "mask")
        self.assertEqual(round_position_mask.action, point.action)
        self.assertEqual(round_position_mask.opponent_rank, point.opponent_rank)
        self.assertGreater(len(tokenizer.encode(round_position_mask).input_ids), 0)

        card_mask = controlled_limit_points((point,), "mask_all_cards")[0]
        self.assertEqual(card_mask.visible_state["public_cards"], [])
        self.assertEqual(card_mask.visible_state["private_cards"], ["mask"])
        self.assertEqual(card_mask.action, point.action)
        self.assertEqual(card_mask.opponent_rank, point.opponent_rank)
        self.assertGreater(len(tokenizer.encode(card_mask).input_ids), 0)

    def test_compound_prior_controls_keep_only_expected_fields(self) -> None:
        point = flatten_decision_points(LimitHoldemRandomGenerator(seed=43).generate_hands(2))[1]
        tokenizer = LimitTokenizer()

        history_only = controlled_limit_points((point,), "history_only")[0]
        self.assertEqual(history_only.visible_state["acting_player"], "mask")
        self.assertEqual(history_only.visible_state["round"], "mask")
        self.assertEqual(history_only.visible_state["public_cards"], [])
        self.assertEqual(history_only.visible_state["private_cards"], ["mask"])
        self.assertEqual(history_only.visible_state["legal_actions"], [])
        self.assertEqual(history_only.visible_state["betting_history"], point.visible_state["betting_history"])
        self.assertGreater(len(tokenizer.encode(history_only).input_ids), 0)

        legal_only = controlled_limit_points((point,), "legal_only")[0]
        self.assertEqual(legal_only.visible_state["acting_player"], "mask")
        self.assertEqual(legal_only.visible_state["round"], "mask")
        self.assertEqual(legal_only.visible_state["public_cards"], [])
        self.assertEqual(legal_only.visible_state["private_cards"], ["mask"])
        self.assertEqual(legal_only.visible_state["betting_history"], [])
        self.assertEqual(legal_only.visible_state["legal_actions"], point.visible_state["legal_actions"])
        self.assertGreater(len(tokenizer.encode(legal_only).input_ids), 0)

        round_position_only = controlled_limit_points((point,), "round_position_only")[0]
        self.assertEqual(round_position_only.visible_state["acting_player"], point.visible_state["acting_player"])
        self.assertEqual(round_position_only.visible_state["round"], point.visible_state["round"])
        self.assertEqual(round_position_only.visible_state["public_cards"], [])
        self.assertEqual(round_position_only.visible_state["private_cards"], ["mask"])
        self.assertEqual(round_position_only.visible_state["betting_history"], [])
        self.assertEqual(round_position_only.visible_state["legal_actions"], [])
        self.assertGreater(len(tokenizer.encode(round_position_only).input_ids), 0)

        no_observation = controlled_limit_points((point,), "no_observation")[0]
        self.assertEqual(no_observation.visible_state["acting_player"], "mask")
        self.assertEqual(no_observation.visible_state["round"], "mask")
        self.assertEqual(no_observation.visible_state["public_cards"], [])
        self.assertEqual(no_observation.visible_state["private_cards"], ["mask"])
        self.assertEqual(no_observation.visible_state["betting_history"], [])
        self.assertEqual(no_observation.visible_state["legal_actions"], [])
        self.assertEqual(no_observation.action, point.action)
        self.assertEqual(no_observation.opponent_rank, point.opponent_rank)
        self.assertGreater(len(tokenizer.encode(no_observation).input_ids), 0)

    def test_history_destruction_controls_preserve_labels(self) -> None:
        point = DecisionPoint(
            hand_id="history-control",
            step_index=3,
            acting_player=0,
            visible_state={
                "game": "limit_holdem_hu",
                "hand_id": "history-control",
                "acting_player": 0,
                "private_cards": ["As", "Kd"],
                "public_cards": ["2c", "7h", "Jd"],
                "round": "flop",
                "pot": 4,
                "contributions": {"0": 2, "1": 2},
                "betting_history": [
                    {"type": "action", "round": "preflop", "player": 0, "action": "bet"},
                    {"type": "action", "round": "preflop", "player": 1, "action": "call"},
                    {"type": "action", "round": "flop", "player": 0, "action": "check"},
                ],
                "legal_actions": ["check", "bet"],
            },
            legal_actions=("check", "bet"),
            action="bet",
            opponent_private_card="Qh Jc",
            opponent_rank="QJo",
            terminal_return=1,
        )
        tokenizer = LimitTokenizer()

        shuffled = controlled_limit_points((point,), "shuffle_history_order")[0]
        self.assertEqual(
            sorted(event["action"] for event in shuffled.visible_state["betting_history"]),
            sorted(event["action"] for event in point.visible_state["betting_history"]),
        )
        self.assertEqual(shuffled.action, point.action)
        self.assertEqual(shuffled.opponent_rank, point.opponent_rank)
        self.assertGreater(len(tokenizer.encode(shuffled).input_ids), 0)

        bag = controlled_limit_points((point,), "bag_of_actions")[0]
        self.assertTrue(all(event["type"] == "bag_action" for event in bag.visible_state["betting_history"]))
        self.assertEqual(
            sorted(event["action"] for event in bag.visible_state["betting_history"]),
            sorted(event["action"] for event in point.visible_state["betting_history"]),
        )
        encoded_bag = tokenizer.encode(bag).input_ids
        indexed_history_token_ids = {
            token_id for token, token_id in tokenizer.token_to_id.items() if token.startswith("hist_0_")
        }
        self.assertTrue(indexed_history_token_ids.isdisjoint(encoded_bag))
        self.assertEqual(bag.action, point.action)
        self.assertEqual(bag.opponent_rank, point.opponent_rank)

        prior = controlled_limit_points((point,), "round_position_action_prior")[0]
        self.assertEqual(prior.visible_state["betting_history"], [])
        self.assertEqual(prior.visible_state["public_cards"], [])
        self.assertEqual(prior.visible_state["private_cards"], ["mask"])
        self.assertEqual(prior.visible_state["acting_player"], point.visible_state["acting_player"])
        self.assertEqual(prior.visible_state["round"], point.visible_state["round"])
        self.assertEqual(prior.visible_state["legal_actions"], point.visible_state["legal_actions"])

    def test_limit_history_composition_controls(self) -> None:
        point = DecisionPoint(
            hand_id="limit-history-composition",
            step_index=2,
            acting_player=0,
            visible_state={
                "game": "limit_holdem_hu",
                "hand_id": "limit-history-composition",
                "acting_player": 0,
                "private_cards": ["As", "Kd"],
                "public_cards": ["2c", "7d", "9h"],
                "round": "flop",
                "pot": 4,
                "contributions": {"0": 2, "1": 2},
                "betting_history": [
                    {"type": "action", "round": "preflop", "player": 0, "action": "bet"},
                    {"type": "action", "round": "preflop", "player": 1, "action": "call"},
                    {"type": "action", "round": "flop", "player": 0, "action": "bet"},
                    {"type": "action", "round": "flop", "player": 1, "action": "fold"},
                ],
                "legal_actions": ["check", "bet"],
            },
            legal_actions=("check", "bet"),
            action="bet",
            opponent_private_card="Qh Jc",
            opponent_rank="QJo",
            terminal_return=1,
        )
        original_actions = [event["action"] for event in point.visible_state["betting_history"]]

        for control in (
            "shuffle_history_order_seed_1",
            "shuffle_history_order_seed_2",
            "shuffle_history_order_seed_3",
            "reverse_history_order",
            "sort_history_actions",
        ):
            controlled = controlled_limit_points((point,), control)[0]
            controlled_actions = [event["action"] for event in controlled.visible_state["betting_history"]]
            self.assertEqual(sorted(controlled_actions), sorted(original_actions))
            self.assertEqual(controlled.action, point.action)
            self.assertEqual(controlled.opponent_rank, point.opponent_rank)

        reversed_point = controlled_limit_points((point,), "reverse_history_order")[0]
        self.assertEqual(
            [event["action"] for event in reversed_point.visible_state["betting_history"]],
            list(reversed(original_actions)),
        )

        sorted_point = controlled_limit_points((point,), "sort_history_actions")[0]
        self.assertEqual(
            [event["action"] for event in sorted_point.visible_state["betting_history"]],
            ["bet", "bet", "call", "fold"],
        )

        presence = controlled_limit_points((point,), "history_presence_only")[0]
        self.assertTrue(all(event["type"] == "bag_action" for event in presence.visible_state["betting_history"]))
        self.assertEqual(
            [event["action"] for event in presence.visible_state["betting_history"]],
            ["bet", "call", "fold"],
        )

        last_action = controlled_limit_points((point,), "history_last_action_only")[0]
        self.assertEqual(last_action.visible_state["betting_history"], [point.visible_state["betting_history"][-1]])

    def test_shuffle_range_labels_is_deterministic_and_keeps_inputs(self) -> None:
        points = flatten_decision_points(LimitHoldemRandomGenerator(seed=37).generate_hands(20))
        shuffled_a = controlled_range_labels(points, "shuffle_within_split", seed=211)
        shuffled_b = controlled_range_labels(points, "shuffle_within_split", seed=211)

        self.assertEqual([point.opponent_rank for point in shuffled_a], [point.opponent_rank for point in shuffled_b])
        self.assertEqual(sorted(point.opponent_rank for point in points), sorted(point.opponent_rank for point in shuffled_a))
        self.assertNotEqual([point.opponent_rank for point in points], [point.opponent_rank for point in shuffled_a])
        self.assertEqual([point.visible_state for point in points], [point.visible_state for point in shuffled_a])
        self.assertEqual([point.action for point in points], [point.action for point in shuffled_a])
        self.assertEqual([point.opponent_private_card for point in points], [point.opponent_private_card for point in shuffled_a])

    def test_load_limit_ar_10k_config(self) -> None:
        config = load_limit_ar_train_config(Path("configs/train/limit_ar_teacher_10k.json"))

        self.assertEqual(config.run_version, "0.4.7")
        self.assertEqual(config.data_source, "teacher")
        self.assertEqual(config.hand_count, 10000)
        self.assertEqual(config.hidden_size, 64)
        self.assertEqual(config.num_heads, 4)

    def test_load_limit_ar_4layer_config(self) -> None:
        config = load_limit_ar_train_config(Path("configs/train/limit_ar_teacher_seed_a_10k_4layer.json"))

        self.assertEqual(config.run_version, "0.8.0")
        self.assertEqual(config.data_source, "teacher")
        self.assertEqual(config.hand_count, 10000)
        self.assertEqual(config.hidden_size, 96)
        self.assertEqual(config.num_heads, 4)
        self.assertEqual(config.num_layers, 4)
        self.assertEqual(config.input_control, "none")
        self.assertTrue(config.save_model)

    def test_load_limit_ar_4layer_seed_replication_configs(self) -> None:
        configs = {
            "configs/train/limit_ar_teacher_seed_b_10k_4layer.json": (103, 71, 89),
            "configs/train/limit_ar_teacher_seed_c_10k_4layer.json": (107, 79, 97),
        }

        for path, seeds in configs.items():
            with self.subTest(path=path):
                config = load_limit_ar_train_config(Path(path))
                seed, data_seed, split_seed = seeds
                self.assertEqual(config.run_version, "0.8.1")
                self.assertEqual(config.seed, seed)
                self.assertEqual(config.data_seed, data_seed)
                self.assertEqual(config.split_seed, split_seed)
                self.assertEqual(config.hand_count, 10000)
                self.assertEqual(config.hidden_size, 96)
                self.assertEqual(config.num_layers, 4)
                self.assertEqual(config.input_control, "none")

    def test_load_limit_ar_50k_data_scale_config(self) -> None:
        config = load_limit_ar_train_config(Path("configs/train/limit_ar_teacher_seed_d_50k_4layer.json"))

        self.assertEqual(config.run_version, "0.9.0")
        self.assertEqual(config.data_source, "teacher")
        self.assertEqual(config.seed, 109)
        self.assertEqual(config.data_seed, 113)
        self.assertEqual(config.split_seed, 127)
        self.assertEqual(config.hand_count, 50000)
        self.assertEqual(config.hidden_size, 96)
        self.assertEqual(config.num_heads, 4)
        self.assertEqual(config.num_layers, 4)
        self.assertEqual(config.input_control, "none")
        self.assertEqual(config.range_label_control, "none")
        self.assertTrue(config.save_model)

    def test_load_limit_ar_8layer_capacity_config(self) -> None:
        config = load_limit_ar_train_config(Path("configs/train/limit_ar_teacher_seed_e_10k_8layer.json"))

        self.assertEqual(config.run_version, "0.9.1")
        self.assertEqual(config.data_source, "teacher")
        self.assertEqual(config.seed, 127)
        self.assertEqual(config.data_seed, 131)
        self.assertEqual(config.split_seed, 137)
        self.assertEqual(config.hand_count, 10000)
        self.assertEqual(config.hidden_size, 96)
        self.assertEqual(config.num_heads, 4)
        self.assertEqual(config.num_layers, 8)
        self.assertEqual(config.input_control, "none")
        self.assertEqual(config.range_label_control, "none")
        self.assertTrue(config.save_model)

    def test_load_limit_ar_stabilized_8layer_config(self) -> None:
        original = load_limit_ar_train_config(Path("configs/train/limit_ar_teacher_seed_e_10k_8layer.json"))
        stable = load_limit_ar_train_config(Path("configs/train/limit_ar_teacher_seed_e_10k_8layer_stable.json"))

        self.assertEqual(stable.run_version, "0.9.2")
        self.assertEqual(stable.seed, original.seed)
        self.assertEqual(stable.data_seed, original.data_seed)
        self.assertEqual(stable.split_seed, original.split_seed)
        self.assertEqual(stable.hand_count, original.hand_count)
        self.assertEqual(stable.hidden_size, original.hidden_size)
        self.assertEqual(stable.num_heads, original.num_heads)
        self.assertEqual(stable.num_layers, original.num_layers)
        self.assertEqual(stable.learning_rate, 0.001)
        self.assertEqual(stable.epochs, 16)
        self.assertNotEqual(stable.artifact_dir, original.artifact_dir)

    def test_load_limit_ar_50k_scale_replication_configs(self) -> None:
        expected = {
            "seed_f": (139, 149, 157),
            "seed_g": (151, 163, 167),
        }

        for seed_name, seeds in expected.items():
            with self.subTest(seed_name=seed_name):
                config = load_limit_ar_train_config(
                    Path(f"configs/train/limit_ar_teacher_{seed_name}_50k_4layer.json")
                )
                seed, data_seed, split_seed = seeds
                self.assertEqual(config.run_version, "0.9.4")
                self.assertEqual(config.seed, seed)
                self.assertEqual(config.data_seed, data_seed)
                self.assertEqual(config.split_seed, split_seed)
                self.assertEqual(config.hand_count, 50000)
                self.assertEqual(config.hidden_size, 96)
                self.assertEqual(config.num_heads, 4)
                self.assertEqual(config.num_layers, 4)
                self.assertEqual(config.epochs, 8)
                self.assertEqual(config.learning_rate, 0.002)

    def test_load_limit_ar_action_value_only_10k_configs(self) -> None:
        expected = {
            "seed_h": (179, 181, 191),
            "seed_i": (193, 197, 199),
            "seed_j": (207, 211, 223),
        }

        for seed_name, seeds in expected.items():
            with self.subTest(seed_name=seed_name):
                config = load_limit_ar_train_config(
                    Path(f"configs/train/limit_ar_action_value_only_{seed_name}_10k_4layer.json")
                )
                seed, data_seed, split_seed = seeds
                self.assertEqual(config.run_version, "1.0.2")
                self.assertEqual(config.seed, seed)
                self.assertEqual(config.data_seed, data_seed)
                self.assertEqual(config.split_seed, split_seed)
                self.assertEqual(config.hand_count, 10000)
                self.assertEqual(config.hidden_size, 96)
                self.assertEqual(config.num_heads, 4)
                self.assertEqual(config.num_layers, 4)
                self.assertEqual(config.policy_loss_weight, 1.0)
                self.assertEqual(config.value_loss_weight, 0.3)
                self.assertEqual(config.opponent_hand_loss_weight, 0.0)

    def test_load_limit_ar_action_value_only_50k_locked_layer_configs(self) -> None:
        expected = {
            "seed_k": (271, 277, 281),
            "seed_l": (283, 293, 307),
            "seed_m": (311, 313, 317),
        }

        for seed_name, seeds in expected.items():
            with self.subTest(seed_name=seed_name):
                config = load_limit_ar_train_config(
                    Path(f"configs/train/limit_ar_action_value_only_{seed_name}_50k_4layer.json")
                )
                seed, data_seed, split_seed = seeds
                self.assertEqual(config.run_version, "1.0.9")
                self.assertEqual(config.seed, seed)
                self.assertEqual(config.data_seed, data_seed)
                self.assertEqual(config.split_seed, split_seed)
                self.assertEqual(config.hand_count, 50000)
                self.assertEqual(config.hidden_size, 96)
                self.assertEqual(config.num_heads, 4)
                self.assertEqual(config.num_layers, 4)
                self.assertEqual(config.policy_loss_weight, 1.0)
                self.assertEqual(config.value_loss_weight, 0.3)
                self.assertEqual(config.opponent_hand_loss_weight, 0.0)

    def test_load_limit_ar_10k_control_configs(self) -> None:
        configs = {
            "configs/train/limit_ar_teacher_drop_history_10k.json": "drop_history",
            "configs/train/limit_ar_teacher_mask_public_10k.json": "mask_public_card",
            "configs/train/limit_ar_teacher_mask_private_10k.json": "mask_private_card",
            "configs/train/limit_ar_teacher_combined_mask_10k.json": "drop_history_mask_cards",
        }

        for path, input_control in configs.items():
            with self.subTest(path=path):
                config = load_limit_ar_train_config(Path(path))
                self.assertEqual(config.hand_count, 10000)
                self.assertEqual(config.input_control, input_control)
                self.assertEqual(config.split_seed, 83)

    def test_run_limit_ar_train_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_limit_ar_train(
                config_path=Path("configs/train/limit_ar_teacher_smoke.json"),
                artifact_dir=Path(temp_dir),
            )
            metrics = json.loads((result.artifact_dir / "metrics.json").read_text(encoding="utf-8"))

        self.assertGreater(result.train_points, 0)
        self.assertTrue(math.isfinite(result.first_loss))
        self.assertTrue(math.isfinite(result.last_loss))
        self.assertGreaterEqual(result.opponent_hand_top5_accuracy, result.opponent_hand_accuracy)
        self.assertGreaterEqual(result.opponent_hand_top10_accuracy, result.opponent_hand_top5_accuracy)
        self.assertEqual(metrics["version"], "0.4.6")
        self.assertEqual(metrics["model"], "tiny_ar")
        self.assertEqual(metrics["input_control"], "none")
        self.assertEqual(metrics["policy_loss_weight"], 1.0)
        self.assertEqual(metrics["value_loss_weight"], 0.3)
        self.assertEqual(metrics["opponent_hand_loss_weight"], 0.7)
        self.assertTrue(metrics["opponent_hand_supervised"])
        self.assertEqual(metrics["checksum"], result.checksum)

    def test_run_limit_ar_action_value_only_smoke_writes_unsupervised_range_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_limit_ar_train(
                config_path=Path("configs/train/limit_ar_action_value_only_smoke.json"),
                artifact_dir=Path(temp_dir),
            )
            metrics = json.loads((result.artifact_dir / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(metrics["version"], "1.0.2-smoke")
        self.assertEqual(metrics["policy_loss_weight"], 1.0)
        self.assertEqual(metrics["value_loss_weight"], 0.3)
        self.assertEqual(metrics["opponent_hand_loss_weight"], 0.0)
        self.assertFalse(metrics["opponent_hand_supervised"])
        self.assertIn("validation_opponent_hand_loss", metrics)
        self.assertEqual(metrics["checksum"], result.checksum)

    def test_run_limit_ar_combined_control_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_limit_ar_train(
                config_path=Path("configs/train/limit_ar_teacher_combined_mask_smoke.json"),
                artifact_dir=Path(temp_dir),
            )
            metrics = json.loads((result.artifact_dir / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(metrics["version"], "0.4.9")
        self.assertEqual(metrics["input_control"], "drop_history_mask_cards")
        self.assertEqual(metrics["checksum"], result.checksum)

    def test_run_limit_ar_shuffle_label_control_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_limit_ar_train(
                config_path=Path("configs/train/limit_ar_teacher_shuffle_labels_smoke.json"),
                artifact_dir=Path(temp_dir),
            )
            metrics = json.loads((result.artifact_dir / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(metrics["version"], "0.4.14-smoke")
        self.assertEqual(metrics["range_label_control"], "shuffle_within_split")
        self.assertEqual(metrics["range_label_seed"], 211)
        self.assertEqual(metrics["checksum"], result.checksum)

    def test_run_limit_ar_abstract_cfr_teacher_smoke_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_limit_ar_train(
                config_path=Path("configs/train/limit_ar_abstract_cfr_teacher_smoke.json"),
                artifact_dir=Path(temp_dir),
            )
            metrics = json.loads((result.artifact_dir / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(metrics["version"], "0.6.3-smoke")
        self.assertEqual(metrics["data_source"], "abstract_cfr_teacher")
        self.assertEqual(metrics["abstract_cfr_iteration_count"], 10)
        self.assertEqual(metrics["abstract_cfr_deal_sample_count"], 20)
        self.assertEqual(metrics["checksum"], result.checksum)

    def test_run_limit_ar_fixed_abstract_cfr_policy_smoke_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_result = run_limit_abstract_cfr_teacher_smoke(
                config_path=Path("configs/data/limit_abstract_cfr_teacher_smoke.json"),
                artifact_dir=Path(temp_dir) / "policy",
            )
            config = json.loads(Path("configs/train/limit_ar_abstract_cfr_fixed_policy_smoke.json").read_text())
            config["abstract_cfr_policy_path"] = str(policy_result.artifact_dir / "average_policy.json")
            config["artifact_dir"] = str(Path(temp_dir) / "train")
            config_path = Path(temp_dir) / "fixed_policy_config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            result = run_limit_ar_train(config_path=config_path)
            metrics = json.loads((result.artifact_dir / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(metrics["version"], "0.6.4-smoke")
        self.assertEqual(metrics["data_source"], "abstract_cfr_teacher")
        self.assertIsNotNone(metrics["abstract_cfr_policy_path"])
        self.assertEqual(metrics["checksum"], result.checksum)


if __name__ == "__main__":
    unittest.main()
