from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from jorjor.config import load_leduc_tiny_probe_config, load_leduc_tiny_train_config
from jorjor.data.schema import DecisionPoint
from jorjor.envs.leduc import LeducRandomGenerator
from jorjor.leduc_random import flatten_decision_points
from jorjor.tokenizer import LeducTokenizer
from jorjor.train.leduc_tiny import run_leduc_tiny_train


class TokenizerTest(unittest.TestCase):
    """Test v0.1.2 tokenizer and training config plumbing."""

    def test_tokenizer_encodes_visible_state_and_labels(self) -> None:
        point = flatten_decision_points(LeducRandomGenerator(seed=11).generate_hands(1))[0]
        tokenizer = LeducTokenizer()
        features = tokenizer.encode(point)

        self.assertGreater(tokenizer.vocab_size, 0)
        self.assertGreater(len(features.input_ids), 0)
        self.assertGreaterEqual(features.policy_label, 0)
        self.assertGreaterEqual(features.opponent_card_label, 0)
        self.assertEqual(features.value_label, float(point.terminal_return))

    def test_tokenizer_preserves_history_order(self) -> None:
        state_left = {
            "game": "leduc",
            "hand_id": "history-order",
            "acting_player": 0,
            "private_card": "K0",
            "public_card": None,
            "round": "preflop",
            "pot": 4,
            "contributions": {"0": 3, "1": 1},
            "betting_history": [
                {"type": "action", "round": "preflop", "player": 0, "action": "bet"},
                {"type": "action", "round": "preflop", "player": 1, "action": "call"},
            ],
            "legal_actions": ["check", "bet"],
        }
        state_right = {
            **state_left,
            "betting_history": [
                {"type": "action", "round": "preflop", "player": 1, "action": "call"},
                {"type": "action", "round": "preflop", "player": 0, "action": "bet"},
            ],
        }
        point = DecisionPoint(
            hand_id="history-order",
            step_index=0,
            acting_player=0,
            visible_state=state_left,
            legal_actions=("check", "bet"),
            action="bet",
            opponent_private_card="Q0",
            opponent_rank="Q",
            terminal_return=1,
        )

        tokenizer = LeducTokenizer()
        left = tokenizer.encode(point)
        right = tokenizer.encode(DecisionPoint(
            hand_id=point.hand_id,
            step_index=point.step_index,
            acting_player=point.acting_player,
            visible_state=state_right,
            legal_actions=point.legal_actions,
            action=point.action,
            opponent_private_card=point.opponent_private_card,
            opponent_rank=point.opponent_rank,
            terminal_return=point.terminal_return,
        ))

        self.assertNotEqual(left.input_ids, right.input_ids)

    def test_tokenizer_encodes_private_mask(self) -> None:
        state = {
            "game": "leduc",
            "hand_id": "private-mask",
            "acting_player": 0,
            "private_card": "mask",
            "public_card": None,
            "round": "preflop",
            "pot": 2,
            "contributions": {"0": 1, "1": 1},
            "betting_history": [],
            "legal_actions": ["check", "bet"],
        }
        point = DecisionPoint(
            hand_id="private-mask",
            step_index=0,
            acting_player=0,
            visible_state=state,
            legal_actions=("check", "bet"),
            action="check",
            opponent_private_card="Q0",
            opponent_rank="Q",
            terminal_return=0,
        )

        features = LeducTokenizer().encode(point)

        self.assertGreater(len(features.input_ids), 0)

    def test_load_leduc_tiny_train_config(self) -> None:
        config = load_leduc_tiny_train_config(Path("configs/train/leduc_tiny_smoke.json"))

        self.assertEqual(config.run_version, "0.1.2")
        self.assertEqual(config.data_source, "random")
        self.assertEqual(config.seed, 23)
        self.assertEqual(config.data_seed, 11)
        self.assertEqual(config.hand_count, 40)
        self.assertEqual(config.epochs, 4)
        self.assertEqual(config.batch_size, 0)
        self.assertEqual(config.device, "cpu")
        self.assertTrue(config.save_model)
        self.assertEqual(config.artifact_dir, Path("outputs/leduc_tiny_smoke"))

    def test_load_leduc_tiny_teacher_train_config(self) -> None:
        config = load_leduc_tiny_train_config(Path("configs/train/leduc_tiny_teacher_10k.json"))

        self.assertEqual(config.run_version, "0.2.1")
        self.assertEqual(config.data_source, "teacher")
        self.assertEqual(config.seed, 47)
        self.assertEqual(config.data_seed, 107)
        self.assertEqual(config.hand_count, 10000)
        self.assertEqual(config.artifact_dir, Path("outputs/leduc_tiny_teacher_10k"))

    def test_load_leduc_tiny_cfr_teacher_train_config(self) -> None:
        config = load_leduc_tiny_train_config(Path("configs/train/leduc_tiny_cfr_teacher_smoke.json"))

        self.assertEqual(config.run_version, "0.5.1-smoke")
        self.assertEqual(config.data_source, "cfr_teacher")
        self.assertEqual(config.data_seed, 149)
        self.assertEqual(config.hand_count, 120)
        self.assertEqual(config.artifact_dir, Path("outputs/leduc_tiny_cfr_teacher_smoke"))

    def test_load_leduc_tiny_probe_config(self) -> None:
        config = load_leduc_tiny_probe_config(Path("configs/probe/leduc_tiny_teacher_opponent_card.json"))

        self.assertEqual(config.run_version, "0.2.2")
        self.assertEqual(config.target, "opponent_card")
        self.assertEqual(config.input_control, "none")
        self.assertEqual(config.layers, ("embedding_pool", "body_hidden"))
        self.assertEqual(config.train_config, Path("configs/train/leduc_tiny_teacher_10k.json"))
        self.assertTrue(config.random_label_control)

    def test_load_leduc_tiny_value_probe_config(self) -> None:
        config = load_leduc_tiny_probe_config(Path("configs/probe/leduc_tiny_teacher_value.json"))

        self.assertEqual(config.run_version, "0.2.3")
        self.assertEqual(config.target, "value")
        self.assertEqual(config.layers, ("embedding_pool", "body_hidden"))
        self.assertEqual(config.learning_rate, 0.01)

    def test_load_leduc_tiny_shuffled_history_probe_config(self) -> None:
        config = load_leduc_tiny_probe_config(
            Path("configs/probe/leduc_tiny_teacher_opponent_card_shuffled_history.json")
        )

        self.assertEqual(config.run_version, "0.2.4")
        self.assertEqual(config.target, "opponent_card")
        self.assertEqual(config.input_control, "shuffle_history")

    def test_load_leduc_tiny_mask_private_card_probe_config(self) -> None:
        config = load_leduc_tiny_probe_config(
            Path("configs/probe/leduc_tiny_teacher_opponent_card_no_private_card.json")
        )

        self.assertEqual(config.run_version, "0.2.5")
        self.assertEqual(config.target, "opponent_card")
        self.assertEqual(config.input_control, "mask_private_card")

    def test_load_leduc_tiny_combined_mask_probe_config(self) -> None:
        config = load_leduc_tiny_probe_config(
            Path("configs/probe/leduc_tiny_teacher_opponent_card_combined_mask.json")
        )

        self.assertEqual(config.run_version, "0.2.6")
        self.assertEqual(config.target, "opponent_card")
        self.assertEqual(config.input_control, "drop_history_mask_cards")

    def test_run_leduc_tiny_cfr_teacher_smoke_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_leduc_tiny_train(
                config_path=Path("configs/train/leduc_tiny_cfr_teacher_smoke.json"),
                artifact_dir=Path(temp_dir),
            )
            metrics = json.loads((result.artifact_dir / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(metrics["version"], "0.5.1-smoke")
        self.assertEqual(metrics["data_source"], "cfr_teacher")
        self.assertEqual(metrics["checksum"], result.checksum)


if __name__ == "__main__":
    unittest.main()
