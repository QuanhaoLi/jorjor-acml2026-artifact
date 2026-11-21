from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from jorjor.data.schema import DecisionPoint
from jorjor.envs.limit_holdem import LimitHoldemRandomGenerator
from jorjor.limit_loader import run_limit_loader_smoke
from jorjor.limit_random import flatten_decision_points
from jorjor.tokenizer import LimitTokenizer


class LimitTokenizerTest(unittest.TestCase):
    """Test v0.3.2 Limit tokenizer and loader smoke behavior."""

    def test_tokenizer_encodes_visible_state_and_169_label(self) -> None:
        point = flatten_decision_points(LimitHoldemRandomGenerator(seed=31).generate_hands(1))[0]
        tokenizer = LimitTokenizer()
        features = tokenizer.encode(point)

        self.assertGreater(tokenizer.vocab_size, 0)
        self.assertEqual(tokenizer.action_size, 4)
        self.assertEqual(tokenizer.opponent_card_size, 169)
        self.assertGreater(len(features.input_ids), 0)
        self.assertGreaterEqual(features.policy_label, 0)
        self.assertGreaterEqual(features.opponent_card_label, 0)
        self.assertLess(features.opponent_card_label, 169)
        self.assertEqual(features.value_label, float(point.terminal_return))

    def test_tokenizer_preserves_history_order(self) -> None:
        state_left = {
            "game": "limit_holdem_hu",
            "hand_id": "limit-history-order",
            "acting_player": 0,
            "private_cards": ["As", "Kd"],
            "public_cards": ["2c", "7d", "9h"],
            "round": "flop",
            "pot": 4,
            "contributions": {"0": 2, "1": 2},
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
            hand_id="limit-history-order",
            step_index=0,
            acting_player=0,
            visible_state=state_left,
            legal_actions=("check", "bet"),
            action="bet",
            opponent_private_card="Qh Js",
            opponent_rank="QJo",
            terminal_return=1,
        )

        tokenizer = LimitTokenizer()
        left = tokenizer.encode(point)
        right = tokenizer.encode(
            DecisionPoint(
                hand_id=point.hand_id,
                step_index=point.step_index,
                acting_player=point.acting_player,
                visible_state=state_right,
                legal_actions=point.legal_actions,
                action=point.action,
                opponent_private_card=point.opponent_private_card,
                opponent_rank=point.opponent_rank,
                terminal_return=point.terminal_return,
            )
        )

        self.assertNotEqual(left.input_ids, right.input_ids)

    def test_tokenizer_encodes_limit_round_and_position_masks(self) -> None:
        state = {
            "game": "limit_holdem_hu",
            "hand_id": "limit-mask-round-position",
            "acting_player": "mask",
            "private_cards": ["As", "Kd"],
            "public_cards": [],
            "round": "mask",
            "pot": 4,
            "contributions": {"0": 2, "1": 2},
            "betting_history": [],
            "legal_actions": [],
        }
        point = DecisionPoint(
            hand_id="limit-mask-round-position",
            step_index=0,
            acting_player=0,
            visible_state=state,
            legal_actions=("check", "bet"),
            action="bet",
            opponent_private_card="Qh Js",
            opponent_rank="QJo",
            terminal_return=1,
        )

        tokenizer = LimitTokenizer()
        features = tokenizer.encode(point)

        self.assertIn(tokenizer.token_to_id["player_mask"], features.input_ids)
        self.assertIn(tokenizer.token_to_id["round_mask"], features.input_ids)

    def test_run_limit_loader_smoke_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_limit_loader_smoke(
                config_path=Path("configs/data/limit_loader_smoke.json"),
                artifact_dir=Path(temp_dir),
            )
            report_path = result.artifact_dir / "loader_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(result.hand_count, 25)
        self.assertEqual(result.opponent_hand_size, 169)
        self.assertEqual(result.action_size, 4)
        self.assertFalse(result.split_overlap)
        self.assertEqual(result.train_rows + result.validation_rows + result.test_rows, result.decision_point_count)
        self.assertEqual(report["checksum"], result.checksum)


if __name__ == "__main__":
    unittest.main()
