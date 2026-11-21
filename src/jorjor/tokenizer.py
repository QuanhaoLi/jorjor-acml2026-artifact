from __future__ import annotations

from dataclasses import dataclass

from .data.schema import DecisionPoint
from .envs.limit_holdem import DECK as LIMIT_DECK
from .envs.limit_holdem import HAND_169_LABELS
from .envs.leduc import DECK


RANKS = ("J", "Q", "K")
ACTIONS = ("bet", "call", "check", "fold")
ROUNDS = ("preflop", "public")
MAX_HISTORY_ACTIONS = 4
LIMIT_ROUNDS = ("preflop", "flop", "turn", "river")
MAX_LIMIT_HISTORY_ACTIONS = 16


@dataclass(frozen=True)
class LeducFeatures:
    """Tokenized Leduc decision point."""

    input_ids: tuple[int, ...]
    policy_label: int
    value_label: float
    opponent_card_label: int


class LeducTokenizer:
    """Small deterministic tokenizer for Leduc visible states."""

    def __init__(self) -> None:
        tokens = [
            "player_0",
            "player_1",
            "public_none",
            "private_mask",
            *[f"private_{card}" for card in DECK],
            *[f"public_{card}" for card in DECK],
            *[f"round_{name}" for name in ROUNDS],
            *[f"hist_{action}" for action in ACTIONS],
            *[f"hist_{index}_{action}" for index in range(MAX_HISTORY_ACTIONS) for action in ACTIONS],
            *[f"legal_{action}" for action in ACTIONS],
        ]
        self.token_to_id = {token: index for index, token in enumerate(tokens)}
        self.card_to_id = {card: index for index, card in enumerate(DECK)}
        self.action_to_id = {action: index for index, action in enumerate(ACTIONS)}

    @property
    def vocab_size(self) -> int:
        """Return the visible-token vocabulary size."""

        return len(self.token_to_id)

    @property
    def action_size(self) -> int:
        """Return the policy action vocabulary size."""

        return len(self.action_to_id)

    @property
    def opponent_card_size(self) -> int:
        """Return the opponent-card class count."""

        return len(self.card_to_id)

    def encode(self, point: DecisionPoint) -> LeducFeatures:
        """Encode one decision point into model-ready ids and labels."""

        state = point.visible_state
        input_ids = [
            self.token_to_id[f"player_{state['acting_player']}"],
            self.token_to_id[f"private_{state['private_card']}"],
            self.token_to_id[f"round_{state['round']}"],
        ]

        public_card = state.get("public_card")
        if public_card is None:
            input_ids.append(self.token_to_id["public_none"])
        else:
            input_ids.append(self.token_to_id[f"public_{public_card}"])

        history_actions = [event["action"] for event in state["betting_history"] if event["type"] == "action"]
        for action in ACTIONS:
            if action in history_actions:
                input_ids.append(self.token_to_id[f"hist_{action}"])
            if action in state["legal_actions"]:
                input_ids.append(self.token_to_id[f"legal_{action}"])
        for index, action in enumerate(history_actions[:MAX_HISTORY_ACTIONS]):
            input_ids.append(self.token_to_id[f"hist_{index}_{action}"])

        return LeducFeatures(
            input_ids=tuple(input_ids),
            policy_label=self.action_to_id[point.action],
            value_label=float(point.terminal_return or 0),
            opponent_card_label=self.card_to_id[point.opponent_private_card],
        )


class LimitTokenizer:
    """Deterministic tokenizer for Limit Hold'em visible states."""

    def __init__(self) -> None:
        tokens = [
            "player_0",
            "player_1",
            "private_mask",
            *[f"private_{card}" for card in LIMIT_DECK],
            *[f"public_{card}" for card in LIMIT_DECK],
            *[f"round_{name}" for name in LIMIT_ROUNDS],
            *[f"hist_{action}" for action in ACTIONS],
            *[f"hist_{index}_{action}" for index in range(MAX_LIMIT_HISTORY_ACTIONS) for action in ACTIONS],
            *[f"legal_{action}" for action in ACTIONS],
            "player_mask",
            "round_mask",
        ]
        self.token_to_id = {token: index for index, token in enumerate(tokens)}
        self.action_to_id = {action: index for index, action in enumerate(ACTIONS)}
        self.opponent_hand_to_id = {label: index for index, label in enumerate(HAND_169_LABELS)}

    @property
    def vocab_size(self) -> int:
        """Return the visible-token vocabulary size."""

        return len(self.token_to_id)

    @property
    def action_size(self) -> int:
        """Return the policy action vocabulary size."""

        return len(self.action_to_id)

    @property
    def opponent_card_size(self) -> int:
        """Return the opponent-hand class count."""

        return len(self.opponent_hand_to_id)

    def encode(self, point: DecisionPoint) -> LeducFeatures:
        """Encode one Limit decision point into model-ready ids and labels."""

        state = point.visible_state
        acting_player = state["acting_player"]
        if acting_player == "mask":
            player_token = "player_mask"
        else:
            player_token = f"player_{acting_player}"

        round_name = state["round"]
        if round_name == "mask":
            round_token = "round_mask"
        else:
            round_token = f"round_{round_name}"

        input_ids = [
            self.token_to_id[player_token],
            self.token_to_id[round_token],
        ]

        for card in state["private_cards"]:
            if card == "mask":
                input_ids.append(self.token_to_id["private_mask"])
            else:
                input_ids.append(self.token_to_id[f"private_{card}"])
        for card in state["public_cards"]:
            input_ids.append(self.token_to_id[f"public_{card}"])

        history_actions = [event["action"] for event in state["betting_history"] if event["type"] == "action"]
        bag_actions = [event["action"] for event in state["betting_history"] if event["type"] == "bag_action"]
        for action in ACTIONS:
            if action in history_actions:
                input_ids.append(self.token_to_id[f"hist_{action}"])
            if action in state["legal_actions"]:
                input_ids.append(self.token_to_id[f"legal_{action}"])
        for index, action in enumerate(history_actions[:MAX_LIMIT_HISTORY_ACTIONS]):
            input_ids.append(self.token_to_id[f"hist_{index}_{action}"])
        for action in ACTIONS:
            for _ in range(bag_actions.count(action)):
                input_ids.append(self.token_to_id[f"hist_{action}"])

        return LeducFeatures(
            input_ids=tuple(input_ids),
            policy_label=self.action_to_id[point.action],
            value_label=float(point.terminal_return or 0),
            opponent_card_label=self.opponent_hand_to_id[point.opponent_rank],
        )
