from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Protocol

from jorjor.data.schema import DecisionPoint


RANK_VALUE = {"J": 0, "Q": 1, "K": 2}
DECK = ("J0", "J1", "Q0", "Q1", "K0", "K1")


@dataclass(frozen=True)
class LeducHand:
    """Generated Leduc hand with decision points and terminal returns."""

    hand_id: str
    private_cards: tuple[str, str]
    public_card: str
    returns: tuple[int, int]
    decision_points: tuple[DecisionPoint, ...]


@dataclass(frozen=True)
class LeducPolicyState:
    """Visible state passed to a Leduc action policy."""

    hand_id: str
    acting_player: int
    private_card: str
    public_card: str | None
    round_name: str
    pot: int
    contributions: tuple[int, int]
    betting_history: tuple[dict[str, object], ...]
    legal_actions: tuple[str, ...]

    def to_visible_state(self) -> dict[str, object]:
        """Serialize the policy state into the public decision-point input."""

        return {
            "game": "leduc",
            "hand_id": self.hand_id,
            "acting_player": self.acting_player,
            "private_card": self.private_card,
            "public_card": self.public_card,
            "round": self.round_name,
            "pot": self.pot,
            "contributions": {
                "0": self.contributions[0],
                "1": self.contributions[1],
            },
            "betting_history": list(self.betting_history),
            "legal_actions": list(self.legal_actions),
        }


class LeducActionPolicy(Protocol):
    """Action-selection interface for the local Leduc ruleset."""

    def choose_action(self, state: LeducPolicyState) -> str:
        """Choose one legal action for the current visible state."""


class RandomLeducPolicy:
    """Random legal Leduc policy."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def choose_action(self, state: LeducPolicyState) -> str:
        """Choose a uniformly random legal action."""

        return self._rng.choice(state.legal_actions)


class HeuristicLeducTeacherPolicy:
    """Simple deterministic Leduc teacher for v0.2.0 smoke data."""

    def choose_action(self, state: LeducPolicyState) -> str:
        """Choose a legal heuristic action from visible Leduc information."""

        rank = card_rank(state.private_card)
        rank_value = RANK_VALUE[rank]
        public_rank = card_rank(state.public_card) if state.public_card else None
        has_pair = public_rank == rank

        if "call" in state.legal_actions:
            return "call"

        if "bet" in state.legal_actions:
            if has_pair or rank_value >= RANK_VALUE["Q"]:
                return "bet"
            return "check"

        raise RuntimeError("teacher received unsupported legal-action set")


class LeducRandomGenerator:
    """Generate random legal Leduc hands with a small fixed-limit ruleset."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def generate_hand(self, hand_index: int) -> LeducHand:
        """Generate one complete random legal Leduc hand."""

        policy = RandomLeducPolicy(self._rng)
        return generate_leduc_hand(
            hand_index=hand_index,
            policy_by_player=(policy, policy),
            deck_rng=self._rng,
        )

    def generate_hands(self, hand_count: int) -> tuple[LeducHand, ...]:
        """Generate multiple complete random legal hands."""

        return tuple(self.generate_hand(index) for index in range(hand_count))


class LeducPolicyGenerator:
    """Generate Leduc hands from explicit player policies."""

    def __init__(self, policy_by_player: tuple[LeducActionPolicy, LeducActionPolicy], seed: int = 0) -> None:
        self._policy_by_player = policy_by_player
        self._rng = random.Random(seed)

    def generate_hand(self, hand_index: int) -> LeducHand:
        """Generate one hand using the configured player policies."""

        return generate_leduc_hand(
            hand_index=hand_index,
            policy_by_player=self._policy_by_player,
            deck_rng=self._rng,
        )

    def generate_hands(self, hand_count: int) -> tuple[LeducHand, ...]:
        """Generate multiple complete hands."""

        return tuple(self.generate_hand(index) for index in range(hand_count))


def generate_leduc_hand(
    hand_index: int,
    policy_by_player: tuple[LeducActionPolicy, LeducActionPolicy],
    deck_rng: random.Random | None = None,
) -> LeducHand:
    """Generate one complete legal Leduc hand with explicit policies."""

    deck = list(DECK)
    rng = random.Random(hand_index) if deck_rng is None else deck_rng
    rng.shuffle(deck)
    private_cards = (deck[0], deck[1])
    public_card = deck[2]
    hand_id = f"leduc-{hand_index:08d}"

    history: list[dict[str, object]] = []
    decision_points: list[DecisionPoint] = []
    contributions = [1, 1]

    terminal_winner = _run_betting_round(
        hand_id=hand_id,
        round_name="preflop",
        bet_size=2,
        public_card=None,
        private_cards=private_cards,
        contributions=contributions,
        history=history,
        decision_points=decision_points,
        policy_by_player=policy_by_player,
    )

    if terminal_winner is None:
        history.append(
            {
                "type": "deal_public",
                "card": public_card,
            }
        )
        terminal_winner = _run_betting_round(
            hand_id=hand_id,
            round_name="public",
            bet_size=4,
            public_card=public_card,
            private_cards=private_cards,
            contributions=contributions,
            history=history,
            decision_points=decision_points,
            policy_by_player=policy_by_player,
        )

    returns = _returns(
        winner=terminal_winner,
        private_cards=private_cards,
        public_card=public_card,
        contributions=contributions,
    )
    finalized = tuple(point.with_terminal_return(returns[point.acting_player]) for point in decision_points)
    return LeducHand(
        hand_id=hand_id,
        private_cards=private_cards,
        public_card=public_card,
        returns=returns,
        decision_points=finalized,
    )


def _run_betting_round(
    hand_id: str,
    round_name: str,
    bet_size: int,
    public_card: str | None,
    private_cards: tuple[str, str],
    contributions: list[int],
    history: list[dict[str, object]],
    decision_points: list[DecisionPoint],
    policy_by_player: tuple[LeducActionPolicy, LeducActionPolicy],
) -> int | None:
    current_player = 0
    checks = 0
    outstanding_bet = False
    bettor: int | None = None

    while True:
        legal_actions = ("fold", "call") if outstanding_bet else ("check", "bet")
        state = LeducPolicyState(
            hand_id=hand_id,
            acting_player=current_player,
            private_card=private_cards[current_player],
            public_card=public_card,
            round_name=round_name,
            pot=sum(contributions),
            contributions=(contributions[0], contributions[1]),
            betting_history=tuple(history),
            legal_actions=legal_actions,
        )
        action = policy_by_player[current_player].choose_action(state)
        if action not in legal_actions:
            raise RuntimeError(f"policy chose illegal action: {action}")

        decision_points.append(
            DecisionPoint(
                hand_id=hand_id,
                step_index=len(decision_points),
                acting_player=current_player,
                visible_state=state.to_visible_state(),
                legal_actions=tuple(legal_actions),
                action=action,
                opponent_private_card=private_cards[1 - current_player],
                opponent_rank=card_rank(private_cards[1 - current_player]),
                terminal_return=None,
            )
        )

        history.append(
            {
                "type": "action",
                "round": round_name,
                "player": current_player,
                "action": action,
            }
        )

        if action == "fold":
            if bettor is None:
                raise RuntimeError("fold is only legal after a bet")
            return bettor
        if action == "call":
            contributions[current_player] += bet_size
            return None
        if action == "bet":
            contributions[current_player] += bet_size
            outstanding_bet = True
            bettor = current_player
            current_player = 1 - current_player
            continue
        if action == "check":
            checks += 1
            if checks == 2:
                return None
            current_player = 1 - current_player
            continue

        raise RuntimeError(f"unknown action: {action}")


def _returns(
    winner: int | None,
    private_cards: tuple[str, str],
    public_card: str,
    contributions: list[int],
) -> tuple[int, int]:
    pot = sum(contributions)
    if winner is None:
        winner = showdown_winner(private_cards=private_cards, public_card=public_card)

    if winner is None:
        return (0, 0)

    returns = [-contributions[0], -contributions[1]]
    returns[winner] = pot - contributions[winner]
    return (returns[0], returns[1])


def card_rank(card: str) -> str:
    """Return the rank component of a Leduc card id."""

    return card[0]


def showdown_winner(private_cards: tuple[str, str], public_card: str) -> int | None:
    """Return the showdown winner, or None for a split pot."""

    public_rank = card_rank(public_card)
    p0_pair = card_rank(private_cards[0]) == public_rank
    p1_pair = card_rank(private_cards[1]) == public_rank
    if p0_pair != p1_pair:
        return 0 if p0_pair else 1

    p0_value = RANK_VALUE[card_rank(private_cards[0])]
    p1_value = RANK_VALUE[card_rank(private_cards[1])]
    if p0_value == p1_value:
        return None
    return 0 if p0_value > p1_value else 1
