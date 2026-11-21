from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import random
from typing import Protocol

from jorjor.data.schema import DecisionPoint


RANKS_ASC = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
RANKS_DESC = tuple(reversed(RANKS_ASC))
SUITS = ("c", "d", "h", "s")
DECK = tuple(f"{rank}{suit}" for rank in RANKS_ASC for suit in SUITS)
RANK_VALUE = {rank: index for index, rank in enumerate(RANKS_ASC, start=2)}
ACTIONS = ("bet", "call", "check", "fold")


def _build_hand_169_labels() -> tuple[str, ...]:
    labels: list[str] = []
    for high_index, high_rank in enumerate(RANKS_DESC):
        for low_rank in RANKS_DESC[high_index:]:
            if high_rank == low_rank:
                labels.append(f"{high_rank}{low_rank}")
                continue
            labels.append(f"{high_rank}{low_rank}s")
            labels.append(f"{high_rank}{low_rank}o")
    return tuple(labels)


HAND_169_LABELS = _build_hand_169_labels()
HAND_169_LABEL_SET = frozenset(HAND_169_LABELS)


@dataclass(frozen=True)
class LimitHoldemHand:
    """Generated heads-up fixed-limit Hold'em hand."""

    hand_id: str
    private_cards: tuple[tuple[str, str], tuple[str, str]]
    public_cards: tuple[str, ...]
    returns: tuple[int, int]
    decision_points: tuple[DecisionPoint, ...]


@dataclass(frozen=True)
class LimitHoldemPolicyState:
    """Visible state passed to a Limit Hold'em action policy."""

    hand_id: str
    acting_player: int
    private_cards: tuple[str, str]
    public_cards: tuple[str, ...]
    round_name: str
    pot: int
    contributions: tuple[int, int]
    betting_history: tuple[dict[str, object], ...]
    legal_actions: tuple[str, ...]

    def to_visible_state(self) -> dict[str, object]:
        """Serialize visible information without opponent private cards."""

        return {
            "game": "limit_holdem_hu",
            "hand_id": self.hand_id,
            "acting_player": self.acting_player,
            "private_cards": list(self.private_cards),
            "public_cards": list(self.public_cards),
            "round": self.round_name,
            "pot": self.pot,
            "contributions": {
                "0": self.contributions[0],
                "1": self.contributions[1],
            },
            "betting_history": list(self.betting_history),
            "legal_actions": list(self.legal_actions),
        }


class LimitHoldemActionPolicy(Protocol):
    """Action-selection interface for the local Limit Hold'em adapter."""

    def choose_action(self, state: LimitHoldemPolicyState) -> str:
        """Choose one legal action for the current visible state."""


class RandomLimitHoldemPolicy:
    """Uniform random legal policy for fixed-limit adapter smoke tests."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def choose_action(self, state: LimitHoldemPolicyState) -> str:
        """Choose a uniformly random legal action."""

        return self._rng.choice(state.legal_actions)


class HeuristicLimitHoldemTeacherPolicy:
    """Simple deterministic Limit Hold'em teacher for smoke data."""

    def choose_action(self, state: LimitHoldemPolicyState) -> str:
        """Choose a legal heuristic action from visible Limit information."""

        strength = _visible_hand_strength(state.private_cards, state.public_cards)
        if "call" in state.legal_actions:
            if strength >= 2:
                return "call"
            return "fold"

        if "bet" in state.legal_actions:
            if strength >= 2:
                return "bet"
            return "check"

        raise RuntimeError("teacher received unsupported legal-action set")


class LimitHoldemRandomGenerator:
    """Generate local heads-up fixed-limit Hold'em random hands."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def generate_hand(self, hand_index: int) -> LimitHoldemHand:
        """Generate one complete random legal hand."""

        policy = RandomLimitHoldemPolicy(self._rng)
        return generate_limit_holdem_hand(
            hand_index=hand_index,
            policy_by_player=(policy, policy),
            deck_rng=self._rng,
        )

    def generate_hands(self, hand_count: int) -> tuple[LimitHoldemHand, ...]:
        """Generate multiple complete legal hands."""

        return tuple(self.generate_hand(index) for index in range(hand_count))


class LimitHoldemPolicyGenerator:
    """Generate Limit Hold'em hands from explicit player policies."""

    def __init__(self, policy_by_player: tuple[LimitHoldemActionPolicy, LimitHoldemActionPolicy], seed: int = 0) -> None:
        self._policy_by_player = policy_by_player
        self._rng = random.Random(seed)

    def generate_hand(self, hand_index: int) -> LimitHoldemHand:
        """Generate one hand using the configured policies."""

        return generate_limit_holdem_hand(
            hand_index=hand_index,
            policy_by_player=self._policy_by_player,
            deck_rng=self._rng,
        )

    def generate_hands(self, hand_count: int) -> tuple[LimitHoldemHand, ...]:
        """Generate multiple complete hands."""

        return tuple(self.generate_hand(index) for index in range(hand_count))


def generate_limit_holdem_hand(
    hand_index: int,
    policy_by_player: tuple[LimitHoldemActionPolicy, LimitHoldemActionPolicy],
    deck_rng: random.Random | None = None,
) -> LimitHoldemHand:
    """Generate one legal local fixed-limit Hold'em hand."""

    deck = list(DECK)
    rng = random.Random(hand_index) if deck_rng is None else deck_rng
    rng.shuffle(deck)

    private_cards = ((deck[0], deck[2]), (deck[1], deck[3]))
    public_cards = tuple(deck[4:9])
    hand_id = f"limit-{hand_index:08d}"
    history: list[dict[str, object]] = []
    decision_points: list[DecisionPoint] = []
    contributions = [1, 1]
    terminal_winner: int | None = None

    street_specs = (
        ("preflop", (), 2, 0),
        ("flop", public_cards[:3], 2, 1),
        ("turn", public_cards[:4], 4, 1),
        ("river", public_cards[:5], 4, 1),
    )
    for round_name, visible_public_cards, bet_size, starting_player in street_specs:
        if round_name != "preflop":
            history.append(
                {
                    "type": "deal_public",
                    "round": round_name,
                    "cards": list(visible_public_cards),
                }
            )

        terminal_winner = _run_betting_round(
            hand_id=hand_id,
            round_name=round_name,
            bet_size=bet_size,
            public_cards=visible_public_cards,
            private_cards=private_cards,
            contributions=contributions,
            history=history,
            decision_points=decision_points,
            policy_by_player=policy_by_player,
            starting_player=starting_player,
        )
        if terminal_winner is not None:
            break

    returns = _returns(
        winner=terminal_winner,
        private_cards=private_cards,
        public_cards=public_cards,
        contributions=contributions,
    )
    finalized = tuple(point.with_terminal_return(returns[point.acting_player]) for point in decision_points)
    return LimitHoldemHand(
        hand_id=hand_id,
        private_cards=private_cards,
        public_cards=public_cards,
        returns=returns,
        decision_points=finalized,
    )


def _run_betting_round(
    hand_id: str,
    round_name: str,
    bet_size: int,
    public_cards: tuple[str, ...],
    private_cards: tuple[tuple[str, str], tuple[str, str]],
    contributions: list[int],
    history: list[dict[str, object]],
    decision_points: list[DecisionPoint],
    policy_by_player: tuple[LimitHoldemActionPolicy, LimitHoldemActionPolicy],
    starting_player: int,
) -> int | None:
    current_player = starting_player
    checks = 0
    outstanding_bet = False
    bettor: int | None = None

    while True:
        legal_actions = ("fold", "call") if outstanding_bet else ("check", "bet")
        state = LimitHoldemPolicyState(
            hand_id=hand_id,
            acting_player=current_player,
            private_cards=private_cards[current_player],
            public_cards=public_cards,
            round_name=round_name,
            pot=sum(contributions),
            contributions=(contributions[0], contributions[1]),
            betting_history=tuple(history),
            legal_actions=legal_actions,
        )
        action = policy_by_player[current_player].choose_action(state)
        if action not in legal_actions:
            raise RuntimeError(f"policy chose illegal action: {action}")

        opponent_cards = private_cards[1 - current_player]
        decision_points.append(
            DecisionPoint(
                hand_id=hand_id,
                step_index=len(decision_points),
                acting_player=current_player,
                visible_state=state.to_visible_state(),
                legal_actions=tuple(legal_actions),
                action=action,
                opponent_private_card=format_private_cards(opponent_cards),
                opponent_rank=hand_169_label(opponent_cards),
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
    private_cards: tuple[tuple[str, str], tuple[str, str]],
    public_cards: tuple[str, ...],
    contributions: list[int],
) -> tuple[int, int]:
    pot = sum(contributions)
    showdown = winner
    if showdown is None:
        showdown = showdown_winner(private_cards=private_cards, public_cards=public_cards)

    if showdown is None:
        return (0, 0)

    returns = [-contributions[0], -contributions[1]]
    returns[showdown] = pot - contributions[showdown]
    return (returns[0], returns[1])


def parse_card(card: str) -> tuple[str, str]:
    """Return rank and suit for a two-character card id."""

    if len(card) != 2 or card[0] not in RANK_VALUE or card[1] not in SUITS:
        raise ValueError(f"invalid card: {card}")
    return card[0], card[1]


def format_private_cards(cards: tuple[str, str]) -> str:
    """Format two private cards in stable rank-descending order."""

    return " ".join(sorted(cards, key=lambda card: (RANK_VALUE[card[0]], card[1]), reverse=True))


def hand_169_label(cards: tuple[str, str]) -> str:
    """Return the standard 169-class Hold'em label for two private cards."""

    if len(cards) != 2:
        raise ValueError("Hold'em private hand must contain exactly two cards")
    if cards[0] == cards[1]:
        raise ValueError("Hold'em private hand cannot contain duplicate cards")

    rank_a, suit_a = parse_card(cards[0])
    rank_b, suit_b = parse_card(cards[1])
    if rank_a == rank_b:
        return f"{rank_a}{rank_b}"

    high_rank, low_rank = sorted((rank_a, rank_b), key=lambda rank: RANK_VALUE[rank], reverse=True)
    suffix = "s" if suit_a == suit_b else "o"
    return f"{high_rank}{low_rank}{suffix}"


def showdown_winner(
    private_cards: tuple[tuple[str, str], tuple[str, str]],
    public_cards: tuple[str, ...],
) -> int | None:
    """Return the showdown winner, or None for a split pot."""

    rank0 = best_seven_card_rank(private_cards[0] + public_cards)
    rank1 = best_seven_card_rank(private_cards[1] + public_cards)
    if rank0 == rank1:
        return None
    return 0 if rank0 > rank1 else 1


def best_seven_card_rank(cards: tuple[str, ...]) -> tuple[int, tuple[int, ...]]:
    """Return the best five-card rank among seven Hold'em cards."""

    if len(cards) != 7:
        raise ValueError("showdown hand must contain seven cards")
    if len(set(cards)) != 7:
        raise ValueError("showdown hand contains duplicate cards")
    for card in cards:
        parse_card(card)

    return max(_five_card_rank(combo) for combo in combinations(cards, 5))


def _five_card_rank(cards: tuple[str, ...]) -> tuple[int, tuple[int, ...]]:
    values = sorted((RANK_VALUE[card[0]] for card in cards), reverse=True)
    suits = [card[1] for card in cards]
    counts = Counter(values)
    count_groups = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    flush = len(set(suits)) == 1
    straight_high = _straight_high(values)

    if flush and straight_high is not None:
        return 8, (straight_high,)

    if count_groups[0][1] == 4:
        quad = count_groups[0][0]
        kicker = max(value for value in values if value != quad)
        return 7, (quad, kicker)

    if count_groups[0][1] == 3 and count_groups[1][1] == 2:
        return 6, (count_groups[0][0], count_groups[1][0])

    if flush:
        return 5, tuple(values)

    if straight_high is not None:
        return 4, (straight_high,)

    if count_groups[0][1] == 3:
        trips = count_groups[0][0]
        kickers = tuple(value for value in values if value != trips)
        return 3, (trips, *kickers)

    if count_groups[0][1] == 2 and count_groups[1][1] == 2:
        high_pair, low_pair = sorted((count_groups[0][0], count_groups[1][0]), reverse=True)
        kicker = max(value for value in values if value not in {high_pair, low_pair})
        return 2, (high_pair, low_pair, kicker)

    if count_groups[0][1] == 2:
        pair = count_groups[0][0]
        kickers = tuple(value for value in values if value != pair)
        return 1, (pair, *kickers)

    return 0, tuple(values)


def _straight_high(values: list[int]) -> int | None:
    unique_values = sorted(set(values), reverse=True)
    if 14 in unique_values:
        unique_values.append(1)
    for index in range(len(unique_values) - 4):
        window = unique_values[index : index + 5]
        if window[0] - window[4] == 4:
            return window[0]
    return None


def _visible_hand_strength(private_cards: tuple[str, str], public_cards: tuple[str, ...]) -> int:
    """Return a coarse visible strength bucket for heuristic action choice."""

    if len(public_cards) == 5:
        category, _ = best_seven_card_rank(private_cards + public_cards)
        return 3 if category >= 1 else 1

    private_ranks = [card[0] for card in private_cards]
    values = [RANK_VALUE[rank] for rank in private_ranks]
    pair = private_ranks[0] == private_ranks[1]
    high_card = max(values)
    suited = private_cards[0][1] == private_cards[1][1]
    public_ranks = [card[0] for card in public_cards]
    made_pair = any(rank in public_ranks for rank in private_ranks)

    if pair and high_card >= RANK_VALUE["T"]:
        return 3
    if pair or made_pair:
        return 2
    if high_card >= RANK_VALUE["A"]:
        return 2
    if suited and high_card >= RANK_VALUE["Q"]:
        return 2
    return 1
