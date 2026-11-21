from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random

from .artifacts import checksum_payload, write_json_artifact
from .config import load_limit_abstract_cfr_teacher_config
from .data.schema import DecisionPoint
from .data.validation import validate_decision_points
from .data.writer import read_decision_points_jsonl, write_decision_points_jsonl
from .envs.limit_holdem import (
    DECK,
    LimitHoldemPolicyGenerator,
    LimitHoldemPolicyState,
    RandomLimitHoldemPolicy,
    generate_limit_holdem_hand,
    showdown_winner,
)
from .limit_abstraction import history_bucket, private_hand_bucket, public_board_bucket
from .limit_random import flatten_decision_points
from .limit_teacher import LimitTeacherEvaluation


@dataclass
class _Node:
    """Regret and average-strategy accumulators for one abstract infoset."""

    legal_actions: tuple[str, ...]
    regret_sum: dict[str, float] = field(default_factory=dict)
    strategy_sum: dict[str, float] = field(default_factory=dict)

    def strategy(self) -> dict[str, float]:
        """Return regret-matching strategy."""

        positive = {action: max(0.0, self.regret_sum.get(action, 0.0)) for action in self.legal_actions}
        total = sum(positive.values())
        if total <= 0.0:
            probability = 1.0 / len(self.legal_actions)
            return {action: probability for action in self.legal_actions}
        return {action: positive[action] / total for action in self.legal_actions}

    def add_strategy(self, strategy: dict[str, float], reach_probability: float) -> None:
        """Accumulate average strategy."""

        for action in self.legal_actions:
            self.strategy_sum[action] = self.strategy_sum.get(action, 0.0) + reach_probability * strategy[action]

    def average_strategy(self) -> dict[str, float]:
        """Return normalized average strategy."""

        total = sum(self.strategy_sum.get(action, 0.0) for action in self.legal_actions)
        if total <= 0.0:
            probability = 1.0 / len(self.legal_actions)
            return {action: probability for action in self.legal_actions}
        return {action: self.strategy_sum.get(action, 0.0) / total for action in self.legal_actions}


@dataclass(frozen=True)
class LimitAbstractCfrTrainResult:
    """Training result for sampled abstract Limit CFR."""

    iteration_count: int
    deal_sample_count: int
    information_set_count: int
    policy_table: dict[str, dict[str, float]]


@dataclass(frozen=True)
class LimitAbstractCfrTeacherSmokeResult:
    """Result summary for v0.6.1 Limit abstract CFR teacher smoke."""

    hand_count: int
    decision_point_count: int
    illegal_action_count: int
    leakage_count: int
    information_set_count: int
    teacher_win_rate: float
    teacher_average_return: float
    checksum: str
    artifact_dir: Path


def _history_events(history: tuple[dict[str, object], ...]) -> tuple[str, ...]:
    """Serialize abstract public history events."""

    tokens = []
    for event in history:
        if event["type"] == "deal_public":
            tokens.append(f"deal:{event['round']}")
        elif event["type"] == "action":
            tokens.append(f"p{event['player']}:{event['action']}")
    return tuple(tokens)


def _abstract_key(
    acting_player: int,
    private_cards: tuple[str, str],
    public_cards: tuple[str, ...],
    round_name: str,
    history: tuple[dict[str, object], ...],
    legal_actions: tuple[str, ...],
) -> str:
    """Build an abstract CFR information-set key."""

    pseudo_point = DecisionPoint(
        hand_id="abstract",
        step_index=0,
        acting_player=acting_player,
        visible_state={
            "acting_player": acting_player,
            "private_cards": list(private_cards),
            "public_cards": list(public_cards),
            "round": round_name,
            "betting_history": list(history),
        },
        legal_actions=legal_actions,
        action=legal_actions[0],
        opponent_private_card="",
        opponent_rank="",
        terminal_return=0,
    )
    legal = ",".join(sorted(legal_actions))
    return (
        f"player={acting_player}|round={round_name}|private={private_hand_bucket(private_cards)}|"
        f"public={public_board_bucket(public_cards)}|history={history_bucket(pseudo_point)}|legal={legal}"
    )


def _sample_deals(deal_sample_count: int, seed: int) -> tuple[tuple[tuple[str, str], tuple[str, str], tuple[str, ...]], ...]:
    """Sample deterministic complete Limit deals."""

    rng = random.Random(seed)
    deals = []
    for _ in range(deal_sample_count):
        deck = list(DECK)
        rng.shuffle(deck)
        deals.append(((deck[0], deck[2]), (deck[1], deck[3]), tuple(deck[4:9])))
    return tuple(deals)


def _node_for(nodes: dict[str, _Node], key: str, legal_actions: tuple[str, ...]) -> _Node:
    """Return or create an abstract CFR node."""

    node = nodes.get(key)
    if node is None:
        node = _Node(
            legal_actions=legal_actions,
            regret_sum={action: 0.0 for action in legal_actions},
            strategy_sum={action: 0.0 for action in legal_actions},
        )
        nodes[key] = node
    return node


def _terminal_utility(
    target_player: int,
    winner: int | None,
    private_cards: tuple[tuple[str, str], tuple[str, str]],
    public_cards: tuple[str, ...],
    contributions: tuple[int, int],
) -> float:
    """Return terminal utility for a target player."""

    showdown = winner if winner is not None else showdown_winner(private_cards=private_cards, public_cards=public_cards)
    if showdown is None:
        return 0.0
    pot = contributions[0] + contributions[1]
    returns = [-contributions[0], -contributions[1]]
    returns[showdown] = pot - contributions[showdown]
    return float(returns[target_player])


def _next_street(round_name: str, full_public_cards: tuple[str, ...]) -> tuple[str, tuple[str, ...], int, int] | None:
    """Return the next street spec after a completed betting round."""

    if round_name == "preflop":
        return "flop", full_public_cards[:3], 2, 1
    if round_name == "flop":
        return "turn", full_public_cards[:4], 4, 1
    if round_name == "turn":
        return "river", full_public_cards[:5], 4, 1
    return None


def _cfr(
    nodes: dict[str, _Node],
    target_player: int,
    private_cards: tuple[tuple[str, str], tuple[str, str]],
    full_public_cards: tuple[str, ...],
    round_name: str,
    visible_public_cards: tuple[str, ...],
    bet_size: int,
    current_player: int,
    contributions: tuple[int, int],
    history: tuple[dict[str, object], ...],
    checks: int,
    outstanding_bet: bool,
    bettor: int | None,
    reach: tuple[float, float],
) -> float:
    """Run sampled full-tree CFR recursion for one abstracted Limit deal."""

    legal_actions = ("fold", "call") if outstanding_bet else ("check", "bet")
    key = _abstract_key(
        acting_player=current_player,
        private_cards=private_cards[current_player],
        public_cards=visible_public_cards,
        round_name=round_name,
        history=history,
        legal_actions=legal_actions,
    )
    node = _node_for(nodes, key, legal_actions)
    strategy = node.strategy()
    node.add_strategy(strategy, reach[current_player])

    action_utilities: dict[str, float] = {}
    for action in legal_actions:
        next_history = (
            *history,
            {
                "type": "action",
                "round": round_name,
                "player": current_player,
                "action": action,
            },
        )
        next_reach = list(reach)
        next_reach[current_player] *= strategy[action]

        if action == "fold":
            if bettor is None:
                raise RuntimeError("fold reached without bettor")
            action_utilities[action] = _terminal_utility(
                target_player,
                winner=bettor,
                private_cards=private_cards,
                public_cards=full_public_cards,
                contributions=contributions,
            )
            continue

        if action == "call":
            next_contributions = list(contributions)
            next_contributions[current_player] += bet_size
            next_street = _next_street(round_name, full_public_cards)
            if next_street is None:
                action_utilities[action] = _terminal_utility(
                    target_player,
                    winner=None,
                    private_cards=private_cards,
                    public_cards=full_public_cards,
                    contributions=(next_contributions[0], next_contributions[1]),
                )
            else:
                next_round, next_public, next_bet_size, next_start = next_street
                action_utilities[action] = _cfr(
                    nodes,
                    target_player,
                    private_cards,
                    full_public_cards,
                    next_round,
                    next_public,
                    next_bet_size,
                    next_start,
                    (next_contributions[0], next_contributions[1]),
                    (
                        *next_history,
                        {
                            "type": "deal_public",
                            "round": next_round,
                            "cards": list(next_public),
                        },
                    ),
                    0,
                    False,
                    None,
                    (next_reach[0], next_reach[1]),
                )
            continue

        if action == "bet":
            next_contributions = list(contributions)
            next_contributions[current_player] += bet_size
            action_utilities[action] = _cfr(
                nodes,
                target_player,
                private_cards,
                full_public_cards,
                round_name,
                visible_public_cards,
                bet_size,
                1 - current_player,
                (next_contributions[0], next_contributions[1]),
                next_history,
                checks,
                True,
                current_player,
                (next_reach[0], next_reach[1]),
            )
            continue

        if action == "check":
            next_checks = checks + 1
            if next_checks == 2:
                next_street = _next_street(round_name, full_public_cards)
                if next_street is None:
                    action_utilities[action] = _terminal_utility(
                        target_player,
                        winner=None,
                        private_cards=private_cards,
                        public_cards=full_public_cards,
                        contributions=contributions,
                    )
                else:
                    next_round, next_public, next_bet_size, next_start = next_street
                    action_utilities[action] = _cfr(
                        nodes,
                        target_player,
                        private_cards,
                        full_public_cards,
                        next_round,
                        next_public,
                        next_bet_size,
                        next_start,
                        contributions,
                        (
                            *next_history,
                            {
                                "type": "deal_public",
                                "round": next_round,
                                "cards": list(next_public),
                            },
                        ),
                        0,
                        False,
                        None,
                        (next_reach[0], next_reach[1]),
                    )
            else:
                action_utilities[action] = _cfr(
                    nodes,
                    target_player,
                    private_cards,
                    full_public_cards,
                    round_name,
                    visible_public_cards,
                    bet_size,
                    1 - current_player,
                    contributions,
                    next_history,
                    next_checks,
                    False,
                    None,
                    (next_reach[0], next_reach[1]),
                )
            continue

        raise RuntimeError(f"unknown action: {action}")

    node_utility = sum(strategy[action] * action_utilities[action] for action in legal_actions)
    if current_player == target_player:
        opponent_reach = reach[1 - current_player]
        for action in legal_actions:
            node.regret_sum[action] = node.regret_sum.get(action, 0.0) + opponent_reach * (
                action_utilities[action] - node_utility
            )
    return node_utility


def train_limit_abstract_cfr(iteration_count: int, deal_sample_count: int, seed: int) -> LimitAbstractCfrTrainResult:
    """Train sampled abstract CFR for local Limit Hold'em."""

    if iteration_count <= 0:
        raise ValueError("iteration_count must be positive")
    if deal_sample_count <= 0:
        raise ValueError("deal_sample_count must be positive")

    nodes: dict[str, _Node] = {}
    deals = list(_sample_deals(deal_sample_count, seed))
    rng = random.Random(seed)
    for _ in range(iteration_count):
        rng.shuffle(deals)
        for private0, private1, public_cards in deals:
            private_cards = (private0, private1)
            for target_player in (0, 1):
                _cfr(
                    nodes=nodes,
                    target_player=target_player,
                    private_cards=private_cards,
                    full_public_cards=public_cards,
                    round_name="preflop",
                    visible_public_cards=(),
                    bet_size=2,
                    current_player=0,
                    contributions=(1, 1),
                    history=(),
                    checks=0,
                    outstanding_bet=False,
                    bettor=None,
                    reach=(1.0, 1.0),
                )

    policy_table = {
        key: {action: round(probability, 12) for action, probability in node.average_strategy().items()}
        for key, node in sorted(nodes.items())
    }
    return LimitAbstractCfrTrainResult(
        iteration_count=iteration_count,
        deal_sample_count=deal_sample_count,
        information_set_count=len(policy_table),
        policy_table=policy_table,
    )


class LimitAbstractCfrPolicy:
    """Policy sampler for a Limit abstract CFR average policy."""

    def __init__(self, policy_table: dict[str, dict[str, float]], seed: int) -> None:
        self._policy_table = policy_table
        self._rng = random.Random(seed)

    def choose_action(self, state: LimitHoldemPolicyState) -> str:
        """Sample one legal action from the abstract average policy."""

        filtered = self.action_distribution(state)
        total = sum(filtered.values())
        threshold = self._rng.random() * total
        cumulative = 0.0
        for action in state.legal_actions:
            cumulative += filtered[action]
            if threshold <= cumulative:
                return action
        return state.legal_actions[-1]

    def action_distribution(self, state: LimitHoldemPolicyState) -> dict[str, float]:
        """Return normalized legal-action probabilities for this state."""

        key = _abstract_key(
            acting_player=state.acting_player,
            private_cards=state.private_cards,
            public_cards=state.public_cards,
            round_name=state.round_name,
            history=state.betting_history,
            legal_actions=state.legal_actions,
        )
        probabilities = self._policy_table.get(key)
        if probabilities is None:
            probability = 1.0 / len(state.legal_actions)
            probabilities = {action: probability for action in state.legal_actions}
        filtered = {action: probabilities.get(action, 0.0) for action in state.legal_actions}
        total = sum(filtered.values())
        if total <= 0.0:
            probability = 1.0 / len(state.legal_actions)
            filtered = {action: probability for action in state.legal_actions}
            total = sum(filtered.values())
        return {action: filtered[action] / total for action in state.legal_actions}

    def generate_points(self, hand_count: int, seed: int) -> tuple[DecisionPoint, ...]:
        """Generate Limit self-play decision points from the abstract policy."""

        generator = LimitHoldemPolicyGenerator(policy_by_player=(self, self), seed=seed)
        return flatten_decision_points(generator.generate_hands(hand_count))


def evaluate_abstract_cfr_against_random(
    policy_table: dict[str, dict[str, float]],
    seed: int,
    hand_count: int,
) -> LimitTeacherEvaluation:
    """Evaluate abstract CFR against random with alternating seats."""

    random_policy = RandomLimitHoldemPolicy(random.Random(seed))
    deck_rng = random.Random(seed + 1_000_003)
    teacher_wins = 0
    random_wins = 0
    ties = 0
    teacher_total_return = 0
    teacher_player0_hands = 0
    teacher_player1_hands = 0

    for index in range(hand_count):
        teacher = LimitAbstractCfrPolicy(policy_table, seed=seed + index + 17)
        teacher_player = index % 2
        if teacher_player == 0:
            teacher_player0_hands += 1
            policies = (teacher, random_policy)
        else:
            teacher_player1_hands += 1
            policies = (random_policy, teacher)
        hand = generate_limit_holdem_hand(
            hand_index=index,
            policy_by_player=policies,
            deck_rng=deck_rng,
        )
        teacher_return = hand.returns[teacher_player]
        teacher_total_return += teacher_return
        if teacher_return > 0:
            teacher_wins += 1
        elif teacher_return < 0:
            random_wins += 1
        else:
            ties += 1

    return LimitTeacherEvaluation(
        hand_count=hand_count,
        teacher_player0_hands=teacher_player0_hands,
        teacher_player1_hands=teacher_player1_hands,
        teacher_wins=teacher_wins,
        random_wins=random_wins,
        ties=ties,
        teacher_total_return=teacher_total_return,
        teacher_win_rate=teacher_wins / hand_count,
        teacher_average_return=teacher_total_return / hand_count,
    )


def run_limit_abstract_cfr_teacher_smoke(
    config_path: Path,
    artifact_dir: Path | None = None,
) -> LimitAbstractCfrTeacherSmokeResult:
    """Train abstract CFR, generate data, and write smoke artifacts."""

    config = load_limit_abstract_cfr_teacher_config(config_path)
    output_dir = artifact_dir if artifact_dir is not None else config.artifact_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_result = train_limit_abstract_cfr(
        iteration_count=config.iteration_count,
        deal_sample_count=config.deal_sample_count,
        seed=config.seed,
    )
    teacher = LimitAbstractCfrPolicy(train_result.policy_table, seed=config.seed + 97)
    points = teacher.generate_points(hand_count=config.hand_count, seed=config.seed + 193)
    report = validate_decision_points(points)
    evaluation = evaluate_abstract_cfr_against_random(
        policy_table=train_result.policy_table,
        seed=config.seed,
        hand_count=config.evaluation_hand_count,
    )

    policy_record = {
        "version": config.run_version,
        "seed": config.seed,
        "iteration_count": config.iteration_count,
        "deal_sample_count": config.deal_sample_count,
        "information_set_count": train_result.information_set_count,
        "policy_table": train_result.policy_table,
    }
    policy_checksum = checksum_payload(policy_record)
    write_json_artifact(output_dir / "average_policy.json", {**policy_record, "checksum": policy_checksum})

    dataset_path = output_dir / "decision_points.jsonl"
    write_decision_points_jsonl(dataset_path, points)
    read_back_count = len(read_decision_points_jsonl(dataset_path))
    report_record = {
        "version": config.run_version,
        "seed": config.seed,
        "iteration_count": config.iteration_count,
        "deal_sample_count": config.deal_sample_count,
        "hand_count": report.hand_count,
        "decision_point_count": report.decision_point_count,
        "illegal_action_count": report.illegal_action_count,
        "leakage_count": report.leakage_count,
        "information_set_count": train_result.information_set_count,
        "read_back_count": read_back_count,
        "read_back_matches": read_back_count == report.decision_point_count,
        "policy_checksum": policy_checksum,
        "teacher_evaluation": evaluation.to_record(),
    }
    checksum = checksum_payload(report_record)
    write_json_artifact(output_dir / "abstract_cfr_teacher_report.json", {**report_record, "checksum": checksum})

    return LimitAbstractCfrTeacherSmokeResult(
        hand_count=report.hand_count,
        decision_point_count=report.decision_point_count,
        illegal_action_count=report.illegal_action_count,
        leakage_count=report.leakage_count,
        information_set_count=train_result.information_set_count,
        teacher_win_rate=evaluation.teacher_win_rate,
        teacher_average_return=evaluation.teacher_average_return,
        checksum=checksum,
        artifact_dir=output_dir,
    )
