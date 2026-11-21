from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random

from .artifacts import checksum_payload, write_json_artifact
from .config import load_leduc_cfr_teacher_config
from .data.schema import DecisionPoint
from .data.validation import validate_decision_points
from .data.writer import read_decision_points_jsonl, write_decision_points_jsonl
from .envs.leduc import (
    DECK,
    LeducPolicyGenerator,
    LeducPolicyState,
    RandomLeducPolicy,
    generate_leduc_hand,
    showdown_winner,
)
from .leduc_random import flatten_decision_points
from .leduc_teacher import TeacherEvaluation


ACTIONS = ("bet", "call", "check", "fold")


@dataclass
class _CfrNode:
    """Regret and average-strategy accumulators for one information set."""

    legal_actions: tuple[str, ...]
    regret_sum: dict[str, float] = field(default_factory=dict)
    strategy_sum: dict[str, float] = field(default_factory=dict)

    def strategy(self) -> dict[str, float]:
        """Return regret-matching strategy for this information set."""

        positive = {action: max(0.0, self.regret_sum.get(action, 0.0)) for action in self.legal_actions}
        total = sum(positive.values())
        if total <= 0.0:
            probability = 1.0 / len(self.legal_actions)
            return {action: probability for action in self.legal_actions}
        return {action: positive[action] / total for action in self.legal_actions}

    def add_strategy(self, strategy: dict[str, float], reach_probability: float) -> None:
        """Accumulate average-strategy mass."""

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
class LeducCfrTrainResult:
    """Tabular CFR training result."""

    iteration_count: int
    information_set_count: int
    policy_table: dict[str, dict[str, float]]


@dataclass(frozen=True)
class LeducCfrTeacherSmokeResult:
    """Result summary for v0.5.0 Leduc CFR teacher smoke."""

    hand_count: int
    decision_point_count: int
    illegal_action_count: int
    leakage_count: int
    information_set_count: int
    teacher_win_rate: float
    teacher_average_return: float
    checksum: str
    artifact_dir: Path


def _history_key(history: tuple[tuple[str, int, str] | tuple[str], ...]) -> str:
    """Serialize public action history for information-set keys."""

    if not history:
        return "empty"
    tokens = []
    for event in history:
        if event[0] == "deal":
            tokens.append("deal")
        else:
            _, player, action = event
            tokens.append(f"p{player}:{action}")
    return "/".join(tokens)


def information_set_key(
    acting_player: int,
    private_card: str,
    public_card: str | None,
    round_name: str,
    history: tuple[tuple[str, int, str] | tuple[str], ...],
    legal_actions: tuple[str, ...],
) -> str:
    """Build a stable information-set key for local Leduc CFR."""

    public_token = "none" if public_card is None else public_card
    return (
        f"p={acting_player}|priv={private_card}|pub={public_token}|round={round_name}|"
        f"hist={_history_key(history)}|legal={','.join(legal_actions)}"
    )


def information_set_key_from_state(state: LeducPolicyState) -> str:
    """Build a CFR information-set key from a policy state."""

    history: list[tuple[str, int, str] | tuple[str]] = []
    for event in state.betting_history:
        if event["type"] == "deal_public":
            history.append(("deal",))
        elif event["type"] == "action":
            history.append(("action", int(event["player"]), str(event["action"])))
    return information_set_key(
        acting_player=state.acting_player,
        private_card=state.private_card,
        public_card=state.public_card,
        round_name=state.round_name,
        history=tuple(history),
        legal_actions=state.legal_actions,
    )


def _deals() -> tuple[tuple[str, str, str], ...]:
    """Return all ordered private-card and public-card deals."""

    deals = []
    for player0_card in DECK:
        for player1_card in DECK:
            if player1_card == player0_card:
                continue
            for public_card in DECK:
                if public_card != player0_card and public_card != player1_card:
                    deals.append((player0_card, player1_card, public_card))
    return tuple(deals)


def _terminal_utility(
    target_player: int,
    winner: int | None,
    private_cards: tuple[str, str],
    public_card: str,
    contributions: tuple[int, int],
) -> float:
    """Return terminal utility for the target player."""

    if winner is None:
        winner = showdown_winner(private_cards=private_cards, public_card=public_card)
    if winner is None:
        return 0.0

    pot = contributions[0] + contributions[1]
    returns = [-contributions[0], -contributions[1]]
    returns[winner] = pot - contributions[winner]
    return float(returns[target_player])


def _node_for(nodes: dict[str, _CfrNode], key: str, legal_actions: tuple[str, ...]) -> _CfrNode:
    """Return an existing CFR node or create one."""

    node = nodes.get(key)
    if node is None:
        node = _CfrNode(
            legal_actions=legal_actions,
            regret_sum={action: 0.0 for action in legal_actions},
            strategy_sum={action: 0.0 for action in legal_actions},
        )
        nodes[key] = node
    return node


def _cfr(
    nodes: dict[str, _CfrNode],
    target_player: int,
    private_cards: tuple[str, str],
    public_card: str,
    round_name: str,
    current_player: int,
    contributions: tuple[int, int],
    history: tuple[tuple[str, int, str] | tuple[str], ...],
    checks: int,
    outstanding_bet: bool,
    bettor: int | None,
    reach: tuple[float, float],
) -> float:
    """Run one full-tree CFR recursion for a fixed deal and target player."""

    legal_actions = ("fold", "call") if outstanding_bet else ("check", "bet")
    key = information_set_key(
        acting_player=current_player,
        private_card=private_cards[current_player],
        public_card=None if round_name == "preflop" else public_card,
        round_name=round_name,
        history=history,
        legal_actions=legal_actions,
    )
    node = _node_for(nodes, key, legal_actions)
    strategy = node.strategy()
    node.add_strategy(strategy, reach[current_player])

    action_utilities: dict[str, float] = {}
    for action in legal_actions:
        next_history = (*history, ("action", current_player, action))
        next_reach = list(reach)
        next_reach[current_player] *= strategy[action]

        if action == "fold":
            if bettor is None:
                raise RuntimeError("fold reached without bettor")
            action_utilities[action] = _terminal_utility(
                target_player=target_player,
                winner=bettor,
                private_cards=private_cards,
                public_card=public_card,
                contributions=contributions,
            )
            continue

        if action == "call":
            next_contributions = list(contributions)
            next_contributions[current_player] += 2 if round_name == "preflop" else 4
            if round_name == "preflop":
                action_utilities[action] = _cfr(
                    nodes=nodes,
                    target_player=target_player,
                    private_cards=private_cards,
                    public_card=public_card,
                    round_name="public",
                    current_player=0,
                    contributions=(next_contributions[0], next_contributions[1]),
                    history=(*next_history, ("deal",)),
                    checks=0,
                    outstanding_bet=False,
                    bettor=None,
                    reach=(next_reach[0], next_reach[1]),
                )
            else:
                action_utilities[action] = _terminal_utility(
                    target_player=target_player,
                    winner=None,
                    private_cards=private_cards,
                    public_card=public_card,
                    contributions=(next_contributions[0], next_contributions[1]),
                )
            continue

        if action == "bet":
            next_contributions = list(contributions)
            next_contributions[current_player] += 2 if round_name == "preflop" else 4
            action_utilities[action] = _cfr(
                nodes=nodes,
                target_player=target_player,
                private_cards=private_cards,
                public_card=public_card,
                round_name=round_name,
                current_player=1 - current_player,
                contributions=(next_contributions[0], next_contributions[1]),
                history=next_history,
                checks=checks,
                outstanding_bet=True,
                bettor=current_player,
                reach=(next_reach[0], next_reach[1]),
            )
            continue

        if action == "check":
            next_checks = checks + 1
            if next_checks == 2:
                if round_name == "preflop":
                    action_utilities[action] = _cfr(
                        nodes=nodes,
                        target_player=target_player,
                        private_cards=private_cards,
                        public_card=public_card,
                        round_name="public",
                        current_player=0,
                        contributions=contributions,
                        history=(*next_history, ("deal",)),
                        checks=0,
                        outstanding_bet=False,
                        bettor=None,
                        reach=(next_reach[0], next_reach[1]),
                    )
                else:
                    action_utilities[action] = _terminal_utility(
                        target_player=target_player,
                        winner=None,
                        private_cards=private_cards,
                        public_card=public_card,
                        contributions=contributions,
                    )
            else:
                action_utilities[action] = _cfr(
                    nodes=nodes,
                    target_player=target_player,
                    private_cards=private_cards,
                    public_card=public_card,
                    round_name=round_name,
                    current_player=1 - current_player,
                    contributions=contributions,
                    history=next_history,
                    checks=next_checks,
                    outstanding_bet=False,
                    bettor=None,
                    reach=(next_reach[0], next_reach[1]),
                )
            continue

        raise RuntimeError(f"unknown action: {action}")

    node_utility = sum(strategy[action] * action_utilities[action] for action in legal_actions)
    if current_player == target_player:
        opponent_reach = reach[1 - current_player]
        for action in legal_actions:
            regret = action_utilities[action] - node_utility
            node.regret_sum[action] = node.regret_sum.get(action, 0.0) + opponent_reach * regret
    return node_utility


def train_leduc_cfr(iteration_count: int, seed: int) -> LeducCfrTrainResult:
    """Train a local tabular CFR policy for Leduc."""

    if iteration_count <= 0:
        raise ValueError("iteration_count must be positive")

    rng = random.Random(seed)
    nodes: dict[str, _CfrNode] = {}
    deals = list(_deals())
    for _ in range(iteration_count):
        rng.shuffle(deals)
        for player0_card, player1_card, public_card in deals:
            private_cards = (player0_card, player1_card)
            for target_player in (0, 1):
                _cfr(
                    nodes=nodes,
                    target_player=target_player,
                    private_cards=private_cards,
                    public_card=public_card,
                    round_name="preflop",
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
    return LeducCfrTrainResult(
        iteration_count=iteration_count,
        information_set_count=len(policy_table),
        policy_table=policy_table,
    )


class LeducCfrPolicy:
    """Action policy backed by a tabular CFR average strategy."""

    def __init__(self, policy_table: dict[str, dict[str, float]], seed: int) -> None:
        self._policy_table = policy_table
        self._rng = random.Random(seed)

    def choose_action(self, state: LeducPolicyState) -> str:
        """Sample one legal action from the average strategy."""

        key = information_set_key_from_state(state)
        probabilities = self._policy_table.get(key)
        if probabilities is None:
            probability = 1.0 / len(state.legal_actions)
            probabilities = {action: probability for action in state.legal_actions}
        filtered = {action: probabilities.get(action, 0.0) for action in state.legal_actions}
        total = sum(filtered.values())
        if total <= 0.0:
            probability = 1.0 / len(state.legal_actions)
            filtered = {action: probability for action in state.legal_actions}
            total = 1.0

        threshold = self._rng.random() * total
        cumulative = 0.0
        for action in state.legal_actions:
            cumulative += filtered[action]
            if threshold <= cumulative:
                return action
        return state.legal_actions[-1]

    def generate_points(self, hand_count: int, seed: int) -> tuple[DecisionPoint, ...]:
        """Generate CFR self-play decision points."""

        generator = LeducPolicyGenerator(policy_by_player=(self, self), seed=seed)
        return flatten_decision_points(generator.generate_hands(hand_count))


def evaluate_cfr_teacher_against_random(
    policy_table: dict[str, dict[str, float]],
    seed: int,
    hand_count: int,
) -> TeacherEvaluation:
    """Evaluate the CFR teacher against random with alternating seats."""

    random_policy = RandomLeducPolicy(random.Random(seed))
    deck_rng = random.Random(seed + 1_000_003)
    teacher_wins = 0
    random_wins = 0
    ties = 0
    teacher_total_return = 0
    teacher_player0_hands = 0
    teacher_player1_hands = 0

    for index in range(hand_count):
        teacher = LeducCfrPolicy(policy_table, seed=seed + index + 17)
        teacher_player = index % 2
        if teacher_player == 0:
            teacher_player0_hands += 1
            policies = (teacher, random_policy)
        else:
            teacher_player1_hands += 1
            policies = (random_policy, teacher)
        hand = generate_leduc_hand(
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

    return TeacherEvaluation(
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


def run_leduc_cfr_teacher_smoke(
    config_path: Path,
    artifact_dir: Path | None = None,
) -> LeducCfrTeacherSmokeResult:
    """Train CFR, generate teacher data, and write v0.5.0 smoke artifacts."""

    config = load_leduc_cfr_teacher_config(config_path)
    output_dir = artifact_dir if artifact_dir is not None else config.artifact_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cfr_result = train_leduc_cfr(iteration_count=config.iteration_count, seed=config.seed)
    teacher = LeducCfrPolicy(cfr_result.policy_table, seed=config.seed + 97)
    points = teacher.generate_points(hand_count=config.hand_count, seed=config.seed + 193)
    report = validate_decision_points(points)
    evaluation = evaluate_cfr_teacher_against_random(
        policy_table=cfr_result.policy_table,
        seed=config.seed,
        hand_count=config.evaluation_hand_count,
    )

    policy_record = {
        "version": config.run_version,
        "seed": config.seed,
        "iteration_count": config.iteration_count,
        "information_set_count": cfr_result.information_set_count,
        "policy_table": cfr_result.policy_table,
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
        "hand_count": report.hand_count,
        "decision_point_count": report.decision_point_count,
        "illegal_action_count": report.illegal_action_count,
        "leakage_count": report.leakage_count,
        "information_set_count": cfr_result.information_set_count,
        "read_back_count": read_back_count,
        "read_back_matches": read_back_count == report.decision_point_count,
        "policy_checksum": policy_checksum,
        "teacher_evaluation": evaluation.to_record(),
    }
    checksum = checksum_payload(report_record)
    write_json_artifact(output_dir / "cfr_teacher_report.json", {**report_record, "checksum": checksum})

    return LeducCfrTeacherSmokeResult(
        hand_count=report.hand_count,
        decision_point_count=report.decision_point_count,
        illegal_action_count=report.illegal_action_count,
        leakage_count=report.leakage_count,
        information_set_count=cfr_result.information_set_count,
        teacher_win_rate=evaluation.teacher_win_rate,
        teacher_average_return=evaluation.teacher_average_return,
        checksum=checksum,
        artifact_dir=output_dir,
    )
