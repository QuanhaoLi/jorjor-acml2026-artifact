from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .artifacts import checksum_payload, write_json_artifact
from .config import load_limit_abstraction_audit_config
from .data.schema import DecisionPoint
from .data.validation import validate_decision_points
from .envs.limit_holdem import RANK_VALUE, LimitHoldemRandomGenerator
from .limit_random import flatten_decision_points
from .limit_teacher import generate_teacher_points


BROADWAY_RANKS = frozenset(("T", "J", "Q", "K", "A"))


@dataclass(frozen=True)
class LimitAbstractionAuditResult:
    """Result summary for v0.6.0 Limit abstraction audit."""

    hand_count: int
    decision_point_count: int
    illegal_action_count: int
    leakage_count: int
    information_set_count: int
    recommendation: str
    checksum: str
    artifact_dir: Path


def private_hand_bucket(cards: tuple[str, str]) -> str:
    """Return a coarse deterministic private-hand bucket."""

    if len(cards) != 2:
        raise ValueError("private hand bucket requires two cards")
    rank_a, rank_b = cards[0][0], cards[1][0]
    value_a, value_b = RANK_VALUE[rank_a], RANK_VALUE[rank_b]
    high, low = max(value_a, value_b), min(value_a, value_b)
    suited = cards[0][1] == cards[1][1]
    pair = rank_a == rank_b
    gap = high - low

    if pair and high >= RANK_VALUE["T"]:
        return "premium_pair"
    if pair:
        return "small_pair"
    if rank_a in BROADWAY_RANKS and rank_b in BROADWAY_RANKS:
        return "broadway_suited" if suited else "broadway_offsuit"
    if suited and high >= RANK_VALUE["A"]:
        return "ace_suited"
    if high >= RANK_VALUE["A"]:
        return "ace_offsuit"
    if suited and gap <= 2:
        return "suited_connector"
    if gap <= 1 and high >= RANK_VALUE["T"]:
        return "high_connector"
    return "trash"


def public_board_bucket(public_cards: tuple[str, ...]) -> str:
    """Return a coarse deterministic public-board bucket."""

    if not public_cards:
        return "preflop"
    if len(public_cards) == 3:
        ranks = [card[0] for card in public_cards]
        suits = [card[1] for card in public_cards]
        if len(set(ranks)) < 3:
            return "paired_flop"
        if all(rank in BROADWAY_RANKS for rank in ranks):
            return "broadway_flop"
        if len(set(suits)) == 1:
            return "monotone_flop"
        return "low_mixed_flop"
    if len(public_cards) == 4:
        return "turn"
    if len(public_cards) == 5:
        return "river"
    raise ValueError("Limit public board must have 0, 3, 4, or 5 cards")


def history_bucket(point: DecisionPoint) -> str:
    """Return a compact action-history bucket."""

    actions = [
        str(event["action"])
        for event in point.visible_state["betting_history"]
        if event["type"] == "action"
    ]
    if not actions:
        return "empty"
    return "-".join(actions[-6:])


def abstract_information_set_key(point: DecisionPoint) -> str:
    """Build an abstract Limit information-set key from a decision point."""

    state = point.visible_state
    private_cards = tuple(state["private_cards"])
    public_cards = tuple(state["public_cards"])
    legal = ",".join(sorted(point.legal_actions))
    return (
        f"player={state['acting_player']}|round={state['round']}|"
        f"private={private_hand_bucket(private_cards)}|public={public_board_bucket(public_cards)}|"
        f"history={history_bucket(point)}|legal={legal}"
    )


def _generate_points(data_source: str, hand_count: int, seed: int) -> tuple[DecisionPoint, ...]:
    """Generate Limit decision points for audit data."""

    if data_source == "random":
        return flatten_decision_points(LimitHoldemRandomGenerator(seed=seed).generate_hands(hand_count))
    if data_source == "teacher":
        return generate_teacher_points(hand_count=hand_count, seed=seed)
    raise ValueError(f"unknown data source: {data_source}")


def run_limit_abstraction_audit(
    config_path: Path,
    artifact_dir: Path | None = None,
    hand_count_override: int | None = None,
) -> LimitAbstractionAuditResult:
    """Run a sampled Limit abstraction audit and write a report."""

    config = load_limit_abstraction_audit_config(config_path)
    hand_count = hand_count_override if hand_count_override is not None else config.hand_count
    if hand_count <= 0:
        raise ValueError("hand_count_override must be positive")

    output_dir = artifact_dir if artifact_dir is not None else config.artifact_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    points = _generate_points(config.data_source, hand_count=hand_count, seed=config.seed)
    report = validate_decision_points(points)

    key_counts = Counter(abstract_information_set_key(point) for point in points)
    private_counts = Counter(private_hand_bucket(tuple(point.visible_state["private_cards"])) for point in points)
    public_counts = Counter(public_board_bucket(tuple(point.visible_state["public_cards"])) for point in points)
    legal_counts = Counter(",".join(sorted(point.legal_actions)) for point in points)
    round_counts = Counter(str(point.visible_state["round"]) for point in points)
    recommendation = "prototype_abstract_cfr" if len(key_counts) <= 5000 else "use_external_solver"

    record = {
        "version": config.run_version,
        "data_source": config.data_source,
        "seed": config.seed,
        "hand_count": report.hand_count,
        "decision_point_count": report.decision_point_count,
        "illegal_action_count": report.illegal_action_count,
        "leakage_count": report.leakage_count,
        "information_set_count": len(key_counts),
        "private_bucket_counts": dict(sorted(private_counts.items())),
        "public_bucket_counts": dict(sorted(public_counts.items())),
        "legal_action_counts": dict(sorted(legal_counts.items())),
        "round_counts": dict(sorted(round_counts.items())),
        "top_information_sets": [
            {"key": key, "count": count}
            for key, count in key_counts.most_common(20)
        ],
        "recommendation": recommendation,
    }
    checksum = checksum_payload(record)
    write_json_artifact(output_dir / "abstraction_report.json", {**record, "checksum": checksum})

    return LimitAbstractionAuditResult(
        hand_count=report.hand_count,
        decision_point_count=report.decision_point_count,
        illegal_action_count=report.illegal_action_count,
        leakage_count=report.leakage_count,
        information_set_count=len(key_counts),
        recommendation=recommendation,
        checksum=checksum,
        artifact_dir=output_dir,
    )
