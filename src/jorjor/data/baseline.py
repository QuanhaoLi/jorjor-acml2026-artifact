from __future__ import annotations

from dataclasses import dataclass
import math

from jorjor.envs.leduc import DECK

from .schema import DecisionPoint


@dataclass(frozen=True)
class BaselineMetrics:
    """Aggregate metrics for an opponent-card baseline."""

    count: int
    accuracy: float
    cross_entropy: float

    def to_record(self) -> dict[str, float | int]:
        """Serialize baseline metrics."""

        return {
            "count": self.count,
            "accuracy": self.accuracy,
            "cross_entropy": self.cross_entropy,
        }


def possible_opponent_cards(point: DecisionPoint) -> tuple[str, ...]:
    """Return opponent cards possible from visible card-removal information."""

    own_card = point.visible_state["private_card"]
    public_card = point.visible_state.get("public_card")
    blocked = {own_card}
    if public_card is not None:
        blocked.add(public_card)
    return tuple(card for card in DECK if card not in blocked)


def card_removal_probability(point: DecisionPoint) -> float:
    """Probability assigned to the true opponent card by card removal."""

    possible_cards = possible_opponent_cards(point)
    if point.opponent_private_card not in possible_cards:
        return 0.0
    return 1.0 / len(possible_cards)


def card_removal_prediction(point: DecisionPoint) -> str:
    """Return a deterministic top-1 prediction from possible cards."""

    return sorted(possible_opponent_cards(point))[0]


def evaluate_card_removal_baseline(points: tuple[DecisionPoint, ...]) -> BaselineMetrics:
    """Evaluate the visible-information/card-removal opponent-card baseline."""

    if not points:
        raise ValueError("cannot evaluate baseline with zero points")

    correct = 0
    total_nll = 0.0
    for point in points:
        probability = card_removal_probability(point)
        if probability <= 0.0:
            raise ValueError("true opponent card is impossible under visible information")
        if card_removal_prediction(point) == point.opponent_private_card:
            correct += 1
        total_nll += -math.log(probability)

    return BaselineMetrics(
        count=len(points),
        accuracy=correct / len(points),
        cross_entropy=total_nll / len(points),
    )
