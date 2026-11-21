from __future__ import annotations

from dataclasses import dataclass

from .schema import DecisionPoint, visible_strings


@dataclass(frozen=True)
class ValidationReport:
    """Summary of legality and leakage validation."""

    hand_count: int
    decision_point_count: int
    illegal_action_count: int
    leakage_count: int

    def to_record(self) -> dict[str, int]:
        """Serialize the validation summary."""

        return {
            "hand_count": self.hand_count,
            "decision_point_count": self.decision_point_count,
            "illegal_action_count": self.illegal_action_count,
            "leakage_count": self.leakage_count,
        }


def has_hidden_card_leakage(point: DecisionPoint) -> bool:
    """Return True if any opponent private card appears in visible input."""

    visible = set(visible_strings(point.visible_state))
    hidden_cards = tuple(card for card in point.opponent_private_card.split() if card)
    if not hidden_cards:
        hidden_cards = (point.opponent_private_card,)
    return any(card in visible for card in hidden_cards)


def has_illegal_action(point: DecisionPoint) -> bool:
    """Return True if the sampled action is not listed as legal."""

    return point.action not in point.legal_actions


def validate_decision_points(points: tuple[DecisionPoint, ...]) -> ValidationReport:
    """Validate a collection of decision points."""

    hand_ids = {point.hand_id for point in points}
    illegal_count = sum(1 for point in points if has_illegal_action(point))
    leakage_count = sum(1 for point in points if has_hidden_card_leakage(point))
    return ValidationReport(
        hand_count=len(hand_ids),
        decision_point_count=len(points),
        illegal_action_count=illegal_count,
        leakage_count=leakage_count,
    )
