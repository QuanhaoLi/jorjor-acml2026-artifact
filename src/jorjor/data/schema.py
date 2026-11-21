from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class DecisionPoint:
    """One supervised sample at a player decision."""

    hand_id: str
    step_index: int
    acting_player: int
    visible_state: dict[str, Any]
    legal_actions: tuple[str, ...]
    action: str
    opponent_private_card: str
    opponent_rank: str
    terminal_return: int | None

    def with_terminal_return(self, terminal_return: int) -> "DecisionPoint":
        """Return a copy with the final return filled in."""

        return replace(self, terminal_return=terminal_return)

    def to_record(self) -> dict[str, Any]:
        """Serialize the point with visible input and hidden labels separated."""

        return {
            "hand_id": self.hand_id,
            "step_index": self.step_index,
            "acting_player": self.acting_player,
            "input": self.visible_state,
            "legal_actions": list(self.legal_actions),
            "labels": {
                "action": self.action,
                "opponent_private_card": self.opponent_private_card,
                "opponent_rank": self.opponent_rank,
                "terminal_return": self.terminal_return,
            },
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "DecisionPoint":
        """Build a decision point from a JSONL record."""

        labels = record["labels"]
        return cls(
            hand_id=record["hand_id"],
            step_index=record["step_index"],
            acting_player=record["acting_player"],
            visible_state=record["input"],
            legal_actions=tuple(record["legal_actions"]),
            action=labels["action"],
            opponent_private_card=labels["opponent_private_card"],
            opponent_rank=labels["opponent_rank"],
            terminal_return=labels["terminal_return"],
        )


def visible_strings(value: Any) -> tuple[str, ...]:
    """Collect exact visible string values from a nested visible-state object."""

    if isinstance(value, str):
        return (value,)
    if isinstance(value, dict):
        strings: list[str] = []
        for item in value.values():
            strings.extend(visible_strings(item))
        return tuple(strings)
    if isinstance(value, list | tuple):
        strings = []
        for item in value:
            strings.extend(visible_strings(item))
        return tuple(strings)
    return ()
