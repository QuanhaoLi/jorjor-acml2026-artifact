from __future__ import annotations

import hashlib
import random

from jorjor.data.schema import DecisionPoint

LIMIT_ACTIONS = ("bet", "call", "check", "fold")


LIMIT_INPUT_CONTROLS = {
    "none",
    "drop_history",
    "shuffle_history_order",
    "shuffle_history_order_seed_1",
    "shuffle_history_order_seed_2",
    "shuffle_history_order_seed_3",
    "reverse_history_order",
    "sort_history_actions",
    "bag_of_actions",
    "history_presence_only",
    "history_last_action_only",
    "mask_public_card",
    "mask_private_card",
    "drop_history_mask_cards",
    "mask_legal_actions",
    "mask_round_position",
    "mask_all_cards",
    "history_only",
    "legal_only",
    "round_position_only",
    "round_position_action_prior",
    "no_observation",
}


def _stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _action_events(state: dict[str, object]) -> list[dict[str, object]]:
    return [dict(event) for event in state["betting_history"] if event["type"] == "action"]


def _shuffle_events(point: DecisionPoint, action_events: list[dict[str, object]], salt: str) -> list[dict[str, object]]:
    shuffled = list(action_events)
    random.Random(_stable_seed(f"{point.hand_id}:{point.step_index}:{salt}")).shuffle(shuffled)
    return shuffled


def controlled_limit_points(points: tuple[DecisionPoint, ...], input_control: str) -> tuple[DecisionPoint, ...]:
    """Return decision points with controlled visible inputs and unchanged labels."""

    if input_control == "none":
        return points

    controlled: list[DecisionPoint] = []
    for point in points:
        state = dict(point.visible_state)
        if input_control == "drop_history":
            state["betting_history"] = []
        elif input_control == "shuffle_history_order":
            action_events = _action_events(state)
            state["betting_history"] = _shuffle_events(point, action_events, "shuffle_history_order")
        elif input_control in {
            "shuffle_history_order_seed_1",
            "shuffle_history_order_seed_2",
            "shuffle_history_order_seed_3",
        }:
            action_events = _action_events(state)
            state["betting_history"] = _shuffle_events(point, action_events, input_control)
        elif input_control == "reverse_history_order":
            state["betting_history"] = list(reversed(_action_events(state)))
        elif input_control == "sort_history_actions":
            state["betting_history"] = sorted(
                _action_events(state),
                key=lambda event: LIMIT_ACTIONS.index(str(event["action"])),
            )
        elif input_control == "bag_of_actions":
            counts = {action: 0 for action in LIMIT_ACTIONS}
            for event in _action_events(state):
                counts[str(event["action"])] += 1
            state["betting_history"] = [
                {"type": "bag_action", "action": action}
                for action in LIMIT_ACTIONS
                for _ in range(counts[action])
            ]
        elif input_control == "history_presence_only":
            present_actions = {
                str(event["action"])
                for event in _action_events(state)
            }
            state["betting_history"] = [
                {"type": "bag_action", "action": action}
                for action in LIMIT_ACTIONS
                if action in present_actions
            ]
        elif input_control == "history_last_action_only":
            action_events = _action_events(state)
            state["betting_history"] = action_events[-1:] if action_events else []
        elif input_control == "mask_public_card":
            state["public_cards"] = []
        elif input_control == "mask_private_card":
            state["private_cards"] = ["mask"]
        elif input_control == "drop_history_mask_cards":
            state["betting_history"] = []
            state["public_cards"] = []
            state["private_cards"] = ["mask"]
        elif input_control == "mask_legal_actions":
            state["legal_actions"] = []
        elif input_control == "mask_round_position":
            state["acting_player"] = "mask"
            state["round"] = "mask"
        elif input_control == "mask_all_cards":
            state["public_cards"] = []
            state["private_cards"] = ["mask"]
        elif input_control == "history_only":
            state["acting_player"] = "mask"
            state["round"] = "mask"
            state["public_cards"] = []
            state["private_cards"] = ["mask"]
            state["legal_actions"] = []
        elif input_control == "legal_only":
            state["acting_player"] = "mask"
            state["round"] = "mask"
            state["public_cards"] = []
            state["private_cards"] = ["mask"]
            state["betting_history"] = []
        elif input_control == "round_position_only":
            state["public_cards"] = []
            state["private_cards"] = ["mask"]
            state["betting_history"] = []
            state["legal_actions"] = []
        elif input_control == "round_position_action_prior":
            state["public_cards"] = []
            state["private_cards"] = ["mask"]
            state["betting_history"] = []
        elif input_control == "no_observation":
            state["acting_player"] = "mask"
            state["round"] = "mask"
            state["public_cards"] = []
            state["private_cards"] = ["mask"]
            state["betting_history"] = []
            state["legal_actions"] = []
        else:
            raise ValueError(f"unknown input control: {input_control}")

        controlled.append(
            DecisionPoint(
                hand_id=point.hand_id,
                step_index=point.step_index,
                acting_player=point.acting_player,
                visible_state=state,
                legal_actions=point.legal_actions,
                action=point.action,
                opponent_private_card=point.opponent_private_card,
                opponent_rank=point.opponent_rank,
                terminal_return=point.terminal_return,
            )
        )

    return tuple(controlled)


def controlled_range_labels(
    points: tuple[DecisionPoint, ...],
    range_label_control: str,
    seed: int,
) -> tuple[DecisionPoint, ...]:
    """Return points with controlled opponent-range labels and unchanged inputs."""

    if range_label_control == "none":
        return points
    if range_label_control != "shuffle_within_split":
        raise ValueError(f"unknown range label control: {range_label_control}")

    labels = [point.opponent_rank for point in points]
    random.Random(seed).shuffle(labels)
    return tuple(
        DecisionPoint(
            hand_id=point.hand_id,
            step_index=point.step_index,
            acting_player=point.acting_player,
            visible_state=point.visible_state,
            legal_actions=point.legal_actions,
            action=point.action,
            opponent_private_card=point.opponent_private_card,
            opponent_rank=shuffled_label,
            terminal_return=point.terminal_return,
        )
        for point, shuffled_label in zip(points, labels, strict=True)
    )
