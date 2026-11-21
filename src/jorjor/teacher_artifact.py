from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.envs.limit_holdem import ACTIONS, LimitHoldemPolicyState
from jorjor.limit_abstraction import history_bucket, private_hand_bucket, public_board_bucket
from jorjor.limit_abstract_cfr_teacher import (
    LimitAbstractCfrPolicy,
    evaluate_abstract_cfr_against_random,
    train_limit_abstract_cfr,
)


FIXED_LIMIT_ACTION_SPACE = tuple(ACTIONS)
FORBIDDEN_STUDENT_INPUT_FIELDS = {
    "opponent_private_card",
    "opponent_rank",
    "hidden_label_for_probe_only",
    "teacher_abstraction_bucket",
    "teacher_posterior_or_range_proxy",
}


def build_limit_abstract_cfr_teacher_artifact(
    *,
    output_dir: Path,
    iteration_count: int = 10,
    deal_sample_count: int = 20,
    hand_count: int = 100,
    evaluation_hand_count: int = 100,
    seed: int = 20260603,
) -> dict[str, Any]:
    """Build the k.0.4 Limit abstract CFR teacher artifact smoke."""

    if iteration_count <= 0:
        raise ValueError("iteration_count must be positive")
    if deal_sample_count <= 0:
        raise ValueError("deal_sample_count must be positive")
    if hand_count <= 0:
        raise ValueError("hand_count must be positive")
    if evaluation_hand_count <= 0:
        raise ValueError("evaluation_hand_count must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_result = train_limit_abstract_cfr(
        iteration_count=iteration_count,
        deal_sample_count=deal_sample_count,
        seed=seed,
    )
    policy = LimitAbstractCfrPolicy(train_result.policy_table, seed=seed + 17)
    generator = _RecordingLimitAbstractCfrPolicy(policy=policy)
    points = generator.generate_points(hand_count=hand_count, seed=seed + 23)
    rows = [_teacher_artifact_row(point, distribution) for point, distribution in zip(points, generator.distributions, strict=True)]
    validation = validate_teacher_artifact_rows(rows)
    evaluation = evaluate_abstract_cfr_against_random(
        policy_table=train_result.policy_table,
        seed=seed,
        hand_count=evaluation_hand_count,
    )
    policy_record = {
        "version": "k.0.4",
        "teacher_family": "local_limit_abstract_cfr",
        "seed": seed,
        "iteration_count": iteration_count,
        "deal_sample_count": deal_sample_count,
        "information_set_count": train_result.information_set_count,
        "policy_table": train_result.policy_table,
    }
    policy_checksum = checksum_payload(policy_record)
    write_json_artifact(output_dir / "average_policy.json", {**policy_record, "checksum": policy_checksum})
    _write_jsonl(output_dir / "teacher_artifact_rows.jsonl", rows)
    payload = {
        "version": "k.0.4",
        "report_type": "k.0.4_limit_abstract_cfr_teacher_artifact",
        "claim_boundary": "teacher_artifact_only; does_not_prove_student_belief_tracking",
        "teacher_id": f"local_limit_abstract_cfr_seed_{seed}",
        "teacher_family": "local_limit_abstract_cfr",
        "environment": "limit_holdem_hu",
        "training_algorithm": "sampled_abstract_cfr",
        "training_iterations": iteration_count,
        "deal_sample_count": deal_sample_count,
        "seed": seed,
        "action_space": list(FIXED_LIMIT_ACTION_SPACE),
        "legal_action_mapping": {
            "no_outstanding_bet": ["check", "bet"],
            "outstanding_bet": ["fold", "call"],
        },
        "state_value_source": "rollout_terminal_return",
        "posterior_or_range_source": "not_available_for_teacher_artifact",
        "teacher_checkpoint_path": str(output_dir / "average_policy.json"),
        "dataset_path": str(output_dir / "teacher_artifact_rows.jsonl"),
        "policy_checksum": policy_checksum,
        "information_set_count": train_result.information_set_count,
        "decision_point_count": len(rows),
        "hand_count": hand_count,
        "evaluation_hand_count": evaluation_hand_count,
        "teacher_strength_proxy": evaluation.to_record(),
        "policy_entropy": _mean_entropy(rows),
        "validation": validation,
        "teacher_artifact_contract_passed": validation["passed"],
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "teacher_artifact_report.json", payload)
    return payload


def validate_teacher_artifact_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate k.0.4 teacher rows for distribution, legality, and input leakage."""

    illegal_action_count = 0
    distribution_sum_failure_count = 0
    distribution_negative_count = 0
    student_input_forbidden_field_count = 0
    missing_required_field_count = 0
    for row in rows:
        if "student_input" in row and _contains_forbidden_student_field(row["student_input"]):
            student_input_forbidden_field_count += 1
        required = {
            "student_input",
            "legal_actions",
            "action_space",
            "action_mask",
            "teacher_action_distribution",
            "sampled_action",
            "teacher_value",
            "hidden_label_for_probe_only",
            "teacher_abstraction_bucket",
            "history_order_key",
            "composition_key",
        }
        if not required.issubset(row):
            missing_required_field_count += 1
            continue
        if row["sampled_action"] not in row["legal_actions"]:
            illegal_action_count += 1
        distribution = row["teacher_action_distribution"]
        if set(distribution) != set(FIXED_LIMIT_ACTION_SPACE):
            distribution_sum_failure_count += 1
        values = [float(distribution.get(action, 0.0)) for action in FIXED_LIMIT_ACTION_SPACE]
        if any(value < 0.0 for value in values):
            distribution_negative_count += 1
        if abs(sum(values) - 1.0) > 1e-9:
            distribution_sum_failure_count += 1
    passed = (
        len(rows) > 0
        and illegal_action_count == 0
        and distribution_sum_failure_count == 0
        and distribution_negative_count == 0
        and student_input_forbidden_field_count == 0
        and missing_required_field_count == 0
    )
    return {
        "passed": passed,
        "row_count": len(rows),
        "illegal_action_count": illegal_action_count,
        "distribution_sum_failure_count": distribution_sum_failure_count,
        "distribution_negative_count": distribution_negative_count,
        "student_input_forbidden_field_count": student_input_forbidden_field_count,
        "missing_required_field_count": missing_required_field_count,
    }


class _RecordingLimitAbstractCfrPolicy:
    """Adapter that records action distributions used during generation."""

    def __init__(self, *, policy: LimitAbstractCfrPolicy) -> None:
        self._policy = policy
        self.distributions: list[dict[str, float]] = []

    def choose_action(self, state: LimitHoldemPolicyState) -> str:
        distribution = self._policy.action_distribution(state)
        self.distributions.append(_fixed_distribution(state.legal_actions, distribution))
        return self._policy.choose_action(state)

    def generate_points(self, *, hand_count: int, seed: int):
        from jorjor.envs.limit_holdem import LimitHoldemPolicyGenerator
        from jorjor.limit_random import flatten_decision_points

        generator = LimitHoldemPolicyGenerator(policy_by_player=(self, self), seed=seed)
        return flatten_decision_points(generator.generate_hands(hand_count))


def _teacher_artifact_row(point, distribution: dict[str, float]) -> dict[str, Any]:
    student_input = dict(point.visible_state)
    private_cards = tuple(student_input["private_cards"])
    public_cards = tuple(student_input["public_cards"])
    action_mask = {action: action in point.legal_actions for action in FIXED_LIMIT_ACTION_SPACE}
    return {
        "teacher_id": "local_limit_abstract_cfr",
        "environment": "limit_holdem_hu",
        "hand_id": point.hand_id,
        "step_index": point.step_index,
        "student_input": student_input,
        "information_state": {
            "acting_player": point.acting_player,
            "round": student_input["round"],
            "private_bucket": private_hand_bucket(private_cards),
            "public_bucket": public_board_bucket(public_cards),
            "history_bucket": history_bucket(point),
        },
        "public_state": {
            "round": student_input["round"],
            "public_cards": student_input["public_cards"],
            "pot": student_input["pot"],
            "betting_history": student_input["betting_history"],
        },
        "player_position": point.acting_player,
        "round_or_street": student_input["round"],
        "legal_actions": list(point.legal_actions),
        "action_space": list(FIXED_LIMIT_ACTION_SPACE),
        "action_mask": action_mask,
        "teacher_action_distribution": distribution,
        "sampled_action": point.action,
        "teacher_value": point.terminal_return,
        "hidden_label_for_probe_only": {
            "opponent_private_card": point.opponent_private_card,
            "opponent_rank": point.opponent_rank,
        },
        "private_label_schema": "opponent_private_card_and_169_rank",
        "teacher_abstraction_bucket": private_hand_bucket(private_cards),
        "teacher_posterior_or_range_proxy": None,
        "posterior_or_range_source": "not_available",
        "history_order_key": _history_order_key(student_input["betting_history"]),
        "composition_key": _composition_key(student_input["betting_history"], point.legal_actions, student_input["round"]),
        "composition_features_version": "limit_action_count_last_round_v1",
    }


def _fixed_distribution(legal_actions: tuple[str, ...], legal_distribution: dict[str, float]) -> dict[str, float]:
    values = {action: 0.0 for action in FIXED_LIMIT_ACTION_SPACE}
    for action in legal_actions:
        values[action] = float(legal_distribution.get(action, 0.0))
    total = sum(values.values())
    if total <= 0.0:
        probability = 1.0 / len(legal_actions)
        for action in legal_actions:
            values[action] = probability
    else:
        values = {action: value / total for action, value in values.items()}
    return values


def _history_order_key(history: list[dict[str, Any]]) -> str:
    tokens = []
    for event in history:
        if event["type"] == "action":
            tokens.append(f"{event['round']}:p{event['player']}:{event['action']}")
        elif event["type"] == "deal_public":
            tokens.append(f"deal:{event['round']}")
    return "|".join(tokens) if tokens else "empty"


def _composition_key(history: list[dict[str, Any]], legal_actions: tuple[str, ...], round_name: str) -> str:
    actions = [str(event["action"]) for event in history if event["type"] == "action"]
    counts = Counter(actions)
    last_action = actions[-1] if actions else "none"
    parts = [f"round={round_name}", f"last={last_action}", "legal=" + ",".join(legal_actions)]
    parts.extend(f"{action}={counts[action]}" for action in FIXED_LIMIT_ACTION_SPACE)
    return "|".join(parts)


def _contains_forbidden_student_field(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_STUDENT_INPUT_FIELDS:
                return True
            if _contains_forbidden_student_field(item):
                return True
    if isinstance(value, list):
        return any(_contains_forbidden_student_field(item) for item in value)
    return False


def _mean_entropy(rows: list[dict[str, Any]]) -> float:
    import math

    if not rows:
        return 0.0
    entropies = []
    for row in rows:
        entropy = 0.0
        for value in row["teacher_action_distribution"].values():
            probability = float(value)
            if probability > 0.0:
                entropy -= probability * math.log(probability)
        entropies.append(entropy)
    return sum(entropies) / len(entropies)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
