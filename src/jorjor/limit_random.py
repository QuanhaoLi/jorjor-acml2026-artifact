from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .artifacts import checksum_payload, write_json_artifact
from .config import load_limit_random_config
from .data.schema import DecisionPoint
from .data.validation import validate_decision_points
from .data.writer import read_decision_points_jsonl, write_decision_points_jsonl
from .envs.limit_holdem import HAND_169_LABELS, HAND_169_LABEL_SET, LimitHoldemHand, LimitHoldemRandomGenerator


@dataclass(frozen=True)
class LimitRandomSmokeResult:
    """Result summary returned by the v0.3.1 Limit random smoke command."""

    seed: int
    hand_count: int
    decision_point_count: int
    illegal_action_count: int
    leakage_count: int
    invalid_opponent_label_count: int
    hand_label_class_count: int
    checksum: str
    artifact_dir: Path


def flatten_decision_points(hands: Iterable[LimitHoldemHand]) -> tuple[DecisionPoint, ...]:
    """Flatten generated Limit Hold'em hands into decision points."""

    points: list[DecisionPoint] = []
    for hand in hands:
        points.extend(hand.decision_points)
    return tuple(points)


def count_invalid_opponent_labels(points: tuple[DecisionPoint, ...]) -> int:
    """Count opponent-hand labels that are outside the 169-class vocabulary."""

    return sum(1 for point in points if point.opponent_rank not in HAND_169_LABEL_SET)


def run_limit_random_smoke(
    config_path: Path,
    artifact_dir: Path | None = None,
) -> LimitRandomSmokeResult:
    """Generate and validate a deterministic Limit random dataset."""

    config = load_limit_random_config(config_path)
    output_dir = artifact_dir if artifact_dir is not None else config.artifact_dir
    generator = LimitHoldemRandomGenerator(seed=config.seed)
    hands = generator.generate_hands(config.hand_count)
    points = flatten_decision_points(hands)
    report = validate_decision_points(points)
    invalid_label_count = count_invalid_opponent_labels(points)

    dataset_path = output_dir / "decision_points.jsonl"
    report_path = output_dir / "validation_report.json"
    write_decision_points_jsonl(dataset_path, points)
    read_back_count = len(read_decision_points_jsonl(dataset_path))

    report_record = {
        "version": config.run_version,
        "seed": config.seed,
        "hand_count": report.hand_count,
        "decision_point_count": report.decision_point_count,
        "illegal_action_count": report.illegal_action_count,
        "leakage_count": report.leakage_count,
        "invalid_opponent_label_count": invalid_label_count,
        "hand_label_class_count": len(HAND_169_LABELS),
        "observed_opponent_label_count": len({point.opponent_rank for point in points}),
        "read_back_count": read_back_count,
        "read_back_matches": read_back_count == report.decision_point_count,
    }
    checksum = checksum_payload(report_record)
    write_json_artifact(report_path, {**report_record, "checksum": checksum})

    return LimitRandomSmokeResult(
        seed=config.seed,
        hand_count=report.hand_count,
        decision_point_count=report.decision_point_count,
        illegal_action_count=report.illegal_action_count,
        leakage_count=report.leakage_count,
        invalid_opponent_label_count=invalid_label_count,
        hand_label_class_count=len(HAND_169_LABELS),
        checksum=checksum,
        artifact_dir=output_dir,
    )
