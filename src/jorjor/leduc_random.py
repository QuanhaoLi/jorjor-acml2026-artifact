from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .artifacts import checksum_payload, write_json_artifact
from .config import load_leduc_random_config
from .data.schema import DecisionPoint
from .data.validation import validate_decision_points
from .data.writer import write_decision_points_jsonl
from .envs.leduc import LeducHand, LeducRandomGenerator


@dataclass(frozen=True)
class LeducRandomSmokeResult:
    """Result summary returned by the v0.1.0 Leduc random smoke command."""

    seed: int
    hand_count: int
    decision_point_count: int
    illegal_action_count: int
    leakage_count: int
    checksum: str
    artifact_dir: Path


def flatten_decision_points(hands: Iterable[LeducHand]) -> tuple[DecisionPoint, ...]:
    """Flatten generated hands into decision points."""

    points: list[DecisionPoint] = []
    for hand in hands:
        points.extend(hand.decision_points)
    return tuple(points)


def run_leduc_random_smoke(
    config_path: Path,
    artifact_dir: Path | None = None,
) -> LeducRandomSmokeResult:
    """Generate and validate a small deterministic Leduc random dataset."""

    config = load_leduc_random_config(config_path)
    output_dir = artifact_dir if artifact_dir is not None else config.artifact_dir
    generator = LeducRandomGenerator(seed=config.seed)
    hands = generator.generate_hands(config.hand_count)
    points = flatten_decision_points(hands)
    report = validate_decision_points(points)

    dataset_path = output_dir / "decision_points.jsonl"
    report_path = output_dir / "validation_report.json"
    write_decision_points_jsonl(dataset_path, points)

    report_record = {
        "version": "0.1.0",
        "seed": config.seed,
        "hand_count": report.hand_count,
        "decision_point_count": report.decision_point_count,
        "illegal_action_count": report.illegal_action_count,
        "leakage_count": report.leakage_count,
    }
    checksum = checksum_payload(report_record)
    write_json_artifact(report_path, {**report_record, "checksum": checksum})

    return LeducRandomSmokeResult(
        seed=config.seed,
        hand_count=report.hand_count,
        decision_point_count=report.decision_point_count,
        illegal_action_count=report.illegal_action_count,
        leakage_count=report.leakage_count,
        checksum=checksum,
        artifact_dir=output_dir,
    )
