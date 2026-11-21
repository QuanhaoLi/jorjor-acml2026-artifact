from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .artifacts import checksum_payload, write_json_artifact
from .config import load_leduc_random_baseline_config
from .data.baseline import evaluate_card_removal_baseline
from .data.split import (
    SplitFractions,
    apply_split,
    build_split_manifest,
    has_split_overlap,
)
from .data.validation import validate_decision_points
from .data.writer import read_decision_points_jsonl, write_decision_points_jsonl
from .envs.leduc import LeducRandomGenerator
from .leduc_random import flatten_decision_points


@dataclass(frozen=True)
class LeducRandomBaselineResult:
    """Result summary returned by the v0.1.1 baseline command."""

    seed: int
    hand_count: int
    decision_point_count: int
    illegal_action_count: int
    leakage_count: int
    split_overlap: bool
    train_count: int
    validation_count: int
    test_count: int
    baseline_accuracy: float
    baseline_cross_entropy: float
    checksum: str
    artifact_dir: Path


def run_leduc_random_baseline(
    config_path: Path,
    artifact_dir: Path | None = None,
) -> LeducRandomBaselineResult:
    """Generate, read, split, validate, and baseline random Leduc data."""

    config = load_leduc_random_baseline_config(config_path)
    output_dir = artifact_dir if artifact_dir is not None else config.artifact_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = LeducRandomGenerator(seed=config.seed)
    points = flatten_decision_points(generator.generate_hands(config.hand_count))
    dataset_path = output_dir / "decision_points.jsonl"
    write_decision_points_jsonl(dataset_path, points)
    read_points = read_decision_points_jsonl(dataset_path)
    report = validate_decision_points(read_points)

    manifest = build_split_manifest(
        points=read_points,
        fractions=SplitFractions(
            train=config.train_fraction,
            validation=config.validation_fraction,
            test=config.test_fraction,
        ),
        seed=config.split_seed,
    )
    splits = apply_split(read_points, manifest)
    baseline = evaluate_card_removal_baseline(splits.test)
    split_overlap = has_split_overlap(manifest)

    report_record = {
        "version": "0.1.1",
        "seed": config.seed,
        "hand_count": report.hand_count,
        "decision_point_count": report.decision_point_count,
        "illegal_action_count": report.illegal_action_count,
        "leakage_count": report.leakage_count,
        "split_seed": config.split_seed,
        "split_overlap": split_overlap,
        "split_counts": {
            "train_points": len(splits.train),
            "validation_points": len(splits.validation),
            "test_points": len(splits.test),
        },
        "baseline": baseline.to_record(),
    }
    checksum = checksum_payload(report_record)
    write_json_artifact(output_dir / "split_manifest.json", manifest.to_record())
    write_json_artifact(output_dir / "baseline_report.json", {**report_record, "checksum": checksum})

    return LeducRandomBaselineResult(
        seed=config.seed,
        hand_count=report.hand_count,
        decision_point_count=report.decision_point_count,
        illegal_action_count=report.illegal_action_count,
        leakage_count=report.leakage_count,
        split_overlap=split_overlap,
        train_count=len(splits.train),
        validation_count=len(splits.validation),
        test_count=len(splits.test),
        baseline_accuracy=baseline.accuracy,
        baseline_cross_entropy=baseline.cross_entropy,
        checksum=checksum,
        artifact_dir=output_dir,
    )
