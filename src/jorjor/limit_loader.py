from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .artifacts import checksum_payload, write_json_artifact
from .config import load_limit_loader_config
from .data.split import SplitFractions, apply_split, build_split_manifest, has_split_overlap
from .data.validation import validate_decision_points
from .envs.limit_holdem import LimitHoldemRandomGenerator
from .limit_random import flatten_decision_points
from .tokenizer import LimitTokenizer


@dataclass(frozen=True)
class LimitLoaderSmokeResult:
    """Result summary for v0.3.2 Limit tokenizer/loader smoke."""

    hand_count: int
    decision_point_count: int
    train_rows: int
    validation_rows: int
    test_rows: int
    vocab_size: int
    action_size: int
    opponent_hand_size: int
    split_overlap: bool
    checksum: str
    artifact_dir: Path


def run_limit_loader_smoke(
    config_path: Path,
    artifact_dir: Path | None = None,
) -> LimitLoaderSmokeResult:
    """Generate, split, and tokenize Limit random data."""

    config = load_limit_loader_config(config_path)
    output_dir = artifact_dir if artifact_dir is not None else config.artifact_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = LimitHoldemRandomGenerator(seed=config.seed)
    points = flatten_decision_points(generator.generate_hands(config.hand_count))
    validation_report = validate_decision_points(points)
    manifest = build_split_manifest(
        points=points,
        fractions=SplitFractions(
            train=config.train_fraction,
            validation=config.validation_fraction,
            test=config.test_fraction,
        ),
        seed=config.split_seed,
    )
    splits = apply_split(points, manifest)
    tokenizer = LimitTokenizer()
    train_features = [tokenizer.encode(point) for point in splits.train]
    validation_features = [tokenizer.encode(point) for point in splits.validation]
    test_features = [tokenizer.encode(point) for point in splits.test]

    record = {
        "version": config.run_version,
        "seed": config.seed,
        "split_seed": config.split_seed,
        "hand_count": validation_report.hand_count,
        "decision_point_count": validation_report.decision_point_count,
        "illegal_action_count": validation_report.illegal_action_count,
        "leakage_count": validation_report.leakage_count,
        "train_rows": len(train_features),
        "validation_rows": len(validation_features),
        "test_rows": len(test_features),
        "vocab_size": tokenizer.vocab_size,
        "action_size": tokenizer.action_size,
        "opponent_hand_size": tokenizer.opponent_card_size,
        "split_overlap": has_split_overlap(manifest),
        "max_input_length": max(len(feature.input_ids) for feature in train_features + validation_features + test_features),
        "split_manifest": manifest.to_record(),
    }
    checksum = checksum_payload(record)
    write_json_artifact(output_dir / "loader_report.json", {**record, "checksum": checksum})

    return LimitLoaderSmokeResult(
        hand_count=validation_report.hand_count,
        decision_point_count=validation_report.decision_point_count,
        train_rows=len(train_features),
        validation_rows=len(validation_features),
        test_rows=len(test_features),
        vocab_size=tokenizer.vocab_size,
        action_size=tokenizer.action_size,
        opponent_hand_size=tokenizer.opponent_card_size,
        split_overlap=has_split_overlap(manifest),
        checksum=checksum,
        artifact_dir=output_dir,
    )
