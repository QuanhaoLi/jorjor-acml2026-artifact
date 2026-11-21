from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from pathlib import Path

from .artifacts import checksum_payload, write_json_artifact
from .config import load_limit_prior_baseline_config
from .data.schema import DecisionPoint
from .data.split import SplitFractions, apply_split, build_split_manifest
from .data.validation import validate_decision_points
from .envs.limit_holdem import HAND_169_LABELS, LimitHoldemRandomGenerator
from .limit_random import flatten_decision_points
from .limit_teacher import generate_teacher_points


@dataclass(frozen=True)
class PriorBaselineMetrics:
    """Aggregate metrics for one empirical prior grouping."""

    name: str
    count: int
    top1_accuracy: float
    top5_accuracy: float
    top10_accuracy: float
    cross_entropy: float
    fallback_count: int
    group_count: int

    def to_record(self) -> dict[str, float | int | str]:
        """Serialize empirical prior metrics."""

        return {
            "name": self.name,
            "count": self.count,
            "top1_accuracy": self.top1_accuracy,
            "top5_accuracy": self.top5_accuracy,
            "top10_accuracy": self.top10_accuracy,
            "cross_entropy": self.cross_entropy,
            "fallback_count": self.fallback_count,
            "group_count": self.group_count,
        }


@dataclass(frozen=True)
class LimitPriorBaselineResult:
    """Result summary for v0.4.12 Limit empirical prior baselines."""

    hand_count: int
    decision_point_count: int
    illegal_action_count: int
    leakage_count: int
    baselines: dict[str, PriorBaselineMetrics]
    checksum: str
    artifact_dir: Path


def _state_value(point: DecisionPoint, key: str) -> str:
    """Read a visible-state value as a stable grouping token."""

    value = point.visible_state.get(key)
    if value is None:
        return "missing"
    return str(value)


def _group_key(point: DecisionPoint, group_name: str) -> tuple[str, ...]:
    """Return the empirical-prior group key for a decision point."""

    if group_name == "global":
        return ("global",)
    if group_name == "round_position":
        return (
            _state_value(point, "round"),
            _state_value(point, "acting_player"),
        )
    if group_name == "legal_actions":
        return tuple(sorted(point.legal_actions))
    if group_name == "action":
        return (point.action,)
    if group_name == "action_round_position":
        return (
            point.action,
            _state_value(point, "round"),
            _state_value(point, "acting_player"),
        )
    raise ValueError(f"unknown prior group: {group_name}")


def _smoothed_probabilities(counts: Counter[str]) -> dict[str, float]:
    """Return Laplace-smoothed probabilities over the 169 hand labels."""

    total = sum(counts.values()) + len(HAND_169_LABELS)
    return {label: (counts[label] + 1) / total for label in HAND_169_LABELS}


def evaluate_grouped_prior(
    train_points: tuple[DecisionPoint, ...],
    test_points: tuple[DecisionPoint, ...],
    group_name: str,
) -> PriorBaselineMetrics:
    """Fit an empirical train prior and evaluate it on held-out points."""

    if not train_points:
        raise ValueError("cannot fit prior baseline with zero train points")
    if not test_points:
        raise ValueError("cannot evaluate prior baseline with zero test points")

    global_counts: Counter[str] = Counter(point.opponent_rank for point in train_points)
    grouped_counts: defaultdict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for point in train_points:
        grouped_counts[_group_key(point, group_name)][point.opponent_rank] += 1

    top1 = 0
    top5 = 0
    top10 = 0
    fallback_count = 0
    total_nll = 0.0
    for point in test_points:
        key = _group_key(point, group_name)
        counts = grouped_counts.get(key)
        if counts is None:
            counts = global_counts
            fallback_count += 1
        probabilities = _smoothed_probabilities(counts)
        ranked_labels = sorted(HAND_169_LABELS, key=lambda label: (-probabilities[label], label))
        true_label = point.opponent_rank
        if true_label == ranked_labels[0]:
            top1 += 1
        if true_label in ranked_labels[:5]:
            top5 += 1
        if true_label in ranked_labels[:10]:
            top10 += 1
        total_nll += -math.log(probabilities[true_label])

    count = len(test_points)
    return PriorBaselineMetrics(
        name=group_name,
        count=count,
        top1_accuracy=top1 / count,
        top5_accuracy=top5 / count,
        top10_accuracy=top10 / count,
        cross_entropy=total_nll / count,
        fallback_count=fallback_count,
        group_count=len(grouped_counts),
    )


def _generate_points(data_source: str, hand_count: int, seed: int) -> tuple[DecisionPoint, ...]:
    """Generate decision points for a configured Limit data source."""

    if data_source == "random":
        generator = LimitHoldemRandomGenerator(seed=seed)
        return flatten_decision_points(generator.generate_hands(hand_count))
    if data_source == "teacher":
        return generate_teacher_points(hand_count=hand_count, seed=seed)
    raise ValueError(f"unknown data source: {data_source}")


def run_limit_prior_baseline(
    config_path: Path,
    artifact_dir: Path | None = None,
    hand_count_override: int | None = None,
) -> LimitPriorBaselineResult:
    """Run v0.4.12 empirical prior baselines and write a report artifact."""

    config = load_limit_prior_baseline_config(config_path)
    hand_count = hand_count_override if hand_count_override is not None else config.hand_count
    if hand_count <= 0:
        raise ValueError("hand_count_override must be positive")

    output_dir = artifact_dir if artifact_dir is not None else config.artifact_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    points = _generate_points(data_source=config.data_source, hand_count=hand_count, seed=config.seed)
    report = validate_decision_points(points)
    fractions = SplitFractions(
        train=config.train_fraction,
        validation=config.validation_fraction,
        test=config.test_fraction,
    )
    split_manifest = build_split_manifest(points=points, fractions=fractions, seed=config.split_seed)
    split_data = apply_split(points=points, manifest=split_manifest)
    train_points = split_data.train + split_data.validation
    test_points = split_data.test

    baseline_names = (
        "global",
        "round_position",
        "legal_actions",
        "action",
        "action_round_position",
    )
    baselines = {
        name: evaluate_grouped_prior(train_points=train_points, test_points=test_points, group_name=name)
        for name in baseline_names
    }

    split_record = split_manifest.to_record()
    record = {
        "version": config.run_version,
        "data_source": config.data_source,
        "seed": config.seed,
        "split_seed": config.split_seed,
        "hand_count": report.hand_count,
        "decision_point_count": report.decision_point_count,
        "illegal_action_count": report.illegal_action_count,
        "leakage_count": report.leakage_count,
        "split": split_record,
        "train_decision_points": len(train_points),
        "test_decision_points": len(test_points),
        "baselines": {name: metrics.to_record() for name, metrics in baselines.items()},
    }
    checksum = checksum_payload(record)
    write_json_artifact(output_dir / "prior_report.json", {**record, "checksum": checksum})

    return LimitPriorBaselineResult(
        hand_count=report.hand_count,
        decision_point_count=report.decision_point_count,
        illegal_action_count=report.illegal_action_count,
        leakage_count=report.leakage_count,
        baselines=baselines,
        checksum=checksum,
        artifact_dir=output_dir,
    )
