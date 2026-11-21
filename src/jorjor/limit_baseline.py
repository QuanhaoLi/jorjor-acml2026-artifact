from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import math
from pathlib import Path

from .artifacts import checksum_payload, write_json_artifact
from .config import load_limit_baseline_config
from .data.schema import DecisionPoint
from .data.validation import validate_decision_points
from .envs.limit_holdem import DECK, HAND_169_LABELS, hand_169_label
from .envs.limit_holdem import LimitHoldemRandomGenerator
from .limit_random import flatten_decision_points
from .limit_teacher import generate_teacher_points


@dataclass(frozen=True)
class LimitCardRemovalMetrics:
    """Aggregate metrics for 169-class Limit card-removal baseline."""

    count: int
    top1_accuracy: float
    top5_accuracy: float
    top10_accuracy: float
    cross_entropy: float
    average_candidate_combos: float
    average_candidate_labels: float

    def to_record(self) -> dict[str, float | int]:
        """Serialize baseline metrics."""

        return {
            "count": self.count,
            "top1_accuracy": self.top1_accuracy,
            "top5_accuracy": self.top5_accuracy,
            "top10_accuracy": self.top10_accuracy,
            "cross_entropy": self.cross_entropy,
            "average_candidate_combos": self.average_candidate_combos,
            "average_candidate_labels": self.average_candidate_labels,
        }


@dataclass(frozen=True)
class LimitBaselineSmokeResult:
    """Result summary for v0.4.4 Limit baseline smoke."""

    hand_count: int
    decision_point_count: int
    illegal_action_count: int
    leakage_count: int
    top1_accuracy: float
    top5_accuracy: float
    top10_accuracy: float
    cross_entropy: float
    checksum: str
    artifact_dir: Path


def possible_opponent_hands(point: DecisionPoint) -> frozenset[frozenset[str]]:
    """Return legal opponent two-card combos consistent with visible cards."""

    state = point.visible_state
    blocked = set(state["private_cards"]) | set(state["public_cards"])
    candidates = []
    for left, right in combinations(DECK, 2):
        if left not in blocked and right not in blocked:
            candidates.append(frozenset((left, right)))
    return frozenset(candidates)


def label_probabilities(point: DecisionPoint) -> dict[str, float]:
    """Return 169-class probabilities induced by uniform card-removal combos."""

    candidates = possible_opponent_hands(point)
    if not candidates:
        raise ValueError("no possible opponent hands")

    counts: Counter[str] = Counter()
    for candidate in candidates:
        cards = tuple(candidate)
        counts[hand_169_label((cards[0], cards[1]))] += 1

    total = len(candidates)
    return {label: count / total for label, count in counts.items()}


def evaluate_limit_card_removal_baseline(points: tuple[DecisionPoint, ...]) -> LimitCardRemovalMetrics:
    """Evaluate the visible card-removal baseline for Limit opponent-hand labels."""

    if not points:
        raise ValueError("cannot evaluate baseline with zero points")

    top1 = 0
    top5 = 0
    top10 = 0
    total_nll = 0.0
    total_candidate_combos = 0
    total_candidate_labels = 0

    for point in points:
        probabilities = label_probabilities(point)
        true_label = point.opponent_rank
        probability = probabilities.get(true_label, 0.0)
        if probability <= 0.0:
            raise ValueError("true opponent hand is impossible under visible information")

        ranked_labels = sorted(HAND_169_LABELS, key=lambda label: (-probabilities.get(label, 0.0), label))
        if true_label == ranked_labels[0]:
            top1 += 1
        if true_label in ranked_labels[:5]:
            top5 += 1
        if true_label in ranked_labels[:10]:
            top10 += 1
        total_nll += -math.log(probability)
        total_candidate_combos += len(possible_opponent_hands(point))
        total_candidate_labels += len(probabilities)

    count = len(points)
    return LimitCardRemovalMetrics(
        count=count,
        top1_accuracy=top1 / count,
        top5_accuracy=top5 / count,
        top10_accuracy=top10 / count,
        cross_entropy=total_nll / count,
        average_candidate_combos=total_candidate_combos / count,
        average_candidate_labels=total_candidate_labels / count,
    )


def run_limit_baseline_smoke(
    config_path: Path,
    artifact_dir: Path | None = None,
) -> LimitBaselineSmokeResult:
    """Generate Limit data and evaluate the 169-way card-removal baseline."""

    config = load_limit_baseline_config(config_path)
    output_dir = artifact_dir if artifact_dir is not None else config.artifact_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.data_source == "random":
        generator = LimitHoldemRandomGenerator(seed=config.seed)
        points = flatten_decision_points(generator.generate_hands(config.hand_count))
    elif config.data_source == "teacher":
        points = generate_teacher_points(hand_count=config.hand_count, seed=config.seed)
    else:
        raise ValueError(f"unknown data source: {config.data_source}")

    report = validate_decision_points(points)
    baseline = evaluate_limit_card_removal_baseline(points)
    record = {
        "version": config.run_version,
        "data_source": config.data_source,
        "seed": config.seed,
        "hand_count": report.hand_count,
        "decision_point_count": report.decision_point_count,
        "illegal_action_count": report.illegal_action_count,
        "leakage_count": report.leakage_count,
        "baseline": baseline.to_record(),
    }
    checksum = checksum_payload(record)
    write_json_artifact(output_dir / "baseline_report.json", {**record, "checksum": checksum})

    return LimitBaselineSmokeResult(
        hand_count=report.hand_count,
        decision_point_count=report.decision_point_count,
        illegal_action_count=report.illegal_action_count,
        leakage_count=report.leakage_count,
        top1_accuracy=baseline.top1_accuracy,
        top5_accuracy=baseline.top5_accuracy,
        top10_accuracy=baseline.top10_accuracy,
        cross_entropy=baseline.cross_entropy,
        checksum=checksum,
        artifact_dir=output_dir,
    )
