from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, product
import math
import random
from typing import Any, Callable

from jorjor.probe.counterfactual import jensen_shannon


HIDDEN_STATES = ("s0", "s1", "s2", "s3")
OBSERVATIONS = ("o0", "o1", "o2", "o3")
PRIOR = tuple(1.0 / len(HIDDEN_STATES) for _ in HIDDEN_STATES)
BEHAVIOR_ONLY_BELIEF_SUPPORTED = "behavior_only_belief_like_supported"
COMPOSITION_SHORTCUT_SUPPORTED = "composition_shortcut_supported"
INCONCLUSIVE = "inconclusive"

# Full-rank reward matrix with different optimal actions in matched-order buckets.
REWARD_MATRIX = (
    (1.25, -0.50, 0.10, -0.65),
    (-0.60, 1.15, -0.35, 0.05),
    (0.00, -0.45, 1.30, -0.50),
    (-0.35, 0.20, -0.60, 1.20),
)


@dataclass(frozen=True)
class BeliefRequiredSequence:
    """One generated k.4 hidden-state sequence."""

    sequence_id: str
    hidden_state: str
    observations: tuple[str, ...]


def emission_probability(
    *,
    hidden_state: str,
    observation: str,
    position_index: int,
    signal_probability: float = 0.76,
) -> float:
    """Return p(observation | hidden_state, position)."""

    if hidden_state not in HIDDEN_STATES:
        raise ValueError(f"unknown hidden_state: {hidden_state}")
    if observation not in OBSERVATIONS:
        raise ValueError(f"unknown observation: {observation}")
    if not (0.0 < signal_probability < 1.0):
        raise ValueError("signal_probability must be between 0 and 1")
    state_index = HIDDEN_STATES.index(hidden_state)
    observation_index = OBSERVATIONS.index(observation)
    preferred = (state_index + position_index) % len(OBSERVATIONS)
    if observation_index == preferred:
        return signal_probability
    return (1.0 - signal_probability) / (len(OBSERVATIONS) - 1)


def exact_posterior(
    observations: tuple[str, ...],
    *,
    signal_probability: float = 0.76,
) -> tuple[float, ...]:
    """Return exact posterior p(z | ordered observations)."""

    if not observations:
        return PRIOR
    weights = []
    for hidden_state, prior in zip(HIDDEN_STATES, PRIOR, strict=True):
        likelihood = prior
        for position_index, observation in enumerate(observations):
            likelihood *= emission_probability(
                hidden_state=hidden_state,
                observation=observation,
                position_index=position_index,
                signal_probability=signal_probability,
            )
        weights.append(likelihood)
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("posterior normalizer is zero")
    return tuple(value / total for value in weights)


def action_value_targets(
    observations: tuple[str, ...],
    *,
    signal_probability: float = 0.76,
) -> dict[str, Any]:
    """Return Bayes-optimal behavior targets from the exact posterior."""

    posterior = exact_posterior(observations, signal_probability=signal_probability)
    action_values = []
    for rewards in REWARD_MATRIX:
        action_values.append(sum(probability * reward for probability, reward in zip(posterior, rewards, strict=True)))
    action_label = max(range(len(action_values)), key=lambda index: action_values[index])
    return {
        "action_label": int(action_label),
        "action_values": tuple(float(value) for value in action_values),
        "q_values": tuple(float(value) for value in action_values),
        "value_label": float(action_values[action_label]),
    }


def generate_sequences(
    *,
    sequence_count: int,
    length: int,
    seed: int,
    signal_probability: float = 0.76,
) -> list[BeliefRequiredSequence]:
    """Generate k.4 POMDP sequences."""

    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")
    if length <= 1:
        raise ValueError("length must be greater than 1")
    rng = random.Random(seed)
    rows = []
    for index in range(sequence_count):
        hidden_state = rng.choice(HIDDEN_STATES)
        observations = tuple(
            _sample_observation(
                hidden_state=hidden_state,
                position_index=position_index,
                rng=rng,
                signal_probability=signal_probability,
            )
            for position_index in range(length)
        )
        rows.append(BeliefRequiredSequence(f"belief-required-{index:06d}", hidden_state, observations))
    return rows


def all_prefixes(length: int) -> list[tuple[str, ...]]:
    """Return all non-empty observation prefixes up to length."""

    if length <= 1:
        raise ValueError("length must be greater than 1")
    prefixes: list[tuple[str, ...]] = []
    for prefix_len in range(1, length + 1):
        prefixes.extend(product(OBSERVATIONS, repeat=prefix_len))
    return prefixes


def composition_key(observations: tuple[str, ...]) -> str:
    """Return visible composition bucket key."""

    if not observations:
        return "len=0|last=none|" + "|".join(f"{observation}=0" for observation in OBSERVATIONS)
    counts = Counter(observations)
    parts = [f"len={len(observations)}", f"last={observations[-1]}"]
    parts.extend(f"{observation}={counts[observation]}" for observation in OBSERVATIONS)
    return "|".join(parts)


def order_key(observations: tuple[str, ...]) -> str:
    """Return ordered observation key."""

    return ",".join(observations) if observations else "empty"


def composition_state(observations: tuple[str, ...]) -> tuple[float, ...]:
    """Return a normalized visible-composition shortcut state."""

    if not observations:
        return PRIOR
    counts = Counter(observations)
    total = float(len(observations))
    count_features = tuple(counts[observation] / total for observation in OBSERVATIONS)
    last_features = tuple(1.0 if observations[-1] == observation else 0.0 for observation in OBSERVATIONS)
    return normalize_distribution(
        tuple(0.75 * count + 0.25 * last for count, last in zip(count_features, last_features, strict=True))
    )


def matched_order_rows(
    *,
    length: int,
    representation_fn: Callable[[tuple[str, ...]], tuple[float, ...]],
    signal_probability: float = 0.76,
    max_order_pairs: int = 4096,
) -> list[dict[str, Any]]:
    """Return same-composition/different-order rows."""

    groups: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for observations in product(OBSERVATIONS, repeat=length):
        groups[composition_key(observations)].append(observations)
    rows = []
    for bucket, candidates in sorted(groups.items()):
        if len(candidates) < 2:
            continue
        for left, right in combinations(candidates, 2):
            posterior_js = jensen_shannon(
                exact_posterior(left, signal_probability=signal_probability),
                exact_posterior(right, signal_probability=signal_probability),
            )
            if posterior_js <= 1e-12:
                continue
            left_targets = action_value_targets(left, signal_probability=signal_probability)
            right_targets = action_value_targets(right, signal_probability=signal_probability)
            rows.append(
                {
                    "composition_key": bucket,
                    "left_order_key": order_key(left),
                    "right_order_key": order_key(right),
                    "left_observations": list(left),
                    "right_observations": list(right),
                    "posterior_js": posterior_js,
                    "representation_js": jensen_shannon(representation_fn(left), representation_fn(right)),
                    "behavior_disagrees": left_targets["action_label"] != right_targets["action_label"],
                    "left_action": left_targets["action_label"],
                    "right_action": right_targets["action_label"],
                }
            )
            if len(rows) >= max_order_pairs:
                return rows
    return rows


def matched_order_summary(
    rows: list[dict[str, Any]],
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Summarize matched-order rows using bucket-bootstrap intervals."""

    posterior_js = [float(row["posterior_js"]) for row in rows]
    representation_js = [float(row["representation_js"]) for row in rows]
    bucket_summary = _bucket_summary(rows, bucket_key="composition_key") if rows else None
    return {
        "pair_count": len(rows),
        "matched_bucket_count": len({str(row["composition_key"]) for row in rows}),
        "estimand": "unweighted_composition_bucket_mean",
        "mean_posterior_js": _mean(bucket_summary["posterior_means"] if bucket_summary else posterior_js),
        "mean_representation_js": _mean(
            bucket_summary["representation_means"] if bucket_summary else representation_js
        ),
        "pair_weighted_mean_posterior_js": _mean(posterior_js),
        "pair_weighted_mean_representation_js": _mean(representation_js),
        "posterior_representation_correlation": _pearson(posterior_js, representation_js),
        "behavior_disagreement_fraction": _mean([1.0 if row["behavior_disagrees"] else 0.0 for row in rows]),
        "representation_js_ci": _bucket_bootstrap_interval(
            rows,
            value_key="representation_js",
            bucket_key="composition_key",
            bootstrap_count=bootstrap_count,
            bootstrap_seed=bootstrap_seed,
            confidence_level=confidence_level,
        )
        if rows
        else None,
    }


def transition_summary(
    sequences: list[BeliefRequiredSequence],
    *,
    representation_fn: Callable[[tuple[str, ...]], tuple[float, ...]],
    signal_probability: float = 0.76,
    min_posterior_js: float = 0.0,
) -> dict[str, Any]:
    """Compare prefix posterior updates with representation updates."""

    posterior_js = []
    representation_js = []
    skipped_transition_count = 0
    for sequence in sequences:
        prefixes = [sequence.observations[:prefix_len] for prefix_len in range(1, len(sequence.observations) + 1)]
        for left, right in zip(prefixes, prefixes[1:], strict=False):
            posterior_delta = jensen_shannon(
                exact_posterior(left, signal_probability=signal_probability),
                exact_posterior(right, signal_probability=signal_probability),
            )
            if posterior_delta < min_posterior_js:
                skipped_transition_count += 1
                continue
            posterior_js.append(posterior_delta)
            representation_js.append(jensen_shannon(representation_fn(left), representation_fn(right)))
    return {
        "transition_count": len(posterior_js),
        "skipped_transition_count": skipped_transition_count,
        "min_posterior_js": min_posterior_js,
        "mean_posterior_js": _mean(posterior_js),
        "mean_representation_js": _mean(representation_js),
        "posterior_representation_correlation": _pearson(posterior_js, representation_js),
    }


def build_preflight_report(
    *,
    length: int,
    signal_probability: float = 0.76,
    max_order_pairs: int = 4096,
    disagreement_threshold: float = 0.25,
) -> dict[str, Any]:
    """Return the task-validity preflight report."""

    rows = matched_order_rows(
        length=length,
        representation_fn=exact_posterior,
        signal_probability=signal_probability,
        max_order_pairs=max_order_pairs,
    )
    grouped_labels: dict[str, list[int]] = defaultdict(list)
    for prefix in product(OBSERVATIONS, repeat=length):
        grouped_labels[composition_key(prefix)].append(
            int(action_value_targets(prefix, signal_probability=signal_probability)["action_label"])
        )
    total = 0
    correct = 0
    for labels in grouped_labels.values():
        counts = Counter(labels)
        total += len(labels)
        correct += max(counts.values())
    composition_action_accuracy_upper_bound = correct / total if total else 0.0
    behavior_disagreement_fraction = _mean([1.0 if row["behavior_disagrees"] else 0.0 for row in rows])
    return {
        "length": length,
        "signal_probability": signal_probability,
        "matched_pair_count": len(rows),
        "matched_bucket_count": len({str(row["composition_key"]) for row in rows}),
        "mean_posterior_js": _mean([float(row["posterior_js"]) for row in rows]),
        "behavior_disagreement_fraction": behavior_disagreement_fraction,
        "behavior_disagreement_threshold": disagreement_threshold,
        "composition_action_accuracy_upper_bound": composition_action_accuracy_upper_bound,
        "belief_required": bool(
            rows
            and behavior_disagreement_fraction >= disagreement_threshold
            and composition_action_accuracy_upper_bound < 0.90
        ),
    }


def normalize_distribution(values: tuple[float, ...]) -> tuple[float, ...]:
    """Return non-negative normalized values."""

    clipped = tuple(max(0.0, float(value)) for value in values)
    total = sum(clipped)
    if total <= 0.0:
        return tuple(1.0 / len(values) for _ in values)
    return tuple(value / total for value in clipped)


def _sample_observation(
    *,
    hidden_state: str,
    position_index: int,
    rng: random.Random,
    signal_probability: float,
) -> str:
    threshold = rng.random()
    cumulative = 0.0
    for observation in OBSERVATIONS:
        cumulative += emission_probability(
            hidden_state=hidden_state,
            observation=observation,
            position_index=position_index,
            signal_probability=signal_probability,
        )
        if threshold <= cumulative:
            return observation
    return OBSERVATIONS[-1]


def _bucket_bootstrap_interval(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    bucket_key: str,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, float]:
    if bootstrap_count <= 0:
        raise ValueError("bootstrap_count must be positive")
    bucket_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        bucket_values[str(row[bucket_key])].append(float(row[value_key]))
    if not bucket_values:
        raise ValueError("bootstrap requires at least one bucket")
    bucket_means = [sum(values) / len(values) for values in bucket_values.values()]
    rng = random.Random(bootstrap_seed)
    means = []
    for _ in range(bootstrap_count):
        means.append(sum(bucket_means[rng.randrange(len(bucket_means))] for _ in bucket_means) / len(bucket_means))
    means.sort()
    lower_index = int(((1.0 - confidence_level) / 2.0) * (bootstrap_count - 1))
    upper_index = int((1.0 - (1.0 - confidence_level) / 2.0) * (bootstrap_count - 1))
    return {
        "bootstrap_mean": sum(bucket_means) / len(bucket_means),
        "bootstrap_lower": means[lower_index],
        "bootstrap_upper": means[upper_index],
    }


def _bucket_summary(rows: list[dict[str, Any]], *, bucket_key: str) -> dict[str, list[float]]:
    bucket_posterior: dict[str, list[float]] = defaultdict(list)
    bucket_representation: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        bucket = str(row[bucket_key])
        bucket_posterior[bucket].append(float(row["posterior_js"]))
        bucket_representation[bucket].append(float(row["representation_js"]))
    buckets = sorted(bucket_posterior)
    return {
        "posterior_means": [_mean(bucket_posterior[bucket]) for bucket in buckets],
        "representation_means": [_mean(bucket_representation[bucket]) for bucket in buckets],
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x <= 0.0 or denom_y <= 0.0:
        return None
    return numerator / (denom_x * denom_y)
