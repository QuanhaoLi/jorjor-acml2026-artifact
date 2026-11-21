from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, product
import math
import random
from typing import Any, Callable

from jorjor.probe.counterfactual import jensen_shannon


LATENT_TYPES = ("A", "B", "C", "D")
LATENT_EVENTS = ("a", "b", "c", "d")
LATENT_PRIOR = tuple(1.0 / len(LATENT_TYPES) for _ in LATENT_TYPES)
PROTOCOL_TRANSFER_SUPPORTED = "belief_like_protocol_transfer_supported"
INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class LatentTypeSequence:
    """One generated non-poker hidden-type sequence."""

    sequence_id: str
    hidden_type: str
    events: tuple[str, ...]


def emission_probability(
    *,
    hidden_type: str,
    event: str,
    position_index: int,
    signal_probability: float = 0.72,
) -> float:
    """Return p(event | hidden_type, position) for the order-sensitive benchmark."""

    if hidden_type not in LATENT_TYPES:
        raise ValueError(f"unknown hidden_type: {hidden_type}")
    if event not in LATENT_EVENTS:
        raise ValueError(f"unknown event: {event}")
    if not (0.0 < signal_probability < 1.0):
        raise ValueError("signal_probability must be between 0 and 1")
    hidden_index = LATENT_TYPES.index(hidden_type)
    event_index = LATENT_EVENTS.index(event)
    preferred_index = (hidden_index + position_index) % len(LATENT_EVENTS)
    if event_index == preferred_index:
        return signal_probability
    return (1.0 - signal_probability) / (len(LATENT_EVENTS) - 1)


def exact_posterior(
    events: tuple[str, ...],
    *,
    signal_probability: float = 0.72,
) -> tuple[float, ...]:
    """Return exact p(z | ordered public events)."""

    if not events:
        return LATENT_PRIOR
    weights = []
    for hidden_type, prior in zip(LATENT_TYPES, LATENT_PRIOR, strict=True):
        likelihood = prior
        for position_index, event in enumerate(events):
            likelihood *= emission_probability(
                hidden_type=hidden_type,
                event=event,
                position_index=position_index,
                signal_probability=signal_probability,
            )
        weights.append(likelihood)
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("posterior has zero normalizer")
    return tuple(value / total for value in weights)


def generate_latent_type_sequences(
    *,
    sequence_count: int,
    length: int,
    seed: int,
    signal_probability: float = 0.72,
) -> list[LatentTypeSequence]:
    """Generate hidden-type sequences from the benchmark process."""

    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")
    if length <= 1:
        raise ValueError("length must be greater than 1")
    rng = random.Random(seed)
    sequences = []
    for index in range(sequence_count):
        hidden_type = rng.choice(LATENT_TYPES)
        events = tuple(
            _sample_event(
                hidden_type=hidden_type,
                position_index=position_index,
                rng=rng,
                signal_probability=signal_probability,
            )
            for position_index in range(length)
        )
        sequences.append(
            LatentTypeSequence(
                sequence_id=f"latent-type-{index:06d}",
                hidden_type=hidden_type,
                events=events,
            )
        )
    return sequences


def all_prefixes(length: int) -> list[tuple[str, ...]]:
    """Return all non-empty public-event prefixes up to length."""

    if length <= 1:
        raise ValueError("length must be greater than 1")
    prefixes: list[tuple[str, ...]] = []
    for prefix_len in range(1, length + 1):
        prefixes.extend(product(LATENT_EVENTS, repeat=prefix_len))
    return prefixes


def prefix_rows(
    sequences: list[LatentTypeSequence],
    *,
    signal_probability: float = 0.72,
) -> list[dict[str, Any]]:
    """Return paper-auditable rows for every generated prefix."""

    rows = []
    for sequence in sequences:
        for prefix_index in range(1, len(sequence.events) + 1):
            events = sequence.events[:prefix_index]
            targets = action_value_targets(events, signal_probability=signal_probability)
            rows.append(
                {
                    "sequence_id": sequence.sequence_id,
                    "prefix_index": prefix_index,
                    "ordered_events": list(events),
                    "event_counts": dict(Counter(events)),
                    "last_event": events[-1],
                    "hidden_type": sequence.hidden_type,
                    "oracle_posterior": exact_posterior(events, signal_probability=signal_probability),
                    "composition_key": composition_key(events),
                    "order_key": order_key(events),
                    "action_label": targets["action_label"],
                    "value_label": targets["value_label"],
                }
            )
    return rows


def composition_key(events: tuple[str, ...]) -> str:
    """Return the visible composition key used for matched buckets."""

    if not events:
        return "len=0|last=none|" + "|".join(f"{event}=0" for event in LATENT_EVENTS)
    counts = Counter(events)
    parts = [f"len={len(events)}", f"last={events[-1]}"]
    parts.extend(f"{event}={counts[event]}" for event in LATENT_EVENTS)
    return "|".join(parts)


def order_key(events: tuple[str, ...]) -> str:
    """Return the ordered event key."""

    return ",".join(events) if events else "empty"


def composition_state(events: tuple[str, ...]) -> tuple[float, ...]:
    """Return a normalized composition shortcut state."""

    if not events:
        return LATENT_PRIOR
    counts = Counter(events)
    total = float(len(events))
    count_features = tuple(counts[event] / total for event in LATENT_EVENTS)
    last_features = tuple(1.0 if events[-1] == event else 0.0 for event in LATENT_EVENTS)
    mixed = tuple(0.75 * count + 0.25 * last for count, last in zip(count_features, last_features, strict=True))
    return normalize_distribution(mixed)


def action_value_targets(
    events: tuple[str, ...],
    *,
    signal_probability: float = 0.72,
) -> dict[str, Any]:
    """Return generated behavior labels for the learned non-poker case."""

    posterior = exact_posterior(events, signal_probability=signal_probability)
    action_scores = (
        0.80 * posterior[0] + 0.15 * posterior[1] - 0.10 * posterior[2],
        0.20 * posterior[0] + 0.75 * posterior[2] + 0.10 * posterior[3],
        0.15 * posterior[1] + 0.25 * posterior[2] + 0.85 * posterior[3],
    )
    action_label = max(range(len(action_scores)), key=lambda index: action_scores[index])
    value_label = 0.9 * posterior[0] + 0.25 * posterior[1] - 0.15 * posterior[2] - 0.7 * posterior[3]
    return {
        "action_label": int(action_label),
        "action_scores": tuple(float(score) for score in action_scores),
        "value_label": float(value_label),
    }


def transition_summary(
    sequences: list[LatentTypeSequence],
    *,
    representation_fn: Callable[[tuple[str, ...]], tuple[float, ...]],
    signal_probability: float = 0.72,
) -> dict[str, Any]:
    """Compare prefix posterior updates to representation updates."""

    posterior_js = []
    representation_js = []
    transition_count = 0
    non_final_transition_count = 0
    for sequence in sequences:
        prefixes = [sequence.events[:prefix_len] for prefix_len in range(1, len(sequence.events) + 1)]
        for left, right in zip(prefixes, prefixes[1:], strict=False):
            transition_count += 1
            if len(right) < len(sequence.events):
                non_final_transition_count += 1
                posterior_js.append(
                    jensen_shannon(
                        exact_posterior(left, signal_probability=signal_probability),
                        exact_posterior(right, signal_probability=signal_probability),
                    )
                )
                representation_js.append(jensen_shannon(representation_fn(left), representation_fn(right)))
    return _summary_values(
        posterior_js=posterior_js,
        representation_js=representation_js,
        extra={
            "transition_count": transition_count,
            "non_final_transition_count": non_final_transition_count,
        },
    )


def matched_order_rows(
    *,
    length: int,
    representation_fn: Callable[[tuple[str, ...]], tuple[float, ...]],
    signal_probability: float = 0.72,
    max_order_pairs: int = 4096,
) -> list[dict[str, Any]]:
    """Return same-composition/same-last different-order rows."""

    groups: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for events in product(LATENT_EVENTS, repeat=length):
        groups[composition_key(events)].append(events)

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
            rows.append(
                {
                    "composition_key": bucket,
                    "left_order_key": order_key(left),
                    "right_order_key": order_key(right),
                    "left_events": list(left),
                    "right_events": list(right),
                    "posterior_js": posterior_js,
                    "representation_js": jensen_shannon(representation_fn(left), representation_fn(right)),
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
    """Summarize matched-order rows with bucket bootstrap intervals."""

    posterior_js = [float(row["posterior_js"]) for row in rows]
    representation_js = [float(row["representation_js"]) for row in rows]
    buckets = sorted({str(row["composition_key"]) for row in rows})
    bucket_summary = _bucket_summary(rows, bucket_key="composition_key") if rows else None
    return _summary_values(
        posterior_js=bucket_summary["posterior_means"] if bucket_summary else posterior_js,
        representation_js=bucket_summary["representation_means"] if bucket_summary else representation_js,
        extra={
            "pair_count": len(rows),
            "matched_bucket_count": len(buckets),
            "estimand": "unweighted_composition_bucket_mean",
            "pair_weighted_mean_posterior_js": _mean(posterior_js),
            "pair_weighted_mean_representation_js": _mean(representation_js),
            "pair_weighted_posterior_representation_correlation": _pearson(posterior_js, representation_js),
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
        },
    )


def normalize_distribution(values: tuple[float, ...]) -> tuple[float, ...]:
    """Return a non-negative normalized probability vector."""

    clipped = tuple(max(0.0, float(value)) for value in values)
    total = sum(clipped)
    if total <= 0.0:
        return tuple(1.0 / len(values) for _ in values)
    return tuple(value / total for value in clipped)


def _sample_event(
    *,
    hidden_type: str,
    position_index: int,
    rng: random.Random,
    signal_probability: float,
) -> str:
    threshold = rng.random()
    cumulative = 0.0
    for event in LATENT_EVENTS:
        cumulative += emission_probability(
            hidden_type=hidden_type,
            event=event,
            position_index=position_index,
            signal_probability=signal_probability,
        )
        if threshold <= cumulative:
            return event
    return LATENT_EVENTS[-1]


def _summary_values(
    *,
    posterior_js: list[float],
    representation_js: list[float],
    extra: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "mean_posterior_js": _mean(posterior_js),
        "mean_representation_js": _mean(representation_js),
        "posterior_representation_correlation": _pearson(posterior_js, representation_js),
    }
    payload.update(extra)
    return payload


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
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
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
