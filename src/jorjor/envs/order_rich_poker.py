from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
import random
from typing import Any, Callable, Iterator

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.probe.counterfactual import jensen_shannon


HIDDEN_STRENGTHS = ("air", "marginal", "value", "trap")
PUBLIC_EVENTS = ("p0_check", "p0_probe", "p0_call", "p1_check", "p1_call", "p1_raise")
BEHAVIOR_ACTIONS = ("check", "call", "pressure")
STRENGTH_PRIOR = tuple(1.0 / len(HIDDEN_STRENGTHS) for _ in HIDDEN_STRENGTHS)
EXACT_POSTERIOR_SIGNAL_SUPPORTED = "exact_posterior_signal_supported"
COMPOSITION_COLLAPSED = "composition_collapsed"
ORDER_RICH_EXACT_POSTERIOR_SUPPORTED = "order_rich_exact_posterior_supported"
INSUFFICIENT_EXACT_POSTERIOR_SUPPORT = "insufficient_exact_posterior_support"

_PREFERRED_EVENTS = {
    "air": ("p0_probe", "p1_call", "p0_check", "p1_check", "p0_probe", "p1_check"),
    "marginal": ("p0_check", "p1_call", "p0_call", "p1_check", "p0_call", "p1_call"),
    "value": ("p0_probe", "p1_raise", "p0_call", "p1_raise", "p0_probe", "p1_raise"),
    "trap": ("p0_check", "p1_check", "p0_probe", "p1_call", "p0_call", "p1_raise"),
}


@dataclass(frozen=True)
class OrderRichSequence:
    """One controlled poker-like public sequence with a hidden opponent strength."""

    sequence_id: str
    hidden_strength: str
    events: tuple[str, ...]


def event_probability(
    *,
    hidden_strength: str,
    event: str,
    position_index: int,
    signal_probability: float = 0.70,
) -> float:
    """Return p(public event | hidden strength, position)."""

    if hidden_strength not in HIDDEN_STRENGTHS:
        raise ValueError(f"unknown hidden strength: {hidden_strength}")
    if event not in PUBLIC_EVENTS:
        raise ValueError(f"unknown public event: {event}")
    if not 0.0 < signal_probability < 1.0:
        raise ValueError("signal_probability must be between 0 and 1")
    preferred = _PREFERRED_EVENTS[hidden_strength][position_index % len(_PREFERRED_EVENTS[hidden_strength])]
    if event == preferred:
        return signal_probability
    return (1.0 - signal_probability) / (len(PUBLIC_EVENTS) - 1)


def exact_posterior(
    events: tuple[str, ...],
    *,
    signal_probability: float = 0.70,
) -> tuple[float, ...]:
    """Return exact p(hidden strength | ordered public events)."""

    if not events:
        return STRENGTH_PRIOR
    weights = []
    for hidden_strength, prior in zip(HIDDEN_STRENGTHS, STRENGTH_PRIOR, strict=True):
        likelihood = prior
        for position_index, event in enumerate(events):
            likelihood *= event_probability(
                hidden_strength=hidden_strength,
                event=event,
                position_index=position_index,
                signal_probability=signal_probability,
            )
        weights.append(likelihood)
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("posterior normalizer is zero")
    return tuple(value / total for value in weights)


def generate_sequences(
    *,
    sequence_count: int,
    length: int,
    seed: int,
    signal_probability: float = 0.70,
) -> list[OrderRichSequence]:
    """Generate controlled poker-like public sequences."""

    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")
    if length <= 1:
        raise ValueError("length must be greater than 1")
    rng = random.Random(seed)
    sequences = []
    for index in range(sequence_count):
        hidden_strength = rng.choice(HIDDEN_STRENGTHS)
        events = tuple(
            _sample_event(
                hidden_strength=hidden_strength,
                position_index=position_index,
                rng=rng,
                signal_probability=signal_probability,
            )
            for position_index in range(length)
        )
        sequences.append(
            OrderRichSequence(
                sequence_id=f"order-rich-poker-{index:06d}",
                hidden_strength=hidden_strength,
                events=events,
            )
        )
    return sequences


def prefix_rows(
    sequences: list[OrderRichSequence],
    *,
    signal_probability: float = 0.70,
) -> list[dict[str, Any]]:
    """Return auditable prefix rows for generated sequences."""

    rows = []
    for sequence in sequences:
        for prefix_index in range(1, len(sequence.events) + 1):
            events = sequence.events[:prefix_index]
            posterior = exact_posterior(events, signal_probability=signal_probability)
            targets = behavior_targets(events, signal_probability=signal_probability)
            rows.append(
                {
                    "sequence_id": sequence.sequence_id,
                    "prefix_index": prefix_index,
                    "hidden_strength": sequence.hidden_strength,
                    "ordered_public_history": list(events),
                    "composition_key": composition_key(events),
                    "order_key": order_key(events),
                    "event_counts": dict(Counter(events)),
                    "last_event": events[-1],
                    "exact_posterior": posterior,
                    "teacher_action_distribution": targets["action_distribution"],
                    "action_label": targets["action_label"],
                    "value_label": targets["value_label"],
                }
            )
    return rows


def behavior_targets(
    events: tuple[str, ...],
    *,
    signal_probability: float = 0.70,
) -> dict[str, Any]:
    """Return behavior labels generated from the exact posterior."""

    posterior = exact_posterior(events, signal_probability=signal_probability)
    scores = (
        0.70 * posterior[0] + 0.25 * posterior[1] + 0.05 * posterior[3],
        0.15 * posterior[0] + 0.70 * posterior[1] + 0.25 * posterior[2],
        0.10 * posterior[1] + 0.85 * posterior[2] + 0.75 * posterior[3],
    )
    distribution = normalize_distribution(scores)
    action_label = max(range(len(scores)), key=lambda index: scores[index])
    value_label = -0.55 * posterior[0] + 0.05 * posterior[1] + 0.75 * posterior[2] + 0.35 * posterior[3]
    return {
        "action_distribution": tuple(float(value) for value in distribution),
        "action_label": BEHAVIOR_ACTIONS[action_label],
        "value_label": float(value_label),
    }


def composition_key(events: tuple[str, ...]) -> str:
    """Return same-composition key that intentionally discards order."""

    if not events:
        return "len=0|last=none|" + "|".join(f"{event}=0" for event in PUBLIC_EVENTS)
    counts = Counter(events)
    parts = [f"len={len(events)}", f"last={events[-1]}"]
    parts.extend(f"{event}={counts[event]}" for event in PUBLIC_EVENTS)
    return "|".join(parts)


def order_key(events: tuple[str, ...]) -> str:
    """Return the ordered public event key."""

    return ",".join(events) if events else "empty"


def composition_state(events: tuple[str, ...]) -> tuple[float, ...]:
    """Return a visible-composition shortcut representation."""

    if not events:
        return tuple(1.0 / len(PUBLIC_EVENTS) for _ in PUBLIC_EVENTS)
    counts = Counter(events)
    total = float(len(events))
    count_features = tuple(counts[event] / total for event in PUBLIC_EVENTS)
    last_features = tuple(1.0 if events[-1] == event else 0.0 for event in PUBLIC_EVENTS)
    return normalize_distribution(
        tuple(0.80 * count + 0.20 * last for count, last in zip(count_features, last_features, strict=True))
    )


def all_event_sequences(length: int) -> list[tuple[str, ...]]:
    """Return all public event sequences of a fixed length."""

    if length <= 1:
        raise ValueError("length must be greater than 1")
    return list(product(PUBLIC_EVENTS, repeat=length))


def matched_order_rows(
    *,
    length: int,
    representation_fn: Callable[[tuple[str, ...]], tuple[float, ...]],
    signal_probability: float = 0.70,
    max_pairs: int = 30000,
) -> list[dict[str, Any]]:
    """Return same-composition/different-order rows with exact posterior JS."""

    groups: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for events in all_event_sequences(length):
        groups[composition_key(events)].append(events)
    eligible = {bucket: values for bucket, values in groups.items() if len(values) >= 2}
    if not eligible:
        return []
    rows = []
    active: list[tuple[str, Iterator[tuple[tuple[str, ...], tuple[str, ...]]]]] = [
        (bucket, combinations(candidates, 2))
        for bucket, candidates in sorted(eligible.items())
    ]
    while active and len(rows) < max_pairs:
        next_active = []
        for bucket, pair_iterator in active:
            row = _next_bucket_pair_row(
            bucket=bucket,
                pair_iterator=pair_iterator,
            representation_fn=representation_fn,
            signal_probability=signal_probability,
        )
            if row is None:
                continue
            rows.append(row)
            next_active.append((bucket, pair_iterator))
            if len(rows) >= max_pairs:
                return rows
        active = next_active
    return rows


def matched_order_summary(
    rows: list[dict[str, Any]],
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Summarize matched-order rows with bucket bootstrap intervals."""

    pair_posterior_js = [float(row["posterior_js"]) for row in rows]
    pair_representation_js = [float(row["representation_js"]) for row in rows]
    buckets = sorted({str(row["composition_key"]) for row in rows})
    bucket_means = _bucket_value_means(rows) if rows else {}
    representation_ci = _bucket_bootstrap_interval(
        rows,
        value_key="representation_js",
        bucket_key="composition_key",
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    ) if rows else {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    posterior_ci = _bucket_bootstrap_interval(
        rows,
        value_key="posterior_js",
        bucket_key="composition_key",
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed + 1,
        confidence_level=confidence_level,
    ) if rows else {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    return {
        "pair_count": len(rows),
        "matched_bucket_count": len(buckets),
        "mean_posterior_js": _mean(bucket_means.get("posterior_js", [])),
        "mean_representation_js": _mean(bucket_means.get("representation_js", [])),
        "posterior_js_ci": posterior_ci,
        "representation_js_ci": representation_ci,
        "pair_table_mean_posterior_js": _mean(pair_posterior_js),
        "pair_table_mean_representation_js": _mean(pair_representation_js),
        "pair_weighted_posterior_representation_correlation": _pearson(pair_posterior_js, pair_representation_js),
        "estimand": "unweighted_composition_bucket_mean",
    }


def posterior_equivalence_rows(
    *,
    length: int,
    max_pairs: int,
    signal_probability: float = 0.70,
    posterior_threshold: float = 0.002,
) -> list[dict[str, Any]]:
    """Return different-surface rows whose exact posteriors are nearly equivalent."""

    sequences = all_event_sequences(length)
    rows = []
    for left, right in combinations(sequences, 2):
        if order_key(left) == order_key(right):
            continue
        posterior_js = _js(
            exact_posterior(left, signal_probability=signal_probability),
            exact_posterior(right, signal_probability=signal_probability),
        )
        if posterior_js > posterior_threshold:
            continue
        rows.append(
            {
                "left_order_key": order_key(left),
                "right_order_key": order_key(right),
                "left_composition_key": composition_key(left),
                "right_composition_key": composition_key(right),
                "posterior_js": posterior_js,
            }
        )
        if len(rows) >= max_pairs:
            return rows
    return rows


def build_order_rich_exact_posterior_report(
    *,
    output_dir: Path,
    sequence_count: int = 20000,
    length: int = 6,
    seed: int = 20260610,
    signal_probability: float = 0.70,
    max_pairs: int = 30000,
    bootstrap_count: int = 1000,
    bootstrap_seed: int = 20260610,
    confidence_level: float = 0.95,
    min_bucket_count: int = 100,
    min_pair_count: int = 2000,
) -> dict[str, Any]:
    """Build the r.0.2 order-rich exact-posterior feasibility report."""

    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")
    if length <= 1:
        raise ValueError("length must be greater than 1")
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    if bootstrap_count <= 0:
        raise ValueError("bootstrap_count must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = generate_sequences(
        sequence_count=sequence_count,
        length=length,
        seed=seed,
        signal_probability=signal_probability,
    )
    rows = prefix_rows(sequences, signal_probability=signal_probability)
    oracle_rows = matched_order_rows(
        length=length,
        representation_fn=lambda events: exact_posterior(events, signal_probability=signal_probability),
        signal_probability=signal_probability,
        max_pairs=max_pairs,
    )
    composition_rows = matched_order_rows(
        length=length,
        representation_fn=composition_state,
        signal_probability=signal_probability,
        max_pairs=max_pairs,
    )
    oracle_summary = matched_order_summary(
        oracle_rows,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    composition_summary = matched_order_summary(
        composition_rows,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed + 101,
        confidence_level=confidence_level,
    )
    equivalence_rows = posterior_equivalence_rows(
        length=length,
        max_pairs=min(max_pairs, 5000),
        signal_probability=signal_probability,
    )
    oracle_label = (
        EXACT_POSTERIOR_SIGNAL_SUPPORTED
        if oracle_summary["representation_js_ci"]["lower"] > 0.0
        and oracle_summary["mean_representation_js"] > 0.0
        else INSUFFICIENT_EXACT_POSTERIOR_SUPPORT
    )
    composition_label = (
        COMPOSITION_COLLAPSED
        if composition_summary["mean_representation_js"] <= 1e-12
        and composition_summary["representation_js_ci"]["upper"] <= 1e-12
        else "composition_not_collapsed"
    )
    feasibility_passed = (
        oracle_summary["matched_bucket_count"] >= min_bucket_count
        and oracle_summary["pair_count"] >= min_pair_count
        and oracle_label == EXACT_POSTERIOR_SIGNAL_SUPPORTED
        and composition_label == COMPOSITION_COLLAPSED
    )
    payload = {
        "version": "r.0.2",
        "report_type": "r.0.2_order_rich_exact_posterior",
        "claim_scope": "exact_posterior_benchmark_feasibility_not_model_mechanism",
        "sequence_count": sequence_count,
        "length": length,
        "seed": seed,
        "signal_probability": signal_probability,
        "max_pairs": max_pairs,
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "min_bucket_count": min_bucket_count,
        "min_pair_count": min_pair_count,
        "hidden_strengths": list(HIDDEN_STRENGTHS),
        "public_events": list(PUBLIC_EVENTS),
        "generated_row_summary": _row_summary(rows),
        "enumerated_sequence_count": len(PUBLIC_EVENTS) ** length,
        "oracle_summary": oracle_summary,
        "composition_summary": composition_summary,
        "posterior_equivalence_summary": _posterior_equivalence_summary(equivalence_rows),
        "oracle_detection_label": oracle_label,
        "composition_detection_label": composition_label,
        "final_diagnostic_label": (
            ORDER_RICH_EXACT_POSTERIOR_SUPPORTED if feasibility_passed else INSUFFICIENT_EXACT_POSTERIOR_SUPPORT
        ),
        "allowed_paper_claim": (
            "The controlled order-rich poker-like benchmark admits exact-posterior same-composition order gates. "
            "It validates the audit domain, not a trained model mechanism."
        ),
        "forbidden_paper_claim": "r.0.2 proves behavior-only AR models learn CBPS or posterior tracking.",
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "r.0.2_order_rich_exact_posterior_report.json", payload)
    (output_dir / "r.0.2_order_rich_exact_posterior_report.md").write_text(_markdown(payload), encoding="utf-8")
    write_json_artifact(output_dir / "r.0.2_order_rich_exact_posterior_manifest.json", _manifest(output_dir, payload))
    return payload


def normalize_distribution(values: tuple[float, ...]) -> tuple[float, ...]:
    """Return a non-negative normalized probability vector."""

    clipped = tuple(max(0.0, float(value)) for value in values)
    total = sum(clipped)
    if total <= 0.0:
        return tuple(1.0 / len(values) for _ in values)
    return tuple(value / total for value in clipped)


def _sample_event(
    *,
    hidden_strength: str,
    position_index: int,
    rng: random.Random,
    signal_probability: float,
) -> str:
    threshold = rng.random()
    cumulative = 0.0
    for event in PUBLIC_EVENTS:
        cumulative += event_probability(
            hidden_strength=hidden_strength,
            event=event,
            position_index=position_index,
            signal_probability=signal_probability,
        )
        if threshold <= cumulative:
            return event
    return PUBLIC_EVENTS[-1]


def _next_bucket_pair_row(
    *,
    bucket: str,
    pair_iterator: Iterator[tuple[tuple[str, ...], tuple[str, ...]]],
    representation_fn: Callable[[tuple[str, ...]], tuple[float, ...]],
    signal_probability: float,
) -> dict[str, Any] | None:
    for left, right in pair_iterator:
        posterior_js = _js(
            exact_posterior(left, signal_probability=signal_probability),
            exact_posterior(right, signal_probability=signal_probability),
        )
        if posterior_js <= 1e-12:
            continue
        return {
            "composition_key": bucket,
            "left_order_key": order_key(left),
            "right_order_key": order_key(right),
            "left_events": list(left),
            "right_events": list(right),
            "posterior_js": posterior_js,
            "representation_js": _js(representation_fn(left), representation_fn(right)),
        }
    return None


def _bucket_bootstrap_interval(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    bucket_key: str,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, float]:
    bucket_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        bucket_values[str(row[bucket_key])].append(float(row[value_key]))
    if not bucket_values:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    bucket_means = [sum(values) / len(values) for values in bucket_values.values()]
    rng = random.Random(bootstrap_seed)
    means = []
    for _ in range(bootstrap_count):
        means.append(sum(bucket_means[rng.randrange(len(bucket_means))] for _ in bucket_means) / len(bucket_means))
    means.sort()
    lower_index = int(((1.0 - confidence_level) / 2.0) * (bootstrap_count - 1))
    upper_index = int((1.0 - (1.0 - confidence_level) / 2.0) * (bootstrap_count - 1))
    return {
        "mean": sum(bucket_means) / len(bucket_means),
        "lower": means[lower_index],
        "upper": means[upper_index],
    }


def _bucket_value_means(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    value_keys = ("posterior_js", "representation_js")
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {key: [] for key in value_keys})
    for row in rows:
        bucket = str(row["composition_key"])
        for key in value_keys:
            grouped[bucket][key].append(float(row[key]))
    means: dict[str, list[float]] = {key: [] for key in value_keys}
    for bucket in sorted(grouped):
        for key in value_keys:
            means[key].append(_mean(grouped[bucket][key]))
    return means


def _row_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts = Counter(str(row["action_label"]) for row in rows)
    prefix_counts = Counter(str(row["prefix_index"]) for row in rows)
    return {
        "row_count": len(rows),
        "composition_bucket_count": len({str(row["composition_key"]) for row in rows}),
        "order_key_count": len({str(row["order_key"]) for row in rows}),
        "action_counts": dict(sorted(action_counts.items())),
        "prefix_counts": dict(sorted(prefix_counts.items(), key=lambda item: int(item[0]))),
    }


def _posterior_equivalence_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [max(0.0, float(row["posterior_js"])) for row in rows]
    return {
        "pair_count": len(rows),
        "mean_posterior_js": _mean(values),
        "max_posterior_js": max(values) if values else 0.0,
        "interpretation": "different surface histories with near-equivalent exact posteriors",
    }


def _manifest(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": payload["version"],
        "report_type": "r.0.2_order_rich_exact_posterior_manifest",
        "claim_scope": payload["claim_scope"],
        "output_dir": str(output_dir),
        "expected_outputs": [
            "r.0.2_order_rich_exact_posterior_report.json",
            "r.0.2_order_rich_exact_posterior_report.md",
            "r.0.2_order_rich_exact_posterior_manifest.json",
        ],
        "report_checksum": payload["checksum"],
        "final_diagnostic_label": payload["final_diagnostic_label"],
        "one_command_reproduction": (
            ".venv/bin/python scripts/build_r002_order_rich_exact_posterior.py "
            f"--output-dir {output_dir} "
            f"--sequence-count {payload['sequence_count']} "
            f"--length {payload['length']} "
            f"--seed {payload['seed']} "
            f"--signal-probability {payload['signal_probability']} "
            f"--max-pairs {payload['max_pairs']} "
            f"--bootstrap-count {payload['bootstrap_count']} "
            f"--bootstrap-seed {payload['bootstrap_seed']}"
        ),
    }


def _markdown(payload: dict[str, Any]) -> str:
    oracle = payload["oracle_summary"]
    composition = payload["composition_summary"]
    lines = [
        "# r.0.2 Order-Rich Exact Posterior Report",
        "",
        f"final_diagnostic_label: `{payload['final_diagnostic_label']}`",
        "",
        "## Oracle Gate",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| matched_bucket_count | {oracle['matched_bucket_count']} |",
        f"| pair_count | {oracle['pair_count']} |",
        f"| mean_representation_js | {oracle['mean_representation_js']:.8f} |",
        f"| representation_js_ci_lower | {oracle['representation_js_ci']['lower']:.8f} |",
        f"| representation_js_ci_upper | {oracle['representation_js_ci']['upper']:.8f} |",
        "",
        "## Composition Gate",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| mean_representation_js | {composition['mean_representation_js']:.8f} |",
        f"| representation_js_ci_upper | {composition['representation_js_ci']['upper']:.8f} |",
        "",
        "## Controls",
        "",
        f"- oracle_detection_label: `{payload['oracle_detection_label']}`",
        f"- composition_detection_label: `{payload['composition_detection_label']}`",
        f"- posterior_equivalence_pair_count: `{payload['posterior_equivalence_summary']['pair_count']}`",
        "",
        "## Claim Discipline",
        "",
        payload["allowed_paper_claim"],
        "",
        f"Forbidden claim: {payload['forbidden_paper_claim']}",
        "",
    ]
    return "\n".join(lines)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _js(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return max(0.0, jensen_shannon(left, right))


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denom_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    denom_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if denom_x <= 0.0 or denom_y <= 0.0:
        return None
    return numerator / (denom_x * denom_y)
