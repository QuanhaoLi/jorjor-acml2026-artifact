from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations, product
import math
from pathlib import Path
import random
from typing import Any, Callable

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.envs.latent_type_sequence import (
    LATENT_EVENTS,
    composition_key,
    composition_state,
    exact_posterior,
    normalize_distribution,
)
from jorjor.probe.counterfactual import jensen_shannon


REPORT_TYPE = "k.5.0_synthetic_cbps_control"
POSTERIOR_TRACKING_SUPPORTED = "posterior_tracking_supported"
CBPS_SUPPORTED = "cbps_supported"
COMPOSITION_SHORTCUT_SUPPORTED = "composition_shortcut_supported"
INCONCLUSIVE = "inconclusive"
POSTERIOR_STATE = "posterior_state"
COMPOSITION_STATE = "composition_state"
CBPS_STATE = "CBPS_state"


def build_synthetic_cbps_report(
    *,
    output_dir: Path,
    sequence_count: int = 4096,
    length: int = 6,
    signal_probability: float = 0.72,
    max_pairs: int = 4096,
    bootstrap_count: int = 1000,
    bootstrap_seed: int = 20260610,
    confidence_level: float = 0.95,
    matched_pair_threshold: int = 512,
    mean_posterior_js_threshold: float = 0.05,
) -> dict[str, Any]:
    """Build the k.5.0 known-control CBPS diagnostic report."""

    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")
    if length <= 1:
        raise ValueError("length must be greater than 1")
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    prefixes = _sample_prefixes(sequence_count=sequence_count, length=length, seed=bootstrap_seed)
    matched_reference = _matched_rows(
        length=length,
        representation_fn=exact_posterior,
        signal_probability=signal_probability,
        max_pairs=max_pairs,
    )
    preflight = _preflight(
        matched_reference,
        matched_pair_threshold=matched_pair_threshold,
        mean_posterior_js_threshold=mean_posterior_js_threshold,
    )
    if not preflight["environment_valid"]:
        raise ValueError(f"k.5.0 synthetic CBPS preflight failed: {preflight}")

    case_specs: list[tuple[str, str, Callable[[tuple[str, ...]], tuple[float, ...]]]] = [
        (POSTERIOR_STATE, "exact_ordered_posterior", lambda prefix: exact_posterior(prefix, signal_probability=signal_probability)),
        (COMPOSITION_STATE, "visible_composition_only", composition_state),
        (CBPS_STATE, "public_history_predictive_compression", cbps_state),
    ]
    cases = []
    for index, (mode, representation_source, representation_fn) in enumerate(case_specs):
        predictive = _predictive_metrics(prefixes, representation_fn=representation_fn)
        latent = _latent_metrics(
            prefixes,
            representation_fn=representation_fn,
            signal_probability=signal_probability,
            shuffle_seed=bootstrap_seed + index,
        )
        boundedness = _composition_boundedness_metrics(
            length=length,
            representation_fn=representation_fn,
            signal_probability=signal_probability,
            max_pairs=max_pairs,
            bootstrap_count=bootstrap_count,
            bootstrap_seed=bootstrap_seed + index,
            confidence_level=confidence_level,
        )
        cases.append(
            {
                "mode": mode,
                "representation_source": representation_source,
                "training_target": "none_analytic_known_control",
                "predictive_metrics": predictive,
                "latent_metrics": latent,
                "composition_boundedness": boundedness,
                "final_diagnostic_label": _verdict(
                    mode=mode,
                    predictive=predictive,
                    latent=latent,
                    boundedness=boundedness,
                ),
            }
        )

    payload = {
        "version": "k.5.0",
        "report_type": REPORT_TYPE,
        "claim_scope": "synthetic_cbps_label_validation_not_poker_claim_upgrade",
        "sequence_count": sequence_count,
        "length": length,
        "signal_probability": signal_probability,
        "max_pairs": max_pairs,
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "matched_pair_threshold": matched_pair_threshold,
        "mean_posterior_js_threshold": mean_posterior_js_threshold,
        "preflight": preflight,
        "cases": cases,
        "synthetic_cbps_control_passed": _control_passed(cases),
        "allowed_paper_claim": (
            "In a known-control latent-order environment, the claim-gated protocol separates exact posterior "
            "tracking, raw composition shortcut, and composition-bounded predictive state."
        ),
        "forbidden_paper_claim": "The synthetic CBPS result proves the poker model uses CBPS.",
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "k.5.0_synthetic_cbps_report.json", payload)
    (output_dir / "k.5.0_synthetic_cbps_report.md").write_text(_markdown(payload), encoding="utf-8")
    write_json_artifact(output_dir / "k.5.0_synthetic_cbps_manifest.json", _manifest(output_dir, payload))
    return payload


def cbps_state(events: tuple[str, ...]) -> tuple[float, ...]:
    """Return a public-history predictive compression that intentionally drops posterior order detail."""

    if not events:
        return tuple(1.0 / len(LATENT_EVENTS) for _ in LATENT_EVENTS)
    event_index = {event: index for index, event in enumerate(LATENT_EVENTS)}
    counts = Counter(events)
    length = float(len(events))
    count_features = [counts[event] / length for event in LATENT_EVENTS]
    transition_denominator = float(max(1, len(events) - 1))
    repeat_rate = sum(1 for left, right in zip(events, events[1:], strict=False) if left == right) / transition_denominator
    cyclic_rate = (
        sum(
            1
            for left, right in zip(events, events[1:], strict=False)
            if (event_index[right] - event_index[left]) % len(LATENT_EVENTS) == 1
        )
        / transition_denominator
    )
    reverse_rate = (
        sum(
            1
            for left, right in zip(events, events[1:], strict=False)
            if (event_index[left] - event_index[right]) % len(LATENT_EVENTS) == 1
        )
        / transition_denominator
    )
    edge_phase = (event_index[events[0]] - event_index[events[-1]]) % len(LATENT_EVENTS)
    edge_rate = edge_phase / float(len(LATENT_EVENTS) - 1)
    values = (
        0.10 + 0.35 * count_features[0] + 0.90 * repeat_rate,
        0.10 + 0.35 * count_features[1] + 0.90 * cyclic_rate,
        0.10 + 0.35 * count_features[2] + 0.90 * reverse_rate,
        0.10 + 0.35 * count_features[3] + 0.50 * edge_rate,
    )
    return normalize_distribution(values)


def cbps_behavior_targets(events: tuple[str, ...]) -> dict[str, Any]:
    """Return behavior targets generated from the public CBPS compression."""

    state = cbps_state(events)
    action_scores = (
        1.10 * state[0] + 0.20 * state[1] - 0.10 * state[3],
        1.05 * state[1] + 0.30 * state[2],
        0.95 * state[2] + 0.55 * state[3] - 0.05 * state[0],
    )
    action_label = max(range(len(action_scores)), key=lambda index: action_scores[index])
    value_label = max(action_scores)
    return {
        "action_label": int(action_label),
        "action_scores": tuple(float(value) for value in action_scores),
        "value_label": float(value_label),
    }


def _sample_prefixes(*, sequence_count: int, length: int, seed: int) -> list[tuple[str, ...]]:
    rng = random.Random(seed)
    prefixes = []
    for _ in range(sequence_count):
        prefix_length = rng.randint(2, length)
        prefixes.append(tuple(rng.choice(LATENT_EVENTS) for _ in range(prefix_length)))
    return prefixes


def _predictive_metrics(
    prefixes: list[tuple[str, ...]],
    *,
    representation_fn: Callable[[tuple[str, ...]], tuple[float, ...]],
) -> dict[str, Any]:
    prior_actions = [cbps_behavior_targets(prefix)["action_label"] for prefix in prefixes]
    majority_action = Counter(prior_actions).most_common(1)[0][0]
    predicted_actions = [_state_action(representation_fn(prefix)) for prefix in prefixes]
    composition_actions = [_state_action(composition_state(prefix)) for prefix in prefixes]
    target_values = [cbps_behavior_targets(prefix)["value_label"] for prefix in prefixes]
    predicted_values = [_state_value(representation_fn(prefix)) for prefix in prefixes]
    composition_values = [_state_value(composition_state(prefix)) for prefix in prefixes]
    prior_value = sum(target_values) / len(target_values)
    action_accuracy = _mean(
        [1.0 if prediction == target else 0.0 for prediction, target in zip(predicted_actions, prior_actions, strict=True)]
    )
    composition_action_accuracy = _mean(
        [1.0 if prediction == target else 0.0 for prediction, target in zip(composition_actions, prior_actions, strict=True)]
    )
    prior_action_accuracy = _mean([1.0 if majority_action == target else 0.0 for target in prior_actions])
    value_mse = _mean(
        [(prediction - target) ** 2 for prediction, target in zip(predicted_values, target_values, strict=True)]
    )
    composition_value_mse = _mean(
        [(prediction - target) ** 2 for prediction, target in zip(composition_values, target_values, strict=True)]
    )
    prior_value_mse = _mean([(prior_value - target) ** 2 for target in target_values])
    return {
        "target": "CBPS_public_behavior_target",
        "action_accuracy": action_accuracy,
        "composition_action_accuracy": composition_action_accuracy,
        "prior_action_accuracy": prior_action_accuracy,
        "predictive_action_delta_vs_prior": action_accuracy - prior_action_accuracy,
        "predictive_action_delta_vs_C_min": action_accuracy - composition_action_accuracy,
        "value_mse": value_mse,
        "composition_value_mse": composition_value_mse,
        "prior_value_mse": prior_value_mse,
        "predictive_value_delta_vs_prior": prior_value_mse - value_mse,
        "predictive_value_delta_vs_C_min": composition_value_mse - value_mse,
    }


def _latent_metrics(
    prefixes: list[tuple[str, ...]],
    *,
    representation_fn: Callable[[tuple[str, ...]], tuple[float, ...]],
    signal_probability: float,
    shuffle_seed: int,
) -> dict[str, Any]:
    posteriors = [exact_posterior(prefix, signal_probability=signal_probability) for prefix in prefixes]
    representations = [representation_fn(prefix) for prefix in prefixes]
    posterior_js = [
        jensen_shannon(posterior, representation)
        for posterior, representation in zip(posteriors, representations, strict=True)
    ]
    shuffled = list(posteriors)
    random.Random(shuffle_seed).shuffle(shuffled)
    shuffled_js = [
        jensen_shannon(posterior, representation)
        for posterior, representation in zip(shuffled, representations, strict=True)
    ]
    top_labels = [_top_index(posterior) for posterior in posteriors]
    predicted_labels = [_top_index(representation) for representation in representations]
    shuffled_labels = list(top_labels)
    random.Random(shuffle_seed + 1).shuffle(shuffled_labels)
    top1_accuracy = _mean(
        [1.0 if prediction == target else 0.0 for prediction, target in zip(predicted_labels, top_labels, strict=True)]
    )
    shuffled_top1_accuracy = _mean(
        [
            1.0 if prediction == target else 0.0
            for prediction, target in zip(predicted_labels, shuffled_labels, strict=True)
        ]
    )
    return {
        "mean_js_to_oracle_posterior": _mean(posterior_js),
        "mean_js_to_shuffled_posterior": _mean(shuffled_js),
        "label_permutation_delta": _mean(shuffled_js) - _mean(posterior_js),
        "top1_latent_accuracy": top1_accuracy,
        "shuffled_top1_latent_accuracy": shuffled_top1_accuracy,
        "top1_label_delta": top1_accuracy - shuffled_top1_accuracy,
        "latent_signal_positive": (_mean(shuffled_js) - _mean(posterior_js)) > 0.0,
    }


def _composition_boundedness_metrics(
    *,
    length: int,
    representation_fn: Callable[[tuple[str, ...]], tuple[float, ...]],
    signal_probability: float,
    max_pairs: int,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    rows = _matched_rows(
        length=length,
        representation_fn=representation_fn,
        signal_probability=signal_probability,
        max_pairs=max_pairs,
    )
    posterior_js = [float(row["posterior_js"]) for row in rows]
    representation_js = [float(row["representation_js"]) for row in rows]
    composition_js = [float(row["composition_js"]) for row in rows]
    bucket_interval = _bucket_bootstrap_interval(
        rows,
        value_key="representation_js",
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    slope = _linear_slope(posterior_js, representation_js)
    slope_interval = _bucket_bootstrap_slope_interval(
        rows,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed + 999,
        confidence_level=confidence_level,
    )
    bin_table = _posterior_gap_bins(rows)
    posterior_gap_tracking_failed = slope_interval["upper"] <= 0.0 or slope_interval["lower"] <= 0.0 <= slope_interval["upper"]
    return {
        "pair_count": len(rows),
        "matched_bucket_count": len({str(row["composition_key"]) for row in rows}),
        "estimand": "unweighted_composition_bucket_mean",
        "mean_posterior_js": _bucket_mean(rows, "posterior_js"),
        "mean_representation_js": _bucket_mean(rows, "representation_js"),
        "mean_composition_js": _bucket_mean(rows, "composition_js"),
        "pair_weighted_mean_posterior_js": _mean(posterior_js),
        "pair_weighted_mean_representation_js": _mean(representation_js),
        "pair_weighted_mean_composition_js": _mean(composition_js),
        "posterior_representation_correlation": _pearson(posterior_js, representation_js),
        "matched_posterior_gap_slope": slope,
        "matched_posterior_gap_slope_ci": slope_interval,
        "representation_js_ci": bucket_interval,
        "posterior_gap_bins": bin_table,
        "posterior_gap_tracking_failed": posterior_gap_tracking_failed,
        "bounded_by_public_history_family": (
            bucket_interval["bootstrap_lower"] <= 0.0 or posterior_gap_tracking_failed
        ),
    }


def _matched_rows(
    *,
    length: int,
    representation_fn: Callable[[tuple[str, ...]], tuple[float, ...]],
    signal_probability: float,
    max_pairs: int,
) -> list[dict[str, Any]]:
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
                    "left_events": list(left),
                    "right_events": list(right),
                    "posterior_js": posterior_js,
                    "representation_js": jensen_shannon(representation_fn(left), representation_fn(right)),
                    "composition_js": jensen_shannon(composition_state(left), composition_state(right)),
                }
            )
            if len(rows) >= max_pairs:
                return rows
    return rows


def _preflight(
    rows: list[dict[str, Any]],
    *,
    matched_pair_threshold: int,
    mean_posterior_js_threshold: float,
) -> dict[str, Any]:
    mean_posterior_js = _bucket_mean(rows, "posterior_js")
    return {
        "matched_composition_pair_count": len(rows),
        "matched_bucket_count": len({str(row["composition_key"]) for row in rows}),
        "mean_matched_posterior_js": mean_posterior_js,
        "behavior_target_not_fully_explained_by_C_min": _behavior_not_explained_by_composition(rows),
        "environment_valid": (
            len(rows) >= matched_pair_threshold
            and mean_posterior_js > mean_posterior_js_threshold
            and _behavior_not_explained_by_composition(rows)
        ),
    }


def _behavior_not_explained_by_composition(rows: list[dict[str, Any]]) -> bool:
    disagreements = 0
    for row in rows:
        left = tuple(row["left_events"])
        right = tuple(row["right_events"])
        if cbps_behavior_targets(left)["action_label"] != cbps_behavior_targets(right)["action_label"]:
            disagreements += 1
    return disagreements > 0


def _state_action(state: tuple[float, ...]) -> int:
    scores = (
        1.10 * state[0] + 0.20 * state[1] - 0.10 * state[3],
        1.05 * state[1] + 0.30 * state[2],
        0.95 * state[2] + 0.55 * state[3] - 0.05 * state[0],
    )
    return int(max(range(len(scores)), key=lambda index: scores[index]))


def _state_value(state: tuple[float, ...]) -> float:
    scores = (
        1.10 * state[0] + 0.20 * state[1] - 0.10 * state[3],
        1.05 * state[1] + 0.30 * state[2],
        0.95 * state[2] + 0.55 * state[3] - 0.05 * state[0],
    )
    return float(max(scores))


def _verdict(
    *,
    mode: str,
    predictive: dict[str, Any],
    latent: dict[str, Any],
    boundedness: dict[str, Any],
) -> str:
    interval = boundedness["representation_js_ci"]
    slope_interval = boundedness["matched_posterior_gap_slope_ci"]
    if mode == POSTERIOR_STATE:
        if interval["bootstrap_lower"] > 0.0 and slope_interval["lower"] > 0.0:
            return POSTERIOR_TRACKING_SUPPORTED
    if mode == COMPOSITION_STATE:
        if interval["bootstrap_upper"] <= 1e-12:
            return COMPOSITION_SHORTCUT_SUPPORTED
    if mode == CBPS_STATE:
        if (
            predictive["predictive_action_delta_vs_C_min"] > 0.0
            and predictive["predictive_value_delta_vs_C_min"] > 0.0
            and latent["latent_signal_positive"]
            and interval["bootstrap_lower"] > 0.0
            and (slope_interval["upper"] <= 0.0 or slope_interval["lower"] <= 0.0 <= slope_interval["upper"])
        ):
            return CBPS_SUPPORTED
    return INCONCLUSIVE


def _control_passed(cases: list[dict[str, Any]]) -> bool:
    labels = {case["mode"]: case["final_diagnostic_label"] for case in cases}
    return (
        labels.get(POSTERIOR_STATE) == POSTERIOR_TRACKING_SUPPORTED
        and labels.get(COMPOSITION_STATE) == COMPOSITION_SHORTCUT_SUPPORTED
        and labels.get(CBPS_STATE) == CBPS_SUPPORTED
    )


def _bucket_bootstrap_interval(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, float]:
    bucket_values = _bucket_values(rows, value_key)
    rng = random.Random(bootstrap_seed)
    samples = []
    for _ in range(bootstrap_count):
        samples.append(sum(bucket_values[rng.randrange(len(bucket_values))] for _ in bucket_values) / len(bucket_values))
    samples.sort()
    lower_index, upper_index = _ci_indices(bootstrap_count, confidence_level)
    return {
        "bootstrap_mean": sum(bucket_values) / len(bucket_values),
        "bootstrap_lower": samples[lower_index],
        "bootstrap_upper": samples[upper_index],
    }


def _bucket_bootstrap_slope_interval(
    rows: list[dict[str, Any]],
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, float]:
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bucket[str(row["composition_key"])].append(row)
    buckets = sorted(by_bucket)
    rng = random.Random(bootstrap_seed)
    slopes = []
    for _ in range(bootstrap_count):
        sampled = []
        for _ in buckets:
            sampled.extend(by_bucket[buckets[rng.randrange(len(buckets))]])
        slopes.append(_linear_slope([row["posterior_js"] for row in sampled], [row["representation_js"] for row in sampled]))
    slopes.sort()
    lower_index, upper_index = _ci_indices(bootstrap_count, confidence_level)
    point = _linear_slope([row["posterior_js"] for row in rows], [row["representation_js"] for row in rows])
    return {
        "point": point,
        "lower": slopes[lower_index],
        "upper": slopes[upper_index],
    }


def _posterior_gap_bins(rows: list[dict[str, Any]]) -> list[dict[str, float | str | int]]:
    if not rows:
        return []
    sorted_rows = sorted(rows, key=lambda row: float(row["posterior_js"]))
    labels = ("low_gap", "mid_gap", "high_gap")
    output = []
    for index, label in enumerate(labels):
        start = int(index * len(sorted_rows) / len(labels))
        end = int((index + 1) * len(sorted_rows) / len(labels))
        chunk = sorted_rows[start:end]
        output.append(
            {
                "bin": label,
                "pair_count": len(chunk),
                "mean_posterior_js": _mean([float(row["posterior_js"]) for row in chunk]),
                "mean_representation_js": _mean([float(row["representation_js"]) for row in chunk]),
            }
        )
    return output


def _bucket_values(rows: list[dict[str, Any]], value_key: str) -> list[float]:
    by_bucket: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_bucket[str(row["composition_key"])].append(float(row[value_key]))
    return [_mean(values) for _, values in sorted(by_bucket.items())]


def _bucket_mean(rows: list[dict[str, Any]], value_key: str) -> float:
    return _mean(_bucket_values(rows, value_key)) if rows else 0.0


def _ci_indices(count: int, confidence_level: float) -> tuple[int, int]:
    if count <= 0:
        raise ValueError("bootstrap count must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    lower = int(((1.0 - confidence_level) / 2.0) * (count - 1))
    upper = int((1.0 - (1.0 - confidence_level) / 2.0) * (count - 1))
    return lower, upper


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator <= 0.0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator


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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _top_index(values: tuple[float, ...]) -> int:
    return int(max(range(len(values)), key=lambda index: values[index]))


def _manifest(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": payload["version"],
        "report_type": "k.5.0_synthetic_cbps_manifest",
        "claim_scope": payload["claim_scope"],
        "output_dir": str(output_dir),
        "expected_outputs": [
            "k.5.0_synthetic_cbps_report.json",
            "k.5.0_synthetic_cbps_report.md",
            "k.5.0_synthetic_cbps_manifest.json",
        ],
        "bootstrap_count": payload["bootstrap_count"],
        "bootstrap_seed": payload["bootstrap_seed"],
        "report_checksum": payload["checksum"],
        "case_labels": {case["mode"]: case["final_diagnostic_label"] for case in payload["cases"]},
        "synthetic_cbps_control_passed": payload["synthetic_cbps_control_passed"],
        "one_command_reproduction": (
            ".venv/bin/python scripts/build_synthetic_cbps_report.py "
            f"--output-dir {output_dir} "
            f"--sequence-count {payload['sequence_count']} "
            f"--length {payload['length']} "
            f"--max-pairs {payload['max_pairs']} "
            f"--bootstrap-count {payload['bootstrap_count']} "
            f"--bootstrap-seed {payload['bootstrap_seed']}"
        ),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# k.5.0 Synthetic CBPS Control Report",
        "",
        "This is a known-control diagnostic-label validation. It does not upgrade the poker AR claim.",
        "",
        f"synthetic_cbps_control_passed: `{payload['synthetic_cbps_control_passed']}`",
        "",
        "| mode | verdict | action delta vs C_min | value delta vs C_min | label delta | matched repr JS | repr lower | repr upper | posterior-gap slope | slope lower | slope upper |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        predictive = case["predictive_metrics"]
        latent = case["latent_metrics"]
        bounded = case["composition_boundedness"]
        interval = bounded["representation_js_ci"]
        slope = bounded["matched_posterior_gap_slope_ci"]
        lines.append(
            "| {mode} | {label} | {action_delta:.6f} | {value_delta:.6f} | {label_delta:.6f} | {repr_js:.6f} | {lower:.6f} | {upper:.6f} | {slope_point:.6f} | {slope_lower:.6f} | {slope_upper:.6f} |".format(
                mode=case["mode"],
                label=case["final_diagnostic_label"],
                action_delta=predictive["predictive_action_delta_vs_C_min"],
                value_delta=predictive["predictive_value_delta_vs_C_min"],
                label_delta=latent["label_permutation_delta"],
                repr_js=bounded["mean_representation_js"],
                lower=interval["bootstrap_lower"],
                upper=interval["bootstrap_upper"],
                slope_point=slope["point"],
                slope_lower=slope["lower"],
                slope_upper=slope["upper"],
            )
        )
    return "\n".join(lines) + "\n"
