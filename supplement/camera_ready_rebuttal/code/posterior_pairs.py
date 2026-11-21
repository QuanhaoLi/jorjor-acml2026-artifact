"""Matched-pair construction and statistics for the exact-posterior poker audit.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
import random
from typing import Any

from jorjor.models.tiny import require_torch
from jorjor.tokenizer import ACTIONS, LIMIT_ROUNDS


__all__ = (
    "batched_jensen_shannon",
    "build_fixed_matched_pairs",
    "dyadic_alignment_bootstrap",
    "dyadic_hand_node_bootstrap",
    "pair_alignment_metrics",
    "paired_mae_improvement_bootstrap",
    "predecision_composition_key",
)


def predecision_composition_key(point) -> str:
    """Return a deterministic composition key containing pre-decision fields only."""

    state = point.visible_state
    acting_player = str(state.get("acting_player", point.acting_player))
    street = str(state.get("round", ""))
    legal_actions = tuple(sorted(str(action) for action in state.get("legal_actions", point.legal_actions)))
    events = tuple(event for event in state.get("betting_history", ()) if event.get("type") == "action")
    actions = tuple(str(event.get("action", "")) for event in events)
    unknown_actions = sorted(set(actions) - set(ACTIONS))
    if unknown_actions:
        raise ValueError(f"unknown betting action(s): {', '.join(unknown_actions)}")

    observed_streets = {str(event.get("round", "")) for event in events}
    if "" in observed_streets:
        raise ValueError("action event is missing round")
    street_order = (*LIMIT_ROUNDS, *sorted(observed_streets - set(LIMIT_ROUNDS)))
    relation_counts: dict[tuple[str, str, str], int] = Counter()
    for event in events:
        if "player" not in event:
            raise ValueError("action event is missing player")
        relation = "self" if str(event["player"]) == acting_player else "opponent"
        relation_counts[(str(event["round"]), relation, str(event["action"]))] += 1

    payload: dict[str, Any] = {
        "actor": acting_player,
        "street": street,
        "legal": legal_actions,
        "history_length": len(actions),
        "action_counts": {action: actions.count(action) for action in ACTIONS},
        "last_action": actions[-1] if actions else "none",
        "street_actor_action_counts": [
            [round_name, relation, action, relation_counts[(round_name, relation, action)]]
            for round_name in street_order
            for relation in ("self", "opponent")
            for action in ACTIONS
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def build_fixed_matched_pairs(
    points,
    *,
    max_pairs_per_bucket: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Select fixed same-composition pairs without within-hand edges."""

    if max_pairs_per_bucket is not None and max_pairs_per_bucket <= 0:
        raise ValueError("max_pairs_per_bucket must be positive or None")
    grouped: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    point_ids: set[str] = set()
    for index, point in enumerate(points):
        hand_id = str(point.hand_id)
        point_id = f"{hand_id}:{int(point.step_index)}"
        if point_id in point_ids:
            raise ValueError(f"duplicate point id: {point_id}")
        point_ids.add(point_id)
        grouped[predecision_composition_key(point)].append((point_id, hand_id, index))

    rows: list[dict[str, Any]] = []
    for bucket, raw_entries in sorted(grouped.items()):
        entries = sorted(raw_entries)
        bucket_seed = _stable_seed(f"{seed}|{bucket}")
        rng = random.Random(bucket_seed)
        selected: list[dict[str, Any]] = []
        eligible_count = 0
        for left, right in itertools.combinations(entries, 2):
            left_point_id, left_hand_id, left_index = left
            right_point_id, right_hand_id, right_index = right
            if left_hand_id == right_hand_id:
                continue
            eligible_count += 1
            replacement_index = None
            if max_pairs_per_bucket is not None and len(selected) >= max_pairs_per_bucket:
                candidate_index = rng.randrange(eligible_count)
                if candidate_index >= max_pairs_per_bucket:
                    continue
                replacement_index = candidate_index
            row = {
                "bucket": bucket,
                "composition_key": bucket,
                "left_index": left_index,
                "right_index": right_index,
                "left_point_id": left_point_id,
                "right_point_id": right_point_id,
                "left_hand_id": left_hand_id,
                "right_hand_id": right_hand_id,
            }
            if replacement_index is None:
                selected.append(row)
            else:
                selected[replacement_index] = row
        rows.extend(sorted(selected, key=_pair_sort_key))
    return rows


def batched_jensen_shannon(left, right, *, batch_size: int = 4096):
    """Return row-wise Jensen-Shannon divergence in nats as a CPU float64 tensor."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    torch, _ = require_torch()
    left_tensor = torch.as_tensor(left, dtype=torch.float64)
    right_tensor = torch.as_tensor(right, dtype=torch.float64, device=left_tensor.device)
    if left_tensor.ndim != 2 or right_tensor.ndim != 2:
        raise ValueError("probability batches must be two-dimensional")
    if left_tensor.shape != right_tensor.shape:
        raise ValueError("probability batches must have the same shape")
    if left_tensor.shape[1] == 0:
        raise ValueError("probability vectors must contain at least one class")
    if not bool(torch.isfinite(left_tensor).all()) or not bool(torch.isfinite(right_tensor).all()):
        raise ValueError("probability batches must be finite")
    if bool((left_tensor < 0.0).any()) or bool((right_tensor < 0.0).any()):
        raise ValueError("probability batches must be non-negative")
    if left_tensor.shape[0] == 0:
        return torch.empty((0,), dtype=torch.float64)

    left_sums = left_tensor.sum(dim=1, keepdim=True)
    right_sums = right_tensor.sum(dim=1, keepdim=True)
    ones = torch.ones_like(left_sums)
    if not bool(torch.allclose(left_sums, ones, atol=1e-6, rtol=1e-6)):
        raise ValueError("left probability rows must sum to one")
    if not bool(torch.allclose(right_sums, ones, atol=1e-6, rtol=1e-6)):
        raise ValueError("right probability rows must sum to one")
    left_tensor = left_tensor / left_sums
    right_tensor = right_tensor / right_sums

    output = []
    tiny = torch.finfo(torch.float64).tiny
    for start in range(0, left_tensor.shape[0], batch_size):
        stop = min(start + batch_size, left_tensor.shape[0])
        left_batch = left_tensor[start:stop]
        right_batch = right_tensor[start:stop]
        middle = 0.5 * (left_batch + right_batch)
        left_term = torch.where(
            left_batch > 0.0,
            left_batch * (torch.log(left_batch.clamp_min(tiny)) - torch.log(middle.clamp_min(tiny))),
            torch.zeros_like(left_batch),
        )
        right_term = torch.where(
            right_batch > 0.0,
            right_batch * (torch.log(right_batch.clamp_min(tiny)) - torch.log(middle.clamp_min(tiny))),
            torch.zeros_like(right_batch),
        )
        divergence = 0.5 * (left_term.sum(dim=1) + right_term.sum(dim=1))
        output.append(divergence.clamp(0.0, math.log(2.0)).detach().cpu())
    return torch.cat(output, dim=0)


def pair_alignment_metrics(target_js, predicted_js) -> dict[str, float | int | None]:
    """Summarize calibration and association between exact and predicted pair JS."""

    target = _float_tuple(target_js, name="target_js")
    predicted = _float_tuple(predicted_js, name="predicted_js")
    if not target or len(target) != len(predicted):
        raise ValueError("target_js and predicted_js must be non-empty and aligned")
    errors = tuple(prediction - truth for truth, prediction in zip(target, predicted, strict=True))
    mean_target = math.fsum(target) / len(target)
    mean_predicted = math.fsum(predicted) / len(predicted)
    centered_target = tuple(value - mean_target for value in target)
    centered_predicted = tuple(value - mean_predicted for value in predicted)
    target_ss = math.fsum(value * value for value in centered_target)
    predicted_ss = math.fsum(value * value for value in centered_predicted)
    cross = math.fsum(
        left * right for left, right in zip(centered_target, centered_predicted, strict=True)
    )
    pearson = None
    target_varies = _has_numeric_variation(target)
    predicted_varies = _has_numeric_variation(predicted)
    if target_varies and predicted_varies:
        pearson = _clip_correlation(cross / math.sqrt(target_ss * predicted_ss))
    slope = cross / target_ss if target_varies else None
    intercept = mean_predicted - slope * mean_target if slope is not None else None
    ranks_target = _average_ranks(target)
    ranks_predicted = _average_ranks(predicted)
    spearman = _pearson(ranks_target, ranks_predicted)
    return {
        "pair_count": len(target),
        "mae": math.fsum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(math.fsum(error * error for error in errors) / len(errors)),
        "bias": math.fsum(errors) / len(errors),
        "pearson": pearson,
        "spearman": spearman,
        "slope": slope,
        "intercept": intercept,
    }


def dyadic_hand_node_bootstrap(
    values,
    left_hand_ids,
    right_hand_ids,
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float = 0.95,
) -> dict[str, float | int | str]:
    """Bootstrap a pair mean by resampling hand nodes and weighting incident edges."""

    numeric_values = _float_tuple(values, name="values")
    left_ids, right_ids = _validated_dyadic_ids(left_hand_ids, right_hand_ids, len(numeric_values))
    _validate_bootstrap_options(bootstrap_count, confidence_level)
    samples, hand_count, invalid_draw_count = _dyadic_mean_samples(
        numeric_values,
        left_ids,
        right_ids,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
    )
    lower, upper = _percentile_bounds(samples, confidence_level)
    return {
        "mean": math.fsum(numeric_values) / len(numeric_values),
        "bootstrap_mean": math.fsum(samples) / len(samples),
        "lower": lower,
        "upper": upper,
        "pair_count": len(numeric_values),
        "hand_count": hand_count,
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "invalid_draw_count": invalid_draw_count,
        "resampling_unit": "dyadic_hand_node",
        "resampling_method": "multinomial_node_resample_with_pair_multiplicity_product",
    }


def dyadic_alignment_bootstrap(
    target_js,
    predicted_js,
    left_hand_ids,
    right_hand_ids,
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Return alignment point metrics and dyadic hand-node bootstrap intervals."""

    target = _float_tuple(target_js, name="target_js")
    predicted = _float_tuple(predicted_js, name="predicted_js")
    if not target or len(target) != len(predicted):
        raise ValueError("target_js and predicted_js must be non-empty and aligned")
    left_ids, right_ids = _validated_dyadic_ids(left_hand_ids, right_hand_ids, len(target))
    _validate_bootstrap_options(bootstrap_count, confidence_level)
    samples, hand_count, invalid_draw_count = _dyadic_alignment_samples(
        target,
        predicted,
        left_ids,
        right_ids,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
    )
    intervals = {
        metric: _bootstrap_metric_interval(values, confidence_level)
        for metric, values in samples.items()
    }
    return {
        "point": pair_alignment_metrics(target, predicted),
        "intervals": intervals,
        "pair_count": len(target),
        "hand_count": hand_count,
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "invalid_draw_count": invalid_draw_count,
        "resampling_unit": "dyadic_hand_node",
        "resampling_method": "multinomial_node_resample_with_pair_multiplicity_product",
        "spearman_uncertainty": "point_estimate_only",
    }


def paired_mae_improvement_bootstrap(
    target_js,
    candidate_js,
    baseline_js,
    left_hand_ids,
    right_hand_ids,
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Compare candidate and baseline pair-JS MAE with a paired dyadic bootstrap."""

    target = _float_tuple(target_js, name="target_js")
    candidate = _float_tuple(candidate_js, name="candidate_js")
    baseline = _float_tuple(baseline_js, name="baseline_js")
    if not target or len(target) != len(candidate) or len(target) != len(baseline):
        raise ValueError("target, candidate, and baseline JS values must be non-empty and aligned")
    candidate_errors = tuple(abs(prediction - truth) for truth, prediction in zip(target, candidate, strict=True))
    baseline_errors = tuple(abs(prediction - truth) for truth, prediction in zip(target, baseline, strict=True))
    improvements = tuple(
        baseline_error - candidate_error
        for baseline_error, candidate_error in zip(baseline_errors, candidate_errors, strict=True)
    )
    interval = dyadic_hand_node_bootstrap(
        improvements,
        left_hand_ids,
        right_hand_ids,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    return {
        "baseline_mae": math.fsum(baseline_errors) / len(baseline_errors),
        "candidate_mae": math.fsum(candidate_errors) / len(candidate_errors),
        "mae_improvement": math.fsum(improvements) / len(improvements),
        "mae_improvement_ci": interval,
        "improvement_sign": "positive_favors_candidate",
    }


def _dyadic_alignment_samples(
    target: tuple[float, ...],
    predicted: tuple[float, ...],
    left_ids: tuple[str, ...],
    right_ids: tuple[str, ...],
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
) -> tuple[dict[str, tuple[float, ...]], int, int]:
    import numpy as np

    hands = tuple(sorted(set(left_ids) | set(right_ids)))
    hand_to_index = {hand_id: index for index, hand_id in enumerate(hands)}
    left_index = np.asarray([hand_to_index[hand_id] for hand_id in left_ids], dtype=np.int64)
    right_index = np.asarray([hand_to_index[hand_id] for hand_id in right_ids], dtype=np.int64)
    target_array = np.asarray(target, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    error_array = predicted_array - target_array
    rng = np.random.default_rng(bootstrap_seed)
    probabilities = np.full(len(hands), 1.0 / len(hands), dtype=np.float64)
    collected: dict[str, list[float]] = {
        "mae": [],
        "rmse": [],
        "bias": [],
        "pearson": [],
        "slope": [],
        "intercept": [],
    }
    accepted_draw_count = 0
    invalid_draw_count = 0
    max_draws = bootstrap_count * 100 + 1000
    draw_count = 0
    while accepted_draw_count < bootstrap_count and draw_count < max_draws:
        remaining = bootstrap_count - accepted_draw_count
        chunk_size = min(128, max(remaining * 2, 1))
        node_counts = rng.multinomial(len(hands), probabilities, size=chunk_size)
        pair_weights = node_counts[:, left_index] * node_counts[:, right_index]
        totals = pair_weights.sum(axis=1).astype(np.float64)
        valid = totals > 0.0
        if not bool(valid.any()):
            invalid_draw_count += chunk_size
            draw_count += chunk_size
            continue
        valid_positions = np.flatnonzero(valid)
        take_count = min(remaining, len(valid_positions))
        selected_positions = valid_positions[:take_count]
        consumed_count = (
            int(selected_positions[-1]) + 1 if take_count == remaining else chunk_size
        )
        invalid_draw_count += consumed_count - take_count
        draw_count += consumed_count
        weights = pair_weights[selected_positions].astype(np.float64)
        totals = totals[selected_positions]
        accepted = len(totals)
        accepted_draw_count += accepted

        means_target = (weights @ target_array) / totals
        means_predicted = (weights @ predicted_array) / totals
        means_target_squared = (weights @ (target_array * target_array)) / totals
        means_predicted_squared = (weights @ (predicted_array * predicted_array)) / totals
        means_cross = (weights @ (target_array * predicted_array)) / totals
        target_variance = np.maximum(0.0, means_target_squared - means_target * means_target)
        predicted_variance = np.maximum(0.0, means_predicted_squared - means_predicted * means_predicted)
        covariance = means_cross - means_target * means_predicted

        collected["mae"].extend(float(value) for value in (weights @ np.abs(error_array)) / totals)
        collected["rmse"].extend(float(value) for value in np.sqrt((weights @ (error_array * error_array)) / totals))
        collected["bias"].extend(float(value) for value in (weights @ error_array) / totals)
        variance_tolerance = np.finfo(np.float64).eps * 32.0
        defined_slope = target_variance > variance_tolerance * np.maximum(1.0, means_target_squared)
        slopes = np.full(accepted, np.nan, dtype=np.float64)
        slopes[defined_slope] = covariance[defined_slope] / target_variance[defined_slope]
        intercepts = means_predicted - slopes * means_target
        defined_pearson = defined_slope & (
            predicted_variance > variance_tolerance * np.maximum(1.0, means_predicted_squared)
        )
        correlations = np.full(accepted, np.nan, dtype=np.float64)
        correlations[defined_pearson] = covariance[defined_pearson] / np.sqrt(
            target_variance[defined_pearson] * predicted_variance[defined_pearson]
        )
        correlations[defined_pearson] = np.clip(correlations[defined_pearson], -1.0, 1.0)
        collected["pearson"].extend(float(value) for value in correlations if np.isfinite(value))
        collected["slope"].extend(float(value) for value in slopes if np.isfinite(value))
        collected["intercept"].extend(float(value) for value in intercepts if np.isfinite(value))
    if accepted_draw_count != bootstrap_count:
        raise RuntimeError("could not obtain enough non-empty dyadic bootstrap draws")
    return {metric: tuple(values) for metric, values in collected.items()}, len(hands), invalid_draw_count


def _bootstrap_metric_interval(values: tuple[float, ...], confidence_level: float) -> dict[str, float | int | None]:
    if not values:
        return {
            "bootstrap_mean": None,
            "lower": None,
            "upper": None,
            "effective_bootstrap_count": 0,
        }
    lower, upper = _percentile_bounds(values, confidence_level)
    return {
        "bootstrap_mean": math.fsum(values) / len(values),
        "lower": lower,
        "upper": upper,
        "effective_bootstrap_count": len(values),
    }


def _validated_dyadic_ids(
    left_hand_ids,
    right_hand_ids,
    expected_count: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    left = tuple(str(value) for value in left_hand_ids)
    right = tuple(str(value) for value in right_hand_ids)
    if expected_count <= 0 or len(left) != expected_count or len(right) != expected_count:
        raise ValueError("values and dyadic hand ids must be non-empty and aligned")
    if any(left_id == right_id for left_id, right_id in zip(left, right, strict=True)):
        raise ValueError("dyadic bootstrap requires cross-hand pairs")
    return left, right


def _validate_bootstrap_options(bootstrap_count: int, confidence_level: float) -> None:
    if bootstrap_count <= 0:
        raise ValueError("bootstrap_count must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")


def _dyadic_mean_samples(
    values: tuple[float, ...],
    left_ids: tuple[str, ...],
    right_ids: tuple[str, ...],
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
) -> tuple[tuple[float, ...], int, int]:
    import numpy as np

    hands = tuple(sorted(set(left_ids) | set(right_ids)))
    if len(hands) < 2:
        raise ValueError("dyadic bootstrap requires at least two hand nodes")
    hand_to_index = {hand_id: index for index, hand_id in enumerate(hands)}
    left_index = np.asarray([hand_to_index[hand_id] for hand_id in left_ids], dtype=np.int64)
    right_index = np.asarray([hand_to_index[hand_id] for hand_id in right_ids], dtype=np.int64)
    value_array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(bootstrap_seed)
    probabilities = np.full(len(hands), 1.0 / len(hands), dtype=np.float64)
    samples: list[float] = []
    invalid_draw_count = 0
    max_draws = bootstrap_count * 100 + 1000
    draw_count = 0
    while len(samples) < bootstrap_count and draw_count < max_draws:
        remaining = bootstrap_count - len(samples)
        chunk_size = min(128, max(remaining * 2, 1))
        node_counts = rng.multinomial(len(hands), probabilities, size=chunk_size)
        pair_weights = node_counts[:, left_index] * node_counts[:, right_index]
        weight_totals = pair_weights.sum(axis=1)
        valid = weight_totals > 0
        if bool(valid.any()):
            valid_positions = np.flatnonzero(valid)
            take_count = min(remaining, len(valid_positions))
            selected_positions = valid_positions[:take_count]
            consumed_count = (
                int(selected_positions[-1]) + 1 if take_count == remaining else chunk_size
            )
            invalid_draw_count += consumed_count - take_count
            draw_count += consumed_count
            estimates = (
                pair_weights[selected_positions] @ value_array
            ) / weight_totals[selected_positions]
            samples.extend(float(value) for value in estimates)
        else:
            invalid_draw_count += chunk_size
            draw_count += chunk_size
    if len(samples) != bootstrap_count:
        raise RuntimeError("could not obtain enough non-empty dyadic bootstrap draws")
    return tuple(samples), len(hands), invalid_draw_count


def _percentile_bounds(samples: tuple[float, ...], confidence_level: float) -> tuple[float, float]:
    ordered = sorted(samples)
    alpha = (1.0 - confidence_level) / 2.0
    lower_index = int(alpha * (len(ordered) - 1))
    upper_index = int((1.0 - alpha) * (len(ordered) - 1))
    return ordered[lower_index], ordered[upper_index]


def _float_tuple(values, *, name: str) -> tuple[float, ...]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    try:
        output = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be one-dimensional numeric values") from exc
    if not all(math.isfinite(value) for value in output):
        raise ValueError(f"{name} must contain only finite values")
    return output


def _average_ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        average_rank = 0.5 * ((start + 1) + stop)
        for position in range(start, stop):
            ranks[ordered[position]] = average_rank
        start = stop
    return tuple(ranks)


def _has_numeric_variation(values: tuple[float, ...]) -> bool:
    return not math.isclose(min(values), max(values), rel_tol=1e-12, abs_tol=1e-15)


def _pearson(left: tuple[float, ...], right: tuple[float, ...]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    left_centered = tuple(value - left_mean for value in left)
    right_centered = tuple(value - right_mean for value in right)
    left_ss = math.fsum(value * value for value in left_centered)
    right_ss = math.fsum(value * value for value in right_centered)
    if left_ss <= 0.0 or right_ss <= 0.0:
        return None
    cross = math.fsum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered, strict=True)
    )
    return _clip_correlation(cross / math.sqrt(left_ss * right_ss))


def _clip_correlation(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _pair_sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["composition_key"]), str(row["left_point_id"]), str(row["right_point_id"])
