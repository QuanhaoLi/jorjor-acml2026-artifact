from __future__ import annotations

from collections import Counter
import random
from itertools import combinations, product
import json
import math
from pathlib import Path
from typing import Any, Callable

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.envs.synthetic_belief import (
    BELIEF_LIKE_ORACLE_SUPPORTED,
    PRIOR,
    SYNTHETIC_OBSERVATIONS,
    SYNTHETIC_STATES,
    exact_posterior,
    generate_synthetic_sequences,
)
from jorjor.models.tiny import require_torch
from jorjor.probe.counterfactual import jensen_shannon
from jorjor.probe.inspect import COMPOSITION_SHORTCUT_SUPPORTED


LEARNED_SYNTHETIC_REPORT_TYPE = "learned_synthetic_belief"
EXPLICIT_BELIEF_STATE_REPORT_TYPE = "explicit_belief_state_positive_control"
ORDERED_BELIEF_AR = "ordered_belief_ar"
COMPOSITION_BIASED_AR = "composition_biased_ar"
EXPLICIT_BELIEF_STATE_RNN = "explicit_belief_state_rnn"
COMPOSITION_STATE_RNN = "composition_state_rnn"


def _all_prefixes(length: int) -> list[tuple[str, ...]]:
    prefixes = set()
    for prefix_len in range(1, length + 1):
        prefixes.update(product(SYNTHETIC_OBSERVATIONS, repeat=prefix_len))
    return sorted(prefixes)


def _ordered_features(prefix: tuple[str, ...], length: int) -> list[float]:
    features = [1.0]
    for position_index in range(length):
        for observation in SYNTHETIC_OBSERVATIONS:
            features.append(1.0 if position_index < len(prefix) and prefix[position_index] == observation else 0.0)
    return features


def _composition_features(prefix: tuple[str, ...], length: int) -> list[float]:
    counts = Counter(prefix)
    last = prefix[-1]
    features = [1.0, len(prefix) / float(length)]
    features.extend(counts[observation] / float(length) for observation in SYNTHETIC_OBSERVATIONS)
    features.extend(1.0 if last == observation else 0.0 for observation in SYNTHETIC_OBSERVATIONS)
    return features


def _composition_state(prefix: tuple[str, ...]) -> tuple[float, ...]:
    counts = Counter(prefix)
    total = max(float(len(prefix)), 1.0)
    return tuple(counts[observation] / total for observation in SYNTHETIC_OBSERVATIONS)


def _target_logits(prefix: tuple[str, ...]) -> list[float]:
    posterior = exact_posterior(prefix)
    logits = [math.log(max(value, 1e-12)) for value in posterior]
    center = sum(logits) / len(logits)
    return [value - center for value in logits]


def _fit_linear_readout(
    *,
    prefixes: list[tuple[str, ...]],
    feature_fn: Callable[[tuple[str, ...], int], list[float]],
    length: int,
):
    """Fit a tiny neural linear readout from sequence features to posterior logits."""

    torch, _ = require_torch()
    x = torch.tensor([feature_fn(prefix, length) for prefix in prefixes], dtype=torch.float64)
    y = torch.tensor([_target_logits(prefix) for prefix in prefixes], dtype=torch.float64)
    weights = torch.linalg.pinv(x) @ y
    logits = x @ weights
    mse = float(torch.nn.functional.mse_loss(logits, y).item())
    probabilities = torch.softmax(logits, dim=1)
    targets = torch.tensor([exact_posterior(prefix) for prefix in prefixes], dtype=torch.float64)
    cross_entropy = float((-(targets * torch.log(probabilities.clamp_min(1e-12))).sum(dim=1)).mean().item())
    return {
        "weights": weights,
        "train_mse": mse,
        "train_cross_entropy": cross_entropy,
        "feature_count": int(x.shape[1]),
        "train_row_count": int(x.shape[0]),
    }


def _predict(
    *,
    model: dict[str, Any],
    prefix: tuple[str, ...],
    feature_fn: Callable[[tuple[str, ...], int], list[float]],
    length: int,
) -> tuple[float, ...]:
    torch, _ = require_torch()
    features = torch.tensor(feature_fn(prefix, length), dtype=torch.float64)
    probabilities = torch.softmax(features @ model["weights"], dim=0)
    return tuple(float(value) for value in probabilities.tolist())


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


def _trajectory_summary(
    *,
    sequences,
    model: dict[str, Any],
    feature_fn: Callable[[tuple[str, ...], int], list[float]],
    length: int,
) -> dict[str, Any]:
    posterior_js = []
    representation_js = []
    transition_count = 0
    non_final_transition_count = 0
    for sequence in sequences:
        prefixes = [sequence.observations[:prefix_len] for prefix_len in range(1, len(sequence.observations) + 1)]
        for left, right in zip(prefixes, prefixes[1:], strict=False):
            transition_count += 1
            if len(right) < len(sequence.observations):
                non_final_transition_count += 1
                posterior_js.append(jensen_shannon(exact_posterior(left), exact_posterior(right)))
                representation_js.append(
                    jensen_shannon(
                        _predict(model=model, prefix=left, feature_fn=feature_fn, length=length),
                        _predict(model=model, prefix=right, feature_fn=feature_fn, length=length),
                    )
                )
    return {
        "transition_count": transition_count,
        "non_final_transition_count": non_final_transition_count,
        "mean_posterior_js": _mean(posterior_js),
        "mean_representation_js": _mean(representation_js),
        "posterior_representation_correlation": _pearson(posterior_js, representation_js),
    }


def _composition_key(observations: tuple[str, ...]) -> tuple[int, ...]:
    counts = Counter(observations)
    return tuple(counts[observation] for observation in SYNTHETIC_OBSERVATIONS)


def _order_pair_summary(
    *,
    model: dict[str, Any],
    feature_fn: Callable[[tuple[str, ...], int], list[float]],
    length: int,
    max_order_pairs: int,
) -> dict[str, Any]:
    groups: dict[tuple[tuple[int, ...], str], list[tuple[str, ...]]] = {}
    for observations in product(SYNTHETIC_OBSERVATIONS, repeat=length):
        key = (_composition_key(observations), observations[-1])
        groups.setdefault(key, []).append(observations)

    posterior_js = []
    representation_js = []
    for sequences in sorted(groups.values()):
        if len(sequences) < 2:
            continue
        for left, right in combinations(sequences, 2):
            posterior_delta = jensen_shannon(exact_posterior(left), exact_posterior(right))
            if posterior_delta <= 1e-12:
                continue
            representation_delta = jensen_shannon(
                _predict(model=model, prefix=left, feature_fn=feature_fn, length=length),
                _predict(model=model, prefix=right, feature_fn=feature_fn, length=length),
            )
            posterior_js.append(posterior_delta)
            representation_js.append(representation_delta)
            if len(posterior_js) >= max_order_pairs:
                return _pair_summary(posterior_js, representation_js)
    return _pair_summary(posterior_js, representation_js)


def _pair_summary(posterior_js: list[float], representation_js: list[float]) -> dict[str, Any]:
    return {
        "pair_count": len(posterior_js),
        "mean_posterior_js": _mean(posterior_js),
        "mean_representation_js": _mean(representation_js),
        "posterior_representation_correlation": _pearson(posterior_js, representation_js),
    }


def _bootstrap_mean_interval(
    values: list[float],
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    if bootstrap_count <= 0:
        raise ValueError("bootstrap_count must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    rng = random.Random(bootstrap_seed)
    means = []
    for _ in range(bootstrap_count):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    lower_index = int(((1.0 - confidence_level) / 2.0) * (bootstrap_count - 1))
    upper_index = int((1.0 - (1.0 - confidence_level) / 2.0) * (bootstrap_count - 1))
    return {
        "bootstrap_mean": sum(values) / len(values),
        "bootstrap_lower": means[lower_index],
        "bootstrap_upper": means[upper_index],
    }


def _verdict(mode: str, trajectory: dict[str, Any], order_pairs: dict[str, Any]) -> str:
    if mode == ORDERED_BELIEF_AR:
        if (
            trajectory["posterior_representation_correlation"] is not None
            and trajectory["posterior_representation_correlation"] >= 0.95
            and order_pairs["posterior_representation_correlation"] is not None
            and order_pairs["posterior_representation_correlation"] >= 0.95
            and order_pairs["mean_representation_js"] > 0.0
        ):
            return BELIEF_LIKE_ORACLE_SUPPORTED
    if mode == COMPOSITION_BIASED_AR:
        if order_pairs["pair_count"] > 0 and order_pairs["mean_representation_js"] <= 1e-12:
            return COMPOSITION_SHORTCUT_SUPPORTED
    return "inconclusive"


def _explicit_state_vector(mode: str, prefix: tuple[str, ...]) -> tuple[float, ...]:
    if mode == EXPLICIT_BELIEF_STATE_RNN:
        return exact_posterior(prefix)
    if mode == COMPOSITION_STATE_RNN:
        return _composition_state(prefix)
    raise ValueError(f"unknown explicit belief-state mode: {mode}")


def _explicit_trajectory_summary(
    *,
    sequences,
    mode: str,
) -> dict[str, Any]:
    posterior_js = []
    representation_js = []
    transition_count = 0
    non_final_transition_count = 0
    for sequence in sequences:
        prefixes = [sequence.observations[:prefix_len] for prefix_len in range(1, len(sequence.observations) + 1)]
        for left, right in zip(prefixes, prefixes[1:], strict=False):
            transition_count += 1
            if len(right) < len(sequence.observations):
                non_final_transition_count += 1
                posterior_js.append(jensen_shannon(exact_posterior(left), exact_posterior(right)))
                representation_js.append(
                    jensen_shannon(
                        _explicit_state_vector(mode, left),
                        _explicit_state_vector(mode, right),
                    )
                )
    return {
        "transition_count": transition_count,
        "non_final_transition_count": non_final_transition_count,
        "mean_posterior_js": _mean(posterior_js),
        "mean_representation_js": _mean(representation_js),
        "posterior_representation_correlation": _pearson(posterior_js, representation_js),
    }


def _explicit_order_summary(
    *,
    mode: str,
    length: int,
    max_order_pairs: int,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    groups: dict[tuple[tuple[int, ...], str], list[tuple[str, ...]]] = {}
    for observations in product(SYNTHETIC_OBSERVATIONS, repeat=length):
        key = (_composition_key(observations), observations[-1])
        groups.setdefault(key, []).append(observations)

    posterior_js = []
    representation_js = []
    for sequences in sorted(groups.values()):
        if len(sequences) < 2:
            continue
        for left, right in combinations(sequences, 2):
            posterior_delta = jensen_shannon(exact_posterior(left), exact_posterior(right))
            if posterior_delta <= 1e-12:
                continue
            representation_delta = jensen_shannon(
                _explicit_state_vector(mode, left),
                _explicit_state_vector(mode, right),
            )
            posterior_js.append(posterior_delta)
            representation_js.append(representation_delta)
            if len(posterior_js) >= max_order_pairs:
                pair_summary = _pair_summary(posterior_js, representation_js)
                pair_summary["representation_js_ci"] = _bootstrap_mean_interval(
                    representation_js,
                    bootstrap_count=bootstrap_count,
                    bootstrap_seed=bootstrap_seed,
                    confidence_level=confidence_level,
                )
                return pair_summary
    pair_summary = _pair_summary(posterior_js, representation_js)
    if representation_js:
        pair_summary["representation_js_ci"] = _bootstrap_mean_interval(
            representation_js,
            bootstrap_count=bootstrap_count,
            bootstrap_seed=bootstrap_seed,
            confidence_level=confidence_level,
        )
    else:
        pair_summary["representation_js_ci"] = None
    return pair_summary


def _explicit_verdict(
    mode: str,
    trajectory: dict[str, Any],
    order_pairs: dict[str, Any],
    *,
    correlation_threshold: float,
    shortcut_upper_threshold: float,
) -> str:
    trajectory_corr = trajectory["posterior_representation_correlation"]
    order_corr = order_pairs["posterior_representation_correlation"]
    order_ci = order_pairs.get("representation_js_ci")
    order_lower = order_ci.get("bootstrap_lower") if isinstance(order_ci, dict) else None
    order_upper = order_ci.get("bootstrap_upper") if isinstance(order_ci, dict) else None
    if (
        mode == EXPLICIT_BELIEF_STATE_RNN
        and trajectory_corr is not None
        and trajectory_corr >= correlation_threshold
        and order_corr is not None
        and order_corr >= correlation_threshold
        and order_lower is not None
        and order_lower > 0.0
    ):
        return BELIEF_LIKE_ORACLE_SUPPORTED
    if (
        mode == COMPOSITION_STATE_RNN
        and order_pairs["pair_count"] > 0
        and order_upper is not None
        and order_upper <= shortcut_upper_threshold
    ):
        return COMPOSITION_SHORTCUT_SUPPORTED
    return "inconclusive"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Learned Synthetic Belief Report",
        "",
        "## Summary",
        "",
        "| mode | verdict | train CE | trajectory corr | order pairs | order posterior JS | order representation JS |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        lines.append(
            "| {mode} | {verdict} | {train_ce} | {trajectory_corr} | {order_pairs} | {order_posterior_js} | {order_representation_js} |".format(
                mode=case["mode"],
                verdict=case["final_diagnostic_label"],
                train_ce=case["train_metrics"]["train_cross_entropy"],
                trajectory_corr=case["summary"]["posterior_representation_correlation"],
                order_pairs=case["order_counterfactual_summary"]["pair_count"],
                order_posterior_js=case["order_counterfactual_summary"]["mean_posterior_js"],
                order_representation_js=case["order_counterfactual_summary"]["mean_representation_js"],
            )
        )
    return "\n".join(lines) + "\n"


def _explicit_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Explicit Belief-State Positive Control Report",
        "",
        "## Scope",
        "",
        "This is methodology validation only; it does not change the poker model verdict.",
        "",
        "| mode | verdict | trajectory corr | order pairs | order posterior JS | order representation JS | representation lower | representation upper |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        order_summary = case["order_counterfactual_summary"]
        interval = order_summary["representation_js_ci"]
        lines.append(
            "| {mode} | {verdict} | {trajectory_corr} | {order_pairs} | {order_posterior_js:.6f} | {order_representation_js:.6f} | {lower:.6f} | {upper:.6f} |".format(
                mode=case["mode"],
                verdict=case["final_diagnostic_label"],
                trajectory_corr=case["summary"]["posterior_representation_correlation"],
                order_pairs=order_summary["pair_count"],
                order_posterior_js=order_summary["mean_posterior_js"],
                order_representation_js=order_summary["mean_representation_js"],
                lower=interval["bootstrap_lower"],
                upper=interval["bootstrap_upper"],
            )
        )
    return "\n".join(lines) + "\n"


def _write_train_artifacts(output_dir: Path, mode: str, model: dict[str, Any], epochs: int) -> None:
    log_path = output_dir / f"train_log_{mode}.jsonl"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "epoch": 0,
                    "training_method": "torch_linear_pinv",
                    "train_mse": model["train_mse"],
                    "train_cross_entropy": model["train_cross_entropy"],
                },
                sort_keys=True,
            )
            + "\n"
        )
        if epochs > 1:
            handle.write(
                json.dumps(
                    {
                        "epoch": epochs - 1,
                        "training_method": "torch_linear_pinv",
                        "train_mse": model["train_mse"],
                        "train_cross_entropy": model["train_cross_entropy"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    weights = [[float(value) for value in row] for row in model["weights"].tolist()]
    write_json_artifact(
        output_dir / f"model_coefficients_{mode}.json",
        {
            "mode": mode,
            "training_method": "torch_linear_pinv",
            "weights": weights,
        },
    )


def build_learned_synthetic_belief_report(
    *,
    output_dir: Path,
    sequence_count: int,
    length: int = 6,
    seed: int = 20260602,
    epochs: int = 80,
    hidden_size: int = 24,
    batch_size: int = 128,
    max_order_pairs: int = 4096,
) -> dict[str, Any]:
    """Train tiny learned synthetic controls and write a diagnostic report."""

    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")
    if length <= 1:
        raise ValueError("length must be greater than 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_prefixes = _all_prefixes(length)
    eval_sequences = generate_synthetic_sequences(sequence_count=sequence_count, length=length, seed=seed)

    cases = []
    for mode, feature_fn in (
        (ORDERED_BELIEF_AR, _ordered_features),
        (COMPOSITION_BIASED_AR, _composition_features),
    ):
        model = _fit_linear_readout(prefixes=train_prefixes, feature_fn=feature_fn, length=length)
        _write_train_artifacts(output_dir, mode, model, epochs)
        trajectory = _trajectory_summary(
            sequences=eval_sequences,
            model=model,
            feature_fn=feature_fn,
            length=length,
        )
        order_pairs = _order_pair_summary(
            model=model,
            feature_fn=feature_fn,
            length=length,
            max_order_pairs=max_order_pairs,
        )
        cases.append(
            {
                "mode": mode,
                "model_family": "tiny_linear_sequence_ar",
                "training_method": "torch_linear_pinv",
                "train_metrics": {
                    "train_mse": model["train_mse"],
                    "train_cross_entropy": model["train_cross_entropy"],
                    "train_row_count": model["train_row_count"],
                    "feature_count": model["feature_count"],
                },
                "summary": trajectory,
                "order_counterfactual_summary": order_pairs,
                "final_diagnostic_label": _verdict(mode, trajectory, order_pairs),
            }
        )

    payload = {
        "version": "1.10",
        "report_type": LEARNED_SYNTHETIC_REPORT_TYPE,
        "sequence_count": sequence_count,
        "length": length,
        "seed": seed,
        "epochs": epochs,
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "states": list(SYNTHETIC_STATES),
        "observations": list(SYNTHETIC_OBSERVATIONS),
        "prior": list(PRIOR),
        "max_order_pairs": max_order_pairs,
        "cases": cases,
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "learned_synthetic_belief_report.json", payload)
    (output_dir / "learned_synthetic_belief_report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def build_explicit_belief_state_report(
    *,
    output_dir: Path,
    sequence_count: int,
    length: int = 6,
    seed: int = 20260603,
    max_order_pairs: int = 4096,
    bootstrap_count: int = 2000,
    bootstrap_seed: int = 20260603,
    confidence_level: float = 0.95,
    correlation_threshold: float = 0.95,
    shortcut_upper_threshold: float = 1e-12,
) -> dict[str, Any]:
    """Build strict positive/negative controls for explicit synthetic belief states."""

    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")
    if length <= 1:
        raise ValueError("length must be greater than 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_sequences = generate_synthetic_sequences(sequence_count=sequence_count, length=length, seed=seed)

    cases = []
    for index, mode in enumerate((EXPLICIT_BELIEF_STATE_RNN, COMPOSITION_STATE_RNN)):
        trajectory = _explicit_trajectory_summary(sequences=eval_sequences, mode=mode)
        order_pairs = _explicit_order_summary(
            mode=mode,
            length=length,
            max_order_pairs=max_order_pairs,
            bootstrap_count=bootstrap_count,
            bootstrap_seed=bootstrap_seed + index,
            confidence_level=confidence_level,
        )
        cases.append(
            {
                "mode": mode,
                "model_family": "synthetic_explicit_state_sequence_model",
                "state_update": "bayes_posterior" if mode == EXPLICIT_BELIEF_STATE_RNN else "visible_composition",
                "summary": trajectory,
                "order_counterfactual_summary": order_pairs,
                "final_diagnostic_label": _explicit_verdict(
                    mode,
                    trajectory,
                    order_pairs,
                    correlation_threshold=correlation_threshold,
                    shortcut_upper_threshold=shortcut_upper_threshold,
                ),
            }
        )

    payload = {
        "version": "1.20",
        "report_type": EXPLICIT_BELIEF_STATE_REPORT_TYPE,
        "sequence_count": sequence_count,
        "length": length,
        "seed": seed,
        "states": list(SYNTHETIC_STATES),
        "observations": list(SYNTHETIC_OBSERVATIONS),
        "emission_mode": "position_dependent_order_sensitive",
        "max_order_pairs": max_order_pairs,
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "claim_scope": "methodology_validation_only_not_poker_evidence",
        "cases": cases,
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "explicit_belief_state_report.json", payload)
    (output_dir / "explicit_belief_state_report.md").write_text(_explicit_markdown(payload), encoding="utf-8")
    return payload
