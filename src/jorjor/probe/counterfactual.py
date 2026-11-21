from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import itertools
import json
import math
from pathlib import Path
from typing import Any

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.config import load_limit_ar_train_config
from jorjor.data.split import SplitFractions, apply_split, build_split_manifest
from jorjor.models.ar import build_tiny_ar_model
from jorjor.models.tiny import require_torch
from jorjor.probe.inspect import COMPOSITION_SHORTCUT_SUPPORTED, INCONCLUSIVE
from jorjor.probe.limit_ar import _load_ar_checkpoint_with_vocab_compat
from jorjor.tokenizer import ACTIONS, LimitTokenizer
from jorjor.train.leduc_tiny import _features_to_tensors, _move_tensors, _resolve_device
from jorjor.train.limit_tiny import _load_points


ORDER_SENSITIVE_SUPPORTED = "order_sensitive_supported"
INSUFFICIENT_PAIRS = "insufficient_pairs"
TRAJECTORY_ALIGNED_SUPPORTED = "trajectory_aligned_supported"
TRAJECTORY_NOT_ALIGNED = "trajectory_not_aligned"


@dataclass(frozen=True)
class OrderGroup:
    """Decision points sharing visible composition but not necessarily order."""

    composition_key: str
    order_key: str
    point_indices: tuple[int, ...]
    posterior: tuple[float, ...]


@dataclass(frozen=True)
class CounterfactualPair:
    """A legal same-composition different-order pair of point groups."""

    composition_key: str
    left_order_key: str
    right_order_key: str
    left_indices: tuple[int, ...]
    right_indices: tuple[int, ...]
    posterior_js: float


def _action_events(point) -> list[str]:
    return [
        str(event["action"])
        for event in point.visible_state["betting_history"]
        if event.get("type") == "action"
    ]


def truncate_point_history(point, prefix_action_count: int):
    """Return a point copy with visible betting history truncated to a prefix."""

    if prefix_action_count < 0:
        raise ValueError("prefix_action_count must be non-negative")
    action_seen = 0
    truncated_history = []
    for event in point.visible_state["betting_history"]:
        if event.get("type") != "action":
            truncated_history.append(dict(event))
            continue
        if action_seen >= prefix_action_count:
            break
        truncated_history.append(dict(event))
        action_seen += 1
    visible_state = dict(point.visible_state)
    visible_state["betting_history"] = truncated_history
    return replace(point, visible_state=visible_state)


def composition_key(point) -> str:
    """Return a strict visible-composition key for legal order counterfactuals."""

    return composition_key_for_profile(point, "strict")


def composition_key_for_profile(point, profile: str) -> str:
    """Return a visible-composition key for one coverage profile."""

    state = point.visible_state
    actions = _action_events(point)
    counts = Counter(actions)
    public_cards = ",".join(str(card) for card in state["public_cards"])
    private_cards = ",".join(str(card) for card in state["private_cards"])
    legal = ",".join(sorted(str(action) for action in point.legal_actions))
    last_action = actions[-1] if actions else "none"
    value = "none" if point.terminal_return is None else str(int(point.terminal_return))
    parts = [
        f"actor={point.acting_player}",
        f"round={state['round']}",
        f"pot={state['pot']}",
        f"legal={legal}",
        f"len={len(actions)}",
        *(f"{action}={counts[action]}" for action in ACTIONS),
        f"last={last_action}",
    ]
    if profile in {"strict", "no_value", "no_private"}:
        parts.append(f"public={public_cards}")
    if profile in {"strict", "no_value"}:
        parts.append(f"private={private_cards}")
    if profile in {"strict", "no_private", "no_cards"}:
        parts.append(f"policy={point.action}")
        parts.append(f"value={value}")
    if profile not in {"strict", "no_private", "no_value", "no_cards", "core"}:
        raise ValueError(f"unknown composition coverage profile: {profile}")
    return "|".join(parts)


def order_key(point) -> str:
    """Return the ordered visible betting-action sequence."""

    actions = _action_events(point)
    return ",".join(actions) if actions else "empty"


def posterior_from_indices(indices: tuple[int, ...], labels: list[int], class_count: int) -> tuple[float, ...]:
    """Estimate an empirical range posterior from grouped labels."""

    if not indices:
        raise ValueError("posterior requires at least one index")
    counts = [0.0] * class_count
    for index in indices:
        counts[labels[index]] += 1.0
    total = float(len(indices))
    return tuple(value / total for value in counts)


def jensen_shannon(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Return Jensen-Shannon distance in bits for two probability vectors."""

    if len(left) != len(right):
        raise ValueError("probability vectors must have the same length")
    if not left:
        raise ValueError("probability vectors must be non-empty")
    middle = tuple((a + b) / 2.0 for a, b in zip(left, right, strict=True))
    return 0.5 * _kl_divergence(left, middle) + 0.5 * _kl_divergence(right, middle)


def _kl_divergence(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    total = 0.0
    for a, b in zip(left, right, strict=True):
        if a <= 0.0:
            continue
        if b <= 0.0:
            raise ValueError("right probability vector has zero mass where left is positive")
        total += a * math.log2(a / b)
    return total


def build_order_groups(points, labels: list[int], class_count: int, *, min_group_size: int) -> list[OrderGroup]:
    """Build empirical posterior groups by strict composition and order."""

    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, point in enumerate(points):
        grouped[(composition_key(point), order_key(point))].append(index)

    groups = []
    for (bucket, order), indices in sorted(grouped.items()):
        if len(indices) < min_group_size:
            continue
        group_indices = tuple(indices)
        groups.append(
            OrderGroup(
                composition_key=bucket,
                order_key=order,
                point_indices=group_indices,
                posterior=posterior_from_indices(group_indices, labels, class_count),
            )
        )
    return groups


def composition_coverage(points) -> list[dict[str, Any]]:
    """Summarize whether legal data supports order-varying composition buckets."""

    rows = []
    for profile in ("strict", "no_private", "no_value", "no_cards", "core"):
        buckets: dict[str, set[str]] = defaultdict(set)
        point_counts: dict[str, int] = defaultdict(int)
        for point in points:
            key = composition_key_for_profile(point, profile)
            buckets[key].add(order_key(point))
            point_counts[key] += 1
        varying_keys = [key for key, orders in buckets.items() if len(orders) >= 2]
        rows.append(
            {
                "profile": profile,
                "bucket_count": len(buckets),
                "order_varying_bucket_count": len(varying_keys),
                "order_pair_group_count": sum(
                    len(buckets[key]) * (len(buckets[key]) - 1) // 2
                    for key in varying_keys
                ),
                "points_in_order_varying_buckets": sum(point_counts[key] for key in varying_keys),
            }
        )
    return rows


def build_counterfactual_pairs(
    groups: list[OrderGroup],
    *,
    min_posterior_js: float,
    max_pairs: int | None,
) -> list[CounterfactualPair]:
    """Create same-composition different-order group pairs."""

    by_composition: dict[str, list[OrderGroup]] = defaultdict(list)
    for group in groups:
        by_composition[group.composition_key].append(group)

    pairs = []
    for bucket in sorted(by_composition):
        bucket_groups = sorted(by_composition[bucket], key=lambda group: group.order_key)
        for left, right in itertools.combinations(bucket_groups, 2):
            posterior_js = jensen_shannon(left.posterior, right.posterior)
            if posterior_js < min_posterior_js:
                continue
            pairs.append(
                CounterfactualPair(
                    composition_key=bucket,
                    left_order_key=left.order_key,
                    right_order_key=right.order_key,
                    left_indices=left.point_indices,
                    right_indices=right.point_indices,
                    posterior_js=posterior_js,
                )
            )
            if max_pairs is not None and len(pairs) >= max_pairs:
                return pairs
    return pairs


def _mean_tensor(tensor, indices: tuple[int, ...], torch):
    index_tensor = torch.tensor(indices, dtype=torch.long, device=tensor.device)
    return tensor.index_select(0, index_tensor).mean(dim=0)


def _cosine_distance(left, right, torch) -> float:
    similarity = torch.nn.functional.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0)).item()
    return float(1.0 - similarity)


def _l2_distance(left, right, torch) -> float:
    return float(torch.linalg.vector_norm(left - right).item())


def _softmax_js(left_logits, right_logits, torch) -> float:
    left = torch.nn.functional.softmax(left_logits, dim=-1).detach().cpu().tolist()
    right = torch.nn.functional.softmax(right_logits, dim=-1).detach().cpu().tolist()
    return jensen_shannon(tuple(float(value) for value in left), tuple(float(value) for value in right))


def _model_outputs(train_config, checkpoint_path: Path, points, layers: tuple[str, ...], device: str):
    torch, _ = require_torch()
    resolved_device = _resolve_device(device, torch)
    tokenizer = LimitTokenizer()
    features = [tokenizer.encode(point) for point in points]
    tensors = _move_tensors(_features_to_tensors(features, torch), resolved_device)
    model = build_tiny_ar_model(
        vocab_size=tokenizer.vocab_size,
        action_size=tokenizer.action_size,
        opponent_card_size=tokenizer.opponent_card_size,
        hidden_size=train_config.hidden_size,
        num_heads=train_config.num_heads,
        num_layers=train_config.num_layers,
    ).to(resolved_device)
    load_info = _load_ar_checkpoint_with_vocab_compat(model, checkpoint_path, torch, resolved_device)
    model.eval()
    with torch.no_grad():
        encoded = model.encode_layers(tensors["input_ids"])
        outputs = model(tensors["input_ids"])
    selected_layers = {}
    for layer in layers:
        if layer not in encoded:
            raise ValueError(f"unknown layer for order-counterfactual report: {layer}")
        selected_layers[layer] = encoded[layer].detach()
    return torch, resolved_device, selected_layers, outputs, load_info


def _encode_model_outputs(train_config, checkpoint_path: Path, points, layers: tuple[str, ...], device: str):
    return _model_outputs(train_config, checkpoint_path, points, layers, device)


def _pair_metrics(pair: CounterfactualPair, layers: dict[str, Any], outputs: dict[str, Any], torch) -> dict[str, Any]:
    layer_rows = []
    for layer_name, tensor in layers.items():
        left = _mean_tensor(tensor, pair.left_indices, torch)
        right = _mean_tensor(tensor, pair.right_indices, torch)
        layer_rows.append(
            {
                "layer": layer_name,
                "hidden_cosine_distance": _cosine_distance(left, right, torch),
                "hidden_l2_distance": _l2_distance(left, right, torch),
            }
        )

    left_policy = _mean_tensor(outputs["policy_logits"], pair.left_indices, torch)
    right_policy = _mean_tensor(outputs["policy_logits"], pair.right_indices, torch)
    left_range = _mean_tensor(outputs["opponent_card_logits"], pair.left_indices, torch)
    right_range = _mean_tensor(outputs["opponent_card_logits"], pair.right_indices, torch)
    left_value = _mean_tensor(outputs["value"], pair.left_indices, torch)
    right_value = _mean_tensor(outputs["value"], pair.right_indices, torch)
    return {
        "composition_key": pair.composition_key,
        "left_order_key": pair.left_order_key,
        "right_order_key": pair.right_order_key,
        "left_count": len(pair.left_indices),
        "right_count": len(pair.right_indices),
        "posterior_js": pair.posterior_js,
        "policy_js": _softmax_js(left_policy, right_policy, torch),
        "range_logit_js": _softmax_js(left_range, right_range, torch),
        "value_abs_delta": float(abs((left_value - right_value).item())),
        "layers": layer_rows,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate_pair_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    layer_names = sorted({layer["layer"] for row in rows for layer in row["layers"]})
    layer_summary = []
    for layer_name in layer_names:
        layer_rows = [
            layer
            for row in rows
            for layer in row["layers"]
            if layer["layer"] == layer_name
        ]
        layer_summary.append(
            {
                "layer": layer_name,
                "mean_hidden_cosine_distance": _mean(
                    [float(row["hidden_cosine_distance"]) for row in layer_rows]
                ),
                "mean_hidden_l2_distance": _mean([float(row["hidden_l2_distance"]) for row in layer_rows]),
            }
        )
    return {
        "mean_posterior_js": _mean([float(row["posterior_js"]) for row in rows]),
        "mean_policy_js": _mean([float(row["policy_js"]) for row in rows]),
        "mean_range_logit_js": _mean([float(row["range_logit_js"]) for row in rows]),
        "mean_value_abs_delta": _mean([float(row["value_abs_delta"]) for row in rows]),
        "layer_summary": layer_summary,
    }


def _diagnostic_label(
    *,
    pair_count: int,
    min_pairs: int,
    mean_posterior_js: float,
    mean_policy_js: float,
    mean_range_logit_js: float,
    mean_hidden_cosine: float,
    separation_threshold: float,
) -> str:
    if pair_count < min_pairs:
        return INSUFFICIENT_PAIRS
    if mean_posterior_js <= 0.0:
        return INCONCLUSIVE
    model_separates = (
        mean_policy_js >= separation_threshold
        or mean_range_logit_js >= separation_threshold
        or mean_hidden_cosine >= separation_threshold
    )
    if model_separates:
        return ORDER_SENSITIVE_SUPPORTED
    return COMPOSITION_SHORTCUT_SUPPORTED


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Order-Counterfactual Causal Report",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| final diagnostic label | {payload['final_diagnostic_label']} |",
        f"| order groups | {payload['order_group_count']} |",
        f"| retained pairs | {payload['pair_count']} |",
        f"| mean posterior JS | {payload['summary']['mean_posterior_js']:.6f} |",
        f"| mean policy JS | {payload['summary']['mean_policy_js']:.6f} |",
        f"| mean range-logit JS | {payload['summary']['mean_range_logit_js']:.6f} |",
        f"| mean value abs delta | {payload['summary']['mean_value_abs_delta']:.6f} |",
        "",
        "## Coverage Profiles",
        "",
        "| profile | buckets | order-varying buckets | order-pair groups | points in order-varying buckets |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["coverage_profiles"]:
        lines.append(
            "| {profile} | {bucket_count} | {order_varying_bucket_count} | "
            "{order_pair_group_count} | {points_in_order_varying_buckets} |".format(**row)
        )
    lines.extend(
        [
        "",
        "## Layer Summary",
        "",
        "| layer | mean hidden cosine distance | mean hidden L2 distance |",
        "|---|---:|---:|",
        ]
    )
    for row in payload["summary"]["layer_summary"]:
        lines.append(
            "| {layer} | {mean_hidden_cosine_distance:.6f} | {mean_hidden_l2_distance:.6f} |".format(**row)
        )
    if payload["gaps"]:
        lines.extend(["", "## Gaps", ""])
        for gap in payload["gaps"]:
            lines.append(f"- {gap}")
    return "\n".join(lines) + "\n"


def build_order_counterfactual_report(
    *,
    train_config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    layers: tuple[str, ...],
    min_group_size: int = 2,
    min_posterior_js: float = 0.01,
    min_pairs: int = 100,
    max_pairs: int | None = None,
    separation_threshold: float = 0.001,
    device: str = "auto",
    write_pairs: bool = False,
) -> dict[str, Any]:
    """Build a legal same-composition different-order diagnostic report."""

    train_config = load_limit_ar_train_config(train_config_path)
    points = _load_points(train_config)
    manifest = build_split_manifest(
        points=points,
        fractions=SplitFractions(
            train=train_config.train_fraction,
            validation=train_config.validation_fraction,
            test=train_config.test_fraction,
        ),
        seed=train_config.split_seed,
    )
    splits = apply_split(points, manifest)
    test_points = list(splits.test)
    tokenizer = LimitTokenizer()
    labels = [tokenizer.opponent_hand_to_id[point.opponent_rank] for point in test_points]
    coverage_rows = composition_coverage(test_points)
    groups = build_order_groups(
        test_points,
        labels,
        tokenizer.opponent_card_size,
        min_group_size=min_group_size,
    )
    pairs = build_counterfactual_pairs(
        groups,
        min_posterior_js=min_posterior_js,
        max_pairs=max_pairs,
    )

    gaps = []
    if len(pairs) < min_pairs:
        gaps.append(f"retained pair count {len(pairs)} is below required minimum {min_pairs}")

    pair_rows: list[dict[str, Any]] = []
    checkpoint_load_info: dict[str, Any] = {"checkpoint_loaded": False}
    resolved_device = device
    if pairs:
        torch, resolved_device, encoded_layers, outputs, checkpoint_load_info = _model_outputs(
            train_config,
            checkpoint_path,
            test_points,
            layers,
            device,
        )
        pair_rows = [_pair_metrics(pair, encoded_layers, outputs, torch) for pair in pairs]

    summary = _aggregate_pair_rows(pair_rows) if pair_rows else {
        "mean_posterior_js": 0.0,
        "mean_policy_js": 0.0,
        "mean_range_logit_js": 0.0,
        "mean_value_abs_delta": 0.0,
        "layer_summary": [
            {
                "layer": layer,
                "mean_hidden_cosine_distance": 0.0,
                "mean_hidden_l2_distance": 0.0,
            }
            for layer in layers
        ],
    }
    mean_hidden_cosine = max(
        (float(row["mean_hidden_cosine_distance"]) for row in summary["layer_summary"]),
        default=0.0,
    )
    final_label = _diagnostic_label(
        pair_count=len(pairs),
        min_pairs=min_pairs,
        mean_posterior_js=float(summary["mean_posterior_js"]),
        mean_policy_js=float(summary["mean_policy_js"]),
        mean_range_logit_js=float(summary["mean_range_logit_js"]),
        mean_hidden_cosine=mean_hidden_cosine,
        separation_threshold=separation_threshold,
    )
    payload = {
        "version": "1.5",
        "report_type": "order_counterfactual_causal",
        "train_config": str(train_config_path),
        "checkpoint_path": str(checkpoint_path),
        "device": resolved_device,
        "layers": list(layers),
        "min_group_size": min_group_size,
        "min_posterior_js": min_posterior_js,
        "min_pairs": min_pairs,
        "max_pairs": max_pairs,
        "separation_threshold": separation_threshold,
        "test_point_count": len(test_points),
        "order_group_count": len(groups),
        "pair_count": len(pairs),
        "coverage_profiles": coverage_rows,
        "summary": summary,
        "final_diagnostic_label": final_label,
        "checkpoint_load_info": checkpoint_load_info,
        "gaps": gaps,
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "order_counterfactual_report.json", payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "order_counterfactual_report.md").write_text(_markdown(payload), encoding="utf-8")
    if write_pairs:
        with (output_dir / "pairs.jsonl").open("w", encoding="utf-8") as handle:
            for row in pair_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return payload


def _prefix_bucket(point) -> str:
    state = point.visible_state
    actions = _action_events(point)
    legal = ",".join(sorted(str(action) for action in point.legal_actions))
    public_cards = ",".join(str(card) for card in state["public_cards"])
    private_cards = ",".join(str(card) for card in state["private_cards"])
    return "|".join(
        [
            f"actor={point.acting_player}",
            f"round={state['round']}",
            f"pot={state['pot']}",
            f"legal={legal}",
            f"private={private_cards}",
            f"public={public_cards}",
            f"prefix={','.join(actions) if actions else 'empty'}",
        ]
    )


def _prefix_composition_bucket(point) -> str:
    state = point.visible_state
    actions = _action_events(point)
    counts = Counter(actions)
    legal = ",".join(sorted(str(action) for action in point.legal_actions))
    first = actions[0] if actions else "none"
    last = actions[-1] if actions else "none"
    return "|".join(
        [
            f"actor={point.acting_player}",
            f"round={state['round']}",
            f"pot={state['pot']}",
            f"legal={legal}",
            f"len={len(actions)}",
            *(f"{action}={counts[action]}" for action in ACTIONS),
            f"first={first}",
            f"last={last}",
        ]
    )


def _prefix_examples(points, max_points: int | None):
    selected = list(points[:max_points]) if max_points is not None else list(points)
    by_hand: dict[str, list[Any]] = defaultdict(list)
    for point in selected:
        by_hand[str(point.hand_id)].append(point)

    prefix_points = []
    metadata = []
    for hand_id in sorted(by_hand):
        hand_points = sorted(by_hand[hand_id], key=lambda point: int(point.step_index))
        for point_index, point in enumerate(hand_points):
            actions = _action_events(point)
            metadata.append(
                {
                    "source_id": hand_id,
                    "step_index": int(point.step_index),
                    "prefix_len": len(actions),
                    "is_final_prefix": point_index == len(hand_points) - 1,
                    "bucket": _prefix_bucket(point),
                    "composition_bucket": _prefix_composition_bucket(point),
                    "target_label": point.opponent_rank,
                }
            )
            prefix_points.append(point)
    return prefix_points, metadata


def _posterior_by_bucket(metadata: list[dict[str, Any]], tokenizer: LimitTokenizer) -> dict[str, tuple[float, ...]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for row in metadata:
        buckets[str(row["bucket"])].append(tokenizer.opponent_hand_to_id[str(row["target_label"])])
    posterior = {}
    for bucket, labels in buckets.items():
        counts = [0.0] * tokenizer.opponent_card_size
        for label in labels:
            counts[label] += 1.0
        total = float(len(labels))
        posterior[bucket] = tuple(value / total for value in counts)
    return posterior


def _prefix_composition_prior(
    train_points,
    tokenizer: LimitTokenizer,
    max_points: int | None,
) -> tuple[dict[str, tuple[float, ...]], tuple[float, ...]]:
    train_prefix_points, train_metadata = _prefix_examples(train_points, max_points)
    global_counts = [1.0] * tokenizer.opponent_card_size
    bucket_counts: dict[str, list[float]] = defaultdict(lambda: [1.0] * tokenizer.opponent_card_size)
    for row in train_metadata:
        label = tokenizer.opponent_hand_to_id[str(row["target_label"])]
        global_counts[label] += 1.0
        bucket_counts[str(row["composition_bucket"])][label] += 1.0
    global_total = sum(global_counts)
    global_prior = tuple(value / global_total for value in global_counts)
    priors = {}
    for bucket, counts in bucket_counts.items():
        total = sum(counts)
        priors[bucket] = tuple(value / total for value in counts)
    return priors, global_prior


def _composition_prior_for_row(
    row: dict[str, Any],
    priors: dict[str, tuple[float, ...]],
    global_prior: tuple[float, ...],
) -> tuple[float, ...]:
    return priors.get(str(row["composition_bucket"]), global_prior)


def _prefix_transition_rows(
    metadata: list[dict[str, Any]],
    posterior: dict[str, tuple[float, ...]],
    layers: dict[str, Any],
    outputs: dict[str, Any],
    torch,
) -> list[dict[str, Any]]:
    rows = []
    by_source: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(metadata):
        by_source[str(row["source_id"])].append((index, row))
    for source_id, indexed_rows in sorted(by_source.items()):
        ordered = sorted(indexed_rows, key=lambda item: int(item[1]["step_index"]))
        for (left_index, left_row), (right_index, right_row) in zip(ordered, ordered[1:], strict=False):
            left_posterior = posterior[str(left_row["bucket"])]
            right_posterior = posterior[str(right_row["bucket"])]
            posterior_js = jensen_shannon(left_posterior, right_posterior)
            left_policy = outputs["policy_logits"][left_index]
            right_policy = outputs["policy_logits"][right_index]
            left_range = outputs["opponent_card_logits"][left_index]
            right_range = outputs["opponent_card_logits"][right_index]
            layer_updates = []
            for layer_name, tensor in layers.items():
                layer_updates.append(
                    {
                        "layer": layer_name,
                        "hidden_cosine_distance": _cosine_distance(tensor[left_index], tensor[right_index], torch),
                        "hidden_l2_distance": _l2_distance(tensor[left_index], tensor[right_index], torch),
                    }
                )
            rows.append(
                {
                    "source_id": source_id,
                    "from_step_index": int(left_row["step_index"]),
                    "to_step_index": int(right_row["step_index"]),
                    "from_prefix_len": int(left_row["prefix_len"]),
                    "to_prefix_len": int(right_row["prefix_len"]),
                    "to_final_prefix": bool(right_row["is_final_prefix"]),
                    "posterior_js": posterior_js,
                    "policy_js": _softmax_js(left_policy, right_policy, torch),
                    "range_logit_js": _softmax_js(left_range, right_range, torch),
                    "value_abs_delta": float(abs((outputs["value"][left_index] - outputs["value"][right_index]).item())),
                    "layers": layer_updates,
                }
            )
    return rows


def _prefix_composition_transition_rows(
    metadata: list[dict[str, Any]],
    empirical_posterior: dict[str, tuple[float, ...]],
    composition_priors: dict[str, tuple[float, ...]],
    global_prior: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows = []
    by_source: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(metadata):
        by_source[str(row["source_id"])].append((index, row))
    for source_id, indexed_rows in sorted(by_source.items()):
        ordered = sorted(indexed_rows, key=lambda item: int(item[1]["step_index"]))
        for (_, left_row), (_, right_row) in zip(ordered, ordered[1:], strict=False):
            left_empirical = empirical_posterior[str(left_row["bucket"])]
            right_empirical = empirical_posterior[str(right_row["bucket"])]
            left_prior = _composition_prior_for_row(left_row, composition_priors, global_prior)
            right_prior = _composition_prior_for_row(right_row, composition_priors, global_prior)
            rows.append(
                {
                    "source_id": source_id,
                    "from_step_index": int(left_row["step_index"]),
                    "to_step_index": int(right_row["step_index"]),
                    "to_final_prefix": bool(right_row["is_final_prefix"]),
                    "posterior_js": jensen_shannon(left_empirical, right_empirical),
                    "composition_prior_js": jensen_shannon(left_prior, right_prior),
                }
            )
    return rows


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


def _trajectory_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    non_final = [row for row in rows if not row["to_final_prefix"]]
    target_rows = non_final if non_final else rows
    layer_names = sorted({layer["layer"] for row in target_rows for layer in row["layers"]})
    layer_summary = []
    for layer_name in layer_names:
        values = [
            float(layer["hidden_cosine_distance"])
            for row in target_rows
            for layer in row["layers"]
            if layer["layer"] == layer_name
        ]
        layer_summary.append({"layer": layer_name, "mean_hidden_cosine_distance": _mean(values)})
    posterior = [float(row["posterior_js"]) for row in target_rows]
    range_updates = [float(row["range_logit_js"]) for row in target_rows]
    policy_updates = [float(row["policy_js"]) for row in target_rows]
    max_hidden = max((row["mean_hidden_cosine_distance"] for row in layer_summary), default=0.0)
    return {
        "transition_count": len(rows),
        "non_final_transition_count": len(non_final),
        "mean_posterior_js": _mean(posterior),
        "mean_range_logit_js": _mean(range_updates),
        "mean_policy_js": _mean(policy_updates),
        "mean_value_abs_delta": _mean([float(row["value_abs_delta"]) for row in target_rows]),
        "posterior_range_update_correlation": _pearson(posterior, range_updates),
        "posterior_policy_update_correlation": _pearson(posterior, policy_updates),
        "max_mean_hidden_cosine_distance": max_hidden,
        "layer_summary": layer_summary,
    }


def _trajectory_label(summary: dict[str, Any], *, min_non_final_transitions: int, alignment_threshold: float) -> str:
    if int(summary["non_final_transition_count"]) < min_non_final_transitions:
        return INSUFFICIENT_PAIRS
    correlations = [
        value
        for value in (
            summary["posterior_range_update_correlation"],
            summary["posterior_policy_update_correlation"],
        )
        if value is not None
    ]
    if correlations and max(correlations) >= alignment_threshold:
        return TRAJECTORY_ALIGNED_SUPPORTED
    return TRAJECTORY_NOT_ALIGNED


def _trajectory_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Prefix Belief Trajectory Report",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| final diagnostic label | {payload['final_diagnostic_label']} |",
        f"| prefix examples | {payload['prefix_example_count']} |",
        f"| transitions | {summary['transition_count']} |",
        f"| non-final transitions | {summary['non_final_transition_count']} |",
        f"| mean posterior JS | {summary['mean_posterior_js']:.6f} |",
        f"| mean range-logit JS | {summary['mean_range_logit_js']:.6f} |",
        f"| mean policy JS | {summary['mean_policy_js']:.6f} |",
        "",
        "## Layer Summary",
        "",
        "| layer | mean hidden cosine distance |",
        "|---|---:|",
    ]
    for row in summary["layer_summary"]:
        lines.append("| {layer} | {mean_hidden_cosine_distance:.6f} |".format(**row))
    if payload["gaps"]:
        lines.extend(["", "## Gaps", ""])
        for gap in payload["gaps"]:
            lines.append(f"- {gap}")
    return "\n".join(lines) + "\n"


def build_prefix_trajectory_report(
    *,
    train_config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    layers: tuple[str, ...],
    max_points: int | None = None,
    min_non_final_transitions: int = 100,
    alignment_threshold: float = 0.05,
    device: str = "auto",
) -> dict[str, Any]:
    """Build a prefix trajectory diagnostic report from re-tokenized visible prefixes."""

    train_config = load_limit_ar_train_config(train_config_path)
    points = _load_points(train_config)
    manifest = build_split_manifest(
        points=points,
        fractions=SplitFractions(
            train=train_config.train_fraction,
            validation=train_config.validation_fraction,
            test=train_config.test_fraction,
        ),
        seed=train_config.split_seed,
    )
    splits = apply_split(points, manifest)
    test_points = list(splits.test)
    prefix_points, metadata = _prefix_examples(test_points, max_points)
    tokenizer = LimitTokenizer()
    posterior = _posterior_by_bucket(metadata, tokenizer)
    gaps = []
    if not prefix_points:
        gaps.append("no prefix examples were built")
        summary = _trajectory_summary([])
        final_label = INSUFFICIENT_PAIRS
        resolved_device = device
        checkpoint_load_info = {"checkpoint_loaded": False}
    else:
        torch, resolved_device, encoded_layers, outputs, checkpoint_load_info = _encode_model_outputs(
            train_config,
            checkpoint_path,
            prefix_points,
            layers,
            device,
        )
        rows = _prefix_transition_rows(metadata, posterior, encoded_layers, outputs, torch)
        summary = _trajectory_summary(rows)
        final_label = _trajectory_label(
            summary,
            min_non_final_transitions=min_non_final_transitions,
            alignment_threshold=alignment_threshold,
        )
        if summary["non_final_transition_count"] < min_non_final_transitions:
            gaps.append(
                "non-final transition count "
                f"{summary['non_final_transition_count']} is below required minimum {min_non_final_transitions}"
            )
    payload = {
        "version": "1.8",
        "report_type": "prefix_belief_trajectory",
        "train_config": str(train_config_path),
        "checkpoint_path": str(checkpoint_path),
        "device": resolved_device,
        "layers": list(layers),
        "max_points": max_points,
        "min_non_final_transitions": min_non_final_transitions,
        "alignment_threshold": alignment_threshold,
        "test_point_count": len(test_points),
        "prefix_example_count": len(prefix_points),
        "summary": summary,
        "final_diagnostic_label": final_label,
        "checkpoint_load_info": checkpoint_load_info,
        "gaps": gaps,
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "prefix_trajectory_report.json", payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prefix_trajectory_report.md").write_text(_trajectory_markdown(payload), encoding="utf-8")
    return payload


def _composition_baseline_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    non_final = [row for row in rows if not row["to_final_prefix"]]
    target_rows = non_final if non_final else rows
    posterior = [float(row["posterior_js"]) for row in target_rows]
    baseline = [float(row["composition_prior_js"]) for row in target_rows]
    return {
        "transition_count": len(rows),
        "non_final_transition_count": len(non_final),
        "mean_posterior_js": _mean(posterior),
        "mean_composition_prior_js": _mean(baseline),
        "posterior_composition_prior_correlation": _pearson(posterior, baseline),
    }


def _composition_baseline_label(summary: dict[str, Any], *, min_non_final_transitions: int) -> str:
    if int(summary["non_final_transition_count"]) < min_non_final_transitions:
        return INSUFFICIENT_PAIRS
    corr = summary["posterior_composition_prior_correlation"]
    if corr is not None and corr > 0.05:
        return COMPOSITION_SHORTCUT_SUPPORTED
    return INCONCLUSIVE


def _composition_baseline_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    return "\n".join(
        [
            "# Prefix Composition Baseline Report",
            "",
            "## Summary",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| final diagnostic label | {payload['final_diagnostic_label']} |",
            f"| transitions | {summary['transition_count']} |",
            f"| non-final transitions | {summary['non_final_transition_count']} |",
            f"| mean posterior JS | {summary['mean_posterior_js']:.6f} |",
            f"| mean composition-prior JS | {summary['mean_composition_prior_js']:.6f} |",
            f"| posterior/composition-prior correlation | {summary['posterior_composition_prior_correlation']} |",
            "",
        ]
    )


def build_prefix_composition_baseline_report(
    *,
    train_config_path: Path,
    output_dir: Path,
    max_points: int | None = None,
    min_non_final_transitions: int = 100,
) -> dict[str, Any]:
    """Build a visible prefix-composition baseline for trajectory evidence."""

    train_config = load_limit_ar_train_config(train_config_path)
    points = _load_points(train_config)
    manifest = build_split_manifest(
        points=points,
        fractions=SplitFractions(
            train=train_config.train_fraction,
            validation=train_config.validation_fraction,
            test=train_config.test_fraction,
        ),
        seed=train_config.split_seed,
    )
    splits = apply_split(points, manifest)
    tokenizer = LimitTokenizer()
    composition_priors, global_prior = _prefix_composition_prior(
        splits.train,
        tokenizer,
        max_points,
    )
    test_prefix_points, test_metadata = _prefix_examples(list(splits.test), max_points)
    empirical = _posterior_by_bucket(test_metadata, tokenizer)
    rows = _prefix_composition_transition_rows(test_metadata, empirical, composition_priors, global_prior)
    summary = _composition_baseline_summary(rows)
    final_label = _composition_baseline_label(summary, min_non_final_transitions=min_non_final_transitions)
    gaps = []
    if summary["non_final_transition_count"] < min_non_final_transitions:
        gaps.append(
            "non-final transition count "
            f"{summary['non_final_transition_count']} is below required minimum {min_non_final_transitions}"
        )
    payload = {
        "version": "1.9",
        "report_type": "prefix_composition_baseline",
        "train_config": str(train_config_path),
        "max_points": max_points,
        "min_non_final_transitions": min_non_final_transitions,
        "test_prefix_example_count": len(test_prefix_points),
        "train_prior_bucket_count": len(composition_priors),
        "summary": summary,
        "final_diagnostic_label": final_label,
        "gaps": gaps,
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "prefix_composition_baseline_report.json", payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prefix_composition_baseline_report.md").write_text(
        _composition_baseline_markdown(payload),
        encoding="utf-8",
    )
    return payload
