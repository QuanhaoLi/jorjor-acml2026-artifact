from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.models.ar import build_tiny_ar_model
from jorjor.models.tiny import require_torch
from jorjor.probe.inspect import COMPOSITION_SHORTCUT_SUPPORTED, INCONCLUSIVE, REALIZED_LABEL_CONTROLS_PASSED
from jorjor.teacher_artifact import FIXED_LIMIT_ACTION_SPACE, validate_teacher_artifact_rows
from jorjor.tokenizer import ACTIONS, LIMIT_ROUNDS, LimitTokenizer
from jorjor.train.leduc_tiny import _batch_indices, _resolve_device
from jorjor.train.teacher_distill import _row_to_decision_point, _rows_to_tensors, _split_rows


@dataclass(frozen=True)
class ProbeEvaluation:
    """Correctness and score for one k.0.6 probe/baseline."""

    top10: float
    correctness: list[bool]


def build_strong_teacher_gate_report(
    *,
    artifact_rows_path: Path,
    student_metrics_path: Path,
    model_path: Path,
    output_dir: Path,
    locked_layer: str = "encoder_layer_2",
    probe_epochs: int = 20,
    probe_batch_size: int = 64,
    probe_learning_rate: float = 0.01,
    bootstrap_count: int = 1000,
    bootstrap_seed: int = 20260603,
    confidence_level: float = 0.95,
    minimum_matched_points: int = 20,
    seed: int = 20260603,
    device: str = "cpu",
) -> dict[str, Any]:
    """Build k.0.6 strong-teacher hidden-state diagnostic gate report."""

    _validate_args(
        probe_epochs=probe_epochs,
        probe_batch_size=probe_batch_size,
        probe_learning_rate=probe_learning_rate,
        bootstrap_count=bootstrap_count,
        confidence_level=confidence_level,
        minimum_matched_points=minimum_matched_points,
    )
    torch, nn = require_torch()
    torch.manual_seed(seed)
    resolved_device = _resolve_device(device, torch)
    rows = _read_jsonl(artifact_rows_path)
    artifact_validation = validate_teacher_artifact_rows(rows)
    if not artifact_validation["passed"]:
        raise ValueError(f"teacher artifact validation failed: {artifact_validation}")
    student_metrics = _read_json_object(student_metrics_path)
    _validate_student_contract(student_metrics)

    splits = _split_rows(
        rows,
        train_fraction=float(student_metrics["train_points"]) / float(student_metrics["decision_point_count"]),
        validation_fraction=float(student_metrics["validation_points"]) / float(student_metrics["decision_point_count"]),
        split_seed=int(student_metrics["split_seed"]),
    )
    tokenizer = LimitTokenizer()
    train_tensors = _move_tensors(_rows_to_tensors(splits["train"], tokenizer, torch), resolved_device)
    test_tensors = _move_tensors(_rows_to_tensors(splits["test"], tokenizer, torch), resolved_device)
    train_points = [_row_to_decision_point(row) for row in splits["train"]]
    test_points = [_row_to_decision_point(row) for row in splits["test"]]
    train_labels = _opponent_rank_labels(splits["train"], tokenizer, torch, resolved_device)
    test_labels = _opponent_rank_labels(splits["test"], tokenizer, torch, resolved_device)

    model = _load_student_model(student_metrics, tokenizer, model_path, torch, resolved_device)
    train_hidden, train_layer_used = _extract_layer(model, train_tensors, locked_layer, torch)
    test_hidden, test_layer_used = _extract_layer(model, test_tensors, locked_layer, torch)
    if train_layer_used != test_layer_used:
        raise RuntimeError("locked layer resolution differed between train and test splits")
    locked_layer_used = test_layer_used

    hidden_eval = _train_linear_classifier(
        train_hidden,
        train_labels,
        test_hidden,
        test_labels,
        output_size=tokenizer.opponent_card_size,
        epochs=probe_epochs,
        batch_size=probe_batch_size,
        learning_rate=probe_learning_rate,
        seed=seed + 11,
        torch=torch,
        nn=nn,
        device=resolved_device,
    )
    shuffled_labels = train_labels[torch.randperm(train_labels.shape[0], device=resolved_device)]
    shuffled_eval = _train_linear_classifier(
        train_hidden,
        shuffled_labels,
        test_hidden,
        test_labels,
        output_size=tokenizer.opponent_card_size,
        epochs=probe_epochs,
        batch_size=probe_batch_size,
        learning_rate=probe_learning_rate,
        seed=seed + 17,
        torch=torch,
        nn=nn,
        device=resolved_device,
    )
    action_value_train = _action_value_features(splits["train"], torch, resolved_device)
    action_value_test = _action_value_features(splits["test"], torch, resolved_device)
    composition_train = _action_value_composition_features(splits["train"], torch, resolved_device)
    composition_test = _action_value_composition_features(splits["test"], torch, resolved_device)
    action_value_eval = _train_linear_classifier(
        action_value_train,
        train_labels,
        action_value_test,
        test_labels,
        output_size=tokenizer.opponent_card_size,
        epochs=probe_epochs,
        batch_size=probe_batch_size,
        learning_rate=probe_learning_rate,
        seed=seed + 23,
        torch=torch,
        nn=nn,
        device=resolved_device,
    )
    composition_eval = _train_linear_classifier(
        composition_train,
        train_labels,
        composition_test,
        test_labels,
        output_size=tokenizer.opponent_card_size,
        epochs=probe_epochs,
        batch_size=probe_batch_size,
        learning_rate=probe_learning_rate,
        seed=seed + 29,
        torch=torch,
        nn=nn,
        device=resolved_device,
    )

    buckets = [_composition_bucket(point) for point in test_points]
    s5 = _gate_from_correctness(
        hidden_eval.correctness,
        action_value_eval.correctness,
        buckets,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed + 5,
        confidence_level=confidence_level,
        minimum_points=0,
    )
    s6 = _gate_from_correctness(
        hidden_eval.correctness,
        composition_eval.correctness,
        buckets,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed + 6,
        confidence_level=confidence_level,
        minimum_points=0,
    )
    s7 = _gate_from_correctness(
        hidden_eval.correctness,
        composition_eval.correctness,
        buckets,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed + 7,
        confidence_level=confidence_level,
        minimum_points=minimum_matched_points,
    )
    final_label = _final_label(s6, s7)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "k.0.6",
        "report_type": "strong_teacher_strict_diagnostic_gates",
        "claim_boundary": "single_strong_teacher_smoke_only; requires_k_0_8_multiseed_for_paper_claim",
        "artifact_rows_path": str(artifact_rows_path),
        "student_metrics_path": str(student_metrics_path),
        "model_path": str(model_path),
        "locked_layer_requested": locked_layer,
        "locked_layer_used": locked_layer_used,
        "probe_family": "linear",
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "minimum_matched_points": minimum_matched_points,
        "student_contract": {
            "opponent_hand_supervised": bool(student_metrics.get("opponent_hand_supervised")),
            "probe_only_hidden_labels_used_for_training": bool(
                student_metrics.get("probe_only_hidden_labels_used_for_training", True)
            ),
            "policy_target": student_metrics.get("policy_target"),
            "value_target": student_metrics.get("value_target"),
        },
        "artifact_validation": artifact_validation,
        "data": {
            "train_points": len(splits["train"]),
            "test_points": len(splits["test"]),
            "matched_bucket_count": s7["matched_bucket_count"],
            "matched_point_count": s7["matched_point_count"],
        },
        "metrics": {
            "hidden_probe_top10": hidden_eval.top10,
            "label_permutation_hidden_top10": shuffled_eval.top10,
            "action_value_nuisance_top10": action_value_eval.top10,
            "action_value_composition_nuisance_top10": composition_eval.top10,
        },
        "gates": {
            "s5_hidden_vs_action_value": s5,
            "s6_hidden_vs_action_value_composition": s6,
            "s7_matched_hidden_vs_composition_prior": s7,
        },
        "final_diagnostic_label": final_label,
        "claim_implication": _claim_implication(final_label),
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "strong_teacher_gate_report.json", payload)
    (output_dir / "strong_teacher_gate_report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _validate_args(
    *,
    probe_epochs: int,
    probe_batch_size: int,
    probe_learning_rate: float,
    bootstrap_count: int,
    confidence_level: float,
    minimum_matched_points: int,
) -> None:
    if probe_epochs <= 0:
        raise ValueError("probe_epochs must be positive")
    if probe_batch_size <= 0:
        raise ValueError("probe_batch_size must be positive")
    if probe_learning_rate <= 0.0:
        raise ValueError("probe_learning_rate must be positive")
    if bootstrap_count <= 0:
        raise ValueError("bootstrap_count must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if minimum_matched_points < 0:
        raise ValueError("minimum_matched_points must be non-negative")


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _validate_student_contract(metrics: dict[str, Any]) -> None:
    if metrics.get("opponent_hand_supervised") is not False:
        raise ValueError("k.0.6 requires a no-range/opponent-unsupervised student")
    if metrics.get("probe_only_hidden_labels_used_for_training") is not False:
        raise ValueError("k.0.6 requires k.0.5 dummy hidden-label training policy")
    if metrics.get("policy_target") != "teacher_action_distribution":
        raise ValueError("k.0.6 expects k.0.5 teacher-action-distribution target")


def _load_student_model(metrics: dict[str, Any], tokenizer: LimitTokenizer, model_path: Path, torch, device: str):
    model = build_tiny_ar_model(
        vocab_size=tokenizer.vocab_size,
        action_size=tokenizer.action_size,
        opponent_card_size=tokenizer.opponent_card_size,
        hidden_size=int(metrics["hidden_size"]),
        num_heads=int(metrics["num_heads"]),
        num_layers=int(metrics["num_layers"]),
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _move_tensors(tensors, device: str):
    return {key: value.to(device) for key, value in tensors.items()}


def _extract_layer(model, tensors, requested_layer: str, torch):
    model.eval()
    with torch.no_grad():
        layers = model.encode_layers(tensors["input_ids"])
    layer = requested_layer if requested_layer in layers else "sequence_hidden"
    return layers[layer].detach(), layer


def _opponent_rank_labels(rows: list[dict[str, Any]], tokenizer: LimitTokenizer, torch, device: str):
    labels = [
        tokenizer.opponent_hand_to_id[row["hidden_label_for_probe_only"]["opponent_rank"]]
        for row in rows
    ]
    return torch.tensor(labels, dtype=torch.long, device=device)


def _train_linear_classifier(
    train_x,
    train_y,
    test_x,
    test_y,
    *,
    output_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    torch,
    nn,
    device: str,
) -> ProbeEvaluation:
    torch.manual_seed(seed)
    probe = nn.Linear(train_x.shape[1], output_size).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    for _ in range(epochs):
        permutation = torch.randperm(train_x.shape[0], device=device)
        shuffled_x = train_x[permutation]
        shuffled_y = train_y[permutation]
        for batch_slice in _batch_indices(train_x.shape[0], batch_size):
            batch_x = shuffled_x[batch_slice]
            batch_y = shuffled_y[batch_slice]
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(probe(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        logits = probe(test_x)
        k = min(10, logits.shape[1])
        topk = torch.topk(logits, k=k, dim=1).indices
        hits = topk.eq(test_y.unsqueeze(1)).any(dim=1)
    correctness = [bool(value) for value in hits.detach().cpu().tolist()]
    top10 = sum(correctness) / len(correctness) if correctness else 0.0
    return ProbeEvaluation(top10=top10, correctness=correctness)


def _action_value_features(rows: list[dict[str, Any]], torch, device: str):
    vectors = []
    for row in rows:
        distribution = [float(row["teacher_action_distribution"][action]) for action in FIXED_LIMIT_ACTION_SPACE]
        action_one_hot = [1.0 if row["sampled_action"] == action else 0.0 for action in FIXED_LIMIT_ACTION_SPACE]
        value = [float(row["teacher_value"] or 0.0)]
        vectors.append(distribution + action_one_hot + value)
    return torch.tensor(vectors, dtype=torch.float32, device=device)


def _action_value_composition_features(rows: list[dict[str, Any]], torch, device: str):
    vectors = []
    round_to_id = {name: index for index, name in enumerate(LIMIT_ROUNDS)}
    for row in rows:
        state = row["student_input"]
        actions = _action_events(state["betting_history"])
        counts = Counter(actions)
        last = actions[-1] if actions else "none"
        action_value = [float(row["teacher_action_distribution"][action]) for action in FIXED_LIMIT_ACTION_SPACE]
        action_value += [1.0 if row["sampled_action"] == action else 0.0 for action in FIXED_LIMIT_ACTION_SPACE]
        action_value += [float(row["teacher_value"] or 0.0)]
        composition = [float(counts[action]) for action in ACTIONS]
        composition += [float(len(actions))]
        composition += [1.0 if last == action else 0.0 for action in ACTIONS]
        composition += [1.0 if action in row["legal_actions"] else 0.0 for action in ACTIONS]
        composition += [1.0 if int(row["player_position"]) == player else 0.0 for player in (0, 1)]
        composition += [1.0 if round_to_id.get(str(state["round"])) == index else 0.0 for index in range(len(LIMIT_ROUNDS))]
        vectors.append(action_value + composition)
    return torch.tensor(vectors, dtype=torch.float32, device=device)


def _action_events(history: list[dict[str, Any]]) -> list[str]:
    return [str(event["action"]) for event in history if event.get("type") == "action"]


def _composition_bucket(point) -> str:
    state = point.visible_state
    actions = _action_events(state["betting_history"])
    counts = Counter(actions)
    last = actions[-1] if actions else "none"
    legal = ",".join(sorted(str(action) for action in point.legal_actions))
    return "|".join(
        [
            f"actor={point.acting_player}",
            f"round={state['round']}",
            f"legal={legal}",
            f"len={len(actions)}",
            *(f"{action}={counts[action]}" for action in ACTIONS),
            f"last={last}",
            f"sampled={point.action}",
            f"value={int(point.terminal_return or 0)}",
        ]
    )


def _gate_from_correctness(
    left_correctness: list[bool],
    right_correctness: list[bool],
    buckets: list[str],
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
    minimum_points: int,
) -> dict[str, Any]:
    if not (len(left_correctness) == len(right_correctness) == len(buckets)):
        raise ValueError("correctness and bucket lengths must match")
    bucket_deltas: dict[str, list[float]] = defaultdict(list)
    for left, right, bucket in zip(left_correctness, right_correctness, buckets, strict=True):
        bucket_deltas[bucket].append((1.0 if left else 0.0) - (1.0 if right else 0.0))
    matched = {bucket: values for bucket, values in bucket_deltas.items() if len(values) >= 2}
    selected = matched if minimum_points > 0 else bucket_deltas
    selected_values = [sum(values) / len(values) for values in selected.values()]
    matched_point_count = sum(len(values) for values in selected.values())
    if not selected_values:
        ci = {"mean_delta": 0.0, "lower_delta": 0.0, "upper_delta": 0.0}
    else:
        ci = _bootstrap_mean_ci(
            selected_values,
            bootstrap_count=bootstrap_count,
            bootstrap_seed=bootstrap_seed,
            confidence_level=confidence_level,
        )
    sufficient = matched_point_count >= minimum_points
    return {
        "delta": ci["mean_delta"],
        "delta_ci": [ci["lower_delta"], ci["upper_delta"]],
        "matched_bucket_count": len(selected),
        "matched_point_count": matched_point_count,
        "sufficient_support": sufficient,
        "estimand": "mean_bucket_delta",
        "bootstrap_unit": "composition_bucket",
    }


def _bootstrap_mean_ci(
    values: list[float],
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, float]:
    if not values:
        raise ValueError("bootstrap requires values")
    rng = random.Random(bootstrap_seed)
    means = []
    for _ in range(bootstrap_count):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    lower_index = int(((1.0 - confidence_level) / 2.0) * (bootstrap_count - 1))
    upper_index = int((1.0 - (1.0 - confidence_level) / 2.0) * (bootstrap_count - 1))
    return {
        "mean_delta": sum(values) / len(values),
        "lower_delta": means[lower_index],
        "upper_delta": means[upper_index],
    }


def _final_label(s6: dict[str, Any], s7: dict[str, Any]) -> str:
    s6_lower, s6_upper = s6["delta_ci"]
    s7_lower, s7_upper = s7["delta_ci"]
    if s6_lower > 0.0 and s7_lower > 0.0 and s7["sufficient_support"]:
        return REALIZED_LABEL_CONTROLS_PASSED
    if s6_upper <= 0.0 or (s7["sufficient_support"] and s7_upper <= 0.0):
        return COMPOSITION_SHORTCUT_SUPPORTED
    return INCONCLUSIVE


def _claim_implication(label: str) -> str:
    if label == REALIZED_LABEL_CONTROLS_PASSED:
        return (
            "The strong-teacher student passes local S6/S7 realized-label residual gates; "
            "independent posterior gates remain required before any belief-like paper claim."
        )
    if label == COMPOSITION_SHORTCUT_SUPPORTED:
        return "The strong-teacher student does not beat composition-aware controls in this smoke."
    return "The local smoke is unresolved or under-supported; do not update the paper claim."


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Strong-Teacher Gate Report",
        "",
        "## Summary",
        "",
        f"- final diagnostic label: `{payload['final_diagnostic_label']}`",
        f"- locked layer: `{payload['locked_layer_used']}`",
        f"- claim implication: {payload['claim_implication']}",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in payload["metrics"].items():
        lines.append(f"| {key} | {float(value):.6f} |")
    lines.extend(["", "## Gates", "", "| gate | delta | lower | upper | points | buckets |", "|---|---:|---:|---:|---:|---:|"])
    for key, gate in payload["gates"].items():
        lower, upper = gate["delta_ci"]
        lines.append(
            f"| {key} | {gate['delta']:.6f} | {lower:.6f} | {upper:.6f} | "
            f"{gate['matched_point_count']} | {gate['matched_bucket_count']} |"
        )
    return "\n".join(lines) + "\n"
