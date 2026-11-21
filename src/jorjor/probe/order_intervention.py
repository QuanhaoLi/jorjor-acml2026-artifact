from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.config import load_limit_ar_train_config
from jorjor.data.split import SplitFractions, apply_split, build_split_manifest
from jorjor.models.ar import build_tiny_ar_model
from jorjor.models.tiny import require_torch
from jorjor.probe.counterfactual import jensen_shannon
from jorjor.probe.limit_ar import _load_ar_checkpoint_with_vocab_compat
from jorjor.tokenizer import ACTIONS, LimitTokenizer
from jorjor.train.leduc_tiny import _features_to_tensors, _move_tensors, _resolve_device
from jorjor.train.limit_controls import controlled_limit_points, controlled_range_labels
from jorjor.train.limit_tiny import _load_points


ORDER_SENSITIVE_STRESS_SUPPORTED = "order_sensitive_stress_supported"
ORDER_INSENSITIVE_STRESS_SUPPORTED = "order_insensitive_stress_supported"
INSUFFICIENT_PAIRS = "insufficient_pairs"
INCONCLUSIVE = "inconclusive"
DEFAULT_SEEDS = ("seed_k", "seed_l", "seed_m")
DEFAULT_MIN_PAIRS_FOR_LABEL = 100
DEFAULT_MOVEMENT_EPSILON = 1e-6


def build_order_intervention_report(
    *,
    archive_root: Path,
    output_dir: Path,
    seed_names: tuple[str, ...] = DEFAULT_SEEDS,
    train_config_dir: Path = Path("configs/train"),
    bootstrap_count: int = 1000,
    bootstrap_seed: int = 20260610,
    confidence_level: float = 0.95,
    device: str = "cpu",
    batch_size: int = 512,
    max_test_points: int | None = None,
    max_pairs: int | None = 10000,
    locked_layer: str = "encoder_layer_2",
    min_pairs_for_label: int = DEFAULT_MIN_PAIRS_FOR_LABEL,
    movement_epsilon: float = DEFAULT_MOVEMENT_EPSILON,
) -> dict[str, Any]:
    """Build k.6.0 input-level betting-order intervention report."""

    if not seed_names:
        raise ValueError("seed_names must not be empty")
    if min_pairs_for_label <= 0:
        raise ValueError("min_pairs_for_label must be positive")
    if movement_epsilon < 0.0:
        raise ValueError("movement_epsilon must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, seed_name in enumerate(seed_names):
        train_config_path = train_config_dir / f"limit_ar_action_value_only_{seed_name}_50k_4layer.json"
        seed_dir = archive_root / "outputs" / f"limit_ar_action_value_only_{seed_name}_50k_4layer"
        rows.append(
            _seed_row(
                seed_name=seed_name,
                train_config_path=train_config_path,
                checkpoint_path=seed_dir / "model.pt",
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed + index,
                confidence_level=confidence_level,
                device=device,
                batch_size=batch_size,
                max_test_points=max_test_points,
                max_pairs=max_pairs,
                locked_layer=locked_layer,
                min_pairs_for_label=min_pairs_for_label,
                movement_epsilon=movement_epsilon,
            )
        )
    aggregate = _aggregate_rows(
        rows,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed + 10000,
        confidence_level=confidence_level,
    )
    payload = {
        "version": "k.6.0",
        "report_type": "k.6.0_order_intervention",
        "claim_scope": "input_level_order_stress_not_legal_poker_causality",
        "archive_root": str(archive_root),
        "seed_names": list(seed_names),
        "train_config_dir": str(train_config_dir),
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "device": device,
        "batch_size": batch_size,
        "max_test_points": max_test_points,
        "max_pairs": max_pairs,
        "locked_layer": locked_layer,
        "min_pairs_for_label": min_pairs_for_label,
        "movement_epsilon": movement_epsilon,
        "rows": rows,
        "aggregate": aggregate,
        "final_diagnostic_label": _aggregate_label(rows, aggregate),
        "allowed_paper_claim": (
            "Under input-level same-composition order permutations, the frozen poker AR can be audited for "
            "order sensitivity. This is a stress test, not a legal poker causal proof."
        ),
        "forbidden_paper_claim": "This input permutation proves or disproves ordered opponent-belief tracking.",
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "k.6.0_order_intervention_report.json", payload)
    (output_dir / "k.6.0_order_intervention_report.md").write_text(_markdown(payload), encoding="utf-8")
    write_json_artifact(output_dir / "k.6.0_order_intervention_manifest.json", _manifest(output_dir, payload))
    return payload


def order_intervention_point(point, *, seed: int):
    """Return a point with history prefix order permuted and last action fixed."""

    history = [dict(event) for event in point.visible_state["betting_history"]]
    action_positions = [index for index, event in enumerate(history) if event.get("type") == "action"]
    if len(action_positions) < 3:
        return None
    prefix_positions = action_positions[:-1]
    prefix_events = [history[index] for index in prefix_positions]
    shuffled = list(prefix_events)
    random.Random(seed + _stable_int(f"{point.hand_id}:{point.step_index}")).shuffle(shuffled)
    if [event.get("action") for event in shuffled] == [event.get("action") for event in prefix_events]:
        shuffled = shuffled[1:] + shuffled[:1]
    if [event.get("action") for event in shuffled] == [event.get("action") for event in prefix_events]:
        return None
    for index, event in zip(prefix_positions, shuffled, strict=True):
        history[index] = dict(event)
    visible_state = dict(point.visible_state)
    visible_state["betting_history"] = history
    return replace(point, visible_state=visible_state)


def _seed_row(
    *,
    seed_name: str,
    train_config_path: Path,
    checkpoint_path: Path,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
    device: str,
    batch_size: int,
    max_test_points: int | None,
    max_pairs: int | None,
    locked_layer: str,
    min_pairs_for_label: int,
    movement_epsilon: float,
) -> dict[str, Any]:
    train_config = load_limit_ar_train_config(train_config_path)
    test_points = _test_points(train_config, max_test_points)
    pair_points = []
    original_points = []
    for point in test_points:
        intervened = order_intervention_point(point, seed=bootstrap_seed)
        if intervened is None:
            continue
        original_points.append(point)
        pair_points.append(intervened)
        if max_pairs is not None and len(pair_points) >= max_pairs:
            break
    metrics = _intervention_metrics(
        train_config=train_config,
        checkpoint_path=checkpoint_path,
        original_points=original_points,
        intervened_points=pair_points,
        device=device,
        batch_size=batch_size,
        locked_layer=locked_layer,
    )
    summary = _summary(
        metrics,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    return {
        "seed": seed_name,
        "train_config_path": str(train_config_path),
        "train_config_sha256": _sha256_file(train_config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "test_points": len(test_points),
        "eligible_point_count": _eligible_count(test_points),
        "intervention_pair_count": len(pair_points),
        "max_test_points": max_test_points,
        "max_pairs": max_pairs,
        "locked_layer": metrics["locked_layer"],
        "summary": summary,
        "final_diagnostic_label": _seed_label(
            summary,
            pair_count=len(pair_points),
            min_pairs_for_label=min_pairs_for_label,
            movement_epsilon=movement_epsilon,
        ),
    }


def _test_points(train_config, max_test_points: int | None):
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
    test_points = controlled_range_labels(
        controlled_limit_points(splits.test, train_config.input_control),
        train_config.range_label_control,
        train_config.range_label_seed + 2,
    )
    return test_points[:max_test_points] if max_test_points is not None else test_points


def _eligible_count(points) -> int:
    return sum(1 for point in points if order_intervention_point(point, seed=0) is not None)


def _intervention_metrics(
    *,
    train_config,
    checkpoint_path: Path,
    original_points,
    intervened_points,
    device: str,
    batch_size: int,
    locked_layer: str,
) -> dict[str, Any]:
    if not original_points:
        return {
            "locked_layer": locked_layer,
            "hidden_cosine_distances": [],
            "hidden_l2_distances": [],
            "policy_js": [],
            "range_head_js": [],
            "value_abs_deltas": [],
            "policy_argmax_changed": [],
            "range_argmax_changed": [],
            "checkpoint_load_info": {"checkpoint_loaded": False},
        }
    torch, _ = require_torch()
    resolved_device = _resolve_device(device, torch)
    tokenizer = LimitTokenizer()
    all_points = [*original_points, *intervened_points]
    tensors = _move_tensors(_features_to_tensors([tokenizer.encode(point) for point in all_points], torch), resolved_device)
    model = build_tiny_ar_model(
        vocab_size=tokenizer.vocab_size,
        action_size=tokenizer.action_size,
        opponent_card_size=tokenizer.opponent_card_size,
        hidden_size=train_config.hidden_size,
        num_heads=train_config.num_heads,
        num_layers=train_config.num_layers,
    ).to(resolved_device)
    load_info = _load_ar_checkpoint_with_vocab_compat(model, checkpoint_path, torch, resolved_device)
    hidden_rows = []
    policy_rows = []
    range_rows = []
    value_rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, tensors["input_ids"].shape[0], batch_size):
            stop = min(start + batch_size, tensors["input_ids"].shape[0])
            layers = model.encode_layers(tensors["input_ids"][start:stop])
            layer_name = locked_layer if locked_layer in layers else "sequence_hidden"
            hidden_rows.append(layers[layer_name].detach())
            hidden = layers["sequence_hidden"]
            policy_rows.append(model.policy_head(hidden).detach())
            range_rows.append(model.opponent_card_head(hidden).detach())
            value_rows.append(model.value_head(hidden).squeeze(-1).detach())
    hidden_all = torch.cat(hidden_rows, dim=0)
    policy_all = torch.cat(policy_rows, dim=0)
    range_all = torch.cat(range_rows, dim=0)
    value_all = torch.cat(value_rows, dim=0)
    n = len(original_points)
    original = {
        "hidden": hidden_all[:n],
        "policy": policy_all[:n],
        "range": range_all[:n],
        "value": value_all[:n],
    }
    intervened = {
        "hidden": hidden_all[n:],
        "policy": policy_all[n:],
        "range": range_all[n:],
        "value": value_all[n:],
    }
    return {
        "locked_layer": locked_layer if locked_layer in model.encode_layers(tensors["input_ids"][:1]) else "sequence_hidden",
        "hidden_cosine_distances": [
            _cosine_distance(original["hidden"][index], intervened["hidden"][index], torch)
            for index in range(n)
        ],
        "hidden_l2_distances": [
            float(torch.linalg.vector_norm(original["hidden"][index] - intervened["hidden"][index]).item())
            for index in range(n)
        ],
        "policy_js": [
            _softmax_js(original["policy"][index], intervened["policy"][index], torch)
            for index in range(n)
        ],
        "range_head_js": [
            _softmax_js(original["range"][index], intervened["range"][index], torch)
            for index in range(n)
        ],
        "value_abs_deltas": [
            float(abs((original["value"][index] - intervened["value"][index]).item()))
            for index in range(n)
        ],
        "policy_argmax_changed": [
            float(torch.argmax(original["policy"][index]).item() != torch.argmax(intervened["policy"][index]).item())
            for index in range(n)
        ],
        "range_argmax_changed": [
            float(torch.argmax(original["range"][index]).item() != torch.argmax(intervened["range"][index]).item())
            for index in range(n)
        ],
        "checkpoint_load_info": load_info,
    }


def _summary(metrics: dict[str, Any], *, bootstrap_count: int, bootstrap_seed: int, confidence_level: float) -> dict[str, Any]:
    hidden_ci = _bootstrap_mean_ci(
        metrics["hidden_cosine_distances"], bootstrap_count, bootstrap_seed, confidence_level
    )
    policy_ci = _bootstrap_mean_ci(metrics["policy_js"], bootstrap_count, bootstrap_seed + 1, confidence_level)
    range_ci = _bootstrap_mean_ci(metrics["range_head_js"], bootstrap_count, bootstrap_seed + 2, confidence_level)
    value_ci = _bootstrap_mean_ci(
        metrics["value_abs_deltas"], bootstrap_count, bootstrap_seed + 3, confidence_level
    )
    return {
        "mean_hidden_cosine_distance": hidden_ci["mean"],
        "hidden_cosine_distance_ci": hidden_ci,
        "mean_hidden_l2_distance": _mean(metrics["hidden_l2_distances"]),
        "mean_policy_js": policy_ci["mean"],
        "policy_js_ci": policy_ci,
        "mean_range_head_js": range_ci["mean"],
        "range_head_js_ci": range_ci,
        "mean_value_abs_delta": value_ci["mean"],
        "value_abs_delta_ci": value_ci,
        "policy_argmax_changed_rate": _mean(metrics["policy_argmax_changed"]),
        "range_argmax_changed_rate": _mean(metrics["range_argmax_changed"]),
    }


def _seed_label(
    summary: dict[str, Any],
    *,
    pair_count: int,
    min_pairs_for_label: int,
    movement_epsilon: float,
) -> str:
    if pair_count < min_pairs_for_label:
        return INSUFFICIENT_PAIRS
    hidden_lower = float(summary["hidden_cosine_distance_ci"]["lower"])
    policy_lower = float(summary["policy_js_ci"]["lower"])
    range_lower = float(summary["range_head_js_ci"]["lower"])
    if max(hidden_lower, policy_lower, range_lower) > movement_epsilon:
        return ORDER_SENSITIVE_STRESS_SUPPORTED
    if max(
        float(summary["hidden_cosine_distance_ci"]["upper"]),
        float(summary["policy_js_ci"]["upper"]),
        float(summary["range_head_js_ci"]["upper"]),
    ) <= movement_epsilon:
        return ORDER_INSENSITIVE_STRESS_SUPPORTED
    return INCONCLUSIVE


def _aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    metrics = {}
    for name in (
        "mean_hidden_cosine_distance",
        "mean_policy_js",
        "mean_range_head_js",
        "mean_value_abs_delta",
        "policy_argmax_changed_rate",
        "range_argmax_changed_rate",
    ):
        values = [float(row["summary"][name]) for row in rows]
        metrics[name] = {
            "mean": _mean(values),
            "seed_ci": _bootstrap_mean_ci(values, bootstrap_count, bootstrap_seed + _stable_int(name) % 1000, confidence_level),
        }
    return {
        "seed_count": len(rows),
        "label_counts": _count_labels(row["final_diagnostic_label"] for row in rows),
        "intervention_pair_count": sum(int(row["intervention_pair_count"]) for row in rows),
        "metrics": metrics,
    }


def _aggregate_label(rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> str:
    labels = set(aggregate["label_counts"])
    if labels == {ORDER_SENSITIVE_STRESS_SUPPORTED}:
        return ORDER_SENSITIVE_STRESS_SUPPORTED
    if labels == {ORDER_INSENSITIVE_STRESS_SUPPORTED}:
        return ORDER_INSENSITIVE_STRESS_SUPPORTED
    if labels == {INSUFFICIENT_PAIRS}:
        return INSUFFICIENT_PAIRS
    return INCONCLUSIVE


def _manifest(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": payload["version"],
        "report_type": "k.6.0_order_intervention_manifest",
        "claim_scope": payload["claim_scope"],
        "output_dir": str(output_dir),
        "expected_outputs": [
            "k.6.0_order_intervention_report.json",
            "k.6.0_order_intervention_report.md",
            "k.6.0_order_intervention_manifest.json",
        ],
        "report_checksum": payload["checksum"],
        "final_diagnostic_label": payload["final_diagnostic_label"],
        "one_command_reproduction": (
            ".venv/bin/python scripts/build_order_intervention_report.py "
            f"--archive-root {payload['archive_root']} "
            f"--output-dir {output_dir} "
            f"--bootstrap-count {payload['bootstrap_count']} "
            f"--bootstrap-seed {payload['bootstrap_seed']} "
            f"--batch-size {payload['batch_size']} "
            f"--locked-layer {payload['locked_layer']} "
            f"--min-pairs-for-label {payload['min_pairs_for_label']} "
            f"--movement-epsilon {payload['movement_epsilon']} "
            + ("--max-test-points {value} ".format(value=payload["max_test_points"]) if payload["max_test_points"] is not None else "")
            + ("--max-pairs {value} ".format(value=payload["max_pairs"]) if payload["max_pairs"] is not None else "")
            + "--seed-names "
            + " ".join(payload["seed_names"])
        ),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# k.6.0 Order-Intervention Stress Report",
        "",
        "This is an input-level order permutation stress test. It is not a legal poker causal proof.",
        "",
        f"final_diagnostic_label: `{payload['final_diagnostic_label']}`",
        "",
        "| seed | label | pairs | hidden cosine | policy JS | range JS | value delta | policy changed |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        summary = row["summary"]
        lines.append(
            "| {seed} | `{label}` | {pairs} | {hidden:.8f} | {policy:.8f} | {range_js:.8f} | {value:.8f} | {changed:.6f} |".format(
                seed=row["seed"],
                label=row["final_diagnostic_label"],
                pairs=row["intervention_pair_count"],
                hidden=summary["mean_hidden_cosine_distance"],
                policy=summary["mean_policy_js"],
                range_js=summary["mean_range_head_js"],
                value=summary["mean_value_abs_delta"],
                changed=summary["policy_argmax_changed_rate"],
            )
        )
    lines.extend(["", "## Aggregate", "", "| metric | mean | lower | upper |", "|---|---:|---:|---:|"])
    for name, row in payload["aggregate"]["metrics"].items():
        ci = row["seed_ci"]
        lines.append(f"| {name} | {row['mean']:.8f} | {ci['lower']:.8f} | {ci['upper']:.8f} |")
    lines.extend(["", "## Claim Discipline", "", payload["allowed_paper_claim"], "", f"Forbidden claim: {payload['forbidden_paper_claim']}"])
    return "\n".join(lines) + "\n"


def _softmax_js(left_logits, right_logits, torch) -> float:
    left = torch.nn.functional.softmax(left_logits, dim=-1).detach().cpu().tolist()
    right = torch.nn.functional.softmax(right_logits, dim=-1).detach().cpu().tolist()
    return jensen_shannon(tuple(float(value) for value in left), tuple(float(value) for value in right))


def _cosine_distance(left, right, torch) -> float:
    denominator = torch.linalg.vector_norm(left.float()) * torch.linalg.vector_norm(right.float())
    if float(denominator.item()) <= 1e-12:
        return 0.0
    similarity = float((torch.dot(left.float(), right.float()) / denominator).clamp(-1.0, 1.0).item())
    return 1.0 - similarity


def _bootstrap_mean_ci(values: list[float], bootstrap_count: int, bootstrap_seed: int, confidence_level: float) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    rng = random.Random(bootstrap_seed)
    means = []
    for _ in range(bootstrap_count):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    lower = int(((1.0 - confidence_level) / 2.0) * (bootstrap_count - 1))
    upper = int((1.0 - (1.0 - confidence_level) / 2.0) * (bootstrap_count - 1))
    return {"mean": _mean(values), "lower": means[lower], "upper": means[upper]}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _count_labels(labels) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[str(label)] = counts.get(str(label), 0) + 1
    return counts


def _stable_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
