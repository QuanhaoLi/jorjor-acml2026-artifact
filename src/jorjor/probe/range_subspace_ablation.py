from __future__ import annotations

from dataclasses import dataclass
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
from jorjor.tokenizer import LimitTokenizer
from jorjor.train.leduc_tiny import _features_to_tensors, _move_tensors, _resolve_device
from jorjor.train.limit_controls import controlled_limit_points, controlled_range_labels
from jorjor.train.limit_tiny import _load_points


RANGE_FOOTPRINT_SUPPORTED = "range_subspace_behavior_footprint_supported"
RANGE_FOOTPRINT_NOT_SUPPORTED = "range_subspace_behavior_footprint_not_supported"
PROBE_NOT_SUPPORTED = "probe_not_supported"
INCONCLUSIVE = "inconclusive"
DEFAULT_SEEDS = ("seed_k", "seed_l", "seed_m")


@dataclass(frozen=True)
class _SplitFeatures:
    train_hidden: Any
    train_labels: Any
    test_hidden: Any
    test_sequence_hidden: Any
    test_policy_logits: Any
    test_value: Any
    test_labels: Any


def build_range_subspace_ablation_report(
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
    max_train_points: int | None = 30000,
    max_test_points: int | None = None,
    locked_layer: str = "sequence_hidden",
    probe_epochs: int = 20,
    probe_learning_rate: float = 0.01,
    random_subspace_count: int = 8,
    max_subspace_rank: int = 16,
    probe_top10_floor: float = 0.08,
) -> dict[str, Any]:
    """Build k.6.3 range-subspace ablation stress report."""

    if not seed_names:
        raise ValueError("seed_names must not be empty")
    if bootstrap_count <= 0:
        raise ValueError("bootstrap_count must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if random_subspace_count <= 0:
        raise ValueError("random_subspace_count must be positive")
    if max_subspace_rank <= 0:
        raise ValueError("max_subspace_rank must be positive")
    if locked_layer != "sequence_hidden":
        raise ValueError("k.6.3 behavior-footprint ablation requires locked_layer=sequence_hidden")
    if probe_epochs <= 0:
        raise ValueError("probe_epochs must be positive")
    if probe_learning_rate <= 0.0:
        raise ValueError("probe_learning_rate must be positive")
    if probe_top10_floor < 0.0:
        raise ValueError("probe_top10_floor must be non-negative")
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
                max_train_points=max_train_points,
                max_test_points=max_test_points,
                locked_layer=locked_layer,
                probe_epochs=probe_epochs,
                probe_learning_rate=probe_learning_rate,
                random_subspace_count=random_subspace_count,
                max_subspace_rank=max_subspace_rank,
                probe_top10_floor=probe_top10_floor,
            )
        )
    aggregate = _aggregate_rows(rows, bootstrap_count, bootstrap_seed + 10000, confidence_level)
    payload = {
        "version": "k.6.3",
        "report_type": "k.6.3_range_subspace_ablation",
        "claim_scope": "range_probe_subspace_behavior_footprint_stress_not_causal_mediation",
        "archive_root": str(archive_root),
        "seed_names": list(seed_names),
        "train_config_dir": str(train_config_dir),
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "device": device,
        "batch_size": batch_size,
        "max_train_points": max_train_points,
        "max_test_points": max_test_points,
        "locked_layer": locked_layer,
        "probe_epochs": probe_epochs,
        "probe_learning_rate": probe_learning_rate,
        "random_subspace_count": random_subspace_count,
        "max_subspace_rank": max_subspace_rank,
        "probe_top10_floor": probe_top10_floor,
        "rows": rows,
        "aggregate": aggregate,
        "final_diagnostic_label": _aggregate_label(aggregate),
        "allowed_paper_claim": (
            "Top range-probe subspace ablation can be compared with same-rank random subspace ablation as a "
            "behavior-footprint stress test. It is not a causal mediation proof."
        ),
        "forbidden_paper_claim": "This subspace ablation proves the range-correlated hidden directions causally drive policy.",
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "k.6.3_range_subspace_ablation_report.json", payload)
    (output_dir / "k.6.3_range_subspace_ablation_report.md").write_text(_markdown(payload), encoding="utf-8")
    write_json_artifact(output_dir / "k.6.3_range_subspace_ablation_manifest.json", _manifest(output_dir, payload))
    return payload


def remove_subspace(hidden, basis):
    """Return hidden vectors after removing the span of an orthonormal basis."""

    if basis.numel() == 0:
        return hidden
    return hidden - (hidden @ basis.T) @ basis


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
    max_train_points: int | None,
    max_test_points: int | None,
    locked_layer: str,
    probe_epochs: int,
    probe_learning_rate: float,
    random_subspace_count: int,
    max_subspace_rank: int,
    probe_top10_floor: float,
) -> dict[str, Any]:
    torch, nn = require_torch()
    train_config = load_limit_ar_train_config(train_config_path)
    resolved_device = _resolve_device(device, torch)
    tokenizer = LimitTokenizer()
    model = build_tiny_ar_model(
        vocab_size=tokenizer.vocab_size,
        action_size=tokenizer.action_size,
        opponent_card_size=tokenizer.opponent_card_size,
        hidden_size=train_config.hidden_size,
        num_heads=train_config.num_heads,
        num_layers=train_config.num_layers,
    ).to(resolved_device)
    load_info = _load_ar_checkpoint_with_vocab_compat(model, checkpoint_path, torch, resolved_device)
    features = _extract_split_features(
        train_config=train_config,
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        device=resolved_device,
        batch_size=batch_size,
        max_train_points=max_train_points,
        max_test_points=max_test_points,
        locked_layer=locked_layer,
    )
    probe = _train_range_probe(
        train_x=features.train_hidden,
        train_y=features.train_labels,
        input_size=features.train_hidden.shape[1],
        output_size=tokenizer.opponent_card_size,
        torch=torch,
        nn=nn,
        device=resolved_device,
        epochs=probe_epochs,
        learning_rate=probe_learning_rate,
        batch_size=batch_size,
        seed=bootstrap_seed,
    )
    with torch.no_grad():
        probe_logits = probe(features.test_hidden)
        probe_top10 = _topk_accuracy(probe_logits, features.test_labels, torch, 10)
    basis, full_rank, singular_values = _basis_from_probe(probe, torch, max_subspace_rank)
    range_metrics = _behavior_movement(
        model=model,
        original_hidden=features.test_sequence_hidden,
        ablated_hidden=remove_subspace(features.test_sequence_hidden, basis),
        original_policy_logits=features.test_policy_logits,
        original_value=features.test_value,
        torch=torch,
    )
    random_rows = []
    for random_index in range(random_subspace_count):
        random_basis = _random_basis(
            rank=basis.shape[0],
            hidden_size=features.test_sequence_hidden.shape[1],
            torch=torch,
            device=resolved_device,
            seed=bootstrap_seed + 1000 + random_index,
        )
        random_rows.append(
            _behavior_movement(
                model=model,
                original_hidden=features.test_sequence_hidden,
                ablated_hidden=remove_subspace(features.test_sequence_hidden, random_basis),
                original_policy_logits=features.test_policy_logits,
                original_value=features.test_value,
                torch=torch,
            )
        )
    summary = _summary(range_metrics, random_rows, bootstrap_count, bootstrap_seed, confidence_level)
    label = _seed_label(summary, probe_top10, probe_top10_floor)
    return {
        "seed": seed_name,
        "train_config_path": str(train_config_path),
        "train_config_sha256": _sha256_file(train_config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_load_info": load_info,
        "locked_layer": locked_layer,
        "train_points": int(features.train_hidden.shape[0]),
        "test_points": int(features.test_hidden.shape[0]),
        "probe_top10_accuracy": probe_top10,
        "probe_top10_floor": probe_top10_floor,
        "subspace_rank": int(basis.shape[0]),
        "probe_full_row_rank": full_rank,
        "max_subspace_rank": max_subspace_rank,
        "probe_top_singular_values": singular_values,
        "random_subspace_count": random_subspace_count,
        "summary": summary,
        "final_diagnostic_label": label,
    }


def _extract_split_features(
    *,
    train_config,
    model,
    tokenizer: LimitTokenizer,
    torch,
    device: str,
    batch_size: int,
    max_train_points: int | None,
    max_test_points: int | None,
    locked_layer: str,
) -> _SplitFeatures:
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
    train_points = controlled_range_labels(
        controlled_limit_points(splits.train, train_config.input_control),
        train_config.range_label_control,
        train_config.range_label_seed,
    )
    test_points = controlled_range_labels(
        controlled_limit_points(splits.test, train_config.input_control),
        train_config.range_label_control,
        train_config.range_label_seed + 2,
    )
    if max_train_points is not None:
        train_points = train_points[:max_train_points]
    if max_test_points is not None:
        test_points = test_points[:max_test_points]
    train_features = [tokenizer.encode(point) for point in train_points]
    test_features = [tokenizer.encode(point) for point in test_points]
    train_tensors = _move_tensors(_features_to_tensors(train_features, torch), device)
    test_tensors = _move_tensors(_features_to_tensors(test_features, torch), device)
    train_hidden = _encode_layer_batches(model, train_tensors["input_ids"], locked_layer, batch_size, torch)
    test_hidden = _encode_layer_batches(model, test_tensors["input_ids"], locked_layer, batch_size, torch)
    test_sequence_hidden = _encode_layer_batches(model, test_tensors["input_ids"], "sequence_hidden", batch_size, torch)
    policy_rows = []
    value_rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, test_sequence_hidden.shape[0], batch_size):
            hidden = test_sequence_hidden[start : start + batch_size]
            policy_rows.append(model.policy_head(hidden).detach())
            value_rows.append(model.value_head(hidden).squeeze(-1).detach())
    return _SplitFeatures(
        train_hidden=train_hidden,
        train_labels=train_tensors["opponent_card"],
        test_hidden=test_hidden,
        test_sequence_hidden=test_sequence_hidden,
        test_policy_logits=torch.cat(policy_rows, dim=0),
        test_value=torch.cat(value_rows, dim=0),
        test_labels=test_tensors["opponent_card"],
    )


def _encode_layer_batches(model, input_ids, layer_name: str, batch_size: int, torch):
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, input_ids.shape[0], batch_size):
            layers = model.encode_layers(input_ids[start : start + batch_size])
            if layer_name not in layers:
                raise ValueError(f"unknown layer for k.6.3 ablation: {layer_name}")
            rows.append(layers[layer_name].detach())
    return torch.cat(rows, dim=0)


def _train_range_probe(
    *,
    train_x,
    train_y,
    input_size: int,
    output_size: int,
    torch,
    nn,
    device: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
):
    torch.manual_seed(seed)
    probe = nn.Linear(input_size, output_size).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    for _ in range(epochs):
        permutation = torch.randperm(train_x.shape[0], device=device)
        shuffled_x = train_x[permutation]
        shuffled_y = train_y[permutation]
        for start in range(0, train_x.shape[0], batch_size):
            stop = min(start + batch_size, train_x.shape[0])
            batch_x = shuffled_x[start:stop]
            batch_y = shuffled_y[start:stop]
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(probe(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    return probe


def _basis_from_probe(probe, torch, max_subspace_rank: int):
    weight = probe.weight.detach().float()
    _, singular_values, vh = torch.linalg.svd(weight, full_matrices=False)
    if singular_values.numel() == 0:
        return weight[:0], 0, []
    threshold = max(float(singular_values.max().item()) * 1e-6, 1e-8)
    full_rank = int((singular_values > threshold).sum().item())
    rank = min(full_rank, max_subspace_rank)
    return vh[:rank].contiguous(), full_rank, [float(value) for value in singular_values[:rank].detach().cpu().tolist()]


def _random_basis(*, rank: int, hidden_size: int, torch, device: str, seed: int):
    if rank <= 0:
        return torch.empty((0, hidden_size), device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    matrix = torch.randn(hidden_size, rank, generator=generator, device=device)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q.T.contiguous()


def _behavior_movement(*, model, original_hidden, ablated_hidden, original_policy_logits, original_value, torch) -> dict[str, list[float]]:
    with torch.no_grad():
        edited_policy_logits = model.policy_head(ablated_hidden)
        edited_value = model.value_head(ablated_hidden).squeeze(-1)
    return {
        "policy_js": [
            _softmax_js(original_policy_logits[index], edited_policy_logits[index], torch)
            for index in range(original_policy_logits.shape[0])
        ],
        "value_abs_delta": [
            float(abs((original_value[index] - edited_value[index]).item()))
            for index in range(original_value.shape[0])
        ],
        "policy_argmax_changed": [
            float(torch.argmax(original_policy_logits[index]).item() != torch.argmax(edited_policy_logits[index]).item())
            for index in range(original_policy_logits.shape[0])
        ],
    }


def _summary(
    range_metrics: dict[str, list[float]],
    random_rows: list[dict[str, list[float]]],
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    random_policy_by_point = _per_point_mean([row["policy_js"] for row in random_rows])
    random_value_by_point = _per_point_mean([row["value_abs_delta"] for row in random_rows])
    random_changed_by_point = _per_point_mean([row["policy_argmax_changed"] for row in random_rows])
    random_policy_means = [_mean(row["policy_js"]) for row in random_rows]
    random_value_means = [_mean(row["value_abs_delta"]) for row in random_rows]
    random_changed_means = [_mean(row["policy_argmax_changed"]) for row in random_rows]
    random_policy_mean = _mean(random_policy_means)
    random_value_mean = _mean(random_value_means)
    random_changed_mean = _mean(random_changed_means)
    policy_deltas = [
        value - random_policy_by_point[index]
        for index, value in enumerate(range_metrics["policy_js"])
    ]
    value_deltas = [
        value - random_value_by_point[index]
        for index, value in enumerate(range_metrics["value_abs_delta"])
    ]
    changed_deltas = [
        value - random_changed_by_point[index]
        for index, value in enumerate(range_metrics["policy_argmax_changed"])
    ]
    return {
        "range_ablation_policy_js": _mean(range_metrics["policy_js"]),
        "range_ablation_value_abs_delta": _mean(range_metrics["value_abs_delta"]),
        "range_ablation_policy_argmax_changed_rate": _mean(range_metrics["policy_argmax_changed"]),
        "random_ablation_policy_js_mean": random_policy_mean,
        "random_ablation_policy_js_min": min(random_policy_means) if random_policy_means else 0.0,
        "random_ablation_policy_js_max": max(random_policy_means) if random_policy_means else 0.0,
        "random_ablation_value_abs_delta_mean": random_value_mean,
        "random_ablation_value_abs_delta_min": min(random_value_means) if random_value_means else 0.0,
        "random_ablation_value_abs_delta_max": max(random_value_means) if random_value_means else 0.0,
        "random_ablation_policy_argmax_changed_rate_mean": random_changed_mean,
        "range_minus_random_policy_js_delta_ci": _bootstrap_mean_ci(policy_deltas, bootstrap_count, bootstrap_seed, confidence_level),
        "range_minus_random_value_delta_ci": _bootstrap_mean_ci(value_deltas, bootstrap_count, bootstrap_seed + 1, confidence_level),
        "range_minus_random_policy_changed_delta_ci": _bootstrap_mean_ci(changed_deltas, bootstrap_count, bootstrap_seed + 2, confidence_level),
    }


def _seed_label(summary: dict[str, Any], probe_top10: float, probe_top10_floor: float) -> str:
    if probe_top10 < probe_top10_floor:
        return PROBE_NOT_SUPPORTED
    policy_lower = float(summary["range_minus_random_policy_js_delta_ci"]["lower_delta"])
    value_lower = float(summary["range_minus_random_value_delta_ci"]["lower_delta"])
    if max(policy_lower, value_lower) > 0.0:
        return RANGE_FOOTPRINT_SUPPORTED
    policy_upper = float(summary["range_minus_random_policy_js_delta_ci"]["upper_delta"])
    value_upper = float(summary["range_minus_random_value_delta_ci"]["upper_delta"])
    if max(policy_upper, value_upper) <= 0.0:
        return RANGE_FOOTPRINT_NOT_SUPPORTED
    return INCONCLUSIVE


def _aggregate_rows(rows: list[dict[str, Any]], bootstrap_count: int, bootstrap_seed: int, confidence_level: float) -> dict[str, Any]:
    metrics = {}
    for name in (
        "probe_top10_accuracy",
        "range_ablation_policy_js",
        "range_ablation_value_abs_delta",
        "random_ablation_policy_js_mean",
        "random_ablation_value_abs_delta_mean",
    ):
        values = [float(row.get(name, row["summary"].get(name, 0.0))) for row in rows]
        metrics[name] = {
            "mean": _mean(values),
            "seed_ci": _bootstrap_mean_ci(values, bootstrap_count, bootstrap_seed + _stable_int(name) % 1000, confidence_level),
        }
    for name in ("range_minus_random_policy_js_delta_ci", "range_minus_random_value_delta_ci"):
        values = [float(row["summary"][name]["mean_delta"]) for row in rows]
        metrics[name.replace("_ci", "")] = {
            "mean": _mean(values),
            "seed_ci": _bootstrap_mean_ci(values, bootstrap_count, bootstrap_seed + _stable_int(name) % 1000, confidence_level),
        }
    return {
        "seed_count": len(rows),
        "label_counts": _count_labels(row["final_diagnostic_label"] for row in rows),
        "metrics": metrics,
    }


def _aggregate_label(aggregate: dict[str, Any]) -> str:
    labels = set(aggregate["label_counts"])
    if labels == {RANGE_FOOTPRINT_SUPPORTED}:
        return RANGE_FOOTPRINT_SUPPORTED
    if labels == {RANGE_FOOTPRINT_NOT_SUPPORTED}:
        return RANGE_FOOTPRINT_NOT_SUPPORTED
    if labels == {PROBE_NOT_SUPPORTED}:
        return PROBE_NOT_SUPPORTED
    return INCONCLUSIVE


def _manifest(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": payload["version"],
        "report_type": "k.6.3_range_subspace_ablation_manifest",
        "claim_scope": payload["claim_scope"],
        "output_dir": str(output_dir),
        "expected_outputs": [
            "k.6.3_range_subspace_ablation_report.json",
            "k.6.3_range_subspace_ablation_report.md",
            "k.6.3_range_subspace_ablation_manifest.json",
        ],
        "report_checksum": payload["checksum"],
        "final_diagnostic_label": payload["final_diagnostic_label"],
        "one_command_reproduction": (
            ".venv/bin/python scripts/build_range_subspace_ablation_report.py "
            f"--archive-root {payload['archive_root']} "
            f"--output-dir {output_dir} "
            f"--bootstrap-count {payload['bootstrap_count']} "
            f"--bootstrap-seed {payload['bootstrap_seed']} "
            f"--batch-size {payload['batch_size']} "
            f"--locked-layer {payload['locked_layer']} "
            f"--probe-epochs {payload['probe_epochs']} "
            f"--probe-learning-rate {payload['probe_learning_rate']} "
            f"--random-subspace-count {payload['random_subspace_count']} "
            f"--max-subspace-rank {payload['max_subspace_rank']} "
            f"--probe-top10-floor {payload['probe_top10_floor']} "
            + ("--max-train-points {value} ".format(value=payload["max_train_points"]) if payload["max_train_points"] is not None else "")
            + ("--max-test-points {value} ".format(value=payload["max_test_points"]) if payload["max_test_points"] is not None else "")
            + "--seed-names "
            + " ".join(payload["seed_names"])
        ),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# k.6.3 Range-Subspace Ablation Stress Report",
        "",
        "This is a representation-edit stress test. It is not a causal mediation proof.",
        "",
        f"final_diagnostic_label: `{payload['final_diagnostic_label']}`",
        "",
        "| seed | label | probe top10 | rank | range policy JS | random policy JS | policy delta | range value delta | random value delta | value delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        summary = row["summary"]
        lines.append(
            "| {seed} | `{label}` | {probe:.6f} | {rank} | {range_policy:.8f} | {random_policy:.8f} | {policy_delta:.8f} | {range_value:.8f} | {random_value:.8f} | {value_delta:.8f} |".format(
                seed=row["seed"],
                label=row["final_diagnostic_label"],
                probe=row["probe_top10_accuracy"],
                rank=row["subspace_rank"],
                range_policy=summary["range_ablation_policy_js"],
                random_policy=summary["random_ablation_policy_js_mean"],
                policy_delta=summary["range_minus_random_policy_js_delta_ci"]["mean_delta"],
                range_value=summary["range_ablation_value_abs_delta"],
                random_value=summary["random_ablation_value_abs_delta_mean"],
                value_delta=summary["range_minus_random_value_delta_ci"]["mean_delta"],
            )
        )
    lines.extend(["", "## Aggregate", "", "| metric | mean | lower | upper |", "|---|---:|---:|---:|"])
    for name, row in payload["aggregate"]["metrics"].items():
        ci = row["seed_ci"]
        lines.append(f"| {name} | {row['mean']:.8f} | {ci['lower_delta']:.8f} | {ci['upper_delta']:.8f} |")
    lines.extend(["", "## Claim Discipline", "", payload["allowed_paper_claim"], "", f"Forbidden claim: {payload['forbidden_paper_claim']}"])
    return "\n".join(lines) + "\n"


def _topk_accuracy(logits, labels, torch, k: int) -> float:
    topk = torch.topk(logits, k=min(k, logits.shape[1]), dim=1).indices
    matches = topk == labels.unsqueeze(1)
    return float(matches.any(dim=1).float().mean().item())


def _softmax_js(left_logits, right_logits, torch) -> float:
    left = torch.nn.functional.softmax(left_logits, dim=-1).detach().cpu().tolist()
    right = torch.nn.functional.softmax(right_logits, dim=-1).detach().cpu().tolist()
    return jensen_shannon(tuple(float(value) for value in left), tuple(float(value) for value in right))


def _bootstrap_mean_ci(values: list[float], bootstrap_count: int, bootstrap_seed: int, confidence_level: float) -> dict[str, float]:
    if not values:
        return {"mean_delta": 0.0, "lower_delta": 0.0, "upper_delta": 0.0}
    rng = random.Random(bootstrap_seed)
    means = []
    for _ in range(bootstrap_count):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    lower = int(((1.0 - confidence_level) / 2.0) * (bootstrap_count - 1))
    upper = int((1.0 - (1.0 - confidence_level) / 2.0) * (bootstrap_count - 1))
    return {"mean_delta": _mean(values), "lower_delta": means[lower], "upper_delta": means[upper]}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _per_point_mean(rows: list[list[float]]) -> list[float]:
    if not rows:
        return []
    return [sum(row[index] for row in rows) / len(rows) for index in range(len(rows[0]))]


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
