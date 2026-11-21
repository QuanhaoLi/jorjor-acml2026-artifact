from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.config import load_limit_ar_probe_config, load_limit_ar_train_config
from jorjor.data.split import SplitFractions, apply_split, build_split_manifest
from jorjor.models.ar import build_tiny_ar_model
from jorjor.models.tiny import require_torch
from jorjor.tokenizer import ACTIONS, LIMIT_ROUNDS, LimitTokenizer
from jorjor.train.leduc_tiny import _batch_indices, _features_to_tensors, _move_tensors, _resolve_device
from jorjor.train.limit_ar import _accuracy
from jorjor.train.limit_controls import controlled_limit_points
from jorjor.train.limit_tiny import _load_points, _topk_accuracy


@dataclass(frozen=True)
class LimitArProbeResult:
    """Summary returned by the v0.7.0 Limit AR probe command."""

    layer_count: int
    best_real_score: float
    best_control_score: float
    metric_name: str
    checksum: str
    artifact_dir: Path
    device: str


def _is_classification_target(target: str) -> bool:
    return target in {"opponent_hand", "action"}


def _num_classes(config, tokenizer: LimitTokenizer) -> int:
    if config.target == "opponent_hand":
        return tokenizer.opponent_card_size
    if config.target == "action":
        return tokenizer.action_size
    raise ValueError(f"class count is not defined for target: {config.target}")


def _point_identity(point, index: int) -> dict[str, object]:
    """Return stable test-point identity for probe correctness rows."""

    return {
        "split": "test",
        "example_index": index,
        "hand_id": point.hand_id,
        "step_index": point.step_index,
        "point_id": f"{point.hand_id}:{point.step_index}",
    }


def _correctness_metadata(records: list[dict[str, object]]) -> dict[str, object]:
    rows_by_layer_control: dict[str, int] = {}
    for record in records:
        key = f"{record['layer']}|{record['control']}"
        rows_by_layer_control[key] = rows_by_layer_control.get(key, 0) + 1
    return {
        "row_count": len(records),
        "rows_by_layer_control": dict(sorted(rows_by_layer_control.items())),
    }


def _jsonl_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classification_probe_module(input_size: int, output_size: int, config, nn):
    if config.probe_family == "linear":
        return nn.Linear(input_size, output_size)
    if config.probe_family == "mlp":
        return nn.Sequential(
            nn.Linear(input_size, config.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(config.mlp_hidden_size, output_size),
        )
    raise ValueError(f"unknown probe_family: {config.probe_family}")


def _regression_probe_module(input_size: int, config, nn):
    if config.probe_family == "linear":
        return nn.Linear(input_size, 1)
    if config.probe_family == "mlp":
        return nn.Sequential(
            nn.Linear(input_size, config.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(config.mlp_hidden_size, 1),
        )
    raise ValueError(f"unknown probe_family: {config.probe_family}")


def _train_classification_probe(train_x, train_y, test_x, test_y, config, tokenizer, torch, nn, device: str):
    torch.manual_seed(config.probe_seed)
    probe = _classification_probe_module(train_x.shape[1], _num_classes(config, tokenizer), config, nn).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    for _ in range(config.epochs):
        permutation = torch.randperm(train_x.shape[0], device=device)
        shuffled_x = train_x[permutation]
        shuffled_y = train_y[permutation]
        epoch_loss = 0.0
        batch_count = 0
        for batch_slice in _batch_indices(train_x.shape[0], config.batch_size):
            batch_x = shuffled_x[batch_slice]
            batch_y = shuffled_y[batch_slice]
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(probe(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batch_count += 1
        losses.append(epoch_loss / batch_count)
    with torch.no_grad():
        logits = probe(test_x)
        top1 = _accuracy(logits, test_y, torch)
        top5 = _topk_accuracy(logits, test_y, torch, k=5)
        top10 = _topk_accuracy(logits, test_y, torch, k=10)
        k10 = min(10, logits.shape[1])
        topk = torch.topk(logits, k=k10, dim=1).indices
        target = test_y.unsqueeze(1)
        correctness_rows = []
        for index in range(test_y.shape[0]):
            row_hits = topk[index].eq(target[index])
            correctness_rows.append(
                {
                    "example_index": index,
                    "target_label": int(test_y[index].detach().cpu().item()),
                    "top1_correct": bool(row_hits[:1].any().detach().cpu().item()),
                    "top5_correct": bool(row_hits[: min(5, k10)].any().detach().cpu().item()),
                    "top10_correct": bool(row_hits.any().detach().cpu().item()),
                }
            )
    if config.target == "opponent_hand":
        score = top10
        metric_name = "top10_accuracy"
    else:
        score = top1
        metric_name = "accuracy"
    return {
        "metric_name": metric_name,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "accuracy": top1,
        "top5_accuracy": top5,
        "top10_accuracy": top10,
        "score": score,
        "correctness_rows": correctness_rows,
    }


def _train_regression_probe(train_x, train_y, test_x, test_y, config, torch, nn, device: str):
    torch.manual_seed(config.probe_seed)
    probe = _regression_probe_module(train_x.shape[1], config, nn).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    train_y = train_y.float()
    test_y = test_y.float()
    for _ in range(config.epochs):
        permutation = torch.randperm(train_x.shape[0], device=device)
        shuffled_x = train_x[permutation]
        shuffled_y = train_y[permutation]
        epoch_loss = 0.0
        batch_count = 0
        for batch_slice in _batch_indices(train_x.shape[0], config.batch_size):
            batch_x = shuffled_x[batch_slice]
            batch_y = shuffled_y[batch_slice]
            optimizer.zero_grad()
            prediction = probe(batch_x).squeeze(-1)
            loss = torch.nn.functional.mse_loss(prediction, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batch_count += 1
        losses.append(epoch_loss / batch_count)
    with torch.no_grad():
        mse = float(torch.nn.functional.mse_loss(probe(test_x).squeeze(-1), test_y).item())
    return {
        "metric_name": "mse",
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mse": mse,
        "score": mse,
    }


def _extract_layers(model, tensors, layers: tuple[str, ...], torch) -> dict[str, object]:
    model.eval()
    with torch.no_grad():
        encoded = model.encode_layers(tensors["input_ids"])
    missing = [layer for layer in layers if layer not in encoded]
    if missing:
        raise ValueError(f"unknown model layer(s): {', '.join(missing)}")
    return {layer: encoded[layer].detach() for layer in layers}


def _target_tensors(config, train_tensors, test_tensors):
    if config.target == "opponent_hand":
        return train_tensors["opponent_card"], test_tensors["opponent_card"]
    if config.target == "action":
        return train_tensors["policy"], test_tensors["policy"]
    if config.target == "value":
        return train_tensors["value"], test_tensors["value"]
    raise ValueError(f"unknown probe target: {config.target}")


def _feature_control_uses_action(feature_control: str) -> bool:
    return feature_control in {
        "action_only",
        "action_value_only",
        "action_value_history_composition_only",
        "residualize_action",
        "residualize_action_value",
        "residualize_action_value_history_composition",
    }


def _feature_control_uses_value(feature_control: str) -> bool:
    return feature_control in {
        "value_only",
        "action_value_only",
        "action_value_history_composition_only",
        "residualize_value",
        "residualize_action_value",
        "residualize_action_value_history_composition",
    }


def _feature_control_uses_history_composition(feature_control: str) -> bool:
    return feature_control in {
        "history_composition_only",
        "action_value_history_composition_only",
        "residualize_action_value_history_composition",
    }


def _is_nuisance_only_feature_control(feature_control: str) -> bool:
    return feature_control in {
        "action_only",
        "value_only",
        "action_value_only",
        "history_composition_only",
        "action_value_history_composition_only",
        "round_position_action_only",
    }


def _is_residual_feature_control(feature_control: str) -> bool:
    return feature_control in {
        "residualize_action",
        "residualize_value",
        "residualize_action_value",
        "residualize_action_value_history_composition",
    }


def _standardized_value_feature(train_value, test_value, torch):
    train_feature = train_value.float().unsqueeze(1)
    test_feature = test_value.float().unsqueeze(1)
    mean = train_feature.mean(dim=0, keepdim=True)
    std = train_feature.std(dim=0, unbiased=False, keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return (train_feature - mean) / std, (test_feature - mean) / std


def _nuisance_features(feature_control: str, train_tensors, test_tensors, tokenizer, torch):
    train_parts = []
    test_parts = []
    if _feature_control_uses_action(feature_control):
        train_parts.append(
            torch.nn.functional.one_hot(train_tensors["policy"], num_classes=tokenizer.action_size).float()
        )
        test_parts.append(torch.nn.functional.one_hot(test_tensors["policy"], num_classes=tokenizer.action_size).float())
    if _feature_control_uses_value(feature_control):
        train_value, test_value = _standardized_value_feature(train_tensors["value"], test_tensors["value"], torch)
        train_parts.append(train_value)
        test_parts.append(test_value)
    if not train_parts:
        raise ValueError(f"feature_control has no nuisance features: {feature_control}")
    return torch.cat(train_parts, dim=1), torch.cat(test_parts, dim=1)


def _round_position_action_prior_features(points, tensors, tokenizer, torch):
    rows = []
    player_values = ("0", "1", "mask")
    round_values = (*LIMIT_ROUNDS, "mask")
    for point in points:
        state = point.visible_state
        acting_player = str(state["acting_player"])
        round_name = str(state["round"])
        legal_actions = set(state["legal_actions"])
        row = []
        row.extend(1.0 if point.action == action else 0.0 for action in ACTIONS)
        row.extend(1.0 if acting_player == value else 0.0 for value in player_values)
        row.extend(1.0 if round_name == value else 0.0 for value in round_values)
        row.extend(1.0 if action in legal_actions else 0.0 for action in ACTIONS)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32, device=tensors["policy"].device)


def _history_action_events(point) -> list[str]:
    return [
        str(event["action"])
        for event in point.visible_state["betting_history"]
        if event["type"] in {"action", "bag_action"}
    ]


def _history_composition_features(points, tensors, torch):
    feature_count = 30
    rows = []
    for point in points:
        actions = _history_action_events(point)
        action_count = max(len(actions), 1)
        row = []
        row.append(float(len(actions)))
        row.append(min(float(len(actions)), 16.0) / 16.0)
        row.extend(float(actions.count(action)) for action in ACTIONS)
        row.extend(float(actions.count(action)) / action_count for action in ACTIONS)
        row.extend(1.0 if action in actions else 0.0 for action in ACTIONS)
        first_action = actions[0] if actions else None
        last_action = actions[-1] if actions else None
        row.extend(1.0 if first_action == action else 0.0 for action in ACTIONS)
        row.extend(1.0 if last_action == action else 0.0 for action in ACTIONS)
        for action in ACTIONS:
            positions = [index / 15.0 for index, history_action in enumerate(actions[:16]) if history_action == action]
            row.append(sum(positions) / len(positions) if positions else 0.0)
            row.append(max(positions) if positions else 0.0)
        rows.append(row)
    if not rows:
        return torch.empty((0, feature_count), dtype=torch.float32, device=tensors["policy"].device)
    return torch.tensor(rows, dtype=torch.float32, device=tensors["policy"].device)


def _controlled_nuisance_features(feature_control: str, train_points, test_points, train_tensors, test_tensors, tokenizer, torch):
    if feature_control == "round_position_action_only":
        return (
            _round_position_action_prior_features(train_points, train_tensors, tokenizer, torch),
            _round_position_action_prior_features(test_points, test_tensors, tokenizer, torch),
        )
    train_parts = []
    test_parts = []
    if _feature_control_uses_action(feature_control) or _feature_control_uses_value(feature_control):
        train_base, test_base = _nuisance_features(feature_control, train_tensors, test_tensors, tokenizer, torch)
        train_parts.append(train_base)
        test_parts.append(test_base)
    if _feature_control_uses_history_composition(feature_control):
        train_parts.append(_history_composition_features(train_points, train_tensors, torch))
        test_parts.append(_history_composition_features(test_points, test_tensors, torch))
    if not train_parts:
        raise ValueError(f"feature_control has no nuisance features: {feature_control}")
    return torch.cat(train_parts, dim=1), torch.cat(test_parts, dim=1)


def _with_intercept(features, torch):
    ones = torch.ones((features.shape[0], 1), dtype=features.dtype, device=features.device)
    return torch.cat([ones, features], dim=1)


def _residualize_layer(train_x, test_x, train_nuisance, test_nuisance, torch):
    train_design = _with_intercept(train_nuisance, torch)
    test_design = _with_intercept(test_nuisance, torch)
    coefficients = torch.linalg.pinv(train_design).matmul(train_x)
    return train_x - train_design.matmul(coefficients), test_x - test_design.matmul(coefficients)


def _apply_feature_control(config, train_layers, test_layers, train_points, test_points, train_tensors, test_tensors, tokenizer, torch):
    if config.feature_control == "none":
        return train_layers, test_layers

    train_nuisance, test_nuisance = _controlled_nuisance_features(
        config.feature_control,
        train_points,
        test_points,
        train_tensors,
        test_tensors,
        tokenizer,
        torch,
    )
    if _is_nuisance_only_feature_control(config.feature_control):
        return {"nuisance": train_nuisance.detach()}, {"nuisance": test_nuisance.detach()}
    if _is_residual_feature_control(config.feature_control):
        residual_train = {}
        residual_test = {}
        for layer in config.layers:
            residual_train[layer], residual_test[layer] = _residualize_layer(
                train_layers[layer],
                test_layers[layer],
                train_nuisance,
                test_nuisance,
                torch,
            )
        return residual_train, residual_test
    raise ValueError(f"unknown feature control: {config.feature_control}")


def _train_probe(train_x, train_y, test_x, test_y, config, tokenizer, torch, nn, device: str):
    if _is_classification_target(config.target):
        return _train_classification_probe(train_x, train_y, test_x, test_y, config, tokenizer, torch, nn, device)
    return _train_regression_probe(train_x, train_y, test_x, test_y, config, torch, nn, device)


def _probe_capacity_metadata(config) -> dict[str, object]:
    return {
        "probe_family": config.probe_family,
        "probe_capacity_role": "main_linear" if config.probe_family == "linear" else "appendix_sensitivity",
        "mlp_hidden_size": config.mlp_hidden_size if config.probe_family == "mlp" else None,
    }


def _load_ar_checkpoint_with_vocab_compat(model, checkpoint_path: Path, torch, device: str) -> dict[str, object]:
    """Load old Limit AR checkpoints after append-only tokenizer growth."""

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_state = model.state_dict()
    load_info = {
        "checkpoint_embedding_compat": False,
        "checkpoint_embedding_rows": int(state_dict["embedding.weight"].shape[0]),
        "model_embedding_rows": int(model_state["embedding.weight"].shape[0]),
    }

    checkpoint_embedding = state_dict["embedding.weight"]
    model_embedding = model_state["embedding.weight"]
    if checkpoint_embedding.shape != model_embedding.shape:
        old_rows, old_dim = checkpoint_embedding.shape
        new_rows, new_dim = model_embedding.shape
        if old_rows > new_rows or old_dim != new_dim:
            raise ValueError(
                "checkpoint embedding shape is incompatible with model embedding shape: "
                f"{tuple(checkpoint_embedding.shape)} vs {tuple(model_embedding.shape)}"
            )
        expanded_embedding = model_embedding.clone()
        expanded_embedding.zero_()
        expanded_embedding[:old_rows] = checkpoint_embedding
        state_dict = dict(state_dict)
        state_dict["embedding.weight"] = expanded_embedding
        load_info["checkpoint_embedding_compat"] = True

    model.load_state_dict(state_dict)
    return load_info


def _stable_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _bucket_key(point) -> tuple[object, ...]:
    state = point.visible_state
    return (
        str(state["acting_player"]),
        str(state["round"]),
        tuple(sorted(state["legal_actions"])),
        point.action,
    )


def _shuffle_labels_global(labels: list[int], seed: int) -> list[int]:
    shuffled = list(labels)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _shuffle_labels_within_bucket(labels: list[int], points, seed: int) -> tuple[list[int], dict[str, int]]:
    grouped: dict[tuple[object, ...], list[int]] = {}
    for index, point in enumerate(points):
        grouped.setdefault(_bucket_key(point), []).append(index)
    shuffled = list(labels)
    singleton_count = 0
    for key, indices in grouped.items():
        if len(indices) == 1:
            singleton_count += 1
            continue
        bucket_labels = [labels[index] for index in indices]
        random.Random(seed + _stable_int(repr(key))).shuffle(bucket_labels)
        for index, label in zip(indices, bucket_labels, strict=True):
            shuffled[index] = label
    return shuffled, {
        "bucket_count": len(grouped),
        "singleton_bucket_count": singleton_count,
    }


def _tensor_from_labels(labels: list[int], like, torch):
    return torch.tensor(labels, dtype=like.dtype, device=like.device)


def _apply_label_control(config, train_y, test_y, train_points, test_points, torch):
    if config.label_control == "none":
        return train_y, test_y, {
            "control": "none",
            "train_unchanged_count": int(train_y.shape[0]),
            "test_unchanged_count": int(test_y.shape[0]),
        }
    if config.target != "opponent_hand":
        raise ValueError("label_control is only supported for opponent_hand probes")

    train_labels = [int(value) for value in train_y.detach().cpu().tolist()]
    test_labels = [int(value) for value in test_y.detach().cpu().tolist()]
    train_extra = {}
    test_extra = {}
    if config.label_control == "shuffle_global":
        controlled_train = _shuffle_labels_global(train_labels, config.control_seed)
        controlled_test = _shuffle_labels_global(test_labels, config.control_seed + 1)
    elif config.label_control == "shuffle_within_bucket":
        controlled_train, train_extra = _shuffle_labels_within_bucket(train_labels, train_points, config.control_seed)
        controlled_test, test_extra = _shuffle_labels_within_bucket(test_labels, test_points, config.control_seed + 1)
    else:
        raise ValueError(f"unknown label_control: {config.label_control}")

    stats = {
        "control": config.label_control,
        "train_unchanged_count": sum(
            1 for original, controlled in zip(train_labels, controlled_train, strict=True) if original == controlled
        ),
        "test_unchanged_count": sum(
            1 for original, controlled in zip(test_labels, controlled_test, strict=True) if original == controlled
        ),
        "train_point_count": len(train_labels),
        "test_point_count": len(test_labels),
        "train_bucket_count": int(train_extra.get("bucket_count", 1)),
        "test_bucket_count": int(test_extra.get("bucket_count", 1)),
        "train_singleton_bucket_count": int(train_extra.get("singleton_bucket_count", 0)),
        "test_singleton_bucket_count": int(test_extra.get("singleton_bucket_count", 0)),
    }
    return _tensor_from_labels(controlled_train, train_y, torch), _tensor_from_labels(controlled_test, test_y, torch), stats


def run_limit_ar_probe(config_path: Path, artifact_dir: Path | None = None) -> LimitArProbeResult:
    """Run a frozen-model Limit AR layer probe."""

    torch, nn = require_torch()
    config = load_limit_ar_probe_config(config_path)
    train_config = load_limit_ar_train_config(config.train_config)
    if config.feature_control != "none" and config.target != "opponent_hand":
        raise ValueError("feature_control is only supported for opponent_hand probes")
    device = _resolve_device(config.device, torch)
    output_dir = artifact_dir if artifact_dir is not None else config.artifact_dir
    output_dir.mkdir(parents=True, exist_ok=True)

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
    train_points = controlled_limit_points(splits.train, config.input_control)
    test_points = controlled_limit_points(splits.test, config.input_control)
    train_features = [tokenizer.encode(point) for point in train_points]
    test_features = [tokenizer.encode(point) for point in test_points]
    train_tensors = _move_tensors(_features_to_tensors(train_features, torch), device)
    test_tensors = _move_tensors(_features_to_tensors(test_features, torch), device)

    checkpoint_load_info = {
        "checkpoint_loaded": False,
        "checkpoint_embedding_compat": False,
        "checkpoint_embedding_rows": None,
        "model_embedding_rows": None,
    }
    train_layers = {}
    test_layers = {}
    if not _is_nuisance_only_feature_control(config.feature_control):
        model = build_tiny_ar_model(
            vocab_size=tokenizer.vocab_size,
            action_size=tokenizer.action_size,
            opponent_card_size=tokenizer.opponent_card_size,
            hidden_size=train_config.hidden_size,
            num_heads=train_config.num_heads,
            num_layers=train_config.num_layers,
        ).to(device)
        checkpoint_load_info = {
            **_load_ar_checkpoint_with_vocab_compat(model, config.checkpoint_path, torch, device),
            "checkpoint_loaded": True,
        }

        train_layers = _extract_layers(model, train_tensors, config.layers, torch)
        test_layers = _extract_layers(model, test_tensors, config.layers, torch)
    train_layers, test_layers = _apply_feature_control(
        config,
        train_layers,
        test_layers,
        train_points,
        test_points,
        train_tensors,
        test_tensors,
        tokenizer,
        torch,
    )
    true_train_y, true_test_y = _target_tensors(config, train_tensors, test_tensors)
    true_train_y, true_test_y, label_control_stats = _apply_label_control(
        config,
        true_train_y,
        true_test_y,
        train_points,
        test_points,
        torch,
    )

    records = []
    correctness_records = []
    test_identities = [_point_identity(point, index) for index, point in enumerate(test_points)]
    control_generator = torch.Generator(device=device)
    control_generator.manual_seed(config.control_seed)
    control_train_y = true_train_y[torch.randperm(true_train_y.shape[0], generator=control_generator, device=device)]
    capacity_metadata = _probe_capacity_metadata(config)

    for layer in config.layers:
        metrics = _train_probe(
            train_layers[layer],
            true_train_y,
            test_layers[layer],
            true_test_y,
            config,
            tokenizer,
            torch,
            nn,
            device,
        )
        correctness_rows = metrics.pop("correctness_rows", [])
        record = {
            "layer": layer,
            "target": config.target,
            "feature_control": config.feature_control,
            "label_control": config.label_control,
            **capacity_metadata,
            "control": "none",
            **metrics,
        }
        records.append(record)
        for correctness_row in correctness_rows:
            point_identity = test_identities[int(correctness_row["example_index"])]
            correctness_records.append(
                {
                    "layer": layer,
                    "target": config.target,
                    "feature_control": config.feature_control,
                    "label_control": config.label_control,
                    **capacity_metadata,
                    "control": "none",
                    **point_identity,
                    **correctness_row,
                }
            )
        if config.random_label_control:
            metrics = _train_probe(
                train_layers[layer],
                control_train_y,
                test_layers[layer],
                true_test_y,
                config,
                tokenizer,
                torch,
                nn,
                device,
            )
            correctness_rows = metrics.pop("correctness_rows", [])
            records.append(
                {
                    "layer": layer,
                    "target": config.target,
                    "feature_control": config.feature_control,
                    "label_control": config.label_control,
                    **capacity_metadata,
                    "control": "random_label",
                    **metrics,
                }
            )
            for correctness_row in correctness_rows:
                point_identity = test_identities[int(correctness_row["example_index"])]
                correctness_records.append(
                    {
                        "layer": layer,
                        "target": config.target,
                        "feature_control": config.feature_control,
                        "label_control": config.label_control,
                        **capacity_metadata,
                        "control": "random_label",
                        **point_identity,
                        **correctness_row,
                    }
                )

    correctness_artifact = "probe_correctness.jsonl" if _is_classification_target(config.target) else None
    correctness_summary = None
    correctness_sha256 = None
    if correctness_records:
        correctness_path = output_dir / "probe_correctness.jsonl"
        with correctness_path.open("w", encoding="utf-8") as handle:
            for record in correctness_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        correctness_summary = _correctness_metadata(correctness_records)
        correctness_sha256 = _jsonl_sha256(correctness_path)

    summary = {
        "version": config.run_version,
        "config_path": str(config_path),
        "train_config": str(config.train_config),
        "checkpoint_path": str(config.checkpoint_path),
        "artifact_dir": str(output_dir),
        "configured_artifact_dir": str(config.artifact_dir),
        "target": config.target,
        "input_control": config.input_control,
        "feature_control": config.feature_control,
        "label_control": config.label_control,
        "probe_seed": config.probe_seed,
        "control_seed": config.control_seed,
        **capacity_metadata,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "random_label_control": config.random_label_control,
        "label_control_stats": label_control_stats,
        "correctness_artifact": correctness_artifact,
        "correctness_sha256": correctness_sha256,
        "correctness_summary": correctness_summary,
        "device": device,
        "train_points": len(splits.train),
        "test_points": len(splits.test),
        "layers": list(config.layers),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        **checkpoint_load_info,
        "records": records,
    }
    checksum = checksum_payload(summary)
    write_json_artifact(output_dir / "probe_metrics.json", {**summary, "checksum": checksum})
    with (output_dir / "probe_table.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    real_scores = [record["score"] for record in records if record["control"] == "none"]
    control_scores = [record["score"] for record in records if record["control"] != "none"]
    if _is_classification_target(config.target):
        best_real_score = max(real_scores)
        best_control_score = max(control_scores) if control_scores else 0.0
    else:
        best_real_score = min(real_scores)
        best_control_score = min(control_scores) if control_scores else 0.0
    return LimitArProbeResult(
        layer_count=len(config.layers),
        best_real_score=best_real_score,
        best_control_score=best_control_score,
        metric_name=records[0]["metric_name"],
        checksum=checksum,
        artifact_dir=output_dir,
        device=device,
    )
