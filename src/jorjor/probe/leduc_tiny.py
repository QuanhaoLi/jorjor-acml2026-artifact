from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.config import load_leduc_tiny_probe_config, load_leduc_tiny_train_config
from jorjor.data.split import SplitFractions, apply_split, build_split_manifest
from jorjor.models.tiny import build_tiny_model, require_torch
from jorjor.tokenizer import LeducTokenizer
from jorjor.train.leduc_tiny import _features_to_tensors, _load_points, _resolve_device


@dataclass(frozen=True)
class LeducTinyProbeResult:
    """Summary returned by the v0.2.2 tiny probe command."""

    layer_count: int
    best_real_score: float
    best_control_score: float
    metric_name: str
    checksum: str
    artifact_dir: Path
    device: str


def _batch_indices(total: int, batch_size: int):
    if batch_size <= 0 or batch_size >= total:
        yield slice(0, total)
        return
    for start in range(0, total, batch_size):
        yield slice(start, min(start + batch_size, total))


def _accuracy(logits, labels, torch) -> float:
    predictions = torch.argmax(logits, dim=1)
    return float((predictions == labels).float().mean().item())


def _is_classification_target(target: str) -> bool:
    return target in {"opponent_card", "action"}


def _num_classes(train_y, torch) -> int:
    return int(torch.max(train_y).item()) + 1


def _train_classification_probe(train_x, train_y, test_x, test_y, config, torch, nn, device: str):
    torch.manual_seed(config.probe_seed)
    probe = nn.Linear(train_x.shape[1], _num_classes(train_y, torch)).to(device)
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
        accuracy = _accuracy(probe(test_x), test_y, torch)
    return {
        "metric_name": "accuracy",
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "accuracy": accuracy,
        "score": accuracy,
    }


def _train_regression_probe(train_x, train_y, test_x, test_y, config, torch, nn, device: str):
    torch.manual_seed(config.probe_seed)
    probe = nn.Linear(train_x.shape[1], 1).to(device)
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
    return {layer: encoded[layer].detach() for layer in layers}


def _target_tensors(config, train_tensors, test_tensors):
    if config.target == "opponent_card":
        return train_tensors["opponent_card"], test_tensors["opponent_card"]
    if config.target == "action":
        return train_tensors["policy"], test_tensors["policy"]
    if config.target == "value":
        return train_tensors["value"], test_tensors["value"]
    raise ValueError(f"unknown probe target: {config.target}")


def _controlled_points(points, input_control: str, seed: int):
    if input_control == "none":
        return points

    rng = random.Random(seed)
    controlled = []
    for point in points:
        state = dict(point.visible_state)
        if input_control == "shuffle_history":
            history = list(state["betting_history"])
            action_indices = [index for index, event in enumerate(history) if event["type"] == "action"]
            action_events = [history[index] for index in action_indices]
            rng.shuffle(action_events)
            for index, event in zip(action_indices, action_events, strict=True):
                history[index] = event
            state["betting_history"] = history
        elif input_control == "drop_history":
            state["betting_history"] = []
        elif input_control == "mask_public_card":
            state["public_card"] = None
        elif input_control == "mask_private_card":
            state["private_card"] = "mask"
        elif input_control == "drop_history_mask_cards":
            state["betting_history"] = []
            state["public_card"] = None
            state["private_card"] = "mask"
        else:
            raise ValueError(f"unknown input control: {input_control}")
        controlled.append(point.__class__(
            hand_id=point.hand_id,
            step_index=point.step_index,
            acting_player=point.acting_player,
            visible_state=state,
            legal_actions=point.legal_actions,
            action=point.action,
            opponent_private_card=point.opponent_private_card,
            opponent_rank=point.opponent_rank,
            terminal_return=point.terminal_return,
        ))
    return tuple(controlled)


def _train_probe(train_x, train_y, test_x, test_y, config, torch, nn, device: str):
    if _is_classification_target(config.target):
        return _train_classification_probe(train_x, train_y, test_x, test_y, config, torch, nn, device)
    return _train_regression_probe(train_x, train_y, test_x, test_y, config, torch, nn, device)


def run_leduc_tiny_probe(config_path: Path, artifact_dir: Path | None = None) -> LeducTinyProbeResult:
    """Run a tiny frozen-model opponent-card probe."""

    torch, nn = require_torch()
    config = load_leduc_tiny_probe_config(config_path)
    train_config = load_leduc_tiny_train_config(config.train_config)
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

    tokenizer = LeducTokenizer()
    train_points = _controlled_points(splits.train, config.input_control, seed=config.control_seed)
    test_points = _controlled_points(splits.test, config.input_control, seed=config.control_seed + 1)
    train_features = [tokenizer.encode(point) for point in train_points]
    test_features = [tokenizer.encode(point) for point in test_points]
    train_tensors = _features_to_tensors(train_features, torch)
    test_tensors = _features_to_tensors(test_features, torch)
    train_tensors = {key: value.to(device) for key, value in train_tensors.items()}
    test_tensors = {key: value.to(device) for key, value in test_tensors.items()}

    model = build_tiny_model(
        vocab_size=tokenizer.vocab_size,
        action_size=tokenizer.action_size,
        opponent_card_size=tokenizer.opponent_card_size,
        hidden_size=train_config.hidden_size,
    ).to(device)
    model.load_state_dict(torch.load(config.checkpoint_path, map_location=device, weights_only=True))

    train_layers = _extract_layers(model, train_tensors, config.layers, torch)
    test_layers = _extract_layers(model, test_tensors, config.layers, torch)
    true_train_y, true_test_y = _target_tensors(config, train_tensors, test_tensors)

    records = []
    control_generator = torch.Generator(device=device)
    control_generator.manual_seed(config.control_seed)
    control_train_y = true_train_y[torch.randperm(true_train_y.shape[0], generator=control_generator, device=device)]

    for layer in config.layers:
        metrics = _train_probe(
            train_layers[layer],
            true_train_y,
            test_layers[layer],
            true_test_y,
            config,
            torch,
            nn,
            device,
        )
        records.append(
            {
                "layer": layer,
                "target": config.target,
                "control": "none",
                **metrics,
            }
        )
        if config.random_label_control:
            metrics = _train_probe(
                train_layers[layer],
                control_train_y,
                test_layers[layer],
                true_test_y,
                config,
                torch,
                nn,
                device,
            )
            records.append(
                {
                    "layer": layer,
                    "target": config.target,
                    "control": "random_label",
                    **metrics,
                }
            )

    summary = {
        "version": config.run_version,
        "train_config": str(config.train_config),
        "checkpoint_path": str(config.checkpoint_path),
        "target": config.target,
        "input_control": config.input_control,
        "device": device,
        "train_points": len(splits.train),
        "test_points": len(splits.test),
        "layers": list(config.layers),
        "records": records,
    }
    checksum = checksum_payload(summary)
    write_json_artifact(output_dir / "probe_metrics.json", {**summary, "checksum": checksum})
    with (output_dir / "probe_table.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    if _is_classification_target(config.target):
        real_scores = [record["score"] for record in records if record["control"] == "none"]
        control_scores = [record["score"] for record in records if record["control"] != "none"]
        best_real_score = max(real_scores)
        best_control_score = max(control_scores) if control_scores else 0.0
    else:
        real_scores = [record["score"] for record in records if record["control"] == "none"]
        control_scores = [record["score"] for record in records if record["control"] != "none"]
        best_real_score = min(real_scores)
        best_control_score = min(control_scores) if control_scores else 0.0
    return LeducTinyProbeResult(
        layer_count=len(config.layers),
        best_real_score=best_real_score,
        best_control_score=best_control_score,
        metric_name=records[0]["metric_name"],
        checksum=checksum,
        artifact_dir=output_dir,
        device=device,
    )
