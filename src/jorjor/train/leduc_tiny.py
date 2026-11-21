from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.config import load_leduc_tiny_train_config
from jorjor.data.baseline import evaluate_card_removal_baseline
from jorjor.data.split import SplitFractions, apply_split, build_split_manifest
from jorjor.data.validation import validate_decision_points
from jorjor.envs.leduc import LeducRandomGenerator
from jorjor.leduc_cfr_teacher import LeducCfrPolicy, train_leduc_cfr
from jorjor.leduc_random import flatten_decision_points
from jorjor.leduc_teacher import generate_teacher_points
from jorjor.models.tiny import build_tiny_model, require_torch
from jorjor.tokenizer import LeducTokenizer


@dataclass(frozen=True)
class LeducTinyTrainResult:
    """Summary for a versioned tiny training run."""

    train_points: int
    validation_points: int
    test_points: int
    first_loss: float
    last_loss: float
    policy_accuracy: float
    value_mse: float
    opponent_card_accuracy: float
    baseline_accuracy: float
    baseline_cross_entropy: float
    checksum: str
    artifact_dir: Path
    device: str


def _features_to_tensors(features, torch):
    max_len = max(len(feature.input_ids) for feature in features)
    input_rows = []
    policy = []
    value = []
    opponent_card = []
    for feature in features:
        row = list(feature.input_ids) + [-1] * (max_len - len(feature.input_ids))
        input_rows.append(row)
        policy.append(feature.policy_label)
        value.append(feature.value_label)
        opponent_card.append(feature.opponent_card_label)
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "policy": torch.tensor(policy, dtype=torch.long),
        "value": torch.tensor(value, dtype=torch.float32),
        "opponent_card": torch.tensor(opponent_card, dtype=torch.long),
    }


def _move_tensors(tensors, device: str):
    return {key: value.to(device) for key, value in tensors.items()}


def _loss(outputs, tensors, torch):
    ce = torch.nn.functional.cross_entropy
    mse = torch.nn.functional.mse_loss
    policy_loss = ce(outputs["policy_logits"], tensors["policy"])
    value_loss = mse(outputs["value"], tensors["value"])
    opponent_loss = ce(outputs["opponent_card_logits"], tensors["opponent_card"])
    return policy_loss + 0.3 * value_loss + 0.7 * opponent_loss


def _accuracy(logits, labels, torch) -> float:
    predictions = torch.argmax(logits, dim=1)
    return float((predictions == labels).float().mean().item())


def _resolve_device(config_device: str, torch) -> str:
    if config_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if config_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return config_device


def _batch_indices(total: int, batch_size: int):
    if batch_size <= 0 or batch_size >= total:
        yield slice(0, total)
        return
    for start in range(0, total, batch_size):
        yield slice(start, min(start + batch_size, total))


def _slice_tensors(tensors, index):
    return {key: value[index] for key, value in tensors.items()}


def _load_points(config):
    if config.data_source == "random":
        generator = LeducRandomGenerator(seed=config.data_seed)
        return flatten_decision_points(generator.generate_hands(config.hand_count))
    if config.data_source == "teacher":
        return generate_teacher_points(hand_count=config.hand_count, seed=config.data_seed)
    if config.data_source == "cfr_teacher":
        cfr_result = train_leduc_cfr(iteration_count=200, seed=config.data_seed)
        policy = LeducCfrPolicy(cfr_result.policy_table, seed=config.data_seed + 97)
        return policy.generate_points(hand_count=config.hand_count, seed=config.data_seed + 193)
    raise ValueError(f"unknown data source: {config.data_source}")


def run_leduc_tiny_train(config_path: Path, artifact_dir: Path | None = None) -> LeducTinyTrainResult:
    """Run a tiny Leduc training experiment."""

    torch, _ = require_torch()
    config = load_leduc_tiny_train_config(config_path)
    torch.manual_seed(config.seed)
    device = _resolve_device(config.device, torch)

    output_dir = artifact_dir if artifact_dir is not None else config.artifact_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    points = _load_points(config)
    validation_report = validate_decision_points(points)
    manifest = build_split_manifest(
        points=points,
        fractions=SplitFractions(
            train=config.train_fraction,
            validation=config.validation_fraction,
            test=config.test_fraction,
        ),
        seed=config.split_seed,
    )
    splits = apply_split(points, manifest)
    baseline = evaluate_card_removal_baseline(splits.test)

    tokenizer = LeducTokenizer()
    train_features = [tokenizer.encode(point) for point in splits.train]
    validation_features = [tokenizer.encode(point) for point in splits.validation]
    test_features = [tokenizer.encode(point) for point in splits.test]
    train_tensors = _move_tensors(_features_to_tensors(train_features, torch), device)
    validation_tensors = _move_tensors(_features_to_tensors(validation_features, torch), device)
    test_tensors = _move_tensors(_features_to_tensors(test_features, torch), device)

    model = build_tiny_model(
        vocab_size=tokenizer.vocab_size,
        action_size=tokenizer.action_size,
        opponent_card_size=tokenizer.opponent_card_size,
        hidden_size=config.hidden_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_log_path = output_dir / "train_log.jsonl"

    losses: list[float] = []
    with train_log_path.open("w", encoding="utf-8") as log:
        for epoch in range(config.epochs):
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            permutation = torch.randperm(len(splits.train), device=device)
            shuffled = {key: value[permutation] for key, value in train_tensors.items()}
            for batch_slice in _batch_indices(len(splits.train), config.batch_size):
                batch = _slice_tensors(shuffled, batch_slice)
                optimizer.zero_grad()
                outputs = model(batch["input_ids"])
                loss = _loss(outputs, batch, torch)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                batch_count += 1
            loss_value = epoch_loss / batch_count
            losses.append(loss_value)
            log.write(json.dumps({"epoch": epoch, "loss": loss_value}, sort_keys=True) + "\n")

    model.eval()
    with torch.no_grad():
        validation_outputs = model(validation_tensors["input_ids"])
        test_outputs = model(test_tensors["input_ids"])
        value_mse = float(torch.nn.functional.mse_loss(test_outputs["value"], test_tensors["value"]).item())
        policy_accuracy = _accuracy(test_outputs["policy_logits"], test_tensors["policy"], torch)
        opponent_card_accuracy = _accuracy(
            test_outputs["opponent_card_logits"],
            test_tensors["opponent_card"],
            torch,
        )
        validation_loss = float(_loss(validation_outputs, validation_tensors, torch).item())

    if config.save_model:
        torch.save(model.state_dict(), output_dir / "model.pt")
    record = {
        "version": config.run_version,
        "data_source": config.data_source,
        "seed": config.seed,
        "data_seed": config.data_seed,
        "split_seed": config.split_seed,
        "hand_count": validation_report.hand_count,
        "decision_point_count": validation_report.decision_point_count,
        "illegal_action_count": validation_report.illegal_action_count,
        "leakage_count": validation_report.leakage_count,
        "train_points": len(splits.train),
        "validation_points": len(splits.validation),
        "test_points": len(splits.test),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "validation_loss": validation_loss,
        "device": device,
        "batch_size": config.batch_size,
        "policy_accuracy": policy_accuracy,
        "value_mse": value_mse,
        "opponent_card_accuracy": opponent_card_accuracy,
        "baseline_accuracy": baseline.accuracy,
        "baseline_cross_entropy": baseline.cross_entropy,
    }
    checksum = checksum_payload(record)
    write_json_artifact(output_dir / "metrics.json", {**record, "checksum": checksum})

    return LeducTinyTrainResult(
        train_points=len(splits.train),
        validation_points=len(splits.validation),
        test_points=len(splits.test),
        first_loss=losses[0],
        last_loss=losses[-1],
        policy_accuracy=policy_accuracy,
        value_mse=value_mse,
        opponent_card_accuracy=opponent_card_accuracy,
        baseline_accuracy=baseline.accuracy,
        baseline_cross_entropy=baseline.cross_entropy,
        checksum=checksum,
        artifact_dir=output_dir,
        device=device,
    )
