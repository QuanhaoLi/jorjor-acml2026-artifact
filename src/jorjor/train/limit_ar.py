from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.config import load_limit_ar_train_config
from jorjor.data.split import SplitFractions, apply_split, build_split_manifest
from jorjor.data.validation import validate_decision_points
from jorjor.models.ar import build_tiny_ar_model
from jorjor.models.tiny import require_torch
from jorjor.tokenizer import LimitTokenizer
from jorjor.train.limit_controls import controlled_limit_points, controlled_range_labels
from jorjor.train.leduc_tiny import _accuracy, _batch_indices, _features_to_tensors, _move_tensors, _resolve_device
from jorjor.train.limit_tiny import _load_points, _topk_accuracy


@dataclass(frozen=True)
class LimitArTrainResult:
    """Summary for a Limit tiny AR smoke run."""

    train_points: int
    validation_points: int
    test_points: int
    first_loss: float
    last_loss: float
    policy_accuracy: float
    value_mse: float
    opponent_hand_accuracy: float
    opponent_hand_top5_accuracy: float
    opponent_hand_top10_accuracy: float
    checksum: str
    artifact_dir: Path
    device: str


def _limit_ar_loss(outputs, tensors, config, torch):
    """Compute the weighted Limit AR objective from explicit config weights."""

    ce = torch.nn.functional.cross_entropy
    mse = torch.nn.functional.mse_loss
    policy_loss = ce(outputs["policy_logits"], tensors["policy"])
    value_loss = mse(outputs["value"], tensors["value"])
    opponent_hand_loss = ce(outputs["opponent_card_logits"], tensors["opponent_card"])
    total_loss = (
        config.policy_loss_weight * policy_loss
        + config.value_loss_weight * value_loss
        + config.opponent_hand_loss_weight * opponent_hand_loss
    )
    return total_loss, {
        "policy_loss": float(policy_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "opponent_hand_loss": float(opponent_hand_loss.detach().item()),
    }


def run_limit_ar_train(config_path: Path, artifact_dir: Path | None = None) -> LimitArTrainResult:
    """Run a tiny AR Limit Hold'em training smoke experiment."""

    torch, _ = require_torch()
    config = load_limit_ar_train_config(config_path)
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

    tokenizer = LimitTokenizer()
    train_points = controlled_range_labels(
        controlled_limit_points(splits.train, config.input_control),
        config.range_label_control,
        config.range_label_seed,
    )
    validation_points = controlled_range_labels(
        controlled_limit_points(splits.validation, config.input_control),
        config.range_label_control,
        config.range_label_seed + 1,
    )
    test_points = controlled_range_labels(
        controlled_limit_points(splits.test, config.input_control),
        config.range_label_control,
        config.range_label_seed + 2,
    )
    train_features = [tokenizer.encode(point) for point in train_points]
    validation_features = [tokenizer.encode(point) for point in validation_points]
    test_features = [tokenizer.encode(point) for point in test_points]
    train_tensors = _move_tensors(_features_to_tensors(train_features, torch), device)
    validation_tensors = _move_tensors(_features_to_tensors(validation_features, torch), device)
    test_tensors = _move_tensors(_features_to_tensors(test_features, torch), device)

    model = build_tiny_ar_model(
        vocab_size=tokenizer.vocab_size,
        action_size=tokenizer.action_size,
        opponent_card_size=tokenizer.opponent_card_size,
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
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
                batch = {key: value[batch_slice] for key, value in shuffled.items()}
                optimizer.zero_grad()
                outputs = model(batch["input_ids"])
                loss, _ = _limit_ar_loss(outputs, batch, config, torch)
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
        opponent_hand_accuracy = _accuracy(test_outputs["opponent_card_logits"], test_tensors["opponent_card"], torch)
        opponent_hand_top5_accuracy = _topk_accuracy(
            test_outputs["opponent_card_logits"],
            test_tensors["opponent_card"],
            torch,
            k=5,
        )
        opponent_hand_top10_accuracy = _topk_accuracy(
            test_outputs["opponent_card_logits"],
            test_tensors["opponent_card"],
            torch,
            k=10,
        )
        validation_loss_tensor, validation_head_losses = _limit_ar_loss(validation_outputs, validation_tensors, config, torch)
        validation_loss = float(validation_loss_tensor.item())

    if config.save_model:
        torch.save(model.state_dict(), output_dir / "model.pt")

    record = {
        "version": config.run_version,
        "model": "tiny_ar",
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
        "hidden_size": config.hidden_size,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "input_control": config.input_control,
        "range_label_control": config.range_label_control,
        "range_label_seed": config.range_label_seed,
        "policy_loss_weight": config.policy_loss_weight,
        "value_loss_weight": config.value_loss_weight,
        "opponent_hand_loss_weight": config.opponent_hand_loss_weight,
        "validation_policy_loss": validation_head_losses["policy_loss"],
        "validation_value_loss": validation_head_losses["value_loss"],
        "validation_opponent_hand_loss": validation_head_losses["opponent_hand_loss"],
        "opponent_hand_supervised": config.opponent_hand_loss_weight > 0,
        "abstract_cfr_iteration_count": config.abstract_cfr_iteration_count,
        "abstract_cfr_deal_sample_count": config.abstract_cfr_deal_sample_count,
        "abstract_cfr_policy_path": str(config.abstract_cfr_policy_path) if config.abstract_cfr_policy_path else None,
        "policy_accuracy": policy_accuracy,
        "value_mse": value_mse,
        "opponent_hand_accuracy": opponent_hand_accuracy,
        "opponent_hand_top5_accuracy": opponent_hand_top5_accuracy,
        "opponent_hand_top10_accuracy": opponent_hand_top10_accuracy,
        "opponent_hand_classes": tokenizer.opponent_card_size,
    }
    checksum = checksum_payload(record)
    write_json_artifact(output_dir / "metrics.json", {**record, "checksum": checksum})

    return LimitArTrainResult(
        train_points=len(splits.train),
        validation_points=len(splits.validation),
        test_points=len(splits.test),
        first_loss=losses[0],
        last_loss=losses[-1],
        policy_accuracy=policy_accuracy,
        value_mse=value_mse,
        opponent_hand_accuracy=opponent_hand_accuracy,
        opponent_hand_top5_accuracy=opponent_hand_top5_accuracy,
        opponent_hand_top10_accuracy=opponent_hand_top10_accuracy,
        checksum=checksum,
        artifact_dir=output_dir,
        device=device,
    )
