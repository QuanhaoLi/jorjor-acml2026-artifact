from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
import random
from typing import Any

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.data.schema import DecisionPoint
from jorjor.models.ar import build_tiny_ar_model
from jorjor.models.tiny import require_torch
from jorjor.teacher_artifact import FIXED_LIMIT_ACTION_SPACE, validate_teacher_artifact_rows
from jorjor.tokenizer import LimitTokenizer
from jorjor.train.leduc_tiny import _batch_indices, _resolve_device


@dataclass(frozen=True)
class LimitTeacherDistillResult:
    """Result handle for a k.0.5 teacher-distillation smoke run."""

    report: dict[str, Any]
    artifact_dir: Path
    device: str


def run_limit_teacher_distill_smoke(
    *,
    artifact_rows_path: Path,
    output_dir: Path,
    epochs: int = 8,
    batch_size: int = 32,
    hidden_size: int = 48,
    num_heads: int = 4,
    num_layers: int = 4,
    learning_rate: float = 0.003,
    value_loss_weight: float = 0.3,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    split_seed: int = 20260603,
    seed: int = 20260603,
    device: str = "cpu",
    save_model: bool = True,
) -> LimitTeacherDistillResult:
    """Train a Limit AR student from k.0.4 teacher action distributions."""

    _validate_train_args(
        epochs=epochs,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        learning_rate=learning_rate,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    torch, _ = require_torch()
    torch.manual_seed(seed)
    resolved_device = _resolve_device(device, torch)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(artifact_rows_path)
    artifact_validation = validate_teacher_artifact_rows(rows)
    if not artifact_validation["passed"]:
        raise ValueError(f"teacher artifact validation failed: {artifact_validation}")

    splits = _split_rows(
        rows,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
    )
    tokenizer = LimitTokenizer()
    train_tensors = _move_tensors(_rows_to_tensors(splits["train"], tokenizer, torch), resolved_device)
    validation_tensors = _move_tensors(_rows_to_tensors(splits["validation"], tokenizer, torch), resolved_device)
    test_tensors = _move_tensors(_rows_to_tensors(splits["test"], tokenizer, torch), resolved_device)

    model = build_tiny_ar_model(
        vocab_size=tokenizer.vocab_size,
        action_size=tokenizer.action_size,
        opponent_card_size=tokenizer.opponent_card_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
    ).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses: list[float] = []
    train_log_path = output_dir / "train_log.jsonl"
    with train_log_path.open("w", encoding="utf-8") as log:
        for epoch in range(epochs):
            model.train()
            permutation = torch.randperm(train_tensors["input_ids"].shape[0], device=resolved_device)
            shuffled = {key: value[permutation] for key, value in train_tensors.items()}
            epoch_loss = 0.0
            batch_count = 0
            for batch_slice in _batch_indices(shuffled["input_ids"].shape[0], batch_size):
                batch = {key: value[batch_slice] for key, value in shuffled.items()}
                optimizer.zero_grad()
                outputs = model(batch["input_ids"])
                loss, loss_parts = _distill_loss(outputs, batch, torch, value_loss_weight)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().item())
                batch_count += 1
            loss_value = epoch_loss / max(batch_count, 1)
            if not isfinite(loss_value):
                raise RuntimeError("non-finite training loss")
            losses.append(loss_value)
            log.write(json.dumps({"epoch": epoch, "loss": loss_value, **loss_parts}, sort_keys=True) + "\n")

    model.eval()
    with torch.no_grad():
        validation_outputs = model(validation_tensors["input_ids"])
        test_outputs = model(test_tensors["input_ids"])
        validation_loss_tensor, validation_parts = _distill_loss(
            validation_outputs,
            validation_tensors,
            torch,
            value_loss_weight,
        )
        test_metrics = _distill_metrics(test_outputs, test_tensors, torch)

    if save_model:
        torch.save(model.state_dict(), output_dir / "model.pt")

    record = {
        "version": "k.0.5",
        "report_type": "limit_teacher_distillation_smoke",
        "claim_boundary": "student_distillation_smoke_only; does_not_prove_belief_tracking",
        "input_artifact_rows_path": str(artifact_rows_path),
        "decision_point_count": len(rows),
        "train_points": len(splits["train"]),
        "validation_points": len(splits["validation"]),
        "test_points": len(splits["test"]),
        "seed": seed,
        "split_seed": split_seed,
        "device": resolved_device,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "value_loss_weight": value_loss_weight,
        "policy_target": "teacher_action_distribution",
        "value_target": "teacher_value_rollout_return",
        "opponent_hand_supervised": False,
        "probe_only_hidden_labels_used_for_training": False,
        "tokenizer_hidden_label_policy": "dummy_valid_labels_for_unused_tokenizer_fields",
        "teacher_artifact_validation": artifact_validation,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "validation_loss": float(validation_loss_tensor.item()),
        "validation_teacher_distribution_cross_entropy": validation_parts["teacher_distribution_cross_entropy"],
        "validation_value_mse": validation_parts["value_mse"],
        **test_metrics,
        "model_path": str(output_dir / "model.pt") if save_model else None,
        "train_log_path": str(train_log_path),
    }
    checksum = checksum_payload(record)
    report = {**record, "checksum": checksum}
    write_json_artifact(output_dir / "metrics.json", report)
    return LimitTeacherDistillResult(report=report, artifact_dir=output_dir, device=resolved_device)


def _validate_train_args(
    *,
    epochs: int,
    batch_size: int,
    hidden_size: int,
    num_heads: int,
    num_layers: int,
    learning_rate: float,
    train_fraction: float,
    validation_fraction: float,
) -> None:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if hidden_size <= 0 or hidden_size % num_heads != 0:
        raise ValueError("hidden_size must be positive and divisible by num_heads")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1")
    if not (0.0 <= validation_fraction < 1.0):
        raise ValueError("validation_fraction must be in [0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must leave a test split")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 3:
        raise ValueError("teacher artifact must contain at least 3 rows")
    return rows


def _split_rows(
    rows: list[dict[str, Any]],
    *,
    train_fraction: float,
    validation_fraction: float,
    split_seed: int,
) -> dict[str, list[dict[str, Any]]]:
    indices = list(range(len(rows)))
    random.Random(split_seed).shuffle(indices)
    train_count = max(1, int(len(rows) * train_fraction))
    validation_count = max(1, int(len(rows) * validation_fraction))
    if train_count + validation_count >= len(rows):
        validation_count = max(1, len(rows) - train_count - 1)
    test_count = len(rows) - train_count - validation_count
    if test_count <= 0:
        train_count = max(1, len(rows) - validation_count - 1)
        test_count = len(rows) - train_count - validation_count
    if test_count <= 0:
        raise ValueError("could not create non-empty test split")
    train_indices = indices[:train_count]
    validation_indices = indices[train_count : train_count + validation_count]
    test_indices = indices[train_count + validation_count :]
    return {
        "train": [rows[index] for index in train_indices],
        "validation": [rows[index] for index in validation_indices],
        "test": [rows[index] for index in test_indices],
    }


def _rows_to_tensors(rows: list[dict[str, Any]], tokenizer: LimitTokenizer, torch):
    action_order = tuple(action for action, _ in sorted(tokenizer.action_to_id.items(), key=lambda item: item[1]))
    if action_order != FIXED_LIMIT_ACTION_SPACE:
        raise ValueError(f"teacher action order {FIXED_LIMIT_ACTION_SPACE} does not match tokenizer order {action_order}")
    features = [tokenizer.encode(_row_to_decision_point(row)) for row in rows]
    max_len = max(len(feature.input_ids) for feature in features)
    input_ids = []
    sampled_action = []
    teacher_distribution = []
    teacher_value = []
    for row, feature in zip(rows, features, strict=True):
        input_ids.append(list(feature.input_ids) + [-1] * (max_len - len(feature.input_ids)))
        sampled_action.append(tokenizer.action_to_id[row["sampled_action"]])
        teacher_distribution.append(
            [float(row["teacher_action_distribution"][action]) for action in action_order]
        )
        teacher_value.append(float(row["teacher_value"] or 0.0))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "sampled_action": torch.tensor(sampled_action, dtype=torch.long),
        "teacher_distribution": torch.tensor(teacher_distribution, dtype=torch.float32),
        "teacher_value": torch.tensor(teacher_value, dtype=torch.float32),
    }


def _row_to_decision_point(row: dict[str, Any]) -> DecisionPoint:
    # The tokenizer interface expects hidden labels, but k.0.5 is an action/value-only
    # distillation smoke. Use dummy valid labels so this training path never reads
    # probe-only opponent labels from the teacher artifact.
    return DecisionPoint(
        hand_id=str(row["hand_id"]),
        step_index=int(row["step_index"]),
        acting_player=int(row["player_position"]),
        visible_state=row["student_input"],
        legal_actions=tuple(row["legal_actions"]),
        action=row["sampled_action"],
        opponent_private_card="As Ah",
        opponent_rank="AA",
        terminal_return=row["teacher_value"],
    )


def _move_tensors(tensors, device: str):
    return {key: value.to(device) for key, value in tensors.items()}


def _distill_loss(outputs, tensors, torch, value_loss_weight: float):
    log_probs = torch.nn.functional.log_softmax(outputs["policy_logits"], dim=1)
    teacher_distribution = tensors["teacher_distribution"]
    policy_cross_entropy = -(teacher_distribution * log_probs).sum(dim=1).mean()
    value_mse = torch.nn.functional.mse_loss(outputs["value"], tensors["teacher_value"])
    loss = policy_cross_entropy + value_loss_weight * value_mse
    return loss, {
        "teacher_distribution_cross_entropy": float(policy_cross_entropy.detach().item()),
        "value_mse": float(value_mse.detach().item()),
    }


def _distill_metrics(outputs, tensors, torch) -> dict[str, float]:
    loss, parts = _distill_loss(outputs, tensors, torch, value_loss_weight=0.0)
    del loss
    probabilities = torch.nn.functional.softmax(outputs["policy_logits"], dim=1)
    teacher_distribution = tensors["teacher_distribution"]
    teacher_entropy = -(
        teacher_distribution * torch.log(teacher_distribution.clamp_min(1.0e-12))
    ).sum(dim=1).mean()
    predictions = torch.argmax(probabilities, dim=1)
    teacher_argmax = torch.argmax(teacher_distribution, dim=1)
    sampled_accuracy = (predictions == tensors["sampled_action"]).float().mean()
    teacher_argmax_accuracy = (predictions == teacher_argmax).float().mean()
    return {
        "test_teacher_distribution_cross_entropy": parts["teacher_distribution_cross_entropy"],
        "test_teacher_distribution_kl": float(
            max(0.0, parts["teacher_distribution_cross_entropy"] - float(teacher_entropy.item()))
        ),
        "test_sampled_action_accuracy": float(sampled_accuracy.item()),
        "test_teacher_argmax_accuracy": float(teacher_argmax_accuracy.item()),
        "test_value_mse": float(torch.nn.functional.mse_loss(outputs["value"], tensors["teacher_value"]).item()),
    }
