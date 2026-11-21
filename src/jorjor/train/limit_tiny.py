from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.config import load_limit_tiny_train_config
from jorjor.data.split import SplitFractions, apply_split, build_split_manifest
from jorjor.data.validation import validate_decision_points
from jorjor.envs.limit_holdem import LimitHoldemRandomGenerator
from jorjor.limit_abstract_cfr_teacher import LimitAbstractCfrPolicy, train_limit_abstract_cfr
from jorjor.limit_random import flatten_decision_points
from jorjor.limit_teacher import generate_teacher_points
from jorjor.models.tiny import build_tiny_model, require_torch
from jorjor.tokenizer import LimitTokenizer
from jorjor.train.leduc_tiny import _accuracy, _batch_indices, _features_to_tensors, _loss, _move_tensors, _resolve_device


@dataclass(frozen=True)
class LimitTinyTrainResult:
    """Summary for a Limit tiny training smoke run."""

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


def run_limit_tiny_train(config_path: Path, artifact_dir: Path | None = None) -> LimitTinyTrainResult:
    """Run a tiny Limit Hold'em training smoke experiment."""

    torch, _ = require_torch()
    config = load_limit_tiny_train_config(config_path)
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
                batch = {key: value[batch_slice] for key, value in shuffled.items()}
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
        opponent_hand_accuracy = _accuracy(
            test_outputs["opponent_card_logits"],
            test_tensors["opponent_card"],
            torch,
        )
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
        "opponent_hand_accuracy": opponent_hand_accuracy,
        "opponent_hand_top5_accuracy": opponent_hand_top5_accuracy,
        "opponent_hand_top10_accuracy": opponent_hand_top10_accuracy,
        "opponent_hand_classes": tokenizer.opponent_card_size,
    }
    checksum = checksum_payload(record)
    write_json_artifact(output_dir / "metrics.json", {**record, "checksum": checksum})

    return LimitTinyTrainResult(
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


def _load_points(config):
    """Load Limit decision points for the configured data source."""

    if config.data_source == "random":
        generator = LimitHoldemRandomGenerator(seed=config.data_seed)
        return flatten_decision_points(generator.generate_hands(config.hand_count))
    if config.data_source == "teacher":
        return generate_teacher_points(hand_count=config.hand_count, seed=config.data_seed)
    if config.data_source == "abstract_cfr_teacher":
        policy_path = getattr(config, "abstract_cfr_policy_path", None)
        if policy_path is None:
            cfr_result = train_limit_abstract_cfr(
                iteration_count=config.abstract_cfr_iteration_count,
                deal_sample_count=config.abstract_cfr_deal_sample_count,
                seed=config.data_seed,
            )
            policy_table = cfr_result.policy_table
        else:
            policy_record = json.loads(policy_path.read_text(encoding="utf-8"))
            policy_table = policy_record["policy_table"]
        policy = LimitAbstractCfrPolicy(policy_table, seed=config.data_seed + 97)
        return policy.generate_points(hand_count=config.hand_count, seed=config.data_seed + 193)
    raise ValueError(f"unknown data source: {config.data_source}")


def _topk_accuracy(logits, labels, torch, k: int) -> float:
    """Return top-k accuracy for a batch of logits."""

    topk = torch.topk(logits, k=min(k, logits.shape[1]), dim=1).indices
    matches = topk == labels.unsqueeze(1)
    return float(matches.any(dim=1).float().mean().item())
