from __future__ import annotations

from collections import Counter, defaultdict
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
from jorjor.tokenizer import ACTIONS, LIMIT_DECK, LIMIT_ROUNDS, LimitTokenizer
from jorjor.train.leduc_tiny import _features_to_tensors, _move_tensors, _resolve_device
from jorjor.train.limit_controls import controlled_limit_points, controlled_range_labels
from jorjor.train.limit_tiny import _load_points


REPORT_TYPE = "k.5.1_poker_predictive_link"
BOUNDEDNESS_REPORT_TYPE = "k.5.2_composition_boundedness"
BEHAVIOR_LINK_REPORT_TYPE = "k.5.5_representation_behavior_link"
PREDICTIVE_SUPPORTED = "predictive_supported"
PREDICTIVE_VISIBLE_ONLY = "predictive_visible_only"
COMPOSITION_BOUNDED_SUPPORTED = "composition_bounded_supported"
ORDERED_PROXY_COMPONENT_SUPPORTED = "ordered_proxy_component_supported"
HIDDEN_BEHAVIOR_LINK_SUPPORTED = "hidden_behavior_link_supported"
COMPOSITION_BOUNDED_BEHAVIOR_LINK = "composition_bounded_behavior_link"
BEHAVIOR_LINK_NOT_SUPPORTED = "behavior_link_not_supported"
INCONCLUSIVE = "inconclusive"
DEFAULT_SEEDS = ("seed_k", "seed_l", "seed_m")


def build_composition_boundedness_report(
    *,
    matched_report_path: Path,
    composition_report_path: Path,
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
    max_pairs_per_bucket: int = 64,
) -> dict[str, Any]:
    """Build k.5.2 composition-boundedness and range-label proxy report."""

    if not seed_names:
        raise ValueError("seed_names must not be empty")
    if max_pairs_per_bucket <= 0:
        raise ValueError("max_pairs_per_bucket must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    matched_report = _read_required_json(matched_report_path)
    composition_report = _read_required_json(composition_report_path)
    composition_rows = _composition_ladder_rows(composition_report)
    seed_rows = []
    for index, seed_name in enumerate(seed_names):
        train_config_path = train_config_dir / f"limit_ar_action_value_only_{seed_name}_50k_4layer.json"
        seed_dir = archive_root / "outputs" / f"limit_ar_action_value_only_{seed_name}_50k_4layer"
        seed_rows.append(
            _boundedness_seed_row(
                seed_name=seed_name,
                train_config_path=train_config_path,
                checkpoint_path=seed_dir / "model.pt",
                matched_report=matched_report,
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed + index,
                confidence_level=confidence_level,
                device=device,
                batch_size=batch_size,
                max_test_points=max_test_points,
                max_pairs_per_bucket=max_pairs_per_bucket,
            )
        )
    aggregate = _boundedness_aggregate(
        composition_rows=composition_rows,
        seed_rows=seed_rows,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed + 10000,
        confidence_level=confidence_level,
    )
    payload = {
        "version": "k.5.2",
        "report_type": BOUNDEDNESS_REPORT_TYPE,
        "claim_scope": "poker_composition_boundedness_with_range_label_proxy_not_exact_oracle_posterior",
        "matched_report_path": str(matched_report_path),
        "matched_report_checksum": _sha256_file(matched_report_path),
        "composition_report_path": str(composition_report_path),
        "composition_report_checksum": _sha256_file(composition_report_path),
        "archive_root": str(archive_root),
        "train_config_dir": str(train_config_dir),
        "seed_names": list(seed_names),
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "device": device,
        "batch_size": batch_size,
        "max_test_points": max_test_points,
        "max_pairs_per_bucket": max_pairs_per_bucket,
        "composition_ladder_rows": composition_rows,
        "seed_rows": seed_rows,
        "aggregate": aggregate,
        "final_diagnostic_label": _boundedness_label(composition_rows, seed_rows, aggregate),
        "allowed_paper_claim": (
            "In the no-range poker AR audit, weaker public-history proxy controls are not decisive, while the "
            "locked matched-composition gate blocks the residual range-probe claim; range-label proxy pair "
            "analysis provides boundedness evidence but is not an exact oracle-posterior test."
        ),
        "forbidden_paper_claim": "This report proves the absence of all ordered opponent-belief tracking.",
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "k.5.2_composition_boundedness_report.json", payload)
    (output_dir / "k.5.2_composition_boundedness_report.md").write_text(
        _boundedness_markdown(payload),
        encoding="utf-8",
    )
    write_json_artifact(output_dir / "k.5.2_composition_boundedness_manifest.json", _boundedness_manifest(output_dir, payload))
    return payload


def build_poker_predictive_link_report(
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
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Build k.5.1 held-out behavior-head predictive-link report."""

    if not seed_names:
        raise ValueError("seed_names must not be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, seed_name in enumerate(seed_names):
        train_config_path = train_config_dir / f"limit_ar_action_value_only_{seed_name}_50k_4layer.json"
        seed_dir = archive_root / "outputs" / f"limit_ar_action_value_only_{seed_name}_50k_4layer"
        rows.append(
            _seed_row_with_cache(
                cache_dir=cache_dir,
                seed_name=seed_name,
                train_config_path=train_config_path,
                checkpoint_path=seed_dir / "model.pt",
                metrics_path=seed_dir / "metrics.json",
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed + index,
                confidence_level=confidence_level,
                device=device,
                batch_size=batch_size,
                max_test_points=max_test_points,
            )
        )

    aggregate = _aggregate_rows(
        rows,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed + 10000,
        confidence_level=confidence_level,
    )
    payload = {
        "version": "k.5.1",
        "report_type": REPORT_TYPE,
        "claim_scope": "poker_behavior_predictive_link_not_causal_not_belief_upgrade",
        "archive_root": str(archive_root),
        "seed_names": list(seed_names),
        "train_config_dir": str(train_config_dir),
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "device": device,
        "batch_size": batch_size,
        "max_test_points": max_test_points,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "rows": rows,
        "aggregate": aggregate,
        "final_diagnostic_label": _aggregate_label(rows, aggregate),
        "allowed_paper_claim": (
            "The no-range poker AR behavior head predicts held-out action/value targets; this supports the "
            "predictive part of CBPS only when paired with boundedness and matched-composition failures."
        ),
        "forbidden_paper_claim": "This held-out prediction audit proves the range-correlated component causally drives policy.",
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "k.5.1_poker_predictive_link_report.json", payload)
    (output_dir / "k.5.1_poker_predictive_link_report.md").write_text(_markdown(payload), encoding="utf-8")
    write_json_artifact(output_dir / "k.5.1_poker_predictive_link_manifest.json", _manifest(output_dir, payload))
    return payload


def build_representation_behavior_link_report(
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
    probe_epochs: int = 20,
    probe_batch_size: int = 512,
    learning_rate: float = 0.01,
) -> dict[str, Any]:
    """Build k.5.5 representation-to-behavior link report."""

    if not seed_names:
        raise ValueError("seed_names must not be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, seed_name in enumerate(seed_names):
        train_config_path = train_config_dir / f"limit_ar_action_value_only_{seed_name}_50k_4layer.json"
        seed_dir = archive_root / "outputs" / f"limit_ar_action_value_only_{seed_name}_50k_4layer"
        rows.append(
            _behavior_link_seed_row(
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
                probe_epochs=probe_epochs,
                probe_batch_size=probe_batch_size,
                learning_rate=learning_rate,
            )
        )
    aggregate = _behavior_link_aggregate(
        rows,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed + 10000,
        confidence_level=confidence_level,
    )
    payload = {
        "version": "k.5.5",
        "report_type": BEHAVIOR_LINK_REPORT_TYPE,
        "claim_scope": "poker_representation_to_behavior_link_not_causal_not_ordered_belief",
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
        "probe_epochs": probe_epochs,
        "probe_batch_size": probe_batch_size,
        "learning_rate": learning_rate,
        "input_only_composition_features": [
            "history_action_counts",
            "history_action_frequencies",
            "first_history_action",
            "last_history_action",
            "round",
            "acting_player",
            "legal_actions",
            "public_cards",
        ],
        "target_leakage_guard": "composition baselines exclude current action label and terminal value target",
        "rows": rows,
        "aggregate": aggregate,
        "final_diagnostic_label": _behavior_link_aggregate_label(rows, aggregate),
        "allowed_paper_claim": (
            "Frozen no-range poker AR hidden states are evaluated for action/value predictiveness against "
            "input-only public-composition baselines; this can support predictive CBPS language but cannot "
            "establish causal mediation or ordered belief tracking."
        ),
        "forbidden_paper_claim": "This report proves that range-correlated hidden dimensions causally drive policy.",
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "k.5.5_representation_behavior_link_report.json", payload)
    (output_dir / "k.5.5_representation_behavior_link_report.md").write_text(
        _behavior_link_markdown(payload),
        encoding="utf-8",
    )
    write_json_artifact(
        output_dir / "k.5.5_representation_behavior_link_manifest.json",
        _behavior_link_manifest(output_dir, payload),
    )
    return payload


def _seed_row_with_cache(
    *,
    cache_dir: Path | None,
    seed_name: str,
    train_config_path: Path,
    checkpoint_path: Path,
    metrics_path: Path,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
    device: str,
    batch_size: int,
    max_test_points: int | None,
) -> dict[str, Any]:
    cache_key = _seed_cache_key(
        seed_name=seed_name,
        train_config_path=train_config_path,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
        device=device,
        batch_size=batch_size,
        max_test_points=max_test_points,
    )
    cache_path = cache_dir / f"k.5.1_{seed_name}_{cache_key}.json" if cache_dir is not None else None
    if cache_path is not None and cache_path.exists():
        row = json.loads(cache_path.read_text(encoding="utf-8"))
        row["cache_status"] = "hit"
        row["cache_path"] = str(cache_path)
        return row

    row = _seed_row(
        seed_name=seed_name,
        train_config_path=train_config_path,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
        device=device,
        batch_size=batch_size,
        max_test_points=max_test_points,
    )
    row["cache_status"] = "miss" if cache_path is not None else "disabled"
    row["cache_path"] = str(cache_path) if cache_path is not None else None
    if cache_path is not None:
        write_json_artifact(cache_path, row)
    return row


def _seed_cache_key(
    *,
    seed_name: str,
    train_config_path: Path,
    checkpoint_path: Path,
    metrics_path: Path,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
    device: str,
    batch_size: int,
    max_test_points: int | None,
) -> str:
    payload = {
        "schema": "k.5.1_seed_row_cache_v1",
        "seed_name": seed_name,
        "train_config_path": str(train_config_path),
        "train_config_sha256": _sha256_file(train_config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "metrics_path": str(metrics_path),
        "metrics_sha256": _sha256_file(metrics_path),
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "device": device,
        "batch_size": batch_size,
        "max_test_points": max_test_points,
    }
    return checksum_payload(payload)[:16]


def _seed_row(
    *,
    seed_name: str,
    train_config_path: Path,
    checkpoint_path: Path,
    metrics_path: Path,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
    device: str,
    batch_size: int,
    max_test_points: int | None,
) -> dict[str, Any]:
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
    test_points = splits.test[:max_test_points] if max_test_points is not None else splits.test
    tokenizer = LimitTokenizer()
    train_features = [tokenizer.encode(point) for point in splits.train]
    test_features = [tokenizer.encode(point) for point in test_points]
    torch, _ = require_torch()
    resolved_device = _resolve_device(device, torch)
    train_tensors = _move_tensors(_features_to_tensors(train_features, torch), resolved_device)
    test_tensors = _move_tensors(_features_to_tensors(test_features, torch), resolved_device)
    model_metrics = _model_predictions(
        checkpoint_path=checkpoint_path,
        train_config=train_config,
        tokenizer=tokenizer,
        test_tensors=test_tensors,
        torch=torch,
        device=resolved_device,
        batch_size=batch_size,
    )
    baselines = _baseline_predictions(
        train_points=splits.train,
        test_points=test_points,
        train_tensors=train_tensors,
        test_tensors=test_tensors,
        tokenizer=tokenizer,
        torch=torch,
    )
    metrics = _compare_predictions(
        seed_name=seed_name,
        model=model_metrics,
        baselines=baselines,
        test_tensors=test_tensors,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
        torch=torch,
    )
    archived_metrics = _read_optional_json(metrics_path)
    return {
        "seed": seed_name,
        "train_config_path": str(train_config_path),
        "train_config_sha256": _sha256_file(train_config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "metrics_path": str(metrics_path),
        "metrics_sha256": _sha256_file(metrics_path) if metrics_path.exists() else None,
        "archived_policy_accuracy": archived_metrics.get("policy_accuracy") if archived_metrics else None,
        "archived_value_mse": archived_metrics.get("value_mse") if archived_metrics else None,
        "train_points": len(splits.train),
        "test_points": len(test_points),
        "full_test_points": len(splits.test),
        "max_test_points": max_test_points,
        "behavior_metrics": metrics,
        "final_diagnostic_label": _seed_label(metrics),
    }


def _behavior_link_seed_row(
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
    probe_epochs: int,
    probe_batch_size: int,
    learning_rate: float,
) -> dict[str, Any]:
    train_config = load_limit_ar_train_config(train_config_path)
    train_points, test_points = _controlled_train_test_points(train_config, max_train_points, max_test_points)
    tokenizer = LimitTokenizer()
    torch, nn = require_torch()
    resolved_device = _resolve_device(device, torch)
    train_tensors = _move_tensors(_features_to_tensors([tokenizer.encode(point) for point in train_points], torch), resolved_device)
    test_tensors = _move_tensors(_features_to_tensors([tokenizer.encode(point) for point in test_points], torch), resolved_device)
    train_model = _model_predictions(
        checkpoint_path=checkpoint_path,
        train_config=train_config,
        tokenizer=tokenizer,
        test_tensors=train_tensors,
        torch=torch,
        device=resolved_device,
        batch_size=batch_size,
    )
    test_model = _model_predictions(
        checkpoint_path=checkpoint_path,
        train_config=train_config,
        tokenizer=tokenizer,
        test_tensors=test_tensors,
        torch=torch,
        device=resolved_device,
        batch_size=batch_size,
    )
    train_composition = _input_only_composition_features(train_points, train_tensors, torch)
    test_composition = _input_only_composition_features(test_points, test_tensors, torch)
    residual_train, residual_test = _linear_residualize(
        train_model["locked_hidden"],
        test_model["locked_hidden"],
        train_composition,
        test_composition,
        torch,
    )
    baselines = _input_only_baseline_predictions(
        train_points=train_points,
        test_points=test_points,
        train_tensors=train_tensors,
        test_tensors=test_tensors,
        torch=torch,
    )
    hidden_action = _train_behavior_action_probe(
        train_x=train_model["locked_hidden"],
        train_y=train_tensors["policy"],
        test_x=test_model["locked_hidden"],
        epochs=probe_epochs,
        batch_size=probe_batch_size,
        learning_rate=learning_rate,
        seed=bootstrap_seed + 101,
        torch=torch,
        nn=nn,
        device=resolved_device,
    )
    residual_action = _train_behavior_action_probe(
        train_x=residual_train,
        train_y=train_tensors["policy"],
        test_x=residual_test,
        epochs=probe_epochs,
        batch_size=probe_batch_size,
        learning_rate=learning_rate,
        seed=bootstrap_seed + 151,
        torch=torch,
        nn=nn,
        device=resolved_device,
    )
    composition_action = _train_behavior_action_probe(
        train_x=train_composition,
        train_y=train_tensors["policy"],
        test_x=test_composition,
        epochs=probe_epochs,
        batch_size=probe_batch_size,
        learning_rate=learning_rate,
        seed=bootstrap_seed + 181,
        torch=torch,
        nn=nn,
        device=resolved_device,
    )
    hidden_value = _train_behavior_value_probe(
        train_x=train_model["locked_hidden"],
        train_y=train_tensors["value"],
        test_x=test_model["locked_hidden"],
        epochs=probe_epochs,
        batch_size=probe_batch_size,
        learning_rate=learning_rate,
        seed=bootstrap_seed + 201,
        torch=torch,
        nn=nn,
        device=resolved_device,
    )
    residual_value = _train_behavior_value_probe(
        train_x=residual_train,
        train_y=train_tensors["value"],
        test_x=residual_test,
        epochs=probe_epochs,
        batch_size=probe_batch_size,
        learning_rate=learning_rate,
        seed=bootstrap_seed + 251,
        torch=torch,
        nn=nn,
        device=resolved_device,
    )
    composition_value = _train_behavior_value_probe(
        train_x=train_composition,
        train_y=train_tensors["value"],
        test_x=test_composition,
        epochs=probe_epochs,
        batch_size=probe_batch_size,
        learning_rate=learning_rate,
        seed=bootstrap_seed + 281,
        torch=torch,
        nn=nn,
        device=resolved_device,
    )
    metrics = _behavior_link_metrics(
        hidden_action=hidden_action,
        residual_action=residual_action,
        composition_action=composition_action,
        hidden_value=hidden_value,
        residual_value=residual_value,
        composition_value=composition_value,
        baselines=baselines,
        test_tensors=test_tensors,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
        torch=torch,
    )
    return {
        "seed": seed_name,
        "train_config_path": str(train_config_path),
        "train_config_sha256": _sha256_file(train_config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "train_points": len(train_points),
        "test_points": len(test_points),
        "max_train_points": max_train_points,
        "max_test_points": max_test_points,
        "locked_layer": test_model["locked_layer"],
        "probe_epochs": probe_epochs,
        "probe_batch_size": probe_batch_size,
        "learning_rate": learning_rate,
        "metrics": metrics,
        "final_diagnostic_label": _behavior_link_seed_label(metrics),
    }


def _model_predictions(
    *,
    checkpoint_path: Path,
    train_config,
    tokenizer: LimitTokenizer,
    test_tensors,
    torch,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint is missing: {checkpoint_path}")
    model = build_tiny_ar_model(
        vocab_size=tokenizer.vocab_size,
        action_size=tokenizer.action_size,
        opponent_card_size=tokenizer.opponent_card_size,
        hidden_size=train_config.hidden_size,
        num_heads=train_config.num_heads,
        num_layers=train_config.num_layers,
    ).to(device)
    load_info = _load_ar_checkpoint_with_vocab_compat(model, checkpoint_path, torch, device)
    model.eval()
    policy_predictions = []
    policy_probabilities = []
    range_probabilities = []
    value_predictions = []
    locked_hidden = []
    sequence_hidden = []
    locked_layer = "encoder_layer_2"
    locked_layer_used = locked_layer
    with torch.no_grad():
        for start in range(0, test_tensors["input_ids"].shape[0], batch_size):
            stop = min(start + batch_size, test_tensors["input_ids"].shape[0])
            layers = model.encode_layers(test_tensors["input_ids"][start:stop])
            layer_name = locked_layer if locked_layer in layers else "sequence_hidden"
            locked_layer_used = layer_name
            hidden = layers["sequence_hidden"]
            outputs = {
                "policy_logits": model.policy_head(hidden),
                "value": model.value_head(hidden).squeeze(-1),
                "opponent_card_logits": model.opponent_card_head(hidden),
            }
            policy_logits = outputs["policy_logits"].detach()
            policy_probabilities.append(torch.softmax(policy_logits, dim=1))
            policy_predictions.append(torch.argmax(policy_logits, dim=1))
            range_probabilities.append(torch.softmax(outputs["opponent_card_logits"].detach(), dim=1))
            value_predictions.append(outputs["value"].detach())
            locked_hidden.append(layers[layer_name].detach())
            sequence_hidden.append(layers["sequence_hidden"].detach())
    return {
        "policy_predictions": torch.cat(policy_predictions, dim=0),
        "policy_probabilities": torch.cat(policy_probabilities, dim=0),
        "value_predictions": torch.cat(value_predictions, dim=0),
        "range_probabilities": torch.cat(range_probabilities, dim=0),
        "locked_hidden": torch.cat(locked_hidden, dim=0),
        "sequence_hidden": torch.cat(sequence_hidden, dim=0),
        "locked_layer": locked_layer_used,
        "checkpoint_load_info": load_info,
    }


def _controlled_train_test_points(train_config, max_train_points: int | None, max_test_points: int | None):
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
    return train_points, test_points


def _input_only_baseline_predictions(*, train_points, test_points, train_tensors, test_tensors, torch) -> dict[str, dict[str, Any]]:
    return {
        "action_prior": _lookup_baseline(
            train_points=train_points,
            test_points=test_points,
            train_tensors=train_tensors,
            test_tensors=test_tensors,
            key_fn=lambda point: ("global",),
            torch=torch,
        ),
        "C_min_lookup": _lookup_baseline(
            train_points=train_points,
            test_points=test_points,
            train_tensors=train_tensors,
            test_tensors=test_tensors,
            key_fn=_input_only_c_min_key,
            fallback_key_fns=[lambda point: ("global",)],
            torch=torch,
        ),
        "C_mid_lookup": _lookup_baseline(
            train_points=train_points,
            test_points=test_points,
            train_tensors=train_tensors,
            test_tensors=test_tensors,
            key_fn=_input_only_c_mid_key,
            fallback_key_fns=[_input_only_c_min_key, lambda point: ("global",)],
            torch=torch,
        ),
        "C_full_lookup": _lookup_baseline(
            train_points=train_points,
            test_points=test_points,
            train_tensors=train_tensors,
            test_tensors=test_tensors,
            key_fn=_input_only_c_full_key,
            fallback_key_fns=[_input_only_c_mid_key, _input_only_c_min_key, lambda point: ("global",)],
            torch=torch,
        ),
    }


def _input_only_composition_features(points, tensors, torch):
    rows = []
    player_values = ("0", "1", "mask")
    round_values = (*LIMIT_ROUNDS, "mask")
    for point in points:
        state = point.visible_state
        actions = _history_actions(point)
        action_count = max(len(actions), 1)
        first_action = actions[0] if actions else None
        last_action = actions[-1] if actions else None
        row = [
            float(len(actions)),
            min(float(len(actions)), 16.0) / 16.0,
        ]
        row.extend(float(actions.count(action)) for action in ACTIONS)
        row.extend(float(actions.count(action)) / action_count for action in ACTIONS)
        row.extend(1.0 if action in actions else 0.0 for action in ACTIONS)
        row.extend(1.0 if first_action == action else 0.0 for action in ACTIONS)
        row.extend(1.0 if last_action == action else 0.0 for action in ACTIONS)
        acting_player = str(state.get("acting_player", point.acting_player))
        row.extend(1.0 if acting_player == value else 0.0 for value in player_values)
        round_name = str(state.get("round", "mask"))
        row.extend(1.0 if round_name == value else 0.0 for value in round_values)
        legal_actions = set(str(action) for action in state.get("legal_actions", point.legal_actions))
        row.extend(1.0 if action in legal_actions else 0.0 for action in ACTIONS)
        private_cards = set(str(card) for card in state.get("private_cards", ()))
        public_cards = set(str(card) for card in state.get("public_cards", ()))
        row.extend(1.0 if card in private_cards else 0.0 for card in LIMIT_DECK)
        row.extend(1.0 if card in public_cards else 0.0 for card in LIMIT_DECK)
        rows.append(row)
    if not rows:
        return torch.empty((0, 0), dtype=torch.float32, device=tensors["policy"].device)
    return torch.tensor(rows, dtype=torch.float32, device=tensors["policy"].device)


def _linear_residualize(train_x, test_x, train_nuisance, test_nuisance, torch):
    train_design = _with_intercept(train_nuisance, torch)
    test_design = _with_intercept(test_nuisance, torch)
    coefficients = torch.linalg.pinv(train_design).matmul(train_x)
    return train_x - train_design.matmul(coefficients), test_x - test_design.matmul(coefficients)


def _with_intercept(features, torch):
    ones = torch.ones((features.shape[0], 1), dtype=features.dtype, device=features.device)
    return torch.cat([ones, features], dim=1)


def _train_behavior_action_probe(
    *,
    train_x,
    train_y,
    test_x,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    torch,
    nn,
    device: str,
) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("probe_epochs must be positive")
    torch.manual_seed(seed)
    probe = nn.Linear(train_x.shape[1], len(ACTIONS)).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    x = train_x.detach()
    y = train_y.detach()
    losses = []
    for _ in range(epochs):
        permutation = torch.randperm(x.shape[0], device=device)
        epoch_loss = 0.0
        batch_count = 0
        for start in range(0, x.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(probe(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            batch_count += 1
        losses.append(epoch_loss / max(batch_count, 1))
    with torch.no_grad():
        logits = probe(test_x.detach())
        return {
            "policy_predictions": torch.argmax(logits, dim=1),
            "policy_probabilities": torch.softmax(logits, dim=1),
            "first_loss": losses[0],
            "last_loss": losses[-1],
        }


def _train_behavior_value_probe(
    *,
    train_x,
    train_y,
    test_x,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    torch,
    nn,
    device: str,
) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("probe_epochs must be positive")
    torch.manual_seed(seed)
    probe = nn.Linear(train_x.shape[1], 1).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    x = train_x.detach()
    y = train_y.detach()
    losses = []
    for _ in range(epochs):
        permutation = torch.randperm(x.shape[0], device=device)
        epoch_loss = 0.0
        batch_count = 0
        for start in range(0, x.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad()
            prediction = probe(x[indices]).squeeze(-1)
            loss = torch.nn.functional.mse_loss(prediction, y[indices])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            batch_count += 1
        losses.append(epoch_loss / max(batch_count, 1))
    with torch.no_grad():
        return {
            "value_predictions": probe(test_x.detach()).squeeze(-1),
            "first_loss": losses[0],
            "last_loss": losses[-1],
        }


def _behavior_link_metrics(
    *,
    hidden_action: dict[str, Any],
    residual_action: dict[str, Any],
    composition_action: dict[str, Any],
    hidden_value: dict[str, Any],
    residual_value: dict[str, Any],
    composition_value: dict[str, Any],
    baselines: dict[str, dict[str, Any]],
    test_tensors,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
    torch,
) -> dict[str, Any]:
    labels = test_tensors["policy"]
    values = test_tensors["value"]
    probe_rows = {
        "hidden": _probe_behavior_row(
            action_predictions=hidden_action["policy_predictions"],
            value_predictions=hidden_value["value_predictions"],
            labels=labels,
            values=values,
            torch=torch,
            extra={
                "action_first_loss": hidden_action["first_loss"],
                "action_last_loss": hidden_action["last_loss"],
                "value_first_loss": hidden_value["first_loss"],
                "value_last_loss": hidden_value["last_loss"],
            },
        ),
        "residual_hidden_after_C_full_linear": _probe_behavior_row(
            action_predictions=residual_action["policy_predictions"],
            value_predictions=residual_value["value_predictions"],
            labels=labels,
            values=values,
            torch=torch,
            extra={
                "action_first_loss": residual_action["first_loss"],
                "action_last_loss": residual_action["last_loss"],
                "value_first_loss": residual_value["first_loss"],
                "value_last_loss": residual_value["last_loss"],
            },
        ),
        "C_full_linear": _probe_behavior_row(
            action_predictions=composition_action["policy_predictions"],
            value_predictions=composition_value["value_predictions"],
            labels=labels,
            values=values,
            torch=torch,
            extra={
                "action_first_loss": composition_action["first_loss"],
                "action_last_loss": composition_action["last_loss"],
                "value_first_loss": composition_value["first_loss"],
                "value_last_loss": composition_value["last_loss"],
            },
        ),
    }
    baseline_rows = {}
    for name, baseline in baselines.items():
        baseline_rows[name] = _probe_behavior_row(
            action_predictions=baseline["policy_predictions"],
            value_predictions=baseline["value_predictions"],
            labels=labels,
            values=values,
            torch=torch,
            extra={
                "primary_coverage": baseline["primary_coverage"],
                "effective_coverage": baseline["effective_coverage"],
                "fallback_counts": baseline["fallback_counts"],
                "global_fallback_count": baseline["global_fallback_count"],
                "bucket_count": baseline["bucket_count"],
            },
        )
    comparisons = {
            "hidden_vs_action_prior": _behavior_delta_row(
                probe_rows["hidden"],
                baseline_rows["action_prior"],
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed + 1,
                confidence_level=confidence_level,
            ),
            "hidden_vs_C_full_lookup": _behavior_delta_row(
                probe_rows["hidden"],
                baseline_rows["C_full_lookup"],
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed + 2,
                confidence_level=confidence_level,
            ),
            "residual_hidden_vs_C_full_lookup": _behavior_delta_row(
                probe_rows["residual_hidden_after_C_full_linear"],
                baseline_rows["C_full_lookup"],
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed + 3,
                confidence_level=confidence_level,
            ),
            "hidden_vs_C_full_linear": _behavior_delta_row(
                probe_rows["hidden"],
                probe_rows["C_full_linear"],
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed + 4,
                confidence_level=confidence_level,
            ),
            "residual_hidden_vs_C_full_linear": _behavior_delta_row(
                probe_rows["residual_hidden_after_C_full_linear"],
                probe_rows["C_full_linear"],
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed + 5,
                confidence_level=confidence_level,
            ),
        }
    return {
        "probe_rows": {name: _strip_pointwise(row) for name, row in probe_rows.items()},
        "lookup_baseline_rows": {name: _strip_pointwise(row) for name, row in baseline_rows.items()},
        "comparisons": comparisons,
    }


def _probe_behavior_row(*, action_predictions, value_predictions, labels, values, torch, extra: dict[str, Any]) -> dict[str, Any]:
    action_hits = (action_predictions == labels).detach().cpu().float().tolist()
    value_errors = ((value_predictions - values) ** 2).detach().cpu().float().tolist()
    return {
        "action_accuracy": _mean([float(value) for value in action_hits]),
        "value_mse": _mean([float(value) for value in value_errors]),
        "per_point_action_hits": [float(value) for value in action_hits],
        "per_point_value_errors": [float(value) for value in value_errors],
        **extra,
    }


def _behavior_delta_row(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    action_deltas = [
        float(left_hit) - float(right_hit)
        for left_hit, right_hit in zip(left["per_point_action_hits"], right["per_point_action_hits"], strict=True)
    ]
    value_deltas = [
        float(right_error) - float(left_error)
        for left_error, right_error in zip(left["per_point_value_errors"], right["per_point_value_errors"], strict=True)
    ]
    return {
        "action_accuracy_delta": _mean(action_deltas),
        "action_accuracy_delta_ci": _bootstrap_mean_ci(
            action_deltas,
            bootstrap_count=bootstrap_count,
            bootstrap_seed=bootstrap_seed,
            confidence_level=confidence_level,
        ),
        "value_mse_delta_right_minus_left": _mean(value_deltas),
        "value_mse_delta_ci": _bootstrap_mean_ci(
            value_deltas,
            bootstrap_count=bootstrap_count,
            bootstrap_seed=bootstrap_seed + 1000,
            confidence_level=confidence_level,
        ),
    }


def _strip_pointwise(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"per_point_action_hits", "per_point_value_errors"}
    }


def _behavior_link_seed_label(metrics: dict[str, Any]) -> str:
    hidden = metrics["comparisons"]["hidden_vs_C_full_linear"]["action_accuracy_delta_ci"]
    residual = metrics["comparisons"]["residual_hidden_vs_C_full_linear"]["action_accuracy_delta_ci"]
    hidden_value = metrics["comparisons"]["hidden_vs_C_full_linear"]["value_mse_delta_ci"]
    residual_value = metrics["comparisons"]["residual_hidden_vs_C_full_linear"]["value_mse_delta_ci"]
    hidden_positive = hidden["lower_delta"] > 0.0 or hidden_value["lower_delta"] > 0.0
    residual_positive = residual["lower_delta"] > 0.0 or residual_value["lower_delta"] > 0.0
    if hidden_positive and residual_positive:
        return HIDDEN_BEHAVIOR_LINK_SUPPORTED
    if hidden_positive and not residual_positive:
        return COMPOSITION_BOUNDED_BEHAVIOR_LINK
    if hidden["upper_delta"] <= 0.0 and hidden_value["upper_delta"] <= 0.0:
        return BEHAVIOR_LINK_NOT_SUPPORTED
    return INCONCLUSIVE


def _behavior_link_aggregate(
    rows: list[dict[str, Any]],
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    labels = [row["final_diagnostic_label"] for row in rows]
    comparisons = {}
    for comparison_name in sorted(rows[0]["metrics"]["comparisons"]):
        action_values = [
            float(row["metrics"]["comparisons"][comparison_name]["action_accuracy_delta"])
            for row in rows
        ]
        value_values = [
            float(row["metrics"]["comparisons"][comparison_name]["value_mse_delta_right_minus_left"])
            for row in rows
        ]
        comparisons[comparison_name] = {
            "mean_action_accuracy_delta": _mean(action_values),
            "action_accuracy_delta_seed_ci": _bootstrap_mean_ci(
                action_values,
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed + _stable_int(comparison_name) % 1000,
                confidence_level=confidence_level,
            ),
            "mean_value_mse_delta_right_minus_left": _mean(value_values),
            "value_mse_delta_seed_ci": _bootstrap_mean_ci(
                value_values,
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed + 2000 + _stable_int(comparison_name) % 1000,
                confidence_level=confidence_level,
            ),
        }
    return {
        "seed_count": len(rows),
        "label_counts": dict(Counter(labels)),
        "comparisons": comparisons,
    }


def _behavior_link_aggregate_label(rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> str:
    labels = {row["final_diagnostic_label"] for row in rows}
    residual = aggregate["comparisons"]["residual_hidden_vs_C_full_linear"]["action_accuracy_delta_seed_ci"]
    residual_value = aggregate["comparisons"]["residual_hidden_vs_C_full_linear"]["value_mse_delta_seed_ci"]
    hidden = aggregate["comparisons"]["hidden_vs_C_full_linear"]["action_accuracy_delta_seed_ci"]
    hidden_value = aggregate["comparisons"]["hidden_vs_C_full_linear"]["value_mse_delta_seed_ci"]
    residual_positive = residual["lower_delta"] > 0.0 or residual_value["lower_delta"] > 0.0
    hidden_positive = hidden["lower_delta"] > 0.0 or hidden_value["lower_delta"] > 0.0
    if labels <= {HIDDEN_BEHAVIOR_LINK_SUPPORTED, COMPOSITION_BOUNDED_BEHAVIOR_LINK} and residual_positive:
        return HIDDEN_BEHAVIOR_LINK_SUPPORTED
    if labels <= {HIDDEN_BEHAVIOR_LINK_SUPPORTED, COMPOSITION_BOUNDED_BEHAVIOR_LINK, INCONCLUSIVE} and hidden_positive:
        return COMPOSITION_BOUNDED_BEHAVIOR_LINK
    if labels == {BEHAVIOR_LINK_NOT_SUPPORTED}:
        return BEHAVIOR_LINK_NOT_SUPPORTED
    return INCONCLUSIVE


def _input_only_c_min_key(point) -> tuple[Any, ...]:
    actions = _history_actions(point)
    counts = tuple(actions.count(action) for action in ACTIONS)
    last_action = actions[-1] if actions else "none"
    return ("len", len(actions), "counts", counts, "last", last_action)


def _input_only_c_mid_key(point) -> tuple[Any, ...]:
    state = point.visible_state
    return (
        *_input_only_c_min_key(point),
        "acting_player",
        str(state.get("acting_player", point.acting_player)),
        "round",
        str(state.get("round", "mask")),
        "legal",
        tuple(sorted(str(action) for action in state.get("legal_actions", point.legal_actions))),
    )


def _input_only_c_full_key(point) -> tuple[Any, ...]:
    state = point.visible_state
    return (
        *_input_only_c_mid_key(point),
        "private_cards",
        tuple(str(card) for card in state.get("private_cards", ())),
        "public_cards",
        tuple(str(card) for card in state.get("public_cards", ())),
    )


def _baseline_predictions(*, train_points, test_points, train_tensors, test_tensors, tokenizer: LimitTokenizer, torch) -> dict[str, dict[str, Any]]:
    return {
        "action_prior": _lookup_baseline(
            train_points=train_points,
            test_points=test_points,
            train_tensors=train_tensors,
            test_tensors=test_tensors,
            key_fn=lambda point: ("global",),
            torch=torch,
        ),
        "C_min": _lookup_baseline(
            train_points=train_points,
            test_points=test_points,
            train_tensors=train_tensors,
            test_tensors=test_tensors,
            key_fn=_c_min_key,
            fallback_key_fns=[lambda point: ("global",)],
            torch=torch,
        ),
        "C_mid": _lookup_baseline(
            train_points=train_points,
            test_points=test_points,
            train_tensors=train_tensors,
            test_tensors=test_tensors,
            key_fn=lambda point: (*_c_min_key(point), *_round_position_legal_key(point)),
            fallback_key_fns=[_c_min_key, lambda point: ("global",)],
            torch=torch,
        ),
        "C_full": _lookup_baseline(
            train_points=train_points,
            test_points=test_points,
            train_tensors=train_tensors,
            test_tensors=test_tensors,
            key_fn=lambda point: (*_c_min_key(point), *_round_position_legal_key(point), *_public_card_key(point)),
            fallback_key_fns=[lambda point: (*_c_min_key(point), *_round_position_legal_key(point)), _c_min_key, lambda point: ("global",)],
            torch=torch,
        ),
    }


def _lookup_baseline(
    *,
    train_points,
    test_points,
    train_tensors,
    test_tensors,
    key_fn,
    fallback_key_fns: list[Any] | None = None,
    torch,
) -> dict[str, Any]:
    train_policy = [int(value) for value in train_tensors["policy"].detach().cpu().tolist()]
    train_value = [float(value) for value in train_tensors["value"].detach().cpu().tolist()]
    global_action = Counter(train_policy).most_common(1)[0][0]
    global_value = sum(train_value) / len(train_value)
    grouped_actions: dict[tuple[Any, ...], Counter[int]] = defaultdict(Counter)
    grouped_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    all_key_fns = [key_fn, *(fallback_key_fns or [])]
    grouped_by_level: list[tuple[dict[tuple[Any, ...], Counter[int]], dict[tuple[Any, ...], list[float]]]] = []
    for level_key_fn in all_key_fns:
        level_actions: dict[tuple[Any, ...], Counter[int]] = defaultdict(Counter)
        level_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        for point, action, value in zip(train_points, train_policy, train_value, strict=True):
            key = level_key_fn(point)
            level_actions[key][action] += 1
            level_values[key].append(value)
        grouped_by_level.append((level_actions, level_values))
    grouped_actions, grouped_values = grouped_by_level[0]
    action_predictions = []
    value_predictions = []
    covered = 0
    fallback_counts = [0 for _ in all_key_fns]
    global_fallback_count = 0
    for point in test_points:
        matched = False
        for level, level_key_fn in enumerate(all_key_fns):
            level_actions, level_values = grouped_by_level[level]
            key = level_key_fn(point)
            if key not in level_actions:
                continue
            if level == 0:
                covered += 1
            fallback_counts[level] += 1
            action_predictions.append(level_actions[key].most_common(1)[0][0])
            value_predictions.append(sum(level_values[key]) / len(level_values[key]))
            matched = True
            break
        if not matched:
            global_fallback_count += 1
            action_predictions.append(global_action)
            value_predictions.append(global_value)
    return {
        "policy_predictions": torch.tensor(action_predictions, dtype=test_tensors["policy"].dtype, device=test_tensors["policy"].device),
        "value_predictions": torch.tensor(value_predictions, dtype=test_tensors["value"].dtype, device=test_tensors["value"].device),
        "primary_coverage": covered / len(test_points),
        "effective_coverage": (len(test_points) - global_fallback_count) / len(test_points),
        "fallback_counts": fallback_counts,
        "global_fallback_count": global_fallback_count,
        "bucket_count": len(grouped_actions),
    }


def _compare_predictions(
    *,
    seed_name: str,
    model: dict[str, Any],
    baselines: dict[str, dict[str, Any]],
    test_tensors,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
    torch,
) -> dict[str, Any]:
    labels = test_tensors["policy"]
    values = test_tensors["value"]
    model_correct = (model["policy_predictions"] == labels).detach().cpu().float().tolist()
    model_value_sq_error = ((model["value_predictions"] - values) ** 2).detach().cpu().tolist()
    model_ce = torch.nn.functional.cross_entropy(model["policy_probabilities"].clamp_min(1e-12).log(), labels, reduction="none")
    baseline_rows = {}
    for index, (name, baseline) in enumerate(baselines.items()):
        baseline_correct = (baseline["policy_predictions"] == labels).detach().cpu().float().tolist()
        baseline_value_sq_error = ((baseline["value_predictions"] - values) ** 2).detach().cpu().tolist()
        action_deltas = [float(model_hit) - float(base_hit) for model_hit, base_hit in zip(model_correct, baseline_correct, strict=True)]
        value_deltas = [float(base_error) - float(model_error) for model_error, base_error in zip(model_value_sq_error, baseline_value_sq_error, strict=True)]
        baseline_rows[name] = {
            "action_accuracy": _mean(baseline_correct),
            "value_mse": _mean(baseline_value_sq_error),
            "primary_coverage": baseline["primary_coverage"],
            "effective_coverage": baseline["effective_coverage"],
            "fallback_counts": baseline["fallback_counts"],
            "global_fallback_count": baseline["global_fallback_count"],
            "bucket_count": baseline["bucket_count"],
            "action_accuracy_delta_model_minus_baseline": _mean(action_deltas),
            "action_accuracy_delta_ci": _bootstrap_mean_ci(
                action_deltas,
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed + index,
                confidence_level=confidence_level,
            ),
            "value_mse_delta_baseline_minus_model": _mean(value_deltas),
            "value_mse_delta_ci": _bootstrap_mean_ci(
                value_deltas,
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed + 1000 + index,
                confidence_level=confidence_level,
            ),
        }
    return {
        "seed": seed_name,
        "model_action_accuracy": _mean(model_correct),
        "model_action_cross_entropy": float(model_ce.mean().item()),
        "model_value_mse": _mean(model_value_sq_error),
        "checkpoint_load_info": model["checkpoint_load_info"],
        "baseline_rows": baseline_rows,
    }


def _aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    baselines = sorted(rows[0]["behavior_metrics"]["baseline_rows"])
    aggregate = {
        "seed_count": len(rows),
        "model_action_accuracy_mean": _mean([row["behavior_metrics"]["model_action_accuracy"] for row in rows]),
        "model_value_mse_mean": _mean([row["behavior_metrics"]["model_value_mse"] for row in rows]),
        "baseline_rows": {},
    }
    for index, baseline in enumerate(baselines):
        action_deltas = [
            row["behavior_metrics"]["baseline_rows"][baseline]["action_accuracy_delta_model_minus_baseline"]
            for row in rows
        ]
        value_deltas = [
            row["behavior_metrics"]["baseline_rows"][baseline]["value_mse_delta_baseline_minus_model"]
            for row in rows
        ]
        aggregate["baseline_rows"][baseline] = {
            "mean_action_accuracy_delta_model_minus_baseline": _mean(action_deltas),
            "action_accuracy_delta_seed_ci": _bootstrap_mean_ci(
                action_deltas,
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed + index,
                confidence_level=confidence_level,
            ),
            "mean_value_mse_delta_baseline_minus_model": _mean(value_deltas),
            "value_mse_delta_seed_ci": _bootstrap_mean_ci(
                value_deltas,
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed + 1000 + index,
                confidence_level=confidence_level,
            ),
        }
    return aggregate


def _seed_label(metrics: dict[str, Any]) -> str:
    prior = metrics["baseline_rows"]["action_prior"]["action_accuracy_delta_ci"]
    c_min = metrics["baseline_rows"]["C_min"]["action_accuracy_delta_ci"]
    if prior["lower_delta"] > 0.0 and c_min["lower_delta"] > 0.0:
        return PREDICTIVE_SUPPORTED
    if prior["lower_delta"] > 0.0:
        return PREDICTIVE_VISIBLE_ONLY
    return INCONCLUSIVE


def _aggregate_label(rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> str:
    labels = {row["final_diagnostic_label"] for row in rows}
    c_min = aggregate["baseline_rows"]["C_min"]["action_accuracy_delta_seed_ci"]
    if labels == {PREDICTIVE_SUPPORTED} and c_min["lower_delta"] > 0.0:
        return PREDICTIVE_SUPPORTED
    if labels <= {PREDICTIVE_SUPPORTED, PREDICTIVE_VISIBLE_ONLY}:
        return PREDICTIVE_VISIBLE_ONLY
    return INCONCLUSIVE


def _boundedness_seed_row(
    *,
    seed_name: str,
    train_config_path: Path,
    checkpoint_path: Path,
    matched_report: dict[str, Any],
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
    device: str,
    batch_size: int,
    max_test_points: int | None,
    max_pairs_per_bucket: int,
) -> dict[str, Any]:
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
    test_points = splits.test[:max_test_points] if max_test_points is not None else splits.test
    tokenizer = LimitTokenizer()
    test_features = [tokenizer.encode(point) for point in test_points]
    torch, _ = require_torch()
    resolved_device = _resolve_device(device, torch)
    test_tensors = _move_tensors(_features_to_tensors(test_features, torch), resolved_device)
    model_metrics = _model_predictions(
        checkpoint_path=checkpoint_path,
        train_config=train_config,
        tokenizer=tokenizer,
        test_tensors=test_tensors,
        torch=torch,
        device=resolved_device,
        batch_size=batch_size,
    )
    pairs = _range_label_proxy_pairs(
        seed_name=seed_name,
        test_points=test_points,
        test_tensors=test_tensors,
        model_metrics=model_metrics,
        max_pairs_per_bucket=max_pairs_per_bucket,
        seed=bootstrap_seed,
        torch=torch,
    )
    matched_row = _matched_row_for_seed(matched_report, seed_name)
    proxy = _proxy_pair_summary(
        pairs,
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
        "full_test_points": len(splits.test),
        "max_test_points": max_test_points,
        "locked_layer": model_metrics["locked_layer"],
        "matched_gate": {
            "matched_point_count": matched_row["matched_point_count"],
            "matched_bucket_count": matched_row["matched_bucket_count"],
            "residual_top10": matched_row["residual_top10"],
            "action_value_composition_top10": matched_row["action_value_composition_top10"],
            "delta_residual_minus_prior": matched_row["delta_residual_minus_prior"],
            "bucket_bootstrap_lower_delta": matched_row["bucket_bootstrap_lower_delta"],
            "bucket_bootstrap_upper_delta": matched_row["bucket_bootstrap_upper_delta"],
        },
        "range_label_proxy": proxy,
        "final_diagnostic_label": _boundedness_seed_label(matched_row, proxy),
    }


def _composition_ladder_rows(composition_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = composition_report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("composition report is missing rows")
    output = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("composition report row must be an object")
        output.append(
            {
                "variant": str(row["variant"]),
                "definition": str(row["definition"]),
                "source_metric": str(row["source_metric"]),
                "mean_delta": float(row["mean_delta"]),
                "lower_delta": float(row["lower_delta"]),
                "upper_delta": None if row.get("upper_delta") is None else float(row["upper_delta"]),
                "evidence_type": str(row["evidence_type"]),
                "gate_label": str(row["gate_label"]),
                "paper_use": str(row["paper_use"]),
                "estimand_note": (
                    "heterogeneous_evidence_ladder"
                    if row["variant"] in {"C_min", "C_mid"}
                    else "locked_matched_composition_bucket_estimand"
                ),
            }
        )
    return output


def _range_label_proxy_pairs(
    *,
    seed_name: str,
    test_points,
    test_tensors,
    model_metrics: dict[str, Any],
    max_pairs_per_bucket: int,
    seed: int,
    torch,
) -> list[dict[str, Any]]:
    labels = [int(value) for value in test_tensors["opponent_card"].detach().cpu().tolist()]
    probabilities = model_metrics["range_probabilities"].detach().cpu()
    hidden = model_metrics["locked_hidden"].detach().cpu()
    by_bucket: dict[str, list[int]] = defaultdict(list)
    for index, point in enumerate(test_points):
        by_bucket[_matched_composition_bucket(point)].append(index)
    rows = []
    for bucket_index, (bucket, indices) in enumerate(sorted(by_bucket.items())):
        if len(indices) < 2:
            continue
        sampled_pairs = _sample_balanced_label_pairs(
            indices,
            labels=labels,
            max_pairs=max_pairs_per_bucket,
            seed=seed + bucket_index + _stable_int(bucket),
        )
        for left_index, right_index in sampled_pairs:
            left_label = labels[left_index]
            right_label = labels[right_index]
            left_probs = tuple(float(value) for value in probabilities[left_index].tolist())
            right_probs = tuple(float(value) for value in probabilities[right_index].tolist())
            rows.append(
                {
                    "seed": seed_name,
                    "composition_key": bucket,
                    "left_point_id": f"{test_points[left_index].hand_id}:{test_points[left_index].step_index}",
                    "right_point_id": f"{test_points[right_index].hand_id}:{test_points[right_index].step_index}",
                    "range_label_gap": 0.0 if left_label == right_label else 1.0,
                    "hidden_cosine_distance": _cosine_distance(hidden[left_index], hidden[right_index], torch),
                    "hidden_l2_distance": _l2_distance(hidden[left_index], hidden[right_index], torch),
                    "range_head_js": jensen_shannon(left_probs, right_probs),
                    "policy_js": _categorical_js(
                        model_metrics["policy_probabilities"][left_index],
                        model_metrics["policy_probabilities"][right_index],
                        torch,
                    ),
                    "value_abs_delta": float(
                        abs(
                            (
                                model_metrics["value_predictions"][left_index]
                                - model_metrics["value_predictions"][right_index]
                            ).detach().cpu().item()
                        )
                    ),
                }
            )
    return rows


def _sample_balanced_label_pairs(
    indices: list[int],
    *,
    labels: list[int],
    max_pairs: int,
    seed: int,
) -> list[tuple[int, int]]:
    same_pairs = []
    different_pairs = []
    for left in range(len(indices)):
        for right in range(left + 1, len(indices)):
            pair = (indices[left], indices[right])
            if labels[pair[0]] == labels[pair[1]]:
                same_pairs.append(pair)
            else:
                different_pairs.append(pair)
    if not same_pairs or not different_pairs:
        return _sample_pairs(same_pairs + different_pairs, max_pairs=max_pairs, seed=seed)
    per_group = max(1, max_pairs // 2)
    sampled_same = _sample_pairs(same_pairs, max_pairs=per_group, seed=seed + 17)
    sampled_different = _sample_pairs(different_pairs, max_pairs=max_pairs - len(sampled_same), seed=seed + 31)
    return sorted(sampled_same + sampled_different)


def _sample_pairs(pairs: list[tuple[int, int]], *, max_pairs: int, seed: int) -> list[tuple[int, int]]:
    if len(pairs) <= max_pairs:
        return sorted(pairs)
    rng = random.Random(seed)
    return sorted(rng.sample(pairs, max_pairs))


def _proxy_pair_summary(
    pairs: list[dict[str, Any]],
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    if not pairs:
        return {
            "pair_count": 0,
            "bucket_count": 0,
            "mean_range_label_gap": 0.0,
            "mean_hidden_cosine_distance": 0.0,
            "mean_hidden_l2_distance": 0.0,
            "mean_range_head_js": 0.0,
            "mean_policy_js": 0.0,
            "mean_value_abs_delta": 0.0,
            "range_label_gap_bins": [],
            "range_label_gap_slope": {"point": 0.0, "lower": 0.0, "upper": 0.0},
            "hidden_cosine_distance_bucket_ci": {"mean_delta": 0.0, "lower_delta": 0.0, "upper_delta": 0.0},
            "posterior_proxy_note": "no eligible same-composition range-label proxy pairs",
            "posterior_proxy_gap_tracking_failed": True,
        }
    hidden_values = [float(row["hidden_cosine_distance"]) for row in pairs]
    slope_interval = _bucket_bootstrap_slope_interval(
        pairs,
        x_key="range_label_gap",
        y_key="hidden_cosine_distance",
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    hidden_ci = _bucket_bootstrap_value_ci(
        pairs,
        value_key="hidden_cosine_distance",
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed + 777,
        confidence_level=confidence_level,
    )
    return {
        "pair_count": len(pairs),
        "bucket_count": len({str(row["composition_key"]) for row in pairs}),
        "mean_range_label_gap": _mean([float(row["range_label_gap"]) for row in pairs]),
        "mean_hidden_cosine_distance": _mean(hidden_values),
        "mean_hidden_l2_distance": _mean([float(row["hidden_l2_distance"]) for row in pairs]),
        "mean_range_head_js": _mean([float(row["range_head_js"]) for row in pairs]),
        "mean_policy_js": _mean([float(row["policy_js"]) for row in pairs]),
        "mean_value_abs_delta": _mean([float(row["value_abs_delta"]) for row in pairs]),
        "range_label_gap_bins": _range_label_gap_bins(pairs),
        "range_label_gap_slope": slope_interval,
        "hidden_cosine_distance_bucket_ci": hidden_ci,
        "posterior_proxy_note": (
            "range_label_gap is a binary same/different opponent 169-class label proxy inside locked "
            "matched-composition buckets; it is not exact oracle posterior JS."
        ),
        "posterior_proxy_gap_tracking_failed": slope_interval["lower"] <= 0.0,
    }


def _boundedness_aggregate(
    *,
    composition_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    matched_deltas = [float(row["matched_gate"]["delta_residual_minus_prior"]) for row in seed_rows]
    proxy_means = [float(row["range_label_proxy"]["mean_hidden_cosine_distance"]) for row in seed_rows]
    slope_points = [float(row["range_label_proxy"]["range_label_gap_slope"]["point"]) for row in seed_rows]
    return {
        "seed_count": len(seed_rows),
        "composition_ladder_label": _composition_ladder_label(composition_rows),
        "matched_delta_mean": _mean(matched_deltas),
        "matched_delta_seed_ci": _bootstrap_mean_ci(
            matched_deltas,
            bootstrap_count=bootstrap_count,
            bootstrap_seed=bootstrap_seed,
            confidence_level=confidence_level,
        ),
        "hidden_cosine_distance_mean_over_seeds": _mean(proxy_means),
        "range_label_gap_slope_mean_over_seeds": _mean(slope_points),
        "range_label_gap_slope_seed_ci": _bootstrap_mean_ci(
            slope_points,
            bootstrap_count=bootstrap_count,
            bootstrap_seed=bootstrap_seed + 1000,
            confidence_level=confidence_level,
        ),
        "proxy_slope_decision_rule": "seed_level_bucket_bootstrap_ci_dominates_descriptive_seed_point_ci",
        "proxy_seed_bucket_ci_positive_count": sum(
            1 for row in seed_rows if float(row["range_label_proxy"]["range_label_gap_slope"]["lower"]) > 0.0
        ),
        "proxy_seed_bucket_ci_cross_zero_count": sum(
            1 for row in seed_rows if float(row["range_label_proxy"]["range_label_gap_slope"]["lower"]) <= 0.0
        ),
        "seed_label_counts": dict(Counter(row["final_diagnostic_label"] for row in seed_rows)),
    }


def _boundedness_label(
    composition_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> str:
    labels = {row["final_diagnostic_label"] for row in seed_rows}
    full = next((row for row in composition_rows if row["variant"] == "C_full"), None)
    if full is None:
        return INCONCLUSIVE
    full_blocks = float(full["lower_delta"]) <= 0.0
    slope_ci = aggregate["range_label_gap_slope_seed_ci"]
    positive_seed_count = int(aggregate["proxy_seed_bucket_ci_positive_count"])
    if full_blocks and labels <= {COMPOSITION_BOUNDED_SUPPORTED} and positive_seed_count == 0:
        return COMPOSITION_BOUNDED_SUPPORTED
    if any(label == ORDERED_PROXY_COMPONENT_SUPPORTED for label in labels) or positive_seed_count > 0:
        return ORDERED_PROXY_COMPONENT_SUPPORTED
    return INCONCLUSIVE


def _boundedness_seed_label(matched_row: dict[str, Any], proxy: dict[str, Any]) -> str:
    matched_blocks = float(matched_row["bucket_bootstrap_lower_delta"]) <= 0.0
    has_proxy_support = int(proxy["pair_count"]) > 0 and int(proxy["bucket_count"]) > 0
    slope = proxy["range_label_gap_slope"]
    proxy_not_positive = float(slope["lower"]) <= 0.0
    if matched_blocks and has_proxy_support and proxy_not_positive:
        return COMPOSITION_BOUNDED_SUPPORTED
    if float(slope["lower"]) > 0.0:
        return ORDERED_PROXY_COMPONENT_SUPPORTED
    return INCONCLUSIVE


def _composition_ladder_label(rows: list[dict[str, Any]]) -> str:
    full = next((row for row in rows if row["variant"] == "C_full"), None)
    if full is None:
        return INCONCLUSIVE
    all_block = all(float(row["lower_delta"]) <= 0.0 for row in rows)
    if all_block:
        return "all_composition_variants_block"
    if float(full["lower_delta"]) <= 0.0:
        return "full_composition_only_blocks"
    return INCONCLUSIVE


def _matched_row_for_seed(matched_report: dict[str, Any], seed_name: str) -> dict[str, Any]:
    for row in matched_report.get("rows", []):
        if isinstance(row, dict) and row.get("seed") == seed_name:
            return row
    raise ValueError(f"matched report has no row for seed: {seed_name}")


def _matched_composition_bucket(point) -> str:
    actions = [str(event["action"]) for event in point.visible_state["betting_history"] if event.get("type") == "action"]
    counts = {action: actions.count(action) for action in ACTIONS}
    value_bucket = int(point.terminal_return or 0)
    legal = ",".join(sorted(str(action) for action in point.legal_actions))
    last_action = actions[-1] if actions else "none"
    return "|".join(
        [
            f"player={point.acting_player}",
            f"round={point.visible_state['round']}",
            f"legal={legal}",
            f"action={point.action}",
            f"value={value_bucket}",
            f"len={len(actions)}",
            *(f"{action}={counts[action]}" for action in ACTIONS),
            f"last={last_action}",
        ]
    )


def _categorical_js(left, right, torch) -> float:
    left_probs = torch.as_tensor(left).detach().float().cpu()
    right_probs = torch.as_tensor(right).detach().float().cpu()
    return jensen_shannon(tuple(float(value) for value in left_probs.tolist()), tuple(float(value) for value in right_probs.tolist()))


def _bucket_bootstrap_slope_interval(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, float]:
    if not rows:
        raise ValueError("slope bootstrap requires rows")
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bucket[str(row["composition_key"])].append(row)
    buckets = sorted(by_bucket)
    rng = random.Random(bootstrap_seed)
    slopes = []
    for _ in range(bootstrap_count):
        sampled = []
        for _ in buckets:
            sampled.extend(by_bucket[buckets[rng.randrange(len(buckets))]])
        slopes.append(_linear_slope([float(row[x_key]) for row in sampled], [float(row[y_key]) for row in sampled]))
    slopes.sort()
    lower_index, upper_index = _ci_indices(bootstrap_count, confidence_level)
    point = _linear_slope([float(row[x_key]) for row in rows], [float(row[y_key]) for row in rows])
    return {"point": point, "lower": slopes[lower_index], "upper": slopes[upper_index]}


def _bucket_bootstrap_value_ci(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, float]:
    if not rows:
        raise ValueError("bucket bootstrap requires rows")
    by_bucket: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_bucket[str(row["composition_key"])].append(float(row[value_key]))
    bucket_values = [_mean(values) for _, values in sorted(by_bucket.items())]
    rng = random.Random(bootstrap_seed)
    means = []
    for _ in range(bootstrap_count):
        means.append(sum(bucket_values[rng.randrange(len(bucket_values))] for _ in bucket_values) / len(bucket_values))
    means.sort()
    lower_index, upper_index = _ci_indices(bootstrap_count, confidence_level)
    return {
        "mean_delta": _mean(bucket_values),
        "lower_delta": means[lower_index],
        "upper_delta": means[upper_index],
    }


def _range_label_gap_bins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    output = []
    for label, target_gap in (("same_range_label", 0.0), ("different_range_label", 1.0)):
        selected = [row for row in rows if float(row["range_label_gap"]) == target_gap]
        output.append(
            {
                "bin": label,
                "pair_count": len(selected),
                "mean_range_label_gap": _mean([float(row["range_label_gap"]) for row in selected]),
                "mean_hidden_cosine_distance": _mean([float(row["hidden_cosine_distance"]) for row in selected]),
                "mean_hidden_l2_distance": _mean([float(row["hidden_l2_distance"]) for row in selected]),
                "mean_range_head_js": _mean([float(row["range_head_js"]) for row in selected]),
                "mean_policy_js": _mean([float(row["policy_js"]) for row in selected]),
                "mean_value_abs_delta": _mean([float(row["value_abs_delta"]) for row in selected]),
            }
        )
    return output


def _cosine_distance(left, right, torch) -> float:
    numerator = torch.dot(left.float(), right.float())
    denominator = torch.linalg.vector_norm(left.float()) * torch.linalg.vector_norm(right.float())
    if float(denominator.item()) <= 1e-12:
        return 0.0
    similarity = float((numerator / denominator).clamp(-1.0, 1.0).item())
    return 1.0 - similarity


def _l2_distance(left, right, torch) -> float:
    return float(torch.linalg.vector_norm(left.float() - right.float()).item())


def _ci_indices(count: int, confidence_level: float) -> tuple[int, int]:
    if count <= 0:
        raise ValueError("bootstrap count must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    lower = int(((1.0 - confidence_level) / 2.0) * (count - 1))
    upper = int((1.0 - (1.0 - confidence_level) / 2.0) * (count - 1))
    return lower, upper


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator <= 0.0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator


def _c_min_key(point) -> tuple[Any, ...]:
    actions = _history_actions(point)
    counts = tuple(actions.count(action) for action in ACTIONS)
    last_action = actions[-1] if actions else "none"
    return ("len", len(actions), "counts", counts, "last", last_action)


def _round_position_legal_key(point) -> tuple[Any, ...]:
    state = point.visible_state
    return (
        "acting_player",
        str(state["acting_player"]),
        "round",
        str(state["round"]),
        "legal",
        tuple(sorted(str(action) for action in state["legal_actions"])),
    )


def _public_card_key(point) -> tuple[Any, ...]:
    state = point.visible_state
    return (
        "public_cards",
        tuple(str(card) for card in state["public_cards"]),
    )


def _history_actions(point) -> list[str]:
    return [
        str(event["action"])
        for event in point.visible_state["betting_history"]
        if event["type"] in {"action", "bag_action"}
    ]


def _bootstrap_mean_ci(
    values: list[float],
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, float]:
    if not values:
        raise ValueError("bootstrap values must not be empty")
    rng = random.Random(bootstrap_seed)
    means = []
    for _ in range(bootstrap_count):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    lower_index = int(((1.0 - confidence_level) / 2.0) * (bootstrap_count - 1))
    upper_index = int((1.0 - (1.0 - confidence_level) / 2.0) * (bootstrap_count - 1))
    return {
        "mean_delta": _mean(values),
        "lower_delta": means[lower_index],
        "upper_delta": means[upper_index],
    }


def _manifest(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": payload["version"],
        "report_type": "k.5.1_poker_predictive_link_manifest",
        "claim_scope": payload["claim_scope"],
        "output_dir": str(output_dir),
        "expected_outputs": [
            "k.5.1_poker_predictive_link_report.json",
            "k.5.1_poker_predictive_link_report.md",
            "k.5.1_poker_predictive_link_manifest.json",
        ],
        "archive_root": payload["archive_root"],
        "seed_names": payload["seed_names"],
        "bootstrap_count": payload["bootstrap_count"],
        "bootstrap_seed": payload["bootstrap_seed"],
        "batch_size": payload["batch_size"],
        "max_test_points": payload["max_test_points"],
        "cache_dir": payload["cache_dir"],
        "report_checksum": payload["checksum"],
        "final_diagnostic_label": payload["final_diagnostic_label"],
        "one_command_reproduction": (
            ".venv/bin/python scripts/build_poker_predictive_link_report.py "
            f"--archive-root {payload['archive_root']} "
            f"--output-dir {output_dir} "
            f"--bootstrap-count {payload['bootstrap_count']} "
            f"--bootstrap-seed {payload['bootstrap_seed']} "
            f"--batch-size {payload['batch_size']} "
            + (f"--cache-dir {payload['cache_dir']} " if payload["cache_dir"] is not None else "")
            + "--seed-names "
            + " ".join(payload["seed_names"])
        ),
    }


def _boundedness_manifest(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": payload["version"],
        "report_type": "k.5.2_composition_boundedness_manifest",
        "claim_scope": payload["claim_scope"],
        "output_dir": str(output_dir),
        "expected_outputs": [
            "k.5.2_composition_boundedness_report.json",
            "k.5.2_composition_boundedness_report.md",
            "k.5.2_composition_boundedness_manifest.json",
        ],
        "matched_report_path": payload["matched_report_path"],
        "composition_report_path": payload["composition_report_path"],
        "archive_root": payload["archive_root"],
        "seed_names": payload["seed_names"],
        "bootstrap_count": payload["bootstrap_count"],
        "bootstrap_seed": payload["bootstrap_seed"],
        "batch_size": payload["batch_size"],
        "max_test_points": payload["max_test_points"],
        "max_pairs_per_bucket": payload["max_pairs_per_bucket"],
        "report_checksum": payload["checksum"],
        "final_diagnostic_label": payload["final_diagnostic_label"],
        "one_command_reproduction": (
            ".venv/bin/python scripts/build_composition_boundedness_report.py "
            f"--matched-report {payload['matched_report_path']} "
            f"--composition-report {payload['composition_report_path']} "
            f"--archive-root {payload['archive_root']} "
            f"--output-dir {output_dir} "
            f"--bootstrap-count {payload['bootstrap_count']} "
            f"--bootstrap-seed {payload['bootstrap_seed']} "
            f"--batch-size {payload['batch_size']} "
            f"--max-pairs-per-bucket {payload['max_pairs_per_bucket']} "
            + ("--max-test-points {value} ".format(value=payload["max_test_points"]) if payload["max_test_points"] is not None else "")
            + "--seed-names "
            + " ".join(payload["seed_names"])
        ),
    }


def _behavior_link_manifest(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": payload["version"],
        "report_type": "k.5.5_representation_behavior_link_manifest",
        "claim_scope": payload["claim_scope"],
        "output_dir": str(output_dir),
        "expected_outputs": [
            "k.5.5_representation_behavior_link_report.json",
            "k.5.5_representation_behavior_link_report.md",
            "k.5.5_representation_behavior_link_manifest.json",
        ],
        "archive_root": payload["archive_root"],
        "seed_names": payload["seed_names"],
        "bootstrap_count": payload["bootstrap_count"],
        "bootstrap_seed": payload["bootstrap_seed"],
        "batch_size": payload["batch_size"],
        "max_train_points": payload["max_train_points"],
        "max_test_points": payload["max_test_points"],
        "probe_epochs": payload["probe_epochs"],
        "probe_batch_size": payload["probe_batch_size"],
        "learning_rate": payload["learning_rate"],
        "target_leakage_guard": payload["target_leakage_guard"],
        "report_checksum": payload["checksum"],
        "final_diagnostic_label": payload["final_diagnostic_label"],
        "one_command_reproduction": (
            ".venv/bin/python scripts/build_representation_behavior_link_report.py "
            f"--archive-root {payload['archive_root']} "
            f"--output-dir {output_dir} "
            f"--bootstrap-count {payload['bootstrap_count']} "
            f"--bootstrap-seed {payload['bootstrap_seed']} "
            f"--batch-size {payload['batch_size']} "
            f"--probe-epochs {payload['probe_epochs']} "
            f"--probe-batch-size {payload['probe_batch_size']} "
            f"--learning-rate {payload['learning_rate']} "
            + ("--max-train-points {value} ".format(value=payload["max_train_points"]) if payload["max_train_points"] is not None else "")
            + ("--max-test-points {value} ".format(value=payload["max_test_points"]) if payload["max_test_points"] is not None else "")
            + "--seed-names "
            + " ".join(payload["seed_names"])
        ),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# k.5.1 Poker Predictive-Link Report",
        "",
        "This is held-out behavior prediction evidence. It is not causal evidence and does not upgrade the poker belief claim by itself.",
        "",
        f"final_diagnostic_label: `{payload['final_diagnostic_label']}`",
        "",
        "| seed | label | model action acc | model value MSE | delta vs prior | prior lower | delta vs C_min | C_min lower | delta vs C_full | C_full lower |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        metrics = row["behavior_metrics"]
        prior = metrics["baseline_rows"]["action_prior"]
        c_min = metrics["baseline_rows"]["C_min"]
        c_full = metrics["baseline_rows"]["C_full"]
        lines.append(
            "| {seed} | {label} | {acc:.6f} | {mse:.6f} | {prior_delta:.6f} | {prior_lower:.6f} | {cmin_delta:.6f} | {cmin_lower:.6f} | {cfull_delta:.6f} | {cfull_lower:.6f} |".format(
                seed=row["seed"],
                label=row["final_diagnostic_label"],
                acc=metrics["model_action_accuracy"],
                mse=metrics["model_value_mse"],
                prior_delta=prior["action_accuracy_delta_model_minus_baseline"],
                prior_lower=prior["action_accuracy_delta_ci"]["lower_delta"],
                cmin_delta=c_min["action_accuracy_delta_model_minus_baseline"],
                cmin_lower=c_min["action_accuracy_delta_ci"]["lower_delta"],
                cfull_delta=c_full["action_accuracy_delta_model_minus_baseline"],
                cfull_lower=c_full["action_accuracy_delta_ci"]["lower_delta"],
            )
        )
    return "\n".join(lines) + "\n"


def _behavior_link_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# k.5.5 Representation-To-Behavior Link Report",
        "",
        "This report tests whether frozen hidden representations predict behavior targets beyond input-only public-composition features. It is not a causal mediation test.",
        "",
        f"final_diagnostic_label: `{payload['final_diagnostic_label']}`",
        "",
        "| seed | label | hidden acc | C_full linear acc | hidden delta | hidden lower | residual acc | residual delta | residual lower | hidden value delta | residual value delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        metrics = row["metrics"]
        hidden = metrics["probe_rows"]["hidden"]
        c_full = metrics["probe_rows"]["C_full_linear"]
        residual = metrics["probe_rows"]["residual_hidden_after_C_full_linear"]
        hidden_delta = metrics["comparisons"]["hidden_vs_C_full_linear"]
        residual_delta = metrics["comparisons"]["residual_hidden_vs_C_full_linear"]
        lines.append(
            "| {seed} | `{label}` | {hidden_acc:.6f} | {cfull_acc:.6f} | {hidden_delta:.6f} | {hidden_lower:.6f} | {residual_acc:.6f} | {residual_delta:.6f} | {residual_lower:.6f} | {hidden_value:.6f} | {residual_value:.6f} |".format(
                seed=row["seed"],
                label=row["final_diagnostic_label"],
                hidden_acc=hidden["action_accuracy"],
                cfull_acc=c_full["action_accuracy"],
                hidden_delta=hidden_delta["action_accuracy_delta"],
                hidden_lower=hidden_delta["action_accuracy_delta_ci"]["lower_delta"],
                residual_acc=residual["action_accuracy"],
                residual_delta=residual_delta["action_accuracy_delta"],
                residual_lower=residual_delta["action_accuracy_delta_ci"]["lower_delta"],
                hidden_value=hidden_delta["value_mse_delta_right_minus_left"],
                residual_value=residual_delta["value_mse_delta_right_minus_left"],
            )
        )
    aggregate = payload["aggregate"]
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| comparison | mean action delta | action lower | action upper | mean value delta | value lower | value upper |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in aggregate["comparisons"].items():
        action_ci = row["action_accuracy_delta_seed_ci"]
        value_ci = row["value_mse_delta_seed_ci"]
        lines.append(
            "| {name} | {action:.6f} | {action_lower:.6f} | {action_upper:.6f} | {value:.6f} | {value_lower:.6f} | {value_upper:.6f} |".format(
                name=name,
                action=row["mean_action_accuracy_delta"],
                action_lower=action_ci["lower_delta"],
                action_upper=action_ci["upper_delta"],
                value=row["mean_value_mse_delta_right_minus_left"],
                value_lower=value_ci["lower_delta"],
                value_upper=value_ci["upper_delta"],
            )
        )
    lines.extend(
        [
            "",
            "## Claim Discipline",
            "",
            payload["allowed_paper_claim"],
            "",
            f"Forbidden claim: {payload['forbidden_paper_claim']}",
        ]
    )
    return "\n".join(lines) + "\n"


def _boundedness_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# k.5.2 Composition-Boundedness Report",
        "",
        "This is a poker composition-boundedness audit. The pair analysis uses opponent 169-class range-label proxy gaps, not exact oracle posterior JS.",
        "",
        f"final_diagnostic_label: `{payload['final_diagnostic_label']}`",
        "",
        "## Composition Ladder",
        "",
        "| variant | lower delta | mean delta | upper delta | label | estimand note |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in payload["composition_ladder_rows"]:
        upper = "NA" if row["upper_delta"] is None else f"{row['upper_delta']:.6f}"
        lines.append(
            "| {variant} | {lower:.6f} | {mean:.6f} | {upper} | `{label}` | {note} |".format(
                variant=row["variant"],
                lower=row["lower_delta"],
                mean=row["mean_delta"],
                upper=upper,
                label=row["gate_label"],
                note=row["estimand_note"],
            )
        )
    lines.extend(
        [
            "",
            "## Seed Rows",
            "",
            "| seed | label | S7 delta | S7 lower | proxy pairs | proxy buckets | same hidden dist | diff hidden dist | gap slope | slope lower | slope upper |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["seed_rows"]:
        proxy = row["range_label_proxy"]
        bins = {bin_row["bin"]: bin_row for bin_row in proxy["range_label_gap_bins"]}
        same = bins.get("same_range_label", {})
        different = bins.get("different_range_label", {})
        slope = proxy["range_label_gap_slope"]
        lines.append(
            "| {seed} | `{label}` | {delta:.6f} | {lower:.6f} | {pairs} | {buckets} | {same:.6f} | {diff:.6f} | {slope:.6f} | {slope_lower:.6f} | {slope_upper:.6f} |".format(
                seed=row["seed"],
                label=row["final_diagnostic_label"],
                delta=row["matched_gate"]["delta_residual_minus_prior"],
                lower=row["matched_gate"]["bucket_bootstrap_lower_delta"],
                pairs=proxy["pair_count"],
                buckets=proxy["bucket_count"],
                same=float(same.get("mean_hidden_cosine_distance", 0.0)),
                diff=float(different.get("mean_hidden_cosine_distance", 0.0)),
                slope=slope["point"],
                slope_lower=slope["lower"],
                slope_upper=slope["upper"],
            )
        )
    aggregate = payload["aggregate"]
    slope_ci = aggregate["range_label_gap_slope_seed_ci"]
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| composition ladder label | `{aggregate['composition_ladder_label']}` |",
            f"| matched delta mean | {aggregate['matched_delta_mean']:.6f} |",
            f"| proxy slope mean | {aggregate['range_label_gap_slope_mean_over_seeds']:.6f} |",
            f"| proxy slope seed lower | {slope_ci['lower_delta']:.6f} |",
            f"| proxy slope seed upper | {slope_ci['upper_delta']:.6f} |",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _read_required_json(path: Path) -> dict[str, Any]:
    data = _read_optional_json(path)
    if data is None:
        raise FileNotFoundError(f"required JSON artifact is missing or not an object: {path}")
    return data


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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
