"""Orchestrate the k.9 exact known-teacher posterior readout product.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

import numpy as np

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.config import load_limit_ar_train_config
from jorjor.data.split import SplitFractions, apply_split, build_split_manifest
from jorjor.models.ar import build_tiny_ar_model
from jorjor.models.tiny import require_torch
from jorjor.probe.limit_posterior_oracle import (
    LimitPosteriorOracleResult,
    compute_limit_posterior_oracle,
    read_limit_posterior_cache,
    write_limit_posterior_cache,
)
from jorjor.probe.limit_ar import _load_ar_checkpoint_with_vocab_compat
from jorjor.probe.posterior_pairs import (
    batched_jensen_shannon,
    build_fixed_matched_pairs,
    dyadic_alignment_bootstrap,
    paired_mae_improvement_bootstrap,
)
from jorjor.probe.soft_posterior import (
    card_prior_offset_logits,
    equal_hand_bootstrap,
    fit_ols_residualizer,
    fit_soft_posterior_readout,
    fit_standardizer,
    five_run_statistics,
    posterior_point_metrics,
)
from jorjor.tokenizer import ACTIONS, LimitTokenizer
from jorjor.train.leduc_tiny import _features_to_tensors, _resolve_device
from jorjor.train.limit_tiny import _load_points


CONDITIONING_ID = "full_player_information_pre_action_teacher_v1"
LOCKED_LAYER = "encoder_layer_2"
FIVE_RUN_SEEDS = ("seed_k", "seed_l", "seed_m", "seed_n", "seed_o")
LEARNED_VARIANTS = (
    "hidden_full_locked",
    "target_side_av_nuisance",
    "history_composition_30d",
    "target_side_av_plus_composition",
    "hidden_residualized_on_av_composition",
    "av_composition_plus_residual_hidden",
)
CONTROL_VARIANTS = ("uniform_169", "card_removal_prior", "identity_oracle")
READOUT_SEED_BASES = {
    "seed_k": 91000,
    "seed_l": 92000,
    "seed_m": 93000,
    "seed_n": 94000,
    "seed_o": 95000,
}
PRIMARY_BASELINE = "target_side_av_plus_composition"
PRIMARY_CANDIDATE = "av_composition_plus_residual_hidden"
REPAIRED_SUPPORT_MIN_BUCKETS = 20
REPAIRED_SUPPORT_MIN_NONZERO_PAIRS = 1500
REPAIRED_SUPPORT_MIN_HAND_NODES = 1000


@dataclass(frozen=True)
class PointSplits:
    """Frozen complete-hand point splits for one end-to-end run."""

    train: tuple[Any, ...]
    validation: tuple[Any, ...]
    test: tuple[Any, ...]


@dataclass(frozen=True)
class HiddenSplits:
    """Full-input locked hidden features aligned to PointSplits."""

    train: Any
    validation: Any
    test: Any
    locked_layer: str
    checkpoint_load_info: dict[str, Any]


@dataclass(frozen=True)
class FeatureSplits:
    """Train/validation/test matrices for one readout variant."""

    train: Any
    validation: Any
    test: Any


@dataclass(frozen=True)
class ReadoutFeatureBundle:
    """Predeclared standardized feature matrices and their audit metadata."""

    variants: dict[str, FeatureSplits]
    dimensions: dict[str, int]
    parameter_counts: dict[str, int]
    residualizer: Any


@dataclass(frozen=True)
class OracleSplits:
    """Exact posterior targets and card priors for frozen data splits."""

    train: LimitPosteriorOracleResult
    validation: LimitPosteriorOracleResult
    test: LimitPosteriorOracleResult
    manifests: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ReadoutEvaluation:
    """Fitted readout metadata and held-out distribution products."""

    records: dict[str, dict[str, Any]]
    logits: dict[str, Any]
    probabilities: dict[str, Any]
    point_metrics: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class SeedEvaluationSummary:
    """Hand-cluster scores and matched-pair products for one frozen run."""

    readout_rows: dict[str, dict[str, Any]]
    primary_improvements: dict[str, dict[str, float]]
    pair_summary: dict[str, Any]
    pair_rows: list[dict[str, Any]]
    final_diagnostic_label: str


def point_id(point) -> str:
    """Return the stable hand/decision identity used across all artifacts."""

    return f"{point.hand_id}:{point.step_index}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _complete_hand_prefix(points: tuple[Any, ...], max_hand_count: int | None) -> tuple[Any, ...]:
    if max_hand_count is None:
        return points
    if max_hand_count <= 0:
        raise ValueError("max hand counts must be positive")
    selected = []
    seen: set[str] = set()
    for point in points:
        if point.hand_id not in seen:
            if len(seen) >= max_hand_count:
                continue
            seen.add(point.hand_id)
        if point.hand_id in seen:
            selected.append(point)
    return tuple(selected)


def load_frozen_point_splits(
    train_config_path: Path,
    *,
    max_train_hands: int | None = None,
    max_validation_hands: int | None = None,
    max_test_hands: int | None = None,
) -> tuple[Any, PointSplits]:
    """Regenerate the exact hand-id split used by a frozen AR checkpoint."""

    config = load_limit_ar_train_config(train_config_path)
    points = _load_points(config)
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
    output = PointSplits(
        train=_complete_hand_prefix(tuple(splits.train), max_train_hands),
        validation=_complete_hand_prefix(tuple(splits.validation), max_validation_hands),
        test=_complete_hand_prefix(tuple(splits.test), max_test_hands),
    )
    train_hands = {point.hand_id for point in output.train}
    validation_hands = {point.hand_id for point in output.validation}
    test_hands = {point.hand_id for point in output.test}
    if train_hands & validation_hands or train_hands & test_hands or validation_hands & test_hands:
        raise ValueError("train/validation/test hand ids overlap")
    return config, output


def load_or_compute_oracle_splits(
    *,
    points: PointSplits,
    cache_dir: Path,
) -> OracleSplits:
    """Load checksummed exact posterior caches or build them from complete hands."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, LimitPosteriorOracleResult] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for split_name, split_points in (
        ("train", points.train),
        ("validation", points.validation),
        ("test", points.test),
    ):
        cache_path = cache_dir / f"{split_name}_posterior_counts.npz"
        manifest_path = cache_path.with_suffix(".manifest.json")
        if cache_path.is_file() or manifest_path.is_file():
            result = read_limit_posterior_cache(cache_path)
            manifest = read_json_object(manifest_path)
        else:
            result = compute_limit_posterior_oracle(split_points)
            manifest = write_limit_posterior_cache(result, cache_path)
        expected_ids = tuple(point_id(point) for point in split_points)
        if result.point_ids != expected_ids:
            raise ValueError(f"{split_name} posterior cache point ids do not match the frozen split")
        if result.conditioning_contract.find("deterministic_heuristic_teacher") < 0:
            raise ValueError("posterior cache conditioning contract changed")
        results[split_name] = result
        manifests[split_name] = manifest
    return OracleSplits(
        train=results["train"],
        validation=results["validation"],
        test=results["test"],
        manifests=manifests,
    )


def _oracle_tensors(result: LimitPosteriorOracleResult):
    """Convert read-only NumPy oracle arrays into writable CPU tensors."""

    torch, _ = require_torch()
    posterior = torch.from_numpy(np.array(result.posterior, copy=True)).float()
    card_prior = torch.from_numpy(np.array(result.card_prior, copy=True)).float()
    log_prior = torch.full_like(card_prior, -torch.inf)
    support = card_prior > 0.0
    log_prior[support] = torch.log(card_prior[support])
    return posterior, card_prior, log_prior


def extract_full_input_locked_hidden(
    *,
    checkpoint_path: Path,
    train_config,
    points: PointSplits,
    device: str,
    batch_size: int,
) -> HiddenSplits:
    """Extract the original unmasked locked layer for all frozen splits."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {checkpoint_path}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    torch, _ = require_torch()
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
    model.eval()

    locked_layer_used = LOCKED_LAYER

    def extract(split_points: tuple[Any, ...]):
        nonlocal locked_layer_used
        features = [tokenizer.encode(point) for point in split_points]
        input_ids = _features_to_tensors(features, torch)["input_ids"]
        rows = []
        with torch.no_grad():
            for start in range(0, input_ids.shape[0], batch_size):
                stop = min(start + batch_size, input_ids.shape[0])
                layers = model.encode_layers(input_ids[start:stop].to(resolved_device))
                layer = LOCKED_LAYER if LOCKED_LAYER in layers else "sequence_hidden"
                locked_layer_used = layer
                rows.append(layers[layer].detach().cpu())
        if not rows:
            return torch.empty((0, train_config.hidden_size), dtype=torch.float32)
        return torch.cat(rows, dim=0)

    train_hidden = extract(points.train)
    validation_hidden = extract(points.validation)
    test_hidden = extract(points.test)
    return HiddenSplits(
        train=train_hidden,
        validation=validation_hidden,
        test=test_hidden,
        locked_layer=locked_layer_used,
        checkpoint_load_info=load_info,
    )


def target_side_av_features(points: PointSplits):
    """Build current-action plus terminal-return nuisance labels, never posterior inputs."""

    torch, _ = require_torch()
    action_to_id = {action: index for index, action in enumerate(ACTIONS)}

    def raw(split_points: tuple[Any, ...]):
        actions = torch.tensor([action_to_id[point.action] for point in split_points], dtype=torch.long)
        action_features = torch.nn.functional.one_hot(actions, num_classes=len(ACTIONS)).float()
        values = torch.tensor([float(point.terminal_return or 0) for point in split_points], dtype=torch.float32).unsqueeze(1)
        return action_features, values

    train_action, train_value = raw(points.train)
    validation_action, validation_value = raw(points.validation)
    test_action, test_value = raw(points.test)
    mean = train_value.mean(dim=0, keepdim=True)
    scale = train_value.std(dim=0, unbiased=False, keepdim=True)
    scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
    return FeatureSplits(
        train=torch.cat([train_action, (train_value - mean) / scale], dim=1),
        validation=torch.cat([validation_action, (validation_value - mean) / scale], dim=1),
        test=torch.cat([test_action, (test_value - mean) / scale], dim=1),
    )


def history_composition_features(points: PointSplits):
    """Build the existing 30-dimensional prior-action composition summary."""

    torch, _ = require_torch()

    def encode(split_points: tuple[Any, ...]):
        rows = []
        for point in split_points:
            actions = [
                str(event["action"])
                for event in point.visible_state["betting_history"]
                if event.get("type") in {"action", "bag_action"}
            ]
            denominator = max(len(actions), 1)
            row = [float(len(actions)), min(float(len(actions)), 16.0) / 16.0]
            row.extend(float(actions.count(action)) for action in ACTIONS)
            row.extend(float(actions.count(action)) / denominator for action in ACTIONS)
            row.extend(1.0 if action in actions else 0.0 for action in ACTIONS)
            first_action = actions[0] if actions else None
            last_action = actions[-1] if actions else None
            row.extend(1.0 if first_action == action else 0.0 for action in ACTIONS)
            row.extend(1.0 if last_action == action else 0.0 for action in ACTIONS)
            for action in ACTIONS:
                positions = [
                    index / 15.0
                    for index, history_action in enumerate(actions[:16])
                    if history_action == action
                ]
                row.append(sum(positions) / len(positions) if positions else 0.0)
                row.append(max(positions) if positions else 0.0)
            if len(row) != 30:
                raise RuntimeError(f"composition feature width changed: {len(row)}")
            rows.append(row)
        if not rows:
            return torch.empty((0, 30), dtype=torch.float32)
        return torch.tensor(rows, dtype=torch.float32)

    return FeatureSplits(
        train=encode(points.train),
        validation=encode(points.validation),
        test=encode(points.test),
    )


def _standardize_splits(features: FeatureSplits) -> tuple[FeatureSplits, Any]:
    standardizer = fit_standardizer(features.train)
    return (
        FeatureSplits(
            train=standardizer.transform(features.train),
            validation=standardizer.transform(features.validation),
            test=standardizer.transform(features.test),
        ),
        standardizer,
    )


def build_readout_feature_bundle(
    *,
    points: PointSplits,
    hidden: HiddenSplits,
) -> ReadoutFeatureBundle:
    """Build all requested rows plus the nested conditional hidden row."""

    torch, _ = require_torch()
    row_counts = (len(points.train), len(points.validation), len(points.test))
    hidden_counts = (hidden.train.shape[0], hidden.validation.shape[0], hidden.test.shape[0])
    if hidden_counts != row_counts:
        raise ValueError(f"hidden features do not align with point splits: {hidden_counts} != {row_counts}")

    av_raw = target_side_av_features(points)
    composition_raw = history_composition_features(points)
    av_composition_raw = FeatureSplits(
        train=torch.cat([av_raw.train, composition_raw.train], dim=1),
        validation=torch.cat([av_raw.validation, composition_raw.validation], dim=1),
        test=torch.cat([av_raw.test, composition_raw.test], dim=1),
    )
    hidden_raw = FeatureSplits(train=hidden.train, validation=hidden.validation, test=hidden.test)
    hidden_standardized, _ = _standardize_splits(hidden_raw)
    av_standardized, _ = _standardize_splits(av_raw)
    composition_standardized, _ = _standardize_splits(composition_raw)
    av_composition_standardized, _ = _standardize_splits(av_composition_raw)

    residualizer = fit_ols_residualizer(hidden.train, av_composition_standardized.train)
    residual_raw = FeatureSplits(
        train=residualizer.transform(hidden.train, av_composition_standardized.train),
        validation=residualizer.transform(hidden.validation, av_composition_standardized.validation),
        test=residualizer.transform(hidden.test, av_composition_standardized.test),
    )
    residual_standardized, _ = _standardize_splits(residual_raw)
    nested = FeatureSplits(
        train=torch.cat([av_composition_standardized.train, residual_standardized.train], dim=1),
        validation=torch.cat(
            [av_composition_standardized.validation, residual_standardized.validation],
            dim=1,
        ),
        test=torch.cat([av_composition_standardized.test, residual_standardized.test], dim=1),
    )
    variants = {
        "hidden_full_locked": hidden_standardized,
        "target_side_av_nuisance": av_standardized,
        "history_composition_30d": composition_standardized,
        "target_side_av_plus_composition": av_composition_standardized,
        "hidden_residualized_on_av_composition": residual_standardized,
        "av_composition_plus_residual_hidden": nested,
    }
    if tuple(variants) != LEARNED_VARIANTS:
        raise RuntimeError("readout feature order changed")
    dimensions = {name: int(split.train.shape[1]) for name, split in variants.items()}
    parameter_counts = {name: (dimension + 1) * 169 for name, dimension in dimensions.items()}
    return ReadoutFeatureBundle(
        variants=variants,
        dimensions=dimensions,
        parameter_counts=parameter_counts,
        residualizer=residualizer,
    )


def _predict_offset_logits(
    *,
    model,
    features,
    card_log_prior,
    device: str,
    batch_size: int,
):
    torch, _ = require_torch()
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            stop = min(start + batch_size, features.shape[0])
            rows.append(
                card_prior_offset_logits(
                    model,
                    features[start:stop].to(device),
                    card_log_prior[start:stop].to(device),
                ).detach().cpu()
            )
    return torch.cat(rows, dim=0)


def _probability_logits(probabilities):
    torch, _ = require_torch()
    values = probabilities.detach().float().cpu()
    logits = torch.full_like(values, -torch.inf)
    support = values > 0.0
    logits[support] = torch.log(values[support])
    return logits


def fit_and_evaluate_readouts(
    *,
    seed_name: str,
    features: ReadoutFeatureBundle,
    oracle: OracleSplits,
    output_dir: Path,
    device: str,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    residual_learning_rate: float,
    weight_decay: float,
) -> ReadoutEvaluation:
    """Fit every predeclared soft-posterior row and evaluate locked test points."""

    if seed_name not in READOUT_SEED_BASES:
        raise ValueError(f"unknown frozen run: {seed_name}")
    torch, _ = require_torch()
    train_target, train_prior, train_log_prior = _oracle_tensors(oracle.train)
    validation_target, validation_prior, validation_log_prior = _oracle_tensors(oracle.validation)
    test_target, test_prior, test_log_prior = _oracle_tensors(oracle.test)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "readout_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict[str, Any]] = {}
    logits: dict[str, Any] = {}
    probabilities: dict[str, Any] = {}
    point_metrics: dict[str, dict[str, Any]] = {}
    for variant_index, variant in enumerate(LEARNED_VARIANTS, start=1):
        split = features.variants[variant]
        probe_seed = READOUT_SEED_BASES[seed_name] + variant_index
        variant_learning_rate = (
            residual_learning_rate
            if variant in {
                "hidden_residualized_on_av_composition",
                "av_composition_plus_residual_hidden",
            }
            else learning_rate
        )
        fit = fit_soft_posterior_readout(
            split.train,
            train_target,
            train_log_prior,
            split.validation,
            validation_target,
            validation_log_prior,
            epochs=epochs,
            learning_rate=variant_learning_rate,
            batch_size=batch_size,
            seed=probe_seed,
            device=device,
            patience=patience,
            weight_decay=weight_decay,
        )
        variant_logits = _predict_offset_logits(
            model=fit.model,
            features=split.test,
            card_log_prior=test_log_prior,
            device=device,
            batch_size=batch_size,
        )
        state_path = weights_dir / f"{variant}.pt"
        torch.save(
            {key: value.detach().cpu() for key, value in fit.model.state_dict().items()},
            state_path,
        )
        logits[variant] = variant_logits
        probabilities[variant] = torch.softmax(variant_logits, dim=1)
        point_metrics[variant] = posterior_point_metrics(test_target, variant_logits)
        records[variant] = {
            "role": "learned_card_prior_offset_linear_readout",
            "feature_dimension": features.dimensions[variant],
            "parameter_count": features.parameter_counts[variant],
            "probe_seed": probe_seed,
            "learning_rate": variant_learning_rate,
            "weight_decay": weight_decay,
            "best_epoch": fit.best_epoch,
            "epochs_completed": fit.epochs_completed,
            "best_validation_cross_entropy_nats": fit.best_validation_cross_entropy_nats,
            "train_cross_entropy_history_nats": list(fit.train_cross_entropy_nats),
            "validation_cross_entropy_history_nats": list(fit.validation_cross_entropy_nats),
            "weights_path": str(state_path.relative_to(output_dir)),
            "weights_sha256": _sha256(state_path),
        }

    control_logits = {
        "uniform_169": torch.zeros_like(test_target),
        "card_removal_prior": test_log_prior,
        "identity_oracle": _probability_logits(test_target),
    }
    for variant, variant_logits in control_logits.items():
        logits[variant] = variant_logits
        probabilities[variant] = (
            test_target.clone()
            if variant == "identity_oracle"
            else torch.softmax(variant_logits, dim=1)
        )
        point_metrics[variant] = posterior_point_metrics(test_target, variant_logits)
        records[variant] = {
            "role": "control",
            "feature_dimension": 0,
            "parameter_count": 0,
        }

    identity = point_metrics["identity_oracle"]
    identity_tolerances = {"kl_nats": 1e-7, "js_nats": 1e-10, "brier": 1e-10}
    for metric, tolerance in identity_tolerances.items():
        if abs(float(identity[metric].abs().max().item())) > tolerance:
            raise ValueError(f"identity oracle {metric} is not zero")
    return ReadoutEvaluation(
        records=records,
        logits=logits,
        probabilities=probabilities,
        point_metrics=point_metrics,
    )


def summarize_seed_evaluation(
    *,
    seed_name: str,
    points: PointSplits,
    oracle: OracleSplits,
    evaluation: ReadoutEvaluation,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
    max_pairs_per_bucket: int,
) -> SeedEvaluationSummary:
    """Build proper-score and direct posterior-JS evidence with hand uncertainty."""

    torch, _ = require_torch()
    metric_names = ("cross_entropy_nats", "kl_nats", "js_nats", "brier")
    hand_ids = tuple(str(point.hand_id) for point in points.test)
    point_values: dict[str, Any] = {}
    for variant, metrics in evaluation.point_metrics.items():
        for metric in metric_names:
            point_values[f"{variant}/{metric}"] = metrics[metric]
    for metric in metric_names:
        baseline = evaluation.point_metrics[PRIMARY_BASELINE][metric]
        candidate = evaluation.point_metrics[PRIMARY_CANDIDATE][metric]
        point_values[f"primary_improvement/{metric}"] = baseline - candidate
    hand_bootstrap = equal_hand_bootstrap(
        hand_ids,
        point_values,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    readout_rows: dict[str, dict[str, Any]] = {}
    for variant, record in evaluation.records.items():
        intervals = {
            metric: hand_bootstrap.intervals[f"{variant}/{metric}"]
            for metric in metric_names
        }
        ce_minus_entropy_minus_kl = (
            evaluation.point_metrics[variant]["cross_entropy_nats"]
            - evaluation.point_metrics[variant]["target_entropy_nats"]
            - evaluation.point_metrics[variant]["kl_nats"]
        )
        if float(ce_minus_entropy_minus_kl.abs().max().item()) > 1e-8:
            raise ValueError(f"CE=entropy+KL identity failed for {seed_name}/{variant}")
        readout_rows[variant] = {
            **record,
            "test_scores": intervals,
        }
    primary_improvements = {
        metric: hand_bootstrap.intervals[f"primary_improvement/{metric}"]
        for metric in metric_names
    }

    pairs = build_fixed_matched_pairs(
        points.test,
        max_pairs_per_bucket=max_pairs_per_bucket,
        seed=bootstrap_seed + 100000,
    )
    if not pairs:
        pair_summary = {
            "pair_count": 0,
            "bucket_count": 0,
            "nonzero_oracle_js_pair_count": 0,
            "readouts": {},
            "primary_mae_improvement": None,
        }
        return SeedEvaluationSummary(
            readout_rows=readout_rows,
            primary_improvements=primary_improvements,
            pair_summary=pair_summary,
            pair_rows=[],
            final_diagnostic_label="insufficient_support",
        )

    left_indices = torch.tensor([int(row["left_index"]) for row in pairs], dtype=torch.long)
    right_indices = torch.tensor([int(row["right_index"]) for row in pairs], dtype=torch.long)
    left_hand_ids = tuple(str(row["left_hand_id"]) for row in pairs)
    right_hand_ids = tuple(str(row["right_hand_id"]) for row in pairs)
    target = torch.from_numpy(np.array(oracle.test.posterior, copy=True)).float()
    target_js = batched_jensen_shannon(target[left_indices], target[right_indices])
    predicted_js: dict[str, Any] = {}
    pair_readouts: dict[str, Any] = {}
    for variant_index, variant in enumerate((*LEARNED_VARIANTS, *CONTROL_VARIANTS)):
        probabilities = evaluation.probabilities[variant]
        variant_js = batched_jensen_shannon(
            probabilities[left_indices],
            probabilities[right_indices],
        )
        predicted_js[variant] = variant_js
        pair_readouts[variant] = dyadic_alignment_bootstrap(
            target_js,
            variant_js,
            left_hand_ids,
            right_hand_ids,
            bootstrap_count=bootstrap_count,
            bootstrap_seed=bootstrap_seed + 200000 + variant_index,
            confidence_level=confidence_level,
        )
    primary_pair_improvement = paired_mae_improvement_bootstrap(
        target_js,
        predicted_js[PRIMARY_CANDIDATE],
        predicted_js[PRIMARY_BASELINE],
        left_hand_ids,
        right_hand_ids,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed + 300000,
        confidence_level=confidence_level,
    )
    pair_rows = []
    target_values = target_js.tolist()
    predicted_values = {variant: values.tolist() for variant, values in predicted_js.items()}
    for index, source in enumerate(pairs):
        pair_rows.append(
            {
                **source,
                "oracle_js_nats": float(target_values[index]),
                "predicted_js_nats": {
                    variant: float(values[index])
                    for variant, values in predicted_values.items()
                },
            }
        )
    bucket_count = len({str(row["composition_key"]) for row in pairs})
    nonzero_pair_count = sum(1 for value in target_values if float(value) > 1e-12)
    pair_summary = {
        "pair_count": len(pairs),
        "bucket_count": bucket_count,
        "nonzero_oracle_js_pair_count": nonzero_pair_count,
        "max_pairs_per_bucket": max_pairs_per_bucket,
        "pair_rule": "fixed_predecision_composition_cross_hand",
        "readouts": pair_readouts,
        "primary_mae_improvement": primary_pair_improvement,
    }

    identity_pair = pair_readouts["identity_oracle"]["point"]
    if (
        float(identity_pair["mae"]) > 1e-10
        or identity_pair["pearson"] is None
        or abs(float(identity_pair["pearson"]) - 1.0) > 1e-9
        or identity_pair["slope"] is None
        or abs(float(identity_pair["slope"]) - 1.0) > 1e-9
    ):
        raise ValueError("identity oracle failed matched-posterior calibration")

    support_ok = bucket_count >= 100 and nonzero_pair_count >= 2000
    proper_positive = all(
        float(primary_improvements[metric]["lower"]) > 0.0
        for metric in ("kl_nats", "js_nats", "brier")
    )
    candidate_alignment = pair_readouts[PRIMARY_CANDIDATE]["intervals"]
    pair_positive = (
        float(primary_pair_improvement["mae_improvement_ci"]["lower"]) > 0.0
        and candidate_alignment["pearson"]["lower"] is not None
        and float(candidate_alignment["pearson"]["lower"]) > 0.0
        and candidate_alignment["slope"]["lower"] is not None
        and float(candidate_alignment["slope"]["lower"]) > 0.0
    )
    any_positive = any(
        float(primary_improvements[metric]["lower"]) > 0.0
        for metric in ("kl_nats", "js_nats", "brier")
    ) or float(primary_pair_improvement["mae_improvement_ci"]["lower"]) > 0.0
    if not support_ok:
        label = "insufficient_support"
    elif proper_positive and pair_positive:
        label = "exact_posterior_increment_supported"
    elif any_positive:
        label = "mixed_exact_posterior_evidence"
    else:
        label = "exact_posterior_increment_not_supported"
    return SeedEvaluationSummary(
        readout_rows=readout_rows,
        primary_improvements=primary_improvements,
        pair_summary=pair_summary,
        pair_rows=pair_rows,
        final_diagnostic_label=label,
    )


def _point_metric_rows(
    *,
    points: tuple[Any, ...],
    oracle: LimitPosteriorOracleResult,
    evaluation: ReadoutEvaluation,
):
    variants = (*LEARNED_VARIANTS, *CONTROL_VARIANTS)
    metric_names = ("cross_entropy_nats", "kl_nats", "js_nats", "brier")
    values = {
        variant: {
            metric: evaluation.point_metrics[variant][metric].detach().cpu().tolist()
            for metric in metric_names
        }
        for variant in variants
    }
    entropies = (
        evaluation.point_metrics["identity_oracle"]["target_entropy_nats"]
        .detach()
        .cpu()
        .tolist()
    )
    for index, point in enumerate(points):
        yield {
            "point_id": point_id(point),
            "hand_id": str(point.hand_id),
            "step_index": int(point.step_index),
            "acting_player": int(point.acting_player),
            "target_entropy_nats": float(entropies[index]),
            "posterior_combo_support": int(oracle.support_combo_counts[index]),
            "card_prior_combo_support": int(oracle.card_prior_combo_counts[index]),
            "readouts": {
                variant: {
                    metric: float(values[variant][metric][index])
                    for metric in metric_names
                }
                for variant in variants
            },
        }


def _seed_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# k.9.0 Exact Posterior Readout - {payload['seed']}",
        "",
        f"- final diagnostic label: `{payload['final_diagnostic_label']}`",
        f"- locked layer: `{payload['locked_layer']}`",
        f"- conditioning: `{payload['conditioning_id']}`",
        f"- test points: {payload['split_counts']['test_points']}",
        "",
        "## Proper Scores",
        "",
        "| readout | CE nats | KL nats | JS nats | Brier |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in (*LEARNED_VARIANTS, *CONTROL_VARIANTS):
        scores = payload["readouts"][variant]["test_scores"]
        lines.append(
            f"| {variant} | {scores['cross_entropy_nats']['mean']:.6f} | "
            f"{scores['kl_nats']['mean']:.6f} | {scores['js_nats']['mean']:.6f} | "
            f"{scores['brier']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Primary Nested Improvement",
            "",
            f"`{PRIMARY_CANDIDATE}` over `{PRIMARY_BASELINE}`; positive favors the nested hidden row.",
            "",
            "| metric | mean | 95% hand CI |",
            "|---|---:|---|",
        ]
    )
    for metric, interval in payload["primary_improvements"].items():
        lines.append(
            f"| {metric} | {interval['mean']:.6f} | "
            f"[{interval['lower']:.6f}, {interval['upper']:.6f}] |"
        )
    pair = payload["matched_posterior"]
    lines.extend(
        [
            "",
            "## Matched Posterior JS",
            "",
            f"- buckets: {pair['bucket_count']}",
            f"- cross-hand pairs: {pair['pair_count']}",
            f"- nonzero oracle-JS pairs: {pair['nonzero_oracle_js_pair_count']}",
        ]
    )
    if pair["primary_mae_improvement"] is not None:
        ci = pair["primary_mae_improvement"]["mae_improvement_ci"]
        lines.append(
            f"- nested-vs-A/V+C JS-MAE improvement: {ci['mean']:.6f} "
            f"[{ci['lower']:.6f}, {ci['upper']:.6f}]"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This is exact only for the frozen deterministic teacher and declared player information set.",
            "It does not establish a causal or general Bayesian belief mechanism.",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_rows(root: Path, *, exclude_names: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = exclude_names or set()
    rows = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path.name in excluded:
            continue
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def build_single_seed_exact_posterior_report(
    *,
    seed_name: str,
    archive_root: Path,
    output_dir: Path,
    source_revision: str,
    device: str = "cpu",
    extraction_batch_size: int = 512,
    readout_batch_size: int = 512,
    epochs: int = 30,
    patience: int = 5,
    learning_rate: float = 0.01,
    residual_learning_rate: float = 0.002,
    weight_decay: float = 1e-4,
    bootstrap_count: int = 10000,
    bootstrap_seed: int = 20260901,
    confidence_level: float = 0.95,
    max_pairs_per_bucket: int = 64,
    max_train_hands: int | None = None,
    max_validation_hands: int | None = None,
    max_test_hands: int | None = None,
) -> dict[str, Any]:
    """Build one complete exact-posterior product row from a frozen checkpoint."""

    if seed_name not in FIVE_RUN_SEEDS:
        raise ValueError(f"unknown five-run seed: {seed_name}")
    train_config_path = (
        archive_root
        / "configs"
        / "train"
        / f"limit_ar_action_value_only_{seed_name}_50k_4layer.json"
    )
    checkpoint_path = (
        archive_root
        / "outputs"
        / f"limit_ar_action_value_only_{seed_name}_50k_4layer"
        / "model.pt"
    )
    train_config, points = load_frozen_point_splits(
        train_config_path,
        max_train_hands=max_train_hands,
        max_validation_hands=max_validation_hands,
        max_test_hands=max_test_hands,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    oracle = load_or_compute_oracle_splits(
        points=points,
        cache_dir=output_dir / "posterior_cache",
    )
    hidden = extract_full_input_locked_hidden(
        checkpoint_path=checkpoint_path,
        train_config=train_config,
        points=points,
        device=device,
        batch_size=extraction_batch_size,
    )
    if hidden.locked_layer != LOCKED_LAYER:
        raise ValueError(f"locked layer changed: {hidden.locked_layer}")
    feature_bundle = build_readout_feature_bundle(points=points, hidden=hidden)
    evaluation = fit_and_evaluate_readouts(
        seed_name=seed_name,
        features=feature_bundle,
        oracle=oracle,
        output_dir=output_dir,
        device=device,
        epochs=epochs,
        patience=patience,
        batch_size=readout_batch_size,
        learning_rate=learning_rate,
        residual_learning_rate=residual_learning_rate,
        weight_decay=weight_decay,
    )
    summary = summarize_seed_evaluation(
        seed_name=seed_name,
        points=points,
        oracle=oracle,
        evaluation=evaluation,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
        max_pairs_per_bucket=max_pairs_per_bucket,
    )
    point_metrics_path = output_dir / "posterior_point_metrics.jsonl.gz"
    pair_path = output_dir / "matched_posterior_pairs.jsonl.gz"
    write_jsonl_gzip(
        point_metrics_path,
        _point_metric_rows(points=points.test, oracle=oracle.test, evaluation=evaluation),
    )
    write_jsonl_gzip(pair_path, summary.pair_rows)
    payload: dict[str, Any] = {
        "created_by": "camera_ready_snapshot",
        "version": "k.9.0",
        "seed": seed_name,
        "source_revision": source_revision,
        "conditioning_id": CONDITIONING_ID,
        "posterior_conditioning_contract": oracle.test.conditioning_contract,
        "representation_conditioning_id": CONDITIONING_ID,
        "train_config_path": str(train_config_path.relative_to(archive_root)),
        "train_config_sha256": _sha256(train_config_path),
        "checkpoint_path": str(checkpoint_path.relative_to(archive_root)),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "locked_layer": hidden.locked_layer,
        "checkpoint_load_info": hidden.checkpoint_load_info,
        "split_counts": {
            "train_points": len(points.train),
            "validation_points": len(points.validation),
            "test_points": len(points.test),
            "train_hands": len({point.hand_id for point in points.train}),
            "validation_hands": len({point.hand_id for point in points.validation}),
            "test_hands": len({point.hand_id for point in points.test}),
        },
        "posterior_cache_manifests": oracle.manifests,
        "readout_family": "card_prior_offset_linear_169",
        "readout_hyperparameters": {
            "epochs": epochs,
            "patience": patience,
            "readout_batch_size": readout_batch_size,
            "learning_rate": learning_rate,
            "residual_learning_rate": residual_learning_rate,
            "weight_decay": weight_decay,
        },
        "readouts": summary.readout_rows,
        "primary_comparison": {
            "baseline": PRIMARY_BASELINE,
            "candidate": PRIMARY_CANDIDATE,
            "sign": "positive_favors_candidate",
        },
        "primary_improvements": summary.primary_improvements,
        "matched_posterior": summary.pair_summary,
        "bootstrap": {
            "count": bootstrap_count,
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
            "point_unit": "equal_hand_mean",
            "pair_unit": "dyadic_hand_node",
        },
        "point_metrics_path": point_metrics_path.name,
        "point_metrics_sha256": _sha256(point_metrics_path),
        "matched_pairs_path": pair_path.name,
        "matched_pairs_sha256": _sha256(pair_path),
        "final_diagnostic_label": summary.final_diagnostic_label,
        "allowed_claim": (
            "known-teacher exact posterior increment under specified controls"
            if summary.final_diagnostic_label == "exact_posterior_increment_supported"
            else "report metric-specific or unsupported exact-posterior increment without absence claims"
        ),
        "forbidden_claims": [
            "general Bayesian belief state",
            "causal posterior mechanism",
            "unknown-opponent posterior tracking",
        ],
    }
    payload["checksum"] = checksum_payload(payload)
    report_json = output_dir / "posterior_readout_metrics.json"
    report_md = output_dir / "posterior_readout_metrics.md"
    write_json_artifact(report_json, payload)
    report_md.write_text(_seed_markdown(payload), encoding="utf-8")
    manifest_name = "manifest.json"
    artifact_rows = _artifact_rows(output_dir, exclude_names={manifest_name})
    manifest = {
        "created_by": "camera_ready_snapshot",
        "version": "k.9.0",
        "status": "complete",
        "seed": seed_name,
        "result_checksum": payload["checksum"],
        "artifacts": artifact_rows,
        "artifact_count": len(artifact_rows),
        "artifact_bytes": sum(int(row["bytes"]) for row in artifact_rows),
        "bundle_sha256": checksum_payload({"artifacts": artifact_rows}),
    }
    manifest["checksum"] = checksum_payload(manifest)
    write_json_artifact(output_dir / manifest_name, manifest)
    return payload


def _verify_completed_seed(output_dir: Path) -> bool:
    report_path = output_dir / "posterior_readout_metrics.json"
    manifest_path = output_dir / "manifest.json"
    if not report_path.is_file() or not manifest_path.is_file():
        return False
    manifest = read_json_object(manifest_path)
    saved_checksum = str(manifest.get("checksum", ""))
    unsigned = {key: value for key, value in manifest.items() if key != "checksum"}
    if checksum_payload(unsigned) != saved_checksum or manifest.get("status") != "complete":
        return False
    for row in manifest.get("artifacts", []):
        path = output_dir / str(row["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or _sha256(path) != str(row["sha256"])
        ):
            return False
    report = read_json_object(report_path)
    return report.get("checksum") == manifest.get("result_checksum")


def _aggregate_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# k.9.0 Five-Run Exact Posterior Audit",
        "",
        f"- final diagnostic label: `{payload['final_diagnostic_label']}`",
        f"- runs: {', '.join(payload['seed_names'])}",
        "",
        "## Primary Nested Proper-Score Improvement",
        "",
        "Positive means A/V+composition+residual hidden improves over A/V+composition.",
        "",
        "| metric | five-run mean | sample SD | 95% t interval |",
        "|---|---:|---:|---|",
    ]
    for metric, row in payload["five_run_primary_improvements"].items():
        lines.append(
            f"| {metric} | {row['mean']:.6f} | {row['sample_standard_deviation']:.6f} | "
            f"[{row['lower']:.6f}, {row['upper']:.6f}] |"
        )
    pair = payload["five_run_pair_mae_improvement"]
    lines.extend(
        [
            "",
            "## Matched Posterior JS",
            "",
            f"Five-run JS-MAE improvement: {pair['mean']:.6f}, "
            f"95% t interval [{pair['lower']:.6f}, {pair['upper']:.6f}].",
            "",
            "## Per-Run Labels",
            "",
            "| run | label | KL improvement | Brier improvement | pair-MAE improvement |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in payload["runs"]:
        lines.append(
            f"| {row['seed']} | {row['final_diagnostic_label']} | "
            f"{row['primary_improvements']['kl_nats']['mean']:.6f} | "
            f"{row['primary_improvements']['brier']['mean']:.6f} | "
            f"{row['matched_posterior']['primary_mae_improvement']['mae_improvement']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            payload["allowed_claim"],
            "",
            "No result in this product licenses a general or causal belief-mechanism claim.",
            "",
        ]
    )
    return "\n".join(lines)


def build_five_run_exact_posterior_report(
    *,
    archive_root: Path,
    output_dir: Path,
    source_revision: str,
    seed_names: tuple[str, ...] = FIVE_RUN_SEEDS,
    reuse_complete: bool = True,
    **single_seed_kwargs,
) -> dict[str, Any]:
    """Build or resume all five exact-posterior rows and aggregate run uncertainty."""

    if seed_names != FIVE_RUN_SEEDS:
        raise ValueError(f"k.9.0 requires exact seed order {FIVE_RUN_SEEDS}")
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed_index, seed_name in enumerate(seed_names):
        seed_dir = output_dir / seed_name
        if reuse_complete and _verify_completed_seed(seed_dir):
            payload = read_json_object(seed_dir / "posterior_readout_metrics.json")
        else:
            payload = build_single_seed_exact_posterior_report(
                seed_name=seed_name,
                archive_root=archive_root,
                output_dir=seed_dir,
                source_revision=source_revision,
                bootstrap_seed=int(single_seed_kwargs.get("bootstrap_seed", 20260901)) + seed_index * 1000,
                **{
                    key: value
                    for key, value in single_seed_kwargs.items()
                    if key != "bootstrap_seed"
                },
            )
        runs.append(payload)

    proper_metrics = ("cross_entropy_nats", "kl_nats", "js_nats", "brier")
    five_run_primary = {
        metric: five_run_statistics(
            [float(run["primary_improvements"][metric]["mean"]) for run in runs]
        )
        for metric in proper_metrics
    }
    pair_values = [
        float(run["matched_posterior"]["primary_mae_improvement"]["mae_improvement"])
        for run in runs
    ]
    five_run_pair = five_run_statistics(pair_values)
    run_labels = [str(run["final_diagnostic_label"]) for run in runs]
    positive_count = run_labels.count("exact_posterior_increment_supported")
    insufficient_count = run_labels.count("insufficient_support")
    reverse_run = any(
        any(
            float(run["primary_improvements"][metric]["upper"]) < 0.0
            for metric in ("kl_nats", "js_nats", "brier")
        )
        or float(
            run["matched_posterior"]["primary_mae_improvement"]["mae_improvement_ci"]["upper"]
        )
        < 0.0
        for run in runs
    )
    aggregate_positive = all(
        float(five_run_primary[metric]["lower"]) > 0.0
        for metric in ("kl_nats", "js_nats", "brier")
    ) and float(five_run_pair["lower"]) > 0.0
    if insufficient_count:
        final_label = "insufficient_support"
    elif positive_count >= 4 and not reverse_run and aggregate_positive:
        final_label = "exact_posterior_increment_supported"
    elif any(
        label in {"exact_posterior_increment_supported", "mixed_exact_posterior_evidence"}
        for label in run_labels
    ):
        final_label = "mixed_exact_posterior_evidence"
    else:
        final_label = "exact_posterior_increment_not_supported"

    payload: dict[str, Any] = {
        "created_by": "camera_ready_snapshot",
        "version": "k.9.0",
        "source_revision": source_revision,
        "seed_names": list(seed_names),
        "seed_count": len(seed_names),
        "conditioning_id": CONDITIONING_ID,
        "locked_layer": LOCKED_LAYER,
        "primary_baseline": PRIMARY_BASELINE,
        "primary_candidate": PRIMARY_CANDIDATE,
        "runs": runs,
        "five_run_primary_improvements": five_run_primary,
        "five_run_pair_mae_improvement": five_run_pair,
        "positive_run_count": positive_count,
        "insufficient_run_count": insufficient_count,
        "significantly_reversed_run_present": reverse_run,
        "final_diagnostic_label": final_label,
        "allowed_claim": (
            "Under the frozen deterministic teacher, exact posterior information is linearly recoverable "
            "beyond the declared card-aware A/V and composition controls."
            if final_label == "exact_posterior_increment_supported"
            else "The tested exact-posterior increment is mixed or unsupported; report every metric and run."
        ),
        "forbidden_claims": [
            "general Bayesian belief state",
            "causal posterior mechanism",
            "transfer to unknown opponents",
        ],
    }
    payload["checksum"] = checksum_payload(payload)
    report_json = output_dir / "k.9.0_exact_posterior_readout_report.json"
    report_md = output_dir / "k.9.0_exact_posterior_readout_report.md"
    write_json_artifact(report_json, payload)
    report_md.write_text(_aggregate_markdown(payload), encoding="utf-8")
    manifest_name = "k.9.0_exact_posterior_readout_manifest.json"
    artifact_rows = _artifact_rows(output_dir, exclude_names={manifest_name})
    manifest = {
        "created_by": "camera_ready_snapshot",
        "version": "k.9.0",
        "status": "complete",
        "result_checksum": payload["checksum"],
        "artifacts": artifact_rows,
        "artifact_count": len(artifact_rows),
        "artifact_bytes": sum(int(row["bytes"]) for row in artifact_rows),
        "bundle_sha256": checksum_payload({"artifacts": artifact_rows}),
    }
    manifest["checksum"] = checksum_payload(manifest)
    write_json_artifact(output_dir / manifest_name, manifest)
    return payload


def _validate_payload_checksum(payload: dict[str, Any], *, role: str) -> None:
    saved = str(payload.get("checksum", ""))
    unsigned = {key: value for key, value in payload.items() if key != "checksum"}
    if not saved or checksum_payload(unsigned) != saved:
        raise ValueError(f"{role} checksum mismatch")


def _repaired_run_label(run: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    matched = run["matched_posterior"]
    candidate_alignment = matched["readouts"][PRIMARY_CANDIDATE]
    support = {
        "bucket_count": int(matched["bucket_count"]),
        "pair_count": int(matched["pair_count"]),
        "nonzero_oracle_js_pair_count": int(matched["nonzero_oracle_js_pair_count"]),
        "hand_node_count": int(candidate_alignment["hand_count"]),
    }
    support["passes"] = (
        support["bucket_count"] >= REPAIRED_SUPPORT_MIN_BUCKETS
        and support["nonzero_oracle_js_pair_count"] >= REPAIRED_SUPPORT_MIN_NONZERO_PAIRS
        and support["hand_node_count"] >= REPAIRED_SUPPORT_MIN_HAND_NODES
    )
    proper_positive = all(
        float(run["primary_improvements"][metric]["lower"]) > 0.0
        for metric in ("kl_nats", "js_nats", "brier")
    )
    pair_mae_ci = matched["primary_mae_improvement"]["mae_improvement_ci"]
    intervals = candidate_alignment["intervals"]
    pair_positive = (
        float(pair_mae_ci["lower"]) > 0.0
        and intervals["pearson"]["lower"] is not None
        and float(intervals["pearson"]["lower"]) > 0.0
        and intervals["slope"]["lower"] is not None
        and float(intervals["slope"]["lower"]) > 0.0
    )
    any_positive = proper_positive or float(pair_mae_ci["lower"]) > 0.0
    if not support["passes"]:
        label = "insufficient_support"
    elif proper_positive and pair_positive:
        label = "exact_posterior_increment_supported"
    elif any_positive:
        label = "mixed_exact_posterior_evidence"
    else:
        label = "exact_posterior_increment_not_supported"
    return label, {
        "support": support,
        "proper_score_increment_positive": proper_positive,
        "pair_alignment_increment_positive": pair_positive,
    }


def build_k901_support_repair(
    *,
    source_report_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Repair only the impossible k.9.0 matched-support feasibility contract."""

    source = read_json_object(source_report_path)
    _validate_payload_checksum(source, role="source aggregate")
    if tuple(source.get("seed_names", ())) != FIVE_RUN_SEEDS:
        raise ValueError("support repair requires the frozen five-run source panel")
    repaired_runs = []
    for run in source["runs"]:
        _validate_payload_checksum(run, role=f"source run {run.get('seed')}")
        repaired_label, diagnostics = _repaired_run_label(run)
        repaired_runs.append(
            {
                "seed": run["seed"],
                "source_result_checksum": run["checksum"],
                "source_label": run["final_diagnostic_label"],
                "repaired_label": repaired_label,
                **diagnostics,
            }
        )
    if any(not bool(row["support"]["passes"]) for row in repaired_runs):
        final_label = "insufficient_support"
    else:
        positive_count = sum(
            row["repaired_label"] == "exact_posterior_increment_supported"
            for row in repaired_runs
        )
        reverse_run = bool(source["significantly_reversed_run_present"])
        aggregate_positive = all(
            float(source["five_run_primary_improvements"][metric]["lower"]) > 0.0
            for metric in ("kl_nats", "js_nats", "brier")
        ) and float(source["five_run_pair_mae_improvement"]["lower"]) > 0.0
        if positive_count >= 4 and not reverse_run and aggregate_positive:
            final_label = "exact_posterior_increment_supported"
        elif any(
            row["repaired_label"] in {
                "exact_posterior_increment_supported",
                "mixed_exact_posterior_evidence",
            }
            for row in repaired_runs
        ):
            final_label = "mixed_exact_posterior_evidence"
        else:
            final_label = "exact_posterior_increment_not_supported"
    def alignment_across_runs(variant: str) -> dict[str, Any]:
        return {
            metric: five_run_statistics(
                [
                    float(run["matched_posterior"]["readouts"][variant]["point"][metric])
                    for run in source["runs"]
                ]
            )
            for metric in ("mae", "rmse", "bias", "pearson", "spearman", "slope", "intercept")
        }

    payload: dict[str, Any] = {
        "created_by": "camera_ready_snapshot",
        "version": "k.9.0.1",
        "repair_type": "matched_support_contract_only",
        "source_report_path": source_report_path.name,
        "source_report_sha256": _sha256(source_report_path),
        "source_result_checksum": source["checksum"],
        "old_feasibility_contract": {
            "minimum_bucket_count": 100,
            "minimum_nonzero_oracle_js_pairs": 2000,
            "max_pairs_per_bucket": 64,
            "structurally_reachable": False,
        },
        "repaired_feasibility_contract": {
            "minimum_bucket_count": REPAIRED_SUPPORT_MIN_BUCKETS,
            "minimum_nonzero_oracle_js_pairs": REPAIRED_SUPPORT_MIN_NONZERO_PAIRS,
            "minimum_hand_nodes": REPAIRED_SUPPORT_MIN_HAND_NODES,
            "max_pairs_per_bucket": 64,
        },
        "immutable_quantities": [
            "posterior_targets",
            "readout_weights",
            "proper_scores",
            "pair_table",
            "oracle_and_predicted_js",
            "bootstrap_draws_and_intervals",
        ],
        "runs": repaired_runs,
        "five_run_primary_improvements": source["five_run_primary_improvements"],
        "five_run_pair_mae_improvement": source["five_run_pair_mae_improvement"],
        "five_run_candidate_pair_alignment": alignment_across_runs(PRIMARY_CANDIDATE),
        "five_run_baseline_pair_alignment": alignment_across_runs(PRIMARY_BASELINE),
        "final_diagnostic_label": final_label,
        "allowed_claim": (
            "Exact known-teacher posterior evidence is metric-specific: proper scores improve beyond "
            "A/V+composition, but calibrated matched-pair JS improvement is not established."
            if final_label == "mixed_exact_posterior_evidence"
            else source["allowed_claim"]
        ),
        "forbidden_claims": source["forbidden_claims"],
    }
    payload["checksum"] = checksum_payload(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "k.9.0.1_exact_posterior_support_repair.json"
    md_path = output_dir / "k.9.0.1_exact_posterior_support_repair.md"
    write_json_artifact(json_path, payload)
    md_lines = [
        "# k.9.0.1 Exact Posterior Support Repair",
        "",
        f"- source checksum: `{source['checksum']}`",
        f"- repaired final label: `{final_label}`",
        "",
        "| run | buckets | pairs | nonzero JS pairs | hand nodes | repaired label |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in repaired_runs:
        support = row["support"]
        md_lines.append(
            f"| {row['seed']} | {support['bucket_count']} | {support['pair_count']} | "
            f"{support['nonzero_oracle_js_pair_count']} | {support['hand_node_count']} | "
            f"{row['repaired_label']} |"
        )
    md_lines.extend(["", payload["allowed_claim"], ""])
    candidate = payload["five_run_candidate_pair_alignment"]
    md_lines.extend(
        [
            (
                f"Candidate pair Pearson mean `{candidate['pearson']['mean']:.6f}` "
                f"(95% t interval [{candidate['pearson']['lower']:.6f}, "
                f"{candidate['pearson']['upper']:.6f}])."
            ),
            (
                f"Candidate pair slope mean `{candidate['slope']['mean']:.6f}` "
                f"(95% t interval [{candidate['slope']['lower']:.6f}, "
                f"{candidate['slope']['upper']:.6f}])."
            ),
            "",
        ]
    )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    artifacts = _artifact_rows(output_dir, exclude_names={
        "k.9.0.1_exact_posterior_support_repair_manifest.json"
    })
    manifest = {
        "created_by": "camera_ready_snapshot",
        "version": "k.9.0.1",
        "status": "complete",
        "result_checksum": payload["checksum"],
        "artifacts": artifacts,
        "bundle_sha256": checksum_payload({"artifacts": artifacts}),
    }
    manifest["checksum"] = checksum_payload(manifest)
    write_json_artifact(
        output_dir / "k.9.0.1_exact_posterior_support_repair_manifest.json",
        manifest,
    )
    return payload


def write_jsonl_gzip(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write deterministic compressed JSONL product rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a required JSON object."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data
