"""Soft-target linear readouts and uncertainty summaries for poker posteriors.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Mapping, Sequence

from jorjor.models.tiny import require_torch


STUDENT_T_975_DF4 = 2.7764451051977987


def five_run_statistics(values: Sequence[float], *, confidence_level: float = 0.95) -> dict[str, object]:
    """Return the equal-run two-sided Student interval for exactly five runs."""

    numeric = [float(value) for value in values]
    if len(numeric) != 5:
        raise ValueError(f"five-run statistics require exactly 5 values, got {len(numeric)}")
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("five-run values must be finite")
    if confidence_level != 0.95:
        raise ValueError("five-run statistics currently support only a 0.95 confidence level")
    mean = statistics.mean(numeric)
    sample_standard_deviation = statistics.stdev(numeric)
    standard_error = sample_standard_deviation / math.sqrt(5.0)
    margin = STUDENT_T_975_DF4 * standard_error
    return {
        "resampling_unit": "independent_end_to_end_run",
        "run_count": 5,
        "degrees_of_freedom": 4,
        "confidence_level": confidence_level,
        "t_critical": STUDENT_T_975_DF4,
        "mean": mean,
        "sample_standard_deviation": sample_standard_deviation,
        "standard_error": standard_error,
        "lower": mean - margin,
        "upper": mean + margin,
    }


@dataclass(frozen=True)
class FeatureStandardizer:
    """Train-fitted column standardizer with explicit constant columns."""

    mean: object
    scale: object
    constant_mask: object

    def transform(self, features):
        """Apply the frozen train statistics to an equal-width feature tensor."""

        if features.ndim != 2 or features.shape[1] != self.mean.shape[1]:
            raise ValueError("features must be rank two with the fitted width")
        return (features - self.mean) / self.scale


def fit_standardizer(train_features, *, minimum_scale: float = 1e-6) -> FeatureStandardizer:
    """Fit column means and population scales using training rows only."""

    torch, _ = require_torch()
    if train_features.ndim != 2 or train_features.shape[0] == 0 or train_features.shape[1] == 0:
        raise ValueError("train_features must be a non-empty rank-2 tensor")
    if minimum_scale <= 0.0:
        raise ValueError("minimum_scale must be positive")
    if not bool(torch.isfinite(train_features).all()):
        raise ValueError("train_features must be finite")
    mean = train_features.mean(dim=0, keepdim=True)
    raw_scale = train_features.std(dim=0, unbiased=False, keepdim=True)
    constant_mask = raw_scale.squeeze(0) < minimum_scale
    scale = torch.where(raw_scale < minimum_scale, torch.ones_like(raw_scale), raw_scale)
    return FeatureStandardizer(mean=mean.detach(), scale=scale.detach(), constant_mask=constant_mask.detach())


@dataclass(frozen=True)
class OLSResidualizer:
    """Train-fitted affine nuisance projection."""

    coefficients: object

    def transform(self, features, nuisance):
        """Remove the nuisance prediction using the frozen train coefficients."""

        torch, _ = require_torch()
        if features.ndim != 2 or nuisance.ndim != 2 or features.shape[0] != nuisance.shape[0]:
            raise ValueError("features and nuisance must be aligned rank-2 tensors")
        if features.shape[1] != self.coefficients.shape[1]:
            raise ValueError("feature width does not match the fitted residualizer")
        if nuisance.shape[1] + 1 != self.coefficients.shape[0]:
            raise ValueError("nuisance width does not match the fitted residualizer")
        ones = torch.ones((nuisance.shape[0], 1), dtype=nuisance.dtype, device=nuisance.device)
        design = torch.cat([ones, nuisance], dim=1)
        return features - design @ self.coefficients


@dataclass(frozen=True)
class ResidualizedFeatureSplit:
    """Train and test residuals produced by one train-fitted projection."""

    train: object
    test: object
    residualizer: OLSResidualizer


def fit_ols_residualizer(train_features, train_nuisance) -> OLSResidualizer:
    """Fit an affine least-squares nuisance prediction on training rows."""

    torch, _ = require_torch()
    if train_features.ndim != 2 or train_nuisance.ndim != 2:
        raise ValueError("train_features and train_nuisance must be rank-2 tensors")
    if train_features.shape[0] == 0 or train_features.shape[0] != train_nuisance.shape[0]:
        raise ValueError("training features and nuisance must have aligned non-empty rows")
    if train_features.device != train_nuisance.device or train_features.dtype != train_nuisance.dtype:
        raise ValueError("training features and nuisance must share dtype and device")
    if not bool(torch.isfinite(train_features).all()) or not bool(torch.isfinite(train_nuisance).all()):
        raise ValueError("training features and nuisance must be finite")
    ones = torch.ones(
        (train_nuisance.shape[0], 1),
        dtype=train_nuisance.dtype,
        device=train_nuisance.device,
    )
    design = torch.cat([ones, train_nuisance], dim=1)
    coefficients = torch.linalg.pinv(design) @ train_features
    return OLSResidualizer(coefficients=coefficients.detach())


def residualize_train_test(
    train_features,
    test_features,
    train_nuisance,
    test_nuisance,
) -> ResidualizedFeatureSplit:
    """Fit nuisance removal on train and apply it unchanged to train and test."""

    residualizer = fit_ols_residualizer(train_features, train_nuisance)
    return ResidualizedFeatureSplit(
        train=residualizer.transform(train_features, train_nuisance),
        test=residualizer.transform(test_features, test_nuisance),
        residualizer=residualizer,
    )


def card_prior_offset_logits(readout, features, card_log_prior):
    """Add a fixed per-point card log prior to learned linear logits.

    ``-inf`` prior entries define structural zeros and remain impossible after
    the learned offset is applied.
    """

    torch, _ = require_torch()
    if features.ndim != 2 or card_log_prior.ndim != 2:
        raise ValueError("features and card_log_prior must be rank-2 tensors")
    if features.shape[0] != card_log_prior.shape[0]:
        raise ValueError("features and card_log_prior must have aligned rows")
    if bool(torch.isnan(card_log_prior).any()) or bool(torch.isposinf(card_log_prior).any()):
        raise ValueError("card_log_prior must not contain NaN or positive infinity")
    support = torch.isfinite(card_log_prior)
    if bool((~support.any(dim=1)).any()):
        raise ValueError("every card prior row must support at least one class")
    learned_logits = readout(features)
    if learned_logits.shape != card_log_prior.shape:
        raise ValueError("readout output width must match the card prior")
    logits = learned_logits + card_log_prior
    return logits.masked_fill(~support, -torch.inf)


def card_prior_offset_probabilities(readout, features, card_log_prior):
    """Return normalized probabilities from a card-prior-offset readout."""

    torch, _ = require_torch()
    return torch.softmax(card_prior_offset_logits(readout, features, card_log_prior), dim=1)


@dataclass(frozen=True)
class SoftPosteriorFitResult:
    """Best-validation state and deterministic training history for one readout."""

    model: object
    best_epoch: int
    epochs_completed: int
    best_validation_cross_entropy_nats: float
    train_cross_entropy_nats: tuple[float, ...]
    validation_cross_entropy_nats: tuple[float, ...]
    patience: int
    weight_decay: float


@dataclass(frozen=True)
class HandBootstrapResult:
    """Equal-hand intervals and shared bootstrap draws for named metrics."""

    point_count: int
    hand_count: int
    bootstrap_count: int
    bootstrap_seed: int
    confidence_level: float
    intervals: dict[str, dict[str, float]]
    bootstrap_means: dict[str, tuple[float, ...]]
    resampling_unit: str = "hand_id"
    estimand: str = "equal_hand_mean"


def equal_hand_bootstrap(
    hand_ids: Sequence[str],
    point_values: Mapping[str, object],
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float = 0.95,
    draw_chunk_size: int = 32,
) -> HandBootstrapResult:
    """Bootstrap equal-weight hand means with shared draws for every metric.

    Callers can include paired-difference columns in ``point_values``; because
    all columns use the same resampled hand indices, their bootstrap draws stay
    paired exactly.
    """

    torch, _ = require_torch()
    if not hand_ids:
        raise ValueError("hand_ids must not be empty")
    if not point_values:
        raise ValueError("point_values must contain at least one metric")
    if bootstrap_count <= 0:
        raise ValueError("bootstrap_count must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if draw_chunk_size <= 0:
        raise ValueError("draw_chunk_size must be positive")

    point_count = len(hand_ids)
    metric_names = tuple(point_values)
    columns = []
    for name in metric_names:
        column = torch.as_tensor(point_values[name], dtype=torch.float64).detach().cpu()
        if column.ndim != 1 or column.shape[0] != point_count:
            raise ValueError(f"point metric {name} must have one finite value per point")
        if not bool(torch.isfinite(column).all()):
            raise ValueError(f"point metric {name} must be finite")
        columns.append(column)
    point_matrix = torch.stack(columns, dim=1)

    unique_hands = sorted({str(hand_id) for hand_id in hand_ids})
    hand_to_index = {hand_id: index for index, hand_id in enumerate(unique_hands)}
    point_hand_indices = torch.tensor(
        [hand_to_index[str(hand_id)] for hand_id in hand_ids],
        dtype=torch.long,
    )
    hand_count = len(unique_hands)
    hand_sums = torch.zeros((hand_count, len(metric_names)), dtype=torch.float64)
    hand_sums.index_add_(0, point_hand_indices, point_matrix)
    hand_sizes = torch.zeros(hand_count, dtype=torch.float64)
    hand_sizes.index_add_(0, point_hand_indices, torch.ones(point_count, dtype=torch.float64))
    hand_means = hand_sums / hand_sizes.unsqueeze(1)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(bootstrap_seed)
    draw_means = torch.empty((bootstrap_count, len(metric_names)), dtype=torch.float64)
    for start in range(0, bootstrap_count, draw_chunk_size):
        stop = min(start + draw_chunk_size, bootstrap_count)
        sampled_indices = torch.randint(
            hand_count,
            (stop - start, hand_count),
            generator=generator,
        )
        draw_means[start:stop] = hand_means[sampled_indices].mean(dim=1)

    alpha = (1.0 - confidence_level) / 2.0
    lower_index = int(alpha * (bootstrap_count - 1))
    upper_index = int((1.0 - alpha) * (bootstrap_count - 1))
    point_means = hand_means.mean(dim=0)
    intervals = {}
    bootstrap_means = {}
    for index, name in enumerate(metric_names):
        sorted_draws = torch.sort(draw_means[:, index]).values
        intervals[name] = {
            "mean": float(point_means[index].item()),
            "lower": float(sorted_draws[lower_index].item()),
            "upper": float(sorted_draws[upper_index].item()),
        }
        bootstrap_means[name] = tuple(float(value) for value in draw_means[:, index].tolist())
    return HandBootstrapResult(
        point_count=point_count,
        hand_count=hand_count,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
        intervals=intervals,
        bootstrap_means=bootstrap_means,
    )


def _validated_readout_tensors(features, targets, card_log_prior, *, name: str, torch):
    if features.ndim != 2 or targets.ndim != 2 or card_log_prior.ndim != 2:
        raise ValueError(f"{name} features, targets, and card_log_prior must be rank-2 tensors")
    if features.shape[0] == 0:
        raise ValueError(f"{name} split must not be empty")
    if targets.shape != card_log_prior.shape or features.shape[0] != targets.shape[0]:
        raise ValueError(f"{name} features, targets, and card_log_prior must have aligned shapes")
    if not features.dtype.is_floating_point:
        raise ValueError(f"{name} features must use a floating dtype")
    if not bool(torch.isfinite(features).all()):
        raise ValueError(f"{name} features must be finite")
    if not bool(torch.isfinite(targets).all()) or bool((targets < 0.0).any()):
        raise ValueError(f"{name} posterior targets must be finite and non-negative")
    if not bool(
        torch.allclose(
            targets.sum(dim=1),
            torch.ones(targets.shape[0], dtype=targets.dtype, device=targets.device),
            atol=1e-6,
            rtol=1e-6,
        )
    ):
        raise ValueError(f"{name} posterior target rows must sum to one")
    if bool(torch.isnan(card_log_prior).any()) or bool(torch.isposinf(card_log_prior).any()):
        raise ValueError(f"{name} card_log_prior must not contain NaN or positive infinity")
    support = torch.isfinite(card_log_prior)
    if bool((~support.any(dim=1)).any()):
        raise ValueError(f"every {name} card prior row must support at least one class")
    if bool(((targets > 0.0) & ~support).any()):
        raise ValueError(f"{name} target has positive mass on a structural-zero card class")


def _soft_cross_entropy_rows(readout, features, targets, card_log_prior, torch):
    logits = card_prior_offset_logits(readout, features, card_log_prior)
    support = torch.isfinite(card_log_prior)
    log_probabilities = torch.log_softmax(logits, dim=1).masked_fill(~support, 0.0)
    return -(targets * log_probabilities).sum(dim=1)


def _mean_soft_cross_entropy(
    readout,
    features,
    targets,
    card_log_prior,
    *,
    batch_size: int,
    torch,
) -> float:
    total = 0.0
    point_count = 0
    readout.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            stop = min(start + batch_size, features.shape[0])
            rows = _soft_cross_entropy_rows(
                readout,
                features[start:stop],
                targets[start:stop],
                card_log_prior[start:stop],
                torch,
            )
            total += float(rows.sum().item())
            point_count += int(rows.shape[0])
    return total / point_count


def fit_soft_posterior_readout(
    train_features,
    train_targets,
    train_card_log_prior,
    validation_features,
    validation_targets,
    validation_card_log_prior,
    *,
    epochs: int = 20,
    learning_rate: float = 0.02,
    batch_size: int = 512,
    seed: int = 0,
    device: str | None = None,
    patience: int = 5,
    weight_decay: float = 1e-4,
    minimum_improvement: float = 0.0,
) -> SoftPosteriorFitResult:
    """Fit a soft-target linear readout and restore its best validation epoch.

    The fixed card log prior is added to every learned linear logit. Structural
    zeros represented by ``-inf`` remain exactly zero throughout optimization.
    Epoch zero participates in validation selection, so harmful training can
    safely restore the initialized readout.
    """

    torch, nn = require_torch()
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if patience <= 0:
        raise ValueError("patience must be positive")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if minimum_improvement < 0.0:
        raise ValueError("minimum_improvement must be non-negative")
    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError("train and validation feature widths must match")
    if train_targets.shape[1] != validation_targets.shape[1]:
        raise ValueError("train and validation class counts must match")

    resolved_device = train_features.device if device is None else torch.device(device)
    dtype = train_features.dtype
    tensors = (
        train_features.to(device=resolved_device, dtype=dtype),
        train_targets.to(device=resolved_device, dtype=dtype),
        train_card_log_prior.to(device=resolved_device, dtype=dtype),
        validation_features.to(device=resolved_device, dtype=dtype),
        validation_targets.to(device=resolved_device, dtype=dtype),
        validation_card_log_prior.to(device=resolved_device, dtype=dtype),
    )
    train_x, train_b, train_prior, validation_x, validation_b, validation_prior = tensors
    _validated_readout_tensors(train_x, train_b, train_prior, name="train", torch=torch)
    _validated_readout_tensors(validation_x, validation_b, validation_prior, name="validation", torch=torch)

    torch.manual_seed(seed)
    readout = nn.Linear(train_x.shape[1], train_b.shape[1]).to(device=resolved_device, dtype=dtype)
    with torch.no_grad():
        readout.weight.zero_()
        readout.bias.zero_()
    optimizer = torch.optim.Adam(
        readout.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    order_generator = torch.Generator(device=resolved_device)
    order_generator.manual_seed(seed + 1)

    train_history = [
        _mean_soft_cross_entropy(
            readout,
            train_x,
            train_b,
            train_prior,
            batch_size=batch_size,
            torch=torch,
        )
    ]
    validation_history = [
        _mean_soft_cross_entropy(
            readout,
            validation_x,
            validation_b,
            validation_prior,
            batch_size=batch_size,
            torch=torch,
        )
    ]
    best_validation = validation_history[0]
    best_epoch = 0
    best_state = {name: value.detach().clone() for name, value in readout.state_dict().items()}
    stale_epochs = 0
    epochs_completed = 0

    for epoch in range(1, epochs + 1):
        readout.train()
        permutation = torch.randperm(train_x.shape[0], generator=order_generator, device=resolved_device)
        for start in range(0, train_x.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad()
            loss = _soft_cross_entropy_rows(
                readout,
                train_x[indices],
                train_b[indices],
                train_prior[indices],
                torch,
            ).mean()
            loss.backward()
            optimizer.step()

        epochs_completed = epoch
        train_ce = _mean_soft_cross_entropy(
            readout,
            train_x,
            train_b,
            train_prior,
            batch_size=batch_size,
            torch=torch,
        )
        validation_ce = _mean_soft_cross_entropy(
            readout,
            validation_x,
            validation_b,
            validation_prior,
            batch_size=batch_size,
            torch=torch,
        )
        train_history.append(train_ce)
        validation_history.append(validation_ce)
        if validation_ce < best_validation - minimum_improvement:
            best_validation = validation_ce
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in readout.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    readout.load_state_dict(best_state)
    readout.eval()
    return SoftPosteriorFitResult(
        model=readout,
        best_epoch=best_epoch,
        epochs_completed=epochs_completed,
        best_validation_cross_entropy_nats=best_validation,
        train_cross_entropy_nats=tuple(train_history),
        validation_cross_entropy_nats=tuple(validation_history),
        patience=patience,
        weight_decay=weight_decay,
    )


def posterior_point_metrics(targets, logits) -> dict[str, object]:
    """Return per-point posterior metrics using natural logarithms.

    Target structural zeros and matching ``-inf`` logits are handled without
    assigning probability mass to impossible classes.
    """

    torch, _ = require_torch()
    if targets.ndim != 2 or logits.ndim != 2 or targets.shape != logits.shape:
        raise ValueError("targets and logits must be equal-shape rank-2 tensors")
    if targets.shape[0] == 0 or targets.shape[1] == 0:
        raise ValueError("posterior metrics require at least one point and class")

    target = targets.detach().to(dtype=torch.float64)
    raw_logits = logits.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(target).all()) or bool((target < 0.0).any()):
        raise ValueError("posterior targets must be finite and non-negative")
    if not bool(
        torch.allclose(
            target.sum(dim=1),
            torch.ones(target.shape[0], device=target.device, dtype=target.dtype),
            atol=1e-6,
            rtol=1e-6,
        )
    ):
        raise ValueError("posterior target rows must sum to one")
    if bool(torch.isnan(raw_logits).any()) or bool(torch.isposinf(raw_logits).any()):
        raise ValueError("posterior logits must not contain NaN or positive infinity")
    finite_support = torch.isfinite(raw_logits)
    if bool((~finite_support.any(dim=1)).any()):
        raise ValueError("every posterior row must have at least one finite logit")
    if bool(((target > 0.0) & ~finite_support).any()):
        raise ValueError("posterior target has mass outside the logit support")

    log_probabilities = torch.log_softmax(raw_logits, dim=1)
    probabilities = log_probabilities.exp()
    safe_log_probabilities = log_probabilities.masked_fill(~finite_support, 0.0)
    tiny = torch.finfo(torch.float64).tiny
    log_target = target.clamp_min(tiny).log()
    target_entropy = -(target * log_target).sum(dim=1)
    cross_entropy = -(target * safe_log_probabilities).sum(dim=1)
    kl = cross_entropy - target_entropy

    middle = 0.5 * (target + probabilities)
    log_middle = middle.clamp_min(tiny).log()
    target_to_middle = (target * (log_target - log_middle)).sum(dim=1)
    log_probabilities_for_js = probabilities.clamp_min(tiny).log()
    predicted_to_middle = (probabilities * (log_probabilities_for_js - log_middle)).sum(dim=1)
    js = 0.5 * (target_to_middle + predicted_to_middle)
    brier = ((probabilities - target) ** 2).sum(dim=1)
    return {
        "target_entropy_nats": target_entropy,
        "cross_entropy_nats": cross_entropy,
        "kl_nats": kl,
        "js_nats": js,
        "brier": brier,
    }
