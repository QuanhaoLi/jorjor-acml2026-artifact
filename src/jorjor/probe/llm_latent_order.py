from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from itertools import combinations, product
import math
import platform
import random
import shlex
import sys
from pathlib import Path
from typing import Any, Callable

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.envs.latent_type_sequence import (
    LATENT_EVENTS,
    composition_key,
    composition_state,
    exact_posterior,
    normalize_distribution,
    order_key,
)
from jorjor.models.tiny import require_torch
from jorjor.probe.counterfactual import jensen_shannon
from jorjor.probe.inspect import COMPOSITION_SHORTCUT_SUPPORTED, INCONCLUSIVE


REPORT_TYPE = "k.2_llm_latent_order_audit"
BELIEF_LIKE_TRANSFER_SUPPORTED = "belief_like_transfer_supported"
ORDER_INFORMATION_SUPPORTED = "order_information_supported"
DEFAULT_BACKENDS = ("oracle_posterior", "composition", "order_only", "surface_noise")
REPRESENTATION_METRIC = "jensen_shannon_after_nonnegative_normalization_or_softmax"
EVENT_WORDS = {"a": "alpha", "b": "beta", "c": "gamma", "d": "delta"}
SURFACE_TEMPLATES = (
    "Case: latent order audit.\nEvents: {then_events}.\nState marker:",
    "Audit item for a hidden-type sequence.\nObserved event stream: {arrow_events}.\nReadout marker:",
    "Latent sequence record. Public observations are {comma_events}.\nBelief marker:",
)


def build_llm_latent_order_report(
    *,
    output_dir: Path,
    length: int = 5,
    signal_probability: float = 0.72,
    max_pairs: int = 512,
    bootstrap_count: int = 1000,
    bootstrap_seed: int = 20260604,
    confidence_level: float = 0.95,
    backends: tuple[str, ...] = DEFAULT_BACKENDS,
    surface_template_count: int = 2,
    posterior_correlation_threshold: float = 0.80,
    equivalence_upper_threshold: float = 1e-6,
    model_id: str | None = None,
    layer_id: int = -1,
    token_selection: str = "final",
    device: str = "cpu",
    dtype: str = "auto",
    max_length: int = 256,
    hf_batch_size: int = 16,
    trust_remote_code: bool = False,
    reproduction_command: str | None = None,
) -> dict[str, Any]:
    """Build the k.2 textual latent-order frozen-LM audit report."""

    if length <= 1:
        raise ValueError("length must be greater than 1")
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    if not backends:
        raise ValueError("at least one backend is required")
    if not (1 < surface_template_count <= len(SURFACE_TEMPLATES)):
        raise ValueError("surface_template_count must be between 2 and the available template count")
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    for case_index, backend in enumerate(backends):
        representation_fn, metadata = _representation_backend(
            backend=backend,
            length=length,
            signal_probability=signal_probability,
            surface_template_count=surface_template_count,
            model_id=model_id,
            layer_id=layer_id,
            token_selection=token_selection,
            device=device,
            dtype=dtype,
            max_length=max_length,
            hf_batch_size=hf_batch_size,
            trust_remote_code=trust_remote_code,
        )
        matched_rows = _matched_composition_rows(
            length=length,
            representation_fn=representation_fn,
            signal_probability=signal_probability,
            max_pairs=max_pairs,
        )
        posterior_equivalence_rows = _posterior_equivalence_rows(
            length=length,
            representation_fn=representation_fn,
            signal_probability=signal_probability,
            max_pairs=max_pairs,
        )
        surface_equivalence_rows = _surface_equivalence_rows(
            length=length,
            representation_fn=representation_fn,
            max_pairs=max_pairs,
            surface_template_count=surface_template_count,
        )
        matched_summary = _distance_summary(
            matched_rows,
            bucket_key="composition_key",
            bootstrap_count=bootstrap_count,
            bootstrap_seed=bootstrap_seed + case_index * 10,
            confidence_level=confidence_level,
        )
        posterior_equivalence_summary = _distance_summary(
            posterior_equivalence_rows,
            bucket_key="posterior_key",
            bootstrap_count=bootstrap_count,
            bootstrap_seed=bootstrap_seed + case_index * 10 + 1,
            confidence_level=confidence_level,
        )
        surface_equivalence_summary = _distance_summary(
            surface_equivalence_rows,
            bucket_key="order_key",
            bootstrap_count=bootstrap_count,
            bootstrap_seed=bootstrap_seed + case_index * 10 + 2,
            confidence_level=confidence_level,
        )
        label, reasons = _verdict(
            backend=backend,
            matched_summary=matched_summary,
            posterior_equivalence_summary=posterior_equivalence_summary,
            surface_equivalence_summary=surface_equivalence_summary,
            posterior_correlation_threshold=posterior_correlation_threshold,
            equivalence_upper_threshold=equivalence_upper_threshold,
        )
        cases.append(
            {
                "backend": backend,
                "backend_metadata": metadata,
                "representation_metric": REPRESENTATION_METRIC,
                "matched_composition_summary": matched_summary,
                "posterior_equivalence_summary": posterior_equivalence_summary,
                "surface_equivalence_summary": surface_equivalence_summary,
                "label_permutation_control": _label_permutation_control(
                    matched_rows,
                    bootstrap_seed=bootstrap_seed + case_index * 10 + 3,
                ),
                "bag_of_events_control": {
                    "definition": "composition backend is the bag-of-events/history-destruction baseline",
                    "expected_matched_representation_distance": 0.0,
                },
                "final_diagnostic_label": label,
                "diagnostic_reasons": reasons,
            }
        )

    payload = {
        "version": "k.2.0",
        "report_type": REPORT_TYPE,
        "claim_scope": "frozen_lm_transfer_stress_test_not_poker_belief_claim_upgrade",
        "length": length,
        "signal_probability": signal_probability,
        "max_pairs": max_pairs,
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "surface_template_count": surface_template_count,
        "hf_batch_size": hf_batch_size,
        "posterior_correlation_threshold": posterior_correlation_threshold,
        "equivalence_upper_threshold": equivalence_upper_threshold,
        "textualization": {
            "event_words": EVENT_WORDS,
            "templates": list(SURFACE_TEMPLATES[:surface_template_count]),
        },
        "environment": _environment_summary(),
        "cases": cases,
        "reproduction_command": reproduction_command or _default_reproduction_command(output_dir),
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "k.2_llm_latent_order_report.json", payload)
    (output_dir / "k.2_llm_latent_order_report.md").write_text(_markdown(payload), encoding="utf-8")
    (output_dir / "k.2_llm_latent_order_table.md").write_text(_table_markdown(payload), encoding="utf-8")
    write_json_artifact(output_dir / "k.2_llm_latent_order_manifest.json", _manifest(output_dir, payload))
    return payload


def textualize_prefix(prefix: tuple[str, ...], *, template_index: int) -> str:
    """Return a surface text for one symbolic latent-order prefix."""

    if template_index < 0 or template_index >= len(SURFACE_TEMPLATES):
        raise ValueError(f"unknown template_index: {template_index}")
    words = [EVENT_WORDS[event] for event in prefix]
    return SURFACE_TEMPLATES[template_index].format(
        then_events=" then ".join(words),
        arrow_events=" -> ".join(words),
        comma_events=", ".join(words),
    )


def _representation_backend(
    *,
    backend: str,
    length: int,
    signal_probability: float,
    surface_template_count: int,
    model_id: str | None,
    layer_id: int,
    token_selection: str,
    device: str,
    dtype: str,
    max_length: int,
    hf_batch_size: int,
    trust_remote_code: bool,
) -> tuple[Callable[[tuple[str, ...], int], tuple[float, ...]], dict[str, Any]]:
    if backend == "oracle_posterior":
        return (
            _tag_backend(
                lambda prefix, template_index=0: exact_posterior(prefix, signal_probability=signal_probability),
                backend,
            ),
            {"backend_type": "fake_oracle", "model_id": "exact_symbolic_posterior", "layer_id": "oracle"},
        )
    if backend == "composition":
        return (
            _tag_backend(lambda prefix, template_index=0: composition_state(prefix), backend),
            {"backend_type": "fake_shortcut", "model_id": "visible_event_composition", "layer_id": "none"},
        )
    if backend == "order_only":
        return (
            _tag_backend(lambda prefix, template_index=0: _order_only_state(prefix, length=length), backend),
            {"backend_type": "fake_control", "model_id": "ordered_sequence_one_hot", "layer_id": "none"},
        )
    if backend == "surface_noise":
        return (
            _tag_backend(
                lambda prefix, template_index=0: _surface_noise_state(
                    prefix,
                    template_index=template_index,
                    surface_template_count=surface_template_count,
                    signal_probability=signal_probability,
                ),
                backend,
            ),
            {"backend_type": "fake_control", "model_id": "posterior_plus_surface_template", "layer_id": "none"},
        )
    if backend == "hf_causal_lm":
        if model_id is None:
            raise ValueError("model_id is required for hf_causal_lm backend")
        return (
            _tag_backend(
                _hf_causal_lm_backend(
                    model_id=model_id,
                    layer_id=layer_id,
                    token_selection=token_selection,
                    device=device,
                    dtype=dtype,
                    max_length=max_length,
                    batch_size=hf_batch_size,
                    length=length,
                    surface_template_count=surface_template_count,
                    trust_remote_code=trust_remote_code,
                ),
                backend,
            ),
            {
                "backend_type": "frozen_hf_causal_lm",
                "model_id": model_id,
                "layer_id": layer_id,
                "token_selection": token_selection,
                "device": device,
                "dtype": dtype,
                "max_length": max_length,
                "batch_size": hf_batch_size,
                "trust_remote_code": trust_remote_code,
            },
        )
    raise ValueError(f"unknown backend: {backend}")


def _tag_backend(
    representation_fn: Callable[[tuple[str, ...], int], tuple[float, ...]],
    backend: str,
) -> Callable[[tuple[str, ...], int], tuple[float, ...]]:
    setattr(representation_fn, "_jorjor_backend", backend)
    return representation_fn


def _matched_composition_rows(
    *,
    length: int,
    representation_fn: Callable[[tuple[str, ...], int], tuple[float, ...]],
    signal_probability: float,
    max_pairs: int,
) -> list[dict[str, Any]]:
    skeletons = _matched_composition_pair_skeletons(
        length=length,
        signal_probability=signal_probability,
        max_pairs=max_pairs,
    )
    return [
        {
            "composition_key": skeleton["composition_key"],
            "left_order_key": order_key(skeleton["left"]),
            "right_order_key": order_key(skeleton["right"]),
            "posterior_distance": skeleton["posterior_distance"],
            "representation_distance": _representation_distance_for(
                representation_fn,
                skeleton["left"],
                skeleton["right"],
            ),
        }
        for skeleton in skeletons
    ]


@lru_cache(maxsize=64)
def _matched_composition_pair_skeletons(
    *,
    length: int,
    signal_probability: float,
    max_pairs: int,
) -> list[dict[str, Any]]:
    """Select matched-composition pair skeletons before representation calls."""

    groups: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for prefix in product(LATENT_EVENTS, repeat=length):
        groups[composition_key(prefix)].append(prefix)
    bucket_rows: dict[str, list[dict[str, Any]]] = {}
    for bucket, candidates in sorted(groups.items()):
        rows = []
        for left, right in combinations(candidates, 2):
            posterior_distance = _posterior_distance(left, right, signal_probability=signal_probability)
            if posterior_distance <= 1e-12:
                continue
            rows.append(
                {
                    "composition_key": bucket,
                    "left": left,
                    "right": right,
                    "posterior_distance": posterior_distance,
                }
            )
        if rows:
            bucket_rows[bucket] = rows
    return _round_robin_cap(bucket_rows, max_pairs=max_pairs)


def _round_robin_cap(bucket_rows: dict[str, list[dict[str, Any]]], *, max_pairs: int) -> list[dict[str, Any]]:
    """Take capped rows while preserving broad bucket coverage."""

    selected = []
    buckets = sorted(bucket_rows)
    row_index = 0
    while len(selected) < max_pairs:
        added = False
        for bucket in buckets:
            rows = bucket_rows[bucket]
            if row_index >= len(rows):
                continue
            selected.append(rows[row_index])
            added = True
            if len(selected) >= max_pairs:
                break
        if not added:
            break
        row_index += 1
    return selected


def _posterior_equivalence_rows(
    *,
    length: int,
    representation_fn: Callable[[tuple[str, ...], int], tuple[float, ...]],
    signal_probability: float,
    max_pairs: int,
) -> list[dict[str, Any]]:
    skeletons = _posterior_equivalence_pair_skeletons(
        length=length,
        signal_probability=signal_probability,
        max_pairs=max_pairs,
    )
    return [
        {
            "posterior_key": skeleton["posterior_key"],
            "left_order_key": order_key(skeleton["left"]),
            "right_order_key": order_key(skeleton["right"]),
            "posterior_distance": skeleton["posterior_distance"],
            "representation_distance": _representation_distance_for(
                representation_fn,
                skeleton["left"],
                skeleton["right"],
            ),
        }
        for skeleton in skeletons
    ]


@lru_cache(maxsize=64)
def _posterior_equivalence_pair_skeletons(
    *,
    length: int,
    signal_probability: float,
    max_pairs: int,
) -> list[dict[str, Any]]:
    rows = []
    groups: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for prefix in product(LATENT_EVENTS, repeat=length):
        groups[_posterior_key(exact_posterior(prefix, signal_probability=signal_probability))].append(prefix)
    for posterior_key, candidates in sorted(groups.items()):
        for left, right in combinations(candidates, 2):
            rows.append(
                {
                    "posterior_key": posterior_key,
                    "left": left,
                    "right": right,
                    "posterior_distance": _posterior_distance(left, right, signal_probability=signal_probability),
                }
            )
            if len(rows) >= max_pairs:
                return rows
    return rows


def _surface_equivalence_rows(
    *,
    length: int,
    representation_fn: Callable[[tuple[str, ...], int], tuple[float, ...]],
    max_pairs: int,
    surface_template_count: int,
) -> list[dict[str, Any]]:
    skeletons = _surface_equivalence_skeletons(length=length, max_pairs=max_pairs, surface_template_count=surface_template_count)
    return [
        {
            "order_key": order_key(skeleton["prefix"]),
            "left_template_index": skeleton["left_template_index"],
            "right_template_index": skeleton["right_template_index"],
            "posterior_distance": 0.0,
            "representation_distance": _representation_distance_for(
                representation_fn,
                skeleton["prefix"],
                skeleton["prefix"],
                left_template_index=skeleton["left_template_index"],
                right_template_index=skeleton["right_template_index"],
            ),
        }
        for skeleton in skeletons
    ]


@lru_cache(maxsize=64)
def _surface_equivalence_skeletons(
    *,
    length: int,
    max_pairs: int,
    surface_template_count: int,
) -> list[dict[str, Any]]:
    rows = []
    if surface_template_count < 2:
        return rows
    for prefix in product(LATENT_EVENTS, repeat=length):
        rows.append(
            {
                "prefix": prefix,
                "left_template_index": 0,
                "right_template_index": 1,
            }
        )
        if len(rows) >= max_pairs:
            return rows
    return rows


def _distance_summary(
    rows: list[dict[str, Any]],
    *,
    bucket_key: str,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    posterior_distances = [float(row["posterior_distance"]) for row in rows]
    representation_distances = [float(row["representation_distance"]) for row in rows]
    bucket_values: dict[str, list[float]] = defaultdict(list)
    bucket_posterior: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        bucket = str(row[bucket_key])
        bucket_values[bucket].append(float(row["representation_distance"]))
        bucket_posterior[bucket].append(float(row["posterior_distance"]))
    bucket_representation_means = [_mean(bucket_values[bucket]) for bucket in sorted(bucket_values)]
    bucket_posterior_means = [_mean(bucket_posterior[bucket]) for bucket in sorted(bucket_posterior)]
    return {
        "pair_count": len(rows),
        "bucket_count": len(bucket_values),
        "bucket_key": bucket_key,
        "estimand": "unweighted_bucket_mean",
        "mean_posterior_distance": _mean(bucket_posterior_means),
        "mean_representation_distance": _mean(bucket_representation_means),
        "pair_weighted_mean_posterior_distance": _mean(posterior_distances),
        "pair_weighted_mean_representation_distance": _mean(representation_distances),
        "posterior_representation_correlation": _pearson(posterior_distances, representation_distances),
        "representation_distance_ci": _bucket_bootstrap_interval(
            bucket_representation_means,
            bootstrap_count=bootstrap_count,
            bootstrap_seed=bootstrap_seed,
            confidence_level=confidence_level,
        )
        if bucket_representation_means
        else None,
    }


def _label_permutation_control(rows: list[dict[str, Any]], *, bootstrap_seed: int) -> dict[str, Any]:
    posterior_distances = [float(row["posterior_distance"]) for row in rows]
    representation_distances = [float(row["representation_distance"]) for row in rows]
    shuffled = list(posterior_distances)
    random.Random(bootstrap_seed).shuffle(shuffled)
    return {
        "definition": "matched-composition posterior distances shuffled against representation distances",
        "row_count": len(rows),
        "real_correlation": _pearson(posterior_distances, representation_distances),
        "shuffled_correlation": _pearson(shuffled, representation_distances),
    }


def _verdict(
    *,
    backend: str,
    matched_summary: dict[str, Any],
    posterior_equivalence_summary: dict[str, Any],
    surface_equivalence_summary: dict[str, Any],
    posterior_correlation_threshold: float,
    equivalence_upper_threshold: float,
) -> tuple[str, list[str]]:
    reasons = []
    matched_ci = matched_summary["representation_distance_ci"] or {}
    posterior_equivalence_ci = posterior_equivalence_summary["representation_distance_ci"] or {}
    surface_equivalence_ci = surface_equivalence_summary["representation_distance_ci"] or {}
    matched_lower = matched_ci.get("bootstrap_lower")
    matched_upper = matched_ci.get("bootstrap_upper")
    posterior_equivalence_upper = posterior_equivalence_ci.get("bootstrap_upper")
    surface_equivalence_upper = surface_equivalence_ci.get("bootstrap_upper")
    matched_corr = matched_summary["posterior_representation_correlation"]

    matched_positive = matched_lower is not None and matched_lower > 0.0
    matched_collapsed = matched_upper is not None and matched_upper <= equivalence_upper_threshold
    corr_ok = matched_corr is not None and matched_corr >= posterior_correlation_threshold
    posterior_equivalence_ok = (
        posterior_equivalence_upper is not None and posterior_equivalence_upper <= equivalence_upper_threshold
    )
    surface_equivalence_ok = surface_equivalence_upper is not None and surface_equivalence_upper <= equivalence_upper_threshold

    if backend == "composition" or matched_collapsed:
        reasons.append("matched_composition_representation_distance_collapsed")
        return COMPOSITION_SHORTCUT_SUPPORTED, reasons
    if matched_positive and corr_ok and posterior_equivalence_ok and surface_equivalence_ok:
        reasons.append("matched_separation_tracks_posterior_and_equivalence_gates_pass")
        return BELIEF_LIKE_TRANSFER_SUPPORTED, reasons
    if matched_positive:
        reasons.append("matched_composition_separates_but_equivalence_or_correlation_gate_blocks_belief_claim")
        if not corr_ok:
            reasons.append("posterior_representation_correlation_failed")
        if not posterior_equivalence_ok:
            reasons.append("posterior_equivalence_failed")
        if not surface_equivalence_ok:
            reasons.append("surface_invariance_failed")
        return ORDER_INFORMATION_SUPPORTED, reasons
    reasons.append("insufficient_matched_composition_separation")
    return INCONCLUSIVE, reasons


def _hf_causal_lm_backend(
    *,
    model_id: str,
    layer_id: int,
    token_selection: str,
    device: str,
    dtype: str,
    max_length: int,
    batch_size: int,
    length: int,
    surface_template_count: int,
    trust_remote_code: bool,
) -> Callable[[tuple[str, ...], int], tuple[float, ...]]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("hf_causal_lm backend requires transformers") from exc
    torch, _ = require_torch()
    torch_dtype = _resolve_torch_dtype(torch, dtype=dtype, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
    model.to(device)
    model.eval()
    cache: dict[tuple[tuple[str, ...], int], tuple[float, ...]] = {}
    _precompute_hf_cache(
        cache=cache,
        tokenizer=tokenizer,
        model=model,
        torch=torch,
        layer_id=layer_id,
        token_selection=token_selection,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        length=length,
        surface_template_count=surface_template_count,
    )

    def encode(prefix: tuple[str, ...], template_index: int = 0) -> tuple[float, ...]:
        key = (prefix, template_index)
        if key in cache:
            return cache[key]
        if token_selection != "final":
            raise ValueError(f"unsupported token_selection: {token_selection}")
        encoded = tokenizer(
            textualize_prefix(prefix, template_index=template_index),
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        selected_layer = outputs.hidden_states[layer_id]
        final_index = int(encoded["attention_mask"].sum(dim=1).item()) - 1
        vector = selected_layer[0, final_index, :].detach().float().cpu().tolist()
        cache[key] = tuple(float(value) for value in vector)
        return cache[key]

    return encode


def _precompute_hf_cache(
    *,
    cache: dict[tuple[tuple[str, ...], int], tuple[float, ...]],
    tokenizer,
    model,
    torch,
    layer_id: int,
    token_selection: str,
    device: str,
    max_length: int,
    batch_size: int,
    length: int,
    surface_template_count: int,
) -> None:
    if batch_size <= 0:
        raise ValueError("hf_batch_size must be positive")
    if token_selection != "final":
        raise ValueError(f"unsupported token_selection: {token_selection}")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    requests = [
        (prefix, template_index)
        for template_index in range(surface_template_count)
        for prefix in product(LATENT_EVENTS, repeat=length)
    ]
    for start in range(0, len(requests), batch_size):
        batch = requests[start : start + batch_size]
        texts = [textualize_prefix(prefix, template_index=template_index) for prefix, template_index in batch]
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        selected_layer = outputs.hidden_states[layer_id].detach().float().cpu()
        final_indices = encoded["attention_mask"].sum(dim=1).detach().cpu().tolist()
        for row_index, (prefix, template_index) in enumerate(batch):
            final_index = int(final_indices[row_index]) - 1
            cache[(prefix, template_index)] = tuple(float(value) for value in selected_layer[row_index, final_index, :].tolist())


def _resolve_torch_dtype(torch, *, dtype: str, device: str):
    if dtype == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"unknown dtype: {dtype}")


def _environment_summary() -> dict[str, Any]:
    torch_version = None
    transformers_version = None
    try:
        import torch

        torch_version = torch.__version__
    except ImportError:
        torch_version = "not_installed"
    try:
        import transformers

        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = "not_installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch_version,
        "transformers": transformers_version,
    }


def _order_only_state(prefix: tuple[str, ...], *, length: int) -> tuple[float, ...]:
    index = 0
    for event in prefix:
        index = index * len(LATENT_EVENTS) + LATENT_EVENTS.index(event)
    width = len(LATENT_EVENTS) ** length
    values = [0.0] * width
    values[index] = 1.0
    return tuple(values)


def _surface_noise_state(
    prefix: tuple[str, ...],
    *,
    template_index: int,
    surface_template_count: int,
    signal_probability: float,
) -> tuple[float, ...]:
    posterior = exact_posterior(prefix, signal_probability=signal_probability)
    surface = tuple(1.0 if template_index == index else 0.0 for index in range(surface_template_count))
    values = tuple(0.5 * value for value in posterior) + tuple(0.5 * value for value in surface)
    return normalize_distribution(values)


def _posterior_key(posterior: tuple[float, ...]) -> str:
    return "|".join(f"{value:.12f}" for value in posterior)


def _posterior_distance(left: tuple[str, ...], right: tuple[str, ...], *, signal_probability: float) -> float:
    return max(0.0, jensen_shannon(exact_posterior(left, signal_probability=signal_probability), exact_posterior(right, signal_probability=signal_probability)))


def _representation_distance_for(
    representation_fn: Callable[[tuple[str, ...], int], tuple[float, ...]],
    left: tuple[str, ...],
    right: tuple[str, ...],
    *,
    left_template_index: int = 0,
    right_template_index: int = 0,
) -> float:
    backend = getattr(representation_fn, "_jorjor_backend", None)
    if backend == "composition":
        return _representation_distance(composition_state(left), composition_state(right))
    if backend == "order_only":
        return 0.0 if left == right and left_template_index == right_template_index else 1.0
    return _representation_distance(
        representation_fn(left, left_template_index),
        representation_fn(right, right_template_index),
    )


def _representation_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return max(0.0, jensen_shannon(_as_distribution(left), _as_distribution(right)))


def _as_distribution(values: tuple[float, ...]) -> tuple[float, ...]:
    if not values:
        raise ValueError("representation vector must be non-empty")
    if all(value >= 0.0 for value in values) and sum(values) > 0.0:
        return normalize_distribution(values)
    max_value = max(values)
    exponentials = [math.exp(max(-60.0, min(60.0, value - max_value))) for value in values]
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def _bucket_bootstrap_interval(
    bucket_means: list[float],
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, float]:
    if bootstrap_count <= 0:
        raise ValueError("bootstrap_count must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    rng = random.Random(bootstrap_seed)
    means = []
    for _ in range(bootstrap_count):
        means.append(sum(bucket_means[rng.randrange(len(bucket_means))] for _ in bucket_means) / len(bucket_means))
    means.sort()
    lower_index = int(((1.0 - confidence_level) / 2.0) * (bootstrap_count - 1))
    upper_index = int((1.0 - (1.0 - confidence_level) / 2.0) * (bootstrap_count - 1))
    return {"bootstrap_mean": _mean(bucket_means), "bootstrap_lower": means[lower_index], "bootstrap_upper": means[upper_index]}


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x <= 0.0 or denom_y <= 0.0:
        return None
    return numerator / (denom_x * denom_y)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# k.2 LLM Latent-Order Audit",
        "",
        f"- claim scope: `{payload['claim_scope']}`",
        f"- length: {payload['length']}",
        f"- bootstrap: {payload['bootstrap_count']} samples, seed {payload['bootstrap_seed']}",
        "",
        _table_markdown(payload),
    ]
    return "\n".join(lines) + "\n"


def _table_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "| backend | verdict | matched distance | matched CI | posterior-eq distance | surface-eq distance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        matched = case["matched_composition_summary"]
        posterior_eq = case["posterior_equivalence_summary"]
        surface_eq = case["surface_equivalence_summary"]
        matched_ci = matched["representation_distance_ci"] or {}
        ci_text = f"[{matched_ci.get('bootstrap_lower', 0.0):.6f}, {matched_ci.get('bootstrap_upper', 0.0):.6f}]" if matched_ci else "NA"
        lines.append(
            "| {backend} | {label} | {matched:.6f} | {ci} | {posterior_eq:.6f} | {surface_eq:.6f} |".format(
                backend=case["backend"],
                label=case["final_diagnostic_label"],
                matched=matched["mean_representation_distance"],
                ci=ci_text,
                posterior_eq=posterior_eq["mean_representation_distance"],
                surface_eq=surface_eq["mean_representation_distance"],
            )
        )
    return "\n".join(lines)


def _manifest(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    files = ["k.2_llm_latent_order_report.json", "k.2_llm_latent_order_report.md", "k.2_llm_latent_order_table.md"]
    return {
        "version": payload["version"],
        "report_type": payload["report_type"],
        "output_dir": str(output_dir),
        "report_checksum": payload["checksum"],
        "reproduction_command": payload["reproduction_command"],
        "files": {name: str(output_dir / name) for name in files},
    }


def _default_reproduction_command(output_dir: Path) -> str:
    return " ".join(shlex.quote(part) for part in ["python", "scripts/build_llm_latent_order_report.py", "--output-dir", str(output_dir)])
