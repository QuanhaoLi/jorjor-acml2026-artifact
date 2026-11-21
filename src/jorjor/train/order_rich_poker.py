from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any, Callable

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.envs.order_rich_poker import (
    BEHAVIOR_ACTIONS,
    COMPOSITION_COLLAPSED,
    EXACT_POSTERIOR_SIGNAL_SUPPORTED,
    HIDDEN_STRENGTHS,
    INSUFFICIENT_EXACT_POSTERIOR_SUPPORT,
    PUBLIC_EVENTS,
    behavior_targets,
    composition_state,
    exact_posterior,
    matched_order_rows,
    matched_order_summary,
    normalize_distribution,
)
from jorjor.models.tiny import require_torch


REPORT_TYPE = "r.1_order_rich_behavior_gates"
LEARNED_POSTERIOR_SUPPORTED = "learned_posterior_supported"
COMPOSITION_SHORTCUT_SUPPORTED = "composition_shortcut_supported"
INCONCLUSIVE = "inconclusive"
ORACLE_POSTERIOR_STATE = "oracle_posterior_state"
COMPOSITION_SHORTCUT_STATE = "composition_shortcut_state"
POSTERIOR_SUPERVISED_HEAD = "posterior_supervised_gru_posterior_head"
POSTERIOR_SUPERVISED_HIDDEN_PROBE = "posterior_supervised_gru_hidden_probe"
BEHAVIOR_ONLY_HIDDEN_PROBE = "behavior_only_gru_hidden_probe"
COMPOSITION_FEATURES_LINEAR_PROBE = "composition_features_linear_probe"


def build_order_rich_behavior_gate_report(
    *,
    output_dir: Path,
    length: int = 6,
    seed: int = 20260610,
    signal_probability: float = 0.70,
    epochs: int = 50,
    hidden_size: int = 64,
    batch_size: int = 512,
    learning_rate: float = 0.01,
    posterior_loss_weight: float = 2.0,
    max_pairs: int = 30000,
    bootstrap_count: int = 1000,
    bootstrap_seed: int = 20260610,
    confidence_level: float = 0.95,
    correlation_threshold: float = 0.80,
    min_pair_count: int = 2000,
    min_bucket_count: int = 100,
    min_policy_accuracy: float = 0.80,
    shortcut_upper_threshold: float = 1e-12,
) -> dict[str, Any]:
    """Build the r.1 learned behavior-model gate report."""

    if length <= 1:
        raise ValueError("length must be greater than 1")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    prefixes = all_prefixes(length)
    posterior_model = _train_gru_model(
        prefixes=prefixes,
        length=length,
        hidden_size=hidden_size,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        signal_probability=signal_probability,
        posterior_loss_weight=posterior_loss_weight,
    )
    behavior_model = _train_gru_model(
        prefixes=prefixes,
        length=length,
        hidden_size=hidden_size,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed + 1,
        signal_probability=signal_probability,
        posterior_loss_weight=0.0,
    )
    posterior_hidden_probe = _fit_hidden_probe(
        model=posterior_model["model"],
        prefixes=prefixes,
        length=length,
        signal_probability=signal_probability,
    )
    behavior_hidden_probe = _fit_hidden_probe(
        model=behavior_model["model"],
        prefixes=prefixes,
        length=length,
        signal_probability=signal_probability,
    )
    composition_probe = _fit_feature_probe(
        prefixes=prefixes,
        feature_fn=lambda prefix: _composition_features(prefix, length),
        signal_probability=signal_probability,
    )

    case_specs: list[tuple[str, str, Callable[[tuple[str, ...]], tuple[float, ...]], dict[str, Any]]] = [
        (
            ORACLE_POSTERIOR_STATE,
            "exact_posterior_state",
            lambda prefix: exact_posterior(prefix, signal_probability=signal_probability),
            {"training_target": "none_oracle_state"},
        ),
        (
            COMPOSITION_SHORTCUT_STATE,
            "visible_composition_state",
            composition_state,
            {"training_target": "none_shortcut_state"},
        ),
        (
            POSTERIOR_SUPERVISED_HEAD,
            "gru_posterior_head",
            lambda prefix: _predict_gru_posterior_head(model=posterior_model["model"], prefix=prefix, length=length),
            {
                "training_target": "action_value_plus_exact_posterior",
                "train_metrics": posterior_model["train_metrics"],
            },
        ),
        (
            POSTERIOR_SUPERVISED_HIDDEN_PROBE,
            "gru_hidden_linear_probe",
            lambda prefix: _predict_hidden_probe(
                model=posterior_model["model"],
                probe=posterior_hidden_probe,
                prefix=prefix,
                length=length,
            ),
            {
                "training_target": "action_value_plus_exact_posterior",
                "train_metrics": posterior_model["train_metrics"],
                "probe_metrics": posterior_hidden_probe["metrics"],
            },
        ),
        (
            BEHAVIOR_ONLY_HIDDEN_PROBE,
            "gru_hidden_linear_probe",
            lambda prefix: _predict_hidden_probe(
                model=behavior_model["model"],
                probe=behavior_hidden_probe,
                prefix=prefix,
                length=length,
            ),
            {
                "training_target": "action_value_only",
                "train_metrics": behavior_model["train_metrics"],
                "probe_metrics": behavior_hidden_probe["metrics"],
            },
        ),
        (
            COMPOSITION_FEATURES_LINEAR_PROBE,
            "composition_features_linear_probe",
            lambda prefix: _predict_feature_probe(probe=composition_probe, features=_composition_features(prefix, length)),
            {
                "training_target": "posterior_probe_on_visible_composition_features",
                "probe_metrics": composition_probe["metrics"],
            },
        ),
    ]

    cases = []
    for case_index, (mode, representation_source, representation_fn, metadata) in enumerate(case_specs):
        cached_representation_fn = _cached_representation(representation_fn)
        rows = matched_order_rows(
            length=length,
            representation_fn=cached_representation_fn,
            signal_probability=signal_probability,
            max_pairs=max_pairs,
        )
        matched = matched_order_summary(
            rows,
            bootstrap_count=bootstrap_count,
            bootstrap_seed=bootstrap_seed + case_index,
            confidence_level=confidence_level,
        )
        cases.append(
            {
                "mode": mode,
                "representation_source": representation_source,
                **metadata,
                "matched_composition_summary": matched,
                "final_diagnostic_label": _verdict(
                    mode=mode,
                    matched_summary=matched,
                    train_metrics=metadata.get("train_metrics"),
                    correlation_threshold=correlation_threshold,
                    min_pair_count=min_pair_count,
                    min_bucket_count=min_bucket_count,
                    min_policy_accuracy=min_policy_accuracy,
                    shortcut_upper_threshold=shortcut_upper_threshold,
                ),
            }
        )

    payload = {
        "version": "r.1",
        "report_type": REPORT_TYPE,
        "claim_scope": "learned_exact_posterior_benchmark_gates_not_causal_mechanism",
        "length": length,
        "seed": seed,
        "signal_probability": signal_probability,
        "epochs": epochs,
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "posterior_loss_weight": posterior_loss_weight,
        "max_pairs": max_pairs,
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "correlation_threshold": correlation_threshold,
        "min_pair_count": min_pair_count,
        "min_bucket_count": min_bucket_count,
        "min_policy_accuracy": min_policy_accuracy,
        "hidden_strengths": list(HIDDEN_STRENGTHS),
        "public_events": list(PUBLIC_EVENTS),
        "behavior_actions": list(BEHAVIOR_ACTIONS),
        "train_prefix_count": len(prefixes),
        "cases": cases,
        "protocol_positive_control_supported": _protocol_positive_control_supported(cases),
        "behavior_only_supported": _case_label(cases, BEHAVIOR_ONLY_HIDDEN_PROBE) == LEARNED_POSTERIOR_SUPPORTED,
        "allowed_paper_claim": (
            "r.1 tests whether learned rows can pass exact-posterior matched-composition gates. "
            "It does not establish causal mechanism without r.2 interventions."
        ),
        "forbidden_paper_claim": "r.1 alone proves CBPS is causally used by the behavior model.",
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "r.1_order_rich_behavior_gates_report.json", payload)
    (output_dir / "r.1_order_rich_behavior_gates_report.md").write_text(_markdown(payload), encoding="utf-8")
    (output_dir / "r.1_order_rich_behavior_gates_table.md").write_text(_table_markdown(payload), encoding="utf-8")
    write_json_artifact(output_dir / "r.1_order_rich_behavior_gates_manifest.json", _manifest(output_dir, payload))
    _write_train_logs(output_dir=output_dir, posterior_model=posterior_model, behavior_model=behavior_model)
    return payload


def all_prefixes(length: int) -> list[tuple[str, ...]]:
    """Return all non-empty event prefixes through the requested length."""

    if length <= 1:
        raise ValueError("length must be greater than 1")
    prefixes: list[tuple[str, ...]] = []
    for prefix_len in range(1, length + 1):
        prefixes.extend(product(PUBLIC_EVENTS, repeat=prefix_len))
    return prefixes


class _OrderRichGRU:
    """Small GRU sequence model for r.1 exact-posterior gates."""

    def __init__(self, *, hidden_size: int, seed: int):
        torch, nn = require_torch()
        torch.manual_seed(seed)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(len(PUBLIC_EVENTS) + 1, hidden_size, padding_idx=0)
                self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
                self.policy_head = nn.Linear(hidden_size, len(BEHAVIOR_ACTIONS))
                self.value_head = nn.Linear(hidden_size, 1)
                self.posterior_head = nn.Linear(hidden_size, len(HIDDEN_STRENGTHS))

            def encode(self, input_ids, lengths):
                embedded = self.embedding(input_ids)
                packed = nn.utils.rnn.pack_padded_sequence(
                    embedded,
                    lengths.cpu(),
                    batch_first=True,
                    enforce_sorted=False,
                )
                _, hidden = self.gru(packed)
                return hidden[-1]

            def forward(self, input_ids, lengths):
                hidden = self.encode(input_ids, lengths)
                return {
                    "hidden": hidden,
                    "policy_logits": self.policy_head(hidden),
                    "value": self.value_head(hidden).squeeze(-1),
                    "posterior_logits": self.posterior_head(hidden),
                }

        self.model = Model()


def _train_gru_model(
    *,
    prefixes: list[tuple[str, ...]],
    length: int,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    signal_probability: float,
    posterior_loss_weight: float,
) -> dict[str, Any]:
    torch, _ = require_torch()
    wrapper = _OrderRichGRU(hidden_size=hidden_size, seed=seed)
    model = wrapper.model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    input_ids, lengths, policy_targets, value_targets, posterior_targets = _prefix_tensors(
        prefixes,
        length=length,
        signal_probability=signal_probability,
    )
    generator = torch.Generator().manual_seed(seed)
    logs = []
    for epoch in range(epochs):
        order = torch.randperm(input_ids.shape[0], generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            outputs = model(input_ids[indices], lengths[indices])
            policy_loss = torch.nn.functional.cross_entropy(outputs["policy_logits"], policy_targets[indices])
            value_loss = torch.nn.functional.mse_loss(outputs["value"], value_targets[indices])
            posterior_loss = (
                -(posterior_targets[indices] * torch.log_softmax(outputs["posterior_logits"], dim=1)).sum(dim=1).mean()
            )
            loss = policy_loss + value_loss + posterior_loss_weight * posterior_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(indices)
        if epoch == 0 or epoch == epochs - 1:
            logs.append({"epoch": epoch, "loss": epoch_loss / input_ids.shape[0]})
    with torch.no_grad():
        outputs = model(input_ids, lengths)
        policy_acc = float((outputs["policy_logits"].argmax(dim=1) == policy_targets).double().mean().item())
        value_mse = float(torch.nn.functional.mse_loss(outputs["value"], value_targets).item())
        posterior_probs = torch.softmax(outputs["posterior_logits"], dim=1)
        posterior_ce = float((-(posterior_targets * torch.log(posterior_probs.clamp_min(1e-12))).sum(dim=1)).mean().item())
    return {
        "model": model,
        "train_metrics": {
            "epoch_logs": logs,
            "policy_accuracy": policy_acc,
            "value_mse": value_mse,
            "posterior_cross_entropy": posterior_ce,
            "posterior_loss_weight": posterior_loss_weight,
            "train_row_count": len(prefixes),
        },
    }


def _fit_hidden_probe(*, model, prefixes: list[tuple[str, ...]], length: int, signal_probability: float) -> dict[str, Any]:
    torch, _ = require_torch()
    input_ids, lengths, _, _, posterior_targets = _prefix_tensors(
        prefixes,
        length=length,
        signal_probability=signal_probability,
    )
    with torch.no_grad():
        hidden = model(input_ids, lengths)["hidden"].double()
    return _fit_probe_from_features(hidden, posterior_targets.double())


def _fit_feature_probe(
    *,
    prefixes: list[tuple[str, ...]],
    feature_fn: Callable[[tuple[str, ...]], list[float]],
    signal_probability: float,
) -> dict[str, Any]:
    torch, _ = require_torch()
    features = torch.tensor([feature_fn(prefix) for prefix in prefixes], dtype=torch.float64)
    targets = torch.tensor(
        [exact_posterior(prefix, signal_probability=signal_probability) for prefix in prefixes],
        dtype=torch.float64,
    )
    return _fit_probe_from_features(features, targets)


def _fit_probe_from_features(features, targets) -> dict[str, Any]:
    torch, _ = require_torch()
    ones = torch.ones((features.shape[0], 1), dtype=torch.float64)
    x = torch.cat([ones, features.double()], dim=1)
    y = torch.log(targets.double().clamp_min(1e-12))
    y = y - y.mean(dim=1, keepdim=True)
    weights = torch.linalg.pinv(x) @ y
    logits = x @ weights
    probs = torch.softmax(logits, dim=1)
    ce = float((-(targets.double() * torch.log(probs.clamp_min(1e-12))).sum(dim=1)).mean().item())
    mse = float(torch.nn.functional.mse_loss(logits, y).item())
    return {
        "weights": weights,
        "metrics": {
            "probe_cross_entropy": ce,
            "probe_mse": mse,
            "feature_count": int(features.shape[1]),
            "row_count": int(features.shape[0]),
        },
    }


def _predict_hidden_probe(*, model, probe: dict[str, Any], prefix: tuple[str, ...], length: int) -> tuple[float, ...]:
    torch, _ = require_torch()
    input_ids, lengths = _single_prefix_tensor(prefix, length=length)
    with torch.no_grad():
        hidden = model(input_ids, lengths)["hidden"].double()
    return _predict_feature_probe(probe=probe, features=[float(value) for value in hidden.squeeze(0).tolist()])


def _predict_gru_posterior_head(*, model, prefix: tuple[str, ...], length: int) -> tuple[float, ...]:
    torch, _ = require_torch()
    input_ids, lengths = _single_prefix_tensor(prefix, length=length)
    with torch.no_grad():
        probs = torch.softmax(model(input_ids, lengths)["posterior_logits"], dim=1).squeeze(0).tolist()
    return normalize_distribution(tuple(float(value) for value in probs))


def _predict_feature_probe(*, probe: dict[str, Any], features: list[float]) -> tuple[float, ...]:
    torch, _ = require_torch()
    vector = torch.tensor([1.0, *features], dtype=torch.float64)
    probs = torch.softmax(vector @ probe["weights"], dim=0)
    return normalize_distribution(tuple(float(value) for value in probs.tolist()))


def _cached_representation(
    representation_fn: Callable[[tuple[str, ...]], tuple[float, ...]],
) -> Callable[[tuple[str, ...]], tuple[float, ...]]:
    cache: dict[tuple[str, ...], tuple[float, ...]] = {}

    def cached(prefix: tuple[str, ...]) -> tuple[float, ...]:
        if prefix not in cache:
            cache[prefix] = representation_fn(prefix)
        return cache[prefix]

    return cached


def _prefix_tensors(
    prefixes: list[tuple[str, ...]],
    *,
    length: int,
    signal_probability: float,
):
    torch, _ = require_torch()
    input_rows = []
    lengths = []
    policy = []
    value = []
    posterior = []
    for prefix in prefixes:
        row = [PUBLIC_EVENTS.index(event) + 1 for event in prefix]
        row.extend([0] * (length - len(row)))
        input_rows.append(row)
        lengths.append(len(prefix))
        targets = behavior_targets(prefix, signal_probability=signal_probability)
        policy.append(BEHAVIOR_ACTIONS.index(str(targets["action_label"])))
        value.append(float(targets["value_label"]))
        posterior.append(exact_posterior(prefix, signal_probability=signal_probability))
    return (
        torch.tensor(input_rows, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(policy, dtype=torch.long),
        torch.tensor(value, dtype=torch.float32),
        torch.tensor(posterior, dtype=torch.float32),
    )


def _single_prefix_tensor(prefix: tuple[str, ...], *, length: int):
    torch, _ = require_torch()
    row = [PUBLIC_EVENTS.index(event) + 1 for event in prefix]
    row.extend([0] * (length - len(row)))
    return torch.tensor([row], dtype=torch.long), torch.tensor([len(prefix)], dtype=torch.long)


def _composition_features(prefix: tuple[str, ...], length: int) -> list[float]:
    features = [1.0, len(prefix) / float(length)]
    counts = {event: prefix.count(event) for event in PUBLIC_EVENTS}
    features.extend(counts[event] / float(length) for event in PUBLIC_EVENTS)
    features.extend(1.0 if prefix[-1] == event else 0.0 for event in PUBLIC_EVENTS)
    return features


def _verdict(
    *,
    mode: str,
    matched_summary: dict[str, Any],
    train_metrics: dict[str, Any] | None,
    correlation_threshold: float,
    min_pair_count: int,
    min_bucket_count: int,
    min_policy_accuracy: float,
    shortcut_upper_threshold: float,
) -> str:
    interval = matched_summary["representation_js_ci"]
    lower = float(interval["lower"])
    upper = float(interval["upper"])
    corr = matched_summary["pair_weighted_posterior_representation_correlation"]
    enough_support = (
        int(matched_summary["pair_count"]) >= min_pair_count
        and int(matched_summary["matched_bucket_count"]) >= min_bucket_count
    )
    if mode == ORACLE_POSTERIOR_STATE:
        if enough_support and lower > 0.0 and corr is not None and float(corr) >= correlation_threshold:
            return EXACT_POSTERIOR_SIGNAL_SUPPORTED
        return INSUFFICIENT_EXACT_POSTERIOR_SUPPORT
    if mode in {COMPOSITION_SHORTCUT_STATE, COMPOSITION_FEATURES_LINEAR_PROBE}:
        if enough_support and upper <= shortcut_upper_threshold:
            return COMPOSITION_SHORTCUT_SUPPORTED
        return INCONCLUSIVE
    policy_ok = train_metrics is None or float(train_metrics["policy_accuracy"]) >= min_policy_accuracy
    if enough_support and lower > 0.0 and corr is not None and float(corr) >= correlation_threshold and policy_ok:
        return LEARNED_POSTERIOR_SUPPORTED
    if enough_support and upper <= shortcut_upper_threshold:
        return COMPOSITION_SHORTCUT_SUPPORTED
    return INCONCLUSIVE


def _protocol_positive_control_supported(cases: list[dict[str, Any]]) -> bool:
    return (
        _case_label(cases, ORACLE_POSTERIOR_STATE) == EXACT_POSTERIOR_SIGNAL_SUPPORTED
        and _case_label(cases, COMPOSITION_SHORTCUT_STATE) == COMPOSITION_SHORTCUT_SUPPORTED
        and _case_label(cases, POSTERIOR_SUPERVISED_HEAD) == LEARNED_POSTERIOR_SUPPORTED
        and _case_label(cases, COMPOSITION_FEATURES_LINEAR_PROBE) == COMPOSITION_SHORTCUT_SUPPORTED
    )


def _case_label(cases: list[dict[str, Any]], mode: str) -> str | None:
    for case in cases:
        if case["mode"] == mode:
            return str(case["final_diagnostic_label"])
    return None


def _manifest(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": payload["version"],
        "report_type": "r.1_order_rich_behavior_gates_manifest",
        "claim_scope": payload["claim_scope"],
        "output_dir": str(output_dir),
        "expected_outputs": [
            "r.1_order_rich_behavior_gates_report.json",
            "r.1_order_rich_behavior_gates_report.md",
            "r.1_order_rich_behavior_gates_table.md",
            "r.1_order_rich_behavior_gates_manifest.json",
            "train_log_posterior_supervised_gru.jsonl",
            "train_log_behavior_only_gru.jsonl",
        ],
        "report_checksum": payload["checksum"],
        "case_labels": {case["mode"]: case["final_diagnostic_label"] for case in payload["cases"]},
        "protocol_positive_control_supported": payload["protocol_positive_control_supported"],
        "behavior_only_supported": payload["behavior_only_supported"],
        "one_command_reproduction": (
            ".venv/bin/python scripts/build_r1_order_rich_behavior_gates.py "
            f"--output-dir {output_dir} "
            f"--length {payload['length']} "
            f"--seed {payload['seed']} "
            f"--epochs {payload['epochs']} "
            f"--hidden-size {payload['hidden_size']} "
            f"--max-pairs {payload['max_pairs']} "
            f"--bootstrap-count {payload['bootstrap_count']} "
            f"--bootstrap-seed {payload['bootstrap_seed']}"
        ),
    }


def _write_train_logs(*, output_dir: Path, posterior_model: dict[str, Any], behavior_model: dict[str, Any]) -> None:
    for name, model_payload in (
        ("posterior_supervised_gru", posterior_model),
        ("behavior_only_gru", behavior_model),
    ):
        path = output_dir / f"train_log_{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in model_payload["train_metrics"]["epoch_logs"]:
                handle.write(json.dumps(row, sort_keys=True) + "\n")


def _markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# r.1 Order-Rich Behavior Gates",
            "",
            "## Scope",
            "",
            payload["allowed_paper_claim"],
            "",
            "## Summary",
            "",
            _table_markdown(payload),
            "",
            f"protocol_positive_control_supported: `{payload['protocol_positive_control_supported']}`",
            f"behavior_only_supported: `{payload['behavior_only_supported']}`",
            "",
            f"Forbidden claim: {payload['forbidden_paper_claim']}",
            "",
        ]
    )


def _table_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "| mode | target | verdict | policy acc | value mse | matched buckets | pairs | repr JS | CI lower | CI upper | corr |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        matched = case["matched_composition_summary"]
        interval = matched["representation_js_ci"]
        train = case.get("train_metrics") or {}
        lines.append(
            "| {mode} | {target} | {verdict} | {policy_acc} | {value_mse} | {buckets} | {pairs} | {repr_js:.6f} | {lower:.6f} | {upper:.6f} | {corr} |".format(
                mode=case["mode"],
                target=case["training_target"],
                verdict=case["final_diagnostic_label"],
                policy_acc=_fmt(train.get("policy_accuracy")),
                value_mse=_fmt(train.get("value_mse")),
                buckets=matched["matched_bucket_count"],
                pairs=matched["pair_count"],
                repr_js=matched["mean_representation_js"],
                lower=interval["lower"],
                upper=interval["upper"],
                corr=_fmt(matched["pair_weighted_posterior_representation_correlation"]),
            )
        )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int | float):
        return f"{float(value):.6f}"
    return str(value)
