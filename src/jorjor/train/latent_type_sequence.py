from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.envs.latent_type_sequence import (
    INCONCLUSIVE,
    LATENT_EVENTS,
    LATENT_TYPES,
    PROTOCOL_TRANSFER_SUPPORTED,
    action_value_targets,
    all_prefixes,
    composition_state,
    exact_posterior,
    generate_latent_type_sequences,
    matched_order_rows,
    matched_order_summary,
    normalize_distribution,
    transition_summary,
)
from jorjor.models.tiny import require_torch
from jorjor.probe.inspect import COMPOSITION_SHORTCUT_SUPPORTED


REPORT_TYPE = "k.1_latent_type_protocol_transfer"
MULTISEED_REPORT_TYPE = "k.1_multiseed_protocol_transfer_pack"
ORACLE_POSTERIOR_STATE = "oracle_posterior_state"
COMPOSITION_SHORTCUT_STATE = "composition_shortcut_state"
BELIEF_AUX_GRU_POSTERIOR_HEAD = "belief_aux_gru_posterior_head"
BELIEF_AUX_GRU = "belief_aux_gru_hidden_probe"
COMPOSITION_BIASED_STATE = "composition_biased_learned_state"
ACTION_VALUE_ONLY_GRU = "action_value_only_gru_hidden_probe"
MATCHED_BUCKET_ESTIMAND = "unweighted_composition_bucket_mean"
REQUIRED_CASE_MODES = (
    ORACLE_POSTERIOR_STATE,
    COMPOSITION_SHORTCUT_STATE,
    BELIEF_AUX_GRU_POSTERIOR_HEAD,
    BELIEF_AUX_GRU,
    COMPOSITION_BIASED_STATE,
    ACTION_VALUE_ONLY_GRU,
)


def build_latent_type_protocol_transfer_report(
    *,
    output_dir: Path,
    sequence_count: int = 512,
    length: int = 6,
    seed: int = 20260603,
    signal_probability: float = 0.72,
    epochs: int = 30,
    hidden_size: int = 32,
    batch_size: int = 256,
    learning_rate: float = 0.01,
    max_order_pairs: int = 4096,
    bootstrap_count: int = 1000,
    bootstrap_seed: int = 20260603,
    confidence_level: float = 0.95,
    correlation_threshold: float = 0.80,
    shortcut_upper_threshold: float = 1e-6,
) -> dict[str, Any]:
    """Build the k.1 non-poker protocol-transfer report."""

    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")
    if length <= 1:
        raise ValueError("length must be greater than 1")
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_sequences = generate_latent_type_sequences(
        sequence_count=sequence_count,
        length=length,
        seed=seed,
        signal_probability=signal_probability,
    )
    train_prefixes = all_prefixes(length)

    belief_aux_model = _train_gru_model(
        prefixes=train_prefixes,
        length=length,
        hidden_size=hidden_size,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        signal_probability=signal_probability,
        posterior_loss_weight=2.0,
    )
    action_value_model = _train_gru_model(
        prefixes=train_prefixes,
        length=length,
        hidden_size=hidden_size,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed + 1,
        signal_probability=signal_probability,
        posterior_loss_weight=0.0,
    )
    belief_aux_probe = _fit_hidden_probe(
        model=belief_aux_model["model"],
        prefixes=train_prefixes,
        length=length,
        signal_probability=signal_probability,
    )
    action_value_probe = _fit_hidden_probe(
        model=action_value_model["model"],
        prefixes=train_prefixes,
        length=length,
        signal_probability=signal_probability,
    )
    composition_probe = _fit_feature_probe(
        prefixes=train_prefixes,
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
            BELIEF_AUX_GRU_POSTERIOR_HEAD,
            "gru_posterior_head",
            lambda prefix: _predict_gru_posterior_head(
                model=belief_aux_model["model"],
                prefix=prefix,
                length=length,
            ),
            {
                "training_target": "action_value_plus_oracle_posterior",
                "train_metrics": belief_aux_model["train_metrics"],
                "interpretation": "learned_supervised_readout_positive_capacity_check",
            },
        ),
        (
            BELIEF_AUX_GRU,
            "gru_hidden_linear_probe",
            lambda prefix: _predict_hidden_probe(
                model=belief_aux_model["model"],
                probe=belief_aux_probe,
                prefix=prefix,
                length=length,
            ),
            {
                "training_target": "action_value_plus_oracle_posterior",
                "train_metrics": belief_aux_model["train_metrics"],
                "probe_metrics": belief_aux_probe["metrics"],
            },
        ),
        (
            COMPOSITION_BIASED_STATE,
            "composition_features_linear_probe",
            lambda prefix: _predict_feature_probe(
                probe=composition_probe,
                features=_composition_features(prefix, length),
            ),
            {
                "training_target": "posterior_probe_on_composition_features",
                "probe_metrics": composition_probe["metrics"],
            },
        ),
        (
            ACTION_VALUE_ONLY_GRU,
            "gru_hidden_linear_probe",
            lambda prefix: _predict_hidden_probe(
                model=action_value_model["model"],
                probe=action_value_probe,
                prefix=prefix,
                length=length,
            ),
            {
                "training_target": "action_value_only",
                "train_metrics": action_value_model["train_metrics"],
                "probe_metrics": action_value_probe["metrics"],
                "interpretation": "exploratory_behavior_only_row_not_poker_claim_upgrade",
            },
        ),
    ]

    cases = []
    for case_index, (mode, representation_source, representation_fn, metadata) in enumerate(case_specs):
        trajectory = transition_summary(
            eval_sequences,
            representation_fn=representation_fn,
            signal_probability=signal_probability,
        )
        order_rows = matched_order_rows(
            length=length,
            representation_fn=representation_fn,
            signal_probability=signal_probability,
            max_order_pairs=max_order_pairs,
        )
        order_summary = matched_order_summary(
            order_rows,
            bootstrap_count=bootstrap_count,
            bootstrap_seed=bootstrap_seed + case_index,
            confidence_level=confidence_level,
        )
        cases.append(
            {
                "mode": mode,
                "representation_source": representation_source,
                **metadata,
                "summary": trajectory,
                "matched_composition_summary": order_summary,
                "final_diagnostic_label": _verdict(
                    mode=mode,
                    trajectory=trajectory,
                    matched_summary=order_summary,
                    correlation_threshold=correlation_threshold,
                    shortcut_upper_threshold=shortcut_upper_threshold,
                ),
            }
        )

    label_preflight = _label_preflight(train_prefixes, length=length, signal_probability=signal_probability)
    payload = {
        "version": "k.1",
        "report_type": REPORT_TYPE,
        "claim_scope": "protocol_transfer_validation_not_poker_belief_claim_upgrade",
        "sequence_count": sequence_count,
        "length": length,
        "seed": seed,
        "signal_probability": signal_probability,
        "epochs": epochs,
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_order_pairs": max_order_pairs,
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "correlation_threshold": correlation_threshold,
        "shortcut_upper_threshold": shortcut_upper_threshold,
        "latent_types": list(LATENT_TYPES),
        "events": list(LATENT_EVENTS),
        "label_preflight": label_preflight,
        "cases": cases,
        "protocol_transfer_supported": _protocol_transfer_supported(cases),
        "learned_positive_capacity_supported": _learned_positive_capacity_supported(cases),
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    report_path = output_dir / "k.1_latent_type_protocol_transfer_report.json"
    write_json_artifact(report_path, payload)
    (output_dir / "k.1_latent_type_protocol_transfer_report.md").write_text(_markdown(payload), encoding="utf-8")
    (output_dir / "k.1_protocol_transfer_table.md").write_text(_table_markdown(payload), encoding="utf-8")
    manifest = _manifest(output_dir=output_dir, payload=payload)
    write_json_artifact(output_dir / "k.1_protocol_transfer_manifest.json", manifest)
    _write_model_sidecars(output_dir=output_dir, belief_aux_model=belief_aux_model, action_value_model=action_value_model)
    return payload


def build_latent_type_multiseed_pack(
    *,
    input_dir: Path,
    output_dir: Path,
    report_glob: str = "**/k.1_latent_type_protocol_transfer_report.json",
) -> dict[str, Any]:
    """Aggregate k.1 seed-level reports into a paper appendix pack."""

    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")
    if not report_glob:
        raise ValueError("report_glob must be non-empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    excluded_reports = []
    for report_path in sorted(input_dir.glob(report_glob)):
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            excluded_reports.append({"path": str(report_path), "reason": f"invalid_json:{exc.msg}"})
            continue
        validation_error = _seed_report_validation_error(payload)
        if validation_error is not None:
            excluded_reports.append({"path": str(report_path), "reason": validation_error})
            continue
        reports.append({"path": report_path, "payload": payload})

    if not reports:
        raise ValueError(f"no valid k.1 reports found under {input_dir} with glob {report_glob}")

    seed_rows = [_seed_row(report["path"], report["payload"]) for report in reports]
    case_summaries = _case_summaries([report["payload"] for report in reports])
    robust_protocol_transfer_supported = _robust_protocol_transfer_supported(
        seed_rows=seed_rows,
        case_summaries=case_summaries,
    )
    learned_positive_capacity_supported_count = sum(1 for row in seed_rows if row["learned_positive_capacity_supported"])
    payload = {
        "version": "k.1.5",
        "report_type": MULTISEED_REPORT_TYPE,
        "claim_scope": "protocol_transfer_validation_not_poker_belief_claim_upgrade; robustness_only",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "report_glob": report_glob,
        "report_count": len(seed_rows),
        "excluded_report_count": len(excluded_reports),
        "excluded_reports": excluded_reports,
        "seed_rows": seed_rows,
        "case_summaries": case_summaries,
        "robust_protocol_transfer_supported": robust_protocol_transfer_supported,
        "learned_positive_capacity_supported_count": learned_positive_capacity_supported_count,
        "action_value_only_supported_count": case_summaries[ACTION_VALUE_ONLY_GRU]["label_counts"].get(PROTOCOL_TRANSFER_SUPPORTED, 0),
        "aggregate_command": (
            ".venv/bin/python scripts/build_latent_type_multiseed_pack.py "
            f"--input-dir {input_dir} --output-dir {output_dir} --report-glob {report_glob}"
        ),
        "source_commands": [row["source_command"] for row in seed_rows if row.get("source_command")],
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "k.1_multiseed_protocol_transfer_pack.json", payload)
    (output_dir / "k.1_multiseed_protocol_transfer_pack.md").write_text(_multiseed_markdown(payload), encoding="utf-8")
    (output_dir / "k.1_multiseed_protocol_transfer_table.md").write_text(_multiseed_table_markdown(payload), encoding="utf-8")
    write_json_artifact(output_dir / "k.1_multiseed_manifest.json", _multiseed_manifest(output_dir=output_dir, payload=payload))
    return payload


class _LatentTypeGRU:
    """Small wrapper around a torch GRU sequence model."""

    def __init__(self, *, hidden_size: int, seed: int):
        torch, nn = require_torch()
        torch.manual_seed(seed)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(len(LATENT_EVENTS) + 1, hidden_size, padding_idx=0)
                self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
                self.policy_head = nn.Linear(hidden_size, 3)
                self.value_head = nn.Linear(hidden_size, 1)
                self.posterior_head = nn.Linear(hidden_size, len(LATENT_TYPES))

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
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    wrapper = _LatentTypeGRU(hidden_size=hidden_size, seed=seed)
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
    return _fit_probe_from_features(hidden, posterior_targets)


def _fit_feature_probe(
    *,
    prefixes: list[tuple[str, ...]],
    feature_fn: Callable[[tuple[str, ...]], list[float]],
    signal_probability: float,
) -> dict[str, Any]:
    torch, _ = require_torch()
    features = torch.tensor([feature_fn(prefix) for prefix in prefixes], dtype=torch.float64)
    targets = torch.tensor([exact_posterior(prefix, signal_probability=signal_probability) for prefix in prefixes], dtype=torch.float64)
    return _fit_probe_from_features(features, targets)


def _fit_probe_from_features(features, targets) -> dict[str, Any]:
    torch, _ = require_torch()
    targets = targets.double()
    ones = torch.ones((features.shape[0], 1), dtype=torch.float64)
    x = torch.cat([ones, features.double()], dim=1)
    y = torch.log(targets.clamp_min(1e-12))
    y = y - y.mean(dim=1, keepdim=True)
    weights = torch.linalg.pinv(x) @ y
    logits = x @ weights
    probs = torch.softmax(logits, dim=1)
    ce = float((-(targets * torch.log(probs.clamp_min(1e-12))).sum(dim=1)).mean().item())
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
        probabilities = torch.softmax(model(input_ids, lengths)["posterior_logits"], dim=1).squeeze(0).tolist()
    return normalize_distribution(tuple(float(value) for value in probabilities))


def _predict_feature_probe(*, probe: dict[str, Any], features: list[float]) -> tuple[float, ...]:
    torch, _ = require_torch()
    vector = torch.tensor([1.0, *features], dtype=torch.float64)
    probs = torch.softmax(vector @ probe["weights"], dim=0)
    return normalize_distribution(tuple(float(value) for value in probs.tolist()))


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
        row = [LATENT_EVENTS.index(event) + 1 for event in prefix]
        row.extend([0] * (length - len(row)))
        input_rows.append(row)
        lengths.append(len(prefix))
        targets = action_value_targets(prefix, signal_probability=signal_probability)
        policy.append(targets["action_label"])
        value.append(targets["value_label"])
        posterior.append(exact_posterior(prefix, signal_probability=signal_probability))
    return (
        torch.tensor(input_rows, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(policy, dtype=torch.long),
        torch.tensor(value, dtype=torch.float64).float(),
        torch.tensor(posterior, dtype=torch.float64).float(),
    )


def _single_prefix_tensor(prefix: tuple[str, ...], *, length: int):
    torch, _ = require_torch()
    row = [LATENT_EVENTS.index(event) + 1 for event in prefix]
    row.extend([0] * (length - len(row)))
    return torch.tensor([row], dtype=torch.long), torch.tensor([len(prefix)], dtype=torch.long)


def _composition_features(prefix: tuple[str, ...], length: int) -> list[float]:
    features = [1.0, len(prefix) / float(length)]
    counts = {event: prefix.count(event) for event in LATENT_EVENTS}
    features.extend(counts[event] / float(length) for event in LATENT_EVENTS)
    features.extend(1.0 if prefix[-1] == event else 0.0 for event in LATENT_EVENTS)
    return features


def _label_preflight(prefixes: list[tuple[str, ...]], *, length: int, signal_probability: float) -> dict[str, Any]:
    rows = []
    for label in ("action_label", "value_label"):
        by_composition: dict[str, set[Any]] = {}
        values = []
        for prefix in prefixes:
            targets = action_value_targets(prefix, signal_probability=signal_probability)
            value = targets[label]
            values.append(value)
            key = tuple(round(feature, 6) for feature in _composition_features(prefix, length))
            by_composition.setdefault(str(key), set()).add(round(float(value), 6) if label == "value_label" else int(value))
        rows.append(
            {
                "label": label,
                "unique_value_count": len(set(round(float(value), 6) if label == "value_label" else int(value) for value in values)),
                "composition_bucket_count": len(by_composition),
                "composition_buckets_with_multiple_values": sum(1 for bucket_values in by_composition.values() if len(bucket_values) > 1),
                "bayes_optimal_dependency_note": "generated_from_exact_ordered_posterior_not_hidden_type_identifier",
            }
        )
    return {
        "rows": rows,
        "allowed_interpretation": "action_value_only_rows_are_exploratory_until_composition_and_matched_gates_pass",
    }


def _verdict(
    *,
    mode: str,
    trajectory: dict[str, Any],
    matched_summary: dict[str, Any],
    correlation_threshold: float,
    shortcut_upper_threshold: float,
) -> str:
    interval = matched_summary["representation_js_ci"]
    lower = interval["bootstrap_lower"] if isinstance(interval, dict) else None
    upper = interval["bootstrap_upper"] if isinstance(interval, dict) else None
    trajectory_corr = trajectory["posterior_representation_correlation"]
    matched_corr = matched_summary["posterior_representation_correlation"]
    if mode in {ORACLE_POSTERIOR_STATE, BELIEF_AUX_GRU_POSTERIOR_HEAD, BELIEF_AUX_GRU}:
        if (
            lower is not None
            and lower > 0.0
            and trajectory_corr is not None
            and trajectory_corr >= correlation_threshold
            and matched_corr is not None
            and matched_corr >= correlation_threshold
        ):
            return PROTOCOL_TRANSFER_SUPPORTED
    if mode in {COMPOSITION_SHORTCUT_STATE, COMPOSITION_BIASED_STATE}:
        if upper is not None and upper <= shortcut_upper_threshold:
            return COMPOSITION_SHORTCUT_SUPPORTED
    if mode == ACTION_VALUE_ONLY_GRU:
        if (
            lower is not None
            and lower > 0.0
            and trajectory_corr is not None
            and trajectory_corr >= correlation_threshold
            and matched_corr is not None
            and matched_corr >= correlation_threshold
        ):
            return PROTOCOL_TRANSFER_SUPPORTED
        if upper is not None and upper <= shortcut_upper_threshold:
            return COMPOSITION_SHORTCUT_SUPPORTED
    return INCONCLUSIVE


def _protocol_transfer_supported(cases: list[dict[str, Any]]) -> bool:
    labels = {case["mode"]: case["final_diagnostic_label"] for case in cases}
    return (
        labels.get(ORACLE_POSTERIOR_STATE) == PROTOCOL_TRANSFER_SUPPORTED
        and labels.get(COMPOSITION_SHORTCUT_STATE) == COMPOSITION_SHORTCUT_SUPPORTED
        and labels.get(COMPOSITION_BIASED_STATE) == COMPOSITION_SHORTCUT_SUPPORTED
    )


def _learned_positive_capacity_supported(cases: list[dict[str, Any]]) -> bool:
    labels = {case["mode"]: case["final_diagnostic_label"] for case in cases}
    return (
        labels.get(BELIEF_AUX_GRU_POSTERIOR_HEAD) == PROTOCOL_TRANSFER_SUPPORTED
        or labels.get(BELIEF_AUX_GRU) == PROTOCOL_TRANSFER_SUPPORTED
    )


def _manifest(*, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    expected_outputs = [
        "k.1_latent_type_protocol_transfer_report.json",
        "k.1_latent_type_protocol_transfer_report.md",
        "k.1_protocol_transfer_manifest.json",
        "k.1_protocol_transfer_table.md",
        "train_log_belief_aux_gru_hidden_probe.jsonl",
        "train_log_action_value_only_gru_hidden_probe.jsonl",
    ]
    return {
        "version": payload["version"],
        "report_type": "k.1_protocol_transfer_manifest",
        "claim_scope": payload["claim_scope"],
        "output_dir": str(output_dir),
        "environment_config": {
            "latent_types": payload["latent_types"],
            "events": payload["events"],
            "length": payload["length"],
            "signal_probability": payload["signal_probability"],
        },
        "expected_outputs": expected_outputs,
        "bootstrap_count": payload["bootstrap_count"],
        "bootstrap_seed": payload["bootstrap_seed"],
        "correlation_threshold": payload["correlation_threshold"],
        "report_checksum": payload["checksum"],
        "case_labels": {case["mode"]: case["final_diagnostic_label"] for case in payload["cases"]},
        "protocol_transfer_supported": payload["protocol_transfer_supported"],
        "learned_positive_capacity_supported": payload["learned_positive_capacity_supported"],
        "one_command_reproduction": (
            ".venv/bin/python scripts/build_latent_type_protocol_transfer_report.py "
            f"--output-dir {output_dir} "
            f"--sequence-count {payload['sequence_count']} "
            f"--length {payload['length']} "
            f"--epochs {payload['epochs']} "
            f"--hidden-size {payload['hidden_size']} "
            f"--bootstrap-count {payload['bootstrap_count']} "
            f"--correlation-threshold {payload['correlation_threshold']} "
            f"--max-order-pairs {payload['max_order_pairs']}"
        ),
    }


def _write_model_sidecars(*, output_dir: Path, belief_aux_model: dict[str, Any], action_value_model: dict[str, Any]) -> None:
    for mode, model_payload in (
        (BELIEF_AUX_GRU, belief_aux_model),
        (ACTION_VALUE_ONLY_GRU, action_value_model),
    ):
        log_path = output_dir / f"train_log_{mode}.jsonl"
        with log_path.open("w", encoding="utf-8") as handle:
            for row in model_payload["train_metrics"]["epoch_logs"]:
                handle.write(f"{row}\n")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# k.1 Latent-Type Protocol-Transfer Report",
        "",
        "## Scope",
        "",
        "This is protocol-transfer validation beyond poker betting histories. It does not upgrade the poker belief-tracking claim.",
        "",
        "## Summary",
        "",
        _table_markdown(payload),
        "",
        f"protocol_transfer_supported: `{payload['protocol_transfer_supported']}`",
        f"learned_positive_capacity_supported: `{payload['learned_positive_capacity_supported']}`",
    ]
    return "\n".join(lines) + "\n"


def _table_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "| mode | target | verdict | trajectory corr | matched buckets | matched pairs | bucket posterior JS | bucket representation JS | bucket lower | bucket upper |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        matched = case["matched_composition_summary"]
        interval = matched["representation_js_ci"]
        lines.append(
            "| {mode} | {target} | {verdict} | {trajectory_corr} | {buckets} | {pairs} | {posterior_js:.6f} | {representation_js:.6f} | {lower:.6f} | {upper:.6f} |".format(
                mode=case["mode"],
                target=case["training_target"],
                verdict=case["final_diagnostic_label"],
                trajectory_corr=case["summary"]["posterior_representation_correlation"],
                buckets=matched["matched_bucket_count"],
                pairs=matched["pair_count"],
                posterior_js=matched["mean_posterior_js"],
                representation_js=matched["mean_representation_js"],
                lower=interval["bootstrap_lower"],
                upper=interval["bootstrap_upper"],
            )
        )
    return "\n".join(lines)


def _seed_report_validation_error(payload: dict[str, Any]) -> str | None:
    if payload.get("report_type") != REPORT_TYPE:
        return f"wrong_report_type:{payload.get('report_type')}"
    cases = payload.get("cases")
    if not isinstance(cases, list):
        return "missing_cases"
    modes = {case.get("mode") for case in cases if isinstance(case, dict)}
    missing_modes = [mode for mode in REQUIRED_CASE_MODES if mode not in modes]
    if missing_modes:
        return "missing_required_modes:" + ",".join(missing_modes)
    for case in cases:
        if not isinstance(case, dict):
            return "invalid_case"
        matched = case.get("matched_composition_summary")
        if not isinstance(matched, dict):
            return f"missing_matched_summary:{case.get('mode')}"
        interval = matched.get("representation_js_ci")
        if not isinstance(interval, dict):
            return f"missing_representation_ci:{case.get('mode')}"
        point = matched.get("mean_representation_js")
        ci_mean = interval.get("bootstrap_mean")
        if not isinstance(point, int | float) or not isinstance(ci_mean, int | float):
            return f"invalid_point_or_ci_mean:{case.get('mode')}"
        if abs(float(point) - float(ci_mean)) > 1e-12:
            return f"pre_k_1_6_estimand_mismatch:{case.get('mode')}"
        estimand = matched.get("estimand")
        if estimand is not None and estimand != MATCHED_BUCKET_ESTIMAND:
            return f"wrong_estimand:{case.get('mode')}:{estimand}"
    return None


def _seed_row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    manifest_path = path.with_name("k.1_protocol_transfer_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return {
        "seed": payload.get("seed"),
        "source_path": str(path),
        "source_checksum": payload.get("checksum"),
        "source_command": manifest.get("one_command_reproduction"),
        "protocol_transfer_supported": bool(payload.get("protocol_transfer_supported")),
        "learned_positive_capacity_supported": bool(payload.get("learned_positive_capacity_supported")),
        "case_labels": {case["mode"]: case["final_diagnostic_label"] for case in payload["cases"]},
    }


def _case_summaries(reports: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = {}
    for mode in REQUIRED_CASE_MODES:
        cases = [_case_by_mode(report, mode) for report in reports]
        labels = [case["final_diagnostic_label"] for case in cases]
        matched_rows = [case["matched_composition_summary"] for case in cases]
        bucket_representation_values = [float(row["mean_representation_js"]) for row in matched_rows]
        bucket_posterior_values = [float(row["mean_posterior_js"]) for row in matched_rows]
        lower_values = [float(row["representation_js_ci"]["bootstrap_lower"]) for row in matched_rows]
        upper_values = [float(row["representation_js_ci"]["bootstrap_upper"]) for row in matched_rows]
        pair_values = [float(row.get("pair_weighted_mean_representation_js", row["mean_representation_js"])) for row in matched_rows]
        summaries[mode] = {
            "report_count": len(cases),
            "label_counts": _count_values(labels),
            "mean_bucket_posterior_js": _mean_values(bucket_posterior_values),
            "mean_bucket_representation_js": _mean_values(bucket_representation_values),
            "min_bucket_representation_js": min(bucket_representation_values),
            "max_bucket_representation_js": max(bucket_representation_values),
            "min_bucket_ci_lower": min(lower_values),
            "max_bucket_ci_upper": max(upper_values),
            "mean_pair_weighted_representation_js": _mean_values(pair_values),
            "total_matched_bucket_count": sum(int(row["matched_bucket_count"]) for row in matched_rows),
            "total_pair_count": sum(int(row["pair_count"]) for row in matched_rows),
        }
    return summaries


def _robust_protocol_transfer_supported(*, seed_rows: list[dict[str, Any]], case_summaries: dict[str, Any]) -> bool:
    report_count = len(seed_rows)
    return (
        report_count > 0
        and all(row["protocol_transfer_supported"] for row in seed_rows)
        and case_summaries[ORACLE_POSTERIOR_STATE]["label_counts"].get(PROTOCOL_TRANSFER_SUPPORTED, 0) == report_count
        and case_summaries[COMPOSITION_SHORTCUT_STATE]["label_counts"].get(COMPOSITION_SHORTCUT_SUPPORTED, 0) == report_count
        and case_summaries[COMPOSITION_BIASED_STATE]["label_counts"].get(COMPOSITION_SHORTCUT_SUPPORTED, 0) == report_count
    )


def _case_by_mode(payload: dict[str, Any], mode: str) -> dict[str, Any]:
    for case in payload["cases"]:
        if case["mode"] == mode:
            return case
    raise ValueError(f"missing case mode {mode}")


def _count_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _mean_values(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _multiseed_manifest(*, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": payload["version"],
        "report_type": "k.1_multiseed_manifest",
        "claim_scope": payload["claim_scope"],
        "output_dir": str(output_dir),
        "expected_outputs": [
            "k.1_multiseed_protocol_transfer_pack.json",
            "k.1_multiseed_protocol_transfer_pack.md",
            "k.1_multiseed_protocol_transfer_table.md",
            "k.1_multiseed_manifest.json",
        ],
        "report_count": payload["report_count"],
        "excluded_report_count": payload["excluded_report_count"],
        "report_checksum": payload["checksum"],
        "robust_protocol_transfer_supported": payload["robust_protocol_transfer_supported"],
        "learned_positive_capacity_supported_count": payload["learned_positive_capacity_supported_count"],
        "one_command_reproduction": payload["aggregate_command"],
    }


def _multiseed_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# k.1.5 Multi-Seed Protocol-Transfer Pack",
        "",
        "## Scope",
        "",
        "This is robustness evidence for protocol transfer beyond poker betting histories. It is not evidence that poker AR models learn ordered opponent belief tracking.",
        "",
        "## Summary",
        "",
        _multiseed_table_markdown(payload),
        "",
        f"robust_protocol_transfer_supported: `{payload['robust_protocol_transfer_supported']}`",
        f"learned_positive_capacity_supported_count: `{payload['learned_positive_capacity_supported_count']}`",
        f"excluded_report_count: `{payload['excluded_report_count']}`",
    ]
    return "\n".join(lines) + "\n"


def _multiseed_table_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "| mode | label counts | mean bucket repr JS | min bucket repr JS | max bucket repr JS | min CI lower | max CI upper | total pairs |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in REQUIRED_CASE_MODES:
        summary = payload["case_summaries"][mode]
        lines.append(
            "| {mode} | {labels} | {mean:.6f} | {min_value:.6f} | {max_value:.6f} | {lower:.6f} | {upper:.6f} | {pairs} |".format(
                mode=mode,
                labels=json.dumps(summary["label_counts"], sort_keys=True),
                mean=summary["mean_bucket_representation_js"],
                min_value=summary["min_bucket_representation_js"],
                max_value=summary["max_bucket_representation_js"],
                lower=summary["min_bucket_ci_lower"],
                upper=summary["max_bucket_ci_upper"],
                pairs=summary["total_pair_count"],
            )
        )
    return "\n".join(lines)
