from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.envs.belief_required_pomdp import (
    BEHAVIOR_ONLY_BELIEF_SUPPORTED,
    COMPOSITION_SHORTCUT_SUPPORTED,
    HIDDEN_STATES,
    INCONCLUSIVE,
    OBSERVATIONS,
    REWARD_MATRIX,
    action_value_targets,
    all_prefixes,
    build_preflight_report,
    composition_state,
    exact_posterior,
    generate_sequences,
    matched_order_rows,
    matched_order_summary,
    normalize_distribution,
    transition_summary,
)
from jorjor.models.tiny import require_torch


REPORT_TYPE = "k.4.0_behavior_required_pomdp"


def build_behavior_required_pomdp_report(
    *,
    output_dir: Path,
    sequence_count: int = 512,
    length: int = 6,
    seed: int = 20260605,
    signal_probability: float = 0.76,
    epochs: int = 80,
    hidden_size: int = 48,
    batch_size: int = 256,
    learning_rate: float = 0.01,
    max_order_pairs: int = 4096,
    bootstrap_count: int = 1000,
    bootstrap_seed: int = 20260605,
    confidence_level: float = 0.95,
    correlation_threshold: float = 0.80,
    action_accuracy_threshold: float = 0.90,
    trajectory_min_posterior_js: float = 0.02,
    shortcut_upper_threshold: float = 1e-6,
) -> dict[str, Any]:
    """Build the k.4.0 behavior-only belief-required POMDP report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = build_preflight_report(
        length=length,
        signal_probability=signal_probability,
        max_order_pairs=max_order_pairs,
    )
    if not preflight["belief_required"]:
        raise ValueError("k.4 POMDP preflight failed: behavior target is not structurally belief-required")

    train_prefixes = all_prefixes(length)
    eval_sequences = generate_sequences(
        sequence_count=sequence_count,
        length=length,
        seed=seed,
        signal_probability=signal_probability,
    )
    model_payload = _train_behavior_gru(
        prefixes=train_prefixes,
        length=length,
        hidden_size=hidden_size,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        signal_probability=signal_probability,
    )
    hidden_probe = _fit_hidden_probe(
        model=model_payload["model"],
        prefixes=train_prefixes,
        length=length,
        signal_probability=signal_probability,
    )
    q_probe = _fit_q_probe(
        model=model_payload["model"],
        prefixes=train_prefixes,
        length=length,
        signal_probability=signal_probability,
    )

    case_specs: list[tuple[str, str, Callable[[tuple[str, ...]], tuple[float, ...]], dict[str, Any]]] = [
        (
            "oracle_posterior_state",
            "exact_posterior_state",
            lambda prefix: exact_posterior(prefix, signal_probability=signal_probability),
            {"training_target": "none_oracle_state"},
        ),
        (
            "composition_shortcut_state",
            "visible_composition_state",
            composition_state,
            {"training_target": "none_shortcut_state"},
        ),
        (
            "behavior_only_gru_q_head",
            "gru_behavior_q_head_inverted_to_posterior",
            lambda prefix: _predict_q_head_posterior(
                model=model_payload["model"],
                prefix=prefix,
                length=length,
            ),
            {
                "training_target": "full_action_value_vector_behavior_only",
                "train_metrics": model_payload["train_metrics"],
                "interpretation": "behavior_output_readout_not_posterior_auxiliary",
            },
        ),
        (
            "behavior_only_gru_hidden_probe",
            "gru_hidden_linear_probe",
            lambda prefix: _predict_hidden_probe(
                model=model_payload["model"],
                probe=hidden_probe,
                prefix=prefix,
                length=length,
            ),
            {
                "training_target": "action_value_only",
                "train_metrics": model_payload["train_metrics"],
                "probe_metrics": hidden_probe["metrics"],
            },
        ),
        (
            "behavior_only_gru_q_probe",
            "gru_hidden_linear_behavior_q_probe_inverted_to_posterior",
            lambda prefix: _predict_q_probe_posterior(
                model=model_payload["model"],
                probe=q_probe,
                prefix=prefix,
                length=length,
            ),
            {
                "training_target": "action_value_only",
                "probe_target": "behavior_q_values_not_posterior",
                "train_metrics": model_payload["train_metrics"],
                "probe_metrics": q_probe["metrics"],
                "interpretation": "behavior_q_probe_not_posterior_auxiliary",
            },
        ),
    ]

    cases = []
    for index, (mode, representation_source, representation_fn, metadata) in enumerate(case_specs):
        trajectory = transition_summary(
            eval_sequences,
            representation_fn=representation_fn,
            signal_probability=signal_probability,
            min_posterior_js=trajectory_min_posterior_js,
        )
        rows = matched_order_rows(
            length=length,
            representation_fn=representation_fn,
            signal_probability=signal_probability,
            max_order_pairs=max_order_pairs,
        )
        matched = matched_order_summary(
            rows,
            bootstrap_count=bootstrap_count,
            bootstrap_seed=bootstrap_seed + index,
            confidence_level=confidence_level,
        )
        cases.append(
            {
                "mode": mode,
                "representation_source": representation_source,
                **metadata,
                "summary": trajectory,
                "matched_composition_summary": matched,
                "final_diagnostic_label": _verdict(
                    mode=mode,
                    train_metrics=metadata.get("train_metrics", {}),
                    trajectory=trajectory,
                    matched_summary=matched,
                    correlation_threshold=correlation_threshold,
                    action_accuracy_threshold=action_accuracy_threshold,
                    shortcut_upper_threshold=shortcut_upper_threshold,
                ),
            }
        )

    payload = {
        "version": "k.4.0",
        "report_type": REPORT_TYPE,
        "claim_scope": "behavior_only_belief_required_non_poker_positive_control_not_poker_claim_upgrade",
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
        "action_accuracy_threshold": action_accuracy_threshold,
        "trajectory_min_posterior_js": trajectory_min_posterior_js,
        "shortcut_upper_threshold": shortcut_upper_threshold,
        "hidden_states": list(HIDDEN_STATES),
        "observations": list(OBSERVATIONS),
        "preflight": preflight,
        "cases": cases,
        "behavior_only_belief_supported": any(
            case["mode"] in {"behavior_only_gru_q_head", "behavior_only_gru_hidden_probe", "behavior_only_gru_q_probe"}
            and case["final_diagnostic_label"] == BEHAVIOR_ONLY_BELIEF_SUPPORTED
            for case in cases
        ),
        "behavior_only_partial_support": any(
            case["mode"] in {"behavior_only_gru_q_head", "behavior_only_gru_hidden_probe", "behavior_only_gru_q_probe"}
            and float(case["matched_composition_summary"]["representation_js_ci"]["bootstrap_lower"]) > 0.0
            for case in cases
        ),
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "k.4.0_behavior_required_pomdp_report.json", payload)
    (output_dir / "k.4.0_behavior_required_pomdp_report.md").write_text(_markdown(payload), encoding="utf-8")
    write_json_artifact(output_dir / "k.4.0_behavior_required_pomdp_manifest.json", _manifest(output_dir, payload))
    _write_train_log(output_dir, model_payload)
    return payload


class _BehaviorGRU:
    """Small behavior-only GRU wrapper."""

    def __init__(self, *, hidden_size: int, seed: int):
        torch, nn = require_torch()
        torch.manual_seed(seed)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(len(OBSERVATIONS) + 1, hidden_size, padding_idx=0)
                self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
                self.policy_head = nn.Linear(hidden_size, len(HIDDEN_STATES))
                self.value_head = nn.Linear(hidden_size, 1)
                self.q_head = nn.Linear(hidden_size, len(HIDDEN_STATES))

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
                    "q_values": self.q_head(hidden),
                }

        self.model = Model()


def _train_behavior_gru(
    *,
    prefixes: list[tuple[str, ...]],
    length: int,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    signal_probability: float,
) -> dict[str, Any]:
    torch, _ = require_torch()
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    wrapper = _BehaviorGRU(hidden_size=hidden_size, seed=seed)
    model = wrapper.model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    input_ids, lengths, policy_targets, value_targets, q_targets, _ = _prefix_tensors(
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
            q_loss = torch.nn.functional.mse_loss(outputs["q_values"], q_targets[indices])
            loss = policy_loss + value_loss + q_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(indices)
        if epoch == 0 or epoch == epochs - 1:
            logs.append({"epoch": epoch, "loss": epoch_loss / input_ids.shape[0]})
    with torch.no_grad():
        outputs = model(input_ids, lengths)
        policy_accuracy = float((outputs["policy_logits"].argmax(dim=1) == policy_targets).double().mean().item())
        value_mse = float(torch.nn.functional.mse_loss(outputs["value"], value_targets).item())
        q_mse = float(torch.nn.functional.mse_loss(outputs["q_values"], q_targets).item())
    return {
        "model": model,
        "train_metrics": {
            "epoch_logs": logs,
            "policy_accuracy": policy_accuracy,
            "value_mse": value_mse,
            "q_mse": q_mse,
            "train_row_count": len(prefixes),
            "posterior_loss_weight": 0.0,
        },
    }


def _fit_hidden_probe(*, model, prefixes: list[tuple[str, ...]], length: int, signal_probability: float) -> dict[str, Any]:
    torch, _ = require_torch()
    input_ids, lengths, _, _, _, posterior_targets = _prefix_tensors(
        prefixes,
        length=length,
        signal_probability=signal_probability,
    )
    with torch.no_grad():
        hidden = model(input_ids, lengths)["hidden"].double()
    return _fit_probe_from_features(hidden, posterior_targets)


def _fit_q_probe(*, model, prefixes: list[tuple[str, ...]], length: int, signal_probability: float) -> dict[str, Any]:
    torch, _ = require_torch()
    input_ids, lengths, _, _, q_targets, _ = _prefix_tensors(
        prefixes,
        length=length,
        signal_probability=signal_probability,
    )
    with torch.no_grad():
        hidden = model(input_ids, lengths)["hidden"].double()
    return _fit_linear_regression_from_features(hidden, q_targets.double())


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


def _fit_linear_regression_from_features(features, targets) -> dict[str, Any]:
    torch, _ = require_torch()
    targets = targets.double()
    ones = torch.ones((features.shape[0], 1), dtype=torch.float64)
    x = torch.cat([ones, features.double()], dim=1)
    weights = torch.linalg.pinv(x) @ targets
    predictions = x @ weights
    mse = float(torch.nn.functional.mse_loss(predictions, targets).item())
    return {
        "weights": weights,
        "metrics": {
            "probe_mse": mse,
            "feature_count": int(features.shape[1]),
            "row_count": int(features.shape[0]),
            "probe_target": "behavior_q_values",
        },
    }


def _predict_hidden_probe(*, model, probe: dict[str, Any], prefix: tuple[str, ...], length: int) -> tuple[float, ...]:
    torch, _ = require_torch()
    input_ids, lengths = _single_prefix_tensor(prefix, length=length)
    with torch.no_grad():
        hidden = model(input_ids, lengths)["hidden"].double()
    vector = torch.tensor([1.0, *[float(value) for value in hidden.squeeze(0).tolist()]], dtype=torch.float64)
    probs = torch.softmax(vector @ probe["weights"], dim=0)
    return normalize_distribution(tuple(float(value) for value in probs.tolist()))


def _predict_q_head_posterior(*, model, prefix: tuple[str, ...], length: int) -> tuple[float, ...]:
    torch, _ = require_torch()
    input_ids, lengths = _single_prefix_tensor(prefix, length=length)
    with torch.no_grad():
        q_values = model(input_ids, lengths)["q_values"].double().squeeze(0)
    reward = torch.tensor(REWARD_MATRIX, dtype=torch.float64)
    posterior = torch.linalg.pinv(reward) @ q_values
    return normalize_distribution(tuple(float(value) for value in posterior.tolist()))


def _predict_q_probe_posterior(*, model, probe: dict[str, Any], prefix: tuple[str, ...], length: int) -> tuple[float, ...]:
    torch, _ = require_torch()
    input_ids, lengths = _single_prefix_tensor(prefix, length=length)
    with torch.no_grad():
        hidden = model(input_ids, lengths)["hidden"].double()
    vector = torch.tensor([1.0, *[float(value) for value in hidden.squeeze(0).tolist()]], dtype=torch.float64)
    q_values = vector @ probe["weights"]
    reward = torch.tensor(REWARD_MATRIX, dtype=torch.float64)
    posterior = torch.linalg.pinv(reward) @ q_values
    return normalize_distribution(tuple(float(value) for value in posterior.tolist()))


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
    q_values = []
    posterior = []
    for prefix in prefixes:
        row = [OBSERVATIONS.index(observation) + 1 for observation in prefix]
        row.extend([0] * (length - len(row)))
        input_rows.append(row)
        lengths.append(len(prefix))
        targets = action_value_targets(prefix, signal_probability=signal_probability)
        policy.append(targets["action_label"])
        value.append(targets["value_label"])
        q_values.append(targets["q_values"])
        posterior.append(exact_posterior(prefix, signal_probability=signal_probability))
    return (
        torch.tensor(input_rows, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(policy, dtype=torch.long),
        torch.tensor(value, dtype=torch.float64).float(),
        torch.tensor(q_values, dtype=torch.float64).float(),
        torch.tensor(posterior, dtype=torch.float64).float(),
    )


def _single_prefix_tensor(prefix: tuple[str, ...], *, length: int):
    torch, _ = require_torch()
    row = [OBSERVATIONS.index(observation) + 1 for observation in prefix]
    row.extend([0] * (length - len(row)))
    return torch.tensor([row], dtype=torch.long), torch.tensor([len(prefix)], dtype=torch.long)


def _verdict(
    *,
    mode: str,
    train_metrics: dict[str, Any],
    trajectory: dict[str, Any],
    matched_summary: dict[str, Any],
    correlation_threshold: float,
    action_accuracy_threshold: float,
    shortcut_upper_threshold: float,
) -> str:
    interval = matched_summary["representation_js_ci"]
    lower = interval["bootstrap_lower"] if isinstance(interval, dict) else None
    upper = interval["bootstrap_upper"] if isinstance(interval, dict) else None
    trajectory_corr = trajectory["posterior_representation_correlation"]
    matched_corr = matched_summary["posterior_representation_correlation"]
    if mode == "composition_shortcut_state":
        if upper is not None and upper <= shortcut_upper_threshold:
            return COMPOSITION_SHORTCUT_SUPPORTED
        return INCONCLUSIVE
    if mode == "oracle_posterior_state":
        if lower is not None and lower > 0.0 and matched_corr is not None and matched_corr >= correlation_threshold:
            return BEHAVIOR_ONLY_BELIEF_SUPPORTED
        return INCONCLUSIVE
    if mode in {"behavior_only_gru_q_head", "behavior_only_gru_hidden_probe", "behavior_only_gru_q_probe"}:
        if (
            float(train_metrics.get("policy_accuracy", 0.0)) >= action_accuracy_threshold
            and float(train_metrics.get("q_mse", 1.0)) <= 0.05
            and lower is not None
            and lower > 0.0
            and trajectory_corr is not None
            and trajectory_corr >= correlation_threshold
            and matched_corr is not None
            and matched_corr >= correlation_threshold
        ):
            return BEHAVIOR_ONLY_BELIEF_SUPPORTED
    return INCONCLUSIVE


def _manifest(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": payload["version"],
        "report_type": "k.4.0_behavior_required_pomdp_manifest",
        "claim_scope": payload["claim_scope"],
        "output_dir": str(output_dir),
        "report_checksum": payload["checksum"],
        "case_labels": {case["mode"]: case["final_diagnostic_label"] for case in payload["cases"]},
        "preflight": payload["preflight"],
        "behavior_only_partial_support": payload["behavior_only_partial_support"],
        "one_command_reproduction": (
            ".venv/bin/python scripts/build_behavior_required_pomdp_report.py "
            f"--output-dir {output_dir} --length {payload['length']} --epochs {payload['epochs']} "
            f"--hidden-size {payload['hidden_size']} --bootstrap-count {payload['bootstrap_count']} "
            f"--max-order-pairs {payload['max_order_pairs']} "
            f"--trajectory-min-posterior-js {payload['trajectory_min_posterior_js']}"
        ),
    }


def _write_train_log(output_dir: Path, model_payload: dict[str, Any]) -> None:
    with (output_dir / "train_log_behavior_only_gru_hidden_probe.jsonl").open("w", encoding="utf-8") as handle:
        for row in model_payload["train_metrics"]["epoch_logs"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# k.4.0 Behavior-Required POMDP Report",
        "",
        "This is a non-poker behavior-only learned positive-control task. It does not upgrade the poker AR claim.",
        "",
        f"behavior_only_belief_supported: `{payload['behavior_only_belief_supported']}`",
        f"behavior_only_partial_support: `{payload['behavior_only_partial_support']}`",
        "",
        "| mode | target | verdict | action acc | trajectory corr | matched buckets | matched pairs | rep JS | lower | upper | matched corr |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        matched = case["matched_composition_summary"]
        interval = matched["representation_js_ci"]
        train = case.get("train_metrics", {})
        lines.append(
            "| {mode} | {target} | {verdict} | {action_acc:.6f} | {trajectory_corr} | {buckets} | {pairs} | {rep_js:.6f} | {lower:.6f} | {upper:.6f} | {matched_corr} |".format(
                mode=case["mode"],
                target=case["training_target"],
                verdict=case["final_diagnostic_label"],
                action_acc=float(train.get("policy_accuracy", 0.0)),
                trajectory_corr=case["summary"]["posterior_representation_correlation"],
                buckets=matched["matched_bucket_count"],
                pairs=matched["pair_count"],
                rep_js=matched["mean_representation_js"],
                lower=interval["bootstrap_lower"],
                upper=interval["bootstrap_upper"],
                matched_corr=matched["posterior_representation_correlation"],
            )
        )
    return "\n".join(lines) + "\n"
