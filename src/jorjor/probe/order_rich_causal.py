from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.envs.order_rich_poker import (
    PUBLIC_EVENTS,
    exact_posterior,
    matched_order_rows,
)
from jorjor.models.tiny import require_torch
from jorjor.probe.counterfactual import jensen_shannon
from jorjor.train.order_rich_poker import (
    BEHAVIOR_ONLY_HIDDEN_PROBE,
    LEARNED_POSTERIOR_SUPPORTED,
    _fit_hidden_probe,
    _prefix_tensors,
    _predict_feature_probe,
    _train_gru_model,
    all_prefixes,
)


REPORT_TYPE = "r.2_order_rich_causal_subspace"
ROBUSTNESS_REPORT_TYPE = "r.3_order_rich_robustness_pack"
CAUSAL_SUBSPACE_SUPPORTED = "causal_subspace_supported"
PROJECTION_CAUSAL_SUPPORTED = "projection_causal_supported_patch_specificity_inconclusive"
DIAGNOSTIC_ONLY = "diagnostic_only"
ROBUST_PROJECTION_SUPPORTED = "robust_projection_supported"
ROBUSTNESS_INCONCLUSIVE = "robustness_inconclusive"
PAPER_PACK_REPORT_TYPE = "r.4_r_series_paper_pack"


def build_order_rich_causal_subspace_report(
    *,
    output_dir: Path,
    length: int = 6,
    seed: int = 20260610,
    signal_probability: float = 0.70,
    epochs: int = 50,
    hidden_size: int = 64,
    batch_size: int = 512,
    learning_rate: float = 0.01,
    max_pairs: int = 30000,
    subspace_rank: int = 3,
    random_seed: int = 20260611,
) -> dict[str, Any]:
    """Build the r.2 causal subspace report for the behavior-only GRU."""

    if length <= 1:
        raise ValueError("length must be greater than 1")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    if subspace_rank <= 0:
        raise ValueError("subspace_rank must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, _ = require_torch()

    prefixes = all_prefixes(length)
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
    model = behavior_model["model"]
    probe = _fit_hidden_probe(
        model=model,
        prefixes=prefixes,
        length=length,
        signal_probability=signal_probability,
    )
    tensors = _cached_tensors(
        model=model,
        prefixes=prefixes,
        length=length,
        signal_probability=signal_probability,
    )
    posterior_basis = _posterior_subspace_basis(probe=probe, hidden_size=hidden_size, rank=subspace_rank)
    random_basis = _random_subspace_basis(hidden_size=hidden_size, rank=posterior_basis.shape[1], seed=random_seed)
    ablation = _ablation_summary(
        model=model,
        tensors=tensors,
        posterior_basis=posterior_basis,
        random_basis=random_basis,
        probe=probe,
    )
    patch_pairs = matched_order_rows(
        length=length,
        representation_fn=lambda prefix: exact_posterior(prefix, signal_probability=signal_probability),
        signal_probability=signal_probability,
        max_pairs=max_pairs,
    )
    patching = _patching_summary(
        model=model,
        tensors=tensors,
        pairs=patch_pairs,
        posterior_basis=posterior_basis,
        random_basis=random_basis,
        probe=probe,
        random_seed=random_seed,
    )
    projection_supported = (
        ablation["posterior_policy_js_mean"] > ablation["random_policy_js_mean"]
        and ablation["posterior_value_abs_delta_mean"] > ablation["random_value_abs_delta_mean"]
        and ablation["posterior_readout_js_mean"] > ablation["random_readout_js_mean"]
    )
    patching_supported = (
        patching["posterior_patch_policy_js_mean"] > patching["random_subspace_patch_policy_js_mean"]
        and patching["posterior_patch_value_abs_delta_mean"] > patching["random_subspace_patch_value_abs_delta_mean"]
        and patching["posterior_patch_posterior_readout_js_mean"]
        > patching["random_subspace_patch_posterior_readout_js_mean"]
    )
    random_pair_specificity_supported = (
        patching["posterior_patch_policy_js_mean"] > patching["random_patch_policy_js_mean"]
        and patching["posterior_patch_value_abs_delta_mean"] > patching["random_patch_value_abs_delta_mean"]
    )
    if projection_supported and patching_supported and random_pair_specificity_supported:
        final_label = CAUSAL_SUBSPACE_SUPPORTED
    elif projection_supported and patching_supported:
        final_label = PROJECTION_CAUSAL_SUPPORTED
    else:
        final_label = DIAGNOSTIC_ONLY
    payload = {
        "version": "r.2",
        "report_type": REPORT_TYPE,
        "claim_scope": "controlled_benchmark_causal_subspace_not_real_poker_claim",
        "length": length,
        "seed": seed,
        "signal_probability": signal_probability,
        "epochs": epochs,
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_pairs": max_pairs,
        "subspace_rank_requested": subspace_rank,
        "subspace_rank_used": int(posterior_basis.shape[1]),
        "random_seed": random_seed,
        "train_metrics": behavior_model["train_metrics"],
        "probe_metrics": probe["metrics"],
        "ablation_summary": ablation,
        "patching_summary": patching,
        "projection_supported": projection_supported,
        "patching_subspace_supported": patching_supported,
        "random_pair_specificity_supported": random_pair_specificity_supported,
        "behavior_only_hidden_probe_label_from_r1": LEARNED_POSTERIOR_SUPPORTED,
        "tested_model_row": BEHAVIOR_ONLY_HIDDEN_PROBE,
        "final_diagnostic_label": final_label,
        "allowed_paper_claim": (
            "On the controlled order-rich benchmark, the behavior-only GRU has a posterior-correlated subspace "
            "whose projection/patching effects can be compared against same-rank random controls."
        ),
        "forbidden_paper_claim": "r.2 proves the original poker AR model causally uses belief tracking.",
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "r.2_order_rich_causal_subspace_report.json", payload)
    (output_dir / "r.2_order_rich_causal_subspace_report.md").write_text(_markdown(payload), encoding="utf-8")
    (output_dir / "r.2_order_rich_causal_subspace_table.md").write_text(_table_markdown(payload), encoding="utf-8")
    write_json_artifact(output_dir / "r.2_order_rich_causal_subspace_manifest.json", _manifest(output_dir, payload))
    return payload


def build_order_rich_robustness_pack(
    *,
    output_dir: Path,
    seeds: tuple[int, ...],
    length: int = 6,
    signal_probability: float = 0.70,
    epochs: int = 50,
    hidden_size: int = 64,
    batch_size: int = 512,
    learning_rate: float = 0.01,
    max_pairs: int = 30000,
    subspace_rank: int = 3,
) -> dict[str, Any]:
    """Build an r.3 multi-seed robustness pack from r.2 reports."""

    if not seeds:
        raise ValueError("seeds must be non-empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_reports = []
    for seed in seeds:
        seed_output = output_dir / f"seed_{seed}"
        report = build_order_rich_causal_subspace_report(
            output_dir=seed_output,
            length=length,
            seed=seed,
            signal_probability=signal_probability,
            epochs=epochs,
            hidden_size=hidden_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_pairs=max_pairs,
            subspace_rank=subspace_rank,
            random_seed=seed + 1,
        )
        seed_reports.append(report)
    label_counts = _count_values([str(report["final_diagnostic_label"]) for report in seed_reports])
    projection_supported_count = sum(1 for report in seed_reports if bool(report["projection_supported"]))
    patching_subspace_supported_count = sum(1 for report in seed_reports if bool(report["patching_subspace_supported"]))
    random_pair_specificity_supported_count = sum(
        1 for report in seed_reports if bool(report["random_pair_specificity_supported"])
    )
    final_label = (
        ROBUST_PROJECTION_SUPPORTED
        if projection_supported_count == len(seed_reports)
        and label_counts.get(DIAGNOSTIC_ONLY, 0) == 0
        else ROBUSTNESS_INCONCLUSIVE
    )
    payload = {
        "version": "r.3",
        "report_type": ROBUSTNESS_REPORT_TYPE,
        "claim_scope": "multi_seed_robustness_for_controlled_benchmark_projection_evidence",
        "seeds": list(seeds),
        "seed_count": len(seed_reports),
        "length": length,
        "signal_probability": signal_probability,
        "epochs": epochs,
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_pairs": max_pairs,
        "subspace_rank": subspace_rank,
        "label_counts": label_counts,
        "projection_supported_count": projection_supported_count,
        "patching_subspace_supported_count": patching_subspace_supported_count,
        "random_pair_specificity_supported_count": random_pair_specificity_supported_count,
        "mean_ablation": _mean_ablation(seed_reports),
        "mean_patching": _mean_patching(seed_reports),
        "seed_rows": [_seed_row(report) for report in seed_reports],
        "final_diagnostic_label": final_label,
        "allowed_paper_claim": (
            "r.3 tests whether r.2 projection-level behavior relevance is stable across seeds in the controlled benchmark."
        ),
        "forbidden_paper_claim": "r.3 proves real poker AR models use causal belief states.",
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "r.3_order_rich_robustness_pack.json", payload)
    (output_dir / "r.3_order_rich_robustness_pack.md").write_text(_robustness_markdown(payload), encoding="utf-8")
    (output_dir / "r.3_order_rich_robustness_table.md").write_text(_robustness_table_markdown(payload), encoding="utf-8")
    write_json_artifact(output_dir / "r.3_order_rich_robustness_manifest.json", _robustness_manifest(output_dir, payload))
    return payload


def _cached_tensors(*, model, prefixes: list[tuple[str, ...]], length: int, signal_probability: float) -> dict[str, Any]:
    torch, _ = require_torch()
    input_ids, lengths, _, _, _ = _prefix_tensors(
        prefixes,
        length=length,
        signal_probability=signal_probability,
    )
    with torch.no_grad():
        outputs = model(input_ids, lengths)
        hidden = outputs["hidden"].double()
        policy = torch.softmax(outputs["policy_logits"].double(), dim=1)
        value = outputs["value"].double()
    return {
        "prefixes": prefixes,
        "index": {prefix: index for index, prefix in enumerate(prefixes)},
        "hidden": hidden,
        "policy": policy,
        "value": value,
    }


def _posterior_subspace_basis(*, probe: dict[str, Any], hidden_size: int, rank: int):
    torch, _ = require_torch()
    weights = probe["weights"][1:, :].double()
    u, singular_values, _ = torch.linalg.svd(weights, full_matrices=False)
    effective_rank = min(rank, hidden_size, int((singular_values > 1e-8).sum().item()))
    if effective_rank <= 0:
        raise ValueError("posterior probe has zero effective rank")
    return u[:, :effective_rank]


def _random_subspace_basis(*, hidden_size: int, rank: int, seed: int):
    torch, _ = require_torch()
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn((hidden_size, rank), dtype=torch.float64, generator=generator)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q[:, :rank]


def _ablation_summary(*, model, tensors: dict[str, Any], posterior_basis, random_basis, probe: dict[str, Any]) -> dict[str, Any]:
    hidden = tensors["hidden"]
    original_policy = tensors["policy"]
    original_value = tensors["value"]
    posterior_hidden = _remove_subspace(hidden, posterior_basis)
    random_hidden = _remove_subspace(hidden, random_basis)
    posterior_outputs = _outputs_from_hidden(model=model, hidden=posterior_hidden)
    random_outputs = _outputs_from_hidden(model=model, hidden=random_hidden)
    return {
        "row_count": int(hidden.shape[0]),
        "posterior_policy_js_mean": _mean_policy_js(original_policy, posterior_outputs["policy"]),
        "random_policy_js_mean": _mean_policy_js(original_policy, random_outputs["policy"]),
        "posterior_value_abs_delta_mean": _mean_abs_delta(original_value, posterior_outputs["value"]),
        "random_value_abs_delta_mean": _mean_abs_delta(original_value, random_outputs["value"]),
        "posterior_readout_js_mean": _mean_probe_js(hidden, posterior_hidden, probe),
        "random_readout_js_mean": _mean_probe_js(hidden, random_hidden, probe),
    }


def _patching_summary(
    *,
    model,
    tensors: dict[str, Any],
    pairs: list[dict[str, Any]],
    posterior_basis,
    random_basis,
    probe: dict[str, Any],
    random_seed: int,
) -> dict[str, Any]:
    torch, _ = require_torch()
    index = tensors["index"]
    hidden = tensors["hidden"]
    original_policy = tensors["policy"]
    original_value = tensors["value"]
    pair_indices = [
        (index[tuple(row["left_events"])], index[tuple(row["right_events"])])
        for row in pairs
        if tuple(row["left_events"]) in index and tuple(row["right_events"]) in index
    ]
    if not pair_indices:
        raise ValueError("patching requires at least one matched pair")
    left = torch.tensor([left for left, _ in pair_indices], dtype=torch.long)
    right = torch.tensor([right for _, right in pair_indices], dtype=torch.long)
    posterior_patched = _patch_subspace(
        base_hidden=hidden[right],
        source_hidden=hidden[left],
        basis=posterior_basis,
    )
    random_subspace_patched = _patch_subspace(
        base_hidden=hidden[right],
        source_hidden=hidden[left],
        basis=random_basis,
    )
    generator = torch.Generator().manual_seed(random_seed)
    random_left = torch.randint(0, hidden.shape[0], (len(pair_indices),), generator=generator)
    random_pair_patched = _patch_subspace(
        base_hidden=hidden[right],
        source_hidden=hidden[random_left],
        basis=posterior_basis,
    )
    posterior_outputs = _outputs_from_hidden(model=model, hidden=posterior_patched)
    random_subspace_outputs = _outputs_from_hidden(model=model, hidden=random_subspace_patched)
    random_pair_outputs = _outputs_from_hidden(model=model, hidden=random_pair_patched)
    base_policy = original_policy[right]
    base_value = original_value[right]
    base_hidden = hidden[right]
    return {
        "matched_pair_count": len(pair_indices),
        "posterior_patch_policy_js_mean": _mean_policy_js(base_policy, posterior_outputs["policy"]),
        "random_subspace_patch_policy_js_mean": _mean_policy_js(base_policy, random_subspace_outputs["policy"]),
        "random_patch_policy_js_mean": _mean_policy_js(base_policy, random_pair_outputs["policy"]),
        "posterior_patch_value_abs_delta_mean": _mean_abs_delta(base_value, posterior_outputs["value"]),
        "random_subspace_patch_value_abs_delta_mean": _mean_abs_delta(base_value, random_subspace_outputs["value"]),
        "random_patch_value_abs_delta_mean": _mean_abs_delta(base_value, random_pair_outputs["value"]),
        "posterior_patch_posterior_readout_js_mean": _mean_probe_js(base_hidden, posterior_patched, probe),
        "random_subspace_patch_posterior_readout_js_mean": _mean_probe_js(base_hidden, random_subspace_patched, probe),
        "random_patch_posterior_readout_js_mean": _mean_probe_js(base_hidden, random_pair_patched, probe),
    }


def _remove_subspace(hidden, basis):
    return hidden - (hidden @ basis) @ basis.T


def _patch_subspace(*, base_hidden, source_hidden, basis):
    return _remove_subspace(base_hidden, basis) + (source_hidden @ basis) @ basis.T


def _outputs_from_hidden(*, model, hidden):
    torch, _ = require_torch()
    with torch.no_grad():
        policy = torch.softmax(model.policy_head(hidden.float()).double(), dim=1)
        value = model.value_head(hidden.float()).squeeze(-1).double()
    return {"policy": policy, "value": value}


def _probe_distribution(hidden, probe: dict[str, Any]):
    torch, _ = require_torch()
    ones = torch.ones((hidden.shape[0], 1), dtype=torch.float64)
    x = torch.cat([ones, hidden.double()], dim=1)
    return torch.softmax(x @ probe["weights"], dim=1)


def _mean_probe_js(left_hidden, right_hidden, probe: dict[str, Any]) -> float:
    left = _probe_distribution(left_hidden, probe)
    right = _probe_distribution(right_hidden, probe)
    return _mean_distribution_js(left, right)


def _mean_policy_js(left, right) -> float:
    return _mean_distribution_js(left, right)


def _mean_distribution_js(left, right) -> float:
    values = [
        max(0.0, jensen_shannon(tuple(a.tolist()), tuple(b.tolist())))
        for a, b in zip(left, right, strict=True)
    ]
    return sum(values) / len(values) if values else 0.0


def _mean_abs_delta(left, right) -> float:
    return float((left - right).abs().mean().item())


def _manifest(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": payload["version"],
        "report_type": "r.2_order_rich_causal_subspace_manifest",
        "claim_scope": payload["claim_scope"],
        "output_dir": str(output_dir),
        "expected_outputs": [
            "r.2_order_rich_causal_subspace_report.json",
            "r.2_order_rich_causal_subspace_report.md",
            "r.2_order_rich_causal_subspace_table.md",
            "r.2_order_rich_causal_subspace_manifest.json",
        ],
        "report_checksum": payload["checksum"],
        "final_diagnostic_label": payload["final_diagnostic_label"],
        "one_command_reproduction": (
            ".venv/bin/python scripts/build_r2_order_rich_causal_subspace.py "
            f"--output-dir {output_dir} "
            f"--length {payload['length']} "
            f"--seed {payload['seed']} "
            f"--epochs {payload['epochs']} "
            f"--hidden-size {payload['hidden_size']} "
            f"--max-pairs {payload['max_pairs']} "
            f"--subspace-rank {payload['subspace_rank_requested']}"
        ),
    }


def _markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# r.2 Order-Rich Causal Subspace",
            "",
            f"final_diagnostic_label: `{payload['final_diagnostic_label']}`",
            "",
            "## Summary",
            "",
            _table_markdown(payload),
            "",
            "## Scope",
            "",
            payload["allowed_paper_claim"],
            "",
            f"Forbidden claim: {payload['forbidden_paper_claim']}",
            "",
        ]
    )


def _table_markdown(payload: dict[str, Any]) -> str:
    ablation = payload["ablation_summary"]
    patching = payload["patching_summary"]
    lines = [
        "| test | metric | posterior subspace | random control |",
        "|---|---|---:|---:|",
        f"| ablation | policy JS | {ablation['posterior_policy_js_mean']:.6f} | {ablation['random_policy_js_mean']:.6f} |",
        f"| ablation | value abs delta | {ablation['posterior_value_abs_delta_mean']:.6f} | {ablation['random_value_abs_delta_mean']:.6f} |",
        f"| ablation | posterior readout JS | {ablation['posterior_readout_js_mean']:.6f} | {ablation['random_readout_js_mean']:.6f} |",
        f"| patching | policy JS | {patching['posterior_patch_policy_js_mean']:.6f} | {patching['random_patch_policy_js_mean']:.6f} |",
        f"| patching | value abs delta | {patching['posterior_patch_value_abs_delta_mean']:.6f} | {patching['random_patch_value_abs_delta_mean']:.6f} |",
        f"| patching | posterior readout JS | {patching['posterior_patch_posterior_readout_js_mean']:.6f} | {patching['random_patch_posterior_readout_js_mean']:.6f} |",
    ]
    return "\n".join(lines)


def _seed_row(report: dict[str, Any]) -> dict[str, Any]:
    ablation = report["ablation_summary"]
    patching = report["patching_summary"]
    return {
        "seed": report["seed"],
        "checksum": report["checksum"],
        "final_diagnostic_label": report["final_diagnostic_label"],
        "projection_supported": report["projection_supported"],
        "patching_subspace_supported": report["patching_subspace_supported"],
        "random_pair_specificity_supported": report["random_pair_specificity_supported"],
        "posterior_policy_js_mean": ablation["posterior_policy_js_mean"],
        "random_policy_js_mean": ablation["random_policy_js_mean"],
        "posterior_value_abs_delta_mean": ablation["posterior_value_abs_delta_mean"],
        "random_value_abs_delta_mean": ablation["random_value_abs_delta_mean"],
        "posterior_patch_policy_js_mean": patching["posterior_patch_policy_js_mean"],
        "random_patch_policy_js_mean": patching["random_patch_policy_js_mean"],
    }


def _mean_ablation(reports: list[dict[str, Any]]) -> dict[str, float]:
    keys = (
        "posterior_policy_js_mean",
        "random_policy_js_mean",
        "posterior_value_abs_delta_mean",
        "random_value_abs_delta_mean",
        "posterior_readout_js_mean",
        "random_readout_js_mean",
    )
    return {key: _mean([float(report["ablation_summary"][key]) for report in reports]) for key in keys}


def _mean_patching(reports: list[dict[str, Any]]) -> dict[str, float]:
    keys = (
        "posterior_patch_policy_js_mean",
        "random_subspace_patch_policy_js_mean",
        "random_patch_policy_js_mean",
        "posterior_patch_value_abs_delta_mean",
        "random_subspace_patch_value_abs_delta_mean",
        "random_patch_value_abs_delta_mean",
        "posterior_patch_posterior_readout_js_mean",
        "random_subspace_patch_posterior_readout_js_mean",
        "random_patch_posterior_readout_js_mean",
    )
    return {key: _mean([float(report["patching_summary"][key]) for report in reports]) for key in keys}


def _count_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _robustness_manifest(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": payload["version"],
        "report_type": "r.3_order_rich_robustness_manifest",
        "claim_scope": payload["claim_scope"],
        "output_dir": str(output_dir),
        "expected_outputs": [
            "r.3_order_rich_robustness_pack.json",
            "r.3_order_rich_robustness_pack.md",
            "r.3_order_rich_robustness_table.md",
            "r.3_order_rich_robustness_manifest.json",
        ],
        "report_checksum": payload["checksum"],
        "final_diagnostic_label": payload["final_diagnostic_label"],
        "one_command_reproduction": (
            ".venv/bin/python scripts/build_r3_order_rich_robustness_pack.py "
            f"--output-dir {output_dir} "
            f"--seeds {','.join(str(seed) for seed in payload['seeds'])} "
            f"--length {payload['length']} "
            f"--epochs {payload['epochs']} "
            f"--hidden-size {payload['hidden_size']} "
            f"--max-pairs {payload['max_pairs']} "
            f"--subspace-rank {payload['subspace_rank']}"
        ),
    }


def _robustness_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# r.3 Order-Rich Robustness Pack",
            "",
            f"final_diagnostic_label: `{payload['final_diagnostic_label']}`",
            "",
            _robustness_table_markdown(payload),
            "",
            "## Scope",
            "",
            payload["allowed_paper_claim"],
            "",
            f"Forbidden claim: {payload['forbidden_paper_claim']}",
            "",
        ]
    )


def _robustness_table_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "| seed | verdict | projection | patch subspace | random-pair specificity | ablation policy JS | random policy JS |",
        "|---:|---|---|---|---|---:|---:|",
    ]
    for row in payload["seed_rows"]:
        lines.append(
            "| {seed} | {final_diagnostic_label} | {projection_supported} | {patching_subspace_supported} | {random_pair_specificity_supported} | {posterior_policy_js_mean:.6f} | {random_policy_js_mean:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            f"projection_supported_count: `{payload['projection_supported_count']}/{payload['seed_count']}`",
            f"patching_subspace_supported_count: `{payload['patching_subspace_supported_count']}/{payload['seed_count']}`",
            f"random_pair_specificity_supported_count: `{payload['random_pair_specificity_supported_count']}/{payload['seed_count']}`",
        ]
    )
    return "\n".join(lines)


def build_r_series_paper_pack(
    *,
    output_dir: Path,
    r001_path: Path,
    r002_path: Path,
    r1_path: Path,
    r2_path: Path,
    r3_path: Path,
) -> dict[str, Any]:
    """Build a paper-facing r-series pack from completed artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "r.0.1": _read_json_artifact(r001_path, "r.0.1_exact_leduc_feasibility"),
        "r.0.2": _read_json_artifact(r002_path, "r.0.2_order_rich_exact_posterior"),
        "r.1": _read_json_artifact(r1_path, "r.1_order_rich_behavior_gates"),
        "r.2": _read_json_artifact(r2_path, "r.2_order_rich_causal_subspace"),
        "r.3": _read_json_artifact(r3_path, "r.3_order_rich_robustness_pack"),
    }
    source_paths = {
        "r.0.1": r001_path,
        "r.0.2": r002_path,
        "r.1": r1_path,
        "r.2": r2_path,
        "r.3": r3_path,
    }
    payload = {
        "version": "r.4",
        "report_type": PAPER_PACK_REPORT_TYPE,
        "claim_scope": "paper_reporting_layer_no_new_evidence",
        "source_artifacts": {
            key: {
                "path": str(source_paths[key]),
                "checksum": artifacts[key].get("checksum"),
                "final_diagnostic_label": artifacts[key].get("final_diagnostic_label"),
            }
            for key in sorted(artifacts)
        },
        "rows": _paper_rows(artifacts),
        "recommended_claim": (
            "The diagnostic protocol avoids overclaiming in poker and can accept controlled learned positives: "
            "in an order-rich exact-posterior benchmark, a behavior-only GRU passes matched-composition diagnostic "
            "gates, and its posterior-correlated subspace has robust projection-level behavior relevance across "
            "three seeds."
        ),
        "required_limitation": (
            "The r-series does not upgrade the original poker AR verdict. Patch specificity is incomplete because "
            "random-source posterior-subspace patching is stronger than matched-order patching."
        ),
        "forbidden_claims": [
            "The original poker AR model learned a causal belief state.",
            "The r-series proves Limit Hold'em hidden states track opponent ranges.",
            "Projection-level evidence is equivalent to full patch-specific causal belief tracking.",
        ],
        "final_diagnostic_label": artifacts["r.3"]["final_diagnostic_label"],
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "r.4_r_series_paper_pack.json", payload)
    (output_dir / "r.4_r_series_paper_pack.md").write_text(_paper_pack_markdown(payload), encoding="utf-8")
    (output_dir / "r.4_r_series_main_table.md").write_text(_paper_table_markdown(payload), encoding="utf-8")
    write_json_artifact(output_dir / "r.4_r_series_manifest.json", _paper_pack_manifest(output_dir, payload))
    return payload


def _read_json_artifact(path: Path, expected_report_type: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing r-series artifact: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    if data.get("report_type") != expected_report_type:
        raise ValueError(f"expected report_type {expected_report_type}, got {data.get('report_type')} at {path}")
    return data


def _paper_rows(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    r001 = artifacts["r.0.1"]
    r002 = artifacts["r.0.2"]
    r1 = artifacts["r.1"]
    r2 = artifacts["r.2"]
    r3 = artifacts["r.3"]
    behavior_case = _case_by_mode(r1, "behavior_only_gru_hidden_probe")
    posterior_head_case = _case_by_mode(r1, "posterior_supervised_gru_posterior_head")
    return [
        {
            "row": "local_leduc_feasibility",
            "result": r001["final_diagnostic_label"],
            "key_metric": f"matched_bucket_count={r001['matched_summary']['matched_bucket_count']}",
            "interpretation": "not every poker-like toy domain supports same-composition order gates",
            "checksum": r001["checksum"],
        },
        {
            "row": "order_rich_oracle_posterior",
            "result": r002["oracle_detection_label"],
            "key_metric": _ci_metric("repr_js", r002["oracle_summary"]["representation_js_ci"]),
            "interpretation": "gate accepts true ordered posterior signal",
            "checksum": r002["checksum"],
        },
        {
            "row": "order_rich_composition_state",
            "result": r002["composition_detection_label"],
            "key_metric": _ci_metric("repr_js", r002["composition_summary"]["representation_js_ci"]),
            "interpretation": "gate rejects pure composition",
            "checksum": r002["checksum"],
        },
        {
            "row": "posterior_supervised_gru_head",
            "result": posterior_head_case["final_diagnostic_label"],
            "key_metric": f"corr={posterior_head_case['matched_composition_summary']['pair_weighted_posterior_representation_correlation']:.6f}",
            "interpretation": "learned positive control passes",
            "checksum": r1["checksum"],
        },
        {
            "row": "behavior_only_gru_hidden_probe",
            "result": behavior_case["final_diagnostic_label"],
            "key_metric": f"corr={behavior_case['matched_composition_summary']['pair_weighted_posterior_representation_correlation']:.6f}",
            "interpretation": "behavior-only diagnostic positive case in the controlled benchmark",
            "checksum": r1["checksum"],
        },
        {
            "row": "projection_ablation",
            "result": "projection_supported" if r2["projection_supported"] else "projection_not_supported",
            "key_metric": (
                f"policy_js={r2['ablation_summary']['posterior_policy_js_mean']:.6f}"
                f" vs random={r2['ablation_summary']['random_policy_js_mean']:.6f}"
            ),
            "interpretation": "posterior-correlated subspace affects behavior outputs",
            "checksum": r2["checksum"],
        },
        {
            "row": "multi_seed_projection_robustness",
            "result": r3["final_diagnostic_label"],
            "key_metric": f"projection_supported={r3['projection_supported_count']}/{r3['seed_count']}",
            "interpretation": "bounded projection-level mechanism evidence is stable",
            "checksum": r3["checksum"],
        },
        {
            "row": "random_source_patch_specificity",
            "result": "not_supported",
            "key_metric": f"random_pair_specificity={r3['random_pair_specificity_supported_count']}/{r3['seed_count']}",
            "interpretation": "do not claim full patch-specific causal belief tracking",
            "checksum": r3["checksum"],
        },
    ]


def _case_by_mode(payload: dict[str, Any], mode: str) -> dict[str, Any]:
    for case in payload["cases"]:
        if case["mode"] == mode:
            return case
    raise ValueError(f"missing case mode {mode}")


def _ci_metric(name: str, interval: dict[str, float]) -> str:
    return f"{name}=[{float(interval['lower']):.6f},{float(interval['upper']):.6f}]"


def _paper_pack_manifest(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": payload["version"],
        "report_type": "r.4_r_series_manifest",
        "claim_scope": payload["claim_scope"],
        "output_dir": str(output_dir),
        "expected_outputs": [
            "r.4_r_series_paper_pack.json",
            "r.4_r_series_paper_pack.md",
            "r.4_r_series_main_table.md",
            "r.4_r_series_manifest.json",
        ],
        "report_checksum": payload["checksum"],
        "source_artifacts": payload["source_artifacts"],
    }


def _paper_pack_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# r.4 R-Series Paper Pack",
        "",
        "## Main Table",
        "",
        _paper_table_markdown(payload),
        "",
        "## Recommended Claim",
        "",
        payload["recommended_claim"],
        "",
        "## Required Limitation",
        "",
        payload["required_limitation"],
        "",
        "## Forbidden Claims",
        "",
    ]
    lines.extend(f"- {claim}" for claim in payload["forbidden_claims"])
    lines.append("")
    return "\n".join(lines)


def _paper_table_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "| row | result | key metric | interpretation | checksum |",
        "|---|---|---:|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {row} | {result} | {key_metric} | {interpretation} | {checksum} |".format(**row)
        )
    return "\n".join(lines)
