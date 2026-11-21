from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.envs.synthetic_belief import BELIEF_LIKE_ORACLE_SUPPORTED
from jorjor.train.synthetic_belief import (
    COMPOSITION_BIASED_AR,
    COMPOSITION_STATE_RNN,
    EXPLICIT_BELIEF_STATE_RNN,
    ORDERED_BELIEF_AR,
)
from jorjor.probe.inspect import COMPOSITION_SHORTCUT_SUPPORTED, INCONCLUSIVE
from jorjor.probe.counterfactual import (
    ORDER_SENSITIVE_SUPPORTED,
    INSUFFICIENT_PAIRS,
    TRAJECTORY_ALIGNED_SUPPORTED,
)


METHODOLOGY_VALIDATION_FAILED = "failed"
METHODOLOGY_VALIDATION_MISSING = "missing"
METHODOLOGY_VALIDATION_PASSED = "passed"
REQUIRED_INSPECT_FILES = (
    "belief_shortcut_report.json",
    "belief_shortcut_report.md",
    "matched_composition_report.json",
    "matched_composition_report.md",
    "claim_gate_matrix.json",
    "claim_gate_matrix.md",
)
STRICT_EXTENSION_SPECS = (
    (
        "range_supervised_strict_gate",
        (
            "limit_ar_range_supervised_positive_control_pack.json",
            "range_supervised_positive_control_pack.json",
        ),
    ),
    ("probe_capacity_s6_s7", ("probe_capacity_s6_s7_report.json",)),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _source_entry(path: Path, *, required: bool, role: str) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "required": required,
        "present": path.exists(),
    }
    if path.exists():
        entry["sha256"] = _sha256(path)
        entry["size_bytes"] = path.stat().st_size
    else:
        entry["gap"] = "missing required artifact" if required else "missing optional artifact"
    return entry


def _optional_reports(root: Path | None, filename: str) -> list[Path]:
    if root is None or not root.exists():
        return []
    return sorted(root.glob(f"**/{filename}"))


def _optional_extension(
    inspect_dir: Path,
    strict_table_dir: Path | None,
    role: str,
    filenames: tuple[str, ...],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Read one optional strict-gate extension from the inspect directory."""

    for root in (strict_table_dir, inspect_dir):
        if root is None:
            continue
        for filename in filenames:
            path = root / filename
            if path.exists():
                return _read_json(path), _source_entry(path, required=False, role=role)
    fallback = inspect_dir / filenames[0]
    entry = _source_entry(fallback, required=False, role=role)
    return None, entry


def _methodology_validation(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize whether synthetic oracle controls validate the audit ladder."""

    if not payloads:
        return {
            "status": METHODOLOGY_VALIDATION_MISSING,
            "passed_report_count": 0,
            "report_count": 0,
            "required_labels": {
                "oracle_belief": BELIEF_LIKE_ORACLE_SUPPORTED,
                "composition_shortcut": COMPOSITION_SHORTCUT_SUPPORTED,
            },
        }

    passed = []
    summaries = []
    for payload in payloads:
        labels = {
            case.get("mode"): case.get("final_diagnostic_label")
            for case in payload.get("cases", [])
            if isinstance(case, dict)
        }
        is_pass = (
            labels.get("oracle_belief") == BELIEF_LIKE_ORACLE_SUPPORTED
            and labels.get("composition_shortcut") == COMPOSITION_SHORTCUT_SUPPORTED
        )
        passed.append(is_pass)
        summaries.append(
            {
                "path_hint": payload.get("report_type", "synthetic_oracle_belief"),
                "labels": labels,
                "passed": is_pass,
            }
        )
    return {
        "status": METHODOLOGY_VALIDATION_PASSED if all(passed) else METHODOLOGY_VALIDATION_FAILED,
        "passed_report_count": sum(1 for value in passed if value),
        "report_count": len(payloads),
        "required_labels": {
            "oracle_belief": BELIEF_LIKE_ORACLE_SUPPORTED,
            "composition_shortcut": COMPOSITION_SHORTCUT_SUPPORTED,
        },
        "reports": summaries,
    }


def _learned_methodology_validation(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize whether learned synthetic controls validate the audit ladder."""

    if not payloads:
        return {
            "status": METHODOLOGY_VALIDATION_MISSING,
            "passed_report_count": 0,
            "report_count": 0,
            "required_labels": {
                ORDERED_BELIEF_AR: BELIEF_LIKE_ORACLE_SUPPORTED,
                COMPOSITION_BIASED_AR: COMPOSITION_SHORTCUT_SUPPORTED,
            },
        }

    passed = []
    summaries = []
    for payload in payloads:
        labels = {
            case.get("mode"): case.get("final_diagnostic_label")
            for case in payload.get("cases", [])
            if isinstance(case, dict)
        }
        is_pass = (
            labels.get(ORDERED_BELIEF_AR) == BELIEF_LIKE_ORACLE_SUPPORTED
            and labels.get(COMPOSITION_BIASED_AR) == COMPOSITION_SHORTCUT_SUPPORTED
        )
        passed.append(is_pass)
        summaries.append(
            {
                "path_hint": payload.get("report_type", "learned_synthetic_belief"),
                "labels": labels,
                "passed": is_pass,
            }
        )
    return {
        "status": METHODOLOGY_VALIDATION_PASSED if all(passed) else METHODOLOGY_VALIDATION_FAILED,
        "passed_report_count": sum(1 for value in passed if value),
        "report_count": len(payloads),
        "required_labels": {
            ORDERED_BELIEF_AR: BELIEF_LIKE_ORACLE_SUPPORTED,
            COMPOSITION_BIASED_AR: COMPOSITION_SHORTCUT_SUPPORTED,
        },
        "reports": summaries,
    }


def _explicit_methodology_validation(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize whether explicit belief-state controls validate strict gates."""

    if not payloads:
        return {
            "status": METHODOLOGY_VALIDATION_MISSING,
            "passed_report_count": 0,
            "report_count": 0,
            "required_labels": {
                EXPLICIT_BELIEF_STATE_RNN: BELIEF_LIKE_ORACLE_SUPPORTED,
                COMPOSITION_STATE_RNN: COMPOSITION_SHORTCUT_SUPPORTED,
            },
        }

    passed = []
    summaries = []
    for payload in payloads:
        labels = {
            case.get("mode"): case.get("final_diagnostic_label")
            for case in payload.get("cases", [])
            if isinstance(case, dict)
        }
        is_pass = (
            labels.get(EXPLICIT_BELIEF_STATE_RNN) == BELIEF_LIKE_ORACLE_SUPPORTED
            and labels.get(COMPOSITION_STATE_RNN) == COMPOSITION_SHORTCUT_SUPPORTED
        )
        passed.append(is_pass)
        summaries.append(
            {
                "path_hint": payload.get("report_type", "explicit_belief_state_positive_control"),
                "labels": labels,
                "passed": is_pass,
            }
        )
    return {
        "status": METHODOLOGY_VALIDATION_PASSED if all(passed) else METHODOLOGY_VALIDATION_FAILED,
        "passed_report_count": sum(1 for value in passed if value),
        "report_count": len(payloads),
        "required_labels": {
            EXPLICIT_BELIEF_STATE_RNN: BELIEF_LIKE_ORACLE_SUPPORTED,
            COMPOSITION_STATE_RNN: COMPOSITION_SHORTCUT_SUPPORTED,
        },
        "reports": summaries,
    }


def _latent_protocol_transfer_validation(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize whether k.1 validates protocol transfer beyond poker."""

    if not payloads:
        return {
            "status": METHODOLOGY_VALIDATION_MISSING,
            "passed_report_count": 0,
            "report_count": 0,
            "required_status": "protocol_transfer_supported",
        }

    passed = []
    summaries = []
    for payload in payloads:
        labels = {
            case.get("mode"): case.get("final_diagnostic_label")
            for case in payload.get("cases", [])
            if isinstance(case, dict)
        }
        is_pass = bool(payload.get("protocol_transfer_supported"))
        passed.append(is_pass)
        summaries.append(
            {
                "path_hint": payload.get("report_type", "k.1_latent_type_protocol_transfer"),
                "protocol_transfer_supported": is_pass,
                "learned_positive_capacity_supported": bool(payload.get("learned_positive_capacity_supported")),
                "labels": labels,
            }
        )
    return {
        "status": METHODOLOGY_VALIDATION_PASSED if all(passed) else METHODOLOGY_VALIDATION_FAILED,
        "passed_report_count": sum(1 for value in passed if value),
        "report_count": len(payloads),
        "required_status": "protocol_transfer_supported",
        "reports": summaries,
    }


def _llm_latent_order_status(payloads: list[dict[str, Any]]) -> str:
    """Return the product-summary label for k.2 frozen-LM latent-order audit rows."""

    if not payloads:
        return "missing"
    labels = []
    for payload in payloads:
        for case in payload.get("cases", []):
            if isinstance(case, dict) and case.get("backend") == "hf_causal_lm":
                label = case.get("final_diagnostic_label")
                if label:
                    labels.append(str(label))
    if labels:
        unique = set(labels)
        return labels[0] if len(unique) == 1 else "mixed"
    return "present_no_hf_row"


def _llm_full_sweep_status(payloads: list[dict[str, Any]]) -> str:
    """Return product-summary label for k.2.6 frozen-LM full-sweep aggregates."""

    if not payloads:
        return "missing"
    labels = []
    for payload in payloads:
        label = payload.get("main_sweep_label") or payload.get("final_sweep_label")
        if label:
            labels.append(str(label))
    if labels:
        unique = set(labels)
        return labels[0] if len(unique) == 1 else "mixed"
    return "present_no_sweep_label"


def _combined_llm_status(
    llm_latent_order_payloads: list[dict[str, Any]],
    llm_full_sweep_payloads: list[dict[str, Any]],
) -> str:
    """Prefer Qwen/main full-sweep aggregates when present, else use single reports."""

    full_sweep_status = _llm_full_sweep_status(llm_full_sweep_payloads)
    if full_sweep_status != "missing":
        return full_sweep_status
    return _llm_latent_order_status(llm_latent_order_payloads)


def _cbps_behavior_link_gate(cbps_payload: dict[str, Any] | None) -> str:
    """Return the poker-row behavior-link gate from a CBPS claim ladder."""

    if cbps_payload is None:
        return "missing"
    for row in cbps_payload.get("rows", []):
        if isinstance(row, dict) and row.get("domain") == "limit_holdem":
            return str(row.get("representation_behavior_link_gate", "not_provided"))
    return "not_provided"


def _order_intervention_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Return product-level status for the k.6.0 order-intervention stress report."""

    if payload is None:
        return {
            "status": "missing",
            "claim_scope": "missing",
            "pair_count": 0,
            "seed_count": 0,
        }
    aggregate = payload.get("aggregate", {})
    return {
        "status": str(payload.get("final_diagnostic_label", "present_unknown")),
        "claim_scope": str(payload.get("claim_scope", "not_provided")),
        "pair_count": int(aggregate.get("intervention_pair_count", 0)),
        "seed_count": int(aggregate.get("seed_count", len(payload.get("rows", [])))),
    }


def _range_subspace_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Return product-level status for the k.6.3 range-subspace stress report."""

    if payload is None:
        return {
            "status": "missing",
            "claim_scope": "missing",
            "probe_top10_mean": 0.0,
            "policy_delta_mean": 0.0,
            "value_delta_mean": 0.0,
        }
    metrics = payload.get("aggregate", {}).get("metrics", {})
    return {
        "status": str(payload.get("final_diagnostic_label", "present_unknown")),
        "claim_scope": str(payload.get("claim_scope", "not_provided")),
        "probe_top10_mean": float(metrics.get("probe_top10_accuracy", {}).get("mean", 0.0)),
        "policy_delta_mean": float(metrics.get("range_minus_random_policy_js_delta", {}).get("mean", 0.0)),
        "value_delta_mean": float(metrics.get("range_minus_random_value_delta", {}).get("mean", 0.0)),
    }


def _range_rank_sensitivity_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Return product-level status for the k.6.5 rank-sensitivity stress report."""

    if payload is None:
        return {
            "status": "missing",
            "claim_scope": "missing",
            "rank_count": 0,
            "policy_delta_mean": 0.0,
            "value_delta_mean": 0.0,
        }
    aggregate = payload.get("aggregate", {})
    return {
        "status": str(payload.get("final_diagnostic_label", "present_unknown")),
        "claim_scope": str(payload.get("claim_scope", "not_provided")),
        "rank_count": int(aggregate.get("rank_count", 0)),
        "policy_delta_mean": float(aggregate.get("mean_policy_delta_across_ranks", 0.0)),
        "value_delta_mean": float(aggregate.get("mean_value_delta_across_ranks", 0.0)),
    }


def _final_verdict(
    belief_report: dict[str, Any] | None,
    matched_report: dict[str, Any] | None,
    counterfactual_payloads: list[dict[str, Any]],
    trajectory_payloads: list[dict[str, Any]],
    baseline_payloads: list[dict[str, Any]],
) -> str:
    if counterfactual_payloads and all(
        payload.get("final_diagnostic_label") == ORDER_SENSITIVE_SUPPORTED
        for payload in counterfactual_payloads
    ):
        return ORDER_SENSITIVE_SUPPORTED
    if trajectory_payloads and all(
        payload.get("final_diagnostic_label") == TRAJECTORY_ALIGNED_SUPPORTED
        for payload in trajectory_payloads
    ):
        if baseline_payloads and any(
            payload.get("final_diagnostic_label") == COMPOSITION_SHORTCUT_SUPPORTED
            for payload in baseline_payloads
        ):
            return COMPOSITION_SHORTCUT_SUPPORTED
        return TRAJECTORY_ALIGNED_SUPPORTED
    labels = [payload.get("final_diagnostic_label") for payload in counterfactual_payloads]
    if labels and all(label == INSUFFICIENT_PAIRS for label in labels):
        return INSUFFICIENT_PAIRS

    belief_label = None
    if belief_report is not None and belief_report.get("cases"):
        belief_label = belief_report["cases"][0].get("final_diagnostic_label")
    matched_label = None
    if matched_report is not None and matched_report.get("decision_time_gate") is True:
        matched_label = matched_report.get("final_diagnostic_label")
    if COMPOSITION_SHORTCUT_SUPPORTED in {belief_label, matched_label, *labels}:
        return COMPOSITION_SHORTCUT_SUPPORTED
    return INCONCLUSIVE


def _summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# JorJor-Inspect Audit Summary",
        "",
        "## Executive Summary",
        "",
        "| item | status |",
        "|---|---|",
        f"| Poker verdict | `{payload['final_verdict']}` |",
        f"| Synthetic validation | `{payload['methodology_validation_status']}` |",
        f"| Toy oracle validation | `{payload['toy_oracle_validation_status']}` |",
        f"| Learned synthetic validation | `{payload['learned_methodology_validation_status']}` |",
        f"| Explicit belief-state validation | `{payload['explicit_methodology_validation_status']}` |",
        f"| Non-poker protocol transfer | `{payload['latent_protocol_transfer_status']}` |",
        f"| Frozen-LM latent-order audit | `{payload['llm_latent_order_status']}` |",
        f"| Range-supervised strict gate | `{payload['range_supervised_strict_gate_status']}` |",
        f"| MLP capacity gate | `{payload['probe_capacity_s6_s7_status']}` |",
        f"| CBPS claim ladder | `{payload['cbps_claim_ladder_status']}` |",
        f"| CBPS behavior link | `{payload['cbps_behavior_link_gate']}` |",
        f"| CBPS claim scope | `{payload['cbps_claim_scope']}` |",
        f"| Order-intervention stress | `{payload['order_intervention_status']}` |",
        f"| Order-intervention scope | `{payload['order_intervention_claim_scope']}` |",
        f"| Range-subspace stress | `{payload['range_subspace_status']}` |",
        f"| Range-subspace scope | `{payload['range_subspace_claim_scope']}` |",
        f"| Range-rank sensitivity | `{payload['range_rank_sensitivity_status']}` |",
        f"| Range-rank scope | `{payload['range_rank_sensitivity_claim_scope']}` |",
        f"| Reproduction manifest | `{payload['reproduction_manifest_status']}` |",
        f"| Gaps | `{len(payload['gaps'])}` |",
        "",
        "## Verdict",
        "",
        f"`{payload['final_verdict']}`",
        "",
        "## Sources",
        "",
        "| role | required | present | path |",
        "|---|---|---|---|",
    ]
    for source in payload["source_manifest"]["sources"]:
        lines.append(f"| {source['role']} | {source['required']} | {source['present']} | {source['path']} |")
    lines.extend(["", "## Gaps", ""])
    if payload["gaps"]:
        for gap in payload["gaps"]:
            lines.append(f"- {gap}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## CBPS Claim Ladder",
            "",
            f"Status: `{payload['cbps_claim_ladder_status']}`",
            "",
            f"Behavior link: `{payload['cbps_behavior_link_gate']}`",
            "",
            f"Scope: `{payload['cbps_claim_scope']}`",
            "",
            payload["cbps_paper_safe_claim"] or "No CBPS claim-ladder artifact was provided.",
            "",
            "## Order-Intervention Stress",
            "",
            f"Status: `{payload['order_intervention_status']}`",
            "",
            f"Scope: `{payload['order_intervention_claim_scope']}`",
            "",
            f"Pairs: `{payload['order_intervention_pair_count']}` across `{payload['order_intervention_seed_count']}` seeds.",
            "",
            "This input-level stress evidence bounds pure bag-of-actions invariance but does not upgrade the poker verdict to legal posterior tracking.",
            "",
            "## Range-Subspace Stress",
            "",
            f"Status: `{payload['range_subspace_status']}`",
            "",
            f"Scope: `{payload['range_subspace_claim_scope']}`",
            "",
            f"Probe top10 mean: `{payload['range_subspace_probe_top10_mean']}`",
            "",
            f"Policy delta mean: `{payload['range_subspace_policy_delta_mean']}`",
            "",
            f"Value delta mean: `{payload['range_subspace_value_delta_mean']}`",
            "",
            "This linear subspace stress evidence cautions against interpreting range decodability as behavior mediation.",
            "",
            "## Range-Rank Sensitivity",
            "",
            f"Status: `{payload['range_rank_sensitivity_status']}`",
            "",
            f"Scope: `{payload['range_rank_sensitivity_claim_scope']}`",
            "",
            f"Ranks: `{payload['range_rank_sensitivity_rank_count']}`",
            "",
            f"Mean policy delta across ranks: `{payload['range_rank_sensitivity_policy_delta_mean']}`",
            "",
            f"Mean value delta across ranks: `{payload['range_rank_sensitivity_value_delta_mean']}`",
            "",
            "This stress evidence checks whether the range-subspace result is stable across predeclared ranks.",
            "",
            "## Methodology Validation",
            "",
            f"`{payload['methodology_validation_status']}`",
            "",
            "Synthetic oracle-belief controls validate the diagnostic protocol only; they do not override the poker model verdict.",
            "Non-poker protocol-transfer validation checks whether the audit maps beyond betting histories; it does not upgrade the poker verdict.",
        ]
    )
    lines.extend(
        [
            "",
            "## Claim Rule",
            "",
            "Positive probe results are insufficient evidence for ordered belief tracking unless they survive shortcut and order-counterfactual gates.",
            "",
        ]
    )
    return "\n".join(lines)


def _reproducibility_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Reproducibility",
            "",
            "## Command",
            "",
            "```text",
            payload["command"],
            "```",
            "",
            "## Inputs",
            "",
            f"- inspect_dir: `{payload['inspect_dir']}`",
            f"- counterfactual_dir: `{payload['counterfactual_dir']}`",
            f"- trajectory_dir: `{payload['trajectory_dir']}`",
            f"- baseline_dir: `{payload['baseline_dir']}`",
            f"- synthetic_dir: `{payload['synthetic_dir']}`",
            f"- toy_oracle_dir: `{payload['toy_oracle_dir']}`",
            f"- learned_synthetic_dir: `{payload['learned_synthetic_dir']}`",
            f"- explicit_belief_state_dir: `{payload['explicit_belief_state_dir']}`",
            f"- latent_protocol_dir: `{payload['latent_protocol_dir']}`",
            f"- llm_latent_order_dir: `{payload['llm_latent_order_dir']}`",
            f"- llm_full_sweep_dir: `{payload['llm_full_sweep_dir']}`",
            f"- cbps_claim_ladder_dir: `{payload['cbps_claim_ladder_dir']}`",
            f"- order_intervention_dir: `{payload['order_intervention_dir']}`",
            f"- range_subspace_dir: `{payload['range_subspace_dir']}`",
            f"- range_rank_sensitivity_dir: `{payload['range_rank_sensitivity_dir']}`",
            "",
        ]
    )


def build_audit_report(
    *,
    inspect_dir: Path,
    output_dir: Path,
    counterfactual_dir: Path | None = None,
    trajectory_dir: Path | None = None,
    baseline_dir: Path | None = None,
    synthetic_dir: Path | None = None,
    toy_oracle_dir: Path | None = None,
    learned_synthetic_dir: Path | None = None,
    explicit_belief_state_dir: Path | None = None,
    latent_protocol_dir: Path | None = None,
    llm_latent_order_dir: Path | None = None,
    llm_full_sweep_dir: Path | None = None,
    cbps_claim_ladder_dir: Path | None = None,
    order_intervention_dir: Path | None = None,
    range_subspace_dir: Path | None = None,
    range_rank_sensitivity_dir: Path | None = None,
    reproduction_manifest_path: Path | None = None,
    strict_table_dir: Path | None = None,
    command: str = "jorjor-inspect audit",
) -> dict[str, Any]:
    """Assemble a product-facing belief-vs-shortcut audit report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    gaps = []
    for relative in REQUIRED_INSPECT_FILES:
        path = inspect_dir / relative
        entry = _source_entry(path, required=True, role=relative)
        sources.append(entry)
        if not entry["present"]:
            gaps.append(f"missing required inspect artifact: {relative}")

    counterfactual_payloads = []
    for path in _optional_reports(counterfactual_dir, "order_counterfactual_report.json"):
        sources.append(_source_entry(path, required=False, role="order_counterfactual_report.json"))
        counterfactual_payloads.append(_read_json(path))
        md_path = path.with_suffix(".md")
        if md_path.exists():
            sources.append(_source_entry(md_path, required=False, role="order_counterfactual_report.md"))
    trajectory_payloads = []
    for path in _optional_reports(trajectory_dir, "prefix_trajectory_report.json"):
        sources.append(_source_entry(path, required=False, role="prefix_trajectory_report.json"))
        trajectory_payloads.append(_read_json(path))
        md_path = path.with_suffix(".md")
        if md_path.exists():
            sources.append(_source_entry(md_path, required=False, role="prefix_trajectory_report.md"))
    baseline_payloads = []
    for path in _optional_reports(baseline_dir, "prefix_composition_baseline_report.json"):
        sources.append(_source_entry(path, required=False, role="prefix_composition_baseline_report.json"))
        baseline_payloads.append(_read_json(path))
        md_path = path.with_suffix(".md")
        if md_path.exists():
            sources.append(_source_entry(md_path, required=False, role="prefix_composition_baseline_report.md"))
    synthetic_payloads = []
    for path in _optional_reports(synthetic_dir, "synthetic_oracle_belief_report.json"):
        sources.append(_source_entry(path, required=False, role="synthetic_oracle_belief_report.json"))
        synthetic_payloads.append(_read_json(path))
        md_path = path.with_suffix(".md")
        if md_path.exists():
            sources.append(_source_entry(md_path, required=False, role="synthetic_oracle_belief_report.md"))
    toy_oracle_payloads = []
    for path in _optional_reports(toy_oracle_dir, "toy_imperfect_info_oracle_report.json"):
        sources.append(_source_entry(path, required=False, role="toy_imperfect_info_oracle_report.json"))
        toy_oracle_payloads.append(_read_json(path))
        md_path = path.with_suffix(".md")
        if md_path.exists():
            sources.append(_source_entry(md_path, required=False, role="toy_imperfect_info_oracle_report.md"))
    learned_synthetic_payloads = []
    for path in _optional_reports(learned_synthetic_dir, "learned_synthetic_belief_report.json"):
        sources.append(_source_entry(path, required=False, role="learned_synthetic_belief_report.json"))
        learned_synthetic_payloads.append(_read_json(path))
        md_path = path.with_suffix(".md")
        if md_path.exists():
            sources.append(_source_entry(md_path, required=False, role="learned_synthetic_belief_report.md"))
    explicit_payloads = []
    for path in _optional_reports(explicit_belief_state_dir, "explicit_belief_state_report.json"):
        sources.append(_source_entry(path, required=False, role="explicit_belief_state_report.json"))
        explicit_payloads.append(_read_json(path))
        md_path = path.with_suffix(".md")
        if md_path.exists():
            sources.append(_source_entry(md_path, required=False, role="explicit_belief_state_report.md"))
    latent_protocol_payloads = []
    for path in _optional_reports(latent_protocol_dir, "k.1_latent_type_protocol_transfer_report.json"):
        sources.append(_source_entry(path, required=False, role="k.1_latent_type_protocol_transfer_report.json"))
        latent_protocol_payloads.append(_read_json(path))
        md_path = path.with_suffix(".md")
        if md_path.exists():
            sources.append(_source_entry(md_path, required=False, role="k.1_latent_type_protocol_transfer_report.md"))
        manifest_path = path.parent / "k.1_protocol_transfer_manifest.json"
        if manifest_path.exists():
            sources.append(_source_entry(manifest_path, required=False, role="k.1_protocol_transfer_manifest.json"))
    llm_latent_order_payloads = []
    for path in _optional_reports(llm_latent_order_dir, "k.2_llm_latent_order_report.json"):
        sources.append(_source_entry(path, required=False, role="k.2_llm_latent_order_report.json"))
        llm_latent_order_payloads.append(_read_json(path))
        md_path = path.with_suffix(".md")
        if md_path.exists():
            sources.append(_source_entry(md_path, required=False, role="k.2_llm_latent_order_report.md"))
        table_path = path.parent / "k.2_llm_latent_order_table.md"
        if table_path.exists():
            sources.append(_source_entry(table_path, required=False, role="k.2_llm_latent_order_table.md"))
        manifest_path = path.parent / "k.2_llm_latent_order_manifest.json"
        if manifest_path.exists():
            sources.append(_source_entry(manifest_path, required=False, role="k.2_llm_latent_order_manifest.json"))
    llm_full_sweep_payloads = []
    for path in _optional_reports(llm_full_sweep_dir, "k.2.6_llm_full_sweep_report.json"):
        sources.append(_source_entry(path, required=False, role="k.2.6_llm_full_sweep_report.json"))
        llm_full_sweep_payloads.append(_read_json(path))
        md_path = path.with_suffix(".md")
        if md_path.exists():
            sources.append(_source_entry(md_path, required=False, role="k.2.6_llm_full_sweep_report.md"))
        manifest_path = path.parent / "k.2.6_llm_full_sweep_manifest.json"
        if manifest_path.exists():
            sources.append(_source_entry(manifest_path, required=False, role="k.2.6_llm_full_sweep_manifest.json"))
    cbps_payload = None
    if cbps_claim_ladder_dir is not None:
        cbps_path = cbps_claim_ladder_dir / "k.5.3_cbps_claim_ladder.json"
        cbps_source = _source_entry(cbps_path, required=False, role="cbps_claim_ladder")
        sources.append(cbps_source)
        if cbps_source["present"]:
            cbps_payload = _read_json(cbps_path)
            for filename, role in (
                ("k.5.3_cbps_claim_ladder.md", "cbps_claim_ladder_md"),
                ("k.5.3_cbps_manifest.json", "cbps_claim_ladder_manifest"),
                ("paper_table_cbps_claim_ladder.md", "cbps_claim_ladder_paper_table"),
            ):
                path = cbps_claim_ladder_dir / filename
                if path.exists():
                    sources.append(_source_entry(path, required=False, role=role))
        else:
            gaps.append("requested CBPS claim-ladder report is missing")
    order_intervention_payload = None
    if order_intervention_dir is not None:
        order_path = order_intervention_dir / "k.6.0_order_intervention_report.json"
        order_source = _source_entry(order_path, required=False, role="order_intervention_report")
        sources.append(order_source)
        if order_source["present"]:
            order_intervention_payload = _read_json(order_path)
            for filename, role in (
                ("k.6.0_order_intervention_report.md", "order_intervention_report_md"),
                ("k.6.0_order_intervention_manifest.json", "order_intervention_manifest"),
            ):
                path = order_intervention_dir / filename
                if path.exists():
                    sources.append(_source_entry(path, required=False, role=role))
        else:
            gaps.append("requested order-intervention report is missing")
    range_subspace_payload = None
    if range_subspace_dir is not None:
        range_path = range_subspace_dir / "k.6.3_range_subspace_ablation_report.json"
        range_source = _source_entry(range_path, required=False, role="range_subspace_report")
        sources.append(range_source)
        if range_source["present"]:
            range_subspace_payload = _read_json(range_path)
            for filename, role in (
                ("k.6.3_range_subspace_ablation_report.md", "range_subspace_report_md"),
                ("k.6.3_range_subspace_ablation_manifest.json", "range_subspace_manifest"),
            ):
                path = range_subspace_dir / filename
                if path.exists():
                    sources.append(_source_entry(path, required=False, role=role))
        else:
            gaps.append("requested range-subspace report is missing")
    range_rank_sensitivity_payload = None
    if range_rank_sensitivity_dir is not None:
        rank_path = range_rank_sensitivity_dir / "k.6.5_range_subspace_rank_sensitivity_report.json"
        rank_source = _source_entry(rank_path, required=False, role="range_rank_sensitivity_report")
        sources.append(rank_source)
        if rank_source["present"]:
            range_rank_sensitivity_payload = _read_json(rank_path)
            for filename, role in (
                ("k.6.5_range_subspace_rank_sensitivity_report.md", "range_rank_sensitivity_report_md"),
                ("k.6.5_range_subspace_rank_sensitivity_manifest.json", "range_rank_sensitivity_manifest"),
            ):
                path = range_rank_sensitivity_dir / filename
                if path.exists():
                    sources.append(_source_entry(path, required=False, role=role))
        else:
            gaps.append("requested range-rank-sensitivity report is missing")
    reproduction_manifest = None
    reproduction_manifest_status = "missing"
    if reproduction_manifest_path is not None:
        reproduction_entry = _source_entry(
            reproduction_manifest_path,
            required=False,
            role="one_command_reproduction_manifest",
        )
        sources.append(reproduction_entry)
        if reproduction_entry["present"]:
            reproduction_manifest = _read_json(reproduction_manifest_path)
            reproduction_manifest_status = "present"
            md_path = reproduction_manifest_path.with_suffix(".md")
            if md_path.exists():
                sources.append(_source_entry(md_path, required=False, role="one_command_reproduction_manifest_md"))
        else:
            reproduction_manifest_status = "missing_requested"
            gaps.append("requested reproduction manifest is missing")

    belief_report = _read_json(inspect_dir / "belief_shortcut_report.json") if (inspect_dir / "belief_shortcut_report.json").exists() else None
    matched_report = _read_json(inspect_dir / "matched_composition_report.json") if (inspect_dir / "matched_composition_report.json").exists() else None
    claim_gate = _read_json(inspect_dir / "claim_gate_matrix.json") if (inspect_dir / "claim_gate_matrix.json").exists() else None
    if not counterfactual_payloads:
        gaps.append("optional order-counterfactual report is missing")
    if not trajectory_payloads:
        gaps.append("optional prefix trajectory report is missing")
    if not baseline_payloads:
        gaps.append("optional prefix composition baseline report is missing")
    if not synthetic_payloads:
        gaps.append("optional synthetic oracle-belief validation report is missing")
    if toy_oracle_dir is not None and not toy_oracle_payloads:
        gaps.append("requested toy imperfect-information oracle validation report is missing")
    if not learned_synthetic_payloads:
        gaps.append("optional learned synthetic belief validation report is missing")
    if not explicit_payloads:
        gaps.append("optional explicit belief-state validation report is missing")
    strict_extensions: dict[str, dict[str, Any] | None] = {}
    for role, filenames in STRICT_EXTENSION_SPECS:
        payload, source = _optional_extension(inspect_dir, strict_table_dir, role, filenames)
        strict_extensions[role] = payload
        sources.append(source)
        if payload is None:
            gaps.append(f"optional {role} report is missing")

    final_verdict = _final_verdict(
        belief_report,
        matched_report,
        counterfactual_payloads,
        trajectory_payloads,
        baseline_payloads,
    )
    methodology_validation = _methodology_validation([*synthetic_payloads, *toy_oracle_payloads])
    toy_oracle_validation = _methodology_validation(toy_oracle_payloads)
    learned_methodology_validation = _learned_methodology_validation(learned_synthetic_payloads)
    explicit_methodology_validation = _explicit_methodology_validation(explicit_payloads)
    latent_protocol_validation = _latent_protocol_transfer_validation(latent_protocol_payloads)
    llm_latent_order_status = _combined_llm_status(llm_latent_order_payloads, llm_full_sweep_payloads)
    source_manifest = {
        "schema_version": "1.0",
        "sources": sources,
    }
    tables = {
        "belief_shortcut": belief_report,
        "matched_composition": matched_report,
        "claim_gate_matrix": claim_gate,
        "order_counterfactual": counterfactual_payloads,
        "prefix_trajectory": trajectory_payloads,
        "prefix_composition_baseline": baseline_payloads,
        "synthetic_oracle_belief": synthetic_payloads,
        "toy_imperfect_info_oracle": toy_oracle_payloads,
        "learned_synthetic_belief": learned_synthetic_payloads,
        "explicit_belief_state": explicit_payloads,
        "latent_protocol_transfer": latent_protocol_payloads,
        "llm_latent_order": llm_latent_order_payloads,
        "llm_full_sweep": llm_full_sweep_payloads,
        "cbps_claim_ladder": cbps_payload,
        "order_intervention": order_intervention_payload,
        "range_subspace": range_subspace_payload,
        "range_rank_sensitivity": range_rank_sensitivity_payload,
        "range_supervised_strict_gate": strict_extensions["range_supervised_strict_gate"],
        "probe_capacity_s6_s7": strict_extensions["probe_capacity_s6_s7"],
        "one_command_reproduction_manifest": reproduction_manifest,
    }
    range_supervised_status = (
        strict_extensions["range_supervised_strict_gate"].get("final_diagnostic_label")
        if strict_extensions["range_supervised_strict_gate"] is not None
        else "missing"
    ) or "present_unknown"
    probe_capacity_status = (
        strict_extensions["probe_capacity_s6_s7"].get("final_verdict")
        if strict_extensions["probe_capacity_s6_s7"] is not None
        else "missing"
    ) or "present_unknown"
    cbps_status = cbps_payload.get("final_label") if cbps_payload is not None else "missing"
    cbps_scope = cbps_payload.get("claim_scope") if cbps_payload is not None else "missing"
    cbps_claim = cbps_payload.get("paper_safe_claim") if cbps_payload is not None else None
    cbps_behavior_link = _cbps_behavior_link_gate(cbps_payload)
    order_summary = _order_intervention_summary(order_intervention_payload)
    range_summary = _range_subspace_summary(range_subspace_payload)
    range_rank_summary = _range_rank_sensitivity_summary(range_rank_sensitivity_payload)
    payload = {
        "version": "1.25",
        "product": "JorJor-Inspect",
        "inspect_dir": str(inspect_dir),
        "counterfactual_dir": str(counterfactual_dir) if counterfactual_dir is not None else None,
        "trajectory_dir": str(trajectory_dir) if trajectory_dir is not None else None,
        "baseline_dir": str(baseline_dir) if baseline_dir is not None else None,
        "synthetic_dir": str(synthetic_dir) if synthetic_dir is not None else None,
        "toy_oracle_dir": str(toy_oracle_dir) if toy_oracle_dir is not None else None,
        "learned_synthetic_dir": str(learned_synthetic_dir) if learned_synthetic_dir is not None else None,
        "explicit_belief_state_dir": str(explicit_belief_state_dir) if explicit_belief_state_dir is not None else None,
        "latent_protocol_dir": str(latent_protocol_dir) if latent_protocol_dir is not None else None,
        "llm_latent_order_dir": str(llm_latent_order_dir) if llm_latent_order_dir is not None else None,
        "llm_full_sweep_dir": str(llm_full_sweep_dir) if llm_full_sweep_dir is not None else None,
        "cbps_claim_ladder_dir": str(cbps_claim_ladder_dir) if cbps_claim_ladder_dir is not None else None,
        "order_intervention_dir": str(order_intervention_dir) if order_intervention_dir is not None else None,
        "range_subspace_dir": str(range_subspace_dir) if range_subspace_dir is not None else None,
        "range_rank_sensitivity_dir": str(range_rank_sensitivity_dir) if range_rank_sensitivity_dir is not None else None,
        "reproduction_manifest_path": str(reproduction_manifest_path) if reproduction_manifest_path is not None else None,
        "strict_table_dir": str(strict_table_dir) if strict_table_dir is not None else None,
        "reproduction_manifest_status": reproduction_manifest_status,
        "reproduction_manifest": reproduction_manifest,
        "command": command,
        "final_verdict": final_verdict,
        "methodology_validation_status": methodology_validation["status"],
        "methodology_validation": methodology_validation,
        "toy_oracle_validation_status": toy_oracle_validation["status"],
        "toy_oracle_validation": toy_oracle_validation,
        "learned_methodology_validation_status": learned_methodology_validation["status"],
        "learned_methodology_validation": learned_methodology_validation,
        "explicit_methodology_validation_status": explicit_methodology_validation["status"],
        "explicit_methodology_validation": explicit_methodology_validation,
        "latent_protocol_transfer_status": latent_protocol_validation["status"],
        "latent_protocol_transfer_validation": latent_protocol_validation,
        "llm_latent_order_status": llm_latent_order_status,
        "cbps_claim_ladder_status": cbps_status,
        "cbps_behavior_link_gate": cbps_behavior_link,
        "cbps_claim_scope": cbps_scope,
        "cbps_paper_safe_claim": cbps_claim,
        "order_intervention_status": order_summary["status"],
        "order_intervention_claim_scope": order_summary["claim_scope"],
        "order_intervention_pair_count": order_summary["pair_count"],
        "order_intervention_seed_count": order_summary["seed_count"],
        "range_subspace_status": range_summary["status"],
        "range_subspace_claim_scope": range_summary["claim_scope"],
        "range_subspace_probe_top10_mean": range_summary["probe_top10_mean"],
        "range_subspace_policy_delta_mean": range_summary["policy_delta_mean"],
        "range_subspace_value_delta_mean": range_summary["value_delta_mean"],
        "range_rank_sensitivity_status": range_rank_summary["status"],
        "range_rank_sensitivity_claim_scope": range_rank_summary["claim_scope"],
        "range_rank_sensitivity_rank_count": range_rank_summary["rank_count"],
        "range_rank_sensitivity_policy_delta_mean": range_rank_summary["policy_delta_mean"],
        "range_rank_sensitivity_value_delta_mean": range_rank_summary["value_delta_mean"],
        "range_supervised_strict_gate_status": range_supervised_status,
        "probe_capacity_s6_s7_status": probe_capacity_status,
        "source_manifest": source_manifest,
        "gaps": gaps,
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})

    write_json_artifact(output_dir / "diagnostic_report.json", payload)
    write_json_artifact(output_dir / "source_manifest.json", source_manifest)
    write_json_artifact(output_dir / "tables.json", tables)
    write_json_artifact(output_dir / "gaps.json", {"gaps": gaps})
    write_json_artifact(
        output_dir / "artifact_index.json",
        {
            "schema_version": "1.0",
            "artifacts": [
                "diagnostic_report.json",
                "diagnostic_report.md",
                "source_manifest.json",
                "tables.json",
                "gaps.json",
                "reproducibility.md",
            ],
        },
    )
    (output_dir / "diagnostic_report.md").write_text(_summary_markdown(payload), encoding="utf-8")
    (output_dir / "reproducibility.md").write_text(_reproducibility_markdown(payload), encoding="utf-8")
    claim_gate_md = inspect_dir / "claim_gate_matrix.md"
    if claim_gate_md.exists():
        (output_dir / "claim_gate_matrix.md").write_text(claim_gate_md.read_text(encoding="utf-8"), encoding="utf-8")
    return payload
