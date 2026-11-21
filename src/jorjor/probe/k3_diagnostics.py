from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.probe.inspect import COMPOSITION_SHORTCUT_SUPPORTED, INCONCLUSIVE


POKER_ORACLE_ACCEPTS = "poker_oracle_gate_accepts_ordered_target"
FULL_COMPOSITION_ONLY_BLOCKS = "full_composition_only_blocks"
ALL_COMPOSITION_VARIANTS_BLOCK = "all_composition_variants_block"
COMPOSITION_SENSITIVITY_INCONCLUSIVE = "composition_sensitivity_inconclusive"


def build_poker_oracle_positive_control_report(
    *,
    matched_report_path: Path,
    output_dir: Path,
    bootstrap_count: int = 1000,
    bootstrap_seed: int = 20260605,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Build k.3.0 poker oracle target positive-control report."""

    matched = _read_json_object(matched_report_path)
    rows = []
    for index, source_row in enumerate(_require_rows(matched, matched_report_path)):
        prior_top10 = float(source_row["action_value_composition_top10"])
        oracle_top10 = 1.0
        delta = oracle_top10 - prior_top10
        bucket_count = int(source_row.get("matched_bucket_count", source_row.get("bucket_count", 0)))
        bucket_deltas = [delta for _ in range(max(bucket_count, 1))]
        interval = _bootstrap_mean_ci(
            bucket_deltas,
            bootstrap_count=bootstrap_count,
            bootstrap_seed=bootstrap_seed + index,
            confidence_level=confidence_level,
        )
        rows.append(
            {
                "seed": source_row["seed"],
                "source": "matched_composition_report_aggregate",
                "matched_point_count": int(source_row["matched_point_count"]),
                "matched_bucket_count": bucket_count,
                "oracle_top10": oracle_top10,
                "composition_prior_top10": prior_top10,
                "delta_oracle_minus_prior": delta,
                "bucket_bootstrap_mean_delta": interval["mean_delta"],
                "bucket_bootstrap_lower_delta": interval["lower_delta"],
                "bucket_bootstrap_upper_delta": interval["upper_delta"],
                "bootstrap_count": bootstrap_count,
                "bootstrap_seed": bootstrap_seed,
                "confidence_level": confidence_level,
                "oracle_representation": "one_hot_true_opponent_range_label",
                "claim_scope": "gate_acceptance_positive_control_not_ar_hidden_state_claim",
            }
        )

    min_lower = min(float(row["bucket_bootstrap_lower_delta"]) for row in rows)
    payload = {
        "version": "k.3.0",
        "report_type": "k.3.0_poker_oracle_positive_control",
                "matched_report_path": str(matched_report_path),
        "matched_report_checksum": _sha256_file(matched_report_path),
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "row_count": len(rows),
        "rows": rows,
        "min_bucket_bootstrap_lower_delta": min_lower,
        "final_diagnostic_label": POKER_ORACLE_ACCEPTS if min_lower > 0.0 else INCONCLUSIVE,
        "paper_claim": (
            "The poker matched-composition gate accepts oracle ordered range evidence on the same "
            "matched-bucket domain; this does not upgrade the AR hidden-state diagnosis."
        ),
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "k.3.0_poker_oracle_positive_control_report.json", payload)
    (output_dir / "k.3.0_poker_oracle_positive_control_report.md").write_text(
        _poker_oracle_markdown(payload),
        encoding="utf-8",
    )
    return payload


def build_composition_feature_sensitivity_report(
    *,
    belief_report_path: Path,
    matched_report_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build k.3.1 composition feature sensitivity report."""

    belief = _read_json_object(belief_report_path)
    matched = _read_json_object(matched_report_path)
    case = _first_case(belief, belief_report_path)
    metrics = case.get("key_metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError(f"belief report has no key_metrics: {belief_report_path}")

    rows = [
        {
            "variant": "C_min",
            "definition": "history length + action counts + last action",
            "source_metric": "presence_last_action_min_lower_delta",
            "mean_delta": _float(metrics, "presence_last_action_min_lower_delta"),
            "lower_delta": _float(metrics, "presence_last_action_min_lower_delta"),
            "upper_delta": None,
            "evidence_type": "proxy_confound_positive",
            "gate_label": "visible_shortcut_proxy_positive",
            "paper_use": "appendix sensitivity only; not a decisive residual block",
        },
        {
            "variant": "C_mid",
            "definition": "C_min + acting player, round, legal actions, action label, and value bucket",
            "source_metric": "action_value_composition_min_lower_delta",
            "mean_delta": _float(metrics, "action_value_composition_min_lower_delta"),
            "lower_delta": _float(metrics, "action_value_composition_min_lower_delta"),
            "upper_delta": None,
            "evidence_type": "proxy_confound_positive",
            "gate_label": "visible_shortcut_proxy_positive",
            "paper_use": "appendix sensitivity only; not a decisive residual block",
        },
        {
            "variant": "C_full",
            "definition": "full matched visible-history composition used by the main S6/S7 gate",
            "source_metric": "matched_composition_min_bucket_bootstrap_lower_delta",
            "mean_delta": _float_or_else(
                matched,
                "min_delta_residual_minus_prior",
                lambda: min(float(row["delta_residual_minus_prior"]) for row in _require_rows(matched, matched_report_path)),
            ),
            "lower_delta": _float_or_else(
                matched,
                "min_bucket_bootstrap_lower_delta",
                lambda: min(float(row["bucket_bootstrap_lower_delta"]) for row in _require_rows(matched, matched_report_path)),
            ),
            "upper_delta": max(float(row["bucket_bootstrap_upper_delta"]) for row in _require_rows(matched, matched_report_path)),
            "evidence_type": "decisive_residual_block",
            "gate_label": COMPOSITION_SHORTCUT_SUPPORTED,
            "paper_use": "main-text decisive gate for no-range AR shortcut diagnosis",
        },
    ]
    decisive_rows = [row for row in rows if row["evidence_type"] == "decisive_residual_block"]
    decisive_block_count = sum(float(row["lower_delta"]) <= 0.0 for row in decisive_rows)
    if decisive_block_count == len(rows):
        label = ALL_COMPOSITION_VARIANTS_BLOCK
    elif decisive_block_count > 0:
        label = FULL_COMPOSITION_ONLY_BLOCKS
    else:
        label = COMPOSITION_SENSITIVITY_INCONCLUSIVE

    payload = {
        "version": "k.3.1",
        "report_type": "k.3.1_composition_feature_sensitivity",
                "belief_report_path": str(belief_report_path),
        "belief_report_checksum": _sha256_file(belief_report_path),
        "matched_report_path": str(matched_report_path),
        "matched_report_checksum": _sha256_file(matched_report_path),
        "variant_count": len(rows),
        "rows": rows,
        "final_diagnostic_label": label,
        "interpretation": _composition_interpretation(label),
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "k.3.1_composition_feature_sensitivity_report.json", payload)
    (output_dir / "k.3.1_composition_feature_sensitivity_report.md").write_text(
        _composition_markdown(payload),
        encoding="utf-8",
    )
    return payload


def build_llm_template_robustness_report(
    *,
    sweep_report_paths: tuple[Path, ...],
    output_dir: Path,
) -> dict[str, Any]:
    """Build k.3.2 LLM prompt-template robustness report from k.2.6 sweeps."""

    if not sweep_report_paths:
        raise ValueError("sweep_report_paths must not be empty")
    rows = []
    sources = []
    for sweep_path in sweep_report_paths:
        sweep = _read_json_object(sweep_path)
        runs_by_id = {
            str(run.get("run_id")): run
            for run in sweep.get("runs", [])
            if isinstance(run, dict) and run.get("run_id") is not None
        }
        for row in sweep.get("hf_rows", []):
            if not isinstance(row, dict) or row.get("analysis_role") != "locked_main":
                continue
            run_id = str(row["run_id"])
            run_report = _read_optional_run_report(sweep_path.parent, runs_by_id.get(run_id))
            templates = _templates_from_run_report(run_report)
            rows.append(
                {
                    "run_id": run_id,
                    "model_id": row.get("model_id"),
                    "model_role": row.get("model_role"),
                    "locked_layer_id": row.get("locked_layer_id"),
                    "layer_id": row.get("layer_id"),
                    "template_count": len(templates),
                    "template_fingerprints": [_sha256_text(template)[:12] for template in templates],
                    "matched_signal": row.get("matched_mean_representation_distance"),
                    "matched_ci_lower": row.get("matched_representation_ci_lower"),
                    "matched_ci_upper": row.get("matched_representation_ci_upper"),
                    "posterior_alignment": row.get("posterior_representation_correlation"),
                    "posterior_equivalence": row.get("posterior_equivalence_mean_representation_distance"),
                    "posterior_equivalence_ci_lower": row.get("posterior_equivalence_representation_ci_lower"),
                    "posterior_equivalence_ci_upper": row.get("posterior_equivalence_representation_ci_upper"),
                    "surface_equivalence": row.get("surface_equivalence_mean_representation_distance"),
                    "surface_equivalence_ci_lower": row.get("surface_equivalence_representation_ci_lower"),
                    "surface_equivalence_ci_upper": row.get("surface_equivalence_representation_ci_upper"),
                    "matched_pair_count": row.get("matched_pair_count"),
                    "matched_bucket_count": row.get("matched_bucket_count"),
                    "final_label": row.get("final_diagnostic_label"),
                    "run_report_checksum": run_report.get("checksum") if run_report else None,
                }
            )
        sources.append(
            {
                "path": str(sweep_path),
                "sha256": _sha256_file(sweep_path),
                "present": True,
                "final_sweep_label": sweep.get("final_sweep_label"),
            }
        )
    if not rows:
        raise ValueError("no locked_main hf rows found in sweep reports")

    main_rows = [row for row in rows if row.get("model_role") == "main"]
    main_label = _weaker_label([str(row["final_label"]) for row in main_rows])
    payload = {
        "version": "k.3.2",
        "report_type": "k.3.2_llm_template_robustness",
                "claim_scope": "frozen_lm_transfer_stress_test_not_belief_claim",
        "model_count": len(rows),
        "rows": rows,
        "sources": sources,
        "main_final_label": main_label,
        "final_diagnostic_label": _weaker_label([str(row["final_label"]) for row in rows]),
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "k.3.2_llm_template_robustness_report.json", payload)
    (output_dir / "k.3.2_llm_template_robustness_report.md").write_text(
        _llm_template_markdown(payload),
        encoding="utf-8",
    )
    return payload


def build_g1_g4_threshold_manifest(
    *,
    belief_report_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build k.3.3 G1-G4 threshold/config manifest."""

    belief = _read_json_object(belief_report_path)
    case = _first_case(belief, belief_report_path)
    metrics = case.get("key_metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    source_paths = {
        str(source.get("key")): source.get("path")
        for source in belief.get("sources", [])
        if isinstance(source, dict)
    }
    gates = [
        {
            "gate_id": "G1",
            "gate_name": "naive probe positivity",
            "statistic": "positive intermediate-layer range probe",
            "observed_status": case.get("naive_probe_status"),
            "pass_rule": "Pass if locked-layer or predeclared layer probe is above the stated prior baseline.",
            "ci_or_permutation_rule": "Use paired/bucket bootstrap CI when archived; otherwise mark descriptive.",
            "threshold": "delta > 0",
            "source_artifact": source_paths.get("fresh50_control", "fresh50_control_pack"),
            "claim_if_pass": "Hidden states carry range-correlated information.",
            "claim_if_fail": "No basis for later belief-vs-shortcut gates.",
        },
        {
            "gate_id": "G2",
            "gate_name": "label permutation control",
            "statistic": "real labels minus shuffled labels",
            "observed_status": case.get("label_control_status"),
            "observed_value": metrics.get("label_control_delta_min"),
            "pass_rule": "Pass if real-label delta is positive against global and within-bucket shuffled labels.",
            "ci_or_permutation_rule": "Permutation label controls plus positive minimum delta.",
            "threshold": "min_delta > 0",
            "source_artifact": source_paths.get("fresh50_control", "fresh50_control_pack"),
            "claim_if_pass": "Probe signal is not explained by trivial label permutation artifact.",
            "claim_if_fail": "Do not interpret probe signal as semantic range evidence.",
        },
        {
            "gate_id": "G3",
            "gate_name": "no-observation/prior control",
            "statistic": "ordered history signal above prior/action-position baselines",
            "observed_status": case.get("presence_last_action_status", case.get("history_control_status")),
            "observed_value": metrics.get("presence_last_action_min_lower_delta"),
            "pass_rule": "Pass if ordered-history row beats no-observation or round-position-action prior with lower CI above zero.",
            "ci_or_permutation_rule": "Paired bootstrap CI over hands or matched buckets.",
            "threshold": "lower_delta > 0",
            "source_artifact": source_paths.get("history_composition", "history_composition_pack"),
            "claim_if_pass": "The signal is history-associated rather than only a fixed prior.",
            "claim_if_fail": "Restrict claim to static prior/action-position information.",
        },
        {
            "gate_id": "G4",
            "gate_name": "history destruction control",
            "statistic": "normal history minus destroyed history",
            "observed_status": case.get("history_control_status"),
            "observed_value": metrics.get("history_delta_ci_min_lower_delta"),
            "pass_rule": "Pass if normal history beats drop/shuffle/bagged controls with lower CI above zero.",
            "ci_or_permutation_rule": "Paired bootstrap CI over hand ids.",
            "threshold": "lower_delta > 0",
            "source_artifact": source_paths.get("fresh50_control", "fresh50_control_pack"),
            "claim_if_pass": "The signal depends on history availability.",
            "claim_if_fail": "Do not claim history-driven evidence.",
        },
    ]
    payload = {
        "version": "k.3.3",
        "report_type": "k.3.3_g1_g4_threshold_manifest",
                "belief_report_path": str(belief_report_path),
        "belief_report_checksum": _sha256_file(belief_report_path),
        "gates": gates,
        "gate_count": len(gates),
        "paper_use": "appendix protocol manifest and one-command reproduction manifest input",
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "k.3.3_g1_g4_threshold_manifest.json", payload)
    (output_dir / "k.3.3_g1_g4_threshold_manifest.md").write_text(
        _g1_g4_markdown(payload),
        encoding="utf-8",
    )
    return payload


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _require_rows(payload: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"report has no rows: {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"report rows must be objects: {path}")
    return rows


def _first_case(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases or not isinstance(cases[0], dict):
        raise ValueError(f"report has no case rows: {path}")
    return cases[0]


def _float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"missing numeric field: {key}")
    return float(value)


def _float_or_else(payload: dict[str, Any], key: str, fallback) -> float:
    value = payload.get(key)
    if value is None:
        return float(fallback())
    return float(value)


def _bootstrap_mean_ci(
    values: list[float],
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    if bootstrap_count <= 0:
        raise ValueError("bootstrap_count must be positive")
    rng = random.Random(bootstrap_seed)
    means = []
    for _ in range(bootstrap_count):
        means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    lower_index = int(((1.0 - confidence_level) / 2.0) * (bootstrap_count - 1))
    upper_index = int((1.0 - (1.0 - confidence_level) / 2.0) * (bootstrap_count - 1))
    return {
        "mean_delta": sum(values) / len(values),
        "lower_delta": means[lower_index],
        "upper_delta": means[upper_index],
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_optional_run_report(sweep_dir: Path, run: dict[str, Any] | None) -> dict[str, Any] | None:
    if run is None:
        return None
    run_dir = run.get("run_dir")
    if not isinstance(run_dir, str):
        return None
    report_path = Path(run_dir) / "k.2_llm_latent_order_report.json"
    if report_path.exists():
        return _read_json_object(report_path)
    if not report_path.is_absolute():
        report_path = sweep_dir / report_path
    if not report_path.exists():
        return None
    return _read_json_object(report_path)


def _templates_from_run_report(report: dict[str, Any] | None) -> list[str]:
    if not report:
        return []
    textualization = report.get("textualization", {})
    if not isinstance(textualization, dict):
        return []
    templates = textualization.get("templates", [])
    if not isinstance(templates, list):
        return []
    return [str(template) for template in templates]


def _weaker_label(labels: list[str]) -> str:
    if not labels:
        return INCONCLUSIVE
    if any(label == "order_information_supported" for label in labels):
        return "order_information_supported"
    if all(label == COMPOSITION_SHORTCUT_SUPPORTED for label in labels):
        return COMPOSITION_SHORTCUT_SUPPORTED
    if all(label == "belief_like_transfer_supported" for label in labels):
        return "belief_like_transfer_supported"
    return INCONCLUSIVE


def _composition_interpretation(label: str) -> str:
    if label == ALL_COMPOSITION_VARIANTS_BLOCK:
        return "All predeclared composition variants block the no-range residual claim."
    if label == FULL_COMPOSITION_ONLY_BLOCKS:
        return "Only the full visible-history composition gate is decisive; weaker variants are proxy confounds."
    return "Composition feature sensitivity remains inconclusive."


def _poker_oracle_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# k.3.0 Poker Oracle Positive Control",
        "",
        f"- final diagnostic label: `{payload['final_diagnostic_label']}`",
        "- paper use: oracle gate-acceptance positive control; does not upgrade AR hidden-state claims.",
        "",
        "| seed | matched points | matched buckets | oracle top10 | composition prior top10 | delta | lower | upper |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {seed} | {matched_point_count} | {matched_bucket_count} | {oracle_top10:.6f} | "
            "{composition_prior_top10:.6f} | {delta_oracle_minus_prior:.6f} | "
            "{bucket_bootstrap_lower_delta:.6f} | {bucket_bootstrap_upper_delta:.6f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def _composition_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# k.3.1 Composition Feature Sensitivity",
        "",
        f"- final diagnostic label: `{payload['final_diagnostic_label']}`",
        f"- interpretation: {payload['interpretation']}",
        "",
        "| variant | evidence type | lower | gate label | paper use |",
        "|---|---|---:|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {variant} | {evidence_type} | {lower_delta:.6f} | {gate_label} | {paper_use} |".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def _llm_template_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# k.3.2 LLM Prompt-Template Robustness",
        "",
        f"- final diagnostic label: `{payload['final_diagnostic_label']}`",
        f"- main final label: `{payload['main_final_label']}`",
        "",
        "| run | role | templates | matched signal | posterior align | posterior eq | surface eq | label |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {run_id} | {model_role} | {template_count} | {matched_signal:.6f} | "
            "{posterior_alignment:.6f} | {posterior_equivalence:.6f} | "
            "{surface_equivalence:.6f} | {final_label} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def _g1_g4_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# k.3.3 G1-G4 Threshold Manifest",
        "",
        "| gate | name | statistic | pass rule | CI/permutation rule | source |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload["gates"]:
        lines.append(
            "| {gate_id} | {gate_name} | {statistic} | {pass_rule} | "
            "{ci_or_permutation_rule} | {source_artifact} |".format(**row)
        )
    return "\n".join(lines) + "\n"
