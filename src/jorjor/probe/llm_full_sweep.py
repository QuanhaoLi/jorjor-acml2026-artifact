from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.probe.inspect import COMPOSITION_SHORTCUT_SUPPORTED, INCONCLUSIVE
from jorjor.probe.llm_latent_order import (
    BELIEF_LIKE_TRANSFER_SUPPORTED,
    ORDER_INFORMATION_SUPPORTED,
    build_llm_latent_order_report,
)


REPORT_TYPE = "k.2.6_llm_full_sweep"


def build_llm_full_sweep(
    *,
    output_dir: Path,
    lengths: tuple[int, ...],
    layer_ids: tuple[int, ...],
    backends: tuple[str, ...],
    max_pairs: int = 512,
    bootstrap_count: int = 1000,
    bootstrap_seed: int = 20260604,
    signal_probability: float = 0.72,
    surface_template_count: int = 2,
    model_id: str | None = None,
    model_role: str = "main_or_single_model",
    locked_layer_id: int | None = None,
    model_specs: tuple[dict[str, Any], ...] | None = None,
    device: str = "cpu",
    dtype: str = "auto",
    max_length: int = 256,
    hf_batch_size: int = 16,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Run the k.2 latent-order report across multiple layer/length configs."""

    if not lengths:
        raise ValueError("lengths must not be empty")
    if not layer_ids:
        raise ValueError("layer_ids must not be empty")
    if not backends:
        raise ValueError("backends must not be empty")

    specs = _normalize_model_specs(
        model_specs=model_specs,
        fallback_model_id=model_id,
        fallback_model_role=model_role,
        fallback_locked_layer_id=locked_layer_id,
        fallback_layer_ids=layer_ids,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    hf_rows = []
    run_index = 0
    for spec in specs:
        spec_layer_ids = tuple(int(value) for value in spec["layer_ids"])
        for length in lengths:
            for layer_id in spec_layer_ids:
                analysis_role = _analysis_role(layer_id=layer_id, locked_layer_id=spec.get("locked_layer_id"))
                run_id = (
                    f"{_model_slug(spec.get('model_id'))}_{spec['model_role']}_"
                    f"length{length}_layer{_layer_slug(layer_id)}"
                )
                run_dir = output_dir / "runs" / run_id
                report = build_llm_latent_order_report(
                    output_dir=run_dir,
                    length=length,
                    signal_probability=signal_probability,
                    max_pairs=max_pairs,
                    bootstrap_count=bootstrap_count,
                    bootstrap_seed=bootstrap_seed + run_index * 100,
                    backends=backends,
                    surface_template_count=surface_template_count,
                    model_id=spec.get("model_id"),
                    layer_id=layer_id,
                    device=device,
                    dtype=dtype,
                    max_length=max_length,
                    hf_batch_size=hf_batch_size,
                    trust_remote_code=trust_remote_code,
                )
                run_summary = _run_summary(
                    run_id=run_id,
                    run_dir=run_dir,
                    model_id=spec.get("model_id"),
                    model_role=spec["model_role"],
                    analysis_role=analysis_role,
                    locked_layer_id=spec.get("locked_layer_id"),
                    layer_id=layer_id,
                    report=report,
                )
                runs.append(run_summary)
                hf_rows.extend(row for row in run_summary["case_summaries"] if row["backend"] == "hf_causal_lm")
                run_index += 1

    main_hf_rows = [
        row
        for row in hf_rows
        if row.get("model_role") == "main" and row.get("analysis_role") == "locked_main"
    ]
    payload = {
        "version": "k.2.6",
        "report_type": REPORT_TYPE,
        "claim_scope": "multi_config_frozen_lm_transfer_stress_test_not_belief_claim",
        "lengths": list(lengths),
        "backends": list(backends),
        "model_specs": specs,
        "run_count": len(runs),
        "hf_run_count": len(hf_rows),
        "main_hf_run_count": len(main_hf_rows),
        "main_sweep_label": _final_label(main_hf_rows),
        "final_sweep_label": _final_label(hf_rows),
        "runs": runs,
        "hf_rows": hf_rows,
        "main_hf_rows": main_hf_rows,
        "layer_curve_rows": [row for row in hf_rows if row.get("analysis_role") == "appendix_layer_curve"],
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "k.2.6_llm_full_sweep_report.json", payload)
    (output_dir / "k.2.6_llm_full_sweep_report.md").write_text(_markdown(payload), encoding="utf-8")
    write_json_artifact(output_dir / "k.2.6_llm_full_sweep_manifest.json", _manifest(output_dir, payload))
    return payload


def rebuild_llm_full_sweep_from_run_reports(
    *,
    output_dir: Path,
    run_report_paths: tuple[Path, ...],
    model_specs: tuple[dict[str, Any], ...],
    lengths: tuple[int, ...],
    layer_ids: tuple[int, ...],
    backends: tuple[str, ...],
) -> dict[str, Any]:
    """Rebuild a k.2.6 aggregate from existing per-run k.2 reports."""

    if not run_report_paths:
        raise ValueError("run_report_paths must not be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = _normalize_model_specs(
        model_specs=model_specs,
        fallback_model_id=None,
        fallback_model_role="main_or_single_model",
        fallback_locked_layer_id=None,
        fallback_layer_ids=layer_ids,
    )
    runs = []
    hf_rows = []
    for report_path in run_report_paths:
        report = _read_run_report(report_path)
        spec = _spec_for_report(report=report, specs=specs)
        layer_id = _layer_for_report(report=report)
        analysis_role = _analysis_role(layer_id=layer_id, locked_layer_id=spec.get("locked_layer_id"))
        run_id = (
            f"{_model_slug(spec.get('model_id'))}_{spec['model_role']}_"
            f"length{report.get('length')}_layer{_layer_slug(layer_id)}"
        )
        run_summary = _run_summary(
            run_id=run_id,
            run_dir=report_path.parent,
            model_id=spec.get("model_id"),
            model_role=spec["model_role"],
            analysis_role=analysis_role,
            locked_layer_id=spec.get("locked_layer_id"),
            layer_id=layer_id,
            report=report,
        )
        runs.append(run_summary)
        hf_rows.extend(row for row in run_summary["case_summaries"] if row["backend"] == "hf_causal_lm")

    main_hf_rows = [
        row
        for row in hf_rows
        if row.get("model_role") == "main" and row.get("analysis_role") == "locked_main"
    ]
    payload = {
        "version": "k.2.6",
        "report_type": REPORT_TYPE,
        "claim_scope": "multi_config_frozen_lm_transfer_stress_test_not_belief_claim",
        "lengths": list(lengths),
        "backends": list(backends),
        "model_specs": specs,
        "run_count": len(runs),
        "hf_run_count": len(hf_rows),
        "main_hf_run_count": len(main_hf_rows),
        "main_sweep_label": _final_label(main_hf_rows),
        "final_sweep_label": _final_label(hf_rows),
        "runs": runs,
        "hf_rows": hf_rows,
        "main_hf_rows": main_hf_rows,
        "layer_curve_rows": [row for row in hf_rows if row.get("analysis_role") == "appendix_layer_curve"],
        "aggregate_mode": "rebuilt_from_existing_run_reports",
        "source_run_reports": [str(path) for path in run_report_paths],
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "k.2.6_llm_full_sweep_report.json", payload)
    (output_dir / "k.2.6_llm_full_sweep_report.md").write_text(_markdown(payload), encoding="utf-8")
    write_json_artifact(output_dir / "k.2.6_llm_full_sweep_manifest.json", _manifest(output_dir, payload))
    return payload


def _normalize_model_specs(
    *,
    model_specs: tuple[dict[str, Any], ...] | None,
    fallback_model_id: str | None,
    fallback_model_role: str,
    fallback_locked_layer_id: int | None,
    fallback_layer_ids: tuple[int, ...],
) -> list[dict[str, Any]]:
    if model_specs is None:
        return [
            {
                "model_id": fallback_model_id,
                "model_role": fallback_model_role,
                "locked_layer_id": fallback_locked_layer_id,
                "layer_ids": list(fallback_layer_ids),
            }
        ]
    normalized = []
    for spec in model_specs:
        if "layer_ids" not in spec or not spec["layer_ids"]:
            raise ValueError("each model spec requires layer_ids")
        normalized.append(
            {
                "model_id": spec.get("model_id"),
                "model_role": str(spec.get("model_role", "appendix")),
                "locked_layer_id": spec.get("locked_layer_id"),
                "layer_ids": [int(value) for value in spec["layer_ids"]],
            }
        )
    if not normalized:
        raise ValueError("model_specs must not be empty")
    return normalized


def _analysis_role(*, layer_id: int, locked_layer_id: int | None) -> str:
    if locked_layer_id is not None and layer_id == locked_layer_id:
        return "locked_main"
    return "appendix_layer_curve"


def _layer_slug(layer_id: int) -> str:
    return f"m{abs(layer_id)}" if layer_id < 0 else str(layer_id)


def _model_slug(model_id: str | None) -> str:
    if model_id is None:
        return "fake_controls"
    name = model_id.rstrip("/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _run_summary(
    *,
    run_id: str,
    run_dir: Path,
    model_id: str | None,
    model_role: str,
    analysis_role: str,
    locked_layer_id: int | None,
    layer_id: int,
    report: dict[str, Any],
) -> dict[str, Any]:
    cases = []
    for case in report.get("cases", []):
        if not isinstance(case, dict):
            continue
        matched = case.get("matched_composition_summary", {})
        posterior_eq = case.get("posterior_equivalence_summary", {})
        surface_eq = case.get("surface_equivalence_summary", {})
        matched_ci = matched.get("representation_distance_ci") or {}
        posterior_eq_ci = posterior_eq.get("representation_distance_ci") or {}
        surface_eq_ci = surface_eq.get("representation_distance_ci") or {}
        cases.append(
            {
                "run_id": run_id,
                "model_id": model_id,
                "model_role": model_role,
                "analysis_role": analysis_role,
                "locked_layer_id": locked_layer_id,
                "layer_id": layer_id,
                "backend": case.get("backend"),
                "final_diagnostic_label": case.get("final_diagnostic_label"),
                "matched_pair_count": matched.get("pair_count"),
                "matched_bucket_count": matched.get("bucket_count"),
                "matched_mean_representation_distance": matched.get("mean_representation_distance"),
                "matched_representation_ci_lower": matched_ci.get("bootstrap_lower"),
                "matched_representation_ci_upper": matched_ci.get("bootstrap_upper"),
                "posterior_representation_correlation": matched.get("posterior_representation_correlation"),
                "posterior_equivalence_pair_count": posterior_eq.get("pair_count"),
                "posterior_equivalence_bucket_count": posterior_eq.get("bucket_count"),
                "posterior_equivalence_mean_representation_distance": posterior_eq.get(
                    "mean_representation_distance"
                ),
                "posterior_equivalence_representation_ci_lower": posterior_eq_ci.get("bootstrap_lower"),
                "posterior_equivalence_representation_ci_upper": posterior_eq_ci.get("bootstrap_upper"),
                "surface_equivalence_pair_count": surface_eq.get("pair_count"),
                "surface_equivalence_bucket_count": surface_eq.get("bucket_count"),
                "surface_equivalence_mean_representation_distance": surface_eq.get("mean_representation_distance"),
                "surface_equivalence_representation_ci_lower": surface_eq_ci.get("bootstrap_lower"),
                "surface_equivalence_representation_ci_upper": surface_eq_ci.get("bootstrap_upper"),
            }
        )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "model_id": model_id,
        "model_role": model_role,
        "analysis_role": analysis_role,
        "length": report.get("length"),
        "layer_id": layer_id,
        "locked_layer_id": locked_layer_id,
        "surface_template_count": report.get("surface_template_count"),
        "checksum": report.get("checksum"),
        "case_summaries": cases,
    }


def _read_run_report(report_path: Path) -> dict[str, Any]:
    import json

    if not report_path.exists():
        raise FileNotFoundError(report_path)
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("report_type") != "k.2_llm_latent_order_audit":
        raise ValueError(f"not a k.2 run report: {report_path}")
    return report


def _spec_for_report(*, report: dict[str, Any], specs: list[dict[str, Any]]) -> dict[str, Any]:
    hf_case = next((case for case in report.get("cases", []) if case.get("backend") == "hf_causal_lm"), None)
    if hf_case is None:
        return specs[0]
    metadata = hf_case.get("backend_metadata", {})
    model_id = metadata.get("model_id")
    for spec in specs:
        if spec.get("model_id") == model_id:
            return spec
    return specs[0]


def _layer_for_report(*, report: dict[str, Any]) -> int:
    hf_case = next((case for case in report.get("cases", []) if case.get("backend") == "hf_causal_lm"), None)
    if hf_case is None:
        return -1
    metadata = hf_case.get("backend_metadata", {})
    layer_id = metadata.get("layer_id")
    if layer_id is None:
        return -1
    return int(layer_id)


def _final_label(hf_rows: list[dict[str, Any]]) -> str:
    if not hf_rows:
        return INCONCLUSIVE
    labels = [row.get("final_diagnostic_label") for row in hf_rows]
    if all(label == BELIEF_LIKE_TRANSFER_SUPPORTED for label in labels):
        return BELIEF_LIKE_TRANSFER_SUPPORTED
    if any(label == ORDER_INFORMATION_SUPPORTED for label in labels):
        return ORDER_INFORMATION_SUPPORTED
    if all(label == COMPOSITION_SHORTCUT_SUPPORTED for label in labels):
        return COMPOSITION_SHORTCUT_SUPPORTED
    return INCONCLUSIVE


def _manifest(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": payload["version"],
        "report_type": f"{REPORT_TYPE}_manifest",
        "output_dir": str(output_dir),
        "model_specs": payload["model_specs"],
        "run_count": payload["run_count"],
        "hf_run_count": payload["hf_run_count"],
        "main_hf_run_count": payload["main_hf_run_count"],
        "main_sweep_label": payload["main_sweep_label"],
        "final_sweep_label": payload["final_sweep_label"],
        "report_checksum": payload["checksum"],
        "run_reports": [
            {
                "run_id": run["run_id"],
                "path": str(Path(run["run_dir"]) / "k.2_llm_latent_order_report.json"),
                "checksum": run["checksum"],
            }
            for run in payload["runs"]
        ],
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# k.2.6 LLM Full Sweep",
        "",
        "## Summary",
        "",
        f"- main sweep label: `{payload['main_sweep_label']}`",
        f"- final sweep label: `{payload['final_sweep_label']}`",
        f"- run count: `{payload['run_count']}`",
        f"- HF run count: `{payload['hf_run_count']}`",
        f"- main HF run count: `{payload['main_hf_run_count']}`",
        f"- checksum: `{payload['checksum']}`",
        "",
        "## Runs",
        "",
        "| run_id | model_role | analysis_role | backend | label | matched pairs/buckets | matched order signal | matched CI | posterior alignment | posterior-equivalence | posterior-eq CI | surface-equivalence | surface-eq CI |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in payload["runs"]:
        for case in run["case_summaries"]:
            lines.append(
                "| {run_id} | {model_role} | {analysis_role} | {backend} | `{label}` | {pairs}/{buckets} | {matched} | {matched_ci} | {corr} | {posterior_eq} | {posterior_eq_ci} | {surface_eq} | {surface_eq_ci} |".format(
                    run_id=run["run_id"],
                    model_role=case["model_role"],
                    analysis_role=case["analysis_role"],
                    backend=case["backend"],
                    label=case["final_diagnostic_label"],
                    pairs=_fmt(case["matched_pair_count"]),
                    buckets=_fmt(case["matched_bucket_count"]),
                    matched=_fmt(case["matched_mean_representation_distance"]),
                    matched_ci=_fmt_ci(
                        case["matched_representation_ci_lower"],
                        case["matched_representation_ci_upper"],
                    ),
                    corr=_fmt(case["posterior_representation_correlation"]),
                    posterior_eq=_fmt(case["posterior_equivalence_mean_representation_distance"]),
                    posterior_eq_ci=_fmt_ci(
                        case["posterior_equivalence_representation_ci_lower"],
                        case["posterior_equivalence_representation_ci_upper"],
                    ),
                    surface_eq=_fmt(case["surface_equivalence_mean_representation_distance"]),
                    surface_eq_ci=_fmt_ci(
                        case["surface_equivalence_representation_ci_lower"],
                        case["surface_equivalence_representation_ci_upper"],
                    ),
                )
            )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This sweep is a frozen-LM transfer stress test. It does not license an LLM belief-tracking claim unless HF rows pass posterior-equivalence and surface-invariance gates.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, int):
        return str(value)
    return ""


def _fmt_ci(lower: Any, upper: Any) -> str:
    if isinstance(lower, float) and isinstance(upper, float):
        return f"[{lower:.6f}, {upper:.6f}]"
    return ""
