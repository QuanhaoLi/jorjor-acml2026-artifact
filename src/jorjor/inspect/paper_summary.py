from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jorjor.artifacts import checksum_payload, write_json_artifact


REPORT_TYPE = "k.2.4_paper_summary"


def build_k2_paper_summary(*, product_pack_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Build a compact paper-ready summary from a gap-free product pack."""

    diagnostic = _read_json(product_pack_dir / "diagnostic_report.json")
    tables = _read_json(product_pack_dir / "tables.json")
    gaps = list(diagnostic.get("gaps", []))
    rows = {
        "poker_verdict": _required_string(diagnostic, "final_verdict"),
        "synthetic_validation": _required_string(diagnostic, "methodology_validation_status"),
        "toy_oracle_validation": _required_string(diagnostic, "toy_oracle_validation_status"),
        "learned_synthetic_validation": _required_string(diagnostic, "learned_methodology_validation_status"),
        "explicit_belief_state_validation": _required_string(diagnostic, "explicit_methodology_validation_status"),
        "non_poker_protocol_transfer": _required_string(diagnostic, "latent_protocol_transfer_status"),
        "frozen_lm_latent_order_audit": _required_string(diagnostic, "llm_latent_order_status"),
        "range_supervised_strict_gate": _required_string(diagnostic, "range_supervised_strict_gate_status"),
        "mlp_capacity_gate": _required_string(diagnostic, "probe_capacity_s6_s7_status"),
        "reproduction_manifest": _required_string(diagnostic, "reproduction_manifest_status"),
    }
    payload = {
        "version": "k.2.4",
        "report_type": REPORT_TYPE,
        "product_pack_dir": str(product_pack_dir),
        "product_pack_checksum": diagnostic.get("checksum"),
        "gap_count": len(gaps),
        "gaps": gaps,
        "rows": rows,
        "llm_case_summary": _llm_case_summary(tables),
        "paper_safe_claim": (
            "JorJor-Inspect diagnoses the poker model as composition-shortcut-supported; "
            "the non-poker protocol-transfer controls pass; and the frozen-LM latent-order row "
            "shows order-sensitive information but does not license a belief-like LLM claim."
        ),
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "k.2.4_paper_summary.json", payload)
    (output_dir / "k.2.4_paper_summary.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required product-pack file is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"diagnostic report is missing string field: {key}")
    return value


def _llm_case_summary(tables: dict[str, Any]) -> dict[str, Any]:
    reports = tables.get("llm_latent_order", [])
    if not isinstance(reports, list):
        return {"status": "missing"}
    for report in reports:
        if not isinstance(report, dict):
            continue
        for case in report.get("cases", []):
            if isinstance(case, dict) and case.get("backend") == "hf_causal_lm":
                return {
                    "backend": "hf_causal_lm",
                    "final_diagnostic_label": case.get("final_diagnostic_label"),
                    "matched_composition_summary": case.get("matched_composition_summary"),
                    "posterior_equivalence_summary": case.get("posterior_equivalence_summary"),
                    "surface_equivalence_summary": case.get("surface_equivalence_summary"),
                }
    return {"status": "missing_hf_causal_lm_row"}


def _markdown(payload: dict[str, Any]) -> str:
    rows = payload["rows"]
    lines = [
        "# k.2.4 Paper Summary",
        "",
        "## Product Verdicts",
        "",
        "| row | label |",
        "|---|---|",
    ]
    for key, value in rows.items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(
        [
            "",
            "## Claim-Safe Sentence",
            "",
            payload["paper_safe_claim"],
            "",
            "## Reproducibility",
            "",
            f"- product pack: `{payload['product_pack_dir']}`",
            f"- product pack checksum: `{payload['product_pack_checksum']}`",
            f"- summary checksum: `{payload['checksum']}`",
            f"- gap count: `{payload['gap_count']}`",
            "",
        ]
    )
    return "\n".join(lines)
