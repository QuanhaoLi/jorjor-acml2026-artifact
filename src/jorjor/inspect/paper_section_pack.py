from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jorjor.artifacts import checksum_payload, write_json_artifact


REPORT_TYPE = "k.2.5_paper_section_pack"
REQUIRED_LABELS = {
    "poker_verdict": "composition_shortcut_supported",
    "frozen_lm_latent_order_audit": "order_information_supported",
    "range_supervised_strict_gate": "composition_shortcut_supported",
    "mlp_capacity_gate": "mlp_inconclusive",
}


def build_k2_paper_section_pack(*, paper_summary_json: Path, output_dir: Path) -> dict[str, Any]:
    """Build manuscript-ready snippets from the k.2.4 paper summary."""

    summary = _read_json(paper_summary_json)
    rows = summary.get("rows")
    if not isinstance(rows, dict):
        raise ValueError("k.2.4 summary is missing rows")
    _validate_required_labels(rows)
    gap_count = summary.get("gap_count")
    if gap_count != 0:
        raise ValueError(f"k.2.4 summary must be gap-free, got gap_count={gap_count!r}")

    sections = _sections(summary, rows)
    payload = {
        "version": "k.2.5",
        "report_type": REPORT_TYPE,
        "source_summary_json": str(paper_summary_json),
        "source_summary_checksum": summary.get("checksum"),
        "product_pack_checksum": summary.get("product_pack_checksum"),
        "gap_count": gap_count,
        "rows": rows,
        "sections": sections,
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "k.2.5_paper_section_pack.json", payload)
    (output_dir / "k.2.5_paper_section_pack.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required summary file is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _validate_required_labels(rows: dict[str, Any]) -> None:
    for key, expected in REQUIRED_LABELS.items():
        actual = rows.get(key)
        if actual != expected:
            raise ValueError(f"required label mismatch for {key}: expected {expected!r}, got {actual!r}")


def _sections(summary: dict[str, Any], rows: dict[str, Any]) -> dict[str, str]:
    """Return copy-safe paper snippets keyed by manuscript use."""

    result_paragraph = (
        "The complete JorJor-Inspect product audit diagnoses the Limit Hold'em poker case as "
        f"`{rows['poker_verdict']}`. Exact-posterior methodology validations pass in the synthetic, "
        "toy, learned, and explicit belief-state controls, while the poker strict-gate rows remain "
        "conservative: the range-supervised strict gate is "
        f"`{rows['range_supervised_strict_gate']}` and the MLP capacity gate is "
        f"`{rows['mlp_capacity_gate']}`."
    )
    transfer_paragraph = (
        "The non-poker latent-order benchmark is used as protocol-transfer validation rather than as a "
        f"poker claim upgrade. Its status is `{rows['non_poker_protocol_transfer']}`, showing that the "
        "same gate vocabulary can be applied beyond betting-action histories."
    )
    llm_caveat_paragraph = (
        "The frozen open-weight LM row is a transfer stress test on textual latent-order histories. Its "
        f"diagnostic label is `{rows['frozen_lm_latent_order_audit']}`: the hidden states carry "
        "order-sensitive information, but the row does not license a belief-like LLM claim."
    )
    repro_paragraph = (
        "The paper section pack is generated from the k.2.4 summary rather than manually copied tables. "
        f"The source summary checksum is `{summary.get('checksum')}` and the product pack checksum is "
        f"`{summary.get('product_pack_checksum')}`; the preserved gap count is `{summary.get('gap_count')}`."
    )
    reviewer_risks = (
        "Do not claim that the frozen LM learns belief tracking; do not claim that range supervision clears "
        "strict poker composition gates; do not claim that MLP capacity recovers belief beyond composition; "
        "and do not use protocol-transfer validation as a poker verdict upgrade."
    )
    return {
        "result_paragraph": result_paragraph,
        "protocol_transfer_paragraph": transfer_paragraph,
        "frozen_lm_caveat_paragraph": llm_caveat_paragraph,
        "reproducibility_paragraph": repro_paragraph,
        "reviewer_risk_notes": reviewer_risks,
    }


def _markdown(payload: dict[str, Any]) -> str:
    rows = payload["rows"]
    sections = payload["sections"]
    lines = [
        "# k.2.5 Paper Section Pack",
        "",
        "## Main Verdict Table",
        "",
        "| row | label |",
        "|---|---|",
    ]
    for key, value in rows.items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(
        [
            "",
            "## Result Paragraph",
            "",
            sections["result_paragraph"],
            "",
            "## Protocol Transfer Paragraph",
            "",
            sections["protocol_transfer_paragraph"],
            "",
            "## Frozen-LM Caveat Paragraph",
            "",
            sections["frozen_lm_caveat_paragraph"],
            "",
            "## Reproducibility Paragraph",
            "",
            sections["reproducibility_paragraph"],
            "",
            "## Reviewer-Risk Notes",
            "",
            sections["reviewer_risk_notes"],
            "",
            "## Checksums",
            "",
            f"- source summary: `{payload['source_summary_json']}`",
            f"- source summary checksum: `{payload['source_summary_checksum']}`",
            f"- product pack checksum: `{payload['product_pack_checksum']}`",
            f"- section pack checksum: `{payload['checksum']}`",
            f"- gap count: `{payload['gap_count']}`",
            "",
        ]
    )
    return "\n".join(lines)
