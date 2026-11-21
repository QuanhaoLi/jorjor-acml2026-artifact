from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jorjor.artifacts import checksum_payload, write_json_artifact


REPORT_TYPE = "k.5.3_cbps_claim_ladder"
CBPS_SUPPORTED = "cbps_supported"
COMPOSITION_SHORTCUT_SUPPORTED = "composition_shortcut_supported"
INCONCLUSIVE = "inconclusive"
REQUIRED_REPORT_ROLES = (
    "synthetic_cbps_report",
    "predictive_link_report",
    "composition_boundedness_report",
)


def build_cbps_claim_ladder(
    *,
    synthetic_cbps_report: Path,
    predictive_link_report: Path,
    composition_boundedness_report: Path,
    output_dir: Path,
    poker_oracle_report: Path | None = None,
    composition_sensitivity_report: Path | None = None,
    threshold_manifest: Path | None = None,
    capacity_report: Path | None = None,
    behavior_link_report: Path | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Assemble the k.5 CBPS claim ladder from completed report artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "synthetic_cbps_report": _required_source(synthetic_cbps_report),
        "predictive_link_report": _required_source(predictive_link_report),
        "composition_boundedness_report": _required_source(composition_boundedness_report),
        "poker_oracle_report": _optional_source(poker_oracle_report, role="poker_oracle_report"),
        "composition_sensitivity_report": _optional_source(
            composition_sensitivity_report,
            role="composition_sensitivity_report",
        ),
        "threshold_manifest": _optional_source(threshold_manifest, role="threshold_manifest"),
        "capacity_report": _optional_source(capacity_report, role="capacity_report"),
        "behavior_link_report": _optional_source(behavior_link_report, role="behavior_link_report"),
    }
    payloads = {role: _read_json(Path(source["path"])) for role, source in sources.items() if source["present"]}
    gaps = [source["gap"] for source in sources.values() if source.get("gap")]
    rows = _claim_rows(payloads)
    final_label = _final_label(rows, gaps)
    payload = {
        "version": "k.5.6",
        "report_type": REPORT_TYPE,
        "claim_scope": "diagnostic_interpretation_not_ordered_belief",
        "git_commit": git_commit,
        "sources": sources,
        "gaps": gaps,
        "rows": rows,
        "final_label": final_label,
        "paper_safe_claim": _paper_safe_claim(final_label),
        "forbidden_claim": "The no-range poker AR model has ordered opponent-belief tracking.",
        "output_contract": [
            "k.5.3_cbps_claim_ladder.json",
            "k.5.3_cbps_claim_ladder.md",
            "k.5.3_cbps_manifest.json",
            "paper_table_cbps_claim_ladder.md",
        ],
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "k.5.3_cbps_claim_ladder.json", payload)
    (output_dir / "k.5.3_cbps_claim_ladder.md").write_text(_markdown(payload), encoding="utf-8")
    (output_dir / "paper_table_cbps_claim_ladder.md").write_text(_paper_table(rows), encoding="utf-8")
    write_json_artifact(output_dir / "k.5.3_cbps_manifest.json", _manifest(output_dir, payload))
    return payload


def _claim_rows(payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    synthetic = payloads["synthetic_cbps_report"]
    predictive = payloads["predictive_link_report"]
    bounded = payloads["composition_boundedness_report"]
    capacity = payloads.get("capacity_report")
    oracle = payloads.get("poker_oracle_report")
    behavior = payloads.get("behavior_link_report")

    synthetic_passed = bool(synthetic.get("synthetic_cbps_control_passed"))
    predictive_label = str(predictive.get("final_diagnostic_label", INCONCLUSIVE))
    bounded_label = str(bounded.get("final_diagnostic_label", INCONCLUSIVE))
    capacity_label = _capacity_label(capacity)
    oracle_label = _oracle_label(oracle)
    behavior_label = _behavior_link_label(behavior)
    final = (
        CBPS_SUPPORTED
        if synthetic_passed
        and predictive_label == "predictive_supported"
        and bounded_label == "composition_bounded_supported"
        and behavior_label
        in {"composition_bounded_behavior_link", "hidden_behavior_link_supported", "not_provided"}
        and capacity_label != "capacity_sensitive"
        else COMPOSITION_SHORTCUT_SUPPORTED
    )
    return [
        {
            "domain": "synthetic_latent_order",
            "model_or_state": "posterior/composition/CBPS known states",
            "supervision": "analytic known-control",
            "predictive_gate": "passed" if synthetic_passed else "failed",
            "representation_behavior_link_gate": "not_applicable_known_control",
            "latent_probe_gate": _synthetic_case_label(synthetic, "CBPS_state"),
            "label_artifact_gate": "not_applicable",
            "composition_bounded_gate": _synthetic_case_label(synthetic, "CBPS_state"),
            "matched_range_label_proxy_gate": "not_applicable_exact_synthetic_posterior_available",
            "capacity_sensitivity": "not_applicable",
            "final_label": "synthetic_cbps_control_passed" if synthetic_passed else "synthetic_cbps_control_failed",
            "claim_scope": "diagnostic_label_validation",
            "allowed_claim": "The audit can distinguish posterior, composition, and CBPS known states in a synthetic control.",
        },
        {
            "domain": "limit_holdem",
            "model_or_state": "no-range 4-layer AR, seeds k/l/m",
            "supervision": "action/value only",
            "predictive_gate": predictive_label,
            "representation_behavior_link_gate": behavior_label,
            "latent_probe_gate": "positive_before_strict_composition_gates",
            "label_artifact_gate": "passed_from_prior_gates",
            "composition_bounded_gate": bounded_label,
            "matched_range_label_proxy_gate": _proxy_gate_label(bounded),
            "capacity_sensitivity": capacity_label,
            "oracle_positive_control": oracle_label,
            "final_label": final,
            "claim_scope": "diagnostic_interpretation_not_ordered_belief",
            "allowed_claim": (
                "The no-range poker AR row is behavior-predictive but composition-bounded under the locked audit."
                if final == CBPS_SUPPORTED
                else "Positive range probes remain insufficient evidence for ordered belief tracking."
            ),
        },
    ]


def _final_label(rows: list[dict[str, Any]], gaps: list[str]) -> str:
    if any(role in gap for role in REQUIRED_REPORT_ROLES for gap in gaps):
        return INCONCLUSIVE
    poker = next(row for row in rows if row["domain"] == "limit_holdem")
    return str(poker["final_label"])


def _synthetic_case_label(payload: dict[str, Any], mode: str) -> str:
    for case in payload.get("cases", []):
        if isinstance(case, dict) and case.get("mode") == mode:
            return str(case.get("final_diagnostic_label", INCONCLUSIVE))
    return INCONCLUSIVE


def _proxy_gate_label(payload: dict[str, Any]) -> str:
    aggregate = payload.get("aggregate", {})
    if not isinstance(aggregate, dict):
        return INCONCLUSIVE
    positive = int(aggregate.get("proxy_seed_bucket_ci_positive_count", 0))
    crossed = int(aggregate.get("proxy_seed_bucket_ci_cross_zero_count", 0))
    if positive == 0 and crossed > 0:
        return "range_label_proxy_tracking_not_supported"
    if positive > 0:
        return "ordered_proxy_component_supported"
    return INCONCLUSIVE


def _capacity_label(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "not_provided"
    label = str(payload.get("final_diagnostic_label", payload.get("final_label", payload.get("final_verdict", INCONCLUSIVE))))
    if "capacity_sensitive" in label:
        return "capacity_sensitive"
    return label


def _oracle_label(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "not_provided"
    return str(payload.get("final_diagnostic_label", payload.get("paper_claim", "provided")))


def _behavior_link_label(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "not_provided"
    return str(payload.get("final_diagnostic_label", INCONCLUSIVE))


def _paper_safe_claim(final_label: str) -> str:
    if final_label == CBPS_SUPPORTED:
        return (
            "Positive hidden-state range probes are insufficient evidence for ordered belief tracking. "
            "For the no-range poker AR row, the best-supported diagnostic interpretation is "
            "composition-bounded predictive support: behavior prediction is positive, while strict "
            "matched-composition gates block an ordered-belief claim."
        )
    if final_label == COMPOSITION_SHORTCUT_SUPPORTED:
        return "The evidence remains strongest for a composition-shortcut diagnosis, not CBPS or ordered belief."
    return "The k.5 evidence is incomplete; keep the paper centered on claim-gated diagnostics."


def _required_source(path: Path) -> dict[str, Any]:
    source = _source(path, required=True, role=path.name)
    if not source["present"]:
        raise FileNotFoundError(f"required report is missing: {path}")
    return source


def _optional_source(path: Path | None, *, role: str) -> dict[str, Any]:
    if path is None:
        return {"role": role, "path": None, "required": False, "present": False, "gap": f"{role} not provided"}
    return _source(path, required=False, role=role)


def _source(path: Path, *, required: bool, role: str) -> dict[str, Any]:
    source: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "required": required,
        "present": path.exists(),
    }
    if path.exists():
        source["sha256"] = _sha256(path)
        source["size_bytes"] = path.stat().st_size
    else:
        source["gap"] = f"{role} missing"
    return source


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['version']} CBPS Claim Ladder",
        "",
        f"final_label: `{payload['final_label']}`",
        "",
        "## Claim-Safe Sentence",
        "",
        payload["paper_safe_claim"],
        "",
        "## Rows",
        "",
        _paper_table(payload["rows"]),
        "",
        "## Gaps",
        "",
    ]
    if payload["gaps"]:
        lines.extend(f"- {gap}" for gap in payload["gaps"])
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _paper_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| domain | model/state | predictive | behavior link | composition bounded | matched proxy | capacity | final label | claim scope |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {domain} | {model} | `{predictive}` | `{behavior}` | `{bounded}` | `{proxy}` | `{capacity}` | `{final}` | `{scope}` |".format(
                domain=row["domain"],
                model=row["model_or_state"],
                predictive=row["predictive_gate"],
                behavior=row.get("representation_behavior_link_gate", "not_applicable"),
                bounded=row["composition_bounded_gate"],
                proxy=row["matched_range_label_proxy_gate"],
                capacity=row["capacity_sensitivity"],
                final=row["final_label"],
                scope=row["claim_scope"],
            )
        )
    return "\n".join(lines) + "\n"


def _manifest(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": payload["version"],
        "report_type": "k.5.3_cbps_manifest",
        "claim_scope": payload["claim_scope"],
        "output_dir": str(output_dir),
        "input_report_paths": {
            role: source["path"] for role, source in payload["sources"].items() if source.get("path") is not None
        },
        "input_sha256": {
            role: source.get("sha256") for role, source in payload["sources"].items() if source.get("sha256")
        },
        "output_paths": payload["output_contract"],
        "output_sha256": payload["checksum"],
        "final_label": payload["final_label"],
        "one_command_reproduction": "see scripts/build_cbps_claim_ladder.py --help for all optional inputs",
    }
