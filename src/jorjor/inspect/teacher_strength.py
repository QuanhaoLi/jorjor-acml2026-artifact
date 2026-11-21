from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jorjor.artifacts import checksum_payload, write_json_artifact


def build_teacher_strength_comparison_pack(
    *,
    teacher_report_path: Path,
    student_report_path: Path,
    gate_report_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build the k.0.7 paper-facing teacher-strength comparison pack."""

    teacher = _read_report(teacher_report_path, required_checksum=True)
    student = _read_report(student_report_path, required_checksum=True)
    gate = _read_report(gate_report_path, required_checksum=True)
    _validate_student_guardrails(student)
    row = _comparison_row(teacher, student, gate)
    payload = {
        "version": "k.0.7",
        "report_type": "teacher_strength_comparison_pack",
        "claim_boundary": "comparison_pack_only; does_not_add_new_empirical_evidence",
        "rows": [row],
        "sources": [
            _source("teacher_report", teacher_report_path, teacher),
            _source("student_report", student_report_path, student),
            _source("gate_report", gate_report_path, gate),
        ],
        "recommended_next_version": "k.0.8",
        "recommended_next_action": _recommended_next_action(row),
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "teacher_strength_comparison_pack.json", payload)
    (output_dir / "teacher_strength_comparison_pack.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _read_report(path: Path, *, required_checksum: bool) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    if required_checksum and not isinstance(data.get("checksum"), str):
        raise ValueError(f"source report is missing checksum: {path}")
    return data


def _validate_student_guardrails(student: dict[str, Any]) -> None:
    if student.get("opponent_hand_supervised") is not False:
        raise ValueError("k.0.7 requires the k.0.5 student to be opponent-unsupervised")
    if student.get("probe_only_hidden_labels_used_for_training") is not False:
        raise ValueError("k.0.7 requires k.0.5 probe-only hidden labels to be unused in training")


def _comparison_row(teacher: dict[str, Any], student: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    gates = gate["gates"]
    s5 = gates["s5_hidden_vs_action_value"]
    s6 = gates["s6_hidden_vs_action_value_composition"]
    s7 = gates["s7_matched_hidden_vs_composition_prior"]
    metrics = gate["metrics"]
    teacher_strength = teacher.get("teacher_strength_proxy") or {}
    win_rate = _first_float(teacher_strength, ("win_rate_vs_random", "teacher_win_rate"))
    average_return = _first_float(teacher_strength, ("average_return_vs_random", "teacher_average_return"))
    return {
        "teacher_family": teacher.get("teacher_family"),
        "teacher_strength_win_rate_vs_random": win_rate,
        "teacher_strength_average_return_vs_random": average_return,
        "policy_entropy": _float_or_none(teacher.get("policy_entropy")),
        "teacher_decision_point_count": int(teacher.get("decision_point_count", 0)),
        "student_policy_kl": _float_or_none(student.get("test_teacher_distribution_kl")),
        "student_policy_ce": _float_or_none(student.get("test_teacher_distribution_cross_entropy")),
        "student_sampled_action_accuracy": _float_or_none(student.get("test_sampled_action_accuracy")),
        "student_teacher_argmax_accuracy": _float_or_none(student.get("test_teacher_argmax_accuracy")),
        "opponent_hand_supervised": bool(student.get("opponent_hand_supervised")),
        "locked_layer": gate.get("locked_layer_used"),
        "hidden_probe_top10": _float_or_none(metrics.get("hidden_probe_top10")),
        "label_permutation_hidden_top10": _float_or_none(metrics.get("label_permutation_hidden_top10")),
        "action_value_nuisance_top10": _float_or_none(metrics.get("action_value_nuisance_top10")),
        "action_value_composition_nuisance_top10": _float_or_none(
            metrics.get("action_value_composition_nuisance_top10")
        ),
        "s5_delta": _float_or_none(s5.get("delta")),
        "s5_lower_delta": _float_or_none(s5.get("delta_ci", [None, None])[0]),
        "s5_upper_delta": _float_or_none(s5.get("delta_ci", [None, None])[1]),
        "s6_delta": _float_or_none(s6.get("delta")),
        "s6_lower_delta": _float_or_none(s6.get("delta_ci", [None, None])[0]),
        "s6_upper_delta": _float_or_none(s6.get("delta_ci", [None, None])[1]),
        "s7_delta": _float_or_none(s7.get("delta")),
        "s7_lower_delta": _float_or_none(s7.get("delta_ci", [None, None])[0]),
        "s7_upper_delta": _float_or_none(s7.get("delta_ci", [None, None])[1]),
        "s7_matched_point_count": int(s7.get("matched_point_count", 0)),
        "final_diagnostic_label": gate.get("final_diagnostic_label"),
        "claim_boundary": "single_seed_smoke_row; requires_k_0_8_for_robust_claim",
    }


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _first_float(data: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _float_or_none(data.get(key))
        if value is not None:
            return value
    return None


def _source(role: str, path: Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "version": report.get("version"),
        "checksum": report.get("checksum"),
    }


def _recommended_next_action(row: dict[str, Any]) -> str:
    if row["final_diagnostic_label"] == "realized_label_controls_passed":
        return "run three-seed k.0.8 to test whether the positive strict-gate result is robust"
    return "run k.0.8 three-seed/scale before deciding whether teacher strength changes the paper claim"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Teacher-Strength Comparison Pack",
        "",
        "## Summary",
        "",
        f"- recommended next version: `{payload['recommended_next_version']}`",
        f"- recommended next action: {payload['recommended_next_action']}",
        "",
        "## Rows",
        "",
        "| teacher | KL | CE | hidden top10 | S6 lower | S7 lower | matched points | label |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {teacher_family} | {student_policy_kl:.6f} | {student_policy_ce:.6f} | "
            "{hidden_probe_top10:.6f} | {s6_lower_delta:.6f} | {s7_lower_delta:.6f} | "
            "{s7_matched_point_count} | {final_diagnostic_label} |".format(**row)
        )
    lines.extend(["", "## Sources", "", "| role | version | checksum | path |", "|---|---|---|---|"])
    for source in payload["sources"]:
        lines.append(f"| {source['role']} | {source['version']} | {source['checksum']} | {source['path']} |")
    return "\n".join(lines) + "\n"
