from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.inspect.teacher_strength import build_teacher_strength_comparison_pack
from jorjor.probe.inspect import COMPOSITION_SHORTCUT_SUPPORTED, INCONCLUSIVE, REALIZED_LABEL_CONTROLS_PASSED
from jorjor.probe.teacher_distill_gates import build_strong_teacher_gate_report
from jorjor.teacher_artifact import build_limit_abstract_cfr_teacher_artifact
from jorjor.train.teacher_distill import run_limit_teacher_distill_smoke


LABELS = (REALIZED_LABEL_CONTROLS_PASSED, COMPOSITION_SHORTCUT_SUPPORTED, INCONCLUSIVE)


def build_teacher_strength_seed_scale_pack(
    *,
    output_dir: Path,
    seeds: tuple[int, ...],
    teacher_iteration_count: int = 4,
    teacher_deal_sample_count: int = 8,
    teacher_hand_count: int = 24,
    teacher_evaluation_hand_count: int = 24,
    student_epochs: int = 3,
    student_batch_size: int = 16,
    student_hidden_size: int = 32,
    student_num_heads: int = 4,
    student_num_layers: int = 2,
    probe_epochs: int = 3,
    probe_batch_size: int = 32,
    bootstrap_count: int = 64,
    minimum_matched_points: int = 2,
    locked_layer: str = "encoder_layer_2",
    local_smoke_only: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run and aggregate the k.0.8 strong-teacher seed/scale pipeline."""

    _validate_args(seeds=seeds, teacher_hand_count=teacher_hand_count, bootstrap_count=bootstrap_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, seed in enumerate(seeds):
        seed_dir = output_dir / f"seed_{seed}"
        teacher_dir = seed_dir / "teacher"
        student_dir = seed_dir / "student"
        gate_dir = seed_dir / "gates"
        comparison_dir = seed_dir / "comparison"
        teacher = build_limit_abstract_cfr_teacher_artifact(
            output_dir=teacher_dir,
            iteration_count=teacher_iteration_count,
            deal_sample_count=teacher_deal_sample_count,
            hand_count=teacher_hand_count,
            evaluation_hand_count=teacher_evaluation_hand_count,
            seed=seed,
        )
        student = run_limit_teacher_distill_smoke(
            artifact_rows_path=teacher_dir / "teacher_artifact_rows.jsonl",
            output_dir=student_dir,
            epochs=student_epochs,
            batch_size=student_batch_size,
            hidden_size=student_hidden_size,
            num_heads=student_num_heads,
            num_layers=student_num_layers,
            split_seed=seed + 101,
            seed=seed + 211,
            device=device,
            save_model=True,
        ).report
        gate = build_strong_teacher_gate_report(
            artifact_rows_path=teacher_dir / "teacher_artifact_rows.jsonl",
            student_metrics_path=student_dir / "metrics.json",
            model_path=student_dir / "model.pt",
            output_dir=gate_dir,
            locked_layer=locked_layer,
            probe_epochs=probe_epochs,
            probe_batch_size=probe_batch_size,
            bootstrap_count=bootstrap_count,
            bootstrap_seed=seed + 307,
            minimum_matched_points=minimum_matched_points,
            seed=seed + 401,
            device=device,
        )
        comparison = build_teacher_strength_comparison_pack(
            teacher_report_path=teacher_dir / "teacher_artifact_report.json",
            student_report_path=student_dir / "metrics.json",
            gate_report_path=gate_dir / "strong_teacher_gate_report.json",
            output_dir=comparison_dir,
        )
        rows.append(_row(seed=seed, teacher=teacher, student=student, gate=gate, comparison=comparison, seed_dir=seed_dir))

    label_counts = {label: 0 for label in LABELS}
    label_counts.update(Counter(row["final_diagnostic_label"] for row in rows))
    payload = {
        "version": "k.0.8",
        "report_type": "teacher_strength_seed_scale_pack",
        "claim_boundary": "local_smoke_only" if local_smoke_only else "remote_size_pack_requires_manual_interpretation",
        "local_smoke_only": local_smoke_only,
        "seed_count": len(seeds),
        "seeds": list(seeds),
        "settings": {
            "teacher_iteration_count": teacher_iteration_count,
            "teacher_deal_sample_count": teacher_deal_sample_count,
            "teacher_hand_count": teacher_hand_count,
            "teacher_evaluation_hand_count": teacher_evaluation_hand_count,
            "student_epochs": student_epochs,
            "student_hidden_size": student_hidden_size,
            "student_num_layers": student_num_layers,
            "probe_epochs": probe_epochs,
            "bootstrap_count": bootstrap_count,
            "minimum_matched_points": minimum_matched_points,
            "locked_layer": locked_layer,
        },
        "rows": rows,
        "label_counts": label_counts,
        "min_s6_lower_delta": min(row["s6_lower_delta"] for row in rows),
        "min_s7_lower_delta": min(row["s7_lower_delta"] for row in rows),
        "robustness_label": _robustness_label(rows),
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "seed_scale_pack.json", payload)
    (output_dir / "seed_scale_pack.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _validate_args(*, seeds: tuple[int, ...], teacher_hand_count: int, bootstrap_count: int) -> None:
    if not seeds:
        raise ValueError("seeds must be non-empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if teacher_hand_count <= 0:
        raise ValueError("teacher_hand_count must be positive")
    if bootstrap_count <= 0:
        raise ValueError("bootstrap_count must be positive")


def _row(
    *,
    seed: int,
    teacher: dict[str, Any],
    student: dict[str, Any],
    gate: dict[str, Any],
    comparison: dict[str, Any],
    seed_dir: Path,
) -> dict[str, Any]:
    comparison_row = comparison["rows"][0]
    return {
        "seed": seed,
        "teacher_family": comparison_row["teacher_family"],
        "teacher_win_rate_vs_random": comparison_row["teacher_strength_win_rate_vs_random"],
        "teacher_average_return_vs_random": comparison_row["teacher_strength_average_return_vs_random"],
        "student_policy_kl": comparison_row["student_policy_kl"],
        "student_policy_ce": comparison_row["student_policy_ce"],
        "hidden_probe_top10": comparison_row["hidden_probe_top10"],
        "label_permutation_hidden_top10": comparison_row["label_permutation_hidden_top10"],
        "s6_lower_delta": comparison_row["s6_lower_delta"],
        "s7_lower_delta": comparison_row["s7_lower_delta"],
        "final_diagnostic_label": comparison_row["final_diagnostic_label"],
        "teacher_report_path": str(seed_dir / "teacher" / "teacher_artifact_report.json"),
        "student_report_path": str(seed_dir / "student" / "metrics.json"),
        "gate_report_path": str(seed_dir / "gates" / "strong_teacher_gate_report.json"),
        "comparison_report_path": str(seed_dir / "comparison" / "teacher_strength_comparison_pack.json"),
        "teacher_checksum": teacher["checksum"],
        "student_checksum": student["checksum"],
        "gate_checksum": gate["checksum"],
        "comparison_checksum": comparison["checksum"],
    }


def _robustness_label(rows: list[dict[str, Any]]) -> str:
    labels = [row["final_diagnostic_label"] for row in rows]
    if all(label == REALIZED_LABEL_CONTROLS_PASSED for label in labels) and all(
        row["s6_lower_delta"] > 0.0 and row["s7_lower_delta"] > 0.0 for row in rows
    ):
        return "robust_belief_like"
    if all(label == COMPOSITION_SHORTCUT_SUPPORTED for label in labels):
        return "robust_shortcut"
    return "mixed_or_inconclusive"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Strong-Teacher Seed/Scale Pack",
        "",
        "## Summary",
        "",
        f"- seed count: `{payload['seed_count']}`",
        f"- robustness label: `{payload['robustness_label']}`",
        f"- local smoke only: `{payload['local_smoke_only']}`",
        "",
        "## Rows",
        "",
        "| seed | teacher | KL | hidden top10 | S6 lower | S7 lower | label |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {seed} | {teacher_family} | {student_policy_kl:.6f} | {hidden_probe_top10:.6f} | "
            "{s6_lower_delta:.6f} | {s7_lower_delta:.6f} | {final_diagnostic_label} |".format(**row)
        )
    lines.extend(["", "## Label Counts", "", "| label | count |", "|---|---:|"])
    for label, count in payload["label_counts"].items():
        lines.append(f"| {label} | {count} |")
    return "\n".join(lines) + "\n"
