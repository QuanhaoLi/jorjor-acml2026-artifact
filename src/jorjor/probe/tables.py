from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import random
import re

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.config import load_limit_ar_train_config
from jorjor.data.split import SplitFractions, apply_split, build_split_manifest
from jorjor.train.limit_tiny import _load_points


SEEDS = ("seed_a", "seed_b", "seed_c")
BASE_RANGE_COMMITS = {
    "seed_a": "ff94fab",
    "seed_b": "3239624",
    "seed_c": "3239624",
}
CORE_CLAIM = (
    "Across three 10k and three 50k independently trained 4-layer Limit Hold'em autoregressive models, hidden "
    "states linearly expose opponent-hand range information from betting history, and this range signal remains "
    "after linearly controlling for action and value labels."
)
DEPTH_BOUNDARY = (
    "Depth robustness is not established: the stabilized 8-layer history-only residual range signal does not "
    "beat the action+value-only nuisance baseline under the tested recipe."
)
CORRECTNESS_SCORE_TOLERANCE = 1e-7
RANGE_SUPERVISED_POSITIVE_CONTROL_SEEDS = ("seed_d", "seed_f", "seed_g")


@dataclass(frozen=True)
class ProbeMetricSummary:
    """Compact best-score summary for one probe metrics file."""

    best_real: float
    best_control: float
    best_layer: str
    checksum: str


@dataclass(frozen=True)
class LayerMetricSummary:
    """Layer-aware summary for one probe metrics file."""

    best_score: float
    best_layer: str
    sequence_hidden_score: float | None
    checksum: str


def _read_metrics(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"required metrics file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"required JSON file is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return data


def _best_classification_metric(path: Path) -> ProbeMetricSummary:
    metrics = _read_metrics(path)
    records = metrics.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"metrics file has no records: {path}")

    real_records = [record for record in records if record.get("control") == "none"]
    control_records = [record for record in records if record.get("control") != "none"]
    if not real_records or not control_records:
        raise ValueError(f"metrics file must include real and control rows: {path}")

    best_real = max(real_records, key=lambda record: float(record["score"]))
    best_control = max(control_records, key=lambda record: float(record["score"]))
    checksum = metrics.get("checksum")
    if not isinstance(checksum, str):
        raise ValueError(f"metrics file is missing checksum: {path}")
    return ProbeMetricSummary(
        best_real=float(best_real["score"]),
        best_control=float(best_control["score"]),
        best_layer=str(best_real["layer"]),
        checksum=checksum,
    )


def _layer_classification_metric(path: Path) -> LayerMetricSummary:
    metrics = _read_metrics(path)
    records = metrics.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"metrics file has no records: {path}")

    real_records = [record for record in records if record.get("control") == "none"]
    if not real_records:
        raise ValueError(f"metrics file must include real rows: {path}")

    best = max(real_records, key=lambda record: float(record["score"]))
    sequence_rows = [record for record in real_records if record.get("layer") == "sequence_hidden"]
    checksum = metrics.get("checksum")
    if not isinstance(checksum, str):
        raise ValueError(f"metrics file is missing checksum: {path}")
    return LayerMetricSummary(
        best_score=float(best["score"]),
        best_layer=str(best["layer"]),
        sequence_hidden_score=float(sequence_rows[0]["score"]) if sequence_rows else None,
        checksum=checksum,
    )


def _range_path(archive_root: Path, commit: str, seed: str, suffix: str) -> Path:
    suffix_text = f"_{suffix}" if suffix else ""
    probe_name = f"limit_ar_teacher_{seed}_4layer_range{suffix_text}_10k"
    candidates = [
        archive_root / commit / "outputs" / "probes" / probe_name / "probe_metrics.json",
        archive_root / commit / "outputs" / probe_name / "probe_metrics.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _no_range_path(archive_root: Path, commit: str, seed: str, suffix: str) -> Path:
    suffix_text = f"_{suffix}" if suffix else ""
    probe_name = f"limit_ar_action_value_only_{seed}_4layer_range{suffix_text}_10k"
    candidates = [
        archive_root / commit / "outputs" / "probes" / probe_name / "probe_metrics.json",
        archive_root / commit / "outputs" / probe_name / "probe_metrics.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _no_range_path_for_size(archive_root: Path, commit: str, seed: str, suffix: str, size: str) -> Path:
    suffix_text = f"_{suffix}" if suffix else ""
    probe_name = f"limit_ar_action_value_only_{seed}_4layer_range{suffix_text}_{size}"
    candidates = [
        archive_root / commit / "outputs" / "probes" / probe_name / "probe_metrics.json",
        archive_root / commit / "outputs" / probe_name / "probe_metrics.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _range_supervised_path_for_size(archive_root: Path, commit: str, seed: str, suffix: str, size: str) -> Path:
    suffix_text = f"_{suffix}" if suffix else ""
    probe_name = f"limit_ar_teacher_{seed}_4layer_range{suffix_text}_{size}"
    candidates = [
        archive_root / commit / "outputs" / "probes" / probe_name / "probe_metrics.json",
        archive_root / commit / "outputs" / probe_name / "probe_metrics.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _no_range_train_path(archive_root: Path, commit: str, seed: str) -> Path:
    return archive_root / commit / "outputs" / f"limit_ar_action_value_only_{seed}_10k_4layer" / "metrics.json"


def _no_range_train_path_for_size(archive_root: Path, commit: str, seed: str, size: str) -> Path:
    return archive_root / commit / "outputs" / f"limit_ar_action_value_only_{seed}_{size}_4layer" / "metrics.json"


def _range_supervised_train_path_for_size(archive_root: Path, commit: str, seed: str, size: str) -> Path:
    return archive_root / commit / "outputs" / f"limit_ar_teacher_{seed}_{size}_4layer" / "metrics.json"


def _row_for_seed(archive_root: Path, seed: str, full_control_commit: str) -> dict[str, object]:
    base_commit = BASE_RANGE_COMMITS[seed]
    history = _best_classification_metric(_range_path(archive_root, base_commit, seed, "history_only"))
    history_only = _best_classification_metric(
        _range_path(archive_root, "cc431cf", seed, "history_only_action_value_only")
    )
    history_residual = _best_classification_metric(
        _range_path(archive_root, "cc431cf", seed, "history_only_residualize_action_value")
    )
    full = _best_classification_metric(_range_path(archive_root, base_commit, seed, ""))
    full_only = _best_classification_metric(_range_path(archive_root, full_control_commit, seed, "action_value_only"))
    full_residual = _best_classification_metric(
        _range_path(archive_root, full_control_commit, seed, "residualize_action_value")
    )

    return {
        "seed": seed,
        "history_only_hidden_top10": history.best_real,
        "history_only_action_value_only_top10": history_only.best_real,
        "history_only_residualize_action_value_top10": history_residual.best_real,
        "history_only_residual_delta_vs_only": history_residual.best_real - history_only.best_real,
        "full_hidden_top10": full.best_real,
        "full_action_value_only_top10": full_only.best_real,
        "full_residualize_action_value_top10": full_residual.best_real,
        "full_residual_delta_vs_only": full_residual.best_real - full_only.best_real,
        "checksums": {
            "history_only_hidden": history.checksum,
            "history_only_action_value_only": history_only.checksum,
            "history_only_residualize_action_value": history_residual.checksum,
            "full_hidden": full.checksum,
            "full_action_value_only": full_only.checksum,
            "full_residualize_action_value": full_residual.checksum,
        },
    }


def _markdown_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| seed | history hidden | history action+value only | history residual | full hidden | full action+value only | full residual |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {seed} | {history_only_hidden_top10:.6f} | {history_only_action_value_only_top10:.6f} | "
            "{history_only_residualize_action_value_top10:.6f} | {full_hidden_top10:.6f} | "
            "{full_action_value_only_top10:.6f} | {full_residualize_action_value_top10:.6f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def build_probe_tables(archive_root: Path, output_dir: Path, full_control_commit: str) -> dict[str, object]:
    """Build deterministic paper-table summaries from archived probe metrics."""

    rows = []
    for seed in SEEDS:
        rows.append(_row_for_seed(archive_root, seed, full_control_commit))

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
                "history_control_commit": "cc431cf",
        "base_range_commits": BASE_RANGE_COMMITS,
        "full_control_commit": full_control_commit,
        "rows": rows,
    }
    write_json_artifact(output_dir / "limit_ar_nuisance_summary.json", payload)
    (output_dir / "limit_ar_nuisance_summary.md").write_text(_markdown_table(rows), encoding="utf-8")
    return payload


def _read_summary(archive_root: Path, commit: str, filename: str) -> dict[str, object]:
    path = archive_root / commit / filename
    if not path.exists():
        raise FileNotFoundError(f"required summary file is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"summary file must contain an object: {path}")
    return data


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return variance ** 0.5


def _core_nuisance_section(core_summary: dict[str, object]) -> dict[str, object]:
    rows = core_summary.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("core nuisance summary must include rows")

    history_deltas = [float(row["history_only_residual_delta_vs_only"]) for row in rows]
    full_deltas = [float(row["full_residual_delta_vs_only"]) for row in rows]
    return {
        "source": "v0.8.4",
        "seed_count": len(rows),
        "history_residual_delta_mean": _mean(history_deltas),
        "history_residual_delta_min": min(history_deltas),
        "full_residual_delta_mean": _mean(full_deltas),
        "full_residual_delta_min": min(full_deltas),
        "rows": rows,
    }


def _robustness_section(summary: dict[str, object], label: str, source: str) -> dict[str, object]:
    train = summary.get("train")
    deltas = summary.get("deltas")
    rows = summary.get("rows")
    if not isinstance(train, dict) or not isinstance(deltas, dict) or not isinstance(rows, list):
        raise ValueError(f"robustness summary is malformed: {label}")
    history_delta = float(deltas["history_residual_vs_action_value_only"])
    full_delta = float(deltas["full_residual_vs_action_value_only"])
    return {
        "label": label,
        "source": source,
        "policy_accuracy": float(train["policy_accuracy"]),
        "range_top10": float(train["opponent_hand_top10_accuracy"]),
        "history_residual_delta_vs_action_value_only": history_delta,
        "full_residual_delta_vs_action_value_only": full_delta,
        "history_passes": history_delta > 0.0,
        "full_passes": full_delta > 0.0,
        "model_sha256": train["model_sha256"],
        "rows": rows,
    }


def _scale_replication_section(seed_d: dict[str, object], seed_fg: dict[str, object]) -> dict[str, object]:
    seed_rows = [
        _robustness_section(seed_d, "50k 4-layer seed-d", "v0.9.0"),
    ]
    extra_seeds = seed_fg.get("seeds")
    if not isinstance(extra_seeds, list) or not extra_seeds:
        raise ValueError("v0.9.4 summary must include seeds")
    for seed_summary in extra_seeds:
        if not isinstance(seed_summary, dict):
            raise ValueError("v0.9.4 seed summary must be an object")
        seed_name = str(seed_summary["seed"]).replace("_", "-")
        seed_rows.append(_robustness_section(seed_summary, f"50k 4-layer {seed_name}", "v0.9.4"))

    history_deltas = [float(row["history_residual_delta_vs_action_value_only"]) for row in seed_rows]
    full_deltas = [float(row["full_residual_delta_vs_action_value_only"]) for row in seed_rows]
    return {
        "label": "50k 4-layer seed-d/f/g",
        "source": "v0.9.0+v0.9.4",
        "seed_count": len(seed_rows),
        "history_residual_delta_mean": _mean(history_deltas),
        "history_residual_delta_min": min(history_deltas),
        "full_residual_delta_mean": _mean(full_deltas),
        "full_residual_delta_min": min(full_deltas),
        "all_history_pass": all(delta > 0.0 for delta in history_deltas),
        "all_full_pass": all(delta > 0.0 for delta in full_deltas),
        "rows": seed_rows,
    }


def _evidence_markdown(payload: dict[str, object]) -> str:
    core = payload["core_4layer"]
    scale = payload["scale_robustness"]
    default_depth = payload["depth_default"]
    stable_depth = payload["depth_stabilized"]
    lines = [
        "# Limit AR Paper Evidence Pack",
        "",
        "## Core Claim",
        "",
        str(payload["claim"]),
        "",
        "## Core 4-Layer Evidence",
        "",
        "| group | history delta mean | history delta min | full delta mean | full delta min |",
        "|---|---:|---:|---:|---:|",
        (
        "| v0.8.4 seed-a/b/c | {history_residual_delta_mean:.6f} | "
            "{history_residual_delta_min:.6f} | {full_residual_delta_mean:.6f} | "
            "{full_residual_delta_min:.6f} |"
        ).format(**core),
        "",
        "## 50k Scale Robustness",
        "",
        "| group | seed count | history delta mean | history delta min | full delta mean | full delta min |",
        "|---|---:|---:|---:|---:|---:|",
        (
            "| 50k 4-layer seed-d/f/g | {seed_count} | {history_residual_delta_mean:.6f} | "
            "{history_residual_delta_min:.6f} | {full_residual_delta_mean:.6f} | "
            "{full_residual_delta_min:.6f} |"
        ).format(**scale),
        "",
        "## Robustness Rows",
        "",
        "| row | policy acc | range top10 | history delta | full delta | interpretation |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in [*scale["rows"], default_depth, stable_depth]:
        interpretation = "positive" if row["history_passes"] and row["full_passes"] else "boundary"
        lines.append(
            "| {label} | {policy_accuracy:.6f} | {range_top10:.6f} | "
            "{history_residual_delta_vs_action_value_only:.6f} | "
            "{full_residual_delta_vs_action_value_only:.6f} | {interpretation} |".format(
                interpretation=interpretation,
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            str(payload["depth_boundary"]),
            "",
            "## Limitations",
            "",
            "- This is linear nuisance-control evidence, not causal independence from the training objective.",
            "- Value probe evidence is weak and should not be a main positive claim.",
            "- Depth robustness is not established by v0.9.1 or v0.9.2.",
            "",
        ]
    )
    return "\n".join(lines)


def _score_or_zero(value: float | None) -> float:
    return value if value is not None else 0.0


def _no_range_row(archive_root: Path, commit: str, seed: str) -> dict[str, object]:
    train = _read_metrics(_no_range_train_path(archive_root, commit, seed))
    history_only = _layer_classification_metric(_no_range_path(archive_root, commit, seed, "history_only"))
    history_action_value = _layer_classification_metric(
        _no_range_path(archive_root, commit, seed, "history_only_action_value_only")
    )
    history_residual = _layer_classification_metric(
        _no_range_path(archive_root, commit, seed, "history_only_residualize_action_value")
    )
    full = _layer_classification_metric(_no_range_path(archive_root, commit, seed, ""))
    full_action_value = _layer_classification_metric(_no_range_path(archive_root, commit, seed, "action_value_only"))
    full_residual = _layer_classification_metric(
        _no_range_path(archive_root, commit, seed, "residualize_action_value")
    )

    history_residual_sequence = _score_or_zero(history_residual.sequence_hidden_score)
    full_residual_sequence = _score_or_zero(full_residual.sequence_hidden_score)
    return {
        "seed": seed,
        "policy_accuracy": float(train["policy_accuracy"]),
        "value_mse": float(train["value_mse"]),
        "opponent_hand_supervised": bool(train["opponent_hand_supervised"]),
        "diagnostic_opponent_hand_top10": float(train["opponent_hand_top10_accuracy"]),
        "history_only_sequence_hidden_top10": history_only.sequence_hidden_score,
        "history_only_best_top10": history_only.best_score,
        "history_only_best_layer": history_only.best_layer,
        "history_action_value_only_top10": history_action_value.best_score,
        "history_residual_sequence_hidden_top10": history_residual.sequence_hidden_score,
        "history_residual_best_top10": history_residual.best_score,
        "history_residual_best_layer": history_residual.best_layer,
        "history_residual_sequence_delta_vs_action_value_only": history_residual_sequence - history_action_value.best_score,
        "history_residual_best_delta_vs_action_value_only": history_residual.best_score - history_action_value.best_score,
        "full_sequence_hidden_top10": full.sequence_hidden_score,
        "full_best_top10": full.best_score,
        "full_best_layer": full.best_layer,
        "full_action_value_only_top10": full_action_value.best_score,
        "full_residual_sequence_hidden_top10": full_residual.sequence_hidden_score,
        "full_residual_best_top10": full_residual.best_score,
        "full_residual_best_layer": full_residual.best_layer,
        "full_residual_sequence_delta_vs_action_value_only": full_residual_sequence - full_action_value.best_score,
        "full_residual_best_delta_vs_action_value_only": full_residual.best_score - full_action_value.best_score,
        "checksums": {
            "train": train["checksum"],
            "history_only": history_only.checksum,
            "history_action_value_only": history_action_value.checksum,
            "history_residual": history_residual.checksum,
            "full": full.checksum,
            "full_action_value_only": full_action_value.checksum,
            "full_residual": full_residual.checksum,
        },
    }


def _no_range_label_control_rows(archive_root: Path, commit: str) -> list[dict[str, object]]:
    rows = []
    for label, suffix in (
        ("real", "history_only_residualize_action_value_real_labels"),
        ("shuffle_global", "history_only_residualize_action_value_shuffle_global_labels"),
        ("shuffle_within_bucket", "history_only_residualize_action_value_shuffle_within_bucket_labels"),
    ):
        metrics_path = _no_range_path(archive_root, commit, "seed_h", suffix)
        metric = _layer_classification_metric(metrics_path)
        raw = _read_metrics(metrics_path)
        rows.append(
            {
                "row": label,
                "sequence_hidden_top10": metric.sequence_hidden_score,
                "best_top10": metric.best_score,
                "best_layer": metric.best_layer,
                "label_control": raw["label_control"],
                "label_control_stats": raw.get("label_control_stats", {}),
                "checksum": metric.checksum,
            }
        )
    return rows


def _no_range_history_control_rows(archive_root: Path, commit: str) -> list[dict[str, object]]:
    rows = []
    for label, suffix in (
        ("drop_history", "drop_history"),
        ("shuffle_history_order", "shuffle_history_order"),
        ("bag_of_actions", "bag_of_actions"),
        ("no_observation", "no_observation"),
        ("round_position_action_prior", "round_position_action_prior"),
        ("history_only_action_only", "history_only_action_only"),
    ):
        metrics_path = _no_range_path(archive_root, commit, "seed_h", suffix)
        metric = _layer_classification_metric(metrics_path)
        raw = _read_metrics(metrics_path)
        rows.append(
            {
                "row": label,
                "input_control": raw["input_control"],
                "feature_control": raw["feature_control"],
                "sequence_hidden_top10": metric.sequence_hidden_score,
                "best_top10": metric.best_score,
                "best_layer": metric.best_layer,
                "checksum": metric.checksum,
            }
        )
    return rows


def build_paper_evidence_pack(archive_root: Path, output_dir: Path) -> dict[str, object]:
    """Build a claim-oriented paper evidence pack from archived summaries."""

    core = _core_nuisance_section(build_probe_tables(archive_root, output_dir, full_control_commit="349304c"))

    scale = _scale_replication_section(
        _read_summary(archive_root, "8502bf8", "v0_9_0_summary.json"),
        _read_summary(archive_root, "8a7a72b", "v0_9_4_summary.json"),
    )
    default_depth = _robustness_section(
        _read_summary(archive_root, "7180a68", "v0_9_1_summary.json"),
        "10k 8-layer default seed-e",
        "v0.9.1",
    )
    stable_depth = _robustness_section(
        _read_summary(archive_root, "6da81c9", "v0_9_2_summary.json"),
        "10k 8-layer stabilized seed-e",
        "v0.9.2",
    )

    payload = {
                "claim": CORE_CLAIM,
        "depth_boundary": DEPTH_BOUNDARY,
        "core_4layer": core,
        "scale_robustness": scale,
        "depth_default": default_depth,
        "depth_stabilized": stable_depth,
        "recommendation": "Use 4-layer multi-seed plus 50k same-architecture robustness as the main claim block.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "limit_ar_paper_evidence_pack.json", payload)
    (output_dir / "limit_ar_paper_evidence_pack.md").write_text(_evidence_markdown(payload), encoding="utf-8")
    return payload


def _no_range_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# No-Range-Supervision Evidence Pack",
        "",
        "## Main Rule",
        "",
        "Primary pass/fail uses `sequence_hidden`. `best_over_layers` is descriptive only.",
        "",
        "## Core Rows",
        "",
        "| seed | policy acc | range supervised | diagnostic range top10 | history seq delta | history best delta | full seq delta | full best delta |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["core_rows"]:
        lines.append(
            "| {seed} | {policy_accuracy:.6f} | {opponent_hand_supervised} | "
            "{diagnostic_opponent_hand_top10:.6f} | "
            "{history_residual_sequence_delta_vs_action_value_only:.6f} | "
            "{history_residual_best_delta_vs_action_value_only:.6f} | "
            "{full_residual_sequence_delta_vs_action_value_only:.6f} | "
            "{full_residual_best_delta_vs_action_value_only:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Label Controls",
            "",
            "| row | sequence_hidden top10 | best top10 | best layer |",
            "|---|---:|---:|---|",
        ]
    )
    for row in payload["label_controls_seed_h"]:
        lines.append(
            "| {row} | {sequence_hidden_top10:.6f} | {best_top10:.6f} | {best_layer} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## History Controls",
            "",
            "| row | input control | feature control | sequence_hidden top10 | best top10 | best layer |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for row in payload["history_controls_seed_h"]:
        sequence_value = row["sequence_hidden_top10"]
        sequence_text = "n/a" if sequence_value is None else f"{float(sequence_value):.6f}"
        lines.append(
            "| {row} | {input_control} | {feature_control} | {sequence} | {best_top10:.6f} | {best_layer} |".format(
                sequence=sequence_text,
                **row,
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_no_range_evidence_pack(
    archive_root: Path,
    output_dir: Path,
    commit: str = "0aac660",
) -> dict[str, object]:
    """Build a no-range-supervision evidence pack with predeclared layer discipline."""

    core_rows = [_no_range_row(archive_root, commit, seed) for seed in ("seed_h", "seed_i", "seed_j")]
    history_sequence_deltas = [float(row["history_residual_sequence_delta_vs_action_value_only"]) for row in core_rows]
    full_sequence_deltas = [float(row["full_residual_sequence_delta_vs_action_value_only"]) for row in core_rows]
    payload = {
                "commit": commit,
        "primary_layer_rule": "sequence_hidden",
        "best_layer_rule": "descriptive_only",
        "core_rows": core_rows,
        "history_sequence_delta_mean": _mean(history_sequence_deltas),
        "history_sequence_delta_min": min(history_sequence_deltas),
        "full_sequence_delta_mean": _mean(full_sequence_deltas),
        "full_sequence_delta_min": min(full_sequence_deltas),
        "label_controls_seed_h": _no_range_label_control_rows(archive_root, commit),
        "history_controls_seed_h": _no_range_history_control_rows(archive_root, commit),
        "recommendation": "Use this pack only if sequence_hidden deltas and label/history controls support the claim.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "limit_ar_no_range_evidence_pack.json", payload)
    (output_dir / "limit_ar_no_range_evidence_pack.md").write_text(_no_range_markdown(payload), encoding="utf-8")
    return payload


def _real_scores_by_layer(path: Path) -> dict[str, float]:
    metrics = _read_metrics(path)
    records = metrics.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"metrics file has no records: {path}")
    return {
        str(record["layer"]): float(record["score"])
        for record in records
        if record.get("control") == "none"
    }


def _repeat_index_from_suffix(suffix: str) -> int:
    match = re.search(r"_repeat_(\d+)$", suffix)
    if match:
        return int(match.group(1))
    return 0


def _config_path_for_no_range_metric(path: Path, seed: str, suffix: str) -> Path:
    probe_name = path.parent.name
    archive_root = path.parents[4]
    commit = path.parents[3].name
    archive_config = archive_root / commit / "configs" / "probe" / f"{probe_name}.json"
    if archive_config.exists():
        return archive_config
    repo_config = Path("configs") / "probe" / f"{probe_name}.json"
    if repo_config.exists():
        return repo_config
    expected_name = f"limit_ar_action_value_only_{seed}_4layer_range_{suffix}_10k.json"
    fallback = Path("configs") / "probe" / expected_name
    if fallback.exists():
        return fallback
    return archive_config


def _probe_metadata(path: Path, seed: str, suffix: str, *, strict: bool = False) -> dict[str, object]:
    metrics = _read_metrics(path)
    config_path = _config_path_for_no_range_metric(path, seed, suffix)
    config_data: dict[str, object] = {}
    if config_path.exists():
        config_data = _read_json_object(config_path)
    if strict and ("probe_seed" not in metrics or "control_seed" not in metrics):
        raise ValueError(f"strict probe metadata requires metrics-embedded seeds: {path}")

    def value(key: str) -> object:
        if key in metrics:
            return metrics[key]
        if key in config_data:
            return config_data[key]
        raise ValueError(f"probe metadata is missing {key}: {path}")

    metadata = {
        "seed": seed,
        "suffix": suffix,
        "repeat_index": _repeat_index_from_suffix(suffix),
        "path": str(path),
        "metadata_source": "metrics" if "probe_seed" in metrics and "control_seed" in metrics else "config_fallback",
        "config_path": str(metrics.get("config_path", config_path)),
        "artifact_dir": str(metrics.get("artifact_dir", config_data.get("artifact_dir", path.parent))),
        "version": str(value("version") if "version" in metrics else value("run_version")),
        "train_config": str(value("train_config")),
        "checkpoint_path": str(value("checkpoint_path")),
        "target": str(value("target")),
        "input_control": str(value("input_control")),
        "feature_control": str(value("feature_control")),
        "label_control": str(value("label_control")),
        "probe_seed": int(value("probe_seed")),
        "control_seed": int(value("control_seed")),
        "layers": list(value("layers")),
        "checksum": str(value("checksum")),
    }
    return metadata


def _range_supervised_probe_metadata(path: Path, seed: str, suffix: str, *, strict: bool = False) -> dict[str, object]:
    metrics = _read_metrics(path)
    config_path = _config_path_for_range_supervised_metric(path, seed, suffix)
    config_data: dict[str, object] = {}
    if config_path.exists():
        config_data = _read_json_object(config_path)
    if strict and ("probe_seed" not in metrics or "control_seed" not in metrics):
        raise ValueError(f"strict probe metadata requires metrics-embedded seeds: {path}")

    def value(key: str) -> object:
        if key in metrics:
            return metrics[key]
        if key in config_data:
            return config_data[key]
        raise ValueError(f"range-supervised probe metadata is missing {key}: {path}")

    return {
        "seed": seed,
        "suffix": suffix,
        "repeat_index": _repeat_index_from_suffix(suffix),
        "path": str(path),
        "metadata_source": "metrics" if "probe_seed" in metrics and "control_seed" in metrics else "config_fallback",
        "config_path": str(metrics.get("config_path", config_path)),
        "artifact_dir": str(metrics.get("artifact_dir", config_data.get("artifact_dir", path.parent))),
        "version": str(value("version") if "version" in metrics else value("run_version")),
        "train_config": str(value("train_config")),
        "checkpoint_path": str(value("checkpoint_path")),
        "target": str(value("target")),
        "input_control": str(value("input_control")),
        "feature_control": str(value("feature_control")),
        "label_control": str(value("label_control")),
        "probe_seed": int(value("probe_seed")),
        "control_seed": int(value("control_seed")),
        "layers": list(value("layers")),
        "checksum": str(value("checksum")),
    }


def _config_path_for_range_supervised_metric(path: Path, seed: str, suffix: str) -> Path:
    probe_name = path.parent.name
    archive_root = path.parents[4]
    commit = path.parents[3].name
    archive_config = archive_root / commit / "configs" / "probe" / f"{probe_name}.json"
    if archive_config.exists():
        return archive_config
    repo_config = Path("configs") / "probe" / f"{probe_name}.json"
    if repo_config.exists():
        return repo_config
    expected_name = f"limit_ar_teacher_{seed}_4layer_range_{suffix}_50k.json"
    fallback = Path("configs") / "probe" / expected_name
    if fallback.exists():
        return fallback
    return archive_config


def _repeat_paths(archive_root: Path, commit: str, seed: str, suffix: str) -> list[Path]:
    paths = [_no_range_path(archive_root, commit, seed, suffix)]
    for index in (1, 2, 3):
        paths.append(
            _no_range_path(
                archive_root,
                commit,
                seed,
                f"{suffix}_repeat_{index}",
            )
        )
    return paths


def _uncertainty_rows(
    archive_root: Path,
    commit: str,
    suffix: str,
    feature_control: str,
) -> list[dict[str, object]]:
    rows = []
    for seed in ("seed_h", "seed_i", "seed_j"):
        layer_scores: dict[str, list[float]] = {}
        checksums = []
        probe_seeds = []
        control_seeds = []
        metadata_sources = []
        for path in _repeat_paths(archive_root, commit, seed, suffix):
            metadata = _probe_metadata(path, seed, suffix)
            checksums.append(metadata["checksum"])
            probe_seeds.append(metadata["probe_seed"])
            control_seeds.append(metadata["control_seed"])
            metadata_sources.append(metadata["metadata_source"])
            for layer, score in _real_scores_by_layer(path).items():
                layer_scores.setdefault(layer, []).append(score)
        for layer, scores in sorted(layer_scores.items()):
            rows.append(
                {
                    "seed": seed,
                    "feature_control": feature_control,
                    "layer": layer,
                    "count": len(scores),
                    "mean_top10": _mean(scores),
                    "std_top10": _std(scores),
                    "min_top10": min(scores),
                    "max_top10": max(scores),
                    "scores": scores,
                    "probe_seeds": probe_seeds,
                    "control_seeds": control_seeds,
                    "checksums": checksums,
                    "metadata_sources": metadata_sources,
                }
            )
    return rows


def _nuisance_score(path: Path) -> float:
    scores = _real_scores_by_layer(path)
    if "nuisance" in scores:
        return scores["nuisance"]
    if len(scores) == 1:
        return next(iter(scores.values()))
    raise ValueError(f"nuisance baseline must include exactly one nuisance score: {path}")


def _paired_delta_rows(archive_root: Path, commit: str) -> list[dict[str, object]]:
    rows = []
    for seed in ("seed_h", "seed_i", "seed_j"):
        residual_paths = _repeat_paths(archive_root, commit, seed, "history_only_residualize_action_value")
        baseline_paths = _repeat_paths(archive_root, commit, seed, "history_only_action_value_only")
        layer_deltas: dict[str, list[float]] = {}
        layer_residual_scores: dict[str, list[float]] = {}
        baseline_scores = []
        checksums = []
        pair_metadata = []
        residual_probe_seeds = []
        residual_control_seeds = []
        baseline_probe_seeds = []
        baseline_control_seeds = []
        for residual_path, baseline_path in zip(residual_paths, baseline_paths, strict=True):
            residual_suffix = residual_path.parent.name.replace(f"limit_ar_action_value_only_{seed}_4layer_range_", "")
            baseline_suffix = baseline_path.parent.name.replace(f"limit_ar_action_value_only_{seed}_4layer_range_", "")
            residual_metadata = _probe_metadata(residual_path, seed, residual_suffix.removesuffix("_10k"))
            baseline_metadata = _probe_metadata(baseline_path, seed, baseline_suffix.removesuffix("_10k"))
            _validate_no_range_pair(residual_metadata, baseline_metadata)
            checksums.append(
                {
                    "residual": residual_metadata["checksum"],
                    "baseline": baseline_metadata["checksum"],
                }
            )
            pair_metadata.append(
                {
                    "repeat_index": residual_metadata["repeat_index"],
                    "residual_probe_seed": residual_metadata["probe_seed"],
                    "residual_control_seed": residual_metadata["control_seed"],
                    "baseline_probe_seed": baseline_metadata["probe_seed"],
                    "baseline_control_seed": baseline_metadata["control_seed"],
                    "metadata_sources": [
                        residual_metadata["metadata_source"],
                        baseline_metadata["metadata_source"],
                    ],
                }
            )
            residual_probe_seeds.append(residual_metadata["probe_seed"])
            residual_control_seeds.append(residual_metadata["control_seed"])
            baseline_probe_seeds.append(baseline_metadata["probe_seed"])
            baseline_control_seeds.append(baseline_metadata["control_seed"])
            baseline_score = _nuisance_score(baseline_path)
            baseline_scores.append(baseline_score)
            for layer, residual_score in _real_scores_by_layer(residual_path).items():
                layer_deltas.setdefault(layer, []).append(residual_score - baseline_score)
                layer_residual_scores.setdefault(layer, []).append(residual_score)

        for layer, deltas in sorted(layer_deltas.items()):
            rows.append(
                {
                    "seed": seed,
                    "layer": layer,
                    "count": len(deltas),
                    "mean_delta": _mean(deltas),
                    "std_delta": _std(deltas),
                    "min_delta": min(deltas),
                    "max_delta": max(deltas),
                    "deltas": deltas,
                    "residual_scores": layer_residual_scores[layer],
                    "baseline_scores": baseline_scores,
                    "residual_probe_seeds": residual_probe_seeds,
                    "residual_control_seeds": residual_control_seeds,
                    "baseline_probe_seeds": baseline_probe_seeds,
                    "baseline_control_seeds": baseline_control_seeds,
                    "pair_metadata": pair_metadata,
                    "checksums": checksums,
                }
            )
    return rows


def _validate_no_range_pair(
    residual_metadata: dict[str, object],
    baseline_metadata: dict[str, object],
    *,
    require_same_version: bool = True,
) -> None:
    keys = ("seed", "repeat_index", "train_config", "checkpoint_path", "target", "input_control", "label_control")
    if require_same_version:
        keys = (*keys, "version")
    for key in keys:
        if residual_metadata[key] != baseline_metadata[key]:
            raise ValueError(
                "paired no-range probe metadata mismatch "
                f"for {key}: {residual_metadata[key]} != {baseline_metadata[key]}"
            )
    if residual_metadata["feature_control"] != "residualize_action_value":
        raise ValueError("paired residual row must use residualize_action_value")
    if baseline_metadata["feature_control"] != "action_value_only":
        raise ValueError("paired baseline row must use action_value_only")


def _validate_range_supervised_pair(
    residual_metadata: dict[str, object],
    baseline_metadata: dict[str, object],
    *,
    residual_feature_control: str,
    baseline_feature_control: str,
) -> None:
    keys = ("seed", "repeat_index", "train_config", "checkpoint_path", "target", "input_control", "label_control")
    for key in keys:
        if residual_metadata[key] != baseline_metadata[key]:
            raise ValueError(
                "paired range-supervised probe metadata mismatch "
                f"for {key}: {residual_metadata[key]} != {baseline_metadata[key]}"
            )
    if residual_metadata["feature_control"] != residual_feature_control:
        raise ValueError(f"range-supervised residual row must use {residual_feature_control}")
    if baseline_metadata["feature_control"] != baseline_feature_control:
        raise ValueError(f"range-supervised baseline row must use {baseline_feature_control}")


def _uncertainty_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# No-Range Probe Uncertainty",
        "",
        "## Residual Rows",
        "",
        "| seed | feature control | layer | count | mean top10 | std top10 | min | max |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {seed} | {feature_control} | {layer} | {count} | {mean_top10:.6f} | {std_top10:.6f} | "
            "{min_top10:.6f} | {max_top10:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Baseline Rows",
            "",
            "| seed | feature control | layer | count | mean top10 | std top10 | min | max |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["baseline_rows"]:
        lines.append(
            "| {seed} | {feature_control} | {layer} | {count} | {mean_top10:.6f} | {std_top10:.6f} | "
            "{min_top10:.6f} | {max_top10:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Paired Deltas",
            "",
            "| seed | layer | count | mean delta | std delta | min | max |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["delta_rows"]:
        lines.append(
            "| {seed} | {layer} | {count} | {mean_delta:.6f} | {std_delta:.6f} | "
            "{min_delta:.6f} | {max_delta:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Decision Inputs",
            "",
            "- Use paired deltas, not raw residual scores, for the no-range claim gate.",
            "- Do not use this table alone as a claim; it measures probe-seed stability for rows selected by v1.0.2-v1.0.4.",
            "",
        ]
    )
    return "\n".join(lines)


def build_no_range_uncertainty_pack(
    archive_root: Path,
    output_dir: Path,
    commit: str = "9a368b2",
) -> dict[str, object]:
    """Build probe-seed uncertainty summaries for no-range-supervision probes."""

    rows = _uncertainty_rows(
        archive_root,
        commit,
        "history_only_residualize_action_value",
        "residualize_action_value",
    )
    baseline_rows = _uncertainty_rows(
        archive_root,
        commit,
        "history_only_action_value_only",
        "action_value_only",
    )
    delta_rows = _paired_delta_rows(archive_root, commit)
    sequence_rows = [row for row in rows if row["layer"] == "sequence_hidden"]
    encoder2_rows = [row for row in rows if row["layer"] == "encoder_layer_2"]
    sequence_delta_rows = [row for row in delta_rows if row["layer"] == "sequence_hidden"]
    encoder2_delta_rows = [row for row in delta_rows if row["layer"] == "encoder_layer_2"]
    payload = {
                "commit": commit,
        "probe_repeat_count": 4,
        "paired_repeat_count": 4,
        "rows": rows,
        "baseline_rows": baseline_rows,
        "delta_rows": delta_rows,
        "sequence_hidden_mean_std": _mean([float(row["std_top10"]) for row in sequence_rows]),
        "encoder_layer_2_mean_std": _mean([float(row["std_top10"]) for row in encoder2_rows]),
        "sequence_hidden_delta_mean": _mean([float(row["mean_delta"]) for row in sequence_delta_rows]),
        "sequence_hidden_delta_min": min(float(row["mean_delta"]) for row in sequence_delta_rows),
        "sequence_hidden_delta_mean_std": _mean([float(row["std_delta"]) for row in sequence_delta_rows]),
        "encoder_layer_2_delta_mean": _mean([float(row["mean_delta"]) for row in encoder2_delta_rows]),
        "encoder_layer_2_delta_min": min(float(row["mean_delta"]) for row in encoder2_delta_rows),
        "encoder_layer_2_delta_mean_std": _mean([float(row["std_delta"]) for row in encoder2_delta_rows]),
        "recommendation": "Use with v1.0.2-v1.0.4 deltas before deciding whether to scale no-range runs.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "limit_ar_no_range_uncertainty_pack.json", payload)
    (output_dir / "limit_ar_no_range_uncertainty_pack.md").write_text(
        _uncertainty_markdown(payload),
        encoding="utf-8",
    )
    return payload


def _locked_layer_row(archive_root: Path, commit: str, seed: str) -> dict[str, object]:
    train = _read_metrics(_no_range_train_path_for_size(archive_root, commit, seed, "50k"))
    residual_path = _no_range_path_for_size(
        archive_root,
        commit,
        seed,
        "history_only_residualize_action_value",
        "50k",
    )
    baseline_path = _no_range_path_for_size(
        archive_root,
        commit,
        seed,
        "history_only_action_value_only",
        "50k",
    )
    residual_metadata = _probe_metadata(residual_path, seed, "history_only_residualize_action_value", strict=True)
    baseline_metadata = _probe_metadata(baseline_path, seed, "history_only_action_value_only", strict=True)
    _validate_no_range_pair(residual_metadata, baseline_metadata)

    residual_scores = _real_scores_by_layer(residual_path)
    baseline_score = _nuisance_score(baseline_path)
    layer_rows = []
    for layer in ("sequence_hidden", "encoder_layer_2"):
        if layer not in residual_scores:
            raise ValueError(f"locked layer is missing from residual probe: {layer}")
        layer_rows.append(
            {
                "layer": layer,
                "residual_top10": residual_scores[layer],
                "baseline_top10": baseline_score,
                "delta": residual_scores[layer] - baseline_score,
            }
        )

    return {
        "seed": seed,
        "policy_accuracy": float(train["policy_accuracy"]),
        "value_mse": float(train["value_mse"]),
        "opponent_hand_supervised": bool(train["opponent_hand_supervised"]),
        "diagnostic_opponent_hand_top10": float(train["opponent_hand_top10_accuracy"]),
        "model_checksum": train["checksum"],
        "layers": layer_rows,
        "residual_probe_seed": residual_metadata["probe_seed"],
        "residual_control_seed": residual_metadata["control_seed"],
        "baseline_probe_seed": baseline_metadata["probe_seed"],
        "baseline_control_seed": baseline_metadata["control_seed"],
        "metadata_sources": [
            residual_metadata["metadata_source"],
            baseline_metadata["metadata_source"],
        ],
        "checksums": {
            "residual": residual_metadata["checksum"],
            "baseline": baseline_metadata["checksum"],
        },
    }


def _locked_layer_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# No-Range Fresh Locked-Layer Verification",
        "",
        "## Rule",
        "",
        "- `sequence_hidden` is the final-hidden boundary row.",
        "- `encoder_layer_2` is the locked candidate layer.",
        "- `best_over_layers` is not used for pass/fail.",
        "",
        "## Training Rows",
        "",
        "| seed | policy acc | value MSE | range supervised | diagnostic range top10 |",
        "|---|---:|---:|---|---:|",
    ]
    for row in payload["seed_rows"]:
        lines.append(
            "| {seed} | {policy_accuracy:.6f} | {value_mse:.6f} | {opponent_hand_supervised} | "
            "{diagnostic_opponent_hand_top10:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Locked Deltas",
            "",
            "| seed | layer | residual top10 | action/value baseline | delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in payload["delta_rows"]:
        lines.append(
            "| {seed} | {layer} | {residual_top10:.6f} | {baseline_top10:.6f} | {delta:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| layer | mean delta | min delta | all positive |",
            "|---|---:|---:|---|",
            (
                "| sequence_hidden | {sequence_hidden_delta_mean:.6f} | "
                "{sequence_hidden_delta_min:.6f} | {sequence_hidden_all_positive} |"
            ).format(**payload),
            (
                "| encoder_layer_2 | {encoder_layer_2_delta_mean:.6f} | "
                "{encoder_layer_2_delta_min:.6f} | {encoder_layer_2_all_positive} |"
            ).format(**payload),
            "",
        ]
    )
    return "\n".join(lines)


def build_no_range_locked_layer_pack(
    archive_root: Path,
    output_dir: Path,
    commit: str,
) -> dict[str, object]:
    """Build the v1.0.9 fresh 50k locked-layer verification pack."""

    seed_rows = [_locked_layer_row(archive_root, commit, seed) for seed in ("seed_k", "seed_l", "seed_m")]
    delta_rows = []
    for seed_row in seed_rows:
        for layer_row in seed_row["layers"]:
            delta_rows.append({"seed": seed_row["seed"], **layer_row})
    sequence_deltas = [float(row["delta"]) for row in delta_rows if row["layer"] == "sequence_hidden"]
    encoder2_deltas = [float(row["delta"]) for row in delta_rows if row["layer"] == "encoder_layer_2"]
    payload = {
                "commit": commit,
        "version": "1.0.9",
        "size": "50k",
        "final_hidden_boundary_layer": "sequence_hidden",
        "locked_candidate_layer": "encoder_layer_2",
        "best_over_layers_rule": "descriptive_only_not_used_for_pass_fail",
        "seed_rows": seed_rows,
        "delta_rows": delta_rows,
        "sequence_hidden_delta_mean": _mean(sequence_deltas),
        "sequence_hidden_delta_min": min(sequence_deltas),
        "sequence_hidden_all_positive": all(delta > 0.0 for delta in sequence_deltas),
        "encoder_layer_2_delta_mean": _mean(encoder2_deltas),
        "encoder_layer_2_delta_min": min(encoder2_deltas),
        "encoder_layer_2_all_positive": all(delta > 0.0 for delta in encoder2_deltas),
        "recommendation": "Upgrade the no-range locked-layer claim only if encoder_layer_2 passes on fresh seeds.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "limit_ar_no_range_locked_layer_pack.json", payload)
    (output_dir / "limit_ar_no_range_locked_layer_pack.md").write_text(
        _locked_layer_markdown(payload),
        encoding="utf-8",
    )
    return payload


def _correctness_artifact_path(metrics_path: Path) -> Path:
    metrics = _read_metrics(metrics_path)
    if metrics.get("target") != "opponent_hand":
        raise ValueError(f"bootstrap correctness requires opponent_hand target: {metrics_path}")
    artifact = metrics.get("correctness_artifact")
    if not isinstance(artifact, str) or not artifact:
        raise ValueError(f"metrics file has no correctness artifact: {metrics_path}")
    return metrics_path.parent / artifact


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_score(metrics_path: Path, layer: str, control: str) -> float:
    metrics = _read_metrics(metrics_path)
    records = metrics.get("records")
    if not isinstance(records, list):
        raise ValueError(f"metrics file has no records: {metrics_path}")
    for record in records:
        if record.get("layer") == layer and record.get("control") == control:
            return float(record["score"])
    raise ValueError(f"metrics file is missing score for {layer}/{control}: {metrics_path}")


def _validate_correctness_provenance(metrics_path: Path, correctness_path: Path, layer: str, control: str, count: int) -> str:
    metrics = _read_metrics(metrics_path)
    sha256 = _file_sha256(correctness_path)
    expected_sha256 = metrics.get("correctness_sha256")
    if expected_sha256 is not None and expected_sha256 != sha256:
        raise ValueError(f"correctness sha256 mismatch: {correctness_path}")
    summary = metrics.get("correctness_summary")
    if isinstance(summary, dict):
        if int(summary.get("row_count", 0)) <= 0:
            raise ValueError(f"correctness summary row_count is invalid: {metrics_path}")
        rows_by_layer_control = summary.get("rows_by_layer_control")
        if isinstance(rows_by_layer_control, dict):
            expected_count = rows_by_layer_control.get(f"{layer}|{control}")
            if expected_count is not None and int(expected_count) != count:
                raise ValueError(
                    "correctness summary count mismatch "
                    f"for {layer}/{control}: {expected_count} != {count}"
                )
    return sha256


def _correctness_by_point(metrics_path: Path, layer: str, control: str = "none") -> tuple[dict[str, dict[str, object]], str]:
    correctness_path = _correctness_artifact_path(metrics_path)
    if not correctness_path.exists():
        raise FileNotFoundError(f"required correctness artifact is missing: {correctness_path}")
    metrics = _read_metrics(metrics_path)
    expected_sha256 = metrics.get("correctness_sha256")
    sha256 = _file_sha256(correctness_path)
    if expected_sha256 is not None and expected_sha256 != sha256:
        raise ValueError(f"correctness sha256 mismatch: {correctness_path}")

    rows: dict[str, dict[str, object]] = {}
    with correctness_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if row.get("control") != control or row.get("layer") != layer:
                continue
            if row.get("split") != "test":
                raise ValueError(f"correctness row split is not test: {correctness_path}:{line_number}")
            example_index = int(row["example_index"])
            point_id = str(row.get("point_id", ""))
            hand_id = str(row.get("hand_id", ""))
            if not point_id or not hand_id:
                raise ValueError(f"correctness row is missing point identity: {correctness_path}:{line_number}")
            if point_id in rows:
                raise ValueError(f"duplicate correctness point_id for layer {layer}: {correctness_path}:{point_id}")
            if not isinstance(row.get("top10_correct"), bool):
                raise ValueError(f"top10_correct must be bool: {correctness_path}:{line_number}")
            target_label = row.get("target_label", row.get("target"))
            rows[point_id] = {
                "example_index": example_index,
                "point_id": point_id,
                "hand_id": hand_id,
                "step_index": int(row["step_index"]),
                "label": int(target_label),
                "top10_correct": bool(row["top10_correct"]),
            }
    if not rows:
        raise ValueError(f"correctness artifact has no rows for layer {layer}: {correctness_path}")
    score = _mean([1.0 if row["top10_correct"] else 0.0 for row in rows.values()])
    expected_score = _expected_score(metrics_path, layer, control)
    if abs(score - expected_score) > CORRECTNESS_SCORE_TOLERANCE:
        raise ValueError(
            "correctness aggregate does not match metrics "
            f"for {layer}/{control}: {score} != {expected_score}"
        )
    return rows, _validate_correctness_provenance(metrics_path, correctness_path, layer, control, len(rows))


def _paired_correctness_deltas_for_layers(
    residual_path: Path,
    residual_layer: str,
    baseline_path: Path,
    baseline_layer: str,
) -> dict[str, object]:
    residual_rows, residual_sha256 = _correctness_by_point(residual_path, residual_layer)
    baseline_rows, baseline_sha256 = _correctness_by_point(baseline_path, baseline_layer)
    residual_ids = sorted(residual_rows)
    baseline_ids = sorted(baseline_rows)
    if residual_ids != baseline_ids:
        raise ValueError(
            "correctness point_id mismatch "
            f"for layers {residual_layer}/{baseline_layer}: residual={len(residual_ids)} "
            f"baseline={len(baseline_ids)}"
        )

    deltas = []
    residual_values = []
    baseline_values = []
    cluster_deltas: dict[str, list[float]] = {}
    point_deltas: dict[str, dict[str, object]] = {}
    paired_outcomes = {
        "both_correct": 0,
        "both_wrong": 0,
        "residual_only_correct": 0,
        "baseline_only_correct": 0,
    }
    for point_id in residual_ids:
        residual_row = residual_rows[point_id]
        baseline_row = baseline_rows[point_id]
        for key in ("example_index", "hand_id", "step_index"):
            if residual_row[key] != baseline_row[key]:
                raise ValueError(f"correctness pair identity mismatch for {key}: {point_id}")
        if residual_row["label"] != baseline_row["label"]:
            raise ValueError(f"correctness target mismatch for point_id {point_id}")
        residual_value = 1.0 if residual_row["top10_correct"] else 0.0
        baseline_value = 1.0 if baseline_row["top10_correct"] else 0.0
        if residual_value and baseline_value:
            paired_outcomes["both_correct"] += 1
        elif not residual_value and not baseline_value:
            paired_outcomes["both_wrong"] += 1
        elif residual_value:
            paired_outcomes["residual_only_correct"] += 1
        else:
            paired_outcomes["baseline_only_correct"] += 1
        residual_values.append(residual_value)
        baseline_values.append(baseline_value)
        delta = residual_value - baseline_value
        deltas.append(delta)
        cluster_deltas.setdefault(str(residual_row["hand_id"]), []).append(delta)
        point_deltas[point_id] = {
            "point_id": point_id,
            "example_index": residual_row["example_index"],
            "hand_id": residual_row["hand_id"],
            "step_index": residual_row["step_index"],
            "label": residual_row["label"],
            "delta": delta,
        }
    return {
        "deltas": deltas,
        "cluster_means": [_mean(values) for values in cluster_deltas.values()],
        "cluster_delta_by_hand": {hand_id: _mean(values) for hand_id, values in cluster_deltas.items()},
        "hand_count": len(cluster_deltas),
        "point_deltas": point_deltas,
        "residual_top10": _mean(residual_values),
        "baseline_top10": _mean(baseline_values),
        "paired_outcomes": paired_outcomes,
        "residual_correctness_sha256": residual_sha256,
        "baseline_correctness_sha256": baseline_sha256,
    }


def _limit_action_events(point) -> list[str]:
    return [
        str(event["action"])
        for event in point.visible_state["betting_history"]
        if event.get("type") == "action"
    ]


def _limit_composition_bucket(point) -> str:
    actions = _limit_action_events(point)
    counts = {action: actions.count(action) for action in ("bet", "call", "check", "fold")}
    value_bucket = int(point.terminal_return or 0)
    legal = ",".join(sorted(str(action) for action in point.legal_actions))
    last_action = actions[-1] if actions else "none"
    return "|".join(
        [
            f"player={point.acting_player}",
            f"round={point.visible_state['round']}",
            f"legal={legal}",
            f"action={point.action}",
            f"value={value_bucket}",
            f"len={len(actions)}",
            *(f"{action}={counts[action]}" for action in ("bet", "call", "check", "fold")),
            f"last={last_action}",
        ]
    )


def _test_point_buckets_from_train_config(train_config_path: Path) -> dict[str, str]:
    train_config = load_limit_ar_train_config(train_config_path)
    points = _load_points(train_config)
    manifest = build_split_manifest(
        points=points,
        fractions=SplitFractions(
            train=train_config.train_fraction,
            validation=train_config.validation_fraction,
            test=train_config.test_fraction,
        ),
        seed=train_config.split_seed,
    )
    splits = apply_split(points, manifest)
    return {
        f"{point.hand_id}:{point.step_index}": _limit_composition_bucket(point)
        for point in splits.test
    }


def _test_point_buckets_for_positive_control(commit_root: Path, seed: str) -> tuple[dict[str, str], str]:
    sidecar = commit_root / "outputs" / "tables" / f"range_supervised_{seed}_matched_composition_buckets.json"
    if sidecar.exists():
        data = _read_json_object(sidecar)
        buckets = data.get("buckets")
        if not isinstance(buckets, dict) or not buckets:
            raise ValueError(f"matched composition bucket sidecar has no buckets: {sidecar}")
        return {str(key): str(value) for key, value in buckets.items()}, "sidecar"
    train_config = commit_root / "configs" / "train" / f"limit_ar_teacher_{seed}_50k_4layer.json"
    if not train_config.exists():
        train_config = Path("configs") / "train" / f"limit_ar_teacher_{seed}_50k_4layer.json"
    return _test_point_buckets_from_train_config(train_config), "train_config"


def _paired_correctness_deltas(
    residual_path: Path,
    baseline_path: Path,
    layer: str,
) -> dict[str, object]:
    return _paired_correctness_deltas_for_layers(residual_path, layer, baseline_path, "nuisance")


def _bootstrap_mean_ci(
    values: list[float],
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, object]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    if bootstrap_count <= 0:
        raise ValueError("bootstrap_count must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    sample_size = len(values)
    value_counts: dict[float, int] = {}
    for value in values:
        float_value = float(value)
        value_counts[float_value] = value_counts.get(float_value, 0) + 1
    unique_values = sorted(value_counts)
    weights = [value_counts[value] / sample_size for value in unique_values]
    bootstrap_engine = "python_random_choices"
    try:
        import numpy as np

        rng = np.random.default_rng(bootstrap_seed)
        counts = rng.multinomial(sample_size, weights, size=bootstrap_count)
        value_array = np.asarray(unique_values, dtype=float)
        bootstrap_means = (counts @ value_array / sample_size).tolist()
        bootstrap_engine = "numpy_multinomial"
    except ImportError:
        rng = random.Random(bootstrap_seed)
        bootstrap_means = [
            sum(rng.choices(unique_values, weights=weights, k=sample_size)) / sample_size
            for _ in range(bootstrap_count)
        ]
    bootstrap_means.sort()

    alpha = (1.0 - confidence_level) / 2.0
    lower_index = int(alpha * (bootstrap_count - 1))
    upper_index = int((1.0 - alpha) * (bootstrap_count - 1))
    return {
        "mean_delta": _mean(values),
        "lower_delta": bootstrap_means[lower_index],
        "upper_delta": bootstrap_means[upper_index],
        "bootstrap_engine": bootstrap_engine,
        "unique_value_count": len(unique_values),
    }


def _bootstrap_layer_row(
    archive_root: Path,
    commit: str,
    seed: str,
    layer: str,
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, object]:
    train_path = _no_range_train_path_for_size(archive_root, commit, seed, "50k")
    train = _read_metrics(train_path)
    if bool(train.get("opponent_hand_supervised", True)):
        raise ValueError(f"no-range bootstrap requires opponent_hand_supervised=false: {train_path}")
    if "opponent_hand_loss_weight" not in train:
        raise ValueError(f"no-range bootstrap requires opponent_hand_loss_weight metadata: {train_path}")
    if float(train["opponent_hand_loss_weight"]) != 0.0:
        raise ValueError(f"no-range bootstrap requires opponent_hand_loss_weight=0.0: {train_path}")
    residual_path = _no_range_path_for_size(
        archive_root,
        commit,
        seed,
        "history_only_residualize_action_value",
        "50k",
    )
    baseline_path = _no_range_path_for_size(
        archive_root,
        commit,
        seed,
        "history_only_action_value_only",
        "50k",
    )
    residual_metadata = _probe_metadata(residual_path, seed, "history_only_residualize_action_value", strict=True)
    baseline_metadata = _probe_metadata(baseline_path, seed, "history_only_action_value_only", strict=True)
    _validate_no_range_pair(residual_metadata, baseline_metadata)

    paired = _paired_correctness_deltas(residual_path, baseline_path, layer)
    decision_interval = _bootstrap_mean_ci(
        list(paired["deltas"]),
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    cluster_interval = _bootstrap_mean_ci(
        list(paired["cluster_means"]),
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed + 100000,
        confidence_level=confidence_level,
    )
    return {
        "seed": seed,
        "layer": layer,
        "resampling_unit": "hand_id",
        "sample_count": len(paired["deltas"]),
        "hand_count": paired["hand_count"],
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "residual_top10": paired["residual_top10"],
        "baseline_top10": paired["baseline_top10"],
        **cluster_interval,
        "decision_point_mean_delta": decision_interval["mean_delta"],
        "decision_point_lower_delta": decision_interval["lower_delta"],
        "decision_point_upper_delta": decision_interval["upper_delta"],
        "paired_outcomes": paired["paired_outcomes"],
        "residual_checksum": residual_metadata["checksum"],
        "baseline_checksum": baseline_metadata["checksum"],
        "residual_correctness_sha256": paired["residual_correctness_sha256"],
        "baseline_correctness_sha256": paired["baseline_correctness_sha256"],
        "residual_probe_seed": residual_metadata["probe_seed"],
        "baseline_probe_seed": baseline_metadata["probe_seed"],
        "train_checksum": train["checksum"],
        "opponent_hand_supervised": bool(train["opponent_hand_supervised"]),
    }


def _bootstrap_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# No-Range Locked-Layer Bootstrap CI",
        "",
        "## Rule",
        "",
        "- Primary bootstrap rows resample hand clusters using paired `top10_correct` deltas.",
        "- Rows fail construction unless residual and nuisance correctness artifacts align by `point_id`.",
        "",
        "## CI Rows",
        "",
        "| seed | layer | hands | points | residual top10 | baseline top10 | cluster mean delta | cluster lower | cluster upper |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["ci_rows"]:
        lines.append(
            "| {seed} | {layer} | {hand_count} | {sample_count} | {residual_top10:.6f} | {baseline_top10:.6f} | "
            "{mean_delta:.6f} | {lower_delta:.6f} | {upper_delta:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| layer | mean delta | min lower | all lower above zero |",
            "|---|---:|---:|---|",
            (
                "| sequence_hidden | {sequence_hidden_mean_delta:.6f} | "
                "{sequence_hidden_min_lower_delta:.6f} | {sequence_hidden_all_lower_above_zero} |"
            ).format(**payload),
            (
                "| encoder_layer_2 | {encoder_layer_2_mean_delta:.6f} | "
                "{encoder_layer_2_min_lower_delta:.6f} | {encoder_layer_2_all_lower_above_zero} |"
            ).format(**payload),
            "",
        ]
    )
    return "\n".join(lines)


def build_no_range_bootstrap_pack(
    archive_root: Path,
    output_dir: Path,
    commit: str,
    *,
    bootstrap_count: int = 1000,
    bootstrap_seed: int = 20260602,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Build paired bootstrap CIs for the fresh no-range locked-layer rows."""

    ci_rows = []
    for seed_index, seed in enumerate(("seed_k", "seed_l", "seed_m")):
        for layer_index, layer in enumerate(("sequence_hidden", "encoder_layer_2")):
            ci_rows.append(
                _bootstrap_layer_row(
                    archive_root,
                    commit,
                    seed,
                    layer,
                    bootstrap_count=bootstrap_count,
                    bootstrap_seed=bootstrap_seed + seed_index * 10 + layer_index,
                    confidence_level=confidence_level,
                )
            )

    sequence_rows = [row for row in ci_rows if row["layer"] == "sequence_hidden"]
    encoder2_rows = [row for row in ci_rows if row["layer"] == "encoder_layer_2"]
    payload = {
                "commit": commit,
        "version": "1.0.10",
        "source_version": "1.0.9",
        "size": "50k",
        "paired_metric": "residual_top10_correct_minus_action_value_only_nuisance_top10_correct",
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "ci_rows": ci_rows,
        "sequence_hidden_mean_delta": _mean([float(row["mean_delta"]) for row in sequence_rows]),
        "sequence_hidden_min_lower_delta": min(float(row["lower_delta"]) for row in sequence_rows),
        "sequence_hidden_all_lower_above_zero": all(float(row["lower_delta"]) > 0.0 for row in sequence_rows),
        "encoder_layer_2_mean_delta": _mean([float(row["mean_delta"]) for row in encoder2_rows]),
        "encoder_layer_2_min_lower_delta": min(float(row["lower_delta"]) for row in encoder2_rows),
        "encoder_layer_2_all_lower_above_zero": all(float(row["lower_delta"]) > 0.0 for row in encoder2_rows),
        "recommendation": "Use this pack as the test-set sampling uncertainty gate for the locked no-range layer.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "limit_ar_no_range_bootstrap_pack.json", payload)
    (output_dir / "limit_ar_no_range_bootstrap_pack.md").write_text(
        _bootstrap_markdown(payload),
        encoding="utf-8",
    )
    return payload


def _score_for_layer_or_nuisance(path: Path, layer: str) -> float:
    scores = _real_scores_by_layer(path)
    if layer in scores:
        return scores[layer]
    if layer == "nuisance" and "nuisance" in scores:
        return scores["nuisance"]
    raise ValueError(f"metrics file is missing layer {layer}: {path}")


def _fresh50_probe_path(archive_root: Path, commit: str, seed: str, suffix: str) -> Path:
    return _no_range_path_for_size(archive_root, commit, seed, suffix, "50k")


def _fresh50_repeat_rows(archive_root: Path, commit: str) -> list[dict[str, object]]:
    rows = []
    for seed in ("seed_k", "seed_l", "seed_m"):
        for repeat_index in (1, 2, 3):
            residual_suffix = f"history_only_residualize_action_value_repeat_{repeat_index}"
            baseline_suffix = f"history_only_action_value_only_repeat_{repeat_index}"
            residual_path = _fresh50_probe_path(archive_root, commit, seed, residual_suffix)
            baseline_path = _fresh50_probe_path(archive_root, commit, seed, baseline_suffix)
            residual_metadata = _probe_metadata(residual_path, seed, residual_suffix, strict=True)
            baseline_metadata = _probe_metadata(baseline_path, seed, baseline_suffix, strict=True)
            _validate_no_range_pair(residual_metadata, baseline_metadata)
            baseline_score = _nuisance_score(baseline_path)
            for layer in ("sequence_hidden", "encoder_layer_2"):
                residual_score = _score_for_layer_or_nuisance(residual_path, layer)
                rows.append(
                    {
                        "seed": seed,
                        "repeat_index": repeat_index,
                        "layer": layer,
                        "residual_top10": residual_score,
                        "baseline_top10": baseline_score,
                        "delta": residual_score - baseline_score,
                        "residual_probe_seed": residual_metadata["probe_seed"],
                        "baseline_probe_seed": baseline_metadata["probe_seed"],
                        "residual_checksum": residual_metadata["checksum"],
                        "baseline_checksum": baseline_metadata["checksum"],
                    }
                )
    return rows


def _fresh50_repeat_ci_rows(
    archive_root: Path,
    commit: str,
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> list[dict[str, object]]:
    rows = []
    layers = ("sequence_hidden", "encoder_layer_2")
    for seed_index, seed in enumerate(("seed_k", "seed_l", "seed_m")):
        for repeat_index in (1, 2, 3):
            residual_suffix = f"history_only_residualize_action_value_repeat_{repeat_index}"
            baseline_suffix = f"history_only_action_value_only_repeat_{repeat_index}"
            residual_path = _fresh50_probe_path(archive_root, commit, seed, residual_suffix)
            baseline_path = _fresh50_probe_path(archive_root, commit, seed, baseline_suffix)
            residual_metadata = _probe_metadata(residual_path, seed, residual_suffix, strict=True)
            baseline_metadata = _probe_metadata(baseline_path, seed, baseline_suffix, strict=True)
            _validate_no_range_pair(residual_metadata, baseline_metadata)
            for layer_index, layer in enumerate(layers):
                paired = _paired_correctness_deltas(residual_path, baseline_path, layer)
                cluster_interval = _bootstrap_mean_ci(
                    list(paired["cluster_means"]),
                    bootstrap_count=bootstrap_count,
                    bootstrap_seed=bootstrap_seed + seed_index * 100 + repeat_index * 10 + layer_index,
                    confidence_level=confidence_level,
                )
                decision_interval = _bootstrap_mean_ci(
                    list(paired["deltas"]),
                    bootstrap_count=bootstrap_count,
                    bootstrap_seed=bootstrap_seed + 100000 + seed_index * 100 + repeat_index * 10 + layer_index,
                    confidence_level=confidence_level,
                )
                rows.append(
                    {
                        "seed": seed,
                        "repeat_index": repeat_index,
                        "layer": layer,
                        "resampling_unit": "hand_id",
                        "sample_count": len(paired["deltas"]),
                        "hand_count": paired["hand_count"],
                        "bootstrap_count": bootstrap_count,
                        "bootstrap_seed": bootstrap_seed,
                        "confidence_level": confidence_level,
                        "residual_top10": paired["residual_top10"],
                        "baseline_top10": paired["baseline_top10"],
                        **cluster_interval,
                        "decision_point_mean_delta": decision_interval["mean_delta"],
                        "decision_point_lower_delta": decision_interval["lower_delta"],
                        "decision_point_upper_delta": decision_interval["upper_delta"],
                        "paired_outcomes": paired["paired_outcomes"],
                        "residual_checksum": residual_metadata["checksum"],
                        "baseline_checksum": baseline_metadata["checksum"],
                        "residual_correctness_sha256": paired["residual_correctness_sha256"],
                        "baseline_correctness_sha256": paired["baseline_correctness_sha256"],
                    }
                )
    return rows


def _fresh50_label_rows(archive_root: Path, commit: str) -> list[dict[str, object]]:
    rows = []
    for seed in ("seed_k", "seed_l", "seed_m"):
        real_path = _fresh50_probe_path(archive_root, commit, seed, "history_only_residualize_action_value")
        real_metadata = _probe_metadata(real_path, seed, "history_only_residualize_action_value", strict=True)
        real_score = _score_for_layer_or_nuisance(real_path, "encoder_layer_2")
        for label_control, suffix in (
            ("shuffle_global", "history_only_residualize_action_value_shuffle_global_labels"),
            ("shuffle_within_bucket", "history_only_residualize_action_value_shuffle_within_bucket_labels"),
        ):
            control_path = _fresh50_probe_path(archive_root, commit, seed, suffix)
            control_metadata = _probe_metadata(control_path, seed, suffix, strict=True)
            if control_metadata["seed"] != real_metadata["seed"]:
                raise ValueError("label control seed mismatch")
            control_score = _score_for_layer_or_nuisance(control_path, "encoder_layer_2")
            rows.append(
                {
                    "seed": seed,
                    "label_control": label_control,
                    "layer": "encoder_layer_2",
                    "real_top10": real_score,
                    "control_top10": control_score,
                    "delta_real_minus_control": real_score - control_score,
                    "real_checksum": real_metadata["checksum"],
                    "control_checksum": control_metadata["checksum"],
                }
            )
    return rows


def _fresh50_destroyed_history_controls() -> tuple[tuple[str, str, str], ...]:
    return (
        ("drop_history", "drop_history_residualize_action_value", "drop_history_action_value_only"),
        (
            "shuffle_history_order",
            "shuffle_history_order_residualize_action_value",
            "shuffle_history_order_action_value_only",
        ),
        ("bag_of_actions", "bag_of_actions_residualize_action_value", "bag_of_actions_action_value_only"),
        ("no_observation", "no_observation_residualize_action_value", "no_observation_action_value_only"),
    )


def _history_composition_controls() -> tuple[tuple[str, str, str], ...]:
    controls = (
        "shuffle_history_order_seed_1",
        "shuffle_history_order_seed_2",
        "shuffle_history_order_seed_3",
        "reverse_history_order",
        "sort_history_actions",
        "history_presence_only",
        "history_last_action_only",
    )
    return tuple(
        (
            control,
            f"{control}_residualize_action_value",
            f"{control}_action_value_only",
        )
        for control in controls
    )


def _fresh50_history_rows(archive_root: Path, commit: str) -> list[dict[str, object]]:
    rows = []
    controls = (
        ("drop_history", "drop_history", "none", "encoder_layer_2"),
        ("shuffle_history_order", "shuffle_history_order", "none", "encoder_layer_2"),
        ("bag_of_actions", "bag_of_actions", "none", "encoder_layer_2"),
        ("no_observation", "no_observation", "none", "encoder_layer_2"),
        ("round_position_action_prior", "none", "round_position_action_only", "nuisance"),
        ("history_only_action_only", "history_only", "action_only", "nuisance"),
        ("drop_history_residualize_action_value", "drop_history", "residualize_action_value", "encoder_layer_2"),
        (
            "shuffle_history_order_residualize_action_value",
            "shuffle_history_order",
            "residualize_action_value",
            "encoder_layer_2",
        ),
        ("bag_of_actions_residualize_action_value", "bag_of_actions", "residualize_action_value", "encoder_layer_2"),
        ("no_observation_residualize_action_value", "no_observation", "residualize_action_value", "encoder_layer_2"),
    )
    for seed in ("seed_k", "seed_l", "seed_m"):
        ordered_path = _fresh50_probe_path(archive_root, commit, seed, "history_only_residualize_action_value")
        ordered_metadata = _probe_metadata(ordered_path, seed, "history_only_residualize_action_value", strict=True)
        ordered_score = _score_for_layer_or_nuisance(ordered_path, "encoder_layer_2")
        for suffix, input_control, feature_control, layer in controls:
            control_path = _fresh50_probe_path(archive_root, commit, seed, suffix)
            control_metadata = _probe_metadata(control_path, seed, suffix, strict=True)
            control_score = _score_for_layer_or_nuisance(control_path, layer)
            rows.append(
                {
                    "seed": seed,
                    "control": suffix,
                    "input_control": input_control,
                    "feature_control": feature_control,
                    "layer": layer,
                    "ordered_encoder_layer_2_top10": ordered_score,
                    "control_top10": control_score,
                    "delta_ordered_minus_control": ordered_score - control_score,
                    "ordered_checksum": ordered_metadata["checksum"],
                    "control_checksum": control_metadata["checksum"],
                }
            )
        ordered_baseline_path = _fresh50_probe_path(archive_root, commit, seed, "history_only_action_value_only")
        ordered_baseline_metadata = _probe_metadata(
            ordered_baseline_path,
            seed,
            "history_only_action_value_only",
            strict=True,
        )
        _validate_no_range_pair(ordered_metadata, ordered_baseline_metadata)
        ordered_baseline_score = _nuisance_score(ordered_baseline_path)
        ordered_delta = ordered_score - ordered_baseline_score
        for control_name, residual_suffix, baseline_suffix in _fresh50_destroyed_history_controls():
            residual_path = _fresh50_probe_path(archive_root, commit, seed, residual_suffix)
            baseline_path = _fresh50_probe_path(archive_root, commit, seed, baseline_suffix)
            residual_metadata = _probe_metadata(residual_path, seed, residual_suffix, strict=True)
            baseline_metadata = _probe_metadata(baseline_path, seed, baseline_suffix, strict=True)
            _validate_no_range_pair(residual_metadata, baseline_metadata, require_same_version=False)
            residual_score = _score_for_layer_or_nuisance(residual_path, "encoder_layer_2")
            baseline_score = _nuisance_score(baseline_path)
            residual_delta = residual_score - baseline_score
            rows.append(
                {
                    "seed": seed,
                    "control": f"{control_name}_matched_delta",
                    "input_control": residual_metadata["input_control"],
                    "feature_control": "matched_residual_vs_action_value_only",
                    "layer": "encoder_layer_2",
                    "ordered_encoder_layer_2_top10": ordered_score,
                    "control_top10": residual_score,
                    "delta_ordered_minus_control": ordered_delta - residual_delta,
                    "ordered_residual_top10": ordered_score,
                    "ordered_baseline_top10": ordered_baseline_score,
                    "ordered_residual_delta": ordered_delta,
                    "control_residual_top10": residual_score,
                    "control_baseline_top10": baseline_score,
                    "control_residual_delta": residual_delta,
                    "ordered_checksum": ordered_metadata["checksum"],
                    "ordered_baseline_checksum": ordered_baseline_metadata["checksum"],
                    "control_checksum": residual_metadata["checksum"],
                    "control_baseline_checksum": baseline_metadata["checksum"],
                }
            )
    return rows


def _history_composition_ci_rows(
    archive_root: Path,
    commit: str,
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> list[dict[str, object]]:
    rows = []
    ordered_residual_suffix = "history_only_residualize_action_value"
    ordered_baseline_suffix = "history_only_action_value_only"
    for seed_index, seed in enumerate(("seed_k", "seed_l", "seed_m")):
        ordered_residual_path = _fresh50_probe_path(archive_root, commit, seed, ordered_residual_suffix)
        ordered_baseline_path = _fresh50_probe_path(archive_root, commit, seed, ordered_baseline_suffix)
        ordered_residual_metadata = _probe_metadata(ordered_residual_path, seed, ordered_residual_suffix, strict=True)
        ordered_baseline_metadata = _probe_metadata(ordered_baseline_path, seed, ordered_baseline_suffix, strict=True)
        _validate_no_range_pair(ordered_residual_metadata, ordered_baseline_metadata)
        ordered_paired = _paired_correctness_deltas(ordered_residual_path, ordered_baseline_path, "encoder_layer_2")
        for control_index, (control_name, residual_suffix, baseline_suffix) in enumerate(
            _history_composition_controls()
        ):
            residual_path = _fresh50_probe_path(archive_root, commit, seed, residual_suffix)
            baseline_path = _fresh50_probe_path(archive_root, commit, seed, baseline_suffix)
            residual_metadata = _probe_metadata(residual_path, seed, residual_suffix, strict=True)
            baseline_metadata = _probe_metadata(baseline_path, seed, baseline_suffix, strict=True)
            _validate_no_range_pair(residual_metadata, baseline_metadata)
            if residual_metadata["input_control"] != control_name:
                raise ValueError(f"history composition input_control mismatch for {seed}/{control_name}")
            control_paired = _paired_correctness_deltas(residual_path, baseline_path, "encoder_layer_2")

            ordered_hands = dict(ordered_paired["cluster_delta_by_hand"])
            control_hands = dict(control_paired["cluster_delta_by_hand"])
            if sorted(ordered_hands) != sorted(control_hands):
                raise ValueError(f"history composition hand_id mismatch for {seed}/{control_name}")
            cluster_diffs = [
                float(ordered_hands[hand_id]) - float(control_hands[hand_id])
                for hand_id in sorted(ordered_hands)
            ]

            ordered_points = dict(ordered_paired["point_deltas"])
            control_points = dict(control_paired["point_deltas"])
            if sorted(ordered_points) != sorted(control_points):
                raise ValueError(f"history composition point_id mismatch for {seed}/{control_name}")
            point_diffs = [
                float(ordered_points[point_id]["delta"]) - float(control_points[point_id]["delta"])
                for point_id in sorted(ordered_points)
            ]
            cluster_interval = _bootstrap_mean_ci(
                cluster_diffs,
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed + 400000 + seed_index * 100 + control_index,
                confidence_level=confidence_level,
            )
            decision_interval = _bootstrap_mean_ci(
                point_diffs,
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed + 500000 + seed_index * 100 + control_index,
                confidence_level=confidence_level,
            )
            rows.append(
                {
                    "seed": seed,
                    "control": control_name,
                    "layer": "encoder_layer_2",
                    "resampling_unit": "hand_id",
                    "sample_count": len(point_diffs),
                    "hand_count": len(cluster_diffs),
                    "bootstrap_count": bootstrap_count,
                    "bootstrap_seed": bootstrap_seed,
                    "confidence_level": confidence_level,
                    "ordered_residual_top10": ordered_paired["residual_top10"],
                    "ordered_baseline_top10": ordered_paired["baseline_top10"],
                    "control_residual_top10": control_paired["residual_top10"],
                    "control_baseline_top10": control_paired["baseline_top10"],
                    "ordered_mean_delta": _mean(list(ordered_paired["cluster_means"])),
                    "control_mean_delta": _mean(list(control_paired["cluster_means"])),
                    **cluster_interval,
                    "decision_point_mean_delta": decision_interval["mean_delta"],
                    "decision_point_lower_delta": decision_interval["lower_delta"],
                    "decision_point_upper_delta": decision_interval["upper_delta"],
                    "ordered_residual_checksum": ordered_residual_metadata["checksum"],
                    "ordered_baseline_checksum": ordered_baseline_metadata["checksum"],
                    "control_residual_checksum": residual_metadata["checksum"],
                    "control_baseline_checksum": baseline_metadata["checksum"],
                    "ordered_residual_correctness_sha256": ordered_paired["residual_correctness_sha256"],
                    "ordered_baseline_correctness_sha256": ordered_paired["baseline_correctness_sha256"],
                    "control_residual_correctness_sha256": control_paired["residual_correctness_sha256"],
                    "control_baseline_correctness_sha256": control_paired["baseline_correctness_sha256"],
                }
            )
    return rows


def _history_composition_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# History Composition Counterfactual Pack",
        "",
        "## Ordered-minus-Control Deltas",
        "",
        "| seed | control | hands | ordered delta | control delta | diff mean | lower | upper |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {seed} | {control} | {hand_count} | {ordered_mean_delta:.6f} | "
            "{control_mean_delta:.6f} | {mean_delta:.6f} | {lower_delta:.6f} | "
            "{upper_delta:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| metric | value |",
            "|---|---:|",
            "| min CI lower | {min_lower_delta:.6f} |".format(**payload),
            "| all CI lower above zero | {all_lower_above_zero} |".format(**payload),
            "| shuffle mean CI lower | {shuffle_mean_lower_delta:.6f} |".format(**payload),
            "| count-preserving min CI lower | {count_preserving_min_lower_delta:.6f} |".format(**payload),
            "",
        ]
    )
    return "\n".join(lines)


def build_no_range_history_composition_pack(
    archive_root: Path,
    output_dir: Path,
    commit: str,
    *,
    bootstrap_count: int = 1000,
    bootstrap_seed: int = 20260602,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Build the v1.0.15 history-composition counterfactual pack."""

    rows = _history_composition_ci_rows(
        archive_root,
        commit,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    shuffle_rows = [row for row in rows if str(row["control"]).startswith("shuffle_history_order_seed_")]
    count_preserving_controls = {
        "shuffle_history_order_seed_1",
        "shuffle_history_order_seed_2",
        "shuffle_history_order_seed_3",
        "reverse_history_order",
        "sort_history_actions",
    }
    count_preserving_rows = [row for row in rows if str(row["control"]) in count_preserving_controls]
    payload = {
                "version": "1.0.15",
        "commit": commit,
        "locked_layer": "encoder_layer_2",
        "tokenizer_scope": "action_identity_and_indexed_action_position_only",
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "rows": rows,
        "control_count": len(_history_composition_controls()),
        "seed_count": 3,
        "min_lower_delta": min(float(row["lower_delta"]) for row in rows),
        "all_lower_above_zero": all(float(row["lower_delta"]) > 0.0 for row in rows),
        "shuffle_mean_delta": _mean([float(row["mean_delta"]) for row in shuffle_rows]),
        "shuffle_mean_lower_delta": _mean([float(row["lower_delta"]) for row in shuffle_rows]),
        "count_preserving_min_lower_delta": min(float(row["lower_delta"]) for row in count_preserving_rows),
        "presence_last_action_min_lower_delta": min(
            float(row["lower_delta"])
            for row in rows
            if row["control"] in {"history_presence_only", "history_last_action_only"}
        ),
        "recommendation": (
            "Use ordered-history wording only if count-preserving controls have aggregate lower deltas above zero."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "limit_ar_no_range_history_composition_pack.json", payload)
    (output_dir / "limit_ar_no_range_history_composition_pack.md").write_text(
        _history_composition_markdown(payload),
        encoding="utf-8",
    )
    return payload


def _composition_nuisance_ci_rows(
    archive_root: Path,
    commit: str,
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> list[dict[str, object]]:
    rows = []
    ordered_residual_suffix = "history_only_residualize_action_value"
    ordered_baseline_suffix = "history_only_action_value_only"
    composition_suffix = "history_composition_only"
    action_value_composition_suffix = "action_value_history_composition_only"
    residual_composition_suffix = "residualize_action_value_history_composition"
    for seed_index, seed in enumerate(("seed_k", "seed_l", "seed_m")):
        ordered_residual_path = _fresh50_probe_path(archive_root, commit, seed, ordered_residual_suffix)
        ordered_baseline_path = _fresh50_probe_path(archive_root, commit, seed, ordered_baseline_suffix)
        composition_path = _fresh50_probe_path(archive_root, commit, seed, composition_suffix)
        action_value_composition_path = _fresh50_probe_path(archive_root, commit, seed, action_value_composition_suffix)
        residual_composition_path = _fresh50_probe_path(archive_root, commit, seed, residual_composition_suffix)

        ordered_residual_metadata = _probe_metadata(ordered_residual_path, seed, ordered_residual_suffix, strict=True)
        ordered_baseline_metadata = _probe_metadata(ordered_baseline_path, seed, ordered_baseline_suffix, strict=True)
        composition_metadata = _probe_metadata(composition_path, seed, composition_suffix, strict=True)
        action_value_composition_metadata = _probe_metadata(
            action_value_composition_path,
            seed,
            action_value_composition_suffix,
            strict=True,
        )
        residual_composition_metadata = _probe_metadata(
            residual_composition_path,
            seed,
            residual_composition_suffix,
            strict=True,
        )
        _validate_no_range_pair(ordered_residual_metadata, ordered_baseline_metadata)
        if composition_metadata["feature_control"] != "history_composition_only":
            raise ValueError(f"composition-only feature_control mismatch for {seed}")
        if action_value_composition_metadata["feature_control"] != "action_value_history_composition_only":
            raise ValueError(f"action/value+composition feature_control mismatch for {seed}")
        if residual_composition_metadata["feature_control"] != "residualize_action_value_history_composition":
            raise ValueError(f"composition residual feature_control mismatch for {seed}")

        ordered_vs_action_value = _paired_correctness_deltas(
            ordered_residual_path,
            ordered_baseline_path,
            "encoder_layer_2",
        )
        residual_vs_action_value = _paired_correctness_deltas(
            residual_composition_path,
            ordered_baseline_path,
            "encoder_layer_2",
        )
        residual_vs_action_value_composition = _paired_correctness_deltas_for_layers(
            residual_composition_path,
            "encoder_layer_2",
            action_value_composition_path,
            "nuisance",
        )
        composition_vs_action_value = _paired_correctness_deltas_for_layers(
            composition_path,
            "nuisance",
            ordered_baseline_path,
            "nuisance",
        )
        action_value_composition_vs_action_value = _paired_correctness_deltas_for_layers(
            action_value_composition_path,
            "nuisance",
            ordered_baseline_path,
            "nuisance",
        )

        for row_index, (comparison, paired) in enumerate(
            (
                ("ordered_residual_vs_action_value", ordered_vs_action_value),
                ("composition_only_vs_action_value", composition_vs_action_value),
                ("action_value_composition_only_vs_action_value", action_value_composition_vs_action_value),
                ("composition_residual_vs_action_value", residual_vs_action_value),
                ("composition_residual_vs_action_value_composition", residual_vs_action_value_composition),
            )
        ):
            cluster_interval = _bootstrap_mean_ci(
                list(paired["cluster_means"]),
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed + 600000 + seed_index * 100 + row_index,
                confidence_level=confidence_level,
            )
            decision_interval = _bootstrap_mean_ci(
                list(paired["deltas"]),
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed + 700000 + seed_index * 100 + row_index,
                confidence_level=confidence_level,
            )
            rows.append(
                {
                    "seed": seed,
                    "comparison": comparison,
                    "layer": "encoder_layer_2" if "residual" in comparison or comparison.startswith("ordered") else "nuisance",
                    "resampling_unit": "hand_id",
                    "sample_count": len(paired["deltas"]),
                    "hand_count": paired["hand_count"],
                    "bootstrap_count": bootstrap_count,
                    "bootstrap_seed": bootstrap_seed,
                    "confidence_level": confidence_level,
                    "real_top10": paired["residual_top10"],
                    "baseline_top10": paired["baseline_top10"],
                    **cluster_interval,
                    "decision_point_mean_delta": decision_interval["mean_delta"],
                    "decision_point_lower_delta": decision_interval["lower_delta"],
                    "decision_point_upper_delta": decision_interval["upper_delta"],
                    "paired_outcomes": paired["paired_outcomes"],
                    "real_correctness_sha256": paired["residual_correctness_sha256"],
                    "baseline_correctness_sha256": paired["baseline_correctness_sha256"],
                    "ordered_residual_checksum": ordered_residual_metadata["checksum"],
                    "ordered_baseline_checksum": ordered_baseline_metadata["checksum"],
                    "composition_checksum": composition_metadata["checksum"],
                    "action_value_composition_checksum": action_value_composition_metadata["checksum"],
                    "residual_composition_checksum": residual_composition_metadata["checksum"],
                }
            )
    return rows


def _range_supervised_paired_row(
    *,
    seed: str,
    gate: str,
    paired: dict[str, object],
    residual_metadata: dict[str, object],
    baseline_metadata: dict[str, object],
    train: dict[str, object],
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
    row_index: int,
) -> dict[str, object]:
    cluster_interval = _bootstrap_mean_ci(
        list(paired["cluster_means"]),
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed + row_index,
        confidence_level=confidence_level,
    )
    decision_interval = _bootstrap_mean_ci(
        list(paired["deltas"]),
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed + 100000 + row_index,
        confidence_level=confidence_level,
    )
    return {
        "model_family": "limit_ar_range_supervised_4layer",
        "seed": seed,
        "gate": gate,
        "layer": "encoder_layer_2",
        "resampling_unit": "hand_id",
        "sample_count": len(paired["deltas"]),
        "hand_count": paired["hand_count"],
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "real_top10": paired["residual_top10"],
        "baseline_top10": paired["baseline_top10"],
        **cluster_interval,
        "decision_point_mean_delta": decision_interval["mean_delta"],
        "decision_point_lower_delta": decision_interval["lower_delta"],
        "decision_point_upper_delta": decision_interval["upper_delta"],
        "paired_outcomes": paired["paired_outcomes"],
        "residual_checksum": residual_metadata["checksum"],
        "baseline_checksum": baseline_metadata["checksum"],
        "residual_correctness_sha256": paired["residual_correctness_sha256"],
        "baseline_correctness_sha256": paired["baseline_correctness_sha256"],
        "residual_probe_seed": residual_metadata["probe_seed"],
        "baseline_probe_seed": baseline_metadata["probe_seed"],
        "train_checksum": train["checksum"],
        "opponent_hand_supervised": bool(train.get("opponent_hand_supervised", True)),
    }


def _range_supervised_matched_row(
    *,
    commit_root: Path,
    seed: str,
    paired: dict[str, object],
    residual_metadata: dict[str, object],
    baseline_metadata: dict[str, object],
    train: dict[str, object],
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
    row_index: int,
) -> dict[str, object]:
    buckets, bucket_source = _test_point_buckets_for_positive_control(commit_root, seed)
    point_deltas = paired["point_deltas"]
    if not isinstance(point_deltas, dict):
        raise ValueError("paired point_deltas must be a dict")
    common_ids = sorted(set(point_deltas) & set(buckets))
    if not common_ids:
        raise ValueError(f"no shared point_ids for range-supervised matched-composition seed: {seed}")

    bucket_to_deltas: dict[str, list[float]] = {}
    dropped_point_count = 0
    for point_id in common_ids:
        bucket_to_deltas.setdefault(buckets[point_id], []).append(float(point_deltas[point_id]["delta"]))

    matched_deltas = []
    bucket_mean_deltas = []
    singleton_bucket_count = 0
    for deltas in bucket_to_deltas.values():
        if len(deltas) >= 2:
            matched_deltas.extend(deltas)
            bucket_mean_deltas.append(_mean(deltas))
        else:
            dropped_point_count += len(deltas)
            singleton_bucket_count += 1
    if not matched_deltas:
        raise ValueError(f"range-supervised matched-composition buckets have no non-singleton support: {seed}")

    interval = _bootstrap_mean_ci(
        bucket_mean_deltas,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed + 200000 + row_index,
        confidence_level=confidence_level,
    )
    all_count = len(common_ids)
    return {
        "model_family": "limit_ar_range_supervised_4layer",
        "seed": seed,
        "gate": "S7_matched_composition",
        "layer": "encoder_layer_2",
        "resampling_unit": "matched_composition_bucket",
        "sample_count": len(matched_deltas),
        "hand_count": paired["hand_count"],
        "bucket_count": len(bucket_to_deltas),
        "matched_bucket_count": len(bucket_mean_deltas),
        "singleton_bucket_count": singleton_bucket_count,
        "dropped_point_count": dropped_point_count,
        "dropped_fraction": dropped_point_count / all_count,
        "bucket_source": bucket_source,
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "real_top10": float(paired["residual_top10"]),
        "baseline_top10": float(paired["baseline_top10"]),
        **interval,
        "decision_point_mean_delta": _mean(matched_deltas),
        "decision_point_lower_delta": min(matched_deltas),
        "decision_point_upper_delta": max(matched_deltas),
        "paired_outcomes": paired["paired_outcomes"],
        "residual_checksum": residual_metadata["checksum"],
        "baseline_checksum": baseline_metadata["checksum"],
        "residual_correctness_sha256": paired["residual_correctness_sha256"],
        "baseline_correctness_sha256": paired["baseline_correctness_sha256"],
        "residual_probe_seed": residual_metadata["probe_seed"],
        "baseline_probe_seed": baseline_metadata["probe_seed"],
        "train_checksum": train["checksum"],
        "opponent_hand_supervised": bool(train.get("opponent_hand_supervised", True)),
    }


def _range_supervision_status(train_path: Path, train: dict[str, object]) -> str:
    if "opponent_hand_supervised" in train:
        if not bool(train["opponent_hand_supervised"]):
            raise ValueError(f"range-supervised positive control requires opponent_hand_supervised=true: {train_path}")
        if float(train.get("opponent_hand_loss_weight", 1.0)) <= 0.0:
            raise ValueError(f"range-supervised positive control requires opponent_hand_loss_weight>0: {train_path}")
        return "explicit_opponent_hand_supervised"
    if "opponent_hand_loss_weight" in train:
        if float(train["opponent_hand_loss_weight"]) <= 0.0:
            raise ValueError(f"range-supervised positive control requires opponent_hand_loss_weight>0: {train_path}")
        return "legacy_loss_weight_supervised"
    if "limit_ar_teacher_" not in str(train_path):
        raise ValueError(f"legacy range-supervised inference only allowed for teacher train paths: {train_path}")
    return "legacy_teacher_range_supervision_inferred"


def _range_supervised_positive_control_rows(
    archive_root: Path,
    commit: str,
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> list[dict[str, object]]:
    rows = []
    commit_root = archive_root / commit
    for seed_index, seed in enumerate(RANGE_SUPERVISED_POSITIVE_CONTROL_SEEDS):
        train_path = _range_supervised_train_path_for_size(archive_root, commit, seed, "50k")
        train = _read_metrics(train_path)
        supervision_status = _range_supervision_status(train_path, train)

        s5_residual_suffix = "history_only_residualize_action_value"
        s5_baseline_suffix = "history_only_action_value_only"
        s6_residual_suffix = "residualize_action_value_history_composition"
        s6_baseline_suffix = "action_value_history_composition_only"
        s5_residual_path = _range_supervised_path_for_size(archive_root, commit, seed, s5_residual_suffix, "50k")
        s5_baseline_path = _range_supervised_path_for_size(archive_root, commit, seed, s5_baseline_suffix, "50k")
        s6_residual_path = _range_supervised_path_for_size(archive_root, commit, seed, s6_residual_suffix, "50k")
        s6_baseline_path = _range_supervised_path_for_size(archive_root, commit, seed, s6_baseline_suffix, "50k")
        s5_residual_metadata = _range_supervised_probe_metadata(s5_residual_path, seed, s5_residual_suffix, strict=True)
        s5_baseline_metadata = _range_supervised_probe_metadata(s5_baseline_path, seed, s5_baseline_suffix, strict=True)
        s6_residual_metadata = _range_supervised_probe_metadata(s6_residual_path, seed, s6_residual_suffix, strict=True)
        s6_baseline_metadata = _range_supervised_probe_metadata(s6_baseline_path, seed, s6_baseline_suffix, strict=True)
        _validate_range_supervised_pair(
            s5_residual_metadata,
            s5_baseline_metadata,
            residual_feature_control="residualize_action_value",
            baseline_feature_control="action_value_only",
        )
        _validate_range_supervised_pair(
            s6_residual_metadata,
            s6_baseline_metadata,
            residual_feature_control="residualize_action_value_history_composition",
            baseline_feature_control="action_value_history_composition_only",
        )
        s5_paired = _paired_correctness_deltas(s5_residual_path, s5_baseline_path, "encoder_layer_2")
        s6_paired = _paired_correctness_deltas_for_layers(
            s6_residual_path,
            "encoder_layer_2",
            s6_baseline_path,
            "nuisance",
        )
        rows.append(
            _range_supervised_paired_row(
                seed=seed,
                gate="S5_action_value_residual",
                paired=s5_paired,
                residual_metadata=s5_residual_metadata,
                baseline_metadata=s5_baseline_metadata,
                train=train,
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed,
                confidence_level=confidence_level,
                row_index=seed_index * 3,
            )
        )
        rows[-1]["supervision_status"] = supervision_status
        rows.append(
            _range_supervised_paired_row(
                seed=seed,
                gate="S6_composition_residual",
                paired=s6_paired,
                residual_metadata=s6_residual_metadata,
                baseline_metadata=s6_baseline_metadata,
                train=train,
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed,
                confidence_level=confidence_level,
                row_index=seed_index * 3 + 1,
            )
        )
        rows[-1]["supervision_status"] = supervision_status
        rows.append(
            _range_supervised_matched_row(
                commit_root=commit_root,
                seed=seed,
                paired=s6_paired,
                residual_metadata=s6_residual_metadata,
                baseline_metadata=s6_baseline_metadata,
                train=train,
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed,
                confidence_level=confidence_level,
                row_index=seed_index * 3 + 2,
            )
        )
        rows[-1]["supervision_status"] = supervision_status
    return rows


def _range_supervised_positive_control_label(rows: list[dict[str, object]]) -> str:
    lowers_by_gate: dict[str, list[float]] = {}
    uppers_by_gate: dict[str, list[float]] = {}
    for row in rows:
        lowers_by_gate.setdefault(str(row["gate"]), []).append(float(row["lower_delta"]))
        uppers_by_gate.setdefault(str(row["gate"]), []).append(float(row["upper_delta"]))
    if all(min(lowers_by_gate.get(gate, [0.0])) > 0.0 for gate in ("S5_action_value_residual", "S6_composition_residual", "S7_matched_composition")):
        return "belief_like_real_poker_positive_control"
    if (
        max(uppers_by_gate.get("S6_composition_residual", [1.0])) < 0.0
        or max(uppers_by_gate.get("S7_matched_composition", [1.0])) < 0.0
    ):
        return "composition_shortcut_supported"
    return "mixed_or_inconclusive_positive_control"


def _range_supervised_positive_control_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Range-Supervised Positive-Control Audit Pack",
        "",
        "## Summary",
        "",
        f"- final label: `{payload['final_diagnostic_label']}`",
        "- model family: range-supervised 50k 4-layer Limit AR",
        "- main layer: locked `encoder_layer_2`",
        "",
        "## Rows",
        "",
        "| seed | gate | unit | real top10 | baseline top10 | mean delta | lower | upper |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {seed} | {gate} | {resampling_unit} | {real_top10:.6f} | {baseline_top10:.6f} | "
            "{mean_delta:.6f} | {lower_delta:.6f} | {upper_delta:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Gate Minimums",
            "",
            "| gate | min lower |",
            "|---|---:|",
            f"| S5 action/value residual | {payload['s5_min_lower_delta']:.6f} |",
            f"| S6 composition residual | {payload['s6_min_lower_delta']:.6f} |",
            f"| S7 matched_composition | {payload['s7_min_lower_delta']:.6f} |",
            "",
        ]
    )
    return "\n".join(lines)


def build_range_supervised_positive_control_pack(
    archive_root: Path,
    output_dir: Path,
    commit: str,
    *,
    bootstrap_count: int = 1000,
    bootstrap_seed: int = 20260603,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Build the v1.17 range-supervised real-poker positive-control audit pack."""

    rows = _range_supervised_positive_control_rows(
        archive_root,
        commit,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )

    def min_lower(gate: str) -> float:
        return min(float(row["lower_delta"]) for row in rows if row["gate"] == gate)

    payload = {
                "version": "1.17",
        "commit": commit,
        "model_family": "limit_ar_range_supervised_4layer",
        "locked_layer": "encoder_layer_2",
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "rows": rows,
        "row_count": len(rows),
        "seed_count": len(RANGE_SUPERVISED_POSITIVE_CONTROL_SEEDS),
        "s5_min_lower_delta": min_lower("S5_action_value_residual"),
        "s6_min_lower_delta": min_lower("S6_composition_residual"),
        "s7_min_lower_delta": min_lower("S7_matched_composition"),
    }
    payload["final_diagnostic_label"] = _range_supervised_positive_control_label(rows)
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "limit_ar_range_supervised_positive_control_pack.json", payload)
    (output_dir / "limit_ar_range_supervised_positive_control_pack.md").write_text(
        _range_supervised_positive_control_markdown(payload),
        encoding="utf-8",
    )
    return payload


def _composition_nuisance_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# No-Range Composition Nuisance Pack",
        "",
        "## Paired Deltas",
        "",
        "| seed | comparison | layer | real top10 | baseline top10 | mean delta | lower | upper |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {seed} | {comparison} | {layer} | {real_top10:.6f} | {baseline_top10:.6f} | "
            "{mean_delta:.6f} | {lower_delta:.6f} | {upper_delta:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| metric | value |",
            "|---|---:|",
            "| ordered residual min lower | {ordered_residual_min_lower_delta:.6f} |".format(**payload),
            "| composition-only min lower | {composition_only_min_lower_delta:.6f} |".format(**payload),
            "| action/value+composition min lower | {action_value_composition_min_lower_delta:.6f} |".format(**payload),
            "| composition-residual vs action/value min lower | {composition_residual_vs_action_value_min_lower_delta:.6f} |".format(**payload),
            "| composition-residual vs action/value+composition min lower | {composition_residual_vs_action_value_composition_min_lower_delta:.6f} |".format(**payload),
            "| composition-residual all lower above zero | {composition_residual_all_lower_above_zero} |".format(**payload),
            "",
        ]
    )
    return "\n".join(lines)


def build_no_range_composition_nuisance_pack(
    archive_root: Path,
    output_dir: Path,
    commit: str,
    *,
    bootstrap_count: int = 1000,
    bootstrap_seed: int = 20260602,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Build the v1.0.16 explicit history-composition nuisance-control pack."""

    rows = _composition_nuisance_ci_rows(
        archive_root,
        commit,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )

    def lower_values(comparison: str) -> list[float]:
        return [float(row["lower_delta"]) for row in rows if row["comparison"] == comparison]

    composition_residual_lowers = lower_values("composition_residual_vs_action_value_composition")
    payload = {
                "version": "1.0.16",
        "commit": commit,
        "locked_layer": "encoder_layer_2",
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "rows": rows,
        "seed_count": 3,
        "comparison_count": 5,
        "ordered_residual_min_lower_delta": min(lower_values("ordered_residual_vs_action_value")),
        "composition_only_min_lower_delta": min(lower_values("composition_only_vs_action_value")),
        "action_value_composition_min_lower_delta": min(
            lower_values("action_value_composition_only_vs_action_value")
        ),
        "composition_residual_vs_action_value_min_lower_delta": min(
            lower_values("composition_residual_vs_action_value")
        ),
        "composition_residual_vs_action_value_composition_min_lower_delta": min(composition_residual_lowers),
        "composition_residual_all_lower_above_zero": all(value > 0.0 for value in composition_residual_lowers),
        "recommendation": (
            "Claim hidden range signal beyond explicit composition only if the composition-residual vs "
            "action/value+composition lower deltas are above zero."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "limit_ar_no_range_composition_nuisance_pack.json", payload)
    (output_dir / "limit_ar_no_range_composition_nuisance_pack.md").write_text(
        _composition_nuisance_markdown(payload),
        encoding="utf-8",
    )
    return payload


def _fresh50_history_delta_ci_rows(
    archive_root: Path,
    commit: str,
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> list[dict[str, object]]:
    rows = []
    ordered_residual_suffix = "history_only_residualize_action_value"
    ordered_baseline_suffix = "history_only_action_value_only"
    for seed_index, seed in enumerate(("seed_k", "seed_l", "seed_m")):
        ordered_residual_path = _fresh50_probe_path(archive_root, commit, seed, ordered_residual_suffix)
        ordered_baseline_path = _fresh50_probe_path(archive_root, commit, seed, ordered_baseline_suffix)
        ordered_residual_metadata = _probe_metadata(ordered_residual_path, seed, ordered_residual_suffix, strict=True)
        ordered_baseline_metadata = _probe_metadata(ordered_baseline_path, seed, ordered_baseline_suffix, strict=True)
        _validate_no_range_pair(ordered_residual_metadata, ordered_baseline_metadata)
        ordered_paired = _paired_correctness_deltas(ordered_residual_path, ordered_baseline_path, "encoder_layer_2")
        for control_index, (control_name, residual_suffix, baseline_suffix) in enumerate(
            _fresh50_destroyed_history_controls()
        ):
            residual_path = _fresh50_probe_path(archive_root, commit, seed, residual_suffix)
            baseline_path = _fresh50_probe_path(archive_root, commit, seed, baseline_suffix)
            residual_metadata = _probe_metadata(residual_path, seed, residual_suffix, strict=True)
            baseline_metadata = _probe_metadata(baseline_path, seed, baseline_suffix, strict=True)
            _validate_no_range_pair(residual_metadata, baseline_metadata, require_same_version=False)
            control_paired = _paired_correctness_deltas(residual_path, baseline_path, "encoder_layer_2")

            ordered_hands = dict(ordered_paired["cluster_delta_by_hand"])
            control_hands = dict(control_paired["cluster_delta_by_hand"])
            if sorted(ordered_hands) != sorted(control_hands):
                raise ValueError(f"history control hand_id mismatch for {seed}/{control_name}")
            cluster_diffs = [float(ordered_hands[hand_id]) - float(control_hands[hand_id]) for hand_id in sorted(ordered_hands)]

            ordered_points = dict(ordered_paired["point_deltas"])
            control_points = dict(control_paired["point_deltas"])
            if sorted(ordered_points) != sorted(control_points):
                raise ValueError(f"history control point_id mismatch for {seed}/{control_name}")
            point_diffs = [
                float(ordered_points[point_id]["delta"]) - float(control_points[point_id]["delta"])
                for point_id in sorted(ordered_points)
            ]
            cluster_interval = _bootstrap_mean_ci(
                cluster_diffs,
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed + 200000 + seed_index * 100 + control_index,
                confidence_level=confidence_level,
            )
            decision_interval = _bootstrap_mean_ci(
                point_diffs,
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed + 300000 + seed_index * 100 + control_index,
                confidence_level=confidence_level,
            )
            rows.append(
                {
                    "seed": seed,
                    "control": control_name,
                    "layer": "encoder_layer_2",
                    "resampling_unit": "hand_id",
                    "sample_count": len(point_diffs),
                    "hand_count": len(cluster_diffs),
                    "bootstrap_count": bootstrap_count,
                    "bootstrap_seed": bootstrap_seed,
                    "confidence_level": confidence_level,
                    "ordered_residual_top10": ordered_paired["residual_top10"],
                    "ordered_baseline_top10": ordered_paired["baseline_top10"],
                    "control_residual_top10": control_paired["residual_top10"],
                    "control_baseline_top10": control_paired["baseline_top10"],
                    "ordered_mean_delta": _mean(list(ordered_paired["cluster_means"])),
                    "control_mean_delta": _mean(list(control_paired["cluster_means"])),
                    **cluster_interval,
                    "decision_point_mean_delta": decision_interval["mean_delta"],
                    "decision_point_lower_delta": decision_interval["lower_delta"],
                    "decision_point_upper_delta": decision_interval["upper_delta"],
                    "ordered_residual_checksum": ordered_residual_metadata["checksum"],
                    "ordered_baseline_checksum": ordered_baseline_metadata["checksum"],
                    "control_residual_checksum": residual_metadata["checksum"],
                    "control_baseline_checksum": baseline_metadata["checksum"],
                    "ordered_residual_correctness_sha256": ordered_paired["residual_correctness_sha256"],
                    "ordered_baseline_correctness_sha256": ordered_paired["baseline_correctness_sha256"],
                    "control_residual_correctness_sha256": control_paired["residual_correctness_sha256"],
                    "control_baseline_correctness_sha256": control_paired["baseline_correctness_sha256"],
                }
            )
    return rows


def _fresh50_scaling_rows(archive_root: Path, commit_10k: str, commit_50k: str) -> list[dict[str, object]]:
    rows = []
    for size, commit, seeds in (
        ("10k", commit_10k, ("seed_h", "seed_i", "seed_j")),
        ("50k", commit_50k, ("seed_k", "seed_l", "seed_m")),
    ):
        for seed in seeds:
            residual_path = _no_range_path_for_size(
                archive_root,
                commit,
                seed,
                "history_only_residualize_action_value",
                size,
            )
            baseline_path = _no_range_path_for_size(
                archive_root,
                commit,
                seed,
                "history_only_action_value_only",
                size,
            )
            residual_scores = _real_scores_by_layer(residual_path)
            baseline_score = _nuisance_score(baseline_path)
            for layer, score in sorted(residual_scores.items()):
                rows.append(
                    {
                        "size": size,
                        "seed": seed,
                        "layer": layer,
                        "residual_top10": score,
                        "baseline_top10": baseline_score,
                        "delta": score - baseline_score,
                    }
                )
    return rows


def _fresh50_control_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Fresh 50k No-Range Control Pack",
        "",
        "## Repeat Deltas",
        "",
        "| seed | repeat | layer | residual top10 | baseline top10 | delta |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in payload["repeat_rows"]:
        lines.append(
            "| {seed} | {repeat_index} | {layer} | {residual_top10:.6f} | {baseline_top10:.6f} | {delta:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Repeat Cluster CIs",
            "",
            "| seed | repeat | layer | hands | mean delta | lower | upper |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["repeat_ci_rows"]:
        lines.append(
            "| {seed} | {repeat_index} | {layer} | {hand_count} | {mean_delta:.6f} | "
            "{lower_delta:.6f} | {upper_delta:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Label Controls",
            "",
            "| seed | label control | real top10 | control top10 | delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in payload["label_rows"]:
        lines.append(
            "| {seed} | {label_control} | {real_top10:.6f} | {control_top10:.6f} | "
            "{delta_real_minus_control:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## History Controls",
            "",
            "| seed | control | input control | feature control | layer | ordered top10 | control top10 | delta |",
            "|---|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in payload["history_rows"]:
        lines.append(
            "| {seed} | {control} | {input_control} | {feature_control} | {layer} | "
            "{ordered_encoder_layer_2_top10:.6f} | {control_top10:.6f} | "
            "{delta_ordered_minus_control:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Matched History Delta CIs",
            "",
            "| seed | control | hands | ordered delta | control delta | diff mean | lower | upper |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["history_delta_ci_rows"]:
        lines.append(
            "| {seed} | {control} | {hand_count} | {ordered_mean_delta:.6f} | "
            "{control_mean_delta:.6f} | {mean_delta:.6f} | {lower_delta:.6f} | "
            "{upper_delta:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| metric | value |",
            "|---|---:|",
            "| encoder_layer_2 repeat mean delta | {encoder_layer_2_repeat_delta_mean:.6f} |".format(**payload),
            "| encoder_layer_2 repeat min delta | {encoder_layer_2_repeat_delta_min:.6f} |".format(**payload),
            "| encoder_layer_2 repeat min CI lower | {encoder_layer_2_repeat_ci_min_lower_delta:.6f} |".format(**payload),
            "| label control min delta | {label_control_delta_min:.6f} |".format(**payload),
            "| history control min delta | {history_control_delta_min:.6f} |".format(**payload),
            "| matched history min CI lower | {history_delta_ci_min_lower_delta:.6f} |".format(**payload),
            "",
        ]
    )
    return "\n".join(lines)


def build_no_range_fresh50_control_pack(
    archive_root: Path,
    output_dir: Path,
    commit_50k: str,
    commit_10k: str = "9a368b2",
    *,
    bootstrap_count: int = 1000,
    bootstrap_seed: int = 20260602,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Build the fresh 50k supplemental no-range control pack."""

    repeat_rows = _fresh50_repeat_rows(archive_root, commit_50k)
    repeat_ci_rows = _fresh50_repeat_ci_rows(
        archive_root,
        commit_50k,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    label_rows = _fresh50_label_rows(archive_root, commit_50k)
    history_rows = _fresh50_history_rows(archive_root, commit_50k)
    history_delta_ci_rows = _fresh50_history_delta_ci_rows(
        archive_root,
        commit_50k,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    scaling_rows = _fresh50_scaling_rows(archive_root, commit_10k, commit_50k)
    encoder_repeat = [row for row in repeat_rows if row["layer"] == "encoder_layer_2"]
    encoder_repeat_ci = [row for row in repeat_ci_rows if row["layer"] == "encoder_layer_2"]
    payload = {
                "version": "1.0.13",
        "commit_50k": commit_50k,
        "commit_10k": commit_10k,
        "locked_layer": "encoder_layer_2",
        "best_over_layers_rule": "descriptive_only_not_used_for_pass_fail",
        "label_control_ci_rule": "not_paired_correctness_because_test_labels_are_shuffled",
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "repeat_rows": repeat_rows,
        "repeat_ci_rows": repeat_ci_rows,
        "label_rows": label_rows,
        "history_rows": history_rows,
        "history_delta_ci_rows": history_delta_ci_rows,
        "scaling_rows": scaling_rows,
        "encoder_layer_2_repeat_delta_mean": _mean([float(row["delta"]) for row in encoder_repeat]),
        "encoder_layer_2_repeat_delta_min": min(float(row["delta"]) for row in encoder_repeat),
        "encoder_layer_2_repeat_ci_min_lower_delta": min(float(row["lower_delta"]) for row in encoder_repeat_ci),
        "encoder_layer_2_repeat_ci_all_lower_above_zero": all(
            float(row["lower_delta"]) > 0.0 for row in encoder_repeat_ci
        ),
        "label_control_delta_min": min(float(row["delta_real_minus_control"]) for row in label_rows),
        "history_control_delta_min": min(float(row["delta_ordered_minus_control"]) for row in history_rows),
        "history_delta_ci_min_lower_delta": min(float(row["lower_delta"]) for row in history_delta_ci_rows),
        "history_delta_ci_all_lower_above_zero": all(
            float(row["lower_delta"]) > 0.0 for row in history_delta_ci_rows
        ),
        "recommendation": "Use this pack only after fresh 50k base verification passes.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "limit_ar_no_range_fresh50_control_pack.json", payload)
    (output_dir / "limit_ar_no_range_fresh50_control_pack.md").write_text(
        _fresh50_control_markdown(payload),
        encoding="utf-8",
    )
    return payload


FINAL_LOCKED_CANDIDATE_LAYER = "encoder_layer_2"
FINAL_BOUNDARY_LAYER = "sequence_hidden"
FINAL_CI_KEYS = ("mean_delta", "lower_delta", "upper_delta", "confidence_level", "resampling_unit")
MAX_FINAL_TABLE_DROPPED_FRACTION = 0.05


def _source_artifact_entry(path: Path, role: str, required: bool) -> dict[str, object]:
    entry: dict[str, object] = {
        "role": role,
        "path": str(path),
        "required": required,
        "present": path.exists(),
    }
    if path.exists():
        entry["sha256"] = _file_sha256(path)
        entry["size_bytes"] = path.stat().st_size
    return entry


def _optional_json_object(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    if not path.exists():
        return None
    return _read_json_object(path)


def _validate_final_ci_row(row: dict[str, object], source: str) -> None:
    missing = [key for key in FINAL_CI_KEYS if key not in row]
    if missing:
        raise ValueError(f"final main table row from {source} is missing CI fields: {missing}")
    layer = str(row.get("layer", ""))
    if "best" in layer:
        raise ValueError(f"final main table row cannot use best-layer selection: {source}/{layer}")
    if int(row.get("sample_count", 0)) <= 0:
        raise ValueError(f"final main table row must include paired sample_count: {source}")
    if int(row.get("hand_count", 0)) <= 0:
        raise ValueError(f"final main table row must include hand_count/bucket_count: {source}")
    dropped_fraction = float(row.get("dropped_fraction", 0.0))
    if dropped_fraction > MAX_FINAL_TABLE_DROPPED_FRACTION:
        raise ValueError(
            f"final main table row exceeds dropped-fraction coverage gate: {source}={dropped_fraction:.6f}"
        )


def _diagnostic_label(lower_delta: float, upper_delta: float) -> str:
    if lower_delta > 0.0:
        return "positive_ci_lower_above_zero"
    if upper_delta < 0.0:
        return "negative_ci_upper_below_zero"
    return "ci_overlaps_zero"


def _layer_role(layer: str) -> str:
    if layer == FINAL_LOCKED_CANDIDATE_LAYER:
        return "locked_candidate"
    if layer == FINAL_BOUNDARY_LAYER:
        return "primary_final_boundary"
    if layer == "nuisance":
        return "nuisance_baseline"
    return "exploratory"


def _table_role(layer: str, source_table: str) -> str:
    if source_table == "no_range_bootstrap" and layer == FINAL_BOUNDARY_LAYER:
        return "boundary"
    if _layer_role(layer) == "exploratory":
        return "appendix"
    return "main"


def _fingerprint_strings(values: list[object]) -> str | None:
    strings = [str(value) for value in values if value is not None]
    if not strings:
        return None
    digest = hashlib.sha256()
    for value in sorted(strings):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _final_row(
    *,
    source_table: str,
    source_sha256: str | None,
    source_path: Path | None,
    source_index: int,
    row: dict[str, object],
    claim_gate: str,
    baseline_label: str,
    code_commit: str,
) -> dict[str, object]:
    _validate_final_ci_row(row, source_table)
    layer = str(row["layer"])
    if _table_role(layer, source_table) == "main" and layer not in {FINAL_LOCKED_CANDIDATE_LAYER, "nuisance"}:
        raise ValueError(f"final main table row must use locked candidate or nuisance baseline layer: {layer}")

    lower_delta = float(row["lower_delta"])
    upper_delta = float(row["upper_delta"])
    seed = str(row.get("seed", "aggregate"))
    comparison = str(row.get("comparison", row.get("control", "residual_vs_baseline")))
    row_id = f"{source_table}:{seed}:{comparison}:{layer}:{source_index}"
    correctness_sha256 = _fingerprint_strings(
        [
            row.get("residual_correctness_sha256"),
            row.get("baseline_correctness_sha256"),
            row.get("real_correctness_sha256"),
            row.get("ordered_residual_correctness_sha256"),
            row.get("ordered_baseline_correctness_sha256"),
            row.get("control_residual_correctness_sha256"),
            row.get("control_baseline_correctness_sha256"),
        ]
    )
    probe_metrics_sha256 = _fingerprint_strings(
        [
            row.get("residual_checksum"),
            row.get("baseline_checksum"),
            row.get("real_checksum"),
            row.get("ordered_residual_checksum"),
            row.get("ordered_baseline_checksum"),
            row.get("control_residual_checksum"),
            row.get("control_baseline_checksum"),
            row.get("composition_checksum"),
            row.get("action_value_composition_checksum"),
            row.get("residual_composition_checksum"),
        ]
    )
    sample_count = int(row["sample_count"])
    hand_count = int(row["hand_count"])
    dropped_point_count = int(row.get("dropped_point_count", 0))
    dropped_fraction = float(row.get("dropped_fraction", 0.0))
    return {
        "artifact_id": f"{source_table}:{source_sha256[:12] if source_sha256 else 'untracked'}",
        "row_id": row_id,
        "claim_gate": claim_gate,
        "diagnostic_label": _diagnostic_label(lower_delta, upper_delta),
        "source_table": source_table,
        "source_path": str(source_path) if source_path is not None else None,
        "source_sha256": source_sha256,
        "table_role": _table_role(layer, source_table),
        "model_family": "limit_ar_action_value_only_4layer",
        "checkpoint_sha256": row.get("train_checksum"),
        "train_config_sha256": None,
        "dataset_id": "limit_holdem_50k",
        "split_seed": None,
        "training_seed": seed,
        "probe_seed": row.get("residual_probe_seed"),
        "control_seed": row.get("baseline_probe_seed"),
        "target": "opponent_hand",
        "metric": "top10_correct_delta",
        "top_k": 10,
        "layer": layer,
        "layer_role": _layer_role(layer),
        "layer_policy": "locked_candidate_encoder_layer_2_no_best_layer_main_rows",
        "input_control": row.get("input_control", "history_only"),
        "feature_control": row.get("feature_control", "residualize_action_value"),
        "label_control": row.get("label_control", "none"),
        "nuisance_features": baseline_label,
        "baseline_artifact_id": f"{source_table}:{baseline_label}",
        "paired_identity_keys": ["point_id", "hand_id", "step_index", "target_label"],
        "resampling_method": "paired_cluster_bootstrap",
        "resampling_unit": row["resampling_unit"],
        "bootstrap_count": row.get("bootstrap_count"),
        "bootstrap_seed": row.get("bootstrap_seed"),
        "confidence_level": row["confidence_level"],
        "mean": row.get("residual_top10", row.get("real_top10")),
        "baseline_mean": row.get("baseline_top10"),
        "delta": float(row["mean_delta"]),
        "mean_delta": float(row["mean_delta"]),
        "ci_lower": lower_delta,
        "ci_upper": upper_delta,
        "lower_delta": lower_delta,
        "upper_delta": upper_delta,
        "hand_count": hand_count,
        "decision_point_count": sample_count,
        "paired_point_count": sample_count,
        "bucket_key_schema": None,
        "bucket_count": hand_count,
        "matched_bucket_count": hand_count,
        "dropped_point_count": dropped_point_count,
        "dropped_fraction": dropped_fraction,
        "max_allowed_dropped_fraction": MAX_FINAL_TABLE_DROPPED_FRACTION,
        "coverage_status": "passed",
        "correctness_sha256": correctness_sha256,
        "probe_metrics_sha256": probe_metrics_sha256,
        "source_manifest_sha256": source_sha256,
        "reproduction_command": "python scripts/reproduce_paper_tables.py --archive-root <archive_root> --output-dir <output_dir>",
        "expected_outputs": ["final_diagnostic_tables.json", "final_diagnostic_tables.md"],
        "code_commit": code_commit,
        "environment_hash": checksum_payload({"python_version": platform.python_version()}),
    }


def _final_rows_from_bootstrap_pack(
    pack: dict[str, object],
    *,
    path: Path,
    source_sha256: str | None,
    code_commit: str,
) -> list[dict[str, object]]:
    rows = pack.get("ci_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"bootstrap pack has no ci_rows: {path}")
    final_rows = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"bootstrap ci row must be an object: {path}:{index}")
        final_rows.append(
            _final_row(
                source_table="no_range_bootstrap",
                source_sha256=source_sha256,
                source_path=path,
                source_index=index,
                row=row,
                claim_gate="no_range_action_value_residual_ci",
                baseline_label="action_value_only",
                code_commit=code_commit,
            )
        )
    return final_rows


def _final_rows_from_composition_pack(
    pack: dict[str, object],
    *,
    path: Path,
    source_sha256: str | None,
    code_commit: str,
) -> list[dict[str, object]]:
    rows = pack.get("rows")
    if not isinstance(rows, list) or not rows:
        return []
    final_rows = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"composition ci row must be an object: {path}:{index}")
        comparison = str(row.get("comparison", ""))
        final_rows.append(
            _final_row(
                source_table="composition_nuisance",
                source_sha256=source_sha256,
                source_path=path,
                source_index=index,
                row=row,
                claim_gate=f"composition_nuisance:{comparison}",
                baseline_label="action_value_or_action_value_composition",
                code_commit=code_commit,
            )
        )
    return final_rows


def _final_rows_from_history_composition_pack(
    pack: dict[str, object],
    *,
    path: Path,
    source_sha256: str | None,
    code_commit: str,
) -> list[dict[str, object]]:
    rows = pack.get("rows")
    if not isinstance(rows, list) or not rows:
        return []
    final_rows = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"history-composition ci row must be an object: {path}:{index}")
        control = str(row.get("control", ""))
        final_rows.append(
            _final_row(
                source_table="history_composition",
                source_sha256=source_sha256,
                source_path=path,
                source_index=index,
                row=row,
                claim_gate=f"history_composition:{control}",
                baseline_label="matched_action_value_only",
                code_commit=code_commit,
            )
        )
    return final_rows


def _final_tables_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Final Diagnostic Tables",
        "",
        "## Rule",
        "",
        "- Main rows require paired/bucket bootstrap CI.",
        "- `encoder_layer_2` is the locked candidate layer.",
        "- Best-layer selection is appendix-only and absent from these main rows.",
        "",
        "## Rows",
        "",
        "| role | gate | seed | layer | layer role | delta | CI lower | CI upper | unit | label |",
        "|---|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {table_role} | {claim_gate} | {training_seed} | {layer} | {layer_role} | "
            "{delta:.6f} | {ci_lower:.6f} | {ci_upper:.6f} | {resampling_unit} | "
            "{diagnostic_label} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Gate Summary",
            "",
            "| item | value |",
            "|---|---|",
            f"| all main rows have CI | `{payload['all_main_rows_have_ci']}` |",
            f"| uses best-layer selection | `{payload['uses_best_layer_selection']}` |",
            f"| locked candidate layer | `{payload['locked_candidate_layer']}` |",
            f"| final hidden boundary layer | `{payload['final_hidden_boundary_layer']}` |",
            "",
        ]
    )
    return "\n".join(lines)


def build_final_diagnostic_tables(
    *,
    output_dir: Path,
    bootstrap_pack_path: Path,
    composition_pack_path: Path | None = None,
    history_composition_pack_path: Path | None = None,
    code_commit: str = "unknown",
) -> dict[str, object]:
    """Assemble final paper tables from CI-backed locked-layer packs."""

    source_entries = [_source_artifact_entry(bootstrap_pack_path, "no_range_bootstrap_pack", True)]
    if not source_entries[0]["present"]:
        raise FileNotFoundError(f"required bootstrap pack is missing: {bootstrap_pack_path}")
    bootstrap_sha256 = str(source_entries[0]["sha256"])
    rows = _final_rows_from_bootstrap_pack(
        _read_json_object(bootstrap_pack_path),
        path=bootstrap_pack_path,
        source_sha256=bootstrap_sha256,
        code_commit=code_commit,
    )

    if composition_pack_path is not None:
        source_entries.append(_source_artifact_entry(composition_pack_path, "composition_nuisance_pack", False))
        composition_pack = _optional_json_object(composition_pack_path)
        if composition_pack is not None:
            rows.extend(
                _final_rows_from_composition_pack(
                    composition_pack,
                    path=composition_pack_path,
                    source_sha256=str(source_entries[-1].get("sha256")),
                    code_commit=code_commit,
                )
            )
    if history_composition_pack_path is not None:
        source_entries.append(
            _source_artifact_entry(history_composition_pack_path, "history_composition_pack", False)
        )
        history_pack = _optional_json_object(history_composition_pack_path)
        if history_pack is not None:
            rows.extend(
                _final_rows_from_history_composition_pack(
                    history_pack,
                    path=history_composition_pack_path,
                    source_sha256=str(source_entries[-1].get("sha256")),
                    code_commit=code_commit,
                )
            )

    main_rows = [row for row in rows if row["table_role"] == "main"]
    all_main_rows_have_ci = all(
        row.get("ci_lower") is not None
        and row.get("ci_upper") is not None
        and row.get("confidence_level") is not None
        for row in main_rows
    )
    uses_best_layer_selection = any("best" in str(row["layer"]) for row in rows)
    if not all_main_rows_have_ci:
        raise ValueError("final diagnostic table contains a main row without CI")
    if uses_best_layer_selection:
        raise ValueError("final diagnostic table contains best-layer main evidence")

    source_manifest = {
        "schema_version": "1.0",
        "sources": source_entries,
    }
    source_manifest_sha256 = checksum_payload(source_manifest)
    for row in rows:
        row["source_manifest_sha256"] = source_manifest_sha256
    payload = {
                "version": "1.12",
        "schema_version": "1.0",
        "locked_candidate_layer": FINAL_LOCKED_CANDIDATE_LAYER,
        "final_hidden_boundary_layer": FINAL_BOUNDARY_LAYER,
        "layer_lock": {
            "candidate_layer": FINAL_LOCKED_CANDIDATE_LAYER,
            "boundary_layer": FINAL_BOUNDARY_LAYER,
            "candidate_set": [FINAL_BOUNDARY_LAYER, FINAL_LOCKED_CANDIDATE_LAYER],
            "lock_basis": "predeclared in v1.0.8/v1.0.9 before final CI table assembly",
            "best_over_layers_rule": "appendix_descriptive_only",
        },
        "required_main_row_ci_fields": list(FINAL_CI_KEYS),
        "main_row_count": len(main_rows),
        "row_count": len(rows),
        "all_main_rows_have_ci": all_main_rows_have_ci,
        "uses_best_layer_selection": uses_best_layer_selection,
        "source_manifest": source_manifest,
        "source_manifest_sha256": source_manifest_sha256,
        "rows": rows,
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "final_diagnostic_tables.json", payload)
    write_json_artifact(output_dir / "final_table_source_manifest.json", source_manifest)
    (output_dir / "final_diagnostic_tables.md").write_text(_final_tables_markdown(payload), encoding="utf-8")
    return payload


def _artifact_output_entry(path: Path, output_dir: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(output_dir)),
        "name": path.name,
        "present": path.exists(),
        "sha256": _file_sha256(path) if path.exists() else None,
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def _reproduction_manifest_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Paper Table Reproduction Manifest",
        "",
        "## Command",
        "",
        "```text",
        str(payload["command"]),
        "```",
        "",
        "## Inputs",
        "",
        f"- archive_root: `{payload['archive_root']}`",
        f"- fresh50_commit: `{payload['fresh50_commit']}`",
        f"- bootstrap_count: `{payload['bootstrap_count']}`",
        f"- bootstrap_seed: `{payload['bootstrap_seed']}`",
        f"- confidence_level: `{payload['confidence_level']}`",
        "",
        "## Outputs",
        "",
    ]
    for output in payload["outputs"]:
        lines.append(f"- `{output['relative_path']}` sha256=`{output['sha256']}`")
    lines.extend(
        [
            "",
            "## Missing Optional Packs",
            "",
        ]
    )
    missing = payload["missing_optional_packs"]
    if missing:
        for item in missing:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def build_paper_table_reproduction(
    *,
    archive_root: Path,
    output_dir: Path,
    fresh50_commit: str,
    bootstrap_count: int = 1000,
    bootstrap_seed: int = 20260602,
    confidence_level: float = 0.95,
    composition_commit: str | None = None,
    history_composition_commit: str | None = None,
    composition_pack_path: Path | None = None,
    history_composition_pack_path: Path | None = None,
    command: str = "python scripts/reproduce_paper_tables.py",
    code_commit: str = "unknown",
) -> dict[str, object]:
    """Run the one-command local paper table reproduction path."""

    pack_dir = output_dir / "packs"
    paper_table_dir = output_dir / "paper_tables"
    build_no_range_bootstrap_pack(
        archive_root=archive_root,
        output_dir=pack_dir,
        commit=fresh50_commit,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    bootstrap_pack_path = pack_dir / "limit_ar_no_range_bootstrap_pack.json"
    if composition_commit is not None:
        build_no_range_composition_nuisance_pack(
            archive_root=archive_root,
            output_dir=pack_dir,
            commit=composition_commit,
            bootstrap_count=bootstrap_count,
            bootstrap_seed=bootstrap_seed,
            confidence_level=confidence_level,
        )
        composition_pack_path = pack_dir / "limit_ar_no_range_composition_nuisance_pack.json"
    if history_composition_commit is not None:
        build_no_range_history_composition_pack(
            archive_root=archive_root,
            output_dir=pack_dir,
            commit=history_composition_commit,
            bootstrap_count=bootstrap_count,
            bootstrap_seed=bootstrap_seed,
            confidence_level=confidence_level,
        )
        history_composition_pack_path = pack_dir / "limit_ar_no_range_history_composition_pack.json"
    missing_optional_packs = []
    if composition_pack_path is not None and not composition_pack_path.exists():
        missing_optional_packs.append(str(composition_pack_path))
    if history_composition_pack_path is not None and not history_composition_pack_path.exists():
        missing_optional_packs.append(str(history_composition_pack_path))

    build_final_diagnostic_tables(
        output_dir=paper_table_dir,
        bootstrap_pack_path=bootstrap_pack_path,
        composition_pack_path=composition_pack_path,
        history_composition_pack_path=history_composition_pack_path,
        code_commit=code_commit,
    )
    output_paths = [
        bootstrap_pack_path,
        pack_dir / "limit_ar_no_range_bootstrap_pack.md",
        paper_table_dir / "final_diagnostic_tables.json",
        paper_table_dir / "final_diagnostic_tables.md",
        paper_table_dir / "final_table_source_manifest.json",
    ]
    if composition_pack_path is not None and composition_pack_path.exists():
        output_paths.extend(
            [
                composition_pack_path,
                pack_dir / "limit_ar_no_range_composition_nuisance_pack.md",
            ]
        )
    if history_composition_pack_path is not None and history_composition_pack_path.exists():
        output_paths.extend(
            [
                history_composition_pack_path,
                pack_dir / "limit_ar_no_range_history_composition_pack.md",
            ]
        )
    outputs = [_artifact_output_entry(path, output_dir) for path in output_paths]
    manifest = {
                "version": "1.13",
        "archive_root": str(archive_root),
        "output_dir": str(output_dir),
        "fresh50_commit": fresh50_commit,
        "composition_commit": composition_commit,
        "history_composition_commit": history_composition_commit,
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "command": command,
        "code_commit": code_commit,
        "environment_hash": checksum_payload({"python_version": platform.python_version()}),
        "expected_outputs": [output["name"] for output in outputs],
        "outputs": outputs,
        "missing_optional_packs": missing_optional_packs,
        "checksum": "",
    }
    manifest["checksum"] = checksum_payload({key: value for key, value in manifest.items() if key != "checksum"})
    write_json_artifact(output_dir / "reproduction_manifest.json", manifest)
    (output_dir / "reproduction_manifest.md").write_text(
        _reproduction_manifest_markdown(manifest),
        encoding="utf-8",
    )
    return manifest
