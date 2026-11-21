from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.probe.tables import (
    _bootstrap_mean_ci,
    _mean,
    _no_range_path_for_size,
    _paired_correctness_deltas,
    _paired_correctness_deltas_for_layers,
    _test_point_buckets_from_train_config,
)


NO_RANGE_CAPACITY_SEEDS = ("seed_k", "seed_l", "seed_m")


def _read_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required probe metrics file is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"probe metrics must be a JSON object: {path}")
    return data


def _locked_layer_score(metrics: dict[str, Any], layer: str) -> float:
    records = metrics.get("records")
    if not isinstance(records, list):
        raise ValueError("probe metrics must include records")
    matches = [
        record
        for record in records
        if isinstance(record, dict) and record.get("control") == "none" and record.get("layer") == layer
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one real record for locked layer {layer}")
    return float(matches[0]["score"])


def _row(path: Path, metrics: dict[str, Any], layer: str, table_role: str) -> dict[str, Any]:
    probe_family = str(metrics.get("probe_family", "linear"))
    if table_role == "main" and probe_family != "linear":
        raise ValueError("main probe-capacity row must be linear")
    if table_role == "appendix" and probe_family != "mlp":
        raise ValueError("appendix probe-capacity row must be mlp")
    return {
        "table_role": table_role,
        "probe_family": probe_family,
        "probe_capacity_role": metrics.get("probe_capacity_role"),
        "mlp_hidden_size": metrics.get("mlp_hidden_size"),
        "target": metrics.get("target"),
        "feature_control": metrics.get("feature_control"),
        "label_control": metrics.get("label_control"),
        "layer": layer,
        "score": _locked_layer_score(metrics, layer),
        "metrics_path": str(path),
        "metrics_checksum": metrics.get("checksum"),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Probe Capacity Sensitivity",
        "",
        "## Rule",
        "",
        "- Linear probe rows are main evidence.",
        "- MLP rows are appendix-only sensitivity.",
        "- MLP rows are appendix-only even when they improve decodability.",
        "",
        "## Rows",
        "",
        "| role | family | layer | score |",
        "|---|---|---|---:|",
    ]
    for row in payload["rows"]:
        lines.append("| {table_role} | {probe_family} | {layer} | {score:.6f} |".format(**row))
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- verdict agreement: `{payload['verdict_agreement']}`",
            f"- score delta MLP-linear: `{payload['score_delta_mlp_minus_linear']:.6f}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_probe_capacity_sensitivity_report(
    *,
    linear_metrics_path: Path,
    mlp_metrics_path: Path,
    output_dir: Path,
    locked_layer: str = "encoder_layer_2",
    tolerance: float = 0.02,
) -> dict[str, Any]:
    """Build appendix-only MLP probe-capacity sensitivity report."""

    linear_metrics = _read_metrics(linear_metrics_path)
    mlp_metrics = _read_metrics(mlp_metrics_path)
    linear_row = _row(linear_metrics_path, linear_metrics, locked_layer, "main")
    mlp_row = _row(mlp_metrics_path, mlp_metrics, locked_layer, "appendix")
    score_delta = float(mlp_row["score"]) - float(linear_row["score"])
    payload = {
                "version": "1.14",
        "locked_layer": locked_layer,
        "main_probe_family": "linear",
        "appendix_probe_family": "mlp",
        "tolerance": tolerance,
        "rows": [linear_row, mlp_row],
        "score_delta_mlp_minus_linear": score_delta,
        "verdict_agreement": abs(score_delta) <= tolerance,
        "claim_rule": "MLP sensitivity can only downgrade the main conclusion, not upgrade a belief claim.",
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "probe_capacity_sensitivity_report.json", payload)
    (output_dir / "probe_capacity_sensitivity_report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _validate_appendix_mlp_metrics(path: Path, metrics: dict[str, Any], feature_control: str) -> None:
    if metrics.get("probe_family") != "mlp":
        raise ValueError(f"appendix MLP capacity row must use probe_family=mlp: {path}")
    if metrics.get("probe_capacity_role") != "appendix_sensitivity":
        raise ValueError(f"appendix MLP capacity row must use appendix_sensitivity role: {path}")
    if metrics.get("target") != "opponent_hand":
        raise ValueError(f"MLP capacity row must target opponent_hand: {path}")
    if metrics.get("input_control") != "history_only":
        raise ValueError(f"MLP capacity row must use history_only input: {path}")
    if metrics.get("feature_control") != feature_control:
        raise ValueError(f"MLP feature_control mismatch: {path}")
    if metrics.get("label_control") != "none":
        raise ValueError(f"MLP capacity row must use real labels: {path}")
    if list(metrics.get("layers", [])) != ["encoder_layer_2"]:
        raise ValueError(f"MLP capacity row must use locked encoder_layer_2: {path}")
    if not (path.parent / "probe_correctness.jsonl").exists():
        raise FileNotFoundError(f"appendix MLP capacity row requires correctness artifact: {path.parent}")


def _validate_linear_nuisance_metrics(path: Path, metrics: dict[str, Any], feature_control: str) -> None:
    if metrics.get("probe_family", "linear") != "linear":
        raise ValueError(f"capacity nuisance baseline must be linear: {path}")
    if metrics.get("target") != "opponent_hand":
        raise ValueError(f"capacity nuisance baseline must target opponent_hand: {path}")
    if metrics.get("input_control") != "history_only":
        raise ValueError(f"capacity nuisance baseline must use history_only input: {path}")
    if metrics.get("feature_control") != feature_control:
        raise ValueError(f"capacity nuisance feature_control mismatch: {path}")
    if list(metrics.get("layers", [])) != ["nuisance"]:
        raise ValueError(f"capacity nuisance baseline must use nuisance layer: {path}")
    if not (path.parent / "probe_correctness.jsonl").exists():
        raise FileNotFoundError(f"capacity nuisance baseline requires correctness artifact: {path.parent}")


def _capacity_test_point_buckets(commit_root: Path, seed: str) -> tuple[dict[str, str], str]:
    sidecar = commit_root / "outputs" / "tables" / f"no_range_capacity_{seed}_matched_composition_buckets.json"
    if sidecar.exists():
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        buckets = data.get("buckets")
        if not isinstance(buckets, dict) or not buckets:
            raise ValueError(f"capacity bucket sidecar has no buckets: {sidecar}")
        return {str(key): str(value) for key, value in buckets.items()}, "sidecar"
    train_config = commit_root / "configs" / "train" / f"limit_ar_action_value_only_{seed}_50k_4layer.json"
    if not train_config.exists():
        train_config = Path("configs") / "train" / f"limit_ar_action_value_only_{seed}_50k_4layer.json"
    return _test_point_buckets_from_train_config(train_config), "train_config"


def _capacity_paired_row(
    *,
    seed: str,
    gate: str,
    paired: dict[str, object],
    mlp_path: Path,
    baseline_path: Path,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
    row_index: int,
) -> dict[str, object]:
    interval = _bootstrap_mean_ci(
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
        "model_family": "limit_ar_no_range_50k_4layer",
        "seed": seed,
        "gate": gate,
        "table_role": "appendix",
        "probe_family": "mlp",
        "baseline_family": "linear_nuisance",
        "layer": "encoder_layer_2",
        "resampling_unit": "hand_id",
        "sample_count": len(paired["deltas"]),
        "hand_count": paired["hand_count"],
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "mlp_top10": paired["residual_top10"],
        "baseline_top10": paired["baseline_top10"],
        **interval,
        "decision_point_mean_delta": decision_interval["mean_delta"],
        "decision_point_lower_delta": decision_interval["lower_delta"],
        "decision_point_upper_delta": decision_interval["upper_delta"],
        "paired_outcomes": paired["paired_outcomes"],
        "mlp_metrics_path": str(mlp_path),
        "baseline_metrics_path": str(baseline_path),
        "mlp_correctness_sha256": paired["residual_correctness_sha256"],
        "baseline_correctness_sha256": paired["baseline_correctness_sha256"],
    }


def _capacity_matched_row(
    *,
    commit_root: Path,
    seed: str,
    paired: dict[str, object],
    mlp_path: Path,
    baseline_path: Path,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
    row_index: int,
) -> dict[str, object]:
    buckets, bucket_source = _capacity_test_point_buckets(commit_root, seed)
    point_deltas = paired["point_deltas"]
    if not isinstance(point_deltas, dict):
        raise ValueError("capacity point_deltas must be a dict")
    common_ids = sorted(set(point_deltas) & set(buckets))
    if not common_ids:
        raise ValueError(f"no shared point_ids for capacity matched-composition seed: {seed}")

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
        raise ValueError(f"capacity matched-composition buckets have no non-singleton support: {seed}")

    interval = _bootstrap_mean_ci(
        bucket_mean_deltas,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed + 200000 + row_index,
        confidence_level=confidence_level,
    )
    return {
        "model_family": "limit_ar_no_range_50k_4layer",
        "seed": seed,
        "gate": "C3_mlp_matched_composition",
        "table_role": "appendix",
        "probe_family": "mlp",
        "baseline_family": "linear_action_value_composition",
        "layer": "encoder_layer_2",
        "resampling_unit": "matched_composition_bucket",
        "sample_count": len(matched_deltas),
        "hand_count": paired["hand_count"],
        "bucket_count": len(bucket_to_deltas),
        "matched_bucket_count": len(bucket_mean_deltas),
        "singleton_bucket_count": singleton_bucket_count,
        "dropped_point_count": dropped_point_count,
        "dropped_fraction": dropped_point_count / len(common_ids),
        "bucket_source": bucket_source,
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "mlp_top10": float(paired["residual_top10"]),
        "baseline_top10": float(paired["baseline_top10"]),
        **interval,
        "decision_point_mean_delta": _mean(matched_deltas),
        "decision_point_lower_delta": min(matched_deltas),
        "decision_point_upper_delta": max(matched_deltas),
        "paired_outcomes": paired["paired_outcomes"],
        "mlp_metrics_path": str(mlp_path),
        "baseline_metrics_path": str(baseline_path),
        "mlp_correctness_sha256": paired["residual_correctness_sha256"],
        "baseline_correctness_sha256": paired["baseline_correctness_sha256"],
    }


def _capacity_s6_s7_rows(
    *,
    archive_root: Path,
    mlp_commit: str,
    linear_commit: str,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> list[dict[str, object]]:
    rows = []
    commit_root = archive_root / mlp_commit
    for seed_index, seed in enumerate(NO_RANGE_CAPACITY_SEEDS):
        c1_mlp_path = _no_range_path_for_size(
            archive_root,
            mlp_commit,
            seed,
            "history_only_residualize_action_value_mlp",
            "50k",
        )
        c2_mlp_path = _no_range_path_for_size(
            archive_root,
            mlp_commit,
            seed,
            "residualize_action_value_history_composition_mlp",
            "50k",
        )
        c1_baseline_path = _no_range_path_for_size(
            archive_root,
            linear_commit,
            seed,
            "history_only_action_value_only",
            "50k",
        )
        c2_baseline_path = _no_range_path_for_size(
            archive_root,
            linear_commit,
            seed,
            "action_value_history_composition_only",
            "50k",
        )
        c1_mlp_metrics = _read_metrics(c1_mlp_path)
        c2_mlp_metrics = _read_metrics(c2_mlp_path)
        c1_baseline_metrics = _read_metrics(c1_baseline_path)
        c2_baseline_metrics = _read_metrics(c2_baseline_path)
        _validate_appendix_mlp_metrics(c1_mlp_path, c1_mlp_metrics, "residualize_action_value")
        _validate_appendix_mlp_metrics(
            c2_mlp_path,
            c2_mlp_metrics,
            "residualize_action_value_history_composition",
        )
        _validate_linear_nuisance_metrics(c1_baseline_path, c1_baseline_metrics, "action_value_only")
        _validate_linear_nuisance_metrics(
            c2_baseline_path,
            c2_baseline_metrics,
            "action_value_history_composition_only",
        )

        c1_paired = _paired_correctness_deltas(c1_mlp_path, c1_baseline_path, "encoder_layer_2")
        c2_paired = _paired_correctness_deltas_for_layers(
            c2_mlp_path,
            "encoder_layer_2",
            c2_baseline_path,
            "nuisance",
        )
        rows.append(
            _capacity_paired_row(
                seed=seed,
                gate="C1_mlp_action_value_residual",
                paired=c1_paired,
                mlp_path=c1_mlp_path,
                baseline_path=c1_baseline_path,
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed,
                confidence_level=confidence_level,
                row_index=seed_index * 3,
            )
        )
        rows.append(
            _capacity_paired_row(
                seed=seed,
                gate="C2_mlp_composition_residual",
                paired=c2_paired,
                mlp_path=c2_mlp_path,
                baseline_path=c2_baseline_path,
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed,
                confidence_level=confidence_level,
                row_index=seed_index * 3 + 1,
            )
        )
        rows.append(
            _capacity_matched_row(
                commit_root=commit_root,
                seed=seed,
                paired=c2_paired,
                mlp_path=c2_mlp_path,
                baseline_path=c2_baseline_path,
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed,
                confidence_level=confidence_level,
                row_index=seed_index * 3 + 2,
            )
        )
    return rows


def _capacity_s6_s7_verdict(rows: list[dict[str, object]]) -> str:
    by_gate: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_gate.setdefault(str(row["gate"]), []).append(row)
    c2_rows = by_gate.get("C2_mlp_composition_residual", [])
    c3_rows = by_gate.get("C3_mlp_matched_composition", [])
    if c2_rows and c3_rows and all(float(row["lower_delta"]) > 0.0 for row in c2_rows + c3_rows):
        return "mlp_capacity_caveat"
    if c2_rows and c3_rows and (
        all(float(row["upper_delta"]) < 0.0 for row in c2_rows)
        or all(float(row["upper_delta"]) < 0.0 for row in c3_rows)
    ):
        return "mlp_agrees_shortcut"
    return "mlp_inconclusive"


def _capacity_s6_s7_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Probe Capacity S6/S7 Report",
        "",
        "## Rule",
        "",
        "- Linear probe rows remain main evidence.",
        "- MLP rows are appendix-only capacity sensitivity.",
        "- MLP cannot upgrade a belief claim unless it beats composition priors under C2 and C3.",
        "",
        "## Summary",
        "",
        f"- final verdict: `{payload['final_verdict']}`",
        f"- C2 min lower: `{payload['c2_min_lower_delta']:.6f}`",
        f"- C3 min lower: `{payload['c3_min_lower_delta']:.6f}`",
        "",
        "## Rows",
        "",
        "| seed | gate | unit | mlp top10 | baseline top10 | mean delta | lower | upper |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {seed} | {gate} | {resampling_unit} | {mlp_top10:.6f} | {baseline_top10:.6f} | "
            "{mean_delta:.6f} | {lower_delta:.6f} | {upper_delta:.6f} |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)


def build_probe_capacity_s6_s7_report(
    *,
    archive_root: Path,
    output_dir: Path,
    mlp_commit: str,
    linear_commit: str,
    bootstrap_count: int = 1000,
    bootstrap_seed: int = 20260603,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Build the v1.18 real no-range MLP capacity-sensitivity gate report."""

    rows = _capacity_s6_s7_rows(
        archive_root=archive_root,
        mlp_commit=mlp_commit,
        linear_commit=linear_commit,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )

    def lower_values(gate: str) -> list[float]:
        return [float(row["lower_delta"]) for row in rows if row["gate"] == gate]

    def upper_values(gate: str) -> list[float]:
        return [float(row["upper_delta"]) for row in rows if row["gate"] == gate]

    payload = {
                "version": "1.18",
        "mlp_commit": mlp_commit,
        "linear_commit": linear_commit,
        "model_family": "limit_ar_no_range_50k_4layer",
        "main_probe_family": "linear",
        "appendix_probe_family": "mlp",
        "locked_layer": "encoder_layer_2",
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "rows": rows,
        "row_count": len(rows),
        "seed_count": len(NO_RANGE_CAPACITY_SEEDS),
        "c1_min_lower_delta": min(lower_values("C1_mlp_action_value_residual")),
        "c2_min_lower_delta": min(lower_values("C2_mlp_composition_residual")),
        "c2_max_upper_delta": max(upper_values("C2_mlp_composition_residual")),
        "c3_min_lower_delta": min(lower_values("C3_mlp_matched_composition")),
        "c3_max_upper_delta": max(upper_values("C3_mlp_matched_composition")),
        "claim_rule": "MLP rows are appendix-only and can only create a capacity caveat, not replace main linear rows.",
    }
    payload["final_verdict"] = _capacity_s6_s7_verdict(rows)
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "probe_capacity_s6_s7_report.json", payload)
    (output_dir / "probe_capacity_s6_s7_report.md").write_text(
        _capacity_s6_s7_markdown(payload),
        encoding="utf-8",
    )
    return payload
