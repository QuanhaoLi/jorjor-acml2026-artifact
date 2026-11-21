"""Build the predeclared five-run G5--G7 rebuttal robustness product.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.probe.inspect import build_matched_composition_report
from jorjor.probe.tables import (
    build_no_range_bootstrap_pack,
    build_no_range_composition_nuisance_pack,
    build_no_range_locked_layer_pack,
)


FIVE_RUN_SEEDS = ("seed_k", "seed_l", "seed_m", "seed_n", "seed_o")
LOCKED_LAYER = "encoder_layer_2"
STUDENT_T_975_DF4 = 2.7764451051977987
ROBUST_SHORTCUT_LABEL = "composition_shortcut_supported_five_run_robust"
MIXED_LABEL = "mixed_or_inconclusive_five_run_robustness"
EXPECTED_TRAIN_SEEDS = {
    "seed_k": {"seed": 271, "data_seed": 277, "split_seed": 281},
    "seed_l": {"seed": 283, "data_seed": 293, "split_seed": 307},
    "seed_m": {"seed": 311, "data_seed": 313, "split_seed": 317},
    "seed_n": {"seed": 331, "data_seed": 337, "split_seed": 347},
    "seed_o": {"seed": 349, "data_seed": 353, "split_seed": 359},
}
EXPECTED_PROBE_SEEDS = {
    "seed_k": ((601, 607), (613, 617), (1601, 1602), (1603, 1604), (1605, 1606)),
    "seed_l": ((619, 631), (641, 643), (1701, 1702), (1703, 1704), (1705, 1706)),
    "seed_m": ((647, 653), (659, 661), (1801, 1802), (1803, 1804), (1805, 1806)),
    "seed_n": ((673, 677), (683, 691), (1901, 1902), (1903, 1904), (1905, 1906)),
    "seed_o": ((701, 709), (719, 727), (2001, 2002), (2003, 2004), (2005, 2006)),
}
PROBE_SUFFIXES = (
    "history_only_action_value_only",
    "history_only_residualize_action_value",
    "history_composition_only",
    "action_value_history_composition_only",
    "residualize_action_value_history_composition",
)


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


def _validate_seed_names(seed_names: tuple[str, ...]) -> None:
    if seed_names != FIVE_RUN_SEEDS:
        raise ValueError(f"five-run product requires exact seed order {FIVE_RUN_SEEDS}, got {seed_names}")
    if len(set(seed_names)) != len(seed_names):
        raise ValueError("seed_names must be unique")


def _interval_label(lower: float, upper: float) -> str:
    if lower > 0.0:
        return "positive"
    if upper < 0.0:
        return "negative"
    return "inconclusive"


def _between_run_statistics(values: list[float]) -> dict[str, Any]:
    if len(values) != 5:
        raise ValueError(f"five-run statistics require 5 values, got {len(values)}")
    mean = statistics.mean(values)
    sample_sd = statistics.stdev(values)
    standard_error = sample_sd / math.sqrt(len(values))
    margin = STUDENT_T_975_DF4 * standard_error
    return {
        "resampling_unit": "independent_end_to_end_run",
        "run_count": len(values),
        "degrees_of_freedom": 4,
        "confidence_level": 0.95,
        "t_critical": STUDENT_T_975_DF4,
        "mean_delta": mean,
        "sample_standard_deviation": sample_sd,
        "standard_error": standard_error,
        "lower_delta": mean - margin,
        "upper_delta": mean + margin,
    }


def _summarize_gate(
    *,
    gate: str,
    rows: list[dict[str, Any]],
    seed_names: tuple[str, ...],
    point_key: str,
    lower_key: str,
    upper_key: str,
    within_run_unit: str,
) -> dict[str, Any]:
    by_seed: dict[str, dict[str, Any]] = {}
    for row in rows:
        seed = str(row["seed"])
        if seed in by_seed:
            raise ValueError(f"duplicate {gate} row for {seed}")
        by_seed[seed] = row
    if set(by_seed) != set(seed_names):
        missing = sorted(set(seed_names) - set(by_seed))
        extra = sorted(set(by_seed) - set(seed_names))
        raise ValueError(f"{gate} seed mismatch missing={missing} extra={extra}")

    run_rows = []
    values = []
    for seed in seed_names:
        row = by_seed[seed]
        point = float(row[point_key])
        lower = float(row[lower_key])
        upper = float(row[upper_key])
        values.append(point)
        run_rows.append(
            {
                "seed": seed,
                "mean_delta": point,
                "lower_delta": lower,
                "upper_delta": upper,
                "within_run_unit": within_run_unit,
                "within_run_label": _interval_label(lower, upper),
            }
        )
    counts = Counter(str(row["within_run_label"]) for row in run_rows)
    return {
        "gate": gate,
        "run_rows": run_rows,
        "label_counts": {
            "positive": counts.get("positive", 0),
            "negative": counts.get("negative", 0),
            "inconclusive": counts.get("inconclusive", 0),
        },
        "between_run": _between_run_statistics(values),
    }


def build_five_seed_summary(
    bootstrap_pack: dict[str, Any],
    composition_pack: dict[str, Any],
    matched_pack: dict[str, Any],
    *,
    seed_names: tuple[str, ...] = FIVE_RUN_SEEDS,
) -> dict[str, Any]:
    """Combine G5/G6 with the legacy-key target-stratified S7 sensitivity."""

    _validate_seed_names(seed_names)
    for name, pack in (
        ("bootstrap", bootstrap_pack),
        ("composition", composition_pack),
        ("matched", matched_pack),
    ):
        pack_seeds = tuple(str(seed) for seed in pack.get("seed_names", []))
        if pack_seeds != seed_names:
            raise ValueError(f"{name} pack seed_names mismatch: {pack_seeds}")

    g5_rows = [row for row in bootstrap_pack["ci_rows"] if row["layer"] == LOCKED_LAYER]
    g6_rows = [
        row
        for row in composition_pack["rows"]
        if row["comparison"] == "composition_residual_vs_action_value_composition"
    ]
    g7_rows = list(matched_pack["rows"])
    gates = {
        "G5_action_value_residual": _summarize_gate(
            gate="G5_action_value_residual",
            rows=g5_rows,
            seed_names=seed_names,
            point_key="mean_delta",
            lower_key="lower_delta",
            upper_key="upper_delta",
            within_run_unit="hand_id",
        ),
        "G6_composition_residual": _summarize_gate(
            gate="G6_composition_residual",
            rows=g6_rows,
            seed_names=seed_names,
            point_key="mean_delta",
            lower_key="lower_delta",
            upper_key="upper_delta",
            within_run_unit="hand_id",
        ),
        "G7_matched_composition": _summarize_gate(
            gate="G7_matched_composition",
            rows=g7_rows,
            seed_names=seed_names,
            point_key="bucket_bootstrap_mean_delta",
            lower_key="bucket_bootstrap_lower_delta",
            upper_key="bucket_bootstrap_upper_delta",
            within_run_unit="matched_composition_bucket",
        ),
    }

    g6 = gates["G6_composition_residual"]
    g7 = gates["G7_matched_composition"]
    robust = (
        g6["label_counts"]["negative"] == 5
        and g7["label_counts"]["negative"] == 5
        and float(g6["between_run"]["upper_delta"]) < 0.0
        and float(g7["between_run"]["upper_delta"]) < 0.0
    )
    payload = {
        "created_by": "camera_ready_snapshot",
        "version": "k.8.0",
        "seed_names": list(seed_names),
        "seed_count": len(seed_names),
        "locked_layer": LOCKED_LAYER,
        "training_run_unit": "independent_end_to_end_run_model_data_split",
        "within_run_intervals": "hand_cluster_for_G5_G6_and_matched_bucket_for_G7",
        "between_run_interval": "equal_weight_two_sided_student_t_df4",
        "gates": gates,
        "final_diagnostic_label": ROBUST_SHORTCUT_LABEL if robust else MIXED_LABEL,
        "allowed_claim": (
            "five-run composition-shortcut robustness supported"
            if robust
            else "five-run robustness is mixed or inconclusive; inspect every run"
        ),
        "forbidden_stronger_claims": [
            "ordered poker belief tracking",
            "causal hidden-state mechanism",
            "decision-level pooled confidence as training-run uncertainty",
        ],
    }
    payload["checksum"] = checksum_payload(payload)
    return payload


def _required_artifact_paths(commit_root: Path, seed_names: tuple[str, ...]) -> list[Path]:
    paths = []
    for seed in seed_names:
        paths.extend(
            [
                commit_root / "configs" / "train" / f"limit_ar_action_value_only_{seed}_50k_4layer.json",
                commit_root / "outputs" / f"limit_ar_action_value_only_{seed}_50k_4layer" / "metrics.json",
                commit_root / "outputs" / f"limit_ar_action_value_only_{seed}_50k_4layer" / "model.pt",
                commit_root / "outputs" / f"limit_ar_action_value_only_{seed}_50k_4layer" / "train_log.jsonl",
            ]
        )
        for suffix in PROBE_SUFFIXES:
            probe_name = f"limit_ar_action_value_only_{seed}_4layer_range_{suffix}_50k"
            paths.extend(
                [
                    commit_root / "configs" / "probe" / f"{probe_name}.json",
                    commit_root / "outputs" / "probes" / probe_name / "probe_metrics.json",
                    commit_root / "outputs" / "probes" / probe_name / "probe_table.jsonl",
                    commit_root / "outputs" / "probes" / probe_name / "probe_correctness.jsonl",
                ]
            )
    return paths


def _validate_frozen_configs(commit_root: Path, seed_names: tuple[str, ...]) -> None:
    seen_random_seeds: set[int] = set()
    for seed in seed_names:
        train_path = commit_root / "configs" / "train" / f"limit_ar_action_value_only_{seed}_50k_4layer.json"
        train = _read_json(train_path)
        for key, expected in EXPECTED_TRAIN_SEEDS[seed].items():
            if int(train[key]) != expected:
                raise ValueError(f"{seed} train {key} changed: {train[key]} != {expected}")
            if expected in seen_random_seeds:
                raise ValueError(f"duplicate frozen random seed {expected}")
            seen_random_seeds.add(expected)
        invariant = {
            "hand_count": 50000,
            "epochs": 8,
            "batch_size": 512,
            "hidden_size": 96,
            "num_heads": 4,
            "num_layers": 4,
            "opponent_hand_loss_weight": 0.0,
            "policy_loss_weight": 1.0,
            "value_loss_weight": 0.3,
        }
        for key, expected in invariant.items():
            if train[key] != expected:
                raise ValueError(f"{seed} train invariant {key} changed: {train[key]} != {expected}")

        for suffix, (probe_seed, control_seed) in zip(PROBE_SUFFIXES, EXPECTED_PROBE_SEEDS[seed]):
            probe_path = (
                commit_root
                / "configs"
                / "probe"
                / f"limit_ar_action_value_only_{seed}_4layer_range_{suffix}_50k.json"
            )
            probe = _read_json(probe_path)
            if (int(probe["probe_seed"]), int(probe["control_seed"])) != (probe_seed, control_seed):
                raise ValueError(f"{seed}/{suffix} probe seeds changed")
            for value in (probe_seed, control_seed):
                if value in seen_random_seeds:
                    raise ValueError(f"duplicate frozen random seed {value}")
                seen_random_seeds.add(value)


def _artifact_manifest(commit_root: Path, paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"required five-run artifact missing: {path}")
        rows.append(
            {
                "path": str(path.relative_to(commit_root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# k.8.0 Five-Run Rebuttal Robustness",
        "",
        f"- final diagnostic label: `{payload['final_diagnostic_label']}`",
        f"- runs: {', '.join(payload['seed_names'])}",
        f"- locked layer: `{payload['locked_layer']}`",
    ]
    for gate_name, gate in payload["gates"].items():
        between = gate["between_run"]
        counts = gate["label_counts"]
        lines.extend(
            [
                "",
                f"## {gate_name}",
                "",
                "| run | mean delta | within-run 95% CI | label |",
                "|---|---:|---|---|",
            ]
        )
        for row in gate["run_rows"]:
            lines.append(
                f"| {row['seed']} | {row['mean_delta']:.6f} | "
                f"[{row['lower_delta']:.6f}, {row['upper_delta']:.6f}] | {row['within_run_label']} |"
            )
        lines.extend(
            [
                "",
                (
                    f"Across runs: mean `{between['mean_delta']:.6f}`, sample SD "
                    f"`{between['sample_standard_deviation']:.6f}`, 95% t interval "
                    f"`[{between['lower_delta']:.6f}, {between['upper_delta']:.6f}]`; "
                    f"positive/negative/inconclusive = "
                    f"{counts['positive']}/{counts['negative']}/{counts['inconclusive']}."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            payload["allowed_claim"],
            "",
            "Forbidden stronger claims:",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in payload["forbidden_stronger_claims"])
    return "\n".join(lines) + "\n"


def build_five_seed_rebuttal_pack(
    archive_root: Path,
    output_dir: Path,
    *,
    commit: str,
    source_revision: str,
    bootstrap_count: int = 10000,
    bootstrap_seed: int = 20260602,
    confidence_level: float = 0.95,
    seed_names: tuple[str, ...] = FIVE_RUN_SEEDS,
) -> dict[str, Any]:
    """Build and validate the complete five-run G5--G7 product."""

    _validate_seed_names(seed_names)
    commit_root = archive_root / commit
    _validate_frozen_configs(commit_root, seed_names)
    source_pack_dir = output_dir / "source_packs"
    locked_pack = build_no_range_locked_layer_pack(
        archive_root,
        source_pack_dir,
        commit,
        seed_names=seed_names,
    )
    bootstrap_pack = build_no_range_bootstrap_pack(
        archive_root,
        source_pack_dir,
        commit,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
        seed_names=seed_names,
    )
    composition_pack = build_no_range_composition_nuisance_pack(
        archive_root,
        source_pack_dir,
        commit,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
        seed_names=seed_names,
    )
    matched_pack = build_matched_composition_report(
        archive_root,
        source_pack_dir,
        commit=commit,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
        seed_names=seed_names,
    )
    payload = build_five_seed_summary(
        bootstrap_pack,
        composition_pack,
        matched_pack,
        seed_names=seed_names,
    )
    payload.update(
        {
            "source_revision": source_revision,
            "run_archive": commit,
            "bootstrap_count": bootstrap_count,
            "bootstrap_seed": bootstrap_seed,
            "confidence_level": confidence_level,
            "training_rows": locked_pack["seed_rows"],
        }
    )
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})

    required_paths = _required_artifact_paths(commit_root, seed_names)
    artifact_rows = _artifact_manifest(commit_root, required_paths)
    source_pack_rows = _artifact_manifest(
        output_dir,
        sorted(path for path in source_pack_dir.iterdir() if path.is_file()),
    )
    manifest = {
        "created_by": "camera_ready_snapshot",
        "version": "k.8.0",
        "status": "complete",
        "source_revision": source_revision,
        "run_archive": commit,
        "seed_names": list(seed_names),
        "physical_gpu_index": 3,
        "locked_layer": LOCKED_LAYER,
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "result_checksum": payload["checksum"],
        "required_artifacts": artifact_rows,
        "source_packs": source_pack_rows,
    }
    manifest["checksum"] = checksum_payload(manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "k.8.0_five_seed_rebuttal.json", payload)
    (output_dir / "k.8.0_five_seed_rebuttal.md").write_text(_markdown(payload), encoding="utf-8")
    write_json_artifact(output_dir / "k.8.0_five_seed_rebuttal_manifest.json", manifest)
    return payload

