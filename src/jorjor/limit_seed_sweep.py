from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import statistics
from typing import Any

from .artifacts import checksum_payload, write_json_artifact
from .config import load_limit_ar_train_config, load_limit_prior_baseline_config, load_limit_seed_sweep_config
from .limit_prior_baseline import run_limit_prior_baseline


@dataclass(frozen=True)
class LimitSeedSweepResult:
    """Result summary for a Limit AR-vs-prior seed sweep."""

    completed_count: int
    missing_count: int
    checksum: str
    artifact_dir: Path


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object: {path}")
    return raw


def _metric_triplet(record: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Extract top-k metrics with optional key prefix."""

    return {
        "top1": float(record[f"{prefix}accuracy"]),
        "top5": float(record[f"{prefix}top5_accuracy"]),
        "top10": float(record[f"{prefix}top10_accuracy"]),
    }


def _delta(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    """Compute paired top-k deltas."""

    return {key: left[key] - right[key] for key in ("top1", "top5", "top10")}


def _mean_std(values: list[float]) -> dict[str, float]:
    """Return sample mean and sample std, using zero std for one row."""

    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": statistics.fmean(values), "std": statistics.stdev(values)}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate completed sweep rows into paper-table statistics."""

    top1_deltas = [float(row["delta_vs_best_prior"]["top1"]) for row in rows]
    top5_deltas = [float(row["delta_vs_best_prior"]["top5"]) for row in rows]
    top10_deltas = [float(row["delta_vs_best_prior"]["top10"]) for row in rows]
    ar_top10 = [float(row["ar"]["top10"]) for row in rows]
    best_prior_top10 = [float(row["best_prior"]["top10"]) for row in rows]
    top1 = _mean_std(top1_deltas)
    top5 = _mean_std(top5_deltas)
    top10 = _mean_std(top10_deltas)
    ar_top10_stats = _mean_std(ar_top10)
    best_prior_top10_stats = _mean_std(best_prior_top10)
    return {
        "completed_count": len(rows),
        "top1_delta_mean": top1["mean"],
        "top1_delta_std": top1["std"],
        "top5_delta_mean": top5["mean"],
        "top5_delta_std": top5["std"],
        "top10_delta_mean": top10["mean"],
        "top10_delta_std": top10["std"],
        "ar_top10_mean": ar_top10_stats["mean"],
        "ar_top10_std": ar_top10_stats["std"],
        "best_prior_top10_mean": best_prior_top10_stats["mean"],
        "best_prior_top10_std": best_prior_top10_stats["std"],
    }


def _load_or_run_prior(prior_config_path: Path, run_missing_priors: bool) -> dict[str, Any] | None:
    """Load a prior report, optionally running the prior baseline if absent."""

    prior_config = load_limit_prior_baseline_config(prior_config_path)
    report_path = prior_config.artifact_dir / "prior_report.json"
    if report_path.exists():
        return _read_json(report_path)
    if not run_missing_priors:
        return None
    result = run_limit_prior_baseline(config_path=prior_config_path)
    return _read_json(result.artifact_dir / "prior_report.json")


def _row_from_reports(name: str, ar_config_path: Path, ar_report: dict[str, Any], prior_report: dict[str, Any]) -> dict[str, Any]:
    """Build one completed sweep row from AR and prior metrics."""

    ar_config = load_limit_ar_train_config(ar_config_path)
    ar_metrics = _metric_triplet(ar_report, prefix="opponent_hand_")
    priors = prior_report["baselines"]
    if not isinstance(priors, dict) or not priors:
        raise ValueError("prior report baselines must be a non-empty object")
    prior_metrics = {
        prior_name: {
            "top1": float(record["top1_accuracy"]),
            "top5": float(record["top5_accuracy"]),
            "top10": float(record["top10_accuracy"]),
        }
        for prior_name, record in priors.items()
    }
    best_prior_name, best_prior = max(prior_metrics.items(), key=lambda item: item[1]["top10"])
    action_round_position = prior_metrics.get("action_round_position")
    if action_round_position is None:
        raise ValueError("prior report must include action_round_position")

    return {
        "name": name,
        "data_seed": ar_config.data_seed,
        "train_seed": ar_config.seed,
        "split_seed": ar_config.split_seed,
        "hand_count": int(ar_report["hand_count"]),
        "decision_point_count": int(ar_report["decision_point_count"]),
        "illegal_action_count": int(ar_report["illegal_action_count"]),
        "leakage_count": int(ar_report["leakage_count"]),
        "policy_accuracy": float(ar_report["policy_accuracy"]),
        "value_mse": float(ar_report["value_mse"]),
        "ar": ar_metrics,
        "best_prior_name": best_prior_name,
        "best_prior": best_prior,
        "action_round_position_prior": action_round_position,
        "delta_vs_best_prior": _delta(ar_metrics, best_prior),
        "delta_vs_action_round_position": _delta(ar_metrics, action_round_position),
    }


def run_limit_seed_sweep(config_path: Path, artifact_dir: Path | None = None) -> LimitSeedSweepResult:
    """Create a v0.4.13 seed-sweep report from completed AR and prior artifacts."""

    config = load_limit_seed_sweep_config(config_path)
    output_dir = artifact_dir if artifact_dir is not None else config.artifact_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for seed in config.seeds:
        ar_config = load_limit_ar_train_config(seed.ar_config)
        ar_report_path = ar_config.artifact_dir / "metrics.json"
        if not ar_report_path.exists():
            missing.append({"name": seed.name, "missing": str(ar_report_path)})
            continue
        prior_report = _load_or_run_prior(seed.prior_config, config.run_missing_priors)
        if prior_report is None:
            prior_config = load_limit_prior_baseline_config(seed.prior_config)
            missing.append({"name": seed.name, "missing": str(prior_config.artifact_dir / "prior_report.json")})
            continue

        ar_report = _read_json(ar_report_path)
        rows.append(_row_from_reports(seed.name, seed.ar_config, ar_report, prior_report))

    record = {
        "version": config.run_version,
        "completed_count": len(rows),
        "missing_count": len(missing),
        "rows": rows,
        "missing": missing,
        "aggregate": _aggregate(rows),
    }
    checksum = checksum_payload(record)
    write_json_artifact(output_dir / "sweep_report.json", {**record, "checksum": checksum})

    return LimitSeedSweepResult(
        completed_count=len(rows),
        missing_count=len(missing),
        checksum=checksum,
        artifact_dir=output_dir,
    )
