from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

from jorjor.artifacts import write_json_artifact
from jorjor.config import load_limit_ar_train_config
from jorjor.data.split import SplitFractions, apply_split, build_split_manifest
from jorjor.train.limit_tiny import _load_points


REALIZED_LABEL_CONTROLS_PASSED = "realized_label_controls_passed"
# Backward-compatible symbol for callers of the pre-posterior audit API. The
# value is intentionally claim-safe: realized-label gates alone do not establish
# posterior fidelity or integrated belief-like support.
BELIEF_LIKE_RESIDUAL_SUPPORTED = REALIZED_LABEL_CONTROLS_PASSED
COMPOSITION_SHORTCUT_SUPPORTED = "composition_shortcut_supported"
INCONCLUSIVE = "inconclusive"
TARGET_STRATIFIED_RESIDUAL_SUPPORTED = "target_stratified_residual_supported"
TARGET_STRATIFIED_COMPOSITION_BOUND_SUPPORTED = "target_stratified_composition_bound_supported"
DEFAULT_BOOTSTRAP_COUNT = 10000
DEFAULT_BOOTSTRAP_SEED = 20260602
RANGE_SUPERVISED_STRICT_FILENAMES = (
    "limit_ar_range_supervised_positive_control_pack.json",
    "range_supervised_positive_control_pack.json",
)
PROBE_CAPACITY_S6_S7_FILENAMES = (
    "probe_capacity_s6_s7_report.json",
)


@dataclass(frozen=True)
class PackSpec:
    """Location and role of one diagnostic evidence pack."""

    key: str
    commit: str
    relative_path: str
    required: bool


@dataclass(frozen=True)
class CorrectnessSpec:
    """One correctness artifact used by matched-composition diagnostics."""

    seed: str
    residual_artifact: str
    prior_artifact: str
    train_config: str


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required diagnostic pack is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"diagnostic pack must contain a JSON object: {path}")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"required JSONL artifact is missing: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(row)
    return rows


def _optional_pack(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"missing artifact: {path}"
    return _read_json_object(path), None


def _optional_named_pack(
    key: str,
    roots: tuple[Path | None, ...],
    filenames: tuple[str, ...],
) -> tuple[dict[str, Any] | None, Path, str | None]:
    """Find one optional named extension pack across candidate roots."""

    checked_paths: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if root is None:
            continue
        for filename in filenames:
            path = root / filename
            path_key = str(path)
            if path_key in seen:
                continue
            seen.add(path_key)
            checked_paths.append(path)
            if path.exists():
                return _read_json_object(path), path, None

    fallback_path = checked_paths[0] if checked_paths else Path(f"<missing:{key}>")
    checked = ", ".join(str(path) for path in checked_paths)
    return None, fallback_path, f"missing optional artifact {key}: {checked}"


def _float_or_none(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None


def _status_from_composition_pack(pack: dict[str, Any] | None) -> tuple[str, list[str]]:
    if pack is None:
        return INCONCLUSIVE, ["composition nuisance pack is missing"]

    residual_lower = _float_or_none(pack, "composition_residual_vs_action_value_composition_min_lower_delta")
    residual_vs_action_lower = _float_or_none(pack, "composition_residual_vs_action_value_min_lower_delta")
    composition_lower = _float_or_none(pack, "action_value_composition_min_lower_delta")
    if residual_lower is None:
        return INCONCLUSIVE, ["composition residual lower delta is missing"]

    notes: list[str] = []
    if residual_lower > 0.0:
        notes.append("residual hidden states beat explicit action/value+composition nuisance with positive CI")
        return REALIZED_LABEL_CONTROLS_PASSED, notes

    if composition_lower is not None and composition_lower > 0.0:
        notes.append("explicit action/value+composition nuisance has positive CI")
    if residual_vs_action_lower is not None and residual_vs_action_lower < 0.0:
        notes.append("composition-residual hidden states are below the action/value nuisance baseline")
    notes.append("composition-residual hidden states do not beat action/value+composition nuisance")
    return COMPOSITION_SHORTCUT_SUPPORTED, notes


def _control_ladder_status(
    fresh50_pack: dict[str, Any] | None,
    history_composition_pack: dict[str, Any] | None,
    composition_pack: dict[str, Any] | None,
) -> dict[str, Any]:
    """Summarize the no-range control ladder as paper-facing statuses."""

    label_min = _float_or_none(fresh50_pack or {}, "label_control_delta_min")
    history_lower = _float_or_none(fresh50_pack or {}, "history_delta_ci_min_lower_delta")
    count_preserving_lower = _float_or_none(history_composition_pack or {}, "count_preserving_min_lower_delta")
    presence_last_lower = _float_or_none(history_composition_pack or {}, "presence_last_action_min_lower_delta")
    composition_status, composition_notes = _status_from_composition_pack(composition_pack)

    return {
        "naive_probe_status": "positive_intermediate_layer_signal",
        "label_control_status": "passes" if label_min is not None and label_min > 0.0 else INCONCLUSIVE,
        "history_control_status": "passes" if history_lower is not None and history_lower > 0.0 else INCONCLUSIVE,
        "order_control_status": (
            "fails_strict_gate"
            if count_preserving_lower is not None and count_preserving_lower <= 0.0
            else "passes"
            if count_preserving_lower is not None
            else INCONCLUSIVE
        ),
        "presence_last_action_status": (
            "passes" if presence_last_lower is not None and presence_last_lower > 0.0 else INCONCLUSIVE
        ),
        "composition_nuisance_status": composition_status,
        "final_diagnostic_label": composition_status,
        "claim_implication": _claim_implication(composition_status),
        "notes": composition_notes,
        "key_metrics": {
            "label_control_delta_min": label_min,
            "history_delta_ci_min_lower_delta": history_lower,
            "count_preserving_min_lower_delta": count_preserving_lower,
            "presence_last_action_min_lower_delta": presence_last_lower,
            "composition_only_min_lower_delta": _float_or_none(composition_pack or {}, "composition_only_min_lower_delta"),
            "action_value_composition_min_lower_delta": _float_or_none(
                composition_pack or {}, "action_value_composition_min_lower_delta"
            ),
            "composition_residual_vs_action_value_min_lower_delta": _float_or_none(
                composition_pack or {}, "composition_residual_vs_action_value_min_lower_delta"
            ),
            "composition_residual_vs_action_value_composition_min_lower_delta": _float_or_none(
                composition_pack or {}, "composition_residual_vs_action_value_composition_min_lower_delta"
            ),
        },
    }


def _claim_implication(status: str) -> str:
    if status == REALIZED_LABEL_CONTROLS_PASSED:
        return (
            "No-range hidden states clear the declared realized-label residual controls. "
            "Independent posterior-fit and predeclared pairwise-posterior fidelity gates remain required for belief-like support."
        )
    if status == COMPOSITION_SHORTCUT_SUPPORTED:
        return "No-range probe signal should be written as composition-shortcut evidence, not ordered belief tracking."
    return "Required controls are missing or unresolved; do not make a belief-tracking claim."


def _source_entry(key: str, path: Path, pack: dict[str, Any] | None, gap: str | None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "key": key,
        "path": str(path),
        "present": pack is not None,
    }
    if gap:
        entry["gap"] = gap
    if pack is not None:
        for field in ("version", "commit", "commit_50k", "locked_layer", "bootstrap_count", "bootstrap_seed"):
            if field in pack:
                entry[field] = pack[field]
        if "rows" in pack and isinstance(pack["rows"], list):
            entry["row_count"] = len(pack["rows"])
    return entry


def _markdown_report(payload: dict[str, Any]) -> str:
    case = payload["cases"][0]
    metrics = case["key_metrics"]
    lines = [
        "# Belief vs Shortcut Diagnostic Report",
        "",
        "## Summary",
        "",
        "| case | final diagnostic label | claim implication |",
        "|---|---|---|",
        f"| {case['case']} | {case['final_diagnostic_label']} | {case['claim_implication']} |",
        "",
        "## Control Ladder",
        "",
        "| control | status |",
        "|---|---|",
        f"| naive probe | {case['naive_probe_status']} |",
        f"| label permutation | {case['label_control_status']} |",
        f"| history destruction | {case['history_control_status']} |",
        f"| count-preserving order control | {case['order_control_status']} |",
        f"| presence/last-action control | {case['presence_last_action_status']} |",
        f"| composition nuisance | {case['composition_nuisance_status']} |",
        "",
        "## Key Metrics",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in metrics.items():
        text = "missing" if value is None else f"{float(value):.6f}"
        lines.append(f"| {key} | {text} |")
    lines.extend(
        [
            "",
            "## Sources",
            "",
            "| key | present | path |",
            "|---|---|---|",
        ]
    )
    for source in payload["sources"]:
        lines.append(f"| {source['key']} | {source['present']} | {source['path']} |")
    if payload["gaps"]:
        lines.extend(["", "## Gaps", ""])
        for gap in payload["gaps"]:
            lines.append(f"- {gap}")
    return "\n".join(lines) + "\n"


def _action_events(point) -> list[str]:
    return [
        str(event["action"])
        for event in point.visible_state["betting_history"]
        if event.get("type") == "action"
    ]


def _composition_bucket(point) -> str:
    actions = _action_events(point)
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


def _test_point_buckets(train_config_path: Path) -> dict[str, str]:
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
        f"{point.hand_id}:{point.step_index}": _composition_bucket(point)
        for point in splits.test
    }


def _correctness_by_point(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    selected = {}
    for row in rows:
        if row.get("control") != "none":
            continue
        point_id = row.get("point_id")
        if not isinstance(point_id, str):
            raise ValueError(f"correctness row is missing point_id: {path}")
        if point_id in selected:
            raise ValueError(f"duplicate point_id in correctness artifact: {path}:{point_id}")
        selected[point_id] = row
    if not selected:
        raise ValueError(f"correctness artifact has no real-control rows: {path}")
    return selected


def _matched_seed_row(archive_root: Path, commit: str, spec: CorrectnessSpec) -> dict[str, Any]:
    commit_root = archive_root / commit
    residual_rows = _correctness_by_point(commit_root / spec.residual_artifact)
    prior_rows = _correctness_by_point(commit_root / spec.prior_artifact)
    buckets = _test_point_buckets(commit_root / spec.train_config)

    common_ids = sorted(set(residual_rows) & set(prior_rows) & set(buckets))
    if not common_ids:
        raise ValueError(f"no shared point_ids for matched-composition seed: {spec.seed}")

    bucket_to_ids: dict[str, list[str]] = defaultdict(list)
    dropped = 0
    for point_id in common_ids:
        residual = residual_rows[point_id]
        prior = prior_rows[point_id]
        if residual.get("target_label") != prior.get("target_label"):
            raise ValueError(f"target label mismatch for point_id {point_id}")
        bucket_to_ids[buckets[point_id]].append(point_id)

    matched_ids = []
    bucket_mean_deltas = []
    singleton_bucket_count = 0
    for ids in bucket_to_ids.values():
        if len(ids) >= 2:
            matched_ids.extend(ids)
            bucket_mean_deltas.append(
                sum(
                    (1.0 if residual_rows[point_id]["top10_correct"] else 0.0)
                    - (1.0 if prior_rows[point_id]["top10_correct"] else 0.0)
                    for point_id in ids
                )
                / len(ids)
            )
        else:
            dropped += len(ids)
            singleton_bucket_count += 1

    if not matched_ids:
        raise ValueError(f"matched-composition buckets have no non-singleton support: {spec.seed}")

    residual_correct = sum(bool(residual_rows[point_id]["top10_correct"]) for point_id in matched_ids)
    prior_correct = sum(bool(prior_rows[point_id]["top10_correct"]) for point_id in matched_ids)
    matched_count = len(matched_ids)
    residual_top10 = residual_correct / matched_count
    prior_top10 = prior_correct / matched_count
    all_count = len(common_ids)
    return {
        "seed": spec.seed,
        "matched_point_count": matched_count,
        "all_joined_point_count": all_count,
        "dropped_point_count": dropped,
        "dropped_fraction": dropped / all_count,
        "bucket_count": len(bucket_to_ids),
        "matched_bucket_count": sum(1 for ids in bucket_to_ids.values() if len(ids) >= 2),
        "singleton_bucket_count": singleton_bucket_count,
        "residual_top10": residual_top10,
        "action_value_composition_top10": prior_top10,
        "delta_residual_minus_prior": residual_top10 - prior_top10,
        "bucket_mean_delta": sum(bucket_mean_deltas) / len(bucket_mean_deltas),
        "bucket_mean_deltas": bucket_mean_deltas,
        "residual_correct": residual_correct,
        "action_value_composition_correct": prior_correct,
    }


def _matched_composition_specs() -> tuple[CorrectnessSpec, ...]:
    specs = []
    for seed in ("seed_k", "seed_l", "seed_m"):
        prefix = f"outputs/probes/limit_ar_action_value_only_{seed}_4layer_range"
        specs.append(
            CorrectnessSpec(
                seed=seed,
                residual_artifact=(
                    f"{prefix}_residualize_action_value_history_composition_50k/probe_correctness.jsonl"
                ),
                prior_artifact=(
                    f"{prefix}_action_value_history_composition_only_50k/probe_correctness.jsonl"
                ),
                train_config=f"configs/train/limit_ar_action_value_only_{seed}_50k_4layer.json",
            )
        )
    return tuple(specs)


def _bootstrap_interval(
    values: list[float],
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    if bootstrap_count <= 0:
        raise ValueError("bootstrap_count must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    rng = random.Random(bootstrap_seed)
    means = []
    for _ in range(bootstrap_count):
        total = 0.0
        for _ in values:
            total += values[rng.randrange(len(values))]
        means.append(total / len(values))
    means.sort()
    lower_index = int(((1.0 - confidence_level) / 2.0) * (bootstrap_count - 1))
    upper_index = int((1.0 - (1.0 - confidence_level) / 2.0) * (bootstrap_count - 1))
    return {
        "bucket_bootstrap_mean_delta": sum(values) / len(values),
        "bucket_bootstrap_lower_delta": means[lower_index],
        "bucket_bootstrap_upper_delta": means[upper_index],
    }


def _matched_label(min_delta: float, min_lower_delta: float) -> str:
    if min_lower_delta > 0.0:
        return TARGET_STRATIFIED_RESIDUAL_SUPPORTED
    if min_delta < 0.0:
        return TARGET_STRATIFIED_COMPOSITION_BOUND_SUPPORTED
    return INCONCLUSIVE


def _matched_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Target-Stratified Matched-Composition Sensitivity Report",
        "",
        "This post-decision diagnostic fixes current action and terminal return as target-side strata. Neither is a model input. The result can corroborate a composition diagnosis but cannot independently establish a decision-time matched-composition gate.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| min residual-prior delta | {payload['min_delta_residual_minus_prior']:.6f} |",
        f"| min bucket-bootstrap lower delta | {payload['min_bucket_bootstrap_lower_delta']:.6f} |",
        f"| final diagnostic label | {payload['final_diagnostic_label']} |",
        "",
        "## Rows",
        "",
        "| seed | matched points | matched buckets | dropped fraction | residual top10 | action/value+composition top10 | delta | bucket lower | bucket upper |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {seed} | {matched_point_count} | {matched_bucket_count} | {dropped_fraction:.6f} | "
            "{residual_top10:.6f} | {action_value_composition_top10:.6f} | "
            "{delta_residual_minus_prior:.6f} | {bucket_bootstrap_lower_delta:.6f} | "
            "{bucket_bootstrap_upper_delta:.6f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def _bool_status(value: bool | None) -> str:
    if value is True:
        return "pass"
    if value is False:
        return "fail"
    return INCONCLUSIVE


def _lower_delta_status(value: float | None) -> str:
    if value is None:
        return INCONCLUSIVE
    return "pass" if value > 0.0 else "fail"


def _interval_delta_status(lower: float | None, upper: float | None) -> str:
    if lower is None:
        return INCONCLUSIVE
    if lower > 0.0:
        return "pass"
    if upper is None or upper < 0.0:
        return "fail"
    return INCONCLUSIVE


def _claim_matrix_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Claim Gate Matrix",
        "",
        "| case | probe positive | label artifact ruled out | history prior ruled out | action/value nuisance ruled out | composition nuisance ruled out | matched-composition residual supported | final diagnostic label |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {case} | {probe_positive} | {label_artifact_ruled_out} | {history_prior_ruled_out} | "
            "{action_value_nuisance_ruled_out} | {composition_nuisance_ruled_out} | "
            "{matched_composition_residual_supported} | {final_diagnostic_label} |".format(**row)
        )
    lines.extend(["", "## Sources", "", "| key | path | present |", "|---|---|---|"])
    for source in payload["sources"]:
        lines.append(f"| {source['key']} | {source['path']} | {source['present']} |")
    if payload["gaps"]:
        lines.extend(["", "## Gaps", ""])
        for gap in payload["gaps"]:
            lines.append(f"- {gap}")
    return "\n".join(lines) + "\n"


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"required text artifact is missing: {path}")
    return path.read_text(encoding="utf-8")


def _acml_manuscript_markdown(inspect_dir: Path) -> str:
    claim_gate = _read_text(inspect_dir / "claim_gate_matrix.md")
    belief_shortcut = _read_text(inspect_dir / "belief_shortcut_report.md")
    matched = _read_text(inspect_dir / "matched_composition_report.md")
    return "\n".join(
        [
            "# ACML Manuscript Pack",
            "",
            "## Title",
            "",
            "Belief or Shortcut? Diagnosing Hidden-State Representations in Autoregressive Models for Imperfect-Information Games",
            "",
            "## Abstract Skeleton",
            "",
            (
                "Autoregressive models trained on imperfect-information game histories can appear to encode "
                "hidden-state beliefs when linear probes recover opponent-range labels from frozen hidden states. "
                "We ask whether such probe evidence distinguishes belief-like tracking from shortcuts induced by "
                "observable action, value, position, and betting-composition features. We introduce JorJor-Inspect, "
                "a reproducible diagnostic suite combining probe baselines, nuisance residualization, label "
                "permutation, history destruction, composition-preserving controls, matched-composition evaluation, "
                "and confidence intervals. In Limit Hold'em case studies, naive and action/value-controlled probes "
                "produce positive range signals, but stricter composition and matched-composition controls show that "
                "the no-range-supervised signal is composition-shortcut supported. These results show that hidden-state "
                "belief claims in imperfect-information models require controls stronger than action/value nuisance alone."
            ),
            "",
            "## Contributions",
            "",
            "1. A control-first diagnostic protocol for hidden-state belief claims in imperfect-information AR models.",
            "2. `JorJor-Inspect`, a local artifact that converts archived probes into claim-gate reports.",
            "3. A Limit Hold'em case study showing that no-range probe positives can fail composition controls, with a target-stratified matched sensitivity as corroboration.",
            "4. A claim-gate matrix that separates probe-positive, shortcut-supported, and unsupported statements.",
            "",
            "## Main Claim",
            "",
            (
                "Naive range probes and action/value nuisance controls are not sufficient evidence for belief tracking. "
                "In the current no-range Limit Hold'em case, the evidence is best written as composition-shortcut supported."
            ),
            "",
            "## Claim Gate Matrix",
            "",
            claim_gate,
            "",
            "## Belief-vs-Shortcut Diagnostic Report",
            "",
            belief_shortcut,
            "",
            "## Target-Stratified Matched Sensitivity",
            "",
            matched,
            "",
            "## Figure Plan",
            "",
            "- Figure 1: control ladder from naive probe to composition nuisance.",
            "- Figure 2: claim-gate matrix heatmap.",
            "- Figure 3: target-stratified residual vs action/value+composition prior.",
            "",
            "## Main Tables",
            "",
            "- Table 1: claim-gate matrix.",
            "- Table 2: no-range diagnostic ladder.",
            "- Table 3: target-stratified matched sensitivity.",
            "- Table 4: range-supervised scoped positive probe evidence.",
            "",
            "## Reproducibility Commands",
            "",
            "```text",
            "python scripts/build_belief_shortcut_report.py --archive-root checkpoints/remote_archive/v100s --output-dir outputs/inspect --commit 2dfcb42",
            "python scripts/build_matched_composition_report.py --archive-root checkpoints/remote_archive/v100s --output-dir outputs/inspect --commit 2dfcb42",
            "python scripts/build_claim_gate_matrix.py --inspect-dir outputs/inspect --paper-table-dir checkpoints/tables --output-dir outputs/inspect",
            "python scripts/build_acml_manuscript_pack.py --inspect-dir outputs/inspect --output-dir outputs/inspect",
            "```",
            "",
            "## Must Not Claim",
            "",
            "- Do not claim no-range models learn ordered betting-history belief tracking.",
            "- Do not claim residual hidden states beat explicit action/value+composition priors.",
            "- Do not claim poker strength or deployable decision assistance.",
            "- Do not claim causal mechanisms from linear probes alone.",
            "",
        ]
    )


def _range_supervised_row(
    paper_pack: dict[str, Any] | None,
    strict_pack: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if strict_pack is not None:
        s5_lower = _float_or_none(strict_pack, "s5_min_lower_delta")
        s6_lower = _float_or_none(strict_pack, "s6_min_lower_delta")
        s7_lower = _float_or_none(strict_pack, "s7_min_lower_delta")
        final_label = strict_pack.get("final_diagnostic_label")
        if not isinstance(final_label, str):
            final_label = (
                REALIZED_LABEL_CONTROLS_PASSED
                if s6_lower is not None and s7_lower is not None and s6_lower > 0.0 and s7_lower > 0.0
                else COMPOSITION_SHORTCUT_SUPPORTED
                if s6_lower is not None and s6_lower <= 0.0
                else INCONCLUSIVE
            )
        return {
            "case": "range_supervised_ar",
            "probe_positive": _lower_delta_status(s5_lower),
            "label_artifact_ruled_out": "direct_range_supervision",
            "history_prior_ruled_out": "not_primary_gate",
            "action_value_nuisance_ruled_out": _lower_delta_status(s5_lower),
            "composition_nuisance_ruled_out": _lower_delta_status(s6_lower),
            "matched_composition_residual_supported": _lower_delta_status(s7_lower),
            "final_diagnostic_label": final_label,
        }

    if paper_pack is None:
        return {
            "case": "range_supervised_ar",
            "probe_positive": INCONCLUSIVE,
            "label_artifact_ruled_out": INCONCLUSIVE,
            "history_prior_ruled_out": INCONCLUSIVE,
            "action_value_nuisance_ruled_out": INCONCLUSIVE,
            "composition_nuisance_ruled_out": INCONCLUSIVE,
            "matched_composition_residual_supported": INCONCLUSIVE,
            "final_diagnostic_label": "missing_optional_evidence",
        }
    core = paper_pack.get("core_4layer")
    scale = paper_pack.get("scale_robustness")
    action_value_pass = None
    if isinstance(core, dict):
        action_value_pass = float(core.get("history_residual_delta_min", 0.0)) > 0.0
    scale_pass = None
    if isinstance(scale, dict):
        scale_pass = bool(scale.get("all_history_pass", False))
    return {
        "case": "range_supervised_ar",
        "probe_positive": _bool_status(action_value_pass),
        "label_artifact_ruled_out": "not_primary_gate",
        "history_prior_ruled_out": "not_primary_gate",
        "action_value_nuisance_ruled_out": _bool_status(action_value_pass),
        "composition_nuisance_ruled_out": "not_tested",
        "matched_composition_residual_supported": "not_tested",
        "final_diagnostic_label": (
            "probe_positive_action_value_controlled"
            if action_value_pass and scale_pass
            else INCONCLUSIVE
        ),
    }


def _mlp_capacity_matrix_row(capacity_pack: dict[str, Any]) -> dict[str, Any]:
    c1_lower = _float_or_none(capacity_pack, "c1_min_lower_delta")
    c2_lower = _float_or_none(capacity_pack, "c2_min_lower_delta")
    c2_upper = _float_or_none(capacity_pack, "c2_max_upper_delta")
    c3_lower = _float_or_none(capacity_pack, "c3_min_lower_delta")
    c3_upper = _float_or_none(capacity_pack, "c3_max_upper_delta")
    final_label = capacity_pack.get("final_verdict")
    return {
        "case": "action_value_only_no_range_ar_mlp_appendix",
        "probe_positive": _lower_delta_status(c1_lower),
        "label_artifact_ruled_out": "inherits_main_linear_gate",
        "history_prior_ruled_out": "inherits_main_linear_gate",
        "action_value_nuisance_ruled_out": _lower_delta_status(c1_lower),
        "composition_nuisance_ruled_out": _interval_delta_status(c2_lower, c2_upper),
        "matched_composition_residual_supported": _interval_delta_status(c3_lower, c3_upper),
        "final_diagnostic_label": final_label if isinstance(final_label, str) else INCONCLUSIVE,
    }


def _no_range_matrix_row(
    belief_report: dict[str, Any],
    matched_report: dict[str, Any] | None,
) -> dict[str, Any]:
    case = belief_report["cases"][0]
    matched_lower = None
    if matched_report is not None:
        matched_lower = _float_or_none(matched_report, "min_bucket_bootstrap_lower_delta")
    return {
        "case": "action_value_only_no_range_ar",
        "probe_positive": "pass",
        "label_artifact_ruled_out": case["label_control_status"],
        "history_prior_ruled_out": case["history_control_status"],
        "action_value_nuisance_ruled_out": "pass",
        "composition_nuisance_ruled_out": (
            "fail"
            if case["composition_nuisance_status"] == COMPOSITION_SHORTCUT_SUPPORTED
            else "pass"
            if case["composition_nuisance_status"] == REALIZED_LABEL_CONTROLS_PASSED
            else INCONCLUSIVE
        ),
        "matched_composition_residual_supported": (
            "fail"
            if matched_lower is not None and matched_lower <= 0.0
            else "pass"
            if matched_lower is not None
            else INCONCLUSIVE
        ),
        "final_diagnostic_label": case["final_diagnostic_label"],
    }


def build_belief_shortcut_report(
    archive_root: Path,
    output_dir: Path,
    *,
    composition_commit: str,
    fresh50_commit: str = "881534f",
    history_composition_commit: str = "1d9574f",
) -> dict[str, Any]:
    """Build a JorJor-Inspect report from archived no-range control packs."""

    specs = (
        PackSpec(
            key="fresh50_control",
            commit=fresh50_commit,
            relative_path="outputs/tables/limit_ar_no_range_fresh50_control_pack.json",
            required=False,
        ),
        PackSpec(
            key="history_composition",
            commit=history_composition_commit,
            relative_path="outputs/tables/limit_ar_no_range_history_composition_pack.json",
            required=False,
        ),
        PackSpec(
            key="composition_nuisance",
            commit=composition_commit,
            relative_path="outputs/tables/limit_ar_no_range_composition_nuisance_pack.json",
            required=True,
        ),
    )

    packs: dict[str, dict[str, Any] | None] = {}
    sources: list[dict[str, Any]] = []
    gaps: list[str] = []
    for spec in specs:
        path = archive_root / spec.commit / spec.relative_path
        if spec.required:
            pack = _read_json_object(path)
            gap = None
        else:
            pack, gap = _optional_pack(path)
        if gap:
            gaps.append(gap)
        packs[spec.key] = pack
        sources.append(_source_entry(spec.key, path, pack, gap))

    no_range_case = {
        "case": "action_value_only_no_range_ar",
        **_control_ladder_status(
            packs["fresh50_control"],
            packs["history_composition"],
            packs["composition_nuisance"],
        ),
    }
    payload = {
                "artifact": "JorJor-Inspect",
        "version": "1.1",
        "archive_root": str(archive_root),
        "cases": [no_range_case],
        "sources": sources,
        "gaps": gaps,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "belief_shortcut_report.json", payload)
    (output_dir / "belief_shortcut_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def build_matched_composition_report(
    archive_root: Path,
    output_dir: Path,
    *,
    commit: str,
    bootstrap_count: int = DEFAULT_BOOTSTRAP_COUNT,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Build the legacy-key, post-decision target-stratified sensitivity report."""

    rows = []
    for index, spec in enumerate(_matched_composition_specs()):
        row = _matched_seed_row(archive_root, commit, spec)
        interval = _bootstrap_interval(
            [float(value) for value in row.pop("bucket_mean_deltas")],
            bootstrap_count=bootstrap_count,
            bootstrap_seed=bootstrap_seed + index,
            confidence_level=confidence_level,
        )
        rows.append({**row, **interval})
    min_delta = min(float(row["delta_residual_minus_prior"]) for row in rows)
    min_lower_delta = min(float(row["bucket_bootstrap_lower_delta"]) for row in rows)
    payload = {
                "artifact": "JorJor-Inspect",
        "version": "1.2",
        "commit": commit,
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "analysis_role": "target_stratified_postdecision_sensitivity",
        "decision_time_gate": False,
        "target_side_strata": ["action_label", "terminal_return"],
        "claim_limit": "corroborates_composition_control_only",
        "bucket_key": [
            "acting_player",
            "round",
            "legal_actions",
            "action_label",
            "terminal_return",
            "history_length",
            "action_counts",
            "last_action",
        ],
        "rows": rows,
        "min_delta_residual_minus_prior": min_delta,
        "min_bucket_bootstrap_lower_delta": min_lower_delta,
        "final_diagnostic_label": _matched_label(min_delta, min_lower_delta),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "matched_composition_report.json", payload)
    (output_dir / "matched_composition_report.md").write_text(_matched_markdown(payload), encoding="utf-8")
    return payload


def build_claim_gate_matrix(
    inspect_dir: Path,
    paper_table_dir: Path,
    output_dir: Path,
    *,
    strict_table_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the ACML-facing diagnostic claim-gate matrix."""

    belief_path = inspect_dir / "belief_shortcut_report.json"
    matched_path = inspect_dir / "matched_composition_report.json"
    paper_path = paper_table_dir / "limit_ar_paper_evidence_pack.json"
    strict_roots = (strict_table_dir, inspect_dir, paper_table_dir)

    belief_report = _read_json_object(belief_path)
    matched_report, matched_gap = _optional_pack(matched_path)
    paper_pack, paper_gap = _optional_pack(paper_path)
    range_strict_pack, range_strict_path, range_strict_gap = _optional_named_pack(
        "range_supervised_strict_gate",
        strict_roots,
        RANGE_SUPERVISED_STRICT_FILENAMES,
    )
    capacity_pack, capacity_path, capacity_gap = _optional_named_pack(
        "probe_capacity_s6_s7",
        strict_roots,
        PROBE_CAPACITY_S6_S7_FILENAMES,
    )
    gaps = [gap for gap in (matched_gap, paper_gap, range_strict_gap, capacity_gap) if gap]
    sources = [
        {"key": "belief_shortcut_report", "path": str(belief_path), "present": True},
        {"key": "matched_composition_report", "path": str(matched_path), "present": matched_report is not None},
        {"key": "paper_evidence_pack", "path": str(paper_path), "present": paper_pack is not None},
        {
            "key": "range_supervised_strict_gate",
            "path": str(range_strict_path),
            "present": range_strict_pack is not None,
        },
        {
            "key": "probe_capacity_s6_s7",
            "path": str(capacity_path),
            "present": capacity_pack is not None,
        },
    ]
    rows = [
        _range_supervised_row(paper_pack, range_strict_pack),
        _no_range_matrix_row(belief_report, matched_report),
    ]
    if capacity_pack is not None:
        rows.append(_mlp_capacity_matrix_row(capacity_pack))
    payload = {
                "artifact": "JorJor-Inspect",
        "version": "1.19",
        "strict_table_dir": str(strict_table_dir) if strict_table_dir is not None else None,
        "rows": rows,
        "sources": sources,
        "gaps": gaps,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "claim_gate_matrix.json", payload)
    (output_dir / "claim_gate_matrix.md").write_text(_claim_matrix_markdown(payload), encoding="utf-8")
    return payload


def build_acml_manuscript_pack(inspect_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Build a paper-facing ACML manuscript skeleton from local diagnostic reports."""

    text = _acml_manuscript_markdown(inspect_dir)
    forbidden = (
        "no-range models learn ordered betting-history belief tracking",
        "residual hidden states beat explicit action/value+composition priors",
    )
    lower_text = text.lower()
    for phrase in forbidden:
        # The forbidden phrases may only appear in the explicit Must Not Claim section.
        if lower_text.count(phrase) > 1:
            raise ValueError(f"forbidden overclaim appears outside boundary section: {phrase}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "acml_manuscript_pack.md"
    output_path.write_text(text, encoding="utf-8")
    payload = {
                "artifact": "JorJor-Inspect",
        "version": "1.4",
        "output_path": str(output_path),
        "required_inputs": [
            str(inspect_dir / "claim_gate_matrix.md"),
            str(inspect_dir / "belief_shortcut_report.md"),
            str(inspect_dir / "matched_composition_report.md"),
        ],
    }
    write_json_artifact(output_dir / "acml_manuscript_pack.json", payload)
    return payload
