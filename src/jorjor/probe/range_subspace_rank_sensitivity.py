from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.config import load_limit_ar_train_config
from jorjor.models.ar import build_tiny_ar_model
from jorjor.models.tiny import require_torch
from jorjor.probe.limit_ar import _load_ar_checkpoint_with_vocab_compat
from jorjor.probe.range_subspace_ablation import (
    DEFAULT_SEEDS,
    INCONCLUSIVE,
    PROBE_NOT_SUPPORTED,
    RANGE_FOOTPRINT_NOT_SUPPORTED,
    RANGE_FOOTPRINT_SUPPORTED,
    _aggregate_rows,
    _basis_from_probe,
    _extract_split_features,
    _mean,
    _random_basis,
    _seed_label,
    _sha256_file,
    _summary,
    _train_range_probe,
    _topk_accuracy,
    remove_subspace,
)
from jorjor.tokenizer import LimitTokenizer
from jorjor.train.leduc_tiny import _resolve_device


RANK_SENSITIVITY_NOT_SUPPORTED = "range_subspace_rank_sensitivity_not_supported"
RANK_SENSITIVITY_SUPPORTED = "range_subspace_rank_sensitivity_supported"
RANK_SENSITIVE_INCONCLUSIVE = "rank_sensitive_inconclusive"
DEFAULT_RANK_GRID = (1, 2, 4, 8, 16, 32)


def build_range_subspace_rank_sensitivity_report(
    *,
    archive_root: Path,
    output_dir: Path,
    seed_names: tuple[str, ...] = DEFAULT_SEEDS,
    rank_grid: tuple[int, ...] = DEFAULT_RANK_GRID,
    train_config_dir: Path = Path("configs/train"),
    bootstrap_count: int = 1000,
    bootstrap_seed: int = 20260610,
    confidence_level: float = 0.95,
    device: str = "cpu",
    batch_size: int = 512,
    max_train_points: int | None = 30000,
    max_test_points: int | None = None,
    locked_layer: str = "sequence_hidden",
    probe_epochs: int = 20,
    probe_learning_rate: float = 0.01,
    random_subspace_count: int = 8,
    probe_top10_floor: float = 0.08,
) -> dict[str, Any]:
    """Build k.6.5 rank-sensitivity report for range-probe subspace ablation."""

    clean_rank_grid = _validate_inputs(
        seed_names=seed_names,
        rank_grid=rank_grid,
        bootstrap_count=bootstrap_count,
        batch_size=batch_size,
        random_subspace_count=random_subspace_count,
        locked_layer=locked_layer,
        probe_epochs=probe_epochs,
        probe_learning_rate=probe_learning_rate,
        probe_top10_floor=probe_top10_floor,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, seed_name in enumerate(seed_names):
        rows.append(
            _seed_rank_rows(
                seed_name=seed_name,
                train_config_path=train_config_dir / f"limit_ar_action_value_only_{seed_name}_50k_4layer.json",
                checkpoint_path=archive_root / "outputs" / f"limit_ar_action_value_only_{seed_name}_50k_4layer" / "model.pt",
                rank_grid=clean_rank_grid,
                bootstrap_count=bootstrap_count,
                bootstrap_seed=bootstrap_seed + index * 1000,
                confidence_level=confidence_level,
                device=device,
                batch_size=batch_size,
                max_train_points=max_train_points,
                max_test_points=max_test_points,
                locked_layer=locked_layer,
                probe_epochs=probe_epochs,
                probe_learning_rate=probe_learning_rate,
                random_subspace_count=random_subspace_count,
                probe_top10_floor=probe_top10_floor,
            )
        )
    aggregate = _aggregate_seed_rows(rows, bootstrap_count, bootstrap_seed + 10000, confidence_level)
    payload = {
        "version": "k.6.5",
        "report_type": "k.6.5_range_subspace_rank_sensitivity",
        "claim_scope": "rank_sensitivity_stress_not_causal_mediation",
        "archive_root": str(archive_root),
        "seed_names": list(seed_names),
        "rank_grid": list(clean_rank_grid),
        "train_config_dir": str(train_config_dir),
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "device": device,
        "batch_size": batch_size,
        "max_train_points": max_train_points,
        "max_test_points": max_test_points,
        "locked_layer": locked_layer,
        "probe_epochs": probe_epochs,
        "probe_learning_rate": probe_learning_rate,
        "random_subspace_count": random_subspace_count,
        "probe_top10_floor": probe_top10_floor,
        "rows": rows,
        "aggregate": aggregate,
        "final_diagnostic_label": _rank_sensitivity_label(aggregate),
        "allowed_paper_claim": (
            "Range-probe behavior-footprint conclusions can be reported as stable or rank-sensitive across "
            "the predeclared rank grid. This is a sensitivity stress test, not causal mediation."
        ),
        "forbidden_paper_claim": "Rank sensitivity proves the presence or absence of belief tracking or a causal circuit.",
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "k.6.5_range_subspace_rank_sensitivity_report.json", payload)
    (output_dir / "k.6.5_range_subspace_rank_sensitivity_report.md").write_text(_markdown(payload), encoding="utf-8")
    write_json_artifact(output_dir / "k.6.5_range_subspace_rank_sensitivity_manifest.json", _manifest(output_dir, payload))
    return payload


def _validate_inputs(
    *,
    seed_names: tuple[str, ...],
    rank_grid: tuple[int, ...],
    bootstrap_count: int,
    batch_size: int,
    random_subspace_count: int,
    locked_layer: str,
    probe_epochs: int,
    probe_learning_rate: float,
    probe_top10_floor: float,
) -> tuple[int, ...]:
    if not seed_names:
        raise ValueError("seed_names must not be empty")
    if not rank_grid:
        raise ValueError("rank_grid must not be empty")
    if any(rank <= 0 for rank in rank_grid):
        raise ValueError("rank_grid ranks must be positive")
    if len(set(rank_grid)) != len(rank_grid):
        raise ValueError("rank_grid must not contain duplicate ranks")
    if bootstrap_count <= 0:
        raise ValueError("bootstrap_count must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if random_subspace_count <= 0:
        raise ValueError("random_subspace_count must be positive")
    if locked_layer != "sequence_hidden":
        raise ValueError("k.6.5 rank sensitivity requires locked_layer=sequence_hidden")
    if probe_epochs <= 0:
        raise ValueError("probe_epochs must be positive")
    if probe_learning_rate <= 0.0:
        raise ValueError("probe_learning_rate must be positive")
    if probe_top10_floor < 0.0:
        raise ValueError("probe_top10_floor must be non-negative")
    return tuple(sorted(rank_grid))


def _seed_rank_rows(
    *,
    seed_name: str,
    train_config_path: Path,
    checkpoint_path: Path,
    rank_grid: tuple[int, ...],
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
    device: str,
    batch_size: int,
    max_train_points: int | None,
    max_test_points: int | None,
    locked_layer: str,
    probe_epochs: int,
    probe_learning_rate: float,
    random_subspace_count: int,
    probe_top10_floor: float,
) -> dict[str, Any]:
    torch, nn = require_torch()
    train_config = load_limit_ar_train_config(train_config_path)
    resolved_device = _resolve_device(device, torch)
    tokenizer = LimitTokenizer()
    model = build_tiny_ar_model(
        vocab_size=tokenizer.vocab_size,
        action_size=tokenizer.action_size,
        opponent_card_size=tokenizer.opponent_card_size,
        hidden_size=train_config.hidden_size,
        num_heads=train_config.num_heads,
        num_layers=train_config.num_layers,
    ).to(resolved_device)
    load_info = _load_ar_checkpoint_with_vocab_compat(model, checkpoint_path, torch, resolved_device)
    features = _extract_split_features(
        train_config=train_config,
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        device=resolved_device,
        batch_size=batch_size,
        max_train_points=max_train_points,
        max_test_points=max_test_points,
        locked_layer=locked_layer,
    )
    probe = _train_probe_once(
        features=features,
        tokenizer=tokenizer,
        torch=torch,
        nn=nn,
        device=resolved_device,
        probe_epochs=probe_epochs,
        probe_learning_rate=probe_learning_rate,
        batch_size=batch_size,
        bootstrap_seed=bootstrap_seed,
    )
    with torch.no_grad():
        probe_logits = probe(features.test_hidden)
        probe_top10 = _topk_accuracy(probe_logits, features.test_labels, torch, 10)
    max_rank = max(rank_grid)
    full_basis, full_rank, singular_values = _basis_from_probe(probe, torch, max_rank)
    rank_rows = []
    for rank_index, requested_rank in enumerate(rank_grid):
        effective_rank = min(int(requested_rank), int(full_basis.shape[0]))
        basis = full_basis[:effective_rank].contiguous()
        range_metrics = _behavior_movement_fast(
            model=model,
            original_hidden=features.test_sequence_hidden,
            ablated_hidden=remove_subspace(features.test_sequence_hidden, basis),
            original_policy_logits=features.test_policy_logits,
            original_value=features.test_value,
            torch=torch,
        )
        random_rows = []
        for random_index in range(random_subspace_count):
            random_basis = _random_basis(
                rank=effective_rank,
                hidden_size=features.test_sequence_hidden.shape[1],
                torch=torch,
                device=resolved_device,
                seed=bootstrap_seed + 100000 + rank_index * 1000 + random_index,
            )
            random_rows.append(
                _behavior_movement_fast(
                    model=model,
                    original_hidden=features.test_sequence_hidden,
                    ablated_hidden=remove_subspace(features.test_sequence_hidden, random_basis),
                    original_policy_logits=features.test_policy_logits,
                    original_value=features.test_value,
                    torch=torch,
                )
            )
        summary = _summary(
            range_metrics=range_metrics,
            random_rows=random_rows,
            bootstrap_count=bootstrap_count,
            bootstrap_seed=bootstrap_seed + rank_index * 10,
            confidence_level=confidence_level,
        )
        rank_rows.append(
            {
                "requested_rank": int(requested_rank),
                "effective_rank": int(effective_rank),
                "probe_top10_accuracy": probe_top10,
                "probe_top10_floor": probe_top10_floor,
                "summary": summary,
                "final_diagnostic_label": _seed_label(summary, probe_top10, probe_top10_floor),
            }
        )
    return {
        "seed": seed_name,
        "train_config_path": str(train_config_path),
        "train_config_sha256": _sha256_file(train_config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_load_info": load_info,
        "locked_layer": locked_layer,
        "train_points": int(features.train_hidden.shape[0]),
        "test_points": int(features.test_hidden.shape[0]),
        "probe_top10_accuracy": probe_top10,
        "probe_full_row_rank": full_rank,
        "probe_top_singular_values": singular_values,
        "rank_rows": rank_rows,
        "label_counts": _count_labels(row["final_diagnostic_label"] for row in rank_rows),
        "final_diagnostic_label": _seed_rank_label(rank_rows),
    }


def _behavior_movement_fast(*, model, original_hidden, ablated_hidden, original_policy_logits, original_value, torch) -> dict[str, list[float]]:
    """Vectorized policy/value movement for repeated rank-sweep ablations."""

    with torch.no_grad():
        edited_policy_logits = model.policy_head(ablated_hidden)
        edited_value = model.value_head(ablated_hidden).squeeze(-1)
        left = torch.nn.functional.softmax(original_policy_logits, dim=-1).clamp_min(1e-12)
        right = torch.nn.functional.softmax(edited_policy_logits, dim=-1).clamp_min(1e-12)
        middle = ((left + right) / 2.0).clamp_min(1e-12)
        log2 = float(torch.log(torch.tensor(2.0, device=left.device)).item())
        left_kl = (left * (torch.log(left / middle) / log2)).sum(dim=-1)
        right_kl = (right * (torch.log(right / middle) / log2)).sum(dim=-1)
        policy_js = 0.5 * left_kl + 0.5 * right_kl
        value_abs_delta = torch.abs(original_value - edited_value)
        policy_changed = torch.argmax(original_policy_logits, dim=-1) != torch.argmax(edited_policy_logits, dim=-1)
    return {
        "policy_js": [float(value) for value in policy_js.detach().cpu().tolist()],
        "value_abs_delta": [float(value) for value in value_abs_delta.detach().cpu().tolist()],
        "policy_argmax_changed": [float(value) for value in policy_changed.float().detach().cpu().tolist()],
    }


def _train_probe_once(
    *,
    features,
    tokenizer: LimitTokenizer,
    torch,
    nn,
    device: str,
    probe_epochs: int,
    probe_learning_rate: float,
    batch_size: int,
    bootstrap_seed: int,
):
    return _train_range_probe(
        train_x=features.train_hidden,
        train_y=features.train_labels,
        input_size=features.train_hidden.shape[1],
        output_size=tokenizer.opponent_card_size,
        torch=torch,
        nn=nn,
        device=device,
        epochs=probe_epochs,
        learning_rate=probe_learning_rate,
        batch_size=batch_size,
        seed=bootstrap_seed,
    )


def _seed_rank_label(rank_rows: list[dict[str, Any]]) -> str:
    labels = {row["final_diagnostic_label"] for row in rank_rows}
    if labels == {RANGE_FOOTPRINT_SUPPORTED}:
        return RANK_SENSITIVITY_SUPPORTED
    if labels == {RANGE_FOOTPRINT_NOT_SUPPORTED}:
        return RANK_SENSITIVITY_NOT_SUPPORTED
    if labels == {PROBE_NOT_SUPPORTED}:
        return PROBE_NOT_SUPPORTED
    return RANK_SENSITIVE_INCONCLUSIVE


def _aggregate_seed_rows(
    rows: list[dict[str, Any]],
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    rank_values = sorted({rank_row["requested_rank"] for row in rows for rank_row in row["rank_rows"]})
    ranks = []
    for rank in rank_values:
        rank_specific_rows = []
        for row in rows:
            matching = [rank_row for rank_row in row["rank_rows"] if rank_row["requested_rank"] == rank]
            if not matching:
                continue
            rank_row = matching[0]
            rank_specific_rows.append(
                {
                    "probe_top10_accuracy": row["probe_top10_accuracy"],
                    "summary": rank_row["summary"],
                    "final_diagnostic_label": rank_row["final_diagnostic_label"],
                }
            )
        rank_aggregate = _aggregate_rows(
            rank_specific_rows,
            bootstrap_count,
            bootstrap_seed + _stable_int(str(rank)) % 1000,
            confidence_level,
        )
        ranks.append(
            {
                "rank": rank,
                "seed_count": len(rank_specific_rows),
                "label_counts": rank_aggregate["label_counts"],
                "metrics": rank_aggregate["metrics"],
                "final_diagnostic_label": _rank_label(rank_aggregate),
            }
        )
    return {
        "seed_count": len(rows),
        "rank_count": len(ranks),
        "seed_label_counts": _count_labels(row["final_diagnostic_label"] for row in rows),
        "rank_label_counts": _count_labels(row["final_diagnostic_label"] for row in ranks),
        "ranks": ranks,
        "mean_policy_delta_across_ranks": _mean(
            [
                rank["metrics"]["range_minus_random_policy_js_delta"]["mean"]
                for rank in ranks
            ]
        ),
        "mean_value_delta_across_ranks": _mean(
            [
                rank["metrics"]["range_minus_random_value_delta"]["mean"]
                for rank in ranks
            ]
        ),
    }


def _rank_label(rank_aggregate: dict[str, Any]) -> str:
    labels = set(rank_aggregate["label_counts"])
    if labels == {RANGE_FOOTPRINT_SUPPORTED}:
        return RANGE_FOOTPRINT_SUPPORTED
    if labels == {RANGE_FOOTPRINT_NOT_SUPPORTED}:
        return RANGE_FOOTPRINT_NOT_SUPPORTED
    if labels == {PROBE_NOT_SUPPORTED}:
        return PROBE_NOT_SUPPORTED
    return INCONCLUSIVE


def _rank_sensitivity_label(aggregate: dict[str, Any]) -> str:
    labels = {rank["final_diagnostic_label"] for rank in aggregate["ranks"]}
    if labels == {RANGE_FOOTPRINT_SUPPORTED}:
        return RANK_SENSITIVITY_SUPPORTED
    if labels == {RANGE_FOOTPRINT_NOT_SUPPORTED}:
        return RANK_SENSITIVITY_NOT_SUPPORTED
    if labels == {PROBE_NOT_SUPPORTED}:
        return PROBE_NOT_SUPPORTED
    return RANK_SENSITIVE_INCONCLUSIVE


def _manifest(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": payload["version"],
        "report_type": "k.6.5_range_subspace_rank_sensitivity_manifest",
        "claim_scope": payload["claim_scope"],
        "output_dir": str(output_dir),
        "expected_outputs": [
            "k.6.5_range_subspace_rank_sensitivity_report.json",
            "k.6.5_range_subspace_rank_sensitivity_report.md",
            "k.6.5_range_subspace_rank_sensitivity_manifest.json",
        ],
        "report_checksum": payload["checksum"],
        "final_diagnostic_label": payload["final_diagnostic_label"],
        "one_command_reproduction": (
            ".venv/bin/python scripts/build_range_subspace_rank_sensitivity_report.py "
            f"--archive-root {payload['archive_root']} "
            f"--output-dir {output_dir} "
            f"--bootstrap-count {payload['bootstrap_count']} "
            f"--bootstrap-seed {payload['bootstrap_seed']} "
            f"--batch-size {payload['batch_size']} "
            f"--locked-layer {payload['locked_layer']} "
            f"--probe-epochs {payload['probe_epochs']} "
            f"--probe-learning-rate {payload['probe_learning_rate']} "
            f"--random-subspace-count {payload['random_subspace_count']} "
            f"--probe-top10-floor {payload['probe_top10_floor']} "
            + ("--max-train-points {value} ".format(value=payload["max_train_points"]) if payload["max_train_points"] is not None else "")
            + ("--max-test-points {value} ".format(value=payload["max_test_points"]) if payload["max_test_points"] is not None else "")
            + "--rank-grid "
            + " ".join(str(rank) for rank in payload["rank_grid"])
            + " --seed-names "
            + " ".join(payload["seed_names"])
        ),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# k.6.5 Range-Subspace Rank-Sensitivity Report",
        "",
        "This is a rank-sensitivity stress test for k.6.3. It is not a causal mediation proof.",
        "",
        f"final_diagnostic_label: `{payload['final_diagnostic_label']}`",
        "",
        "## Per-Rank Aggregate",
        "",
        "| rank | label | policy delta | value delta | probe top10 |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank in payload["aggregate"]["ranks"]:
        metrics = rank["metrics"]
        lines.append(
            "| {rank} | `{label}` | {policy_delta:.8f} | {value_delta:.8f} | {probe:.6f} |".format(
                rank=rank["rank"],
                label=rank["final_diagnostic_label"],
                policy_delta=metrics["range_minus_random_policy_js_delta"]["mean"],
                value_delta=metrics["range_minus_random_value_delta"]["mean"],
                probe=metrics["probe_top10_accuracy"]["mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Per-Seed Per-Rank",
            "",
            "| seed | rank | label | policy delta | value delta | range policy JS | random policy JS |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        for rank_row in row["rank_rows"]:
            summary = rank_row["summary"]
            lines.append(
                "| {seed} | {rank} | `{label}` | {policy_delta:.8f} | {value_delta:.8f} | {range_policy:.8f} | {random_policy:.8f} |".format(
                    seed=row["seed"],
                    rank=rank_row["requested_rank"],
                    label=rank_row["final_diagnostic_label"],
                    policy_delta=summary["range_minus_random_policy_js_delta_ci"]["mean_delta"],
                    value_delta=summary["range_minus_random_value_delta_ci"]["mean_delta"],
                    range_policy=summary["range_ablation_policy_js"],
                    random_policy=summary["random_ablation_policy_js_mean"],
                )
            )
    lines.extend(
        [
            "",
            "## Claim Discipline",
            "",
            payload["allowed_paper_claim"],
            "",
            f"Forbidden claim: {payload['forbidden_paper_claim']}",
            "",
        ]
    )
    return "\n".join(lines)


def _count_labels(labels) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[str(label)] = counts.get(str(label), 0) + 1
    return counts


def _stable_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)
