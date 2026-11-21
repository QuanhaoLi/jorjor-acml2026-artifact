from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.data.validation import validate_decision_points
from jorjor.leduc_cfr_teacher import LeducCfrPolicy, train_leduc_cfr
from jorjor.limit_abstract_cfr_teacher import LimitAbstractCfrPolicy, train_limit_abstract_cfr


EXTERNAL_BACKENDS = ("pyspiel", "open_spiel", "rlcard")


def run_teacher_backend_spike(
    *,
    output_path: Path,
    leduc_iteration_count: int = 20,
    leduc_hand_count: int = 20,
    limit_iteration_count: int = 4,
    limit_deal_sample_count: int = 8,
    limit_hand_count: int = 8,
    seed: int = 20260603,
) -> dict[str, Any]:
    """Probe teacher backends and write the k.0.1 backend decision artifact."""

    if leduc_iteration_count <= 0:
        raise ValueError("leduc_iteration_count must be positive")
    if leduc_hand_count <= 0:
        raise ValueError("leduc_hand_count must be positive")
    if limit_iteration_count <= 0:
        raise ValueError("limit_iteration_count must be positive")
    if limit_deal_sample_count <= 0:
        raise ValueError("limit_deal_sample_count must be positive")
    if limit_hand_count <= 0:
        raise ValueError("limit_hand_count must be positive")

    external = {backend: _probe_import(backend) for backend in EXTERNAL_BACKENDS}
    local = {
        "local_leduc_cfr": _probe_local_leduc_cfr(
            iteration_count=leduc_iteration_count,
            hand_count=leduc_hand_count,
            seed=seed,
        ),
        "local_limit_abstract_cfr": _probe_local_limit_abstract_cfr(
            iteration_count=limit_iteration_count,
            deal_sample_count=limit_deal_sample_count,
            hand_count=limit_hand_count,
            seed=seed,
        ),
    }
    external_available_count = sum(1 for row in external.values() if row["status"] == "available")
    local_backend_pass_count = sum(1 for row in local.values() if row["status"] == "passed")
    payload = {
        "version": "k.0.1",
        "report_type": "k.0.1_teacher_backend_spike",
        "claim_boundary": "backend_selection_only; does_not_prove_belief_tracking",
        "external_backends": external,
        "local_backends": local,
        "external_backend_available_count": external_available_count,
        "local_backend_pass_count": local_backend_pass_count,
        "recommended_backend": _recommend_backend(external=external, local=local),
        "next_version": "k.0.4" if local["local_limit_abstract_cfr"]["status"] == "passed" else "k.0.1_retry",
        "output_path": str(output_path),
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_path, payload)
    return payload


def _probe_import(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {
            "status": "missing",
            "module": module_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {
        "status": "available",
        "module": module_name,
        "version": str(getattr(module, "__version__", "unknown")),
    }


def _probe_local_leduc_cfr(*, iteration_count: int, hand_count: int, seed: int) -> dict[str, Any]:
    result = train_leduc_cfr(iteration_count=iteration_count, seed=seed)
    policy = LeducCfrPolicy(result.policy_table, seed=seed + 17)
    points = policy.generate_points(hand_count=hand_count, seed=seed + 23)
    validation = validate_decision_points(points)
    status = "passed" if validation.illegal_action_count == 0 and validation.leakage_count == 0 else "failed"
    return {
        "status": status,
        "backend_family": "local_tabular_cfr",
        "environment": "leduc",
        "iteration_count": iteration_count,
        "hand_count": validation.hand_count,
        "decision_point_count": validation.decision_point_count,
        "information_set_count": result.information_set_count,
        "illegal_action_count": validation.illegal_action_count,
        "leakage_count": validation.leakage_count,
    }


def _probe_local_limit_abstract_cfr(
    *,
    iteration_count: int,
    deal_sample_count: int,
    hand_count: int,
    seed: int,
) -> dict[str, Any]:
    result = train_limit_abstract_cfr(
        iteration_count=iteration_count,
        deal_sample_count=deal_sample_count,
        seed=seed,
    )
    policy = LimitAbstractCfrPolicy(result.policy_table, seed=seed + 31)
    points = policy.generate_points(hand_count=hand_count, seed=seed + 37)
    validation = validate_decision_points(points)
    status = "passed" if validation.illegal_action_count == 0 and validation.leakage_count == 0 else "failed"
    return {
        "status": status,
        "backend_family": "local_abstract_cfr",
        "environment": "limit_holdem",
        "iteration_count": iteration_count,
        "deal_sample_count": deal_sample_count,
        "hand_count": validation.hand_count,
        "decision_point_count": validation.decision_point_count,
        "information_set_count": result.information_set_count,
        "illegal_action_count": validation.illegal_action_count,
        "leakage_count": validation.leakage_count,
    }


def _recommend_backend(*, external: dict[str, dict[str, Any]], local: dict[str, dict[str, Any]]) -> str:
    if local["local_limit_abstract_cfr"]["status"] == "passed":
        return "local_limit_abstract_cfr_for_k_0_4"
    if external["pyspiel"]["status"] == "available" or external["open_spiel"]["status"] == "available":
        return "openspiel_dependency_spike"
    if external["rlcard"]["status"] == "available":
        return "rlcard_dependency_spike"
    if local["local_leduc_cfr"]["status"] == "passed":
        return "local_leduc_cfr_rehearsal_only"
    return "blocked_no_backend"
