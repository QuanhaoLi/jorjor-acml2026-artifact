from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations, product
import json
import math
from pathlib import Path
import random
from typing import Any

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.probe.counterfactual import jensen_shannon
from jorjor.probe.inspect import COMPOSITION_SHORTCUT_SUPPORTED


BELIEF_LIKE_ORACLE_SUPPORTED = "belief_like_oracle_supported"
SYNTHETIC_STATES = ("A", "B", "C")
SYNTHETIC_OBSERVATIONS = ("x", "y", "z")
PRIOR = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
POSITIONAL_EMISSION = (
    {
        "A": {"x": 0.72, "y": 0.14, "z": 0.14},
        "B": {"x": 0.14, "y": 0.72, "z": 0.14},
        "C": {"x": 0.14, "y": 0.14, "z": 0.72},
    },
    {
        "A": {"x": 0.14, "y": 0.72, "z": 0.14},
        "B": {"x": 0.14, "y": 0.14, "z": 0.72},
        "C": {"x": 0.72, "y": 0.14, "z": 0.14},
    },
    {
        "A": {"x": 0.14, "y": 0.14, "z": 0.72},
        "B": {"x": 0.72, "y": 0.14, "z": 0.14},
        "C": {"x": 0.14, "y": 0.72, "z": 0.14},
    },
)


@dataclass(frozen=True)
class SyntheticSequence:
    """One hidden-state sequence with public observations."""

    sequence_id: str
    hidden_state: str
    observations: tuple[str, ...]


def exact_posterior(observations: tuple[str, ...]) -> tuple[float, ...]:
    """Return exact P(z | ordered observations) under the synthetic emission table."""

    weights = []
    for state, prior in zip(SYNTHETIC_STATES, PRIOR, strict=True):
        likelihood = prior
        for position_index, observation in enumerate(observations):
            likelihood *= _emission_probability(state, observation, position_index)
        weights.append(likelihood)
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("posterior has zero normalizer")
    return tuple(value / total for value in weights)


def generate_synthetic_sequences(
    *,
    sequence_count: int,
    length: int,
    seed: int,
) -> list[SyntheticSequence]:
    """Generate deterministic synthetic hidden-state sequences."""

    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")
    if length <= 1:
        raise ValueError("length must be greater than 1")
    rng = random.Random(seed)
    sequences = []
    for index in range(sequence_count):
        hidden_state = rng.choice(SYNTHETIC_STATES)
        observations = tuple(
            _sample_observation(hidden_state, position_index, rng)
            for position_index in range(length)
        )
        sequences.append(
            SyntheticSequence(
                sequence_id=f"synthetic-{index:06d}",
                hidden_state=hidden_state,
                observations=observations,
            )
        )
    return sequences


def _emission_probability(hidden_state: str, observation: str, position_index: int) -> float:
    table = POSITIONAL_EMISSION[position_index % len(POSITIONAL_EMISSION)]
    return table[hidden_state][observation]


def _sample_observation(hidden_state: str, position_index: int, rng: random.Random) -> str:
    threshold = rng.random()
    cumulative = 0.0
    table = POSITIONAL_EMISSION[position_index % len(POSITIONAL_EMISSION)]
    for observation, probability in table[hidden_state].items():
        cumulative += probability
        if threshold <= cumulative:
            return observation
    return SYNTHETIC_OBSERVATIONS[-1]


def _composition_vector(observations: tuple[str, ...]) -> tuple[float, ...]:
    counts = Counter(observations)
    total = max(float(len(observations)), 1.0)
    return tuple(counts[observation] / total for observation in SYNTHETIC_OBSERVATIONS)


def _shortcut_vector(observations: tuple[str, ...]) -> tuple[float, ...]:
    counts = _composition_vector(observations)
    last = observations[-1] if observations else "none"
    last_one_hot = tuple(1.0 if last == observation else 0.0 for observation in SYNTHETIC_OBSERVATIONS)
    return tuple(0.75 * a + 0.25 * b for a, b in zip(counts, last_one_hot, strict=True))


def _representation(prefix: tuple[str, ...], representation_key: str) -> tuple[float, ...]:
    if representation_key == "oracle_belief_representation":
        return exact_posterior(prefix)
    if representation_key == "composition_shortcut_representation":
        return _normalize(_shortcut_vector(prefix))
    raise ValueError(f"unknown representation: {representation_key}")


def _normalize(vector: tuple[float, ...]) -> tuple[float, ...]:
    total = sum(max(value, 0.0) for value in vector)
    if total <= 0.0:
        return PRIOR
    return tuple(max(value, 0.0) / total for value in vector)


def _prefix_rows(sequences: list[SyntheticSequence]) -> list[dict[str, Any]]:
    rows = []
    for sequence in sequences:
        for prefix_len in range(1, len(sequence.observations) + 1):
            prefix = sequence.observations[:prefix_len]
            posterior = exact_posterior(prefix)
            rows.append(
                {
                    "sequence_id": sequence.sequence_id,
                    "hidden_state": sequence.hidden_state,
                    "prefix_len": prefix_len,
                    "is_final_prefix": prefix_len == len(sequence.observations),
                    "observations": list(prefix),
                    "oracle_posterior": posterior,
                    "oracle_belief_representation": _representation(prefix, "oracle_belief_representation"),
                    "composition_shortcut_representation": _representation(prefix, "composition_shortcut_representation"),
                }
            )
    return rows


def _transition_rows(prefix_rows: list[dict[str, Any]], representation_key: str) -> list[dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in prefix_rows:
        grouped.setdefault(str(row["sequence_id"]), []).append(row)
    transitions = []
    for sequence_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["prefix_len"]))
        for left, right in zip(ordered, ordered[1:], strict=False):
            transitions.append(
                {
                    "sequence_id": sequence_id,
                    "to_final_prefix": bool(right["is_final_prefix"]),
                    "posterior_js": jensen_shannon(left["oracle_posterior"], right["oracle_posterior"]),
                    "representation_js": jensen_shannon(left[representation_key], right[representation_key]),
                }
            )
    return transitions


def _composition_key(observations: tuple[str, ...]) -> tuple[int, ...]:
    counts = Counter(observations)
    return tuple(counts[observation] for observation in SYNTHETIC_OBSERVATIONS)


def _order_counterfactual_rows(
    *,
    length: int,
    representation_key: str,
    max_pairs: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[tuple[int, ...], str], list[tuple[str, ...]]] = {}
    for observations in product(SYNTHETIC_OBSERVATIONS, repeat=length):
        key = (_composition_key(observations), observations[-1])
        groups.setdefault(key, []).append(observations)

    rows = []
    for key, sequences in sorted(groups.items()):
        if len(sequences) < 2:
            continue
        for left, right in combinations(sequences, 2):
            posterior_js = jensen_shannon(exact_posterior(left), exact_posterior(right))
            if posterior_js <= 1e-12:
                continue
            representation_js = jensen_shannon(
                _representation(left, representation_key),
                _representation(right, representation_key),
            )
            rows.append(
                {
                    "composition": list(key[0]),
                    "last_observation": key[1],
                    "left_observations": list(left),
                    "right_observations": list(right),
                    "posterior_js": posterior_js,
                    "representation_js": representation_js,
                }
            )
            if len(rows) >= max_pairs:
                return rows
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x <= 0.0 or denom_y <= 0.0:
        return None
    return numerator / (denom_x * denom_y)


def _pair_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    posterior = [float(row["posterior_js"]) for row in rows]
    representation = [float(row["representation_js"]) for row in rows]
    return {
        "pair_count": len(rows),
        "mean_posterior_js": _mean(posterior),
        "mean_representation_js": _mean(representation),
        "posterior_representation_correlation": _pearson(posterior, representation),
    }


def _summary(rows: list[dict[str, float]]) -> dict[str, Any]:
    non_final = [row for row in rows if not row["to_final_prefix"]]
    target = non_final if non_final else rows
    posterior = [float(row["posterior_js"]) for row in target]
    representation = [float(row["representation_js"]) for row in target]
    return {
        "transition_count": len(rows),
        "non_final_transition_count": len(non_final),
        "mean_posterior_js": _mean(posterior),
        "mean_representation_js": _mean(representation),
        "posterior_representation_correlation": _pearson(posterior, representation),
    }


def _verdict(
    mode: str,
    summary: dict[str, Any],
    order_summary: dict[str, Any],
    *,
    correlation_threshold: float,
) -> str:
    correlation = summary["posterior_representation_correlation"]
    order_correlation = order_summary["posterior_representation_correlation"]
    order_pairs_supported = order_summary["pair_count"] > 0 and order_summary["mean_posterior_js"] > 0.0
    if (
        mode == "oracle_belief"
        and correlation is not None
        and correlation >= correlation_threshold
        and order_pairs_supported
        and order_correlation is not None
        and order_correlation >= correlation_threshold
    ):
        return BELIEF_LIKE_ORACLE_SUPPORTED
    if (
        mode == "composition_shortcut"
        and order_pairs_supported
        and order_summary["mean_representation_js"] <= 1e-12
    ):
        return COMPOSITION_SHORTCUT_SUPPORTED
    return "inconclusive"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Synthetic Oracle-Belief Report",
        "",
        "## Summary",
        "",
        "| mode | verdict | non-final transitions | trajectory corr | order pairs | order posterior JS | order representation JS |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        lines.append(
            "| {mode} | {final_diagnostic_label} | {non_final_transition_count} | {trajectory_corr} | {order_pairs} | {order_posterior_js} | {order_representation_js} |".format(
                mode=case["mode"],
                final_diagnostic_label=case["final_diagnostic_label"],
                non_final_transition_count=case["summary"]["non_final_transition_count"],
                trajectory_corr=case["summary"]["posterior_representation_correlation"],
                order_pairs=case["order_counterfactual_summary"]["pair_count"],
                order_posterior_js=case["order_counterfactual_summary"]["mean_posterior_js"],
                order_representation_js=case["order_counterfactual_summary"]["mean_representation_js"],
            )
        )
    return "\n".join(lines) + "\n"


def build_synthetic_oracle_belief_report(
    *,
    output_dir: Path,
    sequence_count: int,
    length: int = 6,
    seed: int = 20260602,
    correlation_threshold: float = 0.95,
    max_order_pairs: int = 4096,
    write_prefix_rows: bool = False,
) -> dict[str, Any]:
    """Build synthetic positive/negative controls for JorJor-Inspect."""

    sequences = generate_synthetic_sequences(sequence_count=sequence_count, length=length, seed=seed)
    prefix_rows = _prefix_rows(sequences)
    cases = []
    for mode, representation_key in (
        ("oracle_belief", "oracle_belief_representation"),
        ("composition_shortcut", "composition_shortcut_representation"),
    ):
        transitions = _transition_rows(prefix_rows, representation_key)
        summary = _summary(transitions)
        order_rows = _order_counterfactual_rows(
            length=length,
            representation_key=representation_key,
            max_pairs=max_order_pairs,
        )
        order_summary = _pair_summary(order_rows)
        cases.append(
            {
                "mode": mode,
                "summary": summary,
                "order_counterfactual_summary": order_summary,
                "final_diagnostic_label": _verdict(
                    mode,
                    summary,
                    order_summary,
                    correlation_threshold=correlation_threshold,
                ),
            }
        )
    payload = {
        "version": "1.7",
        "report_type": "synthetic_oracle_belief",
        "sequence_count": sequence_count,
        "length": length,
        "seed": seed,
        "states": list(SYNTHETIC_STATES),
        "observations": list(SYNTHETIC_OBSERVATIONS),
        "emission_mode": "position_dependent_order_sensitive",
        "max_order_pairs": max_order_pairs,
        "cases": cases,
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "synthetic_oracle_belief_report.json", payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "synthetic_oracle_belief_report.md").write_text(_markdown(payload), encoding="utf-8")
    if write_prefix_rows:
        with (output_dir / "prefix_rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in prefix_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return payload
