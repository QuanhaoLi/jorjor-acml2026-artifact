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
from jorjor.envs.synthetic_belief import BELIEF_LIKE_ORACLE_SUPPORTED
from jorjor.probe.counterfactual import jensen_shannon
from jorjor.probe.inspect import COMPOSITION_SHORTCUT_SUPPORTED


TOY_STATES = ("low", "mid", "high")
TOY_SIGNALS = ("quiet", "probe", "push")
TOY_PRIOR = (0.25, 0.5, 0.25)
TOY_POSITIONAL_EMISSION = (
    {
        "low": {"quiet": 0.74, "probe": 0.18, "push": 0.08},
        "mid": {"quiet": 0.18, "probe": 0.64, "push": 0.18},
        "high": {"quiet": 0.08, "probe": 0.18, "push": 0.74},
    },
    {
        "low": {"quiet": 0.12, "probe": 0.72, "push": 0.16},
        "mid": {"quiet": 0.16, "probe": 0.12, "push": 0.72},
        "high": {"quiet": 0.72, "probe": 0.16, "push": 0.12},
    },
    {
        "low": {"quiet": 0.18, "probe": 0.12, "push": 0.70},
        "mid": {"quiet": 0.70, "probe": 0.18, "push": 0.12},
        "high": {"quiet": 0.12, "probe": 0.70, "push": 0.18},
    },
)


@dataclass(frozen=True)
class ToyHiddenSequence:
    """One tiny imperfect-information sequence with a hidden state."""

    sequence_id: str
    hidden_state: str
    signals: tuple[str, ...]


def toy_exact_posterior(signals: tuple[str, ...]) -> tuple[float, ...]:
    """Return exact posterior over hidden state given ordered public signals."""

    weights = []
    for hidden_state, prior in zip(TOY_STATES, TOY_PRIOR, strict=True):
        likelihood = prior
        for position_index, signal in enumerate(signals):
            likelihood *= _emission_probability(hidden_state, signal, position_index)
        weights.append(likelihood)
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("toy posterior has zero normalizer")
    return tuple(value / total for value in weights)


def generate_toy_sequences(*, sequence_count: int, length: int, seed: int) -> list[ToyHiddenSequence]:
    """Generate deterministic toy hidden-state sequences."""

    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")
    if length <= 1:
        raise ValueError("length must be greater than 1")
    rng = random.Random(seed)
    sequences = []
    for index in range(sequence_count):
        hidden_state = rng.choices(TOY_STATES, weights=TOY_PRIOR, k=1)[0]
        signals = tuple(_sample_signal(hidden_state, position_index, rng) for position_index in range(length))
        sequences.append(
            ToyHiddenSequence(
                sequence_id=f"toy-{index:06d}",
                hidden_state=hidden_state,
                signals=signals,
            )
        )
    return sequences


def _emission_probability(hidden_state: str, signal: str, position_index: int) -> float:
    table = TOY_POSITIONAL_EMISSION[position_index % len(TOY_POSITIONAL_EMISSION)]
    return table[hidden_state][signal]


def _sample_signal(hidden_state: str, position_index: int, rng: random.Random) -> str:
    threshold = rng.random()
    cumulative = 0.0
    table = TOY_POSITIONAL_EMISSION[position_index % len(TOY_POSITIONAL_EMISSION)]
    for signal, probability in table[hidden_state].items():
        cumulative += probability
        if threshold <= cumulative:
            return signal
    return TOY_SIGNALS[-1]


def _normalize(vector: tuple[float, ...]) -> tuple[float, ...]:
    total = sum(max(value, 0.0) for value in vector)
    if total <= 0.0:
        return TOY_PRIOR
    return tuple(max(value, 0.0) / total for value in vector)


def _composition_representation(signals: tuple[str, ...]) -> tuple[float, ...]:
    counts = Counter(signals)
    total = max(float(len(signals)), 1.0)
    return tuple(counts[signal] / total for signal in TOY_SIGNALS)


def _representation(signals: tuple[str, ...], key: str) -> tuple[float, ...]:
    if key == "oracle_belief_representation":
        return toy_exact_posterior(signals)
    if key == "composition_shortcut_representation":
        return _normalize(_composition_representation(signals))
    raise ValueError(f"unknown toy representation: {key}")


def _prefix_rows(sequences: list[ToyHiddenSequence]) -> list[dict[str, Any]]:
    rows = []
    for sequence in sequences:
        for prefix_len in range(1, len(sequence.signals) + 1):
            prefix = sequence.signals[:prefix_len]
            rows.append(
                {
                    "sequence_id": sequence.sequence_id,
                    "hidden_state": sequence.hidden_state,
                    "prefix_len": prefix_len,
                    "is_final_prefix": prefix_len == len(sequence.signals),
                    "signals": list(prefix),
                    "oracle_posterior": toy_exact_posterior(prefix),
                    "oracle_belief_representation": _representation(prefix, "oracle_belief_representation"),
                    "composition_shortcut_representation": _representation(prefix, "composition_shortcut_representation"),
                }
            )
    return rows


def _transition_rows(prefix_rows: list[dict[str, Any]], representation_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in prefix_rows:
        grouped.setdefault(str(row["sequence_id"]), []).append(row)
    transitions = []
    for sequence_id, rows in grouped.items():
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


def _composition_key(signals: tuple[str, ...]) -> tuple[int, ...]:
    counts = Counter(signals)
    return tuple(counts[signal] for signal in TOY_SIGNALS)


def _order_counterfactual_rows(*, length: int, representation_key: str, max_pairs: int) -> list[dict[str, Any]]:
    groups: dict[tuple[tuple[int, ...], str], list[tuple[str, ...]]] = {}
    for signals in product(TOY_SIGNALS, repeat=length):
        key = (_composition_key(signals), signals[-1])
        groups.setdefault(key, []).append(signals)
    rows = []
    for key, sequences in sorted(groups.items()):
        if len(sequences) < 2:
            continue
        for left, right in combinations(sequences, 2):
            posterior_js = jensen_shannon(toy_exact_posterior(left), toy_exact_posterior(right))
            if posterior_js <= 1e-12:
                continue
            representation_js = jensen_shannon(
                _representation(left, representation_key),
                _representation(right, representation_key),
            )
            rows.append(
                {
                    "composition": list(key[0]),
                    "last_signal": key[1],
                    "left_signals": list(left),
                    "right_signals": list(right),
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


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
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


def _pair_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    posterior = [float(row["posterior_js"]) for row in rows]
    representation = [float(row["representation_js"]) for row in rows]
    return {
        "pair_count": len(rows),
        "mean_posterior_js": _mean(posterior),
        "mean_representation_js": _mean(representation),
        "posterior_representation_correlation": _pearson(posterior, representation),
    }


def _verdict(mode: str, summary: dict[str, Any], order_summary: dict[str, Any], threshold: float) -> str:
    trajectory_corr = summary["posterior_representation_correlation"]
    order_corr = order_summary["posterior_representation_correlation"]
    has_order_pairs = order_summary["pair_count"] > 0 and order_summary["mean_posterior_js"] > 0.0
    if (
        mode == "oracle_belief"
        and trajectory_corr is not None
        and trajectory_corr >= threshold
        and has_order_pairs
        and order_corr is not None
        and order_corr >= threshold
    ):
        return BELIEF_LIKE_ORACLE_SUPPORTED
    if mode == "composition_shortcut" and has_order_pairs and order_summary["mean_representation_js"] <= 1e-12:
        return COMPOSITION_SHORTCUT_SUPPORTED
    return "inconclusive"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Toy Imperfect-Information Oracle Report",
        "",
        "## Scope",
        "",
        "This validates the diagnostic protocol only; it is not poker model evidence.",
        "",
        "| mode | verdict | order pairs | order posterior JS | order representation JS |",
        "|---|---|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        order_summary = case["order_counterfactual_summary"]
        lines.append(
            "| {mode} | {label} | {pairs} | {posterior:.6f} | {representation:.6f} |".format(
                mode=case["mode"],
                label=case["final_diagnostic_label"],
                pairs=order_summary["pair_count"],
                posterior=order_summary["mean_posterior_js"],
                representation=order_summary["mean_representation_js"],
            )
        )
    return "\n".join(lines) + "\n"


def build_toy_imperfect_info_oracle_report(
    *,
    output_dir: Path,
    sequence_count: int,
    length: int = 5,
    seed: int = 20260602,
    correlation_threshold: float = 0.95,
    max_order_pairs: int = 1024,
) -> dict[str, Any]:
    """Build a second exact-belief toy validation for the diagnostic protocol."""

    sequences = generate_toy_sequences(sequence_count=sequence_count, length=length, seed=seed)
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
                    correlation_threshold,
                ),
            }
        )
    payload = {
                "version": "1.15",
        "report_type": "toy_imperfect_info_oracle",
        "sequence_count": sequence_count,
        "length": length,
        "seed": seed,
        "states": list(TOY_STATES),
        "signals": list(TOY_SIGNALS),
        "emission_mode": "toy_imperfect_information_position_dependent",
        "max_order_pairs": max_order_pairs,
        "claim_scope": "methodology_validation_only_not_poker_evidence",
        "cases": cases,
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "toy_imperfect_info_oracle_report.json", payload)
    (output_dir / "toy_imperfect_info_oracle_report.md").write_text(_markdown(payload), encoding="utf-8")
    return payload
