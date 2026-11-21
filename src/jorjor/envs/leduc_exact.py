from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import random
from pathlib import Path
from typing import Any

from jorjor.artifacts import checksum_payload, write_json_artifact
from jorjor.data.schema import DecisionPoint
from jorjor.envs.leduc import DECK, LeducPolicyGenerator, LeducPolicyState, card_rank
from jorjor.probe.counterfactual import jensen_shannon


EXACT_LEDUC_FEASIBILITY_SUPPORTED = "exact_leduc_feasibility_supported"
INSUFFICIENT_EXACT_POSTERIOR_SUPPORT = "insufficient_exact_posterior_support"
EXACT_POSTERIOR_SIGNAL_SUPPORTED = "exact_posterior_signal_supported"
COMPOSITION_COLLAPSED = "composition_collapsed"
LEDUC_ACTIONS = ("bet", "call", "check", "fold")


@dataclass(frozen=True)
class StochasticLeducPolicy:
    """Known stochastic Leduc policy used for exact likelihood computation."""

    rng: random.Random
    aggressiveness: float = 0.08

    def choose_action(self, state: LeducPolicyState) -> str:
        """Sample one legal action from the known policy."""

        probabilities = self.action_probabilities(state)
        threshold = self.rng.random()
        cumulative = 0.0
        for action in state.legal_actions:
            cumulative += probabilities[action]
            if threshold <= cumulative:
                return action
        return state.legal_actions[-1]

    def action_probabilities(self, state: LeducPolicyState) -> dict[str, float]:
        """Return p(action | information state) for all legal actions."""

        strength = _card_strength(state.private_card, state.public_card)
        if state.legal_actions == ("check", "bet"):
            bet_probability = min(0.92, max(0.08, self.aggressiveness + 0.18 + 0.18 * strength))
            if _last_action(state.betting_history) == "check":
                bet_probability = min(0.95, bet_probability + 0.10)
            return {"check": 1.0 - bet_probability, "bet": bet_probability}
        if state.legal_actions == ("fold", "call"):
            call_probability = min(0.94, max(0.06, 0.18 + 0.20 * strength))
            if _last_action(state.betting_history) == "bet":
                call_probability = min(0.96, call_probability + 0.08)
            return {"fold": 1.0 - call_probability, "call": call_probability}
        raise ValueError(f"unsupported legal action set: {state.legal_actions}")


def build_exact_leduc_feasibility_report(
    *,
    output_dir: Path,
    hand_count: int = 50000,
    seed: int = 20260610,
    max_pairs: int = 20000,
    bootstrap_count: int = 1000,
    bootstrap_seed: int = 20260610,
    confidence_level: float = 0.95,
    min_bucket_count: int = 100,
    min_pair_multiplicity: int = 2000,
) -> dict[str, Any]:
    """Build the r.0.1 exact-posterior Leduc feasibility report."""

    if hand_count <= 0:
        raise ValueError("hand_count must be positive")
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    if bootstrap_count <= 0:
        raise ValueError("bootstrap_count must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = StochasticLeducPolicy(random.Random(seed + 1))
    generator = LeducPolicyGenerator((policy, policy), seed=seed)
    points = tuple(point for hand in generator.generate_hands(hand_count) for point in hand.decision_points)
    rows = exact_leduc_rows(points=points, policy=policy)
    matched_rows = matched_exact_posterior_rows(rows, max_pairs=max_pairs)
    matched_summary = matched_exact_posterior_summary(
        matched_rows,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    oracle_label = (
        EXACT_POSTERIOR_SIGNAL_SUPPORTED
        if matched_summary["mean_oracle_representation_js"] > 0.0
        and matched_summary["posterior_js_ci"]["lower"] > 0.0
        else INSUFFICIENT_EXACT_POSTERIOR_SUPPORT
    )
    composition_label = (
        COMPOSITION_COLLAPSED
        if matched_summary["mean_composition_representation_js"] <= 1e-12
        else "composition_not_collapsed"
    )
    feasibility_passed = (
        matched_summary["matched_bucket_count"] >= min_bucket_count
        and matched_summary["matched_pair_multiplicity"] >= min_pair_multiplicity
        and matched_summary["posterior_js_ci"]["lower"] > 0.0
        and oracle_label == EXACT_POSTERIOR_SIGNAL_SUPPORTED
        and composition_label == COMPOSITION_COLLAPSED
    )
    payload = {
        "version": "r.0.1",
        "report_type": "r.0.1_exact_leduc_feasibility",
        "claim_scope": "exact_posterior_domain_feasibility_not_model_mechanism",
        "hand_count": hand_count,
        "decision_point_count": len(points),
        "seed": seed,
        "max_pairs": max_pairs,
        "bootstrap_count": bootstrap_count,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "min_bucket_count": min_bucket_count,
        "min_pair_multiplicity": min_pair_multiplicity,
        "row_summary": _row_summary(rows),
        "matched_summary": matched_summary,
        "oracle_detection_label": oracle_label,
        "composition_detection_label": composition_label,
        "blocking_reasons": _blocking_reasons(
            matched_summary=matched_summary,
            min_bucket_count=min_bucket_count,
            min_pair_multiplicity=min_pair_multiplicity,
        ),
        "next_version_recommendation": (
            "Use r.0.2 to build an order-rich exact-posterior poker-like environment. The local two-round Leduc "
            "ruleset is valuable as a negative feasibility audit, but its legal betting structure leaves no "
            "same-composition/different-order matched buckets for causal CBPS mechanism tests."
        ),
        "final_diagnostic_label": (
            EXACT_LEDUC_FEASIBILITY_SUPPORTED if feasibility_passed else INSUFFICIENT_EXACT_POSTERIOR_SUPPORT
        ),
        "allowed_paper_claim": (
            "Exact-posterior Leduc has or lacks enough matched-composition posterior variation for later "
            "mechanism experiments. This does not claim a trained model has CBPS or belief."
        ),
        "forbidden_paper_claim": "r.0.1 proves behavior-trained AR models learn CBPS or posterior tracking.",
        "checksum": "",
    }
    payload["checksum"] = checksum_payload({key: value for key, value in payload.items() if key != "checksum"})
    write_json_artifact(output_dir / "r.0.1_exact_leduc_feasibility_report.json", payload)
    (output_dir / "r.0.1_exact_leduc_feasibility_report.md").write_text(_markdown(payload), encoding="utf-8")
    write_json_artifact(output_dir / "r.0.1_exact_leduc_feasibility_manifest.json", _manifest(output_dir, payload))
    return payload


def exact_leduc_rows(*, points: tuple[DecisionPoint, ...], policy: StochasticLeducPolicy) -> list[dict[str, Any]]:
    """Return decision-point rows with exact opponent-card posterior vectors."""

    rows = []
    for point in points:
        posterior = exact_opponent_card_posterior(point=point, policy=policy)
        history = _action_history(point.visible_state["betting_history"])
        rows.append(
            {
                "hand_id": point.hand_id,
                "step_index": point.step_index,
                "acting_player": point.acting_player,
                "own_private_card": point.visible_state["private_card"],
                "public_card": point.visible_state.get("public_card"),
                "round": point.visible_state["round"],
                "legal_actions": list(point.legal_actions),
                "action_label": point.action,
                "value_label": point.terminal_return,
                "opponent_private_card": point.opponent_private_card,
                "opponent_rank": point.opponent_rank,
                "action_history": history,
                "action_counts": dict(Counter(history)),
                "last_action": history[-1] if history else "none",
                "exact_posterior": posterior,
                "composition_key": exact_leduc_composition_key(point),
                "order_key": exact_leduc_order_key(point),
            }
        )
    return rows


def exact_opponent_card_posterior(*, point: DecisionPoint, policy: StochasticLeducPolicy) -> tuple[float, ...]:
    """Compute P(opponent private card | visible state) by enumerating Leduc deals."""

    own_card = str(point.visible_state["private_card"])
    public_card = point.visible_state.get("public_card")
    if own_card not in DECK:
        raise ValueError(f"unknown own Leduc card: {own_card}")
    weights = {card: 0.0 for card in DECK}
    for opponent_card in DECK:
        if opponent_card == own_card or opponent_card == public_card:
            continue
        public_candidates = [str(public_card)] if public_card is not None else [
            card for card in DECK if card not in {own_card, opponent_card}
        ]
        for candidate_public in public_candidates:
            likelihood = _prefix_likelihood(
                point=point,
                own_card=own_card,
                opponent_card=opponent_card,
                public_card=candidate_public,
                policy=policy,
            )
            weights[opponent_card] += likelihood
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError(f"posterior normalizer is zero for {point.hand_id}:{point.step_index}")
    return tuple(weights[card] / total for card in DECK)


def exact_leduc_composition_key(point: DecisionPoint) -> str:
    """Return the visible non-ordered composition key for exact Leduc gates."""

    actions = _action_history(point.visible_state["betting_history"])
    counts = Counter(actions)
    legal = ",".join(point.legal_actions)
    parts = [
        f"actor={point.acting_player}",
        f"round={point.visible_state['round']}",
        f"own={point.visible_state['private_card']}",
        f"public={point.visible_state.get('public_card')}",
        f"legal={legal}",
        f"len={len(actions)}",
        f"last={actions[-1] if actions else 'none'}",
    ]
    parts.extend(f"{action}={counts[action]}" for action in LEDUC_ACTIONS)
    return "|".join(parts)


def exact_leduc_order_key(point: DecisionPoint) -> str:
    """Return the ordered public action key for exact Leduc gates."""

    events = []
    for event in point.visible_state["betting_history"]:
        if event.get("type") == "action":
            events.append(f"{event.get('round')}:{event.get('player')}:{event.get('action')}")
    return ",".join(events) if events else "empty"


def matched_exact_posterior_rows(rows: list[dict[str, Any]], *, max_pairs: int) -> list[dict[str, Any]]:
    """Return same-composition/different-order exact-posterior pair rows."""

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        bucket = str(row["composition_key"])
        order = str(row["order_key"])
        if order not in grouped[bucket]:
            grouped[bucket][order] = {
                "order_key": order,
                "posterior": tuple(float(value) for value in row["exact_posterior"]),
                "row_count": 0,
                "example": {
                    "hand_id": row["hand_id"],
                    "step_index": row["step_index"],
                    "action_history": row["action_history"],
                },
            }
        grouped[bucket][order]["row_count"] += 1
    pair_rows = []
    for bucket, order_rows in sorted(grouped.items()):
        if len(order_rows) < 2:
            continue
        for left, right in combinations(sorted(order_rows.values(), key=lambda item: item["order_key"]), 2):
            posterior_js = jensen_shannon(left["posterior"], right["posterior"])
            if posterior_js <= 1e-12:
                continue
            pair_rows.append(
                {
                    "composition_key": bucket,
                    "left_order_key": left["order_key"],
                    "right_order_key": right["order_key"],
                    "left_count": left["row_count"],
                    "right_count": right["row_count"],
                    "pair_multiplicity": left["row_count"] * right["row_count"],
                    "posterior_js": posterior_js,
                    "oracle_representation_js": posterior_js,
                    "composition_representation_js": 0.0,
                    "left_example": left["example"],
                    "right_example": right["example"],
                }
            )
            if len(pair_rows) >= max_pairs:
                return pair_rows
    return pair_rows


def matched_exact_posterior_summary(
    rows: list[dict[str, Any]],
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Summarize exact-posterior matched rows with bucket bootstrap intervals."""

    pair_posterior_js = [float(row["posterior_js"]) for row in rows]
    pair_oracle_js = [float(row["oracle_representation_js"]) for row in rows]
    pair_composition_js = [float(row["composition_representation_js"]) for row in rows]
    buckets = sorted({str(row["composition_key"]) for row in rows})
    bucket_means = _bucket_value_means(rows) if rows else {}
    posterior_ci = _bucket_bootstrap_interval(
        rows,
        value_key="posterior_js",
        bucket_key="composition_key",
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    ) if rows else {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    return {
        "matched_group_pair_count": len(rows),
        "matched_pair_multiplicity": int(sum(int(row["pair_multiplicity"]) for row in rows)),
        "matched_bucket_count": len(buckets),
        "mean_posterior_js": _mean(bucket_means.get("posterior_js", [])),
        "mean_oracle_representation_js": _mean(bucket_means.get("oracle_representation_js", [])),
        "mean_composition_representation_js": _mean(bucket_means.get("composition_representation_js", [])),
        "posterior_js_ci": posterior_ci,
        "pair_table_mean_posterior_js": _mean(pair_posterior_js),
        "pair_table_mean_oracle_representation_js": _mean(pair_oracle_js),
        "pair_table_mean_composition_representation_js": _mean(pair_composition_js),
        "pair_weighted_mean_posterior_js": _weighted_mean(
            [(float(row["posterior_js"]), int(row["pair_multiplicity"])) for row in rows]
        ),
        "estimand": "unweighted_composition_bucket_mean",
    }


def _prefix_likelihood(
    *,
    point: DecisionPoint,
    own_card: str,
    opponent_card: str,
    public_card: str,
    policy: StochasticLeducPolicy,
) -> float:
    acting_player = point.acting_player
    private_cards = [None, None]
    private_cards[acting_player] = own_card
    private_cards[1 - acting_player] = opponent_card
    observed_history = [dict(event) for event in point.visible_state["betting_history"]]
    likelihood = _round_prefix_likelihood(
        point=point,
        private_cards=(str(private_cards[0]), str(private_cards[1])),
        public_card=None,
        round_name="preflop",
        bet_size=2,
        observed_history=observed_history,
        history=[],
        contributions=[1, 1],
        policy=policy,
    )
    if likelihood["matched"]:
        return likelihood["likelihood"]
    if likelihood["terminal"] or not likelihood["round_complete"]:
        return 0.0
    history = likelihood["history"]
    contributions = likelihood["contributions"]
    deal_event = {"type": "deal_public", "card": public_card}
    if len(observed_history) <= len(history) or observed_history[len(history)] != deal_event:
        return 0.0
    history = [*history, deal_event]
    likelihood = _round_prefix_likelihood(
        point=point,
        private_cards=(str(private_cards[0]), str(private_cards[1])),
        public_card=public_card,
        round_name="public",
        bet_size=4,
        observed_history=observed_history,
        history=history,
        contributions=contributions,
        policy=policy,
        initial_likelihood=likelihood["likelihood"],
    )
    return likelihood["likelihood"] if likelihood["matched"] else 0.0


def _round_prefix_likelihood(
    *,
    point: DecisionPoint,
    private_cards: tuple[str, str],
    public_card: str | None,
    round_name: str,
    bet_size: int,
    observed_history: list[dict[str, Any]],
    history: list[dict[str, Any]],
    contributions: list[int],
    policy: StochasticLeducPolicy,
    initial_likelihood: float = 1.0,
) -> dict[str, Any]:
    current_player = 0
    checks = 0
    outstanding_bet = False
    bettor: int | None = None
    likelihood = initial_likelihood
    while True:
        legal_actions = ("fold", "call") if outstanding_bet else ("check", "bet")
        state = LeducPolicyState(
            hand_id=str(point.hand_id),
            acting_player=current_player,
            private_card=private_cards[current_player],
            public_card=public_card,
            round_name=round_name,
            pot=sum(contributions),
            contributions=(contributions[0], contributions[1]),
            betting_history=tuple(history),
            legal_actions=legal_actions,
        )
        if _state_matches_point(state, point, observed_history):
            return {
                "matched": True,
                "terminal": False,
                "round_complete": False,
                "history": history,
                "contributions": contributions,
                "likelihood": likelihood,
            }
        if len(observed_history) <= len(history):
            return {
                "matched": False,
                "terminal": False,
                "round_complete": False,
                "history": history,
                "contributions": contributions,
                "likelihood": likelihood,
            }
        event = observed_history[len(history)]
        if (
            event.get("type") != "action"
            or event.get("round") != round_name
            or int(event.get("player")) != current_player
        ):
            return {
                "matched": False,
                "terminal": False,
                "round_complete": False,
                "history": history,
                "contributions": contributions,
                "likelihood": likelihood,
            }
        action = str(event.get("action"))
        probabilities = policy.action_probabilities(state)
        probability = float(probabilities.get(action, 0.0))
        if probability <= 0.0:
            return {
                "matched": False,
                "terminal": False,
                "round_complete": False,
                "history": history,
                "contributions": contributions,
                "likelihood": 0.0,
            }
        likelihood *= probability
        history = [*history, event]
        if action == "fold":
            return {
                "matched": False,
                "terminal": True,
                "round_complete": False,
                "history": history,
                "contributions": contributions,
                "likelihood": likelihood,
            }
        if action == "call":
            contributions = contributions[:]
            contributions[current_player] += bet_size
            return {
                "matched": False,
                "terminal": False,
                "round_complete": True,
                "history": history,
                "contributions": contributions,
                "likelihood": likelihood,
            }
        if action == "bet":
            contributions = contributions[:]
            contributions[current_player] += bet_size
            outstanding_bet = True
            bettor = current_player
            current_player = 1 - current_player
            continue
        if action == "check":
            checks += 1
            if checks == 2:
                return {
                    "matched": False,
                    "terminal": False,
                    "round_complete": True,
                    "history": history,
                    "contributions": contributions,
                    "likelihood": likelihood,
                }
            current_player = 1 - current_player
            continue
        raise ValueError(f"unknown Leduc action: {action}")


def _state_matches_point(state: LeducPolicyState, point: DecisionPoint, observed_history: list[dict[str, Any]]) -> bool:
    visible = point.visible_state
    return (
        state.acting_player == point.acting_player
        and state.private_card == visible["private_card"]
        and state.public_card == visible.get("public_card")
        and state.round_name == visible["round"]
        and state.pot == visible["pot"]
        and list(state.betting_history) == observed_history
        and tuple(state.legal_actions) == tuple(point.legal_actions)
    )


def _card_strength(private_card: str, public_card: str | None) -> float:
    rank = card_rank(private_card)
    rank_score = {"J": 0.0, "Q": 1.0, "K": 2.0}[rank]
    pair_bonus = 2.5 if public_card is not None and card_rank(public_card) == rank else 0.0
    public_pressure = 0.3 if public_card is not None else 0.0
    return rank_score + pair_bonus + public_pressure


def _action_history(history: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    return tuple(str(event["action"]) for event in history if event.get("type") == "action")


def _last_action(history: tuple[dict[str, object], ...]) -> str | None:
    for event in reversed(history):
        if event.get("type") == "action":
            return str(event.get("action"))
    return None


def _row_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    round_counts = Counter(str(row["round"]) for row in rows)
    action_counts = Counter(str(row["action_label"]) for row in rows)
    return {
        "row_count": len(rows),
        "round_counts": dict(sorted(round_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "composition_bucket_count": len({str(row["composition_key"]) for row in rows}),
        "order_key_count": len({str(row["order_key"]) for row in rows}),
    }


def _bucket_bootstrap_interval(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    bucket_key: str,
    bootstrap_count: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, float]:
    bucket_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        bucket_values[str(row[bucket_key])].append(float(row[value_key]))
    if not bucket_values:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    bucket_means = [sum(values) / len(values) for values in bucket_values.values()]
    rng = random.Random(bootstrap_seed)
    means = []
    for _ in range(bootstrap_count):
        means.append(sum(bucket_means[rng.randrange(len(bucket_means))] for _ in bucket_means) / len(bucket_means))
    means.sort()
    lower_index = int(((1.0 - confidence_level) / 2.0) * (bootstrap_count - 1))
    upper_index = int((1.0 - (1.0 - confidence_level) / 2.0) * (bootstrap_count - 1))
    return {
        "mean": sum(bucket_means) / len(bucket_means),
        "lower": means[lower_index],
        "upper": means[upper_index],
    }


def _bucket_value_means(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    value_keys = ("posterior_js", "oracle_representation_js", "composition_representation_js")
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {key: [] for key in value_keys})
    for row in rows:
        bucket = str(row["composition_key"])
        for key in value_keys:
            grouped[bucket][key].append(float(row[key]))
    means: dict[str, list[float]] = {key: [] for key in value_keys}
    for bucket in sorted(grouped):
        for key in value_keys:
            means[key].append(_mean(grouped[bucket][key]))
    return means


def _manifest(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": payload["version"],
        "report_type": "r.0.1_exact_leduc_feasibility_manifest",
        "claim_scope": payload["claim_scope"],
        "output_dir": str(output_dir),
        "expected_outputs": [
            "r.0.1_exact_leduc_feasibility_report.json",
            "r.0.1_exact_leduc_feasibility_report.md",
            "r.0.1_exact_leduc_feasibility_manifest.json",
        ],
        "report_checksum": payload["checksum"],
        "final_diagnostic_label": payload["final_diagnostic_label"],
        "one_command_reproduction": (
            ".venv/bin/python scripts/build_r001_exact_leduc_feasibility.py "
            f"--output-dir {output_dir} "
            f"--hand-count {payload['hand_count']} "
            f"--seed {payload['seed']} "
            f"--max-pairs {payload['max_pairs']} "
            f"--bootstrap-count {payload['bootstrap_count']} "
            f"--bootstrap-seed {payload['bootstrap_seed']}"
        ),
    }


def _blocking_reasons(
    *,
    matched_summary: dict[str, Any],
    min_bucket_count: int,
    min_pair_multiplicity: int,
) -> list[str]:
    reasons = []
    if matched_summary["matched_bucket_count"] < min_bucket_count:
        reasons.append(
            "matched_bucket_count below requirement; strict public composition does not admit enough different orders"
        )
    if matched_summary["matched_pair_multiplicity"] < min_pair_multiplicity:
        reasons.append("matched_pair_multiplicity below requirement")
    if matched_summary["posterior_js_ci"]["lower"] <= 0.0:
        reasons.append("posterior_js_ci lower bound is not positive")
    return reasons


def _markdown(payload: dict[str, Any]) -> str:
    matched = payload["matched_summary"]
    lines = [
        "# r.0.1 Exact Leduc Feasibility Report",
        "",
        f"final_diagnostic_label: `{payload['final_diagnostic_label']}`",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| decision_point_count | {payload['decision_point_count']} |",
        f"| matched_bucket_count | {matched['matched_bucket_count']} |",
        f"| matched_group_pair_count | {matched['matched_group_pair_count']} |",
        f"| matched_pair_multiplicity | {matched['matched_pair_multiplicity']} |",
        f"| mean_posterior_js | {matched['mean_posterior_js']:.8f} |",
        f"| posterior_js_ci_lower | {matched['posterior_js_ci']['lower']:.8f} |",
        f"| mean_oracle_representation_js | {matched['mean_oracle_representation_js']:.8f} |",
        f"| mean_composition_representation_js | {matched['mean_composition_representation_js']:.8f} |",
        "",
        "## Controls",
        "",
        f"- oracle_detection_label: `{payload['oracle_detection_label']}`",
        f"- composition_detection_label: `{payload['composition_detection_label']}`",
        "",
        "## Blocking Reasons",
        "",
        *[f"- {reason}" for reason in payload["blocking_reasons"]],
        "",
        "## Next Version",
        "",
        payload["next_version_recommendation"],
        "",
        "## Claim Discipline",
        "",
        payload["allowed_paper_claim"],
        "",
        f"Forbidden claim: {payload['forbidden_paper_claim']}",
        "",
    ]
    return "\n".join(lines)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _weighted_mean(values: list[tuple[float, int]]) -> float:
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return 0.0
    return sum(value * weight for value, weight in values) / total_weight
