from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from .artifacts import checksum_payload, write_json_artifact
from .config import load_limit_teacher_config
from .data.schema import DecisionPoint
from .data.validation import validate_decision_points
from .data.writer import read_decision_points_jsonl, write_decision_points_jsonl
from .envs.limit_holdem import (
    HeuristicLimitHoldemTeacherPolicy,
    LimitHoldemPolicyGenerator,
    RandomLimitHoldemPolicy,
    generate_limit_holdem_hand,
)
from .limit_random import flatten_decision_points


@dataclass(frozen=True)
class LimitTeacherEvaluation:
    """Win-rate and return summary for a Limit teacher-vs-random run."""

    hand_count: int
    teacher_player0_hands: int
    teacher_player1_hands: int
    teacher_wins: int
    random_wins: int
    ties: int
    teacher_total_return: int
    teacher_win_rate: float
    teacher_average_return: float

    def to_record(self) -> dict[str, float | int]:
        """Serialize the evaluation summary."""

        return {
            "hand_count": self.hand_count,
            "teacher_player0_hands": self.teacher_player0_hands,
            "teacher_player1_hands": self.teacher_player1_hands,
            "teacher_wins": self.teacher_wins,
            "random_wins": self.random_wins,
            "ties": self.ties,
            "teacher_total_return": self.teacher_total_return,
            "teacher_win_rate": self.teacher_win_rate,
            "teacher_average_return": self.teacher_average_return,
        }


@dataclass(frozen=True)
class LimitTeacherSmokeResult:
    """Result summary returned by v0.4.2 Limit teacher smoke."""

    seed: int
    hand_count: int
    decision_point_count: int
    illegal_action_count: int
    leakage_count: int
    teacher_win_rate: float
    teacher_average_return: float
    checksum: str
    artifact_dir: Path


def generate_teacher_points(hand_count: int, seed: int) -> tuple[DecisionPoint, ...]:
    """Generate Limit teacher self-play decision points."""

    teacher = HeuristicLimitHoldemTeacherPolicy()
    generator = LimitHoldemPolicyGenerator(policy_by_player=(teacher, teacher), seed=seed)
    return flatten_decision_points(generator.generate_hands(hand_count))


def evaluate_teacher_against_random(seed: int, hand_count: int) -> LimitTeacherEvaluation:
    """Evaluate heuristic Limit teacher against random with alternating seats."""

    teacher = HeuristicLimitHoldemTeacherPolicy()
    random_policy = RandomLimitHoldemPolicy(random.Random(seed))
    deck_rng = random.Random(seed + 1_000_003)
    teacher_wins = 0
    random_wins = 0
    ties = 0
    teacher_total_return = 0
    teacher_player0_hands = 0
    teacher_player1_hands = 0

    for index in range(hand_count):
        teacher_player = index % 2
        if teacher_player == 0:
            teacher_player0_hands += 1
            policies = (teacher, random_policy)
        else:
            teacher_player1_hands += 1
            policies = (random_policy, teacher)
        hand = generate_limit_holdem_hand(
            hand_index=index,
            policy_by_player=policies,
            deck_rng=deck_rng,
        )
        teacher_return = hand.returns[teacher_player]
        teacher_total_return += teacher_return
        if teacher_return > 0:
            teacher_wins += 1
        elif teacher_return < 0:
            random_wins += 1
        else:
            ties += 1

    return LimitTeacherEvaluation(
        hand_count=hand_count,
        teacher_player0_hands=teacher_player0_hands,
        teacher_player1_hands=teacher_player1_hands,
        teacher_wins=teacher_wins,
        random_wins=random_wins,
        ties=ties,
        teacher_total_return=teacher_total_return,
        teacher_win_rate=teacher_wins / hand_count,
        teacher_average_return=teacher_total_return / hand_count,
    )


def run_limit_teacher_smoke(
    config_path: Path,
    artifact_dir: Path | None = None,
) -> LimitTeacherSmokeResult:
    """Generate and validate the v0.4.2 Limit heuristic teacher dataset."""

    config = load_limit_teacher_config(config_path)
    output_dir = artifact_dir if artifact_dir is not None else config.artifact_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    points = generate_teacher_points(hand_count=config.hand_count, seed=config.seed)
    report = validate_decision_points(points)
    evaluation = evaluate_teacher_against_random(seed=config.seed, hand_count=config.evaluation_hand_count)

    dataset_path = output_dir / "decision_points.jsonl"
    report_path = output_dir / "teacher_report.json"
    write_decision_points_jsonl(dataset_path, points)
    read_back_count = len(read_decision_points_jsonl(dataset_path))

    report_record = {
        "version": config.run_version,
        "seed": config.seed,
        "hand_count": report.hand_count,
        "decision_point_count": report.decision_point_count,
        "illegal_action_count": report.illegal_action_count,
        "leakage_count": report.leakage_count,
        "read_back_count": read_back_count,
        "read_back_matches": read_back_count == report.decision_point_count,
        "teacher_evaluation": evaluation.to_record(),
    }
    checksum = checksum_payload(report_record)
    write_json_artifact(report_path, {**report_record, "checksum": checksum})

    return LimitTeacherSmokeResult(
        seed=config.seed,
        hand_count=report.hand_count,
        decision_point_count=report.decision_point_count,
        illegal_action_count=report.illegal_action_count,
        leakage_count=report.leakage_count,
        teacher_win_rate=evaluation.teacher_win_rate,
        teacher_average_return=evaluation.teacher_average_return,
        checksum=checksum,
        artifact_dir=output_dir,
    )
