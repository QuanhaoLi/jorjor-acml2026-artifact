from __future__ import annotations

import argparse
from pathlib import Path

from .leduc_cfr_teacher import run_leduc_cfr_teacher_smoke
from .leduc_random_baseline import run_leduc_random_baseline
from .leduc_random import run_leduc_random_smoke
from .leduc_teacher import run_leduc_teacher_smoke
from .limit_abstraction import run_limit_abstraction_audit
from .limit_abstract_cfr_teacher import run_limit_abstract_cfr_teacher_smoke
from .limit_baseline import run_limit_baseline_smoke
from .limit_loader import run_limit_loader_smoke
from .limit_prior_baseline import run_limit_prior_baseline
from .limit_random import run_limit_random_smoke
from .limit_seed_sweep import run_limit_seed_sweep
from .limit_teacher import run_limit_teacher_smoke
from .smoke import run_smoke


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for local research commands."""

    parser = argparse.ArgumentParser(prog="python -m jorjor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke_parser = subparsers.add_parser("smoke", help="run the v0.0 local smoke test")
    smoke_parser.add_argument(
        "--config",
        default="configs/smoke.json",
        help="path to a smoke-test JSON config",
    )
    smoke_parser.add_argument(
        "--artifact-dir",
        default=None,
        help="optional output directory override",
    )

    leduc_parser = subparsers.add_parser(
        "leduc-random-smoke",
        help="run the v0.1.0 Leduc random-data smoke test",
    )
    leduc_parser.add_argument(
        "--config",
        default="configs/leduc_random_smoke.json",
        help="path to a Leduc random-data JSON config",
    )
    leduc_parser.add_argument(
        "--artifact-dir",
        default=None,
        help="optional output directory override",
    )

    baseline_parser = subparsers.add_parser(
        "leduc-random-baseline",
        help="run the v0.1.1 Leduc random baseline smoke test",
    )
    baseline_parser.add_argument(
        "--config",
        default="configs/leduc_random_baseline.json",
        help="path to a Leduc random baseline JSON config",
    )
    baseline_parser.add_argument(
        "--artifact-dir",
        default=None,
        help="optional output directory override",
    )

    train_parser = subparsers.add_parser(
        "train-leduc-tiny-smoke",
        help="run the v0.1.2 tiny Leduc training smoke",
    )
    train_parser.add_argument(
        "--config",
        default="configs/train/leduc_tiny_smoke.json",
        help="path to a tiny Leduc training JSON config",
    )
    train_parser.add_argument(
        "--artifact-dir",
        default=None,
        help="optional output directory override",
    )

    limit_train_parser = subparsers.add_parser(
        "train-limit-tiny-smoke",
        help="run the v0.4.1 tiny Limit Hold'em training smoke",
    )
    limit_train_parser.add_argument(
        "--config",
        default="configs/train/limit_tiny_random_smoke.json",
        help="path to a tiny Limit training JSON config",
    )
    limit_train_parser.add_argument(
        "--artifact-dir",
        default=None,
        help="optional output directory override",
    )

    limit_ar_parser = subparsers.add_parser(
        "train-limit-ar-smoke",
        help="run the v0.4.6 tiny Limit AR training smoke",
    )
    limit_ar_parser.add_argument(
        "--config",
        default="configs/train/limit_ar_teacher_smoke.json",
        help="path to a tiny Limit AR training JSON config",
    )
    limit_ar_parser.add_argument(
        "--artifact-dir",
        default=None,
        help="optional output directory override",
    )

    teacher_parser = subparsers.add_parser(
        "leduc-teacher-smoke",
        help="run the v0.2.0 Leduc heuristic teacher smoke test",
    )
    teacher_parser.add_argument(
        "--config",
        default="configs/data/leduc_teacher_smoke.json",
        help="path to a Leduc teacher JSON config",
    )
    teacher_parser.add_argument(
        "--artifact-dir",
        default=None,
        help="optional output directory override",
    )

    cfr_teacher_parser = subparsers.add_parser(
        "leduc-cfr-teacher-smoke",
        help="run the v0.5.0 Leduc tabular CFR teacher smoke test",
    )
    cfr_teacher_parser.add_argument(
        "--config",
        default="configs/data/leduc_cfr_teacher_smoke.json",
        help="path to a Leduc CFR teacher JSON config",
    )
    cfr_teacher_parser.add_argument(
        "--artifact-dir",
        default=None,
        help="optional output directory override",
    )

    limit_parser = subparsers.add_parser(
        "limit-random-smoke",
        help="run the v0.3.1 Limit Hold'em random adapter smoke test",
    )
    limit_parser.add_argument(
        "--config",
        default="configs/data/limit_random_smoke.json",
        help="path to a Limit random-data JSON config",
    )
    limit_parser.add_argument(
        "--artifact-dir",
        default=None,
        help="optional output directory override",
    )

    limit_abstraction_parser = subparsers.add_parser(
        "limit-abstraction-audit",
        help="run the v0.6.0 Limit abstraction audit",
    )
    limit_abstraction_parser.add_argument(
        "--config",
        default="configs/data/limit_abstraction_audit_smoke.json",
        help="path to a Limit abstraction audit JSON config",
    )
    limit_abstraction_parser.add_argument(
        "--artifact-dir",
        default=None,
        help="optional output directory override",
    )

    limit_abstract_cfr_parser = subparsers.add_parser(
        "limit-abstract-cfr-teacher-smoke",
        help="run the v0.6.1 Limit abstract CFR teacher smoke test",
    )
    limit_abstract_cfr_parser.add_argument(
        "--config",
        default="configs/data/limit_abstract_cfr_teacher_smoke.json",
        help="path to a Limit abstract CFR teacher JSON config",
    )
    limit_abstract_cfr_parser.add_argument(
        "--artifact-dir",
        default=None,
        help="optional output directory override",
    )

    limit_loader_parser = subparsers.add_parser(
        "limit-loader-smoke",
        help="run the v0.3.2 Limit tokenizer/loader smoke test",
    )
    limit_loader_parser.add_argument(
        "--config",
        default="configs/data/limit_loader_smoke.json",
        help="path to a Limit loader JSON config",
    )
    limit_loader_parser.add_argument(
        "--artifact-dir",
        default=None,
        help="optional output directory override",
    )

    limit_baseline_parser = subparsers.add_parser(
        "limit-baseline-smoke",
        help="run the v0.4.4 Limit card-removal baseline smoke test",
    )
    limit_baseline_parser.add_argument(
        "--config",
        default="configs/data/limit_baseline_smoke.json",
        help="path to a Limit baseline JSON config",
    )
    limit_baseline_parser.add_argument(
        "--artifact-dir",
        default=None,
        help="optional output directory override",
    )

    limit_prior_parser = subparsers.add_parser(
        "limit-prior-baseline",
        help="run the v0.4.12 Limit empirical prior baselines",
    )
    limit_prior_parser.add_argument(
        "--config",
        default="configs/data/limit_prior_teacher_10k.json",
        help="path to a Limit prior baseline JSON config",
    )
    limit_prior_parser.add_argument(
        "--artifact-dir",
        default=None,
        help="optional output directory override",
    )

    limit_seed_sweep_parser = subparsers.add_parser(
        "limit-seed-sweep",
        help="run the v0.4.13 Limit AR-vs-prior seed sweep report",
    )
    limit_seed_sweep_parser.add_argument(
        "--config",
        default="configs/limit_seed_sweep_10k.json",
        help="path to a Limit seed sweep JSON config",
    )
    limit_seed_sweep_parser.add_argument(
        "--artifact-dir",
        default=None,
        help="optional output directory override",
    )

    limit_teacher_parser = subparsers.add_parser(
        "limit-teacher-smoke",
        help="run the v0.4.2 Limit heuristic teacher smoke test",
    )
    limit_teacher_parser.add_argument(
        "--config",
        default="configs/data/limit_teacher_smoke.json",
        help="path to a Limit teacher JSON config",
    )
    limit_teacher_parser.add_argument(
        "--artifact-dir",
        default=None,
        help="optional output directory override",
    )

    probe_parser = subparsers.add_parser(
        "probe-leduc-tiny",
        help="run the v0.2.2 tiny Leduc layer-wise probe smoke",
    )
    probe_parser.add_argument(
        "--config",
        default="configs/probe/leduc_tiny_teacher_opponent_card.json",
        help="path to a tiny Leduc probe JSON config",
    )
    probe_parser.add_argument(
        "--artifact-dir",
        default=None,
        help="optional output directory override",
    )

    limit_probe_parser = subparsers.add_parser(
        "probe-limit-ar",
        help="run the v0.7.0 Limit AR layer-wise probe",
    )
    limit_probe_parser.add_argument(
        "--config",
        default="configs/probe/limit_ar_teacher_range_smoke.json",
        help="path to a Limit AR probe JSON config",
    )
    limit_probe_parser.add_argument(
        "--artifact-dir",
        default=None,
        help="optional output directory override",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a JorJor command."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "smoke":
        result = run_smoke(
            config_path=Path(args.config),
            artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        )
        print(
            "smoke ok "
            f"seed={result.seed} "
            f"samples={result.sample_count} "
            f"checksum={result.checksum} "
            f"artifact={result.artifact_path}"
        )
        return 0

    if args.command == "train-leduc-tiny-smoke":
        from .train.leduc_tiny import run_leduc_tiny_train

        result = run_leduc_tiny_train(
            config_path=Path(args.config),
            artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        )
        print(
            "leduc tiny train smoke ok "
            f"train_points={result.train_points} "
            f"validation_points={result.validation_points} "
            f"test_points={result.test_points} "
            f"first_loss={result.first_loss:.6f} "
            f"last_loss={result.last_loss:.6f} "
            f"policy_accuracy={result.policy_accuracy:.6f} "
            f"value_mse={result.value_mse:.6f} "
            f"opponent_card_accuracy={result.opponent_card_accuracy:.6f} "
            f"baseline_accuracy={result.baseline_accuracy:.6f} "
            f"baseline_cross_entropy={result.baseline_cross_entropy:.6f} "
            f"device={result.device} "
            f"checksum={result.checksum} "
            f"artifact_dir={result.artifact_dir}"
        )
        return 0

    if args.command == "train-limit-tiny-smoke":
        from .train.limit_tiny import run_limit_tiny_train

        result = run_limit_tiny_train(
            config_path=Path(args.config),
            artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        )
        print(
            "limit tiny train smoke ok "
            f"train_points={result.train_points} "
            f"validation_points={result.validation_points} "
            f"test_points={result.test_points} "
            f"first_loss={result.first_loss:.6f} "
            f"last_loss={result.last_loss:.6f} "
            f"policy_accuracy={result.policy_accuracy:.6f} "
            f"value_mse={result.value_mse:.6f} "
            f"opponent_hand_accuracy={result.opponent_hand_accuracy:.6f} "
            f"opponent_hand_top5_accuracy={result.opponent_hand_top5_accuracy:.6f} "
            f"opponent_hand_top10_accuracy={result.opponent_hand_top10_accuracy:.6f} "
            f"device={result.device} "
            f"checksum={result.checksum} "
            f"artifact_dir={result.artifact_dir}"
        )
        return 0

    if args.command == "train-limit-ar-smoke":
        from .train.limit_ar import run_limit_ar_train

        result = run_limit_ar_train(
            config_path=Path(args.config),
            artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        )
        print(
            "limit ar train smoke ok "
            f"train_points={result.train_points} "
            f"validation_points={result.validation_points} "
            f"test_points={result.test_points} "
            f"first_loss={result.first_loss:.6f} "
            f"last_loss={result.last_loss:.6f} "
            f"policy_accuracy={result.policy_accuracy:.6f} "
            f"value_mse={result.value_mse:.6f} "
            f"opponent_hand_accuracy={result.opponent_hand_accuracy:.6f} "
            f"opponent_hand_top5_accuracy={result.opponent_hand_top5_accuracy:.6f} "
            f"opponent_hand_top10_accuracy={result.opponent_hand_top10_accuracy:.6f} "
            f"device={result.device} "
            f"checksum={result.checksum} "
            f"artifact_dir={result.artifact_dir}"
        )
        return 0

    if args.command == "leduc-teacher-smoke":
        result = run_leduc_teacher_smoke(
            config_path=Path(args.config),
            artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        )
        print(
            "leduc teacher smoke ok "
            f"seed={result.seed} "
            f"hands={result.hand_count} "
            f"decision_points={result.decision_point_count} "
            f"illegal_actions={result.illegal_action_count} "
            f"leakage={result.leakage_count} "
            f"teacher_win_rate={result.teacher_win_rate:.6f} "
            f"teacher_average_return={result.teacher_average_return:.6f} "
            f"checksum={result.checksum} "
            f"artifact_dir={result.artifact_dir}"
        )
        return 0

    if args.command == "leduc-cfr-teacher-smoke":
        result = run_leduc_cfr_teacher_smoke(
            config_path=Path(args.config),
            artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        )
        print(
            "leduc cfr teacher smoke ok "
            f"hands={result.hand_count} "
            f"decision_points={result.decision_point_count} "
            f"illegal_actions={result.illegal_action_count} "
            f"leakage={result.leakage_count} "
            f"infosets={result.information_set_count} "
            f"teacher_win_rate={result.teacher_win_rate:.6f} "
            f"teacher_average_return={result.teacher_average_return:.6f} "
            f"checksum={result.checksum} "
            f"artifact_dir={result.artifact_dir}"
        )
        return 0

    if args.command == "limit-random-smoke":
        result = run_limit_random_smoke(
            config_path=Path(args.config),
            artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        )
        print(
            "limit random smoke ok "
            f"seed={result.seed} "
            f"hands={result.hand_count} "
            f"decision_points={result.decision_point_count} "
            f"illegal_actions={result.illegal_action_count} "
            f"leakage={result.leakage_count} "
            f"invalid_opponent_labels={result.invalid_opponent_label_count} "
            f"hand_label_classes={result.hand_label_class_count} "
            f"checksum={result.checksum} "
            f"artifact_dir={result.artifact_dir}"
        )
        return 0

    if args.command == "limit-abstraction-audit":
        result = run_limit_abstraction_audit(
            config_path=Path(args.config),
            artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        )
        print(
            "limit abstraction audit ok "
            f"hands={result.hand_count} "
            f"decision_points={result.decision_point_count} "
            f"illegal_actions={result.illegal_action_count} "
            f"leakage={result.leakage_count} "
            f"infosets={result.information_set_count} "
            f"recommendation={result.recommendation} "
            f"checksum={result.checksum} "
            f"artifact_dir={result.artifact_dir}"
        )
        return 0

    if args.command == "limit-abstract-cfr-teacher-smoke":
        result = run_limit_abstract_cfr_teacher_smoke(
            config_path=Path(args.config),
            artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        )
        print(
            "limit abstract cfr teacher smoke ok "
            f"hands={result.hand_count} "
            f"decision_points={result.decision_point_count} "
            f"illegal_actions={result.illegal_action_count} "
            f"leakage={result.leakage_count} "
            f"infosets={result.information_set_count} "
            f"teacher_win_rate={result.teacher_win_rate:.6f} "
            f"teacher_average_return={result.teacher_average_return:.6f} "
            f"checksum={result.checksum} "
            f"artifact_dir={result.artifact_dir}"
        )
        return 0

    if args.command == "limit-loader-smoke":
        result = run_limit_loader_smoke(
            config_path=Path(args.config),
            artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        )
        print(
            "limit loader smoke ok "
            f"hands={result.hand_count} "
            f"decision_points={result.decision_point_count} "
            f"train_rows={result.train_rows} "
            f"validation_rows={result.validation_rows} "
            f"test_rows={result.test_rows} "
            f"vocab_size={result.vocab_size} "
            f"action_size={result.action_size} "
            f"opponent_hand_size={result.opponent_hand_size} "
            f"split_overlap={result.split_overlap} "
            f"checksum={result.checksum} "
            f"artifact_dir={result.artifact_dir}"
        )
        return 0

    if args.command == "limit-baseline-smoke":
        result = run_limit_baseline_smoke(
            config_path=Path(args.config),
            artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        )
        print(
            "limit baseline smoke ok "
            f"hands={result.hand_count} "
            f"decision_points={result.decision_point_count} "
            f"illegal_actions={result.illegal_action_count} "
            f"leakage={result.leakage_count} "
            f"top1={result.top1_accuracy:.6f} "
            f"top5={result.top5_accuracy:.6f} "
            f"top10={result.top10_accuracy:.6f} "
            f"cross_entropy={result.cross_entropy:.6f} "
            f"checksum={result.checksum} "
            f"artifact_dir={result.artifact_dir}"
        )
        return 0

    if args.command == "limit-prior-baseline":
        result = run_limit_prior_baseline(
            config_path=Path(args.config),
            artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        )
        best_top10 = max(metrics.top10_accuracy for metrics in result.baselines.values())
        print(
            "limit prior baseline ok "
            f"hands={result.hand_count} "
            f"decision_points={result.decision_point_count} "
            f"illegal_actions={result.illegal_action_count} "
            f"leakage={result.leakage_count} "
            f"best_top10={best_top10:.6f} "
            f"checksum={result.checksum} "
            f"artifact_dir={result.artifact_dir}"
        )
        return 0

    if args.command == "limit-seed-sweep":
        result = run_limit_seed_sweep(
            config_path=Path(args.config),
            artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        )
        print(
            "limit seed sweep ok "
            f"completed={result.completed_count} "
            f"missing={result.missing_count} "
            f"checksum={result.checksum} "
            f"artifact_dir={result.artifact_dir}"
        )
        return 0

    if args.command == "limit-teacher-smoke":
        result = run_limit_teacher_smoke(
            config_path=Path(args.config),
            artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        )
        print(
            "limit teacher smoke ok "
            f"seed={result.seed} "
            f"hands={result.hand_count} "
            f"decision_points={result.decision_point_count} "
            f"illegal_actions={result.illegal_action_count} "
            f"leakage={result.leakage_count} "
            f"teacher_win_rate={result.teacher_win_rate:.6f} "
            f"teacher_average_return={result.teacher_average_return:.6f} "
            f"checksum={result.checksum} "
            f"artifact_dir={result.artifact_dir}"
        )
        return 0

    if args.command == "probe-leduc-tiny":
        from .probe.leduc_tiny import run_leduc_tiny_probe

        result = run_leduc_tiny_probe(
            config_path=Path(args.config),
            artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        )
        print(
            "leduc tiny probe ok "
            f"layers={result.layer_count} "
            f"metric={result.metric_name} "
            f"best_real_score={result.best_real_score:.6f} "
            f"best_control_score={result.best_control_score:.6f} "
            f"device={result.device} "
            f"checksum={result.checksum} "
            f"artifact_dir={result.artifact_dir}"
        )
        return 0

    if args.command == "probe-limit-ar":
        from .probe.limit_ar import run_limit_ar_probe

        result = run_limit_ar_probe(
            config_path=Path(args.config),
            artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        )
        print(
            "limit ar probe ok "
            f"layers={result.layer_count} "
            f"metric={result.metric_name} "
            f"best_real_score={result.best_real_score:.6f} "
            f"best_control_score={result.best_control_score:.6f} "
            f"device={result.device} "
            f"checksum={result.checksum} "
            f"artifact_dir={result.artifact_dir}"
        )
        return 0

    if args.command == "leduc-random-baseline":
        result = run_leduc_random_baseline(
            config_path=Path(args.config),
            artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        )
        print(
            "leduc random baseline ok "
            f"seed={result.seed} "
            f"hands={result.hand_count} "
            f"decision_points={result.decision_point_count} "
            f"illegal_actions={result.illegal_action_count} "
            f"leakage={result.leakage_count} "
            f"split_overlap={result.split_overlap} "
            f"test_points={result.test_count} "
            f"baseline_accuracy={result.baseline_accuracy:.6f} "
            f"baseline_cross_entropy={result.baseline_cross_entropy:.6f} "
            f"checksum={result.checksum} "
            f"artifact_dir={result.artifact_dir}"
        )
        return 0

    if args.command == "leduc-random-smoke":
        result = run_leduc_random_smoke(
            config_path=Path(args.config),
            artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
        )
        print(
            "leduc random smoke ok "
            f"seed={result.seed} "
            f"hands={result.hand_count} "
            f"decision_points={result.decision_point_count} "
            f"illegal_actions={result.illegal_action_count} "
            f"leakage={result.leakage_count} "
            f"checksum={result.checksum} "
            f"artifact_dir={result.artifact_dir}"
        )
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2
