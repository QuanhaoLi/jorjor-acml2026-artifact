from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from jorjor.train.limit_controls import LIMIT_INPUT_CONTROLS

LIMIT_PROBE_FEATURE_CONTROLS = {
    "none",
    "action_only",
    "value_only",
    "action_value_only",
    "history_composition_only",
    "action_value_history_composition_only",
    "round_position_action_only",
    "residualize_action",
    "residualize_value",
    "residualize_action_value",
    "residualize_action_value_history_composition",
}

LIMIT_PROBE_NUISANCE_ONLY_CONTROLS = {
    "action_only",
    "value_only",
    "action_value_only",
    "history_composition_only",
    "action_value_history_composition_only",
    "round_position_action_only",
}

LIMIT_PROBE_LABEL_CONTROLS = {
    "none",
    "shuffle_global",
    "shuffle_within_bucket",
}

LIMIT_PROBE_FAMILIES = {
    "linear",
    "mlp",
}


@dataclass(frozen=True)
class SmokeConfig:
    """Configuration for the v0.0 deterministic smoke test."""

    seed: int
    sample_count: int
    artifact_dir: Path


@dataclass(frozen=True)
class LeducRandomConfig:
    """Configuration for the v0.1.0 Leduc random-data smoke test."""

    seed: int
    hand_count: int
    artifact_dir: Path


@dataclass(frozen=True)
class LeducRandomBaselineConfig:
    """Configuration for the v0.1.1 Leduc random baseline smoke test."""

    seed: int
    hand_count: int
    artifact_dir: Path
    split_seed: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float


@dataclass(frozen=True)
class LeducTinyTrainConfig:
    """Configuration for versioned tiny Leduc training runs."""

    run_version: str
    data_source: str
    seed: int
    data_seed: int
    split_seed: int
    hand_count: int
    artifact_dir: Path
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    epochs: int
    learning_rate: float
    hidden_size: int
    batch_size: int
    device: str
    save_model: bool


@dataclass(frozen=True)
class LeducTeacherConfig:
    """Configuration for v0.2.0 Leduc teacher data smoke."""

    run_version: str
    seed: int
    hand_count: int
    artifact_dir: Path


@dataclass(frozen=True)
class LeducCfrTeacherConfig:
    """Configuration for v0.5.0 Leduc tabular CFR teacher smoke."""

    run_version: str
    seed: int
    iteration_count: int
    hand_count: int
    evaluation_hand_count: int
    artifact_dir: Path


@dataclass(frozen=True)
class LimitRandomConfig:
    """Configuration for v0.3.1 Limit random-data smoke."""

    run_version: str
    seed: int
    hand_count: int
    artifact_dir: Path


@dataclass(frozen=True)
class LimitAbstractionAuditConfig:
    """Configuration for v0.6.0 Limit abstraction audit."""

    run_version: str
    data_source: str
    seed: int
    hand_count: int
    artifact_dir: Path


@dataclass(frozen=True)
class LimitAbstractCfrTeacherConfig:
    """Configuration for v0.6.1 Limit abstract CFR teacher smoke."""

    run_version: str
    seed: int
    iteration_count: int
    deal_sample_count: int
    hand_count: int
    evaluation_hand_count: int
    artifact_dir: Path


@dataclass(frozen=True)
class LimitLoaderConfig:
    """Configuration for v0.3.2 Limit tokenizer/loader smoke."""

    run_version: str
    seed: int
    hand_count: int
    artifact_dir: Path
    split_seed: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float


@dataclass(frozen=True)
class LimitTeacherConfig:
    """Configuration for v0.4.2 Limit teacher data smoke."""

    run_version: str
    seed: int
    hand_count: int
    evaluation_hand_count: int
    artifact_dir: Path


@dataclass(frozen=True)
class LimitBaselineConfig:
    """Configuration for v0.4.4 Limit card-removal baseline smoke."""

    run_version: str
    data_source: str
    seed: int
    hand_count: int
    artifact_dir: Path


@dataclass(frozen=True)
class LimitPriorBaselineConfig:
    """Configuration for v0.4.12 Limit empirical prior baselines."""

    run_version: str
    data_source: str
    seed: int
    hand_count: int
    artifact_dir: Path
    split_seed: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float


@dataclass(frozen=True)
class LimitSeedSweepSeedConfig:
    """One seed row in a v0.4.13 Limit AR-vs-prior sweep."""

    name: str
    ar_config: Path
    prior_config: Path


@dataclass(frozen=True)
class LimitSeedSweepConfig:
    """Configuration for v0.4.13 Limit multi-seed sweep reports."""

    run_version: str
    artifact_dir: Path
    run_missing_priors: bool
    seeds: tuple[LimitSeedSweepSeedConfig, ...]


@dataclass(frozen=True)
class LimitTinyTrainConfig:
    """Configuration for v0.4.1 Limit tiny training runs."""

    run_version: str
    data_source: str
    seed: int
    data_seed: int
    split_seed: int
    hand_count: int
    artifact_dir: Path
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    epochs: int
    learning_rate: float
    hidden_size: int
    batch_size: int
    device: str
    save_model: bool


@dataclass(frozen=True)
class LimitArTrainConfig:
    """Configuration for v0.4.6 Limit tiny AR training runs."""

    run_version: str
    data_source: str
    seed: int
    data_seed: int
    split_seed: int
    hand_count: int
    artifact_dir: Path
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    epochs: int
    learning_rate: float
    hidden_size: int
    num_heads: int
    num_layers: int
    input_control: str
    range_label_control: str
    range_label_seed: int
    policy_loss_weight: float
    value_loss_weight: float
    opponent_hand_loss_weight: float
    abstract_cfr_iteration_count: int
    abstract_cfr_deal_sample_count: int
    abstract_cfr_policy_path: Path | None
    batch_size: int
    device: str
    save_model: bool


@dataclass(frozen=True)
class LeducTinyProbeConfig:
    """Configuration for v0.2.2 tiny Leduc probe smoke."""

    run_version: str
    train_config: Path
    checkpoint_path: Path
    artifact_dir: Path
    target: str
    input_control: str
    layers: tuple[str, ...]
    probe_seed: int
    control_seed: int
    epochs: int
    learning_rate: float
    batch_size: int
    device: str
    random_label_control: bool


@dataclass(frozen=True)
class LimitArProbeConfig:
    """Configuration for v0.7.0 Limit AR layer probes."""

    run_version: str
    train_config: Path
    checkpoint_path: Path
    artifact_dir: Path
    target: str
    input_control: str
    feature_control: str
    label_control: str
    layers: tuple[str, ...]
    probe_seed: int
    control_seed: int
    epochs: int
    learning_rate: float
    batch_size: int
    device: str
    random_label_control: bool
    probe_family: str
    mlp_hidden_size: int


def _require_int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _require_float(raw: dict[str, Any], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"{key} must be a number")


def load_smoke_config(path: Path) -> SmokeConfig:
    """Load and validate the v0.0 smoke-test config."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("smoke config must be a JSON object")

    seed = _require_int(raw, "seed")
    sample_count = _require_int(raw, "sample_count")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    artifact_dir = raw.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise ValueError("artifact_dir must be a non-empty string")

    return SmokeConfig(
        seed=seed,
        sample_count=sample_count,
        artifact_dir=Path(artifact_dir),
    )


def load_leduc_random_config(path: Path) -> LeducRandomConfig:
    """Load and validate the v0.1.0 Leduc random-data config."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Leduc random config must be a JSON object")

    seed = _require_int(raw, "seed")
    hand_count = _require_int(raw, "hand_count")
    if hand_count <= 0:
        raise ValueError("hand_count must be positive")

    artifact_dir = raw.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise ValueError("artifact_dir must be a non-empty string")

    return LeducRandomConfig(
        seed=seed,
        hand_count=hand_count,
        artifact_dir=Path(artifact_dir),
    )


def load_leduc_random_baseline_config(path: Path) -> LeducRandomBaselineConfig:
    """Load and validate the v0.1.1 Leduc baseline config."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Leduc random baseline config must be a JSON object")

    seed = _require_int(raw, "seed")
    hand_count = _require_int(raw, "hand_count")
    split_seed = _require_int(raw, "split_seed")
    if hand_count <= 0:
        raise ValueError("hand_count must be positive")

    train_fraction = _require_float(raw, "train_fraction")
    validation_fraction = _require_float(raw, "validation_fraction")
    test_fraction = _require_float(raw, "test_fraction")
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1.0")

    artifact_dir = raw.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise ValueError("artifact_dir must be a non-empty string")

    return LeducRandomBaselineConfig(
        seed=seed,
        hand_count=hand_count,
        artifact_dir=Path(artifact_dir),
        split_seed=split_seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )


def load_leduc_tiny_train_config(path: Path) -> LeducTinyTrainConfig:
    """Load and validate the tiny training config."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Leduc tiny train config must be a JSON object")

    run_version = raw.get("run_version")
    if not isinstance(run_version, str) or not run_version:
        raise ValueError("run_version must be a non-empty string")
    data_source = raw.get("data_source")
    if data_source not in {"random", "teacher", "cfr_teacher"}:
        raise ValueError("data_source must be random, teacher, or cfr_teacher")
    seed = _require_int(raw, "seed")
    data_seed = _require_int(raw, "data_seed")
    split_seed = _require_int(raw, "split_seed")
    hand_count = _require_int(raw, "hand_count")
    epochs = _require_int(raw, "epochs")
    hidden_size = _require_int(raw, "hidden_size")
    batch_size = _require_int(raw, "batch_size")
    learning_rate = _require_float(raw, "learning_rate")
    train_fraction = _require_float(raw, "train_fraction")
    validation_fraction = _require_float(raw, "validation_fraction")
    test_fraction = _require_float(raw, "test_fraction")
    if hand_count <= 0:
        raise ValueError("hand_count must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1.0")

    artifact_dir = raw.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise ValueError("artifact_dir must be a non-empty string")
    device = raw.get("device", "auto")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    save_model = raw.get("save_model", True)
    if not isinstance(save_model, bool):
        raise ValueError("save_model must be a boolean")

    return LeducTinyTrainConfig(
        run_version=run_version,
        data_source=data_source,
        seed=seed,
        data_seed=data_seed,
        split_seed=split_seed,
        hand_count=hand_count,
        artifact_dir=Path(artifact_dir),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        batch_size=batch_size,
        device=device,
        save_model=save_model,
    )


def load_leduc_teacher_config(path: Path) -> LeducTeacherConfig:
    """Load and validate the Leduc teacher generation config."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Leduc teacher config must be a JSON object")

    run_version = raw.get("run_version")
    if not isinstance(run_version, str) or not run_version:
        raise ValueError("run_version must be a non-empty string")
    seed = _require_int(raw, "seed")
    hand_count = _require_int(raw, "hand_count")
    if hand_count <= 0:
        raise ValueError("hand_count must be positive")

    artifact_dir = raw.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise ValueError("artifact_dir must be a non-empty string")

    return LeducTeacherConfig(
        run_version=run_version,
        seed=seed,
        hand_count=hand_count,
        artifact_dir=Path(artifact_dir),
    )


def load_leduc_cfr_teacher_config(path: Path) -> LeducCfrTeacherConfig:
    """Load and validate a Leduc tabular CFR teacher config."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Leduc CFR teacher config must be a JSON object")

    run_version = raw.get("run_version")
    if not isinstance(run_version, str) or not run_version:
        raise ValueError("run_version must be a non-empty string")
    seed = _require_int(raw, "seed")
    iteration_count = _require_int(raw, "iteration_count")
    hand_count = _require_int(raw, "hand_count")
    evaluation_hand_count = _require_int(raw, "evaluation_hand_count")
    if iteration_count <= 0:
        raise ValueError("iteration_count must be positive")
    if hand_count <= 0:
        raise ValueError("hand_count must be positive")
    if evaluation_hand_count <= 0:
        raise ValueError("evaluation_hand_count must be positive")

    artifact_dir = raw.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise ValueError("artifact_dir must be a non-empty string")

    return LeducCfrTeacherConfig(
        run_version=run_version,
        seed=seed,
        iteration_count=iteration_count,
        hand_count=hand_count,
        evaluation_hand_count=evaluation_hand_count,
        artifact_dir=Path(artifact_dir),
    )


def load_limit_random_config(path: Path) -> LimitRandomConfig:
    """Load and validate a Limit random-data config."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Limit random config must be a JSON object")

    run_version = raw.get("run_version")
    if not isinstance(run_version, str) or not run_version:
        raise ValueError("run_version must be a non-empty string")
    seed = _require_int(raw, "seed")
    hand_count = _require_int(raw, "hand_count")
    if hand_count <= 0:
        raise ValueError("hand_count must be positive")

    artifact_dir = raw.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise ValueError("artifact_dir must be a non-empty string")

    return LimitRandomConfig(
        run_version=run_version,
        seed=seed,
        hand_count=hand_count,
        artifact_dir=Path(artifact_dir),
    )


def load_limit_abstraction_audit_config(path: Path) -> LimitAbstractionAuditConfig:
    """Load and validate a Limit abstraction audit config."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Limit abstraction audit config must be a JSON object")

    run_version = raw.get("run_version")
    if not isinstance(run_version, str) or not run_version:
        raise ValueError("run_version must be a non-empty string")
    data_source = raw.get("data_source")
    if data_source not in {"random", "teacher", "abstract_cfr_teacher"}:
        raise ValueError("data_source must be random, teacher, or abstract_cfr_teacher")
    seed = _require_int(raw, "seed")
    hand_count = _require_int(raw, "hand_count")
    if hand_count <= 0:
        raise ValueError("hand_count must be positive")

    artifact_dir = raw.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise ValueError("artifact_dir must be a non-empty string")

    return LimitAbstractionAuditConfig(
        run_version=run_version,
        data_source=data_source,
        seed=seed,
        hand_count=hand_count,
        artifact_dir=Path(artifact_dir),
    )


def load_limit_abstract_cfr_teacher_config(path: Path) -> LimitAbstractCfrTeacherConfig:
    """Load and validate a Limit abstract CFR teacher config."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Limit abstract CFR teacher config must be a JSON object")

    run_version = raw.get("run_version")
    if not isinstance(run_version, str) or not run_version:
        raise ValueError("run_version must be a non-empty string")
    seed = _require_int(raw, "seed")
    iteration_count = _require_int(raw, "iteration_count")
    deal_sample_count = _require_int(raw, "deal_sample_count")
    hand_count = _require_int(raw, "hand_count")
    evaluation_hand_count = _require_int(raw, "evaluation_hand_count")
    if iteration_count <= 0:
        raise ValueError("iteration_count must be positive")
    if deal_sample_count <= 0:
        raise ValueError("deal_sample_count must be positive")
    if hand_count <= 0:
        raise ValueError("hand_count must be positive")
    if evaluation_hand_count <= 0:
        raise ValueError("evaluation_hand_count must be positive")

    artifact_dir = raw.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise ValueError("artifact_dir must be a non-empty string")

    return LimitAbstractCfrTeacherConfig(
        run_version=run_version,
        seed=seed,
        iteration_count=iteration_count,
        deal_sample_count=deal_sample_count,
        hand_count=hand_count,
        evaluation_hand_count=evaluation_hand_count,
        artifact_dir=Path(artifact_dir),
    )


def load_limit_loader_config(path: Path) -> LimitLoaderConfig:
    """Load and validate a Limit tokenizer/loader smoke config."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Limit loader config must be a JSON object")

    run_version = raw.get("run_version")
    if not isinstance(run_version, str) or not run_version:
        raise ValueError("run_version must be a non-empty string")
    seed = _require_int(raw, "seed")
    split_seed = _require_int(raw, "split_seed")
    hand_count = _require_int(raw, "hand_count")
    if hand_count <= 0:
        raise ValueError("hand_count must be positive")

    train_fraction = _require_float(raw, "train_fraction")
    validation_fraction = _require_float(raw, "validation_fraction")
    test_fraction = _require_float(raw, "test_fraction")
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1.0")

    artifact_dir = raw.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise ValueError("artifact_dir must be a non-empty string")

    return LimitLoaderConfig(
        run_version=run_version,
        seed=seed,
        hand_count=hand_count,
        artifact_dir=Path(artifact_dir),
        split_seed=split_seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )


def load_limit_tiny_train_config(path: Path) -> LimitTinyTrainConfig:
    """Load and validate a Limit tiny training config."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Limit tiny train config must be a JSON object")

    run_version = raw.get("run_version")
    if not isinstance(run_version, str) or not run_version:
        raise ValueError("run_version must be a non-empty string")
    data_source = raw.get("data_source")
    if data_source not in {"random", "teacher", "abstract_cfr_teacher"}:
        raise ValueError("data_source must be random, teacher, or abstract_cfr_teacher")
    seed = _require_int(raw, "seed")
    data_seed = _require_int(raw, "data_seed")
    split_seed = _require_int(raw, "split_seed")
    hand_count = _require_int(raw, "hand_count")
    epochs = _require_int(raw, "epochs")
    hidden_size = _require_int(raw, "hidden_size")
    batch_size = _require_int(raw, "batch_size")
    learning_rate = _require_float(raw, "learning_rate")
    train_fraction = _require_float(raw, "train_fraction")
    validation_fraction = _require_float(raw, "validation_fraction")
    test_fraction = _require_float(raw, "test_fraction")
    if hand_count <= 0:
        raise ValueError("hand_count must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1.0")

    artifact_dir = raw.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise ValueError("artifact_dir must be a non-empty string")
    device = raw.get("device", "auto")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    save_model = raw.get("save_model", True)
    if not isinstance(save_model, bool):
        raise ValueError("save_model must be a boolean")

    return LimitTinyTrainConfig(
        run_version=run_version,
        data_source=data_source,
        seed=seed,
        data_seed=data_seed,
        split_seed=split_seed,
        hand_count=hand_count,
        artifact_dir=Path(artifact_dir),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        batch_size=batch_size,
        device=device,
        save_model=save_model,
    )


def load_limit_teacher_config(path: Path) -> LimitTeacherConfig:
    """Load and validate a Limit teacher generation config."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Limit teacher config must be a JSON object")

    run_version = raw.get("run_version")
    if not isinstance(run_version, str) or not run_version:
        raise ValueError("run_version must be a non-empty string")
    seed = _require_int(raw, "seed")
    hand_count = _require_int(raw, "hand_count")
    evaluation_hand_count = _require_int(raw, "evaluation_hand_count")
    if hand_count <= 0:
        raise ValueError("hand_count must be positive")
    if evaluation_hand_count <= 0:
        raise ValueError("evaluation_hand_count must be positive")

    artifact_dir = raw.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise ValueError("artifact_dir must be a non-empty string")

    return LimitTeacherConfig(
        run_version=run_version,
        seed=seed,
        hand_count=hand_count,
        evaluation_hand_count=evaluation_hand_count,
        artifact_dir=Path(artifact_dir),
    )


def load_limit_baseline_config(path: Path) -> LimitBaselineConfig:
    """Load and validate a Limit card-removal baseline config."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Limit baseline config must be a JSON object")

    run_version = raw.get("run_version")
    if not isinstance(run_version, str) or not run_version:
        raise ValueError("run_version must be a non-empty string")
    data_source = raw.get("data_source")
    if data_source not in {"random", "teacher", "abstract_cfr_teacher"}:
        raise ValueError("data_source must be random, teacher, or abstract_cfr_teacher")
    seed = _require_int(raw, "seed")
    hand_count = _require_int(raw, "hand_count")
    if hand_count <= 0:
        raise ValueError("hand_count must be positive")

    artifact_dir = raw.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise ValueError("artifact_dir must be a non-empty string")

    return LimitBaselineConfig(
        run_version=run_version,
        data_source=data_source,
        seed=seed,
        hand_count=hand_count,
        artifact_dir=Path(artifact_dir),
    )


def load_limit_prior_baseline_config(path: Path) -> LimitPriorBaselineConfig:
    """Load and validate a Limit empirical prior baseline config."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Limit prior baseline config must be a JSON object")

    run_version = raw.get("run_version")
    if not isinstance(run_version, str) or not run_version:
        raise ValueError("run_version must be a non-empty string")
    data_source = raw.get("data_source")
    if data_source not in {"random", "teacher"}:
        raise ValueError("data_source must be random or teacher")
    seed = _require_int(raw, "seed")
    split_seed = _require_int(raw, "split_seed")
    hand_count = _require_int(raw, "hand_count")
    if hand_count <= 0:
        raise ValueError("hand_count must be positive")

    train_fraction = _require_float(raw, "train_fraction")
    validation_fraction = _require_float(raw, "validation_fraction")
    test_fraction = _require_float(raw, "test_fraction")
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1.0")

    artifact_dir = raw.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise ValueError("artifact_dir must be a non-empty string")

    return LimitPriorBaselineConfig(
        run_version=run_version,
        data_source=data_source,
        seed=seed,
        hand_count=hand_count,
        artifact_dir=Path(artifact_dir),
        split_seed=split_seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )


def load_limit_seed_sweep_config(path: Path) -> LimitSeedSweepConfig:
    """Load and validate a Limit multi-seed sweep config."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Limit seed sweep config must be a JSON object")

    run_version = raw.get("run_version")
    if not isinstance(run_version, str) or not run_version:
        raise ValueError("run_version must be a non-empty string")
    artifact_dir = raw.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise ValueError("artifact_dir must be a non-empty string")
    run_missing_priors = raw.get("run_missing_priors", False)
    if not isinstance(run_missing_priors, bool):
        raise ValueError("run_missing_priors must be a boolean")

    seed_rows = raw.get("seeds")
    if not isinstance(seed_rows, list) or not seed_rows:
        raise ValueError("seeds must be a non-empty list")
    seeds: list[LimitSeedSweepSeedConfig] = []
    seen_names: set[str] = set()
    for row in seed_rows:
        if not isinstance(row, dict):
            raise ValueError("each seed row must be a JSON object")
        name = row.get("name")
        ar_config = row.get("ar_config")
        prior_config = row.get("prior_config")
        if not isinstance(name, str) or not name:
            raise ValueError("seed name must be a non-empty string")
        if name in seen_names:
            raise ValueError("seed names must be unique")
        seen_names.add(name)
        if not isinstance(ar_config, str) or not ar_config:
            raise ValueError("ar_config must be a non-empty string")
        if not isinstance(prior_config, str) or not prior_config:
            raise ValueError("prior_config must be a non-empty string")
        seeds.append(
            LimitSeedSweepSeedConfig(
                name=name,
                ar_config=Path(ar_config),
                prior_config=Path(prior_config),
            )
        )

    return LimitSeedSweepConfig(
        run_version=run_version,
        artifact_dir=Path(artifact_dir),
        run_missing_priors=run_missing_priors,
        seeds=tuple(seeds),
    )


def load_limit_ar_train_config(path: Path) -> LimitArTrainConfig:
    """Load and validate a Limit tiny AR training config."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Limit AR train config must be a JSON object")

    run_version = raw.get("run_version")
    if not isinstance(run_version, str) or not run_version:
        raise ValueError("run_version must be a non-empty string")
    data_source = raw.get("data_source")
    if data_source not in {"random", "teacher", "abstract_cfr_teacher"}:
        raise ValueError("data_source must be random, teacher, or abstract_cfr_teacher")
    seed = _require_int(raw, "seed")
    data_seed = _require_int(raw, "data_seed")
    split_seed = _require_int(raw, "split_seed")
    hand_count = _require_int(raw, "hand_count")
    epochs = _require_int(raw, "epochs")
    hidden_size = _require_int(raw, "hidden_size")
    num_heads = _require_int(raw, "num_heads")
    num_layers = _require_int(raw, "num_layers")
    input_control = raw.get("input_control", "none")
    if input_control not in LIMIT_INPUT_CONTROLS:
        raise ValueError("input_control must be a known Limit AR input control")
    range_label_control = raw.get("range_label_control", "none")
    if range_label_control not in {"none", "shuffle_within_split"}:
        raise ValueError("range_label_control must be none or shuffle_within_split")
    range_label_seed = _require_int(raw, "range_label_seed") if "range_label_seed" in raw else 0
    policy_loss_weight = _require_float(raw, "policy_loss_weight") if "policy_loss_weight" in raw else 1.0
    value_loss_weight = _require_float(raw, "value_loss_weight") if "value_loss_weight" in raw else 0.3
    opponent_hand_loss_weight = (
        _require_float(raw, "opponent_hand_loss_weight") if "opponent_hand_loss_weight" in raw else 0.7
    )
    abstract_cfr_iteration_count = _require_int(raw, "abstract_cfr_iteration_count") if "abstract_cfr_iteration_count" in raw else 10
    abstract_cfr_deal_sample_count = (
        _require_int(raw, "abstract_cfr_deal_sample_count") if "abstract_cfr_deal_sample_count" in raw else 20
    )
    if abstract_cfr_iteration_count <= 0:
        raise ValueError("abstract_cfr_iteration_count must be positive")
    if abstract_cfr_deal_sample_count <= 0:
        raise ValueError("abstract_cfr_deal_sample_count must be positive")
    abstract_cfr_policy_path_raw = raw.get("abstract_cfr_policy_path")
    if abstract_cfr_policy_path_raw is None:
        abstract_cfr_policy_path = None
    elif isinstance(abstract_cfr_policy_path_raw, str) and abstract_cfr_policy_path_raw:
        abstract_cfr_policy_path = Path(abstract_cfr_policy_path_raw)
    else:
        raise ValueError("abstract_cfr_policy_path must be a non-empty string when provided")
    batch_size = _require_int(raw, "batch_size")
    learning_rate = _require_float(raw, "learning_rate")
    train_fraction = _require_float(raw, "train_fraction")
    validation_fraction = _require_float(raw, "validation_fraction")
    test_fraction = _require_float(raw, "test_fraction")
    if hand_count <= 0:
        raise ValueError("hand_count must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if hidden_size % num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if policy_loss_weight < 0 or value_loss_weight < 0 or opponent_hand_loss_weight < 0:
        raise ValueError("loss weights must be non-negative")
    if policy_loss_weight + value_loss_weight + opponent_hand_loss_weight <= 0:
        raise ValueError("at least one loss weight must be positive")
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1.0")

    artifact_dir = raw.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise ValueError("artifact_dir must be a non-empty string")
    device = raw.get("device", "auto")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    save_model = raw.get("save_model", True)
    if not isinstance(save_model, bool):
        raise ValueError("save_model must be a boolean")

    return LimitArTrainConfig(
        run_version=run_version,
        data_source=data_source,
        seed=seed,
        data_seed=data_seed,
        split_seed=split_seed,
        hand_count=hand_count,
        artifact_dir=Path(artifact_dir),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        input_control=input_control,
        range_label_control=range_label_control,
        range_label_seed=range_label_seed,
        policy_loss_weight=policy_loss_weight,
        value_loss_weight=value_loss_weight,
        opponent_hand_loss_weight=opponent_hand_loss_weight,
        abstract_cfr_iteration_count=abstract_cfr_iteration_count,
        abstract_cfr_deal_sample_count=abstract_cfr_deal_sample_count,
        abstract_cfr_policy_path=abstract_cfr_policy_path,
        batch_size=batch_size,
        device=device,
        save_model=save_model,
    )


def load_leduc_tiny_probe_config(path: Path) -> LeducTinyProbeConfig:
    """Load and validate a tiny Leduc probe config."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Leduc tiny probe config must be a JSON object")

    run_version = raw.get("run_version")
    if not isinstance(run_version, str) or not run_version:
        raise ValueError("run_version must be a non-empty string")
    train_config = raw.get("train_config")
    if not isinstance(train_config, str) or not train_config:
        raise ValueError("train_config must be a non-empty string")
    checkpoint_path = raw.get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError("checkpoint_path must be a non-empty string")
    artifact_dir = raw.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise ValueError("artifact_dir must be a non-empty string")
    target = raw.get("target")
    if target not in {"opponent_card", "action", "value"}:
        raise ValueError("target must be opponent_card, action, or value")
    input_control = raw.get("input_control", "none")
    allowed_input_controls = {
        "none",
        "shuffle_history",
        "drop_history",
        "mask_public_card",
        "mask_private_card",
        "drop_history_mask_cards",
    }
    if input_control not in allowed_input_controls:
        raise ValueError("input_control must be a known probe input control")
    layers = raw.get("layers")
    if not isinstance(layers, list) or not layers or not all(isinstance(layer, str) for layer in layers):
        raise ValueError("layers must be a non-empty string list")
    probe_seed = _require_int(raw, "probe_seed")
    control_seed = _require_int(raw, "control_seed")
    epochs = _require_int(raw, "epochs")
    batch_size = _require_int(raw, "batch_size")
    learning_rate = _require_float(raw, "learning_rate")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    device = raw.get("device", "auto")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    random_label_control = raw.get("random_label_control", True)
    if not isinstance(random_label_control, bool):
        raise ValueError("random_label_control must be a boolean")

    return LeducTinyProbeConfig(
        run_version=run_version,
        train_config=Path(train_config),
        checkpoint_path=Path(checkpoint_path),
        artifact_dir=Path(artifact_dir),
        target=target,
        input_control=input_control,
        layers=tuple(layers),
        probe_seed=probe_seed,
        control_seed=control_seed,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device,
        random_label_control=random_label_control,
    )


def load_limit_ar_probe_config(path: Path) -> LimitArProbeConfig:
    """Load and validate a Limit AR probe config."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Limit AR probe config must be a JSON object")

    run_version = raw.get("run_version")
    if not isinstance(run_version, str) or not run_version:
        raise ValueError("run_version must be a non-empty string")
    train_config = raw.get("train_config")
    if not isinstance(train_config, str) or not train_config:
        raise ValueError("train_config must be a non-empty string")
    checkpoint_path = raw.get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError("checkpoint_path must be a non-empty string")
    artifact_dir = raw.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        raise ValueError("artifact_dir must be a non-empty string")
    target = raw.get("target")
    if target not in {"opponent_hand", "action", "value"}:
        raise ValueError("target must be opponent_hand, action, or value")
    input_control = raw.get("input_control", "none")
    if input_control not in LIMIT_INPUT_CONTROLS:
        raise ValueError("input_control must be a known Limit AR input control")
    feature_control = raw.get("feature_control", "none")
    if feature_control not in LIMIT_PROBE_FEATURE_CONTROLS:
        raise ValueError("feature_control must be a known Limit AR probe feature control")
    if target != "opponent_hand" and feature_control != "none":
        raise ValueError("feature_control is only supported for opponent_hand probes")
    label_control = raw.get("label_control", "none")
    if label_control not in LIMIT_PROBE_LABEL_CONTROLS:
        raise ValueError("label_control must be a known Limit AR probe label control")
    if target != "opponent_hand" and label_control != "none":
        raise ValueError("label_control is only supported for opponent_hand probes")
    layers = raw.get("layers")
    if not isinstance(layers, list) or not layers or not all(isinstance(layer, str) for layer in layers):
        raise ValueError("layers must be a non-empty string list")
    if feature_control in LIMIT_PROBE_NUISANCE_ONLY_CONTROLS and layers != ["nuisance"]:
        raise ValueError("nuisance-only feature controls must use the nuisance layer")
    if feature_control not in LIMIT_PROBE_NUISANCE_ONLY_CONTROLS and "nuisance" in layers:
        raise ValueError("nuisance layer is only valid for nuisance-only feature controls")
    probe_seed = _require_int(raw, "probe_seed")
    control_seed = _require_int(raw, "control_seed")
    epochs = _require_int(raw, "epochs")
    batch_size = _require_int(raw, "batch_size")
    learning_rate = _require_float(raw, "learning_rate")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    device = raw.get("device", "auto")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    random_label_control = raw.get("random_label_control", True)
    if not isinstance(random_label_control, bool):
        raise ValueError("random_label_control must be a boolean")
    probe_family = raw.get("probe_family", "linear")
    if probe_family not in LIMIT_PROBE_FAMILIES:
        raise ValueError("probe_family must be linear or mlp")
    mlp_hidden_size = raw.get("mlp_hidden_size", 64)
    if not isinstance(mlp_hidden_size, int) or mlp_hidden_size <= 0:
        raise ValueError("mlp_hidden_size must be a positive integer")
    if probe_family == "linear" and "mlp_hidden_size" in raw and mlp_hidden_size != 64:
        raise ValueError("linear probe configs must not tune mlp_hidden_size")

    return LimitArProbeConfig(
        run_version=run_version,
        train_config=Path(train_config),
        checkpoint_path=Path(checkpoint_path),
        artifact_dir=Path(artifact_dir),
        target=target,
        input_control=input_control,
        feature_control=feature_control,
        label_control=label_control,
        layers=tuple(layers),
        probe_seed=probe_seed,
        control_seed=control_seed,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device,
        random_label_control=random_label_control,
        probe_family=probe_family,
        mlp_hidden_size=mlp_hidden_size,
    )
