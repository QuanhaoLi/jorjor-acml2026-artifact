from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from .artifacts import checksum_payload, write_json_artifact
from .config import load_smoke_config


@dataclass(frozen=True)
class SmokeResult:
    """Result summary returned by the v0.0 smoke command."""

    seed: int
    sample_count: int
    checksum: str
    artifact_path: Path


def build_smoke_payload(seed: int, sample_count: int) -> dict[str, object]:
    """Build a deterministic smoke payload from a seed."""

    rng = random.Random(seed)
    samples = [rng.randrange(0, 10_000) for _ in range(sample_count)]
    return {
        "version": "0.0",
        "purpose": "deterministic local smoke test",
        "seed": seed,
        "sample_count": sample_count,
        "samples": samples,
    }


def run_smoke(config_path: Path, artifact_dir: Path | None = None) -> SmokeResult:
    """Run the v0.0 local smoke test and write a deterministic artifact."""

    config = load_smoke_config(config_path)
    output_dir = artifact_dir if artifact_dir is not None else config.artifact_dir
    payload = build_smoke_payload(seed=config.seed, sample_count=config.sample_count)
    checksum = checksum_payload(payload)
    artifact_payload = {**payload, "checksum": checksum}
    artifact_path = output_dir / "smoke_report.json"
    write_json_artifact(artifact_path, artifact_payload)

    return SmokeResult(
        seed=config.seed,
        sample_count=config.sample_count,
        checksum=checksum,
        artifact_path=artifact_path,
    )
