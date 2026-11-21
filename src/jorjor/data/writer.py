from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import DecisionPoint


def read_decision_points_jsonl(path: Path) -> tuple[DecisionPoint, ...]:
    """Read decision points from JSONL records."""

    points: list[DecisionPoint] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                points.append(DecisionPoint.from_record(json.loads(line)))
    return tuple(points)


def write_decision_points_jsonl(path: Path, points: Iterable[DecisionPoint]) -> None:
    """Write decision points as deterministic JSONL records."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for point in points:
            handle.write(json.dumps(point.to_record(), sort_keys=True) + "\n")
