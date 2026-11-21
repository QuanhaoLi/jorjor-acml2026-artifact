from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(data: dict[str, Any]) -> str:
    """Return deterministic JSON for checksums and paper artifacts."""

    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def checksum_payload(data: dict[str, Any]) -> str:
    """Compute a stable checksum for a JSON-serializable payload."""

    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def write_json_artifact(path: Path, data: dict[str, Any]) -> None:
    """Write a deterministic JSON artifact to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(data), encoding="utf-8")
