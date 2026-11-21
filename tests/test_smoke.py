from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from jorjor.config import load_smoke_config
from jorjor.smoke import build_smoke_payload, run_smoke


class SmokeTest(unittest.TestCase):
    """Test the deterministic v0.0 smoke path."""

    def test_payload_is_deterministic(self) -> None:
        left = build_smoke_payload(seed=7, sample_count=5)
        right = build_smoke_payload(seed=7, sample_count=5)

        self.assertEqual(left, right)
        self.assertEqual(left["samples"], [5305, 2471, 6468, 791, 1186])

    def test_load_smoke_config(self) -> None:
        config = load_smoke_config(Path("configs/smoke.json"))

        self.assertEqual(config.seed, 7)
        self.assertEqual(config.sample_count, 5)
        self.assertEqual(config.artifact_dir, Path("outputs/smoke"))

    def test_run_smoke_writes_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_smoke(
                config_path=Path("configs/smoke.json"),
                artifact_dir=Path(temp_dir),
            )
            artifact = json.loads(result.artifact_path.read_text(encoding="utf-8"))

        self.assertEqual(result.seed, 7)
        self.assertEqual(result.sample_count, 5)
        self.assertEqual(artifact["checksum"], result.checksum)
        self.assertEqual(artifact["samples"], [5305, 2471, 6468, 791, 1186])


if __name__ == "__main__":
    unittest.main()
