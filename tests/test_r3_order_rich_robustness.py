from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jorjor.models.tiny import require_torch
from jorjor.probe.order_rich_causal import (
    ROBUSTNESS_INCONCLUSIVE,
    ROBUST_PROJECTION_SUPPORTED,
    build_order_rich_robustness_pack,
)


class OrderRichRobustnessPackTest(unittest.TestCase):
    """Test r.3 multi-seed robustness pack."""

    def setUp(self) -> None:
        try:
            require_torch()
        except RuntimeError as exc:
            self.skipTest(str(exc))

    def test_robustness_pack_writes_seed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "r3"
            payload = build_order_rich_robustness_pack(
                output_dir=output_dir,
                seeds=(71, 72),
                length=4,
                epochs=6,
                hidden_size=20,
                batch_size=128,
                max_pairs=256,
                subspace_rank=2,
            )
            saved = json.loads((output_dir / "r.3_order_rich_robustness_pack.json").read_text(encoding="utf-8"))
            manifest_exists = (output_dir / "r.3_order_rich_robustness_manifest.json").exists()

        self.assertEqual(saved["report_type"], "r.3_order_rich_robustness_pack")
        self.assertEqual(payload["seed_count"], 2)
        self.assertEqual(len(payload["seed_rows"]), 2)
        self.assertIn(payload["final_diagnostic_label"], {ROBUST_PROJECTION_SUPPORTED, ROBUSTNESS_INCONCLUSIVE})
        self.assertTrue(manifest_exists)

    def test_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_r3_order_rich_robustness_pack.py",
                    "--output-dir",
                    str(Path(temp_dir) / "r3"),
                    "--seeds",
                    "71,72",
                    "--length",
                    "4",
                    "--epochs",
                    "6",
                    "--hidden-size",
                    "20",
                    "--batch-size",
                    "128",
                    "--max-pairs",
                    "256",
                    "--subspace-rank",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote r.3 order-rich robustness pack", result.stdout)


if __name__ == "__main__":
    unittest.main()
