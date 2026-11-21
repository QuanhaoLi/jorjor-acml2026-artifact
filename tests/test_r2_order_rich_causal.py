from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jorjor.models.tiny import require_torch
from jorjor.probe.order_rich_causal import (
    CAUSAL_SUBSPACE_SUPPORTED,
    DIAGNOSTIC_ONLY,
    PROJECTION_CAUSAL_SUPPORTED,
    build_order_rich_causal_subspace_report,
)


class OrderRichCausalSubspaceTest(unittest.TestCase):
    """Test r.2 causal subspace report."""

    def setUp(self) -> None:
        try:
            require_torch()
        except RuntimeError as exc:
            self.skipTest(str(exc))

    def test_report_builder_writes_causal_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "r2"
            payload = build_order_rich_causal_subspace_report(
                output_dir=output_dir,
                length=4,
                seed=61,
                epochs=8,
                hidden_size=24,
                batch_size=128,
                max_pairs=512,
                subspace_rank=2,
            )
            saved = json.loads((output_dir / "r.2_order_rich_causal_subspace_report.json").read_text(encoding="utf-8"))
            manifest_exists = (output_dir / "r.2_order_rich_causal_subspace_manifest.json").exists()

        self.assertEqual(saved["report_type"], "r.2_order_rich_causal_subspace")
        self.assertIn(
            payload["final_diagnostic_label"],
            {CAUSAL_SUBSPACE_SUPPORTED, PROJECTION_CAUSAL_SUPPORTED, DIAGNOSTIC_ONLY},
        )
        self.assertGreater(payload["patching_summary"]["matched_pair_count"], 0)
        self.assertGreaterEqual(payload["ablation_summary"]["posterior_readout_js_mean"], 0.0)
        self.assertTrue(manifest_exists)

    def test_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_r2_order_rich_causal_subspace.py",
                    "--output-dir",
                    str(Path(temp_dir) / "r2"),
                    "--length",
                    "4",
                    "--epochs",
                    "8",
                    "--hidden-size",
                    "24",
                    "--batch-size",
                    "128",
                    "--max-pairs",
                    "512",
                    "--subspace-rank",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote r.2 order-rich causal subspace report", result.stdout)


if __name__ == "__main__":
    unittest.main()
