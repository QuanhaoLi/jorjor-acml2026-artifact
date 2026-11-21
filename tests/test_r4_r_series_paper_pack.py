from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jorjor.artifacts import write_json_artifact
from jorjor.probe.order_rich_causal import build_r_series_paper_pack


class RSeriesPaperPackTest(unittest.TestCase):
    """Test r.4 r-series paper pack generation."""

    def test_pack_builder_writes_table_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = _write_fake_inputs(root)
            output_dir = root / "pack"
            payload = build_r_series_paper_pack(output_dir=output_dir, **paths)
            saved = json.loads((output_dir / "r.4_r_series_paper_pack.json").read_text(encoding="utf-8"))
            table_exists = (output_dir / "r.4_r_series_main_table.md").exists()
            manifest_exists = (output_dir / "r.4_r_series_manifest.json").exists()

        self.assertEqual(saved["report_type"], "r.4_r_series_paper_pack")
        self.assertEqual(len(payload["rows"]), 8)
        self.assertTrue(table_exists)
        self.assertTrue(manifest_exists)
        self.assertIn("does not upgrade", payload["required_limitation"])

    def test_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = _write_fake_inputs(root)
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_r4_r_series_paper_pack.py",
                    "--output-dir",
                    str(root / "pack"),
                    "--r001",
                    str(paths["r001_path"]),
                    "--r002",
                    str(paths["r002_path"]),
                    "--r1",
                    str(paths["r1_path"]),
                    "--r2",
                    str(paths["r2_path"]),
                    "--r3",
                    str(paths["r3_path"]),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote r.4 r-series paper pack", result.stdout)


def _write_fake_inputs(root: Path) -> dict[str, Path]:
    r001 = {
        "report_type": "r.0.1_exact_leduc_feasibility",
        "final_diagnostic_label": "insufficient_exact_posterior_support",
        "matched_summary": {"matched_bucket_count": 0},
        "checksum": "r001",
    }
    r002 = {
        "report_type": "r.0.2_order_rich_exact_posterior",
        "oracle_detection_label": "exact_posterior_signal_supported",
        "composition_detection_label": "composition_collapsed",
        "oracle_summary": {"representation_js_ci": {"lower": 0.1, "upper": 0.2}},
        "composition_summary": {"representation_js_ci": {"lower": 0.0, "upper": 0.0}},
        "final_diagnostic_label": "order_rich_exact_posterior_supported",
        "checksum": "r002",
    }
    r1 = {
        "report_type": "r.1_order_rich_behavior_gates",
        "cases": [
            _case("posterior_supervised_gru_posterior_head", "learned_posterior_supported", 0.99),
            _case("behavior_only_gru_hidden_probe", "learned_posterior_supported", 0.88),
        ],
        "checksum": "r1",
    }
    r2 = {
        "report_type": "r.2_order_rich_causal_subspace",
        "projection_supported": True,
        "ablation_summary": {"posterior_policy_js_mean": 0.04, "random_policy_js_mean": 0.01},
        "final_diagnostic_label": "projection_causal_supported_patch_specificity_inconclusive",
        "checksum": "r2",
    }
    r3 = {
        "report_type": "r.3_order_rich_robustness_pack",
        "projection_supported_count": 3,
        "seed_count": 3,
        "random_pair_specificity_supported_count": 0,
        "final_diagnostic_label": "robust_projection_supported",
        "checksum": "r3",
    }
    paths = {}
    for key, payload in {
        "r001_path": r001,
        "r002_path": r002,
        "r1_path": r1,
        "r2_path": r2,
        "r3_path": r3,
    }.items():
        path = root / f"{key}.json"
        write_json_artifact(path, payload)
        paths[key] = path
    return paths


def _case(mode: str, label: str, corr: float) -> dict[str, object]:
    return {
        "mode": mode,
        "final_diagnostic_label": label,
        "matched_composition_summary": {
            "pair_weighted_posterior_representation_correlation": corr,
        },
    }


if __name__ == "__main__":
    unittest.main()
