from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from jorjor.probe.llm_full_sweep import build_llm_full_sweep, rebuild_llm_full_sweep_from_run_reports


class LLMFullSweepTest(unittest.TestCase):
    """Test multi-configuration LLM latent-order sweep aggregation."""

    def test_build_fake_backend_sweep_writes_manifest_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "sweep"

            payload = build_llm_full_sweep(
                output_dir=output_dir,
                lengths=(4, 5),
                layer_ids=(-1,),
                backends=("oracle_posterior", "composition", "order_only", "surface_noise"),
                max_pairs=64,
                bootstrap_count=100,
                model_role="main",
                locked_layer_id=-1,
            )

            markdown = (output_dir / "k.2.6_llm_full_sweep_report.md").read_text(encoding="utf-8")

        self.assertEqual(payload["run_count"], 2)
        self.assertEqual(payload["hf_run_count"], 0)
        self.assertEqual(payload["final_sweep_label"], "inconclusive")
        self.assertEqual(payload["main_sweep_label"], "inconclusive")
        self.assertIn("oracle_posterior", markdown)
        self.assertIn("locked_main", markdown)
        self.assertIn("composition_shortcut_supported", markdown)
        self.assertIn("matched order signal", markdown)

    def test_locked_layer_and_appendix_layer_curve_are_distinguished(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "sweep"

            payload = build_llm_full_sweep(
                output_dir=output_dir,
                lengths=(4,),
                layer_ids=(-1, -2),
                backends=("oracle_posterior", "composition"),
                max_pairs=64,
                bootstrap_count=100,
                model_id="Qwen/Qwen2.5-0.5B-Instruct",
                model_role="main_qwen2_5_0_5b",
                locked_layer_id=-1,
            )

        self.assertEqual(payload["run_count"], 2)
        self.assertEqual(payload["runs"][0]["model_role"], "main_qwen2_5_0_5b")
        self.assertEqual(payload["runs"][0]["analysis_role"], "locked_main")
        self.assertEqual(payload["runs"][1]["analysis_role"], "appendix_layer_curve")

    def test_case_rows_export_all_paper_table_ci_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "sweep"

            payload = build_llm_full_sweep(
                output_dir=output_dir,
                lengths=(4,),
                layer_ids=(-1,),
                backends=("oracle_posterior",),
                max_pairs=64,
                bootstrap_count=100,
                model_role="main",
                locked_layer_id=-1,
            )

        case = payload["runs"][0]["case_summaries"][0]
        self.assertIn("matched_representation_ci_lower", case)
        self.assertIn("matched_representation_ci_upper", case)
        self.assertIn("posterior_equivalence_representation_ci_lower", case)
        self.assertIn("posterior_equivalence_representation_ci_upper", case)
        self.assertIn("surface_equivalence_representation_ci_lower", case)
        self.assertIn("surface_equivalence_representation_ci_upper", case)

    def test_rebuild_from_existing_run_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "source"
            built = build_llm_full_sweep(
                output_dir=source_dir,
                lengths=(4,),
                layer_ids=(-1,),
                backends=("oracle_posterior",),
                max_pairs=64,
                bootstrap_count=100,
                model_role="main",
                locked_layer_id=-1,
            )
            report_path = Path(built["runs"][0]["run_dir"]) / "k.2_llm_latent_order_report.json"
            rebuilt = rebuild_llm_full_sweep_from_run_reports(
                output_dir=Path(temp_dir) / "rebuilt",
                run_report_paths=(report_path,),
                model_specs=(
                    {
                        "model_id": None,
                        "model_role": "main",
                        "locked_layer_id": -1,
                        "layer_ids": [-1],
                    },
                ),
                lengths=(4,),
                layer_ids=(-1,),
                backends=("oracle_posterior",),
            )

        self.assertEqual(rebuilt["aggregate_mode"], "rebuilt_from_existing_run_reports")
        self.assertEqual(rebuilt["run_count"], 1)
        self.assertEqual(rebuilt["runs"][0]["analysis_role"], "locked_main")

    def test_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "sweep"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_llm_full_sweep.py",
                    "--output-dir",
                    str(output_dir),
                    "--lengths",
                    "4",
                    "--layer-ids",
                    "-1",
                    "--backends",
                    "oracle_posterior,composition,order_only,surface_noise",
                    "--max-pairs",
                    "64",
                    "--bootstrap-count",
                    "100",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote k.2.6 LLM full sweep", result.stdout)

    def test_script_rebuilds_from_run_report_glob(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "source"
            built = build_llm_full_sweep(
                output_dir=source_dir,
                lengths=(4,),
                layer_ids=(-1,),
                backends=("oracle_posterior",),
                max_pairs=64,
                bootstrap_count=100,
                model_role="main",
                locked_layer_id=-1,
            )
            run_report_glob = str(Path(built["runs"][0]["run_dir"]) / "k.2_llm_latent_order_report.json")
            output_dir = Path(temp_dir) / "rebuilt"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_llm_full_sweep.py",
                    "--output-dir",
                    str(output_dir),
                    "--lengths",
                    "4",
                    "--layer-ids",
                    "-1",
                    "--backends",
                    "oracle_posterior",
                    "--model-role",
                    "main",
                    "--locked-layer-id",
                    "-1",
                    "--rebuild-run-report-glob",
                    run_report_glob,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote k.2.6 LLM full sweep", result.stdout)


if __name__ == "__main__":
    unittest.main()
