from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from jorjor.probe.k3_diagnostics import (
    build_composition_feature_sensitivity_report,
    build_g1_g4_threshold_manifest,
    build_llm_template_robustness_report,
    build_poker_oracle_positive_control_report,
)


class K3DiagnosticsTest(unittest.TestCase):
    """Test paper-hardening k.3 diagnostic report builders."""

    def test_poker_oracle_positive_control_accepts_oracle_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            matched_path = root / "matched_composition_report.json"
            output_dir = root / "out"
            _write_matched_report(matched_path)

            payload = build_poker_oracle_positive_control_report(
                matched_report_path=matched_path,
                output_dir=output_dir,
                bootstrap_count=50,
                bootstrap_seed=7,
            )

            report_json = json.loads(
                (output_dir / "k.3.0_poker_oracle_positive_control_report.json").read_text(encoding="utf-8")
            )
            report_md = (output_dir / "k.3.0_poker_oracle_positive_control_report.md").read_text(
                encoding="utf-8"
            )

        self.assertEqual(payload["final_diagnostic_label"], "poker_oracle_gate_accepts_ordered_target")
        self.assertEqual(report_json["row_count"], 2)
        self.assertGreater(report_json["min_bucket_bootstrap_lower_delta"], 0.0)
        self.assertIn("oracle", report_md.lower())
        self.assertIn("does not upgrade", report_md)

    def test_composition_feature_sensitivity_reports_predeclared_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "out"
            _write_matched_report(root / "matched_composition_report.json")
            _write_belief_report(root / "belief_shortcut_report.json")

            payload = build_composition_feature_sensitivity_report(
                belief_report_path=root / "belief_shortcut_report.json",
                matched_report_path=root / "matched_composition_report.json",
                output_dir=output_dir,
            )

            report_md = (output_dir / "k.3.1_composition_feature_sensitivity_report.md").read_text(
                encoding="utf-8"
            )

        self.assertEqual(payload["variant_count"], 3)
        self.assertEqual({row["variant"] for row in payload["rows"]}, {"C_min", "C_mid", "C_full"})
        self.assertEqual(payload["final_diagnostic_label"], "full_composition_only_blocks")
        self.assertEqual(payload["rows"][0]["evidence_type"], "proxy_confound_positive")
        self.assertEqual(payload["rows"][-1]["evidence_type"], "decisive_residual_block")
        self.assertIn("C_full", report_md)

    def test_llm_template_robustness_aggregates_template_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            qwen = root / "qwen" / "k.2.6_llm_full_sweep_report.json"
            smol = root / "smol" / "k.2.6_llm_full_sweep_report.json"
            qwen_run = root / "qwen" / "runs" / "qwen_locked" / "k.2_llm_latent_order_report.json"
            smol_run = root / "smol" / "runs" / "smol_locked" / "k.2_llm_latent_order_report.json"
            _write_llm_sweep_report(qwen, qwen_run, model_role="main", run_id="qwen_locked")
            _write_llm_sweep_report(smol, smol_run, model_role="appendix_smol360", run_id="smol_locked")

            payload = build_llm_template_robustness_report(
                sweep_report_paths=(qwen, smol),
                output_dir=root / "out",
            )

        self.assertEqual(payload["model_count"], 2)
        self.assertEqual(payload["main_final_label"], "order_information_supported")
        self.assertTrue(all(row["template_count"] >= 2 for row in payload["rows"]))
        self.assertTrue(all(row["template_fingerprints"] for row in payload["rows"]))

    def test_llm_template_robustness_accepts_repo_relative_run_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sweep = root / "sweep" / "k.2.6_llm_full_sweep_report.json"
            run_report = root / "relative_runs" / "locked" / "k.2_llm_latent_order_report.json"
            _write_llm_sweep_report(sweep, run_report, model_role="main", run_id="locked")
            data = json.loads(sweep.read_text(encoding="utf-8"))
            data["runs"][0]["run_dir"] = str(run_report.parent)
            sweep.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")

            payload = build_llm_template_robustness_report(
                sweep_report_paths=(sweep,),
                output_dir=root / "out",
            )

        self.assertEqual(payload["rows"][0]["template_count"], 3)

    def test_g1_g4_manifest_records_rules_and_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_belief_report(root / "belief_shortcut_report.json")

            payload = build_g1_g4_threshold_manifest(
                belief_report_path=root / "belief_shortcut_report.json",
                output_dir=root / "out",
            )

            manifest_md = (root / "out" / "k.3.3_g1_g4_threshold_manifest.md").read_text(
                encoding="utf-8"
            )

        self.assertEqual([row["gate_id"] for row in payload["gates"]], ["G1", "G2", "G3", "G4"])
        self.assertTrue(all(row["pass_rule"] for row in payload["gates"]))
        self.assertIn("label permutation", manifest_md.lower())

    def test_k3_scripts_run_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            matched = root / "matched_composition_report.json"
            belief = root / "belief_shortcut_report.json"
            output_dir = root / "out"
            _write_matched_report(matched)
            _write_belief_report(belief)

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_k3_poker_oracle_positive_control.py",
                    "--matched-report",
                    str(matched),
                    "--output-dir",
                    str(output_dir / "oracle"),
                    "--bootstrap-count",
                    "20",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("poker_oracle_gate_accepts_ordered_target", result.stdout)

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_k3_g1_g4_threshold_manifest.py",
                    "--belief-report",
                    str(belief),
                    "--output-dir",
                    str(output_dir / "manifest"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("gates=4", result.stdout)


def _write_matched_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed, prior_top10 in (("seed_k", 0.20), ("seed_l", 0.22)):
        rows.append(
            {
                "seed": seed,
                "matched_point_count": 100,
                "all_joined_point_count": 100,
                "dropped_point_count": 0,
                "dropped_fraction": 0.0,
                "bucket_count": 5,
                "matched_bucket_count": 5,
                "singleton_bucket_count": 0,
                "residual_top10": 0.12,
                "action_value_composition_top10": prior_top10,
                "delta_residual_minus_prior": 0.12 - prior_top10,
                "bucket_bootstrap_mean_delta": 0.12 - prior_top10,
                "bucket_bootstrap_lower_delta": 0.10 - prior_top10,
                "bucket_bootstrap_upper_delta": 0.14 - prior_top10,
            }
        )
    path.write_text(
        json.dumps(
            {
                "artifact": "JorJor-Inspect",
                "version": "1.2",
                "bootstrap_count": 100,
                "bootstrap_seed": 20260602,
                "confidence_level": 0.95,
                "bucket_key": ["history_length", "action_counts", "last_action"],
                "rows": rows,
                "final_diagnostic_label": "composition_shortcut_supported",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_belief_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": "1.1",
                "cases": [
                    {
                        "case": "action_value_only_no_range_ar",
                        "naive_probe_status": "positive_intermediate_layer_signal",
                        "label_control_status": "passes",
                        "history_control_status": "passes",
                        "composition_nuisance_status": "composition_shortcut_supported",
                        "final_diagnostic_label": "composition_shortcut_supported",
                        "key_metrics": {
                            "label_control_delta_min": 0.04,
                            "presence_last_action_min_lower_delta": 0.01,
                            "history_delta_ci_min_lower_delta": 0.02,
                            "composition_only_min_lower_delta": 0.005,
                            "action_value_composition_min_lower_delta": 0.006,
                            "composition_residual_vs_action_value_composition_min_lower_delta": -0.03,
                        },
                    }
                ],
                "sources": [
                    {"key": "fresh50_control", "path": "fresh.json", "present": True},
                    {"key": "history_composition", "path": "history.json", "present": True},
                    {"key": "composition_nuisance", "path": "composition.json", "present": True},
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_llm_sweep_report(path: Path, run_report_path: Path, *, model_role: str, run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    run_report_path.parent.mkdir(parents=True, exist_ok=True)
    run_report_path.write_text(
        json.dumps(
            {
                "report_type": "k.2_llm_latent_order_audit",
                "surface_template_count": 3,
                "textualization": {
                    "templates": [
                        "Case: {then_events}.",
                        "Stream: {arrow_events}.",
                        "Record: {comma_events}.",
                    ]
                },
                "checksum": f"run-{run_id}",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    row = {
        "run_id": run_id,
        "model_id": f"model/{run_id}",
        "model_role": model_role,
        "analysis_role": "locked_main",
        "locked_layer_id": 12,
        "layer_id": 12,
        "backend": "hf_causal_lm",
        "final_diagnostic_label": "order_information_supported",
        "matched_pair_count": 8192,
        "matched_bucket_count": 208,
        "matched_mean_representation_distance": 0.0001,
        "matched_representation_ci_lower": 0.00009,
        "matched_representation_ci_upper": 0.00011,
        "posterior_representation_correlation": 0.04,
        "posterior_equivalence_mean_representation_distance": 0.0002,
        "posterior_equivalence_representation_ci_lower": 0.0001,
        "posterior_equivalence_representation_ci_upper": 0.0003,
        "surface_equivalence_mean_representation_distance": 0.001,
        "surface_equivalence_representation_ci_lower": 0.0009,
        "surface_equivalence_representation_ci_upper": 0.0011,
    }
    path.write_text(
        json.dumps(
            {
                "version": "k.2.6",
                "final_sweep_label": "order_information_supported",
                "main_sweep_label": "order_information_supported",
                "hf_rows": [row],
                "runs": [
                    {
                        "run_id": run_id,
                        "run_dir": str(run_report_path.parent),
                        "surface_template_count": 3,
                        "checksum": f"run-{run_id}",
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
