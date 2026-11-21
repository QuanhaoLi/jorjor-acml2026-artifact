from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from jorjor.probe.inspect import (
    REALIZED_LABEL_CONTROLS_PASSED,
    COMPOSITION_SHORTCUT_SUPPORTED,
    INCONCLUSIVE,
    TARGET_STRATIFIED_COMPOSITION_BOUND_SUPPORTED,
    build_acml_manuscript_pack,
    build_belief_shortcut_report,
    build_claim_gate_matrix,
    build_matched_composition_report,
)


def _write_pack(root: Path, commit: str, relative_path: str, payload: dict[str, object]) -> None:
    path = root / commit / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_required_packs(root: Path, *, composition_lower: float) -> None:
    _write_pack(
        root,
        "fresh",
        "outputs/tables/limit_ar_no_range_fresh50_control_pack.json",
        {
            "version": "1.0.13",
            "commit_50k": "fresh",
            "label_control_delta_min": 0.04,
            "history_delta_ci_min_lower_delta": 0.001,
        },
    )
    _write_pack(
        root,
        "hist",
        "outputs/tables/limit_ar_no_range_history_composition_pack.json",
        {
            "version": "1.0.15",
            "commit": "hist",
            "count_preserving_min_lower_delta": -0.001,
            "presence_last_action_min_lower_delta": 0.002,
        },
    )
    _write_pack(
        root,
        "comp",
        "outputs/tables/limit_ar_no_range_composition_nuisance_pack.json",
        {
            "version": "1.0.16",
            "commit": "comp",
            "rows": [{"seed": "seed_k"}],
            "action_value_composition_min_lower_delta": 0.02,
            "composition_only_min_lower_delta": 0.006,
            "composition_residual_vs_action_value_min_lower_delta": -0.04,
            "composition_residual_vs_action_value_composition_min_lower_delta": composition_lower,
        },
    )


class ProbeInspectTest(unittest.TestCase):
    """Test JorJor-Inspect diagnostic report generation."""

    def test_build_belief_shortcut_report_classifies_composition_shortcut(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "inspect"
            _write_required_packs(root, composition_lower=-0.07)

            payload = build_belief_shortcut_report(
                root,
                output_dir,
                composition_commit="comp",
                fresh50_commit="fresh",
                history_composition_commit="hist",
            )
            report_json = json.loads((output_dir / "belief_shortcut_report.json").read_text(encoding="utf-8"))
            report_md = (output_dir / "belief_shortcut_report.md").read_text(encoding="utf-8")

        case = payload["cases"][0]
        self.assertEqual(case["final_diagnostic_label"], COMPOSITION_SHORTCUT_SUPPORTED)
        self.assertEqual(report_json["cases"][0]["composition_nuisance_status"], COMPOSITION_SHORTCUT_SUPPORTED)
        self.assertIn("composition_shortcut_supported", report_md)
        self.assertIn("composition_residual_vs_action_value_composition_min_lower_delta", report_md)

    def test_build_belief_shortcut_report_classifies_belief_like_residual(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "inspect"
            _write_required_packs(root, composition_lower=0.01)

            payload = build_belief_shortcut_report(
                root,
                output_dir,
                composition_commit="comp",
                fresh50_commit="fresh",
                history_composition_commit="hist",
            )

        self.assertEqual(payload["cases"][0]["final_diagnostic_label"], REALIZED_LABEL_CONTROLS_PASSED)

    def test_build_belief_shortcut_report_surfaces_optional_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "inspect"
            _write_pack(
                root,
                "comp",
                "outputs/tables/limit_ar_no_range_composition_nuisance_pack.json",
                {
                    "version": "1.0.16",
                    "composition_residual_vs_action_value_composition_min_lower_delta": -0.01,
                },
            )

            payload = build_belief_shortcut_report(
                root,
                output_dir,
                composition_commit="comp",
                fresh50_commit="missing-fresh",
                history_composition_commit="missing-hist",
            )

        case = payload["cases"][0]
        self.assertEqual(case["final_diagnostic_label"], COMPOSITION_SHORTCUT_SUPPORTED)
        self.assertEqual(case["label_control_status"], INCONCLUSIVE)
        self.assertEqual(len(payload["gaps"]), 2)

    def test_build_belief_shortcut_report_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "inspect"
            _write_required_packs(root, composition_lower=-0.07)

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_belief_shortcut_report.py",
                    "--archive-root",
                    str(root),
                    "--output-dir",
                    str(output_dir),
                    "--commit",
                    "comp",
                    "--fresh50-commit",
                    "fresh",
                    "--history-composition-commit",
                    "hist",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("final_diagnostic_label=composition_shortcut_supported", result.stdout)

    def test_build_matched_composition_report_classifies_shortcut(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "inspect"
            commit = "comp"
            bucket_map: dict[str, str] = {}
            for seed in ("seed_k", "seed_l", "seed_m"):
                point_ids = [f"{seed}-hand-{index // 2}:{index}" for index in range(4)]
                bucket_map.update({point_id: f"bucket-{index // 2}" for index, point_id in enumerate(point_ids)})
                _write_correctness_pair(root, commit, seed, point_ids)
                train_config = root / commit / "configs" / "train" / f"limit_ar_action_value_only_{seed}_50k_4layer.json"
                train_config.parent.mkdir(parents=True, exist_ok=True)
                train_config.write_text("{}", encoding="utf-8")

            with patch("jorjor.probe.inspect._test_point_buckets", return_value=bucket_map):
                payload = build_matched_composition_report(
                    root,
                    output_dir,
                    commit=commit,
                    bootstrap_count=100,
                    bootstrap_seed=7,
                )
            report_md = (output_dir / "matched_composition_report.md").read_text(encoding="utf-8")

        self.assertEqual(payload["final_diagnostic_label"], TARGET_STRATIFIED_COMPOSITION_BOUND_SUPPORTED)
        self.assertEqual(payload["analysis_role"], "target_stratified_postdecision_sensitivity")
        self.assertIs(payload["decision_time_gate"], False)
        self.assertLess(payload["min_delta_residual_minus_prior"], 0.0)
        self.assertLess(payload["min_bucket_bootstrap_lower_delta"], 0.0)
        self.assertIn("target-stratified matched-composition sensitivity", report_md.lower())

    def test_build_claim_gate_matrix_writes_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inspect_dir = root / "inspect"
            paper_dir = root / "tables"
            output_dir = root / "out"
            inspect_dir.mkdir()
            paper_dir.mkdir()
            _write_minimal_claim_matrix_inputs(inspect_dir)
            (paper_dir / "limit_ar_paper_evidence_pack.json").write_text(
                json.dumps(
                    {
                        "core_4layer": {"history_residual_delta_min": 0.01},
                        "scale_robustness": {"all_history_pass": True},
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            payload = build_claim_gate_matrix(inspect_dir, paper_dir, output_dir)
            matrix_md = (output_dir / "claim_gate_matrix.md").read_text(encoding="utf-8")

        self.assertEqual(len(payload["rows"]), 2)
        self.assertEqual(payload["rows"][1]["final_diagnostic_label"], COMPOSITION_SHORTCUT_SUPPORTED)
        self.assertEqual(payload["rows"][1]["matched_composition_residual_supported"], "fail")
        self.assertIn("Claim Gate Matrix", matrix_md)

    def test_build_claim_gate_matrix_includes_strict_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inspect_dir = root / "inspect"
            paper_dir = root / "tables"
            strict_dir = root / "strict"
            output_dir = root / "out"
            inspect_dir.mkdir()
            paper_dir.mkdir()
            strict_dir.mkdir()
            _write_minimal_claim_matrix_inputs(inspect_dir)
            (strict_dir / "limit_ar_range_supervised_positive_control_pack.json").write_text(
                json.dumps(
                    {
                        "final_diagnostic_label": COMPOSITION_SHORTCUT_SUPPORTED,
                        "s5_min_lower_delta": -0.0003,
                        "s6_min_lower_delta": -0.071,
                        "s7_min_lower_delta": -0.093,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (strict_dir / "probe_capacity_s6_s7_report.json").write_text(
                json.dumps(
                    {
                        "final_verdict": "mlp_inconclusive",
                        "c1_min_lower_delta": 0.008,
                        "c2_min_lower_delta": -0.015,
                        "c2_max_upper_delta": 0.00008,
                        "c3_min_lower_delta": -0.016,
                        "c3_max_upper_delta": 0.014,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            payload = build_claim_gate_matrix(
                inspect_dir,
                paper_dir,
                output_dir,
                strict_table_dir=strict_dir,
            )
            rows = {row["case"]: row for row in payload["rows"]}

        self.assertEqual(len(payload["rows"]), 3)
        self.assertEqual(rows["range_supervised_ar"]["final_diagnostic_label"], COMPOSITION_SHORTCUT_SUPPORTED)
        self.assertEqual(rows["range_supervised_ar"]["composition_nuisance_ruled_out"], "fail")
        self.assertEqual(rows["range_supervised_ar"]["matched_composition_residual_supported"], "fail")
        self.assertEqual(rows["action_value_only_no_range_ar_mlp_appendix"]["final_diagnostic_label"], "mlp_inconclusive")
        self.assertEqual(rows["action_value_only_no_range_ar_mlp_appendix"]["composition_nuisance_ruled_out"], INCONCLUSIVE)
        self.assertEqual(
            rows["action_value_only_no_range_ar_mlp_appendix"]["matched_composition_residual_supported"],
            INCONCLUSIVE,
        )
        present_sources = {source["key"] for source in payload["sources"] if source["present"]}
        self.assertIn("range_supervised_strict_gate", present_sources)
        self.assertIn("probe_capacity_s6_s7", present_sources)

    def test_build_claim_gate_matrix_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inspect_dir = root / "inspect"
            paper_dir = root / "tables"
            output_dir = root / "out"
            inspect_dir.mkdir()
            paper_dir.mkdir()
            _write_minimal_claim_matrix_inputs(inspect_dir)

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_claim_gate_matrix.py",
                    "--inspect-dir",
                    str(inspect_dir),
                    "--paper-table-dir",
                    str(paper_dir),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote claim gate matrix rows=2", result.stdout)

    def test_build_acml_manuscript_pack_writes_required_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            inspect_dir = Path(temp_dir) / "inspect"
            output_dir = Path(temp_dir) / "out"
            inspect_dir.mkdir()
            (inspect_dir / "claim_gate_matrix.md").write_text(
                "| case | final diagnostic label |\n|---|---|\n| no_range | composition_shortcut_supported |\n",
                encoding="utf-8",
            )
            (inspect_dir / "belief_shortcut_report.md").write_text(
                "# Belief vs Shortcut Diagnostic Report\ncomposition_shortcut_supported\n",
                encoding="utf-8",
            )
            (inspect_dir / "matched_composition_report.md").write_text(
                "# Target-Stratified Matched-Composition Sensitivity Report\ntarget_stratified_composition_bound_supported\n",
                encoding="utf-8",
            )

            payload = build_acml_manuscript_pack(inspect_dir, output_dir)
            text = (output_dir / "acml_manuscript_pack.md").read_text(encoding="utf-8")

        self.assertEqual(payload["version"], "1.4")
        self.assertIn("ACML Manuscript Pack", text)
        self.assertIn("composition-shortcut supported", text)
        self.assertIn("Reproducibility Commands", text)
        self.assertNotIn("We discover opponent-range representations", text)

    def test_build_acml_manuscript_pack_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            inspect_dir = Path(temp_dir) / "inspect"
            output_dir = Path(temp_dir) / "out"
            inspect_dir.mkdir()
            (inspect_dir / "claim_gate_matrix.md").write_text("# Claim Gate Matrix\n", encoding="utf-8")
            (inspect_dir / "belief_shortcut_report.md").write_text(
                "# Belief vs Shortcut Diagnostic Report\n",
                encoding="utf-8",
            )
            (inspect_dir / "matched_composition_report.md").write_text(
                "# Target-Stratified Matched-Composition Sensitivity Report\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_acml_manuscript_pack.py",
                    "--inspect-dir",
                    str(inspect_dir),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote ACML manuscript pack", result.stdout)


def _write_minimal_claim_matrix_inputs(inspect_dir: Path) -> None:
    (inspect_dir / "belief_shortcut_report.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case": "action_value_only_no_range_ar",
                        "label_control_status": "passes",
                        "history_control_status": "passes",
                        "composition_nuisance_status": COMPOSITION_SHORTCUT_SUPPORTED,
                        "final_diagnostic_label": COMPOSITION_SHORTCUT_SUPPORTED,
                    }
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (inspect_dir / "matched_composition_report.json").write_text(
        json.dumps(
            {
                "final_diagnostic_label": COMPOSITION_SHORTCUT_SUPPORTED,
                "min_bucket_bootstrap_lower_delta": -0.01,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_correctness_pair(root: Path, commit: str, seed: str, point_ids: list[str]) -> None:
    prefix = root / commit / "outputs" / "probes" / f"limit_ar_action_value_only_{seed}_4layer_range"
    residual_path = prefix.parent / f"{prefix.name}_residualize_action_value_history_composition_50k" / "probe_correctness.jsonl"
    prior_path = prefix.parent / f"{prefix.name}_action_value_history_composition_only_50k" / "probe_correctness.jsonl"
    residual_rows = []
    prior_rows = []
    for index, point_id in enumerate(point_ids):
        base = {
            "control": "none",
            "point_id": point_id,
            "split": "test",
            "target": "opponent_hand",
            "target_label": index % 2,
            "top1_correct": False,
            "top5_correct": False,
        }
        residual_rows.append({**base, "top10_correct": index == 0})
        prior_rows.append({**base, "top10_correct": index in {0, 1, 2}})
    for path, rows in ((residual_path, residual_rows), (prior_path, prior_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
