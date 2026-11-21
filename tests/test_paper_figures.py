from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jorjor.probe.figures import build_paper_figure_pack


class PaperFiguresTest(unittest.TestCase):
    """Test v1.16 paper figure pack generation."""

    def test_build_paper_figure_pack_writes_all_figures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = _write_figure_inputs(root)
            output_dir = root / "figures"

            payload = build_paper_figure_pack(output_dir=output_dir, **paths)
            figure_pack = json.loads((output_dir / "figure_pack.json").read_text(encoding="utf-8"))
            figure2 = json.loads(
                (output_dir / "figure_2_evidence_to_claim_trajectory.json").read_text(encoding="utf-8")
            )
            figure3 = json.loads((output_dir / "figure_3_composition_gate_per_seed.json").read_text(encoding="utf-8"))
            figure4 = json.loads(
                (output_dir / "figure_4_matched_composition_decisive.json").read_text(encoding="utf-8")
            )
            figure5 = json.loads(
                (output_dir / "figure_5_oracle_validation_ordered_vs_shortcut.json").read_text(encoding="utf-8")
            )
            figure_pack_md = (output_dir / "figure_pack.md").read_text(encoding="utf-8")
            svg_texts = [Path(figure["svg_path"]).read_text(encoding="utf-8") for figure in figure_pack["figures"]]

        self.assertEqual(payload["version"], "v1.16")
        self.assertEqual(len(figure_pack["figures"]), 4)
        self.assertEqual([gate["gate"] for gate in figure2["gates"]][-1], "matched-composition")
        self.assertEqual(figure2["gates"][-1]["diagnosis"], "composition_shortcut_supported")
        self.assertGreater(figure2["gates"][4]["source_value"], 0.0)
        self.assertEqual(sorted({row["seed"] for row in figure3["rows"]}), ["seed_k", "seed_l", "seed_m"])
        self.assertEqual(len({row["comparison"] for row in figure3["rows"]}), 3)
        self.assertTrue(all(row["delta"] < 0.0 for row in figure4["rows"]))
        self.assertEqual(len(figure5["rows"]), 6)
        self.assertEqual(
            [row["representation_js"] for row in figure5["rows"] if row["diagnosis"] == "composition_shortcut_supported"],
            [0.0, 0.0, 0.0],
        )
        self.assertIn("Figure 2", figure_pack_md)
        for svg_text in svg_texts:
            self.assertIn("<svg", svg_text)
            self.assertIn("<title>", svg_text)

    def test_build_paper_figure_pack_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = _write_figure_inputs(root)
            output_dir = root / "figures"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_paper_figure_pack.py",
                    "--belief-shortcut-report",
                    str(paths["belief_shortcut_report_path"]),
                    "--final-tables",
                    str(paths["final_tables_path"]),
                    "--matched-composition-report",
                    str(paths["matched_composition_report_path"]),
                    "--synthetic-oracle-report",
                    str(paths["synthetic_oracle_report_path"]),
                    "--learned-synthetic-report",
                    str(paths["learned_synthetic_report_path"]),
                    "--toy-oracle-report",
                    str(paths["toy_oracle_report_path"]),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote paper figure pack figures=4", result.stdout)


def _write_figure_inputs(root: Path) -> dict[str, Path]:
    belief_path = root / "belief_shortcut_report.json"
    final_tables_path = root / "final_diagnostic_tables.json"
    matched_path = root / "matched_composition_report.json"
    synthetic_path = root / "synthetic_oracle_belief_report.json"
    learned_path = root / "learned_synthetic_belief_report.json"
    toy_path = root / "toy_imperfect_info_oracle_report.json"

    _write_json(
        belief_path,
        {
            "cases": [
                {
                    "case_id": "action_value_only_no_range_ar",
                    "final_diagnostic_label": "composition_shortcut_supported",
                    "diagnostic_status": {
                        "naive_probe": "positive",
                        "label_control": "passed",
                        "no_observation": "passed",
                        "history_destruction": "passed",
                        "action_value_control": "passed",
                        "composition_control": "shortcut_supported",
                    },
                    "key_metrics": {
                        "label_control_delta_min": 0.044,
                        "presence_last_action_min_lower_delta": 0.002,
                        "history_delta_ci_min_lower_delta": 0.001,
                        "composition_residual_vs_action_value_min_lower_delta": -0.045,
                        "composition_residual_vs_action_value_composition_min_lower_delta": -0.072,
                    },
                }
            ]
        },
    )
    _write_json(final_tables_path, {"rows": _composition_rows()})
    _write_json(
        matched_path,
        {
            "rows": [
                _matched_row("seed_k", -0.064, -0.081, -0.047),
                _matched_row("seed_l", -0.063, -0.081, -0.045),
                _matched_row("seed_m", -0.075, -0.094, -0.056),
            ]
        },
    )
    _write_oracle_report(synthetic_path, "oracle_belief", "composition_shortcut", 0.461876)
    _write_oracle_report(learned_path, "ordered_belief_ar", "composition_biased_ar", 0.462464)
    _write_oracle_report(toy_path, "oracle_belief", "composition_shortcut", 0.382832)

    return {
        "belief_shortcut_report_path": belief_path,
        "final_tables_path": final_tables_path,
        "matched_composition_report_path": matched_path,
        "synthetic_oracle_report_path": synthetic_path,
        "learned_synthetic_report_path": learned_path,
        "toy_oracle_report_path": toy_path,
    }


def _composition_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    values = {
        "seed_k": (0.013, 0.028, -0.059),
        "seed_l": (0.002, 0.027, -0.061),
        "seed_m": (0.006, 0.026, -0.067),
    }
    for seed, (ordered, prior, residual) in values.items():
        rows.extend(
            [
                _final_table_row(seed, "no_range_action_value_residual_ci", ordered),
                _final_table_row(seed, "composition_nuisance:ordered_residual_vs_action_value", ordered),
                _final_table_row(seed, "composition_nuisance:action_value_composition_only_vs_action_value", prior),
                _final_table_row(seed, "composition_nuisance:composition_residual_vs_action_value_composition", residual),
            ]
        )
    return rows


def _final_table_row(seed: str, claim_gate: str, delta: float) -> dict[str, object]:
    return {
        "claim_gate": claim_gate,
        "training_seed": seed,
        "delta": delta,
        "ci_lower": delta - 0.005,
        "ci_upper": delta + 0.005,
        "table_role": "main",
        "layer": "encoder_layer_2" if "composition_only" not in claim_gate else "nuisance",
        "layer_role": "locked_candidate" if "composition_only" not in claim_gate else "nuisance_baseline",
    }


def _matched_row(seed: str, delta: float, lower: float, upper: float) -> dict[str, object]:
    return {
        "seed": seed,
        "residual_top10": 0.12,
        "action_value_composition_top10": 0.166,
        "delta_residual_minus_prior": delta + 0.02,
        "bucket_bootstrap_mean_delta": delta,
        "bucket_bootstrap_lower_delta": lower,
        "bucket_bootstrap_upper_delta": upper,
        "matched_bucket_count": 146,
        "matched_point_count": 49000,
    }


def _write_oracle_report(path: Path, ordered_mode: str, shortcut_mode: str, posterior_js: float) -> None:
    _write_json(
        path,
        {
            "cases": [
                {
                    "mode": ordered_mode,
                    "final_diagnostic_label": "belief_like_oracle_supported",
                    "order_counterfactual_summary": {
                        "mean_posterior_js": posterior_js,
                        "mean_representation_js": posterior_js,
                    },
                },
                {
                    "mode": shortcut_mode,
                    "final_diagnostic_label": "composition_shortcut_supported",
                    "order_counterfactual_summary": {
                        "mean_posterior_js": posterior_js,
                        "mean_representation_js": 0.0,
                    },
                },
            ]
        },
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
