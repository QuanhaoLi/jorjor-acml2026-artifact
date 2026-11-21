from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jorjor.inspect import build_audit_report
from jorjor.envs.synthetic_belief import BELIEF_LIKE_ORACLE_SUPPORTED
from jorjor.probe.inspect import COMPOSITION_SHORTCUT_SUPPORTED
from jorjor.probe.counterfactual import TRAJECTORY_ALIGNED_SUPPORTED


class InspectAuditTest(unittest.TestCase):
    """Test product-facing JorJor-Inspect audit assembly."""

    def test_build_audit_report_writes_product_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inspect_dir = root / "inspect"
            output_dir = root / "audit"
            _write_inspect_fixture(inspect_dir)

            payload = build_audit_report(inspect_dir=inspect_dir, output_dir=output_dir)
            report = (output_dir / "diagnostic_report.md").read_text(encoding="utf-8")
            artifact_index_exists = (output_dir / "artifact_index.json").exists()
            source_manifest_exists = (output_dir / "source_manifest.json").exists()
            tables_exists = (output_dir / "tables.json").exists()
            reproducibility_exists = (output_dir / "reproducibility.md").exists()

        self.assertEqual(payload["final_verdict"], COMPOSITION_SHORTCUT_SUPPORTED)
        self.assertIn("JorJor-Inspect Audit Summary", report)
        self.assertIn("optional order-counterfactual report is missing", payload["gaps"])
        self.assertTrue(artifact_index_exists)
        self.assertTrue(source_manifest_exists)
        self.assertTrue(tables_exists)
        self.assertTrue(reproducibility_exists)

    def test_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inspect_dir = root / "inspect"
            output_dir = root / "audit"
            _write_inspect_fixture(inspect_dir)

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/jorjor_inspect_audit.py",
                    "--inspect-dir",
                    str(inspect_dir),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote JorJor-Inspect audit", result.stdout)

    def test_audit_can_include_prefix_trajectory_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inspect_dir = root / "inspect"
            trajectory_dir = root / "trajectory" / "seed_k"
            output_dir = root / "audit"
            _write_inspect_fixture(inspect_dir)
            trajectory_dir.mkdir(parents=True)
            (trajectory_dir / "prefix_trajectory_report.json").write_text(
                json.dumps({"final_diagnostic_label": TRAJECTORY_ALIGNED_SUPPORTED}, sort_keys=True),
                encoding="utf-8",
            )

            payload = build_audit_report(
                inspect_dir=inspect_dir,
                output_dir=output_dir,
                trajectory_dir=trajectory_dir.parent,
            )

        self.assertEqual(payload["final_verdict"], TRAJECTORY_ALIGNED_SUPPORTED)

    def test_prefix_baseline_can_override_trajectory_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inspect_dir = root / "inspect"
            trajectory_dir = root / "trajectory" / "seed_k"
            baseline_dir = root / "baseline" / "seed_k"
            output_dir = root / "audit"
            _write_inspect_fixture(inspect_dir)
            trajectory_dir.mkdir(parents=True)
            baseline_dir.mkdir(parents=True)
            (trajectory_dir / "prefix_trajectory_report.json").write_text(
                json.dumps({"final_diagnostic_label": TRAJECTORY_ALIGNED_SUPPORTED}, sort_keys=True),
                encoding="utf-8",
            )
            (baseline_dir / "prefix_composition_baseline_report.json").write_text(
                json.dumps({"final_diagnostic_label": COMPOSITION_SHORTCUT_SUPPORTED}, sort_keys=True),
                encoding="utf-8",
            )

            payload = build_audit_report(
                inspect_dir=inspect_dir,
                output_dir=output_dir,
                trajectory_dir=trajectory_dir.parent,
                baseline_dir=baseline_dir.parent,
            )

        self.assertEqual(payload["final_verdict"], COMPOSITION_SHORTCUT_SUPPORTED)

    def test_audit_records_synthetic_methodology_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inspect_dir = root / "inspect"
            synthetic_dir = root / "synthetic" / "smoke"
            toy_dir = root / "toy" / "smoke"
            output_dir = root / "audit"
            _write_inspect_fixture(inspect_dir)
            synthetic_dir.mkdir(parents=True)
            (synthetic_dir / "synthetic_oracle_belief_report.json").write_text(
                json.dumps(
                    {
                        "report_type": "synthetic_oracle_belief",
                        "cases": [
                            {
                                "mode": "oracle_belief",
                                "final_diagnostic_label": BELIEF_LIKE_ORACLE_SUPPORTED,
                            },
                            {
                                "mode": "composition_shortcut",
                                "final_diagnostic_label": COMPOSITION_SHORTCUT_SUPPORTED,
                            },
                        ],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            toy_dir.mkdir(parents=True)
            (toy_dir / "toy_imperfect_info_oracle_report.json").write_text(
                json.dumps(
                    {
                        "report_type": "toy_imperfect_info_oracle",
                        "cases": [
                            {
                                "mode": "oracle_belief",
                                "final_diagnostic_label": BELIEF_LIKE_ORACLE_SUPPORTED,
                            },
                            {
                                "mode": "composition_shortcut",
                                "final_diagnostic_label": COMPOSITION_SHORTCUT_SUPPORTED,
                            },
                        ],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            payload = build_audit_report(
                inspect_dir=inspect_dir,
                output_dir=output_dir,
                synthetic_dir=synthetic_dir.parent,
                toy_oracle_dir=toy_dir.parent,
            )
            tables = json.loads((output_dir / "tables.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["final_verdict"], COMPOSITION_SHORTCUT_SUPPORTED)
        self.assertEqual(payload["methodology_validation_status"], "passed")
        self.assertEqual(payload["toy_oracle_validation_status"], "passed")
        self.assertEqual(len(tables["synthetic_oracle_belief"]), 1)
        self.assertEqual(len(tables["toy_imperfect_info_oracle"]), 1)

    def test_audit_records_learned_synthetic_validation_and_summary_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inspect_dir = root / "inspect"
            learned_dir = root / "learned" / "smoke"
            output_dir = root / "audit"
            _write_inspect_fixture(inspect_dir)
            learned_dir.mkdir(parents=True)
            (learned_dir / "learned_synthetic_belief_report.json").write_text(
                json.dumps(
                    {
                        "report_type": "learned_synthetic_belief",
                        "cases": [
                            {
                                "mode": "ordered_belief_ar",
                                "final_diagnostic_label": BELIEF_LIKE_ORACLE_SUPPORTED,
                            },
                            {
                                "mode": "composition_biased_ar",
                                "final_diagnostic_label": COMPOSITION_SHORTCUT_SUPPORTED,
                            },
                        ],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            payload = build_audit_report(
                inspect_dir=inspect_dir,
                output_dir=output_dir,
                learned_synthetic_dir=learned_dir.parent,
            )
            report = (output_dir / "diagnostic_report.md").read_text(encoding="utf-8")
            tables = json.loads((output_dir / "tables.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["final_verdict"], COMPOSITION_SHORTCUT_SUPPORTED)
        self.assertEqual(payload["learned_methodology_validation_status"], "passed")
        self.assertIn("| Poker verdict | `composition_shortcut_supported` |", report)
        self.assertIn("| Learned synthetic validation | `passed` |", report)
        self.assertEqual(len(tables["learned_synthetic_belief"]), 1)

    def test_audit_records_one_command_reproduction_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inspect_dir = root / "inspect"
            reproduction_manifest = root / "repro" / "reproduction_manifest.json"
            output_dir = root / "audit"
            _write_inspect_fixture(inspect_dir)
            reproduction_manifest.parent.mkdir(parents=True)
            reproduction_manifest.write_text(
                json.dumps(
                    {
                        "archive_root": "/tmp/archive",
                        "command": "python scripts/reproduce_paper_tables.py --archive-root /tmp/archive",
                        "expected_outputs": ["paper_tables/final_diagnostic_tables.json"],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            payload = build_audit_report(
                inspect_dir=inspect_dir,
                output_dir=output_dir,
                reproduction_manifest_path=reproduction_manifest,
            )
            report = (output_dir / "diagnostic_report.md").read_text(encoding="utf-8")

        roles = {source["role"] for source in payload["source_manifest"]["sources"]}
        self.assertIn("one_command_reproduction_manifest", roles)
        self.assertIn("| Reproduction manifest | `present` |", report)

    def test_audit_records_strict_gate_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inspect_dir = root / "inspect"
            strict_dir = root / "strict"
            output_dir = root / "audit"
            _write_inspect_fixture(inspect_dir)
            strict_dir.mkdir()
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

            payload = build_audit_report(
                inspect_dir=inspect_dir,
                output_dir=output_dir,
                strict_table_dir=strict_dir,
            )
            report = (output_dir / "diagnostic_report.md").read_text(encoding="utf-8")
            tables = json.loads((output_dir / "tables.json").read_text(encoding="utf-8"))

        roles = {source["role"] for source in payload["source_manifest"]["sources"] if source["present"]}
        self.assertIn("range_supervised_strict_gate", roles)
        self.assertIn("probe_capacity_s6_s7", roles)
        self.assertEqual(payload["range_supervised_strict_gate_status"], COMPOSITION_SHORTCUT_SUPPORTED)
        self.assertEqual(payload["probe_capacity_s6_s7_status"], "mlp_inconclusive")
        self.assertEqual(tables["range_supervised_strict_gate"]["final_diagnostic_label"], COMPOSITION_SHORTCUT_SUPPORTED)
        self.assertEqual(tables["probe_capacity_s6_s7"]["final_verdict"], "mlp_inconclusive")
        self.assertIn("| Range-supervised strict gate | `composition_shortcut_supported` |", report)
        self.assertIn("| MLP capacity gate | `mlp_inconclusive` |", report)

    def test_audit_records_explicit_belief_state_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inspect_dir = root / "inspect"
            explicit_dir = root / "explicit" / "smoke"
            output_dir = root / "audit"
            _write_inspect_fixture(inspect_dir)
            explicit_dir.mkdir(parents=True)
            (explicit_dir / "explicit_belief_state_report.json").write_text(
                json.dumps(
                    {
                        "report_type": "explicit_belief_state_positive_control",
                        "cases": [
                            {
                                "mode": "explicit_belief_state_rnn",
                                "final_diagnostic_label": BELIEF_LIKE_ORACLE_SUPPORTED,
                            },
                            {
                                "mode": "composition_state_rnn",
                                "final_diagnostic_label": COMPOSITION_SHORTCUT_SUPPORTED,
                            },
                        ],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            payload = build_audit_report(
                inspect_dir=inspect_dir,
                output_dir=output_dir,
                explicit_belief_state_dir=explicit_dir.parent,
            )
            report = (output_dir / "diagnostic_report.md").read_text(encoding="utf-8")
            tables = json.loads((output_dir / "tables.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["explicit_methodology_validation_status"], "passed")
        self.assertIn("| Explicit belief-state validation | `passed` |", report)
        self.assertEqual(len(tables["explicit_belief_state"]), 1)

    def test_audit_records_non_poker_protocol_transfer_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inspect_dir = root / "inspect"
            latent_dir = root / "k1" / "smoke"
            output_dir = root / "audit"
            _write_inspect_fixture(inspect_dir)
            latent_dir.mkdir(parents=True)
            (latent_dir / "k.1_latent_type_protocol_transfer_report.json").write_text(
                json.dumps(
                    {
                        "report_type": "k.1_latent_type_protocol_transfer",
                        "protocol_transfer_supported": True,
                        "learned_positive_capacity_supported": False,
                        "cases": [
                            {
                                "mode": "oracle_posterior_state",
                                "final_diagnostic_label": "belief_like_protocol_transfer_supported",
                            },
                            {
                                "mode": "composition_shortcut_state",
                                "final_diagnostic_label": COMPOSITION_SHORTCUT_SUPPORTED,
                            },
                        ],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (latent_dir / "k.1_protocol_transfer_manifest.json").write_text(
                json.dumps({"report_type": "k.1_protocol_transfer_manifest"}, sort_keys=True),
                encoding="utf-8",
            )

            payload = build_audit_report(
                inspect_dir=inspect_dir,
                output_dir=output_dir,
                latent_protocol_dir=latent_dir.parent,
            )
            report = (output_dir / "diagnostic_report.md").read_text(encoding="utf-8")
            tables = json.loads((output_dir / "tables.json").read_text(encoding="utf-8"))

        roles = {source["role"] for source in payload["source_manifest"]["sources"] if source["present"]}
        self.assertEqual(payload["latent_protocol_transfer_status"], "passed")
        self.assertIn("| Non-poker protocol transfer | `passed` |", report)
        self.assertIn("k.1_latent_type_protocol_transfer_report.json", roles)
        self.assertIn("k.1_protocol_transfer_manifest.json", roles)
        self.assertEqual(len(tables["latent_protocol_transfer"]), 1)

    def test_audit_records_llm_latent_order_transfer_stress_test(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inspect_dir = root / "inspect"
            llm_dir = root / "k2" / "smoke"
            output_dir = root / "audit"
            _write_inspect_fixture(inspect_dir)
            llm_dir.mkdir(parents=True)
            (llm_dir / "k.2_llm_latent_order_report.json").write_text(
                json.dumps(
                    {
                        "report_type": "k.2_llm_latent_order_audit",
                        "cases": [
                            {
                                "backend": "hf_causal_lm",
                                "final_diagnostic_label": "order_information_supported",
                                "matched_composition_summary": {
                                    "mean_representation_distance": 0.000137,
                                },
                            }
                        ],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (llm_dir / "k.2_llm_latent_order_manifest.json").write_text(
                json.dumps({"report_type": "k.2_llm_latent_order_manifest"}, sort_keys=True),
                encoding="utf-8",
            )

            payload = build_audit_report(
                inspect_dir=inspect_dir,
                output_dir=output_dir,
                llm_latent_order_dir=llm_dir.parent,
            )
            report = (output_dir / "diagnostic_report.md").read_text(encoding="utf-8")
            tables = json.loads((output_dir / "tables.json").read_text(encoding="utf-8"))

        roles = {source["role"] for source in payload["source_manifest"]["sources"] if source["present"]}
        self.assertEqual(payload["llm_latent_order_status"], "order_information_supported")
        self.assertIn("| Frozen-LM latent-order audit | `order_information_supported` |", report)
        self.assertIn("k.2_llm_latent_order_report.json", roles)
        self.assertIn("k.2_llm_latent_order_manifest.json", roles)
        self.assertEqual(tables["llm_latent_order"][0]["cases"][0]["backend"], "hf_causal_lm")

    def test_audit_prefers_llm_full_sweep_aggregate_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inspect_dir = root / "inspect"
            llm_dir = root / "k2" / "single"
            full_sweep_dir = root / "k26"
            output_dir = root / "audit"
            _write_inspect_fixture(inspect_dir)
            llm_dir.mkdir(parents=True)
            full_sweep_dir.mkdir(parents=True)
            (llm_dir / "k.2_llm_latent_order_report.json").write_text(
                json.dumps(
                    {
                        "report_type": "k.2_llm_latent_order_audit",
                        "cases": [
                            {
                                "backend": "hf_causal_lm",
                                "final_diagnostic_label": "composition_shortcut_supported",
                            }
                        ],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (full_sweep_dir / "k.2.6_llm_full_sweep_report.json").write_text(
                json.dumps(
                    {
                        "report_type": "k.2.6_llm_full_sweep",
                        "main_sweep_label": "order_information_supported",
                        "final_sweep_label": "mixed",
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (full_sweep_dir / "k.2.6_llm_full_sweep_report.md").write_text("# k2.6\n", encoding="utf-8")
            (full_sweep_dir / "k.2.6_llm_full_sweep_manifest.json").write_text(
                json.dumps({"report_type": "k.2.6_llm_full_sweep_manifest"}, sort_keys=True),
                encoding="utf-8",
            )

            payload = build_audit_report(
                inspect_dir=inspect_dir,
                output_dir=output_dir,
                llm_latent_order_dir=llm_dir.parent,
                llm_full_sweep_dir=full_sweep_dir,
            )
            report = (output_dir / "diagnostic_report.md").read_text(encoding="utf-8")
            tables = json.loads((output_dir / "tables.json").read_text(encoding="utf-8"))

        roles = {source["role"] for source in payload["source_manifest"]["sources"] if source["present"]}
        self.assertEqual(payload["llm_latent_order_status"], "order_information_supported")
        self.assertIn("| Frozen-LM latent-order audit | `order_information_supported` |", report)
        self.assertIn("k.2.6_llm_full_sweep_report.json", roles)
        self.assertIn("k.2.6_llm_full_sweep_report.md", roles)
        self.assertIn("k.2.6_llm_full_sweep_manifest.json", roles)
        self.assertEqual(tables["llm_full_sweep"][0]["main_sweep_label"], "order_information_supported")

    def test_audit_records_cbps_claim_ladder_without_overriding_poker_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inspect_dir = root / "inspect"
            cbps_dir = root / "k5" / "claim_ladder"
            output_dir = root / "audit"
            _write_inspect_fixture(inspect_dir)
            cbps_dir.mkdir(parents=True)
            paper_safe_claim = (
                "Positive hidden-state range probes are insufficient evidence for ordered belief tracking."
            )
            (cbps_dir / "k.5.3_cbps_claim_ladder.json").write_text(
                json.dumps(
                    {
                        "report_type": "k.5.3_cbps_claim_ladder",
                        "final_label": "cbps_supported",
                        "claim_scope": "diagnostic_interpretation_not_ordered_belief",
                        "paper_safe_claim": paper_safe_claim,
                        "rows": [
                            {
                                "domain": "limit_holdem",
                                "final_label": "cbps_supported",
                                "predictive_gate": "predictive_supported",
                                "representation_behavior_link_gate": "composition_bounded_behavior_link",
                                "matched_range_label_proxy_gate": "range_label_proxy_tracking_not_supported",
                            }
                        ],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (cbps_dir / "k.5.3_cbps_manifest.json").write_text(
                json.dumps({"report_type": "k.5.3_cbps_manifest"}, sort_keys=True),
                encoding="utf-8",
            )
            (cbps_dir / "paper_table_cbps_claim_ladder.md").write_text(
                "| row | label |\n|---|---|\n| poker | cbps_supported |\n",
                encoding="utf-8",
            )

            payload = build_audit_report(
                inspect_dir=inspect_dir,
                output_dir=output_dir,
                cbps_claim_ladder_dir=cbps_dir,
            )
            report = (output_dir / "diagnostic_report.md").read_text(encoding="utf-8")
            tables = json.loads((output_dir / "tables.json").read_text(encoding="utf-8"))

        roles = {source["role"] for source in payload["source_manifest"]["sources"] if source["present"]}
        self.assertEqual(payload["final_verdict"], COMPOSITION_SHORTCUT_SUPPORTED)
        self.assertEqual(payload["cbps_claim_ladder_status"], "cbps_supported")
        self.assertEqual(payload["cbps_behavior_link_gate"], "composition_bounded_behavior_link")
        self.assertEqual(payload["cbps_claim_scope"], "diagnostic_interpretation_not_ordered_belief")
        self.assertEqual(payload["cbps_paper_safe_claim"], paper_safe_claim)
        self.assertIn("cbps_claim_ladder", roles)
        self.assertIn("cbps_claim_ladder_manifest", roles)
        self.assertIn("cbps_claim_ladder_paper_table", roles)
        self.assertIn("| Poker verdict | `composition_shortcut_supported` |", report)
        self.assertIn("| CBPS claim ladder | `cbps_supported` |", report)
        self.assertIn("| CBPS behavior link | `composition_bounded_behavior_link` |", report)
        self.assertIn("| CBPS claim scope | `diagnostic_interpretation_not_ordered_belief` |", report)
        self.assertEqual(tables["cbps_claim_ladder"]["final_label"], "cbps_supported")
        self.assertNotIn("requested CBPS claim-ladder report is missing", payload["gaps"])

    def test_audit_records_order_intervention_without_overriding_poker_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inspect_dir = root / "inspect"
            order_dir = root / "k6" / "order"
            output_dir = root / "audit"
            _write_inspect_fixture(inspect_dir)
            order_dir.mkdir(parents=True)
            (order_dir / "k.6.0_order_intervention_report.json").write_text(
                json.dumps(
                    {
                        "report_type": "k.6.0_order_intervention",
                        "final_diagnostic_label": "order_sensitive_stress_supported",
                        "claim_scope": "input_level_order_stress_not_legal_poker_causality",
                        "aggregate": {
                            "seed_count": 3,
                            "intervention_pair_count": 17337,
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (order_dir / "k.6.0_order_intervention_report.md").write_text("# order\n", encoding="utf-8")
            (order_dir / "k.6.0_order_intervention_manifest.json").write_text(
                json.dumps({"report_type": "k.6.0_order_intervention_manifest"}, sort_keys=True),
                encoding="utf-8",
            )

            payload = build_audit_report(
                inspect_dir=inspect_dir,
                output_dir=output_dir,
                order_intervention_dir=order_dir,
            )
            report = (output_dir / "diagnostic_report.md").read_text(encoding="utf-8")
            tables = json.loads((output_dir / "tables.json").read_text(encoding="utf-8"))

        roles = {source["role"] for source in payload["source_manifest"]["sources"] if source["present"]}
        self.assertEqual(payload["final_verdict"], COMPOSITION_SHORTCUT_SUPPORTED)
        self.assertEqual(payload["order_intervention_status"], "order_sensitive_stress_supported")
        self.assertEqual(
            payload["order_intervention_claim_scope"],
            "input_level_order_stress_not_legal_poker_causality",
        )
        self.assertEqual(payload["order_intervention_pair_count"], 17337)
        self.assertEqual(payload["order_intervention_seed_count"], 3)
        self.assertIn("order_intervention_report", roles)
        self.assertIn("order_intervention_report_md", roles)
        self.assertIn("order_intervention_manifest", roles)
        self.assertIn("| Order-intervention stress | `order_sensitive_stress_supported` |", report)
        self.assertEqual(tables["order_intervention"]["final_diagnostic_label"], "order_sensitive_stress_supported")
        self.assertNotIn("requested order-intervention report is missing", payload["gaps"])

    def test_audit_records_range_subspace_without_overriding_poker_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inspect_dir = root / "inspect"
            range_dir = root / "k63" / "range"
            output_dir = root / "audit"
            _write_inspect_fixture(inspect_dir)
            range_dir.mkdir(parents=True)
            (range_dir / "k.6.3_range_subspace_ablation_report.json").write_text(
                json.dumps(
                    {
                        "report_type": "k.6.3_range_subspace_ablation",
                        "final_diagnostic_label": "range_subspace_behavior_footprint_not_supported",
                        "claim_scope": "range_probe_subspace_behavior_footprint_stress_not_causal_mediation",
                        "aggregate": {
                            "metrics": {
                                "probe_top10_accuracy": {"mean": 0.12538886},
                                "range_minus_random_policy_js_delta": {"mean": -0.00407183},
                                "range_minus_random_value_delta": {"mean": -0.16412876},
                            }
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (range_dir / "k.6.3_range_subspace_ablation_report.md").write_text("# range\n", encoding="utf-8")
            (range_dir / "k.6.3_range_subspace_ablation_manifest.json").write_text(
                json.dumps({"report_type": "k.6.3_range_subspace_ablation_manifest"}, sort_keys=True),
                encoding="utf-8",
            )

            payload = build_audit_report(
                inspect_dir=inspect_dir,
                output_dir=output_dir,
                range_subspace_dir=range_dir,
            )
            report = (output_dir / "diagnostic_report.md").read_text(encoding="utf-8")
            tables = json.loads((output_dir / "tables.json").read_text(encoding="utf-8"))

        roles = {source["role"] for source in payload["source_manifest"]["sources"] if source["present"]}
        self.assertEqual(payload["final_verdict"], COMPOSITION_SHORTCUT_SUPPORTED)
        self.assertEqual(payload["range_subspace_status"], "range_subspace_behavior_footprint_not_supported")
        self.assertEqual(
            payload["range_subspace_claim_scope"],
            "range_probe_subspace_behavior_footprint_stress_not_causal_mediation",
        )
        self.assertEqual(payload["range_subspace_probe_top10_mean"], 0.12538886)
        self.assertEqual(payload["range_subspace_policy_delta_mean"], -0.00407183)
        self.assertEqual(payload["range_subspace_value_delta_mean"], -0.16412876)
        self.assertIn("range_subspace_report", roles)
        self.assertIn("range_subspace_report_md", roles)
        self.assertIn("range_subspace_manifest", roles)
        self.assertIn(
            "| Range-subspace stress | `range_subspace_behavior_footprint_not_supported` |",
            report,
        )
        self.assertEqual(
            tables["range_subspace"]["final_diagnostic_label"],
            "range_subspace_behavior_footprint_not_supported",
        )
        self.assertNotIn("requested range-subspace report is missing", payload["gaps"])

    def test_audit_records_range_rank_sensitivity_without_overriding_poker_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inspect_dir = root / "inspect"
            rank_dir = root / "k65" / "rank"
            output_dir = root / "audit"
            _write_inspect_fixture(inspect_dir)
            rank_dir.mkdir(parents=True)
            (rank_dir / "k.6.5_range_subspace_rank_sensitivity_report.json").write_text(
                json.dumps(
                    {
                        "report_type": "k.6.5_range_subspace_rank_sensitivity",
                        "final_diagnostic_label": "range_subspace_rank_sensitivity_not_supported",
                        "claim_scope": "rank_sensitivity_stress_not_causal_mediation",
                        "aggregate": {
                            "rank_count": 6,
                            "mean_policy_delta_across_ranks": -0.004126965157469471,
                            "mean_value_delta_across_ranks": -0.10528861085259278,
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (rank_dir / "k.6.5_range_subspace_rank_sensitivity_report.md").write_text(
                "# rank\n",
                encoding="utf-8",
            )
            (rank_dir / "k.6.5_range_subspace_rank_sensitivity_manifest.json").write_text(
                json.dumps({"report_type": "k.6.5_range_subspace_rank_sensitivity_manifest"}, sort_keys=True),
                encoding="utf-8",
            )

            payload = build_audit_report(
                inspect_dir=inspect_dir,
                output_dir=output_dir,
                range_rank_sensitivity_dir=rank_dir,
            )
            report = (output_dir / "diagnostic_report.md").read_text(encoding="utf-8")
            tables = json.loads((output_dir / "tables.json").read_text(encoding="utf-8"))

        roles = {source["role"] for source in payload["source_manifest"]["sources"] if source["present"]}
        self.assertEqual(payload["final_verdict"], COMPOSITION_SHORTCUT_SUPPORTED)
        self.assertEqual(payload["range_rank_sensitivity_status"], "range_subspace_rank_sensitivity_not_supported")
        self.assertEqual(payload["range_rank_sensitivity_claim_scope"], "rank_sensitivity_stress_not_causal_mediation")
        self.assertEqual(payload["range_rank_sensitivity_rank_count"], 6)
        self.assertEqual(payload["range_rank_sensitivity_policy_delta_mean"], -0.004126965157469471)
        self.assertEqual(payload["range_rank_sensitivity_value_delta_mean"], -0.10528861085259278)
        self.assertIn("range_rank_sensitivity_report", roles)
        self.assertIn("range_rank_sensitivity_report_md", roles)
        self.assertIn("range_rank_sensitivity_manifest", roles)
        self.assertIn(
            "| Range-rank sensitivity | `range_subspace_rank_sensitivity_not_supported` |",
            report,
        )
        self.assertEqual(
            tables["range_rank_sensitivity"]["final_diagnostic_label"],
            "range_subspace_rank_sensitivity_not_supported",
        )
        self.assertNotIn("requested range-rank-sensitivity report is missing", payload["gaps"])


def _write_inspect_fixture(inspect_dir: Path) -> None:
    inspect_dir.mkdir(parents=True, exist_ok=True)
    (inspect_dir / "belief_shortcut_report.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "final_diagnostic_label": COMPOSITION_SHORTCUT_SUPPORTED,
                    }
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (inspect_dir / "matched_composition_report.json").write_text(
        json.dumps({"final_diagnostic_label": COMPOSITION_SHORTCUT_SUPPORTED}, sort_keys=True),
        encoding="utf-8",
    )
    (inspect_dir / "claim_gate_matrix.json").write_text(
        json.dumps({"rows": []}, sort_keys=True),
        encoding="utf-8",
    )
    for name in (
        "belief_shortcut_report.md",
        "matched_composition_report.md",
        "claim_gate_matrix.md",
    ):
        (inspect_dir / name).write_text(f"# {name}\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
