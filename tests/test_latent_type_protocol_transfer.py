from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jorjor.envs.latent_type_sequence import (
    PROTOCOL_TRANSFER_SUPPORTED,
    composition_state,
    exact_posterior,
    matched_order_rows,
    matched_order_summary,
)
from jorjor.probe.inspect import COMPOSITION_SHORTCUT_SUPPORTED
from jorjor.train.latent_type_sequence import build_latent_type_multiseed_pack, build_latent_type_protocol_transfer_report


class LatentTypeProtocolTransferTest(unittest.TestCase):
    """Test the k.1 non-poker protocol-transfer benchmark."""

    def test_exact_posterior_is_order_sensitive(self) -> None:
        left = ("a", "b", "c", "d")
        right = ("d", "c", "b", "a")

        left_posterior = exact_posterior(left)
        right_posterior = exact_posterior(right)

        self.assertAlmostEqual(sum(left_posterior), 1.0)
        self.assertAlmostEqual(sum(right_posterior), 1.0)
        self.assertNotEqual(left_posterior, right_posterior)

    def test_matched_order_rows_separate_oracle_from_composition(self) -> None:
        oracle_rows = matched_order_rows(
            length=5,
            representation_fn=exact_posterior,
            max_order_pairs=128,
        )
        shortcut_rows = matched_order_rows(
            length=5,
            representation_fn=composition_state,
            max_order_pairs=128,
        )

        self.assertGreater(len(oracle_rows), 0)
        self.assertGreater(sum(float(row["representation_js"]) for row in oracle_rows), 0.0)
        self.assertEqual(sum(float(row["representation_js"]) for row in shortcut_rows), 0.0)

    def test_matched_order_summary_uses_bucket_estimand_for_point_and_ci(self) -> None:
        rows = [
            {"composition_key": "bucket_a", "posterior_js": 1.0, "representation_js": 1.0},
            {"composition_key": "bucket_a", "posterior_js": 1.0, "representation_js": 1.0},
            {"composition_key": "bucket_a", "posterior_js": 1.0, "representation_js": 1.0},
            {"composition_key": "bucket_b", "posterior_js": 0.0, "representation_js": 0.0},
        ]

        summary = matched_order_summary(
            rows,
            bootstrap_count=20,
            bootstrap_seed=11,
        )

        self.assertEqual(summary["estimand"], "unweighted_composition_bucket_mean")
        self.assertAlmostEqual(summary["mean_representation_js"], 0.5)
        self.assertAlmostEqual(summary["representation_js_ci"]["bootstrap_mean"], summary["mean_representation_js"])
        self.assertAlmostEqual(summary["pair_weighted_mean_representation_js"], 0.75)

    def test_protocol_transfer_report_labels_positive_and_shortcut_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "k1"
            payload = build_latent_type_protocol_transfer_report(
                output_dir=output_dir,
                sequence_count=64,
                length=5,
                seed=19,
                epochs=12,
                hidden_size=24,
                batch_size=64,
                max_order_pairs=128,
                bootstrap_count=100,
                bootstrap_seed=23,
                correlation_threshold=0.80,
            )
            saved = json.loads((output_dir / "k.1_latent_type_protocol_transfer_report.json").read_text(encoding="utf-8"))
            manifest_exists = (output_dir / "k.1_protocol_transfer_manifest.json").exists()

            labels = {case["mode"]: case["final_diagnostic_label"] for case in payload["cases"]}
            for case in payload["cases"]:
                matched = case["matched_composition_summary"]
                self.assertAlmostEqual(
                    matched["mean_representation_js"],
                    matched["representation_js_ci"]["bootstrap_mean"],
                )
        self.assertEqual(labels["oracle_posterior_state"], PROTOCOL_TRANSFER_SUPPORTED)
        self.assertEqual(labels["composition_shortcut_state"], COMPOSITION_SHORTCUT_SUPPORTED)
        self.assertIn(labels["belief_aux_gru_posterior_head"], {PROTOCOL_TRANSFER_SUPPORTED, "inconclusive"})
        self.assertIn("belief_aux_gru_hidden_probe", labels)
        self.assertEqual(labels["composition_biased_learned_state"], COMPOSITION_SHORTCUT_SUPPORTED)
        self.assertIn("action_value_only_gru_hidden_probe", labels)
        self.assertTrue(payload["protocol_transfer_supported"])
        self.assertEqual(saved["report_type"], "k.1_latent_type_protocol_transfer")
        self.assertTrue(saved["protocol_transfer_supported"])
        self.assertTrue(manifest_exists)

    def test_multiseed_pack_aggregates_seed_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "inputs"
            output_dir = Path(temp_dir) / "pack"
            _write_fake_seed_report(input_dir / "seed_a", seed=101, learned_supported=True)
            _write_fake_seed_report(input_dir / "seed_b", seed=102, learned_supported=False)

            payload = build_latent_type_multiseed_pack(input_dir=input_dir, output_dir=output_dir)
            saved = json.loads((output_dir / "k.1_multiseed_protocol_transfer_pack.json").read_text(encoding="utf-8"))
            manifest = json.loads((output_dir / "k.1_multiseed_manifest.json").read_text(encoding="utf-8"))

        self.assertTrue(payload["robust_protocol_transfer_supported"])
        self.assertEqual(payload["report_count"], 2)
        self.assertEqual(payload["learned_positive_capacity_supported_count"], 1)
        self.assertEqual(saved["report_type"], "k.1_multiseed_protocol_transfer_pack")
        self.assertEqual(manifest["report_checksum"], saved["checksum"])
        self.assertIn("protocol_transfer_validation_not_poker_belief_claim_upgrade", payload["claim_scope"])
        oracle = payload["case_summaries"]["oracle_posterior_state"]
        self.assertEqual(oracle["label_counts"][PROTOCOL_TRANSFER_SUPPORTED], 2)
        self.assertAlmostEqual(oracle["mean_bucket_representation_js"], 0.47)
        self.assertEqual(len(payload["seed_rows"]), 2)
        self.assertIn("source_checksum", payload["seed_rows"][0])

    def test_multiseed_pack_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "inputs"
            output_dir = Path(temp_dir) / "pack"
            _write_fake_seed_report(input_dir / "seed_a", seed=101, learned_supported=True)
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_latent_type_multiseed_pack.py",
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote k.1 multiseed protocol-transfer pack", result.stdout)

    def test_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_latent_type_protocol_transfer_report.py",
                    "--output-dir",
                    str(Path(temp_dir) / "k1"),
                    "--sequence-count",
                    "64",
                    "--length",
                    "5",
                    "--epochs",
                    "12",
                    "--hidden-size",
                    "24",
                    "--batch-size",
                    "64",
                    "--max-order-pairs",
                    "128",
                    "--bootstrap-count",
                    "100",
                    "--correlation-threshold",
                    "0.80",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote k.1 latent-type protocol-transfer report", result.stdout)


def _write_fake_seed_report(output_dir: Path, *, seed: int, learned_supported: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        _fake_case("oracle_posterior_state", PROTOCOL_TRANSFER_SUPPORTED, 0.47),
        _fake_case("composition_shortcut_state", COMPOSITION_SHORTCUT_SUPPORTED, 0.0),
        _fake_case(
            "belief_aux_gru_posterior_head",
            PROTOCOL_TRANSFER_SUPPORTED if learned_supported else "inconclusive",
            0.45 if learned_supported else 0.20,
        ),
        _fake_case("belief_aux_gru_hidden_probe", "inconclusive", 0.18),
        _fake_case("composition_biased_learned_state", COMPOSITION_SHORTCUT_SUPPORTED, 0.0),
        _fake_case("action_value_only_gru_hidden_probe", "inconclusive", 0.19),
    ]
    payload = {
        "version": "k.1",
        "report_type": "k.1_latent_type_protocol_transfer",
        "claim_scope": "protocol_transfer_validation_not_poker_belief_claim_upgrade",
        "seed": seed,
        "bootstrap_count": 100,
        "bootstrap_seed": 23,
        "protocol_transfer_supported": True,
        "learned_positive_capacity_supported": learned_supported,
        "cases": cases,
        "checksum": f"fake-checksum-{seed}",
    }
    (output_dir / "k.1_latent_type_protocol_transfer_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fake_case(mode: str, verdict: str, representation_js: float) -> dict[str, object]:
    return {
        "mode": mode,
        "training_target": "fake_target",
        "final_diagnostic_label": verdict,
        "summary": {
            "posterior_representation_correlation": 1.0 if representation_js > 0 else 0.0,
        },
        "matched_composition_summary": {
            "estimand": "unweighted_composition_bucket_mean",
            "matched_bucket_count": 2,
            "pair_count": 4,
            "mean_posterior_js": 0.47,
            "mean_representation_js": representation_js,
            "posterior_representation_correlation": 1.0 if representation_js > 0 else 0.0,
            "pair_weighted_mean_representation_js": representation_js + 0.01,
            "representation_js_ci": {
                "bootstrap_mean": representation_js,
                "bootstrap_lower": max(0.0, representation_js - 0.01),
                "bootstrap_upper": representation_js + 0.01,
            },
        },
    }


if __name__ == "__main__":
    unittest.main()
