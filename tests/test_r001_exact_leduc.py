from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from jorjor.envs.leduc import LeducPolicyGenerator
from jorjor.envs.leduc_exact import (
    COMPOSITION_COLLAPSED,
    INSUFFICIENT_EXACT_POSTERIOR_SUPPORT,
    StochasticLeducPolicy,
    build_exact_leduc_feasibility_report,
    exact_leduc_rows,
    exact_opponent_card_posterior,
    matched_exact_posterior_rows,
    matched_exact_posterior_summary,
)


class ExactLeducFeasibilityTest(unittest.TestCase):
    """Test r.0.1 exact-posterior Leduc feasibility helpers."""

    def test_exact_posterior_is_normalized_and_blocks_visible_cards(self) -> None:
        policy = StochasticLeducPolicy(random.Random(11))
        generator = LeducPolicyGenerator((policy, policy), seed=17)
        point = generator.generate_hand(0).decision_points[0]

        posterior = exact_opponent_card_posterior(point=point, policy=policy)

        self.assertAlmostEqual(sum(posterior), 1.0)
        own_index = ("J0", "J1", "Q0", "Q1", "K0", "K1").index(point.visible_state["private_card"])
        self.assertEqual(posterior[own_index], 0.0)

    def test_strict_leduc_lacks_matched_order_support(self) -> None:
        policy = StochasticLeducPolicy(random.Random(23))
        generator = LeducPolicyGenerator((policy, policy), seed=29)
        points = tuple(point for hand in generator.generate_hands(512) for point in hand.decision_points)
        rows = exact_leduc_rows(points=points, policy=policy)
        matched = matched_exact_posterior_rows(rows, max_pairs=512)

        self.assertEqual(len(matched), 0)
        self.assertEqual(sum(float(row["composition_representation_js"]) for row in matched), 0.0)

    def test_matched_summary_uses_bucket_estimand_for_point_and_ci(self) -> None:
        rows = [
            _matched_row("bucket_a", 1.0),
            _matched_row("bucket_a", 1.0),
            _matched_row("bucket_b", 0.0),
        ]

        summary = matched_exact_posterior_summary(
            rows,
            bootstrap_count=20,
            bootstrap_seed=7,
            confidence_level=0.95,
        )

        self.assertEqual(summary["estimand"], "unweighted_composition_bucket_mean")
        self.assertAlmostEqual(summary["mean_posterior_js"], 0.5)
        self.assertAlmostEqual(summary["posterior_js_ci"]["mean"], summary["mean_posterior_js"])
        self.assertAlmostEqual(summary["pair_table_mean_posterior_js"], 2.0 / 3.0)

    def test_report_builder_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "r001"
            payload = build_exact_leduc_feasibility_report(
                output_dir=output_dir,
                hand_count=1000,
                seed=31,
                max_pairs=512,
                bootstrap_count=100,
                min_bucket_count=1,
                min_pair_multiplicity=1,
            )
            saved = json.loads((output_dir / "r.0.1_exact_leduc_feasibility_report.json").read_text(encoding="utf-8"))
            manifest_exists = (output_dir / "r.0.1_exact_leduc_feasibility_manifest.json").exists()

        self.assertEqual(payload["report_type"], "r.0.1_exact_leduc_feasibility")
        self.assertEqual(saved["version"], "r.0.1")
        self.assertEqual(payload["oracle_detection_label"], INSUFFICIENT_EXACT_POSTERIOR_SUPPORT)
        self.assertEqual(payload["composition_detection_label"], COMPOSITION_COLLAPSED)
        self.assertEqual(payload["final_diagnostic_label"], INSUFFICIENT_EXACT_POSTERIOR_SUPPORT)
        self.assertGreater(len(payload["blocking_reasons"]), 0)
        self.assertTrue(manifest_exists)

    def test_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_r001_exact_leduc_feasibility.py",
                    "--output-dir",
                    str(Path(temp_dir) / "r001"),
                    "--hand-count",
                    "1000",
                    "--max-pairs",
                    "512",
                    "--bootstrap-count",
                    "100",
                    "--min-bucket-count",
                    "1",
                    "--min-pair-multiplicity",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote r.0.1 exact Leduc feasibility report", result.stdout)

def _matched_row(bucket: str, posterior_js: float) -> dict[str, object]:
    return {
        "composition_key": bucket,
        "left_order_key": f"{bucket}:left",
        "right_order_key": f"{bucket}:right",
        "left_count": 1,
        "right_count": 1,
        "pair_multiplicity": 1,
        "posterior_js": posterior_js,
        "oracle_representation_js": posterior_js,
        "composition_representation_js": 0.0,
        "left_example": {},
        "right_example": {},
    }


if __name__ == "__main__":
    unittest.main()
