from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jorjor.probe.tables import (
    build_final_diagnostic_tables,
    build_no_range_bootstrap_pack,
    build_no_range_composition_nuisance_pack,
    build_no_range_evidence_pack,
    build_no_range_fresh50_control_pack,
    build_no_range_history_composition_pack,
    build_no_range_locked_layer_pack,
    build_no_range_uncertainty_pack,
    build_paper_evidence_pack,
    build_paper_table_reproduction,
    build_probe_tables,
    build_range_supervised_positive_control_pack,
)


def _write_metrics(path: Path, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "checksum": f"checksum-{path.parent.name}",
                "records": [
                    {"control": "none", "layer": "encoder_layer_1", "score": score},
                    {"control": "random_label", "layer": "encoder_layer_1", "score": 0.09},
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_layer_metrics(
    path: Path,
    sequence_score: float,
    best_score: float,
    *,
    label_control: str = "none",
    metadata: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checksum": f"checksum-{path.parent.name}",
        "feature_control": "none",
        "input_control": "none",
        "label_control": label_control,
        "label_control_stats": {"control": label_control},
        "records": [
            {
                "control": "none",
                "layer": "sequence_hidden",
                "score": sequence_score,
            },
            {
                "control": "none",
                "layer": "encoder_layer_2",
                "score": best_score,
            },
            {
                "control": "random_label",
                "layer": "sequence_hidden",
                "score": 0.08,
            },
        ],
    }
    if metadata:
        payload.update(metadata)
    path.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )


def _path(root: Path, commit: str, seed: str, suffix: str, *, legacy_layout: bool = False) -> Path:
    suffix_text = f"_{suffix}" if suffix else ""
    parent = root / commit / "outputs"
    if not legacy_layout:
        parent = parent / "probes"
    return (
        parent
        / f"limit_ar_teacher_{seed}_4layer_range{suffix_text}_10k"
        / "probe_metrics.json"
    )


def _no_range_path(root: Path, commit: str, seed: str, suffix: str) -> Path:
    suffix_text = f"_{suffix}" if suffix else ""
    return (
        root
        / commit
        / "outputs"
        / "probes"
        / f"limit_ar_action_value_only_{seed}_4layer_range{suffix_text}_10k"
        / "probe_metrics.json"
    )


class ProbeTablesTest(unittest.TestCase):
    """Test archived probe table generation."""

    def test_build_probe_tables_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_required_metrics(root, full_commit="full123")

            payload = build_probe_tables(root, output_dir, full_control_commit="full123")
            table_json = json.loads((output_dir / "limit_ar_nuisance_summary.json").read_text(encoding="utf-8"))
            table_md = (output_dir / "limit_ar_nuisance_summary.md").read_text(encoding="utf-8")

        self.assertEqual(len(payload["rows"]), 3)
        self.assertEqual(table_json["full_control_commit"], "full123")
        self.assertEqual(table_json["base_range_commits"]["seed_a"], "ff94fab")
        self.assertIn("seed_a", table_md)
        self.assertIn("history hidden", table_md)

    def test_build_probe_tables_fails_when_required_metric_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                build_probe_tables(Path(temp_dir) / "archive", Path(temp_dir) / "tables", full_control_commit="full123")

    def test_build_probe_tables_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_required_metrics(root, full_commit="full123")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_probe_tables.py",
                    "--archive-root",
                    str(root),
                    "--output-dir",
                    str(output_dir),
                    "--full-control-commit",
                    "full123",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote rows=3", result.stdout)

    def test_build_paper_evidence_pack_writes_claim_and_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_required_metrics(root, full_commit="349304c")
            _write_robustness_summary(root / "8502bf8" / "v0_9_0_summary.json", 0.84, 0.15, 0.02, 0.01)
            _write_scale_replication_summary(root / "8a7a72b" / "v0_9_4_summary.json")
            _write_robustness_summary(root / "7180a68" / "v0_9_1_summary.json", 0.59, 0.13, -0.01, -0.02)
            _write_robustness_summary(root / "6da81c9" / "v0_9_2_summary.json", 0.84, 0.14, -0.01, 0.002)

            payload = build_paper_evidence_pack(root, output_dir)
            table_json = json.loads((output_dir / "limit_ar_paper_evidence_pack.json").read_text(encoding="utf-8"))
            table_md = (output_dir / "limit_ar_paper_evidence_pack.md").read_text(encoding="utf-8")

        self.assertEqual(payload["depth_stabilized"]["history_passes"], False)
        self.assertEqual(table_json["scale_robustness"]["seed_count"], 3)
        self.assertEqual(table_json["scale_robustness"]["all_history_pass"], True)
        self.assertIn("Depth robustness is not established", table_md)

    def test_build_paper_evidence_pack_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_required_metrics(root, full_commit="349304c")
            _write_robustness_summary(root / "8502bf8" / "v0_9_0_summary.json", 0.84, 0.15, 0.02, 0.01)
            _write_scale_replication_summary(root / "8a7a72b" / "v0_9_4_summary.json")
            _write_robustness_summary(root / "7180a68" / "v0_9_1_summary.json", 0.59, 0.13, -0.01, -0.02)
            _write_robustness_summary(root / "6da81c9" / "v0_9_2_summary.json", 0.84, 0.14, -0.01, 0.002)

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_paper_evidence_pack.py",
                    "--archive-root",
                    str(root),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote paper evidence pack", result.stdout)

    def test_build_no_range_evidence_pack_writes_sequence_and_best_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_no_range_archive(root, commit="abc123")

            payload = build_no_range_evidence_pack(root, output_dir, commit="abc123")
            table_json = json.loads((output_dir / "limit_ar_no_range_evidence_pack.json").read_text(encoding="utf-8"))
            table_md = (output_dir / "limit_ar_no_range_evidence_pack.md").read_text(encoding="utf-8")

        self.assertEqual(payload["primary_layer_rule"], "sequence_hidden")
        self.assertEqual(table_json["best_layer_rule"], "descriptive_only")
        self.assertEqual(len(table_json["core_rows"]), 3)
        self.assertGreater(table_json["core_rows"][0]["history_residual_best_delta_vs_action_value_only"], 0.0)
        self.assertIn("best_over_layers", table_md)

    def test_build_no_range_evidence_pack_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_no_range_archive(root, commit="abc123")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_no_range_evidence_pack.py",
                    "--archive-root",
                    str(root),
                    "--output-dir",
                    str(output_dir),
                    "--commit",
                    "abc123",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote no-range evidence pack", result.stdout)

    def test_build_no_range_uncertainty_pack_writes_repeat_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_no_range_archive(root, commit="abc123")
            _write_no_range_repeat_archive(root, commit="abc123")

            payload = build_no_range_uncertainty_pack(root, output_dir, commit="abc123")
            table_json = json.loads(
                (output_dir / "limit_ar_no_range_uncertainty_pack.json").read_text(encoding="utf-8")
            )
            table_md = (output_dir / "limit_ar_no_range_uncertainty_pack.md").read_text(encoding="utf-8")

        self.assertEqual(payload["probe_repeat_count"], 4)
        self.assertEqual(payload["paired_repeat_count"], 4)
        self.assertGreater(table_json["sequence_hidden_mean_std"], 0.0)
        self.assertGreater(table_json["sequence_hidden_delta_mean"], 0.0)
        self.assertGreater(table_json["encoder_layer_2_delta_mean"], 0.0)
        self.assertEqual(len(table_json["baseline_rows"]), 3)
        self.assertEqual(len(table_json["delta_rows"]), 6)
        self.assertEqual(table_json["rows"][0]["probe_seeds"], [230, 367, 379, 389])
        self.assertEqual(table_json["baseline_rows"][0]["probe_seeds"], [229, 467, 487, 499])
        self.assertEqual(
            table_json["delta_rows"][0]["residual_probe_seeds"],
            [230, 367, 379, 389],
        )
        self.assertEqual(
            table_json["delta_rows"][0]["baseline_probe_seeds"],
            [229, 467, 487, 499],
        )
        self.assertEqual(table_json["delta_rows"][0]["pair_metadata"][0]["repeat_index"], 0)
        self.assertIn("encoder_layer_2", table_md)
        self.assertIn("Paired Deltas", table_md)

    def test_build_no_range_uncertainty_pack_rejects_unmatched_pair_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_no_range_archive(root, commit="abc123")
            _write_no_range_repeat_archive(root, commit="abc123")
            path = _no_range_path(
                root,
                "abc123",
                "seed_h",
                "history_only_action_value_only_repeat_1",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["train_config"] = "configs/train/wrong.json"
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "train_config"):
                build_no_range_uncertainty_pack(root, output_dir, commit="abc123")

    def test_build_no_range_uncertainty_pack_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_no_range_archive(root, commit="abc123")
            _write_no_range_repeat_archive(root, commit="abc123")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_no_range_uncertainty_pack.py",
                    "--archive-root",
                    str(root),
                    "--output-dir",
                    str(output_dir),
                    "--commit",
                    "abc123",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote no-range uncertainty pack", result.stdout)

    def test_build_no_range_locked_layer_pack_writes_locked_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_locked_layer_archive(root, commit="fresh123")

            payload = build_no_range_locked_layer_pack(root, output_dir, commit="fresh123")
            table_json = json.loads(
                (output_dir / "limit_ar_no_range_locked_layer_pack.json").read_text(encoding="utf-8")
            )
            table_md = (output_dir / "limit_ar_no_range_locked_layer_pack.md").read_text(encoding="utf-8")

        self.assertEqual(payload["locked_candidate_layer"], "encoder_layer_2")
        self.assertEqual(len(table_json["seed_rows"]), 3)
        self.assertEqual(len(table_json["delta_rows"]), 6)
        self.assertTrue(table_json["encoder_layer_2_all_positive"])
        self.assertIn("best_over_layers", table_md)

    def test_build_no_range_locked_layer_pack_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_locked_layer_archive(root, commit="fresh123")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_no_range_locked_layer_pack.py",
                    "--archive-root",
                    str(root),
                    "--output-dir",
                    str(output_dir),
                    "--commit",
                    "fresh123",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote no-range locked-layer pack", result.stdout)

    def test_build_no_range_bootstrap_pack_writes_paired_ci_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_locked_layer_archive(root, commit="fresh123")

            payload = build_no_range_bootstrap_pack(
                root,
                output_dir,
                commit="fresh123",
                bootstrap_count=50,
                bootstrap_seed=17,
            )
            table_json = json.loads(
                (output_dir / "limit_ar_no_range_bootstrap_pack.json").read_text(encoding="utf-8")
            )
            table_md = (output_dir / "limit_ar_no_range_bootstrap_pack.md").read_text(encoding="utf-8")

        self.assertEqual(len(payload["ci_rows"]), 6)
        self.assertEqual(table_json["bootstrap_count"], 50)
        self.assertEqual(table_json["ci_rows"][0]["sample_count"], 20)
        self.assertEqual(table_json["ci_rows"][0]["hand_count"], 10)
        self.assertEqual(table_json["ci_rows"][0]["resampling_unit"], "hand_id")
        self.assertGreater(table_json["encoder_layer_2_mean_delta"], 0.0)
        self.assertIn("encoder_layer_2", table_md)
        self.assertIn("point_id", table_md)

    def test_build_no_range_bootstrap_pack_rejects_unmatched_example_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_locked_layer_archive(root, commit="fresh123")
            baseline_path = _no_range_path_for_size(
                root,
                "fresh123",
                "seed_k",
                "history_only_action_value_only",
                "50k",
            )
            correctness_path = baseline_path.parent / "probe_correctness.jsonl"
            rows = correctness_path.read_text(encoding="utf-8").splitlines()
            edited = []
            changed = False
            for line in rows:
                row = json.loads(line)
                if not changed and row["layer"] == "nuisance" and row["control"] == "none":
                    row["point_id"] = "wrong:99"
                    changed = True
                edited.append(json.dumps(row, sort_keys=True))
            correctness_path.write_text("\n".join(edited) + "\n", encoding="utf-8")
            payload = json.loads(baseline_path.read_text(encoding="utf-8"))
            payload["correctness_sha256"] = _file_sha256(correctness_path)
            baseline_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "point_id"):
                build_no_range_bootstrap_pack(root, output_dir, commit="fresh123", bootstrap_count=10)

    def test_build_no_range_bootstrap_pack_rejects_stale_correctness(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_locked_layer_archive(root, commit="fresh123")
            residual_path = _no_range_path_for_size(
                root,
                "fresh123",
                "seed_k",
                "history_only_residualize_action_value",
                "50k",
            )
            correctness_path = residual_path.parent / "probe_correctness.jsonl"
            correctness_path.write_text(correctness_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "sha256"):
                build_no_range_bootstrap_pack(root, output_dir, commit="fresh123", bootstrap_count=10)

    def test_build_no_range_bootstrap_pack_rejects_metrics_correctness_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_locked_layer_archive(root, commit="fresh123")
            residual_path = _no_range_path_for_size(
                root,
                "fresh123",
                "seed_k",
                "history_only_residualize_action_value",
                "50k",
            )
            payload = json.loads(residual_path.read_text(encoding="utf-8"))
            payload["records"][0]["score"] = 0.0
            residual_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "aggregate"):
                build_no_range_bootstrap_pack(root, output_dir, commit="fresh123", bootstrap_count=10)

    def test_build_range_supervised_positive_control_pack_writes_s5_s7_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_range_supervised_positive_control_archive(root, commit="pos123")

            payload = build_range_supervised_positive_control_pack(
                root,
                output_dir,
                commit="pos123",
                bootstrap_count=50,
                bootstrap_seed=23,
            )
            table_json = json.loads(
                (output_dir / "limit_ar_range_supervised_positive_control_pack.json").read_text(encoding="utf-8")
            )
            table_md = (
                output_dir / "limit_ar_range_supervised_positive_control_pack.md"
            ).read_text(encoding="utf-8")

        self.assertEqual(payload["final_diagnostic_label"], "belief_like_real_poker_positive_control")
        self.assertEqual(table_json["row_count"], 9)
        self.assertGreater(table_json["s5_min_lower_delta"], 0.0)
        self.assertGreater(table_json["s6_min_lower_delta"], 0.0)
        self.assertGreater(table_json["s7_min_lower_delta"], 0.0)
        self.assertIn("matched_composition", table_md)
        self.assertIn("range-supervised", table_md)

    def test_build_range_supervised_positive_control_pack_requires_correctness(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_range_supervised_positive_control_archive(root, commit="pos123")
            stale_path = _range_supervised_path_for_size(
                root,
                "pos123",
                "seed_d",
                "history_only_residualize_action_value",
                "50k",
            )
            (stale_path.parent / "probe_correctness.jsonl").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "correctness"):
                build_range_supervised_positive_control_pack(
                    root,
                    output_dir,
                    commit="pos123",
                    bootstrap_count=10,
                    bootstrap_seed=23,
                )

    def test_build_range_supervised_positive_control_pack_allows_legacy_teacher_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_range_supervised_positive_control_archive(root, commit="pos123")
            train_path = (
                root
                / "pos123"
                / "outputs"
                / "limit_ar_teacher_seed_d_50k_4layer"
                / "metrics.json"
            )
            train = json.loads(train_path.read_text(encoding="utf-8"))
            train.pop("opponent_hand_supervised")
            train.pop("opponent_hand_loss_weight")
            train_path.write_text(json.dumps(train, sort_keys=True), encoding="utf-8")

            payload = build_range_supervised_positive_control_pack(
                root,
                output_dir,
                commit="pos123",
                bootstrap_count=20,
                bootstrap_seed=23,
            )

        seed_d_rows = [row for row in payload["rows"] if row["seed"] == "seed_d"]
        self.assertEqual(seed_d_rows[0]["supervision_status"], "legacy_teacher_range_supervision_inferred")

    def test_build_range_supervised_positive_control_pack_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_range_supervised_positive_control_archive(root, commit="pos123")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_range_supervised_positive_control_pack.py",
                    "--archive-root",
                    str(root),
                    "--output-dir",
                    str(output_dir),
                    "--commit",
                    "pos123",
                    "--bootstrap-count",
                    "20",
                    "--bootstrap-seed",
                    "23",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote range-supervised positive-control pack rows=9", result.stdout)

    def test_build_no_range_bootstrap_pack_rejects_range_supervised_train_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_locked_layer_archive(root, commit="fresh123")
            train_path = (
                root
                / "fresh123"
                / "outputs"
                / "limit_ar_action_value_only_seed_k_50k_4layer"
                / "metrics.json"
            )
            payload = json.loads(train_path.read_text(encoding="utf-8"))
            payload["opponent_hand_supervised"] = True
            train_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "opponent_hand_supervised"):
                build_no_range_bootstrap_pack(root, output_dir, commit="fresh123", bootstrap_count=10)

    def test_build_no_range_bootstrap_pack_rejects_config_fallback_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_locked_layer_archive(root, commit="fresh123")
            residual_path = _no_range_path_for_size(
                root,
                "fresh123",
                "seed_k",
                "history_only_residualize_action_value",
                "50k",
            )
            payload = json.loads(residual_path.read_text(encoding="utf-8"))
            del payload["probe_seed"]
            residual_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "strict probe metadata"):
                build_no_range_bootstrap_pack(root, output_dir, commit="fresh123", bootstrap_count=10)

    def test_build_no_range_bootstrap_pack_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_locked_layer_archive(root, commit="fresh123")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_no_range_bootstrap_pack.py",
                    "--archive-root",
                    str(root),
                    "--output-dir",
                    str(output_dir),
                    "--commit",
                    "fresh123",
                    "--bootstrap-count",
                    "50",
                    "--bootstrap-seed",
                    "17",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote no-range bootstrap pack", result.stdout)

    def test_build_no_range_fresh50_control_pack_writes_supplemental_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_no_range_archive(root, commit="base10k")
            _write_locked_layer_archive(root, commit="fresh123")
            _write_fresh50_control_archive(root, commit="fresh123")

            payload = build_no_range_fresh50_control_pack(
                root,
                output_dir,
                commit_50k="fresh123",
                commit_10k="base10k",
                bootstrap_count=50,
                bootstrap_seed=17,
            )
            table_json = json.loads(
                (output_dir / "limit_ar_no_range_fresh50_control_pack.json").read_text(encoding="utf-8")
            )
            table_md = (output_dir / "limit_ar_no_range_fresh50_control_pack.md").read_text(encoding="utf-8")

        self.assertEqual(len(payload["repeat_rows"]), 18)
        self.assertEqual(len(table_json["label_rows"]), 6)
        self.assertEqual(len(table_json["history_rows"]), 42)
        self.assertEqual(len(table_json["repeat_ci_rows"]), 18)
        self.assertEqual(len(table_json["history_delta_ci_rows"]), 12)
        self.assertEqual({row["size"] for row in table_json["scaling_rows"]}, {"10k", "50k"})
        self.assertIn("encoder_layer_2", {row["layer"] for row in table_json["scaling_rows"]})
        self.assertIn("sequence_hidden", {row["layer"] for row in table_json["scaling_rows"]})
        self.assertGreater(table_json["encoder_layer_2_repeat_delta_mean"], 0.0)
        self.assertGreater(table_json["encoder_layer_2_repeat_ci_min_lower_delta"], 0.0)
        self.assertGreater(table_json["history_delta_ci_min_lower_delta"], 0.0)
        self.assertIn("bootstrap_engine", table_json["repeat_ci_rows"][0])
        self.assertIn("unique_value_count", table_json["history_delta_ci_rows"][0])
        self.assertIn("Label Controls", table_md)
        self.assertIn("History Controls", table_md)
        self.assertIn("Matched History Delta CIs", table_md)

    def test_build_no_range_fresh50_control_pack_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_no_range_archive(root, commit="base10k")
            _write_locked_layer_archive(root, commit="fresh123")
            _write_fresh50_control_archive(root, commit="fresh123")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_no_range_fresh50_control_pack.py",
                    "--archive-root",
                    str(root),
                    "--output-dir",
                    str(output_dir),
                    "--commit-50k",
                    "fresh123",
                    "--commit-10k",
                    "base10k",
                    "--bootstrap-count",
                    "50",
                    "--bootstrap-seed",
                    "17",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote no-range fresh50 control pack", result.stdout)

    def test_build_no_range_history_composition_pack_writes_counterfactual_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_locked_layer_archive(root, commit="fresh123")
            _write_history_composition_archive(root, commit="fresh123")

            payload = build_no_range_history_composition_pack(
                root,
                output_dir,
                commit="fresh123",
                bootstrap_count=50,
                bootstrap_seed=17,
            )
            table_json = json.loads(
                (output_dir / "limit_ar_no_range_history_composition_pack.json").read_text(encoding="utf-8")
            )
            table_md = (output_dir / "limit_ar_no_range_history_composition_pack.md").read_text(encoding="utf-8")

        self.assertEqual(len(payload["rows"]), 21)
        self.assertEqual(table_json["locked_layer"], "encoder_layer_2")
        self.assertEqual(table_json["control_count"], 7)
        self.assertGreater(table_json["min_lower_delta"], 0.0)
        self.assertGreater(table_json["count_preserving_min_lower_delta"], 0.0)
        self.assertIn("bootstrap_engine", table_json["rows"][0])
        self.assertIn("History Composition", table_md)

    def test_build_no_range_history_composition_pack_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_locked_layer_archive(root, commit="fresh123")
            _write_history_composition_archive(root, commit="fresh123")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_no_range_history_composition_pack.py",
                    "--archive-root",
                    str(root),
                    "--output-dir",
                    str(output_dir),
                    "--commit",
                    "fresh123",
                    "--bootstrap-count",
                    "50",
                    "--bootstrap-seed",
                    "17",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote no-range history composition pack", result.stdout)

    def test_build_no_range_composition_nuisance_pack_writes_control_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_locked_layer_archive(root, commit="fresh123")
            _write_composition_nuisance_archive(root, commit="fresh123")

            payload = build_no_range_composition_nuisance_pack(
                root,
                output_dir,
                commit="fresh123",
                bootstrap_count=50,
                bootstrap_seed=17,
            )
            table_json = json.loads(
                (output_dir / "limit_ar_no_range_composition_nuisance_pack.json").read_text(encoding="utf-8")
            )
            table_md = (output_dir / "limit_ar_no_range_composition_nuisance_pack.md").read_text(encoding="utf-8")

        self.assertEqual(len(payload["rows"]), 15)
        self.assertEqual(table_json["locked_layer"], "encoder_layer_2")
        self.assertEqual(table_json["comparison_count"], 5)
        self.assertGreater(table_json["composition_residual_vs_action_value_composition_min_lower_delta"], 0.0)
        self.assertTrue(table_json["composition_residual_all_lower_above_zero"])
        self.assertIn("Composition Nuisance", table_md)

    def test_build_no_range_composition_nuisance_pack_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "tables"
            _write_locked_layer_archive(root, commit="fresh123")
            _write_composition_nuisance_archive(root, commit="fresh123")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_no_range_composition_nuisance_pack.py",
                    "--archive-root",
                    str(root),
                    "--output-dir",
                    str(output_dir),
                    "--commit",
                    "fresh123",
                    "--bootstrap-count",
                    "50",
                    "--bootstrap-seed",
                    "17",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote no-range composition nuisance pack", result.stdout)

    def test_build_final_diagnostic_tables_writes_ci_backed_locked_layer_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            pack_dir = Path(temp_dir) / "packs"
            output_dir = Path(temp_dir) / "final"
            _write_locked_layer_archive(root, commit="fresh123")
            build_no_range_bootstrap_pack(
                root,
                pack_dir,
                commit="fresh123",
                bootstrap_count=50,
                bootstrap_seed=17,
            )

            payload = build_final_diagnostic_tables(
                output_dir=output_dir,
                bootstrap_pack_path=pack_dir / "limit_ar_no_range_bootstrap_pack.json",
                code_commit="test-commit",
            )
            table_json = json.loads((output_dir / "final_diagnostic_tables.json").read_text(encoding="utf-8"))
            table_md = (output_dir / "final_diagnostic_tables.md").read_text(encoding="utf-8")

        self.assertTrue(payload["all_main_rows_have_ci"])
        self.assertFalse(payload["uses_best_layer_selection"])
        self.assertEqual(payload["locked_candidate_layer"], "encoder_layer_2")
        self.assertTrue(all("ci_lower" in row and "ci_upper" in row for row in table_json["rows"]))
        self.assertEqual(
            {row["layer_role"] for row in table_json["rows"] if row["table_role"] == "main"},
            {"locked_candidate"},
        )
        self.assertIn("paired/bucket bootstrap", table_md)

    def test_build_final_diagnostic_tables_rejects_point_estimate_only_main_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pack_path = Path(temp_dir) / "bad_bootstrap.json"
            pack_path.write_text(
                json.dumps(
                    {
                        "ci_rows": [
                            {
                                "seed": "seed_k",
                                "layer": "encoder_layer_2",
                                "mean_delta": 0.1,
                                "resampling_unit": "hand_id",
                                "sample_count": 10,
                                "hand_count": 5,
                                "confidence_level": 0.95,
                            }
                        ]
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "CI"):
                build_final_diagnostic_tables(
                    output_dir=Path(temp_dir) / "final",
                    bootstrap_pack_path=pack_path,
                    code_commit="test-commit",
                )

    def test_build_final_diagnostic_tables_rejects_best_layer_main_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pack_path = Path(temp_dir) / "bad_bootstrap.json"
            pack_path.write_text(
                json.dumps(
                    {
                        "ci_rows": [
                            {
                                "seed": "seed_k",
                                "layer": "best_over_layers",
                                "mean_delta": 0.1,
                                "lower_delta": 0.01,
                                "upper_delta": 0.2,
                                "resampling_unit": "hand_id",
                                "sample_count": 10,
                                "hand_count": 5,
                                "confidence_level": 0.95,
                            }
                        ]
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "best-layer"):
                build_final_diagnostic_tables(
                    output_dir=Path(temp_dir) / "final",
                    bootstrap_pack_path=pack_path,
                    code_commit="test-commit",
                )

    def test_build_paper_table_reproduction_writes_manifest_from_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "repro"
            _write_locked_layer_archive(root, commit="fresh123")

            payload = build_paper_table_reproduction(
                archive_root=root,
                output_dir=output_dir,
                fresh50_commit="fresh123",
                bootstrap_count=50,
                bootstrap_seed=17,
                command="python scripts/reproduce_paper_tables.py --archive-root ARCHIVE",
                code_commit="test-commit",
            )
            manifest = json.loads((output_dir / "reproduction_manifest.json").read_text(encoding="utf-8"))
            final_table_exists = (output_dir / "paper_tables" / "final_diagnostic_tables.json").exists()
            manifest_md = (output_dir / "reproduction_manifest.md").read_text(encoding="utf-8")

        self.assertEqual(payload["archive_root"], str(root))
        self.assertEqual(manifest["fresh50_commit"], "fresh123")
        self.assertTrue(final_table_exists)
        self.assertIn("final_diagnostic_tables.json", manifest["expected_outputs"])
        self.assertIn("python scripts/reproduce_paper_tables.py", manifest_md)

    def test_reproduce_paper_tables_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "repro"
            _write_locked_layer_archive(root, commit="fresh123")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/reproduce_paper_tables.py",
                    "--archive-root",
                    str(root),
                    "--output-dir",
                    str(output_dir),
                    "--fresh50-commit",
                    "fresh123",
                    "--bootstrap-count",
                    "50",
                    "--bootstrap-seed",
                    "17",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote paper table reproduction manifest", result.stdout)

    def test_reproduce_paper_tables_script_can_build_optional_ci_packs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "repro"
            _write_locked_layer_archive(root, commit="fresh123")
            _write_history_composition_archive(root, commit="fresh123")
            _write_composition_nuisance_archive(root, commit="fresh123")

            subprocess.run(
                [
                    sys.executable,
                    "scripts/reproduce_paper_tables.py",
                    "--archive-root",
                    str(root),
                    "--output-dir",
                    str(output_dir),
                    "--fresh50-commit",
                    "fresh123",
                    "--composition-commit",
                    "fresh123",
                    "--history-composition-commit",
                    "fresh123",
                    "--bootstrap-count",
                    "50",
                    "--bootstrap-seed",
                    "17",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            final_tables = json.loads(
                (output_dir / "paper_tables" / "final_diagnostic_tables.json").read_text(encoding="utf-8")
            )
            manifest = json.loads((output_dir / "reproduction_manifest.json").read_text(encoding="utf-8"))

        self.assertIn("limit_ar_no_range_composition_nuisance_pack.json", manifest["expected_outputs"])
        self.assertIn("limit_ar_no_range_history_composition_pack.json", manifest["expected_outputs"])
        self.assertGreater(final_tables["main_row_count"], 6)


def _write_required_metrics(root: Path, full_commit: str) -> None:
    for seed, base_commit in {"seed_a": "ff94fab", "seed_b": "3239624", "seed_c": "3239624"}.items():
        _write_metrics(_path(root, base_commit, seed, "", legacy_layout=True), 0.14)
        _write_metrics(_path(root, base_commit, seed, "history_only", legacy_layout=True), 0.145)
        _write_metrics(_path(root, "cc431cf", seed, "history_only_action_value_only"), 0.13)
        _write_metrics(_path(root, "cc431cf", seed, "history_only_residualize_action_value"), 0.142)
        _write_metrics(_path(root, full_commit, seed, "action_value_only"), 0.131)
        _write_metrics(_path(root, full_commit, seed, "residualize_action_value"), 0.143)


def _write_robustness_summary(
    path: Path,
    policy_accuracy: float,
    range_top10: float,
    history_delta: float,
    full_delta: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "train": {
                    "policy_accuracy": policy_accuracy,
                    "opponent_hand_top10_accuracy": range_top10,
                    "model_sha256": f"model-{path.parent.name}",
                },
                "deltas": {
                    "history_residual_vs_action_value_only": history_delta,
                    "full_residual_vs_action_value_only": full_delta,
                },
                "rows": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_scale_replication_summary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seeds = []
    for seed_name in ("seed_f", "seed_g"):
        seeds.append(
            {
                "seed": seed_name,
                "train": {
                    "policy_accuracy": 0.84,
                    "opponent_hand_top10_accuracy": 0.15,
                    "model_sha256": f"model-{seed_name}",
                },
                "deltas": {
                    "history_residual_vs_action_value_only": 0.01,
                    "full_residual_vs_action_value_only": 0.01,
                },
                "rows": [],
            }
        )
    path.write_text(json.dumps({"seeds": seeds}, sort_keys=True), encoding="utf-8")


def _write_no_range_archive(root: Path, commit: str) -> None:
    for index, seed in enumerate(("seed_h", "seed_i", "seed_j")):
        train_dir = root / commit / "outputs" / f"limit_ar_action_value_only_{seed}_10k_4layer"
        train_dir.mkdir(parents=True, exist_ok=True)
        (train_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "checksum": f"train-{seed}",
                    "policy_accuracy": 0.83 + index * 0.001,
                    "value_mse": 31.0,
                    "opponent_hand_supervised": False,
                    "opponent_hand_top10_accuracy": 0.04,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        _write_layer_metrics(_no_range_path(root, commit, seed, ""), 0.13, 0.14)
        _write_layer_metrics(_no_range_path(root, commit, seed, "history_only"), 0.14, 0.145)
        _write_layer_metrics(
            _no_range_path(root, commit, seed, "history_only_action_value_only"),
            0.12,
            0.13,
            metadata=_no_range_probe_metadata(root, commit, seed, "history_only_action_value_only", 229, 241),
        )
        _write_no_range_nuisance_metric(
            _no_range_path(root, commit, seed, "history_only_action_value_only"),
            0.13,
        )
        _write_layer_metrics(
            _no_range_path(root, commit, seed, "history_only_residualize_action_value"),
            0.135,
            0.142,
            metadata=_no_range_probe_metadata(
                root,
                commit,
                seed,
                "history_only_residualize_action_value",
                230,
                242,
            ),
        )
        _write_layer_metrics(_no_range_path(root, commit, seed, "action_value_only"), 0.12, 0.13)
        _write_layer_metrics(_no_range_path(root, commit, seed, "residualize_action_value"), 0.134, 0.141)

    for suffix, label_control, score in (
        ("history_only_residualize_action_value_real_labels", "none", 0.135),
        ("history_only_residualize_action_value_shuffle_global_labels", "shuffle_global", 0.09),
        ("history_only_residualize_action_value_shuffle_within_bucket_labels", "shuffle_within_bucket", 0.10),
    ):
        _write_layer_metrics(_no_range_path(root, commit, "seed_h", suffix), score, score + 0.005, label_control=label_control)

    for suffix, input_control, feature_control, score in (
        ("drop_history", "drop_history", "none", 0.11),
        ("shuffle_history_order", "shuffle_history_order", "none", 0.12),
        ("bag_of_actions", "bag_of_actions", "none", 0.121),
        ("no_observation", "no_observation", "none", 0.10),
        ("round_position_action_prior", "none", "round_position_action_only", 0.125),
        ("history_only_action_only", "history_only", "action_only", 0.126),
    ):
        path = _no_range_path(root, commit, "seed_h", suffix)
        _write_layer_metrics(path, score, score + 0.004)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["input_control"] = input_control
        payload["feature_control"] = feature_control
        if feature_control != "none":
            payload["records"] = [{"control": "none", "layer": "nuisance", "score": score + 0.004}]
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_no_range_repeat_archive(root: Path, commit: str) -> None:
    residual_seeds = ((367, 373), (379, 383), (389, 397))
    baseline_seeds = ((467, 479), (487, 491), (499, 503))
    for seed_index, seed in enumerate(("seed_h", "seed_i", "seed_j")):
        for repeat_index in (1, 2, 3):
            offset = 0.001 * (seed_index + repeat_index)
            _write_layer_metrics(
                _no_range_path(
                    root,
                    commit,
                    seed,
                    f"history_only_residualize_action_value_repeat_{repeat_index}",
                ),
                0.135 + offset,
                0.142 + offset,
                metadata=_no_range_probe_metadata(
                    root,
                    commit,
                    seed,
                    f"history_only_residualize_action_value_repeat_{repeat_index}",
                    residual_seeds[repeat_index - 1][0],
                    residual_seeds[repeat_index - 1][1],
                ),
            )
            path = _no_range_path(
                root,
                commit,
                seed,
                f"history_only_action_value_only_repeat_{repeat_index}",
            )
            _write_layer_metrics(path, 0.12 + offset / 2.0, 0.13 + offset / 2.0)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload.update(
                _no_range_probe_metadata(
                    root,
                    commit,
                    seed,
                    f"history_only_action_value_only_repeat_{repeat_index}",
                    baseline_seeds[repeat_index - 1][0],
                    baseline_seeds[repeat_index - 1][1],
                )
            )
            payload["records"] = [
                {
                    "control": "none",
                    "layer": "nuisance",
                    "score": 0.13 + offset / 2.0,
                },
                {
                    "control": "random_label",
                    "layer": "nuisance",
                    "score": 0.08,
                },
            ]
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_no_range_nuisance_metric(path: Path, score: float) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["feature_control"] = "action_value_only"
    payload["input_control"] = "history_only"
    payload["records"] = [
        {
            "control": "none",
            "layer": "nuisance",
            "score": score,
        },
        {
            "control": "random_label",
            "layer": "nuisance",
            "score": 0.08,
        },
    ]
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_locked_layer_archive(root: Path, commit: str) -> None:
    for index, seed in enumerate(("seed_k", "seed_l", "seed_m")):
        train_dir = root / commit / "outputs" / f"limit_ar_action_value_only_{seed}_50k_4layer"
        train_dir.mkdir(parents=True, exist_ok=True)
        (train_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "checksum": f"train-{seed}",
                    "policy_accuracy": 0.84 + index * 0.001,
                    "value_mse": 30.0,
                    "opponent_hand_loss_weight": 0.0,
                    "opponent_hand_supervised": False,
                    "opponent_hand_top10_accuracy": 0.05,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        baseline_path = _no_range_path_for_size(root, commit, seed, "history_only_action_value_only", "50k")
        residual_path = _no_range_path_for_size(root, commit, seed, "history_only_residualize_action_value", "50k")
        _write_layer_metrics(
            baseline_path,
            0.12,
            0.13,
            metadata=_no_range_probe_metadata_for_size(
                root,
                commit,
                seed,
                "history_only_action_value_only",
                601 + index * 20,
                607 + index * 20,
                "50k",
            ),
        )
        _write_no_range_nuisance_metric(baseline_path, 0.13)
        _write_correctness_artifact(baseline_path, {"nuisance": 0}, correct_indices_by_layer={"nuisance": set()})
        _write_layer_metrics(
            residual_path,
            0.136 + index * 0.001,
            0.142 + index * 0.001,
            metadata=_no_range_probe_metadata_for_size(
                root,
                commit,
                seed,
                "history_only_residualize_action_value",
                613 + index * 20,
                617 + index * 20,
                "50k",
            ),
        )
        _write_correctness_artifact(
            residual_path,
            {
                "sequence_hidden": 20,
                "encoder_layer_2": 20,
            },
            correct_indices_by_layer={
                "sequence_hidden": set(range(20)),
                "encoder_layer_2": set(range(20)),
            },
        )


def _write_fresh50_control_archive(root: Path, commit: str) -> None:
    for seed_index, seed in enumerate(("seed_k", "seed_l", "seed_m")):
        for repeat_index in (1, 2, 3):
            offset = 0.001 * (seed_index + repeat_index)
            baseline_path = _no_range_path_for_size(
                root,
                commit,
                seed,
                f"history_only_action_value_only_repeat_{repeat_index}",
                "50k",
            )
            _write_layer_metrics(
                baseline_path,
                0.12 + offset,
                0.13 + offset,
                metadata=_no_range_probe_metadata_for_size(
                    root,
                    commit,
                    seed,
                    f"history_only_action_value_only_repeat_{repeat_index}",
                    701 + seed_index * 100 + repeat_index,
                    711 + seed_index * 100 + repeat_index,
                    "50k",
                ),
            )
            _write_no_range_nuisance_metric(baseline_path, 0.13 + offset)
            _write_correctness_artifact(
                baseline_path,
                {"nuisance": 10},
                correct_indices_by_layer={"nuisance": set(range(0, 20, 2))},
            )
            residual_path = _no_range_path_for_size(
                root,
                commit,
                seed,
                f"history_only_residualize_action_value_repeat_{repeat_index}",
                "50k",
            )
            _write_layer_metrics(
                residual_path,
                0.136 + offset,
                0.142 + offset,
                metadata=_no_range_probe_metadata_for_size(
                    root,
                    commit,
                    seed,
                    f"history_only_residualize_action_value_repeat_{repeat_index}",
                    703 + seed_index * 100 + repeat_index,
                    713 + seed_index * 100 + repeat_index,
                    "50k",
                ),
            )
            _write_correctness_artifact(
                residual_path,
                {
                    "sequence_hidden": 15,
                    "encoder_layer_2": 20,
                },
                correct_indices_by_layer={
                    "sequence_hidden": {*range(0, 20, 2), *range(1, 10, 2)},
                    "encoder_layer_2": set(range(20)),
                },
            )

        for label_index, (suffix, label_control, score) in enumerate(
            (
                ("history_only_residualize_action_value_shuffle_global_labels", "shuffle_global", 0.09),
                ("history_only_residualize_action_value_shuffle_within_bucket_labels", "shuffle_within_bucket", 0.10),
            ),
            start=1,
        ):
            _write_layer_metrics(
                _no_range_path_for_size(root, commit, seed, suffix, "50k"),
                score + seed_index * 0.001,
                score + 0.002 + seed_index * 0.001,
                label_control=label_control,
                metadata=_no_range_probe_metadata_for_size(
                    root,
                    commit,
                    seed,
                    suffix,
                    803 + seed_index * 100 + label_index,
                    813 + seed_index * 100 + label_index,
                    "50k",
                ),
            )

        history_specs = (
            ("drop_history", "drop_history", "none", 0.11),
            ("shuffle_history_order", "shuffle_history_order", "none", 0.12),
            ("bag_of_actions", "bag_of_actions", "none", 0.121),
            ("no_observation", "no_observation", "none", 0.10),
            ("round_position_action_prior", "none", "round_position_action_only", 0.125),
            ("history_only_action_only", "history_only", "action_only", 0.126),
            ("drop_history_residualize_action_value", "drop_history", "residualize_action_value", 0.112),
            (
                "shuffle_history_order_residualize_action_value",
                "shuffle_history_order",
                "residualize_action_value",
                0.118,
            ),
            ("bag_of_actions_residualize_action_value", "bag_of_actions", "residualize_action_value", 0.119),
            ("no_observation_residualize_action_value", "no_observation", "residualize_action_value", 0.101),
        )
        for control_index, (suffix, input_control, feature_control, score) in enumerate(history_specs, start=1):
            path = _no_range_path_for_size(root, commit, seed, suffix, "50k")
            layers = ["nuisance"] if feature_control in {"action_only", "round_position_action_only"} else None
            _write_layer_metrics(
                path,
                score + seed_index * 0.001,
                score + 0.004 + seed_index * 0.001,
                metadata=_no_range_probe_metadata_for_size(
                    root,
                    commit,
                    seed,
                    suffix,
                    903 + seed_index * 100 + control_index,
                    913 + seed_index * 100 + control_index,
                    "50k",
                ),
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["input_control"] = input_control
            payload["feature_control"] = feature_control
            if layers == ["nuisance"]:
                payload["layers"] = ["nuisance"]
                payload["records"] = [
                    {"control": "none", "layer": "nuisance", "score": score + 0.004 + seed_index * 0.001}
                ]
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            if suffix in {
                "drop_history_residualize_action_value",
                "shuffle_history_order_residualize_action_value",
                "bag_of_actions_residualize_action_value",
                "no_observation_residualize_action_value",
            }:
                _write_correctness_artifact(
                    path,
                    {"encoder_layer_2": 0},
                    correct_indices_by_layer={"encoder_layer_2": set()},
                )

        for control_index, (suffix, input_control, score) in enumerate(
            (
                ("drop_history_action_value_only", "drop_history", 0.101),
                ("shuffle_history_order_action_value_only", "shuffle_history_order", 0.102),
                ("bag_of_actions_action_value_only", "bag_of_actions", 0.103),
                ("no_observation_action_value_only", "no_observation", 0.100),
            ),
            start=1,
        ):
            path = _no_range_path_for_size(root, commit, seed, suffix, "50k")
            _write_layer_metrics(
                path,
                score + seed_index * 0.001,
                score + seed_index * 0.001,
                metadata=_no_range_probe_metadata_for_size(
                    root,
                    commit,
                    seed,
                    suffix,
                    1101 + seed_index * 100 + control_index * 2,
                    1102 + seed_index * 100 + control_index * 2,
                    "50k",
                ),
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["version"] = "1.0.13"
            payload["input_control"] = input_control
            payload["feature_control"] = "action_value_only"
            payload["layers"] = ["nuisance"]
            payload["records"] = [
                {"control": "none", "layer": "nuisance", "score": score + seed_index * 0.001},
                {"control": "random_label", "layer": "nuisance", "score": 0.08},
            ]
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            _write_correctness_artifact(path, {"nuisance": 5}, correct_indices_by_layer={"nuisance": set()})


def _write_history_composition_archive(root: Path, commit: str) -> None:
    controls = (
        "shuffle_history_order_seed_1",
        "shuffle_history_order_seed_2",
        "shuffle_history_order_seed_3",
        "reverse_history_order",
        "sort_history_actions",
        "history_presence_only",
        "history_last_action_only",
    )
    for seed_index, seed in enumerate(("seed_k", "seed_l", "seed_m")):
        for control_index, control in enumerate(controls, start=1):
            residual_suffix = f"{control}_residualize_action_value"
            baseline_suffix = f"{control}_action_value_only"
            residual_path = _no_range_path_for_size(root, commit, seed, residual_suffix, "50k")
            _write_layer_metrics(
                residual_path,
                0.12 + seed_index * 0.001,
                0.13 + seed_index * 0.001,
                metadata=_no_range_probe_metadata_for_size(
                    root,
                    commit,
                    seed,
                    residual_suffix,
                    1301 + seed_index * 100 + control_index * 2,
                    1302 + seed_index * 100 + control_index * 2,
                    "50k",
                ),
            )
            payload = json.loads(residual_path.read_text(encoding="utf-8"))
            payload["version"] = "1.0.15"
            payload["input_control"] = control
            payload["feature_control"] = "residualize_action_value"
            payload["layers"] = ["encoder_layer_2"]
            payload["records"] = [
                {"control": "none", "layer": "encoder_layer_2", "score": 0.13 + seed_index * 0.001},
                {"control": "random_label", "layer": "encoder_layer_2", "score": 0.08},
            ]
            residual_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            _write_correctness_artifact(
                residual_path,
                {"encoder_layer_2": 10},
                correct_indices_by_layer={"encoder_layer_2": set(range(10))},
            )

            baseline_path = _no_range_path_for_size(root, commit, seed, baseline_suffix, "50k")
            _write_layer_metrics(
                baseline_path,
                0.12 + seed_index * 0.001,
                0.10 + seed_index * 0.001,
                metadata=_no_range_probe_metadata_for_size(
                    root,
                    commit,
                    seed,
                    baseline_suffix,
                    1303 + seed_index * 100 + control_index * 2,
                    1304 + seed_index * 100 + control_index * 2,
                    "50k",
                ),
            )
            payload = json.loads(baseline_path.read_text(encoding="utf-8"))
            payload["version"] = "1.0.15"
            payload["input_control"] = control
            payload["feature_control"] = "action_value_only"
            payload["layers"] = ["nuisance"]
            payload["records"] = [
                {"control": "none", "layer": "nuisance", "score": 0.10 + seed_index * 0.001},
                {"control": "random_label", "layer": "nuisance", "score": 0.08},
            ]
            baseline_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            _write_correctness_artifact(
                baseline_path,
                {"nuisance": 8},
                correct_indices_by_layer={"nuisance": set(range(8))},
            )


def _write_composition_nuisance_archive(root: Path, commit: str) -> None:
    for seed_index, seed in enumerate(("seed_k", "seed_l", "seed_m")):
        for suffix, feature_control, layer, correct_indices in (
            ("history_composition_only", "history_composition_only", "nuisance", set(range(9))),
            (
                "action_value_history_composition_only",
                "action_value_history_composition_only",
                "nuisance",
                set(range(6)),
            ),
            (
                "residualize_action_value_history_composition",
                "residualize_action_value_history_composition",
                "encoder_layer_2",
                set(range(14 + seed_index)),
            ),
        ):
            path = _no_range_path_for_size(root, commit, seed, suffix, "50k")
            _write_layer_metrics(
                path,
                0.12 + seed_index * 0.001,
                0.10 + seed_index * 0.001,
                metadata=_no_range_probe_metadata_for_size(
                    root,
                    commit,
                    seed,
                    suffix,
                    1601 + seed_index * 100 + len(suffix),
                    1602 + seed_index * 100 + len(suffix),
                    "50k",
                ),
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["version"] = "1.0.16"
            payload["input_control"] = "history_only"
            payload["feature_control"] = feature_control
            payload["layers"] = [layer]
            payload["records"] = [
                {"control": "none", "layer": layer, "score": 0.10 + seed_index * 0.001},
                {"control": "random_label", "layer": layer, "score": 0.08},
            ]
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            _write_correctness_artifact(
                path,
                {layer: len(correct_indices)},
                correct_indices_by_layer={layer: correct_indices},
            )


def _write_range_supervised_positive_control_archive(root: Path, commit: str) -> None:
    for seed_index, seed in enumerate(("seed_d", "seed_f", "seed_g")):
        train_dir = root / commit / "outputs" / f"limit_ar_teacher_{seed}_50k_4layer"
        train_dir.mkdir(parents=True, exist_ok=True)
        (train_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "checksum": f"range-train-{seed}",
                    "policy_accuracy": 0.86 + seed_index * 0.001,
                    "value_mse": 28.0,
                    "opponent_hand_loss_weight": 1.0,
                    "opponent_hand_supervised": True,
                    "opponent_hand_top10_accuracy": 0.18,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        specs = (
            (
                "history_only_action_value_only",
                "action_value_only",
                "nuisance",
                set(),
            ),
            (
                "history_only_residualize_action_value",
                "residualize_action_value",
                "encoder_layer_2",
                set(range(20)),
            ),
            (
                "action_value_history_composition_only",
                "action_value_history_composition_only",
                "nuisance",
                set(),
            ),
            (
                "residualize_action_value_history_composition",
                "residualize_action_value_history_composition",
                "encoder_layer_2",
                set(range(20)),
            ),
        )
        for suffix, feature_control, layer, correct_indices in specs:
            path = _range_supervised_path_for_size(root, commit, seed, suffix, "50k")
            _write_layer_metrics(
                path,
                0.12 + seed_index * 0.001,
                0.10 + seed_index * 0.001,
                metadata=_range_supervised_probe_metadata_for_size(
                    root,
                    commit,
                    seed,
                    suffix,
                    1701 + seed_index * 100 + len(suffix),
                    1702 + seed_index * 100 + len(suffix),
                    "50k",
                ),
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["version"] = "1.17"
            payload["input_control"] = "history_only"
            payload["feature_control"] = feature_control
            payload["layers"] = [layer]
            payload["records"] = [
                {"control": "none", "layer": layer, "score": 0.10 + seed_index * 0.001},
                {"control": "random_label", "layer": layer, "score": 0.08},
            ]
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            _write_correctness_artifact(
                path,
                {layer: len(correct_indices)},
                correct_indices_by_layer={layer: correct_indices},
            )
        bucket_path = (
            root
            / commit
            / "outputs"
            / "tables"
            / f"range_supervised_{seed}_matched_composition_buckets.json"
        )
        bucket_path.parent.mkdir(parents=True, exist_ok=True)
        bucket_path.write_text(
            json.dumps(
                {
                    "buckets": {
                        f"hand_{example_index // 2}:{example_index % 2}": f"bucket_{example_index // 4}"
                        for example_index in range(20)
                    }
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def _write_correctness_artifact(
    path: Path,
    layer_correct_counts: dict[str, int],
    sample_count: int = 20,
    *,
    correct_indices_by_layer: dict[str, set[int]] | None = None,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["correctness_artifact"] = "probe_correctness.jsonl"

    rows = []
    for layer, correct_count in layer_correct_counts.items():
        score = correct_count / sample_count
        correct_indices = set(range(correct_count))
        if correct_indices_by_layer and layer in correct_indices_by_layer:
            correct_indices = set(correct_indices_by_layer[layer])
            score = len(correct_indices) / sample_count
        for record in payload["records"]:
            if record["layer"] == layer and record["control"] == "none":
                record["score"] = score
                record["top10_accuracy"] = score
        for example_index in range(sample_count):
            is_correct = example_index in correct_indices
            hand_id = f"hand_{example_index // 2}"
            step_index = example_index % 2
            rows.append(
                {
                    "control": "none",
                    "example_index": example_index,
                    "feature_control": payload["feature_control"],
                    "hand_id": hand_id,
                    "label_control": payload["label_control"],
                    "layer": layer,
                    "point_id": f"{hand_id}:{step_index}",
                    "split": "test",
                    "step_index": step_index,
                    "target": payload["target"],
                    "target_label": example_index % 169,
                    "top1_correct": is_correct,
                    "top5_correct": is_correct,
                    "top10_correct": is_correct,
                }
            )
        for example_index in range(sample_count):
            hand_id = f"hand_{example_index // 2}"
            step_index = example_index % 2
            rows.append(
                {
                    "control": "random_label",
                    "example_index": example_index,
                    "feature_control": payload["feature_control"],
                    "hand_id": hand_id,
                    "label_control": payload["label_control"],
                    "layer": layer,
                    "point_id": f"{hand_id}:{step_index}",
                    "split": "test",
                    "step_index": step_index,
                    "target": payload["target"],
                    "target_label": example_index % 169,
                    "top1_correct": False,
                    "top5_correct": False,
                    "top10_correct": False,
                }
            )
    (path.parent / "probe_correctness.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    payload["correctness_sha256"] = _file_sha256(path.parent / "probe_correctness.jsonl")
    rows_by_layer_control = {}
    for row in rows:
        key = f"{row['layer']}|{row['control']}"
        rows_by_layer_control[key] = rows_by_layer_control.get(key, 0) + 1
    payload["correctness_summary"] = {
        "row_count": len(rows),
        "rows_by_layer_control": rows_by_layer_control,
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _no_range_probe_metadata(
    root: Path,
    commit: str,
    seed: str,
    suffix: str,
    probe_seed: int,
    control_seed: int,
) -> dict[str, object]:
    probe_name = f"limit_ar_action_value_only_{seed}_4layer_range_{suffix}_10k"
    feature_control = "action_value_only" if "action_value_only" in suffix else "residualize_action_value"
    layers = ["nuisance"] if feature_control == "action_value_only" else ["sequence_hidden", "encoder_layer_2"]
    artifact_dir = f"outputs/probes/{probe_name}"
    return {
        "version": "1.0.7",
        "config_path": str(root / commit / "configs" / "probe" / f"{probe_name}.json"),
        "train_config": f"configs/train/limit_ar_action_value_only_{seed}_10k_4layer.json",
        "checkpoint_path": f"outputs/limit_ar_action_value_only_{seed}_10k_4layer/model.pt",
        "artifact_dir": artifact_dir,
        "configured_artifact_dir": artifact_dir,
        "target": "opponent_hand",
        "input_control": "history_only",
        "feature_control": feature_control,
        "label_control": "none",
        "probe_seed": probe_seed,
        "control_seed": control_seed,
        "layers": layers,
    }


def _no_range_path_for_size(root: Path, commit: str, seed: str, suffix: str, size: str) -> Path:
    suffix_text = f"_{suffix}" if suffix else ""
    return (
        root
        / commit
        / "outputs"
        / "probes"
        / f"limit_ar_action_value_only_{seed}_4layer_range{suffix_text}_{size}"
        / "probe_metrics.json"
    )


def _range_supervised_path_for_size(root: Path, commit: str, seed: str, suffix: str, size: str) -> Path:
    suffix_text = f"_{suffix}" if suffix else ""
    return (
        root
        / commit
        / "outputs"
        / "probes"
        / f"limit_ar_teacher_{seed}_4layer_range{suffix_text}_{size}"
        / "probe_metrics.json"
    )


def _range_supervised_probe_metadata_for_size(
    root: Path,
    commit: str,
    seed: str,
    suffix: str,
    probe_seed: int,
    control_seed: int,
    size: str,
) -> dict[str, object]:
    probe_name = f"limit_ar_teacher_{seed}_4layer_range_{suffix}_{size}"
    artifact_dir = f"outputs/probes/{probe_name}"
    return {
        "version": "1.17",
        "config_path": str(root / commit / "configs" / "probe" / f"{probe_name}.json"),
        "train_config": f"configs/train/limit_ar_teacher_{seed}_{size}_4layer.json",
        "checkpoint_path": f"outputs/limit_ar_teacher_{seed}_{size}_4layer/model.pt",
        "artifact_dir": artifact_dir,
        "configured_artifact_dir": artifact_dir,
        "target": "opponent_hand",
        "input_control": "history_only",
        "feature_control": "residualize_action_value",
        "label_control": "none",
        "probe_seed": probe_seed,
        "control_seed": control_seed,
        "layers": ["encoder_layer_2"],
    }


def _no_range_probe_metadata_for_size(
    root: Path,
    commit: str,
    seed: str,
    suffix: str,
    probe_seed: int,
    control_seed: int,
    size: str,
) -> dict[str, object]:
    metadata = _no_range_probe_metadata(root, commit, seed, suffix, probe_seed, control_seed)
    probe_name = f"limit_ar_action_value_only_{seed}_4layer_range_{suffix}_{size}"
    metadata["config_path"] = str(root / commit / "configs" / "probe" / f"{probe_name}.json")
    metadata["train_config"] = f"configs/train/limit_ar_action_value_only_{seed}_{size}_4layer.json"
    metadata["checkpoint_path"] = f"outputs/limit_ar_action_value_only_{seed}_{size}_4layer/model.pt"
    metadata["artifact_dir"] = f"outputs/probes/{probe_name}"
    metadata["configured_artifact_dir"] = f"outputs/probes/{probe_name}"
    metadata["version"] = "1.0.9"
    return metadata


if __name__ == "__main__":
    unittest.main()
