from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jorjor.probe.capacity import build_probe_capacity_s6_s7_report, build_probe_capacity_sensitivity_report


class ProbeCapacitySensitivityTest(unittest.TestCase):
    """Test v1.14 linear-main / MLP-appendix sensitivity assembly."""

    def test_build_probe_capacity_sensitivity_report_writes_appendix_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            linear_path = root / "linear" / "probe_metrics.json"
            mlp_path = root / "mlp" / "probe_metrics.json"
            output_dir = root / "report"
            _write_probe_metrics(linear_path, "linear", 0.14)
            _write_probe_metrics(mlp_path, "mlp", 0.145)

            payload = build_probe_capacity_sensitivity_report(
                linear_metrics_path=linear_path,
                mlp_metrics_path=mlp_path,
                output_dir=output_dir,
                locked_layer="encoder_layer_2",
                tolerance=0.02,
            )
            table_json = json.loads((output_dir / "probe_capacity_sensitivity_report.json").read_text(encoding="utf-8"))
            table_md = (output_dir / "probe_capacity_sensitivity_report.md").read_text(encoding="utf-8")

        self.assertEqual(payload["main_probe_family"], "linear")
        self.assertEqual(table_json["appendix_probe_family"], "mlp")
        self.assertTrue(table_json["verdict_agreement"])
        self.assertEqual(table_json["rows"][0]["table_role"], "main")
        self.assertEqual(table_json["rows"][1]["table_role"], "appendix")
        self.assertIn("MLP rows are appendix-only", table_md)

    def test_build_probe_capacity_s6_s7_report_writes_appendix_ci_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "report"
            _write_capacity_archive(root, mlp_commit="mlp123", linear_commit="lin123")

            payload = build_probe_capacity_s6_s7_report(
                archive_root=root,
                output_dir=output_dir,
                mlp_commit="mlp123",
                linear_commit="lin123",
                bootstrap_count=50,
                bootstrap_seed=23,
            )
            table_json = json.loads((output_dir / "probe_capacity_s6_s7_report.json").read_text(encoding="utf-8"))
            table_md = (output_dir / "probe_capacity_s6_s7_report.md").read_text(encoding="utf-8")

        self.assertEqual(payload["row_count"], 9)
        self.assertEqual(table_json["appendix_probe_family"], "mlp")
        self.assertEqual(table_json["final_verdict"], "mlp_agrees_shortcut")
        self.assertLess(table_json["c3_max_upper_delta"], 0.0)
        self.assertTrue(all(row["table_role"] == "appendix" for row in table_json["rows"]))
        self.assertIn("MLP rows are appendix-only", table_md)

    def test_build_probe_capacity_s6_s7_report_requires_mlp_correctness(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "report"
            _write_capacity_archive(root, mlp_commit="mlp123", linear_commit="lin123")
            stale_path = _no_range_path_for_size(
                root,
                "mlp123",
                "seed_k",
                "history_only_residualize_action_value_mlp",
                "50k",
            )
            (stale_path.parent / "probe_correctness.jsonl").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "correctness"):
                build_probe_capacity_s6_s7_report(
                    archive_root=root,
                    output_dir=output_dir,
                    mlp_commit="mlp123",
                    linear_commit="lin123",
                    bootstrap_count=10,
                    bootstrap_seed=23,
                )

    def test_build_probe_capacity_s6_s7_report_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "archive"
            output_dir = Path(temp_dir) / "report"
            _write_capacity_archive(root, mlp_commit="mlp123", linear_commit="lin123")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_probe_capacity_s6_s7_report.py",
                    "--archive-root",
                    str(root),
                    "--output-dir",
                    str(output_dir),
                    "--mlp-commit",
                    "mlp123",
                    "--linear-commit",
                    "lin123",
                    "--bootstrap-count",
                    "20",
                    "--bootstrap-seed",
                    "23",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote probe capacity S6/S7 report rows=9", result.stdout)


def _write_probe_metrics(path: Path, probe_family: str, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "checksum": f"{probe_family}-checksum",
                "probe_family": probe_family,
                "probe_capacity_role": "main_linear" if probe_family == "linear" else "appendix_sensitivity",
                "mlp_hidden_size": None if probe_family == "linear" else 16,
                "target": "opponent_hand",
                "feature_control": "residualize_action_value",
                "label_control": "none",
                "records": [
                    {
                        "control": "none",
                        "layer": "encoder_layer_2",
                        "probe_family": probe_family,
                        "score": score,
                    },
                    {
                        "control": "random_label",
                        "layer": "encoder_layer_2",
                        "probe_family": probe_family,
                        "score": 0.08,
                    },
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_capacity_archive(root: Path, *, mlp_commit: str, linear_commit: str) -> None:
    for seed_index, seed in enumerate(("seed_k", "seed_l", "seed_m")):
        for commit in (mlp_commit, linear_commit):
            train_dir = root / commit / "outputs" / f"limit_ar_action_value_only_{seed}_50k_4layer"
            train_dir.mkdir(parents=True, exist_ok=True)
            (train_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "checksum": f"train-{commit}-{seed}",
                        "opponent_hand_loss_weight": 0.0,
                        "policy_accuracy": 0.44,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        specs = (
            (linear_commit, "history_only_action_value_only", "action_value_only", "nuisance", "linear", set(range(12))),
            (
                linear_commit,
                "action_value_history_composition_only",
                "action_value_history_composition_only",
                "nuisance",
                "linear",
                set(range(20)),
            ),
            (
                mlp_commit,
                "history_only_residualize_action_value_mlp",
                "residualize_action_value",
                "encoder_layer_2",
                "mlp",
                set(range(10)),
            ),
            (
                mlp_commit,
                "residualize_action_value_history_composition_mlp",
                "residualize_action_value_history_composition",
                "encoder_layer_2",
                "mlp",
                set(),
            ),
        )
        for commit, suffix, feature_control, layer, probe_family, correct_indices in specs:
            path = _no_range_path_for_size(root, commit, seed, suffix, "50k")
            _write_capacity_metrics(
                path,
                commit,
                seed,
                suffix,
                feature_control,
                layer,
                probe_family,
                2301 + seed_index * 100 + len(suffix),
                2302 + seed_index * 100 + len(suffix),
                correct_indices,
            )
        bucket_path = (
            root
            / mlp_commit
            / "outputs"
            / "tables"
            / f"no_range_capacity_{seed}_matched_composition_buckets.json"
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


def _write_capacity_metrics(
    path: Path,
    commit: str,
    seed: str,
    suffix: str,
    feature_control: str,
    layer: str,
    probe_family: str,
    probe_seed: int,
    control_seed: int,
    correct_indices: set[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = f"outputs/probes/limit_ar_action_value_only_{seed}_4layer_range_{suffix}_50k"
    rows = []
    for control in ("none", "random_label"):
        for example_index in range(20):
            is_correct = control == "none" and example_index in correct_indices
            hand_id = f"hand_{example_index // 2}"
            step_index = example_index % 2
            rows.append(
                {
                    "control": control,
                    "example_index": example_index,
                    "feature_control": feature_control,
                    "hand_id": hand_id,
                    "label_control": "none",
                    "layer": layer,
                    "point_id": f"{hand_id}:{step_index}",
                    "split": "test",
                    "step_index": step_index,
                    "target": "opponent_hand",
                    "target_label": example_index % 169,
                    "top1_correct": is_correct,
                    "top5_correct": is_correct,
                    "top10_correct": is_correct,
                }
            )
    correctness_path = path.parent / "probe_correctness.jsonl"
    correctness_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    score = len(correct_indices) / 20
    payload = {
        "artifact_dir": str(path.parent),
        "checkpoint_path": f"outputs/limit_ar_action_value_only_{seed}_50k_4layer/model.pt",
        "checksum": f"{commit}-{seed}-{suffix}",
        "configured_artifact_dir": artifact_dir,
        "control_seed": control_seed,
        "correctness_artifact": "probe_correctness.jsonl",
        "correctness_sha256": _file_sha256(correctness_path),
        "feature_control": feature_control,
        "input_control": "history_only",
        "label_control": "none",
        "layers": [layer],
        "mlp_hidden_size": 64 if probe_family == "mlp" else None,
        "probe_capacity_role": "appendix_sensitivity" if probe_family == "mlp" else "main_linear",
        "probe_family": probe_family,
        "probe_seed": probe_seed,
        "records": [
            {
                "control": "none",
                "feature_control": feature_control,
                "label_control": "none",
                "layer": layer,
                "probe_capacity_role": "appendix_sensitivity" if probe_family == "mlp" else "main_linear",
                "probe_family": probe_family,
                "score": score,
                "target": "opponent_hand",
                "top10_accuracy": score,
            },
            {
                "control": "random_label",
                "feature_control": feature_control,
                "label_control": "none",
                "layer": layer,
                "probe_capacity_role": "appendix_sensitivity" if probe_family == "mlp" else "main_linear",
                "probe_family": probe_family,
                "score": 0.0,
                "target": "opponent_hand",
                "top10_accuracy": 0.0,
            },
        ],
        "target": "opponent_hand",
        "train_config": f"configs/train/limit_ar_action_value_only_{seed}_50k_4layer.json",
        "version": "1.18" if probe_family == "mlp" else "1.0.16",
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    unittest.main()
