from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from jorjor.data.schema import DecisionPoint
from jorjor.config import load_limit_ar_probe_config
from jorjor.models.ar import build_tiny_ar_model, require_torch
from jorjor.probe.limit_ar import _history_composition_features, run_limit_ar_probe
from jorjor.tokenizer import LimitTokenizer
from jorjor.train.limit_ar import run_limit_ar_train


class LimitArProbeTest(unittest.TestCase):
    """Test v0.7.0 Limit AR layer probe plumbing."""

    def test_load_limit_ar_probe_config(self) -> None:
        config = load_limit_ar_probe_config(Path("configs/probe/limit_ar_teacher_range_smoke.json"))

        self.assertEqual(config.run_version, "0.7.0-smoke")
        self.assertEqual(config.target, "opponent_hand")
        self.assertEqual(config.input_control, "none")
        self.assertEqual(config.label_control, "none")
        self.assertEqual(config.layers, ("embedding_final", "encoder_layer_0", "sequence_hidden"))
        self.assertEqual(config.train_config, Path("configs/train/limit_ar_teacher_smoke_saved.json"))
        self.assertTrue(config.random_label_control)

    def test_load_limit_ar_probe_control_configs(self) -> None:
        configs = {
            "configs/probe/limit_ar_teacher_seed_a_range_drop_history_10k.json": "drop_history",
            "configs/probe/limit_ar_teacher_seed_a_range_mask_private_10k.json": "mask_private_card",
            "configs/probe/limit_ar_teacher_seed_a_range_combined_mask_10k.json": "drop_history_mask_cards",
        }

        for path, input_control in configs.items():
            with self.subTest(path=path):
                config = load_limit_ar_probe_config(Path(path))
                self.assertEqual(config.run_version, "0.7.1")
                self.assertEqual(config.target, "opponent_hand")
                self.assertEqual(config.input_control, input_control)
                self.assertEqual(config.checkpoint_path, Path("outputs/limit_ar_teacher_seed_a_10k/model.pt"))

    def test_load_limit_ar_probe_strict_control_configs(self) -> None:
        configs = {
            "configs/probe/limit_ar_teacher_seed_a_range_mask_public_10k.json": "mask_public_card",
            "configs/probe/limit_ar_teacher_seed_a_range_mask_legal_10k.json": "mask_legal_actions",
            "configs/probe/limit_ar_teacher_seed_a_range_mask_round_position_10k.json": "mask_round_position",
            "configs/probe/limit_ar_teacher_seed_a_range_mask_all_cards_10k.json": "mask_all_cards",
        }

        for path, input_control in configs.items():
            with self.subTest(path=path):
                config = load_limit_ar_probe_config(Path(path))
                self.assertEqual(config.run_version, "0.7.2")
                self.assertEqual(config.target, "opponent_hand")
                self.assertEqual(config.input_control, input_control)
                self.assertEqual(config.checkpoint_path, Path("outputs/limit_ar_teacher_seed_a_10k/model.pt"))

    def test_load_limit_ar_probe_compound_prior_configs(self) -> None:
        configs = {
            "configs/probe/limit_ar_teacher_seed_a_range_history_only_10k.json": "history_only",
            "configs/probe/limit_ar_teacher_seed_a_range_legal_only_10k.json": "legal_only",
            "configs/probe/limit_ar_teacher_seed_a_range_round_position_only_10k.json": "round_position_only",
            "configs/probe/limit_ar_teacher_seed_a_range_no_observation_10k.json": "no_observation",
        }

        for path, input_control in configs.items():
            with self.subTest(path=path):
                config = load_limit_ar_probe_config(Path(path))
                self.assertEqual(config.run_version, "0.7.3")
                self.assertEqual(config.target, "opponent_hand")
                self.assertEqual(config.input_control, input_control)
                self.assertEqual(config.checkpoint_path, Path("outputs/limit_ar_teacher_seed_a_10k/model.pt"))

    def test_load_limit_ar_4layer_probe_configs(self) -> None:
        configs = {
            "configs/probe/limit_ar_teacher_seed_a_4layer_range_10k.json": "none",
            "configs/probe/limit_ar_teacher_seed_a_4layer_range_history_only_10k.json": "history_only",
            "configs/probe/limit_ar_teacher_seed_a_4layer_range_no_observation_10k.json": "no_observation",
            "configs/probe/limit_ar_teacher_seed_a_4layer_range_legal_only_10k.json": "legal_only",
            "configs/probe/limit_ar_teacher_seed_a_4layer_range_round_position_only_10k.json": "round_position_only",
        }

        for path, input_control in configs.items():
            with self.subTest(path=path):
                config = load_limit_ar_probe_config(Path(path))
                self.assertEqual(config.run_version, "0.8.0")
                self.assertEqual(config.target, "opponent_hand")
                self.assertEqual(config.input_control, input_control)
                self.assertEqual(config.train_config, Path("configs/train/limit_ar_teacher_seed_a_10k_4layer.json"))
                self.assertEqual(
                    config.checkpoint_path,
                    Path("outputs/limit_ar_teacher_seed_a_10k_4layer/model.pt"),
                )
                self.assertEqual(
                    config.layers,
                    (
                        "embedding_final",
                        "encoder_layer_0",
                        "encoder_layer_1",
                        "encoder_layer_2",
                        "encoder_layer_3",
                        "sequence_hidden",
                    ),
                )

    def test_load_limit_ar_4layer_seed_replication_probe_configs(self) -> None:
        configs = {
            "configs/probe/limit_ar_teacher_seed_b_4layer_range_10k.json": ("none", "seed_b"),
            "configs/probe/limit_ar_teacher_seed_b_4layer_range_history_only_10k.json": ("history_only", "seed_b"),
            "configs/probe/limit_ar_teacher_seed_b_4layer_range_no_observation_10k.json": (
                "no_observation",
                "seed_b",
            ),
            "configs/probe/limit_ar_teacher_seed_c_4layer_range_10k.json": ("none", "seed_c"),
            "configs/probe/limit_ar_teacher_seed_c_4layer_range_history_only_10k.json": ("history_only", "seed_c"),
            "configs/probe/limit_ar_teacher_seed_c_4layer_range_no_observation_10k.json": (
                "no_observation",
                "seed_c",
            ),
        }

        for path, (input_control, seed_name) in configs.items():
            with self.subTest(path=path):
                config = load_limit_ar_probe_config(Path(path))
                self.assertEqual(config.run_version, "0.8.1")
                self.assertEqual(config.target, "opponent_hand")
                self.assertEqual(config.input_control, input_control)
                self.assertEqual(
                    config.train_config,
                    Path(f"configs/train/limit_ar_teacher_{seed_name}_10k_4layer.json"),
                )
                self.assertEqual(
                    config.checkpoint_path,
                    Path(f"outputs/limit_ar_teacher_{seed_name}_10k_4layer/model.pt"),
                )
                self.assertEqual(len(config.layers), 6)

    def test_load_limit_ar_4layer_action_value_probe_configs(self) -> None:
        controls = {
            "": "none",
            "_history_only": "history_only",
            "_no_observation": "no_observation",
        }
        targets = {"action", "value"}
        seeds = {"seed_a", "seed_b", "seed_c"}

        for seed_name in seeds:
            for target in targets:
                for suffix, input_control in controls.items():
                    path = Path(f"configs/probe/limit_ar_teacher_{seed_name}_4layer_{target}{suffix}_10k.json")
                    with self.subTest(path=path):
                        config = load_limit_ar_probe_config(path)
                        self.assertEqual(config.run_version, "0.8.2")
                        self.assertEqual(config.target, target)
                        self.assertEqual(config.input_control, input_control)
                        self.assertEqual(
                            config.train_config,
                            Path(f"configs/train/limit_ar_teacher_{seed_name}_10k_4layer.json"),
                        )
                        self.assertEqual(
                            config.checkpoint_path,
                            Path(f"outputs/limit_ar_teacher_{seed_name}_10k_4layer/model.pt"),
                        )
                        self.assertEqual(
                            config.layers,
                            (
                                "embedding_final",
                                "encoder_layer_0",
                                "encoder_layer_1",
                                "encoder_layer_2",
                                "encoder_layer_3",
                                "sequence_hidden",
                            ),
                        )
                        self.assertTrue(config.random_label_control)

    def test_load_limit_ar_4layer_nuisance_conditioned_probe_configs(self) -> None:
        feature_controls = {
            "action_only": ("action_only", ("nuisance",)),
            "value_only": ("value_only", ("nuisance",)),
            "action_value_only": ("action_value_only", ("nuisance",)),
            "residualize_action": (
                "residualize_action",
                (
                    "embedding_final",
                    "encoder_layer_0",
                    "encoder_layer_1",
                    "encoder_layer_2",
                    "encoder_layer_3",
                    "sequence_hidden",
                ),
            ),
            "residualize_value": (
                "residualize_value",
                (
                    "embedding_final",
                    "encoder_layer_0",
                    "encoder_layer_1",
                    "encoder_layer_2",
                    "encoder_layer_3",
                    "sequence_hidden",
                ),
            ),
            "residualize_action_value": (
                "residualize_action_value",
                (
                    "embedding_final",
                    "encoder_layer_0",
                    "encoder_layer_1",
                    "encoder_layer_2",
                    "encoder_layer_3",
                    "sequence_hidden",
                ),
            ),
        }

        for seed_name in {"seed_a", "seed_b", "seed_c"}:
            for suffix, (feature_control, layers) in feature_controls.items():
                path = Path(
                    f"configs/probe/limit_ar_teacher_{seed_name}_4layer_range_history_only_{suffix}_10k.json"
                )
                with self.subTest(path=path):
                    config = load_limit_ar_probe_config(path)
                    self.assertEqual(config.run_version, "0.8.3")
                    self.assertEqual(config.target, "opponent_hand")
                    self.assertEqual(config.input_control, "history_only")
                    self.assertEqual(config.feature_control, feature_control)
                    self.assertEqual(config.layers, layers)
                    self.assertEqual(
                        config.train_config,
                        Path(f"configs/train/limit_ar_teacher_{seed_name}_10k_4layer.json"),
                    )
                    self.assertEqual(
                        config.checkpoint_path,
                        Path(f"outputs/limit_ar_teacher_{seed_name}_10k_4layer/model.pt"),
                    )

    def test_load_limit_ar_4layer_full_observation_nuisance_configs(self) -> None:
        configs = {
            "action_value_only": ("action_value_only", "none", ("nuisance",)),
            "residualize_action_value": (
                "residualize_action_value",
                "none",
                (
                    "embedding_final",
                    "encoder_layer_0",
                    "encoder_layer_1",
                    "encoder_layer_2",
                    "encoder_layer_3",
                    "sequence_hidden",
                ),
            ),
        }

        for seed_name in {"seed_a", "seed_b", "seed_c"}:
            for suffix, (feature_control, input_control, layers) in configs.items():
                path = Path(f"configs/probe/limit_ar_teacher_{seed_name}_4layer_range_{suffix}_10k.json")
                with self.subTest(path=path):
                    config = load_limit_ar_probe_config(path)
                    self.assertEqual(config.run_version, "0.8.4")
                    self.assertEqual(config.target, "opponent_hand")
                    self.assertEqual(config.input_control, input_control)
                    self.assertEqual(config.feature_control, feature_control)
                    self.assertEqual(config.layers, layers)
                    self.assertEqual(
                        config.train_config,
                        Path(f"configs/train/limit_ar_teacher_{seed_name}_10k_4layer.json"),
                    )
                    self.assertEqual(
                        config.checkpoint_path,
                        Path(f"outputs/limit_ar_teacher_{seed_name}_10k_4layer/model.pt"),
                    )

    def test_load_limit_ar_50k_data_scale_probe_configs(self) -> None:
        full_layers = (
            "embedding_final",
            "encoder_layer_0",
            "encoder_layer_1",
            "encoder_layer_2",
            "encoder_layer_3",
            "sequence_hidden",
        )
        configs = {
            "range": ("none", "none", full_layers),
            "range_history_only": ("none", "history_only", full_layers),
            "range_history_only_action_value_only": ("action_value_only", "history_only", ("nuisance",)),
            "range_history_only_residualize_action_value": (
                "residualize_action_value",
                "history_only",
                full_layers,
            ),
            "range_action_value_only": ("action_value_only", "none", ("nuisance",)),
            "range_residualize_action_value": ("residualize_action_value", "none", full_layers),
        }

        for name, (feature_control, input_control, layers) in configs.items():
            path = Path(f"configs/probe/limit_ar_teacher_seed_d_4layer_{name}_50k.json")
            with self.subTest(path=path):
                config = load_limit_ar_probe_config(path)
                self.assertEqual(config.run_version, "0.9.0")
                self.assertEqual(config.target, "opponent_hand")
                self.assertEqual(config.input_control, input_control)
                self.assertEqual(config.feature_control, feature_control)
                self.assertEqual(config.layers, layers)
                self.assertEqual(config.train_config, Path("configs/train/limit_ar_teacher_seed_d_50k_4layer.json"))
                self.assertEqual(
                    config.checkpoint_path,
                    Path("outputs/limit_ar_teacher_seed_d_50k_4layer/model.pt"),
                )
                self.assertTrue(config.random_label_control)

    def test_load_limit_ar_8layer_capacity_probe_configs(self) -> None:
        full_layers = (
            "embedding_final",
            "encoder_layer_0",
            "encoder_layer_1",
            "encoder_layer_2",
            "encoder_layer_3",
            "encoder_layer_4",
            "encoder_layer_5",
            "encoder_layer_6",
            "encoder_layer_7",
            "sequence_hidden",
        )
        configs = {
            "range": ("none", "none", full_layers),
            "range_history_only": ("none", "history_only", full_layers),
            "range_history_only_action_value_only": ("action_value_only", "history_only", ("nuisance",)),
            "range_history_only_residualize_action_value": (
                "residualize_action_value",
                "history_only",
                full_layers,
            ),
            "range_action_value_only": ("action_value_only", "none", ("nuisance",)),
            "range_residualize_action_value": ("residualize_action_value", "none", full_layers),
        }

        for name, (feature_control, input_control, layers) in configs.items():
            path = Path(f"configs/probe/limit_ar_teacher_seed_e_8layer_{name}_10k.json")
            with self.subTest(path=path):
                config = load_limit_ar_probe_config(path)
                self.assertEqual(config.run_version, "0.9.1")
                self.assertEqual(config.target, "opponent_hand")
                self.assertEqual(config.input_control, input_control)
                self.assertEqual(config.feature_control, feature_control)
                self.assertEqual(config.layers, layers)
                self.assertEqual(config.train_config, Path("configs/train/limit_ar_teacher_seed_e_10k_8layer.json"))
                self.assertEqual(
                    config.checkpoint_path,
                    Path("outputs/limit_ar_teacher_seed_e_10k_8layer/model.pt"),
                )
                self.assertTrue(config.random_label_control)

    def test_load_limit_ar_stabilized_8layer_probe_configs(self) -> None:
        full_layers = (
            "embedding_final",
            "encoder_layer_0",
            "encoder_layer_1",
            "encoder_layer_2",
            "encoder_layer_3",
            "encoder_layer_4",
            "encoder_layer_5",
            "encoder_layer_6",
            "encoder_layer_7",
            "sequence_hidden",
        )
        configs = {
            "range": ("none", "none", full_layers),
            "range_history_only": ("none", "history_only", full_layers),
            "range_history_only_action_value_only": ("action_value_only", "history_only", ("nuisance",)),
            "range_history_only_residualize_action_value": (
                "residualize_action_value",
                "history_only",
                full_layers,
            ),
            "range_action_value_only": ("action_value_only", "none", ("nuisance",)),
            "range_residualize_action_value": ("residualize_action_value", "none", full_layers),
        }

        for name, (feature_control, input_control, layers) in configs.items():
            path = Path(f"configs/probe/limit_ar_teacher_seed_e_8layer_stable_{name}_10k.json")
            with self.subTest(path=path):
                config = load_limit_ar_probe_config(path)
                self.assertEqual(config.run_version, "0.9.2")
                self.assertEqual(config.target, "opponent_hand")
                self.assertEqual(config.input_control, input_control)
                self.assertEqual(config.feature_control, feature_control)
                self.assertEqual(config.layers, layers)
                self.assertEqual(
                    config.train_config,
                    Path("configs/train/limit_ar_teacher_seed_e_10k_8layer_stable.json"),
                )
                self.assertEqual(
                    config.checkpoint_path,
                    Path("outputs/limit_ar_teacher_seed_e_10k_8layer_stable/model.pt"),
                )
                self.assertTrue(config.random_label_control)

    def test_load_limit_ar_50k_scale_replication_probe_configs(self) -> None:
        full_layers = (
            "embedding_final",
            "encoder_layer_0",
            "encoder_layer_1",
            "encoder_layer_2",
            "encoder_layer_3",
            "sequence_hidden",
        )
        configs = {
            "range": ("none", "none", full_layers),
            "range_history_only": ("none", "history_only", full_layers),
            "range_history_only_action_value_only": ("action_value_only", "history_only", ("nuisance",)),
            "range_history_only_residualize_action_value": (
                "residualize_action_value",
                "history_only",
                full_layers,
            ),
            "range_action_value_only": ("action_value_only", "none", ("nuisance",)),
            "range_residualize_action_value": ("residualize_action_value", "none", full_layers),
        }

        for seed_name in {"seed_f", "seed_g"}:
            for name, (feature_control, input_control, layers) in configs.items():
                path = Path(f"configs/probe/limit_ar_teacher_{seed_name}_4layer_{name}_50k.json")
                with self.subTest(path=path):
                    config = load_limit_ar_probe_config(path)
                    self.assertEqual(config.run_version, "0.9.4")
                    self.assertEqual(config.target, "opponent_hand")
                    self.assertEqual(config.input_control, input_control)
                    self.assertEqual(config.feature_control, feature_control)
                    self.assertEqual(config.layers, layers)
                    self.assertEqual(
                        config.train_config,
                        Path(f"configs/train/limit_ar_teacher_{seed_name}_50k_4layer.json"),
                    )
                    self.assertEqual(
                        config.checkpoint_path,
                        Path(f"outputs/limit_ar_teacher_{seed_name}_50k_4layer/model.pt"),
                    )
                    self.assertTrue(config.random_label_control)

    def test_load_limit_ar_action_value_only_probe_configs(self) -> None:
        full_layers = (
            "embedding_final",
            "encoder_layer_0",
            "encoder_layer_1",
            "encoder_layer_2",
            "encoder_layer_3",
            "sequence_hidden",
        )
        configs = {
            "range": ("none", "none", full_layers),
            "range_history_only": ("none", "history_only", full_layers),
            "range_history_only_action_value_only": ("action_value_only", "history_only", ("nuisance",)),
            "range_history_only_residualize_action_value": (
                "residualize_action_value",
                "history_only",
                full_layers,
            ),
            "range_action_value_only": ("action_value_only", "none", ("nuisance",)),
            "range_residualize_action_value": ("residualize_action_value", "none", full_layers),
        }

        for seed_name in {"seed_h", "seed_i", "seed_j"}:
            for name, (feature_control, input_control, layers) in configs.items():
                path = Path(f"configs/probe/limit_ar_action_value_only_{seed_name}_4layer_{name}_10k.json")
                with self.subTest(path=path):
                    config = load_limit_ar_probe_config(path)
                    self.assertEqual(config.run_version, "1.0.2")
                    self.assertEqual(config.target, "opponent_hand")
                    self.assertEqual(config.input_control, input_control)
                    self.assertEqual(config.feature_control, feature_control)
                    self.assertEqual(config.label_control, "none")
                    self.assertEqual(config.layers, layers)
                    self.assertEqual(
                        config.train_config,
                        Path(f"configs/train/limit_ar_action_value_only_{seed_name}_10k_4layer.json"),
                    )
                    self.assertEqual(
                        config.checkpoint_path,
                        Path(f"outputs/limit_ar_action_value_only_{seed_name}_10k_4layer/model.pt"),
                    )
                    self.assertTrue(config.random_label_control)

    def test_load_limit_ar_label_permutation_probe_configs(self) -> None:
        expected = {
            "real_labels": "none",
            "shuffle_global_labels": "shuffle_global",
            "shuffle_within_bucket_labels": "shuffle_within_bucket",
        }

        for suffix, label_control in expected.items():
            path = Path(
                "configs/probe/"
                "limit_ar_action_value_only_seed_h_4layer_range_history_only_residualize_action_value_"
                f"{suffix}_10k.json"
            )
            with self.subTest(path=path):
                config = load_limit_ar_probe_config(path)
                self.assertEqual(config.run_version, "1.0.3")
                self.assertEqual(config.target, "opponent_hand")
                self.assertEqual(config.input_control, "history_only")
                self.assertEqual(config.feature_control, "residualize_action_value")
                self.assertEqual(config.label_control, label_control)
                self.assertEqual(
                    config.train_config,
                    Path("configs/train/limit_ar_action_value_only_seed_h_10k_4layer.json"),
                )
                self.assertEqual(
                    config.checkpoint_path,
                    Path("outputs/limit_ar_action_value_only_seed_h_10k_4layer/model.pt"),
                )
                self.assertFalse(config.random_label_control)

    def test_load_limit_ar_history_destruction_probe_configs(self) -> None:
        full_layers = (
            "embedding_final",
            "encoder_layer_0",
            "encoder_layer_1",
            "encoder_layer_2",
            "encoder_layer_3",
            "sequence_hidden",
        )
        expected = {
            "range_drop_history": ("drop_history", "none", full_layers),
            "range_shuffle_history_order": ("shuffle_history_order", "none", full_layers),
            "range_bag_of_actions": ("bag_of_actions", "none", full_layers),
            "range_no_observation": ("no_observation", "none", full_layers),
            "range_round_position_action_prior": ("none", "round_position_action_only", ("nuisance",)),
            "range_history_only_action_only": ("history_only", "action_only", ("nuisance",)),
        }

        for name, (input_control, feature_control, layers) in expected.items():
            path = Path(f"configs/probe/limit_ar_action_value_only_seed_h_4layer_{name}_10k.json")
            with self.subTest(path=path):
                config = load_limit_ar_probe_config(path)
                self.assertEqual(config.run_version, "1.0.4")
                self.assertEqual(config.target, "opponent_hand")
                self.assertEqual(config.input_control, input_control)
                self.assertEqual(config.feature_control, feature_control)
                self.assertEqual(config.label_control, "none")
                self.assertEqual(config.layers, layers)
                self.assertEqual(
                    config.train_config,
                    Path("configs/train/limit_ar_action_value_only_seed_h_10k_4layer.json"),
                )
                self.assertEqual(
                    config.checkpoint_path,
                    Path("outputs/limit_ar_action_value_only_seed_h_10k_4layer/model.pt"),
                )

    def test_load_limit_ar_probe_uncertainty_repeat_configs(self) -> None:
        for seed_name in {"seed_h", "seed_i", "seed_j"}:
            seen_probe_seeds = set()
            for index in (1, 2, 3):
                for suffix, feature_control, layers in (
                    (
                        "range_history_only_residualize_action_value",
                        "residualize_action_value",
                        (
                            "embedding_final",
                            "encoder_layer_0",
                            "encoder_layer_1",
                            "encoder_layer_2",
                            "encoder_layer_3",
                            "sequence_hidden",
                        ),
                    ),
                    ("range_history_only_action_value_only", "action_value_only", ("nuisance",)),
                ):
                    path = Path(
                        f"configs/probe/limit_ar_action_value_only_{seed_name}_4layer_"
                        f"{suffix}_repeat_{index}_10k.json"
                    )
                    with self.subTest(path=path):
                        config = load_limit_ar_probe_config(path)
                        self.assertEqual(config.run_version, "1.0.7")
                        self.assertEqual(config.target, "opponent_hand")
                        self.assertEqual(config.input_control, "history_only")
                        self.assertEqual(config.feature_control, feature_control)
                        self.assertEqual(config.label_control, "none")
                        self.assertEqual(config.layers, layers)
                        self.assertEqual(
                            config.train_config,
                            Path(f"configs/train/limit_ar_action_value_only_{seed_name}_10k_4layer.json"),
                        )
                        self.assertEqual(
                            config.checkpoint_path,
                            Path(f"outputs/limit_ar_action_value_only_{seed_name}_10k_4layer/model.pt"),
                        )
                        if feature_control == "residualize_action_value":
                            self.assertNotIn(config.probe_seed, seen_probe_seeds)
                            seen_probe_seeds.add(config.probe_seed)

    def test_load_limit_ar_action_value_only_50k_locked_layer_probe_configs(self) -> None:
        full_layers = (
            "embedding_final",
            "encoder_layer_0",
            "encoder_layer_1",
            "encoder_layer_2",
            "encoder_layer_3",
            "sequence_hidden",
        )
        expected = {
            "range_history_only_action_value_only": ("action_value_only", ("nuisance",)),
            "range_history_only_residualize_action_value": ("residualize_action_value", full_layers),
        }

        for seed_name in {"seed_k", "seed_l", "seed_m"}:
            for suffix, (feature_control, layers) in expected.items():
                path = Path(
                    f"configs/probe/limit_ar_action_value_only_{seed_name}_4layer_{suffix}_50k.json"
                )
                with self.subTest(path=path):
                    config = load_limit_ar_probe_config(path)
                    self.assertEqual(config.run_version, "1.0.9")
                    self.assertEqual(config.target, "opponent_hand")
                    self.assertEqual(config.input_control, "history_only")
                    self.assertEqual(config.feature_control, feature_control)
                    self.assertEqual(config.label_control, "none")
                    self.assertEqual(config.layers, layers)
                    self.assertEqual(
                        config.train_config,
                        Path(f"configs/train/limit_ar_action_value_only_{seed_name}_50k_4layer.json"),
                    )
                    self.assertEqual(
                        config.checkpoint_path,
                        Path(f"outputs/limit_ar_action_value_only_{seed_name}_50k_4layer/model.pt"),
                    )

    def test_load_limit_ar_action_value_only_50k_control_bundle_configs(self) -> None:
        full_layers = (
            "embedding_final",
            "encoder_layer_0",
            "encoder_layer_1",
            "encoder_layer_2",
            "encoder_layer_3",
            "sequence_hidden",
        )
        expected = {
            **{
                f"range_history_only_action_value_only_repeat_{index}": (
                    "1.0.12",
                    "history_only",
                    "action_value_only",
                    "none",
                    ("nuisance",),
                )
                for index in (1, 2, 3)
            },
            **{
                f"range_history_only_residualize_action_value_repeat_{index}": (
                    "1.0.12",
                    "history_only",
                    "residualize_action_value",
                    "none",
                    full_layers,
                )
                for index in (1, 2, 3)
            },
            "range_history_only_residualize_action_value_shuffle_global_labels": (
                "1.0.12",
                "history_only",
                "residualize_action_value",
                "shuffle_global",
                full_layers,
            ),
            "range_history_only_residualize_action_value_shuffle_within_bucket_labels": (
                "1.0.12",
                "history_only",
                "residualize_action_value",
                "shuffle_within_bucket",
                full_layers,
            ),
            "range_drop_history": ("1.0.12", "drop_history", "none", "none", full_layers),
            "range_shuffle_history_order": ("1.0.12", "shuffle_history_order", "none", "none", full_layers),
            "range_bag_of_actions": ("1.0.12", "bag_of_actions", "none", "none", full_layers),
            "range_no_observation": ("1.0.12", "no_observation", "none", "none", full_layers),
            "range_round_position_action_prior": (
                "1.0.12",
                "none",
                "round_position_action_only",
                "none",
                ("nuisance",),
            ),
            "range_history_only_action_only": ("1.0.12", "history_only", "action_only", "none", ("nuisance",)),
            "range_drop_history_residualize_action_value": (
                "1.0.12",
                "drop_history",
                "residualize_action_value",
                "none",
                full_layers,
            ),
            "range_shuffle_history_order_residualize_action_value": (
                "1.0.12",
                "shuffle_history_order",
                "residualize_action_value",
                "none",
                full_layers,
            ),
            "range_bag_of_actions_residualize_action_value": (
                "1.0.12",
                "bag_of_actions",
                "residualize_action_value",
                "none",
                full_layers,
            ),
            "range_no_observation_residualize_action_value": (
                "1.0.12",
                "no_observation",
                "residualize_action_value",
                "none",
                full_layers,
            ),
            "range_drop_history_action_value_only": (
                "1.0.13",
                "drop_history",
                "action_value_only",
                "none",
                ("nuisance",),
            ),
            "range_shuffle_history_order_action_value_only": (
                "1.0.13",
                "shuffle_history_order",
                "action_value_only",
                "none",
                ("nuisance",),
            ),
            "range_bag_of_actions_action_value_only": (
                "1.0.13",
                "bag_of_actions",
                "action_value_only",
                "none",
                ("nuisance",),
            ),
            "range_no_observation_action_value_only": (
                "1.0.13",
                "no_observation",
                "action_value_only",
                "none",
                ("nuisance",),
            ),
        }

        for seed_name in {"seed_k", "seed_l", "seed_m"}:
            seen_probe_seeds = set()
            for suffix, (run_version, input_control, feature_control, label_control, layers) in expected.items():
                path = Path(f"configs/probe/limit_ar_action_value_only_{seed_name}_4layer_{suffix}_50k.json")
                with self.subTest(path=path):
                    config = load_limit_ar_probe_config(path)
                    self.assertEqual(config.run_version, run_version)
                    self.assertEqual(config.target, "opponent_hand")
                    self.assertEqual(config.input_control, input_control)
                    self.assertEqual(config.feature_control, feature_control)
                    self.assertEqual(config.label_control, label_control)
                    self.assertEqual(config.layers, layers)
                    self.assertNotIn(config.probe_seed, seen_probe_seeds)
                    seen_probe_seeds.add(config.probe_seed)
                    self.assertEqual(
                        config.train_config,
                        Path(f"configs/train/limit_ar_action_value_only_{seed_name}_50k_4layer.json"),
                    )
                    self.assertEqual(
                        config.checkpoint_path,
                        Path(f"outputs/limit_ar_action_value_only_{seed_name}_50k_4layer/model.pt"),
                    )

    def test_load_limit_ar_action_value_only_50k_history_composition_configs(self) -> None:
        controls = (
            "shuffle_history_order_seed_1",
            "shuffle_history_order_seed_2",
            "shuffle_history_order_seed_3",
            "reverse_history_order",
            "sort_history_actions",
            "history_presence_only",
            "history_last_action_only",
        )
        features = {
            "residualize_action_value": ("encoder_layer_2",),
            "action_value_only": ("nuisance",),
        }

        for seed_name in {"seed_k", "seed_l", "seed_m"}:
            seen_probe_seeds = set()
            for control in controls:
                for feature_control, layers in features.items():
                    path = Path(
                        f"configs/probe/limit_ar_action_value_only_{seed_name}_4layer_range_"
                        f"{control}_{feature_control}_50k.json"
                    )
                    with self.subTest(path=path):
                        config = load_limit_ar_probe_config(path)
                        self.assertEqual(config.run_version, "1.0.15")
                        self.assertEqual(config.target, "opponent_hand")
                        self.assertEqual(config.input_control, control)
                        self.assertEqual(config.feature_control, feature_control)
                        self.assertEqual(config.label_control, "none")
                        self.assertEqual(config.layers, layers)
                        self.assertNotIn(config.probe_seed, seen_probe_seeds)
                        seen_probe_seeds.add(config.probe_seed)
                        self.assertEqual(
                            config.train_config,
                            Path(f"configs/train/limit_ar_action_value_only_{seed_name}_50k_4layer.json"),
                        )
                        self.assertEqual(
                            config.checkpoint_path,
                            Path(f"outputs/limit_ar_action_value_only_{seed_name}_50k_4layer/model.pt"),
                        )

    def test_load_limit_ar_action_value_only_50k_composition_nuisance_configs(self) -> None:
        expected = {
            "history_composition_only": ("history_composition_only", ("nuisance",)),
            "action_value_history_composition_only": ("action_value_history_composition_only", ("nuisance",)),
            "residualize_action_value_history_composition": (
                "residualize_action_value_history_composition",
                ("encoder_layer_2",),
            ),
        }

        for seed_name in {"seed_k", "seed_l", "seed_m"}:
            seen_probe_seeds = set()
            for suffix, (feature_control, layers) in expected.items():
                path = Path(
                    f"configs/probe/limit_ar_action_value_only_{seed_name}_4layer_range_{suffix}_50k.json"
                )
                with self.subTest(path=path):
                    config = load_limit_ar_probe_config(path)
                    self.assertEqual(config.run_version, "1.0.16")
                    self.assertEqual(config.target, "opponent_hand")
                    self.assertEqual(config.input_control, "history_only")
                    self.assertEqual(config.feature_control, feature_control)
                    self.assertEqual(config.label_control, "none")
                    self.assertEqual(config.layers, layers)
                    self.assertNotIn(config.probe_seed, seen_probe_seeds)
                    seen_probe_seeds.add(config.probe_seed)
                    self.assertEqual(
                        config.train_config,
                        Path(f"configs/train/limit_ar_action_value_only_{seed_name}_50k_4layer.json"),
                    )
                    self.assertEqual(
                        config.checkpoint_path,
                        Path(f"outputs/limit_ar_action_value_only_{seed_name}_50k_4layer/model.pt"),
                    )

    def test_load_limit_ar_range_supervised_50k_positive_control_configs(self) -> None:
        expected = {
            "action_value_history_composition_only": (
                "action_value_history_composition_only",
                ("nuisance",),
            ),
            "residualize_action_value_history_composition": (
                "residualize_action_value_history_composition",
                ("encoder_layer_2",),
            ),
        }

        for seed_name in {"seed_d", "seed_f", "seed_g"}:
            seen_probe_seeds = set()
            for suffix, (feature_control, layers) in expected.items():
                path = Path(f"configs/probe/limit_ar_teacher_{seed_name}_4layer_range_{suffix}_50k.json")
                with self.subTest(path=path):
                    config = load_limit_ar_probe_config(path)
                    self.assertEqual(config.run_version, "1.17")
                    self.assertEqual(config.target, "opponent_hand")
                    self.assertEqual(config.input_control, "history_only")
                    self.assertEqual(config.feature_control, feature_control)
                    self.assertEqual(config.label_control, "none")
                    self.assertEqual(config.layers, layers)
                    self.assertNotIn(config.probe_seed, seen_probe_seeds)
                    seen_probe_seeds.add(config.probe_seed)
                    self.assertEqual(
                        config.train_config,
                        Path(f"configs/train/limit_ar_teacher_{seed_name}_50k_4layer.json"),
                    )
                    self.assertEqual(
                        config.checkpoint_path,
                        Path(f"outputs/limit_ar_teacher_{seed_name}_50k_4layer/model.pt"),
                    )

    def test_limit_ar_probe_rejects_invalid_feature_control(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe_config = json.loads(Path("configs/probe/limit_ar_teacher_range_smoke.json").read_text())
            probe_config["feature_control"] = "not_a_control"
            probe_config_path = Path(temp_dir) / "probe_config.json"
            probe_config_path.write_text(json.dumps(probe_config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "feature_control"):
                load_limit_ar_probe_config(probe_config_path)

    def test_limit_ar_probe_rejects_invalid_label_control(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe_config = json.loads(Path("configs/probe/limit_ar_teacher_range_smoke.json").read_text())
            probe_config["label_control"] = "not_a_control"
            probe_config_path = Path(temp_dir) / "probe_config.json"
            probe_config_path.write_text(json.dumps(probe_config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "label_control"):
                load_limit_ar_probe_config(probe_config_path)

    def test_load_limit_ar_probe_mlp_capacity_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe_config = json.loads(Path("configs/probe/limit_ar_teacher_range_smoke.json").read_text())
            probe_config["probe_family"] = "mlp"
            probe_config["mlp_hidden_size"] = 8
            probe_config_path = Path(temp_dir) / "probe_config.json"
            probe_config_path.write_text(json.dumps(probe_config), encoding="utf-8")

            config = load_limit_ar_probe_config(probe_config_path)

        self.assertEqual(config.probe_family, "mlp")
        self.assertEqual(config.mlp_hidden_size, 8)

    def test_load_no_range_50k_mlp_capacity_configs(self) -> None:
        for seed in ("k", "l", "m"):
            for suffix in (
                "history_only_residualize_action_value_mlp",
                "residualize_action_value_history_composition_mlp",
            ):
                with self.subTest(seed=seed, suffix=suffix):
                    config = load_limit_ar_probe_config(
                        Path(
                            "configs/probe/"
                            f"limit_ar_action_value_only_seed_{seed}_4layer_range_{suffix}_50k.json"
                        )
                    )

                self.assertEqual(config.run_version, "1.18")
                self.assertEqual(config.probe_family, "mlp")
                self.assertEqual(config.mlp_hidden_size, 64)
                self.assertEqual(config.layers, ("encoder_layer_2",))

    def test_run_limit_ar_probe_nuisance_only_smoke_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            train_config = json.loads(Path("configs/train/limit_ar_teacher_smoke_saved.json").read_text())
            train_config["artifact_dir"] = str(Path(temp_dir) / "train")
            train_config_path = Path(temp_dir) / "train_config.json"
            train_config_path.write_text(json.dumps(train_config), encoding="utf-8")
            train_result = run_limit_ar_train(config_path=train_config_path)

            probe_config = json.loads(Path("configs/probe/limit_ar_teacher_range_smoke.json").read_text())
            probe_config["train_config"] = str(train_config_path)
            probe_config["checkpoint_path"] = str(train_result.artifact_dir / "model.pt")
            probe_config["artifact_dir"] = str(Path(temp_dir) / "probe")
            probe_config["feature_control"] = "action_only"
            probe_config["layers"] = ["nuisance"]
            probe_config_path = Path(temp_dir) / "probe_config.json"
            probe_config_path.write_text(json.dumps(probe_config), encoding="utf-8")

            result = run_limit_ar_probe(config_path=probe_config_path)
            metrics = json.loads((result.artifact_dir / "probe_metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(result.layer_count, 1)
        self.assertEqual(metrics["feature_control"], "action_only")
        self.assertEqual(metrics["label_control"], "none")
        self.assertFalse(metrics["checkpoint_loaded"])
        self.assertEqual(metrics["records"][0]["layer"], "nuisance")

    def test_run_limit_ar_probe_mlp_capacity_smoke_writes_appendix_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            train_config = json.loads(Path("configs/train/limit_ar_teacher_smoke_saved.json").read_text())
            train_config["artifact_dir"] = str(Path(temp_dir) / "train")
            train_config_path = Path(temp_dir) / "train_config.json"
            train_config_path.write_text(json.dumps(train_config), encoding="utf-8")
            train_result = run_limit_ar_train(config_path=train_config_path)

            probe_config = json.loads(Path("configs/probe/limit_ar_teacher_range_smoke.json").read_text())
            probe_config["train_config"] = str(train_config_path)
            probe_config["checkpoint_path"] = str(train_result.artifact_dir / "model.pt")
            probe_config["artifact_dir"] = str(Path(temp_dir) / "probe_mlp")
            probe_config["probe_family"] = "mlp"
            probe_config["mlp_hidden_size"] = 8
            probe_config["layers"] = ["encoder_layer_0"]
            probe_config_path = Path(temp_dir) / "probe_config_mlp.json"
            probe_config_path.write_text(json.dumps(probe_config), encoding="utf-8")

            result = run_limit_ar_probe(config_path=probe_config_path)
            metrics = json.loads((result.artifact_dir / "probe_metrics.json").read_text(encoding="utf-8"))
            correctness_exists = (result.artifact_dir / "probe_correctness.jsonl").exists()

        self.assertEqual(metrics["probe_family"], "mlp")
        self.assertEqual(metrics["probe_capacity_role"], "appendix_sensitivity")
        self.assertEqual(metrics["mlp_hidden_size"], 8)
        self.assertTrue(all(record["probe_family"] == "mlp" for record in metrics["records"]))
        self.assertTrue(correctness_exists)

    def test_history_composition_feature_shape(self) -> None:
        torch, _ = require_torch()
        point = DecisionPoint(
            hand_id="h1",
            step_index=0,
            acting_player=0,
            visible_state={
                "acting_player": 0,
                "round": "preflop",
                "private_cards": ["mask"],
                "public_cards": [],
                "betting_history": [
                    {"type": "action", "action": "bet"},
                    {"type": "action", "action": "call"},
                    {"type": "action", "action": "bet"},
                ],
                "legal_actions": ("check", "bet"),
            },
            legal_actions=("check", "bet"),
            action="bet",
            opponent_private_card="As",
            opponent_rank="AA",
            terminal_return=1,
        )
        features = _history_composition_features((point,), {"policy": torch.tensor([0])}, torch)
        empty_features = _history_composition_features(tuple(), {"policy": torch.empty(0)}, torch)

        # 2 length + 4 counts + 4 normalized counts + 4 presence + 4 first + 4 last + 8 position summaries.
        self.assertEqual(tuple(features.shape), (1, 30))
        self.assertEqual(tuple(empty_features.shape), (0, 30))
        self.assertEqual(float(features[0, 0].item()), 3.0)
        self.assertEqual(float(features[0, 2].item()), 2.0)
        self.assertEqual(float(features[0, 3].item()), 1.0)
        self.assertEqual(float(features[0, 10].item()), 1.0)
        self.assertEqual(float(features[0, 14].item()), 1.0)
        self.assertEqual(float(features[0, 18].item()), 1.0)

    def test_run_limit_ar_probe_residualized_smoke_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            train_config = json.loads(Path("configs/train/limit_ar_teacher_smoke_saved.json").read_text())
            train_config["artifact_dir"] = str(Path(temp_dir) / "train")
            train_config_path = Path(temp_dir) / "train_config.json"
            train_config_path.write_text(json.dumps(train_config), encoding="utf-8")
            train_result = run_limit_ar_train(config_path=train_config_path)

            probe_config = json.loads(Path("configs/probe/limit_ar_teacher_range_smoke.json").read_text())
            probe_config["train_config"] = str(train_config_path)
            probe_config["checkpoint_path"] = str(train_result.artifact_dir / "model.pt")
            probe_config["artifact_dir"] = str(Path(temp_dir) / "probe")
            probe_config["feature_control"] = "residualize_action_value"
            probe_config_path = Path(temp_dir) / "probe_config.json"
            probe_config_path.write_text(json.dumps(probe_config), encoding="utf-8")

            result = run_limit_ar_probe(config_path=probe_config_path)
            metrics = json.loads((result.artifact_dir / "probe_metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(result.layer_count, 3)
        self.assertEqual(metrics["feature_control"], "residualize_action_value")
        self.assertEqual(metrics["label_control"], "none")
        self.assertTrue(metrics["checkpoint_loaded"])
        self.assertEqual(metrics["records"][0]["feature_control"], "residualize_action_value")

    def test_run_limit_ar_probe_label_controls_write_metrics(self) -> None:
        for label_control in ("shuffle_global", "shuffle_within_bucket"):
            with self.subTest(label_control=label_control):
                with tempfile.TemporaryDirectory() as temp_dir:
                    train_config = json.loads(Path("configs/train/limit_ar_teacher_smoke_saved.json").read_text())
                    train_config["artifact_dir"] = str(Path(temp_dir) / "train")
                    train_config_path = Path(temp_dir) / "train_config.json"
                    train_config_path.write_text(json.dumps(train_config), encoding="utf-8")
                    train_result = run_limit_ar_train(config_path=train_config_path)

                    probe_config = json.loads(Path("configs/probe/limit_ar_teacher_range_smoke.json").read_text())
                    probe_config["train_config"] = str(train_config_path)
                    probe_config["checkpoint_path"] = str(train_result.artifact_dir / "model.pt")
                    probe_config["artifact_dir"] = str(Path(temp_dir) / "probe")
                    probe_config["label_control"] = label_control
                    probe_config["random_label_control"] = False
                    probe_config_path = Path(temp_dir) / "probe_config.json"
                    probe_config_path.write_text(json.dumps(probe_config), encoding="utf-8")

                    result = run_limit_ar_probe(config_path=probe_config_path)
                    metrics = json.loads((result.artifact_dir / "probe_metrics.json").read_text(encoding="utf-8"))

                self.assertEqual(metrics["label_control"], label_control)
                self.assertEqual(metrics["label_control_stats"]["control"], label_control)
                self.assertGreater(metrics["label_control_stats"]["train_point_count"], 0)
                self.assertTrue(all(record["label_control"] == label_control for record in metrics["records"]))

    def test_tiny_ar_model_exposes_probe_layers(self) -> None:
        torch, _ = require_torch()
        model = build_tiny_ar_model(
            vocab_size=LimitTokenizer().vocab_size,
            action_size=4,
            opponent_card_size=169,
            hidden_size=16,
            num_heads=4,
            num_layers=1,
        )
        layers = model.encode_layers(torch.tensor([[1, 2, 3, -1], [4, 5, -1, -1]], dtype=torch.long))

        self.assertEqual(layers["embedding_final"].shape, (2, 16))
        self.assertEqual(layers["encoder_layer_0"].shape, (2, 16))
        self.assertEqual(layers["sequence_hidden"].shape, (2, 16))

    def test_tiny_ar_model_exposes_4layer_probe_layers(self) -> None:
        torch, _ = require_torch()
        model = build_tiny_ar_model(
            vocab_size=LimitTokenizer().vocab_size,
            action_size=4,
            opponent_card_size=169,
            hidden_size=16,
            num_heads=4,
            num_layers=4,
        )
        layers = model.encode_layers(torch.tensor([[1, 2, 3, -1], [4, 5, -1, -1]], dtype=torch.long))

        self.assertEqual(
            set(layers),
            {
                "embedding_final",
                "encoder_layer_0",
                "encoder_layer_1",
                "encoder_layer_2",
                "encoder_layer_3",
                "sequence_hidden",
            },
        )
        for layer in layers.values():
            self.assertEqual(layer.shape, (2, 16))

    def test_tiny_ar_model_exposes_8layer_probe_layers(self) -> None:
        torch, _ = require_torch()
        model = build_tiny_ar_model(
            vocab_size=LimitTokenizer().vocab_size,
            action_size=4,
            opponent_card_size=169,
            hidden_size=16,
            num_heads=4,
            num_layers=8,
        )
        layers = model.encode_layers(torch.tensor([[1, 2, 3, -1], [4, 5, -1, -1]], dtype=torch.long))

        self.assertEqual(len(layers), 10)
        self.assertEqual(layers["encoder_layer_7"].shape, (2, 16))
        self.assertEqual(layers["sequence_hidden"].shape, (2, 16))

    def test_run_limit_ar_probe_smoke_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            train_config = json.loads(Path("configs/train/limit_ar_teacher_smoke_saved.json").read_text())
            train_config["artifact_dir"] = str(Path(temp_dir) / "train")
            train_config_path = Path(temp_dir) / "train_config.json"
            train_config_path.write_text(json.dumps(train_config), encoding="utf-8")
            train_result = run_limit_ar_train(config_path=train_config_path)

            probe_config = json.loads(Path("configs/probe/limit_ar_teacher_range_smoke.json").read_text())
            probe_config["train_config"] = str(train_config_path)
            probe_config["checkpoint_path"] = str(train_result.artifact_dir / "model.pt")
            probe_config["artifact_dir"] = str(Path(temp_dir) / "probe")
            probe_config_path = Path(temp_dir) / "probe_config.json"
            probe_config_path.write_text(json.dumps(probe_config), encoding="utf-8")

            result = run_limit_ar_probe(config_path=probe_config_path)
            metrics = json.loads((result.artifact_dir / "probe_metrics.json").read_text(encoding="utf-8"))
            table_lines = (result.artifact_dir / "probe_table.jsonl").read_text(encoding="utf-8").splitlines()
            correctness_lines = (
                result.artifact_dir / "probe_correctness.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            correctness_first = json.loads(correctness_lines[0])

        self.assertEqual(result.layer_count, 3)
        self.assertEqual(result.metric_name, "top10_accuracy")
        self.assertEqual(metrics["target"], "opponent_hand")
        self.assertEqual(metrics["checksum"], result.checksum)
        self.assertEqual(metrics["probe_seed"], probe_config["probe_seed"])
        self.assertEqual(metrics["control_seed"], probe_config["control_seed"])
        self.assertEqual(metrics["config_path"], str(probe_config_path))
        self.assertEqual(metrics["artifact_dir"], str(Path(temp_dir) / "probe"))
        self.assertEqual(metrics["configured_artifact_dir"], str(Path(temp_dir) / "probe"))
        self.assertEqual(metrics["random_label_control"], probe_config["random_label_control"])
        self.assertEqual(metrics["correctness_artifact"], "probe_correctness.jsonl")
        self.assertEqual(metrics["correctness_summary"]["row_count"], len(correctness_lines))
        self.assertEqual(
            metrics["correctness_summary"]["rows_by_layer_control"]["embedding_final|none"],
            metrics["test_points"],
        )
        self.assertIsInstance(metrics["correctness_sha256"], str)
        self.assertEqual(len(metrics["correctness_sha256"]), 64)
        self.assertEqual(len(table_lines), 6)
        self.assertEqual(len(correctness_lines), 6 * metrics["test_points"])
        self.assertEqual(correctness_first["split"], "test")
        self.assertIn("hand_id", correctness_first)
        self.assertIn("step_index", correctness_first)
        self.assertEqual(
            correctness_first["point_id"],
            f"{correctness_first['hand_id']}:{correctness_first['step_index']}",
        )
        self.assertEqual(correctness_first["layer"], "embedding_final")
        self.assertEqual(correctness_first["control"], "none")
        self.assertEqual(correctness_first["target"], "opponent_hand")
        self.assertIn("target_label", correctness_first)
        self.assertIn("top1_correct", correctness_first)
        self.assertIn("top5_correct", correctness_first)
        self.assertIn("top10_correct", correctness_first)
        self.assertTrue(any(record["control"] == "random_label" for record in metrics["records"]))


if __name__ == "__main__":
    unittest.main()
