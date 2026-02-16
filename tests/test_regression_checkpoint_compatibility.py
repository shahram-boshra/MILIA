#!/usr/bin/env python3
# test_regression_checkpoint_compatibility.py
# Section 4.2: Regression Test — Checkpoint Compatibility
# Category: Regression | Priority: HIGH | Est. CI Time: ~15s
#
# What it tests:
#   Checkpoints saved with the current code can be loaded correctly,
#   and the v2.0 format is maintained. Also tests backward compatibility
#   with v1.0 format if applicable.
#
# Modules exercised:
#   - milia_pipeline/models/post_training/checkpoint/checkpoint_manager.py
#     — CheckpointManager, CHECKPOINT_FORMAT_VERSION
#   - milia_pipeline/models/post_training/inference/model_loader.py
#     — ModelLoader.load_from_checkpoint()
#
# Scope:
#   Creates a checkpoint with a known model, saves it, reloads it.
#   Asserts: model weights are identical (bitwise), data_info including
#   structural_features_config is preserved, format version is '2.0'.
#
# Mock Pollution Prevention:
#   - NO sys.modules mocking at module level
#   - All mocks use @patch decorators or context managers at test level
#   - No teardown_module() needed
#
# Author: MILIA Team
# Version: 1.0.0

"""
Regression test suite for checkpoint compatibility (Section 4.2).

Validates that:
1. v2.0 checkpoint format is correctly saved and loaded
2. Model weights survive a save → load round-trip (bitwise identical)
3. hyper_parameters dict is fully preserved
4. data_info including structural_features_config is preserved
5. version_info metadata is correctly populated
6. Backward compatibility with v1.0 format checkpoints
7. CheckpointManager path resolution works correctly
8. ModelLoader.load_from_checkpoint() recreates model correctly
"""

import sys
from pathlib import Path

# ─── Add project root to Python path FIRST ───────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ──────────────────────────────────────────────────────────────────────────────

import logging
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# ─── Module imports (under test) ─────────────────────────────────────────────
from milia_pipeline.models.post_training.checkpoint.checkpoint_manager import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointManager,
)

# ──────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# PYTEST MARKERS
# ═══════════════════════════════════════════════════════════════════════════════

pytestmark = [
    pytest.mark.regression,
]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: MINIMAL DETERMINISTIC MODEL
# ═══════════════════════════════════════════════════════════════════════════════


class _MinimalGNNModel(nn.Module):
    """
    Minimal deterministic model for checkpoint round-trip testing.

    This model is intentionally simple — a two-layer MLP — to decouple
    checkpoint format tests from ModelFactory/registry dependencies.
    Using a known architecture ensures bitwise weight comparison is
    unambiguous and does not depend on external model registration.
    """

    def __init__(self, in_channels: int = 11, hidden_channels: int = 32, out_channels: int = 1):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        return self.lin2(self.relu(self.lin1(x)))


def _build_reference_hyper_parameters() -> dict[str, Any]:
    """
    Build a representative hyper_parameters dict matching the production
    schema documented in CheckpointManager.save() and ModelLoader._load().

    Keys follow the COMPLETE dict pattern:
        model_name, task_type, hyperparameters, model_info,
        wrapper_info, target_selection_config
    """
    return {
        "model_name": "GCN",
        "task_type": "graph_regression",
        "hyperparameters": {
            "in_channels": 11,
            "hidden_channels": 32,
            "out_channels": 1,
            "num_layers": 2,
            "dropout": 0.0,
        },
        "model_info": {
            "task_type": "graph_regression",
            "uses_edge_features": False,
            "hyperparameters_values": {
                "in_channels": 11,
                "hidden_channels": 32,
                "out_channels": 1,
            },
        },
        "wrapper_info": {
            "wrapper_type": "GraphLevelModelWrapper",
            "graph_level_pooling": "mean",
        },
        "target_selection_config": {
            "mode": "properties",
            "selected_properties": ["energy"],
        },
    }


def _build_reference_data_info() -> dict[str, Any]:
    """
    Build a representative data_info dict including structural_features_config.

    This tests the featurization config persistence fix (v1.6.0).
    """
    return {
        "dataset_type": "DFT",
        "num_node_features": 11,
        "num_edge_features": 0,
        "structural_features_config": {
            "atom": ["atomic_number", "degree", "formal_charge"],
            "bond": [],
            "molecule": [],
        },
        "target_properties": ["energy"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def tmp_working_dir(tmp_path):
    """Provide a temporary working_root_dir for DI-pattern tests."""
    return tmp_path


@pytest.fixture
def checkpoint_manager(tmp_working_dir):
    """Provide a CheckpointManager instance."""
    return CheckpointManager(working_root_dir=tmp_working_dir)


@pytest.fixture
def minimal_model():
    """Provide a deterministic minimal model with fixed seed."""
    torch.manual_seed(42)
    model = _MinimalGNNModel(in_channels=11, hidden_channels=32, out_channels=1)
    return model


@pytest.fixture
def reference_hyper_parameters():
    """Provide a representative hyper_parameters dict."""
    return _build_reference_hyper_parameters()


@pytest.fixture
def reference_data_info():
    """Provide a representative data_info dict."""
    return _build_reference_data_info()


@pytest.fixture
def reference_optimizer(minimal_model):
    """Provide an optimizer bound to the minimal model."""
    return torch.optim.Adam(minimal_model.parameters(), lr=0.001)


@pytest.fixture
def reference_scheduler(reference_optimizer):
    """Provide a scheduler bound to the reference optimizer."""
    return torch.optim.lr_scheduler.StepLR(reference_optimizer, step_size=10)


@pytest.fixture
def saved_v2_checkpoint_path(
    checkpoint_manager,
    minimal_model,
    reference_optimizer,
    reference_scheduler,
    reference_hyper_parameters,
    reference_data_info,
    tmp_working_dir,
):
    """
    Save a v2.0 checkpoint and return its path.

    This fixture creates a fully populated checkpoint with all fields
    that the production CheckpointManager.save() supports, including
    hyper_parameters, data_info, optimizer, scheduler, and metrics.
    """
    filepath = "checkpoints/test_model_v2.pt"
    metrics_history = {
        "train_loss": [0.5, 0.3, 0.2],
        "val_loss": [0.6, 0.4, 0.35],
    }

    resolved_path = checkpoint_manager.save(
        filepath=filepath,
        model=minimal_model,
        optimizer=reference_optimizer,
        scheduler=reference_scheduler,
        epoch=10,
        global_step=500,
        metrics_history=metrics_history,
        best_val_loss=0.35,
        hyper_parameters=reference_hyper_parameters,
        data_info=reference_data_info,
    )
    return resolved_path


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 1: v2.0 CHECKPOINT FORMAT INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════════


class TestV2CheckpointFormatIntegrity:
    """
    Verify that the v2.0 checkpoint format is correctly saved and loaded.

    These tests exercise CheckpointManager.save() and .load() directly,
    asserting that every documented field survives a round-trip.
    """

    def test_checkpoint_format_version_is_2_0(self, saved_v2_checkpoint_path, checkpoint_manager):
        """Checkpoint format version must be '2.0'."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        version_info = checkpoint.get("version_info", {})
        assert version_info.get("checkpoint_format_version") == "2.0", (
            f"Expected format version '2.0', got '{version_info.get('checkpoint_format_version')}'"
        )

    def test_checkpoint_format_version_constant_matches(self):
        """Module-level CHECKPOINT_FORMAT_VERSION must be '2.0'."""
        assert CHECKPOINT_FORMAT_VERSION == "2.0", (
            f"CHECKPOINT_FORMAT_VERSION constant is '{CHECKPOINT_FORMAT_VERSION}', expected '2.0'"
        )

    def test_is_v2_checkpoint_returns_true(self, saved_v2_checkpoint_path, checkpoint_manager):
        """is_v2_checkpoint() must return True for v2.0 checkpoints."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        assert checkpoint_manager.is_v2_checkpoint(checkpoint) is True

    def test_version_info_contains_required_keys(
        self, saved_v2_checkpoint_path, checkpoint_manager
    ):
        """version_info must contain all required metadata keys."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        version_info = checkpoint["version_info"]

        required_keys = {
            "milia_version",
            "checkpoint_format_version",
            "pytorch_version",
            "pyg_version",
            "created_at",
        }
        assert required_keys.issubset(version_info.keys()), (
            f"Missing version_info keys: {required_keys - set(version_info.keys())}"
        )

    def test_version_info_created_at_is_valid_iso(
        self, saved_v2_checkpoint_path, checkpoint_manager
    ):
        """version_info.created_at must be a valid ISO 8601 datetime."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        created_at = checkpoint["version_info"]["created_at"]
        # datetime.fromisoformat() will raise ValueError if invalid
        parsed = datetime.fromisoformat(created_at)
        assert isinstance(parsed, datetime)

    def test_version_info_pytorch_version_is_string(
        self, saved_v2_checkpoint_path, checkpoint_manager
    ):
        """version_info.pytorch_version must be a non-empty string."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        pytorch_version = checkpoint["version_info"]["pytorch_version"]
        assert isinstance(pytorch_version, str)
        assert len(pytorch_version) > 0

    def test_checkpoint_file_exists_on_disk(self, saved_v2_checkpoint_path):
        """Saved checkpoint file must physically exist."""
        assert saved_v2_checkpoint_path.exists(), (
            f"Checkpoint file does not exist: {saved_v2_checkpoint_path}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 2: MODEL WEIGHTS BITWISE ROUND-TRIP
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelWeightsBitwiseRoundTrip:
    """
    Verify that model weights are bitwise identical after save → load.

    This is the core regression test: any change to the save/load pipeline
    that corrupts weights must be detected immediately.
    """

    def test_model_state_dict_bitwise_identical(
        self, saved_v2_checkpoint_path, checkpoint_manager, minimal_model
    ):
        """
        model_state_dict must be bitwise identical after round-trip.

        Compares every parameter tensor using torch.equal() which checks
        exact element-wise equality (not approximate).
        """
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        loaded_state_dict = checkpoint["model_state_dict"]
        original_state_dict = minimal_model.state_dict()

        # Same keys
        assert set(loaded_state_dict.keys()) == set(original_state_dict.keys()), (
            f"State dict key mismatch.\n"
            f"  Original: {sorted(original_state_dict.keys())}\n"
            f"  Loaded:   {sorted(loaded_state_dict.keys())}"
        )

        # Bitwise identical values
        for key in original_state_dict:
            assert torch.equal(original_state_dict[key], loaded_state_dict[key]), (
                f"Weight mismatch for key '{key}'.\n"
                f"  Max diff: {(original_state_dict[key] - loaded_state_dict[key]).abs().max().item()}"
            )

    def test_loaded_model_produces_same_output(
        self, saved_v2_checkpoint_path, checkpoint_manager, minimal_model
    ):
        """
        Model loaded from checkpoint must produce identical outputs.

        Creates a fresh model, loads state_dict from checkpoint, and
        verifies that forward() produces the same result as the original.
        """
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)

        # Create fresh model with same architecture
        torch.manual_seed(99)  # Different seed — weights will differ
        fresh_model = _MinimalGNNModel(in_channels=11, hidden_channels=32, out_channels=1)

        # Before loading — models should differ
        test_input = torch.randn(4, 11)
        with torch.no_grad():
            original_output = minimal_model(test_input)
            fresh_output_before = fresh_model(test_input)

        # Outputs should differ before loading (sanity check)
        assert not torch.equal(original_output, fresh_output_before), (
            "Sanity check failed: fresh model already matches original before loading checkpoint"
        )

        # Load state dict
        fresh_model.load_state_dict(checkpoint["model_state_dict"])
        fresh_model.eval()
        minimal_model.eval()

        with torch.no_grad():
            fresh_output_after = fresh_model(test_input)

        # After loading — outputs must be identical
        assert torch.equal(original_output, fresh_output_after), (
            f"Output mismatch after loading checkpoint.\n"
            f"  Max diff: {(original_output - fresh_output_after).abs().max().item()}"
        )

    def test_state_dict_param_count_preserved(
        self, saved_v2_checkpoint_path, checkpoint_manager, minimal_model
    ):
        """Number of parameters must be preserved across round-trip."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        loaded_state_dict = checkpoint["model_state_dict"]
        original_state_dict = minimal_model.state_dict()

        original_count = sum(p.numel() for p in original_state_dict.values())
        loaded_count = sum(p.numel() for p in loaded_state_dict.values())

        assert original_count == loaded_count, (
            f"Parameter count mismatch: original={original_count}, loaded={loaded_count}"
        )

    def test_state_dict_dtypes_preserved(
        self, saved_v2_checkpoint_path, checkpoint_manager, minimal_model
    ):
        """Tensor dtypes must be preserved across round-trip."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        loaded_state_dict = checkpoint["model_state_dict"]
        original_state_dict = minimal_model.state_dict()

        for key in original_state_dict:
            assert original_state_dict[key].dtype == loaded_state_dict[key].dtype, (
                f"dtype mismatch for key '{key}': "
                f"original={original_state_dict[key].dtype}, "
                f"loaded={loaded_state_dict[key].dtype}"
            )

    def test_state_dict_shapes_preserved(
        self, saved_v2_checkpoint_path, checkpoint_manager, minimal_model
    ):
        """Tensor shapes must be preserved across round-trip."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        loaded_state_dict = checkpoint["model_state_dict"]
        original_state_dict = minimal_model.state_dict()

        for key in original_state_dict:
            assert original_state_dict[key].shape == loaded_state_dict[key].shape, (
                f"Shape mismatch for key '{key}': "
                f"original={original_state_dict[key].shape}, "
                f"loaded={loaded_state_dict[key].shape}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 3: HYPER_PARAMETERS PRESERVATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestHyperParametersPreservation:
    """
    Verify that hyper_parameters are fully preserved in the checkpoint.

    Tests both the top-level hyper_parameters dict and its nested
    sub-dicts (hyperparameters, model_info, wrapper_info,
    target_selection_config).
    """

    def test_hyper_parameters_round_trip(
        self,
        saved_v2_checkpoint_path,
        checkpoint_manager,
        reference_hyper_parameters,
    ):
        """Complete hyper_parameters dict must survive round-trip."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        loaded_hp = checkpoint["hyper_parameters"]
        assert loaded_hp == reference_hyper_parameters, (
            f"hyper_parameters mismatch.\n"
            f"  Expected: {reference_hyper_parameters}\n"
            f"  Got:      {loaded_hp}"
        )

    def test_model_name_preserved(
        self,
        saved_v2_checkpoint_path,
        checkpoint_manager,
    ):
        """model_name must be preserved."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        assert checkpoint_manager.get_model_name(checkpoint) == "GCN"

    def test_get_hyper_parameters_method(
        self,
        saved_v2_checkpoint_path,
        checkpoint_manager,
        reference_hyper_parameters,
    ):
        """get_hyper_parameters() must return the full dict."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        hp = checkpoint_manager.get_hyper_parameters(checkpoint)
        assert hp == reference_hyper_parameters

    def test_hyperparameters_inner_dict_preserved(
        self,
        saved_v2_checkpoint_path,
        checkpoint_manager,
    ):
        """hyper_parameters['hyperparameters'] nested dict preserved."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        inner_hp = checkpoint["hyper_parameters"]["hyperparameters"]
        assert inner_hp["in_channels"] == 11
        assert inner_hp["hidden_channels"] == 32
        assert inner_hp["out_channels"] == 1
        assert inner_hp["num_layers"] == 2
        assert inner_hp["dropout"] == 0.0

    def test_model_info_nested_dict_preserved(
        self,
        saved_v2_checkpoint_path,
        checkpoint_manager,
    ):
        """hyper_parameters['model_info'] nested dict preserved."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        model_info = checkpoint["hyper_parameters"]["model_info"]
        assert model_info["task_type"] == "graph_regression"
        assert model_info["uses_edge_features"] is False
        assert "hyperparameters_values" in model_info

    def test_wrapper_info_preserved(
        self,
        saved_v2_checkpoint_path,
        checkpoint_manager,
    ):
        """hyper_parameters['wrapper_info'] preserved."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        wrapper_info = checkpoint["hyper_parameters"]["wrapper_info"]
        assert wrapper_info["wrapper_type"] == "GraphLevelModelWrapper"
        assert wrapper_info["graph_level_pooling"] == "mean"

    def test_target_selection_config_preserved(
        self,
        saved_v2_checkpoint_path,
        checkpoint_manager,
    ):
        """hyper_parameters['target_selection_config'] preserved."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        tsc = checkpoint["hyper_parameters"]["target_selection_config"]
        assert tsc["mode"] == "properties"
        assert tsc["selected_properties"] == ["energy"]

    def test_task_type_preserved(
        self,
        saved_v2_checkpoint_path,
        checkpoint_manager,
    ):
        """hyper_parameters['task_type'] preserved."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        assert checkpoint["hyper_parameters"]["task_type"] == "graph_regression"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 4: DATA_INFO AND STRUCTURAL_FEATURES_CONFIG PRESERVATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataInfoPreservation:
    """
    Verify that data_info, including structural_features_config, is preserved.

    This tests the featurization config persistence fix (v1.6.0): training-time
    featurization config is saved in checkpoints and correctly applied during
    prediction to avoid dimension mismatches.
    """

    def test_data_info_round_trip(
        self,
        saved_v2_checkpoint_path,
        checkpoint_manager,
        reference_data_info,
    ):
        """Complete data_info dict must survive round-trip."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        loaded_data_info = checkpoint["data_info"]
        assert loaded_data_info == reference_data_info, (
            f"data_info mismatch.\n"
            f"  Expected: {reference_data_info}\n"
            f"  Got:      {loaded_data_info}"
        )

    def test_structural_features_config_preserved(
        self,
        saved_v2_checkpoint_path,
        checkpoint_manager,
    ):
        """structural_features_config must be preserved exactly."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        sfc = checkpoint["data_info"]["structural_features_config"]
        assert sfc["atom"] == ["atomic_number", "degree", "formal_charge"]
        assert sfc["bond"] == []
        assert sfc["molecule"] == []

    def test_dataset_type_in_data_info(
        self,
        saved_v2_checkpoint_path,
        checkpoint_manager,
    ):
        """data_info['dataset_type'] preserved."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        assert checkpoint["data_info"]["dataset_type"] == "DFT"

    def test_num_node_features_in_data_info(
        self,
        saved_v2_checkpoint_path,
        checkpoint_manager,
    ):
        """data_info['num_node_features'] preserved."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        assert checkpoint["data_info"]["num_node_features"] == 11

    def test_target_properties_in_data_info(
        self,
        saved_v2_checkpoint_path,
        checkpoint_manager,
    ):
        """data_info['target_properties'] preserved."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        assert checkpoint["data_info"]["target_properties"] == ["energy"]


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 5: TRAINING STATE PRESERVATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestTrainingStatePreservation:
    """
    Verify that training state (epoch, step, optimizer, scheduler,
    metrics_history, best_val_loss) survives round-trip.
    """

    def test_epoch_preserved(self, saved_v2_checkpoint_path, checkpoint_manager):
        """epoch must be preserved."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        assert checkpoint["epoch"] == 10

    def test_global_step_preserved(self, saved_v2_checkpoint_path, checkpoint_manager):
        """global_step must be preserved."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        assert checkpoint["global_step"] == 500

    def test_best_val_loss_preserved(self, saved_v2_checkpoint_path, checkpoint_manager):
        """best_val_loss must be preserved."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        assert checkpoint["best_val_loss"] == pytest.approx(0.35)

    def test_metrics_history_preserved(self, saved_v2_checkpoint_path, checkpoint_manager):
        """metrics_history must be preserved."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        mh = checkpoint["metrics_history"]
        assert mh["train_loss"] == [0.5, 0.3, 0.2]
        assert mh["val_loss"] == [0.6, 0.4, 0.35]

    def test_optimizer_state_dict_preserved(
        self,
        saved_v2_checkpoint_path,
        checkpoint_manager,
    ):
        """optimizer_state_dict must be present and non-empty."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        assert "optimizer_state_dict" in checkpoint
        assert len(checkpoint["optimizer_state_dict"]) > 0

    def test_scheduler_state_dict_preserved(
        self,
        saved_v2_checkpoint_path,
        checkpoint_manager,
    ):
        """scheduler_state_dict must be present and non-empty."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        assert "scheduler_state_dict" in checkpoint
        assert len(checkpoint["scheduler_state_dict"]) > 0

    def test_optimizer_state_can_be_loaded(
        self,
        saved_v2_checkpoint_path,
        checkpoint_manager,
        minimal_model,
    ):
        """optimizer_state_dict must be loadable into a fresh optimizer."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        fresh_optimizer = torch.optim.Adam(minimal_model.parameters(), lr=0.001)
        # Must not raise
        fresh_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def test_scheduler_state_can_be_loaded(
        self,
        saved_v2_checkpoint_path,
        checkpoint_manager,
        minimal_model,
    ):
        """scheduler_state_dict must be loadable into a fresh scheduler."""
        checkpoint = checkpoint_manager.load(saved_v2_checkpoint_path)
        fresh_optimizer = torch.optim.Adam(minimal_model.parameters(), lr=0.001)
        fresh_scheduler = torch.optim.lr_scheduler.StepLR(fresh_optimizer, step_size=10)
        # Must not raise
        fresh_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 6: BACKWARD COMPATIBILITY — v1.0 FORMAT
# ═══════════════════════════════════════════════════════════════════════════════


class TestV1BackwardCompatibility:
    """
    Verify that v1.0 (legacy) checkpoints load correctly with graceful
    defaults and appropriate warnings.

    v1.0 checkpoints lack hyper_parameters, data_info, and version_info.
    CheckpointManager.load() should provide empty defaults for these fields.
    """

    @pytest.fixture
    def v1_checkpoint_path(self, tmp_working_dir, minimal_model):
        """Create a v1.0-style checkpoint (no metadata)."""
        v1_checkpoint = {
            "epoch": 5,
            "global_step": 200,
            "model_state_dict": minimal_model.state_dict(),
            "best_val_loss": 0.42,
            "metrics_history": {"train_loss": [0.8, 0.6]},
        }
        filepath = tmp_working_dir / "checkpoints" / "legacy_v1.pt"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(v1_checkpoint, filepath)
        return filepath

    def test_v1_checkpoint_loads_without_error(
        self,
        v1_checkpoint_path,
        checkpoint_manager,
    ):
        """v1.0 checkpoint must load without raising exceptions."""
        checkpoint = checkpoint_manager.load(v1_checkpoint_path)
        assert isinstance(checkpoint, dict)

    def test_v1_checkpoint_has_empty_hyper_parameters(
        self,
        v1_checkpoint_path,
        checkpoint_manager,
    ):
        """v1.0 checkpoint must have empty hyper_parameters default."""
        checkpoint = checkpoint_manager.load(v1_checkpoint_path)
        hp = checkpoint.get("hyper_parameters", None)
        assert hp is not None, "hyper_parameters key should be added"
        assert hp == {}

    def test_v1_checkpoint_has_empty_data_info(
        self,
        v1_checkpoint_path,
        checkpoint_manager,
    ):
        """v1.0 checkpoint must have empty data_info default."""
        checkpoint = checkpoint_manager.load(v1_checkpoint_path)
        data_info = checkpoint.get("data_info", None)
        assert data_info is not None, "data_info key should be added"
        assert data_info == {}

    def test_v1_checkpoint_has_version_info_fallback(
        self,
        v1_checkpoint_path,
        checkpoint_manager,
    ):
        """v1.0 checkpoint must have version_info with '1.0' format."""
        checkpoint = checkpoint_manager.load(v1_checkpoint_path)
        version_info = checkpoint.get("version_info", {})
        fmt = version_info.get("checkpoint_format_version", None)
        assert fmt == "1.0", f"Expected format version '1.0' for legacy checkpoint, got '{fmt}'"

    def test_v1_is_v2_returns_false(
        self,
        v1_checkpoint_path,
        checkpoint_manager,
    ):
        """is_v2_checkpoint() must return False for v1.0 checkpoints."""
        checkpoint = checkpoint_manager.load(v1_checkpoint_path)
        assert checkpoint_manager.is_v2_checkpoint(checkpoint) is False

    def test_v1_model_state_dict_still_works(
        self,
        v1_checkpoint_path,
        checkpoint_manager,
        minimal_model,
    ):
        """v1.0 model_state_dict must still load into the model."""
        checkpoint = checkpoint_manager.load(v1_checkpoint_path)
        fresh_model = _MinimalGNNModel()
        # Must not raise
        fresh_model.load_state_dict(checkpoint["model_state_dict"])

    def test_v1_get_model_name_returns_none(
        self,
        v1_checkpoint_path,
        checkpoint_manager,
    ):
        """get_model_name() must return None for v1.0 checkpoints."""
        checkpoint = checkpoint_manager.load(v1_checkpoint_path)
        assert checkpoint_manager.get_model_name(checkpoint) is None

    def test_v1_checkpoint_logs_warning(
        self,
        v1_checkpoint_path,
        checkpoint_manager,
        caplog,
    ):
        """Loading a v1.0 checkpoint must emit a warning."""
        with caplog.at_level(logging.WARNING):
            checkpoint_manager.load(v1_checkpoint_path)
        assert any("v1.0" in msg for msg in caplog.messages), (
            "Expected warning about v1.0 checkpoint format"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 7: PATH RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckpointPathResolution:
    """
    Verify that CheckpointManager resolves paths correctly following the
    Dependency Injection pattern.
    """

    def test_relative_path_resolves_to_working_root(
        self,
        checkpoint_manager,
        minimal_model,
        tmp_working_dir,
    ):
        """Relative paths must resolve against working_root_dir."""
        filepath = "checkpoints/relative_test.pt"
        resolved = checkpoint_manager.save(
            filepath=filepath,
            model=minimal_model,
        )
        expected_dir = tmp_working_dir / "checkpoints"
        assert resolved.parent == expected_dir

    def test_absolute_path_used_as_is(
        self,
        checkpoint_manager,
        minimal_model,
        tmp_path,
    ):
        """Absolute paths must be used as-is."""
        filepath = tmp_path / "absolute_test.pt"
        resolved = checkpoint_manager.save(
            filepath=filepath,
            model=minimal_model,
        )
        assert resolved == filepath.resolve()

    def test_parent_directories_created_automatically(
        self,
        checkpoint_manager,
        minimal_model,
    ):
        """Parent directories must be created automatically on save."""
        filepath = "deep/nested/dir/model.pt"
        resolved = checkpoint_manager.save(
            filepath=filepath,
            model=minimal_model,
        )
        assert resolved.parent.exists()

    def test_get_checkpoint_dir_default(self, checkpoint_manager, tmp_working_dir):
        """get_checkpoint_dir() must return working_root/checkpoints."""
        checkpoint_dir = checkpoint_manager.get_checkpoint_dir()
        assert checkpoint_dir == tmp_working_dir / "checkpoints"
        assert checkpoint_dir.exists()

    def test_get_checkpoint_dir_custom_subdir(self, checkpoint_manager, tmp_working_dir):
        """get_checkpoint_dir(subdir) must return working_root/subdir."""
        checkpoint_dir = checkpoint_manager.get_checkpoint_dir("custom_dir")
        assert checkpoint_dir == tmp_working_dir / "custom_dir"
        assert checkpoint_dir.exists()

    def test_load_finds_checkpoint_in_default_dir(
        self,
        checkpoint_manager,
        minimal_model,
    ):
        """load() must find checkpoints in the default checkpoint dir."""
        # Save to default dir
        checkpoint_dir = checkpoint_manager.get_checkpoint_dir()
        filepath = checkpoint_dir / "findme.pt"
        torch.save({"model_state_dict": minimal_model.state_dict()}, filepath)

        # Load using just the filename — should find in default dir
        loaded = checkpoint_manager.load(
            "findme.pt",
            weights_only=False,
        )
        assert "model_state_dict" in loaded


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 8: ModelLoader.load_from_checkpoint() WITH MOCKED FACTORY
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelLoaderFromCheckpoint:
    """
    Verify ModelLoader.load_from_checkpoint() correctly recreates models
    from v2.0 checkpoints.

    ModelFactory is MOCKED here because this is a regression test for
    the checkpoint format, not an integration test for model recreation.
    The mock verifies that ModelLoader passes the correct arguments
    to the factory.
    """

    @pytest.fixture
    def v2_checkpoint_for_loader(
        self,
        tmp_working_dir,
        minimal_model,
        reference_hyper_parameters,
        reference_data_info,
    ):
        """Save a v2.0 checkpoint suitable for ModelLoader testing."""
        manager = CheckpointManager(working_root_dir=tmp_working_dir)
        filepath = manager.save(
            filepath="checkpoints/loader_test.pt",
            model=minimal_model,
            epoch=5,
            global_step=100,
            hyper_parameters=reference_hyper_parameters,
            data_info=reference_data_info,
        )
        return filepath

    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_from_checkpoint_calls_factory_with_correct_args(
        self,
        mock_get_factory,
        v2_checkpoint_for_loader,
        tmp_working_dir,
        minimal_model,
    ):
        """
        ModelLoader must call create_model_with_info() with the correct
        model_name, hyperparameters, and task_type from the checkpoint.
        """
        # Configure mock factory
        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (
            minimal_model,
            {"task_type": "graph_regression", "uses_edge_features": False},
        )
        mock_get_factory.return_value = mock_factory

        from milia_pipeline.models.post_training.inference.model_loader import (
            ModelLoader,
        )

        model, model_info = ModelLoader.load_from_checkpoint(
            checkpoint_path=v2_checkpoint_for_loader,
            working_root_dir=tmp_working_dir,
        )

        # Verify factory was called with correct arguments
        mock_factory.create_model_with_info.assert_called_once()
        call_kwargs = mock_factory.create_model_with_info.call_args
        assert (
            call_kwargs.kwargs.get("name")
            or call_kwargs[1].get("name", call_kwargs[0][0] if call_kwargs[0] else None) == "GCN"
        )

    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_from_checkpoint_returns_model_in_eval_mode(
        self,
        mock_get_factory,
        v2_checkpoint_for_loader,
        tmp_working_dir,
        minimal_model,
    ):
        """Loaded model must be in eval mode."""
        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (
            minimal_model,
            {"task_type": "graph_regression", "uses_edge_features": False},
        )
        mock_get_factory.return_value = mock_factory

        from milia_pipeline.models.post_training.inference.model_loader import (
            ModelLoader,
        )

        model, _ = ModelLoader.load_from_checkpoint(
            checkpoint_path=v2_checkpoint_for_loader,
            working_root_dir=tmp_working_dir,
        )
        assert not model.training, "Model must be in eval mode after loading"

    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_from_checkpoint_includes_data_info_in_model_info(
        self,
        mock_get_factory,
        v2_checkpoint_for_loader,
        tmp_working_dir,
        minimal_model,
        reference_data_info,
    ):
        """
        model_info returned by load_from_checkpoint() must include data_info
        with structural_features_config (FIX 18).
        """
        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (
            minimal_model,
            {"task_type": "graph_regression", "uses_edge_features": False},
        )
        mock_get_factory.return_value = mock_factory

        from milia_pipeline.models.post_training.inference.model_loader import (
            ModelLoader,
        )

        _, model_info = ModelLoader.load_from_checkpoint(
            checkpoint_path=v2_checkpoint_for_loader,
            working_root_dir=tmp_working_dir,
        )

        assert "data_info" in model_info, "model_info must contain 'data_info' key (FIX 18)"
        assert model_info["data_info"] == reference_data_info
        assert "structural_features_config" in model_info["data_info"]

    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_from_checkpoint_file_not_found(
        self,
        mock_get_factory,
        tmp_working_dir,
    ):
        """
        load_from_checkpoint() must raise FileNotFoundError for
        non-existent checkpoints.
        """
        mock_factory = MagicMock()
        mock_get_factory.return_value = mock_factory

        from milia_pipeline.models.post_training.inference.model_loader import (
            ModelLoader,
        )

        with pytest.raises(FileNotFoundError):
            ModelLoader.load_from_checkpoint(
                checkpoint_path="nonexistent_model.pt",
                working_root_dir=tmp_working_dir,
            )

    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_from_checkpoint_v1_raises_without_overrides(
        self,
        mock_get_factory,
        tmp_working_dir,
        minimal_model,
    ):
        """
        Loading a v1.0 checkpoint without model_name/task_type overrides
        must raise ValueError (since these fields are required but missing).
        """
        mock_factory = MagicMock()
        mock_get_factory.return_value = mock_factory

        # Create v1.0 checkpoint (no hyper_parameters)
        _manager = CheckpointManager(working_root_dir=tmp_working_dir)
        v1_path = tmp_working_dir / "checkpoints" / "v1_model.pt"
        v1_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": 3,
                "model_state_dict": minimal_model.state_dict(),
                "best_val_loss": 0.5,
            },
            v1_path,
        )

        from milia_pipeline.models.post_training.inference.model_loader import (
            ModelLoader,
        )

        with pytest.raises(ValueError, match="model_name is required"):
            ModelLoader.load_from_checkpoint(
                checkpoint_path=v1_path,
                working_root_dir=tmp_working_dir,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 9: CHECKPOINT EXTRA DATA
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckpointExtraData:
    """
    Verify that extra data passed via **extra_data to save() is preserved.
    """

    def test_extra_data_preserved(self, checkpoint_manager, minimal_model, tmp_working_dir):
        """Custom extra_data keys must survive round-trip."""
        filepath = checkpoint_manager.save(
            filepath="checkpoints/extra_data_test.pt",
            model=minimal_model,
            custom_metric=42.0,
            experiment_name="regression_test",
        )
        checkpoint = checkpoint_manager.load(filepath)
        assert checkpoint["custom_metric"] == 42.0
        assert checkpoint["experiment_name"] == "regression_test"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 10: CHECKPOINT SAVE WITHOUT OPTIONAL FIELDS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckpointMinimalSave:
    """
    Verify that saving a checkpoint with only required fields works.
    """

    def test_minimal_save_and_load(self, checkpoint_manager, minimal_model):
        """Checkpoint with only model (no optimizer/scheduler) must work."""
        filepath = checkpoint_manager.save(
            filepath="checkpoints/minimal.pt",
            model=minimal_model,
        )
        checkpoint = checkpoint_manager.load(filepath)

        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" not in checkpoint
        assert "scheduler_state_dict" not in checkpoint
        assert checkpoint["epoch"] == 0
        assert checkpoint["global_step"] == 0
        assert checkpoint["best_val_loss"] == float("inf")
        assert checkpoint["hyper_parameters"] == {}
        assert checkpoint["data_info"] == {}

    def test_save_without_optimizer(self, checkpoint_manager, minimal_model):
        """Saving without optimizer must not include optimizer_state_dict."""
        filepath = checkpoint_manager.save(
            filepath="checkpoints/no_opt.pt",
            model=minimal_model,
            epoch=3,
        )
        checkpoint = checkpoint_manager.load(filepath)
        assert "optimizer_state_dict" not in checkpoint

    def test_save_without_scheduler(self, checkpoint_manager, minimal_model):
        """Saving without scheduler must not include scheduler_state_dict."""
        filepath = checkpoint_manager.save(
            filepath="checkpoints/no_sched.pt",
            model=minimal_model,
            epoch=3,
        )
        checkpoint = checkpoint_manager.load(filepath)
        assert "scheduler_state_dict" not in checkpoint


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 11: create_version_info() STATIC METHOD
# ═══════════════════════════════════════════════════════════════════════════════


class TestCreateVersionInfo:
    """
    Verify that CheckpointManager.create_version_info() produces
    well-formed metadata.
    """

    def test_create_version_info_returns_dict(self):
        """create_version_info() must return a dict."""
        info = CheckpointManager.create_version_info()
        assert isinstance(info, dict)

    def test_create_version_info_contains_all_keys(self):
        """create_version_info() must contain all documented keys."""
        info = CheckpointManager.create_version_info()
        required = {
            "milia_version",
            "checkpoint_format_version",
            "pytorch_version",
            "pyg_version",
            "created_at",
        }
        assert required.issubset(info.keys())

    def test_create_version_info_format_version_matches_constant(self):
        """create_version_info() format version must match module constant."""
        info = CheckpointManager.create_version_info()
        assert info["checkpoint_format_version"] == CHECKPOINT_FORMAT_VERSION

    def test_create_version_info_created_at_is_recent(self):
        """created_at must be a recent timestamp (within 60 seconds)."""
        info = CheckpointManager.create_version_info()
        created = datetime.fromisoformat(info["created_at"])
        delta = (datetime.now() - created).total_seconds()
        assert abs(delta) < 60, f"Timestamp is {delta}s old — too stale"


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
