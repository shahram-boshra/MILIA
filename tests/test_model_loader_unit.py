#!/usr/bin/env python3
"""
Extended Production-Ready Unit Test Suite for model_loader.py Module

Comprehensive test coverage for the ModelLoader class and convenience functions.
Tests cover:
- ModelLoader class initialization
- load_from_checkpoint class method
- _load internal method
- get_checkpoint_info class method
- Convenience functions (load_model, load_model_only)
- Error handling and edge cases
- Device handling
- State dict loading (strict/non-strict modes)
- Wrapped/unwrapped model handling
- v1.0/v2.0 checkpoint compatibility
- Thread safety considerations

Author: MILIA Team
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import logging
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# Import the module under test
from milia_pipeline.models.post_training.inference.model_loader import (
    ModelLoader,
    load_model,
    load_model_only,
)
from milia_pipeline.models.post_training.inference.model_loader import (
    logger as model_loader_logger,
)

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_checkpoint_manager():
    """Create a mock CheckpointManager."""
    mock_cm = MagicMock()
    mock_cm.load = MagicMock()
    mock_cm.is_v2_checkpoint = MagicMock(return_value=True)
    return mock_cm


@pytest.fixture
def mock_model_factory():
    """Create a mock ModelFactory."""
    mock_factory = MagicMock()
    mock_model = MagicMock(spec=nn.Module)
    mock_model.eval = MagicMock(return_value=mock_model)
    mock_model.load_state_dict = MagicMock()
    mock_model_info = {
        "uses_edge_features": False,
        "task_type": "graph_regression",
        "out_channels": 1,
    }
    mock_factory.create_model_with_info = MagicMock(return_value=(mock_model, mock_model_info))
    return mock_factory


@pytest.fixture
def sample_state_dict():
    """Create a sample state dict for testing."""
    return {
        "layer1.weight": torch.randn(64, 16),
        "layer1.bias": torch.randn(64),
        "layer2.weight": torch.randn(1, 64),
        "layer2.bias": torch.randn(1),
    }


@pytest.fixture
def v2_checkpoint(sample_state_dict):
    """Create a v2.0 checkpoint dictionary."""
    return {
        "epoch": 100,
        "global_step": 5000,
        "model_state_dict": sample_state_dict,
        "best_val_loss": 0.123,
        "metrics_history": {"train_loss": [0.5, 0.3, 0.2]},
        "hyper_parameters": {
            "model_name": "GCN",
            "task_type": "graph_regression",
            "hyperparameters": {
                "hidden_channels": 64,
                "num_layers": 3,
                "out_channels": 1,
            },
            "model_info": {
                "uses_edge_features": False,
                "requires_edge_features": False,
            },
            "wrapper_info": {
                "wrapper_type": "GraphLevelModelWrapper",
                "pooling_method": "mean",
            },
            "target_selection_config": None,
        },
        "data_info": {
            "dataset_name": "test_dataset",
            "num_samples": 1000,
        },
        "version_info": {
            "checkpoint_format_version": "2.0",
            "milia_version": "1.0.0",
            "pytorch_version": torch.__version__,
        },
    }


@pytest.fixture
def v1_checkpoint(sample_state_dict):
    """Create a v1.0 (legacy) checkpoint dictionary."""
    return {
        "epoch": 50,
        "model_state_dict": sample_state_dict,
        "best_val_loss": 0.456,
        # v1.0 has minimal or missing hyper_parameters
        "hyper_parameters": {},
        "version_info": {
            "checkpoint_format_version": "1.0",
        },
    }


@pytest.fixture
def v2_checkpoint_with_target_selection(sample_state_dict):
    """Create a v2.0 checkpoint with target selection config."""
    return {
        "epoch": 100,
        "model_state_dict": sample_state_dict,
        "best_val_loss": 0.089,
        "hyper_parameters": {
            "model_name": "GAT",
            "task_type": "graph_regression",
            "hyperparameters": {
                "hidden_channels": 128,
                "num_layers": 4,
                "out_channels": 3,
            },
            "model_info": {
                "uses_edge_features": True,
            },
            "wrapper_info": {},
            "target_selection_config": {
                "indices": [0, 2, 5],
                "names": ["energy", "force", "charge"],
            },
        },
        "version_info": {
            "checkpoint_format_version": "2.0",
        },
    }


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoint files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def create_configured_checkpoint_manager_mock(checkpoint, tmp_path):
    """
    Helper function to create a properly configured CheckpointManager mock.

    This centralizes the mock configuration to ensure consistency across tests
    and proper handling of the _resolve_checkpoint_path method.

    Args:
        checkpoint: The checkpoint dict to return from load()
        tmp_path: The temporary path fixture for creating fake files

    Returns:
        Configured MagicMock instance for CheckpointManager
    """
    mock_cm_instance = MagicMock()
    mock_cm_instance.load.return_value = checkpoint
    mock_cm_instance._resolve_checkpoint_path.return_value = tmp_path / "checkpoint.pt"
    mock_cm_instance.get_checkpoint_dir.return_value = tmp_path / "checkpoints"
    mock_cm_instance.is_v2_checkpoint.return_value = (
        checkpoint.get("version_info", {}).get("checkpoint_format_version", "1.0") == "2.0"
    )
    return mock_cm_instance


@pytest.fixture
def configured_cm_mock(v2_checkpoint, tmp_path):
    """Fixture providing a configured CheckpointManager mock with v2 checkpoint."""
    return create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)


@pytest.fixture
def simple_model():
    """Create a simple PyTorch model for testing."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 1)

        def forward(self, x):
            return self.linear(x)

    return SimpleModel()


@pytest.fixture
def wrapped_model(simple_model):
    """Create a wrapped model (has .model attribute) for testing."""

    class WrappedModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x)

    return WrappedModel(simple_model)


# =============================================================================
# MODEL LOADER CLASS TESTS - INITIALIZATION
# =============================================================================


class TestModelLoaderInitialization:
    """Test ModelLoader class initialization."""

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_init_creates_checkpoint_manager(self, mock_get_factory, mock_cm_class, tmp_path):
        """Test that __init__ creates a CheckpointManager instance."""
        mock_cm_instance = MagicMock()
        mock_cm_class.return_value = mock_cm_instance
        mock_get_factory.return_value = MagicMock()

        loader = ModelLoader(working_root_dir=tmp_path)

        mock_cm_class.assert_called_once_with(working_root_dir=tmp_path.expanduser().resolve())
        assert loader.checkpoint_manager == mock_cm_instance

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_init_gets_model_factory(self, mock_get_factory, mock_cm_class, tmp_path):
        """Test that __init__ calls get_factory to get ModelFactory instance."""
        mock_factory = MagicMock()
        mock_get_factory.return_value = mock_factory

        loader = ModelLoader(working_root_dir=tmp_path)

        mock_get_factory.assert_called_once()
        assert loader.model_factory == mock_factory

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_init_attributes_are_set(self, mock_get_factory, mock_cm_class, tmp_path):
        """Test that both checkpoint_manager and model_factory attributes are set."""
        mock_cm_instance = MagicMock()
        mock_cm_class.return_value = mock_cm_instance
        mock_factory = MagicMock()
        mock_get_factory.return_value = mock_factory

        loader = ModelLoader(working_root_dir=tmp_path)

        assert hasattr(loader, "checkpoint_manager")
        assert hasattr(loader, "model_factory")
        assert loader.checkpoint_manager is mock_cm_instance
        assert loader.model_factory is mock_factory

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_init_stores_working_root_dir(self, mock_get_factory, mock_cm_class, tmp_path):
        """Test that __init__ stores the working_root_dir as resolved Path."""
        mock_cm_class.return_value = MagicMock()
        mock_get_factory.return_value = MagicMock()

        loader = ModelLoader(working_root_dir=tmp_path)

        assert hasattr(loader, "_working_root_dir")
        assert loader._working_root_dir == tmp_path.expanduser().resolve()

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_init_expands_user_path(self, mock_get_factory, mock_cm_class, tmp_path):
        """Test that __init__ expands ~ in working_root_dir path."""
        mock_cm_class.return_value = MagicMock()
        mock_get_factory.return_value = MagicMock()

        # Use a path with ~ to verify expansion
        user_path = Path("~/test_dir")
        loader = ModelLoader(working_root_dir=user_path)

        # Verify the path was expanded (no ~ in result)
        assert "~" not in str(loader._working_root_dir)


# =============================================================================
# MODEL LOADER CLASS TESTS - load_from_checkpoint CLASS METHOD
# =============================================================================


class TestLoadFromCheckpoint:
    """Test ModelLoader.load_from_checkpoint class method."""

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_from_checkpoint_creates_instance_and_calls_load(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test that load_from_checkpoint creates ModelLoader and calls _load."""
        # Setup mocks
        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = v2_checkpoint
        mock_cm_instance._resolve_checkpoint_path.return_value = tmp_path / "test_checkpoint.pt"
        mock_cm_instance.get_checkpoint_dir.return_value = tmp_path / "checkpoints"
        mock_cm_class.return_value = mock_cm_instance

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model
        mock_model.load_state_dict = MagicMock()
        mock_model_info = {"uses_edge_features": False}

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, mock_model_info)
        mock_get_factory.return_value = mock_factory

        # Create a fake checkpoint file to satisfy existence check
        checkpoint_file = tmp_path / "test_checkpoint.pt"
        checkpoint_file.touch()

        # Call the method
        model, model_info = ModelLoader.load_from_checkpoint(
            checkpoint_path="test_checkpoint.pt",
            working_root_dir=tmp_path,
            device=torch.device("cpu"),
        )

        # Verify model is returned and in eval mode
        assert model is mock_model
        mock_model.eval.assert_called_once()

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_from_checkpoint_passes_all_parameters(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test that all parameters are correctly passed to _load."""
        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = v2_checkpoint
        mock_cm_instance._resolve_checkpoint_path.return_value = tmp_path / "test.pt"
        mock_cm_instance.get_checkpoint_dir.return_value = tmp_path / "checkpoints"
        mock_cm_class.return_value = mock_cm_instance

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model
        mock_model_info = {}

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, mock_model_info)
        mock_get_factory.return_value = mock_factory

        # Create a fake checkpoint file
        checkpoint_file = tmp_path / "test.pt"
        checkpoint_file.touch()

        override_hyperparams = {"hidden_channels": 128}

        model, model_info = ModelLoader.load_from_checkpoint(
            checkpoint_path="test.pt",
            working_root_dir=tmp_path,
            device=torch.device("cpu"),
            model_name="GAT",
            hyperparameters=override_hyperparams,
            task_type="node_classification",
            strict=False,
        )

        # Verify create_model_with_info was called with override parameters
        mock_factory.create_model_with_info.assert_called_once()
        call_kwargs = mock_factory.create_model_with_info.call_args
        assert call_kwargs[1]["name"] == "GAT"
        assert call_kwargs[1]["task_type"] == "node_classification"
        assert call_kwargs[1]["hyperparameters"] == override_hyperparams

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_from_checkpoint_with_path_object(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test load_from_checkpoint accepts Path objects."""
        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = v2_checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        mock_cm_instance._resolve_checkpoint_path.return_value = checkpoint_path
        mock_cm_instance.get_checkpoint_dir.return_value = tmp_path / "checkpoints"
        mock_cm_class.return_value = mock_cm_instance

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        # Create a fake checkpoint file
        checkpoint_path.touch()

        model, _ = ModelLoader.load_from_checkpoint(
            checkpoint_path=checkpoint_path, working_root_dir=tmp_path
        )

        # Verify load was called with resolved Path
        mock_cm_instance.load.assert_called_once()

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_from_checkpoint_with_string_path(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test load_from_checkpoint accepts string paths."""
        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = v2_checkpoint
        mock_cm_instance._resolve_checkpoint_path.return_value = tmp_path / "checkpoint.pt"
        mock_cm_instance.get_checkpoint_dir.return_value = tmp_path / "checkpoints"
        mock_cm_class.return_value = mock_cm_instance

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        # Create a fake checkpoint file
        checkpoint_file = tmp_path / "checkpoint.pt"
        checkpoint_file.touch()

        checkpoint_path = "/path/to/checkpoint.pt"

        model, _ = ModelLoader.load_from_checkpoint(
            checkpoint_path=checkpoint_path, working_root_dir=tmp_path
        )

        # String should be accepted
        mock_cm_instance.load.assert_called_once()


# =============================================================================
# MODEL LOADER CLASS TESTS - _load INTERNAL METHOD
# =============================================================================


class TestLoadInternalMethod:
    """Test ModelLoader._load internal method."""

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_auto_detects_device_cuda(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test _load auto-detects CUDA device when available."""
        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        with patch(
            "milia_pipeline.models.post_training.inference.model_loader.torch.cuda.is_available",
            return_value=True,
        ):
            model, _ = ModelLoader.load_from_checkpoint(
                "test.pt", working_root_dir=tmp_path, device=None
            )

        # When CUDA is available and no device specified, should use CUDA
        call_kwargs = mock_factory.create_model_with_info.call_args[1]
        assert call_kwargs["device"] == torch.device("cuda")

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_auto_detects_device_cpu(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test _load auto-detects CPU device when CUDA unavailable."""
        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        with patch(
            "milia_pipeline.models.post_training.inference.model_loader.torch.cuda.is_available",
            return_value=False,
        ):
            model, _ = ModelLoader.load_from_checkpoint(
                "test.pt", working_root_dir=tmp_path, device=None
            )

        call_kwargs = mock_factory.create_model_with_info.call_args[1]
        assert call_kwargs["device"] == torch.device("cpu")

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_uses_specified_device(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test _load uses the specified device."""
        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        specified_device = torch.device("cpu")
        model, _ = ModelLoader.load_from_checkpoint(
            "test.pt", working_root_dir=tmp_path, device=specified_device
        )

        # Should use specified device
        mock_cm_instance.load.assert_called_once()
        call_kwargs = mock_cm_instance.load.call_args[1]
        assert call_kwargs["map_location"] == specified_device

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_extracts_hyper_parameters_from_checkpoint(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test _load correctly extracts hyper_parameters from checkpoint."""
        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        model, _ = ModelLoader.load_from_checkpoint("test.pt", working_root_dir=tmp_path)

        # Verify factory was called with checkpoint's hyperparameters
        call_kwargs = mock_factory.create_model_with_info.call_args[1]
        assert call_kwargs["name"] == "GCN"
        assert call_kwargs["task_type"] == "graph_regression"
        assert (
            call_kwargs["hyperparameters"] == v2_checkpoint["hyper_parameters"]["hyperparameters"]
        )

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_extracts_target_selection_config(
        self, mock_get_factory, mock_cm_class, v2_checkpoint_with_target_selection, tmp_path
    ):
        """Test _load correctly extracts target_selection_config from checkpoint."""
        mock_cm_instance = create_configured_checkpoint_manager_mock(
            v2_checkpoint_with_target_selection, tmp_path
        )
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        model, _ = ModelLoader.load_from_checkpoint("test.pt", working_root_dir=tmp_path)

        call_kwargs = mock_factory.create_model_with_info.call_args[1]
        expected_ts_config = v2_checkpoint_with_target_selection["hyper_parameters"][
            "target_selection_config"
        ]
        assert call_kwargs["target_selection_config"] == expected_ts_config

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_overrides_model_name(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test _load uses override model_name when provided."""
        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        # Override model_name (checkpoint has 'GCN')
        model, _ = ModelLoader.load_from_checkpoint(
            "test.pt", working_root_dir=tmp_path, model_name="GAT"
        )

        call_kwargs = mock_factory.create_model_with_info.call_args[1]
        assert call_kwargs["name"] == "GAT"

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_overrides_task_type(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test _load uses override task_type when provided."""
        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        # Override task_type (checkpoint has 'graph_regression')
        model, _ = ModelLoader.load_from_checkpoint(
            "test.pt", working_root_dir=tmp_path, task_type="node_classification"
        )

        call_kwargs = mock_factory.create_model_with_info.call_args[1]
        assert call_kwargs["task_type"] == "node_classification"

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_overrides_hyperparameters(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test _load uses override hyperparameters when provided."""
        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        override_hyperparams = {"hidden_channels": 256, "num_layers": 5}
        model, _ = ModelLoader.load_from_checkpoint(
            "test.pt", working_root_dir=tmp_path, hyperparameters=override_hyperparams
        )

        call_kwargs = mock_factory.create_model_with_info.call_args[1]
        assert call_kwargs["hyperparameters"] == override_hyperparams

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_sets_model_to_eval_mode(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test _load sets model to eval mode."""
        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        model, _ = ModelLoader.load_from_checkpoint("test.pt", working_root_dir=tmp_path)

        mock_model.eval.assert_called_once()

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_loads_state_dict_strict_mode(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test _load loads state_dict with strict=True by default."""
        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model
        mock_model.load_state_dict = MagicMock()

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        model, _ = ModelLoader.load_from_checkpoint(
            "test.pt", working_root_dir=tmp_path, strict=True
        )

        mock_model.load_state_dict.assert_called_once()
        call_kwargs = mock_model.load_state_dict.call_args[1]
        assert call_kwargs["strict"] is True

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_loads_state_dict_non_strict_mode(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test _load loads state_dict with strict=False when specified."""
        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model
        mock_model.load_state_dict = MagicMock()

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        model, _ = ModelLoader.load_from_checkpoint(
            "test.pt", working_root_dir=tmp_path, strict=False
        )

        call_kwargs = mock_model.load_state_dict.call_args[1]
        assert call_kwargs["strict"] is False


# =============================================================================
# MODEL LOADER CLASS TESTS - WRAPPED MODEL HANDLING
# =============================================================================


class TestWrappedModelHandling:
    """Test ModelLoader handling of wrapped models."""

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_state_dict_into_wrapped_model_on_mismatch(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test _load attempts to load state_dict into inner model on mismatch."""
        # Modify checkpoint to have model. prefix (simulating wrapped model checkpoint)
        # This triggers the fallback logic in _load
        wrapped_state_dict = {f"model.{k}": v for k, v in v2_checkpoint["model_state_dict"].items()}
        v2_checkpoint["model_state_dict"] = wrapped_state_dict

        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        # Create a mock wrapped model where outer load fails but inner succeeds
        inner_model = MagicMock(spec=nn.Module)
        inner_model.load_state_dict = MagicMock()

        outer_model = MagicMock(spec=nn.Module)
        outer_model.model = inner_model  # Wrapped model has .model attribute
        outer_model.eval.return_value = outer_model
        outer_model.load_state_dict = MagicMock(side_effect=RuntimeError("Keys mismatch"))
        # state_dict() must return dict without model. prefix so fallback triggers
        outer_model.state_dict.return_value = {
            "layer1.weight": MagicMock(),
            "layer1.bias": MagicMock(),
        }

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (outer_model, {})
        mock_get_factory.return_value = mock_factory

        model, _ = ModelLoader.load_from_checkpoint("test.pt", working_root_dir=tmp_path)

        # Outer load_state_dict should have been called and failed
        outer_model.load_state_dict.assert_called_once()
        # Inner model's load_state_dict should have been called as fallback
        inner_model.load_state_dict.assert_called_once()

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_raises_if_no_inner_model_and_mismatch(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test _load raises RuntimeError if state_dict mismatch and no inner model."""
        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        # Model without .model attribute
        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model
        mock_model.load_state_dict = MagicMock(side_effect=RuntimeError("Keys mismatch"))
        mock_model.state_dict.return_value = {"layer1.weight": MagicMock()}
        # Ensure no .model attribute
        del mock_model.model

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        with pytest.raises(RuntimeError, match="Keys mismatch"):
            ModelLoader.load_from_checkpoint("test.pt", working_root_dir=tmp_path)

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_inner_model_raises_if_also_fails(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test _load raises RuntimeError if both outer and inner model load fail."""
        # Modify checkpoint to have model. prefix (required for fallback logic)
        wrapped_state_dict = {f"model.{k}": v for k, v in v2_checkpoint["model_state_dict"].items()}
        v2_checkpoint["model_state_dict"] = wrapped_state_dict

        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        inner_model = MagicMock(spec=nn.Module)
        inner_model.load_state_dict = MagicMock(side_effect=RuntimeError("Inner also failed"))

        outer_model = MagicMock(spec=nn.Module)
        outer_model.model = inner_model
        outer_model.eval.return_value = outer_model
        outer_model.load_state_dict = MagicMock(side_effect=RuntimeError("Outer failed"))
        # state_dict() must return dict without model. prefix so fallback triggers
        outer_model.state_dict.return_value = {
            "layer1.weight": MagicMock(),
            "layer1.bias": MagicMock(),
        }

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (outer_model, {})
        mock_get_factory.return_value = mock_factory

        with pytest.raises(RuntimeError, match="Inner also failed"):
            ModelLoader.load_from_checkpoint("test.pt", working_root_dir=tmp_path)


# =============================================================================
# MODEL LOADER CLASS TESTS - MODEL INFO MERGING
# =============================================================================


class TestModelInfoMerging:
    """Test ModelLoader model_info merging behavior."""

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_merges_model_info(self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path):
        """Test _load merges saved model_info with recreated model_info."""
        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        recreated_model_info = {
            "uses_edge_features": True,  # Will be overridden by saved
            "new_field": "from_recreated",  # New field from recreated
        }

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, recreated_model_info)
        mock_get_factory.return_value = mock_factory

        model, model_info = ModelLoader.load_from_checkpoint("test.pt", working_root_dir=tmp_path)

        # Saved model_info should override recreated
        assert model_info["uses_edge_features"] is False  # From saved checkpoint
        # New fields from recreated should be preserved
        assert model_info["new_field"] == "from_recreated"

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_saved_model_info_takes_precedence(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test saved model_info values take precedence over recreated."""
        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        # Both saved and recreated have 'requires_edge_features'
        recreated_model_info = {
            "requires_edge_features": True,  # Different from saved
        }

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, recreated_model_info)
        mock_get_factory.return_value = mock_factory

        model, model_info = ModelLoader.load_from_checkpoint("test.pt", working_root_dir=tmp_path)

        # Saved value (False) should override recreated (True)
        assert model_info["requires_edge_features"] is False


# =============================================================================
# MODEL LOADER CLASS TESTS - VALIDATION ERRORS
# =============================================================================


class TestValidationErrors:
    """Test ModelLoader validation and error handling."""

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_raises_value_error_if_model_name_missing(
        self, mock_get_factory, mock_cm_class, v1_checkpoint, tmp_path
    ):
        """Test _load raises ValueError if model_name not in checkpoint and not provided."""
        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = v1_checkpoint  # v1.0 has empty hyper_parameters
        mock_cm_class.return_value = mock_cm_instance

        mock_factory = MagicMock()
        mock_get_factory.return_value = mock_factory

        with pytest.raises(ValueError) as exc_info:
            ModelLoader.load_from_checkpoint("test.pt", working_root_dir=tmp_path)

        assert "model_name is required" in str(exc_info.value)
        assert "v1.0 checkpoint" in str(exc_info.value)

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_raises_value_error_if_task_type_missing(
        self, mock_get_factory, mock_cm_class, v1_checkpoint, tmp_path
    ):
        """Test _load raises ValueError if task_type not in checkpoint and not provided."""
        # Add model_name but not task_type
        v1_checkpoint["hyper_parameters"] = {"model_name": "GCN"}

        mock_cm_instance = create_configured_checkpoint_manager_mock(v1_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_factory = MagicMock()
        mock_get_factory.return_value = mock_factory

        with pytest.raises(ValueError) as exc_info:
            ModelLoader.load_from_checkpoint("test.pt", working_root_dir=tmp_path)

        assert "task_type is required" in str(exc_info.value)
        assert "v1.0 checkpoint" in str(exc_info.value)

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_v1_checkpoint_with_manual_config_succeeds(
        self, mock_get_factory, mock_cm_class, v1_checkpoint, tmp_path
    ):
        """Test _load succeeds for v1.0 checkpoint when config is provided manually."""
        mock_cm_instance = create_configured_checkpoint_manager_mock(v1_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        # Provide required parameters manually for v1.0 checkpoint
        model, _ = ModelLoader.load_from_checkpoint(
            "test.pt",
            working_root_dir=tmp_path,
            model_name="GCN",
            hyperparameters={"hidden_channels": 64},
            task_type="graph_regression",
        )

        # Should succeed
        assert model is mock_model


# =============================================================================
# MODEL LOADER CLASS TESTS - get_checkpoint_info CLASS METHOD
# =============================================================================


class TestGetCheckpointInfo:
    """Test ModelLoader.get_checkpoint_info class method."""

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_get_checkpoint_info_returns_dict(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test get_checkpoint_info returns a dictionary."""
        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = v2_checkpoint
        mock_cm_instance.is_v2_checkpoint.return_value = True
        mock_cm_instance._resolve_checkpoint_path.return_value = tmp_path / "checkpoint.pt"
        mock_cm_class.return_value = mock_cm_instance
        mock_get_factory.return_value = MagicMock()

        info = ModelLoader.get_checkpoint_info("test.pt", working_root_dir=tmp_path)

        assert isinstance(info, dict)

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_get_checkpoint_info_extracts_format_version(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test get_checkpoint_info extracts format_version."""
        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = v2_checkpoint
        mock_cm_instance.is_v2_checkpoint.return_value = True
        mock_cm_instance._resolve_checkpoint_path.return_value = tmp_path / "checkpoint.pt"
        mock_cm_class.return_value = mock_cm_instance
        mock_get_factory.return_value = MagicMock()

        info = ModelLoader.get_checkpoint_info("test.pt", working_root_dir=tmp_path)

        assert info["format_version"] == "2.0"

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_get_checkpoint_info_extracts_is_v2(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test get_checkpoint_info extracts is_v2 status."""
        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = v2_checkpoint
        mock_cm_instance.is_v2_checkpoint.return_value = True
        mock_cm_instance._resolve_checkpoint_path.return_value = tmp_path / "checkpoint.pt"
        mock_cm_class.return_value = mock_cm_instance
        mock_get_factory.return_value = MagicMock()

        info = ModelLoader.get_checkpoint_info("test.pt", working_root_dir=tmp_path)

        assert info["is_v2"] is True

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_get_checkpoint_info_extracts_model_name(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test get_checkpoint_info extracts model_name."""
        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = v2_checkpoint
        mock_cm_instance.is_v2_checkpoint.return_value = True
        mock_cm_instance._resolve_checkpoint_path.return_value = tmp_path / "checkpoint.pt"
        mock_cm_class.return_value = mock_cm_instance
        mock_get_factory.return_value = MagicMock()

        info = ModelLoader.get_checkpoint_info("test.pt", working_root_dir=tmp_path)

        assert info["model_name"] == "GCN"

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_get_checkpoint_info_extracts_task_type(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test get_checkpoint_info extracts task_type."""
        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = v2_checkpoint
        mock_cm_instance.is_v2_checkpoint.return_value = True
        mock_cm_instance._resolve_checkpoint_path.return_value = tmp_path / "checkpoint.pt"
        mock_cm_class.return_value = mock_cm_instance
        mock_get_factory.return_value = MagicMock()

        info = ModelLoader.get_checkpoint_info("test.pt", working_root_dir=tmp_path)

        assert info["task_type"] == "graph_regression"

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_get_checkpoint_info_extracts_epoch(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test get_checkpoint_info extracts epoch."""
        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = v2_checkpoint
        mock_cm_instance.is_v2_checkpoint.return_value = True
        mock_cm_instance._resolve_checkpoint_path.return_value = tmp_path / "checkpoint.pt"
        mock_cm_class.return_value = mock_cm_instance
        mock_get_factory.return_value = MagicMock()

        info = ModelLoader.get_checkpoint_info("test.pt", working_root_dir=tmp_path)

        assert info["epoch"] == 100

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_get_checkpoint_info_extracts_best_val_loss(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test get_checkpoint_info extracts best_val_loss."""
        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = v2_checkpoint
        mock_cm_instance.is_v2_checkpoint.return_value = True
        mock_cm_instance._resolve_checkpoint_path.return_value = tmp_path / "checkpoint.pt"
        mock_cm_class.return_value = mock_cm_instance
        mock_get_factory.return_value = MagicMock()

        info = ModelLoader.get_checkpoint_info("test.pt", working_root_dir=tmp_path)

        assert info["best_val_loss"] == 0.123

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_get_checkpoint_info_extracts_hyper_parameters(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test get_checkpoint_info includes full hyper_parameters dict."""
        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = v2_checkpoint
        mock_cm_instance.is_v2_checkpoint.return_value = True
        mock_cm_instance._resolve_checkpoint_path.return_value = tmp_path / "checkpoint.pt"
        mock_cm_class.return_value = mock_cm_instance
        mock_get_factory.return_value = MagicMock()

        info = ModelLoader.get_checkpoint_info("test.pt", working_root_dir=tmp_path)

        assert info["hyper_parameters"] == v2_checkpoint["hyper_parameters"]

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_get_checkpoint_info_extracts_data_info(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test get_checkpoint_info includes data_info."""
        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = v2_checkpoint
        mock_cm_instance.is_v2_checkpoint.return_value = True
        mock_cm_instance._resolve_checkpoint_path.return_value = tmp_path / "checkpoint.pt"
        mock_cm_class.return_value = mock_cm_instance
        mock_get_factory.return_value = MagicMock()

        info = ModelLoader.get_checkpoint_info("test.pt", working_root_dir=tmp_path)

        assert info["data_info"] == v2_checkpoint["data_info"]

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_get_checkpoint_info_extracts_version_info(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test get_checkpoint_info includes version_info."""
        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = v2_checkpoint
        mock_cm_instance.is_v2_checkpoint.return_value = True
        mock_cm_instance._resolve_checkpoint_path.return_value = tmp_path / "checkpoint.pt"
        mock_cm_class.return_value = mock_cm_instance
        mock_get_factory.return_value = MagicMock()

        info = ModelLoader.get_checkpoint_info("test.pt", working_root_dir=tmp_path)

        assert info["version_info"] == v2_checkpoint["version_info"]

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_get_checkpoint_info_has_wrapper_info_flag(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test get_checkpoint_info includes has_wrapper_info flag."""
        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = v2_checkpoint
        mock_cm_instance.is_v2_checkpoint.return_value = True
        mock_cm_instance._resolve_checkpoint_path.return_value = tmp_path / "checkpoint.pt"
        mock_cm_class.return_value = mock_cm_instance
        mock_get_factory.return_value = MagicMock()

        info = ModelLoader.get_checkpoint_info("test.pt", working_root_dir=tmp_path)

        # v2_checkpoint has wrapper_info
        assert info["has_wrapper_info"] is True

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_get_checkpoint_info_has_target_selection_flag(
        self, mock_get_factory, mock_cm_class, v2_checkpoint_with_target_selection, tmp_path
    ):
        """Test get_checkpoint_info includes has_target_selection flag."""
        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = v2_checkpoint_with_target_selection
        mock_cm_instance.is_v2_checkpoint.return_value = True
        mock_cm_instance._resolve_checkpoint_path.return_value = tmp_path / "checkpoint.pt"
        mock_cm_class.return_value = mock_cm_instance
        mock_get_factory.return_value = MagicMock()

        info = ModelLoader.get_checkpoint_info("test.pt", working_root_dir=tmp_path)

        assert info["has_target_selection"] is True

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_get_checkpoint_info_defaults_for_missing_fields(
        self, mock_get_factory, mock_cm_class, v1_checkpoint, tmp_path
    ):
        """Test get_checkpoint_info provides defaults for missing fields."""
        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = v1_checkpoint
        mock_cm_instance.is_v2_checkpoint.return_value = False
        mock_cm_instance._resolve_checkpoint_path.return_value = tmp_path / "checkpoint.pt"
        mock_cm_class.return_value = mock_cm_instance
        mock_get_factory.return_value = MagicMock()

        info = ModelLoader.get_checkpoint_info("test.pt", working_root_dir=tmp_path)

        assert info["format_version"] == "1.0"
        assert info["is_v2"] is False
        assert info["model_name"] == "UNKNOWN"
        assert info["task_type"] == "UNKNOWN"
        assert info["epoch"] == 50
        assert info["has_wrapper_info"] is False
        assert info["has_target_selection"] is False

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_get_checkpoint_info_accepts_path_object(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test get_checkpoint_info accepts Path object."""
        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = v2_checkpoint
        mock_cm_instance.is_v2_checkpoint.return_value = True
        mock_cm_instance._resolve_checkpoint_path.return_value = tmp_path / "checkpoint.pt"
        mock_cm_class.return_value = mock_cm_instance
        mock_get_factory.return_value = MagicMock()

        checkpoint_path = Path("/path/to/checkpoint.pt")

        info = ModelLoader.get_checkpoint_info(checkpoint_path, working_root_dir=tmp_path)

        assert isinstance(info, dict)
        mock_cm_instance.load.assert_called_once()


# =============================================================================
# CONVENIENCE FUNCTION TESTS - load_model
# =============================================================================


class TestLoadModelFunction:
    """Test load_model convenience function."""

    @patch(
        "milia_pipeline.models.post_training.inference.model_loader.ModelLoader.load_from_checkpoint"
    )
    def test_load_model_calls_load_from_checkpoint(self, mock_load_from_checkpoint, tmp_path):
        """Test load_model calls ModelLoader.load_from_checkpoint."""
        mock_model = MagicMock(spec=nn.Module)
        mock_model_info = {"uses_edge_features": False}
        mock_load_from_checkpoint.return_value = (mock_model, mock_model_info)

        model, model_info = load_model("test.pt", working_root_dir=tmp_path)

        mock_load_from_checkpoint.assert_called_once()

    @patch(
        "milia_pipeline.models.post_training.inference.model_loader.ModelLoader.load_from_checkpoint"
    )
    def test_load_model_passes_checkpoint_path(self, mock_load_from_checkpoint, tmp_path):
        """Test load_model passes checkpoint_path correctly."""
        mock_model = MagicMock(spec=nn.Module)
        mock_load_from_checkpoint.return_value = (mock_model, {})

        load_model("my_checkpoint.pt", working_root_dir=tmp_path)

        call_kwargs = mock_load_from_checkpoint.call_args[1]
        assert call_kwargs["checkpoint_path"] == "my_checkpoint.pt"

    @patch(
        "milia_pipeline.models.post_training.inference.model_loader.ModelLoader.load_from_checkpoint"
    )
    def test_load_model_passes_working_root_dir(self, mock_load_from_checkpoint, tmp_path):
        """Test load_model passes working_root_dir correctly."""
        mock_model = MagicMock(spec=nn.Module)
        mock_load_from_checkpoint.return_value = (mock_model, {})

        load_model("test.pt", working_root_dir=tmp_path)

        call_kwargs = mock_load_from_checkpoint.call_args[1]
        assert call_kwargs["working_root_dir"] == tmp_path

    @patch(
        "milia_pipeline.models.post_training.inference.model_loader.ModelLoader.load_from_checkpoint"
    )
    def test_load_model_passes_device(self, mock_load_from_checkpoint, tmp_path):
        """Test load_model passes device correctly."""
        mock_model = MagicMock(spec=nn.Module)
        mock_load_from_checkpoint.return_value = (mock_model, {})

        device = torch.device("cuda")
        load_model("test.pt", working_root_dir=tmp_path, device=device)

        call_kwargs = mock_load_from_checkpoint.call_args[1]
        assert call_kwargs["device"] == device

    @patch(
        "milia_pipeline.models.post_training.inference.model_loader.ModelLoader.load_from_checkpoint"
    )
    def test_load_model_passes_kwargs(self, mock_load_from_checkpoint, tmp_path):
        """Test load_model passes additional kwargs."""
        mock_model = MagicMock(spec=nn.Module)
        mock_load_from_checkpoint.return_value = (mock_model, {})

        load_model(
            "test.pt",
            working_root_dir=tmp_path,
            model_name="GCN",
            hyperparameters={"hidden_channels": 64},
            task_type="graph_regression",
            strict=False,
        )

        call_kwargs = mock_load_from_checkpoint.call_args[1]
        assert call_kwargs["model_name"] == "GCN"
        assert call_kwargs["hyperparameters"] == {"hidden_channels": 64}
        assert call_kwargs["task_type"] == "graph_regression"
        assert call_kwargs["strict"] is False

    @patch(
        "milia_pipeline.models.post_training.inference.model_loader.ModelLoader.load_from_checkpoint"
    )
    def test_load_model_returns_tuple(self, mock_load_from_checkpoint, tmp_path):
        """Test load_model returns (model, model_info) tuple."""
        mock_model = MagicMock(spec=nn.Module)
        mock_model_info = {"uses_edge_features": True, "task_type": "graph_regression"}
        mock_load_from_checkpoint.return_value = (mock_model, mock_model_info)

        result = load_model("test.pt", working_root_dir=tmp_path)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] is mock_model
        assert result[1] == mock_model_info


# =============================================================================
# CONVENIENCE FUNCTION TESTS - load_model_only
# =============================================================================


class TestLoadModelOnlyFunction:
    """Test load_model_only convenience function."""

    @patch("milia_pipeline.models.post_training.inference.model_loader.load_model")
    def test_load_model_only_calls_load_model(self, mock_load_model, tmp_path):
        """Test load_model_only calls load_model."""
        mock_model = MagicMock(spec=nn.Module)
        mock_load_model.return_value = (mock_model, {"info": "discarded"})

        _model = load_model_only("test.pt", working_root_dir=tmp_path)

        mock_load_model.assert_called_once()

    @patch("milia_pipeline.models.post_training.inference.model_loader.load_model")
    def test_load_model_only_returns_model_only(self, mock_load_model, tmp_path):
        """Test load_model_only returns only the model, not model_info."""
        mock_model = MagicMock(spec=nn.Module)
        mock_model_info = {"uses_edge_features": True, "task_type": "graph_regression"}
        mock_load_model.return_value = (mock_model, mock_model_info)

        result = load_model_only("test.pt", working_root_dir=tmp_path)

        # Should return model only, not a tuple
        assert result is mock_model
        assert not isinstance(result, tuple)

    @patch("milia_pipeline.models.post_training.inference.model_loader.load_model")
    def test_load_model_only_passes_checkpoint_path(self, mock_load_model, tmp_path):
        """Test load_model_only passes checkpoint_path to load_model."""
        mock_model = MagicMock(spec=nn.Module)
        mock_load_model.return_value = (mock_model, {})

        load_model_only("my_model.pt", working_root_dir=tmp_path)

        call_args, call_kwargs = mock_load_model.call_args
        assert call_args[0] == "my_model.pt"

    @patch("milia_pipeline.models.post_training.inference.model_loader.load_model")
    def test_load_model_only_passes_working_root_dir(self, mock_load_model, tmp_path):
        """Test load_model_only passes working_root_dir to load_model."""
        mock_model = MagicMock(spec=nn.Module)
        mock_load_model.return_value = (mock_model, {})

        load_model_only("test.pt", working_root_dir=tmp_path)

        call_kwargs = mock_load_model.call_args[1]
        assert call_kwargs["working_root_dir"] == tmp_path

    @patch("milia_pipeline.models.post_training.inference.model_loader.load_model")
    def test_load_model_only_passes_device(self, mock_load_model, tmp_path):
        """Test load_model_only passes device to load_model."""
        mock_model = MagicMock(spec=nn.Module)
        mock_load_model.return_value = (mock_model, {})

        device = torch.device("cpu")
        load_model_only("test.pt", working_root_dir=tmp_path, device=device)

        call_kwargs = mock_load_model.call_args[1]
        assert call_kwargs["device"] == device

    @patch("milia_pipeline.models.post_training.inference.model_loader.load_model")
    def test_load_model_only_passes_kwargs(self, mock_load_model, tmp_path):
        """Test load_model_only passes additional kwargs to load_model."""
        mock_model = MagicMock(spec=nn.Module)
        mock_load_model.return_value = (mock_model, {})

        load_model_only(
            "test.pt",
            working_root_dir=tmp_path,
            model_name="GAT",
            hyperparameters={"heads": 8},
            strict=False,
        )

        call_kwargs = mock_load_model.call_args[1]
        assert call_kwargs["model_name"] == "GAT"
        assert call_kwargs["hyperparameters"] == {"heads": 8}
        assert call_kwargs["strict"] is False


# =============================================================================
# LOGGING TESTS
# =============================================================================


class TestLogging:
    """Test logging behavior."""

    def test_module_logger_name(self, tmp_path):
        """Test that module uses correct logger name."""
        assert (
            model_loader_logger.name == "milia_pipeline.models.post_training.inference.model_loader"
        )

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_logs_info_on_success(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, caplog, tmp_path
    ):
        """Test that successful load logs info message."""
        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        with caplog.at_level(logging.INFO):
            ModelLoader.load_from_checkpoint("test.pt", working_root_dir=tmp_path)

        assert any("Model loaded successfully" in record.message for record in caplog.records)

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_logs_model_recreation_info(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, caplog, tmp_path
    ):
        """Test that model recreation is logged."""
        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        with caplog.at_level(logging.INFO):
            ModelLoader.load_from_checkpoint("test.pt", working_root_dir=tmp_path)

        assert any(
            "Recreating" in record.message and "GCN" in record.message for record in caplog.records
        )


# =============================================================================
# EDGE CASES TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_handles_empty_hyperparameters_dict(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test _load handles checkpoint with empty hyperparameters dict."""
        v2_checkpoint["hyper_parameters"]["hyperparameters"] = {}

        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        model, _ = ModelLoader.load_from_checkpoint("test.pt", working_root_dir=tmp_path)

        call_kwargs = mock_factory.create_model_with_info.call_args[1]
        assert call_kwargs["hyperparameters"] == {}

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_handles_missing_model_info_in_checkpoint(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test _load handles checkpoint without model_info."""
        del v2_checkpoint["hyper_parameters"]["model_info"]

        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        recreated_info = {"uses_edge_features": True}
        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, recreated_info)
        mock_get_factory.return_value = mock_factory

        model, model_info = ModelLoader.load_from_checkpoint("test.pt", working_root_dir=tmp_path)

        # Should use recreated info when saved is missing
        assert model_info["uses_edge_features"] is True

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_handles_missing_wrapper_info(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test _load handles checkpoint without wrapper_info."""
        del v2_checkpoint["hyper_parameters"]["wrapper_info"]

        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        # Should not raise even without wrapper_info
        model, _ = ModelLoader.load_from_checkpoint("test.pt", working_root_dir=tmp_path)

        assert model is mock_model

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_handles_none_target_selection_config(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test _load handles None target_selection_config."""
        v2_checkpoint["hyper_parameters"]["target_selection_config"] = None

        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        model, _ = ModelLoader.load_from_checkpoint("test.pt", working_root_dir=tmp_path)

        call_kwargs = mock_factory.create_model_with_info.call_args[1]
        assert call_kwargs["target_selection_config"] is None

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_get_checkpoint_info_handles_missing_version_info(
        self, mock_get_factory, mock_cm_class, tmp_path
    ):
        """Test get_checkpoint_info handles checkpoint without version_info."""
        checkpoint = {
            "epoch": 10,
            "model_state_dict": {},
            "hyper_parameters": {},
            # No version_info
        }

        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = checkpoint
        mock_cm_instance.is_v2_checkpoint.return_value = False
        mock_cm_instance._resolve_checkpoint_path.return_value = tmp_path / "checkpoint.pt"
        mock_cm_class.return_value = mock_cm_instance
        mock_get_factory.return_value = MagicMock()

        info = ModelLoader.get_checkpoint_info("test.pt", working_root_dir=tmp_path)

        assert info["format_version"] == "1.0"  # Default
        assert info["version_info"] == {}

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_get_checkpoint_info_handles_missing_epoch(
        self, mock_get_factory, mock_cm_class, tmp_path
    ):
        """Test get_checkpoint_info handles checkpoint without epoch."""
        checkpoint = {
            "model_state_dict": {},
            "hyper_parameters": {},
            "version_info": {},
            # No epoch
        }

        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = checkpoint
        mock_cm_instance.is_v2_checkpoint.return_value = False
        mock_cm_instance._resolve_checkpoint_path.return_value = tmp_path / "checkpoint.pt"
        mock_cm_class.return_value = mock_cm_instance
        mock_get_factory.return_value = MagicMock()

        info = ModelLoader.get_checkpoint_info("test.pt", working_root_dir=tmp_path)

        assert info["epoch"] == 0  # Default

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_get_checkpoint_info_handles_missing_best_val_loss(
        self, mock_get_factory, mock_cm_class, tmp_path
    ):
        """Test get_checkpoint_info handles checkpoint without best_val_loss."""
        checkpoint = {
            "epoch": 10,
            "model_state_dict": {},
            "hyper_parameters": {},
            "version_info": {},
            # No best_val_loss
        }

        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = checkpoint
        mock_cm_instance.is_v2_checkpoint.return_value = False
        mock_cm_instance._resolve_checkpoint_path.return_value = tmp_path / "checkpoint.pt"
        mock_cm_class.return_value = mock_cm_instance
        mock_get_factory.return_value = MagicMock()

        info = ModelLoader.get_checkpoint_info("test.pt", working_root_dir=tmp_path)

        assert info["best_val_loss"] is None  # Default


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================


class TestThreadSafety:
    """Test thread safety of ModelLoader operations."""

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_concurrent_load_from_checkpoint(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test concurrent calls to load_from_checkpoint are safe."""
        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        results = []
        errors = []

        # Capture tmp_path in closure for thread access
        working_dir = tmp_path

        def load_checkpoint(idx):
            try:
                model, info = ModelLoader.load_from_checkpoint(
                    f"test_{idx}.pt", working_root_dir=working_dir
                )
                results.append((idx, model))
            except Exception as e:
                errors.append((idx, str(e)))

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(load_checkpoint, i) for i in range(20)]
            for future in as_completed(futures):
                pass

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 20

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_concurrent_get_checkpoint_info(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test concurrent calls to get_checkpoint_info are safe."""
        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_instance.is_v2_checkpoint.return_value = True
        mock_cm_class.return_value = mock_cm_instance
        mock_get_factory.return_value = MagicMock()

        results = []
        errors = []

        # Capture tmp_path in closure for thread access
        working_dir = tmp_path

        def get_info(idx):
            try:
                info = ModelLoader.get_checkpoint_info(
                    f"test_{idx}.pt", working_root_dir=working_dir
                )
                results.append((idx, info))
            except Exception as e:
                errors.append((idx, str(e)))

        threads = [threading.Thread(target=get_info, args=(i,)) for i in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 30
        # All results should have same format_version
        assert all(info["format_version"] == "2.0" for _, info in results)


# =============================================================================
# TYPE ANNOTATION VERIFICATION TESTS
# =============================================================================


class TestTypeAnnotations:
    """Test type annotation compliance."""

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_from_checkpoint_returns_correct_types(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test load_from_checkpoint returns Tuple[nn.Module, Dict]."""
        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {"key": "value"})
        mock_get_factory.return_value = mock_factory

        result = ModelLoader.load_from_checkpoint("test.pt", working_root_dir=tmp_path)

        assert isinstance(result, tuple)
        assert len(result) == 2
        # First element should be the model
        assert hasattr(result[0], "eval")  # Has model-like interface
        # Second element should be a dict
        assert isinstance(result[1], dict)

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_get_checkpoint_info_returns_dict(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test get_checkpoint_info returns Dict[str, Any]."""
        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = v2_checkpoint
        mock_cm_instance.is_v2_checkpoint.return_value = True
        mock_cm_instance._resolve_checkpoint_path.return_value = tmp_path / "checkpoint.pt"
        mock_cm_class.return_value = mock_cm_instance
        mock_get_factory.return_value = MagicMock()

        result = ModelLoader.get_checkpoint_info("test.pt", working_root_dir=tmp_path)

        assert isinstance(result, dict)
        # All keys should be strings
        assert all(isinstance(k, str) for k in result.keys())

    @patch("milia_pipeline.models.post_training.inference.model_loader.load_model")
    def test_load_model_only_returns_nn_module(self, mock_load_model, tmp_path):
        """Test load_model_only returns nn.Module, not tuple."""
        mock_model = MagicMock(spec=nn.Module)
        mock_load_model.return_value = (mock_model, {})

        result = load_model_only("test.pt", working_root_dir=tmp_path)

        # Should not be a tuple
        assert not isinstance(result, tuple)
        # Should be the model mock
        assert result is mock_model


# =============================================================================
# SPECIAL MODEL TYPES TESTS
# =============================================================================


class TestSpecialModelTypes:
    """Test handling of special model types."""

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_3d_model_checkpoint(
        self, mock_get_factory, mock_cm_class, sample_state_dict, tmp_path
    ):
        """Test loading a 3D model (SchNet, DimeNet) checkpoint."""
        checkpoint = {
            "epoch": 50,
            "model_state_dict": sample_state_dict,
            "best_val_loss": 0.05,
            "hyper_parameters": {
                "model_name": "SchNet",
                "task_type": "graph_regression",
                "hyperparameters": {
                    "hidden_channels": 128,
                    "num_filters": 128,
                    "num_interactions": 6,
                    "cutoff": 10.0,
                },
                "model_info": {},
                "wrapper_info": {
                    "wrapper_type": "GraphLevelModelWrapper",
                    "out_channels": 8,
                },
            },
            "version_info": {
                "checkpoint_format_version": "2.0",
            },
        }

        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = checkpoint
        mock_cm_class.return_value = mock_cm_instance

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        model, _ = ModelLoader.load_from_checkpoint("schnet.pt", working_root_dir=tmp_path)

        call_kwargs = mock_factory.create_model_with_info.call_args[1]
        assert call_kwargs["name"] == "SchNet"
        assert call_kwargs["hyperparameters"]["cutoff"] == 10.0

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_ensemble_model_checkpoint(
        self, mock_get_factory, mock_cm_class, sample_state_dict, tmp_path
    ):
        """Test loading an ensemble model checkpoint."""
        checkpoint = {
            "epoch": 100,
            "model_state_dict": sample_state_dict,
            "hyper_parameters": {
                "model_name": "ensemble",
                "task_type": "graph_regression",
                "hyperparameters": {
                    "ensemble_config": {
                        "models": ["GCN", "GAT"],
                        "fusion": "mean",
                    },
                },
                "model_info": {},
            },
            "version_info": {
                "checkpoint_format_version": "2.0",
            },
        }

        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = checkpoint
        mock_cm_class.return_value = mock_cm_instance

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        model, _ = ModelLoader.load_from_checkpoint("ensemble.pt", working_root_dir=tmp_path)

        call_kwargs = mock_factory.create_model_with_info.call_args[1]
        assert call_kwargs["name"] == "ensemble"
        assert "ensemble_config" in call_kwargs["hyperparameters"]

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_custom_architecture_checkpoint(
        self, mock_get_factory, mock_cm_class, sample_state_dict, tmp_path
    ):
        """Test loading a custom architecture checkpoint."""
        checkpoint = {
            "epoch": 75,
            "model_state_dict": sample_state_dict,
            "hyper_parameters": {
                "model_name": "custom",
                "task_type": "node_classification",
                "hyperparameters": {
                    "architecture_config": {
                        "layers": [
                            {"type": "GCNConv", "out_channels": 64},
                            {"type": "ReLU"},
                            {"type": "GCNConv", "out_channels": 32},
                        ],
                    },
                },
                "model_info": {},
            },
            "version_info": {
                "checkpoint_format_version": "2.0",
            },
        }

        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = checkpoint
        mock_cm_instance._resolve_checkpoint_path.return_value = tmp_path / "checkpoint.pt"
        mock_cm_instance.get_checkpoint_dir.return_value = tmp_path / "checkpoints"
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model
        mock_model.state_dict.return_value = sample_state_dict

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        model, _ = ModelLoader.load_from_checkpoint("custom.pt", working_root_dir=tmp_path)

        call_kwargs = mock_factory.create_model_with_info.call_args[1]
        assert call_kwargs["name"] == "custom"
        assert "architecture_config" in call_kwargs["hyperparameters"]


# =============================================================================
# DIFFERENT TASK TYPES TESTS
# =============================================================================


class TestDifferentTaskTypes:
    """Test loading checkpoints for different task types."""

    @pytest.mark.parametrize(
        "task_type",
        [
            "graph_regression",
            "graph_classification",
            "node_regression",
            "node_classification",
            "link_prediction",
            "edge_regression",
            "edge_classification",
        ],
    )
    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_checkpoint_for_task_type(
        self, mock_get_factory, mock_cm_class, task_type, sample_state_dict, tmp_path
    ):
        """Test loading checkpoint for each task type."""
        checkpoint = {
            "epoch": 50,
            "model_state_dict": sample_state_dict,
            "hyper_parameters": {
                "model_name": "GCN",
                "task_type": task_type,
                "hyperparameters": {"hidden_channels": 64},
                "model_info": {},
            },
            "version_info": {
                "checkpoint_format_version": "2.0",
            },
        }

        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = checkpoint
        mock_cm_class.return_value = mock_cm_instance

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {"task_type": task_type})
        mock_get_factory.return_value = mock_factory

        model, model_info = ModelLoader.load_from_checkpoint("test.pt", working_root_dir=tmp_path)

        call_kwargs = mock_factory.create_model_with_info.call_args[1]
        assert call_kwargs["task_type"] == task_type


# =============================================================================
# SAMPLE_DATA HANDLING TESTS
# =============================================================================


class TestSampleDataHandling:
    """Test that sample_data is correctly handled (always None for inference)."""

    @patch("milia_pipeline.models.post_training.inference.model_loader.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.get_factory")
    def test_load_passes_none_for_sample_data(
        self, mock_get_factory, mock_cm_class, v2_checkpoint, tmp_path
    ):
        """Test _load passes sample_data=None to create_model_with_info."""
        mock_cm_instance = create_configured_checkpoint_manager_mock(v2_checkpoint, tmp_path)
        mock_cm_class.return_value = mock_cm_instance

        # Create fake checkpoint file
        (tmp_path / "checkpoint.pt").touch()

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model

        mock_factory = MagicMock()
        mock_factory.create_model_with_info.return_value = (mock_model, {})
        mock_get_factory.return_value = mock_factory

        model, _ = ModelLoader.load_from_checkpoint("test.pt", working_root_dir=tmp_path)

        call_kwargs = mock_factory.create_model_with_info.call_args[1]
        assert call_kwargs["sample_data"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
