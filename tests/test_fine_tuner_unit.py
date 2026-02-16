#!/usr/bin/env python3
"""
Extended Production-Ready Unit Test Suite for fine_tuner.py Module

Comprehensive test coverage for the FineTuner class and FreezeStrategy enum.
Tests cover:
- FreezeStrategy enum values and behavior
- FineTuner class initialization with Dependency Injection pattern
- from_checkpoint class method with explicit working_root_dir
- prepare_for_finetuning method
- _apply_freeze_strategy internal method
- _freeze_encoder_layers internal method
- _freeze_first_n_layers internal method
- _freeze_all_but_last internal method
- _replace_output_head internal method
- _log_parameter_status internal method
- Device handling (CPU/CUDA)
- Edge cases and error conditions

Dependency Injection Pattern:
- All FineTuner instantiations require explicit working_root_dir: Path parameter
- No hidden config loading (Service Locator anti-pattern removed)
- Follows CallbackFactory pattern from models/training/callbacks.py

Author: MILIA Team
Version: 2.0.0
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import logging
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# Import the module under test
from milia_pipeline.models.post_training.transfer_learning.fine_tuner import (
    FineTuner,
    FreezeStrategy,
)
from milia_pipeline.models.post_training.transfer_learning.fine_tuner import (
    logger as fine_tuner_logger,
)

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_simple_model():
    """Create a simple mock PyTorch model with basic structure."""
    model = MagicMock(spec=nn.Module)
    model.train = MagicMock(return_value=model)
    model.eval = MagicMock(return_value=model)
    model.parameters = MagicMock(
        return_value=iter(
            [
                nn.Parameter(torch.randn(10, 10)),
                nn.Parameter(torch.randn(10)),
            ]
        )
    )
    model.named_parameters = MagicMock(
        return_value=iter(
            [
                ("layer1.weight", nn.Parameter(torch.randn(10, 10))),
                ("layer1.bias", nn.Parameter(torch.randn(10))),
            ]
        )
    )
    model.named_modules = MagicMock(
        return_value=iter(
            [
                ("", model),
            ]
        )
    )
    return model


@pytest.fixture
def simple_linear_model():
    """Create a simple real PyTorch model with linear layers."""

    class SimpleModel(nn.Module):
        def __init__(self, in_features=16, hidden=32, out_features=1):
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden)
            self.fc2 = nn.Linear(hidden, out_features)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    return SimpleModel()


@pytest.fixture
def multi_layer_model():
    """Create a model with multiple layers for freezing tests."""

    class MultiLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Linear(16, 32)  # Named 'conv' for encoder freezing
            self.conv2 = nn.Linear(32, 64)
            self.message_layer = nn.Linear(64, 64)  # Named for encoder freezing
            self.fc_out = nn.Linear(64, 1)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.message_layer(x))
            return self.fc_out(x)

    return MultiLayerModel()


@pytest.fixture
def gnn_like_model():
    """Create a model with GNN-like layer names for encoder freezing tests."""

    class GNNLikeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_layers = nn.ModuleList(
                [
                    nn.Linear(16, 32),
                    nn.Linear(32, 64),
                ]
            )
            self.message_passing = nn.Linear(64, 64)
            self.propagate_layer = nn.Linear(64, 64)
            self.aggr_layer = nn.Linear(64, 32)
            self.output = nn.Linear(32, 1)

        def forward(self, x):
            for conv in self.conv_layers:
                x = torch.relu(conv(x))
            x = torch.relu(self.message_passing(x))
            x = torch.relu(self.propagate_layer(x))
            x = torch.relu(self.aggr_layer(x))
            return self.output(x)

    return GNNLikeModel()


@pytest.fixture
def nested_model():
    """Create a model with nested modules."""

    class NestedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
            )
            self.decoder = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            x = self.encoder(x)
            return self.decoder(x)

    return NestedModel()


@pytest.fixture
def model_with_conv_names():
    """Create a model with various conv-like parameter names."""

    class ConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_block = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
            )
            self.message_block = nn.Linear(32, 32)
            self.propagate_block = nn.Linear(32, 32)
            self.aggr_block = nn.Linear(32, 32)
            self.head = nn.Linear(32, 1)

        def forward(self, x):
            x = self.conv_block(x)
            x = self.message_block(x)
            x = self.propagate_block(x)
            x = self.aggr_block(x)
            return self.head(x)

    return ConvModel()


@pytest.fixture
def basic_hyper_parameters():
    """Create basic hyper_parameters dictionary."""
    return {
        "model_name": "GCN",
        "task_type": "graph_regression",
        "out_channels": 1,
        "hyperparameters": {
            "hidden_channels": 64,
            "num_layers": 3,
        },
    }


@pytest.fixture
def multi_output_hyper_parameters():
    """Create hyper_parameters for multi-output model."""
    return {
        "model_name": "GCN",
        "task_type": "graph_regression",
        "out_channels": 5,
        "hyperparameters": {
            "hidden_channels": 64,
            "num_layers": 3,
            "out_channels": 5,
        },
    }


@pytest.fixture
def classification_hyper_parameters():
    """Create hyper_parameters for classification task."""
    return {
        "model_name": "GCN",
        "task_type": "graph_classification",
        "out_channels": 3,
        "hyperparameters": {
            "hidden_channels": 64,
            "out_channels": 3,
        },
    }


@pytest.fixture
def cpu_device():
    """Return CPU device."""
    return torch.device("cpu")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def working_root_dir(temp_dir):
    """
    Create a working_root_dir for Dependency Injection pattern.

    Following the module's DI pattern, this fixture provides an explicit
    working_root_dir Path that must be passed to FineTuner and related
    classes. This avoids the Service Locator anti-pattern.
    """
    return temp_dir


@pytest.fixture
def mock_checkpoint():
    """Create a mock checkpoint dictionary."""
    return {
        "epoch": 100,
        "model_state_dict": {
            "fc1.weight": torch.randn(32, 16),
            "fc1.bias": torch.randn(32),
            "fc2.weight": torch.randn(1, 32),
            "fc2.bias": torch.randn(1),
        },
        "hyper_parameters": {
            "model_name": "GCN",
            "task_type": "graph_regression",
            "out_channels": 1,
            "hyperparameters": {
                "hidden_channels": 64,
                "num_layers": 3,
            },
        },
        "version_info": {
            "checkpoint_format_version": "2.0",
        },
    }


# =============================================================================
# FREEZESTRATEGY ENUM TESTS
# =============================================================================


class TestFreezeStrategyEnum:
    """Test FreezeStrategy enum values and properties."""

    def test_freeze_strategy_none_value(self):
        """Test NONE strategy has correct value."""
        assert FreezeStrategy.NONE.value == "none"

    def test_freeze_strategy_encoder_value(self):
        """Test ENCODER strategy has correct value."""
        assert FreezeStrategy.ENCODER.value == "encoder"

    def test_freeze_strategy_encoder_partial_value(self):
        """Test ENCODER_PARTIAL strategy has correct value."""
        assert FreezeStrategy.ENCODER_PARTIAL.value == "encoder_partial"

    def test_freeze_strategy_all_but_last_value(self):
        """Test ALL_BUT_LAST strategy has correct value."""
        assert FreezeStrategy.ALL_BUT_LAST.value == "all_but_last"

    def test_freeze_strategy_has_four_members(self):
        """Test FreezeStrategy has exactly four members."""
        assert len(FreezeStrategy) == 4

    def test_freeze_strategy_members_are_unique(self):
        """Test all FreezeStrategy values are unique."""
        values = [s.value for s in FreezeStrategy]
        assert len(values) == len(set(values))

    def test_freeze_strategy_is_enum(self):
        """Test FreezeStrategy is an Enum."""
        from enum import Enum

        assert issubclass(FreezeStrategy, Enum)

    def test_freeze_strategy_can_be_compared(self):
        """Test FreezeStrategy members can be compared."""
        assert FreezeStrategy.NONE == FreezeStrategy.NONE
        assert FreezeStrategy.NONE != FreezeStrategy.ENCODER

    def test_freeze_strategy_can_be_accessed_by_name(self):
        """Test FreezeStrategy can be accessed by name."""
        assert FreezeStrategy["NONE"] == FreezeStrategy.NONE
        assert FreezeStrategy["ENCODER"] == FreezeStrategy.ENCODER

    def test_freeze_strategy_can_be_accessed_by_value(self):
        """Test FreezeStrategy can be instantiated by value."""
        assert FreezeStrategy("none") == FreezeStrategy.NONE
        assert FreezeStrategy("encoder") == FreezeStrategy.ENCODER


# =============================================================================
# FINETUNER INITIALIZATION TESTS
# =============================================================================


class TestFineTunerInitialization:
    """Test FineTuner class initialization."""

    def test_init_stores_model(self, simple_linear_model, basic_hyper_parameters, working_root_dir):
        """Test that __init__ stores the model."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )
        assert fine_tuner.model is simple_linear_model

    def test_init_stores_hyper_parameters(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test that __init__ stores hyper_parameters."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )
        assert fine_tuner.hyper_parameters is basic_hyper_parameters

    def test_init_extracts_original_out_channels(
        self, simple_linear_model, multi_output_hyper_parameters, working_root_dir
    ):
        """Test that __init__ extracts original_out_channels from hyper_parameters."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=multi_output_hyper_parameters,
            working_root_dir=working_root_dir,
        )
        assert fine_tuner.original_out_channels == 5

    def test_init_default_out_channels_is_one(self, simple_linear_model, working_root_dir):
        """Test that default out_channels is 1 when not in hyper_parameters."""
        fine_tuner = FineTuner(
            model=simple_linear_model, hyper_parameters={}, working_root_dir=working_root_dir
        )
        assert fine_tuner.original_out_channels == 1

    def test_init_with_empty_hyper_parameters(self, simple_linear_model, working_root_dir):
        """Test initialization with empty hyper_parameters."""
        fine_tuner = FineTuner(
            model=simple_linear_model, hyper_parameters={}, working_root_dir=working_root_dir
        )
        assert fine_tuner.hyper_parameters == {}
        assert fine_tuner.original_out_channels == 1

    def test_init_with_mock_model(
        self, mock_simple_model, basic_hyper_parameters, working_root_dir
    ):
        """Test initialization with mock model."""
        fine_tuner = FineTuner(
            model=mock_simple_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )
        assert fine_tuner.model is mock_simple_model

    def test_init_with_classification_hyper_parameters(
        self, simple_linear_model, classification_hyper_parameters, working_root_dir
    ):
        """Test initialization with classification hyper_parameters."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=classification_hyper_parameters,
            working_root_dir=working_root_dir,
        )
        assert fine_tuner.original_out_channels == 3

    def test_init_stores_working_root_dir(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test that __init__ stores and resolves working_root_dir."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )
        # The working_root_dir should be stored as a resolved Path
        assert fine_tuner._working_root_dir == Path(working_root_dir).expanduser().resolve()

    def test_init_accepts_string_working_root_dir(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test that __init__ accepts string path for working_root_dir."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=str(working_root_dir),
        )
        assert fine_tuner._working_root_dir == Path(working_root_dir).expanduser().resolve()


# =============================================================================
# FINETUNER FROM_CHECKPOINT TESTS
# =============================================================================


class TestFineTunerFromCheckpoint:
    """Test FineTuner.from_checkpoint class method."""

    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.CheckpointManager")
    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.ModelLoader")
    def test_from_checkpoint_loads_model(
        self,
        mock_loader_class,
        mock_cm_class,
        simple_linear_model,
        mock_checkpoint,
        working_root_dir,
    ):
        """Test from_checkpoint loads model via ModelLoader."""
        checkpoint_path = working_root_dir / "model.pt"

        # Setup mocks
        mock_cm_instance = MagicMock()
        mock_cm_instance._resolve_checkpoint_path.return_value = checkpoint_path
        mock_cm_instance.load.return_value = mock_checkpoint
        mock_cm_class.return_value = mock_cm_instance

        mock_loader_class.load_from_checkpoint.return_value = (simple_linear_model, {})

        _fine_tuner = FineTuner.from_checkpoint(checkpoint_path, working_root_dir=working_root_dir)

        mock_loader_class.load_from_checkpoint.assert_called_once_with(
            checkpoint_path=checkpoint_path, working_root_dir=working_root_dir, device=None
        )

    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.CheckpointManager")
    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.ModelLoader")
    def test_from_checkpoint_with_device(
        self,
        mock_loader_class,
        mock_cm_class,
        simple_linear_model,
        mock_checkpoint,
        working_root_dir,
        cpu_device,
    ):
        """Test from_checkpoint passes device to ModelLoader."""
        checkpoint_path = working_root_dir / "model.pt"

        mock_cm_instance = MagicMock()
        mock_cm_instance._resolve_checkpoint_path.return_value = checkpoint_path
        mock_cm_instance.load.return_value = mock_checkpoint
        mock_cm_class.return_value = mock_cm_instance

        mock_loader_class.load_from_checkpoint.return_value = (simple_linear_model, {})

        _fine_tuner = FineTuner.from_checkpoint(
            checkpoint_path, working_root_dir=working_root_dir, device=cpu_device
        )

        mock_loader_class.load_from_checkpoint.assert_called_once_with(
            checkpoint_path=checkpoint_path, working_root_dir=working_root_dir, device=cpu_device
        )

    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.CheckpointManager")
    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.ModelLoader")
    def test_from_checkpoint_extracts_hyper_parameters(
        self,
        mock_loader_class,
        mock_cm_class,
        simple_linear_model,
        mock_checkpoint,
        working_root_dir,
    ):
        """Test from_checkpoint extracts hyper_parameters from checkpoint."""
        checkpoint_path = working_root_dir / "model.pt"

        mock_cm_instance = MagicMock()
        mock_cm_instance._resolve_checkpoint_path.return_value = checkpoint_path
        mock_cm_instance.load.return_value = mock_checkpoint
        mock_cm_class.return_value = mock_cm_instance

        mock_loader_class.load_from_checkpoint.return_value = (simple_linear_model, {})

        fine_tuner = FineTuner.from_checkpoint(checkpoint_path, working_root_dir=working_root_dir)

        assert fine_tuner.hyper_parameters == mock_checkpoint["hyper_parameters"]

    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.CheckpointManager")
    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.ModelLoader")
    def test_from_checkpoint_returns_fine_tuner_instance(
        self,
        mock_loader_class,
        mock_cm_class,
        simple_linear_model,
        mock_checkpoint,
        working_root_dir,
    ):
        """Test from_checkpoint returns FineTuner instance."""
        checkpoint_path = working_root_dir / "model.pt"

        mock_cm_instance = MagicMock()
        mock_cm_instance._resolve_checkpoint_path.return_value = checkpoint_path
        mock_cm_instance.load.return_value = mock_checkpoint
        mock_cm_class.return_value = mock_cm_instance

        mock_loader_class.load_from_checkpoint.return_value = (simple_linear_model, {})

        fine_tuner = FineTuner.from_checkpoint(checkpoint_path, working_root_dir=working_root_dir)

        assert isinstance(fine_tuner, FineTuner)

    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.CheckpointManager")
    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.ModelLoader")
    def test_from_checkpoint_with_string_path(
        self,
        mock_loader_class,
        mock_cm_class,
        simple_linear_model,
        mock_checkpoint,
        working_root_dir,
    ):
        """Test from_checkpoint accepts string path."""
        checkpoint_path = str(working_root_dir / "model.pt")
        resolved_path = working_root_dir / "model.pt"

        mock_cm_instance = MagicMock()
        mock_cm_instance._resolve_checkpoint_path.return_value = resolved_path
        mock_cm_instance.load.return_value = mock_checkpoint
        mock_cm_class.return_value = mock_cm_instance

        mock_loader_class.load_from_checkpoint.return_value = (simple_linear_model, {})

        fine_tuner = FineTuner.from_checkpoint(checkpoint_path, working_root_dir=working_root_dir)

        assert isinstance(fine_tuner, FineTuner)

    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.CheckpointManager")
    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.ModelLoader")
    def test_from_checkpoint_handles_missing_hyper_parameters(
        self, mock_loader_class, mock_cm_class, simple_linear_model, working_root_dir
    ):
        """Test from_checkpoint handles checkpoint without hyper_parameters."""
        checkpoint_path = working_root_dir / "model.pt"
        checkpoint_without_params = {
            "epoch": 50,
            "model_state_dict": {},
        }

        mock_cm_instance = MagicMock()
        mock_cm_instance._resolve_checkpoint_path.return_value = checkpoint_path
        mock_cm_instance.load.return_value = checkpoint_without_params
        mock_cm_class.return_value = mock_cm_instance

        mock_loader_class.load_from_checkpoint.return_value = (simple_linear_model, {})

        fine_tuner = FineTuner.from_checkpoint(checkpoint_path, working_root_dir=working_root_dir)

        assert fine_tuner.hyper_parameters == {}

    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.CheckpointManager")
    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.ModelLoader")
    def test_from_checkpoint_stores_loaded_model(
        self,
        mock_loader_class,
        mock_cm_class,
        simple_linear_model,
        mock_checkpoint,
        working_root_dir,
    ):
        """Test from_checkpoint stores the loaded model."""
        checkpoint_path = working_root_dir / "model.pt"

        mock_cm_instance = MagicMock()
        mock_cm_instance._resolve_checkpoint_path.return_value = checkpoint_path
        mock_cm_instance.load.return_value = mock_checkpoint
        mock_cm_class.return_value = mock_cm_instance

        mock_loader_class.load_from_checkpoint.return_value = (simple_linear_model, {})

        fine_tuner = FineTuner.from_checkpoint(checkpoint_path, working_root_dir=working_root_dir)

        assert fine_tuner.model is simple_linear_model

    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.CheckpointManager")
    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.ModelLoader")
    def test_from_checkpoint_creates_checkpoint_manager_with_working_root_dir(
        self,
        mock_loader_class,
        mock_cm_class,
        simple_linear_model,
        mock_checkpoint,
        working_root_dir,
    ):
        """Test from_checkpoint creates CheckpointManager with working_root_dir."""
        checkpoint_path = working_root_dir / "model.pt"

        mock_cm_instance = MagicMock()
        mock_cm_instance._resolve_checkpoint_path.return_value = checkpoint_path
        mock_cm_instance.load.return_value = mock_checkpoint
        mock_cm_class.return_value = mock_cm_instance

        mock_loader_class.load_from_checkpoint.return_value = (simple_linear_model, {})

        FineTuner.from_checkpoint(checkpoint_path, working_root_dir=working_root_dir)

        mock_cm_class.assert_called_once_with(working_root_dir=working_root_dir)

    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.CheckpointManager")
    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.ModelLoader")
    def test_from_checkpoint_uses_checkpoint_manager_path_resolution(
        self,
        mock_loader_class,
        mock_cm_class,
        simple_linear_model,
        mock_checkpoint,
        working_root_dir,
    ):
        """Test from_checkpoint uses CheckpointManager for path resolution."""
        checkpoint_path = "relative/path/model.pt"
        resolved_path = working_root_dir / "resolved" / "model.pt"

        mock_cm_instance = MagicMock()
        mock_cm_instance._resolve_checkpoint_path.return_value = resolved_path
        mock_cm_instance.load.return_value = mock_checkpoint
        mock_cm_class.return_value = mock_cm_instance

        mock_loader_class.load_from_checkpoint.return_value = (simple_linear_model, {})

        FineTuner.from_checkpoint(checkpoint_path, working_root_dir=working_root_dir)

        mock_cm_instance._resolve_checkpoint_path.assert_called_once_with(checkpoint_path)


# =============================================================================
# PREPARE_FOR_FINETUNING TESTS
# =============================================================================


class TestPrepareForFinetuning:
    """Test FineTuner.prepare_for_finetuning method."""

    def test_prepare_returns_model(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test prepare_for_finetuning returns a model."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        result = fine_tuner.prepare_for_finetuning()

        assert isinstance(result, nn.Module)

    def test_prepare_sets_model_to_training_mode(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test prepare_for_finetuning sets model to training mode."""
        simple_linear_model.eval()  # Start in eval mode

        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        result = fine_tuner.prepare_for_finetuning()

        assert result.training is True

    def test_prepare_with_default_freeze_strategy(
        self, gnn_like_model, basic_hyper_parameters, working_root_dir
    ):
        """Test prepare_for_finetuning uses ENCODER as default strategy."""
        fine_tuner = FineTuner(
            model=gnn_like_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        result = fine_tuner.prepare_for_finetuning()

        # Check that conv/message/propagate/aggr layers are frozen
        for name, param in result.named_parameters():
            if any(x in name.lower() for x in ["conv", "message", "propagate", "aggr"]):
                assert not param.requires_grad, f"{name} should be frozen"

    def test_prepare_with_none_freeze_strategy(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test prepare_for_finetuning with NONE strategy unfreezes all."""
        # First freeze all parameters
        for param in simple_linear_model.parameters():
            param.requires_grad = False

        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        result = fine_tuner.prepare_for_finetuning(freeze_strategy=FreezeStrategy.NONE)

        for param in result.parameters():
            assert param.requires_grad is True

    def test_prepare_with_new_out_channels(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test prepare_for_finetuning replaces output head with new dimensions."""
        _original_out = simple_linear_model.fc2.out_features
        new_out_channels = 5

        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        result = fine_tuner.prepare_for_finetuning(
            new_out_channels=new_out_channels, freeze_strategy=FreezeStrategy.NONE
        )

        # Find the last linear layer
        last_linear = None
        for module in result.modules():
            if isinstance(module, nn.Linear):
                last_linear = module

        assert last_linear is not None
        assert last_linear.out_features == new_out_channels

    def test_prepare_with_encoder_freeze_strategy(
        self, gnn_like_model, basic_hyper_parameters, working_root_dir
    ):
        """Test prepare_for_finetuning with ENCODER freeze strategy."""
        fine_tuner = FineTuner(
            model=gnn_like_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        result = fine_tuner.prepare_for_finetuning(freeze_strategy=FreezeStrategy.ENCODER)

        # Check encoder layers are frozen
        frozen_count = 0
        unfrozen_count = 0
        for name, param in result.named_parameters():
            if any(x in name.lower() for x in ["conv", "message", "propagate", "aggr"]):
                assert not param.requires_grad
                frozen_count += 1
            else:
                assert param.requires_grad
                unfrozen_count += 1

        assert frozen_count > 0  # Some layers should be frozen
        assert unfrozen_count > 0  # Some layers should be trainable

    def test_prepare_with_encoder_partial_freeze_strategy(
        self, multi_layer_model, basic_hyper_parameters, working_root_dir
    ):
        """Test prepare_for_finetuning with ENCODER_PARTIAL freeze strategy."""
        fine_tuner = FineTuner(
            model=multi_layer_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        result = fine_tuner.prepare_for_finetuning(
            freeze_strategy=FreezeStrategy.ENCODER_PARTIAL, freeze_layers=2
        )

        # First 2 layer groups should be frozen
        layer_names = []
        for name, _ in result.named_parameters():
            layer_base = name.split(".")[0]
            if layer_base not in layer_names:
                layer_names.append(layer_base)

        # Verify freezing pattern
        for name, param in result.named_parameters():
            layer_base = name.split(".")[0]
            layer_idx = layer_names.index(layer_base)
            if layer_idx < 2:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_prepare_with_all_but_last_freeze_strategy(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test prepare_for_finetuning with ALL_BUT_LAST freeze strategy."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        result = fine_tuner.prepare_for_finetuning(freeze_strategy=FreezeStrategy.ALL_BUT_LAST)

        # Get all parameter names
        param_names = list(dict(result.named_parameters()).keys())
        last_layer_prefix = param_names[-1].rsplit(".", 1)[0] if param_names else ""

        # Only last layer should be trainable
        trainable_count = 0
        for name, param in result.named_parameters():
            if name.startswith(last_layer_prefix):
                assert param.requires_grad
                trainable_count += 1
            else:
                assert not param.requires_grad

        assert trainable_count > 0

    def test_prepare_logs_parameter_status(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir, caplog
    ):
        """Test prepare_for_finetuning logs parameter status."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        with caplog.at_level(logging.INFO):
            fine_tuner.prepare_for_finetuning(freeze_strategy=FreezeStrategy.NONE)

        # Check that parameter status was logged
        assert any(
            "Parameter status" in record.message or "trainable" in record.message.lower()
            for record in caplog.records
        )


# =============================================================================
# _APPLY_FREEZE_STRATEGY TESTS
# =============================================================================


class TestApplyFreezeStrategy:
    """Test FineTuner._apply_freeze_strategy internal method."""

    def test_apply_freeze_strategy_none(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test _apply_freeze_strategy with NONE unfreezes all parameters."""
        # First freeze all
        for param in simple_linear_model.parameters():
            param.requires_grad = False

        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        fine_tuner._apply_freeze_strategy(simple_linear_model, FreezeStrategy.NONE)

        for param in simple_linear_model.parameters():
            assert param.requires_grad is True

    def test_apply_freeze_strategy_encoder(
        self, gnn_like_model, basic_hyper_parameters, working_root_dir
    ):
        """Test _apply_freeze_strategy with ENCODER freezes encoder layers."""
        fine_tuner = FineTuner(
            model=gnn_like_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        fine_tuner._apply_freeze_strategy(gnn_like_model, FreezeStrategy.ENCODER)

        for name, param in gnn_like_model.named_parameters():
            if any(x in name.lower() for x in ["conv", "message", "propagate", "aggr"]):
                assert not param.requires_grad, f"{name} should be frozen"
            else:
                assert param.requires_grad, f"{name} should be trainable"

    def test_apply_freeze_strategy_encoder_partial(
        self, multi_layer_model, basic_hyper_parameters, working_root_dir
    ):
        """Test _apply_freeze_strategy with ENCODER_PARTIAL."""
        fine_tuner = FineTuner(
            model=multi_layer_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        fine_tuner._apply_freeze_strategy(
            multi_layer_model, FreezeStrategy.ENCODER_PARTIAL, freeze_layers=2
        )

        # Verify some are frozen, some are not
        frozen = sum(1 for _, p in multi_layer_model.named_parameters() if not p.requires_grad)
        trainable = sum(1 for _, p in multi_layer_model.named_parameters() if p.requires_grad)

        assert frozen > 0
        assert trainable > 0

    def test_apply_freeze_strategy_encoder_partial_default_layers(
        self, multi_layer_model, basic_hyper_parameters, working_root_dir
    ):
        """Test _apply_freeze_strategy with ENCODER_PARTIAL uses default 2 layers."""
        fine_tuner = FineTuner(
            model=multi_layer_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        fine_tuner._apply_freeze_strategy(
            multi_layer_model,
            FreezeStrategy.ENCODER_PARTIAL,
            freeze_layers=None,  # Should default to 2
        )

        # Get layer names
        layer_names = []
        for name, _ in multi_layer_model.named_parameters():
            layer_base = name.split(".")[0]
            if layer_base not in layer_names:
                layer_names.append(layer_base)

        # First 2 layers should be frozen
        for name, param in multi_layer_model.named_parameters():
            layer_base = name.split(".")[0]
            if layer_base in layer_names[:2]:
                assert not param.requires_grad

    def test_apply_freeze_strategy_all_but_last(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test _apply_freeze_strategy with ALL_BUT_LAST."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        fine_tuner._apply_freeze_strategy(simple_linear_model, FreezeStrategy.ALL_BUT_LAST)

        param_names = list(dict(simple_linear_model.named_parameters()).keys())
        last_layer_prefix = param_names[-1].rsplit(".", 1)[0] if param_names else ""

        for name, param in simple_linear_model.named_parameters():
            if name.startswith(last_layer_prefix):
                assert param.requires_grad
            else:
                assert not param.requires_grad


# =============================================================================
# _FREEZE_ENCODER_LAYERS TESTS
# =============================================================================


class TestFreezeEncoderLayers:
    """Test FineTuner._freeze_encoder_layers internal method."""

    def test_freeze_encoder_layers_freezes_conv(
        self, model_with_conv_names, basic_hyper_parameters, working_root_dir
    ):
        """Test _freeze_encoder_layers freezes layers with 'conv' in name."""
        fine_tuner = FineTuner(
            model=model_with_conv_names,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        fine_tuner._freeze_encoder_layers(model_with_conv_names)

        for name, param in model_with_conv_names.named_parameters():
            if "conv" in name.lower():
                assert not param.requires_grad, f"{name} should be frozen"

    def test_freeze_encoder_layers_freezes_message(
        self, model_with_conv_names, basic_hyper_parameters, working_root_dir
    ):
        """Test _freeze_encoder_layers freezes layers with 'message' in name."""
        fine_tuner = FineTuner(
            model=model_with_conv_names,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        fine_tuner._freeze_encoder_layers(model_with_conv_names)

        for name, param in model_with_conv_names.named_parameters():
            if "message" in name.lower():
                assert not param.requires_grad, f"{name} should be frozen"

    def test_freeze_encoder_layers_freezes_propagate(
        self, model_with_conv_names, basic_hyper_parameters, working_root_dir
    ):
        """Test _freeze_encoder_layers freezes layers with 'propagate' in name."""
        fine_tuner = FineTuner(
            model=model_with_conv_names,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        fine_tuner._freeze_encoder_layers(model_with_conv_names)

        for name, param in model_with_conv_names.named_parameters():
            if "propagate" in name.lower():
                assert not param.requires_grad, f"{name} should be frozen"

    def test_freeze_encoder_layers_freezes_aggr(
        self, model_with_conv_names, basic_hyper_parameters, working_root_dir
    ):
        """Test _freeze_encoder_layers freezes layers with 'aggr' in name."""
        fine_tuner = FineTuner(
            model=model_with_conv_names,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        fine_tuner._freeze_encoder_layers(model_with_conv_names)

        for name, param in model_with_conv_names.named_parameters():
            if "aggr" in name.lower():
                assert not param.requires_grad, f"{name} should be frozen"

    def test_freeze_encoder_layers_keeps_head_trainable(
        self, model_with_conv_names, basic_hyper_parameters, working_root_dir
    ):
        """Test _freeze_encoder_layers keeps non-encoder layers trainable."""
        fine_tuner = FineTuner(
            model=model_with_conv_names,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        fine_tuner._freeze_encoder_layers(model_with_conv_names)

        for name, param in model_with_conv_names.named_parameters():
            if "head" in name.lower():
                assert param.requires_grad, f"{name} should be trainable"

    def test_freeze_encoder_layers_case_insensitive(self, basic_hyper_parameters, working_root_dir):
        """Test _freeze_encoder_layers is case insensitive."""

        class MixedCaseModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.CONV_layer = nn.Linear(16, 32)
                self.Message_Layer = nn.Linear(32, 32)
                self.output = nn.Linear(32, 1)

            def forward(self, x):
                x = self.CONV_layer(x)
                x = self.Message_Layer(x)
                return self.output(x)

        model = MixedCaseModel()
        fine_tuner = FineTuner(
            model=model, hyper_parameters=basic_hyper_parameters, working_root_dir=working_root_dir
        )

        fine_tuner._freeze_encoder_layers(model)

        for name, param in model.named_parameters():
            if "conv" in name.lower() or "message" in name.lower():
                assert not param.requires_grad


# =============================================================================
# _FREEZE_FIRST_N_LAYERS TESTS
# =============================================================================


class TestFreezeFirstNLayers:
    """Test FineTuner._freeze_first_n_layers internal method."""

    def test_freeze_first_n_layers_freezes_first_layer(
        self, multi_layer_model, basic_hyper_parameters, working_root_dir
    ):
        """Test _freeze_first_n_layers freezes first layer when n=1."""
        fine_tuner = FineTuner(
            model=multi_layer_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        fine_tuner._freeze_first_n_layers(multi_layer_model, n=1)

        # Get first layer name
        layer_names = []
        for name, _ in multi_layer_model.named_parameters():
            layer_base = name.split(".")[0]
            if layer_base not in layer_names:
                layer_names.append(layer_base)

        first_layer = layer_names[0]

        for name, param in multi_layer_model.named_parameters():
            if name.startswith(first_layer):
                assert not param.requires_grad, f"{name} should be frozen"

    def test_freeze_first_n_layers_freezes_multiple_layers(
        self, multi_layer_model, basic_hyper_parameters, working_root_dir
    ):
        """Test _freeze_first_n_layers freezes first N layers."""
        fine_tuner = FineTuner(
            model=multi_layer_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        fine_tuner._freeze_first_n_layers(multi_layer_model, n=3)

        layer_names = []
        for name, _ in multi_layer_model.named_parameters():
            layer_base = name.split(".")[0]
            if layer_base not in layer_names:
                layer_names.append(layer_base)

        for name, param in multi_layer_model.named_parameters():
            layer_base = name.split(".")[0]
            layer_idx = layer_names.index(layer_base)
            if layer_idx < 3:
                assert not param.requires_grad, f"{name} should be frozen"
            else:
                assert param.requires_grad, f"{name} should be trainable"

    def test_freeze_first_n_layers_keeps_later_layers_trainable(
        self, multi_layer_model, basic_hyper_parameters, working_root_dir
    ):
        """Test _freeze_first_n_layers keeps layers after N trainable."""
        fine_tuner = FineTuner(
            model=multi_layer_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        fine_tuner._freeze_first_n_layers(multi_layer_model, n=2)

        layer_names = []
        for name, _ in multi_layer_model.named_parameters():
            layer_base = name.split(".")[0]
            if layer_base not in layer_names:
                layer_names.append(layer_base)

        trainable_count = 0
        for name, param in multi_layer_model.named_parameters():
            layer_base = name.split(".")[0]
            if layer_names.index(layer_base) >= 2:
                assert param.requires_grad
                trainable_count += 1

        assert trainable_count > 0

    def test_freeze_first_n_layers_with_zero_layers(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test _freeze_first_n_layers with n=0 leaves all trainable."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        fine_tuner._freeze_first_n_layers(simple_linear_model, n=0)

        for param in simple_linear_model.parameters():
            assert param.requires_grad

    def test_freeze_first_n_layers_with_large_n(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test _freeze_first_n_layers with n larger than layer count."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        fine_tuner._freeze_first_n_layers(simple_linear_model, n=100)

        # All should be frozen
        for param in simple_linear_model.parameters():
            assert not param.requires_grad


# =============================================================================
# _FREEZE_ALL_BUT_LAST TESTS
# =============================================================================


class TestFreezeAllButLast:
    """Test FineTuner._freeze_all_but_last internal method."""

    def test_freeze_all_but_last_freezes_earlier_layers(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test _freeze_all_but_last freezes all but last layer."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        fine_tuner._freeze_all_but_last(simple_linear_model)

        param_names = list(dict(simple_linear_model.named_parameters()).keys())
        last_layer_prefix = param_names[-1].rsplit(".", 1)[0] if param_names else ""

        frozen_count = 0
        for name, param in simple_linear_model.named_parameters():
            if not name.startswith(last_layer_prefix):
                assert not param.requires_grad
                frozen_count += 1

        assert frozen_count > 0

    def test_freeze_all_but_last_keeps_last_layer_trainable(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test _freeze_all_but_last keeps last layer trainable."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        fine_tuner._freeze_all_but_last(simple_linear_model)

        param_names = list(dict(simple_linear_model.named_parameters()).keys())
        last_layer_prefix = param_names[-1].rsplit(".", 1)[0] if param_names else ""

        trainable_count = 0
        for name, param in simple_linear_model.named_parameters():
            if name.startswith(last_layer_prefix):
                assert param.requires_grad
                trainable_count += 1

        assert trainable_count > 0

    def test_freeze_all_but_last_with_nested_model(
        self, nested_model, basic_hyper_parameters, working_root_dir
    ):
        """Test _freeze_all_but_last works with nested model structure."""
        fine_tuner = FineTuner(
            model=nested_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        fine_tuner._freeze_all_but_last(nested_model)

        # Should have some frozen and some trainable
        frozen = sum(1 for p in nested_model.parameters() if not p.requires_grad)
        trainable = sum(1 for p in nested_model.parameters() if p.requires_grad)

        assert frozen > 0
        assert trainable > 0

    def test_freeze_all_but_last_single_layer_model(self, basic_hyper_parameters, working_root_dir):
        """Test _freeze_all_but_last with single layer model."""

        class SingleLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 1)

            def forward(self, x):
                return self.linear(x)

        model = SingleLayerModel()
        fine_tuner = FineTuner(
            model=model, hyper_parameters=basic_hyper_parameters, working_root_dir=working_root_dir
        )

        fine_tuner._freeze_all_but_last(model)

        # All parameters should be trainable (only one layer)
        for param in model.parameters():
            assert param.requires_grad


# =============================================================================
# _REPLACE_OUTPUT_HEAD TESTS
# =============================================================================


class TestReplaceOutputHead:
    """Test FineTuner._replace_output_head internal method."""

    def test_replace_output_head_changes_dimensions(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test _replace_output_head changes output dimensions."""
        original_out = simple_linear_model.fc2.out_features
        new_out_channels = 5

        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        result = fine_tuner._replace_output_head(simple_linear_model, new_out_channels)

        # Find the last linear layer
        last_linear = None
        for module in result.modules():
            if isinstance(module, nn.Linear):
                last_linear = module

        assert last_linear.out_features == new_out_channels
        assert last_linear.out_features != original_out

    def test_replace_output_head_preserves_input_features(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test _replace_output_head preserves input features of replaced layer."""
        original_in_features = simple_linear_model.fc2.in_features

        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        result = fine_tuner._replace_output_head(simple_linear_model, 5)

        # Find the last linear layer
        last_linear = None
        for module in result.modules():
            if isinstance(module, nn.Linear):
                last_linear = module

        assert last_linear.in_features == original_in_features

    def test_replace_output_head_returns_model(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test _replace_output_head returns nn.Module."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        result = fine_tuner._replace_output_head(simple_linear_model, 3)

        assert isinstance(result, nn.Module)

    def test_replace_output_head_logs_change(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir, caplog
    ):
        """Test _replace_output_head logs the dimension change."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        with caplog.at_level(logging.INFO):
            fine_tuner._replace_output_head(simple_linear_model, 5)

        assert any(
            "Replaced output head" in record.message or "output" in record.message.lower()
            for record in caplog.records
        )

    def test_replace_output_head_with_nested_model(
        self, nested_model, basic_hyper_parameters, working_root_dir
    ):
        """Test _replace_output_head with nested model structure."""
        fine_tuner = FineTuner(
            model=nested_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        result = fine_tuner._replace_output_head(nested_model, 10)

        # Find the last linear layer
        last_linear = None
        for module in result.modules():
            if isinstance(module, nn.Linear):
                last_linear = module

        assert last_linear.out_features == 10

    def test_replace_output_head_new_layer_has_random_weights(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test _replace_output_head creates new layer with fresh weights."""
        original_weight = simple_linear_model.fc2.weight.clone()

        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        result = fine_tuner._replace_output_head(simple_linear_model, 5)

        # New layer should have different dimensions, hence different weights
        assert result.fc2.weight.shape != original_weight.shape

    def test_replace_output_head_no_linear_layer_warning(
        self, basic_hyper_parameters, working_root_dir, caplog
    ):
        """Test _replace_output_head warns when no linear layer found."""

        class NoLinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(1, 1, 3)

            def forward(self, x):
                return self.conv(x)

        model = NoLinearModel()
        fine_tuner = FineTuner(
            model=model, hyper_parameters=basic_hyper_parameters, working_root_dir=working_root_dir
        )

        with caplog.at_level(logging.WARNING):
            _result = fine_tuner._replace_output_head(model, 5)

        assert any("No linear layer found" in record.message for record in caplog.records)

    def test_replace_output_head_with_sequential_model(
        self, basic_hyper_parameters, working_root_dir
    ):
        """Test _replace_output_head with Sequential model."""
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))

        fine_tuner = FineTuner(
            model=model, hyper_parameters=basic_hyper_parameters, working_root_dir=working_root_dir
        )

        result = fine_tuner._replace_output_head(model, 5)

        # Find the last linear layer
        last_linear = None
        for module in result.modules():
            if isinstance(module, nn.Linear):
                last_linear = module

        assert last_linear.out_features == 5

    def test_replace_output_head_single_output_to_multi(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test replacing single output head with multi-output."""
        assert simple_linear_model.fc2.out_features == 1

        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        result = fine_tuner._replace_output_head(simple_linear_model, 10)

        assert result.fc2.out_features == 10

    def test_replace_output_head_multi_output_to_single(
        self, basic_hyper_parameters, working_root_dir
    ):
        """Test replacing multi-output head with single output."""

        class MultiOutputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(16, 32)
                self.fc2 = nn.Linear(32, 5)  # 5 outputs

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)

        model = MultiOutputModel()
        fine_tuner = FineTuner(
            model=model, hyper_parameters=basic_hyper_parameters, working_root_dir=working_root_dir
        )

        result = fine_tuner._replace_output_head(model, 1)

        assert result.fc2.out_features == 1


# =============================================================================
# _LOG_PARAMETER_STATUS TESTS
# =============================================================================


class TestLogParameterStatus:
    """Test FineTuner._log_parameter_status internal method."""

    def test_log_parameter_status_logs_info(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir, caplog
    ):
        """Test _log_parameter_status logs at INFO level."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        with caplog.at_level(logging.INFO):
            fine_tuner._log_parameter_status(simple_linear_model)

        assert len(caplog.records) > 0

    def test_log_parameter_status_includes_trainable_count(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir, caplog
    ):
        """Test _log_parameter_status includes trainable parameter count."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        with caplog.at_level(logging.INFO):
            fine_tuner._log_parameter_status(simple_linear_model)

        assert any("trainable" in record.message.lower() for record in caplog.records)

    def test_log_parameter_status_includes_frozen_count(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir, caplog
    ):
        """Test _log_parameter_status includes frozen parameter count."""
        # Freeze some parameters
        for param in list(simple_linear_model.parameters())[:2]:
            param.requires_grad = False

        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        with caplog.at_level(logging.INFO):
            fine_tuner._log_parameter_status(simple_linear_model)

        assert any("frozen" in record.message.lower() for record in caplog.records)

    def test_log_parameter_status_includes_percentages(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir, caplog
    ):
        """Test _log_parameter_status includes percentage information."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        with caplog.at_level(logging.INFO):
            fine_tuner._log_parameter_status(simple_linear_model)

        assert any("%" in record.message for record in caplog.records)

    def test_log_parameter_status_all_trainable(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir, caplog
    ):
        """Test _log_parameter_status with all trainable parameters."""
        for param in simple_linear_model.parameters():
            param.requires_grad = True

        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        with caplog.at_level(logging.INFO):
            fine_tuner._log_parameter_status(simple_linear_model)

        # Should show 100% trainable
        assert any("100" in record.message for record in caplog.records)

    def test_log_parameter_status_all_frozen(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir, caplog
    ):
        """Test _log_parameter_status with all frozen parameters."""
        for param in simple_linear_model.parameters():
            param.requires_grad = False

        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        with caplog.at_level(logging.INFO):
            fine_tuner._log_parameter_status(simple_linear_model)

        # Should show 0% trainable or 100% frozen
        log_messages = [r.message for r in caplog.records]
        assert any("0" in msg or "100" in msg for msg in log_messages)


# =============================================================================
# EDGE CASES AND ERROR HANDLING TESTS
# =============================================================================


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_empty_hyper_parameters(self, simple_linear_model, working_root_dir):
        """Test FineTuner with empty hyper_parameters."""
        fine_tuner = FineTuner(
            model=simple_linear_model, hyper_parameters={}, working_root_dir=working_root_dir
        )

        assert fine_tuner.original_out_channels == 1

        result = fine_tuner.prepare_for_finetuning(freeze_strategy=FreezeStrategy.NONE)

        assert isinstance(result, nn.Module)

    def test_model_with_no_parameters(self, basic_hyper_parameters, working_root_dir):
        """Test handling model with no parameters raises ZeroDivisionError in _log_parameter_status."""

        class NoParamModel(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        model = NoParamModel()
        fine_tuner = FineTuner(
            model=model, hyper_parameters=basic_hyper_parameters, working_root_dir=working_root_dir
        )

        # The current implementation raises ZeroDivisionError for models with no parameters
        # This tests the actual behavior of the code
        with pytest.raises(ZeroDivisionError):
            fine_tuner.prepare_for_finetuning(freeze_strategy=FreezeStrategy.NONE)

    def test_prepare_for_finetuning_with_none_new_out_channels(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test prepare_for_finetuning with None new_out_channels."""
        original_out = simple_linear_model.fc2.out_features

        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        result = fine_tuner.prepare_for_finetuning(
            new_out_channels=None, freeze_strategy=FreezeStrategy.NONE
        )

        # Output dimensions should remain unchanged
        assert result.fc2.out_features == original_out

    def test_freeze_encoder_layers_with_no_matching_layers(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test _freeze_encoder_layers with no encoder-like layers."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        fine_tuner._freeze_encoder_layers(simple_linear_model)

        # All parameters should be trainable (no conv/message/etc layers)
        for param in simple_linear_model.parameters():
            assert param.requires_grad

    def test_prepare_multiple_times(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test calling prepare_for_finetuning multiple times."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        # First call - freeze encoder
        _result1 = fine_tuner.prepare_for_finetuning(freeze_strategy=FreezeStrategy.ALL_BUT_LAST)

        # Second call - unfreeze all
        result2 = fine_tuner.prepare_for_finetuning(freeze_strategy=FreezeStrategy.NONE)

        # All parameters should be trainable now
        for param in result2.parameters():
            assert param.requires_grad

    def test_very_deep_model(self, basic_hyper_parameters, working_root_dir):
        """Test with very deep model."""
        layers = []
        for i in range(20):
            layers.append(nn.Linear(32, 32))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(32, 1))

        model = nn.Sequential(*layers)

        fine_tuner = FineTuner(
            model=model, hyper_parameters=basic_hyper_parameters, working_root_dir=working_root_dir
        )

        result = fine_tuner.prepare_for_finetuning(
            freeze_strategy=FreezeStrategy.ENCODER_PARTIAL, freeze_layers=10
        )

        assert isinstance(result, nn.Module)

    def test_model_with_batch_norm(self, basic_hyper_parameters, working_root_dir):
        """Test with model containing BatchNorm layers."""

        class ModelWithBN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Linear(16, 32)
                self.bn1 = nn.BatchNorm1d(32)
                self.fc_out = nn.Linear(32, 1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                return self.fc_out(x)

        model = ModelWithBN()
        fine_tuner = FineTuner(
            model=model, hyper_parameters=basic_hyper_parameters, working_root_dir=working_root_dir
        )

        result = fine_tuner.prepare_for_finetuning(freeze_strategy=FreezeStrategy.ENCODER)

        assert isinstance(result, nn.Module)
        assert result.training is True

    def test_model_with_dropout(self, basic_hyper_parameters, working_root_dir):
        """Test with model containing Dropout layers."""

        class ModelWithDropout(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(16, 32)
                self.dropout = nn.Dropout(0.5)
                self.fc2 = nn.Linear(32, 1)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                return self.fc2(x)

        model = ModelWithDropout()
        fine_tuner = FineTuner(
            model=model, hyper_parameters=basic_hyper_parameters, working_root_dir=working_root_dir
        )

        result = fine_tuner.prepare_for_finetuning(freeze_strategy=FreezeStrategy.NONE)

        assert isinstance(result, nn.Module)
        assert result.training is True


# =============================================================================
# DEVICE HANDLING TESTS
# =============================================================================


class TestDeviceHandling:
    """Test device handling functionality."""

    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.CheckpointManager")
    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.ModelLoader")
    def test_from_checkpoint_with_cpu_device(
        self,
        mock_loader_class,
        mock_cm_class,
        simple_linear_model,
        mock_checkpoint,
        working_root_dir,
    ):
        """Test from_checkpoint with CPU device."""
        checkpoint_path = working_root_dir / "model.pt"
        cpu_device = torch.device("cpu")

        mock_cm_instance = MagicMock()
        mock_cm_instance._resolve_checkpoint_path.return_value = checkpoint_path
        mock_cm_instance.load.return_value = mock_checkpoint
        mock_cm_class.return_value = mock_cm_instance

        mock_loader_class.load_from_checkpoint.return_value = (simple_linear_model, {})

        _fine_tuner = FineTuner.from_checkpoint(
            checkpoint_path, working_root_dir=working_root_dir, device=cpu_device
        )

        mock_loader_class.load_from_checkpoint.assert_called_once_with(
            checkpoint_path=checkpoint_path, working_root_dir=working_root_dir, device=cpu_device
        )

    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.CheckpointManager")
    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.ModelLoader")
    def test_from_checkpoint_with_none_device(
        self,
        mock_loader_class,
        mock_cm_class,
        simple_linear_model,
        mock_checkpoint,
        working_root_dir,
    ):
        """Test from_checkpoint with None device (auto-detect)."""
        checkpoint_path = working_root_dir / "model.pt"

        mock_cm_instance = MagicMock()
        mock_cm_instance._resolve_checkpoint_path.return_value = checkpoint_path
        mock_cm_instance.load.return_value = mock_checkpoint
        mock_cm_class.return_value = mock_cm_instance

        mock_loader_class.load_from_checkpoint.return_value = (simple_linear_model, {})

        _fine_tuner = FineTuner.from_checkpoint(
            checkpoint_path, working_root_dir=working_root_dir, device=None
        )

        mock_loader_class.load_from_checkpoint.assert_called_once_with(
            checkpoint_path=checkpoint_path, working_root_dir=working_root_dir, device=None
        )

    def test_model_on_cpu_after_prepare(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test model remains on CPU after prepare_for_finetuning."""
        simple_linear_model = simple_linear_model.cpu()

        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        result = fine_tuner.prepare_for_finetuning(freeze_strategy=FreezeStrategy.NONE)

        # Check model is on CPU
        for param in result.parameters():
            assert param.device.type == "cpu"


# =============================================================================
# LOGGING TESTS
# =============================================================================


class TestLogging:
    """Test logging behavior."""

    def test_logger_is_module_logger(self):
        """Test that logger is set up correctly."""
        assert (
            fine_tuner_logger.name
            == "milia_pipeline.models.post_training.transfer_learning.fine_tuner"
        )

    def test_freeze_encoder_layers_logs_debug(
        self, gnn_like_model, basic_hyper_parameters, working_root_dir, caplog
    ):
        """Test _freeze_encoder_layers logs at DEBUG level."""
        fine_tuner = FineTuner(
            model=gnn_like_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        with caplog.at_level(logging.DEBUG):
            fine_tuner._freeze_encoder_layers(gnn_like_model)

        # Should have debug logs for frozen parameters
        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(debug_records) > 0


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================


class TestIntegrationStyle:
    """Integration-style tests for complete workflows."""

    def test_full_finetuning_workflow(
        self, multi_layer_model, basic_hyper_parameters, working_root_dir
    ):
        """Test complete fine-tuning workflow."""
        # Initialize FineTuner
        fine_tuner = FineTuner(
            model=multi_layer_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        # Prepare for fine-tuning with new output dimension
        model = fine_tuner.prepare_for_finetuning(
            new_out_channels=3,
            freeze_strategy=FreezeStrategy.ENCODER,
        )

        # Verify model is in training mode
        assert model.training is True

        # Verify some parameters are frozen
        frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        trainable = sum(1 for p in model.parameters() if p.requires_grad)

        assert frozen > 0
        assert trainable > 0

        # Verify output dimension changed
        last_linear = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module

        assert last_linear.out_features == 3

    def test_transfer_learning_pattern(
        self, gnn_like_model, basic_hyper_parameters, working_root_dir
    ):
        """Test typical transfer learning pattern."""
        # Step 1: Create FineTuner from model
        fine_tuner = FineTuner(
            model=gnn_like_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        # Step 2: Freeze encoder, replace head
        model = fine_tuner.prepare_for_finetuning(
            new_out_channels=5, freeze_strategy=FreezeStrategy.ENCODER
        )

        # Step 3: Verify encoder is frozen
        for name, param in model.named_parameters():
            if any(x in name.lower() for x in ["conv", "message", "propagate", "aggr"]):
                assert not param.requires_grad, f"Encoder layer {name} should be frozen"

        # Step 4: Verify output head is trainable and has correct dimensions
        assert model.output.out_features == 5
        assert model.output.weight.requires_grad
        assert model.output.bias.requires_grad

    def test_progressive_unfreezing_pattern(
        self, multi_layer_model, basic_hyper_parameters, working_root_dir
    ):
        """Test progressive unfreezing pattern."""
        fine_tuner = FineTuner(
            model=multi_layer_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        # Stage 1: Freeze all but last
        model = fine_tuner.prepare_for_finetuning(freeze_strategy=FreezeStrategy.ALL_BUT_LAST)

        trainable_stage1 = sum(1 for p in model.parameters() if p.requires_grad)

        # Stage 2: Partial unfreeze
        model = fine_tuner.prepare_for_finetuning(
            freeze_strategy=FreezeStrategy.ENCODER_PARTIAL, freeze_layers=1
        )

        trainable_stage2 = sum(1 for p in model.parameters() if p.requires_grad)

        # Stage 3: Full unfreeze
        model = fine_tuner.prepare_for_finetuning(freeze_strategy=FreezeStrategy.NONE)

        trainable_stage3 = sum(1 for p in model.parameters() if p.requires_grad)

        # Each stage should have more trainable parameters
        assert trainable_stage1 <= trainable_stage2 <= trainable_stage3

    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.CheckpointManager")
    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.ModelLoader")
    def test_from_checkpoint_to_finetuning(
        self,
        mock_loader_class,
        mock_cm_class,
        simple_linear_model,
        mock_checkpoint,
        working_root_dir,
    ):
        """Test complete workflow from checkpoint to fine-tuning."""
        checkpoint_path = working_root_dir / "pretrained.pt"

        mock_cm_instance = MagicMock()
        mock_cm_instance._resolve_checkpoint_path.return_value = checkpoint_path
        mock_cm_instance.load.return_value = mock_checkpoint
        mock_cm_class.return_value = mock_cm_instance

        mock_loader_class.load_from_checkpoint.return_value = (simple_linear_model, {})

        # Step 1: Load from checkpoint
        fine_tuner = FineTuner.from_checkpoint(checkpoint_path, working_root_dir=working_root_dir)

        assert isinstance(fine_tuner, FineTuner)
        assert fine_tuner.model is simple_linear_model

        # Step 2: Prepare for fine-tuning
        model = fine_tuner.prepare_for_finetuning(
            new_out_channels=10, freeze_strategy=FreezeStrategy.ALL_BUT_LAST
        )

        assert model.training is True
        assert model.fc2.out_features == 10


# =============================================================================
# PARAMETER STATE TESTS
# =============================================================================


class TestParameterState:
    """Test parameter state management."""

    def test_freeze_preserves_parameter_values(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test that freezing preserves parameter values."""
        original_weights = {
            name: param.clone() for name, param in simple_linear_model.named_parameters()
        }

        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        fine_tuner._apply_freeze_strategy(simple_linear_model, FreezeStrategy.ALL_BUT_LAST)

        for name, param in simple_linear_model.named_parameters():
            assert torch.equal(param, original_weights[name])

    def test_unfreeze_preserves_parameter_values(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test that unfreezing preserves parameter values."""
        # First freeze all
        for param in simple_linear_model.parameters():
            param.requires_grad = False

        original_weights = {
            name: param.clone() for name, param in simple_linear_model.named_parameters()
        }

        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        fine_tuner._apply_freeze_strategy(simple_linear_model, FreezeStrategy.NONE)

        for name, param in simple_linear_model.named_parameters():
            assert torch.equal(param, original_weights[name])

    def test_new_head_has_independent_weights(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test new output head has independent weights."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        # Replace head twice
        model1 = fine_tuner._replace_output_head(simple_linear_model, 5)
        weights1 = model1.fc2.weight.clone()

        # Create new model and replace again
        new_model = type(simple_linear_model)()
        model2 = fine_tuner._replace_output_head(new_model, 5)
        weights2 = model2.fc2.weight.clone()

        # Weights should be different (randomly initialized)
        assert not torch.equal(weights1, weights2)


# =============================================================================
# TYPE ANNOTATION TESTS
# =============================================================================


class TestTypeAnnotations:
    """Test that type annotations are honored."""

    def test_init_accepts_nn_module(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test __init__ accepts nn.Module."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )
        assert isinstance(fine_tuner.model, nn.Module)

    def test_init_accepts_dict_hyper_parameters(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test __init__ accepts Dict[str, Any] for hyper_parameters."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )
        assert isinstance(fine_tuner.hyper_parameters, dict)

    def test_prepare_returns_nn_module(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test prepare_for_finetuning returns nn.Module."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        result = fine_tuner.prepare_for_finetuning()

        assert isinstance(result, nn.Module)

    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.CheckpointManager")
    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.ModelLoader")
    def test_from_checkpoint_accepts_path(self, mock_loader_class, mock_cm_class, working_root_dir):
        """Test from_checkpoint accepts Path type."""
        checkpoint_path = working_root_dir / "model.pt"

        mock_cm_instance = MagicMock()
        mock_cm_instance._resolve_checkpoint_path.return_value = checkpoint_path
        mock_cm_instance.load.return_value = {"hyper_parameters": {}}
        mock_cm_class.return_value = mock_cm_instance

        mock_loader_class.load_from_checkpoint.return_value = (nn.Linear(10, 1), {})

        fine_tuner = FineTuner.from_checkpoint(checkpoint_path, working_root_dir=working_root_dir)

        assert isinstance(fine_tuner, FineTuner)

    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.CheckpointManager")
    @patch("milia_pipeline.models.post_training.transfer_learning.fine_tuner.ModelLoader")
    def test_from_checkpoint_accepts_string(
        self, mock_loader_class, mock_cm_class, working_root_dir
    ):
        """Test from_checkpoint accepts string path."""
        checkpoint_path = str(working_root_dir / "model.pt")
        resolved_path = working_root_dir / "model.pt"

        mock_cm_instance = MagicMock()
        mock_cm_instance._resolve_checkpoint_path.return_value = resolved_path
        mock_cm_instance.load.return_value = {"hyper_parameters": {}}
        mock_cm_class.return_value = mock_cm_instance

        mock_loader_class.load_from_checkpoint.return_value = (nn.Linear(10, 1), {})

        fine_tuner = FineTuner.from_checkpoint(checkpoint_path, working_root_dir=working_root_dir)

        assert isinstance(fine_tuner, FineTuner)


# =============================================================================
# GRADIENTS AND MEMORY TESTS
# =============================================================================


class TestGradientsAndMemory:
    """Test gradient and memory handling."""

    def test_frozen_parameters_accumulate_no_gradients(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test frozen parameters don't accumulate gradients during forward pass."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        model = fine_tuner.prepare_for_finetuning(freeze_strategy=FreezeStrategy.ALL_BUT_LAST)

        # Forward pass
        x = torch.randn(4, 16)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check frozen parameters have no gradients
        param_names = list(dict(model.named_parameters()).keys())
        last_layer_prefix = param_names[-1].rsplit(".", 1)[0] if param_names else ""

        for name, param in model.named_parameters():
            if not name.startswith(last_layer_prefix):
                assert param.grad is None, f"{name} should have no gradient"

    def test_trainable_parameters_accumulate_gradients(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test trainable parameters accumulate gradients during forward pass."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        model = fine_tuner.prepare_for_finetuning(freeze_strategy=FreezeStrategy.NONE)

        # Forward pass
        x = torch.randn(4, 16)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} should have gradient"

    def test_model_forward_works_after_head_replacement(
        self, simple_linear_model, basic_hyper_parameters, working_root_dir
    ):
        """Test model forward pass works after replacing output head."""
        fine_tuner = FineTuner(
            model=simple_linear_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        model = fine_tuner.prepare_for_finetuning(
            new_out_channels=5, freeze_strategy=FreezeStrategy.NONE
        )

        # Forward pass
        x = torch.randn(4, 16)
        output = model(x)

        assert output.shape == (4, 5)

    def test_backward_pass_works_with_frozen_layers(
        self, gnn_like_model, basic_hyper_parameters, working_root_dir
    ):
        """Test backward pass works with some frozen layers."""
        fine_tuner = FineTuner(
            model=gnn_like_model,
            hyper_parameters=basic_hyper_parameters,
            working_root_dir=working_root_dir,
        )

        model = fine_tuner.prepare_for_finetuning(freeze_strategy=FreezeStrategy.ENCODER)

        # Forward and backward pass
        x = torch.randn(4, 16)
        output = model(x)
        loss = output.sum()

        # Should not raise
        loss.backward()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
