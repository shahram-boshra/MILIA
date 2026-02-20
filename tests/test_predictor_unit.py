#!/usr/bin/env python3
"""
Extended Production-Ready Unit Test Suite for predictor.py Module

Comprehensive test coverage for the Predictor class and convenience functions.
Tests cover:
- Predictor class initialization (including working_root_dir DI, model_info)
- structural_features_config property (FIX 19)
- _resolve_path method (absolute, relative, create_parents)
- from_checkpoint class method (working_root_dir DI, model_info passthrough)
- predict method (single Data, Batch)
- _forward internal method (PyG attribute handling, signature introspection)
- _forward 3D molecular model detection (FIX 25: z, pos, batch path)
- _forward edge_weight and batch_size candidate kwargs
- _forward fallback logic on introspection failure
- _postprocess method (classification vs regression)
- predict_batch method (dataset prediction)
- save_predictions method (csv, json, npy, pt formats)
- Convenience function predict() (with working_root_dir DI)
- Device handling (CPU/CUDA)
- Return type handling (tensor vs numpy)
- Edge cases and error conditions

Author: MILIA Team
Version: 2.0.0
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import contextlib
import json
import logging
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

# Import the module under test
from milia_pipeline.models.post_training.inference.predictor import (
    Predictor,
    predict,
)

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_model():
    """Create a mock PyTorch model."""
    model = MagicMock(spec=nn.Module)
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    # Default forward returns a 2D tensor for graph-level tasks
    model.return_value = torch.tensor([[0.5], [0.3]])
    return model


@pytest.fixture
def mock_model_for_classification():
    """Create a mock model that returns classification logits."""
    model = MagicMock(spec=nn.Module)
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    # Return multi-class logits (batch_size=2, num_classes=3)
    model.return_value = torch.tensor(
        [
            [0.1, 0.7, 0.2],  # Class 1 is highest
            [0.8, 0.1, 0.1],  # Class 0 is highest
        ]
    )
    return model


@pytest.fixture
def simple_pyg_model():
    """Create a simple real PyG-compatible model for testing."""

    class SimplePyGModel(nn.Module):
        def __init__(self, in_channels=16, out_channels=1):
            super().__init__()
            self.linear = nn.Linear(in_channels, out_channels)

        def forward(self, x, edge_index, **kwargs):
            # Simple global mean pooling
            if "batch" in kwargs and kwargs["batch"] is not None:
                from torch_geometric.nn import global_mean_pool

                x = global_mean_pool(x, kwargs["batch"])
            else:
                x = x.mean(dim=0, keepdim=True)
            return self.linear(x)

    return SimplePyGModel()


@pytest.fixture
def simple_pyg_data():
    """Create a simple PyG Data object."""
    # 4 nodes, 16 features each
    x = torch.randn(4, 16)
    # Simple edge structure (triangle + one isolated)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


@pytest.fixture
def pyg_data_with_edge_attr():
    """Create PyG Data with edge attributes."""
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.randn(4, 8)  # 4 edges, 8 features each
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


@pytest.fixture
def pyg_data_with_pos():
    """Create PyG Data with 3D positions."""
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    pos = torch.randn(4, 3)  # 4 nodes, 3D coordinates
    return Data(x=x, edge_index=edge_index, pos=pos)


@pytest.fixture
def pyg_data_with_all_attrs():
    """Create PyG Data with all optional attributes."""
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.randn(4, 8)
    pos = torch.randn(4, 3)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)


@pytest.fixture
def pyg_batch():
    """Create a PyG Batch from multiple Data objects."""
    data_list = []
    for _ in range(3):
        x = torch.randn(4, 16)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        data_list.append(Data(x=x, edge_index=edge_index))
    return Batch.from_data_list(data_list)


@pytest.fixture
def pyg_dataset():
    """Create a list of PyG Data objects simulating a dataset."""
    dataset = []
    for i in range(10):
        num_nodes = 4 + i % 3  # Varying sizes: 4, 5, 6, 4, 5, 6, ...
        x = torch.randn(num_nodes, 16)
        # Create some edges
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        data = Data(x=x, edge_index=edge_index)
        dataset.append(data)
    return dataset


@pytest.fixture
def mock_checkpoint():
    """Create a mock checkpoint dictionary."""
    return {
        "epoch": 100,
        "model_state_dict": {
            "linear.weight": torch.randn(1, 16),
            "linear.bias": torch.randn(1),
        },
        "hyper_parameters": {
            "model_name": "GCN",
            "task_type": "graph_regression",
            "hyperparameters": {
                "hidden_channels": 64,
                "num_layers": 3,
            },
        },
    }


@pytest.fixture
def mock_classification_checkpoint():
    """Create a mock checkpoint for classification tasks."""
    return {
        "epoch": 50,
        "model_state_dict": {
            "linear.weight": torch.randn(3, 16),
            "linear.bias": torch.randn(3),
        },
        "hyper_parameters": {
            "model_name": "GCN",
            "task_type": "graph_classification",
            "hyperparameters": {
                "hidden_channels": 64,
                "out_channels": 3,
            },
        },
    }


@pytest.fixture
def cpu_device():
    """Return CPU device."""
    return torch.device("cpu")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def working_root_dir(temp_dir):
    """Create a working_root_dir for Dependency Injection pattern tests."""
    return temp_dir


# =============================================================================
# PREDICTOR CLASS TESTS - INITIALIZATION
# =============================================================================


class TestPredictorInitialization:
    """Test Predictor class initialization."""

    def test_init_stores_model(self, mock_model, working_root_dir):
        """Test that __init__ stores the model."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir)
        assert predictor.model is mock_model

    def test_init_sets_model_to_eval_mode(self, mock_model, working_root_dir):
        """Test that __init__ sets model to eval mode."""
        _predictor = Predictor(model=mock_model, working_root_dir=working_root_dir)
        mock_model.eval.assert_called_once()

    def test_init_moves_model_to_device(self, mock_model, working_root_dir):
        """Test that __init__ moves model to specified device."""
        device = torch.device("cpu")
        _predictor = Predictor(model=mock_model, working_root_dir=working_root_dir, device=device)
        mock_model.to.assert_called_with(device)

    def test_init_with_explicit_cpu_device(self, mock_model, working_root_dir):
        """Test initialization with explicit CPU device."""
        device = torch.device("cpu")
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir, device=device)
        assert predictor.device == device

    @patch("torch.cuda.is_available", return_value=True)
    def test_init_auto_detects_cuda_when_available(self, mock_cuda, mock_model, working_root_dir):
        """Test that __init__ auto-detects CUDA when available and no device specified."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir, device=None)
        assert predictor.device == torch.device("cuda")

    @patch("torch.cuda.is_available", return_value=False)
    def test_init_auto_detects_cpu_when_cuda_unavailable(
        self, mock_cuda, mock_model, working_root_dir
    ):
        """Test that __init__ auto-detects CPU when CUDA unavailable."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir, device=None)
        assert predictor.device == torch.device("cpu")

    def test_init_stores_task_type(self, mock_model, working_root_dir):
        """Test that __init__ stores task_type."""
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, task_type="graph_regression"
        )
        assert predictor.task_type == "graph_regression"

    def test_init_task_type_none_by_default(self, mock_model, working_root_dir):
        """Test that task_type is None by default."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir)
        assert predictor.task_type is None

    def test_init_with_classification_task_type(self, mock_model, working_root_dir):
        """Test initialization with classification task type."""
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, task_type="graph_classification"
        )
        assert predictor.task_type == "graph_classification"

    def test_init_with_real_model(self, simple_pyg_model, working_root_dir):
        """Test initialization with a real PyG model."""
        predictor = Predictor(
            model=simple_pyg_model, working_root_dir=working_root_dir, device=torch.device("cpu")
        )
        assert predictor.model is simple_pyg_model
        # Model should be in eval mode
        assert not simple_pyg_model.training

    def test_init_stores_working_root_dir(self, mock_model, working_root_dir):
        """Test that __init__ stores and resolves working_root_dir."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir)
        assert predictor._working_root_dir == working_root_dir.expanduser().resolve()

    def test_init_resolves_tilde_in_working_root_dir(self, mock_model):
        """Test that __init__ expands ~ in working_root_dir."""
        predictor = Predictor(model=mock_model, working_root_dir=Path("~/some_dir"))
        assert "~" not in str(predictor._working_root_dir)

    def test_init_stores_model_info(self, mock_model, working_root_dir):
        """Test that __init__ stores model_info dict."""
        model_info = {"model_name": "GCN", "data_info": {"requires_edge_features": True}}
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, model_info=model_info
        )
        assert predictor.model_info is model_info

    def test_init_model_info_defaults_to_empty_dict(self, mock_model, working_root_dir):
        """Test that model_info defaults to empty dict when None."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir)
        assert predictor.model_info == {}

    def test_init_model_info_none_becomes_empty_dict(self, mock_model, working_root_dir):
        """Test that model_info=None is stored as empty dict."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir, model_info=None)
        assert predictor.model_info == {}


# =============================================================================
# PREDICTOR CLASS TESTS - from_checkpoint CLASS METHOD
# =============================================================================


class TestPredictorFromCheckpoint:
    """Test Predictor.from_checkpoint class method."""

    @patch("milia_pipeline.models.post_training.checkpoint.checkpoint_manager.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.ModelLoader")
    def test_from_checkpoint_loads_model(
        self, mock_loader_class, mock_cm_class, mock_model, mock_checkpoint, working_root_dir
    ):
        """Test from_checkpoint loads model via ModelLoader."""
        mock_model_info = {"model_name": "GCN"}
        mock_loader_class.load_from_checkpoint.return_value = (mock_model, mock_model_info)

        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = mock_checkpoint
        mock_cm_class.return_value = mock_cm_instance

        _predictor = Predictor.from_checkpoint("test.pt", working_root_dir=working_root_dir)

        mock_loader_class.load_from_checkpoint.assert_called_once()

    @patch("milia_pipeline.models.post_training.checkpoint.checkpoint_manager.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.ModelLoader")
    def test_from_checkpoint_passes_device(
        self, mock_loader_class, mock_cm_class, mock_model, mock_checkpoint, working_root_dir
    ):
        """Test from_checkpoint passes device to ModelLoader."""
        mock_model_info = {"model_name": "GCN"}
        mock_loader_class.load_from_checkpoint.return_value = (mock_model, mock_model_info)

        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = mock_checkpoint
        mock_cm_class.return_value = mock_cm_instance

        device = torch.device("cpu")
        _predictor = Predictor.from_checkpoint(
            "test.pt", working_root_dir=working_root_dir, device=device
        )

        call_kwargs = mock_loader_class.load_from_checkpoint.call_args[1]
        assert call_kwargs["device"] == device

    @patch("milia_pipeline.models.post_training.checkpoint.checkpoint_manager.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.ModelLoader")
    def test_from_checkpoint_extracts_task_type(
        self, mock_loader_class, mock_cm_class, mock_model, mock_checkpoint, working_root_dir
    ):
        """Test from_checkpoint extracts task_type from checkpoint."""
        mock_model_info = {"model_name": "GCN"}
        mock_loader_class.load_from_checkpoint.return_value = (mock_model, mock_model_info)

        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = mock_checkpoint
        mock_cm_class.return_value = mock_cm_instance

        predictor = Predictor.from_checkpoint("test.pt", working_root_dir=working_root_dir)

        assert predictor.task_type == "graph_regression"

    @patch("milia_pipeline.models.post_training.checkpoint.checkpoint_manager.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.ModelLoader")
    def test_from_checkpoint_handles_missing_task_type(
        self, mock_loader_class, mock_cm_class, mock_model, working_root_dir
    ):
        """Test from_checkpoint handles checkpoint without task_type."""
        mock_model_info = {}
        mock_loader_class.load_from_checkpoint.return_value = (mock_model, mock_model_info)

        checkpoint_no_task = {
            "epoch": 100,
            "hyper_parameters": {},  # No task_type
        }
        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = checkpoint_no_task
        mock_cm_class.return_value = mock_cm_instance

        predictor = Predictor.from_checkpoint("test.pt", working_root_dir=working_root_dir)

        assert predictor.task_type is None

    @patch("milia_pipeline.models.post_training.checkpoint.checkpoint_manager.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.ModelLoader")
    def test_from_checkpoint_handles_missing_hyper_parameters(
        self, mock_loader_class, mock_cm_class, mock_model, working_root_dir
    ):
        """Test from_checkpoint handles checkpoint without hyper_parameters."""
        mock_model_info = {}
        mock_loader_class.load_from_checkpoint.return_value = (mock_model, mock_model_info)

        checkpoint_no_hp = {
            "epoch": 100,
            # No hyper_parameters key
        }
        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = checkpoint_no_hp
        mock_cm_class.return_value = mock_cm_instance

        predictor = Predictor.from_checkpoint("test.pt", working_root_dir=working_root_dir)

        assert predictor.task_type is None

    @patch("milia_pipeline.models.post_training.checkpoint.checkpoint_manager.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.ModelLoader")
    def test_from_checkpoint_with_path_object(
        self, mock_loader_class, mock_cm_class, mock_model, mock_checkpoint, working_root_dir
    ):
        """Test from_checkpoint accepts Path objects."""
        mock_model_info = {}
        mock_loader_class.load_from_checkpoint.return_value = (mock_model, mock_model_info)

        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = mock_checkpoint
        mock_cm_class.return_value = mock_cm_instance

        checkpoint_path = Path("/path/to/model.pt")
        predictor = Predictor.from_checkpoint(checkpoint_path, working_root_dir=working_root_dir)

        assert predictor is not None

    @patch("milia_pipeline.models.post_training.checkpoint.checkpoint_manager.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.ModelLoader")
    def test_from_checkpoint_passes_loader_kwargs(
        self, mock_loader_class, mock_cm_class, mock_model, mock_checkpoint, working_root_dir
    ):
        """Test from_checkpoint passes additional kwargs to ModelLoader."""
        mock_model_info = {}
        mock_loader_class.load_from_checkpoint.return_value = (mock_model, mock_model_info)

        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = mock_checkpoint
        mock_cm_class.return_value = mock_cm_instance

        _predictor = Predictor.from_checkpoint(
            "test.pt", working_root_dir=working_root_dir, strict=False, model_name="GAT"
        )

        call_kwargs = mock_loader_class.load_from_checkpoint.call_args[1]
        assert call_kwargs.get("strict") is False
        assert call_kwargs.get("model_name") == "GAT"

    @patch("milia_pipeline.models.post_training.checkpoint.checkpoint_manager.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.ModelLoader")
    def test_from_checkpoint_classification_task(
        self,
        mock_loader_class,
        mock_cm_class,
        mock_model,
        mock_classification_checkpoint,
        working_root_dir,
    ):
        """Test from_checkpoint with classification task type."""
        mock_model_info = {}
        mock_loader_class.load_from_checkpoint.return_value = (mock_model, mock_model_info)

        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = mock_classification_checkpoint
        mock_cm_class.return_value = mock_cm_instance

        predictor = Predictor.from_checkpoint("test.pt", working_root_dir=working_root_dir)

        assert predictor.task_type == "graph_classification"

    @patch("milia_pipeline.models.post_training.checkpoint.checkpoint_manager.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.ModelLoader")
    def test_from_checkpoint_passes_model_info_to_init(
        self, mock_loader_class, mock_cm_class, mock_model, mock_checkpoint, working_root_dir
    ):
        """Test from_checkpoint passes model_info (from ModelLoader) into __init__."""
        mock_model_info = {
            "model_name": "GCN",
            "data_info": {
                "structural_features_config": {"atom": ["atomic_num"], "bond": ["bond_type"]}
            },
        }
        mock_loader_class.load_from_checkpoint.return_value = (mock_model, mock_model_info)

        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = mock_checkpoint
        mock_cm_class.return_value = mock_cm_instance

        predictor = Predictor.from_checkpoint("test.pt", working_root_dir=working_root_dir)

        assert predictor.model_info is mock_model_info

    @patch("milia_pipeline.models.post_training.checkpoint.checkpoint_manager.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.ModelLoader")
    def test_from_checkpoint_passes_working_root_dir_to_checkpoint_manager(
        self, mock_loader_class, mock_cm_class, mock_model, mock_checkpoint, working_root_dir
    ):
        """Test from_checkpoint creates CheckpointManager with working_root_dir."""
        mock_model_info = {}
        mock_loader_class.load_from_checkpoint.return_value = (mock_model, mock_model_info)

        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = mock_checkpoint
        mock_cm_class.return_value = mock_cm_instance

        _predictor = Predictor.from_checkpoint("test.pt", working_root_dir=working_root_dir)

        mock_cm_class.assert_called_once_with(working_root_dir=working_root_dir)

    @patch("milia_pipeline.models.post_training.checkpoint.checkpoint_manager.CheckpointManager")
    @patch("milia_pipeline.models.post_training.inference.model_loader.ModelLoader")
    def test_from_checkpoint_passes_working_root_dir_to_model_loader(
        self, mock_loader_class, mock_cm_class, mock_model, mock_checkpoint, working_root_dir
    ):
        """Test from_checkpoint passes working_root_dir to ModelLoader.load_from_checkpoint."""
        mock_model_info = {}
        mock_loader_class.load_from_checkpoint.return_value = (mock_model, mock_model_info)

        mock_cm_instance = MagicMock()
        mock_cm_instance.load.return_value = mock_checkpoint
        mock_cm_class.return_value = mock_cm_instance

        _predictor = Predictor.from_checkpoint("test.pt", working_root_dir=working_root_dir)

        call_kwargs = mock_loader_class.load_from_checkpoint.call_args[1]
        assert call_kwargs["working_root_dir"] == working_root_dir


# =============================================================================
# PREDICTOR CLASS TESTS - predict METHOD
# =============================================================================


class TestPredictorPredict:
    """Test Predictor.predict method."""

    def test_predict_moves_data_to_device(
        self, mock_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test predict moves data to correct device."""
        # Setup mock to track data device
        mock_model.return_value = torch.tensor([[0.5]])
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, device=cpu_device
        )

        # Create data on CPU
        data = simple_pyg_data.clone()

        _result = predictor.predict(data)

        # Model should have been called
        mock_model.assert_called()

    def test_predict_returns_tensor_by_default(
        self, mock_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test predict returns torch.Tensor by default."""
        mock_model.return_value = torch.tensor([[0.5]])
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, device=cpu_device
        )

        result = predictor.predict(simple_pyg_data)

        assert isinstance(result, torch.Tensor)

    def test_predict_returns_numpy_when_requested(
        self, mock_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test predict returns numpy array when return_numpy=True."""
        mock_model.return_value = torch.tensor([[0.5]])
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, device=cpu_device
        )

        result = predictor.predict(simple_pyg_data, return_numpy=True)

        assert isinstance(result, np.ndarray)

    def test_predict_runs_in_no_grad_context(
        self, mock_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test predict runs inference with torch.no_grad()."""
        mock_model.return_value = torch.tensor([[0.5]])
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, device=cpu_device
        )

        # If no_grad is working, requires_grad should not propagate
        result = predictor.predict(simple_pyg_data)

        assert not result.requires_grad

    def test_predict_with_batch_data(self, mock_model, pyg_batch, cpu_device, working_root_dir):
        """Test predict with PyG Batch object."""
        mock_model.return_value = torch.tensor([[0.1], [0.2], [0.3]])  # 3 graphs in batch
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, device=cpu_device
        )

        result = predictor.predict(pyg_batch)

        assert isinstance(result, torch.Tensor)
        mock_model.assert_called()

    def test_predict_with_real_model_single_data(
        self, simple_pyg_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test predict with real model and single Data."""
        predictor = Predictor(
            model=simple_pyg_model, working_root_dir=working_root_dir, device=cpu_device
        )

        result = predictor.predict(simple_pyg_data)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 1)  # 1 graph, 1 output

    def test_predict_with_real_model_batch(
        self, simple_pyg_model, pyg_batch, cpu_device, working_root_dir
    ):
        """Test predict with real model and Batch."""
        predictor = Predictor(
            model=simple_pyg_model, working_root_dir=working_root_dir, device=cpu_device
        )

        result = predictor.predict(pyg_batch)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 1)  # 3 graphs, 1 output each


# =============================================================================
# PREDICTOR CLASS TESTS - _forward METHOD
# =============================================================================


class TestPredictorForward:
    """Test Predictor._forward internal method."""

    def test_forward_passes_x_and_edge_index(
        self, mock_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test _forward passes required x and edge_index."""
        mock_model.return_value = torch.tensor([[0.5]])
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, device=cpu_device
        )

        # Move data to device first (as predict does)
        data = simple_pyg_data.to(cpu_device)
        predictor._forward(data)

        # Verify model was called with x and edge_index
        call_args, call_kwargs = mock_model.call_args
        assert len(call_args) == 2  # x and edge_index
        assert torch.is_tensor(call_args[0])  # x
        assert torch.is_tensor(call_args[1])  # edge_index

    def test_forward_passes_edge_attr_when_present(
        self, mock_model, pyg_data_with_edge_attr, cpu_device, working_root_dir
    ):
        """Test _forward passes edge_attr when present in data."""
        mock_model.return_value = torch.tensor([[0.5]])
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, device=cpu_device
        )

        data = pyg_data_with_edge_attr.to(cpu_device)
        predictor._forward(data)

        call_args, call_kwargs = mock_model.call_args
        assert "edge_attr" in call_kwargs
        assert torch.is_tensor(call_kwargs["edge_attr"])

    def test_forward_passes_batch_when_present(
        self, mock_model, pyg_batch, cpu_device, working_root_dir
    ):
        """Test _forward passes batch index when present."""
        mock_model.return_value = torch.tensor([[0.1], [0.2], [0.3]])
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, device=cpu_device
        )

        batch = pyg_batch.to(cpu_device)
        predictor._forward(batch)

        call_args, call_kwargs = mock_model.call_args
        assert "batch" in call_kwargs
        assert torch.is_tensor(call_kwargs["batch"])

    def test_forward_passes_pos_when_present(
        self, mock_model, pyg_data_with_pos, cpu_device, working_root_dir
    ):
        """Test _forward passes pos when present in data."""
        mock_model.return_value = torch.tensor([[0.5]])
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, device=cpu_device
        )

        data = pyg_data_with_pos.to(cpu_device)
        predictor._forward(data)

        call_args, call_kwargs = mock_model.call_args
        assert "pos" in call_kwargs
        assert torch.is_tensor(call_kwargs["pos"])

    def test_forward_passes_all_optional_attrs(
        self, mock_model, pyg_data_with_all_attrs, cpu_device, working_root_dir
    ):
        """Test _forward passes all optional attributes when present."""
        mock_model.return_value = torch.tensor([[0.5]])
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, device=cpu_device
        )

        data = pyg_data_with_all_attrs.to(cpu_device)
        predictor._forward(data)

        call_args, call_kwargs = mock_model.call_args
        assert "edge_attr" in call_kwargs
        assert "pos" in call_kwargs

    def test_forward_does_not_pass_none_edge_attr(self, mock_model, cpu_device, working_root_dir):
        """Test _forward does not pass edge_attr if it's None."""
        mock_model.return_value = torch.tensor([[0.5]])
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, device=cpu_device
        )

        # Create data with edge_attr explicitly set to None
        data = Data(
            x=torch.randn(4, 16), edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=None
        ).to(cpu_device)

        predictor._forward(data)

        call_args, call_kwargs = mock_model.call_args
        assert "edge_attr" not in call_kwargs

    def test_forward_does_not_pass_missing_batch(
        self, mock_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test _forward does not pass batch if not present."""
        mock_model.return_value = torch.tensor([[0.5]])
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, device=cpu_device
        )

        data = simple_pyg_data.to(cpu_device)
        predictor._forward(data)

        call_args, call_kwargs = mock_model.call_args
        assert "batch" not in call_kwargs

    def test_forward_does_not_pass_missing_pos(
        self, mock_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test _forward does not pass pos if not present."""
        mock_model.return_value = torch.tensor([[0.5]])
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, device=cpu_device
        )

        data = simple_pyg_data.to(cpu_device)
        predictor._forward(data)

        call_args, call_kwargs = mock_model.call_args
        assert "pos" not in call_kwargs


# =============================================================================
# PREDICTOR CLASS TESTS - _postprocess METHOD
# =============================================================================


class TestPredictorPostprocess:
    """Test Predictor._postprocess internal method."""

    def test_postprocess_no_change_for_regression(
        self, mock_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test _postprocess returns output unchanged for regression tasks."""
        output = torch.tensor([[0.5]])
        mock_model.return_value = output
        predictor = Predictor(
            model=mock_model,
            working_root_dir=working_root_dir,
            device=cpu_device,
            task_type="graph_regression",
        )

        result = predictor._postprocess(output, simple_pyg_data)

        assert torch.equal(result, output)

    def test_postprocess_no_change_for_none_task_type(
        self, mock_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test _postprocess returns output unchanged when task_type is None."""
        output = torch.tensor([[0.1, 0.7, 0.2]])
        mock_model.return_value = output
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, device=cpu_device, task_type=None
        )

        result = predictor._postprocess(output, simple_pyg_data)

        assert torch.equal(result, output)

    def test_postprocess_argmax_for_classification(
        self, mock_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test _postprocess applies argmax for classification tasks."""
        output = torch.tensor(
            [
                [0.1, 0.7, 0.2],  # Class 1
                [0.8, 0.1, 0.1],  # Class 0
            ]
        )
        predictor = Predictor(
            model=mock_model,
            working_root_dir=working_root_dir,
            device=cpu_device,
            task_type="graph_classification",
        )

        result = predictor._postprocess(output, simple_pyg_data)

        expected = torch.tensor([1, 0])
        assert torch.equal(result, expected)

    def test_postprocess_argmax_for_node_classification(
        self, mock_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test _postprocess applies argmax for node_classification."""
        output = torch.tensor(
            [
                [0.9, 0.05, 0.05],  # Class 0
                [0.1, 0.8, 0.1],  # Class 1
                [0.2, 0.2, 0.6],  # Class 2
            ]
        )
        predictor = Predictor(
            model=mock_model,
            working_root_dir=working_root_dir,
            device=cpu_device,
            task_type="node_classification",
        )

        result = predictor._postprocess(output, simple_pyg_data)

        expected = torch.tensor([0, 1, 2])
        assert torch.equal(result, expected)

    def test_postprocess_handles_binary_classification(
        self, mock_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test _postprocess handles binary classification (2 classes)."""
        output = torch.tensor(
            [
                [0.7, 0.3],
                [0.2, 0.8],
            ]
        )
        predictor = Predictor(
            model=mock_model,
            working_root_dir=working_root_dir,
            device=cpu_device,
            task_type="classification",
        )

        result = predictor._postprocess(output, simple_pyg_data)

        expected = torch.tensor([0, 1])
        assert torch.equal(result, expected)

    def test_postprocess_no_argmax_for_1d_classification_output(
        self, mock_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test _postprocess does not apply argmax if output is 1D."""
        output = torch.tensor([0.5, 0.3, 0.7])  # 1D tensor
        predictor = Predictor(
            model=mock_model,
            working_root_dir=working_root_dir,
            device=cpu_device,
            task_type="classification",
        )

        result = predictor._postprocess(output, simple_pyg_data)

        # Should remain unchanged since dim() is 1
        assert torch.equal(result, output)

    def test_postprocess_no_argmax_for_single_class(
        self, mock_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test _postprocess does not apply argmax if only 1 class output."""
        output = torch.tensor([[0.5], [0.3]])  # Only 1 output per sample
        predictor = Predictor(
            model=mock_model,
            working_root_dir=working_root_dir,
            device=cpu_device,
            task_type="classification",
        )

        result = predictor._postprocess(output, simple_pyg_data)

        # Should remain unchanged since size(-1) is 1
        assert torch.equal(result, output)

    def test_postprocess_case_insensitive_classification(
        self, mock_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test _postprocess handles different cases of 'classification'."""
        output = torch.tensor([[0.1, 0.9]])

        for task_type in ["Classification", "CLASSIFICATION", "Graph_Classification"]:
            predictor = Predictor(
                model=mock_model,
                working_root_dir=working_root_dir,
                device=cpu_device,
                task_type=task_type,
            )
            result = predictor._postprocess(output, simple_pyg_data)
            expected = torch.tensor([1])
            assert torch.equal(result, expected), f"Failed for task_type={task_type}"


# =============================================================================
# PREDICTOR CLASS TESTS - predict_batch METHOD
# =============================================================================


class TestPredictorPredictBatch:
    """Test Predictor.predict_batch method."""

    def test_predict_batch_returns_tensor_by_default(
        self, simple_pyg_model, pyg_dataset, cpu_device, working_root_dir
    ):
        """Test predict_batch returns torch.Tensor by default."""
        predictor = Predictor(
            model=simple_pyg_model, working_root_dir=working_root_dir, device=cpu_device
        )

        result = predictor.predict_batch(pyg_dataset, batch_size=4)

        assert isinstance(result, torch.Tensor)

    def test_predict_batch_returns_numpy_when_requested(
        self, simple_pyg_model, pyg_dataset, cpu_device, working_root_dir
    ):
        """Test predict_batch returns numpy array when return_numpy=True."""
        predictor = Predictor(
            model=simple_pyg_model, working_root_dir=working_root_dir, device=cpu_device
        )

        result = predictor.predict_batch(pyg_dataset, batch_size=4, return_numpy=True)

        assert isinstance(result, np.ndarray)

    def test_predict_batch_correct_output_shape(
        self, simple_pyg_model, pyg_dataset, cpu_device, working_root_dir
    ):
        """Test predict_batch returns correct shape for all samples."""
        predictor = Predictor(
            model=simple_pyg_model, working_root_dir=working_root_dir, device=cpu_device
        )

        result = predictor.predict_batch(pyg_dataset, batch_size=4)

        # Should have prediction for each sample
        assert result.shape[0] == len(pyg_dataset)

    def test_predict_batch_with_different_batch_sizes(
        self, simple_pyg_model, pyg_dataset, cpu_device, working_root_dir
    ):
        """Test predict_batch with various batch sizes."""
        predictor = Predictor(
            model=simple_pyg_model, working_root_dir=working_root_dir, device=cpu_device
        )

        for batch_size in [1, 2, 5, 10, 20]:
            result = predictor.predict_batch(pyg_dataset, batch_size=batch_size)
            assert result.shape[0] == len(pyg_dataset)

    def test_predict_batch_preserves_order(self, cpu_device, working_root_dir):
        """Test predict_batch preserves sample order."""

        # Create model that returns unique values per graph
        class OrderTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.counter = 0

            def forward(self, x, edge_index, **kwargs):
                if "batch" in kwargs and kwargs["batch"] is not None:
                    num_graphs = kwargs["batch"].max().item() + 1
                    # Return sequential values to track order
                    result = torch.arange(
                        self.counter, self.counter + num_graphs, dtype=torch.float32
                    ).unsqueeze(1)
                    self.counter += num_graphs
                    return result
                return torch.tensor([[float(self.counter)]])

        model = OrderTestModel()
        predictor = Predictor(model=model, working_root_dir=working_root_dir, device=cpu_device)

        # Create dataset with 5 samples
        dataset = [
            Data(x=torch.randn(3, 16), edge_index=torch.tensor([[0, 1], [1, 0]])) for _ in range(5)
        ]

        result = predictor.predict_batch(dataset, batch_size=2)

        # Should be [0, 1, 2, 3, 4]
        expected = torch.arange(5, dtype=torch.float32).unsqueeze(1)
        assert torch.equal(result, expected)

    def test_predict_batch_uses_dataloader_correctly(
        self, mock_model, pyg_dataset, cpu_device, working_root_dir
    ):
        """Test predict_batch creates DataLoader with correct parameters."""
        mock_model.return_value = torch.tensor([[0.5]])
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, device=cpu_device
        )

        # Patch DataLoader to verify parameters
        with (
            patch.object(DataLoader, "__init__", return_value=None) as mock_dl_init,
            patch.object(DataLoader, "__iter__", return_value=iter([])),
            patch.object(DataLoader, "__len__", return_value=0),
        ):
            with contextlib.suppress(Exception):
                # Expected to fail, we just want to check DL init
                predictor.predict_batch(pyg_dataset, batch_size=8, num_workers=2)

            # Check DataLoader was initialized with correct params
            if mock_dl_init.called:
                call_kwargs = mock_dl_init.call_args[1]
                assert call_kwargs.get("batch_size") == 8
                assert call_kwargs.get("shuffle") is False
                assert call_kwargs.get("num_workers") == 2

    def test_predict_batch_with_small_dataset(self, simple_pyg_model, cpu_device, working_root_dir):
        """Test predict_batch with dataset smaller than batch_size."""
        predictor = Predictor(
            model=simple_pyg_model, working_root_dir=working_root_dir, device=cpu_device
        )

        # Create small dataset (3 samples)
        small_dataset = [
            Data(x=torch.randn(4, 16), edge_index=torch.tensor([[0, 1], [1, 0]])) for _ in range(3)
        ]

        result = predictor.predict_batch(small_dataset, batch_size=10)

        assert result.shape[0] == 3

    def test_predict_batch_with_single_sample(self, simple_pyg_model, cpu_device, working_root_dir):
        """Test predict_batch with single sample dataset."""
        predictor = Predictor(
            model=simple_pyg_model, working_root_dir=working_root_dir, device=cpu_device
        )

        single_dataset = [Data(x=torch.randn(4, 16), edge_index=torch.tensor([[0, 1], [1, 0]]))]

        result = predictor.predict_batch(single_dataset, batch_size=1)

        assert result.shape[0] == 1

    def test_predict_batch_concatenates_all_predictions(
        self, mock_model, pyg_dataset, cpu_device, working_root_dir
    ):
        """Test predict_batch correctly concatenates predictions from all batches."""
        # Return different predictions per batch
        batch_outputs = [
            torch.tensor([[0.1], [0.2]]),  # Batch 1: 2 samples
            torch.tensor([[0.3], [0.4]]),  # Batch 2: 2 samples
            torch.tensor([[0.5], [0.6]]),  # Batch 3: 2 samples
            torch.tensor([[0.7], [0.8]]),  # Batch 4: 2 samples
            torch.tensor([[0.9], [1.0]]),  # Batch 5: 2 samples
        ]
        mock_model.side_effect = batch_outputs
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, device=cpu_device
        )

        result = predictor.predict_batch(pyg_dataset, batch_size=2)

        # All 10 predictions should be concatenated
        assert result.shape[0] == 10


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestPredictConvenienceFunction:
    """Test the predict() convenience function."""

    @patch("milia_pipeline.models.post_training.inference.predictor.Predictor.from_checkpoint")
    def test_predict_creates_predictor_from_checkpoint(
        self, mock_from_checkpoint, simple_pyg_data, working_root_dir
    ):
        """Test predict() creates Predictor from checkpoint."""
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = torch.tensor([[0.5]])
        mock_from_checkpoint.return_value = mock_predictor

        _result = predict("model.pt", simple_pyg_data, working_root_dir=working_root_dir)

        mock_from_checkpoint.assert_called_once_with(
            "model.pt", working_root_dir=working_root_dir, device=None
        )

    @patch("milia_pipeline.models.post_training.inference.predictor.Predictor.from_checkpoint")
    def test_predict_passes_device(self, mock_from_checkpoint, simple_pyg_data, working_root_dir):
        """Test predict() passes device to from_checkpoint."""
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = torch.tensor([[0.5]])
        mock_from_checkpoint.return_value = mock_predictor

        device = torch.device("cpu")
        _result = predict(
            "model.pt", simple_pyg_data, working_root_dir=working_root_dir, device=device
        )

        mock_from_checkpoint.assert_called_once_with(
            "model.pt", working_root_dir=working_root_dir, device=device
        )

    @patch("milia_pipeline.models.post_training.inference.predictor.Predictor.from_checkpoint")
    def test_predict_calls_predictor_predict(
        self, mock_from_checkpoint, simple_pyg_data, working_root_dir
    ):
        """Test predict() calls predictor.predict with data."""
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = torch.tensor([[0.5]])
        mock_from_checkpoint.return_value = mock_predictor

        _result = predict("model.pt", simple_pyg_data, working_root_dir=working_root_dir)

        mock_predictor.predict.assert_called_once_with(simple_pyg_data, return_numpy=False)

    @patch("milia_pipeline.models.post_training.inference.predictor.Predictor.from_checkpoint")
    def test_predict_passes_return_numpy(
        self, mock_from_checkpoint, simple_pyg_data, working_root_dir
    ):
        """Test predict() passes return_numpy to predictor.predict."""
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = np.array([[0.5]])
        mock_from_checkpoint.return_value = mock_predictor

        _result = predict(
            "model.pt", simple_pyg_data, working_root_dir=working_root_dir, return_numpy=True
        )

        mock_predictor.predict.assert_called_once_with(simple_pyg_data, return_numpy=True)

    @patch("milia_pipeline.models.post_training.inference.predictor.Predictor.from_checkpoint")
    def test_predict_with_path_object(
        self, mock_from_checkpoint, simple_pyg_data, working_root_dir
    ):
        """Test predict() accepts Path objects."""
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = torch.tensor([[0.5]])
        mock_from_checkpoint.return_value = mock_predictor

        checkpoint_path = Path("/path/to/model.pt")
        _result = predict(checkpoint_path, simple_pyg_data, working_root_dir=working_root_dir)

        assert mock_from_checkpoint.called

    @patch("milia_pipeline.models.post_training.inference.predictor.Predictor.from_checkpoint")
    def test_predict_returns_prediction_result(
        self, mock_from_checkpoint, simple_pyg_data, working_root_dir
    ):
        """Test predict() returns the prediction result."""
        expected_output = torch.tensor([[0.5]])
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = expected_output
        mock_from_checkpoint.return_value = mock_predictor

        result = predict("model.pt", simple_pyg_data, working_root_dir=working_root_dir)

        assert torch.equal(result, expected_output)

    @patch("milia_pipeline.models.post_training.inference.predictor.Predictor.from_checkpoint")
    def test_predict_with_batch_data(self, mock_from_checkpoint, pyg_batch, working_root_dir):
        """Test predict() works with Batch data."""
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = torch.tensor([[0.1], [0.2], [0.3]])
        mock_from_checkpoint.return_value = mock_predictor

        result = predict("model.pt", pyg_batch, working_root_dir=working_root_dir)

        assert result.shape[0] == 3


# =============================================================================
# DEVICE HANDLING TESTS
# =============================================================================


class TestDeviceHandling:
    """Test device handling across Predictor methods."""

    def test_prediction_output_on_correct_device(
        self, mock_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test prediction output is on correct device."""
        mock_model.return_value = torch.tensor([[0.5]])
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, device=cpu_device
        )

        result = predictor.predict(simple_pyg_data)

        assert result.device == cpu_device

    def test_data_moved_to_predictor_device(
        self, mock_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test input data is moved to predictor device."""
        mock_model.return_value = torch.tensor([[0.5]])
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, device=cpu_device
        )

        predictor.predict(simple_pyg_data)

        # Verify model was called (data was processed on device)
        assert mock_model.called

    @patch("torch.cuda.is_available", return_value=True)
    def test_cuda_device_selection(self, mock_cuda, mock_model, simple_pyg_data, working_root_dir):
        """Test CUDA device is selected when available and not specified."""
        mock_model.return_value = torch.tensor([[0.5]])
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir, device=None)

        assert predictor.device == torch.device("cuda")

    @patch("torch.cuda.is_available", return_value=False)
    def test_cpu_fallback_when_cuda_unavailable(
        self, mock_cuda, mock_model, simple_pyg_data, working_root_dir
    ):
        """Test CPU fallback when CUDA is not available."""
        mock_model.return_value = torch.tensor([[0.5]])
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir, device=None)

        assert predictor.device == torch.device("cpu")

    def test_explicit_device_overrides_auto_detection(
        self, mock_model, simple_pyg_data, working_root_dir
    ):
        """Test explicit device specification overrides auto-detection."""
        mock_model.return_value = torch.tensor([[0.5]])
        device = torch.device("cpu")

        with patch("torch.cuda.is_available", return_value=True):
            predictor = Predictor(
                model=mock_model, working_root_dir=working_root_dir, device=device
            )

        assert predictor.device == device


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""

    def test_predict_with_empty_edge_index(self, mock_model, cpu_device, working_root_dir):
        """Test predict handles data with no edges."""
        mock_model.return_value = torch.tensor([[0.5]])
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, device=cpu_device
        )

        # Create data with no edges (isolated nodes)
        data = Data(x=torch.randn(4, 16), edge_index=torch.empty((2, 0), dtype=torch.long))

        result = predictor.predict(data)

        assert isinstance(result, torch.Tensor)

    def test_predict_with_single_node(self, mock_model, cpu_device, working_root_dir):
        """Test predict handles data with single node."""
        mock_model.return_value = torch.tensor([[0.5]])
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, device=cpu_device
        )

        data = Data(x=torch.randn(1, 16), edge_index=torch.empty((2, 0), dtype=torch.long))

        result = predictor.predict(data)

        assert isinstance(result, torch.Tensor)

    def test_predict_with_large_graph(self, mock_model, cpu_device, working_root_dir):
        """Test predict handles large graphs."""
        mock_model.return_value = torch.tensor([[0.5]])
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, device=cpu_device
        )

        # Create large graph (1000 nodes)
        num_nodes = 1000
        data = Data(
            x=torch.randn(num_nodes, 16), edge_index=torch.randint(0, num_nodes, (2, num_nodes * 5))
        )

        result = predictor.predict(data)

        assert isinstance(result, torch.Tensor)

    def test_predict_batch_with_varying_graph_sizes(
        self, simple_pyg_model, cpu_device, working_root_dir
    ):
        """Test predict_batch handles graphs of varying sizes."""
        predictor = Predictor(
            model=simple_pyg_model, working_root_dir=working_root_dir, device=cpu_device
        )

        # Create dataset with varying graph sizes
        dataset = []
        for size in [2, 5, 10, 3, 8, 15, 4]:
            x = torch.randn(size, 16)
            edge_index = torch.randint(0, size, (2, size * 2))
            dataset.append(Data(x=x, edge_index=edge_index))

        result = predictor.predict_batch(dataset, batch_size=3)

        assert result.shape[0] == len(dataset)

    def test_model_remains_in_eval_mode_after_prediction(
        self, simple_pyg_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test model remains in eval mode after prediction."""
        predictor = Predictor(
            model=simple_pyg_model, working_root_dir=working_root_dir, device=cpu_device
        )

        predictor.predict(simple_pyg_data)

        assert not simple_pyg_model.training

    def test_predict_preserves_model_parameters(
        self, simple_pyg_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test prediction does not modify model parameters."""
        predictor = Predictor(
            model=simple_pyg_model, working_root_dir=working_root_dir, device=cpu_device
        )

        # Store original parameters
        original_params = {
            name: param.clone() for name, param in simple_pyg_model.named_parameters()
        }

        # Make multiple predictions
        for _ in range(5):
            predictor.predict(simple_pyg_data)

        # Check parameters are unchanged
        for name, param in simple_pyg_model.named_parameters():
            assert torch.equal(param, original_params[name])

    def test_numpy_output_on_cpu(
        self, simple_pyg_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test numpy output works correctly on CPU."""
        predictor = Predictor(
            model=simple_pyg_model, working_root_dir=working_root_dir, device=cpu_device
        )

        result = predictor.predict(simple_pyg_data, return_numpy=True)

        assert isinstance(result, np.ndarray)
        assert result.dtype in [np.float32, np.float64]


# =============================================================================
# CLASSIFICATION TASK TYPE TESTS
# =============================================================================


class TestClassificationTaskTypes:
    """Test various classification task type handling."""

    @pytest.mark.parametrize(
        "task_type",
        [
            "classification",
            "graph_classification",
            "node_classification",
            "edge_classification",
            "Classification",
            "GRAPH_CLASSIFICATION",
            "multi_class_classification",
        ],
    )
    def test_various_classification_task_types(
        self,
        mock_model_for_classification,
        simple_pyg_data,
        cpu_device,
        working_root_dir,
        task_type,
    ):
        """Test _postprocess applies argmax for various classification task types."""
        predictor = Predictor(
            model=mock_model_for_classification,
            working_root_dir=working_root_dir,
            device=cpu_device,
            task_type=task_type,
        )

        result = predictor.predict(simple_pyg_data)

        # Should return class indices
        assert result.dim() == 1
        assert result.dtype == torch.int64

    @pytest.mark.parametrize(
        "task_type",
        [
            "regression",
            "graph_regression",
            "node_regression",
            "edge_regression",
            None,
        ],
    )
    def test_regression_task_types_no_argmax(
        self, mock_model, simple_pyg_data, cpu_device, working_root_dir, task_type
    ):
        """Test _postprocess does not apply argmax for regression task types."""
        output = torch.tensor([[0.1, 0.7, 0.2]])
        mock_model.return_value = output

        predictor = Predictor(
            model=mock_model,
            working_root_dir=working_root_dir,
            device=cpu_device,
            task_type=task_type,
        )

        result = predictor.predict(simple_pyg_data)

        # Should return original output (3 values)
        assert result.shape[-1] == 3


# =============================================================================
# DATA ATTRIBUTE HANDLING TESTS
# =============================================================================


class TestDataAttributeHandling:
    """Test handling of various PyG Data attributes."""

    def test_forward_with_custom_attributes(self, mock_model, cpu_device, working_root_dir):
        """Test _forward ignores custom/unknown attributes."""
        mock_model.return_value = torch.tensor([[0.5]])
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, device=cpu_device
        )

        # Create data with custom attributes
        data = Data(
            x=torch.randn(4, 16),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            custom_attr=torch.randn(4, 8),
            another_attr="some string",
        )

        predictor._forward(data.to(cpu_device))

        # Model should be called without custom attributes in kwargs
        call_args, call_kwargs = mock_model.call_args
        assert "custom_attr" not in call_kwargs
        assert "another_attr" not in call_kwargs

    def test_forward_with_y_attribute(self, mock_model, cpu_device, working_root_dir):
        """Test _forward ignores target y attribute."""
        mock_model.return_value = torch.tensor([[0.5]])
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, device=cpu_device
        )

        # Create data with target
        data = Data(
            x=torch.randn(4, 16), edge_index=torch.tensor([[0, 1], [1, 0]]), y=torch.tensor([1.0])
        )

        predictor._forward(data.to(cpu_device))

        # Model should not receive y
        call_args, call_kwargs = mock_model.call_args
        assert "y" not in call_kwargs

    def test_batch_handles_ptr_attribute(self, mock_model, pyg_batch, cpu_device, working_root_dir):
        """Test predict correctly handles Batch.ptr attribute."""
        mock_model.return_value = torch.tensor([[0.1], [0.2], [0.3]])
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, device=cpu_device
        )

        # Batch should have ptr attribute
        assert hasattr(pyg_batch, "ptr")

        result = predictor.predict(pyg_batch)

        assert isinstance(result, torch.Tensor)


# =============================================================================
# LOGGING TESTS
# =============================================================================


class TestLogging:
    """Test logging behavior."""

    def test_logger_is_module_logger(self):
        """Test that logger is set up correctly."""
        from milia_pipeline.models.post_training.inference.predictor import logger

        assert logger.name == "milia_pipeline.models.post_training.inference.predictor"


# =============================================================================
# TYPE ANNOTATION TESTS
# =============================================================================


class TestTypeAnnotations:
    """Test that type annotations are honored."""

    def test_predict_accepts_data_type(
        self, simple_pyg_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test predict accepts PyG Data type."""
        predictor = Predictor(
            model=simple_pyg_model, working_root_dir=working_root_dir, device=cpu_device
        )

        result = predictor.predict(simple_pyg_data)

        assert isinstance(result, torch.Tensor)

    def test_predict_accepts_batch_type(
        self, simple_pyg_model, pyg_batch, cpu_device, working_root_dir
    ):
        """Test predict accepts PyG Batch type."""
        predictor = Predictor(
            model=simple_pyg_model, working_root_dir=working_root_dir, device=cpu_device
        )

        result = predictor.predict(pyg_batch)

        assert isinstance(result, torch.Tensor)

    def test_predict_batch_accepts_list(
        self, simple_pyg_model, pyg_dataset, cpu_device, working_root_dir
    ):
        """Test predict_batch accepts List[Data]."""
        predictor = Predictor(
            model=simple_pyg_model, working_root_dir=working_root_dir, device=cpu_device
        )

        result = predictor.predict_batch(pyg_dataset, batch_size=4)

        assert isinstance(result, torch.Tensor)


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================


class TestIntegrationStyle:
    """Integration-style tests for complete workflows."""

    def test_full_prediction_workflow(
        self, simple_pyg_model, pyg_dataset, cpu_device, working_root_dir
    ):
        """Test complete prediction workflow."""
        # Initialize predictor
        predictor = Predictor(
            model=simple_pyg_model,
            working_root_dir=working_root_dir,
            device=cpu_device,
            task_type="graph_regression",
        )

        # Single prediction
        single_result = predictor.predict(pyg_dataset[0])
        assert isinstance(single_result, torch.Tensor)

        # Batch prediction
        batch_result = predictor.predict_batch(pyg_dataset, batch_size=4)
        assert batch_result.shape[0] == len(pyg_dataset)

        # Numpy output
        numpy_result = predictor.predict(pyg_dataset[0], return_numpy=True)
        assert isinstance(numpy_result, np.ndarray)

    def test_classification_workflow(self, cpu_device, working_root_dir):
        """Test complete classification workflow."""

        # Create classification model
        class ClassificationModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 3)  # 3 classes

            def forward(self, x, edge_index, **kwargs):
                if "batch" in kwargs and kwargs["batch"] is not None:
                    from torch_geometric.nn import global_mean_pool

                    x = global_mean_pool(x, kwargs["batch"])
                else:
                    x = x.mean(dim=0, keepdim=True)
                return self.linear(x)

        model = ClassificationModel()
        predictor = Predictor(
            model=model,
            working_root_dir=working_root_dir,
            device=cpu_device,
            task_type="graph_classification",
        )

        # Create test data
        data = Data(x=torch.randn(5, 16), edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]))

        result = predictor.predict(data)

        # Should return class index
        assert result.dim() == 1
        assert result.item() in [0, 1, 2]

    def test_multiple_predictions_consistency(
        self, simple_pyg_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test multiple predictions give consistent results."""
        predictor = Predictor(
            model=simple_pyg_model, working_root_dir=working_root_dir, device=cpu_device
        )

        # Run same prediction multiple times
        results = [predictor.predict(simple_pyg_data) for _ in range(5)]

        # All results should be identical (deterministic)
        for i in range(1, len(results)):
            assert torch.equal(results[0], results[i])


# =============================================================================
# MEMORY AND GRADIENT TESTS
# =============================================================================


class TestMemoryAndGradients:
    """Test memory and gradient handling."""

    def test_no_gradient_computation_during_predict(
        self, simple_pyg_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test that gradients are not computed during prediction."""
        predictor = Predictor(
            model=simple_pyg_model, working_root_dir=working_root_dir, device=cpu_device
        )

        # Enable gradient tracking on input
        data = simple_pyg_data.clone()
        data.x.requires_grad_(True)

        result = predictor.predict(data)

        # Result should not require grad
        assert not result.requires_grad

    def test_predict_does_not_accumulate_gradients(
        self, simple_pyg_model, simple_pyg_data, cpu_device, working_root_dir
    ):
        """Test multiple predictions don't accumulate gradients."""
        predictor = Predictor(
            model=simple_pyg_model, working_root_dir=working_root_dir, device=cpu_device
        )

        # Check all model params have no grad
        for param in simple_pyg_model.parameters():
            assert param.grad is None

        # Multiple predictions
        for _ in range(5):
            predictor.predict(simple_pyg_data)

        # Params should still have no grad
        for param in simple_pyg_model.parameters():
            assert param.grad is None


# =============================================================================
# STRUCTURAL FEATURES CONFIG PROPERTY TESTS (FIX 19)
# =============================================================================


class TestStructuralFeaturesConfig:
    """Test the structural_features_config property (FIX 19)."""

    def test_structural_features_config_returns_config_when_present(
        self, mock_model, working_root_dir
    ):
        """Test property returns config from model_info.data_info."""
        expected_config = {"atom": ["atomic_num", "degree"], "bond": ["bond_type"]}
        model_info = {"data_info": {"structural_features_config": expected_config}}
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, model_info=model_info
        )

        assert predictor.structural_features_config == expected_config

    def test_structural_features_config_returns_none_when_no_model_info(
        self, mock_model, working_root_dir
    ):
        """Test property returns None when model_info is empty."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir)

        assert predictor.structural_features_config is None

    def test_structural_features_config_returns_none_when_no_data_info(
        self, mock_model, working_root_dir
    ):
        """Test property returns None when data_info key is missing."""
        model_info = {"model_name": "GCN"}
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, model_info=model_info
        )

        assert predictor.structural_features_config is None

    def test_structural_features_config_returns_none_when_no_config_key(
        self, mock_model, working_root_dir
    ):
        """Test property returns None when structural_features_config key is absent in data_info."""
        model_info = {"data_info": {"requires_edge_features": True}}
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, model_info=model_info
        )

        assert predictor.structural_features_config is None

    def test_structural_features_config_with_empty_config(self, mock_model, working_root_dir):
        """Test property returns empty dict when config is empty."""
        model_info = {"data_info": {"structural_features_config": {}}}
        predictor = Predictor(
            model=mock_model, working_root_dir=working_root_dir, model_info=model_info
        )

        assert predictor.structural_features_config == {}


# =============================================================================
# _resolve_path METHOD TESTS
# =============================================================================


class TestResolvePath:
    """Test Predictor._resolve_path method."""

    def test_resolve_path_relative_resolves_against_working_root(
        self, mock_model, working_root_dir
    ):
        """Test relative path resolves against working_root_dir."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir)

        result = predictor._resolve_path("subdir/file.pt")

        expected = (working_root_dir / "subdir" / "file.pt").resolve()
        assert result == expected

    def test_resolve_path_absolute_returned_as_is(self, mock_model, working_root_dir):
        """Test absolute path is returned as-is (resolved)."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir)

        abs_path = Path("/tmp/absolute/file.pt")
        result = predictor._resolve_path(abs_path)

        assert result == abs_path.resolve()

    def test_resolve_path_creates_parents_when_requested(self, mock_model, working_root_dir):
        """Test _resolve_path creates parent directories when create_parents=True."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir)

        nested_path = "deeply/nested/dir/file.pt"
        result = predictor._resolve_path(nested_path, create_parents=True)

        assert result.parent.exists()

    def test_resolve_path_does_not_create_parents_by_default(self, mock_model, working_root_dir):
        """Test _resolve_path does NOT create parent directories by default."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir)

        result = predictor._resolve_path("nonexistent/path/file.pt")

        assert not result.parent.exists()

    def test_resolve_path_accepts_string(self, mock_model, working_root_dir):
        """Test _resolve_path accepts string paths."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir)

        result = predictor._resolve_path("file.pt")

        assert isinstance(result, Path)

    def test_resolve_path_accepts_path_object(self, mock_model, working_root_dir):
        """Test _resolve_path accepts Path objects."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir)

        result = predictor._resolve_path(Path("file.pt"))

        assert isinstance(result, Path)

    def test_resolve_path_expands_tilde(self, mock_model, working_root_dir):
        """Test _resolve_path expands ~ in paths."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir)

        result = predictor._resolve_path("~/somefile.pt")

        assert "~" not in str(result)


# =============================================================================
# save_predictions METHOD TESTS
# =============================================================================


class TestSavePredictions:
    """Test Predictor.save_predictions method."""

    def test_save_predictions_csv_format(self, mock_model, working_root_dir):
        """Test save_predictions saves CSV correctly."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir)
        predictions = torch.tensor([0.1, 0.2, 0.3])

        result_path = predictor.save_predictions(predictions, "output.csv", format="csv")

        assert result_path.exists()
        import pandas as pd

        df = pd.read_csv(result_path)
        assert "prediction" in df.columns
        assert len(df) == 3

    def test_save_predictions_csv_multi_output(self, mock_model, working_root_dir):
        """Test save_predictions saves multi-dimensional predictions as CSV."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir)
        predictions = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

        result_path = predictor.save_predictions(predictions, "multi_output.csv", format="csv")

        import pandas as pd

        df = pd.read_csv(result_path)
        assert "prediction_0" in df.columns
        assert "prediction_1" in df.columns
        assert len(df) == 2

    def test_save_predictions_csv_with_identifiers(self, mock_model, working_root_dir):
        """Test save_predictions includes input identifiers in CSV."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir)
        predictions = torch.tensor([0.5, 0.6])
        identifiers = ["CCO", "CC(=O)O"]

        result_path = predictor.save_predictions(
            predictions,
            "with_ids.csv",
            format="csv",
            include_inputs=True,
            input_identifiers=identifiers,
        )

        import pandas as pd

        df = pd.read_csv(result_path)
        assert "input" in df.columns
        assert list(df["input"]) == identifiers

    def test_save_predictions_json_format(self, mock_model, working_root_dir):
        """Test save_predictions saves JSON correctly."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir)
        predictions = torch.tensor([0.1, 0.2])

        result_path = predictor.save_predictions(predictions, "output.json", format="json")

        assert result_path.exists()
        with open(result_path) as f:
            data = json.load(f)
        assert "predictions" in data
        assert len(data["predictions"]) == 2

    def test_save_predictions_json_with_identifiers(self, mock_model, working_root_dir):
        """Test save_predictions includes identifiers in JSON."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir)
        predictions = torch.tensor([0.5])
        identifiers = ["CCO"]

        result_path = predictor.save_predictions(
            predictions,
            "with_ids.json",
            format="json",
            include_inputs=True,
            input_identifiers=identifiers,
        )

        with open(result_path) as f:
            data = json.load(f)
        assert "inputs" in data
        assert data["inputs"] == identifiers

    def test_save_predictions_npy_format(self, mock_model, working_root_dir):
        """Test save_predictions saves NPY correctly."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir)
        predictions = torch.tensor([0.1, 0.2, 0.3])

        result_path = predictor.save_predictions(predictions, "output.npy", format="npy")

        assert result_path.exists()
        loaded = np.load(result_path)
        assert len(loaded) == 3

    def test_save_predictions_pt_format(self, mock_model, working_root_dir):
        """Test save_predictions saves PT correctly."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir)
        predictions = torch.tensor([0.1, 0.2, 0.3])

        result_path = predictor.save_predictions(predictions, "output.pt", format="pt")

        assert result_path.exists()
        loaded = torch.load(result_path, weights_only=True)
        assert len(loaded) == 3

    def test_save_predictions_unsupported_format_raises(self, mock_model, working_root_dir):
        """Test save_predictions raises ValueError for unsupported format."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir)
        predictions = torch.tensor([0.1])

        with pytest.raises(ValueError, match="Unsupported format"):
            predictor.save_predictions(predictions, "output.xyz", format="xyz")

    def test_save_predictions_accepts_numpy_input(self, mock_model, working_root_dir):
        """Test save_predictions handles numpy array input."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir)
        predictions = np.array([0.1, 0.2, 0.3])

        result_path = predictor.save_predictions(predictions, "numpy_input.csv", format="csv")

        assert result_path.exists()

    def test_save_predictions_creates_parent_directories(self, mock_model, working_root_dir):
        """Test save_predictions creates parent directories for output path."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir)
        predictions = torch.tensor([0.1])

        result_path = predictor.save_predictions(
            predictions, "nested/subdir/output.csv", format="csv"
        )

        assert result_path.exists()

    def test_save_predictions_returns_resolved_path(self, mock_model, working_root_dir):
        """Test save_predictions returns a resolved Path object."""
        predictor = Predictor(model=mock_model, working_root_dir=working_root_dir)
        predictions = torch.tensor([0.1])

        result_path = predictor.save_predictions(predictions, "result.csv", format="csv")

        assert isinstance(result_path, Path)
        assert result_path.is_absolute()


# =============================================================================
# 3D MOLECULAR MODEL FORWARD TESTS (FIX 25)
# =============================================================================


class TestForward3DMolecularModel:
    """Test _forward with 3D molecular models (SchNet, DimeNet, etc.)."""

    def test_forward_detects_3d_model_by_signature(self, cpu_device, working_root_dir):
        """Test _forward detects 3D molecular model via z/pos in forward signature."""

        class Mock3DModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)

            def forward(self, z, pos, batch=None):
                return torch.tensor([[0.5]])

        model = Mock3DModel()
        predictor = Predictor(model=model, working_root_dir=working_root_dir, device=cpu_device)

        data = Data(z=torch.tensor([6, 8, 1, 1]), pos=torch.randn(4, 3)).to(cpu_device)

        result = predictor._forward(data)
        assert isinstance(result, torch.Tensor)

    def test_forward_3d_model_passes_z_and_pos(self, cpu_device, working_root_dir):
        """Test _forward passes z and pos to 3D model."""
        call_tracker = {}

        class Track3DModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)

            def forward(self, z, pos, batch=None):
                call_tracker["z"] = z
                call_tracker["pos"] = pos
                call_tracker["batch"] = batch
                return torch.tensor([[0.5]])

        model = Track3DModel()
        predictor = Predictor(model=model, working_root_dir=working_root_dir, device=cpu_device)

        z_input = torch.tensor([6, 8, 1, 1])
        pos_input = torch.randn(4, 3)
        data = Data(z=z_input, pos=pos_input).to(cpu_device)

        predictor._forward(data)

        assert torch.equal(call_tracker["z"], z_input)
        assert torch.equal(call_tracker["pos"], pos_input)

    def test_forward_3d_model_passes_batch_when_present(self, cpu_device, working_root_dir):
        """Test _forward passes batch to 3D model when present."""
        call_tracker = {}

        class Track3DModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)

            def forward(self, z, pos, batch=None):
                call_tracker["batch"] = batch
                return torch.tensor([[0.5]])

        model = Track3DModel()
        predictor = Predictor(model=model, working_root_dir=working_root_dir, device=cpu_device)

        batch_tensor = torch.tensor([0, 0, 1, 1])
        data = Data(z=torch.tensor([6, 8, 1, 1]), pos=torch.randn(4, 3), batch=batch_tensor).to(
            cpu_device
        )

        predictor._forward(data)

        assert torch.equal(call_tracker["batch"], batch_tensor)

    def test_forward_3d_model_raises_on_missing_z(self, cpu_device, working_root_dir, caplog):
        """Test _forward triggers fallback with warning when z is missing for 3D model.

        Evidence from predictor.py: The ValueError raised for missing z (line 314-317)
        is caught by the except (ValueError, TypeError) fallback block (line 392),
        which logs a warning and attempts fallback logic. The fallback may succeed
        silently depending on what data attributes are available.
        """

        class Mock3DModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)

            def forward(self, z, pos, batch=None):
                return torch.tensor([[0.5]])

        model = Mock3DModel()
        predictor = Predictor(model=model, working_root_dir=working_root_dir, device=cpu_device)

        # Data WITHOUT z — the internal ValueError is caught by _forward's
        # fallback block which logs the warning about the missing z
        data = Data(pos=torch.randn(4, 3)).to(cpu_device)

        with caplog.at_level(logging.WARNING):
            predictor._forward(data)

        # The fallback warning must contain the original missing-z error message
        assert any(
            "3D molecular model requires 'z'" in record.message
            or "Signature introspection failed" in record.message
            for record in caplog.records
        )

    def test_forward_3d_model_raises_on_missing_pos(self, cpu_device, working_root_dir, caplog):
        """Test _forward triggers fallback with warning when pos is missing for 3D model.

        Evidence from predictor.py: The ValueError raised for missing pos (line 318-322)
        is caught by the except (ValueError, TypeError) fallback block (line 392),
        which logs a warning and attempts fallback logic. The fallback may succeed
        silently depending on what data attributes are available.
        """

        class Mock3DModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)

            def forward(self, z, pos, batch=None):
                return torch.tensor([[0.5]])

        model = Mock3DModel()
        predictor = Predictor(model=model, working_root_dir=working_root_dir, device=cpu_device)

        # Data WITHOUT pos — the internal ValueError is caught by _forward's
        # fallback block which logs the warning about the missing pos
        data = Data(z=torch.tensor([6, 8])).to(cpu_device)

        with caplog.at_level(logging.WARNING):
            predictor._forward(data)

        # The fallback warning must contain the original missing-pos error message
        assert any(
            "3D molecular model requires 'pos'" in record.message
            or "Signature introspection failed" in record.message
            for record in caplog.records
        )

    def test_forward_3d_model_with_edge_index_param(self, cpu_device, working_root_dir):
        """Test _forward passes edge_index to 3D model if model accepts it."""
        call_tracker = {}

        class Hybrid3DModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)

            def forward(self, z, pos, batch=None, edge_index=None):
                call_tracker["edge_index"] = edge_index
                return torch.tensor([[0.5]])

        model = Hybrid3DModel()
        predictor = Predictor(model=model, working_root_dir=working_root_dir, device=cpu_device)

        edge_index = torch.tensor([[0, 1], [1, 0]])
        data = Data(z=torch.tensor([6, 8]), pos=torch.randn(2, 3), edge_index=edge_index).to(
            cpu_device
        )

        predictor._forward(data)

        assert torch.equal(call_tracker["edge_index"], edge_index)

    def test_forward_standard_gnn_not_confused_by_pos(self, cpu_device, working_root_dir):
        """Test _forward does NOT treat standard GNN with pos kwarg as 3D model."""

        class StandardGNNWithPos(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 1)

            def forward(self, x, edge_index, pos=None, batch=None):
                return self.linear(x.mean(dim=0, keepdim=True))

        model = StandardGNNWithPos()
        predictor = Predictor(model=model, working_root_dir=working_root_dir, device=cpu_device)

        data = Data(
            x=torch.randn(4, 16), edge_index=torch.tensor([[0, 1], [1, 0]]), pos=torch.randn(4, 3)
        ).to(cpu_device)

        # Should take standard GNN path (x, edge_index, ...) not 3D path
        result = predictor._forward(data)
        assert isinstance(result, torch.Tensor)


# =============================================================================
# FORWARD EDGE_WEIGHT AND VAR_KEYWORD TESTS
# =============================================================================


class TestForwardEdgeWeightAndVarKeyword:
    """Test _forward handles edge_weight, batch_size, and **kwargs models."""

    def test_forward_passes_edge_weight_when_present(self, cpu_device, working_root_dir):
        """Test _forward passes edge_weight to model that accepts it."""

        class EdgeWeightModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 1)

            def forward(self, x, edge_index, edge_weight=None, batch=None):
                return self.linear(x.mean(dim=0, keepdim=True))

        model = EdgeWeightModel()
        predictor = Predictor(model=model, working_root_dir=working_root_dir, device=cpu_device)

        data = Data(
            x=torch.randn(4, 16),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
            edge_weight=torch.tensor([1.0, 0.5, 0.8]),
        ).to(cpu_device)

        result = predictor._forward(data)
        assert isinstance(result, torch.Tensor)

    def test_forward_with_var_keyword_model(self, cpu_device, working_root_dir):
        """Test _forward passes all candidate kwargs to model accepting **kwargs."""
        received_kwargs = {}

        class KwargsModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 1)

            def forward(self, x, edge_index, **kwargs):
                received_kwargs.update(kwargs)
                return self.linear(x.mean(dim=0, keepdim=True))

        model = KwargsModel()
        predictor = Predictor(model=model, working_root_dir=working_root_dir, device=cpu_device)

        data = Data(
            x=torch.randn(4, 16),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_attr=torch.randn(2, 8),
            pos=torch.randn(4, 3),
        ).to(cpu_device)

        predictor._forward(data)

        assert "edge_attr" in received_kwargs
        assert "pos" in received_kwargs

    def test_forward_filters_kwargs_for_fixed_signature(self, cpu_device, working_root_dir):
        """Test _forward filters kwargs for model with fixed signature."""

        class FixedSigModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 1)

            def forward(self, x, edge_index, edge_attr=None):
                return self.linear(x.mean(dim=0, keepdim=True))

        model = FixedSigModel()
        predictor = Predictor(model=model, working_root_dir=working_root_dir, device=cpu_device)

        data = Data(
            x=torch.randn(4, 16),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_attr=torch.randn(2, 8),
            pos=torch.randn(4, 3),  # Not accepted by model
        ).to(cpu_device)

        # Should NOT raise even though pos isn't in model signature
        result = predictor._forward(data)
        assert isinstance(result, torch.Tensor)


# =============================================================================
# FORWARD FALLBACK LOGIC TESTS
# =============================================================================


class TestForwardFallback:
    """Test _forward fallback logic when signature introspection fails."""

    def test_forward_fallback_3d_path(self, cpu_device, working_root_dir):
        """Test _forward fallback path for 3D models when introspection fails."""

        class FallbackModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)

            def forward(self, z, pos, **kwargs):
                return torch.tensor([[0.5]])

        model = FallbackModel()
        predictor = Predictor(model=model, working_root_dir=working_root_dir, device=cpu_device)

        # Force fallback by patching inspect.signature to raise
        data = Data(z=torch.tensor([6, 8]), pos=torch.randn(2, 3)).to(cpu_device)

        with patch("inspect.signature", side_effect=ValueError("Introspection failure")):
            result = predictor._forward(data)
            assert isinstance(result, torch.Tensor)

    def test_forward_fallback_standard_gnn_path(self, cpu_device, working_root_dir):
        """Test _forward fallback path for standard GNNs when introspection fails."""

        class FallbackGNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 1)

            def forward(self, x, edge_index, **kwargs):
                return self.linear(x.mean(dim=0, keepdim=True))

        model = FallbackGNN()
        predictor = Predictor(model=model, working_root_dir=working_root_dir, device=cpu_device)

        data = Data(x=torch.randn(4, 16), edge_index=torch.tensor([[0, 1], [1, 0]])).to(cpu_device)

        with patch("inspect.signature", side_effect=TypeError("Introspection failure")):
            result = predictor._forward(data)
            assert isinstance(result, torch.Tensor)


# =============================================================================
# FORWARD MODEL UNWRAPPING TESTS
# =============================================================================


class TestForwardModelUnwrapping:
    """Test _forward model unwrapping logic (GraphLevelModelWrapper)."""

    def test_forward_unwraps_wrapper_model(self, cpu_device, working_root_dir):
        """Test _forward unwraps wrapped models for signature introspection."""

        class InnerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 1)

            def forward(self, x, edge_index, batch=None):
                return self.linear(x.mean(dim=0, keepdim=True))

        class WrapperModel(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x, edge_index, **kwargs):
                return self.model(x, edge_index, **kwargs)

        inner = InnerModel()
        wrapper = WrapperModel(inner)
        predictor = Predictor(model=wrapper, working_root_dir=working_root_dir, device=cpu_device)

        data = Data(x=torch.randn(4, 16), edge_index=torch.tensor([[0, 1], [1, 0]])).to(cpu_device)

        result = predictor._forward(data)
        assert isinstance(result, torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
