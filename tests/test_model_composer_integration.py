#!/usr/bin/env python3
"""
Phase 4: Model Composer - Comprehensive Integration Test

Tests all functionality of ModelComposer including:
- Parallel ensemble
- Sequential stacking
- Hierarchical composition
- All fusion methods
- Validation
- Configuration import/export

Run with: pytest tests/test_model_composer_integration.py -v
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import importlib.util
import os
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

# ==============================================================================
# CRITICAL: Mock problematic imports to prevent ModuleNotFoundError
# ==============================================================================
_mock_modules = {}

# Mock torch_geometric modules BEFORE any milia_pipeline imports
mock_pyg_data = Mock()
mock_pyg_data.Data = Mock
mock_pyg_data.Batch = Mock
_mock_modules["torch_geometric.data"] = mock_pyg_data

mock_pyg_utils = Mock()
_mock_modules["torch_geometric.utils"] = mock_pyg_utils

# Mock torch_geometric.nn module with pooling functions
mock_pyg_nn = Mock()


def mock_global_mean_pool(x, batch):
    """Mock global mean pool."""
    if batch is None:
        return x.mean(dim=0, keepdim=True)
    num_graphs = int(batch.max().item()) + 1
    return torch.zeros(num_graphs, x.size(-1))


def mock_global_max_pool(x, batch):
    """Mock global max pool."""
    if batch is None:
        return x.max(dim=0, keepdim=True)[0]
    num_graphs = int(batch.max().item()) + 1
    return torch.zeros(num_graphs, x.size(-1))


def mock_global_add_pool(x, batch):
    """Mock global add pool."""
    if batch is None:
        return x.sum(dim=0, keepdim=True)
    num_graphs = int(batch.max().item()) + 1
    return torch.zeros(num_graphs, x.size(-1))


mock_pyg_nn.global_mean_pool = mock_global_mean_pool
mock_pyg_nn.global_max_pool = mock_global_max_pool
mock_pyg_nn.global_add_pool = mock_global_add_pool
_mock_modules["torch_geometric.nn"] = mock_pyg_nn

mock_pyg = Mock()
mock_pyg.data = mock_pyg_data
mock_pyg.utils = mock_pyg_utils
mock_pyg.nn = mock_pyg_nn
_mock_modules["torch_geometric"] = mock_pyg

# Store original modules for cleanup (populated by setup_module)
_original_modules = {}


# Mock milia_pipeline.exceptions module
class BaseProjectError(Exception):
    """Base exception - mocked."""

    def __init__(self, message: str, details: str = None, **kwargs):
        super().__init__(message)
        self.message = message
        self.details = details
        self.extra_info = kwargs


class ModelError(BaseProjectError):
    """Model error - mocked."""

    pass


mock_exceptions = Mock()
mock_exceptions.BaseProjectError = BaseProjectError
mock_exceptions.ModelError = ModelError
_mock_modules["milia_pipeline.exceptions"] = mock_exceptions

# ---------------------------------------------------------------------------
# Module-level placeholders for classes loaded in setup_module().
# Set to None at import time (safe during collection), populated before tests.
# ---------------------------------------------------------------------------
model_composer_module = None
ModelSpec = None
EnsembleConfig = None
CompositionError = None
ModelComposer = None
ParallelEnsemble = None
SequentialStack = None
HierarchicalComposition = None


def setup_module(module):
    """
    Inject mocked modules into sys.modules and load the module-under-test.

    Runs ONCE before any test — does NOT run during pytest collection,
    preventing sys.modules pollution from breaking other test files.
    """
    global model_composer_module
    global ModelSpec, EnsembleConfig, CompositionError
    global ModelComposer, ParallelEnsemble, SequentialStack, HierarchicalComposition

    # --- Inject torch_geometric mocks into sys.modules ---
    for mod_name in _mock_modules:
        if mod_name in sys.modules:
            _original_modules[mod_name] = sys.modules[mod_name]
        sys.modules[mod_name] = _mock_modules[mod_name]

    # --- Inject milia_pipeline.exceptions mock ---
    if "milia_pipeline.exceptions" in sys.modules:
        _original_modules["milia_pipeline.exceptions"] = sys.modules["milia_pipeline.exceptions"]
    sys.modules["milia_pipeline.exceptions"] = mock_exceptions

    # --- Load model_composer.py directly (bypass package __init__) ---
    model_composer_path = os.path.join(
        str(project_root), "milia_pipeline", "models", "builders", "model_composer.py"
    )
    spec = importlib.util.spec_from_file_location(
        "milia_pipeline.models.builders.model_composer", model_composer_path
    )
    model_composer_module = importlib.util.module_from_spec(spec)
    sys.modules["milia_pipeline.models.builders.model_composer"] = model_composer_module
    spec.loader.exec_module(model_composer_module)

    # --- Extract classes into module-level names ---
    ModelSpec = model_composer_module.ModelSpec
    EnsembleConfig = model_composer_module.EnsembleConfig
    CompositionError = model_composer_module.CompositionError
    ModelComposer = model_composer_module.ModelComposer
    ParallelEnsemble = model_composer_module.ParallelEnsemble
    SequentialStack = model_composer_module.SequentialStack
    HierarchicalComposition = model_composer_module.HierarchicalComposition

    # --- Publish into this module's namespace so tests see them ---
    module.ModelSpec = ModelSpec
    module.EnsembleConfig = EnsembleConfig
    module.CompositionError = CompositionError
    module.ModelComposer = ModelComposer
    module.ParallelEnsemble = ParallelEnsemble
    module.SequentialStack = SequentialStack
    module.HierarchicalComposition = HierarchicalComposition


def teardown_module(module):
    """Cleanup mocked modules to prevent pollution."""
    for module_name in _mock_modules:
        if module_name in sys.modules:
            if module_name in _original_modules:
                sys.modules[module_name] = _original_modules[module_name]
            else:
                del sys.modules[module_name]


# =============================================================================
# MOCK MODELS FOR TESTING
# =============================================================================


class SimpleGCN(nn.Module):
    """Simple GCN model for testing."""

    def __init__(self, in_channels=16, out_channels=1):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, 64)
        self.lin2 = nn.Linear(64, out_channels)

    def forward(self, x, edge_index=None, batch=None):
        x = torch.relu(self.lin1(x))
        if batch is not None:
            batch_size = batch.max().item() + 1
            out = torch.zeros(batch_size, x.size(1), device=x.device)
            for i in range(batch_size):
                mask = batch == i
                out[i] = x[mask].mean(dim=0)
            x = out
        else:
            x = x.mean(dim=0, keepdim=True)
        return self.lin2(x)


class SimpleGAT(nn.Module):
    """Simple GAT model for testing."""

    def __init__(self, in_channels=16, out_channels=1):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, 64)
        self.lin2 = nn.Linear(64, out_channels)

    def forward(self, x, edge_index=None, batch=None):
        x = torch.relu(self.lin1(x))
        if batch is not None:
            batch_size = batch.max().item() + 1
            out = torch.zeros(batch_size, x.size(1), device=x.device)
            for i in range(batch_size):
                mask = batch == i
                out[i] = x[mask].max(dim=0)[0]
            x = out
        else:
            x = x.max(dim=0, keepdim=True)[0]
        return self.lin2(x)


class SimpleEncoder(nn.Module):
    """Encoder for hierarchical testing."""

    def __init__(self, in_channels=16, out_channels=32):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index=None, batch=None):
        return torch.relu(self.lin(x))


class SimpleDecoder(nn.Module):
    """Decoder for hierarchical testing."""

    def __init__(self, in_channels=32, out_channels=1):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index=None, batch=None):
        # Handle both node-level and already-pooled graph-level inputs
        if batch is not None and x.size(0) > 1 and x.size(0) == batch.size(0):
            # Node-level input: pool to graph-level
            batch_size = batch.max().item() + 1
            out = torch.zeros(batch_size, x.size(1), device=x.device)
            for i in range(batch_size):
                mask = batch == i
                out[i] = x[mask].mean(dim=0)
            x = out
        elif batch is None and x.size(0) > 1:
            # No batch info, take mean
            x = x.mean(dim=0, keepdim=True)
        # else: already pooled (x.size(0) != batch.size(0)), use as-is
        return self.lin(x)


# =============================================================================
# PYTEST FIXTURES
# =============================================================================


@pytest.fixture
def sample_input():
    """Fixture providing sample input tensors."""
    return {
        "x": torch.randn(10, 16),
        "edge_index": torch.randint(0, 10, (2, 20)),
        "batch": torch.zeros(10, dtype=torch.long),
    }


# =============================================================================
# INTEGRATION TEST CLASS
# =============================================================================


class TestModelComposerIntegration:
    """Integration tests for ModelComposer."""

    # =========================================================================
    # TEST 1: BASIC MODEL COMPOSER CREATION
    # =========================================================================
    def test_basic_creation(self):
        """Test basic ModelComposer creation."""
        composer = ModelComposer(task_type="graph_regression", name="TestEnsemble")

        assert composer.task_type == "graph_regression"
        assert composer.name == "TestEnsemble"
        assert composer.strategy == "parallel"
        assert composer.fusion == "mean"
        assert len(composer) == 0

    # =========================================================================
    # TEST 2: ADD MODELS
    # =========================================================================
    def test_add_models(self):
        """Test adding models to composer."""
        composer = ModelComposer(task_type="graph_regression", name="TestEnsemble")
        model1 = SimpleGCN(in_channels=16, out_channels=1)
        model2 = SimpleGAT(in_channels=16, out_channels=1)

        composer.add_model(model1, weight=0.6, name="GCN")
        composer.add_model(model2, weight=0.4, name="GAT")

        assert len(composer) == 2
        assert composer.models[0].name == "GCN"
        assert composer.models[1].name == "GAT"

    # =========================================================================
    # TEST 3: SET STRATEGY AND FUSION
    # =========================================================================
    def test_configuration(self):
        """Test strategy and fusion configuration."""
        composer = ModelComposer(task_type="graph_regression")

        composer.set_strategy("parallel")
        assert composer.strategy == "parallel"

        composer.set_fusion("weighted")
        assert composer.fusion == "weighted"

    # =========================================================================
    # TEST 4: VALIDATION
    # =========================================================================
    def test_validation(self):
        """Test composition validation."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(SimpleGCN(), weight=0.6, name="GCN")
        composer.add_model(SimpleGAT(), weight=0.4, name="GAT")
        composer.set_strategy("parallel")
        composer.set_fusion("weighted")

        validation = composer.validate_composition()

        assert validation["valid"] is True
        assert isinstance(validation["errors"], list)
        assert isinstance(validation["warnings"], list)

    # =========================================================================
    # TEST 5: BUILD PARALLEL ENSEMBLE
    # =========================================================================
    def test_build_parallel_ensemble(self):
        """Test building parallel ensemble."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(SimpleGCN(), weight=0.6, name="GCN")
        composer.add_model(SimpleGAT(), weight=0.4, name="GAT")
        composer.set_strategy("parallel")
        composer.set_fusion("weighted")

        ensemble = composer.build()

        assert isinstance(ensemble, ParallelEnsemble)
        assert len(ensemble.models) == 2

    # =========================================================================
    # TEST 6: FORWARD PASS - PARALLEL
    # =========================================================================
    def test_forward_pass_parallel(self, sample_input):
        """Test forward pass with parallel ensemble."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(SimpleGCN(), weight=0.6, name="GCN")
        composer.add_model(SimpleGAT(), weight=0.4, name="GAT")
        composer.set_strategy("parallel")
        composer.set_fusion("weighted")

        ensemble = composer.build()

        with torch.no_grad():
            output = ensemble(
                sample_input["x"], sample_input["edge_index"], batch=sample_input["batch"]
            )

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 1  # Single graph

    # =========================================================================
    # TEST 7: TEST ALL FUSION METHODS
    # =========================================================================
    @pytest.mark.parametrize("fusion_method", ["mean", "weighted", "attention", "voting"])
    def test_all_fusion_methods(self, sample_input, fusion_method):
        """Test all fusion methods."""
        composer = ModelComposer(task_type="graph_regression", name=f"Test_{fusion_method}")
        composer.add_model(SimpleGCN(), weight=0.5, name="GCN")
        composer.add_model(SimpleGAT(), weight=0.5, name="GAT")
        composer.set_strategy("parallel")
        composer.set_fusion(fusion_method)

        ensemble = composer.build()

        with torch.no_grad():
            output = ensemble(
                sample_input["x"], sample_input["edge_index"], batch=sample_input["batch"]
            )

        assert isinstance(output, torch.Tensor)

    # =========================================================================
    # TEST 8: SEQUENTIAL STACK
    # =========================================================================
    def test_sequential_stack(self, sample_input):
        """Test sequential stack composition."""
        composer = ModelComposer(task_type="graph_regression", name="SequentialTest")

        encoder = SimpleEncoder(in_channels=16, out_channels=32)
        decoder = SimpleDecoder(in_channels=32, out_channels=1)

        composer.add_model(encoder, name="Encoder")
        composer.add_model(decoder, name="Decoder")
        composer.set_strategy("sequential")

        validation = composer.validate_composition()
        assert validation["valid"] is True

        stack = composer.build()
        assert isinstance(stack, SequentialStack)

        with torch.no_grad():
            output = stack(
                sample_input["x"], sample_input["edge_index"], batch=sample_input["batch"]
            )

        assert isinstance(output, torch.Tensor)

    # =========================================================================
    # TEST 9: HIERARCHICAL COMPOSITION
    # =========================================================================
    def test_hierarchical_composition(self, sample_input):
        """Test hierarchical composition."""
        composer = ModelComposer(task_type="graph_regression", name="HierarchicalTest")

        enc1 = SimpleEncoder(16, 32)
        enc2 = SimpleEncoder(16, 32)
        dec = SimpleDecoder(32, 1)

        composer.add_model(enc1, weight=0.5, name="Encoder1", level=0)
        composer.add_model(enc2, weight=0.5, name="Encoder2", level=0)
        composer.add_model(dec, weight=1.0, name="Decoder", level=1)
        composer.set_strategy("hierarchical")
        composer.set_fusion("mean")

        validation = composer.validate_composition()
        assert validation["valid"] is True

        hierarchy = composer.build()
        assert isinstance(hierarchy, HierarchicalComposition)

        with torch.no_grad():
            output = hierarchy(
                sample_input["x"], sample_input["edge_index"], batch=sample_input["batch"]
            )

        assert isinstance(output, torch.Tensor)

    # =========================================================================
    # TEST 10: CONFIG EXPORT/IMPORT
    # =========================================================================
    def test_config_export_import(self):
        """Test configuration export and import."""
        composer = ModelComposer(task_type="graph_regression", name="TestEnsemble")
        composer.add_model(SimpleGCN(), weight=0.6, name="GCN")
        composer.add_model(SimpleGAT(), weight=0.4, name="GAT")
        composer.set_strategy("parallel")
        composer.set_fusion("weighted")

        config = composer.to_config()
        assert config.name == "TestEnsemble"
        assert config.task_type == "graph_regression"
        assert config.strategy == "parallel"
        assert config.fusion == "weighted"

        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)

        new_composer = ModelComposer.from_config(config)
        assert new_composer.name == "TestEnsemble"
        assert new_composer.strategy == "parallel"

    # =========================================================================
    # TEST 11: ERROR HANDLING
    # =========================================================================
    def test_error_handling_invalid_strategy(self):
        """Test error handling for invalid strategy."""
        composer = ModelComposer(task_type="graph_regression")

        with pytest.raises(ValueError):
            composer.set_strategy("invalid_strategy")

    def test_error_handling_invalid_fusion(self):
        """Test error handling for invalid fusion."""
        composer = ModelComposer(task_type="graph_regression")

        with pytest.raises(ValueError):
            composer.set_fusion("invalid_fusion")

    def test_error_handling_build_without_models(self):
        """Test error handling for building without models."""
        composer = ModelComposer(task_type="graph_regression")

        with pytest.raises(CompositionError):
            composer.build()

    def test_error_handling_negative_weight(self):
        """Test error handling for negative weight."""
        composer = ModelComposer(task_type="graph_regression")

        with pytest.raises(ValueError):
            composer.add_model(SimpleGCN(), weight=-1.0)

    # =========================================================================
    # TEST 12: UTILITY METHODS
    # =========================================================================
    def test_utility_methods(self):
        """Test utility methods."""
        composer = ModelComposer(task_type="graph_regression", name="TestEnsemble")
        composer.add_model(SimpleGCN(), name="GCN")
        composer.add_model(SimpleGAT(), name="GAT")

        assert len(composer) == 2
        assert "TestEnsemble" in repr(composer)

        summary = composer.summary()
        assert isinstance(summary, str)
        assert "GCN" in summary

    # =========================================================================
    # TEST 13: REMOVE/CLEAR MODELS
    # =========================================================================
    def test_model_management(self):
        """Test model management (remove/clear)."""
        composer = ModelComposer(task_type="graph_regression", name="ManagementTest")
        composer.add_model(SimpleGCN(), name="Model1")
        composer.add_model(SimpleGAT(), name="Model2")
        composer.add_model(SimpleGCN(), name="Model3")

        assert len(composer) == 3

        composer.remove_model(1)
        assert len(composer) == 2

        composer.clear_models()
        assert len(composer) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
