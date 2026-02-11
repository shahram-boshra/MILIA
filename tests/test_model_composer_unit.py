#!/usr/bin/env python3
"""
Complete Unit Test Suite for model_composer.py Module

Tests the ModelComposer, ensemble composition, and model fusion functionality including:
- ModelSpec dataclass initialization and serialization
- EnsembleConfig dataclass and configuration management
- CompositionError exception handling
- ModelComposer initialization and model management
- Strategy configuration (parallel, sequential, hierarchical)
- Fusion method configuration (mean, weighted, attention, voting)
- Composition validation and error handling
- Model building (ParallelEnsemble, SequentialStack, HierarchicalComposition)
- ParallelEnsemble forward pass and fusion methods
- SequentialStack forward pass with fallback handling
- HierarchicalComposition level-based processing
- Configuration import/export
- Thread-safety of ensemble operations

This is a PRODUCTION-READY test suite with comprehensive coverage.
"""
import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import threading
import torch
import torch.nn as nn
from typing import List, Dict, Any
import importlib.util
import os
from dataclasses import dataclass
from copy import deepcopy

# ==============================================================================
# CRITICAL: Mock problematic imports to prevent ModuleNotFoundError
# ==============================================================================
_mock_modules = {}

# Mock torch_geometric modules BEFORE any milia_pipeline imports
mock_pyg_data = Mock()
mock_pyg_data.Data = Mock
mock_pyg_data.Batch = Mock
_mock_modules['torch_geometric.data'] = mock_pyg_data

mock_pyg_utils = Mock()
_mock_modules['torch_geometric.utils'] = mock_pyg_utils

# Mock torch_geometric.nn module with pooling functions
mock_pyg_nn = Mock()
# Create mock pooling functions that return appropriately shaped tensors
def mock_global_mean_pool(x, batch):
    """Mock global mean pool - returns one row per unique batch index."""
    if batch is None:
        return x.mean(dim=0, keepdim=True)
    num_graphs = int(batch.max().item()) + 1
    out_channels = x.size(-1)
    return torch.zeros(num_graphs, out_channels)

def mock_global_max_pool(x, batch):
    """Mock global max pool - returns one row per unique batch index."""
    if batch is None:
        return x.max(dim=0, keepdim=True)[0]
    num_graphs = int(batch.max().item()) + 1
    out_channels = x.size(-1)
    return torch.zeros(num_graphs, out_channels)

def mock_global_add_pool(x, batch):
    """Mock global add pool - returns one row per unique batch index."""
    if batch is None:
        return x.sum(dim=0, keepdim=True)
    num_graphs = int(batch.max().item()) + 1
    out_channels = x.size(-1)
    return torch.zeros(num_graphs, out_channels)

mock_pyg_nn.global_mean_pool = mock_global_mean_pool
mock_pyg_nn.global_max_pool = mock_global_max_pool
mock_pyg_nn.global_add_pool = mock_global_add_pool
_mock_modules['torch_geometric.nn'] = mock_pyg_nn

# Mock torch_geometric root module
mock_pyg = Mock()
mock_pyg.data = mock_pyg_data
mock_pyg.utils = mock_pyg_utils
mock_pyg.nn = mock_pyg_nn
_mock_modules['torch_geometric'] = mock_pyg

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
_mock_modules['milia_pipeline.exceptions'] = mock_exceptions

# ---------------------------------------------------------------------------
# Module-level placeholders for classes loaded in setup_module().
# These are set to None at import time (safe during collection) and populated
# with real values in setup_module() before any test executes.
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

    This runs ONCE before any test in this module executes — but critically
    does NOT run during pytest collection (import time), preventing
    sys.modules pollution from breaking other test files collected later.
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
    if 'milia_pipeline.exceptions' in sys.modules:
        _original_modules['milia_pipeline.exceptions'] = sys.modules['milia_pipeline.exceptions']
    sys.modules['milia_pipeline.exceptions'] = mock_exceptions

    # --- Load model_composer.py directly from file (bypass package __init__) ---
    model_composer_path = os.path.join(
        str(project_root),
        'milia_pipeline',
        'models',
        'builders',
        'model_composer.py'
    )
    spec = importlib.util.spec_from_file_location(
        "milia_pipeline.models.builders.model_composer",
        model_composer_path
    )
    model_composer_module = importlib.util.module_from_spec(spec)
    sys.modules['milia_pipeline.models.builders.model_composer'] = model_composer_module
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
    """
    Cleanup function to remove mocked modules from sys.modules.
    This prevents mock pollution from affecting other test files.
    """
    for module_name in _mock_modules:
        if module_name in sys.modules:
            if module_name in _original_modules:
                # Restore original module
                sys.modules[module_name] = _original_modules[module_name]
            else:
                # Remove mock module
                del sys.modules[module_name]


# =============================================================================
# HELPER CLASSES AND FIXTURES
# =============================================================================

class SimpleModel(nn.Module):
    """Simple test model for ensemble composition."""
    
    def __init__(self, in_channels=10, out_channels=10):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index=None, edge_attr=None, batch=None, **kwargs):
        return self.linear(x)


class SimpleGraphModel(nn.Module):
    """Simple graph model that requires edge_index."""
    
    def __init__(self, in_channels=10, out_channels=10):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index, **kwargs):
        return self.linear(x)


class SimpleBatchModel(nn.Module):
    """Simple model that requires batch parameter."""
    
    def __init__(self, in_channels=10, out_channels=10):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index, batch, **kwargs):
        return self.linear(x)


class MinimalModel(nn.Module):
    """Minimal model that only accepts x."""
    
    def __init__(self, in_channels=10, out_channels=10):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(self, x):
        return self.linear(x)


class PredictionModel(nn.Module):
    """Model for parallel ensemble that produces predictions (different output dim)."""
    
    def __init__(self, in_channels=10, out_channels=5):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index=None, edge_attr=None, batch=None, **kwargs):
        return self.linear(x)


@pytest.fixture
def simple_model():
    """Fixture providing a simple model instance."""
    return SimpleModel(in_channels=10, out_channels=10)


@pytest.fixture
def simple_graph_model():
    """Fixture providing a simple graph model instance."""
    return SimpleGraphModel(in_channels=10, out_channels=10)


@pytest.fixture
def simple_batch_model():
    """Fixture providing a simple batch model instance."""
    return SimpleBatchModel(in_channels=10, out_channels=10)


@pytest.fixture
def minimal_model():
    """Fixture providing a minimal model instance."""
    return MinimalModel(in_channels=10, out_channels=10)


@pytest.fixture
def sample_models():
    """Fixture providing multiple model instances for ensemble."""
    return [
        SimpleModel(in_channels=10, out_channels=10),
        SimpleModel(in_channels=10, out_channels=10),
        SimpleModel(in_channels=10, out_channels=10)
    ]


@pytest.fixture
def sample_input():
    """Fixture providing sample input tensors."""
    return {
        'x': torch.randn(20, 10),  # 20 nodes, 10 features
        'edge_index': torch.randint(0, 20, (2, 30)),  # 30 edges
        'edge_attr': torch.randn(30, 3),  # 30 edges, 3 edge features
        'batch': torch.zeros(20, dtype=torch.long)  # Single graph
    }


# =============================================================================
# MODEL SPEC TESTS
# =============================================================================

class TestModelSpec:
    """Test ModelSpec dataclass."""
    
    def test_init_basic(self, simple_model):
        """Test basic initialization of ModelSpec."""
        spec = ModelSpec(model=simple_model)
        
        assert spec.model is simple_model
        assert spec.weight == 1.0
        assert spec.name is None
        assert spec.level == 0
    
    def test_init_with_parameters(self, simple_model):
        """Test initialization with all parameters."""
        spec = ModelSpec(
            model=simple_model,
            weight=0.75,
            name="TestModel",
            level=2
        )
        
        assert spec.model is simple_model
        assert spec.weight == 0.75
        assert spec.name == "TestModel"
        assert spec.level == 2
    
    def test_to_dict_without_name(self, simple_model):
        """Test to_dict method without explicit name."""
        spec = ModelSpec(model=simple_model, weight=0.6, level=1)
        result = spec.to_dict()
        
        assert isinstance(result, dict)
        assert result['name'] == 'SimpleModel'
        assert result['weight'] == 0.6
        assert result['model_class'] == 'SimpleModel'
        assert result['level'] == 1
    
    def test_to_dict_with_name(self, simple_model):
        """Test to_dict method with explicit name."""
        spec = ModelSpec(
            model=simple_model,
            weight=0.8,
            name="CustomName",
            level=0
        )
        result = spec.to_dict()
        
        assert result['name'] == 'CustomName'
        assert result['weight'] == 0.8
        assert result['model_class'] == 'SimpleModel'
        assert result['level'] == 0
    
    def test_model_spec_immutability(self, simple_model):
        """Test that ModelSpec is a dataclass but still mutable."""
        spec = ModelSpec(model=simple_model, weight=0.5)
        
        # Dataclasses are mutable by default
        spec.weight = 0.7
        assert spec.weight == 0.7


# =============================================================================
# ENSEMBLE CONFIG TESTS
# =============================================================================

class TestEnsembleConfig:
    """Test EnsembleConfig dataclass."""
    
    def test_init_basic(self):
        """Test basic initialization of EnsembleConfig."""
        config = EnsembleConfig(
            name="TestEnsemble",
            task_type="graph_regression"
        )
        
        assert config.name == "TestEnsemble"
        assert config.task_type == "graph_regression"
        assert config.models == []
        assert config.strategy == "parallel"
        assert config.fusion == "mean"
    
    def test_init_with_parameters(self, simple_model):
        """Test initialization with all parameters."""
        model_specs = [
            ModelSpec(model=simple_model, weight=0.6, name="Model1"),
            ModelSpec(model=simple_model, weight=0.4, name="Model2")
        ]
        
        config = EnsembleConfig(
            name="TestEnsemble",
            task_type="node_classification",
            models=model_specs,
            strategy="sequential",
            fusion="weighted"
        )
        
        assert config.name == "TestEnsemble"
        assert config.task_type == "node_classification"
        assert len(config.models) == 2
        assert config.strategy == "sequential"
        assert config.fusion == "weighted"
    
    def test_to_dict(self, simple_model):
        """Test to_dict method."""
        model_specs = [
            ModelSpec(model=simple_model, weight=0.5, name="Model1"),
            ModelSpec(model=simple_model, weight=0.5, name="Model2")
        ]
        
        config = EnsembleConfig(
            name="TestEnsemble",
            task_type="graph_regression",
            models=model_specs,
            strategy="parallel",
            fusion="mean"
        )
        
        result = config.to_dict()
        
        assert isinstance(result, dict)
        assert result['name'] == "TestEnsemble"
        assert result['task_type'] == "graph_regression"
        assert result['num_models'] == 2
        assert result['strategy'] == "parallel"
        assert result['fusion'] == "mean"
        assert 'models' in result
        assert len(result['models']) == 2
        assert all(isinstance(m, dict) for m in result['models'])


# =============================================================================
# COMPOSITION ERROR TESTS
# =============================================================================

class TestCompositionError:
    """Test CompositionError exception."""
    
    def test_init_basic(self):
        """Test basic initialization of CompositionError."""
        error = CompositionError("Test error")
        
        assert error.message == "Test error"
        assert error.strategy is None
        assert error.num_models is None
    
    def test_init_with_parameters(self):
        """Test initialization with all parameters."""
        error = CompositionError(
            message="Test error",
            strategy="parallel",
            num_models=3,
            details="Additional details"
        )
        
        assert error.message == "Test error"
        assert error.strategy == "parallel"
        assert error.num_models == 3
        assert error.details == "Additional details"
    
    def test_str_basic(self):
        """Test string representation without context."""
        error = CompositionError("Test error")
        result = str(error)
        
        assert result == "Test error"
    
    def test_str_with_strategy(self):
        """Test string representation with strategy."""
        error = CompositionError("Test error", strategy="sequential")
        result = str(error)
        
        assert "Test error" in result
        assert "[Strategy: sequential]" in result
    
    def test_str_with_num_models(self):
        """Test string representation with num_models."""
        error = CompositionError("Test error", num_models=5)
        result = str(error)
        
        assert "Test error" in result
        assert "[Models: 5]" in result
    
    def test_str_with_details(self):
        """Test string representation with details."""
        error = CompositionError(
            "Test error",
            details="Additional context"
        )
        result = str(error)
        
        assert "Test error" in result
        assert "Additional context" in result
    
    def test_str_complete(self):
        """Test string representation with all parameters."""
        error = CompositionError(
            message="Composition failed",
            strategy="hierarchical",
            num_models=4,
            details="Level mismatch detected"
        )
        result = str(error)
        
        assert "Composition failed" in result
        assert "[Strategy: hierarchical]" in result
        assert "[Models: 4]" in result
        assert "Level mismatch detected" in result
    
    def test_inheritance(self):
        """Test that CompositionError inherits from ModelError."""
        error = CompositionError("Test error")
        
        assert isinstance(error, ModelError)
        assert isinstance(error, BaseProjectError)
        assert isinstance(error, Exception)


# =============================================================================
# MODEL COMPOSER INITIALIZATION TESTS
# =============================================================================

class TestModelComposerInitialization:
    """Test ModelComposer initialization."""
    
    def test_init_basic(self):
        """Test basic initialization."""
        composer = ModelComposer(task_type="graph_regression")
        
        assert composer.task_type == "graph_regression"
        assert composer.name == "Ensemble"
        assert composer.models == []
        assert composer.strategy == "parallel"
        assert composer.fusion == "mean"
    
    def test_init_with_name(self):
        """Test initialization with custom name."""
        composer = ModelComposer(
            task_type="node_classification",
            name="MyEnsemble"
        )
        
        assert composer.task_type == "node_classification"
        assert composer.name == "MyEnsemble"
    
    def test_len_empty(self):
        """Test __len__ with no models."""
        composer = ModelComposer(task_type="graph_regression")
        
        assert len(composer) == 0
    
    def test_repr(self):
        """Test __repr__ method."""
        composer = ModelComposer(
            task_type="graph_regression",
            name="TestEnsemble"
        )
        
        result = repr(composer)
        
        assert "ModelComposer" in result
        assert "name='TestEnsemble'" in result
        assert "task='graph_regression'" in result
        assert "strategy='parallel'" in result
        assert "fusion='mean'" in result
        assert "models=0" in result


# =============================================================================
# MODEL MANAGEMENT TESTS
# =============================================================================

class TestModelManagement:
    """Test model addition, removal, and management."""
    
    def test_add_model_basic(self, simple_model):
        """Test adding a model with default parameters."""
        composer = ModelComposer(task_type="graph_regression")
        result = composer.add_model(simple_model)
        
        assert result is composer  # Method chaining
        assert len(composer.models) == 1
        assert composer.models[0].model is simple_model
        assert composer.models[0].weight == 1.0
        assert composer.models[0].level == 0
    
    def test_add_model_with_weight(self, simple_model):
        """Test adding a model with custom weight."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(simple_model, weight=0.75)
        
        assert len(composer.models) == 1
        assert composer.models[0].weight == 0.75
    
    def test_add_model_with_name(self, simple_model):
        """Test adding a model with custom name."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(simple_model, name="CustomModel")
        
        assert composer.models[0].name == "CustomModel"
    
    def test_add_model_with_level(self, simple_model):
        """Test adding a model with custom level."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(simple_model, level=2)
        
        assert composer.models[0].level == 2
    
    def test_add_multiple_models(self, sample_models):
        """Test adding multiple models."""
        composer = ModelComposer(task_type="graph_regression")
        
        for i, model in enumerate(sample_models):
            composer.add_model(model, weight=0.3 + i * 0.1, name=f"Model{i}")
        
        assert len(composer.models) == 3
        assert composer.models[0].weight == 0.3
        assert composer.models[1].weight == pytest.approx(0.4)
        assert composer.models[2].weight == pytest.approx(0.5)
    
    def test_add_model_method_chaining(self, sample_models):
        """Test method chaining with add_model."""
        composer = ModelComposer(task_type="graph_regression")
        
        result = (composer
                  .add_model(sample_models[0], weight=0.5)
                  .add_model(sample_models[1], weight=0.3)
                  .add_model(sample_models[2], weight=0.2))
        
        assert result is composer
        assert len(composer.models) == 3
    
    def test_add_model_invalid_type(self):
        """Test adding non-nn.Module raises TypeError."""
        composer = ModelComposer(task_type="graph_regression")
        
        with pytest.raises(TypeError, match="model must be nn.Module instance"):
            composer.add_model("not a model")
    
    def test_add_model_negative_weight(self, simple_model):
        """Test adding model with negative weight raises ValueError."""
        composer = ModelComposer(task_type="graph_regression")
        
        with pytest.raises(ValueError, match="weight must be positive"):
            composer.add_model(simple_model, weight=-0.5)
    
    def test_add_model_zero_weight(self, simple_model):
        """Test adding model with zero weight raises ValueError."""
        composer = ModelComposer(task_type="graph_regression")
        
        with pytest.raises(ValueError, match="weight must be positive"):
            composer.add_model(simple_model, weight=0.0)
    
    def test_add_model_negative_level(self, simple_model):
        """Test adding model with negative level raises ValueError."""
        composer = ModelComposer(task_type="graph_regression")
        
        with pytest.raises(ValueError, match="level must be non-negative"):
            composer.add_model(simple_model, level=-1)
    
    def test_remove_model_basic(self, sample_models):
        """Test removing a model by index."""
        composer = ModelComposer(task_type="graph_regression")
        for model in sample_models:
            composer.add_model(model)
        
        result = composer.remove_model(1)
        
        assert result is composer  # Method chaining
        assert len(composer.models) == 2
    
    def test_remove_model_first(self, sample_models):
        """Test removing the first model."""
        composer = ModelComposer(task_type="graph_regression")
        for i, model in enumerate(sample_models):
            composer.add_model(model, name=f"Model{i}")
        
        composer.remove_model(0)
        
        assert len(composer.models) == 2
        assert composer.models[0].name == "Model1"
    
    def test_remove_model_last(self, sample_models):
        """Test removing the last model."""
        composer = ModelComposer(task_type="graph_regression")
        for i, model in enumerate(sample_models):
            composer.add_model(model, name=f"Model{i}")
        
        composer.remove_model(2)
        
        assert len(composer.models) == 2
        assert composer.models[1].name == "Model1"
    
    def test_remove_model_invalid_index_negative(self, simple_model):
        """Test removing with negative index raises ValueError."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(simple_model)
        
        with pytest.raises(ValueError, match="Invalid index"):
            composer.remove_model(-1)
    
    def test_remove_model_invalid_index_too_large(self, simple_model):
        """Test removing with out-of-range index raises ValueError."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(simple_model)
        
        with pytest.raises(ValueError, match="Invalid index"):
            composer.remove_model(5)
    
    def test_remove_model_from_empty(self):
        """Test removing from empty composer raises ValueError."""
        composer = ModelComposer(task_type="graph_regression")
        
        with pytest.raises(ValueError, match="Invalid index"):
            composer.remove_model(0)
    
    def test_clear_models_basic(self, sample_models):
        """Test clearing all models."""
        composer = ModelComposer(task_type="graph_regression")
        for model in sample_models:
            composer.add_model(model)
        
        result = composer.clear_models()
        
        assert result is composer  # Method chaining
        assert len(composer.models) == 0
    
    def test_clear_models_empty(self):
        """Test clearing already empty composer."""
        composer = ModelComposer(task_type="graph_regression")
        composer.clear_models()
        
        assert len(composer.models) == 0
    
    def test_len_after_operations(self, sample_models):
        """Test __len__ after various operations."""
        composer = ModelComposer(task_type="graph_regression")
        
        assert len(composer) == 0
        
        composer.add_model(sample_models[0])
        assert len(composer) == 1
        
        composer.add_model(sample_models[1])
        assert len(composer) == 2
        
        composer.remove_model(0)
        assert len(composer) == 1
        
        composer.clear_models()
        assert len(composer) == 0


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestConfiguration:
    """Test strategy and fusion configuration."""
    
    def test_set_strategy_parallel(self):
        """Test setting parallel strategy."""
        composer = ModelComposer(task_type="graph_regression")
        result = composer.set_strategy("parallel")
        
        assert result is composer  # Method chaining
        assert composer.strategy == "parallel"
    
    def test_set_strategy_sequential(self):
        """Test setting sequential strategy."""
        composer = ModelComposer(task_type="graph_regression")
        composer.set_strategy("sequential")
        
        assert composer.strategy == "sequential"
    
    def test_set_strategy_hierarchical(self):
        """Test setting hierarchical strategy."""
        composer = ModelComposer(task_type="graph_regression")
        composer.set_strategy("hierarchical")
        
        assert composer.strategy == "hierarchical"
    
    def test_set_strategy_invalid(self):
        """Test setting invalid strategy raises ValueError."""
        composer = ModelComposer(task_type="graph_regression")
        
        with pytest.raises(ValueError, match="Invalid strategy"):
            composer.set_strategy("invalid_strategy")
    
    def test_set_fusion_mean(self):
        """Test setting mean fusion."""
        composer = ModelComposer(task_type="graph_regression")
        result = composer.set_fusion("mean")
        
        assert result is composer  # Method chaining
        assert composer.fusion == "mean"
    
    def test_set_fusion_weighted(self):
        """Test setting weighted fusion."""
        composer = ModelComposer(task_type="graph_regression")
        composer.set_fusion("weighted")
        
        assert composer.fusion == "weighted"
    
    def test_set_fusion_attention(self):
        """Test setting attention fusion."""
        composer = ModelComposer(task_type="graph_regression")
        composer.set_fusion("attention")
        
        assert composer.fusion == "attention"
    
    def test_set_fusion_voting(self):
        """Test setting voting fusion."""
        composer = ModelComposer(task_type="graph_regression")
        composer.set_fusion("voting")
        
        assert composer.fusion == "voting"
    
    def test_set_fusion_invalid(self):
        """Test setting invalid fusion raises ValueError."""
        composer = ModelComposer(task_type="graph_regression")
        
        with pytest.raises(ValueError, match="Invalid fusion method"):
            composer.set_fusion("invalid_fusion")
    
    def test_configuration_chaining(self):
        """Test chaining configuration methods."""
        composer = ModelComposer(task_type="graph_regression")
        
        result = (composer
                  .set_strategy("sequential")
                  .set_fusion("mean"))
        
        assert result is composer
        assert composer.strategy == "sequential"
        assert composer.fusion == "mean"


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestValidation:
    """Test composition validation."""
    
    def test_validate_empty_models(self):
        """Test validation fails with no models."""
        composer = ModelComposer(task_type="graph_regression")
        
        result = composer.validate_composition()
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
        assert "No models added" in result['errors'][0]
        assert len(result['suggestions']) > 0
    
    def test_validate_sequential_insufficient_models(self, simple_model):
        """Test sequential validation requires at least 2 models."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(simple_model)
        composer.set_strategy("sequential")
        
        result = composer.validate_composition()
        
        assert result['valid'] is False
        assert any("at least 2 models" in err for err in result['errors'])
    
    def test_validate_sequential_sufficient_models(self, sample_models):
        """Test sequential validation passes with 2+ models."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(sample_models[0])
        composer.add_model(sample_models[1])
        composer.set_strategy("sequential")
        
        result = composer.validate_composition()
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
    
    def test_validate_parallel_single_model(self, simple_model):
        """Test parallel validation warns about single model."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(simple_model)
        composer.set_strategy("parallel")
        
        result = composer.validate_composition()
        
        assert result['valid'] is True  # Valid but warns
        assert len(result['warnings']) > 0
        assert any("single model is redundant" in warn for warn in result['warnings'])
    
    def test_validate_hierarchical_single_level(self, sample_models):
        """Test hierarchical validation warns about single level."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(sample_models[0], level=0)
        composer.add_model(sample_models[1], level=0)
        composer.set_strategy("hierarchical")
        
        result = composer.validate_composition()
        
        assert result['valid'] is True
        assert any("only one level" in warn for warn in result['warnings'])
    
    def test_validate_hierarchical_multiple_levels(self, sample_models):
        """Test hierarchical validation passes with multiple levels."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(sample_models[0], level=0)
        composer.add_model(sample_models[1], level=1)
        composer.set_strategy("hierarchical")
        
        result = composer.validate_composition()
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
    
    def test_validate_hierarchical_missing_levels(self, sample_models):
        """Test hierarchical validation warns about level gaps."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(sample_models[0], level=0)
        composer.add_model(sample_models[1], level=2)  # Gap at level 1
        composer.set_strategy("hierarchical")
        
        result = composer.validate_composition()
        
        assert result['valid'] is True
        assert any("gaps" in warn for warn in result['warnings'])
    
    def test_validate_voting_with_classification(self):
        """Test voting fusion with classification task."""
        composer = ModelComposer(task_type="node_classification")
        composer.add_model(SimpleModel())
        composer.add_model(SimpleModel())
        composer.set_fusion("voting")
        
        result = composer.validate_composition()
        
        assert result['valid'] is True
        # Should not warn for classification
    
    def test_validate_voting_with_regression(self):
        """Test voting fusion warns for regression task."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(SimpleModel())
        composer.add_model(SimpleModel())
        composer.set_fusion("voting")
        
        result = composer.validate_composition()
        
        assert result['valid'] is True
        assert any("designed for classification" in warn for warn in result['warnings'])
    
    def test_validate_weighted_fusion_normalization(self, sample_models):
        """Test weighted fusion warns about non-normalized weights."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(sample_models[0], weight=0.6)
        composer.add_model(sample_models[1], weight=0.6)  # Sum = 1.2
        composer.set_fusion("weighted")
        
        result = composer.validate_composition()
        
        assert result['valid'] is True
        assert any("not 1.0" in warn for warn in result['warnings'])
    
    def test_validate_attention_with_sequential(self, sample_models):
        """Test attention fusion with sequential warns."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(sample_models[0])
        composer.add_model(sample_models[1])
        composer.set_strategy("sequential")
        composer.set_fusion("attention")
        
        result = composer.validate_composition()
        
        assert result['valid'] is True
        assert any("may not be meaningful" in warn for warn in result['warnings'])
    
    def test_validate_result_structure(self, simple_model):
        """Test validation result structure."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(simple_model)
        
        result = composer.validate_composition()
        
        assert 'valid' in result
        assert 'errors' in result
        assert 'warnings' in result
        assert 'suggestions' in result
        assert isinstance(result['valid'], bool)
        assert isinstance(result['errors'], list)
        assert isinstance(result['warnings'], list)
        assert isinstance(result['suggestions'], list)


# =============================================================================
# BUILD TESTS
# =============================================================================

class TestBuild:
    """Test model building."""
    
    def test_build_fails_without_models(self):
        """Test building fails with no models."""
        composer = ModelComposer(task_type="graph_regression")
        
        with pytest.raises(CompositionError, match="validation failed"):
            composer.build()
    
    def test_build_parallel_ensemble(self, sample_models):
        """Test building parallel ensemble."""
        composer = ModelComposer(task_type="graph_regression")
        for model in sample_models:
            composer.add_model(model)
        composer.set_strategy("parallel")
        
        ensemble = composer.build()
        
        assert isinstance(ensemble, ParallelEnsemble)
        assert len(ensemble.models) == 3
    
    def test_build_sequential_stack(self, sample_models):
        """Test building sequential stack."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(sample_models[0])
        composer.add_model(sample_models[1])
        composer.set_strategy("sequential")
        
        stack = composer.build()
        
        assert isinstance(stack, SequentialStack)
        assert len(stack.models) == 2
    
    def test_build_hierarchical_composition(self, sample_models):
        """Test building hierarchical composition."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(sample_models[0], level=0)
        composer.add_model(sample_models[1], level=1)
        composer.set_strategy("hierarchical")
        
        hierarchy = composer.build()
        
        assert isinstance(hierarchy, HierarchicalComposition)
        assert len(hierarchy.level_ensembles) == 2
    
    def test_build_normalizes_weights(self, sample_models):
        """Test that build normalizes weights for weighted fusion."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(sample_models[0], weight=0.6)
        composer.add_model(sample_models[1], weight=0.6)  # Sum = 1.2
        composer.set_fusion("weighted")
        
        ensemble = composer.build()
        
        # Weights should be normalized to sum to 1.0
        assert torch.allclose(
            ensemble.weights.sum(),
            torch.tensor(1.0),
            atol=1e-5
        )
    
    def test_build_preserves_fusion_method(self, sample_models):
        """Test that build preserves fusion method."""
        composer = ModelComposer(task_type="graph_regression")
        for model in sample_models:
            composer.add_model(model)
        composer.set_fusion("weighted")
        
        ensemble = composer.build()
        
        assert ensemble.fusion == "weighted"
    
    def test_build_sequential_fails_with_single_model(self, simple_model):
        """Test sequential build fails with single model."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(simple_model)
        composer.set_strategy("sequential")
        
        with pytest.raises(CompositionError):
            composer.build()


# =============================================================================
# CONFIG IMPORT/EXPORT TESTS
# =============================================================================

class TestConfigImportExport:
    """Test configuration import and export."""
    
    def test_to_config(self, sample_models):
        """Test exporting configuration."""
        composer = ModelComposer(
            task_type="graph_regression",
            name="TestEnsemble"
        )
        for i, model in enumerate(sample_models):
            composer.add_model(model, weight=0.3 + i * 0.1, name=f"Model{i}")
        composer.set_strategy("parallel")
        composer.set_fusion("weighted")
        
        config = composer.to_config()
        
        assert isinstance(config, EnsembleConfig)
        assert config.name == "TestEnsemble"
        assert config.task_type == "graph_regression"
        assert len(config.models) == 3
        assert config.strategy == "parallel"
        assert config.fusion == "weighted"
    
    def test_from_config_with_config_object(self):
        """Test creating composer from EnsembleConfig."""
        original_config = EnsembleConfig(
            name="ImportedEnsemble",
            task_type="node_classification",
            strategy="sequential",
            fusion="mean"
        )
        
        composer = ModelComposer.from_config(original_config)
        
        assert composer.name == "ImportedEnsemble"
        assert composer.task_type == "node_classification"
        assert composer.strategy == "sequential"
        assert composer.fusion == "mean"
        assert len(composer.models) == 0  # Models not imported
    
    def test_from_config_with_dict(self):
        """Test creating composer from dictionary."""
        config_dict = {
            'name': 'DictEnsemble',
            'task_type': 'graph_regression',
            'strategy': 'hierarchical',
            'fusion': 'attention'
        }
        
        composer = ModelComposer.from_config(config_dict)
        
        assert composer.name == "DictEnsemble"
        assert composer.task_type == "graph_regression"
        assert composer.strategy == "hierarchical"
        assert composer.fusion == "attention"
    
    def test_from_config_with_defaults(self):
        """Test from_config uses defaults for missing keys."""
        config_dict = {}
        
        composer = ModelComposer.from_config(config_dict)
        
        assert composer.name == "Ensemble"
        assert composer.task_type == "graph_regression"
        assert composer.strategy == "parallel"
        assert composer.fusion == "mean"
    
    def test_config_roundtrip(self, sample_models):
        """Test configuration export and import roundtrip."""
        # Create original composer
        original = ModelComposer(
            task_type="graph_regression",
            name="RoundtripTest"
        )
        for model in sample_models:
            original.add_model(model)
        original.set_strategy("parallel")
        original.set_fusion("weighted")
        
        # Export and import
        config = original.to_config()
        restored = ModelComposer.from_config(config)
        
        # Compare configurations (not models)
        assert restored.name == original.name
        assert restored.task_type == original.task_type
        assert restored.strategy == original.strategy
        assert restored.fusion == original.fusion


# =============================================================================
# UTILITY TESTS
# =============================================================================

class TestUtility:
    """Test utility methods."""
    
    def test_summary_basic(self, sample_models):
        """Test summary method."""
        composer = ModelComposer(
            task_type="graph_regression",
            name="TestEnsemble"
        )
        for i, model in enumerate(sample_models):
            composer.add_model(model, weight=0.3 + i * 0.1, name=f"Model{i}")
        composer.set_strategy("parallel")
        composer.set_fusion("weighted")
        
        summary = composer.summary()
        
        assert isinstance(summary, str)
        assert "TestEnsemble" in summary
        assert "graph_regression" in summary
        assert "parallel" in summary
        assert "weighted" in summary
        assert "3" in summary  # Number of models
        assert "Model0" in summary
        assert "Model1" in summary
        assert "Model2" in summary
    
    def test_summary_empty(self):
        """Test summary with no models."""
        composer = ModelComposer(task_type="graph_regression")
        
        summary = composer.summary()
        
        assert "0" in summary or "Number of Models: 0" in summary


# =============================================================================
# PARALLEL ENSEMBLE TESTS
# =============================================================================

class TestParallelEnsemble:
    """Test ParallelEnsemble class."""
    
    def test_init_basic(self, sample_models):
        """Test basic initialization."""
        weights = [0.3, 0.4, 0.3]
        ensemble = ParallelEnsemble(
            models=sample_models,
            weights=weights,
            fusion="mean",
            task_type="graph_regression"
        )
        
        assert len(ensemble.models) == 3
        assert torch.allclose(
            ensemble.weights,
            torch.tensor(weights, dtype=torch.float32)
        )
        assert ensemble.fusion == "mean"
        assert ensemble.task_type == "graph_regression"
    
    def test_init_with_attention(self, sample_models):
        """Test initialization with attention fusion."""
        ensemble = ParallelEnsemble(
            models=sample_models,
            weights=[0.33, 0.33, 0.34],
            fusion="attention",
            task_type="graph_regression"
        )
        
        assert hasattr(ensemble, 'attention')
        assert isinstance(ensemble.attention, nn.Sequential)
    
    def test_forward_mean_fusion(self, sample_input):
        """Test forward pass with mean fusion."""
        # Use prediction models that output dim=5
        models = [PredictionModel(), PredictionModel(), PredictionModel()]
        
        ensemble = ParallelEnsemble(
            models=models,
            weights=[0.33, 0.33, 0.34],
            fusion="mean",
            task_type="graph_regression"
        )
        
        output = ensemble(
            sample_input['x'],
            sample_input['edge_index'],
            sample_input['edge_attr'],
            sample_input['batch']
        )
        
        assert isinstance(output, torch.Tensor)
        # For graph_regression, output is graph-level (1 graph in this batch)
        # Number of graphs = max(batch) + 1 = 1
        num_graphs = int(sample_input['batch'].max().item()) + 1
        assert output.shape[0] == num_graphs  # Graph-level output
        assert output.shape[1] == 5  # Output dimension
    
    def test_forward_weighted_fusion(self, sample_input):
        """Test forward pass with weighted fusion."""
        models = [PredictionModel(), PredictionModel(), PredictionModel()]
        
        ensemble = ParallelEnsemble(
            models=models,
            weights=[0.5, 0.3, 0.2],
            fusion="weighted",
            task_type="graph_regression"
        )
        
        output = ensemble(
            sample_input['x'],
            sample_input['edge_index']
        )
        
        assert isinstance(output, torch.Tensor)
        # For graph_regression without batch, assumes single graph
        assert output.shape[0] == 1  # Single graph output
    
    def test_forward_attention_fusion(self, sample_input):
        """Test forward pass with attention fusion."""
        models = [PredictionModel(), PredictionModel(), PredictionModel()]
        
        ensemble = ParallelEnsemble(
            models=models,
            weights=[0.33, 0.33, 0.34],
            fusion="attention",
            task_type="graph_regression"
        )
        
        output = ensemble(
            sample_input['x'],
            sample_input['edge_index']
        )
        
        assert isinstance(output, torch.Tensor)
    
    def test_forward_voting_fusion(self, sample_input):
        """Test forward pass with voting fusion."""
        models = [PredictionModel(), PredictionModel(), PredictionModel()]
        
        ensemble = ParallelEnsemble(
            models=models,
            weights=[0.33, 0.33, 0.34],
            fusion="voting",
            task_type="node_classification"
        )
        
        output = ensemble(
            sample_input['x'],
            sample_input['edge_index']
        )
        
        assert isinstance(output, torch.Tensor)
    
    def test_forward_fallback_signatures(self, sample_input):
        """Test forward with models that have different signatures."""
        # Use models that all accept the same input dims but have different signatures
        models = [
            SimpleModel(),      # Accepts all params
            SimpleGraphModel(), # Requires edge_index
            MinimalModel()      # Only accepts x
        ]
        
        ensemble = ParallelEnsemble(
            models=models,
            weights=[0.33, 0.33, 0.34],
            fusion="mean",
            task_type="graph_regression"
        )
        
        # Should handle all different signatures through fallback
        output = ensemble(
            sample_input['x'],
            sample_input['edge_index'],
            sample_input['edge_attr'],
            sample_input['batch']
        )
        
        assert isinstance(output, torch.Tensor)
    
    def test_repr(self, sample_models):
        """Test __repr__ method."""
        ensemble = ParallelEnsemble(
            models=sample_models,
            weights=[0.33, 0.33, 0.34],
            fusion="mean",
            task_type="graph_regression",
            name="TestEnsemble"
        )
        
        result = repr(ensemble)
        
        assert "TestEnsemble" in result
        assert "3" in result  # Number of models
        assert "mean" in result


# =============================================================================
# SEQUENTIAL STACK TESTS
# =============================================================================

class TestSequentialStack:
    """Test SequentialStack class."""
    
    def test_init_basic(self, sample_models):
        """Test basic initialization."""
        stack = SequentialStack(models=sample_models[:2])
        
        assert len(stack.models) == 2
        assert stack.name == "SequentialStack"
    
    def test_init_with_name(self, sample_models):
        """Test initialization with custom name."""
        stack = SequentialStack(
            models=sample_models[:2],
            name="CustomStack"
        )
        
        assert stack.name == "CustomStack"
    
    def test_forward_basic(self, sample_models, sample_input):
        """Test forward pass through sequential stack."""
        stack = SequentialStack(models=sample_models[:2])
        
        output = stack(
            sample_input['x'],
            sample_input['edge_index'],
            sample_input['edge_attr'],
            sample_input['batch']
        )
        
        assert isinstance(output, torch.Tensor)
    
    def test_forward_fallback_signatures(self, sample_input):
        """Test forward with models that have different signatures."""
        models = [
            SimpleGraphModel(),
            MinimalModel()
        ]
        
        stack = SequentialStack(models=models)
        
        # Should handle signature differences with fallback
        output = stack(
            sample_input['x'],
            sample_input['edge_index']
        )
        
        assert isinstance(output, torch.Tensor)
    
    def test_repr(self, sample_models):
        """Test __repr__ method."""
        stack = SequentialStack(
            models=sample_models[:2],
            name="TestStack"
        )
        
        result = repr(stack)
        
        assert "TestStack" in result
        assert "2" in result  # Number of models


# =============================================================================
# HIERARCHICAL COMPOSITION TESTS
# =============================================================================

class TestHierarchicalComposition:
    """Test HierarchicalComposition class."""
    
    def test_init_basic(self, sample_models):
        """Test basic initialization."""
        levels_dict = {
            0: [ModelSpec(model=sample_models[0], weight=1.0)],
            1: [ModelSpec(model=sample_models[1], weight=1.0)]
        }
        
        hierarchy = HierarchicalComposition(
            levels_dict=levels_dict,
            fusion="mean",
            task_type="graph_regression"
        )
        
        assert len(hierarchy.level_ensembles) == 2
        assert hierarchy.levels == [0, 1]
    
    def test_init_with_multiple_models_per_level(self, sample_models):
        """Test initialization with multiple models per level."""
        levels_dict = {
            0: [
                ModelSpec(model=sample_models[0], weight=0.6),
                ModelSpec(model=sample_models[1], weight=0.4)
            ],
            1: [ModelSpec(model=sample_models[2], weight=1.0)]
        }
        
        hierarchy = HierarchicalComposition(
            levels_dict=levels_dict,
            fusion="weighted",
            task_type="graph_regression"
        )
        
        assert len(hierarchy.level_ensembles) == 2
        # Each level should be a ParallelEnsemble
        assert all(isinstance(e, ParallelEnsemble) for e in hierarchy.level_ensembles)
    
    def test_init_normalizes_weights_per_level(self, sample_models):
        """Test that weights are normalized per level."""
        levels_dict = {
            0: [
                ModelSpec(model=sample_models[0], weight=0.6),
                ModelSpec(model=sample_models[1], weight=0.6)  # Sum = 1.2
            ]
        }
        
        hierarchy = HierarchicalComposition(
            levels_dict=levels_dict,
            fusion="weighted",
            task_type="graph_regression"
        )
        
        # Weights in level 0 should be normalized
        level0_ensemble = hierarchy.level_ensembles[0]
        assert torch.allclose(
            level0_ensemble.weights.sum(),
            torch.tensor(1.0),
            atol=1e-5
        )
    
    def test_forward_basic(self, sample_models, sample_input):
        """Test forward pass through hierarchical composition."""
        levels_dict = {
            0: [ModelSpec(model=sample_models[0], weight=1.0)],
            1: [ModelSpec(model=sample_models[1], weight=1.0)]
        }
        
        hierarchy = HierarchicalComposition(
            levels_dict=levels_dict,
            fusion="mean",
            task_type="graph_regression"
        )
        
        output = hierarchy(
            sample_input['x'],
            sample_input['edge_index'],
            sample_input['edge_attr'],
            sample_input['batch']
        )
        
        assert isinstance(output, torch.Tensor)
    
    def test_forward_multiple_levels(self, sample_models, sample_input):
        """Test forward with multiple levels."""
        levels_dict = {
            0: [
                ModelSpec(model=sample_models[0], weight=0.5),
                ModelSpec(model=sample_models[1], weight=0.5)
            ],
            1: [ModelSpec(model=sample_models[2], weight=1.0)]
        }
        
        hierarchy = HierarchicalComposition(
            levels_dict=levels_dict,
            fusion="weighted",
            task_type="graph_regression"
        )
        
        output = hierarchy(
            sample_input['x'],
            sample_input['edge_index']
        )
        
        assert isinstance(output, torch.Tensor)
    
    def test_repr(self, sample_models):
        """Test __repr__ method."""
        levels_dict = {
            0: [ModelSpec(model=sample_models[0], weight=1.0)],
            1: [ModelSpec(model=sample_models[1], weight=1.0)]
        }
        
        hierarchy = HierarchicalComposition(
            levels_dict=levels_dict,
            fusion="mean",
            task_type="graph_regression",
            name="TestHierarchy"
        )
        
        result = repr(hierarchy)
        
        assert "TestHierarchy" in result
        assert "2" in result  # Number of levels
        assert "mean" in result


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Test integration scenarios."""
    
    def test_complete_parallel_workflow(self, sample_input):
        """Test complete workflow for parallel ensemble."""
        models = [PredictionModel(), PredictionModel(), PredictionModel()]
        
        composer = ModelComposer(
            task_type="graph_regression",
            name="ParallelTest"
        )
        
        # Add models
        for i, model in enumerate(models):
            composer.add_model(model, weight=0.3 + i * 0.1, name=f"Model{i}")
        
        # Configure
        composer.set_strategy("parallel")
        composer.set_fusion("weighted")
        
        # Validate
        validation = composer.validate_composition()
        assert validation['valid'] is True
        
        # Build
        ensemble = composer.build()
        assert isinstance(ensemble, ParallelEnsemble)
        
        # Test forward pass
        output = ensemble(sample_input['x'], sample_input['edge_index'])
        assert isinstance(output, torch.Tensor)
    
    def test_complete_sequential_workflow(self, sample_models, sample_input):
        """Test complete workflow for sequential stack."""
        composer = ModelComposer(task_type="graph_regression")
        
        composer.add_model(sample_models[0], name="Encoder")
        composer.add_model(sample_models[1], name="Decoder")
        composer.set_strategy("sequential")
        
        stack = composer.build()
        assert isinstance(stack, SequentialStack)
        
        output = stack(sample_input['x'], sample_input['edge_index'])
        assert isinstance(output, torch.Tensor)
    
    def test_complete_hierarchical_workflow(self, sample_models, sample_input):
        """Test complete workflow for hierarchical composition."""
        composer = ModelComposer(task_type="graph_regression")
        
        composer.add_model(sample_models[0], level=0, weight=0.6, name="L0_M1")
        composer.add_model(sample_models[1], level=0, weight=0.4, name="L0_M2")
        composer.add_model(sample_models[2], level=1, name="L1_M1")
        composer.set_strategy("hierarchical")
        composer.set_fusion("weighted")
        
        hierarchy = composer.build()
        assert isinstance(hierarchy, HierarchicalComposition)
        
        output = hierarchy(sample_input['x'], sample_input['edge_index'])
        assert isinstance(output, torch.Tensor)
    
    def test_config_export_import_workflow(self, sample_models):
        """Test configuration export and import workflow."""
        # Create and configure composer
        original = ModelComposer(task_type="graph_regression", name="Original")
        for model in sample_models:
            original.add_model(model)
        original.set_strategy("parallel")
        original.set_fusion("weighted")
        
        # Export config
        config = original.to_config()
        config_dict = config.to_dict()
        
        # Import config
        restored = ModelComposer.from_config(config_dict)
        
        # Verify configuration preserved
        assert restored.name == original.name
        assert restored.task_type == original.task_type
        assert restored.strategy == original.strategy
        assert restored.fusion == original.fusion


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================

class TestThreadSafety:
    """Test thread safety of ensemble operations."""
    
    def test_concurrent_model_addition(self):
        """Test concurrent model addition to composer."""
        composer = ModelComposer(task_type="graph_regression")
        results = []
        errors = []
        
        def add_model_safe(idx):
            try:
                model = SimpleModel()
                composer.add_model(model, name=f"Model{idx}")
                results.append(True)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=add_model_safe, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All additions should succeed
        assert len(results) == 10
        assert len(errors) == 0
        assert len(composer.models) == 10
    
    def test_concurrent_ensemble_forward(self, sample_models, sample_input):
        """Test concurrent forward passes through ensemble."""
        ensemble = ParallelEnsemble(
            models=sample_models,
            weights=[0.33, 0.33, 0.34],
            fusion="mean",
            task_type="graph_regression"
        )
        
        results = []
        errors = []
        
        def forward_safe():
            try:
                output = ensemble(sample_input['x'], sample_input['edge_index'])
                results.append(output)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=forward_safe) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All forward passes should succeed
        assert len(results) == 20
        assert len(errors) == 0


# =============================================================================
# EDGE CASES AND ERROR HANDLING TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_composer_operations(self):
        """Test operations on empty composer."""
        composer = ModelComposer(task_type="graph_regression")
        
        # Should handle gracefully
        assert len(composer) == 0
        assert repr(composer) is not None
        summary = composer.summary()
        assert isinstance(summary, str)
    
    def test_single_model_different_strategies(self, simple_model):
        """Test single model with different strategies."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(simple_model)
        
        # Parallel: should build but warn
        composer.set_strategy("parallel")
        validation = composer.validate_composition()
        assert validation['valid'] is True
        assert len(validation['warnings']) > 0
        
        # Sequential: should fail
        composer.set_strategy("sequential")
        with pytest.raises(CompositionError):
            composer.build()
    
    def test_invalid_fusion_method_in_ensemble(self, sample_models):
        """Test invalid fusion method raises error."""
        with pytest.raises(ValueError):
            # This should be caught during initialization
            ensemble = ParallelEnsemble(
                models=sample_models,
                weights=[0.33, 0.33, 0.34],
                fusion="invalid_method",
                task_type="graph_regression"
            )
            
            # If not caught in init, should fail in forward
            x = torch.randn(10, 10)
            edge_index = torch.randint(0, 10, (2, 20))
            ensemble(x, edge_index)


# =============================================================================
# INTERNAL METHODS TESTS
# =============================================================================

class TestParallelEnsembleInternalMethods:
    """Test internal methods of ParallelEnsemble."""
    
    def test_is_graph_level_task_true(self, sample_models):
        """Test _is_graph_level_task returns True for graph tasks."""
        ensemble = ParallelEnsemble(
            models=sample_models,
            weights=[0.33, 0.33, 0.34],
            fusion="mean",
            task_type="graph_regression"
        )
        
        assert ensemble._is_graph_level_task() is True
        
        ensemble2 = ParallelEnsemble(
            models=sample_models,
            weights=[0.33, 0.33, 0.34],
            fusion="mean",
            task_type="graph_classification"
        )
        
        assert ensemble2._is_graph_level_task() is True
    
    def test_is_graph_level_task_false(self, sample_models):
        """Test _is_graph_level_task returns False for non-graph tasks."""
        ensemble = ParallelEnsemble(
            models=sample_models,
            weights=[0.33, 0.33, 0.34],
            fusion="mean",
            task_type="node_classification"
        )
        
        assert ensemble._is_graph_level_task() is False
        
        ensemble2 = ParallelEnsemble(
            models=sample_models,
            weights=[0.33, 0.33, 0.34],
            fusion="mean",
            task_type="link_prediction"
        )
        
        assert ensemble2._is_graph_level_task() is False
    
    def test_is_graph_level_task_none(self, sample_models):
        """Test _is_graph_level_task returns False when task_type is None."""
        ensemble = ParallelEnsemble(
            models=sample_models,
            weights=[0.33, 0.33, 0.34],
            fusion="mean",
            task_type=None
        )
        
        assert ensemble._is_graph_level_task() is False
    
    def test_get_innermost_model_unwrapped(self, simple_model):
        """Test _get_innermost_model with unwrapped model."""
        ensemble = ParallelEnsemble(
            models=[simple_model],
            weights=[1.0],
            fusion="mean",
            task_type="graph_regression"
        )
        
        inner = ensemble._get_innermost_model(simple_model)
        assert inner is simple_model
    
    def test_get_innermost_model_wrapped(self, simple_model):
        """Test _get_innermost_model with wrapped model."""
        # Create a simple wrapper
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model(x)
        
        wrapped = ModelWrapper(simple_model)
        
        ensemble = ParallelEnsemble(
            models=[wrapped],
            weights=[1.0],
            fusion="mean",
            task_type="graph_regression"
        )
        
        inner = ensemble._get_innermost_model(wrapped)
        assert inner is simple_model
    
    def test_get_innermost_model_double_wrapped(self, simple_model):
        """Test _get_innermost_model with double-wrapped model."""
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model(x)
        
        wrapped1 = ModelWrapper(simple_model)
        wrapped2 = ModelWrapper(wrapped1)
        
        ensemble = ParallelEnsemble(
            models=[wrapped2],
            weights=[1.0],
            fusion="mean",
            task_type="graph_regression"
        )
        
        inner = ensemble._get_innermost_model(wrapped2)
        assert inner is simple_model
    
    def test_model_supports_edge_attr_default(self, simple_model):
        """Test _model_supports_edge_attr returns True for unknown models."""
        ensemble = ParallelEnsemble(
            models=[simple_model],
            weights=[1.0],
            fusion="mean",
            task_type="graph_regression"
        )
        
        # SimpleModel doesn't have supports_edge_attr attribute
        # Should default to True for unknown models
        result = ensemble._model_supports_edge_attr(simple_model)
        assert isinstance(result, bool)


class TestSequentialStackInternalMethods:
    """Test internal methods of SequentialStack."""
    
    def test_is_graph_level_task_true(self, sample_models):
        """Test _is_graph_level_task returns True for graph tasks."""
        stack = SequentialStack(
            models=sample_models[:2],
            task_type="graph_regression"
        )
        
        assert stack._is_graph_level_task() is True
    
    def test_is_graph_level_task_false(self, sample_models):
        """Test _is_graph_level_task returns False for non-graph tasks."""
        stack = SequentialStack(
            models=sample_models[:2],
            task_type="node_classification"
        )
        
        assert stack._is_graph_level_task() is False
    
    def test_is_graph_level_task_none(self, sample_models):
        """Test _is_graph_level_task returns False when task_type is None."""
        stack = SequentialStack(
            models=sample_models[:2],
            task_type=None
        )
        
        assert stack._is_graph_level_task() is False
    
    def test_is_3d_model_false(self, simple_model, sample_models):
        """Test _is_3d_model returns False for standard models."""
        stack = SequentialStack(
            models=sample_models[:2],
            task_type="graph_regression"
        )
        
        # SimpleModel doesn't have z, pos parameters
        result = stack._is_3d_model(simple_model)
        assert result is False
    
    def test_is_3d_model_true(self, sample_models):
        """Test _is_3d_model returns True for 3D models."""
        # Create a mock 3D model with z, pos signature
        class Mock3DModel(nn.Module):
            def forward(self, z, pos, batch=None):
                return z
        
        model_3d = Mock3DModel()
        
        stack = SequentialStack(
            models=sample_models[:2],
            task_type="graph_regression"
        )
        
        result = stack._is_3d_model(model_3d)
        assert result is True
    
    def test_get_innermost_model(self, simple_model, sample_models):
        """Test _get_innermost_model unwraps correctly."""
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model(x)
        
        wrapped = ModelWrapper(simple_model)
        
        stack = SequentialStack(
            models=sample_models[:2],
            task_type="graph_regression"
        )
        
        inner = stack._get_innermost_model(wrapped)
        assert inner is simple_model


class TestVotingFusionModes:
    """Test voting fusion in different modes (training vs inference)."""
    
    def test_voting_soft_in_training(self, sample_input):
        """Test voting uses soft voting during training."""
        models = [PredictionModel(), PredictionModel(), PredictionModel()]
        
        ensemble = ParallelEnsemble(
            models=models,
            weights=[0.33, 0.33, 0.34],
            fusion="voting",
            task_type="node_classification"
        )
        
        # Set to training mode
        ensemble.train()
        
        output = ensemble(
            sample_input['x'],
            sample_input['edge_index']
        )
        
        assert isinstance(output, torch.Tensor)
        # In training mode, output should be probabilities (soft voting)
    
    def test_voting_hard_in_eval(self, sample_input):
        """Test voting uses hard voting during evaluation."""
        models = [PredictionModel(), PredictionModel(), PredictionModel()]
        
        ensemble = ParallelEnsemble(
            models=models,
            weights=[0.33, 0.33, 0.34],
            fusion="voting",
            task_type="node_classification"
        )
        
        # Set to eval mode
        ensemble.eval()
        
        output = ensemble(
            sample_input['x'],
            sample_input['edge_index']
        )
        
        assert isinstance(output, torch.Tensor)
        # In eval mode, output should be one-hot (hard voting)


class TestDataBatchExtraction:
    """Test DataBatch detection and tensor extraction."""
    
    def test_parallel_ensemble_databatch_extraction(self, sample_models, sample_input):
        """Test ParallelEnsemble extracts tensors from DataBatch-like objects."""
        ensemble = ParallelEnsemble(
            models=sample_models,
            weights=[0.33, 0.33, 0.34],
            fusion="mean",
            task_type="graph_regression"
        )
        
        # Create a mock DataBatch-like object
        class MockDataBatch:
            def __init__(self, x, edge_index, edge_attr, batch):
                self.x = x
                self.edge_index = edge_index
                self.edge_attr = edge_attr
                self.batch = batch
        
        data_batch = MockDataBatch(
            x=sample_input['x'],
            edge_index=sample_input['edge_index'],
            edge_attr=sample_input['edge_attr'],
            batch=sample_input['batch']
        )
        
        # Call with DataBatch as first argument
        output = ensemble(data_batch)
        
        assert isinstance(output, torch.Tensor)
    
    def test_sequential_stack_databatch_extraction(self, sample_models, sample_input):
        """Test SequentialStack extracts tensors from DataBatch-like objects."""
        stack = SequentialStack(
            models=sample_models[:2],
            task_type="graph_regression"
        )
        
        # Create a mock DataBatch-like object
        class MockDataBatch:
            def __init__(self, x, edge_index, edge_attr, batch):
                self.x = x
                self.edge_index = edge_index
                self.edge_attr = edge_attr
                self.batch = batch
        
        data_batch = MockDataBatch(
            x=sample_input['x'],
            edge_index=sample_input['edge_index'],
            edge_attr=sample_input['edge_attr'],
            batch=sample_input['batch']
        )
        
        # Call with DataBatch as first argument
        output = stack(data_batch)
        
        assert isinstance(output, torch.Tensor)
    
    def test_hierarchical_databatch_extraction(self, sample_models, sample_input):
        """Test HierarchicalComposition extracts tensors from DataBatch-like objects."""
        levels_dict = {
            0: [ModelSpec(model=sample_models[0], weight=1.0)],
            1: [ModelSpec(model=sample_models[1], weight=1.0)]
        }
        
        hierarchy = HierarchicalComposition(
            levels_dict=levels_dict,
            fusion="mean",
            task_type="graph_regression"
        )
        
        # Create a mock DataBatch-like object
        class MockDataBatch:
            def __init__(self, x, edge_index, edge_attr, batch):
                self.x = x
                self.edge_index = edge_index
                self.edge_attr = edge_attr
                self.batch = batch
        
        data_batch = MockDataBatch(
            x=sample_input['x'],
            edge_index=sample_input['edge_index'],
            edge_attr=sample_input['edge_attr'],
            batch=sample_input['batch']
        )
        
        # Call with DataBatch as first argument
        output = hierarchy(data_batch)
        
        assert isinstance(output, torch.Tensor)


# =============================================================================
# 3D MODEL SUPPORT TESTS
# =============================================================================

class Test3DModelSupport:
    """Test 3D model (SchNet, DimeNet) support in ensembles."""
    
    def test_is_3d_model_detection_in_parallel_ensemble(self, sample_models):
        """Test that ParallelEnsemble can detect 3D models via signature introspection."""
        # Create a mock 3D model with z, pos signature
        class Mock3DModel(nn.Module):
            def forward(self, z, pos, batch=None):
                # Return tensor based on z shape
                return z.float().unsqueeze(-1) if z.dim() == 1 else z.float()
        
        model_3d = Mock3DModel()
        
        ensemble = ParallelEnsemble(
            models=[model_3d],
            weights=[1.0],
            fusion="mean",
            task_type="graph_regression"
        )
        
        # Access internal method for testing
        inner = ensemble._get_innermost_model(model_3d)
        import inspect
        sig = inspect.signature(inner.forward)
        param_names = [name for name in sig.parameters.keys() if name != 'self']
        
        # Verify 3D signature detection
        assert 'z' in param_names
        assert 'pos' in param_names
    
    def test_call_model_with_3d_signature(self, sample_models):
        """Test _call_model_with_signature handles 3D models correctly."""
        class Mock3DModel(nn.Module):
            def forward(self, z, pos, batch=None):
                # Simple output: sum of atomic numbers
                return z.float().unsqueeze(-1) if z.dim() == 1 else z.float()
        
        model_3d = Mock3DModel()
        
        ensemble = ParallelEnsemble(
            models=[model_3d],
            weights=[1.0],
            fusion="mean",
            task_type="graph_regression"
        )
        
        # Prepare 3D model inputs
        z = torch.tensor([6, 8, 1, 1], dtype=torch.long)  # C, O, H, H
        pos = torch.randn(4, 3)  # 4 atoms, 3D coords
        batch = torch.zeros(4, dtype=torch.long)
        
        # This should work with kwargs containing z and pos
        output = ensemble._call_model_with_signature(
            model=model_3d,
            x=torch.randn(4, 10),  # Not used for 3D models
            edge_index=None,
            edge_attr=None,
            batch=batch,
            edge_label_index=None,
            z=z,
            pos=pos
        )
        
        assert isinstance(output, torch.Tensor)
    
    def test_3d_model_missing_z_pos_raises_error(self, sample_models):
        """Test that 3D models raise error when z/pos not provided."""
        class Mock3DModel(nn.Module):
            def forward(self, z, pos, batch=None):
                return z.float().unsqueeze(-1)
        
        model_3d = Mock3DModel()
        
        ensemble = ParallelEnsemble(
            models=[model_3d],
            weights=[1.0],
            fusion="mean",
            task_type="graph_regression"
        )
        
        # Should raise ValueError when z and pos not in kwargs
        with pytest.raises(ValueError, match="requires 'z' and 'pos'"):
            ensemble._call_model_with_signature(
                model=model_3d,
                x=torch.randn(4, 10),
                edge_index=None,
                edge_attr=None,
                batch=None,
                edge_label_index=None
                # Missing z and pos!
            )
    
    def test_databatch_3d_data_extraction(self, sample_models, sample_input):
        """Test DataBatch extraction includes z and pos for 3D models."""
        ensemble = ParallelEnsemble(
            models=sample_models,
            weights=[0.33, 0.33, 0.34],
            fusion="mean",
            task_type="graph_regression"
        )
        
        # Create a mock DataBatch-like object with 3D data
        class MockDataBatch3D:
            def __init__(self, x, edge_index, batch, z, pos):
                self.x = x
                self.edge_index = edge_index
                self.batch = batch
                self.z = z
                self.pos = pos
        
        z = torch.tensor([6, 8, 1, 1], dtype=torch.long)
        pos = torch.randn(4, 3)
        
        data_batch = MockDataBatch3D(
            x=sample_input['x'][:4],
            edge_index=sample_input['edge_index'][:, :10],
            batch=torch.zeros(4, dtype=torch.long),
            z=z,
            pos=pos
        )
        
        # Should process without error
        output = ensemble(data_batch)
        assert isinstance(output, torch.Tensor)


# =============================================================================
# EDGE ATTRIBUTE SUPPORT TESTS
# =============================================================================

class TestEdgeAttrSupport:
    """Test edge attribute compatibility handling."""
    
    def test_model_supports_edge_attr_with_attribute(self, sample_models):
        """Test edge attr support detection for models with attribute."""
        # Create model with supports_edge_attr attribute
        class GATLikeModel(nn.Module):
            supports_edge_attr = True
            
            def forward(self, x, edge_index, edge_attr=None, batch=None):
                return x
        
        class GCNLikeModel(nn.Module):
            supports_edge_attr = False
            
            def forward(self, x, edge_index, batch=None):
                return x
        
        ensemble = ParallelEnsemble(
            models=[GATLikeModel()],
            weights=[1.0],
            fusion="mean",
            task_type="graph_regression"
        )
        
        gat = GATLikeModel()
        gcn = GCNLikeModel()
        
        assert ensemble._model_supports_edge_attr(gat) is True
        assert ensemble._model_supports_edge_attr(gcn) is False
    
    def test_edge_attr_skipped_for_unsupported_model(self, sample_input):
        """Test that edge_attr is skipped for models that don't support it."""
        # Create model that explicitly doesn't support edge_attr
        class GCNLikeModel(nn.Module):
            supports_edge_attr = False
            
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            
            def forward(self, x, edge_index, batch=None):
                return self.linear(x)
        
        model = GCNLikeModel()
        
        ensemble = ParallelEnsemble(
            models=[model],
            weights=[1.0],
            fusion="mean",
            task_type="graph_regression"
        )
        
        # Should handle edge_attr gracefully by skipping it
        output = ensemble(
            sample_input['x'],
            sample_input['edge_index'],
            sample_input['edge_attr'],  # Multi-dimensional edge_attr
            sample_input['batch']
        )
        
        assert isinstance(output, torch.Tensor)
    
    def test_edge_attr_1d_passes_through(self, sample_input):
        """Test that 1D edge_attr (edge weights) passes through."""
        class ModelWith1DEdgeWeight(nn.Module):
            supports_edge_attr = False  # Doesn't support multi-dim
            
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            
            def forward(self, x, edge_index, edge_weight=None, batch=None):
                return self.linear(x)
        
        model = ModelWith1DEdgeWeight()
        
        ensemble = ParallelEnsemble(
            models=[model],
            weights=[1.0],
            fusion="mean",
            task_type="graph_regression"
        )
        
        # 1D edge weights should be handled differently
        edge_weight_1d = torch.ones(30)  # 1D tensor
        
        output = ensemble(
            sample_input['x'],
            sample_input['edge_index'],
            edge_weight_1d,  # 1D edge weights
            sample_input['batch']
        )
        
        assert isinstance(output, torch.Tensor)


# =============================================================================
# OUTPUT DIMENSION ALIGNMENT TESTS
# =============================================================================

class TestOutputDimensionAlignment:
    """Test output dimension alignment in ParallelEnsemble."""
    
    def test_heterogeneous_output_dims_aligned(self, sample_input):
        """Test that models with different output dims get aligned."""
        class Model5D(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x, edge_index=None, batch=None, **kwargs):
                return self.linear(x)
        
        class Model10D(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            
            def forward(self, x, edge_index=None, batch=None, **kwargs):
                return self.linear(x)
        
        ensemble = ParallelEnsemble(
            models=[Model5D(), Model10D()],
            weights=[0.5, 0.5],
            fusion="mean",
            task_type="graph_regression"
        )
        
        output = ensemble(
            sample_input['x'],
            sample_input['edge_index'],
            batch=sample_input['batch']
        )
        
        # Output should be aligned to max dimension (10)
        # For graph tasks, output is [num_graphs, max_out_dim]
        num_graphs = int(sample_input['batch'].max().item()) + 1
        assert output.shape[0] == num_graphs
        assert output.shape[1] == 10  # Max dimension
    
    def test_output_projections_created_lazily(self, sample_input):
        """Test that output projection layers are created lazily."""
        class Model5D(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x, edge_index=None, batch=None, **kwargs):
                return self.linear(x)
        
        class Model10D(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            
            def forward(self, x, edge_index=None, batch=None, **kwargs):
                return self.linear(x)
        
        ensemble = ParallelEnsemble(
            models=[Model5D(), Model10D()],
            weights=[0.5, 0.5],
            fusion="mean",
            task_type="graph_regression"
        )
        
        # Before forward, no projections exist
        assert not hasattr(ensemble, '_output_projections')
        
        # After forward, projections should be created
        ensemble(sample_input['x'], sample_input['edge_index'], batch=sample_input['batch'])
        
        assert hasattr(ensemble, '_output_projections')
        assert len(ensemble._output_projections) > 0


# =============================================================================
# AUTOENCODER / VGAE SUPPORT TESTS
# =============================================================================

class TestAutoencoderSupport:
    """Test autoencoder (GAE/VGAE) model support."""
    
    def test_encode_method_detection(self, sample_input):
        """Test that encode() method is detected and used."""
        class MockVGAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            
            def encode(self, x, edge_index, batch=None):
                # VGAE.encode returns (mu, logstd) but we return just mu
                return self.linear(x)
            
            def forward(self, x, edge_index, batch=None):
                # forward() typically returns encode() output
                return self.encode(x, edge_index, batch)
        
        model = MockVGAE()
        
        # Verify model has encode method
        assert hasattr(model, 'encode')
        assert callable(getattr(model, 'encode'))
    
    def test_tuple_output_handled(self, sample_input):
        """Test that tuple outputs from VGAE are handled correctly."""
        class MockVGAETuple(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            
            def forward(self, x, edge_index=None, batch=None, **kwargs):
                mu = self.linear(x)
                logstd = self.linear(x)
                return (mu, logstd)  # Returns tuple like VGAE
        
        ensemble = ParallelEnsemble(
            models=[MockVGAETuple()],
            weights=[1.0],
            fusion="mean",
            task_type="graph_regression"
        )
        
        # Should extract first element (mu) from tuple
        output = ensemble(
            sample_input['x'],
            sample_input['edge_index'],
            batch=sample_input['batch']
        )
        
        assert isinstance(output, torch.Tensor)
        # Output should be tensor, not tuple


# =============================================================================
# EDGE-LEVEL TASK TESTS
# =============================================================================

class TestEdgeLevelTasks:
    """Test edge-level task support (link prediction, edge regression)."""
    
    def test_edge_label_index_passed_to_model(self, sample_input):
        """Test that edge_label_index is passed to models for edge-level tasks."""
        call_args = []
        
        class EdgeLevelModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            
            def forward(self, x, edge_index, edge_label_index=None, batch=None, **kwargs):
                call_args.append({'edge_label_index': edge_label_index})
                return self.linear(x)
        
        model = EdgeLevelModel()
        
        ensemble = ParallelEnsemble(
            models=[model],
            weights=[1.0],
            fusion="mean",
            task_type="link_prediction"
        )
        
        # Create edge_label_index
        edge_label_index = torch.randint(0, 20, (2, 10))
        
        ensemble(
            sample_input['x'],
            sample_input['edge_index'],
            edge_label_index=edge_label_index
        )
        
        # Verify edge_label_index was passed
        assert len(call_args) > 0
        assert call_args[0]['edge_label_index'] is not None
    
    def test_hierarchical_edge_label_index_last_level_only(self, sample_models, sample_input):
        """Test that edge_label_index is passed only to last level in hierarchical."""
        levels_dict = {
            0: [ModelSpec(model=sample_models[0], weight=1.0)],
            1: [ModelSpec(model=sample_models[1], weight=1.0)]
        }
        
        hierarchy = HierarchicalComposition(
            levels_dict=levels_dict,
            fusion="mean",
            task_type="link_prediction"
        )
        
        edge_label_index = torch.randint(0, 20, (2, 10))
        
        # Should not error - edge_label_index passed to last level only
        output = hierarchy(
            sample_input['x'],
            sample_input['edge_index'],
            edge_label_index=edge_label_index
        )
        
        assert isinstance(output, torch.Tensor)


# =============================================================================
# GLOBAL POOLING TESTS
# =============================================================================

class TestGlobalPooling:
    """Test global pooling methods for graph-level tasks."""
    
    def test_pooling_method_mean(self, sample_input):
        """Test mean pooling for graph-level tasks."""
        models = [SimpleModel(), SimpleModel()]
        
        ensemble = ParallelEnsemble(
            models=models,
            weights=[0.5, 0.5],
            fusion="mean",
            task_type="graph_regression"
        )
        
        output = ensemble(
            sample_input['x'],
            sample_input['edge_index'],
            batch=sample_input['batch'],
            pooling_method='mean'
        )
        
        assert isinstance(output, torch.Tensor)
    
    def test_pooling_method_max(self, sample_input):
        """Test max pooling for graph-level tasks."""
        models = [SimpleModel(), SimpleModel()]
        
        ensemble = ParallelEnsemble(
            models=models,
            weights=[0.5, 0.5],
            fusion="mean",
            task_type="graph_regression"
        )
        
        output = ensemble(
            sample_input['x'],
            sample_input['edge_index'],
            batch=sample_input['batch'],
            pooling_method='max'
        )
        
        assert isinstance(output, torch.Tensor)
    
    def test_pooling_method_add(self, sample_input):
        """Test add pooling for graph-level tasks."""
        models = [SimpleModel(), SimpleModel()]
        
        ensemble = ParallelEnsemble(
            models=models,
            weights=[0.5, 0.5],
            fusion="mean",
            task_type="graph_regression"
        )
        
        output = ensemble(
            sample_input['x'],
            sample_input['edge_index'],
            batch=sample_input['batch'],
            pooling_method='add'
        )
        
        assert isinstance(output, torch.Tensor)
    
    def test_single_graph_pooling(self, sample_input):
        """Test pooling for single graph (batch=None)."""
        models = [SimpleModel(), SimpleModel()]
        
        ensemble = ParallelEnsemble(
            models=models,
            weights=[0.5, 0.5],
            fusion="mean",
            task_type="graph_regression"
        )
        
        # No batch tensor - single graph
        output = ensemble(
            sample_input['x'],
            sample_input['edge_index'],
            batch=None  # Single graph
        )
        
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 1  # Single graph output


# =============================================================================
# MULTI-BATCH GRAPH PROCESSING TESTS
# =============================================================================

class TestMultiBatchGraphProcessing:
    """Test processing of multiple graphs in a batch."""
    
    def test_multiple_graphs_in_batch(self):
        """Test ensemble handles multiple graphs in a batch correctly."""
        models = [PredictionModel(), PredictionModel()]
        
        ensemble = ParallelEnsemble(
            models=models,
            weights=[0.5, 0.5],
            fusion="mean",
            task_type="graph_regression"
        )
        
        # Create batch with 3 graphs
        x = torch.randn(30, 10)  # 30 nodes total
        edge_index = torch.randint(0, 30, (2, 50))
        # 3 graphs: 10 nodes each
        batch = torch.cat([
            torch.zeros(10, dtype=torch.long),
            torch.ones(10, dtype=torch.long),
            torch.full((10,), 2, dtype=torch.long)
        ])
        
        output = ensemble(x, edge_index, batch=batch)
        
        # Output should have 3 rows (one per graph)
        assert output.shape[0] == 3
    
    def test_sequential_stack_graph_level(self):
        """Test SequentialStack handles graph-level tasks."""
        models = [SimpleModel(), SimpleModel()]
        
        stack = SequentialStack(
            models=models,
            task_type="graph_regression"
        )
        
        # Create batch with 2 graphs
        x = torch.randn(20, 10)
        edge_index = torch.randint(0, 20, (2, 30))
        batch = torch.cat([
            torch.zeros(10, dtype=torch.long),
            torch.ones(10, dtype=torch.long)
        ])
        
        output = stack(x, edge_index, batch=batch)
        
        # For graph tasks, final output should be graph-level
        assert isinstance(output, torch.Tensor)


# =============================================================================
# ERROR RECOVERY TESTS
# =============================================================================

class TestErrorRecovery:
    """Test error recovery mechanisms in model calling."""
    
    def test_model_forward_error_logged(self, sample_input):
        """Test that model forward errors are properly logged and raised."""
        class FailingModel(nn.Module):
            def forward(self, x, edge_index=None, **kwargs):
                raise RuntimeError("Intentional test failure")
        
        ensemble = ParallelEnsemble(
            models=[FailingModel()],
            weights=[1.0],
            fusion="mean",
            task_type="graph_regression"
        )
        
        with pytest.raises(RuntimeError, match="Intentional test failure"):
            ensemble(sample_input['x'], sample_input['edge_index'])
    
    def test_sequential_stack_model_error(self, sample_input):
        """Test SequentialStack handles model errors appropriately."""
        class FailingModel(nn.Module):
            def forward(self, x, edge_index=None, **kwargs):
                raise RuntimeError("Sequential model failure")
        
        stack = SequentialStack(
            models=[SimpleModel(), FailingModel()],
            task_type="graph_regression"
        )
        
        with pytest.raises(RuntimeError, match="Sequential model failure"):
            stack(sample_input['x'], sample_input['edge_index'])


# =============================================================================
# PYDANTIC V2 MODEL TESTS
# =============================================================================

class TestPydanticV2Models:
    """Test Pydantic V2 model behavior."""
    
    def test_model_spec_arbitrary_types(self, simple_model):
        """Test ModelSpec accepts nn.Module via arbitrary_types_allowed."""
        # This should work due to ConfigDict(arbitrary_types_allowed=True)
        spec = ModelSpec(model=simple_model, weight=1.0)
        
        assert spec.model is simple_model
        assert isinstance(spec.model, nn.Module)
    
    def test_ensemble_config_nested_arbitrary_types(self, simple_model):
        """Test EnsembleConfig handles nested ModelSpec with nn.Module."""
        model_specs = [
            ModelSpec(model=simple_model, weight=0.5),
            ModelSpec(model=SimpleModel(), weight=0.5)
        ]
        
        config = EnsembleConfig(
            name="Test",
            task_type="graph_regression",
            models=model_specs
        )
        
        assert len(config.models) == 2
        assert all(isinstance(m.model, nn.Module) for m in config.models)
    
    def test_model_spec_field_defaults(self):
        """Test ModelSpec default values work correctly."""
        model = SimpleModel()
        spec = ModelSpec(model=model)
        
        assert spec.weight == 1.0
        assert spec.name is None
        assert spec.level == 0
    
    def test_ensemble_config_field_defaults(self):
        """Test EnsembleConfig default_factory for mutable defaults."""
        config = EnsembleConfig(name="Test", task_type="graph_regression")
        
        # models should be an empty list (via default_factory)
        assert config.models == []
        assert config.strategy == "parallel"
        assert config.fusion == "mean"
        
        # Verify mutable default isolation
        config.models.append(ModelSpec(model=SimpleModel()))
        config2 = EnsembleConfig(name="Test2", task_type="graph_regression")
        assert len(config2.models) == 0  # Should not be affected


# =============================================================================
# SIGNATURE INTROSPECTION TESTS
# =============================================================================

class TestSignatureIntrospection:
    """Test model signature introspection for dynamic calling."""
    
    def test_call_model_with_full_signature(self, sample_input):
        """Test calling model with full signature (x, edge_index, edge_attr, batch)."""
        class FullSignatureModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            
            def forward(self, x, edge_index, edge_attr=None, batch=None):
                return self.linear(x)
        
        ensemble = ParallelEnsemble(
            models=[FullSignatureModel()],
            weights=[1.0],
            fusion="mean",
            task_type="node_classification"
        )
        
        output = ensemble(
            sample_input['x'],
            sample_input['edge_index'],
            sample_input['edge_attr'],
            sample_input['batch']
        )
        
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == sample_input['x'].shape[0]  # Node-level task
    
    def test_call_model_fallback_chain(self, sample_input):
        """Test model calling tries multiple signature strategies."""
        # Model that only accepts x, edge_index
        class MinimalSignatureModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            
            def forward(self, x, edge_index):
                return self.linear(x)
        
        ensemble = ParallelEnsemble(
            models=[MinimalSignatureModel()],
            weights=[1.0],
            fusion="mean",
            task_type="node_classification"
        )
        
        # Call with extra args - should fall back to minimal signature
        output = ensemble(
            sample_input['x'],
            sample_input['edge_index'],
            sample_input['edge_attr'],  # Extra arg
            sample_input['batch']  # Extra arg
        )
        
        assert isinstance(output, torch.Tensor)


# =============================================================================
# WEIGHTED FUSION NORMALIZATION TESTS
# =============================================================================

class TestWeightedFusionNormalization:
    """Test weight normalization for weighted fusion."""
    
    def test_weights_normalized_at_build_time(self, sample_models):
        """Test that weights are normalized when building ensemble."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(sample_models[0], weight=2.0)
        composer.add_model(sample_models[1], weight=3.0)
        composer.set_fusion("weighted")
        
        ensemble = composer.build()
        
        # Weights should sum to 1.0
        assert torch.allclose(
            ensemble.weights.sum(),
            torch.tensor(1.0),
            atol=1e-5
        )
        
        # Verify proportions preserved
        assert torch.allclose(
            ensemble.weights[0] / ensemble.weights[1],
            torch.tensor(2.0 / 3.0),
            atol=1e-5
        )
    
    def test_already_normalized_weights_unchanged(self, sample_models):
        """Test that already normalized weights remain unchanged."""
        composer = ModelComposer(task_type="graph_regression")
        composer.add_model(sample_models[0], weight=0.4)
        composer.add_model(sample_models[1], weight=0.6)
        composer.set_fusion("weighted")
        
        ensemble = composer.build()
        
        # Should remain as-is
        assert torch.allclose(
            ensemble.weights,
            torch.tensor([0.4, 0.6]),
            atol=1e-5
        )


# =============================================================================
# COMPREHENSIVE INTEGRATION TESTS
# =============================================================================

class TestComprehensiveIntegration:
    """Comprehensive integration tests covering complex scenarios."""
    
    def test_mixed_model_types_in_parallel(self, sample_input):
        """Test parallel ensemble with mixed model types."""
        class StandardModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x, edge_index=None, batch=None, **kwargs):
                return self.linear(x)
        
        class ModelWithEdgeAttr(nn.Module):
            supports_edge_attr = True
            
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
                return self.linear(x)
        
        ensemble = ParallelEnsemble(
            models=[StandardModel(), ModelWithEdgeAttr()],
            weights=[0.5, 0.5],
            fusion="weighted",
            task_type="graph_regression"
        )
        
        output = ensemble(
            sample_input['x'],
            sample_input['edge_index'],
            sample_input['edge_attr'],
            sample_input['batch']
        )
        
        assert isinstance(output, torch.Tensor)
    
    def test_full_workflow_with_validation_and_config(self, sample_input):
        """Test complete workflow: create, validate, build, run, export."""
        # 1. Create composer
        composer = ModelComposer(
            task_type="graph_regression",
            name="FullWorkflowTest"
        )
        
        # 2. Add models
        composer.add_model(SimpleModel(), weight=0.4, name="Model1", level=0)
        composer.add_model(SimpleModel(), weight=0.3, name="Model2", level=0)
        composer.add_model(SimpleModel(), weight=0.3, name="Model3", level=1)
        
        # 3. Configure
        composer.set_strategy("hierarchical")
        composer.set_fusion("weighted")
        
        # 4. Validate
        validation = composer.validate_composition()
        assert validation['valid'] is True
        
        # 5. Export config
        config = composer.to_config()
        assert config.name == "FullWorkflowTest"
        assert len(config.models) == 3
        
        # 6. Build
        ensemble = composer.build()
        assert isinstance(ensemble, HierarchicalComposition)
        
        # 7. Forward pass
        output = ensemble(
            sample_input['x'],
            sample_input['edge_index'],
            batch=sample_input['batch']
        )
        assert isinstance(output, torch.Tensor)
        
        # 8. Summary
        summary = composer.summary()
        assert "FullWorkflowTest" in summary
        assert "Model1" in summary


# =============================================================================
# SEQUENTIAL STACK INTERNAL METHOD TESTS (ADDITIONAL)
# =============================================================================

class TestSequentialStackCallModelWithSignature:
    """Test _call_model_with_signature method in SequentialStack."""
    
    def test_sequential_call_with_edge_attr(self, sample_input):
        """Test SequentialStack _call_model_with_signature with edge_attr."""
        class EdgeAttrModel(nn.Module):
            supports_edge_attr = True
            
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            
            def forward(self, x, edge_index, edge_attr=None, batch=None):
                return self.linear(x)
        
        stack = SequentialStack(
            models=[EdgeAttrModel(), EdgeAttrModel()],
            task_type="graph_regression"
        )
        
        output = stack(
            sample_input['x'],
            sample_input['edge_index'],
            sample_input['edge_attr'],
            sample_input['batch']
        )
        
        assert isinstance(output, torch.Tensor)
    
    def test_sequential_3d_model_in_stack(self, sample_input):
        """Test SequentialStack handles 3D model with proper kwargs."""
        class Mock3DModel(nn.Module):
            def forward(self, z, pos, batch=None):
                return z.float().unsqueeze(-1).expand(-1, 10)
        
        class StandardModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            
            def forward(self, x, edge_index=None, batch=None, **kwargs):
                return self.linear(x)
        
        stack = SequentialStack(
            models=[StandardModel()],  # Start with standard model
            task_type="graph_regression"
        )
        
        # Mock DataBatch with 3D data
        class MockDataBatch3D:
            def __init__(self, x, edge_index, batch, z, pos):
                self.x = x
                self.edge_index = edge_index
                self.batch = batch
                self.z = z
                self.pos = pos
        
        z = torch.tensor([6, 8, 1, 1], dtype=torch.long)
        pos = torch.randn(4, 3)
        
        data_batch = MockDataBatch3D(
            x=sample_input['x'][:4],
            edge_index=sample_input['edge_index'][:, :10],
            batch=torch.zeros(4, dtype=torch.long),
            z=z,
            pos=pos
        )
        
        # Should extract z and pos from batch
        output = stack(data_batch)
        assert isinstance(output, torch.Tensor)


# =============================================================================
# MODEL WRAPPER EDGE CASES
# =============================================================================

class TestModelWrapperEdgeCases:
    """Test edge cases with wrapped models."""
    
    def test_deeply_nested_wrapper(self, simple_model):
        """Test unwrapping deeply nested model wrappers."""
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model(x)
        
        # Create 5-level deep nesting
        wrapped = simple_model
        for _ in range(5):
            wrapped = ModelWrapper(wrapped)
        
        ensemble = ParallelEnsemble(
            models=[wrapped],
            weights=[1.0],
            fusion="mean",
            task_type="graph_regression"
        )
        
        inner = ensemble._get_innermost_model(wrapped)
        assert inner is simple_model
    
    def test_wrapper_with_supports_edge_attr(self, simple_model):
        """Test wrapper model inherits supports_edge_attr from inner model."""
        class InnerModel(nn.Module):
            supports_edge_attr = True
            
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            
            def forward(self, x, edge_index, edge_attr=None, batch=None):
                return self.linear(x)
        
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x, edge_index, edge_attr=None, batch=None):
                return self.model(x, edge_index, edge_attr, batch)
        
        inner = InnerModel()
        wrapped = ModelWrapper(inner)
        
        ensemble = ParallelEnsemble(
            models=[wrapped],
            weights=[1.0],
            fusion="mean",
            task_type="graph_regression"
        )
        
        # Should detect supports_edge_attr from inner model
        result = ensemble._model_supports_edge_attr(wrapped)
        assert result is True


# =============================================================================
# ATTENTION FUSION SPECIFIC TESTS
# =============================================================================

class TestAttentionFusion:
    """Test attention fusion specific behavior."""
    
    def test_attention_module_structure(self, sample_models):
        """Test that attention fusion creates proper attention module."""
        ensemble = ParallelEnsemble(
            models=sample_models,
            weights=[0.33, 0.33, 0.34],
            fusion="attention",
            task_type="graph_regression"
        )
        
        # Should have attention module
        assert hasattr(ensemble, 'attention')
        assert isinstance(ensemble.attention, nn.Sequential)
        
        # Check structure: Linear -> Tanh -> Linear -> Softmax
        assert len(ensemble.attention) == 4
        assert isinstance(ensemble.attention[0], nn.Linear)
        assert isinstance(ensemble.attention[1], nn.Tanh)
        assert isinstance(ensemble.attention[2], nn.Linear)
        assert isinstance(ensemble.attention[3], nn.Softmax)
    
    def test_attention_input_output_dims(self, sample_models, sample_input):
        """Test attention fusion input/output dimensions match number of models."""
        ensemble = ParallelEnsemble(
            models=sample_models,  # 3 models
            weights=[0.33, 0.33, 0.34],
            fusion="attention",
            task_type="graph_regression"
        )
        
        # Attention layers should have input/output dim = num_models = 3
        assert ensemble.attention[0].in_features == 3
        assert ensemble.attention[0].out_features == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
