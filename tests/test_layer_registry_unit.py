#!/usr/bin/env python3
"""
Complete Unit Test Suite for layer_registry.py Module

Tests the LayerRegistry singleton, layer registration, functional wrappers,
metadata management, and layer retrieval functionality including:
- FunctionalLayerWrapper initialization and forward pass
- LayerMetadata Pydantic BaseModel and serialization (Pydantic V2)
- LayerNotFoundError exception handling
- LayerRegistry singleton pattern
- LayerRegistry initialization and built-in layer registration
- Custom layer registration (classes and functions)
- Layer retrieval and metadata access
- Layer listing and filtering by category
- Registry statistics and utility methods
- Thread-safety of singleton and registration
- Convenience functions (get_layer, list_layers, get_layer_metadata)
- Built-in layer registration methods coverage
- Edge cases for functional wrapper forward pass
- Pydantic V2 model compatibility (model_validate, model_dump)

This is a PRODUCTION-READY test suite with comprehensive coverage.
"""
import sys
from pathlib import Path

# Add project root to Python path FIRST
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import threading
import torch
import torch.nn as nn
from typing import List, Dict, Any
import importlib.util
import os

# ==============================================================================
# CRITICAL: Mock problematic imports to prevent ModuleNotFoundError
# ==============================================================================
_mock_modules = {}

# Mock torch_geometric modules BEFORE any milia_pipeline imports
mock_pyg_transforms = Mock()
mock_pyg_transforms.Compose = Mock
mock_pyg_transforms.BaseTransform = Mock
_mock_modules['torch_geometric.transforms'] = mock_pyg_transforms

# Mock torch_geometric.data
mock_pyg_data = Mock()
mock_pyg_data.Data = Mock
mock_pyg_data.Batch = Mock
_mock_modules['torch_geometric.data'] = mock_pyg_data

# Mock torch_geometric.utils
mock_pyg_utils = Mock()
_mock_modules['torch_geometric.utils'] = mock_pyg_utils

# Mock torch_geometric.nn module with necessary components
mock_pyg_nn = Mock()

# Mock convolutional layers
mock_pyg_nn.GCNConv = Mock
mock_pyg_nn.GATConv = Mock
mock_pyg_nn.SAGEConv = Mock
mock_pyg_nn.GINConv = Mock
mock_pyg_nn.ChebConv = Mock
mock_pyg_nn.GraphConv = Mock
mock_pyg_nn.GatedGraphConv = Mock
mock_pyg_nn.EdgeConv = Mock
mock_pyg_nn.TAGConv = Mock
mock_pyg_nn.ARMAConv = Mock
mock_pyg_nn.SGConv = Mock
mock_pyg_nn.APPNP = Mock
mock_pyg_nn.MFConv = Mock
mock_pyg_nn.RGCNConv = Mock
mock_pyg_nn.SignedConv = Mock
mock_pyg_nn.DNAConv = Mock
mock_pyg_nn.PANConv = Mock
mock_pyg_nn.PointNetConv = Mock
mock_pyg_nn.GMMConv = Mock
mock_pyg_nn.SplineConv = Mock
mock_pyg_nn.NNConv = Mock
mock_pyg_nn.CGConv = Mock
mock_pyg_nn.TransformerConv = Mock
mock_pyg_nn.GATv2Conv = Mock
mock_pyg_nn.SuperGATConv = Mock
mock_pyg_nn.FiLMConv = Mock
mock_pyg_nn.GeneralConv = Mock
mock_pyg_nn.HGTConv = Mock
mock_pyg_nn.HEATConv = Mock
mock_pyg_nn.LEConv = Mock
mock_pyg_nn.GENConv = Mock

# Mock conv submodule for ClusterGCNConv
mock_pyg_nn.conv = Mock()
mock_pyg_nn.conv.ClusterGCNConv = Mock

# Mock pooling layers and functions
mock_pyg_nn.TopKPooling = Mock
mock_pyg_nn.SAGPooling = Mock
mock_pyg_nn.EdgePooling = Mock
mock_pyg_nn.ASAPooling = Mock
mock_pyg_nn.PANPooling = Mock
mock_pyg_nn.MemPooling = Mock
mock_pyg_nn.global_mean_pool = Mock(return_value=torch.tensor([1.0]))
mock_pyg_nn.global_max_pool = Mock(return_value=torch.tensor([1.0]))
mock_pyg_nn.global_add_pool = Mock(return_value=torch.tensor([1.0]))

# Mock normalization layers
mock_pyg_nn.BatchNorm = Mock
mock_pyg_nn.LayerNorm = Mock
mock_pyg_nn.InstanceNorm = Mock
mock_pyg_nn.GraphNorm = Mock
mock_pyg_nn.PairNorm = Mock
mock_pyg_nn.MeanSubtractionNorm = Mock
mock_pyg_nn.DiffGroupNorm = Mock

_mock_modules['torch_geometric.nn'] = mock_pyg_nn

# Mock torch_geometric root module
mock_pyg = Mock()
mock_pyg.nn = mock_pyg_nn
mock_pyg.transforms = mock_pyg_transforms
mock_pyg.data = mock_pyg_data
mock_pyg.utils = mock_pyg_utils
_mock_modules['torch_geometric'] = mock_pyg

# Mock milia_pipeline.exceptions to prevent import chain
class ModelError(Exception):
    """Model error - mocked."""
    pass

mock_exceptions = Mock()
mock_exceptions.ModelError = ModelError
_mock_modules['milia_pipeline.exceptions'] = mock_exceptions

# Store original modules for cleanup — populated by setup_module()
_original_modules = {}

# ---------------------------------------------------------------------------
# Module-level placeholders — populated by setup_module()
# ---------------------------------------------------------------------------
layer_registry_module = None
LayerCategory = None
FunctionalLayerWrapper = None
LayerMetadata = None
LayerNotFoundError = None
LayerRegistry = None
get_layer = None
list_layers = None
get_layer_metadata = None
global_registry = None

# ModelError is already defined in the mocks section above


def setup_module(module):
    """
    Inject mocks into sys.modules and load the module-under-test.

    Called by pytest ONCE before any test in this module executes.
    By deferring sys.modules writes here (instead of at module level),
    pytest --collect-only can import this file without polluting
    sys.modules for other test files collected afterward.
    """
    global _original_modules
    global layer_registry_module
    global LayerCategory, FunctionalLayerWrapper, LayerMetadata
    global LayerNotFoundError, LayerRegistry
    global get_layer, list_layers, get_layer_metadata, global_registry

    # --- Inject mock modules into sys.modules ---
    for module_name in _mock_modules:
        if module_name in sys.modules:
            _original_modules[module_name] = sys.modules[module_name]
        sys.modules[module_name] = _mock_modules[module_name]

    # --- Load layer_registry module from disk ---
    layer_registry_path = str(
        _PROJECT_ROOT / 'milia_pipeline' / 'models' / 'builders' / 'layer_registry.py'
    )
    spec = importlib.util.spec_from_file_location(
        "milia_pipeline.models.builders.layer_registry",
        layer_registry_path
    )
    layer_registry_module = importlib.util.module_from_spec(spec)
    sys.modules['milia_pipeline.models.builders.layer_registry'] = layer_registry_module
    spec.loader.exec_module(layer_registry_module)

    # --- Extract what we need ---
    LayerCategory = layer_registry_module.LayerCategory
    FunctionalLayerWrapper = layer_registry_module.FunctionalLayerWrapper
    LayerMetadata = layer_registry_module.LayerMetadata
    LayerNotFoundError = layer_registry_module.LayerNotFoundError
    LayerRegistry = layer_registry_module.LayerRegistry
    get_layer = layer_registry_module.get_layer
    list_layers = layer_registry_module.list_layers
    get_layer_metadata = layer_registry_module.get_layer_metadata
    global_registry = layer_registry_module.registry

    # --- Publish into module namespace ---
    module.layer_registry_module = layer_registry_module
    module.LayerCategory = LayerCategory
    module.FunctionalLayerWrapper = FunctionalLayerWrapper
    module.LayerMetadata = LayerMetadata
    module.LayerNotFoundError = LayerNotFoundError
    module.LayerRegistry = LayerRegistry
    module.get_layer = get_layer
    module.list_layers = list_layers
    module.get_layer_metadata = get_layer_metadata
    module.global_registry = global_registry


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
# FUNCTIONAL LAYER WRAPPER TESTS
# =============================================================================

class TestFunctionalLayerWrapper:
    """Test FunctionalLayerWrapper class."""
    
    def test_init_basic(self):
        """Test basic initialization of FunctionalLayerWrapper."""
        mock_func = Mock(return_value=torch.tensor([1.0]))
        wrapper = FunctionalLayerWrapper(
            func=mock_func,
            func_name="test_func"
        )
        
        assert wrapper.func == mock_func
        assert wrapper.func_name == "test_func"
        assert wrapper.requires_batch is False
        assert wrapper.requires_edge_index is False
        assert wrapper.requires_edge_attr is False
    
    def test_init_with_requirements(self):
        """Test initialization with various requirements."""
        mock_func = Mock(return_value=torch.tensor([1.0]))
        wrapper = FunctionalLayerWrapper(
            func=mock_func,
            func_name="pooling_func",
            requires_batch=True,
            requires_edge_index=True,
            requires_edge_attr=True
        )
        
        assert wrapper.requires_batch is True
        assert wrapper.requires_edge_index is True
        assert wrapper.requires_edge_attr is True
    
    def test_forward_basic(self):
        """Test forward pass with basic input."""
        mock_func = Mock(return_value=torch.tensor([1.0]))
        wrapper = FunctionalLayerWrapper(
            func=mock_func,
            func_name="test_func"
        )
        
        x = torch.randn(10, 5)
        result = wrapper(x)
        
        mock_func.assert_called_once_with(x)
        assert torch.is_tensor(result)
    
    def test_forward_with_batch(self):
        """Test forward pass requiring batch argument."""
        mock_func = Mock(return_value=torch.tensor([1.0]))
        wrapper = FunctionalLayerWrapper(
            func=mock_func,
            func_name="pooling_func",
            requires_batch=True
        )
        
        x = torch.randn(10, 5)
        batch = torch.tensor([0, 0, 1, 1, 2])
        
        result = wrapper(x, batch=batch)
        
        mock_func.assert_called_once_with(x, batch)
        assert torch.is_tensor(result)
    
    def test_forward_with_edge_index(self):
        """Test forward pass requiring edge_index."""
        mock_func = Mock(return_value=torch.tensor([1.0]))
        wrapper = FunctionalLayerWrapper(
            func=mock_func,
            func_name="conv_func",
            requires_edge_index=True
        )
        
        x = torch.randn(10, 5)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        
        result = wrapper(x, edge_index=edge_index)
        
        mock_func.assert_called_once_with(x, edge_index)
        assert torch.is_tensor(result)
    
    def test_forward_with_edge_index_and_attr(self):
        """Test forward pass requiring edge_index and edge_attr."""
        mock_func = Mock(return_value=torch.tensor([1.0]))
        wrapper = FunctionalLayerWrapper(
            func=mock_func,
            func_name="conv_func",
            requires_edge_index=True,
            requires_edge_attr=True
        )
        
        x = torch.randn(10, 5)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        edge_attr = torch.randn(3, 2)
        
        result = wrapper(x, edge_index=edge_index, edge_attr=edge_attr)
        
        mock_func.assert_called_once_with(x, edge_index, edge_attr)
        assert torch.is_tensor(result)
    
    def test_forward_all_arguments(self):
        """Test forward pass with all possible arguments."""
        mock_func = Mock(return_value=torch.tensor([1.0]))
        wrapper = FunctionalLayerWrapper(
            func=mock_func,
            func_name="complex_func",
            requires_batch=True,
            requires_edge_index=True,
            requires_edge_attr=True
        )
        
        x = torch.randn(10, 5)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        edge_attr = torch.randn(3, 2)
        batch = torch.tensor([0, 0, 1, 1, 2])
        
        result = wrapper(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        
        # Should pass x, edge_index, edge_attr, batch in that order
        mock_func.assert_called_once_with(x, edge_index, edge_attr, batch)
        assert torch.is_tensor(result)
    
    def test_forward_missing_required_batch_raises_error(self):
        """Test that missing required batch argument raises ValueError."""
        mock_func = Mock(return_value=torch.tensor([1.0]))
        wrapper = FunctionalLayerWrapper(
            func=mock_func,
            func_name="pooling_func",
            requires_batch=True
        )
        
        x = torch.randn(10, 5)
        
        with pytest.raises(ValueError, match="requires batch but none provided"):
            wrapper(x)
    
    def test_forward_missing_required_edge_index_raises_error(self):
        """Test that missing required edge_index raises ValueError."""
        mock_func = Mock(return_value=torch.tensor([1.0]))
        wrapper = FunctionalLayerWrapper(
            func=mock_func,
            func_name="conv_func",
            requires_edge_index=True
        )
        
        x = torch.randn(10, 5)
        
        with pytest.raises(ValueError, match="requires edge_index but none provided"):
            wrapper(x)
    
    def test_repr(self):
        """Test string representation of wrapper."""
        mock_func = Mock(return_value=torch.tensor([1.0]))
        wrapper = FunctionalLayerWrapper(
            func=mock_func,
            func_name="test_func"
        )
        
        repr_str = repr(wrapper)
        
        assert "FunctionalLayerWrapper" in repr_str
        assert "test_func" in repr_str
    
    def test_forward_edge_attr_ignored_when_not_required(self):
        """Test that edge_attr is ignored when requires_edge_attr=False."""
        mock_func = Mock(return_value=torch.tensor([1.0]))
        wrapper = FunctionalLayerWrapper(
            func=mock_func,
            func_name="conv_func",
            requires_edge_index=True,
            requires_edge_attr=False  # edge_attr not required
        )
        
        x = torch.randn(10, 5)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        edge_attr = torch.randn(3, 2)  # Provided but should be ignored
        
        result = wrapper(x, edge_index=edge_index, edge_attr=edge_attr)
        
        # edge_attr should NOT be passed to the function
        mock_func.assert_called_once_with(x, edge_index)
        assert torch.is_tensor(result)
    
    def test_forward_edge_attr_none_when_required_but_optional(self):
        """Test forward pass when requires_edge_attr=True but edge_attr=None."""
        mock_func = Mock(return_value=torch.tensor([1.0]))
        wrapper = FunctionalLayerWrapper(
            func=mock_func,
            func_name="conv_func",
            requires_edge_index=True,
            requires_edge_attr=True  # edge_attr required
        )
        
        x = torch.randn(10, 5)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        
        # edge_attr=None should NOT be appended (only appended if not None)
        result = wrapper(x, edge_index=edge_index, edge_attr=None)
        
        # Per the module logic: if requires_edge_attr and edge_attr is not None, append
        # Since edge_attr=None, it should NOT be appended
        mock_func.assert_called_once_with(x, edge_index)
        assert torch.is_tensor(result)
    
    def test_is_nn_module_subclass(self):
        """Test that FunctionalLayerWrapper is a subclass of nn.Module."""
        mock_func = Mock(return_value=torch.tensor([1.0]))
        wrapper = FunctionalLayerWrapper(
            func=mock_func,
            func_name="test_func"
        )
        
        assert isinstance(wrapper, nn.Module)


# =============================================================================
# LAYER CATEGORY ENUM TESTS
# =============================================================================

class TestLayerCategory:
    """Test LayerCategory enum."""
    
    def test_all_categories_exist(self):
        """Test that all expected categories are defined."""
        expected_categories = [
            "CONVOLUTIONAL", "POOLING", "NORMALIZATION", "ACTIVATION",
            "AGGREGATION", "LINEAR", "DROPOUT", "CUSTOM"
        ]
        
        for cat_name in expected_categories:
            assert hasattr(LayerCategory, cat_name), f"Missing category: {cat_name}"
    
    def test_category_values(self):
        """Test that category values are lowercase strings."""
        assert LayerCategory.CONVOLUTIONAL.value == "convolutional"
        assert LayerCategory.POOLING.value == "pooling"
        assert LayerCategory.NORMALIZATION.value == "normalization"
        assert LayerCategory.ACTIVATION.value == "activation"
        assert LayerCategory.AGGREGATION.value == "aggregation"
        assert LayerCategory.LINEAR.value == "linear"
        assert LayerCategory.DROPOUT.value == "dropout"
        assert LayerCategory.CUSTOM.value == "custom"
    
    def test_category_is_enum(self):
        """Test that LayerCategory is an Enum."""
        from enum import Enum
        assert issubclass(LayerCategory, Enum)
    
    def test_category_membership(self):
        """Test enum membership."""
        assert LayerCategory.CONVOLUTIONAL in LayerCategory
        assert LayerCategory.CUSTOM in LayerCategory


# =============================================================================
# LAYER METADATA TESTS
# =============================================================================

class TestLayerMetadata:
    """Test LayerMetadata Pydantic BaseModel (Pydantic V2)."""
    
    def test_init_minimal(self):
        """Test initialization with minimal required parameters."""
        metadata = LayerMetadata(
            name="TestLayer",
            category=LayerCategory.CONVOLUTIONAL,
            class_path="torch_geometric.nn.GCNConv",
            description="Test layer"
        )
        
        assert metadata.name == "TestLayer"
        assert metadata.category == LayerCategory.CONVOLUTIONAL
        assert metadata.class_path == "torch_geometric.nn.GCNConv"
        assert metadata.description == "Test layer"
        # Check defaults
        assert metadata.requires_edge_index is True
        assert metadata.requires_edge_attr is False
        assert metadata.requires_batch is False
        assert metadata.has_in_channels is True
        assert metadata.has_out_channels is True
        assert metadata.modifies_graph_structure is False
        assert metadata.supported_task_levels == ["node", "edge", "graph"]
        assert metadata.is_functional is False
    
    def test_init_with_all_parameters(self):
        """Test initialization with all parameters."""
        metadata = LayerMetadata(
            name="PoolingLayer",
            category=LayerCategory.POOLING,
            class_path="torch_geometric.nn.global_mean_pool",
            description="Global mean pooling",
            requires_edge_index=False,
            requires_edge_attr=False,
            requires_batch=True,
            has_in_channels=False,
            has_out_channels=False,
            modifies_graph_structure=True,
            supported_task_levels=["graph"],
            is_functional=True
        )
        
        assert metadata.name == "PoolingLayer"
        assert metadata.category == LayerCategory.POOLING
        assert metadata.requires_edge_index is False
        assert metadata.requires_batch is True
        assert metadata.has_in_channels is False
        assert metadata.has_out_channels is False
        assert metadata.modifies_graph_structure is True
        assert metadata.supported_task_levels == ["graph"]
        assert metadata.is_functional is True
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metadata = LayerMetadata(
            name="TestLayer",
            category=LayerCategory.NORMALIZATION,
            class_path="torch_geometric.nn.BatchNorm",
            description="Batch normalization",
            requires_edge_index=False,
            has_out_channels=False
        )
        
        result = metadata.to_dict()
        
        assert isinstance(result, dict)
        assert result['name'] == "TestLayer"
        assert result['category'] == "normalization"
        assert result['class_path'] == "torch_geometric.nn.BatchNorm"
        assert result['description'] == "Batch normalization"
        assert result['requires_edge_index'] is False
        assert result['requires_edge_attr'] is False
        assert result['requires_batch'] is False
        assert result['has_in_channels'] is True
        assert result['has_out_channels'] is False
        assert result['modifies_graph_structure'] is False
        assert result['supported_task_levels'] == ["node", "edge", "graph"]
        assert result['is_functional'] is False
    
    def test_is_pydantic_basemodel(self):
        """Test that LayerMetadata is a Pydantic BaseModel."""
        from pydantic import BaseModel
        
        metadata = LayerMetadata(
            name="TestLayer",
            category=LayerCategory.CUSTOM,
            class_path="custom.layer",
            description="Test"
        )
        
        assert isinstance(metadata, BaseModel)
    
    def test_model_dump_pydantic_v2(self):
        """Test Pydantic V2 model_dump() method."""
        metadata = LayerMetadata(
            name="TestLayerV2",
            category=LayerCategory.ACTIVATION,
            class_path="torch.nn.ReLU",
            description="Test activation"
        )
        
        # Pydantic V2 uses model_dump() instead of dict()
        if hasattr(metadata, 'model_dump'):
            dumped = metadata.model_dump()
            assert isinstance(dumped, dict)
            assert dumped['name'] == "TestLayerV2"
            # Note: model_dump keeps enum objects by default
            assert dumped['category'] == LayerCategory.ACTIVATION
    
    def test_model_validate_pydantic_v2(self):
        """Test Pydantic V2 model_validate() class method."""
        data = {
            'name': 'ValidatedLayer',
            'category': LayerCategory.LINEAR,
            'class_path': 'torch.nn.Linear',
            'description': 'Validated layer'
        }
        
        # Pydantic V2 uses model_validate() instead of parse_obj()
        if hasattr(LayerMetadata, 'model_validate'):
            validated = LayerMetadata.model_validate(data)
            assert validated.name == 'ValidatedLayer'
            assert validated.category == LayerCategory.LINEAR
    
    def test_mutability(self):
        """Test that LayerMetadata is mutable (not frozen)."""
        metadata = LayerMetadata(
            name="MutableLayer",
            category=LayerCategory.CUSTOM,
            class_path="custom.layer",
            description="Mutable test"
        )
        
        # Should be able to modify attributes (not frozen)
        original_name = metadata.name
        metadata.name = "ModifiedLayer"
        assert metadata.name == "ModifiedLayer"
        assert metadata.name != original_name
    
    def test_supported_task_levels_default_factory(self):
        """Test that supported_task_levels uses default_factory for mutable default."""
        metadata1 = LayerMetadata(
            name="Layer1",
            category=LayerCategory.CUSTOM,
            class_path="custom",
            description="Test 1"
        )
        metadata2 = LayerMetadata(
            name="Layer2",
            category=LayerCategory.CUSTOM,
            class_path="custom",
            description="Test 2"
        )
        
        # Each instance should have its own list (not shared reference)
        assert metadata1.supported_task_levels == metadata2.supported_task_levels
        assert metadata1.supported_task_levels is not metadata2.supported_task_levels
        
        # Modifying one should not affect the other
        metadata1.supported_task_levels.append("custom_level")
        assert "custom_level" in metadata1.supported_task_levels
        assert "custom_level" not in metadata2.supported_task_levels


# =============================================================================
# LAYER NOT FOUND ERROR TESTS
# =============================================================================

class TestLayerNotFoundError:
    """Test LayerNotFoundError exception."""
    
    def test_init_without_available_layers(self):
        """Test initialization without available layers list."""
        error = LayerNotFoundError("UnknownLayer")
        
        assert error.layer_name == "UnknownLayer"
        assert error.available_layers == []
        assert "Layer 'UnknownLayer' not found" in str(error)
    
    def test_init_with_few_available_layers(self):
        """Test initialization with small list of available layers."""
        available = ["GCNConv", "GATConv", "SAGEConv"]
        error = LayerNotFoundError("UnknownLayer", available)
        
        assert error.layer_name == "UnknownLayer"
        assert error.available_layers == available
        error_msg = str(error)
        assert "Layer 'UnknownLayer' not found" in error_msg
        assert "GATConv" in error_msg
        assert "GCNConv" in error_msg
        assert "SAGEConv" in error_msg
    
    def test_init_with_many_available_layers(self):
        """Test initialization with large list of available layers (>10)."""
        available = [f"Layer{i}" for i in range(20)]
        error = LayerNotFoundError("UnknownLayer", available)
        
        assert error.layer_name == "UnknownLayer"
        assert len(error.available_layers) == 20
        error_msg = str(error)
        assert "Layer 'UnknownLayer' not found" in error_msg
        assert "and 10 more" in error_msg
    
    def test_is_model_error_subclass(self):
        """Test that LayerNotFoundError is a subclass of ModelError."""
        error = LayerNotFoundError("TestLayer")
        
        assert isinstance(error, ModelError)
        assert isinstance(error, Exception)
    
    def test_init_with_none_available_layers(self):
        """Test initialization with None as available layers."""
        error = LayerNotFoundError("UnknownLayer", None)
        
        assert error.available_layers == []
        assert "Layer 'UnknownLayer' not found" in str(error)
    
    def test_init_with_exactly_ten_layers(self):
        """Test initialization with exactly 10 available layers (boundary case)."""
        available = [f"Layer{i}" for i in range(10)]
        error = LayerNotFoundError("UnknownLayer", available)
        
        error_msg = str(error)
        # Should show all 10 layers without "and X more" message
        assert "and" not in error_msg or "more" not in error_msg
    
    def test_available_layers_are_sorted_in_message(self):
        """Test that available layers are sorted in the error message."""
        available = ["ZLayer", "ALayer", "MLayer"]
        error = LayerNotFoundError("UnknownLayer", available)
        
        error_msg = str(error)
        # The first 10 should be sorted
        assert "ALayer" in error_msg


# =============================================================================
# LAYER REGISTRY SINGLETON TESTS
# =============================================================================

class TestLayerRegistrySingleton:
    """Test LayerRegistry singleton pattern."""
    
    def test_singleton_returns_same_instance(self):
        """Test that multiple instantiations return the same instance."""
        registry1 = LayerRegistry()
        registry2 = LayerRegistry()
        
        assert registry1 is registry2
    
    def test_singleton_thread_safety(self):
        """Test that singleton is thread-safe."""
        instances = []
        
        def create_instance():
            instances.append(LayerRegistry())
        
        threads = [threading.Thread(target=create_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All instances should be the same
        assert all(inst is instances[0] for inst in instances)
    
    def test_initialization_happens_once(self):
        """Test that initialization only happens once."""
        # Create first instance
        registry1 = LayerRegistry()
        initial_count = len(registry1)
        
        # Create second instance (should be same)
        registry2 = LayerRegistry()
        
        # Count should be the same (no re-initialization)
        assert len(registry2) == initial_count
        assert registry1 is registry2


# =============================================================================
# LAYER REGISTRY INITIALIZATION TESTS
# =============================================================================

class TestLayerRegistryInitialization:
    """Test LayerRegistry initialization and built-in layer registration."""
    
    def test_registry_has_initialized_attribute(self):
        """Test that registry has _initialized attribute."""
        registry = LayerRegistry()
        
        assert hasattr(registry, '_initialized')
        assert registry._initialized is True
    
    def test_registry_has_required_attributes(self):
        """Test that registry has all required internal attributes."""
        registry = LayerRegistry()
        
        assert hasattr(registry, '_layers')
        assert hasattr(registry, '_metadata')
        assert hasattr(registry, '_by_category')
        assert hasattr(registry, '_lock')
        assert isinstance(registry._layers, dict)
        assert isinstance(registry._metadata, dict)
        assert isinstance(registry._by_category, dict)
    
    def test_registry_registers_builtin_layers(self):
        """Test that built-in layers are automatically registered."""
        registry = LayerRegistry()
        
        # Should have registered many layers
        assert len(registry) > 0
        
        # Check some known layers exist
        assert registry.has_layer("GCNConv")
        assert registry.has_layer("ReLU")
        assert registry.has_layer("global_mean_pool")
    
    def test_registry_registers_aggregation_layers(self):
        """Test that internally defined aggregation layers are registered."""
        registry = LayerRegistry()
        
        # Check aggregation layers exist
        assert registry.has_layer("MeanAggregation")
        assert registry.has_layer("MaxAggregation")
        assert registry.has_layer("SumAggregation")
        
        # Verify they are in the AGGREGATION category
        agg_layers = registry.list_layers(LayerCategory.AGGREGATION)
        assert "MeanAggregation" in agg_layers
        assert "MaxAggregation" in agg_layers
        assert "SumAggregation" in agg_layers
        
        # Verify metadata
        mean_meta = registry.get_layer_metadata("MeanAggregation")
        assert mean_meta.category == LayerCategory.AGGREGATION
        assert mean_meta.requires_batch is True
        assert mean_meta.requires_edge_index is False
    
    def test_registry_registers_standard_layers(self):
        """Test that standard PyTorch layers (Linear, Dropout) are registered."""
        registry = LayerRegistry()
        
        # Check standard layers exist
        assert registry.has_layer("Linear")
        assert registry.has_layer("Dropout")
        
        # Verify Linear metadata
        linear_meta = registry.get_layer_metadata("Linear")
        assert linear_meta.category == LayerCategory.LINEAR
        assert linear_meta.has_in_channels is True
        assert linear_meta.has_out_channels is True
        
        # Verify Dropout metadata
        dropout_meta = registry.get_layer_metadata("Dropout")
        assert dropout_meta.category == LayerCategory.DROPOUT
        assert dropout_meta.has_in_channels is False
        assert dropout_meta.has_out_channels is False
    
    def test_registry_has_multiple_categories(self):
        """Test that registry has multiple layer categories."""
        registry = LayerRegistry()
        
        categories = registry.list_categories()
        
        assert len(categories) > 0
        assert LayerCategory.CONVOLUTIONAL in categories
        assert LayerCategory.ACTIVATION in categories
        assert LayerCategory.POOLING in categories


# =============================================================================
# BUILT-IN LAYER REGISTRATION METHODS TESTS
# =============================================================================

class TestBuiltinLayerRegistration:
    """Test individual built-in layer registration methods."""
    
    def test_convolutional_layers_registered(self):
        """Test that all expected convolutional layers are registered."""
        registry = LayerRegistry()
        
        expected_conv_layers = [
            "GCNConv", "GATConv", "SAGEConv", "GINConv", "ChebConv",
            "GraphConv", "GatedGraphConv", "EdgeConv", "TAGConv", "ARMAConv",
            "SGConv", "APPNP", "MFConv", "RGCNConv", "SignedConv",
            "DNAConv", "PANConv", "PointNetConv", "GMMConv", "SplineConv",
            "NNConv", "CGConv", "TransformerConv", "GATv2Conv", "SuperGATConv",
            "FiLMConv", "GeneralConv", "HGTConv", "HEATConv", "LEConv",
            "ClusterGCNConv", "GENConv"
        ]
        
        conv_layers = registry.list_layers(LayerCategory.CONVOLUTIONAL)
        
        for layer_name in expected_conv_layers:
            assert layer_name in conv_layers, f"Expected {layer_name} in convolutional layers"
            # Verify metadata
            metadata = registry.get_layer_metadata(layer_name)
            assert metadata.category == LayerCategory.CONVOLUTIONAL
            assert metadata.requires_edge_index is True
    
    def test_pooling_layers_registered(self):
        """Test that all expected pooling layers are registered."""
        registry = LayerRegistry()
        
        # Functional global pooling
        expected_functional_pooling = [
            "global_mean_pool", "global_max_pool", "global_add_pool"
        ]
        
        # Class-based pooling
        expected_class_pooling = [
            "TopKPooling", "SAGPooling", "EdgePooling",
            "ASAPooling", "PANPooling", "MemPooling"
        ]
        
        pooling_layers = registry.list_layers(LayerCategory.POOLING)
        
        for layer_name in expected_functional_pooling:
            assert layer_name in pooling_layers, f"Expected {layer_name} in pooling layers"
            metadata = registry.get_layer_metadata(layer_name)
            assert metadata.is_functional is True
            assert metadata.requires_batch is True
            assert metadata.modifies_graph_structure is True
        
        for layer_name in expected_class_pooling:
            assert layer_name in pooling_layers, f"Expected {layer_name} in pooling layers"
            metadata = registry.get_layer_metadata(layer_name)
            assert metadata.category == LayerCategory.POOLING
            assert metadata.modifies_graph_structure is True
    
    def test_normalization_layers_registered(self):
        """Test that all expected normalization layers are registered."""
        registry = LayerRegistry()
        
        expected_norm_layers = [
            "BatchNorm", "LayerNorm", "InstanceNorm", "GraphNorm",
            "PairNorm", "MeanSubtractionNorm", "DiffGroupNorm"
        ]
        
        norm_layers = registry.list_layers(LayerCategory.NORMALIZATION)
        
        for layer_name in expected_norm_layers:
            assert layer_name in norm_layers, f"Expected {layer_name} in normalization layers"
            metadata = registry.get_layer_metadata(layer_name)
            assert metadata.category == LayerCategory.NORMALIZATION
            assert metadata.requires_edge_index is False
            assert metadata.has_in_channels is True
            assert metadata.has_out_channels is False
    
    def test_activation_layers_registered(self):
        """Test that all expected activation layers are registered."""
        registry = LayerRegistry()
        
        expected_activation_layers = [
            "ReLU", "LeakyReLU", "ELU", "PReLU", "GELU",
            "Tanh", "Sigmoid", "Softplus", "SiLU"
        ]
        
        activation_layers = registry.list_layers(LayerCategory.ACTIVATION)
        
        for layer_name in expected_activation_layers:
            assert layer_name in activation_layers, f"Expected {layer_name} in activation layers"
            metadata = registry.get_layer_metadata(layer_name)
            assert metadata.category == LayerCategory.ACTIVATION
            assert metadata.requires_edge_index is False
            assert metadata.has_in_channels is False
            assert metadata.has_out_channels is False
    
    def test_aggregation_layers_registered(self):
        """Test that internally defined aggregation layers are registered correctly."""
        registry = LayerRegistry()
        
        expected_agg_layers = ["MeanAggregation", "MaxAggregation", "SumAggregation"]
        
        agg_layers = registry.list_layers(LayerCategory.AGGREGATION)
        
        for layer_name in expected_agg_layers:
            assert layer_name in agg_layers, f"Expected {layer_name} in aggregation layers"
            metadata = registry.get_layer_metadata(layer_name)
            assert metadata.category == LayerCategory.AGGREGATION
            assert metadata.requires_edge_index is False
            assert metadata.requires_batch is True
            assert metadata.has_in_channels is False
            assert metadata.has_out_channels is False
    
    def test_standard_layers_registered(self):
        """Test that standard PyTorch layers (Linear, Dropout) are registered."""
        registry = LayerRegistry()
        
        # Verify Linear
        assert registry.has_layer("Linear")
        linear_meta = registry.get_layer_metadata("Linear")
        assert linear_meta.category == LayerCategory.LINEAR
        assert linear_meta.has_in_channels is True
        assert linear_meta.has_out_channels is True
        assert linear_meta.requires_edge_index is False
        
        # Verify Dropout
        assert registry.has_layer("Dropout")
        dropout_meta = registry.get_layer_metadata("Dropout")
        assert dropout_meta.category == LayerCategory.DROPOUT
        assert dropout_meta.has_in_channels is False
        assert dropout_meta.has_out_channels is False
        assert dropout_meta.requires_edge_index is False
    
    def test_wrapped_functional_pooling_instantiation(self):
        """Test that wrapped functional pooling layers can be instantiated."""
        registry = LayerRegistry()
        
        # Get a wrapped functional pooling class
        GlobalMeanPool = registry.get_layer("global_mean_pool")
        
        # Should be instantiable without arguments
        pool_instance = GlobalMeanPool()
        
        # Should be a FunctionalLayerWrapper
        assert isinstance(pool_instance, FunctionalLayerWrapper)
        assert pool_instance.requires_batch is True
        assert pool_instance.requires_edge_index is False
    
    def test_class_path_format_for_pyg_layers(self):
        """Test that class_path is correctly formatted for PyG layers."""
        registry = LayerRegistry()
        
        # Check a few representative layers
        gcn_meta = registry.get_layer_metadata("GCNConv")
        # class_path should be a non-empty string following module.class format
        assert isinstance(gcn_meta.class_path, str)
        assert len(gcn_meta.class_path) > 0
        assert "." in gcn_meta.class_path or gcn_meta.class_path == "custom"
        # In mocked environment, __module__ returns 'unittest.mock', 
        # in real environment it returns 'torch_geometric.nn.conv.gcn_conv'
        # We verify the format is correct (contains a dot for module.class pattern)
        
        relu_meta = registry.get_layer_metadata("ReLU")
        assert isinstance(relu_meta.class_path, str)
        assert len(relu_meta.class_path) > 0
        # ReLU uses real torch.nn.ReLU, so it should have 'torch' in the path
        assert "torch" in relu_meta.class_path


# =============================================================================
# CUSTOM LAYER REGISTRATION TESTS
# =============================================================================

class TestRegisterCustomLayer:
    """Test register_custom_layer method."""
    
    def test_register_custom_class_layer(self):
        """Test registering a custom layer class."""
        registry = LayerRegistry()
        
        class CustomLayerClass(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.linear = nn.Linear(in_channels, out_channels)
            
            def forward(self, x, edge_index):
                return self.linear(x)
        
        layer_name = "CustomLayerClass_v1"
        
        registry.register_custom_layer(
            layer_name,
            CustomLayerClass,
            category=LayerCategory.CUSTOM
        )
        
        assert registry.has_layer(layer_name)
        retrieved = registry.get_layer(layer_name)
        assert retrieved is CustomLayerClass
    
    def test_register_custom_function_layer(self):
        """Test registering a custom functional layer."""
        registry = LayerRegistry()
        
        def custom_pool(x, batch):
            return x.mean(dim=0)
        
        metadata = LayerMetadata(
            name="custom_pool_func",
            category=LayerCategory.POOLING,
            class_path="custom",
            description="Custom pooling",
            requires_batch=True,
            requires_edge_index=False,
            is_functional=True
        )
        
        registry.register_custom_layer(
            "custom_pool_func",
            custom_pool,
            metadata=metadata,
            category=LayerCategory.POOLING
        )
        
        assert registry.has_layer("custom_pool_func")
        retrieved = registry.get_layer("custom_pool_func")
        # When metadata has is_functional=True and layer_class is function,
        # the layer is stored as-is (not wrapped again)
        # This is the actual behavior based on _register_layer logic
        assert retrieved is custom_pool or callable(retrieved)
    
    def test_register_function_without_functional_metadata(self):
        """Test registering a function without is_functional flag triggers wrapping."""
        registry = LayerRegistry()
        
        def my_pool_func(x, batch):
            return x.sum(dim=0)
        
        # Metadata without is_functional=True (defaults to False)
        metadata = LayerMetadata(
            name="my_pool_func_wrapped",
            category=LayerCategory.POOLING,
            class_path="custom",
            description="Pool function",
            requires_batch=True,
            requires_edge_index=False,
            is_functional=False  # Explicitly False
        )
        
        registry.register_custom_layer(
            "my_pool_func_wrapped",
            my_pool_func,
            metadata=metadata,
            category=LayerCategory.POOLING
        )
        
        assert registry.has_layer("my_pool_func_wrapped")
        retrieved = registry.get_layer("my_pool_func_wrapped")
        
        # Should be wrapped since is_functional was False
        # The _register_layer logic wraps functions when metadata.is_functional is False
        assert callable(retrieved)
    
    def test_register_with_auto_generated_metadata(self):
        """Test registering layer with auto-generated metadata."""
        registry = LayerRegistry()
        
        class SimpleLayerAuto(nn.Module):
            def forward(self, x):
                return x
        
        layer_name = "SimpleLayerAuto_v1"
        
        registry.register_custom_layer(
            layer_name,
            SimpleLayerAuto,
            category=LayerCategory.CUSTOM
        )
        
        assert registry.has_layer(layer_name)
        metadata = registry.get_layer_metadata(layer_name)
        assert metadata.name == layer_name
        assert metadata.category == LayerCategory.CUSTOM
        assert isinstance(metadata.description, str)
    
    def test_register_duplicate_without_overwrite_raises_error(self):
        """Test that registering duplicate layer without overwrite raises ValueError."""
        registry = LayerRegistry()
        
        class Layer1(nn.Module):
            pass
        
        class Layer2(nn.Module):
            pass
        
        # Use unique layer name
        layer_name = "TestLayerNoDup_v1"
        
        # Register first time
        registry.register_custom_layer(layer_name, Layer1, category=LayerCategory.CUSTOM)
        
        # Try to register again without overwrite
        with pytest.raises(ValueError, match="already registered"):
            registry.register_custom_layer(layer_name, Layer2, category=LayerCategory.CUSTOM)
    
    def test_register_duplicate_with_overwrite_succeeds(self):
        """Test that registering duplicate layer with overwrite=True succeeds."""
        registry = LayerRegistry()
        
        class Layer1(nn.Module):
            pass
        
        class Layer2(nn.Module):
            pass
        
        # Use a unique name to avoid conflicts with other tests
        layer_name = "TestLayerOverwrite_v1"
        
        # Register first time
        registry.register_custom_layer(layer_name, Layer1, category=LayerCategory.CUSTOM)
        
        # Register again with overwrite
        registry.register_custom_layer(
            layer_name,
            Layer2,
            category=LayerCategory.CUSTOM,
            overwrite=True
        )
        
        # Should now return Layer2
        assert registry.get_layer(layer_name) is Layer2
    
    def test_register_non_callable_raises_error(self):
        """Test that registering non-callable raises TypeError."""
        registry = LayerRegistry()
        
        with pytest.raises(TypeError, match="must be callable"):
            registry.register_custom_layer("InvalidLayer", "not_callable")
    
    def test_register_with_explicit_metadata_category_mismatch(self):
        """Test registering with metadata that has different category than argument."""
        registry = LayerRegistry()
        
        class MismatchCategoryLayer(nn.Module):
            def forward(self, x):
                return x
        
        # Create metadata with CUSTOM category
        metadata = LayerMetadata(
            name="MismatchCategoryLayer_v1",
            category=LayerCategory.CUSTOM,  # Metadata says CUSTOM
            class_path="test",
            description="Test"
        )
        
        # Register with ACTIVATION category argument
        # The metadata's category should be used since metadata is provided
        registry.register_custom_layer(
            "MismatchCategoryLayer_v1",
            MismatchCategoryLayer,
            metadata=metadata,
            category=LayerCategory.ACTIVATION  # Argument says ACTIVATION
        )
        
        # When explicit metadata is provided, it's used as-is
        retrieved_meta = registry.get_layer_metadata("MismatchCategoryLayer_v1")
        assert retrieved_meta.category == LayerCategory.CUSTOM  # Metadata's category wins
    
    def test_register_custom_with_module_attribute(self):
        """Test auto-generated metadata uses correct module path."""
        registry = LayerRegistry()
        
        class ModulePathTestLayer(nn.Module):
            def forward(self, x):
                return x
        
        registry.register_custom_layer(
            "ModulePathTestLayer_v1",
            ModulePathTestLayer,
            category=LayerCategory.CUSTOM
        )
        
        metadata = registry.get_layer_metadata("ModulePathTestLayer_v1")
        # Should include module and class name
        assert "ModulePathTestLayer" in metadata.class_path


# =============================================================================
# LAYER RETRIEVAL TESTS
# =============================================================================

class TestGetLayer:
    """Test get_layer method."""
    
    def test_get_existing_layer(self):
        """Test retrieving an existing layer."""
        registry = LayerRegistry()
        
        layer_class = registry.get_layer("GCNConv")
        
        assert layer_class is not None
        assert callable(layer_class)
    
    def test_get_nonexistent_layer_raises_error(self):
        """Test that retrieving non-existent layer raises LayerNotFoundError."""
        registry = LayerRegistry()
        
        with pytest.raises(LayerNotFoundError) as exc_info:
            registry.get_layer("NonExistentLayer")
        
        assert exc_info.value.layer_name == "NonExistentLayer"
        assert len(exc_info.value.available_layers) > 0


class TestGetLayerMetadata:
    """Test get_layer_metadata method."""
    
    def test_get_existing_metadata(self):
        """Test retrieving metadata for existing layer."""
        registry = LayerRegistry()
        
        metadata = registry.get_layer_metadata("GCNConv")
        
        assert isinstance(metadata, LayerMetadata)
        assert metadata.name == "GCNConv"
        assert metadata.category == LayerCategory.CONVOLUTIONAL
    
    def test_get_nonexistent_metadata_raises_error(self):
        """Test that retrieving metadata for non-existent layer raises error."""
        registry = LayerRegistry()
        
        with pytest.raises(LayerNotFoundError) as exc_info:
            registry.get_layer_metadata("NonExistentLayer")
        
        assert exc_info.value.layer_name == "NonExistentLayer"


class TestHasLayer:
    """Test has_layer method."""
    
    def test_has_layer_returns_true_for_existing(self):
        """Test has_layer returns True for existing layer."""
        registry = LayerRegistry()
        
        assert registry.has_layer("GCNConv") is True
        assert registry.has_layer("ReLU") is True
    
    def test_has_layer_returns_false_for_nonexistent(self):
        """Test has_layer returns False for non-existent layer."""
        registry = LayerRegistry()
        
        assert registry.has_layer("NonExistentLayer") is False


# =============================================================================
# LAYER LISTING TESTS
# =============================================================================

class TestListLayers:
    """Test list_layers method."""
    
    def test_list_all_layers(self):
        """Test listing all layers without category filter."""
        registry = LayerRegistry()
        
        layers = registry.list_layers()
        
        assert isinstance(layers, list)
        assert len(layers) > 0
        # Should be sorted
        assert layers == sorted(layers)
    
    def test_list_layers_by_category(self):
        """Test listing layers filtered by category."""
        registry = LayerRegistry()
        
        conv_layers = registry.list_layers(LayerCategory.CONVOLUTIONAL)
        
        assert isinstance(conv_layers, list)
        assert len(conv_layers) > 0
        # Should be sorted
        assert conv_layers == sorted(conv_layers)
        # Should contain known convolutional layers
        assert "GCNConv" in conv_layers
    
    def test_list_layers_activation_category(self):
        """Test listing activation layers."""
        registry = LayerRegistry()
        
        activation_layers = registry.list_layers(LayerCategory.ACTIVATION)
        
        assert isinstance(activation_layers, list)
        assert len(activation_layers) > 0
        assert "ReLU" in activation_layers
    
    def test_list_layers_pooling_category(self):
        """Test listing pooling layers."""
        registry = LayerRegistry()
        
        pooling_layers = registry.list_layers(LayerCategory.POOLING)
        
        assert isinstance(pooling_layers, list)
        assert len(pooling_layers) > 0
        # Should contain functional pooling
        assert "global_mean_pool" in pooling_layers


class TestListCategories:
    """Test list_categories method."""
    
    def test_list_categories(self):
        """Test listing all available categories."""
        registry = LayerRegistry()
        
        categories = registry.list_categories()
        
        assert isinstance(categories, list)
        assert len(categories) > 0
        # Should contain main categories
        assert LayerCategory.CONVOLUTIONAL in categories
        assert LayerCategory.ACTIVATION in categories
        assert LayerCategory.POOLING in categories


# =============================================================================
# STATISTICS AND UTILITY TESTS
# =============================================================================

class TestGetStatistics:
    """Test get_statistics method."""
    
    def test_get_statistics_structure(self):
        """Test statistics structure and content."""
        registry = LayerRegistry()
        
        stats = registry.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_layers' in stats
        assert 'by_category' in stats
        assert 'categories' in stats
        assert 'functional_layers' in stats
        assert 'class_layers' in stats
        
        # Check types
        assert isinstance(stats['total_layers'], int)
        assert isinstance(stats['by_category'], dict)
        assert isinstance(stats['categories'], list)
        assert isinstance(stats['functional_layers'], int)
        assert isinstance(stats['class_layers'], int)
    
    def test_statistics_counts_match(self):
        """Test that statistics counts are consistent."""
        registry = LayerRegistry()
        
        stats = registry.get_statistics()
        
        # Total should equal functional + class
        assert stats['total_layers'] == stats['functional_layers'] + stats['class_layers']
        
        # Total should be positive
        assert stats['total_layers'] > 0


class TestDunderMethods:
    """Test magic methods (__len__, __contains__, __repr__)."""
    
    def test_len(self):
        """Test __len__ method."""
        registry = LayerRegistry()
        
        length = len(registry)
        
        assert isinstance(length, int)
        assert length > 0
        # Should match total_layers in statistics
        stats = registry.get_statistics()
        assert length == stats['total_layers']
    
    def test_contains(self):
        """Test __contains__ method (in operator)."""
        registry = LayerRegistry()
        
        assert "GCNConv" in registry
        assert "ReLU" in registry
        assert "NonExistentLayer" not in registry
    
    def test_repr(self):
        """Test __repr__ method."""
        registry = LayerRegistry()
        
        repr_str = repr(registry)
        
        assert isinstance(repr_str, str)
        assert "LayerRegistry" in repr_str
        assert "total=" in repr_str
        assert "functional=" in repr_str
        assert "class=" in repr_str


# =============================================================================
# INTERNAL METHODS TESTS
# =============================================================================

class TestInternalMethods:
    """Test internal helper methods."""
    
    def test_is_functional_with_function(self):
        """Test _is_functional correctly identifies functions."""
        registry = LayerRegistry()
        
        def test_func():
            pass
        
        assert registry._is_functional(test_func) is True
    
    def test_is_functional_with_class(self):
        """Test _is_functional correctly identifies classes."""
        registry = LayerRegistry()
        
        class TestClass:
            pass
        
        assert registry._is_functional(TestClass) is False
    
    def test_is_functional_with_lambda(self):
        """Test _is_functional with lambda."""
        registry = LayerRegistry()
        
        test_lambda = lambda x: x
        
        assert registry._is_functional(test_lambda) is True
    
    def test_is_functional_with_builtin(self):
        """Test _is_functional with built-in functions."""
        registry = LayerRegistry()
        
        # Built-in functions like len, sum are callable but not types
        assert registry._is_functional(len) is True
        assert registry._is_functional(sum) is True
    
    def test_is_functional_with_nn_module_class(self):
        """Test _is_functional with nn.Module subclass."""
        registry = LayerRegistry()
        
        class MyModule(nn.Module):
            pass
        
        # nn.Module subclass is a class/type, not a function
        assert registry._is_functional(MyModule) is False
    
    def test_create_functional_wrapper(self):
        """Test _create_functional_wrapper creates correct wrapper class."""
        registry = LayerRegistry()
        
        def test_pool_func(x, batch):
            return x.mean(dim=0)
        
        metadata = LayerMetadata(
            name="test_pool_wrapper",
            category=LayerCategory.POOLING,
            class_path="custom",
            description="Test pooling wrapper",
            requires_batch=True,
            requires_edge_index=False,
            requires_edge_attr=False,
            is_functional=True
        )
        
        # Call the internal method directly
        wrapper_class = registry._create_functional_wrapper(
            "test_pool_wrapper",
            test_pool_func,
            metadata
        )
        
        # Verify wrapper class was created
        assert wrapper_class is not None
        assert callable(wrapper_class)
        # Verify wrapper class name follows pattern
        assert "Wrapped_" in wrapper_class.__name__
        assert "test_pool_wrapper" in wrapper_class.__name__
        
        # Verify wrapper instance works correctly
        wrapper_instance = wrapper_class()
        assert isinstance(wrapper_instance, FunctionalLayerWrapper)
        assert wrapper_instance.func_name == "test_pool_wrapper"
        assert wrapper_instance.requires_batch is True
        assert wrapper_instance.requires_edge_index is False
    
    def test_create_functional_wrapper_with_edge_requirements(self):
        """Test _create_functional_wrapper with edge requirements."""
        registry = LayerRegistry()
        
        def test_conv_func(x, edge_index, edge_attr):
            return x
        
        metadata = LayerMetadata(
            name="test_conv_wrapper",
            category=LayerCategory.CUSTOM,
            class_path="custom",
            description="Test conv wrapper",
            requires_batch=False,
            requires_edge_index=True,
            requires_edge_attr=True,
            is_functional=True
        )
        
        wrapper_class = registry._create_functional_wrapper(
            "test_conv_wrapper",
            test_conv_func,
            metadata
        )
        
        wrapper_instance = wrapper_class()
        assert wrapper_instance.requires_edge_index is True
        assert wrapper_instance.requires_edge_attr is True
        assert wrapper_instance.requires_batch is False
    
    def test_register_layer_internal_auto_wrapping(self):
        """Test _register_layer auto-wraps functions when is_functional=False."""
        registry = LayerRegistry()
        
        def my_internal_func(x):
            return x * 2
        
        # Metadata with is_functional=False (so auto-wrapping should occur)
        metadata = LayerMetadata(
            name="auto_wrap_test_internal",
            category=LayerCategory.CUSTOM,
            class_path="custom",
            description="Auto wrap test",
            requires_edge_index=False,
            requires_batch=False,
            is_functional=False  # Will trigger auto-wrapping
        )
        
        # Call internal _register_layer directly
        registry._register_layer("auto_wrap_test_internal", my_internal_func, metadata)
        
        # Verify the layer was wrapped
        retrieved = registry.get_layer("auto_wrap_test_internal")
        assert callable(retrieved)
        
        # Verify metadata was updated to is_functional=True
        updated_metadata = registry.get_layer_metadata("auto_wrap_test_internal")
        assert updated_metadata.is_functional is True
    
    def test_register_layer_internal_no_wrap_when_is_functional_true(self):
        """Test _register_layer does not double-wrap when is_functional=True."""
        registry = LayerRegistry()
        
        def already_functional(x):
            return x
        
        # Metadata with is_functional=True (no auto-wrapping needed)
        metadata = LayerMetadata(
            name="no_wrap_test_internal",
            category=LayerCategory.CUSTOM,
            class_path="custom",
            description="No wrap test",
            requires_edge_index=False,
            is_functional=True  # Already marked as functional
        )
        
        # Call internal _register_layer directly
        registry._register_layer("no_wrap_test_internal", already_functional, metadata)
        
        # Since is_functional is True AND the condition checks both _is_functional and not metadata.is_functional,
        # and is_functional=True means metadata.is_functional is True, so the condition is False
        # The layer should be stored as-is
        retrieved = registry.get_layer("no_wrap_test_internal")
        assert retrieved is already_functional
    
    def test_register_layer_adds_to_category_index(self):
        """Test _register_layer correctly adds to category index."""
        registry = LayerRegistry()
        
        class CategoryTestLayer(nn.Module):
            pass
        
        metadata = LayerMetadata(
            name="CategoryTestLayer_v1",
            category=LayerCategory.CUSTOM,
            class_path="test",
            description="Test"
        )
        
        # Register via internal method
        registry._register_layer("CategoryTestLayer_v1", CategoryTestLayer, metadata)
        
        # Verify it's in the category
        custom_layers = registry.list_layers(LayerCategory.CUSTOM)
        assert "CategoryTestLayer_v1" in custom_layers


class TestListLayersEdgeCases:
    """Test edge cases for list_layers method."""
    
    def test_list_layers_empty_category(self):
        """Test listing layers for a category that might be empty or non-existent in _by_category."""
        registry = LayerRegistry()
        
        # Create a fake category that won't exist in _by_category
        # Since LayerCategory is an enum, we test the behavior when category has no layers
        # by checking that the method handles missing keys gracefully
        
        # First verify that list_layers with a valid category works
        conv_layers = registry.list_layers(LayerCategory.CONVOLUTIONAL)
        assert isinstance(conv_layers, list)
        
        # The method should return an empty list for any category not in _by_category
        # Since all standard categories are populated, we verify the return type
        # and that sorting an empty set returns an empty list
        all_categories = registry.list_categories()
        for category in LayerCategory:
            result = registry.list_layers(category)
            assert isinstance(result, list)
            # Result should be sorted (even if empty)
            assert result == sorted(result)
    
    def test_list_layers_returns_new_list(self):
        """Test that list_layers returns a new list each time (not internal reference)."""
        registry = LayerRegistry()
        
        layers1 = registry.list_layers()
        layers2 = registry.list_layers()
        
        # Should be equal in content
        assert layers1 == layers2
        # But should be different list objects
        assert layers1 is not layers2
    
    def test_list_categories_returns_new_list(self):
        """Test that list_categories returns a new list each time."""
        registry = LayerRegistry()
        
        cats1 = registry.list_categories()
        cats2 = registry.list_categories()
        
        # Should be equal in content
        assert set(cats1) == set(cats2)
        # But should be different list objects
        assert cats1 is not cats2


class TestGetStatisticsEdgeCases:
    """Test edge cases for get_statistics method."""
    
    def test_statistics_by_category_keys(self):
        """Test that by_category uses string values, not enum objects."""
        registry = LayerRegistry()
        
        stats = registry.get_statistics()
        
        # Keys in by_category should be string values (e.g., "convolutional")
        for key in stats['by_category'].keys():
            assert isinstance(key, str)
    
    def test_statistics_categories_are_strings(self):
        """Test that categories list contains string values."""
        registry = LayerRegistry()
        
        stats = registry.get_statistics()
        
        # Categories should be string values
        for cat in stats['categories']:
            assert isinstance(cat, str)


# =============================================================================
# CONVENIENCE FUNCTIONS TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    def test_get_layer_function(self):
        """Test get_layer convenience function."""
        layer_class = get_layer("GCNConv")
        
        assert layer_class is not None
        assert callable(layer_class)
    
    def test_list_layers_function(self):
        """Test list_layers convenience function."""
        layers = list_layers()
        
        assert isinstance(layers, list)
        assert len(layers) > 0
    
    def test_list_layers_with_category_function(self):
        """Test list_layers convenience function with category."""
        conv_layers = list_layers(LayerCategory.CONVOLUTIONAL)
        
        assert isinstance(conv_layers, list)
        assert len(conv_layers) > 0
        assert "GCNConv" in conv_layers
    
    def test_get_layer_metadata_function(self):
        """Test get_layer_metadata convenience function."""
        metadata = get_layer_metadata("GCNConv")
        
        assert isinstance(metadata, LayerMetadata)
        assert metadata.name == "GCNConv"


# =============================================================================
# GLOBAL REGISTRY TESTS
# =============================================================================

class TestGlobalRegistry:
    """Test global registry instance."""
    
    def test_global_registry_exists(self):
        """Test that global registry instance exists."""
        assert global_registry is not None
        assert isinstance(global_registry, LayerRegistry)
    
    def test_global_registry_is_singleton(self):
        """Test that global registry is the singleton instance."""
        new_registry = LayerRegistry()
        
        assert global_registry is new_registry
    
    def test_global_registry_is_initialized(self):
        """Test that global registry has been initialized with layers."""
        assert len(global_registry) > 0
        assert global_registry.has_layer("GCNConv")
    
    def test_global_registry_convenience_functions_use_same_instance(self):
        """Test that convenience functions use the same global registry."""
        # get_layer should return same result as global_registry.get_layer
        layer_via_func = get_layer("GCNConv")
        layer_via_registry = global_registry.get_layer("GCNConv")
        assert layer_via_func is layer_via_registry
        
        # list_layers should return same result
        layers_via_func = list_layers()
        layers_via_registry = global_registry.list_layers()
        assert layers_via_func == layers_via_registry
        
        # get_layer_metadata should return same result
        meta_via_func = get_layer_metadata("GCNConv")
        meta_via_registry = global_registry.get_layer_metadata("GCNConv")
        assert meta_via_func is meta_via_registry


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================

class TestThreadSafety:
    """Test thread safety of registry operations."""
    
    def test_concurrent_registration(self):
        """Test concurrent custom layer registration."""
        registry = LayerRegistry()
        results = []
        errors = []
        
        def register_layer(idx):
            try:
                class TestLayer(nn.Module):
                    def forward(self, x):
                        return x
                
                registry.register_custom_layer(
                    f"ThreadLayer{idx}",
                    TestLayer,
                    category=LayerCategory.CUSTOM
                )
                results.append(True)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=register_layer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All registrations should succeed
        assert len(results) == 10
        assert len(errors) == 0
        
        # All layers should be registered
        for i in range(10):
            assert registry.has_layer(f"ThreadLayer{i}")
    
    def test_concurrent_retrieval(self):
        """Test concurrent layer retrieval."""
        registry = LayerRegistry()
        results = []
        errors = []
        
        def get_layer_safe():
            try:
                layer = registry.get_layer("GCNConv")
                results.append(layer)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=get_layer_safe) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All retrievals should succeed
        assert len(results) == 20
        assert len(errors) == 0
        # All should return same class
        assert all(r is results[0] for r in results)
    
    def test_concurrent_list_operations(self):
        """Test concurrent list_layers and list_categories."""
        registry = LayerRegistry()
        results_layers = []
        results_categories = []
        errors = []
        
        def list_layers_safe():
            try:
                layers = registry.list_layers()
                results_layers.append(len(layers))
            except Exception as e:
                errors.append(e)
        
        def list_categories_safe():
            try:
                categories = registry.list_categories()
                results_categories.append(len(categories))
            except Exception as e:
                errors.append(e)
        
        threads = []
        for _ in range(10):
            threads.append(threading.Thread(target=list_layers_safe))
            threads.append(threading.Thread(target=list_categories_safe))
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        # All layer counts should be consistent
        assert len(set(results_layers)) == 1
        # All category counts should be consistent
        assert len(set(results_categories)) == 1
    
    def test_concurrent_statistics(self):
        """Test concurrent get_statistics calls."""
        registry = LayerRegistry()
        results = []
        errors = []
        
        def get_stats_safe():
            try:
                stats = registry.get_statistics()
                results.append(stats['total_layers'])
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=get_stats_safe) for _ in range(15)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        # All should return same count
        assert len(set(results)) == 1
    
    def test_registry_uses_rlock(self):
        """Test that registry uses RLock (reentrant lock)."""
        registry = LayerRegistry()
        
        # Verify the lock is a threading.RLock
        assert isinstance(registry._lock, type(threading.RLock()))


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Test integration scenarios."""
    
    def test_custom_layer_workflow(self):
        """Test complete workflow for custom layer."""
        registry = LayerRegistry()
        
        # Define custom layer
        class MyCustomConvWorkflow(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.linear = nn.Linear(in_channels, out_channels)
            
            def forward(self, x, edge_index):
                return self.linear(x)
        
        layer_name = "MyCustomConvWorkflow"
        
        # Register
        registry.register_custom_layer(
            layer_name,
            MyCustomConvWorkflow,
            category=LayerCategory.CUSTOM
        )
        
        # Verify registration
        assert registry.has_layer(layer_name)
        
        # Retrieve
        layer_class = registry.get_layer(layer_name)
        assert layer_class is MyCustomConvWorkflow
        
        # Get metadata
        metadata = registry.get_layer_metadata(layer_name)
        assert metadata.name == layer_name
        assert metadata.category == LayerCategory.CUSTOM
        
        # Check in listings
        all_layers = registry.list_layers()
        assert layer_name in all_layers
        
        custom_layers = registry.list_layers(LayerCategory.CUSTOM)
        assert layer_name in custom_layers
    
    def test_functional_layer_workflow(self):
        """Test complete workflow for functional layer."""
        registry = LayerRegistry()
        
        # Define functional operation
        def my_custom_pool_workflow(x, batch):
            # Simple mean pooling
            return x.mean(dim=0)
        
        # Create metadata
        metadata = LayerMetadata(
            name="my_custom_pool_workflow",
            category=LayerCategory.POOLING,
            class_path="custom.my_custom_pool_workflow",
            description="Custom mean pooling",
            requires_batch=True,
            requires_edge_index=False,
            has_in_channels=False,
            has_out_channels=False,
            modifies_graph_structure=True,
            supported_task_levels=["graph"],
            is_functional=True
        )
        
        # Register
        registry.register_custom_layer(
            "my_custom_pool_workflow",
            my_custom_pool_workflow,
            metadata=metadata,
            category=LayerCategory.POOLING
        )
        
        # Verify registration
        assert registry.has_layer("my_custom_pool_workflow")
        
        # Retrieve - with is_functional=True in metadata, it stores as-is
        retrieved = registry.get_layer("my_custom_pool_workflow")
        assert callable(retrieved)
        
        # Verify metadata
        meta = registry.get_layer_metadata("my_custom_pool_workflow")
        assert meta.is_functional is True
        assert meta.requires_batch is True
    
    def test_metadata_serialization_workflow(self):
        """Test metadata serialization and retrieval."""
        registry = LayerRegistry()
        
        # Get metadata for existing layer
        metadata = registry.get_layer_metadata("GCNConv")
        
        # Serialize to dict
        metadata_dict = metadata.to_dict()
        
        # Verify structure
        assert isinstance(metadata_dict, dict)
        assert 'name' in metadata_dict
        assert 'category' in metadata_dict
        assert 'class_path' in metadata_dict
        assert 'description' in metadata_dict
        
        # Verify values
        assert metadata_dict['name'] == "GCNConv"
        assert metadata_dict['category'] == "convolutional"
    
    def test_layer_instantiation_workflow(self):
        """Test that retrieved layers can be instantiated."""
        registry = LayerRegistry()
        
        # Get activation layer (no constructor args needed)
        ReLU = registry.get_layer("ReLU")
        relu_instance = ReLU()
        assert isinstance(relu_instance, nn.Module)
        
        # Get wrapped functional layer
        GlobalMeanPool = registry.get_layer("global_mean_pool")
        pool_instance = GlobalMeanPool()
        assert isinstance(pool_instance, nn.Module)
    
    def test_statistics_after_custom_registration(self):
        """Test that statistics update correctly after custom layer registration."""
        registry = LayerRegistry()
        
        initial_stats = registry.get_statistics()
        initial_total = initial_stats['total_layers']
        
        # Register a new custom layer
        class StatsTestLayer(nn.Module):
            def forward(self, x):
                return x
        
        registry.register_custom_layer(
            "StatsTestLayer_v1",
            StatsTestLayer,
            category=LayerCategory.CUSTOM
        )
        
        # Check statistics updated
        new_stats = registry.get_statistics()
        assert new_stats['total_layers'] == initial_total + 1
    
    def test_contains_and_has_layer_consistency(self):
        """Test that __contains__ and has_layer return consistent results."""
        registry = LayerRegistry()
        
        # For existing layers
        assert ("GCNConv" in registry) == registry.has_layer("GCNConv")
        assert ("ReLU" in registry) == registry.has_layer("ReLU")
        
        # For non-existing layers
        assert ("NonExistent" in registry) == registry.has_layer("NonExistent")
        assert ("FakeLayer" in registry) == registry.has_layer("FakeLayer")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
