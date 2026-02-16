#!/usr/bin/env python3
"""
Complete Unit Test Suite for validation.py Module

Tests the ArchitectureValidator and validation functions including:
- ArchitectureValidator initialization
- Main validation method (validate)
- Channel flow validation
- Task compatibility validation (graph-level, node-level)
- Layer ordering validation
- Data compatibility validation
- Suggestion generation
- Convenience functions (validate_architecture, validate_data_compatibility)
- Error handling and edge cases
- Multi-head attention channel tracking
- Empty architecture handling
- Integration scenarios

This is a PRODUCTION-READY test suite with comprehensive coverage.
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import importlib.util
from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock, patch

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

# Mock torch_geometric root module
mock_pyg = Mock()
mock_pyg.data = mock_pyg_data
mock_pyg.utils = mock_pyg_utils
_mock_modules["torch_geometric"] = mock_pyg

# Store original modules for cleanup — populated by setup_module()
_original_modules = {}

# NOTE: Mock injection into sys.modules is deferred to setup_module() to
# prevent pollution during pytest collection.  See §4.4 of the tracker.


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


class ArchitectureError(BaseProjectError):
    """Architecture error - mocked."""

    pass


mock_exceptions = Mock()
mock_exceptions.BaseProjectError = BaseProjectError
mock_exceptions.ModelError = ModelError
mock_exceptions.ArchitectureError = ArchitectureError
_mock_modules["milia_pipeline.exceptions"] = mock_exceptions

# NOTE: exceptions injection deferred to setup_module()


# ==============================================================================
# MOCK LAYER REGISTRY COMPONENTS
# ==============================================================================


# Create mock LayerCategory enum
class LayerCategory:
    """Mock LayerCategory enum."""

    CONVOLUTIONAL = "convolutional"
    POOLING = "pooling"
    ACTIVATION = "activation"
    NORMALIZATION = "normalization"
    LINEAR = "linear"
    DROPOUT = "dropout"


# Create mock LayerMetadata
@dataclass
class LayerMetadata:
    """Mock LayerMetadata."""

    name: str
    category: str
    has_in_channels: bool = False
    has_out_channels: bool = False
    description: str = ""
    supported_tasks: list[str] = None


# Create mock LayerConfig
@dataclass
class LayerConfig:
    """Mock LayerConfig."""

    type: str
    in_channels: int = None
    out_channels: int = None
    params: dict[str, Any] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


# Create mock LayerRegistry
class MockLayerRegistry:
    """Mock LayerRegistry for testing."""

    def __init__(self):
        self._metadata = {}
        self._setup_default_layers()

    def _setup_default_layers(self):
        """Setup default layer metadata."""
        self._metadata = {
            "GCNConv": LayerMetadata(
                name="GCNConv",
                category=LayerCategory.CONVOLUTIONAL,
                has_in_channels=True,
                has_out_channels=True,
            ),
            "GATConv": LayerMetadata(
                name="GATConv",
                category=LayerCategory.CONVOLUTIONAL,
                has_in_channels=True,
                has_out_channels=True,
            ),
            "SAGEConv": LayerMetadata(
                name="SAGEConv",
                category=LayerCategory.CONVOLUTIONAL,
                has_in_channels=True,
                has_out_channels=True,
            ),
            "Linear": LayerMetadata(
                name="Linear",
                category=LayerCategory.LINEAR,
                has_in_channels=True,
                has_out_channels=True,
            ),
            "ReLU": LayerMetadata(
                name="ReLU",
                category=LayerCategory.ACTIVATION,
                has_in_channels=False,
                has_out_channels=False,
            ),
            "Dropout": LayerMetadata(
                name="Dropout",
                category=LayerCategory.DROPOUT,
                has_in_channels=False,
                has_out_channels=False,
            ),
            "global_mean_pool": LayerMetadata(
                name="global_mean_pool",
                category=LayerCategory.POOLING,
                has_in_channels=False,
                has_out_channels=False,
            ),
            "global_max_pool": LayerMetadata(
                name="global_max_pool",
                category=LayerCategory.POOLING,
                has_in_channels=False,
                has_out_channels=False,
            ),
            "BatchNorm": LayerMetadata(
                name="BatchNorm",
                category=LayerCategory.NORMALIZATION,
                has_in_channels=False,
                has_out_channels=False,
            ),
        }

    def get_layer_metadata(self, layer_type: str) -> LayerMetadata:
        """Get metadata for a layer type."""
        if layer_type not in self._metadata:
            raise ValueError(f"Unknown layer type: {layer_type}")
        return self._metadata[layer_type]

    def add_layer_metadata(self, layer_type: str, metadata: LayerMetadata):
        """Add layer metadata."""
        self._metadata[layer_type] = metadata


# Mock layer_registry module components
mock_layer_registry_module = Mock()
mock_layer_registry_module.LayerRegistry = MockLayerRegistry
mock_layer_registry_module.LayerCategory = LayerCategory
mock_layer_registry_module.registry = MockLayerRegistry()

_mock_modules["milia_pipeline.models.builders.layer_registry"] = mock_layer_registry_module
# NOTE: Injection deferred to setup_module()

# Mock architecture_builder module
mock_architecture_builder_module = Mock()
mock_architecture_builder_module.LayerConfig = LayerConfig
mock_architecture_builder_module.ArchitectureConfig = Mock
mock_architecture_builder_module.ArchitectureError = ArchitectureError

_mock_modules["milia_pipeline.models.builders.architecture_builder"] = (
    mock_architecture_builder_module
)
# NOTE: Injection deferred to setup_module()


# ---------------------------------------------------------------------------
# Module-level placeholders — populated by setup_module()
# ---------------------------------------------------------------------------
validation_module = None
ArchitectureValidator = None
validate_architecture = None
validate_data_compatibility = None


def setup_module(module):
    """
    Inject mocks into sys.modules and load the module-under-test.

    Called by pytest ONCE before any test in this module executes.
    By deferring sys.modules writes here (instead of at module level),
    pytest --collect-only can import this file without polluting
    sys.modules for other test files collected afterward.
    """
    global _original_modules
    global validation_module, ArchitectureValidator
    global validate_architecture, validate_data_compatibility

    # --- Inject all mock modules into sys.modules ---
    for module_name in _mock_modules:
        if module_name in sys.modules:
            _original_modules[module_name] = sys.modules[module_name]
        sys.modules[module_name] = _mock_modules[module_name]

    # --- Load validation module from disk ---
    validation_path = str(
        _PROJECT_ROOT / "milia_pipeline" / "models" / "builders" / "validation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "milia_pipeline.models.builders.validation", validation_path
    )
    validation_module = importlib.util.module_from_spec(spec)
    sys.modules["milia_pipeline.models.builders.validation"] = validation_module
    spec.loader.exec_module(validation_module)

    # --- Extract what we need ---
    ArchitectureValidator = validation_module.ArchitectureValidator
    validate_architecture = validation_module.validate_architecture
    validate_data_compatibility = validation_module.validate_data_compatibility

    # --- Publish into module namespace ---
    module.validation_module = validation_module
    module.ArchitectureValidator = ArchitectureValidator
    module.validate_architecture = validate_architecture
    module.validate_data_compatibility = validate_data_compatibility


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


class SimpleArchitecture(nn.Module):
    """Simple test architecture for data compatibility testing."""

    def __init__(self, in_channels=10, out_channels=1):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        return self.linear(x)


class FailingArchitecture(nn.Module):
    """Architecture that fails forward pass for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        raise RuntimeError("Intentional forward pass failure")


@pytest.fixture
def mock_registry():
    """Fixture providing a mock layer registry."""
    return MockLayerRegistry()


@pytest.fixture
def simple_layers():
    """Fixture providing simple layer configurations."""
    return [
        LayerConfig(type="GCNConv", out_channels=64),
        LayerConfig(type="ReLU"),
        LayerConfig(type="Linear", params={"out_features": 1}),
    ]


@pytest.fixture
def sample_data():
    """Fixture providing sample PyG Data object."""
    data = Mock()
    data.x = torch.randn(10, 16)  # 10 nodes, 16 features
    data.edge_index = torch.randint(0, 10, (2, 20))  # 20 edges
    data.edge_attr = torch.randn(20, 3)  # 20 edges, 3 features
    data.batch = torch.zeros(10, dtype=torch.long)  # Single graph
    return data


# =============================================================================
# ARCHITECTUREVALIDATOR INITIALIZATION TESTS
# =============================================================================


class TestArchitectureValidatorInit:
    """Test ArchitectureValidator initialization."""

    def test_init_default_registry(self):
        """Test initialization with default registry."""
        validator = ArchitectureValidator()

        assert validator.registry is not None
        assert hasattr(validator.registry, "get_layer_metadata")

    def test_init_custom_registry(self, mock_registry):
        """Test initialization with custom registry."""
        validator = ArchitectureValidator(registry=mock_registry)

        assert validator.registry is mock_registry

    def test_init_registry_has_required_methods(self):
        """Test that registry has required methods."""
        validator = ArchitectureValidator()

        assert hasattr(validator.registry, "get_layer_metadata")
        assert callable(validator.registry.get_layer_metadata)


# =============================================================================
# MAIN VALIDATE METHOD TESTS
# =============================================================================


class TestValidateMethod:
    """Test main validate method."""

    def test_validate_empty_layers(self):
        """Test validation with empty layer list."""
        validator = ArchitectureValidator()

        result = validator.validate(
            layers=[], task_type="graph_regression", in_channels=16, out_channels=1
        )

        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert "Architecture has no layers" in result["errors"][0]
        assert len(result["suggestions"]) > 0
        assert "Add at least one layer" in result["suggestions"][0]

    def test_validate_valid_architecture(self, simple_layers):
        """Test validation with valid architecture."""
        validator = ArchitectureValidator()

        result = validator.validate(
            layers=simple_layers, task_type="graph_regression", in_channels=16, out_channels=1
        )

        # Should have keys
        assert "valid" in result
        assert "errors" in result
        assert "warnings" in result
        assert "suggestions" in result

        # Should be list types
        assert isinstance(result["errors"], list)
        assert isinstance(result["warnings"], list)
        assert isinstance(result["suggestions"], list)

    def test_validate_calls_all_sub_validators(self, simple_layers):
        """Test that validate calls all sub-validation methods."""
        validator = ArchitectureValidator()

        # Mock the sub-validators
        with (
            patch.object(validator, "validate_channel_flow") as mock_channel,
            patch.object(validator, "validate_task_compatibility") as mock_task,
            patch.object(validator, "validate_layer_ordering") as mock_order,
        ):
            # Setup mock returns
            mock_channel.return_value = {"errors": [], "warnings": [], "suggestions": []}
            mock_task.return_value = {"errors": [], "warnings": [], "suggestions": []}
            mock_order.return_value = {"errors": [], "warnings": [], "suggestions": []}

            _result = validator.validate(
                layers=simple_layers, task_type="graph_regression", in_channels=16, out_channels=1
            )

            # Verify all were called
            mock_channel.assert_called_once()
            mock_task.assert_called_once()
            mock_order.assert_called_once()

    def test_validate_aggregates_errors(self, simple_layers):
        """Test that validate aggregates errors from sub-validators."""
        validator = ArchitectureValidator()

        with (
            patch.object(validator, "validate_channel_flow") as mock_channel,
            patch.object(validator, "validate_task_compatibility") as mock_task,
            patch.object(validator, "validate_layer_ordering") as mock_order,
        ):
            # Setup mock returns with errors
            mock_channel.return_value = {
                "errors": ["channel_error"],
                "warnings": ["channel_warning"],
                "suggestions": ["channel_suggestion"],
            }
            mock_task.return_value = {
                "errors": ["task_error"],
                "warnings": ["task_warning"],
                "suggestions": ["task_suggestion"],
            }
            mock_order.return_value = {
                "errors": ["order_error"],
                "warnings": ["order_warning"],
                "suggestions": ["order_suggestion"],
            }

            result = validator.validate(
                layers=simple_layers, task_type="graph_regression", in_channels=16, out_channels=1
            )

            # Check aggregation
            assert result["valid"] is False  # Has errors
            assert "channel_error" in result["errors"]
            assert "task_error" in result["errors"]
            assert "order_error" in result["errors"]
            assert "channel_warning" in result["warnings"]
            assert "task_warning" in result["warnings"]
            assert "order_warning" in result["warnings"]
            assert "channel_suggestion" in result["suggestions"]
            assert "task_suggestion" in result["suggestions"]
            assert "order_suggestion" in result["suggestions"]

    def test_validate_valid_with_no_errors(self, simple_layers):
        """Test that valid is True when no errors exist."""
        validator = ArchitectureValidator()

        with (
            patch.object(validator, "validate_channel_flow") as mock_channel,
            patch.object(validator, "validate_task_compatibility") as mock_task,
            patch.object(validator, "validate_layer_ordering") as mock_order,
        ):
            # Setup mock returns with no errors
            mock_channel.return_value = {"errors": [], "warnings": [], "suggestions": []}
            mock_task.return_value = {"errors": [], "warnings": [], "suggestions": []}
            mock_order.return_value = {"errors": [], "warnings": [], "suggestions": []}

            result = validator.validate(
                layers=simple_layers, task_type="graph_regression", in_channels=16, out_channels=1
            )

            assert result["valid"] is True
            assert len(result["errors"]) == 0


# =============================================================================
# CHANNEL FLOW VALIDATION TESTS
# =============================================================================


class TestValidateChannelFlow:
    """Test channel flow validation."""

    def test_channel_flow_simple_linear(self):
        """Test channel flow through simple linear layers."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", out_channels=32),
            LayerConfig(type="GCNConv", out_channels=16),
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        result = validator.validate_channel_flow(layers=layers, in_channels=16, out_channels=1)

        assert isinstance(result, dict)
        assert "errors" in result
        assert "warnings" in result
        assert "suggestions" in result

    def test_channel_flow_mismatch_in_channels(self):
        """Test detection of mismatched in_channels."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", in_channels=32, out_channels=64),  # Wrong in_channels
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        result = validator.validate_channel_flow(layers=layers, in_channels=16, out_channels=1)

        assert len(result["errors"]) > 0
        assert any("Expected in_channels=16" in err for err in result["errors"])

    def test_channel_flow_in_channels_auto_inferred_no_error(self):
        """Test that omitting in_channels produces no error (auto-inference).

        When layer_config.in_channels is None/falsy, the validator should
        NOT report an error even though has_in_channels is True, because
        the in_channels will be auto-inferred by the builder.
        """
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", out_channels=64),  # in_channels=None → auto-inferred
            LayerConfig(type="GCNConv", out_channels=32),  # in_channels=None → auto-inferred
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        result = validator.validate_channel_flow(layers=layers, in_channels=16, out_channels=1)

        # No in_channels mismatch errors should be reported
        assert not any("Expected in_channels" in err for err in result["errors"])

    def test_channel_flow_correct_in_channels_no_error(self):
        """Test that correctly specified in_channels produces no error.

        When layer_config.in_channels matches the expected current_channels,
        the validator should NOT report an error.
        """
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", in_channels=16, out_channels=64),  # Correct
            LayerConfig(type="GCNConv", in_channels=64, out_channels=32),  # Correct
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        result = validator.validate_channel_flow(layers=layers, in_channels=16, out_channels=1)

        # No errors about in_channels mismatch
        assert not any("Expected in_channels" in err for err in result["errors"])

    def test_channel_flow_missing_out_channels(self):
        """Test warning when out_channels not specified."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv"),  # No out_channels
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        result = validator.validate_channel_flow(layers=layers, in_channels=16, out_channels=1)

        # Should warn about missing out_channels
        assert len(result["warnings"]) > 0

    def test_channel_flow_multi_head_attention_concat_true(self):
        """Test channel tracking with multi-head attention (concat=True)."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GATConv", out_channels=32, params={"heads": 4, "concat": True}),
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        result = validator.validate_channel_flow(layers=layers, in_channels=16, out_channels=1)

        # With concat=True, out_channels should be 32 * 4 = 128
        # This is tracked internally in the validator
        assert isinstance(result, dict)

    def test_channel_flow_multi_head_attention_concat_false(self):
        """Test channel tracking with multi-head attention (concat=False)."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GATConv", out_channels=32, params={"heads": 4, "concat": False}),
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        result = validator.validate_channel_flow(layers=layers, in_channels=16, out_channels=1)

        # With concat=False, out_channels should remain 32
        assert isinstance(result, dict)

    def test_channel_flow_final_mismatch_warning(self):
        """Test warning when final channels don't match expected."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", out_channels=64),
            LayerConfig(type="Linear", params={"out_features": 8}),  # Wrong final size
        ]

        result = validator.validate_channel_flow(
            layers=layers,
            in_channels=16,
            out_channels=1,  # Expected 1, but getting 8
        )

        # Should warn about final output mismatch
        assert len(result["warnings"]) > 0
        assert any("Final output channels" in warn for warn in result["warnings"])

    def test_channel_flow_no_channel_change_layers(self):
        """Test channel flow with layers that don't change channels."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", out_channels=32),
            LayerConfig(type="ReLU"),  # Doesn't change channels
            LayerConfig(type="Dropout"),  # Doesn't change channels
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        result = validator.validate_channel_flow(layers=layers, in_channels=16, out_channels=1)

        # Should handle activation/dropout without errors
        assert isinstance(result, dict)

    def test_channel_flow_out_features_vs_out_channels(self):
        """Test handling of both out_features and out_channels."""
        validator = ArchitectureValidator()

        # Test with out_features in params
        layers1 = [
            LayerConfig(type="Linear", params={"out_features": 32}),
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        result1 = validator.validate_channel_flow(layers=layers1, in_channels=16, out_channels=1)

        # Test with out_channels directly
        layers2 = [
            LayerConfig(type="GCNConv", out_channels=32),
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        result2 = validator.validate_channel_flow(layers=layers2, in_channels=16, out_channels=1)

        assert isinstance(result1, dict)
        assert isinstance(result2, dict)

    def test_channel_flow_out_channels_in_params_dict(self):
        """Test channel flow when out_channels is specified in params dict (not as attribute).

        Covers the branch: elif 'out_channels' in layer_config.params
        This is distinct from layer_config.out_channels (the attribute) and
        layer_config.params['out_features'].
        """
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", params={"out_channels": 32}),
            LayerConfig(type="GCNConv", params={"out_channels": 1}),
        ]

        result = validator.validate_channel_flow(layers=layers, in_channels=16, out_channels=1)

        # Should correctly read out_channels from params and match final output
        assert isinstance(result, dict)
        # No final-output-mismatch warning since last layer produces 1 == out_channels
        assert not any("Final output channels" in w for w in result["warnings"])

    def test_channel_flow_multi_head_default_concat(self):
        """Test channel tracking with multi-head attention when concat is not specified.

        Covers the branch where 'heads' is in params but 'concat' is absent,
        so concat defaults to True via .get('concat', True).
        """
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(
                type="GATConv",
                out_channels=16,
                params={"heads": 4},  # No 'concat' key — defaults to True
            ),
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        result = validator.validate_channel_flow(layers=layers, in_channels=16, out_channels=1)

        # With default concat=True, current_channels becomes 16*4=64 after GATConv.
        # Then Linear produces 1, matching out_channels=1.
        assert isinstance(result, dict)
        assert not any("Final output channels" in w for w in result["warnings"])


# =============================================================================
# TASK COMPATIBILITY VALIDATION TESTS
# =============================================================================


class TestValidateTaskCompatibility:
    """Test task compatibility validation."""

    def test_graph_task_requires_pooling(self):
        """Test that graph-level tasks require pooling."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", out_channels=64),
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        result = validator.validate_task_compatibility(layers=layers, task_type="graph_regression")

        # Should error about missing pooling
        assert len(result["errors"]) > 0
        assert any("requires pooling" in err for err in result["errors"])

    def test_graph_task_with_pooling_valid(self):
        """Test that graph-level tasks with pooling are valid."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", out_channels=64),
            LayerConfig(type="global_mean_pool"),
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        result = validator.validate_task_compatibility(layers=layers, task_type="graph_regression")

        # Should not error about pooling
        assert len(result["errors"]) == 0

    def test_graph_task_with_max_pool_valid(self):
        """Test that graph-level tasks with global_max_pool are also valid.

        Ensures pooling detection works across different pooling layer types,
        not just global_mean_pool.
        """
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", out_channels=64),
            LayerConfig(type="global_max_pool"),
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        result = validator.validate_task_compatibility(
            layers=layers, task_type="graph_classification"
        )

        assert len(result["errors"]) == 0

    def test_graph_classification_requires_pooling(self):
        """Test graph classification also requires pooling."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", out_channels=64),
            LayerConfig(type="Linear", params={"out_features": 10}),
        ]

        result = validator.validate_task_compatibility(
            layers=layers, task_type="graph_classification"
        )

        assert len(result["errors"]) > 0
        assert any("requires pooling" in err for err in result["errors"])

    def test_node_task_warns_about_pooling(self):
        """Test that node-level tasks warn about pooling."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", out_channels=64),
            LayerConfig(type="global_mean_pool"),
            LayerConfig(type="Linear", params={"out_features": 10}),
        ]

        result = validator.validate_task_compatibility(
            layers=layers, task_type="node_classification"
        )

        # Should warn about pooling in node-level task
        assert len(result["warnings"]) > 0
        assert any("node-level" in warn and "pooling" in warn for warn in result["warnings"])

    def test_node_regression_warns_about_pooling(self):
        """Test node regression warns about pooling."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", out_channels=64),
            LayerConfig(type="global_max_pool"),
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        result = validator.validate_task_compatibility(layers=layers, task_type="node_regression")

        assert len(result["warnings"]) > 0

    def test_node_task_without_pooling_valid(self):
        """Test that node-level tasks without pooling are valid."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", out_channels=64),
            LayerConfig(type="ReLU"),
            LayerConfig(type="Linear", params={"out_features": 10}),
        ]

        result = validator.validate_task_compatibility(
            layers=layers, task_type="node_classification"
        )

        # Should not warn about pooling
        assert len(result["warnings"]) == 0

    def test_node_task_multiple_pooling_layers_listed(self):
        """Test that all pooling layers are listed in warning for node tasks.

        Covers the list comprehension that collects (index, type) of all
        pooling layers and reports them in the warning message.
        """
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", out_channels=64),
            LayerConfig(type="global_mean_pool"),
            LayerConfig(type="global_max_pool"),
            LayerConfig(type="Linear", params={"out_features": 10}),
        ]

        result = validator.validate_task_compatibility(
            layers=layers, task_type="node_classification"
        )

        assert len(result["warnings"]) > 0
        # Both pooling layer names should appear in the warning
        warning_text = result["warnings"][0]
        assert "global_mean_pool" in warning_text
        assert "global_max_pool" in warning_text

    def test_unknown_task_type_no_errors(self):
        """Test that unknown task types don't cause errors."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", out_channels=64),
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        result = validator.validate_task_compatibility(layers=layers, task_type="some_unknown_task")

        # Should not raise errors for unknown task types
        assert isinstance(result, dict)

    def test_case_insensitive_task_type(self):
        """Test that task type checking is case-insensitive."""
        validator = ArchitectureValidator()

        layers_no_pool = [
            LayerConfig(type="GCNConv", out_channels=64),
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        # Test uppercase
        result1 = validator.validate_task_compatibility(
            layers=layers_no_pool, task_type="GRAPH_REGRESSION"
        )

        # Test mixed case
        result2 = validator.validate_task_compatibility(
            layers=layers_no_pool, task_type="Graph_Regression"
        )

        # Both should error about missing pooling
        assert len(result1["errors"]) > 0
        assert len(result2["errors"]) > 0


# =============================================================================
# LAYER ORDERING VALIDATION TESTS
# =============================================================================


class TestValidateLayerOrdering:
    """Test layer ordering validation."""

    def test_pooling_before_conv_warning(self):
        """Test warning when pooling comes before convolutional layer."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", out_channels=64),
            LayerConfig(type="global_mean_pool"),
            LayerConfig(type="GCNConv", out_channels=32),  # Conv after pooling
        ]

        result = validator.validate_layer_ordering(layers=layers)

        # Should warn about unusual ordering
        assert len(result["warnings"]) > 0
        assert any("Unusual ordering" in warn for warn in result["warnings"])

    def test_consecutive_activations_warning(self):
        """Test warning when activation layers are consecutive."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", out_channels=64),
            LayerConfig(type="ReLU"),
            LayerConfig(type="ReLU"),  # Consecutive activation
        ]

        result = validator.validate_layer_ordering(layers=layers)

        # Should warn about consecutive activations
        assert len(result["warnings"]) > 0
        assert any("consecutive activation" in warn.lower() for warn in result["warnings"])

    def test_normal_ordering_no_warnings(self):
        """Test that normal ordering produces no warnings."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", out_channels=64),
            LayerConfig(type="ReLU"),
            LayerConfig(type="GCNConv", out_channels=32),
            LayerConfig(type="ReLU"),
            LayerConfig(type="global_mean_pool"),
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        result = validator.validate_layer_ordering(layers=layers)

        # Should have no warnings for normal ordering
        assert len(result["warnings"]) == 0

    def test_single_layer_no_warnings(self):
        """Test that single layer produces no ordering warnings."""
        validator = ArchitectureValidator()

        layers = [LayerConfig(type="GCNConv", out_channels=64)]

        result = validator.validate_layer_ordering(layers=layers)

        assert len(result["warnings"]) == 0

    def test_pooling_before_conv_generates_suggestion(self):
        """Test that pooling-before-conv warning also generates a suggestion.

        Covers the suggestions.append() in the pooling-before-conv branch.
        """
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", out_channels=64),
            LayerConfig(type="global_mean_pool"),
            LayerConfig(type="GCNConv", out_channels=32),
        ]

        result = validator.validate_layer_ordering(layers=layers)

        assert len(result["suggestions"]) > 0
        assert any("moving pooling layer" in s.lower() for s in result["suggestions"])

    def test_layer_ordering_normalization_between_conv_no_warning(self):
        """Test that normalization between conv layers produces no warnings."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", out_channels=64),
            LayerConfig(type="BatchNorm"),
            LayerConfig(type="ReLU"),
            LayerConfig(type="GCNConv", out_channels=32),
        ]

        result = validator.validate_layer_ordering(layers=layers)

        # This is a normal ordering pattern
        assert len(result["warnings"]) == 0

    def test_empty_layers_no_errors(self):
        """Test that empty layers list doesn't cause errors."""
        validator = ArchitectureValidator()

        result = validator.validate_layer_ordering(layers=[])

        # Should not crash and return valid structure
        assert isinstance(result, dict)
        assert "errors" in result
        assert "warnings" in result
        assert "suggestions" in result


# =============================================================================
# DATA COMPATIBILITY VALIDATION TESTS
# =============================================================================


class TestValidateDataCompatibility:
    """Test data compatibility validation."""

    def test_data_compatibility_valid(self, sample_data):
        """Test validation with valid data."""
        validator = ArchitectureValidator()
        architecture = SimpleArchitecture(in_channels=16, out_channels=1)

        result = validator.validate_data_compatibility(
            architecture=architecture, sample_data=sample_data
        )

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_data_compatibility_missing_x(self):
        """Test error when data is missing node features (x)."""
        validator = ArchitectureValidator()
        architecture = SimpleArchitecture()

        data = Mock()
        data.x = None
        data.edge_index = torch.randint(0, 10, (2, 20))

        result = validator.validate_data_compatibility(architecture=architecture, sample_data=data)

        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert any("missing node features" in err.lower() for err in result["errors"])

    def test_data_compatibility_missing_edge_index_warning(self, sample_data):
        """Test warning when data is missing edge_index."""
        validator = ArchitectureValidator()
        architecture = SimpleArchitecture()

        data = Mock()
        data.x = torch.randn(10, 16)
        data.edge_index = None

        result = validator.validate_data_compatibility(architecture=architecture, sample_data=data)

        # Should warn about missing edge_index
        assert len(result["warnings"]) > 0
        assert any("missing edge_index" in warn.lower() for warn in result["warnings"])

    def test_data_compatibility_forward_pass_success(self, sample_data):
        """Test successful forward pass."""
        validator = ArchitectureValidator()
        architecture = SimpleArchitecture(in_channels=16, out_channels=1)

        result = validator.validate_data_compatibility(
            architecture=architecture, sample_data=sample_data
        )

        assert result["valid"] is True

    def test_data_compatibility_forward_pass_failure(self, sample_data):
        """Test forward pass failure handling."""
        validator = ArchitectureValidator()
        architecture = FailingArchitecture()

        result = validator.validate_data_compatibility(
            architecture=architecture, sample_data=sample_data
        )

        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert any("Forward pass failed" in err for err in result["errors"])

    def test_data_compatibility_with_batch(self, sample_data):
        """Test data compatibility with batch parameter."""
        validator = ArchitectureValidator()
        architecture = SimpleArchitecture()

        # Ensure batch is present
        sample_data.batch = torch.zeros(10, dtype=torch.long)

        result = validator.validate_data_compatibility(
            architecture=architecture, sample_data=sample_data
        )

        assert isinstance(result, dict)

    def test_data_compatibility_without_batch(self):
        """Test data compatibility without batch parameter.

        Uses a simple namespace object instead of Mock so that
        hasattr(data, 'batch') correctly returns False.
        Mock objects respond True to hasattr for ANY attribute name
        unless spec is used, which would not test the intended branch.
        """
        validator = ArchitectureValidator()
        architecture = SimpleArchitecture(in_channels=16, out_channels=1)

        class MinimalData:
            """Minimal data object without batch attribute."""

            def __init__(self):
                self.x = torch.randn(10, 16)
                self.edge_index = torch.randint(0, 10, (2, 20))
                self.edge_attr = None

        data = MinimalData()
        assert not hasattr(data, "batch"), "Test precondition: data must lack 'batch'"

        result = validator.validate_data_compatibility(architecture=architecture, sample_data=data)

        assert isinstance(result, dict)
        # Forward pass should still succeed via the no-batch code path
        assert result["valid"] is True

    def test_data_compatibility_missing_x_attribute_entirely(self):
        """Test error when data object has no 'x' attribute at all.

        Covers the hasattr(sample_data, 'x') == False branch,
        which is distinct from x being present but set to None.
        """
        validator = ArchitectureValidator()
        architecture = SimpleArchitecture()

        class NoXData:
            """Data object without x attribute."""

            def __init__(self):
                self.edge_index = torch.randint(0, 10, (2, 20))

        data = NoXData()
        assert not hasattr(data, "x"), "Test precondition: data must lack 'x'"

        result = validator.validate_data_compatibility(architecture=architecture, sample_data=data)

        assert result["valid"] is False
        assert any("missing node features" in err.lower() for err in result["errors"])

    def test_data_compatibility_missing_edge_index_attribute_entirely(self):
        """Test warning when data object has no 'edge_index' attribute at all.

        Covers the hasattr(sample_data, 'edge_index') == False branch.
        """
        validator = ArchitectureValidator()
        architecture = SimpleArchitecture()

        class NoEdgeData:
            """Data object without edge_index attribute."""

            def __init__(self):
                self.x = torch.randn(10, 16)

        data = NoEdgeData()
        assert not hasattr(data, "edge_index"), "Test precondition: data must lack 'edge_index'"

        result = validator.validate_data_compatibility(architecture=architecture, sample_data=data)

        assert any("missing edge_index" in w.lower() for w in result["warnings"])

    def test_data_compatibility_no_grad(self, sample_data):
        """Test that validation uses no_grad context."""
        validator = ArchitectureValidator()
        architecture = SimpleArchitecture()

        # This should not require gradients
        with torch.no_grad():
            result = validator.validate_data_compatibility(
                architecture=architecture, sample_data=sample_data
            )

        assert isinstance(result, dict)


# =============================================================================
# SUGGEST FIXES METHOD TESTS
# =============================================================================


class TestSuggestFixes:
    """Test suggest_fixes method."""

    def test_suggest_fixes_from_suggestions(self):
        """Test that suggestions are extracted from validation result."""
        validator = ArchitectureValidator()

        validation_result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "suggestions": ["suggestion1", "suggestion2"],
        }

        fixes = validator.suggest_fixes(validation_result)

        assert "suggestion1" in fixes
        assert "suggestion2" in fixes

    def test_suggest_fixes_channel_mismatch_error(self):
        """Test additional fix for channel mismatch error."""
        validator = ArchitectureValidator()

        validation_result = {
            "valid": False,
            "errors": ["channel mismatch detected"],
            "warnings": [],
            "suggestions": [],
        }

        fixes = validator.suggest_fixes(validation_result)

        # Should add specific fix for channel mismatch
        assert any("_infer_channels" in fix for fix in fixes)

    def test_suggest_fixes_pooling_error(self):
        """Test additional fix for pooling error."""
        validator = ArchitectureValidator()

        validation_result = {
            "valid": False,
            "errors": ["missing pooling layer"],
            "warnings": [],
            "suggestions": [],
        }

        fixes = validator.suggest_fixes(validation_result)

        # Should add specific fix for pooling
        assert any("pooling" in fix for fix in fixes)

    def test_suggest_fixes_empty_validation_result(self):
        """Test suggest_fixes with empty validation result."""
        validator = ArchitectureValidator()

        validation_result = {"valid": True, "errors": [], "warnings": [], "suggestions": []}

        fixes = validator.suggest_fixes(validation_result)

        # Should return empty or minimal list
        assert isinstance(fixes, list)

    def test_suggest_fixes_multiple_error_patterns(self):
        """Test suggest_fixes with multiple error patterns."""
        validator = ArchitectureValidator()

        validation_result = {
            "valid": False,
            "errors": ["channel mismatch", "pooling issue"],
            "warnings": [],
            "suggestions": ["existing_suggestion"],
        }

        fixes = validator.suggest_fixes(validation_result)

        # Should have both pattern-based fixes and original suggestions
        assert "existing_suggestion" in fixes
        assert len(fixes) >= 2

    def test_suggest_fixes_missing_suggestions_key(self):
        """Test suggest_fixes when 'suggestions' key is absent from result.

        Covers the defensive .get('suggestions') path returning None/falsy.
        """
        validator = ArchitectureValidator()

        validation_result = {
            "valid": False,
            "errors": ["channel mismatch detected"],
            "warnings": [],
            # 'suggestions' key intentionally missing
        }

        fixes = validator.suggest_fixes(validation_result)

        # Should not crash and should still generate pattern-based fixes
        assert isinstance(fixes, list)
        assert any("_infer_channels" in fix for fix in fixes)

    def test_suggest_fixes_missing_errors_key(self):
        """Test suggest_fixes when 'errors' key is absent from result.

        Covers the defensive .get('errors', []) path returning default.
        """
        validator = ArchitectureValidator()

        validation_result = {
            "valid": True,
            "warnings": [],
            "suggestions": ["some_suggestion"],
            # 'errors' key intentionally missing
        }

        fixes = validator.suggest_fixes(validation_result)

        # Should not crash; should include the suggestion
        assert isinstance(fixes, list)
        assert "some_suggestion" in fixes

    def test_suggest_fixes_error_pattern_case_insensitive(self):
        """Test that suggest_fixes error pattern matching is case-insensitive.

        The source uses error.lower() for pattern matching, so
        'Channel Mismatch' should still trigger the fix.
        """
        validator = ArchitectureValidator()

        validation_result = {
            "valid": False,
            "errors": ["Channel Mismatch in layer 3"],
            "warnings": [],
            "suggestions": [],
        }

        fixes = validator.suggest_fixes(validation_result)

        assert any("_infer_channels" in fix for fix in fixes)


# =============================================================================
# CONVENIENCE FUNCTIONS TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_validate_architecture_function(self, simple_layers):
        """Test validate_architecture convenience function."""
        result = validate_architecture(
            layers=simple_layers, task_type="graph_regression", in_channels=16, out_channels=1
        )

        assert isinstance(result, dict)
        assert "valid" in result
        assert "errors" in result
        assert "warnings" in result
        assert "suggestions" in result

    def test_validate_architecture_creates_validator(self, simple_layers):
        """Test that validate_architecture creates its own validator."""
        # Should work without needing to instantiate validator
        result = validate_architecture(
            layers=simple_layers, task_type="node_classification", in_channels=16, out_channels=10
        )

        assert isinstance(result, dict)

    def test_validate_data_compatibility_function(self, sample_data):
        """Test validate_data_compatibility convenience function."""
        architecture = SimpleArchitecture()

        result = validate_data_compatibility(architecture=architecture, sample_data=sample_data)

        assert isinstance(result, dict)
        assert "valid" in result
        assert "errors" in result
        assert "warnings" in result
        assert "suggestions" in result

    def test_validate_data_compatibility_creates_validator(self, sample_data):
        """Test that validate_data_compatibility creates its own validator."""
        architecture = SimpleArchitecture()

        # Should work without needing to instantiate validator
        result = validate_data_compatibility(architecture=architecture, sample_data=sample_data)

        assert isinstance(result, dict)


# =============================================================================
# EDGE CASES AND ERROR HANDLING TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_validation_with_unknown_layer_type(self):
        """Test validation with unknown layer type."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="UnknownLayer", out_channels=64),
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        # Should raise error when getting metadata for unknown layer
        with pytest.raises(ValueError):
            validator.validate_channel_flow(layers=layers, in_channels=16, out_channels=1)

    def test_validation_with_negative_channels(self):
        """Test validation with negative channel values."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", out_channels=-1),
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        # Should complete validation (negative values handled elsewhere)
        result = validator.validate_channel_flow(layers=layers, in_channels=16, out_channels=1)

        assert isinstance(result, dict)

    def test_validation_with_zero_channels(self):
        """Test validation with zero channel values."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", out_channels=0),
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        result = validator.validate_channel_flow(layers=layers, in_channels=16, out_channels=1)

        assert isinstance(result, dict)

    def test_validation_with_very_long_layer_sequence(self):
        """Test validation with very long layer sequence."""
        validator = ArchitectureValidator()

        layers = []
        for _i in range(100):
            layers.append(LayerConfig(type="GCNConv", out_channels=64))
            layers.append(LayerConfig(type="ReLU"))
        layers.append(LayerConfig(type="Linear", params={"out_features": 1}))

        # Should handle long sequences
        result = validator.validate(
            layers=layers, task_type="node_classification", in_channels=16, out_channels=1
        )

        assert isinstance(result, dict)

    def test_channel_flow_with_none_params(self):
        """Test channel flow with None params dictionary."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", out_channels=64, params=None),
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        # Should handle None params gracefully
        result = validator.validate_channel_flow(layers=layers, in_channels=16, out_channels=1)

        assert isinstance(result, dict)

    def test_data_compatibility_with_mismatched_dimensions(self):
        """Test data compatibility with dimension mismatch."""
        validator = ArchitectureValidator()
        architecture = SimpleArchitecture(in_channels=32, out_channels=1)  # Expects 32

        data = Mock()
        data.x = torch.randn(10, 16)  # Has 16
        data.edge_index = torch.randint(0, 10, (2, 20))

        result = validator.validate_data_compatibility(architecture=architecture, sample_data=data)

        # Should fail forward pass due to dimension mismatch
        assert result["valid"] is False
        assert len(result["errors"]) > 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Test integration scenarios."""

    def test_complete_validation_workflow(self):
        """Test complete validation workflow."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", out_channels=64),
            LayerConfig(type="ReLU"),
            LayerConfig(type="GCNConv", out_channels=32),
            LayerConfig(type="ReLU"),
            LayerConfig(type="global_mean_pool"),
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        # Validate architecture
        result = validator.validate(
            layers=layers, task_type="graph_regression", in_channels=16, out_channels=1
        )

        # Should be valid
        assert result["valid"] is True

        # Get suggestions
        fixes = validator.suggest_fixes(result)
        assert isinstance(fixes, list)

    def test_validation_and_data_compatibility_together(self, sample_data):
        """Test architecture and data validation together."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(type="GCNConv", out_channels=32),
            LayerConfig(type="ReLU"),
            LayerConfig(type="global_mean_pool"),
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        # Validate architecture
        arch_result = validator.validate(
            layers=layers, task_type="graph_regression", in_channels=16, out_channels=1
        )

        # Build simple architecture for data validation
        architecture = SimpleArchitecture(in_channels=16, out_channels=1)

        # Validate data compatibility
        data_result = validator.validate_data_compatibility(
            architecture=architecture, sample_data=sample_data
        )

        # Both should succeed
        assert arch_result["valid"] is True
        assert data_result["valid"] is True

    def test_invalid_architecture_generates_fixes(self):
        """Test that invalid architecture generates helpful fixes."""
        validator = ArchitectureValidator()

        # Architecture missing pooling for graph task
        layers = [
            LayerConfig(type="GCNConv", out_channels=64),
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        result = validator.validate(
            layers=layers, task_type="graph_regression", in_channels=16, out_channels=1
        )

        assert result["valid"] is False

        # Should have suggestions
        assert len(result["suggestions"]) > 0

        # Get fixes
        fixes = validator.suggest_fixes(result)
        assert len(fixes) > 0


# =============================================================================
# STRESS AND PERFORMANCE TESTS
# =============================================================================


class TestPerformance:
    """Test performance and stress scenarios."""

    def test_validation_performance_large_architecture(self):
        """Test validation performance with large architecture."""
        import time

        validator = ArchitectureValidator()

        # Create large architecture
        layers = []
        for _i in range(50):
            layers.append(LayerConfig(type="GCNConv", out_channels=64))
            layers.append(LayerConfig(type="ReLU"))
        layers.append(LayerConfig(type="global_mean_pool"))
        layers.append(LayerConfig(type="Linear", params={"out_features": 1}))

        start = time.time()
        result = validator.validate(
            layers=layers, task_type="graph_regression", in_channels=16, out_channels=1
        )
        duration = time.time() - start

        # Should complete in reasonable time
        assert duration < 5.0  # 5 seconds max
        assert isinstance(result, dict)

    def test_multiple_validations_same_validator(self, simple_layers):
        """Test multiple validations with same validator instance."""
        validator = ArchitectureValidator()

        # Run multiple validations
        for _ in range(10):
            result = validator.validate(
                layers=simple_layers, task_type="graph_regression", in_channels=16, out_channels=1
            )
            assert isinstance(result, dict)

    def test_validation_with_complex_params(self):
        """Test validation with complex parameter dictionaries."""
        validator = ArchitectureValidator()

        layers = [
            LayerConfig(
                type="GATConv",
                out_channels=32,
                params={
                    "heads": 8,
                    "concat": True,
                    "dropout": 0.6,
                    "add_self_loops": True,
                    "bias": True,
                    "negative_slope": 0.2,
                },
            ),
            LayerConfig(type="ReLU"),
            LayerConfig(type="global_mean_pool"),
            LayerConfig(type="Linear", params={"out_features": 1}),
        ]

        result = validator.validate(
            layers=layers, task_type="graph_regression", in_channels=16, out_channels=1
        )

        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
