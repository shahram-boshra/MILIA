#!/usr/bin/env python3
"""
Complete Unit Test Suite for model_registry.py Module

Tests model registry system including:
- ModelRegistration Pydantic model (migrated from dataclass in Phase 24)
- ModelRegistry singleton class
- Thread-safe operations
- Auto-discovery mechanism
- Model registration/unregistration
- Query methods (get_model, has_model, list_available_models, etc.)
- Filtering and search functionality
- Plugin management
- Statistics and reporting
- Global registry instance and convenience functions
- Edge cases and error handling

This is a PRODUCTION-READY test suite with comprehensive coverage.

**UPDATED (2025-12-08)**: Imports changed from model_categories.py to pyg_introspector.py
following the dynamic introspection refactoring.

**UPDATED (2025-12-22)**: Fixed test cases that incorrectly mocked module-level functions
instead of introspector instance methods. Added tests for _auto_discovered flag
preventing redundant discovery (Phase 3 feature). Added test for get_metadata
fallback to introspection behavior.

**UPDATED (2025-02-04)**: Updated tests for Pydantic V2 migration (Phase 24).
ModelRegistration is now a Pydantic BaseModel, not a dataclass. Tests updated to use
Pydantic's model_fields instead of dataclasses.fields(), and added tests for
to_dict() method and ConfigDict settings.

Author: milia Team
Version: 1.2.0
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import threading
import time
from datetime import datetime
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
from pydantic import BaseModel

# Import the module under test
from milia_pipeline.models.registry.model_registry import (
    # Pydantic model (Phase 24: migrated from dataclass)
    ModelRegistration,
    # Main class
    ModelRegistry,
    # Convenience functions
    get_model,
    get_model_info,
    has_model,
    list_models,
    # Global instance
    registry,
)

# Import dependencies - UPDATED: Now from pyg_introspector (model_categories.py deleted)
from milia_pipeline.models.registry.pyg_introspector import (
    ModelCategory,
    ModelMetadata,  # Alias for DynamicModelMetadata
)

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_torch_module():
    """Create a mock torch.nn.Module class for testing."""

    class MockModel(torch.nn.Module):
        def __init__(self, in_channels=10, out_channels=5):
            super().__init__()
            self.linear = torch.nn.Linear(in_channels, out_channels)

        def forward(self, x):
            return self.linear(x)

    return MockModel


@pytest.fixture
def sample_metadata():
    """Create sample ModelMetadata for testing.

    Note: ModelMetadata is now an alias for DynamicModelMetadata
    from pyg_introspector.py (model_categories.py has been deleted).
    """
    return ModelMetadata(
        name="TestModel",
        category=ModelCategory.BASIC_GNN,
        import_path="torch_geometric.nn.models.TestModel",
        description="Test model for unit testing",
        supported_tasks=["node_classification", "graph_regression"],
        tags=["test", "mock"],
        requires_edge_features=False,
        requires_edge_weights=False,
        supports_heterogeneous=False,
    )


@pytest.fixture
def fresh_registry():
    """Create a fresh registry instance for each test."""
    # Get the singleton instance
    reg = ModelRegistry.get_instance()
    # Reset it to clean state
    reg.reset()
    yield reg
    # Clean up after test
    reg.reset()


@pytest.fixture
def isolated_registry():
    """Create an isolated registry by clearing the singleton cache."""
    # Store original instance
    original_instances = ModelRegistry._instances.copy()

    # Clear the singleton cache
    ModelRegistry._instances.clear()

    # Create new instance
    with patch.object(ModelRegistry, "auto_discover_pyg_models", return_value=0):
        with patch.object(ModelRegistry, "log_availability_summary"):
            reg = ModelRegistry()

    yield reg

    # Restore original singleton cache
    ModelRegistry._instances.clear()
    ModelRegistry._instances.update(original_instances)


# =============================================================================
# PYDANTIC MODEL TESTS (ModelRegistration)
# =============================================================================


class TestModelRegistrationPydanticModel:
    """Test ModelRegistration Pydantic model.

    Phase 24 Migration: ModelRegistration migrated from @dataclass to Pydantic BaseModel.
    Tests verify Pydantic V2 behavior while maintaining backward compatibility.
    """

    def test_is_pydantic_model(self):
        """Test that ModelRegistration is a Pydantic BaseModel."""
        assert issubclass(ModelRegistration, BaseModel)

    def test_required_fields(self, mock_torch_module, sample_metadata):
        """Test required fields of ModelRegistration."""
        registration = ModelRegistration(
            name="TestModel", model_class=mock_torch_module, metadata=sample_metadata
        )
        assert registration.name == "TestModel"
        assert registration.model_class == mock_torch_module
        assert registration.metadata == sample_metadata

    def test_optional_fields_defaults(self, mock_torch_module, sample_metadata):
        """Test optional fields have correct defaults."""
        registration = ModelRegistration(
            name="TestModel", model_class=mock_torch_module, metadata=sample_metadata
        )
        assert registration.is_builtin is True
        assert registration.plugin_name is None
        assert registration.registered_at is None

    def test_all_fields_present(self):
        """Test all expected fields are present using Pydantic model_fields."""
        field_names = set(ModelRegistration.model_fields.keys())
        expected_fields = {
            "name",
            "model_class",
            "metadata",
            "is_builtin",
            "plugin_name",
            "registered_at",
        }
        assert field_names == expected_fields

    def test_with_all_fields(self, mock_torch_module, sample_metadata):
        """Test creating registration with all fields specified."""
        timestamp = datetime.now().isoformat()
        registration = ModelRegistration(
            name="CustomModel",
            model_class=mock_torch_module,
            metadata=sample_metadata,
            is_builtin=False,
            plugin_name="my_plugin",
            registered_at=timestamp,
        )
        assert registration.name == "CustomModel"
        assert registration.model_class == mock_torch_module
        assert registration.metadata == sample_metadata
        assert registration.is_builtin is False
        assert registration.plugin_name == "my_plugin"
        assert registration.registered_at == timestamp

    def test_field_annotations(self):
        """Test field type annotations are correct using Pydantic model_fields."""
        fields_info = ModelRegistration.model_fields

        assert fields_info["name"].annotation == str
        assert fields_info["model_class"].annotation == type[torch.nn.Module]
        assert fields_info["metadata"].annotation == ModelMetadata
        assert fields_info["is_builtin"].annotation == bool
        assert fields_info["plugin_name"].annotation == Optional[str]
        assert fields_info["registered_at"].annotation == Optional[str]

    def test_arbitrary_types_allowed(self):
        """Test that arbitrary_types_allowed is set in model_config.

        Required for Type[torch.nn.Module] field to work properly.
        """
        assert hasattr(ModelRegistration, "model_config")
        # Pydantic V2 uses ConfigDict
        config = ModelRegistration.model_config
        assert config.get("arbitrary_types_allowed") is True

    def test_to_dict_method(self, mock_torch_module, sample_metadata):
        """Test to_dict() method returns dictionary (backward compatibility).

        Phase 24: to_dict() wraps Pydantic's model_dump() for backward compatibility.
        """
        timestamp = datetime.now().isoformat()
        registration = ModelRegistration(
            name="TestModel",
            model_class=mock_torch_module,
            metadata=sample_metadata,
            is_builtin=False,
            plugin_name="test_plugin",
            registered_at=timestamp,
        )

        result = registration.to_dict()

        assert isinstance(result, dict)
        assert result["name"] == "TestModel"
        assert result["is_builtin"] is False
        assert result["plugin_name"] == "test_plugin"
        assert result["registered_at"] == timestamp

    def test_to_dict_equals_model_dump(self, mock_torch_module, sample_metadata):
        """Test to_dict() is equivalent to model_dump()."""
        registration = ModelRegistration(
            name="TestModel", model_class=mock_torch_module, metadata=sample_metadata
        )

        assert registration.to_dict() == registration.model_dump()

    def test_model_is_mutable(self, mock_torch_module, sample_metadata):
        """Test that ModelRegistration is mutable (not frozen).

        Phase 24: Pydantic model is mutable by default, unlike frozen dataclass.
        """
        registration = ModelRegistration(
            name="OriginalName", model_class=mock_torch_module, metadata=sample_metadata
        )

        # Should be able to modify fields
        registration.name = "NewName"
        assert registration.name == "NewName"

    def test_model_validation_on_invalid_types(self, sample_metadata):
        """Test Pydantic validation rejects invalid types."""
        # Pydantic should reject non-Type for model_class
        with pytest.raises((TypeError, ValueError)):
            ModelRegistration(
                name="TestModel",
                model_class="not_a_class",  # Invalid type
                metadata=sample_metadata,
            )


# =============================================================================
# SINGLETON PATTERN TESTS
# =============================================================================


class TestModelRegistrySingleton:
    """Test ModelRegistry singleton pattern."""

    def test_singleton_pattern(self):
        """Test that ModelRegistry follows singleton pattern."""
        reg1 = ModelRegistry()
        reg2 = ModelRegistry()
        assert reg1 is reg2

    def test_get_instance_returns_singleton(self):
        """Test get_instance() returns the same singleton."""
        reg1 = ModelRegistry.get_instance()
        reg2 = ModelRegistry.get_instance()
        assert reg1 is reg2

    def test_singleton_thread_safety(self):
        """Test singleton creation is thread-safe."""
        instances = []

        def create_instance():
            instances.append(ModelRegistry())

        threads = [threading.Thread(target=create_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All instances should be the same object
        assert all(inst is instances[0] for inst in instances)

    def test_initialized_flag(self):
        """Test _initialized flag is set after first initialization."""
        reg = ModelRegistry()
        assert hasattr(reg, "_initialized")
        assert reg._initialized is True

    def test_multiple_calls_dont_reinitialize(self):
        """Test multiple instantiations don't reinitialize."""
        reg1 = ModelRegistry()
        initial_models_count = len(reg1)

        reg2 = ModelRegistry()
        # Should be same instance with same state
        assert len(reg2) == initial_models_count


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestModelRegistryInitialization:
    """Test ModelRegistry initialization."""

    def test_internal_structures_initialized(self, isolated_registry):
        """Test internal data structures are initialized."""
        assert hasattr(isolated_registry, "_models")
        assert hasattr(isolated_registry, "_by_category")
        assert hasattr(isolated_registry, "_plugin_models")
        assert hasattr(isolated_registry, "_lock")
        assert hasattr(isolated_registry, "_failed_models")
        assert hasattr(isolated_registry, "_discovery_stats")

    def test_lock_is_rlock(self, isolated_registry):
        """Test that _lock is an RLock for reentrant locking."""
        # RLock is a function that returns an instance, check the type name
        assert type(isolated_registry._lock).__name__ == "RLock"

    def test_models_dict_initialized(self, isolated_registry):
        """Test _models dict is initialized."""
        assert isinstance(isolated_registry._models, dict)

    def test_by_category_initialized(self, isolated_registry):
        """Test _by_category is initialized as defaultdict."""
        from collections import defaultdict

        assert isinstance(isolated_registry._by_category, defaultdict)

    def test_plugin_models_initialized(self, isolated_registry):
        """Test _plugin_models dict is initialized."""
        assert isinstance(isolated_registry._plugin_models, dict)

    def test_discovery_stats_structure(self, isolated_registry):
        """Test _discovery_stats has correct structure."""
        stats = isolated_registry._discovery_stats
        assert "total_attempted" in stats
        assert "successful" in stats
        assert "failed" in stats
        assert "last_discovery" in stats


# =============================================================================
# AUTO-DISCOVERY TESTS
# =============================================================================


class TestAutoDiscovery:
    """Test auto-discovery functionality."""

    def test_auto_discover_with_mock_models(self, isolated_registry):
        """Test auto_discover_pyg_models with mocked models.

        Note: Updated for Phase 3 migration - now mocks get_introspector()
        instead of module-level functions, since auto_discover_pyg_models
        uses the introspector singleton directly.
        """
        # Get the already imported module from sys.modules
        registry_module = sys.modules["milia_pipeline.models.registry.model_registry"]

        # Create mock metadata
        mock_metadata = ModelMetadata(
            name="MockModel1",
            category=ModelCategory.BASIC_GNN,
            import_path="torch.nn.Linear",
            description="Test",
        )

        # Create mock introspector
        mock_introspector = MagicMock()
        mock_introspector.get_all_model_names.return_value = ["MockModel1", "MockModel2"]
        mock_introspector.get_model_metadata.return_value = mock_metadata

        # Reset discovery flag to allow re-discovery
        isolated_registry._auto_discovered = False
        isolated_registry._discovery_stats = {"total_attempted": 0, "successful": 0, "failed": 0}

        # Setup mocks - mock get_introspector to return our mock
        with patch.object(registry_module, "get_introspector", return_value=mock_introspector):
            # Run discovery
            count = isolated_registry.auto_discover_pyg_models()

            # Verify results
            assert count >= 0
            assert isolated_registry._discovery_stats["total_attempted"] == 2

    def test_auto_discover_returns_count(self, fresh_registry):
        """Test auto_discover_pyg_models returns integer count."""
        count = fresh_registry.auto_discover_pyg_models()
        assert isinstance(count, int)
        assert count >= 0

    def test_discovery_stats_updated(self, fresh_registry):
        """Test that discovery stats are updated after discovery.

        Note: Models with None metadata are skipped and added to failed_models
        with reason 'introspection_failed'. So successful + failed should equal
        total_attempted.
        """
        fresh_registry.auto_discover_pyg_models()

        stats = fresh_registry._discovery_stats
        assert stats["total_attempted"] >= 0
        assert stats["successful"] >= 0
        assert stats["failed"] >= 0
        assert stats["last_discovery"] is not None
        # successful + failed should equal total_attempted
        # (skipped models are counted as failed with 'introspection_failed' reason)
        assert stats["successful"] + stats["failed"] == stats["total_attempted"]

    def test_discovery_last_timestamp(self, fresh_registry):
        """Test last_discovery timestamp is set."""
        fresh_registry.auto_discover_pyg_models()

        timestamp = fresh_registry._discovery_stats["last_discovery"]
        assert timestamp is not None
        # Should be valid ISO format
        datetime.fromisoformat(timestamp)

    def test_auto_discovered_flag_prevents_redundant_discovery(self, isolated_registry):
        """Test that _auto_discovered flag prevents redundant discovery.

        Phase 3 feature: auto_discover_pyg_models checks _auto_discovered
        and returns early if already run, avoiding redundant introspection.
        """
        registry_module = sys.modules["milia_pipeline.models.registry.model_registry"]

        # Create mock introspector
        mock_introspector = MagicMock()
        mock_introspector.get_all_model_names.return_value = ["Model1"]
        mock_metadata = ModelMetadata(
            name="Model1",
            category=ModelCategory.BASIC_GNN,
            import_path="torch.nn.Linear",
            description="Test",
        )
        mock_introspector.get_model_metadata.return_value = mock_metadata

        # First discovery
        isolated_registry._auto_discovered = False
        with patch.object(registry_module, "get_introspector", return_value=mock_introspector):
            count1 = isolated_registry.auto_discover_pyg_models()

        assert isolated_registry._auto_discovered is True
        first_call_count = mock_introspector.get_all_model_names.call_count

        # Second discovery should return early
        with patch.object(registry_module, "get_introspector", return_value=mock_introspector):
            count2 = isolated_registry.auto_discover_pyg_models()

        # get_all_model_names should NOT have been called again
        assert mock_introspector.get_all_model_names.call_count == first_call_count
        # Should return the cached successful count
        assert count2 == isolated_registry._discovery_stats.get("successful", 0)

    def test_reset_clears_auto_discovered_flag(self, isolated_registry):
        """Test that reset() clears _auto_discovered flag to allow re-discovery."""
        isolated_registry._auto_discovered = True

        isolated_registry.reset()

        assert isolated_registry._auto_discovered is False

    def test_discovery_handles_none_metadata(self, isolated_registry):
        """Test discovery gracefully handles None metadata.

        Note: auto_discover_pyg_models uses introspector.get_model_metadata()
        (instance method), not the module-level get_model_metadata function.
        """
        registry_module = sys.modules["milia_pipeline.models.registry.model_registry"]

        # Create mock introspector that returns None for metadata
        mock_introspector = MagicMock()
        mock_introspector.get_all_model_names.return_value = ["Model1", "Model2"]
        mock_introspector.get_model_metadata.return_value = None

        # Reset discovery flag to allow re-discovery
        isolated_registry._auto_discovered = False
        isolated_registry._discovery_stats = {
            "total_attempted": 0,
            "successful": 0,
            "failed": 0,
            "last_discovery": None,
        }

        with patch.object(registry_module, "get_introspector", return_value=mock_introspector):
            count = isolated_registry.auto_discover_pyg_models()
            assert isinstance(count, int)
            # Since all metadata returns None, no models should be registered
            assert count == 0


# =============================================================================
# IMPORT TESTS
# =============================================================================


class TestModelImport:
    """Test _import_pyg_model method."""

    def test_import_valid_model(self, isolated_registry):
        """Test importing a valid PyTorch model."""
        metadata = ModelMetadata(
            name="Linear",
            category=ModelCategory.UTILITY,
            import_path="torch.nn.Linear",
            description="Linear layer",
        )

        model_class = isolated_registry._import_pyg_model("Linear", metadata)
        assert model_class is torch.nn.Linear

    def test_import_invalid_path(self, isolated_registry):
        """Test importing with invalid path returns None."""
        metadata = ModelMetadata(
            name="Invalid",
            category=ModelCategory.UTILITY,
            import_path="invalid.module.path",
            description="Invalid",
        )

        model_class = isolated_registry._import_pyg_model("Invalid", metadata)
        assert model_class is None

    def test_import_malformed_path(self, isolated_registry):
        """Test importing with malformed path returns None."""
        metadata = ModelMetadata(
            name="Malformed",
            category=ModelCategory.UTILITY,
            import_path="justonepart",
            description="Malformed",
        )

        model_class = isolated_registry._import_pyg_model("Malformed", metadata)
        assert model_class is None

    def test_import_nonexistent_class(self, isolated_registry):
        """Test importing non-existent class returns None."""
        metadata = ModelMetadata(
            name="NonExistent",
            category=ModelCategory.UTILITY,
            import_path="torch.nn.NonExistentClass",
            description="Non-existent",
        )

        model_class = isolated_registry._import_pyg_model("NonExistent", metadata)
        assert model_class is None

    def test_import_non_module_class(self, isolated_registry):
        """Test importing non-Module class returns None."""
        metadata = ModelMetadata(
            name="Dict",
            category=ModelCategory.UTILITY,
            import_path="builtins.dict",
            description="Not a module",
        )

        model_class = isolated_registry._import_pyg_model("Dict", metadata)
        assert model_class is None


# =============================================================================
# REGISTRATION TESTS
# =============================================================================


class TestModelRegistration:
    """Test model registration functionality."""

    def test_register_custom_model(self, fresh_registry, mock_torch_module, sample_metadata):
        """Test registering a custom model."""
        result = fresh_registry.register_model("CustomModel", mock_torch_module, sample_metadata)
        assert result is True
        assert fresh_registry.has_model("CustomModel")

    def test_register_with_plugin_name(self, fresh_registry, mock_torch_module, sample_metadata):
        """Test registering model with plugin name."""
        result = fresh_registry.register_model(
            "PluginModel", mock_torch_module, sample_metadata, plugin_name="test_plugin"
        )
        assert result is True
        assert "PluginModel" in fresh_registry._plugin_models
        assert fresh_registry._plugin_models["PluginModel"] == "test_plugin"

    def test_register_duplicate_without_force(
        self, fresh_registry, mock_torch_module, sample_metadata
    ):
        """Test registering duplicate model without force returns False."""
        fresh_registry.register_model("Duplicate", mock_torch_module, sample_metadata)
        result = fresh_registry.register_model("Duplicate", mock_torch_module, sample_metadata)
        assert result is False

    def test_register_duplicate_with_force(
        self, fresh_registry, mock_torch_module, sample_metadata
    ):
        """Test registering duplicate model with force=True succeeds."""
        fresh_registry.register_model("Duplicate", mock_torch_module, sample_metadata)
        result = fresh_registry.register_model(
            "Duplicate", mock_torch_module, sample_metadata, force=True
        )
        assert result is True

    def test_register_non_module_raises_error(self, fresh_registry, sample_metadata):
        """Test registering non-Module class raises ModelError."""

        class NotAModule:
            pass

        with pytest.raises(Exception):  # Should raise ModelError
            fresh_registry.register_model("NotModule", NotAModule, sample_metadata)

    def test_register_updates_category(self, fresh_registry, mock_torch_module, sample_metadata):
        """Test registration updates _by_category."""
        fresh_registry.register_model("TestModel", mock_torch_module, sample_metadata)

        category = sample_metadata.category
        assert "TestModel" in fresh_registry._by_category[category]

    def test_register_sets_timestamp(self, fresh_registry, mock_torch_module, sample_metadata):
        """Test registration sets registered_at timestamp."""
        fresh_registry.register_model("TimedModel", mock_torch_module, sample_metadata)

        registration = fresh_registry.get_registration("TimedModel")
        assert registration.registered_at is not None
        # Verify it's a valid ISO timestamp
        datetime.fromisoformat(registration.registered_at)


# =============================================================================
# UNREGISTRATION TESTS
# =============================================================================


class TestModelUnregistration:
    """Test model unregistration functionality."""

    def test_unregister_existing_model(self, fresh_registry, mock_torch_module, sample_metadata):
        """Test unregistering an existing model."""
        fresh_registry.register_model("ToRemove", mock_torch_module, sample_metadata)
        result = fresh_registry.unregister_model("ToRemove")

        assert result is True
        assert not fresh_registry.has_model("ToRemove")

    def test_unregister_nonexistent_model(self, fresh_registry):
        """Test unregistering non-existent model returns False."""
        result = fresh_registry.unregister_model("NonExistent")
        assert result is False

    def test_unregister_removes_from_category(
        self, fresh_registry, mock_torch_module, sample_metadata
    ):
        """Test unregistration removes from category."""
        fresh_registry.register_model("CategoryTest", mock_torch_module, sample_metadata)
        category = sample_metadata.category

        fresh_registry.unregister_model("CategoryTest")
        assert "CategoryTest" not in fresh_registry._by_category[category]

    def test_unregister_removes_plugin_association(
        self, fresh_registry, mock_torch_module, sample_metadata
    ):
        """Test unregistration removes plugin association."""
        fresh_registry.register_model(
            "PluginModel", mock_torch_module, sample_metadata, plugin_name="test_plugin"
        )

        fresh_registry.unregister_model("PluginModel")
        assert "PluginModel" not in fresh_registry._plugin_models


# =============================================================================
# QUERY TESTS
# =============================================================================


class TestQueryMethods:
    """Test query methods."""

    def test_get_model(self, fresh_registry, mock_torch_module, sample_metadata):
        """Test get_model returns correct model class."""
        fresh_registry.register_model("QueryTest", mock_torch_module, sample_metadata)

        result = fresh_registry.get_model("QueryTest")
        assert result == mock_torch_module

    def test_get_model_nonexistent(self, fresh_registry):
        """Test get_model returns None for non-existent model."""
        result = fresh_registry.get_model("NonExistent")
        assert result is None

    def test_has_model_true(self, fresh_registry, mock_torch_module, sample_metadata):
        """Test has_model returns True for existing model."""
        fresh_registry.register_model("ExistsTest", mock_torch_module, sample_metadata)
        assert fresh_registry.has_model("ExistsTest") is True

    def test_has_model_false(self, fresh_registry):
        """Test has_model returns False for non-existent model."""
        assert fresh_registry.has_model("NonExistent") is False

    def test_get_registration(self, fresh_registry, mock_torch_module, sample_metadata):
        """Test get_registration returns correct registration."""
        fresh_registry.register_model("RegTest", mock_torch_module, sample_metadata)

        reg = fresh_registry.get_registration("RegTest")
        assert isinstance(reg, ModelRegistration)
        assert reg.name == "RegTest"
        assert reg.model_class == mock_torch_module

    def test_get_registration_nonexistent(self, fresh_registry):
        """Test get_registration returns None for non-existent model."""
        result = fresh_registry.get_registration("NonExistent")
        assert result is None

    def test_get_metadata(self, fresh_registry, mock_torch_module, sample_metadata):
        """Test get_metadata returns correct metadata."""
        fresh_registry.register_model("MetaTest", mock_torch_module, sample_metadata)

        meta = fresh_registry.get_metadata("MetaTest")
        assert isinstance(meta, ModelMetadata)
        assert meta.name == sample_metadata.name

    def test_get_metadata_nonexistent(self, fresh_registry):
        """Test get_metadata falls back to introspection for unregistered model.

        Note: get_metadata() falls back to dynamic introspection via
        get_introspector().get_model_metadata(name) for unregistered models.
        The result depends on whether the introspector can find the model.
        """
        registry_module = sys.modules["milia_pipeline.models.registry.model_registry"]

        # Mock introspector to return None for truly non-existent model
        mock_introspector = MagicMock()
        mock_introspector.get_model_metadata.return_value = None

        with patch.object(registry_module, "get_introspector", return_value=mock_introspector):
            result = fresh_registry.get_metadata("TrulyNonExistentModel")
            assert result is None

    def test_get_metadata_fallback_to_introspection(self, fresh_registry):
        """Test get_metadata falls back to dynamic introspection for unregistered models."""
        registry_module = sys.modules["milia_pipeline.models.registry.model_registry"]

        # Create mock metadata that introspector returns
        mock_metadata = ModelMetadata(
            name="DynamicModel",
            category=ModelCategory.BASIC_GNN,
            import_path="test.DynamicModel",
            description="Model found via introspection",
        )

        mock_introspector = MagicMock()
        mock_introspector.get_model_metadata.return_value = mock_metadata

        with patch.object(registry_module, "get_introspector", return_value=mock_introspector):
            # Model is NOT registered in registry
            assert not fresh_registry.has_model("DynamicModel")

            # But get_metadata should still return it via introspection fallback
            result = fresh_registry.get_metadata("DynamicModel")
            assert result is not None
            assert result.name == "DynamicModel"


# =============================================================================
# LIST AND FILTER TESTS
# =============================================================================


class TestListAndFilter:
    """Test listing and filtering methods."""

    def test_list_available_models_all(self, fresh_registry):
        """Test listing all available models."""
        models = fresh_registry.list_available_models()
        assert isinstance(models, list)
        assert len(models) >= 0
        # Should be sorted
        assert models == sorted(models)

    def test_list_available_models_by_category(self, fresh_registry, mock_torch_module):
        """Test listing models filtered by category."""
        metadata1 = ModelMetadata(
            name="Model1",
            category=ModelCategory.BASIC_GNN,
            import_path="test.Model1",
            description="Test",
        )
        metadata2 = ModelMetadata(
            name="Model2",
            category=ModelCategory.ATTENTION,
            import_path="test.Model2",
            description="Test",
        )

        fresh_registry.register_model("Model1", mock_torch_module, metadata1)
        fresh_registry.register_model("Model2", mock_torch_module, metadata2)

        basic_models = fresh_registry.list_available_models(category=ModelCategory.BASIC_GNN)
        assert "Model1" in basic_models
        assert "Model2" not in basic_models

    def test_list_available_models_by_task(self, fresh_registry, mock_torch_module):
        """Test listing models filtered by task type."""
        metadata = ModelMetadata(
            name="TaskModel",
            category=ModelCategory.BASIC_GNN,
            import_path="test.TaskModel",
            description="Test",
            supported_tasks=["node_classification", "graph_regression"],
        )

        fresh_registry.register_model("TaskModel", mock_torch_module, metadata)

        node_models = fresh_registry.list_available_models(task_type="node_classification")
        assert "TaskModel" in node_models

        link_models = fresh_registry.list_available_models(task_type="link_prediction")
        assert "TaskModel" not in link_models

    def test_list_available_models_by_heterogeneous(self, fresh_registry, mock_torch_module):
        """Test listing models filtered by heterogeneous support."""
        metadata_hetero = ModelMetadata(
            name="HeteroModel",
            category=ModelCategory.BASIC_GNN,
            import_path="test.HeteroModel",
            description="Test",
            supports_heterogeneous=True,
        )
        metadata_homo = ModelMetadata(
            name="HomoModel",
            category=ModelCategory.BASIC_GNN,
            import_path="test.HomoModel",
            description="Test",
            supports_heterogeneous=False,
        )

        fresh_registry.register_model("HeteroModel", mock_torch_module, metadata_hetero)
        fresh_registry.register_model("HomoModel", mock_torch_module, metadata_homo)

        hetero_models = fresh_registry.list_available_models(supports_heterogeneous=True)
        assert "HeteroModel" in hetero_models
        assert "HomoModel" not in hetero_models

    def test_list_available_models_by_tags(self, fresh_registry, mock_torch_module):
        """Test listing models filtered by tags."""
        metadata = ModelMetadata(
            name="TaggedModel",
            category=ModelCategory.BASIC_GNN,
            import_path="test.TaggedModel",
            description="Test",
            tags=["attention", "spectral"],
        )

        fresh_registry.register_model("TaggedModel", mock_torch_module, metadata)

        # Model must have ALL specified tags
        attention_models = fresh_registry.list_available_models(tags=["attention"])
        assert "TaggedModel" in attention_models

        both_tags = fresh_registry.list_available_models(tags=["attention", "spectral"])
        assert "TaggedModel" in both_tags

        missing_tag = fresh_registry.list_available_models(tags=["missing"])
        assert "TaggedModel" not in missing_tag

    def test_list_by_category(self, fresh_registry, mock_torch_module):
        """Test list_by_category returns dict of categories."""
        metadata = ModelMetadata(
            name="CatModel",
            category=ModelCategory.BASIC_GNN,
            import_path="test.CatModel",
            description="Test",
        )

        fresh_registry.register_model("CatModel", mock_torch_module, metadata)

        by_cat = fresh_registry.list_by_category()
        assert isinstance(by_cat, dict)
        assert "basic_gnn" in by_cat
        assert "CatModel" in by_cat["basic_gnn"]


# =============================================================================
# SEARCH TESTS
# =============================================================================


class TestSearchMethods:
    """Test search functionality."""

    def test_search_models_by_name(self, fresh_registry, mock_torch_module):
        """Test searching models by name."""
        metadata = ModelMetadata(
            name="AttentionModel",
            category=ModelCategory.ATTENTION,
            import_path="test.AttentionModel",
            description="A model with attention mechanism",
        )

        fresh_registry.register_model("AttentionModel", mock_torch_module, metadata)

        results = fresh_registry.search_models("attention")
        assert "AttentionModel" in results

    def test_search_models_case_insensitive(self, fresh_registry, mock_torch_module):
        """Test search is case-insensitive."""
        metadata = ModelMetadata(
            name="TestModel",
            category=ModelCategory.BASIC_GNN,
            import_path="test.TestModel",
            description="Test",
        )

        fresh_registry.register_model("TestModel", mock_torch_module, metadata)

        results = fresh_registry.search_models("TESTMODEL")
        assert "TestModel" in results

    def test_search_models_by_description(self, fresh_registry, mock_torch_module):
        """Test searching models by description."""
        metadata = ModelMetadata(
            name="SpecialModel",
            category=ModelCategory.BASIC_GNN,
            import_path="test.SpecialModel",
            description="A special temporal graph model",
        )

        fresh_registry.register_model("SpecialModel", mock_torch_module, metadata)

        results = fresh_registry.search_models("temporal")
        assert "SpecialModel" in results

    def test_search_models_by_tags(self, fresh_registry, mock_torch_module):
        """Test searching models by tags."""
        metadata = ModelMetadata(
            name="TaggedModel",
            category=ModelCategory.BASIC_GNN,
            import_path="test.TaggedModel",
            description="Test",
            tags=["spectral", "convolutional"],
        )

        fresh_registry.register_model("TaggedModel", mock_torch_module, metadata)

        results = fresh_registry.search_models("spectral")
        assert "TaggedModel" in results

    def test_search_models_with_search_in(self, fresh_registry, mock_torch_module):
        """Test search with specific fields."""
        metadata = ModelMetadata(
            name="NameMatch",
            category=ModelCategory.BASIC_GNN,
            import_path="test.NameMatch",
            description="Different description",
        )

        fresh_registry.register_model("NameMatch", mock_torch_module, metadata)

        # Search only in name
        results = fresh_registry.search_models("name", search_in=["name"])
        assert "NameMatch" in results

        # Search only in description should not find it
        results = fresh_registry.search_models("name", search_in=["description"])
        assert "NameMatch" not in results

    def test_search_models_returns_sorted(self, fresh_registry, mock_torch_module):
        """Test search results are sorted."""
        for i in range(5):
            metadata = ModelMetadata(
                name=f"Model{i}",
                category=ModelCategory.BASIC_GNN,
                import_path=f"test.Model{i}",
                description="model description",
            )
            fresh_registry.register_model(f"Model{i}", mock_torch_module, metadata)

        results = fresh_registry.search_models("model")
        assert results == sorted(results)


# =============================================================================
# PLUGIN MANAGEMENT TESTS
# =============================================================================


class TestPluginManagement:
    """Test plugin management functionality."""

    def test_list_plugin_models(self, fresh_registry, mock_torch_module, sample_metadata):
        """Test listing plugin models."""
        fresh_registry.register_model(
            "Plugin1", mock_torch_module, sample_metadata, plugin_name="plugin_a"
        )
        fresh_registry.register_model(
            "Plugin2", mock_torch_module, sample_metadata, plugin_name="plugin_a"
        )
        fresh_registry.register_model(
            "Plugin3", mock_torch_module, sample_metadata, plugin_name="plugin_b"
        )

        plugin_models = fresh_registry.list_plugin_models()
        assert isinstance(plugin_models, dict)
        assert "plugin_a" in plugin_models
        assert "plugin_b" in plugin_models
        assert len(plugin_models["plugin_a"]) == 2
        assert len(plugin_models["plugin_b"]) == 1

    def test_get_builtin_models(self, fresh_registry, mock_torch_module):
        """Test getting list of built-in models."""
        metadata = ModelMetadata(
            name="BuiltinModel",
            category=ModelCategory.BASIC_GNN,
            import_path="test.BuiltinModel",
            description="Test",
        )

        # Manually register as builtin
        fresh_registry._register_internal(
            "BuiltinModel", mock_torch_module, metadata, is_builtin=True
        )

        builtins = fresh_registry.get_builtin_models()
        assert "BuiltinModel" in builtins
        assert builtins == sorted(builtins)

    def test_get_custom_models(self, fresh_registry, mock_torch_module, sample_metadata):
        """Test getting list of custom models."""
        fresh_registry.register_model("CustomModel", mock_torch_module, sample_metadata)

        customs = fresh_registry.get_custom_models()
        assert "CustomModel" in customs
        assert customs == sorted(customs)

    def test_builtin_and_custom_disjoint(self, fresh_registry, mock_torch_module, sample_metadata):
        """Test builtin and custom models are disjoint sets."""
        metadata = ModelMetadata(
            name="BuiltinModel",
            category=ModelCategory.BASIC_GNN,
            import_path="test.BuiltinModel",
            description="Test",
        )

        fresh_registry._register_internal(
            "BuiltinModel", mock_torch_module, metadata, is_builtin=True
        )
        fresh_registry.register_model("CustomModel", mock_torch_module, sample_metadata)

        builtins = set(fresh_registry.get_builtin_models())
        customs = set(fresh_registry.get_custom_models())

        assert len(builtins & customs) == 0


# =============================================================================
# STATISTICS AND REPORTING TESTS
# =============================================================================


class TestStatisticsAndReporting:
    """Test statistics and reporting methods."""

    def test_get_statistics(self, fresh_registry, mock_torch_module, sample_metadata):
        """Test get_statistics returns correct structure."""
        fresh_registry.register_model("StatModel", mock_torch_module, sample_metadata)

        stats = fresh_registry.get_statistics()
        assert isinstance(stats, dict)
        assert "total_models" in stats
        assert "builtin_models" in stats
        assert "plugin_models" in stats
        assert "by_category" in stats
        assert "plugins" in stats
        assert "discovery_stats" in stats
        assert "failed_models_count" in stats

    def test_get_statistics_counts(self, fresh_registry, mock_torch_module, sample_metadata):
        """Test statistics counts are correct."""
        fresh_registry.register_model("Model1", mock_torch_module, sample_metadata)

        stats = fresh_registry.get_statistics()
        assert stats["total_models"] >= 1
        assert stats["builtin_models"] + stats["plugin_models"] == stats["total_models"]

    def test_get_availability_report(self, fresh_registry):
        """Test get_availability_report structure."""
        report = fresh_registry.get_availability_report()

        assert isinstance(report, dict)
        assert "total_registered" in report
        assert "total_attempted" in report
        assert "failed_models" in report
        assert "by_category" in report
        assert "discovery_stats" in report
        assert "success_rate" in report

    def test_get_availability_report_success_rate(self, fresh_registry):
        """Test success rate calculation."""
        report = fresh_registry.get_availability_report()

        if report["total_attempted"] > 0:
            expected_rate = report["total_registered"] / report["total_attempted"] * 100
            assert abs(report["success_rate"] - expected_rate) < 0.01
        else:
            assert report["success_rate"] == 0

    def test_log_availability_summary(self, fresh_registry):
        """Test log_availability_summary runs without error."""
        # Should not raise any exceptions
        fresh_registry.log_availability_summary()


# =============================================================================
# UTILITY METHOD TESTS
# =============================================================================


class TestUtilityMethods:
    """Test utility methods."""

    def test_reset(self, fresh_registry, mock_torch_module, sample_metadata):
        """Test reset clears all data."""
        fresh_registry.register_model("ResetTest", mock_torch_module, sample_metadata)

        fresh_registry.reset()

        assert len(fresh_registry._models) == 0
        assert len(fresh_registry._by_category) == 0
        assert len(fresh_registry._plugin_models) == 0
        assert len(fresh_registry._failed_models) == 0
        assert fresh_registry._discovery_stats["total_attempted"] == 0

    def test_len(self, fresh_registry, mock_torch_module, sample_metadata):
        """Test __len__ returns correct count."""
        initial_len = len(fresh_registry)
        fresh_registry.register_model("LenTest", mock_torch_module, sample_metadata)
        assert len(fresh_registry) == initial_len + 1

    def test_contains(self, fresh_registry, mock_torch_module, sample_metadata):
        """Test __contains__ works correctly."""
        fresh_registry.register_model("ContainsTest", mock_torch_module, sample_metadata)

        assert "ContainsTest" in fresh_registry
        assert "NonExistent" not in fresh_registry

    def test_repr(self, fresh_registry):
        """Test __repr__ returns string representation."""
        repr_str = repr(fresh_registry)

        assert isinstance(repr_str, str)
        assert "ModelRegistry" in repr_str
        assert "total=" in repr_str
        assert "builtin=" in repr_str
        assert "plugin=" in repr_str


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================


class TestThreadSafety:
    """Test thread safety of operations."""

    def test_concurrent_registration(self, fresh_registry, mock_torch_module):
        """Test concurrent model registration is thread-safe."""
        results = []

        def register_model(i):
            metadata = ModelMetadata(
                name=f"ConcurrentModel{i}",
                category=ModelCategory.BASIC_GNN,
                import_path=f"test.Model{i}",
                description="Test",
            )
            result = fresh_registry.register_model(
                f"ConcurrentModel{i}", mock_torch_module, metadata
            )
            results.append(result)

        threads = [threading.Thread(target=register_model, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All registrations should succeed
        assert all(results)
        assert len(fresh_registry._models) >= 10

    def test_concurrent_queries(self, fresh_registry, mock_torch_module, sample_metadata):
        """Test concurrent queries are thread-safe."""
        fresh_registry.register_model("QueryTest", mock_torch_module, sample_metadata)

        results = []

        def query_model():
            model_class = fresh_registry.get_model("QueryTest")
            results.append(model_class)

        threads = [threading.Thread(target=query_model) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All queries should return the same class
        assert all(r == mock_torch_module for r in results)

    def test_concurrent_list_operations(self, fresh_registry):
        """Test concurrent list operations are thread-safe."""
        results = []

        def list_models():
            models = fresh_registry.list_available_models()
            results.append(len(models))

        threads = [threading.Thread(target=list_models) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results should be consistent
        assert len(set(results)) <= 2  # Allow for slight timing differences


# =============================================================================
# GLOBAL INSTANCE AND CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestGlobalInstanceAndConvenienceFunctions:
    """Test global registry instance and convenience functions."""

    def test_global_registry_exists(self):
        """Test global registry instance exists."""
        from milia_pipeline.models.registry.model_registry import registry as global_reg

        assert global_reg is not None
        assert isinstance(global_reg, ModelRegistry)

    def test_global_registry_is_singleton(self):
        """Test global registry is the singleton instance."""
        from milia_pipeline.models.registry.model_registry import registry as global_reg

        assert global_reg is ModelRegistry.get_instance()

    def test_get_model_convenience_function(self, mock_torch_module, sample_metadata):
        """Test get_model convenience function."""
        registry.register_model("ConvFunc1", mock_torch_module, sample_metadata)

        model_class = get_model("ConvFunc1")
        assert model_class == mock_torch_module

        # Cleanup
        registry.unregister_model("ConvFunc1")

    def test_has_model_convenience_function(self, mock_torch_module, sample_metadata):
        """Test has_model convenience function."""
        registry.register_model("ConvFunc2", mock_torch_module, sample_metadata)

        assert has_model("ConvFunc2") is True
        assert has_model("NonExistent") is False

        # Cleanup
        registry.unregister_model("ConvFunc2")

    def test_list_models_convenience_function(self):
        """Test list_models convenience function."""
        models = list_models()
        assert isinstance(models, list)

    def test_list_models_with_filters(self):
        """Test list_models with category filter."""
        models = list_models(category=ModelCategory.BASIC_GNN)
        assert isinstance(models, list)

    def test_get_model_info_convenience_function(self, mock_torch_module, sample_metadata):
        """Test get_model_info convenience function."""
        registry.register_model("ConvFunc3", mock_torch_module, sample_metadata)

        info = get_model_info("ConvFunc3")
        assert isinstance(info, dict)
        assert "name" in info
        assert "class" in info
        assert "description" in info
        assert "category" in info
        assert info["name"] == "ConvFunc3"

        # Cleanup
        registry.unregister_model("ConvFunc3")

    def test_get_model_info_nonexistent(self):
        """Test get_model_info returns None for non-existent model."""
        info = get_model_info("NonExistent")
        assert info is None

    def test_get_model_info_structure(self, mock_torch_module, sample_metadata):
        """Test get_model_info returns complete structure."""
        registry.register_model("InfoTest", mock_torch_module, sample_metadata)

        info = get_model_info("InfoTest")
        expected_keys = {
            "name",
            "class",
            "description",
            "category",
            "supported_tasks",
            "is_builtin",
            "plugin_name",
            "paper_url",
            "tags",
            "requires_edge_features",
            "requires_edge_weights",
            "supports_heterogeneous",
            "registered_at",
        }
        assert set(info.keys()) == expected_keys

        # Cleanup
        registry.unregister_model("InfoTest")


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_registry_operations(self, isolated_registry):
        """Test operations on empty registry."""
        assert len(isolated_registry) == 0
        assert isolated_registry.list_available_models() == []
        assert isolated_registry.get_model("Any") is None
        assert isolated_registry.has_model("Any") is False

    def test_register_with_empty_name(self, fresh_registry, mock_torch_module, sample_metadata):
        """Test registering with empty name."""
        # Should still work, though not recommended
        result = fresh_registry.register_model("", mock_torch_module, sample_metadata)
        assert result is True
        assert fresh_registry.has_model("")

    def test_search_with_empty_query(self, fresh_registry, mock_torch_module, sample_metadata):
        """Test search with empty query."""
        fresh_registry.register_model("Test", mock_torch_module, sample_metadata)

        results = fresh_registry.search_models("")
        # Empty string matches everything (substring of all strings)
        assert "Test" in results

    def test_list_by_nonexistent_category(self, fresh_registry):
        """Test listing by non-existent category."""
        models = fresh_registry.list_available_models(category=ModelCategory.UTILITY)
        # Should return empty list if no models in category
        assert isinstance(models, list)

    def test_multiple_filters_combined(self, fresh_registry, mock_torch_module):
        """Test combining multiple filters."""
        metadata = ModelMetadata(
            name="MultiFilter",
            category=ModelCategory.ATTENTION,
            import_path="test.MultiFilter",
            description="Test",
            supported_tasks=["node_classification"],
            tags=["attention"],
            supports_heterogeneous=False,
        )

        fresh_registry.register_model("MultiFilter", mock_torch_module, metadata)

        # Apply all filters at once
        results = fresh_registry.list_available_models(
            category=ModelCategory.ATTENTION,
            task_type="node_classification",
            supports_heterogeneous=False,
            tags=["attention"],
        )
        assert "MultiFilter" in results

    def test_reset_and_reuse(self, fresh_registry, mock_torch_module, sample_metadata):
        """Test registry can be reset and reused."""
        fresh_registry.register_model("First", mock_torch_module, sample_metadata)
        assert fresh_registry.has_model("First")

        fresh_registry.reset()
        assert not fresh_registry.has_model("First")

        # Should be able to register again
        result = fresh_registry.register_model("Second", mock_torch_module, sample_metadata)
        assert result is True


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Test error handling."""

    def test_register_with_invalid_metadata_type(self, fresh_registry, mock_torch_module):
        """Test registration with invalid metadata type.

        Phase 24: With Pydantic V2, invalid metadata type raises ValidationError
        instead of TypeError/AttributeError.
        """
        from pydantic import ValidationError

        # Should handle gracefully or raise appropriate error
        # Pydantic V2 raises ValidationError for invalid field types
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            fresh_registry.register_model("Invalid", mock_torch_module, "not_a_metadata_object")

    def test_import_with_exception(self, isolated_registry):
        """Test _import_pyg_model handles exceptions gracefully."""
        metadata = ModelMetadata(
            name="ExceptionModel",
            category=ModelCategory.UTILITY,
            import_path="test.module.that.causes.exception",
            description="Test",
        )

        # Should return None instead of raising
        result = isolated_registry._import_pyg_model("ExceptionModel", metadata)
        assert result is None

    def test_register_internal_with_lock(self, fresh_registry, mock_torch_module, sample_metadata):
        """Test _register_internal uses lock correctly."""
        # Should not raise any threading errors
        fresh_registry._register_internal(
            "InternalTest", mock_torch_module, sample_metadata, is_builtin=False
        )
        assert fresh_registry.has_model("InternalTest")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_registration_workflow(
        self, fresh_registry, mock_torch_module, sample_metadata
    ):
        """Test complete registration and query workflow."""
        # Register
        result = fresh_registry.register_model(
            "Workflow1", mock_torch_module, sample_metadata, plugin_name="test_plugin"
        )
        assert result is True

        # Query
        assert fresh_registry.has_model("Workflow1")
        model_class = fresh_registry.get_model("Workflow1")
        assert model_class == mock_torch_module

        # Get details
        registration = fresh_registry.get_registration("Workflow1")
        assert registration.plugin_name == "test_plugin"

        # Search
        results = fresh_registry.search_models("workflow")
        assert "Workflow1" in results

        # Unregister
        unreg_result = fresh_registry.unregister_model("Workflow1")
        assert unreg_result is True
        assert not fresh_registry.has_model("Workflow1")

    def test_multiple_model_categories_workflow(self, fresh_registry, mock_torch_module):
        """Test workflow with models from different categories."""
        categories = [ModelCategory.BASIC_GNN, ModelCategory.ATTENTION, ModelCategory.TRANSFORMER]

        for i, category in enumerate(categories):
            metadata = ModelMetadata(
                name=f"Model{i}",
                category=category,
                import_path=f"test.Model{i}",
                description=f"Model {i}",
            )
            fresh_registry.register_model(f"Model{i}", mock_torch_module, metadata)

        # Check by category
        for i, category in enumerate(categories):
            models = fresh_registry.list_available_models(category=category)
            assert f"Model{i}" in models

        # Check all registered
        all_models = fresh_registry.list_available_models()
        for i in range(len(categories)):
            assert f"Model{i}" in all_models

    def test_plugin_workflow(self, fresh_registry, mock_torch_module):
        """Test complete plugin workflow."""
        # Register multiple plugin models
        for i in range(3):
            metadata = ModelMetadata(
                name=f"PluginModel{i}",
                category=ModelCategory.BASIC_GNN,
                import_path=f"test.PluginModel{i}",
                description="Plugin model",
            )
            fresh_registry.register_model(
                f"PluginModel{i}", mock_torch_module, metadata, plugin_name="test_plugin"
            )

        # List plugin models
        plugin_models = fresh_registry.list_plugin_models()
        assert "test_plugin" in plugin_models
        assert len(plugin_models["test_plugin"]) == 3

        # Get custom models
        customs = fresh_registry.get_custom_models()
        for i in range(3):
            assert f"PluginModel{i}" in customs


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestPerformance:
    """Test performance characteristics."""

    def test_registration_performance(self, fresh_registry, mock_torch_module):
        """Test registration performance."""

        start = time.time()
        for i in range(100):
            metadata = ModelMetadata(
                name=f"PerfModel{i}",
                category=ModelCategory.BASIC_GNN,
                import_path=f"test.PerfModel{i}",
                description="Test",
            )
            fresh_registry.register_model(f"PerfModel{i}", mock_torch_module, metadata)
        duration = time.time() - start

        # Should complete 100 registrations in under 1 second
        assert duration < 1.0

    def test_query_performance(self, fresh_registry, mock_torch_module, sample_metadata):
        """Test query performance."""

        fresh_registry.register_model("QueryPerf", mock_torch_module, sample_metadata)

        start = time.time()
        for _ in range(10000):
            fresh_registry.get_model("QueryPerf")
        duration = time.time() - start

        # Should complete 10000 queries in under 0.5 seconds
        assert duration < 0.5

    def test_search_performance(self, fresh_registry, mock_torch_module):
        """Test search performance."""

        # Register 100 models
        for i in range(100):
            metadata = ModelMetadata(
                name=f"SearchModel{i}",
                category=ModelCategory.BASIC_GNN,
                import_path=f"test.SearchModel{i}",
                description="Model for search testing",
            )
            fresh_registry.register_model(f"SearchModel{i}", mock_torch_module, metadata)

        start = time.time()
        for _ in range(100):
            fresh_registry.search_models("search")
        duration = time.time() - start

        # Should complete 100 searches in under 1 second
        assert duration < 1.0


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
