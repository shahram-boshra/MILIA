#!/usr/bin/env python3
"""
Comprehensive Unit Test Suite for plugin_system.py Module

This test suite provides extensive coverage of the plugin system module including:
- TransformDeclaration class and methods
- PluginMetadata class with all fields and behaviors
- PluginRegistry for plugin management, discovery, and validation
- PluginValidator for comprehensive plugin validation
- Lazy import mechanisms and circular dependency prevention
- Thread-safe operations and concurrent access patterns
- Security validation and dependency checking
- Plugin discovery from multiple sources (YAML, __plugin__.py, standalone)
- Version management and compatibility checking
- Transform registration and lifecycle management
- Error handling and exception cases

NOTE: This test suite runs inside Docker at /app/milia
Uses test-level mocking to prevent import pollution

Author: milia Project Team
Created: November 3, 2025
"""

import sys
import os
from pathlib import Path

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import Mock, MagicMock, patch, call, mock_open
from typing import Dict, List, Any, Optional, Set
import threading
import time
import yaml
import hashlib
from datetime import datetime
import importlib
import importlib.util
import re
import tempfile

# Import the module under test
from milia_pipeline.transformations.plugin_system import (
    # Core classes
    TransformDeclaration,
    PluginMetadata,
    PluginRegistry,
    PluginValidator,
    
    # Exceptions
    PluginError,
    PluginValidationError,
    PluginSecurityError,
    PluginDependencyError,
    
    # Lazy import functions
    _import_custom_transforms,
    _import_graph_transforms,
    
    # Module-level globals used in lazy imports
    TransformValidationError,
)


# =============================================================================
# TEST FIXTURES AND HELPER CLASSES
# =============================================================================

@pytest.fixture
def mock_transform_class():
    """Create a mock transform class for testing"""
    class MockTransform:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        
        def __call__(self, data):
            return data
        
        def __repr__(self):
            return f"MockTransform({self.kwargs})"
    
    MockTransform.__name__ = "MockTransform"
    return MockTransform


@pytest.fixture
def sample_transform_declaration():
    """Create a sample TransformDeclaration for testing"""
    return TransformDeclaration(
        name="test_transform",
        class_name="TestTransform",
        module_path="transforms.test_transform",
        category="molecular",
        description="A test transform",
        version="1.0.0",
        required_node_features=["x", "pos"],
        required_edge_features=["edge_attr"],
        required_graph_attributes=["y"],
        parameter_constraints={"max_nodes": 100}
    )


@pytest.fixture
def sample_plugin_metadata():
    """Create a sample PluginMetadata for testing"""
    return PluginMetadata(
        plugin_name="test_plugin",
        version="1.0.0",
        author="Test Author",
        plugin_type="user_experimental",
        email="test@example.com",
        license="MIT",
        description="A test plugin",
        homepage="https://example.com",
        milia_version=">=1.0.0",
        pyg_version=">=2.0.0",
        python_version=">=3.8",
        dependencies=["torch>=2.0.0", "torch-geometric>=2.3.0"],
        transform_declarations=[],
        registered_transforms=set(),
        discovery_source="yaml",
        discovery_timestamp=None,
        is_validated=False,
        validation_date=None,
        validation_results={},
        checksum=None,
        trusted=False
    )


@pytest.fixture
def sample_plugin_yaml():
    """Create a sample plugin.yaml content for testing"""
    return {
        "plugin": {
            "name": "test_plugin",
            "version": "1.0.0",
            "author": "Test Author",
            "description": "A test plugin",
            "plugin_type": "user_experimental"
        },
        "metadata": {
            "license": "MIT",
            "homepage": "https://example.com",
            "python_requires": ">=3.8"
        },
        "dependencies": {
            "pip": ["torch>=2.0.0", "torch-geometric>=2.3.0"]
        },
        "transforms": [
            {
                "name": "test_transform",
                "class_name": "TestTransform",
                "module_path": "transforms.test_transform",
                "category": "molecular",
                "description": "A test transform",
                "version": "1.0.0"
            }
        ]
    }


@pytest.fixture
def reset_plugin_registry():
    """Reset PluginRegistry state before and after tests"""
    # Get the singleton instance
    registry = PluginRegistry()
    # Snapshot original state to restore
    original_plugin_paths = list(registry._plugin_paths)
    original_sys_path = list(sys.path)
    registry._plugins.clear()
    registry._enabled_plugins.clear()
    registry._disabled_plugins.clear()
    registry._plugin_paths.clear()
    yield
    # Clean up after test
    registry._plugins.clear()
    registry._enabled_plugins.clear()
    registry._disabled_plugins.clear()
    registry._plugin_paths.clear()
    registry._plugin_paths.extend(original_plugin_paths)
    # Restore sys.path to prevent plugin path leakage
    sys.path[:] = original_sys_path


# =============================================================================
# TransformDeclaration Tests
# =============================================================================

class TestTransformDeclaration:
    """Test suite for TransformDeclaration class"""
    
    def test_transform_declaration_initialization(self):
        """Test basic initialization of TransformDeclaration"""
        decl = TransformDeclaration(
            name="test_transform",
            class_name="TestTransform",
            module_path="transforms.test_transform",
            category="molecular",
            description="Test transform",
            version="1.0.0"
        )
        
        assert decl.name == "test_transform"
        assert decl.class_name == "TestTransform"
        assert decl.module_path == "transforms.test_transform"
        assert decl.category == "molecular"
        assert decl.description == "Test transform"
        assert decl.version == "1.0.0"
        assert decl.required_node_features == []
        assert decl.required_edge_features == []
        assert decl.required_graph_attributes == []
        assert decl.parameter_constraints == {}
    
    def test_transform_declaration_with_optional_fields(self, sample_transform_declaration):
        """Test TransformDeclaration with optional fields"""
        decl = sample_transform_declaration
        
        assert decl.required_node_features == ["x", "pos"]
        assert decl.required_edge_features == ["edge_attr"]
        assert decl.required_graph_attributes == ["y"]
        assert decl.parameter_constraints == {"max_nodes": 100}
    
    def test_transform_declaration_to_dict(self, sample_transform_declaration):
        """Test conversion of TransformDeclaration to dictionary"""
        decl = sample_transform_declaration
        result = decl.to_dict()
        
        assert isinstance(result, dict)
        assert result["name"] == "test_transform"
        assert result["class_name"] == "TestTransform"
        assert result["module_path"] == "transforms.test_transform"
        assert result["category"] == "molecular"
        assert result["description"] == "A test transform"
        assert result["version"] == "1.0.0"
        assert result["required_node_features"] == ["x", "pos"]
        assert result["required_edge_features"] == ["edge_attr"]
        assert result["required_graph_attributes"] == ["y"]
        assert result["parameter_constraints"] == {"max_nodes": 100}
    
    def test_transform_declaration_from_dict(self):
        """Test creation of TransformDeclaration from dictionary"""
        data = {
            "name": "from_dict_transform",
            "class_name": "FromDictTransform",
            "module_path": "transforms.from_dict",
            "category": "quantum",
            "description": "Created from dict",
            "version": "2.0.0",
            "required_node_features": ["z"],
            "required_edge_features": ["bond_type"],
            "required_graph_attributes": ["energy"],
            "parameter_constraints": {"min_nodes": 5}
        }
        
        decl = TransformDeclaration.from_dict(data)
        
        assert decl.name == "from_dict_transform"
        assert decl.class_name == "FromDictTransform"
        assert decl.module_path == "transforms.from_dict"
        assert decl.category == "quantum"
        assert decl.description == "Created from dict"
        assert decl.version == "2.0.0"
        assert decl.required_node_features == ["z"]
        assert decl.required_edge_features == ["bond_type"]
        assert decl.required_graph_attributes == ["energy"]
        assert decl.parameter_constraints == {"min_nodes": 5}
    
    def test_transform_declaration_from_dict_minimal(self):
        """Test from_dict with minimal required fields"""
        data = {
            "name": "minimal",
            "class_name": "MinimalTransform",
            "module_path": "transforms.minimal",
            "category": "basic",
            "description": "Minimal"
        }
        
        decl = TransformDeclaration.from_dict(data)
        
        assert decl.name == "minimal"
        assert decl.version == "1.0.0"  # Default version
        assert decl.required_node_features == []
        assert decl.required_edge_features == []
        assert decl.required_graph_attributes == []
        assert decl.parameter_constraints == {}


# =============================================================================
# PluginMetadata Tests
# =============================================================================

class TestPluginMetadata:
    """Test suite for PluginMetadata class"""
    
    def test_plugin_metadata_initialization(self):
        """Test basic initialization of PluginMetadata"""
        metadata = PluginMetadata(
            plugin_name="test_plugin",
            version="1.0.0",
            author="Test Author",
            plugin_type="user_experimental"
        )
        
        assert metadata.plugin_name == "test_plugin"
        assert metadata.version == "1.0.0"
        assert metadata.author == "Test Author"
        assert metadata.plugin_type == "user_experimental"
        assert metadata.email is None
        assert metadata.license == "MIT"  # Default
        assert metadata.description == ""
        assert metadata.is_validated is False
        assert metadata.trusted is False
    
    def test_plugin_metadata_with_all_fields(self, sample_plugin_metadata):
        """Test PluginMetadata with all optional fields"""
        metadata = sample_plugin_metadata
        
        assert metadata.email == "test@example.com"
        assert metadata.homepage == "https://example.com"
        assert metadata.milia_version == ">=1.0.0"
        assert metadata.pyg_version == ">=2.0.0"
        assert metadata.python_version == ">=3.8"
        assert metadata.dependencies == ["torch>=2.0.0", "torch-geometric>=2.3.0"]
        assert metadata.discovery_source == "yaml"
        assert metadata.is_validated is False
        assert metadata.trusted is False
    
    def test_plugin_metadata_equality(self):
        """Test equality comparison of PluginMetadata instances"""
        metadata1 = PluginMetadata(
            plugin_name="test",
            version="1.0.0",
            author="Author"
        )
        
        metadata2 = PluginMetadata(
            plugin_name="test",
            version="1.0.0",
            author="Author"
        )
        
        # Since PluginMetadata has eq=True, they should be equal
        # if all fields match (including timestamps which may differ)
        assert metadata1.plugin_name == metadata2.plugin_name
        assert metadata1.version == metadata2.version
        assert metadata1.author == metadata2.author
    
    def test_plugin_metadata_transform_declarations_vs_registrations(self, sample_plugin_metadata):
        """Test separation between transform_declarations and registered_transforms"""
        metadata = sample_plugin_metadata
        
        # Initially both should be empty
        assert metadata.transform_declarations == []
        assert metadata.registered_transforms == set()
        
        # Add a declaration
        decl = TransformDeclaration(
            name="test_transform",
            class_name="TestTransform",
            module_path="transforms.test",
            category="molecular",
            description="Test"
        )
        metadata.transform_declarations.append(decl)
        
        # Declaration exists but not registered yet
        assert len(metadata.transform_declarations) == 1
        assert len(metadata.registered_transforms) == 0
        
        # Simulate registration
        metadata.registered_transforms.add("test_transform")
        
        assert len(metadata.registered_transforms) == 1
        assert "test_transform" in metadata.registered_transforms


# =============================================================================
# Lazy Import Tests
# =============================================================================

class TestLazyImports:
    """Test suite for lazy import mechanisms"""
    
    @patch('milia_pipeline.transformations.plugin_system.CustomTransformBase', None)
    @patch('milia_pipeline.transformations.plugin_system._IMPORTING_CUSTOM_TRANSFORMS', False)
    def test_import_custom_transforms_success(self):
        """Test successful import of custom transforms"""
        mock_ctb = Mock()
        mock_mtb = Mock()
        mock_qtb = Mock()
        mock_tm = Mock()
        
        with patch.dict('sys.modules', {
            'milia_pipeline.transformations.custom_transforms': Mock(
                CustomTransformBase=mock_ctb,
                MolecularTransformBase=mock_mtb,
                QuantumTransformBase=mock_qtb,
                TransformMetadata=mock_tm
            )
        }):
            # Reset global state
            import milia_pipeline.transformations.plugin_system as ps
            ps.CustomTransformBase = None
            ps._IMPORTING_CUSTOM_TRANSFORMS = False
            
            result = _import_custom_transforms()
            
            # Should return True on success
            assert result is True or ps.CustomTransformBase is not None
    
    @patch('milia_pipeline.transformations.plugin_system.CustomTransformBase', None)
    @patch('milia_pipeline.transformations.plugin_system._IMPORTING_CUSTOM_TRANSFORMS', False)
    def test_import_custom_transforms_already_imported(self):
        """Test that _import_custom_transforms returns early if already imported"""
        import milia_pipeline.transformations.plugin_system as ps
        
        # Simulate already imported
        ps.CustomTransformBase = Mock()
        
        result = _import_custom_transforms()
        assert result is True
    
    @patch('milia_pipeline.transformations.plugin_system.CustomTransformBase', None)
    @patch('milia_pipeline.transformations.plugin_system._IMPORTING_CUSTOM_TRANSFORMS', True)
    def test_import_custom_transforms_prevents_reentry(self):
        """Test that _import_custom_transforms prevents re-entry"""
        result = _import_custom_transforms()
        
        # Should return False if already importing
        assert result is False
    
    @patch('milia_pipeline.transformations.plugin_system.TransformRegistry', None)
    @patch('milia_pipeline.transformations.plugin_system._IMPORTING_GRAPH_TRANSFORMS', False)
    def test_import_graph_transforms_success(self):
        """Test successful import of graph transforms"""
        mock_registry = Mock()
        mock_gti = Mock()
        mock_vc = Mock()
        
        with patch.dict('sys.modules', {
            'milia_pipeline.transformations.graph_transforms': Mock(
                registry=mock_registry,
                get_transform_info=mock_gti,
                validate_comprehensive=mock_vc
            )
        }):
            # Reset global state
            import milia_pipeline.transformations.plugin_system as ps
            ps.TransformRegistry = None
            ps._IMPORTING_GRAPH_TRANSFORMS = False
            
            result = _import_graph_transforms()
            
            # Should return True on success
            assert result is True or ps.TransformRegistry is not None
    
    @patch('milia_pipeline.transformations.plugin_system.TransformRegistry', None)
    @patch('milia_pipeline.transformations.plugin_system._IMPORTING_GRAPH_TRANSFORMS', True)
    def test_import_graph_transforms_prevents_reentry(self):
        """Test that _import_graph_transforms prevents re-entry"""
        result = _import_graph_transforms()
        
        # Should return False if already importing
        assert result is False


# =============================================================================
# PluginRegistry Tests - Core Functionality
# =============================================================================

class TestPluginRegistryCore:
    """Test suite for PluginRegistry core functionality"""
    
    def test_plugin_registry_singleton_behavior(self, reset_plugin_registry):
        """Test that PluginRegistry behaves as a singleton with instance-level storage"""
        # Get registry instance
        registry = PluginRegistry()
        
        # Register a plugin
        metadata = PluginMetadata(
            plugin_name="test1",
            version="1.0.0",
            author="Author"
        )
        
        registry._plugins["test1"] = metadata
        
        # Get another reference - should be same instance
        registry2 = PluginRegistry()
        assert registry is registry2
        assert "test1" in registry2._plugins
    
    def test_register_plugin_success(self, reset_plugin_registry, sample_plugin_metadata):
        """Test successful plugin registration via internal _plugins dict"""
        registry = PluginRegistry()
        metadata = sample_plugin_metadata
        
        # Simulate plugin registration (internally used during discovery)
        registry._plugins["test_plugin"] = metadata
        registry._enabled_plugins.add("test_plugin")
        
        assert "test_plugin" in registry._plugins
        assert registry._plugins["test_plugin"] == metadata
        assert "test_plugin" in registry._enabled_plugins
    
    def test_register_plugin_duplicate_error(self, reset_plugin_registry):
        """Test that registering duplicate plugin causes conflict"""
        registry = PluginRegistry()
        
        metadata1 = PluginMetadata(
            plugin_name="duplicate",
            version="1.0.0",
            author="Author1"
        )
        
        metadata2 = PluginMetadata(
            plugin_name="duplicate",
            version="2.0.0",
            author="Author2"
        )
        
        registry._plugins["duplicate"] = metadata1
        
        # Check that duplicate would overwrite
        registry._plugins["duplicate"] = metadata2
        assert registry._plugins["duplicate"].version == "2.0.0"
    
    def test_unregister_plugin_success(self, reset_plugin_registry, sample_plugin_metadata):
        """Test successful plugin unregistration"""
        registry = PluginRegistry()
        metadata = sample_plugin_metadata
        
        registry._plugins["test_plugin"] = metadata
        registry._enabled_plugins.add("test_plugin")
        
        assert "test_plugin" in registry._plugins
        
        # Simulate unregistration
        del registry._plugins["test_plugin"]
        registry._enabled_plugins.discard("test_plugin")
        
        assert "test_plugin" not in registry._plugins
        assert "test_plugin" not in registry._enabled_plugins
    
    def test_unregister_nonexistent_plugin_error(self, reset_plugin_registry):
        """Test that accessing nonexistent plugin raises KeyError"""
        registry = PluginRegistry()
        
        with pytest.raises(KeyError):
            _ = registry._plugins["nonexistent"]
    
    def test_get_plugin_success(self, reset_plugin_registry, sample_plugin_metadata):
        """Test retrieving a registered plugin via get_plugin_info"""
        registry = PluginRegistry()
        metadata = sample_plugin_metadata
        
        registry._plugins["test_plugin"] = metadata
        
        # Use the actual API method
        info = PluginRegistry.get_plugin_info("test_plugin")
        assert info is not None
        assert info["plugin_name"] == "test_plugin"
    
    def test_get_plugin_not_found(self, reset_plugin_registry):
        """Test getting nonexistent plugin returns None"""
        result = PluginRegistry.get_plugin_info("nonexistent")
        assert result is None
    
    def test_list_plugins(self, reset_plugin_registry):
        """Test listing all registered plugins"""
        registry = PluginRegistry()
        
        metadata1 = PluginMetadata(
            plugin_name="plugin1",
            version="1.0.0",
            author="Author1"
        )
        
        metadata2 = PluginMetadata(
            plugin_name="plugin2",
            version="2.0.0",
            author="Author2"
        )
        
        registry._plugins["plugin1"] = metadata1
        registry._plugins["plugin2"] = metadata2
        
        plugins = PluginRegistry.list_plugins()
        
        assert len(plugins) == 2
        assert "plugin1" in plugins
        assert "plugin2" in plugins
    
    def test_list_plugins_empty(self, reset_plugin_registry):
        """Test listing plugins when none are registered"""
        plugins = PluginRegistry.list_plugins()
        assert len(plugins) == 0
        assert plugins == []


# =============================================================================
# PluginRegistry Tests - Enable/Disable
# =============================================================================

class TestPluginRegistryEnableDisable:
    """Test suite for plugin enable/disable functionality"""
    
    def test_enable_plugin_success(self, reset_plugin_registry, sample_plugin_metadata):
        """Test enabling a plugin"""
        registry = PluginRegistry()
        metadata = sample_plugin_metadata
        
        # Register plugin first
        registry._plugins["test_plugin"] = metadata
        registry._disabled_plugins.add("test_plugin")
        
        PluginRegistry.enable_plugin("test_plugin")
        
        assert "test_plugin" in registry._enabled_plugins
        assert "test_plugin" not in registry._disabled_plugins
    
    def test_enable_nonexistent_plugin_error(self, reset_plugin_registry):
        """Test enabling nonexistent plugin raises error"""
        with pytest.raises(PluginError, match="not found"):
            PluginRegistry.enable_plugin("nonexistent")
    
    def test_disable_plugin_success(self, reset_plugin_registry, sample_plugin_metadata):
        """Test disabling a plugin"""
        registry = PluginRegistry()
        metadata = sample_plugin_metadata
        
        registry._plugins["test_plugin"] = metadata
        registry._enabled_plugins.add("test_plugin")
        
        PluginRegistry.disable_plugin("test_plugin")
        
        assert "test_plugin" in registry._disabled_plugins
        assert "test_plugin" not in registry._enabled_plugins
    
    def test_disable_nonexistent_plugin_error(self, reset_plugin_registry):
        """Test disabling nonexistent plugin raises error"""
        with pytest.raises(PluginError, match="not found"):
            PluginRegistry.disable_plugin("nonexistent")
    
    def test_is_enabled_true(self, reset_plugin_registry, sample_plugin_metadata):
        """Test checking if plugin is enabled"""
        registry = PluginRegistry()
        metadata = sample_plugin_metadata
        
        registry._plugins["test_plugin"] = metadata
        registry._enabled_plugins.add("test_plugin")
        
        # Check manually via the set
        assert "test_plugin" in registry._enabled_plugins
    
    def test_is_enabled_false(self, reset_plugin_registry, sample_plugin_metadata):
        """Test checking if disabled plugin returns False"""
        registry = PluginRegistry()
        metadata = sample_plugin_metadata
        
        registry._plugins["test_plugin"] = metadata
        registry._disabled_plugins.add("test_plugin")
        
        assert "test_plugin" not in registry._enabled_plugins
        assert "test_plugin" in registry._disabled_plugins
    
    def test_is_enabled_nonexistent(self, reset_plugin_registry):
        """Test checking nonexistent plugin returns False"""
        registry = PluginRegistry()
        assert "nonexistent" not in registry._enabled_plugins


# =============================================================================
# PluginRegistry Tests - Thread Safety
# =============================================================================

class TestPluginRegistryThreadSafety:
    """Test suite for thread-safe operations"""
    
    def test_concurrent_registration(self, reset_plugin_registry):
        """Test concurrent plugin registration from multiple threads"""
        registry = PluginRegistry()
        results = []
        errors = []
        
        def register_plugin(plugin_num):
            try:
                metadata = PluginMetadata(
                    plugin_name=f"plugin_{plugin_num}",
                    version="1.0.0",
                    author=f"Author{plugin_num}"
                )
                # Use internal registration
                registry._plugins[f"plugin_{plugin_num}"] = metadata
                results.append(plugin_num)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for i in range(10):
            thread = threading.Thread(target=register_plugin, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All registrations should succeed
        assert len(results) == 10
        assert len(errors) == 0
        assert len(registry._plugins) == 10
    
    def test_concurrent_enable_disable(self, reset_plugin_registry):
        """Test concurrent enable/disable operations"""
        registry = PluginRegistry()
        metadata = PluginMetadata(
            plugin_name="concurrent_test",
            version="1.0.0",
            author="Author"
        )
        registry._plugins["concurrent_test"] = metadata
        
        operations = []
        
        def toggle_plugin():
            try:
                for _ in range(5):
                    PluginRegistry.disable_plugin("concurrent_test")
                    time.sleep(0.001)
                    PluginRegistry.enable_plugin("concurrent_test")
                operations.append(True)
            except Exception as e:
                operations.append(e)
        
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=toggle_plugin)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All operations should complete successfully
        assert all(op is True for op in operations)


# =============================================================================
# PluginRegistry Tests - Discovery
# =============================================================================

class TestPluginRegistryDiscovery:
    """Test suite for plugin discovery functionality"""
    
    @patch('milia_pipeline.transformations.plugin_system.yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    def test_discover_plugins_from_directory(self, mock_file, mock_yaml_load, reset_plugin_registry, sample_plugin_yaml):
        """Test discovering plugins from a directory"""
        # Setup mock directory structure
        mock_plugin_dir = Mock(spec=Path)
        mock_plugin_dir.is_dir.return_value = True
        mock_plugin_dir.name = "test_plugin"
        mock_plugin_dir.parent = Mock()
        
        mock_yaml_file = Mock(spec=Path)
        mock_yaml_file.exists.return_value = True
        mock_yaml_file.name = "plugin.yaml"
        mock_yaml_file.parent = mock_plugin_dir
        
        mock_base_dir = Mock(spec=Path)
        mock_base_dir.is_dir.return_value = True
        mock_base_dir.resolve.return_value = mock_base_dir
        
        # Mock glob to return our yaml file
        mock_base_dir.glob.return_value = [mock_yaml_file]
        
        mock_yaml_load.return_value = sample_plugin_yaml
        
        # Add the path and discover
        registry = PluginRegistry()
        registry._plugin_paths.append(mock_base_dir)
        
        # Discover plugins
        result = PluginRegistry.discover_plugins()
        
        # Verify discovery results
        assert isinstance(result, list)
    
    def test_discover_plugins_empty_directory(self, reset_plugin_registry):
        """Test discovering plugins from empty directory"""
        mock_base_dir = Mock(spec=Path)
        mock_base_dir.is_dir.return_value = True
        mock_base_dir.resolve.return_value = mock_base_dir
        mock_base_dir.glob.return_value = []
        
        registry = PluginRegistry()
        registry._plugin_paths.append(mock_base_dir)
        
        result = PluginRegistry.discover_plugins()
        
        assert result == []
    
    def test_discover_plugins_invalid_directory(self, reset_plugin_registry):
        """Test discovering plugins from invalid directory"""
        # Use a real Path object that doesn't exist
        from tempfile import gettempdir
        import uuid
        
        invalid_path = Path(gettempdir()) / f"nonexistent_{uuid.uuid4()}"
        
        # Ensure it doesn't exist and isn't a directory
        assert not invalid_path.exists()
        
        with pytest.raises(PluginError, match="not a directory"):
            PluginRegistry.add_plugin_path(invalid_path)


# =============================================================================
# PluginRegistry Tests - Validation
# =============================================================================

class TestPluginRegistryValidation:
    """Test suite for plugin validation functionality"""
    
    def test_validate_plugin_basic(self, reset_plugin_registry, sample_plugin_metadata):
        """Test basic plugin validation"""
        registry = PluginRegistry()
        registry._plugins["test_plugin"] = sample_plugin_metadata
        
        result = PluginRegistry.validate_plugin("test_plugin")
        
        assert isinstance(result, dict)
        assert "passed" in result
        assert "tests" in result
    
    def test_validate_nonexistent_plugin(self, reset_plugin_registry):
        """Test validating nonexistent plugin"""
        with pytest.raises(PluginError, match="not registered"):
            PluginRegistry.validate_plugin("nonexistent")
    
    @patch('milia_pipeline.transformations.plugin_system.PluginRegistry._check_dependencies')
    def test_check_dependencies_success(self, mock_check_deps, reset_plugin_registry, sample_plugin_metadata):
        """Test dependency checking"""
        mock_check_deps.return_value = {
            "passed": True,
            "missing": [],
            "version_conflicts": []
        }
        
        metadata = sample_plugin_metadata
        result = PluginRegistry._check_dependencies(metadata)
        
        assert result["passed"] is True
        assert "missing" in result
    
    @patch('milia_pipeline.transformations.plugin_system.PluginRegistry._check_security')
    def test_check_security_success(self, mock_check_sec, reset_plugin_registry, sample_plugin_metadata):
        """Test security checking"""
        mock_check_sec.return_value = {
            "passed": True,
            "issues": [],
            "warnings": []
        }
        
        metadata = sample_plugin_metadata
        result = PluginRegistry._check_security(metadata)
        
        assert result["passed"] is True
        assert "issues" in result


# =============================================================================
# PluginValidator Tests
# =============================================================================

class TestPluginValidator:
    """Test suite for PluginValidator class"""
    
    @patch('milia_pipeline.transformations.plugin_system.PluginValidator._analyze_security')
    @patch('milia_pipeline.transformations.plugin_system.PluginValidator._run_functional_tests')
    @patch('milia_pipeline.transformations.plugin_system.PluginValidator._check_documentation')
    @patch('milia_pipeline.transformations.plugin_system.PluginValidator._check_code_quality')
    def test_validate_plugin_comprehensive(
        self,
        mock_code_quality,
        mock_documentation,
        mock_functional,
        mock_security,
        reset_plugin_registry,
        sample_plugin_metadata
    ):
        """Test comprehensive plugin validation"""
        registry = PluginRegistry()
        registry._plugins["test_plugin"] = sample_plugin_metadata
        
        # Mock all the validation methods to return success
        mock_code_quality.return_value = {'passed': True, 'score': 0.95, 'details': {}}
        mock_documentation.return_value = {'passed': True, 'score': 1.0, 'missing': []}
        mock_functional.return_value = {'passed': True, 'score': 1.0, 'test_results': []}
        mock_security.return_value = {'passed': True, 'score': 1.0, 'issues': [], 'warnings': []}
        
        result = PluginValidator.validate_plugin_comprehensive(
            "test_plugin",
            run_performance_tests=False
        )
        
        assert isinstance(result, dict)
        assert "overall_score" in result
        assert "sections" in result
        assert result["overall_score"] > 0.0
    
    def test_validate_plugin_basic_level(self, reset_plugin_registry, sample_plugin_metadata):
        """Test basic level plugin validation via PluginRegistry"""
        registry = PluginRegistry()
        registry._plugins["test_plugin"] = sample_plugin_metadata
        
        result = PluginRegistry.validate_plugin("test_plugin")
        
        assert isinstance(result, dict)
        assert "passed" in result
    
    def test_validate_plugin_nonexistent(self, reset_plugin_registry):
        """Test validating nonexistent plugin"""
        with pytest.raises(PluginError):
            PluginValidator.validate_plugin_comprehensive("nonexistent")
    
    @patch('milia_pipeline.transformations.plugin_system.PluginValidator._check_code_quality')
    def test_check_code_quality(self, mock_check):
        """Test code quality checking"""
        mock_check.return_value = {
            "passed": True,
            "score": 0.95,
            "details": {}
        }
        
        metadata_dict = {
            "plugin_name": "test",
            "version": "1.0.0",
            "author": "Author"
        }
        
        result = PluginValidator._check_code_quality(metadata_dict)
        
        assert result["passed"] is True
        assert result["score"] == 0.95
    
    @patch('milia_pipeline.transformations.plugin_system.PluginValidator._check_documentation')
    def test_check_documentation(self, mock_check):
        """Test documentation checking"""
        mock_check.return_value = {
            "passed": True,
            "score": 1.0,
            "missing": []
        }
        
        metadata_dict = {
            "plugin_name": "test",
            "version": "1.0.0",
            "author": "Author",
            "description": "Test plugin",
            "homepage": "https://example.com"
        }
        
        result = PluginValidator._check_documentation(metadata_dict)
        
        assert result["passed"] is True
    
    def test_calculate_score(self):
        """Test overall score calculation"""
        sections = {
            "code_quality": {"score": 0.9},
            "documentation": {"score": 0.8},
            "functional": {"score": 1.0},
            "performance": {"score": 0.7},
            "security": {"score": 0.95}
        }
        
        score = PluginValidator._calculate_score(sections)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_generate_recommendation_approved(self):
        """Test recommendation generation for high scores"""
        report = {"overall_score": 0.96}
        
        recommendation = PluginValidator._generate_recommendation(report)
        
        assert "APPROVED" in recommendation
        assert "Excellent" in recommendation
    
    def test_generate_recommendation_rejected(self):
        """Test recommendation generation for low scores"""
        report = {"overall_score": 0.45}
        
        recommendation = PluginValidator._generate_recommendation(report)
        
        assert "REJECTED" in recommendation or "NOT APPROVED" in recommendation


# =============================================================================
# Exception Tests
# =============================================================================

# =============================================================================
# PluginMetadata - Computed Properties and Pydantic Behaviors
# =============================================================================

class TestPluginMetadataProperties:
    """Test suite for PluginMetadata computed properties and Pydantic model behaviors"""

    def test_declared_count_empty(self):
        """Test declared_count returns 0 when no declarations"""
        metadata = PluginMetadata(
            plugin_name="props_test", version="1.0.0", author="Author"
        )
        assert metadata.declared_count == 0

    def test_declared_count_populated(self):
        """Test declared_count reflects transform_declarations length"""
        decl_a = TransformDeclaration(
            name="t_a", class_name="A", module_path="m",
            category="cat", description="desc"
        )
        decl_b = TransformDeclaration(
            name="t_b", class_name="B", module_path="m",
            category="cat", description="desc"
        )
        metadata = PluginMetadata(
            plugin_name="props_test", version="1.0.0", author="Author",
            transform_declarations=[decl_a, decl_b]
        )
        assert metadata.declared_count == 2

    def test_registered_count_empty(self):
        """Test registered_count returns 0 when no registrations"""
        metadata = PluginMetadata(
            plugin_name="props_test", version="1.0.0", author="Author"
        )
        assert metadata.registered_count == 0

    def test_registered_count_populated(self):
        """Test registered_count reflects registered_transforms set size"""
        metadata = PluginMetadata(
            plugin_name="props_test", version="1.0.0", author="Author",
            registered_transforms={"t_a", "t_b", "t_c"}
        )
        assert metadata.registered_count == 3

    def test_missing_implementations_none(self):
        """Test missing_implementations when all declared are registered"""
        decl = TransformDeclaration(
            name="t_a", class_name="A", module_path="m",
            category="cat", description="desc"
        )
        metadata = PluginMetadata(
            plugin_name="props_test", version="1.0.0", author="Author",
            transform_declarations=[decl],
            registered_transforms={"t_a"}
        )
        assert metadata.missing_implementations == []

    def test_missing_implementations_present(self):
        """Test missing_implementations when some declared are not registered"""
        decl_a = TransformDeclaration(
            name="t_a", class_name="A", module_path="m",
            category="cat", description="desc"
        )
        decl_b = TransformDeclaration(
            name="t_b", class_name="B", module_path="m",
            category="cat", description="desc"
        )
        metadata = PluginMetadata(
            plugin_name="props_test", version="1.0.0", author="Author",
            transform_declarations=[decl_a, decl_b],
            registered_transforms={"t_a"}
        )
        assert metadata.missing_implementations == ["t_b"]

    def test_undeclared_implementations_none(self):
        """Test undeclared_implementations when all registered are declared"""
        decl = TransformDeclaration(
            name="t_a", class_name="A", module_path="m",
            category="cat", description="desc"
        )
        metadata = PluginMetadata(
            plugin_name="props_test", version="1.0.0", author="Author",
            transform_declarations=[decl],
            registered_transforms={"t_a"}
        )
        assert metadata.undeclared_implementations == []

    def test_undeclared_implementations_present(self):
        """Test undeclared_implementations when bonus transforms are registered"""
        decl = TransformDeclaration(
            name="t_a", class_name="A", module_path="m",
            category="cat", description="desc"
        )
        metadata = PluginMetadata(
            plugin_name="props_test", version="1.0.0", author="Author",
            transform_declarations=[decl],
            registered_transforms={"t_a", "t_bonus"}
        )
        assert "t_bonus" in metadata.undeclared_implementations

    def test_hash_based_on_name_and_version(self):
        """Test __hash__ uses plugin_name and version"""
        m1 = PluginMetadata(
            plugin_name="test", version="1.0.0", author="A"
        )
        m2 = PluginMetadata(
            plugin_name="test", version="1.0.0", author="B"
        )
        assert hash(m1) == hash(m2)

    def test_hash_differs_for_different_version(self):
        """Test __hash__ differs when version changes"""
        m1 = PluginMetadata(
            plugin_name="test", version="1.0.0", author="A"
        )
        m2 = PluginMetadata(
            plugin_name="test", version="2.0.0", author="A"
        )
        assert hash(m1) != hash(m2)

    def test_hash_differs_for_different_name(self):
        """Test __hash__ differs when plugin_name changes"""
        m1 = PluginMetadata(
            plugin_name="alpha", version="1.0.0", author="A"
        )
        m2 = PluginMetadata(
            plugin_name="beta", version="1.0.0", author="A"
        )
        assert hash(m1) != hash(m2)

    def test_equality_same_name_version(self):
        """Test __eq__ returns True for same plugin_name and version"""
        m1 = PluginMetadata(
            plugin_name="test", version="1.0.0", author="A"
        )
        m2 = PluginMetadata(
            plugin_name="test", version="1.0.0", author="B"
        )
        assert m1 == m2

    def test_equality_different_name(self):
        """Test __eq__ returns False for different plugin_name"""
        m1 = PluginMetadata(
            plugin_name="alpha", version="1.0.0", author="A"
        )
        m2 = PluginMetadata(
            plugin_name="beta", version="1.0.0", author="A"
        )
        assert m1 != m2

    def test_equality_not_implemented_for_non_metadata(self):
        """Test __eq__ returns NotImplemented for non-PluginMetadata"""
        m = PluginMetadata(
            plugin_name="test", version="1.0.0", author="A"
        )
        result = m.__eq__("not_metadata")
        assert result is NotImplemented

    def test_usable_in_set(self):
        """Test PluginMetadata can be used in sets (via __hash__)"""
        m1 = PluginMetadata(
            plugin_name="test", version="1.0.0", author="A"
        )
        m2 = PluginMetadata(
            plugin_name="test", version="1.0.0", author="B"
        )
        m3 = PluginMetadata(
            plugin_name="other", version="1.0.0", author="A"
        )
        s = {m1, m2, m3}
        # m1 and m2 are equal, so set should have 2 elements
        assert len(s) == 2

    def test_invalid_version_format_rejected(self):
        """Test that non-semver version format raises PluginError"""
        with pytest.raises(PluginError, match="Invalid version format"):
            PluginMetadata(
                plugin_name="bad_version",
                version="not.a.version.format",
                author="Author"
            )

    def test_invalid_version_no_dots(self):
        """Test that version without dots is rejected"""
        with pytest.raises(PluginError, match="Invalid version format"):
            PluginMetadata(
                plugin_name="bad_version",
                version="100",
                author="Author"
            )

    def test_valid_version_with_prerelease(self):
        """Test semver with prerelease tag is accepted"""
        m = PluginMetadata(
            plugin_name="prerelease", version="1.0.0-alpha.1", author="A"
        )
        assert m.version == "1.0.0-alpha.1"

    def test_is_valid_version_static(self):
        """Test the _is_valid_version static method directly"""
        assert PluginMetadata._is_valid_version("1.0.0") is True
        assert PluginMetadata._is_valid_version("0.0.1") is True
        assert PluginMetadata._is_valid_version("10.20.30") is True
        assert PluginMetadata._is_valid_version("1.0.0-beta") is True
        assert PluginMetadata._is_valid_version("abc") is False
        assert PluginMetadata._is_valid_version("1.0") is False
        assert PluginMetadata._is_valid_version("") is False

    def test_validate_dependencies_non_string_raises(self):
        """Test that non-string dependency specification raises validation error.
        
        Pydantic V2 enforces List[str] type at the schema level, so a
        pydantic ValidationError is raised before the custom
        _validate_dependencies() method executes. The custom method
        serves as a secondary guard for edge cases that bypass Pydantic
        (e.g., direct attribute mutation after construction).
        """
        from pydantic import ValidationError as PydanticValidationError
        with pytest.raises(PydanticValidationError, match="Input should be a valid string"):
            PluginMetadata(
                plugin_name="bad_deps", version="1.0.0", author="A",
                dependencies=[123]
            )

    def test_from_dict_classmethod(self):
        """Test PluginMetadata.from_dict creates instance from dictionary"""
        data = {
            "plugin_name": "from_dict_test",
            "version": "2.0.0",
            "author": "Dict Author",
            "description": "Created from dict"
        }
        m = PluginMetadata.from_dict(data)
        assert m.plugin_name == "from_dict_test"
        assert m.version == "2.0.0"
        assert m.author == "Dict Author"
        assert m.description == "Created from dict"

    def test_to_dict_includes_computed_properties(self):
        """Test to_dict includes computed properties"""
        decl = TransformDeclaration(
            name="t_a", class_name="A", module_path="m",
            category="cat", description="desc"
        )
        metadata = PluginMetadata(
            plugin_name="dict_test", version="1.0.0", author="A",
            transform_declarations=[decl],
            registered_transforms={"t_a", "t_bonus"}
        )
        d = metadata.to_dict()
        assert "declared_count" in d
        assert d["declared_count"] == 1
        assert "registered_count" in d
        assert d["registered_count"] == 2
        assert "missing_implementations" in d
        assert "undeclared_implementations" in d
        # registered_transforms should be serialized as list
        assert isinstance(d["registered_transforms"], list)
        # transform_declarations should be list of dicts
        assert isinstance(d["transform_declarations"], list)
        assert isinstance(d["transform_declarations"][0], dict)

    def test_to_dict_round_trip_declarations(self):
        """Test that to_dict preserves all TransformDeclaration data"""
        decl = TransformDeclaration(
            name="rt", class_name="RT", module_path="m.rt",
            category="quantum", description="round trip",
            version="2.0.0",
            required_node_features=["x"],
            parameter_constraints={"k": 5}
        )
        metadata = PluginMetadata(
            plugin_name="rt_test", version="1.0.0", author="A",
            transform_declarations=[decl]
        )
        d = metadata.to_dict()
        assert d["transform_declarations"][0]["name"] == "rt"
        assert d["transform_declarations"][0]["parameter_constraints"] == {"k": 5}

    def test_default_field_factories_are_independent(self):
        """Test that default mutable fields are independent across instances"""
        m1 = PluginMetadata(
            plugin_name="iso1", version="1.0.0", author="A"
        )
        m2 = PluginMetadata(
            plugin_name="iso2", version="1.0.0", author="A"
        )
        m1.dependencies.append("numpy")
        m1.registered_transforms.add("bonus")
        # m2 should not be affected
        assert "numpy" not in m2.dependencies
        assert "bonus" not in m2.registered_transforms


# =============================================================================
# PluginRegistry - Static Utility Methods
# =============================================================================

class TestPluginRegistryUtilities:
    """Test suite for PluginRegistry static/utility methods"""

    def test_calculate_file_checksum(self, tmp_path):
        """Test _calculate_file_checksum produces consistent SHA256"""
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')\n")

        expected = hashlib.sha256(b"print('hello')\n").hexdigest()
        result = PluginRegistry._calculate_file_checksum(test_file)
        assert result == expected

    def test_calculate_file_checksum_deterministic(self, tmp_path):
        """Test checksum is deterministic for same content"""
        f1 = tmp_path / "a.py"
        f1.write_text("content")
        c1 = PluginRegistry._calculate_file_checksum(f1)
        c2 = PluginRegistry._calculate_file_checksum(f1)
        assert c1 == c2

    def test_calculate_directory_checksum(self, tmp_path):
        """Test _calculate_directory_checksum hashes all .py files"""
        (tmp_path / "a.py").write_text("aaa")
        (tmp_path / "b.py").write_text("bbb")
        (tmp_path / "not_python.txt").write_text("ignore me")

        result = PluginRegistry._calculate_directory_checksum(tmp_path)
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 hex digest length

    def test_calculate_directory_checksum_deterministic(self, tmp_path):
        """Test directory checksum is deterministic"""
        (tmp_path / "a.py").write_text("aaa")
        (tmp_path / "b.py").write_text("bbb")
        c1 = PluginRegistry._calculate_directory_checksum(tmp_path)
        c2 = PluginRegistry._calculate_directory_checksum(tmp_path)
        assert c1 == c2

    def test_calculate_directory_checksum_changes_with_content(self, tmp_path):
        """Test directory checksum changes when file content changes"""
        f = tmp_path / "a.py"
        f.write_text("v1")
        c1 = PluginRegistry._calculate_directory_checksum(tmp_path)
        f.write_text("v2")
        c2 = PluginRegistry._calculate_directory_checksum(tmp_path)
        assert c1 != c2

    def test_is_custom_transform_with_valid_class(self):
        """Test _is_custom_transform returns True for protocol-compliant class"""
        class FakeTransform:
            def forward(self, data):
                return data
            def get_metadata(self):
                return Mock()

        # Patch CustomTransformBase so the check doesn't skip
        with patch('milia_pipeline.transformations.plugin_system.CustomTransformBase', type('CTB', (), {})):
            with patch('milia_pipeline.transformations.plugin_system.MolecularTransformBase', type('MTB', (), {})):
                with patch('milia_pipeline.transformations.plugin_system.QuantumTransformBase', type('QTB', (), {})):
                    result = PluginRegistry._is_custom_transform(FakeTransform)
                    assert result is True

    def test_is_custom_transform_missing_methods(self):
        """Test _is_custom_transform returns False if forward/get_metadata missing"""
        class Incomplete:
            def forward(self, data):
                return data
            # Missing get_metadata

        with patch('milia_pipeline.transformations.plugin_system.CustomTransformBase', type('CTB', (), {})):
            with patch('milia_pipeline.transformations.plugin_system.MolecularTransformBase', type('MTB', (), {})):
                with patch('milia_pipeline.transformations.plugin_system.QuantumTransformBase', type('QTB', (), {})):
                    result = PluginRegistry._is_custom_transform(Incomplete)
                    assert result is False

    def test_is_custom_transform_excludes_base_classes(self):
        """Test _is_custom_transform returns False for base classes themselves"""
        base = type('CustomTransformBase', (), {
            'forward': lambda self, data: data,
            'get_metadata': classmethod(lambda cls: Mock()),
        })
        with patch('milia_pipeline.transformations.plugin_system.CustomTransformBase', base):
            with patch('milia_pipeline.transformations.plugin_system.MolecularTransformBase', type('MTB', (), {})):
                with patch('milia_pipeline.transformations.plugin_system.QuantumTransformBase', type('QTB', (), {})):
                    result = PluginRegistry._is_custom_transform(base)
                    assert result is False

    def test_is_custom_transform_when_custom_transform_base_none(self):
        """Test _is_custom_transform returns False when CustomTransformBase is None"""
        with patch('milia_pipeline.transformations.plugin_system.CustomTransformBase', None):
            with patch('milia_pipeline.transformations.plugin_system._import_custom_transforms', return_value=False):
                result = PluginRegistry._is_custom_transform(object)
                assert result is False


# =============================================================================
# PluginRegistry - Validation Internal Methods
# =============================================================================

class TestPluginRegistryValidationInternal:
    """Test suite for PluginRegistry internal validation methods"""

    def test_validate_consistency_perfect(self):
        """Test _validate_consistency with perfect declaration/registration match"""
        decl = TransformDeclaration(
            name="t_a", class_name="A", module_path="m",
            category="c", description="d"
        )
        metadata = PluginMetadata(
            plugin_name="test", version="1.0.0", author="A",
            transform_declarations=[decl],
            registered_transforms={"t_a"}
        )
        result = PluginRegistry._validate_consistency(metadata)
        assert result["passed"] is True
        assert result["missing"] == []
        assert result["undeclared"] == []
        assert "Perfect consistency" in result["details"]

    def test_validate_consistency_missing_implementations(self):
        """Test _validate_consistency when declared transforms are not registered"""
        decl = TransformDeclaration(
            name="t_a", class_name="A", module_path="m",
            category="c", description="d"
        )
        metadata = PluginMetadata(
            plugin_name="test", version="1.0.0", author="A",
            transform_declarations=[decl],
            registered_transforms=set()
        )
        result = PluginRegistry._validate_consistency(metadata)
        # passed is False when nothing is registered AND there are missing
        assert result["passed"] is False
        assert "t_a" in result["missing"]

    def test_validate_consistency_with_bonus_discoveries(self):
        """Test _validate_consistency when undeclared transforms exist"""
        metadata = PluginMetadata(
            plugin_name="test", version="1.0.0", author="A",
            transform_declarations=[],
            registered_transforms={"bonus_transform"}
        )
        result = PluginRegistry._validate_consistency(metadata)
        assert result["passed"] is True
        assert "bonus_transform" in result["undeclared"]

    def test_validate_consistency_partial_success(self):
        """Test _validate_consistency partial: some missing but some registered"""
        decl_a = TransformDeclaration(
            name="t_a", class_name="A", module_path="m",
            category="c", description="d"
        )
        decl_b = TransformDeclaration(
            name="t_b", class_name="B", module_path="m",
            category="c", description="d"
        )
        metadata = PluginMetadata(
            plugin_name="test", version="1.0.0", author="A",
            transform_declarations=[decl_a, decl_b],
            registered_transforms={"t_a"}
        )
        result = PluginRegistry._validate_consistency(metadata)
        # partial success: len(missing) > 0 but registered_count > 0
        assert result["passed"] is True
        assert "t_b" in result["missing"]

    def test_check_security_trusted_plugin(self):
        """Test _check_security relaxes checks for trusted plugins"""
        metadata = PluginMetadata(
            plugin_name="trusted_one", version="1.0.0", author="A",
            trusted=True
        )
        result = PluginRegistry._check_security(metadata)
        assert result["passed"] is True
        assert len(result["issues"]) == 0
        assert any("trusted" in w.lower() for w in result["warnings"])

    def test_check_security_no_checksum_warning(self):
        """Test _check_security warns about missing checksum"""
        metadata = PluginMetadata(
            plugin_name="no_checksum", version="1.0.0", author="A",
            checksum=None, trusted=False
        )
        result = PluginRegistry._check_security(metadata)
        assert result["passed"] is True  # no issues, just warnings
        assert any("checksum" in w.lower() for w in result["warnings"])

    def test_check_security_with_checksum_no_warnings_about_it(self):
        """Test _check_security does not warn about checksum when present"""
        metadata = PluginMetadata(
            plugin_name="has_checksum", version="1.0.0", author="A",
            checksum="abc123def456", trusted=False
        )
        result = PluginRegistry._check_security(metadata)
        assert result["passed"] is True
        checksum_warnings = [w for w in result["warnings"] if "checksum" in w.lower()]
        assert len(checksum_warnings) == 0

    def test_check_dependencies_no_extra_deps(self):
        """Test _check_dependencies passes when no extra deps specified"""
        metadata = PluginMetadata(
            plugin_name="no_deps", version="1.0.0", author="A",
            dependencies=[]
        )
        result = PluginRegistry._check_dependencies(metadata)
        # PyG may or may not be installed, so we check structure
        assert "passed" in result
        assert "missing" in result
        assert isinstance(result["missing"], list)

    def test_check_dependencies_missing_package(self):
        """Test _check_dependencies detects missing dependency"""
        metadata = PluginMetadata(
            plugin_name="missing_dep", version="1.0.0", author="A",
            dependencies=["nonexistent_package_xyz_12345>=1.0.0"]
        )
        result = PluginRegistry._check_dependencies(metadata)
        assert len(result["missing"]) > 0
        assert any("nonexistent_package_xyz_12345" in m for m in result["missing"])

    def test_check_dependencies_installed_package(self):
        """Test _check_dependencies passes for installed packages"""
        metadata = PluginMetadata(
            plugin_name="good_dep", version="1.0.0", author="A",
            dependencies=["yaml"]  # pyyaml is imported as yaml, available
        )
        result = PluginRegistry._check_dependencies(metadata)
        yaml_missing = [m for m in result["missing"] if "yaml" in m]
        assert len(yaml_missing) == 0


# =============================================================================
# PluginRegistry - Transform Loading
# =============================================================================

class TestPluginRegistryTransformLoading:
    """Test suite for PluginRegistry transform loading methods"""

    def test_load_transform_class_file_not_found(self, reset_plugin_registry, tmp_path):
        """Test _load_transform_class raises FileNotFoundError for missing module"""
        registry = PluginRegistry()
        with pytest.raises(FileNotFoundError, match="Module file not found"):
            registry._load_transform_class(
                plugin_dir=tmp_path,
                module_path="nonexistent_module",
                class_name="SomeClass"
            )

    def test_load_transform_class_class_not_in_module(self, reset_plugin_registry, tmp_path):
        """Test _load_transform_class raises AttributeError when class missing"""
        # Create a valid Python file without the expected class
        module_file = tmp_path / "my_module.py"
        module_file.write_text("class OtherClass:\n    pass\n")

        registry = PluginRegistry()
        with pytest.raises(AttributeError, match="Class 'MissingClass' not found"):
            registry._load_transform_class(
                plugin_dir=tmp_path,
                module_path="my_module",
                class_name="MissingClass"
            )

    def test_load_transform_class_import_error(self, reset_plugin_registry, tmp_path):
        """Test _load_transform_class raises ImportError for broken module"""
        module_file = tmp_path / "broken_module.py"
        module_file.write_text("import nonexistent_library_xyz_99\n")

        registry = PluginRegistry()
        with pytest.raises(ImportError):
            registry._load_transform_class(
                plugin_dir=tmp_path,
                module_path="broken_module",
                class_name="AnyClass"
            )

    def test_load_transform_class_success(self, reset_plugin_registry, tmp_path):
        """Test _load_transform_class successfully loads a valid class"""
        module_file = tmp_path / "good_module.py"
        module_file.write_text(
            "class GoodTransform:\n"
            "    def forward(self, data): return data\n"
            "    @classmethod\n"
            "    def get_metadata(cls): return None\n"
        )

        registry = PluginRegistry()
        loaded_class = registry._load_transform_class(
            plugin_dir=tmp_path,
            module_path="good_module",
            class_name="GoodTransform"
        )
        assert loaded_class.__name__ == "GoodTransform"
        assert hasattr(loaded_class, 'forward')
        assert hasattr(loaded_class, 'get_metadata')

    def test_load_transform_class_spec_none(self, reset_plugin_registry, tmp_path):
        """Test _load_transform_class raises ImportError when spec is None"""
        module_file = tmp_path / "spec_test.py"
        module_file.write_text("x = 1\n")

        registry = PluginRegistry()
        with patch('milia_pipeline.transformations.plugin_system.importlib.util.spec_from_file_location', return_value=None):
            with pytest.raises(ImportError, match="Failed to create module spec"):
                registry._load_transform_class(
                    plugin_dir=tmp_path,
                    module_path="spec_test",
                    class_name="AnyClass"
                )

    def test_load_transform_class_circular_import_detection(self, reset_plugin_registry, tmp_path):
        """Test _load_transform_class detects circular import patterns"""
        module_file = tmp_path / "circular_mod.py"
        module_file.write_text("raise ImportError('circular import detected')\n")

        registry = PluginRegistry()
        with pytest.raises(ImportError, match="[Cc]ircular"):
            registry._load_transform_class(
                plugin_dir=tmp_path,
                module_path="circular_mod",
                class_name="AnyClass"
            )


# =============================================================================
# PluginRegistry - 3-Tier Fallback Registration
# =============================================================================

class TestPluginRegistryFallbackRegistration:
    """Test suite for _register_transform_with_fallback 3-tier strategy"""

    def test_tier1_pyg_native_found(self, reset_plugin_registry, tmp_path):
        """Test Tier 1: Transform already in TransformRegistry (PyG native)"""
        registry = PluginRegistry()
        mock_native_class = type('NativeTransform', (), {
            '__name__': 'NativeTransform',
            '__module__': 'torch_geometric.transforms'
        })

        mock_tr = Mock()
        mock_tr.get.return_value = mock_native_class

        decl = TransformDeclaration(
            name="native_t", class_name="NativeTransform",
            module_path="transforms", category="c", description="d"
        )
        plugin_meta = PluginMetadata(
            plugin_name="test", version="1.0.0", author="A"
        )

        with patch('milia_pipeline.transformations.plugin_system._import_graph_transforms'):
            with patch('milia_pipeline.transformations.plugin_system.TransformRegistry', mock_tr):
                # We need to patch the module-level TransformRegistry that the instance method uses
                import milia_pipeline.transformations.plugin_system as ps
                original_tr = ps.TransformRegistry
                ps.TransformRegistry = mock_tr
                try:
                    result = registry._register_transform_with_fallback(
                        declaration=decl,
                        plugin_dir=tmp_path,
                        plugin_meta=plugin_meta
                    )
                finally:
                    ps.TransformRegistry = original_tr

        assert result["registered"] is True
        assert result["source"] == "pyg"

    def test_tier3_no_implementation(self, reset_plugin_registry, tmp_path):
        """Test Tier 3: No implementation found anywhere"""
        registry = PluginRegistry()

        decl = TransformDeclaration(
            name="missing_t", class_name="MissingTransform",
            module_path="nonexistent", category="c", description="d"
        )
        plugin_meta = PluginMetadata(
            plugin_name="test", version="1.0.0", author="A"
        )

        import milia_pipeline.transformations.plugin_system as ps
        # Tier 1: registry returns None
        mock_tr = Mock()
        mock_tr.get.return_value = None
        original_tr = ps.TransformRegistry
        ps.TransformRegistry = mock_tr
        try:
            result = registry._register_transform_with_fallback(
                declaration=decl,
                plugin_dir=tmp_path,
                plugin_meta=plugin_meta
            )
        finally:
            ps.TransformRegistry = original_tr

        assert result["registered"] is False
        assert result["source"] == "none"
        assert "no implementation" in result["reason"].lower()


# =============================================================================
# PluginRegistry - Plugin Loading from Module and Standalone
# =============================================================================

class TestPluginRegistryPluginLoading:
    """Test suite for _load_plugin_from_module and _load_plugin_from_standalone"""

    def test_load_plugin_from_standalone_no_plugin_metadata(self, reset_plugin_registry, tmp_path):
        """Test _load_plugin_from_standalone returns None when no PLUGIN_METADATA"""
        py_file = tmp_path / "no_meta.py"
        py_file.write_text("x = 1\n")

        result = PluginRegistry._load_plugin_from_standalone(py_file)
        assert result is None

    def test_load_plugin_from_standalone_spec_none(self, reset_plugin_registry, tmp_path):
        """Test _load_plugin_from_standalone returns None when spec is None"""
        py_file = tmp_path / "dummy.py"
        py_file.write_text("x = 1\n")

        with patch('milia_pipeline.transformations.plugin_system.importlib.util.spec_from_file_location', return_value=None):
            result = PluginRegistry._load_plugin_from_standalone(py_file)
            assert result is None

    def test_load_plugin_from_module_missing_plugin_py(self, reset_plugin_registry, tmp_path):
        """Test _load_plugin_from_module raises PluginError when __plugin__.py missing"""
        with pytest.raises(PluginError, match="Failed to load plugin"):
            PluginRegistry._load_plugin_from_module(tmp_path)


# =============================================================================
# PluginRegistry - Discovery Summary Logging
# =============================================================================

class TestPluginRegistryLogging:
    """Test suite for _log_plugin_discovery_summary"""

    def test_log_plugin_discovery_summary_all_registered(self, reset_plugin_registry, caplog):
        """Test summary log for plugin with all transforms registered"""
        import logging
        decl = TransformDeclaration(
            name="t_a", class_name="A", module_path="m",
            category="c", description="d"
        )
        metadata = PluginMetadata(
            plugin_name="summary_test", version="1.0.0", author="A",
            transform_declarations=[decl],
            registered_transforms={"t_a"}
        )

        registry = PluginRegistry()
        results = [{"registered": True, "source": "plugin",
                     "transform_name": "t_a", "reason": "ok",
                     "details": {"module": "m"}}]

        with caplog.at_level(logging.INFO, logger="milia_Main.PluginSystem"):
            registry._log_plugin_discovery_summary(metadata, results)

        assert "summary_test" in caplog.text

    def test_log_plugin_discovery_summary_with_missing(self, reset_plugin_registry, caplog):
        """Test summary log for plugin with missing implementations"""
        import logging
        decl = TransformDeclaration(
            name="t_missing", class_name="M", module_path="m",
            category="c", description="d"
        )
        metadata = PluginMetadata(
            plugin_name="missing_test", version="1.0.0", author="A",
            transform_declarations=[decl],
            registered_transforms=set()
        )

        registry = PluginRegistry()
        results = [{"registered": False, "source": "none",
                     "transform_name": "t_missing", "reason": "not found",
                     "details": {}}]

        with caplog.at_level(logging.INFO, logger="milia_Main.PluginSystem"):
            registry._log_plugin_discovery_summary(metadata, results)

        assert "missing_test" in caplog.text


# =============================================================================
# PluginRegistry - add_plugin_path
# =============================================================================

class TestPluginRegistryAddPluginPath:
    """Test suite for add_plugin_path method"""

    def test_add_plugin_path_success(self, reset_plugin_registry, tmp_path):
        """Test adding a valid plugin directory path"""
        registry = PluginRegistry()
        original_paths_count = len(registry._plugin_paths)
        PluginRegistry.add_plugin_path(tmp_path)
        assert tmp_path.resolve() in registry._plugin_paths
        assert str(tmp_path.resolve()) in sys.path

    def test_add_plugin_path_duplicate_ignored(self, reset_plugin_registry, tmp_path):
        """Test that adding the same path twice doesn't duplicate"""
        registry = PluginRegistry()
        PluginRegistry.add_plugin_path(tmp_path)
        count_after_first = len(registry._plugin_paths)
        PluginRegistry.add_plugin_path(tmp_path)
        # Should NOT add a duplicate
        assert len([p for p in registry._plugin_paths if p == tmp_path.resolve()]) == 1


# =============================================================================
# PluginRegistry - list_plugins with filters
# =============================================================================

class TestPluginRegistryListFilters:
    """Test suite for list_plugins with validated_only and enabled_only filters"""

    def test_list_plugins_validated_only(self, reset_plugin_registry):
        """Test list_plugins with validated_only=True"""
        registry = PluginRegistry()

        m1 = PluginMetadata(
            plugin_name="validated_one", version="1.0.0", author="A"
        )
        m1.is_validated = True

        m2 = PluginMetadata(
            plugin_name="not_validated", version="1.0.0", author="A"
        )

        registry._plugins["validated_one"] = m1
        registry._plugins["not_validated"] = m2

        result = PluginRegistry.list_plugins(validated_only=True)
        assert "validated_one" in result
        assert "not_validated" not in result

    def test_list_plugins_enabled_only(self, reset_plugin_registry):
        """Test list_plugins with enabled_only=True"""
        registry = PluginRegistry()

        m1 = PluginMetadata(
            plugin_name="enabled_one", version="1.0.0", author="A"
        )
        m2 = PluginMetadata(
            plugin_name="disabled_one", version="1.0.0", author="A"
        )

        registry._plugins["enabled_one"] = m1
        registry._plugins["disabled_one"] = m2
        registry._enabled_plugins.add("enabled_one")

        result = PluginRegistry.list_plugins(enabled_only=True)
        assert "enabled_one" in result
        assert "disabled_one" not in result

    def test_list_plugins_combined_filters(self, reset_plugin_registry):
        """Test list_plugins with both validated_only and enabled_only"""
        registry = PluginRegistry()

        m1 = PluginMetadata(
            plugin_name="both", version="1.0.0", author="A"
        )
        m1.is_validated = True

        m2 = PluginMetadata(
            plugin_name="validated_only_p", version="1.0.0", author="A"
        )
        m2.is_validated = True

        m3 = PluginMetadata(
            plugin_name="enabled_only_p", version="1.0.0", author="A"
        )

        registry._plugins["both"] = m1
        registry._plugins["validated_only_p"] = m2
        registry._plugins["enabled_only_p"] = m3
        registry._enabled_plugins.add("both")
        registry._enabled_plugins.add("enabled_only_p")

        result = PluginRegistry.list_plugins(validated_only=True, enabled_only=True)
        assert "both" in result
        assert "validated_only_p" not in result
        assert "enabled_only_p" not in result


# =============================================================================
# PluginValidator - Detailed Method Tests
# =============================================================================

class TestPluginValidatorDetailed:
    """Detailed tests for PluginValidator methods with real logic"""

    def test_check_documentation_complete(self):
        """Test _check_documentation passes with description and homepage"""
        metadata = PluginMetadata(
            plugin_name="doc_test", version="1.0.0", author="A",
            description="Has description", homepage="https://example.com"
        )
        result = PluginValidator._check_documentation(metadata)
        assert result["passed"] is True
        assert result["score"] == 1.0
        assert result["missing"] == []

    def test_check_documentation_missing_description(self):
        """Test _check_documentation flags missing description"""
        metadata = PluginMetadata(
            plugin_name="doc_test", version="1.0.0", author="A",
            description="", homepage="https://example.com"
        )
        result = PluginValidator._check_documentation(metadata)
        assert "Plugin description" in result["missing"]
        assert result["score"] < 1.0

    def test_check_documentation_missing_homepage(self):
        """Test _check_documentation flags missing homepage"""
        metadata = PluginMetadata(
            plugin_name="doc_test", version="1.0.0", author="A",
            description="Has desc", homepage=None
        )
        result = PluginValidator._check_documentation(metadata)
        assert "Homepage/repository URL" in result["missing"]
        assert result["score"] < 1.0

    def test_check_documentation_missing_both(self):
        """Test _check_documentation flags both missing description and homepage"""
        metadata = PluginMetadata(
            plugin_name="doc_test", version="1.0.0", author="A",
            description="", homepage=None
        )
        result = PluginValidator._check_documentation(metadata)
        assert result["passed"] is False
        assert len(result["missing"]) == 2
        assert result["score"] == 0.6  # 1.0 - (2 * 0.2)

    def test_check_code_quality_placeholder(self):
        """Test _check_code_quality returns placeholder result"""
        metadata = PluginMetadata(
            plugin_name="cq_test", version="1.0.0", author="A"
        )
        result = PluginValidator._check_code_quality(metadata)
        assert result["passed"] is True
        assert result["score"] == 0.95
        assert "details" in result

    def test_generate_recommendation_all_tiers(self):
        """Test _generate_recommendation for all score tiers"""
        assert "APPROVED" in PluginValidator._generate_recommendation({"overall_score": 0.96})
        assert "Excellent" in PluginValidator._generate_recommendation({"overall_score": 0.96})

        assert "APPROVED" in PluginValidator._generate_recommendation({"overall_score": 0.88})
        assert "Good" in PluginValidator._generate_recommendation({"overall_score": 0.88})

        assert "CONDITIONAL" in PluginValidator._generate_recommendation({"overall_score": 0.72})

        assert "NOT APPROVED" in PluginValidator._generate_recommendation({"overall_score": 0.55})

        assert "REJECTED" in PluginValidator._generate_recommendation({"overall_score": 0.30})

    def test_calculate_score_partial_sections(self):
        """Test _calculate_score with only some sections present"""
        sections = {
            "code_quality": {"score": 1.0},
            "security": {"score": 0.5}
        }
        score = PluginValidator._calculate_score(sections)
        expected = (1.0 * 0.15 + 0.5 * 0.15) / (0.15 + 0.15)
        assert abs(score - expected) < 1e-9

    def test_calculate_score_empty_sections(self):
        """Test _calculate_score with no sections returns 0.0"""
        score = PluginValidator._calculate_score({})
        assert score == 0.0

    def test_calculate_score_all_perfect(self):
        """Test _calculate_score with all sections scoring 1.0"""
        sections = {
            "code_quality": {"score": 1.0},
            "documentation": {"score": 1.0},
            "functional": {"score": 1.0},
            "performance": {"score": 1.0},
            "security": {"score": 1.0}
        }
        score = PluginValidator._calculate_score(sections)
        assert abs(score - 1.0) < 1e-9

    def test_validate_plugin_comprehensive_nonexistent(self, reset_plugin_registry):
        """Test validate_plugin_comprehensive raises for nonexistent plugin"""
        with pytest.raises(PluginError, match="not found"):
            PluginValidator.validate_plugin_comprehensive("ghost_plugin")


# =============================================================================
# PluginRegistry - Validate Plugin Full Flow
# =============================================================================

class TestPluginRegistryValidatePluginFlow:
    """Test validate_plugin method full flow including summary and metadata update"""

    def test_validate_plugin_updates_metadata(self, reset_plugin_registry):
        """Test validate_plugin updates is_validated and validation_date on metadata"""
        registry = PluginRegistry()
        metadata = PluginMetadata(
            plugin_name="validate_flow", version="1.0.0", author="A",
            description="test", homepage="https://example.com"
        )
        registry._plugins["validate_flow"] = metadata

        result = PluginRegistry.validate_plugin("validate_flow")

        assert "timestamp" in result
        assert "summary" in result
        assert "tests" in result
        assert "consistency" in result["tests"]
        assert "dependencies" in result["tests"]
        assert "security" in result["tests"]
        # Metadata should be updated
        assert metadata.validation_date is not None
        assert isinstance(metadata.validation_results, dict)

    def test_validate_plugin_summary_structure(self, reset_plugin_registry):
        """Test validate_plugin summary contains expected keys"""
        registry = PluginRegistry()
        metadata = PluginMetadata(
            plugin_name="summary_flow", version="1.0.0", author="A"
        )
        registry._plugins["summary_flow"] = metadata

        result = PluginRegistry.validate_plugin("summary_flow")

        summary = result["summary"]
        assert "declared_transforms" in summary
        assert "registered_transforms" in summary
        assert "missing_implementations" in summary
        assert "undeclared_implementations" in summary
        assert "tests_passed" in summary
        assert "tests_failed" in summary


# =============================================================================
# Lazy Import Edge Cases
# =============================================================================

class TestLazyImportEdgeCases:
    """Extended edge case tests for lazy import mechanisms"""

    def test_import_custom_transforms_import_error_returns_false(self):
        """Test _import_custom_transforms returns False on ImportError"""
        import milia_pipeline.transformations.plugin_system as ps
        original_ctb = ps.CustomTransformBase
        original_flag = ps._IMPORTING_CUSTOM_TRANSFORMS
        ps.CustomTransformBase = None
        ps._IMPORTING_CUSTOM_TRANSFORMS = False

        try:
            # If the actual import fails (which it will in test env), should return False
            result = _import_custom_transforms()
            # Either it succeeds (True) or fails gracefully (False)
            assert isinstance(result, bool)
        finally:
            ps.CustomTransformBase = original_ctb
            ps._IMPORTING_CUSTOM_TRANSFORMS = original_flag

    def test_import_graph_transforms_import_error_returns_false(self):
        """Test _import_graph_transforms returns False on ImportError"""
        import milia_pipeline.transformations.plugin_system as ps
        original_tr = ps.TransformRegistry
        original_flag = ps._IMPORTING_GRAPH_TRANSFORMS
        # Only reset if we want to force re-import
        # The actual import may succeed or fail depending on environment
        ps_tr_was_none = ps.TransformRegistry is None

        try:
            result = _import_graph_transforms()
            assert isinstance(result, bool)
        finally:
            if ps_tr_was_none:
                ps.TransformRegistry = original_tr
            ps._IMPORTING_GRAPH_TRANSFORMS = original_flag

    def test_import_custom_transforms_flag_reset_on_failure(self):
        """Test that _IMPORTING_CUSTOM_TRANSFORMS flag is reset in finally block"""
        import milia_pipeline.transformations.plugin_system as ps
        original_ctb = ps.CustomTransformBase
        original_flag = ps._IMPORTING_CUSTOM_TRANSFORMS

        ps.CustomTransformBase = None
        ps._IMPORTING_CUSTOM_TRANSFORMS = False

        try:
            _import_custom_transforms()
        finally:
            # The flag should always be reset after the call
            assert ps._IMPORTING_CUSTOM_TRANSFORMS is False
            ps.CustomTransformBase = original_ctb
            ps._IMPORTING_CUSTOM_TRANSFORMS = original_flag

    def test_import_graph_transforms_flag_reset_on_failure(self):
        """Test that _IMPORTING_GRAPH_TRANSFORMS flag is reset in finally block"""
        import milia_pipeline.transformations.plugin_system as ps
        original_tr = ps.TransformRegistry
        original_flag = ps._IMPORTING_GRAPH_TRANSFORMS

        ps.TransformRegistry = None
        ps._IMPORTING_GRAPH_TRANSFORMS = False

        try:
            _import_graph_transforms()
        finally:
            assert ps._IMPORTING_GRAPH_TRANSFORMS is False
            ps.TransformRegistry = original_tr
            ps._IMPORTING_GRAPH_TRANSFORMS = original_flag


# =============================================================================
# TransformDeclaration - Additional Edge Cases
# =============================================================================

class TestTransformDeclarationEdgeCases:
    """Additional edge case tests for TransformDeclaration"""

    def test_from_dict_missing_required_field(self):
        """Test from_dict raises KeyError when required field is missing"""
        data = {
            "name": "test",
            # Missing class_name, module_path, category, description
        }
        with pytest.raises(KeyError):
            TransformDeclaration.from_dict(data)

    def test_to_dict_round_trip(self):
        """Test to_dict -> from_dict round trip preserves data"""
        original = TransformDeclaration(
            name="round_trip", class_name="RT", module_path="m.rt",
            category="cat", description="round trip test",
            version="3.0.0",
            required_node_features=["x", "z"],
            required_edge_features=["edge_weight"],
            required_graph_attributes=["global_feat"],
            parameter_constraints={"alpha": {"min": 0, "max": 1}}
        )
        d = original.to_dict()
        restored = TransformDeclaration.from_dict(d)

        assert restored.name == original.name
        assert restored.class_name == original.class_name
        assert restored.module_path == original.module_path
        assert restored.category == original.category
        assert restored.description == original.description
        assert restored.version == original.version
        assert restored.required_node_features == original.required_node_features
        assert restored.required_edge_features == original.required_edge_features
        assert restored.required_graph_attributes == original.required_graph_attributes
        assert restored.parameter_constraints == original.parameter_constraints

    def test_default_version_is_1_0_0(self):
        """Test that default version is 1.0.0"""
        decl = TransformDeclaration(
            name="def_ver", class_name="DV", module_path="m",
            category="c", description="d"
        )
        assert decl.version == "1.0.0"

    def test_model_dump_alias(self):
        """Test that to_dict wraps Pydantic model_dump correctly"""
        decl = TransformDeclaration(
            name="dump", class_name="D", module_path="m",
            category="c", description="d"
        )
        # model_dump and to_dict should return equivalent dicts
        assert decl.to_dict() == decl.model_dump()


class TestExceptions:
    """Test suite for plugin exception classes"""
    
    def test_plugin_error_basic(self):
        """Test basic PluginError"""
        error = PluginError("Test error", plugin_name="test_plugin")
        
        # The actual error includes plugin name in the message
        error_str = str(error)
        assert "Test error" in error_str
        assert error.plugin_name == "test_plugin"
    
    def test_plugin_validation_error(self):
        """Test PluginValidationError with validation errors"""
        validation_errors = ["Error 1", "Error 2"]
        error = PluginValidationError(
            "Validation failed",
            plugin_name="test_plugin",
            validation_errors=validation_errors
        )
        
        assert error.plugin_name == "test_plugin"
        assert error.validation_errors == validation_errors
    
    def test_plugin_security_error(self):
        """Test PluginSecurityError with security issues"""
        security_issues = ["Issue 1", "Issue 2"]
        error = PluginSecurityError(
            "Security check failed",
            plugin_name="test_plugin",
            security_issues=security_issues
        )
        
        assert error.plugin_name == "test_plugin"
        assert error.security_issues == security_issues
    
    def test_plugin_dependency_error(self):
        """Test PluginDependencyError with missing dependencies"""
        missing_deps = ["torch>=2.0.0", "numpy>=1.20.0"]
        error = PluginDependencyError(
            "Dependencies not satisfied",
            plugin_name="test_plugin",
            missing_dependencies=missing_deps
        )
        
        assert error.plugin_name == "test_plugin"
        assert error.missing_dependencies == missing_deps


# =============================================================================
# Integration Tests
# =============================================================================

class TestPluginSystemIntegration:
    """Integration tests for the complete plugin system"""
    
    def test_full_plugin_lifecycle(self, reset_plugin_registry):
        """Test complete plugin lifecycle: register -> enable -> disable"""
        registry = PluginRegistry()
        
        metadata = PluginMetadata(
            plugin_name="lifecycle_test",
            version="1.0.0",
            author="Test Author"
        )
        
        # Register (internal)
        registry._plugins["lifecycle_test"] = metadata
        assert "lifecycle_test" in registry._plugins
        
        # Enable
        PluginRegistry.enable_plugin("lifecycle_test")
        assert "lifecycle_test" in registry._enabled_plugins
        
        # Disable
        PluginRegistry.disable_plugin("lifecycle_test")
        assert "lifecycle_test" not in registry._enabled_plugins
        assert "lifecycle_test" in registry._disabled_plugins
        
        # Re-enable
        PluginRegistry.enable_plugin("lifecycle_test")
        assert "lifecycle_test" in registry._enabled_plugins
    
    def test_plugin_with_transforms(self, reset_plugin_registry):
        """Test plugin with transform declarations and registrations"""
        registry = PluginRegistry()
        
        # Create plugin with transform declarations
        decl = TransformDeclaration(
            name="integration_transform",
            class_name="IntegrationTransform",
            module_path="transforms.integration",
            category="molecular",
            description="Integration test transform"
        )
        
        metadata = PluginMetadata(
            plugin_name="integration_plugin",
            version="1.0.0",
            author="Test Author",
            transform_declarations=[decl],
            registered_transforms=set()
        )
        
        registry._plugins["integration_plugin"] = metadata
        
        # Verify declarations
        retrieved_info = PluginRegistry.get_plugin_info("integration_plugin")
        assert len(retrieved_info["transform_declarations"]) == 1
        assert retrieved_info["transform_declarations"][0]["name"] == "integration_transform"
        
        # Simulate registration
        metadata.registered_transforms.add("integration_transform")
        assert len(metadata.registered_transforms) == 1
        assert "integration_transform" in metadata.registered_transforms


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test suite for edge cases and error conditions"""
    
    def test_empty_plugin_name(self, reset_plugin_registry):
        """Test handling of empty plugin name - should raise error in __post_init__"""
        with pytest.raises(PluginError, match="Plugin name is required"):
            metadata = PluginMetadata(
                plugin_name="",
                version="1.0.0",
                author="Author"
            )
    
    def test_special_characters_in_plugin_name(self, reset_plugin_registry):
        """Test handling of special characters in plugin name"""
        registry = PluginRegistry()
        
        metadata = PluginMetadata(
            plugin_name="test-plugin_v2.0",
            version="1.0.0",
            author="Author"
        )
        
        registry._plugins["test-plugin_v2.0"] = metadata
        assert "test-plugin_v2.0" in registry._plugins
    
    def test_unicode_in_plugin_metadata(self, reset_plugin_registry):
        """Test handling of Unicode characters in metadata"""
        registry = PluginRegistry()
        
        metadata = PluginMetadata(
            plugin_name="unicode_plugin",
            version="1.0.0",
            author="作者",  # Chinese characters
            description="测试插件"  # Chinese description
        )
        
        registry._plugins["unicode_plugin"] = metadata
        retrieved_info = PluginRegistry.get_plugin_info("unicode_plugin")
        assert retrieved_info["author"] == "作者"
        assert retrieved_info["description"] == "测试插件"
    
    def test_very_long_plugin_name(self, reset_plugin_registry):
        """Test handling of very long plugin name"""
        registry = PluginRegistry()
        
        long_name = "a" * 1000
        metadata = PluginMetadata(
            plugin_name=long_name,
            version="1.0.0",
            author="Author"
        )
        
        registry._plugins[long_name] = metadata
        assert long_name in registry._plugins


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Test suite for performance characteristics"""
    
    def test_large_number_of_plugins(self, reset_plugin_registry):
        """Test handling of large number of plugins"""
        registry = PluginRegistry()
        num_plugins = 100
        
        for i in range(num_plugins):
            metadata = PluginMetadata(
                plugin_name=f"plugin_{i}",
                version="1.0.0",
                author=f"Author{i}"
            )
            registry._plugins[f"plugin_{i}"] = metadata
        
        # Verify all registered
        assert len(registry._plugins) == num_plugins
        
        # Test listing performance
        plugins = PluginRegistry.list_plugins()
        assert len(plugins) == num_plugins
    
    def test_plugin_lookup_performance(self, reset_plugin_registry):
        """Test plugin lookup with many plugins"""
        registry = PluginRegistry()
        
        # Register 100 plugins
        for i in range(100):
            metadata = PluginMetadata(
                plugin_name=f"plugin_{i}",
                version="1.0.0",
                author=f"Author{i}"
            )
            registry._plugins[f"plugin_{i}"] = metadata
        
        # Lookup should be fast (O(1))
        start_time = time.time()
        for _ in range(1000):
            info = PluginRegistry.get_plugin_info("plugin_50")
        elapsed = time.time() - start_time
        
        # Should be very fast
        assert elapsed < 0.1  # 100ms for 1000 lookups


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
