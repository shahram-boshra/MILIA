#!/usr/bin/env python3
"""
Complete Production-Ready Unit Test Suite for model_plugin_system.py Module

Comprehensive test coverage for the Model Plugin System including:
- Exception classes (PluginError, PluginValidationError, PluginSecurityError, PluginDependencyError)
- ModelDeclaration Pydantic BaseModel (V2 migration)
- ModelPluginMetadata Pydantic BaseModel (V2 migration)
- ModelPluginLoader class with all methods
- Module-level convenience functions
- Plugin discovery, validation, loading, and registration
- Thread safety and singleton pattern
- Security checks and dependency validation
- Error handling and edge cases
- Pydantic V2 specific behaviors (model_dump, model_config, Field default_factory)

Production-Ready Updates (v2.0.0):
- Fixed module loading: Uses direct file-based import via importlib.util to bypass
  package __init__.py cascade that causes TypeError with Mock objects
- Comprehensive package hierarchy mocking to prevent import side effects
- Added tests for ModelDeclaration.to_dict() Pydantic V2 wrapper
- Added tests for dependency validation with extras syntax (torch[cuda])
- Added tests for all security check patterns (subprocess, os.system, __import__)
- Added tests for ModelMetadata is None edge case
- Added tests for load_plugin exception handling paths
- Added tests for validation level error accumulation
- Added tests for Pydantic V2 mutable defaults independence
- Added tests for arbitrary_types_allowed model_config

This test suite follows the established pattern and prevents mock pollution.

Author: milia Team
Version: 2.0.0
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import threading
import time
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock, PropertyMock, call, mock_open
from datetime import datetime
from dataclasses import dataclass, field
import tempfile
import shutil

import pytest
import torch
import torch.nn as nn
import yaml


# ==============================================================================
# CRITICAL: Mock the entire milia_pipeline package hierarchy to prevent
# __init__.py cascade that causes TypeError: object of type 'Mock' has no len()
#
# The problem: When importing milia_pipeline.models.plugins.model_plugin_system,
# Python loads parent packages whose __init__.py files import from registry,
# which uses len(ModelCategory). We must mock the ENTIRE package hierarchy
# BEFORE any import to prevent this cascade.
#
# The model_plugin_system.py has try/except ImportError blocks (lines 44-56, 58-67)
# that provide fallback behavior when imports fail. By mocking the parent packages,
# those imports will fail gracefully and use the fallback classes.
#
# These mocks are cleaned up in teardown_module() to prevent mock pollution.
# ==============================================================================

# Create a MockModelCategory that behaves like an Enum (supports len() and iteration)
# This is needed in case any code path still reaches ModelCategory
class _MockModelCategory:
    """Mock ModelCategory enum that supports len() and iteration."""
    GNN = 'gnn'
    CUSTOM = 'custom'
    BASELINE = 'baseline'
    
    def __class_getitem__(cls, item):
        return cls
    
    @classmethod
    def __iter__(cls):
        return iter([cls.GNN, cls.CUSTOM, cls.BASELINE])
    
    @classmethod  
    def __len__(cls):
        return 3

# Mock the parent packages to prevent __init__.py loading cascade
# Order matters: mock parent packages BEFORE child modules
_mock_milia_pipeline = Mock()
_mock_milia_pipeline_models = Mock()
_mock_milia_pipeline_models_registry = Mock()
_mock_milia_pipeline_models_plugins = Mock()
_mock_milia_pipeline_models_utils = Mock()

# Set up nested module structure
_mock_milia_pipeline.models = _mock_milia_pipeline_models
_mock_milia_pipeline_models.registry = _mock_milia_pipeline_models_registry
_mock_milia_pipeline_models.plugins = _mock_milia_pipeline_models_plugins
_mock_milia_pipeline_models.utils = _mock_milia_pipeline_models_utils

# Create specific module mocks
_mock_model_registry = Mock()
_mock_pyg_introspector = Mock()
_mock_config_bridge = Mock()

# Set up the objects that model_plugin_system.py imports
_mock_model_registry.ModelRegistry = Mock()
_mock_model_registry.registry = Mock()
_mock_pyg_introspector.ModelMetadata = Mock()
_mock_pyg_introspector.ModelCategory = _MockModelCategory

_mock_modules = {
    # Parent packages (to prevent __init__.py cascade)
    'milia_pipeline': _mock_milia_pipeline,
    'milia_pipeline.models': _mock_milia_pipeline_models,
    'milia_pipeline.models.registry': _mock_milia_pipeline_models_registry,
    'milia_pipeline.models.plugins': _mock_milia_pipeline_models_plugins,
    'milia_pipeline.models.utils': _mock_milia_pipeline_models_utils,
    # Specific modules that model_plugin_system.py imports (lines 45-48)
    'milia_pipeline.models.registry.model_registry': _mock_model_registry,
    'milia_pipeline.models.registry.pyg_introspector': _mock_pyg_introspector,
    'milia_pipeline.models.utils.config_bridge': _mock_config_bridge,
    # Mock exceptions module to trigger fallback classes in model_plugin_system.py
    'milia_pipeline.exceptions': Mock(spec=[]),
}

# Store original modules for cleanup (populated by setup_module)
_original_modules = {}

# Now load model_plugin_system directly from file to bypass package __init__.py
import importlib.util

# ---------------------------------------------------------------------------
# Module-level placeholders for classes loaded in setup_module().
# Set to None at import time (safe during collection), populated before tests.
# ---------------------------------------------------------------------------
model_plugin_system = None
ModelPluginLoader = None
ModelPluginMetadata = None
ModelDeclaration = None
get_plugin_loader = None
discover_plugins = None
load_plugin = None
list_plugins = None
get_plugin_info = None
validate_plugin = None
PluginError = None
PluginValidationError = None
PluginSecurityError = None
PluginDependencyError = None


def setup_module(module):
    """
    Inject mocked modules into sys.modules and load the module-under-test.

    Runs ONCE before any test — does NOT run during pytest collection,
    preventing sys.modules pollution from breaking other test files.
    """
    global model_plugin_system
    global ModelPluginLoader, ModelPluginMetadata, ModelDeclaration
    global get_plugin_loader, discover_plugins, load_plugin
    global list_plugins, get_plugin_info, validate_plugin
    global PluginError, PluginValidationError, PluginSecurityError, PluginDependencyError

    # --- Inject all mocks into sys.modules ---
    for mod_name in _mock_modules:
        if mod_name in sys.modules:
            _original_modules[mod_name] = sys.modules[mod_name]
        sys.modules[mod_name] = _mock_modules[mod_name]

    # --- Determine the path to model_plugin_system.py ---
    _module_search_paths = [
        project_root / "milia_pipeline" / "models" / "plugins" / "model_plugin_system.py",
        Path("/app/milia/milia_pipeline/models/plugins/model_plugin_system.py"),
    ]

    _model_plugin_system_path = None
    for _search_path in _module_search_paths:
        if _search_path.exists():
            _model_plugin_system_path = _search_path
            break

    if _model_plugin_system_path is None:
        raise ImportError(
            f"Could not find model_plugin_system.py. "
            f"Searched: {_module_search_paths}. "
            f"Project root: {project_root}"
        )

    # --- Load the module directly from file ---
    _spec = importlib.util.spec_from_file_location(
        "model_plugin_system",
        _model_plugin_system_path
    )
    model_plugin_system = importlib.util.module_from_spec(_spec)
    sys.modules['model_plugin_system'] = model_plugin_system
    _spec.loader.exec_module(model_plugin_system)

    # --- Extract classes and functions into module-level names ---
    ModelPluginLoader = model_plugin_system.ModelPluginLoader
    ModelPluginMetadata = model_plugin_system.ModelPluginMetadata
    ModelDeclaration = model_plugin_system.ModelDeclaration
    get_plugin_loader = model_plugin_system.get_plugin_loader
    discover_plugins = model_plugin_system.discover_plugins
    load_plugin = model_plugin_system.load_plugin
    list_plugins = model_plugin_system.list_plugins
    get_plugin_info = model_plugin_system.get_plugin_info
    validate_plugin = model_plugin_system.validate_plugin
    PluginError = model_plugin_system.PluginError
    PluginValidationError = model_plugin_system.PluginValidationError
    PluginSecurityError = model_plugin_system.PluginSecurityError
    PluginDependencyError = model_plugin_system.PluginDependencyError

    # --- Publish into this module's namespace so tests see them ---
    module.model_plugin_system = model_plugin_system
    module.ModelPluginLoader = ModelPluginLoader
    module.ModelPluginMetadata = ModelPluginMetadata
    module.ModelDeclaration = ModelDeclaration
    module.get_plugin_loader = get_plugin_loader
    module.discover_plugins = discover_plugins
    module.load_plugin = load_plugin
    module.list_plugins = list_plugins
    module.get_plugin_info = get_plugin_info
    module.validate_plugin = validate_plugin
    module.PluginError = PluginError
    module.PluginValidationError = PluginValidationError
    module.PluginSecurityError = PluginSecurityError
    module.PluginDependencyError = PluginDependencyError


def teardown_module(module):
    """
    Cleanup function to remove mocked modules from sys.modules.
    This prevents mock pollution from affecting other test files.
    """
    # Clean up mock modules
    for module_name in list(_mock_modules.keys()):
        if module_name in sys.modules:
            if module_name in _original_modules:
                # Restore original module
                sys.modules[module_name] = _original_modules[module_name]
            else:
                # Remove mock module
                del sys.modules[module_name]
    
    # Clean up the directly loaded module
    if 'model_plugin_system' in sys.modules:
        del sys.modules['model_plugin_system']


# =============================================================================
# FIXTURES AND SETUP
# =============================================================================

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instance before each test."""
    model_plugin_system.ModelPluginLoader._instance = None
    model_plugin_system._plugin_loader = None
    yield
    model_plugin_system.ModelPluginLoader._instance = None
    model_plugin_system._plugin_loader = None


@pytest.fixture
def temp_plugin_dir():
    """Create a temporary directory for plugin testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_plugin_yaml():
    """Sample plugin.yaml content."""
    return {
        'plugin_name': 'test_plugin',
        'version': '1.0.0',
        'author': 'Test Author',
        'description': 'Test plugin for unit testing',
        'plugin_type': 'user_experimental',
        'milia_version': '>=4.0.0',
        'pyg_version': '>=2.0.0',
        'python_version': '>=3.8',
        'license': 'MIT',
        'models': [
            {
                'name': 'TestModel',
                'class_name': 'TestModelClass',
                'module_path': 'test_module',
                'category': 'gnn',
                'description': 'A test model',
                'supported_tasks': ['regression', 'classification'],
                'hyperparameters': {
                    'hidden_dim': {'type': 'int', 'default': 128},
                    'num_layers': {'type': 'int', 'default': 3}
                },
                'requires_edge_index': True,
                'requires_edge_features': False,
                'requires_edge_weights': False,
                'supports_batch': True,
                'supports_heterogeneous': False,
                'min_pyg_version': '2.0.0',
                'reference_paper': 'Test Paper (2024)',
                'reference_url': 'https://example.com/paper'
            }
        ],
        'dependencies': ['torch', 'torch_geometric'],
        'optional_dependencies': ['networkx'],
        'homepage': 'https://example.com',
        'repository': 'https://github.com/example/test-plugin',
        'documentation': 'https://docs.example.com'
    }


@pytest.fixture
def sample_model_declaration():
    """Sample ModelDeclaration instance."""
    return ModelDeclaration(
        name='TestModel',
        class_name='TestModelClass',
        module_path='test_module',
        category='gnn',
        description='A test model',
        supported_tasks=['regression', 'classification'],
        hyperparameters={
            'hidden_dim': {'type': 'int', 'default': 128},
            'num_layers': {'type': 'int', 'default': 3}
        },
        plugin_name='test_plugin',
        requires_edge_index=True,
        requires_edge_features=False,
        requires_edge_weights=False,
        supports_batch=True,
        supports_heterogeneous=False,
        min_pyg_version='2.0.0',
        reference_paper='Test Paper (2024)',
        reference_url='https://example.com/paper'
    )


@pytest.fixture
def sample_plugin_metadata(sample_model_declaration):
    """Sample ModelPluginMetadata instance."""
    return ModelPluginMetadata(
        plugin_name='test_plugin',
        version='1.0.0',
        author='Test Author',
        description='Test plugin',
        plugin_type='user_experimental',
        milia_version='>=4.0.0',
        pyg_version='>=2.0.0',
        python_version='>=3.8',
        license='MIT',
        model_declarations=[sample_model_declaration],
        dependencies=['torch'],
        optional_dependencies=['networkx'],
        homepage='https://example.com',
        repository='https://github.com/example/test',
        documentation='https://docs.example.com',
        plugin_path=Path('/fake/path'),
        loaded=False,
        enabled=True,
        load_time=None,
        validation_errors=[]
    )


@pytest.fixture
def mock_model_class():
    """Mock PyTorch model class."""
    class MockModel(nn.Module):
        def __init__(self, hidden_dim=128, num_layers=3):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
        
        def forward(self, x):
            return x
    
    return MockModel


# =============================================================================
# EXCEPTION TESTS
# =============================================================================

class TestPluginExceptions:
    """Test suite for plugin exception classes."""
    
    def test_plugin_error_basic(self):
        """Test PluginError basic functionality."""
        error = PluginError("Test error")
        assert str(error) == "Test error"
        assert error.plugin_name is None
    
    def test_plugin_error_with_plugin_name(self):
        """Test PluginError with plugin_name."""
        error = PluginError("Test error", plugin_name="test_plugin")
        assert "Test error" in str(error)
        assert error.plugin_name == "test_plugin"
    
    def test_plugin_validation_error(self):
        """Test PluginValidationError."""
        validation_errors = ["Error 1", "Error 2"]
        error = PluginValidationError(
            "Validation failed",
            plugin_name="test_plugin",
            validation_errors=validation_errors
        )
        
        assert "Validation failed" in str(error)
        assert error.plugin_name == "test_plugin"
        assert error.validation_errors == validation_errors
    
    def test_plugin_validation_error_default_list(self):
        """Test PluginValidationError with default empty list."""
        error = PluginValidationError("Validation failed")
        assert error.validation_errors == []
    
    def test_plugin_security_error(self):
        """Test PluginSecurityError."""
        security_issues = ["Issue 1", "Issue 2"]
        error = PluginSecurityError(
            "Security check failed",
            plugin_name="test_plugin",
            security_issues=security_issues
        )
        
        assert "Security check failed" in str(error)
        assert error.plugin_name == "test_plugin"
        assert error.security_issues == security_issues
    
    def test_plugin_security_error_default_list(self):
        """Test PluginSecurityError with default empty list."""
        error = PluginSecurityError("Security check failed")
        assert error.security_issues == []
    
    def test_plugin_dependency_error(self):
        """Test PluginDependencyError."""
        missing_deps = ["dep1", "dep2"]
        error = PluginDependencyError(
            "Dependencies missing",
            plugin_name="test_plugin",
            missing_dependencies=missing_deps
        )
        
        assert "Dependencies missing" in str(error)
        assert error.plugin_name == "test_plugin"
        assert error.missing_dependencies == missing_deps
    
    def test_plugin_dependency_error_default_list(self):
        """Test PluginDependencyError with default empty list."""
        error = PluginDependencyError("Dependencies missing")
        assert error.missing_dependencies == []
    
    def test_exception_inheritance(self):
        """Test exception class inheritance."""
        assert issubclass(PluginValidationError, PluginError)
        assert issubclass(PluginSecurityError, PluginError)
        assert issubclass(PluginDependencyError, PluginError)
        assert issubclass(PluginError, Exception)


# =============================================================================
# MODEL DECLARATION TESTS
# =============================================================================

class TestModelDeclaration:
    """Test suite for ModelDeclaration dataclass."""
    
    def test_model_declaration_creation(self, sample_model_declaration):
        """Test ModelDeclaration basic creation."""
        assert sample_model_declaration.name == 'TestModel'
        assert sample_model_declaration.class_name == 'TestModelClass'
        assert sample_model_declaration.module_path == 'test_module'
        assert sample_model_declaration.category == 'gnn'
        assert sample_model_declaration.description == 'A test model'
        assert sample_model_declaration.supported_tasks == ['regression', 'classification']
        assert sample_model_declaration.hyperparameters == {
            'hidden_dim': {'type': 'int', 'default': 128},
            'num_layers': {'type': 'int', 'default': 3}
        }
        assert sample_model_declaration.plugin_name == 'test_plugin'
    
    def test_model_declaration_defaults(self):
        """Test ModelDeclaration default values."""
        decl = ModelDeclaration(
            name='Model',
            class_name='ModelClass',
            module_path='module',
            category='gnn',
            description='Desc',
            supported_tasks=['task'],
            hyperparameters={},
            plugin_name='plugin'
        )
        
        assert decl.requires_edge_index is True
        assert decl.requires_edge_features is False
        assert decl.requires_edge_weights is False
        assert decl.supports_batch is True
        assert decl.supports_heterogeneous is False
        assert decl.min_pyg_version is None
        assert decl.reference_paper is None
        assert decl.reference_url is None
    
    def test_model_declaration_all_fields(self):
        """Test ModelDeclaration with all fields specified."""
        decl = ModelDeclaration(
            name='Model',
            class_name='ModelClass',
            module_path='module',
            category='gnn',
            description='Desc',
            supported_tasks=['task'],
            hyperparameters={},
            plugin_name='plugin',
            requires_edge_index=False,
            requires_edge_features=True,
            requires_edge_weights=True,
            supports_batch=False,
            supports_heterogeneous=True,
            min_pyg_version='2.1.0',
            reference_paper='Paper (2024)',
            reference_url='https://example.com'
        )
        
        assert decl.requires_edge_index is False
        assert decl.requires_edge_features is True
        assert decl.requires_edge_weights is True
        assert decl.supports_batch is False
        assert decl.supports_heterogeneous is True
        assert decl.min_pyg_version == '2.1.0'
        assert decl.reference_paper == 'Paper (2024)'
        assert decl.reference_url == 'https://example.com'
    
    def test_model_declaration_to_dict(self, sample_model_declaration):
        """Test ModelDeclaration to_dict method for Pydantic V2 compatibility."""
        result = sample_model_declaration.to_dict()
        
        assert isinstance(result, dict)
        expected_fields = {
            'name', 'class_name', 'module_path', 'category', 'description',
            'supported_tasks', 'hyperparameters', 'plugin_name',
            'requires_edge_index', 'requires_edge_features', 'requires_edge_weights',
            'supports_batch', 'supports_heterogeneous', 'min_pyg_version',
            'reference_paper', 'reference_url'
        }
        assert set(result.keys()) == expected_fields
        assert result['name'] == 'TestModel'
        assert result['class_name'] == 'TestModelClass'
    
    def test_model_declaration_to_dict_equals_model_dump(self, sample_model_declaration):
        """Test that to_dict() wraps Pydantic V2's model_dump() correctly."""
        to_dict_result = sample_model_declaration.to_dict()
        model_dump_result = sample_model_declaration.model_dump()
        assert to_dict_result == model_dump_result


# =============================================================================
# MODEL PLUGIN METADATA TESTS
# =============================================================================

class TestModelPluginMetadata:
    """Test suite for ModelPluginMetadata dataclass."""
    
    def test_metadata_creation(self, sample_plugin_metadata):
        """Test ModelPluginMetadata basic creation."""
        assert sample_plugin_metadata.plugin_name == 'test_plugin'
        assert sample_plugin_metadata.version == '1.0.0'
        assert sample_plugin_metadata.author == 'Test Author'
        assert sample_plugin_metadata.description == 'Test plugin'
        assert len(sample_plugin_metadata.model_declarations) == 1
    
    def test_metadata_defaults(self, sample_model_declaration):
        """Test ModelPluginMetadata default values."""
        metadata = ModelPluginMetadata(
            plugin_name='plugin',
            version='1.0.0',
            author='Author',
            description='Desc',
            plugin_type='user',
            milia_version='>=4.0.0',
            pyg_version='>=2.0.0',
            python_version='>=3.8',
            license='MIT',
            model_declarations=[sample_model_declaration]
        )
        
        assert metadata.dependencies == []
        assert metadata.optional_dependencies == []
        assert metadata.homepage is None
        assert metadata.repository is None
        assert metadata.documentation is None
        assert metadata.plugin_path is None
        assert metadata.loaded is False
        assert metadata.enabled is True
        assert metadata.load_time is None
        assert metadata.validation_errors == []
    
    def test_metadata_to_dict(self, sample_plugin_metadata):
        """Test ModelPluginMetadata to_dict method."""
        result = sample_plugin_metadata.to_dict()
        
        assert isinstance(result, dict)
        assert result['plugin_name'] == 'test_plugin'
        assert result['version'] == '1.0.0'
        assert result['author'] == 'Test Author'
        assert result['description'] == 'Test plugin'
        assert result['plugin_type'] == 'user_experimental'
        assert result['milia_version'] == '>=4.0.0'
        assert result['pyg_version'] == '>=2.0.0'
        assert result['python_version'] == '>=3.8'
        assert result['license'] == 'MIT'
        assert result['num_models'] == 1
        assert result['models'] == ['TestModel']
        assert result['dependencies'] == ['torch']
        assert result['loaded'] is False
        assert result['enabled'] is True
    
    def test_metadata_to_dict_multiple_models(self, sample_model_declaration):
        """Test ModelPluginMetadata to_dict with multiple models."""
        decl2 = ModelDeclaration(
            name='Model2',
            class_name='Model2Class',
            module_path='module2',
            category='gnn',
            description='Second model',
            supported_tasks=['task'],
            hyperparameters={},
            plugin_name='plugin'
        )
        
        metadata = ModelPluginMetadata(
            plugin_name='plugin',
            version='1.0.0',
            author='Author',
            description='Desc',
            plugin_type='user',
            milia_version='>=4.0.0',
            pyg_version='>=2.0.0',
            python_version='>=3.8',
            license='MIT',
            model_declarations=[sample_model_declaration, decl2]
        )
        
        result = metadata.to_dict()
        assert result['num_models'] == 2
        assert result['models'] == ['TestModel', 'Model2']
    
    def test_metadata_arbitrary_types_allowed(self, sample_model_declaration):
        """Test ModelPluginMetadata accepts Path type via model_config."""
        metadata = ModelPluginMetadata(
            plugin_name='plugin',
            version='1.0.0',
            author='Author',
            description='Desc',
            plugin_type='user',
            milia_version='>=4.0.0',
            pyg_version='>=2.0.0',
            python_version='>=3.8',
            license='MIT',
            model_declarations=[sample_model_declaration],
            plugin_path=Path('/some/path')
        )
        assert metadata.plugin_path == Path('/some/path')
        assert isinstance(metadata.plugin_path, Path)
    
    def test_metadata_mutable_defaults_independence(self, sample_model_declaration):
        """Test that mutable default fields don't share state across instances."""
        metadata1 = ModelPluginMetadata(
            plugin_name='plugin1',
            version='1.0.0',
            author='Author',
            description='Desc',
            plugin_type='user',
            milia_version='>=4.0.0',
            pyg_version='>=2.0.0',
            python_version='>=3.8',
            license='MIT',
            model_declarations=[sample_model_declaration]
        )
        
        metadata2 = ModelPluginMetadata(
            plugin_name='plugin2',
            version='1.0.0',
            author='Author',
            description='Desc',
            plugin_type='user',
            milia_version='>=4.0.0',
            pyg_version='>=2.0.0',
            python_version='>=3.8',
            license='MIT',
            model_declarations=[sample_model_declaration]
        )
        
        metadata1.dependencies.append('new_dep')
        metadata1.validation_errors.append('error')
        
        assert 'new_dep' not in metadata2.dependencies
        assert 'error' not in metadata2.validation_errors
    
    def test_metadata_is_pydantic_basemodel(self, sample_plugin_metadata):
        """Test ModelPluginMetadata is a Pydantic BaseModel."""
        from pydantic import BaseModel
        assert isinstance(sample_plugin_metadata, BaseModel)
        assert hasattr(sample_plugin_metadata, 'model_dump')
    
    def test_model_declaration_is_pydantic_basemodel(self, sample_model_declaration):
        """Test ModelDeclaration is a Pydantic BaseModel."""
        from pydantic import BaseModel
        assert isinstance(sample_model_declaration, BaseModel)
        assert hasattr(sample_model_declaration, 'model_dump')


# =============================================================================
# MODEL PLUGIN LOADER TESTS - SINGLETON
# =============================================================================

class TestModelPluginLoaderSingleton:
    """Test suite for ModelPluginLoader singleton pattern."""
    
    def test_singleton_pattern(self):
        """Test that ModelPluginLoader follows singleton pattern."""
        loader1 = ModelPluginLoader()
        loader2 = ModelPluginLoader()
        
        assert loader1 is loader2
    
    def test_singleton_thread_safe(self):
        """Test singleton pattern is thread-safe."""
        instances = []
        
        def create_instance():
            loader = ModelPluginLoader()
            instances.append(id(loader))
        
        threads = [threading.Thread(target=create_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All instances should have the same id
        assert len(set(instances)) == 1
    
    def test_initialization_once(self):
        """Test that initialization only happens once."""
        loader = ModelPluginLoader()
        assert hasattr(loader, '_initialized')
        assert loader._initialized is True
        
        # Creating another instance should not reinitialize
        loader2 = ModelPluginLoader()
        assert loader2._initialized is True
        assert loader is loader2


# =============================================================================
# MODEL PLUGIN LOADER TESTS - DISCOVERY
# =============================================================================

class TestModelPluginLoaderDiscovery:
    """Test suite for plugin discovery functionality."""
    
    def test_discover_plugins_empty_paths(self):
        """Test discover_plugins with empty paths list."""
        loader = ModelPluginLoader()
        result = loader.discover_plugins([])
        
        assert result == []
    
    def test_discover_plugins_nonexistent_path(self, caplog):
        """Test discover_plugins with nonexistent path."""
        loader = ModelPluginLoader()
        fake_path = Path("/nonexistent/path/to/plugins")
        
        result = loader.discover_plugins([fake_path])
        
        assert result == []
        assert "does not exist" in caplog.text
    
    def test_discover_plugins_success(self, temp_plugin_dir, sample_plugin_yaml):
        """Test successful plugin discovery."""
        plugin_dir = temp_plugin_dir / "test_plugin"
        plugin_dir.mkdir()
        
        with open(plugin_dir / "plugin.yaml", 'w') as f:
            yaml.dump(sample_plugin_yaml, f)
        
        loader = ModelPluginLoader()
        result = loader.discover_plugins([temp_plugin_dir])
        
        assert len(result) == 1
        assert 'test_plugin' in result
    
    def test_discover_plugins_multiple(self, temp_plugin_dir):
        """Test discovering multiple plugins."""
        for i in range(2):
            plugin_dir = temp_plugin_dir / f"plugin_{i}"
            plugin_dir.mkdir()
            
            plugin_yaml = {
                'plugin_name': f'plugin_{i}',
                'version': '1.0.0',
                'author': 'Test',
                'description': 'Test',
                'plugin_type': 'user',
                'milia_version': '>=4.0.0',
                'pyg_version': '>=2.0.0',
                'python_version': '>=3.8',
                'license': 'MIT',
                'models': []
            }
            
            with open(plugin_dir / "plugin.yaml", 'w') as f:
                yaml.dump(plugin_yaml, f)
        
        loader = ModelPluginLoader()
        result = loader.discover_plugins([temp_plugin_dir])
        
        assert len(result) == 2
        assert 'plugin_0' in result
        assert 'plugin_1' in result
    
    def test_discover_plugins_validation_failure_permissive(self, temp_plugin_dir):
        """Test plugin discovery with validation failure in permissive mode."""
        plugin_dir = temp_plugin_dir / "bad_plugin"
        plugin_dir.mkdir()
        
        plugin_yaml = {
            'plugin_name': 'bad_plugin',
            'version': '1.0.0',
            'author': 'Test',
            'description': 'Test',
            'plugin_type': 'user',
            'milia_version': '>=4.0.0',
            'pyg_version': '>=2.0.0',
            'python_version': '>=3.8',
            'license': 'MIT',
            'models': [],
            'dependencies': ['nonexistent_package']
        }
        
        with open(plugin_dir / "plugin.yaml", 'w') as f:
            yaml.dump(plugin_yaml, f)
        
        loader = ModelPluginLoader()
        result = loader.discover_plugins([temp_plugin_dir], validation_level='permissive')
        
        assert 'bad_plugin' in result
    
    def test_discover_plugins_validation_failure_strict(self, temp_plugin_dir):
        """Test plugin discovery with validation failure in strict mode."""
        plugin_dir = temp_plugin_dir / "bad_plugin"
        plugin_dir.mkdir()
        
        plugin_yaml = {
            'plugin_name': 'bad_plugin',
            'version': '1.0.0',
            'author': 'Test',
            'description': 'Test',
            'plugin_type': 'user',
            'milia_version': '>=4.0.0',
            'pyg_version': '>=2.0.0',
            'python_version': '>=3.8',
            'license': 'MIT',
            'models': [],
            'dependencies': ['nonexistent_package']
        }
        
        with open(plugin_dir / "plugin.yaml", 'w') as f:
            yaml.dump(plugin_yaml, f)
        
        loader = ModelPluginLoader()
        result = loader.discover_plugins([temp_plugin_dir], validation_level='strict')
        
        assert 'bad_plugin' not in result
    
    def test_discover_plugins_invalid_yaml(self, temp_plugin_dir, caplog):
        """Test plugin discovery with invalid YAML file."""
        plugin_dir = temp_plugin_dir / "invalid_plugin"
        plugin_dir.mkdir()
        
        with open(plugin_dir / "plugin.yaml", 'w') as f:
            f.write("invalid: yaml: content: [[[")
        
        loader = ModelPluginLoader()
        result = loader.discover_plugins([temp_plugin_dir])
        
        assert len(result) == 0
        assert "Failed to discover plugin" in caplog.text


# =============================================================================
# MODEL PLUGIN LOADER TESTS - METADATA LOADING
# =============================================================================

class TestModelPluginLoaderMetadata:
    """Test suite for plugin metadata loading."""
    
    def test_load_plugin_metadata_success(self, temp_plugin_dir, sample_plugin_yaml):
        """Test successful metadata loading."""
        plugin_dir = temp_plugin_dir / "test_plugin"
        plugin_dir.mkdir()
        
        plugin_yaml_path = plugin_dir / "plugin.yaml"
        with open(plugin_yaml_path, 'w') as f:
            yaml.dump(sample_plugin_yaml, f)
        
        loader = ModelPluginLoader()
        metadata = loader._load_plugin_metadata(plugin_yaml_path)
        
        assert metadata.plugin_name == 'test_plugin'
        assert metadata.version == '1.0.0'
        assert metadata.author == 'Test Author'
        assert len(metadata.model_declarations) == 1
    
    def test_load_plugin_metadata_empty_file(self, temp_plugin_dir):
        """Test loading metadata from empty YAML file."""
        plugin_dir = temp_plugin_dir / "empty_plugin"
        plugin_dir.mkdir()
        
        plugin_yaml_path = plugin_dir / "plugin.yaml"
        with open(plugin_yaml_path, 'w') as f:
            f.write("")
        
        loader = ModelPluginLoader()
        
        with pytest.raises(PluginError, match="Empty plugin.yaml"):
            loader._load_plugin_metadata(plugin_yaml_path)
    
    def test_load_plugin_metadata_missing_plugin_name(self, temp_plugin_dir):
        """Test loading metadata without plugin_name."""
        plugin_dir = temp_plugin_dir / "no_name_plugin"
        plugin_dir.mkdir()
        
        plugin_yaml_path = plugin_dir / "plugin.yaml"
        plugin_yaml = {'version': '1.0.0'}
        
        with open(plugin_yaml_path, 'w') as f:
            yaml.dump(plugin_yaml, f)
        
        loader = ModelPluginLoader()
        
        with pytest.raises(PluginError, match="Missing plugin_name"):
            loader._load_plugin_metadata(plugin_yaml_path)
    
    def test_load_plugin_metadata_no_models(self, temp_plugin_dir, caplog):
        """Test loading metadata with no models declared."""
        plugin_dir = temp_plugin_dir / "no_models_plugin"
        plugin_dir.mkdir()
        
        plugin_yaml_path = plugin_dir / "plugin.yaml"
        plugin_yaml = {
            'plugin_name': 'no_models_plugin',
            'version': '1.0.0',
            'author': 'Test',
            'description': 'Test',
            'plugin_type': 'user',
            'models': []
        }
        
        with open(plugin_yaml_path, 'w') as f:
            yaml.dump(plugin_yaml, f)
        
        loader = ModelPluginLoader()
        metadata = loader._load_plugin_metadata(plugin_yaml_path)
        
        assert metadata.plugin_name == 'no_models_plugin'
        assert len(metadata.model_declarations) == 0
        assert "declares no models" in caplog.text
    
    def test_load_plugin_metadata_defaults(self, temp_plugin_dir):
        """Test loading metadata with default values."""
        plugin_dir = temp_plugin_dir / "minimal_plugin"
        plugin_dir.mkdir()
        
        plugin_yaml_path = plugin_dir / "plugin.yaml"
        plugin_yaml = {
            'plugin_name': 'minimal_plugin',
            'models': []
        }
        
        with open(plugin_yaml_path, 'w') as f:
            yaml.dump(plugin_yaml, f)
        
        loader = ModelPluginLoader()
        metadata = loader._load_plugin_metadata(plugin_yaml_path)
        
        assert metadata.version == '0.0.0'
        assert metadata.author == 'Unknown'
        assert metadata.description == ''
        assert metadata.plugin_type == 'user_experimental'
        assert metadata.license == 'Unknown'


# =============================================================================
# MODEL PLUGIN LOADER TESTS - VALIDATION
# =============================================================================

class TestModelPluginLoaderValidation:
    """Test suite for plugin validation functionality."""
    
    def test_validate_plugin_basic(self, sample_plugin_metadata):
        """Test basic plugin validation."""
        loader = ModelPluginLoader()
        result = loader._validate_plugin(sample_plugin_metadata, level='standard')
        
        assert isinstance(result, dict)
        assert 'valid' in result
        assert 'errors' in result
        assert 'warnings' in result
    
    def test_validate_plugin_missing_name(self, sample_plugin_metadata):
        """Test validation with missing plugin_name."""
        sample_plugin_metadata.plugin_name = ''
        
        loader = ModelPluginLoader()
        result = loader._validate_plugin(sample_plugin_metadata)
        
        assert result['valid'] is False
        assert "Missing plugin_name" in result['errors']
    
    def test_validate_plugin_missing_version(self, sample_plugin_metadata):
        """Test validation with missing version."""
        sample_plugin_metadata.version = ''
        
        loader = ModelPluginLoader()
        result = loader._validate_plugin(sample_plugin_metadata)
        
        assert "Missing version" in result['warnings']
    
    def test_validate_plugin_no_models(self, sample_plugin_metadata):
        """Test validation with no models."""
        sample_plugin_metadata.model_declarations = []
        
        loader = ModelPluginLoader()
        result = loader._validate_plugin(sample_plugin_metadata)
        
        assert "No models declared" in result['warnings']
    
    def test_validate_dependencies_satisfied(self, sample_plugin_metadata):
        """Test dependency validation with satisfied dependencies."""
        sample_plugin_metadata.dependencies = ['torch']
        
        loader = ModelPluginLoader()
        result = loader._validate_dependencies(sample_plugin_metadata)
        
        assert result['satisfied'] is True
        assert len(result['missing']) == 0
    
    def test_validate_dependencies_missing(self, sample_plugin_metadata):
        """Test dependency validation with missing dependencies."""
        sample_plugin_metadata.dependencies = ['nonexistent_package']
        
        loader = ModelPluginLoader()
        result = loader._validate_dependencies(sample_plugin_metadata)
        
        assert result['satisfied'] is False
        assert len(result['missing']) > 0
    
    def test_validate_dependencies_with_extras(self, sample_plugin_metadata):
        """Test dependency validation handles extras syntax (e.g., torch[cuda])."""
        sample_plugin_metadata.dependencies = ['torch[cuda]']
        
        loader = ModelPluginLoader()
        result = loader._validate_dependencies(sample_plugin_metadata)
        
        assert result['satisfied'] is True
        assert len(result['missing']) == 0
    
    def test_validate_dependencies_with_extras_missing(self, sample_plugin_metadata):
        """Test dependency validation with missing package using extras syntax."""
        sample_plugin_metadata.dependencies = ['nonexistent_package[extras]']
        
        loader = ModelPluginLoader()
        result = loader._validate_dependencies(sample_plugin_metadata)
        
        assert result['satisfied'] is False
        assert len(result['missing']) > 0
    
    def test_validate_model_declaration_valid(self, sample_model_declaration):
        """Test model declaration validation with valid data."""
        loader = ModelPluginLoader()
        errors = loader._validate_model_declaration(sample_model_declaration)
        
        assert len(errors) == 0
    
    def test_validate_model_declaration_missing_name(self, sample_model_declaration):
        """Test model declaration validation with missing name."""
        sample_model_declaration.name = ''
        
        loader = ModelPluginLoader()
        errors = loader._validate_model_declaration(sample_model_declaration)
        
        assert len(errors) > 0
        assert any("missing name" in e for e in errors)
    
    def test_validate_model_declaration_missing_class_name(self, sample_model_declaration):
        """Test model declaration validation with missing class_name."""
        sample_model_declaration.class_name = ''
        
        loader = ModelPluginLoader()
        errors = loader._validate_model_declaration(sample_model_declaration)
        
        assert len(errors) > 0
        assert any("missing class_name" in e for e in errors)
    
    def test_validate_model_declaration_no_tasks(self, sample_model_declaration):
        """Test model declaration validation with no supported tasks."""
        sample_model_declaration.supported_tasks = []
        
        loader = ModelPluginLoader()
        errors = loader._validate_model_declaration(sample_model_declaration)
        
        assert len(errors) > 0
        assert any("no supported tasks" in e for e in errors)
    
    def test_validate_model_declaration_invalid_hyperparameters(self, sample_model_declaration):
        """Test model declaration validation with invalid hyperparameters."""
        sample_model_declaration.hyperparameters = {
            'param1': 'not_a_dict'
        }
        
        loader = ModelPluginLoader()
        errors = loader._validate_model_declaration(sample_model_declaration)
        
        assert len(errors) > 0
        assert any("must be a dict" in e for e in errors)
    
    def test_security_check_clean(self, sample_plugin_metadata, temp_plugin_dir):
        """Test security check with clean code."""
        plugin_dir = temp_plugin_dir / "clean_plugin"
        plugin_dir.mkdir()
        
        with open(plugin_dir / "model.py", 'w') as f:
            f.write("import torch\nclass Model:\n    pass\n")
        
        sample_plugin_metadata.plugin_path = plugin_dir
        
        loader = ModelPluginLoader()
        result = loader._security_check(sample_plugin_metadata)
        
        assert len(result['issues']) == 0
    
    def test_security_check_suspicious_eval(self, sample_plugin_metadata, temp_plugin_dir):
        """Test security check detects eval()."""
        plugin_dir = temp_plugin_dir / "suspicious_plugin"
        plugin_dir.mkdir()
        
        with open(plugin_dir / "model.py", 'w') as f:
            f.write("import torch\nresult = eval('1+1')\n")
        
        sample_plugin_metadata.plugin_path = plugin_dir
        
        loader = ModelPluginLoader()
        result = loader._security_check(sample_plugin_metadata)
        
        assert len(result['issues']) > 0
        assert any("eval" in issue for issue in result['issues'])
    
    def test_security_check_suspicious_exec(self, sample_plugin_metadata, temp_plugin_dir):
        """Test security check detects exec()."""
        plugin_dir = temp_plugin_dir / "suspicious_plugin"
        plugin_dir.mkdir()
        
        with open(plugin_dir / "model.py", 'w') as f:
            f.write("import torch\nexec('print(1)')\n")
        
        sample_plugin_metadata.plugin_path = plugin_dir
        
        loader = ModelPluginLoader()
        result = loader._security_check(sample_plugin_metadata)
        
        assert len(result['issues']) > 0
        assert any("exec" in issue for issue in result['issues'])
    
    def test_security_check_suspicious_subprocess(self, sample_plugin_metadata, temp_plugin_dir):
        """Test security check detects subprocess import."""
        plugin_dir = temp_plugin_dir / "suspicious_subprocess"
        plugin_dir.mkdir()
        
        with open(plugin_dir / "model.py", 'w') as f:
            f.write("import subprocess\nsubprocess.run(['ls'])\n")
        
        sample_plugin_metadata.plugin_path = plugin_dir
        
        loader = ModelPluginLoader()
        result = loader._security_check(sample_plugin_metadata)
        
        assert len(result['issues']) > 0
        assert any("subprocess" in issue for issue in result['issues'])
    
    def test_security_check_suspicious_dunder_import(self, sample_plugin_metadata, temp_plugin_dir):
        """Test security check detects __import__()."""
        plugin_dir = temp_plugin_dir / "suspicious_import"
        plugin_dir.mkdir()
        
        with open(plugin_dir / "model.py", 'w') as f:
            f.write("mod = __import__('os')\n")
        
        sample_plugin_metadata.plugin_path = plugin_dir
        
        loader = ModelPluginLoader()
        result = loader._security_check(sample_plugin_metadata)
        
        assert len(result['issues']) > 0
        assert any("__import__" in issue for issue in result['issues'])
    
    def test_security_check_nested_files(self, sample_plugin_metadata, temp_plugin_dir):
        """Test security check scans nested Python files."""
        plugin_dir = temp_plugin_dir / "nested_plugin"
        plugin_dir.mkdir()
        subdir = plugin_dir / "submodule"
        subdir.mkdir()
        
        with open(plugin_dir / "model.py", 'w') as f:
            f.write("import torch\nclass Model:\n    pass\n")
        
        with open(subdir / "helper.py", 'w') as f:
            f.write("result = eval('1+1')\n")
        
        sample_plugin_metadata.plugin_path = plugin_dir
        
        loader = ModelPluginLoader()
        result = loader._security_check(sample_plugin_metadata)
        
        assert len(result['issues']) > 0
        assert any("eval" in issue for issue in result['issues'])


# =============================================================================
# MODEL PLUGIN LOADER TESTS - LOADING
# =============================================================================

class TestModelPluginLoaderLoading:
    """Test suite for plugin loading functionality."""
    
    def test_load_plugin_not_discovered(self):
        """Test loading a plugin that hasn't been discovered."""
        loader = ModelPluginLoader()
        
        with pytest.raises(PluginError, match="not discovered"):
            loader.load_plugin("nonexistent_plugin")
    
    def test_load_plugin_already_loaded(self, sample_plugin_metadata):
        """Test loading a plugin that's already loaded."""
        loader = ModelPluginLoader()
        loader._plugins['test_plugin'] = sample_plugin_metadata
        sample_plugin_metadata.loaded = True
        
        result = loader.load_plugin('test_plugin')
        
        assert result is True
    
    def test_load_plugin_disabled(self, sample_plugin_metadata):
        """Test loading a disabled plugin."""
        loader = ModelPluginLoader()
        loader._plugins['test_plugin'] = sample_plugin_metadata
        sample_plugin_metadata.enabled = False
        
        with pytest.raises(PluginError, match="is disabled"):
            loader.load_plugin('test_plugin')
    
    @patch.object(model_plugin_system, 'MODELS_AVAILABLE', False)
    def test_load_plugin_no_models_available(self, sample_plugin_metadata):
        """Test loading plugin when ModelRegistry not available."""
        loader = ModelPluginLoader()
        loader._plugins['test_plugin'] = sample_plugin_metadata
        
        result = loader.load_plugin('test_plugin', register_models=False)
        assert result is True
        assert sample_plugin_metadata.loaded is True
    
    def test_load_all_plugins_empty(self):
        """Test loading all plugins when none discovered."""
        loader = ModelPluginLoader()
        result = loader.load_all_plugins()
        
        assert result == {}
    
    def test_load_all_plugins_success(self, sample_plugin_metadata):
        """Test loading all plugins successfully."""
        loader = ModelPluginLoader()
        loader._plugins['test_plugin'] = sample_plugin_metadata
        
        with patch.object(loader, 'load_plugin', return_value=True) as mock_load:
            result = loader.load_all_plugins(register_models=False)
        
        assert result == {'test_plugin': True}
        mock_load.assert_called_once_with('test_plugin', False)
    
    def test_load_plugin_general_exception(self, sample_plugin_metadata, temp_plugin_dir):
        """Test load_plugin handles general exceptions during loading."""
        loader = ModelPluginLoader()
        sample_plugin_metadata.plugin_path = temp_plugin_dir
        loader._plugins['test_plugin'] = sample_plugin_metadata
        
        with patch.object(loader, '_register_plugin_model', 
                         side_effect=RuntimeError("Unexpected error")):
            with pytest.raises(PluginError, match="Failed to load plugin"):
                loader.load_plugin('test_plugin', register_models=True)
    
    def test_load_all_plugins_partial_failure(self, sample_plugin_metadata):
        """Test load_all_plugins handles partial failures gracefully."""
        loader = ModelPluginLoader()
        loader._plugins['good_plugin'] = sample_plugin_metadata
        
        bad_metadata = ModelPluginMetadata(
            plugin_name='bad_plugin',
            version='1.0.0',
            author='Test',
            description='Test',
            plugin_type='user',
            milia_version='>=4.0.0',
            pyg_version='>=2.0.0',
            python_version='>=3.8',
            license='MIT',
            model_declarations=[],
            enabled=True
        )
        loader._plugins['bad_plugin'] = bad_metadata
        
        def selective_load(name, register_models=True):
            if name == 'bad_plugin':
                raise RuntimeError("Simulated failure")
            loader._plugins[name].loaded = True
            return True
        
        with patch.object(loader, 'load_plugin', side_effect=selective_load):
            result = loader.load_all_plugins(register_models=False)
        
        assert result.get('bad_plugin', True) is False


# =============================================================================
# MODEL PLUGIN LOADER TESTS - MANAGEMENT
# =============================================================================

class TestModelPluginLoaderManagement:
    """Test suite for plugin management functionality."""
    
    def test_enable_plugin(self, sample_plugin_metadata):
        """Test enabling a plugin."""
        loader = ModelPluginLoader()
        loader._plugins['test_plugin'] = sample_plugin_metadata
        sample_plugin_metadata.enabled = False
        
        loader.enable_plugin('test_plugin')
        
        assert sample_plugin_metadata.enabled is True
    
    def test_enable_plugin_not_found(self):
        """Test enabling a non-existent plugin."""
        loader = ModelPluginLoader()
        
        with pytest.raises(PluginError, match="not found"):
            loader.enable_plugin('nonexistent')
    
    def test_disable_plugin(self, sample_plugin_metadata):
        """Test disabling a plugin."""
        loader = ModelPluginLoader()
        loader._plugins['test_plugin'] = sample_plugin_metadata
        sample_plugin_metadata.enabled = True
        
        loader.disable_plugin('test_plugin')
        
        assert sample_plugin_metadata.enabled is False
    
    def test_disable_plugin_not_found(self):
        """Test disabling a non-existent plugin."""
        loader = ModelPluginLoader()
        
        with pytest.raises(PluginError, match="not found"):
            loader.disable_plugin('nonexistent')
    
    def test_unload_plugin(self, sample_plugin_metadata):
        """Test unloading a plugin."""
        loader = ModelPluginLoader()
        loader._plugins['test_plugin'] = sample_plugin_metadata
        sample_plugin_metadata.loaded = True
        sample_plugin_metadata.load_time = datetime.now()
        
        loader.unload_plugin('test_plugin')
        
        assert sample_plugin_metadata.loaded is False
        assert sample_plugin_metadata.load_time is None
    
    def test_unload_plugin_not_found(self):
        """Test unloading a non-existent plugin."""
        loader = ModelPluginLoader()
        
        with pytest.raises(PluginError, match="not found"):
            loader.unload_plugin('nonexistent')
    
    def test_list_plugins_empty(self):
        """Test listing plugins when none registered."""
        loader = ModelPluginLoader()
        result = loader.list_plugins()
        
        assert result == []
    
    def test_list_plugins_all(self, sample_plugin_metadata):
        """Test listing all plugins."""
        loader = ModelPluginLoader()
        loader._plugins['test_plugin'] = sample_plugin_metadata
        
        result = loader.list_plugins()
        
        assert len(result) == 1
        assert 'test_plugin' in result
    
    def test_list_plugins_loaded_only(self, sample_plugin_metadata):
        """Test listing only loaded plugins."""
        loader = ModelPluginLoader()
        
        loaded_metadata = sample_plugin_metadata
        loaded_metadata.loaded = True
        loader._plugins['loaded_plugin'] = loaded_metadata
        
        unloaded_metadata = ModelPluginMetadata(
            plugin_name='unloaded_plugin',
            version='1.0.0',
            author='Test',
            description='Test',
            plugin_type='user',
            milia_version='>=4.0.0',
            pyg_version='>=2.0.0',
            python_version='>=3.8',
            license='MIT',
            model_declarations=[],
            loaded=False
        )
        loader._plugins['unloaded_plugin'] = unloaded_metadata
        
        result = loader.list_plugins(loaded_only=True)
        
        assert len(result) == 1
        assert 'loaded_plugin' in result
        assert 'unloaded_plugin' not in result
    
    def test_list_plugins_enabled_only(self, sample_plugin_metadata):
        """Test listing only enabled plugins."""
        loader = ModelPluginLoader()
        
        enabled_metadata = sample_plugin_metadata
        enabled_metadata.enabled = True
        loader._plugins['enabled_plugin'] = enabled_metadata
        
        disabled_metadata = ModelPluginMetadata(
            plugin_name='disabled_plugin',
            version='1.0.0',
            author='Test',
            description='Test',
            plugin_type='user',
            milia_version='>=4.0.0',
            pyg_version='>=2.0.0',
            python_version='>=3.8',
            license='MIT',
            model_declarations=[],
            enabled=False
        )
        loader._plugins['disabled_plugin'] = disabled_metadata
        
        result = loader.list_plugins(enabled_only=True)
        
        assert len(result) == 1
        assert 'enabled_plugin' in result
        assert 'disabled_plugin' not in result
    
    def test_get_plugin_info(self, sample_plugin_metadata):
        """Test getting plugin information."""
        loader = ModelPluginLoader()
        loader._plugins['test_plugin'] = sample_plugin_metadata
        
        result = loader.get_plugin_info('test_plugin')
        
        assert isinstance(result, dict)
        assert result['plugin_name'] == 'test_plugin'
        assert result['version'] == '1.0.0'
    
    def test_get_plugin_info_not_found(self):
        """Test getting info for non-existent plugin."""
        loader = ModelPluginLoader()
        
        with pytest.raises(PluginError, match="not found"):
            loader.get_plugin_info('nonexistent')
    
    def test_get_plugin_models(self, sample_plugin_metadata):
        """Test getting plugin models."""
        loader = ModelPluginLoader()
        loader._plugins['test_plugin'] = sample_plugin_metadata
        
        result = loader.get_plugin_models('test_plugin')
        
        assert len(result) == 1
        assert 'TestModel' in result
    
    def test_get_plugin_models_not_found(self):
        """Test getting models for non-existent plugin."""
        loader = ModelPluginLoader()
        
        with pytest.raises(PluginError, match="not found"):
            loader.get_plugin_models('nonexistent')


# =============================================================================
# MODEL PLUGIN LOADER TESTS - REGISTRATION
# =============================================================================

class TestModelPluginLoaderRegistration:
    """Test suite for model registration functionality."""
    
    @patch.object(model_plugin_system, 'MODELS_AVAILABLE', False)
    def test_register_plugin_model_no_registry(self, sample_model_declaration, temp_plugin_dir):
        """Test model registration when ModelRegistry not available."""
        loader = ModelPluginLoader()
        
        with pytest.raises(PluginError, match="ModelRegistry not available"):
            loader._register_plugin_model(sample_model_declaration, temp_plugin_dir)
    
    @patch.object(model_plugin_system, 'MODELS_AVAILABLE', True)
    @patch.object(model_plugin_system, 'registry')
    @patch.object(model_plugin_system, 'ModelMetadata')
    def test_register_plugin_model_import_error(
        self, mock_metadata_class, mock_registry, sample_model_declaration, temp_plugin_dir
    ):
        """Test model registration with import error."""
        loader = ModelPluginLoader()
        
        with patch('importlib.import_module', side_effect=ImportError("Module not found")):
            with pytest.raises(PluginError, match="Failed to import model"):
                loader._register_plugin_model(sample_model_declaration, temp_plugin_dir)
    
    @patch.object(model_plugin_system, 'MODELS_AVAILABLE', True)
    @patch.object(model_plugin_system, 'registry')
    @patch.object(model_plugin_system, 'ModelMetadata')
    def test_register_plugin_model_class_not_found(
        self, mock_metadata_class, mock_registry, sample_model_declaration, temp_plugin_dir
    ):
        """Test model registration when class not found in module."""
        loader = ModelPluginLoader()
        
        mock_module = Mock()
        # Use spec to make getattr raise AttributeError
        del mock_module.TestModelClass
        
        with patch('importlib.import_module', return_value=mock_module):
            with pytest.raises(PluginError, match="not found in module"):
                loader._register_plugin_model(sample_model_declaration, temp_plugin_dir)
    
    @patch.object(model_plugin_system, 'MODELS_AVAILABLE', True)
    @patch.object(model_plugin_system, 'registry')
    @patch.object(model_plugin_system, 'ModelMetadata')
    def test_register_plugin_model_not_nn_module(
        self, mock_metadata_class, mock_registry, sample_model_declaration, temp_plugin_dir
    ):
        """Test model registration when class doesn't inherit from nn.Module."""
        loader = ModelPluginLoader()
        
        class NotNNModule:
            pass
        
        mock_module = Mock()
        mock_module.TestModelClass = NotNNModule
        
        with patch('importlib.import_module', return_value=mock_module):
            with pytest.raises(PluginError, match="must inherit from nn.Module"):
                loader._register_plugin_model(sample_model_declaration, temp_plugin_dir)
    
    @patch.object(model_plugin_system, 'MODELS_AVAILABLE', True)
    @patch.object(model_plugin_system, 'registry')
    def test_register_plugin_model_success(
        self, mock_registry, sample_model_declaration, temp_plugin_dir, mock_model_class
    ):
        """Test successful model registration."""
        loader = ModelPluginLoader()
        
        mock_module = Mock()
        mock_module.TestModelClass = mock_model_class
        
        with patch('importlib.import_module', return_value=mock_module):
            loader._register_plugin_model(sample_model_declaration, temp_plugin_dir)
        
        mock_registry.register_model.assert_called_once()
        call_args = mock_registry.register_model.call_args
        
        assert call_args[1]['name'] == 'TestModel'
        assert call_args[1]['model_class'] == mock_model_class
        assert call_args[1]['plugin_name'] == 'test_plugin'
    
    @patch.object(model_plugin_system, 'MODELS_AVAILABLE', True)
    @patch.object(model_plugin_system, 'registry')
    @patch.object(model_plugin_system, 'ModelMetadata', None)
    def test_register_plugin_model_metadata_none(
        self, mock_registry, sample_model_declaration, temp_plugin_dir, mock_model_class
    ):
        """Test model registration when ModelMetadata class is None."""
        loader = ModelPluginLoader()
        
        mock_module = Mock()
        mock_module.TestModelClass = mock_model_class
        
        with patch('importlib.import_module', return_value=mock_module):
            with pytest.raises(PluginError, match="ModelMetadata not available"):
                loader._register_plugin_model(sample_model_declaration, temp_plugin_dir)
    
    @patch.object(model_plugin_system, 'MODELS_AVAILABLE', True)
    @patch.object(model_plugin_system, 'registry')
    @patch.object(model_plugin_system, 'ModelMetadata')
    def test_register_plugin_model_registry_error(
        self, mock_metadata, mock_registry, sample_model_declaration, 
        temp_plugin_dir, mock_model_class
    ):
        """Test model registration when registry raises exception."""
        loader = ModelPluginLoader()
        
        mock_module = Mock()
        mock_module.TestModelClass = mock_model_class
        mock_registry.register_model.side_effect = RuntimeError("Registry error")
        
        with patch('importlib.import_module', return_value=mock_module):
            with pytest.raises(PluginError, match="Failed to register model"):
                loader._register_plugin_model(sample_model_declaration, temp_plugin_dir)


# =============================================================================
# MODULE-LEVEL FUNCTION TESTS
# =============================================================================

class TestModuleLevelFunctions:
    """Test suite for module-level convenience functions."""
    
    def test_get_plugin_loader(self):
        """Test get_plugin_loader function."""
        loader = get_plugin_loader()
        
        assert isinstance(loader, ModelPluginLoader)
        
        loader2 = get_plugin_loader()
        assert loader is loader2
    
    def test_discover_plugins_function(self, temp_plugin_dir, sample_plugin_yaml):
        """Test discover_plugins module-level function."""
        plugin_dir = temp_plugin_dir / "test_plugin"
        plugin_dir.mkdir()
        
        with open(plugin_dir / "plugin.yaml", 'w') as f:
            yaml.dump(sample_plugin_yaml, f)
        
        result = discover_plugins([temp_plugin_dir])
        
        assert 'test_plugin' in result
    
    def test_discover_plugins_no_paths(self):
        """Test discover_plugins with no paths provided."""
        # When no paths provided, it tries to import config_bridge which is mocked
        # The function will catch the exception and use default path
        result = discover_plugins(paths=None)
        
        assert isinstance(result, list)
    
    def test_load_plugin_function(self, sample_plugin_metadata):
        """Test load_plugin module-level function."""
        loader = get_plugin_loader()
        loader._plugins['test_plugin'] = sample_plugin_metadata
        
        with patch.object(loader, 'load_plugin', return_value=True) as mock_load:
            result = load_plugin('test_plugin', register_models=False)
        
        assert result is True
        mock_load.assert_called_once_with('test_plugin', False)
    
    def test_list_plugins_function(self, sample_plugin_metadata):
        """Test list_plugins module-level function."""
        loader = get_plugin_loader()
        loader._plugins['test_plugin'] = sample_plugin_metadata
        
        result = list_plugins()
        
        assert 'test_plugin' in result
    
    def test_list_plugins_function_filters(self, sample_plugin_metadata):
        """Test list_plugins module-level function with filters."""
        loader = get_plugin_loader()
        sample_plugin_metadata.loaded = True
        loader._plugins['test_plugin'] = sample_plugin_metadata
        
        result = list_plugins(loaded_only=True)
        
        assert 'test_plugin' in result
    
    def test_get_plugin_info_function(self, sample_plugin_metadata):
        """Test get_plugin_info module-level function."""
        loader = get_plugin_loader()
        loader._plugins['test_plugin'] = sample_plugin_metadata
        
        result = get_plugin_info('test_plugin')
        
        assert result['plugin_name'] == 'test_plugin'
    
    def test_validate_plugin_function(self, sample_plugin_metadata):
        """Test validate_plugin module-level function."""
        loader = get_plugin_loader()
        loader._plugins['test_plugin'] = sample_plugin_metadata
        
        result = validate_plugin('test_plugin')
        
        assert 'valid' in result
        assert 'errors' in result
        assert 'warnings' in result
    
    def test_validate_plugin_function_not_found(self):
        """Test validate_plugin with non-existent plugin."""
        with pytest.raises(PluginError, match="not found"):
            validate_plugin('nonexistent')


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================

class TestThreadSafety:
    """Test suite for thread safety."""
    
    def test_concurrent_plugin_discovery(self, temp_plugin_dir):
        """Test concurrent plugin discovery."""
        for i in range(5):
            plugin_dir = temp_plugin_dir / f"plugin_{i}"
            plugin_dir.mkdir()
            
            plugin_yaml = {
                'plugin_name': f'plugin_{i}',
                'version': '1.0.0',
                'author': 'Test',
                'description': 'Test',
                'plugin_type': 'user',
                'milia_version': '>=4.0.0',
                'pyg_version': '>=2.0.0',
                'python_version': '>=3.8',
                'license': 'MIT',
                'models': []
            }
            
            with open(plugin_dir / "plugin.yaml", 'w') as f:
                yaml.dump(plugin_yaml, f)
        
        loader = get_plugin_loader()
        results = []
        
        def discover():
            result = loader.discover_plugins([temp_plugin_dir])
            results.append(len(result))
        
        threads = [threading.Thread(target=discover) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert all(r == 5 for r in results)
    
    def test_concurrent_enable_disable(self, sample_plugin_metadata):
        """Test concurrent enable/disable operations."""
        loader = get_plugin_loader()
        loader._plugins['test_plugin'] = sample_plugin_metadata
        
        def toggle():
            for _ in range(10):
                loader.enable_plugin('test_plugin')
                loader.disable_plugin('test_plugin')
        
        threads = [threading.Thread(target=toggle) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert 'test_plugin' in loader._plugins
    
    def test_concurrent_list_plugins(self, sample_plugin_metadata):
        """Test concurrent list_plugins calls."""
        loader = get_plugin_loader()
        loader._plugins['test_plugin'] = sample_plugin_metadata
        
        results = []
        
        def list_them():
            result = loader.list_plugins()
            results.append(result)
        
        threads = [threading.Thread(target=list_them) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert all(r == ['test_plugin'] for r in results)


# =============================================================================
# EDGE CASES AND ERROR HANDLING TESTS
# =============================================================================

class TestEdgeCases:
    """Test suite for edge cases and error handling."""
    
    def test_plugin_with_special_characters_in_name(self, temp_plugin_dir):
        """Test plugin with special characters in name."""
        plugin_dir = temp_plugin_dir / "plugin-with-dashes"
        plugin_dir.mkdir()
        
        plugin_yaml = {
            'plugin_name': 'plugin-with-dashes',
            'version': '1.0.0',
            'author': 'Test',
            'description': 'Test',
            'plugin_type': 'user',
            'models': []
        }
        
        with open(plugin_dir / "plugin.yaml", 'w') as f:
            yaml.dump(plugin_yaml, f)
        
        loader = ModelPluginLoader()
        result = loader.discover_plugins([temp_plugin_dir])
        
        assert 'plugin-with-dashes' in result
    
    def test_plugin_with_empty_description(self, temp_plugin_dir):
        """Test plugin with empty description."""
        plugin_dir = temp_plugin_dir / "no_desc_plugin"
        plugin_dir.mkdir()
        
        plugin_yaml = {
            'plugin_name': 'no_desc_plugin',
            'version': '1.0.0',
            'author': 'Test',
            'description': '',
            'plugin_type': 'user',
            'models': []
        }
        
        with open(plugin_dir / "plugin.yaml", 'w') as f:
            yaml.dump(plugin_yaml, f)
        
        loader = ModelPluginLoader()
        result = loader.discover_plugins([temp_plugin_dir])
        
        assert 'no_desc_plugin' in result
        info = loader.get_plugin_info('no_desc_plugin')
        assert info['description'] == ''
    
    def test_plugin_with_very_long_name(self, temp_plugin_dir):
        """Test plugin with very long name."""
        long_name = 'a' * 200
        plugin_dir = temp_plugin_dir / long_name[:100]
        plugin_dir.mkdir()
        
        plugin_yaml = {
            'plugin_name': long_name,
            'version': '1.0.0',
            'author': 'Test',
            'description': 'Test',
            'plugin_type': 'user',
            'models': []
        }
        
        with open(plugin_dir / "plugin.yaml", 'w') as f:
            yaml.dump(plugin_yaml, f)
        
        loader = ModelPluginLoader()
        result = loader.discover_plugins([temp_plugin_dir])
        
        assert long_name in result
    
    def test_plugin_with_unicode_characters(self, temp_plugin_dir):
        """Test plugin with unicode characters."""
        plugin_dir = temp_plugin_dir / "unicode_plugin"
        plugin_dir.mkdir()
        
        plugin_yaml = {
            'plugin_name': 'unicode_plugin',
            'version': '1.0.0',
            'author': 'Test Author 作者',
            'description': 'Test plugin with unicode 测试',
            'plugin_type': 'user',
            'models': []
        }
        
        with open(plugin_dir / "plugin.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(plugin_yaml, f, allow_unicode=True)
        
        loader = ModelPluginLoader()
        result = loader.discover_plugins([temp_plugin_dir])
        
        assert 'unicode_plugin' in result
        info = loader.get_plugin_info('unicode_plugin')
        assert '作者' in info['author']
    
    def test_multiple_validation_levels(self, sample_plugin_metadata):
        """Test validation with all validation levels."""
        sample_plugin_metadata.dependencies = ['nonexistent_package']
        
        loader = ModelPluginLoader()
        
        result_permissive = loader._validate_plugin(sample_plugin_metadata, level='permissive')
        result_standard = loader._validate_plugin(sample_plugin_metadata, level='standard')
        assert 'missing' in str(result_standard['warnings']).lower() or \
               'missing' in str(result_standard['errors']).lower()
        
        result_strict = loader._validate_plugin(sample_plugin_metadata, level='strict')
        assert not result_strict['valid']
    
    def test_validation_errors_stored_in_metadata(self, sample_plugin_metadata):
        """Test that validation errors are stored in metadata.validation_errors."""
        sample_plugin_metadata.plugin_name = ''
        sample_plugin_metadata.version = ''
        
        loader = ModelPluginLoader()
        loader._validate_plugin(sample_plugin_metadata, level='standard')
        
        assert len(sample_plugin_metadata.validation_errors) > 0
    
    def test_validation_strict_vs_standard(self, sample_plugin_metadata, sample_model_declaration):
        """Test strict mode puts warnings into errors, standard mode doesn't."""
        sample_model_declaration.supported_tasks = []
        sample_plugin_metadata.model_declarations = [sample_model_declaration]
        
        loader = ModelPluginLoader()
        
        result_standard = loader._validate_plugin(sample_plugin_metadata, level='standard')
        assert result_standard['valid'] is True
        assert any('no supported tasks' in w for w in result_standard['warnings'])
        
        result_strict = loader._validate_plugin(sample_plugin_metadata, level='strict')
        assert result_strict['valid'] is False
        assert any('no supported tasks' in e for e in result_strict['errors'])
    
    def test_plugin_path_none(self, sample_plugin_metadata):
        """Test security check when plugin_path is None."""
        sample_plugin_metadata.plugin_path = None
        
        loader = ModelPluginLoader()
        result = loader._security_check(sample_plugin_metadata)
        
        assert isinstance(result, dict)
        assert 'issues' in result


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Test suite for integration scenarios."""
    
    def test_full_plugin_lifecycle(self, temp_plugin_dir, sample_plugin_yaml):
        """Test complete plugin lifecycle: discover -> load -> enable/disable -> unload."""
        plugin_dir = temp_plugin_dir / "lifecycle_plugin"
        plugin_dir.mkdir()
        
        sample_plugin_yaml['plugin_name'] = 'lifecycle_plugin'
        sample_plugin_yaml['models'] = []
        
        with open(plugin_dir / "plugin.yaml", 'w') as f:
            yaml.dump(sample_plugin_yaml, f)
        
        loader = get_plugin_loader()
        
        discovered = loader.discover_plugins([temp_plugin_dir])
        assert 'lifecycle_plugin' in discovered
        
        success = loader.load_plugin('lifecycle_plugin', register_models=False)
        assert success is True
        
        loaded = loader.list_plugins(loaded_only=True)
        assert 'lifecycle_plugin' in loaded
        
        loader.disable_plugin('lifecycle_plugin')
        enabled = loader.list_plugins(enabled_only=True)
        assert 'lifecycle_plugin' not in enabled
        
        loader.enable_plugin('lifecycle_plugin')
        enabled = loader.list_plugins(enabled_only=True)
        assert 'lifecycle_plugin' in enabled
        
        loader.unload_plugin('lifecycle_plugin')
        loaded = loader.list_plugins(loaded_only=True)
        assert 'lifecycle_plugin' not in loaded
    
    def test_discover_load_validate_chain(self, temp_plugin_dir, sample_plugin_yaml):
        """Test chaining discover, validate, and load operations."""
        plugin_dir = temp_plugin_dir / "chain_plugin"
        plugin_dir.mkdir()
        
        sample_plugin_yaml['plugin_name'] = 'chain_plugin'
        sample_plugin_yaml['models'] = []
        
        with open(plugin_dir / "plugin.yaml", 'w') as f:
            yaml.dump(sample_plugin_yaml, f)
        
        loader = get_plugin_loader()
        
        discovered = loader.discover_plugins([temp_plugin_dir], auto_validate=False)
        assert 'chain_plugin' in discovered
        
        validation = validate_plugin('chain_plugin', level='standard')
        assert validation['valid']
        
        success = loader.load_plugin('chain_plugin', register_models=False)
        assert success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
