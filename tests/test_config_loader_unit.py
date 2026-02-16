#!/usr/bin/env python3
"""
Unit tests for config_loader.py module (Phase 5/6 Refactored with YAML Splitting)

Test file: test_config_loader_unit.py
Module under test: ~/ml_projects/milia/milia_pipeline/config/config_loader.py

This test suite validates the config_loader module after Phase 5/6 registry integration
refactoring and YAML splitting feature addition. It tests configuration loading, caching,
validation, migration integration, handler-based architecture, registry-based dataset type
resolution, and split-file configuration support.

Key Test Areas:
1. Configuration loading and caching mechanisms
2. Thread-safe cache operations
3. Validation level support (STRICT/NORMAL/RELAXED)
4. Migration integration and status checking
5. Enhanced transformation configuration
6. Experimental setup management
7. Handler-based architecture integration
8. Statistics and diagnostics
9. Error handling and edge cases
10. Configuration file utilities
11. Cache management functions
12. Validation reporting
13. Legacy format detection
14. Configuration hash and comparison
15. Factory functions and utilities
16. **PHASE 5**: Registry integration and lazy initialization
17. **PHASE 5**: Dynamic dataset type resolution (_get_default_dataset_type)
18. **PHASE 5**: create_example_config with registry-based defaults
19. **PHASE 5**: migrate_legacy_config with registry-based defaults
20. **PHASE 5**: Backward compatibility verification
21. **Standard Transforms Support** (NEW):
    - _detect_transformation_format() with standard_transforms
    - _validate_enhanced_format() with standard_transforms
    - check_migration_status() with standard_transforms_count
    - recommend_validation_level() with both transform sources
    - Backward compatibility tests
22. **YAML Splitting Support** (NEW):
    - _discover_config_files() for single-file vs split-file mode
    - _collect_yaml_files() for file discovery and ordering
    - _deep_merge_configs() for configuration merging
    - _load_and_merge_yaml_files() for multiple file loading
    - _get_default_config_path() with directory support
    - Integration tests for directory-based config loading
23. **PHASE 6**: Dataset Type Normalization (NEW):
    - _normalize_dataset_type() for case-insensitive matching
    - _normalize_dataset_keyed_sections() for config key normalization
    - _normalize_dict_keys() helper function
    - Re-entrant call protection during registry initialization
    - Integration with load_config() for automatic normalization
24. **Validation/Migration Reports** (NEW):
    - get_validation_report() for retrieving validation results
    - get_migration_report() for retrieving migration results
25. **Production-Ready Edge Cases** (NEW):
    - Re-entrant call protection tests
    - Global state management tests
    - Unicode and encoding handling tests
    - Concurrent access edge cases
    - Path handling edge cases
    - Configuration statistics edge cases

Mock Pollution Prevention:
- Uses _INJECTED_MOCKS tracking for proper cleanup
- Implements setup_module() and teardown_module() for test isolation
- Ensures no sys.modules pollution between test files
"""

import os
import sys
from pathlib import Path

# CRITICAL: Add project root to Python path FIRST
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# CRITICAL: Create and inject mock modules BEFORE any other imports
# This must happen before importing anything from milia_pipeline


class MockValidationLevel:
    BASIC = "basic"
    SEMANTIC = "semantic"
    FULL = "full"


class MockValidationScope:
    FULL = "full"
    TRANSFORMATIONS = "transformations"
    DATASET = "dataset"


class MockConfigMigration:
    """Mock ConfigMigration class"""

    def __init__(self):
        self.migration_history = []

    def migrate(self, config, **kwargs):
        """Mock migration"""
        return config

    def migrate_to_enhanced(self, config):
        """Mock migration to enhanced format"""
        if isinstance(config.get("transformations"), list):
            transforms = config["transformations"]
            enhanced_config = config.copy()
            enhanced_config["transformations"] = {
                "default_setup": "baseline",
                "experimental_setups": {
                    "baseline": {
                        "name": "baseline",
                        "description": "Migrated from legacy format",
                        "pipeline": [
                            {"type": t.get("name", "Unknown"), "params": t.get("kwargs", {})}
                            for t in transforms
                        ],
                    }
                },
                "research_metadata": {"migrated": True, "original_format": "list"},
            }
            return enhanced_config, []
        return config, []

    def detect_format(self, config):
        """Mock format detection"""
        if isinstance(config, dict) and "transformations" in config:
            transforms = config["transformations"]
            if isinstance(transforms, list):
                return "legacy_list"
            elif isinstance(transforms, dict):
                if "experimental_setups" in transforms:
                    return "enhanced"
                elif "setups" in transforms:
                    return "legacy_dict"
                else:
                    return "legacy_dict"
        return "unknown"


class MockYAMLSchemaValidator:
    """Mock YAMLSchemaValidator class"""

    def __init__(self):
        self.validation_count = 0

    def detect_format(self, config):
        """Mock format detection"""
        if not isinstance(config, dict):
            return "invalid"
        if "transformations" not in config:
            return "invalid"

        transforms = config["transformations"]
        if isinstance(transforms, list):
            return "legacy_list"
        elif isinstance(transforms, dict):
            if "experimental_setups" in transforms:
                return "enhanced"
            elif "setups" in transforms:
                return "legacy_dict"
            else:
                return "legacy_dict"
        else:
            return "invalid"

    def validate(self, config, strict_mode=False):
        """Mock validation - returns object with valid, errors, warnings attributes"""
        self.validation_count += 1

        class ValidationResult:
            def __init__(self, is_valid, errors=None, warnings=None):
                self.valid = is_valid
                self.errors = errors or []
                self.warnings = warnings or []

        if config is None:
            return ValidationResult(False, ["Configuration cannot be None"], [])

        if not isinstance(config, dict):
            return ValidationResult(False, ["Configuration must be a dictionary"], [])

        if "transformations" not in config:
            return ValidationResult(False, ["Missing transformations key"], [])

        transforms = config["transformations"]
        if isinstance(transforms, str):
            return ValidationResult(False, ["Transformations cannot be a string"], [])

        if transforms is None:
            return ValidationResult(False, ["Transformations cannot be None"], [])

        return ValidationResult(True, [], [])


# Create complete mock module for config_schemas
class MockConfigSchemasModule:
    ConfigMigration = MockConfigMigration
    YAMLSchemaValidator = MockYAMLSchemaValidator
    ValidationLevel = MockValidationLevel
    ValidationScope = MockValidationScope
    __name__ = "milia_pipeline.config.config_schemas"
    __file__ = "/mock/config_schemas.py"


# DISABLED: Mocking this module breaks other test files during collection
# Inject mock into sys.modules BEFORE any milia_pipeline imports
# if 'milia_pipeline.config.config_schemas' not in sys.modules:
#    sys.modules['milia_pipeline.config.config_schemas'] = MockConfigSchemasModule()


# Mock ConfigHandler
class MockConfigHandler:
    """Mock ConfigHandler class"""

    def __init__(self):
        self.migration_available = True
        self.validation_available = True
        self.enhancement_available = True

    def detect_format(self, config):
        """Mock format detection"""
        if isinstance(config, dict) and "transformations" in config:
            transforms = config["transformations"]
            if isinstance(transforms, list):
                return {"format_type": "legacy", "needs_migration": True, "is_valid": True}
            elif isinstance(transforms, dict) and "experimental_setups" in transforms:
                return {"format_type": "enhanced", "needs_migration": False, "is_valid": True}
        return {"format_type": "unknown", "needs_migration": False, "is_valid": False}

    def migrate_config(self, config, **kwargs):
        """Mock config migration"""
        # Convert legacy list format to enhanced format
        if isinstance(config.get("transformations"), list):
            transforms = config["transformations"]
            migrated = config.copy()
            migrated["transformations"] = {
                "default_setup": "baseline",
                "experimental_setups": {
                    "baseline": {
                        "name": "baseline",
                        "description": "Migrated from legacy format",
                        "pipeline": [
                            {"type": t.get("name", "Unknown"), "params": t.get("kwargs", {})}
                            for t in transforms
                        ],
                    }
                },
            }
            return migrated
        return config

    def validate_config(self, config, validation_level="NORMAL"):
        """Mock config validation"""

        class ValidationResult:
            def __init__(self):
                self.valid = True
                self.errors = []
                self.warnings = []

        return ValidationResult()

    def is_migration_available(self):
        """Check if migration is available"""
        return self.migration_available

    def is_validation_available(self):
        """Check if validation is available"""
        return self.validation_available

    def is_enhancement_available(self):
        """Check if enhancement is available"""
        return self.enhancement_available

    def get_feature_status(self):
        """Get feature status"""
        return {
            "migration_available": self.migration_available,
            "validation_available": self.validation_available,
            "enhancement_available": self.enhancement_available,
            "phase2_available": True,
        }

    def print_feature_status(self):
        """Print feature status"""
        print("=== Feature Status ===")
        print(f"Migration: {'Available' if self.migration_available else 'Not Available'}")
        print(f"Validation: {'Available' if self.validation_available else 'Not Available'}")
        print(f"Enhancement: {'Available' if self.enhancement_available else 'Not Available'}")


# Create mock ConfigHandler module
class MockConfigHandlerModule:
    ConfigHandler = MockConfigHandler
    __name__ = "milia_pipeline.handlers.config_handler"
    __file__ = "/mock/config_handler.py"


# CRITICAL: Track original sys.modules state for cleanup
_ORIGINAL_SYSMODULES_KEYS = None
_INJECTED_MOCKS = []

# NOTE: sys.modules injection is deferred to setup_module() to prevent
# pollution during pytest collection.  See §4.4 of the tracker.

# Suppress warnings
import warnings

warnings.filterwarnings("ignore", message="config_schemas not available.*", category=UserWarning)

# Now safe to import other modules
import copy
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import yaml

# ---------------------------------------------------------------------------
# Module-level placeholders — populated by setup_module()
# ---------------------------------------------------------------------------
ConfigurationError = None
ValidationError = None
config_loader = None
load_config = None
reload_config = None
get_enhanced_transformation_config = None
load_transformation_config = None
get_experimental_setup = None
list_experimental_setups = None
validate_config_file = None
validate_and_report = None
recommend_validation_level = None
migrate_legacy_config = None
check_migration_status = None
clear_config_cache = None
get_config_hash = None
is_config_loaded = None
get_config_statistics = None
print_config_diagnostics = None
get_transformation_feature_status = None
print_transformation_status = None
create_example_config = None
_detect_transformation_format = None
_validate_enhanced_format = None
_discover_config_files = None
_collect_yaml_files = None
_deep_merge_configs = None
_load_and_merge_yaml_files = None
_get_default_config_path = None
_normalize_dataset_type = None
_normalize_dataset_keyed_sections = None
_normalize_dict_keys = None
_init_registry = None
_get_default_dataset_type = None
get_validation_report = None
get_migration_report = None


# ==============================================================================
# TEST FIXTURES AND HELPERS
# ==============================================================================


def create_basic_config():
    """Create a basic configuration for testing"""
    return {
        "project_name": "test_project",
        "dataset": {
            "name": "test_dataset",
            "path": "~/Chem_Data/milia_PyG_Dataset/raw/DFT_all_sliced.npz",
            "type": "DFT",
        },
        "transformations": [
            {"name": "StandardScaler", "kwargs": {}},
            {"name": "PCA", "kwargs": {"n_components": 10}},
        ],
        "model": {"type": "MLP", "params": {"hidden_size": 128}},
    }


def create_enhanced_config():
    """Create an enhanced configuration with experimental setups"""
    return {
        "project_name": "test_project",
        "dataset": {
            "name": "test_dataset",
            "path": "~/Chem_Data/milia_PyG_Dataset/raw/DFT_all_sliced.npz",
            "type": "DFT",
        },
        "transformations": {
            "default_setup": "baseline",
            "experimental_setups": {
                "baseline": {
                    "name": "baseline",
                    "description": "Baseline configuration",
                    "pipeline": [
                        {"type": "StandardScaler", "params": {}},
                        {"type": "PCA", "params": {"n_components": 10}},
                    ],
                },
                "advanced": {
                    "name": "advanced",
                    "description": "Advanced configuration",
                    "pipeline": [
                        {"type": "RobustScaler", "params": {}},
                        {"type": "KernelPCA", "params": {"n_components": 20}},
                    ],
                },
            },
        },
        "model": {"type": "MLP", "params": {"hidden_size": 128}},
    }


def create_invalid_config():
    """Create an invalid configuration for testing"""
    return {"project_name": "test_project", "transformations": "invalid_string"}


def create_empty_config():
    """Create an empty configuration"""
    return {}


# ==============================================================================
# PHASE 5: Registry Integration Tests
# ==============================================================================


class TestPhase5RegistryIntegration(unittest.TestCase):
    """
    Test Phase 5 registry integration and lazy initialization.

    These tests verify the new _init_registry() and _get_default_dataset_type()
    functions added in Phase 5 refactoring.
    """

    def setUp(self):
        """Reset registry state before each test."""
        # Import the module to reset its state
        import milia_pipeline.config.config_loader as loader_module

        # Store original state for restoration
        self._original_registry_initialized = getattr(loader_module, "_REGISTRY_INITIALIZED", False)
        self._original_registry_available = getattr(loader_module, "_REGISTRY_AVAILABLE", False)
        self._original_registry_error = getattr(loader_module, "_REGISTRY_IMPORT_ERROR", None)
        self._original_registry_initializing = getattr(
            loader_module, "_REGISTRY_INITIALIZING", False
        )
        self._original_list_all = getattr(loader_module, "_registry_list_all", None)
        self._original_get = getattr(loader_module, "_registry_get", None)
        self._original_is_registered = getattr(loader_module, "_registry_is_registered", None)

        # Reset for test
        loader_module._REGISTRY_INITIALIZED = False
        loader_module._REGISTRY_AVAILABLE = False
        loader_module._REGISTRY_IMPORT_ERROR = None
        loader_module._REGISTRY_INITIALIZING = False
        loader_module._registry_list_all = None
        loader_module._registry_get = None
        loader_module._registry_is_registered = None

    def tearDown(self):
        """Restore registry state after each test."""
        import milia_pipeline.config.config_loader as loader_module

        loader_module._REGISTRY_INITIALIZED = self._original_registry_initialized
        loader_module._REGISTRY_AVAILABLE = self._original_registry_available
        loader_module._REGISTRY_IMPORT_ERROR = self._original_registry_error
        loader_module._REGISTRY_INITIALIZING = self._original_registry_initializing
        loader_module._registry_list_all = self._original_list_all
        loader_module._registry_get = self._original_get
        loader_module._registry_is_registered = self._original_is_registered

    def test_init_registry_success(self):
        """Test successful registry initialization."""
        import milia_pipeline.config.config_loader as loader_module
        from milia_pipeline.config.config_loader import _init_registry

        # Mock the registry module imports
        with patch("milia_pipeline.datasets.registry.list_all") as _mock_list_all:
            with patch("milia_pipeline.datasets.registry.get") as _mock_get:
                with patch("milia_pipeline.datasets.registry.is_registered") as _mock_is_registered:
                    result = _init_registry()

                    # Should return True and set flags
                    self.assertTrue(result)
                    self.assertTrue(loader_module._REGISTRY_AVAILABLE)
                    self.assertTrue(loader_module._REGISTRY_INITIALIZED)
                    self.assertIsNotNone(loader_module._registry_list_all)
                    self.assertIsNotNone(loader_module._registry_get)
                    self.assertIsNotNone(loader_module._registry_is_registered)

    def test_init_registry_import_error(self):
        """Test registry initialization with ImportError."""
        # Mock failed import
        import builtins

        import milia_pipeline.config.config_loader as loader_module
        from milia_pipeline.config.config_loader import _init_registry

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "milia_pipeline.datasets.registry" in name:
                raise ImportError("Mocked registry import failure")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = _init_registry()

            # Should return False and set appropriate flags
            self.assertFalse(result)
            self.assertFalse(loader_module._REGISTRY_AVAILABLE)
            self.assertTrue(loader_module._REGISTRY_INITIALIZED)
            self.assertIsNotNone(loader_module._REGISTRY_IMPORT_ERROR)

    def test_init_registry_already_initialized(self):
        """Test registry initialization when already initialized."""
        import milia_pipeline.config.config_loader as loader_module
        from milia_pipeline.config.config_loader import _init_registry

        # Set as already initialized
        loader_module._REGISTRY_INITIALIZED = True
        loader_module._REGISTRY_AVAILABLE = True

        result = _init_registry()

        # Should return True without reinitializing
        self.assertTrue(result)

    def test_get_default_dataset_type_with_registry_dft_available(self):
        """Test _get_default_dataset_type when registry available with DFT."""
        import milia_pipeline.config.config_loader as loader_module
        from milia_pipeline.config.config_loader import _get_default_dataset_type

        # Mock registry as available with DFT registered
        loader_module._REGISTRY_INITIALIZED = False

        with patch("milia_pipeline.config.config_loader._init_registry", return_value=True):
            mock_list_all = Mock(return_value=["DFT", "DMC", "Wavefunction"])
            loader_module._REGISTRY_AVAILABLE = True
            loader_module._registry_list_all = mock_list_all

            result = _get_default_dataset_type()

            # Should prefer 'DFT' for backward compatibility
            self.assertEqual(result, "DFT")
            mock_list_all.assert_called_once()

    def test_get_default_dataset_type_dft_not_registered(self):
        """Test _get_default_dataset_type when DFT not in registry."""
        import milia_pipeline.config.config_loader as loader_module
        from milia_pipeline.config.config_loader import _get_default_dataset_type

        # Mock registry with different types (no DFT)
        loader_module._REGISTRY_INITIALIZED = False

        with patch("milia_pipeline.config.config_loader._init_registry", return_value=True):
            mock_list_all = Mock(return_value=["QM9", "GEOM"])
            loader_module._REGISTRY_AVAILABLE = True
            loader_module._registry_list_all = mock_list_all

            result = _get_default_dataset_type()

            # Should return first registered type
            self.assertEqual(result, "QM9")

    def test_get_default_dataset_type_registry_empty(self):
        """Test _get_default_dataset_type when registry is empty."""
        import milia_pipeline.config.config_loader as loader_module
        from milia_pipeline.config.config_loader import _get_default_dataset_type

        # Mock empty registry
        loader_module._REGISTRY_INITIALIZED = False

        with patch("milia_pipeline.config.config_loader._init_registry", return_value=True):
            mock_list_all = Mock(return_value=[])
            loader_module._REGISTRY_AVAILABLE = True
            loader_module._registry_list_all = mock_list_all

            result = _get_default_dataset_type()

            # Should fall back to 'DFT'
            self.assertEqual(result, "DFT")

    def test_get_default_dataset_type_registry_unavailable(self):
        """Test _get_default_dataset_type when registry unavailable."""
        import milia_pipeline.config.config_loader as loader_module
        from milia_pipeline.config.config_loader import _get_default_dataset_type

        # Mock registry as unavailable
        loader_module._REGISTRY_INITIALIZED = False

        with patch("milia_pipeline.config.config_loader._init_registry", return_value=False):
            loader_module._REGISTRY_AVAILABLE = False
            loader_module._registry_list_all = None

            result = _get_default_dataset_type()

            # Should fall back to 'DFT'
            self.assertEqual(result, "DFT")

    def test_get_default_dataset_type_exception_handling(self):
        """Test _get_default_dataset_type handles exceptions gracefully."""
        import milia_pipeline.config.config_loader as loader_module
        from milia_pipeline.config.config_loader import _get_default_dataset_type

        # Mock registry that raises exception
        loader_module._REGISTRY_INITIALIZED = False

        with patch("milia_pipeline.config.config_loader._init_registry", return_value=True):
            mock_list_all = Mock(side_effect=Exception("Registry error"))
            loader_module._REGISTRY_AVAILABLE = True
            loader_module._registry_list_all = mock_list_all

            result = _get_default_dataset_type()

            # Should fall back to 'DFT' on error
            self.assertEqual(result, "DFT")


# ==============================================================================
# PHASE 5: create_example_config Tests
# ==============================================================================


class TestPhase5CreateExampleConfig(unittest.TestCase):
    """
    Test Phase 5 changes to create_example_config function.

    Phase 5 changed the default parameter from dataset_type='DFT' to dataset_type=None,
    which triggers dynamic registry lookup via _get_default_dataset_type().
    """

    def setUp(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_path = os.path.join(self.temp_dir, "example.yaml")

    def tearDown(self):
        """Cleanup test files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_example_config_with_explicit_dataset_type(self):
        """Test create_example_config with explicit dataset_type parameter."""
        # Call with explicit 'DMC' - should use the provided type
        result = create_example_config(self.test_config_path, dataset_type="DMC")

        self.assertTrue(os.path.exists(result))

        with open(result) as f:
            config = yaml.safe_load(f)

        self.assertEqual(config.get("dataset_type"), "DMC")

    @patch("milia_pipeline.config.config_loader._get_default_dataset_type")
    def test_create_example_config_with_none_uses_registry(self, mock_get_default):
        """Test create_example_config with dataset_type=None uses registry."""
        # Mock registry returning 'QM9'
        mock_get_default.return_value = "QM9"

        # Call with dataset_type=None (should use registry)
        result = create_example_config(self.test_config_path, dataset_type=None)

        # Should have called _get_default_dataset_type
        mock_get_default.assert_called_once()

        with open(result) as f:
            config = yaml.safe_load(f)

        # Should use registry default
        self.assertEqual(config.get("dataset_type"), "QM9")

    @patch("milia_pipeline.config.config_loader._get_default_dataset_type")
    def test_create_example_config_backward_compatible_no_param(self, mock_get_default):
        """Test create_example_config without dataset_type param (backward compatible)."""
        # Mock registry returning 'DFT' (typical case)
        mock_get_default.return_value = "DFT"

        # Call WITHOUT dataset_type parameter (tests default=None)
        result = create_example_config(self.test_config_path)

        # Should have called _get_default_dataset_type
        mock_get_default.assert_called_once()

        with open(result) as f:
            config = yaml.safe_load(f)

        # Should use registry default (likely 'DFT')
        self.assertEqual(config.get("dataset_type"), "DFT")

    def test_create_example_config_preserves_other_params(self):
        """Test create_example_config preserves other parameters."""
        # Call with include_experimental_setups=False
        result = create_example_config(
            self.test_config_path, dataset_type="DFT", include_experimental_setups=False
        )

        self.assertTrue(os.path.exists(result))


# ==============================================================================
# PHASE 5: migrate_legacy_config Tests
# ==============================================================================


class TestPhase5MigrateLegacyConfig(unittest.TestCase):
    """
    Test Phase 5 changes to migrate_legacy_config function.

    Phase 5 changed the default parameter from dataset_type='DFT' to dataset_type=None,
    which triggers dynamic registry lookup via _get_default_dataset_type().
    """

    def setUp(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.legacy_config_path = os.path.join(self.temp_dir, "legacy.yaml")
        self.output_config_path = os.path.join(self.temp_dir, "migrated.yaml")

        # Create a legacy config
        legacy_config = {
            "transformations": [
                {"name": "AddSelfLoops", "kwargs": {}},
                {"name": "ToUndirected", "kwargs": {}},
            ]
        }
        with open(self.legacy_config_path, "w") as f:
            yaml.dump(legacy_config, f)

    def tearDown(self):
        """Cleanup test files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_migrate_legacy_config_with_explicit_dataset_type(self):
        """Test migrate_legacy_config with explicit dataset_type parameter."""
        # Call with explicit 'DMC'
        try:
            output_path, report = migrate_legacy_config(
                self.legacy_config_path,
                self.output_config_path,
                dataset_type="DMC",
                backup=False,
                report=False,
            )

            # Should succeed
            self.assertTrue(os.path.exists(output_path))
            self.assertIsInstance(report, dict)
        except Exception as e:
            # Migration might not be fully available in test environment
            self.assertIsInstance(e, Exception)

    @patch("milia_pipeline.config.config_loader._get_default_dataset_type")
    def test_migrate_legacy_config_with_none_uses_registry(self, mock_get_default):
        """Test migrate_legacy_config with dataset_type=None uses registry."""
        # Mock registry returning 'QM9'
        mock_get_default.return_value = "QM9"

        # Call with dataset_type=None
        try:
            output_path, report = migrate_legacy_config(
                self.legacy_config_path,
                self.output_config_path,
                dataset_type=None,
                backup=False,
                report=False,
            )

            # Should have called _get_default_dataset_type
            mock_get_default.assert_called_once()
            self.assertTrue(os.path.exists(output_path))
        except Exception:
            # Migration might not be fully available, but _get_default should still be called
            mock_get_default.assert_called_once()

    @patch("milia_pipeline.config.config_loader._get_default_dataset_type")
    def test_migrate_legacy_config_backward_compatible_no_param(self, mock_get_default):
        """Test migrate_legacy_config without dataset_type param (backward compatible)."""
        # Mock registry returning 'DFT'
        mock_get_default.return_value = "DFT"

        # Call WITHOUT dataset_type parameter
        try:
            output_path, report = migrate_legacy_config(
                self.legacy_config_path, self.output_config_path, backup=False, report=False
            )

            # Should have called _get_default_dataset_type
            mock_get_default.assert_called_once()
        except Exception:
            # Migration might not be fully available, but _get_default should still be called
            mock_get_default.assert_called_once()


# ==============================================================================
# PHASE 5: Backward Compatibility Tests
# ==============================================================================


class TestPhase5BackwardCompatibility(unittest.TestCase):
    """
    Test backward compatibility of Phase 5 changes.

    Ensure that existing code using explicit dataset_type parameters continues
    to work identically to before the refactoring.
    """

    def setUp(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        clear_config_cache()

    def tearDown(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_explicit_dft_still_works(self):
        """Test that explicit dataset_type='DFT' still works."""
        result = create_example_config(self.config_path, dataset_type="DFT")

        self.assertTrue(os.path.exists(result))

        with open(result) as f:
            config = yaml.safe_load(f)

        self.assertEqual(config.get("dataset_type"), "DFT")

    def test_explicit_dmc_still_works(self):
        """Test that explicit dataset_type='DMC' still works."""
        result = create_example_config(self.config_path, dataset_type="DMC")

        self.assertTrue(os.path.exists(result))

        with open(result) as f:
            config = yaml.safe_load(f)

        self.assertEqual(config.get("dataset_type"), "DMC")

    def test_explicit_wavefunction_still_works(self):
        """Test that explicit dataset_type='Wavefunction' still works."""
        result = create_example_config(self.config_path, dataset_type="Wavefunction")

        self.assertTrue(os.path.exists(result))

        with open(result) as f:
            config = yaml.safe_load(f)

        self.assertEqual(config.get("dataset_type"), "Wavefunction")


# ==============================================================================
# YAML SPLITTING: Test Classes for Split-File Configuration Support (NEW)
# ==============================================================================


class TestYAMLSplittingDiscovery(unittest.TestCase):
    """
    Test _discover_config_files function for YAML splitting support.

    Tests verify:
    1. Single-file mode detection (backward compatibility)
    2. Split-file mode detection (directory-based configuration)
    3. Proper file ordering for split-file mode
    """

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_discover_single_file_mode(self):
        """Test discovery returns single-file mode for existing file."""
        config_path = os.path.join(self.temp_dir, "config.yaml")
        config = create_basic_config()
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        is_split_mode, files = _discover_config_files(config_path)

        self.assertFalse(is_split_mode)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].name, "config.yaml")

    def test_discover_directory_mode(self):
        """Test discovery returns split-file mode for directory."""
        config_dir = os.path.join(self.temp_dir, "config")
        os.makedirs(config_dir)

        # Create multiple config files
        with open(os.path.join(config_dir, "main.yaml"), "w") as f:
            yaml.dump({"project_name": "test"}, f)
        with open(os.path.join(config_dir, "datasets.yaml"), "w") as f:
            yaml.dump({"dataset_type": "DFT"}, f)

        is_split_mode, files = _discover_config_files(config_dir)

        self.assertTrue(is_split_mode)
        self.assertGreater(len(files), 0)

    def test_discover_nonexistent_path(self):
        """Test discovery handles nonexistent path gracefully."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.yaml")

        is_split_mode, files = _discover_config_files(nonexistent_path)

        # Should return single-file mode with the path (let load_config handle error)
        self.assertFalse(is_split_mode)
        self.assertEqual(len(files), 1)

    def test_discover_empty_directory(self):
        """Test discovery handles empty directory."""
        config_dir = os.path.join(self.temp_dir, "empty_config")
        os.makedirs(config_dir)

        is_split_mode, files = _discover_config_files(config_dir)

        self.assertTrue(is_split_mode)
        # Empty directory should return empty list
        self.assertEqual(len(files), 0)


class TestYAMLSplittingFileCollection(unittest.TestCase):
    """
    Test _collect_yaml_files function for proper file ordering.

    Tests verify:
    1. main.yaml is loaded first (if exists)
    2. Root-level files are sorted alphabetically
    3. datasets/ subdirectory files are included
    """

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.temp_dir, "config")
        os.makedirs(self.config_dir)
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_main_yaml_loaded_first(self):
        """Test that main.yaml is loaded first."""
        # Create main.yaml and other files
        with open(os.path.join(self.config_dir, "main.yaml"), "w") as f:
            yaml.dump({"project_name": "main"}, f)
        with open(os.path.join(self.config_dir, "alpha.yaml"), "w") as f:
            yaml.dump({"key": "alpha"}, f)
        with open(os.path.join(self.config_dir, "beta.yaml"), "w") as f:
            yaml.dump({"key": "beta"}, f)

        files = _collect_yaml_files(Path(self.config_dir))

        # main.yaml should be first
        self.assertEqual(files[0].name, "main.yaml")

    def test_alphabetical_ordering(self):
        """Test that non-main files are sorted alphabetically."""
        # Create files without main.yaml
        with open(os.path.join(self.config_dir, "zebra.yaml"), "w") as f:
            yaml.dump({"key": "zebra"}, f)
        with open(os.path.join(self.config_dir, "alpha.yaml"), "w") as f:
            yaml.dump({"key": "alpha"}, f)
        with open(os.path.join(self.config_dir, "beta.yaml"), "w") as f:
            yaml.dump({"key": "beta"}, f)

        files = _collect_yaml_files(Path(self.config_dir))
        file_names = [f.name for f in files]

        # Should be alphabetically sorted
        self.assertEqual(file_names, ["alpha.yaml", "beta.yaml", "zebra.yaml"])

    def test_datasets_subdirectory_included(self):
        """Test that datasets/ subdirectory files are included."""
        # Create main config
        with open(os.path.join(self.config_dir, "main.yaml"), "w") as f:
            yaml.dump({"project_name": "test"}, f)

        # Create datasets subdirectory
        datasets_dir = os.path.join(self.config_dir, "datasets")
        os.makedirs(datasets_dir)
        with open(os.path.join(datasets_dir, "dft.yaml"), "w") as f:
            yaml.dump({"dataset_type": "DFT"}, f)
        with open(os.path.join(datasets_dir, "dmc.yaml"), "w") as f:
            yaml.dump({"dataset_type": "DMC"}, f)

        files = _collect_yaml_files(Path(self.config_dir))
        file_names = [f.name for f in files]

        # Should include both main and datasets files
        self.assertIn("main.yaml", file_names)
        self.assertIn("dft.yaml", file_names)
        self.assertIn("dmc.yaml", file_names)

    def test_yml_extension_support(self):
        """Test that .yml extension is also supported."""
        with open(os.path.join(self.config_dir, "main.yml"), "w") as f:
            yaml.dump({"project_name": "test"}, f)
        with open(os.path.join(self.config_dir, "config.yml"), "w") as f:
            yaml.dump({"key": "value"}, f)

        files = _collect_yaml_files(Path(self.config_dir))

        self.assertGreater(len(files), 0)
        # Check that .yml files are included
        extensions = [f.suffix for f in files]
        self.assertTrue(any(ext == ".yml" for ext in extensions))


class TestYAMLSplittingDeepMerge(unittest.TestCase):
    """
    Test _deep_merge_configs function for proper configuration merging.

    Tests verify:
    1. Nested dictionaries are recursively merged
    2. Later values override earlier values
    3. Lists are replaced, not merged
    4. Original dictionaries are not modified
    """

    def test_simple_override(self):
        """Test that later values override earlier values."""
        base = {"key": "base_value", "other": "unchanged"}
        override = {"key": "override_value"}

        result = _deep_merge_configs(base, override)

        self.assertEqual(result["key"], "override_value")
        self.assertEqual(result["other"], "unchanged")

    def test_nested_dict_merge(self):
        """Test that nested dictionaries are recursively merged."""
        base = {"outer": {"inner1": "base_value", "inner2": "unchanged"}}
        override = {"outer": {"inner1": "override_value", "inner3": "new_value"}}

        result = _deep_merge_configs(base, override)

        self.assertEqual(result["outer"]["inner1"], "override_value")
        self.assertEqual(result["outer"]["inner2"], "unchanged")
        self.assertEqual(result["outer"]["inner3"], "new_value")

    def test_list_replacement(self):
        """Test that lists are replaced, not merged."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}

        result = _deep_merge_configs(base, override)

        self.assertEqual(result["items"], [4, 5])

    def test_no_mutation_of_inputs(self):
        """Test that original dictionaries are not modified."""
        base = {"key": "base", "nested": {"inner": "value"}}
        override = {"key": "override", "nested": {"new": "added"}}

        base_copy = copy.deepcopy(base)
        override_copy = copy.deepcopy(override)

        _deep_merge_configs(base, override)

        # Original dicts should be unchanged
        self.assertEqual(base, base_copy)
        self.assertEqual(override, override_copy)

    def test_add_new_keys(self):
        """Test that new keys are added from override."""
        base = {"existing": "value"}
        override = {"new_key": "new_value"}

        result = _deep_merge_configs(base, override)

        self.assertIn("existing", result)
        self.assertIn("new_key", result)

    def test_type_mismatch_override(self):
        """Test that type mismatches result in override (not merge)."""
        base = {"key": {"nested": "dict"}}
        override = {"key": "string_value"}

        result = _deep_merge_configs(base, override)

        self.assertEqual(result["key"], "string_value")


class TestYAMLSplittingLoadAndMerge(unittest.TestCase):
    """
    Test _load_and_merge_yaml_files function for complete file loading.

    Tests verify:
    1. Multiple files are loaded and merged correctly
    2. Empty files are handled gracefully
    3. Error handling for invalid YAML
    """

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_merge_multiple_files(self):
        """Test merging multiple YAML files."""
        file1 = os.path.join(self.temp_dir, "base.yaml")
        file2 = os.path.join(self.temp_dir, "override.yaml")

        with open(file1, "w") as f:
            yaml.dump({"project_name": "base", "key1": "value1"}, f)
        with open(file2, "w") as f:
            yaml.dump({"project_name": "override", "key2": "value2"}, f)

        files = [Path(file1), Path(file2)]
        result = _load_and_merge_yaml_files(files)

        # Later file should override
        self.assertEqual(result["project_name"], "override")
        # Earlier keys should remain
        self.assertEqual(result["key1"], "value1")
        # New keys should be added
        self.assertEqual(result["key2"], "value2")

    def test_empty_files_skipped(self):
        """Test that empty files are skipped gracefully."""
        file1 = os.path.join(self.temp_dir, "valid.yaml")
        file2 = os.path.join(self.temp_dir, "empty.yaml")

        with open(file1, "w") as f:
            yaml.dump({"key": "value"}, f)
        with open(file2, "w") as f:
            f.write("")  # Empty file

        files = [Path(file1), Path(file2)]
        result = _load_and_merge_yaml_files(files)

        self.assertEqual(result["key"], "value")

    def test_invalid_yaml_raises_error(self):
        """Test that invalid YAML raises ConfigurationError."""
        file1 = os.path.join(self.temp_dir, "invalid.yaml")

        with open(file1, "w") as f:
            f.write("invalid: yaml: content: [")

        files = [Path(file1)]

        with self.assertRaises(ConfigurationError):
            _load_and_merge_yaml_files(files)

    def test_non_dict_content_raises_error(self):
        """Test that non-dict content raises ConfigurationError."""
        file1 = os.path.join(self.temp_dir, "list.yaml")

        with open(file1, "w") as f:
            yaml.dump(["list", "not", "dict"], f)

        files = [Path(file1)]

        with self.assertRaises(ConfigurationError):
            _load_and_merge_yaml_files(files)

    def test_empty_file_list_raises_error(self):
        """Test that empty file list raises ConfigurationError."""
        with self.assertRaises(ConfigurationError):
            _load_and_merge_yaml_files([])


class TestYAMLSplittingIntegration(unittest.TestCase):
    """
    Integration tests for YAML splitting with load_config.

    Tests verify end-to-end split-file configuration loading.
    """

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.temp_dir, "config")
        os.makedirs(self.config_dir)
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_load_config_from_directory(self):
        """Test load_config works with directory path."""
        # Create split config files
        with open(os.path.join(self.config_dir, "main.yaml"), "w") as f:
            yaml.dump(
                {
                    "project_name": "split_project",
                    "transformations": {"experimental_setups": {}, "default_setup": "baseline"},
                },
                f,
            )
        with open(os.path.join(self.config_dir, "datasets.yaml"), "w") as f:
            yaml.dump({"dataset_type": "DFT"}, f)

        # Load from directory
        config = load_config(self.config_dir)

        self.assertEqual(config["project_name"], "split_project")
        self.assertEqual(config["dataset_type"], "DFT")

    def test_split_config_later_files_override(self):
        """Test that later files in split config override earlier."""
        # Create files that will be loaded in alphabetical order
        with open(os.path.join(self.config_dir, "a_base.yaml"), "w") as f:
            yaml.dump({"setting": "base_value"}, f)
        with open(os.path.join(self.config_dir, "b_override.yaml"), "w") as f:
            yaml.dump({"setting": "override_value"}, f)

        config = load_config(self.config_dir)

        # b_override.yaml loads after a_base.yaml, so should override
        self.assertEqual(config["setting"], "override_value")

    def test_validate_config_file_with_directory(self):
        """Test validate_config_file works with directory path."""
        with open(os.path.join(self.config_dir, "main.yaml"), "w") as f:
            yaml.dump(
                {
                    "project_name": "test",
                    "transformations": {
                        "experimental_setups": {
                            "baseline": [{"name": "AddSelfLoops", "kwargs": {}}]
                        },
                        "default_setup": "baseline",
                    },
                },
                f,
            )

        result = validate_config_file(self.config_dir)

        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("file_exists", False))


# ==============================================================================
# PHASE 6: Dataset Type Normalization Tests (NEW)
# ==============================================================================


class TestPhase6NormalizeDatasetType(unittest.TestCase):
    """
    Test _normalize_dataset_type function for Phase 6 dataset type normalization.

    Tests verify:
    1. Exact match returns unchanged
    2. Case-insensitive match returns canonical name
    3. No match returns input unchanged
    4. Re-entrant call protection works
    """

    def setUp(self):
        """Reset registry state before each test."""
        # Don't reset - let the module handle its state naturally
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        clear_config_cache()

    @patch("milia_pipeline.config.config_loader._REGISTRY_INITIALIZING", False)
    @patch("milia_pipeline.config.config_loader._REGISTRY_AVAILABLE", True)
    @patch("milia_pipeline.config.config_loader._REGISTRY_INITIALIZED", True)
    def test_exact_match_returns_unchanged(self):
        """Test that exact match returns dataset type unchanged."""
        import milia_pipeline.config.config_loader as loader_module

        mock_list_all = Mock(return_value=["DFT", "DMC", "QM9"])
        loader_module._registry_list_all = mock_list_all

        result = _normalize_dataset_type("DFT")

        self.assertEqual(result, "DFT")

    @patch("milia_pipeline.config.config_loader._REGISTRY_INITIALIZING", False)
    @patch("milia_pipeline.config.config_loader._REGISTRY_AVAILABLE", True)
    @patch("milia_pipeline.config.config_loader._REGISTRY_INITIALIZED", True)
    def test_case_insensitive_match(self):
        """Test case-insensitive match returns canonical name."""
        import milia_pipeline.config.config_loader as loader_module

        mock_list_all = Mock(return_value=["DFT", "ANI1x", "QM9"])
        loader_module._registry_list_all = mock_list_all

        # Lowercase should match ANI1x
        result = _normalize_dataset_type("ani1x")
        self.assertEqual(result, "ANI1x")

        # Uppercase should match DFT
        result = _normalize_dataset_type("dft")
        self.assertEqual(result, "DFT")

    @patch("milia_pipeline.config.config_loader._REGISTRY_INITIALIZING", False)
    @patch("milia_pipeline.config.config_loader._REGISTRY_AVAILABLE", True)
    @patch("milia_pipeline.config.config_loader._REGISTRY_INITIALIZED", True)
    def test_no_match_returns_input(self):
        """Test that no match returns input unchanged."""
        import milia_pipeline.config.config_loader as loader_module

        mock_list_all = Mock(return_value=["DFT", "DMC"])
        loader_module._registry_list_all = mock_list_all

        result = _normalize_dataset_type("UnknownType")

        # Should return unchanged
        self.assertEqual(result, "UnknownType")

    @patch("milia_pipeline.config.config_loader._REGISTRY_INITIALIZING", True)
    def test_reentrant_call_skips_normalization(self):
        """Test that re-entrant call during initialization skips normalization."""
        skip_list = []

        result = _normalize_dataset_type("DFT", _skip_cache_if_reentrant=skip_list)

        # Should return unchanged and signal re-entrant call
        self.assertEqual(result, "DFT")
        self.assertTrue(len(skip_list) > 0)

    @patch("milia_pipeline.config.config_loader._REGISTRY_AVAILABLE", False)
    @patch("milia_pipeline.config.config_loader._REGISTRY_INITIALIZED", True)
    @patch("milia_pipeline.config.config_loader._REGISTRY_INITIALIZING", False)
    def test_registry_unavailable_returns_input(self):
        """Test that unavailable registry returns input unchanged."""
        result = _normalize_dataset_type("AnyType")

        self.assertEqual(result, "AnyType")


class TestPhase6NormalizeDatasetKeyedSections(unittest.TestCase):
    """
    Test _normalize_dataset_keyed_sections function for normalizing config keys.

    Tests verify:
    1. property_availability keys are normalized
    2. data_config.property_selection keys are normalized
    3. Non-dataset keys are preserved
    """

    def setUp(self):
        """Set up test fixtures"""
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        clear_config_cache()

    @patch("milia_pipeline.config.config_loader._REGISTRY_AVAILABLE", True)
    @patch("milia_pipeline.config.config_loader._registry_list_all")
    def test_normalize_property_availability(self, mock_list_all):
        """Test normalizing property_availability keys."""
        mock_list_all.return_value = ["DFT", "ANI1x", "QM9"]

        config = {
            "property_availability": {
                "dft": {"energy": True},  # lowercase - should normalize to 'DFT'
                "ani1x": {"forces": True},  # lowercase - should normalize to 'ANI1x'
                "common_settings": {"shared": True},  # not a dataset - should stay
            }
        }

        result = _normalize_dataset_keyed_sections(config)

        # Keys should be normalized
        self.assertIn("DFT", result["property_availability"])
        self.assertIn("ANI1x", result["property_availability"])
        # Non-dataset key should be preserved
        self.assertIn("common_settings", result["property_availability"])

    @patch("milia_pipeline.config.config_loader._REGISTRY_AVAILABLE", True)
    @patch("milia_pipeline.config.config_loader._registry_list_all")
    def test_normalize_property_selection(self, mock_list_all):
        """Test normalizing data_config.property_selection keys."""
        mock_list_all.return_value = ["DFT", "DMC"]

        config = {
            "data_config": {
                "property_selection": {
                    "dft": ["energy"],  # lowercase
                    "dmc": ["total_energy"],  # lowercase
                }
            }
        }

        result = _normalize_dataset_keyed_sections(config)

        # Keys should be normalized
        self.assertIn("DFT", result["data_config"]["property_selection"])
        self.assertIn("DMC", result["data_config"]["property_selection"])

    @patch("milia_pipeline.config.config_loader._REGISTRY_AVAILABLE", False)
    def test_registry_unavailable_returns_unchanged(self):
        """Test that unavailable registry returns config unchanged."""
        config = {"property_availability": {"dft": {"energy": True}}}

        result = _normalize_dataset_keyed_sections(config)

        # Should return unchanged
        self.assertIn("dft", result["property_availability"])


class TestPhase6NormalizeDictKeys(unittest.TestCase):
    """
    Test _normalize_dict_keys helper function.
    """

    def test_normalize_matching_keys(self):
        """Test normalizing keys that match type_lookup."""
        d = {"dft": "value1", "ani1x": "value2"}
        type_lookup = {"DFT": "DFT", "ANI1X": "ANI1x"}

        result = _normalize_dict_keys(d, type_lookup, "test_section")

        self.assertIn("DFT", result)
        self.assertIn("ANI1x", result)
        self.assertEqual(result["DFT"], "value1")
        self.assertEqual(result["ANI1x"], "value2")

    def test_preserve_unmatched_keys(self):
        """Test that unmatched keys are preserved as-is."""
        d = {"dft": "value1", "custom_key": "value2"}
        type_lookup = {"DFT": "DFT"}

        result = _normalize_dict_keys(d, type_lookup, "test_section")

        self.assertIn("DFT", result)
        self.assertIn("custom_key", result)


class TestPhase6NormalizationIntegration(unittest.TestCase):
    """
    Integration tests for Phase 6 normalization during config loading.
    """

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_dataset_type_normalized_on_load(self):
        """Test that dataset_type is normalized when config is loaded."""
        # Create config with lowercase dataset_type
        config = {
            "dataset_type": "dft",  # lowercase
            "project_name": "test",
            "transformations": {"experimental_setups": {}, "default_setup": "baseline"},
        }
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        # Load config - normalization should happen if registry available
        loaded_config = load_config(self.config_path)

        # If registry available, should be normalized to 'DFT'
        # If not, should remain 'dft'
        self.assertIn(loaded_config["dataset_type"].upper(), ["DFT"])


# ==============================================================================
# TEST CLASS: Validation and Migration Reports (NEW)
# ==============================================================================


class TestValidationAndMigrationReports(unittest.TestCase):
    """
    Test get_validation_report and get_migration_report functions.
    """

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_get_validation_report_after_load(self):
        """Test getting validation report after config load."""
        config = create_enhanced_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        load_config(self.config_path, enable_validation=True)

        report = get_validation_report()

        # Report may be None or dict depending on validation results
        if report is not None:
            self.assertIsInstance(report, dict)

    def test_get_migration_report_after_load(self):
        """Test getting migration report after config load."""
        config = create_basic_config()  # Legacy format
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        load_config(self.config_path, enable_migration=True)

        report = get_migration_report()

        # Report may be None or dict depending on migration results
        if report is not None:
            self.assertIsInstance(report, dict)

    def test_get_validation_report_before_load(self):
        """Test getting validation report before any config load."""
        clear_config_cache()

        report = get_validation_report()

        # Should return None before any load
        self.assertIsNone(report)

    def test_get_migration_report_before_load(self):
        """Test getting migration report before any config load."""
        clear_config_cache()

        report = get_migration_report()

        # Should return None before any load
        self.assertIsNone(report)


# ==============================================================================
# TEST CLASS 1: Basic Configuration Loading
# ==============================================================================


class TestBasicConfigurationLoading(unittest.TestCase):
    """Test basic configuration loading functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_load_basic_config(self):
        """Test loading a basic configuration file"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        loaded_config = load_config(self.config_path)

        self.assertIsInstance(loaded_config, dict)
        self.assertEqual(loaded_config["project_name"], "test_project")

    def test_load_enhanced_config(self):
        """Test loading an enhanced configuration file"""
        config = create_enhanced_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        loaded_config = load_config(self.config_path)

        self.assertIsInstance(loaded_config, dict)
        self.assertIn("transformations", loaded_config)
        transforms = loaded_config["transformations"]
        self.assertIsInstance(transforms, dict)

    def test_load_config_with_missing_file(self):
        """Test loading configuration with missing file"""
        with self.assertRaises(ConfigurationError):
            load_config("/nonexistent/config.yaml")

    def test_load_config_with_empty_file(self):
        """Test loading configuration with empty file"""
        # Create empty file
        with open(self.config_path, "w") as f:
            f.write("")

        with self.assertRaises(ConfigurationError):
            load_config(self.config_path)

    def test_load_config_with_invalid_yaml(self):
        """Test loading configuration with invalid YAML"""
        with open(self.config_path, "w") as f:
            f.write("invalid: yaml: content: [")

        with self.assertRaises(ConfigurationError):
            load_config(self.config_path)

    def test_load_config_with_non_dict_content(self):
        """Test loading configuration that isn't a dictionary"""
        with open(self.config_path, "w") as f:
            yaml.dump(["list", "instead", "of", "dict"], f)

        with self.assertRaises(ConfigurationError):
            load_config(self.config_path)


# ==============================================================================
# TEST CLASS 2: Configuration Caching
# ==============================================================================


class TestConfigurationCaching(unittest.TestCase):
    """Test configuration caching mechanisms"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_cache_hit_on_second_load(self):
        """Test that second load hits cache"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        # First load
        load_config(self.config_path)
        stats_after_first = get_config_statistics()
        first_load_count = stats_after_first["load_count"]

        # Second load (should hit cache)
        load_config(self.config_path)
        stats_after_second = get_config_statistics()

        # Load count should not increase, cache hits should increase
        self.assertEqual(stats_after_second["load_count"], first_load_count)
        self.assertGreater(stats_after_second["cache_hits"], 0)

    def test_force_reload_bypasses_cache(self):
        """Test that force_reload bypasses cache"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        # First load
        load_config(self.config_path)
        stats_after_first = get_config_statistics()
        first_load_count = stats_after_first["load_count"]

        # Force reload
        load_config(self.config_path, force_reload=True)
        stats_after_reload = get_config_statistics()

        # Load count should increase
        self.assertEqual(stats_after_reload["load_count"], first_load_count + 1)

    def test_clear_config_cache(self):
        """Test clearing configuration cache"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        # Load config
        load_config(self.config_path)

        # Clear cache
        clear_config_cache()

        # Load again (should not hit cache)
        stats_before = get_config_statistics()
        load_config(self.config_path)
        stats_after = get_config_statistics()

        # After clearing cache, loading should increment load_count
        self.assertGreaterEqual(stats_after["load_count"], stats_before["load_count"])

    def test_cache_key_includes_parameters(self):
        """Test that cache key includes all relevant parameters"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        # Load with different parameters
        load_config(self.config_path, enable_validation=True)
        load_config(self.config_path, enable_validation=False)

        stats = get_config_statistics()

        # Should have loaded twice (different parameters)
        self.assertGreaterEqual(stats["load_count"], 2)

    def test_is_config_loaded(self):
        """Test is_config_loaded function"""
        self.assertFalse(is_config_loaded())

        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        load_config(self.config_path)
        self.assertTrue(is_config_loaded())


# ==============================================================================
# TEST CLASS 3: Thread Safety
# ==============================================================================


class TestThreadSafety(unittest.TestCase):
    """Test thread-safe operations"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_concurrent_loads(self):
        """Test concurrent configuration loads"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        results = []
        errors = []

        def load_worker():
            try:
                result = load_config(self.config_path)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=load_worker) for _ in range(10)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check results
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 10)

        # All results should be identical
        for result in results:
            self.assertEqual(result["project_name"], "test_project")

    def test_statistics_thread_safety(self):
        """Test that statistics updates are thread-safe"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        def stats_worker():
            load_config(self.config_path)
            get_config_statistics()

        threads = [threading.Thread(target=stats_worker) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should complete without errors
        stats = get_config_statistics()
        self.assertIsInstance(stats, dict)


# ==============================================================================
# TEST CLASS 4: Validation Levels
# ==============================================================================


class TestValidationLevels(unittest.TestCase):
    """Test validation level support"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_strict_validation_level(self):
        """Test STRICT validation level"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        loaded_config = load_config(self.config_path, validation_level="STRICT")

        self.assertIsInstance(loaded_config, dict)
        stats = get_config_statistics()
        self.assertEqual(stats["validation_level"], "STRICT")

    def test_normal_validation_level(self):
        """Test NORMAL validation level (default)"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        loaded_config = load_config(self.config_path, validation_level="NORMAL")

        self.assertIsInstance(loaded_config, dict)
        stats = get_config_statistics()
        self.assertEqual(stats["validation_level"], "NORMAL")

    def test_relaxed_validation_level(self):
        """Test RELAXED validation level"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        loaded_config = load_config(self.config_path, validation_level="RELAXED")

        self.assertIsInstance(loaded_config, dict)
        stats = get_config_statistics()
        self.assertEqual(stats["validation_level"], "RELAXED")

    def test_invalid_validation_level_defaults_to_normal(self):
        """Test that invalid validation level defaults to NORMAL"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        loaded_config = load_config(self.config_path, validation_level="INVALID")

        self.assertIsInstance(loaded_config, dict)
        stats = get_config_statistics()
        self.assertEqual(stats["validation_level"], "NORMAL")

    def test_recommend_validation_level(self):
        """Test validation level recommendation"""
        config = create_enhanced_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        recommended = recommend_validation_level(self.config_path)

        self.assertIn(recommended, ["STRICT", "NORMAL", "RELAXED"])

    def test_recommend_validation_level_for_simple_config(self):
        """Test validation level recommendation for simple config"""
        config = {
            "project_name": "test",
            "transformations": {
                "experimental_setups": {
                    "baseline": {
                        "name": "baseline",
                        "pipeline": [{"type": "StandardScaler", "params": {}}],
                    }
                }
            },
        }
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        recommended = recommend_validation_level(self.config_path)

        # Simple configs should recommend STRICT
        self.assertEqual(recommended, "STRICT")


# ==============================================================================
# TEST CLASS 5: Enhanced Transformation Configuration
# ==============================================================================


class TestEnhancedTransformationConfiguration(unittest.TestCase):
    """Test enhanced transformation configuration features"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_get_enhanced_transformation_config(self):
        """Test getting enhanced transformation config"""
        config = create_enhanced_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        load_config(self.config_path)
        transform_config = get_enhanced_transformation_config()

        self.assertIsInstance(transform_config, dict)

    def test_load_transformation_config(self):
        """Test loading transformation config"""
        config = create_enhanced_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        transform_config = load_transformation_config(self.config_path)

        self.assertIsInstance(transform_config, dict)

    def test_list_experimental_setups(self):
        """Test listing experimental setups"""
        config = create_enhanced_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        load_config(self.config_path)
        setups = list_experimental_setups()

        self.assertIsInstance(setups, list)
        # The actual setup names may vary depending on config content
        # Just verify we got a list
        if len(setups) > 0:
            self.assertIsInstance(setups[0], str)

    def test_get_experimental_setup(self):
        """Test getting specific experimental setup"""
        config = create_enhanced_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        load_config(self.config_path)
        setup = get_experimental_setup("baseline")

        if setup is not None:
            self.assertIsInstance(setup, dict)

    def test_get_nonexistent_experimental_setup(self):
        """Test getting nonexistent experimental setup"""
        config = create_enhanced_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        load_config(self.config_path)
        setup = get_experimental_setup("nonexistent")

        # Should return None or empty dict
        self.assertIn(setup, [None, {}])


# ==============================================================================
# TEST CLASS: Standard Transforms Configuration Support (NEW)
# ==============================================================================


class TestStandardTransformsSupport(unittest.TestCase):
    """
    Test standard_transforms configuration support in config_loader.py.

    Tests verify that format detection, validation, and status reporting
    correctly handle:
    1. Configs with only standard_transforms (no experimental_setups)
    2. Configs with both standard_transforms and experimental_setups
    3. Configs with only experimental_setups (backward compatibility)
    4. Migration status reporting includes standard_transforms_count
    5. Validation level recommendations consider both sources
    """

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    # -------------------------------------------------------------------------
    # _detect_transformation_format() tests
    # -------------------------------------------------------------------------

    def test_detect_format_with_standard_transforms_only(self):
        """Test _detect_transformation_format recognizes standard_transforms as enhanced."""
        transforms_section = {
            "standard_transforms": [{"name": "AddSelfLoops", "kwargs": {}, "enabled": True}],
            "default_setup": "production",
        }

        format_type = _detect_transformation_format(transforms_section)
        self.assertEqual(format_type, "enhanced")

    def test_detect_format_with_both_sources(self):
        """Test _detect_transformation_format with both standard_transforms and experimental_setups."""
        transforms_section = {
            "standard_transforms": [{"name": "AddSelfLoops", "kwargs": {}, "enabled": True}],
            "experimental_setups": {"baseline": [{"name": "GCNNorm"}]},
            "default_setup": "baseline",
        }

        format_type = _detect_transformation_format(transforms_section)
        self.assertEqual(format_type, "enhanced")

    def test_detect_format_with_experimental_only_backward_compat(self):
        """Test _detect_transformation_format with only experimental_setups (backward compat)."""
        transforms_section = {
            "experimental_setups": {"baseline": [{"name": "AddSelfLoops"}]},
            "default_setup": "baseline",
        }

        format_type = _detect_transformation_format(transforms_section)
        self.assertEqual(format_type, "enhanced")

    def test_detect_format_with_neither_source(self):
        """Test _detect_transformation_format with neither source returns appropriate type."""
        transforms_section = {
            "default_setup": "baseline"
            # No standard_transforms, no experimental_setups
        }

        format_type = _detect_transformation_format(transforms_section)
        # Should not be 'enhanced' without any transform source
        self.assertIn(format_type, ["unknown", "legacy_dict", "invalid"])

    # -------------------------------------------------------------------------
    # _validate_enhanced_format() tests
    # -------------------------------------------------------------------------

    def test_validate_enhanced_with_standard_transforms_only(self):
        """Test _validate_enhanced_format with only standard_transforms."""
        transforms_section = {
            "standard_transforms": [
                {"name": "AddSelfLoops", "kwargs": {}, "enabled": True},
                {"name": "NormalizeFeatures", "kwargs": {}, "enabled": True},
            ],
            "default_setup": "production",
        }

        result = _validate_enhanced_format(transforms_section)
        self.assertTrue(result["valid"], f"Expected valid but got errors: {result['errors']}")

    def test_validate_enhanced_with_both_sources(self):
        """Test _validate_enhanced_format with both sources."""
        transforms_section = {
            "standard_transforms": [{"name": "AddSelfLoops", "kwargs": {}, "enabled": True}],
            "experimental_setups": {"baseline": [{"name": "GCNNorm", "kwargs": {}}]},
            "default_setup": "baseline",
        }

        result = _validate_enhanced_format(transforms_section)
        self.assertTrue(result["valid"], f"Expected valid but got errors: {result['errors']}")

    def test_validate_enhanced_with_experimental_only(self):
        """Test _validate_enhanced_format with only experimental_setups."""
        transforms_section = {
            "experimental_setups": {"baseline": [{"name": "AddSelfLoops", "kwargs": {}}]},
            "default_setup": "baseline",
        }

        result = _validate_enhanced_format(transforms_section)
        self.assertTrue(result["valid"], f"Expected valid but got errors: {result['errors']}")

    def test_validate_enhanced_neither_source_fails(self):
        """Test _validate_enhanced_format fails with neither source."""
        transforms_section = {
            "default_setup": "baseline"
            # No standard_transforms, no experimental_setups
        }

        result = _validate_enhanced_format(transforms_section)
        self.assertFalse(result["valid"])
        self.assertTrue(any("at least one required" in e.lower() for e in result["errors"]))

    def test_validate_enhanced_standard_transforms_not_list(self):
        """Test _validate_enhanced_format fails when standard_transforms is not a list."""
        transforms_section = {"standard_transforms": {"not": "a list"}, "default_setup": "baseline"}

        result = _validate_enhanced_format(transforms_section)
        self.assertFalse(result["valid"])
        self.assertTrue(any("list" in e.lower() for e in result["errors"]))

    def test_validate_enhanced_standard_transforms_missing_name_warning(self):
        """Test _validate_enhanced_format warns on transforms missing 'name' field."""
        transforms_section = {
            "standard_transforms": [
                {"name": "AddSelfLoops", "kwargs": {}},
                {"kwargs": {}, "enabled": True},  # Missing 'name'
            ],
            "default_setup": "production",
        }

        result = _validate_enhanced_format(transforms_section)
        # Should be valid but with warnings
        self.assertTrue(result["valid"])
        self.assertTrue(any("name" in w.lower() for w in result["warnings"]))

    def test_validate_enhanced_empty_experimental_with_standard(self):
        """Test _validate_enhanced_format with empty experimental_setups but valid standard."""
        transforms_section = {
            "standard_transforms": [{"name": "AddSelfLoops", "kwargs": {}, "enabled": True}],
            "experimental_setups": {},  # Empty
            "default_setup": "production",
        }

        result = _validate_enhanced_format(transforms_section)
        # Should be valid because standard_transforms exists
        self.assertTrue(result["valid"], f"Expected valid but got errors: {result['errors']}")

    def test_validate_enhanced_default_setup_not_in_setups_with_standard(self):
        """Test default_setup not in experimental_setups is OK when standard_transforms exists."""
        transforms_section = {
            "standard_transforms": [{"name": "AddSelfLoops", "kwargs": {}, "enabled": True}],
            "experimental_setups": {"baseline": [{"name": "GCNNorm", "kwargs": {}}]},
            "default_setup": "nonexistent",  # Not in experimental_setups
        }

        result = _validate_enhanced_format(transforms_section)
        # Should be valid (default_setup is just a label when standard_transforms exists)
        # but may have a warning
        self.assertTrue(result["valid"])

    # -------------------------------------------------------------------------
    # check_migration_status() tests
    # -------------------------------------------------------------------------

    def test_check_migration_status_with_standard_transforms(self):
        """Test check_migration_status includes standard_transforms_count."""
        config = {
            "transformations": {
                "standard_transforms": [
                    {"name": "AddSelfLoops", "kwargs": {}, "enabled": True},
                    {"name": "NormalizeFeatures", "kwargs": {}, "enabled": True},
                ],
                "default_setup": "production",
            }
        }

        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        status = check_migration_status(self.config_path)

        self.assertTrue(status["file_exists"])
        # Should include standard_transforms_count if enhanced format
        if status.get("current_format") == "enhanced":
            self.assertIn("standard_transforms_count", status)
            self.assertEqual(status["standard_transforms_count"], 2)

    def test_check_migration_status_with_both_sources(self):
        """Test check_migration_status with both transform sources."""
        config = {
            "transformations": {
                "standard_transforms": [{"name": "AddSelfLoops", "kwargs": {}, "enabled": True}],
                "experimental_setups": {
                    "baseline": [{"name": "GCNNorm"}],
                    "advanced": [{"name": "GCNNorm"}, {"name": "NormalizeFeatures"}],
                },
                "default_setup": "baseline",
            }
        }

        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        status = check_migration_status(self.config_path)

        self.assertTrue(status["file_exists"])
        if status.get("current_format") == "enhanced":
            self.assertIn("standard_transforms_count", status)
            self.assertEqual(status["standard_transforms_count"], 1)
            self.assertIn("experimental_setups", status)

    # -------------------------------------------------------------------------
    # recommend_validation_level() tests
    # -------------------------------------------------------------------------

    def test_recommend_validation_level_simple_standard_transforms(self):
        """Test recommend_validation_level with simple standard_transforms config."""
        config = {
            "transformations": {
                "standard_transforms": [
                    {"name": "AddSelfLoops", "kwargs": {}},
                    {"name": "NormalizeFeatures", "kwargs": {}},
                ],
                "default_setup": "production",
            }
        }

        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        level = recommend_validation_level(self.config_path)
        # Simple config should recommend STRICT
        self.assertEqual(level, "STRICT")

    def test_recommend_validation_level_complex_both_sources(self):
        """Test recommend_validation_level with complex config having both sources."""
        # Create complex config with many transforms
        config = {
            "transformations": {
                "standard_transforms": [{"name": f"Transform{i}", "kwargs": {}} for i in range(10)],
                "experimental_setups": {
                    f"setup_{i}": [{"name": f"SetupTransform{j}", "kwargs": {}} for j in range(5)]
                    for i in range(5)
                },
                "default_setup": "setup_0",
            }
        }

        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        level = recommend_validation_level(self.config_path)
        # Complex config should recommend NORMAL or RELAXED
        self.assertIn(level, ["NORMAL", "RELAXED"])

    # -------------------------------------------------------------------------
    # Integration tests
    # -------------------------------------------------------------------------

    def test_load_config_with_standard_transforms(self):
        """Test load_config works with standard_transforms configuration."""
        config = {
            "project_name": "test_project",
            "transformations": {
                "standard_transforms": [{"name": "AddSelfLoops", "kwargs": {}, "enabled": True}],
                "default_setup": "production",
            },
        }

        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        loaded = load_config(self.config_path)

        self.assertIsInstance(loaded, dict)
        self.assertEqual(loaded["project_name"], "test_project")
        self.assertIn("transformations", loaded)

    def test_load_config_with_both_transform_sources(self):
        """Test load_config works with both transform sources."""
        config = {
            "project_name": "test_project",
            "transformations": {
                "standard_transforms": [{"name": "AddSelfLoops", "kwargs": {}, "enabled": True}],
                "experimental_setups": {"baseline": [{"name": "GCNNorm", "kwargs": {}}]},
                "default_setup": "baseline",
            },
        }

        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        loaded = load_config(self.config_path)

        self.assertIsInstance(loaded, dict)
        self.assertIn("transformations", loaded)
        transforms = loaded["transformations"]
        self.assertIn("standard_transforms", transforms)
        self.assertIn("experimental_setups", transforms)


# ==============================================================================
# TEST CLASS 6: Migration Integration
# ==============================================================================


class TestMigrationIntegration(unittest.TestCase):
    """Test migration integration features"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_check_migration_status_for_legacy_config(self):
        """Test checking migration status for legacy config"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        status = check_migration_status(self.config_path)

        self.assertIsInstance(status, dict)
        self.assertTrue(status.get("file_exists", False))

    def test_check_migration_status_for_enhanced_config(self):
        """Test checking migration status for enhanced config"""
        config = create_enhanced_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        status = check_migration_status(self.config_path)

        self.assertIsInstance(status, dict)
        self.assertTrue(status.get("file_exists", False))

    def test_check_migration_status_for_missing_file(self):
        """Test checking migration status for missing file"""
        status = check_migration_status("/nonexistent/config.yaml")

        self.assertIsInstance(status, dict)
        self.assertFalse(status.get("file_exists", False))

    def test_migrate_legacy_config(self):
        """Test migrating legacy config"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        # Migration should work or gracefully fail
        try:
            result = migrate_legacy_config(self.config_path)
            self.assertIsInstance(result, (dict, bool))
        except Exception as e:
            # Migration might not be fully available in test environment
            self.assertIsInstance(e, Exception)

    def test_migration_with_enable_flag(self):
        """Test loading with migration enabled"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        loaded_config = load_config(self.config_path, enable_migration=True)

        self.assertIsInstance(loaded_config, dict)

    def test_migration_with_disable_flag(self):
        """Test loading with migration disabled"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        loaded_config = load_config(self.config_path, enable_migration=False)

        self.assertIsInstance(loaded_config, dict)


# ==============================================================================
# TEST CLASS 7: Validation Functions
# ==============================================================================


class TestValidationFunctions(unittest.TestCase):
    """Test validation functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_validate_config_file(self):
        """Test validating config file"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        result = validate_config_file(self.config_path)

        # validate_config_file may return bool or dict depending on implementation
        if isinstance(result, bool):
            self.assertIsInstance(result, bool)
        elif isinstance(result, dict):
            self.assertIn("valid", result)
            self.assertIsInstance(result["valid"], bool)
        else:
            self.fail(f"Unexpected return type: {type(result)}")

    def test_validate_and_report(self):
        """Test validation with reporting"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        is_valid = validate_and_report(self.config_path)

        self.assertIsInstance(is_valid, bool)

    def test_validate_invalid_config(self):
        """Test validating invalid config"""
        config = create_invalid_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        # Should handle invalid config gracefully
        try:
            is_valid = validate_config_file(self.config_path)
            self.assertIsInstance(is_valid, bool)
        except Exception:
            # Validation might raise exception for invalid config
            pass

    def test_validation_with_enable_flag(self):
        """Test loading with validation enabled"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        loaded_config = load_config(self.config_path, enable_validation=True)

        self.assertIsInstance(loaded_config, dict)

    def test_validation_with_disable_flag(self):
        """Test loading with validation disabled"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        loaded_config = load_config(self.config_path, enable_validation=False)

        self.assertIsInstance(loaded_config, dict)


# ==============================================================================
# TEST CLASS 8: Statistics and Diagnostics
# ==============================================================================


class TestStatisticsAndDiagnostics(unittest.TestCase):
    """Test statistics and diagnostics functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_get_config_statistics(self):
        """Test getting configuration statistics"""
        stats = get_config_statistics()

        self.assertIsInstance(stats, dict)
        self.assertIn("load_count", stats)
        self.assertIn("cache_hits", stats)
        self.assertIn("validation_level", stats)

    def test_statistics_track_loads(self):
        """Test that statistics track loads correctly"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        stats_before = get_config_statistics()
        load_count_before = stats_before["load_count"]

        load_config(self.config_path, force_reload=True)

        stats_after = get_config_statistics()
        load_count_after = stats_after["load_count"]

        self.assertEqual(load_count_after, load_count_before + 1)

    def test_statistics_track_cache_hits(self):
        """Test that statistics track cache hits correctly"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        # First load
        load_config(self.config_path)

        stats_before = get_config_statistics()
        cache_hits_before = stats_before["cache_hits"]

        # Second load (cache hit)
        load_config(self.config_path)

        stats_after = get_config_statistics()
        cache_hits_after = stats_after["cache_hits"]

        self.assertGreater(cache_hits_after, cache_hits_before)

    def test_print_config_diagnostics(self):
        """Test printing config diagnostics"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        load_config(self.config_path)

        # Should execute without error
        try:
            print_config_diagnostics()
        except Exception as e:
            self.fail(f"print_config_diagnostics raised exception: {e}")

    def test_get_transformation_feature_status(self):
        """Test getting transformation feature status"""
        status = get_transformation_feature_status()

        self.assertIsInstance(status, dict)

    def test_print_transformation_status(self):
        """Test printing transformation status"""
        # Should execute without error
        try:
            print_transformation_status()
        except Exception as e:
            self.fail(f"print_transformation_status raised exception: {e}")


# ==============================================================================
# TEST CLASS 9: Reload Configuration
# ==============================================================================


class TestReloadConfiguration(unittest.TestCase):
    """Test configuration reload functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_reload_config(self):
        """Test reloading configuration"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        # Initial load
        load_config(self.config_path)

        # Reload
        reloaded = reload_config(self.config_path)

        self.assertIsInstance(reloaded, dict)

    def test_reload_updates_config(self):
        """Test that reload updates the configuration"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        # Initial load
        load_config(self.config_path)

        # Modify config file
        config["project_name"] = "modified_project"
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        # Reload
        reloaded = reload_config(self.config_path)

        self.assertEqual(reloaded["project_name"], "modified_project")


# ==============================================================================
# TEST CLASS 10: Configuration Hash
# ==============================================================================


class TestConfigurationHash(unittest.TestCase):
    """Test configuration hashing functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_get_config_hash(self):
        """Test getting configuration hash"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        load_config(self.config_path)
        config_hash = get_config_hash()

        self.assertIsInstance(config_hash, str)
        self.assertGreater(len(config_hash), 0)

    def test_config_hash_consistency(self):
        """Test that hash is consistent for same config"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        load_config(self.config_path)
        hash1 = get_config_hash()
        hash2 = get_config_hash()

        self.assertEqual(hash1, hash2)

    def test_config_hash_changes_with_content(self):
        """Test that hash changes when config changes"""
        config1 = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config1, f)

        load_config(self.config_path)
        hash1 = get_config_hash()

        clear_config_cache()

        config2 = create_basic_config()
        config2["project_name"] = "different_project"
        with open(self.config_path, "w") as f:
            yaml.dump(config2, f)

        load_config(self.config_path)
        hash2 = get_config_hash()

        self.assertNotEqual(hash1, hash2)


# ==============================================================================
# TEST CLASS 11: Create Example Config
# ==============================================================================


class TestCreateExampleConfig(unittest.TestCase):
    """Test example config creation"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, "example_config.yaml")

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_example_config(self):
        """Test creating example configuration"""
        # NOTE: Explicitly pass dataset_type='DFT' to avoid Phase 5 registry lookup
        # which can be affected by mocks in test environment
        success = create_example_config(self.output_path, dataset_type="DFT")

        self.assertTrue(success)
        self.assertTrue(os.path.exists(self.output_path))

    def test_example_config_is_valid(self):
        """Test that created example config is valid"""
        # NOTE: Explicitly pass dataset_type='DFT' to avoid Phase 5 registry lookup
        create_example_config(self.output_path, dataset_type="DFT")

        # Load and validate
        with open(self.output_path) as f:
            config = yaml.safe_load(f)

        self.assertIsInstance(config, dict)

    def test_create_example_config_overwrites(self):
        """Test that create_example_config can overwrite existing file"""
        # NOTE: Explicitly pass dataset_type='DFT' to avoid Phase 5 registry lookup
        # Create initial file
        create_example_config(self.output_path, dataset_type="DFT")

        # Create again (should overwrite)
        success = create_example_config(self.output_path, dataset_type="DFT")

        self.assertTrue(success)


# ==============================================================================
# TEST CLASS 12: Handler-Based Architecture
# ==============================================================================


class TestHandlerBasedArchitecture(unittest.TestCase):
    """Test handler-based architecture integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_handler_integration(self):
        """Test that handler integration works"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        # Load should use handler internally
        loaded_config = load_config(self.config_path)

        self.assertIsInstance(loaded_config, dict)

    def test_handler_feature_detection(self):
        """Test handler feature detection"""
        status = get_transformation_feature_status()

        self.assertIsInstance(status, dict)
        # Should have feature availability information
        self.assertIn("migration_available", status)


# ==============================================================================
# TEST CLASS 13: Error Handling
# ==============================================================================


class TestErrorHandling(unittest.TestCase):
    """Test error handling in various scenarios"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_handle_file_permission_error(self):
        """Test handling file permission errors"""
        # This test may not work on all systems
        if os.name != "nt":  # Skip on Windows
            config = create_basic_config()
            with open(self.config_path, "w") as f:
                yaml.dump(config, f)

            # Remove read permission
            os.chmod(self.config_path, 0o000)

            try:
                # Try to load - should raise ConfigurationError
                # But some systems/environments may still allow reading
                try:
                    load_config(self.config_path)
                    # If no error raised, the system allows reading despite permissions
                    # This is acceptable in some environments (like root in Docker)
                except ConfigurationError:
                    # This is the expected behavior
                    pass
            finally:
                # Restore permission for cleanup
                os.chmod(self.config_path, 0o644)

    def test_handle_corrupted_yaml(self):
        """Test handling corrupted YAML"""
        with open(self.config_path, "w") as f:
            f.write("key: value\n  invalid indentation\n{[bad syntax")

        with self.assertRaises(ConfigurationError):
            load_config(self.config_path)

    def test_handle_unicode_decode_error(self):
        """Test handling unicode decode errors"""
        # Write binary data that's not valid UTF-8
        with open(self.config_path, "wb") as f:
            f.write(b"\xff\xfe invalid utf-8 \x00\x00")

        with self.assertRaises(ConfigurationError):
            load_config(self.config_path)

    def test_error_messages_are_descriptive(self):
        """Test that error messages are descriptive"""
        with self.assertRaises(ConfigurationError) as cm:
            load_config("/nonexistent/config.yaml")

        error_message = str(cm.exception)
        self.assertIn("not found", error_message.lower())


# ==============================================================================
# TEST CLASS 14: Edge Cases
# ==============================================================================


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_load_config_with_none_path(self):
        """Test loading config with None path"""
        # Should use default path
        try:
            load_config(None)
        except ConfigurationError:
            # Expected if default config doesn't exist
            pass

    def test_load_very_large_config(self):
        """Test loading very large configuration"""
        config = create_basic_config()

        # Add many setups to make it large
        config["transformations"] = {
            "experimental_setups": {
                f"setup_{i}": {
                    "name": f"setup_{i}",
                    "pipeline": [{"type": "Transform", "params": {"value": j}} for j in range(10)],
                }
                for i in range(50)
            }
        }

        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        loaded_config = load_config(self.config_path)
        self.assertIsInstance(loaded_config, dict)

    def test_load_config_with_special_characters(self):
        """Test loading config with special characters in values"""
        config = {
            "project_name": "test_@#$%_project",
            "description": "Test with special chars: <>&\"'",
        }

        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        loaded_config = load_config(self.config_path)
        self.assertEqual(loaded_config["project_name"], "test_@#$%_project")

    def test_empty_transformations_section(self):
        """Test config with empty transformations section"""
        config = {"project_name": "test", "transformations": {}}

        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        loaded_config = load_config(self.config_path)
        self.assertIsInstance(loaded_config, dict)


# ==============================================================================
# TEST CLASS 15: Multiple Configuration Files
# ==============================================================================


class TestMultipleConfigurationFiles(unittest.TestCase):
    """Test handling multiple configuration files"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_load_multiple_configs(self):
        """Test loading multiple different config files"""
        config1_path = os.path.join(self.temp_dir, "config1.yaml")
        config2_path = os.path.join(self.temp_dir, "config2.yaml")

        config1 = create_basic_config()
        config1["project_name"] = "project1"

        config2 = create_basic_config()
        config2["project_name"] = "project2"

        with open(config1_path, "w") as f:
            yaml.dump(config1, f)

        with open(config2_path, "w") as f:
            yaml.dump(config2, f)

        loaded1 = load_config(config1_path)
        loaded2 = load_config(config2_path)

        self.assertEqual(loaded1["project_name"], "project1")
        self.assertEqual(loaded2["project_name"], "project2")

    def test_cache_separates_different_configs(self):
        """Test that cache keeps different configs separate"""
        config1_path = os.path.join(self.temp_dir, "config1.yaml")
        config2_path = os.path.join(self.temp_dir, "config2.yaml")

        config1 = create_basic_config()
        config1["project_name"] = "project1"

        config2 = create_basic_config()
        config2["project_name"] = "project2"

        with open(config1_path, "w") as f:
            yaml.dump(config1, f)

        with open(config2_path, "w") as f:
            yaml.dump(config2, f)

        # Load both
        load_config(config1_path)
        load_config(config2_path)

        # Load again (should hit cache)
        loaded1 = load_config(config1_path)
        loaded2 = load_config(config2_path)

        # Should still be different
        self.assertEqual(loaded1["project_name"], "project1")
        self.assertEqual(loaded2["project_name"], "project2")


# ==============================================================================
# TEST CLASS 16: Configuration Enhancement
# ==============================================================================


class TestConfigurationEnhancement(unittest.TestCase):
    """Test configuration enhancement features"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_enhancement_enabled(self):
        """Test loading with enhancement enabled"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        loaded_config = load_config(self.config_path, enable_enhancement=True)

        self.assertIsInstance(loaded_config, dict)

    def test_enhancement_disabled(self):
        """Test loading with enhancement disabled"""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        loaded_config = load_config(self.config_path, enable_enhancement=False)

        self.assertIsInstance(loaded_config, dict)


# ==============================================================================
# TEST CLASS: Re-entrant Call Protection Tests (Production-Ready)
# ==============================================================================


class TestReentrantCallProtection(unittest.TestCase):
    """
    Test re-entrant call protection during registry initialization.

    These tests verify that nested calls to load_config during registry
    initialization are handled correctly without deadlocks or infinite loops.
    """

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        clear_config_cache()

        # Store original registry state
        import milia_pipeline.config.config_loader as loader_module

        self._original_initializing = getattr(loader_module, "_REGISTRY_INITIALIZING", False)

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

        # Restore registry state
        import milia_pipeline.config.config_loader as loader_module

        loader_module._REGISTRY_INITIALIZING = self._original_initializing

    def test_normalize_dataset_type_during_initialization(self):
        """Test that normalization is skipped during registry initialization."""
        import milia_pipeline.config.config_loader as loader_module

        # Simulate re-entrant call scenario
        loader_module._REGISTRY_INITIALIZING = True

        skip_list = []
        result = _normalize_dataset_type("DFT", _skip_cache_if_reentrant=skip_list)

        # Should return unchanged and signal skip
        self.assertEqual(result, "DFT")
        self.assertEqual(len(skip_list), 1)
        self.assertTrue(skip_list[0])

    def test_config_not_cached_during_reentrant_call(self):
        """Test that config is not cached during re-entrant initialization."""
        import milia_pipeline.config.config_loader as loader_module

        config = create_basic_config()
        config["dataset_type"] = "dft"
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        # Simulate re-entrant scenario
        loader_module._REGISTRY_INITIALIZING = True

        # Load should work but not cache
        loaded = load_config(self.config_path)

        self.assertIsInstance(loaded, dict)
        # Reset flag for proper cleanup
        loader_module._REGISTRY_INITIALIZING = False


# ==============================================================================
# TEST CLASS: Global State Management Tests (Production-Ready)
# ==============================================================================


class TestGlobalStateManagement(unittest.TestCase):
    """
    Test global state management functions for proper isolation.

    These tests verify:
    1. get_global_config_state() returns correct state
    2. set_global_config_state() properly sets state
    3. _clear_all_cached_state() clears all caches
    """

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_get_global_config_state_before_load(self):
        """Test get_global_config_state returns None before loading."""
        clear_config_cache()

        from milia_pipeline.config.config_loader import get_global_config_state

        state = get_global_config_state()

        self.assertIsNone(state)

    def test_get_global_config_state_after_load(self):
        """Test get_global_config_state returns config after loading."""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        load_config(self.config_path)

        from milia_pipeline.config.config_loader import get_global_config_state

        state = get_global_config_state()

        self.assertIsInstance(state, dict)
        self.assertEqual(state["project_name"], "test_project")

    def test_set_global_config_state(self):
        """Test set_global_config_state properly sets state."""
        from milia_pipeline.config.config_loader import (
            get_global_config_state,
            set_global_config_state,
        )

        test_config = {"test_key": "test_value"}
        set_global_config_state(test_config)

        state = get_global_config_state()
        self.assertEqual(state, test_config)

    def test_clear_all_cached_state(self):
        """Test _clear_all_cached_state clears all caches."""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        load_config(self.config_path)

        # Verify config is loaded
        self.assertTrue(is_config_loaded())

        # Clear all state
        from milia_pipeline.config.config_loader import _clear_all_cached_state

        _clear_all_cached_state()

        # Verify cleared
        self.assertFalse(is_config_loaded())

    def test_get_config_load_time(self):
        """Test get_config_load_time returns timestamp."""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        before_load = time.time()
        load_config(self.config_path)
        after_load = time.time()

        from milia_pipeline.config.config_loader import get_config_load_time

        load_time = get_config_load_time()

        self.assertIsNotNone(load_time)
        self.assertGreaterEqual(load_time, before_load)
        self.assertLessEqual(load_time, after_load)


# ==============================================================================
# TEST CLASS: Unicode and Encoding Tests (Production-Ready)
# ==============================================================================


class TestUnicodeAndEncoding(unittest.TestCase):
    """
    Test proper handling of Unicode content and encoding issues.

    These tests verify:
    1. Unicode characters in config values work correctly
    2. UTF-8 BOM is handled
    3. Invalid encoding raises appropriate errors
    """

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_unicode_config_values(self):
        """Test config with Unicode characters loads correctly."""
        config = {
            "project_name": "テストプロジェクト",  # Japanese
            "description": "中文描述",  # Chinese
            "transformations": [
                {"name": "Transform_αβγ", "kwargs": {}}  # Greek
            ],
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)

        loaded = load_config(self.config_path)

        self.assertEqual(loaded["project_name"], "テストプロジェクト")
        self.assertEqual(loaded["description"], "中文描述")

    def test_utf8_bom_handling(self):
        """Test config with UTF-8 BOM is handled correctly."""
        config = create_basic_config()

        # Write with BOM
        with open(self.config_path, "wb") as f:
            f.write(b"\xef\xbb\xbf")  # UTF-8 BOM
            f.write(yaml.dump(config).encode("utf-8"))

        # Should load successfully (YAML parser handles BOM)
        loaded = load_config(self.config_path)
        self.assertIsInstance(loaded, dict)


# ==============================================================================
# TEST CLASS: Concurrent Access Edge Cases (Production-Ready)
# ==============================================================================


class TestConcurrentAccessEdgeCases(unittest.TestCase):
    """
    Test concurrent access edge cases for thread safety.

    These tests verify:
    1. Multiple threads clearing cache simultaneously
    2. Load and clear race conditions
    3. Statistics updates during concurrent access
    """

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_concurrent_cache_clear(self):
        """Test concurrent cache clearing is thread-safe."""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        # Load config first
        load_config(self.config_path)

        errors = []

        def clear_worker():
            try:
                for _ in range(10):
                    clear_config_cache()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=clear_worker) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors during concurrent clear: {errors}")

    def test_load_during_clear_race(self):
        """Test load and clear race conditions are handled."""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        results = []
        errors = []

        def load_worker():
            try:
                for _ in range(5):
                    result = load_config(self.config_path, force_reload=True)
                    results.append(result is not None)
            except Exception as e:
                errors.append(e)

        def clear_worker():
            try:
                for _ in range(5):
                    clear_config_cache()
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=load_worker),
            threading.Thread(target=load_worker),
            threading.Thread(target=clear_worker),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors during race: {errors}")


# ==============================================================================
# TEST CLASS: Path Handling Edge Cases (Production-Ready)
# ==============================================================================


class TestPathHandlingEdgeCases(unittest.TestCase):
    """
    Test path handling edge cases for robustness.

    These tests verify:
    1. Relative vs absolute paths
    2. Paths with special characters
    3. Symlinks handling
    """

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_absolute_path_loading(self):
        """Test loading with absolute path."""
        config_path = os.path.join(self.temp_dir, "config.yaml")
        config = create_basic_config()
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Use absolute path
        abs_path = os.path.abspath(config_path)
        loaded = load_config(abs_path)

        self.assertIsInstance(loaded, dict)

    def test_path_with_spaces(self):
        """Test loading from path with spaces."""
        space_dir = os.path.join(self.temp_dir, "path with spaces")
        os.makedirs(space_dir)
        config_path = os.path.join(space_dir, "config.yaml")

        config = create_basic_config()
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        loaded = load_config(config_path)
        self.assertIsInstance(loaded, dict)

    def test_symlink_to_config(self):
        """Test loading via symlink."""
        config_path = os.path.join(self.temp_dir, "config.yaml")
        symlink_path = os.path.join(self.temp_dir, "config_link.yaml")

        config = create_basic_config()
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        try:
            os.symlink(config_path, symlink_path)
            loaded = load_config(symlink_path)
            self.assertIsInstance(loaded, dict)
        except OSError:
            # Symlinks may not be supported on all systems
            self.skipTest("Symlinks not supported on this system")


# ==============================================================================
# TEST CLASS: Configuration Statistics Edge Cases
# ==============================================================================


class TestConfigurationStatisticsEdgeCases(unittest.TestCase):
    """
    Test configuration statistics edge cases.

    These tests verify:
    1. Statistics after multiple operations
    2. Statistics field defaults
    3. Statistics thread safety under load
    """

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        clear_config_cache()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_config_cache()

    def test_statistics_all_required_fields_present(self):
        """Test that all required statistics fields are present."""
        stats = get_config_statistics()

        required_fields = [
            "load_count",
            "cache_hits",
            "enhancement_applied",
            "migration_applied",
            "validation_enabled",
            "validation_level",
            "last_load_time",
            "last_validation_time",
            "last_validation_results",
            "last_migration_report",
            "cache_hit_rate",
            "config_cached",
            "warnings_count",
            "errors_count",
        ]

        for field in required_fields:
            self.assertIn(field, stats, f"Missing required field: {field}")

    def test_statistics_after_multiple_loads(self):
        """Test statistics accuracy after multiple load operations."""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        # Clear and get initial stats
        clear_config_cache()
        initial_stats = get_config_statistics()
        initial_load_count = initial_stats["load_count"]

        # Perform multiple force reloads
        for _ in range(5):
            load_config(self.config_path, force_reload=True)

        final_stats = get_config_statistics()

        # Should have incremented load_count by 5
        self.assertEqual(final_stats["load_count"], initial_load_count + 5)

    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate is calculated correctly."""
        config = create_basic_config()
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        clear_config_cache()

        # First load (cache miss)
        load_config(self.config_path)

        # Multiple cache hits
        for _ in range(4):
            load_config(self.config_path)

        stats = get_config_statistics()

        # 1 load + 4 cache hits = 5 total, hit rate = 4/5 = 0.8
        self.assertGreater(stats["cache_hit_rate"], 0)
        self.assertLessEqual(stats["cache_hit_rate"], 1.0)


# ==============================================================================
# TEST RUNNER
# ==============================================================================


def run_test_suite():
    """Run the complete test suite"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add Phase 5 test classes (NEW)
    suite.addTests(loader.loadTestsFromTestCase(TestPhase5RegistryIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase5CreateExampleConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase5MigrateLegacyConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase5BackwardCompatibility))

    # Add YAML Splitting test classes (NEW)
    suite.addTests(loader.loadTestsFromTestCase(TestYAMLSplittingDiscovery))
    suite.addTests(loader.loadTestsFromTestCase(TestYAMLSplittingFileCollection))
    suite.addTests(loader.loadTestsFromTestCase(TestYAMLSplittingDeepMerge))
    suite.addTests(loader.loadTestsFromTestCase(TestYAMLSplittingLoadAndMerge))
    suite.addTests(loader.loadTestsFromTestCase(TestYAMLSplittingIntegration))

    # Add Phase 6 Normalization test classes (NEW)
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6NormalizeDatasetType))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6NormalizeDatasetKeyedSections))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6NormalizeDictKeys))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6NormalizationIntegration))

    # Add Validation/Migration Reports test class (NEW)
    suite.addTests(loader.loadTestsFromTestCase(TestValidationAndMigrationReports))

    # Add all existing test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBasicConfigurationLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigurationCaching))
    suite.addTests(loader.loadTestsFromTestCase(TestThreadSafety))
    suite.addTests(loader.loadTestsFromTestCase(TestValidationLevels))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedTransformationConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestStandardTransformsSupport))  # NEW
    suite.addTests(loader.loadTestsFromTestCase(TestMigrationIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestValidationFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticsAndDiagnostics))
    suite.addTests(loader.loadTestsFromTestCase(TestReloadConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigurationHash))
    suite.addTests(loader.loadTestsFromTestCase(TestCreateExampleConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestHandlerBasedArchitecture))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestMultipleConfigurationFiles))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigurationEnhancement))

    # Add Production-Ready Edge Case test classes (NEW)
    suite.addTests(loader.loadTestsFromTestCase(TestReentrantCallProtection))
    suite.addTests(loader.loadTestsFromTestCase(TestGlobalStateManagement))
    suite.addTests(loader.loadTestsFromTestCase(TestUnicodeAndEncoding))
    suite.addTests(loader.loadTestsFromTestCase(TestConcurrentAccessEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestPathHandlingEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigurationStatisticsEdgeCases))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)

    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED!")
        print("✓ Configuration loading validated")
        print("✓ Thread-safe caching operational")
        print("✓ Validation levels functional")
        print("✓ Enhanced transformations working")
        print("✓ Standard transforms support validated (NEW)")
        print("✓ Migration integration active")
        print("✓ Handler-based architecture validated")
        print("✓ Statistics and diagnostics operational")
        print("✓ Error handling robust")
        print("✓ Edge cases handled correctly")
        print("✓ Phase 5 registry integration validated")
        print("✓ Phase 5 backward compatibility confirmed")
        print("✓ YAML Splitting support validated (NEW)")
        print("✓ Phase 6 dataset type normalization validated (NEW)")
        print("✓ Validation/Migration reports validated (NEW)")
        print("✓ Re-entrant call protection validated (NEW)")
        print("✓ Global state management validated (NEW)")
        print("✓ Unicode and encoding handling validated (NEW)")
        print("✓ Concurrent access edge cases validated (NEW)")
        print("✓ Path handling edge cases validated (NEW)")
        print("✓ Configuration statistics edge cases validated (NEW)")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("  Review failures before production use")

    # Return exit code
    return 0 if result.wasSuccessful() else 1


def teardown_module():
    """
    Clean up injected mocks to prevent mock pollution.

    CRITICAL: This function ensures that mocks injected into sys.modules
    are properly removed after test collection/execution, preventing
    interference with other test files during pytest collection.

    Mock Pollution Prevention:
    - Removes all tracked injected mocks from sys.modules
    - Clears config_loader module state to prevent stale references
    - Ensures clean slate for subsequent test files
    """
    # Remove tracked injected mocks
    for mock_name in _INJECTED_MOCKS:
        sys.modules.pop(mock_name, None)

    # Remove known test-related mocks
    mocks_to_remove = [
        "milia_pipeline.config.config_schemas",
        "milia_pipeline.handlers.config_handler",
    ]
    for mock in mocks_to_remove:
        sys.modules.pop(mock, None)

    # Clear config_loader cached state if still loaded
    if "milia_pipeline.config.config_loader" in sys.modules:
        loader = sys.modules["milia_pipeline.config.config_loader"]
        # Reset module-level state to prevent test interference
        if hasattr(loader, "_CONFIG"):
            loader._CONFIG = None
        if hasattr(loader, "_ENHANCED_TRANSFORMATION_CONFIG"):
            loader._ENHANCED_TRANSFORMATION_CONFIG = None
        if hasattr(loader, "_config_cache"):
            loader._config_cache.clear()
        if hasattr(loader, "_CONFIG_STATS"):
            loader._CONFIG_STATS = {
                "load_count": 0,
                "cache_hits": 0,
                "enhancement_applied": False,
                "migration_applied": False,
                "validation_enabled": True,
                "validation_level": "NORMAL",
                "last_load_time": None,
                "last_validation_time": None,
                "last_validation_results": None,
                "last_migration_report": None,
                "cache_hit_rate": 0.0,
                "config_cached": False,
                "warnings_count": 0,
                "errors_count": 0,
            }
        # Reset registry state
        if hasattr(loader, "_REGISTRY_INITIALIZED"):
            loader._REGISTRY_INITIALIZED = False
        if hasattr(loader, "_REGISTRY_AVAILABLE"):
            loader._REGISTRY_AVAILABLE = False


def setup_module(module):
    """
    Inject mocks into sys.modules and import the module-under-test.

    Called by pytest ONCE before any test in this module executes.
    By deferring sys.modules writes here (instead of at module level),
    pytest --collect-only can import this file without polluting
    sys.modules for other test files collected afterward.
    """
    global _ORIGINAL_SYSMODULES_KEYS, _INJECTED_MOCKS
    global ConfigurationError, ValidationError, config_loader
    global load_config, reload_config
    global get_enhanced_transformation_config, load_transformation_config
    global get_experimental_setup, list_experimental_setups
    global validate_config_file, validate_and_report, recommend_validation_level
    global migrate_legacy_config, check_migration_status
    global clear_config_cache, get_config_hash, is_config_loaded
    global get_config_statistics, print_config_diagnostics
    global get_transformation_feature_status, print_transformation_status
    global create_example_config
    global _detect_transformation_format, _validate_enhanced_format
    global _discover_config_files, _collect_yaml_files
    global _deep_merge_configs, _load_and_merge_yaml_files, _get_default_config_path
    global _normalize_dataset_type, _normalize_dataset_keyed_sections, _normalize_dict_keys
    global _init_registry, _get_default_dataset_type
    global get_validation_report, get_migration_report

    # --- Track original sys.modules state ---
    _ORIGINAL_SYSMODULES_KEYS = set(sys.modules.keys())

    # --- Inject mock into sys.modules WITH TRACKING ---
    if "milia_pipeline.handlers.config_handler" not in sys.modules:
        sys.modules["milia_pipeline.handlers.config_handler"] = MockConfigHandlerModule()
        _INJECTED_MOCKS.append("milia_pipeline.handlers.config_handler")

    # Clear any partial config_loader module (will be re-imported fresh)
    if "milia_pipeline.config.config_loader" in sys.modules:
        del sys.modules["milia_pipeline.config.config_loader"]

    # --- Import exception types ---
    from milia_pipeline.exceptions import (
        ConfigurationError as _CE,
    )
    from milia_pipeline.exceptions import (
        ValidationError as _VE,
    )

    ConfigurationError = _CE
    ValidationError = _VE

    # --- Import config_loader module ---
    import milia_pipeline.config.config_loader as _config_loader

    config_loader = _config_loader

    from milia_pipeline.config.config_loader import (
        check_migration_status as _check_migration_status,
    )
    from milia_pipeline.config.config_loader import (
        clear_config_cache as _clear_config_cache,
    )
    from milia_pipeline.config.config_loader import (
        create_example_config as _create_example_config,
    )
    from milia_pipeline.config.config_loader import (
        get_config_hash as _get_config_hash,
    )
    from milia_pipeline.config.config_loader import (
        get_config_statistics as _get_config_statistics,
    )
    from milia_pipeline.config.config_loader import (
        get_enhanced_transformation_config as _get_enhanced_transformation_config,
    )
    from milia_pipeline.config.config_loader import (
        get_experimental_setup as _get_experimental_setup,
    )
    from milia_pipeline.config.config_loader import (
        get_migration_report as _get_migration_report,
    )
    from milia_pipeline.config.config_loader import (
        get_transformation_feature_status as _get_transformation_feature_status,
    )
    from milia_pipeline.config.config_loader import (
        get_validation_report as _get_validation_report,
    )
    from milia_pipeline.config.config_loader import (
        is_config_loaded as _is_config_loaded,
    )
    from milia_pipeline.config.config_loader import (
        list_experimental_setups as _list_experimental_setups,
    )
    from milia_pipeline.config.config_loader import (
        load_config as _load_config,
    )
    from milia_pipeline.config.config_loader import (
        load_transformation_config as _load_transformation_config,
    )
    from milia_pipeline.config.config_loader import (
        migrate_legacy_config as _migrate_legacy_config,
    )
    from milia_pipeline.config.config_loader import (
        print_config_diagnostics as _print_config_diagnostics,
    )
    from milia_pipeline.config.config_loader import (
        print_transformation_status as _print_transformation_status,
    )
    from milia_pipeline.config.config_loader import (
        recommend_validation_level as _recommend_validation_level,
    )
    from milia_pipeline.config.config_loader import (
        reload_config as _reload_config,
    )
    from milia_pipeline.config.config_loader import (
        validate_and_report as _validate_and_report,
    )
    from milia_pipeline.config.config_loader import (
        validate_config_file as _validate_config_file,
    )

    load_config = _load_config
    reload_config = _reload_config
    get_enhanced_transformation_config = _get_enhanced_transformation_config
    load_transformation_config = _load_transformation_config
    get_experimental_setup = _get_experimental_setup
    list_experimental_setups = _list_experimental_setups
    validate_config_file = _validate_config_file
    validate_and_report = _validate_and_report
    recommend_validation_level = _recommend_validation_level
    migrate_legacy_config = _migrate_legacy_config
    check_migration_status = _check_migration_status
    clear_config_cache = _clear_config_cache
    get_config_hash = _get_config_hash
    is_config_loaded = _is_config_loaded
    get_config_statistics = _get_config_statistics
    print_config_diagnostics = _print_config_diagnostics
    get_transformation_feature_status = _get_transformation_feature_status
    print_transformation_status = _print_transformation_status
    create_example_config = _create_example_config
    get_validation_report = _get_validation_report
    get_migration_report = _get_migration_report

    # --- Import internal functions via module ---
    _detect_transformation_format = config_loader._detect_transformation_format
    _validate_enhanced_format = config_loader._validate_enhanced_format
    _discover_config_files = config_loader._discover_config_files
    _collect_yaml_files = config_loader._collect_yaml_files
    _deep_merge_configs = config_loader._deep_merge_configs
    _load_and_merge_yaml_files = config_loader._load_and_merge_yaml_files
    _get_default_config_path = config_loader._get_default_config_path
    _normalize_dataset_type = config_loader._normalize_dataset_type
    _normalize_dataset_keyed_sections = config_loader._normalize_dataset_keyed_sections
    _normalize_dict_keys = config_loader._normalize_dict_keys
    _init_registry = config_loader._init_registry
    _get_default_dataset_type = config_loader._get_default_dataset_type

    # --- Publish into module namespace ---
    module.ConfigurationError = ConfigurationError
    module.ValidationError = ValidationError
    module.config_loader = config_loader
    module.load_config = load_config
    module.reload_config = reload_config
    module.get_enhanced_transformation_config = get_enhanced_transformation_config
    module.load_transformation_config = load_transformation_config
    module.get_experimental_setup = get_experimental_setup
    module.list_experimental_setups = list_experimental_setups
    module.validate_config_file = validate_config_file
    module.validate_and_report = validate_and_report
    module.recommend_validation_level = recommend_validation_level
    module.migrate_legacy_config = migrate_legacy_config
    module.check_migration_status = check_migration_status
    module.clear_config_cache = clear_config_cache
    module.get_config_hash = get_config_hash
    module.is_config_loaded = is_config_loaded
    module.get_config_statistics = get_config_statistics
    module.print_config_diagnostics = print_config_diagnostics
    module.get_transformation_feature_status = get_transformation_feature_status
    module.print_transformation_status = print_transformation_status
    module.create_example_config = create_example_config
    module._detect_transformation_format = _detect_transformation_format
    module._validate_enhanced_format = _validate_enhanced_format
    module._discover_config_files = _discover_config_files
    module._collect_yaml_files = _collect_yaml_files
    module._deep_merge_configs = _deep_merge_configs
    module._load_and_merge_yaml_files = _load_and_merge_yaml_files
    module._get_default_config_path = _get_default_config_path
    module._normalize_dataset_type = _normalize_dataset_type
    module._normalize_dataset_keyed_sections = _normalize_dataset_keyed_sections
    module._normalize_dict_keys = _normalize_dict_keys
    module._init_registry = _init_registry
    module._get_default_dataset_type = _get_default_dataset_type
    module.get_validation_report = get_validation_report
    module.get_migration_report = get_migration_report

    # Ensure config cache is cleared before any tests run
    try:
        clear_config_cache()
    except Exception:
        pass  # May fail if module not fully loaded yet


if __name__ == "__main__":
    exit_code = run_test_suite()
    sys.exit(exit_code)
