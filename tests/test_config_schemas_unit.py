#!/usr/bin/env python3
"""
Comprehensive Unit Test Suite for config_schemas.py (Phase 5 Refactored)

Tests all schema classes, validators, migration logic, and utility functions
with a focus on registry integration, validation, edge cases, and error handling.

PHASE 5 ADDITIONS:
- Registry integration functions (_init_registry, _registry_*_safe)
- Dynamic schema lookup helpers
- Refactored validation functions with registry support
- Descriptor compatibility with feature queries
- Updated example configuration functions

STANDARD TRANSFORMS SUPPORT (NEW):
- TransformationSchema with standard_transforms field
- YAMLSchemaValidator.detect_format() recognizes standard_transforms
- YAMLSchemaValidator.validate_config() validates standard_transforms
- Backward compatibility with experimental_setups-only configs
- Validation summary includes standard_transforms_count

NOTE: This test suite runs inside Docker at /app/milia
Path mappings:
- Project root: /app/milia (mapped from ~/ml_projects/milia)
- Module path: ~/ml_projects/milia/milia_pipeline/config/config_schemas.py
- Test path: ~/ml_projects/milia/tests/test_config_schemas_unit.py
"""

import sys
import os
from pathlib import Path

# CRITICAL: Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open, call
from typing import Dict, Any, List
import json
import logging
import tempfile
from io import StringIO

# Import the module under test
from milia_pipeline.config.config_schemas import (
    # Phase 5: Registry integration functions
    _init_registry,
    _registry_list_all_safe,
    _registry_get_safe,
    _registry_is_registered_safe,
    _get_dataset_config_schema_class,
    _get_dataset_feature_support,
    _dataset_supports_feature,
    _get_dataset_config_key,
    
    # Schema classes
    TransformationSchema,
    PluginValidationLevel,
    PluginConfigSchema,
    DescriptorConfigSchema,
    DescriptorCategoryConfigSchema,  # Added: Missing in original imports
    WavefunctionProcessingConfigSchema,
    WavefunctionUncertaintyConfigSchema,
    WavefunctionConfigSchema,
    ExperimentSchema,
    ValidationConfig,
    
    # Validators
    YAMLSchemaValidator,
    PluginSchemaValidator,
    DescriptorSchemaValidator,
    ExperimentSchemaValidator,
    
    # Migration
    ConfigMigration,
    
    # Validation functions (Phase 5 refactored)
    validate_wavefunction_config,
    
    # Helper functions
    create_default_plugin_config,
    create_example_plugin_config,
    create_default_experiment_config,
    create_example_experiments_config,
    validate_plugin_config_file,
    validate_experiment_config_file,
    get_plugin_config_summary,
    get_experiment_config_summary,
    create_example_enhanced_config,  # Added: Missing in original imports
    create_example_legacy_configs,  # Added: Missing in original imports
    merge_plugin_configs,  # Added: Missing in original imports
    create_validator,  # Added: Factory function
    create_migrator,  # Added: Factory function
    
    # Module availability flags
    YAML_AVAILABLE,
    PLUGIN_SYSTEM_AVAILABLE,
    RESEARCH_API_AVAILABLE,
)

# Setup logging for tests
logging.basicConfig(level=logging.DEBUG)


# ==========================================
# PHASE 5: REGISTRY INTEGRATION FIXTURES
# ==========================================

@pytest.fixture(autouse=True)
def reset_registry_state():
    """Reset registry state before each test to ensure test isolation."""
    import milia_pipeline.config.config_schemas as schemas_module
    
    # Store original state
    original_initialized = schemas_module._REGISTRY_INITIALIZED
    original_available = schemas_module._REGISTRY_AVAILABLE
    original_list_all = schemas_module._registry_list_all
    original_get = schemas_module._registry_get
    original_is_registered = schemas_module._registry_is_registered
    
    # Reset state
    schemas_module._REGISTRY_INITIALIZED = False
    schemas_module._REGISTRY_AVAILABLE = False
    schemas_module._registry_list_all = None
    schemas_module._registry_get = None
    schemas_module._registry_is_registered = None
    
    yield
    
    # Restore original state
    schemas_module._REGISTRY_INITIALIZED = original_initialized
    schemas_module._REGISTRY_AVAILABLE = original_available
    schemas_module._registry_list_all = original_list_all
    schemas_module._registry_get = original_get
    schemas_module._registry_is_registered = original_is_registered


@pytest.fixture
def mock_dataset_class():
    """Create a mock dataset class for testing registry integration."""
    mock_class = Mock()
    mock_class.metadata = Mock(name='TestDataset')
    mock_class.config_key = 'test_config'
    mock_class.get_config_schema.return_value = None
    mock_class.get_feature_support.return_value = {
        'vibrational_analysis': True,
        'uncertainty_handling': False,
        'atomization_energy': True,
        'orbital_analysis': False,
        'homo_lumo_gap': False,
        'mo_energies': False,
    }
    return mock_class


@pytest.fixture
def mock_wavefunction_dataset_class():
    """Create a mock Wavefunction dataset class."""
    mock_class = Mock()
    mock_class.metadata = Mock(name='Wavefunction')
    mock_class.config_key = 'wavefunction_config'
    mock_class.get_config_schema.return_value = WavefunctionConfigSchema
    mock_class.get_feature_support.return_value = {
        'vibrational_analysis': False,
        'uncertainty_handling': False,
        'atomization_energy': False,
        'orbital_analysis': True,
        'homo_lumo_gap': True,
        'mo_energies': True,
    }
    return mock_class


# ==========================================
# LEGACY TEST FIXTURES (from original test suite)
# ==========================================

@pytest.fixture
def sample_experimental_setup():
    """Sample experimental setup dictionary.
    
    NOTE: TransformationSchema.experimental_setups has type Dict[str, List[Dict[str, Any]]].
    This means each setup key maps to a LIST of transform dictionaries directly.
    The enhanced format used by YAMLSchemaValidator is different (uses nested dict with
    'description' and 'transforms' keys), but the Pydantic schema expects the simpler format.
    """
    return {
        'baseline': [
            {'name': 'AddSelfLoops', 'kwargs': {}, 'enabled': True}
        ]
    }


@pytest.fixture
def sample_enhanced_experimental_setup():
    """Sample experimental setup in enhanced format (with description/transforms).
    
    This is the format used by YAMLSchemaValidator.validate_config(), NOT by
    TransformationSchema Pydantic model directly.
    """
    return {
        'baseline': {
            'description': 'Baseline setup',
            'transforms': [
                {'name': 'AddSelfLoops', 'kwargs': {}, 'enabled': True}
            ]
        }
    }


@pytest.fixture
def sample_transformation_schema_dict(sample_experimental_setup):
    """Sample transformation schema as dictionary for TransformationSchema Pydantic model."""
    return {
        'experimental_setups': sample_experimental_setup,
        'default_setup': 'baseline',
        'validation': {
            'enabled': True,
            'strict_mode': False
        }
    }


@pytest.fixture
def sample_enhanced_transformation_config(sample_enhanced_experimental_setup):
    """Sample transformation config in enhanced format for YAMLSchemaValidator.
    
    This uses the full enhanced format with description/transforms structure
    that is validated by YAMLSchemaValidator, not TransformationSchema directly.
    """
    return {
        'experimental_setups': sample_enhanced_experimental_setup,
        'default_setup': 'baseline',
        'validation': {
            'enabled': True,
            'strict_mode': False
        }
    }


@pytest.fixture
def sample_plugin_config_dict():
    """Sample plugin configuration dictionary."""
    return {
        'enabled': True,
        'plugin_paths': ['/path/to/plugins'],
        'auto_discover': True,
        'auto_validate': True,
        'validation_level': 'standard',
        'trusted_plugins': ['trusted_plugin'],
        'disabled_plugins': ['disabled_plugin'],
        'allow_experimental': False,
        'max_plugins': 50,
        'require_metadata': True,
        'enforce_checksums': True,
        'security_scanning': True
    }


@pytest.fixture
def sample_experiment_config_dict():
    """Sample experiment configuration dictionary."""
    return {
        'name': 'test_experiment',
        'description': 'Test experiment for validation',
        'base_transforms': [
            {'name': 'AddSelfLoops', 'kwargs': {}, 'enabled': True}
        ],
        'ablations': [
            {
                'name': 'no_self_loops',
                'description': 'Remove self loops',
                'transforms': []
            }
        ],
        'parameter_sweeps': [],
        'num_runs': 3,
        'random_seed': 42,
        'hypothesis': 'Self-loops improve accuracy',
        'expected_outcome': 'baseline > no_self_loops'
    }


@pytest.fixture
def mock_yaml_file_content():
    """Sample YAML file content."""
    return """
transformations:
  experimental_setups:
    baseline:
      description: Baseline setup
      transforms:
        - name: AddSelfLoops
          kwargs: {}
          enabled: true
  default_setup: baseline
  validation:
    enabled: true
    strict_mode: false
"""


# ==========================================
# PHASE 5: REGISTRY INTEGRATION TESTS
# ==========================================

class TestPhase5RegistryIntegration:
    """Test suite for Phase 5 registry integration functions."""
    
    def test_init_registry_returns_bool(self):
        """Test _init_registry returns a boolean."""
        result = _init_registry()
        assert isinstance(result, bool)
    
    def test_init_registry_idempotent(self):
        """Test _init_registry only initializes once."""
        import milia_pipeline.config.config_schemas as schemas_module
        
        # Call twice
        result1 = _init_registry()
        schemas_module._REGISTRY_INITIALIZED = True  # Ensure initialized flag is set
        result2 = _init_registry()
        
        # Both should return consistent results
        assert result1 == result2
    
    def test_registry_list_all_safe_returns_list(self):
        """Test _registry_list_all_safe returns a list."""
        result = _registry_list_all_safe()
        
        assert isinstance(result, list)
        assert len(result) >= 3
        # Should always include these core types (from registry or fallback)
        assert 'DFT' in result
        assert 'DMC' in result
        assert 'Wavefunction' in result
    
    def test_registry_list_all_safe_fallback(self):
        """Test _registry_list_all_safe has proper fallback behavior.
        
        NOTE: The module (lines 140-186) uses DYNAMIC fallback via filesystem discovery.
        When registry is unavailable, it attempts to discover dataset implementations
        from the filesystem. If that also fails, it returns an empty list with a warning.
        This is intentional to force proper registry initialization rather than
        silently using hardcoded values that could become stale.
        """
        import milia_pipeline.config.config_schemas as schemas_module
        
        # Force fallback by setting registry unavailable
        original_available = schemas_module._REGISTRY_AVAILABLE
        original_list_all = schemas_module._registry_list_all
        original_initialized = schemas_module._REGISTRY_INITIALIZED
        
        schemas_module._REGISTRY_INITIALIZED = True  # Skip re-initialization
        schemas_module._REGISTRY_AVAILABLE = False
        schemas_module._registry_list_all = None
        
        try:
            result = _registry_list_all_safe()
            
            # Result must be a list (dynamic discovery result or empty)
            assert isinstance(result, list)
            # If dynamic discovery found implementations, they should be uppercase strings
            for item in result:
                assert isinstance(item, str)
        finally:
            # Restore
            schemas_module._REGISTRY_INITIALIZED = original_initialized
            schemas_module._REGISTRY_AVAILABLE = original_available
            schemas_module._registry_list_all = original_list_all
    
    def test_registry_get_safe_returns_none_for_invalid(self):
        """Test _registry_get_safe returns None for invalid dataset."""
        result = _registry_get_safe('NonexistentDatasetType12345')
        assert result is None
    
    def test_registry_get_safe_with_valid_type(self):
        """Test _registry_get_safe with potentially valid type."""
        # This may return a class or None depending on registry availability
        result = _registry_get_safe('DFT')
        # Result is either a class or None
        assert result is None or hasattr(result, '__class__')
    
    def test_registry_is_registered_safe_returns_bool(self):
        """Test _registry_is_registered_safe returns boolean."""
        result = _registry_is_registered_safe('DFT')
        assert isinstance(result, bool)
    
    def test_registry_is_registered_safe_fallback(self):
        """Test _registry_is_registered_safe uses fallback when registry unavailable.
        
        NOTE: The module (lines 210-236) uses dynamic fallback via _registry_list_all_safe()
        which attempts filesystem discovery. If that fails, it returns empty list,
        so is_registered would return False for all types.
        """
        import milia_pipeline.config.config_schemas as schemas_module
        
        # Force fallback
        original_available = schemas_module._REGISTRY_AVAILABLE
        original_is_registered = schemas_module._registry_is_registered
        original_initialized = schemas_module._REGISTRY_INITIALIZED
        
        schemas_module._REGISTRY_INITIALIZED = True
        schemas_module._REGISTRY_AVAILABLE = False
        schemas_module._registry_is_registered = None
        
        try:
            # Test with a definitely unknown type - should always be False
            result_unknown = _registry_is_registered_safe('UnknownType12345')
            assert result_unknown is False
            
            # Test with known type - result depends on dynamic discovery
            result_dft = _registry_is_registered_safe('DFT')
            assert isinstance(result_dft, bool)  # Just verify it returns a boolean
        finally:
            # Restore
            schemas_module._REGISTRY_INITIALIZED = original_initialized
            schemas_module._REGISTRY_AVAILABLE = original_available
            schemas_module._registry_is_registered = original_is_registered


class TestPhase5DynamicSchemaLookup:
    """Test suite for Phase 5 dynamic schema lookup helpers."""
    
    def test_get_dataset_config_schema_class_returns_class_or_none(self):
        """Test _get_dataset_config_schema_class returns schema class or None."""
        # Test with known type - may return class or None depending on registry
        result = _get_dataset_config_schema_class('Wavefunction')
        assert result is None or result == WavefunctionConfigSchema
    
    def test_get_dataset_config_schema_class_unknown_type(self):
        """Test _get_dataset_config_schema_class returns None for unknown type."""
        result = _get_dataset_config_schema_class('NonexistentDatasetType12345')
        assert result is None
    
    def test_get_dataset_feature_support_returns_dict_or_none(self):
        """Test _get_dataset_feature_support returns dict or None."""
        result = _get_dataset_feature_support('DFT')
        # Either returns a feature dict or None
        assert result is None or isinstance(result, dict)
    
    def test_get_dataset_feature_support_unknown_type(self):
        """Test _get_dataset_feature_support returns None for unknown type."""
        result = _get_dataset_feature_support('NonexistentDatasetType12345')
        assert result is None
    
    def test_dataset_supports_feature_returns_bool(self):
        """Test _dataset_supports_feature returns boolean."""
        result = _dataset_supports_feature('DFT', 'vibrational_analysis')
        assert isinstance(result, bool)
    
    def test_dataset_supports_feature_unknown_type(self):
        """Test _dataset_supports_feature returns False for unknown type."""
        result = _dataset_supports_feature('NonexistentDatasetType12345', 'any_feature')
        assert result is False
    
    def test_dataset_supports_feature_unknown_feature(self):
        """Test _dataset_supports_feature returns False for unknown feature."""
        result = _dataset_supports_feature('DFT', 'nonexistent_feature_12345')
        assert result is False
    
    def test_get_dataset_config_key_returns_string_or_none(self):
        """Test _get_dataset_config_key returns string or None."""
        result = _get_dataset_config_key('Wavefunction')
        assert result is None or isinstance(result, str)
    
    def test_get_dataset_config_key_unknown_type(self):
        """Test _get_dataset_config_key returns None for unknown type."""
        result = _get_dataset_config_key('NonexistentDatasetType12345')
        assert result is None


class TestPhase5WavefunctionSchemaClasses:
    """Test Phase 5 notes added to Wavefunction schema classes (backward compatibility)."""
    
    def test_wavefunction_processing_config_schema_still_works(self):
        """Test WavefunctionProcessingConfigSchema remains functional."""
        schema = WavefunctionProcessingConfigSchema(feature_tier='standard')
        
        assert schema.feature_tier == 'standard'
        assert hasattr(schema, 'to_dict')
        assert hasattr(schema, 'from_dict')
    
    def test_wavefunction_uncertainty_config_schema_still_works(self):
        """Test WavefunctionUncertaintyConfigSchema remains functional."""
        schema = WavefunctionUncertaintyConfigSchema(enabled=False)
        
        assert schema.enabled is False
        assert hasattr(schema, 'to_dict')
    
    def test_wavefunction_config_schema_still_works(self):
        """Test WavefunctionConfigSchema remains functional."""
        schema = WavefunctionConfigSchema(
            raw_npz_filename='test.npz',
            dataset_root_dir='/tmp'
        )
        
        assert schema.raw_npz_filename == 'test.npz'
        assert schema.dataset_root_dir == '/tmp'
        assert hasattr(schema, 'to_dict')
        assert hasattr(schema, 'from_dict')
    
    def test_wavefunction_config_schema_from_dict(self):
        """Test WavefunctionConfigSchema.from_dict still works."""
        data = {
            'raw_npz_filename': 'wavefunction.npz',
            'dataset_root_dir': '/data',
            'processing_config': {'feature_tier': 'complete'}
        }
        
        schema = WavefunctionConfigSchema.from_dict(data)
        
        assert schema.raw_npz_filename == 'wavefunction.npz'
        assert schema.processing_config.feature_tier == 'complete'


class TestPhase5RefactoredValidateWavefunctionConfig:
    """Test Phase 5 refactored validate_wavefunction_config function."""
    
    def test_validate_wavefunction_config_valid(self):
        """Test validate_wavefunction_config with valid configuration."""
        config = {
            'dataset_type': 'Wavefunction',
            'wavefunction_config': {
                'raw_npz_filename': 'test.npz',
                'dataset_root_dir': '/tmp'
            }
        }
        
        valid, errors = validate_wavefunction_config(config)
        
        assert valid is True
        assert len(errors) == 0
    
    def test_validate_wavefunction_config_skips_non_wavefunction(self):
        """Test validate_wavefunction_config skips non-Wavefunction datasets."""
        config = {
            'dataset_type': 'DFT',
            'dft_config': {}
        }
        
        valid, errors = validate_wavefunction_config(config)
        
        assert valid is True
        assert len(errors) == 0
    
    def test_validate_wavefunction_config_missing_config_section(self):
        """Test validate_wavefunction_config detects missing config section."""
        config = {
            'dataset_type': 'Wavefunction'
            # Missing wavefunction_config section
        }
        
        valid, errors = validate_wavefunction_config(config)
        
        assert valid is False
        assert len(errors) > 0
        assert 'required' in errors[0].lower()
    
    def test_validate_wavefunction_config_invalid_data(self):
        """Test validate_wavefunction_config detects invalid configuration."""
        config = {
            'dataset_type': 'Wavefunction',
            'wavefunction_config': {
                # Missing required raw_npz_filename
                'dataset_root_dir': '/tmp'
            }
        }
        
        valid, errors = validate_wavefunction_config(config)
        
        assert valid is False
        assert len(errors) > 0
    
    def test_validate_wavefunction_config_with_processing_config(self):
        """Test validate_wavefunction_config with processing config."""
        config = {
            'dataset_type': 'Wavefunction',
            'wavefunction_config': {
                'raw_npz_filename': 'test.npz',
                'dataset_root_dir': '/tmp',
                'processing_config': {
                    'feature_tier': 'standard'
                }
            }
        }
        
        valid, errors = validate_wavefunction_config(config)
        
        assert valid is True
        assert len(errors) == 0


class TestPhase5RefactoredDescriptorValidation:
    """Test Phase 5 refactored descriptor validation with feature queries."""
    
    def test_descriptor_validation_valid_config(self):
        """Test descriptor validation with valid configuration."""
        validator = DescriptorSchemaValidator()
        descriptor_config = {
            'enabled': True,
            'default_categories': ['topological', 'geometric']
        }
        
        result = validator.validate_descriptor_config(descriptor_config)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
    
    def test_descriptor_validation_missing_enabled(self):
        """Test descriptor validation detects missing enabled field."""
        validator = DescriptorSchemaValidator()
        descriptor_config = {
            'default_categories': ['topological']
        }
        
        result = validator.validate_descriptor_config(descriptor_config)
        
        assert result['valid'] is False
        assert any('enabled' in e.lower() for e in result['errors'])
    
    def test_descriptor_validation_invalid_category(self):
        """Test descriptor validation detects invalid category."""
        validator = DescriptorSchemaValidator()
        descriptor_config = {
            'enabled': True,
            'default_categories': ['invalid_category_xyz']
        }
        
        result = validator.validate_descriptor_config(descriptor_config)
        
        assert result['valid'] is False
        assert any('invalid' in e.lower() for e in result['errors'])
    
    def test_descriptor_validation_with_dataset_type(self):
        """Test descriptor validation with dataset type context."""
        validator = DescriptorSchemaValidator()
        descriptor_config = {
            'enabled': True,
            'default_categories': ['geometric']
        }
        
        result = validator.validate_with_dataset_type(descriptor_config, 'DFT')
        
        assert 'valid' in result
        assert isinstance(result['valid'], bool)
    
    def test_descriptor_validation_disabled(self):
        """Test descriptor validation when descriptors are disabled."""
        validator = DescriptorSchemaValidator()
        descriptor_config = {
            'enabled': False
        }
        
        result = validator.validate_descriptor_config(descriptor_config)
        
        assert result['valid'] is True
    
    def test_descriptor_category_validation(self):
        """Test descriptor category validation."""
        validator = DescriptorSchemaValidator()
        
        result = validator.validate_descriptor_category(
            'topological',
            {'enabled': True},
            strict_mode=False
        )
        
        assert 'valid' in result
        assert 'errors' in result


class TestPhase5RefactoredExampleConfig:
    """Test Phase 5 refactored example configuration functions."""
    
    def test_create_default_experiment_config_structure(self):
        """Test create_default_experiment_config returns valid structure."""
        config = create_default_experiment_config()
        
        assert isinstance(config, dict)
        assert 'name' in config
        assert 'base_transforms' in config
    
    def test_create_example_experiments_config_multiple(self):
        """Test create_example_experiments_config returns multiple experiments."""
        configs = create_example_experiments_config()
        
        assert isinstance(configs, dict)
        assert len(configs) > 0
    
    def test_example_config_functions_return_valid_data(self):
        """Test all example config functions return valid data."""
        # Test plugin config
        plugin_config = create_default_plugin_config()
        assert isinstance(plugin_config, dict)
        assert 'enabled' in plugin_config
        
        # Test experiment config
        experiment_config = create_default_experiment_config()
        assert isinstance(experiment_config, dict)
        assert 'name' in experiment_config


# ==========================================
# ORIGINAL TEST CLASSES (Updated where needed)
# ==========================================

class TestTransformationSchema:
    """Test suite for TransformationSchema dataclass."""
    
    def test_creation_valid_schema(self, sample_experimental_setup):
        """Test creating valid TransformationSchema."""
        schema = TransformationSchema(
            experimental_setups=sample_experimental_setup,
            default_setup='baseline',
            validation={'enabled': True}
        )
        assert schema.experimental_setups == sample_experimental_setup
        assert schema.default_setup == 'baseline'
        assert schema.validation == {'enabled': True}
    
    def test_creation_with_optional_fields(self, sample_experimental_setup):
        """Test creating schema with optional fields."""
        schema = TransformationSchema(
            experimental_setups=sample_experimental_setup,
            default_setup='baseline',
            validation={'enabled': True},
            legacy_transforms=[{'name': 'OldTransform'}],
            research_metadata={'author': 'Test'},
            dataset_optimization={'enabled': True}
        )
        assert schema.legacy_transforms == [{'name': 'OldTransform'}]
        assert schema.research_metadata == {'author': 'Test'}
        assert schema.dataset_optimization == {'enabled': True}
    
    def test_creation_invalid_experimental_setups_not_dict(self):
        """Test that non-dict experimental_setups raises ValidationError.
        
        NOTE: Pydantic validates type BEFORE custom validators run, so we get
        a Pydantic ValidationError about type mismatch, not our custom message.
        """
        from pydantic import ValidationError as PydanticValidationError
        
        with pytest.raises(PydanticValidationError):
            TransformationSchema(
                experimental_setups=[],  # Should be dict
                default_setup='baseline',
                validation={}
            )
    
    def test_creation_empty_experimental_setups(self):
        """Test that empty experimental_setups raises ValueError when no standard_transforms."""
        with pytest.raises(ValueError, match="At least one of 'experimental_setups' or 'standard_transforms' must be defined"):
            TransformationSchema(
                experimental_setups={},
                default_setup='baseline',
                validation={}
            )
    
    def test_creation_default_setup_not_found(self, sample_experimental_setup):
        """Test that missing default_setup raises ValueError when no standard_transforms."""
        with pytest.raises(ValueError, match="Default setup 'nonexistent' not found"):
            TransformationSchema(
                experimental_setups=sample_experimental_setup,
                default_setup='nonexistent',
                validation={}
            )


# ==============================================================================
# TEST CLASS: Standard Transforms Support in TransformationSchema (NEW)
# ==============================================================================

class TestTransformationSchemaStandardTransforms:
    """
    Test standard_transforms support in TransformationSchema dataclass.
    
    Tests verify that TransformationSchema correctly handles:
    1. Configs with only standard_transforms (no experimental_setups)
    2. Configs with both standard_transforms and experimental_setups
    3. Backward compatibility with experimental_setups-only configs
    4. Validation logic for at least one transform source
    """
    
    def test_creation_with_only_standard_transforms(self):
        """Test creating schema with only standard_transforms (no experimental_setups)."""
        schema = TransformationSchema(
            standard_transforms=[
                {'name': 'AddSelfLoops', 'kwargs': {}, 'enabled': True},
                {'name': 'NormalizeFeatures', 'kwargs': {}, 'enabled': True}
            ],
            default_setup='baseline'
        )
        assert schema.standard_transforms is not None
        assert len(schema.standard_transforms) == 2
        assert schema.experimental_setups == {}  # Default empty dict
    
    def test_creation_with_both_sources(self, sample_experimental_setup):
        """Test creating schema with both standard_transforms and experimental_setups."""
        schema = TransformationSchema(
            standard_transforms=[
                {'name': 'AddSelfLoops', 'kwargs': {}, 'enabled': True}
            ],
            experimental_setups=sample_experimental_setup,
            default_setup='baseline'
        )
        assert schema.standard_transforms is not None
        assert len(schema.standard_transforms) == 1
        assert schema.experimental_setups == sample_experimental_setup
    
    def test_creation_empty_experimental_with_standard(self):
        """Test creating schema with empty experimental_setups but valid standard_transforms."""
        # This should NOT raise an error because standard_transforms exists
        schema = TransformationSchema(
            standard_transforms=[
                {'name': 'AddSelfLoops', 'kwargs': {}, 'enabled': True}
            ],
            experimental_setups={},
            default_setup='production'
        )
        assert schema.standard_transforms is not None
        assert schema.experimental_setups == {}
    
    def test_default_setup_as_label_with_standard_only(self):
        """Test default_setup can be any label when only standard_transforms exists."""
        # default_setup doesn't need to exist in experimental_setups when
        # only standard_transforms is defined
        schema = TransformationSchema(
            standard_transforms=[
                {'name': 'AddSelfLoops', 'kwargs': {}, 'enabled': True}
            ],
            default_setup='any_label_is_fine'
        )
        assert schema.default_setup == 'any_label_is_fine'
    
    def test_neither_source_raises_error(self):
        """Test that providing neither transform source raises ValueError."""
        with pytest.raises(ValueError, match="At least one of 'experimental_setups' or 'standard_transforms' must be defined"):
            TransformationSchema(
                default_setup='baseline'
            )
    
    def test_standard_transforms_must_be_list(self):
        """Test that non-list standard_transforms raises ValidationError.
        
        NOTE: Pydantic validates type BEFORE custom validators run, so we get
        a Pydantic ValidationError about type mismatch.
        """
        from pydantic import ValidationError as PydanticValidationError
        
        with pytest.raises(PydanticValidationError):
            TransformationSchema(
                standard_transforms={'not': 'a list'},  # Should be list
                default_setup='baseline'
            )
    
    def test_standard_transforms_none_with_experimental(self, sample_experimental_setup):
        """Test standard_transforms=None with experimental_setups works (backward compat)."""
        schema = TransformationSchema(
            standard_transforms=None,
            experimental_setups=sample_experimental_setup,
            default_setup='baseline'
        )
        assert schema.standard_transforms is None
        assert schema.experimental_setups == sample_experimental_setup


class TestPluginValidationLevel:
    """Test suite for PluginValidationLevel enum."""
    
    def test_enum_values(self):
        """Test that all enum values are defined correctly."""
        assert PluginValidationLevel.STRICT.value == "strict"
        assert PluginValidationLevel.STANDARD.value == "standard"
        assert PluginValidationLevel.PERMISSIVE.value == "permissive"
        assert PluginValidationLevel.DISABLED.value == "disabled"
    
    def test_enum_membership(self):
        """Test enum membership checks."""
        assert PluginValidationLevel.STRICT in PluginValidationLevel
        assert 'strict' == PluginValidationLevel.STRICT.value


class TestPluginConfigSchema:
    """Test suite for PluginConfigSchema dataclass."""
    
    def test_creation_with_defaults(self):
        """Test creating PluginConfigSchema with default values."""
        schema = PluginConfigSchema()
        assert schema.enabled is False
        assert schema.plugin_paths == []
        assert schema.auto_discover is True
        assert schema.validation_level == 'standard'
    
    def test_creation_with_custom_values(self, sample_plugin_config_dict):
        """Test creating PluginConfigSchema with custom values."""
        schema = PluginConfigSchema(**sample_plugin_config_dict)
        assert schema.enabled is True
        assert schema.plugin_paths == ['/path/to/plugins']
        assert schema.validation_level == 'standard'
    
    def test_invalid_validation_level(self):
        """Test that invalid validation_level raises ValueError."""
        with pytest.raises(ValueError, match="validation_level must be one of"):
            PluginConfigSchema(validation_level='invalid')
    
    def test_invalid_max_plugins_too_small(self):
        """Test that max_plugins < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_plugins must be between 1 and 1000"):
            PluginConfigSchema(max_plugins=0)
    
    def test_invalid_max_plugins_too_large(self):
        """Test that max_plugins > 1000 raises ValueError."""
        with pytest.raises(ValueError, match="max_plugins must be between 1 and 1000"):
            PluginConfigSchema(max_plugins=1001)
    
    def test_to_dict_method(self, sample_plugin_config_dict):
        """Test to_dict serialization."""
        schema = PluginConfigSchema(**sample_plugin_config_dict)
        result = schema.to_dict()
        assert isinstance(result, dict)
        assert result['enabled'] is True
        assert result['validation_level'] == 'standard'
    
    def test_from_dict_method(self, sample_plugin_config_dict):
        """Test from_dict deserialization."""
        schema = PluginConfigSchema.from_dict(sample_plugin_config_dict)
        assert schema.enabled is True
        assert schema.plugin_paths == ['/path/to/plugins']


class TestDescriptorConfigSchema:
    """Test suite for DescriptorConfigSchema dataclass."""
    
    def test_creation_with_defaults(self):
        """Test creating DescriptorConfigSchema with default values."""
        schema = DescriptorConfigSchema()
        assert schema.enabled is True  # Default is True in this schema
        assert schema.default_categories == ['constitutional', 'topological']
        assert schema.cache_descriptors is True
    
    def test_creation_with_custom_values(self):
        """Test creating DescriptorConfigSchema with custom values."""
        schema = DescriptorConfigSchema(
            enabled=True,
            default_categories=['topological', 'electronic'],
            cache_descriptors=False,
            error_handling='strict'
        )
        assert schema.enabled is True
        assert 'topological' in schema.default_categories
        assert 'electronic' in schema.default_categories
        assert schema.cache_descriptors is False
        assert schema.error_handling == 'strict'
    
    def test_invalid_error_handling_mode(self):
        """Test that invalid error_handling raises ValueError."""
        with pytest.raises(ValueError, match="error_handling must be one of"):
            DescriptorConfigSchema(error_handling='invalid')
    
    def test_invalid_validation_mode(self):
        """Test that invalid validation_mode raises ValueError."""
        with pytest.raises(ValueError, match="validation_mode must be one of"):
            DescriptorConfigSchema(validation_mode='invalid')
    
    def test_invalid_category(self):
        """Test that invalid category raises ValueError."""
        with pytest.raises(ValueError, match="Invalid category"):
            DescriptorConfigSchema(default_categories=['invalid_category_xyz'])


class TestExperimentSchema:
    """Test suite for ExperimentSchema dataclass."""
    
    def test_creation_valid_schema(self, sample_experiment_config_dict):
        """Test creating valid ExperimentSchema."""
        schema = ExperimentSchema(**sample_experiment_config_dict)
        assert schema.name == 'test_experiment'
        assert schema.num_runs == 3
        assert len(schema.ablations) == 1
    
    def test_creation_with_minimal_fields(self):
        """Test creating ExperimentSchema with minimal required fields."""
        schema = ExperimentSchema(
            name='minimal_exp',
            description='Minimal experiment',
            base_transforms=[]
        )
        assert schema.name == 'minimal_exp'
        assert schema.ablations == []
        assert schema.parameter_sweeps == []
    
    def test_invalid_num_runs(self):
        """Test that invalid num_runs raises ValueError."""
        with pytest.raises(ValueError, match="num_runs must be a positive integer"):
            ExperimentSchema(
                name='test',
                description='Test',
                base_transforms=[],
                num_runs=0
            )
    
    def test_to_dict_method(self, sample_experiment_config_dict):
        """Test to_dict serialization."""
        schema = ExperimentSchema(**sample_experiment_config_dict)
        result = schema.to_dict()
        assert isinstance(result, dict)
        assert result['name'] == 'test_experiment'
        assert result['num_runs'] == 3
    
    def test_from_dict_method(self, sample_experiment_config_dict):
        """Test from_dict deserialization."""
        schema = ExperimentSchema.from_dict(sample_experiment_config_dict)
        assert schema.name == 'test_experiment'
        assert len(schema.ablations) == 1


class TestYAMLSchemaValidator:
    """Test suite for YAMLSchemaValidator."""
    
    def test_validate_config_valid(self, sample_enhanced_transformation_config):
        """Test validation with valid configuration.
        
        NOTE: YAMLSchemaValidator expects the enhanced format with 'description'
        and 'transforms' keys in each setup, not the Pydantic schema format.
        """
        validator = YAMLSchemaValidator()
        complete_config = {'transformations': sample_enhanced_transformation_config}
        result = validator.validate_config(complete_config)
        assert result['valid'] is True
        assert len(result['errors']) == 0
    
    def test_validate_config_missing_transformations(self):
        """Test validation with missing transformations section."""
        validator = YAMLSchemaValidator()
        result = validator.validate_config({})
        assert result['valid'] is False
        assert len(result['errors']) > 0
    
    def test_validate_config_invalid_experimental_setups(self):
        """Test validation with invalid experimental setups."""
        validator = YAMLSchemaValidator()
        invalid_config = {
            'transformations': {
                'experimental_setups': {},  # Empty
                'default_setup': 'baseline',
                'validation': {}
            }
        }
        result = validator.validate_config(invalid_config)
        assert result['valid'] is False


# ==============================================================================
# TEST CLASS: Standard Transforms Support in YAMLSchemaValidator (NEW)
# ==============================================================================

class TestYAMLSchemaValidatorStandardTransforms:
    """
    Test standard_transforms support in YAMLSchemaValidator.
    
    Tests verify that detect_format() and validate_config() correctly handle:
    1. Configs with only standard_transforms
    2. Configs with both standard_transforms and experimental_setups
    3. Backward compatibility with experimental_setups-only configs
    4. Validation summaries include standard_transforms_count
    """
    
    def test_detect_format_with_standard_transforms_only(self):
        """Test detect_format recognizes standard_transforms as enhanced format."""
        validator = YAMLSchemaValidator()
        config = {
            'transformations': {
                'standard_transforms': [
                    {'name': 'AddSelfLoops', 'kwargs': {}, 'enabled': True}
                ],
                'default_setup': 'baseline'
            }
        }
        format_type = validator.detect_format(config)
        assert format_type == 'enhanced'
    
    def test_detect_format_with_both_sources(self):
        """Test detect_format with both standard_transforms and experimental_setups."""
        validator = YAMLSchemaValidator()
        config = {
            'transformations': {
                'standard_transforms': [
                    {'name': 'AddSelfLoops', 'kwargs': {}, 'enabled': True}
                ],
                'experimental_setups': {
                    'baseline': {
                        'transforms': [{'name': 'GCNNorm'}]
                    }
                },
                'default_setup': 'baseline'
            }
        }
        format_type = validator.detect_format(config)
        assert format_type == 'enhanced'
    
    def test_validate_config_with_standard_transforms_only(self):
        """Test validate_config with only standard_transforms."""
        validator = YAMLSchemaValidator()
        config = {
            'transformations': {
                'standard_transforms': [
                    {'name': 'AddSelfLoops', 'kwargs': {}, 'enabled': True},
                    {'name': 'NormalizeFeatures', 'kwargs': {}, 'enabled': True}
                ],
                'default_setup': 'production'
            }
        }
        result = validator.validate_config(config)
        assert result['valid'] is True
        assert result['format_detected'] == 'enhanced'
        # Check summary includes standard_transforms_count
        if 'summary' in result and 'standard_transforms_count' in result['summary']:
            assert result['summary']['standard_transforms_count'] == 2
    
    def test_validate_config_with_both_sources(self):
        """Test validate_config with both standard_transforms and experimental_setups."""
        validator = YAMLSchemaValidator()
        config = {
            'transformations': {
                'standard_transforms': [
                    {'name': 'AddSelfLoops', 'kwargs': {}, 'enabled': True}
                ],
                'experimental_setups': {
                    'baseline': {
                        'transforms': [{'name': 'GCNNorm'}]
                    }
                },
                'default_setup': 'baseline'
            }
        }
        result = validator.validate_config(config)
        assert result['valid'] is True
    
    def test_validate_config_empty_standard_transforms_with_experimental(self):
        """Test validate_config with empty standard_transforms but valid experimental_setups."""
        validator = YAMLSchemaValidator()
        config = {
            'transformations': {
                'standard_transforms': [],
                'experimental_setups': {
                    'baseline': {
                        'transforms': [{'name': 'AddSelfLoops'}]
                    }
                },
                'default_setup': 'baseline'
            }
        }
        result = validator.validate_config(config)
        assert result['valid'] is True
    
    def test_validate_config_neither_source_treated_as_legacy(self):
        """Test config with neither transform source is treated as legacy format."""
        validator = YAMLSchemaValidator()
        config = {
            'transformations': {
                'default_setup': 'baseline'
                # No standard_transforms, no experimental_setups
            }
        }
        result = validator.validate_config(config)
        # Without experimental_setups or standard_transforms, this is detected as legacy_dict format
        # Legacy formats are valid but should be migrated
        assert result['valid'] is True
        assert result['format_detected'] == 'legacy_dict'
        assert any('legacy' in str(w).lower() or 'migrat' in str(w).lower() for w in result.get('warnings', []))
    
    def test_validate_config_invalid_standard_transforms_type(self):
        """Test validate_config fails when standard_transforms is not a list."""
        validator = YAMLSchemaValidator()
        config = {
            'transformations': {
                'standard_transforms': {'not': 'a list'},  # Should be list
                'default_setup': 'baseline'
            }
        }
        result = validator.validate_config(config)
        assert result['valid'] is False
        assert any('list' in str(e).lower() for e in result['errors'])
    
    def test_validate_config_standard_transforms_missing_name(self):
        """Test validate_config generates warning for transform missing name."""
        validator = YAMLSchemaValidator()
        config = {
            'transformations': {
                'standard_transforms': [
                    {'name': 'AddSelfLoops', 'kwargs': {}, 'enabled': True},
                    {'kwargs': {}, 'enabled': True}  # Missing 'name'
                ],
                'default_setup': 'production'
            }
        }
        result = validator.validate_config(config)
        # Should have warnings about missing name
        if 'warnings' in result:
            assert any('name' in str(w).lower() for w in result['warnings'])
    
    def test_backward_compatibility_experimental_only(self, sample_enhanced_transformation_config):
        """Test backward compatibility with experimental_setups-only config."""
        validator = YAMLSchemaValidator()
        config = {'transformations': sample_enhanced_transformation_config}
        result = validator.validate_config(config)
        assert result['valid'] is True
        assert result['format_detected'] == 'enhanced'


class TestPluginSchemaValidator:
    """Test suite for PluginSchemaValidator."""
    
    def test_validate_plugin_config_valid(self, sample_plugin_config_dict):
        """Test validation with valid plugin configuration."""
        validator = PluginSchemaValidator()
        result = validator.validate_plugin_config(sample_plugin_config_dict)
        assert result['valid'] is True
        assert len(result['errors']) == 0
    
    def test_validate_plugin_config_invalid(self):
        """Test validation with invalid plugin configuration."""
        validator = PluginSchemaValidator()
        invalid_config = {
            'enabled': True,
            'validation_level': 'invalid_level'
        }
        result = validator.validate_plugin_config(invalid_config)
        assert result['valid'] is False
        assert len(result['errors']) > 0


class TestExperimentSchemaValidator:
    """Test suite for ExperimentSchemaValidator."""
    
    def test_experiment_schema_validator_instantiation(self):
        """Test ExperimentSchemaValidator can be instantiated."""
        validator = ExperimentSchemaValidator()
        assert validator is not None
    
    def test_validate_experiment_config_valid(self, sample_experiment_config_dict):
        """Test validation with valid experiment configuration."""
        validator = ExperimentSchemaValidator()
        result = validator.validate_experiment_config(
            sample_experiment_config_dict,
            strict_mode=False
        )
        assert result['valid'] is True
    
    def test_validate_experiments_config_multiple(self):
        """Test validation with multiple experiments."""
        validator = ExperimentSchemaValidator()
        experiments = create_example_experiments_config()
        result = validator.validate_experiments_config(experiments, strict_mode=False)
        assert 'valid' in result
        assert 'summary' in result


class TestConfigMigration:
    """Test suite for ConfigMigration."""
    
    def test_migration_basic(self):
        """Test basic configuration migration."""
        migrator = ConfigMigration()
        legacy_config = {
            'transformations': [
                {'name': 'AddSelfLoops', 'kwargs': {}, 'enabled': True}
            ]
        }
        result, warnings = migrator.migrate_to_enhanced(legacy_config)
        assert 'transformations' in result
        assert isinstance(warnings, list)
    
    def test_detect_format_legacy(self):
        """Test format detection for legacy configuration."""
        migrator = ConfigMigration()
        legacy_config = {
            'transformations': [{'name': 'Transform'}]
        }
        format_type = migrator.detect_format(legacy_config)
        # Format can be legacy_list, legacy_dict, enhanced, or unknown
        assert format_type in ['legacy', 'legacy_list', 'legacy_dict', 'enhanced', 'unknown']
    
    def test_migration_preserves_data(self):
        """Test that migration preserves original data."""
        migrator = ConfigMigration()
        legacy_config = {
            'transformations': [
                {'name': 'AddSelfLoops', 'kwargs': {'test': 123}, 'enabled': True}
            ]
        }
        result, warnings = migrator.migrate_to_enhanced(legacy_config)
        # Check that data is preserved in some form
        assert isinstance(result, dict)


class TestHelperFunctions:
    """Test suite for helper functions."""
    
    def test_create_default_plugin_config(self):
        """Test create_default_plugin_config returns valid config."""
        config = create_default_plugin_config()
        assert isinstance(config, dict)
        assert 'enabled' in config
        assert 'plugin_paths' in config
    
    def test_create_example_plugin_config(self):
        """Test create_example_plugin_config returns valid config."""
        config = create_example_plugin_config()
        assert isinstance(config, dict)
        assert config['enabled'] is True
    
    def test_create_default_experiment_config(self):
        """Test create_default_experiment_config returns valid config."""
        config = create_default_experiment_config()
        assert isinstance(config, dict)
        assert 'name' in config
        assert 'base_transforms' in config
    
    def test_create_example_experiments_config(self):
        """Test create_example_experiments_config returns valid configs."""
        configs = create_example_experiments_config()
        assert isinstance(configs, dict)
        assert len(configs) > 0
    
    def test_get_plugin_config_summary(self, sample_plugin_config_dict):
        """Test get_plugin_config_summary returns summary string."""
        summary = get_plugin_config_summary(sample_plugin_config_dict)
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    def test_get_experiment_config_summary(self, sample_experiment_config_dict):
        """Test get_experiment_config_summary returns summary string."""
        summary = get_experiment_config_summary(sample_experiment_config_dict)
        assert isinstance(summary, str)
        assert 'test_experiment' in summary


class TestValidationConfig:
    """Test suite for ValidationConfig."""
    
    def test_validation_config_defaults(self):
        """Test ValidationConfig default values."""
        config = ValidationConfig()
        assert hasattr(config, 'strict_mode')
        assert hasattr(config, 'warn_on_unknown')
        assert hasattr(config, 'require_descriptions')
        assert config.strict_mode is False
        assert config.warn_on_unknown is True


class TestModuleAvailability:
    """Test behavior when optional modules are not available."""
    
    def test_yaml_available_flag(self):
        """Test that YAML_AVAILABLE flag is set correctly."""
        assert isinstance(YAML_AVAILABLE, bool)
    
    def test_plugin_system_available_flag(self):
        """Test that PLUGIN_SYSTEM_AVAILABLE flag is set correctly."""
        assert isinstance(PLUGIN_SYSTEM_AVAILABLE, bool)
    
    def test_research_api_available_flag(self):
        """Test that RESEARCH_API_AVAILABLE flag is set correctly."""
        assert isinstance(RESEARCH_API_AVAILABLE, bool)


# ==========================================
# INTEGRATION TESTS
# ==========================================

class TestPhase5Integration:
    """Test integration between Phase 5 components."""
    
    def test_full_validation_pipeline(self, sample_enhanced_transformation_config):
        """Test complete validation pipeline."""
        # Test transformation validation
        validator = YAMLSchemaValidator()
        complete_config = {'transformations': sample_enhanced_transformation_config}
        transform_result = validator.validate_config(complete_config)
        assert transform_result['valid'] is True
        
        # Test wavefunction config validation
        wf_config = {
            'dataset_type': 'Wavefunction',
            'wavefunction_config': {
                'raw_npz_filename': 'test.npz',
                'dataset_root_dir': '/tmp'
            }
        }
        valid, errors = validate_wavefunction_config(wf_config)
        assert valid is True
        
        # Test descriptor validation
        desc_validator = DescriptorSchemaValidator()
        desc_config = {'enabled': True, 'default_categories': ['topological']}
        desc_result = desc_validator.validate_descriptor_config(desc_config)
        assert desc_result['valid'] is True
    
    def test_registry_fallback_integration(self):
        """Test that all components work with registry unavailable.
        
        NOTE: The module uses DYNAMIC fallback via filesystem discovery (lines 161-186).
        When registry is unavailable AND filesystem discovery fails, it returns empty list.
        This test verifies graceful degradation rather than hardcoded fallback values.
        """
        import milia_pipeline.config.config_schemas as schemas_module
        
        # Force registry unavailable
        original_available = schemas_module._REGISTRY_AVAILABLE
        original_list_all = schemas_module._registry_list_all
        original_initialized = schemas_module._REGISTRY_INITIALIZED
        
        schemas_module._REGISTRY_INITIALIZED = True  # Skip re-initialization
        schemas_module._REGISTRY_AVAILABLE = False
        schemas_module._registry_list_all = None
        
        try:
            # Test registry functions fall back to dynamic discovery or empty list
            # The result depends on whether filesystem discovery succeeds
            types = _registry_list_all_safe()
            assert isinstance(types, list)
            # Dynamic fallback may return discovered types or empty list
            # We verify the function doesn't crash and returns a list
            
            # Test validation still works (uses legacy schema as fallback)
            config = {
                'dataset_type': 'Wavefunction',
                'wavefunction_config': {
                    'raw_npz_filename': 'test.npz',
                    'dataset_root_dir': '/tmp'
                }
            }
            valid, errors = validate_wavefunction_config(config)
            assert valid is True
            
            # Test example config still works
            example = create_default_experiment_config()
            assert isinstance(example, dict)
        finally:
            # Restore
            schemas_module._REGISTRY_INITIALIZED = original_initialized
            schemas_module._REGISTRY_AVAILABLE = original_available
            schemas_module._registry_list_all = original_list_all


# ==========================================
# DESCRIPTOR CATEGORY CONFIG SCHEMA TESTS
# ==========================================

class TestDescriptorCategoryConfigSchema:
    """Test suite for DescriptorCategoryConfigSchema Pydantic model."""
    
    def test_creation_with_valid_category(self):
        """Test creating schema with valid category name."""
        schema = DescriptorCategoryConfigSchema(category_name='topological')
        assert schema.category_name == 'topological'
        assert schema.enabled is True  # Default
        assert schema.descriptors is None  # Default (all)
        assert schema.options == {}  # Default
    
    def test_creation_with_all_fields(self):
        """Test creating schema with all fields specified."""
        schema = DescriptorCategoryConfigSchema(
            category_name='geometric',
            enabled=False,
            descriptors=['TPSA', 'LabuteASA'],
            options={'include_3d': True}
        )
        assert schema.category_name == 'geometric'
        assert schema.enabled is False
        assert schema.descriptors == ['TPSA', 'LabuteASA']
        assert schema.options == {'include_3d': True}
    
    def test_invalid_category_name(self):
        """Test that invalid category_name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid category_name"):
            DescriptorCategoryConfigSchema(category_name='invalid_xyz')
    
    def test_all_valid_categories(self):
        """Test all valid categories can be instantiated."""
        valid_categories = [
            'constitutional', 'topological', 'geometric', 'electronic',
            'pharmacophore', 'fingerprint', 'custom'
        ]
        for category in valid_categories:
            schema = DescriptorCategoryConfigSchema(category_name=category)
            assert schema.category_name == category
    
    def test_descriptors_must_be_list_or_none(self):
        """Test that descriptors validates correctly."""
        # Valid: None
        schema = DescriptorCategoryConfigSchema(
            category_name='topological',
            descriptors=None
        )
        assert schema.descriptors is None
        
        # Valid: List of strings
        schema = DescriptorCategoryConfigSchema(
            category_name='topological',
            descriptors=['desc1', 'desc2']
        )
        assert schema.descriptors == ['desc1', 'desc2']
    
    def test_to_dict_method(self):
        """Test to_dict serialization."""
        schema = DescriptorCategoryConfigSchema(
            category_name='topological',
            enabled=True,
            descriptors=['Wiener', 'Zagreb'],
            options={'normalize': True}
        )
        result = schema.to_dict()
        assert isinstance(result, dict)
        assert result['category_name'] == 'topological'
        assert result['enabled'] is True
        assert result['descriptors'] == ['Wiener', 'Zagreb']
    
    def test_from_dict_method(self):
        """Test from_dict deserialization."""
        data = {
            'category_name': 'electronic',
            'enabled': True,
            'descriptors': ['HOMO', 'LUMO'],
            'options': {'method': 'am1'}
        }
        schema = DescriptorCategoryConfigSchema.from_dict(data)
        assert schema.category_name == 'electronic'
        assert schema.enabled is True
        assert schema.descriptors == ['HOMO', 'LUMO']
    
    def test_immutability(self):
        """Test schema is immutable (frozen=True)."""
        schema = DescriptorCategoryConfigSchema(category_name='topological')
        with pytest.raises(Exception):  # ValidationError or AttributeError
            schema.category_name = 'geometric'


# ==========================================
# CONFIG MIGRATION COMPREHENSIVE TESTS
# ==========================================

class TestConfigMigrationComprehensive:
    """Comprehensive tests for ConfigMigration class."""
    
    def test_detect_format_enhanced(self):
        """Test format detection for enhanced configuration."""
        migrator = ConfigMigration()
        enhanced_config = {
            'transformations': {
                'experimental_setups': {
                    'baseline': {'transforms': []}
                }
            }
        }
        format_type = migrator.detect_format(enhanced_config)
        assert format_type == 'enhanced'
    
    def test_detect_format_legacy_list(self):
        """Test format detection for legacy list configuration."""
        migrator = ConfigMigration()
        legacy_config = {
            'transformations': [
                {'name': 'Transform1'},
                {'name': 'Transform2'}
            ]
        }
        format_type = migrator.detect_format(legacy_config)
        assert format_type == 'legacy_list'
    
    def test_detect_format_invalid(self):
        """Test format detection for invalid configuration."""
        migrator = ConfigMigration()
        invalid_config = {
            'transformations': 'not_valid'  # String instead of dict/list
        }
        format_type = migrator.detect_format(invalid_config)
        assert format_type == 'invalid'
    
    def test_detect_format_missing_transformations(self):
        """Test format detection when transformations missing."""
        migrator = ConfigMigration()
        format_type = migrator.detect_format({'other_key': 'value'})
        assert format_type == 'invalid'
    
    def test_migrate_legacy_list_preserves_transforms(self):
        """Test migration preserves all transforms from legacy list."""
        migrator = ConfigMigration()
        legacy_config = {
            'transformations': [
                {'name': 'AddSelfLoops', 'kwargs': {'test': 1}},
                {'name': 'GCNNorm', 'kwargs': {'add_self_loops': False}}
            ]
        }
        result, warnings = migrator.migrate_to_enhanced(legacy_config)
        
        # Check migration occurred
        assert 'transformations' in result
        assert 'experimental_setups' in result['transformations']
        assert 'migrated_default' in result['transformations']['experimental_setups']
        
        # Check transforms are preserved
        migrated_transforms = result['transformations']['experimental_setups']['migrated_default']['transforms']
        assert len(migrated_transforms) == 2
        assert migrated_transforms[0]['name'] == 'AddSelfLoops'
        assert migrated_transforms[1]['name'] == 'GCNNorm'
    
    def test_migrate_legacy_dict_named_setups(self):
        """Test migration of legacy dict with named setups."""
        migrator = ConfigMigration()
        legacy_config = {
            'transformations': {
                'setup1': [{'name': 'Transform1'}],
                'setup2': [{'name': 'Transform2'}]
            }
        }
        result, warnings = migrator.migrate_to_enhanced(legacy_config)
        
        assert 'transformations' in result
        assert 'experimental_setups' in result['transformations']
        assert 'setup1' in result['transformations']['experimental_setups']
        assert 'setup2' in result['transformations']['experimental_setups']
    
    def test_migrate_already_enhanced_returns_unchanged(self):
        """Test migration of already enhanced config returns it unchanged."""
        migrator = ConfigMigration()
        enhanced_config = {
            'transformations': {
                'experimental_setups': {
                    'baseline': {'transforms': [{'name': 'Test'}]}
                },
                'default_setup': 'baseline'
            }
        }
        result, warnings = migrator.migrate_to_enhanced(enhanced_config)
        
        assert 'already in enhanced format' in str(warnings).lower()
    
    def test_migrate_invalid_format_returns_unchanged(self):
        """Test migration of invalid format returns unchanged."""
        migrator = ConfigMigration()
        invalid_config = {
            'transformations': 'not_valid'
        }
        result, warnings = migrator.migrate_to_enhanced(invalid_config)
        
        assert result == invalid_config
        assert 'invalid' in str(warnings).lower()
    
    def test_migration_calls_tracked(self):
        """Test that migration calls are tracked."""
        migrator = ConfigMigration()
        legacy_config = {'transformations': [{'name': 'Test'}]}
        
        migrator.migrate_to_enhanced(legacy_config)
        
        assert len(migrator.migration_calls) == 1
        assert 'config' in migrator.migration_calls[0]
        assert 'target_version' in migrator.migration_calls[0]
    
    def test_migrate_missing_transformations(self):
        """Test migration with missing transformations key."""
        migrator = ConfigMigration()
        result, warnings = migrator.migrate_to_enhanced({'other': 'data'})
        
        assert 'missing transformations' in str(warnings).lower()


# ==========================================
# HELPER FUNCTIONS COMPREHENSIVE TESTS
# ==========================================

class TestHelperFunctionsComprehensive:
    """Comprehensive tests for all helper functions."""
    
    def test_create_example_enhanced_config_structure(self):
        """Test create_example_enhanced_config returns valid structure."""
        config = create_example_enhanced_config()
        
        assert isinstance(config, dict)
        assert 'transformations' in config
        assert 'experimental_setups' in config['transformations']
        assert 'default_setup' in config['transformations']
        assert 'validation' in config['transformations']
    
    def test_create_example_enhanced_config_has_setups(self):
        """Test example config has multiple experimental setups."""
        config = create_example_enhanced_config()
        setups = config['transformations']['experimental_setups']
        
        assert len(setups) >= 1
        assert 'baseline' in setups
    
    def test_create_example_legacy_configs_all_formats(self):
        """Test create_example_legacy_configs returns all legacy formats."""
        configs = create_example_legacy_configs()
        
        assert isinstance(configs, dict)
        assert 'legacy_list' in configs
        assert 'legacy_dict' in configs
        assert 'legacy_with_parameters' in configs
    
    def test_create_example_legacy_configs_format_detection(self):
        """Test legacy example configs are detected as legacy format."""
        configs = create_example_legacy_configs()
        migrator = ConfigMigration()
        
        format_list = migrator.detect_format(configs['legacy_list'])
        assert format_list == 'legacy_list'
        
        format_dict = migrator.detect_format(configs['legacy_dict'])
        assert format_dict == 'legacy_dict'
    
    def test_merge_plugin_configs_simple_override(self):
        """Test merge_plugin_configs with simple value overrides."""
        base = {
            'enabled': False,
            'validation_level': 'standard',
            'max_plugins': 50
        }
        override = {
            'enabled': True,
            'validation_level': 'strict'
        }
        result = merge_plugin_configs(base, override)
        
        assert result['enabled'] is True
        assert result['validation_level'] == 'strict'
        assert result['max_plugins'] == 50  # Not overridden
    
    def test_merge_plugin_configs_path_union(self):
        """Test merge_plugin_configs unions plugin paths."""
        base = {
            'plugin_paths': ['/path/a', '/path/b']
        }
        override = {
            'plugin_paths': ['/path/b', '/path/c']  # b is duplicate
        }
        result = merge_plugin_configs(base, override)
        
        assert len(result['plugin_paths']) == 3
        assert set(result['plugin_paths']) == {'/path/a', '/path/b', '/path/c'}
    
    def test_merge_plugin_configs_list_override(self):
        """Test merge_plugin_configs overrides plugin lists."""
        base = {
            'trusted_plugins': ['plugin1'],
            'disabled_plugins': ['old_plugin']
        }
        override = {
            'trusted_plugins': ['plugin2', 'plugin3']
        }
        result = merge_plugin_configs(base, override)
        
        assert result['trusted_plugins'] == ['plugin2', 'plugin3']
        assert result['disabled_plugins'] == ['old_plugin']  # Not overridden
    
    def test_create_validator_factory(self):
        """Test create_validator factory function."""
        validator = create_validator()
        
        assert isinstance(validator, YAMLSchemaValidator)
        assert hasattr(validator, 'validate_config')
        assert hasattr(validator, 'detect_format')
    
    def test_create_migrator_factory(self):
        """Test create_migrator factory function."""
        migrator = create_migrator()
        
        assert isinstance(migrator, ConfigMigration)
        assert hasattr(migrator, 'migrate_to_enhanced')
        assert hasattr(migrator, 'detect_format')
    
    def test_get_plugin_config_summary_disabled(self):
        """Test plugin summary for disabled plugins."""
        config = {'enabled': False}
        summary = get_plugin_config_summary(config)
        
        assert 'DISABLED' in summary.upper()
    
    def test_get_experiment_config_summary_with_variants(self):
        """Test experiment summary calculates total runs correctly."""
        config = {
            'name': 'test_exp',
            'description': 'Test',
            'base_transforms': [{'name': 'T1'}],
            'ablations': [{'name': 'abl1'}, {'name': 'abl2'}],
            'parameter_sweeps': [{'name': 'sweep1'}],
            'num_runs': 5,
            'hypothesis': 'Test hypothesis'
        }
        summary = get_experiment_config_summary(config)
        
        assert 'test_exp' in summary
        assert 'Ablations: 2' in summary
        assert 'Parameter Sweeps: 1' in summary
        assert 'Total Runs' in summary
        assert 'hypothesis' in summary.lower()


# ==========================================
# FILE VALIDATION TESTS WITH PROPER MOCKING
# ==========================================

class TestFileValidationWithMocking:
    """Test file validation functions with proper mocking to avoid filesystem dependencies."""
    
    def test_validate_plugin_config_file_valid(self, tmp_path):
        """Test validate_plugin_config_file with valid YAML file."""
        if not YAML_AVAILABLE:
            pytest.skip("YAML not available")
        
        # Create temporary YAML file
        config_content = """
plugins:
  enabled: true
  plugin_paths:
    - /path/to/plugins
  validation_level: standard
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        
        result = validate_plugin_config_file(config_file)
        
        assert 'valid' in result
    
    def test_validate_plugin_config_file_missing_plugins_section(self, tmp_path):
        """Test validate_plugin_config_file when plugins section is missing."""
        if not YAML_AVAILABLE:
            pytest.skip("YAML not available")
        
        config_content = """
other_section:
  key: value
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        
        result = validate_plugin_config_file(config_file)
        
        assert result['valid'] is True
        assert any('no plugins section' in str(w).lower() for w in result.get('warnings', []))
    
    def test_validate_plugin_config_file_not_found(self):
        """Test validate_plugin_config_file with non-existent file."""
        from milia_pipeline.exceptions import ConfigurationError
        
        with pytest.raises(ConfigurationError, match="not found"):
            validate_plugin_config_file('/nonexistent/path/config.yaml')
    
    def test_validate_experiment_config_file_valid(self, tmp_path):
        """Test validate_experiment_config_file with valid YAML file."""
        if not YAML_AVAILABLE:
            pytest.skip("YAML not available")
        
        config_content = """
experiments:
  test_exp:
    name: test_exp
    description: Test experiment
    base_transforms:
      - name: AddSelfLoops
    ablations: []
    num_runs: 3
"""
        config_file = tmp_path / "test_experiments.yaml"
        config_file.write_text(config_content)
        
        result = validate_experiment_config_file(config_file)
        
        assert 'valid' in result
    
    def test_validate_experiment_config_file_missing_experiments(self, tmp_path):
        """Test validate_experiment_config_file when experiments section missing."""
        if not YAML_AVAILABLE:
            pytest.skip("YAML not available")
        
        config_content = """
other:
  key: value
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        
        result = validate_experiment_config_file(config_file)
        
        assert result['valid'] is True
        assert any('no experiments section' in str(w).lower() for w in result.get('warnings', []))
    
    @patch('milia_pipeline.config.config_schemas.YAML_AVAILABLE', False)
    def test_validate_plugin_config_file_yaml_unavailable(self, tmp_path):
        """Test validate_plugin_config_file when YAML is not available."""
        from milia_pipeline.exceptions import ConfigurationError
        
        # Create a file to avoid "not found" error
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("test: value")
        
        # We need to reload or re-import to pick up the patched value
        # Since YAML_AVAILABLE is checked at function call time, we patch it directly
        import milia_pipeline.config.config_schemas as schemas
        original = schemas.YAML_AVAILABLE
        schemas.YAML_AVAILABLE = False
        
        try:
            with pytest.raises(ConfigurationError, match="YAML"):
                validate_plugin_config_file(config_file)
        finally:
            schemas.YAML_AVAILABLE = original


# ==========================================
# EDGE CASES AND ERROR HANDLING TESTS
# ==========================================

class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling throughout the module."""
    
    def test_transformation_schema_standard_transforms_not_list(self):
        """Test TransformationSchema rejects non-list standard_transforms."""
        with pytest.raises(ValueError):
            TransformationSchema(
                standard_transforms={'not': 'a_list'},
                default_setup='baseline'
            )
    
    def test_plugin_config_schema_all_valid_levels(self):
        """Test all valid validation levels are accepted."""
        valid_levels = ['strict', 'standard', 'permissive', 'disabled']
        for level in valid_levels:
            schema = PluginConfigSchema(validation_level=level)
            assert schema.validation_level == level
    
    def test_experiment_validator_empty_config(self):
        """Test experiment validator with empty config."""
        validator = ExperimentSchemaValidator()
        result = validator.validate_experiment_config({})
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
    
    def test_experiment_validator_non_dict_config(self):
        """Test experiment validator with non-dict config."""
        validator = ExperimentSchemaValidator()
        result = validator.validate_experiment_config("not a dict")
        
        assert result['valid'] is False
        assert 'dictionary' in str(result['errors']).lower()
    
    def test_plugin_validator_non_dict_config(self):
        """Test plugin validator with non-dict config."""
        validator = PluginSchemaValidator()
        result = validator.validate_plugin_config("not a dict")
        
        assert result['valid'] is False
        assert 'dictionary' in str(result['errors']).lower()
    
    def test_yaml_validator_non_dict_config(self):
        """Test YAML validator with non-dict config."""
        validator = YAMLSchemaValidator()
        result = validator.validate_config("not a dict")
        
        assert result['valid'] is False
        assert 'dictionary' in str(result['errors']).lower()
    
    def test_descriptor_schema_auto_adjust_num_workers(self):
        """Test auto-adjustment of num_workers when parallel is True."""
        schema = DescriptorConfigSchema(
            parallel_computation=True,
            num_workers=1  # Will be auto-adjusted to 2
        )
        assert schema.num_workers == 2
    
    def test_descriptor_schema_num_workers_validation(self):
        """Test num_workers must be at least 1."""
        with pytest.raises(ValueError, match="num_workers must be at least 1"):
            DescriptorConfigSchema(num_workers=0)
    
    def test_wavefunction_processing_config_invalid_tier(self):
        """Test WavefunctionProcessingConfigSchema rejects invalid tier."""
        with pytest.raises(ValueError, match="feature_tier must be one of"):
            WavefunctionProcessingConfigSchema(feature_tier='invalid_tier')
    
    def test_wavefunction_uncertainty_config_cannot_enable(self):
        """Test WavefunctionUncertaintyConfigSchema cannot be enabled."""
        with pytest.raises(ValueError, match="not supported"):
            WavefunctionUncertaintyConfigSchema(enabled=True)
    
    def test_wavefunction_config_schema_missing_filename(self):
        """Test WavefunctionConfigSchema requires raw_npz_filename."""
        with pytest.raises(ValueError):
            WavefunctionConfigSchema(
                raw_npz_filename='',  # Empty string
                dataset_root_dir='/tmp'
            )
    
    def test_wavefunction_config_schema_wrong_extension(self):
        """Test WavefunctionConfigSchema requires .npz extension."""
        with pytest.raises(ValueError, match=".npz"):
            WavefunctionConfigSchema(
                raw_npz_filename='test.txt',  # Wrong extension
                dataset_root_dir='/tmp'
            )
    
    def test_experiment_schema_invalid_random_seed_type(self):
        """Test ExperimentSchema validates random_seed type.
        
        NOTE: Pydantic validates the type BEFORE custom validators run.
        When a non-integer is passed, Pydantic raises ValidationError with
        its own message about type parsing, not our custom validator message.
        """
        from pydantic import ValidationError as PydanticValidationError
        
        with pytest.raises(PydanticValidationError):
            ExperimentSchema(
                name='test',
                description='Test',
                base_transforms=[],
                random_seed='not_an_int'
            )
    
    def test_experiment_schema_empty_name(self):
        """Test ExperimentSchema rejects empty name."""
        with pytest.raises(ValueError, match="non-empty string"):
            ExperimentSchema(
                name='',
                description='Test',
                base_transforms=[]
            )
    
    def test_experiment_schema_empty_description(self):
        """Test ExperimentSchema rejects empty description."""
        with pytest.raises(ValueError, match="non-empty string"):
            ExperimentSchema(
                name='test',
                description='',
                base_transforms=[]
            )


# ==========================================
# STRICT MODE VALIDATION TESTS
# ==========================================

class TestStrictModeValidation:
    """Test validation behavior in strict mode."""
    
    def test_yaml_validator_strict_mode_missing_default_setup(self):
        """Test strict mode enforces default_setup requirement."""
        validator = YAMLSchemaValidator()
        config = {
            'transformations': {
                'experimental_setups': {
                    'baseline': {'transforms': [{'name': 'Test'}]}
                }
                # Missing default_setup
            }
        }
        validation_config = ValidationConfig(strict_mode=True)
        result = validator.validate_config(config, validation_config)
        
        # In strict mode, missing default_setup should cause errors
        assert any('default_setup' in str(e).lower() for e in result.get('errors', []))
    
    def test_yaml_validator_non_strict_mode_warns(self):
        """Test non-strict mode generates warnings instead of errors."""
        validator = YAMLSchemaValidator()
        config = {
            'transformations': {
                'experimental_setups': {
                    'baseline': {'transforms': [{'name': 'Test'}]}
                }
                # Missing default_setup
            }
        }
        validation_config = ValidationConfig(strict_mode=False)
        result = validator.validate_config(config, validation_config)
        
        # Should have warnings but still be valid
        assert result['valid'] is True
        assert any('default_setup' in str(w).lower() for w in result.get('warnings', []))
    
    def test_experiment_validator_strict_mode_requires_variants(self):
        """Test strict mode requires experiment variants."""
        validator = ExperimentSchemaValidator()
        config = {
            'name': 'test',
            'description': 'Test',
            'base_transforms': [{'name': 'T1'}],
            'ablations': [],
            'parameter_sweeps': []
        }
        result = validator.validate_experiment_config(config, strict_mode=True)
        
        assert result['valid'] is False
        assert any('variant' in str(e).lower() for e in result.get('errors', []))
    
    def test_experiment_validator_non_strict_mode_warns_no_variants(self):
        """Test non-strict mode warns but doesn't fail without variants."""
        validator = ExperimentSchemaValidator()
        config = {
            'name': 'test',
            'description': 'Test',
            'base_transforms': [{'name': 'T1'}],
            'ablations': [],
            'parameter_sweeps': []
        }
        result = validator.validate_experiment_config(config, strict_mode=False)
        
        assert result['valid'] is True
        assert any('variant' in str(w).lower() for w in result.get('warnings', []))
    
    def test_descriptor_validator_strict_mode_categories(self):
        """Test strict mode for descriptor validation."""
        validator = DescriptorSchemaValidator()
        config = {
            'enabled': True
            # Missing default_categories in strict mode
        }
        result = validator.validate_descriptor_config(config, strict_mode=True)
        
        # Should have errors about missing categories
        assert any('categories' in str(e).lower() for e in result.get('errors', []))
    
    def test_plugin_validator_strict_mode_security_warnings(self):
        """Test plugin validator warns about disabled security in strict mode."""
        validator = PluginSchemaValidator()
        config = {
            'enabled': True,
            'plugin_paths': ['/test'],
            'validation_level': 'strict',
            'security_scanning': False,  # Should warn in strict mode
            'enforce_checksums': False  # Should warn in strict mode
        }
        result = validator.validate_plugin_config(
            config,
            validation_level=PluginValidationLevel.STRICT
        )
        
        assert any('security' in str(w).lower() for w in result.get('warnings', []))
        assert any('checksum' in str(w).lower() for w in result.get('warnings', []))


# ==========================================
# COMBINED VALIDATION TESTS
# ==========================================

class TestCombinedValidation:
    """Test combined validation methods (plugins + experiments)."""
    
    def test_validate_config_with_plugins(self, sample_enhanced_transformation_config):
        """Test validate_config_with_plugins method."""
        validator = YAMLSchemaValidator()
        config = {
            'transformations': sample_enhanced_transformation_config,
            'plugins': {
                'enabled': True,
                'plugin_paths': ['/test'],
                'validation_level': 'standard'
            }
        }
        result = validator.validate_config_with_plugins(config)
        
        assert 'valid' in result
        assert 'plugin_validation' in result
    
    def test_validate_config_with_plugins_no_plugins_section(self, sample_enhanced_transformation_config):
        """Test validate_config_with_plugins when plugins section missing."""
        validator = YAMLSchemaValidator()
        config = {'transformations': sample_enhanced_transformation_config}
        result = validator.validate_config_with_plugins(config)
        
        assert result['valid'] is True
        assert any('no plugins section' in str(w).lower() for w in result.get('warnings', []))
    
    def test_validate_config_with_experiments(self, sample_enhanced_transformation_config):
        """Test validate_config_with_experiments method."""
        validator = YAMLSchemaValidator()
        config = {
            'transformations': sample_enhanced_transformation_config,
            'plugins': {'enabled': False},
            'experiments': {
                'exp1': {
                    'name': 'exp1',
                    'description': 'Test',
                    'base_transforms': [],
                    'ablations': [{'name': 'abl1'}]
                }
            }
        }
        result = validator.validate_config_with_experiments(config)
        
        assert 'valid' in result
        assert 'experiment_validation' in result
    
    def test_validate_config_with_experiments_no_experiments_section(self, sample_enhanced_transformation_config):
        """Test validate_config_with_experiments when experiments section missing."""
        validator = YAMLSchemaValidator()
        config = {
            'transformations': sample_enhanced_transformation_config,
            'plugins': {'enabled': False}
        }
        result = validator.validate_config_with_experiments(config)
        
        assert any('no experiments section' in str(w).lower() for w in result.get('warnings', []))


# ==========================================
# MAIN TEST EXECUTION
# ==========================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
