#!/usr/bin/env python3
"""
Unit tests for descriptor configuration system.

Tests:
- Schema validation
- Container creation
- YAML loading
- Validation functions
- Factory functions
"""

# =============================================================================
# PROJECT ROOT SETUP - MUST BE FIRST
# =============================================================================
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# =============================================================================
# IMPORTS
# =============================================================================
import pytest
import yaml
import tempfile
from typing import Dict, Any

from milia_pipeline.config.config_schemas import (
    DescriptorConfigSchema,
    DescriptorCategoryConfigSchema,
    DescriptorSchemaValidator
)
from milia_pipeline.config.config_containers import (
    DescriptorConfig,
    DescriptorCategoryConfig,
    create_descriptor_config_from_yaml,
    create_default_descriptor_config,
    create_minimal_descriptor_config
)
from milia_pipeline.config.validators import (
    validate_descriptor_config,
    validate_descriptor_category_compatibility,
    validate_descriptor_cache_settings
)
from milia_pipeline.exceptions import ConfigurationError, ValidationError


# =============================================================================
# SCHEMA TESTS
# =============================================================================

class TestDescriptorConfigSchema:
    """Tests for DescriptorConfigSchema."""
    
    def test_default_schema(self):
        """Test default schema creation."""
        schema = DescriptorConfigSchema()
        assert schema.enabled == True
        assert schema.default_categories == ['constitutional', 'topological']
        assert schema.error_handling == 'warn'
        assert schema.validation_mode == 'standard'
        assert schema.cache_descriptors == True
        assert schema.num_workers == 1
    
    def test_custom_schema(self):
        """Test custom schema creation."""
        schema = DescriptorConfigSchema(
            enabled=True,
            default_categories=['constitutional'],
            num_workers=4,
            parallel_computation=True,
            error_handling='strict'
        )
        assert schema.enabled == True
        assert len(schema.default_categories) == 1
        assert schema.num_workers == 4
        assert schema.parallel_computation == True
        assert schema.error_handling == 'strict'
    
    def test_invalid_error_handling(self):
        """Test invalid error handling mode."""
        with pytest.raises(ValueError, match="error_handling must be one of"):
            DescriptorConfigSchema(error_handling='invalid')
    
    def test_invalid_validation_mode(self):
        """Test invalid validation mode."""
        with pytest.raises(ValueError, match="validation_mode must be one of"):
            DescriptorConfigSchema(validation_mode='invalid')
    
    def test_invalid_num_workers(self):
        """Test invalid num_workers."""
        with pytest.raises(ValueError, match="num_workers must be at least 1"):
            DescriptorConfigSchema(num_workers=0)
    
    def test_invalid_category(self):
        """Test invalid category in default_categories."""
        with pytest.raises(ValueError, match="Invalid category"):
            DescriptorConfigSchema(default_categories=['invalid_category'])
    
    def test_parallel_auto_adjust(self):
        """Test automatic num_workers adjustment for parallel computation."""
        schema = DescriptorConfigSchema(
            parallel_computation=True,
            num_workers=1
        )
        assert schema.num_workers == 2  # Auto-adjusted
    
    def test_schema_serialization(self):
        """Test schema to_dict and from_dict."""
        original = DescriptorConfigSchema(
            enabled=True,
            default_categories=['constitutional', 'topological'],
            cache_descriptors=False
        )
        
        data = original.to_dict()
        restored = DescriptorConfigSchema.from_dict(data)
        
        assert restored.enabled == original.enabled
        assert restored.default_categories == original.default_categories
        assert restored.cache_descriptors == original.cache_descriptors


class TestDescriptorCategoryConfigSchema:
    """Tests for DescriptorCategoryConfigSchema."""
    
    def test_default_category_schema(self):
        """Test default category schema creation."""
        schema = DescriptorCategoryConfigSchema(
            category_name='constitutional'
        )
        assert schema.category_name == 'constitutional'
        assert schema.enabled == True
        assert schema.descriptors is None
        assert schema.options == {}
    
    def test_custom_category_schema(self):
        """Test custom category schema creation."""
        schema = DescriptorCategoryConfigSchema(
            category_name='topological',
            enabled=True,
            descriptors=['wiener_index', 'zagreb_index'],
            options={'max_path_length': 6}
        )
        assert schema.category_name == 'topological'
        assert len(schema.descriptors) == 2
        assert schema.options['max_path_length'] == 6
    
    def test_invalid_category_name(self):
        """Test invalid category name."""
        with pytest.raises(ValueError, match="Invalid category_name"):
            DescriptorCategoryConfigSchema(category_name='invalid')
    
    def test_invalid_descriptors_type(self):
        """Test invalid descriptors type."""
        # Pydantic V2 performs type validation before field_validator runs,
        # so it raises pydantic ValidationError with 'Input should be a valid list'
        from pydantic import ValidationError as PydanticValidationError
        with pytest.raises(PydanticValidationError, match="Input should be a valid list"):
            DescriptorCategoryConfigSchema(
                category_name='constitutional',
                descriptors="not_a_list"
            )
    
    def test_category_serialization(self):
        """Test category schema serialization."""
        original = DescriptorCategoryConfigSchema(
            category_name='geometric',
            enabled=False,
            descriptors=['radius_of_gyration'],
            options={'use_optimized_coords': True}
        )
        
        data = original.to_dict()
        restored = DescriptorCategoryConfigSchema.from_dict(data)
        
        assert restored.category_name == original.category_name
        assert restored.enabled == original.enabled
        assert restored.descriptors == original.descriptors
        assert restored.options == original.options


# =============================================================================
# CONTAINER TESTS
# =============================================================================

class TestDescriptorConfig:
    """Tests for DescriptorConfig container."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = DescriptorConfig()
        assert config.enabled == True
        assert config.error_handling == 'warn'
        assert config.should_use_cache() == True
        assert config.should_use_parallel() == False
    
    def test_is_category_enabled(self):
        """Test category enabled checking."""
        config = DescriptorConfig(
            default_categories=['constitutional', 'topological']
        )
        assert config.is_category_enabled('constitutional') == True
        assert config.is_category_enabled('geometric') == False
    
    def test_get_enabled_categories(self):
        """Test getting enabled categories."""
        config = DescriptorConfig(
            default_categories=['constitutional', 'topological'],
            categories={
                'geometric': {'enabled': True}
            }
        )
        enabled = config.get_enabled_categories()
        assert 'constitutional' in enabled
        assert 'topological' in enabled
        assert 'geometric' in enabled
    
    def test_get_category_descriptors(self):
        """Test getting category descriptors."""
        config = DescriptorConfig(
            categories={
                'constitutional': {
                    'descriptors': ['molecular_weight', 'num_atoms']
                }
            }
        )
        descriptors = config.get_category_descriptors('constitutional')
        assert descriptors == ['molecular_weight', 'num_atoms']
    
    def test_get_category_options(self):
        """Test getting category options."""
        config = DescriptorConfig(
            categories={
                'topological': {
                    'options': {'max_path_length': 8}
                }
            }
        )
        options = config.get_category_options('topological')
        assert options == {'max_path_length': 8}
    
    def test_to_dict_and_from_dict(self):
        """Test serialization."""
        original = DescriptorConfig(
            enabled=True,
            default_categories=['constitutional'],
            cache_descriptors=False
        )
        
        data = original.to_dict()
        restored = DescriptorConfig.from_dict(data)
        
        assert restored.enabled == original.enabled
        assert restored.default_categories == original.default_categories
        assert restored.cache_descriptors == original.cache_descriptors


class TestDescriptorCategoryConfig:
    """Tests for DescriptorCategoryConfig container."""
    
    def test_default_category_config(self):
        """Test default category config."""
        config = DescriptorCategoryConfig(category_name='constitutional')
        assert config.category_name == 'constitutional'
        assert config.enabled == True
        assert config.descriptors is None
        assert config.options == {}
    
    def test_custom_category_config(self):
        """Test custom category config."""
        config = DescriptorCategoryConfig(
            category_name='topological',
            enabled=True,
            descriptors=['wiener_index'],
            options={'max_path_length': 6}
        )
        assert config.category_name == 'topological'
        assert config.descriptors == ['wiener_index']
        assert config.options['max_path_length'] == 6
    
    def test_serialization(self):
        """Test category config serialization."""
        original = DescriptorCategoryConfig(
            category_name='geometric',
            enabled=False,
            descriptors=['radius_of_gyration']
        )
        
        data = original.to_dict()
        restored = DescriptorCategoryConfig.from_dict(data)
        
        assert restored.category_name == original.category_name
        assert restored.enabled == original.enabled
        assert restored.descriptors == original.descriptors


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_default_descriptor_config(self):
        """Test default config creation."""
        config = create_default_descriptor_config()
        assert config.enabled == True
        assert len(config.default_categories) > 0
        assert config.error_handling == 'warn'
    
    def test_create_minimal_descriptor_config(self):
        """Test minimal config creation."""
        config = create_minimal_descriptor_config()
        assert config.enabled == True
        assert len(config.default_categories) == 1
        assert config.default_categories[0] == 'constitutional'
    
    def test_create_from_yaml_valid(self):
        """Test config creation from valid YAML."""
        yaml_content = """
molecular_descriptors:
  enabled: true
  default_categories:
    - constitutional
  cache_descriptors: true
  num_workers: 1
"""
        yaml_data = yaml.safe_load(yaml_content)
        config = create_descriptor_config_from_yaml(yaml_data)
        assert config.enabled == True
        assert 'constitutional' in config.default_categories
    
    def test_create_from_yaml_missing_descriptors_key(self):
        """Test config creation from YAML missing molecular_descriptors key returns defaults."""
        yaml_content = """
enabled: true
default_categories:
  - constitutional
"""
        yaml_data = yaml.safe_load(yaml_content)
        # Should return default config when 'molecular_descriptors' key is missing
        config = create_descriptor_config_from_yaml(yaml_data)
        assert config.enabled == True
        assert len(config.default_categories) > 0
    
    def test_create_from_yaml_with_categories(self):
        """Test config creation with category configurations."""
        yaml_content = """
molecular_descriptors:
  enabled: true
  default_categories:
    - constitutional
  categories:
    constitutional:
      enabled: true
      descriptors:
        - molecular_weight
      options:
        include_hydrogens: true
"""
        yaml_data = yaml.safe_load(yaml_content)
        config = create_descriptor_config_from_yaml(yaml_data)
        assert config.is_category_enabled('constitutional')
        options = config.get_category_options('constitutional')
        assert options.get('include_hydrogens') == True


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestValidationFunctions:
    """Tests for validation functions."""
    
    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = DescriptorConfig(
            enabled=True,
            default_categories=['constitutional']
        )
        is_valid, errors = validate_descriptor_config(config)
        assert is_valid == True
        assert len(errors) == 0
    
    def test_validate_disabled_config(self):
        """Test validation of disabled configuration."""
        config = DescriptorConfig(enabled=False)
        is_valid, errors = validate_descriptor_config(config)
        assert is_valid == True
        assert len(errors) == 0
    
    def test_validate_empty_categories(self):
        """Test validation with empty categories - should pass as it's valid."""
        config = DescriptorConfig(
            enabled=True,
            default_categories=[]
        )
        is_valid, errors = validate_descriptor_config(config)
        # Empty categories is actually valid - the validator doesn't enforce non-empty
        assert is_valid == True
        assert len(errors) == 0
    
    def test_validate_invalid_cache_path(self):
        """Test validation with invalid cache path."""
        is_valid, errors = validate_descriptor_cache_settings(
            cache_descriptors=True,
            cache_path="/nonexistent/path/to/cache"
        )
        assert is_valid == False
        assert len(errors) > 0
    
    def test_validate_category_compatibility(self):
        """Test category compatibility validation."""
        is_compatible, errors, warnings = validate_descriptor_category_compatibility(
            category='electronic',
            dataset_type='Wavefunction'
        )
        assert is_compatible == True
        assert len(errors) == 0
    
    def test_validate_invalid_category(self):
        """Test validation of invalid category."""
        is_compatible, errors, warnings = validate_descriptor_category_compatibility(
            category='invalid_category',
            dataset_type='DFT'
        )
        assert is_compatible == False
        assert len(errors) > 0


# =============================================================================
# SCHEMA VALIDATOR TESTS
# =============================================================================

class TestDescriptorSchemaValidator:
    """Tests for DescriptorSchemaValidator."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = DescriptorSchemaValidator()
        assert validator is not None
        assert len(validator.validation_history) == 0
    
    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        validator = DescriptorSchemaValidator()
        config = {
            'enabled': True,
            'default_categories': ['constitutional'],
            'error_handling': 'warn'
        }
        
        result = validator.validate_descriptor_config(config)
        assert result['valid'] == True
        assert len(result['errors']) == 0
    
    def test_validate_missing_enabled(self):
        """Test validation with missing enabled field."""
        validator = DescriptorSchemaValidator()
        config = {
            'default_categories': ['constitutional']
        }
        
        result = validator.validate_descriptor_config(config)
        assert result['valid'] == False
        assert any('enabled' in error for error in result['errors'])
    
    def test_validate_invalid_category(self):
        """Test validation with invalid category."""
        validator = DescriptorSchemaValidator()
        config = {
            'enabled': True,
            'default_categories': ['invalid_category']
        }
        
        result = validator.validate_descriptor_config(config)
        assert result['valid'] == False
        assert len(result['errors']) > 0
    
    def test_validate_strict_mode(self):
        """Test validation in strict mode."""
        validator = DescriptorSchemaValidator()
        config = {
            'enabled': True,
            'default_categories': []
        }
        
        result = validator.validate_descriptor_config(config, strict_mode=True)
        assert result['valid'] == False
        assert any('default_categories' in error for error in result['errors'])
    
    def test_validate_category(self):
        """Test individual category validation."""
        validator = DescriptorSchemaValidator()
        category_config = {
            'enabled': True,
            'descriptors': ['molecular_weight', 'num_atoms']
        }
        
        result = validator.validate_descriptor_category(
            'constitutional',
            category_config
        )
        assert result['valid'] == True
        assert len(result['errors']) == 0
    
    def test_validate_invalid_category_name(self):
        """Test validation with invalid category name."""
        validator = DescriptorSchemaValidator()
        category_config = {
            'enabled': True
        }
        
        result = validator.validate_descriptor_category(
            'invalid_category',
            category_config
        )
        assert result['valid'] == False
        assert len(result['errors']) > 0
    
    def test_validate_with_dataset_type(self):
        """Test validation with dataset type compatibility."""
        validator = DescriptorSchemaValidator()
        config = {
            'enabled': True,
            'default_categories': ['electronic']
        }
        
        result = validator.validate_with_dataset_type(config, 'Wavefunction')
        assert result['valid'] == True
        assert len(result['errors']) == 0
    
    def test_validate_disabled_config_dataset_compat(self):
        """Test disabled config has no compatibility issues."""
        validator = DescriptorSchemaValidator()
        config = {
            'enabled': False,
            'default_categories': []
        }
        
        result = validator.validate_with_dataset_type(config, 'DFT')
        assert result['valid'] == True
        assert len(result['compatibility_notes']) > 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestDescriptorConfigIntegration:
    """Integration tests for descriptor configuration system."""
    
    def test_full_yaml_roundtrip(self):
        """Test full YAML configuration roundtrip."""
        yaml_content = """
molecular_descriptors:
  enabled: true
  default_categories:
    - constitutional
    - topological
  categories:
    constitutional:
      enabled: true
      descriptors: null
      options:
        include_hydrogens: true
    topological:
      enabled: true
      descriptors: null
      options:
        max_path_length: 6
  cache_descriptors: true
  cache_path: null
  parallel_computation: false
  num_workers: 1
  error_handling: warn
  validation_mode: standard
"""
        
        yaml_data = yaml.safe_load(yaml_content)
        config = create_descriptor_config_from_yaml(yaml_data)
        
        # Validate configuration
        is_valid, errors = validate_descriptor_config(config)
        assert is_valid == True
        assert len(errors) == 0
        
        # Check configuration properties
        assert config.enabled == True
        assert len(config.get_enabled_categories()) >= 2
        assert config.is_category_enabled('constitutional') == True
        assert config.is_category_enabled('topological') == True
    
    def test_schema_to_container_roundtrip(self):
        """Test schema to container conversion."""
        schema = DescriptorConfigSchema(
            enabled=True,
            default_categories=['constitutional'],
            cache_descriptors=False
        )
        
        # Convert schema to dict
        schema_dict = schema.to_dict()
        
        # Create container from dict
        config = DescriptorConfig.from_dict(schema_dict)
        
        # Validate container
        is_valid, errors = validate_descriptor_config(config)
        assert is_valid == True
        assert config.enabled == schema.enabled
        assert config.cache_descriptors == schema.cache_descriptors
    
    def test_config_in_minimal_bundle(self):
        """Test descriptor config in minimal test bundle."""
        from milia_pipeline.config.config_containers import create_minimal_config_for_testing
        
        bundle = create_minimal_config_for_testing('DFT')
        
        # Check that descriptor_config is in bundle
        assert 'descriptor_config' in bundle
        
        # Validate the descriptor config
        descriptor_config = bundle['descriptor_config']
        assert isinstance(descriptor_config, DescriptorConfig)
        assert descriptor_config.enabled == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
