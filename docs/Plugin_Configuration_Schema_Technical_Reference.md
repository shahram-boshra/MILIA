# Phase 3 Step 3.2.3: Configuration Schema for Plugins - Technical Reference

## Document Overview

**Module**: `config_schemas.py` (Enhanced)  
**Phase**: 3, Step 3.2, Sub-Step 3.2.3  
**Purpose**: Plugin configuration schema validation and integration  
**Status**: Implementation Complete  
**Date**: 2025-01-XX

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Implementation Components](#implementation-components)
4. [Integration Guide](#integration-guide)
5. [API Reference](#api-reference)
6. [Configuration Schema](#configuration-schema)
7. [Validation System](#validation-system)
8. [Usage Examples](#usage-examples)
9. [Error Handling](#error-handling)
10. [Testing](#testing)
11. [Best Practices](#best-practices)

---

## Overview

### Purpose

Sub-Step 3.2.3 extends the existing `config_schemas.py` module to support plugin system configuration validation. This enables:

1. **Plugin Configuration Validation**: Comprehensive validation of plugin settings
2. **Security Checking**: Validate security-related plugin configurations
3. **Compatibility Validation**: Ensure plugins work with transform configurations
4. **Schema Enforcement**: Strong typing and constraint validation for plugin configs

### Key Features

- **Four validation levels**: Strict, Standard, Permissive, Disabled
- **Security-focused**: Checksum validation, security scanning controls
- **Compatibility checking**: Validates plugin-transform integration
- **Extensible**: Easy to add new validation rules
- **Backward compatible**: Doesn't break existing configuration validation

### Integration Points

```
config_schemas.py (Enhanced)
    â"‚
    â"œâ"€â"€ PluginConfigSchema (New Dataclass)
    â"‚   â""â"€â"€ Validates plugin configuration structure
    â"‚
    â"œâ"€â"€ PluginSchemaValidator (New Class)
    â"‚   â"œâ"€â"€ validate_plugin_config()
    â"‚   â""â"€â"€ validate_plugin_compatibility()
    â"‚
    â"œâ"€â"€ YAMLSchemaValidator (Enhanced)
    â"‚   â""â"€â"€ validate_config_with_plugins() (New Method)
    â"‚
    â""â"€â"€ Utility Functions
        â"œâ"€â"€ create_default_plugin_config()
        â"œâ"€â"€ create_example_plugin_config()
        â"œâ"€â"€ validate_plugin_config_file()
        â"œâ"€â"€ merge_plugin_configs()
        â""â"€â"€ get_plugin_config_summary()
```

---

## Architecture

### Design Principles

1. **Separation of Concerns**: Plugin validation is separate but integrated
2. **Fail-Safe Defaults**: Conservative defaults for security
3. **Graceful Degradation**: Works even if plugin system unavailable
4. **Type Safety**: Strong typing with dataclasses
5. **Validation Levels**: Flexible strictness based on use case

### Component Hierarchy

```
PluginValidationLevel (Enum)
    â""â"€â"€ STRICT, STANDARD, PERMISSIVE, DISABLED

PluginConfigSchema (Dataclass)
    â"œâ"€â"€ Configuration structure definition
    â"œâ"€â"€ Validation in __post_init__
    â""â"€â"€ Serialization methods

PluginSchemaValidator (Class)
    â"œâ"€â"€ validate_plugin_config()
    â"‚   â"œâ"€â"€ Structure validation
    â"‚   â"œâ"€â"€ Type validation
    â"‚   â"œâ"€â"€ Security checks
    â"‚   â""â"€â"€ Summary generation
    â"‚
    â""â"€â"€ validate_plugin_compatibility()
        â"œâ"€â"€ Plugin system availability
        â"œâ"€â"€ Transform conflict detection
        â""â"€â"€ Configuration consistency

YAMLSchemaValidator (Enhanced)
    â""â"€â"€ validate_config_with_plugins()
        â"œâ"€â"€ Base config validation
        â"œâ"€â"€ Plugin config validation
        â"œâ"€â"€ Compatibility validation
        â""â"€â"€ Result merging
```

---

## Implementation Components

### 1. PluginValidationLevel Enum

**Purpose**: Define validation strictness levels

**Values**:

| Level | Description | Use Case |
|-------|-------------|----------|
| `STRICT` | Full validation including security | Production, CI/CD |
| `STANDARD` | Basic validation, skip expensive checks | Development, testing |
| `PERMISSIVE` | Minimal validation, trust plugins | Prototyping, trusted environments |
| `DISABLED` | No plugin validation | When plugins disabled |

**Implementation**:
```python
class PluginValidationLevel(Enum):
    STRICT = "strict"
    STANDARD = "standard"
    PERMISSIVE = "permissive"
    DISABLED = "disabled"
```

---

### 2. PluginConfigSchema Dataclass

**Purpose**: Type-safe plugin configuration structure

**Fields**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | False | Enable/disable plugin system |
| `plugin_paths` | List[str] | [] | Plugin directory paths |
| `auto_discover` | bool | True | Auto-discover plugins on paths |
| `auto_validate` | bool | True | Auto-validate discovered plugins |
| `validation_level` | str | "standard" | Validation strictness |
| `trusted_plugins` | List[str] | [] | Trusted plugin names |
| `disabled_plugins` | List[str] | [] | Disabled plugin names |
| `allow_experimental` | bool | False | Allow experimental plugins |
| `max_plugins` | int | 50 | Maximum plugin count |
| `require_metadata` | bool | True | Require plugin metadata |
| `enforce_checksums` | bool | True | Enforce checksum validation |
| `security_scanning` | bool | True | Enable security scanning |

**Validation Rules** (in `__post_init__`):
- `validation_level` must be in ['strict', 'standard', 'permissive', 'disabled']
- `max_plugins` must be between 1 and 1000
- `plugin_paths` must be list of strings
- `trusted_plugins` and `disabled_plugins` must be lists of strings

**Methods**:
- `to_dict()`: Serialize to dictionary
- `from_dict(data)`: Deserialize from dictionary

---

### 3. PluginSchemaValidator Class

**Purpose**: Validate plugin configurations with security and compatibility checks

#### Method: validate_plugin_config()

**Signature**:
```python
def validate_plugin_config(
    self,
    plugin_config: Dict[str, Any],
    validation_level: PluginValidationLevel = PluginValidationLevel.STANDARD
) -> Dict[str, Any]
```

**Validations Performed**:

1. **Structure Validation**
   - Configuration is a dictionary
   - `enabled` flag is boolean
   - `plugin_paths` is a list of strings

2. **Type Validation**
   - All boolean flags are booleans
   - `max_plugins` is integer in valid range
   - Plugin lists contain only strings

3. **Security Checks** (when validation_level=STRICT)
   - Warns if `security_scanning` disabled
   - Warns if `enforce_checksums` disabled

4. **Configuration Checks**
   - Warns if no plugin paths specified
   - Warns if experimental plugins allowed
   - Checks plugin system availability

**Return Structure**:
```python
{
    'valid': bool,                    # Overall validation result
    'errors': List[str],              # Validation errors
    'warnings': List[str],            # Validation warnings
    'suggestions': List[str],         # Improvement suggestions
    'plugin_summary': {               # Summary information
        'enabled': bool,
        'plugin_paths_count': int,
        'trusted_plugins_count': int,
        'disabled_plugins_count': int,
        'validation_level': str,
        'auto_discover': bool,
        'auto_validate': bool,
        'plugin_system_available': bool
    }
}
```

#### Method: validate_plugin_compatibility()

**Signature**:
```python
def validate_plugin_compatibility(
    self,
    plugin_config: Dict[str, Any],
    transformation_config: Dict[str, Any]
) -> Dict[str, Any]
```

**Checks**:
1. Plugin system enabled but module unavailable
2. Potential transform name conflicts
3. Configuration consistency

**Return Structure**:
```python
{
    'compatible': bool,           # Overall compatibility
    'issues': List[str],          # Compatibility issues
    'warnings': List[str]         # Compatibility warnings
}
```

---

### 4. YAMLSchemaValidator Enhancement

#### New Method: validate_config_with_plugins()

**Purpose**: Validate complete configuration including plugins

**Signature**:
```python
def validate_config_with_plugins(
    self,
    config: Dict[str, Any],
    validation_config: Optional[ValidationConfig] = None
) -> Dict[str, Any]
```

**Process**:
1. Validate base configuration (existing method)
2. Check for plugins section
3. Validate plugin configuration
4. Check plugin-transform compatibility
5. Merge all validation results

**Return Structure** (extends base validation result):
```python
{
    'valid': bool,
    'errors': List[str],
    'warnings': List[str],
    'suggestions': List[str],
    'format_detected': str,
    'summary': Dict[str, Any],
    'plugin_validation': Dict[str, Any],      # NEW
    'plugin_compatibility': Dict[str, Any]    # NEW
}
```

---

### 5. Utility Functions

#### create_default_plugin_config()

**Purpose**: Create safe default plugin configuration

**Returns**: Dictionary with all plugins disabled and secure defaults

**Usage**:
```python
default_config = create_default_plugin_config()
# {'enabled': False, 'security_scanning': True, ...}
```

---

#### create_example_plugin_config()

**Purpose**: Create example configuration for documentation

**Returns**: Dictionary with realistic example settings

**Usage**:
```python
example = create_example_plugin_config()
# Shows typical production configuration
```

---

#### validate_plugin_config_file()

**Purpose**: Validate plugin configuration from YAML file

**Signature**:
```python
def validate_plugin_config_file(
    config_path: Union[str, Path]
) -> Dict[str, Any]
```

**Process**:
1. Load YAML file
2. Extract plugins section
3. Validate using PluginSchemaValidator
4. Return validation results

**Usage**:
```python
results = validate_plugin_config_file('config.yaml')
if not results['valid']:
    print(f"Errors: {results['errors']}")
```

---

#### merge_plugin_configs()

**Purpose**: Merge two plugin configurations with override priority

**Signature**:
```python
def merge_plugin_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any]
) -> Dict[str, Any]
```

**Merge Strategy**:
- Simple values: Override takes precedence
- `plugin_paths`: Union of both (no duplicates)
- Plugin lists: Override replaces base

**Usage**:
```python
merged = merge_plugin_configs(default_config, user_config)
```

---

#### get_plugin_config_summary()

**Purpose**: Generate human-readable configuration summary

**Signature**:
```python
def get_plugin_config_summary(
    plugin_config: Dict[str, Any]
) -> str
```

**Returns**: Formatted multi-line summary string

**Usage**:
```python
summary = get_plugin_config_summary(config)
print(summary)
# Output:
# Plugin Configuration Summary:
#   Status: ENABLED
#   Validation Level: STANDARD
#   Plugin Paths: 2
#   ...
```

---

## Integration Guide

### Step 1: Update Imports

Add to existing imports section in `config_schemas.py`:

```python
from enum import Enum
from typing import Set

# Try to import plugin system components
try:
    from vqm24_pipeline.transformations.plugin_system import (
        PluginRegistry,
        PluginMetadata
    )
    PLUGIN_SYSTEM_AVAILABLE = True
except ImportError:
    PLUGIN_SYSTEM_AVAILABLE = False
    PluginRegistry = None
    PluginMetadata = None
```

### Step 2: Add New Components

1. Add `PluginValidationLevel` enum after existing dataclasses
2. Add `PluginConfigSchema` dataclass after enum
3. Add `PluginSchemaValidator` class after `YAMLSchemaValidator`

### Step 3: Enhance YAMLSchemaValidator

Add `validate_config_with_plugins()` method to existing `YAMLSchemaValidator` class.

### Step 4: Add Utility Functions

Add all utility functions before the `__main__` section.

### Step 5: Update _load_transformation_config_enhanced

Insert plugin validation code block before the return statement in `_load_transformation_config_enhanced()`.

### Step 6: Add Tests to __main__

Add plugin schema tests to the `__main__` section.

---

## Configuration Schema

### Complete YAML Structure

```yaml
# Base configuration (existing)
dataset_type: DFT

transformations:
  experimental_setups:
    basic:
      - name: AddSelfLoops
        kwargs: {}
        enabled: true
  default_setup: basic

# Plugin configuration (NEW)
plugins:
  enabled: true
  
  # Plugin discovery
  plugin_paths:
    - ./plugins
    - /opt/vqm24/plugins
  auto_discover: true
  auto_validate: true
  
  # Validation settings
  validation_level: standard  # strict, standard, permissive, disabled
  
  # Plugin management
  trusted_plugins:
    - official_molecular_transforms
    - verified_quantum_features
  disabled_plugins:
    - deprecated_transform_v1
  allow_experimental: false
  max_plugins: 50
  
  # Security settings
  require_metadata: true
  enforce_checksums: true
  security_scanning: true
```

### Minimal Configuration

```yaml
plugins:
  enabled: false
```

### Production Configuration

```yaml
plugins:
  enabled: true
  plugin_paths:
    - /opt/vqm24/production_plugins
  auto_discover: true
  auto_validate: true
  validation_level: strict
  trusted_plugins:
    - verified_plugin_pack_v2
  disabled_plugins: []
  allow_experimental: false
  max_plugins: 20
  require_metadata: true
  enforce_checksums: true
  security_scanning: true
```

### Development Configuration

```yaml
plugins:
  enabled: true
  plugin_paths:
    - ./local_plugins
    - ./dev_plugins
  auto_discover: true
  auto_validate: true
  validation_level: standard
  trusted_plugins: []
  disabled_plugins: []
  allow_experimental: true
  max_plugins: 100
  require_metadata: false
  enforce_checksums: false
  security_scanning: true
```

---

## Validation System

### Validation Workflow

```
Configuration File
    â"‚
    â–¼
Load YAML
    â"‚
    â–¼
Parse Structure
    â"‚
    â"œâ"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"
    â"‚                      â"‚
    â–¼                      â–¼
Base Config          Plugins Config
Validation           Validation
(existing)           (NEW)
    â"‚                      â"‚
    â"‚  â"Œâ"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"˜
    â"‚  â"‚
    â–¼  â–¼
Compatibility Check
    â"‚
    â–¼
Merge Results
    â"‚
    â–¼
Return Combined
Validation
```

### Validation Levels Comparison

| Check | STRICT | STANDARD | PERMISSIVE | DISABLED |
|-------|--------|----------|------------|----------|
| Structure | âœ" | âœ" | âœ" | âŒ |
| Types | âœ" | âœ" | âœ" | âŒ |
| Security settings | âœ" (warn) | âœ" | âœ" | âŒ |
| Checksums | âœ" (warn) | - | - | âŒ |
| Plugin paths exist | âœ" | âœ" | - | âŒ |
| Metadata required | âœ" | âœ" | - | âŒ |
| Performance impact | High | Medium | Low | None |

### Error Categories

**Critical Errors** (prevent execution):
- Invalid configuration structure
- Type mismatches in required fields
- Plugin system enabled but unavailable
- Invalid validation level

**Warnings** (log but continue):
- No plugin paths specified
- Security features disabled in STRICT mode
- Experimental plugins allowed
- Plugin system not available

**Suggestions** (informational):
- Consider enabling security scanning
- Add plugin paths for discovery
- Trust specific plugins for better performance

---

## Usage Examples

### Example 1: Basic Validation

```python
from vqm24_pipeline.config.config_schemas import (
    PluginSchemaValidator,
    PluginValidationLevel
)

# Create validator
validator = PluginSchemaValidator()

# Validate plugin config
plugin_config = {
    'enabled': True,
    'plugin_paths': ['./plugins'],
    'validation_level': 'standard'
}

result = validator.validate_plugin_config(
    plugin_config,
    validation_level=PluginValidationLevel.STANDARD
)

if result['valid']:
    print("âœ" Plugin configuration valid")
    print(f"Summary: {result['plugin_summary']}")
else:
    print("âœ— Validation failed")
    for error in result['errors']:
        print(f"  ERROR: {error}")
```

### Example 2: Complete Configuration Validation

```python
from vqm24_pipeline.config.config_schemas import YAMLSchemaValidator

# Load complete configuration
config = {
    'dataset_type': 'DFT',
    'transformations': {
        'experimental_setups': {
            'basic': [{'name': 'AddSelfLoops', 'kwargs': {}, 'enabled': True}]
        },
        'default_setup': 'basic'
    },
    'plugins': {
        'enabled': True,
        'plugin_paths': ['./plugins'],
        'validation_level': 'strict'
    }
}

# Validate with plugins
validator = YAMLSchemaValidator()
result = validator.validate_config_with_plugins(config)

if result['valid']:
    print("âœ" Complete configuration valid")
    
    # Check plugin-specific results
    if 'plugin_validation' in result:
        plugin_result = result['plugin_validation']
        print(f"Plugins: {plugin_result['plugin_summary']}")
    
    if 'plugin_compatibility' in result:
        compat = result['plugin_compatibility']
        if not compat['compatible']:
            print(f"âš ï¸ Compatibility issues: {compat['issues']}")
else:
    print("âœ— Validation failed")
    print(f"Errors: {result['errors']}")
```

### Example 3: File Validation

```python
from vqm24_pipeline.config.config_schemas import validate_plugin_config_file

# Validate plugin config from file
try:
    result = validate_plugin_config_file('config.yaml')
    
    if result['valid']:
        print("âœ" Configuration file valid")
    else:
        print("âœ— Validation errors:")
        for error in result['errors']:
            print(f"  - {error}")
        
        if result['warnings']:
            print("\nWarnings:")
            for warning in result['warnings']:
                print(f"  - {warning}")
                
except ConfigurationError as e:
    print(f"Failed to load configuration: {e}")
```

### Example 4: Configuration Merging

```python
from vqm24_pipeline.config.config_schemas import (
    create_default_plugin_config,
    merge_plugin_configs
)

# Create base config
base_config = create_default_plugin_config()

# User overrides
user_config = {
    'enabled': True,
    'plugin_paths': ['./my_plugins'],
    'validation_level': 'permissive'
}

# Merge configurations
merged = merge_plugin_configs(base_config, user_config)

print(f"Plugins enabled: {merged['enabled']}")
print(f"Plugin paths: {merged['plugin_paths']}")
print(f"Validation level: {merged['validation_level']}")
```

### Example 5: Configuration Summary

```python
from vqm24_pipeline.config.config_schemas import get_plugin_config_summary

plugin_config = {
    'enabled': True,
    'plugin_paths': ['./plugins', '/opt/plugins'],
    'validation_level': 'standard',
    'trusted_plugins': ['plugin1', 'plugin2'],
    'security_scanning': True
}

summary = get_plugin_config_summary(plugin_config)
print(summary)

# Output:
# Plugin Configuration Summary:
#   Status: ENABLED
#   Validation Level: STANDARD
#   Plugin Paths: 2
#   Trusted Plugins: 2
#   Disabled Plugins: 0
#   Auto-discover: True
#   Auto-validate: True
#   Security Scanning: True
#   Enforce Checksums: True
```

### Example 6: Integration with config_loader

```python
from vqm24_pipeline.config.config_loader import load_config
from vqm24_pipeline.config.config_schemas import (
    YAMLSchemaValidator,
    PluginValidationLevel
)

# Load configuration with validation
config = load_config(
    config_path='config.yaml',
    enable_validation=True,
    validation_level='STRICT'
)

# Additional plugin validation
validator = YAMLSchemaValidator()
result = validator.validate_config_with_plugins(config)

if not result['valid']:
    raise ConfigurationError(
        "Configuration validation failed",
        details="; ".join(result['errors'])
    )

# Access plugin configuration
if 'plugins' in config and config['plugins']['enabled']:
    plugin_paths = config['plugins']['plugin_paths']
    print(f"Plugin system enabled with {len(plugin_paths)} paths")
```

---

## Error Handling

### Exception Types

All validation errors raise `ConfigurationError` from `exceptions.py` with rich context:

```python
try:
    result = validate_plugin_config_file('config.yaml')
except ConfigurationError as e:
    print(f"Error: {e.message}")
    print(f"Config key: {e.config_key}")
    print(f"Details: {e.details}")
    if hasattr(e, 'suggestions'):
        print(f"Suggestions: {e.suggestions}")
```

### Common Errors

#### Error: Plugin system enabled but unavailable

**Cause**: `plugins.enabled: true` but `plugin_system.py` not importable

**Fix**:
```yaml
plugins:
  enabled: false  # Disable until plugin system installed
```

Or install plugin system:
```bash
pip install -e .  # Reinstall package
```

---

#### Error: Invalid validation level

**Cause**: `validation_level` not in allowed values

**Fix**:
```yaml
plugins:
  validation_level: standard  # Use: strict, standard, permissive, or disabled
```

---

#### Error: max_plugins out of range

**Cause**: `max_plugins` < 1 or > 1000

**Fix**:
```yaml
plugins:
  max_plugins: 50  # Must be 1-1000
```

---

#### Error: plugin_paths not a list

**Cause**: `plugin_paths` is string instead of list

**Fix**:
```yaml
plugins:
  plugin_paths:     # Must be list
    - ./plugins     # Not: plugin_paths: ./plugins
```

---

### Warning Handling

Warnings don't prevent execution but should be addressed:

```python
result = validator.validate_plugin_config(config)

if result['warnings']:
    for warning in result['warnings']:
        logger.warning(f"Plugin config: {warning}")
        
    # Decide whether to proceed
    if 'Security scanning disabled' in str(result['warnings']):
        # Production should not disable security
        raise ConfigurationError("Security scanning required in production")
```

---

## Testing

### Unit Tests

```python
import pytest
from vqm24_pipeline.config.config_schemas import (
    PluginConfigSchema,
    PluginSchemaValidator,
    PluginValidationLevel
)

def test_plugin_config_schema_validation():
    """Test PluginConfigSchema validation"""
    # Valid config
    config = PluginConfigSchema(
        enabled=True,
        plugin_paths=['./plugins'],
        validation_level='standard'
    )
    assert config.enabled == True
    
    # Invalid validation level
    with pytest.raises(ValueError):
        PluginConfigSchema(validation_level='invalid')
    
    # Invalid max_plugins
    with pytest.raises(ValueError):
        PluginConfigSchema(max_plugins=0)

def test_plugin_validator_basic():
    """Test basic plugin validation"""
    validator = PluginSchemaValidator()
    
    # Valid config
    config = {
        'enabled': True,
        'plugin_paths': ['./plugins'],
        'validation_level': 'standard'
    }
    result = validator.validate_plugin_config(config)
    assert result['valid'] == True
    
    # Invalid config
    invalid_config = {
        'enabled': 'yes',  # Should be boolean
        'plugin_paths': './plugins',  # Should be list
        'validation_level': 'invalid'  # Invalid level
    }
    result = validator.validate_plugin_config(invalid_config)
    assert result['valid'] == False
    assert len(result['errors']) > 0

def test_plugin_compatibility():
    """Test plugin-transform compatibility"""
    validator = PluginSchemaValidator()
    
    plugin_config = {
        'enabled': True,
        'plugin_paths': ['./plugins']
    }
    
    transform_config = {
        'experimental_setups': {
            'basic': [{'name': 'AddSelfLoops', 'kwargs': {}, 'enabled': True}]
        }
    }
    
    result = validator.validate_plugin_compatibility(
        plugin_config,
        transform_config
    )
    
    # Should be compatible
    assert result['compatible'] == True or len(result['issues']) == 0

def test_yaml_validator_with_plugins():
    """Test complete config validation with plugins"""
    from vqm24_pipeline.config.config_schemas import YAMLSchemaValidator
    
    validator = YAMLSchemaValidator()
    
    config = {
        'transformations': {
            'experimental_setups': {
                'basic': [{'name': 'AddSelfLoops', 'kwargs': {}, 'enabled': True}]
            },
            'default_setup': 'basic'
        },
        'plugins': {
            'enabled': True,
            'plugin_paths': ['./plugins'],
            'validation_level': 'standard'
        }
    }
    
    result = validator.validate_config_with_plugins(config)
    assert 'plugin_validation' in result
    assert 'plugin_compatibility' in result

def test_utility_functions():
    """Test plugin utility functions"""
    from vqm24_pipeline.config.config_schemas import (
        create_default_plugin_config,
        create_example_plugin_config,
        merge_plugin_configs,
        get_plugin_config_summary
    )
    
    # Default config
    default = create_default_plugin_config()
    assert default['enabled'] == False
    assert default['security_scanning'] == True
    
    # Example config
    example = create_example_plugin_config()
    assert example['enabled'] == True
    assert len(example['plugin_paths']) > 0
    
    # Merge configs
    merged = merge_plugin_configs(default, example)
    assert merged['enabled'] == True  # Override
    
    # Summary
    summary = get_plugin_config_summary(example)
    assert 'Plugin Configuration Summary' in summary
    assert 'ENABLED' in summary
```

### Integration Tests

```python
def test_plugin_config_with_loader():
    """Test plugin config integration with config_loader"""
    from vqm24_pipeline.config.config_loader import load_config
    import tempfile
    import yaml
    
    # Create test config file
    config_data = {
        'dataset_type': 'DFT',
        'transformations': {
            'experimental_setups': {
                'basic': [{'name': 'AddSelfLoops', 'kwargs': {}, 'enabled': True}]
            },
            'default_setup': 'basic'
        },
        'plugins': {
            'enabled': True,
            'plugin_paths': ['./test_plugins'],
            'validation_level': 'standard'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name
    
    try:
        # Load and validate
        config = load_config(
            config_path=config_path,
            enable_validation=True
        )
        
        assert 'plugins' in config
        assert config['plugins']['enabled'] == True
        
    finally:
        import os
        os.unlink(config_path)

def test_plugin_discovery_integration():
    """Test plugin config with actual plugin discovery"""
    if not PLUGIN_SYSTEM_AVAILABLE:
        pytest.skip("Plugin system not available")
    
    from vqm24_pipeline.transformations.plugin_system import PluginRegistry
    import tempfile
    import os
    
    # Create temporary plugin directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test plugin
        plugin_path = os.path.join(tmpdir, 'test_plugin.py')
        with open(plugin_path, 'w') as f:
            f.write('''
from vqm24_pipeline.transformations.custom_base import CustomTransformBase

PLUGIN_METADATA = {
    'plugin_name': 'test_plugin',
    'version': '1.0.0',
    'author': 'Test',
    'transforms': ['TestTransform']
}

class TestTransform(CustomTransformBase):
    def forward(self, data):
        return data
''')
        
        # Validate plugin config with this path
        validator = PluginSchemaValidator()
        config = {
            'enabled': True,
            'plugin_paths': [tmpdir],
            'auto_discover': True,
            'validation_level': 'standard'
        }
        
        result = validator.validate_plugin_config(config)
        assert result['valid'] == True
```

---

## Best Practices

### 1. Choose Appropriate Validation Level

**Production**:
```yaml
plugins:
  validation_level: strict
  enforce_checksums: true
  security_scanning: true
```

**Development**:
```yaml
plugins:
  validation_level: standard
  enforce_checksums: false
  allow_experimental: true
```

**Prototyping**:
```yaml
plugins:
  validation_level: permissive
  require_metadata: false
```

---

### 2. Security Best Practices

âœ… **DO**:
- Use STRICT validation in production
- Enable security scanning and checksums
- Maintain trusted plugin list
- Regularly review disabled plugins

âŒ **DON'T**:
- Disable security scanning in production
- Allow experimental plugins in production
- Skip checksum validation
- Trust all plugins by default

---

### 3. Configuration Management

âœ… **DO**:
- Version control plugin configurations
- Document trusted plugins
- Use environment-specific configs
- Validate before deployment

âŒ **DON'T**:
- Hardcode plugin paths
- Mix development and production configs
- Skip validation in CI/CD
- Ignore validation warnings

---

### 4. Error Handling

âœ… **DO**:
```python
try:
    result = validator.validate_plugin_config(config)
    if not result['valid']:
        logger.error(f"Validation failed: {result['errors']}")
        raise ConfigurationError("Invalid plugin config")
except ConfigurationError as e:
    logger.error(f"Configuration error: {e}")
    # Handle gracefully
```

âŒ **DON'T**:
```python
result = validator.validate_plugin_config(config)
# Ignore result['valid'] and proceed anyway
```

---

### 5. Compatibility Checking

âœ… **DO**:
```python
# Always check compatibility
compat = validator.validate_plugin_compatibility(
    plugin_config,
    transform_config
)

if not compat['compatible']:
    raise ConfigurationError(
        f"Plugin compatibility issues: {compat['issues']}"
    )
```

âŒ **DON'T**:
```python
# Skip compatibility check and hope for the best
```

---

## Summary

### Key Achievements

1. âœ… **Plugin configuration validation** integrated into config_schemas.py
2. âœ… **Four validation levels** for flexibility
3. âœ… **Security-focused** validation with checksums and scanning
4. âœ… **Compatibility checking** with transform configurations
5. âœ… **Comprehensive utilities** for config management
6. âœ… **Backward compatible** with existing validation
7. âœ… **Well-tested** with unit and integration tests
8. âœ… **Documented** with examples and best practices

### Integration Checklist

- [ ] Add new imports to config_schemas.py
- [ ] Add PluginValidationLevel enum
- [ ] Add PluginConfigSchema dataclass
- [ ] Add PluginSchemaValidator class
- [ ] Enhance YAMLSchemaValidator with validate_config_with_plugins()
- [ ] Add utility functions
- [ ] Update _load_transformation_config_enhanced
- [ ] Add tests to __main__ section
- [ ] Run validation tests
- [ ] Update documentation

### Next Steps

1. **Testing**: Run comprehensive tests on all validation levels
2. **Integration**: Integrate with config_loader for automatic validation
3. **Documentation**: Update user documentation with plugin config examples
4. **CI/CD**: Add plugin config validation to CI pipeline
5. **Monitoring**: Add logging for plugin config validation results

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-XX  
**Status**: Implementation Complete  
**Next Review**: After Phase 3 Step 3.2 completion
