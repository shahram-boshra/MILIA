# KEY INFO: plugin_system.py

## Overview
**Phase**: 3, Step 3.2.1  
**Purpose**: Secure, extensible plugin system for custom VQM24 transforms  
**Type**: Infrastructure Component  
**Status**: Production-ready core implementation

## Core Purpose
Provides a comprehensive plugin discovery, validation, and management system that allows users to extend VQM24 with custom molecular and quantum transforms while maintaining security, compatibility, and quality standards.

---

## Architecture

### High-Level Design
```
Plugin Sources → Discovery → Validation → Registration → Runtime Use
     ↓              ↓            ↓             ↓            ↓
  YAML/Python   Scanner    Comprehensive   Registry   Transform
  __plugin__.py  System    Test Suite      (Thread    Pipeline
  Standalone                               Safe)
```

### Key Design Patterns
1. **Singleton Registry**: Thread-safe central plugin management
2. **Multi-source Discovery**: Flexible plugin packaging (YAML, modules, standalone)
3. **Comprehensive Validation**: 5-stage validation pipeline
4. **Lazy Loading**: Plugins discovered but loaded on-demand
5. **Security-First**: Checksum verification, dependency validation, trust system

---

## Key Components

### 1. PluginMetadata (Dataclass)
**Purpose**: Store and validate plugin package information

**Critical Fields**:
- **Identity**: `plugin_name`, `version`, `author`
- **Dependencies**: `vqm24_version`, `pyg_version`, `python_version`, `dependencies[]`
- **Contents**: `transforms[]` (list of provided transform names)
- **Validation**: `is_validated`, `validation_date`, `validation_results`
- **Security**: `checksum` (SHA256), `trusted` (bool)

**Key Methods**:
```python
_is_valid_version(version: str) -> bool        # Semantic versioning check
_validate_dependencies() -> None               # Dependency spec validation
to_dict() / from_dict() -> Dict/PluginMetadata # Serialization
```

**Validation Rules**:
- Semantic versioning required (X.Y.Z)
- Plugin name mandatory
- All dependencies must be valid package specs

---

### 2. PluginRegistry (Singleton)
**Purpose**: Central thread-safe registry for all plugin operations

**Core Responsibilities**:
1. Plugin path management
2. Multi-method discovery
3. Transform registration with TransformRegistry
4. Enable/disable plugin functionality
5. Plugin lifecycle management

**Key Methods**:

#### Discovery & Loading
```python
add_plugin_path(path: Path) -> None
  # Add directory to plugin search paths
  # Updates sys.path for imports

discover_plugins(paths: Optional[List[Path]], auto_validate: bool) -> List[PluginMetadata]
  # Scans for plugins using 3 methods:
  # 1. plugin.yaml files
  # 2. __plugin__.py modules  
  # 3. Standalone .py with PLUGIN_METADATA

_load_plugin_from_yaml(yaml_path: Path) -> Optional[PluginMetadata]
_load_plugin_from_module(module_path: Path) -> Optional[PluginMetadata]
_load_plugin_from_standalone(py_file: Path) -> Optional[PluginMetadata]
```

#### Validation
```python
validate_plugin(plugin_name: str) -> Dict[str, Any]
  # Comprehensive 5-stage validation:
  # 1. Dependency checking (VQM24, PyG, Python, packages)
  # 2. Security analysis (checksum, dangerous imports)
  # 3. Instantiation testing (all transforms can be created)
  # 4. Parameter validation (constraints properly defined)
  # 5. Data compatibility (works with VQM24 Data objects)

_check_dependencies(metadata) -> Dict
_check_security(metadata) -> Dict
_test_transform_instantiation(metadata) -> Dict
_test_parameter_validation(metadata) -> Dict
_test_data_compatibility(metadata) -> Dict
```

#### Management
```python
enable_plugin(plugin_name: str) -> None
disable_plugin(plugin_name: str) -> None
  # Also unregisters transforms from TransformRegistry

list_plugins(validated_only: bool, enabled_only: bool) -> List[str]
get_plugin_info(plugin_name: str) -> Optional[PluginMetadata]
```

#### Internal Utilities
```python
_is_custom_transform(obj: Any) -> bool
  # Checks if class inherits from CustomTransformBase
  # Excludes base classes themselves

_register_transform(transform_class: type, plugin_metadata: PluginMetadata) -> None
  # Registers with TransformRegistry from Phase 1
  # Updates plugin's transforms list

_calculate_directory_checksum(directory: Path) -> str
_calculate_file_checksum(file_path: Path) -> str
  # SHA256 checksums for security verification
```

**Internal State**:
```python
_plugins: Dict[str, PluginMetadata]      # All registered plugins
_plugin_paths: List[Path]                # Search directories
_enabled_plugins: Set[str]               # Currently active
_disabled_plugins: Set[str]              # Explicitly disabled
```

---

### 3. PluginValidator
**Purpose**: Comprehensive quality assurance for plugins

**Main Method**:
```python
validate_plugin_comprehensive(
    plugin_name: str,
    test_data_path: Optional[Path],
    run_performance_tests: bool
) -> Dict[str, Any]
```

**Validation Sections** (5 total):

#### 1. Code Quality (`_check_code_quality`)
- PEP 8 compliance (placeholder for flake8/pylint)
- Type hints coverage
- Docstring coverage
- Code complexity metrics
- **Weight**: 15% of overall score

#### 2. Documentation (`_check_documentation`)
- Plugin description present
- Homepage/repository URL provided
- README.md quality
- Example code availability
- **Weight**: 20% of overall score

#### 3. Functional Tests (`_run_functional_tests`)
- Uses PluginRegistry.validate_plugin()
- Integration with real VQM24 data
- Transform instantiation and execution
- **Weight**: 35% of overall score (highest)

#### 4. Performance Benchmarks (`_benchmark_performance`)
- Small molecule execution time (10 nodes)
- Large molecule execution time (100 nodes)
- Scalability analysis
- Target: <10ms for small, <100ms acceptable
- **Weight**: 15% of overall score

#### 5. Security Analysis (`_analyze_security`)
- Uses PluginRegistry._check_security()
- Checksum verification
- Dangerous imports detection
- Trust level assessment
- **Weight**: 15% of overall score

**Scoring System**:
```python
_calculate_score(sections: Dict) -> float
  # Weighted average of section scores (0.0 - 1.0)

_generate_recommendation(report: Dict) -> str
  # ≥0.95: APPROVED - Excellent
  # ≥0.85: APPROVED - Good quality
  # ≥0.70: CONDITIONAL - Needs improvements
  # ≥0.50: NOT APPROVED - Significant issues
  # <0.50: REJECTED - Major issues
```

---

## Plugin Discovery Methods

### Method 1: YAML-based Plugins
**Structure**:
```
my_plugin/
├── plugin.yaml          # Metadata file
├── transform1.py        # Transform implementations
├── transform2.py
└── utils.py            # Supporting code
```

**plugin.yaml Example**:
```yaml
plugin_name: "my_plugin"
version: "1.0.0"
author: "Developer Name"
email: "dev@example.com"
description: "Custom VQM24 transforms"
vqm24_version: ">=1.0.0"
dependencies:
  - "numpy>=1.20.0"
  - "scipy>=1.7.0"
transforms:
  - "MyCustomTransform"
```

### Method 2: Python Package Plugins
**Structure**:
```
my_plugin/
├── __init__.py
├── __plugin__.py        # Contains PLUGIN_METADATA dict
├── transforms.py
└── tests.py
```

**__plugin__.py Example**:
```python
PLUGIN_METADATA = {
    'plugin_name': 'my_plugin',
    'version': '1.0.0',
    'author': 'Developer Name',
    # ... other fields
}
```

### Method 3: Standalone Python Files
**Structure**:
```
my_transform.py          # Single file with PLUGIN_METADATA

# In file:
PLUGIN_METADATA = { ... }

class MyTransform(CustomTransformBase):
    ...
```

---

## Integration Points

### Phase 1 Dependencies (graph_transforms.py)
```python
from vqm24_pipeline.transformations.graph_transforms import (
    TransformRegistry,      # Register discovered transforms
    get_transform_info,     # Introspect transform parameters
    validate_comprehensive  # Additional validation
)
```

**Usage**: Plugin transforms registered with Phase 1's central registry

### Phase 3 Step 3.1 Dependencies (custom_transforms.py)
```python
from vqm24_pipeline.transformations.custom_transforms import (
    CustomTransformBase,      # Base class for all custom transforms
    MolecularTransformBase,   # Molecular-specific base
    QuantumTransformBase,     # Quantum-specific base
    TransformMetadata         # Transform metadata class
)
```

**Usage**: Plugin transforms must inherit from these bases

### Configuration System (01_Relevant_Files)
```python
from vqm24_pipeline.config.config_loader import load_config
```

**Usage**: Load VQM24 configuration for plugin paths and settings

### Exception System (02_Relevant_Files)
```python
from vqm24_pipeline.exceptions import (
    PluginError,              # Base plugin exception
    PluginValidationError,    # Validation failures
    PluginSecurityError,      # Security concerns
    PluginDependencyError,    # Missing dependencies
    TransformValidationError  # Transform-specific errors
)
```

**Fallback**: Module provides fallback exception classes if imports fail

---

## Usage Patterns

### Basic Plugin Discovery
```python
from vqm24_pipeline.transformations.plugin_system import PluginRegistry

# Add plugin directory
PluginRegistry.add_plugin_path(Path("/path/to/plugins"))

# Discover all plugins
plugins = PluginRegistry.discover_plugins(auto_validate=True)

# List discovered plugins
for plugin in plugins:
    print(f"{plugin.plugin_name} v{plugin.version}")
```

### Plugin Validation
```python
# Basic validation (5 stages)
results = PluginRegistry.validate_plugin("my_plugin")
if results['passed']:
    print("Plugin is valid!")
else:
    print("Validation issues:", results['tests'])

# Comprehensive validation (with scoring)
from vqm24_pipeline.transformations.plugin_system import PluginValidator

report = PluginValidator.validate_plugin_comprehensive(
    "my_plugin",
    test_data_path=Path("/path/to/test_data"),
    run_performance_tests=True
)

print(f"Overall Score: {report['overall_score']:.2f}")
print(f"Recommendation: {report['recommendation']}")
```

### Plugin Management
```python
# Enable plugin
PluginRegistry.enable_plugin("my_plugin")

# Use plugin transforms
from vqm24_pipeline.transformations.graph_transforms import TransformRegistry
transform = TransformRegistry.get("MyCustomTransform")
transformed_data = transform(data)

# Disable plugin (unregisters transforms)
PluginRegistry.disable_plugin("my_plugin")

# List enabled plugins only
enabled = PluginRegistry.list_plugins(enabled_only=True)
```

### Query Plugin Information
```python
# Get plugin metadata
metadata = PluginRegistry.get_plugin_info("my_plugin")
print(f"Transforms: {metadata.transforms}")
print(f"Dependencies: {metadata.dependencies}")
print(f"Validated: {metadata.is_validated}")
print(f"Checksum: {metadata.checksum}")
```

---

## Security Features

### 1. Checksum Verification
- SHA256 checksums calculated for all plugin files
- Stored in metadata for verification
- Detects tampering or corruption

### 2. Trust System
```python
metadata.trusted = True  # Manually mark as trusted
# Trusted plugins have relaxed security checks
```

### 3. Dependency Validation
- All dependencies must be importable
- Version constraints validated
- Missing dependencies flagged in validation

### 4. Code Analysis (Placeholder)
- Detection of dangerous imports (os, subprocess, eval, exec)
- AST parsing for security analysis (to be implemented)
- Sandboxing considerations (future enhancement)

### 5. Isolated Loading
- Plugins loaded in separate import contexts
- sys.path manipulation controlled
- No automatic code execution during discovery

---

## Validation Pipeline Details

### Stage 1: Dependencies (CRITICAL)
```python
_check_dependencies(metadata) -> Dict
```
- **Checks**: VQM24 version, PyG installation, Python version, package dependencies
- **Output**: `{'passed': bool, 'missing': list}`
- **Failure Impact**: Plugin cannot be enabled

### Stage 2: Security (CRITICAL)
```python
_check_security(metadata) -> Dict
```
- **Checks**: Checksum verification, dangerous patterns, trust status
- **Output**: `{'passed': bool, 'issues': list, 'warnings': list}`
- **Failure Impact**: Plugin flagged as unsafe

### Stage 3: Instantiation (CRITICAL)
```python
_test_transform_instantiation(metadata) -> Dict
```
- **Checks**: All transforms can be instantiated, transforms are callable
- **Output**: `{'passed': bool, 'failures': list}`
- **Failure Impact**: Plugin unusable

### Stage 4: Parameters (HIGH)
```python
_test_parameter_validation(metadata) -> Dict
```
- **Checks**: Parameter constraints properly defined, constraint format valid
- **Output**: `{'passed': bool, 'issues': list}`
- **Failure Impact**: Transforms may behave unpredictably

### Stage 5: Compatibility (HIGH)
```python
_test_data_compatibility(metadata) -> Dict
```
- **Checks**: Transforms work with VQM24 Data objects
- **Test Data**: Sample molecular graph (10 atoms, 20 edges)
- **Output**: `{'passed': bool, 'failures': list}`
- **Failure Impact**: Runtime failures in pipeline

---

## Thread Safety

### Singleton Pattern
```python
_instance = None
_lock = threading.Lock()

def __new__(cls):
    if cls._instance is None:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
    return cls._instance
```

**Guarantees**:
- Only one registry instance across all threads
- Thread-safe initialization
- No race conditions during discovery/registration

**Usage**: Safe for multi-threaded data loading pipelines

---

## Error Handling

### Exception Hierarchy
```
PluginError (base)
├── PluginValidationError (validation failures)
│   └── validation_errors: List
├── PluginSecurityError (security issues)
│   └── security_issues: List
└── PluginDependencyError (missing dependencies)
    └── missing_dependencies: List
```

### Error Context
All exceptions include:
- `plugin_name`: Affected plugin
- Specific error details (validation_errors, security_issues, etc.)
- Traceback chain for debugging

---

## Performance Considerations

### Lazy Loading
- Plugins discovered but not loaded until needed
- Transform classes imported on first use
- Minimal memory footprint for unused plugins

### Caching
- Plugin metadata cached in registry
- Transform classes cached in TransformRegistry
- Validation results cached in metadata

### Benchmark Targets
- Small molecule (<10 atoms): <10ms per transform
- Large molecule (100+ atoms): <100ms per transform
- Discovery: <1s for 100 plugins

---

## Testing Strategy

### Unit Tests Required
1. PluginMetadata validation
2. Checksum calculation accuracy
3. Version parsing and validation
4. Each discovery method (YAML, module, standalone)
5. Transform registration flow
6. Enable/disable functionality
7. Security checks
8. Dependency validation

### Integration Tests Required
1. Full discovery → validation → enable → use workflow
2. Multi-plugin scenarios
3. Plugin conflicts and resolution
4. Performance under load
5. Thread safety stress tests

### Security Tests Required
1. Malicious code detection
2. Checksum verification
3. Dependency injection prevention
4. Path traversal protection

---

## Configuration

### Expected Config Structure
```yaml
plugins:
  paths:
    - "/opt/vqm24/plugins"
    - "~/.vqm24/plugins"
  auto_discover: true
  auto_validate: false
  enabled_plugins:
    - "my_plugin"
    - "another_plugin"
  trusted_plugins:
    - "official_vqm24_plugin"
```

---

## Logging

### Log Levels Used
- **INFO**: Discovery, registration, enable/disable actions
- **WARNING**: Validation failures, non-critical security issues
- **ERROR**: Critical failures, security violations
- **DEBUG**: Discovery attempts, file scanning details

### Key Log Messages
```python
logger.info(f"Added plugin path: {path}")
logger.info(f"Discovered plugin from YAML: {plugin.plugin_name}")
logger.info(f"Registered transform '{transform_name}' from plugin '{plugin_name}'")
logger.info(f"Enabled plugin: {plugin_name}")
logger.warning(f"Auto-validation failed for {plugin_name}: {e}")
logger.error(f"Failed to register transform {transform_class}: {e}")
```

---

## Future Enhancements

### Phase 3 Future Steps
1. **Hot Reload**: Dynamic plugin reload without restart
2. **Plugin Marketplace**: Central repository for community plugins
3. **Sandboxing**: Isolated execution environments
4. **Code Signing**: Cryptographic verification of plugin authors
5. **Dependency Resolution**: Automatic installation of missing packages
6. **Plugin Conflicts**: Automatic detection and resolution
7. **Performance Profiling**: Built-in profiler for plugin transforms
8. **Documentation Generation**: Auto-generate docs from plugin metadata

### AST-based Security Analysis
```python
# Future implementation
def _analyze_ast_security(source_code: str) -> Dict:
    tree = ast.parse(source_code)
    issues = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            # Check for dangerous imports
        if isinstance(node, ast.Call):
            # Check for eval/exec calls
    
    return {'issues': issues}
```

---

## Dependencies

### Required (Phase 1-3)
- `torch_geometric`: Data structures and transforms
- `vqm24_pipeline.transformations.custom_transforms`: Base classes
- `vqm24_pipeline.transformations.graph_transforms`: Registry integration

### Optional
- `vqm24_pipeline.config.config_loader`: Configuration loading
- `vqm24_pipeline.exceptions`: Exception classes (has fallbacks)

### Standard Library
- `importlib`, `importlib.util`: Dynamic module loading
- `inspect`: Class introspection
- `threading`: Thread-safe singleton
- `pathlib`: Path manipulation
- `yaml`: YAML parsing
- `hashlib`: Checksum calculation
- `re`: Version validation
- `logging`: Event logging

---

## Module Exports

```python
__all__ = [
    'PluginMetadata',          # Plugin metadata dataclass
    'PluginRegistry',          # Central registry (singleton)
    'PluginValidator',         # Comprehensive validation
    'PluginError',             # Base exception
    'PluginValidationError',   # Validation failure exception
    'PluginSecurityError',     # Security issue exception
    'PluginDependencyError'    # Dependency error exception
]
```

---

## Critical Implementation Notes

### 1. Import Fallbacks
Module gracefully handles missing dependencies by providing fallback implementations for exception classes and checking for None before using imported components.

### 2. Semantic Versioning
Version strings must follow `X.Y.Z` or `X.Y.Z-suffix` format. Version comparisons use string parsing.

### 3. Transform Registration
Transforms are registered with Phase 1's TransformRegistry, including metadata that marks them as custom and tracks their source plugin.

### 4. Checksum Security
SHA256 checksums provide integrity verification but not authentication. Future enhancement: code signing.

### 5. Thread Safety
Only the PluginRegistry uses thread safety mechanisms. Individual PluginMetadata objects are not thread-safe but are immutable after creation.

---

## Summary

The plugin system provides a **secure, extensible, and quality-controlled** mechanism for extending VQM24 with custom transforms. It implements:

✅ **3 discovery methods** for maximum flexibility  
✅ **5-stage validation** for quality assurance  
✅ **Comprehensive scoring** (0.0-1.0) with recommendations  
✅ **Security features** (checksums, trust system, dependency validation)  
✅ **Thread-safe singleton** registry  
✅ **Seamless integration** with Phase 1-3 infrastructure  
✅ **Performance benchmarking** for production readiness  
✅ **Enable/disable** functionality for plugin management  

**Production Status**: Core implementation complete, ready for integration testing and security hardening.
