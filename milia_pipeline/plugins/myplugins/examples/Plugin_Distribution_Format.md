# Plugin Distribution Format
## VQM24 Plugin System - Distribution Specification

**Phase:** Plugin Distribution Format  
**Status:** Implementation Ready  
**Version:** 1.0.0  
**Last Updated:** October 15, 2025

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Plugin Package Structure](#plugin-package-structure)
3. [plugin.yaml Specification](#pluginyaml-specification)
4. [Transform Implementation Requirements](#transform-implementation-requirements)
5. [Testing Requirements](#testing-requirements)
6. [Documentation Requirements](#documentation-requirements)
7. [Distribution Formats](#distribution-formats)
8. [Validation & Security](#validation--security)
9. [Example Plugins](#example-plugins)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)

---

## Overview

### Purpose

The Plugin Distribution Format defines the **standard structure, metadata, and requirements** for distributing custom transforms as VQM24 plugins. This ensures:

✅ **Consistency** - All plugins follow the same structure  
✅ **Discoverability** - Plugins can be automatically found and loaded  
✅ **Validation** - Plugins can be verified before use  
✅ **Security** - Plugins can be checked for safety  
✅ **Documentation** - Users know what plugins do and how to use them  

### Key Components

| Component | Purpose | Required |
|-----------|---------|----------|
| `plugin.yaml` | Plugin metadata and requirements | ✅ Yes |
| `__init__.py` | Package initialization | ✅ Yes |
| `transforms/` | Transform implementations | ✅ Yes |
| `tests/` | Plugin test suite | ⚠️ Recommended |
| `README.md` | User documentation | ⚠️ Recommended |
| `LICENSE` | License information | ⚠️ Recommended |
| `CHANGELOG.md` | Version history | ❌ Optional |
| `requirements.txt` | Python dependencies | ❌ Optional |

---

## Plugin Package Structure

### Standard Layout

```
my_plugin/                           # Plugin root directory
├── plugin.yaml                      # ✅ REQUIRED: Plugin metadata
├── __init__.py                      # ✅ REQUIRED: Package initialization
├── transforms/                      # ✅ REQUIRED: Transform implementations
│   ├── __init__.py
│   ├── my_transform.py             # Your transform class
│   └── helper_module.py            # Optional helpers
├── tests/                          # ⚠️ RECOMMENDED: Test suite
│   ├── __init__.py
│   ├── test_my_transform.py
│   └── conftest.py                 # Pytest configuration
├── README.md                       # ⚠️ RECOMMENDED: Documentation
├── LICENSE                         # ⚠️ RECOMMENDED: License
├── CHANGELOG.md                    # ❌ OPTIONAL: Version history
├── requirements.txt                # ❌ OPTIONAL: Dependencies
├── examples/                       # ❌ OPTIONAL: Usage examples
│   ├── basic_usage.py
│   └── advanced_usage.py
└── docs/                          # ❌ OPTIONAL: Extended docs
    ├── api.md
    └── tutorial.md
```

### Naming Conventions

**Plugin Directory Name:**
- Use `snake_case`
- Descriptive and unique
- Examples: `quantum_augmentations`, `molecular_filters`, `advanced_normalization`

**Transform File Names:**
- Use `snake_case`
- Match or relate to transform class name
- Examples: `energy_normalizer.py`, `charge_scaler.py`

**Transform Class Names:**
- Use `PascalCase`
- Clear and descriptive
- Examples: `EnergyNormalizer`, `ChargeScaler`, `VibrationalModeFilter`

---

## plugin.yaml Specification

### Complete Schema

```yaml
# ============================================================================
# MANDATORY FIELDS - Must be present in all plugins
# ============================================================================

plugin_name: "my_plugin"              # Unique identifier (snake_case)
version: "1.0.0"                      # Semantic versioning (MAJOR.MINOR.PATCH)
author: "Your Name <your.email@example.com>"  # Author with contact
description: "Brief description of what this plugin does"

# VQM24 compatibility
vqm24_min_version: "1.0.0"           # Minimum VQM24 version required
vqm24_max_version: "2.0.0"           # Maximum VQM24 version (optional)

# Transform definitions
transforms:
  - name: "MyTransform"               # Transform name (used in config)
    class_name: "MyTransform"         # Python class name
    module_path: "transforms.my_transform"  # Import path (relative)
    category: "quantum"               # Category: molecular/quantum/experimental/augmentation
    description: "What this transform does"
    version: "1.0.0"                  # Transform version
    
    # Data requirements (from TransformMetadata)
    required_node_features: ["x", "pos"]
    required_edge_features: []
    required_graph_attributes: ["energy"]
    
    # Parameter constraints (from get_parameter_constraints)
    parameter_constraints:
      threshold:
        type: "float"
        range: [0.0, 1.0]
        default: 0.5
        description: "Filtering threshold"

# ============================================================================
# OPTIONAL FIELDS - Enhance functionality and documentation
# ============================================================================

# Licensing and links
license: "MIT"                        # SPDX license identifier
homepage: "https://github.com/user/my_plugin"
repository: "https://github.com/user/my_plugin"
documentation: "https://my-plugin.readthedocs.io"
bug_tracker: "https://github.com/user/my_plugin/issues"

# Dependencies (beyond VQM24 base requirements)
dependencies:
  numpy: ">=1.26.0"
  scipy: ">=1.15.0"
  custom_package: "~=2.0.0"          # Use standard pip version specifiers

# Python compatibility
python_requires: ">=3.10,<3.12"      # Python version range

# Plugin flags
experimental: false                   # Is this experimental?
deprecated: false                     # Is this deprecated?
deprecated_message: ""                # Reason for deprecation

# Validation datasets (where tested)
validated_datasets:
  - "VQM24_DFT"
  - "VQM24_DMC"

# Security
checksum: "sha256:abc123..."          # SHA-256 checksum for verification

# Metadata
tags:                                 # Keywords for discovery
  - "normalization"
  - "energy"
  - "quantum"

maintainers:                          # Additional maintainers
  - name: "Maintainer Name"
    email: "maintainer@example.com"

# Change tracking
created_date: "2025-10-15"
updated_date: "2025-10-15"
```

### Field Descriptions

#### Mandatory Fields

**plugin_name** (string)
- Unique identifier for the plugin
- Must be valid Python package name (snake_case)
- Used in configuration and CLI commands
- Example: `"quantum_augmentations"`

**version** (string)
- Semantic versioning: `MAJOR.MINOR.PATCH`
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes
- Example: `"1.2.3"`

**author** (string)
- Plugin author with contact information
- Format: `"Name <email@example.com>"`
- Example: `"Jane Doe <jane@example.com>"`

**description** (string)
- Brief, clear description of plugin purpose
- 1-2 sentences maximum
- Example: `"Advanced quantum property normalization transforms"`

**vqm24_min_version** (string)
- Minimum VQM24 version required
- Semantic versioning format
- Plugin won't load on older versions
- Example: `"1.0.0"`

**vqm24_max_version** (string, optional)
- Maximum VQM24 version supported
- Use when API changes might break plugin
- Example: `"2.0.0"`

**transforms** (list)
- List of transform definitions
- At least one transform required
- See [Transform Definition Schema](#transform-definition-schema)

#### Transform Definition Schema

```yaml
- name: "TransformName"               # ✅ REQUIRED: Registry name
  class_name: "TransformClassName"    # ✅ REQUIRED: Python class
  module_path: "relative.module.path" # ✅ REQUIRED: Import path
  category: "quantum"                 # ✅ REQUIRED: Category
  description: "Transform description" # ✅ REQUIRED: What it does
  version: "1.0.0"                    # ✅ REQUIRED: Transform version
  
  # Data requirements (match TransformMetadata)
  required_node_features: []          # ❌ OPTIONAL: Node features needed
  required_edge_features: []          # ❌ OPTIONAL: Edge features needed
  required_graph_attributes: []       # ❌ OPTIONAL: Graph attributes needed
  
  # Parameter schema (match get_parameter_constraints)
  parameter_constraints: {}           # ❌ OPTIONAL: Parameter validation
  
  # Additional metadata
  paper_reference: ""                 # ❌ OPTIONAL: Citation/DOI
  github_url: ""                      # ❌ OPTIONAL: Source code link
  experimental: false                 # ❌ OPTIONAL: Experimental flag
```

#### Optional Fields

**license** (string)
- SPDX license identifier
- Examples: `"MIT"`, `"Apache-2.0"`, `"GPL-3.0"`
- See: https://spdx.org/licenses/

**homepage** (string)
- Plugin homepage or documentation URL
- Example: `"https://my-plugin.readthedocs.io"`

**repository** (string)
- Source code repository URL
- Example: `"https://github.com/user/my_plugin"`

**documentation** (string)
- Full documentation URL
- Example: `"https://my-plugin.readthedocs.io/en/latest/"`

**bug_tracker** (string)
- Issue tracker URL
- Example: `"https://github.com/user/my_plugin/issues"`

**dependencies** (dict)
- Additional Python packages required
- Format: `package_name: "version_spec"`
- Use standard pip version specifiers
- Examples:
  - `"numpy": ">=1.26.0"` (minimum version)
  - `"scipy": "~=1.15.0"` (compatible release)
  - `"rdkit": ">=2024.03.0,<2026.0.0"` (range)

**python_requires** (string)
- Python version compatibility
- Standard pip format
- Example: `">=3.10,<3.12"`

**experimental** (boolean)
- Mark as experimental/unstable
- Default: `false`
- Blocked when `allow_experimental: false` in config

**deprecated** (boolean)
- Mark as deprecated
- Default: `false`
- Warning shown when loaded

**deprecated_message** (string)
- Deprecation explanation
- Example: `"Use AdvancedNormalizer instead"`

**validated_datasets** (list)
- Datasets where plugin was tested
- Examples: `["VQM24_DFT", "VQM24_DMC"]`

**checksum** (string)
- SHA-256 checksum of plugin files
- Format: `"sha256:hexdigest"`
- Verified when `enforce_checksums: true`

**tags** (list)
- Keywords for discovery/search
- Examples: `["normalization", "energy", "quantum"]`

**maintainers** (list)
- Additional maintainers
- Format:
  ```yaml
  - name: "Name"
    email: "email@example.com"
  ```

**created_date** (string)
- Plugin creation date
- Format: `"YYYY-MM-DD"`

**updated_date** (string)
- Last update date
- Format: `"YYYY-MM-DD"`

### Validation Rules

**Semantic Versioning:**
```python
# Valid versions
"1.0.0"         # Release
"1.2.3"         # Standard release
"2.0.0-alpha"   # Pre-release
"1.5.0-beta.1"  # Pre-release with number

# Invalid versions
"1.0"           # Missing PATCH
"v1.0.0"        # Extra prefix
"latest"        # Not a version number
```

**Category Values:**
```python
VALID_CATEGORIES = {
    "molecular",      # Molecular structure transforms
    "quantum",        # Quantum property transforms
    "experimental",   # Experimental/research transforms
    "augmentation"    # Data augmentation transforms
}
```

**Dependency Specifications:**
```python
# Valid pip version specifiers
">=1.0.0"           # Minimum version
"~=1.2.0"           # Compatible release (~= 1.2.0 means >=1.2.0, <1.3.0)
">=1.0.0,<2.0.0"   # Range
"==1.5.0"           # Exact version (not recommended)
"!=1.3.0"           # Exclude specific version
```

---

## Transform Implementation Requirements

### Required Base Class

All plugin transforms **MUST** inherit from one of:

```python
from vqm24_pipeline.transformations.custom_transforms import (
    CustomTransformBase,        # Generic transforms
    MolecularTransformBase,     # Molecular structure transforms
    QuantumTransformBase        # Quantum property transforms
)
```

### Required Methods

```python
class MyTransform(QuantumTransformBase):
    """Transform docstring."""
    
    def __init__(self, param1: float = 1.0):
        """
        Initialize transform.
        
        Args:
            param1: Parameter description
        """
        super().__init__()  # ✅ REQUIRED: Call parent __init__
        self.param1 = param1
    
    def transform(self, data: Data) -> Data:
        """
        ✅ REQUIRED: Core transformation logic.
        
        Args:
            data: Input molecular graph
        
        Returns:
            Transformed molecular graph (or None to filter)
        """
        # Your implementation
        return data
    
    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        """
        ✅ REQUIRED: Transform metadata.
        
        Returns:
            TransformMetadata with complete information
        """
        return TransformMetadata(
            name="MyTransform",
            version="1.0.0",
            author="Your Name",
            category="quantum",
            description="What this transform does",
            required_graph_attributes=["energy"]
        )
    
    @classmethod
    def get_parameter_constraints(cls) -> Dict[str, Any]:
        """
        ⚠️ RECOMMENDED: Parameter validation constraints.
        
        Returns:
            Dictionary of parameter constraints
        """
        return {
            'param1': {
                'type': float,
                'range': (0.0, 10.0),
                'default': 1.0,
                'description': 'Parameter 1 description'
            }
        }
```

### Transform Implementation Checklist

- [ ] Inherits from appropriate base class
- [ ] Calls `super().__init__()` in `__init__`
- [ ] Implements `transform()` method
- [ ] Implements `get_metadata()` classmethod
- [ ] Implements `get_parameter_constraints()` (recommended)
- [ ] Uses type hints for all parameters
- [ ] Provides default values for parameters
- [ ] Includes comprehensive docstrings
- [ ] Handles missing attributes gracefully
- [ ] Returns `None` for filter transforms
- [ ] Clones tensors when modifying in-place
- [ ] Logs operations at appropriate levels
- [ ] Raises appropriate exceptions on errors
- [ ] Validates input data
- [ ] Validates output data

### Package __init__.py

```python
"""
My Plugin - Brief description.

This plugin provides advanced transforms for VQM24 molecular graphs.
"""

__version__ = "1.0.0"
__author__ = "Your Name <your.email@example.com>"

# Import transforms for easy access
from .transforms.my_transform import MyTransform
from .transforms.another_transform import AnotherTransform

# Define __all__ for clean imports
__all__ = [
    "MyTransform",
    "AnotherTransform",
]

# Package metadata
PLUGIN_METADATA = {
    "name": "my_plugin",
    "version": __version__,
    "author": __author__,
    "description": "Brief plugin description",
}
```

### Transform Module Structure

```python
"""
My Transform - Module docstring.

This module implements MyTransform for processing molecular graphs.
"""

from typing import Optional, Dict, Any
import torch
from torch_geometric.data import Data

from vqm24_pipeline.transformations.custom_transforms import (
    QuantumTransformBase,
    TransformMetadata
)
from vqm24_pipeline.exceptions import (
    TransformValidationError,
    TransformExecutionError
)

import logging
logger = logging.getLogger(__name__)


class MyTransform(QuantumTransformBase):
    """
    Brief transform description.
    
    Longer description with details about:
    - What the transform does
    - When to use it
    - Key features
    
    Example:
        >>> transform = MyTransform(threshold=0.5)
        >>> data = transform(input_data)
    
    Args:
        threshold: Description of threshold parameter
    
    Attributes:
        threshold: Stored threshold value
    """
    
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        logger.debug(f"Initialized {self.__class__.__name__} with threshold={threshold}")
    
    def validate_input(self, data: Data):
        """Validate input data before transformation."""
        # Chemistry validation
        is_valid, issues = self.validate_molecular_structure(data)
        if not is_valid:
            raise TransformValidationError(
                f"Molecular structure invalid: {issues}",
                transform_name=self._metadata.name
            )
        
        # Quantum validation
        is_valid, issues = self.validate_quantum_properties(data)
        if not is_valid:
            logger.warning(f"Quantum property issues: {issues}")
        
        # Custom validation
        if not hasattr(data, 'energy'):
            raise TransformValidationError(
                "Missing required attribute 'energy'",
                transform_name=self._metadata.name,
                required_attributes=['energy']
            )
    
    def transform(self, data: Data) -> Data:
        """
        Apply transformation to molecular graph.
        
        Args:
            data: Input molecular graph
        
        Returns:
            Transformed molecular graph
        
        Raises:
            TransformValidationError: If input invalid
            TransformExecutionError: If processing fails
        """
        try:
            # Clone to avoid modifying input
            data = data.clone()
            
            # Your transformation logic
            logger.debug(f"Processing molecule with {data.num_nodes} atoms")
            
            # Example: Scale energy
            if hasattr(data, 'energy'):
                data.energy = data.energy * self.threshold
                logger.debug(f"Scaled energy by {self.threshold}")
            
            return data
            
        except Exception as e:
            logger.error(f"Transform failed: {e}")
            raise TransformExecutionError(
                f"Failed to process data: {str(e)}",
                transform_name=self._metadata.name,
                original_error=e
            ) from e
    
    def validate_output(self, data: Data):
        """Validate output data after transformation."""
        if not hasattr(data, 'energy'):
            raise TransformValidationError(
                "Missing expected output attribute 'energy'",
                transform_name=self._metadata.name
            )
        
        if not torch.all(torch.isfinite(data.energy)):
            raise TransformValidationError(
                "Non-finite energy values in output",
                transform_name=self._metadata.name
            )
    
    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        """Get transform metadata."""
        return TransformMetadata(
            name="MyTransform",
            version="1.0.0",
            author="Your Name <your.email@example.com>",
            category="quantum",
            description="Scale energy values by threshold",
            paper_reference="doi:10.1234/example",
            github_url="https://github.com/user/my_plugin",
            validated_datasets=["VQM24_DFT", "VQM24_DMC"],
            required_node_features=["x"],
            required_edge_features=[],
            required_graph_attributes=["energy"]
        )
    
    @classmethod
    def get_parameter_constraints(cls) -> Dict[str, Any]:
        """Get parameter validation constraints."""
        return {
            'threshold': {
                'type': float,
                'range': (0.0, 10.0),
                'default': 0.5,
                'description': 'Energy scaling threshold'
            }
        }
    
    @classmethod
    def get_required_node_attributes(cls):
        """Get required node attributes."""
        return {'x'}
    
    @classmethod
    def get_required_graph_attributes(cls):
        """Get required graph attributes."""
        return {'energy'}


# Helper functions (if needed)
def _helper_function(data: Data) -> torch.Tensor:
    """Helper function for internal use."""
    return data.x.mean(dim=0)
```

---

## Testing Requirements

### Test Suite Structure

```
tests/
├── __init__.py
├── conftest.py                    # Pytest configuration
├── test_my_transform.py          # Transform tests
└── test_integration.py           # Integration tests
```

### conftest.py

```python
"""
Pytest configuration and fixtures for plugin tests.
"""

import pytest
import torch
from torch_geometric.data import Data


@pytest.fixture
def sample_molecular_graph():
    """Create sample molecular graph for testing."""
    return Data(
        x=torch.randn(10, 32),           # Node features
        edge_index=torch.randint(0, 10, (2, 20)),  # Edges
        edge_attr=torch.randn(20, 4),    # Edge features
        pos=torch.randn(10, 3),          # 3D coordinates
        z=torch.randint(1, 10, (10,)),   # Atomic numbers
        energy=torch.tensor([-100.5]),   # Energy
        forces=torch.randn(10, 3),       # Forces
        num_nodes=10
    )


@pytest.fixture
def sample_quantum_graph(sample_molecular_graph):
    """Create sample graph with quantum properties."""
    data = sample_molecular_graph
    data.dmc_energy = torch.tensor([-101.2])
    data.dmc_uncertainty = torch.tensor([0.05])
    data.vibmodes = torch.randn(15, 10, 3)  # [n_modes, n_atoms, 3]
    data.charges = torch.randn(10)
    return data
```

### test_my_transform.py

```python
"""
Tests for MyTransform.
"""

import pytest
import torch
from torch_geometric.data import Data

from my_plugin.transforms.my_transform import MyTransform
from vqm24_pipeline.exceptions import (
    TransformValidationError,
    TransformExecutionError
)


class TestMyTransform:
    """Test suite for MyTransform."""
    
    def test_initialization(self):
        """Test transform initialization."""
        transform = MyTransform(threshold=0.5)
        assert transform.threshold == 0.5
    
    def test_initialization_with_defaults(self):
        """Test initialization with default parameters."""
        transform = MyTransform()
        assert transform.threshold == 0.5  # Default value
    
    def test_basic_functionality(self, sample_quantum_graph):
        """Test basic transform operation."""
        transform = MyTransform(threshold=2.0)
        result = transform(sample_quantum_graph)
        
        assert result is not None
        assert result.num_nodes == 10
        assert hasattr(result, 'energy')
        # Check energy was scaled
        expected_energy = sample_quantum_graph.energy * 2.0
        assert torch.allclose(result.energy, expected_energy)
    
    def test_clones_input_data(self, sample_quantum_graph):
        """Test that input data is not modified."""
        original_energy = sample_quantum_graph.energy.clone()
        transform = MyTransform(threshold=2.0)
        result = transform(sample_quantum_graph)
        
        # Original should be unchanged
        assert torch.allclose(sample_quantum_graph.energy, original_energy)
        # Result should be different
        assert not torch.allclose(result.energy, original_energy)
    
    def test_missing_required_attribute(self):
        """Test handling of missing required attributes."""
        transform = MyTransform()
        data = Data(x=torch.randn(5, 3), num_nodes=5)
        
        with pytest.raises(TransformValidationError) as exc_info:
            transform(data)
        
        assert "Missing required attribute" in str(exc_info.value)
    
    def test_parameter_constraints(self):
        """Test parameter constraint definitions."""
        constraints = MyTransform.get_parameter_constraints()
        
        assert 'threshold' in constraints
        assert constraints['threshold']['type'] == float
        assert constraints['threshold']['range'] == (0.0, 10.0)
        assert constraints['threshold']['default'] == 0.5
    
    def test_metadata(self):
        """Test metadata retrieval."""
        metadata = MyTransform.get_metadata()
        
        assert metadata.name == "MyTransform"
        assert metadata.version == "1.0.0"
        assert metadata.category == "quantum"
        assert "energy" in metadata.required_graph_attributes
    
    def test_required_attributes(self):
        """Test required attribute definitions."""
        node_attrs = MyTransform.get_required_node_attributes()
        graph_attrs = MyTransform.get_required_graph_attributes()
        
        assert 'x' in node_attrs
        assert 'energy' in graph_attrs
    
    def test_handles_invalid_values(self, sample_quantum_graph):
        """Test handling of invalid values."""
        transform = MyTransform()
        sample_quantum_graph.energy = torch.tensor([float('nan')])
        
        with pytest.raises((TransformValidationError, TransformExecutionError)):
            transform(sample_quantum_graph)
    
    def test_reproducibility(self, sample_quantum_graph):
        """Test deterministic behavior."""
        transform = MyTransform(threshold=1.5)
        
        result1 = transform(sample_quantum_graph.clone())
        result2 = transform(sample_quantum_graph.clone())
        
        assert torch.allclose(result1.energy, result2.energy)
    
    def test_edge_cases_small_molecule(self):
        """Test with very small molecule."""
        transform = MyTransform()
        data = Data(
            x=torch.randn(1, 32),
            energy=torch.tensor([-10.0]),
            num_nodes=1
        )
        
        result = transform(data)
        assert result is not None
        assert result.num_nodes == 1
    
    def test_edge_cases_large_molecule(self):
        """Test with large molecule."""
        transform = MyTransform()
        data = Data(
            x=torch.randn(1000, 32),
            energy=torch.tensor([-1000.0]),
            num_nodes=1000
        )
        
        result = transform(data)
        assert result is not None
        assert result.num_nodes == 1000
    
    def test_compatibility_validation(self, sample_quantum_graph):
        """Test compatibility validation."""
        transform = MyTransform()
        
        is_valid, warnings = transform.validate_compatibility(
            sample_quantum_graph,
            validation_level='strict'
        )
        
        assert is_valid
        assert isinstance(warnings, list)
    
    def test_usage_statistics(self, sample_quantum_graph):
        """Test usage statistics tracking."""
        transform = MyTransform()
        
        # Apply transform multiple times
        for _ in range(5):
            transform(sample_quantum_graph.clone())
        
        stats = transform.get_usage_statistics()
        assert stats['call_count'] == 5
        assert stats['success_rate'] == 1.0


class TestMyTransformIntegration:
    """Integration tests for MyTransform."""
    
    def test_registry_integration(self):
        """Test integration with TransformRegistry."""
        from vqm24_pipeline.transformations.graph_transforms import TransformRegistry
        
        # Register transform
        TransformRegistry.register("MyTransform", MyTransform)
        
        # Retrieve from registry
        transform = TransformRegistry.get("MyTransform", threshold=3.0)
        
        assert isinstance(transform, MyTransform)
        assert transform.threshold == 3.0
    
    def test_config_integration(self, sample_quantum_graph):
        """Test configuration-based usage."""
        # Simulate config-based initialization
        config_params = {'threshold': 1.5}
        transform = MyTransform(**config_params)
        
        result = transform(sample_quantum_graph)
        assert result is not None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Test Coverage Requirements

⚠️ **Recommended Coverage:** 80% minimum

**Must Test:**
- [ ] Initialization with parameters
- [ ] Initialization with defaults
- [ ] Basic transform functionality
- [ ] Input data not modified
- [ ] Missing required attributes
- [ ] Invalid input handling
- [ ] Edge cases (small/large molecules)
- [ ] Reproducibility
- [ ] Metadata retrieval
- [ ] Parameter constraints
- [ ] Integration with registry
- [ ] Compatibility validation
- [ ] Usage statistics

---

## Documentation Requirements

### README.md Template

```markdown
# My Plugin

Brief one-sentence description of the plugin.

## Overview

Longer description explaining:
- What the plugin does
- Why you would use it
- Key features
- Use cases

## Installation

### From Plugin Directory

```bash
# Copy plugin to VQM24 plugins directory
cp -r my_plugin /path/to/vqm24/plugins/

# Verify installation
python main.py --list-plugins
```

### From Git Repository

```bash
cd /path/to/vqm24/plugins
git clone https://github.com/user/my_plugin.git
```

## Usage

### Basic Usage

```python
from my_plugin.transforms.my_transform import MyTransform

# Create transform
transform = MyTransform(threshold=0.5)

# Apply to data
result = transform(data)
```

### Configuration Usage

Add to your `config.yaml`:

```yaml
transformations:
  experimental_setups:
    with_my_plugin:
      transforms:
        - name: "MyTransform"
          params:
            threshold: 0.5
```

### Advanced Usage

```python
# Custom parameters
transform = MyTransform(
    threshold=0.75,
    # ... other parameters
)

# Validate compatibility
is_valid, warnings = transform.validate_compatibility(
    data,
    validation_level='strict'
)

# Get usage statistics
stats = transform.get_usage_statistics()
print(f"Success rate: {stats['success_rate']:.2%}")
```

## Transforms

### MyTransform

Brief description of what the transform does.

**Parameters:**
- `threshold` (float, default=0.5): Description of threshold parameter
  - Range: [0.0, 10.0]
  - Description: What this parameter controls

**Required Attributes:**
- Node features: `x`
- Graph attributes: `energy`

**Example:**
```python
transform = MyTransform(threshold=1.0)
result = transform(data)
```

**When to use:**
- Use case 1
- Use case 2

**When NOT to use:**
- Don't use if...

## Examples

See the `examples/` directory for:
- `basic_usage.py` - Simple example
- `advanced_usage.py` - Advanced features

## Testing

```bash
# Run tests
cd my_plugin
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=my_plugin --cov-report=html
```

## Development

### Requirements

- Python >= 3.10
- VQM24 >= 1.0.0
- Additional dependencies (if any)

### Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## Citation

If you use this plugin in your research, please cite:

```bibtex
@software{my_plugin,
  author = {Your Name},
  title = {My Plugin: Brief Description},
  year = {2025},
  url = {https://github.com/user/my_plugin}
}
```

## License

This plugin is licensed under the MIT License - see [LICENSE](LICENSE) file.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Support

- Issues: https://github.com/user/my_plugin/issues
- Documentation: https://my-plugin.readthedocs.io
- Email: your.email@example.com

## Acknowledgments

- VQM24 team for the base pipeline
- Contributors
- Funding sources
```

### LICENSE Template (MIT)

```text
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### CHANGELOG.md Template

```markdown
# Changelog

All notable changes to this plugin will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Features in development

## [1.0.0] - 2025-10-15

### Added
- Initial release
- MyTransform implementation
- Basic test suite
- Documentation

### Changed
- N/A (initial release)

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [0.1.0] - 2025-10-01

### Added
- Beta release
- Core functionality

---

[Unreleased]: https://github.com/user/my_plugin/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/user/my_plugin/releases/tag/v1.0.0
[0.1.0]: https://github.com/user/my_plugin/releases/tag/v0.1.0
```

---

## Distribution Formats

### Format 1: Directory Structure (Development)

**Best for:** Local development, testing

```bash
# Plugin as directory
vqm24/plugins/
└── my_plugin/
    ├── plugin.yaml
    ├── __init__.py
    ├── transforms/
    │   └── ...
    └── tests/
        └── ...
```

**Usage:**
```bash
# Copy to plugins directory
cp -r my_plugin /path/to/vqm24/plugins/

# Verify
python main.py --list-plugins
```

### Format 2: Git Repository (Recommended)

**Best for:** Version control, collaboration, distribution

```bash
# Repository structure
my_plugin/
├── .git/
├── .gitignore
├── plugin.yaml
├── __init__.py
├── transforms/
├── tests/
├── examples/
├── README.md
├── LICENSE
├── CHANGELOG.md
└── requirements.txt
```

**.gitignore:**
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
.hypothesis/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
```

**Usage:**
```bash
# Clone to plugins directory
cd /path/to/vqm24/plugins
git clone https://github.com/user/my_plugin.git

# Update
cd my_plugin
git pull

# Verify
python main.py --list-plugins
```

### Format 3: Compressed Archive (Distribution)

**Best for:** Sharing, archival

```bash
# Create archive
tar -czf my_plugin-1.0.0.tar.gz my_plugin/

# Or zip
zip -r my_plugin-1.0.0.zip my_plugin/
```

**Usage:**
```bash
# Extract to plugins directory
cd /path/to/vqm24/plugins
tar -xzf my_plugin-1.0.0.tar.gz

# Or unzip
unzip my_plugin-1.0.0.zip

# Verify
python main.py --list-plugins
```

### Format 4: Python Package (Advanced)

**Best for:** pip installation, PyPI distribution

**setup.py:**
```python
from setuptools import setup, find_packages

setup(
    name="my-plugin",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "vqm24-pipeline>=1.0.0",
        # Additional dependencies
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Brief description",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/user/my_plugin",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
```

**Usage:**
```bash
# Install in development mode
pip install -e /path/to/my_plugin

# Or install from PyPI (when published)
pip install my-plugin

# Plugin will be in site-packages, discoverable if configured
```

---

## Validation & Security

### Validation Levels

**Permissive (Development):**
- Minimal checks
- Fast validation
- Allows experimental plugins
- Warns but doesn't block

**Standard (Default):**
- Balanced validation
- Checks structure and compatibility
- Blocks unsafe plugins
- Good for most users

**Strict (Production):**
- Comprehensive validation
- Security scanning
- Checksum verification
- Trusted plugins only

### Security Checklist

Plugin developers should ensure:

- [ ] No arbitrary code execution
- [ ] No network access in transforms
- [ ] No file system writes (except logs)
- [ ] No import of dangerous modules
- [ ] No eval/exec usage
- [ ] No subprocess calls
- [ ] Validated all user inputs
- [ ] Handled all exceptions
- [ ] No hardcoded credentials
- [ ] No sensitive data in logs

### Validation Command

```bash
# Basic validation
python main.py --validate-plugin my_plugin

# Comprehensive validation
python main.py --comprehensive-validate-plugin my_plugin

# Output example:
# âœ" Structure validation passed
# âœ" Metadata validation passed  
# âœ" Compatibility check passed
# âœ" Import test passed
# âœ" Transform instantiation passed
# ⚠ No tests found (recommended)
# âœ" Security scan passed
# 
# Plugin 'my_plugin' validation: PASSED
```

### Checksum Generation

```python
# Generate SHA-256 checksum
import hashlib
import os

def generate_plugin_checksum(plugin_dir):
    """Generate SHA-256 checksum for plugin."""
    hasher = hashlib.sha256()
    
    for root, dirs, files in os.walk(plugin_dir):
        # Skip tests and examples
        dirs[:] = [d for d in dirs if d not in ['tests', 'examples', '__pycache__']]
        
        for file in sorted(files):
            if file.endswith('.py') or file == 'plugin.yaml':
                filepath = os.path.join(root, file)
                with open(filepath, 'rb') as f:
                    hasher.update(f.read())
    
    return f"sha256:{hasher.hexdigest()}"

# Usage
checksum = generate_plugin_checksum('my_plugin')
print(f"Checksum: {checksum}")
# Add to plugin.yaml
```

---

## Example Plugins

### Example 1: Simple Filter

**plugin.yaml:**
```yaml
plugin_name: "simple_filter"
version: "1.0.0"
author: "Example Author <example@email.com>"
description: "Simple energy-based filtering"
vqm24_min_version: "1.0.0"

transforms:
  - name: "EnergyFilter"
    class_name: "EnergyFilter"
    module_path: "transforms.energy_filter"
    category: "quantum"
    description: "Filter molecules by energy threshold"
    version: "1.0.0"
    required_graph_attributes: ["energy"]
    parameter_constraints:
      max_energy:
        type: "float"
        default: 0.0
        description: "Maximum energy threshold"
```

**transforms/energy_filter.py:**
```python
"""Simple energy filter transform."""

from typing import Optional
import torch
from torch_geometric.data import Data

from vqm24_pipeline.transformations.custom_transforms import (
    QuantumTransformBase,
    TransformMetadata
)


class EnergyFilter(QuantumTransformBase):
    """Filter molecules by energy threshold."""
    
    def __init__(self, max_energy: float = 0.0):
        super().__init__()
        self.max_energy = max_energy
    
    def transform(self, data: Data) -> Optional[Data]:
        """Return None if energy exceeds threshold."""
        if not hasattr(data, 'energy'):
            return data
        
        if data.energy.item() > self.max_energy:
            return None  # Filter out
        
        return data
    
    @classmethod
    def get_metadata(cls):
        return TransformMetadata(
            name="EnergyFilter",
            version="1.0.0",
            author="Example Author",
            category="quantum",
            description="Filter molecules by energy threshold",
            required_graph_attributes=["energy"]
        )
```

### Example 2: Normalization Transform

**plugin.yaml:**
```yaml
plugin_name: "advanced_normalization"
version: "1.0.0"
author: "Example Author <example@email.com>"
description: "Advanced normalization transforms"
vqm24_min_version: "1.0.0"

transforms:
  - name: "AdaptiveNormalizer"
    class_name: "AdaptiveNormalizer"
    module_path: "transforms.adaptive_normalizer"
    category: "quantum"
    description: "Size-adaptive feature normalization"
    version: "1.0.0"
    required_node_features: ["x"]
    parameter_constraints:
      small_threshold:
        type: "int"
        range: [1, 100]
        default: 10
        description: "Threshold for small molecules"
      scale_factor:
        type: "float"
        range: [0.1, 10.0]
        default: 2.0
        description: "Scaling factor for large molecules"
```

**transforms/adaptive_normalizer.py:**
```python
"""Adaptive normalization based on molecule size."""

import torch
from torch_geometric.data import Data

from vqm24_pipeline.transformations.custom_transforms import (
    QuantumTransformBase,
    TransformMetadata
)


class AdaptiveNormalizer(QuantumTransformBase):
    """Normalize features adaptively based on molecule size."""
    
    def __init__(self, small_threshold: int = 10, scale_factor: float = 2.0):
        super().__init__()
        self.small_threshold = small_threshold
        self.scale_factor = scale_factor
    
    def transform(self, data: Data) -> Data:
        """Apply size-dependent normalization."""
        data = data.clone()
        
        if data.num_nodes < self.small_threshold:
            # Small molecule: gentle normalization
            scale = 1.0
        else:
            # Large molecule: aggressive normalization
            scale = self.scale_factor
        
        if hasattr(data, 'x'):
            data.x = data.x / scale
        
        # Store metadata
        data.normalization_scale = torch.tensor(scale)
        
        return data
    
    @classmethod
    def get_metadata(cls):
        return TransformMetadata(
            name="AdaptiveNormalizer",
            version="1.0.0",
            author="Example Author",
            category="quantum",
            description="Size-adaptive feature normalization",
            required_node_features=["x"]
        )
    
    @classmethod
    def get_parameter_constraints(cls):
        return {
            'small_threshold': {
                'type': int,
                'range': (1, 100),
                'default': 10,
                'description': 'Threshold for small molecules'
            },
            'scale_factor': {
                'type': float,
                'range': (0.1, 10.0),
                'default': 2.0,
                'description': 'Scaling factor for large molecules'
            }
        }
```

### Example 3: Multi-Transform Plugin

**plugin.yaml:**
```yaml
plugin_name: "quantum_toolkit"
version: "1.0.0"
author: "Example Author <example@email.com>"
description: "Collection of quantum property transforms"
vqm24_min_version: "1.0.0"
license: "MIT"
repository: "https://github.com/user/quantum_toolkit"

transforms:
  # Transform 1: Energy normalizer
  - name: "EnergyNormalizer"
    class_name: "EnergyNormalizer"
    module_path: "transforms.energy_normalizer"
    category: "quantum"
    description: "Normalize energy values"
    version: "1.0.0"
    required_graph_attributes: ["energy"]
    parameter_constraints:
      method:
        type: "str"
        choices: ["zscore", "minmax", "robust"]
        default: "zscore"
        description: "Normalization method"
  
  # Transform 2: Force scaler
  - name: "ForceScaler"
    class_name: "ForceScaler"
    module_path: "transforms.force_scaler"
    category: "quantum"
    description: "Scale force vectors"
    version: "1.0.0"
    required_graph_attributes: ["forces"]
    parameter_constraints:
      scale_factor:
        type: "float"
        range: [0.1, 10.0]
        default: 1.0
        description: "Force scaling factor"
  
  # Transform 3: Charge normalizer
  - name: "ChargeNormalizer"
    class_name: "ChargeNormalizer"
    module_path: "transforms.charge_normalizer"
    category: "quantum"
    description: "Normalize Mulliken charges"
    version: "1.0.0"
    required_graph_attributes: ["charges"]
    parameter_constraints:
      center:
        type: "bool"
        default: true
        description: "Center charges to mean=0"

validated_datasets:
  - "VQM24_DFT"
  - "VQM24_DMC"

tags:
  - "normalization"
  - "quantum"
  - "energy"
  - "forces"
```

---

## Best Practices

### 1. Plugin Design

**DO:**
- âœ… Keep plugins focused and single-purpose
- âœ… Use clear, descriptive names
- âœ… Provide comprehensive documentation
- âœ… Include examples and tests
- âœ… Follow semantic versioning
- âœ… Handle errors gracefully
- âœ… Log operations appropriately
- âœ… Validate inputs and outputs

**DON'T:**
- âŒ Create monolithic plugins
- âŒ Use ambiguous names
- âŒ Skip documentation
- âŒ Forget to test
- âŒ Break backward compatibility
- âŒ Crash on errors
- âŒ Hide operations from users
- âŒ Assume input validity

### 2. Versioning Strategy

**Semantic Versioning (MAJOR.MINOR.PATCH):**

```
1.0.0 → 1.0.1    # Bug fix (PATCH)
1.0.1 → 1.1.0    # New feature (MINOR)
1.1.0 → 2.0.0    # Breaking change (MAJOR)
```

**When to increment:**

- **MAJOR:** Breaking API changes
  - Changed parameter names
  - Changed default behaviors
  - Removed features
  - Changed output format

- **MINOR:** New features (backward compatible)
  - New transforms added
  - New parameters (with defaults)
  - Enhanced functionality
  - Performance improvements

- **PATCH:** Bug fixes
  - Fixed incorrect behavior
  - Fixed crashes
  - Documentation fixes
  - Minor improvements

### 3. Dependency Management

**Minimize dependencies:**
```yaml
# Good: Only necessary dependencies
dependencies:
  scipy: ">=1.15.0"

# Avoid: Too many dependencies
dependencies:
  scipy: ">=1.15.0"
  pandas: ">=2.0.0"
  numpy: ">=1.26.0"    # Already in VQM24
  torch: ">=2.3.0"     # Already in VQM24
```

**Pin minimum versions:**
```yaml
# Good: Minimum version specified
dependencies:
  custom_lib: ">=2.0.0"

# Avoid: Exact version (too restrictive)
dependencies:
  custom_lib: "==2.1.5"
```

### 4. Error Handling

**Use appropriate exceptions:**
```python
from vqm24_pipeline.exceptions import (
    TransformValidationError,    # For validation issues
    TransformExecutionError,     # For runtime errors
    TransformConfigurationError  # For config issues
)

def transform(self, data: Data) -> Data:
    # Validation errors
    if not hasattr(data, 'required_attr'):
        raise TransformValidationError(
            "Missing required attribute",
            transform_name=self._metadata.name,
            required_attributes=['required_attr']
        )
    
    # Execution errors
    try:
        result = risky_operation(data)
    except Exception as e:
        raise TransformExecutionError(
            "Operation failed",
            transform_name=self._metadata.name,
            original_error=e
        ) from e
    
    return result
```

### 5. Testing Strategy

**Test pyramid:**
```
            /\
           /  \      Integration Tests (few)
          /____\
         /      \    Transform Tests (many)
        /________\
       /          \  Unit Tests (most)
      /____________\
```

**Minimum tests:**
- Basic functionality
- Edge cases
- Error handling
- Parameter validation
- Reproducibility

### 6. Documentation Strategy

**Essential documentation:**
1. **README.md** - Overview, installation, usage
2. **Docstrings** - All classes and methods
3. **Examples** - Basic and advanced usage
4. **CHANGELOG.md** - Version history
5. **LICENSE** - Legal terms

**Documentation checklist:**
- [ ] Plugin purpose clear
- [ ] Installation instructions
- [ ] Basic usage example
- [ ] All parameters documented
- [ ] Return values documented
- [ ] Exceptions documented
- [ ] When to use / not use
- [ ] Examples provided
- [ ] Tests documented

### 7. Performance Considerations

**Optimize for common case:**
```python
# Good: Fast path for common case
def transform(self, data: Data) -> Data:
    if not hasattr(data, 'optional_attr'):
        # Fast path: simple operation
        return self._simple_transform(data)
    else:
        # Slow path: complex operation
        return self._complex_transform(data)

# Avoid: Always using slow path
def transform(self, data: Data) -> Data:
    # Always does expensive operation
    return self._complex_transform(data)
```

**Cache expensive computations:**
```python
def __init__(self):
    super().__init__()
    self._cache = {}

def transform(self, data: Data) -> Data:
    cache_key = self._get_cache_key(data)
    if cache_key in self._cache:
        return self._cache[cache_key]
    
    result = self._expensive_operation(data)
    self._cache[cache_key] = result
    return result
```

---

## Troubleshooting

### Issue: Plugin not discovered

**Symptoms:**
- Plugin doesn't appear in `--list-plugins`
- Transform not found in registry

**Solutions:**

1. **Check plugin location:**
```bash
# Verify plugin in correct directory
ls -la /path/to/vqm24/plugins/my_plugin

# Should see plugin.yaml
```

2. **Verify plugin.yaml:**
```bash
# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('plugin.yaml'))"

# Check required fields
grep -E "plugin_name|version|author|description" plugin.yaml
```

3. **Check configuration:**
```yaml
# In config.yaml
plugins:
  enabled: true
  plugin_paths:
    - ./plugins  # Must match actual location
```

4. **Enable debug logging:**
```bash
export LOG_LEVEL=DEBUG
python main.py --list-plugins
```

### Issue: Validation fails

**Symptoms:**
- `--validate-plugin` reports errors
- Plugin loads but doesn't work

**Solutions:**

1. **Check plugin structure:**
```bash
# Verify required files
ls -la my_plugin/
# Should see:
# - plugin.yaml
# - __init__.py
# - transforms/
```

2. **Validate plugin.yaml:**
```bash
# Comprehensive validation
python main.py --comprehensive-validate-plugin my_plugin

# Check specific issues
grep -E "error|warning" validation_output.txt
```

3. **Test imports:**
```python
# Test manual import
from my_plugin.transforms.my_transform import MyTransform
transform = MyTransform()
print("Import successful!")
```

4. **Check metadata:**
```python
# Verify metadata
from my_plugin.transforms.my_transform import MyTransform
metadata = MyTransform.get_metadata()
print(f"Name: {metadata.name}")
print(f"Version: {metadata.version}")
```

### Issue: Transform not working

**Symptoms:**
- Transform loads but produces wrong results
- Transform crashes on some data

**Solutions:**

1. **Test transform directly:**
```python
from my_plugin.transforms.my_transform import MyTransform
from torch_geometric.data import Data
import torch

transform = MyTransform()
test_data = Data(x=torch.randn(10, 32), num_nodes=10)

result = transform(test_data)
print(f"Result: {result}")
```

2. **Check logs:**
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python main.py

# Check for errors
grep -i "error" vqm24.log
```

3. **Run tests:**
```bash
cd my_plugin
pytest tests/ -v -s
```

4. **Check compatibility:**
```python
transform = MyTransform()
is_valid, warnings = transform.validate_compatibility(
    data,
    validation_level='strict'
)
print(f"Valid: {is_valid}")
print(f"Warnings: {warnings}")
```

### Issue: Dependencies not found

**Symptoms:**
- Import errors for dependencies
- ModuleNotFoundError

**Solutions:**

1. **Install dependencies:**
```bash
# If requirements.txt provided
pip install -r my_plugin/requirements.txt

# Or manually
pip install required_package>=1.0.0
```

2. **Check dependencies in plugin.yaml:**
```yaml
dependencies:
  required_package: ">=1.0.0"
```

3. **Verify installation:**
```python
import required_package
print(f"Version: {required_package.__version__}")
```

### Issue: Checksum verification fails

**Symptoms:**
- Plugin rejected with checksum mismatch
- `enforce_checksums: true` blocks plugin

**Solutions:**

1. **Regenerate checksum:**
```python
# See "Checksum Generation" section
checksum = generate_plugin_checksum('my_plugin')
print(f"New checksum: {checksum}")
```

2. **Update plugin.yaml:**
```yaml
checksum: "sha256:NEW_CHECKSUM_HERE"
```

3. **Disable checksum (temporarily):**
```yaml
# In config.yaml (NOT for production!)
plugins:
  enforce_checksums: false
```

---

## Quick Reference

### Minimum Viable Plugin

```
my_plugin/
├── plugin.yaml              # Metadata
├── __init__.py             # Empty or imports
└── transforms/
    ├── __init__.py         # Empty
    └── my_transform.py     # Transform class
```

### plugin.yaml Minimum

```yaml
plugin_name: "my_plugin"
version: "1.0.0"
author: "Your Name <email@example.com>"
description: "Brief description"
vqm24_min_version: "1.0.0"

transforms:
  - name: "MyTransform"
    class_name: "MyTransform"
    module_path: "transforms.my_transform"
    category: "quantum"
    description: "Transform description"
    version: "1.0.0"
```

### Transform Minimum

```python
from torch_geometric.data import Data
from vqm24_pipeline.transformations.custom_transforms import (
    QuantumTransformBase,
    TransformMetadata
)

class MyTransform(QuantumTransformBase):
    def transform(self, data: Data) -> Data:
        return data
    
    @classmethod
    def get_metadata(cls):
        return TransformMetadata(
            name="MyTransform",
            version="1.0.0",
            author="Your Name",
            category="quantum",
            description="Brief description"
        )
```

### Validation Commands

```bash
# List plugins
python main.py --list-plugins

# Validate plugin
python main.py --validate-plugin my_plugin

# Comprehensive validation
python main.py --comprehensive-validate-plugin my_plugin

# Plugin information
python main.py --plugin-info my_plugin
```

### Installation Methods

```bash
# Method 1: Copy directory
cp -r my_plugin /path/to/vqm24/plugins/

# Method 2: Git clone
cd /path/to/vqm24/plugins
git clone https://github.com/user/my_plugin.git

# Method 3: Extract archive
tar -xzf my_plugin-1.0.0.tar.gz -C /path/to/vqm24/plugins/

# Verify
python main.py --list-plugins
```

---

## Appendix

### A. Complete Example Plugin

See `examples/complete_plugin/` directory for a full, working example plugin with:
- Complete plugin.yaml
- Multiple transforms
- Comprehensive tests
- Full documentation
- Usage examples

### B. Schema Validation

JSON Schema for plugin.yaml validation available in:
`vqm24_pipeline/schemas/plugin_schema.json`

### C. Template Repository

Plugin template repository available at:
`https://github.com/vqm24/plugin-template`

Use as starting point for new plugins.

### D. Community Plugins

Curated list of community plugins:
`https://github.com/vqm24/awesome-plugins`

### E. Support

- Documentation: https://vqm24.readthedocs.io/plugins
- Issues: https://github.com/vqm24/pipeline/issues
- Discussions: https://github.com/vqm24/pipeline/discussions

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-10-15 | Initial release |

---

**End of Plugin Distribution Format Specification**
