# VQM24 Plugin Distribution - Quick Reference Guide
## Fast Start for Plugin Developers

**Version:** 1.0.0  
**Date:** October 15, 2025

---

## 🚀 5-Minute Quick Start

### Create Minimum Viable Plugin

```bash
# 1. Create plugin structure
mkdir my_plugin
cd my_plugin
mkdir transforms tests

# 2. Create plugin.yaml
cat > plugin.yaml << EOF
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
EOF

# 3. Create __init__.py
cat > __init__.py << EOF
from .transforms.my_transform import MyTransform
__version__ = "1.0.0"
__all__ = ["MyTransform"]
EOF

# 4. Create transform
cat > transforms/__init__.py << EOF
EOF

cat > transforms/my_transform.py << EOF
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
EOF

# 5. Install
cp -r ../my_plugin /path/to/vqm24/plugins/

# 6. Verify
python main.py --list-plugins
```

**Done! You have a working plugin in 5 minutes.**

---

## 📋 Essential Checklists

### Plugin Structure Checklist

```
✅ my_plugin/
   ✅ plugin.yaml              # REQUIRED
   ✅ __init__.py             # REQUIRED
   ✅ transforms/
      ✅ __init__.py          # REQUIRED
      ✅ my_transform.py      # REQUIRED
   ⚠️ tests/                  # RECOMMENDED
      ⚠️ __init__.py
      ⚠️ test_my_transform.py
   ⚠️ README.md               # RECOMMENDED
   ⚠️ LICENSE                 # RECOMMENDED
   ❌ CHANGELOG.md            # OPTIONAL
   ❌ examples/               # OPTIONAL
```

### plugin.yaml Mandatory Fields

```yaml
✅ plugin_name: "..."
✅ version: "1.0.0"
✅ author: "..."
✅ description: "..."
✅ vqm24_min_version: "1.0.0"
✅ transforms:
   - name: "..."
     class_name: "..."
     module_path: "..."
     category: "..."
     description: "..."
     version: "1.0.0"
```

### Transform Implementation Checklist

```python
class MyTransform(QuantumTransformBase):  # ✅ Inherit
    def __init__(self, param=1.0):
        super().__init__()                 # ✅ Call super
        self.param = param
    
    def transform(self, data):            # ✅ Implement
        return data
    
    @classmethod
    def get_metadata(cls):                # ✅ Implement
        return TransformMetadata(...)
    
    @classmethod
    def get_parameter_constraints(cls):   # ⚠️ Recommended
        return {...}
```

---

## 🔧 Common Patterns

### Pattern: Simple Transform

```python
class ScaleEnergy(QuantumTransformBase):
    def __init__(self, factor: float = 1.0):
        super().__init__()
        self.factor = factor
    
    def transform(self, data: Data) -> Data:
        if hasattr(data, 'energy'):
            data.energy = data.energy * self.factor
        return data
    
    @classmethod
    def get_metadata(cls):
        return TransformMetadata(
            name="ScaleEnergy",
            version="1.0.0",
            author="Your Name",
            category="quantum",
            description="Scale energy by factor",
            required_graph_attributes=["energy"]
        )
```

### Pattern: Filter Transform

```python
class FilterByEnergy(QuantumTransformBase):
    def __init__(self, max_energy: float = 0.0):
        super().__init__()
        self.max_energy = max_energy
    
    def transform(self, data: Data) -> Optional[Data]:
        if data.energy.item() > self.max_energy:
            return None  # Filter out
        return data
    
    @classmethod
    def get_metadata(cls):
        return TransformMetadata(
            name="FilterByEnergy",
            version="1.0.0",
            author="Your Name",
            category="quantum",
            description="Filter high-energy molecules",
            required_graph_attributes=["energy"]
        )
```

### Pattern: Multi-Step Transform

```python
class NormalizeAndAugment(QuantumTransformBase):
    def __init__(self, normalize: bool = True, noise_std: float = 0.01):
        super().__init__()
        self.normalize = normalize
        self.noise_std = noise_std
    
    def transform(self, data: Data) -> Data:
        data = data.clone()
        
        # Step 1: Normalize
        if self.normalize:
            data.x = (data.x - data.x.mean()) / data.x.std()
        
        # Step 2: Augment
        if self.noise_std > 0:
            noise = torch.randn_like(data.x) * self.noise_std
            data.x = data.x + noise
        
        return data
    
    @classmethod
    def get_metadata(cls):
        return TransformMetadata(
            name="NormalizeAndAugment",
            version="1.0.0",
            author="Your Name",
            category="augmentation",
            description="Normalize and add noise",
            required_node_features=["x"]
        )
```

---

## 🎯 Parameter Constraints

### Basic Constraints

```python
@classmethod
def get_parameter_constraints(cls):
    return {
        # Float with range
        'threshold': {
            'type': float,
            'range': (0.0, 1.0),
            'default': 0.5,
            'description': 'Filtering threshold'
        },
        
        # Integer with range
        'max_atoms': {
            'type': int,
            'range': (1, 1000),
            'default': 100,
            'description': 'Maximum atoms'
        },
        
        # String with choices
        'method': {
            'type': str,
            'choices': ['zscore', 'minmax', 'robust'],
            'default': 'zscore',
            'description': 'Normalization method'
        },
        
        # Boolean
        'enabled': {
            'type': bool,
            'default': True,
            'description': 'Enable feature'
        },
        
        # Optional parameter
        'seed': {
            'type': int,
            'default': None,
            'description': 'Random seed (null for random)'
        }
    }
```

---

## 📦 Distribution Methods

### Method 1: Directory (Development)

```bash
# Copy to plugins directory
cp -r my_plugin /path/to/vqm24/plugins/
```

### Method 2: Git (Recommended)

```bash
# Clone to plugins directory
cd /path/to/vqm24/plugins
git clone https://github.com/user/my_plugin.git
```

### Method 3: Archive (Sharing)

```bash
# Create archive
tar -czf my_plugin-1.0.0.tar.gz my_plugin/

# Extract
tar -xzf my_plugin-1.0.0.tar.gz -C /path/to/vqm24/plugins/
```

### Method 4: pip (Advanced)

```bash
# Install from PyPI
pip install my-plugin

# Install from git
pip install git+https://github.com/user/my_plugin.git
```

---

## 🔍 Testing & Validation

### Quick Test

```python
# test_my_transform.py
import pytest
from my_plugin.transforms.my_transform import MyTransform
from torch_geometric.data import Data
import torch

def test_basic():
    transform = MyTransform()
    data = Data(x=torch.randn(10, 32), num_nodes=10)
    result = transform(data)
    assert result is not None
```

### Validation Commands

```bash
# List plugins
python main.py --list-plugins

# Validate plugin
python main.py --validate-plugin my_plugin

# Comprehensive validation
python main.py --comprehensive-validate-plugin my_plugin
```

---

## 🐛 Quick Troubleshooting

### Plugin not found?

```bash
# Check location
ls -la /path/to/vqm24/plugins/my_plugin

# Check config
grep -A2 "plugin_paths:" config.yaml

# Enable in config
# In config.yaml:
plugins:
  enabled: true
  plugin_paths:
    - ./plugins
```

### Import error?

```python
# Test import directly
python -c "from my_plugin.transforms.my_transform import MyTransform"
```

### Transform not working?

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py --list-plugins
```

---

## 📚 Template Library

### Minimal plugin.yaml

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

### Standard plugin.yaml

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
    required_graph_attributes: ["energy"]
    parameter_constraints:
      threshold:
        type: "float"
        range: [0.0, 1.0]
        default: 0.5
        description: "Threshold parameter"

license: "MIT"
repository: "https://github.com/user/my_plugin"
validated_datasets: ["VQM24_DFT"]
```

### Minimal README.md

```markdown
# My Plugin

Brief description.

## Installation

```bash
cp -r my_plugin /path/to/vqm24/plugins/
```

## Usage

```python
from my_plugin import MyTransform
transform = MyTransform(threshold=0.5)
result = transform(data)
```

## License

MIT
```

---

## 🎓 Best Practices Summary

### DO ✅

- Keep plugins focused and single-purpose
- Use semantic versioning (1.0.0)
- Provide clear documentation
- Include tests
- Handle errors gracefully
- Clone tensors when modifying
- Log important operations
- Validate inputs

### DON'T ❌

- Create monolithic plugins
- Skip documentation
- Forget to test
- Modify input data in-place
- Override __call__() method
- Use hardcoded paths
- Ignore missing attributes
- Crash on errors

---

## 🔗 Quick Links

| Resource | Location |
|----------|----------|
| Full Specification | `Phase3_Step3.2.6_Plugin_Distribution_Format.md` |
| Complete Example | `example_plugin_complete/` |
| Implementation Summary | `Phase3_Step3.2.6_Implementation_Summary.md` |
| Custom Transforms Guide | `custom_transforms_KEY_INFO.md` |
| Migration Guide | `Plugin_System_Migration_Guide.md` |

---

## 📞 Support

- **Documentation:** Full specification document
- **Examples:** Complete example plugin
- **Issues:** GitHub issue tracker
- **Community:** GitHub discussions

---

## ✅ Pre-Release Checklist

Before distributing your plugin:

- [ ] plugin.yaml complete and valid
- [ ] All required files present
- [ ] Transforms implement required methods
- [ ] Tests pass (if included)
- [ ] README.md comprehensive
- [ ] License included
- [ ] Version follows semantic versioning
- [ ] Validated with `--validate-plugin`
- [ ] Tested in actual pipeline
- [ ] Documentation up to date

---

**Quick Reference Version:** 1.0.0  
**Last Updated:** October 15, 2025

**For complete details, see the full specification document.**
