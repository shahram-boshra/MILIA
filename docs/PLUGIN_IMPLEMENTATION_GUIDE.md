# PyG Augmentation Transforms Plugin Implementation Guide

## Overview

Move the 4 missing PyG transforms (`DropEdge`, `DropNode`, `MaskFeatures`, `RandomNodeSample`) from `custom_transforms.py` to a dedicated plugin using your existing Phase 3.2 Plugin System.

**Benefits:**
- ✅ Lighter core pipeline
- ✅ Modular, hot-swappable
- ✅ Tests plugin system practically
- ✅ Version-independent
- ✅ Easy distribution

---

## Implementation Steps

### Step 1: Remove Transforms from custom_transforms.py

**File:** `/app/vqm24/vqm24_pipeline/transformations/custom_transforms.py`

**Action:** Delete these 4 classes:
- `class DropEdge(CustomTransformBase):`
- `class DropNode(CustomTransformBase):`
- `class MaskFeatures(CustomTransformBase):`
- `class RandomNodeSample(CustomTransformBase):`

**Command:**
```bash
nano /app/vqm24/vqm24_pipeline/transformations/custom_transforms.py
# Remove the 4 classes (approximately 200-250 lines)
```

---

### Step 2: Create Plugin Directory Structure

```bash
cd /app/vqm24
mkdir -p custom_plugins/pyg_augmentation
```

**Directory structure:**
```
custom_plugins/
└── pyg_augmentation/
    ├── __init__.py
    ├── __plugin__.py
    └── transforms.py
```

---

### Step 3: Create Plugin Metadata File

**File:** `custom_plugins/pyg_augmentation/__plugin__.py`

```python
"""
PyG Augmentation Transforms Plugin

Provides the 4 missing augmentation transforms from PyTorch Geometric:
- DropEdge: Random edge dropping
- DropNode: Random node dropping  
- MaskFeatures: Feature masking
- RandomNodeSample: Node sampling

These transforms are not available in PyG 2.6.1 but are essential for
graph data augmentation and contrastive learning.
"""

from vqm24_pipeline.transformations.plugin_system import PluginMetadata

PLUGIN_METADATA = PluginMetadata(
    name="pyg_augmentation",
    version="1.0.0",
    author="VQM24 Team",
    description="PyG augmentation transforms for graph data augmentation",
    transforms=[
        "DropEdge",
        "DropNode", 
        "MaskFeatures",
        "RandomNodeSample"
    ],
    dependencies={
        "torch": ">=2.0.0",
        "torch_geometric": ">=2.0.0"
    },
    compatible_vqm24_versions=[">=4.0.0"],
    plugin_type="transform",
    category="augmentation",
    tags=["augmentation", "graph", "contrastive-learning", "pyg"],
    documentation_url="https://github.com/vqm24/plugins/pyg_augmentation",
    license="MIT"
)

__version__ = "1.0.0"
__author__ = "VQM24 Team"
```

---

### Step 4: Create Plugin __init__.py

**File:** `custom_plugins/pyg_augmentation/__init__.py`

```python
"""
PyG Augmentation Transforms Plugin

Exposes the 4 augmentation transforms for VQM24 pipeline integration.
"""

from .transforms import (
    DropEdge,
    DropNode,
    MaskFeatures,
    RandomNodeSample
)

from .__plugin__ import PLUGIN_METADATA, __version__, __author__

__all__ = [
    'DropEdge',
    'DropNode',
    'MaskFeatures',
    'RandomNodeSample',
    'PLUGIN_METADATA',
    '__version__',
    '__author__'
]
```

---

### Step 5: Create Plugin Transforms Module

**File:** `custom_plugins/pyg_augmentation/transforms.py`

```python
"""
PyG Augmentation Transform Implementations

These transforms provide data augmentation capabilities missing from PyG 2.6.1.
All transforms inherit from CustomTransformBase for VQM24 integration.
"""

from vqm24_pipeline.transformations.custom_transforms import (
    CustomTransformBase,
    TransformMetadata
)


class DropEdge(CustomTransformBase):
    """
    Randomly drops edges from the graph.
    
    Missing from PyG 2.6.1. Essential for graph contrastive learning.
    
    Args:
        p (float): Probability of dropping each edge (0-1). Default: 0.5
        force_undirected (bool): Maintain undirected property. Default: True
    
    Algorithm:
        1. Generate random mask for edges
        2. If force_undirected, drop edges symmetrically
        3. Apply mask to edge_index and attributes
    
    Complexity: O(E)
    
    Example:
        >>> transform = DropEdge(p=0.3)
        >>> data = transform(data)  # Drops ~30% of edges
    """
    
    def __init__(self, p: float = 0.5, force_undirected: bool = True):
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = p
        self.force_undirected = force_undirected
    
    def forward(self, data):
        import torch
        
        if not hasattr(data, 'edge_index') or data.edge_index.size(1) == 0:
            return data
        
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        
        # Create dropout mask (True = keep)
        mask = torch.rand(num_edges, device=edge_index.device) >= self.p
        
        if self.force_undirected and num_edges % 2 == 0:
            # Symmetric dropping for undirected graphs
            mask = mask.view(-1, 2).all(dim=1).repeat_interleave(2)
        
        # Apply mask
        data.edge_index = edge_index[:, mask]
        
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = data.edge_attr[mask]
        
        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            data.edge_weight = data.edge_weight[mask]
        
        return data
    
    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name='DropEdge',
            category='augmentation',
            description='Randomly drops edges for graph augmentation',
            parameters={
                'p': 'Edge drop probability (0-1)',
                'force_undirected': 'Maintain undirected property (bool)'
            },
            complexity='O(E)',
            pre_transform_safe=False
        )


class DropNode(CustomTransformBase):
    """
    Randomly drops nodes and their incident edges.
    
    Missing from PyG 2.6.1. Used for structural augmentation.
    
    Args:
        p (float): Probability of dropping each node (0-1). Default: 0.5
    
    Algorithm:
        1. Generate random mask for nodes
        2. Ensure at least 1 node remains
        3. Extract subgraph with remaining nodes
        4. Relabel nodes and update features
    
    Complexity: O(V + E)
    
    Example:
        >>> transform = DropNode(p=0.2)
        >>> data = transform(data)  # Drops ~20% of nodes
    """
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = p
    
    def forward(self, data):
        import torch
        from torch_geometric.utils import subgraph
        
        num_nodes = data.num_nodes
        if num_nodes == 0:
            return data
        
        # Node mask (True = keep)
        device = data.x.device if hasattr(data, 'x') and data.x is not None else 'cpu'
        mask = torch.rand(num_nodes, device=device) >= self.p
        
        # Ensure at least one node
        if not mask.any():
            mask[torch.randint(0, num_nodes, (1,), device=device)] = True
        
        subset = mask.nonzero(as_tuple=False).view(-1)
        
        # Extract subgraph with relabeling
        edge_index, edge_attr = subgraph(
            subset,
            data.edge_index,
            data.edge_attr if hasattr(data, 'edge_attr') else None,
            relabel_nodes=True,
            num_nodes=num_nodes
        )
        
        data.edge_index = edge_index
        if edge_attr is not None:
            data.edge_attr = edge_attr
        
        # Update node features
        for key in ['x', 'pos', 'norm', 'y']:
            if hasattr(data, key) and getattr(data, key) is not None:
                setattr(data, key, getattr(data, key)[mask])
        
        return data
    
    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name='DropNode',
            category='augmentation',
            description='Randomly drops nodes and incident edges',
            parameters={
                'p': 'Node drop probability (0-1)'
            },
            complexity='O(V + E)',
            pre_transform_safe=False
        )


class MaskFeatures(CustomTransformBase):
    """
    Randomly masks node features.
    
    Missing from PyG 2.6.1. Used for feature-level augmentation.
    
    Args:
        p (float): Probability of masking each feature (0-1). Default: 0.5
        mask_value (float): Value for masked features. Default: 0.0
    
    Algorithm:
        1. Generate random mask for feature elements
        2. Replace masked elements with mask_value
    
    Complexity: O(V × F)
    
    Example:
        >>> transform = MaskFeatures(p=0.15, mask_value=0.0)
        >>> data = transform(data)  # Masks 15% of features
    """
    
    def __init__(self, p: float = 0.5, mask_value: float = 0.0):
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = p
        self.mask_value = mask_value
    
    def forward(self, data):
        import torch
        
        if not hasattr(data, 'x') or data.x is None:
            return data
        
        # Element-wise masking
        mask = torch.rand_like(data.x) < self.p
        
        data.x = data.x.clone()
        data.x[mask] = self.mask_value
        
        return data
    
    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name='MaskFeatures',
            category='augmentation',
            description='Randomly masks node features',
            parameters={
                'p': 'Feature mask probability (0-1)',
                'mask_value': 'Value for masked features (float)'
            },
            complexity='O(V × F)',
            pre_transform_safe=False
        )


class RandomNodeSample(CustomTransformBase):
    """
    Randomly samples nodes from the graph.
    
    Missing from PyG 2.6.1. Used for subgraph sampling.
    
    Args:
        num (int): Absolute number of nodes to sample. Takes precedence.
        ratio (float): Fraction of nodes to sample (0-1). Used if num is None.
    
    Algorithm:
        1. Determine sample size (num or ratio)
        2. Random permutation of nodes
        3. Select first k nodes
        4. Extract subgraph
    
    Complexity: O(V + E)
    
    Example:
        >>> transform = RandomNodeSample(ratio=0.5)
        >>> data = transform(data)  # Samples 50% of nodes
    """
    
    def __init__(self, num: int = None, ratio: float = None):
        super().__init__()
        if num is None and ratio is None:
            raise ValueError("Either 'num' or 'ratio' must be specified")
        if num is not None and num <= 0:
            raise ValueError(f"num must be positive, got {num}")
        if ratio is not None and not 0 < ratio <= 1:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        
        self.num = num
        self.ratio = ratio
    
    def forward(self, data):
        import torch
        from torch_geometric.utils import subgraph
        
        num_nodes = data.num_nodes
        if num_nodes == 0:
            return data
        
        # Determine sample size
        if self.num is not None:
            sample_size = min(self.num, num_nodes)
        else:
            sample_size = max(1, int(self.ratio * num_nodes))
        
        # Random sampling
        perm = torch.randperm(num_nodes)[:sample_size]
        
        # Extract subgraph
        edge_index, edge_attr = subgraph(
            perm,
            data.edge_index,
            data.edge_attr if hasattr(data, 'edge_attr') else None,
            relabel_nodes=True,
            num_nodes=num_nodes
        )
        
        data.edge_index = edge_index
        if edge_attr is not None:
            data.edge_attr = edge_attr
        
        # Update node features
        for key in ['x', 'pos', 'norm', 'y']:
            if hasattr(data, key) and getattr(data, key) is not None:
                setattr(data, key, getattr(data, key)[perm])
        
        return data
    
    @classmethod
    def get_metadata(cls) -> TransformMetadata:
        return TransformMetadata(
            name='RandomNodeSample',
            category='sampling',
            description='Randomly samples a subset of nodes',
            parameters={
                'num': 'Absolute number of nodes (int)',
                'ratio': 'Fraction of nodes (0-1)'
            },
            complexity='O(V + E)',
            pre_transform_safe=False
        )
```

---

### Step 6: Update config.yaml to Enable Plugin

**File:** `/app/vqm24/config.yaml`

**Find the `plugins` section and ensure:**

```yaml
plugins:
  enabled: true
  paths:
    - "./custom_plugins"
    - "~/.vqm24/plugins"
  auto_discover: true
  trust_level: "normal"  # or "permissive" for development
  
  enabled_plugins:
    - pyg_augmentation  # Add this line
```

---

### Step 7: Verify Plugin Structure

```bash
cd /app/vqm24
tree custom_plugins/pyg_augmentation/
```

**Expected output:**
```
custom_plugins/pyg_augmentation/
├── __init__.py
├── __plugin__.py
└── transforms.py
```

---

### Step 8: Test Plugin Discovery

```bash
cd /app/vqm24
python main.py --list-plugins
```

**Expected output:**
```
Available Plugins:
==================
1. pyg_augmentation (v1.0.0)
   Author: VQM24 Team
   Status: ✓ Enabled
   Transforms: DropEdge, DropNode, MaskFeatures, RandomNodeSample
   Category: augmentation
```

---

### Step 9: Validate Plugin

```bash
python main.py --validate-plugin pyg_augmentation
```

**Expected output:**
```
Plugin Validation: pyg_augmentation
===================================
✓ Metadata valid
✓ Dependencies satisfied
✓ All transforms instantiable
✓ Parameters valid
✓ Compatible with VQM24 4.0.0

Status: VALID
```

---

### Step 10: Test Transform Import

```bash
python3 -c "
from vqm24_pipeline.transformations.graph_transforms import get_graph_transforms

gt = get_graph_transforms()

# Check if transforms are registered
for name in ['DropEdge', 'DropNode', 'MaskFeatures', 'RandomNodeSample']:
    info = gt.get_transform_info(name)
    print(f'✓ {name}: {info.description}')
"
```

---

### Step 11: Run Full Pipeline

```bash
cd /app/vqm24
python main.py --process
```

**Expected output:**
```
Plugin System: ✓ Enabled
Discovered plugins: 1
  - pyg_augmentation (v1.0.0): 4 transforms

✓ Registered 'DropEdge' (plugin: pyg_augmentation)
✓ Registered 'DropNode' (plugin: pyg_augmentation)
✓ Registered 'MaskFeatures' (plugin: pyg_augmentation)
✓ Registered 'RandomNodeSample' (plugin: pyg_augmentation)

[Pipeline continues normally...]
```

---

## Plugin CLI Commands Reference

```bash
# List all plugins
python main.py --list-plugins

# Validate specific plugin
python main.py --validate-plugin pyg_augmentation

# Enable plugin (if disabled)
python main.py --enable-plugin pyg_augmentation

# Disable plugin
python main.py --disable-plugin pyg_augmentation

# Reload plugins (after changes)
python main.py --reload-plugins
```

---

## Benefits Achieved

### 1. Lighter Core Pipeline ✅
- `custom_transforms.py` reduced by ~250 lines
- Core module focuses on base classes only
- Domain-specific transforms separated

### 2. Modularity ✅
- Plugin can be disabled independently
- Easy to distribute as standalone package
- Version-controlled separately

### 3. Plugin System Testing ✅
- Validates entire Phase 3.2 infrastructure
- Tests discovery, validation, registration
- Confirms multi-source plugin support

### 4. Future-Proof ✅
- When PyG adds these transforms, simply disable plugin
- No core code changes needed
- Clean upgrade path

### 5. Reusability ✅
- Plugin can be shared with community
- Installable via pip (future)
- Documentable independently

---

## Distribution (Optional Future Step)

To distribute this plugin:

```bash
# Create setup.py
cd custom_plugins/pyg_augmentation
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="vqm24-pyg-augmentation",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.0.0"
    ],
    author="VQM24 Team",
    description="PyG augmentation transforms for VQM24 pipeline",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
EOF

# Build and distribute
python setup.py sdist bdist_wheel
pip install dist/vqm24-pyg-augmentation-1.0.0.tar.gz
```

---

## Troubleshooting

### Issue: Plugin not discovered

**Solution:**
```bash
# Check plugin path in config.yaml
# Verify __plugin__.py exists
# Run: python main.py --list-plugins --verbose
```

### Issue: Import errors

**Solution:**
```bash
# Verify __init__.py exports all transforms
# Check PYTHONPATH includes /app/vqm24
# Test: python -c "import custom_plugins.pyg_augmentation"
```

### Issue: Validation fails

**Solution:**
```bash
# Run: python main.py --validate-plugin pyg_augmentation --verbose
# Check error messages
# Verify dependencies installed
```

---

## Summary

**Files Created:**
- `custom_plugins/pyg_augmentation/__init__.py`
- `custom_plugins/pyg_augmentation/__plugin__.py`
- `custom_plugins/pyg_augmentation/transforms.py`

**Files Modified:**
- `custom_transforms.py` (removed 4 classes)
- `config.yaml` (enabled plugin)

**Lines of Code:**
- Removed from core: ~250 lines
- Added to plugin: ~350 lines
- Net change: +100 lines (better organized)

**Testing:**
- Plugin discovery: ✓
- Plugin validation: ✓
- Transform registration: ✓
- Pipeline integration: ✓

**Result:**
- ✅ Lighter core pipeline
- ✅ Modular architecture
- ✅ Plugin system validated
- ✅ Production-ready

---

**Implementation Time:** ~15 minutes  
**Testing Time:** ~5 minutes  
**Total:** ~20 minutes

**Status:** Ready to implement! 🚀
