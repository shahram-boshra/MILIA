# Example Descriptors Plugin

This plugin demonstrates best practices for creating custom descriptor plugins for VQM24.

## Descriptors

### AromaticRatio
- **Category**: Constitutional
- **Description**: Ratio of aromatic atoms to total atoms
- **Range**: 0.0 to 1.0
- **Use Case**: Measuring aromaticity

### HeteroatomRatio
- **Category**: Constitutional
- **Description**: Ratio of heteroatoms (non-C, non-H) to total atoms
- **Range**: 0.0 to 1.0
- **Use Case**: Assessing functional group content

### ChainLength
- **Category**: Topological
- **Description**: Length of longest carbon chain
- **Use Case**: Distinguishing linear from branched structures

## Usage
```python
from rdkit import Chem
from vqm24_pipeline.descriptors.descriptor_plugin_system import discover_plugins
from vqm24_pipeline.descriptors.descriptor_registry import registry

# Discover plugins
discover_plugins()

# Get descriptor function
calc_aromatic = registry.get_descriptor("AromaticRatio")

# Calculate descriptor
mol = Chem.MolFromSmiles("c1ccccc1")  # Benzene
value = calc_aromatic(mol)
print(f"Aromatic Ratio: {value}")  # 1.0
```

## Testing
```python
import pytest
from rdkit import Chem

def test_aromatic_ratio():
    from vqm24_pipeline.plugins.descriptors.example_descriptors.descriptors import (
        calculate_aromatic_ratio
    )

    # Benzene - all aromatic
    mol = Chem.MolFromSmiles("c1ccccc1")
    assert calculate_aromatic_ratio(mol) == 1.0

    # Ethanol - no aromatic
    mol = Chem.MolFromSmiles("CCO")
    assert calculate_aromatic_ratio(mol) == 0.0
```

## Notes

- All descriptors handle errors gracefully (return NaN)
- Descriptors are well-documented
- Examples provided for each descriptor
- `calculate_ring_complexity` is a bonus descriptor (not in plugin.yaml)
