# Custom Transforms Quick Reference
# Phase 3 - Step 3.1.1

## Creating Your First Custom Transform

### 1. Basic Template

```python
from vqm24_pipeline.transformations.custom_transforms import (
    CustomTransformBase,
    TransformMetadata
)
import torch
from torch_geometric.data import Data

class MyCustomTransform(CustomTransformBase):
    """Brief description of your transform."""
    
    def __init__(self, my_param: float = 1.0):
        super().__init__()
        self.my_param = my_param
    
    def transform(self, data: Data) -> Data:
        """Apply transformation logic."""
        # Your transformation code here
        return data
    
    @classmethod
    def get_metadata(cls):
        return TransformMetadata(
            name="MyCustomTransform",
            version="1.0.0",
            author="Your Name",
            category="molecular",  # or "quantum", "experimental", "augmentation"
            description="What this transform does"
        )
```

### 2. Molecular Transform Template

```python
from vqm24_pipeline.transformations.custom_transforms import MolecularTransformBase

class MyMolecularTransform(MolecularTransformBase):
    """Chemistry-aware transform."""
    
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
    
    def transform(self, data: Data) -> Data:
        # Validate molecular structure
        is_valid, issues = self.validate_molecular_structure(data)
        if not is_valid:
            self._logger.warning(f"Validation issues: {issues}")
        
        # Your transformation
        return data
    
    @classmethod
    def get_metadata(cls):
        return TransformMetadata(
            name="MyMolecularTransform",
            version="1.0.0",
            author="Your Name",
            category="molecular",
            description="Molecular transformation",
            required_node_features=["x"]  # Requires atomic numbers
        )
```

### 3. Quantum Transform Template (VQM24-Specific)

```python
from vqm24_pipeline.transformations.custom_transforms import QuantumTransformBase

class MyQuantumTransform(QuantumTransformBase):
    """VQM24 quantum property transform."""
    
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
    
    def transform(self, data: Data) -> Data:
        # Validate quantum properties
        is_valid, issues = self.validate_quantum_properties(data)
        if not is_valid:
            self._logger.warning(f"Quantum validation issues: {issues}")
        
        # Transform quantum properties
        if hasattr(data, 'energy'):
            data.energy = data.energy * self.scale
        
        return data
    
    @classmethod
    def get_metadata(cls):
        return TransformMetadata(
            name="MyQuantumTransform",
            version="1.0.0",
            author="Your Name",
            category="quantum",
            description="Scale quantum energies",
            validated_datasets=["VQM24_DFT", "VQM24_DMC"],
            required_graph_attributes=["energy"]
        )
```

---

## Adding Parameter Constraints

```python
@classmethod
def get_parameter_constraints(cls):
    return {
        'threshold': {
            'type': float,
            'range': (0.0, 1.0),
            'default': 0.5,
            'description': 'Threshold for filtering'
        },
        'k_neighbors': {
            'type': int,
            'min': 1,
            'max': 100,
            'default': 10,
            'description': 'Number of neighbors'
        },
        'method': {
            'type': str,
            'choices': ['average', 'max', 'sum'],
            'default': 'average',
            'description': 'Aggregation method'
        }
    }
```

---

## Adding Custom Validation

### Input Validation

```python
def validate_input(self, data: Data):
    """Validate input data before transformation."""
    if not hasattr(data, 'x'):
        raise TransformValidationError(
            "Missing required node features",
            transform_name=self._metadata.name
        )
    
    if data.num_nodes < 3:
        raise TransformValidationError(
            "Too few atoms for this transform",
            transform_name=self._metadata.name
        )
```

### Output Validation

```python
def validate_output(self, data: Data):
    """Validate output data after transformation."""
    if not torch.all(torch.isfinite(data.x)):
        raise TransformExecutionError(
            "Transform produced non-finite values",
            transform_name=self._metadata.name
        )
```

---

## Using Your Transform

### 1. Direct Usage

```python
from my_transforms import MyCustomTransform
from torch_geometric.data import Data
import torch

# Create transform
transform = MyCustomTransform(my_param=2.0)

# Apply to data
data = Data(x=torch.ones(5, 3))
result = transform(data)

# Check statistics
stats = transform.get_usage_statistics()
print(f"Calls: {stats['call_count']}, Success rate: {stats['success_rate']:.2%}")
```

### 2. In Configuration (After Step 3.1.2)

```yaml
# config.yaml
transformations:
  experimental_setups:
    my_experiment:
      - {name: "AddSelfLoops"}
      - {name: "MyCustomTransform", my_param: 2.0}
      - {name: "GCNNorm"}
```

### 3. Programmatic Registration (After Step 3.1.2)

```python
from vqm24_pipeline.transformations.graph_transforms import register_custom_transforms
import my_transforms

# Register all transforms in module
stats = register_custom_transforms(my_transforms)
print(f"Registered {stats['total']} transforms: {stats['registered']}")

# Or register single transform
stats = register_custom_transforms(my_transforms.MyCustomTransform)
```

---

## Built-in Example Transforms

### 1. NormalizeVibrationalModes

```python
from vqm24_pipeline.transformations.custom_transforms import NormalizeVibrationalModes

# Normalize each mode independently
transform = NormalizeVibrationalModes(normalize_per_mode=True, epsilon=1e-8)

# Use in pipeline
data = Data(
    x=torch.ones(3, 1),
    vibmodes=torch.randn(5, 3, 3),  # 5 modes, 3 atoms, 3D
    num_nodes=3
)
result = transform(data)
```

**Use Cases:**
- Training stability improvement
- Cross-molecule vibrational mode comparison
- Ablation studies

### 2. FilterByDMCUncertainty

```python
from vqm24_pipeline.transformations.custom_transforms import FilterByDMCUncertainty

# Filter high-uncertainty samples
transform = FilterByDMCUncertainty(max_uncertainty=0.1, remove=False)

data = Data(
    x=torch.ones(3, 1),
    dmc_uncertainty=torch.tensor(0.05),
    num_nodes=3
)
result = transform(data)
assert result.is_high_uncertainty == False  # Low uncertainty, passed
```

**Parameters:**
- `max_uncertainty`: Threshold in kcal/mol (default: 0.1)
- `remove`: If True, return None for filtered samples; if False, mark with flag

**Use Cases:**
- Clean training data
- Create difficulty-stratified datasets
- Data quality ablation studies

### 3. ScaleMullikenCharges

```python
from vqm24_pipeline.transformations.custom_transforms import ScaleMullikenCharges

# Scale and center charges
transform = ScaleMullikenCharges(scale_factor=2.0, center=True)

data = Data(
    x=torch.ones(3, 1),
    charges=torch.tensor([0.5, -0.3, -0.2]),
    num_nodes=3
)
result = transform(data)
# Charges are centered (mean=0) then scaled by 2.0
```

**Use Cases:**
- Charge magnitude ablation studies
- Model sensitivity testing
- Feature normalization experiments

---

## Advanced Patterns

### 1. Filter Transform (Return None)

```python
def transform(self, data: Data) -> Optional[Data]:
    """Filter out samples that don't meet criteria."""
    if data.num_nodes < self.min_atoms:
        return None  # Signal to skip this sample
    return data
```

### 2. Multi-Stage Transform

```python
def transform(self, data: Data) -> Data:
    """Apply multiple transformations in sequence."""
    # Stage 1: Normalize
    data = self._normalize(data)
    
    # Stage 2: Add features
    data = self._add_features(data)
    
    # Stage 3: Validate
    is_valid, _ = self.validate_molecular_structure(data)
    if not is_valid:
        raise TransformExecutionError("Invalid output")
    
    return data
```

### 3. Conditional Transform

```python
def transform(self, data: Data) -> Data:
    """Apply different logic based on data properties."""
    if hasattr(data, 'is_periodic'):
        return self._transform_periodic(data)
    else:
        return self._transform_molecular(data)
```

### 4. Stateful Transform with Caching

```python
def __init__(self, cache_size: int = 100):
    super().__init__()
    self._cache = {}
    self._cache_size = cache_size

def transform(self, data: Data) -> Data:
    """Use caching for expensive computations."""
    # Create cache key
    cache_key = hash((data.num_nodes, data.num_edges))
    
    if cache_key in self._cache:
        self._logger.debug("Cache hit")
        result = self._cache[cache_key]
    else:
        result = self._expensive_computation(data)
        
        # Maintain cache size
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        
        self._cache[cache_key] = result
    
    return data
```

---

## Validation Examples

### Compatibility Check

```python
transform = MyCustomTransform()
data = Data(x=torch.ones(5, 3))

# Check compatibility
is_compatible, warnings = transform.validate_compatibility(
    data, 
    validation_level="strict"
)

if not is_compatible:
    print(f"Incompatible: {warnings}")
elif warnings:
    print(f"Warnings: {warnings}")
```

### Manual Structure Validation

```python
# For molecular transforms
transform = MyMolecularTransform()
is_valid, issues = transform.validate_molecular_structure(data)

# For quantum transforms
transform = MyQuantumTransform()
is_valid, issues = transform.validate_quantum_properties(data)
```

---

## Error Handling

### Graceful Error Handling

```python
from vqm24_pipeline.exceptions import TransformExecutionError

try:
    result = transform(data)
except TransformExecutionError as e:
    print(f"Transform failed: {e}")
    print(f"Transform name: {e.transform_name}")
    print(f"Original error: {e.original_error}")
```

### Custom Error Messages

```python
def transform(self, data: Data) -> Data:
    if data.num_nodes > self.max_atoms:
        raise TransformExecutionError(
            f"Molecule too large: {data.num_nodes} > {self.max_atoms}",
            transform_name=self._metadata.name
        )
    return data
```

---

## Testing Your Transform

### Basic Unit Test

```python
import pytest
import torch
from torch_geometric.data import Data

def test_my_transform():
    """Test basic functionality."""
    transform = MyCustomTransform(my_param=2.0)
    
    data = Data(x=torch.ones(5, 3))
    result = transform(data)
    
    assert result is not None
    assert result.num_nodes == 5
    
    # Check statistics
    stats = transform.get_usage_statistics()
    assert stats['call_count'] == 1
    assert stats['error_count'] == 0
```

### Test with Invalid Data

```python
def test_my_transform_invalid_data():
    """Test error handling."""
    transform = MyCustomTransform()
    
    # Missing required attribute
    data = Data(edge_index=torch.zeros((2, 0)))
    
    with pytest.raises(TransformExecutionError):
        transform(data)
```

### Test Parameter Constraints

```python
def test_parameter_constraints():
    """Test parameter validation."""
    constraints = MyCustomTransform.get_parameter_constraints()
    
    assert 'my_param' in constraints
    assert constraints['my_param']['type'] == float
    assert constraints['my_param']['range'] == (0.0, 10.0)
```

---

## Common Pitfalls

### ❌ DON'T: Modify input in-place

```python
def transform(self, data: Data) -> Data:
    data.x *= 2.0  # Modifies original!
    return data
```

### ✅ DO: Clone or create new tensor

```python
def transform(self, data: Data) -> Data:
    data = data.clone()  # Safe copy
    data.x = data.x * 2.0
    return data
```

### ❌ DON'T: Ignore validation failures

```python
def transform(self, data: Data) -> Data:
    # Just apply transform without checking
    return data
```

### ✅ DO: Check and log validation issues

```python
def transform(self, data: Data) -> Data:
    is_valid, issues = self.validate_molecular_structure(data)
    if not is_valid:
        self._logger.warning(f"Validation issues: {issues}")
    return data
```

### ❌ DON'T: Forget to call super().__init__()

```python
def __init__(self, my_param: float):
    self.my_param = my_param  # Missing super().__init__()!
```

### ✅ DO: Always call parent initializer

```python
def __init__(self, my_param: float):
    super().__init__()  # Required!
    self.my_param = my_param
```

---

## Best Practices

### 1. Documentation
- Always provide comprehensive docstrings
- Include usage examples in docstrings
- Document parameter constraints
- Explain use cases

### 2. Validation
- Validate inputs when strict correctness matters
- Log warnings instead of raising errors when possible
- Provide actionable error messages

### 3. Performance
- Avoid expensive operations in __init__
- Use caching for repeated computations
- Profile transforms on large datasets

### 4. Reproducibility
- Make transforms deterministic (no random state)
- Document any randomness clearly
- Use fixed random seeds when needed

### 5. Testing
- Write unit tests for each transform
- Test edge cases (empty graphs, single atoms, etc.)
- Test error handling paths

---

## Resources

### Documentation
- Blueprint: `VQM24_refactoring_07_01_new_modules_ph03_Blueprint_31.txt`
- Implementation: `IMPLEMENTATION_SUMMARY_Step_3_1_1.md`
- Module code: `custom_transforms.py`

### Integration (Coming in Step 3.1.2)
- `register_custom_transforms()` function
- Enhanced `get_transform_info()`
- Enhanced `validate_comprehensive()`

### Examples
- `NormalizeVibrationalModes`: Vibrational mode normalization
- `FilterByDMCUncertainty`: Uncertainty-based filtering
- `ScaleMullikenCharges`: Charge scaling

---

**Quick Start**: Copy one of the template sections above, customize it for your needs, and start transforming! 🚀
