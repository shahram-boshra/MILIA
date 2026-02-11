# Phase 8-5: Entry Point Plugin Loading - Implementation Blueprint

## Document Information

| Field | Value |
|-------|-------|
| **Phase** | 8-5 |
| **Title** | Implement Entry Point Plugin Loading |
| **Parent Document** | MILIA_Dataset_Architecture_Refactoring_Plan_v2_2_0.md |
| **Status** | Implementation Complete |
| **Created** | Based on systematic line-by-line analysis |

---

## 1. Executive Summary

This blueprint documents the implementation of **Phase 8-5: Implement entry point plugin loading** from the MILIA Dataset Architecture Refactoring Plan. The implementation enables external packages to register dataset types with MILIA without modifying any MILIA source code, using the standard Python entry points mechanism.

### 1.1 Objectives Achieved

1. ✅ Created `plugins.py` module for entry point plugin discovery and loading
2. ✅ Updated `__init__.py` to integrate plugin system with module initialization
3. ✅ Implemented Python 3.9+ and 3.10+ compatible `importlib.metadata` API usage
4. ✅ Provided comprehensive validation for external plugin classes
5. ✅ Maintained full backward compatibility with existing code
6. ✅ Added diagnostics and information functions for plugin system

### 1.2 Files Affected

| File | Action | Lines |
|------|--------|-------|
| `milia_pipeline/datasets/plugins.py` | **CREATE** | 320 |
| `milia_pipeline/datasets/__init__.py` | **UPDATE** | 526 (was 508) |

---

## 2. Prerequisites Analysis

### 2.1 Files Analyzed

The following files were systematically analyzed line-by-line before implementation:

| File | Purpose | Key Findings |
|------|---------|--------------|
| `MILIA_Dataset_Architecture_Refactoring_Plan_v2_2_0.md` | Refactoring plan | Phase 8-5 requirements at lines 1821, 1105-1178 |
| `MILIA_Pipeline_Project_Structure.md` | Project structure | Module architecture at lines 56-299 |
| `milia_pipeline/datasets/__init__.py` | Module init | Placeholder `initialize_plugins()` at lines 477-487 |
| `milia_pipeline/datasets/registry.py` | Dataset registry | Registry API: `get_default_registry()`, `register()` |
| `milia_pipeline/datasets/base.py` | Base classes | `BaseDataset` ABC for validation |
| `milia_pipeline/exceptions.py` | Exceptions | `PluginLoadError`, `PluginValidationError` at lines 2816-2890 |

### 2.2 Design Decisions from Refactoring Plan

From Section 2.3.6 (lines 1105-1178) of the refactoring plan:

1. **Entry Point Group Name**: `milia.datasets`
2. **API Style**: Python 3.10+ `importlib.metadata.entry_points()` API
3. **Validation**: Plugin classes must be `BaseDataset` subclasses
4. **Registration**: Use global `DatasetRegistry` via `get_default_registry()`
5. **Error Handling**: Log errors, continue loading other plugins
6. **Return Value**: List of (name, class) tuples for loaded plugins

---

## 3. Implementation Details

### 3.1 New File: `plugins.py`

**Location**: `milia_pipeline/datasets/plugins.py`

**Purpose**: Implement Python entry point plugin discovery and loading for external dataset types.

#### 3.1.1 Module Constants

```python
ENTRY_POINT_GROUP: str = "milia.datasets"
__version__: str = "1.0.0"
```

#### 3.1.2 Core Functions

| Function | Signature | Purpose |
|----------|-----------|---------|
| `_get_entry_points` | `(group: str) -> List[Any]` | Get entry points with Python 3.9/3.10+ compatibility |
| `_validate_plugin_class` | `(entry_point_name: str, loaded_class: Any) -> Optional[str]` | Validate loaded class is proper BaseDataset |
| `load_dataset_plugins` | `() -> List[Tuple[str, Type[BaseDataset]]]` | Main function: discover, load, validate, register |
| `discover_and_load_plugins` | `() -> int` | Convenience wrapper returning count only |
| `get_plugin_info` | `() -> Dict[str, Any]` | Get plugin system diagnostics |
| `list_available_plugins` | `() -> List[str]` | List discovered (not loaded) plugin names |

#### 3.1.3 Python Version Compatibility

The `_get_entry_points()` function handles multiple API styles:

```python
def _get_entry_points(group: str) -> List[Any]:
    from importlib.metadata import entry_points
    eps = entry_points()
    
    # Python 3.10+ with SelectableGroups.select()
    if hasattr(eps, 'select'):
        return list(eps.select(group=group))
    
    # Python 3.9 with SelectableGroups.get()
    elif hasattr(eps, 'get'):
        return list(eps.get(group, []))
    
    # Fallback: entry_points(group=...) parameter
    else:
        return list(entry_points(group=group))
```

#### 3.1.4 Plugin Validation

The `_validate_plugin_class()` function performs these checks:

1. **Is a class**: `isinstance(loaded_class, type)`
2. **Is BaseDataset subclass**: `issubclass(loaded_class, BaseDataset)`
3. **Is not abstract**: No `__abstractmethods__`
4. **Has required attributes**: `metadata`, `schema`, `features`, `config_key`

#### 3.1.5 Main Loading Function

```python
def load_dataset_plugins() -> List[Tuple[str, Type[BaseDataset]]]:
    registry = get_default_registry()
    loaded = []
    
    eps = _get_entry_points(ENTRY_POINT_GROUP)
    
    for ep in eps:
        try:
            # Load entry point
            dataset_class = ep.load()
            
            # Validate
            error = _validate_plugin_class(ep.name, dataset_class)
            if error:
                logger.warning(error)
                continue
            
            # Register
            registry.register(dataset_class)
            loaded.append((ep.name, dataset_class))
            
        except Exception as e:
            logger.error(f"Failed to load plugin '{ep.name}': {e}")
    
    return loaded
```

### 3.2 Updated File: `__init__.py`

**Location**: `milia_pipeline/datasets/__init__.py`

#### 3.2.1 Version Update

```python
# Before
__version__ = "1.3.0"

# After
__version__ = "1.4.0"
```

#### 3.2.2 New Imports

```python
# PHASE 8 ADDITIONS - Plugin Support
from milia_pipeline.datasets.plugins import (
    load_dataset_plugins,
    discover_and_load_plugins,
    get_plugin_info,
    list_available_plugins,
    ENTRY_POINT_GROUP,
)
```

#### 3.2.3 Expanded `__all__`

Added to `__all__`:
- `"initialize_plugins"` (was already listed but now implemented)
- `"load_dataset_plugins"`
- `"discover_and_load_plugins"`
- `"get_plugin_info"`
- `"list_available_plugins"`
- `"ENTRY_POINT_GROUP"`

#### 3.2.4 New Constants

```python
PHASE_8_PLUGIN_VERSION = "8.0.0"
"""Phase 8 entry point plugin loading version."""
```

#### 3.2.5 Updated `initialize_plugins()` Function

**Before** (placeholder):
```python
def initialize_plugins(load_external: bool = True) -> int:
    """Initialize dataset plugins (placeholder for Phase 8)."""
    return 0
```

**After** (fully implemented):
```python
def initialize_plugins(load_external: bool = True) -> int:
    """
    Initialize dataset plugins.
    
    This function loads external dataset plugins via Python entry points.
    External packages can register datasets by declaring the 'milia.datasets'
    entry point group in their pyproject.toml.
    
    Args:
        load_external: Whether to load external plugins via entry points.
        
    Returns:
        Number of external plugins successfully loaded and registered
    """
    if not load_external:
        logger.debug("External plugin loading disabled")
        return 0
    
    logger.info("Initializing external dataset plugins...")
    
    try:
        count = discover_and_load_plugins()
        if count > 0:
            logger.info(f"Successfully loaded {count} external dataset plugin(s)")
        return count
    except Exception as e:
        logger.error(f"Error during plugin initialization: {e}")
        return 0
```

#### 3.2.6 Updated Module Information Functions

**`_initialize_module()`** - Added plugin status logging:
```python
# Phase 8: Log plugin system status
available_plugins = list_available_plugins()
if available_plugins:
    logger.debug(f"Phase 8 plugin system: {len(available_plugins)} external plugin(s) available")
```

**`get_module_info()`** - Added plugin info:
```python
return {
    # ... existing fields ...
    "phase_8_plugin_version": PHASE_8_PLUGIN_VERSION,
    "phase_8_plugins": get_plugin_info(),
}
```

**`check_dependencies()`** - Added importlib check:
```python
# importlib.metadata (for plugins)
try:
    from importlib.metadata import entry_points
    dependencies['importlib_metadata'] = True
except ImportError:
    dependencies['importlib_metadata'] = False

# Phase 8: Plugin support
dependencies['phase_8_plugin_support'] = True
```

---

## 4. Usage Guide

### 4.1 For External Plugin Developers

#### 4.1.1 Step 1: Create Dataset Class

Create a `BaseDataset` subclass with all required attributes and methods:

```python
# my_package/datasets.py

from milia_pipeline.datasets import (
    BaseDataset,
    DatasetMetadata,
    DatasetSchema,
    DatasetFeatures,
)
from typing import Dict, List


class QM9Dataset(BaseDataset):
    """QM9 dataset - 134k molecules with quantum chemical properties."""
    
    metadata = DatasetMetadata(
        name="QM9",
        version="1.0.0",
        description="QM9 dataset with 19 regression targets",
        author="Ramakrishnan et al.",
        license="CC0",
    )
    
    schema = DatasetSchema(
        required_properties=('atoms', 'coordinates', 'homo', 'lumo', 'gap'),
        optional_properties=('zpve', 'u0', 'u298', 'h298', 'g298', 'cv'),
        identifier_keys=(('smiles', 'smiles'), ('inchi', 'inchi')),
        coordinate_units='angstrom',
        energy_units='eV',
    )
    
    features = DatasetFeatures(
        vibrational_analysis=False,
        uncertainty_handling=False,
        atomization_energy=True,
        rotational_constants=True,
        frequency_analysis=False,
        orbital_analysis=True,
    )
    
    config_key = "qm9_config"
    
    @classmethod
    def get_required_properties(cls) -> List[str]:
        return list(cls.schema.required_properties)
    
    @classmethod
    def get_feature_support(cls) -> Dict[str, bool]:
        return cls.features.to_dict()
    
    @classmethod
    def get_molecule_creation_strategy(cls) -> str:
        return 'identifier_coordinate_based'
```

#### 4.1.2 Step 2: Configure Entry Point

Add to your package's `pyproject.toml`:

```toml
[project]
name = "qm9-plugin"
version = "1.0.0"

[project.entry-points."milia.datasets"]
qm9 = "my_package.datasets:QM9Dataset"
```

Or for `setup.py`:

```python
setup(
    name="qm9-plugin",
    entry_points={
        "milia.datasets": [
            "qm9 = my_package.datasets:QM9Dataset",
        ],
    },
)
```

#### 4.1.3 Step 3: Install and Verify

```bash
# Install your package
pip install -e .

# Verify plugin is discovered
python -c "from milia_pipeline.datasets import list_available_plugins; print(list_available_plugins())"
# Output: ['qm9']
```

### 4.2 For MILIA Users

#### 4.2.1 Loading External Plugins

```python
from milia_pipeline.datasets import initialize_plugins, list_all

# Load external plugins
count = initialize_plugins(load_external=True)
print(f"Loaded {count} external plugins")

# See all registered datasets (built-in + plugins)
print(f"All datasets: {list_all()}")
# Output: ['DFT', 'DMC', 'Wavefunction', 'QM9']
```

#### 4.2.2 Using a Plugin Dataset

```python
from milia_pipeline.datasets import get, initialize_plugins

# Load plugins first
initialize_plugins()

# Get the plugin dataset class
QM9Dataset = get('QM9')

# Use it like any other dataset
print(f"Required properties: {QM9Dataset.get_required_properties()}")
print(f"Features: {QM9Dataset.get_feature_support()}")
```

#### 4.2.3 Checking Plugin Status

```python
from milia_pipeline.datasets import get_plugin_info, list_available_plugins

# List available plugins (before loading)
available = list_available_plugins()
print(f"Available plugins: {available}")

# Get detailed plugin info
info = get_plugin_info()
print(f"Entry point group: {info['entry_point_group']}")
print(f"Discovered count: {info['discovered_count']}")
print(f"Python version: {info['python_version']}")
```

---

## 5. Integration with Existing Architecture

### 5.1 Module Dependencies

```
milia_pipeline/datasets/plugins.py
    ├── imports from: base.py (BaseDataset)
    ├── imports from: registry.py (get_default_registry)
    └── imports from: ../exceptions.py (PluginLoadError, PluginValidationError)

milia_pipeline/datasets/__init__.py
    ├── imports from: plugins.py (all plugin functions)
    ├── imports from: base.py (BaseDataset, DatasetMetadata, etc.)
    ├── imports from: registry.py (registry functions)
    └── imports from: implementations/ (DFT, DMC, Wavefunction)
```

### 5.2 Initialization Sequence

```
1. Module import: milia_pipeline.datasets
   │
   ├── 2. Import base.py (BaseDataset, metadata classes)
   │
   ├── 3. Import registry.py (DatasetRegistry, convenience functions)
   │       └── Creates _default_registry instance
   │
   ├── 4. Import implementations/ (DFT, DMC, Wavefunction)
   │       └── @register decorators execute → built-in datasets registered
   │
   ├── 5. Import plugins.py (plugin loading functions)
   │
   └── 6. _initialize_module() executes
           ├── Logs module info
           ├── Checks dependencies
           └── Lists available plugins (does NOT auto-load)
```

### 5.3 Plugin Loading Sequence

```
1. User calls: initialize_plugins(load_external=True)
   │
   ├── 2. discover_and_load_plugins() called
   │       │
   │       └── 3. load_dataset_plugins() called
   │               │
   │               ├── 4. _get_entry_points("milia.datasets")
   │               │       └── Returns list of entry point objects
   │               │
   │               ├── 5. For each entry point:
   │               │       ├── ep.load() → imports module, gets class
   │               │       ├── _validate_plugin_class() → validates
   │               │       └── registry.register() → adds to registry
   │               │
   │               └── 6. Returns list of (name, class) tuples
   │
   └── 7. Returns count of loaded plugins
```

---

## 6. Error Handling

### 6.1 Error Scenarios

| Scenario | Handling | Log Level |
|----------|----------|-----------|
| Entry point module import fails | Log error, skip plugin, continue | ERROR |
| Loaded object is not a class | Log warning, skip plugin, continue | WARNING |
| Class is not BaseDataset subclass | Log warning, skip plugin, continue | WARNING |
| Class is abstract (has abstract methods) | Log warning, skip plugin, continue | WARNING |
| Class missing required attributes | Log warning, skip plugin, continue | WARNING |
| Dataset name already registered | Log warning, skip plugin, continue | WARNING |
| Registry registration fails | Log error, skip plugin, continue | ERROR |

### 6.2 Error Messages

All error messages include:
- Entry point name
- Class name (if available)
- Specific validation failure reason
- Full traceback at DEBUG level

Example:
```
ERROR: Failed to load dataset plugin 'broken_plugin': ModuleNotFoundError: No module named 'missing_dep'
WARNING: Entry point 'bad_class' class 'NotADataset' is not a BaseDataset subclass
```

---

## 7. Testing Recommendations

### 7.1 Unit Tests for `plugins.py`

```python
# tests/datasets/test_plugins.py

import pytest
from unittest.mock import Mock, patch, MagicMock
from milia_pipeline.datasets.plugins import (
    _get_entry_points,
    _validate_plugin_class,
    load_dataset_plugins,
    discover_and_load_plugins,
    get_plugin_info,
    list_available_plugins,
    ENTRY_POINT_GROUP,
)
from milia_pipeline.datasets.base import BaseDataset


class TestGetEntryPoints:
    """Tests for _get_entry_points function."""
    
    def test_returns_empty_list_when_no_plugins(self):
        """Should return empty list when no plugins registered."""
        with patch('milia_pipeline.datasets.plugins.entry_points') as mock_eps:
            mock_eps.return_value = MagicMock(select=lambda group: [])
            result = _get_entry_points(ENTRY_POINT_GROUP)
            assert result == []
    
    def test_handles_python_310_api(self):
        """Should handle Python 3.10+ SelectableGroups.select() API."""
        mock_ep = Mock(name='test_plugin')
        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]
        
        with patch('milia_pipeline.datasets.plugins.entry_points', return_value=mock_eps):
            result = _get_entry_points(ENTRY_POINT_GROUP)
            assert len(result) == 1
            mock_eps.select.assert_called_once_with(group=ENTRY_POINT_GROUP)


class TestValidatePluginClass:
    """Tests for _validate_plugin_class function."""
    
    def test_rejects_non_class(self):
        """Should reject objects that are not classes."""
        error = _validate_plugin_class('test', "not a class")
        assert error is not None
        assert "did not provide a class" in error
    
    def test_rejects_non_base_dataset(self):
        """Should reject classes that don't inherit from BaseDataset."""
        class NotADataset:
            pass
        
        error = _validate_plugin_class('test', NotADataset)
        assert error is not None
        assert "is not a BaseDataset subclass" in error
    
    def test_accepts_valid_dataset(self):
        """Should accept valid BaseDataset subclasses."""
        # Use an existing valid dataset class
        from milia_pipeline.datasets import DFTDataset
        
        error = _validate_plugin_class('test', DFTDataset)
        assert error is None


class TestLoadDatasetPlugins:
    """Tests for load_dataset_plugins function."""
    
    def test_returns_empty_list_when_no_plugins(self):
        """Should return empty list when no plugins discovered."""
        with patch('milia_pipeline.datasets.plugins._get_entry_points', return_value=[]):
            result = load_dataset_plugins()
            assert result == []
    
    def test_handles_load_error_gracefully(self):
        """Should log error and continue when plugin fails to load."""
        mock_ep = Mock()
        mock_ep.name = 'broken_plugin'
        mock_ep.load.side_effect = ImportError("Module not found")
        
        with patch('milia_pipeline.datasets.plugins._get_entry_points', return_value=[mock_ep]):
            result = load_dataset_plugins()
            assert result == []  # Should not raise, returns empty list


class TestDiscoverAndLoadPlugins:
    """Tests for discover_and_load_plugins function."""
    
    def test_returns_count(self):
        """Should return count of loaded plugins."""
        with patch('milia_pipeline.datasets.plugins.load_dataset_plugins', return_value=[]):
            count = discover_and_load_plugins()
            assert count == 0


class TestGetPluginInfo:
    """Tests for get_plugin_info function."""
    
    def test_returns_expected_keys(self):
        """Should return dict with expected keys."""
        info = get_plugin_info()
        assert 'version' in info
        assert 'entry_point_group' in info
        assert 'discovered_plugins' in info
        assert 'python_version' in info
        assert 'api_style' in info
    
    def test_entry_point_group_is_correct(self):
        """Should return correct entry point group name."""
        info = get_plugin_info()
        assert info['entry_point_group'] == 'milia.datasets'


class TestListAvailablePlugins:
    """Tests for list_available_plugins function."""
    
    def test_returns_list_of_strings(self):
        """Should return list of plugin names as strings."""
        result = list_available_plugins()
        assert isinstance(result, list)
        assert all(isinstance(name, str) for name in result)
```

### 7.2 Integration Test

```python
# tests/datasets/test_plugin_integration.py

import pytest
from milia_pipeline.datasets import (
    initialize_plugins,
    list_all,
    get,
    is_registered,
    get_default_registry,
)


class TestPluginIntegration:
    """Integration tests for plugin system with registry."""
    
    def test_initialize_plugins_is_idempotent(self):
        """Calling initialize_plugins multiple times should be safe."""
        count1 = initialize_plugins(load_external=True)
        count2 = initialize_plugins(load_external=True)
        # Second call should not fail
        assert count2 >= 0
    
    def test_builtin_datasets_always_registered(self):
        """Built-in datasets should be registered regardless of plugins."""
        assert is_registered('DFT')
        assert is_registered('DMC')
        assert is_registered('Wavefunction')
    
    def test_disabled_plugin_loading(self):
        """Should not load plugins when load_external=False."""
        registry = get_default_registry()
        initial_count = len(registry)
        
        count = initialize_plugins(load_external=False)
        
        assert count == 0
        assert len(registry) == initial_count
```

---

## 8. Validation Checklist

### 8.1 Implementation Validation

- [x] `plugins.py` created with all required functions
- [x] `__init__.py` updated with plugin imports
- [x] `initialize_plugins()` fully implemented (was placeholder)
- [x] Version bumped to 1.4.0
- [x] New constants added (`PHASE_8_PLUGIN_VERSION`)
- [x] `__all__` updated with plugin exports
- [x] Module info functions updated

### 8.2 Compatibility Validation

- [x] Python 3.9 API supported (`entry_points().get()`)
- [x] Python 3.10+ API supported (`entry_points().select()`)
- [x] Backward compatible (no breaking changes)
- [x] Built-in datasets unaffected

### 8.3 Documentation Validation

- [x] Module docstrings complete
- [x] Function docstrings with examples
- [x] Type hints on all functions
- [x] Usage examples provided

---

## 9. Files for Download

### 9.1 New File: `plugins.py`

**Full path**: `milia_pipeline/datasets/plugins.py`

**Size**: ~320 lines

**Contents**: Entry point plugin discovery, loading, validation, and registration.

### 9.2 Updated File: `__init__.py`

**Full path**: `milia_pipeline/datasets/__init__.py`

**Size**: ~526 lines

**Changes**: 
- Version 1.3.0 → 1.4.0
- New imports from `plugins.py`
- Expanded `__all__`
- New `PHASE_8_PLUGIN_VERSION` constant
- Updated `_initialize_module()` for plugin logging
- Fully implemented `initialize_plugins()`
- Updated `get_module_info()` and `check_dependencies()`

---

## 10. Next Steps (Phase 8-6, 8-7, 8-8)

According to the refactoring plan (lines 1822-1824), the remaining Phase 8 tasks are:

| Task | Description | Status |
|------|-------------|--------|
| 8-6 | Write migration guide for existing code | Pending |
| 8-7 | Update all documentation | Pending |
| 8-8 | Verify cache invalidation works correctly | Pending |

---

## Document End

**Implementation Status**: ✅ Complete

**Files Created**:
1. `milia_pipeline/datasets/plugins.py` (NEW)
2. `milia_pipeline/datasets/__init__.py` (UPDATED)

**Blueprint Version**: 1.0.0
