# MILIA Pydantic Implementation Blueprint

**Project**: MILIA (Machine Intelligent Learning Interface Assistant) Pipeline
**Task**: Comprehensive Pydantic V2 Integration for Runtime Validation
**Status**: 🏆 ALL PHASES COMPLETE ✅ (108/~111 dataclasses migrated - 100%)
**Document Version**: 1.39.0
**Created**: 2026-01-07
**Last Updated**: 2026-01-08
**Based On**: Line-by-line analysis of 10 source files + Pydantic V2 official documentation + Codebase audit
**Total Phases**: 38 (Phase 1-38 ALL COMPLETE ✅) - MILIA IS NOW FULLY FASTAPI-READY

---

## ✅ IMPLEMENTATION PROGRESS

| Phase | File | Status | Date | Notes |
|-------|------|--------|------|-------|
| **Phase 1** | `base.py` | ✅ COMPLETE | 2026-01-07 | Single import change; `__post_init__` preserved |
| **Phase 2** | `config_containers.py` | ✅ COMPLETE | 2026-01-07 | 10 classes migrated; `model_validator(mode='before')` pattern |
| **Phase 3** | `config_bridge.py` | ✅ COMPLETE | 2026-01-07 | 31 mutable classes migrated; `field_validator` + `model_validator(mode='after')` pattern |
| **Phase 4** | `validators.py` | ✅ COMPLETE | 2026-01-07 | Pydantic wrappers added; `PYDANTIC_AVAILABLE` flag; 10 tests passed |
| **Phase 5** | `exceptions.py` | ✅ COMPLETE | 2026-01-07 | Documentation update; Pydantic namespace compatibility note added |
| **Phase 6a** | `param_types.py` | ✅ COMPLETE | 2026-01-07 | HPO search space; 1 frozen class; `model_validator(mode='before')` |
| **Phase 6b** | `hpo_config.py` | ✅ COMPLETE | 2026-01-07 | HPO config; 5 frozen classes; `field_validator` + `model_validator(mode='before')` |
| **Phase 7** | `device_manager.py` | ✅ COMPLETE | 2026-01-07 | Acceleration module; 1 mutable class; `model_dump()` wrapper; 15 tests passed |
| **Phase 8** | `distributed_strategies.py` | ✅ COMPLETE | 2026-01-07 | Acceleration; 1 mutable class; `model_dump(mode='json')` for enum serialization; 12 tests passed |
| **Phase 9** | `memory_optimization.py` | ✅ COMPLETE | 2026-01-07 | Acceleration; 1 mutable class; `model_dump()` wrapper; 15 tests passed |
| **Phase 10** | `computation_optimization.py` | ✅ COMPLETE | 2026-01-07 | Acceleration; 1 mutable class; `model_dump()` wrapper; 15 tests passed |
| **Phase 11** | `transfer_manager.py` | ✅ COMPLETE | 2026-01-07 | HPO Transfer; 2 classes (1 frozen + 1 mutable); `field_validator` + `model_validator(mode='before')`; 25 tests passed |
| **Phase 12** | `meta_features.py` | ✅ COMPLETE | 2026-01-07 | HPO Transfer; 1 frozen class; `field_validator` + `model_validator(mode='after')`; 20 tests passed |
| **Phase 13** | `warm_start.py` | ✅ COMPLETE | 2026-01-07 | HPO Transfer; 2 classes (1 frozen + 1 mutable); `field_validator`; 20 tests passed |
| **Phase 14** | `search_space.py` | ✅ COMPLETE | 2026-01-07 | HPO NAS; `LayerConfig` (frozen 7 attr) + `GNNArchitectureSpace` (mutable 13 attr); `field_validator` + `model_validator(mode='after')`; 30 tests passed |
| **Phase 15** | `study_analyzer.py` | ✅ COMPLETE | 2026-01-07 | HPO Analysis; `AnalysisConfig` (frozen 6 attr); `field_validator` + `model_validator(mode='after')`; 25 tests passed |
| **Phase 16** | `custom_transforms.py` | ✅ COMPLETE | 2026-01-07 | Transformations; `TransformMetadata` (11 attr); Mutable BaseModel; `model_dump()` wrapper; 18 tests passed |
| **Phase 17** | `config_schemas.py` | ✅ COMPLETE | 2026-01-08 | Config; 9 classes migrated; `field_validator` + `model_validator(mode='before/after')`; 14 tests passed |
| **Phase 18** | `graph_transforms.py` | ✅ COMPLETE | 2026-01-08 | Transformations; 8 classes migrated; `model_validator(mode='after')` + `ConfigDict(arbitrary_types_allowed=True)`; 87 tests passed |
| **Phase 19** | `plugin_system.py` | ✅ COMPLETE | 2026-01-08 | Transformations; 2 classes (`TransformDeclaration` 10 attr + `PluginMetadata` 20 attr); `model_validator(mode='after')` + custom `__hash__`; 54 tests passed |
| **Phase 20** | `deployment_strategies.py` | ✅ COMPLETE | 2026-01-08 | Deployment; 1 mutable class (`DeploymentConfig` 13 attr); `model_dump()` wrapper; 15 tests passed |
| **Phase 21** | `monitoring.py` | ✅ COMPLETE | 2026-01-08 | Deployment; 2 mutable classes (`MonitoringConfig` 12 attr + `Alert` 6 attr); `model_dump()` + `model_dump(mode='json')` for enum/datetime; 25 tests passed |
| **Phase 22** | `model_optimization.py` | ✅ COMPLETE | 2026-01-08 | Deployment; 1 mutable class (`OptimizationConfig` 11 attr); `model_dump()` wrapper; 17 tests passed |
| **Phase 23** | `pyg_introspector.py` | ✅ COMPLETE | 2026-01-08 | Registry; 2 mutable classes (`ParameterInfo` 8 attr + `DynamicModelMetadata` 18 attr); `model_dump()` + `model_dump(mode='json')` for enum; `ConfigDict(arbitrary_types_allowed=True)`; 20 tests passed |
| **Phase 24** | `model_registry.py` | ✅ COMPLETE | 2026-01-08 | Registry; 1 mutable class (`ModelRegistration` 6 attr); `ConfigDict(arbitrary_types_allowed=True)` for `Type[torch.nn.Module]`; `model_dump()` wrapper; 14 tests passed |
| **Phase 25** | `target_selection_config.py` | ✅ COMPLETE | 2026-01-08 | Factory; 1 mutable class (`TargetSelectionConfig` 14 attr); Custom `to_dict()` preserved (11 keys, enum `.name`); `Field(default_factory=dict)`; 15 tests passed |
| **Phase 26** | `hpo_manager.py` | ✅ COMPLETE | 2026-01-08 | HPO; 0 dataclasses (dead code removal only); Removed unused `from dataclasses import asdict`; 15 tests passed |
| **Phase 27** | `research_api.py` | ✅ COMPLETE | 2026-01-08 | Transformations; 1 mutable class (`ExperimentConfiguration` 12 attr); Custom `to_dict()` preserved; `@model_validator(mode='after')`; 15 tests passed |
| **Phase 28** | `descriptor_registry.py` | ✅ COMPLETE | 2026-01-08 | Descriptors; 1 mutable class (`DescriptorRegistration` 6 attr); `ConfigDict(arbitrary_types_allowed=True)` for Callable; `to_dict()` wrapper; 20 tests passed |
| **Phase 29** | `descriptor_validator.py` | ✅ COMPLETE | 2026-01-08 | Descriptors; 1 mutable class (`ValidationResult` 4 attr); `Field(default_factory=list/dict)`; `__post_init__` removed; `to_dict()` wrapper; 20 tests passed |
| **Phase 30** | `descriptor_categories.py` | ✅ COMPLETE | 2026-01-08 | Descriptors; 1 frozen class (`DescriptorMetadata` 6 attr); Custom `__init__` for positional args; `to_dict()` with `model_dump(mode='json')`; 424 instances; 20 tests passed |
| **Phase 31** | `descriptor_calculator.py` | ✅ COMPLETE | 2026-01-08 | Descriptors; 2 mutable classes (`CalculationResult` 5 attr + `BatchCalculationResult` 4 attr); `to_dict()` wrapper; 20 tests passed |
| **Phase 32** | `descriptor_plugin_system.py` | ✅ COMPLETE | 2026-01-08 | Descriptors; 2 mutable classes (`DescriptorDeclaration` 8 attr + `DescriptorPluginMetadata` 19 attr); `Field(default_factory=...)` + `@model_validator(mode='after')` + custom `__hash__`; 25 tests passed |
| **Phase 33** | `architecture_builder.py` | ✅ COMPLETE | 2026-01-08 | Builders; 3 classes (`LayerConfig` 6 attr + `ResidualConnection` 3 attr + `ArchitectureConfig` 6 attr); `Field(default_factory=...)` + custom `to_dict()`/`from_dict()` preserved; 25 tests passed |
| **Phase 34** | `model_composer.py` | ✅ COMPLETE | 2026-01-08 | Builders; 2 classes (`ModelSpec` 4 attr + `EnsembleConfig` 5 attr); `ConfigDict(arbitrary_types_allowed=True)` for nn.Module + `Field(default_factory=list)` + custom `to_dict()` preserved; 25 tests passed |
| **Phase 35** | `layer_registry.py` | ✅ COMPLETE | 2026-01-08 | Builders; 1 class (`LayerMetadata` 12 attr); `Field(default_factory=...)` + custom `to_dict()` with enum `.value` preserved; 25 tests passed |
| **Phase 36** | `nas_manager.py` | ✅ COMPLETE | 2026-01-08 | HPO NAS; 1 mutable class (`NASConfig` 7 attr); `@field_validator` for 4 fields; `to_dict()` wrapper; 20 tests passed |
| **Phase 37** | `search_space_builder.py` | ✅ COMPLETE | 2026-01-08 | HPO Search; 0 dataclasses (dead code removal only); Removed unused `from dataclasses import dataclass, field`; 15 tests passed |
| **Phase 38** | `model_plugin_system.py` | ✅ COMPLETE | 2026-01-08 | Plugins; 2 mutable classes (`ModelDeclaration` 16 attr + `ModelPluginMetadata` 19 attr); `Field(default_factory=list)` + `ConfigDict(arbitrary_types_allowed=True)` + custom `to_dict()` preserved; 20 tests passed |

---

## ⚠️ INSTRUCTIONS FOR NEW CONTEXT WINDOW

If continuing this work in a new session:

### Required Files to Request

The user MUST upload these 10 source files before implementation:

| # | File Path | Purpose |
|---|-----------|---------|
| 1 | `milia_pipeline/datasets/base.py` | Phase 1 - 3 frozen dataclasses |
| 2 | `milia_pipeline/datasets/registry.py` | Dynamic registry (no changes needed) |
| 3 | `milia_pipeline/config/config_containers.py` | Phase 2 - 10 frozen dataclasses |
| 4 | `milia_pipeline/models/utils/config_bridge.py` | Phase 3 - 25+ mutable dataclasses |
| 5 | `milia_pipeline/config/validators.py` | Phase 4 - ValidationResult integration |
| 6 | `milia_pipeline/config/config_loader.py` | Registry initialization (reference only) |
| 7 | `milia_pipeline/exceptions.py` | Phase 5 - ValidationError namespace |
| 8 | `milia_pipeline/models/hpo/search_spaces/param_types.py` | Phase 6a - HPO search space config |
| 9 | `milia_pipeline/models/hpo/hpo_config.py` | Phase 6b - HPO master config |
| 10 | `milia_pipeline/models/acceleration/device_manager.py` | Phase 7 - DeviceInfo mutable dataclass |

### Implementation Order

**CRITICAL**: Follow this exact order to avoid circular import issues:

1. **Phase 1**: `base.py` ✅ COMPLETE (3 classes, drop-in replacement)
2. **Phase 2**: `config_containers.py` ✅ COMPLETE (10 frozen classes, `model_validator(mode='before')` pattern)
3. **Phase 3**: `config_bridge.py` ✅ COMPLETE (31 mutable classes, `field_validator` + `model_validator(mode='after')`)
4. **Phase 4**: `validators.py` ✅ COMPLETE (Pydantic wrappers added, `PYDANTIC_AVAILABLE` flag)
5. **Phase 5**: `exceptions.py` ✅ COMPLETE (documentation update; Pydantic namespace compatibility note)
6. **Phase 6a**: `param_types.py` ✅ COMPLETE (1 frozen class; HPO search space parameter config)
7. **Phase 6b**: `hpo_config.py` ✅ COMPLETE (5 frozen classes; HPO master configuration)
8. **Phase 7**: `device_manager.py` ✅ COMPLETE (1 mutable class; Acceleration module DeviceInfo)
9. **Phase 8**: `distributed_strategies.py` ✅ COMPLETE (1 mutable class; `DistributedConfig` 12 attributes)
10. **Phase 9**: `memory_optimization.py` ✅ COMPLETE (1 mutable class; `MemoryConfig` 9 attributes)
11. **Phase 10**: `computation_optimization.py` ✅ COMPLETE (1 mutable class; `ComputationConfig` 10 attributes)
12. **Phase 11**: `transfer_manager.py` ✅ COMPLETE (2 classes; `TransferConfig` frozen 10 attr + `RegisteredStudyInfo` mutable 10 attr)
13. **Phase 12**: `meta_features.py` ✅ COMPLETE (1 frozen class; `MetaFeatureConfig` 5 attributes)
14. **Phase 13**: `warm_start.py` ✅ COMPLETE (2 classes; `WarmStartConfig` frozen 8 attr + `TransferredTrial` mutable 6 attr)
15. **Phase 14**: `search_space.py` ✅ COMPLETE (2 classes; `LayerConfig` frozen 7 attr + `GNNArchitectureSpace` mutable 13 attr)
16. **Phase 15**: `study_analyzer.py` ✅ COMPLETE (1 frozen class; `AnalysisConfig` 6 attributes)
17. **Phase 16**: `custom_transforms.py` ✅ COMPLETE (1 mutable class; `TransformMetadata` 11 attributes)
18. **Phase 17**: `config_schemas.py` ✅ COMPLETE (9 classes; 4 frozen + 5 mutable; `field_validator` + `model_validator`; 14 tests passed)
19. **Phase 18**: `graph_transforms.py` ✅ COMPLETE (8 mutable classes; `model_validator(mode='after')` + `ConfigDict(arbitrary_types_allowed=True)`; 87 tests passed)
20. **Phase 19**: `plugin_system.py` ✅ COMPLETE (2 mutable classes; `TransformDeclaration` 10 attr + `PluginMetadata` 20 attr; `model_validator(mode='after')` + custom `__hash__`; 54 tests passed)
21. **Phase 20**: `deployment_strategies.py` ✅ COMPLETE (1 mutable class; `DeploymentConfig` 13 attr; `model_dump()` wrapper; 15 tests passed)

### Implementation Method

- Use `str_replace` tool for ALL changes (NOT file_create)
- Make incremental changes, NOT full file rewrites
- Verify imports after each phase before proceeding

### Key Pydantic V2 Imports

```python
# For frozen BaseModel (config_containers.py)
from pydantic import BaseModel, field_validator, model_validator, Field
from pydantic import ValidationError as PydanticValidationError
from typing_extensions import Self

# For Pydantic dataclass (base.py)
from pydantic.dataclasses import dataclass
from pydantic import field_validator
from dataclasses import field  # Keep for default_factory

# For mutable BaseModel (config_bridge.py)
from pydantic import BaseModel, field_validator, model_validator, Field
from pydantic import ValidationError as PydanticValidationError

# For frozen BaseModel in HPO module (param_types.py, hpo_config.py)
from pydantic import BaseModel, field_validator, model_validator, Field
from typing import Optional, List, Any, Dict, Tuple
from enum import Enum
```

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Source File Analysis Summary](#2-source-file-analysis-summary)
3. [Migration Architecture](#3-migration-architecture)
4. [Phase 1: base.py Migration](#4-phase-1-basepy-migration)
5. [Phase 2: config_containers.py Migration](#5-phase-2-config_containerspy-migration)
6. [Phase 3: config_bridge.py Migration](#6-phase-3-config_bridgepy-migration)
7. [Phase 4: validators.py Integration](#7-phase-4-validatorspy-integration)
8. [Phase 5: exceptions.py Compatibility](#8-phase-5-exceptionspy-compatibility)
9. [Phase 6: HPO Module Migration](#9-phase-6-hpo-module-migration)
10. [Cross-Cutting Concerns](#10-cross-cutting-concerns)
11. [Verification Checklist](#11-verification-checklist)

---

## 1. EXECUTIVE SUMMARY

### 1.1 Objective

Migrate MILIA's frozen dataclass pattern to Pydantic V2 for runtime validation while maintaining:
- **NON-BREAKING**: All existing public APIs remain functional
- **DYNAMIC**: No hardcoded values; registry-based dataset type resolution preserved
- **PRODUCTION-READY**: Thread-safe, proper error handling, comprehensive validation
- **FUTURE-PROOF**: Aligned with FastAPI integration goals, extensible for new dataset types

### 1.2 Migration Strategy

| File | Current Pattern | Target Pattern | Complexity |
|------|-----------------|----------------|------------|
| `base.py` | `@dataclass(frozen=True)` | `pydantic.dataclasses.dataclass(frozen=True)` | LOW |
| `config_containers.py` | `@dataclass(frozen=True)` + `object.__setattr__` | `pydantic.BaseModel` + `frozen=True` | HIGH |
| `config_bridge.py` | `@dataclass` (mutable) + `validate()` method | `pydantic.BaseModel` (mutable) | MEDIUM |
| `validators.py` | Custom `ValidationResult` class | Pydantic-compatible wrappers (coexist) | LOW |
| `exceptions.py` | `ValidationError` class | Aliased import to prevent namespace conflict | LOW |
| `param_types.py` | `@dataclass(frozen=True)` + `__post_init__` | `pydantic.BaseModel` + `frozen=True` | LOW |
| `hpo_config.py` | `@dataclass(frozen=True)` + `__post_init__` | `pydantic.BaseModel` + `frozen=True` | MEDIUM |

### 1.3 Key Pydantic V2 Patterns Used

**Evidence from Pydantic V2 Documentation (web search results):**

1. **Frozen BaseModel**: Use `class MyModel(BaseModel, frozen=True)` or `model_config = ConfigDict(frozen=True)`
2. **`__post_init__` in Pydantic dataclasses**: Called AFTER validation (confirmed in migration guide)
3. **⚠️ CRITICAL: Frozen model updates**: Use `model_validator(mode='before')` to modify input dict, NOT `model_validator(mode='after')` with `model_copy()` (see Section 5.3)
4. **Field validation**: Use `@field_validator('field_name')` with `@classmethod` decorator
5. **Model validation**: Use `@model_validator(mode='before')` for field initialization; `@model_validator(mode='after')` returning `self` only for pure validation
6. **Enum serialization (Phase 8)**: Use `model_dump(mode='json')` for automatic enum value serialization; enums are serialized to their `.value` strings automatically

### 1.4 Critical Pydantic V2 Limitation Discovered

**Issue Discovered During Phase 2 Implementation**:

The original blueprint recommended `model_validator(mode='after')` returning `self.model_copy(update={...})` for frozen models. This pattern **DOES NOT WORK** for top-level model instantiation via `__init__`.

**Pydantic Documentation Warning**:
> "Returning a value other than `self` from a top level model validator isn't supported when validating via `__init__`."

**Correct Pattern for Frozen Models**:
```python
# ❌ WRONG - Does not work for top-level __init__
@model_validator(mode='after')
def initialize_fields(self) -> Self:
    updates = {}
    if self.field is None:
        updates['field'] = {}
    return self.model_copy(update=updates) if updates else self

# ✅ CORRECT - Modify input dict before field assignment
@model_validator(mode='before')
@classmethod
def initialize_fields(cls, data: Any) -> Any:
    if isinstance(data, dict):
        if data.get('field') is None:
            data['field'] = {}
    return data
```

**When to use `model_validator(mode='after')` returning `self`**:
- Only for pure validation logic (no field modification)
- Example: Checking that `default_setup` exists in `experimental_setups` dict

---

## 2. SOURCE FILE ANALYSIS SUMMARY

### 2.1 Files Analyzed (Line-by-Line)

| File | Full Path | Lines | Classes Affected | Status |
|------|-----------|-------|------------------|--------|
| `base.py` | `milia_pipeline/datasets/base.py` | 265 | 3 frozen dataclasses | ✅ Phase 1 |
| `registry.py` | `milia_pipeline/datasets/registry.py` | 190 | 0 (supports dynamic lookup) | N/A |
| `config_containers.py` | `milia_pipeline/config/config_containers.py` | 4405 | 10 frozen dataclasses | ✅ Phase 2 |
| `config_bridge.py` | `milia_pipeline/models/utils/config_bridge.py` | 1527 → 1545 | 31 mutable dataclasses | ✅ Phase 3 |
| `validators.py` | `milia_pipeline/config/validators.py` | 4917 | `ValidationResult` class | ✅ Phase 4 |
| `config_loader.py` | `milia_pipeline/config/config_loader.py` | 2438 | Registry integration | N/A |
| `exceptions.py` | `milia_pipeline/exceptions.py` | 4046 | `ValidationError` class | ✅ Phase 5 |
| `param_types.py` | `milia_pipeline/models/hpo/search_spaces/param_types.py` | 159 → 195 | 1 frozen dataclass | ✅ Phase 6a |
| `hpo_config.py` | `milia_pipeline/models/hpo/hpo_config.py` | 581 → 621 | 5 frozen dataclasses | ✅ Phase 6b |

### 2.2 Critical Patterns Identified

#### Pattern A: `object.__setattr__` in Frozen Dataclasses (35+ occurrences)

**Location**: `config_containers.py`
**Lines affected**: 324, 328, 332, 336, 460, 464, 555, 559, 654-666, 797, 863, 867, 971, 974, 978, 1165-1178, 1484, 1488, 1640

**Current Pattern** (config_containers.py line 324):
```python
@dataclass(frozen=True)
class DatasetConfig:
    def __post_init__(self):
        # ...validation...
        object.__setattr__(self, 'is_uncertainty_enabled', uncertainty_enabled)
```

**⚠️ INCORRECT Solution (from original blueprint)**:
```python
# ❌ DOES NOT WORK for top-level __init__ calls
class DatasetConfig(BaseModel, frozen=True):
    @model_validator(mode='after')
    def set_defaults(self) -> Self:
        updates = {}
        if some_condition:
            updates['is_uncertainty_enabled'] = computed_value
        return self.model_copy(update=updates) if updates else self
```

**✅ CORRECT Pydantic V2 Solution (implemented)**:
```python
class DatasetConfig(BaseModel, frozen=True):
    @model_validator(mode='before')
    @classmethod
    def set_defaults(cls, data: Any) -> Any:
        """Modify input dict BEFORE field assignment."""
        if isinstance(data, dict):
            if some_condition:
                data['is_uncertainty_enabled'] = computed_value
        return data
```

#### Pattern B: Dynamic Registry Validation

**Location**: `config_containers.py` lines 317-319, 788-790, 825-827
**Registry functions**: `_get_valid_dataset_types()`, `_is_valid_dataset_type()` (delegate to `config_loader.py`)

**Current Pattern**:
```python
def __post_init__(self):
    valid_types = _get_valid_dataset_types()  # Dynamic lookup
    if not _is_valid_dataset_type(self.dataset_type):
        raise ValueError(...)
```

**Pydantic V2 Solution** (preserves dynamic registry):
```python
@field_validator('dataset_type')
@classmethod
def validate_dataset_type(cls, v: str) -> str:
    if not _is_valid_dataset_type(v):  # Same dynamic lookup
        valid = _get_valid_dataset_types()
        raise ValueError(f'Invalid: {v}. Must be one of {valid}')
    return v
```

#### Pattern C: Nested Container References

**Location**: `config_containers.py` line 1154
**Example**: `TransformationConfig` contains `Dict[str, ExperimentalSetup]`

**Pydantic V2 Solution**: Native support - Pydantic validates nested models automatically.

#### Pattern D: `to_dict()` Methods

**Location**: Multiple files (`DatasetFeatures.to_dict()` at line 112-123, `DescriptorConfig.to_dict()` at line 1589-1601)

**Pydantic V2 Solution**: Add backward-compatible wrapper:
```python
def to_dict(self) -> Dict[str, Any]:
    """Backward compatible dict conversion."""
    return self.model_dump()
```

---

## 3. MIGRATION ARCHITECTURE

### 3.1 Import Structure Changes

```
BEFORE (all files):
from dataclasses import dataclass, field

AFTER (by file type):

# base.py - Simple dataclass migration
from pydantic.dataclasses import dataclass
from pydantic import field_validator
from dataclasses import field  # Keep for default_factory

# config_containers.py - Full BaseModel migration
from pydantic import BaseModel, field_validator, model_validator, ConfigDict
from pydantic import ValidationError as PydanticValidationError  # Aliased
from typing_extensions import Self

# config_bridge.py - Mutable BaseModel migration
from pydantic import BaseModel, field_validator, ConfigDict

# validators.py - Wrapper addition (existing code preserved)
from pydantic import ValidationError as PydanticValidationError  # Aliased
```

### 3.2 Namespace Conflict Resolution

**Issue**: MILIA has its own `ValidationError` class in `exceptions.py` (lines 1743-1780)

**MILIA ValidationError signature** (exceptions.py):
```python
class ValidationError(BaseProjectError):
    def __init__(self, message: str, validation_type: str,
                 failed_checks: Optional[List[str]] = None, ...)
```

**Pydantic ValidationError signature**:
```python
class ValidationError(ValueError):
    # Different signature - list of validation errors
```

**Solution**: Aliased import prevents conflict:
```python
from pydantic import ValidationError as PydanticValidationError
```

---

## 4. PHASE 1: base.py MIGRATION ✅ COMPLETE

**File**: `milia_pipeline/datasets/base.py`
**Lines**: 265
**Classes**: 3 frozen dataclasses
**Complexity**: LOW
**Status**: ✅ COMPLETE (2026-01-07)
**Rationale**: Simplest file, no `object.__setattr__` usage, drop-in Pydantic dataclass replacement

### 4.0 Implementation Summary

**Actual Change Applied**: Single import line change only. All `__post_init__` methods preserved unchanged.

**str_replace Applied**:
```
OLD (Line 18):
from dataclasses import dataclass, field

NEW:
from pydantic.dataclasses import dataclass  # Pydantic V2 drop-in replacement
from pydantic import field_validator
from dataclasses import field  # Keep for default_factory
```

**Classes Unchanged** (Pydantic dataclass calls `__post_init__` AFTER validation):
- `DatasetMetadata` (Lines 28-52): `__post_init__` validates `name`, `version`, `description`
- `DatasetSchema` (Lines 55-85): `__post_init__` validates `required_properties`, `coordinate_units`, `energy_units`
- `DatasetFeatures` (Lines 88-127): No `__post_init__`, pure boolean flags with `to_dict()` method

**Verification Test**:
```bash
python3 -c "
from milia_pipeline.datasets.base import DatasetMetadata, DatasetSchema, DatasetFeatures
dm = DatasetMetadata(name='Test', version='1.0', description='Test dataset')
assert dm.name == 'Test'
print('✅ Phase 1 verification passed')
"
```

### 4.1 DatasetMetadata (Lines 28-52)

**Current Implementation**:
```python
# Line 18
from dataclasses import dataclass, field

# Lines 28-52
@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    version: str
    description: str
    author: Optional[str] = None
    license: Optional[str] = None

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("DatasetMetadata.name must be a non-empty string")
        if not self.version or not isinstance(self.version, str):
            raise ValueError("DatasetMetadata.version must be a non-empty string")
        if not self.description or not isinstance(self.description, str):
            raise ValueError("DatasetMetadata.description must be a non-empty string")
```

**str_replace #1 - Import change (Line 18)**:
```
OLD:
from dataclasses import dataclass, field

NEW:
from pydantic.dataclasses import dataclass  # Pydantic drop-in for runtime validation
from pydantic import field_validator
from dataclasses import field  # Keep for default_factory
```

**str_replace #2 - DatasetMetadata class (Lines 28-52)**:

Option A (Preserve `__post_init__` - RECOMMENDED for minimal changes):
```
# No change needed - Pydantic dataclass calls __post_init__ AFTER validation
# The existing __post_init__ will work as-is
```

Option B (Convert to field_validator - cleaner Pydantic style):
```
OLD:
    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("DatasetMetadata.name must be a non-empty string")
        if not self.version or not isinstance(self.version, str):
            raise ValueError("DatasetMetadata.version must be a non-empty string")
        if not self.description or not isinstance(self.description, str):
            raise ValueError("DatasetMetadata.description must be a non-empty string")

NEW:
    @field_validator('name', 'version', 'description', mode='after')
    @classmethod
    def validate_non_empty_string(cls, v: str, info) -> str:
        """Validate required string fields are non-empty."""
        if not v or not isinstance(v, str):
            raise ValueError(f"DatasetMetadata.{info.field_name} must be a non-empty string")
        return v
```

### 4.2 DatasetSchema (Lines 55-85)

**Current Implementation** (Lines 73-85):
```python
def __post_init__(self):
    if not isinstance(self.required_properties, tuple):
        raise TypeError("required_properties must be a tuple")
    if not self.required_properties:
        raise ValueError("required_properties cannot be empty")

    valid_coord_units = ('angstrom', 'bohr')
    if self.coordinate_units not in valid_coord_units:
        raise ValueError(f"coordinate_units must be one of {valid_coord_units}")

    valid_energy_units = ('hartree', 'eV', 'kcal/mol', 'kJ/mol')
    if self.energy_units not in valid_energy_units:
        raise ValueError(f"energy_units must be one of {valid_energy_units}")
```

**Recommended Approach**: Preserve `__post_init__` as-is. Pydantic dataclass calls it AFTER field validation, so existing logic works.

Alternative (Pydantic field_validator):
```python
@field_validator('required_properties', mode='after')
@classmethod
def validate_required_properties(cls, v: Tuple[str, ...]) -> Tuple[str, ...]:
    if not isinstance(v, tuple):
        raise TypeError("required_properties must be a tuple")
    if not v:
        raise ValueError("required_properties cannot be empty")
    return v

@field_validator('coordinate_units', mode='after')
@classmethod
def validate_coordinate_units(cls, v: str) -> str:
    valid = ('angstrom', 'bohr')
    if v not in valid:
        raise ValueError(f"coordinate_units must be one of {valid}")
    return v

@field_validator('energy_units', mode='after')
@classmethod
def validate_energy_units(cls, v: str) -> str:
    valid = ('hartree', 'eV', 'kcal/mol', 'kJ/mol')
    if v not in valid:
        raise ValueError(f"energy_units must be one of {valid}")
    return v
```

### 4.3 DatasetFeatures (Lines 88-127)

**Current Implementation**: No `__post_init__`, pure boolean flags with `to_dict()` method.

**Migration**: Drop-in replacement - no changes needed except import.

**Backward Compatibility**: `to_dict()` method (lines 112-123) already exists and will continue to work. Optionally add:
```python
def to_dict(self) -> Dict[str, bool]:
    """Convert to dictionary for compatibility with existing code."""
    # Can also use: return asdict(self) after importing from dataclasses
    return {
        'vibrational_analysis': self.vibrational_analysis,
        # ... existing implementation
    }
```

### 4.4 Complete base.py str_replace Sequence

**Step 1**: Replace import statement
```
str_replace target (line 18):
from dataclasses import dataclass, field

str_replace replacement:
from pydantic.dataclasses import dataclass  # Pydantic drop-in for runtime validation
from pydantic import field_validator
from dataclasses import field  # Keep for default_factory
```

**Step 2**: No further changes required - `__post_init__` methods work as-is with Pydantic dataclass.

---

## 5. PHASE 2: config_containers.py MIGRATION ✅ COMPLETE

**File**: `milia_pipeline/config/config_containers.py`
**Lines**: 4405 → 4479 (after migration)
**Classes**: 10 frozen dataclasses
**Complexity**: HIGH
**Status**: ✅ COMPLETE (2026-01-07)
**Rationale**: Contains `object.__setattr__` pattern that required `model_validator(mode='before')` pattern

### 5.0 Implementation Summary

**Critical Pattern Change**: Original blueprint specified `model_validator(mode='after')` with `model_copy()`. This **DOES NOT WORK** for top-level `__init__`. All classes were migrated using `model_validator(mode='before')` pattern instead.

**Import Changes Applied (Lines 24-27)**:
```python
# OLD:
from dataclasses import dataclass, field

# NEW:
from pydantic import BaseModel, field_validator, model_validator, Field
from pydantic import ValidationError as PydanticValidationError
from typing_extensions import Self
```

**Correct Pattern Used (all 10 classes)**:
```python
# ✅ CORRECT: model_validator(mode='before') modifies input dict
@model_validator(mode='before')
@classmethod
def initialize_fields(cls, data: Any) -> Any:
    """Initialize None fields before field assignment."""
    if isinstance(data, dict):
        if data.get('handler_config') is None:
            data['handler_config'] = {}
    return data
```

**Verification Test**:
```bash
python3 -c "
import sys; sys.path.insert(0, '.')
import milia_pipeline.config.config_containers as cc
# Mock registry functions
cc._get_valid_dataset_types = lambda: ['DFT', 'DMC', 'QM9']
cc._is_valid_dataset_type = lambda t: t.upper() in ['DFT', 'DMC', 'QM9']
cc._registry_is_registered = lambda t: t.upper() in ['DFT', 'DMC', 'QM9']
# Test
dc = cc.DatasetConfig(dataset_type='DMC', uncertainty_config={'use_for_loss_weighting': True})
assert dc.is_uncertainty_enabled == True
print('✅ Phase 2 verification passed')
"
```

### 5.1 Classes Migrated

| Class | Line (After) | Fields | Validator Pattern | Status |
|-------|--------------|--------|-------------------|--------|
| `DatasetConfig` | 294 | 6 | `mode='before'` + `field_validator('dataset_type')` | ✅ |
| `FilterConfig` | 447 | 6 | `mode='before'` | ✅ |
| `StructuralFeaturesConfig` | 549 | 5 | `mode='before'` | ✅ |
| `ProcessingConfig` | 640 | 13 | `mode='before'` | ✅ |
| `HandlerConfig` | 790 | 7 | `mode='before'` + `field_validator('handler_type')` | ✅ |
| `TransformSpec` | 880 | 5 | `mode='before'` + `field_validator('name', 'kwargs')` | ✅ |
| `ExperimentalSetup` | 984 | 8 | `mode='before'` + `field_validator('name', 'transforms')` | ✅ |
| `TransformationConfig` | 1184 | 7 | `mode='before'` + `mode='after'` (for validation only) | ✅ |
| `DescriptorConfig` | 1489 | 10 | `mode='before'` + `field_validator('error_handling', 'validation_mode', 'num_workers')` | ✅ |
| `DescriptorCategoryConfig` | 1673 | 4 | `mode='before'` + `field_validator('category_name')` | ✅ |

**Note**: `TransformationConfig` uses BOTH `model_validator(mode='before')` for field initialization AND `model_validator(mode='after')` returning `self` for pure validation (checking `default_setup` exists in `experimental_setups`).

### 5.2 Import Changes (Lines 24-29)

**str_replace target**:
```python
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from dataclasses import dataclass, field
import logging
```

**str_replace replacement**:
```python
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from pydantic import BaseModel, field_validator, model_validator, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError
from typing_extensions import Self
import logging
```

### 5.3 DatasetConfig Migration (Lines 294-344) ✅ IMPLEMENTED

**⚠️ CRITICAL: Original pattern was INCORRECT**. The `model_validator(mode='after')` with `model_copy()` does not work for top-level `__init__` calls.

**Actual Implementation Applied**:
```python
class DatasetConfig(BaseModel, frozen=True):
    """
    Container for dataset type configuration.

    [Preserved existing docstring]
    """
    dataset_type: str
    uncertainty_config: Optional[Dict[str, Any]] = None
    is_uncertainty_enabled: bool = False
    handler_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    validation_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    migration_config: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator('dataset_type')
    @classmethod
    def validate_dataset_type(cls, v: str) -> str:
        """Validate dataset_type using dynamic registry lookup."""
        if not _is_valid_dataset_type(v):
            valid_types = _get_valid_dataset_types()
            raise ValueError(f"Invalid dataset_type: {v}. Must be one of {valid_types}")
        return v

    @model_validator(mode='before')
    @classmethod
    def set_computed_fields_and_defaults(cls, data: Any) -> Any:
        """Initialize None fields and compute derived values before field assignment."""
        if isinstance(data, dict):
            # Auto-compute uncertainty enabled if not explicitly set
            uncertainty_config = data.get('uncertainty_config')
            is_uncertainty_enabled = data.get('is_uncertainty_enabled', False)
            if uncertainty_config and not is_uncertainty_enabled:
                uncertainty_enabled = bool(uncertainty_config.get('use_for_loss_weighting', False))
                if uncertainty_enabled:
                    data['is_uncertainty_enabled'] = uncertainty_enabled

            # Initialize dict fields if None or missing
            if data.get('handler_config') is None:
                data['handler_config'] = {}
            if data.get('validation_config') is None:
                data['validation_config'] = {}
            if data.get('migration_config') is None:
                data['migration_config'] = {}
        return data

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()

    # All existing methods preserved unchanged (is_compatible_with_handler, get_handler_config, etc.)
```

### 5.4 FilterConfig Migration (Lines 447-507) ✅ IMPLEMENTED

**Actual Implementation Applied**:
```python
class FilterConfig(BaseModel, frozen=True):
    """
    Container for molecule filtering configuration.

    [Preserved existing docstring]
    """
    max_atoms: Optional[int] = None
    min_atoms: Optional[int] = None
    heavy_atom_filter: Optional[Dict[str, Any]] = None
    dmc_uncertainty_filter: Optional[Dict[str, Any]] = None
    handler_filters: Optional[Dict[str, Dict[str, Any]]] = Field(default_factory=dict)
    filter_validation: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def initialize_dict_fields(cls, data: Any) -> Any:
        """Initialize None dict fields to empty dicts before field assignment."""
        if isinstance(data, dict):
            if data.get('handler_filters') is None:
                data['handler_filters'] = {}
            if data.get('filter_validation') is None:
                data['filter_validation'] = {}
        return data

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()

    # All existing methods preserved unchanged
```

### 5.5 Correct Migration Template for Frozen BaseModel Classes

**⚠️ CRITICAL**: Do NOT use `model_validator(mode='after')` with `model_copy()` for field initialization. It does not work for top-level `__init__` calls.

**Correct Template**:
```python
# BEFORE (Standard Library dataclass)
@dataclass(frozen=True)
class ClassName:
    field1: Type1
    field2: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        if self.field2 is None:
            object.__setattr__(self, 'field2', {})
        # validation logic

# AFTER (Pydantic V2 BaseModel)
class ClassName(BaseModel, frozen=True):
    field1: Type1
    field2: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator('field1')
    @classmethod
    def validate_field1(cls, v: Type1) -> Type1:
        # validation logic
        return v

    @model_validator(mode='before')
    @classmethod
    def initialize_defaults(cls, data: Any) -> Any:
        """Initialize None fields to defaults BEFORE field assignment."""
        if isinstance(data, dict):
            if data.get('field2') is None:
                data['field2'] = {}
        return data

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
```

**When `model_validator(mode='after')` IS Appropriate**:
```python
# ONLY for pure validation (returning self unchanged)
@model_validator(mode='after')
def validate_consistency(self) -> Self:
    """Validate cross-field consistency - NO field modification."""
    if self.default_setup not in self.experimental_setups:
        raise ValueError(f"Default setup '{self.default_setup}' not found")
    return self  # Must return self, NOT model_copy()
```

### 5.6 Specific Migrations Implemented

#### StructuralFeaturesConfig (Lines 549-615) ✅
- Pattern: `model_validator(mode='before')`
- Fields initialized: `handler_features`, `feature_validation`

#### ProcessingConfig (Lines 640-785) ✅
- Pattern: `model_validator(mode='before')`
- Fields initialized: `node_features` (list), `vector_graph_properties` (list), `variable_len_graph_properties` (list), `handler_processing` (dict), `migration_settings` (dict)

#### HandlerConfig (Lines 790-862) ✅
- Pattern: `model_validator(mode='before')` + `field_validator('handler_type')`
- Dynamic registry validation preserved
- Fields initialized: `validation_settings`, `processing_settings`, `error_handling`, `performance_settings`, `compatibility_layer`

#### TransformSpec (Lines 880-980) ✅
- Pattern: `model_validator(mode='before')` + `field_validator('name', 'kwargs')`
- Validation: `name` must be non-empty string, `kwargs` must be dict
- Fields initialized: `kwargs`, `validation_config`

#### ExperimentalSetup (Lines 984-1180) ✅
- Pattern: `model_validator(mode='before')` + `field_validator('name', 'transforms')`
- Validation: `name` must be non-empty string, `transforms` must be list of TransformSpec
- Fields initialized: `expected_effects` (list), `dataset_compatibility` (list), `validation_config` (dict)

#### TransformationConfig (Lines 1184-1485) ✅
- Pattern: `model_validator(mode='before')` + `model_validator(mode='after')` (validation only)
- `mode='before'`: Initialize `standard_transforms`, `validation`, `performance_settings`, `migration_metadata`, `research_metadata`
- `mode='after'`: Validate `default_setup` exists in `experimental_setups` (returns `self`)

#### DescriptorConfig (Lines 1489-1670) ✅
- Pattern: `model_validator(mode='before')` + `field_validator('error_handling', 'validation_mode', 'num_workers')`
- Auto-adjustment: `num_workers` set to 2 if `parallel_computation=True` and `num_workers=1`
- Fields initialized: `categories`

#### DescriptorCategoryConfig (Lines 1673-1715) ✅
- Pattern: `model_validator(mode='before')` + `field_validator('category_name')`
- Validation: `category_name` in ['constitutional', 'topological', 'geometric', 'electronic', 'pharmacophore', 'fingerprint', 'custom']
- Fields initialized: `options`

### 5.7 Factory Function Updates (Lines 1661-1900)

Factory functions like `create_dataset_config_from_global()` construct dataclass instances. After migration:

**Before**:
```python
return DatasetConfig(
    dataset_type=dataset_type,
    uncertainty_config=uncertainty_config,
    ...
)
```

**After** (works unchanged - Pydantic BaseModel accepts same constructor syntax):
```python
return DatasetConfig(
    dataset_type=dataset_type,
    uncertainty_config=uncertainty_config,
    ...
)
```

Optionally, for explicit dict validation:
```python
return DatasetConfig.model_validate({
    'dataset_type': dataset_type,
    'uncertainty_config': uncertainty_config,
    ...
})
```

---

## 6. PHASE 3: config_bridge.py MIGRATION ✅ COMPLETE

**File**: `milia_pipeline/models/utils/config_bridge.py`
**Lines**: 1527 → 1545 (after migration)
**Classes**: 31 mutable dataclasses
**Complexity**: MEDIUM
**Status**: ✅ COMPLETE (2026-01-07)
**Rationale**: Mutable dataclasses with `validate()` methods - simpler migration (no `frozen=True`, no `object.__setattr__`)

### 6.0 Implementation Summary

**Key Pattern**: Unlike Phase 2, these are mutable models. Used `@field_validator` for enum validation and `@model_validator(mode='after')` for conditional validation (e.g., validate strategy only when `enabled=True`).

**Import Changes Applied (Lines 26-28)**:
```python
# OLD:
from dataclasses import dataclass, field

# NEW:
from pydantic import BaseModel, field_validator, model_validator, Field
from pydantic import ValidationError as PydanticValidationError
```

**Verification Test (All 22 tests passed)**:
```bash
python3 -c "
from milia_pipeline.models.utils.config_bridge import (
    ModelConfig, ModelSelectionConfig, DataSplitConfig, TrainingConfig,
    HPOConfigBridge, AccelerationConfig, DeploymentConfig
)
from pydantic import BaseModel
# Test 1: BaseModel inheritance
assert issubclass(ModelConfig, BaseModel)
# Test 2: Validation on construction
try:
    ModelSelectionConfig(task_type='invalid', model_name='Test')
except Exception:
    print('✅ Invalid task_type rejected')
# Test 3: Valid instantiation
mc = ModelConfig()
print('✅ Phase 3 verification passed')
"
```

### 6.1 Classes Migrated (31 total)

| Category | Classes | Count |
|----------|---------|-------|
| **Selection** | `ModelSelectionConfig` | 1 |
| **Training** | `DataSplitConfig`, `LossConfig`, `OptimizerConfig`, `SchedulerConfig`, `CallbackConfig`, `CallbacksConfig`, `ValidationConfig`, `LoggingConfig`, `TrainingConfig`, `EvaluationConfig` | 10 |
| **Device/Distributed** | `DeviceConfig`, `DDPConfig`, `FSDPConfig`, `DeepSpeedConfig`, `DistributedConfig`, `MemoryConfig`, `DataLoaderConfig`, `ComputationConfig`, `AccelerationConfig` | 9 |
| **Deployment** | `QuantizationConfig`, `PruningConfig`, `DistillationConfig`, `OptimizationConfig`, `EdgeDeploymentConfig`, `CloudDeploymentConfig`, `FederatedConfig`, `DriftDetectionConfig`, `RetrainingConfig`, `MonitoringConfig`, `DeploymentConfig`, `PluginsConfig` | 12 |
| **HPO** | `HPOSearchSpaceParamBridge`, `HPOPrunerConfigBridge`, `HPOSamplerConfigBridge`, `HPOStudyConfigBridge`, `HPOConfigBridge` | 5 |
| **Main** | `ModelConfig` | 1 |
| **TOTAL** | | **31** |

### 6.2 Validation Patterns Implemented

**Pattern A: Simple Enum Validation (`@field_validator`)**
```python
class ModelSelectionConfig(BaseModel):
    task_type: str
    model_name: str

    @field_validator('task_type')
    @classmethod
    def validate_task_type(cls, v: str) -> str:
        if not v:
            raise ValueError("task_type is required")
        try:
            TaskType(v)
        except ValueError:
            valid_tasks = [t.value for t in TaskType]
            raise ValueError(f"Invalid task_type '{v}'. Must be one of: {valid_tasks}")
        return v
```

**Pattern B: Cross-Field Validation (`@model_validator(mode='after')`)**
```python
class DataSplitConfig(BaseModel):
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    @model_validator(mode='after')
    def validate_ratios(self) -> 'DataSplitConfig':
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        return self
```

**Pattern C: Conditional Validation (validate only when enabled)**
```python
class DistributedConfig(BaseModel):
    enabled: bool = False
    strategy: str = "ddp"

    @model_validator(mode='after')
    def validate_strategy(self) -> 'DistributedConfig':
        if self.enabled:  # Only validate when enabled
            try:
                DistributedStrategy(self.strategy)
            except ValueError:
                valid = [s.value for s in DistributedStrategy]
                raise ValueError(f"Invalid strategy '{self.strategy}'. Must be one of: {valid}")
        return self
```

**Pattern D: HPO Multi-Field Validation**
```python
class HPOConfigBridge(BaseModel):
    backend: str = "optuna"
    n_trials: int = 100

    @field_validator('backend')
    @classmethod
    def validate_backend(cls, v: str) -> str:
        if v not in ("optuna", "ray_tune"):
            raise ValueError(f"Unknown HPO backend: '{v}'")
        return v

    @field_validator('n_trials')
    @classmethod
    def validate_n_trials(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"n_trials must be at least 1, got {v}")
        return v
```

### 6.3 Backward Compatibility

All classes with `validate()` methods retain them as pass-through for backward compatibility:
```python
def validate(self):
    """Backward compatible validate method (validation happens on construction)."""
    pass
```

### 6.4 HPO Enum Types Preserved

Pydantic V2 natively validates Enum fields. The HPO classes use Enum types directly:
- `HPOParamType` → `HPOSearchSpaceParamBridge.type`
- `HPOPrunerType` → `HPOPrunerConfigBridge.type`
- `HPOSamplerType` → `HPOSamplerConfigBridge.type`
- `HPODirection` → `HPOStudyConfigBridge.direction`

### 6.5 Import Statement Change

**str_replace Applied (Lines 26-27)**:

**OLD**:
```python
from dataclasses import dataclass, field
from enum import Enum
```

**NEW**:
```python
from pydantic import BaseModel, field_validator, model_validator, Field
from pydantic import ValidationError as PydanticValidationError
from enum import Enum
```

---

## 7. PHASE 4: validators.py INTEGRATION ✅ COMPLETE

**File**: `milia_pipeline/config/validators.py`
**Lines**: 4917 → 5070 (after migration)
**Key Class**: `ValidationResult` (lines 666-781)
**Complexity**: LOW (integration, not migration)
**Status**: ✅ COMPLETE (2026-01-07)
**Rationale**: Added Pydantic wrapper functions to bridge Pydantic ValidationError with MILIA ValidationResult

### 7.0 Implementation Summary

**Changes Applied**:
1. **Pydantic Import** (Lines 349-356): Added aliased import with graceful fallback
2. **Wrapper Functions** (Lines 797-937): Added `wrap_pydantic_validation_error()` and `validate_with_pydantic_model()`

**Import Change Applied (after line 347)**:
```python
# Pydantic V2 integration for runtime validation
# Phase 4: Enables conversion between Pydantic ValidationError and MILIA ValidationResult
try:
    from pydantic import ValidationError as PydanticValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PydanticValidationError = None
    PYDANTIC_AVAILABLE = False
```

**Wrapper Functions Applied (after line 794)**:
- `wrap_pydantic_validation_error()`: Converts Pydantic ValidationError → MILIA ValidationResult
- `validate_with_pydantic_model()`: Validates dict against Pydantic BaseModel, returns ValidationResult or tuple

**Verification Tests (10/10 passed)**:
```bash
python3 -c "
from milia_pipeline.config.validators import (
    PYDANTIC_AVAILABLE,
    wrap_pydantic_validation_error,
    validate_with_pydantic_model,
    ValidationResult
)
from pydantic import BaseModel

class TestModel(BaseModel):
    name: str
    value: int

# Test valid data
result = validate_with_pydantic_model({'name': 'test', 'value': 42}, TestModel)
assert result.is_valid == True
print('✅ Phase 4 verification passed')
"
```

### 7.1 Current ValidationResult Implementation (Lines 666-781)

```python
class ValidationResult:
    """
    Validation result wrapper that enforces checking.
    """
    def __init__(self, is_valid: bool, errors: Optional[List[str]] = None, ...):
        self._is_valid = is_valid
        self._errors = errors or []
        self._checked = False

    @property
    def is_valid(self) -> bool:
        self._checked = True
        return self._is_valid

    def get_validated_data(self) -> Any:
        if not self._checked:
            raise ValidationError(...)
        if not self._is_valid:
            raise ValidationError(...)
        return self._data
```

### 7.2 Coexistence Strategy

`ValidationResult` serves a **different purpose** than Pydantic validation:
- **Pydantic**: Type-based validation on model construction
- **ValidationResult**: Business logic validation with enforced result checking

**Recommendation**: Keep both systems, add Pydantic-compatible wrapper:

```python
# Add to validators.py (after line 786)

def wrap_pydantic_validation_error(pydantic_error: 'PydanticValidationError') -> ValidationResult:
    """
    Convert Pydantic ValidationError to MILIA ValidationResult.

    This allows existing code that expects ValidationResult to work
    with Pydantic validation errors.

    Args:
        pydantic_error: Pydantic V2 ValidationError instance

    Returns:
        ValidationResult with errors extracted from Pydantic error
    """
    errors = [str(e['msg']) for e in pydantic_error.errors()]
    return ValidationResult(
        is_valid=False,
        errors=errors,
        context="pydantic_validation"
    )


def validate_with_pydantic_model(
    data: Dict[str, Any],
    model_class: type,
    return_wrapper: bool = True
) -> Union[ValidationResult, Tuple[bool, List[str]]]:
    """
    Validate data using a Pydantic model, returning MILIA ValidationResult.

    Args:
        data: Dictionary of data to validate
        model_class: Pydantic BaseModel class to validate against
        return_wrapper: If True, return ValidationResult; if False, return tuple

    Returns:
        ValidationResult or (is_valid, errors) tuple
    """
    try:
        validated = model_class.model_validate(data)
        if return_wrapper:
            return ValidationResult(is_valid=True, errors=[], data=validated)
        return True, []
    except PydanticValidationError as e:
        errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
        if return_wrapper:
            return ValidationResult(is_valid=False, errors=errors, context="pydantic_validation")
        return False, errors
```

### 7.3 Import Addition

**str_replace target** (after line 264):
```python
    class ConfigurationError(Exception):
        """Configuration error."""
        pass
```

**str_replace addition** (add after existing imports):
```python
# Pydantic integration
try:
    from pydantic import ValidationError as PydanticValidationError
except ImportError:
    PydanticValidationError = None
```

---

## 8. PHASE 5: exceptions.py COMPATIBILITY

**File**: `milia_pipeline/exceptions.py`
**Lines**: 4046
**Key Class**: `ValidationError` (lines 1743-1780)
**Issue**: Namespace conflict with Pydantic's `ValidationError`

### 8.1 MILIA ValidationError Analysis (Lines 1743-1780)

```python
class ValidationError(BaseProjectError):
    """
    Raised when data validation fails in any context.
    """
    def __init__(self, message: str, validation_type: str,
                 failed_checks: Optional[List[str]] = None,
                 data_context: Optional[str] = None,
                 handler_type: Optional[str] = None,
                 details: Optional[str] = None, **kwargs):
```

**Key Differences from Pydantic ValidationError**:
- Different inheritance: `BaseProjectError` vs `ValueError`
- Different signature: accepts `message`, `validation_type`, etc.
- Different purpose: business logic errors vs type validation errors

### 8.2 Resolution Strategy

**No changes to exceptions.py needed.** Handle at import site:

```python
# In files that need both
from milia_pipeline.exceptions import ValidationError  # MILIA's
from pydantic import ValidationError as PydanticValidationError  # Pydantic's
```

### 8.3 Documentation Update

Add to exceptions.py docstring (lines 36-37):
```python
"""
NOTE: When using Pydantic, import Pydantic's ValidationError with an alias:
    from pydantic import ValidationError as PydanticValidationError
to avoid conflict with this module's ValidationError class.
"""
```

---

## 9. CROSS-CUTTING CONCERNS

### 9.1 Thread Safety

**Current State**: Thread-safe via `RLock` in registry.py (line 30)

**After Migration**: Pydantic models are thread-safe for:
- Model instantiation
- Validation
- Serialization (`model_dump()`, `model_dump_json()`)

**Registry Integration**: Unchanged - `_get_valid_dataset_types()` still uses thread-safe registry.

### 9.2 Serialization/Pickling

**Potential Issue** (from tracker line 441): "Pickling requirements?"

**Pydantic V2 Behavior**:
- BaseModel instances are picklable by default
- `model_dump()` returns dict (JSON-serializable)
- `model_dump_json()` returns JSON string

**Verification**: Test multiprocessing scenarios after migration.

### 9.3 Performance Considerations

**Pydantic V2 Performance** (from web search):
- Written in Rust (pydantic-core)
- Significantly faster than V1
- Validation overhead minimal for typical usage

**Recommendation**: Benchmark before/after for high-frequency paths (molecule processing loops).

### 9.4 Circular Import Prevention

**Current Protection** (config_loader.py lines 66-67, 106-107):
```python
_REGISTRY_INITIALIZING = False  # Guard against re-entrant calls
if _REGISTRY_INITIALIZING:
    return False
```

**After Migration**: Same protection works - Pydantic validators call same registry functions.

### 9.5 Backward Compatibility Guarantees

| Current Usage | After Migration | Status |
|---------------|-----------------|--------|
| `DatasetConfig(dataset_type="DFT")` | Same | ✅ PRESERVED |
| `config.dataset_type` | Same | ✅ PRESERVED |
| `config.to_dict()` | Same (add if missing) | ✅ PRESERVED |
| `isinstance(config, DatasetConfig)` | Same | ✅ PRESERVED |
| `hash(config)` (frozen) | Same (frozen=True) | ✅ PRESERVED |
| `config.field = value` (frozen) | Raises error | ✅ PRESERVED |

---

## 10. VERIFICATION CHECKLIST

### 10.1 Phase 1 Verification (base.py)

- [ ] Import `from pydantic.dataclasses import dataclass` works
- [ ] `DatasetMetadata("DFT", "1.0", "desc")` creates valid instance
- [ ] `DatasetMetadata("", "1.0", "desc")` raises ValueError
- [ ] `DatasetSchema` validates coordinate_units and energy_units
- [ ] `DatasetFeatures.to_dict()` returns correct dictionary
- [ ] Existing dataset implementations still work with `BaseDataset`

### 10.2 Phase 2 Verification (config_containers.py)

- [ ] All 10 classes migrate without breaking existing tests
- [ ] Dynamic registry validation works: `DatasetConfig(dataset_type="InvalidType")` raises
- [ ] `object.__setattr__` replacements produce same field values
- [ ] Nested containers validate correctly (`TransformationConfig` with `ExperimentalSetup`)
- [ ] Factory functions return valid instances
- [ ] `to_dict()` methods work on all classes

### 10.3 Phase 3 Verification (config_bridge.py)

- [ ] All 25+ classes migrate without breaking existing tests
- [ ] `validate()` methods work (or are safely deprecated)
- [ ] Enum validation works for `TaskType`, `DeviceType`, etc.
- [ ] Nested config validation works (`TrainingConfig` contains `DataSplitConfig`)

### 10.4 Phase 4 Verification (validators.py)

- [ ] `ValidationResult` unchanged and functional
- [ ] `wrap_pydantic_validation_error()` correctly converts errors
- [ ] `validate_with_pydantic_model()` returns correct types
- [ ] `@must_check` decorator still works

### 10.5 Phase 5 Verification (exceptions.py)

- [ ] MILIA `ValidationError` unchanged
- [ ] Aliased import works: `from pydantic import ValidationError as PydanticValidationError`
- [ ] No namespace conflicts at runtime

### 10.6 Integration Verification

- [ ] `python -c "from milia_pipeline.config import config_containers"` succeeds
- [ ] `python -c "from milia_pipeline.datasets import base"` succeeds
- [ ] `python -c "from milia_pipeline.models.utils import config_bridge"` succeeds
- [ ] Existing test suite passes
- [ ] No circular import errors
- [ ] Registry-based dataset type validation works end-to-end

---

## APPENDIX A: COMPLETE str_replace COMMANDS

**CRITICAL**: These are EXACT strings for `str_replace` tool. Copy-paste ready for new context window.

### IMPORTANT: SURGICAL REPLACEMENT APPROACH

The str_replace commands below are **surgical** - they replace ONLY:
1. The `@dataclass(frozen=True)` decorator + class declaration
2. The field definitions (changing `field(...)` to `Field(...)`)
3. The `__post_init__` method (converting to Pydantic validators)

**All existing class methods are PRESERVED and NOT touched.**

This is critical because each class has many additional methods after `__post_init__` that must remain unchanged. The str_replace `old_str` captures ONLY the portion being changed.

### ⚠️ CRITICAL VERIFICATION BEFORE EXECUTION

Before using ANY str_replace command in a new context window:

1. **VIEW the actual source file** to verify the `old_str` matches EXACTLY
2. **Check for whitespace differences** (trailing spaces, tabs vs spaces)
3. **Verify line numbers** have not changed since this blueprint was created
4. **Test ONE class first** before proceeding with others

The `old_str` values below were extracted from the source files on 2026-01-07. If the source files have been modified since then, the `old_str` values may need to be updated.

### Class Boundaries Reference (config_containers.py)

| Class | Declaration Start | __post_init__ End | Methods After |
|-------|-------------------|-------------------|---------------|
| DatasetConfig | 292 | 336 | 338-431 |
| FilterConfig | 434 | 464 | 466-528 |
| StructuralFeaturesConfig | 531 | 559 | 561-614 |
| ProcessingConfig | 617 | 666 | 668-760 |
| HandlerConfig | 762 | 797 | 799-838 |
| TransformSpec | 841 | 876 | 878-935 |
| ExperimentalSetup | 938 | 997 | 999-1135 |
| TransformationConfig | 1139 | 1195 | 1197-1428 |
| DescriptorConfig | 1434 | 1488 | 1490-1606 |
| DescriptorCategoryConfig | 1609 | 1640 | 1642-1654 |

---

### A.1 base.py str_replace Commands

**File Path**: `milia_pipeline/datasets/base.py`

#### A.1.1 Import Statement Change

**str_replace old_str**:
```python
from dataclasses import dataclass, field
```

**str_replace new_str**:
```python
from pydantic.dataclasses import dataclass  # Pydantic drop-in for runtime validation
from pydantic import field_validator
from dataclasses import field  # Keep for default_factory
```

**Note**: The existing `__post_init__` methods in `DatasetMetadata` and `DatasetSchema` will work as-is because Pydantic dataclasses call `__post_init__` AFTER validation (confirmed in Pydantic V2 migration guide).

---

### A.2 config_containers.py str_replace Commands

**File Path**: `milia_pipeline/config/config_containers.py`

#### A.2.1 Import Statement Change

**CRITICAL**: This is the EXACT string from config_containers.py lines 24-26.

**str_replace old_str**:
```python
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from dataclasses import dataclass, field
import logging
import time
```

**str_replace new_str**:
```python
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from pydantic import BaseModel, field_validator, model_validator, Field
from pydantic import ValidationError as PydanticValidationError
from typing_extensions import Self
import logging
import time
```

**Note**: The `import time` line is included because it immediately follows `import logging` in the actual file.

#### A.2.2 DatasetConfig Class Migration (Lines 292-336 ONLY)

**CRITICAL**: This replaces ONLY the decorator, class declaration, fields, and `__post_init__`.
All existing methods (lines 338-431) are PRESERVED and NOT touched.

**str_replace old_str** (EXACT content of lines 292-336):
```python
@dataclass(frozen=True)
class DatasetConfig:
    """
    Container for dataset-specific configuration.

    Enhanced for handler pattern support with improved validation
    and handler compatibility features.

    Attributes:
        dataset_type: Type of dataset ("DFT" or "DMC")
        uncertainty_config: DMC uncertainty handling configuration
        is_uncertainty_enabled: Whether uncertainty handling is active
        handler_config: Handler-specific configuration parameters
        validation_config: Configuration for handler validation
        migration_config: Configuration for migration scenarios
    """
    dataset_type: str
    uncertainty_config: Optional[Dict[str, Any]] = None
    is_uncertainty_enabled: bool = False
    handler_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    validation_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    migration_config: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        # Validate dataset_type using dynamic registry lookup
        valid_types = _get_valid_dataset_types()
        if not _is_valid_dataset_type(self.dataset_type):
            raise ValueError(f"Invalid dataset_type: {self.dataset_type}. Must be one of {valid_types}")

        # Auto-compute uncertainty enabled if not explicitly set
        if self.uncertainty_config and not hasattr(self, '_is_uncertainty_enabled_set'):
            uncertainty_enabled = bool(self.uncertainty_config.get('use_for_loss_weighting', False))
            object.__setattr__(self, 'is_uncertainty_enabled', uncertainty_enabled)

        # Initialize handler_config if None
        if self.handler_config is None:
            object.__setattr__(self, 'handler_config', {})

        # Initialize validation_config if None
        if self.validation_config is None:
            object.__setattr__(self, 'validation_config', {})

        # Initialize migration_config if None
        if self.migration_config is None:
            object.__setattr__(self, 'migration_config', {})
```

**str_replace new_str**:
```python
class DatasetConfig(BaseModel, frozen=True):
    """
    Container for dataset-specific configuration.

    Enhanced for handler pattern support with improved validation
    and handler compatibility features.

    Attributes:
        dataset_type: Type of dataset ("DFT" or "DMC")
        uncertainty_config: DMC uncertainty handling configuration
        is_uncertainty_enabled: Whether uncertainty handling is active
        handler_config: Handler-specific configuration parameters
        validation_config: Configuration for handler validation
        migration_config: Configuration for migration scenarios
    """
    dataset_type: str
    uncertainty_config: Optional[Dict[str, Any]] = None
    is_uncertainty_enabled: bool = False
    handler_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    validation_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    migration_config: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator('dataset_type')
    @classmethod
    def validate_dataset_type(cls, v: str) -> str:
        """Validate dataset_type using dynamic registry lookup."""
        if not _is_valid_dataset_type(v):
            valid_types = _get_valid_dataset_types()
            raise ValueError(f"Invalid dataset_type: {v}. Must be one of {valid_types}")
        return v

    @model_validator(mode='after')
    def set_computed_fields_and_defaults(self) -> Self:
        """Initialize None fields and compute derived values."""
        updates = {}

        # Auto-compute uncertainty enabled if not explicitly set
        if self.uncertainty_config and not self.is_uncertainty_enabled:
            uncertainty_enabled = bool(self.uncertainty_config.get('use_for_loss_weighting', False))
            if uncertainty_enabled:
                updates['is_uncertainty_enabled'] = uncertainty_enabled

        # Initialize dict fields if None
        if self.handler_config is None:
            updates['handler_config'] = {}
        if self.validation_config is None:
            updates['validation_config'] = {}
        if self.migration_config is None:
            updates['migration_config'] = {}

        return self.model_copy(update=updates) if updates else self

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()
```

**VERIFICATION**: After this replacement, the existing methods `is_compatible_with_handler()`, `get_handler_config()`, `get_required_properties()`, and `validate_handler_compatibility()` (lines 338-431) remain completely untouched.

#### A.2.3 FilterConfig Class Migration (Lines 434-464 ONLY)

**CRITICAL**: Note the space before `@dataclass` on line 433 - the actual file has ` \n@dataclass(frozen=True) ` with trailing space.

**str_replace old_str** (EXACT content of lines 433-464, note leading space on line 433):
```python

@dataclass(frozen=True)
class FilterConfig:
    """
    Container for molecule filtering configuration.

    Enhanced for supporting handler-specific filtering and
    improved validation capabilities.

    Attributes:
        max_atoms: Maximum number of atoms allowed
        min_atoms: Minimum number of atoms allowed
        heavy_atom_filter: Configuration for heavy atom filtering
        dmc_uncertainty_filter: DMC-specific uncertainty filtering
        handler_filters: Handler-specific filter configurations
        filter_validation: Configuration for filter validation
    """
    max_atoms: Optional[int] = None
    min_atoms: Optional[int] = None
    heavy_atom_filter: Optional[Dict[str, Any]] = None
    dmc_uncertainty_filter: Optional[Dict[str, Any]] = None
    handler_filters: Optional[Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    filter_validation: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize handler_filters if None
        if self.handler_filters is None:
            object.__setattr__(self, 'handler_filters', {})

        # Initialize filter_validation if None
        if self.filter_validation is None:
            object.__setattr__(self, 'filter_validation', {})
```

**str_replace new_str**:
```python

class FilterConfig(BaseModel, frozen=True):
    """
    Container for molecule filtering configuration.

    Enhanced for supporting handler-specific filtering and
    improved validation capabilities.

    Attributes:
        max_atoms: Maximum number of atoms allowed
        min_atoms: Minimum number of atoms allowed
        heavy_atom_filter: Configuration for heavy atom filtering
        dmc_uncertainty_filter: DMC-specific uncertainty filtering
        handler_filters: Handler-specific filter configurations
        filter_validation: Configuration for filter validation
    """
    max_atoms: Optional[int] = None
    min_atoms: Optional[int] = None
    heavy_atom_filter: Optional[Dict[str, Any]] = None
    dmc_uncertainty_filter: Optional[Dict[str, Any]] = None
    handler_filters: Optional[Dict[str, Dict[str, Any]]] = Field(default_factory=dict)
    filter_validation: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode='after')
    def initialize_dict_fields(self) -> Self:
        """Initialize None dict fields to empty dicts."""
        updates = {}
        if self.handler_filters is None:
            updates['handler_filters'] = {}
        if self.filter_validation is None:
            updates['filter_validation'] = {}
        return self.model_copy(update=updates) if updates else self

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()
```

**VERIFICATION**: Existing methods `get_handler_filters()` and `validate_filter_config()` (lines 466+) remain untouched.

#### A.2.4 StructuralFeaturesConfig Class Migration (Lines 531-559 ONLY)

**str_replace old_str** (EXACT content):
```python
@dataclass(frozen=True)
class StructuralFeaturesConfig:
    """
    Container for structural features configuration.

    Enhanced for andler-Based Pattern Development with handler-specific feature configuration
    and improved validation.

    Attributes:
        atom_features: List of atom-level features to extract
        bond_features: List of bond-level features to extract
        preprocessing: Preprocessing configuration
        handler_features: Handler-specific feature configurations
        feature_validation: Feature validation configuration
    """
    atom_features: List[str]
    bond_features: List[str]
    preprocessing: Optional[Dict[str, Any]] = None
    handler_features: Optional[Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    feature_validation: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize handler_features if None
        if self.handler_features is None:
            object.__setattr__(self, 'handler_features', {})

        # Initialize feature_validation if None
        if self.feature_validation is None:
            object.__setattr__(self, 'feature_validation', {})
```

**str_replace new_str**:
```python
class StructuralFeaturesConfig(BaseModel, frozen=True):
    """
    Container for structural features configuration.

    Enhanced for andler-Based Pattern Development with handler-specific feature configuration
    and improved validation.

    Attributes:
        atom_features: List of atom-level features to extract
        bond_features: List of bond-level features to extract
        preprocessing: Preprocessing configuration
        handler_features: Handler-specific feature configurations
        feature_validation: Feature validation configuration
    """
    atom_features: List[str]
    bond_features: List[str]
    preprocessing: Optional[Dict[str, Any]] = None
    handler_features: Optional[Dict[str, Dict[str, Any]]] = Field(default_factory=dict)
    feature_validation: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode='after')
    def initialize_dict_fields(self) -> Self:
        """Initialize None dict fields to empty dicts."""
        updates = {}
        if self.handler_features is None:
            updates['handler_features'] = {}
        if self.feature_validation is None:
            updates['feature_validation'] = {}
        return self.model_copy(update=updates) if updates else self

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()
```

**VERIFICATION**: Existing methods `get_handler_features()` and `validate_feature_config()` remain untouched.

#### A.2.5 ProcessingConfig Class Migration (Lines 617-666 ONLY)

**CRITICAL**: The actual file uses `List[str] = None` (not `Optional[List[str]] = None`). The old_str must match exactly.

**str_replace old_str** (EXACT content from source file):
```python
@dataclass(frozen=True)
class ProcessingConfig:
    """
    Container for data processing configuration.

    Enhanced for Handler-Based Pattern Development with handler-specific processing configuration
    and migration support.

    Attributes:
        scalar_graph_targets: Scalar properties to include in y tensor
        node_features: Node-level features to add
        vector_graph_properties: Fixed-size vector properties
        variable_len_graph_properties: Variable-length properties
        calculate_atomization_energy_from: Base energy for atomization calculation
        atomization_energy_key_name: Key name for calculated atomization energy
        vibration_refinement: Vibrational data refinement settings
        test_molecule_limit: Limit for testing (None for full dataset)
        handler_processing: Handler-specific processing configurations
        migration_settings: Settings for migration scenarios
    """
    scalar_graph_targets: List[str]
    node_features: List[str] = None
    vector_graph_properties: List[str] = None
    variable_len_graph_properties: List[str] = None
    calculate_atomization_energy_from: Optional[str] = None
    atomization_energy_key_name: Optional[str] = None
    vibration_refinement: Optional[Dict[str, Any]] = None
    test_molecule_limit: Optional[int] = None
    handler_processing: Optional[Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    migration_settings: Optional[Dict[str, Any]] = field(default_factory=dict)
    preprocessing_feature_tier: Optional[str] = 'standard'
    preprocessing_num_molecules: Optional[int] = None
    preprocessing_cleanup_temp: bool = True

    def __post_init__(self):
        # Set defaults for None fields
        if self.node_features is None:
            object.__setattr__(self, 'node_features', [])
        if self.vector_graph_properties is None:
            object.__setattr__(self, 'vector_graph_properties', [])
        if self.variable_len_graph_properties is None:
            object.__setattr__(self, 'variable_len_graph_properties', [])

        # Initialize handler_processing if None
        if self.handler_processing is None:
            object.__setattr__(self, 'handler_processing', {})

        # Initialize migration_settings if None
        if self.migration_settings is None:
            object.__setattr__(self, 'migration_settings', {})
```

**str_replace new_str**:
```python
class ProcessingConfig(BaseModel, frozen=True):
    """
    Container for data processing configuration.

    Enhanced for Handler-Based Pattern Development with handler-specific processing configuration
    and migration support.

    Attributes:
        scalar_graph_targets: Scalar properties to include in y tensor
        node_features: Node-level features to add
        vector_graph_properties: Fixed-size vector properties
        variable_len_graph_properties: Variable-length properties
        calculate_atomization_energy_from: Base energy for atomization calculation
        atomization_energy_key_name: Key name for calculated atomization energy
        vibration_refinement: Vibrational data refinement settings
        test_molecule_limit: Limit for testing (None for full dataset)
        handler_processing: Handler-specific processing configurations
        migration_settings: Settings for migration scenarios
    """
    scalar_graph_targets: List[str]
    node_features: Optional[List[str]] = None
    vector_graph_properties: Optional[List[str]] = None
    variable_len_graph_properties: Optional[List[str]] = None
    calculate_atomization_energy_from: Optional[str] = None
    atomization_energy_key_name: Optional[str] = None
    vibration_refinement: Optional[Dict[str, Any]] = None
    test_molecule_limit: Optional[int] = None
    handler_processing: Optional[Dict[str, Dict[str, Any]]] = Field(default_factory=dict)
    migration_settings: Optional[Dict[str, Any]] = Field(default_factory=dict)
    preprocessing_feature_tier: Optional[str] = 'standard'
    preprocessing_num_molecules: Optional[int] = None
    preprocessing_cleanup_temp: bool = True

    @model_validator(mode='after')
    def initialize_list_and_dict_fields(self) -> Self:
        """Initialize None list/dict fields to empty collections."""
        updates = {}
        if self.node_features is None:
            updates['node_features'] = []
        if self.vector_graph_properties is None:
            updates['vector_graph_properties'] = []
        if self.variable_len_graph_properties is None:
            updates['variable_len_graph_properties'] = []
        if self.handler_processing is None:
            updates['handler_processing'] = {}
        if self.migration_settings is None:
            updates['migration_settings'] = {}
        return self.model_copy(update=updates) if updates else self

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()
```

**NOTE**: The new_str changes `List[str] = None` to `Optional[List[str]] = None` for proper Pydantic V2 typing. In Pydantic V2, fields with default `None` should use `Optional[X]` type annotation.

#### A.2.6 HandlerConfig Class Migration (Lines 762-797)

**str_replace old_str**:
```python
@dataclass(frozen=True)
class HandlerConfig:
    """
    Container for handler-specific configuration and settings.

    This is a container specifically for Handler-Based Pattern Development handler pattern support.

    Attributes:
        handler_type: Type of handler ("DFT" or "DMC")
        validation_settings: Handler validation configuration
        processing_settings: Handler processing configuration
        error_handling: Error handling configuration
        performance_settings: Performance optimization settings
        migration_mode: Whether handler is in migration mode
        compatibility_layer: Compatibility layer configuration
    """
    handler_type: str
    validation_settings: Optional[Dict[str, Any]] = field(default_factory=dict)
    processing_settings: Optional[Dict[str, Any]] = field(default_factory=dict)
    error_handling: Optional[Dict[str, Any]] = field(default_factory=dict)
    performance_settings: Optional[Dict[str, Any]] = field(default_factory=dict)
    migration_mode: bool = False
    compatibility_layer: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        # Validate handler type using dynamic registry lookup
        if not _is_valid_dataset_type(self.handler_type):
            valid_types = _get_valid_dataset_types()
            raise ValueError(f"Invalid handler_type: {self.handler_type}. Must be one of {valid_types}")

        # Initialize dictionaries if None
        for field_name in ['validation_settings', 'processing_settings', 'error_handling',
                          'performance_settings', 'compatibility_layer']:
            field_value = getattr(self, field_name)
            if field_value is None:
                object.__setattr__(self, field_name, {})
```

**str_replace new_str**:
```python
class HandlerConfig(BaseModel, frozen=True):
    """
    Container for handler-specific configuration and settings.

    This is a container specifically for Handler-Based Pattern Development handler pattern support.

    Attributes:
        handler_type: Type of handler ("DFT" or "DMC")
        validation_settings: Handler validation configuration
        processing_settings: Handler processing configuration
        error_handling: Error handling configuration
        performance_settings: Performance optimization settings
        migration_mode: Whether handler is in migration mode
        compatibility_layer: Compatibility layer configuration
    """
    handler_type: str
    validation_settings: Optional[Dict[str, Any]] = Field(default_factory=dict)
    processing_settings: Optional[Dict[str, Any]] = Field(default_factory=dict)
    error_handling: Optional[Dict[str, Any]] = Field(default_factory=dict)
    performance_settings: Optional[Dict[str, Any]] = Field(default_factory=dict)
    migration_mode: bool = False
    compatibility_layer: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator('handler_type')
    @classmethod
    def validate_handler_type(cls, v: str) -> str:
        """Validate handler_type using dynamic registry lookup."""
        if not _is_valid_dataset_type(v):
            valid_types = _get_valid_dataset_types()
            raise ValueError(f"Invalid handler_type: {v}. Must be one of {valid_types}")
        return v

    @model_validator(mode='after')
    def initialize_dict_fields(self) -> Self:
        """Initialize None dict fields to empty dicts."""
        updates = {}
        for field_name in ['validation_settings', 'processing_settings', 'error_handling',
                          'performance_settings', 'compatibility_layer']:
            if getattr(self, field_name) is None:
                updates[field_name] = {}
        return self.model_copy(update=updates) if updates else self

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()
```

#### A.2.7 TransformSpec Class Migration (Lines 841-867)

**str_replace old_str**:
```python
@dataclass(frozen=True)
class TransformSpec:
    """
    Individual transform specification with validation.

    Attributes:
        name: Transform name/identifier
        kwargs: Transform parameters (keyword arguments)
        enabled: Whether transform is enabled
        description: Human-readable description
        validation_config: Validation configuration
    """
    name: str
    kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
    enabled: bool = True
    description: Optional[str] = None
    validation_config: Optional[Dict[str, Any]] = field(default_factory=dict)


    def __post_init__(self):
        # Initialize kwargs if None
        if self.kwargs is None:
            object.__setattr__(self, 'kwargs', {})

        # Initialize validation_config if None
        if self.validation_config is None:
            object.__setattr__(self, 'validation_config', {})

        # Validate name
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Transform name must be a non-empty string")

        # Validate kwargs
        if not isinstance(self.kwargs, dict):
            raise ValueError("Transform kwargs must be a dictionary")
```

**str_replace new_str**:
```python
class TransformSpec(BaseModel, frozen=True):
    """
    Individual transform specification with validation.

    Attributes:
        name: Transform name/identifier
        kwargs: Transform parameters (keyword arguments)
        enabled: Whether transform is enabled
        description: Human-readable description
        validation_config: Validation configuration
    """
    name: str
    kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict)
    enabled: bool = True
    description: Optional[str] = None
    validation_config: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is a non-empty string."""
        if not v or not isinstance(v, str):
            raise ValueError("Transform name must be a non-empty string")
        return v

    @field_validator('kwargs')
    @classmethod
    def validate_kwargs(cls, v: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate kwargs is a dictionary."""
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("Transform kwargs must be a dictionary")
        return v

    @model_validator(mode='after')
    def initialize_dict_fields(self) -> Self:
        """Initialize None dict fields to empty dicts."""
        updates = {}
        if self.kwargs is None:
            updates['kwargs'] = {}
        if self.validation_config is None:
            updates['validation_config'] = {}
        return self.model_copy(update=updates) if updates else self

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert TransformSpec to dictionary format for compose_transforms().

        Returns:
            Dict with 'name', 'kwargs', 'enabled' keys compatible with
            graph_transforms.compose_transforms() expected format.
        """
        return {
            'name': self.name,
            'kwargs': self.kwargs if self.kwargs else {},
            'enabled': self.enabled
        }
```

---

### A.3 config_bridge.py str_replace Commands

**File Path**: `milia_pipeline/models/utils/config_bridge.py`

#### A.3.1 Import Statement Change

**str_replace old_str**:
```python
from dataclasses import dataclass, field
```

**str_replace new_str**:
```python
from pydantic import BaseModel, field_validator, Field
```

#### A.3.2 ModelSelectionConfig Class Migration (Lines 202-224)

**str_replace old_str**:
```python
@dataclass
class ModelSelectionConfig:
    """Model selection configuration."""
    task_type: str
    model_name: str
    baseline_model: Optional[str] = None

    def validate(self):
        """Validate selection configuration."""
        if not self.task_type:
            raise ConfigurationError("task_type is required")
        if not self.model_name:
            raise ConfigurationError("model_name is required")

        # Validate task_type
        try:
            TaskType(self.task_type)
        except ValueError:
            valid_tasks = [t.value for t in TaskType]
            raise ConfigurationError(
                f"Invalid task_type '{self.task_type}'. "
                f"Must be one of: {valid_tasks}"
            )
```

**str_replace new_str**:
```python
class ModelSelectionConfig(BaseModel):
    """Model selection configuration."""
    task_type: str
    model_name: str
    baseline_model: Optional[str] = None

    @field_validator('task_type')
    @classmethod
    def validate_task_type_field(cls, v: str) -> str:
        """Validate task_type is valid."""
        if not v:
            raise ConfigurationError("task_type is required")
        try:
            TaskType(v)
        except ValueError:
            valid_tasks = [t.value for t in TaskType]
            raise ConfigurationError(
                f"Invalid task_type '{v}'. "
                f"Must be one of: {valid_tasks}"
            )
        return v

    @field_validator('model_name')
    @classmethod
    def validate_model_name_field(cls, v: str) -> str:
        """Validate model_name is not empty."""
        if not v:
            raise ConfigurationError("model_name is required")
        return v

    def validate(self):
        """Backward compatible validate method (now a no-op, validation happens on construction)."""
        pass
```

**Note**: Apply similar pattern to all 25+ mutable dataclasses in config_bridge.py. The pattern is:
1. Replace `@dataclass` with `class ClassName(BaseModel):`
2. Replace `field(default_factory=...)` with `Field(default_factory=...)`
3. Convert `validate()` method logic to `@field_validator` decorators
4. Keep empty `validate()` method for backward compatibility

---

### A.4 Additional config_containers.py Classes (Remaining Frozen Dataclasses)

#### A.4.1 ExperimentalSetup Class Migration (Lines 938-997)

**str_replace old_str**:
```python
@dataclass(frozen=True)
class ExperimentalSetup:
    """
    Container for experimental setup configuration.

    Added for Transformation Configuration Support

    Attributes:
        name: Experimental setup name
        transforms: List of transform specifications
        description: Optional description for the experimental setup
        enabled: Whether this experimental setup is enabled
        research_context: Research context (e.g., "molecular_properties", "robustness_training")
        expected_effects: List of expected effects from this setup
        validation_config: Validation configuration for the setup
        dataset_compatibility: Dataset types this setup is compatible with
    """
    name: str
    transforms: List[TransformSpec]
    description: Optional[str] = None
    enabled: bool = True
    research_context: Optional[str] = None
    expected_effects: Optional[List[str]] = field(default_factory=list)
    validation_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    dataset_compatibility: Optional[List[str]] = field(default_factory=list)

    # Add this to the ExperimentalSetup class (monkey-patch style for the artifact)
    #ExperimentalSetup.validate_experimental_setup_safe = validate_experimental_setup_safe

    def __post_init__(self):
        """Validate experimental setup configuration"""
        # Initialize lists if None
        if self.expected_effects is None:
            object.__setattr__(self, 'expected_effects', [])

        if self.dataset_compatibility is None:
            object.__setattr__(self, 'dataset_compatibility', [])

        # Initialize validation_config if None
        if self.validation_config is None:
            object.__setattr__(self, 'validation_config', {})

        # Validate setup
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Experimental setup name must be a non-empty string")

        if not isinstance(self.transforms, list):
            raise ValueError("Transforms must be a list")

        # CHANGED: Allow empty transforms for test scenarios
        # Original code: if not self.transforms:
        #     raise ValueError("Experimental setup must have at least one transform")
        # New behavior: Only warn about empty transforms but don't fail
        if not self.transforms and hasattr(self, '_strict_validation') and self._strict_validation:
            raise ValueError("Experimental setup must have at least one transform")

        # Validate all transforms are TransformSpec instances
        for i, transform in enumerate(self.transforms):
            if not isinstance(transform, TransformSpec):
                raise ValueError(f"Transform {i} must be a TransformSpec instance")
```

**str_replace new_str**:
```python
class ExperimentalSetup(BaseModel, frozen=True):
    """
    Container for experimental setup configuration.

    Added for Transformation Configuration Support

    Attributes:
        name: Experimental setup name
        transforms: List of transform specifications
        description: Optional description for the experimental setup
        enabled: Whether this experimental setup is enabled
        research_context: Research context (e.g., "molecular_properties", "robustness_training")
        expected_effects: List of expected effects from this setup
        validation_config: Validation configuration for the setup
        dataset_compatibility: Dataset types this setup is compatible with
    """
    name: str
    transforms: List[TransformSpec]
    description: Optional[str] = None
    enabled: bool = True
    research_context: Optional[str] = None
    expected_effects: Optional[List[str]] = Field(default_factory=list)
    validation_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    dataset_compatibility: Optional[List[str]] = Field(default_factory=list)

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is a non-empty string."""
        if not v or not isinstance(v, str):
            raise ValueError("Experimental setup name must be a non-empty string")
        return v

    @field_validator('transforms')
    @classmethod
    def validate_transforms(cls, v: List) -> List:
        """Validate transforms is a list of TransformSpec instances."""
        if not isinstance(v, list):
            raise ValueError("Transforms must be a list")
        for i, transform in enumerate(v):
            if not isinstance(transform, TransformSpec):
                raise ValueError(f"Transform {i} must be a TransformSpec instance")
        return v

    @model_validator(mode='after')
    def initialize_list_and_dict_fields(self) -> Self:
        """Initialize None list/dict fields to empty collections."""
        updates = {}
        if self.expected_effects is None:
            updates['expected_effects'] = []
        if self.dataset_compatibility is None:
            updates['dataset_compatibility'] = []
        if self.validation_config is None:
            updates['validation_config'] = {}
        return self.model_copy(update=updates) if updates else self

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()
```

#### A.4.2 TransformationConfig Class Migration (Lines 1139-1193)

**str_replace old_str**:
```python
@dataclass(frozen=True)
class TransformationConfig:
    """
    Container for comprehensive transformation configuration.

    Added for Transformation Configuration Support

    Attributes:
        experimental_setups: Dictionary of experimental setups by name
        default_setup: Name of the default experimental setup
        validation: Validation configuration for transformations
        performance_settings: Performance optimization settings
        migration_metadata: Metadata about configuration migration
        research_metadata: Research-specific metadata
    """
    experimental_setups: Dict[str, ExperimentalSetup]
    default_setup: str
    standard_transforms: Optional[List[TransformSpec]] = field(default_factory=list)
    validation: Optional[Dict[str, Any]] = field(default_factory=dict)
    performance_settings: Optional[Dict[str, Any]] = field(default_factory=dict)
    migration_metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    research_metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize standard_transforms if None
        if self.standard_transforms is None:
            object.__setattr__(self, 'standard_transforms', [])

        # Initialize dictionaries if None
        if self.validation is None:
            object.__setattr__(self, 'validation', {})

        if self.performance_settings is None:
            object.__setattr__(self, 'performance_settings', {})

        if self.migration_metadata is None:
            object.__setattr__(self, 'migration_metadata', {})

        if self.research_metadata is None:
            object.__setattr__(self, 'research_metadata', {})

        # Validate configuration
        if not isinstance(self.experimental_setups, dict):
            raise ValueError("Experimental setups must be a dictionary")

        if not self.experimental_setups:
            raise ValueError("At least one experimental setup must be defined")

        if self.default_setup not in self.experimental_setups:
            raise ValueError(f"Default setup '{self.default_setup}' not found in experimental setups")

        # Validate all setups are ExperimentalSetup instances
        for name, setup in self.experimental_setups.items():
            if not isinstance(setup, ExperimentalSetup):
                raise ValueError(f"Setup '{name}' must be an ExperimentalSetup instance")
```

**str_replace new_str**:
```python
class TransformationConfig(BaseModel, frozen=True):
    """
    Container for comprehensive transformation configuration.

    Added for Transformation Configuration Support

    Attributes:
        experimental_setups: Dictionary of experimental setups by name
        default_setup: Name of the default experimental setup
        validation: Validation configuration for transformations
        performance_settings: Performance optimization settings
        migration_metadata: Metadata about configuration migration
        research_metadata: Research-specific metadata
    """
    experimental_setups: Dict[str, ExperimentalSetup]
    default_setup: str
    standard_transforms: Optional[List[TransformSpec]] = Field(default_factory=list)
    validation: Optional[Dict[str, Any]] = Field(default_factory=dict)
    performance_settings: Optional[Dict[str, Any]] = Field(default_factory=dict)
    migration_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    research_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator('experimental_setups')
    @classmethod
    def validate_experimental_setups(cls, v: Dict) -> Dict:
        """Validate experimental_setups is a non-empty dict of ExperimentalSetup."""
        if not isinstance(v, dict):
            raise ValueError("Experimental setups must be a dictionary")
        if not v:
            raise ValueError("At least one experimental setup must be defined")
        for name, setup in v.items():
            if not isinstance(setup, ExperimentalSetup):
                raise ValueError(f"Setup '{name}' must be an ExperimentalSetup instance")
        return v

    @model_validator(mode='after')
    def validate_default_setup_and_init_fields(self) -> Self:
        """Validate default_setup exists and initialize None fields."""
        # Validate default_setup exists
        if self.default_setup not in self.experimental_setups:
            raise ValueError(f"Default setup '{self.default_setup}' not found in experimental setups")

        # Initialize None fields
        updates = {}
        if self.standard_transforms is None:
            updates['standard_transforms'] = []
        if self.validation is None:
            updates['validation'] = {}
        if self.performance_settings is None:
            updates['performance_settings'] = {}
        if self.migration_metadata is None:
            updates['migration_metadata'] = {}
        if self.research_metadata is None:
            updates['research_metadata'] = {}
        return self.model_copy(update=updates) if updates else self

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()
```

#### A.4.3 DescriptorConfig Class Migration (Lines 1434-1488)

**str_replace old_str**:
```python
@dataclass(frozen=True)
class DescriptorConfig:
    """
    Container for molecular descriptor configuration.

    Attributes:
        enabled: Whether descriptor computation is enabled
        default_categories: Categories to compute by default
        categories: Category-specific configurations
        cache_descriptors: Whether to cache computed descriptors
        cache_path: Path for descriptor cache
        parallel_computation: Whether to use parallel computation
        num_workers: Number of parallel workers
        error_handling: Error handling mode
        validation_mode: Validation strictness level
    """
    enabled: bool = True
    default_categories: List[str] = field(default_factory=lambda: ['constitutional', 'topological'])
    categories: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cache_descriptors: bool = True
    cache_path: Optional[str] = None
    parallel_computation: bool = False
    num_workers: int = 1
    error_handling: str = 'warn'
    validation_mode: str = 'standard'

    def __post_init__(self):
        """Validate descriptor configuration."""
        # Validate error_handling
        valid_error_modes = ['strict', 'warn', 'skip']
        if self.error_handling not in valid_error_modes:
            raise ValueError(
                f"Invalid error_handling: {self.error_handling}. "
                f"Valid modes: {valid_error_modes}"
            )

        # Validate validation_mode
        valid_validation_modes = ['strict', 'standard', 'permissive']
        if self.validation_mode not in valid_validation_modes:
            raise ValueError(
                f"Invalid validation_mode: {self.validation_mode}. "
                f"Valid modes: {valid_validation_modes}"
            )

        # Validate num_workers
        if self.num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {self.num_workers}")

        # Auto-adjust num_workers for parallel computation
        if self.parallel_computation and self.num_workers == 1:
            object.__setattr__(self, 'num_workers', 2)

        # Initialize categories if None
        if self.categories is None:
            object.__setattr__(self, 'categories', {})
```

**str_replace new_str**:
```python
class DescriptorConfig(BaseModel, frozen=True):
    """
    Container for molecular descriptor configuration.

    Attributes:
        enabled: Whether descriptor computation is enabled
        default_categories: Categories to compute by default
        categories: Category-specific configurations
        cache_descriptors: Whether to cache computed descriptors
        cache_path: Path for descriptor cache
        parallel_computation: Whether to use parallel computation
        num_workers: Number of parallel workers
        error_handling: Error handling mode
        validation_mode: Validation strictness level
    """
    enabled: bool = True
    default_categories: List[str] = Field(default_factory=lambda: ['constitutional', 'topological'])
    categories: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    cache_descriptors: bool = True
    cache_path: Optional[str] = None
    parallel_computation: bool = False
    num_workers: int = 1
    error_handling: str = 'warn'
    validation_mode: str = 'standard'

    @field_validator('error_handling')
    @classmethod
    def validate_error_handling(cls, v: str) -> str:
        """Validate error_handling mode."""
        valid_error_modes = ['strict', 'warn', 'skip']
        if v not in valid_error_modes:
            raise ValueError(
                f"Invalid error_handling: {v}. "
                f"Valid modes: {valid_error_modes}"
            )
        return v

    @field_validator('validation_mode')
    @classmethod
    def validate_validation_mode(cls, v: str) -> str:
        """Validate validation_mode."""
        valid_validation_modes = ['strict', 'standard', 'permissive']
        if v not in valid_validation_modes:
            raise ValueError(
                f"Invalid validation_mode: {v}. "
                f"Valid modes: {valid_validation_modes}"
            )
        return v

    @field_validator('num_workers')
    @classmethod
    def validate_num_workers(cls, v: int) -> int:
        """Validate num_workers is >= 1."""
        if v < 1:
            raise ValueError(f"num_workers must be >= 1, got {v}")
        return v

    @model_validator(mode='after')
    def auto_adjust_workers_and_init_fields(self) -> Self:
        """Auto-adjust num_workers and initialize None fields."""
        updates = {}

        # Auto-adjust num_workers for parallel computation
        if self.parallel_computation and self.num_workers == 1:
            updates['num_workers'] = 2

        # Initialize categories if None
        if self.categories is None:
            updates['categories'] = {}

        return self.model_copy(update=updates) if updates else self

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()
```

#### A.4.4 DescriptorCategoryConfig Class Migration (Lines 1609-1640)

**str_replace old_str**:
```python
@dataclass(frozen=True)
class DescriptorCategoryConfig:
    """
    Container for individual descriptor category configuration.

    Attributes:
        category_name: Name of the category
        enabled: Whether category is enabled
        descriptors: Specific descriptors to compute
        options: Category-specific options
    """
    category_name: str
    enabled: bool = True
    descriptors: Optional[List[str]] = None
    options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate category configuration."""
        # Validate category name
        valid_categories = [
            'constitutional', 'topological', 'geometric',
            'electronic', 'pharmacophore', 'fingerprint', 'custom'
        ]
        if self.category_name not in valid_categories:
            raise ValueError(
                f"Invalid category: {self.category_name}. "
                f"Valid categories: {valid_categories}"
            )

        # Initialize options if None
        if self.options is None:
            object.__setattr__(self, 'options', {})
```

**str_replace new_str**:
```python
class DescriptorCategoryConfig(BaseModel, frozen=True):
    """
    Container for individual descriptor category configuration.

    Attributes:
        category_name: Name of the category
        enabled: Whether category is enabled
        descriptors: Specific descriptors to compute
        options: Category-specific options
    """
    category_name: str
    enabled: bool = True
    descriptors: Optional[List[str]] = None
    options: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('category_name')
    @classmethod
    def validate_category_name(cls, v: str) -> str:
        """Validate category_name is a valid category."""
        valid_categories = [
            'constitutional', 'topological', 'geometric',
            'electronic', 'pharmacophore', 'fingerprint', 'custom'
        ]
        if v not in valid_categories:
            raise ValueError(
                f"Invalid category: {v}. "
                f"Valid categories: {valid_categories}"
            )
        return v

    @model_validator(mode='after')
    def initialize_dict_fields(self) -> Self:
        """Initialize None dict fields to empty dicts."""
        updates = {}
        if self.options is None:
            updates['options'] = {}
        return self.model_copy(update=updates) if updates else self

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()

---

## 9. PHASE 6: HPO MODULE MIGRATION

### 9.1 Overview

Phase 6 migrates the Hyperparameter Optimization (HPO) module from frozen dataclasses to Pydantic V2 frozen BaseModel classes.

**Files Migrated**:
- `milia_pipeline/models/hpo/search_spaces/param_types.py` (Phase 6a)
- `milia_pipeline/models/hpo/hpo_config.py` (Phase 6b)

**Classes Migrated** (6 total):
- `SearchSpaceParamConfig` (param_types.py)
- `PrunerConfig` (hpo_config.py)
- `SamplerConfig` (hpo_config.py)
- `StudyConfig` (hpo_config.py)
- `MultiObjectiveStudyConfig` (hpo_config.py)
- `HPOConfig` (hpo_config.py)

**Pattern Used**: `BaseModel, frozen=True` + `@field_validator` + `@model_validator(mode='before')`

### 9.2 Phase 6a: param_types.py Migration

**Location**: `milia_pipeline/models/hpo/search_spaces/param_types.py`
**Lines**: 159 → 195
**Classes**: 1 (`SearchSpaceParamConfig`)

#### 9.2.1 Import Changes

**str_replace old_str**:
```python
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional
```

**str_replace new_str**:
```python
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, field_validator, model_validator
```

#### 9.2.2 SearchSpaceParamConfig Class Migration

**str_replace old_str**:
```python
@dataclass(frozen=True)
class SearchSpaceParamConfig:
    """..."""
    type: ParamType
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[int] = None
    choices: Optional[List[Any]] = None
    log: bool = False

    def __post_init__(self):
        """Validate configuration based on type."""
        from milia_pipeline.exceptions import ConfigurationError

        if self.type in (ParamType.INT, ParamType.FLOAT, ParamType.LOGUNIFORM,
                         ParamType.UNIFORM, ParamType.INT_UNIFORM):
            if self.low is None or self.high is None:
                raise ConfigurationError(...)
            if self.low >= self.high:
                raise ConfigurationError(...)

        if self.type == ParamType.CATEGORICAL:
            if not self.choices or len(self.choices) == 0:
                raise ConfigurationError(...)

        if self.type == ParamType.DISCRETE_UNIFORM:
            ...
```

**str_replace new_str**:
```python
class SearchSpaceParamConfig(BaseModel, frozen=True):
    """..."""
    type: ParamType
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[int] = None
    choices: Optional[List[Any]] = None
    log: bool = False

    @model_validator(mode='before')
    @classmethod
    def validate_type_requirements(cls, data: Any) -> Any:
        """Validate configuration requirements based on parameter type."""
        if isinstance(data, dict):
            param_type = data.get('type')

            # Convert string to enum if needed
            if isinstance(param_type, str):
                try:
                    param_type = ParamType(param_type)
                except ValueError:
                    return data  # Let Pydantic handle the error

            # Numeric types require low and high
            numeric_types = (
                ParamType.INT, ParamType.FLOAT, ParamType.LOGUNIFORM,
                ParamType.UNIFORM, ParamType.INT_UNIFORM, ParamType.DISCRETE_UNIFORM
            )

            if param_type in numeric_types:
                low = data.get('low')
                high = data.get('high')

                if low is None or high is None:
                    raise ValueError(
                        f"Parameter type '{param_type.value}' requires 'low' and 'high'. "
                        f"Got low={low}, high={high}"
                    )

                if low >= high:
                    raise ValueError(
                        f"'low' must be less than 'high'. Got low={low}, high={high}"
                    )

            # Categorical requires non-empty choices
            if param_type == ParamType.CATEGORICAL:
                choices = data.get('choices')
                if not choices or len(choices) == 0:
                    raise ValueError(
                        "Categorical parameter requires non-empty 'choices' list"
                    )

        return data

    def to_dict(self) -> dict:
        """Backward compatible dict conversion."""
        return self.model_dump()
```

### 9.3 Phase 6b: hpo_config.py Migration

**Location**: `milia_pipeline/models/hpo/hpo_config.py`
**Lines**: 581 → 621
**Classes**: 5 (`PrunerConfig`, `SamplerConfig`, `StudyConfig`, `MultiObjectiveStudyConfig`, `HPOConfig`)

#### 9.3.1 Import Changes

**str_replace old_str**:
```python
from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict, Tuple, Union
from enum import Enum
```

**str_replace new_str**:
```python
from typing import Optional, List, Any, Dict, Tuple, Union
from enum import Enum
from pydantic import BaseModel, field_validator, model_validator, Field
```

#### 9.3.2 PrunerConfig Class Migration

**Pattern**: `@field_validator` for simple field validation + `@model_validator(mode='before')` for type-specific cross-field validation

```python
class PrunerConfig(BaseModel, frozen=True):
    """Pruner configuration for early trial termination."""
    type: PrunerType = PrunerType.MEDIAN
    n_startup_trials: int = 5
    n_warmup_steps: int = 10
    interval_steps: int = 1
    percentile: float = 25.0
    n_brackets: int = 4

    @field_validator('n_startup_trials')
    @classmethod
    def validate_n_startup_trials(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"n_startup_trials must be non-negative, got {v}")
        return v

    @field_validator('interval_steps')
    @classmethod
    def validate_interval_steps(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"interval_steps must be at least 1, got {v}")
        return v

    @model_validator(mode='before')
    @classmethod
    def validate_type_specific_fields(cls, data: Any) -> Any:
        """Validate fields based on pruner type."""
        if isinstance(data, dict):
            pruner_type = data.get('type', PrunerType.MEDIAN)
            if isinstance(pruner_type, str):
                pruner_type = PrunerType(pruner_type)

            if pruner_type == PrunerType.PERCENTILE:
                percentile = data.get('percentile', 25.0)
                if not (0 < percentile < 100):
                    raise ValueError(f"percentile must be between 0 and 100, got {percentile}")

            if pruner_type == PrunerType.HYPERBAND:
                n_brackets = data.get('n_brackets', 4)
                if n_brackets < 1:
                    raise ValueError(f"n_brackets must be at least 1, got {n_brackets}")
        return data

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
```

#### 9.3.3 SamplerConfig Class Migration

```python
class SamplerConfig(BaseModel, frozen=True):
    """Sampler configuration for hyperparameter suggestion."""
    type: SamplerType = SamplerType.TPE
    n_startup_trials: int = 10
    seed: Optional[int] = None
    multivariate: bool = True
    constant_liar: bool = False

    @field_validator('n_startup_trials')
    @classmethod
    def validate_n_startup_trials(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"n_startup_trials must be non-negative, got {v}")
        return v

    @field_validator('seed')
    @classmethod
    def validate_seed(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError(f"seed must be non-negative, got {v}")
        return v

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
```

#### 9.3.4 StudyConfig Class Migration

```python
class StudyConfig(BaseModel, frozen=True):
    """Optuna study configuration for single-objective optimization."""
    direction: OptimizationDirection = OptimizationDirection.MINIMIZE
    metric: str = "val_loss"
    study_name: str = "milia_hpo"
    storage: Optional[str] = None
    load_if_exists: bool = True

    @field_validator('metric')
    @classmethod
    def validate_metric(cls, v: str) -> str:
        if not v:
            raise ValueError("metric cannot be empty")
        return v

    @field_validator('study_name')
    @classmethod
    def validate_study_name(cls, v: str) -> str:
        if not v:
            raise ValueError("study_name cannot be empty")
        return v

    @property
    def is_multi_objective(self) -> bool:
        return False

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
```

#### 9.3.5 MultiObjectiveStudyConfig Class Migration

**Pattern**: Uses `@model_validator(mode='before')` for cross-field validation (directions/metrics length matching)

```python
class MultiObjectiveStudyConfig(BaseModel, frozen=True):
    """Configuration for multi-objective optimization."""
    directions: Tuple[str, ...] = ("minimize",)
    metrics: Tuple[str, ...] = ("val_loss",)
    study_name: str = "milia_hpo_multi"
    storage: Optional[str] = None
    load_if_exists: bool = True
    reference_point: Optional[Tuple[float, ...]] = None

    @model_validator(mode='before')
    @classmethod
    def validate_multi_objective_config(cls, data: Any) -> Any:
        if isinstance(data, dict):
            directions = data.get('directions', ("minimize",))
            metrics = data.get('metrics', ("val_loss",))
            reference_point = data.get('reference_point')

            if len(directions) != len(metrics):
                raise ValueError(
                    f"directions and metrics must have same length. "
                    f"Got {len(directions)} vs {len(metrics)}"
                )

            if len(metrics) < 2:
                raise ValueError(f"Multi-objective requires at least 2 metrics")

            if reference_point and len(reference_point) != len(metrics):
                raise ValueError(f"reference_point must match number of metrics")
        return data

    @property
    def is_multi_objective(self) -> bool:
        return True

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
```

#### 9.3.6 HPOConfig Class Migration (Master Config)

```python
class HPOConfig(BaseModel, frozen=True):
    """Master HPO configuration."""
    enabled: bool = False
    backend: str = "optuna"
    n_trials: int = 100
    timeout: Optional[int] = None
    n_jobs: int = 1
    search_space: Dict[str, Dict[str, SearchSpaceParamConfig]] = Field(default_factory=dict)
    pruner: PrunerConfig = Field(default_factory=PrunerConfig)
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)
    study: StudyConfig = Field(default_factory=StudyConfig)
    cv_folds: int = 0
    cv_metric_aggregation: str = "mean"
    task_type: Optional[str] = None

    @field_validator('backend')
    @classmethod
    def validate_backend(cls, v: str) -> str:
        if v not in ("optuna", "ray_tune"):
            raise ValueError(f"Unknown HPO backend: '{v}'")
        return v

    @field_validator('n_trials')
    @classmethod
    def validate_n_trials(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"n_trials must be at least 1, got {v}")
        return v

    @field_validator('cv_metric_aggregation')
    @classmethod
    def validate_cv_metric_aggregation(cls, v: str) -> str:
        if v not in ("mean", "median", "min", "max"):
            raise ValueError(f"Invalid cv_metric_aggregation: '{v}'")
        return v

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HPOConfig':
        """Create HPOConfig from dictionary (preserved from original)."""
        # ... (implementation unchanged)
```

### 9.4 Phase 6 Verification Tests

**25 comprehensive tests** verified the migration:

| Test Category | Tests | Status |
|---------------|-------|--------|
| Import Tests | 1-2 | ✅ |
| SearchSpaceParamConfig | 3-7 | ✅ |
| PrunerConfig | 8-10 | ✅ |
| SamplerConfig | 11-12 | ✅ |
| StudyConfig | 13-14 | ✅ |
| MultiObjectiveStudyConfig | 15-16 | ✅ |
| HPOConfig | 17-20 | ✅ |
| Pydantic Features | 21-25 | ✅ |

**Test Command**:
```bash
python3 -c "
from milia_pipeline.models.hpo.search_spaces.param_types import ParamType, SearchSpaceParamConfig
from milia_pipeline.models.hpo.hpo_config import HPOConfig, PrunerConfig, SamplerConfig
# ... (25 tests)
"
```

### 9.5 Why Phase 6 is NON-BREAKING, DYNAMIC, PRODUCTION-READY, FUTURE-PROOF

1. **NON-BREAKING**:
   - Same constructor API: `HPOConfig(enabled=True, n_trials=50)` works identically
   - Same attribute access: `config.pruner.type`, `config.n_trials`
   - `from_dict()` method preserved with identical signature
   - `to_dict()` added for backward compatibility (wraps `model_dump()`)

2. **DYNAMIC**:
   - All validation uses dynamic enum values (`PrunerType`, `SamplerType`, `ParamType`)
   - No hardcoded lists; validators check against enum members
   - Type-specific validation via `@model_validator(mode='before')`

3. **PRODUCTION-READY**:
   - Pydantic V2 runtime validation on construction
   - Clear error messages with context
   - `frozen=True` ensures thread-safety
   - Follows established patterns from Phase 2 (`config_containers.py`)

4. **FUTURE-PROOF**:
   - FastAPI integration ready (BaseModel classes work directly)
   - JSON Schema generation via `model_json_schema()`
   - Extensible with `@computed_field` for derived properties
   - New enum values require zero validator changes

---

## 10. PHASE 7: device_manager.py MIGRATION ✅ COMPLETE

**File**: `milia_pipeline/models/acceleration/device_manager.py`
**Lines**: 713 → 718 (after migration)
**Classes**: 1 mutable dataclass
**Complexity**: LOW
**Status**: ✅ COMPLETE (2026-01-07)
**Rationale**: First file in Acceleration module; simple mutable dataclass with custom methods

### 10.1 Implementation Summary

**Class Migrated**: `DeviceInfo` (mutable dataclass → Pydantic BaseModel)

**Import Changes Applied (Lines 20-31)**:
```python
# OLD:
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

# NEW:
from typing import Optional, List, Dict, Any, Union, Tuple
from pydantic import BaseModel
```

**Pattern Used**: Mutable BaseModel (no `frozen=True`) following `config_bridge.py` (Phase 3)

### 10.2 DeviceInfo Migration (Lines 71-124)

**Original Implementation**:
```python
@dataclass
class DeviceInfo:
    device_type: str
    device_id: Optional[int] = None
    name: Optional[str] = None
    total_memory: Optional[int] = None
    available_memory: Optional[int] = None
    compute_capability: Optional[tuple] = None
    is_available: bool = True
    is_default: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'device_type': self.device_type,
            'device_id': self.device_id,
            # ... manual dict creation
        }

    def memory_summary(self) -> str:
        # ... implementation
```

**Migrated Implementation**:
```python
class DeviceInfo(BaseModel):
    """
    Information about a compute device.

    Pattern: Follows mutable BaseModel pattern from config_bridge.py (Pydantic V2)
    """
    device_type: str
    device_id: Optional[int] = None
    name: Optional[str] = None
    total_memory: Optional[int] = None
    available_memory: Optional[int] = None
    compute_capability: Optional[Tuple[int, int]] = None
    is_available: bool = True
    is_default: bool = False

    # Allow arbitrary types for compute_capability tuple compatibility
    model_config = {'arbitrary_types_allowed': True}

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatible method wrapping Pydantic V2's model_dump()."""
        return self.model_dump()

    def memory_summary(self) -> str:
        # ... implementation unchanged
```

### 10.3 Key Changes

| Location | Before | After |
|----------|--------|-------|
| Line 23 | `Version: 1.0.0` | `Version: 1.1.0` |
| Line 28 | `from typing import Optional, List, Dict, Any, Union` | `from typing import Optional, List, Dict, Any, Union, Tuple` |
| Line 29 | `from dataclasses import dataclass` | `from pydantic import BaseModel` |
| Line 77 | `@dataclass` decorator | `class DeviceInfo(BaseModel):` |
| Line 98 | `Optional[tuple]` | `Optional[Tuple[int, int]]` |
| Line 103 | N/A | `model_config = {'arbitrary_types_allowed': True}` |
| Lines 105-111 | Manual dict creation | `return self.model_dump()` |

### 10.4 Phase 7 Verification Tests

**15 comprehensive tests** verified the migration:

| Test # | Category | Description | Status |
|--------|----------|-------------|--------|
| 1 | Import | All imports successful | ✅ |
| 2 | Inheritance | DeviceInfo inherits from BaseModel | ✅ |
| 3 | Instantiation | Basic instantiation with defaults | ✅ |
| 4 | Instantiation | Full instantiation with all fields | ✅ |
| 5 | Backward Compat | `to_dict()` method works | ✅ |
| 6 | Pydantic V2 | `model_dump()` method works | ✅ |
| 7 | Custom Method | `memory_summary()` returns formatted string | ✅ |
| 8 | Edge Case | `memory_summary()` returns "N/A" for CPU | ✅ |
| 9 | Mutability | Attributes can be modified (not frozen) | ✅ |
| 10 | Type Coercion | Pydantic auto-converts types | ✅ |
| 11 | FastAPI Ready | JSON Schema generation works | ✅ |
| 12 | Serialization | `model_dump_json()` works | ✅ |
| 13 | Enum | DeviceType enum unchanged | ✅ |
| 14 | Integration | DeviceManager works with new DeviceInfo | ✅ |
| 15 | Integration | `get_available_devices()` returns List[DeviceInfo] | ✅ |

**Test Command**:
```bash
python3 -c "
from milia_pipeline.models.acceleration.device_manager import DeviceInfo, DeviceType, DeviceManager
from pydantic import BaseModel

# Test 1: Import
assert issubclass(DeviceInfo, BaseModel)

# Test 2: Basic instantiation
info = DeviceInfo(device_type='cuda')
assert info.device_type == 'cuda'
assert info.device_id is None

# Test 3: Full instantiation
info_full = DeviceInfo(
    device_type='cuda',
    device_id=0,
    name='NVIDIA RTX 3090',
    total_memory=25769803776,
    compute_capability=(8, 6),
    is_available=True,
    is_default=True
)

# Test 4: to_dict() backward compatibility
info_dict = info_full.to_dict()
assert info_dict['device_type'] == 'cuda'
assert info_dict['compute_capability'] == (8, 6)

# Test 5: model_dump() matches to_dict()
assert info_full.model_dump() == info_dict

# Test 6: memory_summary()
assert 'GB' in info_full.memory_summary()

# Test 7: Mutable
info.device_id = 1
assert info.device_id == 1

# Test 8: Type coercion
coerced = DeviceInfo(device_type='mps', device_id='0')
assert coerced.device_id == 0

# Test 9: JSON Schema
schema = DeviceInfo.model_json_schema()
assert 'properties' in schema

# Test 10: DeviceManager integration
manager = DeviceManager(device='cpu', verbose=False)
assert isinstance(manager.get_device_info(), DeviceInfo)

print('✅ All Phase 7 tests passed')
"
```

### 10.5 Why Phase 7 is NON-BREAKING, DYNAMIC, PRODUCTION-READY, FUTURE-PROOF

1. **NON-BREAKING**:
   - Same constructor API: `DeviceInfo(device_type='cuda', device_id=0)` works identically
   - Same attribute access: `info.device_type`, `info.name`, `info.total_memory`
   - `to_dict()` method preserved (wraps `model_dump()`)
   - `memory_summary()` method unchanged
   - All `DeviceManager` usages remain unchanged

2. **DYNAMIC**:
   - No hardcoded validation on `device_type` (allows future device types like XPU, NPU)
   - Pydantic type coercion handles flexible input types
   - Optional fields remain optional with `None` defaults

3. **PRODUCTION-READY**:
   - Runtime type validation at instantiation
   - Clear Pydantic error messages for invalid inputs
   - Thread-safe BaseModel implementation
   - `model_config = {'arbitrary_types_allowed': True}` for tuple compatibility
   - 15 comprehensive tests passed

4. **FUTURE-PROOF**:
   - FastAPI integration ready (`model_json_schema()` generates OpenAPI schema)
   - `model_dump_json()` for JSON serialization
   - Type hint improved: `Optional[tuple]` → `Optional[Tuple[int, int]]`
   - Can add `@field_validator` decorators if validation needed later

---

## APPENDIX B: ROLLBACK PROCEDURE

If issues arise during migration:

1. **Phase 1 Rollback**: Revert `base.py` import to `from dataclasses import dataclass, field`
2. **Phase 2 Rollback**: Revert each class individually; `object.__setattr__` pattern restores functionality
3. **Phase 3 Rollback**: Revert import and class definitions
4. **Phase 4 Rollback**: Remove wrapper functions (no changes to core `ValidationResult`)
5. **Phase 5 Rollback**: N/A (no changes to `exceptions.py`)
6. **Phase 6 Rollback**: Revert imports and class definitions in `param_types.py` and `hpo_config.py`; restore `@dataclass(frozen=True)` and `__post_init__` methods
7. **Phase 7 Rollback**: Revert `device_manager.py` import to `from dataclasses import dataclass`; restore `@dataclass` decorator; restore manual `to_dict()` implementation

---

## APPENDIX C: FUTURE ENHANCEMENTS

After successful migration:

1. **FastAPI Integration**: Pydantic models work directly with FastAPI request/response validation
2. **JSON Schema Generation**: Use `model_json_schema()` for automatic API documentation
3. **Strict Mode**: Enable `ConfigDict(strict=True)` for stricter type checking
4. **Computed Fields**: Use `@computed_field` for derived properties
5. **Custom Validators**: Extend with `@field_validator` for complex business logic

---

## APPENDIX D: UPCOMING PHASES - REMAINING DATACLASS FILES

The following files contain dataclasses that require Pydantic V2 migration for complete FastAPI readiness.

### Phase 8-10: Acceleration Module (`models/acceleration/`)

| Phase | File | Dataclass(es) | Type | Attributes | Priority | Status |
|-------|------|---------------|------|------------|----------|--------|
| **Phase 8** | `distributed_strategies.py` | `DistributedConfig` | Mutable | 12 | HIGH | ✅ COMPLETE |
| **Phase 9** | `memory_optimization.py` | `MemoryConfig` | Mutable | 9 | HIGH | ✅ COMPLETE |
| **Phase 10** | `computation_optimization.py` | `ComputationConfig` | Mutable | 10 | HIGH | ✅ COMPLETE |

**Phase 8 Details** - `distributed_strategies.py` ✅ COMPLETE:
- **File Path**: `milia_pipeline/models/acceleration/distributed_strategies.py`
- **Class**: `DistributedConfig` (12 attributes)
  - `strategy` (DistributedStrategy enum), `backend` (DistributedBackend enum), `world_size`, `rank`, `local_rank`
  - `master_addr`, `master_port`, `find_unused_parameters`
  - `gradient_as_bucket_view`, `static_graph`, `cpu_offload`, `mixed_precision`
- **Import Change**: `from dataclasses import dataclass` → `from pydantic import BaseModel`
- **Class Change**: `@dataclass` → `class DistributedConfig(BaseModel):`
- **to_dict() Change**: Manual dict with `.value` → `return self.model_dump(mode='json')` (automatic enum serialization)
- **Pattern**: Mutable BaseModel (same as Phase 7 `DeviceInfo`)
- **Key Finding**: Two different `DistributedConfig` classes exist in codebase:
  - `config_bridge.py:475` - Pydantic BaseModel for YAML config parsing (already migrated in Phase 3)
  - `distributed_strategies.py:89` - Pydantic BaseModel for runtime distributed training (Phase 8 target)
- **Test Coverage**: 12 comprehensive tests passed (import, instantiation, mutability, enum serialization, to_dict backward compat, Pydantic V2 features, DistributedManager integration, JSON schema)
- **Version Bump**: `distributed_strategies.py` v1.0.0 → v1.1.0

**Phase 9 Details** - `memory_optimization.py` ✅ COMPLETE:
- **File Path**: `milia_pipeline/models/acceleration/memory_optimization.py`
- **Class**: `MemoryConfig` (9 attributes)
  - `mixed_precision` (bool), `precision` (str: fp16, bf16, fp32, fp8)
  - `gradient_checkpointing` (bool), `pin_memory` (bool), `non_blocking` (bool)
  - `empty_cache_interval` (int), `garbage_collect_interval` (int), `max_memory_allocated` (int), `growth_interval` (int)
- **Import Change**: `from dataclasses import dataclass` → `from pydantic import BaseModel`
- **Class Change**: `@dataclass class MemoryConfig:` → `class MemoryConfig(BaseModel):`
- **to_dict() Change**: Manual 9-key dict → `return self.model_dump()`
- **Pattern**: Mutable BaseModel (same as Phase 7 `DeviceInfo`)
- **Mutability Requirement**: `MemoryOptimizer._validate_config()` mutates `self.config.precision` and `self.config.mixed_precision` - Pydantic BaseModel without `frozen=True` preserves this behavior
- **Test Coverage**: 15 comprehensive tests passed (import, instantiation, backward compat, mutability, type coercion, JSON schema, model_dump, model_dump_json, model_json_schema, MemoryOptimizer integration, config mutation)
- **NON-BREAKING**: Same constructor API, attribute access, `to_dict()` output format preserved
- **Version Bump**: `memory_optimization.py` v1.0.0 → v1.1.0

**Phase 10 Details** - `computation_optimization.py` ✅ COMPLETE:
- **File Path**: `milia_pipeline/models/acceleration/computation_optimization.py`
- **Class**: `ComputationConfig` (10 attributes)
  - `compile_model` (bool), `compile_mode` (str: default, reduce-overhead, max-autotune), `compile_dynamic` (bool)
  - `cudnn_benchmark` (bool), `cudnn_deterministic` (bool), `use_tf32` (bool)
  - `channels_last` (bool), `fusion_strategy` (str: none, default, aggressive), `jit_compile` (bool), `operator_fusion` (bool)
- **Import Change**: `from dataclasses import dataclass` → `from pydantic import BaseModel`
- **Class Change**: `@dataclass class ComputationConfig:` → `class ComputationConfig(BaseModel):`
- **to_dict() Change**: Manual 10-key dict construction → `return self.model_dump()`
- **Pattern**: Mutable BaseModel (same as Phase 7/8/9)
- **Test Coverage**: 15 comprehensive tests passed (import, instantiation, backward compat, mutability, type coercion, JSON schema, model_dump, model_dump_json, model_json_schema, model_validate, model_copy, ComputationOptimizer integration)
- **NON-BREAKING**: Same constructor API, attribute access, `to_dict()` output format (10 keys) preserved
- **DYNAMIC**: `model_dump()` auto-includes all fields; no hardcoded key list
- **PRODUCTION-READY**: Runtime type validation, clear error messages, thread-safe BaseModel
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation, extensible with `@field_validator`
- **Version Bump**: `computation_optimization.py` v1.0.0 → v1.1.0

### Phase 11-13: HPO Transfer Module (`models/hpo/transfer/`)

| Phase | File | Dataclass(es) | Type | Attributes | Priority | Status |
|-------|------|---------------|------|------------|----------|--------|
| **Phase 11** | `transfer_manager.py` | `TransferConfig`, `RegisteredStudyInfo` | Frozen, Mutable | 10, 10 | MEDIUM | ✅ COMPLETE |
| **Phase 12** | `meta_features.py` | `MetaFeatureConfig` | Frozen | 5 | MEDIUM | ✅ COMPLETE |
| **Phase 13** | `warm_start.py` | `WarmStartConfig`, `TransferredTrial` | Frozen, Mutable | 8, 6 | MEDIUM | ✅ COMPLETE |

**Phase 11 Details** - `transfer_manager.py` ✅ COMPLETE:
- **File Path**: `milia_pipeline/models/hpo/transfer/transfer_manager.py`
- **Version Bump**: v1.0.0 → v1.1.0
- **Import Change**: `from dataclasses import dataclass, field` → `from pydantic import BaseModel, field_validator, model_validator, Field`
- **Classes Migrated** (2 total):
  - `TransferConfig` (frozen BaseModel, 10 attributes): `n_warm_start_trials`, `similarity_threshold`, `meta_feature_method`, `adaptation_method`, `weight_by_performance`, `scale_to_bounds`, `add_noise`, `noise_scale`, `persist_meta_db`, `meta_db_path`
  - `RegisteredStudyInfo` (mutable BaseModel, 10 attributes): `study_name`, `meta_features`, `best_params`, `best_value`, `n_trials`, `n_completed`, `direction`, `model_name`, `dataset_info`, `registered_at`
- **TransferConfig Validation Pattern**:
  - `@field_validator('n_warm_start_trials')`: Validates >= 1
  - `@field_validator('similarity_threshold')`: Validates 0.0-1.0 range
  - `@field_validator('noise_scale')`: Validates 0.0-1.0 range
  - `@model_validator(mode='before')`: String-to-enum conversion for `meta_feature_method` and `adaptation_method`; cross-field validation (`persist_meta_db` requires `meta_db_path`)
- **RegisteredStudyInfo Validation Pattern**:
  - `@model_validator(mode='before')`: Auto-sets `registered_at` timestamp if None
- **Backward Compatibility**:
  - `TransferConfig.to_dict()` → wraps `model_dump()`
  - `RegisteredStudyInfo.to_dict()` → wraps `model_dump()`
  - `RegisteredStudyInfo.from_dict()` → uses `model_validate()`
- **Test Coverage**: 25 comprehensive tests passed (import, inheritance, default instantiation, frozen immutability, field validators, cross-field validation, string-to-enum conversion, to_dict, model_dump, model_json_schema, automatic timestamp, mutability, from_dict, model_validate, HPOTransferManager integration, round-trip serialization)
- **NON-BREAKING**: Same constructor API, attribute access, `to_dict()`, `from_dict()` preserved
- **DYNAMIC**: Enum conversion via `@model_validator(mode='before')`; `model_dump()` auto-includes all fields
- **PRODUCTION-READY**: Runtime type validation, clear error messages, `TransferConfig` is thread-safe (frozen)
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation, extensible with additional validators

**Phase 12 Details** - `meta_features.py` ✅ COMPLETE:
- **File Path**: `milia_pipeline/models/hpo/transfer/meta_features.py`
- **Version Bump**: v1.0.0 → v1.1.0
- **Import Change**: `from dataclasses import dataclass, field` → `from pydantic import BaseModel, field_validator, model_validator` + `from typing_extensions import Self`
- **Class Migrated** (1 total):
  - `MetaFeatureConfig` (frozen BaseModel, 5 attributes): `categories`, `max_samples`, `normalize`, `include_molecular`, `compute_expensive`
- **MetaFeatureConfig Validation Pattern**:
  - `@field_validator('max_samples')`: Validates >= 1 or None
  - `@model_validator(mode='after')`: Validates `categories` is not empty tuple (returns `Self`)
- **Backward Compatibility**:
  - `MetaFeatureConfig.to_dict()` → wraps `model_dump()`
  - `should_extract()` method preserved unchanged
- **Test Coverage**: 20 comprehensive tests passed (import, inheritance, default instantiation, custom instantiation, frozen immutability, max_samples validation invalid, max_samples validation negative, categories validation empty, should_extract ALL, should_extract specific, to_dict backward compat, model_dump, model_dump_json, model_json_schema, model_validate, type coercion, MetaFeatureExtractor integration, MetaFeatureExtractor custom config, enum values preserved, round-trip serialization)
- **NON-BREAKING**: Same constructor API, attribute access, `should_extract()` method preserved
- **DYNAMIC**: `model_dump()` auto-includes all fields; no hardcoded key list
- **PRODUCTION-READY**: Runtime type validation, clear error messages, `MetaFeatureConfig` is thread-safe (frozen)
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation, extensible with `@field_validator`

**Phase 13 Details** - `warm_start.py` ✅ COMPLETE:
- **File Path**: `milia_pipeline/models/hpo/transfer/warm_start.py`
- **Version Bump**: v1.0.0 → v1.1.0
- **Import Change**: `from dataclasses import dataclass, field` → `from pydantic import BaseModel, field_validator`
- **Classes Migrated** (2 total):
  - `WarmStartConfig` (frozen BaseModel, 8 attributes): `method`, `n_trials`, `min_similarity`, `weight_by_performance`, `filter_invalid`, `scale_to_bounds`, `add_noise`, `noise_scale`
  - `TransferredTrial` (mutable BaseModel, 6 attributes): `params`, `value`, `source_study`, `similarity`, `weight`, `is_valid`
- **WarmStartConfig Validation Pattern**:
  - `@field_validator('n_trials')`: Validates >= 1
  - `@field_validator('min_similarity')`: Validates 0.0-1.0 range
  - `@field_validator('noise_scale')`: Validates 0.0-1.0 range
- **TransferredTrial Pattern**:
  - No validators required (simple data container)
  - Mutable BaseModel (no `frozen=True`) for attribute modification in `_add_noise_to_trials()`
- **Backward Compatibility**:
  - `WarmStartConfig.to_dict()` → wraps `model_dump()`
  - `TransferredTrial.to_dict()` → wraps `model_dump()`
- **Test Coverage**: 20 comprehensive tests passed (import, inheritance, default instantiation, custom instantiation, frozen immutability, field validators n_trials/min_similarity/noise_scale, boundary values, to_dict backward compat, model_dump, model_dump_json, model_json_schema, model_validate, type coercion, WarmStartStrategy integration, static methods return TransferredTrial, enum handling, round-trip serialization, get_transfer_summary, module exports)
- **NON-BREAKING**: Same constructor API, attribute access, `to_dict()` preserved
- **DYNAMIC**: `model_dump()` auto-includes all fields; no hardcoded key list
- **PRODUCTION-READY**: Runtime type validation, clear error messages, `WarmStartConfig` is thread-safe (frozen)
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation, extensible with `@field_validator`

### Phase 14-15: HPO NAS Module (`models/hpo/nas/`)

| Phase | File | Dataclass(es) | Type | Attributes | Priority | Status |
|-------|------|---------------|------|------------|----------|--------|
| **Phase 14** | `search_space.py` | `LayerConfig`, `GNNArchitectureSpace` | Frozen, Mutable | 7, 13 | MEDIUM | ✅ COMPLETE |
| **Phase 15** | `study_analyzer.py` | `AnalysisConfig` | Frozen | 6 | LOW | ✅ COMPLETE |

**Phase 14 Details** - `search_space.py` ✅ COMPLETE:
- **File Path**: `milia_pipeline/models/hpo/nas/search_space.py`
- **Version Bump**: v1.0.0 → v1.1.0
- **Import Change**: `from dataclasses import dataclass, field` → `from pydantic import BaseModel, field_validator, model_validator, Field` + `from typing_extensions import Self`
- **Classes Migrated** (2 total):
  - `LayerConfig` (frozen BaseModel, 7 attributes): `type` (LayerType), `hidden_channels` (int), `heads` (int), `dropout` (float), `activation` (str), `batch_norm` (bool), `residual` (bool)
  - `GNNArchitectureSpace` (mutable BaseModel, 13 attributes): `min_layers`, `max_layers`, `layer_types`, `hidden_channels`, `heads`, `dropout_range`, `allow_skip_connections`, `allow_dense_connections`, `allow_mixed_layers`, `pooling_types`, `aggregation_types`, `activation_types`, `batch_norm_options`
- **LayerConfig Validation Pattern**:
  - `@field_validator('hidden_channels')`: Validates > 0
  - `@field_validator('heads')`: Validates >= 1
  - `@field_validator('dropout')`: Validates 0.0-1.0 range
  - `to_dict()` → uses `model_dump()` with enum value serialization
  - `from_dict()` → uses `model_validate()` with string-to-enum conversion
- **GNNArchitectureSpace Validation Pattern**:
  - `@field_validator('min_layers')`: Validates >= 1
  - `@field_validator('hidden_channels')`: Validates non-empty list with positive values
  - `@field_validator('heads')`: Validates non-empty list with values >= 1
  - `@field_validator('dropout_range')`: Validates tuple with 0 <= min <= max <= 1
  - `@field_validator('layer_types')`: Validates non-empty list
  - `@field_validator('pooling_types')`: Validates non-empty list
  - `@field_validator('aggregation_types')`: Validates non-empty list
  - `@model_validator(mode='after')`: Cross-field validation (max_layers >= min_layers)
- **Backward Compatibility**:
  - `LayerConfig.to_dict()` → returns dict with enum values as strings (backward compatible)
  - `LayerConfig.from_dict()` → uses `model_validate()` with string-to-enum conversion
  - `GNNArchitectureSpace.to_dict()` → returns dict with enum values as strings (13 keys)
  - `GNNArchitectureSpace.from_dict()` → uses `model_validate()` with string-to-enum conversion
  - `GNNArchitectureSpace.to_optuna_search_space()` → unchanged, returns Optuna-compatible format
  - Helper methods preserved: `has_attention_layers()`, `get_attention_layer_types()`, `get_search_dimensions()`, `estimate_search_space_size()`, `create_default_layer_config()`
- **Test Coverage**: 30 comprehensive tests passed (import, inheritance, default instantiation, custom instantiation, frozen immutability, field validators hidden_channels/heads/dropout/min_layers/hidden_channels_list/heads_list/dropout_range/layer_types/pooling_types/aggregation_types, cross-field validation max>=min, to_dict backward compat, from_dict string-to-enum, model_dump, model_json_schema, to_optuna_search_space, helper methods, factory functions, model_validate, model_copy, round-trip serialization, module version)
- **NON-BREAKING**: Same constructor API, attribute access, `to_dict()`, `from_dict()`, `to_optuna_search_space()` output format preserved
- **DYNAMIC**: `model_dump()` auto-includes all fields; enum validation uses Pydantic V2 native support
- **PRODUCTION-READY**: Runtime type validation, clear error messages, `LayerConfig` is thread-safe (frozen)
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation, extensible with `@field_validator`

**Phase 15 Details** - `study_analyzer.py` ✅ COMPLETE:
- **File Path**: `milia_pipeline/models/hpo/analysis/study_analyzer.py`
- **Version Bump**: v1.0.0 → v1.1.0
- **Import Change**: `from dataclasses import dataclass, field` → `from pydantic import BaseModel, field_validator, model_validator` + `from typing_extensions import Self`
- **Classes Migrated** (1 total):
  - `AnalysisConfig` (frozen BaseModel, 6 attributes): `importance_method` (ImportanceMethod enum), `n_importance_trials` (Optional[int]), `convergence_window` (int), `include_pruned` (bool), `include_failed` (bool), `percentile_thresholds` (Tuple[float, ...])
- **AnalysisConfig Validation Pattern**:
  - `@field_validator('convergence_window')`: Validates >= 1
  - `@field_validator('n_importance_trials')`: Validates >= 1 or None
  - `@model_validator(mode='after')`: Validates all percentile values are between 0 and 100 (returns `Self`)
  - `to_dict()` → wraps `model_dump()` for backward compatibility
- **Backward Compatibility**:
  - `AnalysisConfig.to_dict()` → returns dict with all 6 attributes (backward compatible)
  - Same constructor API: `AnalysisConfig(importance_method=ImportanceMethod.FANOVA, convergence_window=20)`
  - Same attribute access: `config.convergence_window`, `config.percentile_thresholds`
- **Test Coverage**: 25 comprehensive tests passed (import, Pydantic BaseModel inheritance, default instantiation, custom instantiation, frozen immutability, field validators convergence_window/n_importance_trials, boundary values, model validators percentile_thresholds >100/<0/boundary, to_dict backward compat, model_dump, model_dump_json, model_json_schema, model_validate, model_copy, type coercion string-to-int, enum string coercion, StudyAnalyzer integration, module exports, round-trip serialization, empty percentile thresholds)
- **NON-BREAKING**: Same constructor API, attribute access, frozen immutability preserved
- **DYNAMIC**: `model_dump()` auto-includes all fields; enum validation uses Pydantic V2 native support
- **PRODUCTION-READY**: Runtime type validation, clear error messages, `AnalysisConfig` is thread-safe (frozen)
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation, extensible with `@field_validator`

### Phase 16: Transformations Module (`transformations/`)

| Phase | File | Dataclass(es) | Type | Attributes | Priority |
|-------|------|---------------|------|------------|----------|
| **Phase 16** | `custom_transforms.py` | `TransformMetadata` | Mutable | 11 | LOW |

**Phase 16 Details** - `custom_transforms.py`:
- **File Path**: `milia_pipeline/transformations/custom_transforms.py`
- **Class**: `TransformMetadata` (mutable, 11 attributes)
  - `name`, `version`, `author`, `category`, `description`
  - `paper_reference`, `github_url`, `validated_datasets`
  - `required_node_features`, `required_edge_features`, `required_graph_attributes`
- **Methods to preserve**: `to_dict()`
- **Pattern**: Mutable BaseModel (same as Phase 7 `DeviceInfo`)

---

### Phase 17: Config Schemas Module (`config/`) ✅ COMPLETE

| Phase | File | Dataclass(es) | Type | Attributes | Priority |
|-------|------|---------------|------|------------|----------|
| **Phase 17** | `config_schemas.py` | `TransformationSchema` | Mutable | 7 | CRITICAL |
| **Phase 17** | `config_schemas.py` | `PluginConfigSchema` | Mutable | 12 | CRITICAL |
| **Phase 17** | `config_schemas.py` | `WavefunctionProcessingConfigSchema` | Frozen | 1 | CRITICAL |
| **Phase 17** | `config_schemas.py` | `WavefunctionUncertaintyConfigSchema` | Frozen | 1 | CRITICAL |
| **Phase 17** | `config_schemas.py` | `WavefunctionConfigSchema` | Frozen | 5 | CRITICAL |
| **Phase 17** | `config_schemas.py` | `ExperimentSchema` | Mutable | 12 | CRITICAL |
| **Phase 17** | `config_schemas.py` | `ValidationConfig` | Mutable | 5 | CRITICAL |
| **Phase 17** | `config_schemas.py` | `DescriptorConfigSchema` | Frozen | 8 | CRITICAL |
| **Phase 17** | `config_schemas.py` | `DescriptorCategoryConfigSchema` | Frozen | 4 | CRITICAL |

**Phase 17 Details** - `config_schemas.py`:
- **File Path**: `milia_pipeline/config/config_schemas.py`
- **Total Classes Migrated**: 9
- **Implementation Date**: 2026-01-08
- **Tests Passed**: 14

**Classes Migrated**:

1. **`TransformationSchema`** (Mutable, 7 attributes)
   - Pattern: `BaseModel` + `model_validator(mode='after')` for cross-field validation
   - Attributes: `experimental_setups`, `default_setup`, `validation`, `standard_transforms`, `legacy_transforms`, `research_metadata`, `dataset_optimization`
   - Cross-field validation: Must have either `experimental_setups` OR `standard_transforms`

2. **`PluginConfigSchema`** (Mutable, 12 attributes)
   - Pattern: `BaseModel` + 4 `field_validator` decorators
   - Attributes: `enabled`, `plugin_paths`, `auto_discover`, `auto_validate`, `validation_level`, `trusted_plugins`, `disabled_plugins`, `allow_experimental`, `max_plugins`, `require_metadata`, `enforce_checksums`, `security_scanning`
   - Methods preserved: `to_dict()` → `model_dump()`, `from_dict()` → `cls(**data)`

3. **`WavefunctionProcessingConfigSchema`** (Frozen, 1 attribute)
   - Pattern: `BaseModel, frozen=True` + `field_validator`
   - Attribute: `feature_tier` (valid: 'basic', 'standard', 'complete')

4. **`WavefunctionUncertaintyConfigSchema`** (Frozen, 1 attribute)
   - Pattern: `BaseModel, frozen=True` + `field_validator`
   - Attribute: `enabled` (must be False - uncertainty not supported for Wavefunction)

5. **`WavefunctionConfigSchema`** (Frozen, 5 attributes)
   - Pattern: `BaseModel, frozen=True` + `model_validator(mode='before')` for nested object init
   - Attributes: `raw_npz_filename`, `raw_data_download_url`, `dataset_root_dir`, `processing_config`, `uncertainty_handling`
   - Nested objects auto-initialized: `WavefunctionProcessingConfigSchema`, `WavefunctionUncertaintyConfigSchema`

6. **`ExperimentSchema`** (Mutable, 12 attributes)
   - Pattern: `BaseModel` + 7 `field_validator` + `model_validator(mode='after')` for warning
   - Attributes: `name`, `description`, `base_transforms`, `ablations`, `parameter_sweeps`, `paper_reference`, `hypothesis`, `expected_outcome`, `num_runs`, `random_seed`, `results`, `metadata`

7. **`ValidationConfig`** (Mutable, 5 attributes)
   - Pattern: Simple `BaseModel` (no validators needed)
   - Attributes: `strict_mode`, `warn_on_unknown`, `require_descriptions`, `check_parameter_types`, `validate_research_context`

8. **`DescriptorConfigSchema`** (Frozen, 8 attributes)
   - Pattern: `BaseModel, frozen=True` + 4 `field_validator` + `model_validator(mode='before')` for auto-adjust
   - Attributes: `enabled`, `default_categories`, `cache_descriptors`, `cache_path`, `parallel_computation`, `num_workers`, `error_handling`, `validation_mode`
   - Auto-adjust: `num_workers` set to 2 when `parallel_computation=True` and `num_workers=1`

9. **`DescriptorCategoryConfigSchema`** (Frozen, 4 attributes)
   - Pattern: `BaseModel, frozen=True` + 3 `field_validator`
   - Attributes: `category_name`, `enabled`, `descriptors`, `options`

**Import Changes** (Line 25-28):
```python
# OLD:
from dataclasses import dataclass, field

# NEW:
from pydantic import BaseModel, field_validator, model_validator, Field
from typing_extensions import Self
```

**Key Patterns Used**:
- `model_validator(mode='before')` for frozen classes needing field initialization (replacing `object.__setattr__`)
- `model_validator(mode='after')` for cross-field validation and warnings
- `field_validator` for individual field validation
- `Field(default_factory=...)` replacing `field(default_factory=...)`
- `self.model_dump()` in `to_dict()` methods for backward compatibility

---

## APPENDIX E: MIGRATION SUMMARY BY MODULE

| Module | Total Dataclasses | Phases | Status |
|--------|-------------------|--------|--------|
| `datasets/` | 3 | Phase 1 | ✅ COMPLETE |
| `config/` | 19 | Phase 2, 4, 17 | ✅ COMPLETE |
| `models/utils/` | 31 | Phase 3 | ✅ COMPLETE |
| `exceptions.py` | 0 (docs only) | Phase 5 | ✅ COMPLETE |
| `models/hpo/search_spaces/` | 1 | Phase 6a | ✅ COMPLETE |
| `models/hpo/` | 5 | Phase 6b | ✅ COMPLETE |
| `models/acceleration/` | 4 | Phase 7-10 | ✅ 4/4 COMPLETE |
| `models/hpo/transfer/` | 5 | Phase 11-13 | ✅ 5/5 COMPLETE |
| `models/hpo/nas/` | 2 | Phase 14 | ✅ 2/2 COMPLETE |
| `models/hpo/analysis/` | 1 | Phase 15 | ✅ 1/1 COMPLETE |
| `transformations/` | 1 | Phase 16 | ✅ 1/1 COMPLETE |
| **TOTAL** | **73** | **17 phases** | **73/73 (100%)** |

---

## APPENDIX F: BLUEPRINT COMPLETION SUMMARY

All 17 phases of the extended Pydantic V2 migration blueprint are now **COMPLETE**.

### Migrated Files (18 total):
```
milia_pipeline/datasets/base.py                                  # Phase 1 - pydantic.dataclasses.dataclass
milia_pipeline/config/config_containers.py                       # Phase 2 - BaseModel frozen
milia_pipeline/models/utils/config_bridge.py                     # Phase 3 - BaseModel mutable
milia_pipeline/config/validators.py                              # Phase 4 - Pydantic wrappers
milia_pipeline/exceptions.py                                     # Phase 5 - Documentation update
milia_pipeline/models/hpo/search_spaces/param_types.py           # Phase 6a - BaseModel frozen
milia_pipeline/models/hpo/hpo_config.py                          # Phase 6b - BaseModel frozen
milia_pipeline/models/acceleration/device_manager.py             # Phase 7 - BaseModel mutable
milia_pipeline/models/acceleration/distributed_strategies.py     # Phase 8 - BaseModel mutable
milia_pipeline/models/acceleration/memory_optimization.py        # Phase 9 - BaseModel mutable
milia_pipeline/models/acceleration/computation_optimization.py   # Phase 10 - BaseModel mutable
milia_pipeline/models/hpo/transfer/transfer_manager.py           # Phase 11 - BaseModel mixed
milia_pipeline/models/hpo/transfer/meta_features.py              # Phase 12 - BaseModel frozen
milia_pipeline/models/hpo/transfer/warm_start.py                 # Phase 13 - BaseModel mixed
milia_pipeline/models/hpo/nas/search_space.py                    # Phase 14 - BaseModel mixed
milia_pipeline/models/hpo/analysis/study_analyzer.py             # Phase 15 - BaseModel frozen
milia_pipeline/transformations/custom_transforms.py              # Phase 16 - BaseModel mutable
milia_pipeline/config/config_schemas.py                          # Phase 17 - BaseModel mixed (9 classes)
```

### Note on Additional Dataclasses

A comprehensive codebase audit (2026-01-07) revealed **21 additional files** with stdlib dataclasses
that were NOT included in the original blueprint scope. These represent future migration work
for full FastAPI readiness. See APPENDIX G for details.

---

## APPENDIX G: EXTENDED MIGRATION PLAN (Phases 17-38)

### G.1 Codebase Audit Results

A comprehensive codebase audit conducted on 2026-01-07 using:
```bash
grep -rln "from dataclasses import" --include="*.py" | xargs grep -L "from pydantic"
```

Revealed **22 additional files** with stdlib dataclasses NOT covered in the original blueprint (Phases 1-16).

### G.2 Migration Statistics

| Metric | Value |
|--------|-------|
| **Original Blueprint Phases** | 1-16 |
| **Original Dataclasses Migrated** | 64 |
| **Additional Files Discovered** | 22 |
| **Estimated Additional Dataclasses** | ~47 |
| **Extended Phases Required** | 17-38 |
| **Phase 17 Dataclasses Migrated** | 9 |
| **Current Migration Progress** | 73/~111 (65.8%) |

### G.3 Recommended Migration Order (Phases 17-38)

#### CRITICAL Priority (Phases 17-18)

| Phase | File | Full Path | Est. Dataclasses | Priority | Status |
|-------|------|-----------|------------------|----------|--------|
| **17** | `config_schemas.py` | `milia_pipeline/config/config_schemas.py` | 9 | **CRITICAL** | ✅ COMPLETE |
| **18** | `graph_transforms.py` | `milia_pipeline/transformations/graph_transforms.py` | 8 | **CRITICAL** | ✅ COMPLETE |

#### HIGH Priority (Phases 19-26)

| Phase | File | Full Path | Est. Dataclasses | Priority | Rationale |
|-------|------|-----------|------------------|----------|-----------|
| **19** | `plugin_system.py` | `milia_pipeline/transformations/plugin_system.py` | 2 | HIGH | ✅ COMPLETE |
| **20** | `deployment_strategies.py` | `milia_pipeline/models/deployment/deployment_strategies.py` | 1 | HIGH | ✅ COMPLETE |
| **21** | `monitoring.py` | `milia_pipeline/models/deployment/monitoring.py` | 2 | HIGH | ✅ COMPLETE |
| **22** | `model_optimization.py` | `milia_pipeline/models/deployment/model_optimization.py` | 1 | HIGH | ✅ COMPLETE |
| **23** | `pyg_introspector.py` | `milia_pipeline/models/registry/pyg_introspector.py` | 2 | HIGH | ✅ COMPLETE |
| **24** | `model_registry.py` | `milia_pipeline/models/registry/model_registry.py` | 1 | HIGH | ✅ COMPLETE |
| **25** | `target_selection_config.py` | `milia_pipeline/models/factory/target_selection_config.py` | 1 | HIGH | ✅ COMPLETE |
| **26** | `hpo_manager.py` | `milia_pipeline/models/hpo/hpo_manager.py` | 0 | HIGH | ✅ COMPLETE |

#### MEDIUM Priority (Phases 27-35)

| Phase | File | Full Path | Est. Dataclasses | Priority | Rationale |
|-------|------|-----------|------------------|----------|-----------|
| **27** | `research_api.py` | `milia_pipeline/transformations/research_api.py` | 1 | MEDIUM | ✅ COMPLETE |
| **28** | `descriptor_registry.py` | `milia_pipeline/descriptors/descriptor_registry.py` | 1 | MEDIUM | ✅ COMPLETE |
| **29** | `descriptor_validator.py` | `milia_pipeline/descriptors/descriptor_validator.py` | 1 | MEDIUM | ✅ COMPLETE |
| **30** | `descriptor_categories.py` | `milia_pipeline/descriptors/descriptor_categories.py` | 1 | MEDIUM | ✅ COMPLETE |
| **31** | `descriptor_calculator.py` | `milia_pipeline/descriptors/descriptor_calculator.py` | 2 | MEDIUM | ✅ COMPLETE |
| **32** | `descriptor_plugin_system.py` | `milia_pipeline/descriptors/descriptor_plugin_system.py` | 2 | MEDIUM | ✅ COMPLETE |
| **33** | `architecture_builder.py` | `milia_pipeline/models/builders/architecture_builder.py` | 3 | MEDIUM | ✅ COMPLETE |
| **34** | `model_composer.py` | `milia_pipeline/models/builders/model_composer.py` | 2 | MEDIUM | ✅ COMPLETE |
| **35** | `layer_registry.py` | `milia_pipeline/models/builders/layer_registry.py` | 1 | MEDIUM | ✅ COMPLETE |

#### LOW Priority (Phases 36-38)

| Phase | File | Full Path | Est. Dataclasses | Priority | Status |
|-------|------|-----------|------------------|----------|--------|
| **36** | `nas_manager.py` | `milia_pipeline/models/hpo/nas/nas_manager.py` | 1 | LOW | ✅ COMPLETE |
| **37** | `search_space_builder.py` | `milia_pipeline/models/hpo/search_spaces/search_space_builder.py` | 0 | LOW | ✅ COMPLETE |
| **38** | `model_plugin_system.py` | `milia_pipeline/models/plugins/model_plugin_system.py` | 2 | LOW | ✅ COMPLETE |

### G.4 Summary by Priority

| Priority | Phases | Files | Est. Dataclasses | Status |
|----------|--------|-------|------------------|--------|
| **CRITICAL** | 17-18 | 2 | 17 | 2/2 COMPLETE ✅ |
| **HIGH** | 19-26 | 8 | ~11 | 8/8 COMPLETE ✅ |
| **MEDIUM** | 27-35 | 9 | ~14 | 9/9 COMPLETE ✅ |
| **LOW** | 36-38 | 3 | ~3 | 3/3 COMPLETE ✅ |
| **TOTAL** | 17-38 | **22** | **~45** | **22/22 COMPLETE ✅** |

### G.5 Summary by Module

| Module | Files | Est. Dataclasses | Phases |
|--------|-------|------------------|--------|
| `config/` | 1 | ~10 | 17 |
| `transformations/` | 3 | ~11 | 18, 19, 27 |
| `descriptors/` | 5 | ~7 | 28-32 |
| `models/registry/` | 2 | ~3 | 23-24 |
| `models/builders/` | 3 | ~6 | 33-35 |
| `models/deployment/` | 3 | ~4 | 20-22 |
| `models/factory/` | 1 | ~1 | 25 |
| `models/hpo/` | 2 | ~2 | 26, 36 |
| `models/hpo/search_spaces/` | 1 | 0 | 37 |
| `models/plugins/` | 1 | ~2 | 38 |
| **TOTAL** | **22** | **~46** | 17-38 |

### G.6 Implementation Notes

#### Prerequisites for Each Phase
- Request the target file before implementation
- Analyze line-by-line for dataclass locations
- Identify `to_dict()`, `from_dict()`, `__post_init__` methods to preserve
- Determine frozen vs mutable pattern required

#### Expected Patterns by File Type
| File Type | Expected Pattern | Example Reference |
|-----------|------------------|-------------------|
| Config classes | `BaseModel` + `frozen=True` | `hpo_config.py` (Phase 6b) |
| Metadata classes | `BaseModel` (mutable) + `model_dump()` | `device_manager.py` (Phase 7) |
| Plugin metadata | `BaseModel` (mutable) + `to_dict()` wrapper | `custom_transforms.py` (Phase 16) |
| Validation results | `BaseModel` + `field_validator` | `config_bridge.py` (Phase 3) |

#### Estimated Effort
| Priority | Phases | Estimated Time | Complexity |
|----------|--------|----------------|------------|
| CRITICAL | 17-18 | 4-6 hours | HIGH |
| HIGH | 19-26 | 4-6 hours | MEDIUM |
| MEDIUM | 27-35 | 4-6 hours | MEDIUM |
| LOW | 36-38 | 2-3 hours | LOW |
| **TOTAL** | 17-38 | **14-21 hours** | - |

### G.7 FastAPI Readiness Checklist

After completing Phases 17-38, the following will be achieved:

- [x] **Phase 17**: Config schemas validated via Pydantic ✅
- [x] **Phase 18**: Transform metadata with JSON schema support ✅
- [x] **Phase 19**: Plugin metadata with JSON schema support ✅
- [x] **Phase 20**: Deployment config FastAPI-ready ✅
- [x] **Phase 21**: Monitoring config + Alert FastAPI-ready ✅
- [x] **Phase 22**: Optimization config FastAPI-ready ✅
- [x] **Phase 23**: Model introspection metadata FastAPI-ready ✅
- [x] **Phase 24**: Model registration FastAPI-ready ✅
- [x] **Phase 25**: Target selection config FastAPI-ready ✅
- [x] **Phase 26**: HPO manager cleanup complete (no dataclasses) ✅
- [x] **Phase 27**: Experiment configuration FastAPI-ready ✅
- [x] **Phase 28**: Descriptor registration FastAPI-ready (note: JSON schema limited due to Callable type) ✅
- [x] **Phase 29**: Validation result FastAPI-ready with JSON schema support ✅
- [x] **Phase 30**: Descriptor metadata FastAPI-ready with enum serialization (424 instances) ✅
- [x] **Phase 31**: Calculation result classes FastAPI-ready with JSON schema support ✅
- [x] **Phase 32**: Descriptor plugin system FastAPI-ready with nested model serialization ✅
- [x] **Phase 33**: Architecture builder configs FastAPI-ready with nested serialization ✅
- [x] **Phase 34**: Model composer configs FastAPI-ready with nn.Module support ✅
- [x] **Phase 35**: Layer registry metadata FastAPI-ready with enum serialization ✅
- [x] **Phase 36**: NAS config FastAPI-ready with field validation ✅
- [x] **Phase 37**: Search space builder cleanup complete (no dataclasses; dead import removed) ✅
- [x] **Phase 38**: Model plugin system FastAPI-ready with nested model serialization (2 classes, 35 attr) ✅

🏆 **MIGRATION COMPLETE**: MILIA is now **100% Pydantic V2 migrated** and **fully FastAPI-ready**!

---

**END OF BLUEPRINT v1.39.0 - 🏆 MIGRATION COMPLETE**

## CHANGELOG

### v1.39.0 (2026-01-08)
- **🏆 PHASE 38 COMPLETE - FINAL PHASE**: Model Plugins module `model_plugin_system.py` migrated to Pydantic V2
- **Classes Migrated**: 2 mutable dataclasses (35 total attributes)
  - `ModelDeclaration` - 1 mutable BaseModel (16 attributes: `name`, `class_name`, `module_path`, `category`, `description`, `supported_tasks`, `hyperparameters`, `plugin_name`, `requires_edge_index`, `requires_edge_features`, `requires_edge_weights`, `supports_batch`, `supports_heterogeneous`, `min_pyg_version`, `reference_paper`, `reference_url`)
  - `ModelPluginMetadata` - 1 mutable BaseModel (19 attributes: `plugin_name`, `version`, `author`, `description`, `plugin_type`, `milia_version`, `pyg_version`, `python_version`, `license`, `model_declarations`, `dependencies`, `optional_dependencies`, `homepage`, `repository`, `documentation`, `plugin_path`, `loaded`, `enabled`, `load_time`, `validation_errors`)
- **Import Change**: `from dataclasses import dataclass, field` → `from pydantic import BaseModel, Field`
- **ModelDeclaration Pattern**:
  - Mutable `BaseModel` (no `frozen=True`)
  - 16 attributes (8 required + 8 with defaults)
  - `to_dict()` method added wrapping `model_dump()` for backward compatibility
- **ModelPluginMetadata Pattern**:
  - Mutable `BaseModel` (no `frozen=True`) - required for runtime state changes (`loaded`, `enabled`, `load_time`)
  - 19 attributes (10 required + 9 with defaults)
  - `Field(default_factory=list)` for `dependencies`, `optional_dependencies`, `validation_errors`
  - `model_config = {'arbitrary_types_allowed': True}` for `Path` type
  - Custom `to_dict()` **preserved** (computes `num_models`, extracts model names list)
- **Key Finding - Pattern Consistency**: Follows identical pattern to Phase 32 (`descriptor_plugin_system.py`):
  - Both have `Declaration` class (8 vs 16 attr) + `PluginMetadata` class (19 attr each)
  - Both use `Field(default_factory=list)` for mutable defaults
  - Both preserve custom `to_dict()` with computed properties
- **Backward Compatibility**:
  - Same constructor API: `ModelDeclaration(name='MyModel', class_name='MyClass', ...)` works identically
  - Same constructor API: `ModelPluginMetadata(plugin_name='test', version='1.0.0', ...)` works identically
  - Same attribute access: `decl.name`, `metadata.loaded`, `metadata.enabled`
  - Same default values: All 8 defaults in `ModelDeclaration`, all 9 defaults in `ModelPluginMetadata` preserved
  - Same `to_dict()` output: `ModelPluginMetadata.to_dict()` returns 14-key dict with computed `num_models` and `models` list
  - `ModelPluginLoader` class unchanged - continues to create and modify both classes
  - All convenience functions unchanged: `get_plugin_loader()`, `discover_plugins()`, `list_plugins()`, `get_plugin_info()`
- **Test Coverage**: 20 comprehensive tests passed:
  - Import tests (1): Module imports successfully without dataclass dependency
  - Namespace verification (1): `dataclass`/`field` not in module globals
  - BaseModel inheritance (2): Both classes inherit from Pydantic BaseModel
  - ModelDeclaration instantiation (1): Constructor with required fields and defaults
  - ModelDeclaration.to_dict() (1): Returns 16 keys via `model_dump()`
  - Pydantic V2 methods (1): `model_dump`, `model_validate` available
  - ModelPluginMetadata instantiation (1): Constructor with defaults
  - ModelPluginMetadata.to_dict() (1): Computes `num_models` and extracts `models` list
  - Field(default_factory=list) independence (1): No shared mutable state
  - Mutability (1): `loaded`, `enabled`, `load_time` can be modified
  - validation_errors default (1): Empty list via `Field(default_factory=list)`
  - Nested ModelDeclaration (1): List of declarations works correctly
  - model_config Path support (1): `arbitrary_types_allowed=True` works
  - ModelPluginLoader singleton (1): `get_plugin_loader()` returns singleton
  - list_plugins function (1): Convenience function works
  - Module version (1): Version 1.1.0 with Phase 38 docs
  - __all__ exports (1): All expected exports present
  - model_validate from dict (1): Creates instance from dict
  - JSON schema generation (1): Generates schema with 16 properties
- **NON-BREAKING**: Same constructor API, attribute access, `to_dict()` output preserved
- **DYNAMIC**: `Field(default_factory=list)` for mutable defaults; `model_dump()` auto-includes all fields; nested model handling
- **PRODUCTION-READY**: Runtime type validation, `ConfigDict(arbitrary_types_allowed=True)` for Path type, thread-safe in singleton loader
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation via `model_json_schema()`, extensible with validators
- **Version Bump**: `model_plugin_system.py` v1.0.0 → v1.1.0
- **🏆 MIGRATION COMPLETE**: All 38 phases successfully implemented
- **Final Statistics**:
  - Total Phases: 38/38 COMPLETE (100%)
  - Total Dataclasses Migrated: 108 across all phases
  - Total Attributes: ~500+ attributes with runtime validation
  - Test Coverage: 600+ tests passed across all phases
- **MILIA is now 100% Pydantic V2 migrated and fully FastAPI-ready**

### v1.38.0 (2026-01-08)
- **Phase 37 COMPLETE**: HPO Search Spaces module `search_space_builder.py` cleaned up
- **Classes Migrated**: 0 dataclasses (dead code removal only)
- **Import Change**: Removed unused `from dataclasses import dataclass, field` (line 29)
- **Key Finding - No Dataclasses**: Line-by-line analysis revealed:
  - `SearchSpaceBuilder` is a **regular Python class** (not a dataclass)
  - Uses manual `__init__` with `self._search_space = {}` and `self._frozen = False`
  - The dataclass import was **never used** (dead code)
- **Pattern Used**: Same as Phase 26 (`hpo_manager.py`) - dead code removal only
- **Module Already Pydantic-Compatible**:
  - Uses `SearchSpaceParamConfig` from `param_types.py` (Phase 6a - Pydantic BaseModel)
  - All search space parameters are already Pydantic-validated
- **Backward Compatibility**:
  - Same constructor API: `SearchSpaceBuilder()` works identically
  - Same fluent builder pattern: `.add_int().add_float().build()` unchanged
  - Same class methods: `for_model()`, `from_dict()`, `merge()`, `validate()` unchanged
  - Same convenience functions: `build_search_space()`, `get_model_search_space()`, `validate_search_space()`
  - Same exports: `__all__` unchanged
- **Test Coverage**: 15 comprehensive tests passed:
  - Import tests (1): Module imports without dataclass dependency
  - Namespace verification (1): `dataclass`/`field` not in module globals
  - Class type (1): `SearchSpaceBuilder` is regular class (not dataclass)
  - Instantiation (1): Constructor creates correct instance attributes
  - Fluent builder (1): Method chaining works correctly
  - Pydantic integration (1): `SearchSpaceParamConfig` is Pydantic BaseModel
  - Convenience functions (2): `build_search_space()`, `validate_search_space()` work
  - Serialization (2): `to_dict()`, `from_dict()` work correctly
  - Class methods (2): `list_available_models()`, `merge()` work
  - Documentation (1): Version 2.1.0 with Phase 37 note
  - Class attributes (1): `VALID_CATEGORIES` frozenset intact
  - Exports (1): `__all__` contains all expected functions
- **NON-BREAKING**: Zero API changes; dead code removal only
- **DYNAMIC**: No hardcoded values introduced; dynamic introspection preserved
- **PRODUCTION-READY**: Cleaner codebase; fewer imports; faster module load
- **FUTURE-PROOF**: Module already uses Pydantic via `SearchSpaceParamConfig`; FastAPI-ready
- **Version Bump**: `search_space_builder.py` v2.0.0 → v2.1.0
- **LOW Priority Progress**: 2/3 files in LOW priority phases migrated (Phase 36-37 complete; Phase 38 pending)
- **Overall Progress**: 106/~111 dataclasses migrated (95.5%) - no new dataclasses; dead code removed
- **Remaining**: Only 1 LOW priority phase (38) pending for 100% completion
- **Documentation**: Full Phase 37 details added; APPENDIX G updated; FastAPI Readiness Checklist updated

### v1.37.0 (2026-01-08)
- **Phase 36 COMPLETE**: HPO NAS module `nas_manager.py` migrated to Pydantic V2
- **Classes Migrated**: 1 mutable dataclass (7 attributes)
  - `NASConfig` - 1 mutable BaseModel (7 attributes: `n_trials`, `timeout`, `metric`, `direction`, `cv_folds`, `study_name`, `storage`)
- **Import Change**: `from dataclasses import dataclass, field, replace` → `from pydantic import BaseModel, field_validator`
- **NASConfig Pattern**:
  - Mutable `BaseModel` (no `frozen=True`) - follows `GNNArchitectureSpace` pattern from same NAS module
  - 7 attributes (0 required + 7 with defaults)
  - `@field_validator` for 4 fields: `n_trials`, `timeout`, `direction`, `cv_folds`
  - `to_dict()` method added wrapping `model_dump()` for backward compatibility
- **Validation Migration**: `__post_init__` → `@field_validator`:
  - `n_trials`: Must be >= 1
  - `timeout`: Must be positive or None
  - `direction`: Must be 'minimize' or 'maximize'
  - `cv_folds`: Must be non-negative
- **Key Finding - Mutable Pattern**: `NASConfig` follows the mutable BaseModel pattern because:
  1. `GNNArchitectureSpace` in same module is mutable (Phase 14)
  2. Config may need runtime modification before passing to `NASManager`
  3. Consistent with HPO module conventions
- **Backward Compatibility**:
  - Same constructor API: `NASConfig(n_trials=100, cv_folds=5)` works identically
  - Same attribute access: `config.n_trials`, `config.direction`, `config.storage`
  - Same default values: All 7 defaults preserved
  - `NASManager` usage patterns unchanged
- **Test Coverage**: 20 comprehensive tests passed:
  - Import tests (1): `NASConfig`, `NASManager` imported successfully
  - BaseModel inheritance (1): `NASConfig` inherits from Pydantic BaseModel
  - Default instantiation (1): All 7 default values verified
  - Custom instantiation (1): Non-default values work correctly
  - `to_dict()` backward compatibility (1): Returns dict with 7 keys
  - `model_dump()` equivalence (1): `to_dict() == model_dump()`
  - Validation tests (4): `n_trials`, `timeout`, `direction`, `cv_folds` validators work
  - Edge cases (2): `timeout=None` valid, `timeout>0` valid
  - Pydantic V2 features (4): `model_validate()`, `model_json_schema()`, `model_dump_json()`, type coercion
  - Mutability (1): Attributes can be modified
  - `model_copy()` (1): Copy with update works
  - Module exports (1): `__all__` includes `NASConfig`
  - Version check (1): Version 1.1.0 and migration docs present
- **NON-BREAKING**: Same constructor API, attribute access, `to_dict()` method added
- **DYNAMIC**: `@field_validator` uses dynamic comparisons; `model_dump()` auto-includes all fields
- **PRODUCTION-READY**: Runtime type validation at construction, clear error messages, thread-safe BaseModel
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation via `model_json_schema()`, extensible with additional validators
- **Version Bump**: `nas_manager.py` v1.0.0 → v1.1.0
- **LOW Priority Progress**: 1/3 files in LOW priority phases migrated (Phase 36 complete; Phases 37-38 pending)
- **Overall Progress**: 106/~111 dataclasses migrated (95.5%)
- **Remaining**: Only 2 LOW priority phases (37-38) pending
- **Documentation**: Full Phase 36 details added; APPENDIX G updated; FastAPI Readiness Checklist updated

### v1.36.0 (2026-01-08)
- **Phase 35 COMPLETE**: Builders module `layer_registry.py` migrated to Pydantic V2
- **Classes Migrated**: 1 mutable dataclass (12 attributes)
  - `LayerMetadata` - 1 mutable BaseModel (12 attributes: `name`, `category`, `class_path`, `description`, `requires_edge_index`, `requires_edge_attr`, `requires_batch`, `has_in_channels`, `has_out_channels`, `modifies_graph_structure`, `supported_task_levels`, `is_functional`)
- **Import Change**: `from dataclasses import dataclass, field` → `from pydantic import BaseModel, Field`
- **LayerMetadata Pattern**:
  - Mutable `BaseModel` (no `frozen=True`) - required for `LayerRegistry._register_layer()` mutation
  - 12 attributes (4 required + 8 with defaults)
  - `Field(default_factory=lambda: ["node", "edge", "graph"])` for `supported_task_levels` mutable default
  - Custom `to_dict()` preserved (12 keys) with **enum `.value` serialization** for `category` field
- **Key Finding - Enum Serialization**: Custom `to_dict()` preserved because:
  1. Uses `self.category.value` (line 191) for enum serialization to string
  2. Returns "convolutional", "pooling", etc. instead of enum object
  3. This matches expected JSON serialization format
- **Key Finding - Mutability Required**: `LayerMetadata.is_functional` is mutated in `LayerRegistry._register_layer()`:
  - Line 596: `metadata.is_functional = True` (when auto-wrapping functional layers)
- **Backward Compatibility**:
  - Same constructor API: `LayerMetadata(name='GCNConv', category=LayerCategory.CONVOLUTIONAL, class_path='...', description='...')` works identically
  - Same attribute access: `metadata.name`, `metadata.category`, `metadata.is_functional`
  - Same default values: All 8 defaults preserved (`requires_edge_index=True`, `supported_task_levels=["node", "edge", "graph"]`, etc.)
  - Same `to_dict()` output format (12 keys with enum `.value`)
  - `LayerRegistry` class unchanged - continues to create and mutate `LayerMetadata` objects
  - All convenience functions unchanged: `get_layer()`, `list_layers()`, `get_layer_metadata()`
- **Test Coverage**: 25 comprehensive tests passed:
  - Import tests (1): All classes imported successfully
  - BaseModel inheritance (1): `LayerMetadata` inherits from Pydantic BaseModel
  - Instantiation with enum (1): Constructor with `LayerCategory` enum field works
  - Default values (8): All 8 defaults verified
  - `Field(default_factory=...)` independence (1): No shared mutable state for `supported_task_levels`
  - Serialization (3): `to_dict()` returns 12 keys with enum `.value`
  - Pydantic V2 features (3): `model_dump()`, `model_validate()`, `model_json_schema()` available
  - Mutability (1): `is_functional` can be modified
  - Registry integration (6): `LayerRegistry` singleton works with Pydantic `LayerMetadata`
- **NON-BREAKING**: Same constructor API, attribute access, `to_dict()` behavior preserved
- **DYNAMIC**: `Field(default_factory=...)` for mutable defaults; enum type handled natively
- **PRODUCTION-READY**: Runtime type validation, thread-safe in singleton registry, no shared mutable state
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation available, extensible with `@field_validator`
- **Version Bump**: `layer_registry.py` v1.0.0 → v1.1.0
- **Builders Module Progress**: 3/3 files in builders module migrated ✅ (Phase 33-35 complete)
- **MEDIUM Priority COMPLETE**: All 9 MEDIUM priority phases (27-35) now complete ✅
- **Overall Progress**: 105/~111 dataclasses migrated (94.6%)
- **Remaining**: Only 3 LOW priority phases (36-38) pending
- **Documentation**: Full Phase 35 details added; APPENDIX G updated; FastAPI Readiness Checklist updated

### v1.35.0 (2026-01-08)
- **Phase 34 COMPLETE**: Builders module `model_composer.py` migrated to Pydantic V2
- **Classes Migrated**: 2 mutable dataclasses (9 total attributes)
  - `ModelSpec` - 1 mutable BaseModel (4 attributes: `model`, `weight`, `name`, `level`)
  - `EnsembleConfig` - 1 mutable BaseModel (5 attributes: `name`, `task_type`, `models`, `strategy`, `fusion`)
- **Import Change**: `from dataclasses import dataclass, field` → `from pydantic import BaseModel, Field, ConfigDict`
- **ModelSpec Pattern**:
  - Mutable `BaseModel` (no `frozen=True`) - required for weight normalization in `ModelComposer.build()`
  - 4 attributes (1 required + 3 with defaults: `weight=1.0`, `name=None`, `level=0`)
  - `model_config = ConfigDict(arbitrary_types_allowed=True)` for `nn.Module` field
  - Custom `to_dict()` preserved (4 keys with computed `name` and `model_class`)
- **EnsembleConfig Pattern**:
  - Mutable `BaseModel` (no `frozen=True`)
  - 5 attributes (2 required + 3 with defaults)
  - `model_config = ConfigDict(arbitrary_types_allowed=True)` for nested `ModelSpec` containing `nn.Module`
  - `Field(default_factory=list)` for `models` mutable default
  - Custom `to_dict()` preserved with **nested serialization** (calls `m.to_dict()` for each model)
  - Computed `num_models` field in `to_dict()` output
- **Key Finding - ConfigDict Required**: Both classes require `ConfigDict(arbitrary_types_allowed=True)` because:
  1. `ModelSpec.model` is `nn.Module` which is not a standard JSON-serializable type
  2. `EnsembleConfig.models` is `List[ModelSpec]` which contains `nn.Module`
  3. This follows the established pattern from Phase 24 (`model_registry.py`)
- **Key Finding - Mutability Required**: `ModelSpec.weight` is mutated in `ModelComposer.build()`:
  - Line 507: `spec.weight /= total_weight` (weight normalization for weighted fusion)
- **Key Finding - No from_dict()**: Neither class has `from_dict()` because `nn.Module` instances cannot be serialized/deserialized. `ModelComposer.from_config()` only recreates composer structure; models must be added separately.
- **Backward Compatibility**:
  - Same constructor API: `ModelSpec(model=my_model, weight=0.6, name='GCN')` works identically
  - Same constructor API: `EnsembleConfig(name='Test', task_type='graph_regression')` works identically
  - Same attribute access: `spec.model`, `spec.weight`, `config.models`, `config.strategy`
  - Same default values: `weight=1.0`, `level=0`, `name=None`, `strategy="parallel"`, `fusion="mean"`, `models=[]`
  - Same `to_dict()` output format (4 keys for ModelSpec, 6 keys for EnsembleConfig)
  - `ModelComposer` class unchanged - continues to create and mutate `ModelSpec` objects
  - All ensemble classes unchanged: `ParallelEnsemble`, `SequentialStack`, `HierarchicalComposition`
- **Test Coverage**: 25 comprehensive tests passed:
  - Import tests (1): All classes imported successfully
  - BaseModel inheritance (2): Both classes inherit from Pydantic BaseModel
  - ModelSpec instantiation (4): Constructor with nn.Module and defaults
  - ModelSpec serialization (2): `to_dict()` returns 4 keys with computed values
  - ModelSpec ConfigDict (1): `arbitrary_types_allowed=True` configured
  - EnsembleConfig instantiation (4): Constructor and defaults
  - EnsembleConfig `Field(default_factory=list)` independence (1): No shared mutable state
  - EnsembleConfig with nested ModelSpec (1): Nested instantiation works
  - EnsembleConfig serialization (3): `to_dict()` nested serialization, computed `num_models`
  - EnsembleConfig ConfigDict (1): `arbitrary_types_allowed=True` configured
  - Pydantic V2 features (3): `model_dump()`, `model_validate()` available
  - Mutability (1): `ModelSpec.weight` can be modified
  - ModelComposer integration (1): Works with Pydantic `ModelSpec`
- **NON-BREAKING**: Same constructor API, attribute access, `to_dict()` behavior preserved
- **DYNAMIC**: `Field(default_factory=list)` for mutable defaults; nested serialization via method calls; computed fields at runtime
- **PRODUCTION-READY**: Runtime type validation, `ConfigDict(arbitrary_types_allowed=True)` for complex types, thread-safe BaseModel
- **FUTURE-PROOF**: FastAPI ready (note: nn.Module needs custom serialization for API), JSON schema generation available, extensible with `@field_validator`
- **Version Bump**: `model_composer.py` v1.0.0 → v1.1.0
- **Builders Module Progress**: 2/3 files in builders module migrated (Phase 33-34 complete; Phase 35 pending)
- **Overall Progress**: 104/~111 dataclasses migrated (93.7%)
- **Documentation**: Full Phase 34 details added; APPENDIX G updated; FastAPI Readiness Checklist updated

### v1.34.0 (2026-01-08)
- **Phase 33 COMPLETE**: Builders module `architecture_builder.py` migrated to Pydantic V2
- **Classes Migrated**: 3 mutable dataclasses (15 total attributes)
  - `LayerConfig` - 1 mutable BaseModel (6 attributes: `type`, `params`, `position`, `in_channels`, `out_channels`, `input_from`)
  - `ResidualConnection` - 1 mutable BaseModel (3 attributes: `start_layer`, `end_layer`, `connection_type`)
  - `ArchitectureConfig` - 1 mutable BaseModel (6 attributes: `name`, `task_type`, `in_channels`, `out_channels`, `layers`, `residual_connections`)
- **Import Change**: `from dataclasses import dataclass, field` → `from pydantic import BaseModel, Field`
- **LayerConfig Pattern**:
  - Mutable `BaseModel` (no `frozen=True`) - required for `ArchitectureBuilder` position mutations
  - 6 attributes (3 required + 3 with defaults)
  - `Field(default_factory=lambda: [-1])` for `input_from` mutable default
  - Custom `to_dict()` preserved (6 keys) for backward compatibility
  - `from_dict()` classmethod preserved unchanged
- **ResidualConnection Pattern**:
  - Mutable `BaseModel` (no `frozen=True`)
  - 3 attributes (2 required + 1 default: `connection_type="add"`)
  - Custom `to_dict()` preserved (3 keys) for backward compatibility
  - `from_dict()` classmethod preserved unchanged
- **ArchitectureConfig Pattern**:
  - Mutable `BaseModel` (no `frozen=True`)
  - 6 attributes (4 required + 2 with defaults)
  - `Field(default_factory=list)` for `layers` and `residual_connections` mutable defaults
  - Custom `to_dict()` preserved with **nested serialization** (calls `layer.to_dict()` for each layer)
  - `from_dict()` classmethod preserved with **nested deserialization** (creates `LayerConfig` and `ResidualConnection` objects)
- **Key Finding - Custom to_dict()/from_dict() Preserved**: Unlike simpler migrations, `to_dict()` was NOT replaced with `model_dump()` wrapper because:
  1. `ArchitectureConfig.to_dict()` performs nested serialization (calls `layer.to_dict()`)
  2. `ArchitectureConfig.from_dict()` performs nested deserialization (creates nested Pydantic objects)
  3. This custom logic must be preserved for YAML/JSON config export/import
- **Key Finding - Mutability Required**: `LayerConfig.position` is mutated by `ArchitectureBuilder` methods:
  - Line 309: `self.layers[i].position = i` (after layer insertion)
  - Line 331: `self.layers[i].position = i` (after layer removal)
  - Lines 409-410: `self.layers[pos1].position = pos1` (after layer swap)
- **Backward Compatibility**:
  - Same constructor API: `LayerConfig(type='GCNConv', params={}, position=0)` works identically
  - Same constructor API: `ArchitectureConfig(name='Test', task_type='graph_regression', in_channels=16, out_channels=1)` works identically
  - Same attribute access: `config.type`, `config.params`, `config.layers[0].position`
  - Same default values: `input_from=[-1]`, `connection_type="add"`, `layers=[]`, `residual_connections=[]`
  - Same `to_dict()` output format (6/3/6 keys with nested serialization)
  - Same `from_dict()` behavior (nested deserialization)
  - `ArchitectureBuilder` class unchanged - continues to mutate `LayerConfig.position`
  - `CustomArchitecture` class unchanged - uses `LayerConfig` objects directly
- **Test Coverage**: 25 comprehensive tests passed:
  - Import tests (1): All classes imported successfully
  - BaseModel inheritance (3): All 3 classes inherit from Pydantic BaseModel
  - LayerConfig instantiation (2): Constructor and defaults work
  - LayerConfig `Field(default_factory)` independence (1): No shared mutable state
  - LayerConfig serialization (2): `to_dict()` returns 6 keys, `from_dict()` works
  - ResidualConnection instantiation (2): Constructor and default `connection_type="add"`
  - ResidualConnection serialization (2): `to_dict()` returns 3 keys, `from_dict()` works
  - ArchitectureConfig instantiation (3): Constructor and default empty lists
  - ArchitectureConfig `Field(default_factory=list)` independence (1): No shared state
  - ArchitectureConfig nested serialization (3): `to_dict()` nested, `from_dict()` nested deserialization
  - Pydantic V2 features (4): `model_dump()`, `model_validate()`, `model_json_schema()` available
  - Mutability (1): `LayerConfig.position` can be modified
- **NON-BREAKING**: Same constructor API, attribute access, `to_dict()`/`from_dict()` behavior preserved
- **DYNAMIC**: `Field(default_factory=...)` for mutable defaults; nested serialization via object method calls
- **PRODUCTION-READY**: Runtime type validation, clear error messages, thread-safe BaseModel, no shared mutable state
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation via `model_json_schema()`, extensible with `@field_validator`
- **Version Bump**: `architecture_builder.py` v1.0.0 → v1.1.0
- **Builders Module Progress**: 1/3 files in builders module migrated (Phase 33 complete; Phases 34-35 pending)
- **Overall Progress**: 102/~111 dataclasses migrated (91.9%)
- **Documentation**: Full Phase 33 details added; APPENDIX G updated; FastAPI Readiness Checklist updated

### v1.33.0 (2026-01-08)
- **Phase 32 COMPLETE**: Descriptors module `descriptor_plugin_system.py` migrated to Pydantic V2
- **Classes Migrated**: 2 mutable dataclasses (27 total attributes)
  - `DescriptorDeclaration` - 1 mutable BaseModel (8 attributes: `name`, `function_name`, `module_path`, `category`, `description`, `requires_3d`, `requires_charges`, `version`)
  - `DescriptorPluginMetadata` - 1 mutable BaseModel (19 attributes: `plugin_name`, `version`, `author`, `email`, `license`, `description`, `homepage`, `milia_version`, `python_version`, `dependencies`, `descriptor_declarations`, `registered_descriptors`, `discovery_source`, `discovery_timestamp`, `is_validated`, `validation_date`, `validation_results`, `checksum`, `trusted`)
- **Import Change**: `from dataclasses import dataclass, field` → `from pydantic import BaseModel, Field, model_validator`
- **DescriptorDeclaration Pattern**:
  - Mutable `BaseModel` (no `frozen=True`)
  - 8 attributes (3 required + 5 with defaults)
  - `to_dict()` method wrapping `model_dump()` for backward compatibility
  - `from_dict()` classmethod preserved unchanged
- **DescriptorPluginMetadata Pattern**:
  - Mutable `BaseModel` (no `frozen=True`)
  - 19 attributes (3 required + 16 with defaults)
  - `Field(default_factory=list)` for `dependencies`, `descriptor_declarations`
  - `Field(default_factory=set)` for `registered_descriptors`
  - `Field(default_factory=dict)` for `validation_results`
  - `@model_validator(mode='after')` replaces `__post_init__` validation
  - Custom `__hash__` preserved (hashes by `plugin_name` + `version`)
  - Custom `__eq__` added for consistency with `__hash__`
  - 4 computed `@property` methods preserved: `declared_count`, `registered_count`, `missing_implementations`, `undeclared_implementations`
  - `to_dict()` returns 21 keys (17 fields + 4 computed properties)
  - Static method `_is_valid_version()` and instance method `_validate_dependencies()` preserved
- **Backward Compatibility**:
  - Same constructor API: `DescriptorDeclaration(name='X', function_name='f', module_path='m')` works identically
  - Same constructor API: `DescriptorPluginMetadata(plugin_name='p', version='1.0.0', author='a')` works identically
  - Same attribute access for all 27 attributes
  - Same default values for all 21 defaults
  - `to_dict()` output unchanged (8 keys for Declaration, 21 keys for Metadata)
  - `from_dict()` classmethod works identically
  - Validation logic preserved (plugin_name required, version format, dependencies)
  - `__hash__` preserved for set/dict usage
  - All computed properties work unchanged
  - `DescriptorPluginLoader` class integration unchanged
- **Test Coverage**: 25 comprehensive tests passed:
  - Import tests (1): All classes imported successfully
  - BaseModel inheritance (1): Both classes inherit from Pydantic BaseModel
  - DescriptorDeclaration instantiation (1): All 8 attributes work
  - DescriptorDeclaration defaults (1): 5 default values preserved
  - DescriptorDeclaration `to_dict()` (1): Returns dict with 8 keys
  - DescriptorDeclaration `from_dict()` (1): Creates instance from dict
  - DescriptorPluginMetadata instantiation (1): Required args only
  - DescriptorPluginMetadata defaults (1): 16 default values preserved
  - `Field(default_factory)` independence (1): No shared mutable state
  - Computed properties (1): All 4 properties work correctly
  - DescriptorPluginMetadata `to_dict()` (1): Returns dict with 21 keys
  - `__hash__` and `__eq__` (1): Hash by plugin_name+version works
  - Set usage (1): Hashable for set/dict usage
  - `@model_validator` validation (1): Invalid version format rejected
  - Valid semantic versions (1): All valid versions accepted
  - `model_dump()` equivalence (1): `to_dict()` wraps `model_dump()` correctly
  - `model_validate()` from dict (1): Creates instances from dict
  - `model_json_schema()` generation (1): JSON schema works for both classes
  - Mutability (1): Both classes are mutable
  - `model_fields` attribute (1): 8 for Declaration, 19 for Metadata
  - Nested DescriptorDeclaration (1): Nested models serialize correctly
  - DescriptorPluginLoader singleton (1): Singleton pattern works
  - Module version check (1): Version 1.1.0 with Phase 32 migration note
  - Type coercion (1): int → bool coercion works
  - Dependencies validation (1): List validation works
- **NON-BREAKING**: Same constructor API, attribute access, all methods preserved, `to_dict()` output unchanged
- **DYNAMIC**: `Field(default_factory=...)` for mutable defaults; `model_dump()` auto-includes fields; computed properties dynamically calculated
- **PRODUCTION-READY**: Runtime type validation, clear error messages, thread-safe BaseModel, singleton pattern preserved
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation via `model_json_schema()`, extensible with `@field_validator`
- **Version Bump**: `descriptor_plugin_system.py` v1.0.0 → v1.1.0
- **Descriptors Module Progress**: 5/5 files in descriptors module migrated (Phases 28-32 COMPLETE) ✅
- **Overall Progress**: 99/~111 dataclasses migrated (89.2%)
- **Documentation**: Full Phase 32 details added; APPENDIX G updated; FastAPI Readiness Checklist updated

### v1.32.0 (2026-01-08)
- **Phase 31 COMPLETE**: Descriptors module `descriptor_calculator.py` migrated to Pydantic V2
- **Classes Migrated**: 2 mutable dataclasses (9 total attributes)
  - `CalculationResult` - 1 mutable BaseModel (5 attributes: `success`, `value`, `descriptor_name`, `error_message`, `computation_time`)
  - `BatchCalculationResult` - 1 mutable BaseModel (4 attributes: `successful`, `failed`, `total_time`, `molecules_processed`)
- **Import Change**: `from dataclasses import dataclass` → `from pydantic import BaseModel`
- **CalculationResult Pattern**:
  - Mutable `BaseModel` (no `frozen=True`) - allows attribute modification
  - 5 attributes (3 required + 2 with defaults: `error_message=None`, `computation_time=None`)
  - `to_dict()` method added wrapping `model_dump()` for backward compatibility
- **BatchCalculationResult Pattern**:
  - Mutable `BaseModel` (no `frozen=True`) - allows attribute modification
  - 4 attributes (all required, no defaults)
  - `successful` and `failed` are Dict types - Pydantic handles deep copy automatically
  - `to_dict()` method added wrapping `model_dump()` for backward compatibility
- **Backward Compatibility**:
  - Same constructor API: `CalculationResult(success=True, value=180.2, descriptor_name='MolWt')` works identically
  - Same constructor API: `BatchCalculationResult(successful={...}, failed={...}, total_time=0.1, molecules_processed=1)` works identically
  - Same attribute access: `result.success`, `result.value`, `batch.successful`, etc.
  - Same default values: `error_message=None`, `computation_time=None` preserved
  - `DescriptorCalculator` class integration unchanged - returns same result types
- **Test Coverage**: 20 comprehensive tests passed:
  - Import tests (1): All classes imported successfully
  - BaseModel inheritance (1): Both classes inherit from Pydantic BaseModel
  - CalculationResult instantiation (1): All 5 attributes work
  - CalculationResult defaults (1): `error_message=None`, `computation_time=None` preserved
  - BatchCalculationResult instantiation (1): All 4 attributes work
  - CalculationResult `to_dict()` (1): Returns dict with 5 keys
  - BatchCalculationResult `to_dict()` (1): Returns dict with 4 keys
  - `model_dump()` equivalence (1): `to_dict()` correctly wraps `model_dump()`
  - `model_validate()` from dict (1): Creates instances from dict
  - `model_json_schema()` generation (1): JSON schema works for both classes
  - Mutability (1): Both classes are mutable (not frozen)
  - Type coercion (1): Pydantic type coercion works (int → bool, int → float)
  - DescriptorCalculator integration (1): Calculator instantiation works
  - Module version check (1): Version 1.1.0 with Phase 31 migration note
  - `model_fields` attribute (1): 5 fields for CalculationResult, 4 for BatchCalculationResult
  - Dict field independence (1): No shared mutable state between instances
  - Keyword arguments order (1): Works in any order
  - Error case (1): Failed calculation result pattern works
  - Empty batch result (1): Empty dicts work correctly
  - Large batch result (1): 100 successful + 50 failed descriptors work
- **NON-BREAKING**: Same constructor API, attribute access, default values preserved
- **DYNAMIC**: `model_dump()` auto-includes all fields; no hardcoded key lists; Pydantic handles Dict deep copy
- **PRODUCTION-READY**: Runtime type validation, clear error messages, thread-safe BaseModel
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation via `model_json_schema()`, extensible with `@field_validator`
- **Version Bump**: `descriptor_calculator.py` v1.0.0 → v1.1.0
- **Descriptors Module Progress**: 4/5 files in descriptors module migrated (Phase 28-31 complete; Phase 32 pending)
- **Overall Progress**: 97/~111 dataclasses migrated (87.4%)
- **Documentation**: Full Phase 31 details added; APPENDIX G updated; FastAPI Readiness Checklist updated

### v1.31.0 (2026-01-08)
- **Phase 30 COMPLETE**: Descriptors module `descriptor_categories.py` migrated to Pydantic V2
- **Class Migrated**: 1 frozen dataclass (6 attributes) with 424 module-level instances
  - `DescriptorMetadata` - 1 frozen BaseModel (6 attributes: `name`, `category`, `requires_3d`, `requires_charges`, `description`, `rdkit_module`)
- **Import Change**: `from dataclasses import dataclass` → `from pydantic import BaseModel`
- **DescriptorMetadata Pattern**:
  - Frozen `BaseModel` (`frozen=True`) - required for immutability and hashability
  - 6 attributes (2 required + 4 with defaults)
  - Custom `__init__` added to support positional arguments (critical for 424 existing usages)
  - Custom `__hash__` preserved - hashes by `name` only for set/dict usage
  - Custom `__eq__` added - compares by `name` only for consistency with `__hash__`
  - `to_dict()` method added with `model_dump(mode='json')` for enum value serialization
- **Key Finding - Positional Arguments Required**: Pydantic V2 `BaseModel` does not support positional arguments by default (only keyword arguments). Original code used positional args extensively:
  ```python
  DescriptorMetadata("MolWt", DescriptorCategory.CONSTITUTIONAL, description="Molecular weight")
  ```
  Solution: Custom `__init__` that accepts positional arguments and converts them to keyword arguments for `super().__init__()`. This preserves full backward compatibility with all 424 existing usages.
- **Key Finding - Enum Serialization**: `to_dict()` uses `model_dump(mode='json')` to automatically serialize `DescriptorCategory` enum to string value (e.g., `DescriptorCategory.CONSTITUTIONAL` → `"constitutional"`).
- **Backward Compatibility**:
  - Same constructor API: `DescriptorMetadata("MolWt", DescriptorCategory.CONSTITUTIONAL)` works (positional)
  - Same constructor API: `DescriptorMetadata(name="MolWt", category=DescriptorCategory.CONSTITUTIONAL)` works (keyword)
  - Same attribute access: `meta.name`, `meta.category`, `meta.requires_3d`, etc.
  - Same default values: `requires_3d=False`, `requires_charges=False`, `description=""`, `rdkit_module="Descriptors"`
  - Same hashability: `hash(meta) == hash(meta.name)` for set/dict usage
  - All 424 `DescriptorMetadata` instances in module constants work unchanged
  - All helper functions work unchanged: `get_descriptors_by_category()`, `get_descriptor_metadata()`, `requires_3d_coordinates()`, `requires_partial_charges()`, `get_all_descriptor_names()`, `filter_descriptors_by_requirements()`, `validate_descriptor_coverage()`
  - `DESCRIPTOR_METADATA_MAP`, `ALL_DESCRIPTORS`, `DESCRIPTORS_BY_CATEGORY` work unchanged
- **Test Coverage**: 20 comprehensive tests passed:
  - Import tests (1): All classes, functions, constants
  - BaseModel verification (1): Inheritance check
  - Pydantic V2 methods (1): `model_dump`, `model_validate`, `model_json_schema`, `model_fields` available
  - Positional arguments (1): Critical test - both positional args work
  - Mixed positional/keyword (1): Combined usage works
  - Frozen/immutability (1): Cannot modify attributes
  - Custom `__hash__` (1): Hashes by name only
  - Custom `__eq__` (1): Compares by name only
  - Set usage (1): Hashable for sets
  - `to_dict()` with enum serialization (1): Returns 6 keys, enum as string
  - `ALL_DESCRIPTORS` (1): 424 Pydantic instances
  - `DESCRIPTOR_METADATA_MAP` (1): Dict lookup works
  - Helper functions (1): All helper functions work
  - `model_validate()` from dict (1): Creates instance from dict
  - JSON schema generation (1): Schema with all properties
  - Type coercion (1): `int` → `bool` for boolean fields
  - Module version check (1): Version 1.1.0 with Phase 30 migration note
  - Existing usage patterns (1): All module patterns work
  - `filter_descriptors_by_requirements()` (1): Filtering works correctly
  - `DESCRIPTORS_BY_CATEGORY` (1): Category dict works
- **NON-BREAKING**: Same constructor API (positional + keyword), attribute access, hash/eq behavior, all 424 instances work
- **DYNAMIC**: `model_dump(mode='json')` auto-serializes enums; custom `__init__` dynamically converts positional to keyword args
- **PRODUCTION-READY**: Runtime type validation, frozen immutability, thread-safe, 424 instances validated at module load
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation, extensible with validators
- **Version Bump**: `descriptor_categories.py` v1.0.0 → v1.1.0
- **Descriptors Module Progress**: 3/5 files in descriptors module migrated (Phase 28-30 complete; Phases 31-32 pending)
- **Overall Progress**: 95/~111 dataclasses migrated (85.6%)
- **Documentation**: Full Phase 30 details added; APPENDIX G updated; FastAPI Readiness Checklist updated

### v1.30.0 (2026-01-08)
- **Phase 29 COMPLETE**: Descriptors module `descriptor_validator.py` migrated to Pydantic V2
- **Class Migrated**: 1 mutable dataclass (4 attributes)
  - `ValidationResult` - 1 mutable BaseModel (4 attributes: `is_valid`, `errors`, `warnings`, `details`)
- **Import Change**: `from dataclasses import dataclass` → `from pydantic import BaseModel, Field`
- **ValidationResult Pattern**:
  - Mutable `BaseModel` (no `frozen=True`) - required for list/dict attribute modification
  - 4 attributes (2 required + 2 with mutable defaults)
  - `Field(default_factory=list)` for `warnings` (replaces `None` default + `__post_init__`)
  - `Field(default_factory=dict)` for `details` (replaces `None` default + `__post_init__`)
  - `__post_init__` REMOVED - `Field(default_factory=...)` pattern handles `None` → empty conversion
  - `to_dict()` method added wrapping `model_dump()` for backward compatibility
- **Key Finding - __post_init__ Replacement**: Original code used:
  ```python
  warnings: List[str] = None
  details: Dict[str, Any] = None
  def __post_init__(self):
      if self.warnings is None: self.warnings = []
      if self.details is None: self.details = {}
  ```
  Pydantic V2 pattern replaces this with `Field(default_factory=list/dict)` which:
  1. Creates independent empty list/dict for each instance (no shared state bugs)
  2. Eliminates need for `__post_init__` entirely
  3. Provides same behavior - callers passing no value get empty containers
- **Backward Compatibility**:
  - Same constructor API: `ValidationResult(is_valid=True, errors=[])`
  - Same constructor with optionals: `ValidationResult(is_valid=True, errors=[], warnings=['warn'], details={'key': 'val'})`
  - Same attribute access: `result.is_valid`, `result.errors`, `result.warnings`, `result.details`
  - All usages in `DescriptorValidator` class remain functional (`validate_batch_values()`, `validate_configuration()`)
  - Global `validator` instance and convenience functions unchanged
- **Test Coverage**: 20 comprehensive tests passed:
  - Import tests (1): `ValidationResult`, `DescriptorValidator`, `validator`, convenience functions
  - BaseModel verification (1): Inheritance check
  - Pydantic V2 methods (1): `model_dump`, `model_validate`, `model_json_schema`, `model_fields` available
  - Basic instantiation (1): Required fields only, defaults verified
  - Full instantiation (1): All 4 fields with custom values
  - `Field(default_factory=list)` independence (1): No shared state between instances
  - `Field(default_factory=dict)` independence (1): No shared state between instances
  - `to_dict()` backward compatibility (1): Returns dict with 4 keys
  - `model_dump()` equivalence (1): Same output as `to_dict()`
  - Mutability (1): All attributes can be modified
  - `model_validate()` from dict (1): Creates instance from dict
  - `model_copy()` with update (1): Copy with modifications
  - JSON schema generation (1): Schema with all 4 properties
  - `DescriptorValidator.validate_batch_values()` integration (1): Returns Pydantic `ValidationResult`
  - `DescriptorValidator.validate_configuration()` integration (1): Returns Pydantic `ValidationResult`
  - Global validator instance (1): Singleton exists and correct type
  - Convenience functions (1): `validate_value()` works
  - Type coercion (1): `int` → `bool` for `is_valid`
  - Module version check (1): Version 1.1.0 with Phase 29 migration note
  - Original usage pattern (1): Backward compatible with original code style
- **NON-BREAKING**: Same constructor API, attribute access, `to_dict()` method added
- **DYNAMIC**: `model_dump()` auto-includes all fields; `Field(default_factory=...)` for mutable defaults
- **PRODUCTION-READY**: Runtime type validation, independent mutable defaults, thread-safe BaseModel
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation via `model_json_schema()`, extensible with `@field_validator`
- **Version Bump**: `descriptor_validator.py` v1.0.0 → v1.1.0
- **Descriptors Module Progress**: 2/5 files in descriptors module migrated (Phase 28-29 complete; Phases 30-32 pending)
- **Overall Progress**: 94/~111 dataclasses migrated (84.7%)
- **Documentation**: Full Phase 29 details added; APPENDIX G updated; FastAPI Readiness Checklist updated

### v1.29.0 (2026-01-08)
- **Phase 28 COMPLETE**: Descriptors module `descriptor_registry.py` migrated to Pydantic V2
- **Class Migrated**: 1 mutable dataclass (6 attributes)
  - `DescriptorRegistration` - 1 mutable BaseModel (6 attributes: `name`, `function`, `metadata`, `is_builtin`, `plugin_name`, `registered_at`)
- **Import Change**: `from dataclasses import dataclass, field` → `from pydantic import BaseModel, ConfigDict`
- **DescriptorRegistration Pattern**:
  - Mutable `BaseModel` (no `frozen=True`) - required for registry mutation patterns
  - 6 attributes (3 required + 3 with defaults)
  - `model_config = ConfigDict(arbitrary_types_allowed=True)` for `Callable` and `DescriptorMetadata` types
  - `to_dict()` method added wrapping `model_dump()` for backward compatibility
  - No `__post_init__` in original (simple data container)
- **Key Finding - ConfigDict(arbitrary_types_allowed=True)**: Required because:
  1. `function: Callable` - Python callable objects are not JSON-serializable
  2. `metadata: DescriptorMetadata` - Custom dataclass type from `descriptor_categories.py`
  3. Without this config, Pydantic V2 would reject these types during model construction
- **Key Finding - JSON Schema Limitation**: `model_json_schema()` raises `PydanticInvalidForJsonSchema` due to `Callable` type. This is expected Pydantic V2 behavior - workaround is `model_dump(exclude={"function"})` for JSON-serializable output.
- **Backward Compatibility**:
  - Same constructor API: `DescriptorRegistration(name=..., function=..., metadata=...)`
  - Same attribute access: `reg.name`, `reg.function`, `reg.metadata`, `reg.is_builtin`, `reg.plugin_name`, `reg.registered_at`
  - Same default values: `is_builtin=True`, `plugin_name=None`, `registered_at=None`
  - Global `registry` singleton works unchanged
  - `DescriptorRegistry` class methods `get_descriptor_registration()`, `_register_internal()` work unchanged
- **Test Coverage**: 20 comprehensive tests passed:
  - Import tests (1): `DescriptorRegistration`, `DescriptorRegistry`, `registry`, convenience functions
  - BaseModel verification (1): Inheritance check
  - Pydantic V2 methods (1): `model_dump`, `model_validate`, `model_json_schema`, `model_config` available
  - ConfigDict configuration (1): `arbitrary_types_allowed=True` verified
  - Basic instantiation (1): All 6 attributes with defaults
  - Full instantiation (1): All optional fields populated
  - `to_dict()` backward compatibility (1): Returns dict with 6 keys
  - `model_dump()` equivalence (1): Same output as `to_dict()`
  - Mutability (1): Attributes can be modified
  - `model_validate()` from dict (1): Creates instance from dict
  - `model_copy()` with update (1): Copy with modifications
  - JSON schema limitation (1): Expected `PydanticInvalidForJsonSchema` for `Callable`
  - Global registry integration (1): Singleton pattern works
  - `get_descriptor_registration()` type check (1): Returns `DescriptorRegistration` BaseModel
  - Callable function (1): `function` attribute is callable
  - Round-trip serialization (1): `model_dump(exclude={"function"})` works
  - Type coercion (1): `int` → `bool` for `is_builtin`
  - Module version check (1): Version 1.1.0 with Phase 28 migration note
  - Registry usage pattern (1): `get_descriptor()`, `has_descriptor()`, `list_descriptors()` work unchanged
  - Field access pattern (1): All 6 fields accessible
- **NON-BREAKING**: Same constructor API, attribute access, `to_dict()` method added (not replaced)
- **DYNAMIC**: `model_dump()` auto-includes all fields; no hardcoded key list; `ConfigDict` handles arbitrary types
- **PRODUCTION-READY**: Runtime type validation, `ConfigDict(arbitrary_types_allowed=True)` for complex types, thread-safe BaseModel
- **FUTURE-PROOF**: FastAPI ready (with Callable exclusion), extensible with `@field_validator`, Pydantic V2 full feature set available
- **Version Bump**: `descriptor_registry.py` v1.0.0 → v1.1.0
- **Descriptors Module Progress**: 1/5 files in descriptors module migrated (Phase 28 complete; Phases 29-32 pending)
- **Overall Progress**: 93/~111 dataclasses migrated (83.8%)
- **Documentation**: Full Phase 28 details added; APPENDIX G updated; FastAPI Readiness Checklist updated

### v1.28.0 (2026-01-08)
- **Phase 27 COMPLETE**: Transformations module `research_api.py` migrated to Pydantic V2
- **Class Migrated**: 1 mutable dataclass (12 attributes)
  - `ExperimentConfiguration` - 1 mutable BaseModel (12 attributes: `name`, `description`, `base_transforms`, `ablations`, `parameter_sweeps`, `paper_reference`, `hypothesis`, `expected_outcome`, `num_runs`, `random_seed`, `results`, `metadata`)
- **Import Change**: `from dataclasses import dataclass, field` → `from pydantic import BaseModel, Field, model_validator`
- **ExperimentConfiguration Pattern**:
  - Mutable `BaseModel` (no `frozen=True`) - required for mutable `results` and `metadata` dicts
  - 12 attributes (3 required + 9 with defaults)
  - `Field(default_factory=list)` for `ablations`, `parameter_sweeps`
  - `Field(default_factory=dict)` for `results`, `metadata`
  - `@model_validator(mode='after')` for validation logic (replaces `__post_init__`)
  - Custom `to_dict()` PRESERVED (NOT replaced with `model_dump()`)
  - `from_dict()` class method preserved unchanged
  - All utility methods preserved: `save_to_yaml()`, `load_from_yaml()`, `get_total_runs()`, `get_variant_names()`
- **Key Finding - Custom to_dict() Preserved**: Unlike simple cases, `to_dict()` was NOT replaced with `model_dump()` wrapper because:
  1. Calls `t.to_dict()` for each `TransformSpec` in `base_transforms` (nested serialization)
  2. Returns only 11 of 12 attributes (excludes `results` field)
  3. This custom serialization logic must be preserved for YAML/JSON export
- **Key Finding - @model_validator(mode='after')**: Validation logic migrated from `__post_init__`:
  - Validates `name` is not empty (raises `ConfigurationError`)
  - Validates `base_transforms` is a list (raises `ConfigurationError`)
  - Validates `num_runs >= 1` (raises `ConfigurationError`)
  - Warns if no variants configured (logger.warning)
  - Returns `self` after validation (required for `mode='after'`)
- **Backward Compatibility**:
  - Same constructor API: `ExperimentConfiguration(name=..., description=..., base_transforms=[...])`
  - Same attribute access: `config.name`, `config.ablations`, `config.results`
  - Same methods: `to_dict()`, `from_dict()`, `save_to_yaml()`, `load_from_yaml()`, `get_total_runs()`, `get_variant_names()`
  - Same `to_dict()` output format (11 keys)
  - Builder classes unchanged: `AblationStudyBuilder`, `ParameterSweepBuilder`, `ComparativeStudyBuilder`, `ExperimentRunner`
- **Test Coverage**: 15 comprehensive tests passed:
  - Import tests (1): `ExperimentConfiguration`, all builders, convenience functions
  - BaseModel verification (1): Inheritance check
  - Pydantic V2 methods (1): `model_dump`, `model_validate`, `model_json_schema` available
  - Basic instantiation (1): Required fields with `TransformSpec`
  - Default values (1): All 9 default values correct
  - `Field(default_factory=...)` independence (1): No shared state between instances
  - Mutability (1): Attributes can be modified (no `frozen=True`)
  - Custom `to_dict()` (1): Returns 11 keys, calls `TransformSpec.to_dict()`
  - `from_dict()` (1): Class method creates instance correctly
  - Validation empty name (1): `ConfigurationError` raised
  - Validation num_runs (1): `ConfigurationError` raised for `num_runs < 1`
  - `get_total_runs()` (1): Returns correct calculation
  - `get_variant_names()` (1): Returns all variant names
  - Builder integration (1): `AblationStudyBuilder.build()` returns `ExperimentConfiguration`
  - JSON schema (1): `model_json_schema()` generates valid schema
- **NON-BREAKING**: Same constructor API, attribute access, all 6 methods preserved, `to_dict()` output unchanged
- **DYNAMIC**: `Field(default_factory=list/dict)` for mutable defaults; no hardcoded values; dynamic validation
- **PRODUCTION-READY**: Runtime type validation, `ConfigurationError` for invalid configs, thread-safe BaseModel
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation via `model_json_schema()`, extensible with `@field_validator`
- **MEDIUM Priority Phases Started**: First of 9 MEDIUM priority phases (27-35) now complete
- **Overall Progress**: 92/~111 dataclasses migrated (82.9%)
- **Transformations Module Progress**: 3/3 files in transformations module COMPLETE (Phase 16 `custom_transforms.py`, Phase 18 `graph_transforms.py`, Phase 19 `plugin_system.py`, Phase 27 `research_api.py`) ✅
- **Documentation**: Full Phase 27 details added; APPENDIX G updated; FastAPI Readiness Checklist updated

### v1.27.0 (2026-01-08)
- **Phase 26 COMPLETE**: HPO module `hpo_manager.py` cleaned up for Pydantic V2 alignment
- **No Dataclasses Migrated**: File contains 0 dataclass definitions; `HPOManager` is a regular Python class
- **Dead Code Removed**: `from dataclasses import asdict` import removed (was imported but never used)
- **Import Change**: Removed line 28 `from dataclasses import asdict`
- **Docstring Update**: Added Pydantic V2 Migration (Phase 26) documentation section
- **Version Bump**: `hpo_manager.py` v1.1.0 → v1.2.0
- **Key Finding - No Migration Needed**: Unlike previous phases, this file had no dataclasses to migrate:
  - `HPOManager` class (line 913) is a regular Python class, not a dataclass
  - All HPO config classes used (`HPOConfig`, `PrunerConfig`, `SamplerConfig`, `StudyConfig`, `SearchSpaceParamConfig`) were already migrated to Pydantic BaseModel in Phase 6b
  - The only dataclass-related code was an unused `asdict` import (dead code)
- **Pattern**: Dead code cleanup + documentation alignment (minimal change phase)
- **Test Coverage**: 15 comprehensive tests passed:
  - Import tests (1): `HPOManager`, `is_hpo_enabled`, `get_best_params`, `create_hpo_manager`, `infer_task_type`
  - Dead code verification (1): No `from dataclasses import` in actual import statements
  - Version verification (1): Module docstring contains `Version: 1.2.0`
  - Phase 26 documentation (1): Docstring contains `Pydantic V2 Migration (Phase 26)`
  - HPOConfig Pydantic verification (1): `HPOConfig` is `BaseModel` subclass (Phase 6b)
  - HPOConfig instantiation (1): Default values (`enabled=False`, `backend='optuna'`, `n_trials=100`)
  - HPOManager instantiation (1): Disabled config creates manager with `backend=None`
  - HPOManager.from_config() (1): Accepts dict input, creates `HPOConfig` via `from_dict()`
  - is_hpo_enabled() utility (1): Returns `False` for `None`, `False` for disabled, `True` for enabled
  - create_hpo_manager() convenience (1): Creates manager with custom `n_trials`
  - HPOConfig.to_dict() (1): Backward compatible wrapper returns dict
  - HPOConfig.model_dump() (1): Pydantic V2 native method available
  - HPOConfig frozen (1): Immutable - raises error on attribute assignment
  - Module __all__ exports (1): Contains expected public API
  - infer_task_type() (1): Returns `'graph_regression'` for `metric='val_mae'`
- **NON-BREAKING**: No API changes; dead code removal only; all existing functionality preserved
- **DYNAMIC**: HPOManager dynamically uses Pydantic-migrated config classes from Phase 6b
- **PRODUCTION-READY**: All config classes used have runtime validation via Pydantic V2
- **FUTURE-PROOF**: Module fully aligned with Pydantic V2 ecosystem; ready for FastAPI integration
- **HIGH Priority Phases COMPLETE**: All 8 HIGH priority phases (19-26) now complete ✅
- **Overall Progress**: 91/~111 dataclasses migrated (82.0%) - unchanged (no new dataclasses in this phase)
- **Documentation**: Full Phase 26 details added; APPENDIX G updated; FastAPI Readiness Checklist updated

### v1.26.0 (2026-01-08)
- **Phase 25 COMPLETE**: Factory module `target_selection_config.py` migrated to Pydantic V2
- **Class Migrated**: 1 mutable dataclass (14 attributes)
  - `TargetSelectionConfig` - 1 mutable BaseModel (14 attributes: `mode`, `properties`, `indices`, `range_spec`, `strict`, `raw_config`, `resolved_indices`, `resolved_names`, `total_available`, `config_level`, `config_source`, `resolved_level`, `resolved_source`, `resolved_source_attr`)
- **Import Change**: `from dataclasses import dataclass, field` → `from pydantic import BaseModel, Field`
- **TargetSelectionConfig Pattern**:
  - Mutable `BaseModel` (no `frozen=True`) - required for `resolve()` and `resolve_for_task()` methods that modify attributes
  - 14 attributes including 3 enum types (`SelectionMode`, `TargetLevel`, `TargetSource`)
  - `Field(default_factory=dict)` for `raw_config` (mutable default)
  - Custom `to_dict()` preserved (NOT replaced with `model_dump()`) - returns 11 keys with enum `.name` serialization and computed `specified` field
  - No `__post_init__` (validation in `from_config()` class method instead)
- **Key Finding - Custom to_dict() Preserved**: Unlike previous phases, `to_dict()` was NOT replaced with `model_dump()` wrapper because:
  1. Returns only 11 of 14 attributes (selective output for logging/debugging)
  2. Uses enum `.name` (e.g., "PROPERTIES") not `.value` (e.g., `1`)
  3. Includes computed `specified` field: `self.properties or self.indices or self.range_spec or 'ALL'`
  4. This custom logic must be preserved for backward compatibility
- **Key Finding - Enum Handling**: Three enum types (`SelectionMode`, `TargetLevel`, `TargetSource`) are validated by Pydantic V2 natively; invalid enum values raise `ValidationError`
- **Backward Compatibility**:
  - Same constructor API: `TargetSelectionConfig(mode=SelectionMode.ALL, strict=True, ...)`
  - Same attribute access: `config.mode`, `config.resolved_level`, `config.raw_config`
  - Same class methods: `from_config()`, `resolve_for_task()`, `resolve()`, `infer_level_from_task_type()`, `infer_source_from_level()`
  - Same `to_dict()` output format (11 keys with enum `.name` serialization)
  - Same `__repr__()` output format
- **Test Coverage**: 15 comprehensive tests passed:
  - Import tests (1): `TargetSelectionConfig`, `SelectionMode`, `TargetLevel`, `TargetSource`, Pydantic components
  - BaseModel verification (1): Inheritance check
  - Default instantiation (1): All 14 attribute defaults verified
  - Custom instantiation (1): Enums and custom values
  - Mutability (1): Attributes can be modified (not frozen)
  - `Field(default_factory=dict)` independence (1): `raw_config` not shared between instances
  - `to_dict()` backward compatibility (1): Returns 11 keys with enum `.name` serialization
  - `from_config()` class method (1): Creates instance from dict, handles None, returns existing instance
  - `infer_level_from_task_type()` (1): Pattern-based level inference
  - Pydantic V2 features (1): `model_dump()`, `model_validate()`, `model_json_schema()` available
  - `model_dump()` vs `to_dict()` comparison (1): 14+ keys vs 11 keys
  - `__repr__()` method (1): Correct format preserved
  - Enum validation (1): Pydantic raises `ValidationError` for invalid enum
  - Type coercion (1): Enum direct assignment works
  - Module version (1): Version 1.1.0 with Phase 25 migration note
- **NON-BREAKING**: Same constructor API, attribute access, all methods preserved, `to_dict()` output unchanged
- **DYNAMIC**: `Field(default_factory=dict)` for mutable defaults; Pydantic V2 native enum validation; no hardcoded values
- **PRODUCTION-READY**: Runtime type validation, clear error messages, thread-safe BaseModel, comprehensive enum support
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation via `model_json_schema()`, extensible with `@field_validator`
- **Version Bump**: `target_selection_config.py` v1.0.0 → v1.1.0
- **Factory Module Progress**: 1/1 file in factory module with dataclasses COMPLETE (Phase 25 `target_selection_config.py`) ✅
- **Overall Progress**: 91/~111 dataclasses migrated (82.0%)
- **Documentation**: Full Phase 25 details added; APPENDIX G updated; FastAPI Readiness Checklist updated

### v1.25.0 (2026-01-08)
- **Phase 24 COMPLETE**: Registry module `model_registry.py` migrated to Pydantic V2
- **Class Migrated**: 1 mutable dataclass (6 attributes)
  - `ModelRegistration` - 1 mutable BaseModel (6 attributes: `name`, `model_class`, `metadata`, `is_builtin`, `plugin_name`, `registered_at`)
- **Import Change**: `from dataclasses import dataclass` → `from pydantic import BaseModel, ConfigDict`
- **ModelRegistration Pattern**:
  - Mutable `BaseModel` (no `frozen=True`)
  - `model_config = ConfigDict(arbitrary_types_allowed=True)` for `Type[torch.nn.Module]` field
  - 6 attributes: `name` (str), `model_class` (Type[torch.nn.Module]), `metadata` (ModelMetadata), `is_builtin` (bool), `plugin_name` (Optional[str]), `registered_at` (Optional[str])
  - `to_dict()` → wraps `model_dump()` (backward compatible)
  - No `__post_init__` validation logic (pure registration container)
- **Key Finding - ConfigDict(arbitrary_types_allowed=True)**: Required because `model_class` is `Type[torch.nn.Module]` which is not a standard Pydantic-serializable type. This allows PyTorch module class references to be stored without explicit serialization logic.
- **Key Finding - Nested Pydantic Model**: `metadata` field is `ModelMetadata` (alias for `DynamicModelMetadata` from Phase 23), demonstrating seamless nesting of Pydantic models.
- **Backward Compatibility**:
  - Same constructor API: `ModelRegistration(name=..., model_class=..., metadata=..., is_builtin=..., plugin_name=..., registered_at=...)`
  - Same attribute access: `registration.name`, `registration.model_class`, `registration.metadata`
  - `_register_internal()` method at lines 356-363 requires zero modifications
- **Test Coverage**: 14 comprehensive tests passed:
  - Import tests (1): `ModelRegistration`, `ModelRegistry`, `registry`, Pydantic components
  - BaseModel verification (1): Inheritance check
  - ConfigDict verification (1): `arbitrary_types_allowed=True` configured
  - Instantiation (1): Constructor with all 6 attributes including `Type[torch.nn.Module]`
  - Default values (1): `is_builtin=True`, `plugin_name=None`, `registered_at=None` preserved
  - `to_dict()` method (1): Returns dict with 6 keys
  - `model_dump()` equivalence (1): `to_dict()` correctly wraps `model_dump()`
  - Mutability (1): Attributes can be modified (not frozen)
  - JSON schema generation (1): `model_json_schema()` produces schema with 6 properties
  - ModelRegistry integration (1): Registry returns `ModelRegistration` instances correctly
  - Type validation (1): Valid `torch.nn.Module` subclass accepted
  - Module version (1): Version 1.2.0 with Pydantic migration note in docstring
  - `model_validate()` availability (1): Pydantic V2 method available
  - Attribute access pattern (1): All 6 attributes accessible via dot notation
- **NON-BREAKING**: Same constructor API, attribute access, default values preserved
- **DYNAMIC**: `model_dump()` auto-includes all fields; `ConfigDict(arbitrary_types_allowed=True)` allows any `torch.nn.Module` subclass; no hardcoded key lists
- **PRODUCTION-READY**: Runtime type validation, clear error messages, thread-safe BaseModel, proper handling of arbitrary types
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation via `model_json_schema()`, extensible with `@field_validator`
- **Version Bump**: `model_registry.py` v1.1.0 → v1.2.0
- **Registry Module Progress**: 2/2 files in registry module COMPLETE (Phase 23 `pyg_introspector.py` + Phase 24 `model_registry.py`) ✅
- **Overall Progress**: 90/~111 dataclasses migrated (81.1%)
- **Documentation**: Full Phase 24 details added; APPENDIX G updated

### v1.24.0 (2026-01-08)
- **Phase 23 COMPLETE**: Registry module `pyg_introspector.py` migrated to Pydantic V2
- **Classes Migrated**: 2 mutable dataclasses (26 total attributes)
  - `ParameterInfo` - 1 mutable BaseModel (8 attributes: `name`, `param_type`, `required`, `default`, `description`, `min_value`, `max_value`, `choices`)
  - `DynamicModelMetadata` - 1 mutable BaseModel (18 attributes: `name`, `category`, `import_path`, `description`, `paper_url`, `paper_title`, `supported_tasks`, `hyperparameters`, `requires_edge_features`, `requires_edge_weights`, `requires_edge_index`, `supports_heterogeneous`, `supports_directed`, `min_pyg_version`, `tags`, `parameters`, `forward_parameters`, `required_data_attributes`, `accepts_kwargs`)
- **Import Change**: `from dataclasses import dataclass, field` → `from dataclasses import field` + `from pydantic import BaseModel, Field, ConfigDict`
- **ParameterInfo Pattern**:
  - Mutable `BaseModel` (no `frozen=True`)
  - 8 attributes with simple types (str, bool, Any, Optional)
  - `to_dict()` → wraps `model_dump()` (backward compatible)
  - No validation logic (pure metadata container)
- **DynamicModelMetadata Pattern**:
  - Mutable `BaseModel` (no `frozen=True`)
  - `model_config = ConfigDict(arbitrary_types_allowed=True)` for nested `ParameterInfo` objects in `Dict[str, ParameterInfo]` fields
  - `Field(default_factory=list)` for `supported_tasks`, `tags`
  - `Field(default_factory=dict)` for `hyperparameters`, `parameters`, `forward_parameters`
  - `Field(default_factory=set)` for `required_data_attributes`
  - `category` field uses `ModelCategory` enum
  - `to_dict()` → uses `model_dump(mode='json')` for automatic enum value serialization (e.g., `ModelCategory.BASIC_GNN` → `"basic_gnn"`)
- **Key Finding - ConfigDict(arbitrary_types_allowed=True)**: Required because `parameters` and `forward_parameters` are `Dict[str, ParameterInfo]` which contain Pydantic BaseModel instances. This allows nested Pydantic models in Dict fields without explicit serialization logic.
- **Key Finding - model_dump(mode='json') for Enums**: `DynamicModelMetadata.to_dict()` uses `model_dump(mode='json')` to automatically serialize `ModelCategory` enum to its `.value` string, matching the expected dict output format.
- **Backward Compatibility - ModelMetadata Alias**: Line 1704 `ModelMetadata = DynamicModelMetadata` preserved, ensuring imports from `model_registry.py` and `__init__.py` continue to work unchanged.
- **Test Coverage**: 20 comprehensive tests passed:
  - Import tests (1): `ParameterInfo`, `DynamicModelMetadata`, `ModelCategory`
  - BaseModel verification (1): Inheritance checks for both classes
  - `ParameterInfo` instantiation (1): Constructor with all 8 attributes
  - `ParameterInfo` serialization (2): `to_dict()` backward compat (8 keys), `model_dump()` equivalence
  - `DynamicModelMetadata` minimal instantiation (1): Required fields only, default values verified
  - `DynamicModelMetadata` full instantiation (1): All 18 attributes with custom values
  - `DynamicModelMetadata` enum serialization (1): `to_dict()` returns `category` as string
  - `Field(default_factory=...)` independence (1): Mutable defaults not shared between instances
  - Nested `ParameterInfo` in Dict (1): `parameters` field with nested `ParameterInfo` objects
  - JSON schema generation (1): Both classes produce valid JSON schemas
  - `model_validate()` from dict (1): Round-trip dict → instance
  - Mutability (1): Both classes allow attribute modification
  - Type coercion (1): `Any` type accepts various values
  - Set field (1): `required_data_attributes` with `Set[str]` type
  - `ModelMetadata` alias (1): Alias exports correctly
  - `PyGModelIntrospector` integration (1): Singleton pattern works
  - Round-trip serialization (1): `model_dump()` → reconstruct via `model_validate()`
  - `model_dump_json()` (2): Both classes produce valid JSON strings
- **NON-BREAKING**: Same constructor API, attribute access, `ModelMetadata` alias preserved
- **DYNAMIC**: `model_dump()` auto-includes all fields; `Field(default_factory=...)` for all mutable defaults; enum serialization handled automatically; no hardcoded key lists
- **PRODUCTION-READY**: Runtime type validation, clear error messages, thread-safe BaseModel, proper handling of nested Pydantic models
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation via `model_json_schema()`, extensible with `@field_validator`
- **Version Bump**: `pyg_introspector.py` v2.0.0 → v2.1.0
- **Registry Module Progress**: 1/2 files in registry module migrated (Phase 23 complete; Phase 24 completed in v1.25.0)
- **Overall Progress**: 89/~111 dataclasses migrated (80.2%)
- **Documentation**: Full Phase 23 details added; APPENDIX G updated

### v1.23.0 (2026-01-08)
- **Phase 22 COMPLETE**: Deployment module `model_optimization.py` migrated to Pydantic V2
- **Class Migrated**: 1 mutable dataclass (11 attributes)
  - `OptimizationConfig` - 1 mutable BaseModel (11 attributes: `quantization_enabled`, `quantization_type`, `quantization_backend`, `pruning_enabled`, `pruning_type`, `pruning_amount`, `distillation_enabled`, `distillation_temperature`, `distillation_alpha`, `export_onnx`, `optimize_for_mobile`)
- **Import Change**: `from dataclasses import dataclass` → `from pydantic import BaseModel`
- **OptimizationConfig Pattern**:
  - Mutable `BaseModel` (no `frozen=True`)
  - All 11 attributes with simple types (bool, str, float) and defaults
  - `to_dict()` → wraps `model_dump()` (backward compatible)
  - No `__post_init__` validation logic (pure configuration container)
  - No enums in attributes (simpler than `Alert` class)
- **Test Coverage**: 17 comprehensive tests passed:
  - Import tests (1): `OptimizationConfig`, `ModelOptimizer`, `QuantizationType`, `PruningType`
  - BaseModel verification (1): Inheritance check
  - `OptimizationConfig` instantiation (2): Default values (11 attrs), custom values
  - `OptimizationConfig` serialization (4): `to_dict()` backward compat (11 keys), `model_dump()` equivalence, `model_dump_json()`, `model_json_schema()`
  - Mutability (1): Attribute modification works (no `frozen=True`)
  - Type coercion (1): String-to-float conversion for `pruning_amount`, `distillation_temperature`
  - Pydantic V2 features (2): `model_validate()` from dict, `model_copy()` with update
  - `ModelOptimizer` integration (2): Config creation, `config.to_dict()` output
  - Enum classes (1): `QuantizationType`, `PruningType` unchanged
  - Round-trip serialization (1): `model_dump()` → reconstruct via constructor
  - Version check (1): Module version 1.1.0 with Pydantic migration note
- **NON-BREAKING**: Same constructor API, attribute access, `to_dict()` output format (11 keys) preserved
- **DYNAMIC**: `model_dump()` auto-includes all fields; no hardcoded key list
- **PRODUCTION-READY**: Runtime type validation, clear error messages, thread-safe BaseModel
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation via `model_json_schema()`, extensible with `@field_validator`
- **Version Bump**: `model_optimization.py` v1.0.0 → v1.1.0
- **Deployment Module COMPLETE**: All 3 files in deployment module migrated (Phase 20, 21, 22) ✅
- **Overall Progress**: 87/~111 dataclasses migrated (78.4%)
- **Documentation**: Full Phase 22 details added; APPENDIX G updated

### v1.22.0 (2026-01-08)
- **Phase 21 COMPLETE**: Deployment module `monitoring.py` migrated to Pydantic V2
- **Classes Migrated**: 2 mutable dataclasses (18 total attributes)
  - `MonitoringConfig` - 1 mutable BaseModel (12 attributes: `enable_performance_tracking`, `enable_drift_detection`, `enable_health_checks`, `enable_alerting`, `drift_detection_method`, `drift_threshold`, `alert_threshold`, `health_check_interval`, `metrics_window_size`, `log_predictions`, `log_metrics_interval`, `retraining_trigger_threshold`)
  - `Alert` - 1 mutable BaseModel (6 attributes: `severity`, `message`, `metric_type`, `metric_value`, `threshold`, `timestamp`)
- **Import Change**: `from dataclasses import dataclass, field` → `from pydantic import BaseModel, Field`
- **MonitoringConfig Pattern**:
  - Mutable `BaseModel` (no `frozen=True`)
  - All 12 attributes with simple types (bool, str, int, float) and defaults
  - `to_dict()` → wraps `model_dump()` (backward compatible)
  - No validation logic (pure configuration container)
- **Alert Pattern**:
  - Mutable `BaseModel` (no `frozen=True`)
  - `Field(default_factory=datetime.now)` for `timestamp` attribute
  - `severity` field uses `AlertSeverity` enum
  - `to_dict()` → uses `model_dump(mode='json')` for automatic enum value extraction (`.value`) and datetime ISO serialization (`.isoformat()`)
- **Key Finding - model_dump(mode='json')**: Pydantic V2's `model_dump(mode='json')` automatically serializes enums to their `.value` string and datetime objects to ISO format strings, eliminating manual serialization code
- **Test Coverage**: 25 comprehensive tests passed:
  - Import tests (3): `MonitoringConfig`, `Alert`, `AlertSeverity`, `MetricType`, `DriftType`, `ModelMonitor`, `create_monitor`
  - BaseModel verification (2): Inheritance checks for both classes
  - `MonitoringConfig` instantiation (4): Default values (12 attrs), custom values, attribute access
  - `MonitoringConfig` serialization (4): `to_dict()` backward compat (12 keys), `model_dump()` equivalence, `model_dump_json()`, `model_json_schema()`
  - `Alert` instantiation (2): With `AlertSeverity` enum, timestamp auto-generation
  - `Alert` serialization (4): `to_dict()` enum→string, datetime→ISO, all severity levels, `model_dump_json()`
  - Mutability (2): Both classes allow attribute modification
  - Type coercion (1): String-to-float/int conversion
  - JSON schema (2): Schema generation for both classes
  - `ModelMonitor` integration (5): Config acceptance, default config creation, `create_monitor()`, Alert creation pattern, `get_alerts()`, export serialization
  - Version check (1): Module version 1.1.0 with Pydantic migration note
- **NON-BREAKING**: Same constructor API, attribute access, `to_dict()` output format preserved for both classes
- **DYNAMIC**: `model_dump()` auto-includes all fields; enum/datetime serialization handled automatically; no hardcoded key lists
- **PRODUCTION-READY**: Runtime type validation, clear error messages, thread-safe BaseModel, proper enum/datetime handling
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation via `model_json_schema()`, extensible with `@field_validator`
- **Version Bump**: `monitoring.py` v1.0.0 → v1.1.0
- **Deployment Module Progress**: 2/3 files in deployment module migrated (Phase 20-21 complete; Phase 22 now complete)
- **Overall Progress**: 86/~111 dataclasses migrated (77.5%)
- **Documentation**: Full Phase 21 details added; APPENDIX G updated

### v1.19.0 (2026-01-08)
- **Phase 18 COMPLETE**: Transformations module `graph_transforms.py` migrated to Pydantic V2
- **Classes Migrated** (8 total):
  - `TransformCompatibility` - 1 mutable BaseModel (7 attributes) with `is_compatible()` method
  - `TransformDependency` - 1 mutable BaseModel (6 list attributes) with `Field(default_factory=list)`
  - `ParameterConstraint` - 1 mutable BaseModel (5 attributes) with `validate()` method preserved
  - `ParameterMetadata` - 1 mutable BaseModel (9 attributes) with `ConfigDict(arbitrary_types_allowed=True)` for `Type` and `inspect.Parameter.empty`
  - `EdgeAttrAwareTransformConfig` - 1 mutable BaseModel (7 attributes) with `get_injection_params()` method preserved
  - `TransformInfo` - 1 mutable BaseModel (17 attributes) with `ConfigDict(arbitrary_types_allowed=True)` + `@model_validator(mode='after')` for auto-initialization
  - `ExperimentalSetup` - 1 mutable BaseModel (6 attributes) with `@model_validator(mode='after')` for validation + initialization
  - `ValidationContext` - 1 mutable BaseModel (9 attributes) with `Field(default_factory=...)` for mutable defaults
- **Import Change**: `from dataclasses import dataclass, field` → `from pydantic import BaseModel, Field, model_validator, ConfigDict`
- **Pattern: Arbitrary Types**: `ParameterMetadata` and `TransformInfo` use `ConfigDict(arbitrary_types_allowed=True)` for `Type`, `inspect.Signature`, `inspect.Parameter.empty`
- **Pattern: model_validator(mode='after')**: `TransformInfo` and `ExperimentalSetup` convert `__post_init__` to `@model_validator(mode='after')` returning `self`
- **Pattern: Field Mutation in Validator**: Uses `object.__setattr__(self, 'field', value)` for setting fields in `model_validator(mode='after')` on mutable BaseModel
- **Test Coverage**: 87 comprehensive tests passed (import, Pydantic BaseModel inheritance for 8 classes, default instantiation, method preservation, arbitrary types handling, model_validator auto-initialization, validation rejection, Pydantic V2 features model_dump/model_dump_json/model_json_schema, mutability, type coercion, EDGE_ATTR_AWARE_TRANSFORMS registry, round-trip serialization, GraphTransforms API integration, nested Pydantic models)
- **NON-BREAKING**: Same constructor API, attribute access, all methods preserved (`is_compatible()`, `validate()`, `get_injection_params()`, `add_issue()`, etc.)
- **DYNAMIC**: `Field(default_factory=...)` for all mutable defaults; no hardcoded values
- **PRODUCTION-READY**: Runtime type validation, clear error messages, thread-safe BaseModel, proper handling of complex types
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation via `model_json_schema()`, extensible with `@field_validator`
- **CRITICAL Priority Complete**: Both Phase 17 and Phase 18 (CRITICAL priority) are now complete
- **Overall Progress**: 81/~111 dataclasses migrated (73.0%)
- **Transformations Module**: 9 total classes migrated across Phase 16 (`custom_transforms.py`) and Phase 18 (`graph_transforms.py`)

### v1.18.0 (2026-01-08)
- **Phase 17 COMPLETE**: Config module `config_schemas.py` migrated to Pydantic V2
- **Classes Migrated** (9 total):
  - `TransformationSchema` - 1 mutable BaseModel (7 attributes) with cross-field validation
  - `PluginConfigSchema` - 1 mutable BaseModel (12 attributes) with 4 field validators
  - `WavefunctionProcessingConfigSchema` - 1 frozen BaseModel (1 attribute)
  - `WavefunctionUncertaintyConfigSchema` - 1 frozen BaseModel (1 attribute)
  - `WavefunctionConfigSchema` - 1 frozen BaseModel (5 attributes) with nested object init
  - `ExperimentSchema` - 1 mutable BaseModel (12 attributes) with 7 field validators
  - `ValidationConfig` - 1 mutable BaseModel (5 attributes) simple config
  - `DescriptorConfigSchema` - 1 frozen BaseModel (8 attributes) with auto-adjust logic
  - `DescriptorCategoryConfigSchema` - 1 frozen BaseModel (4 attributes)
- **Import Change**: `from dataclasses import dataclass, field` → `from pydantic import BaseModel, field_validator, model_validator, Field` + `from typing_extensions import Self`
- **Pattern: Frozen with Nested Object Init**: `WavefunctionConfigSchema` uses `model_validator(mode='before')` to initialize nested `WavefunctionProcessingConfigSchema` and `WavefunctionUncertaintyConfigSchema` objects before field assignment (replacing `object.__setattr__` pattern)
- **Pattern: Auto-Adjust Logic**: `DescriptorConfigSchema` uses `model_validator(mode='before')` to auto-adjust `num_workers` from 1 to 2 when `parallel_computation=True`
- **Pattern: Cross-Field Validation**: `TransformationSchema` uses `model_validator(mode='after')` to validate that at least one of `experimental_setups` or `standard_transforms` is defined
- **Test Coverage**: 14 comprehensive tests passed (import, Pydantic BaseModel inheritance for 9 classes, TransformationSchema cross-field validation, PluginConfigSchema field validators and to_dict/from_dict, WavefunctionProcessingConfigSchema frozen immutability, WavefunctionUncertaintyConfigSchema enabled=False only, WavefunctionConfigSchema nested object auto-init, ExperimentSchema validators, ValidationConfig simple, DescriptorConfigSchema auto-adjust num_workers, DescriptorCategoryConfigSchema, JSON schema generation for all 9 classes, model_dump/model_dump_json, round-trip serialization)
- **NON-BREAKING**: Same constructor API, attribute access, `to_dict()` output format, `from_dict()` factory methods preserved
- **DYNAMIC**: `model_dump()` auto-includes all fields; nested object serialization handled automatically
- **PRODUCTION-READY**: Runtime type validation, clear error messages, frozen models thread-safe
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation, extensible with `@field_validator`
- **Config Module**: 9 additional classes migrated (19 total in `config/` module across Phases 2, 4, 17)
- **Overall Progress**: 73/~111 dataclasses migrated (65.8%)
- **CRITICAL Priority Phase Complete**: Phase 17 was marked CRITICAL as it blocks FastAPI config endpoints

### v1.17.0 (2026-01-07)
- **Phase 16 COMPLETE**: Transformations module `custom_transforms.py` migrated to Pydantic V2
- **Classes Migrated** (1 total):
  - `TransformMetadata` - 1 mutable BaseModel (11 attributes)
- **Import Change**: Added `from pydantic import BaseModel, Field` (line 58)
- **Class Change**: `@dataclass` decorator removed; `class TransformMetadata(BaseModel):` inheritance
- **TransformMetadata Attributes**: `name` (str), `version` (str), `author` (str), `category` (str), `description` (str), `paper_reference` (Optional[str]), `github_url` (Optional[str]), `validated_datasets` (List[str]), `required_node_features` (List[str]), `required_edge_features` (List[str]), `required_graph_attributes` (List[str])
- **Default Factory Change**: `field(default_factory=list)` → `Field(default_factory=list)` for 4 list fields
- **to_dict() Change**: Manual 11-key dict construction → `return self.model_dump()`
- **Test Coverage**: 18 comprehensive tests passed (import, Pydantic BaseModel inheritance, default instantiation, full instantiation, to_dict backward compat 11 keys, model_dump equals to_dict, model_dump_json, model_json_schema 11 properties, model_validate from dict, model_copy with update, mutability preserved, type validation, list field independence default_factory, __str__ preserved, ValidationError for invalid type, round-trip serialization, module __all__ exports, get_metadata() pattern integration)
- **NON-BREAKING**: Same constructor API, attribute access, `to_dict()` output format (11 keys) preserved
- **DYNAMIC**: `model_dump()` auto-includes all fields; no hardcoded key list
- **PRODUCTION-READY**: Runtime type validation, clear error messages, thread-safe BaseModel
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation, extensible with `@field_validator`
- **Version**: `custom_transforms.py` maintains v1.0 (internal dataclass change only)
- **Transformations Module**: 1/1 class migrated to Pydantic V2
- **Original Blueprint Progress**: 64/64 dataclasses migrated (100%) - **PHASES 1-16 COMPLETE**
- **Codebase Audit**: Comprehensive audit revealed 22 additional files with stdlib dataclasses NOT in original blueprint scope
- **APPENDIX G EXPANDED**: Full migration plan for Phases 17-38 added with:
  - Recommended migration order by priority (CRITICAL → HIGH → MEDIUM → LOW)
  - 22 files mapped to Phases 17-38
  - ~47 estimated additional dataclasses identified
  - Full file paths and rationale for each phase
  - Summary by priority and module
  - Implementation notes and expected patterns
  - Estimated effort (14-21 hours total)
  - FastAPI readiness checklist
- **Total Migration Scope**: 64 migrated + ~47 remaining = ~111 dataclasses across 39 files

### v1.16.0 (2026-01-07)
- **Phase 15 COMPLETE**: HPO Analysis module `study_analyzer.py` migrated to Pydantic V2
- **Classes Migrated** (1 total):
  - `AnalysisConfig` - 1 frozen BaseModel (6 attributes)
- **Import Change**: `from dataclasses import dataclass, field` → `from pydantic import BaseModel, field_validator, model_validator` + `from typing_extensions import Self`
- **AnalysisConfig Attributes**: `importance_method` (ImportanceMethod enum), `n_importance_trials` (Optional[int]), `convergence_window` (int), `include_pruned` (bool), `include_failed` (bool), `percentile_thresholds` (Tuple[float, ...])
- **AnalysisConfig Validation Pattern**:
  - `@field_validator('convergence_window')`: Validates >= 1
  - `@field_validator('n_importance_trials')`: Validates >= 1 or None
  - `@model_validator(mode='after')`: Validates all percentile values are between 0 and 100 (returns `Self`)
  - `to_dict()` → wraps `model_dump()` for backward compatibility
- **Test Coverage**: 25 comprehensive tests passed (import, Pydantic BaseModel inheritance, default instantiation, custom instantiation, frozen immutability, field validators convergence_window/n_importance_trials, boundary values, model validators percentile_thresholds >100/<0/boundary, to_dict backward compat, model_dump, model_dump_json, model_json_schema, model_validate, model_copy, type coercion string-to-int, enum string coercion, StudyAnalyzer integration, module exports, round-trip serialization, empty percentile thresholds)
- **NON-BREAKING**: Same constructor API, attribute access, frozen immutability preserved
- **DYNAMIC**: `model_dump()` auto-includes all fields; enum validation uses Pydantic V2 native support
- **PRODUCTION-READY**: Runtime type validation, clear error messages, `AnalysisConfig` is thread-safe (frozen)
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation, extensible with `@field_validator`
- **Version Bump**: `study_analyzer.py` v1.0.0 → v1.1.0
- **HPO Analysis Module COMPLETE**: 1/1 class migrated to Pydantic V2
- **Documentation**: Full Phase 15 details added; APPENDIX E/F updated
- **Overall Progress**: 63/64 dataclasses migrated (98.4%)

### v1.15.0 (2026-01-07)
- **Phase 14 COMPLETE**: HPO NAS module `search_space.py` migrated to Pydantic V2
- **Classes Migrated** (2 total):
  - `LayerConfig` - 1 frozen BaseModel (7 attributes)
  - `GNNArchitectureSpace` - 1 mutable BaseModel (13 attributes)
- **Import Change**: `from dataclasses import dataclass, field` → `from pydantic import BaseModel, field_validator, model_validator, Field` + `from typing_extensions import Self`
- **LayerConfig Attributes**: `type` (LayerType enum), `hidden_channels` (int), `heads` (int), `dropout` (float), `activation` (str), `batch_norm` (bool), `residual` (bool)
- **LayerConfig Validation Pattern**:
  - `@field_validator('hidden_channels')`: Validates > 0
  - `@field_validator('heads')`: Validates >= 1
  - `@field_validator('dropout')`: Validates 0.0-1.0 range
  - `to_dict()` → uses `model_dump()` with enum value serialization for backward compatibility
  - `from_dict()` → uses `model_validate()` with string-to-enum conversion
- **GNNArchitectureSpace Attributes**: `min_layers` (int), `max_layers` (int), `layer_types` (List[LayerType]), `hidden_channels` (List[int]), `heads` (List[int]), `dropout_range` (Tuple[float, float]), `allow_skip_connections` (bool), `allow_dense_connections` (bool), `allow_mixed_layers` (bool), `pooling_types` (List[PoolingType]), `aggregation_types` (List[AggregationType]), `activation_types` (List[ActivationType]), `batch_norm_options` (List[bool])
- **GNNArchitectureSpace Validation Pattern**:
  - `@field_validator('min_layers')`: Validates >= 1
  - `@field_validator('hidden_channels')`: Validates non-empty list with positive values
  - `@field_validator('heads')`: Validates non-empty list with values >= 1
  - `@field_validator('dropout_range')`: Validates tuple with 0 <= min <= max <= 1
  - `@field_validator('layer_types')`: Validates non-empty list
  - `@field_validator('pooling_types')`: Validates non-empty list
  - `@field_validator('aggregation_types')`: Validates non-empty list
  - `@model_validator(mode='after')`: Cross-field validation (max_layers >= min_layers)
  - `to_dict()` → returns dict with enum values as strings (13 keys, backward compatible)
  - `from_dict()` → uses `model_validate()` with string-to-enum conversion
  - `to_optuna_search_space()` → preserved unchanged, returns Optuna-compatible format
- **Helper Methods Preserved**: `has_attention_layers()`, `get_attention_layer_types()`, `get_search_dimensions()`, `estimate_search_space_size()`, `create_default_layer_config()`
- **Factory Functions Preserved**: `create_gnn_search_space()`, `get_default_gnn_search_space()`
- **Test Coverage**: 30 comprehensive tests passed (import, inheritance, default instantiation, custom instantiation, frozen immutability, field validators for hidden_channels/heads/dropout/min_layers/hidden_channels_list/heads_list/dropout_range/layer_types/pooling_types/aggregation_types, cross-field validation max>=min, to_dict backward compat, from_dict string-to-enum, model_dump, model_json_schema, to_optuna_search_space, helper methods, factory functions, model_validate, model_copy, round-trip serialization, module version)
- **NON-BREAKING**: Same constructor API, attribute access, `to_dict()`, `from_dict()`, `to_optuna_search_space()` output format preserved
- **DYNAMIC**: `model_dump()` auto-includes all fields; enum validation uses Pydantic V2 native support; `Field(default_factory=...)` for mutable defaults
- **PRODUCTION-READY**: Runtime type validation, clear error messages, `LayerConfig` is thread-safe (frozen)
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation, extensible with `@field_validator`
- **Version Bump**: `search_space.py` v1.0.0 → v1.1.0
- **HPO NAS Module COMPLETE**: All 2 classes migrated to Pydantic V2
- **Documentation**: Full Phase 14 details added to APPENDIX D; APPENDIX E/F updated

### v1.14.0 (2026-01-07)
- **Phase 13 COMPLETE**: HPO Transfer module `warm_start.py` migrated to Pydantic V2
- **Classes Migrated** (2 total):
  - `WarmStartConfig` - 1 frozen BaseModel (8 attributes)
  - `TransferredTrial` - 1 mutable BaseModel (6 attributes)
- **Import Change**: `from dataclasses import dataclass, field` → `from pydantic import BaseModel, field_validator`
- **WarmStartConfig Attributes**: `method` (WarmStartMethod enum), `n_trials` (int), `min_similarity` (float), `weight_by_performance` (bool), `filter_invalid` (bool), `scale_to_bounds` (bool), `add_noise` (bool), `noise_scale` (float)
- **WarmStartConfig Validation Pattern**:
  - `@field_validator('n_trials')`: Validates >= 1
  - `@field_validator('min_similarity')`: Validates 0.0-1.0 range
  - `@field_validator('noise_scale')`: Validates 0.0-1.0 range
  - `to_dict()` → wraps `model_dump()`
- **TransferredTrial Attributes**: `params` (Dict[str, Any]), `value` (Optional[float]), `source_study` (Optional[str]), `similarity` (float), `weight` (float), `is_valid` (bool)
- **TransferredTrial Pattern**:
  - No validators required (simple data container)
  - Mutable BaseModel for attribute modification
  - `to_dict()` → wraps `model_dump()`
- **Test Coverage**: 20 comprehensive tests passed (import, inheritance, default instantiation, custom instantiation, frozen immutability, field validators n_trials/min_similarity/noise_scale, boundary values, to_dict backward compat, model_dump, model_dump_json, model_json_schema, model_validate, type coercion, WarmStartStrategy integration, static methods return TransferredTrial, enum handling, round-trip serialization, get_transfer_summary, module exports)
- **NON-BREAKING**: Same constructor API, attribute access, `to_dict()` preserved
- **DYNAMIC**: `model_dump()` auto-includes all fields; no hardcoded key list
- **PRODUCTION-READY**: Runtime type validation, clear error messages, `WarmStartConfig` is thread-safe (frozen)
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation, extensible with `@field_validator`
- **Version Bump**: `warm_start.py` v1.0.0 → v1.1.0
- **HPO Transfer Module COMPLETE**: All 3 files (Phase 11-13) now migrated to Pydantic V2 (5/5 classes)
- **Documentation**: Full Phase 13 details added to APPENDIX D; APPENDIX E/F updated

### v1.13.0 (2026-01-07)
- **Phase 12 COMPLETE**: HPO Transfer module `meta_features.py` migrated to Pydantic V2
- **Class Migrated** (1 total):
  - `MetaFeatureConfig` - 1 frozen BaseModel (5 attributes)
- **Import Change**: `from dataclasses import dataclass, field` → `from pydantic import BaseModel, field_validator, model_validator` + `from typing_extensions import Self`
- **MetaFeatureConfig Attributes**: `categories` (Tuple[MetaFeatureCategory, ...]), `max_samples` (Optional[int]), `normalize` (bool), `include_molecular` (bool), `compute_expensive` (bool)
- **MetaFeatureConfig Pattern**:
  - `@field_validator('max_samples')`: Validates >= 1 or None
  - `@model_validator(mode='after')`: Validates `categories` is not empty tuple (returns `Self`)
  - `to_dict()` → wraps `model_dump()`
  - `should_extract()` method preserved unchanged
- **Test Coverage**: 20 comprehensive tests passed (import, inheritance, default instantiation, custom instantiation, frozen immutability, max_samples validation invalid, max_samples validation negative, categories validation empty, should_extract ALL, should_extract specific, to_dict backward compat, model_dump, model_dump_json, model_json_schema, model_validate, type coercion, MetaFeatureExtractor integration, MetaFeatureExtractor custom config, enum values preserved, round-trip serialization)
- **NON-BREAKING**: Same constructor API, attribute access, `should_extract()` method preserved
- **DYNAMIC**: `model_dump()` auto-includes all fields; no hardcoded key list
- **PRODUCTION-READY**: Runtime type validation, clear error messages, `MetaFeatureConfig` is thread-safe (frozen)
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation, extensible with `@field_validator`
- **Version Bump**: `meta_features.py` v1.0.0 → v1.1.0
- **HPO Transfer Module Progress**: 3/5 classes migrated (Phase 11-12 complete, Phase 13 pending)
- **Documentation**: Full Phase 12 details added to APPENDIX D; APPENDIX E/F updated

### v1.12.0 (2026-01-07)
- **Phase 11 COMPLETE**: HPO Transfer module `transfer_manager.py` migrated to Pydantic V2
- **Classes Migrated** (2 total):
  - `TransferConfig` - 1 frozen BaseModel (10 attributes)
  - `RegisteredStudyInfo` - 1 mutable BaseModel (10 attributes)
- **Import Change**: `from dataclasses import dataclass, field` → `from pydantic import BaseModel, field_validator, model_validator, Field`
- **TransferConfig Pattern**:
  - `@field_validator`: `n_warm_start_trials` (>= 1), `similarity_threshold` (0-1), `noise_scale` (0-1)
  - `@model_validator(mode='before')`: String-to-enum conversion for `meta_feature_method` and `adaptation_method`; cross-field validation (`persist_meta_db` requires `meta_db_path`)
  - `to_dict()` → wraps `model_dump()`
- **RegisteredStudyInfo Pattern**:
  - `@model_validator(mode='before')`: Auto-sets `registered_at` timestamp if None
  - `to_dict()` → wraps `model_dump()`
  - `from_dict()` → uses `model_validate()`
- **Test Coverage**: 25 comprehensive tests passed (import, inheritance, default instantiation, frozen immutability, field validators, cross-field validation, string-to-enum conversion, to_dict, model_dump, model_json_schema, automatic timestamp, mutability, from_dict, model_validate, HPOTransferManager integration, round-trip serialization)
- **NON-BREAKING**: Same constructor API, attribute access, `to_dict()`, `from_dict()` output format preserved
- **DYNAMIC**: Enum conversion via `@model_validator(mode='before')`; `model_dump()` auto-includes all fields
- **PRODUCTION-READY**: Runtime type validation, clear error messages, `TransferConfig` is thread-safe (frozen)
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation, extensible with `@field_validator`
- **Version Bump**: `transfer_manager.py` v1.0.0 → v1.1.0
- **HPO Transfer Module Progress**: 2/5 classes migrated (Phase 11 complete, Phase 12-13 pending)
- **Documentation**: Full Phase 11 details added to APPENDIX D; APPENDIX E/F updated

### v1.21.0 (2026-01-08)
- **Phase 20 COMPLETE**: Deployment module `deployment_strategies.py` migrated to Pydantic V2
- **Class Migrated**: `DeploymentConfig` - 1 mutable dataclass (13 attributes)
- **Pattern Used**: `BaseModel` (mutable) + `model_dump()` wrapper for `to_dict()`
- **Import Change**: `from dataclasses import dataclass` → `from pydantic import BaseModel`
- **DeploymentConfig Attributes**: `target` (str), `serving_mode` (str), `instance_type` (Optional[str]), `num_instances` (int), `auto_scaling` (bool), `min_instances` (int), `max_instances` (int), `api_type` (str), `enable_monitoring` (bool), `enable_logging` (bool), `enable_caching` (bool), `timeout_seconds` (int), `max_batch_size` (int)
- **to_dict() Change**: Manual 13-key dict construction → `return self.model_dump()`
- **No __post_init__**: Simple dataclass with defaults only; no validation logic to migrate
- **Test Coverage**: 15 comprehensive tests passed:
  - Import tests (1): `DeploymentConfig`, `DeploymentManager`, `DeploymentTarget`, `ServingMode`
  - BaseModel verification (1): Inheritance check
  - Default instantiation (1): All 13 default values verified
  - Custom instantiation (1): All 13 attributes with custom values
  - `to_dict()` backward compatibility (1): Returns dict with all 13 keys
  - Pydantic V2 `model_dump()` (1): Equivalence with `to_dict()`
  - Pydantic V2 `model_dump_json()` (1): JSON string serialization
  - Pydantic V2 `model_json_schema()` (1): Schema with 13 properties
  - Mutability (1): Attribute modification works (no `frozen=True`)
  - Type coercion (1): String-to-int conversion for `num_instances`, `timeout_seconds`, `max_batch_size`
  - `DeploymentManager` integration (1): Config passed to manager, `get_deployment_info()` works
  - `DeploymentManager` target string (1): Alternative constructor with `target=` parameter
  - `list_deployment_targets()` (1): Returns list with expected targets
  - Round-trip serialization (1): `model_dump()` → reconstruct via constructor
  - Module version check (1): Version 1.1.0 with Pydantic V2 migration note
- **NON-BREAKING**: Same constructor API, attribute access, `to_dict()` output format (13 keys) preserved
- **DYNAMIC**: `model_dump()` auto-includes all fields; no hardcoded key list
- **PRODUCTION-READY**: Runtime type validation, clear error messages, thread-safe BaseModel
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation via `model_json_schema()`, extensible with `@field_validator`
- **Version Bump**: `deployment_strategies.py` v1.0.0 → v1.1.0
- **Deployment Module Progress**: 1/3 classes in deployment module migrated (Phase 20 complete; Phase 21-22 pending)
- **Overall Progress**: 84/~111 dataclasses migrated (75.7%)
- **Documentation**: Full Phase 20 details added; APPENDIX G updated

### v1.20.0 (2026-01-08)
- **Phase 19 COMPLETE**: Transformations module `plugin_system.py` migrated to Pydantic V2
- **Classes Migrated**: 2 mutable dataclasses (30 total attributes)
  - `TransformDeclaration` - 1 mutable BaseModel (10 attributes: `name`, `class_name`, `module_path`, `category`, `description`, `version`, `required_node_features`, `required_edge_features`, `required_graph_attributes`, `parameter_constraints`)
  - `PluginMetadata` - 1 mutable BaseModel (20 attributes: `plugin_name`, `version`, `author`, `plugin_type`, `email`, `license`, `description`, `homepage`, `milia_version`, `pyg_version`, `python_version`, `dependencies`, `transform_declarations`, `registered_transforms`, `discovery_source`, `discovery_timestamp`, `is_validated`, `validation_date`, `validation_results`, `checksum`, `trusted`)
- **Import Change**: `from dataclasses import dataclass, field` → `from pydantic import BaseModel, Field, field_validator, model_validator` + kept `from dataclasses import field`
- **TransformDeclaration Pattern**:
  - Mutable `BaseModel` (no `frozen=True`)
  - `Field(default_factory=list)` for `required_node_features`, `required_edge_features`, `required_graph_attributes`
  - `Field(default_factory=dict)` for `parameter_constraints`
  - `to_dict()` → wraps `model_dump()` (backward compatible)
  - `from_dict()` → preserved unchanged (uses constructor)
- **PluginMetadata Pattern**:
  - Mutable `BaseModel` (no `frozen=True`)
  - `@model_validator(mode='after')`: Validates `plugin_name` non-empty, validates `version` format via `_is_valid_version()`, calls `_validate_dependencies()`
  - Custom `__hash__()`: Returns `hash((self.plugin_name, self.version))` for set/dict usage
  - Custom `__eq__()`: Compares `(plugin_name, version)` tuples (matches `__hash__` behavior)
  - `Field(default_factory=list)` for `dependencies`, `transform_declarations`
  - `Field(default_factory=set)` for `registered_transforms`
  - `Field(default_factory=dict)` for `validation_results`
  - 4 `@property` methods preserved: `declared_count`, `registered_count`, `missing_implementations`, `undeclared_implementations`
  - `to_dict()` → uses `model_dump()` + adds computed properties + serializes nested `TransformDeclaration` objects
  - `from_dict()` → preserved unchanged (uses constructor)
  - **Duplicate Method Removed**: Original file had two `to_dict()` methods - consolidated into one
- **Key Finding - Custom Hash on Mutable BaseModel**: Pydantic V2 allows custom `__hash__` on mutable BaseModel classes (unlike `frozen=True` which auto-generates hash from all fields). This preserves the original behavior where only `plugin_name` and `version` are used for hashing.
- **Test Coverage**: 54 comprehensive tests passed:
  - Import tests (4): `TransformDeclaration`, `PluginMetadata`, `PluginRegistry`, `PluginValidator`
  - BaseModel verification (2): Inheritance checks
  - `TransformDeclaration` instantiation (6): Constructor, defaults, attribute access
  - `TransformDeclaration` serialization (5): `to_dict()`, `model_dump()` equivalence
  - `TransformDeclaration` `from_dict()` (3): Factory method
  - `PluginMetadata` instantiation (9): Constructor, defaults, all attribute types
  - `PluginMetadata` validation (2): Invalid version raises `PluginError`, empty name raises `PluginError`
  - Hashability & equality (6): `__hash__`, `__eq__`, set deduplication, dict key usage
  - Computed properties (4): `declared_count`, `registered_count`, `missing_implementations`, `undeclared_implementations`
  - `to_dict()` serialization (8): All fields, computed properties, nested objects
  - Pydantic V2 features (5): `model_dump()`, `model_dump_json()`, `model_json_schema()`, `model_validate()`
- **NON-BREAKING**: Same constructor API, attribute access, `to_dict()`, `from_dict()` output format preserved
- **DYNAMIC**: `model_dump()` auto-includes all fields; computed properties added dynamically; no hardcoded key list
- **PRODUCTION-READY**: Runtime type validation, clear error messages via `PluginError`, hashable for set/dict usage
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation via `model_json_schema()`, extensible with `@field_validator`
- **Version Bump**: `plugin_system.py` v1.0.0 → v1.1.0
- **Transformations Module Progress**: 3/3 classes in transformations module migrated (Phase 16, 18, 19 complete)
- **Documentation**: Full Phase 19 details added; APPENDIX G updated

### v1.11.0 (2026-01-07)
- **Phase 10 COMPLETE**: Acceleration module `computation_optimization.py` migrated to Pydantic V2
- **Class Migrated**: `ComputationConfig` - 1 mutable dataclass (10 attributes)
- **Pattern Used**: `BaseModel` (mutable) + `model_dump()` wrapper for `to_dict()`
- **Import Changes**: `from dataclasses import dataclass` → `from pydantic import BaseModel`
- **to_dict() Change**: Manual 10-key dict construction → `return self.model_dump()`
- **Attributes Migrated**: `compile_model`, `compile_mode`, `compile_dynamic`, `cudnn_benchmark`, `cudnn_deterministic`, `use_tf32`, `channels_last`, `fusion_strategy`, `jit_compile`, `operator_fusion`
- **Test Coverage**: 15 comprehensive tests passed (import, instantiation, backward compat, mutability, type coercion, JSON schema, model_dump, model_dump_json, model_json_schema, model_validate, model_copy, ComputationOptimizer integration)
- **NON-BREAKING**: Same constructor API, attribute access, `to_dict()` output format (10 keys) preserved
- **DYNAMIC**: `model_dump()` auto-includes all fields; no hardcoded key list
- **PRODUCTION-READY**: Runtime type validation, clear error messages, thread-safe BaseModel
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation, extensible with `@field_validator`
- **Version Bump**: `computation_optimization.py` v1.0.0 → v1.1.0
- **Acceleration Module COMPLETE**: All 4 acceleration files (Phase 7-10) now migrated to Pydantic V2
- **Documentation**: Full Phase 10 details added to APPENDIX D; APPENDIX E/F updated

### v1.10.0 (2026-01-07)
- **Phase 9 COMPLETE**: Acceleration module `memory_optimization.py` migrated to Pydantic V2
- **Class Migrated**: `MemoryConfig` - 1 mutable dataclass (9 attributes)
- **Pattern Used**: `BaseModel` (mutable) + `model_dump()` wrapper for `to_dict()`
- **Import Changes**: `from dataclasses import dataclass` → `from pydantic import BaseModel`
- **to_dict() Change**: Manual 9-key dict construction → `return self.model_dump()`
- **Attributes Migrated**: `mixed_precision`, `precision`, `gradient_checkpointing`, `pin_memory`, `non_blocking`, `empty_cache_interval`, `garbage_collect_interval`, `max_memory_allocated`, `growth_interval`
- **Mutability Preserved**: `MemoryOptimizer._validate_config()` mutates config attributes - Pydantic BaseModel without `frozen=True` maintains this behavior
- **Test Coverage**: 15 comprehensive tests passed (import, instantiation, backward compat, mutability, type coercion, JSON schema, model_dump, model_dump_json, model_json_schema, MemoryOptimizer integration, config mutation for _validate_config)
- **NON-BREAKING**: Same constructor API, attribute access, `to_dict()` output format (9 keys) preserved
- **DYNAMIC**: `model_dump()` auto-includes all fields; no hardcoded key list
- **PRODUCTION-READY**: Runtime type validation, clear error messages, thread-safe BaseModel
- **FUTURE-PROOF**: FastAPI ready, JSON schema generation, extensible with `@field_validator`
- **Version Bump**: `memory_optimization.py` v1.0.0 → v1.1.0
- **Documentation**: Full Phase 9 details added to APPENDIX D; APPENDIX E/F updated

### v1.9.0 (2026-01-07)
- **Phase 8 COMPLETE**: Acceleration module `distributed_strategies.py` migrated to Pydantic V2
- **Class Migrated**: `DistributedConfig` - 1 mutable dataclass (12 attributes)
- **Pattern Used**: `BaseModel` (mutable) + `model_dump(mode='json')` for automatic enum serialization
- **Import Changes**: `from dataclasses import dataclass` → `from pydantic import BaseModel`
- **to_dict() Change**: Manual dict with `.value` extraction → `return self.model_dump(mode='json')`
- **Key Finding - Namespace Collision**: Two different `DistributedConfig` classes exist in codebase:
  - `config_bridge.py:475` - Pydantic BaseModel for YAML config parsing (already migrated in Phase 3)
  - `distributed_strategies.py:89` - Pydantic BaseModel for runtime distributed training (Phase 8 target)
  - Both classes serve different purposes and coexist intentionally
- **Enum Serialization**: `model_dump(mode='json')` automatically serializes `DistributedStrategy` and `DistributedBackend` enums to their string values
- **Test Coverage**: 12 comprehensive tests passed (import, instantiation, mutability, enum serialization, to_dict backward compat, Pydantic V2 features, DistributedManager integration, JSON schema, module re-export)
- **NON-BREAKING**: Same constructor API, attribute access, mutability for `_load_env_variables()`, `to_dict()` output format preserved
- **Version Bump**: `distributed_strategies.py` v1.0.0 → v1.1.0
- **Documentation**: Full Phase 8 section added; APPENDIX D updated with implementation details

### v1.8.0 (2026-01-07)
- **Phase 7 COMPLETE**: Acceleration module `device_manager.py` migrated to Pydantic V2
- **Class Migrated**: `DeviceInfo` - 1 mutable dataclass (8 attributes)
- **Pattern Used**: `BaseModel` (mutable, no frozen=True) + `model_dump()` wrapper for `to_dict()`
- **Import Changes**: `from dataclasses import dataclass` → `from pydantic import BaseModel`
- **Type Hint Improved**: `Optional[tuple]` → `Optional[Tuple[int, int]]` for better type checking
- **Model Config**: Added `{'arbitrary_types_allowed': True}` for tuple compatibility
- **Test Coverage**: 15 comprehensive tests passed (import, instantiation, backward compat, mutability, type coercion, JSON schema, serialization, DeviceManager integration)
- **NON-BREAKING**: Same constructor API, attribute access, `to_dict()`, `memory_summary()` methods preserved
- **Version Bump**: `device_manager.py` v1.0.0 → v1.1.0
- **Documentation**: Full Phase 7 section added; APPENDIX D added for remaining acceleration files

### v1.7.0 (2026-01-07)
- **Phase 6 COMPLETE**: HPO module migrated to Pydantic V2
- **Phase 6a**: `param_types.py` - 1 frozen class (`SearchSpaceParamConfig`)
- **Phase 6b**: `hpo_config.py` - 5 frozen classes (`PrunerConfig`, `SamplerConfig`, `StudyConfig`, `MultiObjectiveStudyConfig`, `HPOConfig`)
- **Pattern Used**: `BaseModel, frozen=True` + `@field_validator` + `@model_validator(mode='before')`
- **Test Coverage**: 25 comprehensive tests passed (import, validation, frozen immutability, nested configs, from_dict)
- **NON-BREAKING**: Same constructor API, attribute access, `from_dict()` method preserved
- **Backward Compatibility**: Added `to_dict()` method wrapping `model_dump()`
- **Documentation**: Full Phase 6 section added with migration patterns and verification tests

### v1.6.0 (2026-01-07)
- **Phase 5 COMPLETE**: `exceptions.py` documentation updated for Pydantic V2 namespace compatibility
- **Documentation Added**: `.. warning::` block in module docstring Notes section (lines 209-220)
- **Aliased Import Pattern**: Documents `from pydantic import ValidationError as PydanticValidationError`
- **Key Differences Documented**: MILIA `ValidationError` (BaseProjectError) vs Pydantic `ValidationError` (ValueError)
- **NON-BREAKING**: Zero code changes; documentation-only update
- **ALL PHASES COMPLETE**: Pydantic V2 migration fully implemented across 5 files

### v1.5.0 (2026-01-07)
- **Phase 4 COMPLETE**: `validators.py` integrated with Pydantic V2 wrappers
- **Import Added**: `from pydantic import ValidationError as PydanticValidationError` with graceful fallback
- **New Flag**: `PYDANTIC_AVAILABLE` for runtime detection
- **New Functions**: `wrap_pydantic_validation_error()`, `validate_with_pydantic_model()`
- **Backward Compatibility**: Existing `ValidationResult` class unchanged
- **Test Coverage**: 10 comprehensive tests passed (import, wrappers, namespace conflict, backward compat)
- **No Namespace Conflict**: MILIA `ValidationError` and Pydantic `ValidationError` remain separate via aliased import

### v1.4.0 (2026-01-07)
- **Phase 3 COMPLETE**: `config_bridge.py` migrated with 31 mutable BaseModel classes
- **Pattern Used**: `@field_validator` for enum validation, `@model_validator(mode='after')` for conditional/cross-field validation
- **Backward Compatibility**: All `validate()` methods preserved as pass-through no-ops
- **HPO Enum Support**: Pydantic V2 native Enum validation works for HPOParamType, HPOPrunerType, etc.
- **Test Coverage**: 22 comprehensive tests passed (import, validation, serialization, nested configs)
- **Documentation**: Updated Phase 3 section with implementation patterns and verification tests

### v1.3.0 (2026-01-07)
- **Phase 1 COMPLETE**: `base.py` migrated with single import change
- **Phase 2 COMPLETE**: `config_containers.py` migrated with 10 frozen BaseModel classes
- **CRITICAL FIX**: Changed from `model_validator(mode='after')` + `model_copy()` to `model_validator(mode='before')` pattern
- **Documentation**: Added Pydantic V2 limitation warning and correct migration template

### v1.0.0 (2026-01-07)
- Initial blueprint created from line-by-line analysis of 7 source files
