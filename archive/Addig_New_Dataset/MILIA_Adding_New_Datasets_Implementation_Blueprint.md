# MILIA Pipeline: Adding New Datasets - Implementation Blueprint

**Version:** 2.12.0
**Based on:** MILIA Dataset Architecture Refactoring Plan v2.2.0 (Phase 8-4 Complete) + Handler Module Refactoring v1.2.0 (Phase 7 Migration Complete) + Circular Import Resolution v1.0.0 + QDπ Charged Molecule Support v1.0.0 + Dynamic Discovery Pattern v1.0.0 + Feature Tier Support v1.0.0 + YAML Splitting Architecture v1.1.0 (Full Colocation)
**Architecture:** Protocol + ABC + Explicit Registry + Preprocessing Subsystem + Modular Handler System + Lazy Import Pattern + Dynamic Auto-Discovery + Tier-Aware Validation + YAML Splitting (Single-file/Split-file Configuration Modes with Full Colocation)
**Evidence-Based:** All instructions derived from actual refactored source code analysis
**Updated:** 2026-02-01 - YAML Splitting Architecture v1.1.0: Full Colocation - Each dataset's `configs/datasets/{dataset}.yaml` now contains ALL dataset-specific configuration: `{dataset}_config` + `data_config.property_selection.{DATASET}` + `property_availability.{DATASET}`. The separate `configs/data_config.yaml` file has been removed.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Architecture Summary](#3-architecture-summary)
   - [3.1 CRITICAL: Dataset Class vs Handler Class](#31-critical-dataset-class-vs-handler-class)
4. [Step-by-Step Implementation Guide](#4-step-by-step-implementation-guide)
   - [Step 1: Analyze Your Dataset Requirements](#step-1-analyze-your-dataset-requirements)
   - [Step 2: Create the Dataset Implementation File](#step-2-create-the-dataset-implementation-file)
   - [Step 3: Add Configuration Section](#step-3-add-configuration-section)
   - [Step 4: Validation and Testing](#step-4-validation-and-testing)
5. [Decision Trees](#5-decision-trees)
6. [Complete Implementation Templates](#6-complete-implementation-templates)
7. [Handler Class Integration (Optional)](#7-handler-class-integration-optional)
   - [7.1 Optional: config_constants.py Fallback Entries](#71-optional-config_constantspy-fallback-entries)
   - [7.2 Critical: process_property_value() Dtype Normalization](#72-critical-process_property_value-dtype-normalization)
8. [Preprocessing Module (For Non-NPZ Source Data)](#8-preprocessing-module-for-non-npz-source-data)
   - [8.1 When Preprocessing Is Required](#81-when-preprocessing-is-required)
   - [8.2 Preprocessing Architecture](#82-preprocessing-architecture)
   - [8.3 Creating a Custom Preprocessor](#83-creating-a-custom-preprocessor)
   - [8.4 Available Utility Functions](#84-available-utility-functions)
   - [8.5 Common Pitfalls When Adding New Datasets](#85-common-pitfalls-when-adding-new-datasets)
   - [8.6 Preprocessor Configuration](#86-preprocessor-configuration)
   - [8.7 Integrating Preprocessor with Dataset](#87-integrating-preprocessor-with-dataset)
9. [Troubleshooting Guide](#9-troubleshooting-guide)
10. [Verification Checklist](#10-verification-checklist)
11. [Appendix: Reference Implementation Analysis](#appendix-reference-implementation-analysis)

---

## 1. Overview

### What This Blueprint Covers

This blueprint provides a detailed, step-by-step guide for adding new dataset types to the MILIA Pipeline software. After the refactoring completed through Phase 8-4, adding a new dataset requires:

| Requirement | Action |
|-------------|--------|
| **Create 1 file** | New dataset class in `milia_pipeline/datasets/implementations/` |
| **Add config updates** | **Split-file mode (Recommended):** Create `configs/datasets/your_dataset.yaml` with FULLY COLOCATED config (`your_dataset_config` + `data_config.property_selection.YourDataset` + `property_availability.YourDataset`), update `configs/main.yaml` (Options list) **OR Single-file mode:** Update `config.yaml` with all sections |
| **Optional: Add fallback entries** | In `config_constants.py` (recommended for robustness, see [Section 7.1](#71-optional-config_constantspy-fallback-entries)) |

**If your source data is NOT in NPZ format** (e.g., .molden, .xyz, .tar.gz archives), you will also need:

| Requirement | Action |
|-------------|--------|
| **Create 1 preprocessor file** | Custom preprocessor in `milia_pipeline/preprocessing/preprocessors/` |
| **Possibly create format parser** | Custom parser in `milia_pipeline/preprocessing/utils/` (if format not supported) |

### Files You Will Create/Modify

**For datasets with NPZ source data:**

**Option A: Split-file mode (YAML Splitting Architecture - Recommended)**
```
milia/
├── milia_pipeline/
│   └── datasets/
│       └── implementations/
│           └── your_dataset.py              # CREATE: New dataset class file (auto-discovered)
│
└── configs/                                  # Split configuration directory
    ├── main.yaml                             # UPDATE: Add "YourDataset" to dataset_type Options comment
    └── datasets/
        └── your_dataset.yaml                 # CREATE: FULLY COLOCATED config file containing:
                                              #   - your_dataset_config (dataset settings)
                                              #   - data_config.property_selection.YourDataset (PyG Data properties)
                                              #   - property_availability.YourDataset (available properties)
```

**Option B: Single-file mode (Backward Compatible)**
```
milia/
├── milia_pipeline/
│   └── datasets/
│       └── implementations/
│           └── your_dataset.py              # CREATE: New dataset class file (auto-discovered)
│
└── config.yaml                               # UPDATE: Add dataset_type Options + dataset config + property_availability + data_config.property_selection
```

**For datasets requiring preprocessing (non-NPZ source):**

**Option A: Split-file mode (YAML Splitting Architecture - Recommended)**
```
milia/
├── milia_pipeline/
│   ├── datasets/
│   │   └── implementations/
│   │       └── your_dataset.py              # CREATE: New dataset class file (auto-discovered)
│   │
│   └── preprocessing/
│       ├── preprocessors/
│       │   └── your_dataset.py              # CREATE: New preprocessor file (auto-discovered)
│       └── utils/
│           └── your_format_parser.py        # CREATE: (Optional) Custom format parser
│
└── configs/                                  # Split configuration directory
    ├── main.yaml                             # UPDATE: Add "YourDataset" to dataset_type Options comment
    └── datasets/
        └── your_dataset.yaml                 # CREATE: FULLY COLOCATED config file containing:
                                              #   - your_dataset_config (dataset settings + preprocessing config)
                                              #   - data_config.property_selection.YourDataset (PyG Data properties)
                                              #   - property_availability.YourDataset (available properties)
```

**Option B: Single-file mode (Backward Compatible)**
```
milia/
├── milia_pipeline/
│   ├── datasets/
│   │   └── implementations/
│   │       └── your_dataset.py              # CREATE: New dataset class file (auto-discovered)
│   │
│   └── preprocessing/
│       ├── preprocessors/
│       │   └── your_dataset.py              # CREATE: New preprocessor file (auto-discovered)
│       └── utils/
│           └── your_format_parser.py        # CREATE: (Optional) Custom format parser
│
└── config.yaml                               # UPDATE: Add dataset_type Options + Dataset config + preprocessing config + property_availability + data_config.property_selection
```

---

## 2. Prerequisites

### Required Knowledge

Before implementing a new dataset, you must understand:

1. **Your dataset's data format** - NPZ structure, available properties
2. **Required properties** - What fields must be present in every molecule
3. **Optional properties** - What fields may or may not be present
4. **Molecule creation strategy** - How molecular connectivity is determined
5. **Coordinate units** - Angstrom or Bohr
6. **Energy units** - Hartree, eV, kcal/mol, or kJ/mol
7. **Feature support** - Which analysis features apply

### Critical Decision: Molecule Creation Strategy

The most important decision when adding a new dataset is choosing the molecule creation strategy:

| Strategy | Use When | Examples |
|----------|----------|----------|
| `identifier_coordinate_based` | Dataset includes parseable chemical identifiers (InChI, SMILES) that encode molecular connectivity | DFT, DMC |
| `coordinate_based` | Dataset has only atom types and 3D coordinates; connectivity must be inferred from geometry | Wavefunction |

**Evidence from source code:**

```python
# From dft.py lines 95-109
@classmethod
def get_molecule_creation_strategy(cls) -> str:
    """
    DFT datasets use identifier_coordinate_based strategy.

    DFT molecular data contains InChI identifiers which encode molecular
    connectivity and bonding. These are parsed to create the molecular graph,
    then QM-optimized coordinates are assigned to preserve exact 3D geometry.
    """
    return 'identifier_coordinate_based'

# From wavefunction.py lines 111-137
@classmethod
def get_molecule_creation_strategy(cls) -> str:
    """
    Wavefunction datasets use coordinate_based strategy.

    CRITICAL: Unlike DFT/DMC, Wavefunction compound IDs are NOT parseable
    chemical identifiers. Molecular connectivity must be inferred directly
    from 3D atomic coordinates using rdDetermineBonds algorithm.
    """
    return 'coordinate_based'
```

### How Molecular Graph Structure is Constructed

Understanding **what data is used** to construct the molecular graph (nodes and edges/bonds) is essential before implementation. The table below summarizes how each existing dataset constructs its molecular graph:

| Dataset | Strategy | Identifier Used for Graph Construction | How Molecular Graph is Built |
|---------|----------|----------------------------------------|------------------------------|
| **DFT** | `identifier_coordinate_based` | **InChI** (SMILES as fallback) | InChI string is parsed to obtain molecular connectivity and bond orders; QM-optimized 3D coordinates are then assigned to atoms |
| **DMC** | `identifier_coordinate_based` | **InChI** (SMILES as fallback) | InChI string is parsed to obtain molecular connectivity and bond orders; QMC-optimized 3D coordinates are then assigned to atoms |
| **QM9** | `identifier_coordinate_based` | **InChI** (SMILES as fallback) | InChI string is parsed to obtain molecular connectivity and bond orders; QM-optimized 3D coordinates are then assigned to atoms |
| **Wavefunction** | `coordinate_based` | **None** (compound_id is for tracking/logging only) | Molecular connectivity is inferred directly from 3D atomic coordinates using `rdDetermineBonds.DetermineBonds()` algorithm |

**Evidence from source code:**

| Source File | Lines | Evidence |
|-------------|-------|----------|
| `dft.py` | 55, 106-107, 116 | `identifier_keys=(('inchi', 'inchi'), ('graphs', 'smiles'))`, returns `'identifier_coordinate_based'` |
| `dmc.py` | 56, 107-108, 117 | `identifier_keys=(('inchi', 'inchi'), ('graphs', 'smiles'))`, returns `'identifier_coordinate_based'` |
| `wavefunction.py` | 76, 134-135, 152 | `identifier_keys=(('compounds', 'compound_id'),)` (label only), returns `'coordinate_based'` |
| `mol_conversion_utils.py` | 606-609 | `rdDetermineBonds.DetermineBonds(mol, charge=molecular_charge)` for coordinate_based strategy |

**Key Distinction:**

1. **`identifier_coordinate_based` strategy**: The molecular graph (atoms as nodes, bonds as edges) is constructed by **parsing the InChI string**, which explicitly encodes molecular connectivity and bond orders. The 3D coordinates from QM calculations are assigned to atoms but do **not** determine connectivity.

2. **`coordinate_based` strategy**: The molecular graph is constructed by **inferring bonds from 3D atomic coordinates** using RDKit's `rdDetermineBonds.DetermineBonds()` algorithm. No chemical identifier is parsed for structure. The `compound_id` field (e.g., `'BrCPxSiSxH4_331'`) exists only for tracking/logging purposes and is **not** used for molecular graph construction.

---

## 3. Architecture Summary

### Core Components (from `base.py`)

The refactored architecture uses three immutable dataclasses and one abstract base class:

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATASET CLASS STRUCTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  @register                         ← Decorator for registration │
│  class YourDataset(BaseDataset):                                │
│      │                                                          │
│      ├── metadata: DatasetMetadata   ← Immutable, frozen        │
│      │   ├── name: str               (unique identifier)        │
│      │   ├── version: str            (semantic version)         │
│      │   ├── description: str        (human-readable)           │
│      │   ├── author: Optional[str]                              │
│      │   └── license: Optional[str]                             │
│      │                                                          │
│      ├── schema: DatasetSchema       ← Immutable, frozen        │
│      │   ├── required_properties: Tuple[str, ...]               │
│      │   ├── optional_properties: Tuple[str, ...] = ()          │
│      │   ├── identifier_keys: Tuple[Tuple[str, str], ...] = ()  │
│      │   ├── coordinate_units: str = 'angstrom'                 │
│      │   └── energy_units: str = 'hartree'                      │
│      │                                                          │
│      ├── features: DatasetFeatures   ← Immutable, frozen        │
│      │   ├── vibrational_analysis: bool = False                 │
│      │   ├── uncertainty_handling: bool = False                 │
│      │   ├── atomization_energy: bool = False                   │
│      │   ├── rotational_constants: bool = False                 │
│      │   ├── frequency_analysis: bool = False                   │
│      │   ├── orbital_analysis: bool = False                     │
│      │   ├── homo_lumo_gap: bool = False                        │
│      │   └── mo_energies: bool = False                          │
│      │                                                          │
│      ├── config_key: str             ← Key in config file        │
│      │                                                          │
│      ├── handler_class: Optional[Type] = None  ← Custom handler │
│      │                                                          │
│      └── Abstract Methods (MUST implement):                     │
│          ├── get_required_properties() -> List[str]             │
│          ├── get_feature_support() -> Dict[str, bool]           │
│          └── get_molecule_creation_strategy() -> str            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Registry Pattern (from `registry.py`)

```python
# The @register decorator automatically registers with the global registry
@register
class YourDataset(BaseDataset):
    ...

# This is equivalent to:
class YourDataset(BaseDataset):
    ...
_default_registry.register(YourDataset)
```

### 3.1 CRITICAL: Dataset Class vs Handler Class

MILIA has **TWO class hierarchies** that work together. Understanding the distinction is essential:

| Class Type | Location | Purpose | Defines identifier_keys? |
|------------|----------|---------|---------------------------|
| **Dataset Class** | `datasets/implementations/your_dataset.py` | Schema, metadata, features definition | **YES** - in `DatasetSchema.identifier_keys` tuple |
| **Handler Class** | `handlers/implementations/your_dataset.py` | Runtime processing, validation, enrichment | Has `get_identifier_keys()` method that reads from Dataset |

#### Why This Matters

When you add a new dataset, the **`identifier_keys` ORDER** that determines which identifier is tried first is defined in your **Dataset class** (`DatasetSchema.identifier_keys`), NOT in the handler class.

**Data Flow:**
```
YourDataset.schema.identifier_keys = (('inchi', 'inchi'), ('smiles', 'smiles'))
                    ↓
config_accessors.get_identifier_keys('YourDataset')
                    ↓
    → queries registry → returns YourDataset class
                    ↓
    → reads schema.identifier_keys
                    ↓
molecule_converter_core.py → iterates through keys → uses first available identifier
```

#### Common Mistake

❌ **WRONG:** Editing `YourDatasetHandler.get_identifier_keys()` in handler class
✅ **CORRECT:** Setting `identifier_keys` tuple order in `YourDataset.schema` in `your_dataset.py`

**Evidence from QM9 Implementation:**
- Initial implementation had SMILES first in `qm9.py` → 1% success rate
- Changed to InChI first in `qm9.py` → 100% success rate
- The handler class was never modified - only the Dataset class schema

### Compile-Time Validation (from `base.py` lines 195-228)

The `__init_subclass__` method validates your class at definition time:

```python
def __init_subclass__(cls, **kwargs):
    """Catches missing attributes immediately when the class is defined."""
    # Skip validation for abstract subclasses
    if ABC in cls.__bases__:
        return

    # Validate required class attributes exist
    required_attrs = ['metadata', 'schema', 'features', 'config_key']
    missing = [attr for attr in required_attrs if not hasattr(cls, attr)]

    if missing:
        raise TypeError(
            f"Dataset class '{cls.__name__}' missing required class attributes: {missing}"
        )

    # Validate attribute types
    if not isinstance(cls.metadata, DatasetMetadata):
        raise TypeError(...)
    if not isinstance(cls.schema, DatasetSchema):
        raise TypeError(...)
    if not isinstance(cls.features, DatasetFeatures):
        raise TypeError(...)
    if not isinstance(cls.config_key, str) or not cls.config_key:
        raise TypeError(...)
```

---

## 4. Step-by-Step Implementation Guide

### Step 1: Analyze Your Dataset Requirements

Before writing any code, answer these questions:

#### 1.1 Dataset Identity

| Question | Your Answer |
|----------|-------------|
| Dataset name (unique identifier)? | e.g., "QM9", "ANI1", "GEOM" |
| Dataset version? | e.g., "1.0.0" |
| Dataset description? | e.g., "QM9 dataset with 134k molecules" |
| Author/organization? | e.g., "Your Name" |
| License (if any)? | e.g., "CC0", "MIT", null |

#### 1.2 Data Properties

| Question | Your Answer |
|----------|-------------|
| NPZ file contains which required fields? | e.g., ('atoms', 'coordinates', 'energy') |
| NPZ file contains which optional fields? | e.g., ('dipole', 'gap', 'forces') |
| How are molecules identified? | e.g., (('inchi', 'inchi'), ('smiles', 'smiles')) |
| Coordinate units in source data? | 'angstrom' or 'bohr' |
| Energy units in source data? | 'hartree', 'eV', 'kcal/mol', 'kJ/mol' |

#### 1.2.1 CRITICAL: Identifier Keys Order

**The order of `identifier_keys` determines priority.** The first identifier that exists in the NPZ data will be used for molecule creation. This order is CRITICAL for successful processing.

| Identifier Priority | Rationale |
|---------------------|-----------|
| **InChI FIRST** | InChI encodes complete hydrogen information explicitly (e.g., `InChI=1S/CH4/h1H4` shows all 4 H atoms). No `AddHs()` call needed. |
| **SMILES SECOND** | SMILES may use implicit hydrogens (e.g., `C` for methane = 1 atom). Requires RDKit `AddHs()` which can fail on unsanitized molecules. |

**MILIA Design Principle:** Always specify **InChI first** when both InChI and SMILES are available:

```python
# ✅ CORRECT - InChI first (MILIA's design principle)
identifier_keys=(('inchi', 'inchi'), ('smiles', 'smiles'))

# ❌ WRONG - SMILES first (causes AddHs failures, atom count mismatches)
identifier_keys=(('smiles', 'smiles'), ('inchi', 'inchi'))
```

**Why This Matters - Real Example from QM9:**

| Molecule | SMILES | SMILES Atoms | InChI | InChI Atoms | QM Coords |
|----------|--------|--------------|-------|-------------|-----------|
| Methane | `C` | 1 (implicit H) | `InChI=1S/CH4/h1H4` | 5 (explicit H) | 5 atoms |
| Ammonia | `N` | 1 (implicit H) | `InChI=1S/H3N/h1H3` | 4 (explicit H) | 4 atoms |
| Water | `O` | 1 (implicit H) | `InChI=1S/H2O/h1H2` | 3 (explicit H) | 3 atoms |

- **With SMILES first:** "Atom count mismatch: SMILES has 1 atoms but QM calculation has 5 atoms" → 99% failure
- **With InChI first:** Perfect atom count match → 100% success

#### 1.3 Molecule Creation Strategy

| Question | Your Answer |
|----------|-------------|
| Do you have parseable chemical identifiers (InChI, SMILES)? | Yes → `identifier_coordinate_based` |
| Must connectivity be inferred from 3D geometry? | Yes → `coordinate_based` |

#### 1.4 Feature Support

| Feature | Applies to Your Dataset? |
|---------|--------------------------|
| vibrational_analysis | Yes/No |
| uncertainty_handling | Yes/No (requires 'std' field) |
| atomization_energy | Yes/No |
| rotational_constants | Yes/No |
| frequency_analysis | Yes/No |
| orbital_analysis | Yes/No |
| homo_lumo_gap | Yes/No |
| mo_energies | Yes/No |

---

### Step 2: Create the Dataset Implementation File

Create a new file: `milia_pipeline/datasets/implementations/your_dataset.py`

#### 2.1 File Template

```python
"""
YourDataset dataset implementation.

This module provides the YourDatasetDataset class which encapsulates all
YourDataset-specific metadata and configuration.

[Add notes about your dataset's characteristics, sources, and any
special considerations for molecule creation or property handling.]
"""

from typing import Dict, List

from milia_pipeline.datasets.base import (
    BaseDataset,
    DatasetMetadata,
    DatasetSchema,
    DatasetFeatures,
)
from milia_pipeline.datasets.registry import register

# OPTIONAL: Import custom handler if you have one
# from milia_pipeline.handlers import YourDatasetHandler


@register
class YourDatasetDataset(BaseDataset):
    """
    YourDataset dataset implementation.

    [Detailed docstring describing:
    - What this dataset contains
    - Source/reference for the data
    - Key characteristics
    - Molecule creation strategy rationale
    - Any special processing requirements]
    """

    # =========================================================================
    # REQUIRED: Metadata Definition
    # =========================================================================

    metadata = DatasetMetadata(
        name="YourDataset",                    # UNIQUE identifier (used in registry)
        version="1.0.0",                       # Semantic version
        description="Description of your dataset",
        author="Your Name or Organization",    # Optional
        license="License identifier",          # Optional (e.g., "CC0", "MIT")
    )

    # =========================================================================
    # REQUIRED: Schema Definition
    # =========================================================================

    schema = DatasetSchema(
        # Properties that MUST be present in every molecule
        required_properties=('atoms', 'coordinates', 'energy'),

        # Properties that MAY be present
        optional_properties=('dipole', 'forces', 'gap'),

        # Identifier key mappings: (npz_key, identifier_type)
        # ⚠️ CRITICAL: Order matters! First available identifier will be used.
        # Always put InChI FIRST when both InChI and SMILES are available.
        # InChI encodes complete H information; SMILES may have implicit H issues.
        # Use empty tuple () if no identifiers available.
        identifier_keys=(('inchi', 'inchi'), ('smiles', 'smiles')),

        # Coordinate units in source data
        # Options: 'angstrom', 'bohr'
        coordinate_units='angstrom',

        # Energy units in source data
        # Options: 'hartree', 'eV', 'kcal/mol', 'kJ/mol'
        energy_units='hartree',
    )

    # =========================================================================
    # REQUIRED: Feature Support Definition
    # =========================================================================

    features = DatasetFeatures(
        vibrational_analysis=False,    # Frequency/vibration data
        uncertainty_handling=False,     # Statistical uncertainties (std)
        atomization_energy=False,       # Atomization energy calculation
        rotational_constants=False,     # Rotational constants
        frequency_analysis=False,       # Frequency analysis
        orbital_analysis=False,         # MO analysis
        homo_lumo_gap=False,            # HOMO-LUMO gap
        mo_energies=False,              # MO energies
    )

    # =========================================================================
    # REQUIRED: Configuration Key
    # =========================================================================

    config_key = "your_dataset_config"  # Key in config file (configs/datasets/your_dataset.yaml or config.yaml)

    # =========================================================================
    # OPTIONAL: Custom Handler Class
    # =========================================================================

    # Uncomment and set if you have a custom handler class:
    # handler_class = YourDatasetHandler

    # =========================================================================
    # REQUIRED: Abstract Method Implementations
    # =========================================================================

    @classmethod
    def get_required_properties(cls) -> List[str]:
        """Return list of required properties for this dataset."""
        return list(cls.schema.required_properties)

    @classmethod
    def get_feature_support(cls) -> Dict[str, bool]:
        """Return feature support dictionary for this dataset."""
        return cls.features.to_dict()

    @classmethod
    def get_molecule_creation_strategy(cls) -> str:
        """
        Return the molecule creation strategy for this dataset.

        Options:
        - 'identifier_coordinate_based': Use parseable identifiers (InChI, SMILES)
          for molecular connectivity, assign 3D coordinates from data

        - 'coordinate_based': Infer molecular connectivity from 3D atomic
          coordinates using rdDetermineBonds algorithm

        Returns:
            str: 'identifier_coordinate_based' or 'coordinate_based'
        """
        return 'identifier_coordinate_based'  # or 'coordinate_based'
```

#### 2.2 Validation Rules (from `base.py`)

Your implementation will be automatically validated at class definition time:

| Attribute | Validation Rule |
|-----------|-----------------|
| `metadata` | Must be `DatasetMetadata` instance |
| `metadata.name` | Must be non-empty string |
| `metadata.version` | Must be non-empty string |
| `metadata.description` | Must be non-empty string |
| `schema` | Must be `DatasetSchema` instance |
| `schema.required_properties` | Must be non-empty tuple |
| `schema.coordinate_units` | Must be 'angstrom' or 'bohr' |
| `schema.energy_units` | Must be 'hartree', 'eV', 'kcal/mol', or 'kJ/mol' |
| `features` | Must be `DatasetFeatures` instance |
| `config_key` | Must be non-empty string |

---

---

### Step 3: Add Configuration Section

**Configuration Modes:** MILIA supports two configuration modes:
- **Split-file mode** (`configs/` directory): Recommended for new projects. Each dataset has its own YAML file with **FULLY COLOCATED** configuration (`{dataset}_config` + `data_config.property_selection` + `property_availability`).
- **Single-file mode** (`config.yaml`): Backward compatible. All configuration in one file.

This guide covers both modes. Choose based on your project setup.

#### 3.1 Configuration Template

**Option A: Split-file mode (Recommended)**

Create a new file: `configs/datasets/your_dataset.yaml`

```yaml
# ══════════════════════════════════════════════════════════════════════════════
# YourDataset Dataset Configuration (FULLY COLOCATED)
# ══════════════════════════════════════════════════════════════════════════════
# Part of MILIA YAML Splitting Architecture v1.1.0 (Full Colocation)
# This file contains ALL YourDataset-specific configuration in one place:
#   1. your_dataset_config - Dataset paths and settings
#   2. data_config.property_selection.YourDataset - Property selection for PyG Data
#   3. property_availability.YourDataset - Available properties
# This file is auto-merged with other config files at runtime

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Dataset Configuration
# ─────────────────────────────────────────────────────────────────────────────
your_dataset_config:
  # Filename for the raw .npz dataset file
  raw_npz_filename: your_dataset.npz

  # URL to download the raw dataset file (optional)
  raw_data_download_url: https://example.com/your_dataset.npz

  # Add any dataset-specific configuration options here
  # Examples from existing datasets:

  # For datasets with uncertainty (like DMC):
  # uncertainty_handling:
  #   uncertainty_field_name: std
  #   use_for_loss_weighting: true
  #   max_uncertainty_threshold: null

  # For datasets with preprocessing (like Wavefunction):
  # processing_config:
  #   feature_tier: standard
  #   preprocessing:
  #     num_molecules: null
  #     cleanup_temp: true

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Property Selection for PyG Data Objects (COLOCATED)
# ─────────────────────────────────────────────────────────────────────────────
# This section specifies which properties from the NPZ file should be included
# in the PyTorch Geometric (PyG) Data object during processing.
# Previously in configs/data_config.yaml - now colocated here for single-file editing
data_config:
  property_selection:
    YourDataset:
      # YourDataset Property Selection
      # Reference: Author et al., Journal Name, Year

      # Scalar graph-level targets to be included in pyg_data.y
      scalar_graph_targets_to_include:
        - energy          # Primary energy target
        # Add other scalar targets as needed

      # Node-level features to be added to pyg_data.x
      node_features_to_add:
        - charges         # Atomic charges for node features
        # - forces        # Uncomment if training force fields

      # Fixed-size vector graph properties
      vector_graph_properties_to_include:
        - dipole          # Molecular dipole moment

      # Variable-length graph properties
      variable_len_graph_properties_to_include: []

      # Dataset-specific settings
      # Calculate atomization energy from total energy
      calculate_atomization_energy_from: energy
      # The name of the key for the calculated atomization energy
      atomization_energy_key_name: energy_ATOM

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Property Availability Matrix (COLOCATED)
# ─────────────────────────────────────────────────────────────────────────────
# This section documents all properties available in your dataset's NPZ file.
# This section is automatically merged with other dataset property_availability
# sections at runtime via deep merge
property_availability:
  YourDataset:
    # Dataset header comment with reference information
    # Reference: Author et al., Journal Name, Year
    # DOI: your-doi-here
    # Level of theory: Method/Basis set
    # Coordinate units: Angstrom | Energy units: Hartree

    molecular_identifiers:
      - compounds
      - inchi
    atomic_structure:
      - atoms
      - coordinates
    scalar_graph_targets:
      - energy
    node_features:
      - charges
    vector_graph_properties:
      - dipole
    variable_len_graph_properties: []
    metadata_fields:
      - _metadata
    uncertainty_fields: []
```

**Option B: Single-file mode (Backward Compatible)**

Edit: `config.yaml`

Add a new section matching your `config_key`:

```yaml
# YourDataset Configuration
your_dataset_config:
  # Filename for the raw .npz dataset file
  raw_npz_filename: your_dataset.npz

  # URL to download the raw dataset file (optional)
  raw_data_download_url: https://example.com/your_dataset.npz

  # Add any dataset-specific configuration options here
  # Examples from existing datasets:

  # For datasets with uncertainty (like DMC):
  # uncertainty_handling:
  #   uncertainty_field_name: std
  #   use_for_loss_weighting: true
  #   max_uncertainty_threshold: null

  # For datasets with preprocessing (like Wavefunction):
  # processing_config:
  #   feature_tier: standard
  #   preprocessing:
  #     num_molecules: null
  #     cleanup_temp: true
```

#### 3.2 Update dataset_type Setting

The `dataset_type` setting specifies which dataset to use.

**Location:**
- **Split-file mode:** `configs/main.yaml`
- **Single-file mode:** `config.yaml` (line ~14)

**Two updates are required:**

**A) Add your dataset to the Options list comment:**

**Split-file mode** (`configs/main.yaml`):
```yaml
# BEFORE:
dataset_type: "DFT"  # Options: "DFT", "DMC", "Wavefunction", "QM9", "ANI1X", "ANI1CCX", "RMD17", "ANI2X", "XXMD", "QDPi"

# AFTER (example adding "YourDataset"):
dataset_type: "DFT"  # Options: "DFT", "DMC", "Wavefunction", "QM9", "ANI1X", "ANI1CCX", "RMD17", "ANI2X", "XXMD", "QDPi", "YourDataset"
```

**Single-file mode** (`config.yaml`):
```yaml
# BEFORE:
dataset_type: "DFT"  # Options: "DFT", "DMC", "Wavefunction", "QM9", "ANI1X", "ANI1CCX"

# AFTER (example adding "YourDataset"):
dataset_type: "DFT"  # Options: "DFT", "DMC", "Wavefunction", "QM9", "ANI1X", "ANI1CCX", "YourDataset"
```

**B) Set dataset_type to use your dataset:**

```yaml
# Dataset Type - manually specify your dataset type
dataset_type: "YourDataset"  # Must match metadata.name exactly
```

**Important:** The dataset name in the Options list must exactly match:
- `metadata.name` in your Dataset class
- The key used in `property_availability` section
- The key used in `data_config.property_selection` section

#### 3.3 Add Property Availability Matrix Entry

The `property_availability` section documents all properties available in your dataset's NPZ file. This serves as both documentation and validation reference.

**YAML Splitting Architecture v1.1.0 - Full Colocation Principle:** In split-file mode, `property_availability` is **FULLY COLOCATED** with each dataset's config file in `configs/datasets/your_dataset.yaml`, along with `data_config.property_selection`. This follows the colocation principle: "Things that change together should be located as close as reasonable."

**Location:**
- **Split-file mode:** `configs/datasets/your_dataset.yaml` (FULLY COLOCATED - already included in Section 3.1 template along with `data_config.property_selection`)
- **Single-file mode:** `config.yaml` → search for `property_availability:` section

**For Split-file mode:** The `property_availability` entry is already included in the dataset config file template in Section 3.1. The MILIA config loader's `_deep_merge_configs()` function automatically combines all `property_availability` sections from all dataset files into a single unified dict at runtime.

**For Single-file mode:** Add your dataset entry following this template:

```yaml
property_availability:
  # ... existing entries (DFT, DMC, Wavefunction, QM9, ANI1X, etc.) ...

  YourDataset:
    # Dataset header comment with reference information
    # Reference: Author et al., Journal Name, Year
    # DOI: your-doi-here
    # Level of theory: Method/Basis set
    # Coordinate units: Angstrom | Energy units: Hartree

    # Molecular identifiers (keys used to identify molecules)
    molecular_identifiers:
      - compounds       # Compound identifiers
      - inchi           # InChI string (if available)
      # - smiles        # SMILES string (if available)

    # Atomic structure properties (always required)
    atomic_structure:
      - atoms           # Atomic numbers/symbols
      - coordinates     # Cartesian coordinates

    # Scalar graph-level properties (single values per molecule)
    scalar_graph_targets:
      - energy          # Total energy (primary target)
      # Add other scalar properties from your NPZ

    # Node-level (atomic) properties (per-atom values)
    node_features:
      - charges         # Atomic charges (if available)
      # - forces        # Atomic forces (if available)

    # Fixed-size vector graph properties (e.g., 3D vectors)
    vector_graph_properties:
      - dipole          # Molecular dipole (if available)

    # Variable-length graph properties
    variable_len_graph_properties: []  # Add if applicable

    # Metadata fields (non-target information)
    metadata_fields:
      - _metadata       # Preprocessing metadata

    # Uncertainty fields (for stochastic methods like DMC)
    uncertainty_fields: []  # Add if applicable (e.g., std for DMC)
```

**Important Notes:**
- Property names must match EXACTLY the keys in your preprocessed NPZ file
- This section is used for documentation and validation
- All scalar targets, node features, and vector properties should be listed here

#### 3.4 Add Dataset Configuration for PyG Data Object

The `data_config.property_selection` section specifies which properties from the NPZ file should be included in the PyTorch Geometric (PyG) Data object during processing.

**Location:**
- **Split-file mode:** `configs/datasets/your_dataset.yaml` (FULLY COLOCATED - already included in Section 3.1 template)
- **Single-file mode:** `config.yaml` → search for `data_config:` → `property_selection:`

**For Split-file mode:** The `data_config.property_selection` entry is already included in the dataset config file template in Section 3.1. The MILIA config loader's `_deep_merge_configs()` function automatically combines all `data_config.property_selection` sections from all dataset files into a single unified dict at runtime. **No separate `configs/data_config.yaml` file is needed.**

**For Single-file mode:** Add your dataset entry following this template:

```yaml
data_config:
  property_selection:
    # ... existing entries (DFT, DMC, Wavefunction, QM9, ANI1X, etc.) ...

    YourDataset:
      # YourDataset Property Selection
      # Reference: Author et al., Journal Name, Year

      # Scalar graph-level targets to be included in pyg_data.y
      scalar_graph_targets_to_include:
        - energy          # Primary energy target
        # Add other scalar targets as needed

      # Node-level features to be added to pyg_data.x
      node_features_to_add:
        - charges         # Atomic charges for node features
        # - forces        # Uncomment if training force fields

      # Fixed-size vector graph properties
      vector_graph_properties_to_include:
        - dipole          # Molecular dipole moment

      # Variable-length graph properties
      variable_len_graph_properties_to_include: []

      # Dataset-specific settings
      # Calculate atomization energy from total energy
      calculate_atomization_energy_from: energy
      # The name of the key for the calculated atomization energy
      atomization_energy_key_name: energy_ATOM
```

**Important Notes:**
- Property names must match keys in `property_availability` section
- `scalar_graph_targets_to_include`: These become `pyg_data.y`
- `node_features_to_add`: These become part of `pyg_data.x`
- `vector_graph_properties_to_include`: These become separate attributes (e.g., `pyg_data.dipole`)
- `calculate_atomization_energy_from`: Specifies which energy to use for atomization energy calculation
- `atomization_energy_key_name`: Name for the calculated atomization energy attribute

---

### Step 4: Validation and Testing

#### 4.1 Verify Registration

```python
from milia_pipeline.datasets import list_all, get, is_registered

# Check if dataset is registered
print(f"Registered datasets: {list_all()}")
# Expected: ['DFT', 'DMC', 'Wavefunction', 'YourDataset']

# Check specific registration
print(f"YourDataset registered: {is_registered('YourDataset')}")
# Expected: True

# Get dataset class
YourDatasetClass = get('YourDataset')
print(f"Dataset class: {YourDatasetClass}")
# Expected: <class 'milia_pipeline.datasets.implementations.your_dataset.YourDatasetDataset'>
```

#### 4.2 Verify Metadata

```python
from milia_pipeline.datasets import get

Dataset = get('YourDataset')

# Check metadata
print(f"Name: {Dataset.metadata.name}")
print(f"Version: {Dataset.metadata.version}")
print(f"Description: {Dataset.metadata.description}")

# Check schema
print(f"Required properties: {Dataset.get_required_properties()}")
print(f"Optional properties: {Dataset.get_optional_properties()}")
print(f"Coordinate units: {Dataset.get_coordinate_units()}")
print(f"Energy units: {Dataset.get_energy_units()}")

# Check features
print(f"Feature support: {Dataset.get_feature_support()}")

# Check molecule creation strategy
print(f"Molecule creation strategy: {Dataset.get_molecule_creation_strategy()}")
```

#### 4.3 Verify Config Loading

```python
from milia_pipeline.config import load_config

config = load_config()

# Check dataset_type
print(f"Dataset type: {config.get('dataset_type')}")

# Check your config section exists
your_config = config.get('your_dataset_config')
print(f"Your config: {your_config}")
```

---

## 5. Decision Trees

### Decision Tree 1: Molecule Creation Strategy

```
Does your dataset include parseable chemical identifiers?
│
├── YES (InChI, SMILES, or similar)
│   │
│   └── Do these identifiers encode COMPLETE molecular connectivity?
│       │
│       ├── YES → Use 'identifier_coordinate_based'
│       │         (Parse identifiers for connectivity, assign 3D coordinates)
│       │
│       └── NO (identifiers are labels only, like compound IDs)
│           │
│           └── Use 'coordinate_based'
│               (Infer connectivity from 3D geometry)
│
└── NO (only atom types and coordinates)
    │
    └── Use 'coordinate_based'
        (Infer connectivity using rdDetermineBonds)
```

### Decision Tree 2: Feature Support Selection

```
For each feature, ask:

vibrational_analysis:
  └── Does dataset contain frequency/vibration data? → True/False

uncertainty_handling:
  └── Does dataset contain statistical uncertainties (std field)? → True/False

atomization_energy:
  └── Can atomization energy be calculated from total energy? → True/False

rotational_constants:
  └── Does dataset contain rotational constant data? → True/False

frequency_analysis:
  └── Can frequency analysis be performed? → True/False

orbital_analysis:
  └── Does dataset contain molecular orbital data? → True/False

homo_lumo_gap:
  └── Does dataset contain or can calculate HOMO-LUMO gap? → True/False

mo_energies:
  └── Does dataset contain molecular orbital energies? → True/False
```

### Decision Tree 3: Coordinate Units

```
What units are coordinates stored in your source NPZ file?
│
├── Angstrom (Å) → coordinate_units='angstrom'
│   (Most common for DFT datasets)
│
└── Bohr (a₀) → coordinate_units='bohr'
    (Common for wavefunction/ab initio datasets)
    Note: MILIA will automatically convert to Angstrom during processing
```

---

## 6. Complete Implementation Templates

### Template A: DFT-like Dataset (identifier_coordinate_based)

Use this template when your dataset has:
- Parseable chemical identifiers (InChI, SMILES)
- Coordinates in Angstrom
- Energies in Hartree
- Standard DFT properties

```python
"""
ExampleDFT dataset implementation.
"""

from typing import Dict, List

from milia_pipeline.datasets.base import (
    BaseDataset,
    DatasetMetadata,
    DatasetSchema,
    DatasetFeatures,
)
from milia_pipeline.datasets.registry import register


@register
class ExampleDFTDataset(BaseDataset):
    """
    Example DFT-like dataset with standard properties.

    Uses identifier_coordinate_based strategy since InChI/SMILES
    identifiers are available for molecular connectivity.
    """

    metadata = DatasetMetadata(
        name="ExampleDFT",
        version="1.0.0",
        description="Example DFT dataset with standard properties",
        author="Example Author",
    )

    schema = DatasetSchema(
        required_properties=('Etot', 'atoms', 'coordinates'),
        optional_properties=('dipole', 'gap', 'forces', 'charges'),
        identifier_keys=(('inchi', 'inchi'), ('smiles', 'smiles')),
        coordinate_units='angstrom',
        energy_units='hartree',
    )

    features = DatasetFeatures(
        vibrational_analysis=False,
        uncertainty_handling=False,
        atomization_energy=True,
        rotational_constants=False,
        frequency_analysis=False,
        orbital_analysis=False,
        homo_lumo_gap=True,
        mo_energies=False,
    )

    config_key = "example_dft_config"

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

### Template B: DMC-like Dataset (with uncertainty)

Use this template when your dataset has:
- Statistical uncertainties
- Monte Carlo or stochastic method results

```python
"""
ExampleMC dataset implementation with uncertainty handling.
"""

from typing import Dict, List

from milia_pipeline.datasets.base import (
    BaseDataset,
    DatasetMetadata,
    DatasetSchema,
    DatasetFeatures,
)
from milia_pipeline.datasets.registry import register


@register
class ExampleMCDataset(BaseDataset):
    """
    Example Monte Carlo dataset with uncertainty handling.

    Includes statistical uncertainties (std field) for energy predictions.
    Uses uncertainty-aware processing and loss weighting.
    """

    metadata = DatasetMetadata(
        name="ExampleMC",
        version="1.0.0",
        description="Example Monte Carlo dataset with uncertainties",
        author="Example Author",
    )

    schema = DatasetSchema(
        required_properties=('Etot', 'std', 'atoms', 'coordinates'),  # Note: 'std' required
        optional_properties=('mc_stats', 'correlation_data'),
        identifier_keys=(('inchi', 'inchi'),),
        coordinate_units='angstrom',
        energy_units='hartree',
    )

    features = DatasetFeatures(
        vibrational_analysis=False,
        uncertainty_handling=True,  # CRITICAL: Enable uncertainty handling
        atomization_energy=False,
        rotational_constants=False,
        frequency_analysis=False,
        orbital_analysis=False,
        homo_lumo_gap=False,
        mo_energies=False,
    )

    config_key = "example_mc_config"

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

### Template C: Wavefunction-like Dataset (coordinate_based)

Use this template when your dataset has:
- Non-parseable identifiers (labels only)
- Coordinates in Bohr
- Orbital/wavefunction data
- Molecular connectivity must be inferred from geometry

```python
"""
ExampleWF dataset implementation with coordinate-based molecule creation.
"""

from typing import Dict, List

from milia_pipeline.datasets.base import (
    BaseDataset,
    DatasetMetadata,
    DatasetSchema,
    DatasetFeatures,
)
from milia_pipeline.datasets.registry import register


@register
class ExampleWFDataset(BaseDataset):
    """
    Example wavefunction dataset with orbital analysis.

    CRITICAL DIFFERENCES FROM DFT-LIKE DATASETS:
    1. Uses 'coordinate_based' strategy (compound IDs are NOT parseable)
    2. Coordinates are in Bohr (automatic conversion to Angstrom)
    3. Molecular charge calculated from n_electrons if available
    4. Molecular connectivity inferred from 3D geometry via rdDetermineBonds
    """

    metadata = DatasetMetadata(
        name="ExampleWF",
        version="1.0.0",
        description="Example wavefunction dataset with orbital analysis",
        author="Example Author",
    )

    schema = DatasetSchema(
        required_properties=('atoms', 'coordinates', 'compounds'),  # Note: compounds is label
        optional_properties=('mo_energies', 'mo_occupations', 'homo_lumo_gap_eV',
                            'total_energy', 'n_electrons'),
        identifier_keys=(('compounds', 'compound_id'),),  # Label only, NOT parseable
        coordinate_units='bohr',  # CRITICAL: Bohr units
        energy_units='eV',
    )

    features = DatasetFeatures(
        vibrational_analysis=False,
        uncertainty_handling=False,
        atomization_energy=False,
        rotational_constants=False,
        frequency_analysis=False,
        orbital_analysis=True,   # Enable orbital analysis
        homo_lumo_gap=True,      # Enable HOMO-LUMO
        mo_energies=True,        # Enable MO energies
    )

    config_key = "example_wf_config"

    @classmethod
    def get_required_properties(cls) -> List[str]:
        return list(cls.schema.required_properties)

    @classmethod
    def get_feature_support(cls) -> Dict[str, bool]:
        return cls.features.to_dict()

    @classmethod
    def get_molecule_creation_strategy(cls) -> str:
        """
        Uses coordinate_based strategy.

        Compound IDs are labels (e.g., 'H2O_001') that cannot be parsed
        for molecular connectivity. The rdDetermineBonds algorithm
        infers bonds from 3D atomic coordinates.
        """
        return 'coordinate_based'
```

### Template D: Minimal Dataset (bare minimum)

Use this template as a starting point for any new dataset:

```python
"""
Minimal dataset implementation.
"""

from typing import Dict, List

from milia_pipeline.datasets.base import (
    BaseDataset,
    DatasetMetadata,
    DatasetSchema,
    DatasetFeatures,
)
from milia_pipeline.datasets.registry import register


@register
class MinimalDataset(BaseDataset):
    """Minimal dataset implementation with required elements only."""

    metadata = DatasetMetadata(
        name="Minimal",
        version="1.0.0",
        description="Minimal dataset for testing",
    )

    schema = DatasetSchema(
        required_properties=('atoms', 'coordinates'),
    )

    features = DatasetFeatures()  # All defaults (False)

    config_key = "minimal_config"

    @classmethod
    def get_required_properties(cls) -> List[str]:
        return list(cls.schema.required_properties)

    @classmethod
    def get_feature_support(cls) -> Dict[str, bool]:
        return cls.features.to_dict()

    @classmethod
    def get_molecule_creation_strategy(cls) -> str:
        return 'coordinate_based'  # Safe default
```

---

## 7. Handler Class Integration (Optional)

If your dataset requires custom processing logic beyond what the generic handler provides, you can create a custom handler class.

### Handler Module Architecture (Refactored - Phase 7 Complete)

The handler module has been fully refactored. The legacy `dataset_handlers.py` has been **REMOVED**:

```
milia_pipeline/handlers/
├── __init__.py                      # Backward-compatible exports (v4.0.0, lazy loading + recursion guard)
├── base_handler.py                  # DatasetHandler ABC + factory functions (~1,527 lines)
├── handler_registry.py              # HandlerRegistry + @register_handler decorator
└── implementations/                 # Individual handler implementations
    ├── __init__.py                  # Dynamic discovery pattern
    ├── dft.py                       # DFTDatasetHandler
    ├── dmc.py                       # DMCDatasetHandler
    ├── wavefunction.py              # WavefunctionDatasetHandler
    ├── qm9.py                       # QM9DatasetHandler
    ├── ani1x.py                     # ANI1xDatasetHandler
    ├── ani1ccx.py                   # ANI1ccxDatasetHandler
    ├── ani2x.py                     # ANI2xDatasetHandler
    ├── rmd17.py                     # RMD17DatasetHandler
    ├── xxmd.py                      # XXMDDatasetHandler
    ├── qdpi.py                      # QDPiDatasetHandler ⭐ NEW (charged molecule support)
    └── your_dataset.py              # YOUR NEW HANDLER (create here)
```

**Key Benefits of Modular Structure (Phase 7 Complete):**
- **Dynamic Discovery:** New handlers auto-register when file is added to `implementations/`
- **No Manual Updates:** No need to modify `__init__.py` after adding handler file
- **Thread-Safe Registry:** Uses `RLock` for concurrent access safety
- **Backward Compatible:** All existing imports work unchanged via lazy loading
- **Factory Functions in base_handler.py:** `create_dataset_handler()`, `validate_dataset_handler_compatibility()`, etc.
- **Recursion Guard:** `_DISCOVERING_HANDLERS` flag prevents infinite loops during import

### When to Create a Custom Handler

| Scenario | Need Custom Handler? |
|----------|---------------------|
| Standard DFT-like properties | No |
| Standard property processing | No |
| Custom validation rules | Yes |
| Custom molecular charge calculation | Yes |
| Custom property enrichment | Yes |
| Non-standard data format | Yes |
| **Dataset contains charged molecules** | **Yes** ⭐ (see QDπ pattern) |

**Note on Charged Molecule Support:** If your dataset contains ions, protonated/deprotonated species, or any non-neutral molecules, you MUST create a custom handler that reads `molecular_charge` from the NPZ file instead of returning a hardcoded 0. See `QDPiDatasetHandler` for the reference implementation.

### Handler Protocol (11 Required Methods)

From `protocols.py`, your custom handler must implement:

```python
class DatasetHandlerProtocol(Protocol):
    def get_dataset_type(self) -> str: ...
    def validate_molecule_data(self, raw_properties_dict, molecule_index, identifier) -> None: ...
    def get_required_properties(self) -> List[str]: ...
    def process_property_value(self, key, value, molecule_index, identifier) -> Any: ...
    def enrich_pyg_data(self, pyg_data, raw_properties_dict, molecule_index, identifier) -> Data: ...
    def get_processing_statistics(self, processed_molecules) -> Dict[str, Any]: ...
    def get_supported_structural_features(self) -> Dict[str, List[str]]: ...
    def get_molecular_charge(self, raw_properties_dict, atomic_numbers, mol_identifier) -> int: ...
    def get_molecule_creation_strategy(self) -> str: ...
    def get_transform_recommendations(self) -> Dict[str, List[str]]: ...
    def get_supported_descriptors(self) -> Dict[str, List[str]]: ...
```

### Creating a Handler Implementation File

Create a new file: `milia_pipeline/handlers/implementations/your_dataset.py`

**Required Structure:**

```python
# milia_pipeline/handlers/implementations/your_dataset.py

"""
YourDataset Handler
===================

Handler for YourDataset with exception integration and transformation system support.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple

from torch_geometric.data import Data

from milia_pipeline.config.config_containers import (
    DatasetConfig,
    FilterConfig,
    ProcessingConfig
)
from milia_pipeline.config.validators import (
    validate_molecular_structure,
    is_value_valid_and_not_nan
)
from milia_pipeline.exceptions import (
    PropertyEnrichmentError,
    MoleculeProcessingError,
    HandlerError,
    HandlerConfigurationError,
    HandlerOperationError,
    HandlerValidationError
)

# Import from refactored base handler
from milia_pipeline.handlers.base_handler import (
    DatasetHandler,
    handle_transform_errors
)
from milia_pipeline.handlers.handler_registry import register_handler

logger = logging.getLogger(__name__)


@register_handler  # ← This decorator auto-registers the handler
class YourDatasetHandler(DatasetHandler):
    """
    Handler for YourDataset with exception integration and
    transformation system support.
    """

    def get_dataset_type(self) -> str:
        return "YourDataset"  # Must match dataset_type in config file

    # Implement all 11 required methods from DatasetHandler ABC
    # See existing handlers (dft.py, ani1x.py) for reference implementations
    ...
```

**Key Points:**
1. Use `@register_handler` decorator - handler auto-registers on import
2. Inherit from `DatasetHandler` (from `base_handler.py`)
3. `get_dataset_type()` must return string matching `dataset_type` in config file
4. No modifications to `__init__.py` required - dynamic discovery handles registration

### Linking Custom Handler to Dataset (CRITICAL: Lazy Import Pattern)

**IMPORTANT:** Do NOT use module-level imports for handler classes. This causes circular import dependencies:

```
datasets/implementations/{dataset}.py
    → handlers/implementations/{handler}.py (module-level)
        → config containers → dataset registry → {dataset}.py (CYCLE!)
```

**CORRECT PATTERN:** Override `create_handler()` with lazy import inside the method:

```python
from milia_pipeline.datasets.base import (
    BaseDataset,
    DatasetMetadata,
    DatasetSchema,
    DatasetFeatures,
)
from milia_pipeline.datasets.registry import register


# NOTE: YourDatasetHandler is NOT imported at module level to avoid circular import.
# The handler is registered via @register_handler decorator and discovered dynamically
# through the HandlerRegistry. The create_handler() method uses lazy import to
# instantiate the handler when needed.


@register
class YourDataset(BaseDataset):
    metadata = DatasetMetadata(...)
    schema = DatasetSchema(...)
    features = DatasetFeatures(...)
    config_key = "your_dataset_config"

    # NOTE: handler_class is intentionally NOT set here.
    # YourDatasetHandler is registered via @register_handler decorator and
    # discovered dynamically through the HandlerRegistry by create_dataset_handler().
    # Setting handler_class = None (default from BaseDataset) is correct.
    # We override create_handler() to use lazy import to avoid circular dependency.

    @classmethod
    def create_handler(
        cls,
        dataset_config,
        filter_config,
        processing_config,
        logger,
        experimental_setup=None
    ):
        """
        Factory method to create YourDatasetHandler instance.

        Uses lazy import to avoid circular dependency between
        datasets/implementations/your_dataset.py and handlers/implementations/your_dataset.py.

        This pattern breaks the circular import chain by importing inside the method,
        so the import only happens at runtime when create_handler() is called,
        after all modules are fully loaded.
        """
        # Lazy import to break circular dependency
        from milia_pipeline.handlers.implementations.your_dataset import YourDatasetHandler

        return YourDatasetHandler(
            dataset_config,
            filter_config,
            processing_config,
            logger,
            experimental_setup
        )
```

**DEPRECATED PATTERN (DO NOT USE):**

```python
# ❌ WRONG: Module-level import causes circular dependency
from milia_pipeline.handlers.implementations.your_dataset import YourDatasetHandler

@register
class YourDataset(BaseDataset):
    handler_class = YourDatasetHandler  # ❌ CAUSES CIRCULAR IMPORT
```

**Why Lazy Import Works:**
1. **No module-level import** → No circular dependency at import time
2. **Import inside method** → Import happens at runtime when method called
3. **Handler already registered** → `@register_handler` decorator in `handlers/implementations/` handles registration
4. **Method signature matches** → `create_handler()` signature matches `BaseDataset.create_handler()` exactly
5. **Non-breaking** → Existing code calling `YourDataset.create_handler()` works unchanged
6. **Import from handlers package** → Use `from milia_pipeline.handlers import YourDatasetHandler` (not from implementations directly)

### 7.1 Optional: config_constants.py Fallback Entries

**Architecture Note:** The MILIA pipeline uses a **registry-first** approach where dataset properties are dynamically retrieved from the registered `BaseDataset` subclass. The dictionaries in `config_constants.py` serve as **fallback values** only when registry lookup fails.

**When to Add Fallback Entries:**

| Scenario | Add Fallback Entries? |
|----------|----------------------|
| Registry working correctly (normal operation) | Not required |
| Production deployment requiring maximum robustness | Recommended |
| Debugging registry initialization issues | Recommended |
| Dataset uses non-default properties (e.g., `'energy'` instead of `'Etot'`) | Recommended |

**Fallback Dictionaries to Update:**

If you choose to add fallback entries for robustness, update these dictionaries in `config_constants.py`:

```python
# 1. SUPPORTED_HANDLER_TYPES (line ~382)
SUPPORTED_HANDLER_TYPES: List[str] = ['DFT', 'DMC', 'Wavefunction', 'QM9', 'YourDataset']

# 2. REQUIRED_HANDLER_CONFIG_KEYS (line ~387)
REQUIRED_HANDLER_CONFIG_KEYS: Dict[str, List[str]] = {
    ...
    'YourDataset': ['dataset_type', 'processing_config']
}

# 3. HANDLER_FEATURE_SUPPORT (line ~394)
HANDLER_FEATURE_SUPPORT: Dict[str, Dict[str, bool]] = {
    ...
    'YourDataset': {
        'vibrational_analysis': False,  # Match your DatasetFeatures
        'uncertainty_handling': False,
        'atomization_energy': True,
        # ... other features
    }
}

# 4. HANDLER_REQUIRED_PROPERTIES (line ~443)
HANDLER_REQUIRED_PROPERTIES: Dict[str, List[str]] = {
    ...
    'YourDataset': ['your_energy_key', 'atoms', 'coordinates']  # Match your DatasetSchema
}

# 5. HANDLER_OPTIONAL_PROPERTIES (line ~452)
HANDLER_OPTIONAL_PROPERTIES: Dict[str, List[str]] = {
    ...
    'YourDataset': ['forces', 'charges', ...]  # Match your DatasetSchema
}

# 6. HANDLER_IDENTIFIER_KEYS (line ~461)
HANDLER_IDENTIFIER_KEYS: Dict[str, List[Tuple[str, str]]] = {
    ...
    'YourDataset': [('inchi', 'inchi'), ('smiles', 'smiles')]  # Match your DatasetSchema
    # Or [] for coordinate_based datasets with no identifiers
}

# 7. HANDLER_COORDINATE_UNITS (line ~484)
HANDLER_COORDINATE_UNITS: Dict[str, str] = {
    ...
    'YourDataset': 'angstrom'  # Match your DatasetSchema
}
```

**Important:** All values must exactly match what's defined in your `DatasetSchema` and `DatasetFeatures` classes. The fallback dictionaries should be a mirror of your dataset class definitions.

**ANI-1x Example (coordinate_based dataset with no identifiers):**

```python
# ANI-1x has NO parseable identifiers - uses coordinate_based strategy
'ANI1x': []  # Empty list for HANDLER_IDENTIFIER_KEYS

# ANI-1x uses 'energy' not 'Etot'
'ANI1x': ['energy', 'atoms', 'coordinates']  # For HANDLER_REQUIRED_PROPERTIES
```

### 7.2 Critical: `process_property_value()` Dtype Normalization

The `process_property_value()` method is your **PRIMARY point for dtype normalization**. This method is called for EVERY property BEFORE it reaches core validation.

**Purpose:** Normalize data from NPZ storage format to formats compatible with:
1. `np.isfinite()` validation in `validators.py` (requires numeric dtypes)
2. `torch.tensor()` conversion in `_ensure_tensor()` (requires native dtypes, not object)

**When Required:** Datasets where the preprocessor stores ragged arrays as `dtype=object`.

**Reference Implementation:** `ANI1xDatasetHandler.process_property_value()` in `handlers/implementations/ani1x.py` demonstrates the complete pattern for:
- Detecting object dtype arrays using `arr.dtype == object`
- Converting integer arrays (atoms) to `np.int64`
- Converting float arrays (coordinates, forces, dipole, charges) to `np.float32`/`np.float64`
- Graceful error handling with logging and fallback

**Key Principle:** The handler is the ONLY place to bridge the gap between preprocessor storage format and core pipeline expectations without modifying core files.

**Evidence from ANI-1x Implementation:**
- Without dtype normalization: 0% success rate (all molecules failed)
- With dtype normalization in handler: 98% success rate (49/50 molecules succeeded)

---

## 8. Preprocessing Module (For Non-NPZ Source Data)

This section covers how to create a custom preprocessor when your dataset's source data is **NOT** already in NPZ format. Preprocessing is an **offline, one-time transformation** that converts raw data files into the NPZ format expected by `miliaDataset`.

### 8.1 When Preprocessing Is Required

| Source Data Format | Preprocessing Required? | Notes |
|--------------------|------------------------|-------|
| `.npz` file | **NO** | Directly usable by miliaDataset |
| `.tar.gz` archive with molecular files | **YES** | Need to extract and parse |
| `.molden` files | **YES** | Need WavefunctionPreprocessor or similar |
| `.xyz` files | **YES** | Need custom parser |
| `.log` / `.out` files (QC output) | **YES** | Need custom parser |
| `.json` / `.csv` files | **YES** | Need custom parser |
| Database export | **YES** | Need custom extractor |

**Decision Tree:**
```
Is your source data already in NPZ format?
│
├── YES → Skip preprocessing, proceed to dataset implementation
│
└── NO → You need to create a preprocessor
         │
         ├── Is your format already supported? (check PreprocessorRegistry.list_preprocessors())
         │   │
         │   ├── YES → Use existing preprocessor with your configuration
         │   │
         │   └── NO → Create custom preprocessor (Section 8.3)
         │
         └── Do you need a custom file format parser?
             │
             ├── YES → Create custom parser in utils/ (Section 8.4)
             │
             └── NO → Use existing utilities (archive_handlers, npz_builders)
```

### 8.2 Preprocessing Architecture

The preprocessing subsystem follows a modular, registry-based architecture:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING SUBSYSTEM ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Raw Data (tar.gz, zip, files, etc.)                                    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────┐                        │
│  │         PreprocessorRegistry                 │                        │
│  │  ┌─────────────────────────────────────┐    │                        │
│  │  │ @PreprocessorRegistry.register(...)  │    │                        │
│  │  │ class YourPreprocessor(BasePrep...)  │    │                        │
│  │  └─────────────────────────────────────┘    │                        │
│  └─────────────────────────────────────────────┘                        │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────┐                        │
│  │         BasePreprocessor (ABC)               │                        │
│  │  ├── __init__(config, logger)               │                        │
│  │  ├── _validate_config() [ABSTRACT]          │                        │
│  │  ├── preprocess() -> Path [ABSTRACT]        │                        │
│  │  ├── run() -> Path [CONCRETE]               │                        │
│  │  └── _validate_output(path) [CONCRETE]      │                        │
│  └─────────────────────────────────────────────┘                        │
│           │                                                              │
│           │ Uses                                                         │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────┐                        │
│  │         Utility Functions                    │                        │
│  │  ├── archive_handlers.py                    │                        │
│  │  │   └── extract_from_targz()               │                        │
│  │  ├── format_parsers.py                      │                        │
│  │  │   └── parse_molden_files()  [EXAMPLE]    │                        │
│  │  └── npz_builders.py                        │                        │
│  │      ├── build_npz()                        │                        │
│  │      └── validate_npz_structure()           │                        │
│  └─────────────────────────────────────────────┘                        │
│           │                                                              │
│           ▼                                                              │
│  Output: .npz file ready for miliaDataset                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Module Structure (from source code):**
```
milia_pipeline/preprocessing/
├── __init__.py                  # Module initialization, public API
├── base_preprocessor.py         # BasePreprocessor ABC
├── registry.py                  # PreprocessorRegistry
├── preprocessors/               # Preprocessor implementations
│   ├── __init__.py              # Import triggers registration
│   └── wavefunction.py          # WavefunctionPreprocessor (reference)
└── utils/                       # Shared utilities
    ├── __init__.py
    ├── archive_handlers.py      # Archive extraction (tar.gz)
    ├── format_parsers.py        # Format parsing (molden-specific)
    └── npz_builders.py          # NPZ file construction
```

### 8.3 Creating a Custom Preprocessor

**IMPORTANT:** Preprocessing is highly dataset-specific. The logic for extracting, parsing, and transforming your data will depend entirely on your source data format. The following provides the structural framework, but **you must implement the data extraction and parsing logic specific to your format**.

#### 8.3.1 BasePreprocessor Contract (from `base_preprocessor.py`)

Your preprocessor must inherit from `BasePreprocessor` and implement two abstract methods:

```python
class BasePreprocessor(ABC):
    """
    Abstract base class for dataset preprocessors.

    Preprocessors handle one-time transformation of raw data files
    into the .npz format expected by miliaDataset.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """Initialize with configuration and logger."""
        self.config = config
        self.logger = logger
        self._validate_config()  # Called automatically

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate preprocessor-specific configuration.

        YOU MUST IMPLEMENT: Check that all required config keys exist
        and have valid values.

        Raises:
            ConfigurationError: If configuration is invalid
        """
        pass

    @abstractmethod
    def preprocess(self) -> Path:
        """
        Execute preprocessing logic.

        YOU MUST IMPLEMENT: The actual data transformation logic.

        Returns:
            Path to generated .npz file

        Raises:
            DataProcessingError: If preprocessing fails
        """
        pass

    def run(self) -> Path:
        """
        Execute full preprocessing pipeline with validation.

        DO NOT OVERRIDE: This method calls preprocess() and validates output.

        Returns:
            Path to validated .npz output file
        """
        # ... timing, logging, error handling ...
        output_path = self.preprocess()
        self._validate_output(output_path)
        return output_path

    def _validate_output(self, output_path: Path) -> None:
        """
        Validate generated .npz file structure.

        Checks for required keys: 'compounds', 'metadata'

        You may override to add dataset-specific validation.
        """
        # ... validation logic ...
```

#### 8.3.2 Preprocessor Skeleton Template

```python
"""
YourDataset Preprocessor
========================

Preprocessor for YourDataset (describe source format).

Author: Your Name
Version: 1.0
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, Any

from milia_pipeline.exceptions import ConfigurationError, DataProcessingError
from milia_pipeline.preprocessing.base_preprocessor import BasePreprocessor
from milia_pipeline.preprocessing.registry import PreprocessorRegistry

# Import utilities as needed
from milia_pipeline.preprocessing.utils.npz_builders import build_npz


@PreprocessorRegistry.register("YourDataset")  # Must match dataset metadata.name
class YourDatasetPreprocessor(BasePreprocessor):
    """
    Preprocessor for YourDataset.

    Pipeline:
    ---------
    1. [Describe step 1 - e.g., Extract files from archive]
    2. [Describe step 2 - e.g., Parse source format files]
    3. [Describe step 3 - e.g., Transform to NPZ-compatible arrays]
    4. [Describe step 4 - e.g., Build NPZ file]
    5. [Describe step 5 - e.g., Cleanup temporary files]

    Configuration:
    --------------
    Required keys:
        - raw_data_path: Path to source data (file or directory)
        - output_npz_path: Path for output .npz file

    Optional keys:
        - num_molecules: Limit number of molecules (None = all)
        - cleanup_temp: Remove temporary files after processing (default: True)
        - [Add your dataset-specific options]
    """

    def _validate_config(self) -> None:
        """Validate YourDataset-specific configuration."""
        # Check required keys
        required_keys = ['raw_data_path', 'output_npz_path']
        missing = [k for k in required_keys if k not in self.config]

        if missing:
            raise ConfigurationError(
                f"Missing required configuration keys: {missing}",
                config_key=', '.join(missing)
            )

        # Validate paths exist
        raw_path = Path(self.config['raw_data_path'])
        if not raw_path.exists():
            raise ConfigurationError(
                f"Raw data path not found: {raw_path}",
                config_key='raw_data_path',
                actual_value=str(raw_path)
            )

        # Add dataset-specific validation here
        # Example: validate feature_tier, num_molecules, etc.

        self.logger.debug("Configuration validation passed")

    def preprocess(self) -> Path:
        """
        Execute YourDataset preprocessing pipeline.

        Returns:
            Path to generated .npz file
        """
        raw_path = Path(self.config['raw_data_path'])
        output_npz = Path(self.config['output_npz_path'])
        num_molecules = self.config.get('num_molecules', None)
        cleanup_temp = self.config.get('cleanup_temp', True)

        # Check if output already exists (skip if so)
        if output_npz.exists():
            self.logger.info(f"Output already exists: {output_npz}")
            self.logger.info("Skipping preprocessing - delete file to regenerate")
            return output_npz

        temp_dir = None

        try:
            # ============================================================
            # STEP 1: Extract/Load Source Data
            # ============================================================
            # This is DATASET-SPECIFIC - implement based on your format
            # Examples:
            #   - Extract from tar.gz: use extract_from_targz()
            #   - Read directory of files: use pathlib glob
            #   - Download from URL: use requests
            # ============================================================
            self.logger.info("STEP 1: Loading source data")

            # YOUR IMPLEMENTATION HERE
            # Example for archive:
            # temp_dir = extract_from_targz(raw_path, max_files=num_molecules)

            # ============================================================
            # STEP 2: Parse Source Format
            # ============================================================
            # This is DATASET-SPECIFIC - implement based on your format
            # You may need to create a custom parser in utils/
            # ============================================================
            self.logger.info("STEP 2: Parsing source format")

            # YOUR IMPLEMENTATION HERE
            # Example:
            # features, parse_metadata = parse_your_format(
            #     source_dir=temp_dir,
            #     options=self.config.get('parsing_options', {})
            # )

            # Placeholder - replace with actual parsing
            features = {
                'compounds': [],  # List of molecule identifiers
                'atoms': [],      # List of atomic number arrays
                'coordinates': [],  # List of coordinate arrays
                # Add your dataset-specific features
            }
            parse_metadata = {}

            # ============================================================
            # STEP 3: Build NPZ File
            # ============================================================
            self.logger.info("STEP 3: Building NPZ file")

            # Prepare metadata
            npz_metadata = {
                'version': '1.0',
                'dataset_name': 'YourDataset',
                'source': raw_path.name,
                'preprocessing_version': '1.0',
                **parse_metadata
            }

            # Use the standard NPZ builder
            build_npz(
                features=features,
                metadata=npz_metadata,
                output_path=output_npz,
                logger=self.logger
            )

            # ============================================================
            # STEP 4: Cleanup
            # ============================================================
            if cleanup_temp and temp_dir and temp_dir.exists():
                self.logger.info("STEP 4: Cleaning up temporary files")
                shutil.rmtree(temp_dir)
                self.logger.info(f"Removed: {temp_dir}")

            self.logger.info("PREPROCESSING COMPLETE")
            return output_npz

        except Exception as e:
            # Cleanup on error
            if cleanup_temp and temp_dir and Path(temp_dir).exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass

            raise DataProcessingError(
                f"YourDataset preprocessing failed: {e}",
                operation="your_dataset_preprocessing"
            ) from e
```

#### 8.3.3 Key Implementation Points

1. **Registration Decorator**: The `@PreprocessorRegistry.register("YourDataset")` decorator automatically registers your preprocessor when the module is imported. The string argument should match your dataset's `metadata.name`. **No additional registration steps are required** - the `milia_pipeline/preprocessing/preprocessors/__init__.py` uses dynamic auto-discovery to import all preprocessor modules at package load time, triggering the decorator automatically.

2. **Configuration Validation**: The `_validate_config()` method is called automatically during `__init__`. Validate all required keys and their types/values here.

3. **Preprocessing Logic**: The `preprocess()` method contains your actual transformation logic. This is entirely dataset-specific.

4. **Output Requirements**: The generated NPZ file must contain at minimum:
   - `'compounds'`: Array of molecule identifiers
   - `'atoms'`: Array of atomic number arrays (object dtype)
   - `'coordinates'`: Array of coordinate arrays (object dtype)
   - `'metadata'`: Array containing metadata dictionary

5. **Error Handling**: Use `ConfigurationError` for config issues and `DataProcessingError` for processing failures.

### 8.4 Available Utility Functions

The preprocessing module provides reusable utilities. **Note:** Format parsers are dataset-specific; only `npz_builders` and `archive_handlers` are general-purpose.

#### 8.4.1 Archive Handlers (`archive_handlers.py`)

```python
from milia_pipeline.preprocessing.utils.archive_handlers import extract_from_targz

def extract_from_targz(
    tar_path: Path,
    max_files: Optional[int] = None,
    file_extension: str = '.molden',
    temp_dir: Optional[Path] = None
) -> Path:
    """
    Extract files from tar.gz archive using streaming (memory-efficient).

    Args:
        tar_path: Path to .tar.gz archive
        max_files: Maximum number of files to extract (None = all)
        file_extension: Only extract files with this extension
        temp_dir: Directory for extraction (None = system temp)

    Returns:
        Path to extraction directory containing extracted files
    """
```

**Usage Example:**
```python
# Extract up to 100 .xyz files from archive
extracted_dir = extract_from_targz(
    tar_path=Path("data/molecules.tar.gz"),
    max_files=100,
    file_extension='.xyz'
)
xyz_files = list(extracted_dir.rglob("*.xyz"))
```

#### 8.4.2 NPZ Builders (`npz_builders.py`)

```python
from milia_pipeline.preprocessing.utils.npz_builders import build_npz, validate_npz_structure

def build_npz(
    features: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    output_path: Path,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Build compressed .npz file from features and metadata.

    Args:
        features: Dictionary mapping feature names to numpy arrays
                  REQUIRED keys: 'compounds', 'atoms', 'coordinates'
        metadata: Dictionary with metadata about the dataset
        output_path: Path where .npz file will be created
        logger: Logger instance
    """

def validate_npz_structure(
    npz_path: Path,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Validate .npz file structure and return summary.

    Args:
        npz_path: Path to .npz file to validate

    Returns:
        Dictionary with validation results and file summary
    """
```

**Usage Example:**
```python
import numpy as np
from milia_pipeline.preprocessing.utils.npz_builders import build_npz

# Prepare features (must be numpy arrays)
features = {
    'compounds': np.array(['mol_001', 'mol_002', 'mol_003'], dtype=object),
    'atoms': np.array([
        np.array([6, 1, 1, 1, 1]),  # CH4
        np.array([8, 1, 1]),        # H2O
        np.array([7, 1, 1, 1]),     # NH3
    ], dtype=object),
    'coordinates': np.array([
        np.array([[0, 0, 0], [1, 0, 0], ...]),  # CH4 coords
        np.array([[0, 0, 0], [0.96, 0, 0], ...]),  # H2O coords
        np.array([[0, 0, 0], [1.01, 0, 0], ...]),  # NH3 coords
    ], dtype=object),
    'energy': np.array([-40.5, -76.4, -56.2], dtype=np.float64),
}

metadata = {
    'version': '1.0',
    'dataset_name': 'MyDataset',
    'num_molecules': 3,
}

build_npz(features, metadata, Path('output/dataset.npz'))
```

> ⚠️ **WARNING: Object Array Storage Requires Handler Normalization**
>
> When storing ragged arrays (different sizes per molecule) with `dtype=object` as shown above,
> your handler's `process_property_value()` method MUST convert to native dtypes before core
> pipeline validation. Without this, `np.isfinite()` and `torch.tensor()` will fail.
>
> See `ANI1xDatasetHandler.process_property_value()` in `handlers/implementations/ani1x.py` for the reference
> implementation, and Section 7.2 for detailed guidance.

#### 8.4.3 Format Parsers - DATASET SPECIFIC

**IMPORTANT:** The `format_parsers.py` module contains **molden-specific** parsing logic. It is NOT a general-purpose utility. If your dataset uses a different format, you must create your own parser.

**Existing parser (for reference only):**
```python
# This is MOLDEN-SPECIFIC - do not copy for other formats
from milia_pipeline.preprocessing.utils.format_parsers import parse_molden_files

def parse_molden_files(
    molden_dir: Path,
    feature_tier: str = 'standard',  # 'basic', 'standard', 'complete'
    logger: Optional[logging.Logger] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Parse .molden files and extract quantum chemistry features."""
```

**For other formats, you need to create your own parser:**
```python
# milia_pipeline/preprocessing/utils/your_format_parser.py

def parse_your_format_files(
    source_dir: Path,
    options: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Parse your format files and extract features.

    YOU MUST IMPLEMENT THIS BASED ON YOUR FORMAT.

    Returns:
        Tuple of (features_dict, metadata_dict)
    """
    # YOUR FORMAT-SPECIFIC PARSING LOGIC
    pass
```

### 8.5 Common Pitfalls When Adding New Datasets

Based on real implementation experience (including QM9), these are the most common mistakes to avoid:

#### Pitfall 1: Wrong `identifier_keys` Order

**Symptom:**
```
HandlerValidationError: Atom count mismatch between SMILES and QM coordinates
SMILES: 1, QM coords: 5
```

**Cause:** SMILES specified before InChI in `identifier_keys` tuple

**Why It Fails:** SMILES like `C` (methane) use implicit hydrogens (1 atom), but QM coordinates include all atoms including hydrogens (5 atoms for CH4). RDKit's `AddHs()` fails on unsanitized molecules.

**Prevention:** Always specify InChI first when both are available:
```python
# ✅ CORRECT - InChI first
schema = DatasetSchema(
    identifier_keys=(('inchi', 'inchi'), ('smiles', 'smiles')),
    ...
)

# ❌ WRONG - causes 99% failure rate
schema = DatasetSchema(
    identifier_keys=(('smiles', 'smiles'), ('inchi', 'inchi')),
    ...
)
```

#### Pitfall 2: Editing Handler Instead of Dataset Class

**Symptom:** Changes to identifier order don't take effect

**Cause:** Modified `YourDatasetHandler.get_identifier_keys()` in handler class instead of `YourDataset.schema.identifier_keys` in `datasets/implementations/your_dataset.py`

**Prevention:** Remember the rule:
- **Schema definition** (including `identifier_keys` order) → `datasets/implementations/your_dataset.py`
- **Runtime processing logic** → `handlers/implementations/your_dataset.py`

The handler's `get_identifier_keys()` method reads FROM the Dataset class schema.

#### Pitfall 3: Missing Registry Registration

**Symptom:**
```
HandlerNotAvailableError: Handler type 'YourDataset' is not supported
```

**Cause:**
- Missing `@register` decorator on Dataset class
- Missing import in `datasets/implementations/__init__.py`

**Prevention:** Always verify registration:
```python
from milia_pipeline.datasets.registry import is_registered, list_all

# Should return True
print(is_registered('YourDataset'))

# Should include 'YourDataset'
print(list_all())
```

#### Pitfall 4: Mismatched `config_key`

**Symptom:** Configuration not loaded, using defaults

**Cause:** `config_key` in Dataset class doesn't match section name in config file

**Prevention:** Ensure exact match:
```python
# In your_dataset.py
config_key = "your_dataset_config"

# In config file - MUST match exactly
# Split-file mode: configs/datasets/your_dataset.yaml
# Single-file mode: config.yaml
your_dataset_config:
  raw_npz_filename: your_dataset.npz
  ...
```

#### Pitfall 5: Wrong NPZ Key Names in `identifier_keys`

**Symptom:** Identifiers not found, falling back to index-based

**Cause:** NPZ key name in `identifier_keys` doesn't match actual key in NPZ file

**Prevention:** Verify NPZ keys first:
```python
import numpy as np
data = np.load('your_dataset.npz', allow_pickle=True)
print(list(data.keys()))  # Check exact key names

# Then use exact names in schema
identifier_keys=(('actual_inchi_key', 'inchi'), ('actual_smiles_key', 'smiles'))
```

#### Pitfall 6: Object Array Dtype Issues from Preprocessor

**Symptom:**
- `TypeError: ufunc 'isfinite' not supported for the input types`
- `can't convert np.ndarray of type numpy.object_`

**Cause:** Preprocessors storing ragged arrays (different sizes per molecule) with `dtype=object` causes downstream validation and tensor conversion failures.

**Why It Fails:**
1. `np.isfinite()` in `validators.py` cannot operate on object arrays
2. `torch.tensor()` in handler's `_ensure_tensor()` cannot convert object arrays

**Solution:** Normalize ALL properties to native dtypes in your handler's `process_property_value()` method before they reach the core pipeline.

**Reference Implementation:** See `ANI1xDatasetHandler.process_property_value()` in `handlers/implementations/ani1x.py` for the complete dtype normalization pattern.

**Key Conversions:**
- Integer arrays (atoms, atomic_numbers): `dtype=object` → `np.int64`
- Float arrays (coordinates, forces, dipole, charges): `dtype=object` → `np.float32` or `np.float64`

**Evidence:** ANI-1x preprocessor stores all ragged arrays as `dtype=object` (ani1x.py lines 423, 431, 434, 437, 440). Without dtype normalization in the handler, 100% of molecules failed processing.

#### Pitfall 7: Hardcoded Neutral Charge for Datasets with Charged Molecules

**Symptom:**
- `ValueError: Final molecular charge (0) does not match input (-1); could not find valid bond ordering`
- `ValueError: Valence of atom X is Y, which is larger than the allowed maximum`

**Cause:** Handler's `get_molecular_charge()` returns hardcoded `0` (neutral), but dataset contains charged molecules (ions, protonated/deprotonated species).

**Why It Fails:** The `rdDetermineBonds.DetermineBonds(mol, charge=molecular_charge)` algorithm uses the charge parameter to correctly determine bond orders. A sulfate ion (SO₄²⁻) or protonated amine (R-NH₃⁺) will fail bond order determination if charge=0 is passed.

**Solution:**
1. **Preprocessing:** Store molecular charge in NPZ during preprocessing (inferred from structure or source metadata)
2. **Handler:** Read charge from NPZ in `get_molecular_charge()` instead of returning constant 0

**Reference Implementation:** See `QDPiDatasetHandler.get_molecular_charge()` in `handlers/implementations/qdpi.py`:
```python
def get_molecular_charge(self, raw_properties_dict, atomic_numbers, mol_identifier) -> int:
    # Read charge from NPZ (stored during preprocessing)
    if 'molecular_charge' in raw_properties_dict:
        return int(raw_properties_dict['molecular_charge'])
    return 0  # Default to neutral if not specified
```

**Key Takeaway:** If your dataset contains both neutral AND charged molecules, you MUST:
1. Track charge during preprocessing (from source metadata or directory structure)
2. Store charge as `molecular_charge` property in NPZ
3. Read charge dynamically in handler's `get_molecular_charge()`

#### Pitfall 8: Non-Standard HDF5 Format Key Names

**Symptom:**
- `Skipping {formula}: no 'elements' key found`
- 0 conformers extracted from HDF5 file

**Cause:** Assuming HDF5 uses intuitive key names like `'elements'`, `'coordinates'`, `'energies'` when actual format uses different conventions.

**Why It Fails:** Different HDF5 conventions exist:
- Standard HDF5: `elements`, `coordinates`, `energies`
- DeePMD-kit format: `type_map.raw`, `type.raw`, `set.XXX/coord.npy`, `set.XXX/energy.npy`
- ASE format: `atomic_numbers`, `positions`, `energy`

**Solution:**
1. **Always verify** HDF5 structure BEFORE writing preprocessor
2. **Use official documentation** for the dataset's declared format
3. **Test with h5dump or h5py** to inspect actual key names

**Reference:** QDπ uses DeePMD-kit HDF5 format (NOT standard HDF5):
```python
# WRONG - Intuitive but non-existent keys
elements = group['elements'][:]
coords = group['coordinates'][:]

# CORRECT - DeePMD-kit format
type_map = group['type_map.raw'][:]  # Element symbols as byte strings
type_indices = group['type.raw'][:]   # Atom type indices
coords = group['set.000/coord.npy'][:]  # Flattened coordinates
```

**Key Takeaway:** Never assume HDF5 key names. Always verify format documentation and inspect file structure before implementation.

### 8.6 Preprocessor Configuration

Add preprocessing configuration to your dataset config file.

**Option A: Split-file mode (Recommended)**

Edit or create: `configs/datasets/your_dataset.yaml`

```yaml
# ══════════════════════════════════════════════════════════════════════════════
# YourDataset Dataset Configuration (with Preprocessing + Colocated Property Availability)
# ══════════════════════════════════════════════════════════════════════════════
# Part of MILIA YAML Splitting Architecture

# ─────────────────────────────────────────────────────────────────────────────
# Dataset Configuration with Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
your_dataset_config:
  # Standard dataset config
  raw_npz_filename: your_dataset.npz  # Output from preprocessing
  raw_data_download_url: null  # Optional URL for raw data

  # Preprocessing configuration
  preprocessing:
    # Required: Path to source data
    raw_data_path: raw/your_source_data.tar.gz

    # Required: Output NPZ path
    output_npz_path: processed/your_dataset.npz

    # Optional: Limit number of molecules
    num_molecules: null  # null = process all

    # Optional: Cleanup temporary files
    cleanup_temp: true

    # Dataset-specific options
    # Add your own options here based on your preprocessor's needs
    feature_tier: standard  # Example from WavefunctionPreprocessor

# ─────────────────────────────────────────────────────────────────────────────
# Colocated Property Availability
# ─────────────────────────────────────────────────────────────────────────────
property_availability:
  YourDataset:
    molecular_identifiers:
      - compounds
    atomic_structure:
      - atoms
      - coordinates
    scalar_graph_targets:
      - energy
    node_features: []
    vector_graph_properties: []
    variable_len_graph_properties: []
    metadata_fields:
      - _metadata
    uncertainty_fields: []
```

**Option B: Single-file mode (Backward Compatible)**

Edit: `config.yaml`

```yaml
# YourDataset Configuration
your_dataset_config:
  # Standard dataset config
  raw_npz_filename: your_dataset.npz  # Output from preprocessing
  raw_data_download_url: null  # Optional URL for raw data

  # Preprocessing configuration
  preprocessing:
    # Required: Path to source data
    raw_data_path: raw/your_source_data.tar.gz

    # Required: Output NPZ path
    output_npz_path: processed/your_dataset.npz

    # Optional: Limit number of molecules
    num_molecules: null  # null = process all

    # Optional: Cleanup temporary files
    cleanup_temp: true

    # Dataset-specific options
    # Add your own options here based on your preprocessor's needs
    feature_tier: standard  # Example from WavefunctionPreprocessor
```

### 8.7 Integrating Preprocessor with Dataset

The preprocessing step is **separate from** and **runs before** the dataset class is used. Here's the typical workflow:

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLETE WORKFLOW                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. PREPROCESSING (One-time, offline)                           │
│     ─────────────────────────────────────                       │
│     Input: Raw source data (tar.gz, files, etc.)                │
│     Tool: Your custom preprocessor                              │
│     Output: .npz file                                           │
│                                                                  │
│     >>> from milia_pipeline.preprocessing import PreprocessorRegistry
│     >>> PreprocessorClass = PreprocessorRegistry.get_preprocessor("YourDataset")
│     >>> config = {...}  # From config.yaml preprocessing section
│     >>> preprocessor = PreprocessorClass(config, logger)
│     >>> npz_path = preprocessor.run()                           │
│                                                                  │
│  2. DATASET USAGE (Runtime, repeated)                           │
│     ────────────────────────────────                            │
│     Input: .npz file from step 1                                │
│     Tool: Your dataset class + miliaDataset                     │
│     Output: PyTorch Geometric dataset                           │
│                                                                  │
│     >>> from milia_pipeline.datasets import miliaDataset        │
│     >>> dataset = miliaDataset(root='./data', ...)              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Points:**
1. Preprocessing converts raw data → NPZ (one-time operation)
2. Dataset class defines how NPZ data is interpreted (metadata, schema, features)
3. miliaDataset loads NPZ and creates PyG Data objects (runtime operation)
4. The `config_key` in your dataset class points to the config section with `raw_npz_filename`

---

## 9. Troubleshooting Guide

### Error: "Dataset class 'X' missing required class attributes"

**Cause:** One or more of `metadata`, `schema`, `features`, or `config_key` is missing.

**Solution:**
```python
@register
class YourDataset(BaseDataset):
    metadata = DatasetMetadata(...)    # Required
    schema = DatasetSchema(...)        # Required
    features = DatasetFeatures(...)    # Required
    config_key = "your_config"         # Required
```

### Error: "'X.metadata' must be a DatasetMetadata instance"

**Cause:** `metadata` is not a `DatasetMetadata` instance (possibly a dict or string).

**Solution:**
```python
# Wrong:
metadata = {"name": "X", "version": "1.0.0", ...}

# Correct:
metadata = DatasetMetadata(
    name="X",
    version="1.0.0",
    description="..."
)
```

### Error: "DatasetMetadata.name must be a non-empty string"

**Cause:** Empty or None value for required DatasetMetadata field.

**Solution:**
```python
# Wrong:
metadata = DatasetMetadata(name="", version="1.0.0", description="...")

# Correct:
metadata = DatasetMetadata(name="ValidName", version="1.0.0", description="...")
```

### Error: "required_properties must be a tuple"

**Cause:** Using a list instead of tuple for `required_properties`.

**Solution:**
```python
# Wrong:
schema = DatasetSchema(required_properties=['atoms', 'coordinates'])

# Correct (use tuple):
schema = DatasetSchema(required_properties=('atoms', 'coordinates'))
```

### Error: "required_properties cannot be empty"

**Cause:** Empty tuple for `required_properties`.

**Solution:**
```python
# Wrong:
schema = DatasetSchema(required_properties=())

# Correct:
schema = DatasetSchema(required_properties=('atoms', 'coordinates'))
```

### Error: "coordinate_units must be one of ('angstrom', 'bohr')"

**Cause:** Invalid coordinate unit string.

**Solution:**
```python
# Wrong:
schema = DatasetSchema(..., coordinate_units='Angstrom')  # Wrong case
schema = DatasetSchema(..., coordinate_units='pm')        # Invalid unit

# Correct:
schema = DatasetSchema(..., coordinate_units='angstrom')
schema = DatasetSchema(..., coordinate_units='bohr')
```

### Error: "Dataset 'X' already registered"

**Cause:** Another dataset class has the same `metadata.name`.

**Solution:**
- Ensure `metadata.name` is unique across all registered datasets
- Check for duplicate imports or class definitions

### Error: "Cannot register abstract class 'X'"

**Cause:** Trying to register a class that still has unimplemented abstract methods.

**Solution:**
```python
# Ensure all abstract methods are implemented:
@classmethod
def get_required_properties(cls) -> List[str]:
    return list(cls.schema.required_properties)

@classmethod
def get_feature_support(cls) -> Dict[str, bool]:
    return cls.features.to_dict()

@classmethod
def get_molecule_creation_strategy(cls) -> str:
    return 'identifier_coordinate_based'  # or 'coordinate_based'
```

### Dataset Not Found When Processing

**Cause:** Dataset registered but `dataset_type` in config file doesn't match.

**Solution:**
```yaml
# In config file (configs/main.yaml or config.yaml)
dataset_type: "YourDataset"  # Must exactly match metadata.name
```

### Error: "Missing essential properties: ['Etot', 'atoms']" for Non-DFT Datasets

**Cause:** Registry lookup failed and fallback to `HANDLER_REQUIRED_PROPERTIES` dict returned wrong property names. This occurs when:
1. Your dataset uses a different energy key (e.g., `'energy'` instead of `'Etot'`)
2. The fallback dict in `config_constants.py` doesn't have an entry for your dataset
3. The legacy `_legacy_validation_fallback` function expects `'Etot'`

**Solution:**
1. Ensure your handler's `validate_molecule_data()` succeeds (check `_is_valid_property`)
2. Add fallback entries to `config_constants.py` (see [Section 7.1](#71-optional-config_constantspy-fallback-entries))
3. Verify your `HANDLER_REQUIRED_PROPERTIES` entry uses correct property key names:

```python
# In config_constants.py
HANDLER_REQUIRED_PROPERTIES: Dict[str, List[str]] = {
    ...
    'YourDataset': ['your_energy_key', 'atoms', 'coordinates']  # Not 'Etot'!
}
```

**ANI-1x Example:**
```python
'ANI1x': ['energy', 'atoms', 'coordinates']  # ANI-1x uses 'energy', not 'Etot'
```

### Error: "InChI: N/A (inchi missing for logging)" for coordinate_based Datasets

**Cause:** The logging system expects an InChI identifier, but `coordinate_based` datasets (like Wavefunction, ANI-1x) don't have parseable identifiers.

**This is NOT an error** - it's expected behavior for `coordinate_based` datasets. The processing should still work correctly. Verify:
1. `get_molecule_creation_strategy()` returns `'coordinate_based'`
2. `identifier_keys` in your schema is empty: `identifier_keys=()`
3. `HANDLER_IDENTIFIER_KEYS['YourDataset']` is `[]` (empty list)

### Preprocessing Errors

#### Error: "No preprocessor registered for dataset type 'X'"

**Cause:** Preprocessor not registered (decorator missing or malformed) or file not discovered.

**Solution:**
```python
# 1. Ensure preprocessor file exists in the correct location:
#    milia_pipeline/preprocessing/preprocessors/your_dataset.py
#    (Dynamic discovery scans this directory automatically)

# 2. Ensure @PreprocessorRegistry.register decorator is correct:
@PreprocessorRegistry.register("YourDataset")  # Must match metadata.name
class YourDatasetPreprocessor(BasePreprocessor):
    ...

# 3. Ensure file is not excluded from discovery:
#    - File must be a .py file (not .pyc or other)
#    - Filename must NOT start with underscore
#    - Filename must NOT be in excluded list: __init__, base, registry, utils, common
```

#### Error: "Preprocessor X must inherit from BasePreprocessor"

**Cause:** Custom preprocessor doesn't inherit from the correct base class.

**Solution:**
```python
# Wrong:
class YourPreprocessor:  # Missing inheritance
    ...

# Correct:
from milia_pipeline.preprocessing.base_preprocessor import BasePreprocessor

class YourPreprocessor(BasePreprocessor):
    ...
```

#### Error: "Missing required keys in .npz: ['compounds', 'metadata']"

**Cause:** Output NPZ file missing required keys.

**Solution:**
```python
# Ensure your preprocessor outputs these required keys:
features = {
    'compounds': np.array([...], dtype=object),  # REQUIRED
    'atoms': np.array([...], dtype=object),      # REQUIRED
    'coordinates': np.array([...], dtype=object), # REQUIRED
    # Your additional features...
}

metadata = {'version': '1.0', ...}  # REQUIRED

build_npz(features, metadata, output_path)  # Adds 'metadata' key
```

#### Error: "ConfigurationError: Missing required configuration keys"

**Cause:** Preprocessing config in config file missing required keys.

**Solution:**
```yaml
# In config file (configs/datasets/your_dataset.yaml or config.yaml)
your_dataset_config:
  preprocessing:
    raw_data_path: raw/your_data.tar.gz    # REQUIRED
    output_npz_path: processed/output.npz   # REQUIRED
    # Add other required keys for your preprocessor
```

---

## 10. Verification Checklist

Use this checklist to verify your implementation is complete:

### File Creation Checklist

**For Split-file mode (YAML Splitting Architecture v1.1.0 - Full Colocation - Recommended):**
- [ ] Created `milia_pipeline/datasets/implementations/your_dataset.py` (auto-discovered)
- [ ] Created `configs/datasets/your_dataset.yaml` with FULLY COLOCATED sections:
  - [ ] `your_dataset_config` section (dataset configuration)
  - [ ] `data_config.property_selection.YourDataset` section (PyG Data properties)
  - [ ] `property_availability.YourDataset` section (available properties)
- [ ] Updated `configs/main.yaml`: Added `YourDataset` to `dataset_type` Options list comment

**For Single-file mode (Backward Compatible):**
- [ ] Created `milia_pipeline/datasets/implementations/your_dataset.py` (auto-discovered)
- [ ] Added `YourDataset` to `dataset_type` Options list comment (line ~14 in `config.yaml`)
- [ ] Added `your_dataset_config` section to `config.yaml`
- [ ] Added `YourDataset` entry to `property_availability` section in `config.yaml`
- [ ] Added `YourDataset` entry to `data_config.property_selection` section in `config.yaml`

### Class Definition Checklist

- [ ] Class decorated with `@register`
- [ ] Class extends `BaseDataset`
- [ ] `metadata` is `DatasetMetadata` instance with:
  - [ ] Non-empty `name` (unique)
  - [ ] Non-empty `version`
  - [ ] Non-empty `description`
- [ ] `schema` is `DatasetSchema` instance with:
  - [ ] Non-empty `required_properties` tuple
  - [ ] Valid `coordinate_units` ('angstrom' or 'bohr')
  - [ ] Valid `energy_units`
  - [ ] **`identifier_keys` has InChI FIRST when both InChI and SMILES are available**
- [ ] `features` is `DatasetFeatures` instance
- [ ] `config_key` is non-empty string matching config file section (in `configs/datasets/your_dataset.yaml` or `config.yaml`)

### Abstract Method Checklist

- [ ] `get_required_properties()` implemented and returns `List[str]`
- [ ] `get_feature_support()` implemented and returns `Dict[str, bool]`
- [ ] `get_molecule_creation_strategy()` implemented and returns valid strategy

### Registration Verification Checklist

- [ ] Dataset appears in `list_all()` output
- [ ] `is_registered('YourDataset')` returns `True`
- [ ] `get('YourDataset')` returns your dataset class
- [ ] No errors on import

### Configuration Verification Checklist

- [ ] Config section key matches `config_key` attribute
- [ ] `raw_npz_filename` specified
- [ ] `dataset_type` can be set to your dataset name
- [ ] Added entry to `property_availability` section documenting all NPZ properties
  - **Split-file mode:** FULLY COLOCATED in `configs/datasets/your_dataset.yaml`
  - **Single-file mode:** In `config.yaml` under `property_availability:`
- [ ] Added entry to `data_config.property_selection` section specifying PyG Data properties
  - **Split-file mode:** FULLY COLOCATED in `configs/datasets/your_dataset.yaml`
  - **Single-file mode:** In `config.yaml` under `data_config:`

### Identifier Keys Verification Checklist (CRITICAL)

- [ ] **`identifier_keys` has InChI FIRST** when both InChI and SMILES are available
- [ ] Verified identifier order by checking `YourDataset.schema.identifier_keys` tuple order
- [ ] NPZ key names in `identifier_keys` match actual keys in NPZ file
- [ ] Tested with actual NPZ data to ensure identifiers are found
- [ ] Confirmed **NO "Atom count mismatch" errors** in processing log
- [ ] Verified 100% (or expected) success rate in test run

### Preprocessing Checklist (If Applicable)

- [ ] Created `milia_pipeline/preprocessing/preprocessors/your_dataset.py` (auto-discovered)
- [ ] Preprocessor decorated with `@PreprocessorRegistry.register("YourDataset")`
- [ ] Preprocessor extends `BasePreprocessor`
- [ ] `_validate_config()` implemented
- [ ] `preprocess()` implemented and returns `Path` to NPZ file
- [ ] NPZ output contains required keys: `compounds`, `atoms`, `coordinates`, `metadata`
- [ ] Preprocessing config added to dataset config file:
  - **Split-file mode:** In `configs/datasets/your_dataset.yaml` under `your_dataset_config.preprocessing`
  - **Single-file mode:** In `config.yaml` under `your_dataset_config.preprocessing`
- [ ] `PreprocessorRegistry.supports_preprocessing("YourDataset")` returns `True`

### Optional: config_constants.py Fallback Entries Checklist

For production robustness, consider adding fallback entries (see [Section 7.1](#71-optional-config_constantspy-fallback-entries)):

- [ ] Added to `SUPPORTED_HANDLER_TYPES` list
- [ ] Added to `REQUIRED_HANDLER_CONFIG_KEYS` dict
- [ ] Added to `HANDLER_FEATURE_SUPPORT` dict (matching `DatasetFeatures`)
- [ ] Added to `HANDLER_REQUIRED_PROPERTIES` dict (matching `DatasetSchema.required_properties`)
- [ ] Added to `HANDLER_OPTIONAL_PROPERTIES` dict (matching `DatasetSchema.optional_properties`)
- [ ] Added to `HANDLER_IDENTIFIER_KEYS` dict (matching `DatasetSchema.identifier_keys`)
- [ ] Added to `HANDLER_COORDINATE_UNITS` dict (matching `DatasetSchema.coordinate_units`)
- [ ] All values exactly match your `DatasetSchema` and `DatasetFeatures` definitions

---

## Appendix: Reference Implementation Analysis

### A.1 DFTDataset Analysis (from `dft.py`)

```python
# Key characteristics:
# - Uses identifier_coordinate_based strategy
# - InChI/SMILES identifiers are parseable
# - Coordinates in Angstrom
# - Energies in Hartree
# - Supports vibrational analysis and thermodynamics

metadata = DatasetMetadata(
    name="DFT",
    version="1.0.0",
    description="DFT quantum chemistry dataset...",
    author="MILIA Pipeline Team",
)

schema = DatasetSchema(
    required_properties=('Etot', 'atoms', 'coordinates'),
    optional_properties=('freqs', 'vibmodes', 'rots', 'dipoles'),
    identifier_keys=(('inchi', 'inchi'), ('graphs', 'smiles')),
    coordinate_units='angstrom',
    energy_units='hartree',
)

features = DatasetFeatures(
    vibrational_analysis=True,
    uncertainty_handling=False,
    atomization_energy=True,
    rotational_constants=True,
    frequency_analysis=True,
    orbital_analysis=False,
    homo_lumo_gap=False,
    mo_energies=False,
)
```

### A.2 DMCDataset Analysis (from `dmc.py`)

```python
# Key characteristics:
# - Uses identifier_coordinate_based strategy
# - Has uncertainty handling (std field)
# - No vibrational analysis
# - Monte Carlo specific properties

schema = DatasetSchema(
    required_properties=('Etot', 'std', 'atoms', 'coordinates'),  # Note: std required
    optional_properties=('qmc_stats', 'correlation_data'),
    ...
)

features = DatasetFeatures(
    vibrational_analysis=False,
    uncertainty_handling=True,  # Key difference from DFT
    atomization_energy=False,
    ...
)
```

### A.3 WavefunctionDataset Analysis (from `wavefunction.py`)

```python
# Key characteristics:
# - Uses coordinate_based strategy (compound IDs not parseable)
# - Coordinates in BOHR (requires conversion)
# - Orbital analysis features
# - No total energy as required property

schema = DatasetSchema(
    required_properties=('atoms', 'coordinates', 'compounds'),  # No Etot
    optional_properties=('mo_energies', 'mo_occupations', 'homo_lumo_gap_eV',
                        'total_energy', 'n_electrons'),
    identifier_keys=(('compounds', 'compound_id'),),  # Label only
    coordinate_units='bohr',  # Different from DFT/DMC
    energy_units='eV',
)

features = DatasetFeatures(
    vibrational_analysis=False,
    uncertainty_handling=False,
    atomization_energy=False,
    rotational_constants=False,
    frequency_analysis=False,
    orbital_analysis=True,    # Wavefunction-specific
    homo_lumo_gap=True,       # Wavefunction-specific
    mo_energies=True,         # Wavefunction-specific
)

# Critical: coordinate_based strategy
@classmethod
def get_molecule_creation_strategy(cls) -> str:
    return 'coordinate_based'
```

### A.4 QM9Dataset Analysis (from `qm9.py`) - LESSONS LEARNED

```python
# Key characteristics:
# - Uses identifier_coordinate_based strategy
# - ⚠️ CRITICAL: SMILES MUST NOT be first (learned from implementation)
# - 133,885 stable small organic molecules (CHONF, up to 9 heavy atoms)
# - B3LYP/6-31G(2df,p) level of DFT
# - Coordinates in Angstrom, energies in Hartree

metadata = DatasetMetadata(
    name="QM9",
    version="1.0.0",
    description="QM9 quantum chemistry dataset with 133,885 molecules",
    author="Ramakrishnan, Dral, Rupp, von Lilienfeld",
    license="CC0",
)

schema = DatasetSchema(
    required_properties=('U0', 'atoms', 'coordinates'),
    optional_properties=(
        'A', 'B', 'C',              # Rotational constants (GHz)
        'mu',                        # Dipole moment (Debye)
        'alpha',                     # Isotropic polarizability
        'homo', 'lumo', 'gap',      # Orbital energies (Hartree)
        'r2',                        # Electronic spatial extent
        'zpve',                      # Zero point vibrational energy
        'U', 'H', 'G',              # Thermodynamic energies
        'Cv',                        # Heat capacity
        'Qmulliken',                # Mulliken partial charges
        'freqs',                     # Vibrational frequencies
    ),
    # ⚠️ CRITICAL LESSON: SMILES MUST NOT be the first
    # Original implementation had SMILES first → Almost fails
    # After fix with InChI first → 100% success rate
    identifier_keys=(('inchi', 'inchi'), ('smiles', 'smiles')),
    coordinate_units='angstrom',
    energy_units='hartree',
)

features = DatasetFeatures(
    vibrational_analysis=True,
    uncertainty_handling=False,    # DFT is deterministic
    atomization_energy=True,
    rotational_constants=True,
    frequency_analysis=True,
    orbital_analysis=False,
    homo_lumo_gap=True,            # QM9 has gap property
    mo_energies=False,
)

# Uses identifier_coordinate_based strategy (like DFT)
@classmethod
def get_molecule_creation_strategy(cls) -> str:
    return 'identifier_coordinate_based'
```

**QM9 Implementation Lesson:**

| Configuration | identifier_keys Order | Success Rate | Error |
|---------------|----------------------|--------------|-------|
| ❌ Initial | `(('smiles', 'smiles'), ('inchi', 'inchi'))` | 1% | "Atom count mismatch: SMILES has 1 atoms but QM calculation has 5 atoms" |
| ✅ Fixed | `(('inchi', 'inchi'), ('smiles', 'smiles'))` | 100% | None |

**Root Cause:** SMILES like `C` (methane) have implicit hydrogens (1 atom), while QM coordinates have explicit hydrogens (5 atoms for CH4). InChI explicitly encodes all hydrogens (`InChI=1S/CH4/h1H4`).

### A.5 ANI1xDataset Analysis (from `ani1x.py`) - LESSONS LEARNED

**Key Characteristics:**
- Uses `coordinate_based` strategy (no parseable identifiers)
- Preprocessor stores ragged arrays as `dtype=object`
- ~5 million conformations from ~57k molecules
- ωB97x/6-31G(d) level of DFT
- Coordinates in Angstrom, energies in Hartree

**Source Files:**
- Dataset class: `datasets/implementations/ani1x.py`
- Handler class: `handlers/implementations/ani1x.py` (ANI1xDatasetHandler)
- Preprocessor: `preprocessing/preprocessors/ani1x.py`

**Schema Summary:**
- Required properties: `('energy', 'atoms', 'coordinates')`
- Optional properties: `('forces', 'hirshfeld_charges', 'cm5_charges', 'dipole')`
- Identifier keys: `()` (empty - coordinate_based strategy)
- Coordinate units: `'angstrom'`
- Energy units: `'hartree'`

**Critical Implementation Lesson - Dtype Normalization:**

| Issue | Symptom | Root Cause | Solution Location |
|-------|---------|------------|-------------------|
| Validation failure | `TypeError: ufunc 'isfinite' not supported` | atoms array `dtype=object` | `process_property_value()` converts to `int64` |
| Tensor conversion failure | `can't convert np.ndarray of type numpy.object_` | float arrays `dtype=object` | `process_property_value()` converts to `float32` |

**Before/After Results:**
- Without dtype normalization: 0% success rate
- With dtype normalization: 98% success rate (49/50 molecules)

**Key Takeaway:** For datasets where the preprocessor stores ragged arrays as `dtype=object`, the handler's `process_property_value()` method MUST normalize ALL properties to native dtypes (int64/float32/float64) before they reach the core pipeline. This is non-negotiable for successful processing.

### A.6 QDPiDataset Analysis (from `qdpi.py`) - LESSONS LEARNED

**Key Characteristics:**
- Uses `coordinate_based` strategy (no parseable identifiers)
- **FIRST dataset with charged molecule support** (ions, protonated/deprotonated species)
- DeePMD-kit HDF5 format (NOT standard HDF5 keys)
- ~1.6 million conformations from diverse drug-like molecules
- ωB97M-D3(BJ)/def2-TZVPPD level of DFT (highest accuracy)
- Coordinates in Angstrom, energies converted from eV to Hartree
- 13 elements: H, Li, C, N, O, F, Na, P, S, Cl, K, Br, I

**Source Files:**
- Dataset class: `datasets/implementations/qdpi.py`
- Handler class: `handlers/implementations/qdpi.py` (QDPiDatasetHandler)
- Preprocessor: `preprocessing/preprocessors/qdpi.py`

**Schema Summary:**
- Required properties: `('energy', 'atoms', 'coordinates')`
- Optional properties: `('forces', 'formula', 'molecular_charge', 'charge_type', 'subset')`
- Identifier keys: `()` (empty - coordinate_based strategy)
- Coordinate units: `'angstrom'`
- Energy units: `'hartree'` (converted from eV during preprocessing)

**Critical Implementation Lesson 1 - Charged Molecule Support:**

| Issue | Symptom | Root Cause | Solution |
|-------|---------|------------|----------|
| Bond order determination failure | `ValueError: Final molecular charge (0) does not match input (-1)` | Handler returned hardcoded charge=0 for charged molecules | Store `molecular_charge` in NPZ during preprocessing, read dynamically in handler |

**Charged Molecule Handling Pattern:**
1. **Preprocessing:** Track charge from directory structure (`data/neutral/` vs `data/charged/`)
2. **NPZ Storage:** Store `molecular_charge` and `charge_type` fields
3. **Handler:** `get_molecular_charge()` reads from NPZ, not hardcoded

```python
# In preprocessor - store charge during extraction
molecular_charges.append(inferred_charge)
charge_types.append('neutral' if charge == 0 else 'charged')

# In handler - read charge dynamically
def get_molecular_charge(self, raw_properties_dict, atomic_numbers, mol_identifier) -> int:
    if 'molecular_charge' in raw_properties_dict:
        return int(raw_properties_dict['molecular_charge'])
    return 0  # Default to neutral
```

**Critical Implementation Lesson 2 - DeePMD-kit HDF5 Format:**

| Wrong Key (Assumed) | Correct DeePMD-kit Key | Data Type |
|---------------------|------------------------|-----------|
| `elements` | `type_map.raw` | Byte strings → decode to element symbols |
| `atomic_types` | `type.raw` | Integer indices into type_map |
| `coordinates` | `set.XXX/coord.npy` | Flattened array (Nc × Na × 3) → reshape |
| `energies` | `set.XXX/energy.npy` | Energies per conformer |
| `forces` | `set.XXX/force.npy` | Flattened forces → reshape |

**Key Takeaway:** DeePMD-kit HDF5 format uses `type_map.raw`, `type.raw`, and `set.XXX/*.npy` structure. Always verify format documentation before assuming key names.

**Critical Implementation Lesson 3 - Config Structure Flattening:**

The main CLI flattens preprocessing config before passing to preprocessor:

```yaml
# In config.yaml (nested structure)
qdpi_config:
  processing_config:
    preprocessing:
      num_molecules: 25
      property_keys: [energy, force]

# What preprocessor receives (flattened)
self.config = {
    'num_molecules': 25,
    'property_keys': ['energy', 'force'],
    ...
}
```

**Pattern:** Always read config keys from `self.config.get('key')` at root level, NOT from nested `self.config.get('preprocessing', {}).get('key')`.

**Before/After Results:**
- Without charged molecule handling: ~8% failure rate on charged molecules
- With dynamic charge reading: 92% success rate (remaining 8% are legitimate valence issues)

---

## Document Information

**Version:** 2.10.0
**Created:** Based on systematic analysis of MILIA refactored source code
**Updated:** 2026-01-29 - Feature Tier Support: Added guidance for datasets with tiered feature extraction (e.g., basic/standard/complete)

**Source Files Analyzed - Dataset Module:**
- `base.py` (228 lines) - BaseDataset, DatasetMetadata, DatasetSchema, DatasetFeatures
- `registry.py` (175 lines) - DatasetRegistry, @register decorator
- `protocols.py` (115 lines) - DatasetHandlerProtocol (11 methods)
- `dft.py` (109 lines) - DFTDataset reference implementation
- `dmc.py` (115 lines) - DMCDataset reference implementation
- `wavefunction.py` (137 lines) - WavefunctionDataset reference implementation
- `qm9.py` (195 lines) - QM9Dataset reference implementation
- `ani1x.py` (180 lines) - ANI1xDataset reference implementation (NEW)
- `implementations/__init__.py` (~46 lines) - Dynamic discovery pattern (auto-imports all dataset modules)
- `datasets/__init__.py` (508 lines) - Module API

**Source Files Analyzed - Handler Module (Phase 7 Migration Complete):**
- `handlers/__init__.py` (680 lines) - Backward-compatible exports with lazy loading + recursion guard
- `handlers/base_handler.py` (1,527 lines) - DatasetHandler ABC + factory functions + shared utilities
- `handlers/handler_registry.py` (326 lines) - HandlerRegistry + @register_handler decorator
- `handlers/implementations/__init__.py` (78 lines) - Dynamic discovery pattern
- `handlers/implementations/dft.py` (1,280 lines) - DFTDatasetHandler
- `handlers/implementations/dmc.py` (979 lines) - DMCDatasetHandler
- `handlers/implementations/wavefunction.py` (1,059 lines) - WavefunctionDatasetHandler
- `handlers/implementations/qm9.py` (871 lines) - QM9DatasetHandler
- `handlers/implementations/ani1x.py` (1,015 lines) - ANI1xDatasetHandler
- `handlers/implementations/ani1ccx.py` (1,009 lines) - ANI1ccxDatasetHandler
- `handlers/implementations/ani2x.py` (922 lines) - ANI2xDatasetHandler
- `handlers/implementations/rmd17.py` (947 lines) - RMD17DatasetHandler
- `handlers/implementations/xxmd.py` (950 lines) - XXMDDatasetHandler
- `handlers/implementations/qdpi.py` (950 lines) - QDPiDatasetHandler ⭐ NEW (charged molecule support)

**Source Files Analyzed - Preprocessing Module:**
- `base_preprocessor.py` (127 lines) - BasePreprocessor ABC
- `registry.py` (130 lines) - PreprocessorRegistry
- `archive_handlers.py` (117 lines) - tar.gz extraction utilities
- `format_parsers.py` (420 lines) - Molden format parser + `FEATURE_TIERS` dict (single source of truth for tier-aware validation)
- `npz_builders.py` (175 lines) - NPZ file construction utilities
- `wavefunction.py` (195 lines) - WavefunctionPreprocessor reference implementation
- `ani1x.py` (450 lines) - ANI1xPreprocessor reference implementation (NEW)
- `preprocessing/__init__.py` (476 lines) - Module API
- `preprocessors/__init__.py` (~40 lines) - Dynamic discovery pattern (auto-imports all preprocessor modules)
- `utils/__init__.py` (31 lines) - Utility exports

**Documentation Files Analyzed:**
- `MILIA_Dataset_Architecture_Refactoring_Plan_v2_2_0.md` (2151 lines)
- `MILIA_Pipeline_Project_Structure.md` (2274 lines)

**Lessons Learned - QM9 Implementation (2025-12-27):**
- `identifier_keys` order is CRITICAL: InChI must be first when both InChI and SMILES available
- SMILES implicit hydrogens cause atom count mismatches with QM coordinates
- InChI explicitly encodes all hydrogens, ensuring correct atom counts
- Dataset class schema defines identifier order, not handler class

**Lessons Learned - ANI-1x Implementation (2026-01-01):**
- Preprocessors storing ragged arrays as `dtype=object` require handler-level dtype normalization
- `np.isfinite()` validation fails on object arrays - must convert to native numeric dtypes
- `torch.tensor()` cannot convert object arrays - must convert to float32/float64/int64
- Handler's `process_property_value()` is the correct location for dtype normalization
- This pattern maintains zero-core-file-modification architecture

**Lessons Learned - Handler Module Refactoring (2026-01-04) + Phase 7 Migration (2026-01-05):**
- Modular handler structure (`handlers/implementations/`) improves maintainability
- `@register_handler` decorator enables automatic registration without manual `__init__.py` updates
- Dynamic discovery pattern (`handlers/implementations/__init__.py`) auto-detects new handler files
- Thread-safe `HandlerRegistry` with `RLock` prevents concurrent access issues
- Lazy loading via `__getattr__` in `handlers/__init__.py` maintains 100% backward compatibility
- **Phase 7 Complete:** Legacy `dataset_handlers.py` has been **REMOVED** (~9,713 lines eliminated)
- Factory functions (`create_dataset_handler`, `validate_dataset_handler_compatibility`, etc.) migrated to `base_handler.py`
- Recursion guard (`_DISCOVERING_HANDLERS` flag) prevents infinite loops during implementations/ import
- Adding new handlers requires only: (1) create handler file in `implementations/`, (2) use `@register_handler` decorator
- All existing imports (`from milia_pipeline.handlers import XHandler`) continue to work unchanged

**Lessons Learned - Circular Import Resolution (2026-01-05):**
- Module-level handler imports in dataset files cause circular dependencies
- Circular chain: `datasets/implementations/{dataset}.py` → `handlers/implementations/{handler}.py` → `config containers` → `dataset registry` → `{dataset}.py` (CYCLE!)
- **Solution:** Replace `handler_class = XHandler` with `create_handler()` method using lazy import
- Lazy import inside `create_handler()` breaks the cycle - import happens at runtime, not module load time
- All 9 dataset implementations use correct pattern: DFT, DMC, Wavefunction, QM9, ANI1x, ANI1ccx, ANI2x, RMD17, XXMD
- Pattern is non-breaking: method signature matches `BaseDataset.create_handler()` exactly
- Pattern is dynamic: handlers discovered via `@register_handler` decorator
- Pattern is production-ready: tested with full pipeline runs
- Pattern is future-proof: `dataset_handlers.py` has been completely removed

**Lessons Learned - QDπ Implementation (2026-01-06):**
- **Charged Molecule Support:** First dataset requiring dynamic molecular charge (not hardcoded to 0)
- rdDetermineBonds requires correct charge parameter for proper bond order determination
- Charged molecules (ions, protonated amino acids, deprotonated species) fail with charge=0
- **Solution:** Store `molecular_charge` in NPZ during preprocessing, read in handler's `get_molecular_charge()`
- **DeePMD-kit HDF5 Format:** Uses non-standard key names (`type_map.raw`, `type.raw`, `set.XXX/*.npy`)
- Never assume HDF5 key names - always verify format documentation first
- **Config Flattening:** CLI flattens `processing_config.preprocessing` to root level before passing to preprocessor
- Read config keys from `self.config.get('key')`, not nested structure
- **Unit Conversion:** QDπ source data in eV, must convert to Hartree (factor: 0.0367493)
- This pattern (charged molecule handling) can be reused for future ionic/charged datasets

**Lessons Learned - Dynamic Discovery Pattern (2026-01-26):**
- **Dynamic Auto-Discovery:** Both `datasets/implementations/__init__.py` and `preprocessing/preprocessors/__init__.py` now use fully automatic dynamic discovery
- **Zero Manual Imports Required:** Adding new datasets or preprocessors NO LONGER requires editing `__init__.py` files
- Discovery mechanism: `pathlib.Path(__file__).parent.glob('*.py')` scans directory at package load time
- Modules are imported via `importlib.import_module()`, triggering their `@register` decorators automatically
- **Excluded modules:** `__init__`, `base`, `registry`, `utils`, `common`, `protocols` (and files starting with `_`)
- **Class detection:** Convention-based - classes ending with `Dataset` or `Preprocessor` are detected
- **Validation:** Dynamic discovery verifies classes have required attributes (`metadata`, `metadata.name`)
- Pattern is non-breaking: existing imports like `from milia_pipeline.datasets.implementations import DFTDataset` continue to work
- Pattern is dynamic: new files are automatically discovered on next Python interpreter startup
- Pattern is production-ready: uses try/except with logging for graceful error handling
- Pattern is future-proof: eliminates manual maintenance overhead and reduces human error in registration

**Lessons Learned - Feature Tier Support (2026-01-29):**
- **Tiered Feature Extraction:** Some datasets (e.g., Wavefunction) support multiple `feature_tier` levels (basic/standard/complete) that control computational cost vs. feature richness
- **Single Source of Truth:** Define tier contents in a `FEATURE_TIERS` dict in `format_parsers.py` (or dataset-specific parser) - this dict is the authoritative reference for all tier-aware validation
- **Tier Metadata in NPZ:** Preprocessor MUST store `feature_tier` in NPZ metadata so downstream components know which tier was used
- **Tier-Aware Validation:** `milia_dataset.py` validates critical keys against tier-specific expectations, not against "complete" tier keys
- **Tier-Aware Handler Enrichment:** Handler's `_add_scalar_targets_internal()` must filter `scalar_graph_targets` to only require tier-available keys
- **Logging Best Practices:** Use INFO level for intentional tier exclusions ("14 keys intentionally excluded"), WARNING only for unexpected missing keys
- **Implementation Pattern (3 files):**
  1. `format_parsers.py`: Define `FEATURE_TIERS = {'basic': [...], 'standard': [...], 'complete': [...]}` matching actual extraction logic
  2. `milia_dataset.py`: Read `feature_tier` from NPZ metadata, import `FEATURE_TIERS`, validate only tier-appropriate keys
  3. `{dataset}_handler.py`: Read `_feature_tier` from `raw_properties_dict`, filter `scalar_graph_targets` to tier-available keys
- **Critical:** `FEATURE_TIERS` dict MUST exactly match features extracted by `_extract_molecule_features()` for each tier
- Pattern is non-breaking: datasets without `feature_tier` continue using all configured targets (backward compatible)
- Pattern is dynamic: tier read from NPZ at runtime, no hardcoded assumptions
- Pattern is production-ready: comprehensive error handling with fallbacks at every level
- Pattern is future-proof: adding new tiers requires only updating `FEATURE_TIERS` dict

**Compliance:** This blueprint follows the zero-core-file-modification architecture implemented in Phase 1-8.4 of the MILIA Dataset Architecture Refactoring Plan v2.2.0, includes the preprocessing subsystem for non-NPZ source data formats, reflects the Handler Module Refactoring v1.2.0 (Phase 7 Complete) which removed the legacy `dataset_handlers.py` and migrated factory functions to `base_handler.py`, incorporates the Circular Import Resolution v1.0.0 which replaced the `handler_class` attribute pattern with lazy import `create_handler()` methods to eliminate circular dependencies between dataset and handler modules, includes the QDπ Charged Molecule Support v1.0.0 which establishes patterns for handling datasets with both neutral and charged molecules (critical for drug discovery datasets with ions, protonated amino acids, and deprotonated species), incorporates the Dynamic Discovery Pattern v1.0.0 which eliminates the need for manual import updates when adding new datasets or preprocessors, and now includes the Feature Tier Support v1.0.0 which establishes patterns for datasets with tiered feature extraction (e.g., basic/standard/complete levels) requiring tier-aware validation and handler enrichment.
