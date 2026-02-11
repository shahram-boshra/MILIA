# milia_pipeline/descriptors/__init__.py

"""
Milia Pipeline - Descriptors Module

Production-ready molecular descriptor system with comprehensive feature calculation,
plugin support, and PyTorch Geometric integration.

This module provides a complete framework for computing and managing molecular
descriptors for the milia pipeline, including:

- 400+ RDKit descriptors across 6 categories
- Auto-discovery and registration system
- Plugin architecture for custom descriptors
- Batch processing with caching
- Validation and error handling
- PyTorch Geometric integration utilities

Quick Start
-----------
Basic usage for descriptor calculation:

    >>> from milia_pipeline.descriptors import DescriptorCalculator
    >>> from rdkit import Chem
    >>> 
    >>> # Initialize calculator
    >>> calculator = DescriptorCalculator()
    >>> 
    >>> # Calculate descriptors for a molecule
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> result = calculator.calculate_batch(mol, ["MolWt", "TPSA", "LogP"])
    >>> print(result.successful)
    {'MolWt': 46.07, 'TPSA': 20.23, 'LogP': -0.07}

Module Structure
----------------
The descriptors module is organized into several key components:

1. **Registry System** (descriptor_registry.py)
   - Central registry for all descriptors
   - Auto-discovery of RDKit descriptors
   - Plugin descriptor registration
   - Thread-safe singleton pattern

2. **Categories** (descriptor_categories.py)
   - Descriptor categorization (Constitutional, Topological, Electronic, etc.)
   - Metadata management (requirements, descriptions)
   - Helper functions for filtering and querying

3. **Calculator** (descriptor_calculator.py)
   - Batch calculation engine
   - Conformer generation for 3D descriptors
   - Charge computation when needed
   - Result caching for performance
   - Comprehensive error handling

4. **Validator** (descriptor_validator.py)
   - Value validation (NaN, Inf, range checking)
   - Requirement checking (3D coords, charges)
   - Configuration validation
   - Batch validation utilities

5. **Integration** (descriptor_integration.py)
   - PyTorch Geometric Data integration
   - Tensor conversion utilities
   - Feature merging and broadcasting
   - Dataset statistics

6. **Plugin System** (descriptor_plugin_system.py)
   - Plugin discovery from YAML configs
   - Security and dependency validation
   - Version management
   - Enable/disable functionality

Available Categories
--------------------
- **Constitutional**: Basic molecular properties (35 descriptors)
  Examples: MolWt, NumHDonors, NumHAcceptors, TPSA, LogP
  
- **Topological**: Graph-based descriptors (350+ descriptors)
  Examples: BertzCT, Chi indices, VSA descriptors, Kappa indices
  
- **Electronic**: Electronic properties (8 descriptors)
  Examples: MaxPartialCharge, MinPartialCharge, MaxAbsPartialCharge
  
- **Geometric**: 3D structure descriptors (10 descriptors)
  Examples: RadiusOfGyration, Asphericity, Eccentricity, SpherocityIndex
  
- **Drug-likeness**: Druglikeness metrics (4 descriptors)
  Examples: QED, SPS (Synthetic Pathway Score)
  
- **Fragments**: Functional group counts (85 descriptors)
  Examples: fr_alkyl_halide, fr_amine, fr_benzene, fr_ester

Usage Examples
--------------

Example 1: Calculate specific descriptors
    >>> from milia_pipeline.descriptors import DescriptorCalculator
    >>> from rdkit import Chem
    >>> 
    >>> calculator = DescriptorCalculator()
    >>> mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
    >>> 
    >>> # Calculate specific descriptors
    >>> descriptors = ["MolWt", "TPSA", "NumHDonors", "NumHAcceptors"]
    >>> result = calculator.calculate_batch(mol, descriptors, mol_identifier="aspirin")
    >>> 
    >>> print(f"Successful: {result.successful}")
    >>> print(f"Failed: {result.failed}")
    >>> print(f"Time: {result.total_time:.3f}s")

Example 2: Calculate descriptors by category
    >>> from milia_pipeline.descriptors import (
    ...     DescriptorCalculator,
    ...     DescriptorCategory,
    ...     get_category_descriptor_names
    ... )
    >>> from rdkit import Chem
    >>> 
    >>> # Get all constitutional descriptors
    >>> constitutional_descs = get_category_descriptor_names(
    ...     DescriptorCategory.CONSTITUTIONAL
    ... )
    >>> 
    >>> calculator = DescriptorCalculator()
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> 
    >>> result = calculator.calculate_batch(mol, constitutional_descs)
    >>> print(f"Calculated {len(result.successful)} constitutional descriptors")

Example 3: Filter descriptors by requirements
    >>> from milia_pipeline.descriptors import (
    ...     DescriptorValidator,
    ...     filter_by_requirements
    ... )
    >>> from rdkit import Chem
    >>> 
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> all_descriptors = ["MolWt", "TPSA", "RadiusOfGyration", "Asphericity"]
    >>> 
    >>> # Filter based on what molecule has
    >>> validator = DescriptorValidator()
    >>> filtered = validator.filter_by_requirements(mol, all_descriptors)
    >>> 
    >>> print(f"Valid: {filtered['valid']}")
    >>> print(f"Invalid: {filtered['invalid']}")
    >>> print(f"Has 3D: {filtered['molecule_has_3d']}")

Example 4: Integrate with PyTorch Geometric
    >>> from milia_pipeline.descriptors import (
    ...     DescriptorCalculator,
    ...     add_descriptors_to_pyg_data
    ... )
    >>> from torch_geometric.data import Data
    >>> from rdkit import Chem
    >>> import torch
    >>> 
    >>> # Create PyG Data object
    >>> x = torch.randn(5, 10)  # 5 atoms, 10 features each
    >>> edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    >>> data = Data(x=x, edge_index=edge_index)
    >>> 
    >>> # Calculate descriptors
    >>> mol = Chem.MolFromSmiles("CCCCC")
    >>> calculator = DescriptorCalculator()
    >>> result = calculator.calculate_batch(mol, ["MolWt", "TPSA", "LogP"])
    >>> 
    >>> # Add to PyG Data
    >>> data = add_descriptors_to_pyg_data(
    ...     data,
    ...     result.successful,
    ...     create_feature_vector=True
    ... )
    >>> 
    >>> print(data.descriptor_features)  # Tensor of descriptor values
    >>> print(data.descriptor_names)     # List of descriptor names

Example 5: Load and use custom plugins
    >>> from milia_pipeline.descriptors import (
    ...     DescriptorPluginLoader,
    ...     DescriptorCalculator
    ... )
    >>> from pathlib import Path
    >>> 
    >>> # Discover plugins
    >>> plugin_loader = DescriptorPluginLoader()
    >>> plugin_paths = [Path("./plugins/descriptors")]
    >>> discovered = plugin_loader.discover_plugins(
    ...     paths=plugin_paths,
    ...     auto_validate=True
    ... )
    >>> 
    >>> print(f"Discovered {len(discovered)} plugin(s)")
    >>> 
    >>> # Use custom descriptors
    >>> calculator = DescriptorCalculator()
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> result = calculator.calculate_batch(mol, ["CustomDescriptor1"])

Example 6: Batch processing with statistics
    >>> from milia_pipeline.descriptors import DescriptorCalculator
    >>> from rdkit import Chem
    >>> 
    >>> calculator = DescriptorCalculator(enable_cache=True)
    >>> molecules = [
    ...     (Chem.MolFromSmiles("CCO"), "ethanol"),
    ...     (Chem.MolFromSmiles("CC(C)O"), "isopropanol"),
    ...     (Chem.MolFromSmiles("CCCO"), "propanol")
    ... ]
    >>> 
    >>> descriptors = ["MolWt", "TPSA", "LogP"]
    >>> results = calculator.calculate_for_molecules(molecules, descriptors)
    >>> 
    >>> # Get statistics
    >>> stats = calculator.get_statistics()
    >>> print(f"Total calculations: {stats['total_calculations']}")
    >>> print(f"Success rate: {stats['success_rate']:.1f}%")
    >>> print(f"Cache hit rate: {stats['cache_hit_rate']:.1f}%")

Configuration Integration
-------------------------
The descriptor system integrates with the milia pipeline configuration system
through the config.yaml file:

    descriptors:
      enabled: true
      selection_mode: explicit
      selected_descriptors:
        constitutional:
          - MolWt
          - TPSA
          - NumHDonors
          - NumHAcceptors
        topological:
          - BertzCT
          - Chi0v
      
      computation:
        enable_cache: true
        generate_conformers: true
        fallback_on_error: true
      
      plugins:
        enabled: true
        plugin_paths:
          - ./plugins/descriptors
        auto_discover: true

Performance Considerations
--------------------------
- **Caching**: Enable caching for repeated calculations on same molecules
- **Conformers**: 3D descriptor calculation requires conformer generation
- **Batch Size**: Use batch processing for multiple molecules
- **Error Handling**: Use fallback mode to continue on individual failures
- **Plugin Loading**: Load plugins at initialization to avoid repeated discovery

Thread Safety
-------------
All core components are thread-safe:
- DescriptorRegistry: Thread-safe singleton with locks
- DescriptorCalculator: Thread-safe when using separate instances
- DescriptorPluginLoader: Thread-safe plugin discovery and registration

Error Handling
--------------
The module uses a comprehensive exception hierarchy:
- DescriptorError: Base exception for descriptor operations
- DescriptorCalculationError: Calculation failures
- DescriptorValidationError: Validation failures
- DescriptorPluginError: Plugin-related errors
- DescriptorPluginLoadError: Plugin loading failures
- DescriptorPluginValidationError: Plugin validation failures
- DescriptorPluginConfigError: Plugin configuration errors

API Reference
-------------
See individual module docstrings for detailed API documentation:
- descriptor_registry: Registry and auto-discovery
- descriptor_categories: Category definitions and metadata
- descriptor_calculator: Calculation engine
- descriptor_validator: Validation utilities
- descriptor_integration: PyTorch Geometric integration
- descriptor_plugin_system: Plugin management

Version Information
-------------------
Module Version: 1.0.0
Compatible with: RDKit >= 2022.03.1, PyTorch Geometric >= 2.0.0

Authors
-------
milia Team

License
-------
See project LICENSE file
"""

# Version
__version__ = "1.0.0"

# =============================================================================
# IMPORTS - CORE COMPONENTS
# =============================================================================

# Registry System
from .descriptor_registry import (
    DescriptorRegistry,
    DescriptorRegistration,
    registry,
    get_descriptor,
    has_descriptor,
    list_descriptors,
    auto_discover_rdkit,
)

# Categories and Metadata
from .descriptor_categories import (
    DescriptorCategory,
    DescriptorMetadata,
    get_descriptors_by_category,
    get_descriptor_metadata,
    requires_3d_coordinates,
    requires_partial_charges,
    get_all_descriptor_names,
    get_category_descriptor_names,
    filter_descriptors_by_requirements,
    get_descriptor_count_by_category,
    validate_descriptor_coverage,
    DESCRIPTOR_METADATA_MAP,
    ALL_DESCRIPTORS,
    DESCRIPTORS_BY_CATEGORY,
)

# Calculator
from .descriptor_calculator import (
    DescriptorCalculator,
    CalculationResult,
    BatchCalculationResult,
)

# Validator
from .descriptor_validator import (
    DescriptorValidator,
    ValidationResult,
    validate_value,
    check_requirements,
    filter_by_requirements,
    validator,
)

# Integration with PyTorch Geometric
from .descriptor_integration import (
    descriptors_to_tensor,
    add_descriptors_to_pyg_data,
    merge_descriptors_with_features,
    extract_descriptors_from_pyg_data,
    validate_descriptor_integration,
    get_descriptor_statistics,
)

# Plugin System
from .descriptor_plugin_system import (
    DescriptorPluginLoader,
    DescriptorPluginMetadata,
    DescriptorDeclaration,
    plugin_loader,
    discover_plugins,
    validate_plugin,
    list_plugins,
    get_plugin_info,
)


# =============================================================================
# PUBLIC API DEFINITION
# =============================================================================

__all__ = [
    # Version
    "__version__",
    
    # Registry System
    "DescriptorRegistry",
    "DescriptorRegistration",
    "registry",
    "get_descriptor",
    "has_descriptor",
    "list_descriptors",
    "auto_discover_rdkit",
    
    # Categories and Metadata
    "DescriptorCategory",
    "DescriptorMetadata",
    "get_descriptors_by_category",
    "get_descriptor_metadata",
    "requires_3d_coordinates",
    "requires_partial_charges",
    "get_all_descriptor_names",
    "get_category_descriptor_names",
    "filter_descriptors_by_requirements",
    "get_descriptor_count_by_category",
    "validate_descriptor_coverage",
    "DESCRIPTOR_METADATA_MAP",
    "ALL_DESCRIPTORS",
    "DESCRIPTORS_BY_CATEGORY",
    
    # Calculator
    "DescriptorCalculator",
    "CalculationResult",
    "BatchCalculationResult",
    
    # Validator
    "DescriptorValidator",
    "ValidationResult",
    "validate_value",
    "check_requirements",
    "filter_by_requirements",
    "validator",
    
    # Integration
    "descriptors_to_tensor",
    "add_descriptors_to_pyg_data",
    "merge_descriptors_with_features",
    "extract_descriptors_from_pyg_data",
    "validate_descriptor_integration",
    "get_descriptor_statistics",
    
    # Plugin System
    "DescriptorPluginLoader",
    "DescriptorPluginMetadata",
    "DescriptorDeclaration",
    "plugin_loader",
    "discover_plugins",
    "validate_plugin",
    "list_plugins",
    "get_plugin_info",
]


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

import logging

logger = logging.getLogger(__name__)

# Initialize module
logger.info(f"Milia Descriptors Module v{__version__} initialized")
logger.debug(f"Registry contains {len(registry.list_all_descriptors())} descriptors")

# Note: Descriptor registry auto-discovery happens in descriptor_registry.py
# Plugin discovery should be triggered by the main pipeline or explicitly by user
