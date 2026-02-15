"""
MILIA Pipeline Molecules Package
================================

This package provides comprehensive molecular data processing, conversion,
validation, filtering, and feature enrichment capabilities for the MILIA
machine learning pipeline.

Package Structure:
-----------------
molecules/
├── __init__.py                  # This file - package exports
├── molecule_converter_core.py   # Core molecule conversion (PyG Data creation)
├── molecule_validator.py        # Molecular structure validation
├── molecule_filters.py          # Molecule filtering with registry integration
├── molecule_feature_enricher.py # Feature enrichment and fingerprints
├── property_enrichment.py       # Property enrichment utilities
└── mol_structural_features.py   # Structural feature calculations

Core Components:
---------------
1. MoleculeDataConverter - Main class for converting raw data to PyG Data
2. MoleculeFilter - Filtering with handler integration and transform awareness
3. Validation functions - Structure, compatibility, and completeness checks
4. Enrichment functions - Property enrichment, fingerprints, feature extraction

Usage Examples:
--------------
Basic conversion:
    >>> from milia_pipeline.molecules import MoleculeDataConverter
    >>> converter = MoleculeDataConverter(dataset_config, logger=logger)
    >>> pyg_data = converter.convert(raw_data, molecule_index=0)

Filtering molecules:
    >>> from milia_pipeline.molecules import create_molecule_filter
    >>> mol_filter = create_molecule_filter(
    ...     dataset_config=config,
    ...     filter_config=filter_config,
    ...     handler=handler
    ... )
    >>> mol_filter.apply_filters(pyg_data)

Validation:
    >>> from milia_pipeline.molecules import validate_molecular_structure
    >>> atomic_numbers, coords = validate_molecular_structure(
    ...     atoms=atoms,
    ...     coordinates=coords,
    ...     molecule_index=0,
    ...     handler=handler
    ... )

Feature enrichment:
    >>> from milia_pipeline.molecules import enrich_pyg_data_with_properties
    >>> enriched_data = enrich_pyg_data_with_properties(pyg_data, properties)

Registry integration status (Phase 6):
    >>> from milia_pipeline.molecules import get_registry_integration_status
    >>> status = get_registry_integration_status()
    >>> print(status['phase_6_complete'])  # True

    # Get status from specific modules:
    >>> from milia_pipeline.molecules import get_enricher_registry_status
    >>> enricher_status = get_enricher_registry_status()

    >>> from milia_pipeline.molecules import get_validator_registry_status
    >>> validator_status = get_validator_registry_status()

    >>> from milia_pipeline.molecules import get_filter_registry_status
    >>> filter_status = get_filter_registry_status()

Handler-Only Architecture:
-------------------------
All modules follow the handler-only architecture pattern:
- Handlers are NEVER created within this package
- All handler-dependent functions accept handlers as parameters
- Zero compatibility layer imports
- Zero backward compatibility mechanisms

Phase 6 Registry Integration:
----------------------------
All modules now support dynamic dataset type discovery:
- Dynamic dataset type validation
- Feature-based processing queries
- Automatic support for new dataset types
- Legacy fallbacks when registry unavailable

For advanced usage and internal utilities, import directly from submodules:
    >>> from milia_pipeline.molecules.molecule_converter_core import (
    ...     _get_dataset_feature,
    ...     _get_available_dataset_types
    ... )
"""

import logging

# Package version - Updated for Phase 6 molecule_filters registry integration
__version__ = "1.4.0"

_logger = logging.getLogger(__name__)

# ==============================================================================
# Core Classes
# ==============================================================================

# MoleculeDataConverter - Main class for raw data to PyG Data conversion
# ==============================================================================
# Feature Functions
# ==============================================================================
# Structural features from mol_structural_features
from .mol_structural_features import (
    add_structural_features,
    get_available_features,
)

# ==============================================================================
# Conversion Functions
# ==============================================================================
# Core conversion utilities from molecule_converter_core
from .molecule_converter_core import (
    MoleculeDataConverter,
    create_mol_with_dataset_support,
    create_rdkit_mol,
    mol_to_pyg_data,
)

# Molecular feature enrichment functions
# Phase 6 Updated: Added registry status and handler-only functions
from .molecule_feature_enricher import (
    analyze_capabilities_with_handler,
    analyze_structural_feature_capabilities,
    create_handler_compatible_fingerprint,
    estimate_molecular_properties,
    # Handler-only operation functions
    estimate_properties_with_handler,
    get_feature_extraction_diagnostics,
    get_molecule_identifiers,
    get_structural_feature_summary,
    validate_feature_extraction_with_handler,
)
from .molecule_feature_enricher import (
    # Phase 6: Registry diagnostics
    get_registry_integration_status as get_enricher_registry_status,
)
from .molecule_filters import (
    _REGISTRY_AVAILABLE as _FILTER_REGISTRY_AVAILABLE,
)
from .molecule_filters import (
    _REGISTRY_IMPORT_ERROR as _FILTER_REGISTRY_IMPORT_ERROR,
)
from .molecule_filters import (
    _REGISTRY_INITIALIZED as _FILTER_REGISTRY_INITIALIZED,
)

# MoleculeFilter - Filtering with handler integration and transform awareness
# ==============================================================================
# Filtering Functions
# ==============================================================================
# Core filtering functions
# Phase 6 Updated: Added registry integration functions
from .molecule_filters import (
    MoleculeFilter,
    apply_atom_count_filters,
    apply_dataset_specific_filters,
    apply_heavy_atom_filters,
    # Core filtering functions
    apply_pre_filters,
    create_handler_aware_filter_stats,
    # Factory functions
    create_molecule_filter,
    get_default_molecule_filter,
    # Utility functions
    introspect_transform_filter_parameters,
    validate_filter_compatibility_with_transforms,
    # Validation functions
    validate_filter_configuration,
)
from .molecule_filters import (
    _get_available_dataset_types as _filter_get_available_dataset_types,
)
from .molecule_filters import (
    _get_dataset_feature as _filter_get_dataset_feature,
)
from .molecule_filters import (
    _get_handler_error_type_for_dataset as _filter_get_handler_error_type_for_dataset,
)
from .molecule_filters import (
    # Phase 6: Registry integration functions (internal)
    _init_registry as _filter_init_registry,
)
from .molecule_filters import (
    _is_dataset_type_registered as _filter_is_dataset_type_registered,
)

# ==============================================================================
# Validation Functions
# ==============================================================================
# Core validation functions for molecular structures
# Phase 6 Updated: Added get_registry_status for validator registry diagnostics
from .molecule_validator import (
    check_dataset_compatibility,
    validate_molecular_structure,
    validate_pyg_data_completeness,
)
from .molecule_validator import (
    get_registry_status as get_validator_registry_status,  # Phase 6: Registry diagnostics
)

# ==============================================================================
# Enrichment Functions
# ==============================================================================
# Property enrichment utilities
# Phase 6 Updated: Added get_registry_integration_status
from .property_enrichment import (
    calculate_atomization_energy,
    enrich_pyg_data_with_properties,
    get_registry_integration_status,  # Phase 6: Registry diagnostics
)

# ==============================================================================
# Phase 6: Module-Level Registry Status Function for molecule_filters
# ==============================================================================


def get_filter_registry_status():
    """
    Get registry integration status for molecule_filters module.

    PHASE 6: Module-level convenience function that provides registry
    status without needing to instantiate MoleculeFilter.

    Returns:
        Dict containing registry availability and configuration

    Example:
        >>> status = get_filter_registry_status()
        >>> print(status['registry_available'])
        True
        >>> print(status['available_dataset_types'])
        ['DFT', 'DMC', 'Wavefunction']
    """
    _filter_init_registry()

    return {
        "registry_available": _FILTER_REGISTRY_AVAILABLE,
        "registry_initialized": _FILTER_REGISTRY_INITIALIZED,
        "registry_import_error": _FILTER_REGISTRY_IMPORT_ERROR,
        "available_dataset_types": _filter_get_available_dataset_types(),
        "phase_6_complete": True,
        "features_available": [
            "uncertainty_handling",
            "vibrational_analysis",
            "atomization_energy",
            "rotational_constants",
            "frequency_analysis",
            "orbital_analysis",
            "homo_lumo_gap",
            "mo_energies",
        ],
    }


# ==============================================================================
# Public API Exports
# ==============================================================================

__all__ = [
    # Core classes
    "MoleculeDataConverter",
    "MoleculeFilter",
    # Conversion
    "create_rdkit_mol",
    "mol_to_pyg_data",
    "create_mol_with_dataset_support",
    # Features
    "add_structural_features",
    "get_available_features",
    # Validation
    "validate_molecular_structure",
    "check_dataset_compatibility",
    "validate_pyg_data_completeness",
    # Enrichment
    "enrich_pyg_data_with_properties",
    "calculate_atomization_energy",
    "estimate_molecular_properties",
    "get_molecule_identifiers",
    "get_structural_feature_summary",
    "get_feature_extraction_diagnostics",
    "analyze_structural_feature_capabilities",
    # Handler-only operations (molecule_feature_enricher module)
    "estimate_properties_with_handler",
    "analyze_capabilities_with_handler",
    "create_handler_compatible_fingerprint",
    "validate_feature_extraction_with_handler",
    # Filtering - Factory functions
    "create_molecule_filter",
    "get_default_molecule_filter",
    # Filtering - Core functions
    "apply_pre_filters",
    "apply_atom_count_filters",
    "apply_heavy_atom_filters",
    "apply_dataset_specific_filters",
    # Filtering - Validation
    "validate_filter_configuration",
    "validate_filter_compatibility_with_transforms",
    # Filtering - Utilities
    "introspect_transform_filter_parameters",
    "create_handler_aware_filter_stats",
    # Diagnostics (Phase 6)
    "get_registry_integration_status",  # From property_enrichment
    "get_enricher_registry_status",  # From molecule_feature_enricher (aliased)
    "get_validator_registry_status",  # From molecule_validator (aliased)
    "get_filter_registry_status",  # From this module (Phase 6 molecule_filters)
]


# ==============================================================================
# Module Initialization Logging
# ==============================================================================

_logger.debug(f"MILIA Molecules package v{__version__} loaded")

# Log Phase 6 registry status for molecule_filters
try:
    if _FILTER_REGISTRY_AVAILABLE:
        _logger.debug("Phase 6: molecule_filters registry integration available")
    else:
        _logger.debug("Phase 6: molecule_filters using legacy fallback")
except NameError:
    _logger.debug("Phase 6: molecule_filters registry status not available")
