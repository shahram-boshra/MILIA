# milia_pipeline/datasets/__init__.py

"""
Milia Pipeline - Datasets Module
=================================

This module provides PyTorch Geometric compatible dataset implementations for the Milia pipeline,
with dynamic auto-discovery and registry-based dataset management supporting extensible molecular data formats.

Main Components
--------------
- MiliaDataset: Primary dataset class for Milia molecular data
- Factory functions: Convenience functions for dataset creation
- Dataset utilities: Helper functions for dataset operations

Features
--------
- PyTorch Geometric InMemoryDataset integration
- Dynamic dataset type discovery and registration
- Handler-based processing with comprehensive error handling
- Enhanced transformation system with experimental setup support
- Standard transforms support (always-applied production transforms)
- Combined transforms (standard + experimental) for pipeline execution
- Dynamic transform discovery and validation
- Descriptor system integration
- Uncertainty handling for stochastic methods
- Chunked processing for large datasets
- Configurable filtering and validation
- Phase 6: Registry-based feature queries for dynamic dataset processing

Phase 6 Registry Integration
---------------------------
The datasets module now supports dynamic, registry-based dataset processing:
- Feature-based queries replace hardcoded dataset type checks
- New dataset types automatically get appropriate processing based on features
- Generalized insight extraction methods (uncertainty, vibrational, orbital)
- Generalized metadata extraction methods
- Full backward compatibility with legacy code paths
- Zero modifications required to add new dataset types with appropriate features

Usage Examples
-------------
Basic usage:
    >>> from milia_pipeline.datasets import miliaDataset
    >>> dataset = miliaDataset(root='./data')
    >>> print(f"Dataset size: {len(dataset)}")
    >>> data = dataset[0]  # Access first molecule

Using factory method with configuration:
    >>> from milia_pipeline.datasets import miliaDataset
    >>> dataset = miliaDataset.from_config(
    ...     config_path='config.yaml',
    ...     root='./data',
    ...     dataset_type='DFT'
    ... )

With experimental setup:
    >>> dataset = miliaDataset.from_config(
    ...     config_path='config.yaml',
    ...     root='./data',
    ...     experimental_setup='milia_quantum_enhanced'
    ... )

Get transform configuration info (includes standard transforms):
    >>> dataset = miliaDataset(root='./data')
    >>> transform_info = dataset.get_transform_configuration_info()
    >>> print(f"Has standard transforms: {transform_info['has_standard_transforms']}")
    >>> print(f"Standard transforms count: {transform_info['standard_transforms_count']}")

Check registry integration status (Phase 6):
    >>> dataset = miliaDataset(root='./data')
    >>> status = dataset.get_registry_integration_status()
    >>> print(f"Registry available: {status['registry_available']}")
    >>> print(f"Phase 6 complete: {status['phase_6_complete']}")

List all registered datasets:
    >>> from milia_pipeline.datasets import list_all
    >>> print(f"Available datasets: {list_all()}")

Architecture
-----------
The datasets module implements a handler-only architecture:
- Direct handler creation via create_dataset_handler()
- Comprehensive exception handling with handler-specific errors
- Graceful degradation when handlers unavailable
- Enhanced validation without legacy fallback mechanisms
- Phase 6: Registry-based feature queries for dynamic processing

Integration
----------
This module integrates with:
- config: Configuration management and validation
- molecules: Molecular conversion and feature enrichment
- transformations: Graph transformation system
- handlers: Dataset handler integration
- descriptors: Molecular descriptor calculation
- preprocessing: Wavefunction data preprocessing
- exceptions: Comprehensive error handling
- registry: Dataset type registration and feature queries (Phase 6)

Version Information
------------------
Module Version: 1.4.0
Handler Architecture: Handler-Only (No Backward Compatibility)
Transformation System: Enhanced with Experimental Setup and Standard Transforms Support
Exception Integration: Complete Handler Exception Hierarchy
Phase 1 Registry: Dataset Registry Infrastructure
Phase 2 Datasets: Dynamic Auto-Discovery Pattern (Zero Manual Imports)
Phase 6 Integration: Registry-Based Feature Queries in milia_dataset.py
Standard Transforms: Production-ready transforms applied before experimental setups

Notes
-----
- This module uses ONLY the handler-based system with no fallback to legacy mechanisms
- All handler operations are direct with proper error recovery
- The module maintains full backward compatibility for user-facing APIs
- Enhanced transformation system supports systematic ML/DL research workflows
- Phase 6 enables zero-modification addition of new dataset types
- Dataset implementations are auto-discovered via the implementations submodule

See Also
--------
- milia_dataset.py: Complete implementation details
- config.yaml: Configuration reference
- Milia_Pipeline_Project_Structure.md: Project architecture
- MILIA_Adding_New_Datasets_Implementation_Blueprint.md: Dataset addition guide

Author: milia Pipeline Development Team
License: As per project license
"""

from typing import Any, Dict, List, Optional

# ============================================================================
# PHASE 2 ADDITIONS - Dataset Implementations (via Dynamic Auto-Discovery)
# ============================================================================
# NOTE: Dataset implementations are automatically discovered and registered
# via the dynamic discovery pattern in milia_pipeline.datasets.implementations.
# The @register decorators are triggered automatically when the implementations
# module is imported. No explicit imports are required here.
# See: MILIA_Adding_New_Datasets_Implementation_Blueprint.md (Dynamic Discovery Pattern v1.0.0)
import milia_pipeline.datasets.implementations  # Triggers auto-discovery and @register decorators

# ============================================================================
# PHASE 1 ADDITIONS - Dataset Registry Infrastructure
# ============================================================================
from milia_pipeline.datasets.base import (
    BaseDataset,
    DatasetFeatures,
    DatasetMetadata,
    DatasetSchema,
)
from milia_pipeline.datasets.protocols import (
    DatasetConverterProtocol,
    DatasetHandlerProtocol,
    DatasetValidatorProtocol,
)
from milia_pipeline.datasets.registry import (
    DatasetRegistry,
    get,
    get_default_registry,
    is_registered,
    list_all,
    register,
)
from milia_pipeline.exceptions import (
    DatasetNotFoundError,
    DatasetRegistrationError,
)

# ============================================================================
# Core Dataset Implementation
# ============================================================================
from .milia_dataset import miliaDataset

# ============================================================================
# Version and Metadata
# ============================================================================

__version__ = "1.4.0"
__author__ = "milia Pipeline Development Team"
__module_status__ = "Production Ready - Handler-Only Architecture with Phase 6 Registry Integration and Standard Transforms Support"

# ============================================================================
# Public API Exports
# ============================================================================

__all__ = [
    # Primary Dataset Class
    "miliaDataset",
    # Version and Metadata
    "__version__",
    "__author__",
    "__module_status__",
    # Phase 1: Base Classes
    "BaseDataset",
    "DatasetMetadata",
    "DatasetSchema",
    "DatasetFeatures",
    # Phase 1: Registry
    "DatasetRegistry",
    "get_default_registry",
    "register",
    "get",
    "list_all",
    "is_registered",
    # Phase 1: Protocols
    "DatasetHandlerProtocol",
    "DatasetConverterProtocol",
    "DatasetValidatorProtocol",
    # Phase 1: Exceptions
    "DatasetRegistrationError",
    "DatasetNotFoundError",
    # Phase 1: Plugin Initialization
    "initialize_plugins",
    # Phase 2: Dataset Implementations (accessed via dynamic __getattr__)
    # NOTE: Individual dataset classes (e.g., DFTDataset, DMCDataset) are available
    # via lazy import using the registry. Use get('DatasetName') or import directly.
    # Example: from milia_pipeline.datasets import DFTDataset  # Works via __getattr__
    # Dynamic Dataset Type Discovery
    "get_supported_dataset_types",
    "SUPPORTED_DATASET_TYPES",  # LEGACY - kept for backward compatibility
]


# ============================================================================
# Dynamic Dataset Class Access (Backward Compatibility)
# ============================================================================


def __getattr__(name: str):
    """
    Dynamic attribute access for dataset classes registered in the registry.

    This enables backward-compatible imports like:
        from milia_pipeline.datasets import DFTDataset
        from milia_pipeline.datasets import QM9Dataset

    Without requiring explicit imports in this module, aligning with the
    dynamic auto-discovery pattern.

    Args:
        name: The attribute name being accessed

    Returns:
        The dataset class if registered in the registry

    Raises:
        AttributeError: If the name is not a registered dataset class
    """
    # Check if this looks like a dataset class name (ends with 'Dataset')
    if name.endswith("Dataset"):
        # Extract the dataset type name (e.g., 'DFTDataset' -> 'DFT')
        dataset_type = name[:-7]  # Remove 'Dataset' suffix

        # Try to get from registry
        try:
            from milia_pipeline.datasets.registry import get as registry_get
            from milia_pipeline.datasets.registry import is_registered

            if is_registered(dataset_type):
                return registry_get(dataset_type)
        except Exception:
            pass

        # Try uppercase version (e.g., 'dft' -> 'DFT')
        try:
            from milia_pipeline.datasets.registry import get as registry_get
            from milia_pipeline.datasets.registry import is_registered

            if is_registered(dataset_type.upper()):
                return registry_get(dataset_type.upper())
        except Exception:
            pass

    raise AttributeError(f"module 'milia_pipeline.datasets' has no attribute '{name}'")


# ============================================================================
# Module-Level Documentation Variables
# ============================================================================

# PHASE 6.1 UPDATE: SUPPORTED_DATASET_TYPES is now dynamically populated
# This maintains backward compatibility while ensuring new dataset types are included.
# The constant is populated at module load time using filesystem discovery if registry unavailable.


def _discover_dataset_types_for_constant() -> list:
    """
    Discover dataset types for populating SUPPORTED_DATASET_TYPES constant.

    This function is called once at module load time to populate the legacy
    SUPPORTED_DATASET_TYPES constant with dynamically discovered types.

    DYNAMIC APPROACH:
    1. First tries the registry (primary source of truth)
    2. If registry fails, dynamically discovers from filesystem
    3. Never uses hardcoded dataset type lists

    Returns:
        List of discovered dataset type names

    ADDED Phase 6.1: Dynamic population of legacy constant
    """
    # Try registry first
    try:
        from milia_pipeline.datasets.registry import list_all

        return list_all()
    except Exception:
        pass

    # DYNAMIC FALLBACK: Discover dataset types from implementations directory
    try:
        from pathlib import Path

        implementations_dir = Path(__file__).parent / "implementations"
        if implementations_dir.exists():
            discovered_types = []
            for py_file in implementations_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                # Extract dataset name from filename (e.g., dft.py -> DFT, qm9.py -> QM9)
                dataset_name = py_file.stem.upper()
                # Exclude non-dataset modules
                if dataset_name not in ["BASE", "REGISTRY", "UTILS", "COMMON"]:
                    discovered_types.append(dataset_name)
            if discovered_types:
                return discovered_types
    except Exception:
        pass

    # Final fallback: return empty list (no hardcoded types)
    return []


# LEGACY CONSTANT - Now dynamically populated for backward compatibility
# Use get_supported_dataset_types() for guaranteed up-to-date results
SUPPORTED_DATASET_TYPES = _discover_dataset_types_for_constant()
"""List of supported dataset types in milia pipeline. Note: Use get_supported_dataset_types() for dynamic discovery."""


def get_supported_dataset_types():
    """
    Get list of supported dataset types dynamically.

    DYNAMIC APPROACH: Instead of hardcoded SUPPORTED_DATASET_TYPES, this function:
    1. First tries the registry (primary source of truth)
    2. If registry fails, dynamically discovers dataset implementations from filesystem
    3. Never relies on hardcoded SUPPORTED_DATASET_TYPES constant

    Returns:
        List of all supported dataset type names

    Note:
        Use this function instead of SUPPORTED_DATASET_TYPES constant
        for forward compatibility with new dataset types.
    """
    # Try registry first
    try:
        from milia_pipeline.datasets.registry import list_all

        return list_all()
    except Exception:
        pass

    # DYNAMIC FALLBACK: Discover dataset types from implementations directory
    try:
        from pathlib import Path

        implementations_dir = Path(__file__).parent / "implementations"
        if implementations_dir.exists():
            discovered_types = []
            for py_file in implementations_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                # Extract dataset name from filename (e.g., dft.py -> DFT, qm9.py -> QM9)
                dataset_name = py_file.stem.upper()
                # Exclude non-dataset modules
                if dataset_name not in ["BASE", "REGISTRY", "UTILS", "COMMON"]:
                    discovered_types.append(dataset_name)
            if discovered_types:
                return discovered_types
    except Exception:
        pass

    # Final fallback: return empty list
    return []


HANDLER_ARCHITECTURE_VERSION = "2.0"
"""Current handler architecture version (Handler-Only)."""

TRANSFORMATION_SYSTEM_VERSION = "2.1"
"""Enhanced transformation system with experimental setup and standard transforms support."""

REGISTRY_VERSION = "1.0"
"""Dataset registry infrastructure version (Phase 1)."""

IMPLEMENTATIONS_VERSION = "1.0"
"""Dataset implementations version (Phase 2)."""

PHASE_6_INTEGRATION_VERSION = "6.0.0"
"""Phase 6 registry-based feature query integration version."""

# ============================================================================
# Module Initialization
# ============================================================================


def _initialize_module():
    """
    Initialize the datasets module with validation checks.

    This function performs module-level initialization and validation,
    ensuring all required dependencies are available.

    Note:
        This is called automatically on module import and should not be
        called directly by users.
    """
    import logging

    logger = logging.getLogger(__name__)

    # Log module initialization
    logger.debug(f"Initializing datasets module (version {__version__})")
    logger.debug(f"Handler Architecture: {HANDLER_ARCHITECTURE_VERSION}")
    logger.debug(f"Transformation System: {TRANSFORMATION_SYSTEM_VERSION}")
    logger.debug(f"Registry Version: {REGISTRY_VERSION}")
    logger.debug(f"Implementations Version: {IMPLEMENTATIONS_VERSION}")
    logger.debug(f"Phase 6 Integration: {PHASE_6_INTEGRATION_VERSION}")

    # Check for optional dependencies
    try:
        import torch_geometric

        logger.debug("PyTorch Geometric available")
    except ImportError:
        logger.warning("PyTorch Geometric not available - dataset functionality will be limited")

    try:
        from milia_pipeline.handlers import create_dataset_handler

        logger.debug("Dataset handlers available")
    except ImportError:
        logger.warning("Dataset handlers not available - using basic processing")

    try:
        from milia_pipeline.transformations.graph_transforms import get_graph_transforms

        logger.debug("Enhanced transformation system available")
    except ImportError:
        logger.warning("Enhanced transformation system not available - using basic transforms")

    try:
        from milia_pipeline.descriptors.descriptor_calculator import DescriptorCalculator

        logger.debug("Descriptor system available")
    except ImportError:
        logger.warning("Descriptor system not available - descriptors will not be calculated")

    # Phase 2: Log registered datasets
    registered = list_all()
    logger.debug(f"Registered datasets: {registered}")

    # Phase 6: Log registry integration status
    logger.debug(f"Phase 6 registry integration active (version {PHASE_6_INTEGRATION_VERSION})")


# Perform module initialization
_initialize_module()

# ============================================================================
# Module Information Functions
# ============================================================================


def get_module_info() -> dict[str, Any]:
    """
    Get comprehensive information about the datasets module.

    Returns:
        dict: Dictionary containing module metadata including:
            - version: Module version
            - author: Module author
            - status: Module status
            - supported_types: List of supported dataset types
            - handler_architecture: Handler architecture version
            - transformation_system: Transformation system version
            - available_features: List of available features
            - phase_6_integration: Phase 6 registry integration version

    Example:
        >>> from milia_pipeline.datasets import get_module_info
        >>> info = get_module_info()
        >>> print(f"Version: {info['version']}")
        >>> print(f"Supported types: {info['supported_types']}")
        >>> print(f"Phase 6 integration: {info['phase_6_integration']}")
    """
    # Check feature availability
    features = []

    try:
        import torch_geometric

        features.append("pytorch_geometric")
    except ImportError:
        pass

    try:
        from milia_pipeline.handlers import create_dataset_handler

        features.append("dataset_handlers")
    except ImportError:
        pass

    try:
        from milia_pipeline.transformations.graph_transforms import get_graph_transforms

        features.append("enhanced_transforms")
    except ImportError:
        pass

    try:
        from milia_pipeline.descriptors.descriptor_calculator import DescriptorCalculator

        features.append("descriptors")
    except ImportError:
        pass

    # Phase 1: Registry is always available
    features.append("dataset_registry")

    # Phase 2: Dataset implementations
    features.append("dataset_implementations")

    # Phase 6: Registry-based feature queries
    features.append("phase_6_registry_integration")

    # Standard transforms support
    features.append("standard_transforms")

    return {
        "version": __version__,
        "author": __author__,
        "status": __module_status__,
        "supported_types": get_supported_dataset_types(),  # DYNAMIC: Use function instead of constant
        "handler_architecture": HANDLER_ARCHITECTURE_VERSION,
        "transformation_system": TRANSFORMATION_SYSTEM_VERSION,
        "registry_version": REGISTRY_VERSION,
        "implementations_version": IMPLEMENTATIONS_VERSION,
        "phase_6_integration": PHASE_6_INTEGRATION_VERSION,
        "available_features": features,
        "core_classes": ["miliaDataset", "BaseDataset", "DatasetRegistry"]
        + [f"{name}Dataset" for name in list_all()],  # DYNAMIC: Dataset classes from registry
        "architecture": "Handler-Only (No Backward Compatibility)",
        "exception_integration": "Complete Handler Exception Hierarchy",
        "registered_datasets": list_all(),
        "phase_6_features": {
            "feature_based_queries": True,
            "generalized_insight_extraction": True,
            "generalized_metadata_extraction": True,
            "backward_compatibility": True,
            "zero_modification_extension": True,
        },
        "standard_transforms_features": {
            "standard_transforms_support": True,
            "combined_transforms": True,
            "get_transform_configuration_info": True,
        },
    }


def check_dependencies() -> dict[str, bool]:
    """
    Check availability of optional dependencies.

    Returns:
        dict: Dictionary mapping dependency names to availability status

    Example:
        >>> from milia_pipeline.datasets import check_dependencies
        >>> deps = check_dependencies()
        >>> if deps['pytorch_geometric']:
        ...     print("PyTorch Geometric is available")
        >>> if deps['phase_6_registry_integration']:
        ...     print("Phase 6 registry integration is available")
    """
    dependencies = {}

    # PyTorch Geometric
    try:
        import torch_geometric

        dependencies["pytorch_geometric"] = True
    except ImportError:
        dependencies["pytorch_geometric"] = False

    # Dataset Handlers
    try:
        from milia_pipeline.handlers import create_dataset_handler

        dependencies["dataset_handlers"] = True
    except ImportError:
        dependencies["dataset_handlers"] = False

    # Enhanced Transformation System
    try:
        from milia_pipeline.transformations.graph_transforms import get_graph_transforms

        dependencies["enhanced_transforms"] = True
    except ImportError:
        dependencies["enhanced_transforms"] = False

    # Descriptor System
    try:
        from milia_pipeline.descriptors.descriptor_calculator import DescriptorCalculator

        dependencies["descriptors"] = True
    except ImportError:
        dependencies["descriptors"] = False

    # RDKit
    try:
        import rdkit

        dependencies["rdkit"] = True
    except ImportError:
        dependencies["rdkit"] = False

    # NumPy
    try:
        import numpy

        dependencies["numpy"] = True
    except ImportError:
        dependencies["numpy"] = False

    # Phase 1: Dataset Registry (always available)
    dependencies["dataset_registry"] = True

    # Phase 2: Dataset Implementations (always available after Phase 2)
    dependencies["dataset_implementations"] = True

    # Phase 6: Registry-based feature queries (always available after Phase 6)
    dependencies["phase_6_registry_integration"] = True

    # Standard transforms support (always available)
    dependencies["standard_transforms"] = True

    return dependencies


# ============================================================================
# Phase 1: Plugin Initialization
# ============================================================================


def initialize_plugins(load_external: bool = True) -> int:
    """
    Initialize dataset plugins (placeholder for Phase 8).

    Args:
        load_external: Whether to load external plugins via entry points

    Returns:
        Number of external plugins loaded
    """
    return 0


# ============================================================================
# Module Documentation Update
# ============================================================================


# Update module docstring with runtime information
def _update_module_docstring():
    """Update module __doc__ with runtime dependency information."""
    deps = check_dependencies()
    [name for name, status in deps.items() if status]
    unavailable = [name for name, status in deps.items() if not status]

    if unavailable:
        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"Optional dependencies not available: {', '.join(unavailable)}")


# Update documentation on import
_update_module_docstring()
