# milia_pipeline/datasets/__init__.py

"""
Milia Pipeline - Datasets Module
=================================

This module provides PyTorch Geometric compatible dataset implementations for the Milia pipeline,
including comprehensive molecular dataset handling with support for DFT, DMC, and wavefunction data.

Main Components
--------------
- MiliaDataset: Primary dataset class for Milia molecular data
- Factory functions: Convenience functions for dataset creation
- Dataset utilities: Helper functions for dataset operations

Features
--------
- PyTorch Geometric InMemoryDataset integration
- Multi-format support (DFT, DMC, Wavefunction)
- Handler-based processing with comprehensive error handling
- Enhanced transformation system with experimental setup support
- Dynamic transform discovery and validation
- Descriptor system integration
- DMC uncertainty handling
- Chunked processing for large datasets
- Configurable filtering and validation
- Phase 6: Registry-based feature queries for dynamic dataset processing
- Phase 8: Entry point plugin loading for external datasets

Phase 6 Registry Integration
---------------------------
The datasets module now supports dynamic, registry-based dataset processing:
- Feature-based queries replace hardcoded dataset type checks
- New dataset types automatically get appropriate processing based on features
- Generalized insight extraction methods (uncertainty, vibrational, orbital)
- Generalized metadata extraction methods
- Full backward compatibility with legacy code paths
- Zero modifications required to add new dataset types with appropriate features

Phase 8 Plugin Support
----------------------
External packages can register dataset types via Python entry points:
- Entry point group: 'milia.datasets'
- No MILIA source code modification required
- Plugin classes must be BaseDataset subclasses
- Use initialize_plugins() to load external plugins

Example external package pyproject.toml:
    [project.entry-points."milia.datasets"]
    qm9 = "qm9_plugin.dataset:QM9Dataset"

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

Check registry integration status (Phase 6):
    >>> dataset = miliaDataset(root='./data')
    >>> status = dataset.get_registry_integration_status()
    >>> print(f"Registry available: {status['registry_available']}")
    >>> print(f"Phase 6 complete: {status['phase_6_complete']}")

Load external plugins (Phase 8):
    >>> from milia_pipeline.datasets import initialize_plugins
    >>> count = initialize_plugins(load_external=True)
    >>> print(f"Loaded {count} external plugins")

Architecture
-----------
The datasets module implements a handler-only architecture:
- Direct handler creation via create_dataset_handler()
- Comprehensive exception handling with handler-specific errors
- Graceful degradation when handlers unavailable
- Enhanced validation without legacy fallback mechanisms
- Phase 6: Registry-based feature queries for dynamic processing
- Phase 8: Entry point plugin system for external datasets

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
- plugins: Entry point plugin loading (Phase 8)

Version Information
------------------
Module Version: 1.4.0
Handler Architecture: Handler-Only (No Backward Compatibility)
Transformation System: Enhanced with Experimental Setup Support
Exception Integration: Complete Handler Exception Hierarchy
Phase 1 Registry: Dataset Registry Infrastructure
Phase 2 Datasets: DFT, DMC, Wavefunction Implementations
Phase 6 Integration: Registry-Based Feature Queries in milia_dataset.py
Phase 8 Plugins: Entry Point Plugin Loading

Notes
-----
- This module uses ONLY the handler-based system with no fallback to legacy mechanisms
- All handler operations are direct with proper error recovery
- The module maintains full backward compatibility for user-facing APIs
- Enhanced transformation system supports systematic ML/DL research workflows
- Phase 6 enables zero-modification addition of new dataset types
- Phase 8 enables external plugins via standard Python entry points

See Also
--------
- milia_dataset.py: Complete implementation details
- plugins.py: Entry point plugin loading implementation
- config.yaml: Configuration reference
- Milia_Pipeline_Project_Structure.md: Project architecture

Author: milia Pipeline Development Team
License: As per project license
"""

from typing import Optional, Dict, Any, List

# ============================================================================
# Core Dataset Implementation
# ============================================================================

from .milia_dataset import miliaDataset

# ============================================================================
# PHASE 1 ADDITIONS - Dataset Registry Infrastructure
# ============================================================================

from milia_pipeline.datasets.base import (
    BaseDataset,
    DatasetMetadata,
    DatasetSchema,
    DatasetFeatures,
)
from milia_pipeline.datasets.registry import (
    DatasetRegistry,
    get_default_registry,
    register,
    get,
    list_all,
    is_registered,
)
from milia_pipeline.datasets.protocols import (
    DatasetHandlerProtocol,
    DatasetConverterProtocol,
    DatasetValidatorProtocol,
)
from milia_pipeline.exceptions import (
    DatasetRegistrationError,
    DatasetNotFoundError,
)

# ============================================================================
# PHASE 2 ADDITIONS - Dataset Implementations (triggers @register decorators)
# ============================================================================

from milia_pipeline.datasets.implementations import (
    DFTDataset,
    DMCDataset,
    WavefunctionDataset,
)

# ============================================================================
# PHASE 8 ADDITIONS - Plugin Support
# ============================================================================

from milia_pipeline.datasets.plugins import (
    load_dataset_plugins,
    discover_and_load_plugins,
    get_plugin_info,
    list_available_plugins,
    ENTRY_POINT_GROUP,
)

# ============================================================================
# Version and Metadata
# ============================================================================

__version__ = "1.4.0"
__author__ = "milia Pipeline Development Team"
__module_status__ = "Production Ready - Handler-Only Architecture with Phase 8 Plugin Support"

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
    
    # Phase 8: Plugin Initialization
    "initialize_plugins",
    
    # Phase 8: Plugin Functions
    "load_dataset_plugins",
    "discover_and_load_plugins",
    "get_plugin_info",
    "list_available_plugins",
    "ENTRY_POINT_GROUP",
    
    # Phase 2: Dataset Implementations
    "DFTDataset",
    "DMCDataset",
    "WavefunctionDataset",
    
    # Module Information Functions
    "get_module_info",
    "get_supported_dataset_types",
    "check_dependencies",
]

# ============================================================================
# Module-Level Documentation Variables
# ============================================================================

SUPPORTED_DATASET_TYPES = ["DFT", "DMC", "Wavefunction"]
"""List of supported dataset types in milia pipeline."""

HANDLER_ARCHITECTURE_VERSION = "2.0"
"""Current handler architecture version (Handler-Only)."""

TRANSFORMATION_SYSTEM_VERSION = "2.0"
"""Enhanced transformation system with experimental setup support."""

REGISTRY_VERSION = "1.0"
"""Dataset registry infrastructure version (Phase 1)."""

IMPLEMENTATIONS_VERSION = "1.0"
"""Dataset implementations version (Phase 2)."""

PHASE_6_INTEGRATION_VERSION = "6.0.0"
"""Phase 6 registry-based feature query integration version."""

PHASE_8_PLUGIN_VERSION = "8.0.0"
"""Phase 8 entry point plugin loading version."""

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
    logger.debug(f"Phase 8 Plugin Support: {PHASE_8_PLUGIN_VERSION}")
    
    # Check for optional dependencies
    try:
        import torch_geometric
        logger.debug("PyTorch Geometric available")
    except ImportError:
        logger.warning("PyTorch Geometric not available - dataset functionality will be limited")
    
    try:
        from milia_pipeline.handlers.dataset_handlers import create_dataset_handler
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
    
    # Phase 8: Log plugin system status
    available_plugins = list_available_plugins()
    if available_plugins:
        logger.debug(f"Phase 8 plugin system: {len(available_plugins)} external plugin(s) available")
        logger.debug(f"Available plugins: {available_plugins}")
    else:
        logger.debug("Phase 8 plugin system: No external plugins discovered")

# Perform module initialization
_initialize_module()

# ============================================================================
# Module Information Functions
# ============================================================================

def get_module_info() -> Dict[str, Any]:
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
            - phase_8_plugins: Phase 8 plugin system information
    
    Example:
        >>> from milia_pipeline.datasets import get_module_info
        >>> info = get_module_info()
        >>> print(f"Version: {info['version']}")
        >>> print(f"Supported types: {info['supported_types']}")
        >>> print(f"Phase 8 plugins: {info['phase_8_plugins']}")
    """
    # Check feature availability
    features = []
    
    try:
        import torch_geometric
        features.append("pytorch_geometric")
    except ImportError:
        pass
    
    try:
        from milia_pipeline.handlers.dataset_handlers import create_dataset_handler
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
    
    # Phase 8: Plugin support
    features.append("phase_8_plugin_support")
    
    return {
        "version": __version__,
        "author": __author__,
        "status": __module_status__,
        "supported_types": SUPPORTED_DATASET_TYPES,
        "handler_architecture": HANDLER_ARCHITECTURE_VERSION,
        "transformation_system": TRANSFORMATION_SYSTEM_VERSION,
        "registry_version": REGISTRY_VERSION,
        "implementations_version": IMPLEMENTATIONS_VERSION,
        "phase_6_integration": PHASE_6_INTEGRATION_VERSION,
        "phase_8_plugin_version": PHASE_8_PLUGIN_VERSION,
        "available_features": features,
        "core_classes": ["miliaDataset", "BaseDataset", "DatasetRegistry", 
                        "DFTDataset", "DMCDataset", "WavefunctionDataset"],
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
        "phase_8_plugins": get_plugin_info(),
    }


def get_supported_dataset_types() -> List[str]:
    """
    Get list of supported dataset types.
    
    Returns:
        list: List of supported dataset type strings
    
    Example:
        >>> from milia_pipeline.datasets import get_supported_dataset_types
        >>> types = get_supported_dataset_types()
        >>> print(f"Supported types: {types}")
    """
    return SUPPORTED_DATASET_TYPES.copy()


def check_dependencies() -> Dict[str, bool]:
    """
    Check availability of optional dependencies.
    
    Returns:
        dict: Dictionary mapping dependency names to availability status
    
    Example:
        >>> from milia_pipeline.datasets import check_dependencies
        >>> deps = check_dependencies()
        >>> if deps['pytorch_geometric']:
        ...     print("PyTorch Geometric is available")
        >>> if deps['phase_8_plugin_support']:
        ...     print("Phase 8 plugin support is available")
    """
    dependencies = {}
    
    # PyTorch Geometric
    try:
        import torch_geometric
        dependencies['pytorch_geometric'] = True
    except ImportError:
        dependencies['pytorch_geometric'] = False
    
    # Dataset Handlers
    try:
        from milia_pipeline.handlers.dataset_handlers import create_dataset_handler
        dependencies['dataset_handlers'] = True
    except ImportError:
        dependencies['dataset_handlers'] = False
    
    # Enhanced Transformation System
    try:
        from milia_pipeline.transformations.graph_transforms import get_graph_transforms
        dependencies['enhanced_transforms'] = True
    except ImportError:
        dependencies['enhanced_transforms'] = False
    
    # Descriptor System
    try:
        from milia_pipeline.descriptors.descriptor_calculator import DescriptorCalculator
        dependencies['descriptors'] = True
    except ImportError:
        dependencies['descriptors'] = False
    
    # RDKit
    try:
        import rdkit
        dependencies['rdkit'] = True
    except ImportError:
        dependencies['rdkit'] = False
    
    # NumPy
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        dependencies['numpy'] = False
    
    # importlib.metadata (for plugins)
    try:
        from importlib.metadata import entry_points
        dependencies['importlib_metadata'] = True
    except ImportError:
        try:
            from importlib_metadata import entry_points
            dependencies['importlib_metadata'] = True
        except ImportError:
            dependencies['importlib_metadata'] = False
    
    # Phase 1: Dataset Registry (always available)
    dependencies['dataset_registry'] = True
    
    # Phase 2: Dataset Implementations (always available after Phase 2)
    dependencies['dataset_implementations'] = True
    
    # Phase 6: Registry-based feature queries (always available after Phase 6)
    dependencies['phase_6_registry_integration'] = True
    
    # Phase 8: Plugin support (always available after Phase 8)
    dependencies['phase_8_plugin_support'] = True
    
    return dependencies


# ============================================================================
# Phase 8: Plugin Initialization
# ============================================================================

def initialize_plugins(load_external: bool = True) -> int:
    """
    Initialize dataset plugins.
    
    This function loads external dataset plugins via Python entry points.
    External packages can register datasets by declaring the 'milia.datasets'
    entry point group in their pyproject.toml:
    
        [project.entry-points."milia.datasets"]
        qm9 = "qm9_plugin.dataset:QM9Dataset"
    
    Args:
        load_external: Whether to load external plugins via entry points.
                      Set to False to skip plugin loading (e.g., for testing).
        
    Returns:
        Number of external plugins successfully loaded and registered
        
    Example:
        >>> from milia_pipeline.datasets import initialize_plugins
        >>> count = initialize_plugins(load_external=True)
        >>> print(f"Loaded {count} external dataset plugins")
        
        # List all registered datasets (including plugins)
        >>> from milia_pipeline.datasets import list_all
        >>> print(f"All registered datasets: {list_all()}")
        
    Notes:
        - Built-in datasets (DFT, DMC, Wavefunction) are registered automatically
          on module import; this function loads EXTERNAL plugins only.
        - Each plugin class must be a valid BaseDataset subclass.
        - Plugin loading errors are logged but don't raise exceptions.
        - This function is idempotent; calling it multiple times with the
          same installed plugins will not cause duplicate registrations.
        
    See Also:
        - load_dataset_plugins(): Lower-level function with more details
        - list_available_plugins(): List discovered but not loaded plugins
        - get_plugin_info(): Get plugin system diagnostics
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if not load_external:
        logger.debug("External plugin loading disabled")
        return 0
    
    logger.info("Initializing external dataset plugins...")
    
    try:
        count = discover_and_load_plugins()
        
        if count > 0:
            logger.info(f"Successfully loaded {count} external dataset plugin(s)")
            # Update SUPPORTED_DATASET_TYPES to include newly loaded plugins
            registered = list_all()
            logger.debug(f"All registered datasets after plugin loading: {registered}")
        else:
            logger.debug("No external dataset plugins loaded")
        
        return count
        
    except Exception as e:
        # Log error but don't raise - plugin loading should not break the module
        logger.error(f"Error during plugin initialization: {e}")
        logger.debug("Plugin initialization error details:", exc_info=True)
        return 0


# ============================================================================
# Module Documentation Update
# ============================================================================

# Update module docstring with runtime information
def _update_module_docstring():
    """Update module __doc__ with runtime dependency information."""
    deps = check_dependencies()
    available = [name for name, status in deps.items() if status]
    unavailable = [name for name, status in deps.items() if not status]
    
    if unavailable:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Optional dependencies not available: {', '.join(unavailable)}")

# Update documentation on import
_update_module_docstring()
