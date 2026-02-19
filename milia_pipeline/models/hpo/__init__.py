# Location: milia_pipeline/models/hpo/__init__.py

"""
HPO (Hyperparameter Optimization) Module

Provides comprehensive hyperparameter optimization capabilities for the MILIA
pipeline, including automated hyperparameter tuning, neural architecture search,
transfer learning, and study analysis.

This module is the main entry point for HPO functionality, exporting the primary
classes, configuration dataclasses, convenience functions, and access to
specialized subpackages.

Architecture
------------
The HPO module follows a protocol-based design with pluggable backends:

    HPOManager (orchestrator)
         │
         ├── backends/ ──► OptunaBackend (primary), RayTuneBackend (future)
         │
         ├── callbacks/ ──► OptunaPruningCallback, create_hpo_callback()
         │
         ├── search_spaces/ ──► SearchSpaceBuilder, ParamType
         │
         ├── analysis/ ──► StudyAnalyzer
         │
         ├── transfer/ ──► HPOTransferManager, MetaFeatureExtractor, WarmStartStrategy
         │
         └── nas/ ──► NASManager, GNNArchitectureSpace

Primary Exports
---------------
**Core Classes**:
    HPOManager : Main orchestrator for hyperparameter optimization
    HPOConfig : Master configuration dataclass (frozen, validated)

**Configuration Classes** (from hpo_config.py):
    SearchSpaceConfig : Search space configuration
    PrunerConfig : Pruner configuration for early trial termination
    SamplerConfig : Sampler configuration for hyperparameter suggestion
    StudyConfig : Single-objective study configuration
    MultiObjectiveStudyConfig : Multi-objective study configuration

**Convenience Functions** (from hpo_manager.py):
    is_hpo_enabled : Check if HPO is enabled in configuration
    get_best_params : Get best parameters from completed HPO
    create_hpo_manager : Factory function to create HPOManager
    infer_task_type : Infer task type from dataset/metric for model creation

**Exceptions** (from milia_pipeline.exceptions):
    HPOError : Base exception for HPO-related errors
    HPOConfigurationError : HPO configuration errors
    TrialFailedError : Trial execution failures
    StudyNotFoundError : Study lookup failures
    BackendError : Backend-specific errors
    SearchSpaceError : Search space definition errors
    PruningError : Pruning-related errors

**Backend Access** (from backends subpackage):
    HPOBackendProtocol : Protocol defining backend interface
    OptunaBackend : Primary HPO backend using Optuna
    get_backend : Factory function to get backend by name
    OPTUNA_AVAILABLE : Boolean indicating Optuna availability

**Callback Access** (from callbacks subpackage):
    OptunaPruningCallback : Callback for Optuna trial pruning
    create_hpo_callback : Factory function for backend-specific callbacks
    OPTUNA_AVAILABLE : Boolean indicating Optuna availability

**Search Space Utilities** (from search_spaces subpackage):
    SearchSpaceBuilder : Builder class for constructing search spaces
    ParamType : Enum for parameter types
    SearchSpaceParamConfig : Configuration for single hyperparameter

**Analysis Utilities** (from analysis subpackage):
    StudyAnalyzer : Analyzer for HPO study results
    AnalysisConfig : Configuration for analysis settings
    ImportanceMethod : Enum for parameter importance methods
    ExportFormat : Enum for export formats

**Transfer Learning** (from transfer subpackage):
    HPOTransferManager : Manager for cross-study transfer
    MetaFeatureExtractor : Dataset meta-feature extraction
    WarmStartStrategy : Warm-start strategy implementations
    TransferConfig : Transfer learning configuration

**Neural Architecture Search** (from nas subpackage):
    NASManager : Neural architecture search orchestrator
    GNNArchitectureSpace : GNN architecture search space
    LayerType : Available GNN layer types
    PoolingType : Graph pooling strategies
    NASConfig : NAS configuration

Usage Examples
--------------
Basic HPO via configuration:

    >>> from milia_pipeline.models.hpo import HPOManager, HPOConfig
    >>>
    >>> # Create configuration
    >>> config = HPOConfig(
    ...     enabled=True,
    ...     backend="optuna",
    ...     n_trials=100,
    ... )
    >>>
    >>> # Create manager and run optimization
    >>> manager = HPOManager(config)
    >>> best_params = manager.optimize(
    ...     model_name="GCN",
    ...     dataset=dataset,
    ... )

Using convenience functions:

    >>> from milia_pipeline.models.hpo import (
    ...     create_hpo_manager,
    ...     is_hpo_enabled,
    ...     get_best_params,
    ... )
    >>>
    >>> # Quick setup
    >>> manager = create_hpo_manager(enabled=True, n_trials=50)
    >>>
    >>> # Check if enabled
    >>> if is_hpo_enabled(manager.config):
    ...     best = manager.optimize(model_name="GAT", dataset=dataset)

From YAML configuration:

    >>> from milia_pipeline.models.hpo import HPOManager
    >>>
    >>> # Load from config file
    >>> manager = HPOManager.from_yaml("config.yaml", section="models.hpo")
    >>> best_params = manager.optimize(model_name="GCN", dataset=dataset)

Pattern References
------------------
- Module structure: handlers/__init__.py
- Import pattern: datasets/__init__.py
- Export pattern: callbacks/__init__.py (from milia_pipeline.models.training)
- Blueprint specification: MILIA_HPO_Implementation_Blueprint.md (lines 290-369)

Author: MILIA Team
Version: 1.0.0
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# CORE IMPORTS FROM HPO_CONFIG
# =============================================================================

# =============================================================================
# EXCEPTION IMPORTS FROM CENTRALIZED EXCEPTIONS
# =============================================================================
from milia_pipeline.exceptions import (
    BackendError,
    HPOConfigurationError,
    HPOError,
    PruningError,
    SearchSpaceError,
    StudyNotFoundError,
    TrialFailedError,
)

# =============================================================================
# ANALYSIS SUBPACKAGE IMPORTS
# =============================================================================
from .analysis import (
    # Configuration
    AnalysisConfig,
    ExportFormat,
    # Enums
    ImportanceMethod,
    # Main analyzer
    StudyAnalyzer,
)

# =============================================================================
# BACKEND SUBPACKAGE IMPORTS
# =============================================================================
from .backends import (
    # Availability flags
    OPTUNA_AVAILABLE,
    # Protocol
    HPOBackendProtocol,
    # Backend implementations
    OptunaBackend,
    # Factory function
    get_backend,
)
from .callbacks import (
    # Availability flag (re-exported for convenience)
    OPTUNA_AVAILABLE as CALLBACKS_OPTUNA_AVAILABLE,
)

# =============================================================================
# CALLBACKS SUBPACKAGE IMPORTS
# =============================================================================
from .callbacks import (
    # Primary callback class
    OptunaPruningCallback,
    # Factory function
    create_hpo_callback,
)
from .hpo_config import (
    # Main configuration
    HPOConfig,
    MultiObjectiveStudyConfig,
    OptimizationDirection,
    # Enums
    ParamType,
    PrunerConfig,
    PrunerType,
    SamplerConfig,
    SamplerType,
    # Nested configurations
    SearchSpaceParamConfig,
    StudyConfig,
)

# =============================================================================
# CORE IMPORTS FROM HPO_MANAGER
# =============================================================================
from .hpo_manager import (
    # Main class
    HPOManager,
    _extract_param_categories,
    # Helper functions (for advanced use)
    _flatten_params,
    _run_cross_validation,
    create_hpo_manager,
    get_best_params,
    infer_task_type,
    # Convenience functions
    is_hpo_enabled,
)

# =============================================================================
# NAS (NEURAL ARCHITECTURE SEARCH) SUBPACKAGE IMPORTS
# =============================================================================
from .nas import (
    ActivationType,
    AggregationType,
    GNNArchitectureSpace,
    HeterogeneousGNN,
    # Dataclasses
    LayerConfig,
    # Enums
    LayerType,
    # Configuration
    NASConfig,
    # Main classes
    NASManager,
    PoolingType,
    # Factory functions
    create_gnn_search_space,
    create_nas_manager,
    get_default_gnn_search_space,
)

# =============================================================================
# SEARCH SPACES SUBPACKAGE IMPORTS
# =============================================================================
from .search_spaces import (
    # Parameter types
    ParamType as SearchSpaceParamType,
)
from .search_spaces import (
    # Builder class
    SearchSpaceBuilder,
    # Convenience functions
    build_search_space,
    get_model_search_space,
    validate_search_space,
)
from .search_spaces import (
    SearchSpaceParamConfig as SearchSpaceParam,
)

# =============================================================================
# TRANSFER LEARNING SUBPACKAGE IMPORTS
# =============================================================================
from .transfer import (
    AdaptationMethod,
    # Primary classes
    HPOTransferManager,
    MetaFeatureCategory,
    MetaFeatureConfig,
    MetaFeatureExtractor,
    # Enums
    MetaFeatureMethod,
    # Supporting data classes
    RegisteredStudyInfo,
    # Configuration classes
    TransferConfig,
    TransferredTrial,
    WarmStartConfig,
    WarmStartMethod,
    WarmStartStrategy,
)

# =============================================================================
# PUBLIC API DEFINITION
# =============================================================================

__all__ = [
    # =========================================================================
    # CORE CLASSES
    # =========================================================================
    "HPOManager",
    "HPOConfig",
    # =========================================================================
    # CONFIGURATION CLASSES (from hpo_config.py)
    # =========================================================================
    "SearchSpaceParamConfig",
    "PrunerConfig",
    "SamplerConfig",
    "StudyConfig",
    "MultiObjectiveStudyConfig",
    # =========================================================================
    # CONFIGURATION ENUMS (from hpo_config.py)
    # =========================================================================
    "ParamType",
    "PrunerType",
    "SamplerType",
    "OptimizationDirection",
    # =========================================================================
    # CONVENIENCE FUNCTIONS (from hpo_manager.py)
    # =========================================================================
    "is_hpo_enabled",
    "get_best_params",
    "create_hpo_manager",
    "infer_task_type",
    # Helper functions (for advanced use)
    "_flatten_params",
    "_extract_param_categories",
    "_run_cross_validation",
    # =========================================================================
    # EXCEPTIONS (from milia_pipeline.exceptions)
    # =========================================================================
    "HPOError",
    "HPOConfigurationError",
    "TrialFailedError",
    "StudyNotFoundError",
    "BackendError",
    "SearchSpaceError",
    "PruningError",
    # =========================================================================
    # BACKENDS (from backends subpackage)
    # =========================================================================
    "HPOBackendProtocol",
    "OptunaBackend",
    "get_backend",
    "OPTUNA_AVAILABLE",
    # =========================================================================
    # CALLBACKS (from callbacks subpackage)
    # =========================================================================
    "OptunaPruningCallback",
    "create_hpo_callback",
    # =========================================================================
    # SEARCH SPACES (from search_spaces subpackage)
    # =========================================================================
    "SearchSpaceBuilder",
    "build_search_space",
    "get_model_search_space",
    "validate_search_space",
    # =========================================================================
    # ANALYSIS (from analysis subpackage)
    # =========================================================================
    "StudyAnalyzer",
    "AnalysisConfig",
    "ImportanceMethod",
    "ExportFormat",
    # =========================================================================
    # TRANSFER LEARNING (from transfer subpackage)
    # =========================================================================
    "HPOTransferManager",
    "MetaFeatureExtractor",
    "WarmStartStrategy",
    "TransferConfig",
    "MetaFeatureConfig",
    "WarmStartConfig",
    "RegisteredStudyInfo",
    "TransferredTrial",
    "MetaFeatureMethod",
    "AdaptationMethod",
    "MetaFeatureCategory",
    "WarmStartMethod",
    # =========================================================================
    # NAS (from nas subpackage)
    # =========================================================================
    "NASManager",
    "NASConfig",
    "GNNArchitectureSpace",
    "LayerType",
    "PoolingType",
    "AggregationType",
    "ActivationType",
    "LayerConfig",
    "HeterogeneousGNN",
    "create_gnn_search_space",
    "get_default_gnn_search_space",
    "create_nas_manager",
    # =========================================================================
    # MODULE UTILITIES
    # =========================================================================
    "get_hpo_module_info",
    "check_hpo_dependencies",
]


# =============================================================================
# MODULE METADATA
# =============================================================================

__version__ = "1.0.0"
__author__ = "MILIA Team"


# =============================================================================
# MODULE-LEVEL UTILITY FUNCTIONS
# =============================================================================


def get_hpo_module_info() -> dict[str, Any]:
    """
    Get comprehensive information about the HPO module.

    Returns a dictionary containing version, available components,
    backend availability, and subpackage information.

    Returns
    -------
    dict
        Dictionary containing:
        - version: Module version string
        - author: Module author
        - backends: Available HPO backends
        - optuna_available: Whether Optuna is installed
        - subpackages: List of available subpackages
        - exports: Number of public exports
        - components: Categorized list of components

    Examples
    --------
    >>> from milia_pipeline.models.hpo import get_hpo_module_info
    >>> info = get_hpo_module_info()
    >>> print(f"HPO Module v{info['version']}")
    >>> print(f"Optuna available: {info['optuna_available']}")
    >>> print(f"Backends: {info['backends']}")
    """
    return {
        "version": __version__,
        "author": __author__,
        "module": "milia_pipeline.models.hpo",
        "backends": ["optuna", "ray_tune"],
        "primary_backend": "optuna",
        "optuna_available": OPTUNA_AVAILABLE,
        "subpackages": [
            "backends",
            "callbacks",
            "search_spaces",
            "analysis",
            "transfer",
            "nas",
        ],
        "exports": len(__all__),
        "components": {
            "core": ["HPOManager", "HPOConfig"],
            "configurations": [
                "SearchSpaceParamConfig",
                "PrunerConfig",
                "SamplerConfig",
                "StudyConfig",
                "MultiObjectiveStudyConfig",
            ],
            "convenience_functions": [
                "is_hpo_enabled",
                "get_best_params",
                "create_hpo_manager",
                "infer_task_type",
            ],
            "exceptions": [
                "HPOError",
                "HPOConfigurationError",
                "TrialFailedError",
                "StudyNotFoundError",
                "BackendError",
                "SearchSpaceError",
                "PruningError",
            ],
            "backends": ["HPOBackendProtocol", "OptunaBackend", "get_backend"],
            "callbacks": ["OptunaPruningCallback", "create_hpo_callback"],
            "search_spaces": ["SearchSpaceBuilder", "ParamType"],
            "analysis": ["StudyAnalyzer", "AnalysisConfig"],
            "transfer": ["HPOTransferManager", "MetaFeatureExtractor", "WarmStartStrategy"],
            "nas": ["NASManager", "GNNArchitectureSpace", "LayerType", "PoolingType"],
        },
        "description": (
            "Comprehensive hyperparameter optimization module with support for "
            "automated tuning, neural architecture search, transfer learning, "
            "and multi-objective optimization."
        ),
    }


def check_hpo_dependencies() -> dict[str, Any]:
    """
    Check availability of HPO module dependencies.

    Verifies that required and optional dependencies are available
    for full HPO functionality.

    Returns
    -------
    dict
        Dictionary with dependency availability:
        - optuna: Optuna availability and version
        - ray_tune: Ray Tune availability
        - torch: PyTorch availability and version
        - torch_geometric: PyTorch Geometric availability
        - numpy: NumPy availability and version
        - all_required_available: True if all required dependencies available
        - all_optional_available: True if all optional dependencies available

    Examples
    --------
    >>> from milia_pipeline.models.hpo import check_hpo_dependencies
    >>> deps = check_hpo_dependencies()
    >>> if deps['all_required_available']:
    ...     print("All required HPO dependencies available")
    >>> if not deps['ray_tune']['available']:
    ...     print("Ray Tune not available (optional)")
    """
    deps = {
        "optuna": {"available": False, "version": None},
        "ray_tune": {"available": False, "version": None},
        "torch": {"available": False, "version": None},
        "torch_geometric": {"available": False, "version": None},
        "numpy": {"available": False, "version": None},
        "scikit_learn": {"available": False, "version": None},
    }

    # Check Optuna (required)
    try:
        import optuna

        deps["optuna"]["available"] = True
        deps["optuna"]["version"] = optuna.__version__
    except ImportError:
        pass

    # Check Ray Tune (optional)
    try:
        import ray
        from ray import tune

        deps["ray_tune"]["available"] = True
        deps["ray_tune"]["version"] = ray.__version__
    except ImportError:
        pass

    # Check PyTorch (required)
    try:
        import torch

        deps["torch"]["available"] = True
        deps["torch"]["version"] = torch.__version__
    except ImportError:
        pass

    # Check PyTorch Geometric (required for NAS)
    try:
        import torch_geometric

        deps["torch_geometric"]["available"] = True
        deps["torch_geometric"]["version"] = torch_geometric.__version__
    except ImportError:
        pass

    # Check NumPy (required)
    try:
        import numpy

        deps["numpy"]["available"] = True
        deps["numpy"]["version"] = numpy.__version__
    except ImportError:
        pass

    # Check scikit-learn (optional, for CV utilities)
    try:
        import sklearn

        deps["scikit_learn"]["available"] = True
        deps["scikit_learn"]["version"] = sklearn.__version__
    except ImportError:
        pass

    # Summary flags
    deps["all_required_available"] = all(
        [
            deps["optuna"]["available"],
            deps["torch"]["available"],
            deps["numpy"]["available"],
        ]
    )

    deps["all_optional_available"] = all(
        [
            deps["ray_tune"]["available"],
            deps["torch_geometric"]["available"],
            deps["scikit_learn"]["available"],
        ]
    )

    deps["nas_available"] = all(
        [
            deps["torch"]["available"],
            deps["torch_geometric"]["available"],
            deps["optuna"]["available"],
        ]
    )

    return deps


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Alias for SearchSpaceParamConfig (used in search_spaces subpackage)
SearchSpaceConfig = SearchSpaceParamConfig


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.debug(
    f"HPO module initialized (version {__version__}) - "
    f"Exports: {len(__all__)} items, "
    f"Optuna available: {OPTUNA_AVAILABLE}"
)
