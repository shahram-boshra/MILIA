"""
HPO Transfer Learning Module

This module provides components for hyperparameter optimization transfer learning,
enabling knowledge transfer between related HPO studies for faster convergence
and improved optimization performance on new tasks.

Transfer Learning Use Cases:
1. Transfer from small dataset to large dataset
2. Transfer between related molecular properties (e.g., DFT to experimental)
3. Few-shot optimization on new tasks using prior experience
4. Cross-domain adaptation (e.g., different molecule families)

Module Components:
- HPOTransferManager: Orchestrates cross-study hyperparameter transfer
- MetaFeatureExtractor: Extracts dataset characteristics for similarity
- WarmStartStrategy: Implements trial transfer mechanisms

Configuration Classes:
- TransferConfig: Configuration for HPO transfer learning
- MetaFeatureConfig: Configuration for meta-feature extraction
- WarmStartConfig: Configuration for warm-start strategies

Supporting Classes:
- RegisteredStudyInfo: Metadata for registered studies
- TransferredTrial: Information about trials prepared for transfer

Enums:
- MetaFeatureMethod: Methods for computing meta-features
- AdaptationMethod: Methods for adapting transferred hyperparameters
- MetaFeatureCategory: Categories of meta-features to extract
- WarmStartMethod: Available warm-start methods for HPO transfer

Research Basis:
- Meta-learning for hyperparameter optimization
- Dataset similarity measures using meta-features
- Warm-starting Bayesian optimization
- Transfer learning for AutoML

Author: Milia Team
Version: 1.0.0

Pattern References:
- Module structure: hpo/backends/__init__.py
- Import pattern: hpo/callbacks/__init__.py
- Blueprint specification: MILIA_HPO_Implementation_Blueprint.md (lines 3899-3906)

Usage Examples:
    >>> # Basic usage with default configurations
    >>> from milia_pipeline.models.hpo.transfer import (
    ...     HPOTransferManager,
    ...     MetaFeatureExtractor,
    ...     WarmStartStrategy,
    ... )

    >>> # Initialize transfer manager
    >>> manager = HPOTransferManager()

    >>> # Extract meta-features from a dataset
    >>> features = MetaFeatureExtractor.extract(dataset)

    >>> # Create warm-start trials from a completed study
    >>> trials = WarmStartStrategy.create_from_best_trials(study, n_trials=10)

    >>> # Advanced usage with custom configurations
    >>> from milia_pipeline.models.hpo.transfer import (
    ...     TransferConfig,
    ...     MetaFeatureConfig,
    ...     WarmStartConfig,
    ...     MetaFeatureMethod,
    ...     AdaptationMethod,
    ...     WarmStartMethod,
    ... )

    >>> # Configure transfer with strict similarity
    >>> config = TransferConfig(
    ...     n_warm_start_trials=15,
    ...     similarity_threshold=0.8,
    ...     meta_feature_method=MetaFeatureMethod.STATISTICAL,
    ...     adaptation_method=AdaptationMethod.WEIGHTED,
    ... )
    >>> manager = HPOTransferManager(config)

    >>> # Register a completed study
    >>> info = manager.register_study(
    ...     study_name="gcn_qm9",
    ...     study=completed_study,
    ...     dataset=qm9_dataset,
    ...     model_name="GCN"
    ... )

    >>> # Find similar studies for a new dataset
    >>> similar_studies = manager.find_similar_studies(new_dataset, top_k=3)

    >>> # Warm-start a new study with transferred knowledge
    >>> n_transferred = manager.warm_start_study(
    ...     target_study=new_study,
    ...     source_studies=similar_studies
    ... )
"""

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)


# =============================================================================
# IMPORTS FROM TRANSFER_MANAGER MODULE
# =============================================================================

# =============================================================================
# IMPORTS FROM META_FEATURES MODULE
# =============================================================================
from .meta_features import (
    MetaFeatureCategory,
    MetaFeatureConfig,
    MetaFeatureExtractor,
)
from .transfer_manager import (
    AdaptationMethod,
    HPOTransferManager,
    MetaFeatureMethod,
    RegisteredStudyInfo,
    TransferConfig,
)

# =============================================================================
# IMPORTS FROM WARM_START MODULE
# =============================================================================
from .warm_start import (
    TransferredTrial,
    WarmStartConfig,
    WarmStartMethod,
    WarmStartStrategy,
)

# =============================================================================
# PUBLIC API DEFINITION
# =============================================================================

__all__ = [
    # Primary classes (Blueprint specification lines 3899-3906)
    "HPOTransferManager",
    "MetaFeatureExtractor",
    "WarmStartStrategy",
    # Configuration classes
    "TransferConfig",
    "MetaFeatureConfig",
    "WarmStartConfig",
    # Supporting data classes
    "RegisteredStudyInfo",
    "TransferredTrial",
    # Enums for configuration
    "MetaFeatureMethod",
    "AdaptationMethod",
    "MetaFeatureCategory",
    "WarmStartMethod",
]


# =============================================================================
# MODULE VERSION AND METADATA
# =============================================================================

__version__ = "1.0.0"
__author__ = "Milia Team"


# =============================================================================
# MODULE-LEVEL UTILITY FUNCTIONS
# =============================================================================


def get_module_info() -> dict:
    """
    Get information about the transfer learning module.

    Returns:
        Dictionary containing module metadata and component availability.

    Example:
        >>> info = get_module_info()
        >>> print(f"Version: {info['version']}")
        >>> print(f"Components: {info['components']}")
    """
    return {
        "version": __version__,
        "author": __author__,
        "module": "milia_pipeline.models.hpo.transfer",
        "components": {
            "managers": ["HPOTransferManager"],
            "extractors": ["MetaFeatureExtractor"],
            "strategies": ["WarmStartStrategy"],
            "configs": ["TransferConfig", "MetaFeatureConfig", "WarmStartConfig"],
            "data_classes": ["RegisteredStudyInfo", "TransferredTrial"],
            "enums": [
                "MetaFeatureMethod",
                "AdaptationMethod",
                "MetaFeatureCategory",
                "WarmStartMethod",
            ],
        },
        "description": "HPO Transfer Learning components for cross-study knowledge transfer",
    }


def check_dependencies() -> dict:
    """
    Check availability of optional dependencies for transfer learning.

    Returns:
        Dictionary mapping dependency names to availability status.

    Example:
        >>> deps = check_dependencies()
        >>> if deps['optuna']:
        ...     print("Optuna is available for HPO transfer")
    """
    dependencies = {}

    try:
        import optuna

        dependencies["optuna"] = True
        dependencies["optuna_version"] = optuna.__version__
    except ImportError:
        dependencies["optuna"] = False
        dependencies["optuna_version"] = None

    try:
        import numpy

        dependencies["numpy"] = True
        dependencies["numpy_version"] = numpy.__version__
    except ImportError:
        dependencies["numpy"] = False
        dependencies["numpy_version"] = None

    try:
        import torch

        dependencies["torch"] = True
        dependencies["torch_version"] = torch.__version__
    except ImportError:
        dependencies["torch"] = False
        dependencies["torch_version"] = None

    try:
        from torch_geometric.utils import degree

        dependencies["torch_geometric"] = True
    except ImportError:
        dependencies["torch_geometric"] = False

    return dependencies


logger.debug(f"HPO Transfer module initialized (version {__version__})")
