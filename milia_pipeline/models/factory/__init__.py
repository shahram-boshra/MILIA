# models/factory/__init__.py
"""
Models Factory Module

Factory pattern implementation for model creation with comprehensive validation,
hyperparameter processing, channel inference, and device placement.

**PHASE 7 EXTENSION**: Now supports custom architectures and ensemble models.

This module provides a clean, production-ready API for creating and managing
machine learning models in the milia pipeline.

Public API:
    Classes:
        - ModelFactory: Factory for creating model instances
        - ModelValidator: Validator for hyperparameters and data compatibility
        - GraphLevelModelWrapper: Wrapper for graph-level task handling
        - EdgeLevelModelWrapper: Wrapper for edge-level task handling

    Functions:
        - create_model: Convenience function to create a model
        - get_model_info: Get information about a registered model
        - get_factory: Get singleton ModelFactory instance

Usage:
    Basic model creation (standard models):
        >>> from milia_pipeline.models.factory import create_model
        >>> model = create_model(
        ...     name="GCN",
        ...     hyperparameters={"hidden_channels": 64, "num_layers": 3},
        ...     task_type="graph_regression",
        ...     sample_data=sample_data
        ... )

    Custom architecture creation (Phase 7):
        >>> from milia_pipeline.models.factory import create_model
        >>>
        >>> architecture_config = {
        ...     'layers': [
        ...         {'type': 'GCNConv', 'params': {'out_channels': 64}},
        ...         {'type': 'ReLU', 'params': {}},
        ...         {'type': 'GATConv', 'params': {'out_channels': 32, 'heads': 4}},
        ...         {'type': 'ReLU', 'params': {}},
        ...         {'type': 'global_mean_pool', 'params': {}},
        ...         {'type': 'Linear', 'params': {'out_features': 1}}
        ...     ]
        ... }
        >>>
        >>> model = create_model(
        ...     name="custom",
        ...     hyperparameters={"architecture_config": architecture_config},
        ...     task_type="graph_regression",
        ...     sample_data=sample_data
        ... )

    Ensemble model creation (Phase 7):
        >>> from milia_pipeline.models.factory import create_model
        >>>
        >>> ensemble_config = {
        ...     'models': [
        ...         {
        ...             'name': 'GCN',
        ...             'hyperparameters': {'hidden_channels': 64, 'num_layers': 2},
        ...             'weight': 0.5
        ...         },
        ...         {
        ...             'name': 'GAT',
        ...             'hyperparameters': {'hidden_channels': 64, 'num_layers': 2, 'heads': 4},
        ...             'weight': 0.5
        ...         }
        ...     ],
        ...     'composition': {
        ...         'strategy': 'parallel',
        ...         'fusion': 'weighted'
        ...     }
        ... }
        >>>
        >>> model = create_model(
        ...     name="ensemble",
        ...     hyperparameters={"ensemble_config": ensemble_config},
        ...     task_type="graph_regression",
        ...     sample_data=sample_data
        ... )

    Using ModelFactory directly:
        >>> from milia_pipeline.models.factory import ModelFactory
        >>> factory = ModelFactory()
        >>> model = factory.create_model(
        ...     name="GAT",
        ...     hyperparameters={"hidden_channels": 128, "heads": 4},
        ...     task_type="node_classification",
        ...     sample_data=sample_data,
        ...     device=torch.device("cuda")
        ... )

    Getting model information:
        >>> from milia_pipeline.models.factory import get_model_info
        >>> info = get_model_info("GCN")
        >>> print(info['description'])
        >>> print(info['supported_tasks'])

    Validation:
        >>> from milia_pipeline.models.factory import ModelValidator
        >>> validator = ModelValidator()
        >>> validator.validate_hyperparameters(hparams, schema)
        >>> validator.validate_data_compatibility(data, metadata)

Architecture:
    - Factory Pattern: Centralized model creation with validation
    - Singleton Pattern: Global factory instance via get_factory()
    - Validation: Comprehensive hyperparameter and data validation
    - Channel Inference: Automatic in_channels/out_channels detection
    - Device Management: Flexible device placement (CPU/GPU/TPU)
    - **Phase 7**: Custom architecture and ensemble support

Key Features:
    - Automatic channel inference from sample data
    - Hyperparameter validation against schemas
    - Data compatibility checking
    - Default value application
    - Device placement and multi-device support
    - Comprehensive error handling
    - Parameter counting and model introspection
    - **Phase 7: Custom architecture building via name="custom"**
    - **Phase 7: Ensemble model creation via name="ensemble"**
    - **Phase 7: Graceful degradation when builders module unavailable**

Integration:
    The factory integrates with:
    - model_registry: For model discovery and registration
    - model_categories: For model metadata and categorization
    - Training module: For trainer initialization
    - Deployment module: For model optimization
    - PyTorch Geometric: For graph neural network support
    - **Phase 7: builders module (ArchitectureBuilder, ModelComposer)**

Thread Safety:
    - ModelFactory instances are thread-safe
    - Singleton factory (get_factory) uses thread-safe initialization
    - Model creation can be performed concurrently with separate instances

Performance Considerations:
    - Lazy validation: Validation only performed during creation
    - Caching: Factory maintains internal registry cache
    - Efficient inference: Channel inference optimized for common patterns
    - Device optimization: Automatic device selection available

Backward Compatibility:
    - **100% backward compatible**: All existing code works unchanged
    - Custom and ensemble features only activate when requested
    - Builders module is optional (conditional import)
    - Standard model creation path completely untouched

Author: milia Team
Version: 1.1.0 (Phase 7 Extended)
"""

import logging
from typing import Any

# =============================================================================
# IMPORTS FROM MODEL_FACTORY MODULE
# =============================================================================
from .model_factory import (
    EdgeLevelModelWrapper,
    # Wrapper Classes
    GraphLevelModelWrapper,
    # Main Classes
    ModelFactory,
    ModelValidator,
    # Convenience Functions (Public API)
    create_model,
    get_factory,
    get_model_info,
)

# Target Selection Configuration
from .target_selection_config import (
    SelectionMode,
    TargetSelectionConfig,
)

# =============================================================================
# MODULE METADATA
# =============================================================================

__version__ = "1.1.0"
__author__ = "milia Team"
__all__ = [
    # Main Classes
    "ModelFactory",
    "ModelValidator",
    # Wrapper Classes
    "GraphLevelModelWrapper",
    "EdgeLevelModelWrapper",
    # Public API Functions
    "create_model",
    "get_model_info",
    "get_factory",
    # Target Selection
    "TargetSelectionConfig",
    "SelectionMode",
    # Module Metadata
    "__version__",
]

# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

# Configure module logger
logger = logging.getLogger(__name__)
logger.debug(f"models.factory module initialized (version {__version__}, Phase 7 extended)")

# =============================================================================
# PUBLIC API DOCUMENTATION
# =============================================================================


def _get_module_info() -> dict[str, Any]:
    """
    Get module information for introspection.

    Returns:
        Dictionary with module metadata

    Example:
        >>> from milia_pipeline.models import factory
        >>> info = factory._get_module_info()
        >>> print(info['version'])
        1.1.0
        >>> print(info['phase_7_features'])
        ['custom_architectures', 'ensemble_models']
    """
    return {
        "name": "models.factory",
        "version": __version__,
        "author": __author__,
        "description": "Factory pattern for model creation with validation (Phase 7 extended)",
        "public_api": __all__,
        "classes": [
            "ModelFactory",
            "ModelValidator",
            "GraphLevelModelWrapper",
            "EdgeLevelModelWrapper",
        ],
        "functions": ["create_model", "get_model_info", "get_factory"],
        "capabilities": [
            "Model instantiation with validation",
            "Hyperparameter validation against schemas",
            "Data compatibility checking",
            "Automatic channel inference",
            "Device placement and management",
            "Default value application",
            "Model introspection and information",
            "Custom architecture creation (Phase 7)",
            "Ensemble model creation (Phase 7)",
            "Graceful degradation (Phase 7)",
        ],
        "dependencies": [
            "torch",
            "torch_geometric",
            "model_registry",
            "model_categories",
            "builders (optional, for Phase 7 features)",
        ],
        "thread_safe": True,
        "phase_7_features": [
            "custom_architectures",
            "ensemble_models",
            "architecture_validation",
            "conditional_builders_import",
        ],
        "backward_compatible": True,
    }


# =============================================================================
# CONVENIENCE EXPORTS FOR COMMON USAGE PATTERNS
# =============================================================================

# Note: All main functionality is exported through __all__
# This module serves as the primary interface for model factory operations
# Users should import from this module for a clean, stable API

# Example import patterns:
# 1. from milia_pipeline.models.factory import create_model
# 2. from milia_pipeline.models.factory import ModelFactory
# 3. from milia_pipeline.models.factory import get_factory, get_model_info

# PHASE 7 USAGE EXAMPLES:
#
# Custom Architecture:
#   from milia_pipeline.models.factory import create_model
#   model = create_model("custom", {"architecture_config": {...}}, "graph_regression")
#
# Ensemble:
#   from milia_pipeline.models.factory import create_model
#   model = create_model("ensemble", {"ensemble_config": {...}}, "graph_regression")
