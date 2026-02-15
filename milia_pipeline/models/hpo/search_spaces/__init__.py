# search_spaces/__init__.py - Search Space Module Initialization

"""
Search Space Module for HPO Search Space Configuration.

This module provides utilities for building, validating, and manipulating
hyperparameter search spaces for HPO optimization.

Location: milia_pipeline/models/hpo/search_spaces/__init__.py
Pattern: Follows module initialization patterns from datasets/__init__.py
Dependencies: param_types.py, search_space_builder.py

Exports:
    Classes:
        ParamType: Enum defining supported parameter types for search spaces
        SearchSpaceParamConfig: Frozen dataclass for single hyperparameter configuration
        SearchSpaceBuilder: Builder class for constructing hyperparameter search spaces

    Functions:
        build_search_space: Create a new SearchSpaceBuilder instance
        get_model_search_space: Get predefined search space for a model
        validate_search_space: Validate a search space configuration

Example:
    >>> from milia_pipeline.models.hpo.search_spaces import (
    ...     ParamType,
    ...     SearchSpaceParamConfig,
    ...     SearchSpaceBuilder,
    ...     build_search_space,
    ...     get_model_search_space,
    ... )
    >>>
    >>> # Using ParamType and SearchSpaceParamConfig directly
    >>> lr_config = SearchSpaceParamConfig(
    ...     type=ParamType.LOGUNIFORM,
    ...     low=1e-5,
    ...     high=1e-2,
    ... )
    >>>
    >>> # Using SearchSpaceBuilder fluent API
    >>> search_space = (
    ...     SearchSpaceBuilder()
    ...     .add_int("hidden_channels", 32, 256, step=32)
    ...     .add_loguniform("lr", 1e-5, 1e-2, category="optimizer")
    ...     .add_categorical("activation", ["relu", "gelu", "elu"])
    ...     .build()
    ... )
    >>>
    >>> # Using convenience function
    >>> space = build_search_space().add_int("layers", 2, 6).build()
    >>>
    >>> # Using predefined model search space
    >>> gat_space = get_model_search_space("GAT")

Author: Milia Team
Version: 1.0.0
"""

import logging

# =============================================================================
# IMPORTS FROM SUBMODULES
# =============================================================================
# Parameter type definitions
from .param_types import (
    ParamType,
    SearchSpaceParamConfig,
)

# Search space builder and utilities
from .search_space_builder import (
    SearchSpaceBuilder,
    build_search_space,
    get_model_search_space,
    validate_search_space,
)

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Parameter types
    "ParamType",
    "SearchSpaceParamConfig",
    # Builder class
    "SearchSpaceBuilder",
    # Convenience functions
    "build_search_space",
    "get_model_search_space",
    "validate_search_space",
]


# =============================================================================
# MODULE METADATA
# =============================================================================

__version__ = "1.0.0"
__author__ = "Milia Team"


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger = logging.getLogger(__name__)

logger.debug(
    f"search_spaces module initialized - "
    f"Exports: {len(__all__)} items, "
    f"Available models: {SearchSpaceBuilder.list_available_models()}"
)
