# models/registry/__init__
"""
Models Registry Module

Central registry system for model discovery, registration, and management.
Provides thread-safe singleton registry for PyTorch Geometric models
with comprehensive metadata, category-based organization, and plugin support.

**Version 1.1.0 (Phase 2 Migration)**:
- Core API functions now powered by dynamic PyG introspection
- `get_all_model_names()`, `get_model_metadata()`, etc. use runtime discovery
- `ALL_MODELS` is now a lazy-loading dict from `pyg_introspector`
- Legacy category dictionaries (BASIC_GNN_MODELS, etc.) still available
- New exports: `get_introspector()`, `PyGModelIntrospector`

This module follows the milia Pipeline architectural patterns with:
- Thread-safe singleton registry (like descriptors module)
- Dynamic runtime introspection of PyG models
- Minimal public interface for end users
- Advanced component access for power users
- Factory-based creation patterns
- Comprehensive error handling

Author: milia Team
Version: 1.1.0

Public API Usage:
    >>> from milia_pipeline.models.registry import get_model, list_models
    >>>
    >>> # Get a model class
    >>> GCN = get_model("GCN")
    >>> model = GCN(in_channels=10, out_channels=5, hidden_channels=64)
    >>>
    >>> # List available models
    >>> all_models = list_models()
    >>> basic_gnn = list_models(category=ModelCategory.BASIC_GNN)
    >>> regression_models = list_models(task_type="graph_regression")
    >>>
    >>> # Get model information
    >>> info = get_model_info("GAT")
    >>> print(info['description'])
    >>> print(info['supported_tasks'])
    >>>
    >>> # Check model availability
    >>> if has_model("GraphSAGE"):
    ...     model = get_model("GraphSAGE")
    >>>
    >>> # Register custom/plugin model
    >>> register_model(
    ...     name="CustomGNN",
    ...     model_class=MyCustomGNN,
    ...     metadata=custom_metadata,
    ...     plugin_name="my_plugin"
    ... )

Advanced Usage:
    >>> from milia_pipeline.models.registry import ModelRegistry, ModelCategory
    >>>
    >>> # Access registry instance directly
    >>> registry = ModelRegistry.get_instance()
    >>>
    >>> # Get detailed registration info
    >>> registration = registry.get_registration("GCN")
    >>> print(registration.model_class)
    >>> print(registration.metadata.hyperparameters)
    >>>
    >>> # Get registry statistics
    >>> stats = registry.get_registry_statistics()
    >>> print(f"Total models: {stats['total_models']}")
    >>> print(f"By category: {stats['by_category']}")
    >>>
    >>> # Validate model compatibility
    >>> is_valid = registry.validate_model_compatibility(
    ...     model_name="GCN",
    ...     task_type="graph_classification",
    ...     requires_edge_weights=False
    ... )

Category-Based Access:
    >>> from milia_pipeline.models.registry import (
    ...     ModelCategory,
    ...     get_models_by_category,
    ...     get_models_by_task,
    ...     get_models_by_tag
    ... )
    >>>
    >>> # Get models by category
    >>> attention_models = get_models_by_category(ModelCategory.ATTENTION)
    >>>
    >>> # Get models by task type
    >>> node_classifiers = get_models_by_task("node_classification")
    >>>
    >>> # Get models by tag
    >>> temporal_models = get_models_by_tag("temporal")

Metadata Access:
    >>> from milia_pipeline.models.registry import (
    ...     get_model_metadata,
    ...     get_all_model_names,
    ...     search_models
    ... )
    >>>
    >>> # Get metadata for a specific model
    >>> metadata = get_model_metadata("GCN")
    >>> print(metadata.paper_url)
    >>> print(metadata.hyperparameters)
    >>>
    >>> # Get all registered model names
    >>> all_names = get_all_model_names()
    >>>
    >>> # Search models by keyword
    >>> graph_models = search_models("graph", search_in=["name", "description"])
    >>> attention_models = search_models("attention", search_in=["tags"])

Registry Management:
    >>> from milia_pipeline.models.registry import registry
    >>>
    >>> # Check if model exists
    >>> if "GCN" in registry:
    ...     print("GCN is available")
    >>>
    >>> # Get number of registered models
    >>> print(f"Total models: {len(registry)}")
    >>>
    >>> # Get availability report
    >>> report = registry.get_availability_report()
    >>> print(f"Success rate: {report['success_rate']:.1f}%")
    >>> print(f"Failed models: {len(report['failed_models'])}")

Key Features:
    - 120+ PyTorch Geometric models across 12 categories
    - Thread-safe singleton registry pattern
    - Auto-discovery of PyG models on initialization
    - Comprehensive metadata including:
        * Paper references and URLs
        * Supported task types
        * Hyperparameter schemas
        * Requirements (edge features, weights, etc.)
        * Version compatibility
    - Category-based organization:
        * BASIC_GNN: Core GNN architectures (GCN, GAT, GraphSAGE, etc.)
        * CONVOLUTIONAL: Specialized convolutions (52+ models)
        * ATTENTION: Attention mechanisms (8 models)
        * POOLING: Graph pooling operations (10 models)
        * AGGREGATION: Aggregation functions (8 models)
        * ENCODER: Unsupervised learning (6 models)
        * AUTOENCODER: VAE and adversarial (4 models)
        * TRANSFORMER: Transformer architectures (5 models)
        * TEMPORAL: Dynamic graph models (6 models)
        * META_LEARNING: Meta-learning approaches (3 models)
        * EXPLAINABILITY: GNN explanation (4 models)
        * UTILITY: Helper models (8 models)
    - Plugin support for custom models
    - Graceful handling of version incompatibilities
    - Comprehensive validation and error handling
    - Model search and filtering capabilities
    - Detailed availability reporting

Thread Safety:
    All registry operations are thread-safe using RLock for reentrant locking.
    Multiple threads can safely:
    - Query model availability
    - Get model classes
    - Register new models
    - Access metadata

    The singleton pattern ensures consistent state across the application.

Performance:
    - Lazy model imports (models loaded on-demand)
    - Efficient category-based indexing
    - Fast lookup operations
    - Minimal memory footprint until models are used

Architecture:
    - Singleton pattern for global registry instance
    - Factory pattern for model creation
    - Metadata-driven model specifications
    - Plugin-based extensibility
    - Graceful degradation for missing models

Error Handling:
    - ModelError: Base exception for model-related errors
    - ModelValidationError: Model validation failures
    - Graceful handling of:
        * Missing models in PyG version
        * Import errors
        * Invalid registrations
        * Version incompatibilities
"""

# =============================================================================
# IMPORTS - Dynamic PyG Introspection (Phase 2 Migration)
# =============================================================================

# Core API functions and types now come from dynamic introspector
# This replaces static model_categories.py with runtime introspection
from .pyg_introspector import (
    # Dynamic ALL_MODELS dict (lazy-loading)
    ALL_MODELS,
    # Enums (with fallback if model_categories not available)
    ModelCategory,
    # Dataclasses (DynamicModelMetadata aliased as ModelMetadata)
    ModelMetadata,
    PyGModelIntrospector,
    # Core API Functions (backward-compatible drop-in replacements)
    get_all_model_names,
    get_category_statistics,
    # New Dynamic Introspection API
    get_introspector,
    get_model_metadata,
    get_models_by_category,
    get_models_by_tag,
    get_models_by_task,
    search_models,
)

# =============================================================================
# IMPORTS - Legacy Model Categories (for backward compatibility)
# =============================================================================

# Category-specific model dictionaries are kept from model_categories.py
# These will be deprecated in Phase 7 after full validation
# NOTE: These are static dictionaries and will NOT include dynamically discovered models
try:
    from .model_categories import (
        AGGREGATION_MODELS,
        ATTENTION_MODELS,
        AUTOENCODER_MODELS,
        # Model Dictionaries (for advanced usage - LEGACY)
        BASIC_GNN_MODELS,
        CONVOLUTIONAL_MODELS,
        ENCODER_MODELS,
        EXPLAINABILITY_MODELS,
        META_LEARNING_MODELS,
        MODELS_BY_CATEGORY,
        POOLING_MODELS,
        TEMPORAL_MODELS,
        TRANSFORMER_MODELS,
        UTILITY_MODELS,
    )

    _LEGACY_CATEGORIES_AVAILABLE = True
except ImportError:
    # If model_categories.py is removed, provide empty fallbacks
    BASIC_GNN_MODELS = {}
    CONVOLUTIONAL_MODELS = {}
    ATTENTION_MODELS = {}
    POOLING_MODELS = {}
    AGGREGATION_MODELS = {}
    ENCODER_MODELS = {}
    AUTOENCODER_MODELS = {}
    TRANSFORMER_MODELS = {}
    TEMPORAL_MODELS = {}
    META_LEARNING_MODELS = {}
    EXPLAINABILITY_MODELS = {}
    UTILITY_MODELS = {}
    MODELS_BY_CATEGORY = {}
    _LEGACY_CATEGORIES_AVAILABLE = False

# =============================================================================
# IMPORTS - Model Registry
# =============================================================================

from .model_registry import (
    # Exceptions (re-exported for convenience)
    ModelError,
    # Dataclasses
    ModelRegistration,
    # Main Registry Class
    ModelRegistry,
    ModelValidationError,
    # Convenience Functions (Primary Public API)
    get_model,
    get_model_info,
    has_model,
    list_models,
    # Global Registry Instance
    registry,
)

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # -------------------------------------------------------------------------
    # PRIMARY PUBLIC API (Most Common Usage)
    # -------------------------------------------------------------------------
    # Model Access Functions
    "get_model",  # Get model class by name
    "has_model",  # Check if model exists
    "list_models",  # List available models (with filters)
    "get_model_info",  # Get comprehensive model information
    # Model Registration (for plugins/custom models)
    "register_model",  # Register custom model (via registry instance)
    # -------------------------------------------------------------------------
    # METADATA ACCESS (Now powered by dynamic introspection)
    # -------------------------------------------------------------------------
    # Metadata Functions
    "get_model_metadata",  # Get ModelMetadata for a specific model
    "get_all_model_names",  # Get list of all registered model names
    "search_models",  # Search models by keyword
    # Category-Based Access
    "get_models_by_category",  # Get models in a category
    "get_models_by_task",  # Get models supporting a task
    "get_models_by_tag",  # Get models with a tag
    "get_category_statistics",  # Get model count per category
    # -------------------------------------------------------------------------
    # DYNAMIC INTROSPECTION API (New in Phase 2)
    # -------------------------------------------------------------------------
    "get_introspector",  # Get singleton PyGModelIntrospector instance
    "PyGModelIntrospector",  # Main introspector class
    # -------------------------------------------------------------------------
    # CORE CLASSES (Advanced Usage)
    # -------------------------------------------------------------------------
    # Registry
    "ModelRegistry",  # Main registry class (singleton)
    "registry",  # Global registry instance
    # Data Classes
    "ModelMetadata",  # Model metadata container (now DynamicModelMetadata)
    "ModelRegistration",  # Model registration container
    "ModelCategory",  # Model category enum
    # -------------------------------------------------------------------------
    # MODEL DICTIONARIES (Legacy - will be deprecated in Phase 7)
    # -------------------------------------------------------------------------
    # Category-specific model dictionaries (static, from model_categories.py)
    "BASIC_GNN_MODELS",
    "CONVOLUTIONAL_MODELS",
    "ATTENTION_MODELS",
    "POOLING_MODELS",
    "AGGREGATION_MODELS",
    "ENCODER_MODELS",
    "AUTOENCODER_MODELS",
    "TRANSFORMER_MODELS",
    "TEMPORAL_MODELS",
    "META_LEARNING_MODELS",
    "EXPLAINABILITY_MODELS",
    "UTILITY_MODELS",
    # Aggregate dictionaries
    "MODELS_BY_CATEGORY",  # Dict[ModelCategory, Dict[str, ModelMetadata]] (LEGACY)
    "ALL_MODELS",  # Dict[str, ModelMetadata] - now dynamic via introspector
    # -------------------------------------------------------------------------
    # EXCEPTIONS
    # -------------------------------------------------------------------------
    "ModelError",  # Base model exception
    "ModelValidationError",  # Model validation exception
    # -------------------------------------------------------------------------
    # MODULE INFO
    # -------------------------------------------------------------------------
    "get_module_info",  # Get module information
]

# =============================================================================
# CONVENIENCE FUNCTION ALIASES
# =============================================================================


def register_model(
    name: str,
    model_class,
    metadata: ModelMetadata,
    is_builtin: bool = False,
    plugin_name: str = None,
) -> None:
    """
    Register a custom or plugin model.

    This is a convenience wrapper around registry.register_model() for easier
    access to the registration functionality.

    Args:
        name: Model name (must be unique)
        model_class: PyTorch nn.Module class
        metadata: ModelMetadata with specifications
        is_builtin: True if PyG built-in model, False for custom/plugin
        plugin_name: Name of plugin (for plugin models)

    Raises:
        ValueError: If model name already exists
        ModelValidationError: If model or metadata is invalid

    Example:
        >>> from milia_pipeline.models.registry import register_model, ModelMetadata, ModelCategory
        >>>
        >>> # Create metadata
        >>> metadata = ModelMetadata(
        ...     name="MyCustomGNN",
        ...     category=ModelCategory.BASIC_GNN,
        ...     import_path="my_package.MyCustomGNN",
        ...     description="My custom graph neural network",
        ...     supported_tasks=["graph_classification"],
        ...     hyperparameters={
        ...         "hidden_channels": {"type": "integer", "default": 64},
        ...         "num_layers": {"type": "integer", "default": 3}
        ...     }
        ... )
        >>>
        >>> # Register the model
        >>> register_model(
        ...     name="MyCustomGNN",
        ...     model_class=MyCustomGNN,
        ...     metadata=metadata,
        ...     plugin_name="my_plugin"
        ... )
        >>>
        >>> # Now it's available via get_model
        >>> model_class = get_model("MyCustomGNN")
    """
    registry.register_model(
        name=name,
        model_class=model_class,
        metadata=metadata,
        is_builtin=is_builtin,
        plugin_name=plugin_name,
    )


# =============================================================================
# VERSION AND MODULE INFO
# =============================================================================

__version__ = "1.1.0"
__author__ = "milia Team"
__description__ = "Model registry system for PyTorch Geometric models with dynamic introspection and comprehensive metadata"

# =============================================================================
# MODULE-LEVEL DOCUMENTATION
# =============================================================================

# Add module-level attributes for introspection
_module_info = {
    "version": __version__,
    "author": __author__,
    "description": __description__,
    "total_categories": len(ModelCategory),
    "registry_pattern": "thread-safe singleton",
    "introspection_mode": "dynamic",  # Phase 2: Now uses runtime introspection
    "legacy_categories_available": _LEGACY_CATEGORIES_AVAILABLE,
    "supported_features": [
        "auto-discovery",
        "dynamic-introspection",  # NEW: Phase 2
        "runtime-signature-analysis",  # NEW: Phase 2
        "plugin-support",
        "metadata-driven",
        "thread-safe",
        "category-based-organization",
        "task-based-filtering",
        "tag-based-search",
        "comprehensive-validation",
        "graceful-degradation",
    ],
}


def get_module_info():
    """
    Get information about the models.registry module.

    Returns:
        Dictionary with module information including:
        - version: Module version
        - author: Module author
        - description: Module description
        - total_categories: Number of model categories
        - registry_pattern: Registry design pattern used
        - supported_features: List of module features
        - registry_stats: Current registry statistics

    Example:
        >>> from milia_pipeline.models.registry import get_module_info
        >>> info = get_module_info()
        >>> print(f"Version: {info['version']}")
        >>> print(f"Total models: {info['registry_stats']['total_models']}")
    """
    info = _module_info.copy()
    info["registry_stats"] = registry.get_statistics()
    return info


# =============================================================================
# INITIALIZATION
# =============================================================================

# The registry is automatically initialized when the module is imported
# due to the singleton pattern in ModelRegistry.__init__
# This ensures that PyG models are auto-discovered on first import

# Log initialization (if needed for debugging)
import logging

_logger = logging.getLogger(__name__)
_logger.debug(
    f"Models registry module initialized (v{__version__}). "
    f"Registry contains {len(registry)} models."
)

# =============================================================================
# BACKWARD COMPATIBILITY ALIASES (if needed)
# =============================================================================

# Provide backward compatibility aliases if there are any legacy names
# (Currently not needed, but keeping this section for future use)

# Example:
# get_available_models = list_models  # Legacy alias
# model_exists = has_model            # Legacy alias
