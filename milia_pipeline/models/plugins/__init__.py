# models/plugins/__init__
"""
Model Plugin System for milia Pipeline

This module provides a secure, extensible plugin system for custom models,
enabling dynamic model discovery, validation, and registration without
modifying the core codebase.

Key Features:
    - Plugin discovery from YAML metadata (plugin.yaml)
    - Automatic model registration into ModelRegistry
    - Comprehensive validation (dependencies, security, functional)
    - Version management and compatibility checking
    - Thread-safe registry with enable/disable functionality
    - Singleton pattern for consistent plugin management

Architecture:
    The plugin system follows the established pattern from the transformations
    plugin system and integrates seamlessly with the models module infrastructure.
    It uses a singleton ModelPluginLoader for centralized plugin management and
    provides both class-based and functional APIs for flexibility.

Usage Patterns:

    1. Basic Plugin Discovery:
        >>> from milia_pipeline.models.plugins import discover_plugins
        >>> from pathlib import Path
        >>> plugins = discover_plugins([Path("./plugins/models")])
        >>> print(f"Discovered {len(plugins)} plugins")

    2. Plugin Management:
        >>> from milia_pipeline.models.plugins import (
        ...     load_plugin, list_plugins, get_plugin_info
        ... )
        >>> load_plugin("my_custom_model")
        >>> active_plugins = list_plugins(loaded_only=True)
        >>> info = get_plugin_info("my_custom_model")

    3. Advanced Plugin Control:
        >>> from milia_pipeline.models.plugins import get_plugin_loader
        >>> loader = get_plugin_loader()
        >>> loader.enable_plugin("my_custom_model")
        >>> loader.disable_plugin("my_custom_model")
        >>> loader.validate_plugin("my_custom_model", level="strict")

    4. Plugin Validation:
        >>> from milia_pipeline.models.plugins import validate_plugin
        >>> results = validate_plugin("my_custom_model", level="standard")
        >>> if results["valid"]:
        ...     print("Plugin is valid!")

Plugin Structure:
    Each plugin must contain a plugin.yaml file with metadata:

    plugin_name: my_custom_model
    version: "1.0.0"
    author: "Your Name"
    description: "Custom GNN model for molecular property prediction"
    plugin_type: "model"

    requirements:
      milia_version: ">=1.0.0"
      pyg_version: ">=2.0.0"
      python_version: ">=3.8"

    models:
      - name: "CustomGNN"
        class_name: "CustomGNNModel"
        module_path: "model.py"
        category: "gnn"
        description: "Custom Graph Neural Network"
        supported_tasks: ["regression", "classification"]
        hyperparameters:
          hidden_dim: 128
          num_layers: 3

Thread Safety:
    - ModelPluginLoader uses singleton pattern with thread locks
    - Plugin registration is thread-safe
    - Concurrent plugin lookup and management supported
    - Individual plugin loading operations are synchronized

Performance Considerations:
    - Lazy loading: Plugins loaded on-demand, not at import time
    - Validation caching: Validation results cached per plugin
    - Selective loading: Load only required plugins to reduce overhead
    - Auto-discovery can be deferred for faster initialization

Exception Handling:
    All plugin operations may raise:
    - PluginError: Base exception for plugin-related errors
    - PluginValidationError: When plugin validation fails
    - PluginSecurityError: For security concerns (malicious code patterns)
    - PluginDependencyError: When required dependencies are missing
    - ModelError: For model-specific errors during registration

Integration:
    The plugin system integrates with:
    - ModelRegistry: For automatic model registration
    - ModelMetadata: For model metadata management
    - ModelCategory: For model categorization
    - Config system: For plugin path configuration

Version: 1.0.0
Author: milia Team
"""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
from typing import Any, Dict, List, Optional

# Import all public components from the plugin system implementation
from milia_pipeline.models.plugins.model_plugin_system import (
    ModelDeclaration,
    # Core Classes
    ModelPluginLoader,
    ModelPluginMetadata,
    PluginDependencyError,
    # Exceptions (re-exported for convenience)
    PluginError,
    PluginSecurityError,
    PluginValidationError,
    # Plugin Discovery and Loading
    discover_plugins,
    get_plugin_info,
    # Singleton Access
    get_plugin_loader,
    # Plugin Management
    list_plugins,
    load_plugin,
    validate_plugin,
)

# =============================================================================
# VERSION AND METADATA
# =============================================================================

__version__ = "1.0.0"
__author__ = "milia Team"
__all__ = [
    # Core Classes
    "ModelPluginLoader",
    "ModelPluginMetadata",
    "ModelDeclaration",
    # Singleton Access
    "get_plugin_loader",
    # Plugin Discovery and Loading
    "discover_plugins",
    "load_plugin",
    # Plugin Management
    "list_plugins",
    "get_plugin_info",
    "validate_plugin",
    # Exceptions
    "PluginError",
    "PluginValidationError",
    "PluginSecurityError",
    "PluginDependencyError",
    # Module-level convenience functions
    "get_plugin_loader_instance",
    "discover_and_load_plugins",
    "get_all_plugin_models",
    "is_plugin_loaded",
    "is_plugin_enabled",
    "get_plugin_summary",
    "safe_load_plugin",
    "safe_discover_plugins",
    # Metadata
    "__version__",
    "__author__",
]


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================


def get_plugin_loader_instance() -> ModelPluginLoader:
    """
    Get the singleton plugin loader instance.

    This is an alias for get_plugin_loader() provided for clarity
    in contexts where the singleton pattern needs to be explicit.

    Returns:
        ModelPluginLoader: The singleton plugin loader instance

    Example:
        >>> from milia_pipeline.models.plugins import get_plugin_loader_instance
        >>> loader = get_plugin_loader_instance()
        >>> plugins = loader.list_plugins()
    """
    return get_plugin_loader()


def discover_and_load_plugins(
    paths: list[Path] | None = None,
    auto_validate: bool = True,
    validation_level: str = "standard",
    register_models: bool = True,
) -> dict[str, bool]:
    """
    Discover and load plugins in one operation.

    This is a convenience function that combines discovery and loading,
    useful for initialization scenarios where you want to discover and
    immediately load all available plugins.

    Args:
        paths: List of paths to search for plugins (default: from config)
        auto_validate: Whether to validate plugins automatically during discovery
        validation_level: Validation strictness ('permissive', 'standard', 'strict')
        register_models: Whether to register models in ModelRegistry

    Returns:
        Dictionary mapping plugin names to load success status

    Example:
        >>> from milia_pipeline.models.plugins import discover_and_load_plugins
        >>> from pathlib import Path
        >>> results = discover_and_load_plugins(
        ...     paths=[Path("./plugins/models")],
        ...     validation_level="strict"
        ... )
        >>> print(f"Loaded {sum(results.values())} of {len(results)} plugins")
    """
    loader = get_plugin_loader()

    # Discover plugins
    plugin_names = loader.discover_plugins(
        paths=paths, auto_validate=auto_validate, validation_level=validation_level
    )

    # Load all discovered plugins
    results = {}
    for plugin_name in plugin_names:
        try:
            success = loader.load_plugin(plugin_name, register_models=register_models)
            results[plugin_name] = success
        except Exception:
            results[plugin_name] = False

    return results


def get_all_plugin_models() -> dict[str, list[str]]:
    """
    Get all models provided by all plugins.

    Returns a mapping of plugin names to the list of model names
    they provide. Useful for getting a complete inventory of
    available plugin models.

    Returns:
        Dictionary mapping plugin names to lists of model names

    Example:
        >>> from milia_pipeline.models.plugins import get_all_plugin_models
        >>> models = get_all_plugin_models()
        >>> for plugin, model_list in models.items():
        ...     print(f"{plugin}: {', '.join(model_list)}")
    """
    loader = get_plugin_loader()
    all_plugins = loader.list_plugins()

    result = {}
    for plugin_name in all_plugins:
        try:
            models = loader.get_plugin_models(plugin_name)
            result[plugin_name] = models
        except Exception:
            result[plugin_name] = []

    return result


def is_plugin_loaded(plugin_name: str) -> bool:
    """
    Check if a plugin is loaded.

    Args:
        plugin_name: Name of the plugin to check

    Returns:
        True if plugin is loaded, False otherwise

    Example:
        >>> from milia_pipeline.models.plugins import is_plugin_loaded
        >>> if is_plugin_loaded("my_custom_model"):
        ...     print("Plugin is ready to use")
    """
    loader = get_plugin_loader()
    loaded_plugins = loader.list_plugins(loaded_only=True)
    return plugin_name in loaded_plugins


def is_plugin_enabled(plugin_name: str) -> bool:
    """
    Check if a plugin is enabled.

    Args:
        plugin_name: Name of the plugin to check

    Returns:
        True if plugin is enabled, False otherwise

    Example:
        >>> from milia_pipeline.models.plugins import is_plugin_enabled
        >>> if is_plugin_enabled("my_custom_model"):
        ...     print("Plugin is active")
    """
    loader = get_plugin_loader()
    enabled_plugins = loader.list_plugins(enabled_only=True)
    return plugin_name in enabled_plugins


def get_plugin_summary() -> dict[str, Any]:
    """
    Get a summary of all plugins and their status.

    Returns:
        Dictionary with comprehensive plugin information including:
        - Total number of plugins
        - Number of loaded plugins
        - Number of enabled plugins
        - List of plugin names by status

    Example:
        >>> from milia_pipeline.models.plugins import get_plugin_summary
        >>> summary = get_plugin_summary()
        >>> print(f"Total plugins: {summary['total']}")
        >>> print(f"Loaded: {summary['loaded_count']}")
    """
    loader = get_plugin_loader()

    all_plugins = loader.list_plugins()
    loaded_plugins = loader.list_plugins(loaded_only=True)
    enabled_plugins = loader.list_plugins(enabled_only=True)

    return {
        "total": len(all_plugins),
        "loaded_count": len(loaded_plugins),
        "enabled_count": len(enabled_plugins),
        "all_plugins": all_plugins,
        "loaded_plugins": loaded_plugins,
        "enabled_plugins": enabled_plugins,
        "disabled_plugins": [p for p in all_plugins if p not in enabled_plugins],
        "discovered_but_not_loaded": [p for p in all_plugins if p not in loaded_plugins],
    }


# =============================================================================
# ENHANCED EXCEPTION HANDLING
# =============================================================================


def safe_load_plugin(plugin_name: str, register_models: bool = True) -> tuple[bool, str | None]:
    """
    Safely load a plugin with enhanced error handling.

    This function wraps the load_plugin function with additional
    error handling and returns both success status and error message
    if loading fails.

    Args:
        plugin_name: Name of plugin to load
        register_models: Whether to register models in ModelRegistry

    Returns:
        Tuple of (success: bool, error_message: Optional[str])

    Example:
        >>> from milia_pipeline.models.plugins import safe_load_plugin
        >>> success, error = safe_load_plugin("my_custom_model")
        >>> if not success:
        ...     print(f"Failed to load: {error}")
    """
    try:
        success = load_plugin(plugin_name, register_models=register_models)
        return (success, None)
    except PluginValidationError as e:
        return (False, f"Validation failed: {str(e)}")
    except PluginSecurityError as e:
        return (False, f"Security issue: {str(e)}")
    except PluginDependencyError as e:
        return (False, f"Missing dependencies: {str(e)}")
    except PluginError as e:
        return (False, f"Plugin error: {str(e)}")
    except Exception as e:
        return (False, f"Unexpected error: {str(e)}")


def safe_discover_plugins(
    paths: list[Path] | None = None, auto_validate: bool = True, validation_level: str = "standard"
) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Safely discover plugins with enhanced error handling.

    Returns both successfully discovered plugins and any errors
    encountered during discovery.

    Args:
        paths: List of paths to search for plugins
        auto_validate: Whether to validate plugins automatically
        validation_level: Validation strictness

    Returns:
        Tuple of (discovered_plugins: List[str], errors: List[Tuple[path, error]])

    Example:
        >>> from milia_pipeline.models.plugins import safe_discover_plugins
        >>> from pathlib import Path
        >>> plugins, errors = safe_discover_plugins([Path("./plugins")])
        >>> if errors:
        ...     print(f"Encountered {len(errors)} errors during discovery")
    """
    errors = []
    discovered = []

    try:
        discovered = discover_plugins(
            paths=paths, auto_validate=auto_validate, validation_level=validation_level
        )
    except Exception as e:
        errors.append(("discovery", str(e)))

    return (discovered, errors)


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

# Note: We do NOT auto-discover plugins at import time to keep imports fast
# and allow users to control when plugin discovery happens. Users should
# explicitly call discover_plugins() or discover_and_load_plugins() when ready.

# The singleton loader is created lazily on first access via get_plugin_loader()
