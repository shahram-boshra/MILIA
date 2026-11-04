# models/plugins/__init__
"""
Deprecated alias of :mod:`milia_pipeline.models.model_plugin_system`.

The model plugin loader now lives at ``milia_pipeline.models.model_plugin_system``.
This package re-exports the same objects under the old import paths and will be
removed in MILIA 2.0.0. Importing it emits a ``DeprecationWarning``.

Migration:
    from milia_pipeline.models.plugins import discover_plugins
    → from milia_pipeline.models.model_plugin_system import discover_plugins

    import milia_pipeline.models.plugins.model_plugin_system
    → import milia_pipeline.models.model_plugin_system
"""

# =============================================================================
# DEPRECATION SHIM
# =============================================================================

import sys
import warnings

from milia_pipeline.models import model_plugin_system as _impl

warnings.warn(
    "milia_pipeline.models.plugins is deprecated and will be removed in MILIA 2.0.0; "
    "import from milia_pipeline.models.model_plugin_system instead.",
    DeprecationWarning,
    stacklevel=2,
)

# The old submodule path resolves to the same module object (module identity), so
# attribute access, assignment and patching through either path reach one module.
sys.modules[__name__ + ".model_plugin_system"] = _impl
model_plugin_system = _impl

# Identity re-binding of the public names (the same objects, not copies).
from milia_pipeline.models.model_plugin_system import (
    ModelDeclaration,
    ModelPluginLoader,
    ModelPluginMetadata,
    PluginDependencyError,
    PluginError,
    PluginSecurityError,
    PluginValidationError,
    discover_and_load_plugins,
    discover_plugins,
    get_all_plugin_models,
    get_plugin_info,
    get_plugin_loader,
    get_plugin_loader_instance,
    get_plugin_summary,
    is_plugin_enabled,
    is_plugin_loaded,
    list_plugins,
    load_plugin,
    safe_discover_plugins,
    safe_load_plugin,
    validate_plugin,
)

# =============================================================================
# VERSION AND METADATA
# =============================================================================

__version__ = "1.1.0"
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
