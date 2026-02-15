"""
PyG Augmentation Transforms Plugin

Provides PyG transforms missing from PyG 2.6.1.
These are standard PyG-style transforms for the fallback system.

Production-ready plugin with:
- Lazy loading to avoid circular dependencies
- Full integration with milia PluginRegistry
- Version compatibility validation
- Transform registration with TransformRegistry
- Comprehensive error handling and logging
- Plugin lifecycle management
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

__version__ = "1.0.0"
__author__ = "Shahram Boshra, Ilia Boshra"

# Plugin metadata for milia plugin system
PLUGIN_METADATA = {
    "name": "pyg_augmentation",
    "version": __version__,
    "author": __author__,
    "plugin_type": "pyg_fallback",  # PyG transforms fallback
    "description": "PyG augmentation transforms missing from PyG 2.6.1",
    "license": "MIT",
    "milia_version": ">=4.0.0",
    "pyg_version": ">=2.0.0",
    "python_version": ">=3.8",
}

# Logger for this plugin
logger = logging.getLogger("milia_Main.PluginSystem.pyg_augmentation")

# Transform classes - lazy loaded
_transforms_loaded = False
_transform_classes: dict[str, type] = {}
_transform_names = ["DropEdge", "DropNode", "MaskFeatures", "RandomNodeSample"]

# Plugin registration state
_plugin_registered = False
_plugin_enabled = False
_registration_error: Exception | None = None


def _lazy_import_transforms() -> bool:
    """
    Lazy import of transform classes to avoid circular dependencies.

    Returns:
        bool: True if successful, False otherwise
    """
    global _transforms_loaded, _transform_classes

    if _transforms_loaded:
        return True

    try:
        from .transforms import DropEdge, DropNode, MaskFeatures, RandomNodeSample

        _transform_classes = {
            "DropEdge": DropEdge,
            "DropNode": DropNode,
            "MaskFeatures": MaskFeatures,
            "RandomNodeSample": RandomNodeSample,
        }

        _transforms_loaded = True
        logger.debug("Successfully loaded transform classes")
        return True

    except ImportError as e:
        logger.error(f"Failed to import transform classes: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error loading transforms: {e}")
        return False


def _check_dependencies() -> tuple[bool, list[str]]:
    """
    Check if required dependencies are available.

    Returns:
        tuple: (success: bool, missing_deps: List[str])
    """
    missing = []

    # Check PyTorch Geometric
    try:
        import torch_geometric
        from torch_geometric.transforms import BaseTransform

        # Verify version if possible
        if hasattr(torch_geometric, "__version__"):
            version = torch_geometric.__version__
            logger.debug(f"PyTorch Geometric {version} detected")
    except ImportError:
        missing.append("torch_geometric>=2.0.0")

    # Check PyTorch
    try:
        import torch
    except ImportError:
        missing.append("torch")

    return len(missing) == 0, missing


def _validate_plugin_metadata() -> bool:
    """
    Validate plugin metadata completeness.

    Returns:
        bool: True if metadata is valid
    """
    required_fields = ["name", "version", "author", "plugin_type"]

    for field in required_fields:
        if field not in PLUGIN_METADATA or not PLUGIN_METADATA[field]:
            logger.error(f"Missing required metadata field: {field}")
            return False

    # Validate plugin type
    if PLUGIN_METADATA["plugin_type"] not in ["pyg_fallback", "user_experimental"]:
        logger.error(f"Invalid plugin_type: {PLUGIN_METADATA['plugin_type']}")
        return False

    logger.debug("Plugin metadata validation passed")
    return True


def _register_with_plugin_system() -> bool:
    """
    Register this plugin with milia PluginRegistry.

    Handles:
    - Plugin metadata registration
    - Transform registration with TransformRegistry
    - Version compatibility checks
    - Error recovery

    Returns:
        bool: True if registration successful
    """
    global _plugin_registered, _registration_error

    if _plugin_registered:
        return True

    try:
        # Lazy import plugin system
        from milia_pipeline.transformations.plugin_system import PluginMetadata, PluginRegistry

        # Create plugin metadata object
        plugin_path = Path(__file__).parent

        metadata = PluginMetadata(
            plugin_name=PLUGIN_METADATA["name"],
            version=PLUGIN_METADATA["version"],
            author=PLUGIN_METADATA["author"],
            plugin_type=PLUGIN_METADATA["plugin_type"],
            description=PLUGIN_METADATA.get("description", ""),
            license=PLUGIN_METADATA.get("license", "MIT"),
            milia_version=PLUGIN_METADATA.get("milia_version", ">=4.0.0"),
            pyg_version=PLUGIN_METADATA.get("pyg_version", ">=2.0.0"),
            python_version=PLUGIN_METADATA.get("python_version", ">=3.8"),
            plugin_path=plugin_path,
        )

        # Register with PluginRegistry
        success = PluginRegistry.register_plugin(metadata)

        if success:
            _plugin_registered = True
            logger.info(f"Plugin '{PLUGIN_METADATA['name']}' registered successfully")
            return True
        else:
            logger.warning(f"Plugin '{PLUGIN_METADATA['name']}' registration returned False")
            return False

    except ImportError as e:
        # Plugin system not available - not an error for fallback plugins
        logger.debug(f"PluginRegistry not available (expected for standalone use): {e}")
        _plugin_registered = True  # Mark as registered to avoid repeated attempts
        return True

    except Exception as e:
        _registration_error = e
        logger.error(f"Failed to register plugin: {e}", exc_info=True)
        return False


def _register_transforms() -> bool:
    """
    Register individual transforms with TransformRegistry.

    Returns:
        bool: True if all transforms registered successfully
    """
    if not _lazy_import_transforms():
        return False

    try:
        # Lazy import TransformRegistry
        from milia_pipeline.transformations.graph_transforms import registry as TransformRegistry

        success_count = 0
        for name, transform_class in _transform_classes.items():
            try:
                # Register transform
                TransformRegistry.register(name, transform_class)
                logger.debug(f"Registered transform: {name}")
                success_count += 1
            except Exception as e:
                logger.warning(f"Failed to register transform {name}: {e}")

        if success_count == len(_transform_classes):
            logger.info(f"All {success_count} transforms registered successfully")
            return True
        else:
            logger.warning(f"Only {success_count}/{len(_transform_classes)} transforms registered")
            return success_count > 0

    except ImportError:
        # TransformRegistry not available - fallback to direct exports
        logger.debug("TransformRegistry not available, using direct exports")
        return True
    except Exception as e:
        logger.error(f"Error registering transforms: {e}")
        return False


def enable_plugin() -> bool:
    """
    Enable this plugin and register all transforms.

    Can be called multiple times safely.

    Returns:
        bool: True if plugin enabled successfully
    """
    global _plugin_enabled

    if _plugin_enabled:
        logger.debug("Plugin already enabled")
        return True

    # Check dependencies
    deps_ok, missing = _check_dependencies()
    if not deps_ok:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        return False

    # Validate metadata
    if not _validate_plugin_metadata():
        logger.error("Plugin metadata validation failed")
        return False

    # Load transforms
    if not _lazy_import_transforms():
        logger.error("Failed to load transform classes")
        return False

    # Register with plugin system
    if not _register_with_plugin_system():
        logger.warning("Plugin system registration failed (may be expected)")

    # Register transforms
    if not _register_transforms():
        logger.warning("Transform registration encountered issues")

    _plugin_enabled = True
    logger.info(f"Plugin '{PLUGIN_METADATA['name']}' v{__version__} enabled")
    return True


def disable_plugin() -> bool:
    """
    Disable this plugin and unregister transforms.

    Returns:
        bool: True if plugin disabled successfully
    """
    global _plugin_enabled

    if not _plugin_enabled:
        logger.debug("Plugin not enabled")
        return True

    try:
        # Attempt to unregister from plugin system
        from milia_pipeline.transformations.plugin_system import PluginRegistry

        PluginRegistry.disable_plugin(PLUGIN_METADATA["name"])
        logger.debug("Unregistered from PluginRegistry")
    except Exception as e:
        logger.debug(f"Could not unregister from PluginRegistry: {e}")

    _plugin_enabled = False
    logger.info(f"Plugin '{PLUGIN_METADATA['name']}' disabled")
    return True


def get_plugin_info() -> dict[str, Any]:
    """
    Get comprehensive plugin information.

    Returns:
        dict: Plugin information including metadata, status, and transforms
    """
    return {
        "metadata": PLUGIN_METADATA.copy(),
        "status": {
            "registered": _plugin_registered,
            "enabled": _plugin_enabled,
            "transforms_loaded": _transforms_loaded,
            "registration_error": str(_registration_error) if _registration_error else None,
        },
        "transforms": {
            "available": _transform_names,
            "loaded": list(_transform_classes.keys()) if _transforms_loaded else [],
        },
        "dependencies": {
            "satisfied": _check_dependencies()[0],
            "missing": _check_dependencies()[1],
        },
    }


def get_transform(name: str) -> type | None:
    """
    Get a transform class by name.

    Args:
        name: Transform class name

    Returns:
        Transform class or None if not found
    """
    if not _lazy_import_transforms():
        logger.error("Cannot load transforms")
        return None

    return _transform_classes.get(name)


def list_transforms() -> list[str]:
    """
    List all available transform names.

    Returns:
        List of transform names
    """
    return _transform_names.copy()


# Auto-enable plugin on import (with graceful failure)
try:
    _auto_enable = enable_plugin()
    if not _auto_enable:
        logger.warning("Plugin auto-enable encountered issues but will continue")
except Exception as e:
    logger.error(f"Plugin auto-enable failed: {e}", exc_info=True)
    # Continue anyway - transforms may still be importable


# Direct imports for backward compatibility
# These will be available if transforms loaded successfully
if _lazy_import_transforms():
    DropEdge = _transform_classes.get("DropEdge")
    DropNode = _transform_classes.get("DropNode")
    MaskFeatures = _transform_classes.get("MaskFeatures")
    RandomNodeSample = _transform_classes.get("RandomNodeSample")
else:
    # Fallback: create placeholder warnings
    def _create_import_warning(name: str):
        def _placeholder(*args, **kwargs):
            raise ImportError(f"Transform {name} could not be loaded. Check dependencies.")

        return _placeholder

    DropEdge = _create_import_warning("DropEdge")
    DropNode = _create_import_warning("DropNode")
    MaskFeatures = _create_import_warning("MaskFeatures")
    RandomNodeSample = _create_import_warning("RandomNodeSample")


# Public API
__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "PLUGIN_METADATA",
    # Transform classes (backward compatibility)
    "DropEdge",
    "DropNode",
    "MaskFeatures",
    "RandomNodeSample",
    # Plugin management API
    "enable_plugin",
    "disable_plugin",
    "get_plugin_info",
    "get_transform",
    "list_transforms",
]


# Module-level docstring enhancement
__doc__ = f"""
PyG Augmentation Plugin v{__version__}
{"=" * 50}

Author: {__author__}
License: {PLUGIN_METADATA.get("license", "MIT")}
Plugin Type: {PLUGIN_METADATA.get("plugin_type", "pyg_fallback")}

Description:
{PLUGIN_METADATA.get("description", "PyG augmentation transforms missing from PyG 2.6.1")}

Available Transforms:
{chr(10).join(f"  - {name}" for name in _transform_names)}

Requirements:
  - milia: {PLUGIN_METADATA.get("milia_version", ">=4.0.0")}
  - PyG: {PLUGIN_METADATA.get("pyg_version", ">=2.0.0")}
  - Python: {PLUGIN_METADATA.get("python_version", ">=3.8")}

Usage:
    # Direct import (backward compatible)
    from milia_plugins.pyg_augmentation import DropEdge, DropNode

    # Plugin management
    from milia_plugins.pyg_augmentation import enable_plugin, get_plugin_info
    enable_plugin()
    info = get_plugin_info()

    # Dynamic access
    from milia_plugins.pyg_augmentation import get_transform
    DropEdge = get_transform('DropEdge')

Plugin Status:
  Registered: {_plugin_registered}
  Enabled: {_plugin_enabled}
  Transforms Loaded: {_transforms_loaded}
"""
