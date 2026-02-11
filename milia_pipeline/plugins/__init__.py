"""
Plugins Sub-Package for MILIA Pipeline

Provides a unified namespace for all plugin sub-packages (transform plugins,
descriptor plugins, etc.) with dynamic sub-plugin discovery, lazy loading,
and graceful degradation.

Production-ready with:
- Dynamic sub-plugin discovery at runtime (no hardcoded plugin names)
- Lazy loading to avoid circular dependencies with transformations.plugin_system
- Unified plugin management API delegating to per-plugin APIs
- Thread-safe operations consistent with PluginRegistry singleton pattern
- Graceful handling of missing dependencies or broken plugins
- Forward-compatible: new plugin sub-packages are auto-discovered

Architecture:
    milia_pipeline/plugins/
    ├── __init__.py                  ← This file (package-level API)
    ├── descriptors/                 # Descriptor plugins (filesystem-based, no __init__.py)
    │   ├── example_descriptors/     # Discovered by DescriptorPluginLoader via plugin.yaml
    │   └── user_template/           # Plugin template for users
    ├── pyg_augmentation/            # PyG fallback transforms (plugin_type: pyg_fallback)
    │   ├── __init__.py
    │   ├── transforms.py
    │   └── plugin.yaml
    └── myplugins/                   # User experimental transforms (plugin_type: user_experimental)
        ├── __init__.py
        ├── transforms/
        └── plugin.yaml

Author: MILIA Team
License: MIT
"""

import importlib
import logging
import pkgutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

__version__ = "1.0.0"
__author__ = "MILIA Team"

# Logger consistent with project-wide naming convention
# (see pyg_augmentation: "milia_Main.PluginSystem.pyg_augmentation",
#  myplugins: "milia_Main.PluginSystem.myplugins")
logger = logging.getLogger("milia_Main.PluginSystem")

# ──────────────────────────────────────────────────────────────────────
# Internal state — guarded by _lock for thread safety (consistent with
# PluginRegistry's thread-safe singleton pattern in plugin_system.py)
# ──────────────────────────────────────────────────────────────────────
_lock = threading.Lock()

# Dynamic discovery caches
_discovered_subplugins: Optional[Dict[str, Any]] = None
_discovery_attempted: bool = False


# ──────────────────────────────────────────────────────────────────────
# Dynamic Sub-Plugin Discovery
# ──────────────────────────────────────────────────────────────────────

def _discover_subplugins() -> Dict[str, Any]:
    """
    Dynamically discover all importable sub-plugin packages under this directory.

    Uses ``pkgutil.iter_modules`` on the current package path to find sub-packages
    that are proper Python packages (i.e., contain ``__init__.py``).  Directories
    without ``__init__.py`` (like ``descriptors/``) are intentionally skipped —
    they are filesystem-based plugin directories discovered separately by
    ``DescriptorPluginLoader`` via ``plugin.yaml`` scanning.

    This approach is fully dynamic: adding a new plugin sub-package only requires
    creating a directory with an ``__init__.py`` — no modification to this file.

    Returns:
        Dict[str, module]: Mapping of sub-plugin name → lazily loaded module,
                           or None if import deferred.
    """
    global _discovered_subplugins, _discovery_attempted

    with _lock:
        if _discovered_subplugins is not None:
            return _discovered_subplugins

        _discovery_attempted = True
        _discovered_subplugins = {}

        package_path = Path(__file__).parent

        for importer, modname, ispkg in pkgutil.iter_modules([str(package_path)]):
            if not ispkg:
                # Skip non-package modules (standalone .py files at this level)
                continue

            # Only consider directories that are Python packages
            # (pkgutil.iter_modules already checks for __init__.py presence)
            _discovered_subplugins[modname] = None  # Lazy — module loaded on demand
            logger.debug(f"Discovered plugin sub-package: {modname}")

        if _discovered_subplugins:
            logger.debug(
                f"Discovered {len(_discovered_subplugins)} plugin sub-package(s): "
                f"{', '.join(sorted(_discovered_subplugins.keys()))}"
            )
        else:
            logger.debug("No plugin sub-packages discovered")

        return _discovered_subplugins


def _get_subplugin_module(name: str) -> Optional[Any]:
    """
    Lazily import and cache a sub-plugin module by name.

    Args:
        name: Sub-plugin package name (e.g., 'pyg_augmentation', 'myplugins')

    Returns:
        The imported module, or None if import fails.
    """
    subplugins = _discover_subplugins()

    if name not in subplugins:
        logger.debug(f"Sub-plugin '{name}' not found among discovered packages")
        return None

    # Check if already loaded
    if subplugins[name] is not None:
        return subplugins[name]

    # Lazy import
    with _lock:
        # Double-check after acquiring lock
        if subplugins[name] is not None:
            return subplugins[name]

        try:
            module = importlib.import_module(f".{name}", package=__name__)
            subplugins[name] = module
            logger.debug(f"Loaded sub-plugin module: {name}")
            return module
        except ImportError as e:
            logger.warning(f"Failed to import sub-plugin '{name}': {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error importing sub-plugin '{name}': {e}")
            return None


# ──────────────────────────────────────────────────────────────────────
# Module-level __getattr__ for lazy sub-plugin access
# ──────────────────────────────────────────────────────────────────────

def __getattr__(name: str) -> Any:
    """
    Enable lazy attribute access to sub-plugin packages.

    Allows ``from milia_pipeline.plugins import pyg_augmentation`` or
    ``milia_pipeline.plugins.myplugins`` to work without eagerly importing
    all sub-plugins at package load time, avoiding circular dependencies
    with ``transformations.plugin_system`` → ``plugins`` → ``transformations``.

    This is consistent with the lazy-loading pattern used in:
    - ``myplugins/__init__.py`` (``__getattr__`` for transform classes)
    - ``config/config_constants.py`` (``__getattr__`` for lazy constants)

    Args:
        name: Attribute name to resolve

    Returns:
        The sub-plugin module if found

    Raises:
        AttributeError: If the name is not a known sub-plugin or attribute
    """
    subplugins = _discover_subplugins()

    if name in subplugins:
        module = _get_subplugin_module(name)
        if module is not None:
            return module
        raise AttributeError(
            f"Plugin sub-package '{name}' was discovered but could not be imported. "
            f"Check its dependencies and __init__.py."
        )

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# ──────────────────────────────────────────────────────────────────────
# Unified Plugin Management API
# ──────────────────────────────────────────────────────────────────────

def list_subplugins() -> List[str]:
    """
    List all discovered plugin sub-packages.

    Returns only Python-package sub-plugins (those with ``__init__.py``).
    Filesystem-based plugin directories (like ``descriptors/``) are managed
    separately by their respective loaders (e.g., ``DescriptorPluginLoader``).

    Returns:
        List of sub-plugin package names.
    """
    return sorted(_discover_subplugins().keys())


def get_subplugin_info(name: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a specific sub-plugin.

    Delegates to the sub-plugin's ``get_plugin_info()`` if available,
    otherwise extracts ``PLUGIN_METADATA`` directly.

    Args:
        name: Sub-plugin package name

    Returns:
        Plugin information dict, or None if sub-plugin not found/loadable.
    """
    module = _get_subplugin_module(name)
    if module is None:
        return None

    # Prefer get_plugin_info() — present in pyg_augmentation and extensible
    if hasattr(module, 'get_plugin_info'):
        try:
            return module.get_plugin_info()
        except Exception as e:
            logger.warning(f"get_plugin_info() failed for '{name}': {e}")

    # Fallback to PLUGIN_METADATA dict — present in both pyg_augmentation and myplugins
    if hasattr(module, 'PLUGIN_METADATA'):
        return {'metadata': dict(module.PLUGIN_METADATA)}

    return {'metadata': {'name': name, 'status': 'loaded'}}


def get_all_plugin_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all discovered sub-plugins.

    Returns:
        Dict mapping sub-plugin name → plugin info dict.
    """
    result = {}
    for name in list_subplugins():
        info = get_subplugin_info(name)
        if info is not None:
            result[name] = info
    return result


def enable_subplugin(name: str) -> bool:
    """
    Enable a specific sub-plugin.

    Delegates to the sub-plugin's ``enable_plugin()`` if available.

    Args:
        name: Sub-plugin package name

    Returns:
        True if enabled successfully, False otherwise.
    """
    module = _get_subplugin_module(name)
    if module is None:
        logger.error(f"Cannot enable sub-plugin '{name}': not found or not loadable")
        return False

    if hasattr(module, 'enable_plugin'):
        try:
            return module.enable_plugin()
        except Exception as e:
            logger.error(f"Failed to enable sub-plugin '{name}': {e}")
            return False

    logger.debug(f"Sub-plugin '{name}' has no enable_plugin() — treating as enabled")
    return True


def disable_subplugin(name: str) -> bool:
    """
    Disable a specific sub-plugin.

    Delegates to the sub-plugin's ``disable_plugin()`` if available.

    Args:
        name: Sub-plugin package name

    Returns:
        True if disabled successfully, False otherwise.
    """
    module = _get_subplugin_module(name)
    if module is None:
        logger.warning(f"Cannot disable sub-plugin '{name}': not found or not loadable")
        return False

    if hasattr(module, 'disable_plugin'):
        try:
            return module.disable_plugin()
        except Exception as e:
            logger.error(f"Failed to disable sub-plugin '{name}': {e}")
            return False

    logger.debug(f"Sub-plugin '{name}' has no disable_plugin() — treating as disabled")
    return True


def get_subplugin_transforms(name: str) -> List[str]:
    """
    List available transforms from a specific sub-plugin.

    Delegates to the sub-plugin's ``list_transforms()`` or
    ``list_available_transforms()`` if available.

    Args:
        name: Sub-plugin package name

    Returns:
        List of transform names, or empty list if not applicable.
    """
    module = _get_subplugin_module(name)
    if module is None:
        return []

    # pyg_augmentation uses list_transforms()
    if hasattr(module, 'list_transforms'):
        try:
            return module.list_transforms()
        except Exception as e:
            logger.warning(f"list_transforms() failed for '{name}': {e}")

    # myplugins uses list_available_transforms()
    if hasattr(module, 'list_available_transforms'):
        try:
            return module.list_available_transforms()
        except Exception as e:
            logger.warning(f"list_available_transforms() failed for '{name}': {e}")

    return []


def get_transform(name: str, transform_name: str) -> Optional[Type]:
    """
    Get a specific transform class from a sub-plugin.

    Delegates to the sub-plugin's ``get_transform()`` if available.

    Args:
        name: Sub-plugin package name
        transform_name: Transform class name

    Returns:
        Transform class, or None if not found.
    """
    module = _get_subplugin_module(name)
    if module is None:
        return None

    if hasattr(module, 'get_transform'):
        try:
            return module.get_transform(transform_name)
        except Exception as e:
            logger.warning(
                f"get_transform('{transform_name}') failed for '{name}': {e}"
            )

    # Fallback: try direct attribute access (consistent with __getattr__ in myplugins)
    try:
        return getattr(module, transform_name, None)
    except Exception:
        return None


def get_plugins_directory() -> Path:
    """
    Get the absolute path to the plugins directory.

    Useful for ``PluginRegistry.add_plugin_path()`` and
    ``DescriptorPluginLoader.discover_plugins()`` which take filesystem paths.

    Returns:
        Path to this plugins directory.
    """
    return Path(__file__).parent


def get_descriptor_plugins_directory() -> Path:
    """
    Get the absolute path to the descriptor plugins directory.

    Useful for ``DescriptorPluginLoader`` which scans for ``plugin.yaml``
    files in this directory tree.

    Returns:
        Path to the descriptors plugin directory.
    """
    return Path(__file__).parent / "descriptors"


def get_system_status() -> Dict[str, Any]:
    """
    Get comprehensive status of the plugins sub-package.

    Returns:
        Dict with discovery status, sub-plugin details, and directory paths.
    """
    subplugins = _discover_subplugins()
    plugin_info = {}

    for name in sorted(subplugins.keys()):
        info = get_subplugin_info(name)
        if info is not None:
            plugin_info[name] = info
        else:
            plugin_info[name] = {'status': 'discovered_but_not_loadable'}

    return {
        'version': __version__,
        'plugins_directory': str(get_plugins_directory()),
        'descriptor_plugins_directory': str(get_descriptor_plugins_directory()),
        'discovery': {
            'attempted': _discovery_attempted,
            'subplugins_found': len(subplugins),
            'subplugin_names': sorted(subplugins.keys()),
        },
        'subplugins': plugin_info,
    }


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

__all__ = [
    # Metadata
    '__version__',
    '__author__',

    # Discovery
    'list_subplugins',
    'get_subplugin_info',
    'get_all_plugin_info',

    # Plugin lifecycle
    'enable_subplugin',
    'disable_subplugin',

    # Transform access
    'get_subplugin_transforms',
    'get_transform',

    # Directory paths
    'get_plugins_directory',
    'get_descriptor_plugins_directory',

    # System status
    'get_system_status',
]
