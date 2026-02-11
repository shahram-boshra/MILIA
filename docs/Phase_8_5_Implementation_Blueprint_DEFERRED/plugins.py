# milia_pipeline/datasets/plugins.py

"""
Entry point plugin loading for external dataset types.

This module provides the standard Python mechanism for loading external dataset
plugins via entry points. External packages can register datasets without
modifying MILIA source code.

Phase 8-5 Implementation
------------------------
This module implements Phase 8-5 of the Dataset Architecture Refactoring Plan:
"Implement entry point plugin loading"

Entry Point Group
-----------------
External packages register datasets using the 'milia.datasets' entry point group.

Example pyproject.toml configuration for external plugins:
    [project.entry-points."milia.datasets"]
    qm9 = "qm9_plugin.dataset:QM9Dataset"
    custom = "my_package.datasets:CustomDataset"

Design Decisions
----------------
- Uses Python 3.10+ importlib.metadata API (entry_points with group parameter)
- Validates loaded classes are proper BaseDataset subclasses
- Graceful error handling with detailed logging
- Returns information about loaded plugins for diagnostics
- Thread-safe: uses registry's built-in thread safety
- No auto-loading: explicit call to load_dataset_plugins() required

Integration
-----------
This module integrates with:
- milia_pipeline.datasets.registry: For registering discovered plugins
- milia_pipeline.datasets.base: For validating plugin classes
- milia_pipeline.exceptions: For error handling

Usage
-----
    # Load all external plugins
    from milia_pipeline.datasets.plugins import load_dataset_plugins
    loaded = load_dataset_plugins()
    print(f"Loaded {len(loaded)} plugins")
    
    # Or use the convenience function from __init__.py
    from milia_pipeline.datasets import initialize_plugins
    count = initialize_plugins(load_external=True)

See Also
--------
- MILIA_Dataset_Architecture_Refactoring_Plan_v2_2_0.md: Section 2.3.6
- Python Packaging Guide: Entry Points specification
- importlib.metadata documentation

Author: MILIA Pipeline Development Team
License: As per project license
"""

import logging
import sys
from typing import List, Tuple, Type, Optional, Dict, Any

from milia_pipeline.datasets.base import BaseDataset
from milia_pipeline.datasets.registry import get_default_registry
from milia_pipeline.exceptions import PluginLoadError, PluginValidationError

logger = logging.getLogger(__name__)

# Entry point group name for MILIA dataset plugins
ENTRY_POINT_GROUP: str = "milia.datasets"

# Module version
__version__: str = "1.0.0"


def _get_entry_points(group: str) -> List[Any]:
    """
    Get entry points for the specified group.
    
    Handles both Python 3.10+ API (entry_points with group parameter)
    and Python 3.9 API (entry_points returns dict-like object).
    
    Args:
        group: Entry point group name (e.g., 'milia.datasets')
        
    Returns:
        List of entry point objects for the specified group
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:
        # Fallback for Python < 3.8 (should not happen, but be safe)
        from importlib_metadata import entry_points  # type: ignore
    
    eps = entry_points()
    
    # Python 3.10+ API: entry_points() returns SelectableGroups
    # which can be indexed by group name directly
    if hasattr(eps, 'select'):
        # Python 3.10+ with selectable groups
        return list(eps.select(group=group))
    elif hasattr(eps, 'get'):
        # Python 3.9: entry_points() returns dict-like SelectableGroups
        return list(eps.get(group, []))
    elif isinstance(eps, dict):
        # Fallback: entry_points returns actual dict (older API)
        return list(eps.get(group, []))
    else:
        # Python 3.10+ where entry_points(group=...) works
        try:
            return list(entry_points(group=group))
        except TypeError:
            # If group parameter not supported, filter manually
            all_eps = list(eps)
            return [ep for ep in all_eps if getattr(ep, 'group', None) == group]


def _validate_plugin_class(
    entry_point_name: str,
    loaded_class: Any
) -> Optional[str]:
    """
    Validate that a loaded class is a proper BaseDataset subclass.
    
    Args:
        entry_point_name: Name of the entry point (for error messages)
        loaded_class: The class loaded from the entry point
        
    Returns:
        None if validation passes, error message string if validation fails
    """
    # Check if it's a class
    if not isinstance(loaded_class, type):
        return f"Entry point '{entry_point_name}' did not provide a class, got {type(loaded_class).__name__}"
    
    # Check if it's a BaseDataset subclass
    if not issubclass(loaded_class, BaseDataset):
        return (
            f"Entry point '{entry_point_name}' class '{loaded_class.__name__}' "
            f"is not a BaseDataset subclass"
        )
    
    # Check if it's abstract (has unimplemented abstract methods)
    if hasattr(loaded_class, '__abstractmethods__') and loaded_class.__abstractmethods__:
        return (
            f"Entry point '{entry_point_name}' class '{loaded_class.__name__}' "
            f"is abstract. Missing implementations: {loaded_class.__abstractmethods__}"
        )
    
    # Check required class attributes exist (BaseDataset.__init_subclass__ should catch this,
    # but we double-check for extra safety with external plugins)
    required_attrs = ['metadata', 'schema', 'features', 'config_key']
    missing_attrs = [attr for attr in required_attrs if not hasattr(loaded_class, attr)]
    if missing_attrs:
        return (
            f"Entry point '{entry_point_name}' class '{loaded_class.__name__}' "
            f"missing required class attributes: {missing_attrs}"
        )
    
    return None  # Validation passed


def load_dataset_plugins() -> List[Tuple[str, Type[BaseDataset]]]:
    """
    Load dataset plugins via entry points.
    
    This is the standard Python mechanism for plugins. External packages can
    register datasets without modifying MILIA source code by declaring entry
    points in their pyproject.toml:
    
        [project.entry-points."milia.datasets"]
        my_dataset = "my_package:MyDataset"
    
    The function discovers all packages that declare the 'milia.datasets' entry
    point group, loads each entry point, validates it's a proper BaseDataset
    subclass, and registers it with the global DatasetRegistry.
    
    Returns:
        List of (entry_point_name, dataset_class) tuples for successfully
        loaded plugins. The entry_point_name is the name declared in the
        external package's entry points configuration.
        
    Raises:
        No exceptions are raised. All errors are logged and the function
        continues to attempt loading remaining plugins.
        
    Example:
        >>> plugins = load_dataset_plugins()
        >>> for name, cls in plugins:
        ...     print(f"Loaded plugin: {name} -> {cls.__name__}")
        Loaded plugin: qm9 -> QM9Dataset
        Loaded plugin: ani1 -> ANI1Dataset
    
    Notes:
        - Plugins are registered with the default global registry
        - If a plugin with the same dataset name is already registered,
          registration will fail with a DatasetRegistrationError (logged as warning)
        - Load order is not guaranteed; do not rely on specific plugin order
        - This function is idempotent for the same set of installed plugins
          (re-registering the same class is allowed by the registry)
    """
    registry = get_default_registry()
    loaded: List[Tuple[str, Type[BaseDataset]]] = []
    
    # Get entry points for our group
    eps = _get_entry_points(ENTRY_POINT_GROUP)
    
    if not eps:
        logger.debug(f"No external dataset plugins found in '{ENTRY_POINT_GROUP}' entry point group")
        return loaded
    
    logger.info(f"Discovered {len(eps)} dataset plugin(s) in '{ENTRY_POINT_GROUP}' entry point group")
    
    for ep in eps:
        ep_name = getattr(ep, 'name', str(ep))
        
        try:
            logger.debug(f"Loading dataset plugin: {ep_name}")
            
            # Load the entry point (imports the module and gets the class)
            dataset_class = ep.load()
            
            # Validate the loaded class
            validation_error = _validate_plugin_class(ep_name, dataset_class)
            if validation_error:
                logger.warning(validation_error)
                continue
            
            # Register with the global registry
            registry.register(dataset_class)
            
            loaded.append((ep_name, dataset_class))
            logger.info(
                f"Successfully loaded dataset plugin: {ep_name} "
                f"-> {dataset_class.__name__} (dataset type: '{dataset_class.metadata.name}')"
            )
            
        except Exception as e:
            # Log the error but continue loading other plugins
            logger.error(
                f"Failed to load dataset plugin '{ep_name}': {type(e).__name__}: {e}"
            )
            # Log traceback at debug level for troubleshooting
            logger.debug(f"Plugin load error traceback for '{ep_name}':", exc_info=True)
    
    return loaded


def discover_and_load_plugins() -> int:
    """
    Discover and load all dataset plugins.
    
    This is a convenience function that calls load_dataset_plugins() and
    returns just the count of successfully loaded plugins.
    
    Returns:
        Number of plugins successfully loaded
        
    Example:
        >>> count = discover_and_load_plugins()
        >>> print(f"Loaded {count} external dataset plugins")
        Loaded 2 external dataset plugins
    """
    plugins = load_dataset_plugins()
    return len(plugins)


def get_plugin_info() -> Dict[str, Any]:
    """
    Get information about the plugin system.
    
    Returns a dictionary with plugin system status and diagnostics.
    
    Returns:
        Dict containing:
        - version: Plugin module version
        - entry_point_group: The entry point group name used
        - discovered_plugins: List of discovered entry point names
        - python_version: Current Python version
        - api_style: Which importlib.metadata API style is being used
        
    Example:
        >>> info = get_plugin_info()
        >>> print(f"Entry point group: {info['entry_point_group']}")
        Entry point group: milia.datasets
    """
    eps = _get_entry_points(ENTRY_POINT_GROUP)
    
    # Determine API style
    try:
        from importlib.metadata import entry_points
        test_eps = entry_points()
        if hasattr(test_eps, 'select'):
            api_style = "Python 3.10+ (SelectableGroups with select())"
        elif hasattr(test_eps, 'get'):
            api_style = "Python 3.9 (SelectableGroups with get())"
        else:
            api_style = "Python 3.10+ (entry_points with group parameter)"
    except ImportError:
        api_style = "importlib_metadata backport"
    
    return {
        'version': __version__,
        'entry_point_group': ENTRY_POINT_GROUP,
        'discovered_plugins': [getattr(ep, 'name', str(ep)) for ep in eps],
        'discovered_count': len(eps),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'api_style': api_style,
    }


def list_available_plugins() -> List[str]:
    """
    List names of available (discovered but not necessarily loaded) plugins.
    
    This function only discovers plugins; it does not load them.
    Use load_dataset_plugins() to actually load and register the plugins.
    
    Returns:
        List of entry point names for available plugins
        
    Example:
        >>> available = list_available_plugins()
        >>> print(f"Available plugins: {available}")
        Available plugins: ['qm9', 'ani1', 'custom_dataset']
    """
    eps = _get_entry_points(ENTRY_POINT_GROUP)
    return [getattr(ep, 'name', str(ep)) for ep in eps]
