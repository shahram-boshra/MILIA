"""
MyPlugins - User Experimental Transforms Plugin

Custom transforms for milia Pipeline.
Production-ready with graceful error handling for missing transforms.
"""

import logging
from typing import Any

__version__ = "1.0.0"
__author__ = "milia User"

# Plugin metadata for milia plugin system
PLUGIN_METADATA = {
    "name": "myplugins",
    "version": __version__,
    "author": __author__,
    "plugin_type": "user_experimental",
    "description": "User experimental transforms for milia Pipeline",
}

# Logger for this plugin
logger = logging.getLogger("milia_Main.PluginSystem.myplugins")

# Track what transforms are actually available
_available_transforms = {}
_attempted_imports = set()


def __getattr__(name: str) -> Any:
    """
    Lazy loading of transforms with graceful error handling.

    This allows transforms to be imported on-demand, and handles
    missing transforms gracefully without crashing the plugin system.

    Args:
        name: Name of the attribute/transform to import

    Returns:
        The requested transform class

    Raises:
        AttributeError: If transform not found after all attempts
    """
    # Avoid infinite recursion
    if name in _attempted_imports:
        raise AttributeError(f"Transform '{name}' not available in myplugins")

    _attempted_imports.add(name)

    # Try to import from transforms module
    transform_map = {
        "EnergyNormalizer": "energy_normalizer",
        "DropEdge": "drop_edge",
        "DropNode": "drop_node",
        "MaskFeatures": "mask_features",
        "RandomNodeSample": "random_node_sample",
        # Add more transform mappings as needed
    }

    if name in transform_map:
        module_name = transform_map[name]
        try:
            # Try to import the specific transform
            module = __import__(
                f"milia_pipeline.plugins.myplugins.transforms.{module_name}", fromlist=[name]
            )
            transform_class = getattr(module, name)

            # Cache it for future use
            _available_transforms[name] = transform_class
            logger.debug(f"Successfully imported transform: {name}")

            return transform_class

        except (ImportError, ModuleNotFoundError) as e:
            logger.debug(f"Transform '{name}' not found in myplugins: {e}")
            # Try to fall back to pyg_augmentation plugin if it's a PyG transform
            if name in ["DropEdge", "DropNode", "MaskFeatures", "RandomNodeSample"]:
                try:
                    from milia_pipeline.plugins.pyg_augmentation import get_transform

                    fallback_transform = get_transform(name)
                    if fallback_transform is not None:
                        logger.info(f"Using {name} from pyg_augmentation fallback plugin")
                        _available_transforms[name] = fallback_transform
                        return fallback_transform
                except Exception as fallback_error:
                    logger.debug(f"Could not import from pyg_augmentation: {fallback_error}")

            # Transform doesn't exist anywhere
            raise AttributeError(
                f"Transform '{name}' not found in myplugins. "
                f"Create it at: milia_pipeline/plugins/myplugins/transforms/{module_name}.py"
            )

        except AttributeError as e:
            logger.error(f"Module exists but class '{name}' not found: {e}")
            raise AttributeError(
                f"Transform class '{name}' not found in module '{module_name}'. "
                f"Check that the class is properly defined and exported."
            )

    # Not a known transform name
    raise AttributeError(f"'{__name__}' has no attribute '{name}'")


def list_available_transforms():
    """
    List all available transforms in this plugin.

    Returns:
        list: Names of available transforms
    """
    available = []
    transform_map = {
        "EnergyNormalizer": "energy_normalizer",
        "DropEdge": "drop_edge",
        "DropNode": "drop_node",
        "MaskFeatures": "mask_features",
        "RandomNodeSample": "random_node_sample",
    }

    for name, module_name in transform_map.items():
        try:
            module = __import__(
                f"milia_pipeline.plugins.myplugins.transforms.{module_name}", fromlist=[name]
            )
            if hasattr(module, name):
                available.append(name)
        except (ImportError, ModuleNotFoundError):
            continue

    return available


# Public API
__all__ = [
    "PLUGIN_METADATA",
    "__version__",
    "__author__",
    "list_available_transforms",
]

# Note: Transform classes are loaded via __getattr__ on demand
# Example usage:
#   from milia_pipeline.plugins.myplugins import EnergyNormalizer
#   # This will trigger __getattr__ which loads the transform
