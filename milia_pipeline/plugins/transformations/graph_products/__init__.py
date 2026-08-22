# milia_pipeline/plugins/transformations/graph_products/__init__

"""
Graph Products Plugin (catalogue Family C).

Algebraic graph-product transforms — Cartesian, tensor, strong, lexicographic,
rooted, corona — plus stochastic-Kronecker synthetic-graph generation. NetworkX
-native (no new dependency); MILIA-custom idiom (CustomTransformBase).

Production-ready plugin with lazy loading, graceful degradation, and integration
with the MILIA PluginRegistry / TransformRegistry. The authoritative registration
path is plugin discovery (add_plugin_path + discover_plugins); the lifecycle hooks
below make the plugin usable via direct import as well.
"""

import logging
from typing import Any

__version__ = "1.0.0"
__author__ = "MILIA Team"

PLUGIN_METADATA = {
    "name": "graph_products",
    "version": __version__,
    "author": __author__,
    "plugin_type": "user_experimental",
    "description": "Algebraic graph-product transforms plus stochastic-Kronecker generation.",
    "license": "MIT",
    "milia_version": ">=1.2.2,<2.0.0",
    "pyg_version": ">=2.6.0,<2.7.0",
    "python_version": ">=3.10",
}

logger = logging.getLogger("milia_Main.PluginSystem.graph_products")

_transform_names = [
    "CartesianProduct",
    "TensorProduct",
    "StrongProduct",
    "LexicographicProduct",
    "RootedProduct",
    "CoronaProduct",
    "KroneckerGraphGeneration",
]

_transforms_loaded = False
_transform_classes: dict[str, type] = {}
_plugin_registered = False
_plugin_enabled = False
_registration_error: Exception | None = None


def _lazy_import_transforms() -> bool:
    """Lazily import the transform classes (avoids import-time circular deps)."""
    global _transforms_loaded, _transform_classes

    if _transforms_loaded:
        return True

    try:
        from .transforms import (
            CartesianProduct,
            CoronaProduct,
            KroneckerGraphGeneration,
            LexicographicProduct,
            RootedProduct,
            StrongProduct,
            TensorProduct,
        )

        _transform_classes = {
            "CartesianProduct": CartesianProduct,
            "TensorProduct": TensorProduct,
            "StrongProduct": StrongProduct,
            "LexicographicProduct": LexicographicProduct,
            "RootedProduct": RootedProduct,
            "CoronaProduct": CoronaProduct,
            "KroneckerGraphGeneration": KroneckerGraphGeneration,
        }
        _transforms_loaded = True
        logger.debug("Loaded graph_products transform classes")
        return True
    except Exception as exc:  # noqa: BLE001 - report and degrade gracefully
        logger.error(f"Failed to import graph_products transforms: {exc}")
        return False


def _check_dependencies() -> tuple[bool, list[str]]:
    """Check that required runtime dependencies are importable."""
    missing: list[str] = []
    for module_name, label in (("networkx", "networkx"), ("torch_geometric", "torch_geometric")):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(label)
    return len(missing) == 0, missing


def _register_transforms() -> bool:
    """Register the transform classes with the TransformRegistry (correct signature).

    Plugin-level registration with the PluginRegistry is performed by discovery
    (``add_plugin_path`` + ``discover_plugins``), not here; this hook only makes the
    transforms usable via direct import.
    """
    global _registration_error

    if not _lazy_import_transforms():
        return False

    try:
        from milia_pipeline.transformations.graph_transforms import TransformRegistry
    except ImportError:
        logger.debug("TransformRegistry unavailable; relying on direct exports")
        return True

    registered = 0
    for name, transform_class in _transform_classes.items():
        try:
            TransformRegistry.register(transform_class, name, transform_class.get_metadata())
            registered += 1
        except Exception as exc:  # noqa: BLE001
            _registration_error = exc
            logger.warning(f"Failed to register transform {name}: {exc}")
    return registered == len(_transform_classes)


def enable_plugin() -> bool:
    """Enable the plugin: check deps, register transforms. Idempotent."""
    global _plugin_enabled, _plugin_registered

    if _plugin_enabled:
        return True

    deps_ok, missing = _check_dependencies()
    if not deps_ok:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        return False

    if not _lazy_import_transforms():
        return False

    _plugin_registered = _register_transforms()

    _plugin_enabled = True
    logger.info(f"Plugin 'graph_products' v{__version__} enabled")
    return True


def disable_plugin() -> bool:
    """Disable the plugin (best-effort unregister from PluginRegistry)."""
    global _plugin_enabled

    if not _plugin_enabled:
        return True

    try:
        from milia_pipeline.transformations.plugin_system import PluginRegistry

        PluginRegistry.disable_plugin(PLUGIN_METADATA["name"])
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Could not disable via PluginRegistry: {exc}")

    _plugin_enabled = False
    logger.info("Plugin 'graph_products' disabled")
    return True


def get_transform(name: str) -> type | None:
    """Return a transform class by name (loads lazily)."""
    if not _lazy_import_transforms():
        return None
    return _transform_classes.get(name)


def list_transforms() -> list[str]:
    """List available transform names."""
    return _transform_names.copy()


def get_plugin_info() -> dict[str, Any]:
    """Return plugin metadata, status, transforms, and dependency state."""
    deps_ok, missing = _check_dependencies()
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
        "dependencies": {"satisfied": deps_ok, "missing": missing},
    }


# Auto-enable on import (graceful).
try:
    if not enable_plugin():
        logger.warning("graph_products auto-enable encountered issues; continuing")
except Exception as exc:  # noqa: BLE001
    logger.error(f"graph_products auto-enable failed: {exc}")


__all__ = [
    "__version__",
    "__author__",
    "PLUGIN_METADATA",
    "enable_plugin",
    "disable_plugin",
    "get_transform",
    "list_transforms",
    "get_plugin_info",
]
