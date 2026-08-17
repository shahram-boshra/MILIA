"""
Static Padding & Alignment Plugin (catalogue Family H).

Five array-layout/alignment transforms for static compilation (torch-only; no new
dependency). MILIA-custom idiom. Lazy loading, graceful degradation, and integration
with the MILIA TransformRegistry. Authoritative registration is via plugin discovery
(add_plugin_path + discover_plugins); the hooks below enable direct-import use too.
"""

import logging
from typing import Any

__version__ = "1.0.0"
__author__ = "MILIA Team"

PLUGIN_METADATA = {
    "name": "static_padding",
    "version": __version__,
    "author": __author__,
    "plugin_type": "user_experimental",
    "description": "Array layout & alignment operators for static compilation (Family H).",
    "license": "MIT",
    "milia_version": ">=1.2.2,<2.0.0",
    "pyg_version": ">=2.6.0,<2.7.0",
    "python_version": ">=3.10",
}

logger = logging.getLogger("milia_Main.PluginSystem.static_padding")

_transform_names = [
    "StaticBatchPadding",
    "VirtualNodeMasking",
    "PowerOfTwoPadding",
    "LayerPreprocess",
    "EdgeAttrCanonicalPadding",
]

_transforms_loaded = False
_transform_classes: dict[str, type] = {}
_plugin_registered = False
_plugin_enabled = False
_registration_error: Exception | None = None


def _lazy_import_transforms() -> bool:
    global _transforms_loaded, _transform_classes

    if _transforms_loaded:
        return True

    try:
        from . import transforms as _t

        _transform_classes = {name: getattr(_t, name) for name in _transform_names}
        _transforms_loaded = True
        logger.debug("Loaded static_padding transform classes")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to import static_padding transforms: {exc}")
        return False


def _check_dependencies() -> tuple[bool, list[str]]:
    missing: list[str] = []
    for module_name in ("torch", "torch_geometric"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    return len(missing) == 0, missing


def _register_transforms() -> bool:
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
    logger.info(f"Plugin 'static_padding' v{__version__} enabled")
    return True


def disable_plugin() -> bool:
    global _plugin_enabled

    if not _plugin_enabled:
        return True

    try:
        from milia_pipeline.transformations.plugin_system import PluginRegistry

        PluginRegistry.disable_plugin(PLUGIN_METADATA["name"])
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Could not disable via PluginRegistry: {exc}")

    _plugin_enabled = False
    logger.info("Plugin 'static_padding' disabled")
    return True


def get_transform(name: str) -> type | None:
    if not _lazy_import_transforms():
        return None
    return _transform_classes.get(name)


def list_transforms() -> list[str]:
    return _transform_names.copy()


def get_plugin_info() -> dict[str, Any]:
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


try:
    if not enable_plugin():
        logger.warning("static_padding auto-enable encountered issues; continuing")
except Exception as exc:  # noqa: BLE001
    logger.error(f"static_padding auto-enable failed: {exc}")


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
