# milia_pipeline/datasets/implementations/__init__.py
"""
Dataset implementations subpackage.

This package contains concrete dataset implementations that extend BaseDataset
and register with the global DatasetRegistry.

Each dataset class:
1. Extends BaseDataset from milia_pipeline.datasets.base
2. Defines immutable metadata (DatasetMetadata, DatasetSchema, DatasetFeatures)
3. Implements abstract methods (get_required_properties, get_feature_support, get_molecule_creation_strategy)
4. Uses @register decorator for automatic registration

DYNAMIC LOADING: All dataset modules in this directory are automatically discovered
and imported, triggering their @register decorators. No manual imports or __all__
updates are needed when adding new datasets - just create the file with @register.

Phase 2 Implementation - Dynamic Dataset Discovery
"""

import importlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Modules to exclude from dynamic import (utility modules, not dataset implementations)
_EXCLUDED_MODULES = {"__init__", "base", "registry", "utils", "common", "protocols"}

# Dynamic import: scan this directory for all .py files and import them
_implementations_dir = Path(__file__).parent
_discovered_classes = {}

for _py_file in _implementations_dir.glob("*.py"):
    _module_name = _py_file.stem
    if _module_name in _EXCLUDED_MODULES or _module_name.startswith("_"):
        continue

    try:
        # Import the module - this triggers @register decorator on the dataset class
        _module = importlib.import_module(f".{_module_name}", package=__name__)

        # Find the dataset class (convention: class name ends with 'Dataset')
        for _attr_name in dir(_module):
            if _attr_name.endswith("Dataset") and not _attr_name.startswith("_"):
                _cls = getattr(_module, _attr_name)
                # Verify it's a proper dataset class with metadata
                if hasattr(_cls, "metadata") and hasattr(_cls.metadata, "name"):
                    _discovered_classes[_attr_name] = _cls
                    logger.debug(f"Dynamically loaded dataset: {_attr_name} from {_module_name}.py")
    except Exception as e:
        logger.warning(f"Failed to import dataset module '{_module_name}': {e}")

# Export all discovered classes to module namespace
globals().update(_discovered_classes)

# Dynamically build __all__ from discovered classes
__all__ = list(_discovered_classes.keys())
