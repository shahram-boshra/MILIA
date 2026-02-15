# milia_pipeline/preprocessing/preprocessors/__init__.py

"""
Preprocessor Implementations
============================

Concrete preprocessor implementations for different dataset types.

DYNAMIC LOADING: All preprocessor modules in this directory are automatically
discovered and imported. No manual imports or __all__ updates needed when
adding new preprocessors - just create the file.

Author: milia Pipeline Team
Version: 1.4
Date: December 2025
"""

import importlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Modules to exclude from dynamic import
_EXCLUDED_MODULES = {"__init__", "base", "utils", "common"}

# Dynamic import: scan this directory for all .py files and import them
_preprocessors_dir = Path(__file__).parent
_discovered_modules = []

for _py_file in _preprocessors_dir.glob("*.py"):
    _module_name = _py_file.stem
    if _module_name in _EXCLUDED_MODULES or _module_name.startswith("_"):
        continue

    try:
        # Import the module - this triggers any auto-registration
        _module = importlib.import_module(f".{_module_name}", package=__name__)
        _discovered_modules.append(_module_name)
        logger.debug(f"Dynamically loaded preprocessor: {_module_name}")
    except Exception as e:
        logger.warning(f"Failed to import preprocessor module '{_module_name}': {e}")

# Dynamically build __all__ from discovered modules
__all__ = _discovered_modules
