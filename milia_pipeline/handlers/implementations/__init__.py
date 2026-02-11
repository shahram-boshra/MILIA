# milia_pipeline/handlers/implementations/__init__.py

"""
Handler Implementations Package
===============================

This package contains concrete handler implementations that extend DatasetHandler
and register with the global HandlerRegistry.

Each handler class:
1. Extends DatasetHandler from milia_pipeline.handlers.base_handler
2. Implements all 12 abstract methods + 4 transform validation methods
3. Uses @register_handler decorator for automatic registration

DYNAMIC LOADING: All handler modules in this directory are automatically discovered
and imported, triggering their @register_handler decorators. No manual imports or __all__
updates are needed when adding new handlers - just create the file with @register_handler.

Pattern follows datasets/implementations/__init__.py (proven in production).

Adding a New Handler:
--------------------
1. Create a new file: handlers/implementations/your_dataset.py
2. Import from base_handler and handler_registry
3. Use @register_handler decorator on your handler class
4. Implement all abstract methods

Example:
    # handlers/implementations/your_dataset.py
    from milia_pipeline.handlers.base_handler import DatasetHandler, handle_transform_errors
    from milia_pipeline.handlers.handler_registry import register_handler
    
    @register_handler
    class YourDatasetHandler(DatasetHandler):
        def get_dataset_type(self) -> str:
            return "YourDataset"
        
        # ... implement all other abstract methods
"""

import importlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Modules to exclude from dynamic import (utility modules, not handler implementations)
_EXCLUDED_MODULES = {'__init__', 'base', 'registry', 'utils', 'common', 'base_handler', 'handler_registry'}

# Dynamic import: scan this directory for all .py files and import them
_implementations_dir = Path(__file__).parent
_discovered_classes = {}

for _py_file in _implementations_dir.glob('*.py'):
    _module_name = _py_file.stem
    if _module_name in _EXCLUDED_MODULES or _module_name.startswith('_'):
        continue
    
    try:
        # Import the module - this triggers @register_handler decorator on the handler class
        _module = importlib.import_module(f'.{_module_name}', package=__name__)
        
        # Find the handler class (convention: class name ends with 'DatasetHandler' or 'Handler')
        for _attr_name in dir(_module):
            if (_attr_name.endswith('DatasetHandler') or _attr_name.endswith('Handler')) and not _attr_name.startswith('_'):
                _cls = getattr(_module, _attr_name)
                # Verify it's a proper handler class with get_dataset_type method
                if hasattr(_cls, 'get_dataset_type') and isinstance(_cls, type):
                    _discovered_classes[_attr_name] = _cls
                    logger.debug(f"Dynamically loaded handler: {_attr_name} from {_module_name}.py")
    except Exception as e:
        logger.warning(f"Failed to import handler module '{_module_name}': {e}")

# Export all discovered classes to module namespace
globals().update(_discovered_classes)

# Dynamically build __all__ from discovered classes
__all__ = list(_discovered_classes.keys())
