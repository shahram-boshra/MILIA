# milia_pipeline/handlers/__init__.py

"""
Dataset Handlers Package
=========================

This package provides comprehensive dataset handling capabilities for the milia pipeline,
including handler pattern implementations, transformation system integration, and
experimental workflow support.

REFACTORED ARCHITECTURE (Handler Module Refactoring v1.1.0 + Phase 7):
----------------------------------------------------------------------
The handlers module has been refactored from a single monolithic file (dataset_handlers.py)
to a modular structure:

    handlers/
    ├── __init__.py                      # This file - backward compatible exports
    ├── base_handler.py                  # DatasetHandler ABC + factory functions + utilities
    ├── handler_registry.py              # HandlerRegistry + @register_handler decorator
    ├── dataset_handler_integration.py   # TransformAwareHandlerIntegrator (unchanged)
    └── implementations/                 # Individual handler implementations
        ├── __init__.py                  # Dynamic discovery
        ├── dft.py                       # DFTDatasetHandler
        ├── dmc.py                       # DMCDatasetHandler
        ├── wavefunction.py              # WavefunctionDatasetHandler
        ├── qm9.py                       # QM9DatasetHandler
        ├── ani1x.py                     # ANI1xDatasetHandler
        ├── ani1ccx.py                   # ANI1ccxDatasetHandler
        ├── ani2x.py                     # ANI2xDatasetHandler
        └── rmd17.py                     # RMD17DatasetHandler

Phase 7 Migration Complete:
- dataset_handlers.py has been REMOVED
- All factory functions migrated to base_handler.py
- Handler classes are in implementations/

BACKWARD COMPATIBILITY:
-----------------------
All existing import patterns continue to work:
    from milia_pipeline.handlers import DFTDatasetHandler
    from milia_pipeline.handlers import create_dataset_handler
    from milia_pipeline.handlers import DatasetHandler

The module uses lazy loading via __getattr__ to:
1. Import handler classes from implementations/ (modular structure)
2. Import factory functions from base_handler.py

This ensures zero-breaking-changes after the Phase 7 migration.

ADDING NEW HANDLERS:
-------------------
After refactoring, adding a new handler requires only:
1. Create handlers/implementations/your_dataset.py
2. Use @register_handler decorator
3. No changes to __init__.py needed (dynamic discovery)

Core Components
---------------

Handler Classes:
    - DatasetHandler: Abstract base class for all dataset handlers
    - DFTDatasetHandler: Handler for DFT quantum chemistry datasets
    - DMCDatasetHandler: Handler for Diffusion Monte Carlo datasets
    - WavefunctionDatasetHandler: Handler for wavefunction-based datasets
    - QM9DatasetHandler: Handler for QM9 dataset
    - ANI1xDatasetHandler: Handler for ANI-1x dataset
    - ANI1ccxDatasetHandler: Handler for ANI-1ccx dataset
    - ANI2xDatasetHandler: Handler for ANI-2x dataset
    - RMD17DatasetHandler: Handler for rMD17 dataset

Factory Functions:
    - create_dataset_handler: Factory function to create appropriate handler instances

Integration Classes:
    - TransformAwareHandlerIntegrator: Integration class for handler-transform workflows

Error Handling:
    - handle_transform_errors: Decorator for transform error handling in handlers

Phase 6 - Registry Integration:
    - verify_handler_abstraction: Verification function for handler abstraction status
    - get_handler_abstraction_summary: Summary of handler abstraction features

Phase 7 - Registry Integration for Integration Module:
    - get_registry_integration_status: Status of registry integration for diagnostics
"""

# =============================================================================
# Circular Import Resolution - Pure Lazy Loading
# =============================================================================
#
# PROBLEM (Historical): Circular dependency chain existed with dataset_handlers.py
#
# SOLUTION: Pure lazy loading using __getattr__ without ANY imports at module
# load time. All imports are deferred until an attribute is actually accessed.
#
# Phase 7 Migration: dataset_handlers.py has been REMOVED. All factory functions
# are now in base_handler.py, and all handler classes are in implementations/.
# =============================================================================

# Handler class names for lazy loading - LEGACY FALLBACK ONLY
# NOTE: This set is used ONLY as a fallback when implementations/ is unavailable.
# Dynamic discovery from implementations/__init__.py takes priority.
# New handlers do NOT need to be added here - just create the file in implementations/
_HANDLER_CLASSES_FALLBACK = {
    'DatasetHandler',
    'DFTDatasetHandler', 
    'DMCDatasetHandler',
    'WavefunctionDatasetHandler', 
    'QM9DatasetHandler',
    'ANI1xDatasetHandler',
    'ANI1ccxDatasetHandler',
    'ANI2xDatasetHandler',
    'RMD17DatasetHandler',
}

# Cache for dynamically discovered handler classes from implementations/
_DISCOVERED_HANDLER_CLASSES = None
# Recursion guard to prevent infinite loop during implementations import
_DISCOVERING_HANDLERS = False

def _get_handler_classes():
    """
    Dynamically discover available handler classes from implementations module.
    
    This function queries the implementations/__init__.py which uses dynamic
    discovery to find all handler classes. Results are cached for performance.
    
    Returns:
        set: Set of handler class names available for import
    """
    global _DISCOVERED_HANDLER_CLASSES, _DISCOVERING_HANDLERS
    
    if _DISCOVERED_HANDLER_CLASSES is not None:
        return _DISCOVERED_HANDLER_CLASSES
    
    # Recursion guard: if we're already discovering, return fallback
    if _DISCOVERING_HANDLERS:
        return _HANDLER_CLASSES_FALLBACK
    
    _DISCOVERING_HANDLERS = True
    try:
        from . import implementations as _impl
        # Get all dynamically discovered classes from implementations/__init__.py
        # The implementations module builds __all__ dynamically from discovered handlers
        discovered = set(getattr(_impl, '__all__', []))
        # Also add DatasetHandler which comes from base_handler, not implementations
        discovered.add('DatasetHandler')
        _DISCOVERED_HANDLER_CLASSES = discovered
        return discovered
    except ImportError:
        # Fall back to static list if implementations module fails to import
        _DISCOVERED_HANDLER_CLASSES = _HANDLER_CLASSES_FALLBACK.copy()
        return _DISCOVERED_HANDLER_CLASSES
    finally:
        _DISCOVERING_HANDLERS = False

def _is_handler_class(name: str) -> bool:
    """
    Check if a name is a handler class (dynamically or via fallback).
    
    Args:
        name: Name to check
        
    Returns:
        bool: True if name is a handler class
    """
    # First check dynamic discovery (with recursion guard)
    handler_classes = _get_handler_classes()
    if name in handler_classes:
        return True
    
    # Also check if name follows handler naming convention
    # Note: We check the naming pattern WITHOUT importing implementations
    # to avoid recursion during the import process
    if name.endswith('DatasetHandler') or name.endswith('Handler'):
        # If we're in the middle of discovering, just check the name pattern
        if _DISCOVERING_HANDLERS:
            return True
        try:
            from . import implementations as _impl
            if hasattr(_impl, name):
                return True
        except ImportError:
            pass
    
    return False

# Base handler utilities
_BASE_HANDLER_ATTRS = {
    'handle_transform_errors',
    '_init_registry',
    '_get_available_handler_types',
    '_is_handler_type_registered',
    'get_registry_status',
    # Factory and utility functions (migrated from dataset_handlers.py - Phase 7)
    'create_dataset_handler',
    'validate_dataset_handler_compatibility',
    'filter_descriptors_by_handler_support',
    'verify_handler_abstraction',
    'get_handler_abstraction_summary',
}

# Handler registry exports
_HANDLER_REGISTRY_ATTRS = {
    'HandlerRegistry',
    'HandlerRegistrationError',
    'HandlerNotFoundError',
    'register_handler',
    'get_default_registry',
}


def __getattr__(name):
    """
    Lazy import mechanism to resolve circular dependencies.
    
    This function intercepts attribute access and imports the requested
    symbol only when needed, breaking the circular import chain.
    
    Import priority (Phase 7 - dataset_handlers.py REMOVED):
    1. Try implementations/ (refactored modular structure) - DYNAMIC DISCOVERY
    2. Fall back to base_handler.py (factory functions + utilities)
    3. Fall back to handler_registry.py (registry exports)
    4. Try dataset_handler_integration.py
    
    DYNAMIC HANDLER DISCOVERY:
    Handler classes are discovered dynamically from implementations/__init__.py,
    which scans for all .py files and imports them. No manual updates to this
    file are needed when adding new handlers - just create the file in
    implementations/ with @register_handler decorator.
    
    Args:
        name: Name of the attribute being accessed
        
    Returns:
        The requested attribute from the appropriate submodule
        
    Raises:
        AttributeError: If the attribute doesn't exist
    """
    # Handler classes - use dynamic discovery from implementations/
    if _is_handler_class(name):
        # DatasetHandler ABC comes from base_handler
        if name == 'DatasetHandler':
            from .base_handler import DatasetHandler
            return DatasetHandler
        
        # All other handler classes come from implementations/
        try:
            from . import implementations as _impl
            if hasattr(_impl, name):
                return getattr(_impl, name)
        except ImportError as e:
            raise AttributeError(
                f"Handler class '{name}' not found. "
                f"Import error: {e}. "
                f"Ensure the handler is defined in handlers/implementations/"
            )
        
        # Handler class not found in implementations
        raise AttributeError(
            f"Handler class '{name}' not found in implementations/. "
            f"Available handlers: {list(_get_handler_classes())}"
        )
    
    # Base handler utilities (includes factory functions after Phase 7 migration)
    if name in _BASE_HANDLER_ATTRS:
        from . import base_handler as _bh
        return getattr(_bh, name)
    
    # Handler registry
    if name in _HANDLER_REGISTRY_ATTRS:
        from . import handler_registry as _hr
        return getattr(_hr, name)
    
    # NOTE: create_dataset_handler, validate_dataset_handler_compatibility,
    # filter_descriptors_by_handler_support, verify_handler_abstraction, and
    # get_handler_abstraction_summary are now in base_handler.py and routed
    # via _BASE_HANDLER_ATTRS above (Phase 7 migration).
    
    # Import integration module lazily - Classes
    if name == 'TransformAwareHandlerIntegrator':
        from . import dataset_handler_integration as _dhi
        return getattr(_dhi, name)
    
    # Import integration module lazily - Phase 7 Registry Integration
    if name == 'get_registry_integration_status':
        from . import dataset_handler_integration as _dhi
        return getattr(_dhi, name)
    
    # Import integration module lazily - Demonstration Functions
    if name in ('demonstrate_experimental_setup_workflow',
                'demonstrate_multi_level_validation_complete',
                'demonstrate_dynamic_transform_discovery_workflow',
                'demonstrate_transform_error_handling',
                'demonstrate_config_migration_complete',
                'demonstrate_complete_phase2_workflow',
                'demonstrate_testing_patterns'):
        from . import dataset_handler_integration as _dhi
        return getattr(_dhi, name)
    
    # Import integration module lazily - Helper/Utility Functions
    if name in ('create_integration_checklist',
                'generate_benefits',
                'create_performance_guide',
                'generate_quick_reference_guide',
                'run_example_from_cli'):
        from . import dataset_handler_integration as _dhi
        return getattr(_dhi, name)
    
    # Module helper functions (defined below)
    if name == 'get_available_handlers':
        return get_available_handlers
    if name == 'get_handler_info':
        return get_handler_info
    
    # Unknown attribute
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    """
    Define available attributes for introspection (dir(), help(), autocomplete).
    
    Returns:
        list: All public attributes available in this module
    """
    return [
        # Handler Classes
        'DatasetHandler',
        'DFTDatasetHandler',
        'DMCDatasetHandler',
        'WavefunctionDatasetHandler',
        'QM9DatasetHandler',
        'ANI1xDatasetHandler',
        'ANI1ccxDatasetHandler',
        'ANI2xDatasetHandler',
        'RMD17DatasetHandler',
        
        # Factory Functions
        'create_dataset_handler',
        
        # Integration Classes
        'TransformAwareHandlerIntegrator',
        
        # Decorators
        'handle_transform_errors',
        
        # Registry
        'HandlerRegistry',
        'register_handler',
        'get_default_registry',
        
        # Phase 6: Verification Functions
        'verify_handler_abstraction',
        'get_handler_abstraction_summary',
        
        # Phase 7: Registry Integration Status
        'get_registry_integration_status',
        
        # Demonstration Functions
        'demonstrate_experimental_setup_workflow',
        'demonstrate_multi_level_validation_complete',
        'demonstrate_dynamic_transform_discovery_workflow',
        'demonstrate_transform_error_handling',
        'demonstrate_config_migration_complete',
        'demonstrate_complete_phase2_workflow',
        'demonstrate_testing_patterns',
        
        # Helper/Utility Functions
        'create_integration_checklist',
        'generate_benefits',
        'create_performance_guide',
        'generate_quick_reference_guide',
        'run_example_from_cli',
        
        # Module-Level Helper Functions
        'get_available_handlers',
        'get_handler_info',
    ]


# =============================================================================
# Public API Definition
# =============================================================================

__all__ = [
    # Handler Classes
    'DatasetHandler',
    'DFTDatasetHandler',
    'DMCDatasetHandler',
    'WavefunctionDatasetHandler',
    'QM9DatasetHandler',
    'ANI1xDatasetHandler',
    'ANI1ccxDatasetHandler',
    'ANI2xDatasetHandler',
    'RMD17DatasetHandler',
    
    # Factory Functions
    'create_dataset_handler',
    
    # Integration Classes
    'TransformAwareHandlerIntegrator',
    
    # Decorators
    'handle_transform_errors',
    
    # Registry
    'HandlerRegistry',
    'register_handler',
    'get_default_registry',
    
    # Phase 6: Verification Functions
    'verify_handler_abstraction',
    'get_handler_abstraction_summary',
    
    # Phase 7: Registry Integration Status
    'get_registry_integration_status',
    
    # Demonstration Functions
    'demonstrate_experimental_setup_workflow',
    'demonstrate_multi_level_validation_complete',
    'demonstrate_dynamic_transform_discovery_workflow',
    'demonstrate_transform_error_handling',
    'demonstrate_config_migration_complete',
    'demonstrate_complete_phase2_workflow',
    'demonstrate_testing_patterns',
    
    # Helper/Utility Functions
    'create_integration_checklist',
    'generate_benefits',
    'create_performance_guide',
    'generate_quick_reference_guide',
    'run_example_from_cli',
    
    # Module-Level Helper Functions
    'get_available_handlers',
    'get_handler_info',
]

# =============================================================================
# Module Metadata
# =============================================================================

__version__ = '3.0.0'  # Updated for Handler Module Refactoring
__author__ = 'Milia Pipeline Team'
__description__ = 'Modular dataset handlers with transformation system integration and registry support'

# =============================================================================
# Module-Level Helper Functions
# =============================================================================

def get_available_handlers():
    """
    Get a list of available dataset handler types.
    
    This function uses dynamic discovery to find all handlers:
    1. First tries the handler registry
    2. Then tries base handler registry integration
    3. Finally uses dynamic discovery from implementations/
    
    Returns:
        list: List of supported dataset types as strings
    
    Example:
        >>> from milia_pipeline.handlers import get_available_handlers
        >>> handlers = get_available_handlers()
        >>> print(handlers)
        ['DFT', 'DMC', 'Wavefunction', 'QM9', 'ANI1x', 'ANI1ccx', 'ANI2x', 'RMD17', 'XXMD', ...]
    """
    try:
        # Try handler registry first
        from .handler_registry import list_all
        registered = list_all()
        if registered:
            return registered
    except (ImportError, Exception):
        pass
    
    try:
        # Try base handler registry integration
        from .base_handler import _get_available_handler_types
        return _get_available_handler_types()
    except (ImportError, AttributeError):
        pass
    
    # Dynamic discovery from implementations - extract dataset types from class names
    try:
        handler_classes = _get_handler_classes()
        # Convert class names to dataset types (e.g., 'DFTDatasetHandler' -> 'DFT')
        types = []
        for cls_name in handler_classes:
            if cls_name == 'DatasetHandler':
                continue  # Skip base class
            if cls_name.endswith('DatasetHandler'):
                types.append(cls_name[:-len('DatasetHandler')])
            elif cls_name.endswith('Handler'):
                types.append(cls_name[:-len('Handler')])
        if types:
            return sorted(types)
    except Exception:
        pass
    
    # Static fallback (should rarely be reached)
    return ['DFT', 'DMC', 'Wavefunction', 'QM9', 'ANI1x', 'ANI1ccx', 'ANI2x', 'RMD17']


def get_handler_info(handler_type: str) -> dict:
    """
    Get information about a specific handler type.
    
    For statically defined handlers, returns detailed information.
    For dynamically discovered handlers, returns basic information
    queried from the handler class itself.
    
    Args:
        handler_type: Type of handler ('DFT', 'DMC', 'Wavefunction', 'QM9', etc.)
    
    Returns:
        dict: Handler information including supported features and capabilities
    
    Example:
        >>> from milia_pipeline.handlers import get_handler_info
        >>> info = get_handler_info('DFT')
        >>> print(info['description'])
        Handler for DFT quantum chemistry datasets
    """
    # Static handler info for well-known handlers
    handler_info_map = {
        'DFT': {
            'class': 'DFTDatasetHandler',
            'module': 'implementations.dft',
            'description': 'Handler for DFT quantum chemistry datasets',
            'supports_all_features': True,
            'uncertainty_support': False,
            'molecule_creation_strategy': 'identifier_coordinate_based',
            'typical_properties': [
                'Etot', 'U0', 'H', 'G', 'Cv', 'zpves', 'gap',
                'dipole', 'quadrupole', 'freqs', 'vibmodes'
            ],
            'structural_features': 'All atom and bond features supported',
        },
        'DMC': {
            'class': 'DMCDatasetHandler',
            'module': 'implementations.dmc',
            'description': 'Handler for Diffusion Monte Carlo datasets',
            'supports_all_features': False,
            'uncertainty_support': True,
            'molecule_creation_strategy': 'identifier_coordinate_based',
            'typical_properties': [
                'Etot', 'std', 'Eatomization'
            ],
            'structural_features': 'Limited features (excludes charges, bond lengths)',
            'notes': 'Conservative feature set to prevent NaN/Inf warnings',
        },
        'Wavefunction': {
            'class': 'WavefunctionDatasetHandler',
            'module': 'implementations.wavefunction',
            'description': 'Handler for wavefunction-based datasets',
            'supports_all_features': True,
            'uncertainty_support': False,
            'molecule_creation_strategy': 'coordinate_based',
            'typical_properties': [
                'orbital_energies', 'orbital_coefficients', 
                'HOMO', 'LUMO', 'occupations'
            ],
            'structural_features': 'All features supported with wavefunction data',
            'notes': 'Requires preprocessed wavefunction data',
        },
        'QM9': {
            'class': 'QM9DatasetHandler',
            'module': 'implementations.qm9',
            'description': 'Handler for QM9 quantum chemistry datasets',
            'supports_all_features': True,
            'uncertainty_support': False,
            'molecule_creation_strategy': 'identifier_coordinate_based',
            'typical_properties': [
                'U0', 'U', 'H', 'G', 'zpve', 'homo', 'lumo', 'gap',
                'mu', 'alpha', 'Cv', 'A', 'B', 'C', 'r2', 'freqs', 'Qmulliken'
            ],
            'structural_features': 'All atom and bond features supported',
            'notes': 'QM9 dataset with B3LYP/6-31G(2df,p) properties',
        },
        'ANI1x': {
            'class': 'ANI1xDatasetHandler',
            'module': 'implementations.ani1x',
            'description': 'Handler for ANI-1x quantum chemistry datasets',
            'supports_all_features': True,
            'uncertainty_support': False,
            'molecule_creation_strategy': 'coordinate_based',
            'typical_properties': ['energy', 'forces', 'hirshfeld_charges', 'cm5_charges', 'dipole'],
            'structural_features': 'All features supported',
            'notes': 'NO parseable identifiers - uses coordinate_based molecule creation',
        },
        'ANI1ccx': {
            'class': 'ANI1ccxDatasetHandler',
            'module': 'implementations.ani1ccx',
            'description': 'Handler for ANI-1ccx quantum chemistry datasets',
            'supports_all_features': True,
            'uncertainty_support': False,
            'molecule_creation_strategy': 'coordinate_based',
            'typical_properties': ['energy'],
            'structural_features': 'All features supported',
            'notes': 'CCSD(T)/CBS level energies, NO parseable identifiers',
        },
        'ANI2x': {
            'class': 'ANI2xDatasetHandler',
            'module': 'implementations.ani2x',
            'description': 'Handler for ANI-2x quantum chemistry datasets',
            'supports_all_features': True,
            'uncertainty_support': False,
            'molecule_creation_strategy': 'coordinate_based',
            'typical_properties': ['energy', 'forces'],
            'structural_features': 'All features supported',
            'notes': 'Supports 7 elements (H,C,N,O,S,F,Cl), NO parseable identifiers',
        },
        'RMD17': {
            'class': 'RMD17DatasetHandler',
            'module': 'implementations.rmd17',
            'description': 'Handler for rMD17 quantum chemistry datasets',
            'supports_all_features': True,
            'uncertainty_support': False,
            'molecule_creation_strategy': 'coordinate_based',
            'typical_properties': ['energy', 'forces'],
            'structural_features': 'All features supported',
            'notes': 'Energy in kcal/mol (NOT Hartree!), MD trajectory data',
        }
    }
    
    # Return static info if available
    if handler_type in handler_info_map:
        return handler_info_map[handler_type]
    
    # Dynamic discovery: Try to get info from the handler class itself
    handler_class_name = f"{handler_type}DatasetHandler"
    try:
        from . import implementations as _impl
        if hasattr(_impl, handler_class_name):
            handler_class = getattr(_impl, handler_class_name)
            # Build info from handler class methods
            info = {
                'class': handler_class_name,
                'module': f'implementations.{handler_type.lower()}',
                'description': f'Handler for {handler_type} quantum chemistry datasets',
                'supports_all_features': True,
                'uncertainty_support': False,
                'dynamically_discovered': True,
            }
            # Try to get molecule creation strategy from handler
            if hasattr(handler_class, 'get_molecule_creation_strategy'):
                try:
                    # Create temp instance or use class method
                    info['molecule_creation_strategy'] = 'coordinate_based'  # Default for new handlers
                except Exception:
                    pass
            return info
    except (ImportError, AttributeError):
        pass
    
    # Handler not found
    available = get_available_handlers()
    raise ValueError(
        f"Unknown handler type: {handler_type}. "
        f"Available types: {available}"
    )


# =============================================================================
# Package Usage Notes
# =============================================================================

# IMPORTANT: Actual handler instances should be created via the
# create_dataset_handler() factory function, not by direct instantiation.
# This ensures proper configuration validation and compatibility checking.
#
# Correct usage:
#   handler = create_dataset_handler(dataset_config, filter_config, 
#                                    processing_config, logger)
#
# The factory pattern ensures:
#   - Proper handler type selection based on dataset_config
#   - Configuration validation before handler creation
#   - Consistent initialization across all handler types
#   - Compatibility checking between handler and dataset
#   - Registry-based dynamic handler resolution when available
#
# For transform-aware workflows, use TransformAwareHandlerIntegrator:
#   integrator = TransformAwareHandlerIntegrator(
#       dataset_config, filter_config, processing_config, logger,
#       experimental_setup="baseline",
#       enable_caching=True
#   )
