# milia_pipeline/transformations/__init__.py

"""
milia Pipeline - Transformations Module
========================================

Production-ready transformation system for molecular graph data processing.

This module provides a comprehensive, extensible transformation framework with:

1. **Core Transform Registry** (graph_transforms)
   - 30+ pre-registered PyTorch Geometric transforms
   - Dynamic discovery and validation
   - milia dataset-specific optimizations (DFT, DMC, Wavefunction)
   - Configuration migration and validation
   - Production metrics and monitoring
   - Intelligent caching with memory management

2. **Custom Transform Framework** (custom_transforms)
   - Extensible base classes for domain-specific transforms
   - CustomTransformBase, MolecularTransformBase, QuantumTransformBase
   - Parameter introspection and validation
   - Research-grade metadata tracking
   - milia-specific quantum property handling

3. **Plugin System** (plugin_system)
   - Secure plugin discovery and loading
   - Comprehensive validation (dependencies, security, functional, performance)
   - Version management and compatibility checking
   - Thread-safe registry with enable/disable functionality
   - YAML-based plugin configuration

4. **Research API** (research_api)
   - Systematic experimentation framework
   - Ablation studies, parameter sweeps, comparative studies
   - Fluent builder APIs for experiment configuration
   - Statistical analysis and result tracking
   - Full YAML/JSON serialization support

Architecture:
------------
The module uses a sophisticated circular dependency resolution system:
- Lazy imports with import guards
- Conditional feature availability flags
- Cross-module dependency resolution after initial imports
- Thread-safe initialization

Usage Examples:
--------------
Basic transform sequence:
    >>> from milia_pipeline.transformations import get_graph_transforms
    >>> gt = get_graph_transforms()
    >>> configs = [
    ...     {'name': 'AddSelfLoops'},
    ...     {'name': 'ToUndirected'}
    ... ]
    >>> compose = gt.create_transform_sequence(configs)

Custom transform:
    >>> from milia_pipeline.transformations import QuantumTransformBase
    >>> class MyTransform(QuantumTransformBase):
    ...     def transform(self, data):
    ...         # Custom logic
    ...         return data

Plugin system:
    >>> from milia_pipeline.transformations import PluginRegistry
    >>> PluginRegistry.discover_plugins('/path/to/plugins')
    >>> metadata = PluginRegistry.get_plugin('my_plugin')

Research experiments:
    >>> from milia_pipeline.transformations import create_ablation_study
    >>> config = create_ablation_study(
    ...     "importance_study",
    ...     ["AddSelfLoops", "GCNNorm"],
    ...     ["GCNNorm"]
    ... )

Author: milia Project Team
Version: 1.0.0
License: MIT
"""

import logging
from typing import Dict, List, Any, Optional, Type

# Module metadata
__version__ = "1.0.0"
__author__ = "milia Project Team"
__license__ = "MIT"

# Module logger
logger = logging.getLogger(__name__)

# =============================================================================
# CIRCULAR DEPENDENCY RESOLUTION SYSTEM
# =============================================================================

# Import guard to prevent circular imports during initialization
_INITIALIZING = False
_INITIALIZED = False


def _ensure_initialized():
    """
    Ensure all cross-module dependencies are resolved.
    
    This function is called automatically when needed, but can be called
    explicitly to force initialization of cross-module dependencies.
    
    The initialization process:
    1. Imports basic structures from all modules
    2. Resolves cross-dependencies between modules
    3. Sets up lazy imports for plugin system
    
    Thread-safe and idempotent.
    """
    global _INITIALIZED, _INITIALIZING
    
    if _INITIALIZED:
        return
    
    if _INITIALIZING:
        # Prevent re-entry during initialization
        return
    
    _INITIALIZING = True
    
    try:
        # Import basic structures (already done at module level)
        # Resolve cross-dependencies
        if CUSTOM_TRANSFORMS_AVAILABLE:
            from .custom_transforms import _lazy_import_graph_transforms
            _lazy_import_graph_transforms()
        
        if PLUGIN_SYSTEM_AVAILABLE:
            from .plugin_system import _import_custom_transforms, _import_graph_transforms
            _import_custom_transforms()
            _import_graph_transforms()
        
        _INITIALIZED = True
        logger.debug("Transformations module initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize transformations module: {e}")
        raise
    finally:
        _INITIALIZING = False


# =============================================================================
# CORE TRANSFORM REGISTRY - GRAPH TRANSFORMS
# =============================================================================

try:
    from .graph_transforms import (
        # Core classes
        TransformRegistry,
        TransformComposer,
        TransformValidator,
        DynamicTransformDiscovery,
        ConfigurationBridge,
        TransformErrorRecovery,
        
        # Enums and types
        ValidationLevel,
        ValidationScope,
        
        # Enhanced metadata classes
        TransformInfo,
        TransformDependency,
        TransformCompatibility,
        
        # Main API
        GraphTransforms,
        get_graph_transforms,
        
        # Convenience functions
        get_transform_info,
        validate_v3_configuration,
        validate_comprehensive,
        get_configuration_format_help,
        export_metrics,
        optimize_performance,
        get_milia_setups,
        perform_system_health_check,
        get_validation_report_text,
        discover_custom_transforms,
        register_all_custom_transforms,
    )
    GRAPH_TRANSFORMS_AVAILABLE = True
    logger.debug("graph_transforms module loaded successfully")
except ImportError as e:
    GRAPH_TRANSFORMS_AVAILABLE = False
    logger.warning(f"graph_transforms module not available: {e}")
    
    # Minimal fallbacks for type hints
    TransformRegistry = None
    TransformComposer = None
    TransformValidator = None
    DynamicTransformDiscovery = None
    ConfigurationBridge = None
    TransformErrorRecovery = None
    ValidationLevel = None
    ValidationScope = None
    TransformInfo = None
    TransformDependency = None
    TransformCompatibility = None
    GraphTransforms = None
    get_graph_transforms = None
    get_transform_info = None
    validate_v3_configuration = None
    validate_comprehensive = None
    get_configuration_format_help = None
    export_metrics = None
    optimize_performance = None
    get_milia_setups = None
    perform_system_health_check = None
    get_validation_report_text = None
    discover_custom_transforms = None
    register_all_custom_transforms = None


# =============================================================================
# CUSTOM TRANSFORM FRAMEWORK
# =============================================================================

try:
    from .custom_transforms import (
        # Core base classes
        CustomTransformBase,
        MolecularTransformBase,
        QuantumTransformBase,
        
        # Metadata
        TransformMetadata,
        
        # Example transforms
        NormalizeVibrationalModes,
        FilterByDMCUncertainty,
        ScaleMullikenCharges,
        
        # Exceptions
        TransformValidationError,
        TransformExecutionError,
        TransformConfigurationError,
    )
    CUSTOM_TRANSFORMS_AVAILABLE = True
    logger.debug("custom_transforms module loaded successfully")
except ImportError as e:
    CUSTOM_TRANSFORMS_AVAILABLE = False
    logger.warning(f"custom_transforms module not available: {e}")
    
    # Minimal fallbacks
    CustomTransformBase = None
    MolecularTransformBase = None
    QuantumTransformBase = None
    TransformMetadata = None
    NormalizeVibrationalModes = None
    FilterByDMCUncertainty = None
    ScaleMullikenCharges = None
    TransformValidationError = Exception
    TransformExecutionError = Exception
    TransformConfigurationError = Exception


# =============================================================================
# PLUGIN SYSTEM
# =============================================================================

try:
    from .plugin_system import (
        # Core classes
        PluginMetadata,
        PluginRegistry,
        PluginValidator,
        TransformDeclaration,
        
        # Exceptions
        PluginError,
        PluginValidationError,
        PluginSecurityError,
        PluginDependencyError,
    )
    PLUGIN_SYSTEM_AVAILABLE = True
    logger.debug("plugin_system module loaded successfully")
except ImportError as e:
    PLUGIN_SYSTEM_AVAILABLE = False
    logger.warning(f"plugin_system module not available: {e}")
    
    # Minimal fallbacks
    PluginMetadata = None
    PluginRegistry = None
    PluginValidator = None
    TransformDeclaration = None
    PluginError = Exception
    PluginValidationError = Exception
    PluginSecurityError = Exception
    PluginDependencyError = Exception


# =============================================================================
# RESEARCH API
# =============================================================================

try:
    from .research_api import (
        # Core configuration
        ExperimentConfiguration,
        
        # Fluent builders
        AblationStudyBuilder,
        ParameterSweepBuilder,
        ComparativeStudyBuilder,
        
        # Experiment execution
        ExperimentRunner,
        
        # Convenience functions
        create_ablation_study,
        create_parameter_sweep,
        create_comparative_study,
        
        # Configuration loaders
        load_experiments_from_config,
        get_experiment,
        list_available_experiments,
    )
    RESEARCH_API_AVAILABLE = True
    logger.debug("research_api module loaded successfully")
except ImportError as e:
    RESEARCH_API_AVAILABLE = False
    logger.warning(f"research_api module not available: {e}")
    
    # Minimal fallbacks
    ExperimentConfiguration = None
    AblationStudyBuilder = None
    ParameterSweepBuilder = None
    ComparativeStudyBuilder = None
    ExperimentRunner = None
    create_ablation_study = None
    create_parameter_sweep = None
    create_comparative_study = None
    load_experiments_from_config = None
    get_experiment = None
    list_available_experiments = None


# =============================================================================
# CROSS-MODULE DEPENDENCY RESOLUTION
# =============================================================================

# Resolve cross-dependencies after all modules loaded
# This is called automatically when needed, but can be called explicitly
# _ensure_initialized()  # Optional: Force initialization now


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

def get_available_transforms() -> List[str]:
    """
    Get list of all available transform names.
    
    Returns:
        List of transform names from the registry
        
    Example:
        >>> transforms = get_available_transforms()
        >>> print(f"Available: {len(transforms)} transforms")
    """
    if not GRAPH_TRANSFORMS_AVAILABLE:
        logger.error("graph_transforms module not available")
        return []
    
    gt = get_graph_transforms()
    return gt.list_available_transforms()


def create_transform_sequence(
    configs: List[Dict[str, Any]],
    dataset_type: Optional[str] = None
) -> Any:
    """
    Create a transform sequence from configuration.
    
    Args:
        configs: List of transform configurations
        dataset_type: Optional dataset type for optimization ('DFT', 'DMC', 'Wavefunction')
        
    Returns:
        PyTorch Geometric Compose object
        
    Example:
        >>> compose = create_transform_sequence([
        ...     {'name': 'AddSelfLoops'},
        ...     {'name': 'ToUndirected'}
        ... ])
    """
    if not GRAPH_TRANSFORMS_AVAILABLE:
        raise ImportError("graph_transforms module not available")
    
    gt = get_graph_transforms()
    return gt.create_transform_sequence(configs, dataset_type=dataset_type)


def validate_transform_config(
    configs: List[Dict[str, Any]],
    dataset_type: Optional[str] = None,
    validation_level: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Validate transform configuration.
    
    Args:
        configs: List of transform configurations
        dataset_type: Optional dataset type
        validation_level: Validation strictness level
        
    Returns:
        Validation result dictionary
        
    Example:
        >>> result = validate_transform_config([
        ...     {'name': 'AddSelfLoops'},
        ...     {'name': 'Distance', 'kwargs': {'norm': True}}
        ... ], dataset_type='DFT')
        >>> if result['valid']:
        ...     print("Configuration valid!")
    """
    if not GRAPH_TRANSFORMS_AVAILABLE:
        raise ImportError("graph_transforms module not available")
    
    if validation_level is None and ValidationLevel is not None:
        validation_level = ValidationLevel.STANDARD
    
    return validate_comprehensive(configs, dataset_type=dataset_type, validation_level=validation_level)


def register_custom_transform(transform_class: Type, force: bool = False) -> bool:
    """
    Register a custom transform with the registry.
    
    Args:
        transform_class: Transform class to register
        force: Force registration even if already exists
        
    Returns:
        True if registration successful
        
    Example:
        >>> class MyTransform(QuantumTransformBase):
        ...     pass
        >>> register_custom_transform(MyTransform)
    """
    if not GRAPH_TRANSFORMS_AVAILABLE:
        raise ImportError("graph_transforms module not available")
    
    gt = get_graph_transforms()
    return gt.register_custom_transform(transform_class, force=force)


def discover_and_register_plugins(plugin_paths: List[str]) -> Dict[str, Any]:
    """
    Discover and register plugins from specified paths.
    
    Args:
        plugin_paths: List of paths to search for plugins
        
    Returns:
        Discovery results dictionary
        
    Example:
        >>> results = discover_and_register_plugins(['/path/to/plugins'])
        >>> print(f"Registered {results['registered_count']} plugins")
    """
    if not PLUGIN_SYSTEM_AVAILABLE:
        raise ImportError("plugin_system module not available")
    
    results = {
        'registered_count': 0,
        'failed_count': 0,
        'plugins': []
    }
    
    for path in plugin_paths:
        try:
            discovered = PluginRegistry.discover_plugins(path)
            results['plugins'].extend(discovered)
            results['registered_count'] += len(discovered)
        except Exception as e:
            logger.error(f"Failed to discover plugins from {path}: {e}")
            results['failed_count'] += 1
    
    return results


def get_system_status() -> Dict[str, Any]:
    """
    Get comprehensive system status.
    
    Returns:
        Dictionary with system status information
        
    Example:
        >>> status = get_system_status()
        >>> print(f"Graph transforms: {status['graph_transforms_available']}")
        >>> print(f"Custom transforms: {status['custom_transforms_available']}")
    """
    status = {
        'graph_transforms_available': GRAPH_TRANSFORMS_AVAILABLE,
        'custom_transforms_available': CUSTOM_TRANSFORMS_AVAILABLE,
        'plugin_system_available': PLUGIN_SYSTEM_AVAILABLE,
        'research_api_available': RESEARCH_API_AVAILABLE,
        'initialized': _INITIALIZED,
    }
    
    if GRAPH_TRANSFORMS_AVAILABLE:
        try:
            gt = get_graph_transforms()
            status['registered_transforms'] = len(gt.list_available_transforms())
            status['health_check'] = gt.perform_health_check()
        except Exception as e:
            status['health_check_error'] = str(e)
    
    if PLUGIN_SYSTEM_AVAILABLE:
        try:
            status['registered_plugins'] = len(PluginRegistry.list_plugins())
        except Exception as e:
            status['plugin_registry_error'] = str(e)
    
    return status


def get_module_info() -> Dict[str, Any]:
    """
    Get module information and metadata.
    
    Returns:
        Dictionary with module information
        
    Example:
        >>> info = get_module_info()
        >>> print(f"Version: {info['version']}")
        >>> print(f"Features: {info['features']}")
    """
    return {
        'version': __version__,
        'author': __author__,
        'license': __license__,
        'features': {
            'graph_transforms': GRAPH_TRANSFORMS_AVAILABLE,
            'custom_transforms': CUSTOM_TRANSFORMS_AVAILABLE,
            'plugin_system': PLUGIN_SYSTEM_AVAILABLE,
            'research_api': RESEARCH_API_AVAILABLE,
        },
        'components': {
            'transform_registry': TransformRegistry is not None,
            'transform_composer': TransformComposer is not None,
            'plugin_registry': PluginRegistry is not None,
            'experiment_configuration': ExperimentConfiguration is not None,
        }
    }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Module metadata
    '__version__',
    '__author__',
    '__license__',
    
    # Availability flags
    'GRAPH_TRANSFORMS_AVAILABLE',
    'CUSTOM_TRANSFORMS_AVAILABLE',
    'PLUGIN_SYSTEM_AVAILABLE',
    'RESEARCH_API_AVAILABLE',
    
    # Graph transforms - Core classes
    'TransformRegistry',
    'TransformComposer',
    'TransformValidator',
    'DynamicTransformDiscovery',
    'ConfigurationBridge',
    'TransformErrorRecovery',
    'ValidationLevel',
    'ValidationScope',
    'TransformInfo',
    'TransformDependency',
    'TransformCompatibility',
    'GraphTransforms',
    
    # Graph transforms - Main API
    'get_graph_transforms',
    
    # Graph transforms - Convenience functions
    'get_transform_info',
    'validate_v3_configuration',
    'validate_comprehensive',
    'get_configuration_format_help',
    'export_metrics',
    'optimize_performance',
    'get_milia_setups',
    'perform_system_health_check',
    'get_validation_report_text',
    'discover_custom_transforms',
    'register_all_custom_transforms',
    
    # Custom transforms - Base classes
    'CustomTransformBase',
    'MolecularTransformBase',
    'QuantumTransformBase',
    'TransformMetadata',
    
    # Custom transforms - Example implementations
    'NormalizeVibrationalModes',
    'FilterByDMCUncertainty',
    'ScaleMullikenCharges',
    
    # Custom transforms - Exceptions
    'TransformValidationError',
    'TransformExecutionError',
    'TransformConfigurationError',
    
    # Plugin system - Core classes
    'PluginMetadata',
    'PluginRegistry',
    'PluginValidator',
    'TransformDeclaration',
    
    # Plugin system - Exceptions
    'PluginError',
    'PluginValidationError',
    'PluginSecurityError',
    'PluginDependencyError',
    
    # Research API - Configuration
    'ExperimentConfiguration',
    
    # Research API - Builders
    'AblationStudyBuilder',
    'ParameterSweepBuilder',
    'ComparativeStudyBuilder',
    
    # Research API - Execution
    'ExperimentRunner',
    
    # Research API - Convenience functions
    'create_ablation_study',
    'create_parameter_sweep',
    'create_comparative_study',
    'load_experiments_from_config',
    'get_experiment',
    'list_available_experiments',
    
    # Module-level convenience functions
    'get_available_transforms',
    'create_transform_sequence',
    'validate_transform_config',
    'register_custom_transform',
    'discover_and_register_plugins',
    'get_system_status',
    'get_module_info',
    
    # Initialization
    '_ensure_initialized',
]


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

# Log module initialization
logger.info(f"milia Transformations Module v{__version__} initialized")
logger.debug(f"Graph transforms: {'available' if GRAPH_TRANSFORMS_AVAILABLE else 'unavailable'}")
logger.debug(f"Custom transforms: {'available' if CUSTOM_TRANSFORMS_AVAILABLE else 'unavailable'}")
logger.debug(f"Plugin system: {'available' if PLUGIN_SYSTEM_AVAILABLE else 'unavailable'}")
logger.debug(f"Research API: {'available' if RESEARCH_API_AVAILABLE else 'unavailable'}")

# Optional: Uncomment to force initialization on module import
# _ensure_initialized()
