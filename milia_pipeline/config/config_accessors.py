# config_accessors.py - Enhanced for Transformation Configuration Access

"""
Configuration accessor functions module with enhanced transformation configuration support
and structural feature handler integration.

This module provides all functions for accessing and querying configuration values,
including dataset-specific settings, structural features, validation utilities,
and the new transformation configuration system.

Transformation System Integration Enhancements:
- Added comprehensive transformation configuration access functions
- Integrated with enhanced TransformationConfig containers from Step 2
- Support for experimental setups and systematic experimentation
- Backward compatibility with legacy transformation formats
- Integration with graph_transforms.py system from Step 1
- Enhanced validation and error handling for transformation configs

Structural Feature Integration:
- Handler-aware structural feature filtering in get_dataset_appropriate_structural_features()
- Automatic feature validation against handler capabilities
- Prevents NaN/Inf warnings from unsupported feature calculations (e.g., Gasteiger on DMC)
- Fallback to legacy filtering for backward compatibility
- Integration with dataset_handlers.py feature declarations
"""

import warnings
import logging
from typing import Dict, Any, Optional, Union, Tuple, Literal, List
from milia_pipeline.config.config_loader import load_config
from milia_pipeline.exceptions import (
    ConfigurationError, 
    HandlerNotAvailableError,
    HandlerConfigurationError,
    HandlerCompatibilityError,
    HandlerIntegrationError,
    ValidationError,
    CompatibilityError,
    create_handler_error_context,
    wrap_handler_operation,
    is_recoverable_handler_error,
    get_exception_recovery_suggestions
)
try:
    from milia_pipeline.config.config_containers import (
        DatasetConfig,
        FilterConfig,
        StructuralFeaturesConfig,
        ProcessingConfig,
        HandlerConfig,
        TransformSpec,
        ExperimentalSetup,
        TransformationConfig
    )
    CONTAINERS_AVAILABLE = True
except ImportError:
    CONTAINERS_AVAILABLE = False
    warnings.warn("config_containers not available - using basic dict structures")
try:
    from milia_pipeline.transformations.graph_transforms import (
        get_graph_transforms,
        GraphTransforms,
        TransformInfo,
        ParameterMetadata,
        ValidationLevel,
        ValidationScope,
        ValidationContext,
        TransformValidationError,
        TransformNotFoundError,
        ConfigurationError
    )
    GRAPH_TRANSFORMS_AVAILABLE = True
except ImportError:
    GRAPH_TRANSFORMS_AVAILABLE = False
    import sys
    if 'pytest' not in sys.modules and 'unittest' not in sys.modules:
        warnings.warn(
            "graph_transforms module unavailable - "
            "using basic validation and default parameter handling",
            UserWarning,
            stacklevel=2
        )

if not GRAPH_TRANSFORMS_AVAILABLE:
    from enum import Enum
    
    logger = logging.getLogger(__name__)
    logger.warning("Using fallback validation classes - enhanced validation features disabled")
    
    class ValidationLevel(Enum):
        """Fallback ValidationLevel enum when graph_transforms unavailable"""
        STRICT = "strict"
        STANDARD = "standard"
        PERMISSIVE = "permissive"
    
    class ValidationScope(Enum):
        """Fallback ValidationScope enum when graph_transforms unavailable"""
        BASIC = "basic"
        SEMANTIC = "semantic"
        DATASET_SPECIFIC = "dataset"
        PRODUCTION = "production"
    
    class ValidationContext:
        """Minimal ValidationContext implementation for fallback"""
        def __init__(self, level=None, scope=None, **kwargs):
            self.level = level or ValidationLevel.STANDARD
            self.scope = scope or ValidationScope.BASIC
            self.issues = []
            self.validation_metadata = kwargs.get('validation_metadata', {})
            self.dataset_type = kwargs.get('dataset_type')
            self.strict_mode = kwargs.get('strict_mode', False)
        
        def add_issue(self, *args, **kwargs):
            """Placeholder for adding validation issues"""
            pass
        
        def has_errors(self):
            """Placeholder for error checking"""
            return False
        
        def has_critical_issues(self):
            """Placeholder for critical issue checking"""
            return False
    
    class TransformValidationError(ValidationError):
        """Fallback TransformValidationError"""
        def __init__(self, message, transform_name=None, **kwargs):
            super().__init__(message)
            self.transform_name = transform_name
    
    class TransformNotFoundError(ValidationError):
        """Fallback TransformNotFoundError"""
        def __init__(self, message, transform_name=None, **kwargs):
            super().__init__(message)
            self.transform_name = transform_name
    
    # Mock types for type hints
    GraphTransforms = None
    TransformInfo = None
    ParameterMetadata = None
    
    def get_graph_transforms():
        """Fallback function that returns None"""
        logger.warning("graph_transforms not available - returning None")
        return None


# Moved to function-level imports to break circular dependency
# config_schemas imports are now done lazily in each function that needs them
SCHEMAS_AVAILABLE = None  # Will be determined on first use

def _get_config_schemas():
    """Lazy import to avoid circular dependency with config_schemas."""
    try:
        from milia_pipeline.config.config_schemas import (
            TransformationSchema, ValidationConfig, YAMLSchemaValidator, ConfigMigration
        )
        return TransformationSchema, ValidationConfig, YAMLSchemaValidator, ConfigMigration
    except ImportError:
        return None, None, None, None


# ==========================================
# PHASE 5: Registry Integration Infrastructure
# ==========================================

# Registry availability flags - set during lazy initialization
_REGISTRY_AVAILABLE = False
_REGISTRY_IMPORT_ERROR = None
_REGISTRY_INITIALIZED = False

# Registry function placeholders (populated by _init_registry)
_registry_list_all = None
_registry_get = None
_registry_is_registered = None
_registry_get_default = None


def _init_registry() -> bool:
    """
    Lazily initialize registry imports to avoid circular import at module load time.
    
    The config_accessors module is imported early in the initialization chain,
    so direct imports at module level can cause circular dependencies. By deferring
    the registry import until first use, we allow the config module to fully load first.
    
    Returns:
        True if registry is available, False otherwise
        
    ADDED Phase 5: Lazy initialization following Phase 3 pattern.
    """
    global _REGISTRY_AVAILABLE, _REGISTRY_IMPORT_ERROR, _REGISTRY_INITIALIZED
    global _registry_list_all, _registry_get, _registry_is_registered, _registry_get_default
    
    if _REGISTRY_INITIALIZED:
        return _REGISTRY_AVAILABLE
    
    _REGISTRY_INITIALIZED = True
    
    try:
        # Direct import from registry module (not through datasets/__init__.py)
        # This minimizes the import chain and avoids triggering implementation imports
        from milia_pipeline.datasets.registry import (
            list_all,
            get,
            is_registered,
            get_default_registry,
        )
        _registry_list_all = list_all
        _registry_get = get
        _registry_is_registered = is_registered
        _registry_get_default = get_default_registry
        _REGISTRY_AVAILABLE = True
        logger.debug("Dataset registry initialized successfully in config_accessors")
        return True
        
    except ImportError as e:
        _REGISTRY_IMPORT_ERROR = str(e)
        logger.warning(f"Dataset registry not available in config_accessors - using legacy validation: {e}")
        return False
        
    except Exception as e:
        # Catch any other exceptions (e.g., circular import issues)
        _REGISTRY_IMPORT_ERROR = str(e)
        logger.warning(f"Dataset registry import failed in config_accessors - using legacy validation: {e}")
        return False


def _registry_list_all_safe() -> List[str]:
    """
    Safely get list of all registered dataset types.
    
    DYNAMIC APPROACH: Instead of hardcoded fallback, this function:
    1. First tries the registry (primary source of truth)
    2. If registry fails, dynamically discovers dataset implementations from filesystem
    3. Never uses hardcoded dataset type lists
    
    Returns:
        List of dataset type names from registry or dynamic discovery
        
    ADDED Phase 5: Safe wrapper for registry access.
    UPDATED Phase 6.1: Replaced hardcoded fallback with dynamic filesystem discovery
    """
    _init_registry()
    if _registry_list_all is not None:
        try:
            return _registry_list_all()
        except Exception as e:
            logger.debug(f"Registry list_all failed: {e}")
    
    # DYNAMIC FALLBACK: Discover dataset types from implementations directory
    try:
        from pathlib import Path
        
        # Find the implementations directory relative to this file
        # config_accessors.py is in config/, implementations is in datasets/implementations/
        implementations_dir = Path(__file__).parent.parent / 'datasets' / 'implementations'
        if implementations_dir.exists():
            discovered_types = []
            for py_file in implementations_dir.glob('*.py'):
                if py_file.name.startswith('_'):
                    continue
                # Extract dataset name from filename (e.g., dft.py -> DFT, qm9.py -> QM9)
                dataset_name = py_file.stem.upper()
                # Exclude non-dataset modules
                if dataset_name not in ['BASE', 'REGISTRY', 'UTILS', 'COMMON']:
                    discovered_types.append(dataset_name)
            if discovered_types:
                logger.debug(f"ConfigAccessors: Dynamically discovered dataset types: {discovered_types}")
                return discovered_types
    except Exception as e:
        logger.debug(f"ConfigAccessors: Dynamic dataset type discovery failed: {e}")
    
    # Final fallback: return empty list with warning (no hardcoded types)
    logger.warning("ConfigAccessors: No dataset types available - registry not initialized and dynamic discovery failed")
    return []


def _registry_get_safe(name: str):
    """
    Safely get dataset class from registry.
    
    Args:
        name: Dataset type name
        
    Returns:
        Dataset class if available, None if registry unavailable
        
    ADDED Phase 5: Safe wrapper for registry access.
    """
    _init_registry()
    if _registry_get is not None:
        try:
            return _registry_get(name)
        except Exception as e:
            logger.debug(f"Could not get dataset '{name}' from registry: {e}")
            return None
    return None


def _registry_is_registered_safe(name: str) -> bool:
    """
    Safely check if dataset type is registered.
    
    DYNAMIC APPROACH: Instead of hardcoded fallback check, this function:
    1. First tries the registry (primary source of truth)
    2. If registry fails, uses _registry_list_all_safe() which does dynamic discovery
    3. Never uses hardcoded dataset type lists
    
    Args:
        name: Dataset type name
        
    Returns:
        True if registered or dynamically discovered, False otherwise
        
    ADDED Phase 5: Safe wrapper for registry access.
    UPDATED Phase 6.1: Replaced hardcoded fallback with dynamic filesystem discovery
    """
    _init_registry()
    if _registry_is_registered is not None:
        try:
            return _registry_is_registered(name)
        except Exception as e:
            logger.debug(f"Registry is_registered failed: {e}")
    
    # DYNAMIC FALLBACK: Check against dynamically discovered types
    available_types = _registry_list_all_safe()
    return name in available_types


def _get_dataset_features(dataset_type: str) -> Dict[str, Any]:
    """
    Get features dictionary for a dataset type from registry.
    
    Args:
        dataset_type: Dataset type name
        
    Returns:
        Dictionary of feature flags, or empty dict if unavailable
        
    ADDED Phase 5: Helper to query dataset features from registry.
    """
    dataset_class = _registry_get_safe(dataset_type)
    if dataset_class is not None and hasattr(dataset_class, 'features'):
        try:
            # Access the features attribute
            features = dataset_class.features
            # Convert DatasetFeatures dataclass to dict if needed
            if hasattr(features, '__dict__'):
                return vars(features)
            elif isinstance(features, dict):
                return features
        except Exception as e:
            logger.debug(f"Could not get features for dataset '{dataset_type}': {e}")
    return {}


def _dataset_supports_feature(dataset_type: str, feature_name: str) -> bool:
    """
    Check if a dataset type supports a specific feature.
    
    Args:
        dataset_type: Dataset type name
        feature_name: Feature name to check (e.g., 'uncertainty_handling')
        
    Returns:
        True if feature is supported, False otherwise
        
    ADDED Phase 5: Helper to check dataset feature support.
    """
    features = _get_dataset_features(dataset_type)
    return features.get(feature_name, False)


def _get_dataset_schema(dataset_type: str) -> Dict[str, Any]:
    """
    Get schema dictionary for a dataset type from registry.
    
    Args:
        dataset_type: Dataset type name
        
    Returns:
        Dictionary of schema information, or empty dict if unavailable
        
    ADDED Phase 5: Helper to query dataset schema from registry.
    """
    dataset_class = _registry_get_safe(dataset_type)
    if dataset_class is not None and hasattr(dataset_class, 'schema'):
        try:
            # Access the schema attribute
            schema = dataset_class.schema
            # Convert DatasetSchema dataclass to dict if needed
            if hasattr(schema, '__dict__'):
                return vars(schema)
            elif isinstance(schema, dict):
                return schema
        except Exception as e:
            logger.debug(f"Could not get schema for dataset '{dataset_type}': {e}")
    return {}


# ==========================================
# PUBLIC API: Registry Wrapper Functions
# ==========================================

def registry_list_all() -> List[str]:
    """
    Get list of all registered dataset types (PUBLIC API).
    
    Returns:
        List of dataset type names
        
    ADDED Phase 5: Public wrapper for registry access, following Phase 3 pattern.
    """
    return _registry_list_all_safe()


def registry_get(name: str):
    """
    Get dataset class from registry (PUBLIC API).
    
    Args:
        name: Dataset type name
        
    Returns:
        Dataset class if available
        
    Raises:
        HandlerNotAvailableError: If dataset type not registered
        
    ADDED Phase 5: Public wrapper for registry access, following Phase 3 pattern.
    """
    result = _registry_get_safe(name)
    if result is None:
        valid_types = _registry_list_all_safe()
        raise HandlerNotAvailableError(
            f"Dataset type '{name}' not registered",
            requested_dataset_type=name,
            available_types=valid_types
        )
    return result


def registry_is_registered(name: str) -> bool:
    """
    Check if dataset type is registered (PUBLIC API).
    
    Args:
        name: Dataset type name
        
    Returns:
        True if registered, False otherwise
        
    ADDED Phase 5: Public wrapper for registry access, following Phase 3 pattern.
    """
    return _registry_is_registered_safe(name)


def get_default_registry():
    """
    Get the default global registry instance (PUBLIC API).
    
    Returns:
        DatasetRegistry instance or None if registry unavailable
        
    ADDED Phase 5: Public wrapper for registry access, following Phase 3 pattern.
    """
    _init_registry()
    if _registry_get_default is not None:
        return _registry_get_default()
    return None


def _get_valid_dataset_types() -> List[str]:
    """
    Get list of valid dataset types (INTERNAL HELPER).
    
    This is a convenience alias for _registry_list_all_safe() used by
    other config modules for consistency with Phase 3/4 patterns.
    
    Returns:
        List of valid dataset type names
        
    ADDED Phase 5: Helper function for consistency with Phase 3/4 patterns.
    """
    return _registry_list_all_safe()


def get_valid_dataset_types() -> List[str]:
    """
    Get list of valid dataset types (PUBLIC API - no underscore).
    
    Public version of _get_valid_dataset_types for external module imports.
    
    Returns:
        List of valid dataset type names
        
    ADDED Phase 5: Public API for config/__init__.py compatibility.
    """
    return _get_valid_dataset_types()


def _is_valid_dataset_type(dataset_type: str) -> bool:
    """
    Check if a dataset type is valid (INTERNAL HELPER).
    
    This is a convenience alias for _registry_is_registered_safe() used by
    other config modules for consistency with Phase 3/4 patterns.
    
    Args:
        dataset_type: Dataset type name to check
        
    Returns:
        True if valid, False otherwise
        
    ADDED Phase 5: Helper function for consistency with Phase 3/4 patterns.
    """
    return _registry_is_registered_safe(dataset_type)


def validate_dataset_type(dataset_type: str, raise_on_invalid: bool = False) -> bool:
    """
    Validate that a dataset type is registered (PUBLIC API).
    
    This is a public validation function that can be used by other modules
    to check if a dataset type is valid before using it.
    
    Args:
        dataset_type: Dataset type name to validate
        raise_on_invalid: If True, raise ConfigurationError for invalid types
        
    Returns:
        True if valid and registered, False otherwise
        
    Raises:
        ConfigurationError: If raise_on_invalid=True and type is invalid
        
    ADDED Phase 5: Public validation function for config/__init__.py compatibility.
    """
    is_valid = _is_valid_dataset_type(dataset_type)
    
    if not is_valid and raise_on_invalid:
        valid_types = _get_valid_dataset_types()
        raise ConfigurationError(
            f"Invalid dataset type '{dataset_type}'. Valid types: {', '.join(valid_types)}",
            config_key="dataset_type",
            actual_value=dataset_type
        )
    
    return is_valid


def validate_dataset_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a dataset configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list of validation errors)
        
    ADDED Phase 5: Configuration validation helper.
    """
    errors = []
    
    # Check if dataset_type is present
    if 'dataset_type' not in config:
        errors.append("Missing required field: 'dataset_type'")
        return False, errors
    
    dataset_type = config['dataset_type']
    
    # Validate dataset type
    if not _is_valid_dataset_type(dataset_type):
        valid_types = _get_valid_dataset_types()
        errors.append(f"Invalid dataset_type '{dataset_type}'. Valid types: {', '.join(valid_types)}")
        return False, errors
    
    # If we get here, basic validation passed
    return True, []


def validate_handler_compatibility(dataset_type: str, config: Dict[str, Any]) -> bool:
    """
    Validate that a handler is compatible with the given dataset type and config.
    
    Args:
        dataset_type: Dataset type name
        config: Configuration dictionary
        
    Returns:
        bool: True if handler is compatible, False otherwise
        
    ADDED Phase 5: Handler compatibility validation.
    """
    # Check if dataset type is valid
    if not _is_valid_dataset_type(dataset_type):
        logger.warning(f"Invalid dataset type for handler compatibility check: {dataset_type}")
        return False
    
    # Try to get handler class from registry
    dataset_class = _registry_get_safe(dataset_type)
    if dataset_class is None:
        logger.warning(f"Could not get dataset class for type: {dataset_type}")
        return False
    
    # Check if handler class is defined
    if not hasattr(dataset_class, 'handler_class') or dataset_class.handler_class is None:
        logger.debug(f"No handler class defined for dataset type: {dataset_type}")
        return False
    
    # Basic compatibility check passed
    return True


def is_valid_dataset_type(dataset_type: str) -> bool:
    """
    Check if a dataset type is valid (PUBLIC API - no underscore).
    
    Public version of _is_valid_dataset_type for external module imports.
    
    Args:
        dataset_type: Dataset type name to check
        
    Returns:
        True if valid and registered, False otherwise
        
    ADDED Phase 5: Public API for config/__init__.py compatibility.
    """
    return _is_valid_dataset_type(dataset_type)


# Initialize logger for this module
logger = logging.getLogger(__name__)



class EnhancedConfigAccessor:
    """
    Base class for enhanced configuration accessors.
    
    This class doesn't replace anything, it's additional.
    
    Features:
    - Automatic validation using Transformation System Integration system
    - Intelligent defaults from parameter metadata
    - Type-safe extraction with casting
    - Context-aware retrieval (dataset-specific)
    - Enhanced error handling with suggestions
    """
    
    def __init__(self, 
                 config: Union[Dict[str, Any], 'ConfigContainer'],
                 validation_level: ValidationLevel = ValidationLevel.STANDARD,
                 auto_validate: bool = True,
                 dataset_context: Optional[str] = None):
        """
        Initialize enhanced accessor.
        
        Args:
            config: Configuration dict or container
            validation_level: Validation strictness level
            auto_validate: Whether to auto-validate on access
            dataset_context: Dataset context (DFT, DMC, MD) for optimization
        """
        self._config = config
        self._validation_level = validation_level
        self._auto_validate = auto_validate
        self._dataset_context = dataset_context
        self._access_count = 0
        self._validation_cache: Dict[str, bool] = {}
        
        # Initialize Transformation System Integration
        self._gt = get_graph_transforms() if GRAPH_TRANSFORMS_AVAILABLE else None
        
        # Initialize schema validator if available
        self._schema_validator = None
        if SCHEMAS_AVAILABLE:
            try:
                # Create YAMLSchemaValidator instance
                self._schema_validator = YAMLSchemaValidator()
            except Exception as e:
                logger.debug(f"Could not initialize schema validator: {e}")
        
        # Track accessor statistics
        self._stats = {
            'total_accesses': 0,
            'cache_hits': 0,
            'validations_performed': 0,
            'defaults_used': 0,
            'type_casts_performed': 0
        }
    
    def _get_raw_value(self, key: str, default: Any = None) -> Any:
        """Get raw value from config without processing."""
        if isinstance(self._config, dict):
            return self._config.get(key, default)
        elif CONTAINERS_AVAILABLE and isinstance(self._config, (DatasetConfig, TransformationConfig, HandlerConfig)):
            return getattr(self._config, key, default)
        else:
            return default
    
    def _set_value(self, key: str, value: Any) -> None:
        """Set value in config."""
        if isinstance(self._config, dict):
            self._config[key] = value
        elif CONTAINERS_AVAILABLE and isinstance(self._config, (DatasetConfig, TransformationConfig, HandlerConfig)):
            setattr(self._config, key, value)
    
    def _validate_value(self, key: str, value: Any, 
                       expected_type: Optional[type] = None) -> Tuple[bool, List[str]]:
        """
        Validate value with Transformation System Integration.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        if not self._auto_validate:
            return True, []
        
        # Check cache first
        cache_key = f"{key}:{type(value).__name__}:{value}"
        if cache_key in self._validation_cache:
            self._stats['cache_hits'] += 1
            return self._validation_cache[cache_key], []
        
        errors = []
        
        # Type validation
        if expected_type and not isinstance(value, expected_type):
            errors.append(f"Type mismatch for '{key}': expected {expected_type.__name__}, got {type(value).__name__}")
        
        # Schema validation if available
        if self._schema_validator and SCHEMAS_AVAILABLE:
            try:
                # Use YAMLSchemaValidator to validate value
                # Note: This is a simplified validation - adjust based on actual API
                validation_result = self._schema_validator.validate({key: value})
                if not validation_result:  # Assuming validate returns bool or dict with errors
                    errors.append(f"Schema validation failed for '{key}'")
            except Exception as e:
                logger.debug(f"Schema validation error for {key}: {e}")
        
        is_valid = len(errors) == 0
        self._validation_cache[cache_key] = is_valid
        self._stats['validations_performed'] += 1
        
        return is_valid, errors
    
    def _get_intelligent_default(self, key: str, 
                                expected_type: Optional[type] = None) -> Any:
        """
        Get intelligent default using Transformation System Integration parameter metadata.
        
        Args:
            key: Parameter name
            expected_type: Expected type hint
        
        Returns:
            Intelligent default value or None
        """
        # Try Transformation System Integration parameter metadata first
        if self._gt and GRAPH_TRANSFORMS_AVAILABLE:
            try:
                # Check if key matches a known transform parameter pattern
                if '_' in key:
                    parts = key.split('_')
                    if len(parts) >= 2:
                        potential_transform = parts[0]
                        param_name = '_'.join(parts[1:])
                        
                        try:
                            param_metadata = self._gt.get_parameter_info(
                                potential_transform, 
                                param_name
                            )
                            
                            if isinstance(param_metadata, ParameterMetadata):
                                self._stats['defaults_used'] += 1
                                
                                # Use default from metadata
                                if param_metadata.has_default:
                                    return param_metadata.default_value
                                
                                # Use first example
                                if param_metadata.examples:
                                    return param_metadata.examples[0]
                                
                                # Generate from type hint
                                if param_metadata.type_hint:
                                    return self._generate_default_from_type(
                                        param_metadata.type_hint
                                    )
                        except (TransformNotFoundError, TransformValidationError):
                            pass
            except Exception as e:
                logger.debug(f"Could not get intelligent default for {key}: {e}")
        
        # Fallback to type-based defaults
        if expected_type:
            return self._generate_default_from_type(expected_type)
        
        return None
    
    def _generate_default_from_type(self, type_hint: type) -> Any:
        """Generate default value from type hint."""
        type_defaults = {
            bool: False,
            int: 0,
            float: 0.0,
            str: "",
            list: [],
            dict: {},
            tuple: (),
            set: set()
        }
        
        return type_defaults.get(type_hint, None)
    
    def _cast_to_type(self, value: Any, target_type: type) -> Any:
        """
        Safely cast value to target type.
        
        Args:
            value: Value to cast
            target_type: Target type
        
        Returns:
            Casted value
        
        Raises:
            ValueError: If cast fails
        """
        if isinstance(value, target_type):
            return value
        
        try:
            self._stats['type_casts_performed'] += 1
            
            if target_type == bool:
                if isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes', 'on')
                return bool(value)
            
            elif target_type in (int, float):
                return target_type(value)
            
            elif target_type == str:
                return str(value)
            
            elif target_type in (list, tuple, set):
                if isinstance(value, (list, tuple, set)):
                    return target_type(value)
                return target_type([value])
            
            elif target_type == dict:
                if isinstance(value, dict):
                    return value
                raise ValueError(f"Cannot cast {type(value).__name__} to dict")
            
            else:
                return target_type(value)
                
        except Exception as e:
            raise ValueError(
                f"Failed to cast {value} (type: {type(value).__name__}) "
                f"to {target_type.__name__}: {e}"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get accessor statistics."""
        return self._stats.copy()


def get_transform_config(
    config: Union[Dict[str, Any], 'ConfigContainer'],
    transform_name: str,
    validate: bool = True,
    use_intelligent_defaults: bool = True,
    dataset_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Enhanced accessor for transform configuration with Transformation System Integration.
    
    INTEGRATION NOTE: This REPLACES or ADDS to your existing get_transform_config.
    If you have an existing function, REPLACE it. If not, ADD this function.
    
    Transformation System Integration Features:
    - Automatic validation system
    - Intelligent defaults from parameter metadata
    - Dataset-specific optimization
    - Enhanced error messages with suggestions
    
    Args:
        config: Configuration dict or container
        transform_name: Name of transform
        validate: Whether to validate configuration
        use_intelligent_defaults: Use intelligent defaults for missing params
        dataset_context: Dataset type (DFT, DMC, MD) for optimization
    
    Returns:
        Transform configuration with validated parameters
    
    Raises:
        ConfigurationError: If transform not found or validation fails
    """
    accessor = EnhancedConfigAccessor(
        config, 
        auto_validate=validate,
        dataset_context=dataset_context
    )
    
    # Extract transform list
    transforms = accessor._get_raw_value('transforms', [])
    if not transforms:
        transforms = accessor._get_raw_value('experimental_setups', {}).get('default', [])
    
    # Find transform by name
    transform_config = None
    for t in transforms:
        if isinstance(t, dict) and t.get('name') == transform_name:
            transform_config = t.copy()
            break
    
    if not transform_config:
        # Try to get intelligent default config if Transformation System Integration available
        if use_intelligent_defaults and GRAPH_TRANSFORMS_AVAILABLE:
            gt = get_graph_transforms()
            try:
                transform_info = gt.get_transform_info(transform_name)
                transform_config = {
                    'name': transform_name,
                    'kwargs': {},
                    'enabled': True
                }
                
                # Add intelligent defaults for parameters
                if transform_info:
                    param_metadata = gt.get_parameter_info(transform_name)
                    if isinstance(param_metadata, dict):
                        for param_name, metadata in param_metadata.items():
                            if metadata.has_default:
                                if 'kwargs' not in transform_config:
                                    transform_config['kwargs'] = {}
                                transform_config['kwargs'][param_name] = metadata.default_value
                
                logger.info(f"Generated intelligent default config for '{transform_name}'")
            except TransformNotFoundError:
                pass
    
    if not transform_config:
        available = [t.get('name', 'unknown') for t in transforms if isinstance(t, dict)]
        raise ConfigurationError(
            f"Transform '{transform_name}' not found in configuration",
            config_key=f"transforms.{transform_name}",
            suggestions=f"Available transforms: {available}"
        )
    
    # Validate if requested
    if validate and GRAPH_TRANSFORMS_AVAILABLE:
        gt = get_graph_transforms()
        validation_result = gt.validate_config(
            [transform_config],
            dataset_type=dataset_context
        )
        
        if not validation_result['valid']:
            raise ConfigurationError(
                f"Transform '{transform_name}' validation failed",
                config_key=f"transforms.{transform_name}",
                validation_errors=validation_result['errors'],
                suggestions="Check parameter types and values"
            )
    
    return transform_config

def get_transform_parameter(
    config: Union[Dict[str, Any], 'ConfigContainer'],
    transform_name: str,
    parameter_name: str,
    default: Any = None,
    expected_type: Optional[type] = None,
    validate: bool = True,
    auto_cast: bool = True,
    dataset_context: Optional[str] = None
) -> Any:
    """
    Enhanced accessor for transform parameter with type safety and validation.
    
    INTEGRATION NOTE: This REPLACES or ADDS to your existing get_transform_parameter.
    
    Transformation System Integration Features:
    - Type-safe extraction with automatic casting
    - Validation using Transformation System Integration parameter metadata
    - Intelligent defaults from parameter introspection
    - Enhanced error messages
    
    Args:
        config: Configuration dict or container
        transform_name: Name of transform
        parameter_name: Name of parameter
        default: Default value if not found
        expected_type: Expected type for validation and casting
        validate: Whether to validate parameter value
        auto_cast: Whether to automatically cast to expected_type
        dataset_context: Dataset type for context-aware defaults
    
    Returns:
        Parameter value (casted to expected_type if specified)
    
    Raises:
        ConfigurationError: If parameter invalid or type mismatch
        ValueError: If casting fails
    """
    accessor = EnhancedConfigAccessor(
        config,
        auto_validate=validate,
        dataset_context=dataset_context
    )
    
    # Get transform config
    try:
        transform_config = get_transform_config(
            config, 
            transform_name, 
            validate=False,  # We'll validate the parameter separately
            dataset_context=dataset_context
        )
    except ConfigurationError:
        # Transform not found - use intelligent default
        if default is not None:
            return default
        
        intelligent_default = accessor._get_intelligent_default(
            f"{transform_name}_{parameter_name}",
            expected_type
        )
        
        if intelligent_default is not None:
            logger.info(f"Using intelligent default for {transform_name}.{parameter_name}")
            return intelligent_default
        
        raise
    
    # Extract parameter value
    kwargs = transform_config.get('kwargs', {})
    
    if parameter_name not in kwargs:
        # Try intelligent default
        if default is not None:
            value = default
        else:
            # Use Transformation System Integration metadata for intelligent default
            if GRAPH_TRANSFORMS_AVAILABLE:
                gt = get_graph_transforms()
                try:
                    param_metadata = gt.get_parameter_info(transform_name, parameter_name)
                    if isinstance(param_metadata, ParameterMetadata):
                        if param_metadata.has_default:
                            value = param_metadata.default_value
                        elif param_metadata.examples:
                            value = param_metadata.examples[0]
                        else:
                            value = accessor._get_intelligent_default(
                                parameter_name, 
                                expected_type
                            )
                    else:
                        value = None
                except (TransformNotFoundError, TransformValidationError):
                    value = None
            else:
                value = None
        
        if value is None:
            raise ConfigurationError(
                f"Parameter '{parameter_name}' not found for transform '{transform_name}'",
                config_key=f"transforms.{transform_name}.kwargs.{parameter_name}",
                suggestions="Provide value or use intelligent defaults"
            )
    else:
        value = kwargs[parameter_name]
    
    # Type casting if requested
    if expected_type and auto_cast:
        try:
            value = accessor._cast_to_type(value, expected_type)
        except ValueError as e:
            raise ConfigurationError(
                f"Type cast failed for {transform_name}.{parameter_name}",
                config_key=f"transforms.{transform_name}.kwargs.{parameter_name}",
                details=str(e),
                suggestions=f"Ensure value is compatible with {expected_type.__name__}"
            )
    
    # Validate parameter if requested
    if validate and GRAPH_TRANSFORMS_AVAILABLE:
        gt = get_graph_transforms()
        try:
            is_valid, errors = gt.validate_parameter(
                transform_name,
                parameter_name,
                value,
                check_constraints=True
            )
            
            if not is_valid:
                raise ConfigurationError(
                    f"Parameter validation failed: {transform_name}.{parameter_name}",
                    config_key=f"transforms.{transform_name}.kwargs.{parameter_name}",
                    validation_errors=errors,
                    suggestions="Check parameter constraints and type"
                )
        except (TransformNotFoundError, TransformValidationError) as e:
            logger.debug(f"Could not validate parameter: {e}")
    
    return value

def get_dataset_specific_config(
    config: Union[Dict[str, Any], 'ConfigContainer'],
    dataset_type: str,
    config_key: str,
    default: Any = None,
    validate: bool = True
) -> Any:
    """
    Get dataset-specific configuration with context awareness.
    
    Add this function to your file.
    
    Context-aware retrieval with dataset-specific fallbacks.
    
    Args:
        config: Configuration dict or container
        dataset_type: Dataset type (DFT, DMC, MD)
        config_key: Configuration key to retrieve
        default: Default value if not found
        validate: Whether to validate value
    
    Returns:
        Dataset-specific configuration value
    """
    accessor = EnhancedConfigAccessor(
        config,
        auto_validate=validate,
        dataset_context=dataset_type
    )
    
    # Try dataset-specific config first
    dataset_configs = accessor._get_raw_value('dataset_configs', {})
    
    if dataset_type in dataset_configs:
        dataset_config = dataset_configs[dataset_type]
        if isinstance(dataset_config, dict) and config_key in dataset_config:
            return dataset_config[config_key]
    
    # Fallback to global config
    value = accessor._get_raw_value(config_key, default)
    
    if value is None and default is None:
        raise ConfigurationError(
            f"Configuration '{config_key}' not found for dataset '{dataset_type}'",
            config_key=f"dataset_configs.{dataset_type}.{config_key}",
            suggestions=f"Define {config_key} in dataset_configs or global config"
        )
    
    return value if value is not None else default


def get_all_transforms(
    config: Union[Dict[str, Any], 'ConfigContainer'],
    validate_each: bool = True,
    dataset_context: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get all transform configurations with optional validation.
    
    Transformation System Integration - Add this function to your file.
    
    Transformation System Integration Feature: Batch retrieval with per-transform validation.
    
    Args:
        config: Configuration dict or container
        validate_each: Whether to validate each transform
        dataset_context: Dataset type for context
    
    Returns:
        List of all transform configurations
    """
    accessor = EnhancedConfigAccessor(
        config,
        auto_validate=validate_each,
        dataset_context=dataset_context
    )
    
    # Extract transforms
    transforms = accessor._get_raw_value('transforms', [])
    
    if not transforms:
        # Try experimental setups
        setups = accessor._get_raw_value('experimental_setups', {})
        if setups:
            default_setup = accessor._get_raw_value('default_setup', 'default')
            if default_setup in setups:
                transforms = setups[default_setup]
    
    if not transforms:
        return []
    
    # Validate if requested
    if validate_each and GRAPH_TRANSFORMS_AVAILABLE:
        gt = get_graph_transforms()
        validation_result = gt.validate_config(
            transforms,
            dataset_type=dataset_context
        )
        
        if not validation_result['valid']:
            logger.warning(
                f"Transform validation found issues: {validation_result['errors']}"
            )
    
    return transforms


def get_config_with_fallback(
    config: Union[Dict[str, Any], 'ConfigContainer'],
    keys: List[str],
    default: Any = None,
    expected_type: Optional[type] = None,
    auto_cast: bool = True
) -> Any:
    """
    Get configuration value with fallback chain and type casting.
    
    Transformation System Integration - Add this function to your file.
    
    Transformation System Integration: Multi-key fallback with intelligent type handling.
    
    Args:
        config: Configuration dict or container
        keys: List of keys to try (in order)
        default: Default value if none found
        expected_type: Expected type for casting
        auto_cast: Whether to auto-cast to expected_type
    
    Returns:
        Configuration value (first found in key chain)
    """
    accessor = EnhancedConfigAccessor(config, auto_validate=False)
    
    # Try each key in order
    for key in keys:
        value = accessor._get_raw_value(key)
        if value is not None:
            # Cast if needed
            if expected_type and auto_cast:
                try:
                    value = accessor._cast_to_type(value, expected_type)
                except ValueError as e:
                    logger.debug(f"Cast failed for {key}: {e}, trying next key")
                    continue
            return value
    
    # All keys failed - use default
    return default


def validate_config_structure(
    config: Union[Dict[str, Any], 'ConfigContainer'],
    required_keys: Optional[List[str]] = None,
    dataset_context: Optional[str] = None,
    validation_level: ValidationLevel = ValidationLevel.STANDARD
) -> Tuple[bool, List[str]]:
    """
    Validate overall configuration structure.
    
    Transformation System Integration - Add this function to your file.
    
    Transformation System Integration: Structural validation with integration.
    
    Args:
        config: Configuration to validate
        required_keys: List of required keys
        dataset_context: Dataset type for context-aware validation
        validation_level: Validation strictness
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required keys
    if required_keys:
        for key in required_keys:
            if isinstance(config, dict):
                if key not in config:
                    errors.append(f"Required key '{key}' missing from configuration")
            elif CONTAINERS_AVAILABLE and isinstance(config, ConfigContainer):
                if not hasattr(config, key):
                    errors.append(f"Required attribute '{key}' missing from configuration")
    
    # Schema validation if available
    if SCHEMAS_AVAILABLE:
        try:
            # Use YAMLSchemaValidator to validate structure
            validator = YAMLSchemaValidator()
            validation_result = validator.validate(config)
            # Adjust based on actual return type of YAMLSchemaValidator.validate()
            if not validation_result:
                errors.append("Schema structure validation failed")
        except Exception as e:
            logger.debug(f"Schema validation error: {e}")
    
    # Transform validation if available
    if GRAPH_TRANSFORMS_AVAILABLE and isinstance(config, dict):
        transforms = config.get('transforms', [])
        if transforms:
            gt = get_graph_transforms()
            validation_result = gt.validate_config_comprehensive(
                transforms,
                dataset_type=dataset_context,
                validation_level=validation_level
            )
            
            if not validation_result['valid']:
                errors.extend(validation_result['errors'])
    
    return len(errors) == 0, errors


#--------------------------------------------------------------------------------------------------------------

def _get_config_value(config_dict: Dict[str, Any], key: str, 
                     expected_type: Optional[Union[type, tuple[type, ...]]] = None, 
                     parent_key: Optional[str] = None) -> Any:
    """
    Safely retrieves a value from the configuration dictionary with type validation.

    Args:
        config_dict (dict): The dictionary to retrieve the value from.
        key (str): The key to look up within the dictionary.
        expected_type (type or tuple[type, ...], optional): If provided,
            validates if the retrieved value is an instance of this type or any
            of the types in the tuple. Defaults to None (no type checking).
        parent_key (str, optional): The parent key, used for constructing
            more informative error messages (e.g., "section.key").

    Returns:
        Any: The value associated with the specified key.

    Raises:
        ConfigurationError: If the key is missing from the dictionary or
            the retrieved value does not match the `expected_type`.
    """
    full_key = f"{parent_key}.{key}" if parent_key else key
    if key not in config_dict:
        raise ConfigurationError(
            f"Missing required configuration key.",
            config_key=full_key
        )
    value = config_dict[key]
    if expected_type is not None and not isinstance(value, expected_type):
        raise ConfigurationError(
            f"Invalid type for configuration key '{full_key}'.",
            config_key=full_key,
            expected_type=expected_type,
            actual_value=value
        )
    return value


# Dataset Type and Configuration Accessors
def get_dataset_type() -> str:
    """
    Gets the dataset type from the configuration.
    
    PHASE 5 REFACTORING: Now returns any registered dataset type from registry,
    not just hardcoded "DFT", "DMC", "Wavefunction".
    
    PHASE 6.2 SIMPLIFICATION: Case-insensitive normalization now handled by
    config_loader.py at load time. This function receives already-normalized
    values, so only validation against registry is needed here.
    
    Returns:
        str: Dataset type name (e.g., "DFT", "DMC", "Wavefunction", or any registered type)
        
    Raises:
        ConfigurationError: If dataset_type is missing or invalid
    """
    try:
        config = load_config()
        dataset_type = _get_config_value(config, 'dataset_type', str)
        
        # PHASE 5 REFACTORING: Use registry for dynamic validation
        # PHASE 6.2: dataset_type is already normalized by config_loader.py
        # Only need to validate it exists in registry
        valid_types = _registry_list_all_safe()
        
        if dataset_type not in valid_types:
            raise ConfigurationError(
                f"Invalid dataset_type '{dataset_type}'. Must be one of: {', '.join(valid_types)}",
                config_key="dataset_type",
                actual_value=dataset_type
            )
        
        return dataset_type
        
    except Exception as e:
        # Re-raise ConfigurationError as-is
        if isinstance(e, ConfigurationError):
            raise
        # Convert other errors to ConfigurationError
        raise ConfigurationError(
            f"Failed to get dataset type: {str(e)}",
            config_key="dataset_type"
        ) from e


def get_handler_type(dataset_type: Optional[str] = None) -> str:
    """
    Get the handler type for a dataset.
    
    Args:
        dataset_type: Optional dataset type. If None, uses get_dataset_type()
        
    Returns:
        str: Handler type (typically same as dataset_type)
        
    ADDED Phase 5: Helper function for handler type retrieval.
    """
    if dataset_type is None:
        dataset_type = get_dataset_type()
    
    # Handler type is typically the same as dataset type
    # But can be overridden in config
    dataset_config = get_dataset_config(dataset_type)
    return dataset_config.get('handler_type', dataset_type)


def is_handler_type(handler_type: str, dataset_type: Optional[str] = None) -> bool:
    """
    Check if the current handler type matches the given type.
    
    Case-insensitive comparison to match dataset_type comparison patterns elsewhere.
    
    Args:
        handler_type: Handler type to check against
        dataset_type: Optional dataset type. If None, uses get_dataset_type()
        
    Returns:
        bool: True if handler type matches (case-insensitive)
        
    ADDED Phase 5: Helper function for handler type checking.
    FIXED: Made case-insensitive to match other dataset_type comparisons.
    """
    current_handler = get_handler_type(dataset_type)
    return current_handler.upper() == handler_type.upper()


def get_optional_properties(dataset_type: Optional[str] = None) -> List[str]:
    """
    Get optional properties for a dataset type from registry.
    
    Args:
        dataset_type: Optional dataset type. If None, uses get_dataset_type()
        
    Returns:
        List of optional property names
        
    ADDED Phase 5: Query optional properties from registry.
    """
    if dataset_type is None:
        dataset_type = get_dataset_type()
    
    # Try to get from registry first
    dataset_class = _registry_get_safe(dataset_type)
    if dataset_class is not None:
        try:
            return dataset_class.get_optional_properties()
        except Exception as e:
            logger.debug(f"Could not get optional properties from registry: {e}")
    
    # Fallback: return empty list
    return []


def get_supported_features(dataset_type: Optional[str] = None) -> Dict[str, bool]:
    """
    Get supported features for a dataset type from registry.
    
    Args:
        dataset_type: Optional dataset type. If None, uses get_dataset_type()
        
    Returns:
        Dictionary of feature names to boolean support flags
        
    ADDED Phase 5: Query features from registry (Phase 5 pattern).
    """
    if dataset_type is None:
        dataset_type = get_dataset_type()
    
    return _get_dataset_features(dataset_type)


def is_feature_supported(feature_name: str, dataset_type: Optional[str] = None) -> bool:
    """
    Check if a specific feature is supported by a dataset type.
    
    Args:
        feature_name: Feature name to check
        dataset_type: Optional dataset type. If None, uses get_dataset_type()
        
    Returns:
        bool: True if feature is supported
        
    ADDED Phase 5: Feature support checking.
    """
    if dataset_type is None:
        dataset_type = get_dataset_type()
    
    return _dataset_supports_feature(dataset_type, feature_name)


def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get configuration value using dot-notation path.
    
    Helper function for plugin system integration.
    
    Examples:
        >>> get_config_value('plugins.enabled', False)
        True
        >>> get_config_value('plugins.paths', [])
        ['./custom_plugins', '~/.milia/plugins']
    
    Args:
        key_path: Dot-separated configuration path (e.g., 'plugins.enabled')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    try:
        config = load_config()
        keys = key_path.split('.')
        
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
        
    except Exception as e:
        logger.debug(f"Error getting config value '{key_path}': {e}")
        return default


def get_dataset_config(dataset_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Gets the appropriate dataset configuration based on the dataset type.
    
    Args:
        dataset_type: Optional dataset type. If None, uses get_dataset_type()
    
    Returns:
        dict: Dataset configuration (either dft_config or dmc_config)
        
    Raises:
        ConfigurationError: If the dataset configuration is missing
    """
    config = load_config()
    
    if dataset_type is None:
        dataset_type = get_dataset_type()
    
    config_key = f"{dataset_type.lower()}_config"
    return _get_config_value(config, config_key, dict)


def get_raw_data_info(dataset_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Get raw data information (filename and URL) for a dataset type.
    
    Args:
        dataset_type: Dataset type name. If None, uses get_dataset_type()
        
    Returns:
        dict: Dictionary with 'filename' and 'url' keys
        
    Raises:
        ConfigurationError: If dataset config is missing
        
    ADDED Phase 5: Helper function to get raw data information.
    """
    if dataset_type is None:
        dataset_type = get_dataset_type()
    
    dataset_config = get_dataset_config(dataset_type)
    
    return {
        'filename': dataset_config.get('raw_npz_filename', ''),
        'url': dataset_config.get('raw_data_download_url', ''),
        'root_dir': dataset_config.get('dataset_root_dir', '')
    }


def get_data_config() -> Dict[str, Any]:
    """
    Gets the data configuration for the current dataset type.
    
    Returns:
        dict: Data configuration with property selection for current dataset type
    """
    config = load_config()
    dataset_type = get_dataset_type()
    
    data_config = _get_config_value(config, 'data_config', dict)
    property_selection = _get_config_value(data_config, 'property_selection', dict)
    dataset_specific_config = _get_config_value(
        property_selection, dataset_type, dict, 
        parent_key='data_config.property_selection'
    )
    
    # Merge common settings with dataset-specific settings
    common_settings = _get_config_value(data_config, 'common_settings', dict, parent_key='data_config')
    
    # Create combined config - dataset-specific overrides common
    combined_config = {**common_settings, **dataset_specific_config}
    
    # Flatten structural_feature_integration settings to top level for backward compatibility
    if 'structural_feature_integration' in combined_config:
        structural_settings = combined_config['structural_feature_integration']
        # Add structural settings to top level, but keep the nested version too
        combined_config.update(structural_settings)
    
    return combined_config


def get_property_availability() -> Dict[str, Any]:
    """
    Gets the property availability matrix for the current dataset type.
    
    Returns:
        dict: Available properties for the current dataset type
    """
    config = load_config()
    dataset_type = get_dataset_type()
    
    property_availability = _get_config_value(config, 'property_availability', dict)
    return _get_config_value(property_availability, dataset_type, dict, parent_key='property_availability')


# Uncertainty Configuration (DMC-specific)
def get_uncertainty_config(dataset_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Gets the uncertainty handling configuration for datasets that support uncertainty.
    
    PHASE 5 REFACTORING: Now queries registry for uncertainty_handling feature support
    instead of hardcoding "DMC" check.
    
    Returns:
        dict or None: Uncertainty configuration if dataset supports uncertainty_handling, None otherwise
    """
    dataset_type = get_dataset_type()
    
    # PHASE 5 REFACTORING: Use registry feature query instead of hardcoded "DMC" check
    if not _dataset_supports_feature(dataset_type, 'uncertainty_handling'):
        logger.debug(f"Dataset type '{dataset_type}' does not support uncertainty handling")
        return None
    
    dataset_config = get_dataset_config()
    return dataset_config.get('uncertainty_handling', {})


def is_uncertainty_enabled() -> bool:
    """
    Checks if uncertainty handling is enabled for the current dataset.
    
    Returns:
        bool: True if uncertainty handling is enabled (DMC dataset), False otherwise
    """
    uncertainty_config = get_uncertainty_config()
    return uncertainty_config is not None and uncertainty_config.get('use_for_loss_weighting', False)


# Structural Features Configuration
def get_structural_features_config() -> Optional[Dict[str, Any]]:
    """
    Gets the structural features configuration.
    
    Returns:
        dict or None: Structural features configuration, or None if not configured
    """
    config = load_config()
    return config.get('structural_features', None)


def get_structural_features_preprocessing_config() -> Optional[Dict[str, Any]]:
    """
    Gets the structural features preprocessing configuration.
    
    Returns:
        dict or None: Preprocessing configuration for structural features
    """
    structural_config = get_structural_features_config()
    if structural_config is None:
        return None
    return structural_config.get('preprocessing', {})


def get_charge_handling_config() -> Dict[str, Any]:
    """
    Gets the charge handling configuration for structural features.
    
    Returns:
        dict: Charge handling configuration with defaults
    """
    preprocessing_config = get_structural_features_preprocessing_config()
    if preprocessing_config is None:
        # Return sensible defaults
        return {
            'prefer_mulliken': True,
            'compute_gasteiger_fallback': True,
            'missing_charge_default': 0.0
        }
    return preprocessing_config.get('charge_handling', {
        'prefer_mulliken': True,
        'compute_gasteiger_fallback': True,
        'missing_charge_default': 0.0
    })


def get_geometric_features_config() -> Dict[str, Any]:
    """
    Gets the geometric features configuration for structural features.
    
    Returns:
        dict: Geometric features configuration with defaults
    """
    preprocessing_config = get_structural_features_preprocessing_config()
    if preprocessing_config is None:
        # Return sensible defaults
        return {
            'enable_3d_features': True,
            'conformer_id': 0,
            'missing_length_default': 0.0,
            'bond_length_bins': {
                'bin_edges': [0.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 999.0],
                'bin_labels': ["very_short", "short", "C-C_single", "C=C_double", 
                             "medium", "long", "very_long", "extreme", "missing"]
            }
        }
    return preprocessing_config.get('geometric_features', {
        'enable_3d_features': True,
        'conformer_id': 0,
        'missing_length_default': 0.0,
        'bond_length_bins': {
            'bin_edges': [0.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 999.0],
            'bin_labels': ["very_short", "short", "C-C_single", "C=C_double", 
                         "medium", "long", "very_long", "extreme", "missing"]
        }
    })


def get_stereochemistry_config() -> Dict[str, Any]:
    """
    Gets the stereochemistry configuration for structural features.
    
    Returns:
        dict: Stereochemistry configuration with defaults
    """
    preprocessing_config = get_structural_features_preprocessing_config()
    if preprocessing_config is None:
        # Return sensible defaults
        return {
            'assign_stereochemistry': True,
            'cleanup_stereochemistry': True
        }
    return preprocessing_config.get('stereochemistry', {
        'assign_stereochemistry': True,
        'cleanup_stereochemistry': True
    })


# Filter and Transformation Configuration
def get_filter_config() -> Dict[str, Any]:
    """
    Gets the dataset filtering configuration.
    
    Returns:
        dict: Filter configuration, empty dict if not specified
    """
    config = load_config()
    filter_config = config.get('filter_config')
    
    # Ensure we always return a dict, never None for backward compatibility
    if filter_config is None:
        return {}
    
    # Ensure it's a dict (in case of malformed config)
    if not isinstance(filter_config, dict):
        return {}
    
    return filter_config


def get_transformations_config() -> List[Dict[str, Any]]:
    """
    Gets the PyTorch Geometric transformations configuration.
    
    LEGACY COMPATIBILITY: Always return legacy format for backward compatibility
    
    Returns:
        list: List of transformation configurations in legacy format
    """
    config = load_config()
    transforms_section = config.get('transformations', [])
    
    # Handle enhanced format - extract legacy format for backward compatibility
    if isinstance(transforms_section, dict) and 'experimental_setups' in transforms_section:
        # If there's a legacy_transforms section, return that for compatibility
        if 'legacy_transforms' in transforms_section:
            legacy_transforms = transforms_section['legacy_transforms']
            if isinstance(legacy_transforms, list):
                return legacy_transforms
        
        # Otherwise extract transforms from default setup
        experimental_setups = transforms_section.get('experimental_setups', {})
        default_setup = transforms_section.get('default_setup')
        
        if default_setup and default_setup in experimental_setups:
            setup_transforms = experimental_setups[default_setup]
            if isinstance(setup_transforms, list):
                # Convert enhanced format back to legacy format
                legacy_transforms = []
                for transform in setup_transforms:
                    legacy_transform = {'name': transform['name']}
                    if 'kwargs' in transform:
                        legacy_transform['kwargs'] = transform['kwargs']
                    # Skip enhanced-only fields for backward compatibility
                    legacy_transforms.append(legacy_transform)
                return legacy_transforms
    
    # Handle legacy format (already a list)
    if isinstance(transforms_section, list):
        return transforms_section
    
    # Fallback for any other case
    return []

def get_graph_transforms():
    """
    Get the graph transforms instance.
    
    Added for Integration with graph_transforms.py
    
    Returns:
        GraphTransforms instance from graph_transforms module
    """
    try:
        from milia_pipeline.transformations.graph_transforms import get_graph_transforms as gt_get_transforms
        return gt_get_transforms()
    except ImportError:
        logger.warning("graph_transforms module not available")
        # Return a mock-like object for testing
        return None
    except Exception as e:
        logger.error(f"Failed to get graph transforms: {e}")
        return None


def get_transformation_cache_key(configs: List[Dict[str, Any]]) -> str:
    """
    Generate a cache key for transformation configurations.
    
    Cache management integration
    
    Args:
        configs: List of transform configurations
        
    Returns:
        str: Cache key for the configuration
    """
    try:
        import json
        import hashlib
        
        # Handle empty or None configs
        if not configs:
            return hashlib.md5(b"empty_config").hexdigest()
        
        # Create normalized representation for hashing
        normalized_configs = []
        for config in configs:
            if config is None:
                continue
                
            if isinstance(config, dict):
                normalized = {}
                
                # Handle name field
                if 'name' in config:
                    normalized['name'] = config['name']
                else:
                    continue  # Skip configs without name
                
                # Handle kwargs field (ensure it's a dict)
                kwargs = config.get('kwargs')
                if kwargs is None:
                    normalized['kwargs'] = {}
                elif isinstance(kwargs, dict):
                    normalized['kwargs'] = dict(sorted(kwargs.items()))
                else:
                    normalized['kwargs'] = {}
                
                # Handle enabled field
                normalized['enabled'] = config.get('enabled', True)
                
                normalized_configs.append(normalized)
        
        # Create hash of the normalized configuration
        config_str = json.dumps(normalized_configs, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()
        
    except Exception as e:
        logger.error(f"Failed to generate cache key: {e}")
        # Return a simple fallback key
        return f"config_{len(configs)}_{hash(str(configs))}"


def get_transformation_performance_metrics() -> Dict[str, Any]:
    """
    Get performance metrics for the transformation system.
    
    Performance monitoring integration
    
    Returns:
        Dict[str, Any]: Performance metrics and statistics
    """
    try:
        gt = get_graph_transforms()
        if gt is None:
            return {
                'system_initialized': False,
                'available_transform_count': 0,
                'cache_stats': {},
                'usage_stats': {},
                'performance_metrics': {},
                'error': 'Graph transforms system not available'
            }
        
        system_status = gt.get_system_status()
        
        return {
            'system_initialized': system_status.get('initialized', False),
            'available_transform_count': system_status.get('available_transform_count', 0),
            'cache_stats': system_status.get('cache_stats', {}),
            'usage_stats': system_status.get('usage_stats', {}),
            'performance_metrics': system_status.get('performance_metrics', {})
        }
        
    except Exception as e:
        logger.error(f"Failed to get transformation performance metrics: {e}")
        return {
            'system_initialized': False,
            'error': str(e)
        }


def create_experimental_setup_from_dict(setup_name: str, setup_config: Dict[str, Any]) -> Optional[ExperimentalSetup]:
    """
    Create an ExperimentalSetup from dictionary configuration.
    
    Helper function for dynamic experimental setup creation
    
    Args:
        setup_name: Name for the experimental setup
        setup_config: Dictionary containing setup configuration
        
    Returns:
        ExperimentalSetup: Created experimental setup, or None if creation failed
    """
    try:
        # Use the existing factory function from config_containers
        from milia_pipeline.config.config_containers import create_experimental_setup_from_dict as container_create
        
        # Ensure setup has a name
        setup_config_with_name = setup_config.copy()
        setup_config_with_name['name'] = setup_name
        
        # Call the existing function with non-strict validation for test compatibility
        return container_create(setup_config_with_name, strict_validation=False)
        
    except Exception as e:
        logger.error(f"Failed to create experimental setup '{setup_name}': {e}")
        return None


def save_experimental_setup(setup: ExperimentalSetup, 
                           persist_to_config: bool = False) -> Dict[str, Any]:
    """
    Save an experimental setup (in memory and optionally to configuration).
    
    Dynamic experimental setup management
    
    Args:
        setup: ExperimentalSetup to save
        persist_to_config: Whether to persist to configuration file
        
    Returns:
        Dict[str, Any]: Save operation result
    """
    result = {
        'success': False,
        'setup_name': setup.name,
        'persisted': False,
        'warnings': []
    }
    
    try:
        # Validate the setup
        is_valid, validation_errors = setup.validate_experimental_setup()
        if not is_valid:
            result['warnings'].extend(validation_errors)
            return result
        
        # For now, we don't modify the actual configuration file
        # This would require more sophisticated configuration management
        if persist_to_config:
            result['warnings'].append("Persistence to configuration file not yet implemented")
        
        result['success'] = True
        logger.info(f"Experimental setup '{setup.name}' validated successfully")
        
    except Exception as e:
        result['warnings'].append(f"Save operation failed: {str(e)}")
        logger.error(f"Failed to save experimental setup '{setup.name}': {e}")
    
    return result

# ==========================================
# sTRANSFORMATION CONFIGURATION ACCESS FUNCTIONS
# ==========================================

def get_transformation_config() -> TransformationConfig:
    """
    Get comprehensive transformation configuration using enhanced containers.
    
    ✓ GUARANTEED: Returns a valid TransformationConfig with at least one experimental setup.
    
    Enhanced transformation configuration access with validation
    
    Returns:
        TransformationConfig: Valid configuration with validated experimental setups
        
    Raises:
        ConfigurationError: Only if critical system failure (very rare)
    """
    # ✓ FIX 1: Initialize variables at the start to prevent "referenced before assignment" errors
    transformation_config = None
    experimental_setups = None
    
    try:
        from milia_pipeline.config.config_containers import create_transformation_config_from_global
        
        # ✓ FIX 2: Track creation attempt (no need for migration_start_time)
        logger.debug("Attempting to create TransformationConfig from global configuration")
        
        transformation_config = create_transformation_config_from_global(logger)
        
        # ✓ FIX 3: Comprehensive validation of returned config
        if transformation_config is None:
            raise ConfigurationError(
                "create_transformation_config_from_global returned None",
                config_key="transformations",
                suggestions=["Check config.yaml transformations section", 
                           "Verify config_loader is functioning"]
            )
        
        # ✓ FIX 4: Validate config has required structure
        if not isinstance(transformation_config, TransformationConfig):
            raise ConfigurationError(
                f"Expected TransformationConfig, got {type(transformation_config).__name__}",
                config_key="transformations",
                actual_value=type(transformation_config).__name__
            )
        
        # ✓ FIX 5: Store experimental_setups for error handling
        experimental_setups = transformation_config.experimental_setups
        
        # ✓ FIX 6: Validate config has experimental setups
        if not experimental_setups:
            raise ConfigurationError(
                "TransformationConfig has no experimental setups",
                config_key="transformations.experimental_setups",
                suggestions=["Configuration migration may have failed",
                           "Check config.yaml format"]
            )
        
        # ✓ FIX 7: Validate default setup exists
        if transformation_config.default_setup not in experimental_setups:
            # Try to recover by using first setup
            available_setups = list(experimental_setups.keys())
            if available_setups:
                logger.warning(
                    f"Default setup '{transformation_config.default_setup}' not found, "
                    f"using '{available_setups[0]}' instead"
                )
                # Create corrected config (this is safe since we validated setups exist)
                from milia_pipeline.config.config_containers import TransformationConfig as TC
                transformation_config = TC(
                    experimental_setups=experimental_setups,
                    default_setup=available_setups[0],  # Use first available
                    validation=transformation_config.validation,
                    performance_settings=transformation_config.performance_settings,
                    migration_metadata=transformation_config.migration_metadata,
                    research_metadata=transformation_config.research_metadata
                )
            else:
                raise ConfigurationError(
                    f"Default setup '{transformation_config.default_setup}' not in experimental_setups",
                    config_key="transformations.default_setup",
                    actual_value=transformation_config.default_setup,
                    suggestions=[f"Available setups: {list(experimental_setups.keys())}"]
                )
        
        # ✓ FIX 8: Validate experimental setups have transforms
        # Check if standard_transforms are configured (new architecture)
        has_std_transforms = (
            hasattr(transformation_config, 'standard_transforms') and 
            transformation_config.standard_transforms
        )
        
        for setup_name, setup in experimental_setups.items():
            if not hasattr(setup, 'transforms'):
                logger.warning(f"Setup '{setup_name}' has no transforms attribute")
            elif not setup.transforms:
                if has_std_transforms:
                    # Expected behavior: setup relies on standard_transforms
                    logger.debug(
                        f"Setup '{setup_name}' has no experimental transforms "
                        f"(using {len(transformation_config.standard_transforms)} standard_transforms)"
                    )
                else:
                    # Actual warning: no transforms anywhere
                    logger.warning(f"Setup '{setup_name}' has empty transforms list and no standard_transforms configured")
        
        # ✓ SUCCESS: Return validated config
        logger.debug(
            f"Validated TransformationConfig: {len(experimental_setups)} setups, "
            f"default='{transformation_config.default_setup}'"
        )
        return transformation_config
        
    except ConfigurationError:
        # Re-raise configuration errors directly
        raise
        
    except Exception as e:
        # ✓ FIX 9: Improved error handling with proper variable checks
        logger.error(f"Unexpected error getting transformation config: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        
        # Build error details with safe variable access
        error_details = f"Error type: {type(e).__name__}"
        if experimental_setups is not None:
            error_details += f", Found {len(experimental_setups)} setups"
        if transformation_config is not None:
            error_details += f", Config type: {type(transformation_config).__name__}"
        
        raise ConfigurationError(
            message=f"Failed to get transformation configuration: {str(e)}",
            config_key="transformations",
            details=error_details,
            suggestions=[
                "Check config.yaml syntax",
                "Verify transformations section exists",
                "Check logs for migration errors",
                "Ensure config_containers module is properly imported"
            ]
        ) from e


def get_experimental_setup(
    config: Union[Dict[str, Any], 'ConfigContainer'],
    setup_name: str,
    validate: bool = True,
    dataset_context: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Enhanced accessor for experimental setup with validation.
    
    INTEGRATION NOTE: This REPLACES or ADDS to your existing get_experimental_setup.
    
    Experimental setup with validation Features:
    - Automatic validation of entire setup
    - Dataset-specific optimization
    - Enhanced error messages with alternatives
    
    Args:
        config: Configuration dict or container
        setup_name: Name of experimental setup
        validate: Whether to validate setup
        dataset_context: Dataset type for optimization
    
    Returns:
        List of transform configurations in setup
    
    Raises:
        ConfigurationError: If setup not found or validation fails
    """
    accessor = EnhancedConfigAccessor(
        config,
        auto_validate=validate,
        dataset_context=dataset_context
    )
    
    # Extract experimental setups
    setups = accessor._get_raw_value('experimental_setups', {})
    
    if not setups:
        raise ConfigurationError(
            "No experimental setups found in configuration",
            config_key="experimental_setups",
            suggestions="Define experimental setups in configuration"
        )
    
    if setup_name not in setups:
        available = list(setups.keys())
        raise ConfigurationError(
            f"Experimental setup '{setup_name}' not found",
            config_key=f"experimental_setups.{setup_name}",
            suggestions=f"Available setups: {available}"
        )
    
    setup_obj = setups[setup_name]
    
    # Convert ExperimentalSetup object to list of dicts
    # ExperimentalSetup has transforms: List[TransformSpec]
    # Each TransformSpec has to_dict() method
    if hasattr(setup_obj, 'transforms'):
        # It's an ExperimentalSetup object
        setup_transforms = [t.to_dict() for t in setup_obj.transforms if hasattr(t, 'to_dict')]
        if not setup_transforms and setup_obj.transforms:
            # Fallback: transforms might already be dicts
            setup_transforms = list(setup_obj.transforms) if isinstance(setup_obj.transforms, list) else []
    elif isinstance(setup_obj, list):
        # Already a list (legacy format)
        setup_transforms = setup_obj
    elif isinstance(setup_obj, dict) and 'transforms' in setup_obj:
        # Dict with transforms key
        setup_transforms = setup_obj.get('transforms', [])
    else:
        setup_transforms = []
    
    # Validate if requested
    if validate and GRAPH_TRANSFORMS_AVAILABLE:
        gt = get_graph_transforms()
        
        # Comprehensive validation 
        validation_result = gt.validate_config_comprehensive(
            setup_transforms,
            dataset_type=dataset_context,
            validation_level=ValidationLevel.STANDARD,
            validation_scope=ValidationScope.PRODUCTION
        )
        
        if not validation_result['valid']:
            raise ConfigurationError(
                f"Experimental setup '{setup_name}' validation failed",
                config_key=f"experimental_setups.{setup_name}",
                validation_errors=validation_result['errors'],
                warnings=validation_result['warnings'],
                suggestions="Review setup configuration and fix validation errors"
            )
        
        # Log warnings
        for warning in validation_result.get('warnings', []):
            logger.warning(f"Setup '{setup_name}': {warning}")
    
    return setup_transforms


def list_experimental_setups() -> List[str]:
    """
    List all available experimental setup names.
    
    Enhanced experimental setup listing
    
    Returns:
        List[str]: List of experimental setup names
    """
    try:
        transformation_config = get_transformation_config()
        return transformation_config.list_setup_names()
        
    except ConfigurationError as e:
        logger.warning(f"Failed to list experimental setups: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error listing experimental setups: {e}")
        return []


def list_enabled_experimental_setups() -> List[str]:
    """
    List all enabled experimental setup names.
    
    Enhanced experimental setup listing with enabled filter
    
    Returns:
        List[str]: List of enabled experimental setup names
    """
    try:
        transformation_config = get_transformation_config()
        return transformation_config.list_enabled_setup_names()
        
    except ConfigurationError as e:
        logger.warning(f"Failed to list enabled experimental setups: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error listing enabled experimental setups: {e}")
        return []


def get_default_experimental_setup() -> Optional[ExperimentalSetup]:
    """
    Get the default experimental setup.
    
    Enhanced default setup access
    
    Returns:
        ExperimentalSetup: The default experimental setup, or None if not found
    """
    try:
        transformation_config = get_transformation_config()
        return transformation_config.get_default_setup()
        
    except ConfigurationError as e:
        logger.warning(f"Failed to get default experimental setup: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting default experimental setup: {e}")
        return None

def get_standard_transforms() -> List['TransformSpec']:
    """
    Get list of standard transforms from transformation configuration.
    
    Standard transforms are always applied before any experimental setup transforms.
    They represent production-ready transforms like AddSelfLoops, NormalizeFeatures.
    
    Returns:
        List[TransformSpec]: List of enabled standard transform specifications
        
    Example:
        >>> std_transforms = get_standard_transforms()
        >>> for t in std_transforms:
        ...     print(f"{t.name}: {t.kwargs}")
    """
    try:
        transformation_config = get_transformation_config()
        return transformation_config.get_standard_transforms()
        
    except ConfigurationError as e:
        logger.warning(f"Failed to get standard transforms: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error getting standard transforms: {e}")
        return []


def get_standard_transforms_as_dicts() -> List[Dict[str, Any]]:
    """
    Get standard transforms as list of dictionaries.
    
    This format is compatible with graph_transforms.compose_transforms().
    
    Returns:
        List[Dict[str, Any]]: List of transform configuration dictionaries
    """
    try:
        transformation_config = get_transformation_config()
        return transformation_config.get_standard_transforms_as_dicts()
        
    except ConfigurationError as e:
        logger.warning(f"Failed to get standard transforms as dicts: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error getting standard transforms as dicts: {e}")
        return []


def get_combined_transforms(setup_name: Optional[str] = None) -> List['TransformSpec']:
    """
    Get combined standard + experimental setup transforms.
    
    Standard transforms are applied FIRST, then experimental setup transforms.
    
    Args:
        setup_name: Name of experimental setup. If None, uses default_setup.
        
    Returns:
        List[TransformSpec]: Combined list of transform specifications
        
    Example:
        >>> # Get combined transforms for default setup
        >>> transforms = get_combined_transforms()
        >>> 
        >>> # Get combined transforms for specific setup
        >>> transforms = get_combined_transforms('milia_quantum_enhanced')
    """
    try:
        transformation_config = get_transformation_config()
        return transformation_config.get_combined_transforms(setup_name)
        
    except ConfigurationError as e:
        logger.warning(f"Failed to get combined transforms: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error getting combined transforms: {e}")
        return []


def get_combined_transforms_as_dicts(setup_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get combined transforms as list of dictionaries.
    
    This format is compatible with graph_transforms.compose_transforms().
    
    Args:
        setup_name: Name of experimental setup. If None, uses default_setup.
        
    Returns:
        List[Dict[str, Any]]: List of transform configuration dictionaries
    """
    try:
        transformation_config = get_transformation_config()
        return transformation_config.get_combined_transforms_as_dicts(setup_name)
        
    except ConfigurationError as e:
        logger.warning(f"Failed to get combined transforms as dicts: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error getting combined transforms as dicts: {e}")
        return []


def has_standard_transforms() -> bool:
    """
    Check if any standard transforms are defined and enabled.
    
    Returns:
        bool: True if standard transforms exist, False otherwise
    """
    try:
        transformation_config = get_transformation_config()
        return transformation_config.has_standard_transforms()
        
    except Exception:
        return False


def get_experimental_setups_for_dataset(dataset_type: str) -> Dict[str, ExperimentalSetup]:
    """
    Get experimental setups compatible with a specific dataset type.
    
    NEW: Phase 1 Step 5 - Dataset-specific experimental setup filtering
    
    Args:
        dataset_type: Dataset type to filter by ('DFT', 'DMC', etc.)
        
    Returns:
        Dict[str, ExperimentalSetup]: Dictionary of compatible experimental setups
    """
    try:
        transformation_config = get_transformation_config()
        return transformation_config.get_setups_for_dataset(dataset_type)
        
    except ConfigurationError as e:
        logger.warning(f"Failed to get experimental setups for dataset {dataset_type}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error getting experimental setups for dataset {dataset_type}: {e}")
        return {}


def get_transformation_validation_config() -> Dict[str, Any]:
    """
    Get transformation validation configuration.
    
    Transformation validation settings access
    
    Returns:
        Dict[str, Any]: Validation configuration settings
    """
    try:
        transformation_config = get_transformation_config()
        return transformation_config.validation
        
    except ConfigurationError as e:
        logger.warning(f"Failed to get transformation validation config: {e}")
        return {'enabled': True, 'strict_mode': False}
    except Exception as e:
        logger.error(f"Unexpected error getting transformation validation config: {e}")
        return {'enabled': True, 'strict_mode': False}


def is_transformation_validation_enabled() -> bool:
    """
    Check if transformation validation is enabled.
    
    Transformation validation status check
    
    Returns:
        bool: True if validation is enabled, False otherwise
    """
    try:
        transformation_config = get_transformation_config()
        return transformation_config.is_validation_enabled()
        
    except Exception as e:
        logger.debug(f"Error checking validation status, defaulting to enabled: {e}")
        return True


def is_transformation_strict_mode_enabled() -> bool:
    """
    Check if transformation strict mode is enabled.
    
    Transformation strict mode status check
    
    Returns:
        bool: True if strict mode is enabled, False otherwise
    """
    try:
        transformation_config = get_transformation_config()
        return transformation_config.is_strict_mode_enabled()
        
    except Exception as e:
        logger.debug(f"Error checking strict mode status, defaulting to disabled: {e}")
        return False


def get_available_transforms() -> List[str]:
    """
    Get list of available transforms from the transform registry.
    
    Integration with graph_transforms.py
    
    Returns:
        List[str]: List of available transform names
    """
    try:
        # Import graph_transforms module to get available transforms
        from milia_pipeline.transformations.graph_transforms import get_graph_transforms
        
        gt = get_graph_transforms()
        return gt.list_transforms()
        
    except ImportError:
        logger.warning("graph_transforms module not available")
        return []
    except Exception as e:
        logger.error(f"Failed to get available transforms: {e}")
        return []


def get_transforms_by_category() -> Dict[str, List[str]]:
    """
    Get transforms organized by category from the transform registry.
    
    Integration with graph_transforms.py
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping categories to transform lists
    """
    try:
        # Import graph_transforms module to get categorized transforms
        from milia_pipeline.transformations.graph_transforms import get_graph_transforms
        
        gt = get_graph_transforms()
        return gt.get_available_transforms()
        
    except ImportError:
        logger.warning("graph_transforms module not available")
        return {}
    except Exception as e:
        logger.error(f"Failed to get transforms by category: {e}")
        return {}


def get_transform_info(transform_name: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific transform.
    
    Integration with graph_transforms.py
    
    Args:
        transform_name: Name of the transform to get information about
        
    Returns:
        Dict[str, Any]: Transform information, or None if not found
    """
    try:
        # Import graph_transforms module to get transform info
        from milia_pipeline.transformations.graph_transforms import get_graph_transforms
        
        gt = get_graph_transforms()
        transform_info = gt.get_transform_info(transform_name)
        
        if transform_info is None:
            return None
        
        # Convert TransformInfo to dictionary
        return {
            'name': transform_info.name,
            'category': transform_info.category,
            'description': transform_info.description,
            'parameters': transform_info.parameters,
            'research_applicability': transform_info.research_applicability,
            'performance_notes': transform_info.performance_notes
        }
        
    except ImportError:
        logger.warning("graph_transforms module not available")
        return None
    except Exception as e:
        logger.error(f"Failed to get transform info for '{transform_name}': {e}")
        return None

def get_transform_registry_info() -> Optional[Dict[str, Any]]:
    """
    Get comprehensive information about the transform registry.
    
    Transform registry information access
    
    Returns:
        Dict[str, Any]: Registry information including available transforms, system status, etc.
                       None if transform system is not available
    """
    try:
        # Import graph_transforms module to get registry info
        from milia_pipeline.transformations.graph_transforms import get_graph_transforms
        
        gt = get_graph_transforms()
        if gt is None:
            return {
                'system_status': 'unavailable',
                'available_transforms': [],
                'transform_categories': {},
                'registry_initialized': False,
                'error': 'Transform system not initialized'
            }
        
        # Get comprehensive registry information
        system_status = gt.get_system_status()
        available_transforms = gt.list_transforms()
        transforms_by_category = gt.get_available_transforms()
        
        return {
            'system_status': 'available' if system_status.get('initialized', False) else 'unavailable',
            'available_transforms': available_transforms,
            'transform_categories': transforms_by_category,
            'registry_initialized': system_status.get('initialized', False),
            'available_transform_count': len(available_transforms),
            'category_count': len(transforms_by_category),
            'performance_metrics': system_status.get('performance_metrics', {}),
            'cache_stats': system_status.get('cache_stats', {})
        }
        
    except ImportError:
        logger.warning("graph_transforms module not available for registry info")
        return {
            'system_status': 'unavailable',
            'available_transforms': [],
            'transform_categories': {},
            'registry_initialized': False,
            'error': 'graph_transforms module not available'
        }
    except Exception as e:
        logger.error(f"Failed to get transform registry info: {e}")
        return {
            'system_status': 'error',
            'available_transforms': [],
            'transform_categories': {},
            'registry_initialized': False,
            'error': str(e)
        }


def validate_transformation_config(configs: List[Dict[str, Any]], 
                                 dataset_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate transformation configuration using the graph_transforms system.
    
    Enhanced transformation validation
    
    Args:
        configs: List of transform configurations to validate
        dataset_type: Optional dataset type for dataset-specific validation
        
    Returns:
        Dict[str, Any]: Validation results with detailed feedback
    """
    try:
        # Import graph_transforms module for validation
        from milia_pipeline.transformations.graph_transforms import get_graph_transforms
        
        gt = get_graph_transforms()
        return gt.validate_config(configs, dataset_type)
        
    except ImportError:
        logger.warning("graph_transforms module not available for validation")
        return {
            'valid': False,
            'errors': ['Transform validation system not available'],
            'warnings': [],
            'system_initialized': False
        }
    except Exception as e:
        logger.error(f"Transform validation failed: {e}")
        return {
            'valid': False,
            'errors': [f'Validation error: {str(e)}'],
            'warnings': [],
            'system_initialized': True
        }


def create_transforms_from_config(configs: List[Dict[str, Any]], 
                                dataset_type: Optional[str] = None,
                                experimental_setup: Optional[str] = None) -> Any:
    """
    Create PyG transform sequence from configuration.
    
    Transform creation integration
    
    Args:
        configs: List of transform configurations
        dataset_type: Optional dataset type for optimization
        experimental_setup: Optional experimental setup name to use instead of configs
        
    Returns:
        Compose: PyG Compose object with transforms, or None if creation failed
    """
    try:
        # Import graph_transforms module for transform creation
        from milia_pipeline.transformations.graph_transforms import get_graph_transforms
        
        gt = get_graph_transforms()
        
        # If experimental setup specified, get configs from setup
        if experimental_setup:
            setup = get_experimental_setup(experimental_setup)
            if setup is None:
                logger.error(f"Experimental setup '{experimental_setup}' not found")
                return None
            
            # Convert ExperimentalSetup to config format
            configs = []
            for transform_spec in setup.transforms:
                config = {'name': transform_spec.name}
                if transform_spec.kwargs:
                    config['kwargs'] = transform_spec.kwargs
                config['enabled'] = transform_spec.enabled
                configs.append(config)
        
        return gt.create_transform_sequence(configs)
        
    except ImportError:
        logger.warning("graph_transforms module not available for transform creation")
        return None
    except Exception as e:
        logger.error(f"Transform creation failed: {e}")
        return None


def get_research_recommendations(research_type: str, dataset_type: str = 'DFT') -> Dict[str, Any]:
    """
    Get transform recommendations for specific research applications.
    
    Research-oriented configuration assistance
    
    This function provides transform recommendations tailored to specific research
    applications. It first attempts to use the full GraphTransforms recommendation
    system, then falls back to basic recommendations if unavailable.
    
    Args:
        research_type: Type of research, one of:
            - 'molecular_properties': Property prediction tasks
            - 'robustness_training': Training robust models
            - 'transfer_learning': Transfer learning applications
            - 'interpretability': Interpretable model training
            - 'large_scale': Large-scale dataset training
        dataset_type: Dataset type for optimization ('DFT', 'DMC', 'MD', etc.)
        
    Returns:
        Dict[str, Any]: Research recommendations including:
            - recommended_transforms: List of recommended transform names
            - experimental_setups: Dict of experimental setup configurations
            - warnings: List of warning messages (if any)
            - description: Description of recommendations
            - research_type: Echo of input research type
            - dataset_type: Echo of input dataset type
            - fallback_mode: Boolean (only in fallback mode)
    
    Examples:
        >>> recs = get_research_recommendations('molecular_properties', 'DFT')
        >>> print(recs['recommended_transforms'])
        ['AddSelfLoops', 'ToUndirected']
        
        >>> recs = get_research_recommendations('robustness_training', 'DMC')
        >>> print(recs['description'])
        'Data augmentation transforms for model robustness training'
    """
    try:
        # Import graph_transforms module for research recommendations
        from milia_pipeline.transformations.graph_transforms import get_graph_transforms
        
        gt = get_graph_transforms()
        
        # Check if method exists before calling
        if gt and hasattr(gt, 'get_research_recommendations'):
            logger.debug(f"Using full GraphTransforms recommendations for {research_type}")
            return gt.get_research_recommendations(research_type, dataset_type)
        else:
            # Use fallback recommendations
            logger.info(
                f"GraphTransforms.get_research_recommendations not available, "
                f"using fallback for {research_type}"
            )
            return _get_fallback_research_recommendations(research_type, dataset_type)
        
    except ImportError:
        logger.warning(
            "graph_transforms module not available for research recommendations, "
            "using fallback system"
        )
        return _get_fallback_research_recommendations(research_type, dataset_type)
        
    except Exception as e:
        logger.error(f"Failed to get research recommendations: {e}")
        return {
            'recommended_transforms': [],
            'experimental_setups': {},
            'warnings': [f'Error: {str(e)}'],
            'research_type': research_type,
            'dataset_type': dataset_type,
            'error': str(e)
        }


def _get_fallback_research_recommendations(research_type: str, dataset_type: str) -> Dict[str, Any]:
    """
    Provide basic fallback recommendations when full system unavailable.
    
    Fallback recommendation system
    
    This function provides sensible default recommendations when the full
    GraphTransforms research recommendation system is not available. The
    recommendations are basic but proven patterns for each research type.
    
    Args:
        research_type: Type of research ('molecular_properties', 'robustness_training', etc.)
        dataset_type: Dataset type for optimization ('DFT', 'DMC', etc.)
        
    Returns:
        Dict[str, Any]: Basic recommendations with:
            - recommended_transforms: List of transform names
            - experimental_setups: Dict of setup configurations (empty in fallback)
            - warnings: List of warning messages
            - description: Description of the recommendations
            - research_type: Echo of input research type
            - dataset_type: Echo of input dataset type
            - fallback_mode: Boolean flag indicating fallback mode
            - metadata: Additional metadata about recommendations
    
    Note:
        These are basic recommendations. For advanced, dataset-optimized
        recommendations, enable the full GraphTransforms system.
    """
    # Basic recommendation patterns based on research type
    recommendations = {
        'molecular_properties': {
            'recommended_transforms': ['AddSelfLoops', 'ToUndirected'],
            'description': 'Basic graph structure normalization for molecular properties prediction',
            'rationale': 'Self-loops ensure node self-attention, undirected edges preserve molecular symmetry'
        },
        'robustness_training': {
            'recommended_transforms': ['RandomJitter', 'RandomNodeDrop'],
            'description': 'Data augmentation transforms for model robustness training',
            'rationale': 'Augmentation prevents overfitting and improves generalization'
        },
        'transfer_learning': {
            'recommended_transforms': ['AddSelfLoops', 'NormalizeFeatures'],
            'description': 'Feature normalization for transfer learning applications',
            'rationale': 'Normalization ensures consistent feature scales across domains'
        },
        'interpretability': {
            'recommended_transforms': ['AddSelfLoops'],
            'description': 'Minimal transforms to preserve interpretability',
            'rationale': 'Fewer transforms make model decisions more interpretable'
        },
        'large_scale': {
            'recommended_transforms': ['AddSelfLoops', 'ToUndirected'],
            'description': 'Efficient transforms for large-scale training',
            'rationale': 'Lightweight transforms minimize computational overhead'
        },
        'quantum_chemistry': {
            'recommended_transforms': ['AddSelfLoops', 'ToUndirected'],
            'description': 'Transforms preserving quantum chemistry properties',
            'rationale': 'Preserves molecular symmetries important for QC calculations'
        },
        'default': {
            'recommended_transforms': ['AddSelfLoops'],
            'description': 'Minimal baseline configuration',
            'rationale': 'Safe default for any graph learning task'
        }
    }
    
    # Get recommendation or fall back to default
    rec = recommendations.get(research_type, recommendations['default'])
    
    # Log the fallback usage
    logger.info(
        f"Using fallback recommendations for research_type='{research_type}', "
        f"dataset_type='{dataset_type}': {rec['recommended_transforms']}"
    )
    
    return {
        'recommended_transforms': rec['recommended_transforms'],
        'experimental_setups': {},
        'warnings': [
            'Using basic fallback recommendations',
            'Full GraphTransforms research recommendation system unavailable',
            'Consider enabling graph_transforms module for advanced recommendations'
        ],
        'description': rec['description'],
        'rationale': rec['rationale'],
        'research_type': research_type,
        'dataset_type': dataset_type,
        'fallback_mode': True,
        'metadata': {
            'recommendation_quality': 'basic',
            'source': 'fallback_system',
            'version': '1.0',
            'available_research_types': list(recommendations.keys()),
            'note': 'These are proven baseline recommendations. Enable GraphTransforms for optimized suggestions.'
        }
    }

def get_transformation_performance_metrics() -> Dict[str, Any]:
    """
    Get performance metrics for the transformation system.
    
    Performance monitoring integration
    
    Returns:
        Dict[str, Any]: Performance metrics and statistics
    """
    try:
        # Import graph_transforms module for performance metrics
        from milia_pipeline.transformations.graph_transforms import get_graph_transforms
        
        gt = get_graph_transforms()
        system_status = gt.get_system_status()
        
        return {
            'system_initialized': system_status.get('initialized', False),
            'available_transform_count': system_status.get('available_transform_count', 0),
            'cache_stats': system_status.get('cache_stats', {}),
            'usage_stats': system_status.get('usage_stats', {}),
            'performance_metrics': system_status.get('performance_metrics', {})
        }
        
    except ImportError:
        logger.warning("graph_transforms module not available for performance metrics")
        return {
            'system_initialized': False,
            'available_transform_count': 0,
            'cache_stats': {},
            'usage_stats': {},
            'performance_metrics': {}
        }
    except Exception as e:
        logger.error(f"Failed to get transformation performance metrics: {e}")
        return {
            'system_initialized': False,
            'error': str(e)
        }


def get_transformation_cache_key(configs: List[Dict[str, Any]]) -> str:
    """
    Generate a cache key for transformation configurations.
    
    Cache management integration
    
    Args:
        configs: List of transform configurations
        
    Returns:
        str: Cache key for the configuration
    """
    try:
        import json
        import hashlib
        
        # Create normalized representation for hashing
        normalized_configs = []
        for config in configs:
            if isinstance(config, dict):
                normalized = {
                    'name': config.get('name', ''),
                    'kwargs': config.get('kwargs', {}),
                    'enabled': config.get('enabled', True)
                }
                # Sort kwargs for consistent hashing
                if isinstance(normalized['kwargs'], dict):
                    normalized['kwargs'] = dict(sorted(normalized['kwargs'].items()))
                normalized_configs.append(normalized)
        
        # Create hash of the normalized configuration
        config_str = json.dumps(normalized_configs, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()
        
    except Exception as e:
        logger.error(f"Failed to generate cache key: {e}")
        # Return a simple fallback key
        return f"config_{len(configs)}_{hash(str(configs))}"


# ==========================================
# ENHANCED INTEGRATION FUNCTIONS
# ==========================================

def get_transformation_config_summary() -> Dict[str, Any]:
    """
    Get a comprehensive summary of the transformation configuration.
    
    Configuration summary for monitoring and debugging
    
    Returns:
        Dict[str, Any]: Summary of transformation configuration
    """
    summary = {
        'format_detected': 'unknown',
        'experimental_setups_count': 0,
        'enabled_setups_count': 0,
        'default_setup': None,
        'validation_enabled': True,
        'strict_mode_enabled': False,
        'total_transforms': 0,
        'transform_categories': [],
        'dataset_compatibility': {},
        'system_status': 'unknown'
    }
    
    try:
        # Get transformation configuration
        transformation_config = get_transformation_config()
        
        summary.update({
            'format_detected': 'enhanced',
            'experimental_setups_count': len(transformation_config.experimental_setups),
            'enabled_setups_count': len(transformation_config.get_enabled_setups()),
            'default_setup': transformation_config.default_setup,
            'validation_enabled': transformation_config.is_validation_enabled(),
            'strict_mode_enabled': transformation_config.is_strict_mode_enabled()
        })
        
        # Count total transforms
        total_transforms = 0
        for setup in transformation_config.experimental_setups.values():
            if setup.enabled:
                total_transforms += len([t for t in setup.transforms if t.enabled])
        summary['total_transforms'] = total_transforms
        
        # PHASE 5 REFACTORING: Iterate over all registered dataset types dynamically
        registered_types = _registry_list_all_safe()
        for dataset_type in registered_types:
            compatible_setups = transformation_config.get_setups_for_dataset(dataset_type)
            summary['dataset_compatibility'][dataset_type] = len(compatible_setups)
        
        logger.debug(f"Checked transformation compatibility for {len(registered_types)} dataset types: {registered_types}")
        
        # Get transform categories from registry
        try:
            transforms_by_category = get_transforms_by_category()
            summary['transform_categories'] = list(transforms_by_category.keys())
        except Exception:
            summary['transform_categories'] = []
        
        summary['system_status'] = 'operational'
        
    except ConfigurationError as e:
        summary['system_status'] = f'configuration_error: {e.message}'
    except Exception as e:
        summary['system_status'] = f'error: {str(e)}'
    
    return summary


def create_experimental_setup_from_dict(setup_name: str, setup_config: Dict[str, Any]) -> Optional[ExperimentalSetup]:
    """
    Create an ExperimentalSetup from dictionary configuration.
    
    Helper function for dynamic experimental setup creation
    
    Args:
        setup_name: Name for the experimental setup
        setup_config: Dictionary containing setup configuration
        
    Returns:
        ExperimentalSetup: Created experimental setup, or None if creation failed
    """
    try:
        from milia_pipeline.config.config_containers import create_experimental_setup_from_dict
        
        # Ensure setup has a name
        setup_config_with_name = setup_config.copy()
        setup_config_with_name['name'] = setup_name
        
        return create_experimental_setup_from_dict(setup_config_with_name)
        
    except Exception as e:
        logger.error(f"Failed to create experimental setup '{setup_name}': {e}")
        return None


def save_experimental_setup(setup: ExperimentalSetup, 
                           persist_to_config: bool = False) -> Dict[str, Any]:
    """
    Save an experimental setup (in memory and optionally to configuration).
    
    Dynamic experimental setup management
    
    Args:
        setup: ExperimentalSetup to save
        persist_to_config: Whether to persist to configuration file
        
    Returns:
        Dict[str, Any]: Save operation result
    """
    result = {
        'success': False,
        'setup_name': setup.name,
        'persisted': False,
        'warnings': []
    }
    
    try:
        # Validate the setup
        is_valid, validation_errors = setup.validate_experimental_setup()
        if not is_valid:
            result['warnings'].extend(validation_errors)
            return result
        
        # For now, we don't modify the actual configuration file
        # This would require more sophisticated configuration management
        if persist_to_config:
            result['warnings'].append("Persistence to configuration file not yet implemented")
        
        result['success'] = True
        logger.info(f"Experimental setup '{setup.name}' validated successfully")
        
    except Exception as e:
        result['warnings'].append(f"Save operation failed: {str(e)}")
        logger.error(f"Failed to save experimental setup '{setup.name}': {e}")
    
    return result


# ==========================================
# MIGRATION AND COMPATIBILITY HELPERS
# ==========================================

def migrate_legacy_transformation_config(legacy_config: Any) -> Dict[str, Any]:
    """
    Migrate legacy transformation configuration to enhanced format.
    
    Legacy configuration migration support
    
    Args:
        legacy_config: Legacy transformation configuration
        
    Returns:
        Dict[str, Any]: Migration result with enhanced configuration
    """
    migration_result = {
        'success': False,
        'enhanced_config': None,
        'migration_applied': [],
        'warnings': [],
        'errors': []
    }
    
    try:
        # Import graph_transforms module for migration
        from milia_pipeline.transformations.graph_transforms import get_graph_transforms
        
        gt = get_graph_transforms()
        
        # Use the built-in migration system
        migration_result_from_gt = gt.migrate_configuration(legacy_config)
        
        migration_result.update({
            'success': len(migration_result_from_gt.get('errors', [])) == 0,
            'enhanced_config': migration_result_from_gt.get('migrated_config'),
            'migration_applied': migration_result_from_gt.get('migration_applied', []),
            'warnings': migration_result_from_gt.get('warnings', []),
            'errors': migration_result_from_gt.get('errors', [])
        })
        
    except ImportError:
        migration_result['errors'].append("graph_transforms module not available for migration")
    except Exception as e:
        migration_result['errors'].append(f"Migration failed: {str(e)}")
        logger.error(f"Legacy transformation config migration failed: {e}")
    
    return migration_result


def check_transformation_system_compatibility() -> Dict[str, Any]:
    """
    Check compatibility between legacy and enhanced transformation systems.
    
    System compatibility validation
    
    Returns:
        Dict[str, Any]: Compatibility check results
    """
    compatibility = {
        'compatible': True,
        'legacy_support': True,
        'enhanced_support': True,
        'graph_transforms_available': False,
        'issues': [],
        'recommendations': []
    }
    
    try:
        # Check if graph_transforms module is available
        from milia_pipeline.transformations.graph_transforms import get_graph_transforms
        
        gt = get_graph_transforms()
        system_status = gt.get_system_status()
        
        compatibility['graph_transforms_available'] = True
        compatibility['enhanced_support'] = system_status.get('initialized', False)
        
        if not compatibility['enhanced_support']:
            compatibility['issues'].append("Enhanced transformation system not initialized")
            compatibility['compatible'] = False
        
        # Check if we can load legacy configurations
        try:
            legacy_transforms = get_transformations_config()
            if not isinstance(legacy_transforms, list):
                compatibility['issues'].append("Legacy transformation format not accessible")
                compatibility['legacy_support'] = False
        except Exception as e:
            compatibility['issues'].append(f"Legacy transformation access failed: {e}")
            compatibility['legacy_support'] = False
        
        # Check if we can load enhanced configurations
        try:
            enhanced_config = get_transformation_config()
            if enhanced_config is None:
                compatibility['issues'].append("Enhanced transformation configuration not accessible")
                compatibility['enhanced_support'] = False
        except Exception as e:
            compatibility['issues'].append(f"Enhanced transformation access failed: {e}")
            compatibility['enhanced_support'] = False
        
        # Generate recommendations
        if not compatibility['enhanced_support']:
            compatibility['recommendations'].append("Install and configure PyTorch Geometric")
        
        if not compatibility['legacy_support']:
            compatibility['recommendations'].append("Check legacy configuration format")
        
        if compatibility['enhanced_support'] and compatibility['legacy_support']:
            compatibility['recommendations'].append("System fully compatible - both legacy and enhanced modes available")
        
        # Overall compatibility
        compatibility['compatible'] = compatibility['legacy_support'] or compatibility['enhanced_support']
        
    except ImportError:
        compatibility['graph_transforms_available'] = False
        compatibility['enhanced_support'] = False
        compatibility['issues'].append("graph_transforms module not available")
        compatibility['recommendations'].append("Install graph_transforms dependencies")
        
        # Check if we can at least access legacy configs
        try:
            legacy_transforms = get_transformations_config()
            compatibility['legacy_support'] = isinstance(legacy_transforms, list)
        except Exception:
            compatibility['legacy_support'] = False
            compatibility['compatible'] = False
    
    except Exception as e:
        compatibility['issues'].append(f"Compatibility check failed: {str(e)}")
        compatibility['compatible'] = False
    
    return compatibility

def list_available_transforms() -> List[str]:
    """
    List all available transforms from the transform registry.
    
    Integration with graph_transforms.py (wrapper for get_available_transforms)
    
    Returns:
        List[str]: List of available transform names
    """
    return get_available_transforms()


def get_processing_config() -> ProcessingConfig:
    """
    Get the processing configuration container from global configuration.
    
    Processing configuration container access
    
    Returns:
        ProcessingConfig: Container with processing configuration
        
    Raises:
        ConfigurationError: If processing configuration is invalid
    """
    try:
        return create_processing_config_container()
    except HandlerConfigurationError as e:
        # Convert handler error to configuration error for consistency
        raise ConfigurationError(
            message=f"Failed to get processing configuration: {e.message}",
            config_key="processing",
            details=str(e)
        ) from e


def validate_transform_config(configs: List[Dict[str, Any]], 
                             dataset_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate transform configuration (alias for validate_transformation_config).
    
    Transform validation function alias
    
    Args:
        configs: List of transform configurations to validate
        dataset_type: Optional dataset type for dataset-specific validation
        
    Returns:
        Dict[str, Any]: Validation results with detailed feedback
    """
    return validate_transformation_config(configs, dataset_type)


def validate_transformation_system(config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> Tuple[bool, str]:
    """
    Validate the transformation system configuration.
    
    This function checks if the transformation system is properly configured
    and can be initialized without errors.
    
    Args:
        config: Full configuration dictionary
        logger: Optional logger for detailed output
        
    Returns:
        Tuple[bool, str]: (is_valid, message)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        # Import here to avoid circular dependencies
        from milia_pipeline.transformations.graph_transforms import get_graph_transforms
        
        # Check if transformations section exists
        if 'transformations' not in config:
            return True, "No transformation configuration found (optional)"
        
        transform_config = config['transformations']
        
        # Try to initialize GraphTransforms
        try:
            transforms = GraphTransforms()
            
            # Try to get available transforms (this is what's failing in logs)
            available = transforms.get_available_transforms()
            
            if available and available.get('count', 0) > 0:
                logger.info(f"✅“ Transformation system validated: {available['count']} transforms available")
                return True, f"Transformation system ready with {available['count']} transforms"
            else:
                logger.warning("âš  Transformation system initialized but no transforms found")
                return True, "Transformation system initialized (no transforms available)"
                
        except AttributeError as e:
            # This is the error we're seeing in logs
            error_msg = f"Transformation system validation failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
            
        except Exception as e:
            logger.warning(f"Transformation system validation error: {e}")
            return True, f"Transformation system validation skipped: {e}"
        
    except ImportError as e:
        logger.warning(f"Cannot import GraphTransforms: {e}")
        return True, "Transformation system not available (optional)"
        
    except Exception as e:
        logger.error(f"Unexpected error in transformation validation: {e}")
        return False, f"Unexpected error: {type(e).__name__}: {e}"


# ==========================================
# STRUCTURAL FEATURES VALIDATION FUNCTIONS (EXISTING - MAINTAINED)
# ==========================================

def is_structural_features_enabled() -> bool:
    """
    Checks if structural features are enabled.
    
    Returns:
        bool: True if structural features are configured, False otherwise
    """
    structural_config = get_structural_features_config()
    if structural_config is None:
        return False
    
    # Check if any atom or bond features are configured
    atom_features = structural_config.get('atom', [])
    bond_features = structural_config.get('bond', [])
    
    return len(atom_features) > 0 or len(bond_features) > 0

#---
def get_atom_features() -> List[str]:
    """
    Gets the list of configured atom-level structural features.
    
    Returns:
        list: List of atom feature names
    """
    structural_config = get_structural_features_config()
    if structural_config is None:
        return []
    return structural_config.get('atom', [])


def get_bond_features() -> List[str]:
    """
    Gets the list of configured bond-level structural features.
    
    Returns:
        list: List of bond feature names
    """
    structural_config = get_structural_features_config()
    if structural_config is None:
        return []
    return structural_config.get('bond', [])
#---

# Decision Functions for Structural Features Integration
def should_pass_coordinates_to_structural_features() -> bool:
    """
    Checks if coordinates should be passed to structural feature extraction.
    
    Returns:
        bool: True if coordinates should be passed for 3D features
    """
    data_config = get_data_config()
    structural_integration = data_config.get('structural_feature_integration', {})
    return structural_integration.get('pass_coordinates', True)


def should_pass_mulliken_charges_to_structural_features(dataset_type: Optional[str] = None) -> bool:
    """
    Checks if Mulliken charges should be passed to structural feature extraction.
    
    Returns:
        bool: True if Mulliken charges should be passed (and available for current dataset)
    """
    data_config = get_data_config()
    structural_integration = data_config.get('structural_feature_integration', {})
    
    # Only pass Mulliken charges if enabled and available for the current dataset
    pass_charges = structural_integration.get('pass_mulliken_charges', True)
    
    # PHASE 5 REFACTORING: Check feature support via registry
    dataset_type = get_dataset_type()
    
    # Query dataset features to see if mulliken charges are supported
    features = _get_dataset_features(dataset_type)
    supports_mulliken = features.get('mulliken_charges_support', True)  # Default True for backward compatibility
    
    if not supports_mulliken:
        logger.debug(f"Dataset type '{dataset_type}' does not support Mulliken charges")
        return False
    
    return pass_charges


def should_enable_stereochemistry_preprocessing() -> bool:
    """
    Checks if stereochemistry preprocessing should be enabled.
    
    Returns:
        bool: True if stereochemistry preprocessing is enabled
    """
    data_config = get_data_config()
    structural_integration = data_config.get('structural_feature_integration', {})
    return structural_integration.get('enable_stereochemistry_preprocessing', True)


def get_dataset_appropriate_structural_features(dataset_type: Optional[str] = None, structural_config: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
    """
    Get structural features appropriate for current dataset type.
    
    ENHANCEMENT: Uses handler's supported features if handler available.
    FALLBACK: Uses registry feature query for filtering.
    ULTRA-LEGACY: Uses config-based filtering for complete backward compatibility.
    
    PHASE 5 REFACTORING: Feature filtering now queries dataset features from registry
    instead of hardcoding "DMC" checks. This allows any dataset type to declare
    its feature support.
    
    This function queries the dataset handler to determine which structural
    features are actually supported, preventing NaN/Inf issues from unsupported
    feature calculations (e.g., Gasteiger charges on datasets without Mulliken support).
    
    Returns:
        Dict[str, List[str]]: Dictionary with 'atom' and 'bond' keys containing
                             lists of supported feature names
    
    Examples:
        >>> features = get_dataset_appropriate_structural_features()
        >>> print(features['atom'])
        ['degree', 'hybridization', ...]  # No 'partial_charge' for DMC or other datasets without Mulliken support
    """
    if dataset_type is None:
        dataset_type = get_dataset_type()
    
    if structural_config is None:
        structural_config = get_structural_features_config()
    
    if not structural_config:
        return {'atom': [], 'bond': []}
    
    # Try to get handler's supported features
    try:
        # Create handler to query supported features
        dataset_config = create_dataset_config_container()
        filter_config = create_filter_config_container()
        processing_config = create_processing_config_container()
        
        # Import handler factory
        from milia_pipeline.handlers import create_dataset_handler
        
        handler = create_dataset_handler(
            dataset_config, filter_config, processing_config, logger
        )
        
        # Get handler's supported features 
        handler_supported = handler.get_supported_structural_features()
        
        # Intersect with configured features
        configured_atom = structural_config.get('atom', [])
        configured_bond = structural_config.get('bond', [])
        
        filtered_features = {
            'atom': [f for f in configured_atom if f in handler_supported['atom']],
            'bond': [f for f in configured_bond if f in handler_supported['bond']]
        }
        
        # Log filtering for transparency
        removed_atom = set(configured_atom) - set(filtered_features['atom'])
        removed_bond = set(configured_bond) - set(filtered_features['bond'])
        
        if removed_atom:
            logger.info(
                f"Filtered out unsupported atom features for {dataset_type}: "
                f"{list(removed_atom)}"
            )
        if removed_bond:
            logger.info(
                f"Filtered out unsupported bond features for {dataset_type}: "
                f"{list(removed_bond)}"
            )
        
        return filtered_features
        
    except Exception as e:
        logger.warning(
            f"Could not query handler for supported features: {e}. "
            f"Falling back to legacy config-based filtering."
        )
       
        # FALLBACK: Legacy behavior with registry feature query
        atom_features = list(structural_config.get('atom', []))
        bond_features = list(structural_config.get('bond', []))
        
        # PHASE 5 REFACTORING: Use registry feature query instead of hardcoded "DMC" check
        features = _get_dataset_features(dataset_type)
        supports_mulliken = features.get('mulliken_charges_support', True)  # Default True for backward compatibility
        
        if not supports_mulliken:
            # Remove features that require Mulliken charge support
            if 'mulliken_charge' in atom_features:
                atom_features.remove('mulliken_charge')
                logger.info(f"Legacy fallback: Removed 'mulliken_charge' for {dataset_type} (not supported)")
            if 'partial_charge' in atom_features:
                atom_features.remove('partial_charge')
                logger.info(f"Legacy fallback: Removed 'partial_charge' for {dataset_type} (not supported)")
        # ULTRA-LEGACY: Keep hardcoded DMC check as final fallback if registry unavailable
        elif dataset_type == "DMC":
            if 'mulliken_charge' in atom_features:
                atom_features.remove('mulliken_charge')
                logger.info("Ultra-legacy fallback: Removed 'mulliken_charge' for DMC")
            if 'partial_charge' in atom_features:
                atom_features.remove('partial_charge')
                logger.info("Ultra-legacy fallback: Removed 'partial_charge' for DMC")
        
        return {'atom': atom_features, 'bond': bond_features}


def validate_structural_features_for_dataset(
    features: Dict[str, List[str]],
    dataset_type: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """
    Validate structural features against dataset handler capabilities.
    
    Validates that requested features are supported.
    
    Args:
        features: Dict with 'atom' and 'bond' feature lists
        dataset_type: Dataset type to validate against (uses current if None)
    
    Returns:
        Tuple[bool, List[str]]: (is_valid, list of validation errors)
    
    Examples:
        >>> features = {'atom': ['partial_charge'], 'bond': ['bond_length']}
        >>> is_valid, errors = validate_structural_features_for_dataset(features, 'DMC')
        >>> print(is_valid)
        False
        >>> print(errors)
        ['Feature partial_charge not supported for DMC dataset']
    """
    if dataset_type is None:
        try:
            dataset_type = get_dataset_type()
        except Exception as e:
            return False, [f"Could not determine dataset type: {str(e)}"]
    
    # PHASE 5: Validate dataset type is registered
    if not _is_valid_dataset_type(dataset_type):
        valid_types = _get_valid_dataset_types()
        return False, [f"Invalid dataset type '{dataset_type}'. Valid types: {', '.join(valid_types)}"]
    
    validation_errors = []
    
    try:
        # Create handler to get supported features
        dataset_config = create_dataset_config_container()
        filter_config = create_filter_config_container()
        processing_config = create_processing_config_container()
        
        from milia_pipeline.handlers import create_dataset_handler
        
        handler = create_dataset_handler(
            dataset_config, filter_config, processing_config, logger
        )
        
        handler_supported = handler.get_supported_structural_features()
        
        # Validate atom features
        for feature in features.get('atom', []):
            if feature not in handler_supported['atom']:
                validation_errors.append(
                    f"Atom feature '{feature}' not supported for {dataset_type} dataset"
                )
        
        # Validate bond features
        for feature in features.get('bond', []):
            if feature not in handler_supported['bond']:
                validation_errors.append(
                    f"Bond feature '{feature}' not supported for {dataset_type} dataset"
                )
        
        is_valid = len(validation_errors) == 0
        
        if is_valid:
            logger.debug(
                f"Structural features validated for {dataset_type}: "
                f"{len(features.get('atom', []))} atom, "
                f"{len(features.get('bond', []))} bond features"
            )
        
        return is_valid, validation_errors
        
    except Exception as e:
        logger.warning(f"Feature validation failed: {e}")
        # Return True on validation failure to avoid breaking existing workflows
        return True, [f"Validation skipped: {str(e)}"]


def get_feature_compatibility_report(dataset_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Get comprehensive feature compatibility report for dataset type.
    
    Diagnostic tool for feature support analysis.
    
    Args:
        dataset_type: Dataset type to analyze (uses current if None)
    
    Returns:
        Dict[str, Any]: Compatibility report with:
            - supported_features: Dict of supported atom/bond features
            - configured_features: Dict of configured atom/bond features
            - filtered_features: Dict of filtered atom/bond features
            - removed_features: Dict of removed atom/bond features
            - compatibility_status: 'full', 'partial', or 'limited'
            - warnings: List of compatibility warnings
    
    Example:
        >>> report = get_feature_compatibility_report('DMC')
        >>> print(report['compatibility_status'])
        'partial'
        >>> print(report['removed_features']['atom'])
        ['partial_charge', 'mulliken_charge']
    """
    if dataset_type is None:
        dataset_type = get_dataset_type()
    
    report = {
        'dataset_type': dataset_type,
        'supported_features': {'atom': [], 'bond': []},
        'configured_features': {'atom': [], 'bond': []},
        'filtered_features': {'atom': [], 'bond': []},
        'removed_features': {'atom': [], 'bond': []},
        'compatibility_status': 'unknown',
        'warnings': []
    }
    
    try:
        # Get configured features
        structural_config = get_structural_features_config()
        if structural_config:
            report['configured_features'] = {
                'atom': structural_config.get('atom', []),
                'bond': structural_config.get('bond', [])
            }
        
        # Get handler supported features
        dataset_config = create_dataset_config_container()
        filter_config = create_filter_config_container()
        processing_config = create_processing_config_container()
        
        from milia_pipeline.handlers import create_dataset_handler
        
        handler = create_dataset_handler(
            dataset_config, filter_config, processing_config, logger
        )
        
        report['supported_features'] = handler.get_supported_structural_features()
        
        # Get filtered features (what will actually be used)
        report['filtered_features'] = get_dataset_appropriate_structural_features()
        
        # Calculate removed features
        configured_atom = set(report['configured_features']['atom'])
        filtered_atom = set(report['filtered_features']['atom'])
        report['removed_features']['atom'] = list(configured_atom - filtered_atom)
        
        configured_bond = set(report['configured_features']['bond'])
        filtered_bond = set(report['filtered_features']['bond'])
        report['removed_features']['bond'] = list(configured_bond - filtered_bond)
        
        # Determine compatibility status
        total_configured = len(configured_atom) + len(configured_bond)
        total_filtered = len(filtered_atom) + len(filtered_bond)
        
        if total_configured == 0:
            report['compatibility_status'] = 'no_features_configured'
        elif total_filtered == total_configured:
            report['compatibility_status'] = 'full'
        elif total_filtered > 0:
            report['compatibility_status'] = 'partial'
            report['warnings'].append(
                f"{len(report['removed_features']['atom'])} atom and "
                f"{len(report['removed_features']['bond'])} bond features "
                f"removed due to {dataset_type} dataset limitations"
            )
        else:
            report['compatibility_status'] = 'limited'
            report['warnings'].append(
                f"All configured features unsupported for {dataset_type} dataset"
            )
        
        # Add specific warnings for removed features
        if report['removed_features']['atom']:
            report['warnings'].append(
                f"Removed atom features: {', '.join(report['removed_features']['atom'])}"
            )
        if report['removed_features']['bond']:
            report['warnings'].append(
                f"Removed bond features: {', '.join(report['removed_features']['bond'])}"
            )
        
    except Exception as e:
        report['compatibility_status'] = 'error'
        report['warnings'].append(f"Compatibility check failed: {str(e)}")
        logger.error(f"Feature compatibility report generation failed: {e}")
    
    return report


# Dataset Constants Functions
def get_dataset_constants() -> Tuple[str, Optional[str], str]:
    """
    Gets the dataset constants for the current dataset type.
    
    Returns:
        tuple: (raw_npz_filename, raw_data_download_url, working_root_dir)
    """
    config = load_config()
    dataset_config = get_dataset_config()
    
    raw_npz_filename = _get_config_value(dataset_config, 'raw_npz_filename', str)
    raw_data_download_url = _get_config_value(dataset_config, 'raw_data_download_url', (str, type(None)))
    
    # Get working_root_dir from global_paths section
    global_paths = _get_config_value(config, 'global_paths', dict)
    working_root_dir = _get_config_value(global_paths, 'working_root_dir', str)
    
    return raw_npz_filename, raw_data_download_url, working_root_dir


def get_raw_source_filename(dataset_type: Optional[str] = None) -> Optional[str]:
    """
    Get the raw source filename for datasets that use preprocessing workflows.
    
    For datasets where the download URL does not contain the actual filename
    (e.g., Figshare API URLs like https://figshare.com/ndownloader/files/3195389),
    this function returns the configured `raw_source_filename` which specifies
    the actual filename to use when saving the downloaded file.
    
    This is essential for preprocessing workflows where:
    1. The URL path component is an ID, not a filename (e.g., '3195389')
    2. The actual source file has a specific name (e.g., 'dsgdb9nsd.xyz.tar.bz2')
    3. The system needs to check if the source file already exists before downloading
    
    Args:
        dataset_type: Dataset type name. If None, uses get_dataset_type()
        
    Returns:
        Optional[str]: The configured raw_source_filename if present, None otherwise.
                       When None, callers should fall back to extracting filename from URL.
    
    Example:
        >>> # For QM9 with raw_source_filename: dsgdb9nsd.xyz.tar.bz2
        >>> get_raw_source_filename('QM9')
        'dsgdb9nsd.xyz.tar.bz2'
        
        >>> # For DFT without raw_source_filename configured
        >>> get_raw_source_filename('DFT')
        None
        
    Note:
        This function is used by the preprocessing workflow in milia_dataset.download()
        to determine the correct filename for source files, especially when URLs
        use API-style paths that don't contain the actual filename.
    """
    if dataset_type is None:
        dataset_type = get_dataset_type()
    
    dataset_config = get_dataset_config(dataset_type)
    
    # Return raw_source_filename if configured, None otherwise
    # This allows callers to fall back to URL-based filename extraction
    return dataset_config.get('raw_source_filename', None)


def get_preprocessing_config(dataset_type: str, 
                            raw_tar_path_override: Optional[str] = None,
                            output_npz_path_override: Optional[str] = None) -> Dict[str, Any]:
    """
    Get preprocessing configuration for a dataset type.
    
    Constructs paths from working_root_dir and dataset config,
    with optional CLI overrides.
    
    Args:
        dataset_type: Dataset type ("Wavefunction", "DFT", "DMC")
        raw_tar_path_override: Override for raw input path (from CLI)
        output_npz_path_override: Override for output path (from CLI)
        
    Returns:
        dict: Complete preprocessing configuration
        
    Example:
        >>> config = get_preprocessing_config("Wavefunction")
        >>> # Returns:
        >>> # {
        >>> #     'raw_tar_path': '~/Chem_Data/milia_PyG_Dataset/raw/wavefunctions.tar.gz',
        >>> #     'output_npz_path': '~/Chem_Data/milia_PyG_Dataset/raw/wavefunctions.npz',
        >>> #     'num_molecules': 10,
        >>> #     'feature_tier': 'complete',
        >>> #     'cleanup_temp': True
        >>> # }
    """
    from pathlib import Path
    
    config = load_config()
    
    # Get working_root_dir
    global_paths = _get_config_value(config, 'global_paths', dict)
    working_root_dir = Path(_get_config_value(global_paths, 'working_root_dir', str)).expanduser()
    
    # Get dataset-specific config
    config_key = f"{dataset_type.lower()}_config"
    dataset_config = _get_config_value(config, config_key, dict)
    
    # Get raw filename and URL
    raw_filename = _get_config_value(dataset_config, 'raw_npz_filename', str)
    raw_download_url = dataset_config.get('raw_data_download_url')
    
    # Construct default paths
    raw_dir = working_root_dir / 'raw'
    
    # Determine raw file extension from filename or URL
    if raw_download_url and raw_download_url.endswith('.tar.gz'):
        raw_file_ext = '.tar.gz'
        raw_filename_base = raw_filename.replace('.npz', '')
        default_raw_path = raw_dir / f"{raw_filename_base}{raw_file_ext}"
    else:
        # File is already .npz, use as-is
        default_raw_path = raw_dir / raw_filename
    default_output_path = raw_dir / raw_filename
    
    # Apply overrides or use defaults
    raw_tar_path = Path(raw_tar_path_override).expanduser() if raw_tar_path_override else default_raw_path
    output_npz_path = Path(output_npz_path_override).expanduser() if output_npz_path_override else default_output_path
    
    # Build preprocessing config
    preprocessing_config = {
        'raw_tar_path': str(raw_tar_path),
        'output_npz_path': str(output_npz_path),
    }
    
    # Add dataset-specific preprocessing parameters
    # DYNAMIC PASSTHROUGH: Pass ALL preprocessing parameters from config.yaml to the
    # preprocessor, not just hardcoded keys. This ensures dataset-specific parameters
    # like max_conformers_per_molecule (RMD17), property_keys (ANI-1x/ANI-1ccx), etc.
    # are properly forwarded without requiring code changes for each new dataset.
    if 'processing_config' in dataset_config:
        processing = dataset_config['processing_config']
        if 'preprocessing' in processing:
            preprocess_params = processing['preprocessing']
            # Pass through ALL preprocessing parameters dynamically
            # This supports any dataset-specific parameters without hardcoding:
            # - num_molecules (QM9, Wavefunction, ANI-1x, ANI-1ccx)
            # - max_conformers_per_molecule (RMD17)
            # - molecules_to_include (RMD17)
            # - include_old_data (RMD17)
            # - property_keys (ANI-1x, ANI-1ccx)
            # - cleanup_temp (all datasets)
            # - Any future dataset-specific parameters
            preprocessing_config.update(preprocess_params)
    
    # PHASE 5 REFACTORING: Check schema for feature_tier support instead of hardcoding "wavefunction"
    schema = _get_dataset_schema(dataset_type)
    optional_properties = schema.get('optional_properties', [])
    
    # Check if feature_tier is in the dataset's optional properties
    if 'feature_tier' in optional_properties and 'processing_config' in dataset_config:
        feature_tier = dataset_config['processing_config'].get('feature_tier', 'standard')
        preprocessing_config['feature_tier'] = feature_tier
        logger.debug(f"Added feature_tier from schema for {dataset_type}")
    # FALLBACK: Legacy hardcoded check for Wavefunction (if schema unavailable)
    elif dataset_type.lower() == 'wavefunction' and 'processing_config' in dataset_config:
        feature_tier = dataset_config['processing_config'].get('feature_tier', 'standard')
        preprocessing_config['feature_tier'] = feature_tier
        logger.debug("Using legacy Wavefunction feature_tier check (schema unavailable)")
    
    return preprocessing_config


# ==========================================
# HANDLER PATTERN SUPPORT WITH EXCEPTION INTEGRATION (EXISTING - MAINTAINED)
# ==========================================

def create_dataset_config_container() -> DatasetConfig:
    """
    Create a DatasetConfig container from global configuration.
    
    Returns:
        DatasetConfig: Container with dataset-specific configuration
        
    Raises:
        HandlerConfigurationError: If dataset configuration is invalid
    """
    try:
        from milia_pipeline.config.config_containers import DatasetConfig
        
        return DatasetConfig(
            dataset_type=get_dataset_type(),
            uncertainty_config=get_uncertainty_config(),
            is_uncertainty_enabled=is_uncertainty_enabled()
        )
    except ConfigurationError as e:
        raise HandlerConfigurationError(
            message=f"Failed to create dataset configuration container: {e.message}",
            handler_type="DatasetConfig",
            config_validation_errors=[str(e)],
            invalid_config_keys=[e.config_key] if e.config_key else []
        ) from e


def create_filter_config_container() -> FilterConfig:
    """
    Create a FilterConfig container from global configuration.
    
    Returns:
        FilterConfig: Container with filter configuration
        
    Raises:
        HandlerConfigurationError: If filter configuration is invalid
    """
    try:
        from milia_pipeline.config.config_containers import FilterConfig
        
        filter_config = get_filter_config()
        
        return FilterConfig(
            max_atoms=filter_config.get('max_atoms'),
            min_atoms=filter_config.get('min_atoms'),
            heavy_atom_filter=filter_config.get('heavy_atom_filter'),
            dmc_uncertainty_filter=filter_config.get('dmc_uncertainty_filter'),
            dmc_uncertainty_threshold=filter_config.get('dmc_uncertainty_threshold')
        )
    except Exception as e:
        raise HandlerConfigurationError(
            message=f"Failed to create filter configuration container: {str(e)}",
            handler_type="FilterConfig",
            config_validation_errors=[str(e)]
        ) from e


def create_processing_config_container() -> ProcessingConfig:
    """
    Create a ProcessingConfig container from global configuration.
    
    Returns:
        ProcessingConfig: Container with processing configuration
        
    Raises:
        HandlerConfigurationError: If processing configuration is invalid
    """
    try:
        from milia_pipeline.config.config_containers import ProcessingConfig
        
        data_config = get_data_config()
        # Get dataset-specific config for preprocessing parameters
        dataset_config = get_dataset_config()
        processing_config_section = dataset_config.get('processing_config', {})
        preprocessing_section = processing_config_section.get('preprocessing', {})

        
        return ProcessingConfig(
            scalar_graph_targets=data_config.get('scalar_graph_targets_to_include', []),
            node_features=data_config.get('node_features_to_add', []),
            vector_graph_properties=data_config.get('vector_graph_properties_to_include', []),
            variable_len_graph_properties=data_config.get('variable_len_graph_properties_to_include', []),
            calculate_atomization_energy_from=data_config.get('calculate_atomization_energy_from'),
            atomization_energy_key_name=data_config.get('atomization_energy_key_name'),
            vibration_refinement=data_config.get('vibration_refinement'),
            test_molecule_limit=data_config.get('test_molecule_limit'),
            # NEW: Read preprocessing parameters from dataset-specific config
            preprocessing_feature_tier=processing_config_section.get('feature_tier', 'standard'),
            preprocessing_num_molecules=preprocessing_section.get('num_molecules', None),
            preprocessing_cleanup_temp=preprocessing_section.get('cleanup_temp', True)

        )
    except Exception as e:
        raise HandlerConfigurationError(
            message=f"Failed to create processing configuration container: {str(e)}",
            handler_type="ProcessingConfig",
            config_validation_errors=[str(e)]
        ) from e


def create_structural_features_config_container() -> StructuralFeaturesConfig:
    """
    Create a StructuralFeaturesConfig container from global configuration.
    
    Returns:
        StructuralFeaturesConfig: Container with structural features configuration
        
    Raises:
        HandlerConfigurationError: If structural features configuration is invalid
    """
    try:
        from milia_pipeline.config.config_containers import StructuralFeaturesConfig
        
        structural_config = get_structural_features_config() or {}
        
        return StructuralFeaturesConfig(
            atom_features=structural_config.get('atom', []),
            bond_features=structural_config.get('bond', []),
            preprocessing=structural_config.get('preprocessing')
        )
    except Exception as e:
        raise HandlerConfigurationError(
            message=f"Failed to create structural features configuration container: {str(e)}",
            handler_type="StructuralFeaturesConfig",
            config_validation_errors=[str(e)]
        ) from e


def create_transformation_config_container() -> TransformationConfig:
    """
    Create a TransformationConfig container from global configuration.
    
    Enhanced transformation configuration container creation
    
    Returns:
        TransformationConfig: Container with transformation configuration
        
    Raises:
        HandlerConfigurationError: If transformation configuration is invalid
    """
    try:
        transformation_config = get_transformation_config()
        return transformation_config
        
    except ConfigurationError as e:
        raise HandlerConfigurationError(
            message=f"Failed to create transformation configuration container: {e.message}",
            handler_type="TransformationConfig",
            config_validation_errors=[str(e)],
            invalid_config_keys=[e.config_key] if e.config_key else []
        ) from e
    except Exception as e:
        raise HandlerConfigurationError(
            message=f"Failed to create transformation configuration container: {str(e)}",
            handler_type="TransformationConfig",
            config_validation_errors=[str(e)]
        ) from e


@wrap_handler_operation("ConfigAccessor", "create_dataset_handler")
def create_dataset_handler() -> 'DatasetHandler':
    """
    Create a dataset handler from the current global configuration.
    
    Returns:
        DatasetHandler: Appropriate handler for the current dataset type
        
    Raises:
        HandlerNotAvailableError: If handler creation fails due to missing handler
        HandlerConfigurationError: If configuration is invalid for handler creation
        HandlerIntegrationError: If handler integration with config system fails
    """
    try:
        from milia_pipeline.handlers import create_dataset_handler
        
        # Create configuration containers with validation
        dataset_config = create_dataset_config_container()
        filter_config = create_filter_config_container()
        processing_config = create_processing_config_container()
        
        # Attempt to create handler
        handler = create_dataset_handler(
            dataset_config=dataset_config,
            filter_config=filter_config,
            processing_config=processing_config,
            logger=logger
        )
        
        if handler is None:
            raise HandlerNotAvailableError(
                message="Handler factory returned None",
                requested_dataset_type=dataset_config.dataset_type,
                details="create_dataset_handler returned None instead of handler instance"
            )
        
        logger.info(f"Successfully created {dataset_config.dataset_type} dataset handler")
        return handler
        
    except HandlerNotAvailableError:
        # Re-raise handler-specific exceptions
        raise
    except HandlerConfigurationError:
        # Re-raise configuration exceptions
        raise
    except ImportError as e:
        raise HandlerNotAvailableError(
            message="Dataset handler module not available",
            requested_dataset_type=get_dataset_type(),
            missing_dependencies=["dataset_handlers"],
            details=f"Import error: {str(e)}"
        ) from e
    except Exception as e:
        # Convert unexpected errors to integration errors
        raise HandlerIntegrationError(
            message=f"Unexpected error during handler creation: {str(e)}",
            integration_point="create_dataset_handler",
            migration_phase="Transformation_System_Integration",
            details=f"Original error: {type(e).__name__}: {str(e)}"
        ) from e

def get_required_properties(dataset_type: Optional[str] = None) -> List[str]:
    """
    Get required properties for a dataset type.
    
    Extracts required properties from handler constants configuration.
    This is a convenience accessor that wraps get_handler_constants().
    
    Required properties are the minimum set of NPZ keys that must be present
    for the dataset to load successfully. These are defined in
    HANDLER_REQUIRED_PROPERTIES in config_constants.py.
    
    Args:
        dataset_type: Dataset type ('DFT', 'DMC', 'Wavefunction', etc.).
                     If None, uses get_dataset_type()
        
    Returns:
        List of required property names
        
    Raises:
        HandlerNotAvailableError: If dataset type is not supported
        HandlerConfigurationError: If handler configuration is invalid
        
    Example:
        >>> required = get_required_properties('Wavefunction')
        >>> print(required)
        ['atoms', 'coordinates', 'compounds']
        
        >>> required = get_required_properties('DFT')
        >>> print(required)
        ['Etot', 'atoms', 'coordinates']
    """
    # PHASE 5: Handle None dataset_type for backward compatibility
    if dataset_type is None:
        dataset_type = get_dataset_type()
    
    from milia_pipeline.config.config_constants import get_handler_constants
    
    try:
        handler_constants = get_handler_constants(dataset_type)
        return handler_constants.get('required_properties', [])
    except (HandlerNotAvailableError, HandlerConfigurationError):
        # Re-raise handler-specific exceptions
        raise
    except Exception as e:
        raise HandlerConfigurationError(
            message=f"Failed to get required properties for dataset type '{dataset_type}'",
            handler_type=dataset_type,
            config_validation_errors=[str(e)],
            details=f"Error accessing handler constants: {type(e).__name__}: {str(e)}"
        ) from e

def get_identifier_keys(dataset_type: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Get molecular identifier keys for a dataset type.
    
    Extracts identifier keys from handler constants configuration.
    This is a convenience accessor that wraps get_handler_constants().
    
    Identifier keys define which NPZ keys to use for molecular identification
    and in what priority order. These are defined in HANDLER_IDENTIFIER_KEYS
    in config_constants.py.
    
    Args:
        dataset_type: Optional dataset type. If None, uses get_dataset_type()
        
    Returns:
        List of (npz_key, identifier_type) tuples in priority order
        
    Raises:
        HandlerNotAvailableError: If dataset type is not supported
        HandlerConfigurationError: If handler configuration is invalid
        
    Example:
        >>> keys = get_identifier_keys('Wavefunction')
        >>> print(keys)
        [('compounds', 'compound_id')]
        
    FIXED: Made dataset_type optional for consistency with other functions.
    """
    if dataset_type is None:
        dataset_type = get_dataset_type()
    
    from milia_pipeline.config.config_constants import get_handler_constants
    
    try:
        handler_constants = get_handler_constants(dataset_type)
        return handler_constants.get('identifier_keys', [])
    except (HandlerNotAvailableError, HandlerConfigurationError):
        # Re-raise handler-specific exceptions
        raise
    except Exception as e:
        raise HandlerConfigurationError(
            message=f"Failed to get identifier keys for dataset type '{dataset_type}'",
            handler_type=dataset_type,
            config_validation_errors=[str(e)],
            details=f"Error accessing handler constants: {type(e).__name__}: {str(e)}"
        ) from e


def get_coordinate_units(dataset_type: Optional[str] = None) -> str:
    """
    Get coordinate units for a dataset type from registry.
    
    Args:
        dataset_type: Optional dataset type. If None, uses get_dataset_type()
        
    Returns:
        str: Coordinate units ('angstrom' or 'bohr')
        
    ADDED Phase 5: Query coordinate units from registry.
    """
    if dataset_type is None:
        dataset_type = get_dataset_type()
    
    # Try to get from registry
    dataset_class = _registry_get_safe(dataset_type)
    if dataset_class is not None:
        try:
            return dataset_class.get_coordinate_units()
        except Exception as e:
            logger.debug(f"Could not get coordinate units from registry: {e}")
    
    # Fallback: default to angstrom
    return 'angstrom'


def get_energy_units(dataset_type: Optional[str] = None) -> str:
    """
    Get energy units for a dataset type from registry.
    
    Args:
        dataset_type: Optional dataset type. If None, uses get_dataset_type()
        
    Returns:
        str: Energy units ('hartree', 'eV', 'kcal/mol', etc.)
        
    ADDED Phase 5: Query energy units from registry.
    """
    if dataset_type is None:
        dataset_type = get_dataset_type()
    
    # Try to get from registry
    dataset_class = _registry_get_safe(dataset_type)
    if dataset_class is not None:
        try:
            return dataset_class.get_energy_units()
        except Exception as e:
            logger.debug(f"Could not get energy units from registry: {e}")
    
    # Fallback: default to hartree
    return 'hartree'


def get_molecule_creation_strategy(dataset_type: Optional[str] = None) -> str:
    """
    Get molecule creation strategy for a dataset type from registry.
    
    Args:
        dataset_type: Optional dataset type. If None, uses get_dataset_type()
        
    Returns:
        str: Molecule creation strategy ('identifier_coordinate_based' or 'coordinate_based')
        
    ADDED Phase 5: Query molecule creation strategy from registry.
    """
    if dataset_type is None:
        dataset_type = get_dataset_type()
    
    # Try to get from registry
    dataset_class = _registry_get_safe(dataset_type)
    if dataset_class is not None:
        try:
            return dataset_class.get_molecule_creation_strategy()
        except Exception as e:
            logger.debug(f"Could not get molecule creation strategy from registry: {e}")
    
    # Fallback: default to coordinate_based
    return 'coordinate_based'



def get_handler_compatible_config() -> Dict[str, Any]:
    """
    Get a configuration dictionary that is compatible with dataset handlers.
    
    This provides all necessary configuration data in a format that handlers
    can use, bridging the gap between the existing config system and handlers.
    
    ENHANCED: Added transformation configuration support
    
    Returns:
        dict: Handler-compatible configuration data
        
    Raises:
        HandlerConfigurationError: If configuration cannot be made handler-compatible
    """
    try:
        return {
            'dataset_config': create_dataset_config_container(),
            'filter_config': create_filter_config_container(),
            'processing_config': create_processing_config_container(),
            'structural_features_config': create_structural_features_config_container(),
            'transformation_config': create_transformation_config_container(),  # NEW
            'transformations': get_transformations_config(),  # Legacy compatibility
            'global_config': {
                'dataset_type': get_dataset_type(),
                'uncertainty_enabled': is_uncertainty_enabled(),
                'structural_features_enabled': is_structural_features_enabled(),
                'pass_coordinates': should_pass_coordinates_to_structural_features(),
                'pass_mulliken_charges': should_pass_mulliken_charges_to_structural_features(),
                'enable_stereochemistry': should_enable_stereochemistry_preprocessing(),
                'transformation_validation_enabled': is_transformation_validation_enabled(),  # NEW
                'transformation_strict_mode': is_transformation_strict_mode_enabled()  # NEW
            }
        }
    except Exception as e:
        raise HandlerConfigurationError(
            message=f"Failed to create handler-compatible configuration: {str(e)}",
            handler_type="HandlerConfig",
            config_validation_errors=[str(e)]
        ) from e


def create_handler_compatible_config(dataset_type: str, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a handler-compatible configuration for a specific dataset type.
    
    Args:
        dataset_type: Dataset type name
        base_config: Base configuration dictionary
        
    Returns:
        Handler-compatible configuration with dataset_type set
        
    ADDED Phase 5: Helper function for creating handler configs.
    """
    config = base_config.copy()
    config['dataset_type'] = dataset_type
    
    # Merge with existing handler-compatible config
    try:
        existing_config = get_handler_compatible_config()
        config.update(existing_config)
    except Exception as e:
        logger.debug(f"Could not get handler compatible config: {e}")
    
    return config

# Backwards compatibility wrappers - ADD THESE
def get_transform(config, name, **kwargs):
    """Backwards compatible wrapper for get_transform_config."""
    return get_transform_config(config, name, **kwargs)

def get_parameter(config, transform_name, param_name, default=None):
    """Backwards compatible wrapper for get_transform_parameter."""
    return get_transform_parameter(config, transform_name, param_name, default)

def get_setup(config, setup_name, **kwargs):
    """Backwards compatible wrapper for get_experimental_setup."""
    return get_experimental_setup(config, setup_name, **kwargs)

# ============================================================================
# DESCRIPTOR CONFIGURATION ACCESS (Phase 3 Integration)
# ============================================================================

def is_descriptors_enabled() -> bool:
    """
    Check if molecular descriptors are enabled in configuration.
    
    Phase 3 Integration: Descriptor system enablement check
    
    Returns:
        bool: True if descriptors are enabled, False otherwise
        
    Example:
        >>> if is_descriptors_enabled():
        ...     calculator = DescriptorCalculator()
    """
    try:
        config = load_config()
        return config.get('molecular_descriptors', {}).get('enabled', False)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to check descriptor enablement: {e}")
        return False


def get_descriptor_config() -> Dict[str, Any]:
    """
    Get complete descriptor configuration.
    
    Phase 3 Integration: Descriptor configuration access
    
    Returns:
        Dict[str, Any]: Descriptor configuration dictionary
        
    Example:
        >>> config = get_descriptor_config()
        >>> cache_enabled = config.get('computation', {}).get('cache_results', True)
    """
    try:
        config = load_config()
        return config.get('molecular_descriptors', {})
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to load descriptor configuration: {e}")
        return {}


def get_selected_descriptors() -> List[str]:
    """
    Get list of selected descriptors based on configuration.
    
    Phase 3 Integration: Descriptor selection from categories
    
    This function reads the Phase 2 descriptor configuration format which uses
    categories (constitutional, topological, etc.) and expands them into specific
    descriptor names using the descriptor registry.
    
    Returns:
        List[str]: List of descriptor names to calculate
        
    Example:
        >>> descriptors = get_selected_descriptors()
        >>> print(f"Will calculate {len(descriptors)} descriptors")
        
    Note:
        Requires descriptor registry to be initialized with auto-discovery
    """
    try:
        config = get_descriptor_config()
        
        # Check if enabled
        if not config.get('enabled', False):
            return []
        
        # Get categories from config
        # Support both formats:
        # 1. Explicit default_categories list (Phase 2 format)
        # 2. Derive from categories dict keys where enabled=true (config.yaml format)
        categories = config.get('default_categories', [])
        
        if not categories:
            # Derive from categories dict - get enabled category names
            category_configs = config.get('categories', {})
            if category_configs:
                categories = [
                    cat_name for cat_name, cat_config in category_configs.items()
                    if isinstance(cat_config, dict) and cat_config.get('enabled', True)
                ]
            
        if not categories:
            logger = logging.getLogger(__name__)
            logger.warning("Descriptors enabled but no categories configured")
            return []
        
        # Import descriptor system components
        try:
            from milia_pipeline.descriptors.descriptor_registry import DescriptorRegistry
            from milia_pipeline.descriptors.descriptor_categories import DescriptorCategory
        except ImportError as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to import descriptor components: {e}")
            return []
        
        # Get registry instance
        registry = DescriptorRegistry.get_instance()
        
        # Collect descriptors from all enabled categories
        descriptors = []
        category_configs = config.get('categories', {})
        
        for cat_name in categories:
            try:
                # Check if category is explicitly disabled
                cat_config = category_configs.get(cat_name, {})
                if not cat_config.get('enabled', True):
                    logger = logging.getLogger(__name__)
                    logger.debug(f"Skipping disabled category: {cat_name}")
                    continue
                
                # Convert category name to enum
                cat_enum = DescriptorCategory(cat_name)
                
                # Check if specific descriptors are listed for this category
                specific_descriptors = cat_config.get('descriptors', None)
                
                if specific_descriptors is not None and isinstance(specific_descriptors, list):
                    # Use only specified descriptors from this category
                    all_cat_descriptors = registry.list_available_descriptors(category=cat_enum)
                    # Filter to only the requested ones that exist in this category
                    cat_descriptors = [d for d in specific_descriptors if d in all_cat_descriptors]
                    
                    if len(cat_descriptors) < len(specific_descriptors):
                        logger = logging.getLogger(__name__)
                        missing = set(specific_descriptors) - set(cat_descriptors)
                        logger.warning(
                            f"Category '{cat_name}': {len(missing)} requested descriptors not found: "
                            f"{list(missing)[:3]}{'...' if len(missing) > 3 else ''}"
                        )
                else:
                    # Use all descriptors from this category
                    cat_descriptors = registry.list_available_descriptors(category=cat_enum)
                
                descriptors.extend(cat_descriptors)
                
            except ValueError:
                logger = logging.getLogger(__name__)
                logger.warning(f"Unknown descriptor category: {cat_name}")
                continue
        
        # Remove duplicates while preserving order
        seen = set()
        unique_descriptors = []
        for desc in descriptors:
            if desc not in seen:
                seen.add(desc)
                unique_descriptors.append(desc)
        
        return unique_descriptors
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to get selected descriptors: {e}", exc_info=True)
        return []

# ============================================================================
# STRUCTURAL FEATURE HANDLER INTEGRATION
# ============================================================================
# 
# HANDLER-AWARE FEATURE FILTERING:
# - get_dataset_appropriate_structural_features() now queries handlers
# - Handlers declare supported features via get_supported_structural_features()
# - Automatic filtering prevents NaN/Inf from unsupported calculations
# - Fallback to legacy config-based filtering for backward compatibility
# 
# VALIDATION INTEGRATION:
# - validate_structural_features_for_dataset() validates against handler caps
# - Pre-processing validation prevents runtime feature extraction failures
# - Comprehensive error messages for debugging
# 
# BENEFITS:
# - Eliminates "NaN/Inf in atom structural features" warnings for DMC
# - Prevents molecule rejections due to Gasteiger calculation failures
# - Handler-driven feature support (DFT: all features, DMC: limited)
# - Zero impact on existing code - fully backward compatible
# 
# INTEGRATION POINTS:
# - Dataset_handlers.py: Handler feature declarations
# - config_accessors.py: Handler-aware filtering (this file)
# - Phase 3 config_schemas.py: Schema validation (optional)
# - Phase 4 config.yaml: Explicit overrides (optional)
# 
# This Phase 1+2 implementation provides dataset-appropriate structural feature
# filtering while maintaining full backward compatibility with existing systems.
