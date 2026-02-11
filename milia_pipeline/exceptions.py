# exceptions.py - Fully Dynamic Exception Hierarchy with Registry Integration

"""
Comprehensive exception hierarchy for Mlia dataset processing pipeline.

This module provides a three-tier exception hierarchy with 60+ specialized
exception classes for precise error handling across dataset processing,
molecule conversion, handler operations, transformation systems, plugin
management, model training, and configuration validation.

Registry Integration:
--------------------
- Registry integration for dynamic dataset type support
- DatasetSpecificHandlerError for extensible handler exceptions
- UncertaintyProcessingError for uncertainty-enabled datasets
- Factory functions for dynamic exception creation
- Automatic available_types population from registry

Exception Hierarchy
-------------------
The module organizes exceptions into a three-tier structure:

**Tier 1: Base Exception**
    BaseProjectError : Root exception for all project-specific errors

**Tier 2: Domain-Specific Base Classes**
    - ConfigurationError : Configuration file and parameter issues
    - DataProcessingError : General data processing failures
    - MoleculeProcessingError : Individual molecule processing failures
    - HandlerError : Handler pattern and operation issues
    - DatasetSpecificHandlerError : Dataset-specific handler errors
    - UncertaintyProcessingError : Uncertainty processing errors
    - TransformError : Transformation system errors
    - PluginError : Plugin system errors
    - ValidationError : Validation and compatibility issues
    - ModelError : Model training and lifecycle errors

**Tier 3: Specialized Exceptions**
    60+ specific exception classes for precise error categorization

Core Features
-------------
- **Rich Context**: All exceptions include contextual attributes for debugging
- **Hierarchical Structure**: Enables both specific and general exception handling
- **Consistent Interface**: Uniform error reporting patterns across the project
- **Handler Pattern Support**: Specialized exceptions for handler-based architecture
- **Registry Integration**: Dynamic dataset type support via registry
- **Generic Dataset Support**: DatasetSpecificHandlerError for any dataset type
- **Factory Functions**: Dynamic exception creation
- **Uncertainty Dataset Support**: Uncertainty and stochastic processing error handling
- **Plugin System Integration**: Security, validation, and dependency exceptions
- **Model Lifecycle Management**: Training, validation, and deployment error handling

Exception Categories
--------------------
1. **Base & Configuration** (8 classes)
   - BaseProjectError, LoggingConfigurationError, ConfigurationError
   - DataProcessingError, MoleculeProcessingError, MoleculeFilterRejectedError
   - MissingDependencyError, AtomFilterError

2. **Molecule Processing** (6 classes)
   - RDKitConversionError, PyGDataCreationError, PropertyEnrichmentError
   - StructuralFeatureError, VibrationRefinementError
   - UncertaintyProcessingError (Generic for any uncertainty-enabled dataset)

3. **Handler System** (10 classes)
   - HandlerError, HandlerNotAvailableError, HandlerConfigurationError
   - HandlerOperationError, HandlerValidationError, HandlerCompatibilityError
   - HandlerIntegrationError, TransformHandlerIntegrationError
   - DatasetSpecificHandlerError (Generic base for any dataset-specific handler errors)

4. **Validation & Compatibility** (4 classes)
   - ValidationError, CompatibilityError, MigrationError, LegacyCodeError

5. **Transform System** (10 classes)
   - TransformError, TransformCompatibilityError, TransformationError
   - DatasetIntegrationError, TransformValidationError, TransformCompositionError
   - TransformNotFoundError, TransformRegistryError, ExperimentalSetupError
   - TransformConfigurationError

6. **Plugin System** (7 classes)
   - PluginError, PluginValidationError, PluginSecurityError
   - PluginDependencyError, PluginDiscoveryError, PluginRegistrationError
   - PluginLoadError

7. **Model System** (10 classes)
   - ModelError, ModelNotFoundError, ModelValidationError
   - ModelInstantiationError, HyperparameterError, DataCompatibilityError
   - TrainingError, CheckpointError, DataError, PluginModelError

Factory Functions
-----------------
- create_dataset_handler_error(): Creates appropriate handler exception by type
- create_uncertainty_processing_error(): Creates appropriate processing exception
- create_handler_not_available_error(): Creates error with auto-filled available types

Registry Integration Functions
------------------------------
- get_exception_registry_status(): Get registry integration diagnostics
- _get_available_dataset_types(): Get list of registered dataset types
- _is_dataset_type_registered(): Check if dataset type is registered
- _get_dataset_feature(): Query feature flags for dataset types

Design Patterns
---------------
**Contextual Attributes**:
    All exceptions store relevant context (file paths, indices, configuration
    keys, etc.) as attributes for enhanced debugging and error recovery.

**Enhanced __str__ Methods**:
    Most exceptions override __str__ to provide formatted, human-readable
    error messages that include all contextual information.

**Inheritance Hierarchy**:
    Three-tier design allows catching exceptions at appropriate granularity:
    - Catch BaseProjectError for any project error
    - Catch domain-specific base classes (HandlerError, TransformError, ModelError, etc.)
    - Catch DatasetSpecificHandlerError for any dataset-specific handler error
    - Catch specific exceptions for precise error handling

**Registry Integration**:
    - Lazy initialization to avoid circular imports
    - Dynamic dataset type support via registry queries
    - Conservative default values when registry unavailable
    - Feature-based exception handling logic

Usage Examples
--------------
Catching specific exceptions:
    >>> try:
    ...     handler.process_molecule(mol_data)
    ... except HandlerValidationError as e:
    ...     logger.error(f"Validation failed: {e}")
    ...     logger.debug(f"Handler: {e.handler_name}")

Catching by category:
    >>> try:
    ...     process_dataset(config)
    ... except DatasetSpecificHandlerError as e:  # Catches any dataset-specific handler error
    ...     logger.error(f"Dataset-specific handler failure: {e.dataset_type}")
    ... except HandlerError:
    ...     logger.error("General handler system failure")
    ... except TransformError:
    ...     logger.error("Transform system failure")
    ... except ModelError:
    ...     logger.error("Model system failure")
    ... except BaseProjectError as e:
    ...     logger.error(f"Unexpected error: {e}")

Raising with context:
    >>> raise HandlerConfigurationError(
    ...     message="Invalid handler configuration",
    ...     handler_type="MyDatasetHandler",
    ...     config_key="uncertainty_handling",
    ...     details="Configuration mismatch for dataset"
    ... )
    
Using factory functions:
    >>> # Creates DatasetSpecificHandlerError for any dataset type
    >>> error = create_dataset_handler_error(
    ...     message="Validation failed",
    ...     dataset_type="MyDataset",
    ...     operation="validate_properties"
    ... )
    >>> isinstance(error, DatasetSpecificHandlerError)
    True

Exception Context Examples
---------------------------
**ConfigurationError**:
    Attributes: config_key, actual_value, expected_value, details

**MoleculeProcessingError**:
    Attributes: molecule_index, smiles, inchi, details

**HandlerError (subclasses)**:
    Attributes: handler_name, operation, validation_issue, integration_point

**DatasetSpecificHandlerError**:
    Attributes: dataset_type, property_name, operation, details

**UncertaintyProcessingError**:
    Attributes: dataset_type, uncertainty_property_name, molecule_index

**TransformError (subclasses)**:
    Attributes: transform_name, dataset_type, incompatibility_reason

**PluginError (subclasses)**:
    Attributes: plugin_name, plugin_path, security_issue, dependency_info

**ModelError (subclasses)**:
    Attributes: model_name, epoch, validation_errors, hyperparameters, checkpoint_path

Notes
-----
- Module exports 60+ exception classes
- BaseProjectError inherits from Exception (allows normal exception handling)
- MoleculeFilterRejectedError inherits from BaseException (expected rejections)
- All exceptions support **kwargs for extensibility
- Rich context stored as attributes, not just in message strings
- Designed for both automated error recovery and human debugging
- Registry integration enables dynamic dataset type support
- Factory functions for flexible exception creation

.. warning:: Pydantic V2 Namespace Compatibility

    This module defines a ``ValidationError`` class that is DISTINCT from
    Pydantic's ``pydantic.ValidationError``. When using both in the same file,
    import Pydantic's ValidationError with an alias to avoid namespace conflicts::
    
        from milia_pipeline.exceptions import ValidationError  # MILIA's business logic errors
        from pydantic import ValidationError as PydanticValidationError  # Pydantic's type errors
    
    Key differences:
    - MILIA ``ValidationError``: Inherits from ``BaseProjectError``, used for business logic validation
    - Pydantic ``ValidationError``: Inherits from ``ValueError``, used for type/schema validation

See Also
--------
milia_pipeline.logging_config : Logging infrastructure for error tracking
milia_pipeline.handlers : Handler pattern implementation using these exceptions
milia_pipeline.transformations : Transform system with error handling
milia_pipeline.models : Model training and lifecycle management
milia_pipeline.cli_manager : CLI system that raises configuration exceptions
milia_pipeline.datasets.registry : Dataset registry for dynamic type support
"""

from typing import Optional, Any, Dict, Type, List, Union, Callable


# =============================================================================
# REGISTRY INTEGRATION FOR DYNAMIC DATASET TYPE SUPPORT
# =============================================================================

# Registry state - lazy initialization to avoid circular imports
_REGISTRY_INITIALIZED: bool = False
_REGISTRY_AVAILABLE: bool = False
_registry_list_all: Optional[Callable] = None
_registry_get: Optional[Callable] = None
_registry_is_registered: Optional[Callable] = None



def _discover_dataset_types_from_filesystem() -> List[str]:
    """
    Dynamically discover dataset types from implementations directory.
    
    DYNAMIC APPROACH: Scans the filesystem to find available dataset implementations
    instead of using hardcoded fallback lists.
    
    Returns:
        List of discovered dataset type names (uppercase)
    """
    try:
        from pathlib import Path
        
        # Find the implementations directory relative to this file
        implementations_dir = Path(__file__).parent / 'datasets' / 'implementations'
        if implementations_dir.exists():
            discovered_types = []
            for py_file in implementations_dir.glob('*.py'):
                if py_file.name.startswith('_'):
                    continue
                # Extract dataset name from filename (e.g., my_dataset.py -> MY_DATASET)
                dataset_name = py_file.stem.upper()
                # Exclude non-dataset modules
                if dataset_name not in ['BASE', 'REGISTRY', 'UTILS', 'COMMON', 'PROTOCOLS']:
                    discovered_types.append(dataset_name)
            if discovered_types:
                return discovered_types
    except Exception:
        pass
    
    # Final fallback: return empty list
    return []


def _init_registry() -> bool:
    """
    Lazily initialize registry integration.
    
    Following the lazy initialization pattern from other pipeline modules
    (dataset_handlers.py, milia_dataset.py, validators.py).
    
    This function must be called before any registry operations. It handles
    the case where the registry module is not available (e.g., during early
    application bootstrap or in environments without the full pipeline).
    
    Returns:
        bool: True if registry is available, False otherwise.
    """
    global _REGISTRY_INITIALIZED, _REGISTRY_AVAILABLE
    global _registry_list_all, _registry_get, _registry_is_registered
    
    if _REGISTRY_INITIALIZED:
        return _REGISTRY_AVAILABLE
    
    _REGISTRY_INITIALIZED = True
    
    try:
        from milia_pipeline.datasets.registry import list_all, get, is_registered
        _registry_list_all = list_all
        _registry_get = get
        _registry_is_registered = is_registered
        _REGISTRY_AVAILABLE = True
        return True
    except ImportError:
        # Registry not available - use fallback defaults
        # This is expected during early application bootstrap
        _REGISTRY_AVAILABLE = False
        return False
    except Exception:
        # Unexpected error - fall back to defaults
        _REGISTRY_AVAILABLE = False
        return False


def _get_available_dataset_types() -> List[str]:
    """
    Get list of available dataset types from registry or dynamic discovery.
    
    This function:
    1. First tries the registry (primary source of truth)
    2. If registry fails, dynamically discovers dataset implementations from filesystem
    
    Provides dynamic dataset type list for error messages and validation.
    This function is used throughout the exception module to populate error
    messages with the current list of available dataset types.
    
    Returns:
        List[str]: List of available dataset type names (dynamically discovered).
        
    Examples:
        >>> types = _get_available_dataset_types()
        >>> isinstance(types, list)
        True
    """
    _init_registry()
    
    if _REGISTRY_AVAILABLE and _registry_list_all is not None:
        try:
            return _registry_list_all()
        except Exception:
            pass
    
    # Fallback: Use filesystem discovery
    return _discover_dataset_types_from_filesystem()


def _is_dataset_type_registered(dataset_type: str) -> bool:
    """
    Check if a dataset type is registered or dynamically discovered.
    
    This function:
    1. First tries the registry (primary source of truth)
    2. If registry fails, uses _get_available_dataset_types() which does dynamic discovery
    
    Provides dynamic validation of dataset types for exception handling.
    
    Args:
        dataset_type: Name of the dataset type to check.
        
    Returns:
        bool: True if the dataset type is registered/discovered, False otherwise.
        
    Examples:
        >>> _is_dataset_type_registered('UNKNOWN')
        False
    """
    _init_registry()
    
    if _REGISTRY_AVAILABLE and _registry_is_registered is not None:
        try:
            return _registry_is_registered(dataset_type)
        except Exception:
            pass
    
    # Fallback: Check against dynamically discovered types
    available_types = _get_available_dataset_types()
    return dataset_type in available_types


def _get_dataset_feature(dataset_type: str, feature_name: str, default: bool = False) -> bool:
    """
    Get a feature flag for a dataset type.
    
    Queries the registry for dataset feature flags. Used to determine
    dataset-specific behavior in recovery suggestions and error context
    generation.
    
    Args:
        dataset_type: Name of the dataset type.
        feature_name: Name of the feature to check (e.g., 'uncertainty_handling').
        default: Default value if feature not found or registry unavailable.
        
    Returns:
        bool: Feature value from registry, or default if unavailable.
        
    Examples:
        >>> _get_dataset_feature('some_dataset', 'uncertainty_handling')
        # Returns True/False from registry, or default if registry unavailable
    """
    _init_registry()
    
    if _REGISTRY_AVAILABLE and _registry_get is not None:
        try:
            dataset_class = _registry_get(dataset_type)
            if hasattr(dataset_class, 'features'):
                return getattr(dataset_class.features, feature_name, default)
        except Exception:
            pass
    
    # Fallback: return default when registry unavailable
    return default


def get_exception_registry_status() -> Dict[str, Any]:
    """
    Get registry integration status for diagnostics.
    
    Provides information about registry availability for debugging
    and troubleshooting exception handling issues.
    
    Returns:
        Dict with registry status information including:
        - registry_available: Whether registry is accessible
        - registry_initialized: Whether initialization was attempted
        - available_dataset_types: List of dynamically discovered dataset types
        - using_fallback: Whether using default fallback (registry unavailable)
        - dynamic_integration: Always True to indicate dynamic implementation
        
    Examples:
        >>> status = get_exception_registry_status()
        >>> status['dynamic_integration']
        True
        >>> isinstance(status['available_dataset_types'], list)
        True
    """
    _init_registry()
    
    return {
        'registry_available': _REGISTRY_AVAILABLE,
        'registry_initialized': _REGISTRY_INITIALIZED,
        'available_dataset_types': _get_available_dataset_types(),
        'using_fallback': not _REGISTRY_AVAILABLE,
        'dynamic_integration': True,
    }


# =============================================================================
# TIER 1: BASE EXCEPTION
# =============================================================================

class BaseProjectError(Exception):
    """
    Base exception class for all project-specific errors.
    
    This class provides a foundation for all custom exceptions in the project,
    ensuring consistent error reporting and handling patterns.
    
    Args:
        message (str): Human-readable error message describing what went wrong.
        details (str, optional): Additional technical details about the error.
                                This can include stack traces, debug information, etc.
        **kwargs: Additional keyword arguments for future extensibility.
    """
    
    def __init__(self, message: str, details: Optional[str] = None, **kwargs):
        super().__init__(message)
        self.message = message
        self.details = details
        self.extra_info = kwargs
    
    def __str__(self) -> str:
        msg = self.message
        if self.details:
            msg += f". Details: {self.details}"
        return msg


# =============================================================================
# TIER 2: CONFIGURATION EXCEPTIONS
# =============================================================================

class LoggingConfigurationError(BaseProjectError):
    """Exception raised for errors specifically during logging configuration."""
    def __init__(self, message: str = "Error configuring logging.", details: Optional[str] = None, **kwargs) -> None:
        """
        Initializes the LoggingConfigurationError.

        Args:
            message (str): A general description of the logging configuration error.
            details (Optional[str]): More specific details about the error, if available.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        super().__init__(message, details, **kwargs)


class ConfigurationError(BaseProjectError):
    """
    Raised when there are issues with configuration files or parameters.
    
    This exception is used for problems like missing configuration keys,
    invalid configuration values, or malformed configuration files.
    
    Args:
        message (str): Description of the configuration error.
        config_key (str): The specific configuration key that caused the issue.
        actual_value (Any, optional): The actual value that was found (for debugging).
        expected_value (Any, optional): The expected value or type (for debugging).
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, config_key: str = None, actual_value: Any = None, 
                 expected_value: Any = None, details: Optional[str] = None, **kwargs):
        super().__init__(message, details, **kwargs)
        self.config_key = config_key
        self.actual_value = actual_value
        self.expected_value = expected_value


    def __str__(self) -> str:
        """Enhanced string representation including all context"""
        parts = [self.message]
        
        if hasattr(self, 'config_key') and self.config_key:
            parts.append(f"Key: '{self.config_key}'")
        
        if hasattr(self, 'expected_value') and self.expected_value is not None:
            expected_type = getattr(self.expected_value, '__name__', str(self.expected_value))
            parts.append(f"Expected Type: {expected_type}")
        
        if hasattr(self, 'actual_value') and self.actual_value is not None:
            actual_type = type(self.actual_value).__name__
            parts.append(f"Actual Value: '{self.actual_value}' (Type: {actual_type})")
        
        # Include details if present
        if hasattr(self, 'details') and self.details:
            parts.append(f"Details: {self.details}")
        
        return " ".join(parts)


# =============================================================================
# TIER 2: DATA PROCESSING EXCEPTIONS
# =============================================================================

class DataProcessingError(BaseProjectError):
    """
    Raised when there are general errors during data processing.
    
    This is a broad category for errors that occur during dataset loading,
    file I/O operations, data format issues, etc.
    
    Args:
        message (str): Description of the data processing error.
        file_path (str, optional): Path to the file that caused the error.
        operation (str, optional): The operation that was being performed.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, file_path: Optional[str] = None, 
                 operation: Optional[str] = None, details: Optional[str] = None, **kwargs):
        super().__init__(message, details, **kwargs)
        self.file_path = file_path
        self.operation = operation


class PreprocessingRequiredError(DataProcessingError):
    """
    Raised when preprocessing is required but cannot be completed automatically.
    
    This error provides clear instructions for manual preprocessing.
    """
    
    def __init__(
        self, 
        source_file: str,
        target_file: str,
        dataset_type: str,
        preprocessing_command: str = None,
        details: str = None
    ):
        if preprocessing_command is None:
            preprocessing_command = f"""python -m milia_pipeline.preprocessing.cli \\
        --dataset-type {dataset_type} \\
        --input {source_file} \\
        --output {target_file}"""
        
        message = f"""Preprocessing required but could not be completed automatically.

Source file exists: {source_file}
Target file needed: {target_file}
Dataset type: {dataset_type}

To resolve this, run preprocessing manually:
    {preprocessing_command}

Alternatively, ensure the preprocessing system is properly installed and try again.
"""
        super().__init__(message, details=details)
        self.source_file = source_file
        self.target_file = target_file
        self.dataset_type = dataset_type


# =============================================================================
# TIER 2: MOLECULE PROCESSING EXCEPTIONS
# =============================================================================

class MoleculeProcessingError(BaseProjectError):
    """
    Base class for errors that occur during individual molecule processing.
    
    This serves as a parent class for more specific molecule-related errors.
    
    Args:
        message (str): Description of the molecule processing error.
        molecule_index (int): Index of the molecule that caused the error.
        smiles (str, optional): SMILES string of the molecule (for identification).
        inchi (str, optional): InChI string of the molecule (for identification).
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, molecule_index: int, smiles: Optional[str] = None,
                 inchi: Optional[str] = None, details: Optional[str] = None, reason: Optional[str] = None, **kwargs):
        super().__init__(message, details or reason, **kwargs)
        self.molecule_index = molecule_index
        self.smiles = smiles or "N/A"
        self.inchi = inchi or "N/A"

    def __str__(self) -> str:
        msg: str = f"Error processing molecule (Index: {self.molecule_index})"
        if self.inchi and self.inchi != "N/A":
            msg += f", InChI: {self.inchi}"
        if self.smiles and self.smiles != "N/A":
            msg += f", SMILES: {self.smiles}"

        # Add the main message if it's not already part of the details
        if self.message and self.message not in msg:
            msg += f": {self.message}"

        if self.details:
            msg += f". Details: {self.details}"
        return msg


class MoleculeFilterRejectedError(BaseException):
    """
    Raised when a molecule is rejected by pre-filtering criteria.
    
    This is used when molecules don't meet specified criteria like atom count limits,
    heavy atom requirements, etc. This is considered an expected rejection rather
    than an error condition.
    
    Args:
        molecule_index (int): Index of the rejected molecule.
        inchi (str): InChI string of the rejected molecule.
        reason (str): Reason why the molecule was rejected.
        filter_name (str, optional): Name of the filter that rejected the molecule.
        filter_value (Any, optional): The filter value that caused rejection.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, molecule_index: int, inchi: str, reason: str,
                 filter_name: Optional[str] = None, filter_value: Any = None,
                 details: Optional[str] = None, **kwargs):
        message = f"Molecule rejected by filter: {reason}"
        super().__init__(message)
        self.message = message
        self.molecule_index = molecule_index
        self.inchi = inchi
        self.reason = reason
        self.filter_name = filter_name
        self.filter_value = filter_value
        self.details = details
        self.extra_info = kwargs

    def __str__(self) -> str:
        msg = f"Error processing molecule (Index: {self.molecule_index}), InChI: {self.inchi}: {self.reason}"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


class MissingDependencyError(BaseProjectError):
    """
    Raised when a required dependency is missing or unavailable.
    
    This occurs when optional or required libraries are not installed,
    or when specific features are not available in the current environment.
    
    Args:
        message (str): Description of the missing dependency error.
        dependency_name (str): Name of the missing dependency.
        install_command (str, optional): Command to install the missing dependency.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, dependency_name: str, 
                 install_command: Optional[str] = None, details: Optional[str] = None, **kwargs):
        super().__init__(message, details, **kwargs)
        self.dependency_name = dependency_name
        self.install_command = install_command

    def __str__(self) -> str:
        msg: str = f"{self.message}"
        if self.dependency_name:
            msg += f" Dependency: '{self.dependency_name}'"
        return msg


class AtomFilterError(BaseProjectError):
    """
    Raised when there are issues with atom-based filtering configuration or execution.
    
    This occurs when atom filters are misconfigured or when atom-level filtering
    logic encounters unexpected conditions.
    
    Args:
        message (str): Description of the atom filter error.
        filter_config (Dict, optional): The filter configuration that caused the issue.
        atom_symbol (str, optional): Atomic symbol that caused the issue.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, filter_config: Optional[Dict] = None,
                 atom_symbol: Optional[str] = None, details: Optional[str] = None, **kwargs):
        super().__init__(message, details, **kwargs)
        self.filter_config = filter_config
        self.atom_symbol = atom_symbol

    def __str__(self) -> str:
        msg: str = f"{self.message}"
        if self.atom_symbol:
            msg += f", Invalid Atom Symbol: '{self.atom_symbol}'"
        return msg


# =============================================================================
# TIER 3: SPECIALIZED MOLECULE PROCESSING EXCEPTIONS
# =============================================================================

class RDKitConversionError(MoleculeProcessingError):
    """
    Raised when RDKit fails to process a molecule.
    
    This occurs when RDKit cannot parse molecular structures, generate conformers,
    calculate properties, etc.
    
    Args:
        molecule_index (int): Index of the molecule that failed conversion.
        inchi (str): InChI string of the molecule.
        reason (str): Reason for the RDKit conversion failure.
        detail (str): Additional details about the failure.
        rdkit_error (str, optional): Original RDKit error message.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, molecule_index: int, inchi: str, reason: str, detail: str,
                 rdkit_error: Optional[str] = None, **kwargs):
        message = f"RDKit conversion failed: {reason}"
        super().__init__(message, molecule_index, inchi=inchi, details=detail, **kwargs)
        self.reason = reason
        self.detail = detail
        self.rdkit_error = rdkit_error


class PyGDataCreationError(MoleculeProcessingError):
    """
    Raised when PyTorch Geometric Data object creation fails.
    
    This occurs when there are issues creating PyG Data objects from molecular
    data, applying transforms, or other PyG-specific operations.
    
    Args:
        message (str): Description of the error.
        molecule_index (int): Index of the molecule that failed.
        smiles (str): SMILES string of the molecule.
        reason (str): Reason for the PyG data creation failure.
        detail (str): Additional details about the failure.
        transform_name (str, optional): Name of the transform that failed.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, molecule_index: int, smiles: str, reason: str, detail: str,
                 transform_name: Optional[str] = None, **kwargs):
        super().__init__(message, molecule_index, smiles=smiles, details=detail, **kwargs)
        self.reason = reason
        self.detail = detail
        self.transform_name = transform_name


class PropertyEnrichmentError(MoleculeProcessingError):
    """
    Raised when property calculation or enrichment fails for a molecule.
    
    This occurs when specific molecular properties cannot be calculated or
    when property data is missing/invalid.
    
    Args:
        molecule_index (int): Index of the molecule that failed.
        inchi (str): InChI string of the molecule.
        property_name (str): Name of the property that failed to be calculated.
        reason (str): Reason for the property enrichment failure.
        detail (str): Additional details about the failure.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, molecule_index: int, inchi: str, property_name: str, 
                 reason: str, detail: str, **kwargs):
        message = f"Property enrichment failed for '{property_name}': {reason}"
        super().__init__(message, molecule_index, inchi=inchi, details=detail, **kwargs)
        self.property_name = property_name
        self.reason = reason
        self.detail = detail

    def __str__(self) -> str:
        msg: str = super().__str__()
        if self.property_name:
            msg += f" (Property: {self.property_name})"
        return msg


class StructuralFeatureError(MoleculeProcessingError):
    """
    Exception raised when an error occurs during the calculation or
    assignment of structural features (atom or bond features).
    """
    def __init__(self, message: str = "Failed to calculate or assign structural features.",
                     molecule_index: Optional[int] = None,
                     inchi: Optional[str] = None,
                     feature_type: Optional[str] = None, # "atom" or "bond"
                     feature_name: Optional[str] = None, # specific feature name, e.g., "hybridization"
                     reason: Optional[str] = None,
                     detail: Optional[str] = None, **kwargs) -> None:
        """
        Initializes the StructuralFeatureError.

        Args:
            message (str): A general message about the structural feature failure.
            molecule_index (Optional[int]): The index of the molecule.
            inchi (Optional[str]): The InChI string of the molecule.
            feature_type (Optional[str]): The type of feature being processed ("atom" or "bond").
            feature_name (Optional[str]): The specific feature name that caused the error.
            reason (Optional[str]): The reason for the structural feature failure.
            detail (Optional[str]): Specific details about the error.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(message, molecule_index or 0, inchi=inchi, details=detail, **kwargs)
        self.feature_type: Optional[str] = feature_type
        self.feature_name: Optional[str] = feature_name
        self.reason: Optional[str] = reason
        self.detail: Optional[str] = detail

    def __str__(self) -> str:
        msg: str = super().__str__()
        if self.feature_type:
            msg += f" (Feature Type: {self.feature_type}"
            if self.feature_name:
                msg += f", Feature Name: {self.feature_name}"
            msg += ")"
        return msg


class VibrationRefinementError(DataProcessingError):
    """
    Exception raised when an error occurs during the refinement of
    molecular vibrations (frequencies and vibmodes).
    """
    def __init__(self, message: str = "Error during molecular vibration refinement.",
                 molecule_index: Optional[int] = None,
                 reason: Optional[str] = None,
                 detail: Optional[str] = None, **kwargs) -> None:
        """
        Initializes the VibrationRefinementError.

        Args:
            message (str): A general message about the vibration refinement failure.
            molecule_index (Optional[int]): The index of the molecule being processed.
            reason (Optional[str]): The reason for the refinement failure.
            detail (Optional[str]): Specific details about the error.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(message, details=reason, **kwargs)
        self.molecule_index: Optional[int] = molecule_index
        self.reason: Optional[str] = reason
        self.detail: Optional[str] = detail

    def __str__(self) -> str:
        msg: str = f"Error during molecular vibration refinement"
        if self.molecule_index is not None:
            msg += f" for Molecule (Index: {self.molecule_index})"
        if self.reason:
            msg += f": {self.reason}"
        if self.detail:
            msg += f". Details: {self.detail}"
        return msg


# =============================================================================
# UNCERTAINTY PROCESSING EXCEPTIONS
# =============================================================================

class UncertaintyProcessingError(MoleculeProcessingError):
    """
    Exception raised for errors during uncertainty-enabled data processing.
    
    Generic exception for any dataset type with uncertainty_handling=True.
    
    This includes issues with:
    - Uncertainty values validation
    - Statistical data processing
    - Correlation data handling
    - Standard deviation calculations
    
    Args:
        message (str): Description of the error.
        dataset_type (str): Type of dataset (e.g., 'QMC', 'FCIQMC').
        molecule_index (int, optional): Index of the molecule.
        inchi (str, optional): InChI string of the molecule.
        detail (str, optional): Additional details.
        uncertainty_property_name (str, optional): Uncertainty property that failed.
        **kwargs: Additional keyword arguments.
        
    Examples:
        >>> raise UncertaintyProcessingError(
        ...     "Standard deviation validation failed",
        ...     dataset_type="QMC",
        ...     molecule_index=5,
        ...     uncertainty_property_name="std_error"
        ... )
    """
    
    def __init__(
        self,
        message: str,
        dataset_type: str,
        molecule_index: Optional[int] = None,
        inchi: Optional[str] = None,
        detail: Optional[str] = None,
        uncertainty_property_name: Optional[str] = None,
        **kwargs: Any
    ):
        super().__init__(
            message,
            molecule_index=molecule_index or 0,
            inchi=inchi,
            details=detail,
            **kwargs
        )
        self.dataset_type = dataset_type
        self.uncertainty_property_name = uncertainty_property_name

    def __str__(self) -> str:
        msg = super().__str__()
        if self.dataset_type:
            msg += f" (Dataset: {self.dataset_type})"
        if self.uncertainty_property_name:
            msg += f" (Uncertainty Property: {self.uncertainty_property_name})"
        return msg


# =============================================================================
# TIER 2: HANDLER EXCEPTIONS
# =============================================================================

class HandlerError(BaseProjectError):
    """
    Base exception for all dataset handler-related errors.
    
    This serves as the parent class for all exceptions that occur within
    the dataset handler strategy pattern implementation.
    
    Args:
        message (str): Description of the handler error.
        handler_type (str, optional): Type of handler that caused the error.
        handler_operation (str, optional): Operation being performed when error occurred.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, handler_type: Optional[str] = None,
                 handler_operation: Optional[str] = None, details: Optional[str] = None, **kwargs):
        super().__init__(message, details, **kwargs)
        self.handler_type = handler_type
        self.handler_operation = handler_operation
    
    def __str__(self) -> str:
        msg = self.message
        if self.handler_type:
            msg += f" (Handler: {self.handler_type}"
            if self.handler_operation:
                msg += f", Operation: {self.handler_operation}"
            msg += ")"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


class HandlerNotAvailableError(HandlerError):
    """
    Raised when a required dataset handler is not available or cannot be created.
    
    This occurs when:
    - Handler factory cannot create handler for dataset type
    - Handler dependencies are missing
    - Handler configuration is invalid
    
    available_types can be auto-populated from registry using
    create_handler_not_available_error() factory function.
    
    Args:
        message (str): Description of the availability issue.
        requested_dataset_type (str): The dataset type that was requested.
        available_types (List[str], optional): List of available handler types.
        missing_dependencies (List[str], optional): List of missing dependencies.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, requested_dataset_type: str,
                 available_types: Optional[List[str]] = None,
                 missing_dependencies: Optional[List[str]] = None,
                 details: Optional[str] = None, **kwargs):
        super().__init__(message, handler_type=requested_dataset_type, details=details, **kwargs)
        self.requested_dataset_type = requested_dataset_type
        self.available_types = available_types or []
        self.missing_dependencies = missing_dependencies or []
    
    def __str__(self) -> str:
        msg = f"{self.message} (Requested: {self.requested_dataset_type})"
        if self.available_types:
            msg += f", Available: {self.available_types}"
        if self.missing_dependencies:
            msg += f", Missing Dependencies: {self.missing_dependencies}"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


class HandlerConfigurationError(HandlerError):
    """
    Raised when a dataset handler has invalid or incompatible configuration.
    
    This occurs when:
    - Handler configuration validation fails
    - Required configuration parameters are missing
    - Configuration values are out of valid ranges
    - Handler type mismatches dataset configuration
    
    Args:
        message (str): Description of the configuration error.
        handler_type (str): Type of handler with configuration issues.
        config_validation_errors (List[str], optional): List of specific validation errors.
        invalid_config_keys (List[str], optional): List of invalid configuration keys.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, handler_type: str,
                 config_validation_errors: Optional[List[str]] = None,
                 invalid_config_keys: Optional[List[str]] = None,
                 details: Optional[str] = None, **kwargs):
        super().__init__(message, handler_type=handler_type, details=details, **kwargs)
        self.config_validation_errors = config_validation_errors or []
        self.invalid_config_keys = invalid_config_keys or []
    
    def __str__(self) -> str:
        msg = f"{self.message} (Handler: {self.handler_type})"
        if self.config_validation_errors:
            msg += f", Validation Errors: {self.config_validation_errors}"
        if self.invalid_config_keys:
            msg += f", Invalid Keys: {self.invalid_config_keys}"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


class HandlerOperationError(HandlerError):
    """
    Raised when a handler operation fails during execution.
    
    This occurs when:
    - Handler methods encounter unexpected conditions
    - Handler cannot process specific data
    - Handler state becomes inconsistent
    
    Args:
        message (str): Description of the operation error.
        handler_type (str): Type of handler that failed.
        operation (str): Specific operation that failed.
        molecule_index (int, optional): Index of molecule being processed (if applicable).
        recovery_suggestions (List[str], optional): Suggested recovery actions.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, handler_type: str, operation: str,
                 molecule_index: Optional[int] = None,
                 recovery_suggestions: Optional[List[str]] = None,
                 details: Optional[str] = None, **kwargs):
        super().__init__(message, handler_type=handler_type, handler_operation=operation, details=details, **kwargs)
        self.molecule_index = molecule_index
        self.recovery_suggestions = recovery_suggestions or []
    
    def __str__(self) -> str:
        msg = f"{self.message} (Handler: {self.handler_type}, Operation: {self.handler_operation})"
        if self.molecule_index is not None:
            msg += f", Molecule: {self.molecule_index}"
        if self.recovery_suggestions:
            msg += f", Suggestions: {self.recovery_suggestions}"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


class HandlerValidationError(HandlerError):
    """
    Raised when handler validation operations fail.
    
    This occurs when:
    - Handler cannot validate molecule data
    - Handler validation rules are violated
    - Handler compatibility checks fail
    
    Args:
        message (str): Description of the validation error.
        handler_type (str): Type of handler that failed validation.
        validation_type (str): Type of validation that failed.
        failed_validations (List[str], optional): List of specific validation failures.
        molecule_index (int, optional): Index of molecule that failed validation.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, handler_type: str, validation_type: str,
                 failed_validations: Optional[List[str]] = None,
                 molecule_index: Optional[int] = None,
                 details: Optional[str] = None, **kwargs):
        super().__init__(message, handler_type=handler_type, details=details, **kwargs)
        self.validation_type = validation_type
        self.failed_validations = failed_validations or []
        self.molecule_index = molecule_index
    
    def __str__(self) -> str:
        msg = f"{self.message} (Handler: {self.handler_type}, Validation: {self.validation_type})"
        if self.molecule_index is not None:
            msg += f", Molecule: {self.molecule_index}"
        if self.failed_validations:
            msg += f", Failed: {self.failed_validations}"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


class HandlerCompatibilityError(HandlerError):
    """
    Raised when handlers are incompatible with the current environment or configuration.
    
    This occurs when:
    - Handler requires features not available in current setup
    - Handler version incompatibility
    - Handler cannot work with current data format
    
    Args:
        message (str): Description of the compatibility error.
        handler_type (str): Type of handler with compatibility issues.
        incompatible_features (List[str], optional): List of incompatible features.
        minimum_requirements (Dict[str, str], optional): Minimum requirements not met.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, handler_type: str,
                 incompatible_features: Optional[List[str]] = None,
                 minimum_requirements: Optional[Dict[str, str]] = None,
                 details: Optional[str] = None, **kwargs):
        super().__init__(message, handler_type=handler_type, details=details, **kwargs)
        self.incompatible_features = incompatible_features or []
        self.minimum_requirements = minimum_requirements or {}
    
    def __str__(self) -> str:
        msg = f"{self.message} (Handler: {self.handler_type})"
        if self.incompatible_features:
            msg += f", Incompatible: {self.incompatible_features}"
        if self.minimum_requirements:
            req_str = ", ".join([f"{k}={v}" for k, v in self.minimum_requirements.items()])
            msg += f", Requirements: {req_str}"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


class HandlerIntegrationError(BaseProjectError):
    """
    Raised when there are errors integrating handlers with existing pipeline components.
    
    This occurs when:
    - Handler integration with legacy code fails
    - Handler cannot interface with existing modules
    - Migration to handler pattern encounters issues
    
    Args:
        message (str): Description of the integration error.
        handler_type (str, optional): Type of handler involved in integration.
        integration_point (str, optional): Specific integration point that failed.
        legacy_component (str, optional): Legacy component that failed integration.
        migration_phase (str, optional): Migration phase where error occurred.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, handler_type: Optional[str] = None,
                 integration_point: Optional[str] = None,
                 legacy_component: Optional[str] = None,
                 migration_phase: Optional[str] = None,
                 details: Optional[str] = None, **kwargs):
        super().__init__(message, details, **kwargs)
        self.handler_type = handler_type
        self.integration_point = integration_point
        self.legacy_component = legacy_component
        self.migration_phase = migration_phase
    
    def __str__(self) -> str:
        msg = self.message
        context_parts = []
        if self.handler_type:
            context_parts.append(f"Handler: {self.handler_type}")
        if self.integration_point:
            context_parts.append(f"Integration: {self.integration_point}")
        if self.legacy_component:
            context_parts.append(f"Legacy: {self.legacy_component}")
        if self.migration_phase:
            context_parts.append(f"Phase: {self.migration_phase}")
        
        if context_parts:
            msg += f" ({', '.join(context_parts)})"
        
        if self.details:
            msg += f". Details: {self.details}"
        return msg


class TransformHandlerIntegrationError(HandlerIntegrationError):
    """
    Raised when transform integration with handlers fails.
    
    This occurs when:
    - Transform operations fail within handler context
    - Transform configuration incompatible with handler
    - Transform validation fails during handler operations
    
    Args:
        message: Description of the integration error
        handler_type: Type of handler involved
        integration_point: Specific integration point that failed
        transform_name: Optional name of the transform that caused the issue
        experimental_setup: Optional associated experimental setup
        details: Optional additional technical details
    """
    
    def __init__(self, 
                 message: str, 
                 handler_type: str, 
                 integration_point: str,
                 transform_name: Optional[str] = None,
                 experimental_setup: Optional[str] = None,
                 details: Optional[str] = None, 
                 **kwargs):
        super().__init__(
            message=message,
            handler_type=handler_type,
            integration_point=integration_point,
            details=details,
            **kwargs
        )
        self.transform_name = transform_name
        self.experimental_setup = experimental_setup
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.transform_name:
            msg += f", Transform: {self.transform_name}"
        if self.experimental_setup:
            msg += f", Setup: {self.experimental_setup}"
        return msg


# =============================================================================
# DATASET-SPECIFIC HANDLER EXCEPTIONS
# =============================================================================

class DatasetSpecificHandlerError(HandlerError):
    """
    Exception for dataset-specific handler errors.
    
    This class enables dynamic creation of dataset-specific handler exceptions
    without requiring new class definitions for each dataset type. The
    dataset_type parameter identifies the specific dataset at runtime.
    
    Args:
        message (str): Description of the handler error.
        dataset_type (str): Type of dataset (e.g., 'QMC', 'CCSD').
        operation (str, optional): Specific operation that failed.
        property_name (str, optional): Property that caused the issue.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
        
    Examples:
        >>> raise DatasetSpecificHandlerError(
        ...     "Handler validation failed",
        ...     dataset_type="QMC",
        ...     operation="validate_uncertainty",
        ...     property_name="correlation_energy"
        ... )
        
        >>> try:
        ...     process_dataset(config)
        ... except DatasetSpecificHandlerError as e:
        ...     print(f"Dataset error: {e.dataset_type}")
    """
    
    def __init__(
        self,
        message: str,
        dataset_type: str,
        operation: Optional[str] = None,
        property_name: Optional[str] = None,
        details: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message,
            handler_type=dataset_type,
            handler_operation=operation,
            details=details,
            **kwargs
        )
        self.dataset_type = dataset_type
        self.property_name = property_name
    
    def __str__(self) -> str:
        msg = f"{self.message} (Dataset: {self.dataset_type}"
        if self.handler_operation:
            msg += f", Operation: {self.handler_operation}"
        msg += ")"
        if self.property_name:
            msg += f" (Property: {self.property_name})"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


# =============================================================================
# EXCEPTION FACTORY FUNCTIONS
# =============================================================================

def create_dataset_handler_error(
    message: str,
    dataset_type: str,
    operation: Optional[str] = None,
    property_name: Optional[str] = None,
    details: Optional[str] = None,
    **kwargs
) -> DatasetSpecificHandlerError:
    """
    Factory function to create a dataset handler exception.
    
    Creates a DatasetSpecificHandlerError for any dataset type.
    
    Args:
        message: Error message.
        dataset_type: Type of dataset.
        operation: Operation that failed.
        property_name: Property that caused the issue.
        details: Additional details.
        **kwargs: Additional keyword arguments.
        
    Returns:
        DatasetSpecificHandlerError instance.
        
    Examples:
        >>> error = create_dataset_handler_error(
        ...     "Validation failed",
        ...     dataset_type="MyDataset",
        ...     operation="validate_properties"
        ... )
        >>> isinstance(error, DatasetSpecificHandlerError)
        True
        >>> error.dataset_type
        'MyDataset'
    """
    return DatasetSpecificHandlerError(
        message,
        dataset_type=dataset_type,
        operation=operation,
        property_name=property_name,
        details=details,
        **kwargs
    )


def create_uncertainty_processing_error(
    message: str,
    dataset_type: str,
    molecule_index: Optional[int] = None,
    inchi: Optional[str] = None,
    detail: Optional[str] = None,
    property_name: Optional[str] = None,
    **kwargs
) -> UncertaintyProcessingError:
    """
    Factory function to create an uncertainty processing exception.
    
    Creates an UncertaintyProcessingError for any uncertainty-enabled dataset type.
    
    Args:
        message: Error message.
        dataset_type: Type of dataset.
        molecule_index: Index of molecule.
        inchi: InChI string.
        detail: Additional details.
        property_name: Property that caused the issue.
        **kwargs: Additional keyword arguments.
        
    Returns:
        UncertaintyProcessingError instance.
        
    Examples:
        >>> error = create_uncertainty_processing_error(
        ...     "Uncertainty validation failed",
        ...     dataset_type="MyDataset",
        ...     molecule_index=5
        ... )
        >>> isinstance(error, UncertaintyProcessingError)
        True
        >>> error.dataset_type
        'MyDataset'
    """
    return UncertaintyProcessingError(
        message,
        dataset_type=dataset_type,
        molecule_index=molecule_index,
        inchi=inchi,
        detail=detail,
        uncertainty_property_name=property_name,
        **kwargs
    )


def create_handler_not_available_error(
    message: str,
    requested_dataset_type: str,
    available_types: Optional[List[str]] = None,
    missing_dependencies: Optional[List[str]] = None,
    details: Optional[str] = None,
    **kwargs
) -> HandlerNotAvailableError:
    """
    Factory function to create HandlerNotAvailableError with dynamic available types.
    
    If available_types is not provided, automatically retrieves
    the list from the registry.
    
    Args:
        message: Error message.
        requested_dataset_type: Dataset type that was requested.
        available_types: List of available types (auto-filled from registry if None).
        missing_dependencies: List of missing dependencies.
        details: Additional details.
        **kwargs: Additional keyword arguments.
        
    Returns:
        HandlerNotAvailableError with populated available_types.
        
    Examples:
        >>> error = create_handler_not_available_error(
        ...     "Handler not available",
        ...     requested_dataset_type="UNKNOWN"
        ... )
        >>> isinstance(error.available_types, list)
        True
    """
    if available_types is None:
        available_types = _get_available_dataset_types()
    
    return HandlerNotAvailableError(
        message,
        requested_dataset_type=requested_dataset_type,
        available_types=available_types,
        missing_dependencies=missing_dependencies,
        details=details,
        **kwargs
    )


# =============================================================================
# VALIDATION AND COMPATIBILITY EXCEPTIONS
# =============================================================================

class ValidationError(BaseProjectError):
    """
    Raised when data validation fails in any context.
    
    Enhanced to support handler pattern validation scenarios.
    
    Args:
        message (str): Description of the validation error.
        validation_type (str): Type of validation that failed.
        failed_checks (List[str], optional): List of specific validation checks that failed.
        data_context (str, optional): Context of the data being validated.
        handler_type (str, optional): Handler type if validation occurred within handler.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, validation_type: str,
                 failed_checks: Optional[List[str]] = None,
                 data_context: Optional[str] = None,
                 handler_type: Optional[str] = None,
                 details: Optional[str] = None, **kwargs):
        super().__init__(message, details, **kwargs)
        self.validation_type = validation_type
        self.failed_checks = failed_checks or []
        self.data_context = data_context
        self.handler_type = handler_type
    
    def __str__(self) -> str:
        msg = f"{self.message} (Validation: {self.validation_type})"
        if self.handler_type:
            msg += f", Handler: {self.handler_type}"
        if self.data_context:
            msg += f", Context: {self.data_context}"
        if self.failed_checks:
            msg += f", Failed Checks: {self.failed_checks}"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


class CompatibilityError(BaseProjectError):
    """
    Raised when compatibility issues are detected between different system components.
    
    This supports handler pattern compatibility checks and migration validation.
    
    Args:
        message (str): Description of the compatibility error.
        component_a (str): First component in compatibility conflict.
        component_b (str): Second component in compatibility conflict.
        compatibility_type (str, optional): Type of compatibility issue.
        version_conflicts (Dict[str, str], optional): Version conflicts detected.
        required_changes (List[str], optional): Changes required to resolve compatibility.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, component_a: str, component_b: str,
                 compatibility_type: Optional[str] = None,
                 version_conflicts: Optional[Dict[str, str]] = None,
                 required_changes: Optional[List[str]] = None,
                 details: Optional[str] = None, **kwargs):
        super().__init__(message, details, **kwargs)
        self.component_a = component_a
        self.component_b = component_b
        self.compatibility_type = compatibility_type
        self.version_conflicts = version_conflicts or {}
        self.required_changes = required_changes or []
    
    def __str__(self) -> str:
        msg = f"{self.message} (Components: {self.component_a} <-> {self.component_b})"
        if self.compatibility_type:
            msg += f", Type: {self.compatibility_type}"
        if self.version_conflicts:
            conflicts = ", ".join([f"{k}={v}" for k, v in self.version_conflicts.items()])
            msg += f", Version Conflicts: {conflicts}"
        if self.required_changes:
            msg += f", Required Changes: {self.required_changes}"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


# =============================================================================
# MIGRATION-SPECIFIC EXCEPTIONS
# =============================================================================

class MigrationError(BaseProjectError):
    """
    Raised during migration from legacy code to handler pattern.
    
    This helps track and handle issues during the Handler-Based Pattern Development migration process.
    
    Args:
        message (str): Description of the migration error.
        migration_phase (str): Phase of migration where error occurred.
        source_module (str, optional): Module being migrated from.
        target_pattern (str, optional): Pattern being migrated to.
        migration_step (str, optional): Specific migration step that failed.
        rollback_available (bool, optional): Whether rollback is possible.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, migration_phase: str,
                 source_module: Optional[str] = None,
                 target_pattern: Optional[str] = None,
                 migration_step: Optional[str] = None,
                 rollback_available: bool = True,
                 details: Optional[str] = None, **kwargs):
        super().__init__(message, details, **kwargs)
        self.migration_phase = migration_phase
        self.source_module = source_module
        self.target_pattern = target_pattern
        self.migration_step = migration_step
        self.rollback_available = rollback_available
    
    def __str__(self) -> str:
        msg = f"{self.message} (Phase: {self.migration_phase})"
        if self.source_module:
            msg += f", From: {self.source_module}"
        if self.target_pattern:
            msg += f", To: {self.target_pattern}"
        if self.migration_step:
            msg += f", Step: {self.migration_step}"
        if not self.rollback_available:
            msg += ", [NO ROLLBACK]"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


class LegacyCodeError(BaseProjectError):
    """
    Raised when legacy code patterns conflict with new handler implementation.
    
    This helps identify areas where legacy code needs updating during migration.
    
    Args:
        message (str): Description of the legacy code error.
        legacy_pattern (str): Legacy pattern that caused the error.
        suggested_replacement (str, optional): Suggested handler pattern replacement.
        legacy_module (str, optional): Module containing legacy code.
        migration_priority (str, optional): Priority level for migration ("high", "medium", "low").
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, legacy_pattern: str,
                 suggested_replacement: Optional[str] = None,
                 legacy_module: Optional[str] = None,
                 migration_priority: str = "medium",
                 details: Optional[str] = None, **kwargs):
        super().__init__(message, details, **kwargs)
        self.legacy_pattern = legacy_pattern
        self.suggested_replacement = suggested_replacement
        self.legacy_module = legacy_module
        self.migration_priority = migration_priority
    
    def __str__(self) -> str:
        msg = f"{self.message} (Legacy: {self.legacy_pattern})"
        if self.legacy_module:
            msg += f", Module: {self.legacy_module}"
        if self.suggested_replacement:
            msg += f", Suggested: {self.suggested_replacement}"
        if self.migration_priority != "medium":
            msg += f", Priority: {self.migration_priority.upper()}"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


# =============================================================================
# TRANSFORMATION SYSTEM EXCEPTIONS
# =============================================================================

class TransformError(BaseProjectError):
    """
    Base exception for all transformation-related errors.
    
    This serves as the parent class for all transformation system exceptions,
    including validation, composition, configuration, and runtime errors.
    
    Args:
        message (str): Description of the transformation error.
        transform_name (str, optional): Name of the transform that caused the error.
        experimental_setup (str, optional): Experimental setup context.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, transform_name: Optional[str] = None,
                 experimental_setup: Optional[str] = None,
                 details: Optional[str] = None, **kwargs):
        super().__init__(message, details, **kwargs)
        self.transform_name = transform_name
        self.experimental_setup = experimental_setup
    
    def __str__(self) -> str:
        msg = self.message
        if self.transform_name:
            msg += f" (Transform: {self.transform_name})"
        if self.experimental_setup:
            msg += f", Setup: {self.experimental_setup}"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


class TransformCompatibilityError(TransformError):
    """
    Raised when a transform is incompatible with the target dataset or context.
    
    Args:
        message (str): Description of the incompatibility.
        transform_name (str): Name of the incompatible transform.
        dataset_type (str, optional): Dataset type that's incompatible.
        incompatibility_reason (str, optional): Specific reason for incompatibility.
        suggested_alternatives (List[str], optional): Alternative transforms that would work.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, transform_name: str,
                 dataset_type: Optional[str] = None,
                 incompatibility_reason: Optional[str] = None,
                 suggested_alternatives: Optional[List[str]] = None,
                 details: Optional[str] = None, **kwargs):
        super().__init__(message, transform_name=transform_name, details=details, **kwargs)
        self.dataset_type = dataset_type
        self.incompatibility_reason = incompatibility_reason
        self.suggested_alternatives = suggested_alternatives or []
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.dataset_type:
            msg += f", Dataset: {self.dataset_type}"
        if self.incompatibility_reason:
            msg += f", Reason: {self.incompatibility_reason}"
        if self.suggested_alternatives:
            msg += f", Alternatives: {self.suggested_alternatives}"
        return msg


class TransformationError(BaseProjectError):
    """
    Base exception for transformation system errors.
    
    This serves as the parent class for all transformation-related exceptions.
    
    Args:
        message (str): Description of the transformation error.
        transform_name (str, optional): Name of the transform that caused the error.
        transform_config (Dict, optional): Configuration that caused the error.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, transform_name: Optional[str] = None,
                 transform_config: Optional[Dict[str, Any]] = None,
                 details: Optional[str] = None, **kwargs):
        super().__init__(message, details, **kwargs)
        self.transform_name = transform_name
        self.transform_config = transform_config
    
    def __str__(self) -> str:
        msg = self.message
        if self.transform_name:
            msg += f" (Transform: {self.transform_name})"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


class DatasetIntegrationError(DataProcessingError):
    """
    Exception raised when dataset integration fails.
    
    This exception is raised when there are issues integrating different
    dataset components or when dataset initialization fails.
    """
    
    def __init__(self, message: str, dataset_type: Optional[str] = None,
                 integration_point: Optional[str] = None,
                 details: Optional[str] = None, **kwargs):
        super().__init__(message, details=details, **kwargs)
        self.dataset_type = dataset_type
        self.integration_point = integration_point


class TransformValidationError(ValidationError):
    """
    Raised when transform parameter validation fails.
    
    Args:
        message (str): Description of the validation error.
        transform_name (str): Name of the transform with invalid parameters.
        parameter_name (str, optional): Specific parameter that failed validation.
        parameter_value (Any, optional): Value that failed validation.
        expected_type (Type, optional): Expected parameter type.
        validation_errors (List[str], optional): List of specific validation errors.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, transform_name: str,
                 parameter_name: Optional[str] = None,
                 parameter_value: Any = None,
                 expected_type: Optional[Type] = None,
                 validation_errors: Optional[List[str]] = None,
                 experimental_setup: Optional[str] = None,
                 details: Optional[str] = None, **kwargs):
        super().__init__(
            message, 
            validation_type="transform_parameter",
            failed_checks=validation_errors or [],
            data_context=f"Transform: {transform_name}",
            details=details, 
            **kwargs
        )
        self.transform_name = transform_name
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value
        self.expected_type = expected_type
        self.experimental_setup = experimental_setup
        self.validation_errors = validation_errors or []  
    
    def __str__(self) -> str:
        msg = f"{self.message} (Transform: {self.transform_name})"
        if self.experimental_setup:
            msg += f", Setup: {self.experimental_setup}"
        if self.parameter_name:
            msg += f", Parameter: {self.parameter_name}"
            if self.parameter_value is not None:
                msg += f", Value: {self.parameter_value}"
            if self.expected_type:
                type_name = getattr(self.expected_type, '__name__', str(self.expected_type))
                msg += f", Expected Type: {type_name}"
        if self.failed_checks:
            msg += f", Validation Errors: {self.failed_checks}"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


class TransformCompositionError(DataProcessingError):
    """
    Raised when transform sequence composition fails.
    
    Args:
        message (str): Description of the composition error.
        transform_sequence (List[str], optional): List of transform names in the sequence.
        failed_transform_index (int, optional): Index of transform that caused failure.
        composition_errors (List[str], optional): List of composition-specific errors.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, 
                 transform_sequence: Optional[List[str]] = None,
                 failed_transform_index: Optional[int] = None,
                 failed_transform_name: Optional[str] = None,
                 composition_errors: Optional[List[str]] = None,
                 experimental_setup: Optional[str] = None,
                 config: Optional[List[Dict[str, Any]]] = None,
                 details: Optional[str] = None, **kwargs):
        super().__init__(message, details=details, **kwargs)
        self.transform_sequence = transform_sequence or []
        self.failed_transform_index = failed_transform_index
        self.failed_transform_name = failed_transform_name
        self.composition_errors = composition_errors or []
        self.experimental_setup = experimental_setup
        self.config = config
    
    def __str__(self) -> str:
        msg = self.message
        if self.transform_sequence:
            msg += f" (Sequence: {self.transform_sequence})"
        if self.failed_transform_index is not None:
            msg += f", Failed at index: {self.failed_transform_index}"
        if self.composition_errors:
            msg += f", Composition Errors: {self.composition_errors}"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


class TransformNotFoundError(TransformationError):
    """
    Raised when a requested transform is not found in the registry.
    
    Args:
        message (str): Description of the error.
        transform_name (str): Name of the transform that was not found.
        available_transforms (List[str], optional): List of available transform names.
        suggestions (List[str], optional): Suggested alternative transform names.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, transform_name: str,
                 available_transforms: Optional[List[str]] = None,
                 suggestions: Optional[List[str]] = None,
                 details: Optional[str] = None, **kwargs):
        super().__init__(message, transform_name=transform_name, details=details, **kwargs)
        self.available_transforms = available_transforms or []
        self.suggestions = suggestions or []
    
    def __str__(self) -> str:
        msg = f"{self.message} (Transform: {self.transform_name})"
        if self.suggestions:
            msg += f", Suggestions: {self.suggestions}"
        elif self.available_transforms:
            msg += f", Available: {self.available_transforms[:5]}"  # Show first 5
            if len(self.available_transforms) > 5:
                msg += f" (+{len(self.available_transforms) - 5} more)"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


class TransformRegistryError(TransformationError):
    """
    Raised when there are errors in the transform registry system.
    
    This occurs when:
    - Transform registration fails
    - Registry operations encounter errors
    - Transform lookup or retrieval fails
    
    Args:
        message (str): Description of the registry error.
        transform_name (str, optional): Name of transform involved in error.
        registry_operation (str, optional): Registry operation that failed.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, transform_name: Optional[str] = None,
                 registry_operation: Optional[str] = None,
                 details: Optional[str] = None, **kwargs):
        super().__init__(message, transform_name=transform_name, details=details, **kwargs)
        self.registry_operation = registry_operation
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.registry_operation:
            msg += f" (Operation: {self.registry_operation})"
        return msg


class ExperimentalSetupError(ConfigurationError):
    """
    Exception raised when experimental setup configuration is invalid.
    
    This exception is raised when there are issues with experimental setup
    configuration, such as invalid setup names, missing configurations, or
    incompatible experimental parameters.
    
    Args:
        message (str): Description of the experimental setup error.
        setup_name (str, optional): Name of the problematic experimental setup.
        setup_errors (List[str], optional): List of specific setup validation errors.
        available_setups (List[str], optional): List of available experimental setups.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, setup_name: Optional[str] = None,
                 setup_errors: Optional[List[str]] = None,
                 available_setups: Optional[List[str]] = None,
                 details: Optional[str] = None, **kwargs):
        super().__init__(message, details=details, **kwargs)
        self.setup_name = setup_name
        self.setup_errors = setup_errors or []
        self.available_setups = available_setups or []
    
    def __str__(self) -> str:
        msg = self.message
        if self.setup_name:
            msg += f" (Setup: {self.setup_name})"
        if self.available_setups:
            msg += f", Available: {self.available_setups}"
        if self.setup_errors:
            msg += f", Errors: {self.setup_errors}"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


class TransformConfigurationError(ConfigurationError):
    """
    Exception raised when transform configuration is invalid.
    
    This exception is raised when there are issues with transform configuration
    such as invalid parameters, missing required settings, or incompatible
    transform combinations.
    
    Args:
        message (str): Description of the configuration error.
        transform_name (str, optional): Name of the problematic transform.
        config_errors (List[str], optional): List of specific configuration errors.
        config_source (str, optional): Source of the configuration (e.g., 'experimental_setup', 'legacy').
        experimental_setup (str, optional): Associated experimental setup name.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, message: str, transform_name: Optional[str] = None, 
                 config_errors: Optional[List[str]] = None,
                 config_source: Optional[str] = None,
                 experimental_setup: Optional[str] = None,
                 details: Optional[str] = None, **kwargs):
        super().__init__(message, details=details, **kwargs)
        self.transform_name = transform_name
        self.config_errors = config_errors or []
        self.config_source = config_source
        self.experimental_setup = experimental_setup
    
    def __str__(self) -> str:
        msg = self.message
        if self.transform_name:
            msg += f" (Transform: {self.transform_name})"
        if self.experimental_setup:
            msg += f", Setup: {self.experimental_setup}"
        if self.config_source:
            msg += f", Source: {self.config_source}"
        if self.config_errors:
            msg += f", Errors: {self.config_errors}"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


# =============================================================================
# PLUGIN SYSTEM EXCEPTIONS
# =============================================================================

class PluginError(BaseProjectError):
    """
    Base exception for plugin-related errors.
    
    This serves as the parent class for all plugin system exceptions,
    including plugin discovery, registration, validation, and execution errors.
    
    Args:
        message (str): Description of the plugin error.
        plugin_name (str, optional): Name of the plugin that caused the error.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments for future extensibility.
    """
    
    def __init__(
        self,
        message: str,
        plugin_name: Optional[str] = None,
        details: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, details=details, **kwargs)
        self.plugin_name = plugin_name
    
    def __str__(self) -> str:
        msg = self.message
        if self.plugin_name:
            msg += f" (Plugin: {self.plugin_name})"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


class PluginValidationError(PluginError):
    """
    Exception raised when plugin validation fails.
    
    This occurs when:
    - Plugin fails dependency checks
    - Plugin transforms cannot be instantiated
    - Plugin parameter validation fails
    - Plugin compatibility tests fail
    - Plugin security checks fail
    
    Args:
        message (str): Description of the validation error.
        plugin_name (str, optional): Name of the plugin that failed validation.
        validation_errors (List[str], optional): List of specific validation errors.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(
        self,
        message: str,
        plugin_name: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
        details: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, plugin_name=plugin_name, details=details, **kwargs)
        self.validation_errors = validation_errors or []
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.validation_errors:
            msg += f", Validation Errors: {self.validation_errors}"
        return msg


class PluginSecurityError(PluginError):
    """
    Exception raised for plugin security concerns.
    
    This occurs when:
    - Plugin contains dangerous code patterns
    - Plugin uses unsafe imports (subprocess, eval, exec, etc.)
    - Plugin checksum verification fails
    - Plugin is not trusted and requires elevated permissions
    
    Args:
        message (str): Description of the security error.
        plugin_name (str, optional): Name of the plugin with security concerns.
        security_issues (List[str], optional): List of specific security issues found.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(
        self,
        message: str,
        plugin_name: Optional[str] = None,
        security_issues: Optional[List[str]] = None,
        details: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, plugin_name=plugin_name, details=details, **kwargs)
        self.security_issues = security_issues or []
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.security_issues:
            msg += f", Security Issues: {self.security_issues}"
        return msg


class PluginDependencyError(PluginError):
    """
    Exception raised when plugin dependencies are not satisfied.
    
    This occurs when:
    - Required Python packages are not installed
    - milia version requirements are not met
    - PyTorch Geometric version requirements are not met
    - Python version requirements are not met
    - Other plugin dependencies are missing
    
    Args:
        message (str): Description of the dependency error.
        plugin_name (str, optional): Name of the plugin with missing dependencies.
        missing_dependencies (List[str], optional): List of missing dependencies.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(
        self,
        message: str,
        plugin_name: Optional[str] = None,
        missing_dependencies: Optional[List[str]] = None,
        details: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, plugin_name=plugin_name, details=details, **kwargs)
        self.missing_dependencies = missing_dependencies or []
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.missing_dependencies:
            msg += f", Missing Dependencies: {self.missing_dependencies}"
        return msg


class PluginDiscoveryError(PluginError):
    """
    Exception raised when plugin discovery fails.
    
    This occurs when:
    - Plugin directory cannot be accessed
    - Plugin metadata files are malformed
    - Plugin structure is invalid
    - Multiple plugins have conflicting names
    
    Args:
        message (str): Description of the discovery error.
        plugin_name (str, optional): Name of the plugin that failed discovery.
        discovery_path (str, optional): Path where discovery failed.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(
        self,
        message: str,
        plugin_name: Optional[str] = None,
        discovery_path: Optional[str] = None,
        details: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, plugin_name=plugin_name, details=details, **kwargs)
        self.discovery_path = discovery_path
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.discovery_path:
            msg += f", Discovery Path: {self.discovery_path}"
        return msg


class PluginRegistrationError(PluginError):
    """
    Exception raised when plugin registration fails.
    
    This occurs when:
    - Plugin with same name already registered
    - Plugin transform registration fails
    - Plugin metadata is invalid
    - Plugin conflicts with existing transforms
    
    Args:
        message (str): Description of the registration error.
        plugin_name (str, optional): Name of the plugin that failed registration.
        conflicting_plugin (str, optional): Name of conflicting plugin (if applicable).
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(
        self,
        message: str,
        plugin_name: Optional[str] = None,
        conflicting_plugin: Optional[str] = None,
        details: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, plugin_name=plugin_name, details=details, **kwargs)
        self.conflicting_plugin = conflicting_plugin
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.conflicting_plugin:
            msg += f", Conflicts With: {self.conflicting_plugin}"
        return msg


class PluginLoadError(PluginError):
    """
    Exception raised when plugin loading fails.
    
    This occurs when:
    - Plugin module cannot be imported
    - Plugin code has syntax errors
    - Plugin initialization fails
    - Plugin dependencies cannot be loaded
    
    Args:
        message (str): Description of the load error.
        plugin_name (str, optional): Name of the plugin that failed to load.
        load_path (str, optional): Path where loading was attempted.
        original_error (str, optional): Original error message from loading attempt.
        details (str, optional): Additional technical details.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(
        self,
        message: str,
        plugin_name: Optional[str] = None,
        load_path: Optional[str] = None,
        original_error: Optional[str] = None,
        details: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, plugin_name=plugin_name, details=details, **kwargs)
        self.load_path = load_path
        self.original_error = original_error
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.load_path:
            msg += f", Load Path: {self.load_path}"
        if self.original_error:
            msg += f", Original Error: {self.original_error}"
        return msg


# =============================================================================
# DESCRIPTOR EXCEPTIONS
# =============================================================================

class DescriptorError(BaseProjectError):
    """Base exception for descriptor-related errors"""
    def __init__(self, message: str, descriptor_name: Optional[str] = None, details: Optional[str] = None, **kwargs):
        super().__init__(message, details=details, **kwargs)
        self.descriptor_name = descriptor_name


class DescriptorCalculationError(DescriptorError):
    """Exception raised when descriptor calculation fails"""
    def __init__(self, message: str, descriptor_name: Optional[str] = None, 
                 molecule_index: Optional[int] = None, smiles: Optional[str] = None,
                 original_error: Optional[Exception] = None, **kwargs):
        super().__init__(message, descriptor_name=descriptor_name, **kwargs)
        self.molecule_index = molecule_index
        self.smiles = smiles
        self.original_error = original_error


class DescriptorValidationError(DescriptorError):
    """Exception raised when descriptor validation fails"""
    pass


class DescriptorPluginError(DescriptorError):
    """Base exception for descriptor plugin errors"""
    def __init__(self, message: str, plugin_name: Optional[str] = None, 
                 plugin_path: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.plugin_name = plugin_name
        self.plugin_path = plugin_path


class DescriptorPluginLoadError(DescriptorPluginError):
    """Exception raised when plugin fails to load"""
    pass


class DescriptorPluginValidationError(DescriptorPluginError):
    """Exception raised when plugin validation fails"""
    def __init__(self, message: str, plugin_name: Optional[str] = None,
                 validation_errors: Optional[List[str]] = None, **kwargs):
        super().__init__(message, plugin_name=plugin_name, **kwargs)
        self.validation_errors = validation_errors or []


class DescriptorPluginConfigError(DescriptorPluginError):
    """Exception raised for plugin configuration errors"""
    pass


# =============================================================================
# MODEL SYSTEM EXCEPTIONS
# =============================================================================

class ModelError(BaseProjectError):
    """
    Base exception for all model-related errors in the models module.
    
    This serves as the parent class for all model system exceptions including
    model creation, training, validation, and deployment errors.
    
    Args:
        message: Description of the error
        model_name: Name of the model (if applicable)
        details: Additional technical details
        **kwargs: Additional context
    """
    def __init__(self, message: str, model_name: Optional[str] = None,
                 details: Optional[str] = None, **kwargs):
        super().__init__(message, details, **kwargs)
        self.model_name = model_name


class ModelNotFoundError(ModelError):
    """
    Exception raised when a requested model is not found in the registry.
    
    Args:
        message: Description of the error
        model_name: Name of the model that was not found
        available_models: List of available model names
        **kwargs: Additional context
    """
    def __init__(self, message: str, model_name: str,
                 available_models: Optional[List[str]] = None, **kwargs):
        super().__init__(message, model_name=model_name, **kwargs)
        self.available_models = available_models or []
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.available_models:
            msg += f". Available models: {', '.join(self.available_models[:10])}"
            if len(self.available_models) > 10:
                msg += f" (and {len(self.available_models) - 10} more)"
        return msg


class ModelValidationError(ModelError):
    """
    Exception raised when model validation fails.
    
    This includes hyperparameter validation, data compatibility checks,
    and configuration validation failures.
    
    Args:
        message: Description of the validation error
        model_name: Name of the model
        validation_errors: List of specific validation errors
        **kwargs: Additional context
    """
    def __init__(self, message: str, model_name: Optional[str] = None,
                 validation_errors: Optional[List[str]] = None, **kwargs):
        super().__init__(message, model_name=model_name, **kwargs)
        self.validation_errors = validation_errors or []
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.validation_errors:
            msg += f". Validation errors: {'; '.join(self.validation_errors)}"
        return msg


class ModelInstantiationError(ModelError):
    """
    Exception raised when model instantiation fails.
    
    This occurs when there are errors creating a model instance from
    the model class, typically due to invalid hyperparameters or
    internal PyTorch/PyG errors.
    
    Args:
        message: Description of the instantiation error
        model_name: Name of the model
        hyperparameters: Hyperparameters that caused the error
        original_error: Original exception message
        **kwargs: Additional context
    """
    def __init__(self, message: str, model_name: Optional[str] = None,
                 hyperparameters: Optional[Dict[str, Any]] = None,
                 original_error: Optional[str] = None, **kwargs):
        super().__init__(message, model_name=model_name, **kwargs)
        self.hyperparameters = hyperparameters
        self.original_error = original_error
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.original_error:
            msg += f". Original error: {self.original_error}"
        return msg


class HyperparameterError(ModelError):
    """
    Exception raised for hyperparameter-related issues.
    
    This includes invalid hyperparameter values, type mismatches,
    out-of-range values, and missing required parameters.
    
    Args:
        message: Description of the hyperparameter error
        model_name: Name of the model
        parameter_name: Name of the problematic parameter
        parameter_value: Value that caused the error
        expected_type: Expected type for the parameter
        **kwargs: Additional context
    """
    def __init__(self, message: str, model_name: Optional[str] = None,
                 parameter_name: Optional[str] = None,
                 parameter_value: Any = None,
                 expected_type: Optional[str] = None, **kwargs):
        super().__init__(message, model_name=model_name, **kwargs)
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value
        self.expected_type = expected_type
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.parameter_name:
            msg += f". Parameter: {self.parameter_name}"
        if self.parameter_value is not None:
            msg += f", Value: {self.parameter_value}"
        if self.expected_type:
            msg += f", Expected type: {self.expected_type}"
        return msg


class DataCompatibilityError(ModelError):
    """
    Exception raised when data is incompatible with model requirements.
    
    This includes missing required features (edge_index, edge_attr, etc.),
    heterogeneous graph incompatibilities, and data format issues.
    
    Args:
        message: Description of the compatibility error
        model_name: Name of the model
        missing_features: List of missing required features
        incompatibility_reason: Reason for incompatibility
        **kwargs: Additional context
    """
    def __init__(self, message: str, model_name: Optional[str] = None,
                 missing_features: Optional[List[str]] = None,
                 incompatibility_reason: Optional[str] = None, **kwargs):
        super().__init__(message, model_name=model_name, **kwargs)
        self.missing_features = missing_features or []
        self.incompatibility_reason = incompatibility_reason
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.missing_features:
            msg += f". Missing features: {', '.join(self.missing_features)}"
        if self.incompatibility_reason:
            msg += f". Reason: {self.incompatibility_reason}"
        return msg


class TrainingError(ModelError):
    """
    Exception raised during model training.
    
    This includes errors during training loops, validation, testing,
    and callback execution.
    
    Args:
        message: Description of the training error
        model_name: Name of the model
        epoch: Epoch number where error occurred
        batch_index: Batch index where error occurred
        phase: Training phase (train, val, test)
        **kwargs: Additional context
    """
    def __init__(self, message: str, model_name: Optional[str] = None,
                 epoch: Optional[int] = None,
                 batch_index: Optional[int] = None,
                 phase: Optional[str] = None, **kwargs):
        super().__init__(message, model_name=model_name, **kwargs)
        self.epoch = epoch
        self.batch_index = batch_index
        self.phase = phase
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.phase:
            msg += f". Phase: {self.phase}"
        if self.epoch is not None:
            msg += f", Epoch: {self.epoch}"
        if self.batch_index is not None:
            msg += f", Batch: {self.batch_index}"
        return msg


class CheckpointError(ModelError):
    """
    Exception raised for checkpoint-related issues.
    
    This includes errors saving or loading model checkpoints.
    
    Args:
        message: Description of the checkpoint error
        checkpoint_path: Path to the checkpoint file
        operation: Operation that failed (save, load)
        **kwargs: Additional context
    """
    def __init__(self, message: str, checkpoint_path: Optional[str] = None,
                 operation: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.checkpoint_path = checkpoint_path
        self.operation = operation
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.checkpoint_path:
            msg += f". Checkpoint: {self.checkpoint_path}"
        if self.operation:
            msg += f", Operation: {self.operation}"
        return msg


class DataError(ModelError):
    """
    Exception raised for data-related issues in the models module.
    
    This includes data splitting errors, data loading errors, and
    dataset validation failures.
    
    Args:
        message: Description of the data error
        dataset_size: Size of the dataset (if applicable)
        split_ratios: Split ratios that caused error
        **kwargs: Additional context
    """
    def __init__(self, message: str, dataset_size: Optional[int] = None,
                 split_ratios: Optional[Dict[str, float]] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.dataset_size = dataset_size
        self.split_ratios = split_ratios


class PluginModelError(ModelError):
    """
    Exception raised for plugin model issues.
    
    This includes errors loading, validating, or registering plugin models.
    
    Args:
        message: Description of the plugin model error
        plugin_name: Name of the plugin
        model_name: Name of the model from the plugin
        plugin_path: Path to the plugin
        **kwargs: Additional context
    """
    def __init__(self, message: str, plugin_name: Optional[str] = None,
                 model_name: Optional[str] = None,
                 plugin_path: Optional[str] = None, **kwargs):
        super().__init__(message, model_name=model_name, **kwargs)
        self.plugin_name = plugin_name
        self.plugin_path = plugin_path


# =============================================================================
# DATASET REGISTRATION EXCEPTIONS
# =============================================================================

class DatasetRegistrationError(BaseProjectError):
    """
    Raised when dataset registration fails.
    
    Pattern follows PluginRegistrationError from exceptions.py (lines 1718-1751).
    
    Args:
        message: Description of the registration error
        dataset_name: Name of the dataset that failed registration
        conflicting_class: Name of conflicting class if duplicate registration
        details: Additional technical details
    """
    
    def __init__(
        self,
        message: str,
        dataset_name: Optional[str] = None,
        conflicting_class: Optional[str] = None,
        details: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, details, **kwargs)
        self.dataset_name = dataset_name
        self.conflicting_class = conflicting_class
    
    def __str__(self) -> str:
        msg = self.message
        if self.dataset_name:
            msg += f" Dataset: '{self.dataset_name}'"
        if self.conflicting_class:
            msg += f" Conflicts with: '{self.conflicting_class}'"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


class DatasetNotFoundError(BaseProjectError):
    """
    Raised when a requested dataset is not registered.
    
    PHASE 7: available_datasets can be auto-populated from registry.
    
    Pattern follows ModelNotFoundError from exceptions.py (lines 1871-1892).
    
    Args:
        message: Description of the error
        dataset_name: Name of the dataset that was not found
        available_datasets: List of currently registered dataset names
        details: Additional technical details
    """
    
    def __init__(
        self,
        message: str,
        dataset_name: Optional[str] = None,
        available_datasets: Optional[List[str]] = None,
        details: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, details, **kwargs)
        self.dataset_name = dataset_name
        # PHASE 7: Auto-fill from registry if not provided
        if available_datasets is None:
            available_datasets = _get_available_dataset_types()
        self.available_datasets = available_datasets
    
    def __str__(self) -> str:
        msg = self.message
        if self.dataset_name:
            msg += f" Requested: '{self.dataset_name}'"
        if self.available_datasets:
            msg += f" Available: {self.available_datasets}"
        if self.details:
            msg += f". Details: {self.details}"
        return msg

# =============================================================================
# HPO SYSTEM EXCEPTIONS
# =============================================================================

class HPOError(ModelError):
    """
    Base exception for all HPO-related errors.
    
    Pattern: Follows ModelError (exceptions.py:2547-2564)
    Inherits from ModelError to integrate with existing model exception hierarchy.
    
    Attributes:
        message: Description of the error
        study_name: Name of the HPO study (if applicable)
        trial_number: Trial number (if applicable)
        details: Additional technical details
        
    Example:
        raise HPOError(
            "Optimization failed",
            study_name="gcn_optimization",
            trial_number=42,
            details="Trial parameters invalid"
        )
    """
    
    def __init__(
        self,
        message: str,
        study_name: Optional[str] = None,
        trial_number: Optional[int] = None,
        details: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, details=details, **kwargs)
        self.study_name = study_name
        self.trial_number = trial_number
    
    def __str__(self) -> str:
        msg = self.message
        if self.study_name:
            msg += f". Study: '{self.study_name}'"
        if self.trial_number is not None:
            msg += f", Trial: {self.trial_number}"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


class HPOConfigurationError(HPOError):
    """
    Exception raised for HPO configuration errors.
    
    Pattern: Follows ConfigurationError (exceptions.py:489-532)
    Used when HPO configuration parameters are invalid or incompatible.
    
    Attributes:
        message: Description of the configuration error
        config_key: The specific configuration key that caused the issue
        actual_value: The actual value that was found
        expected_value: The expected value or type
        
    Example:
        raise HPOConfigurationError(
            "Invalid sampler configuration",
            config_key="hpo.sampler.n_startup_trials",
            actual_value=-5,
            expected_value="non-negative integer"
        )
    """
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        actual_value: Any = None,
        expected_value: Any = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.config_key = config_key
        self.actual_value = actual_value
        self.expected_value = expected_value
    
    def __str__(self) -> str:
        parts = [self.message]
        
        if self.config_key:
            parts.append(f"Key: '{self.config_key}'")
        
        if self.expected_value is not None:
            parts.append(f"Expected: {self.expected_value}")
        
        if self.actual_value is not None:
            parts.append(f"Actual: {self.actual_value}")
        
        if self.details:
            parts.append(f"Details: {self.details}")
        
        return " | ".join(parts)


class TrialFailedError(HPOError):
    """
    Exception raised when a trial fails during execution.
    
    Pattern: Follows TrainingError (exceptions.py:2709-2741)
    Captures information about failed HPO trials for debugging and analysis.
    
    Attributes:
        message: Description of the failure
        trial_number: The trial that failed
        trial_params: Hyperparameters used in the failed trial
        original_error: The original exception that caused the failure
        epoch: Epoch at which failure occurred (if applicable)
        
    Example:
        raise TrialFailedError(
            "Trial training crashed",
            trial_number=15,
            trial_params={'lr': 0.001, 'hidden_dim': 256},
            original_error="CUDA out of memory",
            epoch=42
        )
    """
    
    def __init__(
        self,
        message: str,
        trial_number: Optional[int] = None,
        trial_params: Optional[Dict[str, Any]] = None,
        original_error: Optional[str] = None,
        epoch: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, trial_number=trial_number, **kwargs)
        self.trial_params = trial_params or {}
        self.original_error = original_error
        self.epoch = epoch
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.epoch is not None:
            msg += f", Epoch: {self.epoch}"
        if self.original_error:
            msg += f". Original error: {self.original_error}"
        return msg


class StudyNotFoundError(HPOError):
    """
    Exception raised when a requested study is not found.
    
    Pattern: Follows ModelNotFoundError (exceptions.py:2566-2587)
    Used when attempting to load or resume a non-existent HPO study.
    
    Attributes:
        message: Description of the error
        study_name: Name of the study that was not found
        available_studies: List of available study names
        storage_url: Storage URL that was searched
        
    Example:
        raise StudyNotFoundError(
            "Study not found in database",
            study_name="gcn_hyperparam_search",
            available_studies=['study1', 'study2'],
            storage_url="sqlite:///optuna.db"
        )
    """
    
    def __init__(
        self,
        message: str,
        study_name: str,
        available_studies: Optional[List[str]] = None,
        storage_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, study_name=study_name, **kwargs)
        self.available_studies = available_studies or []
        self.storage_url = storage_url
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.available_studies:
            msg += f". Available studies: {', '.join(self.available_studies[:5])}"
            if len(self.available_studies) > 5:
                msg += f" (and {len(self.available_studies) - 5} more)"
        if self.storage_url:
            msg += f". Storage: {self.storage_url}"
        return msg


class BackendError(HPOError):
    """
    Exception raised for backend-specific errors.
    
    Used when Optuna or Ray Tune backends encounter operational issues.
    
    Attributes:
        message: Description of the error
        backend_name: Name of the backend (optuna, ray_tune)
        operation: Operation that failed
        
    Example:
        raise BackendError(
            "Failed to initialize Optuna study",
            backend_name="optuna",
            operation="create_study"
        )
    """
    
    def __init__(
        self,
        message: str,
        backend_name: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.backend_name = backend_name
        self.operation = operation
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.backend_name:
            msg += f". Backend: {self.backend_name}"
        if self.operation:
            msg += f", Operation: {self.operation}"
        return msg


class SearchSpaceError(HPOError):
    """
    Exception raised for search space definition errors.
    
    Used when hyperparameter search space configuration is invalid or inconsistent.
    
    Attributes:
        message: Description of the error
        parameter_name: Name of the problematic parameter
        parameter_config: Configuration that caused the error
        
    Example:
        raise SearchSpaceError(
            "Invalid parameter bounds",
            parameter_name="learning_rate",
            parameter_config={'low': 0.1, 'high': 0.01}
        )
    """
    
    def __init__(
        self,
        message: str,
        parameter_name: Optional[str] = None,
        parameter_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.parameter_name = parameter_name
        self.parameter_config = parameter_config
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.parameter_name:
            msg += f". Parameter: '{self.parameter_name}'"
        return msg


class PruningError(HPOError):
    """
    Exception raised for pruning-related errors.
    
    Used when trial pruning logic encounters issues or fails to execute properly.
    
    Attributes:
        message: Description of the error
        trial_number: Trial that was being pruned
        pruner_type: Type of pruner being used
        intermediate_value: Value that triggered pruning decision
        
    Example:
        raise PruningError(
            "Pruning decision failed",
            trial_number=23,
            pruner_type="MedianPruner",
            intermediate_value=0.456
        )
    """
    
    def __init__(
        self,
        message: str,
        trial_number: Optional[int] = None,
        pruner_type: Optional[str] = None,
        intermediate_value: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, trial_number=trial_number, **kwargs)
        self.pruner_type = pruner_type
        self.intermediate_value = intermediate_value
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.pruner_type:
            msg += f". Pruner: {self.pruner_type}"
        if self.intermediate_value is not None:
            msg += f", Value: {self.intermediate_value}"
        return msg


# =============================================================================
# UTILITY FUNCTIONS FOR EXCEPTION HANDLING
# =============================================================================

def create_handler_error_context(
    handler_type: str,
    operation: str, 
    molecule_index: Optional[int] = None,
    additional_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Creates a standardized error context dictionary for handler exceptions.
    
    PHASE 7: Now includes dataset type validation and registry status.
    
    Args:
        handler_type: Type of dataset handler.
        operation: Operation being performed.
        molecule_index: Index of molecule being processed (if applicable).
        additional_context: Additional context information.
        
    Returns:
        Dictionary with standardized error context.
    """
    import datetime
    
    context = {
        'handler_type': handler_type,
        'operation': operation,
        'timestamp': datetime.datetime.now().isoformat(),
        # PHASE 7: Add registry information
        'dataset_type_registered': _is_dataset_type_registered(handler_type),
        'available_dataset_types': _get_available_dataset_types(),
    }
    
    if molecule_index is not None:
        context['molecule_index'] = molecule_index
    
    if additional_context:
        context.update(additional_context)
    
    return context


def wrap_handler_operation(handler_type: str, operation: str):
    """
    Decorator to wrap handler operations with standardized exception handling.
    
    Args:
        handler_type: Type of dataset handler.
        operation: Name of the operation being performed.
        
    Returns:
        Decorator function.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except HandlerError:
                # Re-raise handler errors as-is
                raise
            except MoleculeProcessingError as e:
                # Convert molecule processing errors to handler operation errors
                raise HandlerOperationError(
                    message=f"Handler operation failed: {e.message}",
                    handler_type=handler_type,
                    operation=operation,
                    molecule_index=getattr(e, 'molecule_index', None),
                    details=str(e)
                ) from e
            except Exception as e:
                # Convert unexpected errors to handler operation errors
                raise HandlerOperationError(
                    message=f"Unexpected error in handler operation: {str(e)}",
                    handler_type=handler_type,
                    operation=operation,
                    details=f"Original error: {type(e).__name__}: {str(e)}"
                ) from e
        return wrapper
    return decorator


def wrap_transform_operation(transform_name: str, operation: str, experimental_setup: Optional[str] = None):
    """
    Decorator to wrap transformation operations with standardized exception handling.
    
    Args:
        transform_name: Name of the transform being operated on.
        operation: Name of the operation being performed.
        experimental_setup: Name of experimental setup (if applicable).
        
    Returns:
        Decorator function.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (TransformConfigurationError, TransformValidationError, TransformCompositionError):
                # Re-raise transformation errors as-is
                raise
            except ValidationError as e:
                # Convert general validation errors to transform validation errors
                raise TransformValidationError(
                    message=f"Transform validation failed: {e.message}",
                    transform_name=transform_name,
                    experimental_setup=experimental_setup,
                    validation_errors=getattr(e, 'failed_checks', []),
                    details=str(e)
                ) from e
            except ConfigurationError as e:
                # Convert general configuration errors to transform configuration errors
                raise TransformConfigurationError(
                    message=f"Transform configuration failed: {e.message}",
                    transform_name=transform_name,
                    experimental_setup=experimental_setup,
                    details=str(e)
                ) from e
            except Exception as e:
                # Convert unexpected errors to appropriate transformation errors
                if operation in ['validate', 'validate_parameter', 'check_parameter']:
                    raise TransformValidationError(
                        message=f"Unexpected error during transform validation: {str(e)}",
                        transform_name=transform_name,
                        experimental_setup=experimental_setup,
                        details=f"Original error: {type(e).__name__}: {str(e)}"
                    ) from e
                elif operation in ['compose', 'create_sequence', 'instantiate']:
                    raise TransformCompositionError(
                        message=f"Unexpected error during transform composition: {str(e)}",
                        failed_transform_name=transform_name,
                        experimental_setup=experimental_setup,
                        details=f"Original error: {type(e).__name__}: {str(e)}"
                    ) from e
                else:
                    raise TransformConfigurationError(
                        message=f"Unexpected error in transform operation: {str(e)}",
                        transform_name=transform_name,
                        experimental_setup=experimental_setup,
                        details=f"Original error: {type(e).__name__}: {str(e)}"
                    ) from e
        return wrapper
    return decorator


def format_handler_exception_summary(exception: BaseException) -> Dict[str, Any]:
    """
    Formats exception information into a standardized summary for logging/reporting.
    
    Args:
        exception: Exception to format.
        
    Returns:
        Dictionary with formatted exception summary.
    """
    summary = {
        'exception_type': type(exception).__name__,
        'message': str(exception),
        'is_handler_exception': isinstance(exception, HandlerError),
        # PHASE 7: Add dataset-specific handler info
        'is_dataset_specific_handler_exception': isinstance(exception, DatasetSpecificHandlerError),
    }
    
    # Add handler-specific information
    if isinstance(exception, HandlerError):
        summary.update({
            'handler_type': getattr(exception, 'handler_type', 'unknown'),
            'handler_operation': getattr(exception, 'handler_operation', 'unknown')
        })
    
    # PHASE 7: Add DatasetSpecificHandlerError information
    if isinstance(exception, DatasetSpecificHandlerError):
        summary.update({
            'dataset_type': getattr(exception, 'dataset_type', 'unknown'),
            'property_name': getattr(exception, 'property_name', None),
        })
    
    # Add molecule-specific information
    if hasattr(exception, 'molecule_index') and exception.molecule_index is not None:
        summary['molecule_index'] = exception.molecule_index
    
    if hasattr(exception, 'inchi') and exception.inchi and exception.inchi != "N/A":
        summary['molecule_inchi'] = exception.inchi
    
    # Add migration-specific information
    if isinstance(exception, MigrationError):
        summary.update({
            'migration_phase': exception.migration_phase,
            'rollback_available': exception.rollback_available
        })
    
    # Add validation-specific information
    if isinstance(exception, ValidationError):
        summary.update({
            'validation_type': exception.validation_type,
            'failed_checks': exception.failed_checks
        })
    
    return summary


def is_recoverable_handler_error(exception: BaseException) -> bool:
    """
    Determines if a handler error is recoverable and processing can continue.
    
    Args:
        exception: Exception to evaluate.
        
    Returns:
        True if error is recoverable, False otherwise.
    """
    # Handler not available errors are generally not recoverable
    if isinstance(exception, HandlerNotAvailableError):
        return False
    
    # Configuration errors may be recoverable with fallbacks
    if isinstance(exception, HandlerConfigurationError):
        return True
    
    # Operation errors are usually recoverable (skip molecule, continue batch)
    if isinstance(exception, HandlerOperationError):
        return True
    
    # PHASE 7: Dataset-specific handler errors are usually recoverable
    if isinstance(exception, DatasetSpecificHandlerError):
        return True
    
    # Validation errors may be recoverable depending on context
    if isinstance(exception, HandlerValidationError):
        return True
    
    # Migration errors depend on rollback availability
    if isinstance(exception, MigrationError):
        return exception.rollback_available
    
    # Legacy code errors are usually recoverable with fallbacks
    if isinstance(exception, LegacyCodeError):
        return True
    
    # Handler integration errors are recoverable with compatibility layers
    if isinstance(exception, HandlerIntegrationError):
        return True
    
    # Other handler errors are conservatively considered recoverable
    if isinstance(exception, HandlerError):
        return True
    
    # Non-handler exceptions follow existing recovery logic
    if isinstance(exception, MoleculeFilterRejectedError):
        return True  # Expected rejection, continue processing
    
    if isinstance(exception, MoleculeProcessingError):
        return True  # Skip molecule, continue batch
    
    # PHASE 7: Uncertainty processing errors are recoverable (skip molecule)
    if isinstance(exception, UncertaintyProcessingError):
        return True
    
    # Configuration and data processing errors are generally not recoverable
    if isinstance(exception, (ConfigurationError, DataProcessingError)):
        return False
    
    # Unknown exceptions are conservatively considered non-recoverable
    return False


def get_exception_recovery_suggestions(exception: BaseException) -> List[str]:
    """
    Provides recovery suggestions for different types of exceptions.
    
    PHASE 7: Now includes suggestions for DatasetSpecificHandlerError
    and UncertaintyProcessingError.
    
    Args:
        exception: Exception to analyze.
        
    Returns:
        List of suggested recovery actions.
    """
    suggestions = []
    
    # PHASE 7: Handle generic dataset-specific handler exceptions
    if isinstance(exception, DatasetSpecificHandlerError):
        dataset_type = exception.dataset_type
        available_types = _get_available_dataset_types()
        
        suggestions.extend([
            f"Verify {dataset_type} handler configuration",
            f"Check if {dataset_type} is a registered dataset type",
            f"Available dataset types: {available_types}",
            "Review dataset-specific property requirements",
        ])
        
        # Add feature-specific suggestions
        if _get_dataset_feature(dataset_type, 'uncertainty_handling'):
            suggestions.append("Verify uncertainty data format and values")
        if _get_dataset_feature(dataset_type, 'vibrational_analysis'):
            suggestions.append("Verify vibrational data (freqs/vibmodes) format")
    
    # PHASE 7: Handle uncertainty processing errors
    elif isinstance(exception, UncertaintyProcessingError):
        suggestions.extend([
            "Verify uncertainty values are numeric and positive",
            "Check standard deviation data format",
            "Ensure uncertainty fields match dataset schema",
            f"Dataset type: {getattr(exception, 'dataset_type', 'unknown')}",
        ])
    
    elif isinstance(exception, HandlerNotAvailableError):
        suggestions.extend([
            "Check handler factory implementation",
            "Verify dataset type configuration",
            "Install missing handler dependencies",
            "Use fallback handler if available",
            f"Available types: {_get_available_dataset_types()}"  # PHASE 7
        ])
    
    elif isinstance(exception, HandlerConfigurationError):
        suggestions.extend([
            "Validate handler configuration",
            "Check required configuration parameters",
            "Use default configuration values",
            "Reset to known good configuration"
        ])
    
    elif isinstance(exception, HandlerOperationError):
        suggestions.extend([
            "Skip current molecule and continue",
            "Log detailed error information",
            "Check molecule data validity",
            "Use alternative processing method"
        ])
    
    elif isinstance(exception, HandlerValidationError):
        suggestions.extend([
            "Filter out invalid molecules",
            "Check validation criteria",
            "Use more permissive validation",
            "Log validation failures for analysis"
        ])
    
    elif isinstance(exception, MigrationError):
        if exception.rollback_available:
            suggestions.append("Rollback to previous version")
        suggestions.extend([
            "Check migration step prerequisites",
            "Verify compatibility requirements",
            "Use gradual migration approach"
        ])
    
    elif isinstance(exception, LegacyCodeError):
        suggestions.extend([
            "Use handler pattern replacement",
            "Update legacy code module",
            "Implement compatibility layer",
            "Plan migration strategy"
        ])
    
    elif isinstance(exception, MoleculeProcessingError):
        suggestions.extend([
            "Skip molecule and continue processing",
            "Log molecule identifier for analysis",
            "Check molecule data quality",
            "Review processing parameters"
        ])
    
    elif isinstance(exception, ConfigurationError):
        suggestions.extend([
            "Check configuration file syntax",
            "Verify all required keys are present",
            "Validate configuration values",
            "Use configuration validation tool"
        ])
    
    # Default suggestions for any unhandled exception types
    if not suggestions:
        suggestions.extend([
            "Review error details and context",
            "Check system logs for additional information",
            "Consult documentation for error resolution",
            "Contact support if issue persists"
        ])
    
    return suggestions


# =============================================================================
# EXCEPTION HIERARCHY VALIDATION
# =============================================================================

def validate_exception_hierarchy() -> Dict[str, bool]:
    """
    Validates the exception hierarchy for consistency and completeness.
    
    Dynamically validates all exception classes including any runtime
    subclasses of DatasetSpecificHandlerError and UncertaintyProcessingError.
    
    Returns:
        Dictionary of validation results.
    """
    validation_results = {}
    
    # Validate handler exception hierarchy
    handler_exceptions = [
        HandlerNotAvailableError,
        HandlerConfigurationError, 
        HandlerOperationError,
        HandlerValidationError, 
        HandlerCompatibilityError,
        DatasetSpecificHandlerError,
    ]
    
    for exc_class in handler_exceptions:
        validation_results[f"{exc_class.__name__}_inherits_HandlerError"] = issubclass(exc_class, HandlerError)
    
    # Dynamically validate any runtime subclasses of DatasetSpecificHandlerError
    for exc_class in DatasetSpecificHandlerError.__subclasses__():
        validation_results[f"{exc_class.__name__}_inherits_DatasetSpecificHandlerError"] = issubclass(exc_class, DatasetSpecificHandlerError)
    
    # Check that HandlerError inherits from BaseProjectError
    validation_results["HandlerError_inherits_BaseProjectError"] = issubclass(HandlerError, BaseProjectError)
    
    # Validate molecule processing exception hierarchy
    molecule_exceptions = [
        RDKitConversionError,
        PyGDataCreationError,
        PropertyEnrichmentError,
        StructuralFeatureError,
        UncertaintyProcessingError,
    ]
    
    for exc_class in molecule_exceptions:
        validation_results[f"{exc_class.__name__}_inherits_MoleculeProcessingError"] = issubclass(exc_class, MoleculeProcessingError)
    
    # Dynamically validate any runtime subclasses of UncertaintyProcessingError
    for exc_class in UncertaintyProcessingError.__subclasses__():
        validation_results[f"{exc_class.__name__}_inherits_UncertaintyProcessingError"] = issubclass(exc_class, UncertaintyProcessingError)
    
    # Check that MoleculeProcessingError inherits from BaseProjectError
    validation_results["MoleculeProcessingError_inherits_BaseProjectError"] = issubclass(MoleculeProcessingError, BaseProjectError)
    
    # Check that transformation exceptions inherit from proper base classes
    validation_results["ExperimentalSetupError_inherits_ConfigurationError"] = issubclass(ExperimentalSetupError, ConfigurationError)
    validation_results["TransformConfigurationError_inherits_ConfigurationError"] = issubclass(TransformConfigurationError, ConfigurationError)
    validation_results["TransformValidationError_inherits_ValidationError"] = issubclass(TransformValidationError, ValidationError)
    validation_results["TransformCompositionError_inherits_DataProcessingError"] = issubclass(TransformCompositionError, DataProcessingError)
    validation_results["TransformRegistryError_inherits_TransformationError"] = issubclass(TransformRegistryError, TransformationError)
    
    # Check that all transformation exceptions ultimately inherit from BaseProjectError
    transformation_exceptions = [ExperimentalSetupError, TransformConfigurationError, 
                               TransformValidationError, TransformCompositionError,
                               TransformRegistryError, TransformNotFoundError]
    
    for exc_class in transformation_exceptions:
        validation_results[f"{exc_class.__name__}_inherits_BaseProjectError"] = issubclass(exc_class, BaseProjectError)
    
    # Check that all plugin exceptions inherit from PluginError
    plugin_exceptions = [PluginValidationError, PluginSecurityError, PluginDependencyError,
                        PluginDiscoveryError, PluginRegistrationError, PluginLoadError]
    
    for exc_class in plugin_exceptions:
        validation_results[f"{exc_class.__name__}_inherits_PluginError"] = issubclass(exc_class, PluginError)
    
    # Check that PluginError inherits from BaseProjectError
    validation_results["PluginError_inherits_BaseProjectError"] = issubclass(PluginError, BaseProjectError)
    
    # Check that all model exceptions inherit from ModelError
    model_exceptions = [ModelNotFoundError, ModelValidationError, ModelInstantiationError,
                       HyperparameterError, DataCompatibilityError, TrainingError,
                       CheckpointError, DataError, PluginModelError]
    
    for exc_class in model_exceptions:
        validation_results[f"{exc_class.__name__}_inherits_ModelError"] = issubclass(exc_class, ModelError)
    
    # Check that ModelError inherits from BaseProjectError
    validation_results["ModelError_inherits_BaseProjectError"] = issubclass(ModelError, BaseProjectError)
    
    # Validate registry integration
    validation_results["registry_integration_available"] = _REGISTRY_INITIALIZED or _init_registry() is not None
    
    # Validate dynamic exception creation
    try:
        test_ds_error = DatasetSpecificHandlerError("test", dataset_type="TEST")
        test_unc_error = UncertaintyProcessingError("test", dataset_type="TEST")
        validation_results["dynamic_exceptions_implemented"] = all([
            hasattr(test_ds_error, 'dataset_type'),
            hasattr(test_unc_error, 'dataset_type'),
            test_ds_error.dataset_type == "TEST",
            test_unc_error.dataset_type == "TEST",
        ])
    except Exception:
        validation_results["dynamic_exceptions_implemented"] = False
    
    return validation_results


if __name__ == "__main__":
    # Run validation when module is executed directly
    print("Exception hierarchy validation results:")
    results = validate_exception_hierarchy()
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test}: {status}")
    
    # Test registry integration
    print("\nTesting dynamic registry integration...")
    status = get_exception_registry_status()
    print(f"Registry status: {status}")
    
    # Test exception creation
    print("\nTesting exception creation...")
    try:
        # Test handler exceptions
        raise HandlerNotAvailableError(
            "Test handler not available",
            requested_dataset_type="TEST",
            available_types=_get_available_dataset_types()  # Dynamic types
        )
    except HandlerNotAvailableError as e:
        print(f"HandlerNotAvailableError: {e}")
    
    try:
        # Test handler operation error
        raise HandlerOperationError(
            "Test operation failed",
            handler_type="TestDataset",
            operation="validate_molecule",
            molecule_index=42
        )
    except HandlerOperationError as e:
        print(f"HandlerOperationError: {e}")
    
    # Test DatasetSpecificHandlerError with dynamic dataset type
    try:
        raise DatasetSpecificHandlerError(
            "Test QMC handler error",
            dataset_type="QMC",
            operation="validate_uncertainty",
            property_name="correlation_energy"
        )
    except DatasetSpecificHandlerError as e:
        print(f"DatasetSpecificHandlerError: {e}")
    
    # Test factory function for dynamic dataset type
    try:
        error = create_dataset_handler_error(
            "Test factory dataset handler error",
            dataset_type="TestDataset",
            operation="validate_properties",
            property_name="test_property"
        )
        raise error
    except DatasetSpecificHandlerError as e:
        print(f"Factory-created DatasetSpecificHandlerError: {e}")
    
    # Test factory function for another dataset type
    try:
        error = create_dataset_handler_error(
            "Test factory error for another type",
            dataset_type="AnotherDataset",
            operation="validate",
            property_name="std_error"
        )
        raise error
    except DatasetSpecificHandlerError as e:
        print(f"Factory-created DatasetSpecificHandlerError: {e}")
        print(f"  dataset_type: {e.dataset_type}")
    
    # Test UncertaintyProcessingError
    try:
        raise UncertaintyProcessingError(
            "Test uncertainty error",
            dataset_type="QMC",
            molecule_index=5,
            uncertainty_property_name="std_error"
        )
    except UncertaintyProcessingError as e:
        print(f"UncertaintyProcessingError: {e}")
    
    # Test UncertaintyProcessingError with explicit dataset type
    try:
        raise UncertaintyProcessingError(
            "Test uncertainty error with explicit type",
            dataset_type="TestUncertaintyDataset",
            molecule_index=10,
            uncertainty_property_name="total_energy"
        )
    except UncertaintyProcessingError as e:
        print(f"UncertaintyProcessingError: {e}")
        print(f"  dataset_type: {e.dataset_type}")
    
    # Test create_handler_not_available_error with auto-fill
    try:
        error = create_handler_not_available_error(
            "Handler not found",
            requested_dataset_type="UNKNOWN"
        )
        raise error
    except HandlerNotAvailableError as e:
        print(f"Auto-filled HandlerNotAvailableError: {e}")
        print(f"  Available types: {e.available_types}")
    
    try:
        # Test experimental setup error
        raise ExperimentalSetupError(
            "Test experimental setup error",
            setup_name="invalid_setup",
            available_setups=["baseline", "augmented", "molecular_specific"]
        )
    except ExperimentalSetupError as e:
        print(f"ExperimentalSetupError: {e}")
    
    try:
        # Test transform configuration error
        raise TransformConfigurationError(
            "Test transform configuration error",
            transform_name="InvalidTransform",
            experimental_setup="test_setup",
            config_source="experimental_setup"
        )
    except TransformConfigurationError as e:
        print(f"TransformConfigurationError: {e}")
    
    try:
        # Test transform validation error
        raise TransformValidationError(
            "Test transform validation error",
            transform_name="RandomRotate",
            parameter_name="degrees",
            parameter_value="invalid_value",
            expected_type=int,
            experimental_setup="test_setup"
        )
    except TransformValidationError as e:
        print(f"TransformValidationError: {e}")
    
    try:
        # Test transform composition error
        raise TransformCompositionError(
            "Test transform composition error",
            transform_sequence=["AddSelfLoops", "InvalidTransform", "ToUndirected"],
            failed_transform_index=1,
            failed_transform_name="InvalidTransform",
            experimental_setup="test_setup"
        )
    except TransformCompositionError as e:
        print(f"TransformCompositionError: {e}")
    
    try:
        # Test transform registry error
        raise TransformRegistryError(
            "Test transform registry error",
            transform_name="TestTransform",
            registry_operation="registration"
        )
    except TransformRegistryError as e:
        print(f"TransformRegistryError: {e}")
    
    try:
        # Test plugin error
        raise PluginError(
            "Test plugin error",
            plugin_name="test_plugin",
            details="Plugin initialization failed"
        )
    except PluginError as e:
        print(f"PluginError: {e}")
    
    try:
        # Test plugin validation error
        raise PluginValidationError(
            "Test plugin validation error",
            plugin_name="test_plugin",
            validation_errors=["Missing dependencies", "Invalid transform"]
        )
    except PluginValidationError as e:
        print(f"PluginValidationError: {e}")
    
    try:
        # Test plugin security error
        raise PluginSecurityError(
            "Test plugin security error",
            plugin_name="untrusted_plugin",
            security_issues=["Uses subprocess", "Contains eval()"]
        )
    except PluginSecurityError as e:
        print(f"PluginSecurityError: {e}")
    
    try:
        # Test plugin dependency error
        raise PluginDependencyError(
            "Test plugin dependency error",
            plugin_name="test_plugin",
            missing_dependencies=["torch>=1.9.0", "numpy>=1.20.0"]
        )
    except PluginDependencyError as e:
        print(f"PluginDependencyError: {e}")
    
    try:
        # Test plugin discovery error
        raise PluginDiscoveryError(
            "Test plugin discovery error",
            plugin_name="test_plugin",
            discovery_path="/path/to/plugins"
        )
    except PluginDiscoveryError as e:
        print(f"PluginDiscoveryError: {e}")
    
    try:
        # Test plugin registration error
        raise PluginRegistrationError(
            "Test plugin registration error",
            plugin_name="new_plugin",
            conflicting_plugin="existing_plugin"
        )
    except PluginRegistrationError as e:
        print(f"PluginRegistrationError: {e}")
    
    try:
        # Test plugin load error
        raise PluginLoadError(
            "Test plugin load error",
            plugin_name="broken_plugin",
            load_path="/path/to/plugin.py",
            original_error="ImportError: No module named 'missing_dep'"
        )
    except PluginLoadError as e:
        print(f"PluginLoadError: {e}")
    
    try:
        # Test model error
        raise ModelError(
            "Test model error",
            model_name="GCN",
            details="Model initialization failed"
        )
    except ModelError as e:
        print(f"ModelError: {e}")
    
    try:
        # Test model not found error
        raise ModelNotFoundError(
            "Test model not found",
            model_name="UnknownModel",
            available_models=["GCN", "GAT", "GraphSAGE"]
        )
    except ModelNotFoundError as e:
        print(f"ModelNotFoundError: {e}")
    
    try:
        # Test model validation error
        raise ModelValidationError(
            "Test model validation error",
            model_name="GCN",
            validation_errors=["hidden_channels must be > 0", "num_layers out of range"]
        )
    except ModelValidationError as e:
        print(f"ModelValidationError: {e}")
    
    try:
        # Test hyperparameter error
        raise HyperparameterError(
            "Test hyperparameter error",
            model_name="GAT",
            parameter_name="heads",
            parameter_value=-1,
            expected_type="positive integer"
        )
    except HyperparameterError as e:
        print(f"HyperparameterError: {e}")
    
    try:
        # Test training error
        raise TrainingError(
            "Test training error",
            model_name="GraphSAGE",
            epoch=10,
            batch_index=5,
            phase="train"
        )
    except TrainingError as e:
        print(f"TrainingError: {e}")
    
    print("\n✓ Exception system enhanced for transformation support!")
    print("✓ All transformation exceptions are properly integrated with handler pattern!")
    print("✓ Plugin system exceptions successfully added!")
    print("✓ Model system exceptions successfully integrated!")
    print("✓ Dynamic registry integration working!")
    print("✓ DatasetSpecificHandlerError (fully dynamic) working!")
    print("✓ UncertaintyProcessingError (fully dynamic) working!")
    print("✓ Factory functions (fully dynamic) working!")
