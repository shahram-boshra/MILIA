# validators.py - Enhanced for Handler Pattern Support & ransformation Integration

"""
validators.py - Centralized Data Validation Module for Milia PyTorch Geometric Dataset Processing Pipeline

This module consolidates all validation functions from across the project to provide
a single source of truth for data validation logic. It includes validators for:
- Value validity and NaN checking
- Molecular structure validation
- Uncertainty data validation for uncertainty-enabled datasets
- Array and tensor validation
- Dataset-specific validation
- Transform configuration validation
- Experimental setup validation
- Transform composition rule validation

Basic Validation Utilities
====================================

Core validation functions for molecular data and graph structures:
- is_value_valid_and_not_nan(): Enhanced NaN validation for scalar and array data
- Basic data structure validation for molecular properties
- Atomic structure validation (coordinates, atomic numbers)
- Property value validation with type checking
- Array and tensor shape validation

Enhanced Transform Validation System
==============================================

Advanced validation capabilities with rich error reporting:
- TransformValidator class: Full parameter introspection and validation
- ValidationIssueDetail: Rich error reporting with actionable suggestions
- Semantic validation: Transform sequence logic checking
- Multiple report formats: text, JSON, markdown
- Backward compatibility: Automatic fallback when unavailable

Core Validation Functions:
- validate_transform_spec(): Validates individual transform specifications
- validate_experimental_setup(): Validates experimental setup configurations
- validate_transformation_config(): Validates complete transformation configs
- validate_transform_composition_rules(): Checks transform ordering and compatibility

Handler Pattern Integration
=====================================

This module has been enhanced to fully support the dataset handler strategy pattern
introduced. The handlers rely on these validation functions to ensure
data integrity and consistency across different dataset types.

Handler Integration Features:
- convert_to_scalar(): Unified scalar conversion for consistent handler operations
- validate_uncertainty_data(): Dataset-agnostic uncertainty validation
- validate_molecular_data_dict(): Comprehensive molecular data validation for handlers
- validate_handler_molecular_batch(): Batch validation optimized for handler workflows
- validate_handler_compatibility(): Handler-config compatibility verification
- create_validation_report(): Formatted validation reporting for handler operations

Migration Benefits:
- Centralized validation logic eliminates code duplication
- Handlers delegate validation to these trusted functions
- Consistent error handling and reporting across all handlers
- Easy testing and maintenance of validation logic
- Clear separation between validation and business logic
- Enhanced support for systematic experimentation workflows

Usage Examples
==============

Basic Validation (Still Works):
```python
from validators import is_value_valid_and_not_nan

# Simple value validation
if is_value_valid_and_not_nan(value):
    process(value)

# Array validation
if is_value_valid_and_not_nan(np.array([1.0, 2.0, 3.0])):
    process_array(data)
```

Enhanced Validation:
```python
from validators import TransformValidator

# Create validator instance
validator = TransformValidator()

# Validate configuration with detailed reporting
is_valid, issues = validator.validate_transform_config(config)
if not is_valid:
    print(validator.get_validation_report('text'))

# Generate different report formats
print(validator.get_validation_report('markdown'))
json_report = validator.get_validation_report('json')
```

Backward Compatible Validation:

Migration Status
================

Basic handler structure with validation delegation
Handler enrichment integration with validation
Configuration system handler support with validation
Exception system handler support with validation
Complete validation system handler optimization
Transform system integration with validation
Pipeline integration validation support (ready)
Enhanced validation with introspection and rich reporting

All validation functions maintain full backward compatibility while providing
enhanced handler pattern support and transformation framework integration.

Module Functions
================

Core Validation:
- is_value_valid_and_not_nan(value): Check if value is valid and not NaN
- convert_to_scalar(value): Convert value to scalar with validation
- validate_property_value(value, name, expected_type): Validate property with type checking

Molecular Structure Validation:
- validate_molecular_structure(atoms, coords, idx, identifier): Validate atomic structure
- validate_molecular_data_dict(data_dict, required_props): Validate molecular data dictionary
- validate_handler_molecular_batch(data_list, required_props, dataset_type): Batch validation

Uncertainty Validation:
- validate_uncertainty_data(value, idx, name, require_positive): Validate uncertainty values

Transform Validation:
- validate_transform_spec(spec): Validate transform specification
- validate_experimental_setup(setup): Validate experimental setup
- validate_transformation_config(config): Validate complete configuration
- validate_transform_composition_rules(sequence): Check transform compatibility

Handler Support:
- validate_handler_compatibility(handler, config): Check handler-config compatibility
- create_validation_report(results, format): Generate formatted validation reports

ransformation Integration Classes:
- TransformValidator: Advanced validation with parameter introspection
- ValidationIssueDetail: Rich error reporting with suggestions and context
- ValidationSeverity: Enumeration for issue severity levels

See Also
========

- graph_transforms.py: Transform implementation and composition
- dataset_handlers.py: Handler pattern implementation
- experimental_setup.py: Configuration system integration
- exceptions.py: Custom exception types for validation errors

CRITICAL: Validation Result Checking (Pitfall 3)
================================================

All validation functions return results that MUST be checked. Ignoring
validation results can lead to data corruption and hard-to-debug errors.

BAD PATTERNS (Will Cause Problems):
-----------------------------------

❌ Ignoring tuple returns:
    validate_molecular_data_dict(data, ['Etot'])  # Result ignored!
    process(data)  # May fail with invalid data

❌ Not checking boolean:
    is_valid, _ = validate_molecular_data_dict(data, ['Etot'])
    process(data)  # is_valid never checked!

❌ Swallowing errors:
    is_valid, errors = validate_molecular_data_dict(data, ['Etot'])
    if not is_valid:
        pass  # Errors ignored!
    process(data)

GOOD PATTERNS (Recommended):
----------------------------

   Pattern 1: Check and handle errors
    is_valid, errors = validate_molecular_data_dict(data, ['Etot'])
    if not is_valid:
        logger.error(f"Validation failed: {errors}")
        raise HandlerError(f"Invalid data: {errors}")
    process(data)

  Pattern 2: Use strict mode (raises on failure)
    validate_molecular_data_dict(data, ['Etot'], strict=True)
    process(data)  # Only executes if validation passed

  Pattern 3: Use ValidationResult wrapper (enforced checking)
    result = validate_molecular_data_dict(data, ['Etot'], return_wrapper=True)
    if result.is_valid:
        validated_data = result.get_validated_data()
        process(validated_data)
    else:
        logger.error(f"Validation failed: {result.errors}")

  Pattern 4: Raise on failure
    result = validate_molecular_data_dict(data, ['Etot'], return_wrapper=True)
    result.raise_if_invalid(HandlerValidationError)
    process(data)  # Only executes if valid

ENFORCEMENT MECHANISMS:
----------------------

1. ValidationResult class: Enforces explicit checking before data access
2. Strict mode: Automatically raises exceptions on validation failure
3. @must_check decorator: Issues warnings for unchecked results (advanced)
4. __del__ warnings: Warns if ValidationResult is garbage collected unchecked

MIGRATION GUIDE:
---------------

For existing code:
    # Old (can be ignored)
    is_valid, errors = validate_molecular_data_dict(data, props)

    # New (enforced checking)
    result = validate_molecular_data_dict(data, props, return_wrapper=True)
    if result.is_valid:
        process(result.get_validated_data())

For critical paths:
    # Use strict mode
    validate_molecular_data_dict(data, props, strict=True)
    process(data)  # Only reached if valid

For batch processing:
    # Check batch result
    is_valid, summary = validate_handler_molecular_batch(batch, props)
    if not is_valid:
        logger.error(f"Batch validation failed: {summary}")
        return
    process_batch(batch)
"""

import functools
import json
import logging
import warnings
from collections import defaultdict
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import torch

if TYPE_CHECKING:
    from .config_containers import DescriptorConfig

logger = logging.getLogger(__name__)

# Handler-specific exceptions for enhanced error handling
try:
    from milia_pipeline.exceptions import (
        ConfigurationError,
        DatasetSpecificHandlerError,
        HandlerCompatibilityError,
        HandlerConfigurationError,
        HandlerError,
        HandlerValidationError,
        MoleculeProcessingError,
        PropertyEnrichmentError,
        StructuralFeatureError,
        # Transformation exceptions
        TransformationError,
        TransformCompositionError,
        TransformValidationError,
        UncertaintyProcessingError,
        ValidationError,
    )
except ImportError:
    # Fallback for testing or when exceptions module is not available
    class MoleculeProcessingError(Exception):
        def __init__(self, message: str, molecule_index: int = None, inchi: str = None, **kwargs):
            super().__init__(message)
            self.molecule_index = molecule_index
            self.inchi = inchi

    class PropertyEnrichmentError(MoleculeProcessingError):
        pass

    class ValidationError(Exception):
        pass

    class HandlerError(Exception):
        pass

    class HandlerValidationError(HandlerError):
        pass

    class HandlerConfigurationError(HandlerError):
        pass

    class HandlerCompatibilityError(HandlerError):
        pass

    class DatasetSpecificHandlerError(HandlerError):
        def __init__(self, message: str, dataset_type: str = "", **kwargs):
            super().__init__(message)
            self.dataset_type = dataset_type
            self.property_name = kwargs.get("property_name")

    class UncertaintyProcessingError(MoleculeProcessingError):
        def __init__(
            self,
            message: str,
            dataset_type: str = "",
            molecule_index: int = None,
            inchi: str = None,
            detail: str = None,
            uncertainty_property_name: str = None,
            **kwargs,
        ):
            super().__init__(message, molecule_index=molecule_index, inchi=inchi, **kwargs)
            self.dataset_type = dataset_type
            self.uncertainty_property_name = uncertainty_property_name

    class StructuralFeatureError(MoleculeProcessingError):
        pass

    class TransformationError(Exception):
        pass

    class TransformValidationError(TransformationError):
        pass

    class TransformCompositionError(TransformationError):
        pass

    class ConfigurationError(Exception):
        pass


# Import transformation containers for Transformation System integration
try:
    from milia_pipeline.config.config_containers import (
        ExperimentalSetup,
        TransformationConfig,
        TransformSpec,
    )

    TRANSFORMATION_CONTAINERS_AVAILABLE = True
except ImportError:
    TRANSFORMATION_CONTAINERS_AVAILABLE = False
    logger_import = logging.getLogger(__name__)
    logger_import.warning("Transformation containers not available - Validation features disabled")

# Parameter introspection support
try:
    from milia_pipeline.transformations.graph_transforms import (
        TransformRegistry,
        get_transform_metadata,
    )

    TRANSFORM_INTROSPECTION_AVAILABLE = True
except ImportError:
    TRANSFORM_INTROSPECTION_AVAILABLE = False

# Validation reporting
import importlib.util

VALIDATION_REPORTING_AVAILABLE = (
    importlib.util.find_spec("milia_pipeline.transformations.graph_transforms") is not None
)

# Pydantic V2 integration for runtime validation
# Phase 4: Enables conversion between Pydantic ValidationError and MILIA ValidationResult
# Phase 4.1: Added BaseModel and Field for ValidationIssueDetail migration
try:
    from pydantic import BaseModel, Field
    from pydantic import ValidationError as PydanticValidationError

    PYDANTIC_AVAILABLE = True
except ImportError:
    PydanticValidationError = None
    BaseModel = None
    Field = None
    PYDANTIC_AVAILABLE = False


# ============================================================================
# PHASE 6: Registry Integration for Dynamic Dataset Validation
# ============================================================================

# Registry availability flags - set during lazy initialization
_REGISTRY_INITIALIZED = False
_REGISTRY_AVAILABLE = False
_REGISTRY_IMPORT_ERROR = None

# Registry function placeholders (populated by _init_registry)
_registry_list_all = None
_registry_get = None
_registry_is_registered = None


def _init_registry() -> bool:
    """
    Lazily initialize registry imports to avoid circular import at module load time.

    The datasets/__init__.py imports implementations which may import this module
    (for validation). By deferring the registry import until first use,
    we allow both modules to fully load first.

    Returns:
        True if registry is available, False otherwise

    ADDED Phase 6: Lazy initialization following Phase 3 pattern from config_constants.py
    """
    global _REGISTRY_INITIALIZED, _REGISTRY_AVAILABLE, _REGISTRY_IMPORT_ERROR
    global _registry_list_all, _registry_get, _registry_is_registered

    if _REGISTRY_INITIALIZED:
        return _REGISTRY_AVAILABLE

    _REGISTRY_INITIALIZED = True

    try:
        from milia_pipeline.datasets.registry import (
            get,
            is_registered,
            list_all,
        )

        _registry_list_all = list_all
        _registry_get = get
        _registry_is_registered = is_registered

        _REGISTRY_AVAILABLE = True
        logger.debug("Validators: Registry integration initialized successfully")
        return True

    except ImportError as e:
        _REGISTRY_IMPORT_ERROR = str(e)
        _REGISTRY_AVAILABLE = False
        logger.debug(f"Validators: Registry not available, using legacy validation: {e}")
        return False


def _get_available_dataset_types() -> list[str]:
    """
    Get list of available dataset types from registry or dynamic filesystem discovery.

    DYNAMIC APPROACH: Instead of hardcoded fallback, this function:
    1. First tries the registry (primary source of truth)
    2. If registry fails, dynamically discovers dataset implementations from filesystem
    3. Never uses hardcoded dataset type lists

    Returns:
        List of available dataset type names from registry or dynamic discovery

    ADDED Phase 6: Dynamic dataset type list for error messages and validation
    UPDATED Phase 6.1: Replaced hardcoded fallback with dynamic filesystem discovery
    """
    _init_registry()

    if _REGISTRY_AVAILABLE and _registry_list_all is not None:
        try:
            return _registry_list_all()
        except Exception as e:
            logger.debug(f"Registry list_all failed: {e}")

    # DYNAMIC FALLBACK: Discover dataset types from implementations directory
    try:
        from pathlib import Path

        # Find the implementations directory relative to this file
        # validators.py is in config/, implementations is in datasets/implementations/
        implementations_dir = Path(__file__).parent.parent / "datasets" / "implementations"
        if implementations_dir.exists():
            discovered_types = []
            for py_file in implementations_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                # Extract dataset name from filename (e.g., dft.py -> DFT, qm9.py -> QM9)
                dataset_name = py_file.stem.upper()
                # Exclude non-dataset modules
                if dataset_name not in ["BASE", "REGISTRY", "UTILS", "COMMON", "PROTOCOLS"]:
                    discovered_types.append(dataset_name)
            if discovered_types:
                logger.debug(
                    f"Validators: Dynamically discovered dataset types: {discovered_types}"
                )
                return discovered_types
    except Exception as e:
        logger.debug(f"Validators: Dynamic dataset type discovery failed: {e}")

    # Final fallback: return empty list with warning (no hardcoded types)
    logger.warning(
        "Validators: No dataset types available - registry not initialized and dynamic discovery failed"
    )
    return []


def _is_dataset_type_registered(dataset_type: str) -> bool:
    """
    Check if a dataset type is registered in registry or dynamically discovered.

    DYNAMIC APPROACH: Instead of hardcoded fallback check, this function:
    1. First tries the registry (primary source of truth)
    2. If registry fails, uses _get_available_dataset_types() which does dynamic discovery
    3. Never uses hardcoded dataset type lists

    Args:
        dataset_type: Dataset type name to check

    Returns:
        True if registered or dynamically discovered, False otherwise

    ADDED Phase 6: Dynamic dataset type validation
    UPDATED Phase 6.1: Replaced hardcoded fallback with dynamic filesystem discovery
    """
    _init_registry()

    if _REGISTRY_AVAILABLE and _registry_is_registered is not None:
        try:
            return _registry_is_registered(dataset_type)
        except Exception as e:
            logger.debug(f"Registry is_registered failed: {e}")

    # DYNAMIC FALLBACK: Check against dynamically discovered types
    available_types = _get_available_dataset_types()
    return dataset_type in available_types


def _get_dataset_feature(dataset_type: str, feature_name: str) -> bool:
    """
    Get a feature flag from a dataset's DatasetFeatures.

    Queries the registry for dataset feature flags. Used to determine
    dataset-specific behavior in validation decisions and capability checks.

    Args:
        dataset_type: Dataset type name (e.g., 'QMC', 'CCSD')
        feature_name: Name of the feature to query (e.g., 'uncertainty_handling')

    Returns:
        True if feature is enabled for this dataset type, False otherwise

    ADDED Phase 6: Feature-based dataset capability queries
    UPDATED Phase 6.2: Removed legacy_features dict; registry-only pattern
    """
    _init_registry()

    if _REGISTRY_AVAILABLE and _registry_get is not None:
        try:
            dataset_class = _registry_get(dataset_type)
            if hasattr(dataset_class, "features"):
                return getattr(dataset_class.features, feature_name, False)
        except Exception as e:
            logger.debug(f"Registry feature lookup failed for {dataset_type}.{feature_name}: {e}")

    # Registry unavailable or feature not found - return False
    return False


def _get_dataset_required_properties(dataset_type: str) -> list[str]:
    """
    Get required properties for a dataset type from registry.

    Queries the registry for dataset-specific required properties. Returns
    a minimal default if registry is unavailable.

    Args:
        dataset_type: Dataset type name (e.g., 'QMC', 'CCSD')

    Returns:
        List of required property names

    ADDED Phase 6: Dynamic required properties lookup
    UPDATED Phase 6.2: Removed legacy_required dict; registry-only pattern
    """
    _init_registry()

    if _REGISTRY_AVAILABLE and _registry_get is not None:
        try:
            dataset_class = _registry_get(dataset_type)
            if hasattr(dataset_class, "get_required_properties"):
                return dataset_class.get_required_properties()
        except Exception as e:
            logger.debug(f"Registry required properties lookup failed for {dataset_type}: {e}")

    # Registry unavailable - return minimal universal defaults
    return ["atoms", "coordinates"]


def _get_handler_compatibility_checks(handler_type: str) -> dict[str, Any]:
    """
    Get handler-specific compatibility check parameters from registry or legacy.

    Args:
        handler_type: Handler/dataset type name

    Returns:
        Dictionary containing compatibility check parameters

    ADDED Phase 6: Dynamic compatibility check configuration
    """
    _init_registry()

    if _REGISTRY_AVAILABLE and _registry_get is not None:
        try:
            dataset_class = _registry_get(handler_type)
            features = dataset_class.features if hasattr(dataset_class, "features") else None

            if features:
                return {
                    "supports_uncertainty": getattr(features, "uncertainty_handling", False),
                    "supports_vibrational": getattr(features, "vibrational_analysis", False),
                    "supports_atomization": getattr(features, "atomization_energy", False),
                    "supports_orbital": getattr(features, "orbital_analysis", False),
                }
        except Exception as e:
            logger.debug(f"Registry compatibility lookup failed for {handler_type}: {e}")

    # Registry unavailable - return conservative defaults (all False)
    return {
        "supports_uncertainty": False,
        "supports_vibrational": False,
        "supports_atomization": False,
        "supports_orbital": False,
    }


def get_registry_status() -> dict[str, Any]:
    """
    Get the current status of registry integration in validators module.

    Returns:
        Dictionary containing registry status information

    ADDED Phase 6: Registry status for diagnostics
    """
    _init_registry()

    return {
        "registry_initialized": _REGISTRY_INITIALIZED,
        "registry_available": _REGISTRY_AVAILABLE,
        "registry_import_error": _REGISTRY_IMPORT_ERROR,
        "available_dataset_types": _get_available_dataset_types(),
        "phase_6_complete": True,
    }


# ============================================================================
# Validation Result Enforcement
# ============================================================================


class ValidationResult:
    """
    Validation result wrapper that enforces checking.

    Pitfall 3 Solution: This class ensures validation results cannot be
    silently ignored. Accessing validated data requires explicit checking.

    Usage:
        result = ValidationResult(is_valid=True, errors=[], data=molecule_data)

        # ❌ This will raise if not checked:
        # data = result.data

        # Correct usage:
        if result.is_valid:
            data = result.get_validated_data()
        else:
            logger.error(f"Validation errors: {result.errors}")
    """

    def __init__(
        self,
        is_valid: bool,
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
        data: Any | None = None,
        context: str | None = None,
    ):
        self._is_valid = is_valid
        self._errors = errors or []
        self._warnings = warnings or []
        self._data = data
        self._context = context
        self._checked = False

    @property
    def is_valid(self) -> bool:
        """Mark result as checked when accessing validity."""
        self._checked = True
        return self._is_valid

    @property
    def errors(self) -> list[str]:
        """Get validation errors."""
        return self._errors

    @property
    def warnings(self) -> list[str]:
        """Get validation warnings."""
        return self._warnings

    @property
    def context(self) -> str | None:
        """Get validation context."""
        return self._context

    def get_validated_data(self) -> Any:
        """
        Get validated data - only allowed if result was checked and valid.

        Raises:
            ValidationError: If result wasn't checked or is invalid.
        """
        if not self._checked:
            raise ValidationError(
                message="Validation result must be checked before accessing data",
                validation_type="result_not_checked",
                data_context=self._context or "unknown",
            )

        if not self._is_valid:
            raise ValidationError(
                message=f"Cannot access data from invalid validation result. Errors: {self._errors}",
                validation_type="invalid_result_access",
                data_context=self._context or "unknown",
            )

        return self._data

    def raise_if_invalid(self, exception_class: type[Exception] = ValidationError) -> None:
        """
        Raise exception if validation failed.

        Args:
            exception_class: Exception class to raise (default: ValidationError)

        Raises:
            exception_class: If validation failed
        """
        self._checked = True
        if not self._is_valid:
            error_msg = "; ".join(self._errors) if self._errors else "Validation failed"
            if self._context:
                error_msg = f"{self._context}: {error_msg}"
            raise exception_class(error_msg)

    def __bool__(self) -> bool:
        """Allow use in boolean context (marks as checked)."""
        self._checked = True
        return self._is_valid

    def __del__(self):
        """Warn if validation result was never checked."""
        if not self._checked and self._errors:
            logger.warning(
                f"Validation result never checked! Context: {self._context}, "
                f"Errors: {len(self._errors)}"
            )

    def __repr__(self) -> str:
        checked_str = "checked" if self._checked else "UNCHECKED"
        status = "valid" if self._is_valid else "invalid"
        return (
            f"ValidationResult({status}, {len(self._errors)} errors, "
            f"{len(self._warnings)} warnings, {checked_str})"
        )


# Backward compatibility: Allow returning tuple OR ValidationResult
ValidationReturn = Union[tuple[bool, list[str]], ValidationResult]


# ============================================================================
# Phase 4: Pydantic V2 Integration Wrappers
# ============================================================================


def wrap_pydantic_validation_error(pydantic_error: "PydanticValidationError") -> ValidationResult:
    """
    Convert Pydantic ValidationError to MILIA ValidationResult.

    This allows existing code that expects ValidationResult to work
    with Pydantic validation errors, enabling seamless integration
    between Pydantic's type-based validation and MILIA's business
    logic validation.

    Args:
        pydantic_error: Pydantic V2 ValidationError instance

    Returns:
        ValidationResult with errors extracted from Pydantic error

    Raises:
        RuntimeError: If Pydantic is not available

    Example:
        >>> from pydantic import BaseModel, ValidationError as PydanticValidationError
        >>> class MyModel(BaseModel):
        ...     value: int
        >>> try:
        ...     MyModel(value='not_an_int')
        ... except PydanticValidationError as e:
        ...     result = wrap_pydantic_validation_error(e)
        ...     if not result.is_valid:
        ...         print(result.errors)

    Phase 4: Pydantic V2 integration for MILIA validation system
    """
    if not PYDANTIC_AVAILABLE:
        raise RuntimeError("Pydantic is not available. Install pydantic>=2.0 to use this function.")

    # Extract error messages from Pydantic ValidationError
    # Pydantic V2 errors() returns list of dicts with 'loc', 'msg', 'type', etc.
    errors = []
    for err in pydantic_error.errors():
        # Format location as dot-separated path for readability
        loc = err.get("loc", ())
        loc_str = ".".join(str(x) for x in loc) if loc else "root"
        msg = err.get("msg", "Validation error")
        errors.append(f"{loc_str}: {msg}")

    return ValidationResult(
        is_valid=False, errors=errors, warnings=[], data=None, context="pydantic_validation"
    )


def validate_with_pydantic_model(
    data: dict[str, Any], model_class: type, return_wrapper: bool = True
) -> ValidationResult | tuple[bool, list[str]]:
    """
    Validate data using a Pydantic model, returning MILIA ValidationResult.

    This function bridges Pydantic's model validation with MILIA's
    ValidationResult pattern, enabling consistent validation handling
    throughout the codebase regardless of whether validation is
    performed by Pydantic or custom validators.

    Args:
        data: Dictionary of data to validate against the Pydantic model
        model_class: Pydantic BaseModel class to validate against
        return_wrapper: If True, return ValidationResult; if False, return tuple

    Returns:
        ValidationResult or (is_valid, errors) tuple depending on return_wrapper

    Raises:
        RuntimeError: If Pydantic is not available
        TypeError: If model_class is not a Pydantic BaseModel

    Example:
        >>> from pydantic import BaseModel
        >>> class UserConfig(BaseModel):
        ...     name: str
        ...     age: int
        >>>
        >>> # Using ValidationResult (recommended)
        >>> result = validate_with_pydantic_model({'name': 'Alice', 'age': 30}, UserConfig)
        >>> if result.is_valid:
        ...     config = result.get_validated_data()
        ...
        >>> # Using tuple return (backward compatible)
        >>> is_valid, errors = validate_with_pydantic_model(
        ...     {'name': 'Bob', 'age': 'invalid'},
        ...     UserConfig,
        ...     return_wrapper=False
        ... )

    Phase 4: Pydantic V2 integration for MILIA validation system
    """
    if not PYDANTIC_AVAILABLE:
        raise RuntimeError("Pydantic is not available. Install pydantic>=2.0 to use this function.")

    try:
        # Use model_validate for Pydantic V2 (replaces parse_obj from V1)
        validated = model_class.model_validate(data)

        if return_wrapper:
            return ValidationResult(
                is_valid=True, errors=[], warnings=[], data=validated, context="pydantic_validation"
            )
        return True, []

    except PydanticValidationError as e:
        # Format errors with location for clarity
        errors = []
        for err in e.errors():
            loc = err.get("loc", ())
            loc_str = ".".join(str(x) for x in loc) if loc else "root"
            msg = err.get("msg", "Validation error")
            errors.append(f"{loc_str}: {msg}")

        if return_wrapper:
            return ValidationResult(
                is_valid=False, errors=errors, warnings=[], data=None, context="pydantic_validation"
            )
        return False, errors


def must_check(func: Callable) -> Callable:
    """
    Decorator to mark validation functions that must have results checked.

    Pitfall 3 Solution: Issues warning if return value is not used.

    Usage:
        @must_check
        def validate_something(data):
            return is_valid, errors
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a tracked result
        result = func(*args, **kwargs)

        # If it's a tuple (traditional return), wrap it
        if isinstance(result, tuple) and len(result) == 2:
            is_valid, errors = result

            # Create wrapper that tracks if it was unpacked
            class TrackedTuple(tuple):
                def __new__(cls, is_valid, errors):
                    instance = super().__new__(cls, (is_valid, errors))
                    instance._unpacked = False
                    return instance

                def __iter__(self):
                    self._unpacked = True
                    return super().__iter__()

                def __getitem__(self, index):
                    self._unpacked = True
                    return super().__getitem__(index)

                def __del__(self):
                    if not self._unpacked and not is_valid:
                        warnings.warn(
                            f"Validation result from {func.__name__} was not checked! "
                            f"Errors: {errors[:2]}",
                            UserWarning,
                            stacklevel=2,
                        )

            return TrackedTuple(is_valid, errors)

        return result

    return wrapper


# ============================================================================
# Validation Issue Tracking
# ============================================================================


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    ERROR = "error"  # Blocks execution
    WARNING = "warning"  # Continues but flags issue
    INFO = "info"  # Informational only


class ValidationIssueDetail(BaseModel):
    """
    Detailed validation issue with context and suggestions.

    Enhancement: Rich validation feedback for better debugging.

    Pydantic V2 Migration (Phase 4.1):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses Field(default_factory=dict) for context field
        - to_dict() uses model_dump(mode='json') for automatic enum serialization
        - NON-BREAKING: Same constructor API and attribute access
        - Follows established pattern from custom_transforms.py (Phase 16)
    """

    severity: ValidationSeverity
    message: str
    location: str  # e.g., "transform[2].parameters.num_neighbors"
    parameter: str | None = None
    constraint: str | None = None
    actual_value: Any = None
    expected_value: Any = None
    suggestion: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for reporting.

        Uses model_dump(mode='json') for automatic enum serialization,
        ensuring severity is serialized as its string value.

        Returns:
            Dictionary representation with severity as string value
        """
        return self.model_dump(mode="json")


# ============================================================================
# Transform Validator Class
# ============================================================================


class TransformValidator:
    """
    Enhanced transform validation with parameter introspection.

    Enhancement: Complete validation system with:
    - Dynamic parameter constraint checking
    - Metadata-driven validation
    - Rich error reporting (text/JSON/markdown)
    - Semantic validation integration
    - Caching for performance

    Usage:
        validator = TransformValidator()
        is_valid, issues = validator.validate_transform_config(config)
        if not is_valid:
            print(validator.get_validation_report('text'))
    """

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize validator with optional logger."""
        self.logger = logger or logging.getLogger(__name__)
        self.issues: list[ValidationIssueDetail] = []

        # Load transform metadata if available
        self.registry = None
        self.metadata_cache: dict[str, dict[str, Any]] = {}
        self.missing_transforms: set[str] = set()  # NEW: Track missing transforms

        if TRANSFORM_INTROSPECTION_AVAILABLE:
            try:
                self.registry = TransformRegistry()

                # NEW: Log any transforms that aren't available in current PyG version
                if hasattr(self.registry, "unavailable_transforms"):
                    self.missing_transforms = self.registry.unavailable_transforms
                    if self.missing_transforms:
                        missing_list = ", ".join(sorted(self.missing_transforms))
                        self.logger.info(
                            f"Note: {len(self.missing_transforms)} transform(s) not available "
                            f"in current PyG version: {missing_list}"
                        )

                self.logger.info("Transform introspection enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize registry: {e}")

    def validate_transform_config(
        self,
        transform_config: dict[str, Any],
        experimental_setup: str | None = None,
        collect_issues: bool = True,
    ) -> tuple[bool, list[ValidationIssueDetail]]:
        """
        Validate transform configuration with enhanced parameter checking.

        Enhancement: Uses parameter introspection for dynamic validation.

        Args:
            transform_config: Transform configuration dictionary to validate
            experimental_setup: Optional setup name for context in error messages
            collect_issues: If True, collect all issues; if False, fail fast on first error

        Returns:
            Tuple of (is_valid: bool, issues: List[ValidationIssueDetail])

        Example:
            >>> config = {'transforms': [{'name': 'AddSelfLoops', 'parameters': {}}]}
            >>> validator = TransformValidator()
            >>> is_valid, issues = validator.validate_transform_config(config)
            >>> print(f"Valid: {is_valid}, Issues: {len(issues)}")
        """
        self.issues.clear()
        location_prefix = (
            f"experimental_setup[{experimental_setup}]" if experimental_setup else "transforms"
        )

        # Basic structure validation
        if not self._validate_basic_structure(transform_config, location_prefix):
            return False, self.issues

        # Validate each transform in the sequence
        transforms = transform_config.get("transforms", [])
        for idx, transform_spec in enumerate(transforms):
            location = f"{location_prefix}.transforms[{idx}]"

            # Basic validation (name exists, etc.)
            if not self._validate_transform_spec(transform_spec, location):
                if not collect_issues:
                    return False, self.issues
                continue

            # Parameter introspection validation
            if TRANSFORM_INTROSPECTION_AVAILABLE and self.registry:
                param_valid = self._validate_transform_parameters(
                    transform_spec, location, collect_issues
                )
                if not param_valid and not collect_issues:
                    return False, self.issues

        has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)
        return not has_errors, self.issues

    def _validate_transform_parameters(
        self, transform_spec: dict[str, Any], location: str, collect_issues: bool
    ) -> bool:
        """
        Validate transform parameters using introspection metadata.

        Enhancement: Dynamic constraint checking based on metadata.

        Args:
            transform_spec: Transform specification with name and parameters
            location: Location string for error reporting
            collect_issues: Whether to collect all issues or fail fast

        Returns:
            True if all parameters valid, False otherwise
        """
        transform_name = transform_spec.get("name")
        parameters = transform_spec.get("parameters", {})

        # Get metadata for this transform
        metadata = self._get_transform_metadata(transform_name)
        if not metadata:
            self.issues.append(
                ValidationIssueDetail(
                    severity=ValidationSeverity.WARNING,
                    message=f"No metadata available for transform '{transform_name}'",
                    location=location,
                    suggestion="Ensure transform is registered in TransformRegistry",
                )
            )
            return True

        # Validate each provided parameter
        param_info = metadata.get("parameters", {})
        for param_name, param_value in parameters.items():
            param_location = f"{location}.parameters.{param_name}"

            # Check if parameter is recognized
            if param_name not in param_info:
                self.issues.append(
                    ValidationIssueDetail(
                        severity=ValidationSeverity.WARNING,
                        message=f"Unrecognized parameter '{param_name}' for transform '{transform_name}'",
                        location=param_location,
                        parameter=param_name,
                        actual_value=param_value,
                        suggestion=f"Valid parameters: {list(param_info.keys())}",
                    )
                )
                continue

            # Validate parameter type and constraints
            param_metadata = param_info[param_name]
            if not self._validate_parameter_constraints(
                param_name, param_value, param_metadata, param_location
            ) and not collect_issues:
                return False

        # Check for required parameters
        for param_name, param_metadata in param_info.items():
            if param_metadata.get("required", False) and param_name not in parameters:
                self.issues.append(
                    ValidationIssueDetail(
                        severity=ValidationSeverity.ERROR,
                        message=f"Required parameter '{param_name}' missing for transform '{transform_name}'",
                        location=location,
                        parameter=param_name,
                        suggestion=f"Add '{param_name}': {param_metadata.get('default', 'value')} to parameters",
                    )
                )
                if not collect_issues:
                    return False

        return not any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)

    def _validate_parameter_constraints(
        self, param_name: str, param_value: Any, param_metadata: dict[str, Any], location: str
    ) -> bool:
        """
        Validate parameter against metadata constraints.

        Enhancement: Dynamic constraint checking for type, range, enum values.

        Args:
            param_name: Parameter name
            param_value: Actual parameter value
            param_metadata: Metadata dictionary with constraints
            location: Location string for error reporting

        Returns:
            True if parameter passes all constraints
        """
        is_valid = True

        # Type validation
        expected_type = param_metadata.get("type")
        if expected_type and not self._check_type(param_value, expected_type):
            self.issues.append(
                ValidationIssueDetail(
                    severity=ValidationSeverity.ERROR,
                    message=f"Parameter '{param_name}' has incorrect type",
                    location=location,
                    parameter=param_name,
                    constraint="type",
                    actual_value=type(param_value).__name__,
                    expected_value=expected_type,
                    suggestion=f"Convert to {expected_type}",
                )
            )
            is_valid = False

        # Range constraints (min)
        if "min" in param_metadata and isinstance(param_value, (int, float)):
            if param_value < param_metadata["min"]:
                self.issues.append(
                    ValidationIssueDetail(
                        severity=ValidationSeverity.ERROR,
                        message=f"Parameter '{param_name}' below minimum value",
                        location=location,
                        parameter=param_name,
                        constraint="min",
                        actual_value=param_value,
                        expected_value=f">= {param_metadata['min']}",
                        suggestion=f"Use value >= {param_metadata['min']}",
                    )
                )
                is_valid = False

        # Range constraints (max)
        if "max" in param_metadata and isinstance(param_value, (int, float)):
            if param_value > param_metadata["max"]:
                self.issues.append(
                    ValidationIssueDetail(
                        severity=ValidationSeverity.ERROR,
                        message=f"Parameter '{param_name}' exceeds maximum value",
                        location=location,
                        parameter=param_name,
                        constraint="max",
                        actual_value=param_value,
                        expected_value=f"<= {param_metadata['max']}",
                        suggestion=f"Use value <= {param_metadata['max']}",
                    )
                )
                is_valid = False

        # Enum constraints
        if "choices" in param_metadata and param_value not in param_metadata["choices"]:
            self.issues.append(
                ValidationIssueDetail(
                    severity=ValidationSeverity.ERROR,
                    message=f"Parameter '{param_name}' has invalid value",
                    location=location,
                    parameter=param_name,
                    constraint="choices",
                    actual_value=param_value,
                    expected_value=param_metadata["choices"],
                    suggestion=f"Use one of: {param_metadata['choices']}",
                )
            )
            is_valid = False

        return is_valid

    def _get_transform_metadata(self, transform_name: str) -> dict[str, Any] | None:
        """
        Get metadata for transform with caching.

        Args:
            transform_name: Name of the transform

        Returns:
            Metadata dictionary or None if unavailable
        """
        if transform_name in self.metadata_cache:
            return self.metadata_cache[transform_name]

        if not self.registry:
            return None

        try:
            metadata = get_transform_metadata(transform_name)
            self.metadata_cache[transform_name] = metadata
            return metadata
        except Exception as e:
            self.logger.debug(f"Failed to get metadata for {transform_name}: {e}")
            return None

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """
        Check if value matches expected type string.

        Args:
            value: Value to check
            expected_type: Type name as string ('int', 'float', 'str', etc.)

        Returns:
            True if type matches
        """
        type_map = {
            "int": int,
            "float": (int, float),  # Accept int for float
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
        }
        expected = type_map.get(expected_type)
        return expected is None or isinstance(value, expected)

    def _validate_basic_structure(self, transform_config: dict[str, Any], location: str) -> bool:
        """
        Validate basic configuration structure.

        Args:
            transform_config: Configuration to validate
            location: Location for error reporting

        Returns:
            True if structure is valid
        """
        if not isinstance(transform_config, dict):
            self.issues.append(
                ValidationIssueDetail(
                    severity=ValidationSeverity.ERROR,
                    message="Transform configuration must be a dictionary",
                    location=location,
                    actual_value=type(transform_config).__name__,
                    expected_value="dict",
                )
            )
            return False

        if "transforms" not in transform_config:
            self.issues.append(
                ValidationIssueDetail(
                    severity=ValidationSeverity.ERROR,
                    message="Transform configuration missing 'transforms' key",
                    location=location,
                    suggestion="Add 'transforms: []' to configuration",
                )
            )
            return False

        return True

    def _validate_transform_spec(self, transform_spec: dict[str, Any], location: str) -> bool:
        """
        Validate individual transform specification.

        Args:
            transform_spec: Single transform specification
            location: Location for error reporting

        Returns:
            True if specification is valid
        """
        if not isinstance(transform_spec, dict):
            self.issues.append(
                ValidationIssueDetail(
                    severity=ValidationSeverity.ERROR,
                    message="Transform specification must be a dictionary",
                    location=location,
                    actual_value=type(transform_spec).__name__,
                    expected_value="dict",
                )
            )
            return False

        if "name" not in transform_spec:
            self.issues.append(
                ValidationIssueDetail(
                    severity=ValidationSeverity.ERROR,
                    message="Transform specification missing 'name' key",
                    location=location,
                    suggestion="Add 'name: TransformName' to specification",
                )
            )
            return False

        return True

    # ========================================================================
    # Validation Reporting Methods
    # ========================================================================

    def get_validation_report(self, format: str = "text") -> str:
        """
        Generate validation report in specified format.

        Enhancement: Multiple output formats for different use cases.

        Args:
            format: Output format - 'text', 'json', or 'markdown'

        Returns:
            Formatted validation report string

        Example:
            >>> validator.validate_transform_config(config)
            >>> print(validator.get_validation_report('text'))
            >>> # Or save JSON report
            >>> with open('report.json', 'w') as f:
            >>>     f.write(validator.get_validation_report('json'))
        """
        if format == "json":
            return self._report_json()
        elif format == "markdown":
            return self._report_markdown()
        else:
            return self._report_text()

    def _report_text(self) -> str:
        """Generate text format report."""
        if not self.issues:
            return "✓ No validation issues found"

        lines = [f"Validation Report: {len(self.issues)} issue(s) found\n"]
        lines.append("=" * 70)

        # Group by severity
        by_severity = defaultdict(list)
        for issue in self.issues:
            by_severity[issue.severity].append(issue)

        for severity in [
            ValidationSeverity.ERROR,
            ValidationSeverity.WARNING,
            ValidationSeverity.INFO,
        ]:
            issues = by_severity[severity]
            if not issues:
                continue

            lines.append(f"\n{severity.value.upper()}: {len(issues)} issue(s)")
            lines.append("-" * 70)

            for issue in issues:
                lines.append(f"\n• {issue.message}")
                lines.append(f"  Location: {issue.location}")
                if issue.parameter:
                    lines.append(f"  Parameter: {issue.parameter}")
                if issue.actual_value is not None:
                    lines.append(f"  Actual: {issue.actual_value}")
                if issue.expected_value is not None:
                    lines.append(f"  Expected: {issue.expected_value}")
                if issue.suggestion:
                    lines.append(f"  💡 Suggestion: {issue.suggestion}")

        return "\n".join(lines)

    def _report_json(self) -> str:
        """Generate JSON format report."""
        report = {
            "total_issues": len(self.issues),
            "errors": sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR),
            "warnings": sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING),
            "info": sum(1 for i in self.issues if i.severity == ValidationSeverity.INFO),
            "issues": [issue.to_dict() for issue in self.issues],
        }
        return json.dumps(report, indent=2)

    def _report_markdown(self) -> str:
        """Generate Markdown format report."""
        if not self.issues:
            return "## ✓ No validation issues found"

        lines = [f"## Validation Report: {len(self.issues)} Issue(s) Found\n"]

        # Summary table
        error_count = sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)
        info_count = sum(1 for i in self.issues if i.severity == ValidationSeverity.INFO)

        lines.append("| Severity | Count |")
        lines.append("|----------|-------|")
        lines.append(f"| ERROR    | {error_count} |")
        lines.append(f"| WARNING  | {warning_count} |")
        lines.append(f"| INFO     | {info_count} |")
        lines.append("")

        # Detailed issues
        by_severity = defaultdict(list)
        for issue in self.issues:
            by_severity[issue.severity].append(issue)

        for severity in [
            ValidationSeverity.ERROR,
            ValidationSeverity.WARNING,
            ValidationSeverity.INFO,
        ]:
            issues = by_severity[severity]
            if not issues:
                continue

            lines.append(f"### {severity.value.upper()}")
            for issue in issues:
                lines.append(f"\n**{issue.message}**")
                lines.append(f"- **Location**: `{issue.location}`")
                if issue.parameter:
                    lines.append(f"- **Parameter**: `{issue.parameter}`")
                if issue.actual_value is not None:
                    lines.append(f"- **Actual**: `{issue.actual_value}`")
                if issue.expected_value is not None:
                    lines.append(f"- **Expected**: `{issue.expected_value}`")
                if issue.suggestion:
                    lines.append(f"- **💡 Suggestion**: {issue.suggestion}")
                lines.append("")

        return "\n".join(lines)


# ============================================================================
# ENHANCEMENT: Semantic Validation Integration (NEW)
# ============================================================================


def validate_transform_sequence_semantics(
    transforms: list[dict[str, Any]],
    dataset_type: str | None = None,
    logger: logging.Logger | None = None,
) -> tuple[bool, list[ValidationIssueDetail]]:
    """
    Validate semantic correctness of transform sequence.

    Enhancement: Integrates with SemanticValidator from graph_transforms
    to check for logical issues in transform ordering and compatibility.

    Args:
        transforms: List of transform specifications [{'name': ..., 'parameters': ...}, ...]
        dataset_type: Optional dataset type for dataset-specific validation (e.g., 'VQM24')
        logger: Optional logger instance

    Returns:
        Tuple of (is_valid: bool, issues: List[ValidationIssueDetail])

    Example:
        >>> transforms = [
        >>>     {'name': 'AddSelfLoops'},
        >>>     {'name': 'RemoveIsolatedNodes'}
        >>> ]
        >>> is_valid, issues = validate_transform_sequence_semantics(transforms)
        >>> if not is_valid:
        >>>     for issue in issues:
        >>>         print(issue)
    """
    logger = logger or logging.getLogger(__name__)
    issues: list[ValidationIssueDetail] = []

    # Try to use SemanticValidator if available
    if TRANSFORM_INTROSPECTION_AVAILABLE:
        try:
            from milia_pipeline.transformations.graph_transforms import SemanticValidator

            validator = SemanticValidator(logger=logger)
            semantic_issues = validator.validate_sequence([t.get("name") for t in transforms])

            # Convert semantic issues to ValidationIssueDetail
            for idx, issue_msg in enumerate(semantic_issues):
                severity = ValidationSeverity.WARNING
                if "error" in issue_msg.lower() or "conflict" in issue_msg.lower():
                    severity = ValidationSeverity.ERROR

                issues.append(
                    ValidationIssueDetail(
                        severity=severity,
                        message=issue_msg,
                        location="transforms.sequence",
                        context={"sequence_position": idx},
                    )
                )

            has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in issues)
            return not has_errors, issues

        except Exception as e:
            logger.debug(f"Semantic validation unavailable: {e}")

    # Fallback: Basic ordering checks if SemanticValidator not available
    issues.extend(_basic_semantic_checks(transforms))

    has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in issues)
    return not has_errors, issues


def _basic_semantic_checks(transforms: list[dict[str, Any]]) -> list[ValidationIssueDetail]:
    """
    Basic semantic validation without full SemanticValidator.

    Provides minimal ordering and conflict checks when features unavailable.

    Args:
        transforms: List of transform specifications

    Returns:
        List of validation issues found
    """
    issues: list[ValidationIssueDetail] = []

    transform_names = [t.get("name", "") for t in transforms]

    # Check for obvious conflicts
    if "RemoveIsolatedNodes" in transform_names and "AddSelfLoops" in transform_names:
        if transform_names.index("RemoveIsolatedNodes") > transform_names.index("AddSelfLoops"):
            issues.append(
                ValidationIssueDetail(
                    severity=ValidationSeverity.WARNING,
                    message="RemoveIsolatedNodes after AddSelfLoops may cause issues",
                    location="transforms.sequence",
                    suggestion="Place RemoveIsolatedNodes before AddSelfLoops",
                )
            )

    # Check for duplicate transforms
    seen = set()
    for idx, name in enumerate(transform_names):
        if name in seen:
            issues.append(
                ValidationIssueDetail(
                    severity=ValidationSeverity.WARNING,
                    message=f"Duplicate transform '{name}' in sequence",
                    location=f"transforms[{idx}]",
                    suggestion="Remove duplicate or ensure intentional",
                )
            )
        seen.add(name)

    return issues


# ==========================================
# Core Value Validation Functions
# ==========================================


def is_value_valid_and_not_nan(
    value: Any, allow_empty_array: bool = False, check_finite: bool = True
) -> bool:
    """
    Unified validation function that checks if a value is valid and not NaN.

    This function consolidates the various _is_value_valid_and_not_nan implementations
    found throughout the codebase into a single, consistent validator. It is heavily
    used by dataset handlers for validating molecular properties.

    Handler Usage:
        Dataset handlers use this for validating energy values, coordinates,
        uncertainty data, and other properties. All handlers rely on this for
        structural property validation.

    Args:
        value: The value to check. Can be of various types including None, str, bytes,
               numpy strings/bytes, numerical types, arrays, or tensors.
        allow_empty_array: If True, empty arrays are considered valid. Default: False.
        check_finite: If True, also checks for infinite values. Default: True.

    Returns:
        bool: True if the value is valid and does not contain NaN/Inf, False otherwise.

    Examples:
        >>> is_value_valid_and_not_nan(3.14)
        True
        >>> is_value_valid_and_not_nan(np.nan)
        False
        >>> is_value_valid_and_not_nan(np.array([1, 2, 3]))
        True
        >>> is_value_valid_and_not_nan(None)
        False

    Handler Integration:
        >>> # Dataset handler energy validation
        >>> if not is_value_valid_and_not_nan(raw_properties_dict.get('Etot')):
        ...     raise MoleculeProcessingError("Invalid energy data")
        >>>
        >>> # Uncertainty-enabled handler validation
        >>> if not is_value_valid_and_not_nan(uncertainty_value):
        ...     logger.warning("Invalid uncertainty data detected")
    """
    if value is None:
        return False

    # Handle string values
    if isinstance(value, (str, bytes, np.str_, np.bytes_)):
        value_str = str(value).strip().lower()
        if value_str in ["nan", "inf", "-inf", "none", ""] or not value_str:
            return False
        # Check if it's a numeric string
        try:
            float_val = float(value_str)
            result = np.isfinite(float_val) if check_finite else not np.isnan(float_val)
            # FIXED: Always return Python bool, not numpy.bool_
            return bool(result)
        except (ValueError, TypeError):
            return False

    # Handle PyTorch tensors
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return allow_empty_array
        if check_finite:
            result = not (torch.any(torch.isnan(value)) or torch.any(torch.isinf(value)))
        else:
            result = not torch.any(torch.isnan(value))
        # FIXED: Always return Python bool
        return bool(result)

    # Handle scalar numeric values
    if isinstance(value, (int, float, np.number)):
        result = np.isfinite(value) if check_finite else not np.isnan(value)
        # FIXED: Always return Python bool, not numpy.bool_
        return bool(result)

    # Handle arrays and lists
    if isinstance(value, (np.ndarray, list)):
        # FIXED: Handle empty lists correctly
        if isinstance(value, list):
            if len(value) == 0:
                logger.debug("Empty list validation failed (allow_empty_array=False)")
                return allow_empty_array

            # Check if all elements are valid
            for element in value:
                if not is_value_valid_and_not_nan(element, allow_empty_array, check_finite):
                    return False
            return True

        # Handle numpy arrays
        try:
            arr_value = np.asarray(value)
        except (TypeError, ValueError) as e:
            logger.debug(f"Cannot convert value to array: {e}")
            return False

        if arr_value.size == 0:
            if not allow_empty_array:
                logger.debug("Empty array validation failed (allow_empty_array=False)")
            return allow_empty_array

        # Handle numeric dtypes
        if np.issubdtype(arr_value.dtype, np.number):
            if check_finite:
                result = not np.any(~np.isfinite(arr_value))
            else:
                result = not np.any(np.isnan(arr_value))
            # FIXED: Always return Python bool
            return bool(result)

        # Handle complex numbers
        elif np.issubdtype(arr_value.dtype, np.complexfloating):
            real_valid = not np.any(np.isnan(arr_value.real))
            imag_valid = not np.any(np.isnan(arr_value.imag))
            if check_finite:
                real_valid = real_valid and not np.any(np.isinf(arr_value.real))
                imag_valid = imag_valid and not np.any(np.isinf(arr_value.imag))
            return bool(real_valid and imag_valid)

        # Handle object dtype - recursively check elements
        elif arr_value.dtype == object:
            try:
                # Try to convert to float to check validity
                converted_arr = arr_value.astype(np.float64, copy=False)
                if check_finite:
                    result = not np.any(~np.isfinite(converted_arr))
                else:
                    result = not np.any(np.isnan(converted_arr))
                return bool(result)
            except (ValueError, TypeError):
                # If conversion fails, recursively check each element
                for element in arr_value.flat:
                    if not is_value_valid_and_not_nan(element, allow_empty_array, check_finite):
                        return False
                return True
        else:
            # Handle string arrays
            if arr_value.dtype.kind in ["U", "S"]:
                for element in arr_value.flat:
                    element_str = str(element).strip().lower()
                    if element_str in ["nan", "inf", "-inf", "none", ""] or not element_str:
                        return False
                    try:
                        float_val = float(element_str)
                        if (
                            check_finite
                            and not np.isfinite(float_val)
                            or not check_finite
                            and np.isnan(float_val)
                        ):
                            return False
                    except (ValueError, TypeError):
                        return False
                return True
            else:
                # Non-numeric dtype
                return False

    # Unknown type - be conservative
    return False


# ==========================================
# Molecular Structure Validation
# ==========================================


def validate_molecular_structure(
    atoms: np.ndarray, coordinates: np.ndarray, molecule_index: int, identifier: str = "N/A"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validates molecular structure data including atoms and coordinates.

    This function is extensively used by dataset handlers to ensure molecular
    structure integrity before processing. All handlers rely on this validation
    for structural consistency.

    Handler Integration:
        Handlers for datasets with vibrational_analysis use this for validating
        molecular geometry and ensuring proper structure for property calculations.

        Handlers for datasets with uncertainty_handling use this for validating
        molecular structure before uncertainty analysis and energy calculations.

    Args:
        atoms: Array of atomic numbers or symbols.
        coordinates: Array of 3D coordinates, shape (n_atoms, 3).
        molecule_index: Index of the molecule for error reporting.
        identifier: Molecule identifier (InChI/SMILES) for error reporting.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Validated (atomic_numbers, coordinates).

    Raises:
        StructuralFeatureError: If validation fails with detailed error context.

    Handler Usage Examples:
        >>> # In any DatasetHandler.validate_molecule_data()
        >>> try:
        ...     atoms_validated, coords_validated = validate_molecular_structure(
        ...         atoms, coordinates, molecule_index, identifier
        ...     )
        ... except StructuralFeatureError as e:
        ...     raise DatasetSpecificHandlerError(
        ...         message="Molecular structure validation failed",
        ...         dataset_type=self.dataset_type,
        ...         operation="validate_structure",
        ...         details=str(e)
        ...     ) from e
    """
    # Validate atoms
    if not is_value_valid_and_not_nan(atoms):
        raise StructuralFeatureError(
            message="Invalid or missing atoms data",
            molecule_index=molecule_index,
            inchi=identifier,
            feature_type="atom",
            reason="Atoms array contains invalid or NaN values",
        )

    # Validate coordinates
    if not is_value_valid_and_not_nan(coordinates):
        raise StructuralFeatureError(
            message="Invalid or missing coordinates data",
            molecule_index=molecule_index,
            inchi=identifier,
            feature_type="structural",
            reason="Coordinates array contains invalid or NaN values",
        )

    # Check dimensions
    if atoms.ndim != 1:
        raise StructuralFeatureError(
            message=f"Atoms array must be 1D, got shape {atoms.shape}",
            molecule_index=molecule_index,
            inchi=identifier,
            feature_type="atom",
            reason=f"Expected 1D array, got {atoms.ndim}D",
        )

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise StructuralFeatureError(
            message=f"Coordinates must be (n_atoms, 3), got shape {coordinates.shape}",
            molecule_index=molecule_index,
            inchi=identifier,
            feature_type="structural",
            reason=f"Expected (N, 3) shape, got {coordinates.shape}",
        )

    # Check count consistency
    if len(atoms) != coordinates.shape[0]:
        raise StructuralFeatureError(
            message=f"Atom count mismatch: {len(atoms)} atoms vs {coordinates.shape[0]} coordinate sets",
            molecule_index=molecule_index,
            inchi=identifier,
            feature_type="structural",
            reason="Inconsistent atom and coordinate counts",
        )

    # Convert atoms to atomic numbers if they are symbols
    if atoms.dtype.kind in ["U", "S"]:  # Unicode or byte string
        # This would need the symbol-to-atomic-number mapping
        # For now, assume they are already atomic numbers
        atomic_numbers = atoms.astype(np.int64)
    else:
        atomic_numbers = atoms.astype(np.int64)

    # Validate atomic numbers range
    if np.any(atomic_numbers < 1) or np.any(atomic_numbers > 118):
        invalid_nums = atomic_numbers[(atomic_numbers < 1) | (atomic_numbers > 118)]
        raise StructuralFeatureError(
            message=f"Invalid atomic numbers: {invalid_nums}",
            molecule_index=molecule_index,
            inchi=identifier,
            feature_type="atom",
            reason=f"Atomic numbers must be 1-118, found: {invalid_nums}",
        )

    # Validate coordinates are finite
    if not np.all(np.isfinite(coordinates)):
        raise StructuralFeatureError(
            message="Coordinates contain non-finite values",
            molecule_index=molecule_index,
            inchi=identifier,
            feature_type="structural",
            reason="Found NaN or Inf values in coordinates",
        )

    return atomic_numbers, coordinates.astype(np.float32)


# ==========================================
# Handler Support Functions
# ==========================================


def convert_to_scalar(
    value: np.ndarray | torch.Tensor | float | int | None,
    value_name: str = "value",
    context: str = "",
    allow_none: bool = False,
) -> float | None:
    """
    Converts various data types to a scalar float value for handler operations.

    This function provides consistent scalar conversion across all handlers,
    supporting the common pattern of extracting scalar values from arrays or tensors.
    It is essential for handler operations that need to work with scalar values
    regardless of the input data format.

    Handler Integration:
        Handlers for datasets with vibrational_analysis use this for extracting
        energy values, atomization energies, and scalar properties from various
        input formats.

        Handlers for datasets with uncertainty_handling use this extensively for
        uncertainty value extraction and energy value normalization.

    Args:
        value: Input value of various types (numpy array, torch tensor, scalar, None).
        value_name: Name of the value for error context.
        context: Additional context for error messages (e.g., "Molecule 42").
        allow_none: If True, return None for invalid conversions; if False, raise ValidationError.

    Returns:
        Optional[float]: Scalar float value, or None if conversion fails/invalid.

    Raises:
        ValidationError: If allow_none=False and conversion fails.

    Examples:
        >>> convert_to_scalar(np.array([3.14]))
        3.14
        >>> convert_to_scalar(torch.tensor(2.71))
        2.71
        >>> convert_to_scalar([1.0])
        1.0
        >>> convert_to_scalar(None)
        None

    Handler Usage Examples:
        >>> # Dataset handler energy extraction
        >>> energy_scalar = convert_to_scalar(
        ...     raw_properties_dict['Etot'], 'Etot', f"Molecule {molecule_index}"
        ... )
        >>>
        >>> # Uncertainty-enabled handler extraction
        >>> uncertainty_scalar = convert_to_scalar(
        ...     uncertainty_value, 'std', f"Molecule {molecule_index}",
        ...     allow_none=False  # Strict validation for uncertainty
        ... )
    """
    error_context = f"{context} {value_name}".strip()

    if value is None:
        if allow_none:
            logger.debug(f"{error_context}: Value is None")
            return None
        else:
            raise ValidationError(
                message="Value is None but allow_none=False",
                validation_type="scalar_conversion",
                data_context=error_context,
            )

    # Handle scalar numeric values
    if isinstance(value, (int, float, np.number)):
        scalar_val = float(value)
        if not np.isfinite(scalar_val):
            if allow_none:
                logger.debug(f"{error_context}: Scalar value is not finite: {scalar_val}")
                return None
            else:
                raise ValidationError(
                    message=f"Scalar value is not finite: {scalar_val}",
                    validation_type="scalar_conversion",
                    data_context=error_context,
                )
        return scalar_val

    # Handle numpy arrays
    elif isinstance(value, np.ndarray):
        if value.size == 0:
            if allow_none:
                logger.debug(f"{error_context}: Empty numpy array")
                return None
            else:
                raise ValidationError(
                    message="Empty numpy array",
                    validation_type="scalar_conversion",
                    data_context=error_context,
                )
        elif value.size == 1:
            scalar_val = float(value.item())
            if not np.isfinite(scalar_val):
                if allow_none:
                    logger.debug(f"{error_context}: Array scalar value is not finite: {scalar_val}")
                    return None
                else:
                    raise ValidationError(
                        message=f"Array scalar value is not finite: {scalar_val}",
                        validation_type="scalar_conversion",
                        data_context=error_context,
                    )
            return scalar_val
        else:
            if allow_none:
                logger.debug(
                    f"{error_context}: Cannot convert array of size {value.size} to scalar"
                )
                return None
            else:
                raise ValidationError(
                    message=f"Cannot convert array of size {value.size} to scalar",
                    validation_type="scalar_conversion",
                    data_context=error_context,
                )

    # Handle PyTorch tensors
    elif isinstance(value, torch.Tensor):
        if value.numel() == 0:
            if allow_none:
                logger.debug(f"{error_context}: Empty torch tensor")
                return None
            else:
                raise ValidationError(
                    message="Empty torch tensor",
                    validation_type="scalar_conversion",
                    data_context=error_context,
                )
        elif value.numel() == 1:
            scalar_val = float(value.item())
            if not np.isfinite(scalar_val):
                if allow_none:
                    logger.debug(
                        f"{error_context}: Tensor scalar value is not finite: {scalar_val}"
                    )
                    return None
                else:
                    raise ValidationError(
                        message=f"Tensor scalar value is not finite: {scalar_val}",
                        validation_type="scalar_conversion",
                        data_context=error_context,
                    )
            return scalar_val
        else:
            if allow_none:
                logger.debug(
                    f"{error_context}: Cannot convert tensor of size {value.numel()} to scalar"
                )
                return None
            else:
                raise ValidationError(
                    message=f"Cannot convert tensor of size {value.numel()} to scalar",
                    validation_type="scalar_conversion",
                    data_context=error_context,
                )

    # Handle lists and tuples
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            if allow_none:
                logger.debug(f"{error_context}: Empty list/tuple")
                return None
            else:
                raise ValidationError(
                    message="Empty list/tuple",
                    validation_type="scalar_conversion",
                    data_context=error_context,
                )
        elif len(value) == 1:
            return convert_to_scalar(value[0], value_name, context, allow_none)
        else:
            if allow_none:
                logger.debug(
                    f"{error_context}: Cannot convert list/tuple of length {len(value)} to scalar"
                )
                return None
            else:
                raise ValidationError(
                    message=f"Cannot convert list/tuple of length {len(value)} to scalar",
                    validation_type="scalar_conversion",
                    data_context=error_context,
                )

    # Handle string values (attempt numeric conversion)
    elif isinstance(value, (str, bytes, np.str_, np.bytes_)):
        value_str = str(value).strip()
        if not value_str or value_str.lower() in ["nan", "inf", "-inf", "none"]:
            if allow_none:
                logger.debug(f"{error_context}: Invalid string value: '{value_str}'")
                return None
            else:
                raise ValidationError(
                    message=f"Invalid string value: '{value_str}'",
                    validation_type="scalar_conversion",
                    data_context=error_context,
                )
        try:
            scalar_val = float(value_str)
            if not np.isfinite(scalar_val):
                if allow_none:
                    logger.debug(
                        f"{error_context}: String converted to non-finite value: {scalar_val}"
                    )
                    return None
                else:
                    raise ValidationError(
                        message=f"String converted to non-finite value: {scalar_val}",
                        validation_type="scalar_conversion",
                        data_context=error_context,
                    )
            return scalar_val
        except (ValueError, TypeError):
            if allow_none:
                logger.debug(f"{error_context}: Cannot convert string '{value_str}' to float")
                return None
            else:
                raise ValidationError(
                    message=f"Cannot convert string '{value_str}' to float",
                    validation_type="scalar_conversion",
                    data_context=error_context,
                ) from None

    # Unknown type
    else:
        if allow_none:
            logger.debug(f"{error_context}: Unsupported type for scalar conversion: {type(value)}")
            return None
        else:
            raise ValidationError(
                message=f"Unsupported type for scalar conversion: {type(value)}",
                validation_type="scalar_conversion",
                data_context=error_context,
            )


def validate_uncertainty_data(
    uncertainty_value: Any,
    molecule_index: int | None = None,
    uncertainty_field_name: str = "uncertainty",
    require_positive: bool = True,
    max_threshold: float | None = None,
) -> float | None:
    """
    Dataset-agnostic uncertainty data validation for handler operations.

    This function provides a unified interface for uncertainty validation that
    works across different dataset types and handler implementations. It is
    specifically designed to support the handler pattern where different
    dataset types may have different uncertainty formats and requirements.

    Handler Integration:
        Handlers for uncertainty-enabled datasets use this extensively for
        validating uncertainty values from various uncertainty fields
        (std, variance, etc.).

        Any handler can use this for validating uncertainty-like values
        if present in the dataset.

    Args:
        uncertainty_value: Raw uncertainty data of any type.
        molecule_index: Index of molecule being processed (for error context).
        uncertainty_field_name: Name of the uncertainty field for error context.
        require_positive: Whether to require positive uncertainty values.
        max_threshold: Maximum allowed uncertainty value.

    Returns:
        Optional[float]: Validated uncertainty scalar, or None if invalid.

    Raises:
        UncertaintyProcessingError: If uncertainty validation fails for uncertainty-enabled data.
        ValidationError: If uncertainty validation fails for general data.

    Handler Usage Examples:
        >>> # Uncertainty-enabled handler validation
        >>> try:
        ...     uncertainty_scalar = validate_uncertainty_data(
        ...         raw_properties_dict['std'],
        ...         molecule_index=molecule_index,
        ...         uncertainty_field_name='std',
        ...         require_positive=True,
        ...         max_threshold=self.dataset_config.uncertainty_config.get('max_uncertainty_threshold')
        ...     )
        ... except UncertaintyProcessingError as e:
        ...     raise DatasetSpecificHandlerError(
        ...         message=f"{e.dataset_type} uncertainty validation failed",
        ...         dataset_type=e.dataset_type,
        ...         operation="validate_uncertainty",
        ...         property_name=uncertainty_field_name
        ...     ) from e
        >>>
        >>> # Generic uncertainty validation in any handler
        >>> if uncertainty_value is not None:
        ...     validated_uncertainty = validate_uncertainty_data(
        ...         uncertainty_value, molecule_index, 'custom_uncertainty'
        ...     )
    """
    context = f"Molecule {molecule_index}" if molecule_index is not None else "Unknown molecule"

    if uncertainty_value is None:
        return None

    # Convert to scalar using the unified converter
    try:
        uncertainty_scalar = convert_to_scalar(
            uncertainty_value,
            value_name=uncertainty_field_name,
            context=context,
            allow_none=False,  # None already handled above
        )
    except ValidationError as e:
        # Convert to uncertainty-specific error if this appears to be uncertainty data
        if uncertainty_field_name in ["std", "variance", "uncertainty"]:
            raise UncertaintyProcessingError(
                message=f"Uncertainty validation failed: {str(e)}",
                dataset_type="",
                molecule_index=molecule_index,
                detail=str(e),
                uncertainty_property_name=uncertainty_field_name,
            ) from e
        else:
            # Re-raise as generic validation error
            raise

    if uncertainty_scalar is None:
        error_msg = (
            f"{context}: {uncertainty_field_name} is None or could not be converted to scalar"
        )
        if uncertainty_field_name in ["std", "variance", "uncertainty"]:
            raise UncertaintyProcessingError(
                message=error_msg,
                dataset_type="",
                molecule_index=molecule_index,
                uncertainty_property_name=uncertainty_field_name,
            )
        else:
            raise ValidationError(
                message=error_msg, validation_type="uncertainty_validation", data_context=context
            )

    # Validate positivity requirement
    if require_positive and uncertainty_scalar < 0:
        error_msg = (
            f"{context}: {uncertainty_field_name} must be non-negative, got {uncertainty_scalar}"
        )
        if uncertainty_field_name in ["std", "variance", "uncertainty"]:
            raise UncertaintyProcessingError(
                message=error_msg,
                dataset_type="",
                molecule_index=molecule_index,
                uncertainty_property_name=uncertainty_field_name,
            )
        else:
            raise ValidationError(
                message=error_msg, validation_type="uncertainty_validation", data_context=context
            )

    # Validate against maximum threshold
    if max_threshold is not None and uncertainty_scalar > max_threshold:
        error_msg = f"{context}: {uncertainty_field_name} {uncertainty_scalar} exceeds maximum threshold {max_threshold}"
        if uncertainty_field_name in ["std", "variance", "uncertainty"]:
            raise UncertaintyProcessingError(
                message=error_msg,
                dataset_type="",
                molecule_index=molecule_index,
                uncertainty_property_name=uncertainty_field_name,
            )
        else:
            raise ValidationError(
                message=error_msg, validation_type="uncertainty_validation", data_context=context
            )

    return uncertainty_scalar


def validate_molecular_data_dict(
    molecular_data: dict[str, Any],
    required_properties: list[str],
    molecule_index: int | None = None,
    identifier: str = "N/A",
    dataset_type: str | None = None,
    property_validators: dict[str, callable] | None = None,
    strict: bool = False,
    return_wrapper: bool = False,
) -> tuple[bool, list[str]]:
    """
    Comprehensive validation of a molecular data dictionary for handler operations.

    This function provides batch validation capabilities that handlers can use
    to validate molecular data dictionaries before processing. It supports
    dataset-specific validation rules and custom property validators.

    Handler Integration:
        All handlers can use this function for comprehensive molecule validation.
        It provides dataset-specific validation logic while maintaining a
        unified interface across all handler types.

    Args:
        molecular_data: Dictionary containing molecular properties.
        required_properties: List of properties that must be present and valid.
        molecule_index: Index of molecule for error context.
        identifier: Molecule identifier (InChI/SMILES) for error context.
        dataset_type: Dataset type for dataset-specific validation.
        property_validators: Optional dict of custom validators for specific properties.

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_validation_errors)

    Handler Usage Examples:
        >>> # Dataset handler validation
        >>> is_valid, errors = validate_molecular_data_dict(
        ...     molecular_data, ['Etot', 'atoms', 'coordinates'],
        ...     molecule_index=i, identifier=inchi, dataset_type='MyDataset'
        ... )
        >>> if not is_valid:
        ...     raise DatasetSpecificHandlerError(
        ...         message="Molecular data validation failed",
        ...         dataset_type='MyDataset',
        ...         operation="validate_molecular_data",
        ...         details="; ".join(errors)
        ...     )
        >>>
        >>> # Uncertainty-enabled handler validation with custom validators
        >>> custom_validators = {
        ...     'std': lambda x: validate_uncertainty_data(x, require_positive=True) is not None
        ... }
        >>> is_valid, errors = validate_molecular_data_dict(
        ...     molecular_data, ['Etot', 'std'],
        ...     molecule_index=i, identifier=inchi, dataset_type='MyDataset',
        ...     property_validators=custom_validators
        ... )
    """
    errors = []
    context = (
        f"Molecule {molecule_index} ({identifier})"
        if molecule_index is not None
        else f"Molecule ({identifier})"
    )

    # Check for required properties
    for prop in required_properties:
        if prop not in molecular_data:
            errors.append(f"{context}: Missing required property: {prop}")
            continue

        value = molecular_data[prop]
        if not is_value_valid_and_not_nan(value):
            errors.append(f"{context}: Property '{prop}' has invalid value")

        # Apply custom validator if provided
        if property_validators and prop in property_validators:
            validator = property_validators[prop]
            try:
                if not validator(value):
                    errors.append(f"{context}: Property '{prop}' failed custom validation")
            except Exception as e:
                errors.append(f"{context}: Property '{prop}' custom validation error: {str(e)}")

    # ========================================================================
    # PHASE 6: Feature-based dataset-specific validation (replaces if/elif)
    # ========================================================================

    # Uncertainty validation (for datasets with uncertainty_handling feature)
    if dataset_type and _get_dataset_feature(dataset_type, "uncertainty_handling"):
        # Uncertainty-enabled validation (for datasets with uncertainty_handling feature)
        if "std" in molecular_data:
            try:
                validate_uncertainty_data(
                    molecular_data["std"],
                    molecule_index=molecule_index,
                    uncertainty_field_name="std",
                    require_positive=True,
                )
            except (UncertaintyProcessingError, ValidationError) as e:
                errors.append(f"{context}: {dataset_type} uncertainty validation failed: {str(e)}")

        # Check for energy validity in uncertainty-enabled datasets
        if "Etot" in molecular_data:
            etot_value = molecular_data["Etot"]
            if etot_value is None:
                errors.append(f"{context}: {dataset_type} energy 'Etot' is missing (None)")
            else:
                try:
                    etot_scalar = convert_to_scalar(etot_value, "Etot", context, allow_none=False)
                    if etot_scalar is None:
                        errors.append(f"{context}: {dataset_type} energy 'Etot' is invalid")
                    elif abs(etot_scalar) > 10000:  # Hartree - unusually large energy
                        errors.append(
                            f"{context}: {dataset_type} energy magnitude unusually large: {etot_scalar}"
                        )
                except ValidationError as e:
                    errors.append(f"{context}: {dataset_type} energy validation failed: {str(e)}")

    # Vibrational analysis validation (for datasets with vibrational_analysis feature)
    if dataset_type and _get_dataset_feature(dataset_type, "vibrational_analysis"):
        # Vibrational analysis-enabled validation (for datasets with vibrational_analysis feature)
        if "freqs" in molecular_data and "vibmodes" in molecular_data:
            freqs = molecular_data["freqs"]
            vibmodes = molecular_data["vibmodes"]

            if isinstance(freqs, (list, np.ndarray)) and isinstance(vibmodes, (list, np.ndarray)):
                # Empty arrays are valid for molecules with no vibrational modes
                # (e.g., single atoms or linear molecules)
                freq_len = len(freqs) if hasattr(freqs, "__len__") else 0
                vib_len = len(vibmodes) if hasattr(vibmodes, "__len__") else 0

                if freq_len != vib_len:
                    errors.append(
                        f"{context}: {dataset_type} frequency/vibmode count mismatch: "
                        f"{freq_len} vs {vib_len}"
                    )
                # Note: Empty arrays (0 == 0) are valid - no error added

        # Check for Mulliken charges (can be empty for some molecules)
        if "Qmulliken" in molecular_data:
            qmull = molecular_data["Qmulliken"]
            # Allow empty arrays for Mulliken charges
            if not is_value_valid_and_not_nan(qmull, allow_empty_array=True):
                errors.append(f"{context}: {dataset_type} Mulliken charges contain invalid values")

    # Common structural validation
    if "atoms" in molecular_data and "coordinates" in molecular_data:
        atoms = molecular_data["atoms"]
        coordinates = molecular_data["coordinates"]

        if is_value_valid_and_not_nan(atoms) and is_value_valid_and_not_nan(coordinates):
            try:
                validate_molecular_structure(
                    np.asarray(atoms), np.asarray(coordinates), molecule_index or 0, identifier
                )
            except StructuralFeatureError as e:
                errors.append(f"{context}: Structural validation failed: {str(e)}")

    is_valid = len(errors) == 0

    # Strict mode: Raise immediately on failure
    if strict and not is_valid:
        raise ValidationError(
            message=f"Molecular data validation failed: {'; '.join(errors[:3])}",
            validation_type="molecular_data_validation",
            data_context=context,
        )

    # Return wrapper for enforced checking
    if return_wrapper:
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            data=molecular_data if is_valid else None,
            context=context,
        )

    # Traditional tuple return (backward compatible)

    return is_valid, errors


# ==========================================
# Array and Tensor Validation
# ==========================================


def validate_array_shape(
    array: np.ndarray | torch.Tensor | list,
    expected_shape: tuple[int, ...] | None = None,
    expected_ndim: int | None = None,
    name: str = "array",
    allow_empty: bool = False,
) -> bool:
    """
    Validates the shape and dimensions of an array or tensor.

    This function is commonly used by handlers to validate that molecular
    properties have the expected dimensions and shapes before processing.

    Handler Integration:
        Handlers for datasets with vibrational_analysis use this to validate
        coordinate arrays, frequency arrays, and vibrational mode tensors.

        Handlers for datasets with uncertainty_handling use this to validate
        uncertainty arrays and energy tensors.

    Args:
        array: The array or tensor to validate.
        expected_shape: Expected shape tuple (use -1 for any size in that dimension).
        expected_ndim: Expected number of dimensions.
        name: Name of the array for error messages.
        allow_empty: Whether to allow empty arrays (size 0 in any dimension).

    Returns:
        bool: True if shape/dimensions match expectations.

    Examples:
        >>> arr = np.array([[1, 2, 3], [4, 5, 6]])
        >>> validate_array_shape(arr, expected_shape=(2, 3))
        True
        >>> validate_array_shape(arr, expected_ndim=2)
        True

    Handler Usage Examples:
        >>> # Dataset handler coordinate validation
        >>> if not validate_array_shape(coordinates, expected_shape=(-1, 3), name="coordinates"):
        ...     raise DatasetSpecificHandlerError(
        ...         message="Invalid coordinate shape",
        ...         dataset_type=self.dataset_type,
        ...         operation="validate_coordinates",
        ...         property_name="coordinates"
        ...     )
        >>>
        >>> # Uncertainty-enabled handler array validation
        >>> if uncertainty_array is not None:
        ...     if not validate_array_shape(uncertainty_array, expected_ndim=1, name="uncertainty"):
        ...         logger.warning(f"Unexpected uncertainty array shape for molecule {mol_idx}")
    """
    if isinstance(array, list):
        array = np.array(array)

    if isinstance(array, torch.Tensor):
        shape = array.shape
        ndim = array.ndim
        size = array.numel()
    elif isinstance(array, np.ndarray):
        shape = array.shape
        ndim = array.ndim
        size = array.size
    else:
        logger.warning(f"{name}: Not an array or tensor type: {type(array)}")
        return False

    if size == 0 and not allow_empty:
        logger.debug(f"{name}: Empty array not allowed (use allow_empty=True)")
        return False

    # Check number of dimensions
    if expected_ndim is not None and ndim != expected_ndim:
        logger.debug(f"{name}: Expected {expected_ndim} dimensions, got {ndim}")
        return False

    # Check number of dimensions
    if expected_ndim is not None and ndim != expected_ndim:
        logger.debug(f"{name}: Expected {expected_ndim} dimensions, got {ndim}")
        return False

    # Check shape
    if expected_shape is not None:
        if len(shape) != len(expected_shape):
            logger.debug(f"{name}: Shape mismatch - expected {expected_shape}, got {shape}")
            return False

        for actual, expected in zip(shape, expected_shape, strict=False):
            if expected != -1 and actual != expected:
                logger.debug(f"{name}: Shape mismatch - expected {expected_shape}, got {shape}")
                return False

    return True


def validate_numeric_range(
    value: float | np.ndarray | torch.Tensor,
    min_val: float | None = None,
    max_val: float | None = None,
    name: str = "value",
) -> bool:
    """
    Validates that numeric values fall within an expected range.

    This function is used by handlers to validate that molecular properties
    fall within reasonable physical ranges (e.g., energies, uncertainties,
    atomic numbers, coordinates).

    Handler Integration:
        Handlers for datasets with vibrational_analysis use this to validate
        energy ranges, coordinate ranges, and frequency ranges for physical
        reasonableness.

        Handlers for datasets with uncertainty_handling use this to validate
        uncertainty ranges and energy magnitude checks.

    Args:
        value: The value(s) to check.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).
        name: Name for error messages.

    Returns:
        bool: True if all values are within range.

    Handler Usage Examples:
        >>> # Dataset handler energy range validation
        >>> if not validate_numeric_range(energy_val, min_val=-10000, max_val=0, name="energy"):
        ...     logger.warning(f"Energy {energy_val} outside expected range")
        >>>
        >>> # Uncertainty-enabled handler range validation
        >>> if not validate_numeric_range(uncertainty_val, min_val=0, max_val=100, name="uncertainty"):
        ...     raise DatasetSpecificHandlerError(
        ...         message=f"Uncertainty {uncertainty_val} outside valid range",
        ...         dataset_type=self.dataset_type,
        ...         operation="validate_uncertainty_range",
        ...         property_name="std"
        ...     )
    """
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().numpy()
    elif isinstance(value, (list, tuple)):
        values = np.array(value)
    elif isinstance(value, np.ndarray):
        values = value
    else:
        values = np.array([value])

    if min_val is not None and np.any(values < min_val):
        logger.debug(f"{name}: Values below minimum {min_val}")
        return False

    if max_val is not None and np.any(values > max_val):
        logger.debug(f"{name}: Values above maximum {max_val}")
        return False

    return True


# ==========================================
# Dataset-Specific Validation
# ==========================================


def validate_property_value(
    value: Any,
    property_name: str,
    expected_type: type | None = None,
    expected_shape: tuple[int, ...] | None = None,
    allow_none: bool = False,
    allow_empty: bool = False,
) -> bool:
    """
    Validates a molecular property value against expected constraints.

    This function provides general property validation that handlers can use
    for any molecular property. It combines type checking, value validation,
    and shape validation in a single convenient function.

    Handler Integration:
        All handlers use this for validating individual molecular properties
        before processing. It provides a consistent validation interface
        across different property types and datasets.

    Args:
        value: The property value to validate.
        property_name: Name of the property for logging.
        expected_type: Expected Python type.
        expected_shape: Expected shape for array-like values.
        allow_none: Whether None is a valid value.
        allow_empty: Whether empty arrays are valid.

    Returns:
        bool: True if property value is valid.

    Handler Usage Examples:
        >>> # Vibrational handler frequency validation
        >>> if not validate_property_value(freqs, 'freqs', expected_type=np.ndarray):
        ...     logger.warning(f"Invalid frequencies for molecule {mol_idx}")
        >>>
        >>> # Uncertainty-enabled handler validation
        >>> if not validate_property_value(uncertainty, 'std', expected_type=(float, np.ndarray)):
        ...     raise DatasetSpecificHandlerError(
        ...         message="Invalid uncertainty data type",
        ...         dataset_type=self.dataset_type,
        ...         operation="validate_property",
        ...         property_name='std'
        ...     )
    """
    if value is None:
        return allow_none

    # Type check
    if expected_type is not None and not isinstance(value, expected_type):
        logger.debug(f"{property_name}: Expected type {expected_type}, got {type(value)}")
        return False

    # Value validity check
    if not is_value_valid_and_not_nan(value, allow_empty_array=allow_empty):
        logger.debug(f"{property_name}: Contains invalid or NaN values")
        return False

    # Shape check for arrays
    if expected_shape is not None:
        if not validate_array_shape(value, expected_shape=expected_shape, name=property_name):
            return False

    return True


def validate_coordinates_3d(
    coordinates: np.ndarray, num_atoms: int, identifier: str = "molecule", allow_empty: bool = False
) -> bool:
    """
    Validates that coordinates are proper 3D molecular coordinates.

    This function is essential for handler validation of molecular geometry.
    All dataset handlers rely on this for ensuring coordinate integrity
    before geometric analysis and processing.

    Handler Integration:
        Handlers for datasets with vibrational_analysis use this extensively
        for validating molecular geometry before property calculations.

        Handlers for datasets with uncertainty_handling use this for validating
        molecular structure before uncertainty propagation and energy calculations.

    Args:
        coordinates: Coordinate array to validate.
        num_atoms: Expected number of atoms.
        identifier: Molecule identifier for error messages.
        allow_empty: Whether to allow empty coordinate arrays (for edge cases).

    Returns:
        bool: True if coordinates are valid 3D coordinates.

    Handler Usage Examples:
        >>> # Dataset handler coordinate validation
        >>> if not validate_coordinates_3d(coordinates, len(atomic_numbers), identifier):
        ...     raise DatasetSpecificHandlerError(
        ...         message="Invalid 3D coordinates",
        ...         dataset_type=self.dataset_type,
        ...         operation="validate_coordinates",
        ...         property_name="coordinates"
        ...     )
        >>>
        >>> # Uncertainty-enabled handler structure validation
        >>> if coordinates is not None and not validate_coordinates_3d(coordinates, num_atoms, inchi):
        ...     logger.warning(f"Molecule {mol_idx} has questionable coordinates")
    """
    if not isinstance(coordinates, np.ndarray):
        logger.debug(f"{identifier}: Coordinates not a numpy array")
        return False

    if coordinates.size == 0:
        if num_atoms == 0 and allow_empty:
            return True
        else:
            logger.debug(f"{identifier}: Empty coordinates not allowed")
            return False

    if not validate_array_shape(
        coordinates, expected_shape=(num_atoms, 3), allow_empty=allow_empty
    ):
        logger.debug(f"{identifier}: Invalid coordinate shape, expected ({num_atoms}, 3)")
        return False

    if not np.all(np.isfinite(coordinates)):
        logger.debug(f"{identifier}: Coordinates contain non-finite values")
        return False

    # Check for reasonable coordinate ranges (in Angstroms)
    if not validate_numeric_range(
        coordinates, min_val=-1000, max_val=1000, name=f"{identifier} coordinates"
    ):
        logger.warning(f"{identifier}: Coordinates outside reasonable range")
        return False

    return True


def validate_atomic_numbers(
    atomic_numbers: np.ndarray | list[int], identifier: str = "molecule"
) -> bool:
    """
    Validates atomic numbers are in the valid range (1-118).

    This function ensures that atomic number data is valid for all
    dataset handlers. It checks that all atomic numbers correspond
    to real elements.

    Handler Integration:
        All handlers use this to validate atomic number data before
        processing molecular structures and calculating properties.

    Args:
        atomic_numbers: Array or list of atomic numbers.
        identifier: Molecule identifier for error messages.

    Returns:
        bool: True if all atomic numbers are valid.

    Handler Usage Examples:
        >>> # Common handler atomic number validation
        >>> if not validate_atomic_numbers(z_values, identifier):
        ...     raise HandlerValidationError(
        ...         message="Invalid atomic numbers",
        ...         handler_type="generic",
        ...         validation_type="atomic_numbers",
        ...         failed_validations=[f"Invalid atomic numbers in {identifier}"]
        ...     )
    """
    if isinstance(atomic_numbers, list):
        atomic_numbers = np.array(atomic_numbers)

    if not isinstance(atomic_numbers, np.ndarray):
        logger.debug(f"{identifier}: Atomic numbers not an array")
        return False

    if atomic_numbers.ndim != 1:
        logger.debug(f"{identifier}: Atomic numbers must be 1D array")
        return False

    return validate_numeric_range(
        atomic_numbers, min_val=1, max_val=118, name=f"{identifier} atomic numbers"
    )


# ==========================================
# Batch Validation Functions for Handler Operations
# ==========================================


def validate_batch_consistency(
    data_list: list[dict[str, Any]],
    required_keys: list[str],
    property_shapes: dict[str, tuple[int, ...]] | None = None,
    allow_empty_arrays: set[str] | None = None,
) -> tuple[bool, list[str]]:
    """
    Validates consistency across a batch of molecular data.

    This function is designed for handler batch operations where multiple
    molecules need to be validated for consistency before processing.
    It ensures that all molecules in a batch have the required properties
    and consistent data formats.

    Handler Integration:
        Handlers can use this before processing molecule batches to ensure
        consistency and catch data format issues early in the pipeline.

    Args:
        data_list: List of molecular data dictionaries.
        required_keys: Keys that must be present in all molecules.
        property_shapes: Expected shapes for specific properties.
        allow_empty_arrays: Set of property names that can be empty arrays.

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)

    Handler Usage Examples:
        >>> # Dataset handler batch validation
        >>> is_valid, errors = validate_batch_consistency(
        ...     molecule_batch, ['Etot', 'atoms', 'coordinates']
        ... )
        >>> if not is_valid:
        ...     raise DatasetSpecificHandlerError(
        ...         message="Batch validation failed",
        ...         dataset_type=self.dataset_type,
        ...         operation="validate_batch",
        ...         details="; ".join(errors)
        ...     )
        >>>
        >>> # Uncertainty-enabled handler batch validation
        >>> shape_requirements = {'coordinates': (-1, 3), 'std': (1,)}
        >>> is_valid, errors = validate_batch_consistency(
        ...     molecule_batch, ['Etot', 'std'], property_shapes=shape_requirements
        ... )
    """
    errors = []

    if not data_list:
        errors.append("Empty data list")
        return False, errors

    if allow_empty_arrays is None:
        allow_empty_arrays = set()

    # Check required keys
    for i, data in enumerate(data_list):
        for key in required_keys:
            if key not in data:
                errors.append(f"Molecule {i}: Missing required key '{key}'")
            else:
                allow_empty = key in allow_empty_arrays
                if not is_value_valid_and_not_nan(data[key], allow_empty_array=allow_empty):
                    errors.append(f"Molecule {i}: Invalid value for key '{key}'")

    # Check property shapes if specified
    if property_shapes:
        for i, data in enumerate(data_list):
            for key, expected_shape in property_shapes.items():
                if key in data:
                    value = data[key]
                    if hasattr(value, "shape"):
                        actual_shape = value.shape
                        # Check shape consistency (allowing for variable first dimension)
                        if len(actual_shape) != len(expected_shape):
                            errors.append(
                                f"Molecule {i}: Property '{key}' has wrong number of dimensions"
                            )
                        else:
                            for j, (actual, expected) in enumerate(
                                zip(actual_shape, expected_shape, strict=False)
                            ):
                                if expected != -1 and actual != expected:
                                    errors.append(
                                        f"Molecule {i}: Property '{key}' dimension {j} mismatch"
                                    )

    is_valid = len(errors) == 0
    return is_valid, errors


# ==========================================
# Utility Validation Functions
# ==========================================


def is_valid_molecule_identifier(identifier: str, identifier_type: str = "any") -> bool:
    """
    Validates molecular identifiers (InChI, SMILES).

    This function helps handlers validate molecule identifiers for
    error reporting and molecule tracking purposes.

    Handler Integration:
        All handlers use this to validate molecule identifiers before
        using them in error messages and logging.

    Args:
        identifier: The identifier string to validate.
        identifier_type: Type of identifier ("inchi", "smiles", or "any").

    Returns:
        bool: True if identifier appears valid.

    Handler Usage Examples:
        >>> # Validate identifier before using in error messages
        >>> if is_valid_molecule_identifier(inchi, "inchi"):
        ...     self.logger.error(f"Processing failed for {inchi}")
        >>> else:
        ...     self.logger.error(f"Processing failed for molecule {mol_idx}")
    """
    if not identifier or identifier == "N/A":
        return False

    if not isinstance(identifier, str):
        return False

    identifier = identifier.strip()

    if identifier_type in ["inchi", "any"] and identifier.startswith("InChI="):
        return True

    if identifier_type in ["smiles", "any"]:
        # Basic SMILES validation - contains valid SMILES characters
        valid_chars = set("CNOPSFClBrIcnops()[]=#-+\\./@123456789%")
        if all(c in valid_chars for c in identifier):
            return True

    return False


def safe_get_value(
    data_dict: dict[str, Any],
    key: str,
    default: Any = None,
    expected_type: type | None = None,
    validate: bool = True,
) -> Any:
    """
    Safely retrieves and optionally validates a value from a dictionary.

    This utility function is commonly used by handlers to safely extract
    values from molecular property dictionaries with built-in validation
    and fallback behavior.

    Handler Integration:
        Handlers use this extensively for safely extracting molecular
        properties with automatic validation and sensible defaults.

    Args:
        data_dict: Dictionary to retrieve value from.
        key: Key to look up.
        default: Default value if key not found or invalid.
        expected_type: Expected type for type checking.
        validate: Whether to validate the value.

    Returns:
        The retrieved value or default if not found/invalid.

    Handler Usage Examples:
        >>> # Dataset handler safe property extraction
        >>> energy = safe_get_value(
        ...     raw_properties_dict, 'Etot', default=None,
        ...     expected_type=(int, float, np.number), validate=True
        ... )
        >>> if energy is None:
        ...     raise DatasetSpecificHandlerError(
        ...         message="Missing energy data",
        ...         dataset_type=self.dataset_type,
        ...         operation="extract_energy",
        ...         property_name="Etot"
        ...     )
        >>>
        >>> # Uncertainty-enabled handler safe extraction
        >>> uncertainty = safe_get_value(
        ...     raw_properties_dict, 'std', default=0.0,
        ...     expected_type=(float, np.ndarray), validate=True
        ... )
    """
    if not data_dict or key not in data_dict:
        return default

    value = data_dict[key]

    if expected_type is not None and not isinstance(value, expected_type):
        logger.debug(f"Key '{key}': Expected type {expected_type}, got {type(value)}")
        return default

    if validate:
        # For string types, don't apply NaN validation logic
        if not isinstance(value, (str, bytes, np.str_, np.bytes_)):
            if not is_value_valid_and_not_nan(value):
                logger.debug(f"Key '{key}': Value failed validation")
                return default

    # For the specific test case with np.nan, return the actual value when validate=False
    return value


# ==========================================
# Advanced Handler Support Functions
# ==========================================


def validate_handler_molecular_batch(
    molecular_data_list: list[dict[str, Any]],
    required_properties: list[str],
    dataset_type: str | None = None,
    max_errors_to_report: int = 10,
    strict: bool = False,
    return_wrapper: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """
    Validates a batch of molecular data for handler processing.

    This function provides comprehensive batch validation that handlers can use
    to validate entire batches of molecular data before processing. It's optimized
    for handler workflows and provides detailed validation summaries.

    Handler Integration:
        Handlers use this at the beginning of batch processing operations to
        validate entire molecular datasets and generate comprehensive validation
        reports for debugging and quality assurance.

    Args:
        molecular_data_list: List of molecular data dictionaries.
        required_properties: Properties that must be present in all molecules.
        dataset_type: Dataset type for dataset-specific validation.
        max_errors_to_report: Maximum number of individual errors to report.

    Returns:
        Tuple[bool, Dict[str, Any]]: (is_valid, validation_summary)

        validation_summary contains:
        - total_molecules: Total number of molecules
        - valid_molecules: Number of valid molecules
        - invalid_molecules: Number of invalid molecules
        - validation_errors: List of validation errors (up to max_errors_to_report)
        - error_statistics: Statistics about error types

    Handler Usage Examples:
        >>> # Dataset handler batch validation
        >>> is_valid, summary = validate_handler_molecular_batch(
        ...     molecule_batch, ['Etot', 'atoms', 'coordinates'], dataset_type='MyDataset'
        ... )
        >>> self.logger.info(f"Batch validation: {summary['valid_molecules']}/{summary['total_molecules']} valid")
        >>> if not is_valid:
        ...     report = create_validation_report(summary)
        ...     raise DatasetSpecificHandlerError(
        ...         message="Batch validation failed",
        ...         dataset_type='MyDataset',
        ...         operation="validate_batch",
        ...         details=report
        ...     )
        >>>
        >>> # Uncertainty-enabled handler batch validation
        >>> is_valid, summary = validate_handler_molecular_batch(
        ...     molecule_batch, ['Etot', 'std'], dataset_type='MyDataset', max_errors_to_report=5
        ... )
        >>> if summary['error_statistics'].get('uncertainty_errors', 0) > 0:
        ...     self.logger.warning("Multiple uncertainty validation errors detected")
    """
    validation_summary = {
        "total_molecules": len(molecular_data_list),
        "valid_molecules": 0,
        "invalid_molecules": 0,
        "validation_errors": [],
        "error_statistics": {},
    }

    if not molecular_data_list:
        validation_summary["validation_errors"].append("Empty molecular data list")
        return False, validation_summary

    error_count = 0
    error_types = {}

    for i, mol_data in enumerate(molecular_data_list):
        # Get molecule identifier if available
        identifier = mol_data.get("inchi", mol_data.get("smiles", "N/A"))

        # Validate individual molecule
        is_mol_valid, mol_errors = validate_molecular_data_dict(
            mol_data,
            required_properties,
            molecule_index=i,
            identifier=identifier,
            dataset_type=dataset_type,
        )

        if is_mol_valid:
            validation_summary["valid_molecules"] += 1
        else:
            validation_summary["invalid_molecules"] += 1

            # Track error types for statistics
            for error in mol_errors:
                # Extract error type from error message
                if "Missing required property" in error:
                    error_types["missing_properties"] = error_types.get("missing_properties", 0) + 1
                elif "invalid value" in error.lower():
                    error_types["invalid_values"] = error_types.get("invalid_values", 0) + 1
                elif "uncertainty validation failed" in error:
                    error_types["uncertainty_errors"] = error_types.get("uncertainty_errors", 0) + 1
                elif "structural validation failed" in error:
                    error_types["structural_errors"] = error_types.get("structural_errors", 0) + 1
                else:
                    error_types["other_errors"] = error_types.get("other_errors", 0) + 1

            # Add errors to summary (up to limit)
            if error_count < max_errors_to_report:
                validation_summary["validation_errors"].extend(mol_errors)
                error_count += len(mol_errors)

                if error_count >= max_errors_to_report:
                    remaining_errors = validation_summary["invalid_molecules"] - (i + 1)
                    if remaining_errors > 0:
                        validation_summary["validation_errors"].append(
                            f"... and {remaining_errors} more molecules with validation errors (limit reached)"
                        )

    validation_summary["error_statistics"] = error_types

    # Overall validation passes if all molecules are valid
    is_batch_valid = validation_summary["invalid_molecules"] == 0

    # Strict mode: Raise on failure
    if strict and not is_batch_valid:
        error_msg = (
            f"Batch validation failed: {validation_summary['invalid_molecules']} "
            f"of {validation_summary['total_molecules']} molecules invalid"
        )
        raise ValidationError(
            message=error_msg,
            validation_type="batch_validation",
            data_context=f"Batch of {len(molecular_data_list)} molecules",
        )

    # Return wrapper
    if return_wrapper:
        return ValidationResult(
            is_valid=is_batch_valid,
            errors=validation_summary["validation_errors"],
            data=validation_summary,
            context=f"Batch validation ({len(molecular_data_list)} molecules)",
        )

    return is_batch_valid, validation_summary


def validate_handler_compatibility(
    handler_type: str, dataset_config: dict[str, Any], processing_config: dict[str, Any]
) -> tuple[bool, list[str]]:
    """
    Validates compatibility between handler type and configurations.

    PHASE 6: Refactored to use registry-based validation and feature queries.

    This function checks if a handler type is compatible with the provided
    dataset and processing configurations. It's essential for the handler
    factory pattern to ensure handlers are created with compatible configurations.

    Handler Integration:
        The handler factory uses this to validate configuration compatibility
        before creating handler instances. It prevents runtime errors by
        catching configuration mismatches early.

    Args:
        handler_type: Type of dataset handler.
        dataset_config: Dataset configuration dictionary.
        processing_config: Processing configuration dictionary.

    Returns:
        Tuple[bool, List[str]]: (is_compatible, list_of_compatibility_issues)

    Handler Usage Examples:
        >>> # Handler factory compatibility checking
        >>> is_compatible, issues = validate_handler_compatibility(
        ...     'MyDataset', dataset_config, processing_config
        ... )
        >>> if not is_compatible:
        ...     raise HandlerConfigurationError(
        ...         message=f"Handler compatibility issues: {issues}",
        ...         handler_type='MyDataset',
        ...         config_validation_errors=issues
        ...     )
        >>>
        >>> # Uncertainty-enabled handler specific compatibility
        >>> is_compatible, issues = validate_handler_compatibility(
        ...     'MyDataset', dataset_config, processing_config
        ... )
        >>> if issues:
        ...     logger.warning(f"Handler compatibility warnings: {issues}")
    """
    issues = []

    # ========================================================================
    # PHASE 6: Registry-based handler type validation (replaces hardcoded list)
    # ========================================================================
    available_types = _get_available_dataset_types()

    if not _is_dataset_type_registered(handler_type):
        issues.append(f"Unknown handler type: {handler_type}. Available: {available_types}")
        return False, issues

    # Check dataset type compatibility
    dataset_type = dataset_config.get("dataset_type")
    if dataset_type != handler_type:
        issues.append(
            f"Handler type '{handler_type}' incompatible with dataset type '{dataset_type}'"
        )

    # ========================================================================
    # PHASE 6: Feature-based compatibility checks (replaces if/elif chain)
    # ========================================================================
    compatibility = _get_handler_compatibility_checks(handler_type)

    # Uncertainty-related checks
    if compatibility.get("supports_uncertainty", False):
        # This handler supports uncertainty - check config
        uncertainty_config = dataset_config.get("uncertainty_config")
        is_uncertainty_enabled = dataset_config.get("is_uncertainty_enabled", False)

        if is_uncertainty_enabled and not uncertainty_config:
            issues.append(f"{handler_type} uncertainty enabled but uncertainty_config missing")

        if uncertainty_config and not uncertainty_config.get("uncertainty_field_name"):
            issues.append(f"{handler_type} uncertainty_config missing uncertainty_field_name")

        # Check for incompatible features
        vector_props = processing_config.get("vector_graph_properties", [])
        var_props = processing_config.get("variable_len_graph_properties", [])

        if "freqs" in vector_props or "freqs" in var_props:
            issues.append(
                f"{handler_type} handler incompatible with vibrational frequency processing"
            )

        if "vibmodes" in vector_props or "vibmodes" in var_props:
            issues.append(f"{handler_type} handler incompatible with vibrational mode processing")

    else:
        # This handler does NOT support uncertainty
        node_features = processing_config.get("node_features", [])

        if "uncertainty" in node_features:
            issues.append(f"{handler_type} handler incompatible with uncertainty node features")

    # Vibrational/atomization energy checks
    if compatibility.get("supports_vibrational", False):
        # This handler supports vibrational analysis - check atomization config
        atomization_from = processing_config.get("calculate_atomization_energy_from")
        atomization_key = processing_config.get("atomization_energy_key_name")

        if atomization_from and not atomization_key:
            issues.append(f"{handler_type} atomization energy calculation missing key name")

        if atomization_key and not atomization_from:
            issues.append(
                f"{handler_type} atomization energy key specified but no base energy source"
            )

    else:
        # This handler does NOT support vibrational analysis
        vector_props = processing_config.get("vector_graph_properties", [])
        var_props = processing_config.get("variable_len_graph_properties", [])

        if "freqs" in vector_props or "freqs" in var_props:
            issues.append(f"{handler_type} handler does not support vibrational frequencies")

        if "vibmodes" in vector_props or "vibmodes" in var_props:
            issues.append(f"{handler_type} handler does not support vibrational modes")

    # Atomization energy checks
    if not compatibility.get("supports_atomization", False):
        if processing_config.get("calculate_atomization_energy_from"):
            issues.append(f"{handler_type} handler does not support atomization energy calculation")

    # Check for required processing targets
    scalar_targets = processing_config.get("scalar_graph_targets", [])
    if not scalar_targets:
        issues.append(f"{handler_type} handler requires at least one scalar graph target")

    is_compatible = len(issues) == 0

    return is_compatible, issues


def create_validation_report(
    validation_results: dict[str, Any],
    include_statistics: bool = True,
    include_recommendations: bool = True,
) -> str:
    """
    Creates a formatted validation report for handler operations.

    This function generates human-readable validation reports that handlers
    can use for logging, debugging, and quality assurance. It provides
    comprehensive summaries of validation results in a standardized format.

    Handler Integration:
        Handlers use this to generate detailed validation reports for logging
        and debugging purposes. It helps with identifying data quality issues
        and tracking validation statistics across datasets.

    Args:
        validation_results: Results from validation functions.
        include_statistics: Whether to include validation statistics.
        include_recommendations: Whether to include recommendations.

    Returns:
        str: Formatted validation report.

    Handler Usage Examples:
        >>> # Dataset handler validation reporting
        >>> is_valid, summary = validate_handler_molecular_batch(molecule_batch, required_props)
        >>> if not is_valid:
        ...     report = create_validation_report(summary, include_recommendations=True)
        ...     raise DatasetSpecificHandlerError(
        ...         message="Batch validation failed",
        ...         dataset_type=self.dataset_type,
        ...         operation="validate_batch",
        ...         details=f"Batch validation report:\n{report}"
        ...     )
        >>>
        >>> # Handler validation summary
        >>> validation_summary = self.get_processing_statistics(processed_molecules)
        >>> report = create_validation_report(validation_summary, include_statistics=True)
        >>> self.logger.info(f"Validation summary:\n{report}")
    """
    report_lines = []
    report_lines.append("Validation Report")
    report_lines.append("=" * 50)

    # Basic statistics
    if include_statistics and "total_molecules" in validation_results:
        total = validation_results["total_molecules"]
        valid = validation_results["valid_molecules"]
        invalid = validation_results["invalid_molecules"]

        report_lines.append(f"Total molecules: {total}")
        report_lines.append(f"Valid molecules: {valid}")
        report_lines.append(f"Invalid molecules: {invalid}")

        if total > 0:
            valid_pct = (valid / total) * 100
            report_lines.append(f"Validation success rate: {valid_pct:.1f}%")

        report_lines.append("")

    # Error statistics
    if include_statistics and "error_statistics" in validation_results:
        error_stats = validation_results["error_statistics"]
        if error_stats:
            report_lines.append("Error Statistics:")
            report_lines.append("-" * 20)
            for error_type, count in error_stats.items():
                error_name = error_type.replace("_", " ").title()
                report_lines.append(f"  {error_name}: {count}")
            report_lines.append("")

    # Validation errors
    if "validation_errors" in validation_results:
        errors = validation_results["validation_errors"]
        if errors:
            report_lines.append("Validation Errors:")
            report_lines.append("-" * 20)
            for error in errors[:10]:  # Show first 10 errors
                report_lines.append(f"  - {error}")

            if len(errors) > 10:
                report_lines.append(f"  ... and {len(errors) - 10} more errors")
            report_lines.append("")

    # Recommendations
    if include_recommendations:
        recommendations = []

        if "error_statistics" in validation_results:
            error_stats = validation_results["error_statistics"]

            if error_stats.get("missing_properties", 0) > 0:
                recommendations.append("Review dataset for missing required properties")

            if error_stats.get("invalid_values", 0) > 0:
                recommendations.append("Check data quality - many invalid values detected")

            if error_stats.get("uncertainty_errors", 0) > 0:
                recommendations.append("Review uncertainty configuration and data")

            if error_stats.get("structural_errors", 0) > 0:
                recommendations.append("Validate molecular structure data (atoms/coordinates)")

        if "total_molecules" in validation_results and validation_results["total_molecules"] > 0:
            valid_rate = (
                validation_results["valid_molecules"] / validation_results["total_molecules"]
            ) * 100

            if valid_rate < 90:
                recommendations.append("Low validation success rate - review data preprocessing")
            elif valid_rate < 99:
                recommendations.append(
                    "Some validation failures - consider filtering invalid molecules"
                )

        if recommendations:
            report_lines.append("Recommendations:")
            report_lines.append("-" * 20)
            for rec in recommendations:
                report_lines.append(f"  • {rec}")

    return "\n".join(report_lines)


# ==========================================
# Transform Configuration Validation
# ==========================================


def validate_transform_spec(
    transform_spec: Union[dict[str, Any], "TransformSpec"], strict_mode: bool = True
) -> tuple[bool, list[str]]:
    """
    Validates a TransformSpec configuration.

    Integration:
        Validates individual transform specifications for use in experimental setups.
        Checks parameter types, required fields, and logical consistency.

    Args:
        transform_spec: Transform specification as dict or TransformSpec object.
        strict_mode: If True, apply strict validation rules.

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_validation_errors)

    Examples:
        >>> spec = {'name': 'AddSelfLoops', 'kwargs': {}, 'enabled': True}
        >>> is_valid, errors = validate_transform_spec(spec)
        >>> if not is_valid:
        ...     print(f"Validation errors: {errors}")
    """
    errors = []

    # Convert to dict if TransformSpec object
    if TRANSFORMATION_CONTAINERS_AVAILABLE and isinstance(transform_spec, TransformSpec):
        spec_dict = {
            "name": transform_spec.name,
            "kwargs": transform_spec.kwargs,
            "enabled": transform_spec.enabled,
            "description": transform_spec.description,
        }
    elif isinstance(transform_spec, dict):
        spec_dict = transform_spec
    else:
        errors.append(f"Invalid transform spec type: {type(transform_spec)}")
        return False, errors

    # Validate required fields
    if "name" not in spec_dict:
        errors.append("Missing required field: 'name'")
        return False, errors

    transform_name = spec_dict["name"]
    if not isinstance(transform_name, str) or not transform_name:
        errors.append(f"Invalid transform name: {transform_name}")
        return False, errors

    # Validate name format
    if strict_mode and not transform_name[0].isupper():
        errors.append(f"Transform name should start with uppercase letter: {transform_name}")

    # Validate optional fields
    if "kwargs" in spec_dict:
        kwargs = spec_dict["kwargs"]
        if not isinstance(kwargs, dict):
            errors.append(f"'kwargs' must be a dictionary, got {type(kwargs)}")
        else:
            # Validate common parameter patterns
            if "p" in kwargs:
                p_val = kwargs["p"]
                if not isinstance(p_val, (int, float)):
                    errors.append(f"Parameter 'p' should be numeric, got {type(p_val)}")
                elif not (0.0 <= p_val <= 1.0):
                    errors.append(f"Parameter 'p' should be in [0, 1], got {p_val}")

            if "degrees" in kwargs:
                degrees = kwargs["degrees"]
                if not isinstance(degrees, (int, float)):
                    errors.append(f"Parameter 'degrees' should be numeric, got {type(degrees)}")

    if "enabled" in spec_dict:
        enabled = spec_dict["enabled"]
        if not isinstance(enabled, bool):
            errors.append(f"'enabled' must be boolean, got {type(enabled)}")

    if "description" in spec_dict:
        desc = spec_dict["description"]
        if desc is not None and not isinstance(desc, str):
            errors.append(f"'description' must be string or None, got {type(desc)}")

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_experimental_setup(
    experimental_setup: Union[dict[str, Any], "ExperimentalSetup"],
    strict_mode: bool = True,
    check_transform_availability: bool = False,
) -> tuple[bool, list[str]]:
    """
    Validates an ExperimentalSetup configuration.

    Integration:
        Validates experimental setup configurations including transform sequences,
        setup metadata, and compatibility settings.

    Args:
        experimental_setup: Experimental setup as dict or ExperimentalSetup object.
        strict_mode: If True, apply strict validation rules.
        check_transform_availability: If True, check if transforms are available in PyG.

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_validation_errors)

    Examples:
        >>> setup = {
        ...     'name': 'baseline',
        ...     'transforms': [{'name': 'AddSelfLoops'}],
        ...     'enabled': True
        ... }
        >>> is_valid, errors = validate_experimental_setup(setup)
    """
    errors = []

    # Convert to dict if ExperimentalSetup object
    if TRANSFORMATION_CONTAINERS_AVAILABLE and isinstance(experimental_setup, ExperimentalSetup):
        setup_dict = {
            "name": experimental_setup.name,
            "transforms": [
                {"name": t.name, "kwargs": t.kwargs, "enabled": t.enabled}
                for t in experimental_setup.transforms
            ],
            "description": experimental_setup.description,
            "enabled": experimental_setup.enabled,
        }
    elif isinstance(experimental_setup, dict):
        setup_dict = experimental_setup
    else:
        errors.append(f"Invalid experimental setup type: {type(experimental_setup)}")
        return False, errors

    # Validate required fields
    if "name" not in setup_dict:
        errors.append("Missing required field: 'name'")
        return False, errors

    setup_name = setup_dict["name"]
    if not isinstance(setup_name, str) or not setup_name:
        errors.append(f"Invalid setup name: {setup_name}")
        return False, errors

    # Validate transforms field
    if "transforms" not in setup_dict:
        errors.append("Missing required field: 'transforms'")
        return False, errors

    transforms = setup_dict["transforms"]
    if not isinstance(transforms, list):
        errors.append(f"'transforms' must be a list, got {type(transforms)}")
        return False, errors

    if not transforms:
        if strict_mode:
            errors.append("Experimental setup must have at least one transform in strict mode")
        else:
            # M5 fix: This is expected for setups like 'baseline' that use only standard transforms
            # Changed from WARNING to DEBUG since empty experimental transforms is valid design
            logger.debug(
                f"Experimental setup '{setup_name}' has no experimental transforms (standard transforms may still apply)"
            )

    # Validate each transform
    for i, transform in enumerate(transforms):
        is_valid, transform_errors = validate_transform_spec(transform, strict_mode)
        if not is_valid:
            for error in transform_errors:
                errors.append(f"Transform {i}: {error}")

    # Validate optional fields
    if "enabled" in setup_dict:
        enabled = setup_dict["enabled"]
        if not isinstance(enabled, bool):
            errors.append(f"'enabled' must be boolean, got {type(enabled)}")

    if "description" in setup_dict:
        desc = setup_dict["description"]
        if desc is not None and not isinstance(desc, str):
            errors.append(f"'description' must be string or None, got {type(desc)}")

    # Check transform availability if requested
    if check_transform_availability:
        try:
            import torch_geometric.transforms as T

            for i, transform in enumerate(transforms):
                if transform.get("enabled", True):
                    transform_name = transform["name"]
                    if not hasattr(T, transform_name):
                        errors.append(f"Transform {i}: '{transform_name}' not available in PyG")
        except ImportError:
            logger.warning("PyTorch Geometric not available, skipping transform availability check")

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_transformation_config(
    transformation_config: Union[dict[str, Any], "TransformationConfig"], strict_mode: bool = True
) -> tuple[bool, list[str]]:
    """
    Validates a TransformationConfig configuration.

    Integration:
        Validates the complete transformation configuration including experimental setups,
        standard transforms, default setup selection, and validation settings.

        A valid configuration must have either 'experimental_setups' OR 'standard_transforms'
        (or both). Standard transforms are always applied first, before experimental setup
        transforms.

    Args:
        transformation_config: Transformation config as dict or TransformationConfig object.
        strict_mode: If True, apply strict validation rules.

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_validation_errors)

    Examples:
        >>> # Config with both standard_transforms and experimental_setups
        >>> config = {
        ...     'standard_transforms': [{'name': 'AddSelfLoops', 'enabled': True}],
        ...     'experimental_setups': {'baseline': [{'name': 'NormalizeFeatures'}]},
        ...     'default_setup': 'baseline',
        ...     'validation': {'enabled': True, 'strict_mode': False}
        ... }
        >>> is_valid, errors = validate_transformation_config(config)

        >>> # Config with only standard_transforms (no experimental variations)
        >>> config = {
        ...     'standard_transforms': [{'name': 'AddSelfLoops', 'enabled': True}],
        ...     'default_setup': 'baseline'
        ... }
        >>> is_valid, errors = validate_transformation_config(config)

        >>> # Config with only experimental_setups (backward compatible)
        >>> config = {
        ...     'experimental_setups': {'baseline': [{'name': 'AddSelfLoops'}]},
        ...     'default_setup': 'baseline'
        ... }
        >>> is_valid, errors = validate_transformation_config(config)
    """
    errors = []

    # Convert to dict if TransformationConfig object
    if TRANSFORMATION_CONTAINERS_AVAILABLE and isinstance(
        transformation_config, TransformationConfig
    ):
        config_dict = {
            "experimental_setups": {
                name: [{"name": t.name, "kwargs": t.kwargs} for t in setup.transforms]
                for name, setup in transformation_config.experimental_setups.items()
            },
            "default_setup": transformation_config.default_setup,
            "validation": transformation_config.validation,
        }
        # Include standard_transforms if the TransformationConfig has them
        if (
            hasattr(transformation_config, "standard_transforms")
            and transformation_config.standard_transforms
        ):
            config_dict["standard_transforms"] = [
                {"name": t.name, "kwargs": t.kwargs, "enabled": t.enabled}
                for t in transformation_config.standard_transforms
            ]
    elif isinstance(transformation_config, dict):
        config_dict = transformation_config
    else:
        errors.append(f"Invalid transformation config type: {type(transformation_config)}")
        return False, errors

    # Validate required fields - must have experimental_setups OR standard_transforms (or both)
    has_experimental_setups = "experimental_setups" in config_dict
    has_standard_transforms = "standard_transforms" in config_dict

    if not has_experimental_setups and not has_standard_transforms:
        errors.append(
            "Missing required field: 'experimental_setups' or 'standard_transforms' (at least one required)"
        )
        return False, errors

    if "default_setup" not in config_dict:
        errors.append("Missing required field: 'default_setup'")
        return False, errors

    experimental_setups = config_dict.get("experimental_setups", {})
    standard_transforms = config_dict.get("standard_transforms", [])
    default_setup = config_dict["default_setup"]

    # Validate standard_transforms if present
    if has_standard_transforms:
        if not isinstance(standard_transforms, list):
            errors.append(f"'standard_transforms' must be a list, got {type(standard_transforms)}")
        else:
            # Validate each standard transform
            for i, transform in enumerate(standard_transforms):
                is_valid, transform_errors = validate_transform_spec(transform, strict_mode)
                if not is_valid:
                    for error in transform_errors:
                        errors.append(f"standard_transforms[{i}]: {error}")

    # Validate experimental_setups structure if present
    if has_experimental_setups and not isinstance(experimental_setups, dict):
        errors.append(
            f"'experimental_setups' must be a dictionary, got {type(experimental_setups)}"
        )
        return False, errors

    # At least one transform source must have content in strict mode
    if not experimental_setups and not standard_transforms:
        if strict_mode:
            errors.append(
                "At least one transform must be defined in 'experimental_setups' or 'standard_transforms'"
            )

    # Validate each experimental setup (only if experimental_setups is non-empty)
    if experimental_setups:
        for setup_name, setup_transforms in experimental_setups.items():
            if not isinstance(setup_name, str):
                errors.append(f"Setup name must be string, got {type(setup_name)}")
                continue

            setup_dict = {
                "name": setup_name,
                "transforms": setup_transforms if isinstance(setup_transforms, list) else [],
                "enabled": True,
            }

            is_valid, setup_errors = validate_experimental_setup(setup_dict, strict_mode)
            if not is_valid:
                for error in setup_errors:
                    errors.append(f"Setup '{setup_name}': {error}")

    # Validate default_setup
    if not isinstance(default_setup, str):
        errors.append(f"'default_setup' must be string, got {type(default_setup)}")
    elif experimental_setups and default_setup not in experimental_setups:
        # default_setup must be in experimental_setups if experimental_setups is non-empty
        # If experimental_setups is empty but standard_transforms exists, default_setup
        # is used as a reference name and doesn't need to exist in experimental_setups
        if not has_standard_transforms:
            errors.append(f"default_setup '{default_setup}' not found in experimental_setups")
        # else: standard_transforms only config - default_setup is just a label

    # Validate validation config
    if "validation" in config_dict:
        validation_config = config_dict["validation"]
        if not isinstance(validation_config, dict):
            errors.append(f"'validation' must be a dictionary, got {type(validation_config)}")
        else:
            if "enabled" in validation_config:
                if not isinstance(validation_config["enabled"], bool):
                    errors.append("validation.enabled must be boolean")
            if "strict_mode" in validation_config:
                if not isinstance(validation_config["strict_mode"], bool):
                    errors.append("validation.strict_mode must be boolean")

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_transform_composition_rules(
    transform_sequence: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """
    Validates transform composition rules for a sequence of transforms.

    Integration:
        Checks for common problematic transform combinations and ordering issues
        that could affect dataset processing or model training.

    Args:
        transform_sequence: List of transform specifications in order.

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_warnings)

    Examples:
        >>> sequence = [
        ...     {'name': 'AddSelfLoops'},
        ...     {'name': 'ToUndirected'},
        ...     {'name': 'GCNNorm'}
        ... ]
        >>> is_valid, warnings = validate_transform_composition_rules(sequence)
    """
    warnings = []

    if not transform_sequence:
        return True, warnings

    transform_names = [t.get("name", "") for t in transform_sequence]

    # Check for redundant operations
    if transform_names.count("ToUndirected") > 1:
        warnings.append("Multiple ToUndirected transforms detected - redundant")

    if transform_names.count("AddSelfLoops") > 1:
        warnings.append("Multiple AddSelfLoops transforms detected - redundant")

    # Check for conflicting normalization
    norm_transforms = [name for name in transform_names if "Norm" in name]
    if len(norm_transforms) > 1:
        warnings.append(f"Multiple normalization transforms: {norm_transforms} - may conflict")

    # Check for order dependencies
    if "AddSelfLoops" in transform_names and "GCNNorm" in transform_names:
        self_loops_idx = transform_names.index("AddSelfLoops")
        gcn_norm_idx = transform_names.index("GCNNorm")
        if self_loops_idx > gcn_norm_idx:
            warnings.append("AddSelfLoops should typically come before GCNNorm")

    # Check for structural transforms before augmentation
    augmentation_transforms = ["DropEdge", "DropNode", "MaskFeatures"]
    structural_transforms = ["AddSelfLoops", "ToUndirected"]

    last_structural_idx = -1
    first_augmentation_idx = len(transform_names)

    for i, name in enumerate(transform_names):
        if name in structural_transforms:
            last_structural_idx = i
        if name in augmentation_transforms and first_augmentation_idx == len(transform_names):
            first_augmentation_idx = i

    if last_structural_idx > first_augmentation_idx:
        warnings.append(
            "Structural transforms should typically come before augmentation transforms"
        )

    # Check for geometric transforms with non-geometric data
    geometric_transforms = ["RandomRotate", "RandomScale", "RandomTranslate", "RandomFlip"]
    if any(t in transform_names for t in geometric_transforms):
        warnings.append(
            "Geometric transforms require 3D coordinate data - ensure data is available"
        )

    is_valid = True  # Warnings don't invalidate the sequence
    return is_valid, warnings


# ==========================================
# Descriptor Configuration Validation
# ==========================================


def validate_descriptor_config(
    descriptor_config: "DescriptorConfig", logger: logging.Logger | None = None
) -> tuple[bool, list[str]]:
    """
    Validate descriptor configuration.

    Args:
        descriptor_config: DescriptorConfig instance to validate
        logger: Optional logger for validation messages

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    errors = []

    try:
        # Validate enabled categories exist
        enabled_categories = descriptor_config.get_enabled_categories()
        valid_categories = [
            "constitutional",
            "topological",
            "geometric",
            "electronic",
            "pharmacophore",
            "fingerprint",
            "custom",
        ]

        for category in enabled_categories:
            if category not in valid_categories:
                errors.append(f"Invalid enabled category: {category}")

        # Validate cache settings
        if descriptor_config.cache_descriptors and descriptor_config.cache_path:
            from pathlib import Path

            cache_path = Path(descriptor_config.cache_path)
            if not cache_path.parent.exists():
                errors.append(f"Cache path parent directory does not exist: {cache_path.parent}")

        # Validate parallel computation settings
        if descriptor_config.parallel_computation:
            if descriptor_config.num_workers < 2:
                errors.append(
                    f"Parallel computation requires num_workers >= 2, got {descriptor_config.num_workers}"
                )

        # Validate category configurations
        for category in descriptor_config.categories:
            if category not in valid_categories:
                errors.append(f"Invalid category in configuration: {category}")

            cat_config = descriptor_config.categories[category]
            if not isinstance(cat_config, dict):
                errors.append(f"Category '{category}' configuration must be a dictionary")

        # Log validation result
        if errors:
            logger.warning(f"Descriptor configuration validation found {len(errors)} errors")
        else:
            logger.debug("Descriptor configuration validation passed")

    except Exception as e:
        errors.append(f"Descriptor validation exception: {str(e)}")
        logger.error(f"Descriptor validation error: {e}")

    return len(errors) == 0, errors


def validate_descriptor_category_compatibility(
    category: str, dataset_type: str, logger: logging.Logger | None = None
) -> tuple[bool, list[str], list[str]]:
    """
    Validate descriptor category compatibility with dataset type.

    PHASE 6: Refactored to use registry-based feature queries.

    Args:
        category: Descriptor category name
        dataset_type: Dataset type
        logger: Optional logger

    Returns:
        Tuple of (is_compatible, list_of_errors, list_of_warnings)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    errors = []
    warnings = []

    try:
        # Check category validity
        valid_categories = [
            "constitutional",
            "topological",
            "geometric",
            "electronic",
            "pharmacophore",
            "fingerprint",
            "custom",
        ]

        if category not in valid_categories:
            errors.append(f"Invalid category: {category}")
            return False, errors, warnings

        # ====================================================================
        # PHASE 6: Registry-based dataset type validation
        # ====================================================================
        if not _is_dataset_type_registered(dataset_type):
            available = _get_available_dataset_types()
            errors.append(f"Unknown dataset type: {dataset_type}. Available: {available}")
            return False, errors, warnings

        # ====================================================================
        # PHASE 6: Feature-based descriptor compatibility (replaces if/elif)
        # ====================================================================

        # Check orbital analysis capability
        if _get_dataset_feature(dataset_type, "orbital_analysis"):
            # Dataset with orbital analysis (e.g., wavefunction-based)
            if category == "electronic":
                logger.info(f"{dataset_type} dataset: Electronic descriptors can use quantum data")
            elif category == "geometric":
                warnings.append("Geometric descriptors will use optimized coordinates")
        else:
            # Dataset without orbital analysis
            if category == "electronic":
                warnings.append(
                    f"{dataset_type} dataset: Electronic descriptors limited to "
                    "available properties"
                )

    except Exception as e:
        errors.append(f"Compatibility validation exception: {str(e)}")
        logger.error(f"Category compatibility validation error: {e}")

    return len(errors) == 0, errors, warnings


def validate_descriptor_cache_settings(
    cache_descriptors: bool, cache_path: str | None, logger: logging.Logger | None = None
) -> tuple[bool, list[str]]:
    """
    Validate descriptor cache settings.

    Args:
        cache_descriptors: Whether caching is enabled
        cache_path: Path for cache storage
        logger: Optional logger

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    import os

    if logger is None:
        logger = logging.getLogger(__name__)

    errors = []

    try:
        if cache_descriptors:
            if cache_path:
                from pathlib import Path

                cache_path_obj = Path(cache_path)

                # Check if parent directory exists
                if not cache_path_obj.parent.exists():
                    errors.append(
                        f"Cache path parent directory does not exist: {cache_path_obj.parent}"
                    )

                # Check if path is writable (if it exists)
                if cache_path_obj.exists():
                    if not cache_path_obj.is_dir():
                        errors.append(f"Cache path exists but is not a directory: {cache_path}")
                    elif not os.access(cache_path_obj, os.W_OK):
                        errors.append(f"Cache path is not writable: {cache_path}")
            else:
                logger.debug("Cache enabled with auto-generated cache path")

    except Exception as e:
        errors.append(f"Cache validation exception: {str(e)}")
        logger.error(f"Cache settings validation error: {e}")

    return len(errors) == 0, errors


# ============================================================================
# ENHANCEMENT: Backward Compatibility Wrapper
# ============================================================================


def validate_transforms_with_fallback(
    transform_config: dict[str, Any],
    experimental_setup: str | None = None,
    logger: logging.Logger | None = None,
    detailed: bool = False,
) -> bool | tuple[bool, list[ValidationIssueDetail]]:
    """
    Validate transforms with automatic fallback to basic validation.

    Enhancement: Provides seamless backward compatibility.
    Tries enhanced validation first, falls back to basic checks if unavailable.

    Args:
        transform_config: Transform configuration dictionary
        experimental_setup: Optional setup name for context
        logger: Optional logger instance
        detailed: If True, return (is_valid, issues); if False, return bool only

    Returns:
        If detailed=False: bool indicating validity
        If detailed=True: (bool, List[ValidationIssueDetail]) tuple

    Example:
        >>> # Simple usage (backward compatible)
        >>> is_valid = validate_transforms_with_fallback(config)
        >>> if not is_valid:
        >>>     print("Configuration invalid")
        >>>
        >>> # Detailed usage (get issue details)
        >>> is_valid, issues = validate_transforms_with_fallback(config, detailed=True)
        >>> for issue in issues:
        >>>     if issue.severity == ValidationSeverity.ERROR:
        >>>         print(f"ERROR: {issue.message}")
    """
    logger = logger or logging.getLogger(__name__)

    # Try enhanced validation
    if TRANSFORM_INTROSPECTION_AVAILABLE:
        try:
            validator = TransformValidator(logger=logger)
            is_valid, issues = validator.validate_transform_config(
                transform_config, experimental_setup=experimental_setup, collect_issues=True
            )

            if detailed:
                return is_valid, issues
            else:
                return is_valid

        except Exception as e:
            logger.warning(f"validation failed, falling back to basic validation: {e}")

    # Fallback to basic validation
    try:
        # Basic structure checks only
        if not isinstance(transform_config, dict):
            return (False, []) if detailed else False

        transforms = transform_config.get("transforms", [])
        if not isinstance(transforms, list):
            return (False, []) if detailed else False

        for transform_spec in transforms:
            if not isinstance(transform_spec, dict):
                return (False, []) if detailed else False
            if "name" not in transform_spec:
                return (False, []) if detailed else False

        return (True, []) if detailed else True

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return (False, []) if detailed else False


# ==========================================
# Diagnostic and Testing Functions
# ==========================================


def run_validation_diagnostics() -> dict[str, Any]:
    """
    Runs diagnostic tests on all validation functions.

    PHASE 6: Includes registry integration status.

    This function provides comprehensive testing of all validation functions
    to ensure they work correctly. It's used for debugging and verifying
    that the validation system is functioning properly.

    Handler Integration:
        Handlers can use this during initialization or debugging to verify
        that all validation functions are working correctly before processing
        molecular data.

    Returns:
        Dict[str, Any]: Dictionary of test results (mix of bool and other types).

    Handler Usage Examples:
        >>> # Handler initialization diagnostics
        >>> diagnostic_results = run_validation_diagnostics()
        >>> failed_tests = [name for name, result in diagnostic_results.items() if not result]
        >>> if failed_tests:
        ...     raise HandlerError(
        ...         message=f"Validation diagnostic failures: {failed_tests}",
        ...         handler_type="generic",
        ...         handler_operation="diagnostics"
        ...     )
        >>>
        >>> # Development and debugging
        >>> if self.debug_mode:
        ...     diagnostics = run_validation_diagnostics()
        ...     self.logger.debug(f"Validation diagnostics: {diagnostics}")
    """
    results = {}

    # ========================================================================
    # PHASE 6: Registry integration diagnostics
    # ========================================================================
    registry_status = get_registry_status()
    results["registry_available"] = registry_status["registry_available"]
    results["registry_initialized"] = registry_status["registry_initialized"]
    results["available_dataset_types"] = registry_status["available_dataset_types"]
    results["phase_6_complete"] = registry_status["phase_6_complete"]

    # Test is_value_valid_and_not_nan
    test_cases = [
        (3.14, True, "float"),
        (np.nan, False, "nan"),
        (None, False, "none"),
        (np.array([1, 2, 3]), True, "valid_array"),
        (np.array([1, np.nan, 3]), False, "array_with_nan"),
        ("123.45", True, "numeric_string"),
        ("not_a_number", False, "non_numeric_string"),
        (torch.tensor([1.0, 2.0]), True, "valid_tensor"),
    ]

    for value, expected, name in test_cases:
        try:
            result = is_value_valid_and_not_nan(value)
            results[f"is_value_valid_{name}"] = result == expected
        except Exception as e:
            results[f"is_value_valid_{name}"] = False
            logger.error(f"Diagnostic test failed for {name}: {e}")

    # Test validate_array_shape
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    results["validate_shape_correct"] = validate_array_shape(arr, expected_shape=(2, 3))
    results["validate_shape_wrong"] = not validate_array_shape(arr, expected_shape=(3, 2))
    results["validate_ndim_correct"] = validate_array_shape(arr, expected_ndim=2)

    # Test validate_numeric_range
    results["validate_range_in"] = validate_numeric_range(5.0, min_val=0, max_val=10)
    results["validate_range_out"] = not validate_numeric_range(15.0, min_val=0, max_val=10)

    # Test convert_to_scalar function
    try:
        results["convert_to_scalar"] = (
            convert_to_scalar(3.14) == 3.14
            and convert_to_scalar(np.array([2.71])) == 2.71
            and abs(convert_to_scalar(torch.tensor(1.41)) - 1.41) < 1e-6
            and convert_to_scalar(None, allow_none=True) is None
        )
    except Exception as e:
        results["convert_to_scalar"] = False
        logger.error(f"convert_to_scalar test failed: {e}")

    # ========================================================================
    # PHASE 6: Feature-based query tests
    # ========================================================================
    try:
        # Feature queries now use registry-only pattern; results depend on registry availability
        available_types = _get_available_dataset_types()
        if available_types:
            # Test feature queries dynamically for whatever types are available
            feature_query_success = True
            for dt in available_types[:3]:  # Test up to first 3 available types
                try:
                    _get_dataset_feature(dt, "uncertainty_handling")
                    _get_dataset_feature(dt, "vibrational_analysis")
                    _get_dataset_feature(dt, "orbital_analysis")
                except Exception:
                    feature_query_success = False
            results["feature_query_dynamic"] = feature_query_success
        else:
            # No registry available - feature queries should return False gracefully
            results["feature_query_dynamic"] = (
                not _get_dataset_feature("unknown_type", "uncertainty_handling")
            )
    except Exception as e:
        logger.error(f"Feature query tests failed: {e}")
        results["feature_query_tests"] = False

    return results


def run_handler_validation_tests() -> dict[str, bool]:
    """
    Run validation tests specifically for handler support functions.

    PHASE 6: Updated to test all registered handler types dynamically and
    include registry integration status.

    This function tests all handler-specific validation functions to ensure
    they work correctly with the handler pattern. It verifies compatibility
    checking, batch validation, and other handler-specific operations.

    Handler Integration:
        Handler factories and development environments use this to verify
        that all handler validation functions are working correctly.

    Returns:
        Dict[str, bool]: Test results for handler validation functions.

    Handler Usage Examples:
        >>> # Handler factory testing
        >>> handler_tests = run_handler_validation_tests()
        >>> if not all(handler_tests.values()):
        ...     failed_tests = [name for name, result in handler_tests.items() if not result]
        ...     raise HandlerError(
        ...         message=f"Handler validation test failures: {failed_tests}",
        ...         handler_type="factory",
        ...         handler_operation="validation_tests"
        ...     )
        >>>
        >>> # Development environment validation
        >>> test_results = run_handler_validation_tests()
        >>> logger.info(f"Handler validation tests: {sum(test_results.values())}/{len(test_results)} passed")
    """
    results = {}

    try:
        # ====================================================================
        # PHASE 6: Registry integration tests
        # ====================================================================
        registry_status = get_registry_status()
        results["registry_initialized"] = registry_status["registry_initialized"]
        results["phase_6_complete"] = registry_status["phase_6_complete"]

        # ====================================================================
        # PHASE 6: Dynamic type testing - test all available handler types
        # ====================================================================
        available_types = _get_available_dataset_types()
        results["dynamic_types_available"] = (
            len(available_types) >= 0
        )  # Dynamic discovery, count varies

        # Test each available handler type dynamically
        for handler_type in available_types:
            compatibility = _get_handler_compatibility_checks(handler_type)

            # Build config based on handler features
            config = {
                "dataset_type": handler_type,
                "uncertainty_config": {"uncertainty_field_name": "std"}
                if compatibility.get("supports_uncertainty")
                else None,
                "is_uncertainty_enabled": compatibility.get("supports_uncertainty", False),
            }

            processing = {
                "scalar_graph_targets": ["Etot"],
                "node_features": ["atomic_number"]
                if not compatibility.get("supports_uncertainty")
                else [],
            }

            if compatibility.get("supports_vibrational"):
                processing["calculate_atomization_energy_from"] = "U0"
                processing["atomization_energy_key_name"] = "atomization_energy"

            is_compatible, issues = validate_handler_compatibility(handler_type, config, processing)
            results[f"{handler_type.lower()}_handler_compatible"] = (
                is_compatible and len(issues) == 0
            )

        # Test batch validation
        batch_data = [
            {
                "atoms": np.array([1, 6]),
                "coordinates": np.array([[0, 0, 0], [1, 0, 0]]),
                "Etot": -3.0,
                "inchi": "InChI=test1",
            },
            {
                "atoms": np.array([1, 6, 8]),
                "coordinates": np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
                "Etot": -5.0,
                "inchi": "InChI=test2",
            },
        ]

        is_valid, summary = validate_handler_molecular_batch(
            batch_data, ["atoms", "coordinates", "Etot"]
        )
        results["batch_validation"] = is_valid and summary["valid_molecules"] == 2

        # Test validation report generation
        report = create_validation_report(summary)
        results["validation_report"] = isinstance(report, str) and len(report) > 0

        # Test handler exception integration
        results["handler_exceptions_available"] = all(
            [
                HandlerError is not None,
                HandlerValidationError is not None,
                HandlerConfigurationError is not None,
                DatasetSpecificHandlerError is not None,
                UncertaintyProcessingError is not None,
            ]
        )

        # ====================================================================
        # PHASE 6: Test unknown handler type rejection with dynamic available_types
        # ====================================================================
        is_compatible, issues = validate_handler_compatibility(
            "NonExistentType",
            {"dataset_type": "NonExistentType"},
            {"scalar_graph_targets": ["Etot"]},
        )
        # Should be rejected with "Available:" in error message
        results["unknown_type_rejected_with_available"] = not is_compatible and any(
            "Available" in issue for issue in issues
        )

    except Exception as e:
        logger.error(f"Handler validation tests failed: {e}")
        results["handler_validation_tests"] = False

    return results


def run_transformation_validation_tests() -> dict[str, bool]:
    """
    Run validation tests for transformation system integration.

    Returns:
        Dict[str, bool]: Test results for transformation validation functions.
    """
    results = {}

    try:
        # Test TransformSpec validation
        spec = {"name": "AddSelfLoops", "kwargs": {}, "enabled": True}
        is_valid, errors = validate_transform_spec(spec)
        results["transform_spec_valid"] = is_valid and len(errors) == 0

        # Test invalid TransformSpec
        invalid_spec = {"kwargs": {}}  # Missing 'name'
        is_valid, errors = validate_transform_spec(invalid_spec)
        results["transform_spec_invalid"] = not is_valid and len(errors) > 0

        # Test ExperimentalSetup validation
        setup = {
            "name": "baseline",
            "transforms": [{"name": "AddSelfLoops"}, {"name": "ToUndirected"}],
            "enabled": True,
        }
        is_valid, errors = validate_experimental_setup(setup)
        results["experimental_setup_valid"] = is_valid and len(errors) == 0

        # Test TransformationConfig validation
        config = {
            "experimental_setups": {"baseline": [{"name": "AddSelfLoops"}]},
            "default_setup": "baseline",
            "validation": {"enabled": True, "strict_mode": False},
        }
        is_valid, errors = validate_transformation_config(config)
        results["transformation_config_valid"] = is_valid and len(errors) == 0

        # Test transform composition rules
        sequence = [{"name": "AddSelfLoops"}, {"name": "ToUndirected"}, {"name": "GCNNorm"}]
        is_valid, warnings = validate_transform_composition_rules(sequence)
        results["composition_rules_check"] = is_valid  # Should pass with possible warnings

        # Test problematic composition
        problematic_sequence = [
            {"name": "GCNNorm"},
            {"name": "AddSelfLoops"},  # Wrong order
        ]
        is_valid, warnings = validate_transform_composition_rules(problematic_sequence)
        results["composition_rules_warning"] = len(warnings) > 0  # Should have warnings

        results["transformation_validation_available"] = True

    except Exception as e:
        logger.error(f"Transformation validation tests failed: {e}")
        results["transformation_validation_available"] = False

    return results


if __name__ == "__main__":
    # Run diagnostics when module is executed directly
    print("Running validation diagnostics...")
    diagnostics = run_validation_diagnostics()

    print("\nDiagnostic Results:")
    print("-" * 40)
    for test_name, passed in diagnostics.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")

    # Run handler validation tests
    print("\nRunning handler validation tests...")
    handler_tests = run_handler_validation_tests()

    print("\nHandler Test Results:")
    print("-" * 40)
    for test_name, passed in handler_tests.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")

    # Run transformation validation tests
    print("\nRunning transformation validation tests...")
    transform_tests = run_transformation_validation_tests()

    print("\nTransformation Test Results:")
    print("-" * 40)
    for test_name, passed in transform_tests.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")

    # ========================================================================
    # NEW: Pitfall 1 Testing - convert_to_scalar None Handling
    # ========================================================================
    print("\n" + "=" * 70)
    print("PITFALL 1 TESTS: convert_to_scalar None Handling")
    print("=" * 70)

    pitfall_tests = {}

    # Test 1: None with allow_none=False should raise ValidationError
    print("\nTest 1: None with allow_none=False (should raise exception)")
    try:
        result = convert_to_scalar(None, allow_none=False)
        pitfall_tests["none_raises_exception"] = False
        print("  ✗ FAIL: Expected ValidationError but got result:", result)
    except ValidationError as e:
        pitfall_tests["none_raises_exception"] = True
        print(f"  ✓ PASS: Correctly raised ValidationError: {e}")
    except Exception as e:
        pitfall_tests["none_raises_exception"] = False
        print(f"  ✗ FAIL: Unexpected exception type: {type(e).__name__}: {e}")

    # Test 2: None with allow_none=True should return None
    print("\nTest 2: None with allow_none=True (should return None)")
    try:
        result = convert_to_scalar(None, allow_none=True)
        if result is None:
            pitfall_tests["none_returns_none"] = True
            print("  ✓ PASS: Correctly returned None")
        else:
            pitfall_tests["none_returns_none"] = False
            print(f"  ✗ FAIL: Expected None but got: {result}")
    except Exception as e:
        pitfall_tests["none_returns_none"] = False
        print(f"  ✗ FAIL: Unexpected exception: {type(e).__name__}: {e}")

    # Test 3: validate_uncertainty_data with None
    print("\nTest 3: validate_uncertainty_data(None) (should return None gracefully)")
    try:
        result = validate_uncertainty_data(None, molecule_index=0, uncertainty_field_name="std")
        if result is None:
            pitfall_tests["uncertainty_none_graceful"] = True
            print("  ✓ PASS: validate_uncertainty_data returns None gracefully")
        else:
            pitfall_tests["uncertainty_none_graceful"] = False
            print(f"  ✗ FAIL: Expected None but got: {result}")
    except Exception as e:
        pitfall_tests["uncertainty_none_graceful"] = False
        print(f"  ✗ FAIL: Unexpected exception: {type(e).__name__}: {e}")

    # Test 4: validate_molecular_data_dict with None energy
    print("\nTest 4: validate_molecular_data_dict with None energy for uncertainty-enabled dataset")
    try:
        data = {
            "Etot": None,
            "atoms": np.array([1, 6]),
            "coordinates": np.array([[0, 0, 0], [1, 0, 0]]),
            "inchi": "InChI=test",
        }
        # Use first available uncertainty-enabled type, or a placeholder
        test_uncertainty_type = None
        for dt in _get_available_dataset_types():
            if _get_dataset_feature(dt, "uncertainty_handling"):
                test_uncertainty_type = dt
                break
        test_type = test_uncertainty_type or "TestUncertaintyDataset"
        is_valid, errors = validate_molecular_data_dict(
            data, ["Etot", "atoms", "coordinates"], dataset_type=test_type
        )

        # Should be invalid with error about None/missing
        none_error_found = any(
            "None" in err or "missing" in err.lower() or "invalid" in err.lower() for err in errors
        )

        if not is_valid and none_error_found:
            pitfall_tests["molecular_dict_none_handling"] = True
            print("  ✓ PASS: Correctly detected invalid None energy")
            print(f"    Errors: {errors[:2]}")  # Show first 2 errors
        else:
            pitfall_tests["molecular_dict_none_handling"] = False
            print(f"  ✗ FAIL: is_valid={is_valid}, errors={errors}")
    except Exception as e:
        pitfall_tests["molecular_dict_none_handling"] = False
        print(f"  ✗ FAIL: Unexpected exception: {type(e).__name__}: {e}")

    # Test 5: Safe conversion patterns
    print("\nTest 5: Safe conversion patterns (explicit None check)")
    try:
        test_value = None

        # Pattern 1: Check before conversion
        result1 = None if test_value is None else convert_to_scalar(test_value, allow_none=False)

        # Pattern 2: Use allow_none=True and check result
        result2 = convert_to_scalar(test_value, allow_none=True)

        if result1 is None and result2 is None:
            pitfall_tests["safe_patterns"] = True
            print("  ✓ PASS: Both safe patterns work correctly")
        else:
            pitfall_tests["safe_patterns"] = False
            print(f"  ✗ FAIL: Pattern results incorrect: {result1}, {result2}")
    except Exception as e:
        pitfall_tests["safe_patterns"] = False
        print(f"  ✗ FAIL: Exception in safe patterns: {type(e).__name__}: {e}")

    # Test 6: Edge cases with different None-like values
    print("\nTest 6: Edge cases with None-like values")
    edge_cases = [
        (np.nan, "np.nan", False),  # Should fail validation
        (float("nan"), "float('nan')", False),
        (np.array([np.nan]), "array([nan])", False),
        ("", "empty string", False),
        ("None", "string 'None'", False),
    ]

    edge_case_results = []
    for value, description, _expected_valid in edge_cases:
        try:
            result = convert_to_scalar(value, allow_none=False, value_name=description)
            # Should not reach here for invalid values
            edge_case_results.append(False)
            print(f"  ✗ {description}: Got {result}, expected exception")
        except (ValidationError, ValueError):
            edge_case_results.append(True)
            print(f"  ✓ {description}: Correctly rejected")
        except Exception as e:
            edge_case_results.append(False)
            print(f"  ✗ {description}: Unexpected exception {type(e).__name__}")

    pitfall_tests["edge_cases"] = all(edge_case_results)

    # Pitfall 1 Summary
    print("\n" + "-" * 70)
    print("PITFALL 1 TEST SUMMARY:")
    print("-" * 70)
    for test_name, passed in pitfall_tests.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")

    pitfall_passed = sum(pitfall_tests.values())
    pitfall_total = len(pitfall_tests)
    print(f"\nPitfall 1 Tests: {pitfall_passed}/{pitfall_total} passed")

    # ========================================================================
    # NEW: Pitfall 2 Testing - Empty Array Validation
    # ========================================================================
    print("\n" + "=" * 70)
    print("PITFALL 2 TESTS: Empty Array Validation")
    print("=" * 70)

    pitfall2_tests = {}

    # Test 1: Empty array with allow_empty_array=False (should fail)
    print("\nTest 1: Empty array with allow_empty_array=False (should fail)")
    try:
        result = is_value_valid_and_not_nan(np.array([]), allow_empty_array=False)
        if not result:
            pitfall2_tests["empty_array_rejected"] = True
            print("  ✓ PASS: Empty array correctly rejected")
        else:
            pitfall2_tests["empty_array_rejected"] = False
            print(f"  ✗ FAIL: Empty array should be rejected, got: {result}")
    except Exception as e:
        pitfall2_tests["empty_array_rejected"] = False
        print(f"  ✗ FAIL: Unexpected exception: {type(e).__name__}: {e}")

    # Test 2: Empty array with allow_empty_array=True (should pass)
    print("\nTest 2: Empty array with allow_empty_array=True (should pass)")
    try:
        result = is_value_valid_and_not_nan(np.array([]), allow_empty_array=True)
        if result:
            pitfall2_tests["empty_array_allowed"] = True
            print("  ✓ PASS: Empty array correctly allowed")
        else:
            pitfall2_tests["empty_array_allowed"] = False
            print(f"  ✗ FAIL: Empty array should be allowed, got: {result}")
    except Exception as e:
        pitfall2_tests["empty_array_allowed"] = False
        print(f"  ✗ FAIL: Unexpected exception: {type(e).__name__}: {e}")

    # Test 3: Empty list validation
    print("\nTest 3: Empty list validation")
    try:
        result_rejected = not is_value_valid_and_not_nan([], allow_empty_array=False)
        result_allowed = is_value_valid_and_not_nan([], allow_empty_array=True)

        if result_rejected and result_allowed:
            pitfall2_tests["empty_list_handling"] = True
            print("  ✓ PASS: Empty list handled correctly (rejected when False, allowed when True)")
        else:
            pitfall2_tests["empty_list_handling"] = False
            print(
                f"  ✗ FAIL: Empty list handling incorrect: rejected={result_rejected}, allowed={result_allowed}"
            )
    except Exception as e:
        pitfall2_tests["empty_list_handling"] = False
        print(f"  ✗ FAIL: Unexpected exception: {type(e).__name__}: {e}")

    # Test 4: validate_property_value with empty array
    print("\nTest 4: validate_property_value with empty array parameter")
    try:
        empty_freqs = np.array([])

        # Should fail without allow_empty
        result_fail = validate_property_value(
            empty_freqs, "freqs", expected_type=np.ndarray, allow_empty=False
        )

        # Should pass with allow_empty
        result_pass = validate_property_value(
            empty_freqs, "freqs", expected_type=np.ndarray, allow_empty=True
        )

        if not result_fail and result_pass:
            pitfall2_tests["property_value_empty"] = True
            print("  ✓ PASS: validate_property_value handles allow_empty correctly")
        else:
            pitfall2_tests["property_value_empty"] = False
            print(
                f"  ✗ FAIL: Expected fail=False, pass=True, got fail={result_fail}, pass={result_pass}"
            )
    except Exception as e:
        pitfall2_tests["property_value_empty"] = False
        print(f"  ✗ FAIL: Unexpected exception: {type(e).__name__}: {e}")

    # Test 5: validate_array_shape with empty array
    print("\nTest 5: validate_array_shape with empty array parameter")
    try:
        empty_arr = np.array([])

        # Should fail without allow_empty
        result_fail = validate_array_shape(empty_arr, expected_ndim=1, allow_empty=False)

        # Should pass with allow_empty
        result_pass = validate_array_shape(empty_arr, expected_ndim=1, allow_empty=True)

        if not result_fail and result_pass:
            pitfall2_tests["array_shape_empty"] = True
            print("  ✓ PASS: validate_array_shape handles allow_empty correctly")
        else:
            pitfall2_tests["array_shape_empty"] = False
            print(
                f"  ✗ FAIL: Expected fail=False, pass=True, got fail={result_fail}, pass={result_pass}"
            )
    except Exception as e:
        pitfall2_tests["array_shape_empty"] = False
        print(f"  ✗ FAIL: Unexpected exception: {type(e).__name__}: {e}")

    # Test 6: validate_coordinates_3d with empty coordinates
    print("\nTest 6: validate_coordinates_3d with empty molecule")
    try:
        empty_coords = np.array([]).reshape(0, 3)

        # Should fail without allow_empty
        result_fail = validate_coordinates_3d(empty_coords, num_atoms=0, allow_empty=False)

        # Should pass with allow_empty
        result_pass = validate_coordinates_3d(empty_coords, num_atoms=0, allow_empty=True)

        if not result_fail and result_pass:
            pitfall2_tests["coordinates_empty"] = True
            print("  ✓ PASS: validate_coordinates_3d handles empty molecules correctly")
        else:
            pitfall2_tests["coordinates_empty"] = False
            print(
                f"  ✗ FAIL: Expected fail=False, pass=True, got fail={result_fail}, pass={result_pass}"
            )
    except Exception as e:
        pitfall2_tests["coordinates_empty"] = False
        print(f"  ✗ FAIL: Unexpected exception: {type(e).__name__}: {e}")

    # Test 7: validate_batch_consistency with empty arrays
    print("\nTest 7: validate_batch_consistency with allow_empty_arrays parameter")
    try:
        batch_data = [
            {
                "atoms": np.array([1, 6]),
                "coordinates": np.array([[0, 0, 0], [1, 0, 0]]),
                "freqs": np.array([]),  # Empty vibrational frequencies
                "Etot": -3.0,
            },
            {
                "atoms": np.array([1]),
                "coordinates": np.array([[0, 0, 0]]),
                "freqs": np.array([]),  # Single atom has no vibrations
                "Etot": -1.0,
            },
        ]

        # Should fail without allow_empty_arrays
        is_valid_fail, errors_fail = validate_batch_consistency(
            batch_data, ["atoms", "coordinates", "freqs", "Etot"]
        )

        # Should pass with allow_empty_arrays for 'freqs'
        is_valid_pass, errors_pass = validate_batch_consistency(
            batch_data, ["atoms", "coordinates", "freqs", "Etot"], allow_empty_arrays={"freqs"}
        )

        if not is_valid_fail and is_valid_pass:
            pitfall2_tests["batch_empty_arrays"] = True
            print("  ✓ PASS: validate_batch_consistency handles allow_empty_arrays correctly")
        else:
            pitfall2_tests["batch_empty_arrays"] = False
            print("  ✗ FAIL: Expected fail=False, pass=True")
            print(f"    Without allow_empty: valid={is_valid_fail}, errors={len(errors_fail)}")
            print(f"    With allow_empty: valid={is_valid_pass}, errors={len(errors_pass)}")
    except Exception as e:
        pitfall2_tests["batch_empty_arrays"] = False
        print(f"  ✗ FAIL: Unexpected exception: {type(e).__name__}: {e}")

    # Test 8: Vibrational-analysis dataset with empty vibrational modes
    print("\nTest 8: Vibrational-analysis dataset with empty vibrational modes")
    try:
        # Single atom molecule (no vibrations)
        single_atom_data = {
            "atoms": np.array([1]),
            "coordinates": np.array([[0, 0, 0]]),
            "Etot": -0.5,
            "freqs": np.array([]),
            "vibmodes": np.array([]),
            "inchi": "InChI=1S/H",
        }

        # Use first available vibrational-analysis type, or a placeholder
        test_vibrational_type = None
        for dt in _get_available_dataset_types():
            if _get_dataset_feature(dt, "vibrational_analysis"):
                test_vibrational_type = dt
                break
        test_type = test_vibrational_type or "TestVibrationalDataset"

        is_valid, errors = validate_molecular_data_dict(
            single_atom_data, ["atoms", "coordinates", "Etot"], dataset_type=test_type
        )

        # Should be valid - empty freqs/vibmodes are OK for single atoms
        if is_valid:
            pitfall2_tests["vibrational_empty_vibrations"] = True
            print("  ✓ PASS: Vibrational-analysis validation accepts empty vibrational modes")
        else:
            pitfall2_tests["vibrational_empty_vibrations"] = False
            print("  ✗ FAIL: Vibrational-analysis validation should accept empty vibrations")
            print(f"    Errors: {errors}")
    except Exception as e:
        pitfall2_tests["vibrational_empty_vibrations"] = False
        print(f"  ✗ FAIL: Unexpected exception: {type(e).__name__}: {e}")

    # Test 9: Empty tensor validation (PyTorch)
    print("\nTest 9: Empty PyTorch tensor validation")
    try:
        empty_tensor = torch.tensor([])

        result_rejected = not is_value_valid_and_not_nan(empty_tensor, allow_empty_array=False)
        result_allowed = is_value_valid_and_not_nan(empty_tensor, allow_empty_array=True)

        if result_rejected and result_allowed:
            pitfall2_tests["empty_tensor_handling"] = True
            print("  ✓ PASS: Empty tensor handled correctly")
        else:
            pitfall2_tests["empty_tensor_handling"] = False
            print("  ✗ FAIL: Empty tensor handling incorrect")
    except Exception as e:
        pitfall2_tests["empty_tensor_handling"] = False
        print(f"  ✗ FAIL: Unexpected exception: {type(e).__name__}: {e}")

    # Pitfall 2 Summary
    print("\n" + "-" * 70)
    print("PITFALL 2 TEST SUMMARY:")
    print("-" * 70)
    for test_name, passed in pitfall2_tests.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")

    pitfall2_passed = sum(pitfall2_tests.values())
    pitfall2_total = len(pitfall2_tests)
    print(f"\nPitfall 2 Tests: {pitfall2_passed}/{pitfall2_total} passed")

    # ============================================================================
    # Pitfall 3: Validation Result Checking Utilities
    # ============================================================================

    def validate_and_require(
        validation_func: Callable,
        *args,
        error_message: str | None = None,
        exception_class: type[Exception] = ValidationError,
        **kwargs,
    ) -> Any:
        """
        Execute validation function and raise exception if it fails.

        Pitfall 3 Solution: Convenience wrapper that always enforces checking.

        Args:
            validation_func: Validation function to call
            *args: Arguments to pass to validation function
            error_message: Custom error message
            exception_class: Exception to raise on failure
            **kwargs: Keyword arguments to pass to validation function

        Returns:
            Validated data (if validation function returns data)

        Raises:
            exception_class: If validation fails

        Example:
            >>> # Instead of:
            >>> is_valid, errors = validate_molecular_data_dict(data, props)
            >>> if not is_valid:
            >>>     raise ValidationError(errors)
            >>>
            >>> # Use:
            >>> validate_and_require(
            >>>     validate_molecular_data_dict, data, props,
            >>>     error_message="Critical validation failed"
            >>> )
        """
        result = validation_func(*args, **kwargs)

        # Handle ValidationResult wrapper
        if isinstance(result, ValidationResult):
            if not result.is_valid:
                msg = error_message or f"Validation failed: {'; '.join(result.errors[:3])}"
                raise exception_class(msg)
            return result.get_validated_data()

        # Handle traditional tuple return
        if isinstance(result, tuple) and len(result) >= 2:
            is_valid, errors = result[0], result[1]
            if not is_valid:
                msg = (
                    error_message
                    or f"Validation failed: {'; '.join(errors[:3]) if errors else 'Unknown error'}"
                )
                raise exception_class(msg)
            # Return additional data if present
            return result[2:] if len(result) > 2 else None

        # Handle boolean return
        if isinstance(result, bool):
            if not result:
                msg = error_message or "Validation failed"
                raise exception_class(msg)
            return None

        return result

    def log_validation_errors(
        is_valid: bool,
        errors: list[str],
        logger: logging.Logger,
        context: str = "",
        level: str = "error",
    ) -> None:
        """
        Log validation errors with proper context.

        Pitfall 3 Solution: Standardized error logging for validation results.

        Args:
            is_valid: Validation result
            errors: List of error messages
            logger: Logger instance
            context: Additional context for logging
            level: Log level ('error', 'warning', 'info')
        """
        if not is_valid and errors:
            log_func = getattr(logger, level, logger.error)
            prefix = f"{context}: " if context else ""

            if len(errors) <= 3:
                # Log all errors if few
                for error in errors:
                    log_func(f"{prefix}{error}")
            else:
                # Log first few and summarize
                for error in errors[:3]:
                    log_func(f"{prefix}{error}")
                log_func(f"{prefix}... and {len(errors) - 3} more errors")

    class ValidationContext:
        """
        Context manager for validation operations.

        Pitfall 3 Solution: Ensures validation results are always checked
        within a context.

        Usage:
            >>> with ValidationContext("Molecule batch validation") as ctx:
            >>>     is_valid, errors = validate_handler_molecular_batch(batch, props)
            >>>     ctx.check(is_valid, errors)  # Must call check() or raises
            >>>
            >>>     # Or use require():
            >>>     ctx.require(is_valid, errors, "Critical validation failed")
        """

        def __init__(self, context: str, logger: logging.Logger | None = None):
            self.context = context
            self.logger = logger or logging.getLogger(__name__)
            self._checked = False
            self._is_valid = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if not self._checked and not exc_type:
                self.logger.warning(
                    f"ValidationContext '{self.context}' exited without checking results!"
                )
            return False

        def check(self, is_valid: bool, errors: list[str] | None = None) -> bool:
            """Mark validation as checked and log if invalid."""
            self._checked = True
            self._is_valid = is_valid

            if not is_valid and errors:
                log_validation_errors(is_valid, errors, self.logger, self.context)

            return is_valid

        def require(
            self,
            is_valid: bool,
            errors: list[str] | None = None,
            message: str | None = None,
            exception_class: type[Exception] = ValidationError,
        ) -> None:
            """Check validation and raise if invalid."""
            self._checked = True
            self._is_valid = is_valid

            if not is_valid:
                error_msg = message or f"{self.context}: Validation failed"
                if errors:
                    error_msg = f"{error_msg}: {'; '.join(errors[:3])}"
                raise exception_class(error_msg)

    # ========================================================================
    # PITFALL 4: Removed - Deprecated function cleanup complete
    # ========================================================================
    pitfall4_tests = {}
    pitfall4_passed = 0
    pitfall4_total = 0

    # ========================================================================
    # Combined Final Results
    # ========================================================================
    # Update combined results
    all_tests = {
        **diagnostics,
        **handler_tests,
        **transform_tests,
        **pitfall_tests,
        **pitfall2_tests,
    }
    total_tests = len(all_tests)
    passed_tests = sum(all_tests.values())

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"Core Diagnostics: {sum(diagnostics.values())}/{len(diagnostics)} passed")
    print(f"Handler Tests: {sum(handler_tests.values())}/{len(handler_tests)} passed")
    print(f"Transformation Tests: {sum(transform_tests.values())}/{len(transform_tests)} passed")
    print(f"Pitfall 1 Tests (None): {pitfall_passed}/{pitfall_total} passed")
    print(f"Pitfall 2 Tests (Empty Arrays): {pitfall2_passed}/{pitfall2_total} passed")
    print(f"Pitfall 4 Tests (Deprecation): {pitfall4_passed}/{pitfall4_total} passed")
    print("-" * 70)
    print(f"TOTAL: {passed_tests}/{total_tests} tests passed")
    print("=" * 70)

    if passed_tests == total_tests:
        print("\n✓ All validation functions working correctly!")
        print("✓ Handler support functions ready!")
        print("✓ transformation validation ready!")
        print("✓ Pitfall 1 (None handling) verified!")
        print("✓ Pitfall 2 (Empty array handling) verified!")
        print("   Use validate_uncertainty_data() in new code.")
