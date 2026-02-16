# molecule_validator.py

"""
Molecular validation module for the conversion pipeline.

This module handles validation of molecular structure data, dataset compatibility checking,
and atomic symbol/number conversions. It provides comprehensive validation with detailed
error reporting for both DFT and DMC datasets.

COMPLETE - HANDLER-ONLY ARCHITECTURE: All validation operations now require
dataset handlers. Legacy fallback mechanisms have been removed. Uses dataset handler
strategy pattern exclusively for all dataset-specific validation logic with comprehensive
handler-specific exception handling and improved error context.

PHASE 6 REFACTORED: All hardcoded dataset type references have been replaced with
dynamic registry-based lookups and feature queries. This enables:
- Zero-core-file-modification when adding new dataset types
- Dynamic dataset-specific validation via registry feature queries
- Automatic support for any dataset type registered in the registry
- Backward compatibility when registry is unavailable

Handler Integration Benefits:
- Centralized dataset-specific validation logic in handlers
- Consistent error handling through handler-specific exceptions
- Better separation of concerns between general and dataset-specific validation
- Improved testability with handler mocking capabilities
- Enhanced error context and recovery suggestions
- Handler is now REQUIRED - no fallback to legacy implementations

Breaking Changes (Handler Cleanup):
- All validation functions now require a valid handler instance
- Fallback to legacy implementations has been removed
- Handler parameter cannot be None - will raise ValueError if not provided
"""

import logging
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from milia_pipeline.config.config_accessors import (
    get_dataset_type,
    get_uncertainty_config,
    is_uncertainty_enabled,
)
from milia_pipeline.config.config_constants import HEAVY_ATOM_SYMBOLS_TO_Z
from milia_pipeline.config.config_containers import (
    DatasetConfig,
    create_dataset_config_from_global,
    create_filter_config_from_global,
    create_processing_config_from_global,
)
from milia_pipeline.exceptions import (
    DatasetSpecificHandlerError,
    HandlerError,
    HandlerNotAvailableError,
    HandlerOperationError,
    HandlerValidationError,
    MoleculeProcessingError,
    create_handler_error_context,
)
from milia_pipeline.handlers import DatasetHandler, create_dataset_handler

# ============================================================================
# PHASE 6: Registry Integration for Dynamic Validation
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
    (for validation operations). By deferring the registry import until first use,
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

        return True
    except ImportError as e:
        _REGISTRY_IMPORT_ERROR = str(e)
        _REGISTRY_AVAILABLE = False
        return False


def _get_available_dataset_types() -> list[str]:
    """
    Get list of available dataset types from registry or dynamic discovery.

    DYNAMIC APPROACH: Instead of hardcoded fallback, this function:
    1. First tries the registry (primary source of truth)
    2. If registry fails, dynamically discovers dataset implementations from filesystem
    3. Never uses hardcoded dataset type lists

    Returns:
        List[str]: Available dataset type names

    ADDED Phase 6: Dynamic dataset type list
    """
    _init_registry()

    if _REGISTRY_AVAILABLE and _registry_list_all is not None:
        try:
            return _registry_list_all()
        except Exception as e:
            logger.debug(f"Registry list_all() failed: {e}")

    # DYNAMIC FALLBACK: Discover dataset types from implementations directory
    try:
        from pathlib import Path

        # Find the implementations directory
        implementations_dir = Path(__file__).parent.parent / "datasets" / "implementations"
        if implementations_dir.exists():
            discovered_types = []
            for py_file in implementations_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                # Extract dataset name from filename (e.g., dft.py -> DFT, qm9.py -> QM9)
                dataset_name = py_file.stem.upper()
                # Exclude non-dataset modules
                if dataset_name not in ["BASE", "REGISTRY", "UTILS", "COMMON"]:
                    discovered_types.append(dataset_name)
            if discovered_types:
                logger.debug(f"Dynamically discovered dataset types: {discovered_types}")
                return discovered_types
    except Exception as e:
        logger.debug(f"Dynamic dataset type discovery failed: {e}")

    # Final fallback: return empty list with warning
    logger.warning(
        "No dataset types available - registry not initialized and dynamic discovery failed"
    )
    return []


def _is_dataset_type_registered(dataset_type: str) -> bool:
    """
    Check if a dataset type is registered in the registry or dynamically discovered.

    DYNAMIC APPROACH: Instead of hardcoded fallback check, this function:
    1. First tries the registry (primary source of truth)
    2. If registry fails, uses _get_available_dataset_types() which does dynamic discovery
    3. Never uses hardcoded dataset type lists

    Args:
        dataset_type: Name of the dataset type to check

    Returns:
        bool: True if dataset type is registered/discovered

    ADDED Phase 6: Dynamic type validation
    """
    _init_registry()

    if _REGISTRY_AVAILABLE and _registry_is_registered is not None:
        try:
            return _registry_is_registered(dataset_type)
        except Exception as e:
            logger.debug(f"Registry is_registered() failed for '{dataset_type}': {e}")

    # DYNAMIC FALLBACK: Check against dynamically discovered types
    available_types = _get_available_dataset_types()
    return dataset_type in available_types


def _get_dataset_feature(dataset_type: str, feature_name: str) -> bool:
    """
    Query a specific feature flag for a dataset type.

    Args:
        dataset_type: Name of the dataset type
        feature_name: Name of the feature to query (e.g., 'uncertainty_handling')

    Returns:
        bool: True if feature is enabled for this dataset type

    ADDED Phase 6: Feature-based queries replace hardcoded type checks
    """
    _init_registry()

    if _REGISTRY_AVAILABLE and _registry_get is not None:
        try:
            dataset_class = _registry_get(dataset_type)
            if hasattr(dataset_class, "features"):
                return getattr(dataset_class.features, feature_name, False)
        except Exception:
            pass

    # Legacy fallback - hardcoded feature matrix
    legacy_features = {
        "DFT": {
            "vibrational_analysis": True,
            "uncertainty_handling": False,
            "atomization_energy": True,
            "rotational_constants": True,
            "frequency_analysis": True,
            "orbital_analysis": False,
            "homo_lumo_gap": False,
            "mo_energies": False,
        },
        "DMC": {
            "vibrational_analysis": False,
            "uncertainty_handling": True,
            "atomization_energy": False,
            "rotational_constants": False,
            "frequency_analysis": False,
            "orbital_analysis": False,
            "homo_lumo_gap": False,
            "mo_energies": False,
        },
        "Wavefunction": {
            "vibrational_analysis": False,
            "uncertainty_handling": False,
            "atomization_energy": False,
            "rotational_constants": False,
            "frequency_analysis": False,
            "orbital_analysis": True,
            "homo_lumo_gap": True,
            "mo_energies": True,
        },
    }

    return legacy_features.get(dataset_type, {}).get(feature_name, False)


def _get_dataset_optional_properties(dataset_type: str) -> list[str]:
    """
    Get optional properties for a dataset type from registry.

    Args:
        dataset_type: Name of the dataset type

    Returns:
        List[str]: List of optional property names

    ADDED Phase 6: Dynamic optional properties
    """
    _init_registry()

    if _REGISTRY_AVAILABLE and _registry_get is not None:
        try:
            dataset_class = _registry_get(dataset_type)
            if hasattr(dataset_class, "get_optional_properties"):
                return dataset_class.get_optional_properties()
        except Exception:
            pass

    # Legacy fallback
    legacy_optional = {
        "DFT": ["freqs", "vibmodes", "rots", "dipoles"],
        "DMC": ["qmc_stats", "correlation_data"],
        "Wavefunction": ["mo_energies", "mo_occupations", "homo_lumo_gap_eV", "total_energy"],
    }

    return legacy_optional.get(dataset_type, [])


def _create_handler_specific_error(
    handler: DatasetHandler,
    message: str,
    operation: str,
    details: str,
    molecule_index: int | None = None,
    original_error: Exception | None = None,
) -> Exception:
    """
    Create a handler-specific error based on registry features.

    Uses dataset type features to determine which error class to use,
    replacing hardcoded handler class name checks.

    Args:
        handler: The dataset handler instance
        message: Error message
        operation: Operation that failed
        details: Additional error details
        molecule_index: Optional molecule index for error context
        original_error: Original exception if any

    Returns:
        Appropriate handler error instance

    ADDED Phase 6: Dynamic error creation replacing handler class name checks
    """
    dataset_type = handler.get_dataset_type()

    # Check if this dataset type has uncertainty handling (DMC-like)
    if _get_dataset_feature(dataset_type, "uncertainty_handling") or _get_dataset_feature(
        dataset_type, "vibrational_analysis"
    ):
        error = DatasetSpecificHandlerError(
            message=f"{dataset_type} {message}",
            dataset_type=dataset_type,
            operation=operation,
            details=details,
        )
    else:
        # Default to generic handler validation error
        error = HandlerValidationError(
            message=f"{dataset_type} {message}",
            handler_type=dataset_type,
            validation_type=operation,
            molecule_index=molecule_index,
            details=details,
        )

    if original_error is not None:
        error.__cause__ = original_error

    return error


def get_registry_status() -> dict[str, Any]:
    """
    Get current status of registry integration.

    Returns:
        Dict with registry availability and status information

    ADDED Phase 6: Registry status for diagnostics
    """
    _init_registry()

    return {
        "registry_available": _REGISTRY_AVAILABLE,
        "registry_initialized": _REGISTRY_INITIALIZED,
        "registry_import_error": _REGISTRY_IMPORT_ERROR,
        "available_dataset_types": _get_available_dataset_types(),
        "phase_6_complete": True,
        "features_available": [
            "uncertainty_handling",
            "vibrational_analysis",
            "atomization_energy",
            "rotational_constants",
            "frequency_analysis",
            "orbital_analysis",
            "homo_lumo_gap",
            "mo_energies",
        ],
    }


logger = logging.getLogger(__name__)


def validate_molecular_structure(
    atoms: np.ndarray,
    coordinates: np.ndarray,
    molecule_index: int,
    inchi: str,
    handler: DatasetHandler | None = None,
    raw_properties_dict: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
        Enhanced molecular structure validation with comprehensive error handling.

        HANDLER-ONLY (Complete): Requires dataset handler for all validation.
        No fallback to legacy implementations. Handler performs dataset-specific validation
        with enhanced exception handling and error context.

        PHASE 6: Error handling uses feature-based dynamic error creation.

        Args:
            atoms: Array of atomic numbers or symbols
            coordinates: Array of 3D coordinates, shape (n_atoms, 3)
            molecule_index: Index of the molecule for error reporting
            inchi: InChI string for error reporting
            handler: Dataset handler for dataset-specific validation (REQUIRED)
            raw_properties_dict: Optional complete raw properties dictionary for full validation

        Returns:
            Tuple[np.ndarray, np.ndarray]: Validated (atomic_numbers, coordinates)

        Raises:
            ValueError: If handler is None
            HandlerValidationError: If handler validation fails
            HandlerOperationError: If handler operation encounters errors
            MoleculeProcessingError: If general validation fails

        Note:
            Handler parameter is REQUIRED and cannot be None. Create handler using:
    ```python
            from milia_pipeline.handlers import create_dataset_handler
            from milia_pipeline.config.config_containers import (
                create_dataset_config_from_global,
                create_filter_config_from_global,
                create_processing_config_from_global
            )

            handler = create_dataset_handler(
                create_dataset_config_from_global(),
                create_filter_config_from_global(),
                create_processing_config_from_global(),
                logger
            )
    ```
    """

    def _handler_validate_molecular_structure(
        handler,
        atoms,
        coordinates,
        molecule_index,
        inchi,
        raw_properties_dict: dict[str, Any] | None = None,
    ):
        """
        Handler-based validation with enhanced exception handling.

        PHASE 6: Uses feature-based error creation instead of handler class name checks.

        If raw_properties_dict is provided, performs full molecule validation including
        all required properties (Etot, etc.). Otherwise, performs structure-only validation.
        """
        create_handler_error_context(
            handler.get_dataset_type(),
            "validate_molecular_structure",
            molecule_index,
            {"inchi": inchi},
        )

        try:
            # Use centralized validators first for basic structure validation
            from milia_pipeline.config.validators import (
                validate_molecular_structure as centralized_validator,
            )

            atomic_numbers, validated_coords = centralized_validator(
                atoms, coordinates, molecule_index, inchi
            )

            # Let handler perform additional dataset-specific validation
            # Use complete raw_properties_dict if available, otherwise create minimal dict
            if raw_properties_dict is not None:
                # FULL VALIDATION: Use complete data dictionary
                validation_dict = raw_properties_dict.copy()
                validation_dict["atoms"] = atomic_numbers  # Update with validated data
                validation_dict["coordinates"] = validated_coords  # Update with validated data
                logger.debug(
                    f"Molecule {molecule_index} ({inchi}): "
                    f"Performing full validation with {len(validation_dict)} properties"
                )
            else:
                # STRUCTURE-ONLY VALIDATION: Minimal dict
                validation_dict = {
                    "atoms": atomic_numbers,
                    "coordinates": validated_coords,
                    "inchi": inchi,
                }
                logger.debug(
                    f"Molecule {molecule_index} ({inchi}): "
                    f"Performing structure-only validation (no complete data available)"
                )

            # This will trigger dataset-specific validation logic
            try:
                handler.validate_molecule_data(validation_dict, molecule_index, inchi)
            except Exception as e:
                # ====================================================================
                # PHASE 6: Feature-based error creation (replaces handler class name checks)
                # ====================================================================
                raise _create_handler_specific_error(
                    handler=handler,
                    message=f"molecular structure validation failed: {str(e)}",
                    operation="validate_molecular_structure",
                    details=f"Handler validation failed for molecule {molecule_index}",
                    molecule_index=molecule_index,
                    original_error=e,
                )

            logger.debug(
                f"Molecule {molecule_index} passed {handler.get_dataset_type()} structural validation"
            )
            return atomic_numbers, validated_coords

        except HandlerError:
            # Re-raise handler errors as-is
            raise
        except ValueError as e:
            raise HandlerOperationError(
                message=f"Molecular structure validation failed: {str(e)}",
                handler_type=handler.get_dataset_type(),
                operation="validate_molecular_structure",
                molecule_index=molecule_index,
                recovery_suggestions=[
                    "Check molecular structure data quality",
                    "Verify atom and coordinate array consistency",
                    "Review input data format",
                ],
                details=str(e),
            ) from e
        except Exception as e:
            raise HandlerOperationError(
                message=f"Unexpected error in molecular structure validation: {str(e)}",
                handler_type=handler.get_dataset_type(),
                operation="validate_molecular_structure",
                molecule_index=molecule_index,
                recovery_suggestions=[
                    "Skip molecule and continue processing",
                    "Review molecular data integrity",
                    "Check for data corruption",
                ],
                details=f"Original error: {type(e).__name__}: {str(e)}",
            ) from e

    if handler is None:
        raise ValueError("Handler is required for molecular structure validation")

    return _handler_validate_molecular_structure(
        handler, atoms, coordinates, molecule_index, inchi, raw_properties_dict
    )


def convert_symbols_to_atomic_numbers(
    symbols: np.ndarray, molecule_index: int, inchi: str = "N/A"
) -> np.ndarray:
    """
    Enhanced atomic symbol to number conversion with comprehensive error handling.

    This function remains unchanged as it doesn't have dataset-specific logic.
    Enhanced with better error context for Handler-Based Pattern Development.

    Args:
        symbols: Array of atomic symbols
        molecule_index: Index of the molecule for error reporting
        inchi: InChI string for error reporting

    Returns:
        np.ndarray: Array of atomic numbers

    Raises:
        MoleculeProcessingError: If conversion fails
    """
    atomic_numbers = []
    unknown_symbols = set()

    # Create complete symbol mapping
    symbol_to_z = dict(HEAVY_ATOM_SYMBOLS_TO_Z)
    symbol_to_z["H"] = 1

    for symbol in symbols:
        symbol_str = str(symbol).strip()
        if symbol_str in symbol_to_z:
            atomic_numbers.append(symbol_to_z[symbol_str])
        else:
            unknown_symbols.add(symbol_str)

    if unknown_symbols:
        raise MoleculeProcessingError(
            molecule_index=molecule_index,
            inchi=inchi,
            message=f"Unknown atom symbols: {list(unknown_symbols)}",
            reason=f"Unknown atom symbols: {list(unknown_symbols)}",
            details=f"Cannot convert unknown atom symbols to atomic numbers. "
            f"Available symbols: {list(symbol_to_z.keys())[:10]}...",
        )

    return np.array(atomic_numbers, dtype=np.int64)


def check_dataset_compatibility(
    raw_properties_dict: dict[str, Any],
    dataset_type: str,
    molecule_index: int,
    inchi: str = "N/A",
    handler: DatasetHandler | None = None,
) -> bool:
    """
    HANDLER-ONLY (Complete): Dataset compatibility checking using handlers exclusively.

    Uses dataset handler strategy pattern with comprehensive exception handling.
    Legacy conditional logic has been removed - all validation now delegated to handlers.

    PHASE 6: Error handling uses feature-based dynamic error creation.

    Args:
        raw_properties_dict: Dictionary containing raw molecular data
        dataset_type: Type of dataset ('DFT' or 'DMC') - for logging/error reporting
        molecule_index: Index of the molecule for error reporting
        inchi: InChI string for error reporting
        handler: Dataset handler for dataset-specific validation (REQUIRED)

    Returns:
        bool: True if molecule is compatible with dataset requirements

    Raises:
        ValueError: If handler is None
        HandlerValidationError: If handler validation fails
        HandlerOperationError: If handler operation encounters errors
        MoleculeProcessingError: If general validation fails

    Note:
        The dataset_type parameter is retained for backward compatibility and error
        reporting but is not used for validation logic. Handler determines all
        dataset-specific validation rules.
    """

    def _handler_check_dataset_compatibility(
        handler, raw_properties_dict, dataset_type, molecule_index, inchi
    ):
        """
        Handler-based compatibility checking with enhanced exception handling.

        PHASE 6: Uses feature-based error creation instead of handler class name checks.
        """
        create_handler_error_context(
            handler.get_dataset_type(),
            "check_dataset_compatibility",
            molecule_index,
            {"inchi": inchi, "input_dataset_type": dataset_type},
        )

        try:
            # Verify handler matches expected dataset type
            handler_type = handler.get_dataset_type()
            if dataset_type != handler_type:
                logger.warning(
                    f"Dataset type mismatch: expected {dataset_type}, handler is {handler_type}"
                )

            # Let the handler perform comprehensive dataset-specific validation
            handler.validate_molecule_data(raw_properties_dict, molecule_index, inchi)

            # If no exception is raised, the molecule is compatible
            logger.debug(f"Molecule {molecule_index} passed {handler_type} compatibility check")
            return True

        except HandlerError:
            # Re-raise handler errors as-is
            raise
        except MoleculeProcessingError as e:
            # ====================================================================
            # PHASE 6: Feature-based error creation (replaces handler class name checks)
            # ====================================================================
            raise _create_handler_specific_error(
                handler=handler,
                message=f"compatibility validation failed: {e.message}",
                operation="check_dataset_compatibility",
                details=f"Molecule {molecule_index} failed validation: {str(e)}",
                molecule_index=molecule_index,
                original_error=e,
            )
        except Exception as e:
            # Convert other exceptions to handler operation errors
            raise HandlerOperationError(
                message=f"Dataset compatibility check failed: {str(e)}",
                handler_type=handler.get_dataset_type(),
                operation="check_dataset_compatibility",
                molecule_index=molecule_index,
                recovery_suggestions=[
                    "Verify input data format",
                    "Check required properties are present",
                    "Review dataset configuration",
                ],
                details=f"Handler {handler.__class__.__name__} compatibility check error: {str(e)}",
            ) from e

    if handler is None:
        raise ValueError("Handler is required for dataset compatibility checking")

    return _handler_check_dataset_compatibility(
        handler, raw_properties_dict, dataset_type, molecule_index, inchi
    )


def validate_uncertainty_data(
    pyg_data: Data, molecule_index: int | str, smiles: str, handler: DatasetHandler | None = None
) -> float | None:
    """
    HANDLER-ONLY (Complete): Uncertainty validation requiring dataset handler.

    Uses dataset handler for dataset-aware uncertainty validation with comprehensive
    error handling. Legacy global config access has been removed.

    PHASE 6: Uses feature-based check instead of hardcoded DMC type check.
    Only datasets with uncertainty_handling feature and uncertainty enabled
    will perform validation.

    Args:
        pyg_data: PyG Data object containing uncertainty information
        molecule_index: Index of the molecule for error reporting
        smiles: SMILES string for error reporting
        handler: Dataset handler for dataset-specific validation (REQUIRED)

    Returns:
        Optional[float]: Validated uncertainty value or None (if uncertainty not supported/enabled)

    Raises:
        ValueError: If handler is None
        DatasetSpecificHandlerError: If dataset-specific uncertainty validation fails
        HandlerOperationError: If handler operation encounters errors

    Note:
        - Returns None for datasets without uncertainty_handling feature
        - Returns None if uncertainty is not enabled in handler config
        - Only datasets with uncertainty_handling feature perform validation
    """

    def _handler_validate_uncertainty_data(handler, pyg_data, molecule_index, smiles):
        """
        Handler-based uncertainty validation with enhanced exception handling.

        PHASE 6: Uses feature-based check instead of hardcoded DMC type check.
        """
        create_handler_error_context(
            handler.get_dataset_type(),
            "validate_uncertainty_data",
            molecule_index,
            {"smiles": smiles},
        )

        # ====================================================================
        # PHASE 6: Feature-based uncertainty check (replaces DMC type check)
        # ====================================================================
        dataset_type = handler.get_dataset_type()

        # Check if dataset type supports uncertainty handling
        if not _get_dataset_feature(dataset_type, "uncertainty_handling"):
            logger.debug(
                f"Uncertainty validation skipped for {dataset_type} dataset (no uncertainty_handling feature)"
            )
            return None

        # Check if uncertainty is enabled in handler config
        if not getattr(handler.dataset_config, "is_uncertainty_enabled", False):
            logger.debug(
                f"Uncertainty validation skipped for {dataset_type} dataset (uncertainty not enabled)"
            )
            return None

        try:
            uncertainty_config = handler.dataset_config.uncertainty_config
            if not uncertainty_config:
                return None

            uncertainty_field = uncertainty_config.get("uncertainty_field_name", "std")
            uncertainty_value = None

            # Try to get uncertainty from various sources
            if hasattr(pyg_data, "uncertainty"):
                uncertainty_value = pyg_data.uncertainty
            elif hasattr(pyg_data, uncertainty_field):
                uncertainty_value = getattr(pyg_data, uncertainty_field)

            if uncertainty_value is None:
                return None

            # Use handler's property processing for uncertainty validation
            try:
                validated_uncertainty = handler.process_property_value(
                    uncertainty_field, uncertainty_value, molecule_index, smiles
                )

                if validated_uncertainty is not None:
                    if isinstance(validated_uncertainty, (torch.Tensor, np.ndarray)):
                        if hasattr(validated_uncertainty, "item"):
                            uncertainty_scalar = float(validated_uncertainty.item())
                        else:
                            uncertainty_scalar = float(validated_uncertainty.flat[0])
                    else:
                        uncertainty_scalar = float(validated_uncertainty)

                    logger.debug(
                        f"Validated uncertainty {uncertainty_scalar} for molecule {molecule_index}"
                    )
                    return round(uncertainty_scalar, 6)

            except Exception as e:
                # Use feature-based error for datasets with uncertainty_handling
                raise DatasetSpecificHandlerError(
                    message=f"{dataset_type} uncertainty validation failed: {str(e)}",
                    dataset_type=dataset_type,
                    operation="validate_uncertainty_data",
                    property_name=uncertainty_field,
                    details=f"Failed to validate uncertainty for field '{uncertainty_field}' in molecule {molecule_index}",
                ) from e

        except HandlerError:
            # Re-raise handler errors as-is
            raise
        except Exception as e:
            raise HandlerOperationError(
                message=f"Uncertainty validation operation failed: {str(e)}",
                handler_type=handler.get_dataset_type(),
                operation="validate_uncertainty_data",
                molecule_index=molecule_index,
                recovery_suggestions=[
                    "Check uncertainty data format",
                    "Verify uncertainty field configuration",
                    f"Review {dataset_type} dataset settings",
                ],
                details=f"Uncertainty validation error: {str(e)}",
            ) from e

        return None

    if handler is None:
        raise ValueError("Handler is required for uncertainty data validation")

    return _handler_validate_uncertainty_data(handler, pyg_data, molecule_index, smiles)


def validate_pyg_data_completeness(
    pyg_data: Data,
    dataset_type: str,
    molecule_index: int | None = None,
    handler: DatasetHandler | None = None,
) -> dict[str, bool]:
    """
    HANDLER-ONLY (Complete): PyG data completeness validation using handlers exclusively.

    Uses dataset handler for dataset-aware completeness validation with comprehensive
    error handling. Legacy validation logic has been consolidated into handlers.

    PHASE 6: Uses feature-based validation enhancements instead of hardcoded type checks.

    Args:
        pyg_data: PyG Data object to validate
        dataset_type: Type of dataset ('DFT' or 'DMC') - for backward compatibility
        molecule_index: Index of the molecule for error reporting
        handler: Dataset handler for dataset-specific validation (REQUIRED)

    Returns:
        Dict[str, bool]: Dictionary of validation results including:
            - has_basic_structure: Basic PyG structure present
            - has_coordinates: Valid 3D coordinates
            - has_atomic_numbers: Valid atomic numbers
            - has_target_values: Target energy values present
            - has_structural_features: Atom/bond features present
            - has_handler_required_props: Handler-required properties present
            - handler_type: Type of handler used
            - missing_handler_props: List of missing properties (if any)
            - has_uncertainty: Uncertainty data (for uncertainty-enabled datasets)
            - has_vibrational_data: Vibrational data (for vibrational-enabled datasets)
            - has_atomization_energy: Atomization energy (for atomization-enabled datasets)
            - has_orbital_data: Orbital data (for orbital-enabled datasets)

    Raises:
        ValueError: If handler is None
        HandlerOperationError: If validation operation fails
        ValidationError: If general validation fails
    """

    def _handler_validate_pyg_data_completeness(handler, pyg_data, dataset_type, molecule_index):
        """
        Handler-based completeness validation with enhanced exception handling.

        PHASE 6: Uses feature-based validation enhancements instead of hardcoded type checks.
        """
        create_handler_error_context(
            handler.get_dataset_type(), "validate_pyg_data_completeness", molecule_index
        )

        try:
            # Start with general validation
            validation_results = _original_validate_pyg_data_completeness(
                pyg_data, dataset_type, molecule_index
            )

            # Add handler-specific validation
            required_props = handler.get_required_properties()

            # Check if PyG data has the properties required by the handler
            missing_handler_props = []
            for prop in required_props:
                if not hasattr(pyg_data, prop) or getattr(pyg_data, prop) is None:
                    missing_handler_props.append(prop)

            # Add handler-specific validation results
            validation_results["has_handler_required_props"] = len(missing_handler_props) == 0
            validation_results["missing_handler_props"] = missing_handler_props
            validation_results["handler_type"] = handler.get_dataset_type()

            # ========================================================================
            # PHASE 6: Feature-based validation enhancements (replaces if/elif)
            # ========================================================================
            handler_dataset_type = handler.get_dataset_type()

            # Uncertainty handling validation (for datasets with uncertainty_handling feature)
            if _get_dataset_feature(handler_dataset_type, "uncertainty_handling"):
                if getattr(handler.dataset_config, "is_uncertainty_enabled", False):
                    uncertainty_config = getattr(handler.dataset_config, "uncertainty_config", {})
                    if uncertainty_config:
                        uncertainty_field = uncertainty_config.get("uncertainty_field_name", "std")
                        has_uncertainty = (
                            hasattr(pyg_data, "uncertainty")
                            or hasattr(pyg_data, uncertainty_field)
                            or hasattr(pyg_data, "uncertainty_metadata")
                        )
                        validation_results["has_uncertainty"] = has_uncertainty

                        if not has_uncertainty:
                            logger.warning(
                                f"{handler_dataset_type} molecule {molecule_index} missing uncertainty data "
                                "but uncertainty is enabled"
                            )

            # Vibrational analysis validation (for datasets with vibrational_analysis feature)
            if _get_dataset_feature(handler_dataset_type, "vibrational_analysis"):
                has_vibrational_data = hasattr(pyg_data, "freqs") and hasattr(pyg_data, "vibmodes")
                validation_results["has_vibrational_data"] = has_vibrational_data

                # Check for atomization energy if atomization_energy feature is enabled
                if _get_dataset_feature(handler_dataset_type, "atomization_energy"):
                    has_atomization_energy = hasattr(pyg_data, "atomization_energy")
                    validation_results["has_atomization_energy"] = has_atomization_energy

            # Orbital analysis validation (for datasets with orbital_analysis feature)
            if _get_dataset_feature(handler_dataset_type, "orbital_analysis"):
                has_orbital_data = hasattr(pyg_data, "mo_energies") or hasattr(
                    pyg_data, "homo_lumo_gap_eV"
                )
                validation_results["has_orbital_data"] = has_orbital_data

            logger.debug(
                f"Handler-based validation for molecule {molecule_index}: {validation_results}"
            )
            return validation_results

        except HandlerError:
            # Re-raise handler errors as-is
            raise
        except Exception as e:
            raise HandlerOperationError(
                message=f"PyG data completeness validation failed: {str(e)}",
                handler_type=handler.get_dataset_type(),
                operation="validate_pyg_data_completeness",
                molecule_index=molecule_index,
                recovery_suggestions=[
                    "Check PyG data structure",
                    "Verify required attributes are present",
                    "Review data creation process",
                ],
                details=f"Completeness validation error: {str(e)}",
            ) from e

    if handler is None:
        raise ValueError("Handler is required for PyG data completeness validation")

    return _handler_validate_pyg_data_completeness(handler, pyg_data, dataset_type, molecule_index)


def _original_validate_pyg_data_completeness(
    pyg_data: Data, dataset_type: str, molecule_index: int | None = None
) -> dict[str, bool]:
    """
    Original PyG data completeness validation for fallback compatibility.

    PHASE 6: Uses feature-based metadata validation instead of hardcoded DMC check.

    This preserves the original validation logic for backward compatibility.
    """
    validation_results = {
        "has_basic_structure": True,
        "has_coordinates": True,
        "has_atomic_numbers": True,
        "has_target_values": True,
        "has_metadata": True,
        "has_structural_features": False,
        "structural_features_valid": False,
    }

    # Basic structure validation
    required_basic = ["z", "pos", "num_nodes"]
    for attr in required_basic:
        if not hasattr(pyg_data, attr) or getattr(pyg_data, attr) is None:
            validation_results["has_basic_structure"] = False
            break

    # Coordinates validation
    if hasattr(pyg_data, "pos") and pyg_data.pos is not None:
        if isinstance(pyg_data.pos, torch.Tensor):
            if torch.any(torch.isnan(pyg_data.pos)) or torch.any(torch.isinf(pyg_data.pos)):
                validation_results["has_coordinates"] = False
    else:
        validation_results["has_coordinates"] = False

    # Atomic numbers validation
    if hasattr(pyg_data, "z") and pyg_data.z is not None:
        if isinstance(pyg_data.z, torch.Tensor):
            if torch.any(pyg_data.z < 1) or torch.any(pyg_data.z > 118):
                validation_results["has_atomic_numbers"] = False
    else:
        validation_results["has_atomic_numbers"] = False

    # Target values validation
    if not hasattr(pyg_data, "y") or pyg_data.y is None:
        validation_results["has_target_values"] = False
    elif isinstance(pyg_data.y, torch.Tensor):
        if torch.any(torch.isnan(pyg_data.y)) or torch.any(torch.isinf(pyg_data.y)):
            validation_results["has_target_values"] = False

    # Enhanced structural features validation
    has_atom_features = hasattr(pyg_data, "x") and pyg_data.x is not None
    has_bond_features = hasattr(pyg_data, "edge_attr") and pyg_data.edge_attr is not None

    if has_atom_features or has_bond_features:
        validation_results["has_structural_features"] = True

        # Validate structural features quality
        features_valid = True
        if has_atom_features and isinstance(pyg_data.x, torch.Tensor):
            if torch.any(torch.isnan(pyg_data.x)) or torch.any(torch.isinf(pyg_data.x)):
                features_valid = False

        if has_bond_features and isinstance(pyg_data.edge_attr, torch.Tensor):
            if torch.any(torch.isnan(pyg_data.edge_attr)) or torch.any(
                torch.isinf(pyg_data.edge_attr)
            ):
                features_valid = False

        validation_results["structural_features_valid"] = features_valid

    # ========================================================================
    # PHASE 6: Feature-based metadata validation (replaces DMC type check)
    # ========================================================================
    # Check if dataset type supports uncertainty handling
    if _get_dataset_feature(dataset_type, "uncertainty_handling") and is_uncertainty_enabled():
        uncertainty_config = get_uncertainty_config()
        uncertainty_field = uncertainty_config.get("uncertainty_field_name", "std")
        has_uncertainty = (
            hasattr(pyg_data, "uncertainty")
            or hasattr(pyg_data, uncertainty_field)
            or hasattr(pyg_data, "uncertainty_metadata")
        )
        if not has_uncertainty:
            validation_results["has_metadata"] = False

    # Basic metadata
    basic_metadata = ["original_mol_idx", "dataset_type"]
    for attr in basic_metadata:
        if not hasattr(pyg_data, attr):
            validation_results["has_metadata"] = False
            break

    return validation_results


# ==========================================
# ENHANCED HANDLER INTEGRATION FUNCTIONS
# ==========================================


def create_validator_with_handler(dataset_config: DatasetConfig | None = None) -> DatasetHandler:
    """
    Creates a dataset handler for validation purposes with enhanced error handling.

    Enhanced with handler-specific exception handling and better error context.

    Args:
        dataset_config: Optional dataset configuration. If None, uses global config.

    Returns:
        DatasetHandler: Handler instance for validation operations

    Raises:
        HandlerNotAvailableError: If handler cannot be created
    """
    try:
        if dataset_config is None:
            dataset_config = create_dataset_config_from_global()

        filter_config = create_filter_config_from_global()
        processing_config = create_processing_config_from_global()

        handler = create_dataset_handler(dataset_config, filter_config, processing_config, logger)
        logger.debug(f"Created {handler.get_dataset_type()} handler for validation")
        return handler

    except Exception as e:
        raise HandlerNotAvailableError(
            message=f"Failed to create validation handler: {str(e)}",
            requested_dataset_type=dataset_config.dataset_type if dataset_config else "unknown",
            details=f"Handler creation failed: {str(e)}",
        ) from e


def validate_molecule_with_handler(
    raw_properties_dict: dict[str, Any],
    molecule_index: int,
    inchi: str = "N/A",
    handler: DatasetHandler | None = None,
) -> bool:
    """
    High-level molecule validation using dataset handlers with comprehensive error handling.

    HANDLER-ONLY (Complete): Comprehensive validation requiring dataset handler.
    If handler is None, creates one from global config using explicit handler creation pattern.

    PHASE 6: Uses feature-based error creation instead of handler class name checks.

    This function demonstrates the complete handler-based validation workflow with three stages:
    1. Basic structure validation
    2. Dataset compatibility validation
    3. Handler-specific validation

    Args:
        raw_properties_dict: Dictionary containing raw molecular data
        molecule_index: Index of the molecule for error reporting
        inchi: InChI string for error reporting
        handler: Dataset handler. If None, creates one from global config. (Default: None)

    Returns:
        bool: True if molecule passes all validation checks

    Raises:
        HandlerNotAvailableError: If handler cannot be created from global config
        HandlerValidationError: If validation fails
        HandlerOperationError: If handler operation fails
        MoleculeProcessingError: If general validation fails

    Example:
        >>> # With explicit handler
        >>> handler = create_dataset_handler(dataset_config, filter_config,
        ...                                  processing_config, logger)
        >>> is_valid = validate_molecule_with_handler(raw_data, 0, inchi, handler)

        >>> # With automatic handler creation
        >>> is_valid = validate_molecule_with_handler(raw_data, 0, inchi)  # Creates handler
    """
    if handler is None:
        try:
            # Create configs
            dataset_config = create_dataset_config_from_global()
            filter_config = create_filter_config_from_global()
            processing_config = create_processing_config_from_global()

            # Create handler (already validated)
            handler = create_dataset_handler(
                dataset_config, filter_config, processing_config, logger
            )
        except Exception as e:
            raise HandlerNotAvailableError(
                message=f"Cannot create handler for molecule validation: {str(e)}",
                requested_dataset_type="from_global_config",
                details=f"Handler creation failed during molecule validation: {str(e)}",
            ) from e

    # Comprehensive validation using handler
    validation_errors = []
    dataset_type = handler.get_dataset_type()

    try:
        # 1. Basic structure validation
        atoms = raw_properties_dict.get("atoms")
        coordinates = raw_properties_dict.get("coordinates")

        if atoms is not None and coordinates is not None:
            try:
                validate_molecular_structure(
                    atoms, coordinates, molecule_index, inchi, handler=handler
                )
            except HandlerError as e:
                validation_errors.append(f"Structure validation: {str(e)}")
                raise
            except Exception as e:
                validation_errors.append(f"Structure validation: {str(e)}")
                raise HandlerOperationError(
                    message=f"Structural validation failed: {str(e)}",
                    handler_type=dataset_type,
                    operation="validate_molecular_structure",
                    molecule_index=molecule_index,
                    details=str(e),
                ) from e

        # 2. Dataset compatibility validation
        try:
            check_dataset_compatibility(
                raw_properties_dict, dataset_type, molecule_index, inchi, handler
            )
        except HandlerError as e:
            validation_errors.append(f"Compatibility validation: {str(e)}")
            raise
        except Exception as e:
            validation_errors.append(f"Compatibility validation: {str(e)}")
            raise HandlerOperationError(
                message=f"Compatibility validation failed: {str(e)}",
                handler_type=dataset_type,
                operation="check_dataset_compatibility",
                molecule_index=molecule_index,
                details=str(e),
            ) from e

        # 3. Handler-specific validation
        try:
            handler.validate_molecule_data(raw_properties_dict, molecule_index, inchi)
        except Exception as e:
            validation_errors.append(f"Handler validation: {str(e)}")
            # ====================================================================
            # PHASE 6: Feature-based error creation (replaces handler class name checks)
            # ====================================================================
            raise _create_handler_specific_error(
                handler=handler,
                message=f"handler validation failed: {str(e)}",
                operation="validate_molecule_data",
                details=f"Handler-specific validation failed for molecule {molecule_index}",
                molecule_index=molecule_index,
                original_error=e,
            )

        logger.debug(f"Molecule {molecule_index} passed comprehensive {dataset_type} validation")
        return True

    except HandlerError:
        # Re-raise handler errors as-is with context
        logger.error(
            f"Handler validation failed for molecule {molecule_index}: {validation_errors}"
        )
        raise
    except MoleculeProcessingError as e:
        # Convert to handler validation error for better context
        raise HandlerValidationError(
            message=f"Molecule validation failed: {e.message}",
            handler_type=dataset_type,
            validation_type="comprehensive",
            molecule_index=molecule_index,
            failed_validations=validation_errors,
            details=f"Comprehensive validation failed: {str(e)}",
        ) from e
    except Exception as e:
        # Convert unexpected errors to handler operation errors
        raise HandlerOperationError(
            message=f"Comprehensive validation failed: {str(e)}",
            handler_type=dataset_type,
            operation="validate_molecule_with_handler",
            molecule_index=molecule_index,
            recovery_suggestions=[
                "Check molecular data integrity",
                "Verify handler configuration",
                "Review validation parameters",
                "Skip molecule and continue processing",
            ],
            details=f"Validation with {handler.__class__.__name__} failed: {str(e)}",
        ) from e


def get_validation_summary(validation_results: dict[str, bool]) -> str:
    """
    Generates a human-readable summary of validation results with enhanced formatting.

    PHASE 6: Uses feature-based recommendations instead of hardcoded type checks.

    Enhanced with handler-specific result interpretation and better formatting.

    Args:
        validation_results: Dictionary of validation results from validate_pyg_data_completeness

    Returns:
        str: Human-readable validation summary
    """
    passed_checks = sum(1 for v in validation_results.values() if isinstance(v, bool) and v)
    total_checks = sum(1 for v in validation_results.values() if isinstance(v, bool))

    # Determine overall status
    overall_status = "PASS" if passed_checks == total_checks else "PARTIAL"
    if passed_checks == 0:
        overall_status = "FAIL"

    summary_lines = [
        f"Validation Summary: {passed_checks}/{total_checks} checks passed ({overall_status})",
        "=" * 60,
    ]

    # Handler-specific information
    if "handler_type" in validation_results:
        summary_lines.append(f"Handler Type: {validation_results['handler_type']}")
        summary_lines.append("")

    # Categorize results
    critical_checks = [
        "has_basic_structure",
        "has_coordinates",
        "has_atomic_numbers",
        "has_target_values",
    ]
    handler_checks = [
        "has_handler_required_props",
        "has_uncertainty",
        "has_vibrational_data",
        "has_atomization_energy",
        "has_orbital_data",
    ]
    feature_checks = ["has_structural_features", "structural_features_valid"]
    metadata_checks = ["has_metadata"]

    def format_check_group(check_names, group_title):
        group_lines = []
        group_passed = 0
        group_total = 0

        for check_name in check_names:
            if check_name in validation_results and isinstance(
                validation_results[check_name], bool
            ):
                result = validation_results[check_name]
                status = "✓ PASS" if result else "✗ FAIL"
                group_lines.append(f"    {check_name}: {status}")
                group_total += 1
                if result:
                    group_passed += 1

        if group_lines:
            group_status = f"({group_passed}/{group_total})"
            return [f"  {group_title} {group_status}:"] + group_lines + [""]
        return []

    # Add categorized results
    summary_lines.extend(format_check_group(critical_checks, "Critical Checks"))
    summary_lines.extend(format_check_group(handler_checks, "Handler-Specific Checks"))
    summary_lines.extend(format_check_group(feature_checks, "Feature Checks"))
    summary_lines.extend(format_check_group(metadata_checks, "Metadata Checks"))

    # Add missing properties information
    if "missing_handler_props" in validation_results:
        missing_props = validation_results["missing_handler_props"]
        if isinstance(missing_props, list) and missing_props:
            summary_lines.extend(["  Missing Handler Properties:", f"    {missing_props}", ""])

    # ========================================================================
    # PHASE 6: Feature-based recommendations (replaces type-specific checks)
    # ========================================================================
    recommendations = []
    handler_type = validation_results.get("handler_type", "")

    if not validation_results.get("has_basic_structure", True):
        recommendations.append("• Ensure PyG data has basic structure (z, pos, num_nodes)")
    if not validation_results.get("has_target_values", True):
        recommendations.append("• Add target values (y) to PyG data")
    if not validation_results.get("has_handler_required_props", True):
        recommendations.append("• Verify all handler-required properties are present")

    # Uncertainty recommendations (for datasets with uncertainty_handling feature)
    if handler_type and _get_dataset_feature(handler_type, "uncertainty_handling"):
        if not validation_results.get("has_uncertainty", True):
            recommendations.append(f"• Add uncertainty data for {handler_type} dataset")

    # Vibrational recommendations (for datasets with vibrational_analysis feature)
    if handler_type and _get_dataset_feature(handler_type, "vibrational_analysis"):
        if not validation_results.get("has_vibrational_data", True):
            recommendations.append(
                f"• Consider adding vibrational data for enhanced {handler_type} analysis"
            )
        if not validation_results.get("has_atomization_energy", True):
            if _get_dataset_feature(handler_type, "atomization_energy"):
                recommendations.append(
                    f"• Consider adding atomization energy for {handler_type} dataset"
                )

    # Orbital recommendations (for datasets with orbital_analysis feature)
    if handler_type and _get_dataset_feature(handler_type, "orbital_analysis"):
        if not validation_results.get("has_orbital_data", True):
            recommendations.append(f"• Consider adding orbital data for {handler_type} analysis")

    if recommendations:
        summary_lines.extend(["Recommendations:"] + recommendations + [""])

    return "\n".join(summary_lines)


def validate_molecule_legacy(
    raw_properties_dict: dict[str, Any], molecule_index: int, inchi: str = "N/A"
) -> bool:
    """
    Legacy validation function for backward compatibility with enhanced error handling.

    Enhanced with handler-specific exception handling while maintaining compatibility.

    This function maintains the original interface while using the new handler-based system.

    Args:
        raw_properties_dict: Dictionary containing raw molecular data
        molecule_index: Index of the molecule for error reporting
        inchi: InChI string for error reporting

    Returns:
        bool: True if molecule passes validation
    """
    try:
        dataset_type = get_dataset_type()
        # Create configs
        dataset_config = create_dataset_config_from_global()
        filter_config = create_filter_config_from_global()
        processing_config = create_processing_config_from_global()

        # Create handler (already validated)
        handler = create_dataset_handler(dataset_config, filter_config, processing_config, logger)

        return check_dataset_compatibility(
            raw_properties_dict, dataset_type, molecule_index, inchi, handler
        )

    except HandlerError as e:
        logger.error(f"Handler-based legacy validation failed for molecule {molecule_index}: {e}")
        return False
    except Exception as e:
        logger.error(f"Legacy validation failed for molecule {molecule_index}: {e}")
        return False


def get_dataset_requirements(
    dataset_type: str, handler: DatasetHandler | None = None
) -> dict[str, list[str]]:
    """
    Gets dataset-specific requirements using handlers with enhanced error handling.

    PHASE 6: Uses feature-based requirement categorization instead of hardcoded type checks.

    Enhanced with handler-specific exception handling and better requirement categorization.

    Args:
        dataset_type: Type of dataset ('DFT' or 'DMC') - for backward compatibility
        handler: Optional dataset handler. If None, creates one from global config.

    Returns:
        Dict[str, List[str]]: Dictionary containing required properties by category

    Raises:
        HandlerNotAvailableError: If handler cannot be created
    """
    if handler is None:
        try:
            # Create configs
            dataset_config = create_dataset_config_from_global()
            filter_config = create_filter_config_from_global()
            processing_config = create_processing_config_from_global()

            # Create handler (already validated)
            handler = create_dataset_handler(
                dataset_config, filter_config, processing_config, logger
            )
        except Exception as e:
            raise HandlerNotAvailableError(
                message=f"Cannot create handler for requirements query: {str(e)}",
                requested_dataset_type=dataset_type,
                details=f"Handler creation failed while getting dataset requirements: {str(e)}",
            ) from e

    try:
        requirements = {
            "dataset_type": handler.get_dataset_type(),
            "required_properties": handler.get_required_properties(),
            "common_properties": handler.get_common_required_properties(),
            "handler_class": handler.__class__.__name__,
        }

        # ========================================================================
        # PHASE 6: Feature-based requirement categorization (replaces if/elif)
        # ========================================================================
        handler_dataset_type = handler.get_dataset_type()

        # Uncertainty requirements (uncertainty_handling feature)
        if _get_dataset_feature(handler_dataset_type, "uncertainty_handling"):
            if getattr(handler.dataset_config, "is_uncertainty_enabled", False):
                uncertainty_config = getattr(handler.dataset_config, "uncertainty_config", {})
                if uncertainty_config:
                    uncertainty_field = uncertainty_config.get("uncertainty_field_name", "std")
                    requirements["uncertainty_properties"] = [uncertainty_field]
                    requirements["uncertainty_enabled"] = True
                else:
                    requirements["uncertainty_enabled"] = False
            else:
                requirements["uncertainty_enabled"] = False
        else:
            requirements["uncertainty_enabled"] = False

        # Vibrational requirements (vibrational_analysis feature)
        if _get_dataset_feature(handler_dataset_type, "vibrational_analysis"):
            requirements["optional_properties"] = _get_dataset_optional_properties(
                handler_dataset_type
            )
            if _get_dataset_feature(handler_dataset_type, "atomization_energy"):
                requirements["derived_properties"] = ["atomization_energy"]

        # Orbital requirements (orbital_analysis feature)
        if _get_dataset_feature(handler_dataset_type, "orbital_analysis"):
            requirements["orbital_properties"] = _get_dataset_optional_properties(
                handler_dataset_type
            )

        # Add validation requirements
        requirements["structural_requirements"] = ["atoms", "coordinates"]
        requirements["energy_requirements"] = ["Etot"]

        logger.debug(
            f"Retrieved requirements for {handler.get_dataset_type()} dataset: {len(requirements['required_properties'])} properties"
        )
        return requirements

    except Exception as e:
        raise HandlerOperationError(
            message=f"Failed to get dataset requirements: {str(e)}",
            handler_type=handler.get_dataset_type() if handler else dataset_type,
            operation="get_dataset_requirements",
            details=f"Requirements query failed: {str(e)}",
        ) from e


def create_validation_context(
    handler: DatasetHandler, molecule_index: int, inchi: str = "N/A"
) -> dict[str, Any]:
    """
    Creates a comprehensive validation context for error reporting and debugging.

    PHASE 6: Uses feature-based context addition instead of hardcoded type checks.

    New function for enhanced error context and debugging support.

    Args:
        handler: Dataset handler instance
        molecule_index: Index of the molecule being validated
        inchi: InChI string for identification

    Returns:
        Dict[str, Any]: Comprehensive validation context
    """
    context = {
        "molecule_index": molecule_index,
        "inchi": inchi,
        "handler_type": handler.get_dataset_type(),
        "handler_class": handler.__class__.__name__,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "required_properties": handler.get_required_properties(),
        "validation_config": {
            "uncertainty_enabled": getattr(handler.dataset_config, "is_uncertainty_enabled", False),
            "dataset_type": handler.get_dataset_type(),
        },
    }

    # ========================================================================
    # PHASE 6: Feature-based context addition (replaces DMC type check)
    # ========================================================================
    dataset_type = handler.get_dataset_type()

    # Add uncertainty config for datasets with uncertainty_handling feature
    if _get_dataset_feature(dataset_type, "uncertainty_handling"):
        if hasattr(handler.dataset_config, "uncertainty_config"):
            context["uncertainty_config"] = handler.dataset_config.uncertainty_config

    # Add vibrational context for datasets with vibrational_analysis feature
    if _get_dataset_feature(dataset_type, "vibrational_analysis"):
        context["supports_vibrational_analysis"] = True
        context["supports_atomization_energy"] = _get_dataset_feature(
            dataset_type, "atomization_energy"
        )

    # Add orbital context for datasets with orbital_analysis feature
    if _get_dataset_feature(dataset_type, "orbital_analysis"):
        context["supports_orbital_analysis"] = True

    return context


def validate_with_detailed_feedback(
    raw_properties_dict: dict[str, Any],
    molecule_index: int,
    inchi: str = "N/A",
    handler: DatasetHandler | None = None,
) -> dict[str, Any]:
    """
    Performs validation with detailed feedback and diagnostic information.

    PHASE 6: Includes registry integration status in results.

    New function for enhanced validation reporting and debugging.

    Args:
        raw_properties_dict: Dictionary containing raw molecular data
        molecule_index: Index of the molecule for error reporting
        inchi: InChI string for error reporting
        handler: Optional dataset handler

    Returns:
        Dict[str, Any]: Detailed validation results and feedback
    """
    if handler is None:
        # Create configs
        dataset_config = create_dataset_config_from_global()
        filter_config = create_filter_config_from_global()
        processing_config = create_processing_config_from_global()

        # Create handler (already validated)
        handler = create_dataset_handler(dataset_config, filter_config, processing_config, logger)

    validation_context = create_validation_context(handler, molecule_index, inchi)
    detailed_results = {
        "context": validation_context,
        "validation_passed": False,
        "validation_errors": [],
        "validation_warnings": [],
        "validation_steps": {},
        "recovery_suggestions": [],
    }

    validation_steps = [
        (
            "structure_validation",
            lambda: validate_molecular_structure(
                raw_properties_dict.get("atoms"),
                raw_properties_dict.get("coordinates"),
                molecule_index,
                inchi,
                handler,
            ),
        ),
        (
            "compatibility_validation",
            lambda: check_dataset_compatibility(
                raw_properties_dict, handler.get_dataset_type(), molecule_index, inchi, handler
            ),
        ),
        (
            "handler_validation",
            lambda: handler.validate_molecule_data(raw_properties_dict, molecule_index, inchi),
        ),
    ]

    for step_name, validation_func in validation_steps:
        try:
            result = validation_func()
            detailed_results["validation_steps"][step_name] = {
                "passed": True,
                "result": result,
                "error": None,
            }
        except Exception as e:
            detailed_results["validation_steps"][step_name] = {
                "passed": False,
                "result": None,
                "error": str(e),
                "error_type": type(e).__name__,
            }
            detailed_results["validation_errors"].append(f"{step_name}: {str(e)}")

            # Add specific recovery suggestions based on error type
            if isinstance(e, HandlerValidationError):
                detailed_results["recovery_suggestions"].extend(e.failed_validations)
            elif isinstance(e, HandlerOperationError):
                detailed_results["recovery_suggestions"].extend(e.recovery_suggestions)

    # Overall validation status
    detailed_results["validation_passed"] = all(
        step["passed"] for step in detailed_results["validation_steps"].values()
    )

    # Add summary statistics
    passed_steps = sum(
        1 for step in detailed_results["validation_steps"].values() if step["passed"]
    )
    total_steps = len(detailed_results["validation_steps"])
    detailed_results["summary"] = f"{passed_steps}/{total_steps} validation steps passed"

    # ========================================================================
    # PHASE 6: Add registry integration status to results
    # ========================================================================
    detailed_results["registry_status"] = get_registry_status()

    return detailed_results
