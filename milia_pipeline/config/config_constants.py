# config_constants.py - Enhanced for Handler-Based Pattern Development & Transformation System Support

"""
Configuration constants module with full support for dataset handler strategy pattern
and transformation system integration.

This module defines all static constants derived from configuration,
including atomic energies, conversion factors, dataset-specific constants,
handler-specific constants, and transformation system constants.

Handler-Based Pattern Development Enhancements:
- Added handler factory support functions
- Enhanced handler-compatible constant access
- Added handler configuration validation
- Improved backward compatibility for handler migration
- Added handler-specific constant caching
- Added handler integration utilities

Transformation System Integration:
- Added transformation system constants and categories
- Added experimental setup configuration constants
- Added handler-transform compatibility matrix
- Added transform accessor functions with caching
- Added handler-transform integration validation
- Enhanced debugging and diagnostics for complete system

Registry Integration (MILIA_Dataset_Architecture_Refactoring_Plan_v2.1.0):
- Added registry imports for dynamic dataset type resolution
- Added cache invalidation callbacks (FIX #5)
- Added dynamic wrapper functions delegating to registry
- Refactored get_handler_constants() to use registry
- Added deprecation warnings for legacy constant access
- Updated cache management with registry integration status
"""

import logging
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

# DO NOT import from config_accessors here - causes circular import
# Local implementations below avoid the circular dependency
from milia_pipeline.exceptions import (
    ConfigurationError,
    HandlerConfigurationError,
    HandlerNotAvailableError,
)

# Initialize logger for this module
logger = logging.getLogger(__name__)


# ==========================================
# PHASE 3: Registry Integration Imports
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

    The datasets/__init__.py imports implementations which import handlers which
    may import config modules. By deferring the registry import until first use,
    we allow the config module to fully load first.

    Returns:
        True if registry is available, False otherwise

    ADDED Phase 3: Lazy initialization to resolve circular import issues.
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
            get,
            get_default_registry,
            is_registered,
            list_all,
        )

        _registry_list_all = list_all
        _registry_get = get
        _registry_is_registered = is_registered
        _registry_get_default = get_default_registry
        _REGISTRY_AVAILABLE = True
        logger.debug("Dataset registry initialized successfully")
        return True

    except ImportError as e:
        _REGISTRY_IMPORT_ERROR = str(e)
        logger.warning(f"Dataset registry not available - using legacy hardcoded constants: {e}")
        return False

    except Exception as e:
        # Catch any other exceptions (e.g., circular import issues)
        _REGISTRY_IMPORT_ERROR = str(e)
        logger.warning(f"Dataset registry import failed - using legacy hardcoded constants: {e}")
        return False


def _discover_dataset_types_from_filesystem() -> list[str]:
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


def registry_list_all():
    """
    Wrapper for registry.list_all() with lazy initialization and dynamic fallback.

    DYNAMIC APPROACH: Instead of hardcoded SUPPORTED_HANDLER_TYPES fallback,
    uses filesystem discovery when registry unavailable.
    """
    _init_registry()
    if _registry_list_all is not None:
        try:
            return _registry_list_all()
        except Exception as e:
            logger.debug(f"Registry list_all() failed: {e}")

    # DYNAMIC FALLBACK: Use filesystem discovery instead of hardcoded list
    return _discover_dataset_types_from_filesystem()


def registry_get(name: str):
    """
    Wrapper for registry.get() with lazy initialization.

    DYNAMIC APPROACH: Uses registry_list_all() for available_types in error messages,
    which does dynamic filesystem discovery when registry unavailable.
    """
    _init_registry()
    if _registry_get is not None:
        return _registry_get(name)
    raise HandlerNotAvailableError(
        f"Registry not available, cannot get dataset '{name}'",
        requested_dataset_type=name,
        available_types=registry_list_all(),  # DYNAMIC: Use discovery instead of hardcoded
    )


def registry_is_registered(name: str) -> bool:
    """
    Wrapper for registry.is_registered() with lazy initialization and dynamic fallback.

    DYNAMIC APPROACH: Instead of checking against hardcoded SUPPORTED_HANDLER_TYPES,
    uses registry_list_all() which does dynamic filesystem discovery.
    """
    _init_registry()
    if _registry_is_registered is not None:
        try:
            return _registry_is_registered(name)
        except Exception as e:
            logger.debug(f"Registry is_registered() failed: {e}")

    # DYNAMIC FALLBACK: Check against dynamically discovered types
    return name in registry_list_all()


def get_default_registry():
    """Wrapper for registry.get_default_registry() with lazy initialization."""
    _init_registry()
    if _registry_get_default is not None:
        return _registry_get_default()
    return None


# ==========================================
# Local Config Helper Functions (Avoid Circular Import)
# ==========================================


def _get_config_value(
    config: dict[str, Any], key: str, expected_type, parent_key: str | None = None
):
    """
    Local implementation of config value extraction to avoid circular import.

    This duplicates the logic from config_accessors._get_config_value to prevent
    circular dependency between config_constants and config_accessors.

    Args:
        config: Configuration dictionary
        key: Key to extract
        expected_type: Expected type or tuple of types
        parent_key: Optional parent key for better error messages

    Returns:
        The configuration value

    Raises:
        ConfigurationError: If key is missing or type is wrong
    """
    if key not in config:
        parent_info = f" in '{parent_key}'" if parent_key else ""
        raise ConfigurationError(f"Missing required config key: '{key}'{parent_info}")

    value = config[key]
    if not isinstance(value, expected_type):
        raise ConfigurationError(
            f"Config key '{key}' has wrong type. Expected {expected_type}, got {type(value)}"
        )
    return value


def _get_dataset_type_local() -> str:
    """Local implementation to get dataset type without importing config_accessors."""
    from milia_pipeline.config.config_loader import load_config  # Safe

    config = load_config()
    return _get_config_value(config, "dataset_type", str)


def _get_dataset_config_local() -> dict[str, Any]:
    """Local implementation to get dataset config without importing config_accessors."""
    from milia_pipeline.config.config_loader import load_config  # Safe

    config = load_config()
    dataset_type = _get_dataset_type_local()
    config_key = f"{dataset_type.lower()}_config"
    return _get_config_value(config, config_key, dict)


def _get_dataset_constants_local() -> tuple[str, str | None, str]:
    """Local implementation to get dataset constants without importing config_accessors."""
    from milia_pipeline.config.config_loader import load_config

    config = load_config()
    dataset_config = _get_dataset_config_local()

    raw_npz_filename = _get_config_value(dataset_config, "raw_npz_filename", str)
    raw_data_download_url = _get_config_value(
        dataset_config, "raw_data_download_url", (str, type(None))
    )

    # Get working_root_dir from global_paths section
    global_paths = _get_config_value(config, "global_paths", dict)
    working_root_dir = _get_config_value(global_paths, "working_root_dir", str)

    return raw_npz_filename, raw_data_download_url, working_root_dir


# For compatibility: expose these as module-level functions that other code might import
get_dataset_constants = _get_dataset_constants_local
get_dataset_config = _get_dataset_config_local


# ==========================================
# Lazy Config Loading (Avoid Circular Import)
# ==========================================

# Don't load config at module level - defer until first access via __getattr__
# This prevents circular import when config_loader calls back into config_constants
_TEMP_CONFIG: dict[str, Any] | None = None
_CONSTANTS_CACHE: dict[str, Any] = {}


# ==========================================
# Core Scientific Constants
# ==========================================

# NOTE: These constants are now loaded lazily via __getattr__ at the end of this file
# ATOMIC_ENERGIES_HARTREE, HEAVY_ATOM_SYMBOLS_TO_Z, and HAR2EV are accessed normally
# but loaded on-demand to avoid circular imports


# ==========================================
# Dataset Constants (Dynamic Based on Type)
# ==========================================

# For backward compatibility, provide constants that match the original interface
# These will be determined dynamically based on the dataset type


@property
def RAW_NPZ_FILENAME() -> str:
    """Gets the raw NPZ filename for the current dataset type."""
    return get_dataset_constants()[0]


@property
def RAW_DATA_DOWNLOAD_URL() -> str | None:
    """Gets the raw data download URL for the current dataset type."""
    return get_dataset_constants()[1]


@property
def DATASET_ROOT_DIR() -> str:
    """Gets the dataset root directory for the current dataset type."""
    return get_dataset_constants()[2]


### For immediate access without function calls (cached)
##_dataset_constants = get_dataset_constants()
##RAW_NPZ_FILENAME_CACHED = _dataset_constants[0]
##RAW_NPZ_DOWNLOAD_URL_CACHED = _dataset_constants[1]
##DATASET_ROOT_DIR_CACHED = _dataset_constants[2]
##
### Dynamically generates the processed dataset filename based on the raw NPZ filename.
### Example: 'DFT_all.npz' becomes 'DFT_all.pt'.
##if RAW_NPZ_FILENAME_CACHED:
##    PROCESSED_DATA_FILENAME: str = Path(RAW_NPZ_FILENAME_CACHED).stem + '.pt'
##else:
##    # Log a warning and fall back to the old filename
##    logger.warning("The raw NPZ filename is not specified in config.yaml. Falling back to default 'data.pt'.")
##    PROCESSED_DATA_FILENAME: str = 'data.pt'
##
##
### ==========================================
### Legacy Constants for Backward Compatibility
### ==========================================
##
##try:
##    # Try to access the old structure for backward compatibility
##    # DATASET_CONFIG: Dict[str, Any] = _get_config_value(_TEMP_CONFIG, 'dataset_config', dict)
##    RAW_NPZ_FILENAME_LEGACY: str = _get_config_value(DATASET_CONFIG, 'raw_npz_filename', str)
##    RAW_NPZ_DOWNLOAD_URL_LEGACY: Optional[str] = _get_config_value(DATASET_CONFIG, 'raw_npz_download_url', (str, type(None)))
##    DATASET_ROOT_DIR_LEGACY: str = _get_config_value(DATASET_CONFIG, 'dataset_root_dir', str)
##except ConfigurationError:
##    # Old structure not present, use new dynamic constants
##    DATASET_CONFIG = get_dataset_config()
##    RAW_NPZ_FILENAME_LEGACY = RAW_NPZ_FILENAME_CACHED
##    RAW_NPZ_DOWNLOAD_URL_LEGACY = RAW_NPZ_DOWNLOAD_URL_CACHED


# ==========================================
# HANDLER PATTERN SUPPORT
# ==========================================

# Handler type constants
# DEPRECATED: Use get_supported_handler_types() or registry_list_all() instead
# This constant is kept for backward compatibility only and may not include
# all dynamically registered dataset types (e.g., QM9, MD, etc.)
SUPPORTED_HANDLER_TYPES: list[str] = [
    "DFT",
    "DMC",
    "Wavefunction",
    "QM9",
    "ANI1x",
    "ANI1CCX",
    "RMD17",
    "ANI2x",
    "XXMD",
    "QDPi",
]  # LEGACY - use get_supported_handler_types()
DEFAULT_HANDLER_TYPE: str = "DFT"


# Handler configuration validation constants
REQUIRED_HANDLER_CONFIG_KEYS: dict[str, list[str]] = {
    "DFT": ["dataset_type", "processing_config"],
    "DMC": ["dataset_type", "processing_config", "uncertainty_config"],
    "Wavefunction": ["dataset_type", "processing_config"],
    "QM9": ["dataset_type", "processing_config"],
    "ANI1x": ["dataset_type", "processing_config"],
    "ANI1CCX": ["dataset_type", "processing_config"],
    "RMD17": ["dataset_type", "processing_config"],
    "ANI2x": ["dataset_type", "processing_config"],
    "XXMD": ["dataset_type", "processing_config"],
    "QDPi": ["dataset_type", "processing_config"],
}

# Handler feature support matrix
HANDLER_FEATURE_SUPPORT: dict[str, dict[str, bool]] = {
    "DFT": {
        "vibrational_analysis": True,
        "uncertainty_handling": False,
        "atomization_energy": True,
        "rotational_constants": True,
        "frequency_analysis": True,
    },
    "DMC": {
        "vibrational_analysis": False,
        "uncertainty_handling": True,
        "atomization_energy": False,
        "rotational_constants": False,
        "frequency_analysis": False,
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
    "QM9": {
        "vibrational_analysis": True,
        "uncertainty_handling": False,
        "atomization_energy": True,
        "rotational_constants": True,
        "frequency_analysis": True,
        "orbital_analysis": False,
        "homo_lumo_gap": True,
        "mo_energies": False,
    },
    "ANI1x": {
        "vibrational_analysis": False,
        "uncertainty_handling": False,
        "atomization_energy": True,
        "rotational_constants": False,
        "frequency_analysis": False,
        "orbital_analysis": False,
        "homo_lumo_gap": False,
        "mo_energies": False,
    },
    "ANI1CCX": {
        "vibrational_analysis": False,
        "uncertainty_handling": False,
        "atomization_energy": True,
        "rotational_constants": False,
        "frequency_analysis": False,
        "orbital_analysis": False,
        "homo_lumo_gap": False,
        "mo_energies": False,
    },
    "RMD17": {
        "vibrational_analysis": False,
        "uncertainty_handling": False,
        "atomization_energy": True,
        "rotational_constants": False,
        "frequency_analysis": False,
        "orbital_analysis": False,
        "homo_lumo_gap": False,
        "mo_energies": False,
        "forces": True,  # RMD17 has atomic forces
    },
    "ANI2x": {
        "vibrational_analysis": False,
        "uncertainty_handling": False,
        "atomization_energy": True,
        "rotational_constants": False,
        "frequency_analysis": False,
        "orbital_analysis": False,
        "homo_lumo_gap": False,
        "mo_energies": False,
        "forces": True,  # ANI-2x has atomic forces
    },
    "XXMD": {
        "vibrational_analysis": False,
        "uncertainty_handling": False,
        "atomization_energy": True,
        "rotational_constants": False,
        "frequency_analysis": False,
        "orbital_analysis": False,
        "homo_lumo_gap": False,
        "mo_energies": False,
        "forces": True,  # xxMD has atomic forces
    },
    "QDPi": {
        "vibrational_analysis": False,
        "uncertainty_handling": False,
        "atomization_energy": True,
        "rotational_constants": False,
        "frequency_analysis": False,
        "orbital_analysis": False,
        "homo_lumo_gap": False,
        "mo_energies": False,
        "forces": True,  # QDπ has atomic forces
    },
}

# Handler property requirements
HANDLER_REQUIRED_PROPERTIES: dict[str, list[str]] = {
    "DFT": ["Etot", "atoms", "coordinates"],
    "DMC": ["Etot", "std", "atoms", "coordinates"],
    "Wavefunction": ["atoms", "coordinates", "compounds"],
    "QM9": ["U0", "atoms", "coordinates"],
    "ANI1x": ["energy", "atoms", "coordinates"],
    "ANI1CCX": ["ccsd_energy", "atoms", "coordinates"],
    "RMD17": ["energies", "atoms", "coordinates"],
    "ANI2x": ["energy", "atoms", "coordinates"],
    "XXMD": ["energy", "atoms", "coordinates"],
    "QDPi": ["energy", "atoms", "coordinates"],
}

# Handler optional properties
HANDLER_OPTIONAL_PROPERTIES: dict[str, list[str]] = {
    "DFT": ["freqs", "vibmodes", "rots", "dipoles"],
    "DMC": ["qmc_stats", "correlation_data"],
    "Wavefunction": ["mo_energies", "mo_occupations", "homo_lumo_gap_eV", "total_energy"],
    "QM9": [
        "A",
        "B",
        "C",
        "mu",
        "alpha",
        "homo",
        "lumo",
        "gap",
        "r2",
        "zpve",
        "U",
        "H",
        "G",
        "Cv",
        "freqs",
        "Qmulliken",
    ],
    "ANI1x": ["forces", "hirshfeld_charges", "cm5_charges", "dipole", "molecule_id"],
    "ANI1CCX": [
        "dft_energy",
        "forces",
        "hirshfeld_charges",
        "cm5_charges",
        "dipole",
        "molecule_id",
    ],
    "RMD17": ["forces", "molecule_name", "old_energies", "old_forces", "old_indices"],
    "ANI2x": ["forces", "molecule_id"],
    "XXMD": ["forces", "molecule_name", "split"],
    "QDPi": ["forces", "formula"],
}

# Handler molecular identifier keys
# Defines the NPZ keys to use for molecular identification, in priority order
# Format: List of tuples (npz_key, identifier_type)
# The converter will try each key in order until it finds a valid identifier
HANDLER_IDENTIFIER_KEYS: dict[str, list[tuple[str, str]]] = {
    "DFT": [
        ("inchi", "inchi"),  # Primary: InChI identifier
        ("graphs", "smiles"),  # Fallback: SMILES identifier
    ],
    "DMC": [
        ("inchi", "inchi"),  # Primary: InChI identifier
        ("graphs", "smiles"),  # Fallback: SMILES identifier
    ],
    "Wavefunction": [
        ("compounds", "compound_id")  # Wavefunction uses compound IDs
    ],
    "QM9": [
        ("inchi", "inchi"),  # Primary: InChI identifier (explicit H atoms)
        ("smiles", "smiles"),  # Fallback: SMILES identifier
    ],
    "ANI1x": [],  # ANI-1x has NO parseable identifiers - uses coordinate_based strategy
    "ANI1CCX": [],  # ANI-1ccx has NO parseable identifiers - uses coordinate_based strategy
    "RMD17": [],  # rMD17 has NO parseable identifiers - uses coordinate_based strategy
    "ANI2x": [],  # ANI-2x has NO parseable identifiers - uses coordinate_based strategy
    "XXMD": [],  # xxMD has NO parseable identifiers - uses coordinate_based strategy
    "QDPi": [],  # QDπ has NO parseable identifiers - uses coordinate_based strategy
}

# Handler coordinate units (for unit conversion)
HANDLER_COORDINATE_UNITS: dict[str, str] = {
    "DFT": "angstrom",
    "DMC": "angstrom",
    "Wavefunction": "bohr",
    "QM9": "angstrom",
    "ANI1x": "angstrom",
    "ANI1CCX": "angstrom",
    "RMD17": "angstrom",
    "ANI2x": "angstrom",
    "XXMD": "angstrom",
    "QDPi": "angstrom",
}


# ==========================================
# PHASE 3: Legacy Constant Deprecation Helpers
# ==========================================

_DEPRECATION_WARNED: dict[str, bool] = {}


def _warn_legacy_constant_access(constant_name: str, replacement: str) -> None:
    """
    Issue a deprecation warning for legacy constant access.

    Only warns once per constant to avoid spam.

    Args:
        constant_name: Name of the legacy constant
        replacement: Name of the replacement function
    """
    if constant_name not in _DEPRECATION_WARNED:
        warnings.warn(
            f"Direct access to {constant_name} is deprecated. "
            f"Use {replacement}() instead for dynamic registry support.",
            DeprecationWarning,
            stacklevel=3,
        )
        _DEPRECATION_WARNED[constant_name] = True


def get_supported_handler_types_legacy() -> list[str]:
    """
    Legacy accessor for SUPPORTED_HANDLER_TYPES with deprecation warning.

    DEPRECATED: Use get_supported_handler_types() instead.
    """
    _warn_legacy_constant_access("SUPPORTED_HANDLER_TYPES", "get_supported_handler_types")
    return SUPPORTED_HANDLER_TYPES


def get_handler_feature_support_legacy(handler_type: str) -> dict[str, bool]:
    """
    Legacy accessor for HANDLER_FEATURE_SUPPORT with deprecation warning.

    DEPRECATED: Use get_handler_feature_support() instead.
    """
    _warn_legacy_constant_access("HANDLER_FEATURE_SUPPORT", "get_handler_feature_support")
    return HANDLER_FEATURE_SUPPORT.get(handler_type, {})


def get_handler_required_properties_legacy(handler_type: str) -> list[str]:
    """
    Legacy accessor for HANDLER_REQUIRED_PROPERTIES with deprecation warning.

    DEPRECATED: Use get_handler_required_properties() instead.
    """
    _warn_legacy_constant_access("HANDLER_REQUIRED_PROPERTIES", "get_handler_required_properties")
    return HANDLER_REQUIRED_PROPERTIES.get(handler_type, [])


# ==========================================
# PHASE 3: Registry Cache Invalidation Setup (FIX #5)
# ==========================================


def _invalidate_all_handler_caches() -> None:
    """
    Invalidate all handler-related LRU caches.

    This function is registered as a callback with the DatasetRegistry
    and is called whenever datasets are registered or unregistered.
    This ensures cached values are refreshed when new dataset types
    become available.

    ADDED Phase 3 (FIX #5 from v2.1.0)
    """
    try:
        get_handler_constants.cache_clear()
        get_handler_identifier_keys.cache_clear()
        get_cached_handler_config.cache_clear()
        logger.debug("Handler caches invalidated due to registry change")
    except Exception as e:
        logger.warning(f"Cache invalidation failed: {e}")


def _setup_registry_cache_invalidation() -> bool:
    """
    Register cache invalidation callback with the dataset registry.

    Returns:
        True if callback was registered successfully, False otherwise

    ADDED Phase 3 (FIX #5 from v2.1.0)
    """
    # Ensure registry is initialized
    if not _init_registry():
        logger.debug("Registry not available - cache invalidation not configured")
        return False

    try:
        registry = get_default_registry()
        if registry is not None:
            registry.add_on_change_callback(_invalidate_all_handler_caches)
            logger.debug("Registry cache invalidation callback registered")
            return True
        return False
    except Exception as e:
        logger.warning(f"Failed to register cache invalidation callback: {e}")
        return False


# Register cache invalidation on module load (deferred to avoid circular import)
_CACHE_INVALIDATION_REGISTERED = False


def _ensure_cache_invalidation_registered() -> None:
    """Ensure cache invalidation is registered (call on first use)."""
    global _CACHE_INVALIDATION_REGISTERED
    if not _CACHE_INVALIDATION_REGISTERED:
        _CACHE_INVALIDATION_REGISTERED = _setup_registry_cache_invalidation()


# ==========================================
# PHASE 3: Dynamic Registry Wrapper Functions
# ==========================================


def get_supported_handler_types() -> list[str]:
    """
    Get list of supported dataset/handler types dynamically from registry.

    DYNAMIC APPROACH: Uses registry_list_all() which:
    1. First tries the registry (primary source of truth)
    2. If registry fails, uses filesystem discovery via _discover_dataset_types_from_filesystem()
    3. Never relies on hardcoded SUPPORTED_HANDLER_TYPES constant

    Returns:
        List of registered dataset type names

    ADDED Phase 3: Replaces direct access to SUPPORTED_HANDLER_TYPES constant.
    UPDATED: Now uses dynamic filesystem discovery as fallback instead of hardcoded list.
    """
    _ensure_cache_invalidation_registered()

    if _init_registry() and _REGISTRY_AVAILABLE:
        try:
            return registry_list_all()
        except Exception as e:
            logger.warning(f"Registry lookup failed, using dynamic discovery: {e}")

    # DYNAMIC FALLBACK: Use filesystem discovery instead of hardcoded SUPPORTED_HANDLER_TYPES
    return _discover_dataset_types_from_filesystem()


def get_default_handler_type() -> str:
    """
    Get the default handler type.

    Returns:
        Default handler type name (currently 'DFT')

    ADDED Phase 3: Returns DEFAULT_HANDLER_TYPE constant.
    In future, this could be made configurable via registry metadata.
    """
    return DEFAULT_HANDLER_TYPE


def get_handler_feature_support(handler_type: str) -> dict[str, bool]:
    """
    Get feature support dictionary for a handler type from registry.

    Args:
        handler_type: Type of handler ('DFT', 'DMC', 'Wavefunction', etc.)

    Returns:
        Dictionary mapping feature names to support status

    Raises:
        HandlerNotAvailableError: If handler type is not registered

    ADDED Phase 3: Delegates to dataset class get_feature_support() method.
    """
    _ensure_cache_invalidation_registered()

    if _init_registry() and _REGISTRY_AVAILABLE:
        try:
            dataset_class = registry_get(handler_type)
            return dataset_class.get_feature_support()
        except Exception as e:
            logger.debug(f"Registry lookup failed for {handler_type}: {e}")

    # DYNAMIC: Use registry_is_registered() which does filesystem discovery as fallback
    if not registry_is_registered(handler_type):
        raise HandlerNotAvailableError(
            f"Handler type '{handler_type}' is not supported",
            requested_dataset_type=handler_type,
            available_types=get_supported_handler_types(),
        )

    return HANDLER_FEATURE_SUPPORT.get(handler_type, {})


def get_handler_required_properties(handler_type: str) -> list[str]:
    """
    Get required properties for a handler type from registry.

    Args:
        handler_type: Type of handler ('DFT', 'DMC', 'Wavefunction', etc.)

    Returns:
        List of required property names

    Raises:
        HandlerNotAvailableError: If handler type is not registered

    ADDED Phase 3: Delegates to dataset class get_required_properties() method.
    """
    _ensure_cache_invalidation_registered()

    if _init_registry() and _REGISTRY_AVAILABLE:
        try:
            dataset_class = registry_get(handler_type)
            return dataset_class.get_required_properties()
        except Exception as e:
            logger.debug(f"Registry lookup failed for {handler_type}: {e}")

    # DYNAMIC: Use registry_is_registered() which does filesystem discovery as fallback
    if not registry_is_registered(handler_type):
        raise HandlerNotAvailableError(
            f"Handler type '{handler_type}' is not supported",
            requested_dataset_type=handler_type,
            available_types=get_supported_handler_types(),
        )

    return HANDLER_REQUIRED_PROPERTIES.get(handler_type, [])


def get_handler_optional_properties(handler_type: str) -> list[str]:
    """
    Get optional properties for a handler type from registry.

    Args:
        handler_type: Type of handler ('DFT', 'DMC', 'Wavefunction', etc.)

    Returns:
        List of optional property names

    Raises:
        HandlerNotAvailableError: If handler type is not registered

    ADDED Phase 3: Delegates to dataset class get_optional_properties() method.
    """
    _ensure_cache_invalidation_registered()

    if _init_registry() and _REGISTRY_AVAILABLE:
        try:
            dataset_class = registry_get(handler_type)
            return dataset_class.get_optional_properties()
        except Exception as e:
            logger.debug(f"Registry lookup failed for {handler_type}: {e}")

    # DYNAMIC: Use registry_is_registered() which does filesystem discovery as fallback
    if not registry_is_registered(handler_type):
        raise HandlerNotAvailableError(
            f"Handler type '{handler_type}' is not supported",
            requested_dataset_type=handler_type,
            available_types=get_supported_handler_types(),
        )

    return HANDLER_OPTIONAL_PROPERTIES.get(handler_type, [])


def get_handler_identifier_keys_dynamic(handler_type: str) -> list[tuple[str, str]]:
    """
    Get identifier keys for a handler type from registry.

    Args:
        handler_type: Type of handler ('DFT', 'DMC', 'Wavefunction', etc.)

    Returns:
        List of (npz_key, identifier_type) tuples

    Raises:
        HandlerNotAvailableError: If handler type is not registered

    ADDED Phase 3: Delegates to dataset class get_identifier_keys() method.
    """
    _ensure_cache_invalidation_registered()

    if _init_registry() and _REGISTRY_AVAILABLE:
        try:
            dataset_class = registry_get(handler_type)
            return dataset_class.get_identifier_keys()
        except Exception as e:
            logger.debug(f"Registry lookup failed for {handler_type}: {e}")

    # DYNAMIC: Use registry_is_registered() which does filesystem discovery as fallback
    if not registry_is_registered(handler_type):
        raise HandlerNotAvailableError(
            f"Handler type '{handler_type}' is not supported",
            requested_dataset_type=handler_type,
            available_types=get_supported_handler_types(),
        )

    return HANDLER_IDENTIFIER_KEYS.get(handler_type, [])


def get_handler_coordinate_units_dynamic(handler_type: str) -> str:
    """
    Get coordinate units for a handler type from registry.

    Args:
        handler_type: Type of handler ('DFT', 'DMC', 'Wavefunction', etc.)

    Returns:
        Coordinate unit string ('angstrom' or 'bohr')

    Raises:
        HandlerNotAvailableError: If handler type is not registered

    ADDED Phase 3: Delegates to dataset class get_coordinate_units() method.
    """
    _ensure_cache_invalidation_registered()

    if _init_registry() and _REGISTRY_AVAILABLE:
        try:
            dataset_class = registry_get(handler_type)
            return dataset_class.get_coordinate_units()
        except Exception as e:
            logger.debug(f"Registry lookup failed for {handler_type}: {e}")

    # DYNAMIC: Use registry_is_registered() which does filesystem discovery as fallback
    if not registry_is_registered(handler_type):
        raise HandlerNotAvailableError(
            f"Handler type '{handler_type}' is not supported",
            requested_dataset_type=handler_type,
            available_types=get_supported_handler_types(),
        )

    return HANDLER_COORDINATE_UNITS.get(handler_type, "angstrom")


def get_handler_molecule_creation_strategy(handler_type: str) -> str:
    """
    Get molecule creation strategy for a handler type from registry.

    Args:
        handler_type: Type of handler ('DFT', 'DMC', 'Wavefunction', etc.)

    Returns:
        Strategy string ('identifier_coordinate_based' or 'coordinate_based')

    Raises:
        HandlerNotAvailableError: If handler type is not registered

    ADDED Phase 3: Delegates to dataset class get_molecule_creation_strategy() method.
    """
    _ensure_cache_invalidation_registered()

    if _init_registry() and _REGISTRY_AVAILABLE:
        try:
            dataset_class = registry_get(handler_type)
            return dataset_class.get_molecule_creation_strategy()
        except Exception as e:
            logger.debug(f"Registry lookup failed for {handler_type}: {e}")

    # DYNAMIC: Use registry_is_registered() which does filesystem discovery as fallback
    if not registry_is_registered(handler_type):
        raise HandlerNotAvailableError(
            f"Handler type '{handler_type}' is not supported",
            requested_dataset_type=handler_type,
            available_types=get_supported_handler_types(),
        )

    # Fallback based on known strategies
    if handler_type == "Wavefunction":
        return "coordinate_based"
    return "identifier_coordinate_based"


def is_handler_type_supported(handler_type: str) -> bool:
    """
    Check if a handler type is supported (registered in registry).

    Args:
        handler_type: Type of handler to check

    Returns:
        True if handler type is registered, False otherwise

    ADDED Phase 3: Uses registry is_registered() for dynamic check.
    """
    _ensure_cache_invalidation_registered()

    if _init_registry() and _REGISTRY_AVAILABLE:
        try:
            return registry_is_registered(handler_type)
        except Exception as e:
            logger.debug(f"Registry check failed for {handler_type}: {e}")

    # DYNAMIC: Use registry_is_registered() which does filesystem discovery as fallback
    return registry_is_registered(handler_type)


# ==========================================
# PHASE 3: Refactored Handler Constants Function
# ==========================================


@lru_cache(maxsize=32)
def get_handler_constants(handler_type: str) -> dict[str, Any]:
    """
    Get handler-specific constants with caching.

    REFACTORED Phase 3: Now uses dynamic registry lookups instead of
    hardcoded dictionaries. Falls back to legacy constants if registry
    unavailable.

    Args:
        handler_type: Type of handler ('DFT', 'DMC', 'Wavefunction', etc.)

    Returns:
        Dictionary of handler-specific constants

    Raises:
        HandlerNotAvailableError: If handler type is not supported
        HandlerConfigurationError: If handler configuration is invalid
    """
    _ensure_cache_invalidation_registered()

    # Validate handler type using dynamic check
    if not is_handler_type_supported(handler_type):
        raise HandlerNotAvailableError(
            f"Handler type '{handler_type}' is not supported",
            requested_dataset_type=handler_type,
            available_types=get_supported_handler_types(),
        )

    try:
        # Get dataset-specific configuration
        dataset_config = get_dataset_config()

        # Build handler constants using dynamic wrapper functions
        constants = {
            "handler_type": handler_type,
            "required_properties": get_handler_required_properties(handler_type),
            "optional_properties": get_handler_optional_properties(handler_type),
            "feature_support": get_handler_feature_support(handler_type),
            "identifier_keys": get_handler_identifier_keys_dynamic(handler_type),
            "coordinate_units": get_handler_coordinate_units_dynamic(handler_type),
            "molecule_creation_strategy": get_handler_molecule_creation_strategy(handler_type),
            "dataset_config": dataset_config,
        }

        # Add handler-specific constants using registry-aware dispatch
        handler_specific = _get_handler_specific_constants(handler_type, dataset_config)
        constants.update(handler_specific)

        return constants

    except Exception as e:
        if isinstance(e, (HandlerNotAvailableError, HandlerConfigurationError)):
            raise
        raise HandlerConfigurationError(
            f"Failed to get constants for handler '{handler_type}'",
            handler_type=handler_type,
            details=str(e),
        )


def _get_handler_specific_constants(
    handler_type: str, dataset_config: dict[str, Any]
) -> dict[str, Any]:
    """
    Get handler-specific constants based on handler type.

    ADDED Phase 3: Centralizes handler-specific constant retrieval.
    Uses registry to determine handler class and delegates accordingly.

    Args:
        handler_type: Type of handler
        dataset_config: Dataset configuration dictionary

    Returns:
        Dictionary of handler-specific constants
    """
    # Try to get handler class from registry for custom constant retrieval
    if _init_registry() and _REGISTRY_AVAILABLE:
        try:
            dataset_class = registry_get(handler_type)
            # Check if dataset class has custom get_handler_specific_constants method
            if hasattr(dataset_class, "get_handler_specific_constants"):
                return dataset_class.get_handler_specific_constants(dataset_config)
        except Exception as e:
            logger.debug(f"Registry-based constant retrieval failed for {handler_type}: {e}")

    # Fallback to legacy handler-specific functions
    if handler_type == "DFT":
        return _get_dft_handler_constants(dataset_config)
    elif handler_type == "DMC":
        return _get_dmc_handler_constants(dataset_config)
    elif handler_type == "Wavefunction":
        return _get_wavefunction_handler_constants(dataset_config)

    # For new handler types registered via registry without legacy support
    return {}


def _get_dft_handler_constants(dataset_config: dict[str, Any]) -> dict[str, Any]:
    """
    Get DFT-specific handler constants.

    UPDATED Phase 3: Added feature_support retrieval from registry.
    """
    # Import here to avoid circular import and use lazy loading
    from milia_pipeline.config import config_constants

    constants = {
        "supports_vibrations": True,
        "supports_atomization_energy": True,
        "energy_conversion_factor": config_constants.HAR2EV,
        "atomic_energies": config_constants.ATOMIC_ENERGIES_HARTREE,
        "heavy_atom_mapping": config_constants.HEAVY_ATOM_SYMBOLS_TO_Z,
        "default_frequency_threshold": 50.0,
        "imaginary_frequency_tolerance": -50.0,
    }

    # Add config-specific keys if needed
    if "required_config_keys" not in constants:
        constants["required_config_keys"] = REQUIRED_HANDLER_CONFIG_KEYS.get("DFT", [])

    return constants


def _get_dmc_handler_constants(dataset_config: dict[str, Any]) -> dict[str, Any]:
    """
    Get DMC-specific handler constants.

    UPDATED Phase 3: Added feature_support retrieval from registry.
    """
    # Import here to avoid circular import and use lazy loading
    from milia_pipeline.config import config_constants

    uncertainty_config = dataset_config.get("uncertainty_config", {})

    constants = {
        "supports_uncertainty": True,
        "uncertainty_enabled": uncertainty_config.get("enabled", False),
        "max_uncertainty_threshold": uncertainty_config.get("max_uncertainty_threshold"),
        "uncertainty_validation_mode": uncertainty_config.get("validation_mode", "strict"),
        "energy_conversion_factor": config_constants.HAR2EV,
        "default_uncertainty_fields": ["std", "tau_corr", "statistical_error"],
    }

    # Add config-specific keys if needed
    if "required_config_keys" not in constants:
        constants["required_config_keys"] = REQUIRED_HANDLER_CONFIG_KEYS.get("DMC", [])

    return constants


def _get_wavefunction_handler_constants(dataset_config: dict[str, Any]) -> dict[str, Any]:
    """
    Get Wavefunction-specific handler constants.

    UPDATED Phase 3: Added Bohr-to-Angstrom conversion factor.
    """
    # Import here to avoid circular import and use lazy loading
    from milia_pipeline.config import config_constants

    constants = {
        "supports_orbital_analysis": True,
        "supports_homo_lumo_gap": True,
        "supports_mo_energies": True,
        "energy_unit": "eV",
        "default_mo_energy_range": (-50.0, 50.0),
        "homo_lumo_gap_min": 0.0,
        "mo_occupation_threshold": 0.1,
        "heavy_atom_mapping": config_constants.HEAVY_ATOM_SYMBOLS_TO_Z,
        "coordinate_unit_conversion": config_constants.BOHR_TO_ANGSTROM,
    }

    # Add config-specific keys if needed
    if "required_config_keys" not in constants:
        constants["required_config_keys"] = REQUIRED_HANDLER_CONFIG_KEYS.get("Wavefunction", [])

    return constants


# ==========================================
# PHASE 3: Refactored Handler Identifier Keys Function
# ==========================================


@lru_cache(maxsize=32)
def get_handler_identifier_keys(handler_type: str) -> list[tuple[str, str]]:
    """
    Get identifier extraction keys for a specific handler type.

    REFACTORED Phase 3: Uses dynamic registry lookup with caching.

    Returns list of (npz_key, identifier_type) tuples in priority order.

    Args:
        handler_type: Type of handler ('DFT', 'DMC', 'Wavefunction', etc.)

    Returns:
        List of (npz_key, identifier_type) tuples in priority order

    Raises:
        HandlerNotAvailableError: If handler type is not supported
    """
    _ensure_cache_invalidation_registered()

    if not is_handler_type_supported(handler_type):
        raise HandlerNotAvailableError(
            f"Handler type '{handler_type}' is not supported",
            requested_dataset_type=handler_type,
            available_types=get_supported_handler_types(),
        )

    return get_handler_identifier_keys_dynamic(handler_type)


# ==========================================
# PHASE 3: Refactored Validation Functions
# ==========================================


def validate_handler_configuration(handler_type: str, config: dict[str, Any]) -> list[str]:
    """
    Validate handler configuration for completeness and correctness.

    REFACTORED Phase 3: Uses dynamic registry check for handler type validation.

    Args:
        handler_type: Type of handler to validate
        config: Configuration dictionary to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    validation_errors = []

    # Check if handler type is supported using dynamic check
    if not is_handler_type_supported(handler_type):
        validation_errors.append(f"Unsupported handler type: {handler_type}")
        return validation_errors

    # Check required configuration keys
    required_keys = _get_required_config_keys_for_handler(handler_type)
    for key in required_keys:
        if key not in config:
            validation_errors.append(f"Missing required configuration key: {key}")
        elif config[key] is None:
            validation_errors.append(f"Configuration key '{key}' cannot be None")

    # Handler-specific validation using registry-aware dispatch
    specific_errors = _validate_handler_specific_config(handler_type, config)
    validation_errors.extend(specific_errors)

    return validation_errors


def _get_required_config_keys_for_handler(handler_type: str) -> list[str]:
    """
    Get required configuration keys for a handler type.

    ADDED Phase 3: Retrieves from registry if available, else uses legacy.

    Args:
        handler_type: Type of handler

    Returns:
        List of required configuration key names
    """
    if _init_registry() and _REGISTRY_AVAILABLE:
        try:
            dataset_class = registry_get(handler_type)
            if hasattr(dataset_class, "get_required_config_keys"):
                return dataset_class.get_required_config_keys()
        except Exception as e:
            logger.debug(f"Registry lookup for config keys failed: {e}")

    return REQUIRED_HANDLER_CONFIG_KEYS.get(handler_type, ["dataset_type", "processing_config"])


def _validate_handler_specific_config(handler_type: str, config: dict[str, Any]) -> list[str]:
    """
    Run handler-specific validation.

    ADDED Phase 3: Centralizes handler-specific validation dispatch.
    Uses registry to check for custom validation methods.

    Args:
        handler_type: Type of handler
        config: Configuration dictionary

    Returns:
        List of validation error messages
    """
    # Try registry-based validation first
    if _init_registry() and _REGISTRY_AVAILABLE:
        try:
            dataset_class = registry_get(handler_type)
            if hasattr(dataset_class, "validate_config"):
                return dataset_class.validate_config(config)
        except Exception as e:
            logger.debug(f"Registry-based validation failed: {e}")

    # Fallback to legacy validation functions
    if handler_type == "DMC":
        return _validate_dmc_handler_config(config)
    elif handler_type == "DFT":
        return _validate_dft_handler_config(config)
    elif handler_type == "Wavefunction":
        return _validate_wavefunction_handler_config(config)

    return []


def _validate_dmc_handler_config(config: dict[str, Any]) -> list[str]:
    """Validate DMC-specific handler configuration."""
    errors = []

    uncertainty_config = config.get("uncertainty_config", {})
    if isinstance(uncertainty_config, dict):
        # Validate uncertainty threshold
        max_threshold = uncertainty_config.get("max_uncertainty_threshold")
        if max_threshold is not None:
            try:
                threshold_val = float(max_threshold)
                if threshold_val <= 0:
                    errors.append("max_uncertainty_threshold must be positive")
            except (ValueError, TypeError):
                errors.append("max_uncertainty_threshold must be a numeric value")

        # Validate validation mode
        validation_mode = uncertainty_config.get("validation_mode", "strict")
        valid_modes = ["strict", "permissive", "disabled"]
        if validation_mode not in valid_modes:
            errors.append(
                f"Invalid uncertainty validation_mode: {validation_mode}. Must be one of {valid_modes}"
            )

    return errors


def _validate_dft_handler_config(config: dict[str, Any]) -> list[str]:
    """Validate DFT-specific handler configuration."""
    errors = []

    processing_config = config.get("processing_config", {})
    if isinstance(processing_config, dict):
        # Validate frequency thresholds
        freq_threshold = processing_config.get("frequency_threshold")
        if freq_threshold is not None:
            try:
                freq_val = float(freq_threshold)
                if freq_val < 0:
                    errors.append("frequency_threshold cannot be negative")
            except (ValueError, TypeError):
                errors.append("frequency_threshold must be a numeric value")

    return errors


def _validate_wavefunction_handler_config(config: dict[str, Any]) -> list[str]:
    """
    Validate Wavefunction-specific handler configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of validation error messages
    """
    errors = []

    processing_config = config.get("processing_config", {})
    if isinstance(processing_config, dict):
        # Validate feature_tier if specified
        feature_tier = processing_config.get("feature_tier")
        if feature_tier is not None:
            valid_tiers = ["basic", "standard", "complete"]
            if feature_tier not in valid_tiers:
                errors.append(f"Invalid feature_tier: {feature_tier}. Must be one of {valid_tiers}")

        # Validate scalar targets include appropriate wavefunction properties
        scalar_targets = processing_config.get("scalar_graph_targets", [])
        if isinstance(scalar_targets, list):
            # Recommend including homo_lumo_gap_eV if available
            if "homo_lumo_gap_eV" not in scalar_targets:
                # This is a warning, not an error - don't add to errors list
                pass

    return errors


# ==========================================
# PHASE 3: Refactored Compatibility and Integration Functions
# ==========================================


def get_handler_compatibility_info(handler_type: str) -> dict[str, Any]:
    """
    Get compatibility information for a handler type.

    REFACTORED Phase 3: Uses dynamic registry lookups.

    Args:
        handler_type: Type of handler

    Returns:
        Dictionary with compatibility information

    Raises:
        HandlerNotAvailableError: If handler type is not supported
    """
    if not is_handler_type_supported(handler_type):
        raise HandlerNotAvailableError(
            f"Handler type '{handler_type}' is not supported",
            requested_dataset_type=handler_type,
            available_types=get_supported_handler_types(),
        )

    feature_support = get_handler_feature_support(handler_type)

    return {
        "handler_type": handler_type,
        "supported_features": [k for k, v in feature_support.items() if v],
        "unsupported_features": [k for k, v in feature_support.items() if not v],
        "required_properties": get_handler_required_properties(handler_type),
        "optional_properties": get_handler_optional_properties(handler_type),
        "molecule_creation_strategy": get_handler_molecule_creation_strategy(handler_type),
        "coordinate_units": get_handler_coordinate_units_dynamic(handler_type),
        "backward_compatible": True,
        "migration_phase": "Phase 3 - Registry Integration",
        "configuration_requirements": _get_required_config_keys_for_handler(handler_type),
    }


def check_handler_feature_support(handler_type: str, feature: str) -> bool:
    """
    Check if a handler supports a specific feature.

    REFACTORED Phase 3: Uses dynamic registry lookup.

    Args:
        handler_type: Type of handler
        feature: Feature to check

    Returns:
        True if feature is supported, False otherwise

    Raises:
        HandlerNotAvailableError: If handler type is not supported
    """
    if not is_handler_type_supported(handler_type):
        raise HandlerNotAvailableError(
            f"Handler type '{handler_type}' is not supported",
            requested_dataset_type=handler_type,
            available_types=get_supported_handler_types(),
        )

    feature_support = get_handler_feature_support(handler_type)
    return feature_support.get(feature, False)


def get_handler_property_requirements(handler_type: str) -> tuple[list[str], list[str]]:
    """
    Get required and optional properties for a handler type.

    REFACTORED Phase 3: Uses dynamic registry lookup.

    Args:
        handler_type: Type of handler

    Returns:
        Tuple of (required_properties, optional_properties)

    Raises:
        HandlerNotAvailableError: If handler type is not supported
    """
    if not is_handler_type_supported(handler_type):
        raise HandlerNotAvailableError(
            f"Handler type '{handler_type}' is not supported",
            requested_dataset_type=handler_type,
            available_types=get_supported_handler_types(),
        )

    required = get_handler_required_properties(handler_type)
    optional = get_handler_optional_properties(handler_type)

    return required, optional


# ==========================================
# TRANSFORMATION SYSTEM CONSTANTS
# ==========================================

# Transform system availability flag (set during module initialization)
TRANSFORMATION_SYSTEM_AVAILABLE: bool | None = None

# Core transform categories (matches graph_transforms.py)
TRANSFORM_CATEGORIES: list[str] = [
    "structural",  # AddSelfLoops, ToUndirected, etc.
    "geometric",  # RandomRotate, RandomScale, etc.
    "normalization",  # Normalize, GCNNorm, etc.
    "augmentation",  # DropEdge, DropNode, etc.
    "spatial",  # Distance, Cartesian, etc.
    "custom",  # User-defined transforms
]

# Core transforms registry (from graph_transforms.py discovery)
CORE_TRANSFORMS: dict[str, str] = {
    # Structural modifications
    "AddSelfLoops": "structural",
    "ToUndirected": "structural",
    "RemoveIsolatedNodes": "structural",
    "VirtualNode": "structural",
    # Geometric transformations
    "RandomRotate": "geometric",
    "RandomScale": "geometric",
    "RandomTranslate": "geometric",
    "RandomFlip": "geometric",
    # Normalization
    "Normalize": "normalization",
    "GCNNorm": "normalization",
    "NormalizeFeatures": "normalization",
    # Augmentation
    "DropEdge": "augmentation",
    "DropNode": "augmentation",
    "RandomNodeSample": "augmentation",
    "MaskFeatures": "augmentation",
    # Distance/spatial
    "Distance": "spatial",
    "Cartesian": "spatial",
    "LocalCartesian": "spatial",
}

# Transform validation modes
TRANSFORM_VALIDATION_MODES: list[str] = ["strict", "permissive", "disabled"]
DEFAULT_TRANSFORM_VALIDATION_MODE: str = "permissive"

# Experimental setup configuration
DEFAULT_EXPERIMENTAL_SETUP_NAME: str = "default"
MAX_EXPERIMENTAL_SETUPS: int = 50  # Reasonable limit
MAX_TRANSFORMS_PER_SETUP: int = 20  # Performance consideration

# Transform-handler compatibility matrix
HANDLER_TRANSFORM_COMPATIBILITY: dict[str, dict[str, str]] = {
    "DFT": {
        "RandomRotate": "warning",  # May affect vibrational modes
        "NormalizeFeatures": "compatible",
        "GCNNorm": "compatible",
        "AddSelfLoops": "compatible",
        "ToUndirected": "compatible",
        "Distance": "compatible",
        "Cartesian": "compatible",
        "LocalCartesian": "compatible",
        "RandomScale": "compatible",
        "RandomTranslate": "warning",  # May affect spatial features
        "DropEdge": "compatible",
        "DropNode": "warning",
    },
    "DMC": {
        "NormalizeFeatures": "incompatible",  # Uncertainty data shouldn't be normalized
        "Normalize": "incompatible",
        "AddSelfLoops": "compatible",
        "ToUndirected": "compatible",
        "DropEdge": "warning",  # May affect uncertainty propagation
        "DropNode": "warning",
        "RandomRotate": "compatible",
        "RandomScale": "compatible",
        "RandomTranslate": "compatible",
        "Distance": "compatible",
        "GCNNorm": "warning",  # Check normalization interaction with uncertainties
    },
    "Wavefunction": {
        "Distance": "recommended",
        "Cartesian": "recommended",
        "ToUndirected": "recommended",
        "AddSelfLoops": "recommended",
        "NormalizeFeatures": "recommended",
        "GCNNorm": "compatible",
        "DropEdge": "warning",
        "MaskFeatures": "warning",
        "VirtualNode": "incompatible",
        "RandomNodeSplit": "incompatible",
        "DropNode": "incompatible",
    },
}

# Transform caching configuration
TRANSFORM_CACHE_SIZE: int = 32  # LRU cache size for transform compositions
TRANSFORM_VALIDATION_CACHE_SIZE: int = 64  # Validation result cache


@lru_cache(maxsize=TRANSFORM_CACHE_SIZE)
def get_transformation_constants(include_registry_info: bool = False) -> dict[str, Any]:
    """
    Get transformation system constants with optional registry information.

    Args:
        include_registry_info: If True, include detailed transform registry info

    Returns:
        Dictionary of transformation-related constants

    Raises:
        ConfigurationError: If transformation config cannot be loaded
    """
    try:
        # Check if transformation system is available
        transformation_available = _check_transformation_system_availability()

        constants = {
            "system_available": transformation_available,
            "core_transforms": CORE_TRANSFORMS.copy(),
            "categories": TRANSFORM_CATEGORIES.copy(),
            "validation_modes": TRANSFORM_VALIDATION_MODES.copy(),
            "default_validation_mode": DEFAULT_TRANSFORM_VALIDATION_MODE,
            "default_setup_name": DEFAULT_EXPERIMENTAL_SETUP_NAME,
            "max_setups": MAX_EXPERIMENTAL_SETUPS,
            "max_transforms_per_setup": MAX_TRANSFORMS_PER_SETUP,
            "handler_compatibility": HANDLER_TRANSFORM_COMPATIBILITY.copy(),
        }

        # Add configuration-based constants if available
        if transformation_available:
            try:
                config_constants = _get_config_transformation_constants()
                constants.update(config_constants)
            except Exception as e:
                logger.warning(f"Could not load transformation config constants: {e}")

        # Add registry info if requested and available
        if include_registry_info and transformation_available:
            try:
                registry_info = _get_transform_registry_info()
                constants["registry_info"] = registry_info
            except Exception as e:
                logger.warning(f"Could not load transform registry info: {e}")

        return constants

    except Exception as e:
        raise ConfigurationError(
            f"Failed to get transformation constants: {e}",
            context={"phase": "Phase1_transformation_integration"},
        )


def _check_transformation_system_availability() -> bool:
    """Check if transformation system modules are available."""
    global TRANSFORMATION_SYSTEM_AVAILABLE

    if TRANSFORMATION_SYSTEM_AVAILABLE is not None:
        return TRANSFORMATION_SYSTEM_AVAILABLE

    try:
        # Check for graph_transforms module
        import graph_transforms

        # Check for enhanced config_accessors functions
        from milia_pipeline.config.config_accessors import get_transformation_config

        # Check for transformation containers
        from milia_pipeline.config.config_containers import ExperimentalSetup, TransformationConfig

        TRANSFORMATION_SYSTEM_AVAILABLE = True
        logger.debug("Transformation system is available")

    except ImportError as e:
        TRANSFORMATION_SYSTEM_AVAILABLE = False
        logger.debug(f"Transformation system not available: {e}")

    return TRANSFORMATION_SYSTEM_AVAILABLE


def _get_config_transformation_constants() -> dict[str, Any]:
    """Get transformation constants from configuration."""
    from milia_pipeline.config.config_accessors import (
        get_transformation_config,
        list_experimental_setups,
    )

    try:
        transform_config = get_transformation_config()
        setups = list_experimental_setups()

        return {
            "experimental_setups": setups,
            "default_setup": transform_config.default_setup
            if hasattr(transform_config, "default_setup")
            else DEFAULT_EXPERIMENTAL_SETUP_NAME,
            "validation_enabled": transform_config.validation.enabled
            if hasattr(transform_config, "validation")
            else True,
            "setup_count": len(setups),
        }
    except Exception as e:
        logger.debug(f"Could not load transformation config: {e}")
        return {}


def _get_transform_registry_info() -> dict[str, Any]:
    """Get information from transform registry."""
    try:
        from milia_pipeline.transformations.graph_transforms import get_graph_transforms

        gt = get_graph_transforms()
        available_transforms = gt.get_available_transforms()

        return {
            "total_transforms": sum(
                len(transforms) for transforms in available_transforms.values()
            ),
            "transforms_by_category": {
                cat: len(transforms) for cat, transforms in available_transforms.items()
            },
            "available_categories": list(available_transforms.keys()),
        }
    except Exception as e:
        logger.debug(f"Could not get transform registry info: {e}")
        return {}


def get_handler_transform_compatibility(handler_type: str, transform_name: str) -> str:
    """
    Check compatibility between a handler and transform.

    Args:
        handler_type: Type of handler ('DFT', 'DMC', etc.)
        transform_name: Name of the transform

    Returns:
        Compatibility status: 'compatible', 'warning', 'incompatible', or 'unknown'

    Raises:
        HandlerNotAvailableError: If handler type is not supported
    """
    if not is_handler_type_supported(handler_type):
        raise HandlerNotAvailableError(
            f"Handler type '{handler_type}' is not supported",
            requested_dataset_type=handler_type,
            available_types=get_supported_handler_types(),
        )

    handler_compat = HANDLER_TRANSFORM_COMPATIBILITY.get(handler_type, {})
    return handler_compat.get(transform_name, "unknown")


def get_compatible_transforms_for_handler(
    handler_type: str, include_warnings: bool = False
) -> list[str]:
    """
    Get list of compatible transforms for a handler type.

    Args:
        handler_type: Type of handler
        include_warnings: If True, include transforms with warnings

    Returns:
        List of compatible transform names

    Raises:
        HandlerNotAvailableError: If handler type is not supported
    """
    if not is_handler_type_supported(handler_type):
        raise HandlerNotAvailableError(
            f"Handler type '{handler_type}' is not supported",
            requested_dataset_type=handler_type,
            available_types=get_supported_handler_types(),
        )

    handler_compat = HANDLER_TRANSFORM_COMPATIBILITY.get(handler_type, {})

    compatible = []
    for transform_name, status in handler_compat.items():
        if status == "compatible" or status == "warning" and include_warnings:
            compatible.append(transform_name)

    return compatible


def get_incompatible_transforms_for_handler(handler_type: str) -> list[str]:
    """
    Get list of incompatible transforms for a handler type.

    Args:
        handler_type: Type of handler

    Returns:
        List of incompatible transform names

    Raises:
        HandlerNotAvailableError: If handler type is not supported
    """
    if not is_handler_type_supported(handler_type):
        raise HandlerNotAvailableError(
            f"Handler type '{handler_type}' is not supported",
            requested_dataset_type=handler_type,
            available_types=get_supported_handler_types(),
        )

    handler_compat = HANDLER_TRANSFORM_COMPATIBILITY.get(handler_type, {})

    return [name for name, status in handler_compat.items() if status == "incompatible"]


def validate_experimental_setup_for_handler(handler_type: str, setup_name: str) -> dict[str, Any]:
    """
    Validate an experimental setup against handler compatibility requirements.

    Args:
        handler_type: Type of handler
        setup_name: Name of experimental setup

    Returns:
        Dictionary with validation results including compatibility issues

    Raises:
        HandlerNotAvailableError: If handler type is not supported
        ConfigurationError: If setup cannot be loaded
    """
    if not is_handler_type_supported(handler_type):
        raise HandlerNotAvailableError(
            f"Handler type '{handler_type}' is not supported",
            requested_dataset_type=handler_type,
            available_types=get_supported_handler_types(),
        )

    try:
        from milia_pipeline.config.config_accessors import get_experimental_setup

        setup = get_experimental_setup(setup_name)

        validation_result = {
            "setup_name": setup_name,
            "handler_type": handler_type,
            "is_compatible": True,
            "warnings": [],
            "errors": [],
            "transform_count": len(setup.transforms) if hasattr(setup, "transforms") else 0,
        }

        # Check each transform in the setup
        for transform_spec in setup.transforms:
            transform_name = (
                transform_spec.name
                if hasattr(transform_spec, "name")
                else transform_spec.get("name")
            )

            if transform_name:
                compat_status = get_handler_transform_compatibility(handler_type, transform_name)

                if compat_status == "incompatible":
                    validation_result["errors"].append(
                        f"Transform '{transform_name}' is incompatible with {handler_type} handler"
                    )
                    validation_result["is_compatible"] = False
                elif compat_status == "warning":
                    validation_result["warnings"].append(
                        f"Transform '{transform_name}' may have issues with {handler_type} handler"
                    )

        return validation_result

    except Exception as e:
        raise ConfigurationError(
            f"Failed to validate experimental setup '{setup_name}' for handler '{handler_type}'",
            context={"setup_name": setup_name, "handler_type": handler_type},
            details=str(e),
        )


# ==========================================
# HANDLER INTEGRATION UTILITIES
# ==========================================


def create_handler_config_from_constants(handler_type: str) -> dict[str, Any]:
    """
    Create a handler configuration dictionary from constants.

    Args:
        handler_type: Type of handler to create config for

    Returns:
        Configuration dictionary suitable for handler creation

    Raises:
        HandlerConfigurationError: If configuration cannot be created
    """
    try:
        # Get base configuration
        dataset_config = get_dataset_config()
        handler_constants = get_handler_constants(handler_type)

        # Build handler configuration
        handler_config = {
            "dataset_type": handler_type,
            "processing_config": dataset_config.get("processing_config", {}),
            "filter_config": dataset_config.get("filter_config", {}),
            "handler_constants": handler_constants,
            "feature_support": handler_constants["feature_support"],
        }

        # Add handler-specific configuration
        if handler_type == "DMC":
            handler_config["uncertainty_config"] = dataset_config.get("uncertainty_config", {})
        elif handler_type == "DFT":
            handler_config["vibrational_config"] = dataset_config.get("vibrational_config", {})

        # Validate the created configuration
        validation_errors = validate_handler_configuration(handler_type, handler_config)
        if validation_errors:
            raise HandlerConfigurationError(
                "Handler configuration validation failed",
                handler_type=handler_type,
                config_validation_errors=validation_errors,
            )

        return handler_config

    except Exception as e:
        if isinstance(e, HandlerConfigurationError):
            raise

        raise HandlerConfigurationError(
            "Failed to create handler configuration", handler_type=handler_type, details=str(e)
        )


def get_migration_compatibility_constants() -> dict[str, Any]:
    """
    Get constants for ensuring backward compatibility during migration.

    UPDATED Phase 3: Now includes registry integration status.

    Returns:
        Dictionary with migration compatibility constants
    """
    return {
        "migration_phase": "Phase 3 - Registry Integration",
        "handler_pattern_enabled": True,
        "legacy_fallback_enabled": True,
        "supported_handler_types": get_supported_handler_types(),
        "default_handler_type": get_default_handler_type(),
        "handler_factory_available": True,
        "backward_compatibility_mode": "full",
        "migration_validation_enabled": True,
        "handler_integration_testing": True,
        "registry_integration": {
            "available": _REGISTRY_AVAILABLE,
            "cache_invalidation_enabled": _CACHE_INVALIDATION_REGISTERED,
            "dynamic_lookup_enabled": _REGISTRY_AVAILABLE,
        },
    }


def validate_handler_environment() -> dict[str, bool]:
    """
    Validate that the environment supports handler pattern operations.

    UPDATED Phase 3: Includes registry validation.

    Returns:
        Dictionary of validation results
    """
    # Import the module to access lazy-loaded constants
    import milia_pipeline.config.config_constants as const_module

    validation_results = {}

    # Always set registry integration status first (these don't require config)
    _init_registry()  # Ensure registry is initialized
    validation_results["registry_available"] = _REGISTRY_AVAILABLE
    validation_results["cache_invalidation_registered"] = _CACHE_INVALIDATION_REGISTERED

    try:
        # Check configuration availability
        validation_results["config_available"] = bool(_TEMP_CONFIG)

        # Check required constants (these are lazy-loaded, access through module)
        try:
            validation_results["atomic_energies_available"] = bool(
                const_module.ATOMIC_ENERGIES_HARTREE
            )
        except Exception:
            validation_results["atomic_energies_available"] = False

        try:
            validation_results["conversion_factors_available"] = bool(const_module.HAR2EV)
        except Exception:
            validation_results["conversion_factors_available"] = False

        try:
            validation_results["dataset_constants_available"] = bool(
                const_module._dataset_constants
            )
        except Exception:
            validation_results["dataset_constants_available"] = False

        # Check handler support using dynamic functions
        validation_results["handler_types_defined"] = len(get_supported_handler_types()) > 0

        # Check handler configuration capabilities
        for handler_type in get_supported_handler_types():
            try:
                get_handler_constants(handler_type)
                validation_results[f"{handler_type.lower()}_handler_constants_available"] = True
            except Exception:
                validation_results[f"{handler_type.lower()}_handler_constants_available"] = False

        # Check migration compatibility
        validation_results["migration_compatibility_available"] = bool(
            get_migration_compatibility_constants()
        )

        # Overall validation
        validation_results["handler_environment_ready"] = all(
            [
                validation_results.get("config_available", False),
                validation_results["handler_types_defined"],
                validation_results.get("registry_available", False),
            ]
        )

    except Exception as e:
        logger.error(f"Handler environment validation failed: {e}")
        validation_results["validation_error"] = str(e)
        validation_results["handler_environment_ready"] = False

    return validation_results


# ==========================================
# TRANSFORMATION CACHE MANAGEMENT
# ==========================================


def clear_transformation_caches():
    """Clear all transformation-related caches."""
    get_transformation_constants.cache_clear()
    logger.info("Transformation caches cleared")


def get_transformation_cache_info() -> dict[str, Any]:
    """
    Get information about transformation cache usage.

    Returns:
        Dictionary with cache statistics
    """
    return {
        "transformation_constants_cache": get_transformation_constants.cache_info()._asdict(),
        "cache_enabled": True,
        "max_cache_size": TRANSFORM_CACHE_SIZE,
    }


# ==========================================
# PHASE 3: Updated Handler Caching and Performance
# ==========================================


@lru_cache(maxsize=16)
def get_cached_handler_config(handler_type: str, config_hash: int) -> dict[str, Any]:
    """
    Get cached handler configuration to improve performance.

    Args:
        handler_type: Type of handler
        config_hash: Hash of configuration for cache key

    Returns:
        Cached handler configuration
    """
    return create_handler_config_from_constants(handler_type)


def clear_handler_caches():
    """
    Clear all handler-related caches.

    UPDATED Phase 3: Also triggers registry cache invalidation callback.
    """
    get_handler_constants.cache_clear()
    get_handler_identifier_keys.cache_clear()
    get_cached_handler_config.cache_clear()
    logger.info("Handler caches cleared")


def get_handler_cache_info() -> dict[str, Any]:
    """
    Get information about handler cache usage.

    UPDATED Phase 3: Added registry integration status.

    Returns:
        Dictionary with cache statistics
    """
    return {
        "handler_constants_cache": get_handler_constants.cache_info()._asdict(),
        "handler_identifier_keys_cache": get_handler_identifier_keys.cache_info()._asdict(),
        "handler_config_cache": get_cached_handler_config.cache_info()._asdict(),
        "cache_enabled": True,
        "max_cache_size": 32,
        "registry_integration": {
            "available": _REGISTRY_AVAILABLE,
            "cache_invalidation_registered": _CACHE_INVALIDATION_REGISTERED,
        },
    }


def clear_all_caches():
    """
    Clear all caches (handlers and transformations).

    UPDATED Phase 3: Uses centralized cache invalidation.
    """
    _invalidate_all_handler_caches()
    clear_transformation_caches()
    logger.info("All caches cleared")


def get_all_cache_info() -> dict[str, Any]:
    """
    Get comprehensive cache information for all systems.

    Returns:
        Dictionary with all cache statistics
    """
    return {
        "handler_caches": get_handler_cache_info(),
        "transformation_caches": get_transformation_cache_info(),
        "timestamp": __import__("datetime").datetime.now().isoformat(),
    }


# ==========================================
# LEGACY INTEGRATION AND FALLBACKS
# ==========================================


def get_legacy_compatible_constants(handler_type: str | None = None) -> dict[str, Any]:
    """
    Get constants in current handler-based architecture format.

    This function ensures all lazy-loaded constants are properly initialized
    before building the constants dictionary.

    Note: Legacy DATASET_CONFIG and *_LEGACY constants have been removed
    as they reference obsolete configuration structure. Use handler-based
    constants instead.

    Args:
        handler_type: Optional handler type for context

    Returns:
        Dictionary with handler-compatible constants
    """
    # Import here to ensure module is fully loaded
    import milia_pipeline.config.config_constants as const_module

    # Force lazy loading of all required constants by accessing them
    # This triggers __getattr__ and populates _CONSTANTS_CACHE
    atomic_energies = const_module.ATOMIC_ENERGIES_HARTREE
    heavy_atoms = const_module.HEAVY_ATOM_SYMBOLS_TO_Z
    har2ev = const_module.HAR2EV
    raw_filename = const_module.RAW_NPZ_FILENAME_CACHED
    raw_url = const_module.RAW_DATA_DOWNLOAD_URL_CACHED
    root_dir = const_module.DATASET_ROOT_DIR_CACHED
    processed_filename = const_module.PROCESSED_DATA_FILENAME

    # Build the dictionary with handler-based constants
    constants = {
        # Core constants
        "ATOMIC_ENERGIES_HARTREE": atomic_energies,
        "HEAVY_ATOM_SYMBOLS_TO_Z": heavy_atoms,
        "HAR2EV": har2ev,
        # Handler-based dataset constants
        "RAW_NPZ_FILENAME": raw_filename,
        "RAW_DATA_DOWNLOAD_URL": raw_url,
        "DATASET_ROOT_DIR": root_dir,
        "PROCESSED_DATA_FILENAME": processed_filename,
    }

    # Add handler-specific constants if requested
    if handler_type and is_handler_type_supported(handler_type):
        try:
            handler_constants = get_handler_constants(handler_type)
            constants[f"{handler_type}_HANDLER_CONSTANTS"] = handler_constants
        except Exception as e:
            logger.warning(f"Could not get handler constants for {handler_type}: {e}")

    return constants


def ensure_handler_constant_compatibility() -> bool:
    """
    Ensure that handler constants are compatible with legacy code expectations.

    Returns:
        True if all compatibility checks pass
    """
    # Import the module to access lazy-loaded constants
    import milia_pipeline.config.config_constants as const_module

    try:
        # Check that legacy constants are still available
        legacy_checks = [
            bool(const_module.ATOMIC_ENERGIES_HARTREE),
            bool(const_module.HEAVY_ATOM_SYMBOLS_TO_Z),
            bool(const_module.HAR2EV),
            bool(const_module.RAW_NPZ_FILENAME_CACHED),
            bool(const_module.DATASET_ROOT_DIR_CACHED),
        ]

        # Check handler constants can be created
        handler_checks = []
        for handler_type in get_supported_handler_types():
            try:
                constants = get_handler_constants(handler_type)
                handler_checks.append(bool(constants))
            except Exception:
                handler_checks.append(False)

        # Check environment validation
        env_validation = validate_handler_environment()
        environment_ready = env_validation.get("handler_environment_ready", False)

        return all(legacy_checks) and all(handler_checks) and environment_ready

    except Exception as e:
        logger.error(f"Handler constant compatibility check failed: {e}")
        return False


# ==========================================
# COMPREHENSIVE DEBUGGING AND DIAGNOSTICS
# ==========================================


def get_complete_constants_debug_info() -> dict[str, Any]:
    """
    Get comprehensive debug information about all constants (handlers + transformations).

    Returns:
        Dictionary with comprehensive debug information
    """
    # Import the module to access lazy-loaded constants
    import milia_pipeline.config.config_constants as const_module

    debug_info = {
        "config_status": {
            "temp_config_loaded": bool(_TEMP_CONFIG),
            "dataset_constants_available": bool(const_module._dataset_constants),
        },
        "handler_support": {
            "supported_types": get_supported_handler_types(),
            "default_type": get_default_handler_type(),
            "feature_matrix": HANDLER_FEATURE_SUPPORT,
            "property_requirements": HANDLER_REQUIRED_PROPERTIES,
            "optional_properties": HANDLER_OPTIONAL_PROPERTIES,
        },
        "transformation_support": {
            "system_available": TRANSFORMATION_SYSTEM_AVAILABLE,
            "core_transforms": len(CORE_TRANSFORMS),
            "categories": TRANSFORM_CATEGORIES,
            "validation_modes": TRANSFORM_VALIDATION_MODES,
            "default_validation_mode": DEFAULT_TRANSFORM_VALIDATION_MODE,
            "compatibility_matrix": HANDLER_TRANSFORM_COMPATIBILITY,
        },
        "constants_status": {
            "atomic_energies_count": len(const_module.ATOMIC_ENERGIES_HARTREE),
            "heavy_atoms_count": len(const_module.HEAVY_ATOM_SYMBOLS_TO_Z),
            "conversion_factor": const_module.HAR2EV,
            "processed_filename": const_module.PROCESSED_DATA_FILENAME,
        },
        "registry_integration": {
            "available": _REGISTRY_AVAILABLE,
            "cache_invalidation_registered": _CACHE_INVALIDATION_REGISTERED,
            "import_error": _REGISTRY_IMPORT_ERROR,
        },
        "cache_status": get_all_cache_info(),
        "environment_validation": validate_complete_environment(),
        "compatibility_status": ensure_complete_compatibility(),
    }

    # Add handler-specific debug info
    debug_info["handler_constants"] = {}
    for handler_type in get_supported_handler_types():
        try:
            constants = get_handler_constants(handler_type)
            debug_info["handler_constants"][handler_type] = {
                "constants_available": True,
                "constant_count": len(constants),
                "required_properties": constants.get("required_properties", []),
                "feature_support": constants.get("feature_support", {}),
                "molecule_creation_strategy": constants.get(
                    "molecule_creation_strategy", "unknown"
                ),
            }
        except Exception as e:
            debug_info["handler_constants"][handler_type] = {
                "constants_available": False,
                "error": str(e),
            }

    # Add transformation-specific debug info
    if TRANSFORMATION_SYSTEM_AVAILABLE:
        try:
            transform_constants = get_transformation_constants(include_registry_info=True)
            debug_info["transformation_constants"] = {
                "constants_available": True,
                "experimental_setups": transform_constants.get("setup_count", 0),
                "registry_info": transform_constants.get("registry_info", {}),
            }
        except Exception as e:
            debug_info["transformation_constants"] = {"constants_available": False, "error": str(e)}
    else:
        debug_info["transformation_constants"] = {
            "constants_available": False,
            "reason": "Transformation system not available",
        }

    return debug_info


def validate_complete_environment() -> dict[str, bool]:
    """
    Validate that the environment supports both handler and transformation operations.

    Returns:
        Dictionary of validation results
    """
    validation_results = validate_handler_environment()

    # Add transformation system validation
    try:
        validation_results["transformation_system_available"] = (
            _check_transformation_system_availability()
        )

        if validation_results["transformation_system_available"]:
            transform_constants = get_transformation_constants()
            validation_results["transformation_config_loadable"] = bool(transform_constants)
            validation_results["experimental_setups_available"] = (
                transform_constants.get("setup_count", 0) > 0
            )
        else:
            validation_results["transformation_config_loadable"] = False
            validation_results["experimental_setups_available"] = False

        # Overall validation
        validation_results["complete_environment_ready"] = all(
            [
                validation_results.get("handler_environment_ready", False),
                validation_results.get("transformation_system_available", False),
            ]
        )

    except Exception as e:
        logger.error(f"Transformation environment validation failed: {e}")
        validation_results["transformation_validation_error"] = str(e)
        validation_results["complete_environment_ready"] = False

    return validation_results


def ensure_complete_compatibility() -> bool:
    """
    Ensure that both handler and transformation constants are compatible.

    Returns:
        True if all compatibility checks pass
    """
    handler_compat = ensure_handler_constant_compatibility()

    try:
        # Check transformation system compatibility
        if not _check_transformation_system_availability():
            logger.debug("Transformation system not available - compatibility check skipped")
            return handler_compat

        transform_constants = get_transformation_constants()
        transformation_compat = bool(transform_constants)

        # Check handler-transform compatibility matrix
        compat_matrix_valid = bool(HANDLER_TRANSFORM_COMPATIBILITY)

        return handler_compat and transformation_compat and compat_matrix_valid

    except Exception as e:
        logger.error(f"Transformation compatibility check failed: {e}")
        return False


# Deprecated alias - for backward compatibility
get_handler_constants_debug_info = get_complete_constants_debug_info


# ==========================================
# MODULE SELF-TEST
# ==========================================

if __name__ == "__main__":
    # Module self-test and diagnostics
    # Import the module to access lazy-loaded constants
    import milia_pipeline.config.config_constants as const_module

    print("Config Constants - Handler-Based Pattern Development & Transformation System Support")
    print("Phase 3: Registry Integration")
    print("=" * 70)

    # Initialize registry
    _init_registry()

    # Test registry integration
    print("\nRegistry Integration:")
    print(f"  Registry available: {_REGISTRY_AVAILABLE}")
    print(f"  Cache invalidation registered: {_CACHE_INVALIDATION_REGISTERED}")

    # Test basic constants (access through module for lazy-loaded)
    print("\nBasic Constants:")
    print(f"  Atomic energies available: {len(const_module.ATOMIC_ENERGIES_HARTREE)}")
    print(f"  Heavy atoms mapping: {len(const_module.HEAVY_ATOM_SYMBOLS_TO_Z)}")
    print(f"  Conversion factor (HAR2EV): {const_module.HAR2EV}")
    print(f"  Dataset filename: {const_module.PROCESSED_DATA_FILENAME}")

    # Test dynamic handler support
    print("\nHandler Support (Dynamic):")
    print(f"  Supported types: {get_supported_handler_types()}")
    print(f"  Default type: {get_default_handler_type()}")

    # Test handler constants
    for handler_type in get_supported_handler_types():
        try:
            constants = get_handler_constants(handler_type)
            print(f"\n{handler_type} handler: {len(constants)} constants")

            features = constants.get("feature_support", {})
            supported_features = [k for k, v in features.items() if v]
            print(f"  Supported features: {supported_features}")

            required_props = constants.get("required_properties", [])
            print(f"  Required properties: {required_props}")

            strategy = constants.get("molecule_creation_strategy", "unknown")
            print(f"  Molecule creation strategy: {strategy}")

        except Exception as e:
            print(f"\n{handler_type} handler error: {e}")

    # Test transformation support
    print("\nTransformation Support:")
    print(f"  System available: {_check_transformation_system_availability()}")
    print(f"  Core transforms: {len(CORE_TRANSFORMS)}")
    print(f"  Categories: {TRANSFORM_CATEGORIES}")

    if _check_transformation_system_availability():
        try:
            transform_constants = get_transformation_constants(include_registry_info=True)
            print(f"  Experimental setups: {transform_constants.get('setup_count', 0)}")

            registry_info = transform_constants.get("registry_info", {})
            if registry_info:
                print(f"  Total transforms in registry: {registry_info.get('total_transforms', 0)}")
                print(f"  Categories in registry: {registry_info.get('available_categories', [])}")
        except Exception as e:
            print(f"  Transformation constants error: {e}")

    # Test handler-transform compatibility
    print("\nHandler-Transform Compatibility:")
    for handler_type in get_supported_handler_types():
        compatible = get_compatible_transforms_for_handler(handler_type)
        incompatible = get_incompatible_transforms_for_handler(handler_type)
        print(f"  {handler_type}:")
        print(f"    Compatible transforms: {len(compatible)}")
        print(f"    Incompatible transforms: {len(incompatible)}")
        if incompatible:
            print(f"    Incompatible: {incompatible}")

    # Test environment validation
    print("\nEnvironment Validation:")
    env_results = validate_complete_environment()
    for check, result in env_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {check}: {status}")

    # Test complete compatibility
    print("\nComplete Compatibility:")
    compat_result = ensure_complete_compatibility()
    print(f"  Status: {'✓ PASS' if compat_result else '✗ FAIL'}")

    # Cache statistics
    print("\nCache Information:")
    cache_info = get_all_cache_info()

    # Handler caches
    handler_caches = cache_info.get("handler_caches", {})
    for cache_name, stats in handler_caches.items():
        if isinstance(stats, dict) and "hits" in stats:
            print(f"  {cache_name}: {stats['hits']} hits, {stats['misses']} misses")

    # Transformation caches
    transform_caches = cache_info.get("transformation_caches", {})
    for cache_name, stats in transform_caches.items():
        if isinstance(stats, dict) and "hits" in stats:
            print(f"  {cache_name}: {stats['hits']} hits, {stats['misses']} misses")

    print("\n✓ Config constants ready for Phase 3 - Registry Integration!")


# ==========================================
# Lazy Module-Level Constant Loading (Avoid Circular Import)
# ==========================================


def __getattr__(name: str):
    """
    Lazy-load constants on first access to avoid circular imports.

    This module-level __getattr__ is called when an attribute is not found,
    allowing us to defer config loading until constants are actually needed.

    This prevents the circular import:
    config_constants (module load) -> load_config() -> config_handler ->
    config_constants (partial) -> ImportError
    """
    global _TEMP_CONFIG

    # Lazy initialize config on first constant access
    if _TEMP_CONFIG is None:
        from milia_pipeline.config.config_loader import load_config

        _TEMP_CONFIG = load_config()

    # Load and cache the requested constant
    if name == "ATOMIC_ENERGIES_HARTREE":
        if name not in _CONSTANTS_CACHE:
            _CONSTANTS_CACHE[name] = _get_config_value(
                _TEMP_CONFIG, "atomic_energies_hartree", dict
            )
        return _CONSTANTS_CACHE[name]

    elif name == "HEAVY_ATOM_SYMBOLS_TO_Z":
        if name not in _CONSTANTS_CACHE:
            _CONSTANTS_CACHE[name] = _get_config_value(
                _TEMP_CONFIG, "heavy_atom_symbols_to_z", dict
            )
        return _CONSTANTS_CACHE[name]

    elif name == "HAR2EV":
        if name not in _CONSTANTS_CACHE:
            global_constants = _get_config_value(_TEMP_CONFIG, "global_constants", dict)
            _CONSTANTS_CACHE[name] = _get_config_value(
                global_constants, "har2ev", (int, float), parent_key="global_constants"
            )
        return _CONSTANTS_CACHE[name]

    elif name == "BOHR_TO_ANGSTROM":
        if name not in _CONSTANTS_CACHE:
            global_constants = _get_config_value(_TEMP_CONFIG, "global_constants", dict)
            _CONSTANTS_CACHE[name] = _get_config_value(
                global_constants, "bohr_to_angstrom", (int, float), parent_key="global_constants"
            )
        return _CONSTANTS_CACHE[name]

    elif name == "RAW_NPZ_FILENAME_CACHED":
        if name not in _CONSTANTS_CACHE:
            constants = get_dataset_constants()
            _CONSTANTS_CACHE[name] = constants[0]
        return _CONSTANTS_CACHE[name]

    elif name == "RAW_DATA_DOWNLOAD_URL_CACHED":
        if name not in _CONSTANTS_CACHE:
            constants = get_dataset_constants()
            _CONSTANTS_CACHE[name] = constants[1]
        return _CONSTANTS_CACHE[name]

    elif name == "DATASET_ROOT_DIR_CACHED":
        if name not in _CONSTANTS_CACHE:
            constants = get_dataset_constants()
            _CONSTANTS_CACHE[name] = constants[2]
        return _CONSTANTS_CACHE[name]
    # ---
    elif name == "PROCESSED_DATA_FILENAME":
        if name not in _CONSTANTS_CACHE:
            # Always fetch fresh to ensure config is loaded
            constants = get_dataset_constants()
            raw_filename = constants[0]

            if raw_filename:
                _CONSTANTS_CACHE[name] = Path(raw_filename).stem + ".pt"
                # Also update RAW_NPZ_FILENAME_CACHED for consistency
                _CONSTANTS_CACHE["RAW_NPZ_FILENAME_CACHED"] = raw_filename
            else:
                # Retry fetching config - it may not have been loaded on first attempt
                from milia_pipeline.config.config_loader import load_config

                load_config()
                dataset_config = _get_dataset_config_local()
                raw_filename = _get_config_value(dataset_config, "raw_npz_filename", str)
                if raw_filename:
                    _CONSTANTS_CACHE[name] = Path(raw_filename).stem + ".pt"
                else:
                    logger.warning(
                        "The raw NPZ filename is not specified in config.yaml. Falling back to default 'data.pt'."
                    )
                    _CONSTANTS_CACHE[name] = "data.pt"
        return _CONSTANTS_CACHE[name]
        # ---

    elif name == "_dataset_constants":
        if name not in _CONSTANTS_CACHE:
            _CONSTANTS_CACHE[name] = get_dataset_constants()
        return _CONSTANTS_CACHE[name]
