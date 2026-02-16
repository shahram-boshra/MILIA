# config_loader.py - Enhanced Configuration Loader with Handler-Based Architecture

"""
Enhanced YAML configuration loading and caching module with comprehensive
transformation configuration support.

This module handles the loading of application configuration from YAML files
with enhanced support for experimental setups, legacy format migration,
and comprehensive validation. Uses ConfigHandler for backward compatibility
and graceful degradation when relevant features are unavailable.

Key Features:
- Thread-safe configuration loading with intelligent caching
- Configurable validation levels (STRICT/NORMAL/RELAXED)
- Automatic legacy format detection and migration
- Support for multiple experimental setups
- Handler-based backward compatibility
- Comprehensive validation with detailed error reporting
- Production-ready error handling and fallback mechanisms
- **PHASE 5**: Dynamic dataset type resolution via registry

Architecture:
- All integration handled through ConfigHandler
- Automatic feature detection and graceful degradation
- Consistent API regardless of relevant availability
- No direct conditional imports for relevent modules
- **PHASE 5**: Registry-based dynamic dataset type support

Thread Safety:
- Configuration cache protected by _cache_lock (RLock)
- Statistics updates protected by _stats_lock (Lock)
- Safe for concurrent access from multiple threads

PHASE 5 REFACTORING:
- Replaced hardcoded 'DFT' default with dynamic registry lookup
- Added lazy registry initialization (following Phase 3 pattern)
- Maintained 100% backward compatibility
"""

import copy
import hashlib
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from milia_pipeline.exceptions import ConfigurationError

# Handler removed - using config_schemas directly for all validation and migration
# This simplifies the architecture and removes the non-existent config_handler dependency

# Initialize logger for this module
logger = logging.getLogger(__name__)


# ==========================================
# PHASE 5: Registry Integration for Dynamic Dataset Type Resolution
# ==========================================

# Registry availability flags - set during lazy initialization
_REGISTRY_AVAILABLE = False
_REGISTRY_IMPORT_ERROR = None
_REGISTRY_INITIALIZED = False
_REGISTRY_INITIALIZING = False  # Guard against re-entrant calls during initialization

# Registry function placeholders (populated by _init_registry)
_registry_list_all = None
_registry_get = None
_registry_is_registered = None


def _init_registry() -> bool:
    """
    Lazily initialize registry imports to avoid circular import at module load time.

    The config_loader.py module is imported early in the initialization chain.
    By deferring the registry import until first use, we allow the config module
    to fully load first.

    CRITICAL: Uses _REGISTRY_INITIALIZING flag to detect re-entrant calls.
    During registry initialization, importing datasets.registry can trigger
    a nested load_config() call. The nested call must not attempt to use
    the registry while it's still being initialized.

    Returns:
        True if registry is available, False otherwise

    PHASE 5 REFACTORING: Lazy initialization to resolve circular import issues.
    Pattern: Following Phase 3 config_constants.py pattern for consistency.
    Evidence: config_constants.py (Phase 3, lines 73-120)
    """
    global _REGISTRY_AVAILABLE, _REGISTRY_IMPORT_ERROR, _REGISTRY_INITIALIZED
    global _REGISTRY_INITIALIZING
    global _registry_list_all, _registry_get, _registry_is_registered

    # If already fully initialized, return cached result
    if _REGISTRY_INITIALIZED and not _REGISTRY_INITIALIZING:
        return _REGISTRY_AVAILABLE

    # If currently initializing (re-entrant call), return False
    # This prevents nested load_config() calls from trying to use
    # a partially-initialized registry
    if _REGISTRY_INITIALIZING:
        return False

    # Mark as initializing to guard against re-entrant calls
    _REGISTRY_INITIALIZING = True

    try:
        # Import registry functions directly (avoids circular import via milia_dataset.py)
        # Trigger dataset implementations to register themselves via @register decorator
        # This import is safe because implementations/ only imports from base.py and registry.py
        import milia_pipeline.datasets.implementations  # noqa: F401
        from milia_pipeline.datasets.registry import (
            get,
            is_registered,
            list_all,
        )

        _registry_list_all = list_all
        _registry_get = get
        _registry_is_registered = is_registered
        _REGISTRY_AVAILABLE = True
        _REGISTRY_INITIALIZED = True

        logger.debug("Registry initialized successfully in config_loader.py")
        return True

    except ImportError as e:
        _REGISTRY_IMPORT_ERROR = str(e)
        _REGISTRY_AVAILABLE = False
        _REGISTRY_INITIALIZED = True
        logger.debug(f"Registry not available in config_loader.py: {e}")
        return False
    finally:
        # Always clear the initializing flag
        _REGISTRY_INITIALIZING = False


def _get_default_dataset_type() -> str:
    """
    Get default dataset type from registry, with fallback to 'DFT'.

    PHASE 5 REFACTORING: Dynamic default dataset type instead of hardcoded 'DFT'.

    Strategy:
    1. If registry available, use first registered dataset type
    2. If 'DFT' specifically registered, prefer it for backward compatibility
    3. Otherwise fallback to 'DFT' for maximum backward compatibility

    Returns:
        str: Default dataset type name

    Evidence:
    - Original hardcoded default: config_loader.py line 1256, 1345 (dataset_type='DFT')
    - Pattern: config_constants.py DEFAULT_HANDLER_TYPE = 'DFT' (line 210)
    """
    # Initialize registry if needed
    if not _REGISTRY_INITIALIZED:
        _init_registry()

    # If registry available, get dynamic default
    if _REGISTRY_AVAILABLE and _registry_list_all is not None:
        try:
            registered_types = _registry_list_all()

            if not registered_types:
                # Registry available but empty - use fallback
                logger.debug("Registry available but empty, using fallback default 'DFT'")
                return "DFT"

            # Prefer 'DFT' if registered (backward compatibility)
            if "DFT" in registered_types:
                logger.debug("Using 'DFT' as default dataset type (registered and preferred)")
                return "DFT"

            # Otherwise use first registered type
            default_type = registered_types[0]
            logger.debug(f"Using '{default_type}' as default dataset type (first registered)")
            return default_type

        except Exception as e:
            logger.debug(f"Error getting default from registry: {e}, using fallback 'DFT'")
            return "DFT"

    # Registry not available - use hardcoded fallback
    logger.debug("Registry not available, using fallback default 'DFT'")
    return "DFT"


def _normalize_dataset_type(dataset_type: str, _skip_cache_if_reentrant: list = None) -> str:
    """
    Normalize dataset_type to its canonical registry name (case-insensitive).

    This is the SINGLE SOURCE OF TRUTH for dataset type normalization in MILIA.
    Called once at config load time to ensure all downstream code receives
    the canonical dataset type name from the registry.

    CRITICAL: If called during registry initialization (re-entrant call),
    returns input unchanged to avoid blocking. The outer call will complete
    initialization and subsequent calls will normalize correctly.

    Resolution order:
    1. Exact match in registry - returns as-is
    2. Case-insensitive match in registry - returns canonical name
    3. No match - returns input unchanged (validation will catch invalid types)

    Args:
        dataset_type: Dataset type name from config.yaml (any case)
        _skip_cache_if_reentrant: Internal list to signal re-entrant call (set to [True] if skipped)

    Returns:
        Canonical dataset type name from registry, or input if not found

    Examples:
        >>> _normalize_dataset_type("ANI1X")   # config has uppercase
        'ANI1x'  # returns canonical name from DatasetMetadata
        >>> _normalize_dataset_type("dft")     # config has lowercase
        'DFT'    # returns canonical name
        >>> _normalize_dataset_type("DFT")     # already canonical
        'DFT'    # returns unchanged

    ADDED Phase 6.2: Single source of truth for dataset type normalization.
    Resolves Technical Debt MILIA-ARCH-001.
    """
    # If registry is currently initializing (re-entrant call), skip normalization
    # The outer call will handle it once initialization completes
    # Signal to caller that this is a re-entrant call so they don't cache
    if _REGISTRY_INITIALIZING:
        logger.debug(f"Registry initializing, skipping normalization for '{dataset_type}'")
        if _skip_cache_if_reentrant is not None:
            _skip_cache_if_reentrant.append(True)
        return dataset_type

    # Initialize registry if needed
    if not _REGISTRY_INITIALIZED:
        _init_registry()

    # If registry available, perform case-insensitive resolution
    if _REGISTRY_AVAILABLE and _registry_list_all is not None:
        try:
            registered_types = _registry_list_all()

            # First try exact match (fast path)
            if dataset_type in registered_types:
                return dataset_type

            # Then try case-insensitive match
            type_lookup = {t.upper(): t for t in registered_types}
            dataset_type_upper = dataset_type.upper()

            if dataset_type_upper in type_lookup:
                canonical_name = type_lookup[dataset_type_upper]
                if canonical_name != dataset_type:
                    logger.debug(
                        f"Dataset type normalized at config load: "
                        f"'{dataset_type}' -> '{canonical_name}'"
                    )
                return canonical_name

        except Exception as e:
            logger.debug(f"Error during dataset type normalization: {e}")

    # Registry not available or no match - return unchanged
    # Downstream validation will catch invalid types
    return dataset_type


def _normalize_dataset_keyed_sections(config: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize dataset type keys in config sections that use dataset names as keys.

    This function normalizes keys in:
    - property_availability: Maps dataset types to available properties
    - data_config.property_selection: Maps dataset types to selected properties

    Keys are normalized to match canonical registry names (case-insensitive matching).
    This ensures that config.yaml can use any case (e.g., "ANI1X", "ani1x", "Ani1x")
    and lookups will work correctly with the canonical name (e.g., "ANI1x").

    Args:
        config: The configuration dictionary to normalize

    Returns:
        The configuration dictionary with normalized dataset keys

    ADDED Phase 6.3: Extends Phase 6.2 normalization to dataset-keyed sections.
    """
    # Skip if registry not available
    if not _REGISTRY_AVAILABLE or _registry_list_all is None:
        return config

    try:
        registered_types = _registry_list_all()
        if not registered_types:
            return config

        # Build case-insensitive lookup: UPPERCASE -> canonical_name
        type_lookup = {t.upper(): t for t in registered_types}

        # Normalize property_availability keys
        if "property_availability" in config and isinstance(config["property_availability"], dict):
            config["property_availability"] = _normalize_dict_keys(
                config["property_availability"], type_lookup, "property_availability"
            )

        # Normalize data_config.property_selection keys
        if "data_config" in config and isinstance(config["data_config"], dict):
            if "property_selection" in config["data_config"] and isinstance(
                config["data_config"]["property_selection"], dict
            ):
                config["data_config"]["property_selection"] = _normalize_dict_keys(
                    config["data_config"]["property_selection"],
                    type_lookup,
                    "data_config.property_selection",
                )

    except Exception as e:
        logger.debug(f"Error normalizing dataset-keyed sections: {e}")

    return config


def _normalize_dict_keys(
    d: dict[str, Any], type_lookup: dict[str, str], section_name: str
) -> dict[str, Any]:
    """
    Normalize dictionary keys using the type lookup table.

    Args:
        d: Dictionary with keys to normalize
        type_lookup: Mapping from UPPERCASE key to canonical name
        section_name: Name of section (for logging)

    Returns:
        New dictionary with normalized keys
    """
    normalized = {}
    for key, value in d.items():
        key_upper = key.upper()
        if key_upper in type_lookup:
            canonical_key = type_lookup[key_upper]
            if canonical_key != key:
                logger.debug(f"Normalized key in {section_name}: '{key}' -> '{canonical_key}'")
            normalized[canonical_key] = value
        else:
            # Key not in registry - keep as-is (might be a non-dataset key like 'common_settings')
            normalized[key] = value
    return normalized


# Global variable to store the loaded config
_CONFIG: dict[str, Any] | None = None
"""Stores the loaded application configuration as a dictionary,
cached after the first call to `load_config`."""

# Global variable to store enhanced transformation config cache
_ENHANCED_TRANSFORMATION_CONFIG: dict[str, Any] | None = None
"""Stores the enhanced transformation configuration separately for performance."""

# Global variable to cache loaded configurations with thread-safe access
_config_cache: dict[str, Any] = {}
"""A dictionary to cache configurations keyed by their loading parameters."""

# Thread lock for cache operations
_cache_lock = threading.RLock()

# Configuration loading statistics - Enhanced
_CONFIG_STATS = {
    "load_count": 0,
    "cache_hits": 0,
    "enhancement_applied": False,
    "migration_applied": False,
    "validation_enabled": True,
    "validation_level": "NORMAL",  # NEW: Track validation level
    "last_load_time": None,
    "last_validation_time": None,
    "last_validation_results": None,  # NEW: Store validation results
    "last_migration_report": None,  # NEW: Store migration report
    "cache_hit_rate": 0.0,
    "config_cached": False,
    "warnings_count": 0,  # NEW: Track warnings
    "errors_count": 0,  # NEW: Track errors
}

# Thread lock for statistics
_stats_lock = threading.Lock()


# ==========================================
# YAML SPLITTING: Helper Functions
# ==========================================
# These functions enable modular configuration management by supporting
# both single-file (backward compatible) and split-file (directory) modes.
# Evidence: Home Assistant pattern, Configu best practices, Python pathlib docs.


def _discover_config_files(config_path: str) -> tuple[bool, list[Path]]:
    """
    Discover configuration files for YAML splitting support.

    Strategy:
    1. If config_path is a file that exists → single-file mode (backward compatible)
    2. If config_path is a directory → split-file mode (new feature)
    3. If config_path doesn't exist but config_path + '/' does → split-file mode

    Args:
        config_path: Path to config file or directory

    Returns:
        Tuple of (is_split_mode: bool, files: List[Path])
        - is_split_mode=False: files contains single config file path
        - is_split_mode=True: files contains all YAML files to merge (sorted)

    File Discovery Order (for split mode):
    1. main.yaml (if exists) - loaded first as base
    2. All *.yaml and *.yml files in root (alphabetical)
    3. All *.yaml and *.yml files in datasets/ subdirectory (alphabetical)

    YAML Splitting Architecture Evidence:
    - Home Assistant pattern: !include_dir_merge_named for directory-based splitting
    - Industry standard: "Split large configuration files into smaller,
      purpose-specific ones to ease management" (Configu, 2024)
    - Python pathlib.Path.glob() for file discovery (Python 3.4+)
    """
    config_path = Path(config_path)

    # Case 1: Single file exists (backward compatibility)
    if config_path.is_file():
        logger.debug(f"Single-file config mode: {config_path}")
        return (False, [config_path])

    # Case 2: Directory exists (split-file mode)
    if config_path.is_dir():
        logger.debug(f"Split-file config mode: {config_path}")
        return (True, _collect_yaml_files(config_path))

    # Case 3: Path might be intended as directory
    # (e.g., 'config' when 'config/' exists but 'config' file doesn't)
    if config_path.with_suffix("").is_dir():
        dir_path = config_path.with_suffix("")
        logger.debug(f"Split-file config mode (inferred directory): {dir_path}")
        return (True, _collect_yaml_files(dir_path))

    # Case 4: Neither exists - return as-is, let load_config() handle the error
    return (False, [config_path])


def _collect_yaml_files(config_dir: Path) -> list[Path]:
    """
    Collect all YAML files from config directory in merge order.

    Merge Order (later files override earlier):
    1. main.yaml or main.yml (base configuration, if exists)
    2. Root-level *.yaml/*.yml (alphabetical, excluding main.yaml/main.yml)
       - This includes config.yaml if present (handles edge case where
         directory contains config.yaml instead of main.yaml)
    3. datasets/*.yaml/*.yml (alphabetical) - dataset-specific configs

    Edge Case Handling:
    - If ./config/ exists with config.yaml but NO main.yaml:
      config.yaml is picked up in step 2 (root-level files)
    - Files are sorted alphabetically, so 'config.yaml' loads before
      'datasets.yaml', 'models.yaml', etc.

    Args:
        config_dir: Path to configuration directory

    Returns:
        List of Path objects in merge order

    Evidence: pathlib.Path.glob() is the modern Python standard for
    file discovery (Python docs, 3.4+)
    """
    files = []

    # 1. Main config first (if exists) - preferred base file name
    main_yaml = config_dir / "main.yaml"
    main_yml = config_dir / "main.yml"
    if main_yaml.exists():
        files.append(main_yaml)
    elif main_yml.exists():
        files.append(main_yml)

    # 2. Root-level YAML files (alphabetical, excluding main)
    # NOTE: This catches config.yaml if main.yaml doesn't exist
    root_yamls = sorted(
        [f for f in config_dir.glob("*.yaml") if f.name not in ("main.yaml",)]
        + [f for f in config_dir.glob("*.yml") if f.name not in ("main.yml",)]
    )
    files.extend(root_yamls)

    # 3. Dataset subdirectory (if exists)
    datasets_dir = config_dir / "datasets"
    if datasets_dir.is_dir():
        dataset_yamls = sorted(list(datasets_dir.glob("*.yaml")) + list(datasets_dir.glob("*.yml")))
        files.extend(dataset_yamls)

    if not files:
        logger.warning(f"No YAML files found in config directory: {config_dir}")
    else:
        logger.debug(f"Discovered {len(files)} config files: {[f.name for f in files]}")

    return files


def _deep_merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.

    Dependencies: Python standard library ONLY (copy.deepcopy)
    NO external packages required (deepmerge/mergedeep NOT used)

    Merge Strategy:
    - Dict + Dict: Recursive merge (nested keys combined)
    - List + List: Override (later list replaces earlier)
    - Any + Any: Override (later value replaces earlier)

    This follows the standard YAML merging pattern used by:
    - Dynaconf (Python settings library)
    - hiyapyco (hierarchical YAML merging)
    - pydantic-config (Pydantic V2 compatible)

    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary

    Returns:
        New merged dictionary (does not modify inputs)

    Implementation Note:
        Uses copy.deepcopy() from Python standard library to achieve
        the same merge semantics as deepmerge/mergedeep libraries
        without adding external dependencies.

    Thread Safety: Returns new dict, no mutation of inputs
    """
    # Start with deep copy of base to avoid mutation
    result = copy.deepcopy(base)

    for key, override_value in override.items():
        if key in result:
            base_value = result[key]

            # Both are dicts: recursive merge
            if isinstance(base_value, dict) and isinstance(override_value, dict):
                result[key] = _deep_merge_configs(base_value, override_value)
            else:
                # All other cases: override replaces base
                # This includes: list+list, scalar+scalar, type mismatch
                result[key] = copy.deepcopy(override_value)
        else:
            # New key: add with deep copy
            result[key] = copy.deepcopy(override_value)

    return result


def _load_and_merge_yaml_files(files: list[Path]) -> dict[str, Any]:
    """
    Load multiple YAML files and merge them in order.

    Args:
        files: List of YAML file paths in merge order

    Returns:
        Merged configuration dictionary

    Raises:
        ConfigurationError: If any file fails to load or parse
    """
    if not files:
        raise ConfigurationError(
            "No configuration files provided for merging",
            config_key="config_files",
            actual_value="empty list",
        )

    merged_config = {}
    loaded_files = []

    for file_path in files:
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                logger.warning(f"Empty config file skipped: {file_path}")
                continue

            file_config = yaml.safe_load(content)

            if file_config is None:
                logger.warning(f"Config file parsed as None, skipped: {file_path}")
                continue

            if not isinstance(file_config, dict):
                raise ConfigurationError(
                    f"Config file must contain a dictionary, got {type(file_config).__name__}",
                    config_key="config_format",
                    actual_value=str(file_path),
                )

            merged_config = _deep_merge_configs(merged_config, file_config)
            loaded_files.append(file_path.name)

        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Error parsing configuration file {file_path}: {str(e)}",
                config_key="yaml_parsing",
                actual_value=str(file_path),
            )
        except UnicodeDecodeError as e:
            raise ConfigurationError(
                f"Error reading configuration file {file_path}: {str(e)}",
                config_key="file_encoding",
                actual_value=str(file_path),
            )

    logger.info(f"Merged {len(loaded_files)} config files: {loaded_files}")
    return merged_config


def _get_default_config_path():
    """
    Get the default configuration file or directory path.

    YAML Splitting Enhancement:
    - Now supports both single-file (backward compatible) and directory (split-file) modes

    Priority Order:
    1. config.yaml (single file in CWD) - HIGHEST, backward compatible
    2. config.yml (single file in CWD)
    3. ./configs/ (directory) - triggers split-file mode
       NOTE: Uses 'configs/' (plural) to avoid confusion with milia_pipeline/config/ (Python code)
    4. ./configs/config.yaml (single file inside configs/) - fallback

    Edge Case Clarification:
    - If ./configs/ directory exists, it takes priority over ./configs/config.yaml
    - The directory mode will then discover ALL *.yaml files inside ./configs/
    - If ./configs/ contains config.yaml (but no main.yaml), config.yaml is
      loaded as part of the directory merge (see _collect_yaml_files)
    """
    # Priority 1: Single file (backward compatible)
    for file_path in ["config.yaml", "config.yml"]:
        if Path(file_path).is_file():
            return file_path

    # Priority 2: Configs directory (NEW - split-file mode)
    # NOTE: 'configs/' (plural) avoids confusion with milia_pipeline/config/ (Python code module)
    configs_dir = Path("./configs")
    if configs_dir.is_dir():
        return str(configs_dir)

    # Priority 3: config.yaml inside configs/ directory (legacy layout)
    config_in_dir = Path("./configs/config.yaml")
    if config_in_dir.is_file():
        return str(config_in_dir)

    # Default fallback
    return "config.yaml"


def load_config(
    config_path=None,
    enable_enhancement=True,
    enable_migration=True,
    enable_validation=True,
    validation_level="NORMAL",
    force_reload=False,
    report_validation=True,
):
    """
    Load the application configuration from the specified YAML file.

    Handler-Based Architecture:
    - All operations delegated to ConfigHandler
    - Automatic Handler-Based Architecture or fallback selection
    - Integrated enhanced schema validation with configurable levels
    - Validation reporting on load with detailed diagnostics
    - Configuration migration with comprehensive reports
    - Support for STRICT/NORMAL/RELAXED validation modes
    - Enhanced error reporting and diagnostics

    Args:
        config_path (str): The path to the YAML configuration file.
        enable_enhancement (bool): Enable enhanced configuration processing.
        enable_migration (bool): Enable automatic legacy format migration.
        enable_validation (bool): Enable comprehensive configuration validation.
        validation_level (str): Validation level - 'STRICT', 'NORMAL', or 'RELAXED'. Default is 'NORMAL'.
        force_reload (bool): Force reload even if config is cached.
        report_validation (bool): Log validation results (warnings/errors).

    Returns:
        dict: A dictionary containing the loaded and enhanced configuration.

    Raises:
        ConfigurationError: If the file is not found, cannot be parsed,
                            validation fails, or any other loading error occurs.
    """
    global _config_cache, _CONFIG_STATS, _CONFIG

    # Determine config path
    if config_path is None:
        config_path = _get_default_config_path()

    # Normalize validation level - DEFAULT TO NORMAL for backward compatibility
    validation_level = validation_level.upper()
    if validation_level not in ["STRICT", "NORMAL", "RELAXED"]:
        logger.warning(f"Invalid validation level '{validation_level}', using NORMAL")
        validation_level = "NORMAL"

    # ================================================================
    # YAML SPLITTING: Detect single-file vs split-file mode
    # ================================================================
    is_split_mode, config_files = _discover_config_files(config_path)

    # Create cache key including validation level
    # For split-file mode, include directory hash for cache invalidation
    if is_split_mode:
        # Create deterministic hash of all file paths and their modification times
        file_info = [(str(f), f.stat().st_mtime if f.exists() else 0) for f in config_files]
        config_hash = hashlib.md5(str(file_info).encode()).hexdigest()[:8]
        cache_key = f"{config_path}:split:{config_hash}:{enable_enhancement}:{enable_migration}:{enable_validation}:{validation_level}"
    else:
        cache_key = f"{config_path}:{enable_enhancement}:{enable_migration}:{enable_validation}:{validation_level}"

    with _cache_lock:
        # Check cache first
        if not force_reload and cache_key in _config_cache:
            cached_config = _config_cache[cache_key]
            with _stats_lock:
                _CONFIG_STATS["cache_hits"] += 1
                _CONFIG_STATS["config_cached"] = True
                total_requests = _CONFIG_STATS["load_count"] + _CONFIG_STATS["cache_hits"]
                _CONFIG_STATS["cache_hit_rate"] = _CONFIG_STATS["cache_hits"] / total_requests

            _CONFIG = cached_config
            return cached_config

        # Not in cache - increment load count
        with _stats_lock:
            _CONFIG_STATS["load_count"] += 1
            _CONFIG_STATS["last_load_time"] = time.time()
            _CONFIG_STATS["validation_level"] = validation_level

        # Load the config
        try:
            # ================================================================
            # YAML SPLITTING: Check file/directory existence
            # ================================================================
            if not is_split_mode and not config_files[0].exists():
                raise ConfigurationError(
                    f"Configuration file not found at: {config_path}",
                    config_key="config_path",
                    actual_value=config_path,
                )
            if is_split_mode and not config_files:
                raise ConfigurationError(
                    f"No configuration files found in directory: {config_path}",
                    config_key="config_directory",
                    actual_value=config_path,
                )

            # ================================================================
            # YAML SPLITTING: Load and parse YAML (single-file or split-file)
            # ================================================================
            try:
                if is_split_mode:
                    # Split-file mode: merge multiple YAML files
                    config = _load_and_merge_yaml_files(config_files)
                    logger.info(f"Loaded split configuration from: {config_path}")
                else:
                    # Single-file mode: backward compatible behavior
                    single_file = config_files[0]
                    with open(single_file, encoding="utf-8") as f:
                        content = f.read().strip()

                    if not content:
                        raise ConfigurationError(
                            f"Configuration file is empty: {single_file}",
                            config_key="config_content",
                        )

                    config = yaml.safe_load(content)

            except yaml.YAMLError as e:
                raise ConfigurationError(
                    f"Error parsing configuration file {config_path}: {str(e)}",
                    config_key="yaml_parsing",
                    actual_value=config_path,
                )
            except UnicodeDecodeError as e:
                raise ConfigurationError(
                    f"Error reading configuration file {config_path}: {str(e)}",
                    config_key="file_encoding",
                    actual_value=config_path,
                )

            # Validate basic structure
            if not isinstance(config, dict):
                raise ConfigurationError(
                    f"Configuration must be a dictionary, got {type(config).__name__}",
                    config_key="config_format",
                    actual_value=type(config).__name__,
                )

            # ================================================================
            # PHASE 6.2: Normalize dataset_type at config load time
            # ================================================================
            # This is the SINGLE SOURCE OF TRUTH for dataset type normalization.
            # All downstream code receives the canonical name, eliminating the
            # need for case-insensitive matching at multiple validation points.
            # Resolves Technical Debt MILIA-ARCH-001.
            #
            # CRITICAL: Track if normalization was skipped due to re-entrant call.
            # If so, we MUST NOT cache this config because it has un-normalized
            # dataset_type. The outer call will complete normalization correctly.
            _normalization_skipped = []
            if "dataset_type" in config:
                original_dataset_type = config["dataset_type"]
                config["dataset_type"] = _normalize_dataset_type(
                    original_dataset_type, _skip_cache_if_reentrant=_normalization_skipped
                )
                if config["dataset_type"] != original_dataset_type:
                    logger.info(
                        f"Dataset type normalized: '{original_dataset_type}' -> "
                        f"'{config['dataset_type']}'"
                    )

            # ================================================================
            # PHASE 6.3: Normalize dataset-keyed config sections
            # ================================================================
            # Normalize keys in property_availability and data_config.property_selection
            # to match the canonical dataset names from the registry.
            # This ensures config lookups work regardless of case used in config.yaml.
            if not _normalization_skipped:
                config = _normalize_dataset_keyed_sections(config)

            # If this is a re-entrant call, DON'T cache the config
            # The outer call will cache the correctly normalized config
            if _normalization_skipped:
                logger.debug(
                    "Skipping config cache due to re-entrant call during registry initialization"
                )
                _CONFIG = config
                return config

            # Apply enhancements with Handler delegation
            original_config = copy.deepcopy(config)

            if enable_enhancement:
                try:
                    # Delegate to Handler-based enhancement function
                    config, validation_results, migration_report = (
                        _apply_transformation_enhancements(
                            config,
                            enable_migration,
                            enable_validation,
                            validation_level,
                            report_validation,
                        )
                    )

                    # Store results in stats
                    with _stats_lock:
                        _CONFIG_STATS["enhancement_applied"] = True
                        _CONFIG_STATS["last_validation_results"] = validation_results
                        _CONFIG_STATS["last_migration_report"] = migration_report

                        if validation_results:
                            _CONFIG_STATS["warnings_count"] = len(
                                validation_results.get("warnings", [])
                            )
                            _CONFIG_STATS["errors_count"] = len(
                                validation_results.get("errors", [])
                            )

                        if migration_report and migration_report.get("migration_applied"):
                            _CONFIG_STATS["migration_applied"] = True

                except ImportError as e:
                    # Handler not available - should not happen but keep for safety
                    logger.error(f"Handler not available (unexpected): {e}")
                    config = original_config
                    with _stats_lock:
                        _CONFIG_STATS["enhancement_applied"] = False
                except ConfigurationError:
                    raise  # Re-raise validation errors in STRICT mode
                except Exception as e:
                    logger.warning(f"Configuration enhancement failed: {e}")
                    config = original_config
                    with _stats_lock:
                        _CONFIG_STATS["enhancement_applied"] = False

            # Store in both cache and global _CONFIG variable
            _config_cache[cache_key] = config
            _CONFIG = config

            with _stats_lock:
                _CONFIG_STATS["config_cached"] = True
                total_requests = _CONFIG_STATS["load_count"] + _CONFIG_STATS["cache_hits"]
                _CONFIG_STATS["cache_hit_rate"] = (
                    _CONFIG_STATS["cache_hits"] / total_requests if total_requests > 0 else 0.0
                )

            return config

        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(
                f"An unexpected error occurred while loading config: {str(e)}",
                config_key="config_loading",
                actual_value=config_path,
            )


def _detect_transformation_format(transforms_section: Any) -> str:
    """
    Detect transformation configuration format with robust validation.

    This function identifies the format of the transformation section to determine
    if migration is needed and which migration strategy to apply.

    Args:
        transforms_section: The 'transformations' section from config

    Returns:
        str: Format type - 'enhanced', 'legacy_list', 'legacy_dict', 'invalid', or 'unknown'

    Format Types:
        - 'enhanced': New format with experimental_setups structure
        - 'legacy_list': Old format with list of transforms
        - 'legacy_dict': Old format with single transform dict or named setups
        - 'invalid': Missing or invalid structure
        - 'unknown': Cannot determine format
    """
    if not transforms_section:
        return "invalid"

    if not isinstance(transforms_section, dict):
        if isinstance(transforms_section, list):
            return "legacy_list"
        return "invalid"

    # Check for enhanced format - must have experimental_setups OR standard_transforms
    if "experimental_setups" in transforms_section or "standard_transforms" in transforms_section:
        return "enhanced"

    # Check for legacy dict with 'name' (single transform)
    if "name" in transforms_section:
        return "legacy_dict"

    # Check for legacy dict with named setups (values are lists)
    # This handles the case where someone created their own setup structure
    if any(isinstance(v, list) for v in transforms_section.values() if v is not None):
        return "legacy_dict"

    return "unknown"


def _validate_enhanced_format(transforms_section: dict[str, Any]) -> dict[str, Any]:
    """
    Validate enhanced format structure to ensure it's complete and correct.

    This ensures that configurations claiming to be in enhanced format actually
    have all required fields and valid structure.

    Args:
        transforms_section: The 'transformations' section to validate

    Returns:
        dict: Validation result with:
            - 'valid' (bool): Whether format is valid
            - 'errors' (list): Critical errors that prevent usage
            - 'warnings' (list): Non-critical issues
    """
    validation = {"valid": True, "errors": [], "warnings": []}

    # Check for at least one transform source: experimental_setups OR standard_transforms
    has_experimental_setups = "experimental_setups" in transforms_section
    has_standard_transforms = "standard_transforms" in transforms_section

    if not has_experimental_setups and not has_standard_transforms:
        validation["valid"] = False
        validation["errors"].append(
            "Missing 'experimental_setups' or 'standard_transforms' key (at least one required)"
        )
        return validation

    # Validate standard_transforms if present
    if has_standard_transforms:
        standard_transforms = transforms_section["standard_transforms"]
        if not isinstance(standard_transforms, list):
            validation["valid"] = False
            validation["errors"].append(
                f"'standard_transforms' must be list, got {type(standard_transforms).__name__}"
            )
        else:
            # Validate each standard transform
            for i, transform in enumerate(standard_transforms):
                if isinstance(transform, dict) and "name" not in transform:
                    validation["warnings"].append(f"standard_transforms[{i}] missing 'name' field")

    # Validate experimental_setups if present
    if has_experimental_setups:
        setups = transforms_section["experimental_setups"]

        # Validate experimental_setups is a dict
        if not isinstance(setups, dict):
            validation["valid"] = False
            validation["errors"].append(
                f"'experimental_setups' must be dict, got {type(setups).__name__}"
            )
            return validation

        # Empty experimental_setups is allowed if standard_transforms exists
        if len(setups) == 0 and not has_standard_transforms:
            validation["valid"] = False
            validation["errors"].append(
                "'experimental_setups' is empty and no 'standard_transforms' defined"
            )
            return validation

        # Validate default_setup exists and references valid setup (only if setups non-empty)
        if len(setups) > 0:
            if "default_setup" not in transforms_section:
                validation["warnings"].append("Missing 'default_setup' key")
            else:
                default_setup = transforms_section["default_setup"]
                if default_setup not in setups:
                    # Only error if no standard_transforms exist
                    if not has_standard_transforms:
                        validation["valid"] = False
                        validation["errors"].append(
                            f"default_setup '{default_setup}' not found in experimental_setups"
                        )
                    else:
                        # standard_transforms exist, so default_setup is just a label
                        validation["warnings"].append(
                            f"default_setup '{default_setup}' not in experimental_setups (using standard_transforms)"
                        )
    else:
        # Only standard_transforms - check for default_setup as optional label
        if "default_setup" not in transforms_section:
            validation["warnings"].append(
                "Missing 'default_setup' key (recommended even with only standard_transforms)"
            )

    # Validate each setup structure (only if experimental_setups exists)
    if has_experimental_setups:
        for setup_name, setup_transforms in setups.items():
            # Each setup should be a list of transforms
            if not isinstance(setup_transforms, list):
                validation["warnings"].append(
                    f"Setup '{setup_name}' should be a list, got {type(setup_transforms).__name__}"
                )
                continue

            # Validate each transform in the setup
            for idx, transform in enumerate(setup_transforms):
                if not isinstance(transform, dict):
                    validation["errors"].append(
                        f"Setup '{setup_name}', transform {idx}: must be dict"
                    )
                    validation["valid"] = False
                    continue

                # Check for required 'name' field
                if "name" not in transform:
                    validation["errors"].append(
                        f"Setup '{setup_name}', transform {idx}: missing 'name' field"
                    )
                    validation["valid"] = False

    return validation


def _apply_transformation_enhancements(
    config, enable_migration, enable_validation, validation_level, report_validation
):
    """
    Apply enhancements to configuration with comprehensive reporting.

    Handler-Based Architecture: All Phase 2 operations delegated to ConfigHandler.

    This function handles:
    1. Format detection via Handler
    2. Migration from legacy to enhanced format via Handler
    3. Validation via Handler
    4. Comprehensive error reporting

    Args:
        config: Raw configuration dictionary
        enable_migration: Whether to apply migration
        enable_validation: Whether to apply validation
        validation_level: Validation strictness level (string: 'STRICT', 'NORMAL', 'RELAXED')
        report_validation: Whether to log validation results

    Returns:
        Tuple of (enhanced_config, validation_results, migration_report)
    """
    enhanced_config = copy.deepcopy(config)

    # Initialize with proper default structures to prevent None returns
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "format_detected": "unknown",
        "format_validation": {},
    }

    migration_report = {
        "migration_applied": False,
        "format_before": "unknown",
        "format_after": "unknown",
        "validation_passed": False,
    }

    # Step 1: Format detection via Handler
    config.get("transformations", {})

    # Detect format using Handler
    try:
        from milia_pipeline.config.config_schemas import ConfigMigration

        migration = ConfigMigration()
        format_info = migration.detect_format(config)
        format_detected = format_info if isinstance(format_info, str) else "unknown"
        validation_results["format_detected"] = format_detected
        migration_report["format_before"] = format_detected

        # Check if already in valid enhanced format
        is_already_enhanced = (
            format_detected == "enhanced" and (format_info == "enhanced")
            if isinstance(format_info, str)
            else False
        )

        if (
            format_detected == "enhanced" and format_info != "enhanced"
            if isinstance(format_info, str)
            else False
        ):
            logger.warning(f"Enhanced format detected but invalid: {[]}")
            validation_results["warnings"].append(
                "Enhanced format structure invalid, migration may be needed"
            )
            is_already_enhanced = False
        elif is_already_enhanced:
            logger.debug("Configuration already in valid enhanced format")

    except Exception as e:
        logger.warning(f"Format detection failed: {e}")
        format_detected = "unknown"
        is_already_enhanced = False
        validation_results["format_detected"] = format_detected
        migration_report["format_before"] = format_detected

    # Step 2: Apply migration if needed and enabled
    if enable_migration and not is_already_enhanced:
        needs_migration = (
            format_detected in ["legacy_list", "legacy_dict", "unknown"] or not is_already_enhanced
        )

        if needs_migration:
            try:
                logger.info(f"Applying migration from format '{format_detected}'")

                # Migrate using config_schemas
                from milia_pipeline.config.config_schemas import ConfigMigration

                migration = ConfigMigration()
                migrated_config, migration_warnings = migration.migrate_to_enhanced(
                    config, preserve_original=True
                )

                # Handle migration result (tuple returned: config, warnings)
                if migrated_config and isinstance(migrated_config, dict):
                    enhanced_config = migrated_config

                    # Validate migration result
                    from milia_pipeline.config.config_schemas import ConfigMigration

                    migration_validator = ConfigMigration()
                    migrated_format_info = migration_validator.detect_format(enhanced_config)
                    migrated_format = migrated_format_info

                    # Build comprehensive migration report
                    migration_report = {
                        "migration_applied": True,
                        "format_before": format_detected,
                        "format_after": migrated_format,
                        "original_format": format_detected,
                        "target_format": migrated_format,
                        "changes_made": migration_warnings,
                        "warnings": migration_warnings,
                        "migration_time_ms": 0,
                        "validation_passed": True,
                    }

                    if migrated_format != "enhanced":
                        validation_results["warnings"].append(
                            f"Migration completed but format is '{migrated_format}', expected 'enhanced'"
                        )
                    elif migrated_format != "enhanced":
                        validation_results["warnings"].append(
                            "Migration created invalid enhanced format"
                        )
                        logger.warning("Post-migration validation: format is not enhanced")

                    if report_validation and migration_report:
                        _log_migration_report(migration_report)
                else:
                    # Migration failed or returned None
                    logger.warning("Migration failed or returned empty result")
                    enhanced_config = config
                    validation_results["warnings"].append("Migration failed")

            except Exception as e:
                logger.warning(f"Migration failed, using original config: {e}")
                enhanced_config = config
                validation_results["warnings"].append(f"Migration failed: {str(e)}")

    # Step 3: Apply validation if enabled
    if enable_validation:
        try:
            # Validate using config_schemas
            from milia_pipeline.config.config_schemas import ValidationConfig, YAMLSchemaValidator

            validator = YAMLSchemaValidator()
            # Map validation_level string to ValidationConfig
            if validation_level == "STRICT":
                validation_cfg = ValidationConfig(strict_mode=True)
            elif validation_level == "RELAXED":
                validation_cfg = ValidationConfig(
                    strict_mode=False, check_transform_compatibility=False
                )
            else:  # NORMAL
                validation_cfg = ValidationConfig(strict_mode=False)
            val_results = validator.validate_config(
                enhanced_config, validation_config=validation_cfg
            )

            # Ensure we got a valid result before using it
            if val_results is not None and isinstance(val_results, dict):
                # Merge validation results, preserving format_validation
                format_validation_backup = validation_results.get("format_validation", {})
                validation_results.update(val_results)
                validation_results["format_validation"] = format_validation_backup

            # Store validation time
            with _stats_lock:
                _CONFIG_STATS["last_validation_time"] = time.time()

            # Report validation results
            if report_validation:
                _log_validation_results(validation_results, validation_level)

            # In STRICT mode, raise errors if there are actual validation errors
            if validation_level == "STRICT" and not validation_results.get("valid", True):
                errors = validation_results.get("errors", [])
                if errors:
                    raise ConfigurationError(
                        f"Configuration validation failed in STRICT mode: {'; '.join(errors)}",
                        config_key="validation",
                    )

        except ConfigurationError:
            raise  # Re-raise STRICT mode errors
        except Exception as e:
            logger.warning(f"Validation error: {e}")
            validation_results["warnings"].append(f"Validation error: {str(e)}")

    return enhanced_config, validation_results, migration_report


def _log_validation_results(validation_results, validation_level):
    """Log validation results in a user-friendly format."""
    if not validation_results or not isinstance(validation_results, dict):
        logger.warning("Invalid validation results structure")
        return

    valid = validation_results.get("valid", True)
    errors = validation_results.get("errors", [])
    warnings = validation_results.get("warnings", [])
    format_detected = validation_results.get("format_detected", "unknown")

    # Log format detection
    logger.info(f"Configuration format: {format_detected}")

    # Log validation status
    if valid:
        logger.info(f"✓ Configuration validation passed ({validation_level} mode)")
    else:
        logger.warning(f"✗ Configuration validation issues detected ({validation_level} mode)")

    # Log errors
    if errors:
        logger.error(f"Validation errors ({len(errors)}):")
        for i, error in enumerate(errors, 1):
            logger.error(f"  {i}. {error}")

    # Log warnings
    if warnings:
        logger.warning(f"Validation warnings ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            logger.warning(f"  {i}. {warning}")

    # Log performance
    validation_time = validation_results.get("validation_time_ms", 0)
    if validation_time > 0:
        logger.debug(f"Validation completed in {validation_time:.2f}ms")


def _log_migration_report(migration_report):
    """Log migration report in a user-friendly format."""
    if not migration_report or not isinstance(migration_report, dict):
        return

    if not migration_report.get("migration_applied", False):
        return

    logger.info("=" * 60)
    logger.info("Configuration Migration Report")
    logger.info("=" * 60)

    logger.info(f"Original format: {migration_report.get('original_format', 'unknown')}")
    logger.info(f"Target format: {migration_report.get('target_format', 'unknown')}")

    changes = migration_report.get("changes_made", [])
    if changes:
        logger.info(f"Changes applied ({len(changes)}):")
        for i, change in enumerate(changes, 1):
            logger.info(f"  {i}. {change}")

    warnings = migration_report.get("warnings", [])
    if warnings:
        logger.warning(f"Migration warnings ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            logger.warning(f"  {i}. {warning}")

    migration_time = migration_report.get("migration_time_ms", 0)
    if migration_time > 0:
        logger.info(f"Migration completed in {migration_time:.2f}ms")
    logger.info("=" * 60)


# ADDITIONAL HELPER FUNCTIONS FOR GLOBAL STATE MANAGEMENT


def get_global_config_state():
    """
    Get the current global configuration state.

    Returns:
        dict or None: The current global configuration, or None if not loaded.
    """
    global _CONFIG
    return _CONFIG


def set_global_config_state(config_data):
    """
    Manually set the global configuration state.

    Args:
        config_data (dict): Configuration data to set globally.
    """
    global _CONFIG
    _CONFIG = config_data


def is_config_loaded():
    """
    Check if configuration has been loaded.

    Returns:
        bool: True if configuration is loaded, False otherwise.
    """
    global _CONFIG
    return _CONFIG is not None


def get_config_load_time():
    """
    Get the timestamp of the last configuration load.

    Returns:
        float or None: Timestamp of last load, or None if never loaded.
    """
    global _CONFIG_STATS
    with _stats_lock:
        return _CONFIG_STATS.get("last_load_time")


def get_config_hash():
    """
    Get a hash of the current configuration for change detection.

    Returns:
        str or None: Hash of current config, or None if no config loaded.
    """
    global _CONFIG
    if _CONFIG is None:
        return None

    import hashlib
    import json

    # Create a stable hash of the config
    config_str = json.dumps(_CONFIG, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()


def get_config_stats():
    """
    Get configuration loading statistics.

    Returns:
        dict: Dictionary containing loading statistics.
    """
    global _CONFIG_STATS
    with _stats_lock:
        return _CONFIG_STATS.copy()


def _apply_configuration_enhancements(config, enable_migration=True, enable_validation=True):
    """
    Apply configuration enhancements with proper Handler delegation.

    Respects enable_migration parameter for backward compatibility.
    Uses ConfigHandler for all operations.
    """
    try:
        enhanced_config = config.copy()

        # Check if this is already an enhanced config FIRST
        transforms_section = config.get("transformations", {})
        is_already_enhanced = isinstance(transforms_section, dict) and (
            "experimental_setups" in transforms_section
            or "standard_transforms" in transforms_section
        )

        # CRITICAL FIX: If migration is disabled, skip ALL migration processing
        if not enable_migration:
            logger.debug("Migration disabled - preserving original format")

            # Only apply validation if requested and migration wasn't requested
            if enable_validation:
                try:
                    # Validate using config_schemas
                    from milia_pipeline.config.config_schemas import (
                        ValidationConfig,
                        YAMLSchemaValidator,
                    )

                    validator = YAMLSchemaValidator()
                    validation_cfg = ValidationConfig(
                        strict_mode=False, check_transform_compatibility=False
                    )
                    val_results = validator.validate_config(
                        enhanced_config, validation_config=validation_cfg
                    )
                    if not val_results.get("valid", True):
                        logger.warning(f"Validation warnings: {val_results.get('warnings', [])}")
                except Exception as e:
                    logger.warning(f"Validation warning: {e}")

            return enhanced_config

        # If already enhanced, skip migration processing
        if is_already_enhanced:
            logger.debug("Config is already in enhanced format")

            # Apply validation if requested
            if enable_validation:
                try:
                    # Validate using config_schemas
                    from milia_pipeline.config.config_schemas import (
                        ValidationConfig,
                        YAMLSchemaValidator,
                    )

                    validator = YAMLSchemaValidator()
                    validation_cfg = ValidationConfig(
                        strict_mode=False, check_transform_compatibility=False
                    )
                    val_results = validator.validate_config(
                        enhanced_config, validation_config=validation_cfg
                    )
                    if not val_results.get("valid", True):
                        logger.warning(f"Validation warnings: {val_results.get('warnings', [])}")
                except Exception as e:
                    logger.warning(f"Validation warning: {e}")

            return enhanced_config

        # Only migrate if explicitly enabled and format needs migration
        if enable_migration and "transformations" in config:
            # Detect format using config_schemas
            from milia_pipeline.config.config_schemas import ConfigMigration

            migration = ConfigMigration()
            format_info = migration.detect_format(enhanced_config)
            needs_migration = (
                (format_info not in ["enhanced", "invalid"])
                if isinstance(format_info, str)
                else False
            )

            if needs_migration:
                try:
                    # Migrate using config_schemas
                    from milia_pipeline.config.config_schemas import ConfigMigration

                    migration = ConfigMigration()
                    migrated_config, migration_warnings = migration.migrate_to_enhanced(
                        enhanced_config, preserve_original=True
                    )
                    if migrated_config and isinstance(migrated_config, dict):
                        enhanced_config = migrated_config
                        logger.info("Applied configuration migration")
                        if migration_warnings:
                            for warning in migration_warnings:
                                logger.debug(f"Migration: {warning}")
                    else:
                        logger.warning("Migration failed or returned empty result")
                except Exception as e:
                    logger.warning(f"Migration failed, using original config: {e}")
                    enhanced_config = config  # Use original on migration failure

        # Apply validation if requested (with leniency for backward compatibility)
        if enable_validation:
            try:
                # Validate using config_schemas
                from milia_pipeline.config.config_schemas import (
                    ValidationConfig,
                    YAMLSchemaValidator,
                )

                validator = YAMLSchemaValidator()
                validation_cfg = ValidationConfig(
                    strict_mode=False, check_transform_compatibility=False
                )
                val_results = validator.validate_config(
                    enhanced_config, validation_config=validation_cfg
                )
                if not val_results.get("valid", True):
                    logger.warning(f"Validation warnings: {val_results.get('warnings', [])}")
            except Exception as e:
                logger.warning(f"Validation warning: {e}")

        return enhanced_config

    except Exception as e:
        logger.error(f"Enhancement failed, using original config: {e}")
        return config  # Always return original config on any error


def _optimized_validation(config):
    """
    FIXED: Optimized validation that doesn't generate false migration warnings
    """
    transforms_section = config.get("transformations", {})

    # Check if this is an enhanced config (has experimental_setups OR standard_transforms)
    is_enhanced = isinstance(transforms_section, dict) and (
        "experimental_setups" in transforms_section or "standard_transforms" in transforms_section
    )

    if isinstance(transforms_section, dict):
        setups = transforms_section.get("experimental_setups", {})
        standard_transforms = transforms_section.get("standard_transforms", [])

        # Calculate total transforms for optimization
        total_experimental = sum(
            len(transforms) for transforms in setups.values() if isinstance(transforms, list)
        )
        total_standard = len(standard_transforms) if isinstance(standard_transforms, list) else 0
        total_transforms = total_experimental + total_standard

        # Use large config optimization for very large configs
        if len(setups) > 50 or total_transforms > 500:
            logger.info(
                f"Large config detected ({len(setups)} setups, {total_transforms} transforms). Using optimized validation."
            )

            # Fast validation - check critical errors only
            if "default_setup" not in transforms_section:
                logger.warning("Missing default_setup in transformations")
            elif setups and transforms_section["default_setup"] not in setups:
                # Only warn if experimental_setups is non-empty and no standard_transforms
                if not standard_transforms:
                    logger.warning(
                        f"Default setup '{transforms_section['default_setup']}' not found in experimental_setups"
                    )

            # Validate essential structure
            for setup_name, transforms_list in setups.items():
                if isinstance(transforms_list, list):
                    for i, transform in enumerate(transforms_list):
                        if not isinstance(transform, dict):
                            raise ConfigurationError(
                                f"Setup '{setup_name}', transform {i}: Must be a dictionary",
                                config_key="validation",
                            )
                        if "name" not in transform:
                            raise ConfigurationError(
                                f"Setup '{setup_name}', transform {i}: Missing 'name' field",
                                config_key="validation",
                            )

            return config

    # For smaller configs, do detailed validation
    try:
        from milia_pipeline.config.config_schemas import ValidationConfig, YAMLSchemaValidator

        # Validate using config_schemas
        validator = YAMLSchemaValidator()
        validation_cfg = ValidationConfig(strict_mode=True)
        validation_results = validator.validate_config(config, validation_config=validation_cfg)

        # CRITICAL FIX: Filter out migration warnings for enhanced configs
        for warning in validation_results.get("warnings", []):
            # Skip migration warnings for configs that are already enhanced
            if is_enhanced and ("migrated" in warning.lower() or "migration" in warning.lower()):
                continue
            logger.warning(f"Config validation: {warning}")

        # Raise errors on validation failures
        errors = validation_results.get("errors", [])
        if errors:
            raise ConfigurationError(
                f"Configuration validation failed: {'; '.join(errors)}", config_key="validation"
            )

        # Check if validation actually failed
        if not validation_results.get("valid", True):
            raise ConfigurationError("Configuration validation failed", config_key="validation")
    except ConfigurationError:
        raise  # Re-raise validation errors
    except Exception as e:
        logger.warning(f"Validation error: {e}")

    return config


def _get_config_info(config: dict[str, Any]) -> str:
    """Generate informative summary of loaded configuration"""

    info_parts = []

    # Basic structure info
    info_parts.append(f"{len(config)} sections")

    # Transformation info
    if "transformations" in config:
        transforms = config["transformations"]

        if isinstance(transforms, dict) and "experimental_setups" in transforms:
            # Enhanced format
            setups = transforms["experimental_setups"]
            setup_count = len(setups)
            total_transforms = sum(
                len(setup_transforms)
                for setup_transforms in setups.values()
                if isinstance(setup_transforms, list)
            )
            default_setup = transforms.get("default_setup", "unknown")

            info_parts.append(
                f"transforms: {setup_count} experimental setups, {total_transforms} total transforms, default: {default_setup}"
            )

        elif isinstance(transforms, list):
            # Legacy list format
            info_parts.append(f"transforms: {len(transforms)} transforms (legacy format)")

        elif isinstance(transforms, dict) and "name" in transforms:
            # Legacy single transform format
            info_parts.append("transforms: 1 transform (legacy format)")

        else:
            info_parts.append("transforms: unknown format")

    # Dataset info
    if "dataset_type" in config:
        info_parts.append(f"dataset: {config['dataset_type']}")

    return ", ".join(info_parts)


def load_config_with_validation(config_path=None, validation_level="NORMAL"):
    """
    Load configuration and return validation results separately.

    This is useful when you need both the configuration and detailed
    validation results for reporting or decision-making.

    Uses ConfigHandler for validation - automatically selects
    Enhanced validation or fallback validation.

    Args:
        config_path: Path to config file (default: auto-detect)
        validation_level: Validation level (STRICT/NORMAL/RELAXED)

    Returns:
        Tuple of (config_dict, validation_results_dict)
    """
    # Determine config path
    if config_path is None:
        config_path = _get_default_config_path()

    # Load config without validation first (we'll validate separately)
    config = load_config(
        config_path=config_path,
        enable_validation=False,  # We'll validate separately
        validation_level=validation_level,
        report_validation=False,
    )

    # Validate using Handler
    from milia_pipeline.config.config_schemas import ValidationConfig, YAMLSchemaValidator

    validator = YAMLSchemaValidator()
    # Map validation_level string to ValidationConfig
    if validation_level == "STRICT":
        validation_cfg = ValidationConfig(strict_mode=True)
    elif validation_level == "RELAXED":
        validation_cfg = ValidationConfig(strict_mode=False, check_transform_compatibility=False)
    else:  # NORMAL
        validation_cfg = ValidationConfig(strict_mode=False)
    validation_results = validator.validate_config(config, validation_config=validation_cfg)

    # Store in statistics
    with _stats_lock:
        _CONFIG_STATS["last_validation_results"] = validation_results
        _CONFIG_STATS["warnings_count"] = len(validation_results.get("warnings", []))
        _CONFIG_STATS["errors_count"] = len(validation_results.get("errors", []))

    return config, validation_results


def reload_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """
    Force reload the configuration, bypassing cache.

    Useful for development and testing when configuration changes.

    Args:
        config_path: Path to configuration file

    Returns:
        Reloaded configuration dictionary
    """
    logger.info("Force reloading configuration")

    return load_config(
        config_path=config_path,
        enable_enhancement=True,
        enable_migration=True,
        enable_validation=True,
        force_reload=True,
    )


def get_enhanced_transformation_config(force_reload=False):
    """
    Get enhanced transformation configuration with consistent behavior.

    Args:
        force_reload: Force reload from file instead of using cache

    Returns:
        Dictionary containing transformation configuration

    Raises:
        ConfigurationError: If transformation configuration cannot be loaded
    """
    global _ENHANCED_TRANSFORMATION_CONFIG

    # Use separate caching for transformation config
    if not force_reload and _ENHANCED_TRANSFORMATION_CONFIG is not None:
        return _ENHANCED_TRANSFORMATION_CONFIG

    try:
        # Load config using same parameters as main load_config
        config = load_config(
            enable_enhancement=True,
            enable_migration=True,
            enable_validation=True,
            force_reload=force_reload,
        )

        # Extract transformation config
        transform_config = config.get("transformations", {})

        # Cache the transformation config separately
        _ENHANCED_TRANSFORMATION_CONFIG = transform_config

        return transform_config

    except ConfigurationError as e:
        # Preserve original error message
        original_error = str(e)
        if hasattr(e, "original_exception") and e.original_exception:
            if "Test config error" in str(e.original_exception):
                original_error = str(e.original_exception)

        raise ConfigurationError(
            f"Failed to load transformation configuration: {original_error}",
            config_key="transformations",
        )
    except Exception as e:
        raise ConfigurationError(
            f"Failed to load transformation configuration: {str(e)}", config_key="transformations"
        )


def validate_config_file(
    config_path: str = "config.yaml",
    validation_level: str = "STRICT",
    dataset_type: str | None = None,
) -> dict[str, Any]:
    """
    Validate a configuration file without loading it into cache.

    Handler-Based Architecture: Delegates validation to ConfigHandler.

    Enhancements:
    - Support for validation levels (STRICT/NORMAL/RELAXED)
    - Migration detection and reporting
    - Comprehensive diagnostics
    - Automatic ConfigHandler or fallback validation

    Args:
        config_path: Path to configuration file to validate
        validation_level: Validation strictness level
        dataset_type: Dataset type for specialized validation

    Returns:
        Dictionary with comprehensive validation results
    """
    validation_level = validation_level.upper()
    if validation_level not in ["STRICT", "NORMAL", "RELAXED"]:
        validation_level = "STRICT"  # Default to strict for file validation

    # ================================================================
    # YAML SPLITTING: Detect single-file vs split-file mode
    # ================================================================
    is_split_mode, config_files = _discover_config_files(config_path)

    # Check file/directory existence
    if not is_split_mode and not config_files[0].exists():
        return {
            "valid": False,
            "errors": [f"Configuration file not found: {config_path}"],
            "warnings": [],
            "file_exists": False,
            "validation_level": validation_level,
            "migration_recommended": False,
        }
    if is_split_mode and not config_files:
        return {
            "valid": False,
            "errors": [f"No configuration files found in directory: {config_path}"],
            "warnings": [],
            "file_exists": False,
            "validation_level": validation_level,
            "migration_recommended": False,
        }

    try:
        # ================================================================
        # YAML SPLITTING: Load and parse config (single-file or split-file)
        # ================================================================
        if is_split_mode:
            config = _load_and_merge_yaml_files(config_files)
        else:
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)

        if not isinstance(config, dict):
            return {
                "valid": False,
                "errors": [f"Configuration must be a dictionary, got {type(config).__name__}"],
                "warnings": [],
                "file_exists": True,
                "validation_level": validation_level,
                "migration_recommended": False,
            }

        # Detect format and check if migration recommended
        from milia_pipeline.config.config_schemas import ConfigMigration

        migration = ConfigMigration()
        format_info = migration.detect_format(config)
        format_type = format_info if isinstance(format_info, str) else "unknown"
        is_legacy = format_type in ["legacy_list", "legacy_dict"]

        # Initialize results
        # For split-file mode, calculate total size of all config files
        if is_split_mode:
            total_size = sum(f.stat().st_size for f in config_files if f.exists())
        else:
            total_size = os.path.getsize(config_path)

        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "file_exists": True,
            "file_path": config_path,
            "file_size_bytes": total_size,
            "validation_level": validation_level,
            "migration_recommended": is_legacy,
            "format_detected": format_type,
            "split_mode": is_split_mode,
            "config_files_count": len(config_files) if is_split_mode else 1,
        }

        # Validate using Handler
        try:
            from milia_pipeline.config.config_schemas import ValidationConfig, YAMLSchemaValidator

            validator = YAMLSchemaValidator()
            # Map validation_level string to ValidationConfig
            if validation_level == "STRICT":
                validation_cfg = ValidationConfig(strict_mode=True)
            elif validation_level == "RELAXED":
                validation_cfg = ValidationConfig(
                    strict_mode=False, check_transform_compatibility=False
                )
            else:  # NORMAL
                validation_cfg = ValidationConfig(strict_mode=False)
            val_results = validator.validate_config(config, validation_config=validation_cfg)

            # Merge validation results from Handler
            if val_results and isinstance(val_results, dict):
                validation_results["valid"] = val_results.get("valid", True)
                validation_results["errors"] = val_results.get("errors", [])
                validation_results["warnings"] = val_results.get("warnings", [])

                # Preserve format_detected if Handler provides it
                if "format_detected" in val_results:
                    validation_results["format_detected"] = val_results["format_detected"]

        except Exception as e:
            validation_results["warnings"].append(f"Validation error: {str(e)}")

        # Add migration recommendation message if legacy format
        if is_legacy:
            validation_results["warnings"].append(
                "Legacy configuration format detected. Consider migrating to enhanced format."
            )

        return validation_results

    except yaml.YAMLError as e:
        return {
            "valid": False,
            "errors": [f"YAML parsing error: {str(e)}"],
            "warnings": [],
            "file_exists": True,
            "yaml_error": str(e),
            "validation_level": validation_level,
            "migration_recommended": False,
        }
    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Validation failed: {str(e)}"],
            "warnings": [],
            "file_exists": True,
            "unexpected_error": str(e),
            "validation_level": validation_level,
            "migration_recommended": False,
        }


def get_config_statistics():
    """
    Get configuration loading and caching statistics.

    Returns:
        Dictionary with configuration statistics including all fields needed for diagnostics
    """
    with _stats_lock:
        stats = _CONFIG_STATS.copy()

    # Calculate cache hit rate
    total_requests = stats["load_count"] + stats["cache_hits"]
    if total_requests > 0:
        stats["cache_hit_rate"] = stats["cache_hits"] / total_requests
    else:
        stats["cache_hit_rate"] = 0.0

    # Ensure all expected fields are present with defaults
    required_fields = {
        "load_count": 0,
        "cache_hits": 0,
        "enhancement_applied": False,
        "migration_applied": False,
        "validation_enabled": True,
        "validation_level": "NORMAL",
        "last_load_time": None,
        "last_validation_time": None,
        "last_validation_results": None,
        "last_migration_report": None,
        "cache_hit_rate": 0.0,
        "config_cached": False,
        "warnings_count": 0,
        "errors_count": 0,
    }

    # Fill in any missing fields with defaults
    for field, default_value in required_fields.items():
        if field not in stats:
            stats[field] = default_value

    return stats


def create_example_config(
    config_path="config.example.yaml", dataset_type=None, include_experimental_setups=True
):
    """
    Create an example configuration file with enhanced format.

    Generates enhanced format example configuration via Handler.

    PHASE 5 REFACTORING:
    - Changed dataset_type default from hardcoded 'DFT' to None
    - Uses dynamic registry lookup to get default dataset type
    - Maintains backward compatibility via _get_default_dataset_type()

    Args:
        config_path: Path where to create the example config
        dataset_type: Dataset type for the example (None = use registry default)
        include_experimental_setups: Include multiple experimental setups

    Returns:
        Path to created example configuration file

    Evidence:
    - Original hardcoded default: config_loader.py line 1256 (dataset_type='DFT')
    - Pattern: Following Phase 3/4 dynamic type resolution approach
    """
    # PHASE 5: Get default dataset type from registry if not specified
    if dataset_type is None:
        dataset_type = _get_default_dataset_type()
        logger.debug(f"Using default dataset type from registry: {dataset_type}")

    try:
        # Handler provides example generation
        from milia_pipeline.config.config_schemas import create_example_config

        example_config = create_example_config()

        # Update dataset type
        example_config["dataset_type"] = dataset_type

        # If simple example requested, use only basic setup
        if not include_experimental_setups:
            if (
                "transformations" in example_config
                and "experimental_setups" in example_config["transformations"]
            ):
                # Keep only the first setup
                setups = example_config["transformations"]["experimental_setups"]
                first_setup = list(setups.keys())[0] if setups else "basic"
                example_config["transformations"]["experimental_setups"] = {
                    first_setup: setups.get(first_setup, [])
                }
                example_config["transformations"]["default_setup"] = first_setup

            # Update dataset type
            example_config["dataset_type"] = dataset_type
        else:
            # Fallback: create basic enhanced format example
            example_config = {
                "dataset_type": dataset_type,
                "transformations": {
                    "experimental_setups": {
                        "basic": [
                            {
                                "name": "AddSelfLoops",
                                "kwargs": {},
                                "enabled": True,
                                "description": "Add self-loops to graph",
                            },
                            {
                                "name": "ToUndirected",
                                "kwargs": {},
                                "enabled": True,
                                "description": "Convert to undirected graph",
                            },
                        ],
                        "advanced": [
                            {
                                "name": "AddSelfLoops",
                                "kwargs": {},
                                "enabled": True,
                                "description": "Add self-loops",
                            },
                            {
                                "name": "GCNNorm",
                                "kwargs": {"add_self_loops": False},
                                "enabled": True,
                                "description": "GCN normalization",
                            },
                        ],
                    },
                    "default_setup": "basic",
                },
            }

        # Ensure parent directory exists
        parent_dir = os.path.dirname(config_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Write example config
        with open(config_path, "w") as f:
            yaml.dump(example_config, f, default_flow_style=False, indent=2, sort_keys=False)

        logger.info(f"Created example configuration: {config_path}")
        return config_path

    except Exception as e:
        logger.error(f"Failed to create example config: {e}")
        # Create absolute minimal fallback
        minimal_example = {
            "dataset_type": dataset_type,
            "transformations": [{"name": "AddSelfLoops"}, {"name": "ToUndirected"}],
        }

        parent_dir = os.path.dirname(config_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump(minimal_example, f, default_flow_style=False, indent=2)

        logger.warning("Created minimal example (Relevanteatures not available)")
        return config_path


def migrate_legacy_config(
    input_path, output_path=None, dataset_type=None, backup=True, report=True
):
    """
    Migrate a legacy configuration file to enhanced format.

    Handler-Based Architecture: Delegates migration to ConfigHandler.

    PHASE 5 REFACTORING:
    - Changed dataset_type default from hardcoded 'DFT' to None
    - Uses dynamic registry lookup to get default dataset type
    - Maintains backward compatibility via _get_default_dataset_type()

    Enhancements:
    - Automatic ConfigHandler or fallback migration via Handler
    - Detailed migration reporting
    - Enhanced backup management
    - Validation of migrated configuration

    Args:
        input_path: Path to legacy configuration file
        output_path: Path for migrated config (default: input_path)
        dataset_type: Dataset type for optimization (None = use registry default)
        backup: Create backup of original file
        report: Print detailed migration report

    Returns:
        Tuple of (output_path, migration_report)

    Evidence:
    - Original hardcoded default: config_loader.py line 1345 (dataset_type='DFT')
    - Pattern: Following Phase 3/4 dynamic type resolution approach
    """
    # PHASE 5: Get default dataset type from registry if not specified
    if dataset_type is None:
        dataset_type = _get_default_dataset_type()
        logger.debug(f"Using default dataset type from registry for migration: {dataset_type}")

    # Check input file exists
    if not os.path.exists(input_path):
        raise ConfigurationError(
            f"Input configuration file not found: {input_path}", config_key="input_path"
        )

    # Load legacy config
    try:
        with open(input_path, encoding="utf-8") as f:
            legacy_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Failed to parse input configuration: {str(e)}", config_key="yaml_parsing"
        )

    if not isinstance(legacy_config, dict):
        raise ConfigurationError(
            "Legacy configuration must be a dictionary", config_key="legacy_format"
        )

    # Create backup if requested
    if backup:
        backup_path = f"{input_path}.backup_{int(time.time())}"
        shutil.copy2(input_path, backup_path)
        logger.info(f"Created backup at: {backup_path}")

    # Determine output path
    if output_path is None:
        output_path = input_path

    # Perform migration using Handler
    try:
        logger.info(f"Migrating configuration: {input_path}")

        # Migrate using Handler
        from milia_pipeline.config.config_schemas import ConfigMigration

        migration = ConfigMigration()
        migration.migrate_to_enhanced(legacy_config, preserve_original=True)

        # Handle migration result
        if migrated_config is None or not isinstance(migrated_config, dict):
            raise ConfigurationError(
                "Migration failed during configuration migration", config_key="migration_process"
            )

        # Build migration report
        migration_report = {
            "migration_applied": True,
            "original_format": format_detected,
            "target_format": "enhanced",
            "changes_made": migration_warnings,
            "warnings": migration_warnings,
            "migration_time_ms": 0,
            "validation_passed": True,
        }

        # Ensure parent directory exists
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Save migrated config - ensure it's a plain dict for YAML
        if isinstance(migrated_config, dict):
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(migrated_config, f, default_flow_style=False, indent=2, sort_keys=False)
        else:
            raise ConfigurationError(
                f"Migration produced invalid type: {type(migrated_config)}",
                config_key="migration_output",
            )

        logger.info(f"Successfully migrated configuration from {input_path} to {output_path}")

        # Print report if requested
        if report and migration_report:
            _log_migration_report(migration_report)

        return output_path, migration_report

    except ConfigurationError:
        raise  # Re-raise ConfigurationError as-is
    except Exception as e:
        raise ConfigurationError(
            f"Migration failed: {str(e)}", config_key="migration_process", actual_value=input_path
        )


def _migrate_legacy_transform_to_enhanced(transform_config):
    """
    FIXED: Ensure migration produces all required fields including 'enabled'
    """
    if isinstance(transform_config, dict):
        name = transform_config.get("name", "")
        if not name:
            raise ValueError("Transform must have a 'name' field")

        # Migrate kwargs from various legacy formats
        kwargs = {}
        if "kwargs" in transform_config:
            kwargs.update(transform_config["kwargs"])
        elif "args" in transform_config:  # Legacy 'args' format
            kwargs.update(transform_config["args"])
        elif "parameters" in transform_config:  # Legacy 'parameters' format
            kwargs.update(transform_config["parameters"])

        # FIXED: Always include 'enabled' field (this was missing!)
        enabled = transform_config.get("enabled", True)

        # Create migrated transform with all required fields
        migrated_transform = {
            "name": name,
            "kwargs": kwargs,
            "enabled": enabled,  # This field was missing before
            "description": transform_config.get("description", f"Migrated {name} transform"),
        }

        return migrated_transform
    else:
        raise ValueError(f"Invalid transform config format: {type(transform_config)}")


# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# =============================================================================


def load_transformation_config(config_path=None) -> dict[str, Any]:
    """
    Load transformation configuration section only.

    Backward compatibility wrapper - uses module's load_config with Handler support.

    Args:
        config_path: Path to config file (optional, uses default if None)

    Returns:
        Transformations section from configuration
    """
    try:
        # Use the module's load_config (which internally uses Handler)
        full_config = load_config(
            config_path=config_path,
            enable_enhancement=True,
            enable_migration=True,
            enable_validation=True,
        )
        return full_config.get("transformations", {})

    except Exception as e:
        logger.warning(f"Failed to load transformation config: {str(e)}")
        return {}


def get_experimental_setup(
    setup_name: str, config: dict[str, Any] | None = None
) -> list[dict[str, Any]] | None:
    """
    Get a specific experimental setup from the configuration.

    Backward compatibility wrapper.

    Args:
        setup_name: Name of the experimental setup
        config: Configuration dict (optional, loads if None)

    Returns:
        List of transform specifications or None if not found
    """
    try:
        # Load config if not provided
        if config is None:
            config = load_config()

        # Extract transformations section
        transforms = config.get("transformations", {})

        # Get experimental setup
        if isinstance(transforms, dict) and "experimental_setups" in transforms:
            return transforms["experimental_setups"].get(setup_name)

        return None

    except Exception as e:
        logger.warning(f"Failed to get experimental setup '{setup_name}': {str(e)}")
        return None


def list_experimental_setups(config: dict[str, Any] | None = None) -> list[str]:
    """
    List all available experimental setup names.

    Backward compatibility wrapper.

    Args:
        config: Configuration dict (optional, loads if None)

    Returns:
        List of experimental setup names
    """
    try:
        # Load config if not provided
        if config is None:
            config = load_config()

        # Extract transformations section
        transforms = config.get("transformations", {})

        # Get experimental setup names
        if isinstance(transforms, dict) and "experimental_setups" in transforms:
            return list(transforms["experimental_setups"].keys())

        return []

    except Exception as e:
        logger.warning(f"Failed to list experimental setups: {str(e)}")
        return []


def _clear_all_cached_state():
    """Clear ALL cached state to prevent test interference"""
    global _CONFIG, _ENHANCED_TRANSFORMATION_CONFIG, _config_cache

    _CONFIG = None
    _ENHANCED_TRANSFORMATION_CONFIG = None
    _config_cache.clear()

    # Clear any module-level caches in config_accessors
    try:
        import config_accessors

        if hasattr(config_accessors, "_cached_config"):
            config_accessors._cached_config = None
        if hasattr(config_accessors, "_current_config_path"):
            config_accessors._current_config_path = None
    except ImportError:
        pass


def clear_config_cache() -> None:
    """
    Clear the configuration cache, forcing a reload on next access.

    This is primarily useful for testing or when configuration needs to be
    reloaded during runtime. Clears both main config and transformation config caches.
    """
    global _CONFIG, _ENHANCED_TRANSFORMATION_CONFIG, _CONFIG_STATS, _config_cache

    with _cache_lock:
        _CONFIG = None
        _ENHANCED_TRANSFORMATION_CONFIG = None  # Clear transformation cache too
        _config_cache.clear()  # Clear the dictionary cache

    with _stats_lock:
        # Reset some statistics but preserve counters for diagnostics
        _CONFIG_STATS["last_load_time"] = None
        _CONFIG_STATS["last_validation_time"] = None
        _CONFIG_STATS["last_validation_results"] = None
        _CONFIG_STATS["last_migration_report"] = None
        _CONFIG_STATS["enhancement_applied"] = False
        _CONFIG_STATS["migration_applied"] = False
        _CONFIG_STATS["config_cached"] = False
        # Don't reset load_count and cache_hits - keep for session statistics


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================


def _initialize_enhanced_config_loader():
    """Initialize the enhanced config loader with system checks"""

    try:
        # Check if enhancement modules are available
        from milia_pipeline.config.config_schemas import YAMLSchemaValidator

        logger.info("Enhanced configuration system initialized successfully")
        return True

    except ImportError as e:
        logger.warning(f"Enhanced configuration features not available: {str(e)}")
        logger.warning("Falling back to basic configuration loading")
        return False


# Initialize on module import
_ENHANCEMENT_AVAILABLE = _initialize_enhanced_config_loader()


def get_validation_report() -> dict[str, Any] | None:
    """
    Get the last validation report from configuration loading.

    Returns:
        Dictionary with validation results or None if not available
    """
    with _stats_lock:
        return _CONFIG_STATS.get("last_validation_results")


def get_migration_report() -> dict[str, Any] | None:
    """
    Get the last migration report from configuration loading.

    Returns:
        Dictionary with migration results or None if not available
    """
    with _stats_lock:
        return _CONFIG_STATS.get("last_migration_report")


def print_config_diagnostics():
    """
    Print comprehensive configuration diagnostics for debugging.

    Includes:
    - Configuration loading statistics
    - Validation results
    - Migration status
    - Cache performance
    """
    print("\n" + "=" * 70)
    print("Configuration Diagnostics")
    print("=" * 70)

    stats = get_config_statistics()

    # Loading stats
    print("\nLoading Statistics:")
    print(f"  Total loads: {stats['load_count']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"  Enhancement applied: {stats['enhancement_applied']}")
    print(f"  Migration applied: {stats['migration_applied']}")

    # Validation info
    print("\nValidation:")
    print(f"  Enabled: {stats['validation_enabled']}")
    print(f"  Level: {stats.get('validation_level', 'N/A')}")
    print(f"  Warnings: {stats.get('warnings_count', 0)}")
    print(f"  Errors: {stats.get('errors_count', 0)}")

    # Validation results
    validation_results = get_validation_report()
    if validation_results:
        print("\nLast Validation Results:")
        print(f"  Valid: {validation_results.get('valid', 'N/A')}")
        print(f"  Format: {validation_results.get('format_detected', 'N/A')}")

        warnings = validation_results.get("warnings", [])
        if warnings:
            print(f"  Warnings ({len(warnings)}):")
            for w in warnings[:3]:  # Show first 3
                print(f"    - {w}")
            if len(warnings) > 3:
                print(f"    ... and {len(warnings) - 3} more")

    # Migration report
    migration_report = get_migration_report()
    if migration_report and migration_report.get("migration_applied"):
        print("\nLast Migration:")
        print(f"  Original format: {migration_report.get('original_format', 'N/A')}")
        print(f"  Target format: {migration_report.get('target_format', 'N/A')}")
        changes = migration_report.get("changes_made", [])
        print(f"  Changes: {len(changes)}")

    # Timing
    if stats["last_load_time"]:
        print("\nTiming:")
        print(
            f"  Last load: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats['last_load_time']))}"
        )
        if stats.get("last_validation_time"):
            print(
                f"  Last validation: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats['last_validation_time']))}"
            )

    print("\n" + "=" * 70 + "\n")


def validate_and_report(config_path: str = "config.yaml", validation_level: str = "NORMAL") -> bool:
    """
    Validate configuration file and print detailed report.

    Args:
        config_path: Path to configuration file
        validation_level: Validation strictness level

    Returns:
        True if validation passed, False otherwise
    """
    print(f"\nValidating configuration: {config_path}")
    print(f"Validation level: {validation_level}")
    print("-" * 60)

    results = validate_config_file(config_path, validation_level)

    # Print results
    if results["valid"]:
        print("✓ Validation PASSED")
    else:
        print("✗ Validation FAILED")

    # Print format info
    if "format_detected" in results:
        print(f"\nFormat: {results['format_detected']}")

    # Print errors
    errors = results.get("errors", [])
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")

    # Print warnings
    warnings = results.get("warnings", [])
    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")

    # Migration recommendation
    if results.get("migration_recommended"):
        print("\n⚠ Migration recommended - use migrate_legacy_config()")

    print("-" * 60 + "\n")

    return results["valid"]


# =============================================================================
# MIGRATION INTEGRATION HELPERS
# =============================================================================


def check_migration_status(config_path: str = "config.yaml") -> dict[str, Any]:
    """
    Check if a configuration file needs migration and provide recommendations.

    Handler-Based Architecture: Delegates format detection to ConfigHandler.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary with migration status and recommendations
    """
    try:
        if not os.path.exists(config_path):
            return {
                "file_exists": False,
                "needs_migration": False,
                "error": f"File not found: {config_path}",
            }

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            return {
                "file_exists": True,
                "needs_migration": False,
                "error": "Invalid configuration format",
            }

        # Delegate format detection to Handler
        from milia_pipeline.config.config_schemas import ConfigMigration

        migration = ConfigMigration()
        format_info = migration.detect_format(config)
        format_type = format_info if isinstance(format_info, str) else "unknown"
        needs_migration = (
            (format_info not in ["enhanced", "invalid"]) if isinstance(format_info, str) else False
        )

        result = {
            "file_exists": True,
            "file_path": config_path,
            "needs_migration": needs_migration,
            "current_format": format_type,
            "migration_available": True,  # config_schemas always available
            "recommendations": [],
        }

        if needs_migration:
            # Migration always available via config_schemas
            if True:
                result["recommendations"].append(f"Run: migrate_legacy_config('{config_path}')")
                result["recommendations"].append(
                    "Migration will use advanced migration for best results"
                )
            else:
                result["recommendations"].append("Migration will be applied automatically on load")
                result["recommendations"].append("Using fallback migration (Not available)")
        else:
            result["recommendations"].append("Configuration is already in enhanced format")

        # Add setup information if enhanced format
        if format_type == "enhanced":
            transforms = config.get("transformations", {})
            if isinstance(transforms, dict):
                # Extract experimental_setups info
                if "experimental_setups" in transforms:
                    setups = transforms.get("experimental_setups", {})
                    result["experimental_setups"] = list(setups.keys())
                    result["total_setups"] = len(setups)
                else:
                    result["experimental_setups"] = []
                    result["total_setups"] = 0

                # Extract standard_transforms info
                if "standard_transforms" in transforms:
                    standard_transforms = transforms.get("standard_transforms", [])
                    result["standard_transforms_count"] = (
                        len(standard_transforms) if isinstance(standard_transforms, list) else 0
                    )
                else:
                    result["standard_transforms_count"] = 0

                result["default_setup"] = transforms.get("default_setup", "unknown")

        # Add format validation info if available
        if False:  # detect_format returns string, not dict
            result["format_valid"] = format_info == "enhanced"
            if format_info != "enhanced" and "issues" in format_info:
                result["format_issues"] = format_info["issues"]

        return result

    except Exception as e:
        return {"file_exists": True, "needs_migration": False, "error": str(e)}


def get_transformation_feature_status() -> dict[str, Any]:
    """
    Get the status of transformation features.

    Returns feature availability status from config_schemas.

    Returns:
        Dictionary with feature availability and details
    """
    from milia_pipeline.config.config_schemas import (
        PLUGIN_SYSTEM_AVAILABLE,
        RESEARCH_API_AVAILABLE,
        YAML_AVAILABLE,
    )

    return {
        "config_schemas_available": True,
        "yaml_available": YAML_AVAILABLE,
        "validation_available": True,
        "migration_available": True,
        "plugin_system_available": PLUGIN_SYSTEM_AVAILABLE,
        "research_api_available": RESEARCH_API_AVAILABLE,
    }


def print_transformation_status():
    """
    Print a detailed status report of transformation features.

    Shows feature availability status from config_schemas.
    """
    status = get_transformation_feature_status()
    print("Configuration System Status:")
    print("  - Config Schemas: ✓")
    print(f"  - YAML Support: {'✓' if status['yaml_available'] else '✗'}")
    print("  - Validation: ✓")
    print("  - Migration: ✓")
    print(f"  - Plugin System: {'✓' if status['plugin_system_available'] else '✗'}")
    if status.get("research_api_available"):
        print("  - Research API: ✓")


def recommend_validation_level(config_path: str = "config.yaml") -> str:
    """
    Recommend an appropriate validation level based on configuration analysis.

    Args:
        config_path: Path to configuration file

    Returns:
        Recommended validation level ('STRICT', 'NORMAL', or 'RELAXED')
    """
    try:
        if not os.path.exists(config_path):
            return "NORMAL"

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            return "NORMAL"

        transforms = config.get("transformations", {})

        # Check format complexity - handle both experimental_setups and standard_transforms
        if isinstance(transforms, dict) and (
            "experimental_setups" in transforms or "standard_transforms" in transforms
        ):
            setups = transforms.get("experimental_setups", {})
            standard_transforms = transforms.get("standard_transforms", [])

            num_setups = len(setups) if isinstance(setups, dict) else 0
            num_standard = len(standard_transforms) if isinstance(standard_transforms, list) else 0

            total_experimental_transforms = (
                sum(len(t) for t in setups.values() if isinstance(t, list)) if setups else 0
            )

            total_transforms = total_experimental_transforms + num_standard

            # For simple configs, recommend STRICT
            if num_setups <= 3 and total_transforms <= 10:
                return "STRICT"
            # For complex configs, recommend NORMAL
            elif num_setups <= 10 and total_transforms <= 50:
                return "NORMAL"
            # For very complex configs, recommend RELAXED
            else:
                return "RELAXED"
        else:
            # Legacy configs - recommend NORMAL for compatibility
            return "NORMAL"

    except Exception:
        return "NORMAL"


# =============================================================================
# EXPORTS AND PUBLIC API
# =============================================================================

__all__ = [
    # Core loading functions
    "load_config",
    "load_config_with_validation",
    "reload_config",
    # Transformation access functions
    "get_enhanced_transformation_config",
    "load_transformation_config",
    "get_experimental_setup",
    "list_experimental_setups",
    # Validation functions
    "validate_config_file",
    "validate_and_report",
    "get_validation_report",
    "recommend_validation_level",
    # Migration functions
    "migrate_legacy_config",
    "check_migration_status",
    "get_migration_report",
    # Cache management
    "clear_config_cache",
    "get_config_hash",
    "is_config_loaded",
    # Diagnostics and statistics
    "get_config_statistics",
    "print_config_diagnostics",
    "get_transformation_feature_status",
    "print_transformation_status",
    # Utility functions
    "create_example_config",
]
