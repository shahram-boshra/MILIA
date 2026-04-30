# main.py - Enhanced for Experimental Setup Integration Transformation System Integration

"""
Entry point orchestration script for Milia molecular dataset pipeline.

This module serves as the primary command-line interface and orchestration layer
for the milia pipeline, coordinating dataset loading, validation, transformation,
and analysis workflows. It integrates multiple subsystems including CLI management,
handler pattern architecture, transformation system, and plugin infrastructure.

Core Responsibilities:
    - CLI argument parsing with interactive mode support
    - System initialization (handlers, transforms, plugins)
    - Configuration loading, validation, and merging
    - Dataset creation with comprehensive error handling
    - Multiple processing modes (validation, statistics, dry-run, full processing)
    - Experimental setup management for systematic research workflows
    - Plugin discovery, validation, and registration
    - Custom transform auto-discovery and registration

Architecture Integration:
    The script implements a handler-first architecture with three-tier validation:
    1. Configuration validation (syntax and structure)
    2. Handler validation (creation and interface testing)
    3. Transform validation (experimental setup and pipeline composition)

Processing Modes:
    - Configuration Validation: Validates config.yaml structure (--validate-config)
    - Transform Validation: Validates transformation system (--validate-transforms-only)
    - Handler Testing: Tests handler creation (--test-handlers-only)
    - List Operations: Shows setups/transforms (--list-experimental-setups, --list-transforms)
    - Quick Validation: Loads existing data without reprocessing (--quick-validation)
    - Statistics Only: Generates statistics from existing data (--stats-only)
    - Dry Run: Validates configuration without processing (--dry-run)
    - Full Processing: Complete dataset processing workflow (default)

System Initialization Flow:
    1. CLI Parsing (CLIManager) → Parse and validate arguments
    2. Logging Setup → Initialize comprehensive logging system
    3. Custom Transform Registration → Auto-discover custom transforms
    4. Plugin System Initialization → Discover and validate plugins
    5. Configuration Loading → Load config.yaml and merge CLI overrides
    6. Configuration Validation → Validate handlers, transforms, and datasets
    7. Processing Mode Execution → Execute requested operation

Handler Pattern Integration:
    The script uses the factory pattern to create dataset-specific handlers:
    create_dataset_handler() → DatasetHandler (Abstract)
                             ├── DFTDatasetHandler (deterministic quantum chemistry)
                             └── DMCDatasetHandler (stochastic with uncertainty)

Transformation System:
    Supports four-tier transform system:
    1. Built-in PyG Transforms (always available)
    2. Custom Transforms (auto-discovered from custom_transforms module)
    3. Plugin Transforms (from plugin system)
    4. Experimental Setups (configurations for research workflows)

Plugin System (8-Step Initialization):
    1. Feature Check → Verify plugin system availability
    2. Read Configuration → Load from config.yaml
    3. Get Settings → Auto-discover, validation parameters
    4. Discover Plugins → Call PluginRegistry.discover_plugins()
    5. Validate Plugins → Run plugin validation (if required)
    6. Apply Filters → Process enabled/disabled lists
    7. Enable Plugins → Call PluginRegistry.enable_plugin()
    8. Report Results → Log initialization summary

Error Handling Strategy:
    - Handler errors: Categorized as recoverable/non-recoverable with suggestions
    - Transform errors: Graceful degradation to legacy transforms
    - Plugin errors: Non-blocking, system continues without problematic plugins
    - Configuration errors: Fail-fast with actionable error messages
    - Data processing errors: Chunk-level recovery with partial results

Dependencies:
    Core Systems:
        - milia_pipeline.cli_manager: CLI argument parsing and validation
        - milia_pipeline.config: Configuration management and accessors
        - milia_pipeline.handlers: Dataset handler implementations
        - milia_pipeline.datasets: miliaDataset implementation
        - milia_pipeline.exceptions: Comprehensive exception hierarchy

    Optional Systems (with graceful degradation):
        - milia_pipeline.transformations.graph_transforms: Transform system
        - milia_pipeline.transformations.plugin_system: Plugin infrastructure

Module Constants:
    HANDLERS_AVAILABLE (bool): True if handler system is available
    GRAPH_TRANSFORMS_AVAILABLE (bool): True if transformation system is available
    PLUGIN_SYSTEM_AVAILABLE (bool): True if plugin system is available
    *_IMPORT_ERROR (str): Import error message if system unavailable

Notes:
    - This module implements handler-only architecture (no legacy fallbacks)
    - All operations require handlers; legacy mode removed
    - CLI-first override model: CLI Args > Config File > System Defaults
    - Non-blocking initialization for optional systems (transforms, plugins)
    - Comprehensive logging to both file and console
    - Type-safe configuration containers throughout

PHASE 7 ENHANCEMENTS:
--------------------
- Registry integration for dynamic dataset type support
- Dynamic handler availability validation from registry
- Feature-based configuration validation (not type-name based)
- Generic validation functions for uncertainty, atomization, orbital analysis
- Automatic support for new dataset types when registered
- Backward compatibility with legacy validation functions

Registry Integration Functions (Internal):
-----------------------------------------
- _init_registry(): Lazy initialization of registry imports
- _get_available_dataset_types(): Get list of registered dataset types
- _is_dataset_type_registered(): Check if dataset type is registered
- _get_dataset_feature(): Query feature flags for dataset types
- _get_dataset_config_key(): Get configuration key for dataset type
- _get_dataset_schema_attribute(): Get schema attributes for dataset type
- get_main_registry_status(): Get registry integration diagnostics

Adding New Dataset Types:
------------------------
After Phase 7, adding a new dataset type that is automatically supported by main.py:
1. Create dataset class with @register decorator (Phase 2 pattern)
2. Define features attribute with appropriate flags
3. Define config_key attribute with YAML section name
4. Add ONE import line in datasets/__init__.py
5. main.py automatically:
   - Discovers handler during validation
   - Validates configuration based on features
   - Shows feature-appropriate information
   - Collects feature-appropriate statistics

See Also:
    - milia_pipeline.cli_manager: CLI management system documentation
    - milia_pipeline.handlers: Handler pattern implementation (modular architecture)
    - milia_pipeline.transformations.graph_transforms: Transform system
    - project_hierarchical_structure.md: Complete architecture documentation

Examples:
    Basic usage:
        $ milia --config config.yaml

    Quick validation:
        $ milia --quick-validation

    Experimental setup:
        $ milia --experimental-setup baseline --force-reload

    Transform validation:
        $ milia --validate-transforms-only --experimental-setup ablation

    Handler testing:
        $ milia --test-handlers-only

    List available setups:
        $ milia --list-experimental-setups

    Note: ``python main.py`` also works as a fallback when running from
    the project root without a package install.
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

# CLI system
from milia_pipeline.cli_manager import (
    CLIManager,
    CLIValidationError,
    parse_cli_args,
)
from milia_pipeline.config.config_accessors import (
    get_dataset_constants,
    get_dataset_type,
    get_property_availability,
)
from milia_pipeline.config.config_constants import (
    ATOMIC_ENERGIES_HARTREE,
    PROCESSED_DATA_FILENAME,
)
from milia_pipeline.config.config_containers import (
    DatasetConfig,
    FilterConfig,
    ProcessingConfig,
    create_dataset_config_from_global,
    create_filter_config_from_global,
    create_processing_config_from_global,
)

# Configuration
from milia_pipeline.config.config_loader import load_config

# Transformations
try:
    from milia_pipeline.config.config_accessors import (
        get_combined_transforms_as_dicts,
        get_experimental_setup,
        get_transformation_config,
        list_available_transforms,  # noqa: F401 — module-level attr for mock.patch in tests
        list_experimental_setups,  # noqa: F401 — module-level attr for mock.patch in tests
    )
    from milia_pipeline.transformations.graph_transforms import get_graph_transforms

    GRAPH_TRANSFORMS_AVAILABLE = True
    GRAPH_TRANSFORMS_IMPORT_ERROR = None
except ImportError as e:
    GRAPH_TRANSFORMS_AVAILABLE = False
    GRAPH_TRANSFORMS_IMPORT_ERROR = str(e)

try:
    from milia_pipeline.config.config_accessors import get_config_value
    from milia_pipeline.transformations.plugin_system import PluginRegistry

    PLUGIN_SYSTEM_AVAILABLE = True
    PLUGIN_SYSTEM_IMPORT_ERROR = None
except ImportError as e:
    PLUGIN_SYSTEM_AVAILABLE = False
    PLUGIN_SYSTEM_IMPORT_ERROR = str(e)

# Import plugin-specific exceptions (always available from Plugin Exception Framework)
from milia_pipeline.exceptions import PluginDiscoveryError, PluginError, PluginValidationError

# Preprocessing System (conditional import)
try:
    from milia_pipeline.preprocessing import PreprocessorRegistry

    PREPROCESSING_AVAILABLE = True
    PREPROCESSING_IMPORT_ERROR = None
except ImportError as e:
    PREPROCESSING_AVAILABLE = False
    PREPROCESSING_IMPORT_ERROR = str(e)

# Dataset
import contextlib

from milia_pipeline.datasets.milia_dataset import miliaDataset

# Exceptions
from milia_pipeline.exceptions import (
    BaseProjectError,
    ConfigurationError,
    DataCompatibilityError,
    ExperimentalSetupError,
    HandlerCompatibilityError,
    HandlerConfigurationError,
    HandlerError,
    HandlerNotAvailableError,
    HandlerOperationError,
    HPOError,
    ModelError,
    TrainingError,
    TransformCompositionError,
    TransformConfigurationError,
    create_handler_error_context,
    format_handler_exception_summary,
    get_exception_recovery_suggestions,
    is_recoverable_handler_error,
)

# Handlers
try:
    from milia_pipeline.handlers import DatasetHandler, create_dataset_handler

    HANDLERS_AVAILABLE = True
except ImportError as e:
    HANDLERS_AVAILABLE = False
    HANDLER_IMPORT_ERROR = str(e)

# MODELS TRAINING SYSTEM (Conditional Import)
try:
    from milia_pipeline.models import (
        DataSplitter,
        # Training
        Trainer,
        get_factory,
    )

    # Phase 1 Refactor: Import optimizer and scheduler registries
    # Phase 2 Refactor: Import loss registry
    # Phase 4 Refactor: Import task data preparer
    # Phase 5 Refactor: Import metrics registry and visualization
    # These registries/factories provide DYNAMIC creation via string names
    # with automatic parameter validation and filtering
    from milia_pipeline.models.training import (
        LossRegistry,
        OptimizerRegistry,
        SchedulerRegistry,
        TaskDataPreparer,
        # NEW: Metrics and Visualization (Phase 5)
        get_metrics_for_task,
    )
    from milia_pipeline.models.training.callbacks import (
        CallbackFactory,
        EarlyStopping,  # noqa: F401 — module-level attr for mock.patch in tests
        ModelCheckpoint,  # noqa: F401 — module-level attr for mock.patch in tests
    )

    MODELS_TRAINING_AVAILABLE = True
    MODELS_TRAINING_IMPORT_ERROR = None
    METRICS_AVAILABLE = True  # NEW: Metrics available flag
except ImportError as e:
    MODELS_TRAINING_AVAILABLE = False
    MODELS_TRAINING_IMPORT_ERROR = str(e)
    METRICS_AVAILABLE = False  # NEW: Metrics not available
    # Ensure registry classes are not available if import fails
    OptimizerRegistry = None
    SchedulerRegistry = None
    LossRegistry = None
    CallbackFactory = None
    TaskDataPreparer = None

# HPO System (Conditional Import)
try:
    from milia_pipeline.models.hpo import (
        OPTUNA_AVAILABLE,
        HPOConfig,
        HPOManager,
    )

    HPO_AVAILABLE = True
    HPO_IMPORT_ERROR = None
except ImportError as e:
    HPO_AVAILABLE = False
    HPO_IMPORT_ERROR = str(e)
    OPTUNA_AVAILABLE = False

# POST-TRAINING SYSTEM (Conditional Import) - Phase 5b + Phase 7 (DI Refactoring)
try:
    from milia_pipeline.models.post_training import (
        # Inference
        Predictor,
        convert_sdf_to_pyg_list,  # FIX 24: Multi-molecule SDF support
        # Data conversion
        convert_to_pyg,
    )

    POST_TRAINING_AVAILABLE = True
    POST_TRAINING_IMPORT_ERROR = None
except ImportError as e:
    POST_TRAINING_AVAILABLE = False
    POST_TRAINING_IMPORT_ERROR = str(e)
    Predictor = None

# Target Selection Config (Conditional Import)
try:
    from milia_pipeline.models.factory.target_selection_config import TargetSelectionConfig

    TARGET_SELECTION_AVAILABLE = True
except ImportError:
    TARGET_SELECTION_AVAILABLE = False
    TargetSelectionConfig = None


# Module-level logger for registry utility functions.
# These functions may be called before setup_logging() configures the main logger.
# logging.getLogger() is safe at module level — returns a no-op logger until handlers are added.
logger = logging.getLogger(__name__)


# Registry Integration for Dynamic Dataset Type Support
# This section adds lazy initialization infrastructure for registry imports,
# following the exact pattern from dataset_handlers.py, exceptions.py
# , and cli_manager.py.
#
# The main.py module imports many components early in the application lifecycle.
# By deferring the registry import until first use, we avoid circular dependency
# issues that could occur during module initialization.

# Registry availability flags - set during lazy initialization
_REGISTRY_INITIALIZED = False
_REGISTRY_AVAILABLE = False
_REGISTRY_IMPORT_ERROR = None

# Registry function placeholders (populated by _init_registry)
_registry_list_all = None
_registry_get = None
_registry_is_registered = None

# Legacy fallback dataset types (used when registry unavailable)
_LEGACY_DATASET_TYPES = ["DFT", "DMC", "Wavefunction"]

# Legacy feature fallback (used when registry unavailable)
_LEGACY_FEATURES = {
    "DFT": {
        "vibrational_analysis": True,
        "uncertainty_handling": False,
        "atomization_energy": True,
        "rotational_constants": True,
        "frequency_analysis": True,
        "orbital_analysis": False,
    },
    "DMC": {
        "vibrational_analysis": False,
        "uncertainty_handling": True,
        "atomization_energy": False,
        "rotational_constants": False,
        "frequency_analysis": False,
        "orbital_analysis": False,
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

# Legacy config key fallback (used when registry unavailable)
_LEGACY_CONFIG_KEYS = {
    "DFT": "dft_config",
    "DMC": "dmc_config",
    "Wavefunction": "wavefunction_config",
}


def _init_registry() -> bool:
    """
    Lazily initialize registry imports to avoid circular import at module load time.

    Returns:
        True if registry is available, False otherwise

    ADDED Phase 7: Lazy initialization following established pattern from
    dataset_handlers.py (Phase 6), exceptions.py (Phase 7), cli_manager.py (Phase 7).
    """
    global _REGISTRY_INITIALIZED, _REGISTRY_AVAILABLE, _REGISTRY_IMPORT_ERROR
    global _registry_list_all, _registry_get, _registry_is_registered

    if _REGISTRY_INITIALIZED:
        return _REGISTRY_AVAILABLE

    _REGISTRY_INITIALIZED = True

    try:
        # Direct import from registry module
        from milia_pipeline.datasets.registry import (
            get,
            is_registered,
            list_all,
        )

        _registry_list_all = list_all
        _registry_get = get
        _registry_is_registered = is_registered
        _REGISTRY_AVAILABLE = True
        # Use module-level logger if available, otherwise defer logging
        with contextlib.suppress(NameError):  # Logger not yet available at module load time
            logger.debug("Dataset registry initialized successfully for main.py")
        return True

    except ImportError as e:
        _REGISTRY_IMPORT_ERROR = str(e)
        with contextlib.suppress(NameError):  # Logger not yet available at module load time
            logger.debug(f"Dataset registry not available - using legacy validation: {e}")
        return False

    except Exception as e:
        _REGISTRY_IMPORT_ERROR = str(e)
        with contextlib.suppress(NameError):  # Logger not yet available at module load time
            logger.debug(f"Dataset registry import failed - using legacy validation: {e}")
        return False


def _get_available_dataset_types() -> list[str]:
    """
    Get list of available dataset types from registry or fallback.

    Returns:
        List of registered dataset type names, or legacy fallback list

    ADDED Phase 7: Dynamic dataset type discovery.
    """
    _init_registry()
    if _REGISTRY_AVAILABLE and _registry_list_all is not None:
        try:
            return _registry_list_all()
        except Exception as e:
            with contextlib.suppress(NameError):
                logger.debug(f"Registry list_all failed: {e}")
    return _LEGACY_DATASET_TYPES.copy()


def _resolve_canonical_dataset_type(dataset_type: str) -> str:
    """
    Resolve a dataset type name to its canonical registry name.

    PHASE 6.2 SIMPLIFICATION: Case-insensitive normalization is now handled by
    config_loader.py at load time. This function receives already-normalized
    values and simply returns them. Kept for backward compatibility.

    Args:
        dataset_type: Dataset type name (already normalized by config_loader)

    Returns:
        The input dataset_type (already canonical)
    """
    # Config is already normalized by config_loader.py
    return dataset_type


def _is_dataset_type_registered(dataset_type: str) -> bool:
    """
    Check if dataset type is registered in registry or known legacy type.

    Args:
        dataset_type: Type name to check

    Returns:
        True if registered or known legacy type, False otherwise

    ADDED Phase 7: Dynamic dataset type validation.
    PHASE 6.2 SIMPLIFICATION: Case-insensitive normalization now handled by
                              config_loader.py at load time. This function
                              receives already-normalized canonical names.
    """
    _init_registry()
    if _REGISTRY_AVAILABLE and _registry_is_registered is not None:
        try:
            return _registry_is_registered(dataset_type)
        except Exception as e:
            with contextlib.suppress(NameError):
                logger.debug(f"Registry is_registered failed: {e}")
    # Simple check for legacy types (already canonical)
    return dataset_type in _LEGACY_DATASET_TYPES


def _get_dataset_feature(dataset_type: str, feature_name: str, default: bool = False) -> bool:
    """
    Query a specific feature flag for a dataset type from registry.

    Args:
        dataset_type: Dataset type name (e.g., 'DFT', 'DMC')
        feature_name: Feature flag name (e.g., 'uncertainty_handling')
        default: Default value if feature not found

    Returns:
        Feature flag value

    ADDED Phase 7: Feature-based logic support.
    PHASE 6.2 SIMPLIFICATION: dataset_type is already normalized by config_loader.py.

    Evidence: base.py DatasetFeatures dataclass provides:
    - vibrational_analysis, uncertainty_handling, atomization_energy
    - rotational_constants, frequency_analysis, orbital_analysis
    - homo_lumo_gap, mo_energies
    """
    _init_registry()

    # dataset_type is already canonical from config_loader.py
    if _REGISTRY_AVAILABLE and _registry_get is not None:
        try:
            dataset_class = _registry_get(dataset_type)
            if hasattr(dataset_class, "features"):
                features = dataset_class.features
                # DatasetFeatures is a dataclass with direct attribute access
                if hasattr(features, feature_name):
                    return getattr(features, feature_name)
                # Also check via to_dict() method
                if hasattr(features, "to_dict"):
                    features_dict = features.to_dict()
                    return features_dict.get(feature_name, default)
        except Exception as e:
            with contextlib.suppress(NameError):
                logger.debug(f"Feature query failed for {dataset_type}.{feature_name}: {e}")

    # Legacy fallback (dataset_type already canonical)
    if dataset_type in _LEGACY_FEATURES:
        return _LEGACY_FEATURES[dataset_type].get(feature_name, default)
    return default


def _get_dataset_config_key(dataset_type: str) -> str | None:
    """
    Get the configuration key for a dataset type from registry.

    Args:
        dataset_type: Dataset type name (e.g., 'DFT', 'DMC')

    Returns:
        Config key (e.g., 'dft_config') or None if not found

    ADDED Phase 7: Dynamic config key lookup.
    PHASE 6.2 SIMPLIFICATION: dataset_type is already normalized by config_loader.py.

    Evidence: base.py BaseDataset.config_key class attribute.
    """
    _init_registry()

    # dataset_type is already canonical from config_loader.py
    if _REGISTRY_AVAILABLE and _registry_get is not None:
        try:
            dataset_class = _registry_get(dataset_type)
            if hasattr(dataset_class, "config_key"):
                return dataset_class.config_key
        except Exception as e:
            with contextlib.suppress(NameError):
                logger.debug(f"Config key query failed for {dataset_type}: {e}")

    # Legacy fallback (dataset_type already canonical)
    return _LEGACY_CONFIG_KEYS.get(dataset_type)


def _get_dataset_schema_attribute(dataset_type: str, attr_name: str, default: Any = None) -> Any:
    """
    Get a schema attribute for a dataset type from registry.

    Args:
        dataset_type: Dataset type name
        attr_name: Schema attribute name (e.g., 'coordinate_units')
        default: Default value if not found

    Returns:
        Schema attribute value

    ADDED Phase 7: Schema-based logic support.
    PHASE 6.2 SIMPLIFICATION: dataset_type is already normalized by config_loader.py.

    Evidence: base.py DatasetSchema provides:
    - required_properties, optional_properties, identifier_keys
    - coordinate_units, energy_units
    """
    _init_registry()

    # dataset_type is already canonical from config_loader.py
    if _REGISTRY_AVAILABLE and _registry_get is not None:
        try:
            dataset_class = _registry_get(dataset_type)
            if hasattr(dataset_class, "schema"):
                schema = dataset_class.schema
                if hasattr(schema, attr_name):
                    return getattr(schema, attr_name)
        except Exception as e:
            with contextlib.suppress(NameError):
                logger.debug(f"Schema query failed for {dataset_type}.{attr_name}: {e}")

    return default


def get_main_registry_status() -> dict[str, Any]:
    """
    Get registry integration status for main.py diagnostics.

    Returns:
        Dict with registry status information

    ADDED Phase 7: Diagnostic function for troubleshooting.
    """
    _init_registry()

    return {
        "registry_available": _REGISTRY_AVAILABLE,
        "registry_initialized": _REGISTRY_INITIALIZED,
        "registry_import_error": _REGISTRY_IMPORT_ERROR,
        "available_dataset_types": _get_available_dataset_types(),
        "using_legacy_fallback": not _REGISTRY_AVAILABLE,
        "phase_7_integration": True,
    }


# =============================================================================
# TASK-SPECIFIC DATA PREPARATION (Phase 4 Refactor)
# =============================================================================


def prepare_data_for_task(
    train_data,
    val_data,
    test_data,
    task_type: str,
    logger: logging.Logger,
    target_selection_config: Any | None = None,
) -> tuple[Any, Any, Any, int | None]:
    """
    Prepare split data for specific task types using TaskDataPreparer.

    Delegates to TaskDataPreparer.prepare_for_task() for:
    - DYNAMIC: Supports 7 task types via registry pattern
    - PRODUCTION-READY: Automatic target extraction, discretization, validation
    - FUTURE-PROOF: New task types auto-available when registered

    This function applies task-specific transformations AFTER dataset splitting
    and BEFORE DataLoader creation, following PyG conventions.

    Args:
        train_data: Training subset from DataSplitter
        val_data: Validation subset from DataSplitter
        test_data: Test subset from DataSplitter
        task_type: Task type string from config
        logger: Logger instance
        target_selection_config: Optional TargetSelectionConfig for specifying
            target source (x, y, edge_attr) and indices

    Returns:
        Tuple of (train_data, val_data, test_data, num_classes) where:
        - train_data, val_data, test_data: potentially transformed data
        - num_classes: Number of classes for classification, or None

    Raises:
        DataCompatibilityError: If data is incompatible with task_type
        RuntimeError: If MODELS_TRAINING system is not available
    """
    # Validate preparer availability
    if not MODELS_TRAINING_AVAILABLE or TaskDataPreparer is None:
        raise RuntimeError(
            f"TaskDataPreparer not available. Import error: {MODELS_TRAINING_IMPORT_ERROR}"
        )

    # Delegate to TaskDataPreparer
    return TaskDataPreparer.prepare_for_task(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        task_type=task_type,
        logger=logger,
        target_selection_config=target_selection_config,
    )


def _register_custom_transforms_on_startup(logger=None):
    """
    Auto-discover and register custom transforms during pipeline initialization.

    This function implements non-blocking custom transform registration that extends
    the pipeline's transformation capabilities beyond built-in PyG transforms. It
    discovers custom transforms from the custom_transforms module and registers them
    with the transform registry for use in experimental setups.

    The registration process is non-blocking: if it fails, the pipeline continues
    with only built-in PyG transforms. This ensures the pipeline remains functional
    even when custom transforms are unavailable or misconfigured.

    Discovery Sources:
        1. Built-in custom_transforms module (milia_pipeline/transformations/custom_transforms)
        2. User-specified plugin paths (future enhancement via plugin system)

    Registration Process:
        1. Check custom_transforms module availability
        2. Import register_all_custom_transforms() function
        3. Call registration function (returns count of registered transforms)
        4. Log registration results (success/failure/count)
        5. Continue pipeline operation regardless of registration outcome

    Failure Handling:
        - ImportError: Custom transforms module not found (logged as debug, not error)
        - Exception: Other registration errors logged as error with warning
        - All failures are non-fatal: pipeline continues with built-in transforms

    Args:
        logger (logging.Logger, optional): Logger instance for registration messages.
            If not provided, creates a basic StreamHandler logger with INFO level.
            Default: None (creates fallback logger).

    Returns:
        None: Results logged via logger, no return value.

    Side Effects:
        - Registers custom transforms with global transform registry
        - Logs registration header, results, and any errors
        - Creates fallback logger if none provided

    Logging Output:
        On success:
            INFO: "Custom Transform Registration Phase" (with separator)
            INFO: "✓ Successfully registered N custom transform(s)"

        On no transforms found:
            INFO: "No custom transforms found to register"

        On module unavailable:
            INFO: "Custom transforms module not available - skipping registration"
            DEBUG: "Pipeline will use built-in PyG transforms only"

        On error:
            ERROR: "Error during custom transform registration: {error}"
            WARNING: "Continuing without custom transforms"

    Notes:
        - Called automatically during main() initialization
        - Non-blocking design ensures pipeline robustness
        - Custom transforms extend but don't replace built-in transforms
        - Registration occurs before dataset creation
        - Transform count returned by register_all_custom_transforms()

    See Also:
        milia_pipeline.transformations.graph_transforms.register_all_custom_transforms:
            Function that performs actual registration
        milia_pipeline.transformations.custom_transforms: Custom transform module

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> _register_custom_transforms_on_startup(logger)
        ============================================================
        Custom Transform Registration Phase
        ============================================================
        ✓ Successfully registered 3 custom transform(s)
        ============================================================
    """
    # Create fallback logger if not provided
    if logger is None:
        import logging

        logger = logging.getLogger("CustomTransformRegistration")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            logger.addHandler(handler)

    try:
        # Import custom transform registration functions
        from milia_pipeline.transformations.graph_transforms import (
            CUSTOM_TRANSFORMS_AVAILABLE,
            register_all_custom_transforms,
        )

        # Check if custom transforms module is available
        if not CUSTOM_TRANSFORMS_AVAILABLE:
            logger.info("Custom transforms module not available - skipping registration")
            logger.debug("Pipeline will use built-in PyG transforms only")
            return

        # Print registration header
        logger.info("=" * 60)
        logger.info("Custom Transform Registration Phase")
        logger.info("=" * 60)

        # Register all discovered custom transforms
        # register_all_custom_transforms() returns int (count of registered transforms)
        count = register_all_custom_transforms()

        if count > 0:
            logger.info(f"✓ Successfully registered {count} custom transform(s)")
        else:
            logger.info("No custom transforms found to register")

        logger.info("=" * 60)

    except ImportError as e:
        # Custom transform module not found - not an error
        logger.debug(f"Custom transform registration skipped: {e}")
        logger.debug("Pipeline will use built-in PyG transforms only")

    except Exception as e:
        # Other errors during registration - log but don't crash
        logger.error(f"Error during custom transform registration: {e}")
        logger.warning("Continuing without custom transforms")
        logger.debug(f"Registration error details: {type(e).__name__}: {str(e)}")


def _discover_and_register_plugins(logger=None):
    """
    Discover and register plugins from configured paths during pipeline startup.

    Plugin System Integration: Main pipeline integration for plugin system.

    This function:
    1. Reads plugin configuration from config.yaml
    2. Discovers plugins in configured paths
    3. Validates plugins (if required)
    4. Enables/disables plugins based on configuration
    5. Registers plugin transforms with the transform registry

    The function is called during pipeline initialization and is non-blocking.
    If plugin discovery/registration fails, the pipeline continues with only
    built-in and custom transforms.

    Args:
        logger: Logger instance (optional, creates basic logger if not provided)

    Returns:
        None (logs results)
    """
    # ========================================================================
    # STEP 1: Setup and Feature Check
    # ========================================================================

    # Create fallback logger if not provided
    if logger is None:
        import logging

        logger = logging.getLogger("PluginDiscovery")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            logger.addHandler(handler)

    # Check if plugin system is available
    if not PLUGIN_SYSTEM_AVAILABLE:
        logger.debug(f"Plugin system not available: {PLUGIN_SYSTEM_IMPORT_ERROR}")
        logger.debug("Pipeline will use built-in and custom transforms only")
        return

    try:
        # ====================================================================
        # STEP 2: Read Plugin Configuration
        # ====================================================================

        logger.info("=" * 60)
        logger.info("Plugin System Initialization")
        logger.info("=" * 60)

        # Load configuration - this is what the tests mock
        from milia_pipeline.config.config_loader import load_config

        config = load_config()

        # Get plugins section
        plugins_config = config.get("plugins", {})

        # Check if plugins enabled in config.yaml
        plugins_enabled = plugins_config.get("enabled", False)

        if not plugins_enabled:
            logger.info("Plugin system disabled in configuration (plugins.enabled = false)")
            logger.debug("To enable plugins, set 'plugins.enabled: true' in config.yaml")
            logger.info("=" * 60)
            return

        logger.info("Plugin system enabled in configuration")

        # Read plugin paths from configuration
        plugin_paths_config = plugins_config.get("plugin_paths", [])
        # Also try 'paths' as alternative key (for backward compatibility)
        if not plugin_paths_config:
            plugin_paths_config = plugins_config.get("paths", [])

        if not plugin_paths_config:
            logger.warning("No plugin paths configured in config.yaml")
            logger.info("Add plugin paths under 'plugins.paths' to discover plugins")
            logger.info("=" * 60)
            return

        # Expand user paths (~ expansion) and convert to Path objects
        from pathlib import Path

        plugin_paths = []
        for path_str in plugin_paths_config:
            expanded_path = Path(path_str).expanduser()
            plugin_paths.append(expanded_path)
            logger.debug(f"Plugin path configured: {expanded_path}")

        # ====================================================================
        # STEP 3: Get Plugin System Configuration
        # ====================================================================

        # Auto-discover setting
        auto_discover = plugins_config.get("auto_discover", True)
        logger.debug(f"Auto-discover: {auto_discover}")

        # Validation settings
        require_validation = plugins_config.get("require_validation", True)
        auto_validate = plugins_config.get("auto_validate", True)
        logger.debug(f"Require validation: {require_validation}")
        logger.debug(f"Auto-validate on discovery: {auto_validate}")

        # Trust mode
        trust_mode = plugins_config.get("trust_mode", "normal")
        logger.debug(f"Trust mode: {trust_mode}")

        # Plugin lists
        enabled_plugins_list = plugins_config.get("enabled_plugins", [])
        disabled_plugins_list = plugins_config.get("disabled_plugins", [])

        if enabled_plugins_list:
            logger.debug(f"Enabled plugins list: {enabled_plugins_list}")
        if disabled_plugins_list:
            logger.debug(f"Disabled plugins list: {disabled_plugins_list}")

        # ====================================================================
        # STEP 4: Discover Plugins
        # ====================================================================

        if not auto_discover:
            logger.info("Auto-discover disabled - skipping plugin discovery")
            logger.info("=" * 60)
            return

        logger.info(f"Discovering plugins in {len(plugin_paths)} path(s)...")

        discovered_plugins = []

        # Call PluginRegistry.discover_plugins - this is what tests expect
        try:
            # The tests mock PluginRegistry.discover_plugins to return a list
            discovered_plugins = PluginRegistry.discover_plugins(
                plugin_paths, auto_validate=auto_validate
            )
            logger.debug(f"Found {len(discovered_plugins)} plugin(s)")

        except PluginDiscoveryError as e:
            logger.error(f"Plugin discovery failed: {e}")
            return
        except Exception as e:
            logger.error(f"Unexpected error discovering plugins: {e}")
            logger.debug(f"Error details: {type(e).__name__}: {str(e)}", exc_info=True)
            return

        if not discovered_plugins:
            logger.info("No plugins discovered")
            logger.info("=" * 60)
            return

        logger.info(f"✓ Discovered {len(discovered_plugins)} plugin(s)")

        # ====================================================================
        # STEP 5: Validate Plugins (if required)
        # ====================================================================

        validated_plugins = []
        validation_failures = []

        if require_validation:
            logger.info("Validating discovered plugins...")

            for plugin_name in discovered_plugins:
                try:
                    # Get plugin info - tests mock this
                    plugin_info = PluginRegistry.get_plugin_info(plugin_name)

                    # Check if already validated during discovery
                    if auto_validate and plugin_info and plugin_info.get("is_validated", False):
                        validated_plugins.append(plugin_name)
                        logger.debug(f"✓ {plugin_name}: Already validated")
                        continue

                    # Validate plugin - tests mock this
                    validation_result = PluginRegistry.validate_plugin(plugin_name)

                    if validation_result.get("passed", False):
                        validated_plugins.append(plugin_name)
                        logger.debug(f"✓ {plugin_name}: Validation passed")
                    else:
                        validation_failures.append(plugin_name)
                        tests = validation_result.get("tests", {})
                        failed_tests = [
                            name
                            for name, result in tests.items()
                            if not result.get("passed", False)
                        ]
                        logger.warning(
                            f"✗ {plugin_name}: Validation failed - Failed tests: {failed_tests}"
                        )

                except PluginValidationError as e:
                    validation_failures.append(plugin_name)
                    logger.error(f"✗ {plugin_name}: Validation error - {e}")
                except Exception as e:
                    validation_failures.append(plugin_name)
                    logger.error(f"✗ {plugin_name}: Unexpected validation error - {e}")

            if validation_failures:
                logger.warning(f"Validation failed for {len(validation_failures)} plugin(s)")
                logger.info(f"Successfully validated: {len(validated_plugins)} plugin(s)")
            else:
                logger.info(f"✓ All {len(validated_plugins)} plugin(s) validated successfully")

            # Use only validated plugins
            plugins_to_enable = validated_plugins
        else:
            logger.warning("Plugin validation disabled - using all discovered plugins")
            plugins_to_enable = discovered_plugins

        # ====================================================================
        # STEP 6: Apply Enabled/Disabled Lists
        # ====================================================================

        # Filter based on enabled_plugins list (if specified)
        if enabled_plugins_list:
            logger.debug("Filtering by enabled_plugins list...")
            plugins_to_enable = [p for p in plugins_to_enable if p in enabled_plugins_list]
            logger.debug(f"Plugins after enabled filter: {plugins_to_enable}")

        # Remove disabled plugins
        if disabled_plugins_list:
            logger.debug("Removing disabled plugins...")
            plugins_to_enable = [p for p in plugins_to_enable if p not in disabled_plugins_list]
            logger.debug(f"Plugins after disabled filter: {plugins_to_enable}")

        if not plugins_to_enable:
            logger.warning("No plugins remain after filtering")
            logger.info("=" * 60)
            return

        # ====================================================================
        # STEP 7: Enable Plugins
        # ====================================================================

        logger.info(f"Enabling {len(plugins_to_enable)} plugin(s)...")

        enabled_count = 0
        enable_failures = []

        for plugin_name in plugins_to_enable:
            try:
                # This is what tests expect to be called
                PluginRegistry.enable_plugin(plugin_name)
                enabled_count += 1
                logger.debug(f"✓ Enabled: {plugin_name}")

            except PluginError as e:
                enable_failures.append(plugin_name)
                logger.error(f"✗ Failed to enable {plugin_name}: {e}")
            except Exception as e:
                enable_failures.append(plugin_name)
                logger.error(f"✗ Unexpected error enabling {plugin_name}: {e}")

        # ====================================================================
        # STEP 8: Report Results
        # ====================================================================

        logger.info("=" * 60)
        logger.info("Plugin System Initialization Complete")
        logger.info("=" * 60)
        logger.info(f"Total discovered: {len(discovered_plugins)}")

        if require_validation:
            logger.info(f"Total validated: {len(validated_plugins)}")
            if validation_failures:
                logger.info(f"Validation failures: {len(validation_failures)}")

        logger.info(f"Total enabled: {enabled_count}")

        if enable_failures:
            logger.info(f"Enable failures: {len(enable_failures)}")

        if enabled_count > 0:
            logger.info("✓ Plugin system ready")
            logger.info("  Use --list-plugins to see available plugin transforms")
        else:
            logger.warning("No plugins successfully enabled")

        logger.info("=" * 60)

    except PluginError as e:
        # Plugin-specific errors - log but don't crash
        logger.error(f"Plugin system error: {e}")
        logger.warning("Continuing without plugins")
        logger.debug(f"Error details: {type(e).__name__}: {str(e)}", exc_info=True)

    except Exception as e:
        # Unexpected errors - log but don't crash
        logger.error(f"Unexpected error during plugin initialization: {e}")
        logger.warning("Continuing without plugins")
        logger.debug(f"Error details: {type(e).__name__}: {str(e)}", exc_info=True)


def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """
    Setup logging configuration.

    Configures both the root logger and milia_Main logger to ensure all module
    loggers (using __name__) inherit the same handlers and configuration.

    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (str, optional): Path to log file. If None, logs to console only.

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)

    # File handler (if specified)
    file_handler = None
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always debug level for file
        file_handler.setFormatter(formatter)

    # Configure ROOT logger so all module loggers inherit handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Allow all levels, handlers filter
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    if file_handler:
        root_logger.addHandler(file_handler)

    # Create milia_Main logger (inherits from root, no need for own handlers)
    logger = logging.getLogger("milia_Main")
    logger.setLevel(getattr(logging, log_level.upper()))
    # Ensure propagation to root (default is True, but be explicit)
    logger.propagate = True

    if log_file:
        logger.info(f"Logging to file: {log_file}")

    return logger


def validate_handler_availability(logger: logging.Logger) -> bool:
    """
    Validates that dataset handlers are available and functional.

    REFACTORED Phase 7: Now uses registry to dynamically discover and validate
    all registered dataset types instead of hardcoded DFT/DMC testing.

    Args:
        logger (logging.Logger): Logger instance

    Returns:
        bool: True if handlers are available and functional

    Raises:
        HandlerNotAvailableError: If handlers cannot be imported or used
        HandlerCompatibilityError: If handlers are incompatible
    """
    logger.info("Validating dataset handler availability...")

    if not HANDLERS_AVAILABLE:
        error_msg = f"Dataset handlers not available: {HANDLER_IMPORT_ERROR}"
        logger.error(error_msg)
        raise HandlerNotAvailableError(
            message="Dataset handler modules could not be imported",
            requested_dataset_type="unknown",
            missing_dependencies=[HANDLER_IMPORT_ERROR],
            details=f"Import error: {HANDLER_IMPORT_ERROR}",
        )

    try:
        # Verify handler system is available by checking core handler classes
        try:
            from milia_pipeline.handlers import create_dataset_handler
        except ImportError as e:
            raise HandlerNotAvailableError(
                message="Handler module not available",
                requested_dataset_type="unknown",
                missing_dependencies=[str(e)],
                details=f"Import error: {str(e)}",
            ) from e

        # ================================================================
        # PHASE 7: Dynamic handler availability testing from registry
        # ================================================================
        available_types = []

        # Get all registered dataset types dynamically
        registered_types = _get_available_dataset_types()
        logger.debug(f"Testing handlers for registered types: {registered_types}")

        for dataset_type in registered_types:
            try:
                # Create test config for this dataset type
                test_config = DatasetConfig(dataset_type=dataset_type)
                filter_cfg = create_filter_config_from_global()
                proc_cfg = create_processing_config_from_global()

                # Attempt to create handler
                test_handler = create_dataset_handler(test_config, filter_cfg, proc_cfg, logger)

                if test_handler:
                    available_types.append(dataset_type)
                    logger.debug(f"Handler available for: {dataset_type}")

            except Exception as e:
                # Handler unavailable for this type - log and continue
                logger.debug(f"{dataset_type} handler unavailable: {e}")

        if not available_types:
            raise HandlerNotAvailableError(
                message="No dataset handlers available (Requested: any)",
                requested_dataset_type="any",
                available_types=[],
                details=f"Handler factory could not create any handlers. "
                f"Registered types tested: {registered_types}",
            )

        logger.info(f"Available handler types: {', '.join(available_types)}")

        # ================================================================
        # Verify essential handler types are available
        # PHASE 7: Use registry to determine "essential" types
        # ================================================================
        # If registry available, all registered types are essential
        # If registry unavailable, DFT and DMC are essential (legacy)
        if _REGISTRY_AVAILABLE:
            # All registered types should ideally be available
            missing_types = set(registered_types) - set(available_types)
        else:
            # Legacy: DFT and DMC are required
            required_types = {"DFT", "DMC"}
            missing_types = required_types - set(available_types)

        if missing_types:
            logger.warning(f"Some registered handler types are unavailable: {missing_types}")
            # Don't fail - some handlers being unavailable is acceptable
            # Only fail if NO handlers are available (checked above)

        logger.info("✅ Dataset handlers validated successfully")
        return True

    except HandlerError:
        # Re-raise handler errors as-is
        raise
    except Exception as e:
        logger.error(f"Handler validation failed with unexpected error: {e}")
        raise HandlerCompatibilityError(
            message="Handler validation failed",
            handler_type="factory",
            incompatible_features=["handler_factory_test"],
            details=f"Unexpected error: {type(e).__name__}: {str(e)}",
        ) from e


def validate_transformation_system(
    logger: logging.Logger, experimental_setup: str | None = None
) -> bool:
    """
    NEW: Experimental Setup Integration - Validate transformation system availability and configuration.

    Args:
        logger (logging.Logger): Logger instance
        experimental_setup (Optional[str]): Experimental setup to validate

    Returns:
        bool: True if transformation system is available and valid

    Raises:
        TransformConfigurationError: If transform configuration is invalid
        ExperimentalSetupError: If experimental setup is invalid
    """
    logger.info("Validating transformation system...")

    if not GRAPH_TRANSFORMS_AVAILABLE:
        logger.warning(f"Transformation system not available: {GRAPH_TRANSFORMS_IMPORT_ERROR}")
        return False

    try:
        # Validate graph transforms module
        gt = get_graph_transforms()

        # ============================================================================
        # Get available transforms with robust error handling
        # ============================================================================
        try:
            available_transforms = gt.get_available_transforms()

            # Calculate total transforms based on return structure
            if isinstance(available_transforms, dict):
                # New format: {'basic': [...], 'augmentation': [...], 'count': N}
                total_transforms = available_transforms.get("count", 0)
                if total_transforms == 0:
                    # Fallback: calculate from categories
                    total_transforms = sum(
                        len(v) for k, v in available_transforms.items() if isinstance(v, list)
                    )
            else:
                # Old format: might be a list or other structure
                total_transforms = (
                    len(available_transforms) if hasattr(available_transforms, "__len__") else 0
                )

            if total_transforms > 0:
                logger.info(f"Transform system available with {total_transforms} transforms")
            else:
                logger.debug("Transform system available (transform count unavailable)")

        except AttributeError:
            # Method doesn't exist - not an error, just graceful degradation
            logger.debug(
                "Transform discovery method not available - continuing with basic validation"
            )
            total_transforms = 0

        except Exception as e:
            # Other errors - log at debug level since validation continues
            logger.debug(f"Could not determine transform count: {e}")
            total_transforms = 0

        # ============================================================================
        # NEW: Custom Transform Autodiscovery - Validate custom transforms
        # ============================================================================
        try:
            from milia_pipeline.transformations.graph_transforms import (
                CUSTOM_TRANSFORMS_AVAILABLE,
                get_custom_transform_count,
            )

            if CUSTOM_TRANSFORMS_AVAILABLE:
                custom_count = get_custom_transform_count()
                if custom_count > 0:
                    logger.info(f"Custom transforms available: {custom_count}")
                else:
                    logger.debug("Custom transform system available but no transforms registered")
            else:
                logger.debug("Custom transforms not available - using built-in transforms only")

        except ImportError:
            logger.debug("Custom transform validation skipped - module not available")
        except Exception as e:
            logger.debug(f"Could not validate custom transforms: {e}")

        # ============================================================================
        # Validate transformation configuration
        # ============================================================================
        try:
            transform_config = get_transformation_config()

            # Check if transform_config is valid
            has_experimental = hasattr(transform_config, "experimental_setups")
            has_standard = (
                hasattr(transform_config, "has_standard_transforms")
                and transform_config.has_standard_transforms()
            )

            if not has_experimental and not has_standard:
                # NOTE: Missing both experimental_setups and standard_transforms is acceptable for legacy configs
                logger.debug(
                    "Transform configuration missing 'experimental_setups' and 'standard_transforms' - using legacy format"
                )
                return True  # Not an error - legacy configs work fine

            # Log standard transforms info if available
            if has_standard:
                standard_count = len(transform_config.get_standard_transforms())
                logger.info(f"Standard transforms configured: {standard_count}")

            available_setups = []
            if has_experimental:
                available_setups = list(transform_config.experimental_setups.keys())
                logger.info(f"Available experimental setups: {available_setups}")

            # Validate default setup
            if hasattr(transform_config, "default_setup"):
                default_setup = transform_config.default_setup
                if available_setups and default_setup not in available_setups:
                    if has_standard:
                        logger.info(
                            f"Default setup '{default_setup}' not in experimental_setups - will use standard transforms"
                        )
                    else:
                        logger.warning(
                            f"Default setup '{default_setup}' not found in available setups"
                        )
                else:
                    logger.info(f"Default experimental setup: '{default_setup}'")
            else:
                logger.warning("Transform configuration missing 'default_setup' attribute")

            # Validate specific experimental setup if requested
            if experimental_setup:
                if experimental_setup not in available_setups:
                    logger.error(f"❌ Requested setup '{experimental_setup}' not found")
                    logger.error(f"Available setups: {', '.join(available_setups)}")
                    raise ExperimentalSetupError(
                        f"Experimental setup '{experimental_setup}' not found",
                        setup_name=experimental_setup,
                        available_setups=available_setups,
                    )

                logger.debug(f"Validating experimental setup: '{experimental_setup}'")
                # ---
                # Validate setup configuration
                try:
                    setup_config = get_experimental_setup(transform_config, experimental_setup)

                    # Validate setup configuration if validation method is available
                    if hasattr(gt, "validate_config"):
                        validation_result = gt.validate_config(setup_config)

                        if not validation_result.get("valid", False):
                            error_details = "; ".join(validation_result.get("errors", []))
                            raise ExperimentalSetupError(
                                f"Experimental setup '{experimental_setup}' validation failed: {error_details}",
                                setup_name=experimental_setup,
                                validation_errors=validation_result.get("errors", []),
                                setup_config=setup_config,
                            )

                        logger.info(
                            f"✅ Experimental setup '{experimental_setup}' validated successfully"
                        )

                        # Log warnings if any
                        for warning in validation_result.get("warnings", []):
                            logger.warning(f"Setup validation: {warning}")
                    else:
                        # Basic validation without detailed checks
                        logger.debug(
                            f"Detailed validation not available for '{experimental_setup}' - using basic checks"
                        )
                        logger.info(
                            f"✅ Experimental setup '{experimental_setup}' loaded successfully"
                        )
                # ----
                except Exception as setup_error:
                    logger.warning(
                        f"Could not validate experimental setup '{experimental_setup}': {setup_error}"
                    )
                    # Continue - non-fatal for now

        except AttributeError as e:
            # Transform configuration attributes missing
            logger.warning(f"Transform configuration validation failed: {e}")
            logger.warning(
                "Transformation system validation incomplete - continuing with legacy transforms"
            )
            return True

        except Exception as e:
            logger.warning(f"Transform configuration validation error: {e}")
            logger.warning("Continuing with basic transformation system validation")
        # ----
        # Validation successful
        logger.info("✅ Transformation system validated successfully")
        return True

    except (TransformConfigurationError, ExperimentalSetupError):
        # Re-raise critical transform errors
        raise

    except Exception as e:
        # Non-critical errors - allow pipeline to continue with legacy transforms
        logger.warning(f"Transformation system validation encountered error: {e}")
        logger.debug(f"Error details: {type(e).__name__}: {str(e)}", exc_info=True)
        logger.info("Continuing with legacy transform system")

        # Return True to allow pipeline to continue
        return True


def list_experimental_setups_info(logger: logging.Logger) -> None:
    """
    List available experimental setups and standard transforms with details.

    Shows both standard transforms (always applied) and experimental setups
    (for research variations). Combined transforms are applied in order:
    standard_transforms first, then experimental setup transforms.

    Args:
        logger (logging.Logger): Logger instance
    """
    if not GRAPH_TRANSFORMS_AVAILABLE:
        logger.error("Transformation system not available - cannot list experimental setups")
        logger.error(f"Error: {GRAPH_TRANSFORMS_IMPORT_ERROR}")
        return

    try:
        # Get transformation configuration
        transform_config = get_transformation_config()

        logger.info("=" * 60)
        logger.info("TRANSFORMATION CONFIGURATION")
        logger.info("=" * 60)

        # Display standard transforms first
        if (
            hasattr(transform_config, "has_standard_transforms")
            and transform_config.has_standard_transforms()
        ):
            standard_transforms = transform_config.get_standard_transforms()
            logger.info(f"\nSTANDARD TRANSFORMS ({len(standard_transforms)} transforms)")
            logger.info("-" * 40)
            logger.info("(Always applied first, before experimental setups)")

            for i, transform in enumerate(standard_transforms):
                transform_name = transform.name if hasattr(transform, "name") else str(transform)
                enabled = getattr(transform, "enabled", True)
                status = "✓" if enabled else "✗"
                logger.info(f"  {i + 1}. [{status}] {transform_name}")
        else:
            logger.info("\nNo standard transforms configured")

        # Display experimental setups
        setups = (
            list(transform_config.experimental_setups.keys())
            if hasattr(transform_config, "experimental_setups")
            else []
        )

        if setups:
            logger.info(f"\nEXPERIMENTAL SETUPS ({len(setups)} setups)")
            logger.info("-" * 40)

            for setup_name in setups:
                try:
                    setup_config = get_experimental_setup(transform_config, setup_name)
                    is_default = setup_name == transform_config.default_setup
                    default_marker = " *** DEFAULT ***" if is_default else ""

                    logger.info(f"\nSetup: '{setup_name}'{default_marker}")

                    # Display the setup's enabled state so the 'Combined total' below is
                    # always interpretable. When a setup has enabled=false in YAML, its
                    # experimental transforms exist on the container but are skipped by
                    # TransformationConfig.get_combined_transforms() (which honours
                    # ExperimentalSetup.enabled). Showing the toggle state up-front prevents
                    # the otherwise-confusing 'Experimental transforms: 3 / Combined total: 2'
                    # discrepancy that would arise if any setup is disabled in the future.
                    setup_enabled = True
                    setup_obj = None
                    if hasattr(transform_config, "experimental_setups"):
                        setup_obj = transform_config.experimental_setups.get(setup_name)
                        if setup_obj is not None and hasattr(setup_obj, "enabled"):
                            setup_enabled = bool(setup_obj.enabled)
                    state_marker = "enabled" if setup_enabled else "disabled"
                    logger.info(f"  Status: {state_marker}")

                    if setup_config and len(setup_config) > 0:
                        logger.info(f"  Experimental transforms: {len(setup_config)}")
                        transform_names = [t.get("name", "unknown") for t in setup_config]
                        logger.info(f"  Transform sequence: {' -> '.join(transform_names)}")
                    else:
                        logger.info("  Experimental transforms: 0 (uses standard transforms only)")

                    # Show combined count if standard transforms exist
                    if (
                        hasattr(transform_config, "has_standard_transforms")
                        and transform_config.has_standard_transforms()
                    ):
                        combined = get_combined_transforms_as_dicts(setup_name)
                        if setup_enabled:
                            logger.info(f"  Combined total: {len(combined)} transforms")
                        else:
                            # Setup is disabled — runtime combined excludes its experimental
                            # transforms. Show what would be combined if enabled, so the
                            # display is internally consistent with 'Experimental transforms'
                            # above and the toggle state is the only thing the reader needs
                            # to reconcile.
                            would_be_combined = len(combined) + len(setup_config or [])
                            logger.info(
                                f"  Combined total: {len(combined)} transforms "
                                f"(setup disabled — would be {would_be_combined} if enabled)"
                            )

                except Exception as e:
                    logger.error(f"  Error loading setup '{setup_name}': {e}")
        else:
            logger.info("\nNo experimental setups found in configuration")

        logger.info(f"\nDefault setup: '{transform_config.default_setup}'")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to list experimental setups: {e}")


def create_handler_for_validation(
    dataset_config: DatasetConfig,
    filter_config: FilterConfig,
    processing_config: ProcessingConfig,
    logger: logging.Logger,
) -> DatasetHandler | None:
    """
    Creates a dataset handler for configuration validation.

    Handler-Based Pattern Development ENHANCEMENT: New function to create handlers for validation.

    Args:
        dataset_config (DatasetConfig): Dataset configuration container
        filter_config (FilterConfig): Filter configuration container
        processing_config (ProcessingConfig): Processing configuration container
        logger (logging.Logger): Logger instance

    Returns:
        DatasetHandler: Created handler instance, or None if creation fails

    Raises:
        HandlerConfigurationError: If handler configuration is invalid
        HandlerNotAvailableError: If handler cannot be created
    """
    try:
        logger.debug(f"Creating {dataset_config.dataset_type} handler for validation...")

        handler = create_dataset_handler(dataset_config, filter_config, processing_config, logger)

        logger.debug(f"✅“ Handler created successfully: {type(handler).__name__}")
        return handler

    except HandlerError:
        # Re-raise handler errors as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating handler: {e}")
        raise HandlerNotAvailableError(
            message=f"Failed to create {dataset_config.dataset_type} handler",
            requested_dataset_type=dataset_config.dataset_type,
            details=f"Unexpected error: {type(e).__name__}: {str(e)}",
        ) from e


def validate_configuration(
    logger: logging.Logger,
    dataset_config: DatasetConfig | None = None,
    filter_config: FilterConfig | None = None,
    processing_config: ProcessingConfig | None = None,
    validate_handlers: bool = True,
    validate_transforms: bool = True,
    experimental_setup: str | None = None,
) -> tuple[DatasetConfig, FilterConfig, ProcessingConfig, dict[str, Any]]:
    """
    Validates the current configuration and returns key components.

    ENHANCED Experimental Setup Integration: Now includes transformation system validation.
    """
    logger.info("Validating configuration...")

    try:
        # Load full config for legacy compatibility
        config = load_config()

        # Handle configuration containers with fallback to global config
        if dataset_config is None:
            dataset_config = create_dataset_config_from_global()
            logger.debug("Using global dataset configuration fallback")

        if filter_config is None:
            filter_config = create_filter_config_from_global()
            logger.debug("Using global filter configuration fallback")

        if processing_config is None:
            processing_config = create_processing_config_from_global()
            logger.debug("Using global processing configuration fallback")

        # Validate handler availability if requested
        if validate_handlers:
            validate_handler_availability(logger)

        logger.info(f"Dataset type: {dataset_config.dataset_type}")

        # Handler-Based Pattern Development ENHANCEMENT: Validate handlers first if requested
        # Only validate if handlers are actually available
        if validate_handlers and HANDLERS_AVAILABLE:
            try:
                validate_handler_availability(logger)

                # Test handler creation with current configuration
                test_handler = create_handler_for_validation(
                    dataset_config, filter_config, processing_config, logger
                )

                logger.info(f"✅“ Handler validation successful: {type(test_handler).__name__}")

            except HandlerError as e:
                logger.error(f"Handler validation failed: {e}")

                # Check if error is recoverable
                if is_recoverable_handler_error(e):
                    logger.warning("Handler error is recoverable - continuing with legacy fallback")
                    suggestions = get_exception_recovery_suggestions(e)
                    for suggestion in suggestions[:3]:  # Show top 3 suggestions
                        logger.warning(f"Suggestion: {suggestion}")
                else:
                    logger.error("Handler error is not recoverable - aborting")
                    raise
        elif validate_handlers and not HANDLERS_AVAILABLE:
            # Don't fail if handlers aren't available - just log warning
            logger.warning("Handler validation requested but handlers not available - skipping")

        # NEW: Experimental Setup Integration - Transformation system validation
        if validate_transforms:
            try:
                validate_transformation_system(logger, experimental_setup)
                logger.info("✅“ Transformation system validation successful")

            except (TransformConfigurationError, ExperimentalSetupError) as e:
                logger.error(f"Transformation validation failed: {e}")

                # For research workflows, transformation errors are often recoverable
                logger.warning("Transformation error detected - continuing with legacy transforms")

                # Provide helpful suggestions
                if isinstance(e, ExperimentalSetupError):
                    available_setups = getattr(e, "available_setups", [])
                    if available_setups:
                        logger.info(f"Available setups: {available_setups}")

            except Exception as e:
                logger.warning(f"Transformation system validation failed: {e}")
                # Don't fail the entire validation for transform issues

        # ================================================================
        # PHASE 7: Dynamic dataset-specific configuration validation
        # ================================================================
        # Use generic validation that queries registry for features and config keys
        validate_dataset_specific_configuration(config, logger, dataset_config, processing_config)

        # Validate common configuration elements using containers
        validate_common_configuration(config, logger, filter_config, processing_config)

        logger.info("Configuration validation completed successfully")
        return dataset_config, filter_config, processing_config, config

    except HandlerError:
        # Re-raise handler errors as-is
        raise
    except (TransformConfigurationError, ExperimentalSetupError):
        # Re-raise transform errors as-is
        raise
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")

        # Enhanced error context for handler debugging
        if validate_handlers:
            error_context = create_handler_error_context(
                handler_type=dataset_config.dataset_type if dataset_config else "unknown",
                operation="configuration_validation",
                additional_context={"validate_handlers": validate_handlers},
            )
            logger.debug(f"Error context: {error_context}")

        raise


def validate_dmc_configuration(
    config: dict[str, Any], logger: logging.Logger, dataset_config: DatasetConfig | None = None
) -> None:
    """
    Validates DMC-specific configuration settings.

    REFACTORED Phase 7: Now delegates to generic uncertainty validation
    while maintaining backward compatibility for direct calls.

    Args:
        config (Dict[str, Any]): Full configuration dictionary
        logger (logging.Logger): Logger instance
        dataset_config (Optional[DatasetConfig]): Dataset configuration container

    Raises:
        ConfigurationError: If DMC configuration is invalid
        HandlerConfigurationError: If DMC handler configuration is invalid
    """
    # Handle configuration with fallback
    if dataset_config is None:
        dataset_config = create_dataset_config_from_global()
        logger.debug("Using global dataset configuration fallback for DMC validation")

    logger.info("Validating DMC-specific configuration...")

    try:
        # Check DMC dataset configuration exists
        dmc_config = config.get("dmc_config")
        if not dmc_config:
            raise ConfigurationError(
                message="DMC configuration section missing", config_key="dmc_config"
            )

        # PHASE 7: Delegate to generic uncertainty validation
        _validate_uncertainty_configuration(config, logger, dataset_config)

        # Validate property availability for DMC
        property_availability = get_property_availability()
        dmc_properties = property_availability.get(dataset_config.dataset_type, {})
        if (
            "scalar_graph_targets" in dmc_properties
            and "Etot" not in dmc_properties["scalar_graph_targets"]
        ):
            logger.warning("DMC datasets typically should include 'Etot' as a scalar target")

        logger.info("DMC configuration validated successfully")

    except HandlerConfigurationError:
        # Re-raise handler configuration errors as-is
        raise
    except Exception as e:
        logger.error(f"DMC configuration validation failed: {e}")
        raise HandlerConfigurationError(
            message="DMC configuration validation failed",
            handler_type=dataset_config.dataset_type,
            config_validation_errors=[str(e)],
            details=f"Validation error: {type(e).__name__}: {str(e)}",
        ) from e


def validate_dft_configuration(
    config: dict[str, Any],
    logger: logging.Logger,
    dataset_config: DatasetConfig | None = None,
    processing_config: ProcessingConfig | None = None,
) -> None:
    """
    Validates DFT-specific configuration settings.

    REFACTORED Phase 7: Now delegates to generic atomization validation
    while maintaining backward compatibility for direct calls.

    Args:
        config (Dict[str, Any]): Full configuration dictionary
        logger (logging.Logger): Logger instance
        dataset_config (Optional[DatasetConfig]): Dataset configuration container
        processing_config (Optional[ProcessingConfig]): Processing configuration container

    Raises:
        ConfigurationError: If DFT configuration is invalid
        HandlerConfigurationError: If DFT handler configuration is invalid
    """
    # Handle configuration with fallback
    if dataset_config is None:
        dataset_config = create_dataset_config_from_global()
        logger.debug("Using global dataset configuration fallback for DFT validation")

    if processing_config is None:
        processing_config = create_processing_config_from_global()
        logger.debug("Using global processing configuration fallback for DFT validation")

    logger.info("Validating DFT-specific configuration...")

    try:
        # Check DFT dataset configuration exists
        dft_config = config.get("dft_config")
        if not dft_config:
            raise ConfigurationError(
                message="DFT configuration section missing", config_key="dft_config"
            )

        # PHASE 7: Delegate to generic atomization validation
        _validate_atomization_configuration(config, logger, dataset_config, processing_config)

        logger.info("DFT configuration validated successfully")

    except HandlerConfigurationError:
        # Re-raise handler configuration errors as-is
        raise
    except Exception as e:
        logger.error(f"DFT configuration validation failed: {e}")
        raise HandlerConfigurationError(
            message="DFT configuration validation failed",
            handler_type=dataset_config.dataset_type,
            config_validation_errors=[str(e)],
            details=f"Validation error: {type(e).__name__}: {str(e)}",
        ) from e


def validate_dataset_specific_configuration(
    config: dict[str, Any],
    logger: logging.Logger,
    dataset_config: DatasetConfig,
    processing_config: ProcessingConfig | None = None,
) -> None:
    """
    Generic dataset-specific configuration validation using registry features.

    ADDED Phase 7: Replaces hardcoded if/elif/else chain with dynamic
    feature-based validation that works with any registered dataset type.

    Args:
        config: Full configuration dictionary
        logger: Logger instance
        dataset_config: Dataset configuration container
        processing_config: Optional processing configuration container

    Raises:
        ConfigurationError: If dataset configuration is invalid
        HandlerConfigurationError: If handler-specific configuration is invalid
    """
    dataset_type = dataset_config.dataset_type

    # ================================================================
    # Step 1: Validate dataset type is registered
    # ================================================================
    if not _is_dataset_type_registered(dataset_type):
        available = _get_available_dataset_types()
        raise ConfigurationError(
            message=f"Unknown dataset type: {dataset_type}",
            config_key="dataset_type",
            actual_value=dataset_type,
            expected_values=available,
        )

    logger.info(f"Validating {dataset_type}-specific configuration...")

    # ================================================================
    # Step 2: Validate config section exists
    # ================================================================
    config_key = _get_dataset_config_key(dataset_type)
    if config_key:
        dataset_type_config = config.get(config_key)
        if not dataset_type_config:
            raise ConfigurationError(
                message=f"{dataset_type} configuration section missing", config_key=config_key
            )
        logger.debug(f"Found configuration section: {config_key}")

    # ================================================================
    # Step 3: Feature-based validation
    # ================================================================

    # Uncertainty-enabled datasets (DMC, QMC, CCSD(T), etc.)
    if _get_dataset_feature(dataset_type, "uncertainty_handling"):
        _validate_uncertainty_configuration(config, logger, dataset_config)

    # Atomization-energy datasets (DFT, etc.)
    if _get_dataset_feature(dataset_type, "atomization_energy"):
        _validate_atomization_configuration(config, logger, dataset_config, processing_config)

    # Vibrational analysis datasets (DFT, etc.)
    if _get_dataset_feature(dataset_type, "vibrational_analysis"):
        _validate_vibrational_configuration(config, logger, dataset_config)

    # Orbital analysis datasets (Wavefunction, etc.)
    if _get_dataset_feature(dataset_type, "orbital_analysis"):
        _validate_orbital_configuration(config, logger, dataset_config)

    # ================================================================
    # Step 4: Legacy fallback for backward compatibility
    # ================================================================
    # Call legacy validation functions if they exist and match dataset type
    # This ensures existing code that calls validate_dmc_configuration() directly
    # still works correctly.

    if dataset_type == "DMC":
        # Already covered by uncertainty handling above
        pass
    elif dataset_type == "DFT":
        # Already covered by atomization/vibrational above
        pass
    elif dataset_type == "Wavefunction":
        # Already covered by orbital analysis above
        pass
    # New dataset types handled generically - no modification needed

    logger.info(f"{dataset_type} configuration validated successfully")


def _validate_uncertainty_configuration(
    config: dict[str, Any], logger: logging.Logger, dataset_config: DatasetConfig
) -> None:
    """
    Validate configuration for uncertainty-enabled datasets.

    ADDED Phase 7: Extracted from validate_dmc_configuration() for reuse.

    Args:
        config: Full configuration dictionary
        logger: Logger instance
        dataset_config: Dataset configuration container

    Raises:
        HandlerConfigurationError: If uncertainty configuration is invalid
    """
    if not dataset_config.is_uncertainty_enabled:
        logger.info(f"{dataset_config.dataset_type} uncertainty handling is DISABLED")
        return

    uncertainty_config = dataset_config.uncertainty_config
    if uncertainty_config:
        logger.info(f"{dataset_config.dataset_type} uncertainty handling is ENABLED")
        logger.info(
            f"  - Uncertainty field: {uncertainty_config.get('uncertainty_field_name', 'std')}"
        )
        logger.info(
            f"  - Use for loss weighting: {uncertainty_config.get('use_for_loss_weighting', False)}"
        )
        logger.info(
            f"  - Weighting strategy: {uncertainty_config.get('uncertainty_weighting', 'inverse_variance')}"
        )

        max_uncertainty = uncertainty_config.get("max_uncertainty_threshold")
        if max_uncertainty is not None:
            logger.info(f"  - Maximum uncertainty threshold: {max_uncertainty}")

            if max_uncertainty <= 0:
                raise HandlerConfigurationError(
                    message="Invalid uncertainty threshold",
                    handler_type=dataset_config.dataset_type,
                    config_validation_errors=[
                        f"max_uncertainty_threshold must be > 0, got {max_uncertainty}"
                    ],
                    invalid_config_keys=["max_uncertainty_threshold"],
                )


def _validate_atomization_configuration(
    config: dict[str, Any],
    logger: logging.Logger,
    dataset_config: DatasetConfig,
    processing_config: ProcessingConfig | None = None,
) -> None:
    """
    Validate configuration for atomization-energy-enabled datasets.

    ADDED Phase 7: Extracted from validate_dft_configuration() for reuse.

    Args:
        config: Full configuration dictionary
        logger: Logger instance
        dataset_config: Dataset configuration container
        processing_config: Optional processing configuration container

    Raises:
        HandlerConfigurationError: If atomization configuration is invalid
    """
    if processing_config is None:
        processing_config = create_processing_config_from_global()

    atomization_base = processing_config.calculate_atomization_energy_from
    atomization_key = processing_config.atomization_energy_key_name

    if atomization_base and atomization_key:
        logger.info(f"Atomization energy calculation: {atomization_base} -> {atomization_key}")

        # Validate atomic energies are available
        if not ATOMIC_ENERGIES_HARTREE:
            raise HandlerConfigurationError(
                message="Atomic energies required for atomization energy calculation but not found",
                handler_type=dataset_config.dataset_type,
                config_validation_errors=["Missing atomic_energies_hartree"],
                invalid_config_keys=["atomic_energies_hartree"],
            )

        # DYNAMIC VALIDATION: Validate atomization_base against the dataset's
        # scalar_graph_targets from property_availability in config.yaml.
        # This ensures validation automatically supports new datasets without code changes.
        property_availability = get_property_availability()
        dataset_properties = property_availability.get(dataset_config.dataset_type, {})
        valid_scalar_targets = dataset_properties.get("scalar_graph_targets", [])

        if valid_scalar_targets and atomization_base not in valid_scalar_targets:
            logger.warning(
                f"Atomization energy base property '{atomization_base}' is not in "
                f"{dataset_config.dataset_type}'s scalar_graph_targets: {valid_scalar_targets}"
            )
    else:
        logger.info("Atomization energy calculation: DISABLED")


def _validate_vibrational_configuration(
    config: dict[str, Any], logger: logging.Logger, dataset_config: DatasetConfig
) -> None:
    """
    Validate configuration for vibrational-analysis-enabled datasets.

    ADDED Phase 7: Feature-based validation for vibrational analysis.

    Args:
        config: Full configuration dictionary
        logger: Logger instance
        dataset_config: Dataset configuration container
    """
    logger.debug(f"{dataset_config.dataset_type} supports vibrational analysis")
    # Add specific vibrational configuration checks if needed


def _validate_orbital_configuration(
    config: dict[str, Any], logger: logging.Logger, dataset_config: DatasetConfig
) -> None:
    """
    Validate configuration for orbital-analysis-enabled datasets.

    ADDED Phase 7: Feature-based validation for orbital/wavefunction analysis.

    Args:
        config: Full configuration dictionary
        logger: Logger instance
        dataset_config: Dataset configuration container
    """
    logger.debug(f"{dataset_config.dataset_type} supports orbital analysis")
    # Wavefunction-specific: Check for feature_tier if applicable
    config_key = _get_dataset_config_key(dataset_config.dataset_type)
    if config_key:
        dataset_type_config = config.get(config_key, {})
        if "processing_config" in dataset_type_config:
            feature_tier = dataset_type_config["processing_config"].get("feature_tier", "standard")
            logger.info(f"  - Feature tier: {feature_tier}")


def inspect_transform_object(transform: Any, logger: logging.Logger) -> dict[str, Any]:
    """
    Inspect a transform object to extract useful information.

    Args:
        transform: Transform object to inspect
        logger: Logger instance

    Returns:
        Dictionary with transform information
    """
    info = {"type": type(transform).__name__, "valid": False, "name": None, "format": "unknown"}

    try:
        # Dict-based specification (proper format)
        if isinstance(transform, dict):
            info["format"] = "dict"
            info["name"] = transform.get("name", "Unknown")
            info["valid"] = "name" in transform
            info["has_kwargs"] = "kwargs" in transform
            info["has_enabled"] = "enabled" in transform

        # PyG transform object
        elif hasattr(transform, "__class__"):
            info["format"] = "pyg_object"
            info["name"] = transform.__class__.__name__
            info["valid"] = True
            info["module"] = transform.__class__.__module__

        # String reference
        elif isinstance(transform, str):
            info["format"] = "string"
            info["name"] = transform
            info["valid"] = len(transform) > 0

        else:
            logger.warning(f"Unknown transform format: {type(transform)}")

    except Exception as e:
        logger.error(f"Error inspecting transform: {e}")
        info["error"] = str(e)

    return info


def validate_common_configuration(
    config: dict[str, Any],
    logger: logging.Logger,
    filter_config: FilterConfig | None = None,
    processing_config: ProcessingConfig | None = None,
) -> None:
    """
    Validates common configuration settings for both DFT and DMC datasets.

    ENHANCED Handler-Based Pattern Development: Added handler-aware validation considerations.
    Handle both string and dictionary transform formats for robustness.
    ENHANCED Experimental Setup Integration: Properly handle enhanced transformation format.

    Args:
        config (Dict[str, Any]): Full configuration dictionary
        logger (logging.Logger): Logger instance
        filter_config (Optional[FilterConfig]): Filter configuration container
        processing_config (Optional[ProcessingConfig]): Processing configuration container
    """
    # Handle configuration with fallback
    if filter_config is None:
        filter_config = create_filter_config_from_global()
        logger.debug("Using global filter configuration fallback for common validation")

    if processing_config is None:
        processing_config = create_processing_config_from_global()
        logger.debug("Using global processing configuration fallback for common validation")

    logger.info("Validating common configuration...")

    # Validate filter configuration using container
    if filter_config:
        logger.info("Filters configured:")
        if filter_config.max_atoms is not None:
            logger.info(f"  - max_atoms: {filter_config.max_atoms}")
        if filter_config.min_atoms is not None:
            logger.info(f"  - min_atoms: {filter_config.min_atoms}")
        if filter_config.heavy_atom_filter is not None:
            logger.info(f"  - heavy_atom_filter: {filter_config.heavy_atom_filter}")
        if filter_config.dmc_uncertainty_filter is not None:
            logger.info(f"  - dmc_uncertainty_filter: {filter_config.dmc_uncertainty_filter}")
    else:
        logger.info("No filters configured")

    # Validate transformations with enhanced error handling
    transformations = config.get("transformations", [])

    # Handle enhanced format (dict with experimental_setups and/or standard_transforms)
    if isinstance(transformations, dict) and (
        "experimental_setups" in transformations or "standard_transforms" in transformations
    ):
        logger.info("Enhanced transformation format detected")

        # Log standard transforms if present
        standard_transforms = transformations.get("standard_transforms", [])
        if standard_transforms:
            logger.info(f"Standard transforms configured: {len(standard_transforms)}")

        # Log experimental setups
        setups = transformations.get("experimental_setups", {})
        if setups:
            logger.info(f"Experimental setups configured: {len(setups)}")

        default_setup = transformations.get("default_setup")
        if default_setup:
            logger.info(f"Default setup: '{default_setup}'")

        # Log setup details
        for setup_name, setup_config in setups.items():
            # Handle both dict with 'transforms' key and direct list
            if isinstance(setup_config, dict):
                setup_transforms = setup_config.get("transforms", [])
            elif isinstance(setup_config, list):
                setup_transforms = setup_config
            else:
                setup_transforms = []
            logger.info(f"  Setup '{setup_name}': {len(setup_transforms)} experimental transforms")

    elif isinstance(transformations, list) and transformations:
        # Legacy format (list of transforms) - Better handling of transform objects
        logger.info(f"PyG transformations configured: {len(transformations)}")

        valid_transforms = 0
        for i, transform in enumerate(transformations):
            try:
                # Handle dict-based transform specs (proper format)
                if isinstance(transform, dict):
                    transform_name = transform.get("name", "Unknown")
                    logger.info(f"  - {i + 1}. {transform_name}")
                    valid_transforms += 1

                # Handle PyG transform objects directly (legacy compatibility)
                elif hasattr(transform, "__class__") and hasattr(transform.__class__, "__name__"):
                    transform_name = transform.__class__.__name__
                    logger.info(f"  - {i + 1}. {transform_name} (PyG object)")
                    valid_transforms += 1

                # Handle string-based transform names (rare legacy case)
                elif isinstance(transform, str):
                    logger.info(f"  - {i + 1}. {transform} (string reference)")
                    valid_transforms += 1

                else:
                    # Unknown format - warn but don't skip
                    logger.warning(
                        f"Transform {i + 1}: Unrecognized format {type(transform).__name__}"
                    )
                    logger.debug(f"Transform data: {transform}")

            except Exception as e:
                logger.error(f"Error processing transform {i + 1}: {e}")
                logger.debug(f"Transform data: {transform} (type: {type(transform)})")

        if valid_transforms == 0:
            logger.warning("No valid transforms found in configuration")
        else:
            logger.info(f"Total valid transforms: {valid_transforms}/{len(transformations)}")

    else:
        logger.info("No PyG transformations configured")

    # Validate test molecule limit using container
    test_limit = processing_config.test_molecule_limit if processing_config else None
    if test_limit:
        logger.warning(f"*** TEST MODE: Processing limited to {test_limit} molecules ***")


def print_dataset_info(
    logger: logging.Logger,
    dataset_config: DatasetConfig | None = None,
    processing_config: ProcessingConfig | None = None,
    experimental_setup: str | None = None,
) -> None:
    """
    Prints comprehensive information about the dataset configuration.

    ENHANCED Experimental Setup Integration: Added experimental setup and transformation system information.

    Args:
        logger (logging.Logger): Logger instance
        dataset_config (Optional[DatasetConfig]): Dataset configuration container
        processing_config (Optional[ProcessingConfig]): Processing configuration container
        experimental_setup (Optional[str]): Current experimental setup
    """
    # Handle configuration with fallback
    if dataset_config is None:
        dataset_config = create_dataset_config_from_global()
        logger.debug("Using global dataset configuration fallback for info display")

    if processing_config is None:
        processing_config = create_processing_config_from_global()
        logger.debug("Using global processing configuration fallback for info display")

    logger.info("=" * 60)
    logger.info(f"DATASET INFORMATION - {dataset_config.dataset_type}")
    logger.info("=" * 60)

    # Handler-Based Pattern Development ENHANCEMENT: Handler pattern status
    logger.info(
        f"Handler pattern support: {'✅“ Available' if HANDLERS_AVAILABLE else 'âœ— Not Available'}"
    )
    if HANDLERS_AVAILABLE:
        try:
            from milia_pipeline.handlers import get_available_handlers

            # Dynamically discover available handler types via registry
            available_types = get_available_handlers()

            if available_types:
                logger.info(f"Available handlers: {', '.join(available_types)}")
            else:
                logger.info("Available handlers: Unable to determine (no handler classes found)")
        except Exception as e:
            logger.debug(f"Could not determine available handlers: {e}")
            logger.info("Available handlers: Unable to determine")

    # Experimental Setup Integration - Transformation system status
    logger.info(
        f"Transformation system: {'✅ Available' if GRAPH_TRANSFORMS_AVAILABLE else '❌ Not Available'}"
    )

    # Plugin System Integration - Plugin system status
    logger.info(
        f"Plugin system: {'✅ Available' if PLUGIN_SYSTEM_AVAILABLE else '❌ Not Available'}"
    )

    if PLUGIN_SYSTEM_AVAILABLE:
        try:
            # Check if plugins enabled
            plugins_enabled = get_config_value("plugins.enabled", default=False)

            if plugins_enabled:
                # Get plugin counts
                all_plugins = PluginRegistry.list_plugins()
                enabled_plugins = [p for p in all_plugins if PluginRegistry.is_plugin_enabled(p)]

                logger.info(f"Plugins discovered: {len(all_plugins)}")
                logger.info(f"Plugins enabled: {len(enabled_plugins)}")

                if enabled_plugins:
                    logger.info(
                        f"Active plugins: {', '.join(enabled_plugins[:3])}"
                        + (
                            f" (+{len(enabled_plugins) - 3} more)"
                            if len(enabled_plugins) > 3
                            else ""
                        )
                    )
            else:
                logger.info("Plugins: Disabled in configuration")

        except Exception as e:
            logger.debug(f"Could not retrieve plugin info: {e}")
    else:
        logger.info(f"Plugin system error: {PLUGIN_SYSTEM_IMPORT_ERROR}")

    if GRAPH_TRANSFORMS_AVAILABLE:
        try:
            from milia_pipeline.transformations.graph_transforms import (
                CUSTOM_TRANSFORMS_AVAILABLE,
                get_custom_transform_count,
            )

            if CUSTOM_TRANSFORMS_AVAILABLE:
                custom_count = get_custom_transform_count()
                if custom_count > 0:
                    logger.info(f"Custom transforms: ✅ Available ({custom_count} registered)")
                else:
                    logger.info("Custom transforms: ⚠️  Available but none registered")
            else:
                logger.info("Custom transforms: ⚠️  Module not available")

        except Exception as e:
            logger.debug(f"Could not retrieve custom transform info: {e}")

    if GRAPH_TRANSFORMS_AVAILABLE:
        try:
            gt = get_graph_transforms()

            # Handle get_available_transforms() return value safely
            try:
                available_transforms = gt.get_available_transforms()

                # Handle different return formats
                if isinstance(available_transforms, int):
                    # Direct integer count (actual production case)
                    total_transforms = available_transforms
                    logger.info(f"Available transforms: {total_transforms}")

                elif isinstance(available_transforms, dict):
                    # Dict format with categories
                    total_transforms = available_transforms.get("count", 0)
                    if total_transforms == 0:
                        total_transforms = sum(
                            len(v) for k, v in available_transforms.items() if isinstance(v, list)
                        )
                    num_categories = len(
                        [k for k, v in available_transforms.items() if isinstance(v, list)]
                    )
                    logger.info(
                        f"Available transforms: {total_transforms} across {num_categories} categories"
                    )

                elif isinstance(available_transforms, (list, tuple)):
                    # List/tuple format
                    total_transforms = len(available_transforms)
                    logger.info(f"Available transforms: {total_transforms}")

                else:
                    logger.info(
                        f"Available transforms: Unable to count (type: {type(available_transforms).__name__})"
                    )

            except AttributeError:
                logger.info("Available transforms: Method not available")
            except Exception as e:
                logger.info(f"Available transforms: Error - {e}")

            # Display transformation configuration information
            transform_config = get_transformation_config()

            # Show standard transforms info
            if (
                hasattr(transform_config, "has_standard_transforms")
                and transform_config.has_standard_transforms()
            ):
                standard_count = len(transform_config.get_standard_transforms())
                logger.info(f"Standard transforms: {standard_count} configured")

            # Show experimental setups info
            available_setups = (
                list(transform_config.experimental_setups.keys())
                if hasattr(transform_config, "experimental_setups")
                else []
            )
            logger.info(f"Experimental setups: {len(available_setups)} available")

            if experimental_setup:
                logger.info(f"Current experimental setup: '{experimental_setup}'")
                try:
                    # Show combined transforms (standard + experimental)
                    combined_config = get_combined_transforms_as_dicts(experimental_setup)
                    setup_config = get_experimental_setup(transform_config, experimental_setup)
                    experimental_count = len(setup_config) if setup_config else 0
                    logger.info(f"  - Experimental transforms: {experimental_count}")
                    logger.info(
                        f"  - Combined transforms (standard + experimental): {len(combined_config)}"
                    )
                    transform_names = [t.get("name", "unknown") for t in combined_config]
                    logger.info(f"  - Transform sequence: {' -> '.join(transform_names)}")
                except Exception as e:
                    logger.warning(f"  - Error loading setup details: {e}")
            else:
                default_setup = transform_config.default_setup
                logger.info(f"Using default experimental setup: '{default_setup}'")

        except Exception as e:
            logger.info(f"Transformation system: Error getting details - {e}")
    else:
        logger.info(f"Transformation system error: {GRAPH_TRANSFORMS_IMPORT_ERROR}")

    # Dataset constants
    raw_npz_filename, raw_npz_download_url, dataset_root_dir = get_dataset_constants()
    logger.info(f"Raw NPZ filename: {raw_npz_filename}")
    logger.info(f"Download URL: {raw_npz_download_url or 'Not specified'}")
    logger.info(f"Dataset root directory: {dataset_root_dir}")

    # Property configuration using container
    logger.info("\nConfigured Properties:")

    scalar_targets = processing_config.scalar_graph_targets
    if scalar_targets:
        logger.info(f"  Scalar graph targets ({len(scalar_targets)}): {', '.join(scalar_targets)}")

    node_features = processing_config.node_features
    if node_features:
        logger.info(f"  Node features ({len(node_features)}): {', '.join(node_features)}")

    vector_props = processing_config.vector_graph_properties
    if vector_props:
        logger.info(f"  Vector properties ({len(vector_props)}): {', '.join(vector_props)}")

    variable_props = processing_config.variable_len_graph_properties
    if variable_props:
        logger.info(
            f"  Variable-length properties ({len(variable_props)}): {', '.join(variable_props)}"
        )

    # ================================================================
    # PHASE 7: Feature-based uncertainty display
    # ================================================================
    # Display uncertainty info for ANY dataset with uncertainty_handling feature
    if (
        _get_dataset_feature(dataset_config.dataset_type, "uncertainty_handling")
        and dataset_config.is_uncertainty_enabled
    ):
        uncertainty_config = dataset_config.uncertainty_config
        logger.info(f"\n{dataset_config.dataset_type} Uncertainty Configuration:")
        logger.info(f"  Field name: {uncertainty_config.get('uncertainty_field_name', 'std')}")
        logger.info(f"  Loss weighting: {uncertainty_config.get('use_for_loss_weighting', False)}")
        logger.info(
            f"  Weighting strategy: {uncertainty_config.get('uncertainty_weighting', 'inverse_variance')}"
        )

        max_uncertainty = uncertainty_config.get("max_uncertainty_threshold")
        if max_uncertainty is not None:
            logger.info(f"  Max uncertainty threshold: {max_uncertainty}")

    logger.info("=" * 60)


def analyze_dataset_statistics(
    dataset: miliaDataset, logger: logging.Logger, dataset_config: DatasetConfig | None = None
) -> dict[str, Any]:
    """
    Analyzes and reports statistics about the processed dataset.

    ENHANCED Handler-Based Pattern Development: Added handler-aware statistics collection.

    Args:
        dataset (miliaDataset): Processed dataset
        logger (logging.Logger): Logger instance
        dataset_config (Optional[DatasetConfig]): Dataset configuration container

    Returns:
        Dict[str, Any]: Dictionary containing dataset statistics
    """
    # Handle configuration with fallback
    if dataset_config is None:
        dataset_config = create_dataset_config_from_global()
        logger.debug("Using global dataset configuration fallback for statistics")

    logger.info("Analyzing dataset statistics...")

    if len(dataset) == 0:
        logger.warning("Dataset is empty - no statistics to compute")
        return {}

    stats = {
        "total_molecules": len(dataset),
        "dataset_type": dataset_config.dataset_type,
        "handler_pattern_used": HANDLERS_AVAILABLE,  # Handler-Based Pattern Development ENHANCEMENT
        "transformation_system_used": GRAPH_TRANSFORMS_AVAILABLE,  # NEW: Experimental Setup Integration
    }

    # Handler-Based Pattern Development ENHANCEMENT: Try to get handler statistics if available
    try:
        if hasattr(dataset, "get_handler_statistics"):
            handler_stats = dataset.get_handler_statistics()
            if handler_stats:
                stats["handler_statistics"] = handler_stats
                logger.info(f"Handler statistics available: {list(handler_stats.keys())}")
    except Exception as e:
        logger.debug(f"Could not retrieve handler statistics: {e}")

    # NEW: Experimental Setup Integration - Try to get transformation statistics if available
    try:
        if hasattr(dataset, "get_transform_info"):
            transform_info = dataset.get_transform_info()
            if transform_info:
                stats["transformation_statistics"] = transform_info
                logger.info(
                    f"Transformation statistics available: transform_count={transform_info.get('transform_count', 0)}"
                )
    except Exception as e:
        logger.debug(f"Could not retrieve transformation statistics: {e}")

    # Sample first few molecules to analyze structure
    sample_size = min(10, len(dataset))
    logger.info(f"Analyzing sample of {sample_size} molecules for statistics...")

    node_counts = []
    y_dimensions = []
    x_dimensions = []
    has_uncertainty = 0
    uncertainty_values = []

    for i in range(sample_size):
        try:
            data = dataset[i]

            # Node count statistics
            if hasattr(data, "num_nodes"):
                node_counts.append(data.num_nodes)

            # Target dimensions
            if hasattr(data, "y") and data.y is not None:
                if data.y.dim() == 0:  # Scalar
                    y_dimensions.append(1)
                else:
                    y_dimensions.append(data.y.shape[0])

            # Feature dimensions
            if hasattr(data, "x") and data.x is not None:
                x_dimensions.append(data.x.shape[1])

            # ================================================================
            # PHASE 7: Feature-based uncertainty statistics
            # ================================================================
            # Collect uncertainty for ANY dataset with uncertainty_handling feature
            if (
                _get_dataset_feature(dataset_config.dataset_type, "uncertainty_handling")
                and hasattr(data, "uncertainty")
                and data.uncertainty is not None
            ):
                has_uncertainty += 1
                uncertainty_values.append(data.uncertainty.item())

        except Exception as e:
            logger.warning(f"Error analyzing molecule {i}: {e}")

    # Compute statistics
    if node_counts:
        stats["avg_nodes"] = np.mean(node_counts)
        stats["min_nodes"] = np.min(node_counts)
        stats["max_nodes"] = np.max(node_counts)

    if y_dimensions:
        stats["y_dim"] = (
            y_dimensions[0] if len(set(y_dimensions)) == 1 else f"Variable: {set(y_dimensions)}"
        )

    if x_dimensions:
        stats["x_dim"] = (
            x_dimensions[0] if len(set(x_dimensions)) == 1 else f"Variable: {set(x_dimensions)}"
        )

    if uncertainty_values:
        stats["uncertainty_stats"] = {
            "count": len(uncertainty_values),
            "mean": np.mean(uncertainty_values),
            "std": np.std(uncertainty_values),
            "min": np.min(uncertainty_values),
            "max": np.max(uncertainty_values),
        }

    # Report statistics
    logger.info("=" * 50)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 50)
    logger.info(f"Total molecules: {stats['total_molecules']}")
    logger.info(f"Dataset type: {stats['dataset_type']}")
    logger.info(
        f"Handler pattern: {'✅“ Used' if stats['handler_pattern_used'] else 'âœ— Legacy fallback'}"
    )
    logger.info(
        f"Transformation system: {'✅“ Used' if stats['transformation_system_used'] else 'âœ— Legacy fallback'}"
    )

    if "avg_nodes" in stats:
        logger.info(f"Average nodes per molecule: {stats['avg_nodes']:.1f}")
        logger.info(f"Node count range: {stats['min_nodes']} - {stats['max_nodes']}")

    if "y_dim" in stats:
        logger.info(f"Target tensor dimension: {stats['y_dim']}")

    if "x_dim" in stats:
        logger.info(f"Feature tensor dimension: {stats['x_dim']}")

    if "uncertainty_stats" in stats:
        unc_stats = stats["uncertainty_stats"]
        logger.info(f"DMC Uncertainty statistics ({unc_stats['count']} molecules):")
        logger.info(f"  Mean: {unc_stats['mean']:.6f}")
        logger.info(f"  Std:  {unc_stats['std']:.6f}")
        logger.info(f"  Range: {unc_stats['min']:.6f} - {unc_stats['max']:.6f}")

    # Handler-Based Pattern Development ENHANCEMENT: Report handler statistics if available
    if "handler_statistics" in stats:
        handler_stats = stats["handler_statistics"]
        logger.info("\nHandler Statistics:")
        for key, value in handler_stats.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    logger.info(f"    {sub_key}: {sub_value}")
            else:
                logger.info(f"  {key}: {value}")

    # NEW: Experimental Setup Integration - Report transformation statistics if available
    if "transformation_statistics" in stats:
        transform_stats = stats["transformation_statistics"]
        logger.info("\nTransformation Statistics:")

        # Core statistics
        logger.info(
            f"  Current experimental setup: {transform_stats.get('current_experimental_setup', 'None')}"
        )
        logger.info(f"  Transform count: {transform_stats.get('transform_count', 0)}")
        logger.info(
            f"  Transform pipeline active: {transform_stats.get('transform_pipeline_active', False)}"
        )

        # Transform details if available
        if "transforms" in transform_stats:
            logger.info("  Transform sequence:")
            for i, transform in enumerate(transform_stats["transforms"], 1):
                logger.info(f"    {i}. {transform.get('name', 'Unknown')}")

    logger.info("=" * 50)

    return stats


def dataset_access_test(
    dataset: miliaDataset,
    logger: logging.Logger,
    num_samples: int = 3,
    dataset_config: DatasetConfig | None = None,
) -> bool:
    """
    Tests dataset access and validates data integrity.

    ENHANCED Handler-Based Pattern Development: Added handler-aware testing and enhanced error handling.

    Args:
        dataset (miliaDataset): Dataset to test
        logger (logging.Logger): Logger instance
        num_samples (int): Number of samples to test
        dataset_config (Optional[DatasetConfig]): Dataset configuration container

    Returns:
        bool: True if all tests pass, False otherwise
    """
    # Handle configuration with fallback
    if dataset_config is None:
        dataset_config = create_dataset_config_from_global()
        logger.debug("Using global dataset configuration fallback for access testing")

    logger.info(f"Testing dataset access with {num_samples} samples...")

    if len(dataset) == 0:
        logger.error("Dataset is empty - cannot test access")
        return False

    num_samples = min(num_samples, len(dataset))

    try:
        for i in range(num_samples):
            data = dataset[i]

            # Basic structure validation
            if not hasattr(data, "z") or data.z is None:
                logger.error(f"Sample {i}: Missing atomic numbers (z)")
                return False

            if not hasattr(data, "pos") or data.pos is None:
                logger.error(f"Sample {i}: Missing positions (pos)")
                return False

            if not hasattr(data, "edge_index") or data.edge_index is None:
                logger.error(f"Sample {i}: Missing edge indices (edge_index)")
                return False

            # Dataset-specific validation using container
            if (
                dataset_config.dataset_type == "DMC"
                and dataset_config.is_uncertainty_enabled
                and not hasattr(data, "uncertainty")
            ):
                logger.warning(f"Sample {i}: Missing uncertainty data for DMC dataset")

            # Handler-Based Pattern Development ENHANCEMENT: Check for handler-generated attributes
            handler_attributes = ["handler_processed", "dataset_handler_type"]
            for attr in handler_attributes:
                if hasattr(data, attr):
                    logger.debug(
                        f"Sample {i}: Handler attribute '{attr}' found: {getattr(data, attr)}"
                    )

            logger.debug(
                f"Sample {i}: ✅“ Valid (nodes: {data.num_nodes}, edges: {data.edge_index.shape[1]})"
            )

        logger.info("✅“ Dataset access tests passed")
        return True

    except Exception as e:
        logger.error(f"Dataset access test failed: {e}")

        # Handler-Based Pattern Development ENHANCEMENT: Enhanced error diagnosis
        if isinstance(e, HandlerError):
            error_summary = format_handler_exception_summary(e)
            logger.error(f"Handler error details: {error_summary}")

        return False


def run_quick_validation(
    root_dir: str,
    logger: logging.Logger,
    dataset_config: DatasetConfig | None = None,
    filter_config: FilterConfig | None = None,
    processing_config: ProcessingConfig | None = None,
    experimental_setup: str | None = None,
) -> bool:
    """
    Runs a quick validation of existing processed data without full reprocessing.

    ENHANCED Experimental Setup Integration: Added experimental setup support and transformation validation.

    Args:
        root_dir (str): Root directory for dataset
        logger (logging.Logger): Logger instance
        dataset_config (Optional[DatasetConfig]): Dataset configuration container
        filter_config (Optional[FilterConfig]): Filter configuration container
        processing_config (Optional[ProcessingConfig]): Processing configuration container
        experimental_setup (Optional[str]): Experimental setup to validate

    Returns:
        bool: True if validation passes, False otherwise
    """
    logger.info("Running quick validation mode...")

    try:
        # NEW: Experimental Setup Integration - Validate experimental setup if provided
        if experimental_setup and GRAPH_TRANSFORMS_AVAILABLE:
            try:
                validate_transformation_system(logger, experimental_setup)
                logger.info(f"✅“ Experimental setup '{experimental_setup}' validated")
            except (TransformConfigurationError, ExperimentalSetupError) as e:
                logger.warning(f"Experimental setup validation failed: {e}")
                # Continue with validation but note the issue

        # Try to load existing dataset using containers if provided
        if (
            dataset_config is not None
            and filter_config is not None
            and processing_config is not None
        ):
            dataset = miliaDataset.create_with_containers(
                root=root_dir,
                logger=logger,
                dataset_config=dataset_config,
                filter_config=filter_config,
                processing_config=processing_config,
                chunk_size=1000,  # Smaller chunk size for validation
                force_reload=False,  # Don't force reload
                experimental_setup=experimental_setup,  # NEW: Pass experimental setup
            )
        else:
            # Fallback to regular constructor
            dataset = miliaDataset(
                root=root_dir,
                logger=logger,
                chunk_size=1000,  # Smaller chunk size for validation
                force_reload=False,  # Don't force reload
                experimental_setup=experimental_setup,  # NEW: Pass experimental setup
            )

        if len(dataset) == 0:
            logger.warning("No existing processed data found or dataset is empty")
            return False

        logger.info(f"Found existing processed dataset with {len(dataset)} molecules")

        # Run basic tests with container
        if not dataset_access_test(dataset, logger, num_samples=5, dataset_config=dataset_config):
            return False

        # Generate statistics with container
        analyze_dataset_statistics(dataset, logger, dataset_config=dataset_config)

        logger.info("✅“ Quick validation completed successfully")
        return True

    except HandlerError as e:
        logger.error(f"Handler error during quick validation: {e}")

        # Provide recovery suggestions
        suggestions = get_exception_recovery_suggestions(e)
        logger.info("Recovery suggestions:")
        for i, suggestion in enumerate(suggestions[:3], 1):
            logger.info(f"  {i}. {suggestion}")

        return False

    except (TransformConfigurationError, ExperimentalSetupError) as e:
        logger.error(f"Transformation error during quick validation: {e}")
        return False

    except Exception as e:
        logger.error(f"Quick validation failed: {e}")
        return False


def create_dataset_with_error_handling(
    root_dir: str,
    logger: logging.Logger,
    dataset_config: DatasetConfig,
    filter_config: FilterConfig,
    processing_config: ProcessingConfig,
    chunk_size: int,
    force_reload: bool,
    experimental_setup: str | None = None,
) -> miliaDataset | None:
    """
    Creates dataset with comprehensive error handling and recovery.

    ENHANCED Experimental Setup Integration: Added experimental setup support and transformation error handling.

    Args:
        root_dir (str): Root directory for dataset
        logger (logging.Logger): Logger instance
        dataset_config (DatasetConfig): Dataset configuration container
        filter_config (FilterConfig): Filter configuration container
        processing_config (ProcessingConfig): Processing configuration container
        chunk_size (int): Chunk size for processing
        force_reload (bool): Whether to force reload
        experimental_setup (Optional[str]): Experimental setup to use

    Returns:
        miliaDataset: Created dataset, or None if creation failed

    Raises:
        HandlerError: If handler errors are not recoverable
        TransformConfigurationError: If transform errors are not recoverable
        BaseProjectError: If dataset creation fails completely
    """
    logger.info("Creating dataset with enhanced error handling...")

    # NEW: Experimental Setup Integration - Validate experimental setup before dataset creation
    if experimental_setup and GRAPH_TRANSFORMS_AVAILABLE:
        try:
            validate_transformation_system(logger, experimental_setup)
            logger.info(f"✅“ Experimental setup '{experimental_setup}' pre-validated")
        except (TransformConfigurationError, ExperimentalSetupError) as e:
            logger.warning(f"Experimental setup validation failed: {e}")

            # Check if we should continue or fail
            if isinstance(e, ExperimentalSetupError):
                # Try to fall back to default setup
                try:
                    transform_config = get_transformation_config()
                    default_setup = transform_config.default_setup
                    logger.warning(f"Falling back to default experimental setup: '{default_setup}'")
                    experimental_setup = default_setup
                    validate_transformation_system(logger, experimental_setup)
                except Exception:
                    logger.warning("Disabling experimental setup due to validation failures")
                    experimental_setup = None

    try:
        # Primary attempt: Use containers with handler pattern and experimental setup
        dataset = miliaDataset.create_with_containers(
            root=root_dir,
            logger=logger,
            dataset_config=dataset_config,
            filter_config=filter_config,
            processing_config=processing_config,
            chunk_size=chunk_size,
            force_reload=force_reload,
            experimental_setup=experimental_setup,  # NEW: Pass experimental setup
        )

        # Check if dataset actually contains data before declaring success
        if len(dataset) > 0:
            setup_info = (
                f" with experimental setup '{experimental_setup}'" if experimental_setup else ""
            )
            logger.info(
                f"✅“ Dataset created successfully with handler pattern{setup_info} ({len(dataset)} molecules)"
            )
            return dataset
        else:
            # Dataset creation succeeded but no data - this should trigger fallback
            logger.warning("Dataset created with handler pattern but is empty (0 molecules)")
            raise HandlerOperationError(
                message="Dataset creation succeeded but resulted in empty dataset",
                handler_type=dataset_config.dataset_type,
                operation="dataset_creation",
            )

    except HandlerError as e:
        logger.error(f"Handler error during dataset creation: {e}")

        # Check if error is recoverable
        if is_recoverable_handler_error(e):
            logger.warning("Attempting recovery with fallback approach...")

            # Get recovery suggestions
            suggestions = get_exception_recovery_suggestions(e)
            for suggestion in suggestions[:2]:  # Try top 2 suggestions
                logger.info(f"Recovery suggestion: {suggestion}")

            try:
                # Fallback: Try without handlers
                logger.warning("Falling back to legacy dataset creation...")
                dataset = miliaDataset(
                    root=root_dir,
                    logger=logger,
                    chunk_size=chunk_size,
                    force_reload=force_reload,
                    experimental_setup=experimental_setup,  # NEW: Still try experimental setup
                )

                if len(dataset) > 0:
                    logger.warning(f"✅“ Fallback successful ({len(dataset)} molecules)")
                    return dataset
                else:
                    logger.error("Fallback created empty dataset")
                    return None

            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise
        else:
            logger.error("Handler error is not recoverable")
            raise

    except (TransformConfigurationError, ExperimentalSetupError) as e:
        logger.error(f"Transformation error during dataset creation: {e}")

        # Try to create dataset without experimental setup
        if experimental_setup:
            logger.warning("Attempting dataset creation without experimental setup...")
            try:
                dataset = miliaDataset.create_with_containers(
                    root=root_dir,
                    logger=logger,
                    dataset_config=dataset_config,
                    filter_config=filter_config,
                    processing_config=processing_config,
                    chunk_size=chunk_size,
                    force_reload=force_reload,
                    experimental_setup=None,  # Disable experimental setup
                )

                if len(dataset) > 0:
                    logger.warning(
                        f"✅“ Dataset created without experimental setup ({len(dataset)} molecules)"
                    )
                    return dataset
                else:
                    logger.error("Dataset creation without experimental setup also failed")
                    return None

            except Exception as no_setup_error:
                logger.error(
                    f"Dataset creation without experimental setup failed: {no_setup_error}"
                )
                raise
        else:
            # Original error was not related to experimental setup
            raise

    except Exception as e:
        logger.error(f"Unexpected error during dataset creation: {e}")

        # Log error context for debugging
        error_context = create_handler_error_context(
            handler_type=dataset_config.dataset_type,
            operation="dataset_creation",
            additional_context={
                "chunk_size": chunk_size,
                "force_reload": force_reload,
                "handlers_available": HANDLERS_AVAILABLE,
                "transforms_available": GRAPH_TRANSFORMS_AVAILABLE,
                "experimental_setup": experimental_setup,
            },
        )
        logger.debug(f"Error context: {error_context}")

        raise


# TRAINING SYSTEM HANDLERS


def handle_training_mode(
    args: argparse.Namespace,
    logger: logging.Logger,
    dataset: "miliaDataset",
    config: dict[str, Any],
) -> int:
    """
    Handle model training mode execution.

    Phase 9: Complete training workflow with optional HPO.

    Args:
        args: CLI arguments
        logger: Logger instance
        dataset: Loaded miliaDataset
        config: Full configuration dict (from cli_manager.config)

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    if not MODELS_TRAINING_AVAILABLE:
        logger.error("Models training system not available")
        logger.error(f"Import error: {MODELS_TRAINING_IMPORT_ERROR}")
        return 1

    # Get models configuration
    models_config = config.get("models", {})

    # HPO enabled from merged config (CLI overrides already applied by cli_manager)
    hpo_enabled = models_config.get("hpo", {}).get("enabled", False)

    if hpo_enabled:
        return _run_hpo_training(args, logger, dataset, config)
    else:
        return _run_standard_training(args, logger, dataset, config)


def handle_predict_mode(
    args: argparse.Namespace, logger: logging.Logger, config: dict[str, Any]
) -> int:
    """
    Handle prediction mode execution.

    Phase 5b + Phase 7: Post-training inference with Dependency Injection pattern.

    DYNAMIC: Works with ANY model via checkpoint recreation
    PRODUCTION-READY: Explicit working_root_dir passed to all components
    FUTURE-PROOF: New input formats automatically supported via DataConverterRegistry

    Args:
        args: CLI arguments
        logger: Logger instance
        config: Full configuration dict

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    logger.info("=" * 60)
    logger.info("PREDICTION MODE (Phase 5b + Phase 7 DI)")
    logger.info("=" * 60)

    try:
        # Get prediction config
        prediction_config = config.get("models", {}).get("prediction", {})

        # =================================================================
        # Compute working_root_dir (Dependency Injection pattern)
        # =================================================================
        working_root_dir = _get_working_root_dir(config, logger)

        # Resolve model path (--model-path)
        model_path_arg = getattr(args, "model_path", None)
        if model_path_arg is None:
            logger.error("--model-path is required for prediction mode")
            return 1
        model_path = Path(model_path_arg).expanduser()
        if not model_path.is_absolute():
            # Check in checkpoint directory first
            checkpoint_dir = working_root_dir / "checkpoints"
            candidate = checkpoint_dir / model_path.name
            model_path = candidate if candidate.exists() else working_root_dir / model_path
        logger.info(f"Model checkpoint: {model_path}")

        # Resolve test path (--test-path)
        test_path_arg = getattr(args, "test_path", None)
        if test_path_arg is None:
            logger.error("--test-path is required for prediction mode")
            return 1
        test_path = Path(test_path_arg).expanduser()
        if not test_path.is_absolute():
            test_path = working_root_dir / test_path
        logger.info(f"Test data path: {test_path}")

        # Resolve output path (--preds-path)
        preds_path_arg = getattr(args, "preds_path", None) or prediction_config.get(
            "output_path", "./predictions.csv"
        )
        preds_path = Path(preds_path_arg).expanduser()
        if not preds_path.is_absolute():
            preds_path = working_root_dir / preds_path
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Predictions output: {preds_path}")

        # =================================================================
        # Get CLI overrides (or config defaults)
        # =================================================================
        batch_size = getattr(args, "predict_batch_size", None) or prediction_config.get(
            "batch_size", 32
        )
        device_str = getattr(args, "predict_device", None) or prediction_config.get(
            "device", "auto"
        )
        input_format = getattr(args, "predict_format", None) or prediction_config.get(
            "format", "auto"
        )
        output_format = getattr(args, "predict_output_format", None) or prediction_config.get(
            "output_format", "csv"
        )
        split = getattr(args, "predict_split", None) or prediction_config.get("split", "all")
        num_samples = getattr(args, "predict_num_samples", None) or prediction_config.get(
            "num_samples", None
        )
        include_inputs = getattr(args, "predict_include_inputs", False) or prediction_config.get(
            "include_inputs", False
        )
        getattr(args, "predict_uncertainty", False) or prediction_config.get("uncertainty", False)

        # Determine device
        if device_str == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_str)
        logger.info(f"Device: {device}")

        # =================================================================
        # Load predictor (Dependency Injection: pass working_root_dir)
        # =================================================================
        logger.info("Loading model from checkpoint...")
        predictor = Predictor.from_checkpoint(
            checkpoint_path=model_path, working_root_dir=working_root_dir, device=device
        )
        logger.info(f"Model loaded successfully. Task type: {predictor.task_type}")

        # ====================================================================
        # FIX 21: GET FEATURIZATION CONFIG FROM CHECKPOINT
        # ====================================================================
        # DYNAMIC: Uses whatever structural_features_config is in checkpoint
        # PRODUCTION-READY: Provides clear logging for debugging
        # FUTURE-PROOF: Works with any featurization config structure
        # ====================================================================
        structural_features_config = predictor.structural_features_config
        if structural_features_config:
            logger.info(
                f"Using featurization from checkpoint: "
                f"atom={list(structural_features_config.get('atom', []))}, "
                f"bond={list(structural_features_config.get('bond', []))}"
            )
        else:
            logger.warning(
                "No structural_features_config in checkpoint - using default featurization. "
                "This may cause dimension mismatch if training used different features. "
                "Consider re-training with updated code that saves featurization config."
            )

        # =================================================================
        # Load/prepare input data
        # =================================================================
        logger.info(f"Loading test data from: {test_path}")

        # Track input identifiers for include_inputs option
        input_identifiers = None

        if test_path.is_dir():
            # Assume miliaDataset directory
            from milia_pipeline.datasets import miliaDataset

            dataset = miliaDataset(root=str(test_path))

            # Get split if specified
            if split != "all" and hasattr(dataset, "get_split"):
                data_list = dataset.get_split(split)
            else:
                data_list = [dataset[i] for i in range(len(dataset))]
        elif test_path.suffix == ".pt":
            # =================================================================
            # FIX 22: HANDLE ALL PYG .PT FILE FORMATS
            # =================================================================
            # DYNAMIC: Detects and handles multiple PyG data formats:
            #   1. List[Data]: Direct list of Data objects
            #   2. Tuple (data, slices): PyG < 2.4 InMemoryDataset collated format
            #   3. Tuple (data, slices, data_cls): PyG >= 2.4 collated format
            #   4. Single Data object: Wrapped into list
            # PRODUCTION-READY: Uses official PyG separate() for reconstruction
            # FUTURE-PROOF: Compatible with all PyG versions and formats
            # =================================================================
            loaded_data = torch.load(test_path)

            if isinstance(loaded_data, list):
                # Format 1: Direct list of Data objects
                data_list = loaded_data
            elif isinstance(loaded_data, tuple) and len(loaded_data) in (2, 3):
                # Format 2/3: PyG InMemoryDataset collated format (data, slices[, data_cls])
                # This is the internal storage format of InMemoryDataset
                from torch_geometric.data import Data
                from torch_geometric.data.separate import separate

                if len(loaded_data) == 2:
                    # PyG < 2.4 format: (collated_data, slices)
                    collated_data, slices = loaded_data
                    data_cls = Data
                else:
                    # PyG >= 2.4 format: (collated_data, slices, data_cls)
                    collated_data, slices, data_cls = loaded_data
                    # Handle dict-based storage (PyG >= 2.4)
                    if isinstance(collated_data, dict):
                        collated_data = data_cls.from_dict(collated_data)

                # Determine number of graphs from slices
                # The slices dict contains start/end indices for each attribute
                # We can infer count from any attribute's slice tensor
                num_graphs = None
                if slices is not None:
                    for _key, slice_tensor in slices.items():
                        if hasattr(slice_tensor, "__len__"):
                            num_graphs = len(slice_tensor) - 1
                            break

                if num_graphs is not None and num_graphs > 0:
                    # Reconstruct individual Data objects using PyG's separate()
                    data_list = []
                    for idx in range(num_graphs):
                        data = separate(
                            cls=collated_data.__class__,
                            batch=collated_data,
                            idx=idx,
                            slice_dict=slices,
                            decrement=False,
                        )
                        data_list.append(data)
                    logger.info(f"Reconstructed {num_graphs} graphs from PyG collated format")
                elif slices is None:
                    # Single graph stored without slices
                    data_list = [collated_data]
                else:
                    raise ValueError(f"Could not determine number of graphs from slices: {slices}")
            elif hasattr(loaded_data, "x") or hasattr(loaded_data, "edge_index"):
                # Format 4: Single Data/HeteroData object (duck-typing check)
                data_list = [loaded_data]
            else:
                raise ValueError(
                    f"Unsupported .pt file format. Expected one of:\n"
                    f"  1. List[Data]: List of PyG Data objects\n"
                    f"  2. Tuple[Data, Dict]: PyG InMemoryDataset collated format\n"
                    f"  3. Data: Single PyG Data object\n"
                    f"Got: {type(loaded_data)}"
                )
        else:
            # File input (CSV, XYZ, SDF, etc.)
            if test_path.suffix == ".csv":
                import pandas as pd

                df = pd.read_csv(test_path)
                # Assume SMILES column
                smiles_col = "smiles" if "smiles" in df.columns else df.columns[0]
                inputs = df[smiles_col].tolist()
                input_identifiers = inputs  # Store for include_inputs

                # ================================================================
                # FIX 21: CONVERT TO PYG WITH SAME FEATURIZATION AS TRAINING
                # ================================================================
                # DYNAMIC: Passes structural_features_config from checkpoint
                # PRODUCTION-READY: Falls back gracefully if no config available
                # FUTURE-PROOF: Works with any converter that accepts the config
                # ================================================================
                data_list = [
                    convert_to_pyg(
                        inp,
                        format=input_format if input_format != "auto" else None,
                        structural_features_config=structural_features_config,  # FIX 21: Same featurization as training
                    )
                    for inp in inputs
                ]
            elif test_path.suffix.lower() in (".sdf", ".mol"):
                # ================================================================
                # FIX 24: MULTI-MOLECULE SDF FILE SUPPORT
                # ================================================================
                # DYNAMIC: Uses convert_sdf_to_pyg_list to load ALL molecules
                # PRODUCTION-READY: Handles multi-molecule SDF files correctly
                # FUTURE-PROOF: Works with any number of molecules in SDF file
                # ================================================================
                data_list = convert_sdf_to_pyg_list(
                    test_path,
                    structural_features_config=structural_features_config,
                    working_root_dir=working_root_dir,
                )
                # Generate identifiers for each molecule (use index if no names)
                input_identifiers = [f"mol_{i}" for i in range(len(data_list))]
            else:
                # Single file (XYZ, etc.)
                inputs = [str(test_path)]
                input_identifiers = [test_path.name]

                # ================================================================
                # FIX 21: CONVERT TO PYG WITH SAME FEATURIZATION AS TRAINING
                # ================================================================
                data_list = [
                    convert_to_pyg(
                        inp,
                        format=input_format if input_format != "auto" else None,
                        structural_features_config=structural_features_config,
                    )
                    for inp in inputs
                ]

        # Apply sample limit
        if num_samples is not None and num_samples < len(data_list):
            data_list = data_list[:num_samples]
            if input_identifiers is not None:
                input_identifiers = input_identifiers[:num_samples]

        logger.info(f"Loaded {len(data_list)} samples for prediction")

        # =================================================================
        # VALIDATE INPUT DATA DIMENSIONS
        # =================================================================
        # DYNAMIC: Checks any loaded data against model's expected dimensions
        # PRODUCTION-READY: Provides clear, actionable error messages
        # FUTURE-PROOF: Works with any model type and any data source
        # =================================================================
        if data_list:
            # Get expected in_channels from model
            model_in_channels = None

            # Try to get from predictor's model_info (checkpoint metadata)
            if hasattr(predictor, "model_info") and predictor.model_info:
                hyperparams = predictor.model_info.get("hyperparameters_values", {})
                model_in_channels = hyperparams.get("in_channels")

            # Fallback: Try to get from model directly
            if model_in_channels is None and hasattr(predictor, "model"):
                actual_model = predictor.model
                # Unwrap if necessary
                while hasattr(actual_model, "model"):
                    actual_model = actual_model.model
                if hasattr(actual_model, "in_channels"):
                    model_in_channels = actual_model.in_channels

            # Get actual feature dimension from test data
            test_sample = data_list[0]
            if hasattr(test_sample, "x") and test_sample.x is not None:
                test_in_channels = test_sample.x.size(-1)

                # Validate dimensions match
                if model_in_channels is not None and test_in_channels != model_in_channels:
                    raise ValueError(
                        f"\n"
                        f"{'=' * 70}\n"
                        f"FEATURE DIMENSION MISMATCH\n"
                        f"{'=' * 70}\n"
                        f"Model expects {model_in_channels} node features (in_channels={model_in_channels})\n"
                        f"Test data has {test_in_channels} node features\n"
                        f"\n"
                        f"This typically happens when:\n"
                        f"  1. Training data used different featurization than test data\n"
                        f"  2. Training: DFT/QM dataset with {model_in_channels} computed features\n"
                        f"  3. Testing: SMILES strings with basic {test_in_channels}-feature encoding\n"
                        f"\n"
                        f"SOLUTIONS:\n"
                        f"  Option 1: Use test data with matching featurization\n"
                        f"            - Process test molecules through same pipeline as training\n"
                        f"            - Use miliaDataset with same config as training\n"
                        f"\n"
                        f"  Option 2: Provide pre-processed .pt file\n"
                        f"            - Use --test-path with a .pt file containing\n"
                        f"              properly featurized PyG Data objects\n"
                        f"\n"
                        f"  Option 3: Train a new model on SMILES-featurized data\n"
                        f"            - Use SMILESConverter-compatible training data\n"
                        f"{'=' * 70}"
                    )

                logger.info(
                    f"Test data feature dimension: {test_in_channels} (model expects: {model_in_channels})"
                )

        # =================================================================
        # Run predictions
        # =================================================================
        logger.info("Running predictions...")
        predictions = predictor.predict_batch(data_list, batch_size=batch_size, return_numpy=True)
        logger.info(f"Predictions shape: {predictions.shape}")

        # =================================================================
        # Save predictions
        # =================================================================
        logger.info(f"Saving predictions to: {preds_path}")
        predictor.save_predictions(
            predictions=predictions,
            output_path=preds_path,
            format=output_format,
            include_inputs=include_inputs,
            input_identifiers=input_identifiers,
        )

        logger.info("=" * 60)
        logger.info("PREDICTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        logger.debug("Error details:", exc_info=True)
        return 1


def _run_standard_training(
    args: argparse.Namespace,
    logger: logging.Logger,
    dataset: "miliaDataset",
    config: dict[str, Any],
) -> int:
    """Execute standard (non-HPO) training workflow."""
    logger.info("=" * 60)
    logger.info("STANDARD TRAINING MODE")
    logger.info("=" * 60)

    try:
        # 1. Get configuration
        models_config = config.get("models", {})
        selection = models_config.get("selection", {})
        training_config = models_config.get("training", {})

        # CLI overrides (getattr with None default, then fallback to config)
        mode = getattr(args, "mode", None) or selection.get("mode", "single")
        model_name = getattr(args, "model_name", None) or selection.get("model_name", "GCN")
        task_type_override = getattr(args, "task_type", None) or selection.get("task_type", None)
        epochs = getattr(args, "epochs", None) or training_config.get("epochs", 100)
        batch_size = getattr(args, "batch_size", None) or training_config.get("batch_size", 32)
        lr = getattr(args, "learning_rate", None) or training_config.get("optimizer", {}).get(
            "params", {}
        ).get("lr", 0.001)

        # Determine task type with validation
        from milia_pipeline.models.hpo import infer_task_type

        sample_data = dataset[0] if hasattr(dataset, "__getitem__") and len(dataset) > 0 else None
        inferred_task_type = infer_task_type(
            dataset=dataset,
            metric=training_config.get("loss", {}).get("name", "mse"),
            sample_data=sample_data,
        )

        if task_type_override is not None:
            task_type = task_type_override
            # Validate CLI/config task type against inferred type
            if task_type != inferred_task_type:
                logger.info(
                    f"Task override: '{task_type}' (data suggests '{inferred_task_type}'). "
                    f"Pipeline will automatically prepare data for {task_type}."
                )
        else:
            task_type = inferred_task_type

        logger.info(f"Mode: {mode}")
        logger.info(f"Model: {model_name}")
        logger.info(
            f"Task: {task_type}"
            + (f" (inferred: {inferred_task_type})" if task_type != inferred_task_type else "")
        )
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Learning rate: {lr}")

        # 2. Split dataset
        data_split_config = training_config.get("data_split", {})
        train_data, val_data, test_data = DataSplitter.random_split(
            dataset,
            train_ratio=data_split_config.get("train_ratio", 0.8),
            val_ratio=data_split_config.get("val_ratio", 0.1),
            test_ratio=data_split_config.get("test_ratio", 0.1),
            random_seed=data_split_config.get("random_seed", 42),
        )

        logger.info(
            f"Data split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}"
        )

        # 2.5 Parse target_selection config BEFORE data preparation
        # ================================================================
        # TARGET SELECTION: Parse config for data preparation and model creation
        # ================================================================
        # DYNAMIC: Reads target_selection from models.selection config
        # PRODUCTION-READY: Handles missing config gracefully (None = all)
        # FUTURE-PROOF: Selection resolved before data prep, used by both prep and factory
        # ================================================================
        target_selection_raw = models_config.get("selection", {}).get("target_selection", None)
        target_selection_config = None

        if (
            target_selection_raw is not None
            and TARGET_SELECTION_AVAILABLE
            and TargetSelectionConfig is not None
        ):
            target_selection_config = TargetSelectionConfig.from_config(target_selection_raw)
            logger.info(
                f"Target selection configured: level={target_selection_config.config_level}, source={target_selection_config.config_source}"
            )

        # 2.6 Apply task-specific data preparation
        # Returns num_classes for classification tasks (from discretization n_bins or counted)
        num_classes_override = None
        try:
            train_data, val_data, test_data, num_classes_override = prepare_data_for_task(
                train_data, val_data, test_data, task_type, logger, target_selection_config
            )
            if num_classes_override is not None:
                logger.info(
                    f"Classification: num_classes={num_classes_override} (will be passed to model factory)"
                )
        except DataCompatibilityError as e:
            logger.error(f"Data incompatible with task type '{task_type}': {e}")
            return 1

        # 3. Create data loaders
        from torch_geometric.loader import DataLoader

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # 4. Create model via factory
        factory = get_factory()

        # Initialize model_info to None (will be populated below based on mode)
        model_info = None

        # Helper: Infer out_channels from sample data for graph-level tasks
        def _infer_out_channels(sample_data, task_type):
            """Infer output channels from sample data's y attribute."""
            if hasattr(sample_data, "y") and sample_data.y is not None:
                y = sample_data.y
                if y.dim() == 0:  # Scalar (single target or class index)
                    return 1
                elif y.dim() == 1:  # Multi-target [num_targets]
                    return y.size(0)
                else:  # Higher dimensional
                    return y.size(-1)
            return 1  # Default fallback

        # Determine model creation parameters based on mode
        if mode == "custom":
            # Custom architecture mode
            arch_config_path = getattr(args, "architecture_config", None)
            if arch_config_path:
                import yaml

                with open(arch_config_path) as f:
                    architecture_config = yaml.safe_load(f)
            else:
                architecture_config = models_config.get("custom_architecture", {})

            # Use train_data[0] as sample (after task-specific data prep)
            sample_data = train_data[0] if len(train_data) > 0 else dataset[0]

            # ================================================================
            # CLASSIFICATION FIX: Inject num_classes_override into hyperparameters
            # ================================================================
            # For classification custom models, inject out_channels so the
            # architecture builder creates layers with correct output dimension.
            #
            # DYNAMIC: Uses num_classes_override from prepare_data_for_task
            # PRODUCTION-READY: Ensures custom model has correct output dimension
            # FUTURE-PROOF: Works with any custom architecture configuration
            # ================================================================
            custom_hyperparameters = {"architecture_config": architecture_config}
            if num_classes_override is not None:
                custom_hyperparameters["out_channels"] = num_classes_override
                logger.info(
                    f"Classification custom model: Injecting out_channels={num_classes_override} "
                    f"(num_classes) into hyperparameters for model factory"
                )

            model = factory.create_model(
                name="custom",
                hyperparameters=custom_hyperparameters,
                task_type=task_type,
                sample_data=sample_data,
            )

            # ================================================================
            # FIX 22: CREATE COMPLETE model_info FOR CHECKPOINT COMPATIBILITY
            # ================================================================
            # PROBLEM: Minimal model_info was missing 'name' and 'hyperparameters_values'
            #          which callbacks.py requires for checkpoint['hyper_parameters']
            #          (line 549: 'model_name': trainer.model_info.get('name'))
            #          causing model_loader.py to fail with:
            #          "model_name is required but not found in checkpoint"
            #
            # DYNAMIC: Uses the same field names as create_model_with_info()
            # PRODUCTION-READY: Enables post_training.load_model() to work
            # FUTURE-PROOF: Consistent structure for all model modes
            # ================================================================
            is_classification = "classification" in task_type.lower()
            # For classification, use num_classes_override; otherwise infer from sample data
            if is_classification and num_classes_override is not None:
                out_channels = num_classes_override
            else:
                out_channels = _infer_out_channels(sample_data, task_type)

            # Extract in_channels from sample_data for checkpoint recreation
            in_channels = None
            if sample_data is not None and hasattr(sample_data, "x") and sample_data.x is not None:
                in_channels = sample_data.x.size(-1)

            # Build hyperparameters_values with all info needed for model recreation
            custom_hyperparams_for_checkpoint = dict(custom_hyperparameters)
            if in_channels is not None:
                custom_hyperparams_for_checkpoint["in_channels"] = in_channels
            custom_hyperparams_for_checkpoint["out_channels"] = out_channels

            model_info = {
                "name": "custom",  # CRITICAL: Required by callbacks.py line 549
                "task_type": task_type,
                "is_classification": is_classification,
                "out_channels": out_channels,
                "uses_edge_features": False,  # Custom model handles edge features internally
                "target_selection": None,  # Custom doesn't support target selection yet
                "hyperparameters_values": custom_hyperparams_for_checkpoint,  # CRITICAL: Required for model recreation
                "wrapper_info": {},  # For consistency with create_model_with_info()
            }
        elif mode == "ensemble":
            # Ensemble mode
            ensemble_config_path = getattr(args, "ensemble_config", None)
            if ensemble_config_path:
                import yaml

                with open(ensemble_config_path) as f:
                    ensemble_config = yaml.safe_load(f)
            else:
                ensemble_config = models_config.get("ensemble", {})

            # Use train_data[0] as sample (after task-specific data prep)
            sample_data = train_data[0] if len(train_data) > 0 else dataset[0]

            # ================================================================
            # CLASSIFICATION FIX: Inject num_classes_override into hyperparameters
            # ================================================================
            # For node/graph classification ensembles, each individual model needs
            # the correct out_channels (num_classes) to produce the right output shape.
            # The factory extracts num_classes_override from hyperparameters['out_channels']
            # and propagates it to individual models in the ensemble.
            #
            # EXCEPTION: For edge_classification, models should output node EMBEDDINGS
            # (not class logits). The EdgeLevelModelWrapper handles combining node
            # embeddings into edge predictions, and its MLP decoder outputs num_classes.
            # Injecting out_channels=num_classes would make models output class logits
            # instead of embeddings, breaking the edge decoding architecture.
            #
            # DYNAMIC: Uses num_classes_override from prepare_data_for_task
            # PRODUCTION-READY: Ensures correct output dimension per task type
            # FUTURE-PROOF: Works with any ensemble strategy (parallel, hierarchical, etc.)
            # ================================================================
            ensemble_hyperparameters = {"ensemble_config": ensemble_config}
            is_edge_classification = task_type.lower() == "edge_classification"

            # For edge_classification: DON'T inject out_channels into model hyperparameters
            # Models output embeddings; the EdgeLevelModelWrapper decoder outputs num_classes
            # For other classification tasks: inject out_channels so models output class logits
            if num_classes_override is not None and not is_edge_classification:
                ensemble_hyperparameters["out_channels"] = num_classes_override
                logger.info(
                    f"Classification ensemble: Injecting out_channels={num_classes_override} "
                    f"(num_classes) into hyperparameters for model factory"
                )
            elif num_classes_override is not None and is_edge_classification:
                # For edge_classification, pass num_classes to factory for EdgeLevelModelWrapper's decoder
                # but DON'T set out_channels in ensemble_hyperparameters (models output embeddings)
                # Use 'num_classes' key which factory will use for edge_out_channels in wrapper
                ensemble_hyperparameters["num_classes"] = num_classes_override
                logger.info(
                    f"edge_classification ensemble: NOT injecting out_channels into model hyperparameters. "
                    f"Models will output embeddings; EdgeLevelModelWrapper decoder will output {num_classes_override} classes."
                )

            model = factory.create_model(
                name="ensemble",
                hyperparameters=ensemble_hyperparameters,
                task_type=task_type,
                sample_data=sample_data,
            )

            # ================================================================
            # FIX 22: CREATE COMPLETE model_info FOR CHECKPOINT COMPATIBILITY
            # ================================================================
            # PROBLEM: Minimal model_info was missing 'name' and 'hyperparameters_values'
            #          which callbacks.py requires for checkpoint['hyper_parameters']
            #          (line 549: 'model_name': trainer.model_info.get('name'))
            #          causing model_loader.py to fail with:
            #          "model_name is required but not found in checkpoint"
            #
            # DYNAMIC: Uses the same field names as create_model_with_info()
            # PRODUCTION-READY: Enables post_training.load_model() to work
            # FUTURE-PROOF: Consistent structure for all model modes
            # ================================================================
            is_classification = "classification" in task_type.lower()
            # For classification, use num_classes_override; otherwise infer from sample data
            if is_classification and num_classes_override is not None:
                out_channels = num_classes_override
            else:
                out_channels = _infer_out_channels(sample_data, task_type)

            # Extract in_channels from sample_data for checkpoint recreation
            in_channels = None
            if sample_data is not None and hasattr(sample_data, "x") and sample_data.x is not None:
                in_channels = sample_data.x.size(-1)

            # Build hyperparameters_values with all info needed for model recreation
            ensemble_hyperparams_for_checkpoint = dict(ensemble_hyperparameters)
            if in_channels is not None:
                ensemble_hyperparams_for_checkpoint["in_channels"] = in_channels
            ensemble_hyperparams_for_checkpoint["out_channels"] = out_channels

            model_info = {
                "name": "ensemble",  # CRITICAL: Required by callbacks.py line 549
                "task_type": task_type,
                "is_classification": is_classification,
                "out_channels": out_channels,
                "uses_edge_features": False,  # Ensemble uses inner model edge handling
                "target_selection": None,  # Ensemble doesn't support target selection yet
                "hyperparameters_values": ensemble_hyperparams_for_checkpoint,  # CRITICAL: Required for model recreation
                "wrapper_info": {},  # For consistency with create_model_with_info()
            }

        else:
            # Single model mode (default)
            # ================================================================
            # TARGET SELECTION: Use already-parsed config from data preparation
            # ================================================================
            # The target_selection_config was already parsed before prepare_data_for_task
            # and used for data extraction. We reuse the same config for model creation.
            # ================================================================

            # Use train_data[0] as sample_data (after target extraction) for proper inference
            # This ensures model sees the actual data shape it will receive during training
            sample_data = train_data[0] if len(train_data) > 0 else dataset[0]

            model, model_info = factory.create_model_with_info(
                name=model_name,
                hyperparameters=models_config.get("hyperparameters", {}),
                task_type=task_type,
                sample_data=sample_data,
                target_selection_config=target_selection_config,
                num_classes_override=num_classes_override,  # From discretization
            )

        logger.info(f"Model created: {model.__class__.__name__}")

        # Log target selection result if applied
        if model_info and model_info.get("target_selection"):
            ts = model_info["target_selection"]
            logger.info(
                f"Target selection active: mode={ts['mode']}, "
                f"selected={ts.get('resolved_names') or ts.get('resolved_indices')}, "
                f"out_channels={model_info['out_channels']}"
            )

        # 5. Create callbacks
        # Pass full config for working_root_dir resolution when paths are null
        callbacks = _create_callbacks(training_config, logger, config=config)

        # 6. Create loss function and optimizer
        # DYNAMIC: _get_loss_function now auto-selects appropriate loss based on task_type
        # This prevents dtype mismatch errors between loss functions and targets

        loss_fn = _get_loss_function(training_config, task_type=task_type)
        optimizer = _get_optimizer(model, training_config)
        scheduler = _get_scheduler(optimizer, training_config)

        # 6.5 Create metrics via MetricsRegistry (NEW)
        # DYNAMIC: Uses task-aware metric selection with fallback
        # PRODUCTION-READY: Validates config metrics against task type
        # FUTURE-PROOF: Reads from EvaluationConfig, extensible via registry
        metrics = {}
        if METRICS_AVAILABLE:
            eval_config = training_config.get("evaluation", {})
            config_metrics = eval_config.get("metrics", None)

            try:
                metrics = get_metrics_for_task(
                    task_type=task_type,
                    metric_names=config_metrics,
                    num_classes=num_classes_override,
                )
                logger.info(f"Metrics created for task '{task_type}': {list(metrics.keys())}")
            except Exception as e:
                logger.warning(f"Failed to create metrics: {e}. Continuing without metrics.")
                metrics = {}

        # ====================================================================
        # FIX 16: CAPTURE FEATURIZATION CONFIG FOR CHECKPOINT
        # ====================================================================
        # DYNAMIC: Captures whatever structural_features_config the dataset has
        # PRODUCTION-READY: Enables identical featurization at inference time
        # FUTURE-PROOF: Works with ANY dataset that stores structural_features_config
        # ====================================================================
        if model_info is None:
            model_info = {}

        # Capture structural_features_config from dataset for checkpoint saving
        if hasattr(dataset, "structural_features_config") and dataset.structural_features_config:
            model_info["structural_features_config"] = dataset.structural_features_config
            logger.info(
                f"Featurization config captured for checkpoint: "
                f"atom_features={list(dataset.structural_features_config.get('atom', []))}, "
                f"bond_features={list(dataset.structural_features_config.get('bond', []))}"
            )
        else:
            logger.debug(
                "No structural_features_config found on dataset - checkpoint will use default featurization"
            )

        # 7. Create trainer with model_info for target selection
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            max_epochs=epochs,
            callbacks=callbacks,
            model_info=model_info,  # NEW: Pass model_info for target selection
            metrics=metrics,  # NEW: Pass metrics for evaluation
        )

        # 8. Check for checkpoint resume
        checkpoint_path = getattr(args, "checkpoint", None)
        if checkpoint_path:
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            trainer.load_checkpoint(Path(checkpoint_path))

        # ====================================================================
        # EVALUATE-ONLY MODE: Skip training, only run evaluation
        # ====================================================================
        # DYNAMIC: Checks args.evaluate_only flag
        # PRODUCTION-READY: Requires checkpoint for evaluate-only mode
        # FUTURE-PROOF: Separate code path avoids interference with training
        # ====================================================================
        evaluate_only = getattr(args, "evaluate_only", False)

        if evaluate_only:
            logger.info("=" * 60)
            logger.info("EVALUATE-ONLY MODE")
            logger.info("=" * 60)

            # Validate: evaluate-only requires a checkpoint
            if not checkpoint_path:
                logger.error(
                    "Evaluate-only mode requires --checkpoint argument. "
                    "Please provide a checkpoint path to evaluate."
                )
                return 1

            # Run evaluation on test set
            results = {}
            if hasattr(trainer, "test") and callable(trainer.test):
                test_results = trainer.test()
                results["test_loss"] = test_results.get("test_loss")
                logger.info(f"Test Results: {test_results}")
            else:
                logger.error("Trainer does not have test() method for evaluation")
                return 1

            # Optionally run validation evaluation
            if hasattr(trainer, "validate") and callable(trainer.validate):
                val_results = trainer.validate()
                results["val_loss"] = val_results.get("val_loss")
                logger.info(f"Validation Results: {val_results}")

            logger.info("✅ Evaluation completed successfully!")
            logger.info(f"Checkpoint evaluated: {checkpoint_path}")
            return 0

        # ====================================================================
        # STANDARD TRAINING MODE
        # ====================================================================
        # 9. Train
        logger.info("Starting training...")
        results = trainer.fit()

        # 10. Evaluate on test set (if not already done in fit())
        # The MILIA Trainer may evaluate on test_loader during fit() if provided.
        # Check if test results are already in results, or use test() method if available.
        eval_config = training_config.get("evaluation", {})
        if eval_config.get("test_after_training", True):
            # Check if Trainer has test() method (standard pattern)
            if hasattr(trainer, "test") and callable(trainer.test):
                # MILIA Trainer.test() takes no arguments - uses test_loader from constructor
                test_results = trainer.test()
                logger.info(f"Test Results: {test_results}")
            # Otherwise, check if test_loss is already in results from fit()
            elif "test_loss" in results:
                logger.info(f"Test Results (from fit): test_loss={results['test_loss']}")
            else:
                logger.debug(
                    "No separate test evaluation method available; test may have been run during fit()"
                )

        # 10.5 Generate training visualization (NEW)
        # DYNAMIC: Uses metrics_history from Trainer, reads config from evaluation.visualization
        # PRODUCTION-READY: Saves plots to output directory, configurable via config.yaml
        # FUTURE-PROOF: Supports multiple output formats and plot types
        if METRICS_AVAILABLE and hasattr(trainer, "metrics_history") and trainer.metrics_history:
            try:
                # Read visualization config from evaluation section
                viz_config = eval_config.get("visualization", {})
                viz_enabled = viz_config.get("enabled", True)

                if viz_enabled:
                    # Determine output directory for plots
                    output_dir = config.get("global_paths", {}).get("working_root_dir", "./output")
                    output_dir = Path(output_dir).expanduser() / "training_plots"

                    # Get configurable formats (default: png, html)
                    viz_formats = viz_config.get("formats", ["png", "html"])

                    # Get style configuration if provided
                    viz_style = viz_config.get("style", None)

                    # Get plot type toggles
                    viz_plots = viz_config.get("plots", {})
                    plot_loss = viz_plots.get("loss_curves", True)
                    plot_metrics = viz_plots.get("metrics", True)
                    plot_lr = viz_plots.get("learning_rate", True)
                    plot_interactive = viz_plots.get("interactive", True)

                    # Create visualizer with optional style config
                    from milia_pipeline.models.training import TrainingVisualizer

                    visualizer = TrainingVisualizer(
                        metrics_history=dict(trainer.metrics_history), style=viz_style
                    )

                    # Generate requested plots
                    output_dir.mkdir(parents=True, exist_ok=True)
                    from datetime import datetime

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_prefix = f"{model_name}_{timestamp}" if model_name else timestamp
                    saved_plots = {}

                    # Static plots (matplotlib - PNG)
                    if "png" in viz_formats:
                        if plot_loss:
                            loss_path = output_dir / f"{file_prefix}_loss_curves.png"
                            if visualizer.plot_loss_curves(save_path=loss_path):
                                saved_plots["loss_curves_png"] = loss_path

                        if plot_metrics:
                            metrics_path = output_dir / f"{file_prefix}_metrics.png"
                            if visualizer.plot_metrics(save_path=metrics_path):
                                saved_plots["metrics_png"] = metrics_path

                        if plot_lr:
                            lr_path = output_dir / f"{file_prefix}_learning_rate.png"
                            if visualizer.plot_learning_rate(save_path=lr_path):
                                saved_plots["learning_rate_png"] = lr_path

                    # Interactive plots (plotly - HTML)
                    if "html" in viz_formats and plot_interactive:
                        interactive_path = output_dir / f"{file_prefix}_interactive.html"
                        if visualizer.plot_interactive(save_path=interactive_path):
                            saved_plots["interactive_html"] = interactive_path

                    # PDF export (requires kaleido)
                    if "pdf" in viz_formats and plot_interactive:
                        pdf_path = output_dir / f"{file_prefix}_summary.pdf"
                        if visualizer.plot_interactive(save_path=pdf_path):
                            saved_plots["summary_pdf"] = pdf_path

                    logger.info(
                        f"Training visualization saved: {len(saved_plots)} files to {output_dir}"
                    )
                    for plot_type, path in saved_plots.items():
                        logger.debug(f"  {plot_type}: {path}")
                else:
                    logger.debug(
                        "Visualization disabled in config (evaluation.visualization.enabled: false)"
                    )

            except Exception as e:
                logger.warning(f"Failed to generate training visualization: {e}")

        # 11. Save results
        _save_training_results(trainer, results, args, logger, config)

        logger.info("✅ Training completed successfully!")
        return 0

    except ModelError as e:
        logger.error(f"Model error: {e}")
        if hasattr(e, "model_name"):
            logger.error(f"Model: {e.model_name}")
        return 1
    except TrainingError as e:
        logger.error(f"Training error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


def _run_hpo_training(
    args: argparse.Namespace,
    logger: logging.Logger,
    dataset: "miliaDataset",
    config: dict[str, Any],
) -> int:
    """Execute HPO training workflow."""
    logger.info("=" * 60)
    logger.info("HPO TRAINING MODE")
    logger.info("=" * 60)

    if not HPO_AVAILABLE:
        logger.error("HPO not available")
        logger.error(f"Import error: {HPO_IMPORT_ERROR}")
        return 1

    if not OPTUNA_AVAILABLE:
        logger.error("Optuna not installed - HPO requires Optuna")
        logger.error("Install with: pip install optuna")
        return 1

    try:
        # 1. Get HPO configuration
        models_config = config.get("models", {})
        hpo_config_dict = models_config.get("hpo", {})

        # CLI overrides
        n_trials = getattr(args, "n_trials", None) or hpo_config_dict.get("n_trials", 100)
        timeout = getattr(args, "hpo_timeout", None) or hpo_config_dict.get("timeout", None)
        cv_folds = getattr(args, "cv_folds", None) or hpo_config_dict.get("cv_folds", 0)
        backend = getattr(args, "hpo_backend", None) or hpo_config_dict.get("backend", "optuna")

        # CLI overrides for sampler/pruner
        sampler_type = getattr(args, "sampler", None) or hpo_config_dict.get("sampler", {}).get(
            "type", "tpe"
        )
        pruner_type = getattr(args, "pruner", None) or hpo_config_dict.get("pruner", {}).get(
            "type", "median"
        )

        logger.info(f"HPO Trials: {n_trials}")
        logger.info(f"HPO Timeout: {timeout}")
        logger.info(f"HPO Backend: {backend}")
        logger.info(f"CV Folds: {cv_folds}")
        logger.info(f"Sampler: {sampler_type}")
        logger.info(f"Pruner: {pruner_type}")

        # 2. Get model name and task type FIRST (needed for HPOConfig)
        selection = models_config.get("selection", {})
        mode = getattr(args, "mode", None) or selection.get("mode", "single")
        model_name = getattr(args, "model_name", None) or selection.get("model_name", "GCN")
        task_type = getattr(args, "task_type", None) or selection.get(
            "task_type", "graph_regression"
        )

        # 3. Build HPOConfig with CLI overrides (including task_type)
        hpo_config_dict_merged = hpo_config_dict.copy()
        hpo_config_dict_merged["enabled"] = True
        hpo_config_dict_merged["n_trials"] = n_trials
        hpo_config_dict_merged["timeout"] = timeout
        hpo_config_dict_merged["cv_folds"] = cv_folds
        hpo_config_dict_merged["backend"] = backend
        hpo_config_dict_merged["task_type"] = task_type

        # Override sampler/pruner types if specified via CLI
        if "sampler" not in hpo_config_dict_merged:
            hpo_config_dict_merged["sampler"] = {}
        hpo_config_dict_merged["sampler"]["type"] = sampler_type

        if "pruner" not in hpo_config_dict_merged:
            hpo_config_dict_merged["pruner"] = {}
        hpo_config_dict_merged["pruner"]["type"] = pruner_type

        hpo_config = HPOConfig.from_dict(hpo_config_dict_merged)

        # 4. Create HPO Manager
        manager = HPOManager(hpo_config)

        # 5. Resume study if requested
        resume_study_name = getattr(args, "resume_study", None)
        if resume_study_name:
            logger.info(f"Resuming study: {resume_study_name}")
            # HPOManager handles study resumption internally via study config

        # For custom/ensemble modes, use the mode name
        if mode == "custom":
            model_name = "custom"
        elif mode == "ensemble":
            model_name = "ensemble"

        logger.info(f"Model: {model_name}")
        logger.info(f"Task: {task_type}")

        # 6. Prepare base hyperparameters (non-optimized params)
        base_hyperparameters = models_config.get("hyperparameters", {})

        # 6b. Inject mode-specific configurations into base_hyperparameters
        # These define model structure (not optimized by HPO). HPO optimizes
        # individual hyperparameters within these structures.
        # Pattern: Mirrors _run_standard_training() lines 2957-2970
        if mode == "ensemble":
            # Load ensemble configuration from CLI arg or config.yaml
            ensemble_config_path = getattr(args, "ensemble_config", None)
            if ensemble_config_path:
                import yaml

                with open(ensemble_config_path) as f:
                    ensemble_config = yaml.safe_load(f)
                logger.info(f"Loaded ensemble config from: {ensemble_config_path}")
            else:
                ensemble_config = models_config.get("ensemble", {})
                logger.info("Using ensemble config from config.yaml")

            if not ensemble_config:
                raise HPOError(
                    "Ensemble mode requires ensemble configuration",
                    details="Provide --ensemble-config or configure models.ensemble in config.yaml",
                )

            base_hyperparameters["ensemble_config"] = ensemble_config
            logger.info(
                f"Ensemble config: {len(ensemble_config.get('models', []))} models, "
                f"strategy={ensemble_config.get('strategy', 'parallel')}, "
                f"fusion={ensemble_config.get('fusion', {}).get('method', 'mean')}"
            )

        elif mode == "custom":
            # Load custom architecture configuration from CLI arg or config.yaml
            architecture_config_path = getattr(args, "architecture_config", None)
            if architecture_config_path:
                import yaml

                with open(architecture_config_path) as f:
                    architecture_config = yaml.safe_load(f)
                logger.info(f"Loaded architecture config from: {architecture_config_path}")
            else:
                architecture_config = models_config.get("custom_architecture", {})
                logger.info("Using custom architecture config from config.yaml")

            if not architecture_config:
                raise HPOError(
                    "Custom mode requires architecture configuration",
                    details="Provide --architecture-config or configure models.custom_architecture in config.yaml",
                )

            base_hyperparameters["architecture_config"] = architecture_config
            logger.info(
                f"Custom architecture config: {len(architecture_config.get('layers', []))} layers"
            )

        # 7. Prepare trainer kwargs
        training_config = models_config.get("training", {})
        trainer_kwargs = {
            "max_epochs": training_config.get("epochs", 100),
        }

        # 8. Run optimization
        logger.info("Starting HPO optimization...")
        best_params = manager.optimize(
            model_name=model_name,
            dataset=dataset,
            base_hyperparameters=base_hyperparameters,
            trainer_kwargs=trainer_kwargs,
            config_dict=models_config,  # NEW: For target selection
        )

        # 9. Report results
        logger.info("=" * 60)
        logger.info("HPO RESULTS")
        logger.info("=" * 60)
        logger.info(f"Best Parameters: {best_params}")
        logger.info(f"Best Value: {manager.get_best_value()}")

        # 10. Get study statistics
        stats = manager.get_study_statistics()
        logger.info(f"Total Trials: {stats.get('n_trials', 'N/A')}")
        logger.info(f"Completed: {stats.get('n_completed', 'N/A')}")
        logger.info(f"Pruned: {stats.get('n_pruned', 'N/A')}")
        logger.info(f"Failed: {stats.get('n_failed', 'N/A')}")

        # 11. Save HPO results
        _save_hpo_results(manager, best_params, args, logger, config)

        # 12. Retrain final model with best hyperparameters
        # Best Practice: After HPO, retrain with best params and save the model
        # Reference: https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/010_reuse_best_trial.html
        # Implementation: Uses HPOManager.train_final_model() which leverages
        # the refactored registries (LossRegistry, OptimizerRegistry, SchedulerRegistry)
        # for DYNAMIC, PRODUCTION-READY, FUTURE-PROOF component creation.

        try:
            # Create callbacks (main.py responsibility - uses config for paths)
            callbacks = _create_callbacks(training_config, logger, config=config)

            # Train final model using HPOManager method
            # This uses the refactored registries with automatic parameter filtering
            model, final_trainer, final_results = manager.train_final_model(
                dataset=dataset,
                model_name=model_name,
                base_hyperparameters=base_hyperparameters,
                training_config=training_config,
                callbacks=callbacks,
                config_dict=models_config,
            )

            # Save final training results (main.py responsibility - uses config for paths)
            _save_training_results(final_trainer, final_results, args, logger, config)

            logger.info(f"Final model best_val_loss: {final_results.get('best_val_loss', 'N/A')}")
            logger.info("✅ Final model trained and saved successfully!")

        except Exception as e:
            logger.warning(f"Final model training failed: {e}")
            logger.warning(
                "HPO results were saved successfully. You can manually retrain with best_params.json"
            )

        logger.info("✅ HPO completed successfully!")
        return 0

    except HPOError as e:
        logger.error(f"HPO error: {e}")
        if hasattr(e, "trial_number"):
            logger.error(f"Trial: {e.trial_number}")
        return 1
    except Exception as e:
        logger.error(f"HPO failed: {e}", exc_info=True)
        return 1


# HELPER FUNCTIONS FOR TRAINING


def _get_working_root_dir(config: dict, logger: logging.Logger) -> Path:
    """
    Get the working root directory from config with proper fallbacks.

    Priority:
    1. config['global_paths']['working_root_dir']
    2. get_dataset_constants() root directory
    3. Current directory fallback

    Args:
        config: Full configuration dictionary
        logger: Logger for debug messages

    Returns:
        Path to working root directory
    """
    working_root_dir: Path | None = None

    # Priority 1: From config
    if config is not None:
        global_paths = config.get("global_paths", {})
        working_root_dir_str = global_paths.get("working_root_dir")
        if working_root_dir_str:
            working_root_dir = Path(working_root_dir_str).expanduser()
            logger.debug(f"Using working_root_dir from config: {working_root_dir}")

    # Priority 2: From get_dataset_constants
    if working_root_dir is None:
        try:
            _, _, dataset_root_dir = get_dataset_constants()
            working_root_dir = Path(dataset_root_dir).expanduser()
            logger.debug(f"Using working_root_dir from dataset constants: {working_root_dir}")
        except Exception:
            pass

    # Priority 3: Ultimate fallback to current directory
    if working_root_dir is None:
        working_root_dir = Path(".").resolve()
        logger.debug(f"Using current directory as working_root_dir: {working_root_dir}")

    return working_root_dir


def _create_callbacks(
    training_config: dict, logger: logging.Logger, config: dict[str, Any] | None = None
) -> list:
    """
    Create training callbacks from configuration using CallbackFactory.

    Delegates to CallbackFactory.from_config() for:
    - DYNAMIC: Supports 6 callback types via factory registry
    - PRODUCTION-READY: Automatic path resolution, parameter filtering, enabled flags
    - FUTURE-PROOF: New callbacks auto-available when registered in CallbackFactory

    Args:
        training_config: Training section of models config containing 'callbacks' key
            Expected structure:
            {
                'callbacks': {
                    'early_stopping': {'enabled': True, 'params': {...}},
                    'model_checkpoint': {'enabled': True, 'params': {...}},
                    'tensorboard': {'enabled': True, 'params': {...}},
                    'lr_monitor': {'enabled': True, 'params': {...}},
                    'progress_bar': {'enabled': True, 'params': {...}},
                }
            }
        logger: Logger instance for debug messages
        config: Full configuration dict (used for resolving working_root_dir
                when paths like dirpath or log_dir are set to null)

    Returns:
        List of callback instances

    Raises:
        RuntimeError: If MODELS_TRAINING system is not available

    Note:
        When dirpath or log_dir is null in config.yaml, CallbackFactory auto-generates
        paths under working_root_dir as documented in the configuration comments.
    """
    # Validate factory availability
    if not MODELS_TRAINING_AVAILABLE or CallbackFactory is None:
        raise RuntimeError(
            f"CallbackFactory not available. Import error: {MODELS_TRAINING_IMPORT_ERROR}"
        )

    # Resolve working_root_dir for auto-generated paths using helper
    # This ensures all outputs go to the configured working directory, not project root
    working_root_dir = _get_working_root_dir(config or {}, logger)

    # Extract callback configuration
    callback_config = training_config.get("callbacks", {})

    # Delegate to CallbackFactory
    # Factory handles: enabled flags, path resolution, parameter filtering, instantiation
    return CallbackFactory.from_config(
        callback_config=callback_config, working_root_dir=working_root_dir, callback_logger=logger
    )


def _get_loss_function(training_config: dict, task_type: str | None = None):
    """
    Get loss function from configuration using LossRegistry with task-aware selection.

    Delegates to LossRegistry.get_loss_for_task() for:
    - DYNAMIC: Supports 20+ loss functions via registry, auto-selects based on task_type
    - PRODUCTION-READY: Automatic parameter filtering, task-loss compatibility checking,
                        prevents dtype mismatch errors between loss functions and targets
    - FUTURE-PROOF: New loss functions auto-available when registered, extensible
                    task-to-default mappings in LossRegistry

    Loss Selection Strategy (handled by LossRegistry):
    1. If user explicitly configured a task-compatible loss → use it
    2. If user configured a task-incompatible loss → override with warning
    3. If no loss configured → auto-select based on task_type

    This prevents the common dtype mismatch errors:
    - "Found dtype Long but expected Float" (regression loss with classification targets)
    - "Expected Long but got Float" (classification loss with regression targets)

    Args:
        training_config: Training configuration dictionary containing 'loss' section
            Expected structure:
            {
                'loss': {
                    'name': str,      # e.g., 'mse', 'cross_entropy', 'focal'
                    'params': dict    # e.g., {'reduction': 'mean'}
                }
            }
        task_type: Task type string (e.g., 'graph_classification', 'graph_regression')
                   Used for automatic loss selection when needed.

    Returns:
        nn.Module: Instantiated loss function appropriate for the task type

    Raises:
        ValueError: If loss name not found in registry
        RuntimeError: If MODELS_TRAINING system is not available
    """
    # Validate registry availability
    if not MODELS_TRAINING_AVAILABLE or LossRegistry is None:
        raise RuntimeError(
            f"LossRegistry not available. Import error: {MODELS_TRAINING_IMPORT_ERROR}"
        )

    # Extract configuration
    loss_config = training_config.get("loss", {})
    loss_name = loss_config.get("name", None)  # None triggers auto-selection
    loss_params = loss_config.get("params", {})

    # Convert empty string to None for consistent auto-selection behavior
    if loss_name == "":
        loss_name = None

    # Delegate to LossRegistry
    # Registry handles: task-aware selection, compatibility checking, parameter filtering
    return LossRegistry.get_loss_for_task(
        task_type=task_type or "", name=loss_name, params=loss_params
    )


def _get_optimizer(model, training_config: dict):
    """
    Get optimizer from configuration using OptimizerRegistry.

    Delegates to OptimizerRegistry.get_optimizer() for:
    - DYNAMIC: Supports all 13+ PyTorch optimizers via registry
    - PRODUCTION-READY: Automatic parameter filtering, proper error handling
    - FUTURE-PROOF: New optimizers auto-available when registered

    Args:
        model: PyTorch model whose parameters to optimize
        training_config: Training configuration dict with 'optimizer' section
            Expected structure:
            {
                'optimizer': {
                    'name': str,      # e.g., 'adam', 'adamw', 'sgd', 'rmsprop', etc.
                    'params': dict    # e.g., {'lr': 0.001, 'weight_decay': 0.0001}
                }
            }

    Returns:
        torch.optim.Optimizer: Instantiated optimizer

    Raises:
        ValueError: If optimizer name not found in registry
        RuntimeError: If MODELS_TRAINING system is not available
    """
    # Validate registry availability
    if not MODELS_TRAINING_AVAILABLE or OptimizerRegistry is None:
        raise RuntimeError(
            f"OptimizerRegistry not available. Import error: {MODELS_TRAINING_IMPORT_ERROR}"
        )

    # Extract configuration
    opt_config = training_config.get("optimizer", {})
    opt_name = opt_config.get("name", "adam").lower()
    params = opt_config.get("params", {})

    # Delegate to OptimizerRegistry
    # Registry handles: parameter merging, filtering, validation, instantiation
    return OptimizerRegistry.get_optimizer(
        name=opt_name, model_parameters=model.parameters(), params=params
    )


def _get_scheduler(optimizer, training_config: dict):
    """
    Get learning rate scheduler from configuration using SchedulerRegistry.

    Delegates to SchedulerRegistry.get_scheduler() for:
    - DYNAMIC: Supports all 13+ PyTorch schedulers via registry
    - PRODUCTION-READY: Automatic parameter filtering, proper error handling
    - FUTURE-PROOF: New schedulers auto-available when registered

    Args:
        optimizer: PyTorch optimizer instance to attach scheduler to
        training_config: Training configuration dict with 'scheduler' section
            Expected structure:
            {
                'scheduler': {
                    'enabled': bool,  # Must be True to create scheduler
                    'name': str,      # e.g., 'reduce_on_plateau', 'cosine_annealing'
                    'params': dict    # e.g., {'factor': 0.5, 'patience': 10}
                }
            }

    Returns:
        LRScheduler or None: Instantiated scheduler, or None if disabled

    Raises:
        ValueError: If scheduler name not found in registry
        RuntimeError: If MODELS_TRAINING system is not available
    """
    # Extract configuration
    sched_config = training_config.get("scheduler", {})

    # Check if scheduler is enabled (preserve existing behavior)
    if not sched_config.get("enabled", False):
        return None

    # Validate registry availability
    if not MODELS_TRAINING_AVAILABLE or SchedulerRegistry is None:
        raise RuntimeError(
            f"SchedulerRegistry not available. Import error: {MODELS_TRAINING_IMPORT_ERROR}"
        )

    # Extract scheduler name and params
    sched_name = sched_config.get("name", "reduce_on_plateau").lower()
    params = sched_config.get("params", {})

    # Delegate to SchedulerRegistry
    # Registry handles: parameter merging, filtering, validation, instantiation
    return SchedulerRegistry.get_scheduler(name=sched_name, optimizer=optimizer, params=params)


def _save_training_results(trainer, results: dict, args, logger, config: dict):
    """
    Save training results and final checkpoint using Trainer.save_results().

    Delegates to Trainer.save_results() for:
    - DYNAMIC: Handles any results dictionary structure
    - PRODUCTION-READY: Comprehensive error handling, JSON serialization
    - FUTURE-PROOF: Extensible via Trainer class methods

    Args:
        trainer: Trainer instance after training
        results: Training results dictionary from fit()
        args: Command-line arguments (unused, kept for API compatibility)
        logger: Logger instance
        config: Configuration dictionary

    Raises:
        RuntimeError: If Trainer doesn't have save_results method (graceful fallback)
    """
    # Get working root directory from config
    working_root_dir = _get_working_root_dir(config, logger)

    # Get output subdirectory from config
    output_subdir = config.get("models", {}).get("training_output_subdir", "training_output")
    output_dir = working_root_dir / output_subdir

    logger.info(f"Training output directory: {output_dir}")

    # Delegate to Trainer.save_results() if available
    if hasattr(trainer, "save_results") and callable(trainer.save_results):
        saved_paths = trainer.save_results(
            output_dir=output_dir,
            results=results,
            save_checkpoint=True,
            checkpoint_filename="final_model.pt",
            results_filename="training_results.json",
        )
        logger.debug(f"Saved paths: {saved_paths}")
    else:
        # Fallback for backward compatibility with older Trainer versions
        logger.warning("Trainer.save_results() not available. Using legacy save logic.")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save checkpoint
        final_checkpoint_path = output_dir / "final_model.pt"
        trainer.save_checkpoint(final_checkpoint_path)
        logger.info(f"Final checkpoint saved: {final_checkpoint_path}")

        # Save results JSON
        import json

        results_path = output_dir / "training_results.json"
        with open(results_path, "w") as f:
            serializable_results = {}
            for k, v in results.items():
                if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                    serializable_results[k] = v
                else:
                    serializable_results[k] = str(v)
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Training results saved: {results_path}")


def _save_hpo_results(manager, best_params: dict, args, logger, config: dict):
    """
    Save HPO results and study information using HPOManager.save_results().

    Delegates to HPOManager.save_results() for:
    - DYNAMIC: Saves all available study information
    - PRODUCTION-READY: Comprehensive error handling, JSON serialization
    - FUTURE-PROOF: Extensible via HPOManager class methods

    Args:
        manager: HPOManager instance after optimization
        best_params: Best hyperparameters dictionary (unused, kept for API compatibility)
        args: Command-line arguments (unused, kept for API compatibility)
        logger: Logger instance
        config: Configuration dictionary

    Raises:
        RuntimeError: If HPOManager doesn't have save_results method (graceful fallback)
    """
    # Get working root directory from config
    working_root_dir = _get_working_root_dir(config, logger)

    # Get output subdirectory from config
    output_subdir = config.get("models", {}).get("hpo_output_subdir", "hpo_output")
    output_dir = working_root_dir / output_subdir

    logger.info(f"HPO output directory: {output_dir}")

    # Delegate to HPOManager.save_results() if available
    if hasattr(manager, "save_results") and callable(manager.save_results):
        saved_paths = manager.save_results(
            output_dir=output_dir,
            best_params_filename="best_params.json",
            statistics_filename="study_statistics.json",
            trials_filename="all_trials.json",
        )
        logger.debug(f"Saved paths: {saved_paths}")
    else:
        # Fallback for backward compatibility with older HPOManager versions
        logger.warning("HPOManager.save_results() not available. Using legacy save logic.")
        import json

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save best parameters
        best_params_path = output_dir / "best_params.json"
        with open(best_params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        logger.info(f"Best parameters saved: {best_params_path}")

        # Save study statistics
        stats = manager.get_study_statistics()
        stats_path = output_dir / "study_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Study statistics saved: {stats_path}")

        # Save all trials info
        all_trials = manager.get_all_trials()
        trials_path = output_dir / "all_trials.json"
        with open(trials_path, "w") as f:
            json.dump(all_trials, f, indent=2)
        logger.info(f"All trials saved: {trials_path}")


def main():
    """
    Main function that orchestrates the dataset processing pipeline.

    ENHANCED CLI Manager Enhancement: Now uses enhanced CLI manager for improved usability.
    """
    # Initialize logger early to ensure it's available in all exception handlers
    logger = None

    try:
        # Create CLI manager with basic logger for initialization
        basic_logger = logging.getLogger("Main_Init")
        basic_logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        basic_logger.addHandler(handler)

        # Parse arguments using enhanced CLI manager
        try:
            args, cli_manager = parse_cli_args(logger=basic_logger)
        except CLIValidationError as e:
            basic_logger.error(f"CLI validation error: {e}")
            sys.exit(1)

        # Handle interactive mode first
        if args.interactive:
            args = cli_manager.run_interactive_mode()

        # Setup full logging after we have the arguments
        datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = args.log_file

        logger = setup_logging(args.log_level, log_file)  # Back to args.log_level

        # Print startup information
        logger.info("=" * 80)
        logger.info("milia Dataset Processing - Enhanced CLI (CLI Manager Enhancement)")
        logger.info("=" * 80)
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Handle plugin operations (Plugin Operations Handling)
        if hasattr(cli_manager, "handle_plugin_operations"):
            try:
                should_exit = cli_manager.handle_plugin_operations(args)
                if should_exit:
                    logger.info("Plugin operation completed successfully")
                    return 0
            except Exception as e:
                logger.error(f"Plugin operation failed: {e}")
                return 1

        # Research API Operations: Research API Operations
        if hasattr(cli_manager, "handle_research_api_commands"):
            try:
                should_exit = cli_manager.handle_research_api_commands(args)
                if should_exit:
                    logger.info("Research API operation completed successfully")
                    return 0
            except Exception as e:
                logger.error(f"Research API operation failed: {e}")
                logger.debug("Error details:", exc_info=True)
                return 1

        # NEW: Descriptor Operations (Phase 3 Integration)
        if hasattr(cli_manager, "handle_descriptor_operations"):
            try:
                should_exit = cli_manager.handle_descriptor_operations(args)
                if should_exit:
                    logger.info("Descriptor operation completed successfully")
                    return 0
            except Exception as e:
                logger.error(f"Descriptor operation failed: {e}")
                logger.debug("Error details:", exc_info=True)
                return 1

        start_time = time.time()

        logger.info("Initializing transformation system...")
        _register_custom_transforms_on_startup(logger)

        # ====================================================================
        # Plugin System Integration: Plugin System Initialization
        # ====================================================================
        logger.info("Initializing plugin system...")
        _discover_and_register_plugins(logger)

        # Handle special modes that exit early
        if args.list_experimental_setups:
            list_experimental_setups_info(logger)
            return

        if args.list_transforms:
            list_available_transforms_info(logger)
            return

        if args.validate_transforms_only:
            handle_transform_validation(logger, args.experimental_setup)
            return

        if args.validate_config:
            handle_config_validation(logger, args, cli_manager)
            return

        # PREPROCESSING MODES
        if getattr(args, "preprocess", False):
            if not PREPROCESSING_AVAILABLE:
                logger.error("Preprocessing system not available")
                logger.error(f"Import error: {PREPROCESSING_IMPORT_ERROR}")
                logger.info("Hint: Ensure Phase 1-3 (preprocessing subsystem) is implemented")
                sys.exit(1)
            return handle_preprocessing_mode(args, logger)

        if getattr(args, "validate_preprocessing_only", False):
            if not PREPROCESSING_AVAILABLE:
                logger.error("Preprocessing system not available")
                sys.exit(1)
            return handle_preprocessing_validation(args, logger)

        if getattr(args, "test_preprocessor_only", False):
            if not PREPROCESSING_AVAILABLE:
                logger.error("Preprocessing system not available")
                sys.exit(1)
            return handle_preprocessor_testing(args, logger)

        # PREDICTION MODE (Phase 5b + Phase 6)
        if getattr(args, "predict", False):
            if not POST_TRAINING_AVAILABLE:
                logger.error("Post-training system not available")
                logger.error(f"Import error: {POST_TRAINING_IMPORT_ERROR}")
                logger.info("Hint: Ensure Phase 5b (post_training module) is implemented")
                sys.exit(1)

            # Load config for path resolution
            try:
                config = cli_manager.load_and_merge_config(args)
            except CLIValidationError as e:
                logger.error(f"Configuration error: {e}")
                sys.exit(1)

            return handle_predict_mode(args, logger, config)

        # Load and merge configuration
        try:
            config = cli_manager.load_and_merge_config(args)
        except CLIValidationError as e:
            logger.error(f"Configuration error: {e}")
            sys.exit(1)

        # Validate configuration if not skipped
        if not args.skip_validation:
            try:
                cli_manager.validate_configuration(args)
            except CLIValidationError as e:
                logger.error(f"Configuration validation failed: {e}")
                sys.exit(1)

        # Print configuration summary
        cli_manager.print_configuration_summary(args)

        # Determine validation approach
        validate_handlers = HANDLERS_AVAILABLE  # Handlers always used
        validate_transforms = not args.disable_transforms and GRAPH_TRANSFORMS_AVAILABLE

        if args.disable_transforms:
            logger.warning("Transformation system disabled by CLI")

        # Validate configuration and get containers
        dataset_config, filter_config, processing_config, config = validate_configuration(
            logger,
            validate_handlers=validate_handlers,
            validate_transforms=validate_transforms,
            experimental_setup=args.experimental_setup,
        )

        # Handle handler testing mode
        if args.test_handlers_only:
            success = handler_integration_test(
                dataset_config, filter_config, processing_config, logger
            )
            sys.exit(0 if success else 1)

        # Determine root directory
        root_dir = args.root_dir or Path(get_dataset_constants()[2]).expanduser()
        logger.info(f"Using root directory: {root_dir}")

        # Construct processed file path
        processed_data_path = Path(root_dir) / "processed" / PROCESSED_DATA_FILENAME

        # Print dataset information
        print_dataset_info(logger, dataset_config, processing_config, args.experimental_setup)

        # Handle different processing modes
        if args.quick_validation:
            success = run_quick_validation(
                str(root_dir),
                logger,
                dataset_config,
                filter_config,
                processing_config,
                args.experimental_setup,
            )
            sys.exit(0 if success else 1)

        if args.stats_only:
            handle_stats_only_mode(
                str(root_dir),
                logger,
                dataset_config,
                filter_config,
                processing_config,
                args.experimental_setup,
            )
            return

        if args.dry_run:
            logger.info("Dry run mode - configuration validated successfully")
            logger.info("No dataset processing performed")
            return

        # Main processing mode with enhanced error handling
        logger.info("Starting main dataset processing...")

        dataset = create_dataset_with_error_handling(
            str(root_dir),
            logger,
            dataset_config,
            filter_config,
            processing_config,
            args.chunk_size,
            args.force_reload,
            experimental_setup=args.experimental_setup,
        )

        if dataset is None or len(dataset) == 0:
            logger.critical("Dataset creation failed or resulted in empty dataset!")
            sys.exit(1)

        logger.info("✅“ Dataset processing completed successfully!")
        logger.info(f"Final dataset contains {len(dataset)} molecules")

        # NEW: Verify transform pipeline functionality
        if hasattr(dataset, "test_transform_pipeline") and dataset.pre_transform_pipeline:
            try:
                test_results = dataset.test_transform_pipeline(num_samples=3)
                if test_results["successful_applications"] > 0:
                    logger.info("✅ Transform pipeline verified - transforms are being applied")
            except Exception as e:
                logger.debug(f"Transform pipeline test skipped: {e}")

        # Handle experimental setup switching
        if args.switch_experimental_setup:
            handle_setup_switching(dataset, args.switch_experimental_setup, logger)

        # Run access tests
        if not dataset_access_test(dataset, logger, dataset_config=dataset_config):
            logger.error("Dataset access tests failed")
            sys.exit(1)

        # Training Mode Handling
        if getattr(args, "train", False) or getattr(args, "evaluate_only", False):
            logger.info("=" * 60)
            logger.info("ENTERING TRAINING MODE")
            logger.info("=" * 60)

            exit_code = handle_training_mode(args, logger, dataset, cli_manager.config)

            if exit_code != 0:
                logger.error("Training workflow failed")
                sys.exit(exit_code)

            logger.info("Training workflow completed successfully")

        # Generate statistics
        analyze_dataset_statistics(dataset, logger, dataset_config=dataset_config)

        # Print final summary
        print_final_summary(
            logger,
            args,
            dataset,
            dataset_config,
            start_time,
            root_dir,
            processed_data_path,
            validate_handlers,
            validate_transforms,
        )

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user (Ctrl+C)")
        sys.exit(1)
    except CLIValidationError as e:
        logger.error(f"CLI validation error: {e}")
        sys.exit(1)
    except HandlerError as e:
        handle_handler_error(e, logger)
        sys.exit(1)
    except (TransformConfigurationError, ExperimentalSetupError, TransformCompositionError) as e:
        handle_transform_error(e, logger)
        sys.exit(1)

    # ========================================================================
    # Plugin System Integration: Plugin-Specific Exception Handling
    # ========================================================================
    except PluginError as e:
        logger.error(f"Plugin system error: {e}")

        # Get error context
        error_type = type(e).__name__
        logger.error(f"Error type: {error_type}")

        # Check for error details
        if hasattr(e, "plugin_name") and e.plugin_name:
            logger.error(f"Plugin: {e.plugin_name}")
        if hasattr(e, "details") and e.details:
            logger.error(f"Details: {e.details}")

        # Provide recovery suggestions
        logger.info("Suggested actions:")
        logger.info("  1. Check plugin configuration in config.yaml")
        logger.info("  2. Use --list-plugins to see plugin status")
        logger.info("  3. Use --validate-plugin PLUGIN_NAME for details")
        logger.info("  4. Disable plugin in config.yaml if problematic")
        logger.info("  5. Set plugins.enabled: false to disable plugin system")

        sys.exit(1)

    # Model/HPO Exception Handling
    except ModelError as e:
        logger.error(f"Model error: {e}")
        if hasattr(e, "model_name"):
            logger.error(f"Model: {e.model_name}")
        if hasattr(e, "details"):
            logger.error(f"Details: {e.details}")
        logger.info("Suggested actions:")
        logger.info("  1. Check model name exists in registry")
        logger.info("  2. Verify hyperparameters are valid for the model")
        logger.info("  3. Check dataset compatibility with model requirements")
        logger.info("  4. Use --mode to specify: single, custom, or ensemble")
        sys.exit(1)

    except HPOError as e:
        logger.error(f"HPO error: {e}")
        if hasattr(e, "trial_number"):
            logger.error(f"Failed trial: {e.trial_number}")
        if hasattr(e, "study_name"):
            logger.error(f"Study: {e.study_name}")
        logger.info("Suggested actions:")
        logger.info("  1. Check HPO configuration in config.yaml models.hpo section")
        logger.info("  2. Verify search space parameters are valid")
        logger.info("  3. Ensure Optuna is installed: pip install optuna")
        logger.info("  4. Check if n_trials or timeout are reasonable")
        logger.info("  5. Use --resume-study to continue a failed study")
        sys.exit(1)

    except TrainingError as e:
        logger.error(f"Training error: {e}")
        if hasattr(e, "epoch"):
            logger.error(f"Failed at epoch: {e.epoch}")
        logger.info("Suggested actions:")
        logger.info("  1. Check training configuration in config.yaml")
        logger.info("  2. Verify dataset is compatible with the model")
        logger.info("  3. Try reducing batch size if memory errors occur")
        logger.info("  4. Use --checkpoint to resume from last checkpoint")
        sys.exit(1)

    except BaseProjectError as e:
        logger.error(f"Project error: {e}")
        if hasattr(e, "details") and e.details:
            logger.error(f"Details: {e.details}")
        sys.exit(1)
    except Exception as e:
        # Use logger if available, otherwise fall back to basic_logger
        active_logger = logger if logger is not None else basic_logger
        active_logger.critical(f"Unexpected error: {e}", exc_info=True)

        sys.exit(1)


def handler_integration_test(
    dataset_config: DatasetConfig,
    filter_config: FilterConfig,
    processing_config: ProcessingConfig,
    logger: logging.Logger,
) -> bool:
    """
    Tests dataset handler integration and functionality.

    Handler-Based Pattern Development ENHANCEMENT: Function to test handler pattern integration.

    Args:
        dataset_config (DatasetConfig): Dataset configuration container
        filter_config (FilterConfig): Filter configuration container
        processing_config (ProcessingConfig): Processing configuration container
        logger (logging.Logger): Logger instance

    Returns:
        bool: True if handler integration tests pass, False otherwise
    """
    logger.info("Testing handler integration...")

    if not HANDLERS_AVAILABLE:
        logger.warning("Handlers not available - skipping integration tests")
        return True  # Not a failure if handlers aren't available

    try:
        # Test handler creation
        handler = create_handler_for_validation(
            dataset_config, filter_config, processing_config, logger
        )

        # Test basic handler interface
        required_methods = [
            "validate_molecule_data",
            "get_required_properties",
            "process_property_value",
            "enrich_pyg_data",
            "get_processing_statistics",
        ]

        for method_name in required_methods:
            if not hasattr(handler, method_name):
                logger.error(f"Handler missing required method: {method_name}")
                return False

            method = getattr(handler, method_name)
            if not callable(method):
                logger.error(f"Handler attribute {method_name} is not callable")
                return False

        logger.debug(f"✅“ Handler interface validation passed: {type(handler).__name__}")

        # Test handler-specific functionality
        try:
            required_props = handler.get_required_properties()
            logger.debug(f"Handler required properties: {required_props}")

            stats = handler.get_processing_statistics()
            logger.debug(f"Handler statistics: {stats}")

        except Exception as e:
            logger.warning(f"Handler method test failed: {e}")
            # Not critical - may fail if handler hasn't processed any data yet

        logger.info("✅“ Handler integration tests passed")
        return True

    except HandlerError as e:
        logger.error(f"Handler integration test failed: {e}")

        # Check if this is recoverable
        if is_recoverable_handler_error(e):
            logger.warning("Handler error is recoverable - system can use fallback")
            return True
        else:
            logger.error("Handler error is not recoverable")
            return False

    except Exception as e:
        logger.error(f"Unexpected error in handler integration test: {e}")
        return False


def handle_preprocessing_mode(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Execute preprocessing workflow with 5-step orchestration"""

    # Step 1: Build config from config.yaml + CLI overrides
    from milia_pipeline.config.config_accessors import get_preprocessing_config

    config = get_preprocessing_config(
        dataset_type=args.preprocess_dataset,
        raw_tar_path_override=getattr(args, "preprocess_input", None),
        output_npz_path_override=getattr(args, "preprocess_output", None),
    )

    # Apply CLI overrides for optional parameters
    if hasattr(args, "preprocess_num_molecules") and args.preprocess_num_molecules is not None:
        config["num_molecules"] = args.preprocess_num_molecules
    if hasattr(args, "preprocess_feature_tier") and args.preprocess_feature_tier is not None:
        config["feature_tier"] = args.preprocess_feature_tier
    if hasattr(args, "preprocess_cleanup") and args.preprocess_cleanup is not None:
        config["cleanup_temp"] = args.preprocess_cleanup

    # Step 2: Get preprocessor class via PreprocessorRegistry
    preprocessor_class = PreprocessorRegistry.get_preprocessor(args.preprocess_dataset)

    # Step 3: Initialize preprocessor
    preprocessor = preprocessor_class(config, logger)

    # Step 4: Execute preprocessing
    preprocessor.run()

    # Step 5: Validate output
    # (already done in preprocessor.run())

    return 0  # Success


def handle_preprocessing_validation(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Validate preprocessing configuration"""

    logger.info("=" * 80)
    logger.info("PREPROCESSING VALIDATION MODE")
    logger.info("=" * 80)

    try:
        # Check preprocessor exists
        if not PreprocessorRegistry.supports_preprocessing(args.preprocess_dataset):
            logger.error(f"Preprocessor not found: {args.preprocess_dataset}")
            return 1

        logger.info(f"✅ Preprocessor available: {args.preprocess_dataset}")

        # Build config from config.yaml + CLI overrides
        from milia_pipeline.config.config_accessors import get_preprocessing_config

        config = get_preprocessing_config(
            dataset_type=args.preprocess_dataset,
            raw_tar_path_override=getattr(args, "preprocess_input", None),
            output_npz_path_override=getattr(args, "preprocess_output", None),
        )

        # Apply CLI overrides
        if hasattr(args, "preprocess_num_molecules") and args.preprocess_num_molecules is not None:
            config["num_molecules"] = args.preprocess_num_molecules
        if hasattr(args, "preprocess_feature_tier") and args.preprocess_feature_tier is not None:
            config["feature_tier"] = args.preprocess_feature_tier

        logger.info("Configuration built:")
        logger.info(f"  Input: {config['raw_tar_path']}")
        logger.info(f"  Output: {config['output_npz_path']}")

        # Get preprocessor class and test instantiation
        preprocessor_class = PreprocessorRegistry.get_preprocessor(args.preprocess_dataset)
        logger.info(f"✅ Preprocessor class: {preprocessor_class.__name__}")

        preprocessor_class(config, logger)
        logger.info("✅ Preprocessor instantiation successful")

        logger.info("=" * 80)
        logger.info("✅ VALIDATION PASSED")
        logger.info("=" * 80)
        return 0

    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return 1


def handle_preprocessor_testing(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Test preprocessor availability"""

    # List all preprocessors
    available = PreprocessorRegistry.list_preprocessors()
    logger.info(f"Found {len(available)} preprocessor(s): {available}")

    # Test each preprocessor
    for name in available:
        preprocessor_class = PreprocessorRegistry.get_preprocessor(name)
        logger.info(f"✅ {name}: {preprocessor_class.__name__}")

    return 0


def list_available_transforms_info(logger: logging.Logger) -> None:
    """List available transforms by category."""
    if not GRAPH_TRANSFORMS_AVAILABLE:
        logger.error("Transformation system not available")
        return

    try:
        gt = get_graph_transforms()
        available_transforms = gt.get_available_transforms()

        logger.info("=" * 60)
        logger.info("AVAILABLE TRANSFORMS BY CATEGORY")
        logger.info("=" * 60)

        # 'all' is a meta-category in the API contract — a flat union of every
        # transform across all per-category buckets. Skip it during rendering so
        # the listing isn't duplicated, and skip it in the count-fallback so the
        # per-bucket totals can't double-count.
        for category, transforms in available_transforms.items():
            if not isinstance(transforms, list):
                continue
            if category == "all":
                continue
            logger.info(f"\n{category}:")
            for transform_name in sorted(transforms):
                logger.info(f"  - {transform_name}")

        total = available_transforms.get(
            "count",
            sum(
                len(v)
                for k, v in available_transforms.items()
                if isinstance(v, list) and k != "all"
            ),
        )
        logger.info(f"\nTotal transforms: {total}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to list transforms: {e}")


def handle_transform_validation(logger: logging.Logger, experimental_setup: str | None) -> None:
    """Handle transform validation mode."""
    if not GRAPH_TRANSFORMS_AVAILABLE:
        logger.error("Transformation system not available")
        sys.exit(1)

    try:
        validate_transformation_system(logger, experimental_setup)
        logger.info("✅“ Transformation system validation completed successfully")
    except (TransformConfigurationError, ExperimentalSetupError) as e:
        logger.error(f"Transformation validation failed: {e}")
        sys.exit(1)


def handle_config_validation(
    logger: logging.Logger, args: argparse.Namespace, cli_manager: CLIManager
) -> None:
    """Handle configuration validation mode."""
    try:
        config = cli_manager.load_and_merge_config(args)
        cli_manager.validate_configuration(args)

        logger.info("=" * 60)
        logger.info("CONFIGURATION VALIDATION SUCCESSFUL")
        logger.info("=" * 60)
        logger.info(f"Configuration file: {args.config}")
        logger.info(f"Dataset type: {get_dataset_type()}")

        # Show key configuration elements
        dataset_type = config.get("dataset_type", "unknown")
        logger.info(f"Dataset type: {dataset_type}")

        if "transformations" in config:
            transform_config = config["transformations"]
            if isinstance(transform_config, dict):
                # Show standard transforms if present
                if "standard_transforms" in transform_config:
                    standard_count = len(transform_config["standard_transforms"])
                    logger.info(f"Standard transforms: {standard_count}")

                # Show experimental setups if present
                if "experimental_setups" in transform_config:
                    setups = list(transform_config["experimental_setups"].keys())
                    logger.info(f"Experimental setups: {len(setups)}")

                logger.info(f"Default setup: {transform_config.get('default_setup', 'none')}")

        logger.info("=" * 60)

    except CLIValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)


def handle_stats_only_mode(
    root_dir: str,
    logger: logging.Logger,
    dataset_config: DatasetConfig,
    filter_config: FilterConfig,
    processing_config: ProcessingConfig,
    experimental_setup: str | None,
) -> None:
    """Handle statistics-only mode."""
    logger.info("Statistics-only mode - loading existing dataset...")
    try:
        dataset = miliaDataset.create_with_containers(
            root=root_dir,
            logger=logger,
            dataset_config=dataset_config,
            filter_config=filter_config,
            processing_config=processing_config,
            force_reload=False,
            experimental_setup=experimental_setup,
        )

        analyze_dataset_statistics(dataset, logger, dataset_config)

    except Exception as e:
        logger.error(f"Failed to load dataset for statistics: {e}")
        sys.exit(1)


def handle_setup_switching(dataset: miliaDataset, setup_name: str, logger: logging.Logger) -> None:
    """Handle experimental setup switching."""
    if not hasattr(dataset, "switch_experimental_setup"):
        logger.error("Dataset does not support experimental setup switching")
        sys.exit(1)

    logger.info(f"Switching to experimental setup: '{setup_name}'")
    success = dataset.switch_experimental_setup(setup_name)

    if success:
        logger.info(f"✅ Successfully switched to experimental setup: '{setup_name}'")
    else:
        logger.error(f"Failed to switch to experimental setup: '{setup_name}'")
        sys.exit(1)


def handle_handler_error(e: HandlerError, logger: logging.Logger) -> None:
    """Handle handler pattern errors with helpful output."""
    logger.error(f"Handler pattern error: {e}")

    error_summary = format_handler_exception_summary(e)
    logger.error(f"Error details: {error_summary}")

    suggestions = get_exception_recovery_suggestions(e)
    if suggestions:
        logger.info("Suggested recovery actions:")
        for i, suggestion in enumerate(suggestions[:3], 1):
            logger.info(f"  {i}. {suggestion}")


def handle_transform_error(e: Exception, logger: logging.Logger) -> None:
    """Handle transformation system errors with helpful output."""
    logger.error(f"Transformation system error: {e}")

    if hasattr(e, "experimental_setup") and e.experimental_setup:
        logger.error(f"Experimental setup: '{e.experimental_setup}'")
    if hasattr(e, "available_setups") and e.available_setups:
        logger.error(f"Available setups: {e.available_setups}")
    if hasattr(e, "validation_errors") and e.validation_errors:
        logger.error(f"Validation errors: {e.validation_errors}")

    logger.info("Suggested actions:")
    logger.info("  1. Use --list-experimental-setups to see available setups")
    logger.info("  2. Use --validate-transforms-only to test transformation system")
    logger.info("  3. Use --disable-transforms to bypass transformation system")


def print_final_summary(
    logger: logging.Logger,
    args: argparse.Namespace,
    dataset: miliaDataset,
    dataset_config: DatasetConfig,
    start_time: float,
    root_dir: Path,
    processed_data_path: Path,
    validate_handlers: bool,
    validate_transforms: bool,
) -> None:
    """Print comprehensive final summary."""
    elapsed_time = time.time() - start_time

    logger.info("=" * 80)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Dataset type: {dataset_config.dataset_type}")
    logger.info(f"Root directory: {root_dir}")
    logger.info(f"Processed file: {processed_data_path}")
    logger.info(f"Final molecule count: {len(dataset)}")
    logger.info(f"Processing time: {elapsed_time:.2f}s ({elapsed_time / 60:.1f} min)")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Force reload: {args.force_reload}")

    # System status
    logger.info(f"Handler pattern: {'✅“ Used' if validate_handlers else '❌— Legacy'}")
    logger.info(f"Transformation system: {'✅“ Used' if validate_transforms else '❌— Legacy'}")

    # Experimental setup info
    if args.experimental_setup:
        logger.info(f"Experimental setup: '{args.experimental_setup}'")

    # Filter information
    active_filters = []
    if hasattr(args, "max_atoms") and args.max_atoms:
        active_filters.append(f"max_atoms={args.max_atoms}")
    if hasattr(args, "min_atoms") and args.min_atoms:
        active_filters.append(f"min_atoms={args.min_atoms}")
    if hasattr(args, "max_uncertainty") and args.max_uncertainty:
        active_filters.append(f"max_uncertainty={args.max_uncertainty}")

    if active_filters:
        logger.info(f"Active filters: {', '.join(active_filters)}")
    elif hasattr(args, "no_filters") and args.no_filters:
        logger.info("Filters: Disabled")
    else:
        logger.info("Filters: From config.yaml")

    logger.info("✅ All processing completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
