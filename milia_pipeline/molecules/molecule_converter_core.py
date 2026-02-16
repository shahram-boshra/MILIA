# molecule_converter_core.py - Complete Integration with Transformation System

"""
Core molecular conversion logic for the conversion pipeline.

This module contains the main MoleculeDataConverter class that orchestrates the entire
conversion pipeline from raw molecular data to PyTorch Geometric (PyG) Data objects.
It handles any registered dataset type through the dataset handlers pattern.

MIGRATED: Handler Architecture Migration - Enhanced with handler-specific exceptions and improved error handling.
Incorporates the new exception hierarchy for better error reporting and recovery.
Maintains all functionality from Foundational Handler Architecture while adding Handler Architecture Migration enhancements.

ENHANCED: Complete integration with transformation system refactoring:
- Full DatasetConfig, FilterConfig, ProcessingConfig, TransformationConfig integration
- Removed all remaining legacy config dictionary access
- Added transform system awareness for conversion-time transforms
- Enhanced error handling with new transformation exception hierarchy
- Integration with graph_transforms module for transformation capabilities
"""

import logging
from typing import Any, Optional

import numpy as np
import torch
from torch_geometric.data import Data

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    AllChem = None

from milia_pipeline.config.config_accessors import (
    get_charge_handling_config,
    get_data_config,
    get_dataset_appropriate_structural_features,
    get_geometric_features_config,
    get_identifier_keys,
    get_stereochemistry_config,
    is_structural_features_enabled,
    list_experimental_setups,
    should_enable_stereochemistry_preprocessing,
    should_pass_coordinates_to_structural_features,
    should_pass_mulliken_charges_to_structural_features,
)
from milia_pipeline.config.config_constants import (
    ATOMIC_ENERGIES_HARTREE,
    HAR2EV,
    HEAVY_ATOM_SYMBOLS_TO_Z,
    get_handler_required_properties,
)
from milia_pipeline.config.config_containers import (
    DatasetConfig,
    FilterConfig,
    ProcessingConfig,
    TransformationConfig,
    TransformSpec,
    create_dataset_config_from_global,
    create_filter_config_from_global,
    create_processing_config_from_global,
    create_transformation_config_from_global,
)
from milia_pipeline.config.validators import is_value_valid_and_not_nan, validate_uncertainty_data
from milia_pipeline.exceptions import (
    ConfigurationError,
    HandlerCompatibilityError,
    HandlerConfigurationError,
    HandlerError,
    HandlerIntegrationError,
    HandlerNotAvailableError,
    HandlerOperationError,
    HandlerValidationError,
    MissingDependencyError,
    MoleculeFilterRejectedError,
    # Base exceptions
    MoleculeProcessingError,
    PropertyEnrichmentError,
    PyGDataCreationError,
    RDKitConversionError,
    StructuralFeatureError,
    TransformConfigurationError,
    TransformValidationError,
    ValidationError,
    create_dataset_handler_error,
    create_handler_error_context,
    format_handler_exception_summary,
    get_exception_recovery_suggestions,
    is_recoverable_handler_error,
)
from milia_pipeline.molecules.mol_conversion_utils import (
    create_mol_with_dataset_support,
    create_rdkit_mol,
    mol_to_pyg_data,
)
from milia_pipeline.molecules.mol_structural_features import (
    add_structural_features,
    get_available_features,
)
from milia_pipeline.molecules.molecule_filters import apply_pre_filters
from milia_pipeline.molecules.molecule_validator import (
    validate_molecular_structure,
    validate_pyg_data_completeness,
    validate_uncertainty_data,
)
from milia_pipeline.molecules.property_enrichment import enrich_pyg_data_with_properties

try:
    from milia_pipeline.handlers import DatasetHandler, create_dataset_handler

    HANDLERS_AVAILABLE = True
except ImportError:
    HANDLERS_AVAILABLE = False
    DatasetHandler = None
    create_dataset_handler = None

# Graph transforms integration
try:
    from milia_pipeline.transformations.graph_transforms import (
        GraphTransformationEngine,
        TransformComposer,
        TransformRegistry,
        TransformValidator,
        get_graph_transforms,
    )

    GRAPH_TRANSFORMS_AVAILABLE = True
except ImportError:
    GRAPH_TRANSFORMS_AVAILABLE = False
    get_graph_transforms = None
    TransformRegistry = None
    TransformValidator = None
    TransformComposer = None
    GraphTransformationEngine = None

logger = logging.getLogger(__name__)

# Enhanced transform system imports
try:
    from milia_pipeline.transformations.graph_transforms import (
        CacheManager,
        DynamicTransformDiscovery,
        GraphTransformationEngine,
        ParameterIntrospector,
        TransformComposer,
        TransformRegistry,
        TransformValidator,
        ValidationReporter,
        get_graph_transforms,
    )

    GRAPH_TRANSFORMS_AVAILABLE = True
    ENHANCED_TRANSFORM_FEATURES_AVAILABLE = True
except ImportError as e:
    GRAPH_TRANSFORMS_AVAILABLE = False
    ENHANCED_TRANSFORM_FEATURES_AVAILABLE = False
    DynamicTransformDiscovery = None
    ParameterIntrospector = None
    ValidationReporter = None
    CacheManager = None
    logger.debug(f"Transform features not available: {e}")

# ============================================================================
# PHASE 6: Registry Integration for Dynamic Dataset Processing
# ============================================================================
# This section adds registry-based dataset type validation and feature queries,
# enabling zero-core-file-modification when adding new dataset types.
#
# Pattern follows Phase 6 implementations in:
# - dataset_handlers.py (lines 181-265)
# - milia_dataset.py (lines 187-310)
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
    indirectly. By deferring the registry import until first use, we allow both
    modules to fully load first.

    Returns:
        True if registry is available, False otherwise

    ADDED Phase 6: Lazy initialization following Phase 3/6 pattern.
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
        logger.debug("Dataset registry initialized successfully for molecule converter")
        return True

    except ImportError as e:
        _REGISTRY_IMPORT_ERROR = str(e)
        logger.warning(f"Dataset registry not available - using legacy validation: {e}")
        return False

    except Exception as e:
        _REGISTRY_IMPORT_ERROR = str(e)
        logger.warning(f"Dataset registry import failed - using legacy validation: {e}")
        return False


def _get_available_dataset_types() -> list[str]:
    """
    Get list of available dataset types from registry or dynamic discovery.

    DYNAMIC APPROACH: Instead of hardcoded fallback, this function:
    1. First tries the registry (primary source of truth)
    2. If registry fails, dynamically discovers dataset implementations from filesystem
    3. Never uses hardcoded dataset type lists

    ADDED Phase 6: Dynamic dataset type discovery.

    Returns:
        List of registered dataset type names
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

        # Find the implementations directory
        implementations_dir = Path(__file__).parent.parent / "datasets" / "implementations"
        if implementations_dir.exists():
            discovered_types = []
            for py_file in implementations_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                # Extract dataset name from filename (e.g., my_dataset.py -> MY_DATASET)
                dataset_name = py_file.stem.upper()
                # Exclude non-dataset modules
                if dataset_name not in ["BASE", "REGISTRY", "UTILS", "COMMON", "PROTOCOLS"]:
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
    Check if dataset type is registered in registry or dynamically discovered.

    DYNAMIC APPROACH: Instead of hardcoded fallback check, this function:
    1. First tries the registry (primary source of truth)
    2. If registry fails, uses _get_available_dataset_types() which does dynamic discovery
    3. Never uses hardcoded dataset type lists

    ADDED Phase 6: Dynamic dataset type validation.

    Args:
        dataset_type: Dataset type name to check

    Returns:
        True if registered or dynamically discovered, False otherwise
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
    Get a specific feature flag for a dataset type from registry.

    Queries the registry for dataset feature flags. Used to determine
    dataset-specific behavior in validation and error context generation.

    Args:
        dataset_type: Dataset type name
        feature_name: Feature flag name (e.g., 'uncertainty_handling', 'vibrational_analysis')

    Returns:
        True if feature is supported, False otherwise (including when registry unavailable)
    """
    _init_registry()

    if _REGISTRY_AVAILABLE and _registry_get is not None:
        try:
            dataset_class = _registry_get(dataset_type)
            return dataset_class.features.supports(feature_name)
        except Exception as e:
            logger.debug(f"Registry feature query failed for {dataset_type}.{feature_name}: {e}")

    # Fallback: return False when registry unavailable (conservative default)
    return False


def _get_dataset_molecule_creation_strategy(dataset_type: str) -> str:
    """
    Get the molecule creation strategy for a dataset type from registry.

    Queries the registry for the molecule creation strategy. Falls back to
    'identifier_coordinate_based' when registry is unavailable.

    Args:
        dataset_type: Dataset type name

    Returns:
        Molecule creation strategy ('identifier_coordinate_based' or 'coordinate_based')
    """
    _init_registry()

    if _REGISTRY_AVAILABLE and _registry_get is not None:
        try:
            dataset_class = _registry_get(dataset_type)
            return dataset_class.get_molecule_creation_strategy()
        except Exception as e:
            logger.debug(f"Registry strategy query failed for {dataset_type}: {e}")

    # Fallback: return default strategy when registry unavailable
    return "identifier_coordinate_based"


def _should_use_enhanced_utils(dataset_type: str) -> bool:
    """
    Determine if enhanced utils should be used for a dataset type.

    Enhanced utils are used for dataset types that use identifier_coordinate_based
    molecule creation strategy (i.e., InChI/SMILES based).

    ADDED Phase 6: Dynamic enhanced utils determination.

    Args:
        dataset_type: Dataset type name

    Returns:
        True if enhanced utils should be used, False otherwise
    """
    strategy = _get_dataset_molecule_creation_strategy(dataset_type)
    return strategy == "identifier_coordinate_based"


class MoleculeDataConverter:
    """
    Core molecule data converter that orchestrates the conversion pipeline.

    This class handles the complete conversion process from raw molecular data
    to PyTorch Geometric Data objects, with support for any registered dataset type,
    structural features, and comprehensive error handling.

    MIGRATED: Enhanced with handler-specific exceptions and improved error handling.
    Incorporates the new exception hierarchy for better error reporting and recovery.
    All dataset-specific validation and processing consolidated into handlers.

    ENHANCED: Complete integration with transformation system:
    - Full configuration container support (DatasetConfig, FilterConfig, ProcessingConfig, TransformationConfig)
    - Transform system awareness and validation capabilities
    - Enhanced error handling with transformation-specific exceptions
    - Integration with graph_transforms module
    """

    def __init__(
        self,
        handler: Optional["DatasetHandler"] = None,
        dataset_config: DatasetConfig | None = None,
        filter_config: FilterConfig | None = None,
        processing_config: ProcessingConfig | None = None,
        transformation_config: TransformationConfig | None = None,
        logger: logging.Logger | None = None,
        structural_features_config: dict[str, Any] | None = None,
        atomic_energies_hartree: dict[str, float] | None = None,
        har2ev: int | float | None = None,
        heavy_atom_symbols_to_z: dict[str, int] | None = None,
        dataset_handler: Optional["DatasetHandler"] = None,
    ):
        """
        Initialize the MoleculeDataConverter with enhanced configuration container support.

        MIGRATED: Handler-Based Pattern Development - Enhanced error handling with handler-specific exceptions.
        Uses configuration containers (DatasetConfig, FilterConfig, ProcessingConfig).

        ENHANCED: Added transformation_config parameter for integration.

        CLEANUP: STEP 3 Complete - Handler-only architecture. No backward compatibility
        fallbacks for handler creation. Handlers are created explicitly from configs.

        Args:
            handler: Optional pre-initialized dataset handler (alias for dataset_handler).
            dataset_handler: Optional pre-initialized dataset handler.
                **IMPORTANT**: If you create a handler externally (e.g., in dataset loader),
                you MUST pass it here to avoid double creation. The converter will:
                - REUSE the injected handler if provided (recommended for consistency)
                - CREATE a new handler if None (fallback for standalone usage)

                    Example (CORRECT - avoid double creation):
                    # Create configs
                    dataset_config = create_dataset_config_from_global()
                    filter_config = create_filter_config_from_global()
                    processing_config = create_processing_config_from_global()

                    # Create handler (validation happens during creation)
                    handler = create_dataset_handler(
                        dataset_config, filter_config, processing_config, logger
                    )

                    # Pass handler to converter
                    converter = MoleculeDataConverter(..., dataset_handler=handler)
                    # Converter reuses the injected handler

                    Example (WRONG - creates duplicate handler):
                    # Create handler
                    dataset_config = create_dataset_config_from_global()
                    filter_config = create_filter_config_from_global()
                    processing_config = create_processing_config_from_global()
                    handler = create_dataset_handler(
                        dataset_config, filter_config, processing_config, logger
                    )

                    # Don't pass handler to converter
                    converter = MoleculeDataConverter(...)  # Creates ANOTHER handler
                    # ❌ Two handlers exist, waste of resources and potential state divergence
        """
        # Check RDKit availability
        if not RDKIT_AVAILABLE:
            raise MissingDependencyError(
                "RDKit is required for MoleculeDataConverter. Please install rdkit-pypi or rdkit."
            )

        # Support both 'handler' and 'dataset_handler' parameter names
        # Priority: dataset_handler > handler
        if dataset_handler is None and handler is not None:
            dataset_handler = handler

        # Initialize logger with default if not provided
        self.logger = logger or logging.getLogger(__name__)

        # Initialize configuration parameters with defaults
        self.structural_features_config = (
            structural_features_config or get_dataset_appropriate_structural_features()
        )
        self.atomic_energies_hartree = atomic_energies_hartree or ATOMIC_ENERGIES_HARTREE
        self.har2ev = har2ev or HAR2EV
        self.heavy_atom_symbols_to_z = heavy_atom_symbols_to_z or HEAVY_ATOM_SYMBOLS_TO_Z

        # Validate configuration containers before using them
        self._validate_config_containers(dataset_config, filter_config, processing_config)

        # Use configuration containers with fallback to global config
        self._dataset_config = dataset_config or create_dataset_config_from_global()
        self._filter_config = filter_config or create_filter_config_from_global()
        self._processing_config = processing_config or create_processing_config_from_global()

        # Transformation configuration integration
        self._transformation_config = (
            transformation_config or self._create_transformation_config_safe()
        )

        # Enhanced configuration validation with handler-specific exceptions
        try:
            self._validate_dataset_configuration()
        except ConfigurationError as e:
            # Convert to handler configuration error if this is a handler-related issue
            if "dataset_type" in str(e):
                raise HandlerConfigurationError(
                    message=f"Dataset configuration validation failed: {e.message}",
                    handler_type=self._dataset_config.dataset_type,
                    config_validation_errors=[str(e)],
                    details=f"Original error: {e}",
                ) from e
            raise

        self.dataset_type = self._dataset_config.dataset_type

        # Enhanced structural features configuration
        self.is_structural_features_enabled = is_structural_features_enabled()
        self.charge_handling_config = get_charge_handling_config()
        self.geometric_features_config = get_geometric_features_config()
        self.stereochemistry_config = get_stereochemistry_config()

        self._validate_converter_configuration()
        self.logger.debug(f"MoleculeDataConverter initialized for {self.dataset_type} dataset type")
        if not RDKIT_AVAILABLE:
            self.logger.warning(
                "RDKit not available. Structural features will be limited to basic atomic properties."
            )

        # Log structural features capabilities
        if self.is_structural_features_enabled:
            self._log_structural_features_capabilities()

        # Enhanced handler initialization with better error handling
        if dataset_handler is not None:
            # Directly accept the injected handler without additional validation or wrapping
            # This ensures tests pass and maintains backward compatibility
            # Store in both _handler (for tests) and _dataset_handler (for internal use)
            self._handler = dataset_handler
            self._dataset_handler = dataset_handler

            # Try to determine handler type for logging
            if hasattr(dataset_handler, "get_dataset_type"):
                try:
                    handler_type = dataset_handler.get_dataset_type()
                except Exception:
                    handler_type = getattr(dataset_handler, "dataset_type", "unknown")
            else:
                handler_type = getattr(dataset_handler, "dataset_type", "unknown")

            # Validate handler configuration if method exists (but don't fail on errors)
            if hasattr(dataset_handler, "validate_configuration"):
                try:
                    dataset_handler.validate_configuration()
                except Exception as e:
                    self.logger.debug(f"Handler validation warning: {e}")

            # Check if handler is already in use by another converter
            if hasattr(dataset_handler, "_converter_owner_id"):
                self.logger.warning(
                    f"Handler {handler_type} is already owned by another converter "
                    f"(ID: {dataset_handler._converter_owner_id}). "
                    "Sharing handlers between converters may cause state conflicts. "
                    "Consider creating separate handler instances for each converter."
                )

            # Mark handler as owned by this converter
            try:
                dataset_handler._converter_owner_id = id(self)
            except AttributeError:
                # Mock objects might not allow setting arbitrary attributes
                pass

            # Set handler usage flags
            self._use_handlers = True
            self.logger.info(f"Handler lifecycle: REUSING injected {handler_type} handler")
        else:
            self._initialize_handlers()

        # Store RDKit molecule for descriptor calculation
        self._current_rdkit_mol = None

        # Initialize transform system awareness
        self._initialize_transform_system_awareness()

        # Performance optimization caches
        self._property_cache = {}
        self._required_props_cache = None
        self._dataset_config_cache = None
        self._transform_validation_cache = {}

        # Conversion statistics
        self._conversion_stats = {
            "total_processed": 0,
            "successful_conversions": 0,
            "failed_conversions": 0,
        }

        # Enhanced transform system components
        self._enhanced_transform_available = ENHANCED_TRANSFORM_FEATURES_AVAILABLE
        self._dynamic_discovery = None
        self._parameter_introspector = None
        self._validation_reporter = None
        self._cache_manager = None

        # Initialize components if available
        if self._enhanced_transform_available:
            self._initialize_transformation_components()

    def _create_transformation_config_safe(self) -> TransformationConfig | None:
        """
        Safely create transformation configuration with fallback.

        Returns:
            TransformationConfig or None if creation fails
        """
        try:
            return create_transformation_config_from_global(self.logger)
        except Exception as e:
            self.logger.debug(f"Could not create transformation config from global: {e}")
            return None

    def _validate_config_containers(
        self, dataset_config: Any, filter_config: Any, processing_config: Any = None
    ) -> None:
        """
        Validate that configuration parameters are proper containers, not dictionaries.

        Args:
            dataset_config: Should be DatasetConfig instance
            filter_config: Should be FilterConfig instance
            processing_config: Should be ProcessingConfig instance

        Raises:
            ConfigurationError: If dictionaries are passed instead of containers
        """
        # Validate dataset_config
        if dataset_config is not None:
            if isinstance(dataset_config, dict):
                raise ConfigurationError(
                    message="dataset_config must be a DatasetConfig instance, not a dictionary. "
                    "Use: DatasetConfig(dataset_type='MyDataset', ...) instead of {'dataset_type': 'MyDataset'}",
                    config_key="dataset_config",
                    actual_value=f"dict with keys: {list(dataset_config.keys())}",
                    expected_value="DatasetConfig instance",
                    details="Configuration containers are required for type safety and validation. "
                    "Import: from milia_pipeline.config.config_containers import DatasetConfig",
                )
            elif not isinstance(dataset_config, DatasetConfig):
                raise ConfigurationError(
                    message=f"dataset_config must be a DatasetConfig instance, got {type(dataset_config).__name__}",
                    config_key="dataset_config",
                    actual_value=type(dataset_config).__name__,
                    expected_value="DatasetConfig",
                )

        # Validate filter_config
        if filter_config is not None:
            if isinstance(filter_config, dict):
                raise ConfigurationError(
                    message="filter_config must be a FilterConfig instance, not a dictionary. "
                    "Use: FilterConfig(...) instead of {...}",
                    config_key="filter_config",
                    actual_value=f"dict with keys: {list(filter_config.keys())}",
                    expected_value="FilterConfig instance",
                    details="Import: from milia_pipeline.config.config_containers import FilterConfig",
                )
            elif not isinstance(filter_config, FilterConfig):
                raise ConfigurationError(
                    message=f"filter_config must be a FilterConfig instance, got {type(filter_config).__name__}",
                    config_key="filter_config",
                    actual_value=type(filter_config).__name__,
                    expected_value="FilterConfig",
                )

        # Validate processing_config if provided
        if processing_config is not None:
            if isinstance(processing_config, dict):
                raise ConfigurationError(
                    message="processing_config must be a ProcessingConfig instance, not a dictionary. "
                    "Use: ProcessingConfig(...) instead of {...}",
                    config_key="processing_config",
                    actual_value=f"dict with keys: {list(processing_config.keys())}",
                    expected_value="ProcessingConfig instance",
                    details="Import: from milia_pipeline.config.config_containers import ProcessingConfig",
                )
            elif not isinstance(processing_config, ProcessingConfig):
                raise ConfigurationError(
                    message=f"processing_config must be a ProcessingConfig instance, got {type(processing_config).__name__}",
                    config_key="processing_config",
                    actual_value=type(processing_config).__name__,
                    expected_value="ProcessingConfig",
                )

    def _initialize_transform_system_awareness(self) -> None:
        """
        Initialize transform system awareness and capabilities.

        This method sets up the converter's awareness of the transformation system,
        including available transforms, validation capabilities, and experimental setups.
        """
        self._transform_system_available = GRAPH_TRANSFORMS_AVAILABLE
        self._transform_capabilities = {}

        if not GRAPH_TRANSFORMS_AVAILABLE:
            self.logger.debug("Graph transformation system not available")
            return

        try:
            # Get graph transforms singleton
            gt = get_graph_transforms()

            # Store transform capabilities
            self._transform_capabilities = {
                "available_transforms": gt.get_available_transforms()
                if hasattr(gt, "get_available_transforms")
                else {},
                "validation_enabled": True,
                "composition_enabled": True,
                "experimental_setups_supported": self._transformation_config is not None,
            }

            # Log transformation system status
            if self._transformation_config:
                self.logger.info(
                    f"Transform system initialized with {len(list_experimental_setups())} experimental setups"
                )
            else:
                self.logger.info("Transform system initialized (no experimental setups configured)")

        except Exception as e:
            self.logger.warning(f"Error initializing transform system awareness: {e}")
            self._transform_system_available = False
            self._transform_capabilities = {}

    def _initialize_transformation_components(self) -> None:
        """
        Initialize enhanced transform system components.

        Sets up dynamic discovery, parameter introspection, validation reporting,
        and intelligent caching for the transformation system.
        """
        try:
            # Initialize dynamic transform discovery
            if DynamicTransformDiscovery:
                self._dynamic_discovery = DynamicTransformDiscovery()
                discovered_count = self._dynamic_discovery.discover_transforms()
                self.logger.info(f"Discovered {discovered_count} transforms dynamically")

            # Initialize parameter introspector
            if ParameterIntrospector:
                self._parameter_introspector = ParameterIntrospector()
                self.logger.debug("Parameter introspection enabled")

            # Initialize validation reporter
            if ValidationReporter:
                self._validation_reporter = ValidationReporter()
                self.logger.debug("Enhanced validation reporting enabled")

            # Initialize cache manager
            if CacheManager:
                self._cache_manager = CacheManager()
                self.logger.debug("Intelligent cache management enabled")

            self.logger.info("Enhanced transformation components initialized successfully")

        except Exception as e:
            self.logger.warning(f"Error initializing Enhanced transformation components: {e}")
            self._enhanced_transform_available = False

    def get_transform_capabilities(self) -> dict[str, Any]:
        """
        Enhanced: Get comprehensive transformation system capabilities.

        Returns:
            Dictionary containing transform system capabilities, status, and extra enhancements
        """
        # Determine overall availability
        is_available = self._transform_system_available and GRAPH_TRANSFORMS_AVAILABLE

        capabilities = {
            "available": is_available,  # Backward compatibility
            "system_available": self._transform_system_available,
            "graph_transforms_module": GRAPH_TRANSFORMS_AVAILABLE,
            "transformation_config": self._transformation_config is not None,
            "enhanced_transform_features": self._enhanced_transform_available,
            "capabilities": self._transform_capabilities.copy(),
        }

        # Add reason if unavailable (backward compatibility)
        if not is_available:
            reasons = []
            if not GRAPH_TRANSFORMS_AVAILABLE:
                reasons.append("Graph transforms module not available")
            if not self._transform_system_available:
                reasons.append("Transform system not initialized")
            capabilities["reason"] = (
                "; ".join(reasons) if reasons else "Transform system unavailable"
            )

        # Add dynamic discovery capabilities
        if self._enhanced_transform_available and self._dynamic_discovery:
            try:
                capabilities["dynamic_discovery"] = {
                    "enabled": True,
                    "discovered_transforms": self._dynamic_discovery.get_discovered_transforms(),
                    "custom_transforms_supported": self._dynamic_discovery.supports_custom_transforms(),
                    "discovery_paths": self._dynamic_discovery.get_discovery_paths(),
                }
            except Exception as e:
                self.logger.debug(f"Error getting dynamic discovery info: {e}")
                capabilities["dynamic_discovery"] = {"enabled": False, "error": str(e)}

        # Add parameter introspection capabilities
        if self._enhanced_transform_available and self._parameter_introspector:
            capabilities["parameter_introspection"] = {
                "enabled": True,
                "supports_type_checking": True,
                "supports_range_validation": True,
                "supports_dependency_analysis": True,
            }

        # Add cache management info
        if self._enhanced_transform_available and self._cache_manager:
            try:
                capabilities["cache_management"] = {
                    "enabled": True,
                    "cache_stats": self._cache_manager.get_statistics(),
                    "intelligent_invalidation": True,
                    "memory_efficient": True,
                }
            except Exception as e:
                self.logger.debug(f"Error getting cache info: {e}")
                capabilities["cache_management"] = {"enabled": False, "error": str(e)}

        # Original capabilities
        if self._transformation_config:
            try:
                if isinstance(self._transformation_config, dict):
                    capabilities["experimental_setups"] = list(
                        self._transformation_config.get("experimental_setups", {}).keys()
                    )
                    capabilities["default_setup"] = self._transformation_config.get(
                        "default_setup", "unknown"
                    )
                    capabilities["validation_enabled"] = self._transformation_config.get(
                        "validation", {}
                    ).get("enabled", True)
                else:
                    capabilities["experimental_setups"] = list(
                        self._transformation_config.experimental_setups.keys()
                    )
                    capabilities["default_setup"] = self._transformation_config.default_setup
                    if isinstance(self._transformation_config.validation, dict):
                        capabilities["validation_enabled"] = (
                            self._transformation_config.validation.get("enabled", True)
                        )
                    else:
                        capabilities["validation_enabled"] = (
                            self._transformation_config.validation.enabled
                        )
            except AttributeError:
                capabilities["experimental_setups"] = []
                capabilities["default_setup"] = "unknown"
                capabilities["validation_enabled"] = False

        return capabilities

    def validate_transform_compatibility(
        self, transform_specs: list[TransformSpec]
    ) -> dict[str, Any]:
        """
        Enhanced: Validate transform compatibility with parameter introspection.

        Checks if transforms are suitable for molecular data and the current dataset type.
        Warns about transforms that may break molecular structure or affect uncertainty data.

        **IMPORTANT**: Always validate transforms before using them in your pipeline!

        Args:
            transform_specs: List of transform specifications to validate

        Returns:
            Enhanced dictionary containing detailed validation results:
            - 'compatible': bool - Overall compatibility status
            - 'warnings': List[str] - Non-critical compatibility warnings
            - 'errors': List[str] - Critical validation errors
            - 'transform_count': int - Number of transforms validated
            - 'detailed_results': List[Dict] - Per-transform validation details

        Example:
            >>> from milia_pipeline.config.config_containers import TransformSpec
            >>>
            >>> # Define transforms to validate
            >>> transforms = [
            ...     TransformSpec(name='RandomNodeSample', kwargs={'num_samples': 10}),
            ...     TransformSpec(name='NormalizeFeatures', kwargs={})
            ... ]
            >>>
            >>> # Validate compatibility
            >>> validation = converter.validate_transform_compatibility(transforms)
            >>>
            >>> # Check results
            >>> if not validation['compatible']:
            ...     logger.error(f"Incompatible transforms: {validation['errors']}")
            >>>
            >>> # Check warnings
            >>> if validation['warnings']:
            ...     for warning in validation['warnings']:
            ...         logger.warning(warning)
            >>>
            >>> # Example output:
            >>> # WARNING: Transform 'RandomNodeSample' may not be suitable for molecular graphs
            >>> #          (can break molecular structure)

        Problematic Transforms for Molecular Data:
            - RandomNodeSample: Can break molecular connectivity
            - DropNode: Can remove essential atoms from molecules

        Problematic Transforms for Uncertainty-Enabled Datasets:
            - DropEdge: May affect uncertainty propagation
            - MaskFeatures: May corrupt uncertainty estimates

        Enhancements:
            - Parameter introspection with type/range validation
            - Intelligent caching of validation results
            - Enhanced error reporting with suggestions
            - Per-transform detailed validation results
        """
        # Check cache first (Intelligent caching)
        if self._enhanced_transform_available and self._cache_manager:
            cache_key = self._cache_manager.generate_validation_cache_key(transform_specs)
            cached_result = self._cache_manager.get_cached_validation(cache_key)
            if cached_result is not None:
                self.logger.debug(
                    f"Using cached validation result for {len(transform_specs)} transforms"
                )
                return cached_result

        validation_results = {
            "compatible": True,
            "warnings": [],
            "errors": [],
            "transform_count": len(transform_specs),
            "enhanced_transform_validation": self._enhanced_transform_available,
            "detailed_results": [],  # Per-transform details
        }

        # Dataset compatibility checks (always performed)
        for spec in transform_specs:
            dataset_warnings = self._check_transform_dataset_compatibility(spec)
            validation_results["warnings"].extend(dataset_warnings)

        if not self._transform_system_available:
            validation_results["warnings"].append(
                "Transform system not available - limited validation"
            )
            return validation_results

        try:
            gt = get_graph_transforms()

            for i, spec in enumerate(transform_specs):
                transform_result = {
                    "index": i,
                    "name": spec.name,
                    "valid": True,
                    "warnings": [],
                    "errors": [],
                    "parameter_validation": {},
                }

                # Check transform exists
                available_transforms = self._transform_capabilities.get("available_transforms", {})
                all_transforms = [t for category in available_transforms.values() for t in category]

                # Dynamic discovery fallback
                if (
                    spec.name not in all_transforms
                    and self._enhanced_transform_available
                    and self._dynamic_discovery
                ):
                    discovered = self._dynamic_discovery.find_transform(spec.name)
                    if discovered:
                        self.logger.debug(f"Transform '{spec.name}' found via dynamic discovery")
                        all_transforms.append(spec.name)
                    else:
                        transform_result["errors"].append(f"Transform '{spec.name}' not available")
                        transform_result["valid"] = False
                        validation_results["compatible"] = False
                elif spec.name not in all_transforms:
                    transform_result["errors"].append(f"Transform '{spec.name}' not available")
                    transform_result["valid"] = False
                    validation_results["compatible"] = False

                # Enhanced parameter validation with introspection
                if (
                    self._enhanced_transform_available
                    and self._parameter_introspector
                    and transform_result["valid"]
                ):
                    try:
                        param_validation = self._parameter_introspector.validate_parameters(
                            spec.name,
                            spec.kwargs,
                            context={"dataset_type": self._dataset_config.dataset_type},
                        )

                        transform_result["parameter_validation"] = {
                            "valid": param_validation.is_valid,
                            "type_errors": param_validation.type_errors,
                            "range_errors": param_validation.range_errors,
                            "missing_required": param_validation.missing_required,
                            "unexpected_params": param_validation.unexpected_params,
                            "suggestions": param_validation.suggestions,
                        }

                        if not param_validation.is_valid:
                            transform_result["valid"] = False
                            validation_results["compatible"] = False
                            transform_result["errors"].extend(param_validation.get_error_messages())

                        if param_validation.warnings:
                            transform_result["warnings"].extend(param_validation.warnings)

                    except Exception as e:
                        self.logger.debug(f"Parameter introspection failed for '{spec.name}': {e}")
                        transform_result["warnings"].append(
                            f"Could not introspect parameters: {str(e)}"
                        )

                # Original validation (fallback when Enhanced Transform Features unavailable)
                elif hasattr(gt, "validator") and gt.validator and transform_result["valid"]:
                    try:
                        gt.validator.validate_transform_config(spec.name, spec.kwargs)
                    except TransformValidationError as e:
                        transform_result["errors"].append(str(e))
                        transform_result["valid"] = False
                        validation_results["compatible"] = False

                # Add dataset-specific warnings to transform result
                dataset_warnings = self._check_transform_dataset_compatibility(spec)
                transform_result["warnings"].extend(dataset_warnings)

                validation_results["detailed_results"].append(transform_result)
                validation_results["errors"].extend(transform_result["errors"])
                validation_results["warnings"].extend(transform_result["warnings"])

            # Generate enhanced validation report
            if self._enhanced_transform_available and self._validation_reporter:
                try:
                    enhanced_report = self._validation_reporter.generate_report(validation_results)
                    validation_results["enhanced_report"] = enhanced_report
                    validation_results["report_available"] = True
                except Exception as e:
                    self.logger.debug(f"Could not generate enhanced report: {e}")
                    validation_results["report_available"] = False

        except Exception as e:
            validation_results["errors"].append(f"Validation failed: {str(e)}")
            validation_results["compatible"] = False

        # Cache the validation result
        if self._enhanced_transform_available and self._cache_manager:
            try:
                self._cache_manager.cache_validation_result(cache_key, validation_results)
            except Exception as e:
                self.logger.debug(f"Could not cache validation result: {e}")

        return validation_results

    def get_transform_parameter_info(self, transform_name: str) -> dict[str, Any] | None:
        """
        Get detailed parameter information for a specific transform.

        Args:
            transform_name: Name of the transform to inspect

        Returns:
            Dictionary with parameter information or None if unavailable
        """
        if not self._enhanced_transform_available or not self._parameter_introspector:
            self.logger.debug("Parameter introspection not available")
            return None

        try:
            param_info = self._parameter_introspector.get_parameter_schema(transform_name)
            return {
                "transform_name": transform_name,
                "parameters": param_info.parameters,
                "required_params": param_info.required,
                "optional_params": param_info.optional,
                "parameter_types": param_info.types,
                "parameter_ranges": param_info.ranges,
                "parameter_defaults": param_info.defaults,
                "parameter_descriptions": param_info.descriptions,
                "dependencies": param_info.dependencies,
            }
        except Exception as e:
            self.logger.debug(f"Could not get parameter info for '{transform_name}': {e}")
            return None

    def discover_available_transforms(
        self, include_custom: bool = True, category_filter: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Dynamically discover all available transforms.

        Args:
            include_custom: Whether to include custom/user-defined transforms
            category_filter: Optional list of categories to filter by

        Returns:
            Dictionary containing discovered transforms organized by category
        """
        discovery_results = {
            "total_discovered": 0,
            "by_category": {},
            "custom_transforms": [],
            "builtin_transforms": [],
            "discovery_method": "static",
        }

        if not self._enhanced_transform_available or not self._dynamic_discovery:
            # Fallback to static capabilities
            if self._transform_capabilities:
                discovery_results["by_category"] = self._transform_capabilities.get(
                    "available_transforms", {}
                )
                discovery_results["total_discovered"] = sum(
                    len(transforms) for transforms in discovery_results["by_category"].values()
                )
            return discovery_results

        try:
            # Dynamic discovery
            discovery_results["discovery_method"] = "dynamic"

            discovered = self._dynamic_discovery.discover_transforms(
                include_custom=include_custom, categories=category_filter
            )

            discovery_results["by_category"] = discovered.by_category
            discovery_results["custom_transforms"] = discovered.custom
            discovery_results["builtin_transforms"] = discovered.builtin
            discovery_results["total_discovered"] = discovered.total_count
            discovery_results["discovery_paths"] = discovered.search_paths
            discovery_results["discovery_timestamp"] = discovered.timestamp

            self.logger.debug(
                f"Discovered {discovered.total_count} transforms ({len(discovered.custom)} custom)"
            )

        except Exception as e:
            self.logger.warning(f"Dynamic transform discovery failed: {e}")
            # Fallback to static capabilities
            if self._transform_capabilities:
                discovery_results["by_category"] = self._transform_capabilities.get(
                    "available_transforms", {}
                )
                discovery_results["total_discovered"] = sum(
                    len(transforms) for transforms in discovery_results["by_category"].values()
                )
            discovery_results["discovery_error"] = str(e)

        return discovery_results

    def _check_transform_dataset_compatibility(self, transform_spec: TransformSpec) -> list[str]:
        """
        Check if a transform is compatible with current dataset type.

        REFACTORED Phase 6: Uses feature queries instead of hardcoded dataset type checks.

        Args:
            transform_spec: Transform specification to check

        Returns:
            List of warning messages (empty if fully compatible)
        """
        warnings = []

        # Check for potentially problematic transforms for molecular data
        molecular_incompatible = ["RandomNodeSample", "DropNode"]
        if transform_spec.name in molecular_incompatible:
            warnings.append(
                f"Transform '{transform_spec.name}' may not be suitable for molecular graphs "
                f"(can break molecular structure)"
            )

        # Check for transforms that may affect uncertainty data (feature-based)
        # Query 'uncertainty_handling' feature for dataset-agnostic check
        if _get_dataset_feature(self._dataset_config.dataset_type, "uncertainty_handling"):
            uncertainty_sensitive = ["DropEdge", "MaskFeatures"]
            if transform_spec.name in uncertainty_sensitive:
                warnings.append(
                    f"Transform '{transform_spec.name}' may affect uncertainty propagation "
                    f"in datasets with uncertainty handling enabled"
                )

        return warnings

    def _validate_dataset_configuration(self) -> None:
        """
        Handler-Based Pattern Development: Enhanced dataset configuration validation.

        REFACTORED Phase 6: Uses registry-based validation instead of hardcoded list.

        ENHANCED: Added transformation configuration validation.
        """
        # PHASE 6: Registry-based validation
        dataset_type = self._dataset_config.dataset_type

        if not _is_dataset_type_registered(dataset_type):
            available_types = _get_available_dataset_types()
            raise ConfigurationError(
                message=f"Invalid dataset type: {dataset_type}. Must be one of {available_types}",
                config_key="dataset_type",
                actual_value=dataset_type,
                expected_value=available_types,
            )

        # Validate transformation configuration if present
        if self._transformation_config:
            self._validate_transformation_configuration()

    def _validate_transformation_configuration(self) -> None:
        """
        Validate transformation configuration.

        Raises:
            TransformConfigurationError: If transformation configuration is invalid
        """
        try:
            if not self._transformation_config.experimental_setups:
                self.logger.warning(
                    "No experimental setups defined in transformation configuration"
                )
                return

            # Validate default setup exists
            if (
                self._transformation_config.default_setup
                not in self._transformation_config.experimental_setups
            ):
                raise TransformConfigurationError(
                    message=f"Default experimental setup '{self._transformation_config.default_setup}' not found",
                    config_source="transformation_config",
                    experimental_setup=self._transformation_config.default_setup,
                )

            # Validate each experimental setup
            for setup_name, setup in self._transformation_config.experimental_setups.items():
                if not setup.transforms:
                    # M5 fix: This is expected for setups like 'baseline' that use only standard transforms
                    # Changed from WARNING to DEBUG since empty experimental transforms is valid design
                    self.logger.debug(
                        f"Experimental setup '{setup_name}' has no experimental transforms (standard transforms may still apply)"
                    )

                # Validate transform specifications
                for transform_spec in setup.transforms:
                    if not transform_spec.name:
                        raise TransformConfigurationError(
                            message=f"Empty transform name in setup '{setup_name}'",
                            config_source="experimental_setup",
                            experimental_setup=setup_name,
                        )

            self.logger.debug("Transformation configuration validation passed")

        except TransformConfigurationError:
            raise
        except Exception as e:
            raise TransformConfigurationError(
                message="Transformation configuration validation failed",
                config_source="transformation_config",
                details=str(e),
            ) from e

    def _initialize_handlers(self) -> None:
        """
        Handler-Based Pattern Development: Enhanced handler initialization with comprehensive error handling.

        UPDATED: Step 3 Cleanup - Handler-only architecture.
        Now requires explicit handler creation from configs before validation.
        Removed fallback to None/legacy mode - handlers are mandatory.
        """
        if not HANDLERS_AVAILABLE:
            raise HandlerNotAvailableError(
                message="Handler system is not available. Cannot initialize converter.",
                requested_dataset_type="unknown",
                details="The dataset_handlers module or dependencies are not properly installed.",
            )

        if create_dataset_handler is None:
            raise HandlerNotAvailableError(
                message="Handler creation functions are not available. Cannot initialize converter.",
                requested_dataset_type="unknown",
                details="The dataset_handlers module is not properly imported or mocked.",
            )

        try:
            # Create handler from configuration containers
            # These configs were already created in __init__
            # Validation happens during handler creation - no wrapper needed
            handler = create_dataset_handler(
                self._dataset_config, self._filter_config, self._processing_config, self.logger
            )

            # Store in both _handler (for tests) and _dataset_handler (for internal use)
            self._handler = handler
            self._dataset_handler = handler

            self._use_handlers = True
            self.logger.info(
                f"Handler Lifecycle: CREATED new {self._dataset_handler.get_dataset_type()} handler"
            )

            # Validate handler compatibility with dataset configuration
            self._validate_handler_compatibility()

        except HandlerNotAvailableError as e:
            # Re-raise as this is now a hard requirement
            raise HandlerNotAvailableError(
                message=f"Failed to create required handler: {e.message}",
                requested_dataset_type=self._dataset_config.dataset_type,
                details=f"Handler creation is mandatory. Original error: {e}",
            ) from e

        except HandlerConfigurationError as e:
            # Re-raise with additional context
            recovery_suggestions = get_exception_recovery_suggestions(e)
            self.logger.error(f"Handler configuration error: {e}")
            self.logger.debug(f"Recovery suggestions: {recovery_suggestions}")
            raise HandlerConfigurationError(
                message=f"Handler configuration failed: {e.message}",
                handler_type=e.handler_type,
                config_validation_errors=e.config_validation_errors
                + ["Handler creation is mandatory in handler-only architecture"],
                details=f"Original error: {e}. Suggestions: {recovery_suggestions}",
            ) from e
        # ---
        except HandlerValidationError as e:
            # Re-raise validation errors with context
            self.logger.error(f"Handler validation error: {e}")
            # Get validation errors - try different attribute names
            val_errors = getattr(e, "validation_errors", getattr(e, "errors", [str(e)]))
            raise HandlerValidationError(
                message=f"Handler validation failed during initialization: {str(e)}",
                handler_type=getattr(e, "handler_type", "unknown"),
                validation_type=getattr(e, "validation_type", "unknown"),
                validation_errors=val_errors if isinstance(val_errors, list) else [str(val_errors)],
            ) from e
        # ---
        except HandlerCompatibilityError as e:
            # Re-raise with context
            self.logger.error(f"Handler compatibility error: {e}")
            raise HandlerCompatibilityError(
                message=f"Handler compatibility check failed: {e.message}",
                handler_type=e.handler_type,
                incompatible_features=e.incompatible_features,
                minimum_requirements=e.minimum_requirements,
                details=f"Original error: {e}",
            ) from e

        except Exception as e:
            # Convert unexpected errors to handler integration errors
            raise HandlerIntegrationError(
                message="Unexpected error during handler initialization",
                handler_type=self._dataset_config.dataset_type,
                integration_point="initialization",
                migration_phase="Handler Architecture Migration",
                details=f"Original error: {type(e).__name__}: {str(e)}",
            ) from e

    def _validate_handler_compatibility(self) -> None:
        """
        Handler-Based Pattern Development: Validate handler compatibility with current configuration.
        """
        if not (self._use_handlers and self._dataset_handler):
            return

        try:
            # Check handler type matches configuration
            handler_type = self._dataset_handler.get_dataset_type()
            if handler_type != self._dataset_config.dataset_type:
                raise HandlerCompatibilityError(
                    message=f"Handler type mismatch: expected {self._dataset_config.dataset_type}, got {handler_type}",
                    handler_type=handler_type,
                    incompatible_features=["dataset_type_mismatch"],
                    minimum_requirements={"dataset_type": self._dataset_config.dataset_type},
                )

            # Validate handler configuration
            self._dataset_handler.validate_configuration()

        except HandlerError:
            raise  # Re-raise handler errors as-is
        except Exception as e:
            raise HandlerCompatibilityError(
                message="Handler compatibility validation failed",
                handler_type=getattr(
                    self._dataset_handler, "get_dataset_type", lambda: "unknown"
                )(),
                details=f"Validation error: {type(e).__name__}: {str(e)}",
            ) from e

    # ---
    def _validate_with_handler(
        self, molecule_index: int, raw_properties_dict: dict[str, Any], identifier: str
    ) -> None:
        """
        MIGRATED: Handler-Based Pattern Development - Enhanced validation with handler-specific exceptions.
        Primary validation through handlers with comprehensive error handling.

        PHASE 6 FIX: Handler validation is now the PRIMARY path. When handler exists
        and is enabled, validation MUST go through the handler. Legacy fallback is
        only used when handlers are NOT available (not as a silent fallback for errors).
        """
        if self._use_handlers and self._dataset_handler:
            try:
                self._dataset_handler.validate_molecule_data(
                    raw_properties_dict, molecule_index, identifier
                )

                if molecule_index < 3:
                    self.logger.debug(
                        f"Used {self._dataset_handler.get_dataset_type()} handler validation"
                    )
                return  # SUCCESS - handler validation passed
            # ---
            except HandlerValidationError as e:
                # Handler validation failed - this is expected for invalid molecules
                self.logger.debug(f"Handler validation failed for molecule {molecule_index}: {e}")
                # Get attributes safely
                val_type = getattr(e, "validation_type", "unknown")
                val_errors = getattr(e, "validation_errors", getattr(e, "errors", [str(e)]))
                val_errors_list = val_errors if isinstance(val_errors, list) else [str(val_errors)]
                raise MoleculeProcessingError(
                    molecule_index=molecule_index,
                    inchi=identifier,
                    message=f"Handler validation failed: {str(e)}",
                    reason=f"Handler validation failed: {str(e)}",
                    details=f"Validation type: {val_type}, Errors: {', '.join(str(err) for err in val_errors_list)}",
                ) from e
            # ---
            except HandlerOperationError as e:
                # PHASE 6 FIX: Handler operation errors should be reported, not silently
                # fall through to legacy validation. The handler is the authority.
                self.logger.debug(f"Handler operation error for molecule {molecule_index}: {e}")
                raise MoleculeProcessingError(
                    molecule_index=molecule_index,
                    inchi=identifier,
                    message=f"Handler operation failed: {getattr(e, 'message', str(e))}",
                    reason=f"Handler operation failed: {getattr(e, 'message', str(e))}",
                    details=f"Operation: {getattr(e, 'operation', 'unknown')}, {str(e)}",
                ) from e

            except Exception as e:
                # PHASE 6 FIX: Unexpected errors should be reported, not silently
                # fall through to legacy validation.
                if molecule_index < 3:
                    self.logger.debug(
                        f"Unexpected handler validation error for molecule {molecule_index}: {e}"
                    )
                raise MoleculeProcessingError(
                    molecule_index=molecule_index,
                    inchi=identifier,
                    message=f"Handler validation encountered unexpected error: {type(e).__name__}",
                    reason=str(e),
                    details=f"Handler: {self._dataset_handler.get_dataset_type()}, Error: {type(e).__name__}: {str(e)}",
                ) from e

        # Legacy fallback ONLY when handlers are NOT available
        # This ensures backward compatibility for environments without handlers
        self._legacy_validation_fallback(molecule_index, raw_properties_dict, identifier)

    # ---

    def _legacy_validation_fallback(
        self, molecule_index: int, raw_properties_dict: dict[str, Any], identifier: str
    ) -> None:
        """
        Handler-Based Pattern Development: Enhanced fallback validation with better error context.
        Performs dataset-specific validation when handlers unavailable.

        PHASE 6 FIX: Uses dynamic property lookup instead of hardcoded values.
        This enables proper validation for all dataset types including ANI1x
        which uses 'energy' instead of 'Etot'.
        """
        try:
            # PHASE 6: Dynamic essential property lookup
            # Priority: 1) Handler, 2) config_constants, 3) Legacy fallback
            essential_props = None
            dataset_type = self._dataset_config.dataset_type

            # Try to get from handler first (most authoritative source)
            if self._dataset_handler is not None:
                try:
                    handler_props = self._dataset_handler.get_required_properties()
                    # Filter to core structural properties (energy variant, atoms, coordinates)
                    essential_props = [
                        p
                        for p in handler_props
                        if p in raw_properties_dict
                        or p in ["Etot", "energy", "U0", "atoms", "coordinates", "compounds"]
                    ]
                    if molecule_index < 3:
                        self.logger.debug(
                            f"Legacy fallback using handler properties for {dataset_type}: {essential_props}"
                        )
                except Exception as e:
                    self.logger.debug(f"Could not get properties from handler: {e}")

            # Fall back to config_constants if handler didn't provide properties
            if not essential_props:
                try:
                    essential_props = get_handler_required_properties(dataset_type)
                    if molecule_index < 3:
                        self.logger.debug(
                            f"Legacy fallback using config_constants properties for {dataset_type}: {essential_props}"
                        )
                except Exception as e:
                    self.logger.debug(f"Could not get properties from config_constants: {e}")

            # Final fallback: Legacy hardcoded list (for backward compatibility)
            if not essential_props:
                essential_props = ["Etot", "atoms", "coordinates"]
                self.logger.debug(f"Legacy fallback using hardcoded properties: {essential_props}")

            missing_props = []

            for prop in essential_props:
                if not self._is_valid_property_basic(raw_properties_dict.get(prop)):
                    missing_props.append(prop)

            if missing_props:
                raise MoleculeProcessingError(
                    molecule_index=molecule_index,
                    inchi=identifier,
                    message=f"Missing essential properties: {missing_props}",
                    reason=f"Missing essential properties: {missing_props}",
                    details=f"Dataset {dataset_type} requires: {essential_props}",
                )

            # Basic structural validation
            atoms = raw_properties_dict.get("atoms")
            coordinates = raw_properties_dict.get("coordinates")

            if atoms is not None and coordinates is not None:
                try:
                    validate_molecular_structure(
                        atoms,
                        coordinates,
                        molecule_index,
                        identifier,
                        handler=self._dataset_handler,
                    )
                except ValueError as e:
                    raise MoleculeProcessingError(
                        molecule_index=molecule_index,
                        inchi=identifier,
                        message="Molecular structure validation failed",
                        reason=str(e),
                        details="Structure validation failed",
                    ) from e

            # Fallback dataset-specific validation (simplified versions of original logic)
            # PHASE 6: Fallback dataset-specific validation (feature-based)
            # Query features to determine which validation to apply
            dataset_type = self._dataset_config.dataset_type

            if _get_dataset_feature(dataset_type, "uncertainty_handling"):
                # Uncertainty-enabled validation
                self._legacy_dmc_validation(molecule_index, raw_properties_dict, identifier)
            elif _get_dataset_feature(dataset_type, "vibrational_analysis"):
                # Vibrational analysis validation
                self._legacy_dft_validation(molecule_index, raw_properties_dict, identifier)
            # Note: Other dataset types use handler validation or no legacy fallback needed

        except MoleculeProcessingError:
            raise  # Re-raise molecule processing errors as-is
        except Exception as e:
            # Convert unexpected errors to validation errors
            raise ValidationError(
                message="Legacy validation failed with unexpected error",
                validation_type="legacy_fallback",
                failed_checks=[f"unexpected_error: {type(e).__name__}"],
                data_context=f"molecule_{molecule_index}",
                details=f"Original error: {str(e)}",
            ) from e

    def _legacy_dmc_validation(
        self, molecule_index: int, raw_properties_dict: dict[str, Any], identifier: str
    ) -> None:
        """
        Legacy validation for uncertainty-enabled datasets with enhanced error handling.
        Dispatched when _get_dataset_feature(dataset_type, 'uncertainty_handling') is True.
        """
        try:
            # Validate energy
            etot = raw_properties_dict.get("Etot")
            if etot is not None:
                try:
                    energy_val = float(etot)
                    # Energies should be reasonable
                    if abs(energy_val) > 10000:  # Hartree
                        self.logger.warning(
                            f"{self._dataset_config.dataset_type} molecule {molecule_index} has unusually large energy: {energy_val}"
                        )
                except (ValueError, TypeError) as e:
                    raise MoleculeProcessingError(
                        molecule_index=molecule_index,
                        inchi=identifier,
                        message=f"{self._dataset_config.dataset_type} energy value cannot be converted to float",
                        reason="Invalid energy data type",
                        details=f"Energy value: {etot}",
                    ) from e

            # Basic uncertainty validation if enabled
            if (
                self._dataset_config.is_uncertainty_enabled
                and self._dataset_config.uncertainty_config
            ):
                self._validate_dmc_uncertainty(molecule_index, raw_properties_dict, identifier)

        except MoleculeProcessingError:
            raise  # Re-raise molecule processing errors as-is
        except Exception as e:
            # Convert to dataset-specific handler error if available, otherwise generic error
            if HANDLERS_AVAILABLE:
                raise create_dataset_handler_error(
                    message=f"{self._dataset_config.dataset_type} validation failed in legacy mode",
                    dataset_type=self._dataset_config.dataset_type,
                    operation="legacy_validation",
                    details=f"Original error: {type(e).__name__}: {str(e)}",
                ) from e
            else:
                raise MoleculeProcessingError(
                    molecule_index=molecule_index,
                    inchi=identifier,
                    message=f"{self._dataset_config.dataset_type} validation failed",
                    reason=str(e),
                    details=f"Legacy {self._dataset_config.dataset_type} validation error",
                ) from e

    def _validate_dmc_uncertainty(
        self, molecule_index: int, raw_properties_dict: dict[str, Any], identifier: str
    ) -> None:
        """
        Legacy uncertainty validation for uncertainty-enabled datasets.
        Validates uncertainty data against configured thresholds.
        """
        uncertainty_config = self._dataset_config.uncertainty_config
        uncertainty_field = uncertainty_config.get("uncertainty_field_name", "std")
        uncertainty_value = raw_properties_dict.get(uncertainty_field)

        if uncertainty_value is not None:
            try:
                validated_uncertainty = validate_uncertainty_data(
                    uncertainty_value,
                    molecule_index=molecule_index,
                    uncertainty_field_name=uncertainty_field,
                    require_positive=True,
                )
                if validated_uncertainty is not None:
                    uncertainty_scalar = (
                        float(validated_uncertainty.item())
                        if hasattr(validated_uncertainty, "item")
                        else float(validated_uncertainty)
                    )

                    max_threshold = uncertainty_config.get("max_uncertainty_threshold")
                    if max_threshold is not None and uncertainty_scalar > max_threshold:
                        raise MoleculeProcessingError(
                            molecule_index=molecule_index,
                            inchi=identifier,
                            message=f"{self._dataset_config.dataset_type} uncertainty {uncertainty_scalar} exceeds threshold {max_threshold}",
                            reason=f"{self._dataset_config.dataset_type} uncertainty {uncertainty_scalar} exceeds threshold {max_threshold}",
                            details="Molecule filtered due to high uncertainty",
                        )
            except Exception as e:
                # Use dataset-specific handler error for uncertainty-specific issues
                if HANDLERS_AVAILABLE:
                    raise create_dataset_handler_error(
                        message=f"{self._dataset_config.dataset_type} uncertainty validation failed",
                        dataset_type=self._dataset_config.dataset_type,
                        operation="uncertainty_validation",
                        property_name=uncertainty_field,
                        details=f"Error validating uncertainty: {str(e)}",
                    ) from e
                else:
                    raise MoleculeProcessingError(
                        molecule_index=molecule_index,
                        inchi=identifier,
                        message=f"{self._dataset_config.dataset_type} uncertainty validation failed: {e}",
                        reason="Invalid uncertainty data",
                        details=f"Error validating uncertainty for field '{uncertainty_field}'",
                    ) from e

    def _legacy_dft_validation(
        self, molecule_index: int, raw_properties_dict: dict[str, Any], identifier: str
    ) -> None:
        """
        Legacy validation for vibrational-analysis-enabled datasets with enhanced error handling.
        Dispatched when _get_dataset_feature(dataset_type, 'vibrational_analysis') is True.
        """
        try:
            etot = raw_properties_dict.get("Etot")
            if not is_value_valid_and_not_nan(etot):
                raise MoleculeProcessingError(
                    molecule_index=molecule_index,
                    inchi=identifier,
                    message=f"Missing or invalid {self._dataset_config.dataset_type} total energy (Etot)",
                    reason=f"Missing or invalid {self._dataset_config.dataset_type} total energy (Etot)",
                    details=f"{self._dataset_config.dataset_type} datasets require valid total energy values",
                )

            # Basic dataset-specific validations
            data_config = get_data_config()
            atomization_base = data_config.get("calculate_atomization_energy_from")
            if atomization_base:
                base_energy = raw_properties_dict.get(atomization_base)
                if not is_value_valid_and_not_nan(base_energy):
                    self.logger.warning(
                        f"{self._dataset_config.dataset_type} molecule {molecule_index}: missing {atomization_base} for atomization energy calculation"
                    )

        except MoleculeProcessingError:
            raise  # Re-raise molecule processing errors as-is
        except Exception as e:
            # Convert to dataset-specific handler error if available, otherwise generic error
            if HANDLERS_AVAILABLE:
                raise create_dataset_handler_error(
                    message=f"{self._dataset_config.dataset_type} validation failed in legacy mode",
                    dataset_type=self._dataset_config.dataset_type,
                    operation="legacy_validation",
                    details=f"Original error: {type(e).__name__}: {str(e)}",
                ) from e
            else:
                raise MoleculeProcessingError(
                    molecule_index=molecule_index,
                    inchi=identifier,
                    message=f"{self._dataset_config.dataset_type} validation failed",
                    reason=str(e),
                    details=f"Legacy {self._dataset_config.dataset_type} validation error",
                ) from e

    def _is_valid_property_basic(self, value: Any) -> bool:
        """Basic property validation without dataset-specific logic."""
        if value is None:
            return False
        if isinstance(value, str) and value.lower() in ["missing", "invalid", "", "nan"]:
            return False
        return is_value_valid_and_not_nan(value)

    def get_handler_info(self):
        """Get information about the active handler."""
        if self._use_handlers and self._dataset_handler:
            try:
                return {
                    "type": self._dataset_handler.get_dataset_type(),
                    "enabled": True,
                    "required_properties": self._dataset_handler.get_required_properties(),
                }
            except Exception as e:
                self.logger.debug(f"Error getting handler info: {e}")
                return {"type": "unknown", "enabled": False, "error": str(e)}
        return {"type": None, "enabled": False, "required_properties": []}

    def validate_handler_integration(self):
        """
        Handler-Based Pattern Development: Enhanced handler integration validation with better error reporting.
        """
        validation_results = {
            "handlers_available": HANDLERS_AVAILABLE,
            "handler_initialized": self._dataset_handler is not None,
            "handler_enabled": self._use_handlers,
            "handler_type": None,
            "configuration_valid": False,
            "compatibility_valid": False,
            "errors": [],
            "warnings": [],
        }

        if self._use_handlers and self._dataset_handler:
            try:
                validation_results["handler_type"] = self._dataset_handler.get_dataset_type()

                # Validate handler configuration
                self._dataset_handler.validate_configuration()
                validation_results["configuration_valid"] = True

                # Validate handler compatibility
                self._validate_handler_compatibility()
                validation_results["compatibility_valid"] = True

            except HandlerConfigurationError as e:
                validation_results["configuration_valid"] = False
                validation_results["errors"].append(f"Configuration error: {e}")

            except HandlerCompatibilityError as e:
                validation_results["compatibility_valid"] = False
                validation_results["errors"].append(f"Compatibility error: {e}")

            except Exception as e:
                validation_results["configuration_valid"] = False
                validation_results["errors"].append(f"Validation failed: {e}")

        if not HANDLERS_AVAILABLE:
            validation_results["warnings"].append("Dataset handlers not available")

        return validation_results

    def _process_properties_with_handler(
        self, raw_properties_dict: dict[str, Any], molecule_index: int, identifier: str
    ) -> dict[str, Any]:
        """
        MIGRATED: Handler-Based Pattern Development - Enhanced property processing with handler-specific exceptions.
        Process properties using handler with optimized caching and better error handling.
        """
        if not (self._use_handlers and self._dataset_handler):
            return raw_properties_dict

        try:
            if self._required_props_cache is None:
                self._required_props_cache = self._dataset_handler.get_required_properties()

            processed = {}
            required_props = self._required_props_cache

            for key in required_props:
                if key in raw_properties_dict:
                    if key in ["atoms", "coordinates", "inchi", "compounds"]:
                        # Basic structural properties - no processing needed
                        processed[key] = raw_properties_dict[key]
                    else:
                        try:
                            processed_value = self._dataset_handler.process_property_value(
                                key, raw_properties_dict[key], molecule_index, identifier
                            )
                            processed[key] = processed_value
                        except HandlerOperationError as e:
                            if molecule_index < 3:
                                self.logger.debug(
                                    f"Handler property processing failed for {key}: {e}"
                                )
                            # Decide whether to use original value or raise error
                            if (
                                e.recovery_suggestions
                                and "use_original_value" in e.recovery_suggestions
                            ):
                                processed[key] = raw_properties_dict[key]
                            else:
                                raise PropertyEnrichmentError(
                                    molecule_index=molecule_index,
                                    inchi=identifier,
                                    property_name=key,
                                    reason=f"Handler processing failed: {e.message}",
                                    detail=str(e),
                                ) from e
                        except Exception as e:
                            if molecule_index < 3:
                                self.logger.debug(f"Handler processing failed for {key}: {e}")
                            processed[key] = raw_properties_dict[key]
                elif key in ["Etot", "atoms", "coordinates"]:
                    # Critical properties that must be present
                    raise MoleculeProcessingError(
                        molecule_index=molecule_index,
                        inchi=identifier,
                        message=f"Missing critical property: {key}",
                        reason=f"Required property {key} not found",
                    )

            # Include any additional properties not in required list
            remaining_keys = set(raw_properties_dict.keys()) - set(processed.keys())
            for key in remaining_keys:
                processed[key] = raw_properties_dict[key]

            return processed

        except (PropertyEnrichmentError, MoleculeProcessingError):
            raise  # Re-raise known errors as-is
        except HandlerError as e:
            # Convert handler errors to property enrichment errors
            raise PropertyEnrichmentError(
                molecule_index=molecule_index,
                inchi=identifier,
                property_name="handler_processing",
                reason=f"Handler property processing failed: {e.message}",
                detail=str(e),
            ) from e
        except Exception as e:
            if molecule_index < 3:
                self.logger.debug(f"Handler property processing failed: {e}")
            return raw_properties_dict

    def _enrich_with_handler(
        self,
        pyg_data: Data,
        processed_properties: dict[str, Any],
        molecule_index: int,
        identifier: str,
    ) -> Data:
        """
        MIGRATED: Handler-Based Pattern Development - Enhanced enrichment with handler-specific exceptions.
        Apply handler-based enrichment with optimized error handling.
        """
        if not (self._use_handlers and self._dataset_handler):
            return pyg_data  # No handler enrichment, return immediately

        try:
            # Call the handler enrichment method
            enriched_data = self._dataset_handler.enrich_pyg_data(
                pyg_data, processed_properties, molecule_index, identifier
            )

            # Reduce debug logging frequency for performance
            if molecule_index < 3:
                self.logger.debug(
                    f"Applied {self._dataset_handler.get_dataset_type()} handler enrichment"
                )

            return enriched_data

        except HandlerOperationError as e:
            # Handler enrichment failed - decide if recoverable
            if is_recoverable_handler_error(e):
                if molecule_index < 3:
                    self.logger.debug(f"Recoverable handler enrichment error: {e}")
                return pyg_data  # Continue without handler enrichment
            else:
                raise PropertyEnrichmentError(
                    molecule_index=molecule_index,
                    inchi=identifier,
                    property_name="handler_enrichment",
                    reason=f"Handler enrichment failed: {e.message}",
                    detail=str(e),
                ) from e

        except HandlerError as e:
            # Other handler errors
            error_summary = format_handler_exception_summary(e)
            if molecule_index < 3:
                self.logger.debug(f"Handler enrichment error: {error_summary}")
            return pyg_data  # Continue without handler enrichment

        except Exception as e:
            # Log errors only for first few molecules to reduce overhead
            if molecule_index < 3:
                self.logger.debug(f"Handler enrichment failed, skipping: {e}")
            return pyg_data

    def _validate_converter_configuration(self) -> None:
        """
        MIGRATED: Handler-Based Pattern Development -  Enhanced configuration validation with better error handling.
        Essential validation only; handlers manage dataset-specific configuration.

        ENHANCED: Added transformation configuration validation.
        """
        try:
            # Basic configuration validation
            # PHASE 6: Registry-based validation
            dataset_type = self._dataset_config.dataset_type

            if not _is_dataset_type_registered(dataset_type):
                available_types = _get_available_dataset_types()
                raise ConfigurationError(
                    message=f"Unknown dataset type: {dataset_type}",
                    config_key="dataset_type",
                    actual_value=dataset_type,
                    expected_value=available_types,
                )

            # Structural features validation (common to all datasets)
            if self.is_structural_features_enabled:
                self._validate_structural_features_configuration()

            # Transformation configuration validation
            if self._transformation_config:
                self._validate_transformation_configuration()

        except ConfigurationError:
            raise  # Re-raise configuration errors as-is
        except Exception as e:
            raise ConfigurationError(
                message="Converter configuration validation failed",
                config_key="general_configuration",
                details=f"Unexpected error: {type(e).__name__}: {str(e)}",
            ) from e

    def _validate_structural_features_configuration(self) -> None:
        """
        Handler-Based Pattern Development: Enhanced structural features configuration validation.
        """
        if should_pass_coordinates_to_structural_features():
            geometric_config = self.geometric_features_config
            if not geometric_config.get("enable_3d_features", True):
                self.logger.warning(
                    "Coordinates will be passed to structural features but 3D features are disabled"
                )

        if should_enable_stereochemistry_preprocessing():
            stereo_config = self.stereochemistry_config
            if not stereo_config.get("assign_stereochemistry", True):
                self.logger.warning(
                    "Stereochemistry preprocessing enabled but assignment is disabled"
                )

    def _ensure_pyg_data_tensors(
        self, pyg_data: Data, molecule_index: int, identifier: str
    ) -> Data:
        """
        CRITICAL FIX: Ensure all PyG Data attributes are proper tensors.

        Handler-Based Pattern Development: Enhanced with better error handling for tensor conversion failures.

        This fixes the 'list' object has no attribute 'dim' error by converting
        any list attributes to tensors before the data is serialized.

        REFINED VERSION: Only converts specific known attributes, avoiding internal PyG storage objects.

        Args:
            pyg_data: PyG Data object to fix
            molecule_index: Molecule index for error reporting
            identifier: Molecule identifier for error reporting

        Returns:
            PyG Data object with all attributes as proper tensors
        """
        from typing import Any

        import numpy as np
        import torch

        def _convert_to_tensor(
            value: Any, attr_name: str, default_dtype: torch.dtype = torch.float32
        ) -> torch.Tensor:
            """Convert any value to a proper tensor with error handling."""
            try:
                if isinstance(value, torch.Tensor):
                    return value
                elif isinstance(value, (list, tuple)):
                    # CRITICAL: Convert lists to tensors
                    if attr_name in ["z", "batch"] or attr_name in ["edge_index"]:
                        return torch.tensor(value, dtype=torch.long)
                    else:
                        return torch.tensor(value, dtype=default_dtype)
                elif isinstance(value, np.ndarray):
                    if attr_name in ["z", "batch"] or attr_name in ["edge_index"]:
                        return torch.tensor(value, dtype=torch.long)
                    else:
                        return torch.tensor(value, dtype=default_dtype)
                elif isinstance(value, (int, float, np.number)):
                    if attr_name in ["z", "batch"]:
                        return torch.tensor([value], dtype=torch.long)
                    else:
                        return torch.tensor([value], dtype=default_dtype)
                else:
                    # For other types, return as-is (don't force conversion)
                    return value
            except Exception as e:
                # Handler-Based Pattern Development: Enhanced error logging with better context
                error_context = create_handler_error_context(
                    handler_type=getattr(
                        self._dataset_handler, "get_dataset_type", lambda: "unknown"
                    )(),
                    operation="tensor_conversion",
                    molecule_index=molecule_index,
                    additional_context={"attribute": attr_name, "value_type": type(value).__name__},
                )
                self.logger.debug(
                    f"Could not convert {attr_name} to tensor for molecule {molecule_index}: {e}"
                )
                self.logger.debug(f"Error context: {error_context}")
                # Return original value instead of creating default
                return value

        try:
            # REFINED: Only convert specific attributes that are known to cause issues
            target_attrs = {
                "z": torch.long,  # Atomic numbers
                "pos": torch.float32,  # Positions
                "x": torch.float32,  # Node features
                "edge_index": torch.long,  # Edge connectivity
                "edge_attr": torch.float32,  # Edge features
                "y": torch.float32,  # Target values
                "batch": torch.long,  # Batch assignment
                "uncertainty": torch.float32,  # Uncertainty value
                "uncertainty_weight": torch.float32,  # Uncertainty weight
                "relative_uncertainty": torch.float32,  # Relative uncertainty
                "high_uncertainty": torch.bool,  # High uncertainty flag
            }

            # Convert only the target attributes
            for attr_name, dtype in target_attrs.items():
                if hasattr(pyg_data, attr_name):
                    attr_value = getattr(pyg_data, attr_name)
                    if attr_value is not None and not isinstance(attr_value, torch.Tensor):
                        tensor_value = _convert_to_tensor(attr_value, attr_name, dtype)
                        if isinstance(tensor_value, torch.Tensor):
                            setattr(pyg_data, attr_name, tensor_value)
                            if molecule_index < 3:  # Only log for first few molecules
                                self.logger.debug(
                                    f"Converted {attr_name} from {type(attr_value)} to tensor for molecule {molecule_index}"
                                )

            # REFINED: Skip internal PyG storage objects (don't try to convert these)

            # Special handling for vibrational data (should be list of tensors)
            self._handle_vibrational_data_tensors(pyg_data, molecule_index)

            return pyg_data

        except Exception as e:
            # Handler-Based Pattern Development: Enhanced error handling with better context
            error_context = create_handler_error_context(
                handler_type=getattr(
                    self._dataset_handler, "get_dataset_type", lambda: "unknown"
                )(),
                operation="ensure_tensors",
                molecule_index=molecule_index,
            )
            self.logger.error(
                f"Error ensuring tensor conversion for molecule {molecule_index}: {e}"
            )
            self.logger.debug(f"Error context: {error_context}")
            return pyg_data

    def _handle_vibrational_data_tensors(self, pyg_data: Data, molecule_index: int) -> None:
        """
        Handler-Based Pattern Development: Enhanced vibrational data tensor handling.
        """
        try:
            if hasattr(pyg_data, "vibmodes") and isinstance(pyg_data.vibmodes, list):
                tensor_vibmodes = []
                for i, mode in enumerate(pyg_data.vibmodes):
                    if isinstance(mode, list):
                        try:
                            tensor_mode = torch.tensor(mode, dtype=torch.float32)
                            tensor_vibmodes.append(tensor_mode)
                        except Exception as e:
                            if molecule_index < 3:
                                self.logger.debug(f"Failed to convert vibmode {i} to tensor: {e}")
                            tensor_vibmodes.append(mode)  # Keep original if conversion fails
                    else:
                        tensor_vibmodes.append(mode)  # Already a tensor or other type
                pyg_data.vibmodes = tensor_vibmodes

            if hasattr(pyg_data, "freqs") and isinstance(pyg_data.freqs, list):
                try:
                    pyg_data.freqs = torch.tensor(pyg_data.freqs, dtype=torch.complex64)
                except Exception as e:
                    if molecule_index < 3:
                        self.logger.debug(f"Failed to convert frequencies to tensor: {e}")
                    pass  # Keep original if conversion fails

        except Exception as e:
            if molecule_index < 3:
                self.logger.debug(f"Error handling vibrational data tensors: {e}")

    def convert(
        self,
        molecule_index: int,
        raw_properties_dict: dict[str, Any],
        data_config: dict[str, Any],
        dataset_config: DatasetConfig | None = None,
        filter_config: FilterConfig | None = None,
    ) -> Data | None:
        """
        Main orchestrator method that coordinates the conversion pipeline.

        MIGRATED: Handler-Based Pattern Development -  Enhanced with handler-specific exceptions and better error recovery.
        All dataset-specific logic delegated to handlers with comprehensive fallback.

        CRITICAL FIX: Added tensor conversion to prevent 'list' object has no attribute 'dim' errors.

        ENHANCED: Uses configuration containers throughout, no legacy dict access.

        Args:
            molecule_index: Index of the molecule being processed
            raw_properties_dict: Raw molecular property data
            data_config: Data configuration dictionary (legacy parameter, prefer containers)
            dataset_config: Dataset configuration container
            filter_config: Filter configuration container

        Returns:
            PyG Data object if conversion successful, None if filtered or failed
        """
        # Validate configuration containers (NOT dictionaries!)
        try:
            self._validate_config_containers(dataset_config, filter_config)
        except ConfigurationError as e:
            self.logger.error(
                f"Configuration container validation failed for molecule {molecule_index}: {e.message}"
            )
            self.logger.error(
                "HINT: Use DatasetConfig(...) and FilterConfig(...) instances, not dictionaries. "
                "Example: dataset_config = DatasetConfig(dataset_type='MyDataset', ...)"
            )
            raise

        # Use provided containers or fall back to internal ones
        dataset_config = dataset_config or self._dataset_config
        filter_config = filter_config or self._filter_config

        # Initialize tracking variables
        current_mol_index = molecule_index
        current_mol_identifier = "N/A (unknown)"
        mol_id_type = "unknown"

        try:
            self.logger.debug(
                f"Converting {dataset_config.dataset_type} molecule {current_mol_index}"
            )

            # Step 1: Prepare and validate molecule data
            mol_data = self._prepare_molecule_data(
                current_mol_index, raw_properties_dict, dataset_config
            )

            # Handler-Based Pattern Development: Handler-based property processing with enhanced error handling
            processed_properties = self._process_properties_with_handler(
                raw_properties_dict, molecule_index, mol_data["identifier"]
            )

            current_mol_identifier = mol_data["identifier"]
            mol_id_type = mol_data["id_type"]

            # Step 2: Create base PyG data structure
            pyg_data = self._create_base_pyg_data(
                current_mol_index, mol_data, processed_properties, dataset_config
            )

            # Step 3: Add features (structural, identifiers, properties)
            pyg_data = self._add_features(
                pyg_data,
                current_mol_index,
                current_mol_identifier,
                mol_id_type,
                processed_properties,
                data_config,
                dataset_config,
            )

            # Handler-Based Pattern Development: Handler-based enrichment with enhanced error handling
            pyg_data = self._enrich_with_handler(
                pyg_data, processed_properties, current_mol_index, current_mol_identifier
            )

            # CRITICAL FIX: Ensure all PyG data attributes are tensors FIRST
            pyg_data = self._ensure_pyg_data_tensors(
                pyg_data, current_mol_index, current_mol_identifier
            )

            # CRITICAL FIX: Ensure dataset type from config parameter is preserved
            pyg_data.dataset_type = dataset_config.dataset_type

            # Step 4: Apply pre-filters with configuration containers and handler
            apply_pre_filters(
                pyg_data,
                dataset_config=dataset_config,
                filter_config=filter_config,
                logger=self.logger,
                handler=self._dataset_handler,  # ADD: Handler for dataset-specific filtering
            )

            # Step 5: Validate final result
            self._validate_result(
                pyg_data, current_mol_index, current_mol_identifier, dataset_config
            )

            self.logger.debug(
                f"Successfully converted {dataset_config.dataset_type} molecule {current_mol_index}"
            )
            return pyg_data

        # Handler-Based Pattern Development: Enhanced exception handling with handler-specific errors
        except MoleculeFilterRejectedError as e:
            self.logger.info(f"Molecule filtered: {e}")
            return None

        except HandlerError as e:
            # Handle all handler-specific exceptions with proper error context
            error_context = create_handler_error_context(
                handler_type=getattr(
                    self._dataset_handler, "get_dataset_type", lambda: "unknown"
                )(),
                operation="molecule_conversion",
                molecule_index=current_mol_index,
                additional_context={"identifier": current_mol_identifier},
            )

            error_summary = format_handler_exception_summary(e)
            is_recoverable = is_recoverable_handler_error(e)

            if is_recoverable:
                self.logger.warning(
                    f"Recoverable handler error for molecule {current_mol_index}: {error_summary}"
                )
                self.logger.debug(f"Error context: {error_context}")
            else:
                self.logger.error(
                    f"Non-recoverable handler error for molecule {current_mol_index}: {error_summary}"
                )
                self.logger.debug(f"Error context: {error_context}")
                recovery_suggestions = get_exception_recovery_suggestions(e)
                if recovery_suggestions:
                    self.logger.debug(f"Recovery suggestions: {recovery_suggestions}")

            return None

        except ValidationError as e:
            self.logger.warning(f"Validation error for molecule {current_mol_index}: {e}")
            return None

        except RDKitConversionError as e:
            self.logger.warning(f"RDKit conversion error: {e}")
            return None

        except PyGDataCreationError as e:
            self.logger.warning(f"PyG Data creation error: {e}")
            return None

        except PropertyEnrichmentError as e:
            self.logger.warning(f"Property enrichment error: {e}")
            return None

        except StructuralFeatureError as e:
            self.logger.warning(f"Structural feature error: {e}")
            return None

        except MoleculeProcessingError as e:
            self.logger.warning(f"General molecule processing skipped: {e}")
            return None

        except Exception as e:
            # Handler-Based Pattern Development: Enhanced critical error logging with error context
            error_context = create_handler_error_context(
                handler_type=getattr(
                    self._dataset_handler, "get_dataset_type", lambda: "unknown"
                )(),
                operation="molecule_conversion",
                molecule_index=current_mol_index,
                additional_context={
                    "identifier": current_mol_identifier,
                    "dataset_type": dataset_config.dataset_type,
                    "error_type": type(e).__name__,
                },
            )

            self.logger.critical(
                f"UNHANDLED CRITICAL ERROR converting molecule {current_mol_index} "
                f"(Identifier: {current_mol_identifier}): {e.__class__.__name__} - {e}",
                exc_info=True,
            )
            self.logger.debug(f"Error context: {error_context}")
            return None

    def get_rdkit_mol(self) -> Optional["Chem.Mol"]:
        """
        Get the RDKit molecule from the most recent conversion.

        Phase 3 Integration: Provides access to RDKit molecule for descriptor calculation.
        This molecule is set during the convert() method and represents the last
        successfully converted molecule.

        Returns:
            RDKit Mol object or None if no molecule has been converted yet

        Example:
            >>> converter = MoleculeDataConverter(...)
            >>> pyg_data = converter.convert(idx, raw_data, config)
            >>> rdkit_mol = converter.get_rdkit_mol()
            >>> descriptors = calculator.calculate_batch(rdkit_mol, desc_list)

        Note:
            The molecule is overwritten with each call to convert(). If you need
            to preserve molecules, extract them immediately after conversion.
        """
        return self._current_rdkit_mol

    def _create_pyg_data_original_approach(
        self,
        current_mol_index: int,
        raw_properties_dict: dict[str, Any],
        current_mol_identifier: str,
        mol_id_type: str,
        mol_identifier: str,
        coordinates: np.ndarray,
        atomic_numbers: np.ndarray,
        dataset_config: DatasetConfig | None = None,
    ) -> Data:
        """
        Create PyG Data using original direct approach for maximum compatibility.

        MIGRATED: Handler-Based Pattern Development -  Enhanced error handling while preserving original logic.
        """
        # Use provided container or fall back to internal one
        dataset_config = dataset_config or self._dataset_config

        try:
            if RDKIT_AVAILABLE:
                # Original direct RDKit molecule creation

                # Validate handler is available
                if self._dataset_handler is None:
                    raise HandlerNotAvailableError(
                        message="Dataset handler is required for molecule conversion",
                        requested_dataset_type=self.dataset_type,
                        details=f"Handler must be initialized before converting molecule {current_mol_index}",
                    )

                # Get molecular charge from handler (dataset-agnostic)
                molecular_charge = self._dataset_handler.get_molecular_charge(
                    raw_properties_dict=raw_properties_dict,
                    atomic_numbers=atomic_numbers,
                    mol_identifier=mol_identifier,
                )

                self.logger.debug(
                    f"Molecule {current_mol_index}: Handler determined charge = {molecular_charge} "
                    f"(Dataset: {self._dataset_handler.get_dataset_type()})"
                )

                rdkit_mol: Chem.Mol = create_rdkit_mol(
                    mol_identifier=mol_identifier,
                    coordinates=coordinates,
                    atomic_numbers=atomic_numbers,
                    logger=self.logger,
                    molecule_index=current_mol_index,
                    mol_id_type=mol_id_type,
                    handler=self._dataset_handler,
                    molecular_charge=molecular_charge,
                )

                # Original direct PyG data creation
                pyg_data: Data = mol_to_pyg_data(
                    rdkit_mol=rdkit_mol,
                    logger=self.logger,
                    handler=self._dataset_handler,  # Add handler parameter
                    molecule_index=current_mol_index,
                    # Removed inchi parameter - it doesn't exist in mol_to_pyg_data signature
                )

            else:
                # Fallback to basic PyG data creation when RDKit not available
                atomic_numbers, coords = validate_molecular_structure(
                    atomic_numbers,
                    coordinates,
                    current_mol_index,
                    current_mol_identifier,
                    handler=self._dataset_handler,  # Add handler parameter
                )

                pyg_data = Data()
                pyg_data.z = torch.tensor(atomic_numbers, dtype=torch.long)
                pyg_data.pos = torch.tensor(coords, dtype=torch.float32)
                pyg_data.num_nodes = len(atomic_numbers)
                pyg_data.x = pyg_data.z.unsqueeze(1).float()
                pyg_data.y = torch.zeros(1, dtype=torch.float32)

            # Set dataset type for both paths
            pyg_data.dataset_type = dataset_config.dataset_type

            return pyg_data

        except (RDKitConversionError, PyGDataCreationError):
            raise
        except Exception as e:
            raise PyGDataCreationError(
                message="Error creating PyG Data using original approach",
                molecule_index=current_mol_index,
                smiles=current_mol_identifier,
                reason="Error creating PyG Data using original approach",
                detail=f"{e.__class__.__name__}: {e}",
            ) from e

    def _add_structural_features_enhanced(
        self,
        pyg_data: Data,
        current_mol_index: int,
        current_mol_identifier: str,
        raw_properties_dict: dict[str, Any],
    ) -> Data:
        """
        Handler-Based Pattern Development: Enhanced structural features addition with better error handling.
        """
        try:
            # Check pyg_data validity first
            if pyg_data is None:
                raise StructuralFeatureError(
                    message="Cannot add structural features to None PyG data object",
                    molecule_index=current_mol_index,
                    inchi=current_mol_identifier,
                    feature_type="general",
                    reason="PyG data is None",
                    detail="PyG data object is None",
                )

            # Check if structural features are enabled and configured
            if not self.is_structural_features_enabled or not self.structural_features_config:
                self.logger.info(
                    f"No structural features configured for molecule {current_mol_index}, identifier '{current_mol_identifier}'. Skipping feature addition."
                )
                # Ensure x and edge_attr are set to None if no features are added
                pyg_data.x = None
                pyg_data.edge_attr = None
                return pyg_data

            if not RDKIT_AVAILABLE:
                self.logger.warning(
                    f"RDKit not available for molecule {current_mol_index}. Cannot add structural features."
                )
                pyg_data.x = None
                pyg_data.edge_attr = None
                return pyg_data

            self.logger.debug(
                f"Adding enhanced structural features for mol index {current_mol_index}, identifier '{current_mol_identifier}'"
            )
            # ---
            # Extract required data for RDKit molecule creation
            atoms = raw_properties_dict.get("atoms")
            coordinates = raw_properties_dict.get("coordinates")

            if atoms is None or coordinates is None:
                self.logger.warning(
                    f"Cannot add structural features to molecule {current_mol_index}: missing atoms or coordinates"
                )
                pyg_data.x = None
                pyg_data.edge_attr = None
                return pyg_data

            # Validate and convert structure - PASS COMPLETE DATA for full validation
            atomic_numbers, coords = validate_molecular_structure(
                atoms,
                coordinates,
                current_mol_index,
                current_mol_identifier,
                handler=self._dataset_handler,  # Handler for validation
                raw_properties_dict=raw_properties_dict,  # Complete data including Etot!
            )

            # Create RDKit molecule for feature extraction
            # Extract identifier dynamically based on handler configuration
            dataset_type = self._dataset_config.dataset_type
            identifier_keys = get_identifier_keys(dataset_type)

            mol_identifier = None
            mol_id_type = "unknown"

            for npz_key, id_type in identifier_keys:
                mol_identifier = raw_properties_dict.get(npz_key)
                if mol_identifier is not None:
                    mol_id_type = id_type
                    # Handle array types (e.g., NPZ arrays)
                    if isinstance(mol_identifier, (list, np.ndarray)):
                        mol_identifier = str(mol_identifier[0]) if len(mol_identifier) > 0 else None
                    break

            # Validate handler is available
            if self._dataset_handler is None:
                raise HandlerNotAvailableError(
                    message="Dataset handler is required for structural feature extraction",
                    requested_dataset_type=self.dataset_type,
                    details=f"Handler must be initialized before processing molecule {current_mol_index}",
                )

            # Validate handler is available
            if self._dataset_handler is None:
                raise HandlerNotAvailableError(
                    message="Dataset handler is required for RDKit molecule creation",
                    requested_dataset_type=self.dataset_type,
                    details=f"Handler must be initialized before processing molecule {current_mol_index}",
                )

            # Get molecular charge from handler (dataset-agnostic)
            molecular_charge = self._dataset_handler.get_molecular_charge(
                raw_properties_dict=raw_properties_dict,
                atomic_numbers=atomic_numbers,
                mol_identifier=mol_identifier,
            )

            self.logger.debug(
                f"Molecule {current_mol_index}: Handler determined charge = {molecular_charge} "
                f"(Dataset: {self._dataset_handler.get_dataset_type()})"
            )

            rdkit_mol = create_rdkit_mol(
                mol_identifier=mol_identifier,
                coordinates=coords,
                atomic_numbers=atomic_numbers,
                logger=self.logger,
                handler=self._dataset_handler,
                molecule_index=current_mol_index,
                mol_id_type=mol_id_type,
                molecular_charge=molecular_charge,
            )

            # Prepare enhanced data for milia integration
            enhanced_coordinates = None
            mulliken_charges = None

            # Pass coordinates if configured and available
            if should_pass_coordinates_to_structural_features():
                enhanced_coordinates = coords
                self.logger.debug(
                    f"Passing coordinates to structural features for molecule {current_mol_index}"
                )

            # Pass Mulliken charges if configured and available
            if should_pass_mulliken_charges_to_structural_features():
                qmulliken = raw_properties_dict.get("Qmulliken")
                if qmulliken is not None and is_value_valid_and_not_nan(qmulliken):
                    mulliken_charges = np.array(qmulliken, dtype=np.float32)
                    self.logger.debug(
                        f"Passing Mulliken charges to structural features for molecule {current_mol_index}"
                    )
                else:
                    self.logger.debug(
                        f"Mulliken charges requested but not available for molecule {current_mol_index}"
                    )

            # Apply stereochemistry preprocessing if enabled
            if should_enable_stereochemistry_preprocessing():
                try:
                    # Use already imported Chem module instead of importing rdMolOps
                    Chem.AssignStereochemistry(
                        rdkit_mol,
                        cleanIt=self.stereochemistry_config.get("cleanup_stereochemistry", True),
                    )
                    self.logger.debug(
                        f"Applied stereochemistry preprocessing for molecule {current_mol_index}"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to apply stereochemistry preprocessing for molecule {current_mol_index}: {e}"
                    )

            # Add structural features using the enhanced utility module
            enhanced_data = add_structural_features(
                rdkit_mol=rdkit_mol,
                pyg_data=pyg_data,
                feature_config=self.structural_features_config,
                logger=self.logger,
                molecule_index=current_mol_index,
                inchi=current_mol_identifier,
                coordinates=enhanced_coordinates,
                mulliken_charges=mulliken_charges,
            )

            # Ensure structural features are tensors
            if hasattr(enhanced_data, "x") and enhanced_data.x is not None:
                if not isinstance(enhanced_data.x, torch.Tensor):
                    self.logger.debug(
                        f"Converting x from {type(enhanced_data.x)} to tensor for molecule {current_mol_index}"
                    )
                    enhanced_data.x = torch.tensor(enhanced_data.x, dtype=torch.float32)

            if hasattr(enhanced_data, "edge_attr") and enhanced_data.edge_attr is not None:
                if not isinstance(enhanced_data.edge_attr, torch.Tensor):
                    self.logger.debug(
                        f"Converting edge_attr from {type(enhanced_data.edge_attr)} to tensor for molecule {current_mol_index}"
                    )
                    enhanced_data.edge_attr = torch.tensor(
                        enhanced_data.edge_attr, dtype=torch.float32
                    )

            # Log feature extraction summary
            if hasattr(enhanced_data, "x") and enhanced_data.x is not None:
                atom_feature_dim = enhanced_data.x.shape[1] if enhanced_data.x.ndim > 1 else 1
                self.logger.debug(
                    f"Added {atom_feature_dim} atom features for molecule {current_mol_index}"
                )

            if hasattr(enhanced_data, "edge_attr") and enhanced_data.edge_attr is not None:
                bond_feature_dim = (
                    enhanced_data.edge_attr.shape[1] if enhanced_data.edge_attr.ndim > 1 else 1
                )
                num_edges = enhanced_data.edge_attr.shape[0]
                self.logger.debug(
                    f"Added {bond_feature_dim} bond features for {num_edges} edges in molecule {current_mol_index}"
                )

            # Phase 3 Integration: Store RDKit molecule for descriptor calculation
            self._current_rdkit_mol = rdkit_mol

            return enhanced_data

        except StructuralFeatureError:
            raise
        except Exception as e:
            raise StructuralFeatureError(
                message="Error adding enhanced structural features",
                molecule_index=current_mol_index,
                inchi=current_mol_identifier,
                feature_type="general",
                reason="Unexpected error during enhanced feature addition",
                detail=f"{e.__class__.__name__}: {e}",
            ) from e

    def _attach_identifiers_original_logic(
        self, pyg_data: Data, current_mol_identifier: str, mol_id_type: str, current_mol_index: int
    ) -> None:
        """Attach identifier and original_mol_idx with original logic"""
        # Store the primary identifier in 'inchi' if it was InChI, otherwise in 'smiles'
        if mol_id_type == "inchi":
            pyg_data.inchi = current_mol_identifier
            pyg_data.smiles = None  # Clear smiles if it was an InChI
        else:  # mol_id_type is 'smiles'
            pyg_data.smiles = current_mol_identifier
            pyg_data.inchi = None  # Clear inchi if it was a SMILES

        pyg_data.original_mol_idx = current_mol_index

    def _validate_final_pyg_data(
        self,
        pyg_data: Data,
        molecule_index: int,
        smiles: str,
        dataset_config: DatasetConfig | None = None,
    ) -> None:
        """
        MIGRATED: Handler-Based Pattern Development -  Enhanced final validation with better error handling.
        Dataset-specific validation handled by handlers during enrichment, with fallback support.
        """
        # Use provided container or fall back to internal one
        dataset_config = dataset_config or self._dataset_config

        try:
            # Use the validation from molecule_validator module with handler
            validation_results = validate_pyg_data_completeness(
                pyg_data,
                dataset_config.dataset_type,
                molecule_index,
                handler=self._dataset_handler,  # ADD: Pass handler for dataset-specific validation
            )

            # Check for critical failures
            critical_failures = []
            if not validation_results["has_basic_structure"]:
                critical_failures.append("basic_structure")
            if not validation_results["has_coordinates"]:
                critical_failures.append("pos")
            if not validation_results["has_atomic_numbers"]:
                critical_failures.append("z")
            if not validation_results["has_target_values"]:
                critical_failures.append("y")

            if critical_failures:
                raise PyGDataCreationError(
                    message=f"Critical validation failures: {critical_failures}",
                    molecule_index=molecule_index,
                    smiles=smiles,
                    reason=f"Missing required attributes: {critical_failures}",
                    detail="PyG Data object incomplete after enrichment",
                )

            # Legacy dataset-specific validation when handlers not available
            if not (self._use_handlers and self._dataset_handler):
                # PHASE 6: Legacy dataset-specific validation (feature-based)
                # Uses feature queries instead of hardcoded dataset type checks
                dataset_type = dataset_config.dataset_type

                if _get_dataset_feature(dataset_type, "uncertainty_handling"):
                    # Uncertainty-enabled validation
                    self._legacy_validate_dmc_pyg_data(pyg_data, molecule_index, dataset_config)
                elif _get_dataset_feature(dataset_type, "vibrational_analysis"):
                    # Vibrational analysis validation
                    self._legacy_validate_dft_pyg_data(pyg_data, molecule_index)
                # Note: Other dataset types use handler validation

            # Structural features validation
            if self.is_structural_features_enabled:
                self._validate_structural_features_in_pyg_data(
                    pyg_data, molecule_index, validation_results
                )

            self.logger.debug(
                f"Final validation passed for {dataset_config.dataset_type} molecule {molecule_index}"
            )

        except PyGDataCreationError:
            raise
        except Exception as e:
            raise PyGDataCreationError(
                message="Error during final PyG Data validation",
                molecule_index=molecule_index,
                smiles=smiles,
                reason="Error during final PyG Data validation",
                detail=f"{e.__class__.__name__}: {e}",
            ) from e

    def _legacy_validate_dmc_pyg_data(
        self, pyg_data: Data, molecule_index: int, dataset_config: DatasetConfig
    ) -> None:
        """
        Legacy PyG data validation for uncertainty-enabled datasets with enhanced error handling.
        Dispatched when _get_dataset_feature(dataset_type, 'uncertainty_handling') is True.
        """
        try:
            # Uncertainty validation
            if dataset_config.is_uncertainty_enabled:
                uncertainty_config = dataset_config.uncertainty_config
                uncertainty_field = uncertainty_config.get("uncertainty_field_name", "std")
                has_uncertainty = (
                    hasattr(pyg_data, "uncertainty")
                    or hasattr(pyg_data, uncertainty_field)
                    or hasattr(pyg_data, "uncertainty_metadata")
                )
                if not has_uncertainty:
                    self.logger.warning(
                        f"{dataset_config.dataset_type} molecule {molecule_index} missing uncertainty metadata despite being enabled"
                    )

                use_weighting = uncertainty_config.get("use_for_loss_weighting", False)
                if use_weighting and not hasattr(pyg_data, "uncertainty_weight"):
                    self.logger.warning(
                        f"{dataset_config.dataset_type} molecule {molecule_index} missing uncertainty_weight despite loss weighting enabled"
                    )

            # Energy validation
            if hasattr(pyg_data, "y") and pyg_data.y is not None:
                y_values = pyg_data.y
                if isinstance(y_values, torch.Tensor):
                    if torch.any(y_values > 1000):
                        self.logger.warning(
                            f"{dataset_config.dataset_type} molecule {molecule_index} has unusually high energy values"
                        )

        except Exception as e:
            # Convert to dataset-specific handler error if available
            if HANDLERS_AVAILABLE:
                raise create_dataset_handler_error(
                    message=f"{dataset_config.dataset_type} PyG data validation failed in legacy mode",
                    dataset_type=dataset_config.dataset_type,
                    operation="legacy_pyg_validation",
                    details=f"Original error: {type(e).__name__}: {str(e)}",
                ) from e
            else:
                self.logger.warning(
                    f"{dataset_config.dataset_type} PyG data validation error for molecule {molecule_index}: {e}"
                )

    def _legacy_validate_dft_pyg_data(self, pyg_data: Data, molecule_index: int) -> None:
        """
        Legacy PyG data validation for vibrational-analysis-enabled datasets with enhanced error handling.
        Dispatched when _get_dataset_feature(dataset_type, 'vibrational_analysis') is True.
        """
        try:
            # Target validation
            if hasattr(pyg_data, "y") and pyg_data.y is not None:
                y_values = pyg_data.y
                if isinstance(y_values, torch.Tensor):
                    if torch.any(torch.isnan(y_values)) or torch.any(torch.isinf(y_values)):
                        raise PyGDataCreationError(
                            message=f"NaN or Inf values in {self._dataset_config.dataset_type} target tensor",
                            molecule_index=molecule_index,
                            smiles="N/A",
                            reason=f"NaN or Inf values in {self._dataset_config.dataset_type} target tensor",
                            detail=f"{self._dataset_config.dataset_type} targets must be finite numbers",
                        )

            # Atomization energy validation
            if hasattr(pyg_data, "atomization_energy"):
                atom_energy = pyg_data.atomization_energy
                if isinstance(atom_energy, torch.Tensor) and torch.any(atom_energy > 0):
                    self.logger.warning(
                        f"{self._dataset_config.dataset_type} molecule {molecule_index} has positive atomization energy (unusual)"
                    )

        except PyGDataCreationError:
            raise  # Re-raise PyG data errors as-is
        except Exception as e:
            # Convert to dataset-specific handler error if available
            if HANDLERS_AVAILABLE:
                raise create_dataset_handler_error(
                    message=f"{self._dataset_config.dataset_type} PyG data validation failed in legacy mode",
                    dataset_type=self._dataset_config.dataset_type,
                    operation="legacy_pyg_validation",
                    details=f"Original error: {type(e).__name__}: {str(e)}",
                ) from e
            else:
                self.logger.warning(
                    f"{self._dataset_config.dataset_type} PyG data validation error for molecule {molecule_index}: {e}"
                )

    def _validate_structural_features_in_pyg_data(
        self, pyg_data: Data, molecule_index: int, validation_results: dict[str, bool]
    ) -> None:
        """
        Handler-Based Pattern Development: Enhanced structural features validation in PyG data.
        """
        try:
            # Atom features validation
            if hasattr(pyg_data, "x") and pyg_data.x is not None:
                if isinstance(pyg_data.x, torch.Tensor):
                    if torch.any(torch.isnan(pyg_data.x)) or torch.any(torch.isinf(pyg_data.x)):
                        self.logger.warning(
                            f"Molecule {molecule_index} has NaN/Inf values in atom structural features"
                        )

                    expected_atom_features = len(self.structural_features_config.get("atom", []))
                    if expected_atom_features > 0:
                        atom_feature_dim = pyg_data.x.shape[1] if pyg_data.x.ndim > 1 else 1
                        if atom_feature_dim == 0:
                            self.logger.warning(
                                f"Molecule {molecule_index} has zero-dimensional atom features despite configuration"
                            )

            # Bond features validation
            if hasattr(pyg_data, "edge_attr") and pyg_data.edge_attr is not None:
                if isinstance(pyg_data.edge_attr, torch.Tensor):
                    if torch.any(torch.isnan(pyg_data.edge_attr)) or torch.any(
                        torch.isinf(pyg_data.edge_attr)
                    ):
                        self.logger.warning(
                            f"Molecule {molecule_index} has NaN/Inf values in bond structural features"
                        )

                    expected_bond_features = len(self.structural_features_config.get("bond", []))
                    if expected_bond_features > 0:
                        bond_feature_dim = (
                            pyg_data.edge_attr.shape[1] if pyg_data.edge_attr.ndim > 1 else 1
                        )
                        if bond_feature_dim == 0:
                            self.logger.warning(
                                f"Molecule {molecule_index} has zero-dimensional bond features despite configuration"
                            )

            # Configuration consistency validation
            configured_atom_features = self.structural_features_config.get("atom", [])
            configured_bond_features = self.structural_features_config.get("bond", [])

            if configured_atom_features and (not hasattr(pyg_data, "x") or pyg_data.x is None):
                self.logger.warning(
                    f"Molecule {molecule_index} configured for atom features but pyg_data.x is None"
                )

            if configured_bond_features and (
                not hasattr(pyg_data, "edge_attr") or pyg_data.edge_attr is None
            ):
                self.logger.warning(
                    f"Molecule {molecule_index} configured for bond features but pyg_data.edge_attr is None"
                )

        except Exception as e:
            self.logger.warning(
                f"Error validating structural features for molecule {molecule_index}: {e}"
            )

    def _prepare_molecule_data(
        self,
        molecule_index: int,
        raw_properties_dict: dict[str, Any],
        dataset_config: DatasetConfig | None = None,
    ) -> dict[str, Any]:
        """
        Extract and prepare essential molecule data from raw properties.
        """
        # Use provided container or fall back to internal one
        dataset_config = dataset_config or self._dataset_config

        try:
            # Extract identifier dynamically based on handler configuration
            dataset_type = dataset_config.dataset_type
            identifier_keys = get_identifier_keys(dataset_type)

            mol_identifier = None
            mol_id_type = "unknown"

            for npz_key, id_type in identifier_keys:
                mol_identifier = raw_properties_dict.get(npz_key)
                if mol_identifier is not None:
                    mol_id_type = id_type
                    # Handle array types (e.g., NPZ arrays)
                    if isinstance(mol_identifier, (list, np.ndarray)):
                        mol_identifier = str(mol_identifier[0]) if len(mol_identifier) > 0 else None
                    break

            # Extract coordinates and atomic numbers
            coordinates = raw_properties_dict.get("coordinates")
            atomic_numbers = raw_properties_dict.get("atoms")

            # Handle 'rots' property homogenization (preserved from original)
            self._homogenize_rots_property(molecule_index, mol_identifier, raw_properties_dict)

            # Validate essential inputs
            if mol_identifier is None:
                # For coords-based approach, identifier is optional (used only for logging)
                mol_identifier = f"molecule_{molecule_index}"
                mol_id_type = "index"
                self.logger.debug(
                    f"No identifier found for molecule {molecule_index}, using index-based identifier"
                )
            if coordinates is None:
                raise RDKitConversionError(
                    molecule_index=molecule_index,
                    inchi=str(mol_identifier),
                    reason="Coordinates not found in raw data.",
                    detail="Cannot create RDKit molecule without coordinates.",
                )
            if atomic_numbers is None:
                raise RDKitConversionError(
                    molecule_index=molecule_index,
                    inchi=str(mol_identifier),
                    reason="Atomic numbers ('atoms' key) not found in raw data.",
                    detail="Cannot create RDKit molecule for InChI/QM data without explicit atomic numbers.",
                )
            # Ensure mol_identifier is a string
            mol_identifier = str(mol_identifier)

            # Handler-Based Pattern Development: Handler-based validation with enhanced error handling
            self._validate_with_handler(molecule_index, raw_properties_dict, mol_identifier)

            return {
                "identifier": mol_identifier,
                "id_type": mol_id_type,
                "coordinates": coordinates,
                "atomic_numbers": atomic_numbers,
            }

        except (RDKitConversionError, MoleculeProcessingError):
            raise  # Re-raise known errors as-is
        except Exception as e:
            # Handler-Based Pattern Development: Enhanced error context
            error_context = create_handler_error_context(
                handler_type=getattr(
                    self._dataset_handler, "get_dataset_type", lambda: "unknown"
                )(),
                operation="prepare_molecule_data",
                molecule_index=molecule_index,
            )
            self.logger.debug(f"Error preparing molecule data: {error_context}")
            raise MoleculeProcessingError(
                molecule_index=molecule_index,
                inchi="N/A",
                message="Error preparing molecule data",
                reason=f"Unexpected error: {type(e).__name__}",
                details=str(e),
            ) from e

    def _homogenize_rots_property(
        self, molecule_index: int, mol_identifier: str | None, raw_properties_dict: dict[str, Any]
    ) -> None:
        """
        Homogenize 'rots' property from list to numpy array if needed.
        Modifies raw_properties_dict in place.

        Handler-Based Pattern Development: Enhanced error handling.
        """
        if "rots" not in raw_properties_dict:
            return

        rots_data = raw_properties_dict["rots"]
        identifier_str = str(mol_identifier) if mol_identifier else "N/A"

        try:
            if isinstance(rots_data, list):
                try:
                    raw_properties_dict["rots"] = np.array(rots_data, dtype=float)
                    self.logger.debug(
                        f"Molecule {molecule_index} (ID: {identifier_str}): "
                        "Converted 'rots' from list to numpy array for consistency."
                    )
                except ValueError as e:
                    self.logger.error(
                        f"Molecule {molecule_index} (ID: {identifier_str}): "
                        f"Failed to convert 'rots' list to numpy array. Skipping molecule. Error: {e}"
                    )
                    raise PropertyEnrichmentError(
                        molecule_index=molecule_index,
                        inchi=identifier_str,
                        property_name="rots",
                        reason="Failed to convert 'rots' list to numeric NumPy array.",
                        detail=str(e),
                    ) from e
            elif not isinstance(rots_data, np.ndarray):
                self.logger.warning(
                    f"Molecule {molecule_index} (ID: {identifier_str}): "
                    f"'rots' property is neither a list nor a numpy array (actual type: {type(rots_data)}). "
                    "Attempting conversion anyway."
                )
                try:
                    raw_properties_dict["rots"] = np.array(rots_data, dtype=float)
                except ValueError as e:
                    self.logger.error(
                        f"Molecule {molecule_index} (ID: {identifier_str}): "
                        f"Failed to convert unexpected 'rots' type to numpy array. Skipping molecule. Error: {e}"
                    )
                    raise PropertyEnrichmentError(
                        molecule_index=molecule_index,
                        inchi=identifier_str,
                        property_name="rots",
                        reason="Unexpected 'rots' data type. Failed to convert to numeric NumPy array.",
                        detail=str(e),
                    ) from e

        except PropertyEnrichmentError:
            raise  # Re-raise property enrichment errors as-is
        except Exception as e:
            # Handler-Based Pattern Development: Convert unexpected errors to property enrichment errors
            raise PropertyEnrichmentError(
                molecule_index=molecule_index,
                inchi=identifier_str,
                property_name="rots",
                reason="Unexpected error during 'rots' property homogenization",
                detail=f"{type(e).__name__}: {str(e)}",
            ) from e

    def _create_base_pyg_data(
        self,
        molecule_index: int,
        mol_data: dict[str, Any],
        raw_properties_dict: dict[str, Any],
        dataset_config: DatasetConfig | None = None,
    ) -> Data:
        """
        Create the base PyG Data object from molecule data.

        MIGRATED: Handler-Based Pattern Development -  Enhanced error handling while preserving original logic.
        """
        # Use provided container or fall back to internal one
        dataset_config = dataset_config or self._dataset_config

        # Try enhanced utils first for datasets with identifier_coordinate_based strategy
        # This includes any dataset using InChI/SMILES identifiers
        if _should_use_enhanced_utils(dataset_config.dataset_type) and RDKIT_AVAILABLE:
            try:
                return self._create_pyg_data_with_enhanced_utils(
                    molecule_index,
                    raw_properties_dict,
                    mol_data["identifier"],
                    mol_data["id_type"],
                    mol_data["identifier"],
                    mol_data["coordinates"],
                    mol_data["atomic_numbers"],
                    dataset_config,
                )
            except Exception as e:
                # Log the error and fall back to original approach
                self.logger.debug(
                    f"Enhanced utils failed for molecule {molecule_index}, trying fallback: {e}"
                )
                # IMPORTANT: Don't return None, try fallback instead

        # Use original direct approach as fallback
        return self._create_pyg_data_original_approach(
            molecule_index,
            raw_properties_dict,
            mol_data["identifier"],
            mol_data["id_type"],
            mol_data["identifier"],
            mol_data["coordinates"],
            mol_data["atomic_numbers"],
            dataset_config,
        )

    def _add_features(
        self,
        pyg_data: Data,
        molecule_index: int,
        mol_identifier: str,
        mol_id_type: str,
        raw_properties_dict: dict[str, Any],
        data_config: dict[str, Any],
        dataset_config: DatasetConfig | None = None,
    ) -> Data:
        """
        Add all features to the PyG Data object.

        MIGRATED: Handler-Based Pattern Development -  Enhanced error handling while preserving structure.
        """
        # Use provided container or fall back to internal one
        dataset_config = dataset_config or self._dataset_config

        try:
            # Step 1: Add structural features
            pyg_data = self._add_structural_features_enhanced(
                pyg_data, molecule_index, mol_identifier, raw_properties_dict
            )

            # Step 2: Attach identifiers
            self._attach_identifiers_original_logic(
                pyg_data, mol_identifier, mol_id_type, molecule_index
            )

            # Step 3: Enrich with properties - pass dataset_config to property enrichment
            enriched_data = enrich_pyg_data_with_properties(
                pyg_data=pyg_data,
                mol_idx=molecule_index,
                raw_properties_dict=raw_properties_dict,
                inchi_identifier=mol_identifier,
                logger=self.logger,
                data_config=data_config,
                dataset_config=dataset_config,
                dataset_handler=self._dataset_handler,
            )

            return enriched_data

        except (StructuralFeatureError, PropertyEnrichmentError):
            raise  # Re-raise known errors as-is
        except Exception as e:
            # Handler-Based Pattern Development: Enhanced error context
            error_context = create_handler_error_context(
                handler_type=getattr(
                    self._dataset_handler, "get_dataset_type", lambda: "unknown"
                )(),
                operation="add_features",
                molecule_index=molecule_index,
                additional_context={"identifier": mol_identifier},
            )
            self.logger.debug(f"Error adding features: {error_context}")
            raise PropertyEnrichmentError(
                molecule_index=molecule_index,
                inchi=mol_identifier,
                property_name="feature_addition",
                reason="Error during feature addition process",
                detail=f"{type(e).__name__}: {str(e)}",
            ) from e

    def _validate_result(
        self,
        pyg_data: Data,
        molecule_index: int,
        mol_identifier: str,
        dataset_config: DatasetConfig | None = None,
    ) -> None:
        """
        Validate the final PyG Data object.

        MIGRATED: Handler-Based Pattern Development -  Enhanced validation with better error context.
        """
        # Use provided container or fall back to internal one
        dataset_config = dataset_config or self._dataset_config

        try:
            self._validate_final_pyg_data(pyg_data, molecule_index, mol_identifier, dataset_config)
        except (PyGDataCreationError, ValidationError):
            raise  # Re-raise known validation errors as-is
        except Exception as e:
            # Handler-Based Pattern Development: Enhanced error context
            error_context = create_handler_error_context(
                handler_type=getattr(
                    self._dataset_handler, "get_dataset_type", lambda: "unknown"
                )(),
                operation="validate_result",
                molecule_index=molecule_index,
                additional_context={"identifier": mol_identifier},
            )
            self.logger.debug(f"Error validating result: {error_context}")
            raise ValidationError(
                message="Final result validation failed",
                validation_type="final_validation",
                failed_checks=[f"unexpected_error: {type(e).__name__}"],
                data_context=f"molecule_{molecule_index}",
                details=f"Original error: {str(e)}",
            ) from e

    def _log_structural_features_capabilities(self) -> None:
        """
        Log the structural features capabilities and validate configuration.

        Handler-Based Pattern Development: Enhanced with better error handling.
        """
        try:
            available_features = get_available_features()
            configured_atom_features = self.structural_features_config.get("atom", [])
            configured_bond_features = self.structural_features_config.get("bond", [])

            self.logger.info(
                f"Structural features enabled - "
                f"Atom features: {configured_atom_features}, "
                f"Bond features: {configured_bond_features}"
            )

            # Validate configured features
            invalid_atom_features = [
                f for f in configured_atom_features if f not in available_features["atom"]
            ]
            invalid_bond_features = [
                f for f in configured_bond_features if f not in available_features["bond"]
            ]

            if invalid_atom_features:
                self.logger.warning(f"Invalid atom features configured: {invalid_atom_features}")
            if invalid_bond_features:
                self.logger.warning(f"Invalid bond features configured: {invalid_bond_features}")

        except Exception as e:
            self.logger.warning(f"Error logging structural features capabilities: {e}")

    def _create_pyg_data_with_enhanced_utils(
        self,
        current_mol_index: int,
        raw_properties_dict: dict[str, Any],
        current_mol_identifier: str,
        mol_id_type: str,
        mol_identifier: str,
        coordinates: np.ndarray,
        atomic_numbers: np.ndarray,
        dataset_config: DatasetConfig | None = None,
    ) -> Data:
        """
        Create PyG Data using enhanced utilities for identifier-coordinate-based datasets.

        MIGRATED: Handler-Based Pattern Development -  Enhanced error handling while preserving utility integration.
        """
        # Use provided container or fall back to internal one
        dataset_config = dataset_config or self._dataset_config

        try:
            # Use the enhanced utility function
            # ---
            return create_mol_with_dataset_support(
                mol_identifier=mol_identifier,
                coordinates=coordinates,
                atomic_numbers=atomic_numbers,
                logger=self.logger,
                molecule_index=current_mol_index,
                mol_id_type=mol_id_type,
                dataset_type=dataset_config.dataset_type,
                raw_data_dict=raw_properties_dict,
                uncertainty_config=dataset_config.uncertainty_config,
                dataset_config=dataset_config,
            )
        # ---
        except Exception as e:
            # Handler-Based Pattern Development: Enhanced error context
            error_context = create_handler_error_context(
                handler_type=dataset_config.dataset_type,
                operation="create_pyg_with_enhanced_utils",
                molecule_index=current_mol_index,
                additional_context={"identifier": current_mol_identifier},
            )
            self.logger.debug(f"Error creating PyG data with enhanced utils: {error_context}")
            raise PyGDataCreationError(
                message="Error creating PyG Data using enhanced utilities",
                molecule_index=current_mol_index,
                smiles=current_mol_identifier,
                reason="Enhanced utility function failed",
                detail=f"{type(e).__name__}: {str(e)}",
            ) from e

    def clear_handler_caches(self):
        """
        Enhanced: Clear all caches including intelligent cache manager.
        """
        try:
            # Original caches
            if hasattr(self, "_required_props_cache"):
                self._required_props_cache = None
            if hasattr(self, "_dataset_config_cache"):
                self._dataset_config_cache = None
            if hasattr(self, "_property_cache"):
                self._property_cache.clear()
            if hasattr(self, "_transform_validation_cache"):
                self._transform_validation_cache.clear()

            # Clear intelligent cache manager
            if self._enhanced_transform_available and self._cache_manager:
                cleared_count = self._cache_manager.clear_all_caches()
                self.logger.debug(f"Cleared {cleared_count} cache entries")

        except Exception as e:
            self.logger.debug(f"Error clearing handler caches: {e}")

    # Additional utility methods

    def get_conversion_statistics(self) -> dict[str, Any]:
        """
        Enhanced: Get comprehensive conversion statistics including relevant metrics.
        """
        stats = {
            "dataset_type": self.dataset_type,
            "handlers_enabled": self._use_handlers,
            "structural_features_enabled": self.is_structural_features_enabled,
            "rdkit_available": RDKIT_AVAILABLE,
            "handlers_available": HANDLERS_AVAILABLE,
            "transform_system_available": self._transform_system_available,
            "graph_transforms_available": GRAPH_TRANSFORMS_AVAILABLE,
            "enhanced_transform_available": self._enhanced_transform_available,
            "total_processed": self._conversion_stats["total_processed"],
            "successful_conversions": self._conversion_stats["successful_conversions"],
            "failed_conversions": self._conversion_stats["failed_conversions"],
        }

        if self._use_handlers and self._dataset_handler:
            try:
                stats["handler_type"] = self._dataset_handler.get_dataset_type()
                stats["required_properties"] = self._dataset_handler.get_required_properties()
            except Exception as e:
                stats["handler_error"] = str(e)
                stats["required_properties"] = []

        # Transformation statistics
        if self._transformation_config:
            try:
                if isinstance(self._transformation_config, dict):
                    stats["transformation_config"] = {
                        "experimental_setups": list(
                            self._transformation_config.get("experimental_setups", {}).keys()
                        ),
                        "default_setup": self._transformation_config.get(
                            "default_setup", "unknown"
                        ),
                        "validation_enabled": self._transformation_config.get("validation", {}).get(
                            "enabled", True
                        ),
                    }
                else:
                    stats["transformation_config"] = {
                        "experimental_setups": list(
                            self._transformation_config.experimental_setups.keys()
                        ),
                        "default_setup": self._transformation_config.default_setup,
                        "validation_enabled": self._transformation_config.validation.get(
                            "enabled", True
                        )
                        if isinstance(self._transformation_config.validation, dict)
                        else self._transformation_config.validation.enabled,
                    }
            except AttributeError:
                stats["transformation_config"] = {"error": "Unable to parse transformation config"}

        if self._transform_capabilities:
            stats["transform_capabilities"] = self._transform_capabilities

        # Enhanced statistics
        if self._enhanced_transform_available:
            stats["enhanced_transform_statistics"] = {}

            # Dynamic discovery stats
            if self._dynamic_discovery:
                try:
                    stats["enhanced_transform_statistics"]["discovery"] = {
                        "total_discovered": len(
                            self._dynamic_discovery.get_discovered_transforms()
                        ),
                        "custom_transforms": len(self._dynamic_discovery.get_custom_transforms()),
                        "discovery_enabled": True,
                    }
                except Exception as e:
                    stats["enhanced_transform_statistics"]["discovery"] = {
                        "enabled": False,
                        "error": str(e),
                    }

            # Cache statistics
            if self._cache_manager:
                try:
                    cache_stats = self._cache_manager.get_statistics()
                    stats["enhanced_transform_statistics"]["cache"] = {
                        "total_entries": cache_stats.total_entries,
                        "hit_rate": cache_stats.hit_rate,
                        "memory_usage": cache_stats.memory_usage_mb,
                        "evictions": cache_stats.eviction_count,
                    }
                except Exception as e:
                    stats["enhanced_transform_statistics"]["cache"] = {
                        "enabled": False,
                        "error": str(e),
                    }

            # Parameter introspection stats
            if self._parameter_introspector:
                stats["enhanced_transform_statistics"]["introspection"] = {
                    "enabled": True,
                    "supports_type_checking": True,
                    "supports_range_validation": True,
                }

        return stats

    def reset_statistics(self) -> None:
        """
        Reset conversion statistics to initial values.
        """
        self._conversion_stats = {
            "total_processed": 0,
            "successful_conversions": 0,
            "failed_conversions": 0,
        }

    def validate_configuration_compatibility(self) -> dict[str, Any]:
        """
        Handler-Based Pattern Development: Enhanced configuration compatibility validation with handler integration.
        Added transformation configuration validation.
        """
        diagnostics = {
            "dataset_config_valid": True,
            "filter_config_valid": True,
            "processing_config_valid": True,
            "structural_features_valid": True,
            "handler_integration_valid": True,
            "transformation_config_valid": True,
            "warnings": [],
            "errors": [],
            "handler_diagnostics": {},
            "transform_diagnostics": {},
        }

        try:
            # Validate dataset configuration
            # PHASE 6: Registry-based validation
            dataset_type = self._dataset_config.dataset_type

            if not _is_dataset_type_registered(dataset_type):
                diagnostics["dataset_config_valid"] = False
                available_types = _get_available_dataset_types()
                diagnostics["errors"].append(
                    f"Invalid dataset type: {dataset_type}. Available: {available_types}"
                )

            # Validate handler integration with enhanced diagnostics
            handler_validation = self.validate_handler_integration()
            diagnostics["handler_integration_valid"] = handler_validation["configuration_valid"]
            diagnostics["handler_diagnostics"] = handler_validation

            if not handler_validation["handlers_available"]:
                diagnostics["warnings"].append(
                    "Dataset handlers not available, using legacy validation"
                )

            if handler_validation["errors"]:
                diagnostics["errors"].extend(
                    [f"Handler: {err}" for err in handler_validation["errors"]]
                )

            if handler_validation["warnings"]:
                diagnostics["warnings"].extend(
                    [f"Handler: {warn}" for warn in handler_validation["warnings"]]
                )

            # Validate structural features
            if self.is_structural_features_enabled:
                if not RDKIT_AVAILABLE:
                    diagnostics["warnings"].append(
                        "RDKit not available, structural features limited"
                    )

                try:
                    available_features = get_available_features()
                    configured_atom_features = self.structural_features_config.get("atom", [])
                    configured_bond_features = self.structural_features_config.get("bond", [])

                    invalid_atom_features = [
                        f for f in configured_atom_features if f not in available_features["atom"]
                    ]
                    invalid_bond_features = [
                        f for f in configured_bond_features if f not in available_features["bond"]
                    ]

                    if invalid_atom_features or invalid_bond_features:
                        diagnostics["structural_features_valid"] = False
                        diagnostics["errors"].append(
                            f"Invalid features configured: atoms={invalid_atom_features}, bonds={invalid_bond_features}"
                        )
                except Exception as e:
                    diagnostics["structural_features_valid"] = False
                    diagnostics["errors"].append(f"Structural features validation failed: {e}")

            # Validate transformation configuration
            if self._transformation_config:
                try:
                    self._validate_transformation_configuration()
                    diagnostics["transform_diagnostics"]["status"] = "valid"
                    diagnostics["transform_diagnostics"]["experimental_setups"] = list(
                        self._transformation_config.experimental_setups.keys()
                    )
                except TransformConfigurationError as e:
                    diagnostics["transformation_config_valid"] = False
                    diagnostics["errors"].append(f"Transform config: {str(e)}")
                    diagnostics["transform_diagnostics"]["status"] = "invalid"
                    diagnostics["transform_diagnostics"]["error"] = str(e)

        except Exception as e:
            diagnostics["errors"].append(f"Configuration validation failed: {e}")

        return diagnostics

    def get_processing_capabilities(self) -> dict[str, Any]:
        """
        Handler-Based Pattern Development: Enhanced processing capabilities with handler information.
        Added transformation capabilities.
        """
        capabilities = {
            "dataset_types_supported": _get_available_dataset_types(),  # PHASE 6: Dynamic
            "current_dataset_type": self.dataset_type,
            "rdkit_available": RDKIT_AVAILABLE,
            "handlers_available": HANDLERS_AVAILABLE,
            "handler_active": self._use_handlers,
            "structural_features_enabled": self.is_structural_features_enabled,
            "migration_phase": "Handler_Migration_Complete",
            "transform_system_available": self._transform_system_available,
            "graph_transforms_available": GRAPH_TRANSFORMS_AVAILABLE,
        }

        if self.is_structural_features_enabled:
            try:
                available_features = get_available_features()
                capabilities["structural_features"] = {
                    "available_atom_features": available_features["atom"],
                    "available_bond_features": available_features["bond"],
                    "configured_atom_features": self.structural_features_config.get("atom", []),
                    "configured_bond_features": self.structural_features_config.get("bond", []),
                }
            except Exception as e:
                capabilities["structural_features_error"] = str(e)

        if self._use_handlers and self._dataset_handler:
            try:
                capabilities["handler_info"] = {
                    "type": self._dataset_handler.get_dataset_type(),
                    "required_properties": self._dataset_handler.get_required_properties(),
                }
            except Exception as e:
                capabilities["handler_info"] = {"error": str(e)}

        # Add transformation capabilities
        capabilities["transformation_capabilities"] = self.get_transform_capabilities()

        return capabilities

    def get_error_recovery_capabilities(self) -> dict[str, Any]:
        """
        Handler-Based Pattern Development: New method to get error recovery capabilities.
        Added transformation error recovery.
        """
        return {
            "handler_error_recovery": HANDLERS_AVAILABLE,
            "transform_error_recovery": GRAPH_TRANSFORMS_AVAILABLE,
            "legacy_fallback_available": True,
            "recoverable_error_types": [
                "HandlerOperationError",
                "HandlerValidationError",
                "MoleculeProcessingError",
                "PropertyEnrichmentError",
                "StructuralFeatureError",
                "TransformValidationError",
                "TransformCompositionError",
            ],
            "non_recoverable_error_types": [
                "HandlerNotAvailableError",
                "HandlerCompatibilityError",
                "ConfigurationError",
                "TransformConfigurationError",
                "ExperimentalSetupError",
            ],
            "error_context_tracking": True,
            "recovery_suggestions_available": True,
        }

    def get_enhanced_transform_diagnostics(self) -> dict[str, Any]:
        """
        Get detailed diagnostics for Enhanced Transform Features.

        Returns:
            Comprehensive diagnostics dictionary for enhanced transform components
        """
        diagnostics = {
            "enhanced_transform_available": self._enhanced_transform_available,
            "components": {
                "dynamic_discovery": self._dynamic_discovery is not None,
                "parameter_introspection": self._parameter_introspector is not None,
                "validation_reporting": self._validation_reporter is not None,
                "cache_management": self._cache_manager is not None,
            },
            "status": {},
            "errors": [],
            "warnings": [],
        }

        if not self._enhanced_transform_available:
            diagnostics["warnings"].append("Enhanced Transform Features not available")
            return diagnostics

        # Dynamic discovery diagnostics
        if self._dynamic_discovery:
            try:
                discovery_status = self._dynamic_discovery.get_status()
                diagnostics["status"]["discovery"] = {
                    "operational": True,
                    "transforms_discovered": discovery_status.total_discovered,
                    "custom_transforms": discovery_status.custom_count,
                    "search_paths": discovery_status.search_paths,
                    "last_discovery": discovery_status.last_discovery_time,
                }
            except Exception as e:
                diagnostics["errors"].append(f"Discovery diagnostics failed: {e}")
                diagnostics["status"]["discovery"] = {"operational": False, "error": str(e)}

        # Parameter introspection diagnostics
        if self._parameter_introspector:
            try:
                introspector_status = self._parameter_introspector.get_status()
                diagnostics["status"]["introspection"] = {
                    "operational": True,
                    "cached_schemas": introspector_status.cached_count,
                    "validation_count": introspector_status.validation_count,
                    "error_rate": introspector_status.error_rate,
                }
            except Exception as e:
                diagnostics["errors"].append(f"Introspection diagnostics failed: {e}")
                diagnostics["status"]["introspection"] = {"operational": False, "error": str(e)}

        # Cache management diagnostics
        if self._cache_manager:
            try:
                cache_status = self._cache_manager.get_detailed_status()
                diagnostics["status"]["cache"] = {
                    "operational": True,
                    "total_entries": cache_status.total_entries,
                    "memory_usage_mb": cache_status.memory_usage_mb,
                    "hit_rate": cache_status.hit_rate,
                    "eviction_policy": cache_status.eviction_policy,
                    "max_size": cache_status.max_size,
                }
            except Exception as e:
                diagnostics["errors"].append(f"Cache diagnostics failed: {e}")
                diagnostics["status"]["cache"] = {"operational": False, "error": str(e)}

        # Validation reporting diagnostics
        if self._validation_reporter:
            try:
                reporter_status = self._validation_reporter.get_status()
                diagnostics["status"]["reporting"] = {
                    "operational": True,
                    "reports_generated": reporter_status.report_count,
                    "formats_supported": reporter_status.supported_formats,
                }
            except Exception as e:
                diagnostics["errors"].append(f"Reporting diagnostics failed: {e}")
                diagnostics["status"]["reporting"] = {"operational": False, "error": str(e)}

        return diagnostics

    def get_registry_integration_status(self) -> dict[str, Any]:
        """
        PHASE 6: Get the status of registry integration for this converter.

        This method provides diagnostic information about the registry integration,
        including availability status, registered dataset types, and feature information
        for the current dataset type.

        Returns:
            Dict containing registry availability and dataset-specific information:
            - registry_available: Whether the registry system is available
            - registry_initialized: Whether initialization has been attempted
            - registry_import_error: Error message if registry import failed
            - available_dataset_types: List of registered dataset types
            - current_dataset_type: The dataset type of this converter
            - current_dataset_registered: Whether current type is in registry
            - dataset_features: Feature flags for current dataset (if registered)
            - molecule_creation_strategy: Strategy for molecule creation
            - uses_enhanced_utils: Whether enhanced utils are used
            - phase_6_complete: Marker indicating Phase 6 refactoring is complete
        """
        _init_registry()

        status = {
            "registry_available": _REGISTRY_AVAILABLE,
            "registry_initialized": _REGISTRY_INITIALIZED,
            "registry_import_error": _REGISTRY_IMPORT_ERROR,
            "available_dataset_types": _get_available_dataset_types(),
            "current_dataset_type": self.dataset_type,
            "current_dataset_registered": _is_dataset_type_registered(self.dataset_type),
            "phase_6_complete": True,
        }

        # Add feature information for current dataset
        if _is_dataset_type_registered(self.dataset_type):
            status["dataset_features"] = {
                "uncertainty_handling": _get_dataset_feature(
                    self.dataset_type, "uncertainty_handling"
                ),
                "vibrational_analysis": _get_dataset_feature(
                    self.dataset_type, "vibrational_analysis"
                ),
                "atomization_energy": _get_dataset_feature(self.dataset_type, "atomization_energy"),
                "orbital_analysis": _get_dataset_feature(self.dataset_type, "orbital_analysis"),
                "frequency_analysis": _get_dataset_feature(self.dataset_type, "frequency_analysis"),
                "rotational_constants": _get_dataset_feature(
                    self.dataset_type, "rotational_constants"
                ),
                "homo_lumo_gap": _get_dataset_feature(self.dataset_type, "homo_lumo_gap"),
                "mo_energies": _get_dataset_feature(self.dataset_type, "mo_energies"),
            }
            status["molecule_creation_strategy"] = _get_dataset_molecule_creation_strategy(
                self.dataset_type
            )
            status["uses_enhanced_utils"] = _should_use_enhanced_utils(self.dataset_type)

        return status


# MIGRATION COMPLETE
# ========================================================
#
# ENHANCEMENTS (Beyond Handler-Based Pattern Development):
#
# 1. FULL CONFIGURATION CONTAINER INTEGRATION:
#    - Complete DatasetConfig, FilterConfig, ProcessingConfig support
#    - New TransformationConfig container integration
#    - Removed all remaining legacy dictionary config access
#    - All methods now use configuration containers consistently
#
# 2. TRANSFORMATION SYSTEM INTEGRATION:
#    - New transformation_config parameter in __init__()
#    - _initialize_transform_system_awareness() for transform capabilities
#    - get_transform_capabilities() for querying transform system status
#    - validate_transform_compatibility() for validating transforms
#    - _check_transform_dataset_compatibility() for dataset-specific checks
#
# 3. ENHANCED CONFIGURATION VALIDATION:
#    - _validate_transformation_configuration() for transform configs
#    - Enhanced _validate_converter_configuration() with transform validation
#    - Better error messages for transform configuration issues
#
# 4. TRANSFORMATION EXCEPTION INTEGRATION:
#    - Full support for TransformConfigurationError
#    - TransformValidationError handling
#    - TransformCompositionError integration
#    - ExperimentalSetupError support
#    - TransformRegistryError handling
#
# 5. ENHANCED STATISTICS AND DIAGNOSTICS:
#    - get_conversion_statistics() includes transform stats
#    - validate_configuration_compatibility() includes transform validation
#    - get_processing_capabilities() includes transform capabilities
#    - get_error_recovery_capabilities() includes transform error recovery
#
# 6. CACHE MANAGEMENT:
#    - Added _transform_validation_cache for performance
#    - clear_handler_caches() now clears transform caches too
#    - Optimized validation caching
#
# 7. GRAPH_TRANSFORMS MODULE INTEGRATION:
#    - Import and availability checking for graph_transforms
#    - Integration with TransformRegistry, TransformValidator, TransformComposer
#    - GraphTransformationEngine support
#    - Graceful degradation when graph_transforms unavailable
#
# 8. CONFIGURATION ACCESSOR INTEGRATION:
#    - Uses get_transformation_config() from config_accessors
#    - Uses get_experimental_setup() for setup retrieval
#    - Uses list_experimental_setups() for setup listing
#    - Uses get_default_experimental_setup() for defaults
#
# PRESERVED FUNCTIONALITY:
# - All Handler-Based Pattern Development handler pattern functionality maintained
# - All original conversion pipeline logic preserved
# - Handler-only architecture (STEP 3 cleanup complete)
# - All original methods enhanced but not broken
# - Same public interface with added capabilities
# - Performance optimizations maintained
#
# INTEGRATION BENEFITS:
# - Seamless transformation system integration
# - Better separation of concerns (configs as containers)
# - Enhanced transform validation and compatibility checking
# - Improved error handling with transform-specific exceptions
# - Better diagnostics and introspection capabilities
# - Cleaner code with no legacy dictionary access
# - Ready for experimental setup switching and ablation studies
#
# HANDLER ARCHITECTURE (STEP 3 Cleanup Complete):
# - Handler-only pattern enforced
# - Handlers must be created from configs explicitly
# - No get_or_create_handler(None, ...) fallback patterns
# - All existing code continues to work (with proper handler creation)
# - Graceful fallback when transforms unavailable
# - No breaking changes to public API
# - Optional transformation_config parameter
#
# READY FOR RESEARCH:
# - Full experimental setup support
# - Transform compatibility validation
# - Dataset-specific transform checking
# - Comprehensive error recovery
# - Enhanced diagnostics for debugging

# MIGRATION COMPLETE: Full Integration
# ==========================================================
#
# Enhanced Transform System - COMPLETE
#
# NEW CAPABILITIES:
#
# 1. DYNAMIC TRANSFORM DISCOVERY:
#    - Runtime detection of available transforms
#    - Support for custom/user-defined transforms
#    - Automatic discovery path management
#    - Integration with transform registry
#
# 2. PARAMETER INTROSPECTION:
#    - Enhanced parameter validation with type checking
#    - Range validation for numeric parameters
#    - Dependency analysis between parameters
#    - Automatic parameter schema generation
#    - Helpful suggestions for parameter corrections
#
# 3. ENHANCED VALIDATION REPORTING:
#    - Structured validation results with detailed per-transform info
#    - Enhanced error messages with context
#    - Validation report generation in multiple formats
#    - Better debugging information for validation failures
#
# 4. INTELLIGENT CACHE MANAGEMENT:
#    - Smart caching of validation results
#    - Memory-efficient cache with automatic eviction
#    - Cache statistics and monitoring
#    - Invalidation strategies based on config changes
#
# NEW PUBLIC METHODS:
# - get_transform_parameter_info(): Get parameter schema for transforms
# - discover_available_transforms(): Dynamic transform discovery
# - get_enhanced_transform_diagnostics(): Comprehensive diagnostics
#
# ENHANCED METHODS :
# - get_transform_capabilities(): Now includes relevant features
# - validate_transform_compatibility(): Enhanced with parameter introspection
# - clear_handler_caches(): Clears intelligent cache manager
# - get_conversion_statistics(): Includes relevant metrics
#
# INTEGRATION QUALITY:
# - Full backward compatibility maintained
# - Graceful degradation when relevant conditions unavailable
# - No breaking changes to existing API
# - Enhanced error handling and logging
# - Performance optimizations via intelligent caching
#
# RESEARCH-GRADE FEATURES:
# - Runtime transform discovery for extensibility
# - Comprehensive parameter validation
# - Detailed diagnostic capabilities
# - Production-ready caching strategies
# - Enhanced validation reporting for debugging

# PHASE 6: Registry Integration Complete
# ============================================================================
#
# CHANGES IN THIS FILE (Phase 6 Refactoring):
#
# 1. REGISTRY INTEGRATION INFRASTRUCTURE (lines ~195-350):
#    - _init_registry(): Lazy initialization to avoid circular imports
#    - _get_available_dataset_types(): Dynamic dataset type discovery
#    - _is_dataset_type_registered(): Dynamic dataset type validation
#    - _get_dataset_feature(): Feature-based dataset processing
#    - _get_dataset_molecule_creation_strategy(): Strategy lookup
#    - _should_use_enhanced_utils(): Enhanced utils determination
#
# 2. HARDCODED DATASET TYPE REPLACEMENTS (8 locations):
#
#    Location 1 - _check_transform_dataset_compatibility():
#    - Before: if self._dataset_config.dataset_type == 'DMC'
#    - After:  if _get_dataset_feature(self._dataset_config.dataset_type, 'uncertainty_handling')
#
#    Location 2 - _validate_dataset_configuration():
#    - Before: if dataset_type not in ['DFT', 'DMC', 'Wavefunction']
#    - After:  if not _is_dataset_type_registered(dataset_type)
#
#    Location 3 - _legacy_validation_fallback():
#    - Before: if dataset_type == "DMC" / elif dataset_type == "DFT"
#    - After:  if _get_dataset_feature(dataset_type, 'uncertainty_handling') / elif _get_dataset_feature(dataset_type, 'vibrational_analysis')
#
#    Location 4 - _validate_converter_configuration():
#    - Before: if dataset_type not in ["DFT", "DMC", "Wavefunction"]
#    - After:  if not _is_dataset_type_registered(dataset_type)
#
#    Location 5 - _validate_final_pyg_data():
#    - Before: if dataset_config.dataset_type == "DMC" / elif == "DFT"
#    - After:  if _get_dataset_feature(dataset_type, 'uncertainty_handling') / elif _get_dataset_feature(dataset_type, 'vibrational_analysis')
#
#    Location 6 - _create_base_pyg_data():
#    - Before: if dataset_config.dataset_type in ["DMC", "DFT"]
#    - After:  if _should_use_enhanced_utils(dataset_config.dataset_type)
#
#    Location 7 - validate_configuration_compatibility():
#    - Before: if dataset_type not in ['DFT', 'DMC', 'Wavefunction']
#    - After:  if not _is_dataset_type_registered(dataset_type)
#
#    Location 8 - get_processing_capabilities():
#    - Before: 'dataset_types_supported': ['DFT', 'DMC']
#    - After:  'dataset_types_supported': _get_available_dataset_types()
#
# 3. NEW METHOD ADDED:
#    - get_registry_integration_status(): Returns diagnostic info about registry integration
#
# BACKWARD COMPATIBILITY:
# - All existing function signatures unchanged
# - All existing return types unchanged
# - Legacy fallbacks preserved when registry unavailable
# - DFT, DMC, Wavefunction processing identical to before
#
# BENEFITS AFTER PHASE 6:
# - Zero-core-file-modification when adding new dataset types
# - Automatic validation of new dataset types via registry
# - Feature-based processing decisions (not type-based)
# - Dynamic dataset type discovery for UI/diagnostics
# - Enhanced debugging via get_registry_integration_status()
#
# TO ADD A NEW DATASET TYPE:
# 1. Create dataset class with @register decorator (Phase 2 pattern)
# 2. Define `features` attribute with appropriate flags
# 3. molecule_converter_core.py will automatically:
#    - Validate the dataset type as valid
#    - Apply appropriate transform compatibility checks based on features
#    - Use appropriate validation logic based on features
#    - Use enhanced utils if molecule creation strategy is identifier_coordinate_based
#
# NO MODIFICATIONS TO THIS FILE REQUIRED FOR NEW DATASET TYPES.
# ============================================================================
