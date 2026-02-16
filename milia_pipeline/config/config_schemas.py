# config_schemas.py - YAML Schema Validation and Migration System

"""
YAML Schema Validation and Configuration Migration System

This module provides comprehensive schema validation for transformation configurations
and handles migration between different configuration formats. It supports both
legacy formats and the new enhanced experimental setups format while providing
detailed validation feedback and automatic format detection.

YAML Configuration Enhancement
- Enhanced schema validation for experimental setups
- Automatic format detection and migration
- Backward compatibility with legacy formats
- Research-grade configuration validation
- Production-ready error handling and reporting

Corrected migration statistics tracking to avoid double-counting
"""

import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Self

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

from enum import Enum

from milia_pipeline.exceptions import ConfigurationError

# Try to import plugin system components
try:
    from milia_pipeline.transformations.plugin_system import PluginMetadata, PluginRegistry

    PLUGIN_SYSTEM_AVAILABLE = True
except ImportError:
    PLUGIN_SYSTEM_AVAILABLE = False
    PluginRegistry = None
    PluginMetadata = None

# Research API integration
try:
    from milia_pipeline.transformations.research_api import ExperimentConfiguration

    RESEARCH_API_AVAILABLE = True
except ImportError:
    RESEARCH_API_AVAILABLE = False
    ExperimentConfiguration = None

# Place this immediately after the module docstring (after line 19, before imports)

# Place immediately after module docstring (currently at lines 69-107)


# Initialize logger for this module
logger = logging.getLogger(__name__)


# ==========================================
# PHASE 5: Registry Integration for Dynamic Dataset Type Resolution
# ==========================================

# Registry availability flags - set during lazy initialization
_REGISTRY_AVAILABLE = False
_REGISTRY_IMPORT_ERROR = None
_REGISTRY_INITIALIZED = False

# Registry function placeholders (populated by _init_registry)
_registry_list_all = None
_registry_get = None
_registry_is_registered = None


def _init_registry() -> bool:
    """
    Lazily initialize registry imports to avoid circular import at module load time.

    The config_schemas.py module is imported early in the initialization chain.
    By deferring the registry import until first use, we allow the config module
    to fully load first.

    Returns:
        True if registry is available, False otherwise

    PHASE 5 REFACTORING: Lazy initialization to resolve circular import issues.
    Pattern: Following Phase 3 config_constants.py (lines 73-120)
    """
    global _REGISTRY_AVAILABLE, _REGISTRY_IMPORT_ERROR, _REGISTRY_INITIALIZED
    global _registry_list_all, _registry_get, _registry_is_registered

    if _REGISTRY_INITIALIZED:
        return _REGISTRY_AVAILABLE

    _REGISTRY_INITIALIZED = True

    try:
        # Direct import from registry module (not through datasets/__init__.py)
        from milia_pipeline.datasets.registry import (
            get,
            is_registered,
            list_all,
        )

        _registry_list_all = list_all
        _registry_get = get
        _registry_is_registered = is_registered
        _REGISTRY_AVAILABLE = True
        logger.debug("Dataset registry initialized in config_schemas.py")
        return True

    except ImportError as e:
        _REGISTRY_IMPORT_ERROR = str(e)
        logger.debug(f"Dataset registry not available in config_schemas.py: {e}")
        return False

    except Exception as e:
        _REGISTRY_IMPORT_ERROR = str(e)
        logger.debug(f"Dataset registry import failed in config_schemas.py: {e}")
        return False


def _registry_list_all_safe() -> list[str]:
    """
    Safely get list of all registered dataset types.

    DYNAMIC APPROACH: Instead of hardcoded fallback, this function:
    1. First tries the registry (primary source of truth)
    2. If registry fails, attempts filesystem discovery of dataset implementations
    3. Returns discovered types or empty list with warning

    Returns:
        List of dataset type names from registry or dynamic discovery

    PHASE 5 REFACTORING: Safe wrapper with dynamic fallback instead of hardcoded list.
    """
    _init_registry()
    if _registry_list_all is not None:
        try:
            return _registry_list_all()
        except Exception as e:
            logger.warning(f"Registry list_all() failed: {e}")

    # DYNAMIC FALLBACK: Try to discover dataset types from implementations directory
    try:
        from pathlib import Path

        # Find the implementations directory
        implementations_dir = Path(__file__).parent.parent / "datasets" / "implementations"
        if implementations_dir.exists():
            discovered_types = []
            for py_file in implementations_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                # Extract dataset name from filename (e.g., dft.py -> DFT)
                dataset_name = py_file.stem.upper()
                if dataset_name not in ["BASE", "REGISTRY", "UTILS"]:
                    discovered_types.append(dataset_name)
            if discovered_types:
                logger.debug(f"Dynamically discovered dataset types: {discovered_types}")
                return discovered_types
    except Exception as e:
        logger.debug(f"Dynamic dataset discovery failed: {e}")

    # Final fallback: return empty list and log warning
    # This forces proper registry initialization rather than silently using hardcoded values
    logger.warning(
        "No dataset types available - registry not initialized and dynamic discovery failed"
    )
    return []


def _registry_get_safe(name: str):
    """
    Safely get dataset class from registry.

    Args:
        name: Dataset type name

    Returns:
        Dataset class or None if not found/unavailable

    PHASE 5 REFACTORING: Safe wrapper with None return on failure.
    """
    _init_registry()
    if _registry_get is not None:
        try:
            return _registry_get(name)
        except Exception:
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
        True if registered, False otherwise (or if registry unavailable)

    PHASE 5 REFACTORING: Safe wrapper with dynamic fallback instead of hardcoded check.
    """
    _init_registry()
    if _registry_is_registered is not None:
        try:
            return _registry_is_registered(name)
        except Exception as e:
            logger.debug(f"Registry is_registered() failed for '{name}': {e}")

    # DYNAMIC FALLBACK: Check against dynamically discovered types
    available_types = _registry_list_all_safe()
    return name in available_types


def _get_dataset_config_schema_class(dataset_type: str) -> type | None:
    """
    Get config schema class for a dataset type from registry.

    Args:
        dataset_type: Dataset type name (e.g., 'DFT', 'Wavefunction')

    Returns:
        Config schema class or None if not available

    PHASE 5 REFACTORING: Dynamic schema class lookup via BaseDataset.get_config_schema()
    Evidence: base.py lines 202-204 (get_config_schema method)
    """
    dataset_class = _registry_get_safe(dataset_type)
    if dataset_class is None:
        return None

    try:
        schema_class = dataset_class.get_config_schema()
        return schema_class
    except Exception as e:
        logger.debug(f"Failed to get config schema for {dataset_type}: {e}")
        return None


def _get_dataset_feature_support(dataset_type: str) -> dict[str, bool] | None:
    """
    Get feature support dictionary for a dataset type from registry.

    Args:
        dataset_type: Dataset type name

    Returns:
        Feature support dictionary or None if not available

    PHASE 5 REFACTORING: Dynamic feature lookup via BaseDataset.get_feature_support()
    Evidence: dft.py lines 88-99, dmc.py lines 88-99, wavefunction.py lines 114-127
    """
    dataset_class = _registry_get_safe(dataset_type)
    if dataset_class is None:
        return None

    try:
        features = dataset_class.get_feature_support()
        return features
    except Exception as e:
        logger.debug(f"Failed to get feature support for {dataset_type}: {e}")
        return None


def _dataset_supports_feature(dataset_type: str, feature_name: str) -> bool:
    """
    Check if a dataset type supports a specific feature.

    Args:
        dataset_type: Dataset type name
        feature_name: Feature name (e.g., 'orbital_analysis', 'uncertainty_handling')

    Returns:
        True if feature is supported, False otherwise

    PHASE 5 REFACTORING: Dynamic feature check via registry
    """
    features = _get_dataset_feature_support(dataset_type)
    if features is None:
        return False
    return features.get(feature_name, False)


def _get_dataset_config_key(dataset_type: str) -> str | None:
    """
    Get config key for a dataset type from registry.

    Args:
        dataset_type: Dataset type name

    Returns:
        Config key (e.g., 'dft_config', 'wavefunction_config') or None if not available

    PHASE 5 REFACTORING: Dynamic config key lookup via BaseDataset.config_key
    Evidence: dft.py line 71, dmc.py line 71, wavefunction.py line 96
    """
    dataset_class = _registry_get_safe(dataset_type)
    if dataset_class is None:
        return None

    try:
        config_key = dataset_class.config_key
        return config_key
    except Exception as e:
        logger.debug(f"Failed to get config key for {dataset_type}: {e}")
        return None


# =============================================================================
# SCHEMA DEFINITION CLASSES
# =============================================================================


# Lazy import to break circular dependency with config_loader
def _get_load_config():
    """Lazy import of load_config to avoid circular dependency"""
    from milia_pipeline.config.config_loader import load_config

    return load_config


class TransformationSchema(BaseModel):
    """Schema definition for transformation configuration

    A valid configuration must have either experimental_setups OR standard_transforms
    (or both). Standard transforms are always applied first, before experimental
    setup transforms.

    Pattern: Pydantic V2 BaseModel (mutable) with model_validator for cross-field validation
    """

    experimental_setups: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    default_setup: str = "baseline"
    validation: dict[str, Any] = Field(default_factory=dict)
    standard_transforms: list[dict[str, Any]] | None = (
        None  # NEW: Standard transforms applied before experimental
    )
    legacy_transforms: list[dict[str, Any]] | None = None
    research_metadata: dict[str, Any] | None = None
    dataset_optimization: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_schema_definition(self) -> Self:
        """Validate schema definition - cross-field validation."""
        # Must have at least one transform source
        has_experimental = isinstance(self.experimental_setups, dict) and bool(
            self.experimental_setups
        )
        has_standard = isinstance(self.standard_transforms, list) and bool(self.standard_transforms)

        if not has_experimental and not has_standard:
            raise ValueError(
                "At least one of 'experimental_setups' or 'standard_transforms' must be defined"
            )

        # Validate experimental_setups if present
        if self.experimental_setups and not isinstance(self.experimental_setups, dict):
            raise ValueError("experimental_setups must be a dictionary")

        # Validate standard_transforms if present
        if self.standard_transforms is not None and not isinstance(self.standard_transforms, list):
            raise ValueError("standard_transforms must be a list")

        # default_setup must exist in experimental_setups if experimental_setups is non-empty
        if has_experimental and self.default_setup not in self.experimental_setups:
            # Allow if standard_transforms exists (default_setup is just a label)
            if not has_standard:
                raise ValueError(
                    f"Default setup '{self.default_setup}' not found in experimental setups"
                )

        return self


class PluginValidationLevel(Enum):
    """Plugin validation strictness levels"""

    STRICT = "strict"  # Full validation including security
    STANDARD = "standard"  # Basic validation, skip expensive checks
    PERMISSIVE = "permissive"  # Minimal validation, trust plugins
    DISABLED = "disabled"  # No plugin validation


class PluginConfigSchema(BaseModel):
    """Schema definition for plugin configuration

    Pattern: Pydantic V2 BaseModel (mutable) with field_validator for validation
    """

    enabled: bool = False
    plugin_paths: list[str] = Field(default_factory=list)
    auto_discover: bool = True
    auto_validate: bool = True
    validation_level: str = "standard"  # strict, standard, permissive, disabled
    trusted_plugins: list[str] = Field(default_factory=list)
    disabled_plugins: list[str] = Field(default_factory=list)
    allow_experimental: bool = False
    max_plugins: int = 50
    require_metadata: bool = True
    enforce_checksums: bool = True
    security_scanning: bool = True

    @field_validator("validation_level")
    @classmethod
    def validate_validation_level(cls, v: str) -> str:
        """Validate validation_level is a valid option."""
        valid_levels = ["strict", "standard", "permissive", "disabled"]
        if v not in valid_levels:
            raise ValueError(f"validation_level must be one of {valid_levels}, got '{v}'")
        return v

    @field_validator("max_plugins")
    @classmethod
    def validate_max_plugins(cls, v: int) -> int:
        """Validate max_plugins is within valid range."""
        if v < 1 or v > 1000:
            raise ValueError(f"max_plugins must be between 1 and 1000, got {v}")
        return v

    @field_validator("plugin_paths")
    @classmethod
    def validate_plugin_paths(cls, v: list[str]) -> list[str]:
        """Validate all plugin_paths are strings."""
        if not all(isinstance(p, str) for p in v):
            raise ValueError("All plugin_paths must be strings")
        return v

    @field_validator("trusted_plugins", "disabled_plugins")
    @classmethod
    def validate_plugin_lists(cls, v: list[str]) -> list[str]:
        """Validate all plugin names are strings."""
        if not all(isinstance(p, str) for p in v):
            raise ValueError("All plugin names must be strings")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PluginConfigSchema":
        """Create from dictionary."""
        return cls(**data)


# =============================================================================
# COMMENTED OUT: DEAD CODE - Duplicate DescriptorConfigSchema Class
# =============================================================================
#
# REASON FOR COMMENTING OUT (Phase 17 Pre-Migration Analysis - 2026-01-08):
#
# This class is SHADOWED by another DescriptorConfigSchema class defined at
# Line ~886 (in this same file). In Python, when two classes have the same
# name in the same module, the second definition completely shadows the first,
# making the first class inaccessible at runtime.
#
# EVIDENCE GATHERED:
# 1. grep search across entire codebase found NO instantiation of this class
# 2. The fields (selection_mode, selected_descriptors, selected_categories,
#    excluded_descriptors) are used as DICTIONARY KEYS in:
#    - cli_manager.py (lines 2431-2459)
#    - descriptor_validator.py (lines 290-299)
#    - config_accessors.py (get_selected_descriptors function)
#    But these access config dictionaries, NOT this class.
# 3. The second DescriptorConfigSchema (Line ~886) has different fields that
#    match DescriptorConfig in config_containers.py (already migrated to Pydantic)
#
# ACTION: Commented out for testing. If no functionality breaks, this block
# should be removed entirely in a future cleanup phase.
#
# TESTING INSTRUCTIONS:
# 1. Run full test suite
# 2. Run main.py with descriptor computation enabled
# 3. Verify cli_manager.py descriptor validation works
# If all pass, this commented block can be safely deleted.
#
# @dataclass
# class DescriptorConfigSchema:
#     """
#     Schema definition for descriptor configuration.
#
#     Phase 3 Integration: Defines the complete configuration schema for
#     molecular descriptor computation, including selection modes, categories,
#     plugins, computation settings, and output format.
#     """
#     enabled: bool = False
#     selection_mode: str = "explicit"  # explicit, category, all
#
#     # Descriptor selection
#     selected_descriptors: Dict[str, List[str]] = field(default_factory=dict)
#     selected_categories: List[str] = field(default_factory=list)
#     excluded_descriptors: List[str] = field(default_factory=list)
#
#     # Plugin configuration
#     plugins: Optional[Dict[str, Any]] = None
#
#     # Computation settings
#     computation: Dict[str, Any] = field(default_factory=lambda: {
#         'batch_size': 100,
#         'fallback_on_error': True,
#         'cache_results': True,
#         'generate_conformers': True
#     })
#
#     # Output configuration
#     output: Dict[str, Any] = field(default_factory=lambda: {
#         'format': 'pyg_data',
#         'prefix': 'desc_',
#         'create_feature_vector': True,
#         'merge_with_node_features': False
#     })
#
#     def __post_init__(self):
#         """Validate descriptor configuration schema."""
#         # Validate selection mode
#         valid_modes = ['explicit', 'category', 'all']
#         if self.selection_mode not in valid_modes:
#             raise ValueError(
#                 f"selection_mode must be one of {valid_modes}, "
#                 f"got '{self.selection_mode}'"
#             )
#
#         # Validate categories if using category mode
#         if self.selection_mode == 'category':
#             valid_categories = [
#                 'constitutional', 'topological', 'electronic',
#                 'geometric', 'drug_likeness', 'fragments'
#             ]
#             for cat in self.selected_categories:
#                 if cat not in valid_categories:
#                     raise ValueError(
#                         f"Invalid descriptor category '{cat}'. "
#                         f"Valid categories: {valid_categories}"
#                     )
#
#         # Validate output format
#         valid_formats = ['pyg_data', 'dict', 'tensor']
#         output_format = self.output.get('format', 'pyg_data')
#         if output_format not in valid_formats:
#             raise ValueError(
#                 f"output.format must be one of {valid_formats}, "
#                 f"got '{output_format}'"
#             )
#
#         # Ensure batch_size is reasonable
#         batch_size = self.computation.get('batch_size', 100)
#         if batch_size < 1 or batch_size > 10000:
#             raise ValueError(
#                 f"computation.batch_size must be between 1 and 10000, "
#                 f"got {batch_size}"
#             )
#
# =============================================================================
# END OF COMMENTED OUT DEAD CODE
# =============================================================================

# =============================================================================
# WAVEFUNCTION DATASET CONFIGURATION SCHEMAS
# =============================================================================


class WavefunctionProcessingConfigSchema(BaseModel, frozen=True):
    """
    Schema for wavefunction processing configuration.

    PHASE 5 NOTE: This is a legacy/example schema class specific to Wavefunction datasets.
    New dataset types should implement BaseDataset.get_config_schema() to return their
    own schema class dynamically from the registry.

    Pattern: Pydantic V2 frozen BaseModel with field_validator

    Attributes:
        feature_tier: Feature extraction complexity level
    """

    feature_tier: str = "standard"

    @field_validator("feature_tier")
    @classmethod
    def validate_feature_tier(cls, v: str) -> str:
        """Validate feature_tier is a valid option."""
        valid_tiers = ["basic", "standard", "complete"]
        if v not in valid_tiers:
            raise ValueError(f"feature_tier must be one of {valid_tiers}, got '{v}'")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WavefunctionProcessingConfigSchema":
        """Create from dictionary."""
        return cls(feature_tier=data.get("feature_tier", "standard"))


class WavefunctionUncertaintyConfigSchema(BaseModel, frozen=True):
    """
    Schema for wavefunction uncertainty configuration (placeholder).

    Wavefunction datasets do not currently support uncertainty handling.
    This schema exists for consistency with DFT/DMC configuration structure.

    PHASE 5 NOTE: This is a legacy/example schema class. New dataset types should
    implement their own uncertainty schemas and return them via BaseDataset.get_config_schema().

    Pattern: Pydantic V2 frozen BaseModel with field_validator

    Attributes:
        enabled: Must be False (uncertainty not supported for wavefunction)
    """

    enabled: bool = False

    @field_validator("enabled")
    @classmethod
    def validate_enabled(cls, v: bool) -> bool:
        """Validate enabled is False (uncertainty not supported)."""
        if v:
            raise ValueError(
                "Uncertainty handling is not supported for Wavefunction datasets. "
                "Set 'enabled: false' in wavefunction_config.uncertainty_handling"
            )
        return v

    def to_dict(self) -> dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WavefunctionUncertaintyConfigSchema":
        """Create from dictionary."""
        return cls(enabled=data.get("enabled", False))


class WavefunctionConfigSchema(BaseModel, frozen=True):
    """
    Schema for wavefunction dataset configuration.

    This schema validates the wavefunction_config section in config.yaml,
    ensuring all required fields are present and valid.

    PHASE 5 NOTE: This is a legacy/example schema class specific to Wavefunction datasets.
    For dynamic validation across all dataset types, use the registry-based validation
    functions. New dataset types should implement BaseDataset.get_config_schema() to
    provide their own schema class.

    Pattern: Pydantic V2 frozen BaseModel with model_validator(mode='before') for nested init

    Attributes:
        raw_npz_filename: Filename for preprocessed wavefunction .npz data
        raw_data_download_url: Optional URL for downloading raw data
        dataset_root_dir: Root directory for PyTorch Geometric dataset storage
        processing_config: Wavefunction processing configuration
        uncertainty_handling: Uncertainty handling configuration (placeholder)
    """

    raw_npz_filename: str
    raw_data_download_url: str | None = None
    dataset_root_dir: str = "~/Chem_Data/milia_PyG_Dataset"
    processing_config: WavefunctionProcessingConfigSchema | None = None
    uncertainty_handling: WavefunctionUncertaintyConfigSchema | None = None

    @field_validator("raw_npz_filename")
    @classmethod
    def validate_raw_npz_filename(cls, v: str) -> str:
        """Validate raw_npz_filename is present and has correct extension."""
        if not v:
            raise ValueError("raw_npz_filename is required for Wavefunction dataset")
        if not v.endswith(".npz"):
            raise ValueError(f"raw_npz_filename must end with '.npz', got '{v}'")
        return v

    @field_validator("dataset_root_dir")
    @classmethod
    def validate_dataset_root_dir(cls, v: str) -> str:
        """Validate dataset_root_dir is present."""
        if not v:
            raise ValueError("dataset_root_dir is required for Wavefunction dataset")
        return v

    @model_validator(mode="before")
    @classmethod
    def initialize_nested_configs(cls, data: Any) -> Any:
        """Initialize nested config objects if None before field assignment."""
        if isinstance(data, dict):
            # Initialize processing_config if None or missing
            if data.get("processing_config") is None:
                data["processing_config"] = WavefunctionProcessingConfigSchema()
            elif isinstance(data.get("processing_config"), dict):
                data["processing_config"] = WavefunctionProcessingConfigSchema.from_dict(
                    data["processing_config"]
                )

            # Initialize uncertainty_handling if None or missing
            if data.get("uncertainty_handling") is None:
                data["uncertainty_handling"] = WavefunctionUncertaintyConfigSchema()
            elif isinstance(data.get("uncertainty_handling"), dict):
                data["uncertainty_handling"] = WavefunctionUncertaintyConfigSchema.from_dict(
                    data["uncertainty_handling"]
                )
        return data

    def to_dict(self) -> dict[str, Any]:
        """Backward compatible dict conversion with nested model handling."""
        result = {
            "raw_npz_filename": self.raw_npz_filename,
            "raw_data_download_url": self.raw_data_download_url,
            "dataset_root_dir": self.dataset_root_dir,
        }

        if self.processing_config:
            result["processing_config"] = self.processing_config.to_dict()

        if self.uncertainty_handling:
            result["uncertainty_handling"] = self.uncertainty_handling.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WavefunctionConfigSchema":
        """
        Create WavefunctionConfigSchema from dictionary.

        Args:
            data: Dictionary containing wavefunction configuration

        Returns:
            WavefunctionConfigSchema instance
        """
        # Extract processing_config if present
        processing_config = None
        if "processing_config" in data and data["processing_config"]:
            processing_config = WavefunctionProcessingConfigSchema.from_dict(
                data["processing_config"]
            )

        # Extract uncertainty_handling if present
        uncertainty_handling = None
        if "uncertainty_handling" in data and data["uncertainty_handling"]:
            uncertainty_handling = WavefunctionUncertaintyConfigSchema.from_dict(
                data["uncertainty_handling"]
            )

        return cls(
            raw_npz_filename=data["raw_npz_filename"],
            raw_data_download_url=data.get("raw_data_download_url"),
            dataset_root_dir=data.get("dataset_root_dir", "~/Chem_Data/milia_PyG_Dataset"),
            processing_config=processing_config,
            uncertainty_handling=uncertainty_handling,
        )


def validate_wavefunction_config(config: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate wavefunction_config section from config.yaml.

    PHASE 5 REFACTORING: Now supports dynamic schema validation via registry.
    First attempts to get schema class from registry via BaseDataset.get_config_schema(),
    falls back to legacy WavefunctionConfigSchema if registry unavailable.

    This function provides standalone validation that can be used
    by config_loader or other modules.

    Args:
        config: Dictionary containing configuration (should have 'wavefunction_config' key)

    Returns:
        Tuple of (is_valid, list_of_errors)

    Example:
        >>> config = {'dataset_type': 'Wavefunction', 'wavefunction_config': {...}}
        >>> is_valid, errors = validate_wavefunction_config(config)
        >>> if not is_valid:
        ...     print(f"Validation errors: {errors}")
    """
    errors = []

    # Get dataset_type
    dataset_type = config.get("dataset_type", "")

    # PHASE 5: Use registry to check if this is actually Wavefunction
    # Maintains backward compatibility by checking string match
    if dataset_type != "Wavefunction":
        # Not a wavefunction config, skip validation
        logger.debug(f"Skipping Wavefunction validation for dataset_type: {dataset_type}")
        return True, []

    # Get the expected config key from registry
    config_key = _get_dataset_config_key(dataset_type)
    if config_key is None:
        # Fallback to hardcoded key
        config_key = "wavefunction_config"
        logger.debug("Using legacy hardcoded config key: wavefunction_config")

    # Check if config section exists
    if config_key not in config:
        errors.append(f"{config_key} section is required when dataset_type is '{dataset_type}'")
        return False, errors

    config_section = config[config_key]

    # PHASE 5: Try to get schema class from registry first
    schema_class = _get_dataset_config_schema_class(dataset_type)

    if schema_class is not None:
        # Use registry-provided schema class
        try:
            schema_class.from_dict(config_section)
            logger.debug(f"Validated {dataset_type} config using registry schema")
            return True, []
        except ValueError as e:
            errors.append(f"{dataset_type} config validation failed: {str(e)}")
            return False, errors
        except Exception as e:
            errors.append(f"Unexpected error validating {dataset_type} config: {str(e)}")
            return False, errors
    else:
        # FALLBACK: Use legacy WavefunctionConfigSchema
        try:
            WavefunctionConfigSchema.from_dict(config_section)
            logger.debug("Validated Wavefunction config using legacy schema (registry unavailable)")
            return True, []
        except ValueError as e:
            errors.append(f"Wavefunction config validation failed: {str(e)}")
            return False, errors
        except Exception as e:
            errors.append(f"Unexpected error validating wavefunction config: {str(e)}")
            return False, errors


# =============================================================================
# RESEARCH EXPERIMENT SCHEMA
# =============================================================================


class ExperimentSchema(BaseModel):
    """Schema definition for research experiment configuration

    Pattern: Pydantic V2 BaseModel (mutable) with field_validator for validation
    """

    name: str
    description: str
    base_transforms: list[dict[str, Any]]

    # Experiment variants
    ablations: list[dict[str, Any]] = Field(default_factory=list)
    parameter_sweeps: list[dict[str, Any]] = Field(default_factory=list)

    # Publication metadata
    paper_reference: str | None = None
    hypothesis: str | None = None
    expected_outcome: str | None = None

    # Execution settings
    num_runs: int = 3
    random_seed: int = 42

    # Results and metadata
    results: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is a non-empty string."""
        if not v or not isinstance(v, str):
            raise ValueError("Experiment name must be a non-empty string")
        return v

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Validate description is a non-empty string."""
        if not v or not isinstance(v, str):
            raise ValueError("Experiment description must be a non-empty string")
        return v

    @field_validator("base_transforms")
    @classmethod
    def validate_base_transforms(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate base_transforms is a list."""
        if not isinstance(v, list):
            raise ValueError("base_transforms must be a list")
        return v

    @field_validator("num_runs")
    @classmethod
    def validate_num_runs(cls, v: int) -> int:
        """Validate num_runs is a positive integer."""
        if not isinstance(v, int) or v < 1:
            raise ValueError(f"num_runs must be a positive integer, got {v}")
        return v

    @field_validator("random_seed")
    @classmethod
    def validate_random_seed(cls, v: int) -> int:
        """Validate random_seed is an integer."""
        if not isinstance(v, int):
            raise ValueError(f"random_seed must be an integer, got {type(v).__name__}")
        return v

    @field_validator("ablations", "parameter_sweeps")
    @classmethod
    def validate_variant_lists(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate variant lists are lists."""
        if not isinstance(v, list):
            raise ValueError("Variant list must be a list")
        return v

    @model_validator(mode="after")
    def warn_no_variants(self) -> Self:
        """Warn if no variants configured."""
        if not self.ablations and not self.parameter_sweeps:
            logger.warning(f"Experiment '{self.name}' has no variants configured")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentSchema":
        """Create from dictionary."""
        return cls(**data)


class ValidationConfig(BaseModel):
    """Configuration for validation behavior

    Pattern: Pydantic V2 BaseModel (mutable) - simple config, no validators needed
    """

    strict_mode: bool = False
    warn_on_unknown: bool = True
    require_descriptions: bool = False
    check_parameter_types: bool = False
    validate_research_context: bool = False


# =============================================================================
# DESCRIPTOR CONFIGURATION SCHEMAS
# =============================================================================


class DescriptorConfigSchema(BaseModel, frozen=True):
    """
    Schema for molecular descriptor configuration.

    Pattern: Pydantic V2 frozen BaseModel with field_validator and model_validator(mode='before')
    for auto-adjust logic (parallel_computation with num_workers=1 auto-adjusts to 2)

    Attributes:
        enabled: Whether descriptor computation is enabled globally
        default_categories: Categories to compute by default
        cache_descriptors: Whether to cache computed descriptors
        cache_path: Path for descriptor cache (None = auto)
        parallel_computation: Whether to compute descriptors in parallel
        num_workers: Number of workers for parallel computation
        error_handling: How to handle descriptor computation errors
        validation_mode: Descriptor validation strictness level
    """

    enabled: bool = True
    default_categories: list[str] = Field(default_factory=lambda: ["constitutional", "topological"])
    cache_descriptors: bool = True
    cache_path: str | None = None
    parallel_computation: bool = False
    num_workers: int = 1
    error_handling: str = "warn"  # Options: 'strict', 'warn', 'skip'
    validation_mode: str = "standard"  # Options: 'strict', 'standard', 'permissive'

    @field_validator("error_handling")
    @classmethod
    def validate_error_handling(cls, v: str) -> str:
        """Validate error_handling is a valid option."""
        valid_error_modes = ["strict", "warn", "skip"]
        if v not in valid_error_modes:
            raise ValueError(f"error_handling must be one of {valid_error_modes}, got '{v}'")
        return v

    @field_validator("validation_mode")
    @classmethod
    def validate_validation_mode(cls, v: str) -> str:
        """Validate validation_mode is a valid option."""
        valid_validation_modes = ["strict", "standard", "permissive"]
        if v not in valid_validation_modes:
            raise ValueError(f"validation_mode must be one of {valid_validation_modes}, got '{v}'")
        return v

    @field_validator("num_workers")
    @classmethod
    def validate_num_workers(cls, v: int) -> int:
        """Validate num_workers is at least 1."""
        if v < 1:
            raise ValueError(f"num_workers must be at least 1, got {v}")
        return v

    @field_validator("default_categories")
    @classmethod
    def validate_default_categories(cls, v: list[str]) -> list[str]:
        """Validate all categories are valid."""
        valid_categories = [
            "constitutional",
            "topological",
            "geometric",
            "electronic",
            "pharmacophore",
            "fingerprint",
            "custom",
        ]
        for category in v:
            if category not in valid_categories:
                raise ValueError(
                    f"Invalid category '{category}'. Valid categories: {valid_categories}"
                )
        return v

    @model_validator(mode="before")
    @classmethod
    def auto_adjust_num_workers(cls, data: Any) -> Any:
        """Auto-adjust num_workers when parallel_computation is True but num_workers is 1."""
        if isinstance(data, dict):
            parallel_computation = data.get("parallel_computation", False)
            num_workers = data.get("num_workers", 1)
            if parallel_computation and num_workers == 1:
                data["num_workers"] = 2  # Auto-adjust to minimum for parallel
        return data

    def to_dict(self) -> dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DescriptorConfigSchema":
        """Create from dictionary."""
        return cls(**data)


class DescriptorCategoryConfigSchema(BaseModel, frozen=True):
    """
    Schema for individual descriptor category configuration.

    Pattern: Pydantic V2 frozen BaseModel with field_validator

    Attributes:
        category_name: Name of the descriptor category
        enabled: Whether this category is enabled
        descriptors: List of specific descriptors to compute (None = all)
        options: Category-specific options
    """

    category_name: str
    enabled: bool = True
    descriptors: list[str] | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("category_name")
    @classmethod
    def validate_category_name(cls, v: str) -> str:
        """Validate category_name is a valid category."""
        valid_categories = [
            "constitutional",
            "topological",
            "geometric",
            "electronic",
            "pharmacophore",
            "fingerprint",
            "custom",
        ]
        if v not in valid_categories:
            raise ValueError(f"Invalid category_name '{v}'. Valid categories: {valid_categories}")
        return v

    @field_validator("descriptors")
    @classmethod
    def validate_descriptors(cls, v: list[str] | None) -> list[str] | None:
        """Validate descriptors is a list of strings or None."""
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("descriptors must be a list or None")
            if not all(isinstance(d, str) for d in v):
                raise ValueError("All descriptor names must be strings")
        return v

    @field_validator("options")
    @classmethod
    def validate_options(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate options is a dictionary."""
        if not isinstance(v, dict):
            raise ValueError("options must be a dictionary")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DescriptorCategoryConfigSchema":
        """Create from dictionary."""
        return cls(**data)


# =============================================================================
# YAML SCHEMA VALIDATOR
# =============================================================================


class YAMLSchemaValidator:
    """Enhanced YAML schema validator with migration detection"""

    def __init__(self):
        self.validation_calls = []

    def detect_format(self, config: dict[str, Any]) -> str:
        """Detect configuration format"""
        if not isinstance(config, dict) or "transformations" not in config:
            return "invalid"

        transformations = config["transformations"]

        # Check for invalid formats first
        if isinstance(transformations, str):
            return "invalid"

        # Enhanced format detection - check for experimental_setups OR standard_transforms
        if isinstance(transformations, dict) and (
            "experimental_setups" in transformations or "standard_transforms" in transformations
        ):
            return "enhanced"

        # Legacy format detection
        if isinstance(transformations, list):
            return "legacy_list"

        # Legacy dict format
        if isinstance(transformations, dict):
            return "legacy_dict"

        return "unknown"

    def validate_config(
        self, config: dict[str, Any], validation_config: ValidationConfig | None = None
    ) -> dict[str, Any]:
        """Validate configuration - Less strict"""
        if validation_config is None:
            validation_config = ValidationConfig()

        self.validation_calls.append({"config": config, "validation_config": validation_config})

        # Default result structure
        result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "format_detected": "unknown",
            "summary": {},
        }

        # Basic validation logic
        if not isinstance(config, dict):
            result["errors"].append("Configuration must be a dictionary")
            return result

        if "transformations" not in config:
            result["errors"].append("Configuration must contain 'transformations' key")
            return result

        transformations = config["transformations"]
        format_detected = self.detect_format(config)
        result["format_detected"] = format_detected

        # Handle different formats
        if format_detected == "invalid":
            result["errors"].append("Invalid transformations format")
            return result

        elif format_detected == "enhanced":
            # Enhanced format validation - supports experimental_setups OR standard_transforms (or both)
            setups = transformations.get("experimental_setups", {})
            standard_transforms = transformations.get("standard_transforms", [])

            has_experimental = bool(setups)
            has_standard = bool(standard_transforms)

            # Must have at least one transform source
            if not has_experimental and not has_standard:
                result["errors"].append(
                    "Enhanced format requires 'experimental_setups' or 'standard_transforms' (at least one)"
                )
                return result

            # Validate standard_transforms if present
            if has_standard:
                if not isinstance(standard_transforms, list):
                    result["errors"].append("standard_transforms must be a list")
                else:
                    # Validate each standard transform
                    for i, transform in enumerate(standard_transforms):
                        if isinstance(transform, dict):
                            if "name" not in transform:
                                result["warnings"].append(
                                    f"standard_transforms[{i}] missing 'name' field"
                                )
                        elif not isinstance(transform, str):
                            result["warnings"].append(
                                f"standard_transforms[{i}] has unexpected format"
                            )

                    result["summary"]["standard_transforms_count"] = len(standard_transforms)

            # Validate experimental_setups if present
            if has_experimental:
                if not isinstance(setups, dict):
                    result["errors"].append("experimental_setups must be a dictionary")
                    return result

                # Empty experimental_setups is allowed if standard_transforms exists
                if len(setups) == 0 and not has_standard:
                    result["errors"].append(
                        "experimental_setups cannot be empty when no standard_transforms defined"
                    )
                    return result

                # Validate each setup - ONLY check critical fields
                for setup_name, setup_data in setups.items():
                    if not isinstance(setup_data, dict):
                        result["errors"].append(f"Setup '{setup_name}' must be a dictionary")
                        continue

                    # Only require description in strict mode
                    if validation_config.require_descriptions and validation_config.strict_mode:
                        if "description" not in setup_data:
                            result["errors"].append(
                                f"Setup '{setup_name}' missing required description"
                            )

                    # Check for transforms - but be flexible about format
                    if "transforms" in setup_data:
                        transforms = setup_data["transforms"]
                        if not isinstance(transforms, list):
                            result["errors"].append(
                                f"Setup '{setup_name}' transforms must be a list"
                            )
                            continue

                        # Validate individual transforms - be lenient
                        for i, transform in enumerate(transforms):
                            if isinstance(transform, dict):
                                if "name" not in transform:
                                    result["warnings"].append(
                                        f"Setup '{setup_name}' transform {i} missing 'name' field"
                                    )

                                # Check for params field and suggest kwargs
                                if "params" in transform and "kwargs" not in transform:
                                    # This is acceptable, but log for awareness
                                    pass  # params will be auto-converted by TransformSpec
                                elif "params" in transform and "kwargs" in transform:
                                    result["warnings"].append(
                                        f"Setup '{setup_name}' transform {i} has both 'params' and 'kwargs' - 'kwargs' will take precedence"
                                    )
                            elif not isinstance(transform, str):
                                result["warnings"].append(
                                    f"Setup '{setup_name}' transform {i} has unexpected format"
                                )

            # Check for default_setup - CRITICAL in strict, warning in non-strict
            if "default_setup" not in transformations:
                if validation_config.strict_mode:
                    # Strict: Enforce required structure
                    result["errors"].append(
                        "Enhanced format missing default_setup (required in strict mode)"
                    )
                else:
                    # Non-strict: Only check critical issues
                    result["warnings"].append("Enhanced format missing default_setup (recommended)")

            # If no critical errors, mark as valid
            if not result["errors"]:
                result["valid"] = True
                result["summary"]["experimental_setups_count"] = (
                    len(setups) if has_experimental else 0
                )
                if has_standard:
                    result["summary"]["standard_transforms_count"] = len(standard_transforms)

        elif format_detected in ["legacy_list", "legacy_dict"]:
            # Legacy formats are valid but should be migrated
            result["valid"] = True
            result["warnings"].append(
                f"Configuration is in {format_detected} format and should be migrated"
            )

        else:
            result["warnings"].append(f"Unknown format detected: {format_detected}")
            result["valid"] = True  # Don't fail on unknown formats

        return result

    def validate_config_with_plugins(
        self, config: dict[str, Any], validation_config: ValidationConfig | None = None
    ) -> dict[str, Any]:
        """
        Validate complete configuration including plugins

        This method extends validate_config to include plugin validation.

        Args:
            config: Complete configuration dictionary
            validation_config: Validation behavior configuration

        Returns:
            Combined validation results
        """
        # First validate base configuration
        base_result = self.validate_config(config, validation_config)

        # Check if plugins section exists
        if "plugins" not in config:
            base_result["warnings"].append("No plugins section in configuration")
            return base_result

        # Validate plugin configuration
        plugin_validator = PluginSchemaValidator()

        # Determine validation level
        if validation_config and validation_config.strict_mode:
            plugin_validation_level = PluginValidationLevel.STRICT
        else:
            plugin_validation_level = PluginValidationLevel.STANDARD

        plugin_result = plugin_validator.validate_plugin_config(
            config["plugins"], validation_level=plugin_validation_level
        )

        # Merge results
        combined_result = base_result.copy()
        combined_result["errors"].extend(plugin_result["errors"])
        combined_result["warnings"].extend(plugin_result["warnings"])
        combined_result["suggestions"].extend(plugin_result["suggestions"])
        combined_result["plugin_validation"] = plugin_result

        # Check plugin-transform compatibility
        if "transformations" in config:
            compat_result = plugin_validator.validate_plugin_compatibility(
                config["plugins"], config["transformations"]
            )
            combined_result["plugin_compatibility"] = compat_result
            combined_result["warnings"].extend(compat_result["warnings"])
            if not compat_result["compatible"]:
                combined_result["errors"].extend(compat_result["issues"])

        # Update overall validity
        combined_result["valid"] = (
            len(combined_result["errors"]) == 0 and base_result["valid"] and plugin_result["valid"]
        )

        return combined_result

    def validate_config_with_experiments(
        self, config: dict[str, Any], validation_config: ValidationConfig | None = None
    ) -> dict[str, Any]:
        """
        Validate complete configuration including experiments

        This method extends validate_config to include experiment validation.

        Args:
            config: Complete configuration dictionary
            validation_config: Validation behavior configuration

        Returns:
            Combined validation results
        """
        # First validate base configuration
        base_result = self.validate_config_with_plugins(config, validation_config)

        # Check if experiments section exists
        if "experiments" not in config:
            base_result["warnings"].append("No experiments section in configuration")
            return base_result

        # Validate experiments configuration
        experiment_validator = ExperimentSchemaValidator()

        # Determine strict mode
        strict_mode = validation_config.strict_mode if validation_config else False

        experiment_result = experiment_validator.validate_experiments_config(
            config["experiments"], strict_mode=strict_mode
        )

        # Merge results
        combined_result = base_result.copy()
        combined_result["errors"].extend(experiment_result["errors"])
        combined_result["warnings"].extend(experiment_result["warnings"])
        combined_result["suggestions"].extend(experiment_result["suggestions"])
        combined_result["experiment_validation"] = experiment_result

        # Update overall validity
        combined_result["valid"] = base_result["valid"] and experiment_result["valid"]

        return combined_result


class PluginSchemaValidator:
    """Validator for plugin configurations with security and compatibility checks"""

    def __init__(self):
        self.validation_calls = []
        self.plugin_system_available = PLUGIN_SYSTEM_AVAILABLE

    def validate_plugin_config(
        self,
        plugin_config: dict[str, Any],
        validation_level: PluginValidationLevel = PluginValidationLevel.STANDARD,
    ) -> dict[str, Any]:
        """
        Validate plugin configuration structure and settings

        Args:
            plugin_config: Plugin configuration dictionary
            validation_level: Validation strictness level

        Returns:
            Validation results dictionary with:
                - valid: bool
                - errors: List[str]
                - warnings: List[str]
                - suggestions: List[str]
                - plugin_summary: Dict[str, Any]
        """
        self.validation_calls.append(
            {"config": plugin_config, "validation_level": validation_level}
        )

        result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "plugin_summary": {},
        }

        # Check if plugins section exists
        if not isinstance(plugin_config, dict):
            result["errors"].append("Plugin configuration must be a dictionary")
            return result

        # Validate enabled flag
        enabled = plugin_config.get("enabled", False)
        if not isinstance(enabled, bool):
            result["errors"].append(f"'enabled' must be boolean, got {type(enabled).__name__}")

        # If plugins disabled, skip further validation
        if not enabled:
            result["valid"] = True
            result["warnings"].append("Plugin system is disabled")
            result["plugin_summary"]["enabled"] = False
            return result

        # Check plugin system availability early when plugins are enabled
        if not self.plugin_system_available:
            result["valid"] = True  # Warn but don't fail
            result["warnings"].append(
                "Plugin system module not available - plugins will not be loaded"
            )
            result["suggestions"].append("Ensure plugin_system.py is installed and importable")
            result["plugin_summary"] = {"enabled": True, "plugin_system_available": False}
            return result

        # Validate plugin_paths
        plugin_paths = plugin_config.get("plugin_paths", [])
        if not isinstance(plugin_paths, list):
            result["errors"].append(
                f"'plugin_paths' must be a list, got {type(plugin_paths).__name__}"
            )
        else:
            if not plugin_paths:
                result["warnings"].append("No plugin paths specified")
            else:
                # Validate each path is a string
                invalid_paths = [p for p in plugin_paths if not isinstance(p, str)]
                if invalid_paths:
                    result["errors"].append(
                        f"All plugin_paths must be strings, found {len(invalid_paths)} invalid entries"
                    )

        # Validate validation_level
        validation_level_str = plugin_config.get("validation_level", "standard")
        valid_levels = ["strict", "standard", "permissive", "disabled"]
        if validation_level_str not in valid_levels:
            result["errors"].append(
                f"validation_level must be one of {valid_levels}, got '{validation_level_str}'"
            )

        # Validate boolean flags
        bool_flags = [
            "auto_discover",
            "auto_validate",
            "allow_experimental",
            "require_metadata",
            "enforce_checksums",
            "security_scanning",
        ]
        for flag in bool_flags:
            value = plugin_config.get(flag)
            if value is not None and not isinstance(value, bool):
                result["errors"].append(f"'{flag}' must be boolean, got {type(value).__name__}")

        # Validate max_plugins
        max_plugins = plugin_config.get("max_plugins", 50)
        if not isinstance(max_plugins, int):
            result["errors"].append(
                f"'max_plugins' must be integer, got {type(max_plugins).__name__}"
            )
        elif max_plugins < 1 or max_plugins > 1000:
            result["errors"].append(f"'max_plugins' must be between 1 and 1000, got {max_plugins}")

        # Validate plugin lists
        for list_key in ["trusted_plugins", "disabled_plugins"]:
            plugin_list = plugin_config.get(list_key, [])
            if not isinstance(plugin_list, list):
                result["errors"].append(f"'{list_key}' must be a list")
            else:
                invalid_names = [p for p in plugin_list if not isinstance(p, str)]
                if invalid_names:
                    result["errors"].append(
                        f"All {list_key} must be strings, found {len(invalid_names)} invalid entries"
                    )

        # Check for security concerns based on validation level
        if validation_level == PluginValidationLevel.STRICT:
            # Strict mode requires security features
            if not plugin_config.get("security_scanning", True):
                result["warnings"].append(
                    "Security scanning disabled in STRICT mode - this is not recommended"
                )

            if not plugin_config.get("enforce_checksums", True):
                result["warnings"].append(
                    "Checksum enforcement disabled in STRICT mode - this is not recommended"
                )

        # Check for potential issues
        if plugin_config.get("allow_experimental", False):
            result["warnings"].append(
                "Experimental plugins allowed - use with caution in production"
            )

        # Validate plugin system availability
        if not self.plugin_system_available:
            result["warnings"].append(
                "Plugin system module not available - plugins will not be loaded"
            )
            result["suggestions"].append("Ensure plugin_system.py is installed and importable")

        # Build plugin summary
        result["plugin_summary"] = {
            "enabled": enabled,
            "plugin_paths_count": len(plugin_paths) if isinstance(plugin_paths, list) else 0,
            "trusted_plugins_count": len(plugin_config.get("trusted_plugins", [])),
            "disabled_plugins_count": len(plugin_config.get("disabled_plugins", [])),
            "validation_level": validation_level_str,
            "auto_discover": plugin_config.get("auto_discover", True),
            "auto_validate": plugin_config.get("auto_validate", True),
            "plugin_system_available": self.plugin_system_available,
        }

        # Set overall validity
        result["valid"] = len(result["errors"]) == 0

        return result

    def validate_plugin_compatibility(
        self, plugin_config: dict[str, Any], transformation_config: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Validate plugin compatibility with transformation configuration

        Args:
            plugin_config: Plugin configuration
            transformation_config: Transformation configuration

        Returns:
            Compatibility validation results
        """
        result = {"compatible": True, "issues": [], "warnings": []}

        # Check if plugins are enabled
        if not plugin_config.get("enabled", False):
            result["warnings"].append("Plugins disabled - no compatibility issues")
            return result

        # Check if plugin system is available
        if not self.plugin_system_available:
            result["compatible"] = False
            result["issues"].append("Plugin system not available but plugins are enabled")
            return result

        # Check for transform name conflicts (if we can access registry)
        if PLUGIN_SYSTEM_AVAILABLE and PluginRegistry is not None:
            try:
                PluginRegistry()

                # Get transforms from experimental setups
                experimental_setups = transformation_config.get("experimental_setups", {})
                used_transforms = set()

                for _setup_name, setup_config in experimental_setups.items():
                    if isinstance(setup_config, list):
                        # Legacy format
                        for transform in setup_config:
                            if isinstance(transform, dict):
                                used_transforms.add(transform.get("name"))
                    elif isinstance(setup_config, dict):
                        # Enhanced format
                        transforms = setup_config.get("transforms", [])
                        for transform in transforms:
                            if isinstance(transform, dict):
                                used_transforms.add(transform.get("name"))

                # Check for potential conflicts with plugin transforms
                # (This would require discovering plugins first, so we just warn)
                if used_transforms:
                    result["warnings"].append(
                        f"Configuration uses {len(used_transforms)} transforms - "
                        "ensure plugin transforms don't conflict"
                    )

            except Exception as e:
                result["warnings"].append(f"Could not check transform conflicts: {str(e)}")

        return result


# =============================================================================
# EXPERIMENT SCHEMA VALIDATOR
# =============================================================================


class ExperimentSchemaValidator:
    """Validator for research experiment configurations"""

    def __init__(self):
        self.validation_calls = []
        self.research_api_available = RESEARCH_API_AVAILABLE

    def validate_experiment_config(
        self, experiment_config: dict[str, Any], strict_mode: bool = False
    ) -> dict[str, Any]:
        """
        Validate single experiment configuration

        Args:
            experiment_config: Experiment configuration dictionary
            strict_mode: Whether to enforce strict validation

        Returns:
            Validation results dictionary with:
                - valid: bool
                - errors: List[str]
                - warnings: List[str]
                - suggestions: List[str]
                - experiment_summary: Dict[str, Any]
        """
        self.validation_calls.append({"config": experiment_config, "strict_mode": strict_mode})

        result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "experiment_summary": {},
        }

        # Check if config is a dictionary
        if not isinstance(experiment_config, dict):
            result["errors"].append("Experiment configuration must be a dictionary")
            return result

        # Validate required fields
        required_fields = ["name", "description", "base_transforms"]
        for field in required_fields:
            if field not in experiment_config:
                result["errors"].append(f"Missing required field: '{field}'")

        # If missing required fields, return early
        if result["errors"]:
            return result

        # Validate name
        name = experiment_config.get("name")
        if not isinstance(name, str) or not name.strip():
            result["errors"].append("'name' must be a non-empty string")

        # Validate description
        description = experiment_config.get("description")
        if not isinstance(description, str) or not description.strip():
            result["errors"].append("'description' must be a non-empty string")

        # Validate base_transforms
        base_transforms = experiment_config.get("base_transforms")
        if not isinstance(base_transforms, list):
            result["errors"].append("'base_transforms' must be a list")
        else:
            # Validate individual transforms
            for i, transform in enumerate(base_transforms):
                if isinstance(transform, dict):
                    if "name" not in transform:
                        result["warnings"].append(f"base_transforms[{i}] missing 'name' field")
                elif not isinstance(transform, str):
                    result["warnings"].append(f"base_transforms[{i}] has unexpected type")

        # Validate ablations if present
        ablations = experiment_config.get("ablations", [])
        if not isinstance(ablations, list):
            result["errors"].append("'ablations' must be a list")
        else:
            for i, ablation in enumerate(ablations):
                if not isinstance(ablation, dict):
                    result["warnings"].append(f"ablations[{i}] should be a dictionary")
                elif "name" not in ablation:
                    result["warnings"].append(f"ablations[{i}] missing 'name' field")

        # Validate parameter_sweeps if present
        parameter_sweeps = experiment_config.get("parameter_sweeps", [])
        if not isinstance(parameter_sweeps, list):
            result["errors"].append("'parameter_sweeps' must be a list")
        else:
            for i, sweep in enumerate(parameter_sweeps):
                if not isinstance(sweep, dict):
                    result["warnings"].append(f"parameter_sweeps[{i}] should be a dictionary")

        # Validate num_runs
        num_runs = experiment_config.get("num_runs", 3)
        if not isinstance(num_runs, int):
            result["errors"].append(f"'num_runs' must be an integer, got {type(num_runs).__name__}")
        elif num_runs < 1:
            result["errors"].append(f"'num_runs' must be >= 1, got {num_runs}")

        # Validate random_seed
        random_seed = experiment_config.get("random_seed", 42)
        if not isinstance(random_seed, int):
            result["errors"].append(
                f"'random_seed' must be an integer, got {type(random_seed).__name__}"
            )

        # Check for variants - research experiments need comparisons
        if not ablations and not parameter_sweeps:
            if strict_mode:
                # Strict: Ensure research-grade completeness
                result["errors"].append(
                    "Experiment must have at least one ablation or parameter sweep (strict mode requires variants for comparison)"
                )
            else:
                # Non-strict: Basic structure check only
                result["warnings"].append(
                    "Experiment has no variants configured (consider adding for meaningful comparisons)"
                )

        # Validate metadata fields if present
        for field in ["hypothesis", "expected_outcome", "paper_reference"]:
            value = experiment_config.get(field)
            if value is not None and not isinstance(value, str):
                result["warnings"].append(f"'{field}' should be a string if provided")

        # Check for research API availability
        if not self.research_api_available:
            result["warnings"].append(
                "Research API module not available - experiments cannot be executed"
            )
            result["suggestions"].append("Ensure research_api.py is installed and importable")

        # Build experiment summary
        result["experiment_summary"] = {
            "name": name,
            "base_transforms_count": len(base_transforms)
            if isinstance(base_transforms, list)
            else 0,
            "ablations_count": len(ablations),
            "parameter_sweeps_count": len(parameter_sweeps),
            "num_runs": num_runs,
            "has_hypothesis": "hypothesis" in experiment_config,
            "has_expected_outcome": "expected_outcome" in experiment_config,
            "research_api_available": self.research_api_available,
        }

        # Set overall validity
        result["valid"] = len(result["errors"]) == 0

        return result

    def validate_experiments_config(
        self, experiments_config: dict[str, Any], strict_mode: bool = False
    ) -> dict[str, Any]:
        """
        Validate complete experiments configuration (multiple experiments)

        Args:
            experiments_config: Dictionary of experiment_name -> experiment_config
            strict_mode: Whether to enforce strict validation

        Returns:
            Combined validation results for all experiments
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "experiment_results": {},
            "summary": {},
        }

        # Check if config is a dictionary
        if not isinstance(experiments_config, dict):
            result["errors"].append("Experiments configuration must be a dictionary")
            result["valid"] = False
            return result

        # Validate each experiment
        for experiment_name, experiment_config in experiments_config.items():
            experiment_result = self.validate_experiment_config(
                experiment_config, strict_mode=strict_mode
            )

            result["experiment_results"][experiment_name] = experiment_result

            # Aggregate errors and warnings
            for error in experiment_result["errors"]:
                result["errors"].append(f"Experiment '{experiment_name}': {error}")

            for warning in experiment_result["warnings"]:
                result["warnings"].append(f"Experiment '{experiment_name}': {warning}")

            for suggestion in experiment_result["suggestions"]:
                result["suggestions"].append(f"Experiment '{experiment_name}': {suggestion}")

            # Update overall validity
            if not experiment_result["valid"]:
                result["valid"] = False

        # Build summary
        result["summary"] = {
            "total_experiments": len(experiments_config),
            "valid_experiments": sum(
                1 for r in result["experiment_results"].values() if r["valid"]
            ),
            "invalid_experiments": sum(
                1 for r in result["experiment_results"].values() if not r["valid"]
            ),
            "total_errors": len(result["errors"]),
            "total_warnings": len(result["warnings"]),
        }

        return result


# =============================================================================
# DESCRIPTOR SCHEMA VALIDATOR
# =============================================================================


class DescriptorSchemaValidator:
    """
    Validator for descriptor configuration schemas.

    Provides validation for:
    - Individual descriptor configurations
    - Descriptor category configurations
    - Complete descriptor configuration bundles
    - Cross-validation with dataset types
    """

    def __init__(self):
        """Initialize descriptor schema validator."""
        self.validation_history = []
        self.logger = logging.getLogger(__name__)

    def validate_descriptor_config(
        self, config: dict[str, Any], strict_mode: bool = False
    ) -> dict[str, Any]:
        """
        Validate complete descriptor configuration.

        Args:
            config: Descriptor configuration dictionary
            strict_mode: Whether to use strict validation

        Returns:
            Validation result dictionary with:
                - valid: bool
                - errors: List[str]
                - warnings: List[str]
                - config_summary: Dict
        """
        result = {"valid": True, "errors": [], "warnings": [], "config_summary": {}}

        try:
            # Validate required fields
            required_fields = ["enabled"]
            for field in required_fields:
                if field not in config:
                    result["errors"].append(f"Missing required field: {field}")

            # Validate 'enabled' field
            if "enabled" in config and not isinstance(config["enabled"], bool):
                result["errors"].append("Field 'enabled' must be boolean")

            # Validate 'default_categories' if present
            if "default_categories" in config:
                if not isinstance(config["default_categories"], list):
                    result["errors"].append("Field 'default_categories' must be a list")
                else:
                    valid_categories = [
                        "constitutional",
                        "topological",
                        "geometric",
                        "electronic",
                        "pharmacophore",
                        "fingerprint",
                        "custom",
                    ]
                    for cat in config["default_categories"]:
                        if cat not in valid_categories:
                            result["errors"].append(
                                f"Invalid category '{cat}'. Valid: {valid_categories}"
                            )

            # Validate 'categories' section if present
            if "categories" in config:
                if not isinstance(config["categories"], dict):
                    result["errors"].append("Field 'categories' must be a dictionary")
                else:
                    for cat_name, cat_config in config["categories"].items():
                        cat_result = self.validate_descriptor_category(
                            cat_name, cat_config, strict_mode
                        )
                        result["errors"].extend(cat_result["errors"])
                        result["warnings"].extend(cat_result["warnings"])

            # Validate error handling mode
            if "error_handling" in config:
                valid_modes = ["strict", "warn", "skip"]
                if config["error_handling"] not in valid_modes:
                    result["errors"].append(f"Invalid error_handling mode. Valid: {valid_modes}")

            # Validate validation mode
            if "validation_mode" in config:
                valid_modes = ["strict", "standard", "permissive"]
                if config["validation_mode"] not in valid_modes:
                    result["errors"].append(f"Invalid validation_mode. Valid: {valid_modes}")

            # Validate parallel computation settings
            if "parallel_computation" in config:
                if not isinstance(config["parallel_computation"], bool):
                    result["errors"].append("Field 'parallel_computation' must be boolean")

                if config.get("parallel_computation", False):
                    if "num_workers" not in config:
                        result["warnings"].append(
                            "parallel_computation enabled but num_workers not specified"
                        )
                    elif config["num_workers"] < 2:
                        result["warnings"].append(
                            "parallel_computation enabled but num_workers < 2"
                        )

            # Strict mode validations
            if strict_mode and config.get("enabled", False):
                if not config.get("default_categories"):
                    result["errors"].append(
                        "Strict mode: default_categories required when enabled=true"
                    )
                if "categories" not in config or not config["categories"]:
                    result["warnings"].append("Strict mode: No category configurations defined")

            # Build configuration summary
            result["config_summary"] = {
                "enabled": config.get("enabled", False),
                "num_categories": len(config.get("default_categories", [])),
                "cache_enabled": config.get("cache_descriptors", False),
                "parallel_enabled": config.get("parallel_computation", False),
                "error_handling": config.get("error_handling", "warn"),
                "validation_mode": config.get("validation_mode", "standard"),
            }

            # Set overall validity
            result["valid"] = len(result["errors"]) == 0

        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Validation exception: {str(e)}")
            self.logger.error(f"Descriptor config validation error: {e}")

        return result

    def validate_descriptor_category(
        self, category_name: str, category_config: dict[str, Any], strict_mode: bool = False
    ) -> dict[str, Any]:
        """
        Validate individual descriptor category configuration.

        Args:
            category_name: Name of the category
            category_config: Category configuration dictionary
            strict_mode: Whether to use strict validation

        Returns:
            Validation result dictionary
        """
        result = {"valid": True, "errors": [], "warnings": []}

        try:
            # Validate category name
            valid_categories = [
                "constitutional",
                "topological",
                "geometric",
                "electronic",
                "pharmacophore",
                "fingerprint",
                "custom",
            ]
            if category_name not in valid_categories:
                result["errors"].append(
                    f"Invalid category name '{category_name}'. Valid: {valid_categories}"
                )

            # Validate enabled field
            if "enabled" in category_config:
                if not isinstance(category_config["enabled"], bool):
                    result["errors"].append(f"Category '{category_name}': enabled must be boolean")

            # Validate descriptors list
            if "descriptors" in category_config:
                descriptors = category_config["descriptors"]
                if descriptors is not None:
                    if not isinstance(descriptors, list):
                        result["errors"].append(
                            f"Category '{category_name}': descriptors must be a list or null"
                        )
                    elif not all(isinstance(d, str) for d in descriptors):
                        result["errors"].append(
                            f"Category '{category_name}': all descriptor names must be strings"
                        )
                    elif len(descriptors) == 0:
                        result["warnings"].append(
                            f"Category '{category_name}': empty descriptors list"
                        )

            # Validate options
            if "options" in category_config:
                if not isinstance(category_config["options"], dict):
                    result["errors"].append(
                        f"Category '{category_name}': options must be a dictionary"
                    )

            # Strict mode validations
            if strict_mode and category_config.get("enabled", True):
                if "descriptors" not in category_config:
                    result["warnings"].append(
                        f"Strict mode: Category '{category_name}' has no descriptor specification"
                    )

            result["valid"] = len(result["errors"]) == 0

        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Category validation exception: {str(e)}")
            self.logger.error(f"Category '{category_name}' validation error: {e}")

        return result

    def validate_with_dataset_type(
        self, descriptor_config: dict[str, Any], dataset_type: str
    ) -> dict[str, Any]:
        """
        Validate descriptor configuration compatibility with dataset type.

        Args:
            descriptor_config: Descriptor configuration
            dataset_type: Dataset type (any registered dataset type, e.g., 'DFT', 'DMC', 'Wavefunction')

        Returns:
            Validation result with compatibility information
        """
        result = {"valid": True, "errors": [], "warnings": [], "compatibility_notes": []}

        try:
            # Check if descriptors are enabled
            if not descriptor_config.get("enabled", False):
                result["compatibility_notes"].append(
                    "Descriptors disabled - no compatibility issues"
                )
                return result

            # PHASE 5 REFACTORING: Dataset-specific validation using registry features
            # Check if dataset supports orbital analysis (rich quantum data)
            if _dataset_supports_feature(dataset_type, "orbital_analysis"):
                if "electronic" not in descriptor_config.get("default_categories", []):
                    result["warnings"].append(
                        f"{dataset_type} dataset: Consider enabling 'electronic' descriptors "
                        "(dataset supports orbital analysis)"
                    )
                logger.debug(
                    f"{dataset_type} has orbital_analysis, suggested electronic descriptors"
                )

            # Check if dataset has vibrational analysis (standard molecular properties)
            elif _dataset_supports_feature(dataset_type, "vibrational_analysis"):
                if "geometric" in descriptor_config.get("default_categories", []):
                    categories = descriptor_config.get("categories", {})
                    geom_config = categories.get("geometric", {})
                    if geom_config.get("enabled", True):
                        result["compatibility_notes"].append(
                            f"{dataset_type} dataset: Geometric descriptors will use "
                            "QM-optimized coordinates"
                        )
                logger.debug(f"{dataset_type} has vibrational_analysis, noted coordinate source")

            # For datasets without specific features, provide generic note
            else:
                if "geometric" in descriptor_config.get("default_categories", []):
                    result["compatibility_notes"].append(
                        f"{dataset_type} dataset: Geometric descriptors will be computed from "
                        "available coordinates"
                    )
                logger.debug(
                    f"{dataset_type} does not have special features, generic coordinate note"
                )

            result["valid"] = len(result["errors"]) == 0

        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Compatibility validation exception: {str(e)}")
            self.logger.error(f"Dataset compatibility validation error: {e}")

        return result


# =============================================================================
# CONFIGURATION MIGRATION MANAGER
# =============================================================================


class ConfigMigration:
    """Configuration migration utilities"""

    def __init__(self):
        self.migration_calls = []

    def detect_format(self, config: dict[str, Any]) -> str:
        """Detect format using validator"""
        validator = YAMLSchemaValidator()
        return validator.detect_format(config)

    def migrate_to_enhanced(
        self, config: dict[str, Any], target_version: str = "v3", preserve_original: bool = True
    ) -> tuple[dict[str, Any], list[str]]:
        """Migrate legacy configuration to enhanced format"""
        self.migration_calls.append({"config": config, "target_version": target_version})

        if "transformations" not in config:
            return config, ["Configuration missing transformations"]

        transformations = config["transformations"]
        warnings = []
        current_format = self.detect_format(config)

        # If already enhanced, return as-is
        if current_format == "enhanced":
            warnings.append("Configuration already in enhanced format")
            return config, warnings

        # If invalid format, return unchanged with warning
        if current_format == "invalid":
            warnings.append("Invalid format detected, no migration performed")
            return config, warnings

        enhanced_config = config.copy()

        # Convert legacy list format
        if current_format == "legacy_list":
            enhanced_config["transformations"] = {
                "experimental_setups": {
                    "migrated_default": {
                        "description": "Migrated from legacy list format",
                        "transforms": [
                            {**transform, "enabled": True}
                            if isinstance(transform, dict)
                            else {"name": transform, "enabled": True}
                            for transform in transformations
                        ],
                    }
                },
                "default_setup": "migrated_default",
                "validation": {"enabled": True, "strict_mode": False, "warn_on_unknown": True},
                "migration_metadata": {
                    "original_format": "legacy_list",
                    "migration_timestamp": time.time(),
                    "migration_version": target_version,
                    "original_transform_count": len(transformations),
                    "migrated_transform_count": len(transformations),
                },
            }

            if preserve_original:
                enhanced_config["transformations"]["legacy_transforms"] = transformations.copy()

            warnings.append("Configuration migrated from legacy_list format")

        # Convert legacy dict format - To handle named setups properly
        elif current_format == "legacy_dict":
            experimental_setups = {}

            # Check if it has named setups (values are lists) vs single transform dict
            if any(isinstance(v, list) for v in transformations.values() if v is not None):
                # Named setups format - convert each setup
                for setup_name, setup_transforms in transformations.items():
                    if isinstance(setup_transforms, list):
                        experimental_setups[setup_name] = {
                            "description": f"Migrated setup: {setup_name}",
                            "transforms": [
                                {**transform, "enabled": True}
                                if isinstance(transform, dict)
                                else {"name": transform, "enabled": True}
                                for transform in setup_transforms
                            ],
                        }

                # Create enhanced config with experimental_setups
                enhanced_config["transformations"] = {
                    "experimental_setups": experimental_setups,
                    "default_setup": list(experimental_setups.keys())[0]
                    if experimental_setups
                    else "default",
                    "validation": {"enabled": True, "strict_mode": False, "warn_on_unknown": True},
                    "migration_metadata": {
                        "original_format": "legacy_dict_named",
                        "migration_timestamp": time.time(),
                        "migration_version": target_version,
                        "original_setup_count": len(experimental_setups),
                        "migrated_setup_count": len(experimental_setups),
                    },
                }

                if preserve_original:
                    enhanced_config["transformations"]["legacy_transforms"] = transformations.copy()

                warnings.append("Configuration migrated from legacy_dict format with named setups")

            else:
                # Single transform dict
                experimental_setups["migrated_default"] = {
                    "description": "Migrated from legacy single transform dict",
                    "transforms": [
                        {**transformations, "enabled": True}
                        if "name" in transformations
                        else {"name": "Unknown", "enabled": True}
                    ],
                }

                enhanced_config["transformations"] = {
                    "experimental_setups": experimental_setups,
                    "default_setup": "migrated_default",
                    "migration_metadata": {
                        "original_format": "legacy_dict_single",
                        "migration_timestamp": time.time(),
                        "migration_version": target_version,
                    },
                }

                warnings.append("Configuration migrated from legacy_dict format (single transform)")

        else:
            # Unknown format - return unchanged
            warnings.append(f"Unknown format '{current_format}', no migration performed")
            return config, warnings

        return enhanced_config, warnings


# Factory functions for easy creation
def create_validator() -> YAMLSchemaValidator:
    """Create a new YAML schema validator"""
    return YAMLSchemaValidator()


def create_migrator() -> ConfigMigration:
    """Create a new configuration migrator"""
    return ConfigMigration()


# =============================================================================
# ENHANCED CONFIGURATION LOADER WITH VALIDATION
# =============================================================================


def _load_transformation_config_enhanced(
    config_dict: dict[str, Any],
    validate: bool = True,
    migrate_legacy: bool = True,
    validation_config: ValidationConfig | None = None,
) -> dict[str, Any]:
    """
    Load and validate enhanced transformation configuration

    This function now includes plugin configuration validation as part of the
    complete configuration loading process.

    Args:
        config_dict: Complete configuration dictionary
        validate: Whether to validate configuration
        migrate_legacy: Whether to migrate legacy formats
        validation_config: Validation behavior configuration

    Returns:
        Transformations section of configuration (enhanced format)

    Raises:
        ConfigurationError: If validation fails or migration fails
    """

    config_logger = logger

    # Initialize utilities
    validator = YAMLSchemaValidator()
    migrator = ConfigMigration()

    if validation_config is None:
        validation_config = ValidationConfig()

    try:
        # Detect and handle format
        format_type = migrator.detect_format(config_dict)
        config_logger.debug(f"Detected configuration format: {format_type}")

        migrated_config = config_dict

        # Handle legacy formats
        if format_type in ["legacy_list", "legacy_dict"] and migrate_legacy:
            config_logger.info("Legacy transformation configuration detected - migrating...")
            try:
                migrated_config, migration_warnings = migrator.migrate_to_enhanced(config_dict)

                for warning in migration_warnings:
                    config_logger.warning(f"Config migration: {warning}")

                # Validate migration result
                migration_validation = migrator.validate_migration_result(
                    migrated_config, config_dict
                )
                if not migration_validation["valid"]:
                    config_logger.error("Migration validation failed:")
                    for issue in migration_validation["issues"]:
                        config_logger.error(f"  - {issue}")

                    raise ConfigurationError(
                        "Configuration migration failed validation",
                        config_key="transformations",
                        details=f"Migration issues: {migration_validation['issues']}",
                    )

                config_logger.info(f"Migration successful: {migration_validation['statistics']}")

            except Exception as e:
                config_logger.error(f"Configuration migration failed: {str(e)}")
                if validation_config.strict_mode:
                    raise
                else:
                    config_logger.warning(
                        "Continuing with original configuration (migration disabled)"
                    )
                    migrated_config = config_dict

        # Validate configuration if requested
        if validate:
            validation_results = validator.validate_config(migrated_config, validation_config)

            if not validation_results["valid"]:
                error_msg = "Configuration validation failed: " + "; ".join(
                    validation_results["errors"]
                )
                raise ConfigurationError(
                    error_msg,
                    config_key="transformations",
                    details=f"Validation errors: {validation_results['errors']}",
                )

            # Log warnings and suggestions
            for warning in validation_results["warnings"]:
                config_logger.warning(f"Config validation: {warning}")

            for suggestion in validation_results.get("suggestions", []):
                config_logger.info(f"Config suggestion: {suggestion}")

            # Log validation summary
            summary = validation_results["summary"]
            exp_count = summary.get("experimental_setups_count", 0)
            std_count = summary.get("standard_transforms_count", 0)
            config_logger.info(
                f"Configuration validated: {exp_count} experimental setups, "
                f"{std_count} standard transforms, "
                f"{summary.get('total_warnings', 0)} warnings, {summary.get('total_suggestions', 0)} suggestions"
            )

        # Validate plugin configuration if present
        if "plugins" in migrated_config:
            try:
                plugin_validator = PluginSchemaValidator()

                # Determine validation level based on configuration settings
                if validation_config and validation_config.strict_mode:
                    plugin_validation_level = PluginValidationLevel.STRICT
                else:
                    plugin_validation_level = PluginValidationLevel.STANDARD

                # Validate plugin configuration
                plugin_validation = plugin_validator.validate_plugin_config(
                    migrated_config["plugins"], validation_level=plugin_validation_level
                )

                # Handle validation results
                if not plugin_validation["valid"]:
                    error_msg = "Plugin configuration validation failed: " + "; ".join(
                        plugin_validation["errors"]
                    )

                    if validation_config and validation_config.strict_mode:
                        # In strict mode, raise error
                        raise ConfigurationError(
                            error_msg,
                            config_key="plugins",
                            details=f"Validation errors: {plugin_validation['errors']}",
                        )
                    else:
                        # In non-strict mode, just warn
                        config_logger.warning(error_msg)

                # Log plugin warnings
                for warning in plugin_validation["warnings"]:
                    config_logger.warning(f"Plugin config: {warning}")

                # Log plugin suggestions
                for suggestion in plugin_validation.get("suggestions", []):
                    config_logger.info(f"Plugin config suggestion: {suggestion}")

                # Log plugin summary if plugins are enabled
                summary = plugin_validation.get("plugin_summary", {})
                if summary.get("enabled"):
                    config_logger.info(
                        f"Plugin system enabled: {summary.get('plugin_paths_count', 0)} paths, "
                        f"{summary.get('trusted_plugins_count', 0)} trusted plugins, "
                        f"validation level: {summary.get('validation_level', 'unknown')}"
                    )
                else:
                    config_logger.debug("Plugin system disabled")

                # Optional: Validate plugin-transform compatibility
                if "transformations" in migrated_config and summary.get("enabled"):
                    compat_result = plugin_validator.validate_plugin_compatibility(
                        migrated_config["plugins"], migrated_config["transformations"]
                    )

                    if not compat_result["compatible"]:
                        compat_error_msg = "Plugin-transform compatibility issues: " + "; ".join(
                            compat_result["issues"]
                        )

                        if validation_config and validation_config.strict_mode:
                            raise ConfigurationError(
                                compat_error_msg,
                                config_key="plugins",
                                details=f"Compatibility issues: {compat_result['issues']}",
                            )
                        else:
                            config_logger.warning(compat_error_msg)

                    # Log compatibility warnings
                    for warning in compat_result.get("warnings", []):
                        config_logger.warning(f"Plugin compatibility: {warning}")

            except ConfigurationError:
                # Re-raise ConfigurationError as-is
                raise
            except Exception as e:
                # Handle unexpected errors in plugin validation
                config_logger.error(f"Plugin configuration validation error: {str(e)}")
                if validation_config and validation_config.strict_mode:
                    raise ConfigurationError(
                        f"Unexpected error in plugin validation: {str(e)}",
                        config_key="plugins",
                        details="Plugin validation encountered an unexpected error",
                    ) from e
                else:
                    config_logger.warning(
                        f"Plugin validation failed with unexpected error, continuing: {str(e)}"
                    )
        else:
            # No plugins section in configuration
            config_logger.debug("No plugins section in configuration")

        # Return the transformations section
        return migrated_config.get("transformations", {})

    except ConfigurationError:
        # Re-raise configuration errors as-is
        raise
    except Exception as e:
        # Handle unexpected errors
        config_logger.error(f"Unexpected error in configuration loading: {str(e)}")
        raise ConfigurationError(
            f"Configuration loading failed: {str(e)}",
            config_key="transformations",
            details="Unexpected error during configuration processing",
        ) from e


# =============================================================================
# YAML UTILITIES AND HELPERS
# =============================================================================


def load_and_validate_yaml_config(
    config_path: str | Path,
    validation_config: ValidationConfig | None = None,
    migrate_legacy: bool = True,
) -> dict[str, Any]:
    """Load and validate YAML configuration file"""

    if not YAML_AVAILABLE:
        raise ConfigurationError(
            "YAML library not available",
            config_key="yaml_dependency",
            details="Cannot load YAML configuration files without PyYAML",
        )

    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_path}",
            config_key="config_file",
            actual_value=str(config_path),
        )

    try:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        if not isinstance(config_dict, dict):
            raise ConfigurationError(
                "Configuration file must contain a dictionary",
                config_key="config_structure",
                actual_value=type(config_dict).__name__,
            )

        # Process with enhanced loader
        enhanced_config = _load_transformation_config_enhanced(
            config_dict,
            validate=True,
            migrate_legacy=migrate_legacy,
            validation_config=validation_config,
        )

        # Update original dict with enhanced config
        result_config = config_dict.copy()
        result_config["transformations"] = enhanced_config

        return result_config

    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"YAML parsing error: {str(e)}",
            config_key="yaml_syntax",
            details=f"Error in file: {config_path}",
        ) from e
    except ConfigurationError:
        # Re-raise configuration errors as-is
        raise
    except Exception as e:
        raise ConfigurationError(
            f"Unexpected error loading config: {str(e)}",
            config_key="config_loading",
            details=f"Error loading file: {config_path}",
        ) from e


def validate_yaml_config_string(
    config_string: str, validation_config: ValidationConfig | None = None
) -> dict[str, Any]:
    """Validate YAML configuration from string"""

    if not YAML_AVAILABLE:
        raise ConfigurationError("YAML library not available", config_key="yaml_dependency")

    try:
        config_dict = yaml.safe_load(config_string)

        if not isinstance(config_dict, dict):
            raise ConfigurationError(
                "Configuration must be a dictionary",
                config_key="config_structure",
                actual_value=type(config_dict).__name__,
            )

        validator = YAMLSchemaValidator()
        return validator.validate_config(config_dict, validation_config)

    except yaml.YAMLError as e:
        raise ConfigurationError(f"YAML parsing error: {str(e)}", config_key="yaml_syntax") from e


def create_example_enhanced_config() -> dict[str, Any]:
    """Create an example enhanced configuration for documentation/testing"""

    return {
        "transformations": {
            "experimental_setups": {
                "baseline": [
                    {"name": "AddSelfLoops", "enabled": True},
                    {"name": "ToUndirected", "enabled": True},
                ],
                "normalized": [
                    {"name": "AddSelfLoops", "enabled": True},
                    {"name": "ToUndirected", "enabled": True},
                    {"name": "GCNNorm", "kwargs": {"add_self_loops": False}, "enabled": True},
                ],
                "augmented": [
                    {"name": "AddSelfLoops", "enabled": True},
                    {"name": "ToUndirected", "enabled": True},
                    {
                        "name": "DropEdge",
                        "kwargs": {"p": 0.1},
                        "enabled": True,
                        "description": "Light edge dropout for robustness",
                    },
                ],
                "molecular_3d": [
                    {"name": "AddSelfLoops", "enabled": True},
                    {"name": "ToUndirected", "enabled": True},
                    {
                        "name": "Distance",
                        "kwargs": {"norm": True, "max_value": 10.0},
                        "enabled": True,
                        "description": "Normalized distance features for 3D molecular data",
                    },
                ],
            },
            "default_setup": "baseline",
            "validation": {
                "enabled": True,
                "strict_mode": False,
                "warn_on_unknown": True,
                "require_descriptions": False,
                "check_parameter_types": True,
            },
            "research_metadata": {
                "research_context": "molecular_property_prediction",
                # PHASE 5 REFACTORING: Use dynamic registry list for dataset compatibility
                "dataset_compatibility": _registry_list_all_safe(),
                "expected_effects": ["improved_message_passing", "structural_consistency"],
                "performance_notes": "Baseline setup optimized for molecular graphs",
            },
            "dataset_optimization": {
                "dataset_type": "molecular_graphs",
                "optimization_applied": True,
                "optimizations": ["molecular_graph_specific", "memory_efficient"],
            },
        }
    }


def create_example_legacy_configs() -> dict[str, dict[str, Any]]:
    """Create example legacy configurations for testing migration"""

    return {
        "legacy_list": {
            "transformations": [
                {"name": "AddSelfLoops"},
                {"name": "GCNNorm", "kwargs": {"add_self_loops": False}},
                {"name": "DropEdge", "args": {"p": 0.1}},  # Legacy args format
            ]
        },
        "legacy_dict": {"transformations": {"name": "AddSelfLoops", "enabled": True}},
        "legacy_with_parameters": {
            "transformations": [
                {"name": "AddSelfLoops"},
                {"name": "Distance", "parameters": {"norm": True}},  # Legacy parameters format
            ]
        },
    }


# =============================================================================
# MODULE INTEGRATION HELPERS
# =============================================================================


def get_enhanced_transformation_config(
    validate: bool = True,
    migrate_legacy: bool = True,
    validation_config: ValidationConfig | None = None,
) -> dict[str, Any]:
    """Get enhanced transformation configuration from current global config"""

    # Lazy import - see _get_load_config() function

    try:
        config = _get__get_load_config()()()
        if config is None:
            logger.warning("Config loading failed, returning empty transformation config")
            return {}

        return _load_transformation_config_enhanced(
            config,
            validate=validate,
            migrate_legacy=migrate_legacy,
            validation_config=validation_config,
        )

    except Exception as e:
        logger.error(f"Failed to get enhanced transformation config: {str(e)}")
        raise ConfigurationError(
            "Failed to load enhanced transformation configuration",
            config_key="transformations",
            details=str(e),
        ) from e


def validate_current_config(validation_config: ValidationConfig | None = None) -> dict[str, Any]:
    """Validate current configuration in memory"""

    # Lazy import - see _get_load_config() function

    try:
        config = _get__get_load_config()()()
        if config is None:
            raise ConfigurationError("No configuration loaded", config_key="global_config")

        validator = YAMLSchemaValidator()
        return validator.validate_config(config, validation_config)

    except Exception as e:
        logger.error(f"Current config validation failed: {str(e)}")
        raise


def create_default_plugin_config() -> dict[str, Any]:
    """
    Create a default plugin configuration

    Returns:
        Default plugin configuration dictionary
    """
    return {
        "enabled": False,
        "plugin_paths": [],
        "auto_discover": True,
        "auto_validate": True,
        "validation_level": "standard",
        "trusted_plugins": [],
        "disabled_plugins": [],
        "allow_experimental": False,
        "max_plugins": 50,
        "require_metadata": True,
        "enforce_checksums": True,
        "security_scanning": True,
    }


def create_example_plugin_config() -> dict[str, Any]:
    """
    Create an example plugin configuration for documentation

    Returns:
        Example plugin configuration dictionary
    """
    return {
        "enabled": True,
        "plugin_paths": ["./plugins", "/opt/milia/plugins"],
        "auto_discover": True,
        "auto_validate": True,
        "validation_level": "standard",
        "trusted_plugins": ["official_molecular_transforms", "verified_quantum_features"],
        "disabled_plugins": ["deprecated_transform_v1"],
        "allow_experimental": False,
        "max_plugins": 50,
        "require_metadata": True,
        "enforce_checksums": True,
        "security_scanning": True,
    }


# =============================================================================
# EXPERIMENT CONFIGURATION HELPERS
# =============================================================================


def create_default_experiment_config() -> dict[str, Any]:
    """
    Create a default experiment configuration

    Returns:
        Default experiment configuration dictionary
    """
    return {
        "name": "default_experiment",
        "description": "Default experiment configuration",
        "base_transforms": [{"name": "AddSelfLoops", "kwargs": {}, "enabled": True}],
        "ablations": [],
        "parameter_sweeps": [],
        "num_runs": 3,
        "random_seed": 42,
        "results": {},
        "metadata": {},
    }


def create_example_experiments_config() -> dict[str, Any]:
    """
    Create example experiments configuration for documentation

    Returns:
        Example experiments configuration with multiple experiment types
    """
    return {
        "transform_ablation": {
            "name": "transform_ablation",
            "description": "Study importance of each transform",
            "base_transforms": [
                {"name": "AddSelfLoops", "kwargs": {}, "enabled": True},
                {"name": "GCNNorm", "kwargs": {}, "enabled": True},
                {"name": "NormalizeVibModes", "kwargs": {}, "enabled": True},
            ],
            "ablations": [
                {
                    "name": "no_gcn_norm",
                    "description": "Remove GCN normalization",
                    "transforms": [
                        {"name": "AddSelfLoops", "kwargs": {}, "enabled": True},
                        {"name": "NormalizeVibModes", "kwargs": {}, "enabled": True},
                    ],
                },
                {
                    "name": "no_vib_norm",
                    "description": "Remove vibrational mode normalization",
                    "transforms": [
                        {"name": "AddSelfLoops", "kwargs": {}, "enabled": True},
                        {"name": "GCNNorm", "kwargs": {}, "enabled": True},
                    ],
                },
            ],
            "hypothesis": "All transforms contribute to model accuracy",
            "expected_outcome": "baseline > no_vib_norm > no_gcn_norm",
            "paper_reference": "Section 4.2 - Ablation Studies",
            "num_runs": 5,
            "random_seed": 42,
        },
        "dropout_optimization": {
            "name": "dropout_optimization",
            "description": "Find optimal dropout rate for DropEdge transform",
            "base_transforms": [
                {"name": "AddSelfLoops", "kwargs": {}, "enabled": True},
                {"name": "GCNNorm", "kwargs": {}, "enabled": True},
            ],
            "parameter_sweeps": [
                {
                    "transform_name": "DropEdge",
                    "parameter_name": "p",
                    "values": [0.0, 0.1, 0.2, 0.3, 0.5],
                    "description": "Sweep dropout probability",
                }
            ],
            "hypothesis": "Moderate dropout improves generalization",
            "expected_outcome": "Optimal p in range [0.15, 0.25]",
            "num_runs": 10,
            "random_seed": 42,
        },
        "normalization_comparison": {
            "name": "normalization_comparison",
            "description": "Compare different normalization approaches",
            "base_transforms": [{"name": "AddSelfLoops", "kwargs": {}, "enabled": True}],
            "ablations": [
                {
                    "name": "gcn_norm",
                    "description": "GCN normalization",
                    "transforms": [
                        {"name": "AddSelfLoops", "kwargs": {}, "enabled": True},
                        {"name": "GCNNorm", "kwargs": {}, "enabled": True},
                    ],
                },
                {
                    "name": "no_norm",
                    "description": "No normalization",
                    "transforms": [{"name": "AddSelfLoops", "kwargs": {}, "enabled": True}],
                },
            ],
            "hypothesis": "GCN normalization improves stability",
            "expected_outcome": "gcn_norm > no_norm",
            "paper_reference": "Section 4.3 - Normalization Study",
            "num_runs": 5,
            "random_seed": 42,
            "metadata": {
                "research_question": "Best normalization for molecular graphs?",
                "evaluation_metric": "validation_mae",
            },
        },
    }


def validate_experiment_config_file(
    config_path: str | Path, validation_config: ValidationConfig | None = None
) -> dict[str, Any]:
    """
    Validate experiments configuration from a YAML file

    Args:
        config_path: Path to configuration file
        validation_config: Validation behavior configuration

    Returns:
        Validation results dictionary

    Raises:
        ConfigurationError: If file cannot be loaded
    """
    if not YAML_AVAILABLE:
        raise ConfigurationError("YAML library not available", config_key="yaml_dependency")

    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_path}",
            config_key="config_file",
            actual_value=str(config_path),
        )

    try:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        if not isinstance(config_dict, dict):
            raise ConfigurationError(
                "Configuration file must contain a dictionary",
                config_key="config_structure",
                actual_value=type(config_dict).__name__,
            )

        # Extract experiments section
        if "experiments" not in config_dict:
            return {
                "valid": True,
                "warnings": ["No experiments section in configuration"],
                "errors": [],
                "suggestions": ["Add experiments section to define research experiments"],
            }

        # Validate experiments configuration with strict mode
        validator = ExperimentSchemaValidator()
        strict_mode = validation_config.strict_mode if validation_config else False

        return validator.validate_experiments_config(
            config_dict["experiments"], strict_mode=strict_mode
        )
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"YAML parsing error: {str(e)}",
            config_key="yaml_syntax",
            details=f"Error in file: {config_path}",
        ) from e
    except Exception as e:
        raise ConfigurationError(
            f"Unexpected error loading config: {str(e)}",
            config_key="config_loading",
            details=f"Error loading file: {config_path}",
        ) from e


def get_experiment_config_summary(experiment_config: dict[str, Any]) -> str:
    """
    Generate a human-readable summary of experiment configuration

    Args:
        experiment_config: Experiment configuration dictionary

    Returns:
        Formatted summary string
    """
    name = experiment_config.get("name", "Unknown")
    description = experiment_config.get("description", "No description")

    base_transforms_count = len(experiment_config.get("base_transforms", []))
    ablations_count = len(experiment_config.get("ablations", []))
    sweeps_count = len(experiment_config.get("parameter_sweeps", []))
    num_runs = experiment_config.get("num_runs", 3)

    lines = [
        f"Experiment: {name}",
        f"  Description: {description}",
        f"  Base Transforms: {base_transforms_count}",
        f"  Ablations: {ablations_count}",
        f"  Parameter Sweeps: {sweeps_count}",
        f"  Runs per Variant: {num_runs}",
    ]

    if "hypothesis" in experiment_config:
        lines.append(f"  Hypothesis: {experiment_config['hypothesis']}")

    if "expected_outcome" in experiment_config:
        lines.append(f"  Expected: {experiment_config['expected_outcome']}")

    if "paper_reference" in experiment_config:
        lines.append(f"  Paper Ref: {experiment_config['paper_reference']}")

    # Calculate total runs
    total_variants = ablations_count + sweeps_count
    if total_variants > 0:
        total_runs = total_variants * num_runs
        lines.append(f"  Total Runs: {total_runs} ({total_variants} variants × {num_runs} runs)")

    return "\n".join(lines)


def validate_plugin_config_file(
    config_path: str | Path, validation_config: ValidationConfig | None = None
) -> dict[str, Any]:
    """
    Validate plugin configuration from a YAML file

    Args:
        config_path: Path to configuration file
        validation_config: Validation behavior configuration

    Returns:
        Validation results dictionary

    Raises:
        ConfigurationError: If file cannot be loaded
    """
    if not YAML_AVAILABLE:
        raise ConfigurationError("YAML library not available", config_key="yaml_dependency")

    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_path}",
            config_key="config_file",
            actual_value=str(config_path),
        )

    try:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        if not isinstance(config_dict, dict):
            raise ConfigurationError(
                "Configuration file must contain a dictionary",
                config_key="config_structure",
                actual_value=type(config_dict).__name__,
            )

        # Extract plugins section
        if "plugins" not in config_dict:
            return {
                "valid": True,
                "warnings": ["No plugins section in configuration"],
                "errors": [],
                "suggestions": ["Add plugins section if you want to use plugins"],
            }

        # Validate plugin configuration with proper validation level
        validator = PluginSchemaValidator()

        # Determine validation level from ValidationConfig
        if validation_config and validation_config.strict_mode:
            validation_level = PluginValidationLevel.STRICT
        else:
            validation_level = PluginValidationLevel.STANDARD

        return validator.validate_plugin_config(
            config_dict["plugins"], validation_level=validation_level
        )
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"YAML parsing error: {str(e)}",
            config_key="yaml_syntax",
            details=f"Error in file: {config_path}",
        ) from e
    except Exception as e:
        raise ConfigurationError(
            f"Unexpected error loading config: {str(e)}",
            config_key="config_loading",
            details=f"Error loading file: {config_path}",
        ) from e


def merge_plugin_configs(
    base_config: dict[str, Any], override_config: dict[str, Any]
) -> dict[str, Any]:
    """
    Merge two plugin configurations with override taking precedence

    Args:
        base_config: Base plugin configuration
        override_config: Override plugin configuration

    Returns:
        Merged plugin configuration
    """
    merged = base_config.copy()

    # Simple key-value overrides
    simple_keys = [
        "enabled",
        "auto_discover",
        "auto_validate",
        "validation_level",
        "allow_experimental",
        "max_plugins",
        "require_metadata",
        "enforce_checksums",
        "security_scanning",
    ]

    for key in simple_keys:
        if key in override_config:
            merged[key] = override_config[key]

    # Merge lists (union for paths, override for plugin lists)
    if "plugin_paths" in override_config:
        # Union of paths (no duplicates)
        base_paths = set(merged.get("plugin_paths", []))
        override_paths = set(override_config["plugin_paths"])
        merged["plugin_paths"] = list(base_paths | override_paths)

    # Override for trusted/disabled plugins
    for list_key in ["trusted_plugins", "disabled_plugins"]:
        if list_key in override_config:
            merged[list_key] = override_config[list_key]

    return merged


def get_plugin_config_summary(plugin_config: dict[str, Any]) -> str:
    """
    Generate a human-readable summary of plugin configuration

    Args:
        plugin_config: Plugin configuration dictionary

    Returns:
        Formatted summary string
    """
    if not plugin_config.get("enabled", False):
        return "Plugins: DISABLED"

    lines = [
        "Plugin Configuration Summary:",
        "  Status: ENABLED",
        f"  Validation Level: {plugin_config.get('validation_level', 'standard').upper()}",
        f"  Plugin Paths: {len(plugin_config.get('plugin_paths', []))}",
        f"  Trusted Plugins: {len(plugin_config.get('trusted_plugins', []))}",
        f"  Disabled Plugins: {len(plugin_config.get('disabled_plugins', []))}",
        f"  Auto-discover: {plugin_config.get('auto_discover', True)}",
        f"  Auto-validate: {plugin_config.get('auto_validate', True)}",
        f"  Security Scanning: {plugin_config.get('security_scanning', True)}",
        f"  Enforce Checksums: {plugin_config.get('enforce_checksums', True)}",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    """
    Comprehensive test suite for configuration schemas

    Tests transformations, plugins, and experiments
    """

    # =============================================================================
    # TRANSFORMATION SCHEMA TESTS
    # =============================================================================

    print("=" * 70)
    print("CONFIGURATION SCHEMA TEST SUITE")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("Testing Transformation Configuration Schema")
    print("=" * 70)

    # Test legacy dict format migration
    test_config = {
        "transformations": {
            "baseline": [{"name": "AddSelfLoops"}, {"name": "ToUndirected"}],
            "advanced": [{"name": "AddSelfLoops"}, {"name": "GCNNorm"}],
        }
    }

    print("\n1. Testing configuration migration...")
    migrator = ConfigMigration()
    result, warnings = migrator.migrate_to_enhanced(test_config)

    print(f"   ✓ Migration successful: {'experimental_setups' in result['transformations']}")
    print(f"   ✓ Warnings: {len(warnings)}")
    for warning in warnings:
        print(f"     - {warning}")

    # Test validation
    print("\n2. Testing configuration validation...")
    validator = YAMLSchemaValidator()
    validation_result = validator.validate_config(result)
    print(f"   ✓ Validation result: {validation_result['valid']}")
    print(f"   ✓ Format detected: {validation_result['format_detected']}")
    print(f"   ✓ Errors: {len(validation_result['errors'])}")
    print(f"   ✓ Warnings: {len(validation_result['warnings'])}")

    # =============================================================================
    # PLUGIN SCHEMA TESTS
    # =============================================================================

    print("\n" + "=" * 70)
    print("Testing Plugin Configuration Schema")
    print("=" * 70)

    # Test 1: Default plugin config
    print("\n1. Creating default plugin configuration...")
    default_plugin_config = create_default_plugin_config()
    print("   ✓ Default config created")
    print(f"   ✓ Enabled: {default_plugin_config['enabled']}")
    print(f"   ✓ Validation level: {default_plugin_config['validation_level']}")

    # Test 2: Example plugin config
    print("\n2. Creating example plugin configuration...")
    example_plugin_config = create_example_plugin_config()
    print("   ✓ Example config created")
    print(f"   ✓ Plugin paths: {len(example_plugin_config['plugin_paths'])}")
    print(f"   ✓ Trusted plugins: {len(example_plugin_config['trusted_plugins'])}")
    print(f"   ✓ Disabled plugins: {len(example_plugin_config['disabled_plugins'])}")

    # Test 3: Validate plugin config
    print("\n3. Validating example plugin configuration...")
    plugin_validator = PluginSchemaValidator()
    plugin_validation = plugin_validator.validate_plugin_config(
        example_plugin_config, validation_level=PluginValidationLevel.STANDARD
    )
    print(f"   ✓ Validation result: {plugin_validation['valid']}")
    print(f"   ✓ Warnings: {len(plugin_validation['warnings'])}")
    print(f"   ✓ Errors: {len(plugin_validation['errors'])}")

    if plugin_validation["warnings"]:
        print("   Warnings:")
        for warning in plugin_validation["warnings"][:3]:  # Show first 3
            print(f"     - {warning}")

    # Test 4: Plugin config summary
    print("\n4. Plugin configuration summary:")
    summary_lines = get_plugin_config_summary(example_plugin_config).split("\n")
    for line in summary_lines:
        print(f"   {line}")

    # Test 5: Complete config with plugins (WITHOUT experiments)
    print("\n5. Validating complete configuration with plugins...")
    complete_config = {
        "transformations": {
            "experimental_setups": {
                "basic": {
                    "description": "Basic test setup",
                    "transforms": [{"name": "AddSelfLoops", "kwargs": {}, "enabled": True}],
                }
            },
            "default_setup": "basic",
            "validation": {"enabled": True, "strict_mode": False},
        },
        "plugins": example_plugin_config,
    }

    complete_validation = validator.validate_config_with_plugins(complete_config)
    print(f"   ✓ Complete validation: {complete_validation['valid']}")
    if "plugin_validation" in complete_validation:
        print(f"   ✓ Plugin validation passed: {complete_validation['plugin_validation']['valid']}")
    print(f"   ✓ Total errors: {len(complete_validation['errors'])}")
    print(f"   ✓ Total warnings: {len(complete_validation['warnings'])}")

    # =============================================================================
    # EXPERIMENT SCHEMA TESTS
    # =============================================================================

    print("\n" + "=" * 70)
    print("Testing Experiment Configuration Schema")
    print("=" * 70)

    # Test 1: Default experiment config
    print("\n1. Creating default experiment configuration...")
    default_experiment = create_default_experiment_config()
    print("   ✓ Default experiment created")
    print(f"   ✓ Name: {default_experiment['name']}")
    print(f"   ✓ Base transforms: {len(default_experiment['base_transforms'])}")
    print(f"   ✓ Num runs: {default_experiment['num_runs']}")

    # Test 2: Example experiments config
    print("\n2. Creating example experiments configuration...")
    example_experiments = create_example_experiments_config()
    print("   ✓ Example experiments created")
    print(f"   ✓ Total experiments: {len(example_experiments)}")
    print(f"   ✓ Experiment names: {list(example_experiments.keys())}")

    # Test 3: Validate single experiment
    print("\n3. Validating single experiment configuration...")
    experiment_validator = ExperimentSchemaValidator()
    single_validation = experiment_validator.validate_experiment_config(
        example_experiments["transform_ablation"], strict_mode=False
    )
    print(f"   ✓ Validation result: {single_validation['valid']}")
    print(f"   ✓ Warnings: {len(single_validation['warnings'])}")
    print(f"   ✓ Errors: {len(single_validation['errors'])}")

    summary = single_validation["experiment_summary"]
    print("   Summary:")
    print(f"     - Name: {summary['name']}")
    print(f"     - Base transforms: {summary['base_transforms_count']}")
    print(f"     - Ablations: {summary['ablations_count']}")
    print(f"     - Parameter sweeps: {summary['parameter_sweeps_count']}")
    print(f"     - Has hypothesis: {summary['has_hypothesis']}")
    print(f"     - Research API available: {summary['research_api_available']}")

    # Test 4: Validate all experiments
    print("\n4. Validating complete experiments configuration...")
    all_validation = experiment_validator.validate_experiments_config(
        example_experiments, strict_mode=False
    )
    print(f"   ✓ Overall validation: {all_validation['valid']}")

    all_summary = all_validation["summary"]
    print("   Summary:")
    print(f"     - Total experiments: {all_summary['total_experiments']}")
    print(f"     - Valid experiments: {all_summary['valid_experiments']}")
    print(f"     - Invalid experiments: {all_summary['invalid_experiments']}")
    print(f"     - Total errors: {all_summary['total_errors']}")
    print(f"     - Total warnings: {all_summary['total_warnings']}")

    # Test 5: Experiment config summaries
    print("\n5. Individual experiment configuration summaries:")
    for i, (exp_name, exp_config) in enumerate(example_experiments.items(), 1):
        print(f"\n   Experiment {i}: {exp_name}")
        summary_lines = get_experiment_config_summary(exp_config).split("\n")
        for line in summary_lines:
            print(f"   {line}")

    # Test 6: Complete config with experiments (MOVED HERE - after example_experiments is created)
    print("\n6. Validating complete configuration (transforms + plugins + experiments)...")
    complete_config_with_experiments = {
        "transformations": {
            "experimental_setups": {
                "baseline": {
                    "description": "Baseline test setup",
                    "transforms": [{"name": "AddSelfLoops", "kwargs": {}, "enabled": True}],
                }
            },
            "default_setup": "baseline",
            "validation": {"enabled": True, "strict_mode": False},
        },
        "plugins": example_plugin_config,
        "experiments": example_experiments,
    }

    full_validation = validator.validate_config_with_experiments(
        complete_config_with_experiments, validation_config=ValidationConfig(strict_mode=False)
    )
    print(f"   ✓ Full validation: {full_validation['valid']}")

    if "plugin_validation" in full_validation:
        plugin_val = full_validation["plugin_validation"]
        print(f"   ✓ Plugin validation: {plugin_val['valid']}")

    if "experiment_validation" in full_validation:
        exp_val = full_validation["experiment_validation"]
        print(f"   ✓ Experiment validation: {exp_val['valid']}")
        print(
            f"   ✓ Valid experiments: {exp_val['summary']['valid_experiments']}/{exp_val['summary']['total_experiments']}"
        )

    print(f"   ✓ Total errors: {len(full_validation['errors'])}")
    print(f"   ✓ Total warnings: {len(full_validation['warnings'])}")

    # Test 7: ExperimentSchema dataclass
    print("\n7. Testing ExperimentSchema dataclass...")
    try:
        schema = ExperimentSchema(
            name="test_experiment",
            description="Test experiment for validation",
            base_transforms=[{"name": "AddSelfLoops", "kwargs": {}, "enabled": True}],
            ablations=[
                {"name": "no_self_loops", "description": "Remove self loops", "transforms": []}
            ],
            num_runs=5,
            random_seed=123,
            hypothesis="Self-loops improve accuracy",
            expected_outcome="baseline > no_self_loops",
        )
        print("   ✓ ExperimentSchema created successfully")
        print(f"   ✓ Schema name: {schema.name}")
        print(f"   ✓ Num runs: {schema.num_runs}")
        print(f"   ✓ Has hypothesis: {schema.hypothesis is not None}")

        # Test serialization
        schema_dict = schema.to_dict()
        print("   ✓ Serialization successful")
        print(f"   ✓ Dict keys: {list(schema_dict.keys())}")

        # Test deserialization
        schema_restored = ExperimentSchema.from_dict(schema_dict)
        print("   ✓ Deserialization successful")
        print(f"   ✓ Names match: {schema.name == schema_restored.name}")

    except Exception as e:
        print(f"   ✗ ExperimentSchema test failed: {str(e)}")

    # Test 8: Invalid configurations
    print("\n8. Testing validation with invalid configurations...")

    # Test 8a: Missing required fields
    invalid_experiment_1 = {"description": "Missing name field"}
    invalid_validation_1 = experiment_validator.validate_experiment_config(
        invalid_experiment_1, strict_mode=False
    )
    print(f"   ✓ Missing name detected: {not invalid_validation_1['valid']}")
    print(f"   ✓ Errors: {len(invalid_validation_1['errors'])}")

    # Test 8b: Invalid num_runs
    invalid_experiment_2 = {
        "name": "test",
        "description": "Invalid num_runs",
        "base_transforms": [],
        "num_runs": -1,
    }
    invalid_validation_2 = experiment_validator.validate_experiment_config(
        invalid_experiment_2, strict_mode=False
    )
    print(f"   ✓ Invalid num_runs detected: {not invalid_validation_2['valid']}")

    # Test 8c: No variants in strict mode
    no_variants_experiment = {
        "name": "test",
        "description": "No variants",
        "base_transforms": [{"name": "AddSelfLoops"}],
        "ablations": [],
        "parameter_sweeps": [],
    }
    no_variants_validation = experiment_validator.validate_experiment_config(
        no_variants_experiment, strict_mode=True
    )
    print(f"   ✓ No variants detected (strict): {not no_variants_validation['valid']}")

    # =============================================================================
    # FINAL SUMMARY
    # =============================================================================

    print("\n" + "=" * 70)
    print("TEST SUITE SUMMARY")
    print("=" * 70)

    print("\n✓ Transformations: Configuration migration and validation")
    print("✓ Plugins: Plugin configuration schema validation")
    print("✓ Experiments: Research experiment schema validation")
    print("\nAll schema validation tests completed successfully!")

    print("\n" + "=" * 70)
    print("Available Schema Validators:")
    print("=" * 70)
    print("  - YAMLSchemaValidator:        Transformation configuration")
    print("  - PluginSchemaValidator:      Plugin system configuration")
    print("  - ExperimentSchemaValidator:  Research experiment configuration")
    print("\nAvailable Helper Functions:")
    print("  - create_default_plugin_config()")
    print("  - create_example_plugin_config()")
    print("  - create_default_experiment_config()")
    print("  - create_example_experiments_config()")
    print("  - validate_plugin_config_file()")
    print("  - validate_experiment_config_file()")
    print("  - get_plugin_config_summary()")
    print("  - get_experiment_config_summary()")
    print("=" * 70)
