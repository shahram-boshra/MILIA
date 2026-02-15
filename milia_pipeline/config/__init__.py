# milia_pipeline/config/__init__.py

"""
milia Pipeline Configuration System
====================================

This module provides a comprehensive configuration management system for the milia
molecular data processing pipeline. It supports YAML-based configuration with schema
validation, type checking, handler-based architecture, and transformation management.

Key Features:
-------------
- **Thread-safe configuration loading** with intelligent caching
- **Schema validation** using Pydantic with multiple validation levels
- **Handler-based architecture** for dataset-specific processing (DFT, DMC, Wavefunction)
- **Transformation system** with experimental setups and plugin support
- **Data refinement** for molecular properties and vibrational data
- **Comprehensive validation** with detailed error reporting
- **Migration support** for legacy configuration formats

Phase 3 Registry Integration:
-----------------------------
- **Dynamic registry lookups** for handler types and properties
- **Cache invalidation** when registry changes
- **Backward compatibility** with legacy hardcoded constants
- **Deprecation warnings** for direct constant access

Phase 4 Container Registry Integration:
---------------------------------------
- **Dynamic type validation** in all configuration containers
- **Registry-based property lookups** in DatasetConfig
- **Feature-based handler customization** in factory functions
- **Verification utilities** for registry integration status

Phase 5 Accessor Registry Integration:
---------------------------------------
- **Dynamic dataset type validation** in all accessor functions
- **Registry-based type enumeration** replacing hardcoded lists
- **Cache invalidation callbacks** for accessor consistency
- **Validation helper functions** for dataset type checking

Phase 5 Config Loader Registry Integration:
-------------------------------------------
- **Dynamic default dataset type** via _get_default_dataset_type()
- **Lazy registry initialization** avoiding circular imports
- **Backward compatible defaults** (falls back to 'DFT' if registry unavailable)
- **create_example_config()** now uses registry for default dataset type
- **migrate_legacy_config()** now uses registry for default dataset type

Phase 6 Data Refining Registry Integration:
-------------------------------------------
- **Dynamic feature queries** via _get_dataset_feature()
- **Feature-based refinement category routing** via _get_dataset_refinement_category()
- **Registry-based dataset type validation** in refinement functions
- **Zero-core-file-modification** for new dataset types with appropriate features
- **Backward compatible fallback** when registry unavailable
- **Refactored functions**: refine_molecular_vibrations, detect_dmc_statistical_outliers,
  calculate_dmc_uncertainty_weights, log_data_refinement_status

Phase 6 Validators Registry Integration:
----------------------------------------
- **Dynamic dataset type validation** via _validators_is_dataset_type_registered()
- **Feature-based validation queries** via _validators_get_dataset_feature()
- **Dynamic required properties lookup** via _validators_get_dataset_required_properties()
- **Handler compatibility checks** via _validators_get_handler_compatibility_checks()
- **Registry status reporting** via get_validators_registry_status()
- **Backward compatible fallback** when registry unavailable
- **Refactored functions**: validate_molecular_data_dict, validate_handler_compatibility,
  validate_descriptor_category_compatibility, run_handler_validation_tests

Standard Transforms Configuration Support:
------------------------------------------
- **validators.py**: validate_transformation_config() now accepts configs with only
  standard_transforms (no experimental_setups required)
- **config_schemas.py**: TransformationSchema supports standard_transforms field;
  YAMLSchemaValidator.validate_config() validates standard_transforms
- **config_loader.py**: detect_transforms_format() recognizes standard_transforms as
  enhanced format; _validate_enhanced_format() validates standard_transforms;
  check_migration_status() reports standard_transforms_count

Architecture:
-------------
The configuration system is organized into several specialized modules:

1. **config_loader**: Core configuration loading with caching and validation
2. **config_containers**: Type-safe dataclass containers for configuration data
3. **config_accessors**: Accessor functions for querying configuration values
4. **config_schemas**: Pydantic schemas for validation and migration
5. **config_constants**: Constants and handler compatibility utilities
6. **validators**: Validation functions for molecular data and transforms
7. **data_refining**: Data refinement for molecular properties

Main Components:
----------------

Configuration Loading:
    - load_config: Main function to load and cache YAML configuration
    - Configuration automatically validated against schemas
    - Thread-safe with RLock protection for concurrent access

Configuration Containers (Dataclasses):
    - DatasetConfig: Dataset-specific configuration (DFT/DMC/Wavefunction)
    - FilterConfig: Molecular filtering rules
    - StructuralFeaturesConfig: Structural feature extraction settings
    - ProcessingConfig: General processing parameters
    - HandlerConfig: Handler-specific configuration
    - TransformSpec: Individual transformation specification
    - ExperimentalSetup: Experimental configuration for research
    - TransformationConfig: Complete transformation system configuration
    - DescriptorConfig: Molecular descriptor calculation settings
    - DescriptorCategoryConfig: Descriptor categorization settings

Accessor Functions:
    - get_dataset_type, get_dataset_config, get_data_config
    - get_uncertainty_config, is_uncertainty_enabled
    - get_structural_features_config, get_filter_config
    - get_transformation_config, get_experimental_setup
    - get_descriptor_config, is_descriptors_enabled
    - Plus 60+ specialized accessor functions

Schema & Validation:
    - YAMLSchemaValidator: Complete YAML configuration validation
    - ConfigMigration: Legacy format migration utilities
    - ValidationResult: Structured validation results
    - TransformValidator: Transform-specific validation

Data Refinement:
    - refine_molecular_data: Refine molecular properties
    - refine_molecular_vibrations: Process vibrational data
    - detect_dmc_statistical_outliers: DMC-specific outlier detection
    - validate_refined_data_quality: Post-refinement validation

Usage Examples:
---------------

Basic configuration loading:
    >>> from milia_pipeline.config import load_config
    >>> config = load_config('config.yaml')

Using configuration containers:
    >>> from milia_pipeline.config import (
    ...     create_dataset_config_from_global,
    ...     create_transformation_config_from_global
    ... )
    >>> dataset_config = create_dataset_config_from_global()
    >>> transform_config = create_transformation_config_from_global()

Accessing configuration values:
    >>> from milia_pipeline.config import (
    ...     get_dataset_type,
    ...     get_uncertainty_config,
    ...     is_uncertainty_enabled
    ... )
    >>> dataset_type = get_dataset_type()  # Returns 'DFT', 'DMC', or 'Wavefunction'
    >>> if is_uncertainty_enabled():
    ...     uncertainty_cfg = get_uncertainty_config()

Working with transformations:
    >>> from milia_pipeline.config import (
    ...     get_transformation_config,
    ...     get_experimental_setup,
    ...     list_experimental_setups
    ... )
    >>> transform_cfg = get_transformation_config()
    >>> available_setups = list_experimental_setups()
    >>> setup = get_experimental_setup('baseline')

Working with standard transforms (NEW):
    >>> from milia_pipeline.config import (
    ...     get_standard_transforms,
    ...     get_combined_transforms,
    ...     has_standard_transforms
    ... )
    >>> if has_standard_transforms():
    ...     standard = get_standard_transforms()  # TransformSpec objects
    ...     combined = get_combined_transforms()  # standard + experimental

Validation:
    >>> from milia_pipeline.config import (
    ...     validate_molecular_data_dict,
    ...     validate_transformation_config,
    ...     create_validation_report
    ... )
    >>> is_valid = validate_molecular_data_dict(data_dict)
    >>> report = create_validation_report(validation_results)

Data refinement:
    >>> from milia_pipeline.config import (
    ...     refine_molecular_data,
    ...     validate_refined_data_quality
    ... )
    >>> refined_data = refine_molecular_data(raw_data)
    >>> is_valid = validate_refined_data_quality(refined_data)

Phase 3 Registry Integration (NEW):
    >>> from milia_pipeline.config import (
    ...     get_supported_handler_types,
    ...     get_handler_feature_support,
    ...     get_handler_molecule_creation_strategy,
    ...     is_handler_type_supported
    ... )
    >>> handler_types = get_supported_handler_types()  # Dynamic from registry
    >>> features = get_handler_feature_support('DFT')  # From dataset class
    >>> strategy = get_handler_molecule_creation_strategy('Wavefunction')  # 'coordinate_based'
    >>> is_supported = is_handler_type_supported('DFT')  # True

Phase 4 Container Registry Integration (NEW):
    >>> from milia_pipeline.config import (
    ...     verify_container_registry_integration,
    ...     _get_valid_dataset_types,
    ...     _is_valid_dataset_type
    ... )
    >>> results = verify_container_registry_integration()
    >>> print(results['overall_status'])  # 'ok' if all containers work
    >>> valid_types = _get_valid_dataset_types()  # ['DFT', 'DMC', 'Wavefunction']
    >>> is_valid = _is_valid_dataset_type('DFT')  # True

Phase 5 Accessor Registry Integration (NEW):
    >>> from milia_pipeline.config import (
    ...     validate_dataset_type,
    ...     is_valid_dataset_type,
    ...     registry_list_all,
    ...     registry_is_registered
    ... )
    >>> validate_dataset_type('DFT')  # Validates against registry, raises if invalid
    >>> is_valid = is_valid_dataset_type('DFT')  # Non-throwing validation
    >>> all_types = registry_list_all()  # ['DFT', 'DMC', 'Wavefunction']
    >>> registered = registry_is_registered('DFT')  # True

Phase 6 Data Refining Registry Integration (NEW):
    >>> from milia_pipeline.config import (
    ...     get_refining_registry_status,
    ...     _get_dataset_feature,
    ...     _get_dataset_refinement_category
    ... )
    >>> status = get_refining_registry_status()  # Registry status for data_refining
    >>> has_vibrations = _get_dataset_feature('DFT', 'vibrational_analysis')  # True
    >>> has_uncertainty = _get_dataset_feature('DMC', 'uncertainty_handling')  # True
    >>> category = _get_dataset_refinement_category('DFT')  # 'vibrational'

Phase 6 Validators Registry Integration (NEW):
    >>> from milia_pipeline.config import (
    ...     get_validators_registry_status,
    ...     _validators_get_dataset_feature,
    ...     _validators_is_dataset_type_registered,
    ...     _validators_get_handler_compatibility_checks
    ... )
    >>> status = get_validators_registry_status()  # Registry status for validators
    >>> has_uncertainty = _validators_get_dataset_feature('DMC', 'uncertainty_handling')  # True
    >>> is_valid = _validators_is_dataset_type_registered('DFT')  # True
    >>> checks = _validators_get_handler_compatibility_checks('DFT')  # {'supports_vibrational': True, ...}

Thread Safety:
--------------
The configuration system is thread-safe with proper locking mechanisms:
- Configuration cache protected by RLock
- Statistics updates protected by Lock
- Safe for concurrent access from multiple threads

For detailed documentation, see individual module and function docstrings.
"""

# =============================================================================
# Core Configuration Loading
# =============================================================================

from .config_accessors import (
    # Registry Status Flags
    _REGISTRY_AVAILABLE as _ACCESSORS_REGISTRY_AVAILABLE,  # Renamed to avoid conflict
)
from .config_accessors import (
    _REGISTRY_IMPORT_ERROR as _ACCESSORS_REGISTRY_IMPORT_ERROR,
)
from .config_accessors import (
    _REGISTRY_INITIALIZED as _ACCESSORS_REGISTRY_INITIALIZED,
)

# =============================================================================
# Configuration Accessor Functions
# =============================================================================
from .config_accessors import (
    # Enhanced Config Accessor Class
    EnhancedConfigAccessor,
    # Registry Initialization
    _init_registry,
    check_transformation_system_compatibility,
    # Container Creation Functions
    create_dataset_config_container,
    # Handler Creation
    create_dataset_handler,
    create_experimental_setup_from_dict,
    create_filter_config_container,
    # Phase 5: Configuration Helpers (NEW)
    create_handler_compatible_config,
    create_processing_config_container,
    create_structural_features_config_container,
    create_transformation_config_container,
    create_transforms_from_config,
    get_all_transforms,
    get_atom_features,
    # Transform Registry & Discovery
    get_available_transforms,
    get_bond_features,
    get_charge_handling_config,
    get_combined_transforms,
    get_combined_transforms_as_dicts,
    get_config_value,
    get_config_with_fallback,
    get_coordinate_units,
    get_data_config,
    get_dataset_appropriate_structural_features,
    get_dataset_config,
    # Dataset Constants & Properties
    get_dataset_constants,
    get_dataset_specific_config,
    # Core Configuration Accessors
    get_dataset_type,
    get_default_experimental_setup,
    get_default_registry,
    get_descriptor_config,
    get_energy_units,
    # Experimental Setup Management
    get_experimental_setup,
    get_experimental_setups_for_dataset,
    get_feature_compatibility_report,
    # Filter Configuration
    get_filter_config,
    get_geometric_features_config,
    get_handler_compatible_config,
    get_identifier_keys,
    get_molecule_creation_strategy,
    get_optional_properties,
    get_parameter,
    get_preprocessing_config,
    # Processing Configuration
    get_processing_config,
    get_property_availability,
    # Phase 5: Dataset Information (NEW)
    get_raw_data_info,
    get_required_properties,
    # Research API
    get_research_recommendations,
    get_selected_descriptors,
    get_setup,
    # Standard Transforms Accessors (NEW)
    get_standard_transforms,
    get_standard_transforms_as_dicts,
    get_stereochemistry_config,
    # Structural Features Configuration
    get_structural_features_config,
    get_structural_features_preprocessing_config,
    get_supported_features,
    # Utility Functions
    get_transform,
    # Transform Configuration Accessors
    get_transform_config,
    get_transform_info,
    get_transform_parameter,
    get_transform_registry_info,
    get_transformation_cache_key,
    get_transformation_config,
    get_transformation_config_summary,
    get_transformation_performance_metrics,
    get_transformation_validation_config,
    # Transformation Configuration
    get_transformations_config,
    get_transforms_by_category,
    # Uncertainty Configuration
    get_uncertainty_config,
    # -------------------------------------------------------------------------
    # Phase 5: Registry Public API (NEW)
    get_valid_dataset_types,
    has_standard_transforms,
    # Descriptor Configuration
    is_descriptors_enabled,
    is_feature_supported,
    is_handler_type,
    is_structural_features_enabled,
    is_transformation_strict_mode_enabled,
    is_transformation_validation_enabled,
    is_uncertainty_enabled,
    is_valid_dataset_type,
    list_available_transforms,
    list_enabled_experimental_setups,
    list_experimental_setups,
    # Legacy Compatibility
    migrate_legacy_transformation_config,
    registry_get,
    registry_is_registered,
    # Phase 5: Registry Integration for Accessor Functions (NEW)
    # -------------------------------------------------------------------------
    # Registry Wrapper Functions
    registry_list_all,
    save_experimental_setup,
    should_enable_stereochemistry_preprocessing,
    should_pass_coordinates_to_structural_features,
    should_pass_mulliken_charges_to_structural_features,
    validate_config_structure,
    validate_dataset_config,
    # Dataset Type Validation Functions
    validate_dataset_type,
    validate_structural_features_for_dataset,
    validate_transform_config,
    validate_transformation_config,
    validate_transformation_system,
)

# =============================================================================
# Configuration Constants
# =============================================================================
from .config_constants import (
    _CACHE_INVALIDATION_REGISTERED,  # Flag indicating if cache invalidation is set up
    # -------------------------------------------------------------------------
    # Phase 3: Registry Integration Status (NEW)
    # -------------------------------------------------------------------------
    _REGISTRY_AVAILABLE,  # Flag indicating if registry is available
    _REGISTRY_IMPORT_ERROR,  # Error message if registry import failed (for debugging)
    _REGISTRY_INITIALIZED,  # Flag indicating if registry initialization was attempted
    DATASET_ROOT_DIR,
    RAW_DATA_DOWNLOAD_URL,
    # Dataset Constants
    RAW_NPZ_FILENAME,
    _init_registry,  # Function to initialize registry (for testing/debugging)
    check_handler_feature_support,
    clear_all_caches,
    clear_handler_caches,
    # Cache Management
    clear_transformation_caches,
    create_handler_config_from_constants,
    ensure_complete_compatibility,
    ensure_handler_constant_compatibility,
    get_all_cache_info,
    get_cached_handler_config,
    get_compatible_transforms_for_handler,
    # Debug & Diagnostics
    get_complete_constants_debug_info,
    get_default_handler_type,  # Default handler type ('DFT')
    get_handler_cache_info,
    get_handler_compatibility_info,
    # -------------------------------------------------------------------------
    # Handler Constants & Utilities (Existing, now registry-aware)
    # -------------------------------------------------------------------------
    get_handler_constants,
    get_handler_coordinate_units_dynamic,  # Coordinate units from registry
    # Handler Property Accessors (Dynamic)
    get_handler_feature_support,  # Feature support dict from registry
    get_handler_feature_support_legacy,  # Deprecated: use get_handler_feature_support()
    get_handler_identifier_keys,
    get_handler_identifier_keys_dynamic,  # Identifier keys from registry
    get_handler_molecule_creation_strategy,  # Molecule creation strategy from registry
    get_handler_optional_properties,  # Optional properties from registry
    get_handler_property_requirements,
    get_handler_required_properties,  # Required properties from registry
    get_handler_required_properties_legacy,  # Deprecated: use get_handler_required_properties()
    get_handler_transform_compatibility,
    get_incompatible_transforms_for_handler,
    get_legacy_compatible_constants,
    # Migration Constants
    get_migration_compatibility_constants,
    # -------------------------------------------------------------------------
    # Phase 3: Dynamic Registry Wrapper Functions (NEW)
    # -------------------------------------------------------------------------
    # These functions delegate to the dataset registry when available,
    # falling back to hardcoded constants for backward compatibility.
    # Handler Type Discovery
    get_supported_handler_types,  # Dynamic list of registered handler types
    # -------------------------------------------------------------------------
    # Phase 3: Legacy Deprecation Helpers (NEW)
    # -------------------------------------------------------------------------
    get_supported_handler_types_legacy,  # Deprecated: use get_supported_handler_types()
    get_transformation_cache_info,
    # Transformation Constants
    get_transformation_constants,
    is_handler_type_supported,  # Check if handler type is registered
    validate_complete_environment,
    validate_experimental_setup_for_handler,
    validate_handler_configuration,
    # Environment Validation
    validate_handler_environment,
)

# =============================================================================
# Configuration Container Classes
# =============================================================================
from .config_containers import (
    _FALLBACK_VALID_TYPES,
    # Primary Configuration Containers
    DatasetConfig,
    DescriptorCategoryConfig,
    DescriptorConfig,
    ExperimentalSetup,
    FilterConfig,
    HandlerConfig,
    ProcessingConfig,
    StructuralFeaturesConfig,
    TransformationConfig,
    TransformSpec,
    # Dynamic Type Validation Helpers
    _get_valid_dataset_types,
    _is_valid_dataset_type,
    check_configuration_compatibility,
    create_ablation_study_setups,
    # Container Factory Functions
    create_dataset_config_from_global,
    create_default_descriptor_config,
    create_default_experimental_setups,
    create_descriptor_config_from_yaml,
    create_experimental_setup_from_dict,
    create_filter_config_from_global,
    # Handler Configuration Functions
    create_handler_config,
    create_handler_configuration_bundle,
    create_migration_compatible_config,
    create_minimal_config_for_testing,
    create_minimal_descriptor_config,
    create_processing_config_from_global,
    create_structural_features_config_from_global,
    # Transform Specification Functions
    create_transform_spec_from_dict,
    create_transformation_config_from_dict,
    create_transformation_config_from_global,
    get_config_summary,
    validate_handler_configuration_bundle,
    # Configuration Validation & Compatibility
    validate_transformation_compatibility,
    # Registry Integration Verification
    verify_container_registry_integration,
)
from .config_containers import (
    # -------------------------------------------------------------------------
    # Phase 4: Registry Integration for Container Classes (NEW)
    # -------------------------------------------------------------------------
    # Registry Status
    _REGISTRY_AVAILABLE as _CONTAINERS_REGISTRY_AVAILABLE,  # Renamed to avoid conflict with config_constants
)
from .config_loader import (
    # -------------------------------------------------------------------------
    # Phase 5: Config Loader Registry Integration (NEW)
    # -------------------------------------------------------------------------
    _REGISTRY_AVAILABLE as _LOADER_REGISTRY_AVAILABLE,  # Renamed to avoid conflict
)
from .config_loader import (
    _REGISTRY_IMPORT_ERROR as _LOADER_REGISTRY_IMPORT_ERROR,
)
from .config_loader import (
    _REGISTRY_INITIALIZED as _LOADER_REGISTRY_INITIALIZED,
)
from .config_loader import (
    _get_default_dataset_type,
    # Migration & Validation Helpers (standard_transforms support)
    check_migration_status,
    clear_config_cache,
    create_example_config,
    get_config_hash,
    get_config_load_time,
    get_config_statistics,
    get_config_stats,
    get_global_config_state,
    get_migration_report,
    get_transformation_feature_status,
    get_validation_report,
    is_config_loaded,
    load_config,
    load_config_with_validation,
    load_transformation_config,
    migrate_legacy_config,
    print_transformation_status,
    recommend_validation_level,
    reload_config,
    set_global_config_state,
    validate_config_file,
)
from .config_loader import (
    _init_registry as _loader_init_registry,
)

# =============================================================================
# Schema Validation & Migration
# =============================================================================
from .config_schemas import (
    # Migration Class
    ConfigMigration,
    DescriptorCategoryConfigSchema,
    DescriptorConfigSchema,
    DescriptorSchemaValidator,
    ExperimentSchema,
    ExperimentSchemaValidator,
    PluginConfigSchema,
    PluginSchemaValidator,
    PluginValidationLevel,
    # Schema Classes
    TransformationSchema,
    ValidationConfig,
    WavefunctionConfigSchema,
    WavefunctionProcessingConfigSchema,
    WavefunctionUncertaintyConfigSchema,
    # Validator Classes
    YAMLSchemaValidator,
    create_default_experiment_config,
    create_default_plugin_config,
    # Configuration Creation Functions
    create_example_enhanced_config,
    create_example_experiments_config,
    create_example_legacy_configs,
    create_example_plugin_config,
    create_migrator,
    # Factory Functions
    create_validator,
    get_enhanced_transformation_config,
    get_experiment_config_summary,
    get_plugin_config_summary,
    # Validation Functions
    load_and_validate_yaml_config,
    merge_plugin_configs,
    validate_current_config,
    # Plugin & Experiment Validation
    validate_experiment_config_file,
    validate_plugin_config_file,
    validate_wavefunction_config,
    validate_yaml_config_string,
)
from .data_refining import (
    # -------------------------------------------------------------------------
    # Phase 6: Data Refining Registry Integration (NEW)
    # -------------------------------------------------------------------------
    # Registry Status
    _REGISTRY_AVAILABLE as _REFINING_REGISTRY_AVAILABLE,  # Renamed to avoid conflict
)
from .data_refining import (
    _REGISTRY_IMPORT_ERROR as _REFINING_REGISTRY_IMPORT_ERROR,
)
from .data_refining import (
    _REGISTRY_INITIALIZED as _REFINING_REGISTRY_INITIALIZED,
)
from .data_refining import (
    # Dynamic Dataset Type Queries
    _get_available_dataset_types as _refining_get_available_dataset_types,
)

# =============================================================================
# Data Refinement Functions
# =============================================================================
from .data_refining import (
    # Feature-Based Queries
    _get_dataset_feature,
    _get_dataset_refinement_category,
    apply_dataset_specific_refinement,
    calculate_dmc_uncertainty_weights,
    # Handler-Based Refinement
    create_refinement_handler,
    demonstrate_migration_patterns,
    # DMC-Specific
    detect_dmc_statistical_outliers,
    # Diagnostics
    diagnose_vibrational_data_structure,
    get_migration_benefits,
    get_module_migration_summary,
    get_refinement_statistics_with_handler,
    # Logging
    log_data_refinement_status,
    log_vibration_refinement_status,
    # Migration Support
    migrate_refinement_call_to_handler,
    # Core Refinement
    refine_molecular_data,
    refine_molecular_data_with_handler,
    refine_molecular_vibrations,
    validate_refined_data_quality,
    validate_refined_data_with_handler,
    verify_migration_completeness,
)
from .data_refining import (
    _init_registry as _refining_init_registry,
)
from .data_refining import (
    _is_dataset_type_registered as _refining_is_dataset_type_registered,
)
from .data_refining import (
    # Registry Status Reporting
    get_registry_status as get_refining_registry_status,
)
from .validators import (
    # -------------------------------------------------------------------------
    # Phase 6: Validators Registry Integration (NEW)
    # -------------------------------------------------------------------------
    # Registry Status Flags
    _REGISTRY_AVAILABLE as _VALIDATORS_REGISTRY_AVAILABLE,  # Renamed to avoid conflict
)
from .validators import (
    _REGISTRY_IMPORT_ERROR as _VALIDATORS_REGISTRY_IMPORT_ERROR,
)
from .validators import (
    _REGISTRY_INITIALIZED as _VALIDATORS_REGISTRY_INITIALIZED,
)

# =============================================================================
# Validation Functions
# =============================================================================
from .validators import (
    # Validator Classes
    TransformValidator,
    ValidationIssueDetail,
    # Validation Result Classes
    ValidationResult,
    ValidationSeverity,
    convert_to_scalar,
    # Reporting
    create_validation_report,
    is_valid_molecule_identifier,
    # Property & Value Validation
    is_value_valid_and_not_nan,
    # Decorators
    must_check,
    run_handler_validation_tests,
    run_transformation_validation_tests,
    run_validation_diagnostics,
    safe_get_value,
    validate_array_shape,
    validate_atomic_numbers,
    validate_batch_consistency,
    validate_coordinates_3d,
    validate_descriptor_cache_settings,
    validate_descriptor_category_compatibility,
    # Descriptor Validation
    validate_descriptor_config,
    validate_experimental_setup,
    # Handler Validation
    validate_handler_compatibility,
    validate_handler_molecular_batch,
    validate_molecular_data_dict,
    # Molecular Data Validation
    validate_molecular_structure,
    validate_numeric_range,
    validate_property_value,
    validate_transform_composition_rules,
    validate_transform_sequence_semantics,
    # Transform Validation
    validate_transform_spec,
    validate_transformation_config,
    validate_transforms_with_fallback,
    validate_uncertainty_data,
)
from .validators import (
    # Dynamic Dataset Type Queries
    _get_available_dataset_types as _validators_get_available_dataset_types,
)
from .validators import (
    # Feature-Based Queries
    _get_dataset_feature as _validators_get_dataset_feature,
)
from .validators import (
    _get_dataset_required_properties as _validators_get_dataset_required_properties,
)
from .validators import (
    _get_handler_compatibility_checks as _validators_get_handler_compatibility_checks,
)
from .validators import (
    # Registry Initialization
    _init_registry as _validators_init_registry,
)
from .validators import (
    _is_dataset_type_registered as _validators_is_dataset_type_registered,
)
from .validators import (
    # Registry Status Reporting
    get_registry_status as get_validators_registry_status,
)

# =============================================================================
# Public API (__all__)
# =============================================================================

__all__ = [
    # -------------------------------------------------------------------------
    # Configuration Loading
    # -------------------------------------------------------------------------
    "load_config",
    "get_config_stats",
    "clear_config_cache",
    "get_global_config_state",
    "set_global_config_state",
    "is_config_loaded",
    "get_config_load_time",
    "get_config_hash",
    "load_config_with_validation",
    "reload_config",
    "validate_config_file",
    "get_config_statistics",
    "create_example_config",
    "migrate_legacy_config",
    "load_transformation_config",
    "get_validation_report",
    "get_migration_report",
    # Migration & Validation Helpers (standard_transforms support)
    "check_migration_status",
    "recommend_validation_level",
    "get_transformation_feature_status",
    "print_transformation_status",
    # Phase 5: Config Loader Registry (NEW)
    "_LOADER_REGISTRY_AVAILABLE",
    "_LOADER_REGISTRY_IMPORT_ERROR",
    "_LOADER_REGISTRY_INITIALIZED",
    "_loader_init_registry",
    "_get_default_dataset_type",
    # -------------------------------------------------------------------------
    # Configuration Containers
    # -------------------------------------------------------------------------
    # Container Classes
    "DatasetConfig",
    "FilterConfig",
    "StructuralFeaturesConfig",
    "ProcessingConfig",
    "HandlerConfig",
    "TransformSpec",
    "ExperimentalSetup",
    "TransformationConfig",
    "DescriptorConfig",
    "DescriptorCategoryConfig",
    # Container Factory Functions
    "create_dataset_config_from_global",
    "create_filter_config_from_global",
    "create_processing_config_from_global",
    "create_structural_features_config_from_global",
    "create_transformation_config_from_global",
    "create_descriptor_config_from_yaml",
    "create_default_descriptor_config",
    "create_minimal_descriptor_config",
    # Handler Configuration
    "create_handler_config",
    "create_handler_configuration_bundle",
    "validate_handler_configuration_bundle",
    "create_migration_compatible_config",
    # Transform Specification
    "create_transform_spec_from_dict",
    "create_experimental_setup_from_dict",
    "create_transformation_config_from_dict",
    "create_default_experimental_setups",
    "create_ablation_study_setups",
    # Configuration Validation
    "validate_transformation_compatibility",
    "get_config_summary",
    "check_configuration_compatibility",
    "create_minimal_config_for_testing",
    # Phase 4: Container Registry Integration (NEW)
    "_CONTAINERS_REGISTRY_AVAILABLE",
    "_FALLBACK_VALID_TYPES",
    "_get_valid_dataset_types",
    "_is_valid_dataset_type",
    "verify_container_registry_integration",
    # -------------------------------------------------------------------------
    # Configuration Accessors
    # -------------------------------------------------------------------------
    # Classes
    "EnhancedConfigAccessor",
    # Core Accessors
    "get_dataset_type",
    "get_config_value",
    "get_dataset_config",
    "get_data_config",
    "get_property_availability",
    # Uncertainty
    "get_uncertainty_config",
    "is_uncertainty_enabled",
    # Structural Features
    "get_structural_features_config",
    "get_structural_features_preprocessing_config",
    "get_charge_handling_config",
    "get_geometric_features_config",
    "get_stereochemistry_config",
    "is_structural_features_enabled",
    "get_atom_features",
    "get_bond_features",
    "should_pass_coordinates_to_structural_features",
    "should_pass_mulliken_charges_to_structural_features",
    "should_enable_stereochemistry_preprocessing",
    "get_dataset_appropriate_structural_features",
    "validate_structural_features_for_dataset",
    "get_feature_compatibility_report",
    # Filtering
    "get_filter_config",
    # Transformations
    "get_transformations_config",
    "get_transformation_config",
    "get_transformation_validation_config",
    "is_transformation_validation_enabled",
    "is_transformation_strict_mode_enabled",
    "get_transformation_cache_key",
    "get_transformation_performance_metrics",
    "get_transformation_config_summary",
    "validate_transformation_config",
    "create_transforms_from_config",
    "validate_transformation_system",
    # Transform Registry
    "get_available_transforms",
    "get_transforms_by_category",
    "get_transform_info",
    "get_transform_registry_info",
    "list_available_transforms",
    "get_all_transforms",
    # Transform Configuration
    "get_transform_config",
    "get_transform_parameter",
    "get_dataset_specific_config",
    "validate_transform_config",
    # Experimental Setups
    "get_experimental_setup",
    "list_experimental_setups",
    "list_enabled_experimental_setups",
    "get_default_experimental_setup",
    "get_experimental_setups_for_dataset",
    "create_experimental_setup_from_dict",
    "save_experimental_setup",
    # Standard Transforms (NEW)
    "get_standard_transforms",
    "get_standard_transforms_as_dicts",
    "get_combined_transforms",
    "get_combined_transforms_as_dicts",
    "has_standard_transforms",
    # Research
    "get_research_recommendations",
    # Processing
    "get_processing_config",
    "get_config_with_fallback",
    "validate_config_structure",
    # Dataset Properties
    "get_dataset_constants",
    "get_required_properties",
    "get_identifier_keys",
    "get_handler_compatible_config",
    "get_preprocessing_config",
    # Container Creation
    "create_dataset_config_container",
    "create_filter_config_container",
    "create_processing_config_container",
    "create_structural_features_config_container",
    "create_transformation_config_container",
    # Handler
    "create_dataset_handler",
    # Legacy
    "migrate_legacy_transformation_config",
    "check_transformation_system_compatibility",
    # Descriptors
    "is_descriptors_enabled",
    "get_descriptor_config",
    "get_selected_descriptors",
    # Utilities
    "get_transform",
    "get_parameter",
    "get_setup",
    # Phase 5: Registry Public API (NEW)
    "registry_list_all",
    "registry_get",
    "registry_is_registered",
    "get_default_registry",
    "get_valid_dataset_types",
    "is_valid_dataset_type",
    "validate_dataset_type",
    "is_handler_type",
    "get_optional_properties",
    "get_supported_features",
    "is_feature_supported",
    "get_raw_data_info",
    "get_coordinate_units",
    "get_energy_units",
    "get_molecule_creation_strategy",
    "create_handler_compatible_config",
    "validate_dataset_config",
    "_init_registry",
    "_ACCESSORS_REGISTRY_AVAILABLE",
    "_ACCESSORS_REGISTRY_INITIALIZED",
    "_ACCESSORS_REGISTRY_IMPORT_ERROR",
    # -------------------------------------------------------------------------
    # Schema Validation & Migration
    # -------------------------------------------------------------------------
    # Schema Classes
    "TransformationSchema",
    "PluginValidationLevel",
    "PluginConfigSchema",
    "DescriptorConfigSchema",
    "WavefunctionProcessingConfigSchema",
    "WavefunctionUncertaintyConfigSchema",
    "WavefunctionConfigSchema",
    "ExperimentSchema",
    "ValidationConfig",
    "DescriptorCategoryConfigSchema",
    # Validators
    "YAMLSchemaValidator",
    "PluginSchemaValidator",
    "ExperimentSchemaValidator",
    "DescriptorSchemaValidator",
    # Migration
    "ConfigMigration",
    # Factory Functions
    "create_validator",
    "create_migrator",
    # Validation Functions
    "load_and_validate_yaml_config",
    "validate_yaml_config_string",
    "validate_wavefunction_config",
    "validate_current_config",
    # Configuration Creation
    "create_example_enhanced_config",
    "create_example_legacy_configs",
    "get_enhanced_transformation_config",
    "create_default_plugin_config",
    "create_example_plugin_config",
    "create_default_experiment_config",
    "create_example_experiments_config",
    # Plugin & Experiment
    "validate_experiment_config_file",
    "get_experiment_config_summary",
    "validate_plugin_config_file",
    "merge_plugin_configs",
    "get_plugin_config_summary",
    # -------------------------------------------------------------------------
    # Configuration Constants
    # -------------------------------------------------------------------------
    # Dataset Constants
    "RAW_NPZ_FILENAME",
    "RAW_DATA_DOWNLOAD_URL",
    "DATASET_ROOT_DIR",
    # Phase 3: Registry Functions (NEW)
    "get_supported_handler_types",
    "get_default_handler_type",
    "is_handler_type_supported",
    "get_handler_feature_support",
    "get_handler_required_properties",
    "get_handler_optional_properties",
    "get_handler_identifier_keys_dynamic",
    "get_handler_coordinate_units_dynamic",
    "get_handler_molecule_creation_strategy",
    # Handler Utilities
    "get_handler_constants",
    "get_handler_identifier_keys",
    "validate_handler_configuration",
    "get_handler_compatibility_info",
    "check_handler_feature_support",
    "get_handler_property_requirements",
    "create_handler_config_from_constants",
    # Transformation Constants
    "get_transformation_constants",
    "get_handler_transform_compatibility",
    "get_compatible_transforms_for_handler",
    "get_incompatible_transforms_for_handler",
    "validate_experimental_setup_for_handler",
    # Migration Constants
    "get_migration_compatibility_constants",
    "get_legacy_compatible_constants",
    # Environment Validation
    "validate_handler_environment",
    "validate_complete_environment",
    "ensure_handler_constant_compatibility",
    "ensure_complete_compatibility",
    # Cache Management
    "clear_transformation_caches",
    "get_transformation_cache_info",
    "get_cached_handler_config",
    "clear_handler_caches",
    "get_handler_cache_info",
    "clear_all_caches",
    "get_all_cache_info",
    # Diagnostics
    "get_complete_constants_debug_info",
    # -------------------------------------------------------------------------
    # Phase 3: Registry Integration Status (NEW)
    # -------------------------------------------------------------------------
    "_REGISTRY_AVAILABLE",
    "_CACHE_INVALIDATION_REGISTERED",
    "_REGISTRY_IMPORT_ERROR",
    "_REGISTRY_INITIALIZED",
    "_init_registry",
    # -------------------------------------------------------------------------
    # Phase 3: Legacy Deprecation Helpers (NEW)
    # -------------------------------------------------------------------------
    "get_supported_handler_types_legacy",
    "get_handler_feature_support_legacy",
    "get_handler_required_properties_legacy",
    # -------------------------------------------------------------------------
    # Validation Functions
    # -------------------------------------------------------------------------
    # Validation Classes
    "ValidationResult",
    "ValidationSeverity",
    "ValidationIssueDetail",
    "TransformValidator",
    # Decorators
    "must_check",
    # Molecular Validation
    "validate_molecular_structure",
    "validate_molecular_data_dict",
    "validate_uncertainty_data",
    "validate_coordinates_3d",
    "validate_atomic_numbers",
    "validate_batch_consistency",
    "validate_handler_molecular_batch",
    # Property Validation
    "is_value_valid_and_not_nan",
    "validate_array_shape",
    "validate_numeric_range",
    "validate_property_value",
    "is_valid_molecule_identifier",
    "safe_get_value",
    "convert_to_scalar",
    # Transform Validation
    "validate_transform_spec",
    "validate_experimental_setup",
    "validate_transformation_config",
    "validate_transform_sequence_semantics",
    "validate_transform_composition_rules",
    "validate_transforms_with_fallback",
    # Descriptor Validation
    "validate_descriptor_config",
    "validate_descriptor_category_compatibility",
    "validate_descriptor_cache_settings",
    # Handler Validation
    "validate_handler_compatibility",
    # Reporting
    "create_validation_report",
    "run_validation_diagnostics",
    "run_handler_validation_tests",
    "run_transformation_validation_tests",
    # -------------------------------------------------------------------------
    # Phase 6: Validators Registry Integration (NEW)
    # -------------------------------------------------------------------------
    # Registry Status
    "_VALIDATORS_REGISTRY_AVAILABLE",
    "_VALIDATORS_REGISTRY_INITIALIZED",
    "_VALIDATORS_REGISTRY_IMPORT_ERROR",
    "_validators_init_registry",
    # Dynamic Dataset Type Queries
    "_validators_get_available_dataset_types",
    "_validators_is_dataset_type_registered",
    # Feature-Based Queries
    "_validators_get_dataset_feature",
    "_validators_get_dataset_required_properties",
    "_validators_get_handler_compatibility_checks",
    # Registry Status Reporting
    "get_validators_registry_status",
    # -------------------------------------------------------------------------
    # Data Refinement Functions
    # -------------------------------------------------------------------------
    # Core Refinement
    "refine_molecular_data",
    "refine_molecular_vibrations",
    "validate_refined_data_quality",
    "apply_dataset_specific_refinement",
    # DMC-Specific
    "detect_dmc_statistical_outliers",
    "calculate_dmc_uncertainty_weights",
    # Handler-Based
    "create_refinement_handler",
    "refine_molecular_data_with_handler",
    "validate_refined_data_with_handler",
    "get_refinement_statistics_with_handler",
    # Logging
    "log_data_refinement_status",
    "log_vibration_refinement_status",
    # Migration
    "migrate_refinement_call_to_handler",
    "demonstrate_migration_patterns",
    "verify_migration_completeness",
    "get_migration_benefits",
    "get_module_migration_summary",
    # Diagnostics
    "diagnose_vibrational_data_structure",
    # -------------------------------------------------------------------------
    # Phase 6: Data Refining Registry Integration (NEW)
    # -------------------------------------------------------------------------
    # Registry Status
    "_REFINING_REGISTRY_AVAILABLE",
    "_REFINING_REGISTRY_INITIALIZED",
    "_REFINING_REGISTRY_IMPORT_ERROR",
    "_refining_init_registry",
    # Dynamic Dataset Type Queries
    "_refining_get_available_dataset_types",
    "_refining_is_dataset_type_registered",
    # Feature-Based Queries
    "_get_dataset_feature",
    "_get_dataset_refinement_category",
    # Registry Status Reporting
    "get_refining_registry_status",
]

# =============================================================================
# Version & Metadata
# =============================================================================

__version__ = "1.8.0"  # Updated for standard_transforms validation support in validators, config_schemas, config_loader
__author__ = "milia Pipeline Development Team"
__description__ = "Configuration management system for milia molecular graph ML/DL pipeline"

# =============================================================================
# Module Initialization
# =============================================================================

# Import logger for module-level logging
import logging

_logger = logging.getLogger(__name__)
_logger.debug("Configuration module initialized successfully")

# Validate critical imports on module load
try:
    # Verify core components are available
    from .config_accessors import get_dataset_type
    from .config_containers import DatasetConfig
    from .config_loader import _CONFIG

    _logger.debug("Core configuration components validated")
except ImportError as e:
    _logger.warning(f"Some configuration components may be unavailable: {e}")

# Phase 3: Log registry integration status
try:
    from .config_constants import _CACHE_INVALIDATION_REGISTERED, _REGISTRY_AVAILABLE

    if _REGISTRY_AVAILABLE:
        _logger.debug("Phase 3: Dataset registry integration available")
        if _CACHE_INVALIDATION_REGISTERED:
            _logger.debug("Phase 3: Cache invalidation callback registered")
    else:
        _logger.debug("Phase 3: Dataset registry not available, using legacy constants")
except ImportError:
    _logger.debug("Phase 3: Registry status not available")

# Phase 4: Log container registry integration status
try:
    from .config_containers import _REGISTRY_AVAILABLE as _CONTAINERS_REGISTRY_AVAILABLE
    from .config_containers import verify_container_registry_integration

    if _CONTAINERS_REGISTRY_AVAILABLE:
        _logger.debug("Phase 4: Container registry integration available")
        # Optionally verify integration on module load (can be expensive)
        # results = verify_container_registry_integration()
        # _logger.debug(f"Phase 4: Container integration status: {results['overall_status']}")
    else:
        _logger.debug("Phase 4: Container registry not available, using fallback validation")
except ImportError:
    _logger.debug("Phase 4: Container registry status not available")

# Phase 5: Log accessor registry integration status
try:
    from .config_accessors import _REGISTRY_AVAILABLE as _ACCESSORS_REGISTRY_AVAILABLE
    from .config_accessors import _REGISTRY_INITIALIZED as _ACCESSORS_REGISTRY_INITIALIZED

    if _ACCESSORS_REGISTRY_AVAILABLE:
        _logger.debug("Phase 5: Accessor registry integration available")
        if _ACCESSORS_REGISTRY_INITIALIZED:
            _logger.debug("Phase 5: Accessor registry initialized successfully")
    else:
        _logger.debug("Phase 5: Accessor registry not available, using legacy fallback")
except ImportError:
    _logger.debug("Phase 5: Accessor registry status not available")

# Phase 5: Log config_loader registry integration status
try:
    from .config_loader import _REGISTRY_AVAILABLE as _LOADER_REGISTRY_AVAILABLE_CHECK
    from .config_loader import _REGISTRY_INITIALIZED as _LOADER_REGISTRY_INITIALIZED_CHECK

    if _LOADER_REGISTRY_AVAILABLE_CHECK:
        _logger.debug("Phase 5: Config loader registry integration available")
        if _LOADER_REGISTRY_INITIALIZED_CHECK:
            _logger.debug("Phase 5: Config loader registry initialized successfully")
    else:
        _logger.debug("Phase 5: Config loader registry not available, using 'DFT' fallback")
except ImportError:
    _logger.debug("Phase 5: Config loader registry status not available")

# Phase 6: Log data_refining registry integration status
try:
    from .data_refining import _REGISTRY_AVAILABLE as _REFINING_REGISTRY_AVAILABLE_CHECK
    from .data_refining import _REGISTRY_INITIALIZED as _REFINING_REGISTRY_INITIALIZED_CHECK

    if _REFINING_REGISTRY_AVAILABLE_CHECK:
        _logger.debug("Phase 6: Data refining registry integration available")
        if _REFINING_REGISTRY_INITIALIZED_CHECK:
            _logger.debug("Phase 6: Data refining registry initialized successfully")
    else:
        _logger.debug("Phase 6: Data refining registry not available, using legacy fallback")
except ImportError:
    _logger.debug("Phase 6: Data refining registry status not available")

# Phase 6: Log validators registry integration status
try:
    from .validators import _REGISTRY_AVAILABLE as _VALIDATORS_REGISTRY_AVAILABLE_CHECK
    from .validators import _REGISTRY_INITIALIZED as _VALIDATORS_REGISTRY_INITIALIZED_CHECK

    if _VALIDATORS_REGISTRY_AVAILABLE_CHECK:
        _logger.debug("Phase 6: Validators registry integration available")
        if _VALIDATORS_REGISTRY_INITIALIZED_CHECK:
            _logger.debug("Phase 6: Validators registry initialized successfully")
    else:
        _logger.debug("Phase 6: Validators registry not available, using legacy fallback")
except ImportError:
    _logger.debug("Phase 6: Validators registry status not available")

# Clean up namespace
del logging
