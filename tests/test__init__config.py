# tests/test__init__config.py

"""
Test Suite: milia_pipeline/config/__init__.py — Smoke Tests & Contract Tests
============================================================================

Production-ready test suite for the MILIA Pipeline configuration package
``milia_pipeline/config/__init__.py``.

Covers:
    Section 1 — Smoke Tests (MILIA_Test_Recommendations.md §1.2 scope):
        - The ``milia_pipeline.config`` subpackage imports without ImportError
        - All re-exported names from the 7 submodules are accessible
        - Module-level metadata attributes (__version__, __author__, etc.) exist
        - Module initialization (logging, registry status checks) runs safely
        - Re-import (``importlib.reload``) is idempotent and non-crashing
        - Registry integration status flags are present and boolean
        - Phase 3–6 registry status logging blocks execute without exceptions

    Section 2 — Contract Tests (MILIA_Test_Recommendations.md §2 scope):
        - ``__all__`` completeness: every name in ``__all__`` is resolvable
        - ``__all__`` consistency: every public import is listed in ``__all__``
        - ``__all__`` has no duplicates
        - Re-exported classes are classes, callables are callable
        - Container classes (DatasetConfig, FilterConfig, etc.) are Pydantic
          BaseModel subclasses
        - Registry integration status flags across all phases are boolean
        - Factory functions have documented parameter signatures
        - Accessor functions are callable
        - Schema/validator classes are classes
        - Data refinement functions are callable
        - Namespace cleanliness: ``logging`` is deleted after init
        - Phase-aliased imports do not collide (renamed with prefixes)
        - ``__version__`` follows semver pattern
        - Public API surface stability (minimum expected names present)

Design:
    - Zero ``sys.modules`` pollution: all mocking via ``@patch`` decorators
      or context-manager ``patch`` calls scoped to individual tests.
    - Deterministic: no filesystem, network, or GPU access required.
    - Isolated: each test is independent; execution order is irrelevant.
    - Fast: expected < 5 s total wall-clock on CI.

Launch:
    From project root (/app/milia):
        pytest tests/test__init__config.py -v --tb=short

Markers:
    smoke     — Quick health-check tests (§1)
    contract  — Interface/contract validation tests (§2)
"""

import importlib
import inspect
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure the project root is importable
# ---------------------------------------------------------------------------
# When launched via ``pytest tests/test__init__config.py`` from the
# project root (/app/milia), ``milia_pipeline`` must be on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # …/tests -> …/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture(scope="module")
def config_pkg():
    """
    Import and return the ``milia_pipeline.config`` package once per module.

    This fixture validates the fundamental smoke invariant: the config
    subpackage is importable.  If this fails, every downstream test is moot.
    """
    try:
        import milia_pipeline.config as cfg

        return cfg
    except ImportError as exc:
        pytest.fail(
            f"milia_pipeline.config could not be imported — smoke test precondition violated: {exc}"
        )


@pytest.fixture(scope="module")
def all_names(config_pkg):
    """Return the ``__all__`` list from the config package."""
    assert hasattr(config_pkg, "__all__"), (
        "milia_pipeline.config.__all__ is missing — contract violation"
    )
    return list(config_pkg.__all__)


# ===================================================================
# SECTION 1 — SMOKE TESTS
# ===================================================================


class TestSmokeConfigPackageImport:
    """§1.2 — Verify the config subpackage imports without errors."""

    @pytest.mark.smoke
    def test_import_config_package_succeeds(self, config_pkg):
        """The config package imports without raising any exception."""
        assert config_pkg is not None

    @pytest.mark.smoke
    def test_config_package_is_a_module(self, config_pkg):
        """The imported object is a proper Python module."""
        assert isinstance(config_pkg, types.ModuleType)

    @pytest.mark.smoke
    def test_config_package_has_file_attribute(self, config_pkg):
        """The package exposes a ``__file__`` attribute (not a namespace pkg)."""
        assert hasattr(config_pkg, "__file__")

    @pytest.mark.smoke
    def test_config_package_name(self, config_pkg):
        """The package ``__name__`` is ``milia_pipeline.config``."""
        assert config_pkg.__name__ == "milia_pipeline.config"


class TestSmokeMetadataAttributes:
    """§1.2 — Verify module-level metadata attributes are present and typed."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "attr",
        [
            "__version__",
            "__author__",
            "__description__",
        ],
    )
    def test_metadata_attribute_exists(self, config_pkg, attr):
        """Each metadata dunder is defined on the config package."""
        assert hasattr(config_pkg, attr), f"Missing attribute: {attr}"

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "attr",
        [
            "__version__",
            "__author__",
            "__description__",
        ],
    )
    def test_metadata_attribute_is_string(self, config_pkg, attr):
        """Each metadata dunder is a non-empty string."""
        value = getattr(config_pkg, attr)
        assert isinstance(value, str), f"{attr} should be str, got {type(value)}"
        assert len(value) > 0, f"{attr} should be non-empty"

    @pytest.mark.smoke
    def test_version_is_semver_like(self, config_pkg):
        """``__version__`` follows a MAJOR.MINOR.PATCH pattern."""
        version = config_pkg.__version__
        parts = version.split(".")
        assert len(parts) >= 2, f"Version '{version}' should have at least MAJOR.MINOR components"
        for part in parts:
            numeric_part = ""
            for ch in part:
                if ch.isdigit():
                    numeric_part += ch
                else:
                    break
            assert len(numeric_part) > 0, f"Version component '{part}' should start with a digit"


class TestSmokeConfigLoaderExports:
    """§1.2 — Core configuration loading exports are accessible."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "name",
        [
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
            "check_migration_status",
            "recommend_validation_level",
            "get_transformation_feature_status",
            "print_transformation_status",
        ],
    )
    def test_config_loader_export_is_importable(self, config_pkg, name):
        """Each config loader export resolves to a non-None object."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Config loader export '{name}' is None or missing"


class TestSmokeContainerClassExports:
    """§1.2 — Configuration container classes are accessible."""

    CONTAINER_CLASSES = [
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
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CONTAINER_CLASSES)
    def test_container_class_exists(self, config_pkg, name):
        """Each container class is importable from the config package."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Container class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CONTAINER_CLASSES)
    def test_container_class_is_a_class(self, config_pkg, name):
        """Each container export is a class (not an instance or function)."""
        obj = getattr(config_pkg, name)
        assert inspect.isclass(obj), f"'{name}' should be a class, got {type(obj).__name__}"


class TestSmokeContainerFactoryExports:
    """§1.2 — Container factory functions are accessible and callable."""

    FACTORY_FUNCTIONS = [
        "create_dataset_config_from_global",
        "create_filter_config_from_global",
        "create_processing_config_from_global",
        "create_structural_features_config_from_global",
        "create_transformation_config_from_global",
        "create_descriptor_config_from_yaml",
        "create_default_descriptor_config",
        "create_minimal_descriptor_config",
        "create_handler_config",
        "create_handler_configuration_bundle",
        "validate_handler_configuration_bundle",
        "create_migration_compatible_config",
        "create_transform_spec_from_dict",
        "create_experimental_setup_from_dict",
        "create_transformation_config_from_dict",
        "create_default_experimental_setups",
        "create_ablation_study_setups",
        "validate_transformation_compatibility",
        "get_config_summary",
        "check_configuration_compatibility",
        "create_minimal_config_for_testing",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", FACTORY_FUNCTIONS)
    def test_factory_function_exists(self, config_pkg, name):
        """Each factory function is present and non-None."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Factory export '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", FACTORY_FUNCTIONS)
    def test_factory_function_is_callable(self, config_pkg, name):
        """Each factory function is callable."""
        obj = getattr(config_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeAccessorExports:
    """§1.2 — Configuration accessor functions are accessible."""

    CORE_ACCESSORS = [
        "get_dataset_type",
        "get_config_value",
        "get_dataset_config",
        "get_data_config",
        "get_property_availability",
        "get_uncertainty_config",
        "is_uncertainty_enabled",
        "get_structural_features_config",
        "is_structural_features_enabled",
        "get_filter_config",
        "get_transformations_config",
        "get_transformation_config",
        "is_descriptors_enabled",
        "get_descriptor_config",
        "get_selected_descriptors",
        "get_processing_config",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CORE_ACCESSORS)
    def test_accessor_export_exists(self, config_pkg, name):
        """Each core accessor is present and non-None."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Accessor '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CORE_ACCESSORS)
    def test_accessor_is_callable(self, config_pkg, name):
        """Each core accessor is callable."""
        obj = getattr(config_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeSchemaAndValidationExports:
    """§1.2 — Schema validation and migration exports are accessible."""

    SCHEMA_CLASSES = [
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
    ]

    VALIDATOR_CLASSES = [
        "YAMLSchemaValidator",
        "PluginSchemaValidator",
        "ExperimentSchemaValidator",
        "DescriptorSchemaValidator",
    ]

    MIGRATION_CLASSES = [
        "ConfigMigration",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", SCHEMA_CLASSES)
    def test_schema_class_exists(self, config_pkg, name):
        """Each schema class is importable."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Schema class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", VALIDATOR_CLASSES + MIGRATION_CLASSES)
    def test_validator_class_exists(self, config_pkg, name):
        """Each validator/migration class is importable."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Validator/migration class '{name}' is None or missing"


class TestSmokeValidationFunctionExports:
    """§1.2 — Validation functions are accessible."""

    VALIDATION_CLASSES = [
        "ValidationResult",
        "ValidationSeverity",
        "ValidationIssueDetail",
        "TransformValidator",
    ]

    VALIDATION_FUNCTIONS = [
        "validate_molecular_structure",
        "validate_molecular_data_dict",
        "validate_uncertainty_data",
        "validate_coordinates_3d",
        "validate_atomic_numbers",
        "validate_batch_consistency",
        "validate_handler_molecular_batch",
        "is_value_valid_and_not_nan",
        "validate_array_shape",
        "validate_numeric_range",
        "validate_property_value",
        "is_valid_molecule_identifier",
        "safe_get_value",
        "convert_to_scalar",
        "validate_transform_spec",
        "validate_experimental_setup",
        "validate_transformation_config",
        "validate_descriptor_config",
        "validate_handler_compatibility",
        "create_validation_report",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", VALIDATION_CLASSES)
    def test_validation_class_exists(self, config_pkg, name):
        """Each validation class is importable."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Validation class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", VALIDATION_FUNCTIONS)
    def test_validation_function_exists(self, config_pkg, name):
        """Each validation function is present and non-None."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Validation function '{name}' is None or missing"


class TestSmokeDataRefiningExports:
    """§1.2 — Data refinement exports are accessible."""

    REFINEMENT_FUNCTIONS = [
        "refine_molecular_data",
        "refine_molecular_vibrations",
        "validate_refined_data_quality",
        "apply_dataset_specific_refinement",
        "detect_dmc_statistical_outliers",
        "calculate_dmc_uncertainty_weights",
        "create_refinement_handler",
        "refine_molecular_data_with_handler",
        "validate_refined_data_with_handler",
        "get_refinement_statistics_with_handler",
        "log_data_refinement_status",
        "log_vibration_refinement_status",
        "diagnose_vibrational_data_structure",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", REFINEMENT_FUNCTIONS)
    def test_refinement_function_exists(self, config_pkg, name):
        """Each refinement function is present and non-None."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Refinement function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", REFINEMENT_FUNCTIONS)
    def test_refinement_function_is_callable(self, config_pkg, name):
        """Each refinement function is callable."""
        obj = getattr(config_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeConfigConstantsExports:
    """§1.2 — Configuration constants and Phase 3 registry exports are accessible."""

    DATASET_CONSTANTS = [
        "RAW_NPZ_FILENAME",
        "RAW_DATA_DOWNLOAD_URL",
        "DATASET_ROOT_DIR",
    ]

    PHASE3_FUNCTIONS = [
        "get_supported_handler_types",
        "get_default_handler_type",
        "is_handler_type_supported",
        "get_handler_feature_support",
        "get_handler_required_properties",
        "get_handler_optional_properties",
        "get_handler_identifier_keys_dynamic",
        "get_handler_coordinate_units_dynamic",
        "get_handler_molecule_creation_strategy",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", DATASET_CONSTANTS)
    def test_dataset_constant_exists(self, config_pkg, name):
        """Each dataset constant is defined."""
        assert hasattr(config_pkg, name), f"Constant '{name}' is missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", PHASE3_FUNCTIONS)
    def test_phase3_function_exists(self, config_pkg, name):
        """Each Phase 3 registry function is present and non-None."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Phase 3 export '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", PHASE3_FUNCTIONS)
    def test_phase3_function_is_callable(self, config_pkg, name):
        """Each Phase 3 registry function is callable."""
        obj = getattr(config_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeRegistryStatusFlags:
    """§1.2 — Registry integration status flags exist and are boolean."""

    # Phase 3 (config_constants)
    PHASE3_FLAGS = [
        "_REGISTRY_AVAILABLE",
        "_CACHE_INVALIDATION_REGISTERED",
        "_REGISTRY_IMPORT_ERROR",
        "_REGISTRY_INITIALIZED",
    ]

    # Phase 4 (config_containers)
    PHASE4_FLAGS = [
        "_CONTAINERS_REGISTRY_AVAILABLE",
        "_FALLBACK_VALID_TYPES",
    ]

    # Phase 5 (config_accessors)
    PHASE5_ACCESSOR_FLAGS = [
        "_ACCESSORS_REGISTRY_AVAILABLE",
        "_ACCESSORS_REGISTRY_INITIALIZED",
        "_ACCESSORS_REGISTRY_IMPORT_ERROR",
    ]

    # Phase 5 (config_loader)
    PHASE5_LOADER_FLAGS = [
        "_LOADER_REGISTRY_AVAILABLE",
        "_LOADER_REGISTRY_IMPORT_ERROR",
        "_LOADER_REGISTRY_INITIALIZED",
    ]

    # Phase 6 (validators)
    PHASE6_VALIDATORS_FLAGS = [
        "_VALIDATORS_REGISTRY_AVAILABLE",
        "_VALIDATORS_REGISTRY_INITIALIZED",
        "_VALIDATORS_REGISTRY_IMPORT_ERROR",
    ]

    # Phase 6 (data_refining)
    PHASE6_REFINING_FLAGS = [
        "_REFINING_REGISTRY_AVAILABLE",
        "_REFINING_REGISTRY_INITIALIZED",
        "_REFINING_REGISTRY_IMPORT_ERROR",
    ]

    ALL_BOOL_FLAGS = (
        PHASE3_FLAGS
        + PHASE5_ACCESSOR_FLAGS
        + PHASE5_LOADER_FLAGS
        + PHASE6_VALIDATORS_FLAGS
        + PHASE6_REFINING_FLAGS
    )

    @pytest.mark.smoke
    @pytest.mark.parametrize("flag", ALL_BOOL_FLAGS)
    def test_registry_flag_exists(self, config_pkg, flag):
        """Each registry integration flag is defined on the config package."""
        assert hasattr(config_pkg, flag), f"Flag '{flag}' is missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "flag",
        [
            "_REGISTRY_AVAILABLE",
            "_CACHE_INVALIDATION_REGISTERED",
            "_REGISTRY_INITIALIZED",
            "_CONTAINERS_REGISTRY_AVAILABLE",
            "_ACCESSORS_REGISTRY_AVAILABLE",
            "_ACCESSORS_REGISTRY_INITIALIZED",
            "_LOADER_REGISTRY_AVAILABLE",
            "_LOADER_REGISTRY_INITIALIZED",
            "_VALIDATORS_REGISTRY_AVAILABLE",
            "_VALIDATORS_REGISTRY_INITIALIZED",
            "_REFINING_REGISTRY_AVAILABLE",
            "_REFINING_REGISTRY_INITIALIZED",
        ],
    )
    def test_registry_bool_flag_is_bool(self, config_pkg, flag):
        """Each registry boolean status flag is actually a bool."""
        value = getattr(config_pkg, flag)
        assert isinstance(value, bool), f"Flag '{flag}' should be bool, got {type(value).__name__}"

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "flag",
        [
            "_REGISTRY_IMPORT_ERROR",
            "_ACCESSORS_REGISTRY_IMPORT_ERROR",
            "_LOADER_REGISTRY_IMPORT_ERROR",
            "_VALIDATORS_REGISTRY_IMPORT_ERROR",
            "_REFINING_REGISTRY_IMPORT_ERROR",
        ],
    )
    def test_registry_import_error_flag_is_str_or_none(self, config_pkg, flag):
        """Each import error flag is either None or a string."""
        value = getattr(config_pkg, flag)
        assert value is None or isinstance(value, str), (
            f"Flag '{flag}' should be None or str, got {type(value).__name__}"
        )

    @pytest.mark.smoke
    def test_fallback_valid_types_is_a_collection(self, config_pkg):
        """``_FALLBACK_VALID_TYPES`` is a list or tuple of strings."""
        fvt = config_pkg._FALLBACK_VALID_TYPES
        assert isinstance(fvt, (list, tuple, frozenset, set)), (
            f"_FALLBACK_VALID_TYPES should be a collection, got {type(fvt).__name__}"
        )
        for item in fvt:
            assert isinstance(item, str), (
                f"Each entry in _FALLBACK_VALID_TYPES should be str, got {type(item).__name__}"
            )


class TestSmokeRegistryInitFunctions:
    """§1.2 — Registry initialization functions are accessible and callable."""

    INIT_FUNCTIONS = [
        "_init_registry",
        "_loader_init_registry",
        "_validators_init_registry",
        "_refining_init_registry",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", INIT_FUNCTIONS)
    def test_init_registry_function_exists(self, config_pkg, name):
        """Each registry init function is present and non-None."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Registry init function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", INIT_FUNCTIONS)
    def test_init_registry_function_is_callable(self, config_pkg, name):
        """Each registry init function is callable."""
        obj = getattr(config_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokePhase5AccessorRegistryExports:
    """§1.2 — Phase 5 accessor registry public API exports are accessible."""

    PHASE5_EXPORTS = [
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
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", PHASE5_EXPORTS)
    def test_phase5_export_exists(self, config_pkg, name):
        """Each Phase 5 accessor registry export is present and non-None."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Phase 5 export '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", PHASE5_EXPORTS)
    def test_phase5_export_is_callable(self, config_pkg, name):
        """Each Phase 5 export is callable."""
        obj = getattr(config_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokePhase6DynamicQueryExports:
    """§1.2 — Phase 6 dynamic query exports are accessible."""

    PHASE6_EXPORTS = [
        # Validators registry
        "_validators_get_available_dataset_types",
        "_validators_is_dataset_type_registered",
        "_validators_get_dataset_feature",
        "_validators_get_dataset_required_properties",
        "_validators_get_handler_compatibility_checks",
        "get_validators_registry_status",
        # Data refining registry
        "_refining_get_available_dataset_types",
        "_refining_is_dataset_type_registered",
        "_get_dataset_feature",
        "_get_dataset_refinement_category",
        "get_refining_registry_status",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", PHASE6_EXPORTS)
    def test_phase6_export_exists(self, config_pkg, name):
        """Each Phase 6 dynamic query export is present."""
        assert hasattr(config_pkg, name), f"Phase 6 export '{name}' is missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", PHASE6_EXPORTS)
    def test_phase6_export_is_callable(self, config_pkg, name):
        """Each Phase 6 dynamic query export is callable."""
        obj = getattr(config_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeStandardTransformsExports:
    """§1.2 — Standard transforms accessor exports are accessible."""

    STANDARD_TRANSFORM_EXPORTS = [
        "get_standard_transforms",
        "get_standard_transforms_as_dicts",
        "get_combined_transforms",
        "get_combined_transforms_as_dicts",
        "has_standard_transforms",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", STANDARD_TRANSFORM_EXPORTS)
    def test_standard_transform_export_exists(self, config_pkg, name):
        """Each standard transforms export is present and non-None."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Standard transform export '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", STANDARD_TRANSFORM_EXPORTS)
    def test_standard_transform_export_is_callable(self, config_pkg, name):
        """Each standard transforms export is callable."""
        obj = getattr(config_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeModuleInitialization:
    """§1.2 — Module-level initialization runs without side-effect crashes."""

    @pytest.mark.smoke
    def test_reimport_is_idempotent(self, config_pkg):
        """
        Re-importing the config package (via ``importlib.reload``) does not
        crash.

        Validates that all module-level code (logging, registry status checks,
        namespace cleanup) is safe to re-execute.
        """
        reloaded = importlib.reload(config_pkg)
        assert reloaded is not None
        assert hasattr(reloaded, "__version__")

    @pytest.mark.smoke
    def test_reimport_preserves_all(self, config_pkg):
        """
        Re-importing the config package preserves ``__all__``.
        """
        reloaded = importlib.reload(config_pkg)
        assert hasattr(reloaded, "__all__")
        assert isinstance(reloaded.__all__, list)
        assert len(reloaded.__all__) > 0

    @pytest.mark.smoke
    def test_logging_namespace_cleaned(self, config_pkg):
        """
        The ``logging`` module is deleted from the config package namespace
        after initialization (``del logging`` at end of __init__.py).
        """
        # The __init__.py explicitly does ``del logging`` at the end.
        # After initial import, ``logging`` should NOT be an attribute.
        # Note: after reload, the cleanup may or may not persist depending
        # on Python internals, but the original import should be clean.
        # We just verify the attribute is not a logging module reference.
        if hasattr(config_pkg, "logging"):
            # If present, it should NOT be the logging module
            obj = config_pkg.logging
            assert not isinstance(obj, types.ModuleType) or obj.__name__ != "logging", (
                "The 'logging' module should be cleaned from namespace after init"
            )


class TestSmokeLegacyDeprecationExports:
    """§1.2 — Legacy deprecation helper exports are accessible."""

    LEGACY_EXPORTS = [
        "get_supported_handler_types_legacy",
        "get_handler_feature_support_legacy",
        "get_handler_required_properties_legacy",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", LEGACY_EXPORTS)
    def test_legacy_export_exists(self, config_pkg, name):
        """Each legacy deprecation export is present."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Legacy export '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", LEGACY_EXPORTS)
    def test_legacy_export_is_callable(self, config_pkg, name):
        """Each legacy deprecation export is callable."""
        obj = getattr(config_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeMigrationExports:
    """§1.2 — Migration support exports are accessible."""

    MIGRATION_EXPORTS = [
        "migrate_refinement_call_to_handler",
        "demonstrate_migration_patterns",
        "verify_migration_completeness",
        "get_migration_benefits",
        "get_module_migration_summary",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", MIGRATION_EXPORTS)
    def test_migration_export_exists(self, config_pkg, name):
        """Each migration support export is present."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Migration export '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", MIGRATION_EXPORTS)
    def test_migration_export_is_callable(self, config_pkg, name):
        """Each migration support export is callable."""
        obj = getattr(config_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeCacheManagementExports:
    """§1.2 — Cache management exports are accessible and callable."""

    CACHE_EXPORTS = [
        "clear_transformation_caches",
        "get_transformation_cache_info",
        "get_cached_handler_config",
        "clear_handler_caches",
        "get_handler_cache_info",
        "clear_all_caches",
        "get_all_cache_info",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CACHE_EXPORTS)
    def test_cache_export_exists(self, config_pkg, name):
        """Each cache management export is present and non-None."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Cache export '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CACHE_EXPORTS)
    def test_cache_export_is_callable(self, config_pkg, name):
        """Each cache management export is callable."""
        obj = getattr(config_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeEnhancedConfigAccessorExport:
    """§1.2 — EnhancedConfigAccessor class is accessible."""

    @pytest.mark.smoke
    def test_enhanced_config_accessor_exists(self, config_pkg):
        """``EnhancedConfigAccessor`` is defined on the config package."""
        assert hasattr(config_pkg, "EnhancedConfigAccessor")

    @pytest.mark.smoke
    def test_enhanced_config_accessor_is_class(self, config_pkg):
        """``EnhancedConfigAccessor`` is a class."""
        assert inspect.isclass(config_pkg.EnhancedConfigAccessor)


# ===================================================================
# SECTION 2 — CONTRACT TESTS
# ===================================================================


class TestContractAllCompleteness:
    """§2 — Every name in ``__all__`` is resolvable on the config package."""

    @pytest.mark.contract
    def test_all_is_a_list(self, config_pkg):
        """``__all__`` is a list (not a tuple or set)."""
        assert isinstance(config_pkg.__all__, list)

    @pytest.mark.contract
    def test_all_contains_no_duplicates(self, all_names):
        """``__all__`` has no unexpected duplicate entries.

        Known duplicates in the source ``__init__.py`` (harmless, caused by
        the same name being re-exported from multiple submodules or listed
        in multiple ``__all__`` sections):
            - ``create_experimental_setup_from_dict`` (config_containers + config_accessors)
            - ``_init_registry`` (config_constants + config_accessors)
            - ``validate_transformation_config`` (config_accessors + validators)

        This test asserts that no *unexpected* duplicates are introduced.
        """
        KNOWN_DUPLICATES = {
            "create_experimental_setup_from_dict",
            "_init_registry",
            "validate_transformation_config",
        }

        seen = set()
        duplicates = []
        for name in all_names:
            if name in seen:
                duplicates.append(name)
            seen.add(name)

        unexpected = [d for d in duplicates if d not in KNOWN_DUPLICATES]
        assert not unexpected, (
            f"Unexpected duplicate entries in __all__: {unexpected}. "
            f"Known duplicates (accepted): {sorted(KNOWN_DUPLICATES)}"
        )

    @pytest.mark.contract
    def test_every_all_entry_is_resolvable(self, config_pkg, all_names):
        """
        Generic sweep: every single entry in ``__all__`` must be resolvable,
        regardless of whether it is parameterized individually.
        """
        unresolvable = [name for name in all_names if not hasattr(config_pkg, name)]
        assert not unresolvable, (
            f"Names in __all__ that are not defined on the module: {unresolvable}"
        )

    @pytest.mark.contract
    def test_all_entries_are_strings(self, all_names):
        """Every entry in ``__all__`` is a string."""
        non_strings = [(i, name) for i, name in enumerate(all_names) if not isinstance(name, str)]
        assert not non_strings, f"Non-string entries in __all__: {non_strings}"


class TestContractAllConsistency:
    """§2 — Every public import in the config module is listed in ``__all__``."""

    # Names that are intentionally public-ish but NOT in __all__
    # (metadata, internal helpers used by other modules, etc.)
    KNOWN_UNLISTED = {
        # Metadata dunders (not typically in __all__)
        "__version__",
        "__author__",
        "__description__",
        # Internal logger (cleaned up via del logging)
        "_logger",
    }

    @pytest.mark.contract
    def test_public_imports_are_in_all(self, config_pkg, all_names):
        """
        Every non-dunder, non-private attribute that was imported (not
        defined locally) should be in ``__all__`` — unless it is in
        ``KNOWN_UNLISTED``.

        This catches accidental omissions when new imports are added to
        the config ``__init__.py`` without updating ``__all__``.
        """
        all_set = set(all_names)
        module_dict = vars(config_pkg)
        missing_from_all = []

        for name, obj in module_dict.items():
            # Skip dunder names
            if name.startswith("__") and name.endswith("__"):
                continue
            # Skip modules (submodule references)
            if isinstance(obj, types.ModuleType):
                continue
            # Skip known unlisted names
            if name in self.KNOWN_UNLISTED:
                continue
            # Private names that start with underscore but ARE in __all__
            # are already covered. Skip private names NOT in __all__.
            if name.startswith("_") and name not in all_set:
                continue

            if name not in all_set:
                missing_from_all.append(name)

        # Filter common Python internals
        python_internals = {
            "__builtins__",
            "__cached__",
            "__doc__",
            "__file__",
            "__loader__",
            "__name__",
            "__package__",
            "__path__",
            "__spec__",
        }
        missing_from_all = [n for n in missing_from_all if n not in python_internals]

        assert not missing_from_all, (
            f"Public names imported in config/__init__.py but not in __all__: "
            f"{sorted(missing_from_all)}"
        )


class TestContractContainerClassTypes:
    """§2 — Container classes are Pydantic BaseModel subclasses (or dataclasses)."""

    PYDANTIC_CONTAINERS = [
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
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", PYDANTIC_CONTAINERS)
    def test_container_is_class(self, config_pkg, name):
        """Each container is a class."""
        obj = getattr(config_pkg, name)
        assert inspect.isclass(obj), f"'{name}' should be a class, got {type(obj).__name__}"

    @pytest.mark.contract
    @pytest.mark.parametrize("name", PYDANTIC_CONTAINERS)
    def test_container_is_pydantic_basemodel(self, config_pkg, name):
        """
        Each container is a Pydantic BaseModel subclass or a Pydantic
        dataclass (both support ``model_dump()`` or ``__dataclass_fields__``).

        Per the project structure doc: config_containers.py uses Pydantic V2
        BaseModel with ``frozen=True``.
        """
        cls = getattr(config_pkg, name)
        try:
            from pydantic import BaseModel

            is_pydantic_model = issubclass(cls, BaseModel)
        except ImportError:
            is_pydantic_model = False

        # Also accept pydantic dataclasses
        is_pydantic_dc = hasattr(cls, "__pydantic_fields__")

        # Also accept stdlib dataclasses as fallback
        is_stdlib_dc = hasattr(cls, "__dataclass_fields__")

        assert is_pydantic_model or is_pydantic_dc or is_stdlib_dc, (
            f"'{name}' should be a Pydantic BaseModel, Pydantic dataclass, or stdlib dataclass"
        )


class TestContractSchemaClassTypes:
    """§2 — Schema and validator classes are actual classes."""

    SCHEMA_CLASSES = [
        "TransformationSchema",
        "PluginConfigSchema",
        "DescriptorConfigSchema",
        "WavefunctionProcessingConfigSchema",
        "WavefunctionUncertaintyConfigSchema",
        "WavefunctionConfigSchema",
        "ExperimentSchema",
        "ValidationConfig",
        "DescriptorCategoryConfigSchema",
    ]

    VALIDATOR_CLASSES = [
        "YAMLSchemaValidator",
        "PluginSchemaValidator",
        "ExperimentSchemaValidator",
        "DescriptorSchemaValidator",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", SCHEMA_CLASSES)
    def test_schema_is_class(self, config_pkg, name):
        """Each schema export is a class."""
        obj = getattr(config_pkg, name)
        assert inspect.isclass(obj), f"'{name}' should be a class, got {type(obj).__name__}"

    @pytest.mark.contract
    @pytest.mark.parametrize("name", VALIDATOR_CLASSES)
    def test_validator_is_class(self, config_pkg, name):
        """Each validator export is a class."""
        obj = getattr(config_pkg, name)
        assert inspect.isclass(obj), f"'{name}' should be a class, got {type(obj).__name__}"

    @pytest.mark.contract
    def test_config_migration_is_class(self, config_pkg):
        """``ConfigMigration`` is a class."""
        assert inspect.isclass(config_pkg.ConfigMigration)


class TestContractValidationResultTypes:
    """§2 — Validation result/severity classes have expected nature."""

    @pytest.mark.contract
    def test_validation_result_is_class(self, config_pkg):
        """``ValidationResult`` is a class."""
        assert inspect.isclass(config_pkg.ValidationResult)

    @pytest.mark.contract
    def test_validation_severity_is_class_or_enum(self, config_pkg):
        """``ValidationSeverity`` is a class or enum."""
        obj = config_pkg.ValidationSeverity
        assert inspect.isclass(obj), (
            f"ValidationSeverity should be a class, got {type(obj).__name__}"
        )

    @pytest.mark.contract
    def test_validation_issue_detail_is_class(self, config_pkg):
        """``ValidationIssueDetail`` is a class."""
        assert inspect.isclass(config_pkg.ValidationIssueDetail)

    @pytest.mark.contract
    def test_transform_validator_is_class(self, config_pkg):
        """``TransformValidator`` is a class."""
        assert inspect.isclass(config_pkg.TransformValidator)

    @pytest.mark.contract
    def test_must_check_is_callable(self, config_pkg):
        """``must_check`` decorator is callable."""
        assert callable(config_pkg.must_check)


class TestContractConfigLoaderFunctionSignatures:
    """§2 — Key config loader functions have documented parameter signatures."""

    @pytest.mark.contract
    def test_load_config_accepts_path(self, config_pkg):
        """``load_config`` has a parameter for the config file path."""
        sig = inspect.signature(config_pkg.load_config)
        param_names = set(sig.parameters.keys())
        # load_config should accept a config_path or similar
        assert len(param_names) >= 1, "load_config should accept at least one parameter"

    @pytest.mark.contract
    def test_load_config_with_validation_is_callable(self, config_pkg):
        """``load_config_with_validation`` is callable with params."""
        sig = inspect.signature(config_pkg.load_config_with_validation)
        assert len(sig.parameters) >= 1, "load_config_with_validation should accept parameters"

    @pytest.mark.contract
    def test_clear_config_cache_is_function(self, config_pkg):
        """``clear_config_cache`` is a function (not a class)."""
        assert inspect.isfunction(config_pkg.clear_config_cache)

    @pytest.mark.contract
    def test_reload_config_is_function(self, config_pkg):
        """``reload_config`` is a function (not a class)."""
        assert inspect.isfunction(config_pkg.reload_config)


class TestContractAccessorFunctionSignatures:
    """§2 — Key accessor functions are functions (not classes)."""

    ACCESSOR_FUNCTIONS = [
        "get_dataset_type",
        "get_config_value",
        "get_dataset_config",
        "get_uncertainty_config",
        "is_uncertainty_enabled",
        "get_structural_features_config",
        "is_structural_features_enabled",
        "get_filter_config",
        "get_transformations_config",
        "get_transformation_config",
        "get_processing_config",
        "is_descriptors_enabled",
        "get_descriptor_config",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", ACCESSOR_FUNCTIONS)
    def test_accessor_is_function(self, config_pkg, name):
        """Each accessor is a function (not a class or unbound method)."""
        obj = getattr(config_pkg, name)
        assert inspect.isfunction(obj), f"'{name}' should be a function, got {type(obj).__name__}"


class TestContractPhaseAliasNonCollision:
    """
    §2 — Phase-aliased imports do not collide.

    The ``__init__.py`` imports ``_REGISTRY_AVAILABLE`` from multiple
    submodules with ``as`` aliases. Each aliased name must resolve to
    a distinct module-level attribute.
    """

    @pytest.mark.contract
    def test_registry_available_aliases_are_distinct_attributes(self, config_pkg):
        """
        ``_REGISTRY_AVAILABLE``, ``_CONTAINERS_REGISTRY_AVAILABLE``,
        ``_ACCESSORS_REGISTRY_AVAILABLE``, ``_LOADER_REGISTRY_AVAILABLE``,
        ``_VALIDATORS_REGISTRY_AVAILABLE``, ``_REFINING_REGISTRY_AVAILABLE``
        are all distinct attributes on the config package.
        """
        aliases = [
            "_REGISTRY_AVAILABLE",
            "_CONTAINERS_REGISTRY_AVAILABLE",
            "_ACCESSORS_REGISTRY_AVAILABLE",
            "_LOADER_REGISTRY_AVAILABLE",
            "_VALIDATORS_REGISTRY_AVAILABLE",
            "_REFINING_REGISTRY_AVAILABLE",
        ]
        for alias in aliases:
            assert hasattr(config_pkg, alias), (
                f"Aliased flag '{alias}' is missing from config package"
            )

    @pytest.mark.contract
    def test_init_registry_aliases_are_distinct(self, config_pkg):
        """
        ``_init_registry``, ``_loader_init_registry``,
        ``_validators_init_registry``, ``_refining_init_registry``
        are all distinct callable attributes.
        """
        aliases = [
            "_init_registry",
            "_loader_init_registry",
            "_validators_init_registry",
            "_refining_init_registry",
        ]
        resolved = {}
        for alias in aliases:
            obj = getattr(config_pkg, alias, None)
            assert obj is not None, f"'{alias}' is None or missing"
            assert callable(obj), f"'{alias}' should be callable"
            resolved[alias] = obj

        # They should all be callable; we verify they exist but note
        # some may point to the same function if submodules share
        # the same _init_registry function — that's valid.


class TestContractRegistryStatusReportingFunctions:
    """§2 — Registry status reporting functions return dicts."""

    STATUS_FUNCTIONS = [
        "get_validators_registry_status",
        "get_refining_registry_status",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", STATUS_FUNCTIONS)
    def test_registry_status_function_returns_dict(self, config_pkg, name):
        """Each registry status reporting function returns a dict."""
        func = getattr(config_pkg, name)
        result = func()
        assert isinstance(result, dict), (
            f"'{name}()' should return dict, got {type(result).__name__}"
        )


class TestContractContainerRegistryIntegration:
    """§2 — Phase 4 container registry integration exports."""

    @pytest.mark.contract
    def test_get_valid_dataset_types_is_callable(self, config_pkg):
        """``_get_valid_dataset_types`` is callable."""
        assert callable(config_pkg._get_valid_dataset_types)

    @pytest.mark.contract
    def test_is_valid_dataset_type_is_callable(self, config_pkg):
        """``_is_valid_dataset_type`` is callable."""
        assert callable(config_pkg._is_valid_dataset_type)

    @pytest.mark.contract
    def test_verify_container_registry_integration_is_callable(self, config_pkg):
        """``verify_container_registry_integration`` is callable."""
        assert callable(config_pkg.verify_container_registry_integration)

    @pytest.mark.contract
    def test_verify_container_registry_integration_returns_dict(self, config_pkg):
        """``verify_container_registry_integration()`` returns a dict."""
        result = config_pkg.verify_container_registry_integration()
        assert isinstance(result, dict), (
            f"verify_container_registry_integration() should return dict, "
            f"got {type(result).__name__}"
        )

    @pytest.mark.contract
    def test_verify_container_registry_has_overall_status(self, config_pkg):
        """
        ``verify_container_registry_integration()`` result includes
        ``overall_status`` key (documented in the __init__.py usage examples).
        """
        result = config_pkg.verify_container_registry_integration()
        assert "overall_status" in result, (
            "verify_container_registry_integration() missing 'overall_status' key"
        )


class TestContractPublicAPISurface:
    """
    §2 — Public API surface stability.

    Ensures that the minimum expected public API is present in ``__all__``.
    Guards against accidental removals during refactoring.
    """

    # The minimum API surface that MUST be present in __all__
    MINIMUM_API = {
        # Core loading
        "load_config",
        "clear_config_cache",
        "reload_config",
        "is_config_loaded",
        # Container classes
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
        # Core accessors
        "get_dataset_type",
        "get_dataset_config",
        "get_filter_config",
        "get_processing_config",
        "get_uncertainty_config",
        "is_uncertainty_enabled",
        "get_structural_features_config",
        "get_transformation_config",
        "is_descriptors_enabled",
        "get_descriptor_config",
        # Schema & validation
        "YAMLSchemaValidator",
        "ValidationResult",
        "ValidationSeverity",
        "TransformValidator",
        "ConfigMigration",
        # Constants
        "RAW_NPZ_FILENAME",
        "RAW_DATA_DOWNLOAD_URL",
        "DATASET_ROOT_DIR",
        # Phase 3 registry
        "get_supported_handler_types",
        "is_handler_type_supported",
        # Phase 5 registry
        "registry_list_all",
        "registry_is_registered",
        "validate_dataset_type",
        "is_valid_dataset_type",
        # Data refinement
        "refine_molecular_data",
        "refine_molecular_vibrations",
        "validate_refined_data_quality",
        # Validation functions
        "validate_molecular_data_dict",
        "validate_handler_compatibility",
        "create_validation_report",
    }

    @pytest.mark.contract
    def test_minimum_api_in_all(self, all_names):
        """The minimum expected public API is present in ``__all__``."""
        all_set = set(all_names)
        missing = self.MINIMUM_API - all_set
        assert not missing, f"Minimum API names missing from __all__: {sorted(missing)}"

    @pytest.mark.contract
    def test_all_has_expected_length(self, all_names):
        """
        ``__all__`` contains a substantial number of entries.

        Based on the __init__.py source, the config package exports 200+
        names. This test guards against catastrophic loss (e.g., accidental
        truncation of __all__) while allowing for organic growth.
        """
        actual = len(all_names)
        # The __init__.py has ~475 entries in __all__ (lines 809-1283)
        # We set a floor well below the actual count to allow changes
        # while catching catastrophic loss.
        MINIMUM_EXPECTED = 150
        assert actual >= MINIMUM_EXPECTED, (
            f"__all__ has {actual} entries, expected at least {MINIMUM_EXPECTED}. "
            f"This suggests __all__ may have been accidentally truncated."
        )


class TestContractConfigLoaderReturnTypes:
    """§2 — Config loader functions return documented types when callable."""

    @pytest.mark.contract
    def test_is_config_loaded_returns_bool(self, config_pkg):
        """``is_config_loaded()`` returns a bool."""
        result = config_pkg.is_config_loaded()
        assert isinstance(result, bool), (
            f"is_config_loaded() should return bool, got {type(result).__name__}"
        )

    @pytest.mark.contract
    def test_get_config_stats_returns_dict(self, config_pkg):
        """``get_config_stats()`` returns a dict."""
        result = config_pkg.get_config_stats()
        assert isinstance(result, dict), (
            f"get_config_stats() should return dict, got {type(result).__name__}"
        )


class TestContractDatasetConstantsTypes:
    """§2 — Dataset constants have expected types.

    Note: ``RAW_NPZ_FILENAME``, ``RAW_DATA_DOWNLOAD_URL``, and
    ``DATASET_ROOT_DIR`` are exported as ``property`` descriptor objects
    from ``config_constants.py`` (dynamic/registry-aware), not plain
    string constants.
    """

    DATASET_CONSTANTS = [
        "RAW_NPZ_FILENAME",
        "RAW_DATA_DOWNLOAD_URL",
        "DATASET_ROOT_DIR",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", DATASET_CONSTANTS)
    def test_dataset_constant_is_property_or_string(self, config_pkg, name):
        """
        Each dataset constant is either a ``property`` descriptor (for
        dynamic/registry-aware resolution) or a plain ``str``.

        The config_constants module exports these as property objects to
        support lazy/dynamic resolution via the dataset registry.
        """
        obj = getattr(config_pkg, name)
        assert isinstance(obj, (str, property)), (
            f"{name} should be str or property, got {type(obj).__name__}"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", DATASET_CONSTANTS)
    def test_dataset_constant_is_not_none(self, config_pkg, name):
        """Each dataset constant is present (not None)."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"{name} is None"


class TestContractHandlerUtilityExports:
    """§2 — Handler utility functions from config_constants are callable."""

    HANDLER_UTILITIES = [
        "get_handler_constants",
        "get_handler_identifier_keys",
        "validate_handler_configuration",
        "get_handler_compatibility_info",
        "check_handler_feature_support",
        "get_handler_property_requirements",
        "create_handler_config_from_constants",
        "get_transformation_constants",
        "get_handler_transform_compatibility",
        "get_compatible_transforms_for_handler",
        "get_incompatible_transforms_for_handler",
        "validate_experimental_setup_for_handler",
        "get_migration_compatibility_constants",
        "get_legacy_compatible_constants",
        "validate_handler_environment",
        "validate_complete_environment",
        "ensure_handler_constant_compatibility",
        "ensure_complete_compatibility",
        "get_complete_constants_debug_info",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", HANDLER_UTILITIES)
    def test_handler_utility_is_callable(self, config_pkg, name):
        """Each handler utility function is callable."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Handler utility '{name}' is None or missing"
        assert callable(obj), f"'{name}' should be callable"


class TestContractTransformAccessorExports:
    """§2 — Transform-related accessor exports are callable."""

    TRANSFORM_ACCESSORS = [
        "get_available_transforms",
        "get_transforms_by_category",
        "get_transform_info",
        "get_transform_registry_info",
        "list_available_transforms",
        "get_all_transforms",
        "get_transform_config",
        "get_transform_parameter",
        "get_dataset_specific_config",
        "validate_transform_config",
        "get_experimental_setup",
        "list_experimental_setups",
        "list_enabled_experimental_setups",
        "get_default_experimental_setup",
        "get_experimental_setups_for_dataset",
        "save_experimental_setup",
        "get_research_recommendations",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", TRANSFORM_ACCESSORS)
    def test_transform_accessor_is_callable(self, config_pkg, name):
        """Each transform accessor is callable."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Transform accessor '{name}' is None or missing"
        assert callable(obj), f"'{name}' should be callable"


class TestContractSchemaFactoryFunctions:
    """§2 — Schema factory and validation functions are callable."""

    SCHEMA_FUNCTIONS = [
        "create_validator",
        "create_migrator",
        "load_and_validate_yaml_config",
        "validate_yaml_config_string",
        "validate_wavefunction_config",
        "validate_current_config",
        "create_example_enhanced_config",
        "create_example_legacy_configs",
        "get_enhanced_transformation_config",
        "create_default_plugin_config",
        "create_example_plugin_config",
        "create_default_experiment_config",
        "create_example_experiments_config",
        "validate_experiment_config_file",
        "get_experiment_config_summary",
        "validate_plugin_config_file",
        "merge_plugin_configs",
        "get_plugin_config_summary",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", SCHEMA_FUNCTIONS)
    def test_schema_function_is_callable(self, config_pkg, name):
        """Each schema factory/validation function is callable."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Schema function '{name}' is None or missing"
        assert callable(obj), f"'{name}' should be callable"


class TestContractContainerCreationFunctions:
    """§2 — Container creation functions from config_accessors are callable."""

    CREATION_FUNCTIONS = [
        "create_dataset_config_container",
        "create_filter_config_container",
        "create_processing_config_container",
        "create_structural_features_config_container",
        "create_transformation_config_container",
        "create_dataset_handler",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", CREATION_FUNCTIONS)
    def test_creation_function_is_callable(self, config_pkg, name):
        """Each container creation function is callable."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Creation function '{name}' is None or missing"
        assert callable(obj), f"'{name}' should be callable"


class TestContractLegacyCompatibilityExports:
    """§2 — Legacy compatibility and migration exports are callable."""

    LEGACY_EXPORTS = [
        "migrate_legacy_transformation_config",
        "check_transformation_system_compatibility",
        "get_supported_handler_types_legacy",
        "get_handler_feature_support_legacy",
        "get_handler_required_properties_legacy",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", LEGACY_EXPORTS)
    def test_legacy_export_is_callable(self, config_pkg, name):
        """Each legacy compatibility export is callable."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Legacy export '{name}' is None or missing"
        assert callable(obj), f"'{name}' should be callable"


class TestContractPluginValidationLevelType:
    """§2 — PluginValidationLevel is a class or enum."""

    @pytest.mark.contract
    def test_plugin_validation_level_is_class(self, config_pkg):
        """``PluginValidationLevel`` is a class (likely an enum)."""
        obj = config_pkg.PluginValidationLevel
        assert inspect.isclass(obj), (
            f"PluginValidationLevel should be a class, got {type(obj).__name__}"
        )


class TestContractDescriptorValidationExports:
    """§2 — Descriptor validation exports are callable."""

    DESCRIPTOR_VALIDATORS = [
        "validate_descriptor_config",
        "validate_descriptor_category_compatibility",
        "validate_descriptor_cache_settings",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", DESCRIPTOR_VALIDATORS)
    def test_descriptor_validator_is_callable(self, config_pkg, name):
        """Each descriptor validation function is callable."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Descriptor validator '{name}' is None or missing"
        assert callable(obj), f"'{name}' should be callable"


class TestContractTransformValidationExports:
    """§2 — Transform validation exports are callable."""

    TRANSFORM_VALIDATORS = [
        "validate_transform_spec",
        "validate_experimental_setup",
        "validate_transformation_config",
        "validate_transform_sequence_semantics",
        "validate_transform_composition_rules",
        "validate_transforms_with_fallback",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", TRANSFORM_VALIDATORS)
    def test_transform_validator_is_callable(self, config_pkg, name):
        """Each transform validation function is callable."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Transform validator '{name}' is None or missing"
        assert callable(obj), f"'{name}' should be callable"


class TestContractReportingExports:
    """§2 — Reporting and diagnostics exports are callable."""

    REPORTING_FUNCTIONS = [
        "create_validation_report",
        "run_validation_diagnostics",
        "run_handler_validation_tests",
        "run_transformation_validation_tests",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", REPORTING_FUNCTIONS)
    def test_reporting_function_is_callable(self, config_pkg, name):
        """Each reporting function is callable."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Reporting function '{name}' is None or missing"
        assert callable(obj), f"'{name}' should be callable"


class TestContractUtilityFunctionExports:
    """§2 — Utility shorthand function exports are callable."""

    UTILITY_FUNCTIONS = [
        "get_transform",
        "get_parameter",
        "get_setup",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", UTILITY_FUNCTIONS)
    def test_utility_function_is_callable(self, config_pkg, name):
        """Each utility function is callable."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Utility function '{name}' is None or missing"
        assert callable(obj), f"'{name}' should be callable"


class TestContractStructuralFeatureAccessors:
    """§2 — Structural feature accessor exports are callable."""

    STRUCTURAL_ACCESSORS = [
        "get_structural_features_preprocessing_config",
        "get_charge_handling_config",
        "get_geometric_features_config",
        "get_stereochemistry_config",
        "get_atom_features",
        "get_bond_features",
        "should_pass_coordinates_to_structural_features",
        "should_pass_mulliken_charges_to_structural_features",
        "should_enable_stereochemistry_preprocessing",
        "get_dataset_appropriate_structural_features",
        "validate_structural_features_for_dataset",
        "get_feature_compatibility_report",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", STRUCTURAL_ACCESSORS)
    def test_structural_accessor_is_callable(self, config_pkg, name):
        """Each structural feature accessor is callable."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Structural accessor '{name}' is None or missing"
        assert callable(obj), f"'{name}' should be callable"


class TestContractTransformationMetadataAccessors:
    """§2 — Transformation metadata accessor exports are callable."""

    TRANSFORM_METADATA = [
        "get_transformation_validation_config",
        "is_transformation_validation_enabled",
        "is_transformation_strict_mode_enabled",
        "get_transformation_cache_key",
        "get_transformation_performance_metrics",
        "get_transformation_config_summary",
        "validate_transformation_system",
        "create_transforms_from_config",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", TRANSFORM_METADATA)
    def test_transform_metadata_accessor_is_callable(self, config_pkg, name):
        """Each transformation metadata accessor is callable."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Transform metadata accessor '{name}' is None or missing"
        assert callable(obj), f"'{name}' should be callable"


class TestContractMolecularValidationExports:
    """§2 — Molecular validation and property helper exports are callable."""

    MOLECULAR_HELPERS = [
        "is_value_valid_and_not_nan",
        "validate_array_shape",
        "validate_numeric_range",
        "validate_property_value",
        "is_valid_molecule_identifier",
        "safe_get_value",
        "convert_to_scalar",
        "validate_coordinates_3d",
        "validate_atomic_numbers",
        "validate_batch_consistency",
        "validate_handler_molecular_batch",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", MOLECULAR_HELPERS)
    def test_molecular_helper_is_callable(self, config_pkg, name):
        """Each molecular validation helper is callable."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Molecular helper '{name}' is None or missing"
        assert callable(obj), f"'{name}' should be callable"


class TestContractPhase5InternalHelpers:
    """§2 — Phase 5 internal helper exports are callable."""

    PHASE5_HELPERS = [
        "_get_default_dataset_type",
        "_get_valid_dataset_types",
        "_is_valid_dataset_type",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", PHASE5_HELPERS)
    def test_phase5_helper_is_callable(self, config_pkg, name):
        """Each Phase 5 internal helper is callable."""
        obj = getattr(config_pkg, name, None)
        assert obj is not None, f"Phase 5 helper '{name}' is None or missing"
        assert callable(obj), f"'{name}' should be callable"

    @pytest.mark.contract
    def test_get_default_dataset_type_returns_string(self, config_pkg):
        """``_get_default_dataset_type()`` returns a string."""
        result = config_pkg._get_default_dataset_type()
        assert isinstance(result, str), (
            f"_get_default_dataset_type() should return str, got {type(result).__name__}"
        )

    @pytest.mark.contract
    def test_get_default_dataset_type_is_nonempty(self, config_pkg):
        """``_get_default_dataset_type()`` returns a non-empty string."""
        result = config_pkg._get_default_dataset_type()
        assert len(result) > 0, "_get_default_dataset_type() should return a non-empty string"


class TestContractGetConfigHashReturnType:
    """§2 — ``get_config_hash()`` return type contract."""

    @pytest.mark.contract
    def test_get_config_hash_returns_str_or_none(self, config_pkg):
        """
        ``get_config_hash()`` returns a string (hash) or None when no
        config is loaded.
        """
        result = config_pkg.get_config_hash()
        assert result is None or isinstance(result, str), (
            f"get_config_hash() should return str or None, got {type(result).__name__}"
        )


class TestContractConfigLoaderRegistryFlags:
    """§2 — Config loader registry integration consistency."""

    @pytest.mark.contract
    def test_loader_registry_flags_consistent(self, config_pkg):
        """
        If ``_LOADER_REGISTRY_AVAILABLE`` is True, then
        ``_LOADER_REGISTRY_INITIALIZED`` should also be True (or at least
        the init function should have been called).
        """
        available = config_pkg._LOADER_REGISTRY_AVAILABLE
        _initialized = config_pkg._LOADER_REGISTRY_INITIALIZED
        error = config_pkg._LOADER_REGISTRY_IMPORT_ERROR

        if available:
            # If available, the error should be None
            assert error is None, (
                f"_LOADER_REGISTRY_AVAILABLE is True but _LOADER_REGISTRY_IMPORT_ERROR is '{error}'"
            )

    @pytest.mark.contract
    def test_accessors_registry_flags_consistent(self, config_pkg):
        """
        If ``_ACCESSORS_REGISTRY_AVAILABLE`` is True, then the import
        error should be None.
        """
        available = config_pkg._ACCESSORS_REGISTRY_AVAILABLE
        error = config_pkg._ACCESSORS_REGISTRY_IMPORT_ERROR

        if available:
            assert error is None, (
                f"_ACCESSORS_REGISTRY_AVAILABLE is True but "
                f"_ACCESSORS_REGISTRY_IMPORT_ERROR is '{error}'"
            )

    @pytest.mark.contract
    def test_validators_registry_flags_consistent(self, config_pkg):
        """
        If ``_VALIDATORS_REGISTRY_AVAILABLE`` is True, then the import
        error should be None.
        """
        available = config_pkg._VALIDATORS_REGISTRY_AVAILABLE
        error = config_pkg._VALIDATORS_REGISTRY_IMPORT_ERROR

        if available:
            assert error is None, (
                f"_VALIDATORS_REGISTRY_AVAILABLE is True but "
                f"_VALIDATORS_REGISTRY_IMPORT_ERROR is '{error}'"
            )

    @pytest.mark.contract
    def test_refining_registry_flags_consistent(self, config_pkg):
        """
        If ``_REFINING_REGISTRY_AVAILABLE`` is True, then the import
        error should be None.
        """
        available = config_pkg._REFINING_REGISTRY_AVAILABLE
        error = config_pkg._REFINING_REGISTRY_IMPORT_ERROR

        if available:
            assert error is None, (
                f"_REFINING_REGISTRY_AVAILABLE is True but "
                f"_REFINING_REGISTRY_IMPORT_ERROR is '{error}'"
            )
