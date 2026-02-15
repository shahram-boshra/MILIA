# tests/test__init__milia_pipeline.py

"""
Test Suite: milia_pipeline/__init__.py — Smoke Tests & Contract Tests
=====================================================================

Production-ready test suite for the MILIA Pipeline root package ``__init__.py``.

Covers:
    Section 1 — Smoke Tests (MILIA_Test_Recommendations.md §1.2 scope):
        - All top-level package imports succeed without ImportError
        - Module-level attributes are accessible and correctly typed
        - Convenience functions execute without exceptions
        - Conditional import branches degrade gracefully
        - Module initialization (disable_verbose_third_party_logging) runs safely

    Section 2 — Contract Tests (MILIA_Test_Recommendations.md §2 scope):
        - ``__all__`` completeness: every name in ``__all__`` is resolvable
        - ``__all__`` consistency: every public import is listed in ``__all__``
        - Backward-compatibility aliases maintain documented contracts
        - Exception hierarchy contract: Tier-1 → Tier-2 → Tier-3 inheritance
        - Convenience function return-type contracts
        - Conditional availability flags contract (bool type, ternary semantics)
        - Public API surface stability (no accidental removals)

Design:
    - Zero ``sys.modules`` pollution: all mocking via ``@patch`` decorators
      or context-manager ``patch`` calls scoped to individual tests.
    - Deterministic: no filesystem, network, or GPU access required.
    - Isolated: each test is independent; execution order is irrelevant.
    - Fast: expected < 5 s total wall-clock on CI.

Launch:
    From project root (/app/milia):
        pytest tests/test__init__milia_pipeline.py -v --tb=short

Markers:
    smoke     — Quick health-check tests (§1)
    contract  — Interface/contract validation tests (§2)
"""

import importlib
import inspect
import logging
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure the project root is importable
# ---------------------------------------------------------------------------
# When launched via ``pytest tests/test__init__milia_pipeline.py`` from the
# project root (/app/milia), ``milia_pipeline`` must be on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # …/tests -> …/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture(scope="module")
def milia_pkg():
    """
    Import and return the ``milia_pipeline`` package once per module.

    This fixture validates the fundamental smoke invariant: the package
    is importable.  If this fails, every downstream test is moot.
    """
    try:
        import milia_pipeline

        return milia_pipeline
    except ImportError as exc:
        pytest.fail(
            f"milia_pipeline could not be imported — smoke test precondition violated: {exc}"
        )


@pytest.fixture(scope="module")
def all_names(milia_pkg):
    """Return the ``__all__`` list from the package."""
    assert hasattr(milia_pkg, "__all__"), "milia_pipeline.__all__ is missing — contract violation"
    return list(milia_pkg.__all__)


# ===================================================================
# SECTION 1 — SMOKE TESTS
# ===================================================================


class TestSmokePackageImport:
    """§1.2 — Verify the top-level package imports without errors."""

    @pytest.mark.smoke
    def test_import_milia_pipeline_succeeds(self, milia_pkg):
        """The root package imports without raising any exception."""
        assert milia_pkg is not None

    @pytest.mark.smoke
    def test_package_is_a_module(self, milia_pkg):
        """The imported object is a proper Python module."""
        assert isinstance(milia_pkg, types.ModuleType)

    @pytest.mark.smoke
    def test_package_has_file_attribute(self, milia_pkg):
        """The package exposes a ``__file__`` attribute (not a namespace pkg)."""
        assert hasattr(milia_pkg, "__file__")


class TestSmokeMetadataAttributes:
    """§1.2 — Verify module-level metadata attributes are present and typed."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "attr",
        [
            "__version__",
            "__author__",
            "__license__",
            "__maintainer__",
            "__status__",
        ],
    )
    def test_metadata_attribute_exists(self, milia_pkg, attr):
        """Each metadata dunder is defined on the package."""
        assert hasattr(milia_pkg, attr), f"Missing attribute: {attr}"

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "attr",
        [
            "__version__",
            "__author__",
            "__license__",
            "__maintainer__",
            "__status__",
        ],
    )
    def test_metadata_attribute_is_string(self, milia_pkg, attr):
        """Each metadata dunder is a non-empty string."""
        value = getattr(milia_pkg, attr)
        assert isinstance(value, str), f"{attr} should be str, got {type(value)}"
        assert len(value) > 0, f"{attr} should be non-empty"

    @pytest.mark.smoke
    def test_version_is_semver_like(self, milia_pkg):
        """``__version__`` follows a MAJOR.MINOR.PATCH pattern."""
        version = milia_pkg.__version__
        parts = version.split(".")
        assert len(parts) >= 2, f"Version '{version}' should have at least MAJOR.MINOR components"
        # Each part should be numeric (allows pre-release suffixes like 1.1.0rc1)
        for part in parts:
            numeric_part = ""
            for ch in part:
                if ch.isdigit():
                    numeric_part += ch
                else:
                    break
            assert len(numeric_part) > 0, f"Version component '{part}' should start with a digit"

    @pytest.mark.smoke
    def test_status_is_recognized_value(self, milia_pkg):
        """``__status__`` is one of the standard PyPI classifier values."""
        recognized = {
            "Planning",
            "Pre-Alpha",
            "Alpha",
            "Beta",
            "Production",
            "Production/Stable",
            "Mature",
            "Inactive",
        }
        assert milia_pkg.__status__ in recognized, (
            f"Unexpected __status__ = '{milia_pkg.__status__}'"
        )


class TestSmokeCLIExports:
    """§1.2 + §1.3 — CLI management exports are accessible."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "name",
        [
            "CLIManager",
            "CLIValidationError",
            "create_cli_manager",
            "parse_cli_args",
            "get_cli_registry_status",
        ],
    )
    def test_cli_export_is_importable(self, milia_pkg, name):
        """Each CLI export resolves to a non-None object."""
        obj = getattr(milia_pkg, name, None)
        assert obj is not None, f"CLI export '{name}' is None or missing"


class TestSmokeLoggingExports:
    """§1.2 — Logging configuration exports are accessible."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "name",
        [
            "setup_logging",
            "HandlerLoggerAdapter",
            "MigrationLoggerAdapter",
            "TransformLoggerAdapter",
            "create_handler_logger",
            "create_migration_logger",
            "create_transform_logger",
            "log_exception_with_context",
            "configure_debug_logging_for_handlers",
            "configure_debug_logging_for_transforms",
            "disable_verbose_third_party_logging",
        ],
    )
    def test_logging_export_is_importable(self, milia_pkg, name):
        """Each logging export resolves to a non-None object."""
        obj = getattr(milia_pkg, name, None)
        assert obj is not None, f"Logging export '{name}' is None or missing"


class TestSmokeExceptionExports:
    """§1.2 — Exception class exports are importable and are classes."""

    # Tier 1 + Tier 2 base exceptions
    TIER_1_AND_2 = [
        "BaseProjectError",
        "ConfigurationError",
        "DataProcessingError",
        "MoleculeProcessingError",
        "HandlerError",
        "TransformError",
        "PluginError",
        "ValidationError",
        "DescriptorError",
        "UncertaintyProcessingError",
        "DatasetSpecificHandlerError",
    ]

    # Tier 3 specialized exceptions (complete list from __init__.py)
    TIER_3 = [
        "LoggingConfigurationError",
        "PreprocessingRequiredError",
        "MissingDependencyError",
        "MoleculeFilterRejectedError",
        "AtomFilterError",
        "RDKitConversionError",
        "PyGDataCreationError",
        "PropertyEnrichmentError",
        "StructuralFeatureError",
        "VibrationRefinementError",
        "HandlerNotAvailableError",
        "HandlerConfigurationError",
        "HandlerOperationError",
        "HandlerValidationError",
        "HandlerCompatibilityError",
        "HandlerIntegrationError",
        "TransformHandlerIntegrationError",
        "CompatibilityError",
        "MigrationError",
        "LegacyCodeError",
        "TransformCompatibilityError",
        "TransformationError",
        "DatasetIntegrationError",
        "TransformValidationError",
        "TransformCompositionError",
        "TransformNotFoundError",
        "TransformRegistryError",
        "ExperimentalSetupError",
        "TransformConfigurationError",
        "PluginValidationError",
        "PluginSecurityError",
        "PluginDependencyError",
        "PluginDiscoveryError",
        "PluginRegistrationError",
        "PluginLoadError",
        "DescriptorCalculationError",
        "DescriptorValidationError",
        "DescriptorPluginError",
        "DescriptorPluginLoadError",
        "DescriptorPluginValidationError",
        "DescriptorPluginConfigError",
    ]

    ALL_EXCEPTIONS = TIER_1_AND_2 + TIER_3

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", ALL_EXCEPTIONS)
    def test_exception_class_exists(self, milia_pkg, name):
        """Each documented exception is importable from the package."""
        obj = getattr(milia_pkg, name, None)
        assert obj is not None, f"Exception '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", ALL_EXCEPTIONS)
    def test_exception_is_a_class(self, milia_pkg, name):
        """Each exception export is a class (not an instance or function)."""
        obj = getattr(milia_pkg, name)
        assert inspect.isclass(obj), f"'{name}' should be a class, got {type(obj).__name__}"


class TestSmokeExceptionFactoryExports:
    """§1.2 — Exception factory functions and registry utilities are accessible."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "name",
        [
            "create_dataset_handler_error",
            "create_uncertainty_processing_error",
            "create_handler_not_available_error",
            "get_exception_registry_status",
            "validate_exception_hierarchy",
        ],
    )
    def test_factory_export_exists(self, milia_pkg, name):
        """Each factory/utility export is present and non-None."""
        obj = getattr(milia_pkg, name, None)
        assert obj is not None, f"Factory export '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "name",
        [
            "create_dataset_handler_error",
            "create_uncertainty_processing_error",
            "create_handler_not_available_error",
            "get_exception_registry_status",
            "validate_exception_hierarchy",
        ],
    )
    def test_factory_export_is_callable(self, milia_pkg, name):
        """Each factory/utility export is callable."""
        obj = getattr(milia_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeConditionalPostTrainingFlags:
    """§1.2 — Conditional availability flags exist and are boolean."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "flag",
        [
            "_POST_TRAINING_AVAILABLE",
            "_DATA_PREPARATION_AVAILABLE",
            "_TRANSFER_LEARNING_AVAILABLE",
        ],
    )
    def test_availability_flag_exists(self, milia_pkg, flag):
        """Each conditional availability flag is defined."""
        assert hasattr(milia_pkg, flag), f"Flag '{flag}' is missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "flag",
        [
            "_POST_TRAINING_AVAILABLE",
            "_DATA_PREPARATION_AVAILABLE",
            "_TRANSFER_LEARNING_AVAILABLE",
        ],
    )
    def test_availability_flag_is_bool(self, milia_pkg, flag):
        """Each conditional availability flag is a boolean."""
        value = getattr(milia_pkg, flag)
        assert isinstance(value, bool), f"Flag '{flag}' should be bool, got {type(value).__name__}"


class TestSmokeConditionalPostTrainingExports:
    """§1.2 — Post-training exports exist (may be None if unavailable)."""

    # Post-training model loading & prediction
    POST_TRAINING_NAMES = [
        "ModelLoader",
        "load_model",
        "load_model_only",
        "Predictor",
        "predict",
        "CheckpointManager",
        "CHECKPOINT_FORMAT_VERSION",
    ]

    # Data preparation exports
    DATA_PREP_NAMES = [
        "convert_to_pyg",
        "convert_batch_to_pyg",
        "list_available_formats",
        "DataConverterRegistry",
    ]

    # Transfer learning exports
    TRANSFER_NAMES = [
        "FineTuner",
        "FreezeStrategy",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", POST_TRAINING_NAMES)
    def test_post_training_name_defined(self, milia_pkg, name):
        """
        Post-training exports are defined on the package regardless of
        availability (they may be ``None`` if the submodule is absent).
        """
        assert hasattr(milia_pkg, name), f"Post-training export '{name}' is not defined"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", DATA_PREP_NAMES)
    def test_data_preparation_name_defined(self, milia_pkg, name):
        """Data-preparation exports are defined (possibly None)."""
        assert hasattr(milia_pkg, name), f"Data-preparation export '{name}' is not defined"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", TRANSFER_NAMES)
    def test_transfer_learning_name_defined(self, milia_pkg, name):
        """Transfer-learning exports are defined (possibly None)."""
        assert hasattr(milia_pkg, name), f"Transfer-learning export '{name}' is not defined"

    @pytest.mark.smoke
    def test_post_training_consistency_when_available(self, milia_pkg):
        """
        If ``_POST_TRAINING_AVAILABLE`` is True, none of the post-training
        exports should be None.
        """
        if not milia_pkg._POST_TRAINING_AVAILABLE:
            pytest.skip("Post-training module not available")
        for name in self.POST_TRAINING_NAMES:
            obj = getattr(milia_pkg, name)
            assert obj is not None, f"_POST_TRAINING_AVAILABLE is True but '{name}' is None"

    @pytest.mark.smoke
    def test_data_preparation_consistency_when_available(self, milia_pkg):
        """
        If ``_DATA_PREPARATION_AVAILABLE`` is True, none of the data-prep
        exports should be None.
        """
        if not milia_pkg._DATA_PREPARATION_AVAILABLE:
            pytest.skip("Data preparation module not available")
        for name in self.DATA_PREP_NAMES:
            obj = getattr(milia_pkg, name)
            assert obj is not None, f"_DATA_PREPARATION_AVAILABLE is True but '{name}' is None"

    @pytest.mark.smoke
    def test_transfer_learning_consistency_when_available(self, milia_pkg):
        """
        If ``_TRANSFER_LEARNING_AVAILABLE`` is True, none of the transfer-
        learning exports should be None.
        """
        if not milia_pkg._TRANSFER_LEARNING_AVAILABLE:
            pytest.skip("Transfer learning module not available")
        for name in self.TRANSFER_NAMES:
            obj = getattr(milia_pkg, name)
            assert obj is not None, f"_TRANSFER_LEARNING_AVAILABLE is True but '{name}' is None"

    @pytest.mark.smoke
    def test_post_training_none_when_unavailable(self, milia_pkg):
        """
        Simulate ``_POST_TRAINING_AVAILABLE = False`` by reloading the
        package with the post-training import patched to raise ImportError.

        Verifies the graceful-degradation contract: when the submodule is
        absent, every post-training export is set to ``None``.
        """
        import importlib

        target = "milia_pipeline.models.post_training"
        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def _mock_import(name, *args, **kwargs):
            if name == target or name.startswith(target + "."):
                raise ImportError(f"Mocked: {name} unavailable")
            return original_import(name, *args, **kwargs)

        # Snapshot sys.modules BEFORE the mocked reload so we can fully
        # restore it afterwards.  importlib.reload() with a mocked
        # builtins.__import__ can evict or corrupt submodule entries in
        # sys.modules, which would break subsequent test-file collection
        # in the same pytest process.
        saved_modules = sys.modules.copy()

        try:
            with patch("builtins.__import__", side_effect=_mock_import):
                reloaded = importlib.reload(milia_pkg)

            assert reloaded._POST_TRAINING_AVAILABLE is False, (
                "_POST_TRAINING_AVAILABLE should be False when import fails"
            )
            for name in self.POST_TRAINING_NAMES:
                obj = getattr(reloaded, name)
                assert obj is None, f"_POST_TRAINING_AVAILABLE is False but '{name}' is not None"
        finally:
            # Fully restore sys.modules to the pre-reload snapshot, then
            # reload the package cleanly so subsequent tests (and test
            # files collected later in the same process) see a healthy
            # milia_pipeline with all submodules intact.
            sys.modules.clear()
            sys.modules.update(saved_modules)
            importlib.reload(milia_pkg)

    @pytest.mark.smoke
    def test_data_preparation_none_when_unavailable(self, milia_pkg):
        """
        Simulate ``_DATA_PREPARATION_AVAILABLE = False`` by reloading with
        the data_preparation import patched to raise ImportError.
        """
        import importlib

        target = "milia_pipeline.models.post_training.data_preparation"
        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def _mock_import(name, *args, **kwargs):
            if name == target or name.startswith(target + "."):
                raise ImportError(f"Mocked: {name} unavailable")
            return original_import(name, *args, **kwargs)

        saved_modules = sys.modules.copy()

        try:
            with patch("builtins.__import__", side_effect=_mock_import):
                reloaded = importlib.reload(milia_pkg)

            assert reloaded._DATA_PREPARATION_AVAILABLE is False, (
                "_DATA_PREPARATION_AVAILABLE should be False when import fails"
            )
            for name in self.DATA_PREP_NAMES:
                obj = getattr(reloaded, name)
                assert obj is None, f"_DATA_PREPARATION_AVAILABLE is False but '{name}' is not None"
        finally:
            sys.modules.clear()
            sys.modules.update(saved_modules)
            importlib.reload(milia_pkg)

    @pytest.mark.smoke
    def test_transfer_learning_none_when_unavailable(self, milia_pkg):
        """
        Simulate ``_TRANSFER_LEARNING_AVAILABLE = False`` by reloading with
        the transfer_learning import patched to raise ImportError.
        """
        import importlib

        target = "milia_pipeline.models.post_training.transfer_learning"
        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def _mock_import(name, *args, **kwargs):
            if name == target or name.startswith(target + "."):
                raise ImportError(f"Mocked: {name} unavailable")
            return original_import(name, *args, **kwargs)

        saved_modules = sys.modules.copy()

        try:
            with patch("builtins.__import__", side_effect=_mock_import):
                reloaded = importlib.reload(milia_pkg)

            assert reloaded._TRANSFER_LEARNING_AVAILABLE is False, (
                "_TRANSFER_LEARNING_AVAILABLE should be False when import fails"
            )
            for name in self.TRANSFER_NAMES:
                obj = getattr(reloaded, name)
                assert obj is None, (
                    f"_TRANSFER_LEARNING_AVAILABLE is False but '{name}' is not None"
                )
        finally:
            sys.modules.clear()
            sys.modules.update(saved_modules)
            importlib.reload(milia_pkg)


class TestSmokeConvenienceFunctions:
    """§1.2 — Module-level convenience functions execute without errors."""

    @pytest.mark.smoke
    def test_get_version_returns_string(self, milia_pkg):
        """``get_version()`` returns a non-empty string."""
        result = milia_pkg.get_version()
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.smoke
    def test_get_version_matches_dunder(self, milia_pkg):
        """``get_version()`` returns the same value as ``__version__``."""
        assert milia_pkg.get_version() == milia_pkg.__version__

    @pytest.mark.smoke
    def test_get_package_info_returns_dict(self, milia_pkg):
        """``get_package_info()`` returns a dictionary."""
        result = milia_pkg.get_package_info()
        assert isinstance(result, dict)

    @pytest.mark.smoke
    def test_get_package_info_has_required_keys(self, milia_pkg):
        """``get_package_info()`` contains all documented keys."""
        info = milia_pkg.get_package_info()
        required_keys = {
            "version",
            "author",
            "license",
            "maintainer",
            "status",
            "python_requires",
        }
        missing = required_keys - set(info.keys())
        assert not missing, f"get_package_info() missing keys: {missing}"

    @pytest.mark.smoke
    def test_get_package_info_version_consistency(self, milia_pkg):
        """The ``version`` key in ``get_package_info()`` matches ``__version__``."""
        info = milia_pkg.get_package_info()
        assert info["version"] == milia_pkg.__version__

    @pytest.mark.smoke
    def test_check_dependencies_returns_dict(self, milia_pkg):
        """``check_dependencies()`` returns a dictionary."""
        result = milia_pkg.check_dependencies()
        assert isinstance(result, dict)

    @pytest.mark.smoke
    def test_check_dependencies_has_core_keys(self, milia_pkg):
        """``check_dependencies()`` includes all documented dependency keys."""
        deps = milia_pkg.check_dependencies()
        expected_keys = {
            "rdkit",
            "torch_geometric",
            "iodata",
            "transforms_available",
            "plugins_available",
            "config_validation_available",
            "post_training_available",
            "data_preparation_available",
            "transfer_learning_available",
        }
        missing = expected_keys - set(deps.keys())
        assert not missing, f"check_dependencies() missing keys: {missing}"

    @pytest.mark.smoke
    def test_check_dependencies_values_are_bool(self, milia_pkg):
        """All values returned by ``check_dependencies()`` are boolean."""
        deps = milia_pkg.check_dependencies()
        for key, value in deps.items():
            assert isinstance(value, bool), (
                f"check_dependencies()['{key}'] = {value!r}, expected bool"
            )

    @pytest.mark.smoke
    def test_check_dependencies_post_training_flag_consistency(self, milia_pkg):
        """
        ``check_dependencies()['post_training_available']`` matches the
        module-level ``_POST_TRAINING_AVAILABLE`` flag.
        """
        deps = milia_pkg.check_dependencies()
        assert deps["post_training_available"] == milia_pkg._POST_TRAINING_AVAILABLE

    @pytest.mark.smoke
    def test_check_dependencies_data_prep_flag_consistency(self, milia_pkg):
        """
        ``check_dependencies()['data_preparation_available']`` matches the
        module-level ``_DATA_PREPARATION_AVAILABLE`` flag.
        """
        deps = milia_pkg.check_dependencies()
        assert deps["data_preparation_available"] == milia_pkg._DATA_PREPARATION_AVAILABLE

    @pytest.mark.smoke
    def test_check_dependencies_transfer_flag_consistency(self, milia_pkg):
        """
        ``check_dependencies()['transfer_learning_available']`` matches the
        module-level ``_TRANSFER_LEARNING_AVAILABLE`` flag.
        """
        deps = milia_pkg.check_dependencies()
        assert deps["transfer_learning_available"] == milia_pkg._TRANSFER_LEARNING_AVAILABLE


class TestSmokeInitializePipeline:
    """§1.2 — ``initialize_pipeline()`` function is accessible and callable."""

    @pytest.mark.smoke
    def test_initialize_pipeline_exists(self, milia_pkg):
        """``initialize_pipeline`` is defined on the package."""
        assert hasattr(milia_pkg, "initialize_pipeline")

    @pytest.mark.smoke
    def test_initialize_pipeline_is_callable(self, milia_pkg):
        """``initialize_pipeline`` is callable."""
        assert callable(milia_pkg.initialize_pipeline)

    @pytest.mark.smoke
    def test_initialize_pipeline_signature(self, milia_pkg):
        """``initialize_pipeline`` accepts the documented parameters."""
        sig = inspect.signature(milia_pkg.initialize_pipeline)
        param_names = set(sig.parameters.keys())
        expected = {"config_path", "log_level", "enable_plugins", "validate_config"}
        assert expected.issubset(param_names), (
            f"initialize_pipeline missing params: {expected - param_names}"
        )

    @pytest.mark.smoke
    def test_initialize_pipeline_defaults(self, milia_pkg):
        """``initialize_pipeline`` parameters have the documented defaults."""
        sig = inspect.signature(milia_pkg.initialize_pipeline)
        params = sig.parameters

        assert params["config_path"].default is None
        assert params["log_level"].default == "INFO"
        assert params["enable_plugins"].default is True
        assert params["validate_config"].default is True

    @pytest.mark.smoke
    def test_initialize_pipeline_no_config_returns_tuple(self, milia_pkg):
        """
        Calling ``initialize_pipeline()`` without a config path returns a
        3-tuple of (logger, None, cli_manager).
        """
        result = milia_pkg.initialize_pipeline(
            config_path=None,
            log_level="WARNING",
            enable_plugins=False,
            validate_config=False,
        )
        assert isinstance(result, tuple), "Should return a tuple"
        assert len(result) == 3, "Should return a 3-tuple"

        logger, config, cli_manager = result
        assert isinstance(logger, logging.Logger), (
            f"First element should be Logger, got {type(logger)}"
        )
        assert config is None, "Config should be None when no config_path given"
        # cli_manager should be an instance of CLIManager
        assert cli_manager is not None, "cli_manager should not be None"


class TestSmokeBackwardCompatibilityAliases:
    """§1.2 — Backward-compatibility aliases are importable."""

    @pytest.mark.smoke
    def test_project_error_alias_exists(self, milia_pkg):
        """``ProjectError`` alias is defined."""
        assert hasattr(milia_pkg, "ProjectError")

    @pytest.mark.smoke
    def test_processing_error_alias_exists(self, milia_pkg):
        """``ProcessingError`` alias is defined."""
        assert hasattr(milia_pkg, "ProcessingError")


class TestSmokeModuleInitialization:
    """§1.2 — Module-level initialization runs without side-effect crashes."""

    @pytest.mark.smoke
    def test_reimport_is_idempotent(self, milia_pkg):
        """
        Re-importing the package (via ``importlib.reload``) does not crash.

        This validates that ``disable_verbose_third_party_logging()`` and
        any other module-level side effects are safe to re-execute.
        """
        # Snapshot sys.modules before the reload.  importlib.reload()
        # re-executes the module-level code in milia_pipeline/__init__.py,
        # which performs ``from milia_pipeline.X import ...`` statements.
        # These can mutate sys.modules entries for subpackages.  We restore
        # the snapshot afterwards so that subsequent test files collected
        # in the same pytest process still find all submodules intact.
        saved_modules = sys.modules.copy()

        try:
            reloaded = importlib.reload(milia_pkg)
            assert reloaded is not None
            assert hasattr(reloaded, "__version__")
        finally:
            sys.modules.clear()
            sys.modules.update(saved_modules)
            importlib.reload(milia_pkg)


# ===================================================================
# SECTION 2 — CONTRACT TESTS
# ===================================================================


class TestContractAllCompleteness:
    """§2 — Every name in ``__all__`` is resolvable on the package."""

    @pytest.mark.contract
    def test_all_is_a_list(self, milia_pkg):
        """``__all__`` is a list (not a tuple or set)."""
        assert isinstance(milia_pkg.__all__, list)

    @pytest.mark.contract
    def test_all_contains_no_duplicates(self, all_names):
        """``__all__`` has no duplicate entries."""
        seen = set()
        duplicates = []
        for name in all_names:
            if name in seen:
                duplicates.append(name)
            seen.add(name)
        assert not duplicates, f"Duplicate entries in __all__: {duplicates}"

    @pytest.mark.contract
    @pytest.mark.parametrize(
        "name",
        [
            # Package Metadata
            "__version__",
            "__author__",
            "__license__",
            "__maintainer__",
            "__status__",
            # CLI Management
            "CLIManager",
            "CLIValidationError",
            "create_cli_manager",
            "parse_cli_args",
            "get_cli_registry_status",
            # Post-Training (conditional — names still listed in __all__)
            "ModelLoader",
            "load_model",
            "load_model_only",
            "Predictor",
            "predict",
            "CheckpointManager",
            "CHECKPOINT_FORMAT_VERSION",
            "convert_to_pyg",
            "convert_batch_to_pyg",
            "list_available_formats",
            "DataConverterRegistry",
            "FineTuner",
            "FreezeStrategy",
            "_POST_TRAINING_AVAILABLE",
            "_DATA_PREPARATION_AVAILABLE",
            "_TRANSFER_LEARNING_AVAILABLE",
            # Logging
            "setup_logging",
            "HandlerLoggerAdapter",
            "MigrationLoggerAdapter",
            "TransformLoggerAdapter",
            "create_handler_logger",
            "create_migration_logger",
            "create_transform_logger",
            "log_exception_with_context",
            "configure_debug_logging_for_handlers",
            "configure_debug_logging_for_transforms",
            "disable_verbose_third_party_logging",
            # Base + Config Exceptions
            "BaseProjectError",
            "LoggingConfigurationError",
            "ConfigurationError",
            "DataProcessingError",
            "PreprocessingRequiredError",
            "MissingDependencyError",
            # Molecule Exceptions
            "MoleculeProcessingError",
            "MoleculeFilterRejectedError",
            "AtomFilterError",
            "RDKitConversionError",
            "PyGDataCreationError",
            "PropertyEnrichmentError",
            "StructuralFeatureError",
            "VibrationRefinementError",
            # Phase 7
            "UncertaintyProcessingError",
            # Handler Exceptions
            "HandlerError",
            "HandlerNotAvailableError",
            "HandlerConfigurationError",
            "HandlerOperationError",
            "HandlerValidationError",
            "HandlerCompatibilityError",
            "HandlerIntegrationError",
            "TransformHandlerIntegrationError",
            "DatasetSpecificHandlerError",
            # Validation/Compat Exceptions
            "ValidationError",
            "CompatibilityError",
            "MigrationError",
            "LegacyCodeError",
            # Transform Exceptions
            "TransformError",
            "TransformCompatibilityError",
            "TransformationError",
            "DatasetIntegrationError",
            "TransformValidationError",
            "TransformCompositionError",
            "TransformNotFoundError",
            "TransformRegistryError",
            "ExperimentalSetupError",
            "TransformConfigurationError",
            # Plugin Exceptions
            "PluginError",
            "PluginValidationError",
            "PluginSecurityError",
            "PluginDependencyError",
            "PluginDiscoveryError",
            "PluginRegistrationError",
            "PluginLoadError",
            # Descriptor Exceptions
            "DescriptorError",
            "DescriptorCalculationError",
            "DescriptorValidationError",
            "DescriptorPluginError",
            "DescriptorPluginLoadError",
            "DescriptorPluginValidationError",
            "DescriptorPluginConfigError",
            # Factory Functions
            "create_dataset_handler_error",
            "create_uncertainty_processing_error",
            "create_handler_not_available_error",
            "get_exception_registry_status",
            "validate_exception_hierarchy",
        ],
    )
    def test_all_name_is_resolvable(self, milia_pkg, name):
        """
        Every name that appears in ``__all__`` resolves to an attribute on
        the package (may be ``None`` for conditional exports).
        """
        assert name in milia_pkg.__all__, (
            f"'{name}' not found in __all__ — possible accidental removal"
        )
        assert hasattr(milia_pkg, name), (
            f"'{name}' is listed in __all__ but not defined on the module"
        )

    @pytest.mark.contract
    def test_every_all_entry_is_resolvable(self, milia_pkg, all_names):
        """
        Generic sweep: every single entry in ``__all__`` must be resolvable,
        regardless of whether it is parameterized above.
        """
        unresolvable = [name for name in all_names if not hasattr(milia_pkg, name)]
        assert not unresolvable, (
            f"Names in __all__ that are not defined on the module: {unresolvable}"
        )


class TestContractAllConsistency:
    """§2 — Every public import in the module is listed in ``__all__``."""

    # Names that are intentionally public imports but NOT in __all__
    # (backward-compatibility aliases, convenience functions, etc.)
    KNOWN_UNLISTED = {
        # Backward-compat aliases (documented in module but not in __all__)
        "ProjectError",
        "ProcessingError",
        # Convenience functions (public API but exported separately)
        "get_version",
        "get_package_info",
        "check_dependencies",
        "initialize_pipeline",
    }

    @pytest.mark.contract
    def test_public_imports_are_in_all(self, milia_pkg, all_names):
        """
        Every non-dunder, non-private attribute that was imported (not
        defined locally) should be in ``__all__`` — unless it is in
        ``KNOWN_UNLISTED``.

        This catches accidental omissions when new imports are added to
        ``__init__.py`` without updating ``__all__``.
        """
        all_set = set(all_names)
        module_dict = vars(milia_pkg)
        missing_from_all = []

        for name, obj in module_dict.items():
            # Skip private/dunder names
            if name.startswith("_") and name not in all_set:
                # Exception: _POST_TRAINING_AVAILABLE etc. ARE in __all__
                continue
            # Skip modules (submodule references)
            if isinstance(obj, types.ModuleType):
                continue
            # Skip known unlisted names
            if name in self.KNOWN_UNLISTED:
                continue
            # Skip names that start with underscore but are in __all__
            # (already handled by the first condition)

            if name not in all_set and not name.startswith("_"):
                missing_from_all.append(name)

        # Filter out common Python internals that might leak through
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
            f"Public names imported in __init__.py but not in __all__: {sorted(missing_from_all)}"
        )


class TestContractBackwardCompatibility:
    """§2 — Backward-compatibility aliases point to the correct targets."""

    @pytest.mark.contract
    def test_project_error_is_base_project_error(self, milia_pkg):
        """``ProjectError`` is an alias for ``BaseProjectError``."""
        assert milia_pkg.ProjectError is milia_pkg.BaseProjectError

    @pytest.mark.contract
    def test_processing_error_is_data_processing_error(self, milia_pkg):
        """``ProcessingError`` is an alias for ``DataProcessingError``."""
        assert milia_pkg.ProcessingError is milia_pkg.DataProcessingError


class TestContractExceptionHierarchy:
    """
    §2 — Exception hierarchy contracts.

    Validates the three-tier inheritance documented in the module docstring:
        Tier 1: BaseProjectError(Exception)
        Tier 2: Domain bases inherit from BaseProjectError (or Exception for
                 MoleculeFilterRejectedError which inherits BaseException)
        Tier 3: Specialized exceptions inherit from their domain base
    """

    @pytest.mark.contract
    def test_tier1_inherits_from_exception(self, milia_pkg):
        """Tier 1: ``BaseProjectError`` is a subclass of ``Exception``."""
        assert issubclass(milia_pkg.BaseProjectError, Exception)

    @pytest.mark.contract
    @pytest.mark.parametrize(
        "name",
        [
            "ConfigurationError",
            "DataProcessingError",
            "MoleculeProcessingError",
            "HandlerError",
            "TransformError",
            "PluginError",
            "ValidationError",
            "DescriptorError",
        ],
    )
    def test_tier2_inherits_from_base_project_error(self, milia_pkg, name):
        """Tier 2: Domain base classes inherit from ``BaseProjectError``."""
        cls = getattr(milia_pkg, name)
        assert issubclass(cls, milia_pkg.BaseProjectError), (
            f"{name} should be a subclass of BaseProjectError"
        )

    @pytest.mark.contract
    def test_molecule_filter_rejected_inherits_base_exception(self, milia_pkg):
        """
        ``MoleculeFilterRejectedError`` inherits from ``BaseException``
        (not ``Exception``) — documented as an expected-rejection signal.
        """
        assert issubclass(milia_pkg.MoleculeFilterRejectedError, BaseException)

    # Tier 3 → direct parent mapping (verified from exceptions.py source)
    TIER3_TO_PARENT = {
        # Config-area exceptions
        "LoggingConfigurationError": "BaseProjectError",
        "TransformConfigurationError": "ConfigurationError",
        "ExperimentalSetupError": "ConfigurationError",
        # Data processing exceptions
        "PreprocessingRequiredError": "DataProcessingError",
        "VibrationRefinementError": "DataProcessingError",
        "DatasetIntegrationError": "DataProcessingError",
        "TransformCompositionError": "DataProcessingError",
        # Molecule processing exceptions
        "RDKitConversionError": "MoleculeProcessingError",
        "PyGDataCreationError": "MoleculeProcessingError",
        "PropertyEnrichmentError": "MoleculeProcessingError",
        "StructuralFeatureError": "MoleculeProcessingError",
        # Handler exceptions → HandlerError
        "HandlerNotAvailableError": "HandlerError",
        "HandlerConfigurationError": "HandlerError",
        "HandlerOperationError": "HandlerError",
        "HandlerValidationError": "HandlerError",
        "HandlerCompatibilityError": "HandlerError",
        # HandlerIntegrationError inherits BaseProjectError (not HandlerError)
        "HandlerIntegrationError": "BaseProjectError",
        # TransformHandlerIntegrationError inherits HandlerIntegrationError
        "TransformHandlerIntegrationError": "HandlerIntegrationError",
        # DatasetSpecificHandlerError inherits HandlerError
        "DatasetSpecificHandlerError": "HandlerError",
        # Transform exceptions
        "TransformCompatibilityError": "TransformError",
        # TransformationError inherits BaseProjectError (not TransformError)
        "TransformationError": "BaseProjectError",
        "TransformNotFoundError": "TransformationError",
        "TransformRegistryError": "TransformationError",
        # TransformValidationError inherits ValidationError (cross-hierarchy)
        "TransformValidationError": "ValidationError",
        # Plugin exceptions → PluginError
        "PluginValidationError": "PluginError",
        "PluginSecurityError": "PluginError",
        "PluginDependencyError": "PluginError",
        "PluginDiscoveryError": "PluginError",
        "PluginRegistrationError": "PluginError",
        "PluginLoadError": "PluginError",
        # Descriptor exceptions → DescriptorError
        "DescriptorCalculationError": "DescriptorError",
        "DescriptorValidationError": "DescriptorError",
        "DescriptorPluginError": "DescriptorError",
        "DescriptorPluginLoadError": "DescriptorPluginError",
        "DescriptorPluginValidationError": "DescriptorPluginError",
        "DescriptorPluginConfigError": "DescriptorPluginError",
        # Validation/Compat → BaseProjectError
        "CompatibilityError": "BaseProjectError",
        "MigrationError": "BaseProjectError",
        "LegacyCodeError": "BaseProjectError",
        # MissingDependencyError → BaseProjectError
        "MissingDependencyError": "BaseProjectError",
        # AtomFilterError → BaseProjectError
        "AtomFilterError": "BaseProjectError",
        # UncertaintyProcessingError → MoleculeProcessingError
        "UncertaintyProcessingError": "MoleculeProcessingError",
    }

    @pytest.mark.contract
    @pytest.mark.parametrize(
        "child_name,parent_name",
        list(TIER3_TO_PARENT.items()),
        ids=[f"{c}->{p}" for c, p in TIER3_TO_PARENT.items()],
    )
    def test_tier3_inherits_from_documented_parent(self, milia_pkg, child_name, parent_name):
        """Tier 3 exceptions inherit from their actual documented parent."""
        child = getattr(milia_pkg, child_name)
        parent = getattr(milia_pkg, parent_name)
        assert issubclass(child, parent), f"{child_name} should be a subclass of {parent_name}"

    @pytest.mark.contract
    def test_transform_validation_error_inherits_validation_error(self, milia_pkg):
        """
        ``TransformValidationError`` inherits from ``ValidationError``
        (cross-hierarchy inheritance).
        """
        assert issubclass(
            milia_pkg.TransformValidationError,
            milia_pkg.ValidationError,
        )

    @pytest.mark.contract
    def test_all_tier3_ultimately_inherit_base_project_error(self, milia_pkg):
        """
        Every Tier 3 exception (except MoleculeFilterRejectedError) should
        ultimately be a subclass of ``BaseProjectError``.
        """
        # Exceptions that inherit from BaseException rather than Exception
        exempt = {"MoleculeFilterRejectedError"}

        for name in TestSmokeExceptionExports.TIER_3:
            if name in exempt:
                continue
            cls = getattr(milia_pkg, name, None)
            if cls is None:
                continue  # conditional export
            assert issubclass(cls, milia_pkg.BaseProjectError), (
                f"{name} should ultimately inherit from BaseProjectError"
            )


class TestContractConvenienceFunctionReturnTypes:
    """§2 — Convenience functions return the documented types."""

    @pytest.mark.contract
    def test_get_version_returns_str(self, milia_pkg):
        """``get_version()`` → ``str``."""
        assert isinstance(milia_pkg.get_version(), str)

    @pytest.mark.contract
    def test_get_package_info_returns_dict_str_str(self, milia_pkg):
        """``get_package_info()`` → ``dict`` with string keys and string values."""
        info = milia_pkg.get_package_info()
        assert isinstance(info, dict)
        for key, value in info.items():
            assert isinstance(key, str), f"Key {key!r} should be str"
            assert isinstance(value, str), (
                f"Value for key '{key}' should be str, got {type(value).__name__}"
            )

    @pytest.mark.contract
    def test_check_dependencies_returns_dict_str_bool(self, milia_pkg):
        """``check_dependencies()`` → ``dict`` with string keys and bool values."""
        deps = milia_pkg.check_dependencies()
        assert isinstance(deps, dict)
        for key, value in deps.items():
            assert isinstance(key, str), f"Key {key!r} should be str"
            assert isinstance(value, bool), (
                f"Value for key '{key}' should be bool, got {type(value).__name__}"
            )


class TestContractCLIExportTypes:
    """§2 — CLI exports have the correct type categories."""

    @pytest.mark.contract
    def test_cli_manager_is_a_class(self, milia_pkg):
        """``CLIManager`` is a class."""
        assert inspect.isclass(milia_pkg.CLIManager)

    @pytest.mark.contract
    def test_cli_validation_error_is_exception_class(self, milia_pkg):
        """``CLIValidationError`` is an exception class."""
        assert inspect.isclass(milia_pkg.CLIValidationError)
        assert issubclass(milia_pkg.CLIValidationError, Exception)

    @pytest.mark.contract
    def test_create_cli_manager_is_callable(self, milia_pkg):
        """``create_cli_manager`` is a callable (factory function)."""
        assert callable(milia_pkg.create_cli_manager)

    @pytest.mark.contract
    def test_parse_cli_args_is_callable(self, milia_pkg):
        """``parse_cli_args`` is a callable."""
        assert callable(milia_pkg.parse_cli_args)

    @pytest.mark.contract
    def test_get_cli_registry_status_is_callable(self, milia_pkg):
        """``get_cli_registry_status`` is a callable."""
        assert callable(milia_pkg.get_cli_registry_status)


class TestContractLoggingExportTypes:
    """§2 — Logging exports have the correct type categories."""

    LOGGER_ADAPTER_CLASSES = [
        "HandlerLoggerAdapter",
        "MigrationLoggerAdapter",
        "TransformLoggerAdapter",
    ]

    LOGGER_FACTORY_FUNCTIONS = [
        "create_handler_logger",
        "create_migration_logger",
        "create_transform_logger",
    ]

    LOGGING_CALLABLES = [
        "setup_logging",
        "log_exception_with_context",
        "configure_debug_logging_for_handlers",
        "configure_debug_logging_for_transforms",
        "disable_verbose_third_party_logging",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", LOGGER_ADAPTER_CLASSES)
    def test_logger_adapter_is_a_class(self, milia_pkg, name):
        """Logger adapter exports are classes."""
        obj = getattr(milia_pkg, name)
        assert inspect.isclass(obj), f"'{name}' should be a class"

    @pytest.mark.contract
    @pytest.mark.parametrize("name", LOGGER_FACTORY_FUNCTIONS + LOGGING_CALLABLES)
    def test_logging_callable(self, milia_pkg, name):
        """Logging utility exports are callable."""
        obj = getattr(milia_pkg, name)
        assert callable(obj), f"'{name}' should be callable"

    @pytest.mark.contract
    def test_setup_logging_is_a_function(self, milia_pkg):
        """``setup_logging`` is a function (not a class)."""
        assert inspect.isfunction(milia_pkg.setup_logging)


class TestContractPublicAPISurface:
    """
    §2 — Public API surface stability.

    Ensures that the minimum expected public API is present. This guards
    against accidental removals during refactoring.
    """

    # The minimum API surface that MUST be present in __all__
    MINIMUM_API = {
        # Metadata
        "__version__",
        "__author__",
        "__license__",
        # CLI
        "CLIManager",
        "create_cli_manager",
        "parse_cli_args",
        # Logging
        "setup_logging",
        # Base exceptions
        "BaseProjectError",
        "ConfigurationError",
        "DataProcessingError",
        # Availability flags
        "_POST_TRAINING_AVAILABLE",
        "_DATA_PREPARATION_AVAILABLE",
        "_TRANSFER_LEARNING_AVAILABLE",
    }

    @pytest.mark.contract
    def test_minimum_api_in_all(self, all_names):
        """The minimum expected public API is present in ``__all__``."""
        all_set = set(all_names)
        missing = self.MINIMUM_API - all_set
        assert not missing, f"Minimum API names missing from __all__: {sorted(missing)}"

    @pytest.mark.contract
    def test_all_length_matches_expected(self, all_names):
        """
        ``__all__`` contains the expected number of entries.

        Based on the __init__.py source, __all__ has exactly 79 entries.
        This test guards against silent additions or removals. Update
        the expected count when intentionally changing the public API.
        """
        # Counted from lines 450-579 of __init__.py (94 entries)
        EXPECTED_COUNT = 94
        actual = len(all_names)
        assert actual == EXPECTED_COUNT, (
            f"__all__ has {actual} entries, expected {EXPECTED_COUNT}. "
            f"If this change is intentional, update EXPECTED_COUNT."
        )


class TestContractExceptionFactoryFunctionSignatures:
    """§2 — Exception factory functions have the documented signatures."""

    @pytest.mark.contract
    def test_create_dataset_handler_error_params(self, milia_pkg):
        """
        ``create_dataset_handler_error`` accepts the documented parameters:
        message, dataset_type, operation, property_name, **kwargs.
        """
        sig = inspect.signature(milia_pkg.create_dataset_handler_error)
        param_names = list(sig.parameters.keys())
        # Must include at least: message, dataset_type
        assert "message" in param_names, "Missing 'message' parameter"
        assert "dataset_type" in param_names, "Missing 'dataset_type' parameter"

    @pytest.mark.contract
    def test_create_uncertainty_processing_error_params(self, milia_pkg):
        """
        ``create_uncertainty_processing_error`` accepts the documented parameters:
        message, dataset_type, molecule_index, ...
        """
        sig = inspect.signature(milia_pkg.create_uncertainty_processing_error)
        param_names = list(sig.parameters.keys())
        assert "message" in param_names, "Missing 'message' parameter"
        assert "dataset_type" in param_names, "Missing 'dataset_type' parameter"

    @pytest.mark.contract
    def test_create_handler_not_available_error_params(self, milia_pkg):
        """
        ``create_handler_not_available_error`` accepts the documented parameters:
        message, requested_dataset_type, ...
        """
        sig = inspect.signature(milia_pkg.create_handler_not_available_error)
        param_names = list(sig.parameters.keys())
        assert "message" in param_names, "Missing 'message' parameter"

    @pytest.mark.contract
    def test_get_exception_registry_status_returns_dict(self, milia_pkg):
        """``get_exception_registry_status()`` returns a dict."""
        result = milia_pkg.get_exception_registry_status()
        assert isinstance(result, dict), f"Expected dict, got {type(result).__name__}"


class TestContractConditionalExportSemantics:
    """
    §2 — Conditional export ternary semantics.

    When a conditional module IS available:
        - The availability flag is True
        - The exports are non-None and are classes/functions

    When a conditional module is NOT available:
        - The availability flag is False
        - The exports are None

    Both states must be internally consistent.
    """

    @pytest.mark.contract
    def test_post_training_ternary_consistency(self, milia_pkg):
        """
        ``_POST_TRAINING_AVAILABLE`` and its exports are mutually consistent.
        """
        flag = milia_pkg._POST_TRAINING_AVAILABLE
        names = [
            "ModelLoader",
            "load_model",
            "load_model_only",
            "Predictor",
            "predict",
            "CheckpointManager",
            "CHECKPOINT_FORMAT_VERSION",
        ]
        for name in names:
            obj = getattr(milia_pkg, name)
            if flag:
                assert obj is not None, f"Flag is True but '{name}' is None"
            else:
                assert obj is None, f"Flag is False but '{name}' is {obj!r}"

    @pytest.mark.contract
    def test_data_preparation_ternary_consistency(self, milia_pkg):
        """
        ``_DATA_PREPARATION_AVAILABLE`` and its exports are mutually consistent.
        """
        flag = milia_pkg._DATA_PREPARATION_AVAILABLE
        names = [
            "convert_to_pyg",
            "convert_batch_to_pyg",
            "list_available_formats",
            "DataConverterRegistry",
        ]
        for name in names:
            obj = getattr(milia_pkg, name)
            if flag:
                assert obj is not None, f"Flag is True but '{name}' is None"
            else:
                assert obj is None, f"Flag is False but '{name}' is {obj!r}"

    @pytest.mark.contract
    def test_transfer_learning_ternary_consistency(self, milia_pkg):
        """
        ``_TRANSFER_LEARNING_AVAILABLE`` and its exports are mutually consistent.
        """
        flag = milia_pkg._TRANSFER_LEARNING_AVAILABLE
        names = ["FineTuner", "FreezeStrategy"]
        for name in names:
            obj = getattr(milia_pkg, name)
            if flag:
                assert obj is not None, f"Flag is True but '{name}' is None"
            else:
                assert obj is None, f"Flag is False but '{name}' is {obj!r}"

    @pytest.mark.contract
    def test_post_training_types_when_available(self, milia_pkg):
        """
        When ``_POST_TRAINING_AVAILABLE`` is True, core exports are classes.
        """
        if not milia_pkg._POST_TRAINING_AVAILABLE:
            pytest.skip("Post-training module not available")

        assert inspect.isclass(milia_pkg.ModelLoader)
        assert inspect.isclass(milia_pkg.Predictor)
        assert inspect.isclass(milia_pkg.CheckpointManager)
        assert callable(milia_pkg.load_model)
        assert callable(milia_pkg.load_model_only)
        assert callable(milia_pkg.predict)

    @pytest.mark.contract
    def test_data_preparation_types_when_available(self, milia_pkg):
        """
        When ``_DATA_PREPARATION_AVAILABLE`` is True, core exports are callable/class.
        """
        if not milia_pkg._DATA_PREPARATION_AVAILABLE:
            pytest.skip("Data preparation module not available")

        assert callable(milia_pkg.convert_to_pyg)
        assert callable(milia_pkg.convert_batch_to_pyg)
        assert callable(milia_pkg.list_available_formats)
        assert inspect.isclass(milia_pkg.DataConverterRegistry)

    @pytest.mark.contract
    def test_transfer_learning_types_when_available(self, milia_pkg):
        """
        When ``_TRANSFER_LEARNING_AVAILABLE`` is True, exports are classes.
        """
        if not milia_pkg._TRANSFER_LEARNING_AVAILABLE:
            pytest.skip("Transfer learning module not available")

        assert inspect.isclass(milia_pkg.FineTuner)
        # FreezeStrategy could be a class or an enum
        assert inspect.isclass(milia_pkg.FreezeStrategy) or callable(milia_pkg.FreezeStrategy)


class TestContractExceptionInstantiation:
    """
    §2 — All exported exception classes can be instantiated.

    Uses ``inspect.signature`` to discover required positional parameters
    and provides type-appropriate dummy values so that every exception can
    be constructed without hardcoding per-class knowledge.
    """

    # Mapping of Python type annotation names → dummy values for required
    # positional parameters (no default).  The key is the annotation string
    # as it appears in inspect.Parameter.annotation.__name__ or str().
    _DUMMY_VALUES = {
        "str": "test_value",
        "int": 0,
        "float": 0.0,
        "bool": False,
        "list": [],
        "dict": {},
        "List": [],
        "Dict": {},
        "type": str,
        "Type": str,
    }

    @classmethod
    def _build_dummy_args(cls, exception_cls):
        """
        Inspect the ``__init__`` signature of *exception_cls* and return a
        list of positional dummy arguments sufficient to satisfy all
        required (no-default) parameters (excluding ``self``).
        """
        try:
            sig = inspect.signature(exception_cls.__init__)
        except (ValueError, TypeError):
            return ["test message"]

        args = []
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            # **kwargs — skip
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                continue
            # *args — skip
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                continue
            # Has a default → not required
            if param.default is not inspect.Parameter.empty:
                continue

            # Required positional parameter — provide a dummy value
            annotation = param.annotation
            if annotation is not inspect.Parameter.empty:
                ann_name = getattr(annotation, "__name__", str(annotation))
                # Handle Optional[X] → str fallback
                if "Optional" in str(ann_name):
                    args.append("test_value")
                elif ann_name in cls._DUMMY_VALUES:
                    args.append(cls._DUMMY_VALUES[ann_name])
                else:
                    args.append("test_value")
            else:
                # No annotation — assume str
                args.append("test_value")
        return args

    @pytest.mark.contract
    @pytest.mark.parametrize("name", TestSmokeExceptionExports.ALL_EXCEPTIONS)
    def test_exception_instantiation(self, milia_pkg, name):
        """
        Each exception class can be instantiated without crashing, using
        dynamically-generated dummy arguments from ``inspect.signature``.
        """
        cls = getattr(milia_pkg, name, None)
        if cls is None:
            pytest.skip(f"{name} is not available (conditional export)")

        dummy_args = self._build_dummy_args(cls)

        try:
            instance = cls(*dummy_args)
        except (TypeError, ValueError) as exc:
            pytest.fail(f"Could not instantiate {name} with dummy args {dummy_args!r}: {exc}")

        assert isinstance(instance, BaseException), (
            f"Instantiated {name} is not a BaseException subclass"
        )


class TestContractInitializePipelineReturnContract:
    """
    §2 — ``initialize_pipeline()`` return type contract.

    Documents and enforces the (logger, config, cli_manager) tuple shape.
    """

    @pytest.mark.contract
    def test_return_type_is_tuple_of_three(self, milia_pkg):
        """Return value is a 3-tuple."""
        result = milia_pkg.initialize_pipeline(
            config_path=None,
            log_level="WARNING",
            enable_plugins=False,
            validate_config=False,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3

    @pytest.mark.contract
    def test_first_element_is_logger(self, milia_pkg):
        """First element of the return tuple is a ``logging.Logger``."""
        logger, _, _ = milia_pkg.initialize_pipeline(
            config_path=None,
            log_level="WARNING",
            enable_plugins=False,
            validate_config=False,
        )
        assert isinstance(logger, logging.Logger)

    @pytest.mark.contract
    def test_second_element_is_none_without_config(self, milia_pkg):
        """Second element is ``None`` when ``config_path=None``."""
        _, config, _ = milia_pkg.initialize_pipeline(
            config_path=None,
            log_level="WARNING",
            enable_plugins=False,
            validate_config=False,
        )
        assert config is None

    @pytest.mark.contract
    def test_third_element_is_cli_manager(self, milia_pkg):
        """Third element is a ``CLIManager`` instance."""
        _, _, cli_manager = milia_pkg.initialize_pipeline(
            config_path=None,
            log_level="WARNING",
            enable_plugins=False,
            validate_config=False,
        )
        assert isinstance(cli_manager, milia_pkg.CLIManager)


# ===================================================================
# EDGE CASE & ROBUSTNESS TESTS
# ===================================================================


class TestEdgeCaseCheckDependencies:
    """Edge cases for ``check_dependencies()``."""

    @pytest.mark.smoke
    def test_check_dependencies_is_idempotent(self, milia_pkg):
        """Calling ``check_dependencies()`` twice yields identical results."""
        first = milia_pkg.check_dependencies()
        second = milia_pkg.check_dependencies()
        assert first == second

    @pytest.mark.smoke
    def test_check_dependencies_does_not_mutate_state(self, milia_pkg):
        """
        ``check_dependencies()`` does not change the module-level
        availability flags as a side effect.
        """
        before_pt = milia_pkg._POST_TRAINING_AVAILABLE
        before_dp = milia_pkg._DATA_PREPARATION_AVAILABLE
        before_tl = milia_pkg._TRANSFER_LEARNING_AVAILABLE

        milia_pkg.check_dependencies()

        assert before_pt == milia_pkg._POST_TRAINING_AVAILABLE
        assert before_dp == milia_pkg._DATA_PREPARATION_AVAILABLE
        assert before_tl == milia_pkg._TRANSFER_LEARNING_AVAILABLE


class TestEdgeCaseGetPackageInfo:
    """Edge cases for ``get_package_info()``."""

    @pytest.mark.contract
    def test_get_package_info_python_requires_is_set(self, milia_pkg):
        """``python_requires`` key is present and non-empty."""
        info = milia_pkg.get_package_info()
        assert "python_requires" in info
        assert len(info["python_requires"]) > 0

    @pytest.mark.contract
    def test_get_package_info_python_requires_format(self, milia_pkg):
        """``python_requires`` starts with a comparison operator."""
        info = milia_pkg.get_package_info()
        pr = info["python_requires"]
        assert pr.startswith(">=") or pr.startswith(">") or pr.startswith("=="), (
            f"python_requires '{pr}' should start with >=, >, or =="
        )
