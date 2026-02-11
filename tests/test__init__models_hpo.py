# tests/test__init__models_hpo.py

"""
Test Suite: milia_pipeline/models/hpo/__init__.py — Smoke Tests & Contract Tests
=================================================================================

Production-ready test suite for the MILIA Pipeline HPO (Hyperparameter
Optimization) package ``milia_pipeline/models/hpo/__init__.py``.

Covers:
    Section 1 — Smoke Tests (MILIA_Test_Recommendations.md §1.2 scope):
        - The ``milia_pipeline.models.hpo`` subpackage imports without ImportError
        - All re-exported names from the 8 submodules are accessible
        - Module-level metadata attributes (__version__, __author__) exist
        - Module initialization (logging) runs safely
        - Re-import (``importlib.reload``) is idempotent and non-crashing
        - Availability flags (OPTUNA_AVAILABLE, CALLBACKS_OPTUNA_AVAILABLE) are
          present and boolean
        - Core classes, config classes, convenience functions, exceptions,
          backends, callbacks, search spaces, analysis, transfer, and NAS
          exports are all accessible

    Section 2 — Contract Tests (MILIA_Test_Recommendations.md §2 scope):
        - ``__all__`` completeness: every name in ``__all__`` is resolvable
        - ``__all__`` consistency: every public import is listed in ``__all__``
        - ``__all__`` has no duplicates
        - Re-exported classes are classes, callables are callable
        - Configuration dataclasses are dataclasses or Pydantic BaseModel
          subclasses
        - Enums are subclasses of ``enum.Enum``
        - Exception classes are subclasses of ``Exception``
        - Protocol class uses ``@runtime_checkable``
        - Convenience functions have documented parameter signatures
        - Module utility functions return documented types
        - ``__version__`` follows semver pattern
        - Public API surface stability (minimum expected names present)
        - Backward compatibility alias (SearchSpaceConfig) is present
        - ``get_hpo_module_info()`` returns documented dict structure
        - ``check_hpo_dependencies()`` returns documented dict structure

Design:
    - Zero ``sys.modules`` pollution: all mocking via ``@patch`` decorators
      or context-manager ``patch`` calls scoped to individual tests.
    - Deterministic: no filesystem, network, or GPU access required.
    - Isolated: each test is independent; execution order is irrelevant.
    - Fast: expected < 5 s total wall-clock on CI.

Launch:
    From project root (/app/milia):
        pytest tests/test__init__models_hpo.py -v --tb=short

Markers:
    smoke     — Quick health-check tests (§1)
    contract  — Interface/contract validation tests (§2)
"""

import importlib
import inspect
import sys
import types
import enum
import logging
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure the project root is importable
# ---------------------------------------------------------------------------
# When launched via ``pytest tests/test__init__models_hpo.py`` from the
# project root (/app/milia), ``milia_pipeline`` must be on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # …/tests -> …/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def hpo_pkg():
    """
    Import and return the ``milia_pipeline.models.hpo`` package once per module.

    This fixture validates the fundamental smoke invariant: the HPO
    subpackage is importable.  If this fails, every downstream test is moot.
    """
    try:
        import milia_pipeline.models.hpo as hpo
        return hpo
    except ImportError as exc:
        pytest.fail(
            f"milia_pipeline.models.hpo could not be imported — smoke test "
            f"precondition violated: {exc}"
        )


@pytest.fixture(scope="module")
def all_names(hpo_pkg):
    """Return the ``__all__`` list from the HPO package."""
    assert hasattr(hpo_pkg, "__all__"), (
        "milia_pipeline.models.hpo.__all__ is missing — contract violation"
    )
    return list(hpo_pkg.__all__)


# ===================================================================
# SECTION 1 — SMOKE TESTS
# ===================================================================


class TestSmokeHPOPackageImport:
    """§1.2 — Verify the HPO subpackage imports without errors."""

    @pytest.mark.smoke
    def test_import_hpo_package_succeeds(self, hpo_pkg):
        """The HPO package imports without raising any exception."""
        assert hpo_pkg is not None

    @pytest.mark.smoke
    def test_hpo_package_is_a_module(self, hpo_pkg):
        """The imported object is a proper Python module."""
        assert isinstance(hpo_pkg, types.ModuleType)

    @pytest.mark.smoke
    def test_hpo_package_has_file_attribute(self, hpo_pkg):
        """The package exposes a ``__file__`` attribute (not a namespace pkg)."""
        assert hasattr(hpo_pkg, "__file__")

    @pytest.mark.smoke
    def test_hpo_package_name(self, hpo_pkg):
        """The package ``__name__`` is ``milia_pipeline.models.hpo``."""
        assert hpo_pkg.__name__ == "milia_pipeline.models.hpo"


class TestSmokeMetadataAttributes:
    """§1.2 — Verify module-level metadata attributes are present and typed."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("attr", [
        "__version__",
        "__author__",
    ])
    def test_metadata_attribute_exists(self, hpo_pkg, attr):
        """Each metadata dunder is defined on the HPO package."""
        assert hasattr(hpo_pkg, attr), f"Missing attribute: {attr}"

    @pytest.mark.smoke
    @pytest.mark.parametrize("attr", [
        "__version__",
        "__author__",
    ])
    def test_metadata_attribute_is_string(self, hpo_pkg, attr):
        """Each metadata dunder is a non-empty string."""
        value = getattr(hpo_pkg, attr)
        assert isinstance(value, str), f"{attr} should be str, got {type(value)}"
        assert len(value) > 0, f"{attr} should be non-empty"

    @pytest.mark.smoke
    def test_version_is_semver_like(self, hpo_pkg):
        """``__version__`` follows a MAJOR.MINOR.PATCH pattern."""
        version = hpo_pkg.__version__
        parts = version.split(".")
        assert len(parts) >= 2, (
            f"Version '{version}' should have at least MAJOR.MINOR components"
        )
        for part in parts:
            numeric_part = ""
            for ch in part:
                if ch.isdigit():
                    numeric_part += ch
                else:
                    break
            assert len(numeric_part) > 0, (
                f"Version component '{part}' should start with a digit"
            )


class TestSmokeCoreClassExports:
    """§1.2 — Core HPO classes are accessible."""

    CORE_CLASSES = [
        "HPOManager",
        "HPOConfig",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CORE_CLASSES)
    def test_core_class_exists(self, hpo_pkg, name):
        """Each core class is importable from the HPO package."""
        obj = getattr(hpo_pkg, name, None)
        assert obj is not None, f"Core class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CORE_CLASSES)
    def test_core_class_is_a_class(self, hpo_pkg, name):
        """Each core export is a class (not an instance or function)."""
        obj = getattr(hpo_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )


class TestSmokeConfigClassExports:
    """§1.2 — Configuration dataclass exports are accessible."""

    CONFIG_CLASSES = [
        "SearchSpaceParamConfig",
        "PrunerConfig",
        "SamplerConfig",
        "StudyConfig",
        "MultiObjectiveStudyConfig",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CONFIG_CLASSES)
    def test_config_class_exists(self, hpo_pkg, name):
        """Each configuration class is importable from the HPO package."""
        obj = getattr(hpo_pkg, name, None)
        assert obj is not None, f"Config class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CONFIG_CLASSES)
    def test_config_class_is_a_class(self, hpo_pkg, name):
        """Each configuration export is a class."""
        obj = getattr(hpo_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )


class TestSmokeConfigEnumExports:
    """§1.2 — Configuration enum exports are accessible."""

    CONFIG_ENUMS = [
        "ParamType",
        "PrunerType",
        "SamplerType",
        "OptimizationDirection",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CONFIG_ENUMS)
    def test_config_enum_exists(self, hpo_pkg, name):
        """Each configuration enum is importable from the HPO package."""
        obj = getattr(hpo_pkg, name, None)
        assert obj is not None, f"Config enum '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CONFIG_ENUMS)
    def test_config_enum_is_a_class(self, hpo_pkg, name):
        """Each configuration enum is a class (enum type)."""
        obj = getattr(hpo_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )


class TestSmokeConvenienceFunctionExports:
    """§1.2 — Convenience functions from hpo_manager.py are accessible."""

    CONVENIENCE_FUNCTIONS = [
        "is_hpo_enabled",
        "get_best_params",
        "create_hpo_manager",
        "infer_task_type",
    ]

    HELPER_FUNCTIONS = [
        "_flatten_params",
        "_extract_param_categories",
        "_run_cross_validation",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CONVENIENCE_FUNCTIONS + HELPER_FUNCTIONS)
    def test_function_exists(self, hpo_pkg, name):
        """Each convenience/helper function is present and non-None."""
        obj = getattr(hpo_pkg, name, None)
        assert obj is not None, f"Function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CONVENIENCE_FUNCTIONS + HELPER_FUNCTIONS)
    def test_function_is_callable(self, hpo_pkg, name):
        """Each convenience/helper function is callable."""
        obj = getattr(hpo_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeExceptionExports:
    """§1.2 — HPO exception classes are accessible."""

    EXCEPTION_CLASSES = [
        "HPOError",
        "HPOConfigurationError",
        "TrialFailedError",
        "StudyNotFoundError",
        "BackendError",
        "SearchSpaceError",
        "PruningError",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", EXCEPTION_CLASSES)
    def test_exception_exists(self, hpo_pkg, name):
        """Each exception class is importable from the HPO package."""
        obj = getattr(hpo_pkg, name, None)
        assert obj is not None, f"Exception '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", EXCEPTION_CLASSES)
    def test_exception_is_a_class(self, hpo_pkg, name):
        """Each exception export is a class."""
        obj = getattr(hpo_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )


class TestSmokeBackendExports:
    """§1.2 — Backend subpackage exports are accessible."""

    BACKEND_CLASSES = [
        "HPOBackendProtocol",
        "OptunaBackend",
    ]

    BACKEND_FUNCTIONS = [
        "get_backend",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", BACKEND_CLASSES)
    def test_backend_class_exists(self, hpo_pkg, name):
        """Each backend class is importable."""
        obj = getattr(hpo_pkg, name, None)
        assert obj is not None, f"Backend class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", BACKEND_FUNCTIONS)
    def test_backend_function_exists(self, hpo_pkg, name):
        """Each backend function is present and callable."""
        obj = getattr(hpo_pkg, name, None)
        assert obj is not None, f"Backend function '{name}' is None or missing"
        assert callable(obj), f"'{name}' should be callable"

    @pytest.mark.smoke
    def test_optuna_available_flag_exists(self, hpo_pkg):
        """``OPTUNA_AVAILABLE`` flag is defined on the HPO package."""
        assert hasattr(hpo_pkg, "OPTUNA_AVAILABLE"), (
            "OPTUNA_AVAILABLE flag is missing"
        )

    @pytest.mark.smoke
    def test_optuna_available_is_bool(self, hpo_pkg):
        """``OPTUNA_AVAILABLE`` is a boolean."""
        assert isinstance(hpo_pkg.OPTUNA_AVAILABLE, bool), (
            f"OPTUNA_AVAILABLE should be bool, got "
            f"{type(hpo_pkg.OPTUNA_AVAILABLE).__name__}"
        )


class TestSmokeCallbackExports:
    """§1.2 — Callback subpackage exports are accessible."""

    @pytest.mark.smoke
    def test_optuna_pruning_callback_exists(self, hpo_pkg):
        """``OptunaPruningCallback`` is importable from the HPO package."""
        obj = getattr(hpo_pkg, "OptunaPruningCallback", None)
        assert obj is not None, "OptunaPruningCallback is None or missing"

    @pytest.mark.smoke
    def test_create_hpo_callback_exists(self, hpo_pkg):
        """``create_hpo_callback`` is present and callable."""
        obj = getattr(hpo_pkg, "create_hpo_callback", None)
        assert obj is not None, "create_hpo_callback is None or missing"
        assert callable(obj), "create_hpo_callback should be callable"

    @pytest.mark.smoke
    def test_callbacks_optuna_available_exists(self, hpo_pkg):
        """``CALLBACKS_OPTUNA_AVAILABLE`` flag is defined."""
        assert hasattr(hpo_pkg, "CALLBACKS_OPTUNA_AVAILABLE"), (
            "CALLBACKS_OPTUNA_AVAILABLE flag is missing"
        )

    @pytest.mark.smoke
    def test_callbacks_optuna_available_is_bool(self, hpo_pkg):
        """``CALLBACKS_OPTUNA_AVAILABLE`` is a boolean."""
        assert isinstance(hpo_pkg.CALLBACKS_OPTUNA_AVAILABLE, bool), (
            f"CALLBACKS_OPTUNA_AVAILABLE should be bool, got "
            f"{type(hpo_pkg.CALLBACKS_OPTUNA_AVAILABLE).__name__}"
        )


class TestSmokeSearchSpaceExports:
    """§1.2 — Search space subpackage exports are accessible."""

    SEARCH_SPACE_CLASSES = [
        "SearchSpaceBuilder",
    ]

    SEARCH_SPACE_FUNCTIONS = [
        "build_search_space",
        "get_model_search_space",
        "validate_search_space",
    ]

    SEARCH_SPACE_ALIASES = [
        "SearchSpaceParamType",
        "SearchSpaceParam",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", SEARCH_SPACE_CLASSES)
    def test_search_space_class_exists(self, hpo_pkg, name):
        """Each search space class is importable."""
        obj = getattr(hpo_pkg, name, None)
        assert obj is not None, f"Search space class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", SEARCH_SPACE_FUNCTIONS)
    def test_search_space_function_exists(self, hpo_pkg, name):
        """Each search space function is present and callable."""
        obj = getattr(hpo_pkg, name, None)
        assert obj is not None, f"Search space function '{name}' is None or missing"
        assert callable(obj), f"'{name}' should be callable"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", SEARCH_SPACE_ALIASES)
    def test_search_space_alias_exists(self, hpo_pkg, name):
        """Each search space alias is present."""
        obj = getattr(hpo_pkg, name, None)
        assert obj is not None, f"Search space alias '{name}' is None or missing"


class TestSmokeAnalysisExports:
    """§1.2 — Analysis subpackage exports are accessible."""

    ANALYSIS_CLASSES = [
        "StudyAnalyzer",
        "AnalysisConfig",
    ]

    ANALYSIS_ENUMS = [
        "ImportanceMethod",
        "ExportFormat",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", ANALYSIS_CLASSES)
    def test_analysis_class_exists(self, hpo_pkg, name):
        """Each analysis class is importable."""
        obj = getattr(hpo_pkg, name, None)
        assert obj is not None, f"Analysis class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", ANALYSIS_ENUMS)
    def test_analysis_enum_exists(self, hpo_pkg, name):
        """Each analysis enum is importable."""
        obj = getattr(hpo_pkg, name, None)
        assert obj is not None, f"Analysis enum '{name}' is None or missing"


class TestSmokeTransferExports:
    """§1.2 — Transfer learning subpackage exports are accessible."""

    TRANSFER_CLASSES = [
        "HPOTransferManager",
        "MetaFeatureExtractor",
        "WarmStartStrategy",
    ]

    TRANSFER_CONFIGS = [
        "TransferConfig",
        "MetaFeatureConfig",
        "WarmStartConfig",
    ]

    TRANSFER_DATA_CLASSES = [
        "RegisteredStudyInfo",
        "TransferredTrial",
    ]

    TRANSFER_ENUMS = [
        "MetaFeatureMethod",
        "AdaptationMethod",
        "MetaFeatureCategory",
        "WarmStartMethod",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", TRANSFER_CLASSES)
    def test_transfer_class_exists(self, hpo_pkg, name):
        """Each transfer learning class is importable."""
        obj = getattr(hpo_pkg, name, None)
        assert obj is not None, f"Transfer class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", TRANSFER_CONFIGS)
    def test_transfer_config_exists(self, hpo_pkg, name):
        """Each transfer config class is importable."""
        obj = getattr(hpo_pkg, name, None)
        assert obj is not None, f"Transfer config '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", TRANSFER_DATA_CLASSES)
    def test_transfer_data_class_exists(self, hpo_pkg, name):
        """Each transfer data class is importable."""
        obj = getattr(hpo_pkg, name, None)
        assert obj is not None, f"Transfer data class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", TRANSFER_ENUMS)
    def test_transfer_enum_exists(self, hpo_pkg, name):
        """Each transfer enum is importable."""
        obj = getattr(hpo_pkg, name, None)
        assert obj is not None, f"Transfer enum '{name}' is None or missing"


class TestSmokeNASExports:
    """§1.2 — Neural Architecture Search subpackage exports are accessible."""

    NAS_ENUMS = [
        "LayerType",
        "PoolingType",
        "AggregationType",
        "ActivationType",
    ]

    NAS_CLASSES = [
        "LayerConfig",
        "GNNArchitectureSpace",
        "NASConfig",
        "NASManager",
        "HeterogeneousGNN",
    ]

    NAS_FUNCTIONS = [
        "create_gnn_search_space",
        "get_default_gnn_search_space",
        "create_nas_manager",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", NAS_ENUMS)
    def test_nas_enum_exists(self, hpo_pkg, name):
        """Each NAS enum is importable."""
        obj = getattr(hpo_pkg, name, None)
        assert obj is not None, f"NAS enum '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", NAS_CLASSES)
    def test_nas_class_exists(self, hpo_pkg, name):
        """Each NAS class is importable."""
        obj = getattr(hpo_pkg, name, None)
        assert obj is not None, f"NAS class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", NAS_FUNCTIONS)
    def test_nas_function_exists(self, hpo_pkg, name):
        """Each NAS function is present and callable."""
        obj = getattr(hpo_pkg, name, None)
        assert obj is not None, f"NAS function '{name}' is None or missing"
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeModuleUtilityFunctions:
    """§1.2 — Module-level utility functions are accessible and callable."""

    UTILITY_FUNCTIONS = [
        "get_hpo_module_info",
        "check_hpo_dependencies",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", UTILITY_FUNCTIONS)
    def test_utility_function_exists(self, hpo_pkg, name):
        """Each module utility function is present and non-None."""
        obj = getattr(hpo_pkg, name, None)
        assert obj is not None, f"Utility function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", UTILITY_FUNCTIONS)
    def test_utility_function_is_callable(self, hpo_pkg, name):
        """Each module utility function is callable."""
        obj = getattr(hpo_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeAvailabilityFlags:
    """§1.2 — Availability flags exist and are boolean."""

    BOOL_FLAGS = [
        "OPTUNA_AVAILABLE",
        "CALLBACKS_OPTUNA_AVAILABLE",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("flag", BOOL_FLAGS)
    def test_availability_flag_exists(self, hpo_pkg, flag):
        """Each availability flag is defined on the HPO package."""
        assert hasattr(hpo_pkg, flag), f"Flag '{flag}' is missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("flag", BOOL_FLAGS)
    def test_availability_flag_is_bool(self, hpo_pkg, flag):
        """Each availability flag is actually a bool."""
        value = getattr(hpo_pkg, flag)
        assert isinstance(value, bool), (
            f"Flag '{flag}' should be bool, got {type(value).__name__}"
        )

    @pytest.mark.smoke
    def test_optuna_flags_consistent(self, hpo_pkg):
        """
        ``OPTUNA_AVAILABLE`` from backends and ``CALLBACKS_OPTUNA_AVAILABLE``
        from callbacks should agree (both check for the same dependency).
        """
        backends_flag = hpo_pkg.OPTUNA_AVAILABLE
        callbacks_flag = hpo_pkg.CALLBACKS_OPTUNA_AVAILABLE
        assert backends_flag == callbacks_flag, (
            f"OPTUNA_AVAILABLE ({backends_flag}) and "
            f"CALLBACKS_OPTUNA_AVAILABLE ({callbacks_flag}) should agree"
        )


class TestSmokeBackwardCompatibility:
    """§1.2 — Backward compatibility alias is present."""

    @pytest.mark.smoke
    def test_search_space_config_alias_exists(self, hpo_pkg):
        """``SearchSpaceConfig`` alias is present on the HPO package."""
        assert hasattr(hpo_pkg, "SearchSpaceConfig"), (
            "Backward compatibility alias 'SearchSpaceConfig' is missing"
        )

    @pytest.mark.smoke
    def test_search_space_config_alias_is_correct(self, hpo_pkg):
        """
        ``SearchSpaceConfig`` is an alias for ``SearchSpaceParamConfig``.
        """
        assert hpo_pkg.SearchSpaceConfig is hpo_pkg.SearchSpaceParamConfig, (
            "SearchSpaceConfig should be an alias for SearchSpaceParamConfig"
        )


class TestSmokeModuleInitialization:
    """§1.2 — Module-level initialization runs without side-effect crashes."""

    @pytest.mark.smoke
    def test_reimport_is_idempotent(self, hpo_pkg):
        """
        Re-importing the HPO package (via ``importlib.reload``) does not
        crash.

        Validates that all module-level code (logging, availability checks)
        is safe to re-execute.
        """
        reloaded = importlib.reload(hpo_pkg)
        assert reloaded is not None
        assert hasattr(reloaded, "__version__")

    @pytest.mark.smoke
    def test_reimport_preserves_all(self, hpo_pkg):
        """
        Re-importing the HPO package preserves ``__all__``.
        """
        reloaded = importlib.reload(hpo_pkg)
        assert hasattr(reloaded, "__all__")
        assert isinstance(reloaded.__all__, list)
        assert len(reloaded.__all__) > 0

    @pytest.mark.smoke
    def test_logger_exists(self, hpo_pkg):
        """
        The HPO package defines a module-level ``logger`` attribute.
        """
        assert hasattr(hpo_pkg, "logger"), (
            "Module-level 'logger' attribute is missing"
        )
        assert isinstance(hpo_pkg.logger, logging.Logger), (
            f"'logger' should be a logging.Logger, got "
            f"{type(hpo_pkg.logger).__name__}"
        )


# ===================================================================
# SECTION 2 — CONTRACT TESTS
# ===================================================================


class TestContractAllCompleteness:
    """§2 — Every name in ``__all__`` is resolvable on the HPO package."""

    @pytest.mark.contract
    def test_all_is_a_list(self, hpo_pkg):
        """``__all__`` is a list (not a tuple or set)."""
        assert isinstance(hpo_pkg.__all__, list)

    @pytest.mark.contract
    def test_all_contains_no_duplicates(self, all_names):
        """``__all__`` has no duplicate entries."""
        seen = set()
        duplicates = []
        for name in all_names:
            if name in seen:
                duplicates.append(name)
            seen.add(name)

        assert not duplicates, (
            f"Duplicate entries in __all__: {duplicates}"
        )

    @pytest.mark.contract
    def test_every_all_entry_is_resolvable(self, hpo_pkg, all_names):
        """
        Generic sweep: every single entry in ``__all__`` must be resolvable,
        regardless of whether it is parameterized individually.
        """
        unresolvable = [
            name for name in all_names
            if not hasattr(hpo_pkg, name)
        ]
        assert not unresolvable, (
            f"Names in __all__ that are not defined on the module: {unresolvable}"
        )

    @pytest.mark.contract
    def test_all_entries_are_strings(self, all_names):
        """Every entry in ``__all__`` is a string."""
        non_strings = [
            (i, name) for i, name in enumerate(all_names)
            if not isinstance(name, str)
        ]
        assert not non_strings, (
            f"Non-string entries in __all__: {non_strings}"
        )


class TestContractAllConsistency:
    """§2 — Every public import in the HPO module is listed in ``__all__``."""

    # Names that are intentionally public-ish but NOT in __all__
    KNOWN_UNLISTED = {
        # Metadata dunders
        "__version__",
        "__author__",
        # Module-level logger
        "logger",
        # Backward compatibility alias (not in __all__ per source code)
        "SearchSpaceConfig",
        # Re-aliased names from subpackages (internal re-export aliases)
        "SearchSpaceParamType",
        "SearchSpaceParam",
        # Re-exported availability flag alias
        "CALLBACKS_OPTUNA_AVAILABLE",
        # typing imports used at module level (not part of public API)
        "TYPE_CHECKING",
        "Dict",
        "Any",
        "List",
        "Optional",
    }

    @pytest.mark.contract
    def test_public_imports_are_in_all(self, hpo_pkg, all_names):
        """
        Every non-dunder, non-private attribute that was imported (not
        defined locally) should be in ``__all__`` — unless it is in
        ``KNOWN_UNLISTED``.

        This catches accidental omissions when new imports are added to
        the HPO ``__init__.py`` without updating ``__all__``.
        """
        all_set = set(all_names)
        module_dict = vars(hpo_pkg)
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
            "__builtins__", "__cached__", "__doc__", "__file__",
            "__loader__", "__name__", "__package__", "__path__",
            "__spec__",
        }
        missing_from_all = [
            n for n in missing_from_all if n not in python_internals
        ]

        assert not missing_from_all, (
            f"Public names imported in hpo/__init__.py but not in __all__: "
            f"{sorted(missing_from_all)}"
        )


class TestContractCoreClassTypes:
    """§2 — Core classes are actual classes with expected nature."""

    @pytest.mark.contract
    def test_hpo_manager_is_class(self, hpo_pkg):
        """``HPOManager`` is a class."""
        assert inspect.isclass(hpo_pkg.HPOManager)

    @pytest.mark.contract
    def test_hpo_config_is_class(self, hpo_pkg):
        """``HPOConfig`` is a class."""
        assert inspect.isclass(hpo_pkg.HPOConfig)

    @pytest.mark.contract
    def test_hpo_config_is_dataclass_or_pydantic(self, hpo_pkg):
        """
        ``HPOConfig`` is a dataclass or Pydantic BaseModel (frozen, validated).

        Per project structure: HPOConfig is a frozen, validated dataclass.
        """
        cls = hpo_pkg.HPOConfig
        is_stdlib_dc = hasattr(cls, "__dataclass_fields__")

        try:
            from pydantic import BaseModel
            is_pydantic = issubclass(cls, BaseModel)
        except ImportError:
            is_pydantic = False

        is_pydantic_dc = hasattr(cls, "__pydantic_fields__")

        assert is_stdlib_dc or is_pydantic or is_pydantic_dc, (
            "HPOConfig should be a dataclass or Pydantic BaseModel"
        )


class TestContractConfigDataclassTypes:
    """§2 — Configuration classes are dataclasses or Pydantic BaseModel subclasses."""

    FROZEN_DATACLASSES = [
        "SearchSpaceParamConfig",
        "PrunerConfig",
        "SamplerConfig",
        "StudyConfig",
        "MultiObjectiveStudyConfig",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", FROZEN_DATACLASSES)
    def test_config_is_class(self, hpo_pkg, name):
        """Each configuration is a class."""
        obj = getattr(hpo_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", FROZEN_DATACLASSES)
    def test_config_is_dataclass_or_pydantic(self, hpo_pkg, name):
        """
        Each configuration class is a dataclass or Pydantic BaseModel.

        Per project structure: HPO config classes are frozen dataclasses
        or Pydantic V2 BaseModel subclasses.
        """
        cls = getattr(hpo_pkg, name)
        is_stdlib_dc = hasattr(cls, "__dataclass_fields__")

        try:
            from pydantic import BaseModel
            is_pydantic = issubclass(cls, BaseModel)
        except ImportError:
            is_pydantic = False

        is_pydantic_dc = hasattr(cls, "__pydantic_fields__")

        assert is_stdlib_dc or is_pydantic or is_pydantic_dc, (
            f"'{name}' should be a dataclass or Pydantic BaseModel"
        )


class TestContractEnumTypes:
    """§2 — Enum exports are subclasses of ``enum.Enum``."""

    # All enums from hpo_config.py
    HPO_CONFIG_ENUMS = [
        "ParamType",
        "PrunerType",
        "SamplerType",
        "OptimizationDirection",
    ]

    # Analysis enums
    ANALYSIS_ENUMS = [
        "ImportanceMethod",
        "ExportFormat",
    ]

    # Transfer enums
    TRANSFER_ENUMS = [
        "MetaFeatureMethod",
        "AdaptationMethod",
        "MetaFeatureCategory",
        "WarmStartMethod",
    ]

    # NAS enums
    NAS_ENUMS = [
        "LayerType",
        "PoolingType",
        "AggregationType",
        "ActivationType",
    ]

    ALL_ENUMS = (
        HPO_CONFIG_ENUMS
        + ANALYSIS_ENUMS
        + TRANSFER_ENUMS
        + NAS_ENUMS
    )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", ALL_ENUMS)
    def test_enum_is_class(self, hpo_pkg, name):
        """Each enum export is a class."""
        obj = getattr(hpo_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", ALL_ENUMS)
    def test_enum_is_enum_subclass(self, hpo_pkg, name):
        """Each enum export is a subclass of ``enum.Enum``."""
        obj = getattr(hpo_pkg, name)
        assert issubclass(obj, enum.Enum), (
            f"'{name}' should be a subclass of enum.Enum, got MRO: "
            f"{[c.__name__ for c in obj.__mro__]}"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", ALL_ENUMS)
    def test_enum_has_members(self, hpo_pkg, name):
        """Each enum has at least one member."""
        obj = getattr(hpo_pkg, name)
        members = list(obj)
        assert len(members) > 0, (
            f"'{name}' should have at least one member"
        )


class TestContractEnumMemberCounts:
    """§2 — Enum member counts match project structure documentation."""

    @pytest.mark.contract
    def test_param_type_has_expected_members(self, hpo_pkg):
        """
        ``ParamType`` has 7 types per project structure:
        INT, FLOAT, CATEGORICAL, LOGUNIFORM, UNIFORM, INT_UNIFORM, DISCRETE_UNIFORM.
        """
        members = list(hpo_pkg.ParamType)
        assert len(members) == 7, (
            f"ParamType should have 7 members, got {len(members)}: "
            f"{[m.name for m in members]}"
        )

    @pytest.mark.contract
    def test_pruner_type_has_expected_members(self, hpo_pkg):
        """
        ``PrunerType`` has 7 types per project structure:
        MEDIAN, PERCENTILE, HYPERBAND, SUCCESSIVE_HALVING, THRESHOLD,
        PATIENT, NONE.
        """
        members = list(hpo_pkg.PrunerType)
        assert len(members) == 7, (
            f"PrunerType should have 7 members, got {len(members)}: "
            f"{[m.name for m in members]}"
        )

    @pytest.mark.contract
    def test_sampler_type_has_expected_members(self, hpo_pkg):
        """
        ``SamplerType`` has 7 types per project structure:
        TPE, RANDOM, CMAES, GRID, NSGAII, MOTPE, QMCSAMPLER.
        """
        members = list(hpo_pkg.SamplerType)
        assert len(members) == 7, (
            f"SamplerType should have 7 members, got {len(members)}: "
            f"{[m.name for m in members]}"
        )

    @pytest.mark.contract
    def test_optimization_direction_has_expected_members(self, hpo_pkg):
        """
        ``OptimizationDirection`` has 2 types: MINIMIZE, MAXIMIZE.
        """
        members = list(hpo_pkg.OptimizationDirection)
        assert len(members) == 2, (
            f"OptimizationDirection should have 2 members, got {len(members)}: "
            f"{[m.name for m in members]}"
        )

    @pytest.mark.contract
    def test_layer_type_has_expected_members(self, hpo_pkg):
        """
        ``LayerType`` has 7 types per project structure:
        GCN, GAT, SAGE, GIN, GATV2, TRANSFORMER, PNA.
        """
        members = list(hpo_pkg.LayerType)
        assert len(members) == 7, (
            f"LayerType should have 7 members, got {len(members)}: "
            f"{[m.name for m in members]}"
        )

    @pytest.mark.contract
    def test_pooling_type_has_expected_members(self, hpo_pkg):
        """
        ``PoolingType`` has 6 types per project structure:
        MEAN, MAX, SUM, ATTENTION, SET2SET, TOPK.
        """
        members = list(hpo_pkg.PoolingType)
        assert len(members) == 6, (
            f"PoolingType should have 6 members, got {len(members)}: "
            f"{[m.name for m in members]}"
        )

    @pytest.mark.contract
    def test_aggregation_type_has_expected_members(self, hpo_pkg):
        """
        ``AggregationType`` has 5 types per project structure:
        MEAN, MAX, SUM, LSTM, MULTI.
        """
        members = list(hpo_pkg.AggregationType)
        assert len(members) == 5, (
            f"AggregationType should have 5 members, got {len(members)}: "
            f"{[m.name for m in members]}"
        )

    @pytest.mark.contract
    def test_activation_type_has_expected_members(self, hpo_pkg):
        """
        ``ActivationType`` has 7 types per project structure:
        RELU, GELU, ELU, LEAKY_RELU, SILU, TANH, PRELU.
        """
        members = list(hpo_pkg.ActivationType)
        assert len(members) == 7, (
            f"ActivationType should have 7 members, got {len(members)}: "
            f"{[m.name for m in members]}"
        )

    @pytest.mark.contract
    def test_importance_method_has_expected_members(self, hpo_pkg):
        """
        ``ImportanceMethod`` has 2 types per project structure: FANOVA, MDI.
        """
        members = list(hpo_pkg.ImportanceMethod)
        assert len(members) == 2, (
            f"ImportanceMethod should have 2 members, got {len(members)}: "
            f"{[m.name for m in members]}"
        )

    @pytest.mark.contract
    def test_export_format_has_expected_members(self, hpo_pkg):
        """
        ``ExportFormat`` has 4 types per project structure:
        JSON, CSV, DATAFRAME, DICT.
        """
        members = list(hpo_pkg.ExportFormat)
        assert len(members) == 4, (
            f"ExportFormat should have 4 members, got {len(members)}: "
            f"{[m.name for m in members]}"
        )

    @pytest.mark.contract
    def test_meta_feature_method_has_expected_members(self, hpo_pkg):
        """
        ``MetaFeatureMethod`` has 3 types per project structure:
        STATISTICAL, LEARNED, LANDMARK.
        """
        members = list(hpo_pkg.MetaFeatureMethod)
        assert len(members) == 3, (
            f"MetaFeatureMethod should have 3 members, got {len(members)}: "
            f"{[m.name for m in members]}"
        )

    @pytest.mark.contract
    def test_adaptation_method_has_expected_members(self, hpo_pkg):
        """
        ``AdaptationMethod`` has 4 types per project structure:
        WEIGHTED, FILTERED, FULL, ADAPTIVE.
        """
        members = list(hpo_pkg.AdaptationMethod)
        assert len(members) == 4, (
            f"AdaptationMethod should have 4 members, got {len(members)}: "
            f"{[m.name for m in members]}"
        )

    @pytest.mark.contract
    def test_meta_feature_category_has_expected_members(self, hpo_pkg):
        """
        ``MetaFeatureCategory`` has 7 types per project structure:
        STATISTICAL, GRAPH, MOLECULAR, TARGET, NODE_FEATURES,
        EDGE_FEATURES, ALL.
        """
        members = list(hpo_pkg.MetaFeatureCategory)
        assert len(members) == 7, (
            f"MetaFeatureCategory should have 7 members, got {len(members)}: "
            f"{[m.name for m in members]}"
        )

    @pytest.mark.contract
    def test_warm_start_method_has_expected_members(self, hpo_pkg):
        """
        ``WarmStartMethod`` has 4 types per project structure:
        WEIGHTED, FILTERED, FULL, ADAPTIVE.
        """
        members = list(hpo_pkg.WarmStartMethod)
        assert len(members) == 4, (
            f"WarmStartMethod should have 4 members, got {len(members)}: "
            f"{[m.name for m in members]}"
        )


class TestContractExceptionHierarchy:
    """§2 — Exception classes are proper Exception subclasses."""

    EXCEPTION_CLASSES = [
        "HPOError",
        "HPOConfigurationError",
        "TrialFailedError",
        "StudyNotFoundError",
        "BackendError",
        "SearchSpaceError",
        "PruningError",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", EXCEPTION_CLASSES)
    def test_exception_is_exception_subclass(self, hpo_pkg, name):
        """Each exception is a subclass of ``Exception``."""
        cls = getattr(hpo_pkg, name)
        assert issubclass(cls, Exception), (
            f"'{name}' should be a subclass of Exception"
        )

    @pytest.mark.contract
    def test_hpo_error_is_base(self, hpo_pkg):
        """
        ``HPOError`` is the base HPO exception. All other HPO exceptions
        should be its subclasses.
        """
        base = hpo_pkg.HPOError
        derived = [
            "HPOConfigurationError",
            "TrialFailedError",
            "StudyNotFoundError",
            "BackendError",
            "SearchSpaceError",
            "PruningError",
        ]
        for name in derived:
            cls = getattr(hpo_pkg, name)
            assert issubclass(cls, base), (
                f"'{name}' should be a subclass of HPOError"
            )


class TestContractBackendProtocol:
    """§2 — HPOBackendProtocol is a runtime-checkable Protocol."""

    @pytest.mark.contract
    def test_backend_protocol_is_class(self, hpo_pkg):
        """``HPOBackendProtocol`` is a class."""
        assert inspect.isclass(hpo_pkg.HPOBackendProtocol)

    @pytest.mark.contract
    def test_backend_protocol_is_runtime_checkable(self, hpo_pkg):
        """
        ``HPOBackendProtocol`` is decorated with ``@runtime_checkable``.

        Per project structure: HPOBackendProtocol is a @runtime_checkable
        Protocol with 7 methods.
        """
        protocol = hpo_pkg.HPOBackendProtocol
        # runtime_checkable protocols have _is_runtime_protocol attribute
        is_runtime = getattr(protocol, "_is_runtime_protocol", False)
        assert is_runtime, (
            "HPOBackendProtocol should be @runtime_checkable"
        )

    @pytest.mark.contract
    def test_optuna_backend_is_class(self, hpo_pkg):
        """``OptunaBackend`` is a class."""
        assert inspect.isclass(hpo_pkg.OptunaBackend)


class TestContractConvenienceFunctionSignatures:
    """§2 — Convenience functions have documented parameter signatures."""

    @pytest.mark.contract
    def test_is_hpo_enabled_accepts_config(self, hpo_pkg):
        """``is_hpo_enabled`` accepts at least one parameter (config)."""
        sig = inspect.signature(hpo_pkg.is_hpo_enabled)
        assert len(sig.parameters) >= 1, (
            "is_hpo_enabled should accept at least one parameter"
        )

    @pytest.mark.contract
    def test_create_hpo_manager_is_callable(self, hpo_pkg):
        """``create_hpo_manager`` is callable with parameters."""
        sig = inspect.signature(hpo_pkg.create_hpo_manager)
        assert len(sig.parameters) >= 1, (
            "create_hpo_manager should accept parameters"
        )

    @pytest.mark.contract
    def test_get_best_params_is_callable(self, hpo_pkg):
        """``get_best_params`` accepts at least one parameter."""
        sig = inspect.signature(hpo_pkg.get_best_params)
        assert len(sig.parameters) >= 1, (
            "get_best_params should accept at least one parameter"
        )

    @pytest.mark.contract
    def test_infer_task_type_is_callable(self, hpo_pkg):
        """``infer_task_type`` is callable."""
        assert callable(hpo_pkg.infer_task_type)
        sig = inspect.signature(hpo_pkg.infer_task_type)
        assert len(sig.parameters) >= 1, (
            "infer_task_type should accept at least one parameter"
        )

    @pytest.mark.contract
    def test_is_hpo_enabled_is_function(self, hpo_pkg):
        """``is_hpo_enabled`` is a function (not a class)."""
        assert inspect.isfunction(hpo_pkg.is_hpo_enabled)

    @pytest.mark.contract
    def test_get_best_params_is_function(self, hpo_pkg):
        """``get_best_params`` is a function (not a class)."""
        assert inspect.isfunction(hpo_pkg.get_best_params)

    @pytest.mark.contract
    def test_create_hpo_manager_is_function(self, hpo_pkg):
        """``create_hpo_manager`` is a function."""
        assert inspect.isfunction(hpo_pkg.create_hpo_manager)

    @pytest.mark.contract
    def test_infer_task_type_is_function(self, hpo_pkg):
        """``infer_task_type`` is a function."""
        assert inspect.isfunction(hpo_pkg.infer_task_type)


class TestContractHelperFunctionTypes:
    """§2 — Internal helper functions are functions."""

    HELPERS = [
        "_flatten_params",
        "_extract_param_categories",
        "_run_cross_validation",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", HELPERS)
    def test_helper_is_function(self, hpo_pkg, name):
        """Each helper is a function (not a class or bound method)."""
        obj = getattr(hpo_pkg, name)
        assert inspect.isfunction(obj), (
            f"'{name}' should be a function, got {type(obj).__name__}"
        )


class TestContractSearchSpaceFunctionSignatures:
    """§2 — Search space functions are callable with documented signatures."""

    @pytest.mark.contract
    def test_build_search_space_is_function(self, hpo_pkg):
        """``build_search_space`` is a function."""
        assert inspect.isfunction(hpo_pkg.build_search_space)

    @pytest.mark.contract
    def test_get_model_search_space_is_function(self, hpo_pkg):
        """``get_model_search_space`` is a function."""
        assert inspect.isfunction(hpo_pkg.get_model_search_space)

    @pytest.mark.contract
    def test_validate_search_space_is_function(self, hpo_pkg):
        """``validate_search_space`` is a function."""
        assert inspect.isfunction(hpo_pkg.validate_search_space)

    @pytest.mark.contract
    def test_search_space_builder_is_class(self, hpo_pkg):
        """``SearchSpaceBuilder`` is a class."""
        assert inspect.isclass(hpo_pkg.SearchSpaceBuilder)


class TestContractNASClassTypes:
    """§2 — NAS classes have expected types."""

    @pytest.mark.contract
    def test_nas_manager_is_class(self, hpo_pkg):
        """``NASManager`` is a class."""
        assert inspect.isclass(hpo_pkg.NASManager)

    @pytest.mark.contract
    def test_gnn_architecture_space_is_class(self, hpo_pkg):
        """``GNNArchitectureSpace`` is a class."""
        assert inspect.isclass(hpo_pkg.GNNArchitectureSpace)

    @pytest.mark.contract
    def test_heterogeneous_gnn_is_class(self, hpo_pkg):
        """``HeterogeneousGNN`` is a class (nn.Module)."""
        assert inspect.isclass(hpo_pkg.HeterogeneousGNN)

    @pytest.mark.contract
    def test_layer_config_is_dataclass_or_pydantic(self, hpo_pkg):
        """
        ``LayerConfig`` is a frozen dataclass per project structure.
        """
        cls = hpo_pkg.LayerConfig
        is_stdlib_dc = hasattr(cls, "__dataclass_fields__")

        try:
            from pydantic import BaseModel
            is_pydantic = issubclass(cls, BaseModel)
        except ImportError:
            is_pydantic = False

        is_pydantic_dc = hasattr(cls, "__pydantic_fields__")

        assert is_stdlib_dc or is_pydantic or is_pydantic_dc, (
            "LayerConfig should be a dataclass or Pydantic BaseModel"
        )

    @pytest.mark.contract
    def test_nas_config_is_dataclass_or_pydantic(self, hpo_pkg):
        """
        ``NASConfig`` is a dataclass per project structure.
        """
        cls = hpo_pkg.NASConfig
        is_stdlib_dc = hasattr(cls, "__dataclass_fields__")

        try:
            from pydantic import BaseModel
            is_pydantic = issubclass(cls, BaseModel)
        except ImportError:
            is_pydantic = False

        is_pydantic_dc = hasattr(cls, "__pydantic_fields__")

        assert is_stdlib_dc or is_pydantic or is_pydantic_dc, (
            "NASConfig should be a dataclass or Pydantic BaseModel"
        )

    @pytest.mark.contract
    def test_create_gnn_search_space_is_function(self, hpo_pkg):
        """``create_gnn_search_space`` is a function."""
        assert inspect.isfunction(hpo_pkg.create_gnn_search_space)

    @pytest.mark.contract
    def test_get_default_gnn_search_space_is_function(self, hpo_pkg):
        """``get_default_gnn_search_space`` is a function."""
        assert inspect.isfunction(hpo_pkg.get_default_gnn_search_space)

    @pytest.mark.contract
    def test_create_nas_manager_is_function(self, hpo_pkg):
        """``create_nas_manager`` is a function."""
        assert inspect.isfunction(hpo_pkg.create_nas_manager)


class TestContractAnalysisClassTypes:
    """§2 — Analysis classes have expected types."""

    @pytest.mark.contract
    def test_study_analyzer_is_class(self, hpo_pkg):
        """``StudyAnalyzer`` is a class."""
        assert inspect.isclass(hpo_pkg.StudyAnalyzer)

    @pytest.mark.contract
    def test_analysis_config_is_dataclass_or_pydantic(self, hpo_pkg):
        """
        ``AnalysisConfig`` is a frozen dataclass per project structure.
        """
        cls = hpo_pkg.AnalysisConfig
        is_stdlib_dc = hasattr(cls, "__dataclass_fields__")

        try:
            from pydantic import BaseModel
            is_pydantic = issubclass(cls, BaseModel)
        except ImportError:
            is_pydantic = False

        is_pydantic_dc = hasattr(cls, "__pydantic_fields__")

        assert is_stdlib_dc or is_pydantic or is_pydantic_dc, (
            "AnalysisConfig should be a dataclass or Pydantic BaseModel"
        )


class TestContractTransferClassTypes:
    """§2 — Transfer learning classes have expected types."""

    TRANSFER_MAIN_CLASSES = [
        "HPOTransferManager",
        "MetaFeatureExtractor",
        "WarmStartStrategy",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", TRANSFER_MAIN_CLASSES)
    def test_transfer_class_is_class(self, hpo_pkg, name):
        """Each transfer learning class is a class."""
        obj = getattr(hpo_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )

    @pytest.mark.contract
    def test_transfer_config_is_pydantic_basemodel(self, hpo_pkg):
        """
        ``TransferConfig`` is a Pydantic V2 BaseModel (frozen) per project
        structure documentation.
        """
        cls = hpo_pkg.TransferConfig
        try:
            from pydantic import BaseModel
            is_pydantic = issubclass(cls, BaseModel)
        except ImportError:
            is_pydantic = False

        is_pydantic_dc = hasattr(cls, "__pydantic_fields__")
        is_stdlib_dc = hasattr(cls, "__dataclass_fields__")

        assert is_pydantic or is_pydantic_dc or is_stdlib_dc, (
            "TransferConfig should be a Pydantic BaseModel or dataclass"
        )

    @pytest.mark.contract
    def test_meta_feature_config_is_pydantic_or_dataclass(self, hpo_pkg):
        """
        ``MetaFeatureConfig`` is a frozen BaseModel or dataclass per project
        structure.
        """
        cls = hpo_pkg.MetaFeatureConfig
        try:
            from pydantic import BaseModel
            is_pydantic = issubclass(cls, BaseModel)
        except ImportError:
            is_pydantic = False

        is_pydantic_dc = hasattr(cls, "__pydantic_fields__")
        is_stdlib_dc = hasattr(cls, "__dataclass_fields__")

        assert is_pydantic or is_pydantic_dc or is_stdlib_dc, (
            "MetaFeatureConfig should be a Pydantic BaseModel or dataclass"
        )

    @pytest.mark.contract
    def test_warm_start_config_is_dataclass(self, hpo_pkg):
        """
        ``WarmStartConfig`` is a frozen dataclass per project structure.
        """
        cls = hpo_pkg.WarmStartConfig
        is_stdlib_dc = hasattr(cls, "__dataclass_fields__")

        try:
            from pydantic import BaseModel
            is_pydantic = issubclass(cls, BaseModel)
        except ImportError:
            is_pydantic = False

        is_pydantic_dc = hasattr(cls, "__pydantic_fields__")

        assert is_stdlib_dc or is_pydantic or is_pydantic_dc, (
            "WarmStartConfig should be a dataclass or Pydantic BaseModel"
        )

    @pytest.mark.contract
    def test_registered_study_info_is_pydantic_or_dataclass(self, hpo_pkg):
        """
        ``RegisteredStudyInfo`` is a mutable Pydantic V2 BaseModel per
        project structure.
        """
        cls = hpo_pkg.RegisteredStudyInfo
        try:
            from pydantic import BaseModel
            is_pydantic = issubclass(cls, BaseModel)
        except ImportError:
            is_pydantic = False

        is_pydantic_dc = hasattr(cls, "__pydantic_fields__")
        is_stdlib_dc = hasattr(cls, "__dataclass_fields__")

        assert is_pydantic or is_pydantic_dc or is_stdlib_dc, (
            "RegisteredStudyInfo should be a Pydantic BaseModel or dataclass"
        )

    @pytest.mark.contract
    def test_transferred_trial_is_dataclass(self, hpo_pkg):
        """
        ``TransferredTrial`` is a dataclass per project structure.
        """
        cls = hpo_pkg.TransferredTrial
        is_stdlib_dc = hasattr(cls, "__dataclass_fields__")

        try:
            from pydantic import BaseModel
            is_pydantic = issubclass(cls, BaseModel)
        except ImportError:
            is_pydantic = False

        is_pydantic_dc = hasattr(cls, "__pydantic_fields__")

        assert is_stdlib_dc or is_pydantic or is_pydantic_dc, (
            "TransferredTrial should be a dataclass or Pydantic BaseModel"
        )


class TestContractGetHPOModuleInfoReturnType:
    """§2 — ``get_hpo_module_info()`` return type and structure contract."""

    @pytest.mark.contract
    def test_returns_dict(self, hpo_pkg):
        """``get_hpo_module_info()`` returns a dict."""
        result = hpo_pkg.get_hpo_module_info()
        assert isinstance(result, dict), (
            f"get_hpo_module_info() should return dict, got "
            f"{type(result).__name__}"
        )

    @pytest.mark.contract
    def test_has_version_key(self, hpo_pkg):
        """Result includes 'version' key."""
        result = hpo_pkg.get_hpo_module_info()
        assert "version" in result, "Missing 'version' key"
        assert isinstance(result["version"], str)

    @pytest.mark.contract
    def test_has_author_key(self, hpo_pkg):
        """Result includes 'author' key."""
        result = hpo_pkg.get_hpo_module_info()
        assert "author" in result, "Missing 'author' key"
        assert isinstance(result["author"], str)

    @pytest.mark.contract
    def test_has_module_key(self, hpo_pkg):
        """Result includes 'module' key."""
        result = hpo_pkg.get_hpo_module_info()
        assert "module" in result, "Missing 'module' key"
        assert result["module"] == "milia_pipeline.models.hpo"

    @pytest.mark.contract
    def test_has_backends_key(self, hpo_pkg):
        """Result includes 'backends' key as a list."""
        result = hpo_pkg.get_hpo_module_info()
        assert "backends" in result, "Missing 'backends' key"
        assert isinstance(result["backends"], list)
        assert len(result["backends"]) > 0

    @pytest.mark.contract
    def test_has_primary_backend_key(self, hpo_pkg):
        """Result includes 'primary_backend' key with value 'optuna'."""
        result = hpo_pkg.get_hpo_module_info()
        assert "primary_backend" in result, "Missing 'primary_backend' key"
        assert result["primary_backend"] == "optuna"

    @pytest.mark.contract
    def test_has_optuna_available_key(self, hpo_pkg):
        """Result includes 'optuna_available' key as a bool."""
        result = hpo_pkg.get_hpo_module_info()
        assert "optuna_available" in result, "Missing 'optuna_available' key"
        assert isinstance(result["optuna_available"], bool)

    @pytest.mark.contract
    def test_has_subpackages_key(self, hpo_pkg):
        """Result includes 'subpackages' key as a list with expected entries."""
        result = hpo_pkg.get_hpo_module_info()
        assert "subpackages" in result, "Missing 'subpackages' key"
        subpackages = result["subpackages"]
        assert isinstance(subpackages, list)
        expected = {"backends", "callbacks", "search_spaces", "analysis",
                    "transfer", "nas"}
        actual = set(subpackages)
        missing = expected - actual
        assert not missing, (
            f"Missing subpackages: {missing}"
        )

    @pytest.mark.contract
    def test_has_exports_key(self, hpo_pkg):
        """Result includes 'exports' key matching len(__all__)."""
        result = hpo_pkg.get_hpo_module_info()
        assert "exports" in result, "Missing 'exports' key"
        assert isinstance(result["exports"], int)
        assert result["exports"] == len(hpo_pkg.__all__)

    @pytest.mark.contract
    def test_has_components_key(self, hpo_pkg):
        """Result includes 'components' key as a dict."""
        result = hpo_pkg.get_hpo_module_info()
        assert "components" in result, "Missing 'components' key"
        assert isinstance(result["components"], dict)
        # Verify expected component categories
        components = result["components"]
        expected_categories = {
            "core", "configurations", "convenience_functions",
            "exceptions", "backends", "callbacks", "search_spaces",
            "analysis", "transfer", "nas",
        }
        actual_categories = set(components.keys())
        missing = expected_categories - actual_categories
        assert not missing, (
            f"Missing component categories: {missing}"
        )

    @pytest.mark.contract
    def test_has_description_key(self, hpo_pkg):
        """Result includes 'description' key as a non-empty string."""
        result = hpo_pkg.get_hpo_module_info()
        assert "description" in result, "Missing 'description' key"
        assert isinstance(result["description"], str)
        assert len(result["description"]) > 0


class TestContractCheckHPODependenciesReturnType:
    """§2 — ``check_hpo_dependencies()`` return type and structure contract."""

    @pytest.mark.contract
    def test_returns_dict(self, hpo_pkg):
        """``check_hpo_dependencies()`` returns a dict."""
        result = hpo_pkg.check_hpo_dependencies()
        assert isinstance(result, dict), (
            f"check_hpo_dependencies() should return dict, got "
            f"{type(result).__name__}"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("dep", [
        "optuna",
        "ray_tune",
        "torch",
        "torch_geometric",
        "numpy",
        "scikit_learn",
    ])
    def test_dependency_entry_has_expected_structure(self, hpo_pkg, dep):
        """Each dependency entry has 'available' (bool) and 'version' keys."""
        result = hpo_pkg.check_hpo_dependencies()
        assert dep in result, f"Missing dependency key: {dep}"
        entry = result[dep]
        assert isinstance(entry, dict), (
            f"Dependency '{dep}' entry should be a dict"
        )
        assert "available" in entry, (
            f"Dependency '{dep}' missing 'available' key"
        )
        assert isinstance(entry["available"], bool), (
            f"Dependency '{dep}' 'available' should be bool"
        )
        assert "version" in entry, (
            f"Dependency '{dep}' missing 'version' key"
        )
        # version is a string when available, None otherwise
        if entry["available"]:
            assert isinstance(entry["version"], str), (
                f"Dependency '{dep}' version should be str when available"
            )
        else:
            assert entry["version"] is None, (
                f"Dependency '{dep}' version should be None when not available"
            )

    @pytest.mark.contract
    def test_has_all_required_available_key(self, hpo_pkg):
        """Result includes 'all_required_available' summary flag."""
        result = hpo_pkg.check_hpo_dependencies()
        assert "all_required_available" in result
        assert isinstance(result["all_required_available"], bool)

    @pytest.mark.contract
    def test_has_all_optional_available_key(self, hpo_pkg):
        """Result includes 'all_optional_available' summary flag."""
        result = hpo_pkg.check_hpo_dependencies()
        assert "all_optional_available" in result
        assert isinstance(result["all_optional_available"], bool)

    @pytest.mark.contract
    def test_has_nas_available_key(self, hpo_pkg):
        """Result includes 'nas_available' summary flag."""
        result = hpo_pkg.check_hpo_dependencies()
        assert "nas_available" in result
        assert isinstance(result["nas_available"], bool)

    @pytest.mark.contract
    def test_required_deps_consistency(self, hpo_pkg):
        """
        ``all_required_available`` is True iff optuna, torch, and numpy
        are all available.
        """
        result = hpo_pkg.check_hpo_dependencies()
        expected = all([
            result["optuna"]["available"],
            result["torch"]["available"],
            result["numpy"]["available"],
        ])
        assert result["all_required_available"] == expected, (
            f"all_required_available ({result['all_required_available']}) "
            f"should equal all([optuna={result['optuna']['available']}, "
            f"torch={result['torch']['available']}, "
            f"numpy={result['numpy']['available']}]) = {expected}"
        )

    @pytest.mark.contract
    def test_nas_deps_consistency(self, hpo_pkg):
        """
        ``nas_available`` is True iff torch, torch_geometric, and optuna
        are all available.
        """
        result = hpo_pkg.check_hpo_dependencies()
        expected = all([
            result["torch"]["available"],
            result["torch_geometric"]["available"],
            result["optuna"]["available"],
        ])
        assert result["nas_available"] == expected, (
            f"nas_available ({result['nas_available']}) "
            f"should equal all([torch={result['torch']['available']}, "
            f"torch_geometric={result['torch_geometric']['available']}, "
            f"optuna={result['optuna']['available']}]) = {expected}"
        )


class TestContractPublicAPISurface:
    """
    §2 — Public API surface stability.

    Ensures that the minimum expected public API is present in ``__all__``.
    Guards against accidental removals during refactoring.
    """

    # The minimum API surface that MUST be present in __all__
    MINIMUM_API = {
        # Core classes
        "HPOManager",
        "HPOConfig",
        # Configuration
        "SearchSpaceParamConfig",
        "PrunerConfig",
        "SamplerConfig",
        "StudyConfig",
        "MultiObjectiveStudyConfig",
        # Config enums
        "ParamType",
        "PrunerType",
        "SamplerType",
        "OptimizationDirection",
        # Convenience functions
        "is_hpo_enabled",
        "get_best_params",
        "create_hpo_manager",
        "infer_task_type",
        # Exceptions
        "HPOError",
        "HPOConfigurationError",
        "TrialFailedError",
        "StudyNotFoundError",
        "BackendError",
        "SearchSpaceError",
        "PruningError",
        # Backends
        "HPOBackendProtocol",
        "OptunaBackend",
        "get_backend",
        "OPTUNA_AVAILABLE",
        # Callbacks
        "OptunaPruningCallback",
        "create_hpo_callback",
        # Search spaces
        "SearchSpaceBuilder",
        "build_search_space",
        "get_model_search_space",
        "validate_search_space",
        # Analysis
        "StudyAnalyzer",
        "AnalysisConfig",
        "ImportanceMethod",
        "ExportFormat",
        # Transfer
        "HPOTransferManager",
        "MetaFeatureExtractor",
        "WarmStartStrategy",
        "TransferConfig",
        # NAS
        "NASManager",
        "GNNArchitectureSpace",
        "LayerType",
        "PoolingType",
        # Module utilities
        "get_hpo_module_info",
        "check_hpo_dependencies",
    }

    @pytest.mark.contract
    def test_minimum_api_in_all(self, all_names):
        """The minimum expected public API is present in ``__all__``."""
        all_set = set(all_names)
        missing = self.MINIMUM_API - all_set
        assert not missing, (
            f"Minimum API names missing from __all__: {sorted(missing)}"
        )

    @pytest.mark.contract
    def test_all_has_expected_length(self, all_names):
        """
        ``__all__`` contains a substantial number of entries.

        Based on the __init__.py source, the HPO package exports 60+
        names. This test guards against catastrophic loss (e.g., accidental
        truncation of __all__) while allowing for organic growth.
        """
        actual = len(all_names)
        # The __init__.py has ~60 entries in __all__ (lines 325-439)
        # We set a floor below the actual count to allow changes
        # while catching catastrophic loss.
        MINIMUM_EXPECTED = 50
        assert actual >= MINIMUM_EXPECTED, (
            f"__all__ has {actual} entries, expected at least {MINIMUM_EXPECTED}. "
            f"This suggests __all__ may have been accidentally truncated."
        )


class TestContractCallbackClassType:
    """§2 — OptunaPruningCallback is a class."""

    @pytest.mark.contract
    def test_optuna_pruning_callback_is_class(self, hpo_pkg):
        """``OptunaPruningCallback`` is a class."""
        assert inspect.isclass(hpo_pkg.OptunaPruningCallback)

    @pytest.mark.contract
    def test_create_hpo_callback_is_function(self, hpo_pkg):
        """``create_hpo_callback`` is a function."""
        assert inspect.isfunction(hpo_pkg.create_hpo_callback)


class TestContractGetBackendSignature:
    """§2 — ``get_backend`` factory function has expected signature."""

    @pytest.mark.contract
    def test_get_backend_is_function(self, hpo_pkg):
        """``get_backend`` is a function."""
        assert inspect.isfunction(hpo_pkg.get_backend)

    @pytest.mark.contract
    def test_get_backend_accepts_name_param(self, hpo_pkg):
        """``get_backend`` accepts at least one parameter (backend name)."""
        sig = inspect.signature(hpo_pkg.get_backend)
        assert len(sig.parameters) >= 1, (
            "get_backend should accept at least one parameter"
        )


class TestContractVersionConsistency:
    """§2 — Version consistency between module and info function."""

    @pytest.mark.contract
    def test_version_matches_info(self, hpo_pkg):
        """
        ``__version__`` matches the version returned by
        ``get_hpo_module_info()``.
        """
        info = hpo_pkg.get_hpo_module_info()
        assert hpo_pkg.__version__ == info["version"], (
            f"__version__ ({hpo_pkg.__version__}) does not match "
            f"get_hpo_module_info()['version'] ({info['version']})"
        )

    @pytest.mark.contract
    def test_optuna_available_matches_info(self, hpo_pkg):
        """
        ``OPTUNA_AVAILABLE`` matches the value in ``get_hpo_module_info()``.
        """
        info = hpo_pkg.get_hpo_module_info()
        assert hpo_pkg.OPTUNA_AVAILABLE == info["optuna_available"], (
            f"OPTUNA_AVAILABLE ({hpo_pkg.OPTUNA_AVAILABLE}) does not match "
            f"get_hpo_module_info()['optuna_available'] "
            f"({info['optuna_available']})"
        )


class TestContractSearchSpaceAliasConsistency:
    """§2 — Search space alias imports are consistent."""

    @pytest.mark.contract
    def test_search_space_param_type_is_param_type(self, hpo_pkg):
        """
        ``SearchSpaceParamType`` (alias from search_spaces) is the same
        object as ``ParamType`` (from hpo_config).
        """
        assert hpo_pkg.SearchSpaceParamType is hpo_pkg.ParamType, (
            "SearchSpaceParamType should be the same as ParamType"
        )

    @pytest.mark.contract
    def test_search_space_param_is_search_space_param_config(self, hpo_pkg):
        """
        ``SearchSpaceParam`` (alias from search_spaces) is the same
        object as ``SearchSpaceParamConfig`` (from hpo_config).
        """
        assert hpo_pkg.SearchSpaceParam is hpo_pkg.SearchSpaceParamConfig, (
            "SearchSpaceParam should be the same as SearchSpaceParamConfig"
        )
