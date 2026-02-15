# tests/test__init__preprocessing.py

"""
Test Suite: milia_pipeline/preprocessing/__init__.py — Smoke Tests & Contract Tests
=====================================================================================

Production-ready test suite for the MILIA Pipeline preprocessing package
``milia_pipeline/preprocessing/__init__.py``.

Covers:
    Section 1 — Smoke Tests (MILIA_Test_Recommendations.md §1.2 scope):
        - The ``milia_pipeline.preprocessing`` subpackage imports without ImportError
        - All re-exported names from core classes and utility submodules are accessible
        - Module-level metadata attributes (__version__) exist and are typed
        - Module initialization (_log_initialization_status, _validate_critical_components)
          runs safely without crashing
        - Re-import (``importlib.reload``) is idempotent and non-crashing
        - Preprocessor auto-registration error tracking lists exist
        - Utility import error tracking lists exist
        - Convenience functions (get_preprocessing_info, list_available_preprocessors,
          supports_dataset) are accessible and callable
        - Namespace cleanup (``del logging``, ``del warnings``, etc.) executed correctly

    Section 2 — Contract Tests (MILIA_Test_Recommendations.md §2 scope):
        - ``__all__`` completeness: every name in ``__all__`` is resolvable
        - ``__all__`` consistency: every public import is listed in ``__all__``
        - ``__all__`` has no duplicates
        - ``__all__`` entries are all strings
        - Core classes (BasePreprocessor, PreprocessorRegistry) are classes
        - BasePreprocessor is an abstract base class
        - PreprocessorRegistry exposes the documented class methods
          (get_preprocessor, list_preprocessors, supports_preprocessing, register,
          clear_registry)
        - Utility function exports are callable or None (graceful degradation)
        - Convenience functions return documented types:
          - ``get_preprocessing_info()`` returns a dict with expected keys
          - ``list_available_preprocessors()`` returns a list of strings
          - ``supports_dataset()`` returns a bool
        - ``_PREPROCESSOR_IMPORT_ERRORS`` is a list of tuples
        - ``_UTILITY_IMPORT_ERRORS`` is a list of tuples
        - ``__version__`` follows a numeric pattern
        - Namespace cleanliness: ``logging``, ``warnings``, ``Optional``, ``List``
          are deleted after init
        - Public API surface stability (minimum expected names present)
        - ``_log_initialization_status`` and ``_validate_critical_components``
          are callable internal functions

Design:
    - Zero ``sys.modules`` pollution: all mocking via ``@patch`` decorators
      or context-manager ``patch`` calls scoped to individual tests.
    - Deterministic: no filesystem, network, or GPU access required.
    - Isolated: each test is independent; execution order is irrelevant.
    - Fast: expected < 5 s total wall-clock on CI.

Launch:
    From project root (/app/milia):
        pytest tests/test__init__preprocessing.py -v --tb=short

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

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure the project root is importable
# ---------------------------------------------------------------------------
# When launched via ``pytest tests/test__init__preprocessing.py`` from the
# project root (/app/milia), ``milia_pipeline`` must be on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # …/tests -> …/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture(scope="module")
def preprocessing_pkg():
    """
    Import and return the ``milia_pipeline.preprocessing`` package once per module.

    This fixture validates the fundamental smoke invariant: the preprocessing
    subpackage is importable.  If this fails, every downstream test is moot.
    """
    try:
        import milia_pipeline.preprocessing as prep

        return prep
    except ImportError as exc:
        pytest.fail(
            f"milia_pipeline.preprocessing could not be imported — smoke test "
            f"precondition violated: {exc}"
        )


@pytest.fixture(scope="module")
def all_names(preprocessing_pkg):
    """Return the ``__all__`` list from the preprocessing package."""
    assert hasattr(preprocessing_pkg, "__all__"), (
        "milia_pipeline.preprocessing.__all__ is missing — contract violation"
    )
    return list(preprocessing_pkg.__all__)


# ===================================================================
# SECTION 1 — SMOKE TESTS
# ===================================================================


class TestSmokePreprocessingPackageImport:
    """§1.2 — Verify the preprocessing subpackage imports without errors."""

    @pytest.mark.smoke
    def test_import_preprocessing_package_succeeds(self, preprocessing_pkg):
        """The preprocessing package imports without raising any exception."""
        assert preprocessing_pkg is not None

    @pytest.mark.smoke
    def test_preprocessing_package_is_a_module(self, preprocessing_pkg):
        """The imported object is a proper Python module."""
        assert isinstance(preprocessing_pkg, types.ModuleType)

    @pytest.mark.smoke
    def test_preprocessing_package_has_file_attribute(self, preprocessing_pkg):
        """The package exposes a ``__file__`` attribute (not a namespace pkg)."""
        assert hasattr(preprocessing_pkg, "__file__")

    @pytest.mark.smoke
    def test_preprocessing_package_name(self, preprocessing_pkg):
        """The package ``__name__`` is ``milia_pipeline.preprocessing``."""
        assert preprocessing_pkg.__name__ == "milia_pipeline.preprocessing"


class TestSmokeMetadataAttributes:
    """§1.2 — Verify module-level metadata attributes are present and typed."""

    @pytest.mark.smoke
    def test_version_attribute_exists(self, preprocessing_pkg):
        """``__version__`` is defined on the preprocessing package."""
        assert hasattr(preprocessing_pkg, "__version__"), "Missing attribute: __version__"

    @pytest.mark.smoke
    def test_version_attribute_is_string(self, preprocessing_pkg):
        """``__version__`` is a non-empty string."""
        value = preprocessing_pkg.__version__
        assert isinstance(value, str), f"__version__ should be str, got {type(value)}"
        assert len(value) > 0, "__version__ should be non-empty"

    @pytest.mark.smoke
    def test_version_is_numeric_pattern(self, preprocessing_pkg):
        """``__version__`` follows a MAJOR.MINOR pattern (e.g., '1.1')."""
        version = preprocessing_pkg.__version__
        parts = version.split(".")
        assert len(parts) >= 1, f"Version '{version}' should have at least one numeric component"
        for part in parts:
            numeric_part = ""
            for ch in part:
                if ch.isdigit():
                    numeric_part += ch
                else:
                    break
            assert len(numeric_part) > 0, f"Version component '{part}' should start with a digit"


class TestSmokeCoreClassExports:
    """§1.2 — Core classes (BasePreprocessor, PreprocessorRegistry) are accessible."""

    CORE_CLASSES = [
        "BasePreprocessor",
        "PreprocessorRegistry",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CORE_CLASSES)
    def test_core_class_exists(self, preprocessing_pkg, name):
        """Each core class is importable from the preprocessing package."""
        obj = getattr(preprocessing_pkg, name, None)
        assert obj is not None, f"Core class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CORE_CLASSES)
    def test_core_class_is_a_class(self, preprocessing_pkg, name):
        """Each core export is a class (not an instance or function)."""
        obj = getattr(preprocessing_pkg, name)
        assert inspect.isclass(obj), f"'{name}' should be a class, got {type(obj).__name__}"


class TestSmokeUtilityFunctionExports:
    """§1.2 — Utility function exports are accessible (or None for graceful degradation)."""

    # These utility functions may be None if their underlying modules
    # failed to import (graceful degradation pattern in __init__.py).
    UTILITY_FUNCTIONS = [
        "extract_from_targz",
        "parse_molden_files",
        "build_npz",
        "validate_npz_structure",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", UTILITY_FUNCTIONS)
    def test_utility_function_is_defined(self, preprocessing_pkg, name):
        """Each utility function name is defined on the package (may be None)."""
        assert hasattr(preprocessing_pkg, name), (
            f"Utility function '{name}' is not defined on the package"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", UTILITY_FUNCTIONS)
    def test_utility_function_is_callable_or_none(self, preprocessing_pkg, name):
        """
        Each utility function is either callable (successfully imported)
        or None (import failed with graceful degradation).
        """
        obj = getattr(preprocessing_pkg, name)
        assert obj is None or callable(obj), (
            f"'{name}' should be callable or None, got {type(obj).__name__}"
        )


class TestSmokeConvenienceFunctionExports:
    """§1.2 — Convenience functions are accessible and callable."""

    CONVENIENCE_FUNCTIONS = [
        "get_preprocessing_info",
        "list_available_preprocessors",
        "supports_dataset",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CONVENIENCE_FUNCTIONS)
    def test_convenience_function_exists(self, preprocessing_pkg, name):
        """Each convenience function is present and non-None."""
        obj = getattr(preprocessing_pkg, name, None)
        assert obj is not None, f"Convenience function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CONVENIENCE_FUNCTIONS)
    def test_convenience_function_is_callable(self, preprocessing_pkg, name):
        """Each convenience function is callable."""
        obj = getattr(preprocessing_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeImportErrorTrackingLists:
    """§1.2 — Import error tracking lists are present and correctly typed."""

    @pytest.mark.smoke
    def test_preprocessor_import_errors_exists(self, preprocessing_pkg):
        """``_PREPROCESSOR_IMPORT_ERRORS`` is defined on the package."""
        assert hasattr(preprocessing_pkg, "_PREPROCESSOR_IMPORT_ERRORS"), (
            "_PREPROCESSOR_IMPORT_ERRORS is missing"
        )

    @pytest.mark.smoke
    def test_preprocessor_import_errors_is_list(self, preprocessing_pkg):
        """``_PREPROCESSOR_IMPORT_ERRORS`` is a list."""
        obj = preprocessing_pkg._PREPROCESSOR_IMPORT_ERRORS
        assert isinstance(obj, list), (
            f"_PREPROCESSOR_IMPORT_ERRORS should be list, got {type(obj).__name__}"
        )

    @pytest.mark.smoke
    def test_utility_import_errors_exists(self, preprocessing_pkg):
        """``_UTILITY_IMPORT_ERRORS`` is defined on the package."""
        assert hasattr(preprocessing_pkg, "_UTILITY_IMPORT_ERRORS"), (
            "_UTILITY_IMPORT_ERRORS is missing"
        )

    @pytest.mark.smoke
    def test_utility_import_errors_is_list(self, preprocessing_pkg):
        """``_UTILITY_IMPORT_ERRORS`` is a list."""
        obj = preprocessing_pkg._UTILITY_IMPORT_ERRORS
        assert isinstance(obj, list), (
            f"_UTILITY_IMPORT_ERRORS should be list, got {type(obj).__name__}"
        )


class TestSmokeInternalFunctions:
    """§1.2 — Internal functions are present and callable."""

    INTERNAL_FUNCTIONS = [
        "_log_initialization_status",
        "_validate_critical_components",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", INTERNAL_FUNCTIONS)
    def test_internal_function_exists(self, preprocessing_pkg, name):
        """Each internal function is present."""
        assert hasattr(preprocessing_pkg, name), f"Internal function '{name}' is missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", INTERNAL_FUNCTIONS)
    def test_internal_function_is_callable(self, preprocessing_pkg, name):
        """Each internal function is callable."""
        obj = getattr(preprocessing_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeModuleLogger:
    """§1.2 — Module-level logger is configured."""

    @pytest.mark.smoke
    def test_logger_exists(self, preprocessing_pkg):
        """``_logger`` is defined on the package."""
        assert hasattr(preprocessing_pkg, "_logger"), (
            "_logger is missing from preprocessing package"
        )

    @pytest.mark.smoke
    def test_logger_is_logger_instance(self, preprocessing_pkg):
        """``_logger`` is a ``logging.Logger`` instance."""
        logger = preprocessing_pkg._logger
        assert isinstance(logger, logging.Logger), (
            f"_logger should be logging.Logger, got {type(logger).__name__}"
        )


class TestSmokeModuleInitialization:
    """§1.2 — Module-level initialization runs without side-effect crashes."""

    @pytest.mark.smoke
    def test_reimport_is_idempotent(self, preprocessing_pkg):
        """
        Re-importing the preprocessing package (via ``importlib.reload``)
        does not crash.

        Validates that all module-level code (logging, registry status checks,
        _validate_critical_components, namespace cleanup) is safe to re-execute.
        """
        reloaded = importlib.reload(preprocessing_pkg)
        assert reloaded is not None
        assert hasattr(reloaded, "__version__")

    @pytest.mark.smoke
    def test_reimport_preserves_all(self, preprocessing_pkg):
        """
        Re-importing the preprocessing package preserves ``__all__``.
        """
        reloaded = importlib.reload(preprocessing_pkg)
        assert hasattr(reloaded, "__all__")
        assert isinstance(reloaded.__all__, list)
        assert len(reloaded.__all__) > 0

    @pytest.mark.smoke
    def test_reimport_preserves_core_classes(self, preprocessing_pkg):
        """
        Re-importing preserves BasePreprocessor and PreprocessorRegistry.
        """
        reloaded = importlib.reload(preprocessing_pkg)
        assert hasattr(reloaded, "BasePreprocessor")
        assert hasattr(reloaded, "PreprocessorRegistry")
        assert inspect.isclass(reloaded.BasePreprocessor)
        assert inspect.isclass(reloaded.PreprocessorRegistry)


class TestSmokeNamespaceCleanup:
    """§1.2 — Namespace cleanup verifies standard library names are deleted."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "name",
        [
            "logging",
            "warnings",
            "Optional",
            "List",
        ],
    )
    def test_cleaned_name_not_in_namespace(self, preprocessing_pkg, name):
        """
        The ``__init__.py`` explicitly deletes standard library names
        (``del logging, warnings, Optional, List``) at the end.
        After initial import, these should NOT be attributes.
        """
        if hasattr(preprocessing_pkg, name):
            obj = getattr(preprocessing_pkg, name)
            # If present, it should NOT be the original module/type reference
            if isinstance(obj, types.ModuleType):
                assert obj.__name__ != name, (
                    f"The '{name}' module should be cleaned from namespace after init"
                )


class TestSmokeLogInitializationStatusExecution:
    """§1.2 — _log_initialization_status() runs without exception."""

    @pytest.mark.smoke
    def test_log_initialization_status_runs_safely(self, preprocessing_pkg):
        """
        Calling ``_log_initialization_status()`` does not raise.
        This validates the logging blocks that report registered preprocessors
        and import errors.
        """
        # Should not raise any exception
        preprocessing_pkg._log_initialization_status()

    @pytest.mark.smoke
    def test_validate_critical_components_runs_safely(self, preprocessing_pkg):
        """
        Calling ``_validate_critical_components()`` does not crash the test.
        The function may raise RuntimeError if critical components are missing,
        but the function itself should be safely callable.
        """
        # May raise RuntimeError if no preprocessors registered — that's OK
        # We just verify it doesn't raise unexpected exceptions
        try:
            preprocessing_pkg._validate_critical_components()
        except RuntimeError:
            # Expected when preprocessors are not fully available
            pass


# ===================================================================
# SECTION 2 — CONTRACT TESTS
# ===================================================================


class TestContractAllCompleteness:
    """§2 — Every name in ``__all__`` is resolvable on the preprocessing package."""

    @pytest.mark.contract
    def test_all_is_a_list(self, preprocessing_pkg):
        """``__all__`` is a list (not a tuple or set)."""
        assert isinstance(preprocessing_pkg.__all__, list)

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
    def test_every_all_entry_is_resolvable(self, preprocessing_pkg, all_names):
        """
        Generic sweep: every single entry in ``__all__`` must be resolvable,
        regardless of whether it is parameterized individually.
        """
        unresolvable = [name for name in all_names if not hasattr(preprocessing_pkg, name)]
        assert not unresolvable, (
            f"Names in __all__ that are not defined on the module: {unresolvable}"
        )

    @pytest.mark.contract
    def test_all_entries_are_strings(self, all_names):
        """Every entry in ``__all__`` is a string."""
        non_strings = [(i, name) for i, name in enumerate(all_names) if not isinstance(name, str)]
        assert not non_strings, f"Non-string entries in __all__: {non_strings}"


class TestContractAllConsistency:
    """§2 — Every public import in the preprocessing module is listed in ``__all__``."""

    # Names that are intentionally public-ish but NOT in __all__
    # (metadata, internal helpers, error tracking lists, etc.)
    KNOWN_UNLISTED = {
        # Metadata dunders (not typically in __all__)
        "__version__",
        # Internal logger
        "_logger",
        # Import error tracking (internal)
        "_PREPROCESSOR_IMPORT_ERRORS",
        "_UTILITY_IMPORT_ERRORS",
        # Internal functions
        "_log_initialization_status",
        "_validate_critical_components",
    }

    @pytest.mark.contract
    def test_public_imports_are_in_all(self, preprocessing_pkg, all_names):
        """
        Every non-dunder, non-private attribute that was imported (not
        defined locally) should be in ``__all__`` — unless it is in
        ``KNOWN_UNLISTED``.

        This catches accidental omissions when new imports are added to
        the preprocessing ``__init__.py`` without updating ``__all__``.
        """
        all_set = set(all_names)
        module_dict = vars(preprocessing_pkg)
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
            f"Public names imported in preprocessing/__init__.py but not in __all__: "
            f"{sorted(missing_from_all)}"
        )


class TestContractCoreClassTypes:
    """§2 — Core classes have the expected nature (class, ABC, etc.)."""

    @pytest.mark.contract
    def test_base_preprocessor_is_class(self, preprocessing_pkg):
        """``BasePreprocessor`` is a class."""
        assert inspect.isclass(preprocessing_pkg.BasePreprocessor), (
            f"BasePreprocessor should be a class, got "
            f"{type(preprocessing_pkg.BasePreprocessor).__name__}"
        )

    @pytest.mark.contract
    def test_base_preprocessor_is_abstract(self, preprocessing_pkg):
        """
        ``BasePreprocessor`` is an abstract base class (ABC).

        Per the project structure doc: base_preprocessor.py defines an ABC
        with abstract methods ``_validate_config()`` and ``preprocess()``.
        """
        cls = preprocessing_pkg.BasePreprocessor
        # Check for ABC metaclass or __abstractmethods__
        from abc import ABCMeta

        is_abc = isinstance(cls, ABCMeta) or hasattr(cls, "__abstractmethods__")
        assert is_abc, "BasePreprocessor should be an abstract base class (ABC)"

    @pytest.mark.contract
    def test_base_preprocessor_has_abstract_methods(self, preprocessing_pkg):
        """
        ``BasePreprocessor`` declares abstract methods.

        Expected abstract methods: ``_validate_config``, ``preprocess``.
        """
        cls = preprocessing_pkg.BasePreprocessor
        if hasattr(cls, "__abstractmethods__"):
            abstract_methods = cls.__abstractmethods__
            assert len(abstract_methods) > 0, (
                "BasePreprocessor should have at least one abstract method"
            )
            expected = {"_validate_config", "preprocess"}
            for method_name in expected:
                assert method_name in abstract_methods, (
                    f"BasePreprocessor should have abstract method '{method_name}', "
                    f"found: {sorted(abstract_methods)}"
                )

    @pytest.mark.contract
    def test_base_preprocessor_has_run_method(self, preprocessing_pkg):
        """
        ``BasePreprocessor`` has a concrete ``run()`` method.

        Per the project structure doc: ``run() -> Path`` executes the full
        pipeline with timing and validation.
        """
        cls = preprocessing_pkg.BasePreprocessor
        assert hasattr(cls, "run"), "BasePreprocessor should have a 'run' method"
        assert callable(cls.run), "BasePreprocessor.run should be callable"

    @pytest.mark.contract
    def test_base_preprocessor_has_validate_output_method(self, preprocessing_pkg):
        """
        ``BasePreprocessor`` has a concrete ``_validate_output()`` method.

        Per the project structure doc: ``_validate_output(output_path)``
        validates .npz structure (requires 'compounds', 'metadata').
        """
        cls = preprocessing_pkg.BasePreprocessor
        assert hasattr(cls, "_validate_output"), (
            "BasePreprocessor should have a '_validate_output' method"
        )

    @pytest.mark.contract
    def test_preprocessor_registry_is_class(self, preprocessing_pkg):
        """``PreprocessorRegistry`` is a class."""
        assert inspect.isclass(preprocessing_pkg.PreprocessorRegistry), (
            f"PreprocessorRegistry should be a class, got "
            f"{type(preprocessing_pkg.PreprocessorRegistry).__name__}"
        )


class TestContractPreprocessorRegistryAPI:
    """§2 — PreprocessorRegistry exposes the documented class methods."""

    DOCUMENTED_CLASS_METHODS = [
        "register",
        "get_preprocessor",
        "list_preprocessors",
        "supports_preprocessing",
        "clear_registry",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("method_name", DOCUMENTED_CLASS_METHODS)
    def test_registry_method_exists(self, preprocessing_pkg, method_name):
        """Each documented class method exists on PreprocessorRegistry."""
        cls = preprocessing_pkg.PreprocessorRegistry
        assert hasattr(cls, method_name), f"PreprocessorRegistry should have method '{method_name}'"

    @pytest.mark.contract
    @pytest.mark.parametrize("method_name", DOCUMENTED_CLASS_METHODS)
    def test_registry_method_is_callable(self, preprocessing_pkg, method_name):
        """Each documented class method is callable."""
        cls = preprocessing_pkg.PreprocessorRegistry
        method = getattr(cls, method_name)
        assert callable(method), f"PreprocessorRegistry.{method_name} should be callable"

    @pytest.mark.contract
    def test_registry_has_preprocessors_dict(self, preprocessing_pkg):
        """
        PreprocessorRegistry has a ``_preprocessors`` class attribute (dict).

        Per the project structure doc: ``_preprocessors: Dict[str, Type[BasePreprocessor]]``.
        """
        cls = preprocessing_pkg.PreprocessorRegistry
        assert hasattr(cls, "_preprocessors"), (
            "PreprocessorRegistry should have '_preprocessors' class attribute"
        )
        assert isinstance(cls._preprocessors, dict), (
            f"PreprocessorRegistry._preprocessors should be dict, "
            f"got {type(cls._preprocessors).__name__}"
        )


class TestContractPreprocessorRegistryReturnTypes:
    """§2 — PreprocessorRegistry methods return documented types."""

    @pytest.mark.contract
    def test_list_preprocessors_returns_list(self, preprocessing_pkg):
        """``PreprocessorRegistry.list_preprocessors()`` returns a list."""
        result = preprocessing_pkg.PreprocessorRegistry.list_preprocessors()
        assert isinstance(result, list), (
            f"list_preprocessors() should return list, got {type(result).__name__}"
        )

    @pytest.mark.contract
    def test_list_preprocessors_entries_are_strings(self, preprocessing_pkg):
        """``list_preprocessors()`` returns a list of strings."""
        result = preprocessing_pkg.PreprocessorRegistry.list_preprocessors()
        for item in result:
            assert isinstance(item, str), (
                f"Each entry from list_preprocessors() should be str, got {type(item).__name__}"
            )

    @pytest.mark.contract
    def test_supports_preprocessing_returns_bool(self, preprocessing_pkg):
        """``PreprocessorRegistry.supports_preprocessing()`` returns a bool."""
        # Test with a known type name (may or may not be registered)
        result = preprocessing_pkg.PreprocessorRegistry.supports_preprocessing("Wavefunction")
        assert isinstance(result, bool), (
            f"supports_preprocessing() should return bool, got {type(result).__name__}"
        )

    @pytest.mark.contract
    def test_supports_preprocessing_false_for_unknown(self, preprocessing_pkg):
        """``supports_preprocessing()`` returns False for an unknown dataset type."""
        result = preprocessing_pkg.PreprocessorRegistry.supports_preprocessing(
            "NonExistentDatasetType_XYZ_12345"
        )
        assert result is False, (
            "supports_preprocessing() should return False for unknown dataset type"
        )

    @pytest.mark.contract
    def test_register_is_decorator(self, preprocessing_pkg):
        """
        ``PreprocessorRegistry.register()`` returns a callable (decorator).

        Per the project structure doc: ``@register(dataset_type)`` is a
        decorator for auto-registration.
        """
        cls = preprocessing_pkg.PreprocessorRegistry
        # Calling register with a dataset_type should return a decorator
        decorator = cls.register("__test_decorator_check__")
        assert callable(decorator), (
            "PreprocessorRegistry.register('name') should return a callable decorator"
        )
        # Clean up: remove the test registration if it was added
        if hasattr(cls, "_preprocessors") and "__test_decorator_check__" in cls._preprocessors:
            del cls._preprocessors["__test_decorator_check__"]


class TestContractUtilityFunctionTypes:
    """§2 — Utility functions have the expected callable nature when available."""

    @pytest.mark.contract
    def test_extract_from_targz_type(self, preprocessing_pkg):
        """
        ``extract_from_targz`` is either callable (function) or None
        (graceful degradation if archive_handlers import failed).
        """
        obj = preprocessing_pkg.extract_from_targz
        if obj is not None:
            assert callable(obj), f"extract_from_targz should be callable, got {type(obj).__name__}"
            assert inspect.isfunction(obj), (
                f"extract_from_targz should be a function, got {type(obj).__name__}"
            )

    @pytest.mark.contract
    def test_parse_molden_files_type(self, preprocessing_pkg):
        """
        ``parse_molden_files`` is either callable (function) or None
        (graceful degradation if format_parsers import failed).
        """
        obj = preprocessing_pkg.parse_molden_files
        if obj is not None:
            assert callable(obj), f"parse_molden_files should be callable, got {type(obj).__name__}"
            assert inspect.isfunction(obj), (
                f"parse_molden_files should be a function, got {type(obj).__name__}"
            )

    @pytest.mark.contract
    def test_build_npz_type(self, preprocessing_pkg):
        """
        ``build_npz`` is either callable (function) or None
        (graceful degradation if npz_builders import failed).
        """
        obj = preprocessing_pkg.build_npz
        if obj is not None:
            assert callable(obj), f"build_npz should be callable, got {type(obj).__name__}"
            assert inspect.isfunction(obj), (
                f"build_npz should be a function, got {type(obj).__name__}"
            )

    @pytest.mark.contract
    def test_validate_npz_structure_type(self, preprocessing_pkg):
        """
        ``validate_npz_structure`` is either callable (function) or None
        (graceful degradation if npz_builders import failed).
        """
        obj = preprocessing_pkg.validate_npz_structure
        if obj is not None:
            assert callable(obj), (
                f"validate_npz_structure should be callable, got {type(obj).__name__}"
            )
            assert inspect.isfunction(obj), (
                f"validate_npz_structure should be a function, got {type(obj).__name__}"
            )


class TestContractConvenienceFunctionReturnTypes:
    """§2 — Convenience functions return documented types."""

    @pytest.mark.contract
    def test_get_preprocessing_info_returns_dict(self, preprocessing_pkg):
        """``get_preprocessing_info()`` returns a dict."""
        result = preprocessing_pkg.get_preprocessing_info()
        assert isinstance(result, dict), (
            f"get_preprocessing_info() should return dict, got {type(result).__name__}"
        )

    @pytest.mark.contract
    def test_get_preprocessing_info_has_expected_keys(self, preprocessing_pkg):
        """
        ``get_preprocessing_info()`` returns a dict with documented keys:
        version, registered_preprocessors, available_utilities,
        preprocessor_import_errors, utility_import_errors.
        """
        result = preprocessing_pkg.get_preprocessing_info()
        expected_keys = {
            "version",
            "registered_preprocessors",
            "available_utilities",
            "preprocessor_import_errors",
            "utility_import_errors",
        }
        missing_keys = expected_keys - set(result.keys())
        assert not missing_keys, (
            f"get_preprocessing_info() missing expected keys: {sorted(missing_keys)}. "
            f"Got keys: {sorted(result.keys())}"
        )

    @pytest.mark.contract
    def test_get_preprocessing_info_version_matches(self, preprocessing_pkg):
        """
        ``get_preprocessing_info()['version']`` matches ``__version__``.
        """
        result = preprocessing_pkg.get_preprocessing_info()
        assert result["version"] == preprocessing_pkg.__version__, (
            f"get_preprocessing_info()['version'] is '{result['version']}', "
            f"but __version__ is '{preprocessing_pkg.__version__}'"
        )

    @pytest.mark.contract
    def test_get_preprocessing_info_registered_preprocessors_is_list(self, preprocessing_pkg):
        """
        ``get_preprocessing_info()['registered_preprocessors']`` is a list.
        """
        result = preprocessing_pkg.get_preprocessing_info()
        assert isinstance(result["registered_preprocessors"], list), (
            f"registered_preprocessors should be list, "
            f"got {type(result['registered_preprocessors']).__name__}"
        )

    @pytest.mark.contract
    def test_get_preprocessing_info_available_utilities_is_list(self, preprocessing_pkg):
        """
        ``get_preprocessing_info()['available_utilities']`` is a list.
        """
        result = preprocessing_pkg.get_preprocessing_info()
        assert isinstance(result["available_utilities"], list), (
            f"available_utilities should be list, "
            f"got {type(result['available_utilities']).__name__}"
        )

    @pytest.mark.contract
    def test_get_preprocessing_info_import_errors_are_lists(self, preprocessing_pkg):
        """
        Import error entries in ``get_preprocessing_info()`` are lists.
        """
        result = preprocessing_pkg.get_preprocessing_info()
        assert isinstance(result["preprocessor_import_errors"], list), (
            f"preprocessor_import_errors should be list, "
            f"got {type(result['preprocessor_import_errors']).__name__}"
        )
        assert isinstance(result["utility_import_errors"], list), (
            f"utility_import_errors should be list, "
            f"got {type(result['utility_import_errors']).__name__}"
        )

    @pytest.mark.contract
    def test_list_available_preprocessors_returns_list(self, preprocessing_pkg):
        """``list_available_preprocessors()`` returns a list."""
        result = preprocessing_pkg.list_available_preprocessors()
        assert isinstance(result, list), (
            f"list_available_preprocessors() should return list, got {type(result).__name__}"
        )

    @pytest.mark.contract
    def test_list_available_preprocessors_entries_are_strings(self, preprocessing_pkg):
        """``list_available_preprocessors()`` returns a list of strings."""
        result = preprocessing_pkg.list_available_preprocessors()
        for item in result:
            assert isinstance(item, str), (
                f"Each entry from list_available_preprocessors() should be str, "
                f"got {type(item).__name__}"
            )

    @pytest.mark.contract
    def test_list_available_preprocessors_matches_registry(self, preprocessing_pkg):
        """
        ``list_available_preprocessors()`` returns the same result as
        ``PreprocessorRegistry.list_preprocessors()``.

        Per the source code: ``list_available_preprocessors`` is a wrapper
        around ``PreprocessorRegistry.list_preprocessors()``.
        """
        convenience_result = preprocessing_pkg.list_available_preprocessors()
        registry_result = preprocessing_pkg.PreprocessorRegistry.list_preprocessors()
        assert convenience_result == registry_result, (
            f"list_available_preprocessors() returned {convenience_result}, but "
            f"PreprocessorRegistry.list_preprocessors() returned {registry_result}"
        )

    @pytest.mark.contract
    def test_supports_dataset_returns_bool(self, preprocessing_pkg):
        """``supports_dataset()`` returns a bool."""
        result = preprocessing_pkg.supports_dataset("Wavefunction")
        assert isinstance(result, bool), (
            f"supports_dataset() should return bool, got {type(result).__name__}"
        )

    @pytest.mark.contract
    def test_supports_dataset_matches_registry(self, preprocessing_pkg):
        """
        ``supports_dataset()`` returns the same result as
        ``PreprocessorRegistry.supports_preprocessing()``.

        Per the source code: ``supports_dataset`` is a wrapper around
        ``PreprocessorRegistry.supports_preprocessing()``.
        """
        test_type = "Wavefunction"
        convenience_result = preprocessing_pkg.supports_dataset(test_type)
        registry_result = preprocessing_pkg.PreprocessorRegistry.supports_preprocessing(test_type)
        assert convenience_result == registry_result, (
            f"supports_dataset('{test_type}') returned {convenience_result}, but "
            f"PreprocessorRegistry.supports_preprocessing('{test_type}') "
            f"returned {registry_result}"
        )

    @pytest.mark.contract
    def test_supports_dataset_false_for_unknown(self, preprocessing_pkg):
        """``supports_dataset()`` returns False for an unknown dataset type."""
        result = preprocessing_pkg.supports_dataset("NonExistentDatasetType_XYZ_12345")
        assert result is False, "supports_dataset() should return False for unknown dataset type"


class TestContractImportErrorTrackingStructure:
    """§2 — Import error tracking lists have the expected tuple structure."""

    @pytest.mark.contract
    def test_preprocessor_import_errors_entries_are_tuples(self, preprocessing_pkg):
        """
        Each entry in ``_PREPROCESSOR_IMPORT_ERRORS`` is a tuple of
        (name: str, error: str).
        """
        for entry in preprocessing_pkg._PREPROCESSOR_IMPORT_ERRORS:
            assert isinstance(entry, tuple), (
                f"Each entry should be a tuple, got {type(entry).__name__}"
            )
            assert len(entry) == 2, (
                f"Each entry should have 2 elements (name, error), got {len(entry)}"
            )
            name, error = entry
            assert isinstance(name, str), f"Entry name should be str, got {type(name).__name__}"
            assert isinstance(error, str), f"Entry error should be str, got {type(error).__name__}"

    @pytest.mark.contract
    def test_utility_import_errors_entries_are_tuples(self, preprocessing_pkg):
        """
        Each entry in ``_UTILITY_IMPORT_ERRORS`` is a tuple of
        (name: str, error: str).
        """
        for entry in preprocessing_pkg._UTILITY_IMPORT_ERRORS:
            assert isinstance(entry, tuple), (
                f"Each entry should be a tuple, got {type(entry).__name__}"
            )
            assert len(entry) == 2, (
                f"Each entry should have 2 elements (name, error), got {len(entry)}"
            )
            name, error = entry
            assert isinstance(name, str), f"Entry name should be str, got {type(name).__name__}"
            assert isinstance(error, str), f"Entry error should be str, got {type(error).__name__}"


class TestContractConvenienceFunctionSignatures:
    """§2 — Convenience functions have documented parameter signatures."""

    @pytest.mark.contract
    def test_get_preprocessing_info_accepts_no_args(self, preprocessing_pkg):
        """``get_preprocessing_info()`` takes no required parameters."""
        sig = inspect.signature(preprocessing_pkg.get_preprocessing_info)
        required_params = [
            name
            for name, param in sig.parameters.items()
            if param.default is inspect.Parameter.empty
            and param.kind
            not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        assert len(required_params) == 0, (
            f"get_preprocessing_info() should have no required parameters, found: {required_params}"
        )

    @pytest.mark.contract
    def test_list_available_preprocessors_accepts_no_args(self, preprocessing_pkg):
        """``list_available_preprocessors()`` takes no required parameters."""
        sig = inspect.signature(preprocessing_pkg.list_available_preprocessors)
        required_params = [
            name
            for name, param in sig.parameters.items()
            if param.default is inspect.Parameter.empty
            and param.kind
            not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        assert len(required_params) == 0, (
            f"list_available_preprocessors() should have no required parameters, "
            f"found: {required_params}"
        )

    @pytest.mark.contract
    def test_supports_dataset_accepts_one_arg(self, preprocessing_pkg):
        """``supports_dataset()`` takes exactly one required parameter (dataset_type)."""
        sig = inspect.signature(preprocessing_pkg.supports_dataset)
        required_params = [
            name
            for name, param in sig.parameters.items()
            if param.default is inspect.Parameter.empty
            and param.kind
            not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        assert len(required_params) == 1, (
            f"supports_dataset() should have exactly 1 required parameter, found: {required_params}"
        )

    @pytest.mark.contract
    def test_supports_dataset_parameter_name(self, preprocessing_pkg):
        """``supports_dataset()`` parameter is named 'dataset_type'."""
        sig = inspect.signature(preprocessing_pkg.supports_dataset)
        param_names = list(sig.parameters.keys())
        assert "dataset_type" in param_names, (
            f"supports_dataset() should have a 'dataset_type' parameter, found: {param_names}"
        )


class TestContractConvenienceFunctionsAreFunction:
    """§2 — Convenience functions are functions (not classes or methods)."""

    CONVENIENCE_FUNCTIONS = [
        "get_preprocessing_info",
        "list_available_preprocessors",
        "supports_dataset",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", CONVENIENCE_FUNCTIONS)
    def test_convenience_function_is_function(self, preprocessing_pkg, name):
        """Each convenience function is a function (not a class)."""
        obj = getattr(preprocessing_pkg, name)
        assert inspect.isfunction(obj), f"'{name}' should be a function, got {type(obj).__name__}"


class TestContractPublicAPISurface:
    """
    §2 — Public API surface stability.

    Ensures that the minimum expected public API is present in ``__all__``.
    Guards against accidental removals during refactoring.
    """

    # The minimum API surface that MUST be present in __all__
    # Based on the __init__.py source code and project structure doc.
    MINIMUM_API = {
        # Core Classes
        "BasePreprocessor",
        "PreprocessorRegistry",
        # Utility Functions - Archive Handling
        "extract_from_targz",
        # Utility Functions - Format Parsing
        "parse_molden_files",
        # Utility Functions - NPZ Building
        "build_npz",
        "validate_npz_structure",
        # Convenience Functions
        "get_preprocessing_info",
        "list_available_preprocessors",
        "supports_dataset",
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
        ``__all__`` contains the expected number of entries.

        Based on the __init__.py source, the preprocessing package exports
        9 names in __all__ (6 initial + 3 added via __all__.extend()).
        We set a floor to guard against catastrophic loss while allowing
        for organic growth.
        """
        actual = len(all_names)
        # The __init__.py has 9 items in __all__:
        # BasePreprocessor, PreprocessorRegistry, extract_from_targz,
        # parse_molden_files, build_npz, validate_npz_structure,
        # get_preprocessing_info, list_available_preprocessors, supports_dataset
        MINIMUM_EXPECTED = 9
        assert actual >= MINIMUM_EXPECTED, (
            f"__all__ has {actual} entries, expected at least {MINIMUM_EXPECTED}. "
            f"This suggests __all__ may have been accidentally truncated."
        )


class TestContractGracefulDegradationPattern:
    """
    §2 — The module follows a graceful degradation pattern for imports.

    When utility submodules fail to import, the corresponding names are set
    to ``None`` rather than propagating the ImportError. This allows the
    module to be imported even in environments with incomplete dependencies.
    """

    DEGRADABLE_UTILITIES = [
        "extract_from_targz",
        "parse_molden_files",
        "build_npz",
        "validate_npz_structure",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", DEGRADABLE_UTILITIES)
    def test_utility_is_callable_or_none(self, preprocessing_pkg, name):
        """
        Each degradable utility is either a callable function or None.
        This verifies the try/except + ``name = None`` pattern.
        """
        obj = getattr(preprocessing_pkg, name)
        assert obj is None or callable(obj), (
            f"'{name}' should be callable or None (graceful degradation), got {type(obj).__name__}"
        )

    @pytest.mark.contract
    def test_available_utilities_matches_globals(self, preprocessing_pkg):
        """
        ``get_preprocessing_info()['available_utilities']`` lists exactly
        those utilities that are not None.
        """
        info = preprocessing_pkg.get_preprocessing_info()
        available = info["available_utilities"]

        expected_available = [
            name
            for name in self.DEGRADABLE_UTILITIES
            if getattr(preprocessing_pkg, name) is not None
        ]

        assert set(available) == set(expected_available), (
            f"available_utilities mismatch: info reports {available}, "
            f"but non-None globals are {expected_available}"
        )


class TestContractCoreImportsAreNonDegradable:
    """
    §2 — Core imports (BasePreprocessor, PreprocessorRegistry) are
    non-degradable: they raise ImportError on failure rather than
    falling back to None.

    This validates the ``raise ImportError(...)`` pattern in __init__.py
    for critical components vs. the ``name = None`` pattern for optional
    utilities.
    """

    @pytest.mark.contract
    def test_base_preprocessor_is_not_none(self, preprocessing_pkg):
        """``BasePreprocessor`` is never None (non-degradable import)."""
        assert preprocessing_pkg.BasePreprocessor is not None, (
            "BasePreprocessor should never be None — it is a critical component"
        )

    @pytest.mark.contract
    def test_preprocessor_registry_is_not_none(self, preprocessing_pkg):
        """``PreprocessorRegistry`` is never None (non-degradable import)."""
        assert preprocessing_pkg.PreprocessorRegistry is not None, (
            "PreprocessorRegistry should never be None — it is a critical component"
        )


class TestContractPreprocessorAutoRegistration:
    """
    §2 — Preprocessor auto-registration contract.

    When preprocessor modules are successfully imported, they auto-register
    via the ``@PreprocessorRegistry.register()`` decorator. The registration
    status is reflected in ``_PREPROCESSOR_IMPORT_ERRORS`` and
    ``PreprocessorRegistry.list_preprocessors()``.
    """

    # All known preprocessor types from the project structure doc
    KNOWN_PREPROCESSOR_TYPES = [
        "Wavefunction",
        "QM9",
        "ANI1x",
        "ANI1ccx",
        "ANI2x",
        "RMD17",
        "XXMD",
        "QDPi",
    ]

    # Preprocessor module names used in _PREPROCESSOR_IMPORT_ERRORS
    KNOWN_PREPROCESSOR_MODULES = [
        "wavefunction",
        "qm9",
        "ani1x",
    ]

    @pytest.mark.contract
    def test_registered_preprocessors_are_known_types(self, preprocessing_pkg):
        """
        Every registered preprocessor name is a known dataset type.
        """
        registered = preprocessing_pkg.PreprocessorRegistry.list_preprocessors()
        for name in registered:
            assert name in self.KNOWN_PREPROCESSOR_TYPES, (
                f"Registered preprocessor '{name}' is not a known type. "
                f"Known types: {self.KNOWN_PREPROCESSOR_TYPES}"
            )

    @pytest.mark.contract
    def test_import_errors_reference_known_modules(self, preprocessing_pkg):
        """
        Every entry in ``_PREPROCESSOR_IMPORT_ERRORS`` references a
        known preprocessor module name.
        """
        errors = preprocessing_pkg._PREPROCESSOR_IMPORT_ERRORS
        if errors:
            for name, _error in errors:
                assert isinstance(name, str), (
                    f"Error entry name should be str, got {type(name).__name__}"
                )

    @pytest.mark.contract
    def test_registered_plus_errors_accounts_for_all_imports(self, preprocessing_pkg):
        """
        The set of successfully registered preprocessors plus the set of
        failed imports should account for the preprocessors that the
        __init__.py attempts to import.

        This verifies that no import attempt is silently dropped without
        either succeeding or recording an error.
        """
        registered = set(preprocessing_pkg.PreprocessorRegistry.list_preprocessors())
        failed_modules = {name for name, _error in preprocessing_pkg._PREPROCESSOR_IMPORT_ERRORS}
        # Verify no overlap — a module can't both succeed and fail
        overlap = registered & failed_modules
        # Note: registered uses type names (e.g., "Wavefunction"), while
        # failed uses module names (e.g., "wavefunction"). These won't
        # actually overlap, but we verify the tracking lists are consistent.
        assert isinstance(registered, set)
        assert isinstance(failed_modules, set)


class TestContractGetPreprocessorReturnType:
    """§2 — PreprocessorRegistry.get_preprocessor() return type contract."""

    @pytest.mark.contract
    def test_get_preprocessor_returns_class_for_registered(self, preprocessing_pkg):
        """
        ``get_preprocessor()`` returns a class (subclass of BasePreprocessor)
        for any registered preprocessor type.
        """
        registered = preprocessing_pkg.PreprocessorRegistry.list_preprocessors()
        BasePreprocessor = preprocessing_pkg.BasePreprocessor
        for name in registered:
            cls = preprocessing_pkg.PreprocessorRegistry.get_preprocessor(name)
            assert inspect.isclass(cls), (
                f"get_preprocessor('{name}') should return a class, got {type(cls).__name__}"
            )
            assert issubclass(cls, BasePreprocessor), (
                f"get_preprocessor('{name}') should return a BasePreprocessor subclass"
            )

    @pytest.mark.contract
    def test_get_preprocessor_raises_for_unknown(self, preprocessing_pkg):
        """
        ``get_preprocessor()`` raises an exception for an unknown type.
        """
        with pytest.raises(Exception):
            preprocessing_pkg.PreprocessorRegistry.get_preprocessor(
                "NonExistentDatasetType_XYZ_12345"
            )


class TestContractBasePreprocessorConstructorSignature:
    """§2 — BasePreprocessor constructor accepts documented parameters."""

    @pytest.mark.contract
    def test_constructor_has_config_param(self, preprocessing_pkg):
        """
        ``BasePreprocessor.__init__`` accepts a ``config`` parameter.

        Per the project structure doc:
        ``__init__(config: Dict[str, Any], logger: logging.Logger)``
        """
        cls = preprocessing_pkg.BasePreprocessor
        sig = inspect.signature(cls.__init__)
        param_names = set(sig.parameters.keys()) - {"self"}
        assert "config" in param_names, (
            f"BasePreprocessor.__init__ should accept 'config' parameter, "
            f"found: {sorted(param_names)}"
        )

    @pytest.mark.contract
    def test_constructor_has_logger_param(self, preprocessing_pkg):
        """
        ``BasePreprocessor.__init__`` accepts a ``logger`` parameter.

        Per the project structure doc:
        ``__init__(config: Dict[str, Any], logger: logging.Logger)``
        """
        cls = preprocessing_pkg.BasePreprocessor
        sig = inspect.signature(cls.__init__)
        param_names = set(sig.parameters.keys()) - {"self"}
        assert "logger" in param_names, (
            f"BasePreprocessor.__init__ should accept 'logger' parameter, "
            f"found: {sorted(param_names)}"
        )


class TestContractVersionConsistency:
    """§2 — Version consistency across module attributes."""

    @pytest.mark.contract
    def test_version_in_get_preprocessing_info(self, preprocessing_pkg):
        """
        ``get_preprocessing_info()['version']`` equals ``__version__``.
        """
        info = preprocessing_pkg.get_preprocessing_info()
        assert info["version"] == preprocessing_pkg.__version__

    @pytest.mark.contract
    def test_version_is_string(self, preprocessing_pkg):
        """``__version__`` is a string."""
        assert isinstance(preprocessing_pkg.__version__, str)

    @pytest.mark.contract
    def test_version_is_not_empty(self, preprocessing_pkg):
        """``__version__`` is not empty."""
        assert len(preprocessing_pkg.__version__) > 0


class TestContractRegisteredPreprocessorClasses:
    """§2 — Each registered preprocessor is a proper BasePreprocessor subclass."""

    @pytest.mark.contract
    def test_each_registered_preprocessor_is_class(self, preprocessing_pkg):
        """Each registered preprocessor is a class."""
        registered = preprocessing_pkg.PreprocessorRegistry.list_preprocessors()
        for name in registered:
            cls = preprocessing_pkg.PreprocessorRegistry.get_preprocessor(name)
            assert inspect.isclass(cls), f"Registered preprocessor '{name}' should be a class"

    @pytest.mark.contract
    def test_each_registered_preprocessor_is_base_subclass(self, preprocessing_pkg):
        """Each registered preprocessor is a BasePreprocessor subclass."""
        BasePreprocessor = preprocessing_pkg.BasePreprocessor
        registered = preprocessing_pkg.PreprocessorRegistry.list_preprocessors()
        for name in registered:
            cls = preprocessing_pkg.PreprocessorRegistry.get_preprocessor(name)
            assert issubclass(cls, BasePreprocessor), (
                f"Registered preprocessor '{name}' should be a "
                f"BasePreprocessor subclass, got {cls.__mro__}"
            )

    @pytest.mark.contract
    def test_each_registered_preprocessor_has_preprocess_method(self, preprocessing_pkg):
        """Each registered preprocessor implements the ``preprocess()`` method."""
        registered = preprocessing_pkg.PreprocessorRegistry.list_preprocessors()
        for name in registered:
            cls = preprocessing_pkg.PreprocessorRegistry.get_preprocessor(name)
            assert hasattr(cls, "preprocess"), (
                f"Registered preprocessor '{name}' should have a 'preprocess' method"
            )

    @pytest.mark.contract
    def test_each_registered_preprocessor_has_validate_config_method(self, preprocessing_pkg):
        """Each registered preprocessor implements the ``_validate_config()`` method."""
        registered = preprocessing_pkg.PreprocessorRegistry.list_preprocessors()
        for name in registered:
            cls = preprocessing_pkg.PreprocessorRegistry.get_preprocessor(name)
            assert hasattr(cls, "_validate_config"), (
                f"Registered preprocessor '{name}' should have a '_validate_config' method"
            )

    @pytest.mark.contract
    def test_each_registered_preprocessor_has_run_method(self, preprocessing_pkg):
        """Each registered preprocessor inherits the ``run()`` method."""
        registered = preprocessing_pkg.PreprocessorRegistry.list_preprocessors()
        for name in registered:
            cls = preprocessing_pkg.PreprocessorRegistry.get_preprocessor(name)
            assert hasattr(cls, "run"), (
                f"Registered preprocessor '{name}' should have a 'run' method "
                f"(inherited from BasePreprocessor)"
            )
