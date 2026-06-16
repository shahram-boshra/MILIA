#!/usr/bin/env python3
# tests/test__init__models_factory.py

"""
Test Suite: milia_pipeline/models/factory/__init__.py — Smoke Tests & Contract Tests
=====================================================================================

Production-ready test suite for the MILIA Pipeline models factory package
``milia_pipeline/models/factory/__init__.py``.

Covers:
    Section 1 — Smoke Tests (MILIA_Test_Recommendations.md §1.2 scope):
        - The ``milia_pipeline.models.factory`` subpackage imports without ImportError
        - All re-exported names from the underlying submodules are accessible
        - Module-level metadata attributes (__version__, __author__) exist
        - ``__version__`` is unified with the canonical package version
        - Re-import (``importlib.reload``) is idempotent and non-crashing
        - Main factory classes are accessible
        - Task-level wrapper classes are accessible
        - Public API functions are accessible and callable
        - Target-selection components are accessible

    Section 2 — Contract Tests (MILIA_Test_Recommendations.md §2 scope):
        - ``__all__`` completeness: every name in ``__all__`` is resolvable
        - ``__all__`` consistency: every public import is listed in ``__all__``
        - ``__all__`` has no duplicates
        - Re-exported classes are classes, functions are callable
        - Enum exports are enums (SelectionMode)
        - Convenience functions have documented parameter signatures
        - ``__version__`` follows semver pattern
        - Public API surface stability (minimum expected names present)

Design:
    - Zero ``sys.modules`` pollution: this suite mocks nothing; it relies only
      on a real import of the subpackage plus introspection.  No ``@patch`` and
      no ``sys.modules[...]`` assignment are used, so it cannot pollute the
      global import system or break sibling test files during collection.
    - Deterministic: no filesystem, network, or GPU access required.
    - Isolated: each test is independent; execution order is irrelevant.
    - Fast: expected < 5 s total wall-clock on CI.

Launch:
    From project root (/app/milia):
        pytest tests/test__init__models_factory.py -v --tb=short

Markers:
    smoke     — Quick health-check tests (§1)
    contract  — Interface/contract validation tests (§2)
"""

import enum
import importlib
import inspect
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure the project root is importable
# ---------------------------------------------------------------------------
# When launched via ``pytest tests/test__init__models_factory.py`` from the
# project root (/app/milia), ``milia_pipeline`` must be on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # …/tests -> …/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture(scope="module")
def factory_pkg():
    """
    Import and return the ``milia_pipeline.models.factory`` package once
    per module.

    This fixture validates the fundamental smoke invariant: the factory
    subpackage is importable.  If this fails, every downstream test is moot.
    """
    try:
        import milia_pipeline.models.factory as fct

        return fct
    except ImportError as exc:
        pytest.fail(
            f"milia_pipeline.models.factory could not be imported — smoke "
            f"test precondition violated: {exc}"
        )


@pytest.fixture(scope="module")
def all_names(factory_pkg):
    """Return the ``__all__`` list from the factory package."""
    assert hasattr(factory_pkg, "__all__"), (
        "milia_pipeline.models.factory.__all__ is missing — contract violation"
    )
    return list(factory_pkg.__all__)


# ===================================================================
# SECTION 1 — SMOKE TESTS
# ===================================================================


class TestSmokeFactoryPackageImport:
    """§1.2 — Verify the factory subpackage imports without errors."""

    @pytest.mark.smoke
    def test_import_factory_package_succeeds(self, factory_pkg):
        """The factory package imports without raising any exception."""
        assert factory_pkg is not None

    @pytest.mark.smoke
    def test_factory_package_is_a_module(self, factory_pkg):
        """The imported object is a proper Python module."""
        assert isinstance(factory_pkg, types.ModuleType)

    @pytest.mark.smoke
    def test_factory_package_has_file_attribute(self, factory_pkg):
        """The package exposes a ``__file__`` attribute (not a namespace pkg)."""
        assert hasattr(factory_pkg, "__file__")

    @pytest.mark.smoke
    def test_factory_package_reimport_is_idempotent(self, factory_pkg):
        """Re-importing the factory package via reload does not crash."""
        reloaded = importlib.reload(factory_pkg)
        assert reloaded is not None
        assert hasattr(reloaded, "__all__")


class TestSmokeMetadataAttributes:
    """§1.2 — Verify module-level metadata attributes are present and typed."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "attr",
        [
            "__version__",
            "__author__",
        ],
    )
    def test_metadata_attribute_exists(self, factory_pkg, attr):
        """Each metadata dunder is defined on the factory package."""
        assert hasattr(factory_pkg, attr), f"Missing attribute: {attr}"

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "attr",
        [
            "__version__",
            "__author__",
        ],
    )
    def test_metadata_attribute_is_string(self, factory_pkg, attr):
        """Each metadata dunder is a non-empty string."""
        value = getattr(factory_pkg, attr)
        assert isinstance(value, str), f"{attr} should be str, got {type(value)}"
        assert len(value) > 0, f"{attr} should be non-empty"

    @pytest.mark.smoke
    def test_version_is_semver_like(self, factory_pkg):
        """``__version__`` follows a MAJOR.MINOR.PATCH pattern."""
        version = factory_pkg.__version__
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

    @pytest.mark.smoke
    def test_version_matches_package_source_of_truth(self, factory_pkg):
        """
        ``__version__`` tracks the canonical package version.

        The factory subpackage version must stay unified with
        ``milia_pipeline.__version__`` — the single source of truth that
        ``pyproject.toml`` reads dynamically. Asserting equality against that
        attribute (rather than a hardcoded literal) keeps this contract correct
        across future version bumps with no test edits required.
        """
        import milia_pipeline

        assert factory_pkg.__version__ == milia_pipeline.__version__, (
            f"factory __version__ ({factory_pkg.__version__}) must match the "
            f"canonical package version milia_pipeline.__version__ "
            f"({milia_pipeline.__version__})"
        )


class TestSmokeMainClassExports:
    """§1.2 — Main factory classes are accessible from the factory package."""

    MAIN_CLASS_EXPORTS = [
        "ModelFactory",
        "ModelValidator",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", MAIN_CLASS_EXPORTS)
    def test_main_class_export_exists(self, factory_pkg, name):
        """Each main class export is present and non-None."""
        obj = getattr(factory_pkg, name, None)
        assert obj is not None, f"Main class export '{name}' is None or missing"


class TestSmokeWrapperClassExports:
    """§1.2 — Task-level wrapper classes are accessible from the factory package."""

    WRAPPER_CLASS_EXPORTS = [
        "GraphLevelModelWrapper",
        "EdgeLevelModelWrapper",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", WRAPPER_CLASS_EXPORTS)
    def test_wrapper_class_export_exists(self, factory_pkg, name):
        """Each wrapper class export is present and non-None."""
        obj = getattr(factory_pkg, name, None)
        assert obj is not None, f"Wrapper class export '{name}' is None or missing"


class TestSmokePublicFunctionExports:
    """§1.2 — Public API functions are accessible and callable."""

    PUBLIC_FUNCTIONS = [
        "create_model",
        "get_model_info",
        "get_factory",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", PUBLIC_FUNCTIONS)
    def test_public_function_exists(self, factory_pkg, name):
        """Each public function export is present and non-None."""
        obj = getattr(factory_pkg, name, None)
        assert obj is not None, f"Public function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", PUBLIC_FUNCTIONS)
    def test_public_function_is_callable(self, factory_pkg, name):
        """Each public function export is callable."""
        obj = getattr(factory_pkg, name)
        assert callable(obj), f"Public function '{name}' should be callable"


class TestSmokeTargetSelectionExports:
    """§1.2 — Target-selection components are accessible from the factory package."""

    TARGET_SELECTION_EXPORTS = [
        "TargetSelectionConfig",
        "SelectionMode",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", TARGET_SELECTION_EXPORTS)
    def test_target_selection_export_exists(self, factory_pkg, name):
        """Each target-selection export is present and non-None."""
        obj = getattr(factory_pkg, name, None)
        assert obj is not None, f"Target-selection export '{name}' is None or missing"


class TestSmokeModuleInitialization:
    """§1.2 — Module-level introspection helpers behave as documented."""

    @pytest.mark.smoke
    def test_get_module_info_returns_dict(self, factory_pkg):
        """The private ``_get_module_info`` helper returns a dict with a version."""
        assert hasattr(factory_pkg, "_get_module_info"), (
            "_get_module_info introspection helper is missing"
        )
        info = factory_pkg._get_module_info()
        assert isinstance(info, dict), "_get_module_info() should return a dict"
        assert "version" in info, "_get_module_info() result missing 'version' key"

    @pytest.mark.smoke
    def test_get_module_info_version_matches_dunder(self, factory_pkg):
        """``_get_module_info()['version']`` matches ``__version__``."""
        info = factory_pkg._get_module_info()
        assert info["version"] == factory_pkg.__version__, (
            f"_get_module_info() version '{info['version']}' does not match "
            f"__version__ '{factory_pkg.__version__}'"
        )


class TestSmokeAllExportsGenericSweep:
    """§1.2 — Every name advertised in ``__all__`` resolves on the package."""

    @pytest.mark.smoke
    def test_every_all_entry_resolves(self, factory_pkg, all_names):
        """Each entry listed in ``__all__`` is a real, resolvable attribute."""
        unresolved = [name for name in all_names if not hasattr(factory_pkg, name)]
        assert not unresolved, f"Names in __all__ not resolvable on package: {unresolved}"


# ===================================================================
# SECTION 2 — CONTRACT TESTS
# ===================================================================


class TestContractAllCompleteness:
    """§2 — ``__all__`` is well-formed and every entry resolves."""

    # Minimum public names that MUST remain exported for API stability.
    EXPECTED_MINIMUM = {
        "ModelFactory",
        "ModelValidator",
        "GraphLevelModelWrapper",
        "EdgeLevelModelWrapper",
        "create_model",
        "get_model_info",
        "get_factory",
        "TargetSelectionConfig",
        "SelectionMode",
        "__version__",
    }

    @pytest.mark.contract
    def test_all_is_a_list(self, factory_pkg):
        """``__all__`` is declared as a list."""
        assert isinstance(factory_pkg.__all__, list), "__all__ should be a list"

    @pytest.mark.contract
    def test_all_contains_no_duplicates(self, all_names):
        """``__all__`` has no duplicate entries."""
        duplicates = [name for name in set(all_names) if all_names.count(name) > 1]
        assert not duplicates, f"__all__ contains duplicates: {sorted(duplicates)}"

    @pytest.mark.contract
    def test_all_entries_are_strings(self, all_names):
        """Every entry in ``__all__`` is a string."""
        non_strings = [n for n in all_names if not isinstance(n, str)]
        assert not non_strings, f"__all__ entries must be strings; offenders: {non_strings}"

    @pytest.mark.contract
    def test_all_entries_resolve(self, factory_pkg, all_names):
        """Every name in ``__all__`` resolves to a real attribute."""
        unresolved = [name for name in all_names if not hasattr(factory_pkg, name)]
        assert not unresolved, f"__all__ names not resolvable: {unresolved}"

    @pytest.mark.contract
    def test_all_includes_expected_minimum(self, all_names):
        """``__all__`` exposes at least the documented stable surface."""
        missing = self.EXPECTED_MINIMUM - set(all_names)
        assert not missing, f"__all__ missing expected stable names: {sorted(missing)}"


class TestContractAllConsistency:
    """§2 — Every public import in the factory module is listed in ``__all__``."""

    # Names that are intentionally public-ish but NOT in __all__.
    KNOWN_UNLISTED = {
        # Metadata dunder (not listed besides __version__)
        "__author__",
        # Module-level logger instance
        "logger",
        # typing imports used at module level
        "Any",
    }

    @pytest.mark.contract
    def test_public_imports_are_in_all(self, factory_pkg, all_names):
        """
        Every non-dunder, non-private attribute that was imported (not a
        submodule reference) should be in ``__all__`` — unless it is in
        ``KNOWN_UNLISTED``.

        This catches accidental omissions when new imports are added to the
        factory ``__init__.py`` without updating ``__all__``.
        """
        all_set = set(all_names)
        module_dict = vars(factory_pkg)
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
            # Skip private names NOT in __all__
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
            f"Public names imported in factory/__init__.py but not in __all__: "
            f"{sorted(missing_from_all)}"
        )


class TestContractClassExports:
    """§2 — Class exports are actually classes."""

    CLASS_EXPORTS = [
        "ModelFactory",
        "ModelValidator",
        "GraphLevelModelWrapper",
        "EdgeLevelModelWrapper",
        "TargetSelectionConfig",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", CLASS_EXPORTS)
    def test_class_export_is_a_class(self, factory_pkg, name):
        """Each advertised class export is a ``type`` (class object)."""
        obj = getattr(factory_pkg, name)
        assert isinstance(obj, type), f"Export '{name}' should be a class, got {type(obj)}"


class TestContractEnumExports:
    """§2 — Enum exports are enums."""

    @pytest.mark.contract
    def test_selection_mode_is_enum(self, factory_pkg):
        """``SelectionMode`` is an ``enum.Enum`` subclass."""
        obj = factory_pkg.SelectionMode
        assert isinstance(obj, type) and issubclass(obj, enum.Enum), (
            "SelectionMode should be an enum.Enum subclass"
        )

    @pytest.mark.contract
    def test_selection_mode_has_members(self, factory_pkg):
        """``SelectionMode`` declares at least one member."""
        members = list(factory_pkg.SelectionMode)
        assert len(members) > 0, "SelectionMode should declare at least one member"


class TestContractFunctionSignatures:
    """§2 — Convenience functions are callables exposing parameters."""

    FUNCTION_EXPORTS = [
        "create_model",
        "get_model_info",
        "get_factory",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", FUNCTION_EXPORTS)
    def test_function_export_is_callable(self, factory_pkg, name):
        """Each advertised function export is callable."""
        obj = getattr(factory_pkg, name)
        assert callable(obj), f"Export '{name}' should be callable"

    @pytest.mark.contract
    @pytest.mark.parametrize("name", FUNCTION_EXPORTS)
    def test_function_signature_is_introspectable(self, factory_pkg, name):
        """Each function export exposes an introspectable signature."""
        obj = getattr(factory_pkg, name)
        signature = inspect.signature(obj)
        assert signature is not None, f"Function '{name}' should expose a signature"


class TestContractVersionFormat:
    """§2 — ``__version__`` follows the documented semver contract."""

    @pytest.mark.contract
    def test_version_has_three_components(self, factory_pkg):
        """``__version__`` has exactly three MAJOR.MINOR.PATCH components."""
        version = factory_pkg.__version__
        parts = version.split(".")
        assert len(parts) == 3, (
            f"Expected 3 version components (MAJOR.MINOR.PATCH), got {len(parts)} in '{version}'"
        )

    @pytest.mark.contract
    def test_version_in_all(self, all_names):
        """``__version__`` is listed in ``__all__``."""
        assert "__version__" in all_names, "__version__ should be listed in __all__"
