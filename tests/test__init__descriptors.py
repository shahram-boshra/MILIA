# tests/test__init__descriptors.py

"""
Test Suite: milia_pipeline/descriptors/__init__.py — Smoke Tests & Contract Tests
=================================================================================

Production-ready test suite for the MILIA Pipeline descriptors package
``milia_pipeline/descriptors/__init__.py``.

Covers:
    Section 1 — Smoke Tests (MILIA_Test_Recommendations.md §1.2 scope):
        - The ``milia_pipeline.descriptors`` subpackage imports without ImportError
        - All re-exported names from the 6 submodules are accessible
        - Module-level metadata attributes (__version__) exist
        - Module initialization (logging, registry info) runs safely
        - Re-import (``importlib.reload``) is idempotent and non-crashing
        - Registry System exports are accessible (DescriptorRegistry, etc.)
        - Categories and Metadata exports are accessible
        - Calculator exports are accessible
        - Validator exports are accessible
        - Integration exports are accessible
        - Plugin System exports are accessible

    Section 2 — Contract Tests (MILIA_Test_Recommendations.md §2 scope):
        - ``__all__`` completeness: every name in ``__all__`` is resolvable
        - ``__all__`` consistency: every public import is listed in ``__all__``
        - ``__all__`` has no duplicates
        - Re-exported classes are classes, callables are callable
        - DescriptorCategory is an enum-like class
        - Registry singleton ``registry`` is an instance of DescriptorRegistry
        - Validator singleton ``validator`` is an instance of DescriptorValidator
        - Plugin loader singleton ``plugin_loader`` is an instance of
          DescriptorPluginLoader
        - Factory functions and convenience functions are callable with
          documented parameter signatures
        - Integration functions are callable
        - Constants (DESCRIPTOR_METADATA_MAP, ALL_DESCRIPTORS, DESCRIPTORS_BY_CATEGORY)
          are dict-like
        - ``__version__`` follows semver pattern
        - Public API surface stability (minimum expected names present)
        - Namespace cleanliness

Design:
    - Zero ``sys.modules`` pollution: all mocking via ``@patch`` decorators
      or context-manager ``patch`` calls scoped to individual tests.
    - Deterministic: no filesystem, network, or GPU access required.
    - Isolated: each test is independent; execution order is irrelevant.
    - Fast: expected < 5 s total wall-clock on CI.

Launch:
    From project root (/app/milia):
        pytest tests/test__init__descriptors.py -v --tb=short

Markers:
    smoke     — Quick health-check tests (§1)
    contract  — Interface/contract validation tests (§2)
"""

import importlib
import inspect
import sys
import types
import logging
import re
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure the project root is importable
# ---------------------------------------------------------------------------
# When launched via ``pytest tests/test__init__descriptors.py`` from the
# project root (/app/milia), ``milia_pipeline`` must be on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # …/tests -> …/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def desc_pkg():
    """
    Import and return the ``milia_pipeline.descriptors`` package once per module.

    This fixture validates the fundamental smoke invariant: the descriptors
    subpackage is importable.  If this fails, every downstream test is moot.
    """
    try:
        import milia_pipeline.descriptors as desc
        return desc
    except ImportError as exc:
        pytest.fail(
            f"milia_pipeline.descriptors could not be imported — smoke test "
            f"precondition violated: {exc}"
        )


@pytest.fixture(scope="module")
def all_names(desc_pkg):
    """Return the ``__all__`` list from the descriptors package."""
    assert hasattr(desc_pkg, "__all__"), (
        "milia_pipeline.descriptors.__all__ is missing — contract violation"
    )
    return list(desc_pkg.__all__)


# ===================================================================
# SECTION 1 — SMOKE TESTS
# ===================================================================


class TestSmokeDescriptorsPackageImport:
    """§1.2 — Verify the descriptors subpackage imports without errors."""

    @pytest.mark.smoke
    def test_import_descriptors_package_succeeds(self, desc_pkg):
        """The descriptors package imports without raising any exception."""
        assert desc_pkg is not None

    @pytest.mark.smoke
    def test_descriptors_package_is_a_module(self, desc_pkg):
        """The imported object is a proper Python module."""
        assert isinstance(desc_pkg, types.ModuleType)

    @pytest.mark.smoke
    def test_descriptors_package_has_file_attribute(self, desc_pkg):
        """The package exposes a ``__file__`` attribute (not a namespace pkg)."""
        assert hasattr(desc_pkg, "__file__")

    @pytest.mark.smoke
    def test_descriptors_package_name(self, desc_pkg):
        """The package ``__name__`` is ``milia_pipeline.descriptors``."""
        assert desc_pkg.__name__ == "milia_pipeline.descriptors"


class TestSmokeMetadataAttributes:
    """§1.2 — Verify module-level metadata attributes are present and typed."""

    @pytest.mark.smoke
    def test_version_attribute_exists(self, desc_pkg):
        """``__version__`` is defined on the descriptors package."""
        assert hasattr(desc_pkg, "__version__"), "Missing attribute: __version__"

    @pytest.mark.smoke
    def test_version_attribute_is_string(self, desc_pkg):
        """``__version__`` is a non-empty string."""
        value = desc_pkg.__version__
        assert isinstance(value, str), (
            f"__version__ should be str, got {type(value)}"
        )
        assert len(value) > 0, "__version__ should be non-empty"

    @pytest.mark.smoke
    def test_version_is_semver_like(self, desc_pkg):
        """``__version__`` follows a MAJOR.MINOR.PATCH pattern."""
        version = desc_pkg.__version__
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

    @pytest.mark.smoke
    def test_version_value_is_1_0_0(self, desc_pkg):
        """``__version__`` is '1.0.0' as documented in the module."""
        assert desc_pkg.__version__ == "1.0.0"


class TestSmokeRegistrySystemExports:
    """§1.2 — Registry System exports from descriptor_registry.py are accessible."""

    REGISTRY_EXPORTS = [
        "DescriptorRegistry",
        "DescriptorRegistration",
        "registry",
        "get_descriptor",
        "has_descriptor",
        "list_descriptors",
        "auto_discover_rdkit",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", REGISTRY_EXPORTS)
    def test_registry_export_exists(self, desc_pkg, name):
        """Each registry system export is present and non-None."""
        obj = getattr(desc_pkg, name, None)
        assert obj is not None, f"Registry export '{name}' is None or missing"


class TestSmokeCategoriesAndMetadataExports:
    """§1.2 — Categories and Metadata exports from descriptor_categories.py."""

    CATEGORY_CLASSES = [
        "DescriptorCategory",
        "DescriptorMetadata",
    ]

    CATEGORY_FUNCTIONS = [
        "get_descriptors_by_category",
        "get_descriptor_metadata",
        "requires_3d_coordinates",
        "requires_partial_charges",
        "get_all_descriptor_names",
        "get_category_descriptor_names",
        "filter_descriptors_by_requirements",
        "get_descriptor_count_by_category",
        "validate_descriptor_coverage",
    ]

    CATEGORY_CONSTANTS = [
        "DESCRIPTOR_METADATA_MAP",
        "ALL_DESCRIPTORS",
        "DESCRIPTORS_BY_CATEGORY",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CATEGORY_CLASSES)
    def test_category_class_exists(self, desc_pkg, name):
        """Each category class is importable from the descriptors package."""
        obj = getattr(desc_pkg, name, None)
        assert obj is not None, f"Category class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CATEGORY_FUNCTIONS)
    def test_category_function_exists(self, desc_pkg, name):
        """Each category function is present and non-None."""
        obj = getattr(desc_pkg, name, None)
        assert obj is not None, f"Category function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CATEGORY_FUNCTIONS)
    def test_category_function_is_callable(self, desc_pkg, name):
        """Each category function is callable."""
        obj = getattr(desc_pkg, name)
        assert callable(obj), f"'{name}' should be callable"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CATEGORY_CONSTANTS)
    def test_category_constant_exists(self, desc_pkg, name):
        """Each category constant is defined on the descriptors package."""
        assert hasattr(desc_pkg, name), f"Constant '{name}' is missing"


class TestSmokeCalculatorExports:
    """§1.2 — Calculator exports from descriptor_calculator.py are accessible."""

    CALCULATOR_CLASSES = [
        "DescriptorCalculator",
        "CalculationResult",
        "BatchCalculationResult",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CALCULATOR_CLASSES)
    def test_calculator_class_exists(self, desc_pkg, name):
        """Each calculator class is importable from the descriptors package."""
        obj = getattr(desc_pkg, name, None)
        assert obj is not None, f"Calculator class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CALCULATOR_CLASSES)
    def test_calculator_class_is_a_class(self, desc_pkg, name):
        """Each calculator export is a class (not an instance or function)."""
        obj = getattr(desc_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )


class TestSmokeValidatorExports:
    """§1.2 — Validator exports from descriptor_validator.py are accessible."""

    VALIDATOR_CLASSES = [
        "DescriptorValidator",
        "ValidationResult",
    ]

    VALIDATOR_FUNCTIONS = [
        "validate_value",
        "check_requirements",
        "filter_by_requirements",
    ]

    VALIDATOR_SINGLETONS = [
        "validator",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", VALIDATOR_CLASSES)
    def test_validator_class_exists(self, desc_pkg, name):
        """Each validator class is importable from the descriptors package."""
        obj = getattr(desc_pkg, name, None)
        assert obj is not None, f"Validator class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", VALIDATOR_FUNCTIONS)
    def test_validator_function_exists(self, desc_pkg, name):
        """Each validator function is present and non-None."""
        obj = getattr(desc_pkg, name, None)
        assert obj is not None, f"Validator function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", VALIDATOR_FUNCTIONS)
    def test_validator_function_is_callable(self, desc_pkg, name):
        """Each validator function is callable."""
        obj = getattr(desc_pkg, name)
        assert callable(obj), f"'{name}' should be callable"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", VALIDATOR_SINGLETONS)
    def test_validator_singleton_exists(self, desc_pkg, name):
        """Each validator singleton is present and non-None."""
        obj = getattr(desc_pkg, name, None)
        assert obj is not None, f"Validator singleton '{name}' is None or missing"


class TestSmokeIntegrationExports:
    """§1.2 — Integration exports from descriptor_integration.py are accessible."""

    INTEGRATION_FUNCTIONS = [
        "descriptors_to_tensor",
        "add_descriptors_to_pyg_data",
        "merge_descriptors_with_features",
        "extract_descriptors_from_pyg_data",
        "validate_descriptor_integration",
        "get_descriptor_statistics",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", INTEGRATION_FUNCTIONS)
    def test_integration_function_exists(self, desc_pkg, name):
        """Each integration function is present and non-None."""
        obj = getattr(desc_pkg, name, None)
        assert obj is not None, f"Integration function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", INTEGRATION_FUNCTIONS)
    def test_integration_function_is_callable(self, desc_pkg, name):
        """Each integration function is callable."""
        obj = getattr(desc_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokePluginSystemExports:
    """§1.2 — Plugin System exports from descriptor_plugin_system.py are accessible."""

    PLUGIN_CLASSES = [
        "DescriptorPluginLoader",
        "DescriptorPluginMetadata",
        "DescriptorDeclaration",
    ]

    PLUGIN_FUNCTIONS = [
        "discover_plugins",
        "validate_plugin",
        "list_plugins",
        "get_plugin_info",
    ]

    PLUGIN_SINGLETONS = [
        "plugin_loader",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", PLUGIN_CLASSES)
    def test_plugin_class_exists(self, desc_pkg, name):
        """Each plugin class is importable from the descriptors package."""
        obj = getattr(desc_pkg, name, None)
        assert obj is not None, f"Plugin class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", PLUGIN_CLASSES)
    def test_plugin_class_is_a_class(self, desc_pkg, name):
        """Each plugin export is a class (not an instance or function)."""
        obj = getattr(desc_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", PLUGIN_FUNCTIONS)
    def test_plugin_function_exists(self, desc_pkg, name):
        """Each plugin function is present and non-None."""
        obj = getattr(desc_pkg, name, None)
        assert obj is not None, f"Plugin function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", PLUGIN_FUNCTIONS)
    def test_plugin_function_is_callable(self, desc_pkg, name):
        """Each plugin function is callable."""
        obj = getattr(desc_pkg, name)
        assert callable(obj), f"'{name}' should be callable"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", PLUGIN_SINGLETONS)
    def test_plugin_singleton_exists(self, desc_pkg, name):
        """Each plugin singleton is present and non-None."""
        obj = getattr(desc_pkg, name, None)
        assert obj is not None, f"Plugin singleton '{name}' is None or missing"


class TestSmokeModuleInitialization:
    """§1.2 — Module-level initialization runs without side-effect crashes."""

    @pytest.mark.smoke
    def test_reimport_is_idempotent(self, desc_pkg):
        """
        Re-importing the descriptors package (via ``importlib.reload``) does
        not crash.

        Validates that all module-level code (logging, registry info logging)
        is safe to re-execute.
        """
        reloaded = importlib.reload(desc_pkg)
        assert reloaded is not None
        assert hasattr(reloaded, "__version__")

    @pytest.mark.smoke
    def test_reimport_preserves_all(self, desc_pkg):
        """
        Re-importing the descriptors package preserves ``__all__``.
        """
        reloaded = importlib.reload(desc_pkg)
        assert hasattr(reloaded, "__all__")
        assert isinstance(reloaded.__all__, list)
        assert len(reloaded.__all__) > 0

    @pytest.mark.smoke
    def test_reimport_preserves_version(self, desc_pkg):
        """
        Re-importing the descriptors package preserves ``__version__``.
        """
        reloaded = importlib.reload(desc_pkg)
        assert reloaded.__version__ == "1.0.0"

    @pytest.mark.smoke
    def test_module_docstring_exists(self, desc_pkg):
        """The descriptors package has a module-level docstring."""
        assert desc_pkg.__doc__ is not None
        assert len(desc_pkg.__doc__) > 0

    @pytest.mark.smoke
    def test_module_docstring_mentions_descriptors(self, desc_pkg):
        """The docstring references 'descriptor' to confirm it's the right module."""
        assert "descriptor" in desc_pkg.__doc__.lower() or "Descriptor" in desc_pkg.__doc__


class TestSmokeAllExportsAccessible:
    """§1.2 — Quick sweep: every name in ``__all__`` is accessible."""

    @pytest.mark.smoke
    def test_all_exports_are_non_none(self, desc_pkg, all_names):
        """
        Every name in ``__all__`` resolves to a non-None attribute on
        the descriptors package.

        This is the broadest smoke test: if any export is broken (e.g.,
        missing submodule, renamed class), this catches it immediately.
        """
        missing = []
        for name in all_names:
            if not hasattr(desc_pkg, name):
                missing.append(name)
        assert not missing, (
            f"Names in __all__ that are not accessible on the module: {missing}"
        )


# ===================================================================
# SECTION 2 — CONTRACT TESTS
# ===================================================================


class TestContractAllCompleteness:
    """§2 — Every name in ``__all__`` is resolvable on the descriptors package."""

    @pytest.mark.contract
    def test_all_is_a_list(self, desc_pkg):
        """``__all__`` is a list (not a tuple or set)."""
        assert isinstance(desc_pkg.__all__, list)

    @pytest.mark.contract
    def test_all_contains_no_duplicates(self, all_names):
        """``__all__`` has no duplicate entries.

        Unlike the config module, the descriptors ``__init__.py`` has a
        cleanly organized ``__all__`` with no known duplicates. Any
        duplicate is unexpected and should be flagged.
        """
        seen = set()
        duplicates = []
        for name in all_names:
            if name in seen:
                duplicates.append(name)
            seen.add(name)

        assert not duplicates, (
            f"Unexpected duplicate entries in __all__: {duplicates}"
        )

    @pytest.mark.contract
    def test_every_all_entry_is_resolvable(self, desc_pkg, all_names):
        """
        Generic sweep: every single entry in ``__all__`` must be resolvable,
        regardless of whether it is parameterized individually.
        """
        unresolvable = [
            name for name in all_names
            if not hasattr(desc_pkg, name)
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
    """§2 — Every public import in the descriptors module is listed in ``__all__``."""

    # Names that are intentionally public-ish but NOT in __all__
    KNOWN_UNLISTED = {
        # Metadata dunders (not typically in __all__)
        "__version__",
        # Internal logger
        "logger",
    }

    @pytest.mark.contract
    def test_public_imports_are_in_all(self, desc_pkg, all_names):
        """
        Every non-dunder, non-private attribute that was imported (not
        defined locally) should be in ``__all__`` — unless it is in
        ``KNOWN_UNLISTED``.

        This catches accidental omissions when new imports are added to
        the descriptors ``__init__.py`` without updating ``__all__``.
        """
        all_set = set(all_names)
        module_dict = vars(desc_pkg)
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
            # Skip private names that start with underscore and are NOT in __all__
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
            f"Public names imported in descriptors/__init__.py but not in __all__: "
            f"{sorted(missing_from_all)}"
        )


class TestContractVersionInAll:
    """§2 — ``__version__`` is listed in ``__all__``."""

    @pytest.mark.contract
    def test_version_in_all(self, all_names):
        """``__version__`` is included in ``__all__``."""
        assert "__version__" in all_names, (
            "__version__ should be listed in __all__ as documented in the source"
        )


class TestContractRegistryClassTypes:
    """§2 — Registry System exports have the correct types."""

    @pytest.mark.contract
    def test_descriptor_registry_is_class(self, desc_pkg):
        """``DescriptorRegistry`` is a class."""
        assert inspect.isclass(desc_pkg.DescriptorRegistry), (
            f"DescriptorRegistry should be a class, got "
            f"{type(desc_pkg.DescriptorRegistry).__name__}"
        )

    @pytest.mark.contract
    def test_descriptor_registration_is_class(self, desc_pkg):
        """``DescriptorRegistration`` is a class."""
        assert inspect.isclass(desc_pkg.DescriptorRegistration), (
            f"DescriptorRegistration should be a class, got "
            f"{type(desc_pkg.DescriptorRegistration).__name__}"
        )

    @pytest.mark.contract
    def test_registry_singleton_is_instance_of_descriptor_registry(self, desc_pkg):
        """
        ``registry`` is an instance of ``DescriptorRegistry``.

        The descriptor_registry.py module exposes a module-level singleton
        ``registry`` that should be an instance of ``DescriptorRegistry``.
        """
        assert isinstance(desc_pkg.registry, desc_pkg.DescriptorRegistry), (
            f"registry should be an instance of DescriptorRegistry, got "
            f"{type(desc_pkg.registry).__name__}"
        )

    @pytest.mark.contract
    def test_get_descriptor_is_callable(self, desc_pkg):
        """``get_descriptor`` is callable."""
        assert callable(desc_pkg.get_descriptor)

    @pytest.mark.contract
    def test_has_descriptor_is_callable(self, desc_pkg):
        """``has_descriptor`` is callable."""
        assert callable(desc_pkg.has_descriptor)

    @pytest.mark.contract
    def test_list_descriptors_is_callable(self, desc_pkg):
        """``list_descriptors`` is callable."""
        assert callable(desc_pkg.list_descriptors)

    @pytest.mark.contract
    def test_auto_discover_rdkit_is_callable(self, desc_pkg):
        """``auto_discover_rdkit`` is callable."""
        assert callable(desc_pkg.auto_discover_rdkit)


class TestContractRegistryFunctionSignatures:
    """§2 — Registry convenience functions have proper signatures."""

    @pytest.mark.contract
    def test_get_descriptor_accepts_name_parameter(self, desc_pkg):
        """``get_descriptor`` accepts at least one parameter (descriptor name)."""
        sig = inspect.signature(desc_pkg.get_descriptor)
        assert len(sig.parameters) >= 1, (
            "get_descriptor should accept at least one parameter"
        )

    @pytest.mark.contract
    def test_has_descriptor_accepts_name_parameter(self, desc_pkg):
        """``has_descriptor`` accepts at least one parameter (descriptor name)."""
        sig = inspect.signature(desc_pkg.has_descriptor)
        assert len(sig.parameters) >= 1, (
            "has_descriptor should accept at least one parameter"
        )


class TestContractCategoryClassTypes:
    """§2 — Category and Metadata exports have the correct types."""

    @pytest.mark.contract
    def test_descriptor_category_is_class(self, desc_pkg):
        """``DescriptorCategory`` is a class."""
        assert inspect.isclass(desc_pkg.DescriptorCategory), (
            f"DescriptorCategory should be a class, got "
            f"{type(desc_pkg.DescriptorCategory).__name__}"
        )

    @pytest.mark.contract
    def test_descriptor_category_is_enum_like(self, desc_pkg):
        """
        ``DescriptorCategory`` behaves like an enum (has members that can
        be iterated or accessed as attributes).

        Per the __init__.py docs: 6 categories — Constitutional, Topological,
        Electronic, Geometric, Drug-likeness, Fragments.
        """
        cls = desc_pkg.DescriptorCategory
        # Check that it has at least one of the documented category names
        # as an attribute or member
        documented_categories = [
            "CONSTITUTIONAL", "TOPOLOGICAL", "ELECTRONIC",
            "GEOMETRIC",
        ]
        found_any = False
        for cat_name in documented_categories:
            if hasattr(cls, cat_name):
                found_any = True
                break
        assert found_any, (
            f"DescriptorCategory should have at least one of "
            f"{documented_categories} as a member/attribute"
        )

    @pytest.mark.contract
    def test_descriptor_metadata_is_class(self, desc_pkg):
        """``DescriptorMetadata`` is a class."""
        assert inspect.isclass(desc_pkg.DescriptorMetadata), (
            f"DescriptorMetadata should be a class, got "
            f"{type(desc_pkg.DescriptorMetadata).__name__}"
        )


class TestContractCategoryFunctionTypes:
    """§2 — Category functions are functions (not classes)."""

    CATEGORY_FUNCTIONS = [
        "get_descriptors_by_category",
        "get_descriptor_metadata",
        "requires_3d_coordinates",
        "requires_partial_charges",
        "get_all_descriptor_names",
        "get_category_descriptor_names",
        "filter_descriptors_by_requirements",
        "get_descriptor_count_by_category",
        "validate_descriptor_coverage",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", CATEGORY_FUNCTIONS)
    def test_category_function_is_function_or_callable(self, desc_pkg, name):
        """Each category function is a function or callable object."""
        obj = getattr(desc_pkg, name)
        assert callable(obj), (
            f"'{name}' should be callable, got {type(obj).__name__}"
        )


class TestContractCategoryConstants:
    """§2 — Category constants have correct types."""

    @pytest.mark.contract
    def test_descriptor_metadata_map_is_dict_like(self, desc_pkg):
        """``DESCRIPTOR_METADATA_MAP`` is a dict or dict-like mapping."""
        obj = desc_pkg.DESCRIPTOR_METADATA_MAP
        assert isinstance(obj, dict), (
            f"DESCRIPTOR_METADATA_MAP should be a dict, got {type(obj).__name__}"
        )

    @pytest.mark.contract
    def test_all_descriptors_is_collection(self, desc_pkg):
        """``ALL_DESCRIPTORS`` is a list, tuple, set, frozenset, or dict."""
        obj = desc_pkg.ALL_DESCRIPTORS
        assert isinstance(obj, (list, tuple, set, frozenset, dict)), (
            f"ALL_DESCRIPTORS should be a collection, got {type(obj).__name__}"
        )

    @pytest.mark.contract
    def test_descriptors_by_category_is_dict_like(self, desc_pkg):
        """``DESCRIPTORS_BY_CATEGORY`` is a dict or dict-like mapping."""
        obj = desc_pkg.DESCRIPTORS_BY_CATEGORY
        assert isinstance(obj, dict), (
            f"DESCRIPTORS_BY_CATEGORY should be a dict, got {type(obj).__name__}"
        )

    @pytest.mark.contract
    def test_descriptor_metadata_map_is_non_empty(self, desc_pkg):
        """``DESCRIPTOR_METADATA_MAP`` is non-empty (400+ descriptors documented)."""
        obj = desc_pkg.DESCRIPTOR_METADATA_MAP
        assert len(obj) > 0, (
            "DESCRIPTOR_METADATA_MAP should be non-empty (400+ descriptors documented)"
        )

    @pytest.mark.contract
    def test_descriptors_by_category_is_non_empty(self, desc_pkg):
        """``DESCRIPTORS_BY_CATEGORY`` is non-empty (6 categories documented)."""
        obj = desc_pkg.DESCRIPTORS_BY_CATEGORY
        assert len(obj) > 0, (
            "DESCRIPTORS_BY_CATEGORY should be non-empty (6 categories documented)"
        )


class TestContractCalculatorClassTypes:
    """§2 — Calculator classes are classes with correct nature."""

    CALCULATOR_CLASSES = [
        "DescriptorCalculator",
        "CalculationResult",
        "BatchCalculationResult",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", CALCULATOR_CLASSES)
    def test_calculator_is_class(self, desc_pkg, name):
        """Each calculator export is a class."""
        obj = getattr(desc_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )


class TestContractValidatorClassTypes:
    """§2 — Validator classes and singletons have correct types."""

    @pytest.mark.contract
    def test_descriptor_validator_is_class(self, desc_pkg):
        """``DescriptorValidator`` is a class."""
        assert inspect.isclass(desc_pkg.DescriptorValidator), (
            f"DescriptorValidator should be a class, got "
            f"{type(desc_pkg.DescriptorValidator).__name__}"
        )

    @pytest.mark.contract
    def test_validation_result_is_class(self, desc_pkg):
        """``ValidationResult`` is a class."""
        assert inspect.isclass(desc_pkg.ValidationResult), (
            f"ValidationResult should be a class, got "
            f"{type(desc_pkg.ValidationResult).__name__}"
        )

    @pytest.mark.contract
    def test_validator_singleton_is_instance_of_descriptor_validator(self, desc_pkg):
        """
        ``validator`` is an instance of ``DescriptorValidator``.

        The descriptor_validator.py module exposes a module-level singleton
        ``validator`` that should be an instance of ``DescriptorValidator``.
        """
        assert isinstance(desc_pkg.validator, desc_pkg.DescriptorValidator), (
            f"validator should be an instance of DescriptorValidator, got "
            f"{type(desc_pkg.validator).__name__}"
        )

    @pytest.mark.contract
    def test_validate_value_is_function_or_callable(self, desc_pkg):
        """``validate_value`` is callable."""
        assert callable(desc_pkg.validate_value)

    @pytest.mark.contract
    def test_check_requirements_is_callable(self, desc_pkg):
        """``check_requirements`` is callable."""
        assert callable(desc_pkg.check_requirements)

    @pytest.mark.contract
    def test_filter_by_requirements_is_callable(self, desc_pkg):
        """``filter_by_requirements`` is callable."""
        assert callable(desc_pkg.filter_by_requirements)


class TestContractValidatorFunctionSignatures:
    """§2 — Validator functions have proper parameter signatures."""

    @pytest.mark.contract
    def test_validate_value_accepts_parameters(self, desc_pkg):
        """``validate_value`` accepts at least one parameter."""
        sig = inspect.signature(desc_pkg.validate_value)
        assert len(sig.parameters) >= 1, (
            "validate_value should accept at least one parameter"
        )

    @pytest.mark.contract
    def test_check_requirements_accepts_parameters(self, desc_pkg):
        """``check_requirements`` accepts at least one parameter."""
        sig = inspect.signature(desc_pkg.check_requirements)
        assert len(sig.parameters) >= 1, (
            "check_requirements should accept at least one parameter"
        )

    @pytest.mark.contract
    def test_filter_by_requirements_accepts_parameters(self, desc_pkg):
        """``filter_by_requirements`` accepts at least one parameter."""
        sig = inspect.signature(desc_pkg.filter_by_requirements)
        assert len(sig.parameters) >= 1, (
            "filter_by_requirements should accept at least one parameter"
        )


class TestContractIntegrationFunctionTypes:
    """§2 — Integration functions are functions (not classes)."""

    INTEGRATION_FUNCTIONS = [
        "descriptors_to_tensor",
        "add_descriptors_to_pyg_data",
        "merge_descriptors_with_features",
        "extract_descriptors_from_pyg_data",
        "validate_descriptor_integration",
        "get_descriptor_statistics",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", INTEGRATION_FUNCTIONS)
    def test_integration_function_is_callable(self, desc_pkg, name):
        """Each integration function is callable."""
        obj = getattr(desc_pkg, name)
        assert callable(obj), (
            f"'{name}' should be callable, got {type(obj).__name__}"
        )


class TestContractIntegrationFunctionSignatures:
    """§2 — Key integration functions have documented parameter signatures."""

    @pytest.mark.contract
    def test_descriptors_to_tensor_accepts_parameters(self, desc_pkg):
        """``descriptors_to_tensor`` accepts at least one parameter."""
        sig = inspect.signature(desc_pkg.descriptors_to_tensor)
        assert len(sig.parameters) >= 1, (
            "descriptors_to_tensor should accept at least one parameter"
        )

    @pytest.mark.contract
    def test_add_descriptors_to_pyg_data_accepts_parameters(self, desc_pkg):
        """``add_descriptors_to_pyg_data`` accepts at least two parameters."""
        sig = inspect.signature(desc_pkg.add_descriptors_to_pyg_data)
        assert len(sig.parameters) >= 2, (
            "add_descriptors_to_pyg_data should accept at least two parameters "
            "(data and descriptors)"
        )

    @pytest.mark.contract
    def test_merge_descriptors_with_features_accepts_parameters(self, desc_pkg):
        """``merge_descriptors_with_features`` accepts at least one parameter."""
        sig = inspect.signature(desc_pkg.merge_descriptors_with_features)
        assert len(sig.parameters) >= 1, (
            "merge_descriptors_with_features should accept at least one parameter"
        )


class TestContractPluginClassTypes:
    """§2 — Plugin System classes have correct types."""

    @pytest.mark.contract
    def test_descriptor_plugin_loader_is_class(self, desc_pkg):
        """``DescriptorPluginLoader`` is a class."""
        assert inspect.isclass(desc_pkg.DescriptorPluginLoader), (
            f"DescriptorPluginLoader should be a class, got "
            f"{type(desc_pkg.DescriptorPluginLoader).__name__}"
        )

    @pytest.mark.contract
    def test_descriptor_plugin_metadata_is_class(self, desc_pkg):
        """``DescriptorPluginMetadata`` is a class."""
        assert inspect.isclass(desc_pkg.DescriptorPluginMetadata), (
            f"DescriptorPluginMetadata should be a class, got "
            f"{type(desc_pkg.DescriptorPluginMetadata).__name__}"
        )

    @pytest.mark.contract
    def test_descriptor_declaration_is_class(self, desc_pkg):
        """``DescriptorDeclaration`` is a class."""
        assert inspect.isclass(desc_pkg.DescriptorDeclaration), (
            f"DescriptorDeclaration should be a class, got "
            f"{type(desc_pkg.DescriptorDeclaration).__name__}"
        )

    @pytest.mark.contract
    def test_plugin_loader_singleton_is_instance_of_descriptor_plugin_loader(self, desc_pkg):
        """
        ``plugin_loader`` is an instance of ``DescriptorPluginLoader``.

        The descriptor_plugin_system.py module exposes a module-level singleton
        ``plugin_loader`` that should be an instance of ``DescriptorPluginLoader``.
        """
        assert isinstance(desc_pkg.plugin_loader, desc_pkg.DescriptorPluginLoader), (
            f"plugin_loader should be an instance of DescriptorPluginLoader, got "
            f"{type(desc_pkg.plugin_loader).__name__}"
        )


class TestContractPluginFunctionTypes:
    """§2 — Plugin convenience functions are callable."""

    PLUGIN_FUNCTIONS = [
        "discover_plugins",
        "validate_plugin",
        "list_plugins",
        "get_plugin_info",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", PLUGIN_FUNCTIONS)
    def test_plugin_function_is_callable(self, desc_pkg, name):
        """Each plugin function is callable."""
        obj = getattr(desc_pkg, name)
        assert callable(obj), (
            f"'{name}' should be callable, got {type(obj).__name__}"
        )


class TestContractPluginFunctionSignatures:
    """§2 — Plugin functions have proper parameter signatures."""

    @pytest.mark.contract
    def test_validate_plugin_accepts_parameters(self, desc_pkg):
        """``validate_plugin`` accepts at least one parameter."""
        sig = inspect.signature(desc_pkg.validate_plugin)
        assert len(sig.parameters) >= 1, (
            "validate_plugin should accept at least one parameter"
        )

    @pytest.mark.contract
    def test_get_plugin_info_accepts_parameters(self, desc_pkg):
        """``get_plugin_info`` accepts at least one parameter."""
        sig = inspect.signature(desc_pkg.get_plugin_info)
        assert len(sig.parameters) >= 1, (
            "get_plugin_info should accept at least one parameter"
        )


class TestContractRegistrySingletonHasListAll:
    """§2 — The ``registry`` singleton has the ``list_all_descriptors`` method."""

    @pytest.mark.contract
    def test_registry_has_list_all_descriptors_method(self, desc_pkg):
        """
        ``registry.list_all_descriptors()`` is callable.

        The __init__.py module-level code calls
        ``registry.list_all_descriptors()`` during initialization, so this
        method must exist.
        """
        assert hasattr(desc_pkg.registry, "list_all_descriptors"), (
            "registry should have a 'list_all_descriptors' method "
            "(called during module initialization)"
        )
        assert callable(desc_pkg.registry.list_all_descriptors), (
            "registry.list_all_descriptors should be callable"
        )

    @pytest.mark.contract
    def test_registry_list_all_descriptors_returns_collection(self, desc_pkg):
        """
        ``registry.list_all_descriptors()`` returns a collection whose
        ``len()`` is defined (used in the __init__.py logging statement).
        """
        result = desc_pkg.registry.list_all_descriptors()
        # The __init__.py does len(registry.list_all_descriptors()),
        # so the result must support len()
        try:
            count = len(result)
        except TypeError:
            pytest.fail(
                "registry.list_all_descriptors() must return a sized collection "
                "(supports len())"
            )
        # The project documents 400+ descriptors
        assert count >= 0, (
            f"registry.list_all_descriptors() returned a collection with "
            f"negative length — this should not happen"
        )


class TestContractPublicAPISurface:
    """
    §2 — Public API surface stability.

    Ensures that the minimum expected public API is present in ``__all__``.
    Guards against accidental removals during refactoring.
    """

    # The minimum API surface that MUST be present in __all__
    MINIMUM_API = {
        # Version
        "__version__",

        # Registry System
        "DescriptorRegistry",
        "DescriptorRegistration",
        "registry",
        "get_descriptor",
        "has_descriptor",
        "list_descriptors",
        "auto_discover_rdkit",

        # Categories and Metadata
        "DescriptorCategory",
        "DescriptorMetadata",
        "get_descriptors_by_category",
        "get_descriptor_metadata",
        "requires_3d_coordinates",
        "requires_partial_charges",
        "get_all_descriptor_names",
        "get_category_descriptor_names",
        "filter_descriptors_by_requirements",
        "get_descriptor_count_by_category",
        "validate_descriptor_coverage",
        "DESCRIPTOR_METADATA_MAP",
        "ALL_DESCRIPTORS",
        "DESCRIPTORS_BY_CATEGORY",

        # Calculator
        "DescriptorCalculator",
        "CalculationResult",
        "BatchCalculationResult",

        # Validator
        "DescriptorValidator",
        "ValidationResult",
        "validate_value",
        "check_requirements",
        "filter_by_requirements",
        "validator",

        # Integration
        "descriptors_to_tensor",
        "add_descriptors_to_pyg_data",
        "merge_descriptors_with_features",
        "extract_descriptors_from_pyg_data",
        "validate_descriptor_integration",
        "get_descriptor_statistics",

        # Plugin System
        "DescriptorPluginLoader",
        "DescriptorPluginMetadata",
        "DescriptorDeclaration",
        "plugin_loader",
        "discover_plugins",
        "validate_plugin",
        "list_plugins",
        "get_plugin_info",
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

        Based on the __init__.py source, the descriptors package exports
        approximately 45 names. This test guards against catastrophic
        loss (e.g., accidental truncation of __all__) while allowing for
        organic growth.
        """
        actual = len(all_names)
        # The __init__.py has approximately 45 entries in __all__
        # We set a floor well below the actual count to allow changes
        # while catching catastrophic loss.
        MINIMUM_EXPECTED = 35
        assert actual >= MINIMUM_EXPECTED, (
            f"__all__ has {actual} entries, expected at least {MINIMUM_EXPECTED}. "
            f"This suggests __all__ may have been accidentally truncated."
        )

    @pytest.mark.contract
    def test_all_has_exact_documented_count(self, all_names):
        """
        ``__all__`` has 45 entries as defined in the source code.

        This is an exact count test. If the count changes, it means
        someone added or removed exports and should also update this
        test to acknowledge the change.
        """
        actual = len(all_names)
        EXPECTED_COUNT = 45
        assert actual == EXPECTED_COUNT, (
            f"__all__ has {actual} entries, expected exactly {EXPECTED_COUNT}. "
            f"If exports were intentionally added or removed, update this test."
        )


class TestContractSubmoduleOrganization:
    """
    §2 — Verify the 6 submodules documented in the __init__.py are
    importable individually.

    Per the project structure:
        descriptors/
        ├── __init__.py
        ├── descriptor_registry.py
        ├── descriptor_categories.py
        ├── descriptor_calculator.py
        ├── descriptor_validator.py
        ├── descriptor_integration.py
        └── descriptor_plugin_system.py
    """

    SUBMODULES = [
        "milia_pipeline.descriptors.descriptor_registry",
        "milia_pipeline.descriptors.descriptor_categories",
        "milia_pipeline.descriptors.descriptor_calculator",
        "milia_pipeline.descriptors.descriptor_validator",
        "milia_pipeline.descriptors.descriptor_integration",
        "milia_pipeline.descriptors.descriptor_plugin_system",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("submodule", SUBMODULES)
    def test_submodule_is_importable(self, submodule):
        """Each documented submodule can be imported without error."""
        try:
            importlib.import_module(submodule)
        except ImportError as exc:
            pytest.fail(
                f"Submodule '{submodule}' could not be imported: {exc}"
            )


class TestContractCrossSubmoduleConsistency:
    """
    §2 — Verify that the top-level descriptors package re-exports are
    consistent with what the submodules actually define.
    """

    @pytest.mark.contract
    def test_descriptor_registry_class_is_from_registry_submodule(self, desc_pkg):
        """
        ``DescriptorRegistry`` on the descriptors package is the same class
        as ``DescriptorRegistry`` from the descriptor_registry submodule.
        """
        import milia_pipeline.descriptors.descriptor_registry as reg_mod
        assert desc_pkg.DescriptorRegistry is reg_mod.DescriptorRegistry, (
            "DescriptorRegistry on the package should be the same object "
            "as in descriptor_registry submodule"
        )

    @pytest.mark.contract
    def test_descriptor_calculator_class_is_from_calculator_submodule(self, desc_pkg):
        """
        ``DescriptorCalculator`` on the descriptors package is the same class
        as ``DescriptorCalculator`` from the descriptor_calculator submodule.
        """
        import milia_pipeline.descriptors.descriptor_calculator as calc_mod
        assert desc_pkg.DescriptorCalculator is calc_mod.DescriptorCalculator, (
            "DescriptorCalculator on the package should be the same object "
            "as in descriptor_calculator submodule"
        )

    @pytest.mark.contract
    def test_descriptor_category_class_is_from_categories_submodule(self, desc_pkg):
        """
        ``DescriptorCategory`` on the descriptors package is the same class
        as ``DescriptorCategory`` from the descriptor_categories submodule.
        """
        import milia_pipeline.descriptors.descriptor_categories as cat_mod
        assert desc_pkg.DescriptorCategory is cat_mod.DescriptorCategory, (
            "DescriptorCategory on the package should be the same object "
            "as in descriptor_categories submodule"
        )

    @pytest.mark.contract
    def test_descriptor_validator_class_is_from_validator_submodule(self, desc_pkg):
        """
        ``DescriptorValidator`` on the descriptors package is the same class
        as ``DescriptorValidator`` from the descriptor_validator submodule.
        """
        import milia_pipeline.descriptors.descriptor_validator as val_mod
        assert desc_pkg.DescriptorValidator is val_mod.DescriptorValidator, (
            "DescriptorValidator on the package should be the same object "
            "as in descriptor_validator submodule"
        )

    @pytest.mark.contract
    def test_descriptor_plugin_loader_class_is_from_plugin_submodule(self, desc_pkg):
        """
        ``DescriptorPluginLoader`` on the descriptors package is the same class
        as ``DescriptorPluginLoader`` from the descriptor_plugin_system submodule.
        """
        import milia_pipeline.descriptors.descriptor_plugin_system as plugin_mod
        assert desc_pkg.DescriptorPluginLoader is plugin_mod.DescriptorPluginLoader, (
            "DescriptorPluginLoader on the package should be the same object "
            "as in descriptor_plugin_system submodule"
        )


class TestContractSingletonIdentity:
    """
    §2 — Singletons exported from the package are identical to those
    in their respective submodules (not copies).
    """

    @pytest.mark.contract
    def test_registry_singleton_identity(self, desc_pkg):
        """
        ``registry`` on the descriptors package is the same object (``is``)
        as ``registry`` from the descriptor_registry submodule.
        """
        import milia_pipeline.descriptors.descriptor_registry as reg_mod
        assert desc_pkg.registry is reg_mod.registry, (
            "registry singleton on the package should be the same object "
            "as in descriptor_registry submodule (identity, not equality)"
        )

    @pytest.mark.contract
    def test_validator_singleton_identity(self, desc_pkg):
        """
        ``validator`` on the descriptors package is the same object (``is``)
        as ``validator`` from the descriptor_validator submodule.
        """
        import milia_pipeline.descriptors.descriptor_validator as val_mod
        assert desc_pkg.validator is val_mod.validator, (
            "validator singleton on the package should be the same object "
            "as in descriptor_validator submodule (identity, not equality)"
        )

    @pytest.mark.contract
    def test_plugin_loader_singleton_identity(self, desc_pkg):
        """
        ``plugin_loader`` on the descriptors package is the same object (``is``)
        as ``plugin_loader`` from the descriptor_plugin_system submodule.
        """
        import milia_pipeline.descriptors.descriptor_plugin_system as plugin_mod
        assert desc_pkg.plugin_loader is plugin_mod.plugin_loader, (
            "plugin_loader singleton on the package should be the same object "
            "as in descriptor_plugin_system submodule (identity, not equality)"
        )


class TestContractNamespaceCleanlinessLogging:
    """
    §2 — The ``logging`` module is retained in namespace since the
    __init__.py defines ``logger`` at module level without deleting
    ``logging``. Verify logger is a Logger instance.
    """

    @pytest.mark.contract
    def test_logger_is_proper_logger(self, desc_pkg):
        """
        If ``logger`` is present on the descriptors package, it should be
        a ``logging.Logger`` instance (not the logging module itself).
        """
        if hasattr(desc_pkg, "logger"):
            obj = desc_pkg.logger
            assert isinstance(obj, logging.Logger), (
                f"logger should be a logging.Logger instance, got "
                f"{type(obj).__name__}"
            )


class TestContractDescriptorCalculatorStructure:
    """§2 — DescriptorCalculator class has expected documented methods."""

    @pytest.mark.contract
    def test_calculator_has_calculate_batch_method(self, desc_pkg):
        """
        ``DescriptorCalculator`` has a ``calculate_batch`` method.

        Per the __init__.py usage examples:
            result = calculator.calculate_batch(mol, descriptors)
        """
        assert hasattr(desc_pkg.DescriptorCalculator, "calculate_batch"), (
            "DescriptorCalculator should have a 'calculate_batch' method "
            "(documented in usage examples)"
        )

    @pytest.mark.contract
    def test_calculator_calculate_batch_accepts_parameters(self, desc_pkg):
        """``DescriptorCalculator.calculate_batch`` accepts parameters."""
        method = desc_pkg.DescriptorCalculator.calculate_batch
        sig = inspect.signature(method)
        # Expect at least 'self', 'mol', 'descriptors'
        param_count = len(sig.parameters)
        assert param_count >= 2, (
            f"calculate_batch should accept at least 2 parameters "
            f"(mol, descriptors), got {param_count} "
            f"(params: {list(sig.parameters.keys())})"
        )


class TestContractVersionFormat:
    """§2 — ``__version__`` format validation."""

    @pytest.mark.contract
    def test_version_matches_semver_regex(self, desc_pkg):
        """
        ``__version__`` matches a semantic versioning pattern
        (MAJOR.MINOR.PATCH with optional pre-release suffix).
        """
        version = desc_pkg.__version__
        # Accept patterns like: 1.0.0, 1.2.3, 1.0.0-alpha, 1.0.0rc1
        pattern = r"^\d+\.\d+\.\d+([a-zA-Z0-9._-]*)?$"
        assert re.match(pattern, version), (
            f"__version__ '{version}' does not match semver pattern "
            f"MAJOR.MINOR.PATCH"
        )
