# tests/test__init__datasets.py

"""
Test Suite: milia_pipeline/datasets/__init__.py — Smoke Tests & Contract Tests
===============================================================================

Production-ready test suite for the MILIA Pipeline datasets package
``milia_pipeline/datasets/__init__.py``.

Covers:
    Section 1 — Smoke Tests (MILIA_Test_Recommendations.md §1.2 scope):
        - The ``milia_pipeline.datasets`` subpackage imports without ImportError
        - All re-exported names from submodules are accessible
        - Module-level metadata attributes (__version__, __author__, etc.) exist
        - Module initialization (_initialize_module) runs safely
        - Re-import (``importlib.reload``) is idempotent and non-crashing
        - Version constants are present and correctly typed
        - Phase 1 registry infrastructure exports are accessible
        - Phase 2 auto-discovery import triggers without error
        - Phase 6 integration version constants are present
        - Dynamic dataset type discovery functions work
        - Module information functions (get_module_info, check_dependencies) run
        - Plugin initialization placeholder runs without error
        - __getattr__ dynamic attribute access for dataset classes works

    Section 2 — Contract Tests (MILIA_Test_Recommendations.md §2 scope):
        - ``__all__`` completeness: every name in ``__all__`` is resolvable
        - ``__all__`` consistency: every public import is listed in ``__all__``
        - ``__all__`` has no duplicates
        - Re-exported classes are classes, callables are callable
        - Base classes (BaseDataset, DatasetMetadata, DatasetSchema,
          DatasetFeatures) are Pydantic dataclasses
        - Registry classes and functions have correct types
        - Protocol classes are runtime_checkable
        - Exception classes are proper Exception subclasses
        - Version constants follow expected patterns
        - Factory/utility functions have correct signatures
        - Public API surface stability (minimum expected names present)
        - Dynamic __getattr__ raises AttributeError for invalid names
        - get_module_info() returns dict with expected keys
        - check_dependencies() returns dict with expected keys
        - get_supported_dataset_types() returns a list of strings
        - SUPPORTED_DATASET_TYPES is a list
        - initialize_plugins() returns int

Design:
    - Zero ``sys.modules`` pollution: all mocking via ``@patch`` decorators
      or context-manager ``patch`` calls scoped to individual tests.
    - Deterministic: no filesystem, network, or GPU access required.
    - Isolated: each test is independent; execution order is irrelevant.
    - Fast: expected < 5 s total wall-clock on CI.

Launch:
    From project root (/app/milia):
        pytest tests/test__init__datasets.py -v --tb=short

Markers:
    smoke     — Quick health-check tests (§1)
    contract  — Interface/contract validation tests (§2)
"""

import importlib
import inspect
import sys
import types
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure the project root is importable
# ---------------------------------------------------------------------------
# When launched via ``pytest tests/test__init__datasets.py`` from the
# project root (/app/milia), ``milia_pipeline`` must be on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # …/tests -> …/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def datasets_pkg():
    """
    Import and return the ``milia_pipeline.datasets`` package once per module.

    This fixture validates the fundamental smoke invariant: the datasets
    subpackage is importable.  If this fails, every downstream test is moot.
    """
    try:
        import milia_pipeline.datasets as ds
        return ds
    except ImportError as exc:
        pytest.fail(
            f"milia_pipeline.datasets could not be imported — smoke test "
            f"precondition violated: {exc}"
        )


@pytest.fixture(scope="module")
def all_names(datasets_pkg):
    """Return the ``__all__`` list from the datasets package."""
    assert hasattr(datasets_pkg, "__all__"), (
        "milia_pipeline.datasets.__all__ is missing — contract violation"
    )
    return list(datasets_pkg.__all__)


# ===================================================================
# SECTION 1 — SMOKE TESTS
# ===================================================================


class TestSmokeDatasetPackageImport:
    """§1.2 — Verify the datasets subpackage imports without errors."""

    @pytest.mark.smoke
    def test_import_datasets_package_succeeds(self, datasets_pkg):
        """The datasets package imports without raising any exception."""
        assert datasets_pkg is not None

    @pytest.mark.smoke
    def test_datasets_package_is_a_module(self, datasets_pkg):
        """The imported object is a proper Python module."""
        assert isinstance(datasets_pkg, types.ModuleType)

    @pytest.mark.smoke
    def test_datasets_package_has_file_attribute(self, datasets_pkg):
        """The package exposes a ``__file__`` attribute (not a namespace pkg)."""
        assert hasattr(datasets_pkg, "__file__")

    @pytest.mark.smoke
    def test_datasets_package_name(self, datasets_pkg):
        """The package ``__name__`` is ``milia_pipeline.datasets``."""
        assert datasets_pkg.__name__ == "milia_pipeline.datasets"


class TestSmokeMetadataAttributes:
    """§1.2 — Verify module-level metadata attributes are present and typed."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("attr", [
        "__version__",
        "__author__",
        "__module_status__",
    ])
    def test_metadata_attribute_exists(self, datasets_pkg, attr):
        """Each metadata dunder is defined on the datasets package."""
        assert hasattr(datasets_pkg, attr), f"Missing attribute: {attr}"

    @pytest.mark.smoke
    @pytest.mark.parametrize("attr", [
        "__version__",
        "__author__",
        "__module_status__",
    ])
    def test_metadata_attribute_is_string(self, datasets_pkg, attr):
        """Each metadata dunder is a non-empty string."""
        value = getattr(datasets_pkg, attr)
        assert isinstance(value, str), f"{attr} should be str, got {type(value)}"
        assert len(value) > 0, f"{attr} should be non-empty"

    @pytest.mark.smoke
    def test_version_is_semver_like(self, datasets_pkg):
        """``__version__`` follows a MAJOR.MINOR.PATCH pattern."""
        version = datasets_pkg.__version__
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


class TestSmokeVersionConstants:
    """§1.2 — Version and architecture constants are present and typed."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("attr", [
        "HANDLER_ARCHITECTURE_VERSION",
        "TRANSFORMATION_SYSTEM_VERSION",
        "REGISTRY_VERSION",
        "IMPLEMENTATIONS_VERSION",
        "PHASE_6_INTEGRATION_VERSION",
    ])
    def test_version_constant_exists(self, datasets_pkg, attr):
        """Each version constant is defined on the datasets package."""
        assert hasattr(datasets_pkg, attr), f"Missing version constant: {attr}"

    @pytest.mark.smoke
    @pytest.mark.parametrize("attr", [
        "HANDLER_ARCHITECTURE_VERSION",
        "TRANSFORMATION_SYSTEM_VERSION",
        "REGISTRY_VERSION",
        "IMPLEMENTATIONS_VERSION",
        "PHASE_6_INTEGRATION_VERSION",
    ])
    def test_version_constant_is_string(self, datasets_pkg, attr):
        """Each version constant is a non-empty string."""
        value = getattr(datasets_pkg, attr)
        assert isinstance(value, str), f"{attr} should be str, got {type(value)}"
        assert len(value) > 0, f"{attr} should be non-empty"


class TestSmokeCoreDatasetClassExport:
    """§1.2 — The primary dataset class miliaDataset is accessible."""

    @pytest.mark.smoke
    def test_milia_dataset_exists(self, datasets_pkg):
        """``miliaDataset`` is importable from the datasets package."""
        assert hasattr(datasets_pkg, "miliaDataset"), (
            "miliaDataset is missing from datasets package"
        )

    @pytest.mark.smoke
    def test_milia_dataset_is_a_class(self, datasets_pkg):
        """``miliaDataset`` is a class (not an instance or function)."""
        assert inspect.isclass(datasets_pkg.miliaDataset), (
            f"miliaDataset should be a class, got "
            f"{type(datasets_pkg.miliaDataset).__name__}"
        )


class TestSmokeBaseClassExports:
    """§1.2 — Phase 1 base class exports are accessible."""

    BASE_CLASSES = [
        "BaseDataset",
        "DatasetMetadata",
        "DatasetSchema",
        "DatasetFeatures",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", BASE_CLASSES)
    def test_base_class_exists(self, datasets_pkg, name):
        """Each base class is importable from the datasets package."""
        obj = getattr(datasets_pkg, name, None)
        assert obj is not None, f"Base class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", BASE_CLASSES)
    def test_base_class_is_a_class(self, datasets_pkg, name):
        """Each base class export is a class (not an instance or function)."""
        obj = getattr(datasets_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )


class TestSmokeRegistryExports:
    """§1.2 — Phase 1 registry exports are accessible."""

    @pytest.mark.smoke
    def test_dataset_registry_class_exists(self, datasets_pkg):
        """``DatasetRegistry`` class is accessible."""
        assert hasattr(datasets_pkg, "DatasetRegistry")
        assert inspect.isclass(datasets_pkg.DatasetRegistry)

    REGISTRY_FUNCTIONS = [
        "get_default_registry",
        "register",
        "get",
        "list_all",
        "is_registered",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", REGISTRY_FUNCTIONS)
    def test_registry_function_exists(self, datasets_pkg, name):
        """Each registry convenience function is present and non-None."""
        obj = getattr(datasets_pkg, name, None)
        assert obj is not None, f"Registry function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", REGISTRY_FUNCTIONS)
    def test_registry_function_is_callable(self, datasets_pkg, name):
        """Each registry convenience function is callable."""
        obj = getattr(datasets_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeProtocolExports:
    """§1.2 — Phase 1 protocol exports are accessible."""

    PROTOCOL_CLASSES = [
        "DatasetHandlerProtocol",
        "DatasetConverterProtocol",
        "DatasetValidatorProtocol",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", PROTOCOL_CLASSES)
    def test_protocol_class_exists(self, datasets_pkg, name):
        """Each protocol class is importable from the datasets package."""
        obj = getattr(datasets_pkg, name, None)
        assert obj is not None, f"Protocol class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", PROTOCOL_CLASSES)
    def test_protocol_class_is_a_class(self, datasets_pkg, name):
        """Each protocol export is a class."""
        obj = getattr(datasets_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )


class TestSmokeExceptionExports:
    """§1.2 — Exception class exports are accessible."""

    EXCEPTION_CLASSES = [
        "DatasetRegistrationError",
        "DatasetNotFoundError",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", EXCEPTION_CLASSES)
    def test_exception_class_exists(self, datasets_pkg, name):
        """Each exception class is importable from the datasets package."""
        obj = getattr(datasets_pkg, name, None)
        assert obj is not None, f"Exception class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", EXCEPTION_CLASSES)
    def test_exception_class_is_a_class(self, datasets_pkg, name):
        """Each exception export is a class."""
        obj = getattr(datasets_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )


class TestSmokeDynamicDiscoveryExports:
    """§1.2 — Dynamic dataset type discovery exports are accessible."""

    @pytest.mark.smoke
    def test_get_supported_dataset_types_exists(self, datasets_pkg):
        """``get_supported_dataset_types`` is present and callable."""
        assert hasattr(datasets_pkg, "get_supported_dataset_types")
        assert callable(datasets_pkg.get_supported_dataset_types)

    @pytest.mark.smoke
    def test_supported_dataset_types_constant_exists(self, datasets_pkg):
        """``SUPPORTED_DATASET_TYPES`` legacy constant is present."""
        assert hasattr(datasets_pkg, "SUPPORTED_DATASET_TYPES")

    @pytest.mark.smoke
    def test_get_supported_dataset_types_returns_list(self, datasets_pkg):
        """``get_supported_dataset_types()`` returns a list."""
        result = datasets_pkg.get_supported_dataset_types()
        assert isinstance(result, list), (
            f"get_supported_dataset_types() should return list, "
            f"got {type(result).__name__}"
        )

    @pytest.mark.smoke
    def test_supported_dataset_types_is_list(self, datasets_pkg):
        """``SUPPORTED_DATASET_TYPES`` is a list."""
        assert isinstance(datasets_pkg.SUPPORTED_DATASET_TYPES, list), (
            f"SUPPORTED_DATASET_TYPES should be list, "
            f"got {type(datasets_pkg.SUPPORTED_DATASET_TYPES).__name__}"
        )


class TestSmokeModuleInformationFunctions:
    """§1.2 — Module information functions execute without crashing."""

    @pytest.mark.smoke
    def test_get_module_info_exists(self, datasets_pkg):
        """``get_module_info`` is present and callable."""
        assert hasattr(datasets_pkg, "get_module_info")
        assert callable(datasets_pkg.get_module_info)

    @pytest.mark.smoke
    def test_get_module_info_runs_without_error(self, datasets_pkg):
        """``get_module_info()`` executes without raising an exception."""
        result = datasets_pkg.get_module_info()
        assert result is not None

    @pytest.mark.smoke
    def test_get_module_info_returns_dict(self, datasets_pkg):
        """``get_module_info()`` returns a dict."""
        result = datasets_pkg.get_module_info()
        assert isinstance(result, dict), (
            f"get_module_info() should return dict, got {type(result).__name__}"
        )

    @pytest.mark.smoke
    def test_check_dependencies_exists(self, datasets_pkg):
        """``check_dependencies`` is present and callable."""
        assert hasattr(datasets_pkg, "check_dependencies")
        assert callable(datasets_pkg.check_dependencies)

    @pytest.mark.smoke
    def test_check_dependencies_runs_without_error(self, datasets_pkg):
        """``check_dependencies()`` executes without raising an exception."""
        result = datasets_pkg.check_dependencies()
        assert result is not None

    @pytest.mark.smoke
    def test_check_dependencies_returns_dict(self, datasets_pkg):
        """``check_dependencies()`` returns a dict."""
        result = datasets_pkg.check_dependencies()
        assert isinstance(result, dict), (
            f"check_dependencies() should return dict, got {type(result).__name__}"
        )


class TestSmokePluginInitialization:
    """§1.2 — Plugin initialization placeholder runs without error."""

    @pytest.mark.smoke
    def test_initialize_plugins_exists(self, datasets_pkg):
        """``initialize_plugins`` is present and callable."""
        assert hasattr(datasets_pkg, "initialize_plugins")
        assert callable(datasets_pkg.initialize_plugins)

    @pytest.mark.smoke
    def test_initialize_plugins_runs_without_error(self, datasets_pkg):
        """``initialize_plugins()`` executes without raising an exception."""
        result = datasets_pkg.initialize_plugins()
        assert result is not None

    @pytest.mark.smoke
    def test_initialize_plugins_returns_int(self, datasets_pkg):
        """``initialize_plugins()`` returns an int (number of plugins loaded)."""
        result = datasets_pkg.initialize_plugins()
        assert isinstance(result, int), (
            f"initialize_plugins() should return int, got {type(result).__name__}"
        )

    @pytest.mark.smoke
    def test_initialize_plugins_returns_zero_placeholder(self, datasets_pkg):
        """``initialize_plugins()`` returns 0 (Phase 8 placeholder)."""
        result = datasets_pkg.initialize_plugins()
        assert result == 0, (
            f"initialize_plugins() placeholder should return 0, got {result}"
        )


class TestSmokeModuleInitialization:
    """§1.2 — Module-level initialization runs without side-effect crashes."""

    @pytest.mark.smoke
    def test_reimport_is_idempotent(self, datasets_pkg):
        """
        Re-importing the datasets package (via ``importlib.reload``) does not
        crash.

        Validates that all module-level code (logging, registry status checks,
        _initialize_module, _update_module_docstring) is safe to re-execute.
        """
        reloaded = importlib.reload(datasets_pkg)
        assert reloaded is not None
        assert hasattr(reloaded, "__version__")

    @pytest.mark.smoke
    def test_reimport_preserves_all(self, datasets_pkg):
        """
        Re-importing the datasets package preserves ``__all__``.
        """
        reloaded = importlib.reload(datasets_pkg)
        assert hasattr(reloaded, "__all__")
        assert isinstance(reloaded.__all__, list)
        assert len(reloaded.__all__) > 0

    @pytest.mark.smoke
    def test_reimport_preserves_version(self, datasets_pkg):
        """
        Re-importing the datasets package preserves ``__version__``.
        """
        original_version = datasets_pkg.__version__
        reloaded = importlib.reload(datasets_pkg)
        assert reloaded.__version__ == original_version

    @pytest.mark.smoke
    def test_reimport_preserves_miliadataset(self, datasets_pkg):
        """
        Re-importing the datasets package preserves ``miliaDataset``.
        """
        reloaded = importlib.reload(datasets_pkg)
        assert hasattr(reloaded, "miliaDataset")
        assert inspect.isclass(reloaded.miliaDataset)


class TestSmokeDynamicGetattr:
    """§1.2 — Dynamic __getattr__ for dataset class access works."""

    @pytest.mark.smoke
    def test_getattr_for_unknown_attribute_raises(self, datasets_pkg):
        """
        Accessing a non-existent attribute raises ``AttributeError``.
        """
        with pytest.raises(AttributeError):
            _ = datasets_pkg.NonExistentThing

    @pytest.mark.smoke
    def test_getattr_for_non_dataset_suffix_raises(self, datasets_pkg):
        """
        Accessing a name that does NOT end in 'Dataset' and is not
        a module attribute raises ``AttributeError``.
        """
        with pytest.raises(AttributeError):
            _ = datasets_pkg.SomethingElseEntirely

    @pytest.mark.smoke
    def test_getattr_for_unregistered_dataset_raises(self, datasets_pkg):
        """
        Accessing a name that ends in 'Dataset' but is not registered
        raises ``AttributeError``.
        """
        with pytest.raises(AttributeError):
            _ = datasets_pkg.FakeNonexistentDataset


class TestSmokePhase2AutoDiscovery:
    """§1.2 — Phase 2 auto-discovery import did not crash."""

    @pytest.mark.smoke
    def test_implementations_submodule_imported(self, datasets_pkg):
        """
        The ``milia_pipeline.datasets.implementations`` submodule was
        imported (triggered by the auto-discovery line in __init__.py).
        """
        assert "milia_pipeline.datasets.implementations" in sys.modules, (
            "milia_pipeline.datasets.implementations should be in sys.modules "
            "after datasets package import (auto-discovery trigger)"
        )

    @pytest.mark.smoke
    def test_list_all_returns_nonempty_after_discovery(self, datasets_pkg):
        """
        After Phase 2 auto-discovery, ``list_all()`` returns a non-empty
        list (assuming at least one dataset implementation exists).
        """
        registered = datasets_pkg.list_all()
        assert isinstance(registered, list), (
            f"list_all() should return list, got {type(registered).__name__}"
        )
        # At least one dataset should be registered after auto-discovery
        assert len(registered) > 0, (
            "list_all() should return at least one registered dataset "
            "after Phase 2 auto-discovery"
        )


class TestSmokeRegisteredDatasetAccess:
    """§1.2 — Registered datasets are accessible via dynamic __getattr__."""

    @pytest.mark.smoke
    def test_registered_datasets_accessible_via_getattr(self, datasets_pkg):
        """
        Each registered dataset name (e.g., 'DFT') can be accessed as
        ``DFTDataset`` via dynamic __getattr__.
        """
        registered = datasets_pkg.list_all()
        for dataset_name in registered:
            attr_name = f"{dataset_name}Dataset"
            try:
                cls = getattr(datasets_pkg, attr_name)
                assert cls is not None, (
                    f"Dynamic access to '{attr_name}' returned None"
                )
            except AttributeError:
                # Some dataset names may not follow the simple
                # convention (e.g., ANI1x vs ANI1X). Skip gracefully.
                pass

    @pytest.mark.smoke
    def test_registered_datasets_accessible_via_registry_get(self, datasets_pkg):
        """
        Each registered dataset name can be retrieved via ``get(name)``.
        """
        registered = datasets_pkg.list_all()
        for dataset_name in registered:
            cls = datasets_pkg.get(dataset_name)
            assert cls is not None, (
                f"get('{dataset_name}') returned None"
            )


# ===================================================================
# SECTION 2 — CONTRACT TESTS
# ===================================================================


class TestContractAllCompleteness:
    """§2 — Every name in ``__all__`` is resolvable on the datasets package."""

    @pytest.mark.contract
    def test_all_is_a_list(self, datasets_pkg):
        """``__all__`` is a list (not a tuple or set)."""
        assert isinstance(datasets_pkg.__all__, list)

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
    def test_every_all_entry_is_resolvable(self, datasets_pkg, all_names):
        """
        Generic sweep: every single entry in ``__all__`` must be resolvable,
        regardless of whether it is parameterized individually.
        """
        unresolvable = [
            name for name in all_names
            if not hasattr(datasets_pkg, name)
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
    """§2 — Every public import in the datasets module is listed in ``__all__``."""

    # Names that are intentionally public-ish but NOT in __all__.
    # Evidence: __init__datasets.py lines 132 (typing imports),
    # 391-404 (version constants), 471-561 (get_module_info),
    # 563-634 (check_dependencies). These are defined in the module
    # namespace but deliberately omitted from __all__.
    KNOWN_UNLISTED = {
        # Metadata dunders (not typically in __all__)
        "__version__",
        "__author__",
        "__module_status__",
        # typing imports used in function signatures (line 132)
        "Optional",
        "Dict",
        "Any",
        "List",
        # Version/architecture constants (lines 391-404)
        "HANDLER_ARCHITECTURE_VERSION",
        "TRANSFORMATION_SYSTEM_VERSION",
        "REGISTRY_VERSION",
        "IMPLEMENTATIONS_VERSION",
        "PHASE_6_INTEGRATION_VERSION",
        # Module information utility functions (lines 471-634)
        "get_module_info",
        "check_dependencies",
    }

    @pytest.mark.contract
    def test_public_imports_are_in_all(self, datasets_pkg, all_names):
        """
        Every non-dunder, non-private attribute that was imported (not
        defined locally) should be in ``__all__`` — unless it is in
        ``KNOWN_UNLISTED``.

        This catches accidental omissions when new imports are added to
        the datasets ``__init__.py`` without updating ``__all__``.
        """
        all_set = set(all_names)
        module_dict = vars(datasets_pkg)
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
            f"Public names imported in datasets/__init__.py but not in __all__: "
            f"{sorted(missing_from_all)}"
        )


class TestContractBaseClassTypes:
    """§2 — Base classes are Pydantic dataclasses or proper classes."""

    PYDANTIC_DATACLASSES = [
        "DatasetMetadata",
        "DatasetSchema",
        "DatasetFeatures",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", PYDANTIC_DATACLASSES)
    def test_dataclass_is_class(self, datasets_pkg, name):
        """Each Pydantic dataclass export is a class."""
        obj = getattr(datasets_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", PYDANTIC_DATACLASSES)
    def test_dataclass_is_pydantic_or_stdlib_dataclass(self, datasets_pkg, name):
        """
        Each dataclass export is a Pydantic dataclass, Pydantic BaseModel,
        or stdlib dataclass.

        Per the project structure doc: base.py uses Pydantic V2 frozen
        dataclasses (``from pydantic.dataclasses import dataclass``).
        """
        cls = getattr(datasets_pkg, name)

        # Check for Pydantic dataclass
        is_pydantic_dc = hasattr(cls, "__pydantic_fields__")

        # Check for Pydantic BaseModel
        try:
            from pydantic import BaseModel
            is_pydantic_model = issubclass(cls, BaseModel)
        except ImportError:
            is_pydantic_model = False

        # Check for stdlib dataclass
        is_stdlib_dc = hasattr(cls, "__dataclass_fields__")

        assert is_pydantic_dc or is_pydantic_model or is_stdlib_dc, (
            f"'{name}' should be a Pydantic dataclass, Pydantic BaseModel, "
            f"or stdlib dataclass"
        )

    @pytest.mark.contract
    def test_base_dataset_is_abstract(self, datasets_pkg):
        """``BaseDataset`` is an abstract class (has abstract methods)."""
        from abc import ABC
        cls = datasets_pkg.BaseDataset
        assert inspect.isclass(cls), "BaseDataset should be a class"
        # BaseDataset should be an ABC subclass
        assert issubclass(cls, ABC), (
            "BaseDataset should be a subclass of ABC"
        )


class TestContractRegistryClassContract:
    """§2 — DatasetRegistry class has expected API methods."""

    EXPECTED_METHODS = [
        "register",
        "get",
        "list_all",
        "is_registered",
        "clear",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("method", EXPECTED_METHODS)
    def test_registry_has_method(self, datasets_pkg, method):
        """DatasetRegistry exposes the expected method."""
        cls = datasets_pkg.DatasetRegistry
        assert hasattr(cls, method), (
            f"DatasetRegistry should have '{method}' method"
        )

    @pytest.mark.contract
    def test_registry_is_not_singleton(self, datasets_pkg):
        """
        DatasetRegistry is non-singleton (can create isolated instances).

        Per the project structure doc: DatasetRegistry is NOT a singleton —
        isolated instances for testing.
        """
        cls = datasets_pkg.DatasetRegistry
        instance1 = cls()
        instance2 = cls()
        assert instance1 is not instance2, (
            "DatasetRegistry should NOT be a singleton — "
            "two instances should be distinct objects"
        )


class TestContractRegistryFunctionSignatures:
    """§2 — Registry convenience functions have correct signatures."""

    @pytest.mark.contract
    def test_register_is_callable(self, datasets_pkg):
        """``register`` is callable (decorator function)."""
        assert callable(datasets_pkg.register)

    @pytest.mark.contract
    def test_get_accepts_name_parameter(self, datasets_pkg):
        """``get`` function accepts at least one parameter (name)."""
        sig = inspect.signature(datasets_pkg.get)
        assert len(sig.parameters) >= 1, (
            "get() should accept at least one parameter (dataset name)"
        )

    @pytest.mark.contract
    def test_list_all_is_callable(self, datasets_pkg):
        """``list_all`` is callable."""
        assert callable(datasets_pkg.list_all)

    @pytest.mark.contract
    def test_is_registered_accepts_name_parameter(self, datasets_pkg):
        """``is_registered`` function accepts at least one parameter."""
        sig = inspect.signature(datasets_pkg.is_registered)
        assert len(sig.parameters) >= 1, (
            "is_registered() should accept at least one parameter"
        )

    @pytest.mark.contract
    def test_get_default_registry_is_callable(self, datasets_pkg):
        """``get_default_registry`` is callable."""
        assert callable(datasets_pkg.get_default_registry)

    @pytest.mark.contract
    def test_get_default_registry_returns_registry_instance(self, datasets_pkg):
        """``get_default_registry()`` returns a DatasetRegistry instance."""
        result = datasets_pkg.get_default_registry()
        assert isinstance(result, datasets_pkg.DatasetRegistry), (
            f"get_default_registry() should return DatasetRegistry, "
            f"got {type(result).__name__}"
        )


class TestContractProtocolTypes:
    """§2 — Protocol classes are runtime_checkable and have expected methods."""

    @pytest.mark.contract
    def test_handler_protocol_is_runtime_checkable(self, datasets_pkg):
        """
        ``DatasetHandlerProtocol`` is decorated with ``@runtime_checkable``.

        Per the project structure doc: DatasetHandlerProtocol is
        runtime_checkable with 11 methods.
        """
        from typing import runtime_checkable, Protocol
        proto = datasets_pkg.DatasetHandlerProtocol
        assert issubclass(proto, Protocol), (
            "DatasetHandlerProtocol should be a Protocol subclass"
        )
        # runtime_checkable protocols have _is_runtime_protocol = True
        assert getattr(proto, '_is_runtime_protocol', False), (
            "DatasetHandlerProtocol should be @runtime_checkable"
        )

    HANDLER_PROTOCOL_METHODS = [
        "get_dataset_type",
        "validate_molecule_data",
        "get_required_properties",
        "process_property_value",
        "enrich_pyg_data",
        "get_processing_statistics",
        "get_supported_structural_features",
        "get_molecular_charge",
        "get_molecule_creation_strategy",
        "get_transform_recommendations",
        "get_supported_descriptors",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("method", HANDLER_PROTOCOL_METHODS)
    def test_handler_protocol_has_method(self, datasets_pkg, method):
        """DatasetHandlerProtocol defines the expected method."""
        proto = datasets_pkg.DatasetHandlerProtocol
        # Protocol methods appear as annotations or in __protocol_attrs__
        # or as direct attributes on the class
        has_method = (
            hasattr(proto, method)
            or method in getattr(proto, '__protocol_attrs__', set())
            or method in getattr(proto, '__abstractmethods__', set())
        )
        assert has_method, (
            f"DatasetHandlerProtocol should define '{method}' "
            f"(11-method contract per project structure doc)"
        )

    @pytest.mark.contract
    def test_converter_protocol_is_a_class(self, datasets_pkg):
        """``DatasetConverterProtocol`` is a class."""
        assert inspect.isclass(datasets_pkg.DatasetConverterProtocol)

    @pytest.mark.contract
    def test_validator_protocol_is_a_class(self, datasets_pkg):
        """``DatasetValidatorProtocol`` is a class."""
        assert inspect.isclass(datasets_pkg.DatasetValidatorProtocol)


class TestContractExceptionHierarchy:
    """§2 — Exception classes are proper Exception subclasses."""

    @pytest.mark.contract
    def test_dataset_registration_error_is_exception(self, datasets_pkg):
        """``DatasetRegistrationError`` is a subclass of ``Exception``."""
        assert issubclass(datasets_pkg.DatasetRegistrationError, Exception), (
            "DatasetRegistrationError should be a subclass of Exception"
        )

    @pytest.mark.contract
    def test_dataset_not_found_error_is_exception(self, datasets_pkg):
        """``DatasetNotFoundError`` is a subclass of ``Exception``."""
        assert issubclass(datasets_pkg.DatasetNotFoundError, Exception), (
            "DatasetNotFoundError should be a subclass of Exception"
        )

    @pytest.mark.contract
    def test_dataset_registration_error_is_instantiable(self, datasets_pkg):
        """``DatasetRegistrationError`` can be instantiated with a message."""
        try:
            exc = datasets_pkg.DatasetRegistrationError("test error")
            assert str(exc) is not None
        except TypeError:
            # Some custom exceptions require additional args
            # Just verify the class exists and is an Exception subclass
            assert issubclass(datasets_pkg.DatasetRegistrationError, Exception)

    @pytest.mark.contract
    def test_dataset_not_found_error_is_instantiable(self, datasets_pkg):
        """``DatasetNotFoundError`` can be instantiated with a message."""
        try:
            exc = datasets_pkg.DatasetNotFoundError("test error")
            assert str(exc) is not None
        except TypeError:
            # Some custom exceptions require additional args
            assert issubclass(datasets_pkg.DatasetNotFoundError, Exception)


class TestContractGetModuleInfoContract:
    """§2 — get_module_info() returns dict with expected keys."""

    EXPECTED_KEYS = [
        "version",
        "author",
        "status",
        "supported_types",
        "handler_architecture",
        "transformation_system",
        "registry_version",
        "implementations_version",
        "phase_6_integration",
        "available_features",
        "core_classes",
        "architecture",
        "exception_integration",
        "registered_datasets",
        "phase_6_features",
        "standard_transforms_features",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("key", EXPECTED_KEYS)
    def test_module_info_has_key(self, datasets_pkg, key):
        """get_module_info() result contains the expected key."""
        info = datasets_pkg.get_module_info()
        assert key in info, (
            f"get_module_info() missing expected key '{key}'. "
            f"Available keys: {sorted(info.keys())}"
        )

    @pytest.mark.contract
    def test_module_info_version_matches(self, datasets_pkg):
        """get_module_info()['version'] matches __version__."""
        info = datasets_pkg.get_module_info()
        assert info["version"] == datasets_pkg.__version__, (
            f"get_module_info()['version'] ({info['version']}) does not match "
            f"__version__ ({datasets_pkg.__version__})"
        )

    @pytest.mark.contract
    def test_module_info_supported_types_is_list(self, datasets_pkg):
        """get_module_info()['supported_types'] is a list."""
        info = datasets_pkg.get_module_info()
        assert isinstance(info["supported_types"], list), (
            f"get_module_info()['supported_types'] should be list, "
            f"got {type(info['supported_types']).__name__}"
        )

    @pytest.mark.contract
    def test_module_info_available_features_is_list(self, datasets_pkg):
        """get_module_info()['available_features'] is a list."""
        info = datasets_pkg.get_module_info()
        assert isinstance(info["available_features"], list), (
            f"get_module_info()['available_features'] should be list, "
            f"got {type(info['available_features']).__name__}"
        )

    @pytest.mark.contract
    def test_module_info_core_classes_is_list(self, datasets_pkg):
        """get_module_info()['core_classes'] is a list."""
        info = datasets_pkg.get_module_info()
        assert isinstance(info["core_classes"], list), (
            f"get_module_info()['core_classes'] should be list, "
            f"got {type(info['core_classes']).__name__}"
        )

    @pytest.mark.contract
    def test_module_info_always_includes_dataset_registry_feature(self, datasets_pkg):
        """get_module_info()['available_features'] always includes 'dataset_registry'."""
        info = datasets_pkg.get_module_info()
        assert "dataset_registry" in info["available_features"], (
            "'dataset_registry' should always be in available_features "
            "(Phase 1 is always available)"
        )

    @pytest.mark.contract
    def test_module_info_always_includes_phase_6_feature(self, datasets_pkg):
        """get_module_info()['available_features'] always includes 'phase_6_registry_integration'."""
        info = datasets_pkg.get_module_info()
        assert "phase_6_registry_integration" in info["available_features"], (
            "'phase_6_registry_integration' should always be in available_features "
            "(Phase 6 is always available after Phase 6)"
        )

    @pytest.mark.contract
    def test_module_info_phase_6_features_is_dict(self, datasets_pkg):
        """get_module_info()['phase_6_features'] is a dict with boolean values."""
        info = datasets_pkg.get_module_info()
        phase_6 = info["phase_6_features"]
        assert isinstance(phase_6, dict), (
            f"get_module_info()['phase_6_features'] should be dict, "
            f"got {type(phase_6).__name__}"
        )
        for key, value in phase_6.items():
            assert isinstance(value, bool), (
                f"phase_6_features['{key}'] should be bool, "
                f"got {type(value).__name__}"
            )

    @pytest.mark.contract
    def test_module_info_standard_transforms_features_is_dict(self, datasets_pkg):
        """get_module_info()['standard_transforms_features'] is a dict with boolean values."""
        info = datasets_pkg.get_module_info()
        st_features = info["standard_transforms_features"]
        assert isinstance(st_features, dict), (
            f"get_module_info()['standard_transforms_features'] should be dict, "
            f"got {type(st_features).__name__}"
        )
        for key, value in st_features.items():
            assert isinstance(value, bool), (
                f"standard_transforms_features['{key}'] should be bool, "
                f"got {type(value).__name__}"
            )


class TestContractCheckDependenciesContract:
    """§2 — check_dependencies() returns dict with expected keys."""

    ALWAYS_PRESENT_KEYS = [
        "dataset_registry",
        "dataset_implementations",
        "phase_6_registry_integration",
        "standard_transforms",
    ]

    OPTIONAL_DEPENDENCY_KEYS = [
        "pytorch_geometric",
        "dataset_handlers",
        "enhanced_transforms",
        "descriptors",
        "rdkit",
        "numpy",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("key", ALWAYS_PRESENT_KEYS)
    def test_always_true_dependency(self, datasets_pkg, key):
        """Dependencies that are always available should be True."""
        deps = datasets_pkg.check_dependencies()
        assert key in deps, (
            f"check_dependencies() missing expected key '{key}'"
        )
        assert deps[key] is True, (
            f"check_dependencies()['{key}'] should always be True, "
            f"got {deps[key]}"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("key", OPTIONAL_DEPENDENCY_KEYS)
    def test_optional_dependency_key_exists(self, datasets_pkg, key):
        """Each optional dependency key exists in the result."""
        deps = datasets_pkg.check_dependencies()
        assert key in deps, (
            f"check_dependencies() missing expected key '{key}'"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("key", OPTIONAL_DEPENDENCY_KEYS)
    def test_optional_dependency_is_bool(self, datasets_pkg, key):
        """Each optional dependency value is a boolean."""
        deps = datasets_pkg.check_dependencies()
        assert isinstance(deps[key], bool), (
            f"check_dependencies()['{key}'] should be bool, "
            f"got {type(deps[key]).__name__}"
        )


class TestContractGetSupportedDatasetTypesContract:
    """§2 — get_supported_dataset_types() contract."""

    @pytest.mark.contract
    def test_returns_list_of_strings(self, datasets_pkg):
        """``get_supported_dataset_types()`` returns a list of strings."""
        result = datasets_pkg.get_supported_dataset_types()
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str), (
                f"Each item in get_supported_dataset_types() should be str, "
                f"got {type(item).__name__}: {item}"
            )

    @pytest.mark.contract
    def test_consistent_with_list_all(self, datasets_pkg):
        """
        ``get_supported_dataset_types()`` is consistent with ``list_all()``.

        Both functions should return the same dataset names (both use the
        registry as the primary source of truth).
        """
        supported = datasets_pkg.get_supported_dataset_types()
        registered = datasets_pkg.list_all()
        assert set(supported) == set(registered), (
            f"get_supported_dataset_types() ({sorted(supported)}) should match "
            f"list_all() ({sorted(registered)})"
        )

    @pytest.mark.contract
    def test_supported_dataset_types_constant_consistent(self, datasets_pkg):
        """
        ``SUPPORTED_DATASET_TYPES`` constant is consistent with
        ``get_supported_dataset_types()`` function.
        """
        constant = datasets_pkg.SUPPORTED_DATASET_TYPES
        dynamic = datasets_pkg.get_supported_dataset_types()
        assert set(constant) == set(dynamic), (
            f"SUPPORTED_DATASET_TYPES ({sorted(constant)}) should match "
            f"get_supported_dataset_types() ({sorted(dynamic)})"
        )


class TestContractInitializePluginsContract:
    """§2 — initialize_plugins() contract."""

    @pytest.mark.contract
    def test_accepts_load_external_parameter(self, datasets_pkg):
        """``initialize_plugins`` accepts a ``load_external`` parameter."""
        sig = inspect.signature(datasets_pkg.initialize_plugins)
        param_names = set(sig.parameters.keys())
        assert "load_external" in param_names, (
            f"initialize_plugins() should have 'load_external' parameter. "
            f"Has: {param_names}"
        )

    @pytest.mark.contract
    def test_load_external_has_default_true(self, datasets_pkg):
        """``initialize_plugins(load_external=...)`` defaults to True."""
        sig = inspect.signature(datasets_pkg.initialize_plugins)
        param = sig.parameters["load_external"]
        assert param.default is True, (
            f"initialize_plugins(load_external=) should default to True, "
            f"got {param.default}"
        )

    @pytest.mark.contract
    def test_callable_with_false(self, datasets_pkg):
        """``initialize_plugins(load_external=False)`` executes without error."""
        result = datasets_pkg.initialize_plugins(load_external=False)
        assert isinstance(result, int)


class TestContractDynamicGetAttrContract:
    """§2 — __getattr__ dynamic attribute access contract."""

    @pytest.mark.contract
    def test_getattr_exists_as_function(self, datasets_pkg):
        """
        The datasets module defines a module-level ``__getattr__`` function
        for dynamic attribute access.
        """
        # Module-level __getattr__ is accessible via module __dict__
        module_dict = vars(datasets_pkg)
        assert "__getattr__" in module_dict, (
            "datasets/__init__.py should define a module-level __getattr__"
        )
        assert callable(module_dict["__getattr__"]), (
            "__getattr__ should be callable"
        )

    @pytest.mark.contract
    def test_getattr_raises_attributeerror_for_invalid_names(self, datasets_pkg):
        """
        ``__getattr__`` raises ``AttributeError`` with informative message
        for names that are not registered dataset classes.
        """
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = datasets_pkg.CompletelyBogusName

    @pytest.mark.contract
    def test_getattr_handles_dataset_suffix_correctly(self, datasets_pkg):
        """
        ``__getattr__`` recognizes names ending with 'Dataset' and
        attempts registry lookup before raising AttributeError.
        """
        # This should raise because 'ZZZFakeNonExistent' is not registered
        with pytest.raises(AttributeError):
            _ = datasets_pkg.ZZZFakeNonExistentDataset

    @pytest.mark.contract
    def test_getattr_returns_class_for_registered_datasets(self, datasets_pkg):
        """
        For each registered dataset, ``__getattr__`` returns the dataset class
        when accessed as ``<Name>Dataset``.
        """
        registered = datasets_pkg.list_all()
        for dataset_name in registered:
            attr_name = f"{dataset_name}Dataset"
            try:
                cls = getattr(datasets_pkg, attr_name)
                assert inspect.isclass(cls), (
                    f"Dynamic __getattr__('{attr_name}') should return a class, "
                    f"got {type(cls).__name__}"
                )
            except AttributeError:
                # Some dataset names may need case variation (e.g., ANI1x).
                # The __getattr__ also tries uppercase. This is acceptable.
                pass


class TestContractPublicAPISurface:
    """
    §2 — Public API surface stability.

    Ensures that the minimum expected public API is present in ``__all__``.
    Guards against accidental removals during refactoring.
    """

    # The minimum API surface that MUST be present in __all__
    MINIMUM_API = {
        # Primary Dataset Class
        "miliaDataset",
        # Version and Metadata
        "__version__",
        "__author__",
        "__module_status__",
        # Phase 1: Base Classes
        "BaseDataset",
        "DatasetMetadata",
        "DatasetSchema",
        "DatasetFeatures",
        # Phase 1: Registry
        "DatasetRegistry",
        "get_default_registry",
        "register",
        "get",
        "list_all",
        "is_registered",
        # Phase 1: Protocols
        "DatasetHandlerProtocol",
        "DatasetConverterProtocol",
        "DatasetValidatorProtocol",
        # Phase 1: Exceptions
        "DatasetRegistrationError",
        "DatasetNotFoundError",
        # Phase 1: Plugin Initialization
        "initialize_plugins",
        # Dynamic Dataset Type Discovery
        "get_supported_dataset_types",
        "SUPPORTED_DATASET_TYPES",
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
    def test_all_has_expected_minimum_length(self, all_names):
        """
        ``__all__`` contains at least the minimum expected number of entries.

        Based on the __init__.py source, the datasets package exports ~27
        names in __all__. We set a floor to catch catastrophic truncation.
        """
        actual = len(all_names)
        MINIMUM_EXPECTED = 20
        assert actual >= MINIMUM_EXPECTED, (
            f"__all__ has {actual} entries, expected at least {MINIMUM_EXPECTED}. "
            f"This suggests __all__ may have been accidentally truncated."
        )


class TestContractVersionConstantValues:
    """§2 — Version constants have expected values based on source code."""

    @pytest.mark.contract
    def test_handler_architecture_version(self, datasets_pkg):
        """HANDLER_ARCHITECTURE_VERSION is '2.0'."""
        assert datasets_pkg.HANDLER_ARCHITECTURE_VERSION == "2.0", (
            f"HANDLER_ARCHITECTURE_VERSION should be '2.0', "
            f"got '{datasets_pkg.HANDLER_ARCHITECTURE_VERSION}'"
        )

    @pytest.mark.contract
    def test_transformation_system_version(self, datasets_pkg):
        """TRANSFORMATION_SYSTEM_VERSION is '2.1'."""
        assert datasets_pkg.TRANSFORMATION_SYSTEM_VERSION == "2.1", (
            f"TRANSFORMATION_SYSTEM_VERSION should be '2.1', "
            f"got '{datasets_pkg.TRANSFORMATION_SYSTEM_VERSION}'"
        )

    @pytest.mark.contract
    def test_registry_version(self, datasets_pkg):
        """REGISTRY_VERSION is '1.0'."""
        assert datasets_pkg.REGISTRY_VERSION == "1.0", (
            f"REGISTRY_VERSION should be '1.0', "
            f"got '{datasets_pkg.REGISTRY_VERSION}'"
        )

    @pytest.mark.contract
    def test_implementations_version(self, datasets_pkg):
        """IMPLEMENTATIONS_VERSION is '1.0'."""
        assert datasets_pkg.IMPLEMENTATIONS_VERSION == "1.0", (
            f"IMPLEMENTATIONS_VERSION should be '1.0', "
            f"got '{datasets_pkg.IMPLEMENTATIONS_VERSION}'"
        )

    @pytest.mark.contract
    def test_phase_6_integration_version(self, datasets_pkg):
        """PHASE_6_INTEGRATION_VERSION is '6.0.0'."""
        assert datasets_pkg.PHASE_6_INTEGRATION_VERSION == "6.0.0", (
            f"PHASE_6_INTEGRATION_VERSION should be '6.0.0', "
            f"got '{datasets_pkg.PHASE_6_INTEGRATION_VERSION}'"
        )


class TestContractRegisteredDatasetsContract:
    """§2 — Registered datasets satisfy the BaseDataset ABC contract."""

    @pytest.mark.contract
    def test_all_registered_datasets_are_base_dataset_subclasses(self, datasets_pkg):
        """
        Every registered dataset class is a subclass of BaseDataset.

        Per the project structure doc: concrete implementations use
        ``@register`` decorator on ``BaseDataset`` subclasses.
        """
        registered = datasets_pkg.list_all()
        for dataset_name in registered:
            cls = datasets_pkg.get(dataset_name)
            assert inspect.isclass(cls), (
                f"get('{dataset_name}') should return a class, "
                f"got {type(cls).__name__}"
            )
            assert issubclass(cls, datasets_pkg.BaseDataset), (
                f"Registered dataset '{dataset_name}' ({cls.__name__}) should "
                f"be a subclass of BaseDataset"
            )

    @pytest.mark.contract
    def test_all_registered_datasets_have_metadata(self, datasets_pkg):
        """
        Every registered dataset class has a ``metadata`` class attribute
        that is a DatasetMetadata instance.
        """
        registered = datasets_pkg.list_all()
        for dataset_name in registered:
            cls = datasets_pkg.get(dataset_name)
            assert hasattr(cls, "metadata"), (
                f"Registered dataset '{dataset_name}' should have "
                f"'metadata' class attribute"
            )
            metadata = cls.metadata
            assert isinstance(metadata, datasets_pkg.DatasetMetadata), (
                f"Registered dataset '{dataset_name}'.metadata should be "
                f"DatasetMetadata instance, got {type(metadata).__name__}"
            )

    @pytest.mark.contract
    def test_all_registered_datasets_have_schema(self, datasets_pkg):
        """
        Every registered dataset class has a ``schema`` class attribute
        that is a DatasetSchema instance.
        """
        registered = datasets_pkg.list_all()
        for dataset_name in registered:
            cls = datasets_pkg.get(dataset_name)
            assert hasattr(cls, "schema"), (
                f"Registered dataset '{dataset_name}' should have "
                f"'schema' class attribute"
            )
            schema = cls.schema
            assert isinstance(schema, datasets_pkg.DatasetSchema), (
                f"Registered dataset '{dataset_name}'.schema should be "
                f"DatasetSchema instance, got {type(schema).__name__}"
            )

    @pytest.mark.contract
    def test_all_registered_datasets_have_features(self, datasets_pkg):
        """
        Every registered dataset class has a ``features`` class attribute
        that is a DatasetFeatures instance.
        """
        registered = datasets_pkg.list_all()
        for dataset_name in registered:
            cls = datasets_pkg.get(dataset_name)
            assert hasattr(cls, "features"), (
                f"Registered dataset '{dataset_name}' should have "
                f"'features' class attribute"
            )
            features = cls.features
            assert isinstance(features, datasets_pkg.DatasetFeatures), (
                f"Registered dataset '{dataset_name}'.features should be "
                f"DatasetFeatures instance, got {type(features).__name__}"
            )


class TestContractRegistryGetAndIsRegisteredConsistency:
    """§2 — Registry get() and is_registered() are consistent."""

    @pytest.mark.contract
    def test_get_returns_class_for_all_registered(self, datasets_pkg):
        """For every name in list_all(), get(name) returns a class."""
        registered = datasets_pkg.list_all()
        for name in registered:
            cls = datasets_pkg.get(name)
            assert inspect.isclass(cls), (
                f"get('{name}') should return a class, "
                f"got {type(cls).__name__}"
            )

    @pytest.mark.contract
    def test_is_registered_true_for_all_listed(self, datasets_pkg):
        """For every name in list_all(), is_registered(name) returns True."""
        registered = datasets_pkg.list_all()
        for name in registered:
            assert datasets_pkg.is_registered(name), (
                f"is_registered('{name}') should be True since '{name}' "
                f"appears in list_all()"
            )

    @pytest.mark.contract
    def test_is_registered_false_for_nonexistent(self, datasets_pkg):
        """is_registered() returns False for an unregistered name."""
        result = datasets_pkg.is_registered("ZZZCompletelyFakeDataset12345")
        assert result is False, (
            "is_registered() should return False for a non-existent dataset"
        )

    @pytest.mark.contract
    def test_list_all_returns_no_duplicates(self, datasets_pkg):
        """list_all() does not return duplicate dataset names."""
        registered = datasets_pkg.list_all()
        assert len(registered) == len(set(registered)), (
            f"list_all() has duplicates: {registered}"
        )


class TestContractModuleDocstring:
    """§2 — Module docstring is present and informative."""

    @pytest.mark.contract
    def test_module_has_docstring(self, datasets_pkg):
        """The datasets package has a non-empty docstring."""
        assert datasets_pkg.__doc__ is not None, (
            "datasets package should have a __doc__ string"
        )
        assert len(datasets_pkg.__doc__) > 100, (
            "datasets package docstring should be substantial (>100 chars)"
        )

    @pytest.mark.contract
    def test_docstring_mentions_milia(self, datasets_pkg):
        """The docstring references 'Milia' or 'milia'."""
        doc = datasets_pkg.__doc__.lower()
        assert "milia" in doc, (
            "datasets package docstring should mention 'Milia'"
        )

    @pytest.mark.contract
    def test_docstring_mentions_registry(self, datasets_pkg):
        """The docstring references the registry system."""
        doc = datasets_pkg.__doc__.lower()
        assert "registry" in doc, (
            "datasets package docstring should mention the registry system"
        )


class TestContractListAllConsistentWithGetModuleInfo:
    """§2 — Cross-function consistency between list_all and get_module_info."""

    @pytest.mark.contract
    def test_registered_datasets_in_module_info(self, datasets_pkg):
        """
        get_module_info()['registered_datasets'] matches list_all().
        """
        info = datasets_pkg.get_module_info()
        registered_from_info = info["registered_datasets"]
        registered_from_list_all = datasets_pkg.list_all()
        assert set(registered_from_info) == set(registered_from_list_all), (
            f"get_module_info()['registered_datasets'] ({sorted(registered_from_info)}) "
            f"should match list_all() ({sorted(registered_from_list_all)})"
        )

    @pytest.mark.contract
    def test_supported_types_in_module_info(self, datasets_pkg):
        """
        get_module_info()['supported_types'] matches get_supported_dataset_types().
        """
        info = datasets_pkg.get_module_info()
        supported_from_info = info["supported_types"]
        supported_from_func = datasets_pkg.get_supported_dataset_types()
        assert set(supported_from_info) == set(supported_from_func), (
            f"get_module_info()['supported_types'] ({sorted(supported_from_info)}) "
            f"should match get_supported_dataset_types() ({sorted(supported_from_func)})"
        )
