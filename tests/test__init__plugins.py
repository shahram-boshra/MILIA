#!/usr/bin/env python3
# tests/test__init__plugins.py

"""
Test Suite: milia_pipeline/plugins/__init__.py — Smoke Tests & Contract Tests
=============================================================================

Production-ready test suite for the MILIA Pipeline plugins sub-package
``milia_pipeline/plugins/__init__.py``.

Covers:
    Section 1 — Smoke Tests (MILIA_Test_Recommendations.md §1.2 scope):
        - The ``milia_pipeline.plugins`` subpackage imports without ImportError
        - All public API names from ``__all__`` are accessible
        - Module-level metadata attributes (__version__, __author__) exist
        - Module initialization (logging, threading lock, discovery state) runs safely
        - Re-import (``importlib.reload``) is idempotent and non-crashing
        - Internal state variables (_lock, _discovered_subplugins, etc.) are present
        - Discovery functions are accessible and callable
        - Plugin management API functions are accessible and callable
        - Directory path functions are accessible and callable
        - System status function is accessible and callable

    Section 2 — Contract Tests (MILIA_Test_Recommendations.md §2 scope):
        - ``__all__`` completeness: every name in ``__all__`` is resolvable
        - ``__all__`` consistency: every public import is listed in ``__all__``
        - ``__all__`` has no duplicates
        - ``__all__`` entries are all strings
        - Public API surface stability (minimum expected names present)
        - ``__version__`` follows semver pattern
        - ``list_subplugins()`` returns a list of strings
        - ``get_plugins_directory()`` returns a Path
        - ``get_descriptor_plugins_directory()`` returns a Path
        - ``get_system_status()`` returns a dict with documented keys
        - ``get_subplugin_info()`` returns None for unknown plugins
        - ``get_all_plugin_info()`` returns a dict
        - ``enable_subplugin()`` returns False for unknown plugins
        - ``disable_subplugin()`` returns False for unknown plugins
        - ``get_subplugin_transforms()`` returns a list for unknown plugins
        - ``get_transform()`` returns None for unknown plugins
        - ``__getattr__`` raises AttributeError for unknown attributes
        - ``_discover_subplugins()`` returns a dict (idempotent, cached)
        - ``_get_subplugin_module()`` returns None for unknown names
        - Thread-safety primitives (_lock) are threading.Lock instances
        - Discovery functions are idempotent (repeated calls return same result)
        - Directory path functions return paths within the package directory
        - Function signatures match documented parameter expectations

Design:
    - Zero ``sys.modules`` pollution: all mocking via ``@patch`` decorators
      or context-manager ``patch`` calls scoped to individual tests.
    - Deterministic: no filesystem, network, or GPU access required.
    - Isolated: each test is independent; execution order is irrelevant.
    - Fast: expected < 5 s total wall-clock on CI.

Launch:
    From project root (/app/milia):
        pytest tests/test__init__plugins.py -v --tb=short

Markers:
    smoke     — Quick health-check tests (§1)
    contract  — Interface/contract validation tests (§2)
"""

import importlib
import inspect
import logging
import sys
import threading
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure the project root is importable
# ---------------------------------------------------------------------------
# When launched via ``pytest tests/test__init__plugins.py`` from the
# project root (/app/milia), ``milia_pipeline`` must be on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # …/tests -> …/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture(scope="module")
def plugins_pkg():
    """
    Import and return the ``milia_pipeline.plugins`` package once per module.

    This fixture validates the fundamental smoke invariant: the plugins
    subpackage is importable.  If this fails, every downstream test is moot.
    """
    try:
        import milia_pipeline.plugins as plugins

        return plugins
    except ImportError as exc:
        pytest.fail(
            f"milia_pipeline.plugins could not be imported — smoke test "
            f"precondition violated: {exc}"
        )


@pytest.fixture(scope="module")
def all_names(plugins_pkg):
    """Return the ``__all__`` list from the plugins package."""
    assert hasattr(plugins_pkg, "__all__"), (
        "milia_pipeline.plugins.__all__ is missing — contract violation"
    )
    return list(plugins_pkg.__all__)


# ===================================================================
# SECTION 1 — SMOKE TESTS
# ===================================================================


class TestSmokePluginsPackageImport:
    """§1.2 — Verify the plugins subpackage imports without errors."""

    @pytest.mark.smoke
    def test_import_plugins_package_succeeds(self, plugins_pkg):
        """The plugins package imports without raising any exception."""
        assert plugins_pkg is not None

    @pytest.mark.smoke
    def test_plugins_package_is_a_module(self, plugins_pkg):
        """The imported object is a proper Python module."""
        assert isinstance(plugins_pkg, types.ModuleType)

    @pytest.mark.smoke
    def test_plugins_package_has_file_attribute(self, plugins_pkg):
        """The package exposes a ``__file__`` attribute (not a namespace pkg)."""
        assert hasattr(plugins_pkg, "__file__")

    @pytest.mark.smoke
    def test_plugins_package_name(self, plugins_pkg):
        """The package ``__name__`` is ``milia_pipeline.plugins``."""
        assert plugins_pkg.__name__ == "milia_pipeline.plugins"


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
    def test_metadata_attribute_exists(self, plugins_pkg, attr):
        """Each metadata dunder is defined on the plugins package."""
        assert hasattr(plugins_pkg, attr), f"Missing attribute: {attr}"

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "attr",
        [
            "__version__",
            "__author__",
        ],
    )
    def test_metadata_attribute_is_string(self, plugins_pkg, attr):
        """Each metadata dunder is a non-empty string."""
        value = getattr(plugins_pkg, attr)
        assert isinstance(value, str), f"{attr} should be str, got {type(value)}"
        assert len(value) > 0, f"{attr} should be non-empty"

    @pytest.mark.smoke
    def test_version_is_semver_like(self, plugins_pkg):
        """``__version__`` follows a MAJOR.MINOR.PATCH pattern."""
        version = plugins_pkg.__version__
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


class TestSmokeInternalStateVariables:
    """§1.2 — Internal state variables exist and have expected initial types."""

    @pytest.mark.smoke
    def test_lock_exists(self, plugins_pkg):
        """The module-level ``_lock`` is present."""
        assert hasattr(plugins_pkg, "_lock")

    @pytest.mark.smoke
    def test_discovered_subplugins_exists(self, plugins_pkg):
        """The module-level ``_discovered_subplugins`` is present."""
        assert hasattr(plugins_pkg, "_discovered_subplugins")

    @pytest.mark.smoke
    def test_discovery_attempted_exists(self, plugins_pkg):
        """The module-level ``_discovery_attempted`` is present."""
        assert hasattr(plugins_pkg, "_discovery_attempted")

    @pytest.mark.smoke
    def test_discovery_attempted_is_bool(self, plugins_pkg):
        """``_discovery_attempted`` is a boolean."""
        value = plugins_pkg._discovery_attempted
        assert isinstance(value, bool), (
            f"_discovery_attempted should be bool, got {type(value).__name__}"
        )


class TestSmokeDiscoveryFunctions:
    """§1.2 — Discovery functions are accessible and callable."""

    DISCOVERY_FUNCTIONS = [
        "_discover_subplugins",
        "_get_subplugin_module",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", DISCOVERY_FUNCTIONS)
    def test_discovery_function_exists(self, plugins_pkg, name):
        """Each discovery function is present and non-None."""
        obj = getattr(plugins_pkg, name, None)
        assert obj is not None, f"Discovery function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", DISCOVERY_FUNCTIONS)
    def test_discovery_function_is_callable(self, plugins_pkg, name):
        """Each discovery function is callable."""
        obj = getattr(plugins_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokePluginManagementAPIFunctions:
    """§1.2 — Unified plugin management API functions are accessible."""

    API_FUNCTIONS = [
        "list_subplugins",
        "get_subplugin_info",
        "get_all_plugin_info",
        "enable_subplugin",
        "disable_subplugin",
        "get_subplugin_transforms",
        "get_transform",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", API_FUNCTIONS)
    def test_api_function_exists(self, plugins_pkg, name):
        """Each plugin management API function is present and non-None."""
        obj = getattr(plugins_pkg, name, None)
        assert obj is not None, f"API function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", API_FUNCTIONS)
    def test_api_function_is_callable(self, plugins_pkg, name):
        """Each plugin management API function is callable."""
        obj = getattr(plugins_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeDirectoryPathFunctions:
    """§1.2 — Directory path functions are accessible and callable."""

    PATH_FUNCTIONS = [
        "get_plugins_directory",
        "get_descriptor_plugins_directory",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", PATH_FUNCTIONS)
    def test_path_function_exists(self, plugins_pkg, name):
        """Each directory path function is present and non-None."""
        obj = getattr(plugins_pkg, name, None)
        assert obj is not None, f"Path function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", PATH_FUNCTIONS)
    def test_path_function_is_callable(self, plugins_pkg, name):
        """Each directory path function is callable."""
        obj = getattr(plugins_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeSystemStatusFunction:
    """§1.2 — System status function is accessible and callable."""

    @pytest.mark.smoke
    def test_get_system_status_exists(self, plugins_pkg):
        """``get_system_status`` is present and non-None."""
        obj = getattr(plugins_pkg, "get_system_status", None)
        assert obj is not None, "get_system_status is None or missing"

    @pytest.mark.smoke
    def test_get_system_status_is_callable(self, plugins_pkg):
        """``get_system_status`` is callable."""
        assert callable(plugins_pkg.get_system_status)


class TestSmokeLoggerConfiguration:
    """§1.2 — Logger is configured with the expected naming convention."""

    @pytest.mark.smoke
    def test_logger_exists(self, plugins_pkg):
        """The module-level ``logger`` is present."""
        assert hasattr(plugins_pkg, "logger")

    @pytest.mark.smoke
    def test_logger_is_logging_logger(self, plugins_pkg):
        """``logger`` is a ``logging.Logger`` instance."""
        assert isinstance(plugins_pkg.logger, logging.Logger), (
            f"logger should be logging.Logger, got {type(plugins_pkg.logger).__name__}"
        )

    @pytest.mark.smoke
    def test_logger_name_follows_convention(self, plugins_pkg):
        """``logger`` name follows the ``milia_Main.PluginSystem`` convention."""
        assert plugins_pkg.logger.name == "milia_Main.PluginSystem", (
            f"Logger name should be 'milia_Main.PluginSystem', got '{plugins_pkg.logger.name}'"
        )


class TestSmokeModuleGetattr:
    """§1.2 — Module-level ``__getattr__`` is defined for lazy loading."""

    @pytest.mark.smoke
    def test_getattr_is_defined(self, plugins_pkg):
        """The module defines a ``__getattr__`` function for lazy imports."""
        # __getattr__ is a module-level function, accessible via the module dict
        module_dict = vars(plugins_pkg)
        assert "__getattr__" in module_dict, "Module-level __getattr__ function is not defined"

    @pytest.mark.smoke
    def test_getattr_is_callable(self, plugins_pkg):
        """The module-level ``__getattr__`` is callable."""
        module_dict = vars(plugins_pkg)
        assert callable(module_dict["__getattr__"])


class TestSmokeModuleReimport:
    """§1.2 — Module-level initialization is safe to re-execute."""

    @pytest.mark.smoke
    def test_reimport_is_idempotent(self, plugins_pkg):
        """
        Re-importing the plugins package (via ``importlib.reload``) does not
        crash.

        Validates that all module-level code (logging, threading lock,
        discovery state initialization) is safe to re-execute.
        """
        reloaded = importlib.reload(plugins_pkg)
        assert reloaded is not None
        assert hasattr(reloaded, "__version__")

    @pytest.mark.smoke
    def test_reimport_preserves_all(self, plugins_pkg):
        """
        Re-importing the plugins package preserves ``__all__``.
        """
        reloaded = importlib.reload(plugins_pkg)
        assert hasattr(reloaded, "__all__")
        assert isinstance(reloaded.__all__, list)
        assert len(reloaded.__all__) > 0

    @pytest.mark.smoke
    def test_reimport_preserves_version(self, plugins_pkg):
        """
        Re-importing the plugins package preserves ``__version__``.
        """
        original_version = plugins_pkg.__version__
        reloaded = importlib.reload(plugins_pkg)
        assert reloaded.__version__ == original_version


# ===================================================================
# SECTION 2 — CONTRACT TESTS
# ===================================================================


class TestContractAllCompleteness:
    """§2 — Every name in ``__all__`` is resolvable on the plugins package."""

    @pytest.mark.contract
    def test_all_is_a_list(self, plugins_pkg):
        """``__all__`` is a list (not a tuple or set)."""
        assert isinstance(plugins_pkg.__all__, list)

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
    def test_every_all_entry_is_resolvable(self, plugins_pkg, all_names):
        """
        Generic sweep: every single entry in ``__all__`` must be resolvable,
        regardless of whether it is parameterized individually.
        """
        unresolvable = [name for name in all_names if not hasattr(plugins_pkg, name)]
        assert not unresolvable, (
            f"Names in __all__ that are not defined on the module: {unresolvable}"
        )

    @pytest.mark.contract
    def test_all_entries_are_strings(self, all_names):
        """Every entry in ``__all__`` is a string."""
        non_strings = [(i, name) for i, name in enumerate(all_names) if not isinstance(name, str)]
        assert not non_strings, f"Non-string entries in __all__: {non_strings}"


class TestContractAllConsistency:
    """§2 — Every public import in the plugins module is listed in ``__all__``."""

    # Names that are intentionally public-ish but NOT in __all__
    KNOWN_UNLISTED = {
        # Metadata dunders (not typically in __all__)
        "__version__",
        "__author__",
        # Internal state (prefixed with underscore, not in __all__)
        "_lock",
        "_discovered_subplugins",
        "_discovery_attempted",
        # Internal discovery functions (prefixed with underscore)
        "_discover_subplugins",
        "_get_subplugin_module",
        # Logger
        "logger",
        # Standard library / typing imports used internally (not public API)
        "Path",
        "Any",
        "Dict",
        "List",
        "Optional",
        "Type",
    }

    @pytest.mark.contract
    def test_public_imports_are_in_all(self, plugins_pkg, all_names):
        """
        Every non-dunder, non-private attribute that was imported (not
        defined locally) should be in ``__all__`` — unless it is in
        ``KNOWN_UNLISTED``.

        This catches accidental omissions when new imports are added to
        the plugins ``__init__.py`` without updating ``__all__``.
        """
        all_set = set(all_names)
        module_dict = vars(plugins_pkg)
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
            f"Public names imported in plugins/__init__.py but not in __all__: "
            f"{sorted(missing_from_all)}"
        )


class TestContractPublicAPISurface:
    """
    §2 — Public API surface stability.

    Ensures that the minimum expected public API is present in ``__all__``.
    Guards against accidental removals during refactoring.
    """

    # The minimum API surface that MUST be present in __all__
    MINIMUM_API = {
        # Metadata
        "__version__",
        "__author__",
        # Discovery
        "list_subplugins",
        "get_subplugin_info",
        "get_all_plugin_info",
        # Plugin lifecycle
        "enable_subplugin",
        "disable_subplugin",
        # Transform access
        "get_subplugin_transforms",
        "get_transform",
        # Directory paths
        "get_plugins_directory",
        "get_descriptor_plugins_directory",
        # System status
        "get_system_status",
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

        Based on the ``__init__.py`` source, the plugins package exports
        13 names. This test guards against catastrophic loss (e.g., accidental
        truncation of ``__all__``) while allowing for organic growth.
        """
        actual = len(all_names)
        # The __init__.py has 13 entries in __all__
        # We set a floor to allow minor changes while catching catastrophic loss.
        MINIMUM_EXPECTED = 10
        assert actual >= MINIMUM_EXPECTED, (
            f"__all__ has {actual} entries, expected at least {MINIMUM_EXPECTED}. "
            f"This suggests __all__ may have been accidentally truncated."
        )


class TestContractListSubplugins:
    """§2 — ``list_subplugins()`` return type and invariants."""

    @pytest.mark.contract
    def test_list_subplugins_returns_list(self, plugins_pkg):
        """``list_subplugins()`` returns a list."""
        result = plugins_pkg.list_subplugins()
        assert isinstance(result, list), (
            f"list_subplugins() should return list, got {type(result).__name__}"
        )

    @pytest.mark.contract
    def test_list_subplugins_returns_strings(self, plugins_pkg):
        """``list_subplugins()`` returns a list of strings."""
        result = plugins_pkg.list_subplugins()
        for item in result:
            assert isinstance(item, str), (
                f"Each item in list_subplugins() should be str, got {type(item).__name__}: {item}"
            )

    @pytest.mark.contract
    def test_list_subplugins_is_sorted(self, plugins_pkg):
        """``list_subplugins()`` returns a sorted list."""
        result = plugins_pkg.list_subplugins()
        assert result == sorted(result), "list_subplugins() should return a sorted list"

    @pytest.mark.contract
    def test_list_subplugins_is_idempotent(self, plugins_pkg):
        """Calling ``list_subplugins()`` twice yields the same result."""
        result1 = plugins_pkg.list_subplugins()
        result2 = plugins_pkg.list_subplugins()
        assert result1 == result2, (
            "list_subplugins() should be idempotent (same result on repeated calls)"
        )


class TestContractGetPluginsDirectory:
    """§2 — ``get_plugins_directory()`` return type and invariants."""

    @pytest.mark.contract
    def test_get_plugins_directory_returns_path(self, plugins_pkg):
        """``get_plugins_directory()`` returns a ``Path`` object."""
        result = plugins_pkg.get_plugins_directory()
        assert isinstance(result, Path), (
            f"get_plugins_directory() should return Path, got {type(result).__name__}"
        )

    @pytest.mark.contract
    def test_get_plugins_directory_is_absolute(self, plugins_pkg):
        """``get_plugins_directory()`` returns an absolute path."""
        result = plugins_pkg.get_plugins_directory()
        assert result.is_absolute(), (
            f"get_plugins_directory() should return an absolute path, got '{result}'"
        )

    @pytest.mark.contract
    def test_get_plugins_directory_ends_with_plugins(self, plugins_pkg):
        """``get_plugins_directory()`` path ends with 'plugins'."""
        result = plugins_pkg.get_plugins_directory()
        assert result.name == "plugins", (
            f"get_plugins_directory() path should end with 'plugins', got '{result.name}'"
        )


class TestContractGetDescriptorPluginsDirectory:
    """§2 — ``get_descriptor_plugins_directory()`` return type and invariants."""

    @pytest.mark.contract
    def test_get_descriptor_plugins_directory_returns_path(self, plugins_pkg):
        """``get_descriptor_plugins_directory()`` returns a ``Path`` object."""
        result = plugins_pkg.get_descriptor_plugins_directory()
        assert isinstance(result, Path), (
            f"get_descriptor_plugins_directory() should return Path, got {type(result).__name__}"
        )

    @pytest.mark.contract
    def test_get_descriptor_plugins_directory_is_absolute(self, plugins_pkg):
        """``get_descriptor_plugins_directory()`` returns an absolute path."""
        result = plugins_pkg.get_descriptor_plugins_directory()
        assert result.is_absolute(), (
            f"get_descriptor_plugins_directory() should return an absolute path, got '{result}'"
        )

    @pytest.mark.contract
    def test_get_descriptor_plugins_directory_ends_with_descriptors(self, plugins_pkg):
        """``get_descriptor_plugins_directory()`` path ends with 'descriptors'."""
        result = plugins_pkg.get_descriptor_plugins_directory()
        assert result.name == "descriptors", (
            f"get_descriptor_plugins_directory() path should end with 'descriptors', "
            f"got '{result.name}'"
        )

    @pytest.mark.contract
    def test_descriptor_dir_is_child_of_plugins_dir(self, plugins_pkg):
        """
        ``get_descriptor_plugins_directory()`` is a child of
        ``get_plugins_directory()``.
        """
        plugins_dir = plugins_pkg.get_plugins_directory()
        descriptors_dir = plugins_pkg.get_descriptor_plugins_directory()
        assert descriptors_dir.parent == plugins_dir, (
            f"Descriptor plugins dir '{descriptors_dir}' should be a child "
            f"of plugins dir '{plugins_dir}'"
        )


class TestContractGetSystemStatus:
    """§2 — ``get_system_status()`` return type and structure."""

    @pytest.mark.contract
    def test_get_system_status_returns_dict(self, plugins_pkg):
        """``get_system_status()`` returns a dict."""
        result = plugins_pkg.get_system_status()
        assert isinstance(result, dict), (
            f"get_system_status() should return dict, got {type(result).__name__}"
        )

    @pytest.mark.contract
    def test_get_system_status_has_version_key(self, plugins_pkg):
        """``get_system_status()`` result includes 'version' key."""
        result = plugins_pkg.get_system_status()
        assert "version" in result, "get_system_status() missing 'version' key"

    @pytest.mark.contract
    def test_get_system_status_has_plugins_directory_key(self, plugins_pkg):
        """``get_system_status()`` result includes 'plugins_directory' key."""
        result = plugins_pkg.get_system_status()
        assert "plugins_directory" in result, "get_system_status() missing 'plugins_directory' key"

    @pytest.mark.contract
    def test_get_system_status_has_descriptor_plugins_directory_key(self, plugins_pkg):
        """``get_system_status()`` result includes 'descriptor_plugins_directory' key."""
        result = plugins_pkg.get_system_status()
        assert "descriptor_plugins_directory" in result, (
            "get_system_status() missing 'descriptor_plugins_directory' key"
        )

    @pytest.mark.contract
    def test_get_system_status_has_discovery_key(self, plugins_pkg):
        """``get_system_status()`` result includes 'discovery' key."""
        result = plugins_pkg.get_system_status()
        assert "discovery" in result, "get_system_status() missing 'discovery' key"

    @pytest.mark.contract
    def test_get_system_status_has_subplugins_key(self, plugins_pkg):
        """``get_system_status()`` result includes 'subplugins' key."""
        result = plugins_pkg.get_system_status()
        assert "subplugins" in result, "get_system_status() missing 'subplugins' key"

    @pytest.mark.contract
    def test_get_system_status_discovery_structure(self, plugins_pkg):
        """
        ``get_system_status()['discovery']`` has the documented keys:
        'attempted', 'subplugins_found', 'subplugin_names'.
        """
        result = plugins_pkg.get_system_status()
        discovery = result["discovery"]
        assert isinstance(discovery, dict), (
            f"discovery should be dict, got {type(discovery).__name__}"
        )
        expected_keys = {"attempted", "subplugins_found", "subplugin_names"}
        missing = expected_keys - set(discovery.keys())
        assert not missing, f"discovery dict missing keys: {sorted(missing)}"

    @pytest.mark.contract
    def test_get_system_status_discovery_types(self, plugins_pkg):
        """Discovery sub-dict values have expected types."""
        result = plugins_pkg.get_system_status()
        discovery = result["discovery"]

        assert isinstance(discovery["attempted"], bool), (
            f"discovery['attempted'] should be bool, got {type(discovery['attempted']).__name__}"
        )
        assert isinstance(discovery["subplugins_found"], int), (
            f"discovery['subplugins_found'] should be int, "
            f"got {type(discovery['subplugins_found']).__name__}"
        )
        assert isinstance(discovery["subplugin_names"], list), (
            f"discovery['subplugin_names'] should be list, "
            f"got {type(discovery['subplugin_names']).__name__}"
        )

    @pytest.mark.contract
    def test_get_system_status_version_matches_module(self, plugins_pkg):
        """``get_system_status()['version']`` matches ``__version__``."""
        result = plugins_pkg.get_system_status()
        assert result["version"] == plugins_pkg.__version__, (
            f"Status version '{result['version']}' does not match "
            f"module __version__ '{plugins_pkg.__version__}'"
        )

    @pytest.mark.contract
    def test_get_system_status_subplugins_is_dict(self, plugins_pkg):
        """``get_system_status()['subplugins']`` is a dict."""
        result = plugins_pkg.get_system_status()
        assert isinstance(result["subplugins"], dict), (
            f"subplugins should be dict, got {type(result['subplugins']).__name__}"
        )


class TestContractGetSubpluginInfo:
    """§2 — ``get_subplugin_info()`` contract for unknown plugins."""

    @pytest.mark.contract
    def test_get_subplugin_info_unknown_returns_none(self, plugins_pkg):
        """``get_subplugin_info()`` returns None for an unknown plugin name."""
        result = plugins_pkg.get_subplugin_info("__nonexistent_plugin_xyz__")
        assert result is None, (
            f"get_subplugin_info() for unknown plugin should return None, got {result}"
        )

    @pytest.mark.contract
    def test_get_subplugin_info_is_function(self, plugins_pkg):
        """``get_subplugin_info`` is a function (not a class)."""
        assert inspect.isfunction(plugins_pkg.get_subplugin_info)


class TestContractGetAllPluginInfo:
    """§2 — ``get_all_plugin_info()`` return type."""

    @pytest.mark.contract
    def test_get_all_plugin_info_returns_dict(self, plugins_pkg):
        """``get_all_plugin_info()`` returns a dict."""
        result = plugins_pkg.get_all_plugin_info()
        assert isinstance(result, dict), (
            f"get_all_plugin_info() should return dict, got {type(result).__name__}"
        )

    @pytest.mark.contract
    def test_get_all_plugin_info_keys_are_strings(self, plugins_pkg):
        """``get_all_plugin_info()`` dict keys are strings."""
        result = plugins_pkg.get_all_plugin_info()
        for key in result:
            assert isinstance(key, str), (
                f"get_all_plugin_info() key should be str, got {type(key).__name__}: {key}"
            )

    @pytest.mark.contract
    def test_get_all_plugin_info_values_are_dicts(self, plugins_pkg):
        """``get_all_plugin_info()`` dict values are dicts."""
        result = plugins_pkg.get_all_plugin_info()
        for name, info in result.items():
            assert isinstance(info, dict), (
                f"get_all_plugin_info()['{name}'] should be dict, got {type(info).__name__}"
            )


class TestContractEnableSubplugin:
    """§2 — ``enable_subplugin()`` contract for unknown plugins."""

    @pytest.mark.contract
    def test_enable_subplugin_unknown_returns_false(self, plugins_pkg):
        """``enable_subplugin()`` returns False for an unknown plugin name."""
        result = plugins_pkg.enable_subplugin("__nonexistent_plugin_xyz__")
        assert result is False, (
            f"enable_subplugin() for unknown plugin should return False, got {result}"
        )

    @pytest.mark.contract
    def test_enable_subplugin_returns_bool(self, plugins_pkg):
        """``enable_subplugin()`` always returns a bool."""
        result = plugins_pkg.enable_subplugin("__nonexistent_plugin_xyz__")
        assert isinstance(result, bool), (
            f"enable_subplugin() should return bool, got {type(result).__name__}"
        )

    @pytest.mark.contract
    def test_enable_subplugin_is_function(self, plugins_pkg):
        """``enable_subplugin`` is a function (not a class)."""
        assert inspect.isfunction(plugins_pkg.enable_subplugin)


class TestContractDisableSubplugin:
    """§2 — ``disable_subplugin()`` contract for unknown plugins."""

    @pytest.mark.contract
    def test_disable_subplugin_unknown_returns_false(self, plugins_pkg):
        """``disable_subplugin()`` returns False for an unknown plugin name."""
        result = plugins_pkg.disable_subplugin("__nonexistent_plugin_xyz__")
        assert result is False, (
            f"disable_subplugin() for unknown plugin should return False, got {result}"
        )

    @pytest.mark.contract
    def test_disable_subplugin_returns_bool(self, plugins_pkg):
        """``disable_subplugin()`` always returns a bool."""
        result = plugins_pkg.disable_subplugin("__nonexistent_plugin_xyz__")
        assert isinstance(result, bool), (
            f"disable_subplugin() should return bool, got {type(result).__name__}"
        )

    @pytest.mark.contract
    def test_disable_subplugin_is_function(self, plugins_pkg):
        """``disable_subplugin`` is a function (not a class)."""
        assert inspect.isfunction(plugins_pkg.disable_subplugin)


class TestContractGetSubpluginTransforms:
    """§2 — ``get_subplugin_transforms()`` contract for unknown plugins."""

    @pytest.mark.contract
    def test_get_subplugin_transforms_unknown_returns_empty_list(self, plugins_pkg):
        """``get_subplugin_transforms()`` returns [] for an unknown plugin."""
        result = plugins_pkg.get_subplugin_transforms("__nonexistent_plugin_xyz__")
        assert result == [], (
            f"get_subplugin_transforms() for unknown plugin should return [], got {result}"
        )

    @pytest.mark.contract
    def test_get_subplugin_transforms_returns_list(self, plugins_pkg):
        """``get_subplugin_transforms()`` returns a list."""
        result = plugins_pkg.get_subplugin_transforms("__nonexistent_plugin_xyz__")
        assert isinstance(result, list), (
            f"get_subplugin_transforms() should return list, got {type(result).__name__}"
        )

    @pytest.mark.contract
    def test_get_subplugin_transforms_is_function(self, plugins_pkg):
        """``get_subplugin_transforms`` is a function (not a class)."""
        assert inspect.isfunction(plugins_pkg.get_subplugin_transforms)


class TestContractGetTransform:
    """§2 — ``get_transform()`` contract for unknown plugins."""

    @pytest.mark.contract
    def test_get_transform_unknown_plugin_returns_none(self, plugins_pkg):
        """``get_transform()`` returns None for an unknown plugin name."""
        result = plugins_pkg.get_transform("__nonexistent_plugin_xyz__", "SomeTransform")
        assert result is None, (
            f"get_transform() for unknown plugin should return None, got {result}"
        )

    @pytest.mark.contract
    def test_get_transform_is_function(self, plugins_pkg):
        """``get_transform`` is a function (not a class)."""
        assert inspect.isfunction(plugins_pkg.get_transform)


class TestContractModuleLevelGetattr:
    """§2 — Module-level ``__getattr__`` raises AttributeError for unknowns."""

    @pytest.mark.contract
    def test_getattr_raises_attribute_error_for_unknown(self, plugins_pkg):
        """
        Accessing an unknown attribute raises ``AttributeError``.

        Validates the ``__getattr__`` function's documented contract:
        it should raise ``AttributeError`` when the name is not a known
        sub-plugin or attribute.
        """
        with pytest.raises(AttributeError):
            _ = plugins_pkg.__completely_nonexistent_attr_xyz__

    @pytest.mark.contract
    def test_getattr_error_message_includes_module_name(self, plugins_pkg):
        """
        The ``AttributeError`` from ``__getattr__`` mentions the module name.
        """
        with pytest.raises(AttributeError, match=r"milia_pipeline\.plugins"):
            _ = plugins_pkg.__completely_nonexistent_attr_xyz__


class TestContractDiscoverSubplugins:
    """§2 — ``_discover_subplugins()`` return type and idempotency."""

    @pytest.mark.contract
    def test_discover_subplugins_returns_dict(self, plugins_pkg):
        """``_discover_subplugins()`` returns a dict."""
        result = plugins_pkg._discover_subplugins()
        assert isinstance(result, dict), (
            f"_discover_subplugins() should return dict, got {type(result).__name__}"
        )

    @pytest.mark.contract
    def test_discover_subplugins_keys_are_strings(self, plugins_pkg):
        """``_discover_subplugins()`` dict keys are strings."""
        result = plugins_pkg._discover_subplugins()
        for key in result:
            assert isinstance(key, str), (
                f"_discover_subplugins() key should be str, got {type(key).__name__}: {key}"
            )

    @pytest.mark.contract
    def test_discover_subplugins_is_idempotent(self, plugins_pkg):
        """Calling ``_discover_subplugins()`` twice returns the same object."""
        result1 = plugins_pkg._discover_subplugins()
        result2 = plugins_pkg._discover_subplugins()
        assert result1 is result2, (
            "_discover_subplugins() should return the cached dict on repeated calls"
        )

    @pytest.mark.contract
    def test_discover_subplugins_sets_discovery_attempted(self, plugins_pkg):
        """After calling ``_discover_subplugins()``, ``_discovery_attempted`` is True."""
        plugins_pkg._discover_subplugins()
        assert plugins_pkg._discovery_attempted is True, (
            "_discovery_attempted should be True after calling _discover_subplugins()"
        )


class TestContractGetSubpluginModule:
    """§2 — ``_get_subplugin_module()`` contract for unknown names."""

    @pytest.mark.contract
    def test_get_subplugin_module_unknown_returns_none(self, plugins_pkg):
        """``_get_subplugin_module()`` returns None for an unknown name."""
        result = plugins_pkg._get_subplugin_module("__nonexistent_plugin_xyz__")
        assert result is None, (
            f"_get_subplugin_module() for unknown name should return None, got {result}"
        )

    @pytest.mark.contract
    def test_get_subplugin_module_is_function(self, plugins_pkg):
        """``_get_subplugin_module`` is a function (not a class)."""
        assert inspect.isfunction(plugins_pkg._get_subplugin_module)


class TestContractThreadSafetyPrimitives:
    """§2 — Thread-safety primitives have expected types."""

    @pytest.mark.contract
    def test_lock_is_threading_lock(self, plugins_pkg):
        """``_lock`` is a ``threading.Lock`` instance."""
        assert isinstance(plugins_pkg._lock, type(threading.Lock())), (
            f"_lock should be threading.Lock, got {type(plugins_pkg._lock).__name__}"
        )


class TestContractFunctionSignatures:
    """§2 — Key API functions have documented parameter signatures."""

    @pytest.mark.contract
    def test_get_subplugin_info_accepts_name_param(self, plugins_pkg):
        """``get_subplugin_info`` has a parameter for the plugin name."""
        sig = inspect.signature(plugins_pkg.get_subplugin_info)
        param_names = list(sig.parameters.keys())
        assert len(param_names) >= 1, (
            "get_subplugin_info should accept at least one parameter (name)"
        )
        assert param_names[0] == "name", (
            f"First parameter of get_subplugin_info should be 'name', got '{param_names[0]}'"
        )

    @pytest.mark.contract
    def test_enable_subplugin_accepts_name_param(self, plugins_pkg):
        """``enable_subplugin`` has a parameter for the plugin name."""
        sig = inspect.signature(plugins_pkg.enable_subplugin)
        param_names = list(sig.parameters.keys())
        assert len(param_names) >= 1, "enable_subplugin should accept at least one parameter (name)"
        assert param_names[0] == "name", (
            f"First parameter of enable_subplugin should be 'name', got '{param_names[0]}'"
        )

    @pytest.mark.contract
    def test_disable_subplugin_accepts_name_param(self, plugins_pkg):
        """``disable_subplugin`` has a parameter for the plugin name."""
        sig = inspect.signature(plugins_pkg.disable_subplugin)
        param_names = list(sig.parameters.keys())
        assert len(param_names) >= 1, (
            "disable_subplugin should accept at least one parameter (name)"
        )
        assert param_names[0] == "name", (
            f"First parameter of disable_subplugin should be 'name', got '{param_names[0]}'"
        )

    @pytest.mark.contract
    def test_get_subplugin_transforms_accepts_name_param(self, plugins_pkg):
        """``get_subplugin_transforms`` has a parameter for the plugin name."""
        sig = inspect.signature(plugins_pkg.get_subplugin_transforms)
        param_names = list(sig.parameters.keys())
        assert len(param_names) >= 1, (
            "get_subplugin_transforms should accept at least one parameter (name)"
        )
        assert param_names[0] == "name", (
            f"First parameter of get_subplugin_transforms should be 'name', got '{param_names[0]}'"
        )

    @pytest.mark.contract
    def test_get_transform_accepts_two_params(self, plugins_pkg):
        """``get_transform`` has parameters for plugin name and transform name."""
        sig = inspect.signature(plugins_pkg.get_transform)
        param_names = list(sig.parameters.keys())
        assert len(param_names) >= 2, (
            "get_transform should accept at least two parameters (name, transform_name)"
        )
        assert param_names[0] == "name", (
            f"First parameter of get_transform should be 'name', got '{param_names[0]}'"
        )
        assert param_names[1] == "transform_name", (
            f"Second parameter of get_transform should be 'transform_name', got '{param_names[1]}'"
        )

    @pytest.mark.contract
    def test_list_subplugins_no_required_params(self, plugins_pkg):
        """``list_subplugins`` takes no required parameters."""
        sig = inspect.signature(plugins_pkg.list_subplugins)
        required_params = [
            name
            for name, param in sig.parameters.items()
            if param.default is inspect.Parameter.empty
        ]
        assert len(required_params) == 0, (
            f"list_subplugins should take no required parameters, got {required_params}"
        )

    @pytest.mark.contract
    def test_get_plugins_directory_no_required_params(self, plugins_pkg):
        """``get_plugins_directory`` takes no required parameters."""
        sig = inspect.signature(plugins_pkg.get_plugins_directory)
        required_params = [
            name
            for name, param in sig.parameters.items()
            if param.default is inspect.Parameter.empty
        ]
        assert len(required_params) == 0, (
            f"get_plugins_directory should take no required parameters, got {required_params}"
        )

    @pytest.mark.contract
    def test_get_descriptor_plugins_directory_no_required_params(self, plugins_pkg):
        """``get_descriptor_plugins_directory`` takes no required parameters."""
        sig = inspect.signature(plugins_pkg.get_descriptor_plugins_directory)
        required_params = [
            name
            for name, param in sig.parameters.items()
            if param.default is inspect.Parameter.empty
        ]
        assert len(required_params) == 0, (
            f"get_descriptor_plugins_directory should take no required parameters, "
            f"got {required_params}"
        )

    @pytest.mark.contract
    def test_get_system_status_no_required_params(self, plugins_pkg):
        """``get_system_status`` takes no required parameters."""
        sig = inspect.signature(plugins_pkg.get_system_status)
        required_params = [
            name
            for name, param in sig.parameters.items()
            if param.default is inspect.Parameter.empty
        ]
        assert len(required_params) == 0, (
            f"get_system_status should take no required parameters, got {required_params}"
        )

    @pytest.mark.contract
    def test_get_all_plugin_info_no_required_params(self, plugins_pkg):
        """``get_all_plugin_info`` takes no required parameters."""
        sig = inspect.signature(plugins_pkg.get_all_plugin_info)
        required_params = [
            name
            for name, param in sig.parameters.items()
            if param.default is inspect.Parameter.empty
        ]
        assert len(required_params) == 0, (
            f"get_all_plugin_info should take no required parameters, got {required_params}"
        )


class TestContractFunctionTypes:
    """§2 — All public API entries are functions (not classes or methods)."""

    PUBLIC_FUNCTIONS = [
        "list_subplugins",
        "get_subplugin_info",
        "get_all_plugin_info",
        "enable_subplugin",
        "disable_subplugin",
        "get_subplugin_transforms",
        "get_transform",
        "get_plugins_directory",
        "get_descriptor_plugins_directory",
        "get_system_status",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", PUBLIC_FUNCTIONS)
    def test_public_api_entry_is_function(self, plugins_pkg, name):
        """Each public API entry is a function (not a class or method)."""
        obj = getattr(plugins_pkg, name)
        assert inspect.isfunction(obj), f"'{name}' should be a function, got {type(obj).__name__}"


class TestContractConsistencyBetweenAPIs:
    """§2 — Consistency between related API functions."""

    @pytest.mark.contract
    def test_list_subplugins_consistent_with_system_status(self, plugins_pkg):
        """
        ``list_subplugins()`` is consistent with
        ``get_system_status()['discovery']['subplugin_names']``.
        """
        listed = plugins_pkg.list_subplugins()
        status = plugins_pkg.get_system_status()
        status_names = status["discovery"]["subplugin_names"]
        assert listed == status_names, (
            f"list_subplugins() result {listed} does not match "
            f"get_system_status() subplugin_names {status_names}"
        )

    @pytest.mark.contract
    def test_subplugins_count_consistent_with_system_status(self, plugins_pkg):
        """
        ``len(list_subplugins())`` matches
        ``get_system_status()['discovery']['subplugins_found']``.
        """
        listed = plugins_pkg.list_subplugins()
        status = plugins_pkg.get_system_status()
        found_count = status["discovery"]["subplugins_found"]
        assert len(listed) == found_count, (
            f"list_subplugins() count {len(listed)} does not match "
            f"get_system_status() subplugins_found {found_count}"
        )

    @pytest.mark.contract
    def test_get_all_plugin_info_keys_subset_of_list_subplugins(self, plugins_pkg):
        """
        Keys of ``get_all_plugin_info()`` are a subset of
        ``list_subplugins()`` (only successfully loadable plugins appear).
        """
        all_info = plugins_pkg.get_all_plugin_info()
        listed = set(plugins_pkg.list_subplugins())
        extra_keys = set(all_info.keys()) - listed
        assert not extra_keys, (
            f"get_all_plugin_info() has keys not in list_subplugins(): {sorted(extra_keys)}"
        )

    @pytest.mark.contract
    def test_plugins_directory_consistent_with_system_status(self, plugins_pkg):
        """
        ``get_plugins_directory()`` is consistent with
        ``get_system_status()['plugins_directory']``.
        """
        dir_path = plugins_pkg.get_plugins_directory()
        status = plugins_pkg.get_system_status()
        assert str(dir_path) == status["plugins_directory"], (
            f"get_plugins_directory() '{dir_path}' does not match "
            f"get_system_status() plugins_directory '{status['plugins_directory']}'"
        )

    @pytest.mark.contract
    def test_descriptor_directory_consistent_with_system_status(self, plugins_pkg):
        """
        ``get_descriptor_plugins_directory()`` is consistent with
        ``get_system_status()['descriptor_plugins_directory']``.
        """
        dir_path = plugins_pkg.get_descriptor_plugins_directory()
        status = plugins_pkg.get_system_status()
        assert str(dir_path) == status["descriptor_plugins_directory"], (
            f"get_descriptor_plugins_directory() '{dir_path}' does not match "
            f"get_system_status() descriptor_plugins_directory "
            f"'{status['descriptor_plugins_directory']}'"
        )
