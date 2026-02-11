# tests/test__init__transformations.py

"""
Test Suite: milia_pipeline/transformations/__init__.py — Smoke Tests & Contract Tests
=====================================================================================

Production-ready test suite for the MILIA Pipeline transformations package
``milia_pipeline/transformations/__init__.py``.

Covers:
    Section 1 — Smoke Tests (MILIA_Test_Recommendations.md §1.2 scope):
        - The ``milia_pipeline.transformations`` subpackage imports without ImportError
        - All re-exported names from the 4 submodules are accessible
        - Module-level metadata attributes (__version__, __author__, __license__) exist
        - Module initialization (logging, availability flags) runs safely
        - Re-import (``importlib.reload``) is idempotent and non-crashing
        - Feature availability flags are present and boolean
        - Circular dependency resolution system (_INITIALIZING, _INITIALIZED) is present
        - Module-level convenience functions are accessible and callable
        - Graph transforms core class exports are accessible
        - Custom transforms base class exports are accessible
        - Plugin system exports are accessible
        - Research API exports are accessible
        - Fallback values for unavailable submodules are set correctly

    Section 2 — Contract Tests (MILIA_Test_Recommendations.md §2 scope):
        - ``__all__`` completeness: every name in ``__all__`` is resolvable
        - ``__all__`` consistency: every public import is listed in ``__all__``
        - ``__all__`` has no duplicates
        - Re-exported classes are classes (when submodule is available)
        - Availability flags control access to submodule exports
        - Convenience functions have documented parameter signatures
        - Module-level convenience functions raise ImportError when submodule unavailable
        - ``__version__`` follows semver pattern
        - Public API surface stability (minimum expected names present)
        - ``_ensure_initialized`` is callable and idempotent
        - ``get_system_status()`` returns dict with expected keys
        - ``get_module_info()`` returns dict with expected keys
        - Exception fallback classes are Exception subclasses
        - Feature availability flags are mutually consistent

Design:
    - Zero ``sys.modules`` pollution: all mocking via ``@patch`` decorators
      or context-manager ``patch`` calls scoped to individual tests.
    - Deterministic: no filesystem, network, or GPU access required.
    - Isolated: each test is independent; execution order is irrelevant.
    - Fast: expected < 5 s total wall-clock on CI.

Launch:
    From project root (/app/milia):
        pytest tests/test__init__transformations.py -v --tb=short

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
# When launched via ``pytest tests/test__init__transformations.py`` from the
# project root (/app/milia), ``milia_pipeline`` must be on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # …/tests -> …/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def transformations_pkg():
    """
    Import and return the ``milia_pipeline.transformations`` package once per module.

    This fixture validates the fundamental smoke invariant: the transformations
    subpackage is importable.  If this fails, every downstream test is moot.
    """
    try:
        import milia_pipeline.transformations as tfm
        return tfm
    except ImportError as exc:
        pytest.fail(
            f"milia_pipeline.transformations could not be imported — smoke test "
            f"precondition violated: {exc}"
        )


@pytest.fixture(scope="module")
def all_names(transformations_pkg):
    """Return the ``__all__`` list from the transformations package."""
    assert hasattr(transformations_pkg, "__all__"), (
        "milia_pipeline.transformations.__all__ is missing — contract violation"
    )
    return list(transformations_pkg.__all__)


# ===================================================================
# SECTION 1 — SMOKE TESTS
# ===================================================================


class TestSmokeTransformationsPackageImport:
    """§1.2 — Verify the transformations subpackage imports without errors."""

    @pytest.mark.smoke
    def test_import_transformations_package_succeeds(self, transformations_pkg):
        """The transformations package imports without raising any exception."""
        assert transformations_pkg is not None

    @pytest.mark.smoke
    def test_transformations_package_is_a_module(self, transformations_pkg):
        """The imported object is a proper Python module."""
        assert isinstance(transformations_pkg, types.ModuleType)

    @pytest.mark.smoke
    def test_transformations_package_has_file_attribute(self, transformations_pkg):
        """The package exposes a ``__file__`` attribute (not a namespace pkg)."""
        assert hasattr(transformations_pkg, "__file__")

    @pytest.mark.smoke
    def test_transformations_package_name(self, transformations_pkg):
        """The package ``__name__`` is ``milia_pipeline.transformations``."""
        assert transformations_pkg.__name__ == "milia_pipeline.transformations"


class TestSmokeMetadataAttributes:
    """§1.2 — Verify module-level metadata attributes are present and typed."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("attr", [
        "__version__",
        "__author__",
        "__license__",
    ])
    def test_metadata_attribute_exists(self, transformations_pkg, attr):
        """Each metadata dunder is defined on the transformations package."""
        assert hasattr(transformations_pkg, attr), f"Missing attribute: {attr}"

    @pytest.mark.smoke
    @pytest.mark.parametrize("attr", [
        "__version__",
        "__author__",
        "__license__",
    ])
    def test_metadata_attribute_is_string(self, transformations_pkg, attr):
        """Each metadata dunder is a non-empty string."""
        value = getattr(transformations_pkg, attr)
        assert isinstance(value, str), f"{attr} should be str, got {type(value)}"
        assert len(value) > 0, f"{attr} should be non-empty"

    @pytest.mark.smoke
    def test_version_is_semver_like(self, transformations_pkg):
        """``__version__`` follows a MAJOR.MINOR.PATCH pattern."""
        version = transformations_pkg.__version__
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


class TestSmokeFeatureAvailabilityFlags:
    """§1.2 — Feature availability flags exist and are boolean."""

    AVAILABILITY_FLAGS = [
        "GRAPH_TRANSFORMS_AVAILABLE",
        "CUSTOM_TRANSFORMS_AVAILABLE",
        "PLUGIN_SYSTEM_AVAILABLE",
        "RESEARCH_API_AVAILABLE",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("flag", AVAILABILITY_FLAGS)
    def test_availability_flag_exists(self, transformations_pkg, flag):
        """Each feature availability flag is defined on the transformations package."""
        assert hasattr(transformations_pkg, flag), f"Flag '{flag}' is missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("flag", AVAILABILITY_FLAGS)
    def test_availability_flag_is_bool(self, transformations_pkg, flag):
        """Each feature availability flag is actually a bool."""
        value = getattr(transformations_pkg, flag)
        assert isinstance(value, bool), (
            f"Flag '{flag}' should be bool, got {type(value).__name__}"
        )


class TestSmokeCircularDependencyResolutionSystem:
    """§1.2 — Circular dependency resolution system attributes are present."""

    @pytest.mark.smoke
    def test_initializing_flag_exists(self, transformations_pkg):
        """``_INITIALIZING`` flag is defined."""
        assert hasattr(transformations_pkg, "_INITIALIZING")

    @pytest.mark.smoke
    def test_initialized_flag_exists(self, transformations_pkg):
        """``_INITIALIZED`` flag is defined."""
        assert hasattr(transformations_pkg, "_INITIALIZED")

    @pytest.mark.smoke
    def test_initializing_flag_is_bool(self, transformations_pkg):
        """``_INITIALIZING`` is a bool."""
        value = transformations_pkg._INITIALIZING
        assert isinstance(value, bool), (
            f"_INITIALIZING should be bool, got {type(value).__name__}"
        )

    @pytest.mark.smoke
    def test_initialized_flag_is_bool(self, transformations_pkg):
        """``_INITIALIZED`` is a bool."""
        value = transformations_pkg._INITIALIZED
        assert isinstance(value, bool), (
            f"_INITIALIZED should be bool, got {type(value).__name__}"
        )

    @pytest.mark.smoke
    def test_ensure_initialized_exists(self, transformations_pkg):
        """``_ensure_initialized`` function is present."""
        assert hasattr(transformations_pkg, "_ensure_initialized")

    @pytest.mark.smoke
    def test_ensure_initialized_is_callable(self, transformations_pkg):
        """``_ensure_initialized`` is callable."""
        assert callable(transformations_pkg._ensure_initialized)


class TestSmokeGraphTransformsExports:
    """§1.2 — Graph transforms core class exports are accessible."""

    GRAPH_TRANSFORM_CORE_CLASSES = [
        "TransformRegistry",
        "TransformComposer",
        "TransformValidator",
        "DynamicTransformDiscovery",
        "ConfigurationBridge",
        "TransformErrorRecovery",
    ]

    GRAPH_TRANSFORM_ENUMS = [
        "ValidationLevel",
        "ValidationScope",
    ]

    GRAPH_TRANSFORM_METADATA_CLASSES = [
        "TransformInfo",
        "TransformDependency",
        "TransformCompatibility",
    ]

    GRAPH_TRANSFORM_API = [
        "GraphTransforms",
        "get_graph_transforms",
    ]

    GRAPH_TRANSFORM_CONVENIENCE = [
        "get_transform_info",
        "validate_v3_configuration",
        "validate_comprehensive",
        "get_configuration_format_help",
        "export_metrics",
        "optimize_performance",
        "get_milia_setups",
        "perform_system_health_check",
        "get_validation_report_text",
        "discover_custom_transforms",
        "register_all_custom_transforms",
    ]

    ALL_GRAPH_EXPORTS = (
        GRAPH_TRANSFORM_CORE_CLASSES
        + GRAPH_TRANSFORM_ENUMS
        + GRAPH_TRANSFORM_METADATA_CLASSES
        + GRAPH_TRANSFORM_API
        + GRAPH_TRANSFORM_CONVENIENCE
    )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", ALL_GRAPH_EXPORTS)
    def test_graph_transform_export_exists(self, transformations_pkg, name):
        """Each graph transform export is defined on the transformations package."""
        assert hasattr(transformations_pkg, name), (
            f"Graph transform export '{name}' is missing"
        )

    @pytest.mark.smoke
    def test_graph_transforms_available_controls_exports(self, transformations_pkg):
        """
        When GRAPH_TRANSFORMS_AVAILABLE is True, core classes are non-None.
        When False, they are set to None as fallbacks.
        """
        if transformations_pkg.GRAPH_TRANSFORMS_AVAILABLE:
            for name in self.GRAPH_TRANSFORM_CORE_CLASSES:
                obj = getattr(transformations_pkg, name)
                assert obj is not None, (
                    f"'{name}' should be non-None when GRAPH_TRANSFORMS_AVAILABLE is True"
                )
        else:
            for name in self.GRAPH_TRANSFORM_CORE_CLASSES:
                obj = getattr(transformations_pkg, name)
                assert obj is None, (
                    f"'{name}' should be None when GRAPH_TRANSFORMS_AVAILABLE is False"
                )


class TestSmokeCustomTransformsExports:
    """§1.2 — Custom transforms base class exports are accessible."""

    CUSTOM_TRANSFORM_BASE_CLASSES = [
        "CustomTransformBase",
        "MolecularTransformBase",
        "QuantumTransformBase",
    ]

    CUSTOM_TRANSFORM_METADATA = [
        "TransformMetadata",
    ]

    CUSTOM_TRANSFORM_EXAMPLES = [
        "NormalizeVibrationalModes",
        "FilterByDMCUncertainty",
        "ScaleMullikenCharges",
    ]

    CUSTOM_TRANSFORM_EXCEPTIONS = [
        "TransformValidationError",
        "TransformExecutionError",
        "TransformConfigurationError",
    ]

    ALL_CUSTOM_EXPORTS = (
        CUSTOM_TRANSFORM_BASE_CLASSES
        + CUSTOM_TRANSFORM_METADATA
        + CUSTOM_TRANSFORM_EXAMPLES
        + CUSTOM_TRANSFORM_EXCEPTIONS
    )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", ALL_CUSTOM_EXPORTS)
    def test_custom_transform_export_exists(self, transformations_pkg, name):
        """Each custom transform export is defined on the transformations package."""
        assert hasattr(transformations_pkg, name), (
            f"Custom transform export '{name}' is missing"
        )

    @pytest.mark.smoke
    def test_custom_transforms_available_controls_base_classes(self, transformations_pkg):
        """
        When CUSTOM_TRANSFORMS_AVAILABLE is True, base classes are non-None.
        When False, they are set to None as fallbacks.
        """
        if transformations_pkg.CUSTOM_TRANSFORMS_AVAILABLE:
            for name in self.CUSTOM_TRANSFORM_BASE_CLASSES:
                obj = getattr(transformations_pkg, name)
                assert obj is not None, (
                    f"'{name}' should be non-None when CUSTOM_TRANSFORMS_AVAILABLE is True"
                )
        else:
            for name in self.CUSTOM_TRANSFORM_BASE_CLASSES:
                obj = getattr(transformations_pkg, name)
                assert obj is None, (
                    f"'{name}' should be None when CUSTOM_TRANSFORMS_AVAILABLE is False"
                )


class TestSmokePluginSystemExports:
    """§1.2 — Plugin system exports are accessible."""

    PLUGIN_CORE_CLASSES = [
        "PluginMetadata",
        "PluginRegistry",
        "PluginValidator",
        "TransformDeclaration",
    ]

    PLUGIN_EXCEPTIONS = [
        "PluginError",
        "PluginValidationError",
        "PluginSecurityError",
        "PluginDependencyError",
    ]

    ALL_PLUGIN_EXPORTS = PLUGIN_CORE_CLASSES + PLUGIN_EXCEPTIONS

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", ALL_PLUGIN_EXPORTS)
    def test_plugin_export_exists(self, transformations_pkg, name):
        """Each plugin system export is defined on the transformations package."""
        assert hasattr(transformations_pkg, name), (
            f"Plugin system export '{name}' is missing"
        )

    @pytest.mark.smoke
    def test_plugin_system_available_controls_core_classes(self, transformations_pkg):
        """
        When PLUGIN_SYSTEM_AVAILABLE is True, core classes are non-None.
        When False, they are set to None as fallbacks.
        """
        if transformations_pkg.PLUGIN_SYSTEM_AVAILABLE:
            for name in self.PLUGIN_CORE_CLASSES:
                obj = getattr(transformations_pkg, name)
                assert obj is not None, (
                    f"'{name}' should be non-None when PLUGIN_SYSTEM_AVAILABLE is True"
                )
        else:
            for name in self.PLUGIN_CORE_CLASSES:
                obj = getattr(transformations_pkg, name)
                assert obj is None, (
                    f"'{name}' should be None when PLUGIN_SYSTEM_AVAILABLE is False"
                )


class TestSmokeResearchAPIExports:
    """§1.2 — Research API exports are accessible."""

    RESEARCH_CONFIGURATION = [
        "ExperimentConfiguration",
    ]

    RESEARCH_BUILDERS = [
        "AblationStudyBuilder",
        "ParameterSweepBuilder",
        "ComparativeStudyBuilder",
    ]

    RESEARCH_EXECUTION = [
        "ExperimentRunner",
    ]

    RESEARCH_CONVENIENCE = [
        "create_ablation_study",
        "create_parameter_sweep",
        "create_comparative_study",
        "load_experiments_from_config",
        "get_experiment",
        "list_available_experiments",
    ]

    ALL_RESEARCH_EXPORTS = (
        RESEARCH_CONFIGURATION
        + RESEARCH_BUILDERS
        + RESEARCH_EXECUTION
        + RESEARCH_CONVENIENCE
    )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", ALL_RESEARCH_EXPORTS)
    def test_research_api_export_exists(self, transformations_pkg, name):
        """Each research API export is defined on the transformations package."""
        assert hasattr(transformations_pkg, name), (
            f"Research API export '{name}' is missing"
        )

    @pytest.mark.smoke
    def test_research_api_available_controls_exports(self, transformations_pkg):
        """
        When RESEARCH_API_AVAILABLE is True, core classes are non-None.
        When False, they are set to None as fallbacks.
        """
        if transformations_pkg.RESEARCH_API_AVAILABLE:
            for name in self.RESEARCH_CONFIGURATION + self.RESEARCH_BUILDERS:
                obj = getattr(transformations_pkg, name)
                assert obj is not None, (
                    f"'{name}' should be non-None when RESEARCH_API_AVAILABLE is True"
                )
        else:
            for name in self.RESEARCH_CONFIGURATION + self.RESEARCH_BUILDERS:
                obj = getattr(transformations_pkg, name)
                assert obj is None, (
                    f"'{name}' should be None when RESEARCH_API_AVAILABLE is False"
                )


class TestSmokeConvenienceFunctionExports:
    """§1.2 — Module-level convenience functions are accessible and callable."""

    CONVENIENCE_FUNCTIONS = [
        "get_available_transforms",
        "create_transform_sequence",
        "validate_transform_config",
        "register_custom_transform",
        "discover_and_register_plugins",
        "get_system_status",
        "get_module_info",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CONVENIENCE_FUNCTIONS)
    def test_convenience_function_exists(self, transformations_pkg, name):
        """Each module-level convenience function is present and non-None."""
        obj = getattr(transformations_pkg, name, None)
        assert obj is not None, f"Convenience function '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CONVENIENCE_FUNCTIONS)
    def test_convenience_function_is_callable(self, transformations_pkg, name):
        """Each module-level convenience function is callable."""
        obj = getattr(transformations_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeModuleInitialization:
    """§1.2 — Module-level initialization runs without side-effect crashes."""

    @pytest.mark.smoke
    def test_reimport_is_idempotent(self, transformations_pkg):
        """
        Re-importing the transformations package (via ``importlib.reload``)
        does not crash.

        Validates that all module-level code (logging, availability flag checks,
        circular dependency resolution) is safe to re-execute.
        """
        reloaded = importlib.reload(transformations_pkg)
        assert reloaded is not None
        assert hasattr(reloaded, "__version__")

    @pytest.mark.smoke
    def test_reimport_preserves_all(self, transformations_pkg):
        """
        Re-importing the transformations package preserves ``__all__``.
        """
        reloaded = importlib.reload(transformations_pkg)
        assert hasattr(reloaded, "__all__")
        assert isinstance(reloaded.__all__, list)
        assert len(reloaded.__all__) > 0

    @pytest.mark.smoke
    def test_reimport_preserves_availability_flags(self, transformations_pkg):
        """
        Re-importing preserves feature availability flags.
        """
        original_graph = transformations_pkg.GRAPH_TRANSFORMS_AVAILABLE
        original_custom = transformations_pkg.CUSTOM_TRANSFORMS_AVAILABLE
        original_plugin = transformations_pkg.PLUGIN_SYSTEM_AVAILABLE
        original_research = transformations_pkg.RESEARCH_API_AVAILABLE

        reloaded = importlib.reload(transformations_pkg)

        assert reloaded.GRAPH_TRANSFORMS_AVAILABLE == original_graph
        assert reloaded.CUSTOM_TRANSFORMS_AVAILABLE == original_custom
        assert reloaded.PLUGIN_SYSTEM_AVAILABLE == original_plugin
        assert reloaded.RESEARCH_API_AVAILABLE == original_research

    @pytest.mark.smoke
    def test_logger_is_present(self, transformations_pkg):
        """
        The package-level ``logger`` attribute is a logging.Logger instance.
        """
        assert hasattr(transformations_pkg, "logger")
        assert isinstance(transformations_pkg.logger, logging.Logger)

    @pytest.mark.smoke
    def test_logger_name_matches_package(self, transformations_pkg):
        """
        The logger name matches the package name.
        """
        assert transformations_pkg.logger.name == "milia_pipeline.transformations"


class TestSmokeExceptionFallbacks:
    """§1.2 — Exception fallback assignments are accessible."""

    EXCEPTION_NAMES = [
        "TransformValidationError",
        "TransformExecutionError",
        "TransformConfigurationError",
        "PluginError",
        "PluginValidationError",
        "PluginSecurityError",
        "PluginDependencyError",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", EXCEPTION_NAMES)
    def test_exception_export_exists(self, transformations_pkg, name):
        """Each exception export is defined on the transformations package."""
        assert hasattr(transformations_pkg, name), (
            f"Exception export '{name}' is missing"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", EXCEPTION_NAMES)
    def test_exception_export_is_not_none(self, transformations_pkg, name):
        """
        Each exception export is never None — it is either the real exception
        class from the submodule, or the ``Exception`` fallback.
        """
        obj = getattr(transformations_pkg, name)
        assert obj is not None, f"Exception '{name}' should never be None"


# ===================================================================
# SECTION 2 — CONTRACT TESTS
# ===================================================================


class TestContractAllCompleteness:
    """§2 — Every name in ``__all__`` is resolvable on the transformations package."""

    @pytest.mark.contract
    def test_all_is_a_list(self, transformations_pkg):
        """``__all__`` is a list (not a tuple or set)."""
        assert isinstance(transformations_pkg.__all__, list)

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
    def test_every_all_entry_is_resolvable(self, transformations_pkg, all_names):
        """
        Generic sweep: every single entry in ``__all__`` must be resolvable,
        regardless of whether it is parameterized individually.
        """
        unresolvable = [
            name for name in all_names
            if not hasattr(transformations_pkg, name)
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
    """§2 — Every public import in the transformations module is listed in ``__all__``."""

    # Names that are intentionally public-ish but NOT in __all__
    KNOWN_UNLISTED = {
        # Internal state flags
        "_INITIALIZING",
        "_INITIALIZED",
        # Module logger
        "logger",
        # Standard library imports used in module-level convenience functions
        # (typing symbols imported at module level for type hints)
        "logging",
        "Dict",
        "List",
        "Any",
        "Optional",
        "Type",
    }

    @pytest.mark.contract
    def test_public_imports_are_in_all(self, transformations_pkg, all_names):
        """
        Every non-dunder, non-private attribute that was imported (not
        defined locally) should be in ``__all__`` — unless it is in
        ``KNOWN_UNLISTED``.

        This catches accidental omissions when new imports are added to
        the transformations ``__init__.py`` without updating ``__all__``.
        """
        all_set = set(all_names)
        module_dict = vars(transformations_pkg)
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
            f"Public names imported in transformations/__init__.py but not in __all__: "
            f"{sorted(missing_from_all)}"
        )


class TestContractGraphTransformClassTypes:
    """§2 — Graph transform core exports are classes when available."""

    CORE_CLASSES = [
        "TransformRegistry",
        "TransformComposer",
        "TransformValidator",
        "DynamicTransformDiscovery",
        "ConfigurationBridge",
        "TransformErrorRecovery",
    ]

    ENUM_CLASSES = [
        "ValidationLevel",
        "ValidationScope",
    ]

    METADATA_CLASSES = [
        "TransformInfo",
        "TransformDependency",
        "TransformCompatibility",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", CORE_CLASSES)
    def test_core_class_is_class_when_available(self, transformations_pkg, name):
        """Each core graph transform export is a class when GRAPH_TRANSFORMS_AVAILABLE."""
        if not transformations_pkg.GRAPH_TRANSFORMS_AVAILABLE:
            pytest.skip("graph_transforms submodule not available")
        obj = getattr(transformations_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", ENUM_CLASSES)
    def test_enum_class_is_class_when_available(self, transformations_pkg, name):
        """Each enum export is a class when GRAPH_TRANSFORMS_AVAILABLE."""
        if not transformations_pkg.GRAPH_TRANSFORMS_AVAILABLE:
            pytest.skip("graph_transforms submodule not available")
        obj = getattr(transformations_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class (enum), got {type(obj).__name__}"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", METADATA_CLASSES)
    def test_metadata_class_is_class_when_available(self, transformations_pkg, name):
        """Each metadata export is a class when GRAPH_TRANSFORMS_AVAILABLE."""
        if not transformations_pkg.GRAPH_TRANSFORMS_AVAILABLE:
            pytest.skip("graph_transforms submodule not available")
        obj = getattr(transformations_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )

    @pytest.mark.contract
    def test_graph_transforms_main_api_is_class_when_available(self, transformations_pkg):
        """``GraphTransforms`` is a class when GRAPH_TRANSFORMS_AVAILABLE."""
        if not transformations_pkg.GRAPH_TRANSFORMS_AVAILABLE:
            pytest.skip("graph_transforms submodule not available")
        assert inspect.isclass(transformations_pkg.GraphTransforms), (
            f"GraphTransforms should be a class, got "
            f"{type(transformations_pkg.GraphTransforms).__name__}"
        )

    @pytest.mark.contract
    def test_get_graph_transforms_is_callable_when_available(self, transformations_pkg):
        """``get_graph_transforms`` is callable when GRAPH_TRANSFORMS_AVAILABLE."""
        if not transformations_pkg.GRAPH_TRANSFORMS_AVAILABLE:
            pytest.skip("graph_transforms submodule not available")
        assert callable(transformations_pkg.get_graph_transforms), (
            "get_graph_transforms should be callable"
        )


class TestContractCustomTransformClassTypes:
    """§2 — Custom transform base exports are classes when available."""

    BASE_CLASSES = [
        "CustomTransformBase",
        "MolecularTransformBase",
        "QuantumTransformBase",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", BASE_CLASSES)
    def test_base_class_is_class_when_available(self, transformations_pkg, name):
        """Each custom transform base class is a class when CUSTOM_TRANSFORMS_AVAILABLE."""
        if not transformations_pkg.CUSTOM_TRANSFORMS_AVAILABLE:
            pytest.skip("custom_transforms submodule not available")
        obj = getattr(transformations_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )

    @pytest.mark.contract
    def test_transform_metadata_is_class_when_available(self, transformations_pkg):
        """``TransformMetadata`` is a class when CUSTOM_TRANSFORMS_AVAILABLE."""
        if not transformations_pkg.CUSTOM_TRANSFORMS_AVAILABLE:
            pytest.skip("custom_transforms submodule not available")
        assert inspect.isclass(transformations_pkg.TransformMetadata), (
            f"TransformMetadata should be a class, got "
            f"{type(transformations_pkg.TransformMetadata).__name__}"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", [
        "NormalizeVibrationalModes",
        "FilterByDMCUncertainty",
        "ScaleMullikenCharges",
    ])
    def test_example_transform_is_class_when_available(self, transformations_pkg, name):
        """Each example transform is a class when CUSTOM_TRANSFORMS_AVAILABLE."""
        if not transformations_pkg.CUSTOM_TRANSFORMS_AVAILABLE:
            pytest.skip("custom_transforms submodule not available")
        obj = getattr(transformations_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )


class TestContractPluginSystemClassTypes:
    """§2 — Plugin system core exports are classes when available."""

    CORE_CLASSES = [
        "PluginMetadata",
        "PluginRegistry",
        "PluginValidator",
        "TransformDeclaration",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", CORE_CLASSES)
    def test_plugin_class_is_class_when_available(self, transformations_pkg, name):
        """Each plugin system core class is a class when PLUGIN_SYSTEM_AVAILABLE."""
        if not transformations_pkg.PLUGIN_SYSTEM_AVAILABLE:
            pytest.skip("plugin_system submodule not available")
        obj = getattr(transformations_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )


class TestContractResearchAPIClassTypes:
    """§2 — Research API core exports are classes when available."""

    @pytest.mark.contract
    def test_experiment_configuration_is_class_when_available(self, transformations_pkg):
        """``ExperimentConfiguration`` is a class when RESEARCH_API_AVAILABLE."""
        if not transformations_pkg.RESEARCH_API_AVAILABLE:
            pytest.skip("research_api submodule not available")
        assert inspect.isclass(transformations_pkg.ExperimentConfiguration), (
            f"ExperimentConfiguration should be a class, got "
            f"{type(transformations_pkg.ExperimentConfiguration).__name__}"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", [
        "AblationStudyBuilder",
        "ParameterSweepBuilder",
        "ComparativeStudyBuilder",
    ])
    def test_builder_is_class_when_available(self, transformations_pkg, name):
        """Each builder class is a class when RESEARCH_API_AVAILABLE."""
        if not transformations_pkg.RESEARCH_API_AVAILABLE:
            pytest.skip("research_api submodule not available")
        obj = getattr(transformations_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )

    @pytest.mark.contract
    def test_experiment_runner_is_class_when_available(self, transformations_pkg):
        """``ExperimentRunner`` is a class when RESEARCH_API_AVAILABLE."""
        if not transformations_pkg.RESEARCH_API_AVAILABLE:
            pytest.skip("research_api submodule not available")
        assert inspect.isclass(transformations_pkg.ExperimentRunner), (
            f"ExperimentRunner should be a class, got "
            f"{type(transformations_pkg.ExperimentRunner).__name__}"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", [
        "create_ablation_study",
        "create_parameter_sweep",
        "create_comparative_study",
        "load_experiments_from_config",
        "get_experiment",
        "list_available_experiments",
    ])
    def test_research_convenience_is_callable_when_available(self, transformations_pkg, name):
        """Each research convenience function is callable when RESEARCH_API_AVAILABLE."""
        if not transformations_pkg.RESEARCH_API_AVAILABLE:
            pytest.skip("research_api submodule not available")
        obj = getattr(transformations_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestContractExceptionFallbackTypes:
    """§2 — Exception fallback classes are always Exception subclasses."""

    CUSTOM_TRANSFORM_EXCEPTIONS = [
        "TransformValidationError",
        "TransformExecutionError",
        "TransformConfigurationError",
    ]

    PLUGIN_EXCEPTIONS = [
        "PluginError",
        "PluginValidationError",
        "PluginSecurityError",
        "PluginDependencyError",
    ]

    ALL_EXCEPTIONS = CUSTOM_TRANSFORM_EXCEPTIONS + PLUGIN_EXCEPTIONS

    @pytest.mark.contract
    @pytest.mark.parametrize("name", ALL_EXCEPTIONS)
    def test_exception_is_class(self, transformations_pkg, name):
        """Each exception export is a class."""
        obj = getattr(transformations_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", ALL_EXCEPTIONS)
    def test_exception_is_exception_subclass(self, transformations_pkg, name):
        """
        Each exception export is a subclass of Exception — whether it is the
        real exception from the submodule or the ``Exception`` fallback.
        """
        cls = getattr(transformations_pkg, name)
        assert issubclass(cls, Exception), (
            f"'{name}' should be a subclass of Exception, got MRO: "
            f"{[c.__name__ for c in cls.__mro__]}"
        )


class TestContractConvenienceFunctionSignatures:
    """§2 — Key convenience functions have documented parameter signatures."""

    @pytest.mark.contract
    def test_get_available_transforms_signature(self, transformations_pkg):
        """``get_available_transforms`` accepts no required arguments."""
        sig = inspect.signature(transformations_pkg.get_available_transforms)
        required = [
            p for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        assert len(required) == 0, (
            f"get_available_transforms should take no required args, "
            f"found: {[p.name for p in required]}"
        )

    @pytest.mark.contract
    def test_create_transform_sequence_accepts_configs(self, transformations_pkg):
        """``create_transform_sequence`` has a ``configs`` parameter."""
        sig = inspect.signature(transformations_pkg.create_transform_sequence)
        param_names = set(sig.parameters.keys())
        assert "configs" in param_names, (
            f"create_transform_sequence should accept 'configs', "
            f"found params: {param_names}"
        )

    @pytest.mark.contract
    def test_create_transform_sequence_accepts_dataset_type(self, transformations_pkg):
        """``create_transform_sequence`` has a ``dataset_type`` parameter."""
        sig = inspect.signature(transformations_pkg.create_transform_sequence)
        param_names = set(sig.parameters.keys())
        assert "dataset_type" in param_names, (
            f"create_transform_sequence should accept 'dataset_type', "
            f"found params: {param_names}"
        )

    @pytest.mark.contract
    def test_validate_transform_config_accepts_configs(self, transformations_pkg):
        """``validate_transform_config`` has a ``configs`` parameter."""
        sig = inspect.signature(transformations_pkg.validate_transform_config)
        param_names = set(sig.parameters.keys())
        assert "configs" in param_names, (
            f"validate_transform_config should accept 'configs', "
            f"found params: {param_names}"
        )

    @pytest.mark.contract
    def test_validate_transform_config_accepts_dataset_type(self, transformations_pkg):
        """``validate_transform_config`` has a ``dataset_type`` parameter."""
        sig = inspect.signature(transformations_pkg.validate_transform_config)
        param_names = set(sig.parameters.keys())
        assert "dataset_type" in param_names, (
            f"validate_transform_config should accept 'dataset_type', "
            f"found params: {param_names}"
        )

    @pytest.mark.contract
    def test_validate_transform_config_accepts_validation_level(self, transformations_pkg):
        """``validate_transform_config`` has a ``validation_level`` parameter."""
        sig = inspect.signature(transformations_pkg.validate_transform_config)
        param_names = set(sig.parameters.keys())
        assert "validation_level" in param_names, (
            f"validate_transform_config should accept 'validation_level', "
            f"found params: {param_names}"
        )

    @pytest.mark.contract
    def test_register_custom_transform_accepts_transform_class(self, transformations_pkg):
        """``register_custom_transform`` has a ``transform_class`` parameter."""
        sig = inspect.signature(transformations_pkg.register_custom_transform)
        param_names = set(sig.parameters.keys())
        assert "transform_class" in param_names, (
            f"register_custom_transform should accept 'transform_class', "
            f"found params: {param_names}"
        )

    @pytest.mark.contract
    def test_register_custom_transform_accepts_force(self, transformations_pkg):
        """``register_custom_transform`` has a ``force`` parameter."""
        sig = inspect.signature(transformations_pkg.register_custom_transform)
        param_names = set(sig.parameters.keys())
        assert "force" in param_names, (
            f"register_custom_transform should accept 'force', "
            f"found params: {param_names}"
        )

    @pytest.mark.contract
    def test_discover_and_register_plugins_accepts_plugin_paths(self, transformations_pkg):
        """``discover_and_register_plugins`` has a ``plugin_paths`` parameter."""
        sig = inspect.signature(transformations_pkg.discover_and_register_plugins)
        param_names = set(sig.parameters.keys())
        assert "plugin_paths" in param_names, (
            f"discover_and_register_plugins should accept 'plugin_paths', "
            f"found params: {param_names}"
        )

    @pytest.mark.contract
    def test_get_system_status_signature(self, transformations_pkg):
        """``get_system_status`` accepts no required arguments."""
        sig = inspect.signature(transformations_pkg.get_system_status)
        required = [
            p for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        assert len(required) == 0, (
            f"get_system_status should take no required args, "
            f"found: {[p.name for p in required]}"
        )

    @pytest.mark.contract
    def test_get_module_info_signature(self, transformations_pkg):
        """``get_module_info`` accepts no required arguments."""
        sig = inspect.signature(transformations_pkg.get_module_info)
        required = [
            p for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        assert len(required) == 0, (
            f"get_module_info should take no required args, "
            f"found: {[p.name for p in required]}"
        )


class TestContractConvenienceFunctionTypes:
    """§2 — Convenience functions are functions (not classes)."""

    CONVENIENCE_FUNCTIONS = [
        "get_available_transforms",
        "create_transform_sequence",
        "validate_transform_config",
        "register_custom_transform",
        "discover_and_register_plugins",
        "get_system_status",
        "get_module_info",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", CONVENIENCE_FUNCTIONS)
    def test_convenience_is_function(self, transformations_pkg, name):
        """Each convenience function is a function (not a class or unbound method)."""
        obj = getattr(transformations_pkg, name)
        assert inspect.isfunction(obj), (
            f"'{name}' should be a function, got {type(obj).__name__}"
        )


class TestContractConvenienceFunctionUnavailableBehavior:
    """
    §2 — Convenience functions raise ImportError or return empty results
    when the required submodule is unavailable.
    """

    @pytest.mark.contract
    @pytest.mark.xfail(
        strict=True,
        raises=AttributeError,
        reason=(
            "Source bug: __init__.py line 377 calls "
            "gt.list_available_transforms() but GraphTransforms exposes "
            "gt.get_available_transforms() — method name mismatch"
        ),
    )
    def test_get_available_transforms_returns_list(self, transformations_pkg):
        """
        ``get_available_transforms()`` returns a list (possibly empty if
        graph_transforms is unavailable).

        Known source bug: The module-level ``get_available_transforms()``
        delegates to ``GraphTransforms.list_available_transforms()`` which
        does not exist; the actual method is ``get_available_transforms()``.
        Marked ``xfail(strict=True)`` so CI will alert when the source
        bug is fixed (XPASS → FAIL).
        """
        result = transformations_pkg.get_available_transforms()
        assert isinstance(result, list), (
            f"get_available_transforms() should return list, got "
            f"{type(result).__name__}"
        )

    @pytest.mark.contract
    def test_create_transform_sequence_raises_when_unavailable(self, transformations_pkg):
        """
        ``create_transform_sequence()`` raises ImportError when
        graph_transforms is not available.
        """
        if transformations_pkg.GRAPH_TRANSFORMS_AVAILABLE:
            pytest.skip("graph_transforms is available — cannot test unavailable path")

        with pytest.raises(ImportError, match="graph_transforms"):
            transformations_pkg.create_transform_sequence([{"name": "AddSelfLoops"}])

    @pytest.mark.contract
    def test_validate_transform_config_raises_when_unavailable(self, transformations_pkg):
        """
        ``validate_transform_config()`` raises ImportError when
        graph_transforms is not available.
        """
        if transformations_pkg.GRAPH_TRANSFORMS_AVAILABLE:
            pytest.skip("graph_transforms is available — cannot test unavailable path")

        with pytest.raises(ImportError, match="graph_transforms"):
            transformations_pkg.validate_transform_config([{"name": "AddSelfLoops"}])

    @pytest.mark.contract
    def test_register_custom_transform_raises_when_unavailable(self, transformations_pkg):
        """
        ``register_custom_transform()`` raises ImportError when
        graph_transforms is not available.
        """
        if transformations_pkg.GRAPH_TRANSFORMS_AVAILABLE:
            pytest.skip("graph_transforms is available — cannot test unavailable path")

        with pytest.raises(ImportError, match="graph_transforms"):
            transformations_pkg.register_custom_transform(type("FakeTransform", (), {}))

    @pytest.mark.contract
    def test_discover_and_register_plugins_raises_when_unavailable(self, transformations_pkg):
        """
        ``discover_and_register_plugins()`` raises ImportError when
        plugin_system is not available.
        """
        if transformations_pkg.PLUGIN_SYSTEM_AVAILABLE:
            pytest.skip("plugin_system is available — cannot test unavailable path")

        with pytest.raises(ImportError, match="plugin_system"):
            transformations_pkg.discover_and_register_plugins(["/fake/path"])


class TestContractGetSystemStatusReturnType:
    """§2 — ``get_system_status()`` return type and structure contract."""

    @pytest.mark.contract
    def test_get_system_status_returns_dict(self, transformations_pkg):
        """``get_system_status()`` returns a dict."""
        result = transformations_pkg.get_system_status()
        assert isinstance(result, dict), (
            f"get_system_status() should return dict, got {type(result).__name__}"
        )

    @pytest.mark.contract
    def test_get_system_status_has_availability_keys(self, transformations_pkg):
        """
        ``get_system_status()`` result includes availability status keys
        for all four submodule systems.
        """
        result = transformations_pkg.get_system_status()
        expected_keys = [
            "graph_transforms_available",
            "custom_transforms_available",
            "plugin_system_available",
            "research_api_available",
            "initialized",
        ]
        for key in expected_keys:
            assert key in result, (
                f"get_system_status() missing expected key '{key}'. "
                f"Available keys: {sorted(result.keys())}"
            )

    @pytest.mark.contract
    def test_get_system_status_availability_values_are_bool(self, transformations_pkg):
        """
        The availability values in ``get_system_status()`` are booleans.
        """
        result = transformations_pkg.get_system_status()
        bool_keys = [
            "graph_transforms_available",
            "custom_transforms_available",
            "plugin_system_available",
            "research_api_available",
            "initialized",
        ]
        for key in bool_keys:
            assert isinstance(result[key], bool), (
                f"get_system_status()['{key}'] should be bool, "
                f"got {type(result[key]).__name__}"
            )

    @pytest.mark.contract
    def test_get_system_status_consistent_with_flags(self, transformations_pkg):
        """
        The availability values in ``get_system_status()`` match the
        module-level availability flags.
        """
        result = transformations_pkg.get_system_status()
        assert result["graph_transforms_available"] == transformations_pkg.GRAPH_TRANSFORMS_AVAILABLE
        assert result["custom_transforms_available"] == transformations_pkg.CUSTOM_TRANSFORMS_AVAILABLE
        assert result["plugin_system_available"] == transformations_pkg.PLUGIN_SYSTEM_AVAILABLE
        assert result["research_api_available"] == transformations_pkg.RESEARCH_API_AVAILABLE


class TestContractGetModuleInfoReturnType:
    """§2 — ``get_module_info()`` return type and structure contract."""

    @pytest.mark.contract
    def test_get_module_info_returns_dict(self, transformations_pkg):
        """``get_module_info()`` returns a dict."""
        result = transformations_pkg.get_module_info()
        assert isinstance(result, dict), (
            f"get_module_info() should return dict, got {type(result).__name__}"
        )

    @pytest.mark.contract
    def test_get_module_info_has_version(self, transformations_pkg):
        """``get_module_info()`` result includes a ``version`` key."""
        result = transformations_pkg.get_module_info()
        assert "version" in result, (
            f"get_module_info() missing 'version' key. "
            f"Available keys: {sorted(result.keys())}"
        )

    @pytest.mark.contract
    def test_get_module_info_version_matches_dunder(self, transformations_pkg):
        """``get_module_info()['version']`` matches ``__version__``."""
        result = transformations_pkg.get_module_info()
        assert result["version"] == transformations_pkg.__version__, (
            f"get_module_info()['version'] ({result['version']}) does not match "
            f"__version__ ({transformations_pkg.__version__})"
        )

    @pytest.mark.contract
    def test_get_module_info_has_author(self, transformations_pkg):
        """``get_module_info()`` result includes an ``author`` key."""
        result = transformations_pkg.get_module_info()
        assert "author" in result, (
            f"get_module_info() missing 'author' key. "
            f"Available keys: {sorted(result.keys())}"
        )

    @pytest.mark.contract
    def test_get_module_info_has_license(self, transformations_pkg):
        """``get_module_info()`` result includes a ``license`` key."""
        result = transformations_pkg.get_module_info()
        assert "license" in result, (
            f"get_module_info() missing 'license' key. "
            f"Available keys: {sorted(result.keys())}"
        )

    @pytest.mark.contract
    def test_get_module_info_has_features_dict(self, transformations_pkg):
        """``get_module_info()['features']`` is a dict of booleans."""
        result = transformations_pkg.get_module_info()
        assert "features" in result, (
            f"get_module_info() missing 'features' key. "
            f"Available keys: {sorted(result.keys())}"
        )
        features = result["features"]
        assert isinstance(features, dict), (
            f"get_module_info()['features'] should be dict, got "
            f"{type(features).__name__}"
        )
        expected_feature_keys = [
            "graph_transforms",
            "custom_transforms",
            "plugin_system",
            "research_api",
        ]
        for key in expected_feature_keys:
            assert key in features, (
                f"get_module_info()['features'] missing '{key}'. "
                f"Available keys: {sorted(features.keys())}"
            )
            assert isinstance(features[key], bool), (
                f"get_module_info()['features']['{key}'] should be bool, "
                f"got {type(features[key]).__name__}"
            )

    @pytest.mark.contract
    def test_get_module_info_has_components_dict(self, transformations_pkg):
        """``get_module_info()['components']`` is a dict of booleans."""
        result = transformations_pkg.get_module_info()
        assert "components" in result, (
            f"get_module_info() missing 'components' key. "
            f"Available keys: {sorted(result.keys())}"
        )
        components = result["components"]
        assert isinstance(components, dict), (
            f"get_module_info()['components'] should be dict, got "
            f"{type(components).__name__}"
        )
        expected_component_keys = [
            "transform_registry",
            "transform_composer",
            "plugin_registry",
            "experiment_configuration",
        ]
        for key in expected_component_keys:
            assert key in components, (
                f"get_module_info()['components'] missing '{key}'. "
                f"Available keys: {sorted(components.keys())}"
            )
            assert isinstance(components[key], bool), (
                f"get_module_info()['components']['{key}'] should be bool, "
                f"got {type(components[key]).__name__}"
            )

    @pytest.mark.contract
    def test_get_module_info_features_consistent_with_flags(self, transformations_pkg):
        """
        ``get_module_info()['features']`` values match the module-level
        availability flags.
        """
        result = transformations_pkg.get_module_info()
        features = result["features"]
        assert features["graph_transforms"] == transformations_pkg.GRAPH_TRANSFORMS_AVAILABLE
        assert features["custom_transforms"] == transformations_pkg.CUSTOM_TRANSFORMS_AVAILABLE
        assert features["plugin_system"] == transformations_pkg.PLUGIN_SYSTEM_AVAILABLE
        assert features["research_api"] == transformations_pkg.RESEARCH_API_AVAILABLE


class TestContractEnsureInitializedIdempotency:
    """§2 — ``_ensure_initialized()`` is idempotent and safe to call."""

    @pytest.mark.contract
    def test_ensure_initialized_is_function(self, transformations_pkg):
        """``_ensure_initialized`` is a function (not a class)."""
        assert inspect.isfunction(transformations_pkg._ensure_initialized)

    @pytest.mark.contract
    def test_ensure_initialized_accepts_no_args(self, transformations_pkg):
        """``_ensure_initialized`` takes no parameters."""
        sig = inspect.signature(transformations_pkg._ensure_initialized)
        required = [
            p for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        assert len(required) == 0, (
            f"_ensure_initialized should take no required args, "
            f"found: {[p.name for p in required]}"
        )

    @pytest.mark.contract
    @pytest.mark.xfail(
        strict=True,
        raises=NameError,
        reason=(
            "Source bug: custom_transforms._lazy_import_graph_transforms() "
            "references undefined global '_IMPORTING_GRAPH_TRANSFORMS'"
        ),
    )
    def test_ensure_initialized_returns_none(self, transformations_pkg):
        """
        ``_ensure_initialized()`` returns None (side-effect only).

        Known source bug: ``_ensure_initialized()`` triggers
        ``custom_transforms._lazy_import_graph_transforms()`` which
        references an undefined global ``_IMPORTING_GRAPH_TRANSFORMS``,
        raising ``NameError``.  Marked ``xfail(strict=True)`` so CI
        will alert when the source bug is fixed (XPASS → FAIL).
        """
        result = transformations_pkg._ensure_initialized()
        assert result is None, (
            f"_ensure_initialized() should return None, got {type(result).__name__}"
        )

    @pytest.mark.contract
    @pytest.mark.xfail(
        strict=True,
        raises=NameError,
        reason=(
            "Source bug: custom_transforms._lazy_import_graph_transforms() "
            "references undefined global '_IMPORTING_GRAPH_TRANSFORMS'"
        ),
    )
    def test_ensure_initialized_double_call_safe(self, transformations_pkg):
        """
        Calling ``_ensure_initialized()`` twice does not crash.

        Known source bug: raises ``NameError`` due to undefined
        ``_IMPORTING_GRAPH_TRANSFORMS`` in custom_transforms.py.
        Marked ``xfail(strict=True)`` so CI will alert when the source
        bug is fixed (XPASS → FAIL).
        """
        transformations_pkg._ensure_initialized()
        transformations_pkg._ensure_initialized()
        # No assertion needed — the test passes if no exception is raised


class TestContractGraphTransformConvenienceFunctionCallability:
    """§2 — Graph transform convenience function exports are callable when available."""

    GRAPH_CONVENIENCE_FUNCTIONS = [
        "get_transform_info",
        "validate_v3_configuration",
        "validate_comprehensive",
        "get_configuration_format_help",
        "export_metrics",
        "optimize_performance",
        "get_milia_setups",
        "perform_system_health_check",
        "get_validation_report_text",
        "discover_custom_transforms",
        "register_all_custom_transforms",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", GRAPH_CONVENIENCE_FUNCTIONS)
    def test_graph_convenience_is_callable_when_available(self, transformations_pkg, name):
        """Each graph transform convenience function is callable when available."""
        if not transformations_pkg.GRAPH_TRANSFORMS_AVAILABLE:
            pytest.skip("graph_transforms submodule not available")
        obj = getattr(transformations_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestContractPublicAPISurface:
    """
    §2 — Public API surface stability.

    Ensures that the minimum expected public API is present in ``__all__``.
    Guards against accidental removals during refactoring.
    """

    # The minimum API surface that MUST be present in __all__
    MINIMUM_API = {
        # Module metadata
        "__version__",
        "__author__",
        "__license__",

        # Availability flags
        "GRAPH_TRANSFORMS_AVAILABLE",
        "CUSTOM_TRANSFORMS_AVAILABLE",
        "PLUGIN_SYSTEM_AVAILABLE",
        "RESEARCH_API_AVAILABLE",

        # Graph transforms - Core classes
        "TransformRegistry",
        "TransformComposer",
        "TransformValidator",
        "DynamicTransformDiscovery",
        "ConfigurationBridge",
        "TransformErrorRecovery",
        "ValidationLevel",
        "ValidationScope",
        "TransformInfo",
        "TransformDependency",
        "TransformCompatibility",
        "GraphTransforms",

        # Graph transforms - Main API
        "get_graph_transforms",

        # Graph transforms - Convenience functions
        "get_transform_info",
        "validate_v3_configuration",
        "validate_comprehensive",
        "get_configuration_format_help",
        "export_metrics",
        "optimize_performance",
        "get_milia_setups",
        "perform_system_health_check",
        "get_validation_report_text",
        "discover_custom_transforms",
        "register_all_custom_transforms",

        # Custom transforms - Base classes
        "CustomTransformBase",
        "MolecularTransformBase",
        "QuantumTransformBase",
        "TransformMetadata",

        # Custom transforms - Example implementations
        "NormalizeVibrationalModes",
        "FilterByDMCUncertainty",
        "ScaleMullikenCharges",

        # Custom transforms - Exceptions
        "TransformValidationError",
        "TransformExecutionError",
        "TransformConfigurationError",

        # Plugin system - Core classes
        "PluginMetadata",
        "PluginRegistry",
        "PluginValidator",
        "TransformDeclaration",

        # Plugin system - Exceptions
        "PluginError",
        "PluginValidationError",
        "PluginSecurityError",
        "PluginDependencyError",

        # Research API - Configuration
        "ExperimentConfiguration",

        # Research API - Builders
        "AblationStudyBuilder",
        "ParameterSweepBuilder",
        "ComparativeStudyBuilder",

        # Research API - Execution
        "ExperimentRunner",

        # Research API - Convenience functions
        "create_ablation_study",
        "create_parameter_sweep",
        "create_comparative_study",
        "load_experiments_from_config",
        "get_experiment",
        "list_available_experiments",

        # Module-level convenience functions
        "get_available_transforms",
        "create_transform_sequence",
        "validate_transform_config",
        "register_custom_transform",
        "discover_and_register_plugins",
        "get_system_status",
        "get_module_info",

        # Initialization
        "_ensure_initialized",
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

        Based on the __init__.py source, the transformations package exports
        70+ names. This test guards against catastrophic loss (e.g., accidental
        truncation of __all__) while allowing for organic growth.
        """
        actual = len(all_names)
        # The __init__.py has ~70 entries in __all__ (lines 570-670)
        # We set a floor well below the actual count to allow changes
        # while catching catastrophic loss.
        MINIMUM_EXPECTED = 50
        assert actual >= MINIMUM_EXPECTED, (
            f"__all__ has {actual} entries, expected at least {MINIMUM_EXPECTED}. "
            f"This suggests __all__ may have been accidentally truncated."
        )


class TestContractAvailabilityFlagConsistency:
    """
    §2 — Feature availability flags are mutually consistent with
    the actual state of submodule exports.
    """

    @pytest.mark.contract
    def test_graph_transforms_flag_consistency(self, transformations_pkg):
        """
        When GRAPH_TRANSFORMS_AVAILABLE is True, ``TransformRegistry`` is
        not None. When False, it is None.
        """
        flag = transformations_pkg.GRAPH_TRANSFORMS_AVAILABLE
        obj = transformations_pkg.TransformRegistry

        if flag:
            assert obj is not None, (
                "GRAPH_TRANSFORMS_AVAILABLE is True but TransformRegistry is None"
            )
        else:
            assert obj is None, (
                "GRAPH_TRANSFORMS_AVAILABLE is False but TransformRegistry is not None"
            )

    @pytest.mark.contract
    def test_custom_transforms_flag_consistency(self, transformations_pkg):
        """
        When CUSTOM_TRANSFORMS_AVAILABLE is True, ``CustomTransformBase`` is
        not None. When False, it is None.
        """
        flag = transformations_pkg.CUSTOM_TRANSFORMS_AVAILABLE
        obj = transformations_pkg.CustomTransformBase

        if flag:
            assert obj is not None, (
                "CUSTOM_TRANSFORMS_AVAILABLE is True but CustomTransformBase is None"
            )
        else:
            assert obj is None, (
                "CUSTOM_TRANSFORMS_AVAILABLE is False but CustomTransformBase is not None"
            )

    @pytest.mark.contract
    def test_plugin_system_flag_consistency(self, transformations_pkg):
        """
        When PLUGIN_SYSTEM_AVAILABLE is True, ``PluginRegistry`` is
        not None. When False, it is None.
        """
        flag = transformations_pkg.PLUGIN_SYSTEM_AVAILABLE
        obj = transformations_pkg.PluginRegistry

        if flag:
            assert obj is not None, (
                "PLUGIN_SYSTEM_AVAILABLE is True but PluginRegistry is None"
            )
        else:
            assert obj is None, (
                "PLUGIN_SYSTEM_AVAILABLE is False but PluginRegistry is not None"
            )

    @pytest.mark.contract
    def test_research_api_flag_consistency(self, transformations_pkg):
        """
        When RESEARCH_API_AVAILABLE is True, ``ExperimentConfiguration`` is
        not None. When False, it is None.
        """
        flag = transformations_pkg.RESEARCH_API_AVAILABLE
        obj = transformations_pkg.ExperimentConfiguration

        if flag:
            assert obj is not None, (
                "RESEARCH_API_AVAILABLE is True but ExperimentConfiguration is None"
            )
        else:
            assert obj is None, (
                "RESEARCH_API_AVAILABLE is False but ExperimentConfiguration is not None"
            )


class TestContractNullFallbackCompleteness:
    """
    §2 — When a submodule is unavailable, all its exports are set to
    the documented fallback values (None for classes, Exception for exceptions).
    """

    @pytest.mark.contract
    def test_graph_transforms_null_fallbacks_complete(self, transformations_pkg):
        """
        When GRAPH_TRANSFORMS_AVAILABLE is False, all graph transform
        class exports are None.
        """
        if transformations_pkg.GRAPH_TRANSFORMS_AVAILABLE:
            pytest.skip("graph_transforms is available — cannot test fallbacks")

        null_expected = [
            "TransformRegistry", "TransformComposer", "TransformValidator",
            "DynamicTransformDiscovery", "ConfigurationBridge",
            "TransformErrorRecovery", "ValidationLevel", "ValidationScope",
            "TransformInfo", "TransformDependency", "TransformCompatibility",
            "GraphTransforms", "get_graph_transforms", "get_transform_info",
            "validate_v3_configuration", "validate_comprehensive",
            "get_configuration_format_help", "export_metrics",
            "optimize_performance", "get_milia_setups",
            "perform_system_health_check", "get_validation_report_text",
            "discover_custom_transforms", "register_all_custom_transforms",
        ]
        for name in null_expected:
            obj = getattr(transformations_pkg, name)
            assert obj is None, (
                f"'{name}' should be None when GRAPH_TRANSFORMS_AVAILABLE is False, "
                f"got {type(obj).__name__}"
            )

    @pytest.mark.contract
    def test_custom_transforms_null_fallbacks_complete(self, transformations_pkg):
        """
        When CUSTOM_TRANSFORMS_AVAILABLE is False, all custom transform
        class exports are None and exception exports are Exception.
        """
        if transformations_pkg.CUSTOM_TRANSFORMS_AVAILABLE:
            pytest.skip("custom_transforms is available — cannot test fallbacks")

        null_expected = [
            "CustomTransformBase", "MolecularTransformBase",
            "QuantumTransformBase", "TransformMetadata",
            "NormalizeVibrationalModes", "FilterByDMCUncertainty",
            "ScaleMullikenCharges",
        ]
        for name in null_expected:
            obj = getattr(transformations_pkg, name)
            assert obj is None, (
                f"'{name}' should be None when CUSTOM_TRANSFORMS_AVAILABLE is False, "
                f"got {type(obj).__name__}"
            )

        exception_expected = [
            "TransformValidationError",
            "TransformExecutionError",
            "TransformConfigurationError",
        ]
        for name in exception_expected:
            obj = getattr(transformations_pkg, name)
            assert obj is Exception, (
                f"'{name}' should be Exception when CUSTOM_TRANSFORMS_AVAILABLE is False, "
                f"got {obj}"
            )

    @pytest.mark.contract
    def test_plugin_system_null_fallbacks_complete(self, transformations_pkg):
        """
        When PLUGIN_SYSTEM_AVAILABLE is False, all plugin system class
        exports are None and exception exports are Exception.
        """
        if transformations_pkg.PLUGIN_SYSTEM_AVAILABLE:
            pytest.skip("plugin_system is available — cannot test fallbacks")

        null_expected = [
            "PluginMetadata", "PluginRegistry",
            "PluginValidator", "TransformDeclaration",
        ]
        for name in null_expected:
            obj = getattr(transformations_pkg, name)
            assert obj is None, (
                f"'{name}' should be None when PLUGIN_SYSTEM_AVAILABLE is False, "
                f"got {type(obj).__name__}"
            )

        exception_expected = [
            "PluginError", "PluginValidationError",
            "PluginSecurityError", "PluginDependencyError",
        ]
        for name in exception_expected:
            obj = getattr(transformations_pkg, name)
            assert obj is Exception, (
                f"'{name}' should be Exception when PLUGIN_SYSTEM_AVAILABLE is False, "
                f"got {obj}"
            )

    @pytest.mark.contract
    def test_research_api_null_fallbacks_complete(self, transformations_pkg):
        """
        When RESEARCH_API_AVAILABLE is False, all research API exports
        are None.
        """
        if transformations_pkg.RESEARCH_API_AVAILABLE:
            pytest.skip("research_api is available — cannot test fallbacks")

        null_expected = [
            "ExperimentConfiguration", "AblationStudyBuilder",
            "ParameterSweepBuilder", "ComparativeStudyBuilder",
            "ExperimentRunner", "create_ablation_study",
            "create_parameter_sweep", "create_comparative_study",
            "load_experiments_from_config", "get_experiment",
            "list_available_experiments",
        ]
        for name in null_expected:
            obj = getattr(transformations_pkg, name)
            assert obj is None, (
                f"'{name}' should be None when RESEARCH_API_AVAILABLE is False, "
                f"got {type(obj).__name__}"
            )


class TestContractVersionMetadata:
    """§2 — Version and metadata consistency."""

    @pytest.mark.contract
    def test_version_matches_module_info(self, transformations_pkg):
        """``__version__`` matches ``get_module_info()['version']``."""
        info = transformations_pkg.get_module_info()
        assert transformations_pkg.__version__ == info["version"]

    @pytest.mark.contract
    def test_author_matches_module_info(self, transformations_pkg):
        """``__author__`` matches ``get_module_info()['author']``."""
        info = transformations_pkg.get_module_info()
        assert transformations_pkg.__author__ == info["author"]

    @pytest.mark.contract
    def test_license_matches_module_info(self, transformations_pkg):
        """``__license__`` matches ``get_module_info()['license']``."""
        info = transformations_pkg.get_module_info()
        assert transformations_pkg.__license__ == info["license"]

    @pytest.mark.contract
    def test_version_is_1_0_0(self, transformations_pkg):
        """
        ``__version__`` is ``'1.0.0'`` as documented in the module source
        (line 88 of __init__.py).
        """
        assert transformations_pkg.__version__ == "1.0.0", (
            f"Expected version '1.0.0', got '{transformations_pkg.__version__}'"
        )

    @pytest.mark.contract
    def test_license_is_mit(self, transformations_pkg):
        """
        ``__license__`` is ``'MIT'`` as documented in the module source
        (line 90 of __init__.py).
        """
        assert transformations_pkg.__license__ == "MIT", (
            f"Expected license 'MIT', got '{transformations_pkg.__license__}'"
        )
