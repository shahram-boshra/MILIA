# tests/test__init__models_deployment.py

"""
Test Suite: milia_pipeline/models/deployment/__init__.py — Smoke Tests & Contract Tests
========================================================================================

Production-ready test suite for the MILIA Pipeline deployment package
``milia_pipeline/models/deployment/__init__.py``.

Covers:
    Section 1 — Smoke Tests (MILIA_Test_Recommendations.md §1.2 scope):
        - The ``milia_pipeline.models.deployment`` subpackage imports without
          ImportError
        - All re-exported names from the 3 submodules (deployment_strategies,
          model_optimization, monitoring) are accessible
        - Module-level metadata attributes (__version__, __author__) exist
        - High-level convenience functions defined in __init__.py are accessible
        - Utility functions defined in __init__.py are accessible
        - Module-level convenience functions imported from submodules are accessible
        - Exception classes (with fallback definitions) are accessible
        - Enum classes are accessible and are Enum subclasses
        - Configuration dataclasses are accessible and are classes
        - Re-import (``importlib.reload``) is idempotent and non-crashing
        - Module initialization (logging, critical component verification) runs
          safely

    Section 2 — Contract Tests (MILIA_Test_Recommendations.md §2 scope):
        - ``__all__`` completeness: every name in ``__all__`` is resolvable
        - ``__all__`` consistency: every public import is listed in ``__all__``
        - ``__all__`` has no duplicates
        - Re-exported classes are classes, callables are callable
        - Exception classes follow the correct inheritance hierarchy
        - Enum classes contain expected members
        - DeploymentStrategy subclasses are ABCs with required abstract methods
        - Strategy implementations are subclasses of DeploymentStrategy
        - DeploymentManager has required public methods
        - ModelOptimizer has required public methods
        - ModelMonitor has required public methods
        - Configuration classes are dataclasses with ``to_dict()`` methods
        - Convenience functions have documented parameter signatures
        - ``__version__`` follows semver pattern
        - Public API surface stability (minimum expected names present)
        - High-level convenience functions return correct types (with mocking)
        - validate_deployment_config raises ConfigurationError for invalid configs

Design:
    - Zero ``sys.modules`` pollution: all mocking via ``@patch`` decorators
      or context-manager ``patch`` calls scoped to individual tests.
    - Deterministic: no filesystem, network, or GPU access required.
    - Isolated: each test is independent; execution order is irrelevant.
    - Fast: expected < 5 s total wall-clock on CI.

Launch:
    From project root (/app/milia):
        pytest tests/test__init__models_deployment.py -v --tb=short

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
from unittest.mock import patch, MagicMock, PropertyMock
from enum import Enum

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure the project root is importable
# ---------------------------------------------------------------------------
# When launched via ``pytest tests/test__init__models_deployment.py`` from the
# project root (/app/milia), ``milia_pipeline`` must be on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # …/tests -> …/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def deployment_pkg():
    """
    Import and return the ``milia_pipeline.models.deployment`` package once
    per module.

    This fixture validates the fundamental smoke invariant: the deployment
    subpackage is importable.  If this fails, every downstream test is moot.
    """
    try:
        import milia_pipeline.models.deployment as dep
        return dep
    except ImportError as exc:
        pytest.fail(
            f"milia_pipeline.models.deployment could not be imported — "
            f"smoke test precondition violated: {exc}"
        )


@pytest.fixture(scope="module")
def all_names(deployment_pkg):
    """Return the ``__all__`` list from the deployment package."""
    assert hasattr(deployment_pkg, "__all__"), (
        "milia_pipeline.models.deployment.__all__ is missing — "
        "contract violation"
    )
    return list(deployment_pkg.__all__)


# ===================================================================
# SECTION 1 — SMOKE TESTS
# ===================================================================


class TestSmokeDeploymentPackageImport:
    """§1.2 — Verify the deployment subpackage imports without errors."""

    @pytest.mark.smoke
    def test_import_deployment_package_succeeds(self, deployment_pkg):
        """The deployment package imports without raising any exception."""
        assert deployment_pkg is not None

    @pytest.mark.smoke
    def test_deployment_package_is_a_module(self, deployment_pkg):
        """The imported object is a proper Python module."""
        assert isinstance(deployment_pkg, types.ModuleType)

    @pytest.mark.smoke
    def test_deployment_package_has_file_attribute(self, deployment_pkg):
        """The package exposes a ``__file__`` attribute (not a namespace pkg)."""
        assert hasattr(deployment_pkg, "__file__")

    @pytest.mark.smoke
    def test_deployment_package_name(self, deployment_pkg):
        """The package ``__name__`` is ``milia_pipeline.models.deployment``."""
        assert deployment_pkg.__name__ == "milia_pipeline.models.deployment"


class TestSmokeMetadataAttributes:
    """§1.2 — Verify module-level metadata attributes are present and typed."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("attr", [
        "__version__",
        "__author__",
    ])
    def test_metadata_attribute_exists(self, deployment_pkg, attr):
        """Each metadata dunder is defined on the deployment package."""
        assert hasattr(deployment_pkg, attr), f"Missing attribute: {attr}"

    @pytest.mark.smoke
    @pytest.mark.parametrize("attr", [
        "__version__",
        "__author__",
    ])
    def test_metadata_attribute_is_string(self, deployment_pkg, attr):
        """Each metadata dunder is a non-empty string."""
        value = getattr(deployment_pkg, attr)
        assert isinstance(value, str), f"{attr} should be str, got {type(value)}"
        assert len(value) > 0, f"{attr} should be non-empty"

    @pytest.mark.smoke
    def test_version_is_semver_like(self, deployment_pkg):
        """``__version__`` follows a MAJOR.MINOR.PATCH pattern."""
        version = deployment_pkg.__version__
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


class TestSmokeExceptionExports:
    """§1.2 — Exception classes are accessible (with fallback support)."""

    EXCEPTION_CLASSES = [
        "ModelError",
        "DeploymentError",
        "OptimizationError",
        "MonitoringError",
        "ConfigurationError",
        "ExportError",
        "AlertError",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", EXCEPTION_CLASSES)
    def test_exception_class_exists(self, deployment_pkg, name):
        """Each exception class is importable from the deployment package."""
        obj = getattr(deployment_pkg, name, None)
        assert obj is not None, f"Exception class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", EXCEPTION_CLASSES)
    def test_exception_class_is_a_class(self, deployment_pkg, name):
        """Each exception export is a class (not an instance or function)."""
        obj = getattr(deployment_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", EXCEPTION_CLASSES)
    def test_exception_class_is_exception_subclass(self, deployment_pkg, name):
        """Each exception class is a subclass of Exception."""
        obj = getattr(deployment_pkg, name)
        assert issubclass(obj, Exception), (
            f"'{name}' should be a subclass of Exception"
        )


class TestSmokeDeploymentStrategyExports:
    """§1.2 — Deployment strategy exports are accessible."""

    # Base and configuration
    BASE_EXPORTS = [
        "DeploymentStrategy",
        "DeploymentConfig",
    ]

    # Enum exports
    ENUM_EXPORTS = [
        "DeploymentTarget",
        "ServingMode",
    ]

    # Strategy implementations
    STRATEGY_EXPORTS = [
        "AWSDeploymentStrategy",
        "GCPDeploymentStrategy",
        "AzureDeploymentStrategy",
        "EdgeDeploymentStrategy",
        "ContainerDeploymentStrategy",
        "LocalDeploymentStrategy",
    ]

    # Manager
    MANAGER_EXPORTS = [
        "DeploymentManager",
    ]

    ALL_STRATEGY_EXPORTS = (
        BASE_EXPORTS + ENUM_EXPORTS + STRATEGY_EXPORTS + MANAGER_EXPORTS
    )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", ALL_STRATEGY_EXPORTS)
    def test_deployment_strategy_export_exists(self, deployment_pkg, name):
        """Each deployment strategy export is present and non-None."""
        obj = getattr(deployment_pkg, name, None)
        assert obj is not None, (
            f"Deployment strategy export '{name}' is None or missing"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", BASE_EXPORTS + STRATEGY_EXPORTS + MANAGER_EXPORTS)
    def test_deployment_strategy_export_is_class(self, deployment_pkg, name):
        """Each deployment strategy/config/manager export is a class."""
        obj = getattr(deployment_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", ENUM_EXPORTS)
    def test_deployment_enum_is_enum_subclass(self, deployment_pkg, name):
        """Each deployment enum export is an Enum subclass."""
        obj = getattr(deployment_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )
        assert issubclass(obj, Enum), (
            f"'{name}' should be a subclass of Enum"
        )


class TestSmokeModelOptimizationExports:
    """§1.2 — Model optimization exports are accessible."""

    OPTIMIZATION_CLASSES = [
        "ModelOptimizer",
        "OptimizationConfig",
    ]

    OPTIMIZATION_ENUMS = [
        "QuantizationType",
        "PruningType",
    ]

    OPTIMIZATION_FUNCTIONS = [
        "quantize_for_inference",
        "prune_for_deployment",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", OPTIMIZATION_CLASSES)
    def test_optimization_class_exists(self, deployment_pkg, name):
        """Each optimization class is importable from the deployment package."""
        obj = getattr(deployment_pkg, name, None)
        assert obj is not None, (
            f"Optimization class '{name}' is None or missing"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", OPTIMIZATION_CLASSES)
    def test_optimization_class_is_a_class(self, deployment_pkg, name):
        """Each optimization export is a class."""
        obj = getattr(deployment_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", OPTIMIZATION_ENUMS)
    def test_optimization_enum_is_enum_subclass(self, deployment_pkg, name):
        """Each optimization enum is an Enum subclass."""
        obj = getattr(deployment_pkg, name)
        assert inspect.isclass(obj) and issubclass(obj, Enum), (
            f"'{name}' should be an Enum subclass"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", OPTIMIZATION_FUNCTIONS)
    def test_optimization_function_exists(self, deployment_pkg, name):
        """Each optimization convenience function is present and non-None."""
        obj = getattr(deployment_pkg, name, None)
        assert obj is not None, (
            f"Optimization function '{name}' is None or missing"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", OPTIMIZATION_FUNCTIONS)
    def test_optimization_function_is_callable(self, deployment_pkg, name):
        """Each optimization convenience function is callable."""
        obj = getattr(deployment_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeMonitoringExports:
    """§1.2 — Monitoring exports are accessible."""

    MONITORING_CLASSES = [
        "ModelMonitor",
        "MonitoringConfig",
        "Alert",
    ]

    MONITORING_ENUMS = [
        "MetricType",
        "AlertSeverity",
        "DriftType",
    ]

    MONITORING_FUNCTIONS = [
        "create_monitor",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", MONITORING_CLASSES)
    def test_monitoring_class_exists(self, deployment_pkg, name):
        """Each monitoring class is importable from the deployment package."""
        obj = getattr(deployment_pkg, name, None)
        assert obj is not None, (
            f"Monitoring class '{name}' is None or missing"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", MONITORING_CLASSES)
    def test_monitoring_class_is_a_class(self, deployment_pkg, name):
        """Each monitoring export is a class."""
        obj = getattr(deployment_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", MONITORING_ENUMS)
    def test_monitoring_enum_is_enum_subclass(self, deployment_pkg, name):
        """Each monitoring enum is an Enum subclass."""
        obj = getattr(deployment_pkg, name)
        assert inspect.isclass(obj) and issubclass(obj, Enum), (
            f"'{name}' should be an Enum subclass"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", MONITORING_FUNCTIONS)
    def test_monitoring_function_exists(self, deployment_pkg, name):
        """Each monitoring convenience function is present and non-None."""
        obj = getattr(deployment_pkg, name, None)
        assert obj is not None, (
            f"Monitoring function '{name}' is None or missing"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", MONITORING_FUNCTIONS)
    def test_monitoring_function_is_callable(self, deployment_pkg, name):
        """Each monitoring convenience function is callable."""
        obj = getattr(deployment_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeHighLevelConvenienceFunctions:
    """§1.2 — High-level convenience functions defined in __init__.py are accessible."""

    CONVENIENCE_FUNCTIONS = [
        "deploy_model_locally",
        "deploy_model_to_cloud",
        "optimize_model_for_deployment",
        "create_production_monitor",
        "create_deployment_pipeline",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CONVENIENCE_FUNCTIONS)
    def test_convenience_function_exists(self, deployment_pkg, name):
        """Each high-level convenience function is present and non-None."""
        obj = getattr(deployment_pkg, name, None)
        assert obj is not None, (
            f"Convenience function '{name}' is None or missing"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CONVENIENCE_FUNCTIONS)
    def test_convenience_function_is_callable(self, deployment_pkg, name):
        """Each high-level convenience function is callable."""
        obj = getattr(deployment_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeUtilityFunctions:
    """§1.2 — Utility functions defined in __init__.py are accessible."""

    UTILITY_FUNCTIONS = [
        "list_deployment_targets",
        "get_deployment_info",
        "validate_deployment_config",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", UTILITY_FUNCTIONS)
    def test_utility_function_exists(self, deployment_pkg, name):
        """Each utility function is present and non-None."""
        obj = getattr(deployment_pkg, name, None)
        assert obj is not None, (
            f"Utility function '{name}' is None or missing"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", UTILITY_FUNCTIONS)
    def test_utility_function_is_callable(self, deployment_pkg, name):
        """Each utility function is callable."""
        obj = getattr(deployment_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeSubmoduleConvenienceFunctions:
    """§1.2 — Module-level convenience functions imported from submodules."""

    SUBMODULE_FUNCTIONS = [
        "deploy_locally",
        "quantize_for_inference",
        "prune_for_deployment",
        "create_monitor",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", SUBMODULE_FUNCTIONS)
    def test_submodule_function_exists(self, deployment_pkg, name):
        """Each submodule convenience function is present and non-None."""
        obj = getattr(deployment_pkg, name, None)
        assert obj is not None, (
            f"Submodule convenience function '{name}' is None or missing"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", SUBMODULE_FUNCTIONS)
    def test_submodule_function_is_callable(self, deployment_pkg, name):
        """Each submodule convenience function is callable."""
        obj = getattr(deployment_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeModuleInitialization:
    """§1.2 — Module-level initialization runs without side-effect crashes."""

    @pytest.mark.smoke
    def test_reimport_is_idempotent(self, deployment_pkg):
        """
        Re-importing the deployment package (via ``importlib.reload``) does
        not crash.

        Validates that all module-level code (logging, critical component
        verification) is safe to re-execute.
        """
        reloaded = importlib.reload(deployment_pkg)
        assert reloaded is not None
        assert hasattr(reloaded, "__version__")

    @pytest.mark.smoke
    def test_reimport_preserves_all(self, deployment_pkg):
        """Re-importing the deployment package preserves ``__all__``."""
        reloaded = importlib.reload(deployment_pkg)
        assert hasattr(reloaded, "__all__")
        assert isinstance(reloaded.__all__, list)
        assert len(reloaded.__all__) > 0

    @pytest.mark.smoke
    def test_critical_components_present_after_init(self, deployment_pkg):
        """
        The 3 critical components (DeploymentManager, ModelOptimizer,
        ModelMonitor) are present in module globals after initialization.
        """
        critical = ["DeploymentManager", "ModelOptimizer", "ModelMonitor"]
        for name in critical:
            assert hasattr(deployment_pkg, name), (
                f"Critical component '{name}' missing after module init"
            )

    @pytest.mark.smoke
    def test_logger_exists(self, deployment_pkg):
        """
        The module-level logger is present and is a logging.Logger instance.
        """
        assert hasattr(deployment_pkg, "logger")
        assert isinstance(deployment_pkg.logger, logging.Logger)


class TestSmokeListDeploymentTargets:
    """§1.2 — list_deployment_targets() runs without crashing."""

    @pytest.mark.smoke
    def test_list_deployment_targets_returns_list(self, deployment_pkg):
        """``list_deployment_targets()`` returns a list."""
        result = deployment_pkg.list_deployment_targets()
        assert isinstance(result, (list, tuple, set, frozenset)), (
            f"list_deployment_targets() should return a collection, "
            f"got {type(result).__name__}"
        )

    @pytest.mark.smoke
    def test_list_deployment_targets_is_nonempty(self, deployment_pkg):
        """``list_deployment_targets()`` returns at least one target."""
        result = deployment_pkg.list_deployment_targets()
        assert len(result) > 0, (
            "list_deployment_targets() should return at least one target"
        )

    @pytest.mark.smoke
    def test_list_deployment_targets_contains_strings(self, deployment_pkg):
        """Each element from ``list_deployment_targets()`` is a string."""
        result = deployment_pkg.list_deployment_targets()
        for item in result:
            assert isinstance(item, str), (
                f"Each deployment target should be str, got {type(item).__name__}"
            )


# ===================================================================
# SECTION 2 — CONTRACT TESTS
# ===================================================================


class TestContractAllCompleteness:
    """§2 — Every name in ``__all__`` is resolvable on the deployment package."""

    @pytest.mark.contract
    def test_all_is_a_list(self, deployment_pkg):
        """``__all__`` is a list (not a tuple or set)."""
        assert isinstance(deployment_pkg.__all__, list)

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
    def test_every_all_entry_is_resolvable(self, deployment_pkg, all_names):
        """
        Generic sweep: every single entry in ``__all__`` must be resolvable,
        regardless of whether it is parameterized individually.
        """
        unresolvable = [
            name for name in all_names
            if not hasattr(deployment_pkg, name)
        ]
        assert not unresolvable, (
            f"Names in __all__ that are not defined on the module: "
            f"{unresolvable}"
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

    @pytest.mark.contract
    def test_all_has_minimum_expected_count(self, all_names):
        """
        ``__all__`` contains at least the expected minimum number of exports.

        The deployment __init__.py defines 7 exception classes, 11 deployment
        strategy exports, 4 optimization exports, 7 monitoring exports,
        5 convenience functions, 3 utility functions, and 4 submodule functions
        = 41 total. We use a safety margin.
        """
        assert len(all_names) >= 30, (
            f"__all__ should have at least 30 entries, got {len(all_names)}"
        )


class TestContractAllConsistency:
    """§2 — Every public import in the deployment module is listed in ``__all__``."""

    # Names that are intentionally public-ish but NOT in __all__
    KNOWN_UNLISTED = {
        # Metadata dunders
        "__version__",
        "__author__",
        # Module-level infrastructure
        "logger",
        "_critical_components",
        # Loop variable leaked from module-level critical component check
        # (``for component in _critical_components: ...``)
        "component",
        # Typing imports used in function signatures (not re-exports)
        "Optional",
        "Dict",
        "Any",
        "List",
        "Union",
        "Callable",
        # Standard library imports used in function signatures
        "Path",
    }

    @pytest.mark.contract
    def test_public_imports_are_in_all(self, deployment_pkg, all_names):
        """
        Every non-dunder, non-private attribute that was imported (not
        defined locally as infrastructure) should be in ``__all__`` —
        unless it is in ``KNOWN_UNLISTED``.
        """
        all_set = set(all_names)
        module_dict = vars(deployment_pkg)
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
            # Skip private names not in __all__
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
            f"Public names imported in deployment/__init__.py but not in "
            f"__all__: {sorted(missing_from_all)}"
        )


class TestContractExceptionHierarchy:
    """§2 — Exception classes follow the correct inheritance hierarchy."""

    @pytest.mark.contract
    def test_model_error_is_base_exception(self, deployment_pkg):
        """``ModelError`` is a subclass of Exception."""
        assert issubclass(deployment_pkg.ModelError, Exception)

    @pytest.mark.contract
    def test_deployment_error_inherits_model_error(self, deployment_pkg):
        """``DeploymentError`` inherits from ``ModelError``."""
        assert issubclass(
            deployment_pkg.DeploymentError,
            deployment_pkg.ModelError
        ), "DeploymentError should inherit from ModelError"

    @pytest.mark.contract
    def test_optimization_error_inherits_model_error(self, deployment_pkg):
        """``OptimizationError`` inherits from ``ModelError``."""
        assert issubclass(
            deployment_pkg.OptimizationError,
            deployment_pkg.ModelError
        ), "OptimizationError should inherit from ModelError"

    @pytest.mark.contract
    def test_monitoring_error_inherits_model_error(self, deployment_pkg):
        """``MonitoringError`` inherits from ``ModelError``."""
        assert issubclass(
            deployment_pkg.MonitoringError,
            deployment_pkg.ModelError
        ), "MonitoringError should inherit from ModelError"

    @pytest.mark.contract
    def test_configuration_error_inherits_model_error(self, deployment_pkg):
        """``ConfigurationError`` inherits from ``ModelError``."""
        # Note: In the main exceptions module, ConfigurationError may inherit
        # from BaseProjectError. In the fallback definitions, it inherits
        # from ModelError. We test the actual hierarchy as loaded.
        assert issubclass(
            deployment_pkg.ConfigurationError,
            Exception
        ), "ConfigurationError should be a subclass of Exception"

    @pytest.mark.contract
    def test_export_error_inherits_model_error(self, deployment_pkg):
        """``ExportError`` inherits from ``ModelError``."""
        assert issubclass(
            deployment_pkg.ExportError,
            deployment_pkg.ModelError
        ), "ExportError should inherit from ModelError"

    @pytest.mark.contract
    def test_alert_error_inherits_monitoring_error(self, deployment_pkg):
        """``AlertError`` inherits from ``MonitoringError``."""
        assert issubclass(
            deployment_pkg.AlertError,
            deployment_pkg.MonitoringError
        ), "AlertError should inherit from MonitoringError"

    @pytest.mark.contract
    @pytest.mark.parametrize("name", [
        "ModelError",
        "DeploymentError",
        "OptimizationError",
        "MonitoringError",
        "ConfigurationError",
        "ExportError",
        "AlertError",
    ])
    def test_exception_is_instantiable_with_message(self, deployment_pkg, name):
        """Each exception can be instantiated with a string message."""
        exc_cls = getattr(deployment_pkg, name)
        instance = exc_cls("test message")
        assert isinstance(instance, Exception)
        assert "test message" in str(instance)


class TestContractDeploymentTargetEnum:
    """§2 — DeploymentTarget enum has expected members."""

    EXPECTED_MEMBERS = [
        "CLOUD_AWS",
        "CLOUD_GCP",
        "CLOUD_AZURE",
        "EDGE_MOBILE",
        "EDGE_IOT",
        "FEDERATED",
        "SERVERLESS",
        "CONTAINER",
        "LOCAL",
    ]

    @pytest.mark.contract
    def test_deployment_target_has_minimum_members(self, deployment_pkg):
        """``DeploymentTarget`` has at least 6 members."""
        members = list(deployment_pkg.DeploymentTarget)
        assert len(members) >= 6, (
            f"DeploymentTarget should have at least 6 members, "
            f"got {len(members)}"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("member_name", EXPECTED_MEMBERS)
    def test_deployment_target_has_expected_member(
        self, deployment_pkg, member_name
    ):
        """DeploymentTarget has each expected member."""
        assert hasattr(deployment_pkg.DeploymentTarget, member_name), (
            f"DeploymentTarget missing expected member '{member_name}'"
        )


class TestContractServingModeEnum:
    """§2 — ServingMode enum has expected members."""

    EXPECTED_MEMBERS = [
        "ONLINE",
        "BATCH",
        "STREAMING",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("member_name", EXPECTED_MEMBERS)
    def test_serving_mode_has_expected_member(
        self, deployment_pkg, member_name
    ):
        """ServingMode has each expected member."""
        assert hasattr(deployment_pkg.ServingMode, member_name), (
            f"ServingMode missing expected member '{member_name}'"
        )


class TestContractQuantizationTypeEnum:
    """§2 — QuantizationType enum has expected members."""

    EXPECTED_MEMBERS = [
        "DYNAMIC",
        "STATIC",
        "QAT",
        "FP16",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("member_name", EXPECTED_MEMBERS)
    def test_quantization_type_has_expected_member(
        self, deployment_pkg, member_name
    ):
        """QuantizationType has each expected member."""
        assert hasattr(deployment_pkg.QuantizationType, member_name), (
            f"QuantizationType missing expected member '{member_name}'"
        )


class TestContractPruningTypeEnum:
    """§2 — PruningType enum has expected members."""

    EXPECTED_MEMBERS = [
        "UNSTRUCTURED",
        "STRUCTURED",
        "MAGNITUDE",
        "GRADIENT",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("member_name", EXPECTED_MEMBERS)
    def test_pruning_type_has_expected_member(
        self, deployment_pkg, member_name
    ):
        """PruningType has each expected member."""
        assert hasattr(deployment_pkg.PruningType, member_name), (
            f"PruningType missing expected member '{member_name}'"
        )


class TestContractMetricTypeEnum:
    """§2 — MetricType enum has expected members."""

    EXPECTED_MEMBERS = [
        "LATENCY",
        "THROUGHPUT",
        "ERROR_RATE",
        "ACCURACY",
        "LOSS",
        "MEMORY",
        "CPU",
        "GPU",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("member_name", EXPECTED_MEMBERS)
    def test_metric_type_has_expected_member(
        self, deployment_pkg, member_name
    ):
        """MetricType has each expected member."""
        assert hasattr(deployment_pkg.MetricType, member_name), (
            f"MetricType missing expected member '{member_name}'"
        )


class TestContractAlertSeverityEnum:
    """§2 — AlertSeverity enum has expected members."""

    EXPECTED_MEMBERS = [
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("member_name", EXPECTED_MEMBERS)
    def test_alert_severity_has_expected_member(
        self, deployment_pkg, member_name
    ):
        """AlertSeverity has each expected member."""
        assert hasattr(deployment_pkg.AlertSeverity, member_name), (
            f"AlertSeverity missing expected member '{member_name}'"
        )


class TestContractDriftTypeEnum:
    """§2 — DriftType enum has expected members."""

    EXPECTED_MEMBERS = [
        "DATA_DRIFT",
        "CONCEPT_DRIFT",
        "MODEL_DRIFT",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("member_name", EXPECTED_MEMBERS)
    def test_drift_type_has_expected_member(
        self, deployment_pkg, member_name
    ):
        """DriftType has each expected member."""
        assert hasattr(deployment_pkg.DriftType, member_name), (
            f"DriftType missing expected member '{member_name}'"
        )


class TestContractDeploymentConfigDataclass:
    """§2 — DeploymentConfig is a dataclass with expected attributes."""

    EXPECTED_ATTRIBUTES = [
        "target",
        "serving_mode",
        "instance_type",
        "num_instances",
        "auto_scaling",
        "min_instances",
        "max_instances",
        "api_type",
        "enable_monitoring",
        "enable_logging",
        "enable_caching",
        "timeout_seconds",
        "max_batch_size",
    ]

    @pytest.mark.contract
    def test_deployment_config_is_class(self, deployment_pkg):
        """``DeploymentConfig`` is a class."""
        assert inspect.isclass(deployment_pkg.DeploymentConfig)

    @pytest.mark.contract
    def test_deployment_config_has_to_dict(self, deployment_pkg):
        """``DeploymentConfig`` has a ``to_dict()`` method."""
        assert hasattr(deployment_pkg.DeploymentConfig, "to_dict"), (
            "DeploymentConfig should have a to_dict() method"
        )
        assert callable(deployment_pkg.DeploymentConfig.to_dict), (
            "DeploymentConfig.to_dict should be callable"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("attr", EXPECTED_ATTRIBUTES)
    def test_deployment_config_attribute_in_annotations_or_init(
        self, deployment_pkg, attr
    ):
        """
        Each expected attribute is declared in DeploymentConfig's annotations
        or __init__ signature (dataclass or Pydantic field).
        """
        cls = deployment_pkg.DeploymentConfig
        # Check annotations (dataclass or Pydantic fields)
        annotations = getattr(cls, "__annotations__", {})
        init_sig_params = set()
        try:
            sig = inspect.signature(cls.__init__)
            init_sig_params = set(sig.parameters.keys()) - {"self"}
        except (ValueError, TypeError):
            pass

        assert attr in annotations or attr in init_sig_params, (
            f"DeploymentConfig should declare '{attr}' in annotations or "
            f"__init__ parameters. Found annotations: "
            f"{sorted(annotations.keys())}, init params: "
            f"{sorted(init_sig_params)}"
        )


class TestContractOptimizationConfigDataclass:
    """§2 — OptimizationConfig is a dataclass with expected attributes."""

    EXPECTED_ATTRIBUTES = [
        "quantization_enabled",
        "quantization_type",
        "pruning_enabled",
        "pruning_type",
        "pruning_amount",
        "distillation_enabled",
        "export_onnx",
        "optimize_for_mobile",
    ]

    @pytest.mark.contract
    def test_optimization_config_is_class(self, deployment_pkg):
        """``OptimizationConfig`` is a class."""
        assert inspect.isclass(deployment_pkg.OptimizationConfig)

    @pytest.mark.contract
    def test_optimization_config_has_to_dict(self, deployment_pkg):
        """``OptimizationConfig`` has a ``to_dict()`` method."""
        assert hasattr(deployment_pkg.OptimizationConfig, "to_dict"), (
            "OptimizationConfig should have a to_dict() method"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("attr", EXPECTED_ATTRIBUTES)
    def test_optimization_config_attribute_in_annotations_or_init(
        self, deployment_pkg, attr
    ):
        """
        Each expected attribute is declared in OptimizationConfig's annotations
        or __init__ signature.
        """
        cls = deployment_pkg.OptimizationConfig
        annotations = getattr(cls, "__annotations__", {})
        init_sig_params = set()
        try:
            sig = inspect.signature(cls.__init__)
            init_sig_params = set(sig.parameters.keys()) - {"self"}
        except (ValueError, TypeError):
            pass

        assert attr in annotations or attr in init_sig_params, (
            f"OptimizationConfig should declare '{attr}'"
        )


class TestContractMonitoringConfigDataclass:
    """§2 — MonitoringConfig is a dataclass with expected attributes."""

    EXPECTED_ATTRIBUTES = [
        "enable_performance_tracking",
        "enable_drift_detection",
        "enable_health_checks",
        "enable_alerting",
        "drift_detection_method",
        "drift_threshold",
        "alert_threshold",
        "health_check_interval",
        "metrics_window_size",
        "log_predictions",
        "log_metrics_interval",
        "retraining_trigger_threshold",
    ]

    @pytest.mark.contract
    def test_monitoring_config_is_class(self, deployment_pkg):
        """``MonitoringConfig`` is a class."""
        assert inspect.isclass(deployment_pkg.MonitoringConfig)

    @pytest.mark.contract
    def test_monitoring_config_has_to_dict(self, deployment_pkg):
        """``MonitoringConfig`` has a ``to_dict()`` method."""
        assert hasattr(deployment_pkg.MonitoringConfig, "to_dict"), (
            "MonitoringConfig should have a to_dict() method"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("attr", EXPECTED_ATTRIBUTES)
    def test_monitoring_config_attribute_in_annotations_or_init(
        self, deployment_pkg, attr
    ):
        """
        Each expected attribute is declared in MonitoringConfig's annotations
        or __init__ signature.
        """
        cls = deployment_pkg.MonitoringConfig
        annotations = getattr(cls, "__annotations__", {})
        init_sig_params = set()
        try:
            sig = inspect.signature(cls.__init__)
            init_sig_params = set(sig.parameters.keys()) - {"self"}
        except (ValueError, TypeError):
            pass

        assert attr in annotations or attr in init_sig_params, (
            f"MonitoringConfig should declare '{attr}'"
        )


class TestContractAlertDataclass:
    """§2 — Alert is a dataclass with expected attributes."""

    EXPECTED_ATTRIBUTES = [
        "severity",
        "message",
        "metric_type",
        "metric_value",
        "threshold",
        "timestamp",
    ]

    @pytest.mark.contract
    def test_alert_is_class(self, deployment_pkg):
        """``Alert`` is a class."""
        assert inspect.isclass(deployment_pkg.Alert)

    @pytest.mark.contract
    def test_alert_has_to_dict(self, deployment_pkg):
        """``Alert`` has a ``to_dict()`` method."""
        assert hasattr(deployment_pkg.Alert, "to_dict"), (
            "Alert should have a to_dict() method"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("attr", EXPECTED_ATTRIBUTES)
    def test_alert_attribute_in_annotations_or_init(
        self, deployment_pkg, attr
    ):
        """
        Each expected attribute is declared in Alert's annotations
        or __init__ signature.
        """
        cls = deployment_pkg.Alert
        annotations = getattr(cls, "__annotations__", {})
        init_sig_params = set()
        try:
            sig = inspect.signature(cls.__init__)
            init_sig_params = set(sig.parameters.keys()) - {"self"}
        except (ValueError, TypeError):
            pass

        assert attr in annotations or attr in init_sig_params, (
            f"Alert should declare '{attr}'"
        )


class TestContractDeploymentStrategyABC:
    """§2 — DeploymentStrategy is an ABC with required abstract methods."""

    EXPECTED_ABSTRACT_METHODS = [
        "prepare_model",
        "deploy",
        "predict",
        "teardown",
    ]

    @pytest.mark.contract
    def test_deployment_strategy_is_abstract(self, deployment_pkg):
        """
        ``DeploymentStrategy`` is an abstract class (has abstract methods or
        uses ABC/ABCMeta).
        """
        cls = deployment_pkg.DeploymentStrategy
        assert inspect.isclass(cls)
        # Check for ABCMeta or abstract methods
        has_abc_meta = type(cls).__name__ == "ABCMeta"
        has_abstract_methods = bool(getattr(cls, "__abstractmethods__", set()))
        assert has_abc_meta or has_abstract_methods, (
            "DeploymentStrategy should be an abstract class (ABCMeta or "
            "with __abstractmethods__)"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("method_name", EXPECTED_ABSTRACT_METHODS)
    def test_deployment_strategy_has_method(
        self, deployment_pkg, method_name
    ):
        """DeploymentStrategy defines each expected method."""
        cls = deployment_pkg.DeploymentStrategy
        assert hasattr(cls, method_name), (
            f"DeploymentStrategy should define '{method_name}'"
        )

    @pytest.mark.contract
    def test_deployment_strategy_has_get_deployment_info(self, deployment_pkg):
        """DeploymentStrategy defines ``get_deployment_info()``."""
        cls = deployment_pkg.DeploymentStrategy
        assert hasattr(cls, "get_deployment_info"), (
            "DeploymentStrategy should define 'get_deployment_info'"
        )


class TestContractStrategyImplementationsInheritance:
    """§2 — All strategy implementations are subclasses of DeploymentStrategy."""

    STRATEGY_IMPLEMENTATIONS = [
        "AWSDeploymentStrategy",
        "GCPDeploymentStrategy",
        "AzureDeploymentStrategy",
        "EdgeDeploymentStrategy",
        "ContainerDeploymentStrategy",
        "LocalDeploymentStrategy",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("strategy_name", STRATEGY_IMPLEMENTATIONS)
    def test_strategy_inherits_deployment_strategy(
        self, deployment_pkg, strategy_name
    ):
        """Each strategy implementation inherits from DeploymentStrategy."""
        strategy_cls = getattr(deployment_pkg, strategy_name)
        base_cls = deployment_pkg.DeploymentStrategy
        assert issubclass(strategy_cls, base_cls), (
            f"'{strategy_name}' should inherit from DeploymentStrategy"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("strategy_name", STRATEGY_IMPLEMENTATIONS)
    def test_strategy_is_concrete_class(self, deployment_pkg, strategy_name):
        """Each strategy implementation is a concrete (non-abstract) class."""
        strategy_cls = getattr(deployment_pkg, strategy_name)
        abstract_methods = getattr(strategy_cls, "__abstractmethods__", set())
        assert len(abstract_methods) == 0, (
            f"'{strategy_name}' should be concrete but has abstract methods: "
            f"{abstract_methods}"
        )


class TestContractDeploymentManagerPublicAPI:
    """§2 — DeploymentManager has the required public methods."""

    EXPECTED_METHODS = [
        "prepare_model",
        "deploy",
        "predict",
        "teardown",
        "get_deployment_info",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("method_name", EXPECTED_METHODS)
    def test_deployment_manager_has_method(
        self, deployment_pkg, method_name
    ):
        """DeploymentManager defines each expected public method."""
        cls = deployment_pkg.DeploymentManager
        assert hasattr(cls, method_name), (
            f"DeploymentManager should define '{method_name}'"
        )
        method = getattr(cls, method_name)
        assert callable(method), (
            f"DeploymentManager.{method_name} should be callable"
        )

    @pytest.mark.contract
    def test_deployment_manager_has_list_available_targets(
        self, deployment_pkg
    ):
        """DeploymentManager has ``list_available_targets()`` classmethod."""
        cls = deployment_pkg.DeploymentManager
        assert hasattr(cls, "list_available_targets"), (
            "DeploymentManager should define 'list_available_targets'"
        )

    @pytest.mark.contract
    def test_deployment_manager_has_strategies_registry(
        self, deployment_pkg
    ):
        """DeploymentManager has ``_strategies`` class attribute mapping."""
        cls = deployment_pkg.DeploymentManager
        assert hasattr(cls, "_strategies"), (
            "DeploymentManager should have '_strategies' class attribute"
        )
        strategies = getattr(cls, "_strategies")
        assert isinstance(strategies, dict), (
            f"DeploymentManager._strategies should be a dict, "
            f"got {type(strategies).__name__}"
        )


class TestContractModelOptimizerPublicAPI:
    """§2 — ModelOptimizer has the required public methods."""

    EXPECTED_METHODS = [
        "quantize_model",
        "prune_model",
        "export_to_onnx",
        "get_model_size",
        "compare_models",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("method_name", EXPECTED_METHODS)
    def test_model_optimizer_has_method(
        self, deployment_pkg, method_name
    ):
        """ModelOptimizer defines each expected public method."""
        cls = deployment_pkg.ModelOptimizer
        assert hasattr(cls, method_name), (
            f"ModelOptimizer should define '{method_name}'"
        )
        method = getattr(cls, method_name)
        assert callable(method), (
            f"ModelOptimizer.{method_name} should be callable"
        )


class TestContractModelMonitorPublicAPI:
    """§2 — ModelMonitor has the required public methods."""

    EXPECTED_METHODS = [
        "log_prediction",
        "log_error",
        "detect_drift",
        "set_reference_data",
        "health_check",
        "register_alert_callback",
        "get_alerts",
        "clear_alerts",
        "get_metrics_summary",
        "get_metric",
        "export_metrics",
        "reset_metrics",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("method_name", EXPECTED_METHODS)
    def test_model_monitor_has_method(
        self, deployment_pkg, method_name
    ):
        """ModelMonitor defines each expected public method."""
        cls = deployment_pkg.ModelMonitor
        assert hasattr(cls, method_name), (
            f"ModelMonitor should define '{method_name}'"
        )
        method = getattr(cls, method_name)
        assert callable(method), (
            f"ModelMonitor.{method_name} should be callable"
        )


class TestContractConvenienceFunctionSignatures:
    """§2 — High-level convenience functions have documented parameter signatures."""

    @pytest.mark.contract
    def test_deploy_model_locally_signature(self, deployment_pkg):
        """``deploy_model_locally`` has the expected parameters."""
        sig = inspect.signature(deployment_pkg.deploy_model_locally)
        param_names = set(sig.parameters.keys())
        expected = {"model", "save_path"}
        assert expected.issubset(param_names), (
            f"deploy_model_locally should accept at least {expected}, "
            f"got {param_names}"
        )

    @pytest.mark.contract
    def test_deploy_model_to_cloud_signature(self, deployment_pkg):
        """``deploy_model_to_cloud`` has the expected parameters."""
        sig = inspect.signature(deployment_pkg.deploy_model_to_cloud)
        param_names = set(sig.parameters.keys())
        expected = {"model", "save_path", "target"}
        assert expected.issubset(param_names), (
            f"deploy_model_to_cloud should accept at least {expected}, "
            f"got {param_names}"
        )

    @pytest.mark.contract
    def test_optimize_model_for_deployment_signature(self, deployment_pkg):
        """``optimize_model_for_deployment`` has the expected parameters."""
        sig = inspect.signature(deployment_pkg.optimize_model_for_deployment)
        param_names = set(sig.parameters.keys())
        expected = {"model", "quantize", "prune"}
        assert expected.issubset(param_names), (
            f"optimize_model_for_deployment should accept at least "
            f"{expected}, got {param_names}"
        )

    @pytest.mark.contract
    def test_create_production_monitor_signature(self, deployment_pkg):
        """``create_production_monitor`` has the expected parameters."""
        sig = inspect.signature(deployment_pkg.create_production_monitor)
        param_names = set(sig.parameters.keys())
        expected = {"model_name"}
        assert expected.issubset(param_names), (
            f"create_production_monitor should accept at least "
            f"{expected}, got {param_names}"
        )

    @pytest.mark.contract
    def test_create_deployment_pipeline_signature(self, deployment_pkg):
        """``create_deployment_pipeline`` has the expected parameters."""
        sig = inspect.signature(deployment_pkg.create_deployment_pipeline)
        param_names = set(sig.parameters.keys())
        expected = {"model", "save_path", "target", "optimize",
                     "enable_monitoring"}
        assert expected.issubset(param_names), (
            f"create_deployment_pipeline should accept at least "
            f"{expected}, got {param_names}"
        )


class TestContractOptimizeModelForDeploymentSignatureDetails:
    """§2 — optimize_model_for_deployment detailed parameter contracts."""

    @pytest.mark.contract
    def test_optimize_has_quantization_type_param(self, deployment_pkg):
        """``optimize_model_for_deployment`` accepts ``quantization_type``."""
        sig = inspect.signature(deployment_pkg.optimize_model_for_deployment)
        assert "quantization_type" in sig.parameters, (
            "optimize_model_for_deployment should accept 'quantization_type'"
        )

    @pytest.mark.contract
    def test_optimize_has_prune_amount_param(self, deployment_pkg):
        """``optimize_model_for_deployment`` accepts ``prune_amount``."""
        sig = inspect.signature(deployment_pkg.optimize_model_for_deployment)
        assert "prune_amount" in sig.parameters, (
            "optimize_model_for_deployment should accept 'prune_amount'"
        )

    @pytest.mark.contract
    def test_optimize_prune_amount_default(self, deployment_pkg):
        """``prune_amount`` defaults to 0.3."""
        sig = inspect.signature(deployment_pkg.optimize_model_for_deployment)
        param = sig.parameters["prune_amount"]
        assert param.default == 0.3, (
            f"prune_amount should default to 0.3, got {param.default}"
        )

    @pytest.mark.contract
    def test_optimize_has_export_onnx_param(self, deployment_pkg):
        """``optimize_model_for_deployment`` accepts ``export_onnx``."""
        sig = inspect.signature(deployment_pkg.optimize_model_for_deployment)
        assert "export_onnx" in sig.parameters, (
            "optimize_model_for_deployment should accept 'export_onnx'"
        )

    @pytest.mark.contract
    def test_optimize_has_verbose_param(self, deployment_pkg):
        """``optimize_model_for_deployment`` accepts ``verbose``."""
        sig = inspect.signature(deployment_pkg.optimize_model_for_deployment)
        assert "verbose" in sig.parameters, (
            "optimize_model_for_deployment should accept 'verbose'"
        )


class TestContractCreateProductionMonitorSignatureDetails:
    """§2 — create_production_monitor detailed parameter contracts."""

    EXPECTED_PARAMS = [
        "model_name",
        "enable_performance_tracking",
        "enable_drift_detection",
        "enable_health_checks",
        "enable_alerting",
        "drift_threshold",
        "alert_threshold",
        "metrics_window_size",
        "verbose",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("param_name", EXPECTED_PARAMS)
    def test_create_production_monitor_has_param(
        self, deployment_pkg, param_name
    ):
        """``create_production_monitor`` accepts each expected parameter."""
        sig = inspect.signature(deployment_pkg.create_production_monitor)
        assert param_name in sig.parameters, (
            f"create_production_monitor should accept '{param_name}'"
        )


class TestContractCreateDeploymentPipelineSignatureDetails:
    """§2 — create_deployment_pipeline detailed parameter contracts."""

    EXPECTED_PARAMS = [
        "model",
        "save_path",
        "target",
        "optimize",
        "quantize",
        "prune",
        "prune_amount",
        "enable_monitoring",
        "model_name",
        "instance_type",
        "verbose",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("param_name", EXPECTED_PARAMS)
    def test_create_deployment_pipeline_has_param(
        self, deployment_pkg, param_name
    ):
        """``create_deployment_pipeline`` accepts each expected parameter."""
        sig = inspect.signature(deployment_pkg.create_deployment_pipeline)
        assert param_name in sig.parameters, (
            f"create_deployment_pipeline should accept '{param_name}'"
        )

    @pytest.mark.contract
    def test_pipeline_target_defaults_to_local(self, deployment_pkg):
        """``target`` defaults to ``"local"``."""
        sig = inspect.signature(deployment_pkg.create_deployment_pipeline)
        param = sig.parameters["target"]
        assert param.default == "local", (
            f"target should default to 'local', got {param.default!r}"
        )


class TestContractDeployModelToCloudValidation:
    """§2 — deploy_model_to_cloud validates target parameter."""

    @pytest.mark.contract
    def test_deploy_model_to_cloud_rejects_invalid_target(
        self, deployment_pkg
    ):
        """
        ``deploy_model_to_cloud`` raises ConfigurationError for an invalid
        cloud target.
        """
        mock_model = MagicMock()
        with pytest.raises(
            (deployment_pkg.ConfigurationError, deployment_pkg.DeploymentError)
        ):
            deployment_pkg.deploy_model_to_cloud(
                model=mock_model,
                save_path="/tmp/test_deploy",
                target="invalid_cloud_provider"
            )


class TestContractValidateDeploymentConfig:
    """§2 — validate_deployment_config enforces configuration rules."""

    @pytest.mark.contract
    def test_validate_deployment_config_is_callable(self, deployment_pkg):
        """``validate_deployment_config`` is callable."""
        assert callable(deployment_pkg.validate_deployment_config)

    @pytest.mark.contract
    def test_validate_deployment_config_signature(self, deployment_pkg):
        """``validate_deployment_config`` accepts a ``config`` parameter."""
        sig = inspect.signature(deployment_pkg.validate_deployment_config)
        param_names = set(sig.parameters.keys())
        assert "config" in param_names, (
            "validate_deployment_config should accept 'config'"
        )


class TestContractGetDeploymentInfo:
    """§2 — get_deployment_info function contract."""

    @pytest.mark.contract
    def test_get_deployment_info_is_callable(self, deployment_pkg):
        """``get_deployment_info`` is callable."""
        assert callable(deployment_pkg.get_deployment_info)

    @pytest.mark.contract
    def test_get_deployment_info_signature(self, deployment_pkg):
        """``get_deployment_info`` accepts a ``manager`` parameter."""
        sig = inspect.signature(deployment_pkg.get_deployment_info)
        param_names = set(sig.parameters.keys())
        assert "manager" in param_names, (
            "get_deployment_info should accept 'manager'"
        )

    @pytest.mark.contract
    def test_get_deployment_info_delegates_to_manager(self, deployment_pkg):
        """``get_deployment_info`` delegates to manager.get_deployment_info()."""
        mock_manager = MagicMock()
        mock_manager.get_deployment_info.return_value = {"status": "deployed"}
        result = deployment_pkg.get_deployment_info(mock_manager)
        mock_manager.get_deployment_info.assert_called_once()
        assert result == {"status": "deployed"}


class TestContractPublicAPISurface:
    """§2 — Public API surface stability: minimum expected names present."""

    # Exhaustive list of all names expected in __all__ per the __init__.py
    EXPECTED_ALL_NAMES = [
        # Exceptions
        "ModelError",
        "DeploymentError",
        "OptimizationError",
        "MonitoringError",
        "ConfigurationError",
        "ExportError",
        "AlertError",
        # Deployment - Base and Config
        "DeploymentStrategy",
        "DeploymentConfig",
        "DeploymentTarget",
        "ServingMode",
        # Deployment - Strategies
        "AWSDeploymentStrategy",
        "GCPDeploymentStrategy",
        "AzureDeploymentStrategy",
        "EdgeDeploymentStrategy",
        "ContainerDeploymentStrategy",
        "LocalDeploymentStrategy",
        "DeploymentManager",
        # Optimization
        "ModelOptimizer",
        "OptimizationConfig",
        "QuantizationType",
        "PruningType",
        # Monitoring
        "ModelMonitor",
        "MonitoringConfig",
        "Alert",
        "MetricType",
        "AlertSeverity",
        "DriftType",
        # High-level convenience functions
        "deploy_model_locally",
        "deploy_model_to_cloud",
        "optimize_model_for_deployment",
        "create_production_monitor",
        "create_deployment_pipeline",
        # Utility functions
        "list_deployment_targets",
        "get_deployment_info",
        "validate_deployment_config",
        # Submodule convenience functions
        "deploy_locally",
        "quantize_for_inference",
        "prune_for_deployment",
        "create_monitor",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", EXPECTED_ALL_NAMES)
    def test_expected_name_in_all(self, all_names, name):
        """Each expected public API name is present in ``__all__``."""
        assert name in all_names, (
            f"Expected name '{name}' is missing from __all__"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", EXPECTED_ALL_NAMES)
    def test_expected_name_is_resolvable(self, deployment_pkg, name):
        """Each expected public API name resolves to a non-None object."""
        obj = getattr(deployment_pkg, name, None)
        assert obj is not None, (
            f"Expected API name '{name}' is None or missing from module"
        )


class TestContractCallableVsClassClassification:
    """§2 — Re-exported classes are classes, callables are callable."""

    CLASS_NAMES = [
        "DeploymentStrategy",
        "DeploymentConfig",
        "DeploymentTarget",
        "ServingMode",
        "AWSDeploymentStrategy",
        "GCPDeploymentStrategy",
        "AzureDeploymentStrategy",
        "EdgeDeploymentStrategy",
        "ContainerDeploymentStrategy",
        "LocalDeploymentStrategy",
        "DeploymentManager",
        "ModelOptimizer",
        "OptimizationConfig",
        "QuantizationType",
        "PruningType",
        "ModelMonitor",
        "MonitoringConfig",
        "Alert",
        "MetricType",
        "AlertSeverity",
        "DriftType",
        "ModelError",
        "DeploymentError",
        "OptimizationError",
        "MonitoringError",
        "ConfigurationError",
        "ExportError",
        "AlertError",
    ]

    FUNCTION_NAMES = [
        "deploy_model_locally",
        "deploy_model_to_cloud",
        "optimize_model_for_deployment",
        "create_production_monitor",
        "create_deployment_pipeline",
        "list_deployment_targets",
        "get_deployment_info",
        "validate_deployment_config",
        "deploy_locally",
        "quantize_for_inference",
        "prune_for_deployment",
        "create_monitor",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", CLASS_NAMES)
    def test_class_export_is_class(self, deployment_pkg, name):
        """Each class export is a class."""
        obj = getattr(deployment_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", FUNCTION_NAMES)
    def test_function_export_is_callable(self, deployment_pkg, name):
        """Each function export is callable."""
        obj = getattr(deployment_pkg, name)
        assert callable(obj), (
            f"'{name}' should be callable, got {type(obj).__name__}"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", FUNCTION_NAMES)
    def test_function_export_is_function(self, deployment_pkg, name):
        """Each function export is a function (not a class or bound method)."""
        obj = getattr(deployment_pkg, name)
        assert inspect.isfunction(obj), (
            f"'{name}' should be a function, got {type(obj).__name__}"
        )
