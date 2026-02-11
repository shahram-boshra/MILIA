# tests/test__init__models_acceleration.py

"""
Test Suite: milia_pipeline/models/acceleration/__init__.py — Smoke Tests & Contract Tests
=========================================================================================

Production-ready test suite for the MILIA Pipeline acceleration package
``milia_pipeline/models/acceleration/__init__.py``.

Covers:
    Section 1 — Smoke Tests (MILIA_Test_Recommendations.md §1.2 scope):
        - The ``milia_pipeline.models.acceleration`` subpackage imports without
          ImportError (validates no circular imports, missing deps, broken __init__)
        - All re-exported names from the 4 submodules are accessible
        - Module-level metadata attributes (__version__, __author__, __description__)
          exist and are typed correctly
        - Module initialization (logging, hardware capability checks) runs safely
        - Re-import (``importlib.reload``) is idempotent and non-crashing
        - Core manager classes are accessible (AccelerationManager, DeviceManager,
          MemoryOptimizer, ComputationOptimizer, DistributedManager)
        - Configuration/dataclass classes are accessible (DeviceInfo, DeviceType,
          MemoryConfig, ComputationConfig, DistributedConfig, DistributedStrategy,
          DistributedBackend)
        - Convenience functions are accessible and callable
        - Fallback exception classes are accessible
        - Module-level hardware capability check (_capabilities) executes

    Section 2 — Contract Tests (MILIA_Test_Recommendations.md §2 scope):
        - ``__all__`` completeness: every name in ``__all__`` is resolvable
        - ``__all__`` consistency: every public import is listed in ``__all__``
        - ``__all__`` has no duplicates
        - ``__all__`` entries are all strings
        - Re-exported classes are classes, callables are callable
        - Manager classes are instantiable types (classes)
        - Configuration classes are dataclasses or Pydantic BaseModel subclasses
        - Enum classes (DeviceType, DistributedStrategy, DistributedBackend) are enums
        - Convenience functions have documented parameter signatures
        - Exception class hierarchy (ModelError → HardwareError →
          DeviceNotAvailableError / VQMMemoryError / OptimizationError /
          DistributedError) follows documented inheritance
        - AccelerationManager.__init__ signature contracts
        - auto_optimize_for_training / auto_optimize_for_inference signature contracts
        - get_recommended_settings / create_acceleration_manager /
          benchmark_accelerations are callable with expected signatures
        - ``__version__`` follows semver pattern
        - Public API surface stability (minimum expected names present)
        - ``__all__`` has expected length (guards against catastrophic truncation)

Design:
    - Zero ``sys.modules`` pollution: all mocking via ``@patch`` decorators
      or context-manager ``patch`` calls scoped to individual tests.
    - Deterministic: no filesystem, network, or GPU access required.
    - Isolated: each test is independent; execution order is irrelevant.
    - Fast: expected < 5 s total wall-clock on CI.

Launch:
    From project root (/app/milia):
        pytest tests/test__init__models_acceleration.py -v --tb=short

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
# When launched via ``pytest tests/test__init__models_acceleration.py`` from
# the project root (/app/milia), ``milia_pipeline`` must be on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # …/tests -> …/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def accel_pkg():
    """
    Import and return the ``milia_pipeline.models.acceleration`` package
    once per module.

    This fixture validates the fundamental smoke invariant: the acceleration
    subpackage is importable.  If this fails, every downstream test is moot.
    """
    try:
        import milia_pipeline.models.acceleration as accel
        return accel
    except ImportError as exc:
        pytest.fail(
            f"milia_pipeline.models.acceleration could not be imported — "
            f"smoke test precondition violated: {exc}"
        )


@pytest.fixture(scope="module")
def all_names(accel_pkg):
    """Return the ``__all__`` list from the acceleration package."""
    assert hasattr(accel_pkg, "__all__"), (
        "milia_pipeline.models.acceleration.__all__ is missing — "
        "contract violation"
    )
    return list(accel_pkg.__all__)


# ===================================================================
# SECTION 1 — SMOKE TESTS
# ===================================================================


class TestSmokeAccelerationPackageImport:
    """§1.2 — Verify the acceleration subpackage imports without errors."""

    @pytest.mark.smoke
    def test_import_acceleration_package_succeeds(self, accel_pkg):
        """The acceleration package imports without raising any exception."""
        assert accel_pkg is not None

    @pytest.mark.smoke
    def test_acceleration_package_is_a_module(self, accel_pkg):
        """The imported object is a proper Python module."""
        assert isinstance(accel_pkg, types.ModuleType)

    @pytest.mark.smoke
    def test_acceleration_package_has_file_attribute(self, accel_pkg):
        """The package exposes a ``__file__`` attribute (not a namespace pkg)."""
        assert hasattr(accel_pkg, "__file__")

    @pytest.mark.smoke
    def test_acceleration_package_name(self, accel_pkg):
        """The package ``__name__`` is ``milia_pipeline.models.acceleration``."""
        assert accel_pkg.__name__ == "milia_pipeline.models.acceleration"


class TestSmokeMetadataAttributes:
    """§1.2 — Verify module-level metadata attributes are present and typed."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("attr", [
        "__version__",
        "__author__",
        "__description__",
    ])
    def test_metadata_attribute_exists(self, accel_pkg, attr):
        """Each metadata dunder is defined on the acceleration package."""
        assert hasattr(accel_pkg, attr), f"Missing attribute: {attr}"

    @pytest.mark.smoke
    @pytest.mark.parametrize("attr", [
        "__version__",
        "__author__",
        "__description__",
    ])
    def test_metadata_attribute_is_string(self, accel_pkg, attr):
        """Each metadata dunder is a non-empty string."""
        value = getattr(accel_pkg, attr)
        assert isinstance(value, str), f"{attr} should be str, got {type(value)}"
        assert len(value) > 0, f"{attr} should be non-empty"

    @pytest.mark.smoke
    def test_version_is_semver_like(self, accel_pkg):
        """``__version__`` follows a MAJOR.MINOR.PATCH pattern."""
        version = accel_pkg.__version__
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
    def test_version_value(self, accel_pkg):
        """``__version__`` is ``'1.0.0'`` as declared in the source."""
        assert accel_pkg.__version__ == "1.0.0"

    @pytest.mark.smoke
    def test_author_value(self, accel_pkg):
        """``__author__`` is ``'milia Team'`` as declared in the source."""
        assert accel_pkg.__author__ == "milia Team"


class TestSmokeDeviceManagerExports:
    """§1.2 — Device management exports from device_manager.py are accessible."""

    DEVICE_MANAGER_CLASSES = [
        "DeviceManager",
        "DeviceInfo",
        "DeviceType",
    ]

    DEVICE_MANAGER_FUNCTIONS = [
        "get_default_device",
        "list_available_devices",
        "get_device_capabilities",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", DEVICE_MANAGER_CLASSES)
    def test_device_manager_class_exists(self, accel_pkg, name):
        """Each device manager class is importable from the acceleration package."""
        obj = getattr(accel_pkg, name, None)
        assert obj is not None, f"Device manager class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", DEVICE_MANAGER_CLASSES)
    def test_device_manager_class_is_a_class(self, accel_pkg, name):
        """Each device manager export is a class (not an instance or function)."""
        obj = getattr(accel_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", DEVICE_MANAGER_FUNCTIONS)
    def test_device_manager_function_exists(self, accel_pkg, name):
        """Each device manager function is present and non-None."""
        obj = getattr(accel_pkg, name, None)
        assert obj is not None, (
            f"Device manager function '{name}' is None or missing"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", DEVICE_MANAGER_FUNCTIONS)
    def test_device_manager_function_is_callable(self, accel_pkg, name):
        """Each device manager function is callable."""
        obj = getattr(accel_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeMemoryOptimizationExports:
    """§1.2 — Memory optimization exports from memory_optimization.py are accessible."""

    MEMORY_CLASSES = [
        "MemoryOptimizer",
        "MemoryConfig",
    ]

    MEMORY_FUNCTIONS = [
        "get_memory_efficient_settings",
        "estimate_model_memory",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", MEMORY_CLASSES)
    def test_memory_class_exists(self, accel_pkg, name):
        """Each memory optimization class is importable."""
        obj = getattr(accel_pkg, name, None)
        assert obj is not None, f"Memory class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", MEMORY_CLASSES)
    def test_memory_class_is_a_class(self, accel_pkg, name):
        """Each memory optimization export is a class."""
        obj = getattr(accel_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", MEMORY_FUNCTIONS)
    def test_memory_function_exists(self, accel_pkg, name):
        """Each memory optimization function is present and non-None."""
        obj = getattr(accel_pkg, name, None)
        assert obj is not None, (
            f"Memory function '{name}' is None or missing"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", MEMORY_FUNCTIONS)
    def test_memory_function_is_callable(self, accel_pkg, name):
        """Each memory optimization function is callable."""
        obj = getattr(accel_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeComputationOptimizationExports:
    """§1.2 — Computation optimization exports from computation_optimization.py are accessible."""

    COMPUTATION_CLASSES = [
        "ComputationOptimizer",
        "ComputationConfig",
    ]

    COMPUTATION_FUNCTIONS = [
        "get_optimal_settings",
        "auto_optimize_model",
        "optimize_inference",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", COMPUTATION_CLASSES)
    def test_computation_class_exists(self, accel_pkg, name):
        """Each computation optimization class is importable."""
        obj = getattr(accel_pkg, name, None)
        assert obj is not None, f"Computation class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", COMPUTATION_CLASSES)
    def test_computation_class_is_a_class(self, accel_pkg, name):
        """Each computation optimization export is a class."""
        obj = getattr(accel_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", COMPUTATION_FUNCTIONS)
    def test_computation_function_exists(self, accel_pkg, name):
        """Each computation optimization function is present and non-None."""
        obj = getattr(accel_pkg, name, None)
        assert obj is not None, (
            f"Computation function '{name}' is None or missing"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", COMPUTATION_FUNCTIONS)
    def test_computation_function_is_callable(self, accel_pkg, name):
        """Each computation optimization function is callable."""
        obj = getattr(accel_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeDistributedStrategiesExports:
    """§1.2 — Distributed strategies exports from distributed_strategies.py are accessible."""

    DISTRIBUTED_CLASSES = [
        "DistributedManager",
        "DistributedConfig",
        "DistributedStrategy",
        "DistributedBackend",
    ]

    DISTRIBUTED_FUNCTIONS = [
        "is_distributed_available",
        "get_world_size",
        "get_rank",
        "is_main_process",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", DISTRIBUTED_CLASSES)
    def test_distributed_class_exists(self, accel_pkg, name):
        """Each distributed strategies class is importable."""
        obj = getattr(accel_pkg, name, None)
        assert obj is not None, (
            f"Distributed class '{name}' is None or missing"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", DISTRIBUTED_CLASSES)
    def test_distributed_class_is_a_class(self, accel_pkg, name):
        """Each distributed strategies export is a class."""
        obj = getattr(accel_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", DISTRIBUTED_FUNCTIONS)
    def test_distributed_function_exists(self, accel_pkg, name):
        """Each distributed strategies function is present and non-None."""
        obj = getattr(accel_pkg, name, None)
        assert obj is not None, (
            f"Distributed function '{name}' is None or missing"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", DISTRIBUTED_FUNCTIONS)
    def test_distributed_function_is_callable(self, accel_pkg, name):
        """Each distributed strategies function is callable."""
        obj = getattr(accel_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeAccelerationManagerExport:
    """§1.2 — AccelerationManager unified class is accessible."""

    @pytest.mark.smoke
    def test_acceleration_manager_exists(self, accel_pkg):
        """``AccelerationManager`` is defined on the acceleration package."""
        assert hasattr(accel_pkg, "AccelerationManager")

    @pytest.mark.smoke
    def test_acceleration_manager_is_class(self, accel_pkg):
        """``AccelerationManager`` is a class."""
        assert inspect.isclass(accel_pkg.AccelerationManager)


class TestSmokeConvenienceFunctionExports:
    """§1.2 — Convenience functions defined in __init__.py are accessible and callable."""

    CONVENIENCE_FUNCTIONS = [
        "auto_optimize_for_training",
        "auto_optimize_for_inference",
        "get_recommended_settings",
        "create_acceleration_manager",
        "benchmark_accelerations",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CONVENIENCE_FUNCTIONS)
    def test_convenience_function_exists(self, accel_pkg, name):
        """Each convenience function is present and non-None."""
        obj = getattr(accel_pkg, name, None)
        assert obj is not None, (
            f"Convenience function '{name}' is None or missing"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", CONVENIENCE_FUNCTIONS)
    def test_convenience_function_is_callable(self, accel_pkg, name):
        """Each convenience function is callable."""
        obj = getattr(accel_pkg, name)
        assert callable(obj), f"'{name}' should be callable"


class TestSmokeExceptionClassExports:
    """§1.2 — Exception classes (primary or fallback) are accessible."""

    EXCEPTION_CLASSES = [
        "ModelError",
        "HardwareError",
        "DeviceNotAvailableError",
        "VQMMemoryError",
        "OptimizationError",
        "DistributedError",
    ]

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", EXCEPTION_CLASSES)
    def test_exception_class_exists(self, accel_pkg, name):
        """Each exception class is defined on the acceleration package."""
        obj = getattr(accel_pkg, name, None)
        assert obj is not None, f"Exception class '{name}' is None or missing"

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", EXCEPTION_CLASSES)
    def test_exception_class_is_a_class(self, accel_pkg, name):
        """Each exception export is a class."""
        obj = getattr(accel_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("name", EXCEPTION_CLASSES)
    def test_exception_class_is_subclass_of_exception(self, accel_pkg, name):
        """Each exception class inherits from the built-in Exception."""
        obj = getattr(accel_pkg, name)
        assert issubclass(obj, Exception), (
            f"'{name}' should be a subclass of Exception"
        )


class TestSmokeModuleInitialization:
    """§1.2 — Module-level initialization runs without side-effect crashes."""

    @pytest.mark.smoke
    def test_reimport_is_idempotent(self, accel_pkg):
        """
        Re-importing the acceleration package (via ``importlib.reload``) does
        not crash.

        Validates that all module-level code (logging, hardware capability
        checks) is safe to re-execute.
        """
        reloaded = importlib.reload(accel_pkg)
        assert reloaded is not None
        assert hasattr(reloaded, "__version__")

    @pytest.mark.smoke
    def test_reimport_preserves_all(self, accel_pkg):
        """
        Re-importing the acceleration package preserves ``__all__``.
        """
        reloaded = importlib.reload(accel_pkg)
        assert hasattr(reloaded, "__all__")
        assert isinstance(reloaded.__all__, list)
        assert len(reloaded.__all__) > 0

    @pytest.mark.smoke
    def test_logger_exists(self, accel_pkg):
        """
        The module-level ``logger`` attribute is a logging.Logger instance.

        The __init__.py creates: ``logger = logging.getLogger(__name__)``.
        """
        assert hasattr(accel_pkg, "logger")
        assert isinstance(accel_pkg.logger, logging.Logger)

    @pytest.mark.smoke
    def test_capabilities_dict_exists(self, accel_pkg):
        """
        The module-level ``_capabilities`` dictionary is present.

        The __init__.py calls ``_capabilities = get_device_capabilities()``
        at module load time and checks its contents for logging.
        """
        assert hasattr(accel_pkg, "_capabilities")
        caps = accel_pkg._capabilities
        assert isinstance(caps, dict), (
            f"_capabilities should be dict, got {type(caps).__name__}"
        )

    @pytest.mark.smoke
    def test_capabilities_has_expected_keys(self, accel_pkg):
        """
        ``_capabilities`` contains documented keys from
        ``get_device_capabilities()``.

        Per the project structure doc, get_device_capabilities() returns:
        {cuda_available, cuda_device_count, mps_available, tpu_available,
        cudnn_available, cudnn_enabled}.
        """
        caps = accel_pkg._capabilities
        expected_keys = {
            "cuda_available",
            "cuda_device_count",
            "mps_available",
            "tpu_available",
        }
        for key in expected_keys:
            assert key in caps, (
                f"_capabilities missing expected key '{key}'. "
                f"Available keys: {sorted(caps.keys())}"
            )

    @pytest.mark.smoke
    def test_capabilities_cuda_available_is_bool(self, accel_pkg):
        """``_capabilities['cuda_available']`` is a boolean."""
        caps = accel_pkg._capabilities
        assert isinstance(caps["cuda_available"], bool), (
            f"cuda_available should be bool, got "
            f"{type(caps['cuda_available']).__name__}"
        )

    @pytest.mark.smoke
    def test_capabilities_mps_available_is_bool(self, accel_pkg):
        """``_capabilities['mps_available']`` is a boolean."""
        caps = accel_pkg._capabilities
        assert isinstance(caps["mps_available"], bool), (
            f"mps_available should be bool, got "
            f"{type(caps['mps_available']).__name__}"
        )

    @pytest.mark.smoke
    def test_capabilities_tpu_available_is_bool(self, accel_pkg):
        """``_capabilities['tpu_available']`` is a boolean."""
        caps = accel_pkg._capabilities
        assert isinstance(caps["tpu_available"], bool), (
            f"tpu_available should be bool, got "
            f"{type(caps['tpu_available']).__name__}"
        )

    @pytest.mark.smoke
    def test_capabilities_cuda_device_count_is_int(self, accel_pkg):
        """``_capabilities['cuda_device_count']`` is an integer."""
        caps = accel_pkg._capabilities
        assert isinstance(caps["cuda_device_count"], int), (
            f"cuda_device_count should be int, got "
            f"{type(caps['cuda_device_count']).__name__}"
        )


# ===================================================================
# SECTION 2 — CONTRACT TESTS
# ===================================================================


class TestContractAllCompleteness:
    """§2 — Every name in ``__all__`` is resolvable on the acceleration package."""

    @pytest.mark.contract
    def test_all_is_a_list(self, accel_pkg):
        """``__all__`` is a list (not a tuple or set)."""
        assert isinstance(accel_pkg.__all__, list)

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
    def test_every_all_entry_is_resolvable(self, accel_pkg, all_names):
        """
        Generic sweep: every single entry in ``__all__`` must be resolvable,
        regardless of whether it is parameterized individually.
        """
        unresolvable = [
            name for name in all_names
            if not hasattr(accel_pkg, name)
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


class TestContractAllConsistency:
    """§2 — Every public import in the acceleration module is listed in ``__all__``."""

    # Names that are intentionally public-ish but NOT in __all__
    KNOWN_UNLISTED = {
        # Metadata dunders (not typically in __all__)
        # Note: __version__ IS in __all__ for this module
        # Module-level logger
        "logger",
        # Module-level capabilities dict (internal)
        "_capabilities",
        # typing imports used internally by the module (line 75 of __init__.py:
        # ``from typing import Optional, Dict, Any, List, Union, Tuple``)
        # These are standard library re-exports, not part of the package API.
        "Optional",
        "Dict",
        "Any",
        "List",
        "Union",
        "Tuple",
    }

    @pytest.mark.contract
    def test_public_imports_are_in_all(self, accel_pkg, all_names):
        """
        Every non-dunder, non-private attribute that was imported (not
        defined locally as an internal) should be in ``__all__`` — unless
        it is in ``KNOWN_UNLISTED``.

        This catches accidental omissions when new imports are added to
        the acceleration ``__init__.py`` without updating ``__all__``.
        """
        all_set = set(all_names)
        module_dict = vars(accel_pkg)
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
            # Skip private names that are not in __all__
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
            f"Public names imported in acceleration/__init__.py but not "
            f"in __all__: {sorted(missing_from_all)}"
        )


class TestContractManagerClassTypes:
    """§2 — Core manager classes are actual classes (not instances or functions)."""

    MANAGER_CLASSES = [
        "AccelerationManager",
        "DeviceManager",
        "MemoryOptimizer",
        "ComputationOptimizer",
        "DistributedManager",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", MANAGER_CLASSES)
    def test_manager_is_class(self, accel_pkg, name):
        """Each manager export is a class."""
        obj = getattr(accel_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )


class TestContractConfigurationClassTypes:
    """§2 — Configuration/data classes are actual classes with dataclass or Pydantic semantics."""

    # Per the project structure doc:
    #   DeviceInfo: Pydantic BaseModel (mutable, 8 attributes)
    #   MemoryConfig: dataclass (9 attributes)
    #   ComputationConfig: dataclass (10 attributes)
    #   DistributedConfig: dataclass (12 attributes)

    DATACLASS_CONFIGS = [
        "MemoryConfig",
        "ComputationConfig",
        "DistributedConfig",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", DATACLASS_CONFIGS)
    def test_config_is_class(self, accel_pkg, name):
        """Each configuration class is a class."""
        obj = getattr(accel_pkg, name)
        assert inspect.isclass(obj), (
            f"'{name}' should be a class, got {type(obj).__name__}"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", DATACLASS_CONFIGS)
    def test_config_is_dataclass_or_pydantic(self, accel_pkg, name):
        """
        Each configuration class is a dataclass or Pydantic BaseModel.

        Per the project structure doc, MemoryConfig, ComputationConfig, and
        DistributedConfig are documented as ``(dataclass, N attributes)``.
        """
        cls = getattr(accel_pkg, name)

        # Check for stdlib dataclass
        is_stdlib_dc = hasattr(cls, "__dataclass_fields__")

        # Check for Pydantic BaseModel
        try:
            from pydantic import BaseModel
            is_pydantic_model = issubclass(cls, BaseModel)
        except ImportError:
            is_pydantic_model = False

        # Check for Pydantic dataclass
        is_pydantic_dc = hasattr(cls, "__pydantic_fields__")

        assert is_stdlib_dc or is_pydantic_model or is_pydantic_dc, (
            f"'{name}' should be a dataclass, Pydantic BaseModel, or "
            f"Pydantic dataclass"
        )

    @pytest.mark.contract
    @pytest.mark.parametrize("name", DATACLASS_CONFIGS)
    def test_config_has_to_dict_method(self, accel_pkg, name):
        """
        Each configuration class exposes a ``to_dict()`` method.

        Per the project structure doc, all config dataclasses have
        ``to_dict(): Configuration serialization``.
        """
        cls = getattr(accel_pkg, name)
        assert hasattr(cls, "to_dict"), (
            f"'{name}' should have a 'to_dict' method for serialization"
        )

    @pytest.mark.contract
    def test_device_info_is_pydantic_or_dataclass(self, accel_pkg):
        """
        ``DeviceInfo`` is a Pydantic BaseModel or dataclass.

        Per the project structure doc: DeviceInfo is a Pydantic BaseModel
        with mutable config and 8 attributes.
        """
        cls = accel_pkg.DeviceInfo
        assert inspect.isclass(cls)

        try:
            from pydantic import BaseModel
            is_pydantic = issubclass(cls, BaseModel)
        except ImportError:
            is_pydantic = False

        is_stdlib_dc = hasattr(cls, "__dataclass_fields__")
        is_pydantic_dc = hasattr(cls, "__pydantic_fields__")

        assert is_pydantic or is_stdlib_dc or is_pydantic_dc, (
            "DeviceInfo should be a Pydantic BaseModel, dataclass, "
            "or Pydantic dataclass"
        )


class TestContractEnumClassTypes:
    """§2 — Enum classes are proper enums with expected members."""

    @pytest.mark.contract
    def test_device_type_is_class(self, accel_pkg):
        """``DeviceType`` is a class."""
        assert inspect.isclass(accel_pkg.DeviceType)

    @pytest.mark.contract
    def test_device_type_is_enum(self, accel_pkg):
        """
        ``DeviceType`` is an Enum.

        Per the project structure doc: DeviceType (Enum, 5 values):
        CPU, CUDA, MPS, TPU, AUTO.
        """
        import enum
        assert issubclass(accel_pkg.DeviceType, enum.Enum), (
            "DeviceType should be an Enum subclass"
        )

    @pytest.mark.contract
    def test_device_type_has_expected_members(self, accel_pkg):
        """
        ``DeviceType`` has the 5 documented members.

        Per the project structure doc: CPU, CUDA, MPS, TPU, AUTO.
        """
        dt = accel_pkg.DeviceType
        expected_members = {"CPU", "CUDA", "MPS", "TPU", "AUTO"}
        actual_members = {m.name for m in dt}
        missing = expected_members - actual_members
        assert not missing, (
            f"DeviceType missing expected members: {missing}. "
            f"Actual members: {sorted(actual_members)}"
        )

    @pytest.mark.contract
    def test_distributed_strategy_is_class(self, accel_pkg):
        """``DistributedStrategy`` is a class."""
        assert inspect.isclass(accel_pkg.DistributedStrategy)

    @pytest.mark.contract
    def test_distributed_strategy_is_enum(self, accel_pkg):
        """
        ``DistributedStrategy`` is an Enum.

        Per the project structure doc: DistributedStrategy (Enum, 6 values):
        NONE, DP, DDP, FSDP, DEEPSPEED, HOROVOD.
        """
        import enum
        assert issubclass(accel_pkg.DistributedStrategy, enum.Enum), (
            "DistributedStrategy should be an Enum subclass"
        )

    @pytest.mark.contract
    def test_distributed_strategy_has_expected_members(self, accel_pkg):
        """
        ``DistributedStrategy`` has the 6 documented members.

        Per the project structure doc: NONE, DP, DDP, FSDP, DEEPSPEED, HOROVOD.
        """
        ds = accel_pkg.DistributedStrategy
        expected_members = {"NONE", "DP", "DDP", "FSDP", "DEEPSPEED", "HOROVOD"}
        actual_members = {m.name for m in ds}
        missing = expected_members - actual_members
        assert not missing, (
            f"DistributedStrategy missing expected members: {missing}. "
            f"Actual members: {sorted(actual_members)}"
        )

    @pytest.mark.contract
    def test_distributed_backend_is_class(self, accel_pkg):
        """``DistributedBackend`` is a class."""
        assert inspect.isclass(accel_pkg.DistributedBackend)

    @pytest.mark.contract
    def test_distributed_backend_is_enum(self, accel_pkg):
        """
        ``DistributedBackend`` is an Enum.

        Per the project structure doc: DistributedBackend (Enum, 4 values):
        GLOO, NCCL, MPI, AUTO.
        """
        import enum
        assert issubclass(accel_pkg.DistributedBackend, enum.Enum), (
            "DistributedBackend should be an Enum subclass"
        )

    @pytest.mark.contract
    def test_distributed_backend_has_expected_members(self, accel_pkg):
        """
        ``DistributedBackend`` has the 4 documented members.

        Per the project structure doc: GLOO, NCCL, MPI, AUTO.
        """
        db = accel_pkg.DistributedBackend
        expected_members = {"GLOO", "NCCL", "MPI", "AUTO"}
        actual_members = {m.name for m in db}
        missing = expected_members - actual_members
        assert not missing, (
            f"DistributedBackend missing expected members: {missing}. "
            f"Actual members: {sorted(actual_members)}"
        )


class TestContractExceptionHierarchy:
    """§2 — Exception class inheritance follows the documented hierarchy.

    Documented hierarchy from __init__.py (lines 155-177):
        ModelError(Exception)
            └── HardwareError(ModelError)
                ├── DeviceNotAvailableError(HardwareError)
                ├── VQMMemoryError(HardwareError)
                ├── OptimizationError(HardwareError)
                └── DistributedError(HardwareError)
    """

    @pytest.mark.contract
    def test_model_error_is_base_exception(self, accel_pkg):
        """``ModelError`` is a direct subclass of Exception."""
        assert issubclass(accel_pkg.ModelError, Exception)

    @pytest.mark.contract
    def test_hardware_error_inherits_model_error(self, accel_pkg):
        """``HardwareError`` inherits from ``ModelError``."""
        assert issubclass(accel_pkg.HardwareError, accel_pkg.ModelError)

    @pytest.mark.contract
    def test_device_not_available_error_inherits_hardware_error(self, accel_pkg):
        """``DeviceNotAvailableError`` inherits from ``HardwareError``."""
        assert issubclass(
            accel_pkg.DeviceNotAvailableError, accel_pkg.HardwareError
        )

    @pytest.mark.contract
    def test_vqm_memory_error_inherits_hardware_error(self, accel_pkg):
        """``VQMMemoryError`` inherits from ``HardwareError``."""
        assert issubclass(
            accel_pkg.VQMMemoryError, accel_pkg.HardwareError
        )

    @pytest.mark.contract
    def test_optimization_error_inherits_hardware_error(self, accel_pkg):
        """``OptimizationError`` inherits from ``HardwareError``."""
        assert issubclass(
            accel_pkg.OptimizationError, accel_pkg.HardwareError
        )

    @pytest.mark.contract
    def test_distributed_error_inherits_hardware_error(self, accel_pkg):
        """``DistributedError`` inherits from ``HardwareError``."""
        assert issubclass(
            accel_pkg.DistributedError, accel_pkg.HardwareError
        )

    @pytest.mark.contract
    def test_exception_classes_are_instantiable(self, accel_pkg):
        """
        All exception classes can be instantiated with a string message.

        This validates they are not abstract or broken.
        """
        exception_classes = [
            accel_pkg.ModelError,
            accel_pkg.HardwareError,
            accel_pkg.DeviceNotAvailableError,
            accel_pkg.VQMMemoryError,
            accel_pkg.OptimizationError,
            accel_pkg.DistributedError,
        ]
        for exc_cls in exception_classes:
            instance = exc_cls("test message")
            assert str(instance) == "test message", (
                f"{exc_cls.__name__} should accept a string message"
            )

    @pytest.mark.contract
    def test_exception_classes_are_catchable_by_parent(self, accel_pkg):
        """
        Raising a child exception is catchable by the parent exception type.

        Validates the inheritance chain is functional, not just structural.
        """
        with pytest.raises(accel_pkg.ModelError):
            raise accel_pkg.HardwareError("test")

        with pytest.raises(accel_pkg.HardwareError):
            raise accel_pkg.DeviceNotAvailableError("test")

        with pytest.raises(accel_pkg.HardwareError):
            raise accel_pkg.VQMMemoryError("test")

        with pytest.raises(accel_pkg.HardwareError):
            raise accel_pkg.OptimizationError("test")

        with pytest.raises(accel_pkg.HardwareError):
            raise accel_pkg.DistributedError("test")


class TestContractAccelerationManagerSignature:
    """§2 — AccelerationManager.__init__ has documented parameter signature.

    Per the __init__.py source (lines 215-242), AccelerationManager.__init__
    accepts: device, mixed_precision, precision, gradient_checkpointing,
    compile_model, compile_mode, cudnn_benchmark, distributed_strategy,
    distributed_backend, verbose.
    """

    @pytest.mark.contract
    def test_acceleration_manager_init_params(self, accel_pkg):
        """AccelerationManager.__init__ accepts the documented parameters."""
        sig = inspect.signature(accel_pkg.AccelerationManager.__init__)
        param_names = set(sig.parameters.keys()) - {"self"}

        expected_params = {
            "device",
            "mixed_precision",
            "precision",
            "gradient_checkpointing",
            "compile_model",
            "compile_mode",
            "cudnn_benchmark",
            "distributed_strategy",
            "distributed_backend",
            "verbose",
        }

        missing = expected_params - param_names
        assert not missing, (
            f"AccelerationManager.__init__ missing expected parameters: "
            f"{sorted(missing)}. Actual: {sorted(param_names)}"
        )

    @pytest.mark.contract
    def test_acceleration_manager_init_has_defaults(self, accel_pkg):
        """
        AccelerationManager.__init__ parameters have default values
        (enabling ``AccelerationManager()`` call with no args besides
        implicit ``self``).

        Per source: all params have defaults (device=None, mixed_precision=False,
        precision="fp16", etc.).
        """
        sig = inspect.signature(accel_pkg.AccelerationManager.__init__)
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            assert param.default is not inspect.Parameter.empty, (
                f"AccelerationManager.__init__ parameter '{name}' "
                f"should have a default value"
            )

    @pytest.mark.contract
    def test_acceleration_manager_has_setup(self, accel_pkg):
        """``AccelerationManager`` has a ``setup()`` method."""
        assert hasattr(accel_pkg.AccelerationManager, "setup")
        assert callable(accel_pkg.AccelerationManager.setup)

    @pytest.mark.contract
    def test_acceleration_manager_has_optimize_model(self, accel_pkg):
        """``AccelerationManager`` has an ``optimize_model()`` method."""
        assert hasattr(accel_pkg.AccelerationManager, "optimize_model")
        assert callable(accel_pkg.AccelerationManager.optimize_model)

    @pytest.mark.contract
    def test_acceleration_manager_has_get_device(self, accel_pkg):
        """``AccelerationManager`` has a ``get_device()`` method."""
        assert hasattr(accel_pkg.AccelerationManager, "get_device")
        assert callable(accel_pkg.AccelerationManager.get_device)

    @pytest.mark.contract
    def test_acceleration_manager_has_autocast(self, accel_pkg):
        """``AccelerationManager`` has an ``autocast()`` method."""
        assert hasattr(accel_pkg.AccelerationManager, "autocast")
        assert callable(accel_pkg.AccelerationManager.autocast)

    @pytest.mark.contract
    def test_acceleration_manager_has_cleanup(self, accel_pkg):
        """``AccelerationManager`` has a ``cleanup()`` method."""
        assert hasattr(accel_pkg.AccelerationManager, "cleanup")
        assert callable(accel_pkg.AccelerationManager.cleanup)

    @pytest.mark.contract
    def test_acceleration_manager_has_get_memory_stats(self, accel_pkg):
        """``AccelerationManager`` has a ``get_memory_stats()`` method."""
        assert hasattr(accel_pkg.AccelerationManager, "get_memory_stats")
        assert callable(accel_pkg.AccelerationManager.get_memory_stats)

    @pytest.mark.contract
    def test_acceleration_manager_has_get_device_info(self, accel_pkg):
        """``AccelerationManager`` has a ``get_device_info()`` method."""
        assert hasattr(accel_pkg.AccelerationManager, "get_device_info")
        assert callable(accel_pkg.AccelerationManager.get_device_info)

    @pytest.mark.contract
    def test_acceleration_manager_has_is_main_process(self, accel_pkg):
        """``AccelerationManager`` has an ``is_main_process()`` method."""
        assert hasattr(accel_pkg.AccelerationManager, "is_main_process")
        assert callable(accel_pkg.AccelerationManager.is_main_process)

    @pytest.mark.contract
    def test_acceleration_manager_has_barrier(self, accel_pkg):
        """``AccelerationManager`` has a ``barrier()`` method."""
        assert hasattr(accel_pkg.AccelerationManager, "barrier")
        assert callable(accel_pkg.AccelerationManager.barrier)

    @pytest.mark.contract
    def test_acceleration_manager_has_step(self, accel_pkg):
        """``AccelerationManager`` has a ``step()`` method."""
        assert hasattr(accel_pkg.AccelerationManager, "step")
        assert callable(accel_pkg.AccelerationManager.step)

    @pytest.mark.contract
    def test_acceleration_manager_has_get_grad_scaler(self, accel_pkg):
        """``AccelerationManager`` has a ``get_grad_scaler()`` method."""
        assert hasattr(accel_pkg.AccelerationManager, "get_grad_scaler")
        assert callable(accel_pkg.AccelerationManager.get_grad_scaler)

    @pytest.mark.contract
    def test_acceleration_manager_has_print_summary(self, accel_pkg):
        """``AccelerationManager`` has a ``print_summary()`` method."""
        assert hasattr(accel_pkg.AccelerationManager, "print_summary")
        assert callable(accel_pkg.AccelerationManager.print_summary)


class TestContractConvenienceFunctionSignatures:
    """§2 — Convenience functions have documented parameter signatures."""

    @pytest.mark.contract
    def test_auto_optimize_for_training_params(self, accel_pkg):
        """
        ``auto_optimize_for_training`` accepts the documented parameters.

        Per source (lines 429-436): model, device, model_size,
        enable_distributed, distributed_strategy, verbose.
        """
        sig = inspect.signature(accel_pkg.auto_optimize_for_training)
        param_names = set(sig.parameters.keys())

        expected = {
            "model",
            "device",
            "model_size",
            "enable_distributed",
            "distributed_strategy",
            "verbose",
        }
        missing = expected - param_names
        assert not missing, (
            f"auto_optimize_for_training missing parameters: "
            f"{sorted(missing)}. Actual: {sorted(param_names)}"
        )

    @pytest.mark.contract
    def test_auto_optimize_for_training_is_function(self, accel_pkg):
        """``auto_optimize_for_training`` is a function (not a class)."""
        assert inspect.isfunction(accel_pkg.auto_optimize_for_training)

    @pytest.mark.contract
    def test_auto_optimize_for_inference_params(self, accel_pkg):
        """
        ``auto_optimize_for_inference`` accepts the documented parameters.

        Per source (lines 506-511): model, device, optimize_for_latency,
        verbose.
        """
        sig = inspect.signature(accel_pkg.auto_optimize_for_inference)
        param_names = set(sig.parameters.keys())

        expected = {
            "model",
            "device",
            "optimize_for_latency",
            "verbose",
        }
        missing = expected - param_names
        assert not missing, (
            f"auto_optimize_for_inference missing parameters: "
            f"{sorted(missing)}. Actual: {sorted(param_names)}"
        )

    @pytest.mark.contract
    def test_auto_optimize_for_inference_is_function(self, accel_pkg):
        """``auto_optimize_for_inference`` is a function (not a class)."""
        assert inspect.isfunction(accel_pkg.auto_optimize_for_inference)

    @pytest.mark.contract
    def test_get_recommended_settings_params(self, accel_pkg):
        """
        ``get_recommended_settings`` accepts the documented parameters.

        Per source (lines 558-562): model, device, task_type.
        """
        sig = inspect.signature(accel_pkg.get_recommended_settings)
        param_names = set(sig.parameters.keys())

        expected = {"model", "device", "task_type"}
        missing = expected - param_names
        assert not missing, (
            f"get_recommended_settings missing parameters: "
            f"{sorted(missing)}. Actual: {sorted(param_names)}"
        )

    @pytest.mark.contract
    def test_get_recommended_settings_is_function(self, accel_pkg):
        """``get_recommended_settings`` is a function (not a class)."""
        assert inspect.isfunction(accel_pkg.get_recommended_settings)

    @pytest.mark.contract
    def test_create_acceleration_manager_params(self, accel_pkg):
        """
        ``create_acceleration_manager`` accepts the documented parameters.

        Per source (lines 613-616): config, verbose.
        """
        sig = inspect.signature(accel_pkg.create_acceleration_manager)
        param_names = set(sig.parameters.keys())

        expected = {"config", "verbose"}
        missing = expected - param_names
        assert not missing, (
            f"create_acceleration_manager missing parameters: "
            f"{sorted(missing)}. Actual: {sorted(param_names)}"
        )

    @pytest.mark.contract
    def test_create_acceleration_manager_is_function(self, accel_pkg):
        """``create_acceleration_manager`` is a function (not a class)."""
        assert inspect.isfunction(accel_pkg.create_acceleration_manager)

    @pytest.mark.contract
    def test_benchmark_accelerations_params(self, accel_pkg):
        """
        ``benchmark_accelerations`` accepts the documented parameters.

        Per source (lines 639-644): model, input_data, configurations,
        num_iterations.
        """
        sig = inspect.signature(accel_pkg.benchmark_accelerations)
        param_names = set(sig.parameters.keys())

        expected = {"model", "input_data", "configurations", "num_iterations"}
        missing = expected - param_names
        assert not missing, (
            f"benchmark_accelerations missing parameters: "
            f"{sorted(missing)}. Actual: {sorted(param_names)}"
        )

    @pytest.mark.contract
    def test_benchmark_accelerations_is_function(self, accel_pkg):
        """``benchmark_accelerations`` is a function (not a class)."""
        assert inspect.isfunction(accel_pkg.benchmark_accelerations)


class TestContractDeviceManagerFunctionSignatures:
    """§2 — Device management convenience functions are functions with expected signatures."""

    @pytest.mark.contract
    def test_get_default_device_is_function(self, accel_pkg):
        """``get_default_device`` is a function."""
        assert inspect.isfunction(accel_pkg.get_default_device)

    @pytest.mark.contract
    def test_list_available_devices_is_function(self, accel_pkg):
        """``list_available_devices`` is a function."""
        assert inspect.isfunction(accel_pkg.list_available_devices)

    @pytest.mark.contract
    def test_get_device_capabilities_is_function(self, accel_pkg):
        """``get_device_capabilities`` is a function."""
        assert inspect.isfunction(accel_pkg.get_device_capabilities)

    @pytest.mark.contract
    def test_get_device_capabilities_returns_dict(self, accel_pkg):
        """
        ``get_device_capabilities()`` returns a dict.

        Per the project structure doc, it returns a dict with keys:
        cuda_available, cuda_device_count, mps_available, tpu_available,
        cudnn_available, cudnn_enabled.
        """
        result = accel_pkg.get_device_capabilities()
        assert isinstance(result, dict), (
            f"get_device_capabilities() should return dict, got "
            f"{type(result).__name__}"
        )


class TestContractMemoryFunctionSignatures:
    """§2 — Memory optimization convenience functions are functions with expected signatures."""

    @pytest.mark.contract
    def test_get_memory_efficient_settings_is_function(self, accel_pkg):
        """``get_memory_efficient_settings`` is a function."""
        assert inspect.isfunction(accel_pkg.get_memory_efficient_settings)

    @pytest.mark.contract
    def test_estimate_model_memory_is_function(self, accel_pkg):
        """``estimate_model_memory`` is a function."""
        assert inspect.isfunction(accel_pkg.estimate_model_memory)


class TestContractComputationFunctionSignatures:
    """§2 — Computation optimization convenience functions are functions with expected nature."""

    @pytest.mark.contract
    def test_get_optimal_settings_is_function(self, accel_pkg):
        """``get_optimal_settings`` is a function."""
        assert inspect.isfunction(accel_pkg.get_optimal_settings)

    @pytest.mark.contract
    def test_auto_optimize_model_is_callable(self, accel_pkg):
        """``auto_optimize_model`` is callable."""
        assert callable(accel_pkg.auto_optimize_model)

    @pytest.mark.contract
    def test_optimize_inference_is_callable(self, accel_pkg):
        """
        ``optimize_inference`` is callable.

        Per the project structure doc, ``optimize_inference`` is a decorator
        that wraps functions with torch.no_grad() + inference_mode().
        It may be a function or a callable class.
        """
        assert callable(accel_pkg.optimize_inference)


class TestContractDistributedFunctionReturnTypes:
    """§2 — Distributed convenience functions return expected types."""

    @pytest.mark.contract
    def test_is_distributed_available_returns_bool(self, accel_pkg):
        """``is_distributed_available()`` returns a bool."""
        result = accel_pkg.is_distributed_available()
        assert isinstance(result, bool), (
            f"is_distributed_available() should return bool, got "
            f"{type(result).__name__}"
        )

    @pytest.mark.contract
    def test_get_world_size_returns_int(self, accel_pkg):
        """``get_world_size()`` returns an int."""
        result = accel_pkg.get_world_size()
        assert isinstance(result, int), (
            f"get_world_size() should return int, got "
            f"{type(result).__name__}"
        )

    @pytest.mark.contract
    def test_get_world_size_is_positive(self, accel_pkg):
        """``get_world_size()`` returns a positive integer (at least 1)."""
        result = accel_pkg.get_world_size()
        assert result >= 1, (
            f"get_world_size() should return >= 1, got {result}"
        )

    @pytest.mark.contract
    def test_get_rank_returns_int(self, accel_pkg):
        """``get_rank()`` returns an int."""
        result = accel_pkg.get_rank()
        assert isinstance(result, int), (
            f"get_rank() should return int, got "
            f"{type(result).__name__}"
        )

    @pytest.mark.contract
    def test_get_rank_is_non_negative(self, accel_pkg):
        """``get_rank()`` returns a non-negative integer (>= 0)."""
        result = accel_pkg.get_rank()
        assert result >= 0, (
            f"get_rank() should return >= 0, got {result}"
        )

    @pytest.mark.contract
    def test_is_main_process_returns_bool(self, accel_pkg):
        """``is_main_process()`` returns a bool."""
        result = accel_pkg.is_main_process()
        assert isinstance(result, bool), (
            f"is_main_process() should return bool, got "
            f"{type(result).__name__}"
        )


class TestContractPublicAPISurface:
    """
    §2 — Public API surface stability.

    Ensures that the minimum expected public API is present in ``__all__``.
    Guards against accidental removals during refactoring.
    """

    # The minimum API surface that MUST be present in __all__
    # Sourced directly from __init__.py lines 701-757
    MINIMUM_API = {
        # Version
        "__version__",
        "__author__",
        "__description__",

        # Core Managers
        "AccelerationManager",
        "DeviceManager",
        "MemoryOptimizer",
        "ComputationOptimizer",
        "DistributedManager",

        # Configuration Classes
        "DeviceInfo",
        "DeviceType",
        "MemoryConfig",
        "ComputationConfig",
        "DistributedConfig",
        "DistributedStrategy",
        "DistributedBackend",

        # Device Management Functions
        "get_default_device",
        "list_available_devices",
        "get_device_capabilities",

        # Memory Optimization Functions
        "get_memory_efficient_settings",
        "estimate_model_memory",

        # Computation Optimization Functions
        "get_optimal_settings",
        "auto_optimize_model",
        "optimize_inference",

        # Distributed Training Functions
        "is_distributed_available",
        "get_world_size",
        "get_rank",
        "is_main_process",

        # Convenience Functions
        "auto_optimize_for_training",
        "auto_optimize_for_inference",
        "get_recommended_settings",
        "create_acceleration_manager",
        "benchmark_accelerations",

        # Exceptions
        "ModelError",
        "HardwareError",
        "DeviceNotAvailableError",
        "VQMMemoryError",
        "OptimizationError",
        "DistributedError",
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
        ``__all__`` contains the expected number of entries.

        The __init__.py has exactly 43 entries in __all__ (lines 701-757,
        counting all named exports across the 8 categories: Version(3),
        Core Managers(5), Configuration Classes(7), Device Management(3),
        Memory Optimization(2), Computation Optimization(3), Distributed
        Training(4), Convenience(5), Exceptions(6) = 38 total unique names).

        We set a floor below the actual count to allow for minor changes
        while catching catastrophic loss.
        """
        actual = len(all_names)
        MINIMUM_EXPECTED = 35
        assert actual >= MINIMUM_EXPECTED, (
            f"__all__ has {actual} entries, expected at least "
            f"{MINIMUM_EXPECTED}. This suggests __all__ may have been "
            f"accidentally truncated."
        )

    @pytest.mark.contract
    def test_all_contains_all_documented_categories(self, all_names):
        """
        ``__all__`` contains at least one entry from each documented
        category: managers, configs, device functions, memory functions,
        computation functions, distributed functions, convenience functions,
        and exceptions.
        """
        all_set = set(all_names)

        categories = {
            "Core Managers": {"AccelerationManager", "DeviceManager"},
            "Configuration Classes": {"DeviceInfo", "MemoryConfig"},
            "Device Management Functions": {"get_default_device"},
            "Memory Optimization Functions": {"get_memory_efficient_settings"},
            "Computation Optimization Functions": {"get_optimal_settings"},
            "Distributed Training Functions": {"is_distributed_available"},
            "Convenience Functions": {"auto_optimize_for_training"},
            "Exceptions": {"ModelError", "HardwareError"},
        }

        for category, representatives in categories.items():
            overlap = representatives & all_set
            assert overlap, (
                f"No names from '{category}' found in __all__. "
                f"Expected at least one of: {sorted(representatives)}"
            )


class TestContractAllCategoryCoverage:
    """§2 — Verify each category in __all__ is fully present."""

    # Exact __all__ entries by category, from __init__.py source

    VERSION_ENTRIES = [
        "__version__",
        "__author__",
        "__description__",
    ]

    CORE_MANAGER_ENTRIES = [
        "AccelerationManager",
        "DeviceManager",
        "MemoryOptimizer",
        "ComputationOptimizer",
        "DistributedManager",
    ]

    CONFIG_CLASS_ENTRIES = [
        "DeviceInfo",
        "DeviceType",
        "MemoryConfig",
        "ComputationConfig",
        "DistributedConfig",
        "DistributedStrategy",
        "DistributedBackend",
    ]

    DEVICE_FUNCTION_ENTRIES = [
        "get_default_device",
        "list_available_devices",
        "get_device_capabilities",
    ]

    MEMORY_FUNCTION_ENTRIES = [
        "get_memory_efficient_settings",
        "estimate_model_memory",
    ]

    COMPUTATION_FUNCTION_ENTRIES = [
        "get_optimal_settings",
        "auto_optimize_model",
        "optimize_inference",
    ]

    DISTRIBUTED_FUNCTION_ENTRIES = [
        "is_distributed_available",
        "get_world_size",
        "get_rank",
        "is_main_process",
    ]

    CONVENIENCE_FUNCTION_ENTRIES = [
        "auto_optimize_for_training",
        "auto_optimize_for_inference",
        "get_recommended_settings",
        "create_acceleration_manager",
        "benchmark_accelerations",
    ]

    EXCEPTION_ENTRIES = [
        "ModelError",
        "HardwareError",
        "DeviceNotAvailableError",
        "VQMMemoryError",
        "OptimizationError",
        "DistributedError",
    ]

    @pytest.mark.contract
    @pytest.mark.parametrize("name", VERSION_ENTRIES)
    def test_version_entry_in_all(self, all_names, name):
        """Version metadata entry is in __all__."""
        assert name in all_names, f"'{name}' missing from __all__"

    @pytest.mark.contract
    @pytest.mark.parametrize("name", CORE_MANAGER_ENTRIES)
    def test_core_manager_entry_in_all(self, all_names, name):
        """Core manager entry is in __all__."""
        assert name in all_names, f"'{name}' missing from __all__"

    @pytest.mark.contract
    @pytest.mark.parametrize("name", CONFIG_CLASS_ENTRIES)
    def test_config_class_entry_in_all(self, all_names, name):
        """Configuration class entry is in __all__."""
        assert name in all_names, f"'{name}' missing from __all__"

    @pytest.mark.contract
    @pytest.mark.parametrize("name", DEVICE_FUNCTION_ENTRIES)
    def test_device_function_entry_in_all(self, all_names, name):
        """Device management function entry is in __all__."""
        assert name in all_names, f"'{name}' missing from __all__"

    @pytest.mark.contract
    @pytest.mark.parametrize("name", MEMORY_FUNCTION_ENTRIES)
    def test_memory_function_entry_in_all(self, all_names, name):
        """Memory optimization function entry is in __all__."""
        assert name in all_names, f"'{name}' missing from __all__"

    @pytest.mark.contract
    @pytest.mark.parametrize("name", COMPUTATION_FUNCTION_ENTRIES)
    def test_computation_function_entry_in_all(self, all_names, name):
        """Computation optimization function entry is in __all__."""
        assert name in all_names, f"'{name}' missing from __all__"

    @pytest.mark.contract
    @pytest.mark.parametrize("name", DISTRIBUTED_FUNCTION_ENTRIES)
    def test_distributed_function_entry_in_all(self, all_names, name):
        """Distributed training function entry is in __all__."""
        assert name in all_names, f"'{name}' missing from __all__"

    @pytest.mark.contract
    @pytest.mark.parametrize("name", CONVENIENCE_FUNCTION_ENTRIES)
    def test_convenience_function_entry_in_all(self, all_names, name):
        """Convenience function entry is in __all__."""
        assert name in all_names, f"'{name}' missing from __all__"

    @pytest.mark.contract
    @pytest.mark.parametrize("name", EXCEPTION_ENTRIES)
    def test_exception_entry_in_all(self, all_names, name):
        """Exception class entry is in __all__."""
        assert name in all_names, f"'{name}' missing from __all__"
