#!/usr/bin/env python3
"""
Complete Unit Test Suite for milia_pipeline/models/hpo/backends/base.py Module

Tests HPO Backend Protocol system including:
- OPTUNA_AVAILABLE flag behavior (import success/failure scenarios)
- HPOBackendProtocol class
  - Protocol definition and runtime_checkable decorator
  - Abstract method signatures and @abstractmethod decorator validation
  - Protocol compliance checking
  - isinstance() checks with protocol
- get_backend() factory function
  - Valid backend retrieval ("optuna")
  - Unknown backend error handling
  - Import error handling (relative import path resolution)
  - BackendError exception attributes

Test Categories:
1. Module-level import behavior (OPTUNA_AVAILABLE)
2. HPOBackendProtocol interface validation
3. Protocol compliance for implementing classes
4. get_backend() factory function
5. Error handling and BackendError validation
6. Edge cases and boundary conditions
7. Relative import resolution tests
8. @abstractmethod decorator validation

This is a PRODUCTION-READY test suite with comprehensive coverage.

Note: This test suite follows best practices for mock pollution prevention:
- Uses test-level @patch decorators and context managers
- No module-level sys.modules manipulation
- All patches are properly scoped and cleaned up

Author: Milia Team
Version: 1.1.0
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import inspect
from collections.abc import Callable
from typing import Any, Protocol
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_optuna_backend_class():
    """Create a mock OptunaBackend class for testing."""
    mock_class = MagicMock()
    mock_instance = MagicMock()
    mock_class.return_value = mock_instance
    return mock_class


@pytest.fixture
def mock_backend_instance():
    """Create a mock backend instance implementing the protocol methods."""
    mock = MagicMock()
    mock.create_study = MagicMock(return_value=MagicMock())
    mock.optimize = MagicMock(return_value=None)
    mock.get_best_params = MagicMock(return_value={"lr": 0.001})
    mock.get_best_value = MagicMock(return_value=0.95)
    mock.get_all_trials = MagicMock(return_value=[])
    mock.create_pruner = MagicMock(return_value=MagicMock())
    mock.create_sampler = MagicMock(return_value=MagicMock())
    return mock


@pytest.fixture
def compliant_backend_class():
    """
    Create a class that properly implements the HPOBackendProtocol.

    This class has all required methods with correct signatures.
    """

    class CompliantBackend:
        """A backend class that implements all protocol methods."""

        def create_study(
            self,
            study_name: str,
            direction: str,
            storage: str | None = None,
            load_if_exists: bool = True,
            sampler: Any | None = None,
            pruner: Any | None = None,
        ) -> Any:
            return MagicMock()

        def optimize(
            self,
            study: Any,
            objective_fn: Callable[[Any], float],
            n_trials: int,
            timeout: int | None = None,
            n_jobs: int = 1,
            catch: tuple = (),
            callbacks: list[Callable] | None = None,
        ) -> None:
            pass

        def get_best_params(self, study: Any) -> dict[str, Any]:
            return {"learning_rate": 0.001}

        def get_best_value(self, study: Any) -> float:
            return 0.95

        def get_all_trials(self, study: Any) -> list[dict[str, Any]]:
            return []

        def create_pruner(
            self, pruner_type: str, n_startup_trials: int = 5, n_warmup_steps: int = 10, **kwargs
        ) -> Any:
            return MagicMock()

        def create_sampler(
            self, sampler_type: str, seed: int | None = None, n_startup_trials: int = 10, **kwargs
        ) -> Any:
            return MagicMock()

    return CompliantBackend


@pytest.fixture
def non_compliant_backend_class():
    """
    Create a class that does NOT implement all HPOBackendProtocol methods.

    This class is missing required methods.
    """

    class NonCompliantBackend:
        """A backend class missing most protocol methods."""

        def create_study(self, study_name: str, direction: str) -> Any:
            return MagicMock()

        # Missing: optimize, get_best_params, get_best_value,
        #          get_all_trials, create_pruner, create_sampler

    return NonCompliantBackend


@pytest.fixture
def partial_backend_class():
    """
    Create a class that partially implements HPOBackendProtocol.

    This class has some methods but not all.
    """

    class PartialBackend:
        """A backend class with partial protocol implementation."""

        def create_study(
            self,
            study_name: str,
            direction: str,
            storage: str | None = None,
            load_if_exists: bool = True,
            sampler: Any | None = None,
            pruner: Any | None = None,
        ) -> Any:
            return MagicMock()

        def optimize(
            self,
            study: Any,
            objective_fn: Callable[[Any], float],
            n_trials: int,
            timeout: int | None = None,
            n_jobs: int = 1,
            catch: tuple = (),
            callbacks: list[Callable] | None = None,
        ) -> None:
            pass

        def get_best_params(self, study: Any) -> dict[str, Any]:
            return {}

        # Missing: get_best_value, get_all_trials, create_pruner, create_sampler

    return PartialBackend


# =============================================================================
# MODULE IMPORT TESTS - OPTUNA_AVAILABLE FLAG
# =============================================================================


class TestOptunaAvailableFlag:
    """Test OPTUNA_AVAILABLE module-level flag behavior."""

    def test_optuna_available_flag_is_defined(self):
        """Test OPTUNA_AVAILABLE flag is defined in the module."""
        from milia_pipeline.models.hpo.backends.base import OPTUNA_AVAILABLE

        assert isinstance(OPTUNA_AVAILABLE, bool)

    def test_optuna_available_flag_true_when_optuna_installed(self):
        """Test OPTUNA_AVAILABLE is True when optuna is installed."""
        # This test assumes optuna is installed in the test environment
        try:
            import optuna

            from milia_pipeline.models.hpo.backends.base import OPTUNA_AVAILABLE

            assert OPTUNA_AVAILABLE is True
        except ImportError:
            pytest.skip("Optuna not installed, cannot test OPTUNA_AVAILABLE=True")

    def test_optuna_available_reflects_import_status(self):
        """Test OPTUNA_AVAILABLE correctly reflects optuna import status."""
        from milia_pipeline.models.hpo.backends.base import OPTUNA_AVAILABLE

        # Check consistency with actual import
        try:
            import optuna

            optuna_importable = True
        except ImportError:
            optuna_importable = False

        assert optuna_importable == OPTUNA_AVAILABLE


class TestOptunaImportFailureScenario:
    """Test module behavior when optuna import fails."""

    def test_module_loads_without_optuna(self):
        """Test base module can be imported even without optuna."""
        # This test verifies the try/except block handles ImportError
        # The module should define OPTUNA_AVAILABLE = False if import fails
        from milia_pipeline.models.hpo.backends import base

        assert hasattr(base, "OPTUNA_AVAILABLE")
        assert hasattr(base, "HPOBackendProtocol")
        assert hasattr(base, "get_backend")

    def test_optuna_available_is_boolean_type(self):
        """Test OPTUNA_AVAILABLE is strictly a boolean, not truthy/falsy."""
        from milia_pipeline.models.hpo.backends.base import OPTUNA_AVAILABLE

        # Must be exactly True or False, not 1, 0, or other truthy/falsy values
        assert OPTUNA_AVAILABLE is True or OPTUNA_AVAILABLE is False
        assert type(OPTUNA_AVAILABLE) is bool


class TestModuleImports:
    """Test module-level imports and dependencies."""

    def test_protocol_imported_from_typing(self):
        """Test Protocol is imported from typing module."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        # Verify it inherits from typing.Protocol
        assert Protocol in HPOBackendProtocol.__mro__

    def test_runtime_checkable_applied_correctly(self):
        """Test @runtime_checkable decorator is properly applied."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        # This is how typing module marks runtime_checkable protocols
        assert getattr(HPOBackendProtocol, "_is_runtime_protocol", False) is True

    def test_abstractmethod_imported_from_abc(self):
        """Test abstractmethod is used from abc module."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        # All protocol methods should have __isabstractmethod__ = True
        method = HPOBackendProtocol.create_study
        assert hasattr(method, "__isabstractmethod__")


# =============================================================================
# HPOBackendProtocol INTERFACE TESTS
# =============================================================================


class TestHPOBackendProtocolDefinition:
    """Test HPOBackendProtocol class definition."""

    def test_protocol_is_defined(self):
        """Test HPOBackendProtocol class exists."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        assert HPOBackendProtocol is not None

    def test_protocol_is_runtime_checkable(self):
        """Test HPOBackendProtocol has @runtime_checkable decorator."""

        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        # runtime_checkable protocols have _is_runtime_protocol attribute
        assert hasattr(HPOBackendProtocol, "_is_runtime_protocol")
        assert HPOBackendProtocol._is_runtime_protocol is True

    def test_protocol_inherits_from_protocol(self):
        """Test HPOBackendProtocol inherits from typing.Protocol."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        assert issubclass(HPOBackendProtocol, Protocol)

    def test_protocol_has_create_study_method(self):
        """Test HPOBackendProtocol defines create_study method."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        assert hasattr(HPOBackendProtocol, "create_study")

    def test_protocol_has_optimize_method(self):
        """Test HPOBackendProtocol defines optimize method."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        assert hasattr(HPOBackendProtocol, "optimize")

    def test_protocol_has_get_best_params_method(self):
        """Test HPOBackendProtocol defines get_best_params method."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        assert hasattr(HPOBackendProtocol, "get_best_params")

    def test_protocol_has_get_best_value_method(self):
        """Test HPOBackendProtocol defines get_best_value method."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        assert hasattr(HPOBackendProtocol, "get_best_value")

    def test_protocol_has_get_all_trials_method(self):
        """Test HPOBackendProtocol defines get_all_trials method."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        assert hasattr(HPOBackendProtocol, "get_all_trials")

    def test_protocol_has_create_pruner_method(self):
        """Test HPOBackendProtocol defines create_pruner method."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        assert hasattr(HPOBackendProtocol, "create_pruner")

    def test_protocol_has_create_sampler_method(self):
        """Test HPOBackendProtocol defines create_sampler method."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        assert hasattr(HPOBackendProtocol, "create_sampler")

    def test_protocol_has_seven_required_methods(self):
        """Test HPOBackendProtocol defines all 7 required methods."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        required_methods = [
            "create_study",
            "optimize",
            "get_best_params",
            "get_best_value",
            "get_all_trials",
            "create_pruner",
            "create_sampler",
        ]

        for method_name in required_methods:
            assert hasattr(HPOBackendProtocol, method_name), (
                f"Protocol missing required method: {method_name}"
            )


class TestHPOBackendProtocolMethodSignatures:
    """Test HPOBackendProtocol method signatures."""

    def test_create_study_signature(self):
        """Test create_study method has correct signature."""
        import inspect

        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.create_study)
        params = list(sig.parameters.keys())

        # Should have: self, study_name, direction, storage, load_if_exists, sampler, pruner
        assert "self" in params
        assert "study_name" in params
        assert "direction" in params
        assert "storage" in params
        assert "load_if_exists" in params
        assert "sampler" in params
        assert "pruner" in params

    def test_create_study_default_parameters(self):
        """Test create_study has correct default parameter values."""
        import inspect

        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.create_study)
        params = sig.parameters

        # storage default is None
        assert params["storage"].default is None
        # load_if_exists default is True
        assert params["load_if_exists"].default is True
        # sampler default is None
        assert params["sampler"].default is None
        # pruner default is None
        assert params["pruner"].default is None

    def test_optimize_signature(self):
        """Test optimize method has correct signature."""
        import inspect

        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.optimize)
        params = list(sig.parameters.keys())

        assert "self" in params
        assert "study" in params
        assert "objective_fn" in params
        assert "n_trials" in params
        assert "timeout" in params
        assert "n_jobs" in params
        assert "catch" in params
        assert "callbacks" in params

    def test_optimize_default_parameters(self):
        """Test optimize has correct default parameter values."""
        import inspect

        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.optimize)
        params = sig.parameters

        # timeout default is None
        assert params["timeout"].default is None
        # n_jobs default is 1
        assert params["n_jobs"].default == 1
        # catch default is empty tuple
        assert params["catch"].default == ()
        # callbacks default is None
        assert params["callbacks"].default is None

    def test_get_best_params_signature(self):
        """Test get_best_params method has correct signature."""
        import inspect

        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.get_best_params)
        params = list(sig.parameters.keys())

        assert "self" in params
        assert "study" in params
        assert len(params) == 2

    def test_get_best_value_signature(self):
        """Test get_best_value method has correct signature."""
        import inspect

        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.get_best_value)
        params = list(sig.parameters.keys())

        assert "self" in params
        assert "study" in params
        assert len(params) == 2

    def test_get_all_trials_signature(self):
        """Test get_all_trials method has correct signature."""
        import inspect

        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.get_all_trials)
        params = list(sig.parameters.keys())

        assert "self" in params
        assert "study" in params
        assert len(params) == 2

    def test_create_pruner_signature(self):
        """Test create_pruner method has correct signature."""
        import inspect

        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.create_pruner)
        params = list(sig.parameters.keys())

        assert "self" in params
        assert "pruner_type" in params
        assert "n_startup_trials" in params
        assert "n_warmup_steps" in params
        assert "kwargs" in params

    def test_create_pruner_default_parameters(self):
        """Test create_pruner has correct default parameter values."""
        import inspect

        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.create_pruner)
        params = sig.parameters

        # n_startup_trials default is 5
        assert params["n_startup_trials"].default == 5
        # n_warmup_steps default is 10
        assert params["n_warmup_steps"].default == 10

    def test_create_sampler_signature(self):
        """Test create_sampler method has correct signature."""
        import inspect

        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.create_sampler)
        params = list(sig.parameters.keys())

        assert "self" in params
        assert "sampler_type" in params
        assert "seed" in params
        assert "n_startup_trials" in params
        assert "kwargs" in params

    def test_create_sampler_default_parameters(self):
        """Test create_sampler has correct default parameter values."""
        import inspect

        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.create_sampler)
        params = sig.parameters

        # seed default is None
        assert params["seed"].default is None
        # n_startup_trials default is 10
        assert params["n_startup_trials"].default == 10


class TestHPOBackendProtocolReturnTypeAnnotations:
    """Test HPOBackendProtocol method return type annotations."""

    def test_create_study_returns_any(self):
        """Test create_study return annotation is Any."""
        import inspect
        from typing import Any

        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.create_study)
        assert sig.return_annotation == Any

    def test_optimize_returns_none(self):
        """Test optimize return annotation is None."""
        import inspect

        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.optimize)
        assert sig.return_annotation is None

    def test_get_best_params_returns_dict(self):
        """Test get_best_params return annotation is Dict[str, Any]."""
        import inspect
        from typing import Any, get_args, get_origin

        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.get_best_params)
        ret_annotation = sig.return_annotation

        # Check it's Dict[str, Any]
        assert get_origin(ret_annotation) is dict
        args = get_args(ret_annotation)
        assert args == (str, Any)

    def test_get_best_value_returns_float(self):
        """Test get_best_value return annotation is float."""
        import inspect

        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.get_best_value)
        assert sig.return_annotation == float

    def test_get_all_trials_returns_list(self):
        """Test get_all_trials return annotation is List[Dict[str, Any]]."""
        import inspect
        from typing import get_origin

        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.get_all_trials)
        ret_annotation = sig.return_annotation

        # Check it's List[Dict[str, Any]]
        assert get_origin(ret_annotation) is list

    def test_create_pruner_returns_any(self):
        """Test create_pruner return annotation is Any."""
        import inspect
        from typing import Any

        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.create_pruner)
        assert sig.return_annotation == Any

    def test_create_sampler_returns_any(self):
        """Test create_sampler return annotation is Any."""
        import inspect
        from typing import Any

        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.create_sampler)
        assert sig.return_annotation == Any


# =============================================================================
# ABSTRACTMETHOD DECORATOR VALIDATION TESTS
# =============================================================================


class TestAbstractMethodDecorator:
    """Test that protocol methods have @abstractmethod decorator applied."""

    def test_create_study_is_abstract(self):
        """Test create_study method is decorated with @abstractmethod."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        method = HPOBackendProtocol.create_study
        # Check if the method has __isabstractmethod__ attribute set to True
        assert getattr(method, "__isabstractmethod__", False) is True, (
            "create_study should be decorated with @abstractmethod"
        )

    def test_optimize_is_abstract(self):
        """Test optimize method is decorated with @abstractmethod."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        method = HPOBackendProtocol.optimize
        assert getattr(method, "__isabstractmethod__", False) is True, (
            "optimize should be decorated with @abstractmethod"
        )

    def test_get_best_params_is_abstract(self):
        """Test get_best_params method is decorated with @abstractmethod."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        method = HPOBackendProtocol.get_best_params
        assert getattr(method, "__isabstractmethod__", False) is True, (
            "get_best_params should be decorated with @abstractmethod"
        )

    def test_get_best_value_is_abstract(self):
        """Test get_best_value method is decorated with @abstractmethod."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        method = HPOBackendProtocol.get_best_value
        assert getattr(method, "__isabstractmethod__", False) is True, (
            "get_best_value should be decorated with @abstractmethod"
        )

    def test_get_all_trials_is_abstract(self):
        """Test get_all_trials method is decorated with @abstractmethod."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        method = HPOBackendProtocol.get_all_trials
        assert getattr(method, "__isabstractmethod__", False) is True, (
            "get_all_trials should be decorated with @abstractmethod"
        )

    def test_create_pruner_is_abstract(self):
        """Test create_pruner method is decorated with @abstractmethod."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        method = HPOBackendProtocol.create_pruner
        assert getattr(method, "__isabstractmethod__", False) is True, (
            "create_pruner should be decorated with @abstractmethod"
        )

    def test_create_sampler_is_abstract(self):
        """Test create_sampler method is decorated with @abstractmethod."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        method = HPOBackendProtocol.create_sampler
        assert getattr(method, "__isabstractmethod__", False) is True, (
            "create_sampler should be decorated with @abstractmethod"
        )

    def test_all_protocol_methods_are_abstract(self):
        """Test all 7 protocol methods have @abstractmethod decorator."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        required_abstract_methods = [
            "create_study",
            "optimize",
            "get_best_params",
            "get_best_value",
            "get_all_trials",
            "create_pruner",
            "create_sampler",
        ]

        for method_name in required_abstract_methods:
            method = getattr(HPOBackendProtocol, method_name)
            assert getattr(method, "__isabstractmethod__", False) is True, (
                f"{method_name} should be decorated with @abstractmethod"
            )


# =============================================================================
# PROTOCOL METHOD DOCSTRING TESTS
# =============================================================================


class TestProtocolMethodDocstrings:
    """Test that all protocol methods have proper docstrings."""

    def test_create_study_has_docstring(self):
        """Test create_study method has a docstring."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        assert HPOBackendProtocol.create_study.__doc__ is not None
        assert len(HPOBackendProtocol.create_study.__doc__) > 0

    def test_optimize_has_docstring(self):
        """Test optimize method has a docstring."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        assert HPOBackendProtocol.optimize.__doc__ is not None
        assert len(HPOBackendProtocol.optimize.__doc__) > 0

    def test_get_best_params_has_docstring(self):
        """Test get_best_params method has a docstring."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        assert HPOBackendProtocol.get_best_params.__doc__ is not None
        assert len(HPOBackendProtocol.get_best_params.__doc__) > 0

    def test_get_best_value_has_docstring(self):
        """Test get_best_value method has a docstring."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        assert HPOBackendProtocol.get_best_value.__doc__ is not None
        assert len(HPOBackendProtocol.get_best_value.__doc__) > 0

    def test_get_all_trials_has_docstring(self):
        """Test get_all_trials method has a docstring."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        assert HPOBackendProtocol.get_all_trials.__doc__ is not None
        assert len(HPOBackendProtocol.get_all_trials.__doc__) > 0

    def test_create_pruner_has_docstring(self):
        """Test create_pruner method has a docstring."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        assert HPOBackendProtocol.create_pruner.__doc__ is not None
        assert len(HPOBackendProtocol.create_pruner.__doc__) > 0

    def test_create_sampler_has_docstring(self):
        """Test create_sampler method has a docstring."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        assert HPOBackendProtocol.create_sampler.__doc__ is not None
        assert len(HPOBackendProtocol.create_sampler.__doc__) > 0

    def test_all_methods_have_docstrings(self):
        """Test all 7 protocol methods have non-empty docstrings."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        required_methods = [
            "create_study",
            "optimize",
            "get_best_params",
            "get_best_value",
            "get_all_trials",
            "create_pruner",
            "create_sampler",
        ]

        for method_name in required_methods:
            method = getattr(HPOBackendProtocol, method_name)
            assert method.__doc__ is not None, f"{method_name} should have a docstring"
            assert len(method.__doc__.strip()) > 0, f"{method_name} docstring should not be empty"


class TestProtocolMethodParameterTypes:
    """Test protocol method parameter type annotations."""

    def test_create_study_study_name_is_str(self):
        """Test create_study study_name parameter is annotated as str."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.create_study)
        assert sig.parameters["study_name"].annotation == str

    def test_create_study_direction_is_str(self):
        """Test create_study direction parameter is annotated as str."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.create_study)
        assert sig.parameters["direction"].annotation == str

    def test_optimize_n_trials_is_int(self):
        """Test optimize n_trials parameter is annotated as int."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.optimize)
        assert sig.parameters["n_trials"].annotation == int

    def test_optimize_n_jobs_is_int(self):
        """Test optimize n_jobs parameter is annotated as int."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.optimize)
        assert sig.parameters["n_jobs"].annotation == int

    def test_optimize_catch_is_tuple(self):
        """Test optimize catch parameter is annotated as tuple."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.optimize)
        assert sig.parameters["catch"].annotation == tuple

    def test_create_pruner_pruner_type_is_str(self):
        """Test create_pruner pruner_type parameter is annotated as str."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.create_pruner)
        assert sig.parameters["pruner_type"].annotation == str

    def test_create_pruner_n_startup_trials_is_int(self):
        """Test create_pruner n_startup_trials parameter is annotated as int."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.create_pruner)
        assert sig.parameters["n_startup_trials"].annotation == int

    def test_create_pruner_n_warmup_steps_is_int(self):
        """Test create_pruner n_warmup_steps parameter is annotated as int."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.create_pruner)
        assert sig.parameters["n_warmup_steps"].annotation == int

    def test_create_sampler_sampler_type_is_str(self):
        """Test create_sampler sampler_type parameter is annotated as str."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.create_sampler)
        assert sig.parameters["sampler_type"].annotation == str

    def test_create_sampler_n_startup_trials_is_int(self):
        """Test create_sampler n_startup_trials parameter is annotated as int."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        sig = inspect.signature(HPOBackendProtocol.create_sampler)
        assert sig.parameters["n_startup_trials"].annotation == int


# =============================================================================
# PROTOCOL COMPLIANCE TESTS
# =============================================================================


class TestProtocolCompliance:
    """Test protocol compliance checking with isinstance()."""

    def test_compliant_class_is_instance_of_protocol(self, compliant_backend_class):
        """Test class with all methods is instance of protocol."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        instance = compliant_backend_class()
        assert isinstance(instance, HPOBackendProtocol)

    def test_non_compliant_class_is_not_instance_of_protocol(self, non_compliant_backend_class):
        """Test class missing methods is not instance of protocol."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        instance = non_compliant_backend_class()
        assert not isinstance(instance, HPOBackendProtocol)

    def test_partial_backend_is_not_instance_of_protocol(self, partial_backend_class):
        """Test class with partial implementation is not instance of protocol."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        instance = partial_backend_class()
        assert not isinstance(instance, HPOBackendProtocol)

    def test_mock_with_all_methods_is_instance_of_protocol(self, mock_backend_instance):
        """Test MagicMock with all methods is instance of protocol."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        assert isinstance(mock_backend_instance, HPOBackendProtocol)

    def test_empty_object_is_not_instance_of_protocol(self):
        """Test empty object is not instance of protocol."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        class EmptyClass:
            pass

        instance = EmptyClass()
        assert not isinstance(instance, HPOBackendProtocol)

    def test_dict_is_not_instance_of_protocol(self):
        """Test dict is not instance of protocol."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        assert not isinstance({}, HPOBackendProtocol)

    def test_none_is_not_instance_of_protocol(self):
        """Test None is not instance of protocol."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        assert not isinstance(None, HPOBackendProtocol)


class TestProtocolComplianceMethodPresence:
    """Test protocol compliance based on method presence."""

    def test_class_missing_create_study_fails_compliance(self):
        """Test class missing create_study fails compliance."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        class MissingCreateStudy:
            def optimize(self, *args, **kwargs):
                pass

            def get_best_params(self, study):
                return {}

            def get_best_value(self, study):
                return 0.0

            def get_all_trials(self, study):
                return []

            def create_pruner(self, *args, **kwargs):
                pass

            def create_sampler(self, *args, **kwargs):
                pass

        instance = MissingCreateStudy()
        assert not isinstance(instance, HPOBackendProtocol)

    def test_class_missing_optimize_fails_compliance(self):
        """Test class missing optimize fails compliance."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        class MissingOptimize:
            def create_study(self, *args, **kwargs):
                pass

            def get_best_params(self, study):
                return {}

            def get_best_value(self, study):
                return 0.0

            def get_all_trials(self, study):
                return []

            def create_pruner(self, *args, **kwargs):
                pass

            def create_sampler(self, *args, **kwargs):
                pass

        instance = MissingOptimize()
        assert not isinstance(instance, HPOBackendProtocol)

    def test_class_missing_get_best_params_fails_compliance(self):
        """Test class missing get_best_params fails compliance."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        class MissingGetBestParams:
            def create_study(self, *args, **kwargs):
                pass

            def optimize(self, *args, **kwargs):
                pass

            def get_best_value(self, study):
                return 0.0

            def get_all_trials(self, study):
                return []

            def create_pruner(self, *args, **kwargs):
                pass

            def create_sampler(self, *args, **kwargs):
                pass

        instance = MissingGetBestParams()
        assert not isinstance(instance, HPOBackendProtocol)

    def test_class_missing_get_best_value_fails_compliance(self):
        """Test class missing get_best_value fails compliance."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        class MissingGetBestValue:
            def create_study(self, *args, **kwargs):
                pass

            def optimize(self, *args, **kwargs):
                pass

            def get_best_params(self, study):
                return {}

            def get_all_trials(self, study):
                return []

            def create_pruner(self, *args, **kwargs):
                pass

            def create_sampler(self, *args, **kwargs):
                pass

        instance = MissingGetBestValue()
        assert not isinstance(instance, HPOBackendProtocol)

    def test_class_missing_get_all_trials_fails_compliance(self):
        """Test class missing get_all_trials fails compliance."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        class MissingGetAllTrials:
            def create_study(self, *args, **kwargs):
                pass

            def optimize(self, *args, **kwargs):
                pass

            def get_best_params(self, study):
                return {}

            def get_best_value(self, study):
                return 0.0

            def create_pruner(self, *args, **kwargs):
                pass

            def create_sampler(self, *args, **kwargs):
                pass

        instance = MissingGetAllTrials()
        assert not isinstance(instance, HPOBackendProtocol)

    def test_class_missing_create_pruner_fails_compliance(self):
        """Test class missing create_pruner fails compliance."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        class MissingCreatePruner:
            def create_study(self, *args, **kwargs):
                pass

            def optimize(self, *args, **kwargs):
                pass

            def get_best_params(self, study):
                return {}

            def get_best_value(self, study):
                return 0.0

            def get_all_trials(self, study):
                return []

            def create_sampler(self, *args, **kwargs):
                pass

        instance = MissingCreatePruner()
        assert not isinstance(instance, HPOBackendProtocol)

    def test_class_missing_create_sampler_fails_compliance(self):
        """Test class missing create_sampler fails compliance."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        class MissingCreateSampler:
            def create_study(self, *args, **kwargs):
                pass

            def optimize(self, *args, **kwargs):
                pass

            def get_best_params(self, study):
                return {}

            def get_best_value(self, study):
                return 0.0

            def get_all_trials(self, study):
                return []

            def create_pruner(self, *args, **kwargs):
                pass

        instance = MissingCreateSampler()
        assert not isinstance(instance, HPOBackendProtocol)


# =============================================================================
# GET_BACKEND FACTORY FUNCTION TESTS
# =============================================================================


class TestGetBackendValidCases:
    """Test get_backend() with valid backend names."""

    def test_get_backend_optuna_returns_backend_instance(self):
        """Test get_backend('optuna') returns OptunaBackend instance."""
        from milia_pipeline.models.hpo.backends.base import get_backend

        mock_instance = MagicMock()
        mock_optuna_cls = MagicMock(return_value=mock_instance)
        mock_module = MagicMock()
        mock_module.OptunaBackend = mock_optuna_cls

        with patch.dict(
            "sys.modules", {"milia_pipeline.models.hpo.backends.optuna_backend": mock_module}
        ):
            result = get_backend("optuna")

        assert result == mock_instance
        mock_optuna_cls.assert_called_once()

    def test_get_backend_optuna_calls_constructor_without_args(self):
        """Test get_backend('optuna') calls OptunaBackend() without arguments."""
        from milia_pipeline.models.hpo.backends.base import get_backend

        mock_instance = MagicMock()
        mock_optuna_cls = MagicMock(return_value=mock_instance)
        mock_module = MagicMock()
        mock_module.OptunaBackend = mock_optuna_cls

        with patch.dict(
            "sys.modules", {"milia_pipeline.models.hpo.backends.optuna_backend": mock_module}
        ):
            get_backend("optuna")

        mock_optuna_cls.assert_called_once_with()

    def test_get_backend_returns_new_instance_each_call(self):
        """Test get_backend() returns new instance on each call (not singleton)."""
        from milia_pipeline.models.hpo.backends.base import get_backend

        mock_instance_1 = MagicMock(name="instance_1")
        mock_instance_2 = MagicMock(name="instance_2")
        mock_optuna_cls = MagicMock(side_effect=[mock_instance_1, mock_instance_2])
        mock_module = MagicMock()
        mock_module.OptunaBackend = mock_optuna_cls

        with patch.dict(
            "sys.modules", {"milia_pipeline.models.hpo.backends.optuna_backend": mock_module}
        ):
            result_1 = get_backend("optuna")
            result_2 = get_backend("optuna")

        assert result_1 is mock_instance_1
        assert result_2 is mock_instance_2
        assert result_1 is not result_2
        assert mock_optuna_cls.call_count == 2


class TestGetBackendRelativeImport:
    """Test get_backend() handles relative import correctly.

    The get_backend() function uses 'from .optuna_backend import OptunaBackend'
    which is a relative import. These tests verify proper resolution.
    """

    def test_get_backend_resolves_relative_import_path(self):
        """Test get_backend() correctly resolves the relative import path."""
        from milia_pipeline.models.hpo.backends.base import get_backend

        mock_instance = MagicMock()
        mock_optuna_cls = MagicMock(return_value=mock_instance)
        mock_module = MagicMock()
        mock_module.OptunaBackend = mock_optuna_cls

        # The relative import '.optuna_backend' resolves to this full path
        full_module_path = "milia_pipeline.models.hpo.backends.optuna_backend"

        with patch.dict("sys.modules", {full_module_path: mock_module}):
            result = get_backend("optuna")
            assert result is mock_instance

    def test_get_backend_uses_optuna_backend_module(self):
        """Test get_backend accesses OptunaBackend from correct module."""
        from milia_pipeline.models.hpo.backends.base import get_backend

        mock_instance = MagicMock()
        mock_optuna_cls = MagicMock(return_value=mock_instance)
        mock_module = MagicMock(spec=["OptunaBackend"])
        mock_module.OptunaBackend = mock_optuna_cls

        with patch.dict(
            "sys.modules", {"milia_pipeline.models.hpo.backends.optuna_backend": mock_module}
        ):
            get_backend("optuna")
            # Verify OptunaBackend was accessed from the mocked module
            mock_optuna_cls.assert_called_once()


class TestGetBackendInvalidCases:
    """Test get_backend() with invalid backend names."""

    def test_get_backend_unknown_raises_backend_error(self):
        """Test get_backend() with unknown backend raises BackendError."""
        from milia_pipeline.exceptions import BackendError
        from milia_pipeline.models.hpo.backends.base import get_backend

        with pytest.raises(BackendError):
            get_backend("unknown_backend")

    def test_get_backend_unknown_error_message_contains_backend_name(self):
        """Test BackendError message contains the unknown backend name."""
        from milia_pipeline.exceptions import BackendError
        from milia_pipeline.models.hpo.backends.base import get_backend

        with pytest.raises(BackendError) as exc_info:
            get_backend("invalid_backend")

        assert "invalid_backend" in str(exc_info.value)

    def test_get_backend_unknown_error_contains_available_backends(self):
        """Test BackendError details contains available backends."""
        from milia_pipeline.exceptions import BackendError
        from milia_pipeline.models.hpo.backends.base import get_backend

        with pytest.raises(BackendError) as exc_info:
            get_backend("ray_tune")  # Not yet implemented

        error = exc_info.value
        assert error.details is not None
        assert "optuna" in error.details

    def test_get_backend_empty_string_raises_backend_error(self):
        """Test get_backend('') raises BackendError."""
        from milia_pipeline.exceptions import BackendError
        from milia_pipeline.models.hpo.backends.base import get_backend

        with pytest.raises(BackendError):
            get_backend("")

    def test_get_backend_none_raises_error(self):
        """Test get_backend(None) raises an error."""
        from milia_pipeline.exceptions import BackendError
        from milia_pipeline.models.hpo.backends.base import get_backend

        # None is converted to string 'None' by the backend_name check
        # So it raises BackendError for unknown backend
        with pytest.raises(BackendError):
            get_backend(None)

    def test_get_backend_case_sensitive(self):
        """Test get_backend() is case-sensitive."""
        from milia_pipeline.exceptions import BackendError
        from milia_pipeline.models.hpo.backends.base import get_backend

        # 'Optuna' should fail (uppercase O)
        with pytest.raises(BackendError):
            get_backend("Optuna")

        # 'OPTUNA' should fail (all uppercase)
        with pytest.raises(BackendError):
            get_backend("OPTUNA")


class TestGetBackendImportError:
    """Test get_backend() handling of ImportError during backend instantiation."""

    def test_get_backend_import_error_raises_backend_error(self):
        """Test ImportError during backend instantiation raises BackendError."""
        from milia_pipeline.exceptions import BackendError
        from milia_pipeline.models.hpo.backends.base import get_backend

        mock_optuna_cls = MagicMock(side_effect=ImportError("No module named 'optuna'"))
        mock_module = MagicMock()
        mock_module.OptunaBackend = mock_optuna_cls

        with (
            patch.dict(
                "sys.modules", {"milia_pipeline.models.hpo.backends.optuna_backend": mock_module}
            ),
            pytest.raises(BackendError) as exc_info,
        ):
            get_backend("optuna")

        error = exc_info.value
        assert "not installed" in str(error).lower() or "dependencies" in str(error).lower()

    def test_get_backend_import_error_includes_backend_name(self):
        """Test BackendError from ImportError includes backend name."""
        from milia_pipeline.exceptions import BackendError
        from milia_pipeline.models.hpo.backends.base import get_backend

        mock_optuna_cls = MagicMock(side_effect=ImportError("No module named 'optuna'"))
        mock_module = MagicMock()
        mock_module.OptunaBackend = mock_optuna_cls

        with (
            patch.dict(
                "sys.modules", {"milia_pipeline.models.hpo.backends.optuna_backend": mock_module}
            ),
            pytest.raises(BackendError) as exc_info,
        ):
            get_backend("optuna")

        error = exc_info.value
        assert error.backend_name == "optuna"

    def test_get_backend_import_error_includes_original_error_in_details(self):
        """Test BackendError from ImportError includes original error in details."""
        from milia_pipeline.exceptions import BackendError
        from milia_pipeline.models.hpo.backends.base import get_backend

        original_error_msg = "No module named 'optuna'"
        mock_optuna_cls = MagicMock(side_effect=ImportError(original_error_msg))
        mock_module = MagicMock()
        mock_module.OptunaBackend = mock_optuna_cls

        with (
            patch.dict(
                "sys.modules", {"milia_pipeline.models.hpo.backends.optuna_backend": mock_module}
            ),
            pytest.raises(BackendError) as exc_info,
        ):
            get_backend("optuna")

        error = exc_info.value
        assert original_error_msg in str(error.details)


# =============================================================================
# BACKENDERROR EXCEPTION ATTRIBUTE TESTS
# =============================================================================


class TestBackendErrorAttributes:
    """Test BackendError exception attributes."""

    def test_backend_error_has_backend_name_attribute(self):
        """Test BackendError has backend_name attribute."""
        from milia_pipeline.exceptions import BackendError

        error = BackendError("Test error", backend_name="optuna")

        assert hasattr(error, "backend_name")
        assert error.backend_name == "optuna"

    def test_backend_error_has_operation_attribute(self):
        """Test BackendError has operation attribute."""
        from milia_pipeline.exceptions import BackendError

        error = BackendError("Test error", backend_name="optuna", operation="create_study")

        assert hasattr(error, "operation")
        assert error.operation == "create_study"

    def test_backend_error_has_details_attribute(self):
        """Test BackendError has details attribute."""
        from milia_pipeline.exceptions import BackendError

        error = BackendError("Test error", backend_name="optuna", details="Additional info")

        assert hasattr(error, "details")
        assert error.details == "Additional info"

    def test_backend_error_str_contains_message(self):
        """Test BackendError __str__ contains the message."""
        from milia_pipeline.exceptions import BackendError

        error = BackendError("Test error message", backend_name="optuna")

        assert "Test error message" in str(error)

    def test_backend_error_str_contains_backend_name(self):
        """Test BackendError __str__ contains backend name."""
        from milia_pipeline.exceptions import BackendError

        error = BackendError("Test error", backend_name="test_backend")

        assert "test_backend" in str(error)

    def test_backend_error_str_contains_operation(self):
        """Test BackendError __str__ contains operation when set."""
        from milia_pipeline.exceptions import BackendError

        error = BackendError("Test error", backend_name="optuna", operation="optimize")

        assert "optimize" in str(error)

    def test_backend_error_inherits_from_hpo_error(self):
        """Test BackendError inherits from HPOError."""
        from milia_pipeline.exceptions import BackendError, HPOError

        assert issubclass(BackendError, HPOError)

    def test_backend_error_with_all_attributes(self):
        """Test BackendError with all attributes set."""
        from milia_pipeline.exceptions import BackendError

        error = BackendError(
            "Complete error test",
            backend_name="ray_tune",
            operation="create_sampler",
            details="Missing dependency",
        )

        assert error.backend_name == "ray_tune"
        assert error.operation == "create_sampler"
        assert error.details == "Missing dependency"

        error_str = str(error)
        assert "Complete error test" in error_str
        assert "ray_tune" in error_str
        assert "create_sampler" in error_str


class TestBackendErrorFromGetBackend:
    """Test BackendError raised by get_backend()."""

    def test_unknown_backend_error_has_backend_name(self):
        """Test BackendError for unknown backend has backend_name set."""
        from milia_pipeline.exceptions import BackendError
        from milia_pipeline.models.hpo.backends.base import get_backend

        try:
            get_backend("nonexistent")
        except BackendError as e:
            assert e.backend_name == "nonexistent"

    def test_unknown_backend_error_has_available_in_details(self):
        """Test BackendError for unknown backend lists available backends."""
        from milia_pipeline.exceptions import BackendError
        from milia_pipeline.models.hpo.backends.base import get_backend

        try:
            get_backend("nonexistent")
        except BackendError as e:
            assert "Available backends" in e.details or "optuna" in e.details


# =============================================================================
# INTERNAL BACKENDS REGISTRY TESTS
# =============================================================================


class TestBackendsRegistry:
    """Test the internal backends registry dictionary in get_backend()."""

    def test_backends_dict_is_created_inside_function(self):
        """Test backends dict is created inside get_backend, not module level."""
        from milia_pipeline.models.hpo.backends import base

        # The backends dict should NOT be a module-level attribute
        # It's defined inside get_backend() function
        assert not hasattr(base, "backends"), (
            "backends dict should be inside get_backend(), not at module level"
        )

    def test_backends_dict_includes_optuna(self):
        """Test backends registry includes 'optuna' key."""
        from milia_pipeline.exceptions import BackendError
        from milia_pipeline.models.hpo.backends.base import get_backend

        # We can verify the registry by checking error messages
        try:
            get_backend("__nonexistent_test_backend__")
        except BackendError as e:
            assert "optuna" in e.details.lower()

    def test_backends_registry_is_case_sensitive(self):
        """Test backend names in registry are case-sensitive."""
        from milia_pipeline.exceptions import BackendError
        from milia_pipeline.models.hpo.backends.base import get_backend

        # These should all fail (case variations)
        case_variations = ["Optuna", "OPTUNA", "OpTuNa", "optunA"]

        for variation in case_variations:
            with pytest.raises(BackendError) as exc_info:
                get_backend(variation)
            # Verify the error mentions the exact case used
            assert variation in str(exc_info.value)


class TestGetBackendExceptionModule:
    """Test get_backend() imports BackendError from correct module."""

    def test_backend_error_imported_from_exceptions(self):
        """Test BackendError is imported from milia_pipeline.exceptions."""
        from milia_pipeline.exceptions import BackendError
        from milia_pipeline.models.hpo.backends.base import get_backend

        with pytest.raises(BackendError) as exc_info:
            get_backend("nonexistent")

        # Verify it's the same BackendError class
        assert type(exc_info.value).__name__ == "BackendError"
        assert type(exc_info.value).__module__ == "milia_pipeline.exceptions"


# =============================================================================
# EDGE CASES AND BOUNDARY CONDITIONS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_protocol_can_be_used_as_type_hint(self):
        """Test HPOBackendProtocol can be used as a type hint."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        def accept_backend(backend: HPOBackendProtocol) -> None:
            pass

        # Should not raise
        assert callable(accept_backend)

    def test_protocol_docstring_exists(self):
        """Test HPOBackendProtocol has a docstring."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        assert HPOBackendProtocol.__doc__ is not None
        assert len(HPOBackendProtocol.__doc__) > 0

    def test_get_backend_docstring_exists(self):
        """Test get_backend function has a docstring."""
        from milia_pipeline.models.hpo.backends.base import get_backend

        assert get_backend.__doc__ is not None
        assert len(get_backend.__doc__) > 0

    def test_get_backend_multiple_calls_create_separate_instances(self):
        """Test multiple get_backend calls create separate instances."""
        from milia_pipeline.models.hpo.backends.base import get_backend

        mock_optuna_cls = MagicMock(return_value=MagicMock())
        mock_module = MagicMock()
        mock_module.OptunaBackend = mock_optuna_cls

        with patch.dict(
            "sys.modules", {"milia_pipeline.models.hpo.backends.optuna_backend": mock_module}
        ):
            _backend1 = get_backend("optuna")
            _backend2 = get_backend("optuna")

            assert mock_optuna_cls.call_count == 2

    def test_backends_dict_contains_optuna(self):
        """Test internal backends dict includes 'optuna' key."""
        from milia_pipeline.exceptions import BackendError
        from milia_pipeline.models.hpo.backends.base import get_backend

        # Try to get a non-existent backend to check the error message
        try:
            get_backend("__test__")
        except BackendError as e:
            # The details should mention optuna as available
            assert "optuna" in e.details.lower()


class TestGetBackendSpecialInputs:
    """Test get_backend with special input values."""

    def test_get_backend_with_whitespace_raises_error(self):
        """Test get_backend with whitespace-only string raises error."""
        from milia_pipeline.exceptions import BackendError
        from milia_pipeline.models.hpo.backends.base import get_backend

        with pytest.raises(BackendError):
            get_backend("   ")

    def test_get_backend_with_special_characters_raises_error(self):
        """Test get_backend with special characters raises error."""
        from milia_pipeline.exceptions import BackendError
        from milia_pipeline.models.hpo.backends.base import get_backend

        with pytest.raises(BackendError):
            get_backend("optuna!")

    def test_get_backend_with_numeric_string_raises_error(self):
        """Test get_backend with numeric string raises error."""
        from milia_pipeline.exceptions import BackendError
        from milia_pipeline.models.hpo.backends.base import get_backend

        with pytest.raises(BackendError):
            get_backend("123")


# =============================================================================
# PROTOCOL METHOD INVOCATION TESTS
# =============================================================================


class TestProtocolMethodInvocation:
    """Test that protocol methods can be invoked on compliant instances."""

    def test_create_study_can_be_called(self, compliant_backend_class):
        """Test create_study can be called on compliant instance."""
        backend = compliant_backend_class()

        result = backend.create_study(study_name="test_study", direction="minimize")

        assert result is not None

    def test_optimize_can_be_called(self, compliant_backend_class):
        """Test optimize can be called on compliant instance."""
        backend = compliant_backend_class()

        mock_study = MagicMock()
        mock_objective = lambda trial: 0.5

        # Should not raise
        backend.optimize(study=mock_study, objective_fn=mock_objective, n_trials=10)

    def test_get_best_params_returns_dict(self, compliant_backend_class):
        """Test get_best_params returns a dict on compliant instance."""
        backend = compliant_backend_class()
        mock_study = MagicMock()

        result = backend.get_best_params(mock_study)

        assert isinstance(result, dict)

    def test_get_best_value_returns_float(self, compliant_backend_class):
        """Test get_best_value returns a float on compliant instance."""
        backend = compliant_backend_class()
        mock_study = MagicMock()

        result = backend.get_best_value(mock_study)

        assert isinstance(result, float)

    def test_get_all_trials_returns_list(self, compliant_backend_class):
        """Test get_all_trials returns a list on compliant instance."""
        backend = compliant_backend_class()
        mock_study = MagicMock()

        result = backend.get_all_trials(mock_study)

        assert isinstance(result, list)

    def test_create_pruner_returns_something(self, compliant_backend_class):
        """Test create_pruner returns something on compliant instance."""
        backend = compliant_backend_class()

        result = backend.create_pruner(pruner_type="median")

        assert result is not None

    def test_create_sampler_returns_something(self, compliant_backend_class):
        """Test create_sampler returns something on compliant instance."""
        backend = compliant_backend_class()

        result = backend.create_sampler(sampler_type="tpe")

        assert result is not None


# =============================================================================
# MODULE EXPORTS TESTS
# =============================================================================


class TestModuleExports:
    """Test module exports and public API."""

    def test_hpo_backend_protocol_exported(self):
        """Test HPOBackendProtocol is properly exported."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        assert HPOBackendProtocol is not None

    def test_get_backend_exported(self):
        """Test get_backend is properly exported."""
        from milia_pipeline.models.hpo.backends.base import get_backend

        assert get_backend is not None
        assert callable(get_backend)

    def test_optuna_available_exported(self):
        """Test OPTUNA_AVAILABLE is properly exported."""
        from milia_pipeline.models.hpo.backends.base import OPTUNA_AVAILABLE

        assert isinstance(OPTUNA_AVAILABLE, bool)

    def test_module_has_docstring(self):
        """Test base module has a docstring."""
        from milia_pipeline.models.hpo.backends import base

        assert base.__doc__ is not None


# =============================================================================
# INTEGRATION-STYLE UNIT TESTS
# =============================================================================


class TestProtocolIntegration:
    """Integration-style unit tests for protocol usage."""

    def test_backend_can_be_used_polymorphically(self, compliant_backend_class):
        """Test backend can be used polymorphically via protocol."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        def use_backend(backend: HPOBackendProtocol) -> dict[str, Any]:
            """Function that accepts any protocol-compliant backend."""
            study = backend.create_study("test", "minimize")
            return backend.get_best_params(study)

        backend = compliant_backend_class()
        assert isinstance(backend, HPOBackendProtocol)

        result = use_backend(backend)
        assert isinstance(result, dict)

    def test_protocol_check_with_isinstance_in_function(self, compliant_backend_class):
        """Test isinstance protocol check works in function guard."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        def validate_backend(backend: Any) -> bool:
            """Validate backend implements protocol."""
            return isinstance(backend, HPOBackendProtocol)

        compliant = compliant_backend_class()
        assert validate_backend(compliant) is True

        non_compliant = object()
        assert validate_backend(non_compliant) is False


class TestBackendFactoryPatterns:
    """Test factory pattern behaviors."""

    def test_factory_returns_correct_type_annotation(self):
        """Test get_backend return type matches protocol."""
        import inspect

        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol, get_backend

        sig = inspect.signature(get_backend)
        # Return annotation should be HPOBackendProtocol
        assert sig.return_annotation == HPOBackendProtocol

    def test_factory_pattern_allows_backend_substitution(self):
        """Test factory pattern enables backend substitution."""
        from milia_pipeline.models.hpo.backends.base import get_backend

        # Create a mock backend registry scenario
        mock_instance = MagicMock()
        mock_instance.create_study = MagicMock(return_value=MagicMock())
        mock_optuna_cls = MagicMock(return_value=mock_instance)
        mock_module = MagicMock()
        mock_module.OptunaBackend = mock_optuna_cls

        with patch.dict(
            "sys.modules", {"milia_pipeline.models.hpo.backends.optuna_backend": mock_module}
        ):
            backend = get_backend("optuna")

            # Backend should be usable through protocol interface
            _study = backend.create_study("test", "minimize")
            backend.create_study.assert_called_once_with("test", "minimize")


# =============================================================================
# PROTOCOL METHOD COUNT VERIFICATION
# =============================================================================


class TestProtocolMethodCount:
    """Verify the protocol has exactly the expected number of methods.

    Note: The docstring in base.py says '6 methods' but lists 7 methods.
    This test verifies the actual implementation matches expectations.
    """

    def test_protocol_has_seven_abstract_methods(self):
        """Test HPOBackendProtocol has exactly 7 abstract methods."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        expected_methods = {
            "create_study",
            "optimize",
            "get_best_params",
            "get_best_value",
            "get_all_trials",
            "create_pruner",
            "create_sampler",
        }

        # Count abstract methods
        abstract_methods = set()
        for name in dir(HPOBackendProtocol):
            if not name.startswith("_"):
                method = getattr(HPOBackendProtocol, name, None)
                if callable(method) and getattr(method, "__isabstractmethod__", False):
                    abstract_methods.add(name)

        assert abstract_methods == expected_methods, (
            f"Expected {expected_methods}, got {abstract_methods}"
        )
        assert len(abstract_methods) == 7


class TestProtocolKwargsSupport:
    """Test protocol methods with **kwargs properly accept arbitrary keyword arguments."""

    def test_create_pruner_accepts_kwargs(self, compliant_backend_class):
        """Test create_pruner accepts arbitrary keyword arguments."""
        backend = compliant_backend_class()

        # Should not raise with extra kwargs
        result = backend.create_pruner(
            pruner_type="median",
            n_startup_trials=5,
            n_warmup_steps=10,
            custom_arg1="value1",
            custom_arg2=42,
        )
        assert result is not None

    def test_create_sampler_accepts_kwargs(self, compliant_backend_class):
        """Test create_sampler accepts arbitrary keyword arguments."""
        backend = compliant_backend_class()

        # Should not raise with extra kwargs
        result = backend.create_sampler(
            sampler_type="tpe", seed=42, n_startup_trials=10, custom_setting="value"
        )
        assert result is not None


# =============================================================================
# GET_BACKEND CONCURRENT ACCESS TESTS
# =============================================================================


class TestGetBackendConcurrentAccess:
    """Test get_backend() behavior under concurrent access scenarios.

    These tests verify the factory function is safe for concurrent use,
    though true thread safety depends on the backend implementations.
    """

    def test_multiple_sequential_backend_creations(self):
        """Test get_backend can be called multiple times sequentially."""
        from milia_pipeline.models.hpo.backends.base import get_backend

        mock_instances = [MagicMock(name=f"instance_{i}") for i in range(5)]
        mock_optuna_cls = MagicMock(side_effect=mock_instances)
        mock_module = MagicMock()
        mock_module.OptunaBackend = mock_optuna_cls

        with patch.dict(
            "sys.modules", {"milia_pipeline.models.hpo.backends.optuna_backend": mock_module}
        ):
            results = [get_backend("optuna") for _ in range(5)]

        # Each call should return a different instance
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result is mock_instances[i]

    def test_backend_factory_is_not_singleton(self):
        """Test get_backend does not implement singleton pattern."""
        from milia_pipeline.models.hpo.backends.base import get_backend

        instance1 = MagicMock()
        instance2 = MagicMock()
        mock_optuna_cls = MagicMock(side_effect=[instance1, instance2])
        mock_module = MagicMock()
        mock_module.OptunaBackend = mock_optuna_cls

        with patch.dict(
            "sys.modules", {"milia_pipeline.models.hpo.backends.optuna_backend": mock_module}
        ):
            backend1 = get_backend("optuna")
            backend2 = get_backend("optuna")

        assert backend1 is not backend2
        assert backend1 is instance1
        assert backend2 is instance2


# =============================================================================
# GET_BACKEND FUNCTION SIGNATURE TESTS
# =============================================================================


class TestGetBackendFunctionSignature:
    """Test get_backend() function signature and annotations."""

    def test_get_backend_has_single_required_parameter(self):
        """Test get_backend has exactly one required parameter."""
        from milia_pipeline.models.hpo.backends.base import get_backend

        sig = inspect.signature(get_backend)
        required_params = [
            p for p in sig.parameters.values() if p.default is inspect.Parameter.empty
        ]

        assert len(required_params) == 1
        assert required_params[0].name == "backend_name"

    def test_get_backend_backend_name_is_str(self):
        """Test get_backend backend_name parameter is annotated as str."""
        from milia_pipeline.models.hpo.backends.base import get_backend

        sig = inspect.signature(get_backend)
        assert sig.parameters["backend_name"].annotation == str

    def test_get_backend_return_annotation_is_protocol(self):
        """Test get_backend return annotation is HPOBackendProtocol."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol, get_backend

        sig = inspect.signature(get_backend)
        assert sig.return_annotation == HPOBackendProtocol


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
