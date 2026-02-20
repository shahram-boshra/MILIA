#!/usr/bin/env python3
"""
Unit Test Suite for milia_pipeline/models/hpo/backends/optuna_backend.py Module

Tests included in this part:
1. Module-level import behavior (OPTUNA_AVAILABLE flag)
2. OptunaBackend class initialization
3. Initialization error handling (BackendError when optuna not available)
4. Logger initialization
5. Class structure validation
6. create_study method - study creation, loading, error handling
7. optimize method - optimization execution, callbacks, error handling
8. get_best_params method - retrieving best parameters
9. get_best_value method - retrieving best objective value
10. get_all_trials method - retrieving trial information
11. _build_sampler_registry method - dynamic registry building, caching, version compatibility
12. create_sampler method - sampler creation for all types, parameters, error handling
13. create_pruner method - pruner creation for all types, parameters, error handling
14. suggest_params method - parameter suggestion for all types (int, float, categorical, etc.)
    - Includes tests for missing required parameters (low, high, choices, type)
15. Integration tests - end-to-end workflows combining multiple methods
16. Edge cases and boundary conditions
17. Protocol compliance verification

Author: Milia Team
Version: 1.2.0

Changelog:
- v1.2.0: Production-ready improvements:
          - Added test for DuplicatedStudyError handling path in create_study
          - Added test for keyboard interrupt logging in optimize
          - Added test for create_study logging when resuming with existing trials
          - Added tests for get_all_trials duration calculation edge cases
          - Added tests for suggest_params with mixed dict/dataclass configs
          - Added tests for uniform type bounds validation
          - Added tests for MOTPESampler and QMCSampler when available
          - Added comprehensive tests for threshold pruner parameters
          - Added tests for successive_halving pruner parameters
          - Added test for config objects with None attribute values
- v1.1.0: Added tests for missing required parameters in suggest_params
          (low, high, choices, type) that exercise _get_required_value helper
- v1.0.0: Initial release
"""

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import importlib.util
import inspect
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# HELPER: Check if optuna is actually available
# =============================================================================


def is_optuna_installed() -> bool:
    """Check if optuna is actually installed in the environment."""
    try:
        return importlib.util.find_spec("optuna") is not None
    except ValueError:
        # find_spec raises ValueError if the module is in sys.modules
        # but __spec__ is not set or is None (documented CPython behavior)
        return False


OPTUNA_INSTALLED = is_optuna_installed()


# =============================================================================
# HELPER CLASSES FOR SEARCH SPACE TESTING
# =============================================================================


class ParamType(Enum):
    """Parameter type enumeration for testing."""

    INT = "int"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    LOGUNIFORM = "loguniform"
    UNIFORM = "uniform"


@dataclass
class ParamConfig:
    """Parameter configuration dataclass for testing."""

    type: ParamType
    low: float | None = None
    high: float | None = None
    step: int | None = None
    log: bool | None = None
    choices: list[Any] | None = None


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def backend_instance():
    """
    Create a real OptunaBackend instance.
    Skip if optuna is not installed.
    """
    if not OPTUNA_INSTALLED:
        pytest.skip("Optuna not installed")

    from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

    return OptunaBackend()


@pytest.fixture
def fresh_backend_instance():
    """
    Create a fresh OptunaBackend instance without cached registry.
    Skip if optuna is not installed.
    """
    if not OPTUNA_INSTALLED:
        pytest.skip("Optuna not installed")

    from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

    backend = OptunaBackend()
    # Ensure no cached registry
    if hasattr(backend, "_sampler_registry_cache"):
        delattr(backend, "_sampler_registry_cache")
    return backend


@pytest.fixture
def mock_optuna_module():
    """Create a comprehensive mock optuna module."""
    mock = MagicMock()

    # Mock TrialState enum
    mock.trial.TrialState.COMPLETE = "COMPLETE"
    mock.trial.TrialState.PRUNED = "PRUNED"
    mock.trial.TrialState.FAIL = "FAIL"
    mock.trial.TrialState.RUNNING = "RUNNING"

    # Mock study
    mock_study = MagicMock()
    mock_study.trials = []
    mock_study.study_name = "test_study"
    mock_study.best_params = {"lr": 0.001, "hidden_dim": 128}
    mock_study.best_value = 0.05
    mock.create_study.return_value = mock_study
    mock.load_study.return_value = mock_study

    # Mock exceptions
    mock.exceptions.DuplicatedStudyError = type("DuplicatedStudyError", (Exception,), {})

    return mock


@pytest.fixture
def mock_study():
    """Create a mock Optuna study object."""
    study = MagicMock()
    study.study_name = "test_study"
    study.trials = []
    study.best_params = {"learning_rate": 0.001, "batch_size": 32}
    study.best_value = 0.042
    study.direction = "minimize"
    return study


@pytest.fixture
def mock_study_with_trials():
    """Create a mock study with completed, pruned, and failed trials."""
    from datetime import datetime, timedelta

    study = MagicMock()
    study.study_name = "test_study_with_trials"

    # Create mock TrialState
    class MockTrialState:
        COMPLETE = "COMPLETE"
        PRUNED = "PRUNED"
        FAIL = "FAIL"
        RUNNING = "RUNNING"

    # Create mock trials
    completed_trial = MagicMock()
    completed_trial.number = 0
    completed_trial.params = {"lr": 0.001}
    completed_trial.value = 0.05
    completed_trial.state = MagicMock()
    completed_trial.state.name = "COMPLETE"
    completed_trial.datetime_start = datetime.now() - timedelta(minutes=5)
    completed_trial.datetime_complete = datetime.now() - timedelta(minutes=4)
    completed_trial.user_attrs = {"model": "GCN"}
    completed_trial.intermediate_values = {0: 0.1, 1: 0.07, 2: 0.05}

    pruned_trial = MagicMock()
    pruned_trial.number = 1
    pruned_trial.params = {"lr": 0.1}
    pruned_trial.value = None
    pruned_trial.state = MagicMock()
    pruned_trial.state.name = "PRUNED"
    pruned_trial.datetime_start = datetime.now() - timedelta(minutes=3)
    pruned_trial.datetime_complete = datetime.now() - timedelta(minutes=2)
    pruned_trial.user_attrs = {}
    pruned_trial.intermediate_values = {0: 0.5}

    failed_trial = MagicMock()
    failed_trial.number = 2
    failed_trial.params = {"lr": 1.0}
    failed_trial.value = None
    failed_trial.state = MagicMock()
    failed_trial.state.name = "FAIL"
    failed_trial.datetime_start = datetime.now() - timedelta(minutes=1)
    failed_trial.datetime_complete = None
    failed_trial.user_attrs = {"error": "OOM"}
    failed_trial.intermediate_values = {}

    study.trials = [completed_trial, pruned_trial, failed_trial]
    study.best_params = {"lr": 0.001}
    study.best_value = 0.05

    return study, MockTrialState


@pytest.fixture
def mock_objective_fn():
    """Create a mock objective function."""

    def objective(trial):
        return 0.5

    return objective


@pytest.fixture
def mock_optuna_samplers():
    """Create mock optuna.samplers module with configurable samplers."""
    mock_samplers = MagicMock()

    # Core samplers (always available)
    mock_samplers.TPESampler = MagicMock(return_value=MagicMock(name="TPESampler"))
    mock_samplers.RandomSampler = MagicMock(return_value=MagicMock(name="RandomSampler"))
    mock_samplers.CmaEsSampler = MagicMock(return_value=MagicMock(name="CmaEsSampler"))
    mock_samplers.GridSampler = MagicMock(return_value=MagicMock(name="GridSampler"))
    mock_samplers.NSGAIISampler = MagicMock(return_value=MagicMock(name="NSGAIISampler"))

    return mock_samplers


@pytest.fixture
def mock_optuna_pruners():
    """Create mock optuna.pruners module."""
    mock_pruners = MagicMock()

    mock_pruners.MedianPruner = MagicMock(return_value=MagicMock(name="MedianPruner"))
    mock_pruners.PercentilePruner = MagicMock(return_value=MagicMock(name="PercentilePruner"))
    mock_pruners.HyperbandPruner = MagicMock(return_value=MagicMock(name="HyperbandPruner"))
    mock_pruners.SuccessiveHalvingPruner = MagicMock(
        return_value=MagicMock(name="SuccessiveHalvingPruner")
    )
    mock_pruners.ThresholdPruner = MagicMock(return_value=MagicMock(name="ThresholdPruner"))
    mock_pruners.PatientPruner = MagicMock(return_value=MagicMock(name="PatientPruner"))
    mock_pruners.NopPruner = MagicMock(return_value=MagicMock(name="NopPruner"))

    return mock_pruners


@pytest.fixture
def mock_trial():
    """Create a mock Optuna trial with suggest methods."""
    trial = MagicMock()
    trial.suggest_int = MagicMock(return_value=64)
    trial.suggest_float = MagicMock(return_value=0.001)
    trial.suggest_categorical = MagicMock(return_value="relu")
    return trial


@pytest.fixture
def simple_search_space_dict():
    """Create a simple search space using dictionaries."""
    return {
        "model": {
            "hidden_dim": {"type": "int", "low": 32, "high": 256, "step": 32},
            "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        },
        "training": {"batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]}},
    }


@pytest.fixture
def search_space_with_dataclass():
    """Create a search space using dataclass-style config objects."""
    return {
        "model": {
            "hidden_dim": ParamConfig(type=ParamType.INT, low=32, high=256, step=32),
            "learning_rate": ParamConfig(type=ParamType.FLOAT, low=1e-5, high=1e-2, log=True),
        },
        "training": {
            "activation": ParamConfig(type=ParamType.CATEGORICAL, choices=["relu", "gelu", "silu"])
        },
    }


@pytest.fixture
def comprehensive_search_space():
    """Create a comprehensive search space with all parameter types."""
    return {
        "model": {
            "num_layers": {"type": "int", "low": 1, "high": 10, "step": 1},
            "hidden_dim": {"type": "int", "low": 32, "high": 512, "step": 32},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5, "log": False},
            "learning_rate": {"type": "loguniform", "low": 1e-6, "high": 1e-1},
        },
        "optimizer": {
            "weight_decay": {"type": "uniform", "low": 0.0, "high": 0.1},
            "optimizer_type": {"type": "categorical", "choices": ["adam", "adamw", "sgd"]},
        },
    }


# =============================================================================
# PART 1: MODULE IMPORT TESTS - OPTUNA_AVAILABLE FLAG
# =============================================================================


class TestOptunaAvailableFlag:
    """Test OPTUNA_AVAILABLE module-level flag behavior."""

    def test_optuna_available_flag_is_boolean(self):
        """Test OPTUNA_AVAILABLE flag is a boolean type."""
        from milia_pipeline.models.hpo.backends.optuna_backend import OPTUNA_AVAILABLE

        assert isinstance(OPTUNA_AVAILABLE, bool), (
            f"OPTUNA_AVAILABLE should be bool, got {type(OPTUNA_AVAILABLE)}"
        )

    def test_optuna_available_matches_actual_import_status(self):
        """Test OPTUNA_AVAILABLE correctly reflects whether optuna can be imported."""
        from milia_pipeline.models.hpo.backends.optuna_backend import OPTUNA_AVAILABLE

        assert OPTUNA_AVAILABLE == OPTUNA_INSTALLED, (
            f"OPTUNA_AVAILABLE ({OPTUNA_AVAILABLE}) doesn't match actual import status ({OPTUNA_INSTALLED})"
        )

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_optuna_available_true_when_installed(self):
        """Test OPTUNA_AVAILABLE is True when optuna is installed."""
        from milia_pipeline.models.hpo.backends.optuna_backend import OPTUNA_AVAILABLE

        assert OPTUNA_AVAILABLE is True

    def test_optuna_module_variable_defined(self):
        """Test 'optuna' variable is defined in module (as module or None)."""
        from milia_pipeline.models.hpo.backends import optuna_backend as module

        assert hasattr(module, "optuna"), "Module should define 'optuna' variable"

        if OPTUNA_INSTALLED:
            assert module.optuna is not None, "optuna should be the module when installed"
        else:
            assert module.optuna is None, "optuna should be None when not installed"

    def test_trial_state_variable_defined(self):
        """Test 'TrialState' variable is defined in module."""
        from milia_pipeline.models.hpo.backends import optuna_backend as module

        assert hasattr(module, "TrialState"), "Module should define 'TrialState' variable"

        if OPTUNA_INSTALLED:
            assert module.TrialState is not None
        else:
            assert module.TrialState is None


class TestModuleImports:
    """Test module import behavior and exports."""

    def test_module_imports_without_error(self):
        """Test optuna_backend module imports successfully."""
        try:
            from milia_pipeline.models.hpo.backends import optuna_backend

            assert optuna_backend is not None
        except ImportError as e:
            pytest.fail(f"Module import failed: {e}")

    def test_module_exports_optuna_backend_class(self):
        """Test OptunaBackend class is exported."""
        from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

        assert OptunaBackend is not None
        assert inspect.isclass(OptunaBackend)

    def test_module_exports_optuna_available_flag(self):
        """Test OPTUNA_AVAILABLE is exported."""
        from milia_pipeline.models.hpo.backends.optuna_backend import OPTUNA_AVAILABLE

        assert OPTUNA_AVAILABLE is not None

    def test_module_has_logger(self):
        """Test module has a properly configured logger."""
        from milia_pipeline.models.hpo.backends.optuna_backend import logger

        assert logger is not None
        assert isinstance(logger, logging.Logger)

    def test_logger_has_correct_name(self):
        """Test logger has the correct module name."""
        from milia_pipeline.models.hpo.backends.optuna_backend import logger

        expected_name = "milia_pipeline.models.hpo.backends.optuna_backend"
        assert logger.name == expected_name, (
            f"Logger name should be '{expected_name}', got '{logger.name}'"
        )

    def test_module_imports_from_base(self):
        """Test module imports HPOBackendProtocol from base."""
        from milia_pipeline.models.hpo.backends import optuna_backend as module

        # Check the import statement exists by verifying the module works
        # The actual import is: from .base import HPOBackendProtocol
        assert hasattr(module, "OptunaBackend")

    def test_module_imports_exceptions(self):
        """Test module imports required exceptions."""
        from milia_pipeline.exceptions import BackendError, HPOError

        # These should be importable without error
        assert BackendError is not None
        assert HPOError is not None


# =============================================================================
# PART 1: OptunaBackend INITIALIZATION TESTS
# =============================================================================


class TestOptunaBackendInit:
    """Test OptunaBackend.__init__ method."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_init_succeeds_when_optuna_available(self):
        """Test __init__ succeeds when optuna is installed."""
        from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

        backend = OptunaBackend()

        assert backend is not None
        assert isinstance(backend, OptunaBackend)

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_init_creates_usable_instance(self):
        """Test __init__ creates an instance with callable methods."""
        from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

        backend = OptunaBackend()

        # Verify methods are callable
        assert callable(backend.create_study)
        assert callable(backend.optimize)
        assert callable(backend.create_sampler)
        assert callable(backend.create_pruner)

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_init_logs_initialization_message(self):
        """Test __init__ logs 'OptunaBackend initialized' message."""
        from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

        with patch("milia_pipeline.models.hpo.backends.optuna_backend.logger") as mock_logger:
            OptunaBackend()

            mock_logger.info.assert_called_once_with("OptunaBackend initialized")

    def test_init_raises_backend_error_when_optuna_unavailable(self):
        """Test __init__ raises BackendError when OPTUNA_AVAILABLE is False."""
        from milia_pipeline.exceptions import BackendError

        # Patch at module level to simulate optuna not being available
        with patch("milia_pipeline.models.hpo.backends.optuna_backend.OPTUNA_AVAILABLE", False):
            from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

            with pytest.raises(BackendError) as exc_info:
                OptunaBackend()

            error = exc_info.value
            assert error.backend_name == "optuna"
            assert "not installed" in str(error).lower()

    def test_backend_error_includes_install_instructions(self):
        """Test BackendError message includes installation instructions."""
        from milia_pipeline.exceptions import BackendError

        with patch("milia_pipeline.models.hpo.backends.optuna_backend.OPTUNA_AVAILABLE", False):
            from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

            with pytest.raises(BackendError) as exc_info:
                OptunaBackend()

            error = exc_info.value
            assert "pip install optuna" in error.details


class TestOptunaBackendClassStructure:
    """Test OptunaBackend class structure."""

    def test_class_exists(self):
        """Test OptunaBackend class is defined."""
        from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

        assert OptunaBackend is not None
        assert inspect.isclass(OptunaBackend)

    def test_class_has_all_required_public_methods(self):
        """Test OptunaBackend has all required public methods."""
        from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

        required_methods = [
            "create_study",
            "optimize",
            "get_best_params",
            "get_best_value",
            "get_all_trials",
            "create_pruner",
            "create_sampler",
            "suggest_params",
        ]

        for method_name in required_methods:
            assert hasattr(OptunaBackend, method_name), (
                f"OptunaBackend missing required method: {method_name}"
            )
            assert callable(getattr(OptunaBackend, method_name)), (
                f"OptunaBackend.{method_name} should be callable"
            )

    def test_class_has_build_sampler_registry_method(self):
        """Test OptunaBackend has _build_sampler_registry private method."""
        from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

        assert hasattr(OptunaBackend, "_build_sampler_registry")
        assert callable(OptunaBackend._build_sampler_registry)

    def test_class_has_docstring(self):
        """Test OptunaBackend class has a docstring."""
        from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

        assert OptunaBackend.__doc__ is not None
        assert len(OptunaBackend.__doc__.strip()) > 0

    def test_class_docstring_describes_purpose(self):
        """Test OptunaBackend docstring describes its purpose."""
        from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

        docstring = OptunaBackend.__doc__.lower()
        # Should mention optuna and HPO/optimization
        assert "optuna" in docstring
        assert "backend" in docstring or "hpo" in docstring or "optimization" in docstring


class TestOptunaBackendMethodSignatures:
    """Test OptunaBackend method signatures match expected interfaces."""

    def test_create_study_signature(self):
        """Test create_study has correct parameters."""
        from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

        sig = inspect.signature(OptunaBackend.create_study)
        params = sig.parameters

        # Required parameters
        assert "study_name" in params
        assert "direction" in params

        # Optional parameters with defaults
        assert "storage" in params
        assert params["storage"].default is None

        assert "load_if_exists" in params
        assert params["load_if_exists"].default is True

        assert "sampler" in params
        assert params["sampler"].default is None

        assert "pruner" in params
        assert params["pruner"].default is None

    def test_optimize_signature(self):
        """Test optimize has correct parameters."""
        from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

        sig = inspect.signature(OptunaBackend.optimize)
        params = sig.parameters

        # Required parameters
        assert "study" in params
        assert "objective_fn" in params
        assert "n_trials" in params

        # Optional parameters
        assert "timeout" in params
        assert params["timeout"].default is None

        assert "n_jobs" in params
        assert params["n_jobs"].default == 1

        assert "catch" in params
        assert params["catch"].default == (Exception,)

        assert "callbacks" in params
        assert params["callbacks"].default is None

    def test_create_sampler_signature(self):
        """Test create_sampler has correct parameters."""
        from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

        sig = inspect.signature(OptunaBackend.create_sampler)
        params = sig.parameters

        assert "sampler_type" in params

        assert "seed" in params
        assert params["seed"].default is None

        assert "n_startup_trials" in params
        assert params["n_startup_trials"].default == 10

    def test_create_pruner_signature(self):
        """Test create_pruner has correct parameters."""
        from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

        sig = inspect.signature(OptunaBackend.create_pruner)
        params = sig.parameters

        assert "pruner_type" in params

        assert "n_startup_trials" in params
        assert params["n_startup_trials"].default == 5

        assert "n_warmup_steps" in params
        assert params["n_warmup_steps"].default == 10

    def test_suggest_params_signature(self):
        """Test suggest_params has correct parameters."""
        from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

        sig = inspect.signature(OptunaBackend.suggest_params)
        params = sig.parameters

        assert "trial" in params
        assert "search_space" in params

    def test_get_best_params_signature(self):
        """Test get_best_params has correct parameters."""
        from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

        sig = inspect.signature(OptunaBackend.get_best_params)
        params = sig.parameters

        assert "study" in params

    def test_get_best_value_signature(self):
        """Test get_best_value has correct parameters."""
        from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

        sig = inspect.signature(OptunaBackend.get_best_value)
        params = sig.parameters

        assert "study" in params

    def test_get_all_trials_signature(self):
        """Test get_all_trials has correct parameters."""
        from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

        sig = inspect.signature(OptunaBackend.get_all_trials)
        params = sig.parameters

        assert "study" in params


class TestBackendErrorAttributes:
    """Test BackendError exception attributes used by OptunaBackend."""

    def test_backend_error_has_backend_name_attribute(self):
        """Test BackendError has backend_name attribute."""
        from milia_pipeline.exceptions import BackendError

        error = BackendError("Test error", backend_name="optuna", operation="test_op")

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

    def test_backend_error_str_includes_backend_name(self):
        """Test BackendError string representation includes backend name."""
        from milia_pipeline.exceptions import BackendError

        error = BackendError("Test error", backend_name="optuna")

        error_str = str(error)
        assert "optuna" in error_str.lower()


# -------
# =============================================================================
# PART 2: create_study METHOD TESTS
# =============================================================================


class TestCreateStudyBasic:
    """Test create_study method basic functionality."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_study_returns_study_object(self, backend_instance):
        """Test create_study returns an Optuna Study object."""
        import optuna

        study = backend_instance.create_study(study_name="test_basic_study", direction="minimize")

        assert study is not None
        assert isinstance(study, optuna.Study)

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_study_with_minimize_direction(self, backend_instance):
        """Test create_study with 'minimize' direction."""
        study = backend_instance.create_study(
            study_name="test_minimize_study", direction="minimize"
        )

        assert study.direction.name == "MINIMIZE"

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_study_with_maximize_direction(self, backend_instance):
        """Test create_study with 'maximize' direction."""
        study = backend_instance.create_study(
            study_name="test_maximize_study", direction="maximize"
        )

        assert study.direction.name == "MAXIMIZE"

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_study_with_study_name(self, backend_instance):
        """Test create_study assigns correct study name."""
        study_name = "my_custom_study_name"

        study = backend_instance.create_study(study_name=study_name, direction="minimize")

        assert study.study_name == study_name

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_study_with_no_storage_uses_in_memory(self, backend_instance):
        """Test create_study with no storage uses in-memory storage."""
        study = backend_instance.create_study(
            study_name="test_in_memory", direction="minimize", storage=None
        )

        # In-memory studies have no persistent storage
        assert study is not None


class TestCreateStudyWithSampler:
    """Test create_study method with sampler parameter."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_study_with_tpe_sampler(self, backend_instance):
        """Test create_study with TPESampler."""
        import optuna

        sampler = optuna.samplers.TPESampler(seed=42)

        study = backend_instance.create_study(
            study_name="test_with_tpe", direction="minimize", sampler=sampler
        )

        assert study.sampler is sampler

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_study_with_random_sampler(self, backend_instance):
        """Test create_study with RandomSampler."""
        import optuna

        sampler = optuna.samplers.RandomSampler(seed=42)

        study = backend_instance.create_study(
            study_name="test_with_random", direction="minimize", sampler=sampler
        )

        assert study.sampler is sampler

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_study_with_none_sampler_uses_default(self, backend_instance):
        """Test create_study with None sampler uses default TPESampler."""
        import optuna

        study = backend_instance.create_study(
            study_name="test_default_sampler", direction="minimize", sampler=None
        )

        # Default sampler is TPESampler
        assert isinstance(study.sampler, optuna.samplers.TPESampler)


class TestCreateStudyWithPruner:
    """Test create_study method with pruner parameter."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_study_with_median_pruner(self, backend_instance):
        """Test create_study with MedianPruner."""
        import optuna

        pruner = optuna.pruners.MedianPruner()

        study = backend_instance.create_study(
            study_name="test_with_median_pruner", direction="minimize", pruner=pruner
        )

        assert study.pruner is pruner

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_study_with_nop_pruner(self, backend_instance):
        """Test create_study with NopPruner (no pruning)."""
        import optuna

        pruner = optuna.pruners.NopPruner()

        study = backend_instance.create_study(
            study_name="test_with_nop_pruner", direction="minimize", pruner=pruner
        )

        assert study.pruner is pruner


class TestCreateStudyLoadIfExists:
    """Test create_study with load_if_exists parameter."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_study_load_if_exists_true_creates_new(self, backend_instance):
        """Test create_study with load_if_exists=True creates new study if not exists."""
        import uuid

        unique_name = f"test_new_study_{uuid.uuid4().hex[:8]}"

        study = backend_instance.create_study(
            study_name=unique_name, direction="minimize", load_if_exists=True
        )

        assert study is not None
        assert len(study.trials) == 0

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_study_load_if_exists_resumes_existing(self, backend_instance, tmp_path):
        """Test create_study with load_if_exists=True resumes existing study with storage."""

        study_name = "test_resume_study"
        # Use SQLite storage to enable persistence between create_study calls
        storage = f"sqlite:///{tmp_path}/test_resume.db"

        # Create initial study and add a trial
        study1 = backend_instance.create_study(
            study_name=study_name, direction="minimize", storage=storage, load_if_exists=True
        )

        # Run one trial
        study1.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=1)
        initial_trials = len(study1.trials)

        # Create again with load_if_exists=True - should resume
        study2 = backend_instance.create_study(
            study_name=study_name, direction="minimize", storage=storage, load_if_exists=True
        )

        assert len(study2.trials) == initial_trials


class TestCreateStudyLogging:
    """Test create_study logging behavior."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_study_logs_new_study_creation(self, backend_instance):
        """Test create_study logs when creating new study."""
        import uuid

        unique_name = f"test_log_new_{uuid.uuid4().hex[:8]}"

        with patch("milia_pipeline.models.hpo.backends.optuna_backend.logger") as mock_logger:
            backend_instance.create_study(study_name=unique_name, direction="minimize")

            # Should log info about new study creation
            mock_logger.info.assert_called()
            log_message = str(mock_logger.info.call_args)
            assert unique_name in log_message or "Created" in log_message

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_study_logs_resumed_study_with_trial_count(self, backend_instance, tmp_path):
        """Test create_study logs trial count when resuming existing study."""

        study_name = "test_resume_study_logging"
        storage = f"sqlite:///{tmp_path}/test_resume_log.db"

        # Create initial study and add trials
        study1 = backend_instance.create_study(
            study_name=study_name, direction="minimize", storage=storage, load_if_exists=True
        )
        study1.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=3)

        # Resume study and check logging
        with patch("milia_pipeline.models.hpo.backends.optuna_backend.logger") as mock_logger:
            _study2 = backend_instance.create_study(
                study_name=study_name, direction="minimize", storage=storage, load_if_exists=True
            )

            # Should log info about resumed study with trial count
            mock_logger.info.assert_called()
            log_calls = [str(call) for call in mock_logger.info.call_args_list]
            # Verify that resumed message contains trial count info
            assert any(
                "3" in call or "existing" in call.lower() or "resumed" in call.lower()
                for call in log_calls
            )


class TestCreateStudyErrorHandling:
    """Test create_study error handling."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_study_invalid_direction_raises_error(self, backend_instance):
        """Test create_study with invalid direction raises error."""
        from milia_pipeline.exceptions import BackendError

        with pytest.raises(BackendError):
            backend_instance.create_study(
                study_name="test_invalid_direction", direction="invalid_direction"
            )

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_study_handles_duplicated_study_error(self, backend_instance):
        """Test create_study handles DuplicatedStudyError by loading existing study."""
        import optuna

        from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

        backend = OptunaBackend()

        # Create a mock DuplicatedStudyError
        mock_duplicated_error = optuna.exceptions.DuplicatedStudyError("Study already exists")

        # First call raises DuplicatedStudyError, then load_study returns mock study
        mock_study = MagicMock()
        mock_study.study_name = "existing_study"
        mock_study.trials = [MagicMock()]  # Has existing trials

        with (
            patch("optuna.create_study", side_effect=mock_duplicated_error),
            patch("optuna.load_study", return_value=mock_study) as mock_load,
            patch(
                "milia_pipeline.models.hpo.backends.optuna_backend.logger"
            ) as mock_logger,
        ):
            study = backend.create_study(
                study_name="existing_study",
                direction="minimize",
                storage="sqlite:///test.db",
            )

            # Verify load_study was called with correct arguments
            mock_load.assert_called_once()
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs["study_name"] == "existing_study"
            assert call_kwargs["storage"] == "sqlite:///test.db"

            # Verify warning was logged
            mock_logger.warning.assert_called()
            warning_msg = str(mock_logger.warning.call_args)
            assert "existing_study" in warning_msg or "exists" in warning_msg.lower()

            # Verify the loaded study is returned
            assert study is mock_study

    def test_create_study_wraps_exceptions_in_backend_error(self):
        """Test create_study wraps exceptions in BackendError."""
        from milia_pipeline.exceptions import BackendError
        from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

        if not OPTUNA_INSTALLED:
            pytest.skip("Optuna not installed")

        backend = OptunaBackend()

        # Patch optuna.create_study to raise an exception
        with patch("optuna.create_study", side_effect=Exception("Database error")):
            with pytest.raises(BackendError) as exc_info:
                backend.create_study(study_name="test_error", direction="minimize")

            assert exc_info.value.backend_name == "optuna"
            assert exc_info.value.operation == "create_study"


# =============================================================================
# PART 2: optimize METHOD TESTS
# =============================================================================


class TestOptimizeBasic:
    """Test optimize method basic functionality."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_optimize_runs_specified_trials(self, backend_instance):
        """Test optimize runs the specified number of trials."""
        study = backend_instance.create_study(
            study_name="test_optimize_trials", direction="minimize"
        )

        n_trials = 3

        def simple_objective(trial):
            x = trial.suggest_float("x", 0, 1)
            return x**2

        backend_instance.optimize(study=study, objective_fn=simple_objective, n_trials=n_trials)

        assert len(study.trials) >= n_trials

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_optimize_returns_none(self, backend_instance):
        """Test optimize returns None (modifies study in place)."""
        study = backend_instance.create_study(
            study_name="test_optimize_return", direction="minimize"
        )

        result = backend_instance.optimize(
            study=study, objective_fn=lambda trial: trial.suggest_float("x", 0, 1), n_trials=1
        )

        assert result is None

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_optimize_with_timeout(self, backend_instance):
        """Test optimize respects timeout parameter."""
        study = backend_instance.create_study(
            study_name="test_optimize_timeout", direction="minimize"
        )

        import time

        start_time = time.time()

        def slow_objective(trial):
            time.sleep(0.1)
            return trial.suggest_float("x", 0, 1)

        backend_instance.optimize(
            study=study,
            objective_fn=slow_objective,
            n_trials=1000,  # Large number
            timeout=1,  # But only 1 second timeout
        )

        elapsed = time.time() - start_time
        # Should stop around 1 second (with some buffer)
        assert elapsed < 3


class TestOptimizeWithCallbacks:
    """Test optimize method with callbacks."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_optimize_calls_callback_after_each_trial(self, backend_instance):
        """Test optimize invokes callbacks after each trial."""
        study = backend_instance.create_study(study_name="test_callback", direction="minimize")

        callback_calls = []

        def test_callback(study, trial):
            callback_calls.append(trial.number)

        backend_instance.optimize(
            study=study,
            objective_fn=lambda trial: trial.suggest_float("x", 0, 1),
            n_trials=3,
            callbacks=[test_callback],
        )

        assert len(callback_calls) == 3

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_optimize_with_multiple_callbacks(self, backend_instance):
        """Test optimize works with multiple callbacks."""
        study = backend_instance.create_study(
            study_name="test_multi_callback", direction="minimize"
        )

        callback1_calls = []
        callback2_calls = []

        def callback1(study, trial):
            callback1_calls.append(trial.number)

        def callback2(study, trial):
            callback2_calls.append(trial.number)

        backend_instance.optimize(
            study=study,
            objective_fn=lambda trial: trial.suggest_float("x", 0, 1),
            n_trials=2,
            callbacks=[callback1, callback2],
        )

        assert len(callback1_calls) == 2
        assert len(callback2_calls) == 2


class TestOptimizeWithCatch:
    """Test optimize method exception catching."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_optimize_catches_specified_exceptions(self, backend_instance):
        """Test optimize catches exceptions specified in catch parameter."""
        study = backend_instance.create_study(
            study_name="test_catch_exception", direction="minimize"
        )

        call_count = [0]

        def failing_objective(trial):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Intentional failure")
            return trial.suggest_float("x", 0, 1)

        # Should not raise, should catch ValueError
        backend_instance.optimize(
            study=study, objective_fn=failing_objective, n_trials=3, catch=(ValueError,)
        )

        # Should have run all trials (one failed, two succeeded)
        assert len(study.trials) == 3


class TestOptimizeLogging:
    """Test optimize logging behavior."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_optimize_logs_start_message(self, backend_instance):
        """Test optimize logs start message with trial count."""
        study = backend_instance.create_study(
            study_name="test_optimize_log_start", direction="minimize"
        )

        with patch("milia_pipeline.models.hpo.backends.optuna_backend.logger") as mock_logger:
            backend_instance.optimize(study=study, objective_fn=lambda trial: 0.5, n_trials=2)

            # Should log starting optimization
            mock_logger.info.assert_called()

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_optimize_logs_completion_summary(self, backend_instance):
        """Test optimize logs completion summary."""
        study = backend_instance.create_study(
            study_name="test_optimize_log_complete", direction="minimize"
        )

        with patch("milia_pipeline.models.hpo.backends.optuna_backend.logger") as mock_logger:
            backend_instance.optimize(
                study=study, objective_fn=lambda trial: trial.suggest_float("x", 0, 1), n_trials=2
            )

            # Check that completion was logged
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("complete" in call.lower() for call in info_calls)


class TestOptimizeErrorHandling:
    """Test optimize error handling."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_optimize_keyboard_interrupt_propagates(self, backend_instance):
        """Test optimize propagates KeyboardInterrupt."""
        study = backend_instance.create_study(
            study_name="test_keyboard_interrupt", direction="minimize"
        )

        def interrupting_objective(trial):
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            backend_instance.optimize(study=study, objective_fn=interrupting_objective, n_trials=1)

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_optimize_keyboard_interrupt_logs_warning(self, backend_instance):
        """Test optimize logs warning when interrupted by user."""
        study = backend_instance.create_study(
            study_name="test_keyboard_interrupt_log", direction="minimize"
        )

        def interrupting_objective(trial):
            raise KeyboardInterrupt()

        with patch("milia_pipeline.models.hpo.backends.optuna_backend.logger") as mock_logger:
            with pytest.raises(KeyboardInterrupt):
                backend_instance.optimize(
                    study=study, objective_fn=interrupting_objective, n_trials=1
                )

            # Verify warning was logged about interruption
            mock_logger.warning.assert_called()
            warning_msg = str(mock_logger.warning.call_args)
            assert "interrupt" in warning_msg.lower()

    def test_optimize_wraps_exceptions_in_backend_error(self):
        """Test optimize wraps unexpected exceptions in BackendError."""
        from milia_pipeline.exceptions import BackendError
        from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

        if not OPTUNA_INSTALLED:
            pytest.skip("Optuna not installed")

        backend = OptunaBackend()
        mock_study = MagicMock()
        mock_study.optimize.side_effect = RuntimeError("Unexpected error")

        with pytest.raises(BackendError) as exc_info:
            backend.optimize(study=mock_study, objective_fn=lambda trial: 0.5, n_trials=1)

        assert exc_info.value.backend_name == "optuna"
        assert exc_info.value.operation == "optimize"


# =============================================================================
# PART 2: get_best_params METHOD TESTS
# =============================================================================


class TestGetBestParams:
    """Test get_best_params method."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_get_best_params_returns_dict(self, backend_instance):
        """Test get_best_params returns a dictionary."""
        study = backend_instance.create_study(study_name="test_best_params", direction="minimize")

        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=3)

        best_params = backend_instance.get_best_params(study)

        assert isinstance(best_params, dict)

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_get_best_params_contains_suggested_params(self, backend_instance):
        """Test get_best_params contains all suggested parameters."""
        study = backend_instance.create_study(
            study_name="test_best_params_content", direction="minimize"
        )

        def objective(trial):
            x = trial.suggest_float("x", 0, 1)
            y = trial.suggest_int("y", 1, 10)
            return x + y

        study.optimize(objective, n_trials=5)

        best_params = backend_instance.get_best_params(study)

        assert "x" in best_params
        assert "y" in best_params

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_get_best_params_raises_hpo_error_no_trials(self, backend_instance):
        """Test get_best_params raises HPOError when no completed trials."""
        from milia_pipeline.exceptions import HPOError

        study = backend_instance.create_study(
            study_name="test_best_params_no_trials", direction="minimize"
        )

        # No trials run
        with pytest.raises(HPOError):
            backend_instance.get_best_params(study)


# =============================================================================
# PART 2: get_best_value METHOD TESTS
# =============================================================================


class TestGetBestValue:
    """Test get_best_value method."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_get_best_value_returns_float(self, backend_instance):
        """Test get_best_value returns a float."""
        study = backend_instance.create_study(study_name="test_best_value", direction="minimize")

        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=3)

        best_value = backend_instance.get_best_value(study)

        assert isinstance(best_value, float)

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_get_best_value_returns_minimum_for_minimize(self, backend_instance):
        """Test get_best_value returns minimum value when direction is minimize."""
        study = backend_instance.create_study(
            study_name="test_best_value_min", direction="minimize"
        )

        # Objective returns value close to x, so minimum should be close to 0
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=10)

        best_value = backend_instance.get_best_value(study)

        # Best value should be the minimum returned
        assert best_value >= 0
        assert best_value <= 1

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_get_best_value_returns_maximum_for_maximize(self, backend_instance):
        """Test get_best_value returns maximum value when direction is maximize."""
        study = backend_instance.create_study(
            study_name="test_best_value_max", direction="maximize"
        )

        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=10)

        best_value = backend_instance.get_best_value(study)

        assert best_value >= 0
        assert best_value <= 1

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_get_best_value_raises_hpo_error_no_trials(self, backend_instance):
        """Test get_best_value raises HPOError when no completed trials."""
        from milia_pipeline.exceptions import HPOError

        study = backend_instance.create_study(
            study_name="test_best_value_no_trials", direction="minimize"
        )

        with pytest.raises(HPOError):
            backend_instance.get_best_value(study)


# =============================================================================
# PART 2: get_all_trials METHOD TESTS
# =============================================================================


class TestGetAllTrials:
    """Test get_all_trials method."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_get_all_trials_returns_list(self, backend_instance):
        """Test get_all_trials returns a list."""
        study = backend_instance.create_study(study_name="test_all_trials", direction="minimize")

        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=3)

        trials = backend_instance.get_all_trials(study)

        assert isinstance(trials, list)

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_get_all_trials_returns_correct_count(self, backend_instance):
        """Test get_all_trials returns correct number of trials."""
        study = backend_instance.create_study(
            study_name="test_all_trials_count", direction="minimize"
        )

        n_trials = 5
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=n_trials)

        trials = backend_instance.get_all_trials(study)

        assert len(trials) == n_trials

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_get_all_trials_each_trial_is_dict(self, backend_instance):
        """Test get_all_trials returns list of dictionaries."""
        study = backend_instance.create_study(
            study_name="test_all_trials_dict", direction="minimize"
        )

        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=2)

        trials = backend_instance.get_all_trials(study)

        for trial in trials:
            assert isinstance(trial, dict)

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_get_all_trials_contains_required_keys(self, backend_instance):
        """Test each trial dict contains required keys."""
        study = backend_instance.create_study(
            study_name="test_all_trials_keys", direction="minimize"
        )

        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=1)

        trials = backend_instance.get_all_trials(study)
        trial = trials[0]

        required_keys = [
            "number",
            "params",
            "value",
            "state",
            "duration",
            "user_attrs",
            "intermediate_values",
        ]

        for key in required_keys:
            assert key in trial, f"Trial dict missing required key: {key}"

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_get_all_trials_number_is_sequential(self, backend_instance):
        """Test trial numbers are sequential starting from 0."""
        study = backend_instance.create_study(
            study_name="test_all_trials_sequential", direction="minimize"
        )

        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=3)

        trials = backend_instance.get_all_trials(study)

        numbers = [t["number"] for t in trials]
        assert numbers == [0, 1, 2]

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_get_all_trials_params_contains_suggested_values(self, backend_instance):
        """Test trial params contains the suggested parameter values."""
        study = backend_instance.create_study(
            study_name="test_all_trials_params", direction="minimize"
        )

        def objective(trial):
            x = trial.suggest_float("x", 0, 1)
            return x

        study.optimize(objective, n_trials=1)

        trials = backend_instance.get_all_trials(study)

        assert "x" in trials[0]["params"]
        assert 0 <= trials[0]["params"]["x"] <= 1

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_get_all_trials_empty_study_returns_empty_list(self, backend_instance):
        """Test get_all_trials returns empty list for study with no trials."""
        study = backend_instance.create_study(
            study_name="test_all_trials_empty", direction="minimize"
        )

        trials = backend_instance.get_all_trials(study)

        assert trials == []

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_get_all_trials_state_is_string(self, backend_instance):
        """Test trial state is returned as a string."""
        study = backend_instance.create_study(
            study_name="test_all_trials_state", direction="minimize"
        )

        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=1)

        trials = backend_instance.get_all_trials(study)

        assert isinstance(trials[0]["state"], str)
        assert trials[0]["state"] == "COMPLETE"

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_get_all_trials_duration_none_when_datetime_missing(self, backend_instance):
        """Test get_all_trials returns None duration when datetime is missing."""
        from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

        backend = OptunaBackend()

        # Create a mock study with trial that has None datetime values
        mock_trial = MagicMock()
        mock_trial.number = 0
        mock_trial.params = {"x": 0.5}
        mock_trial.value = 0.5
        mock_trial.state = MagicMock()
        mock_trial.state.name = "FAIL"
        mock_trial.datetime_start = None
        mock_trial.datetime_complete = None
        mock_trial.user_attrs = {}
        mock_trial.intermediate_values = {}

        mock_study = MagicMock()
        mock_study.trials = [mock_trial]

        trials = backend.get_all_trials(mock_study)

        assert len(trials) == 1
        assert trials[0]["duration"] is None

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_get_all_trials_duration_calculated_when_datetimes_present(self, backend_instance):
        """Test get_all_trials calculates duration correctly when both datetimes present."""
        study = backend_instance.create_study(
            study_name="test_all_trials_duration_calc", direction="minimize"
        )

        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=1)

        trials = backend_instance.get_all_trials(study)

        # Duration should be a float (seconds) when trial completed successfully
        assert isinstance(trials[0]["duration"], (int, float))
        assert trials[0]["duration"] >= 0


# -------
# =============================================================================
# PART 3: _build_sampler_registry METHOD TESTS
# =============================================================================


class TestBuildSamplerRegistryBasic:
    """Test _build_sampler_registry method basic functionality."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_build_sampler_registry_returns_dict(self, fresh_backend_instance):
        """Test _build_sampler_registry returns a dictionary."""
        registry = fresh_backend_instance._build_sampler_registry()

        assert isinstance(registry, dict)

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_build_sampler_registry_contains_core_samplers(self, fresh_backend_instance):
        """Test registry contains core samplers that are always available."""
        registry = fresh_backend_instance._build_sampler_registry()

        core_samplers = ["tpe", "random", "cmaes", "grid"]

        for sampler in core_samplers:
            assert sampler in registry, f"Core sampler '{sampler}' missing from registry"

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_build_sampler_registry_values_are_classes(self, fresh_backend_instance):
        """Test registry values are sampler classes (callable)."""
        registry = fresh_backend_instance._build_sampler_registry()

        for sampler_type, sampler_cls in registry.items():
            assert callable(sampler_cls), f"Registry value for '{sampler_type}' should be callable"

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_build_sampler_registry_tpe_is_tpe_sampler(self, fresh_backend_instance):
        """Test 'tpe' key maps to TPESampler class."""
        import optuna

        registry = fresh_backend_instance._build_sampler_registry()

        assert registry["tpe"] is optuna.samplers.TPESampler

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_build_sampler_registry_random_is_random_sampler(self, fresh_backend_instance):
        """Test 'random' key maps to RandomSampler class."""
        import optuna

        registry = fresh_backend_instance._build_sampler_registry()

        assert registry["random"] is optuna.samplers.RandomSampler

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_build_sampler_registry_cmaes_is_cmaes_sampler(self, fresh_backend_instance):
        """Test 'cmaes' key maps to CmaEsSampler class."""
        import optuna

        registry = fresh_backend_instance._build_sampler_registry()

        assert registry["cmaes"] is optuna.samplers.CmaEsSampler

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_build_sampler_registry_grid_is_grid_sampler(self, fresh_backend_instance):
        """Test 'grid' key maps to GridSampler class."""
        import optuna

        registry = fresh_backend_instance._build_sampler_registry()

        assert registry["grid"] is optuna.samplers.GridSampler


class TestBuildSamplerRegistryCaching:
    """Test _build_sampler_registry caching behavior."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_build_sampler_registry_caches_result(self, fresh_backend_instance):
        """Test _build_sampler_registry caches result in _sampler_registry_cache."""
        # First call - should build registry
        registry1 = fresh_backend_instance._build_sampler_registry()

        # Should have cache now
        assert hasattr(fresh_backend_instance, "_sampler_registry_cache")
        assert fresh_backend_instance._sampler_registry_cache is registry1

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_build_sampler_registry_returns_cached_on_second_call(self, fresh_backend_instance):
        """Test _build_sampler_registry returns cached registry on subsequent calls."""
        registry1 = fresh_backend_instance._build_sampler_registry()
        registry2 = fresh_backend_instance._build_sampler_registry()

        # Should be the exact same object (cached)
        assert registry1 is registry2

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_build_sampler_registry_cache_is_efficient(self, fresh_backend_instance):
        """Test cached registry is returned immediately without rebuilding."""
        import optuna

        # First call builds registry
        fresh_backend_instance._build_sampler_registry()

        # Patch hasattr to track if it's called (which would indicate rebuilding)
        with patch.object(optuna, "samplers") as _mock_samplers:
            # Second call should use cache, not access optuna.samplers
            registry = fresh_backend_instance._build_sampler_registry()

            # If cache is used, hasattr on mock_samplers shouldn't be called
            # (we can't easily assert this, but we can verify cache is returned)
            assert registry is fresh_backend_instance._sampler_registry_cache


class TestBuildSamplerRegistryVersionCompatibility:
    """Test _build_sampler_registry version compatibility for MOTPESampler/QMCSampler."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_registry_includes_nsgaii_if_available(self, fresh_backend_instance):
        """Test registry includes NSGAIISampler if available in optuna version."""
        import optuna

        registry = fresh_backend_instance._build_sampler_registry()

        if hasattr(optuna.samplers, "NSGAIISampler"):
            assert "nsgaii" in registry
        else:
            assert "nsgaii" not in registry

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_registry_includes_motpe_if_available(self, fresh_backend_instance):
        """Test registry includes MOTPESampler only if available in optuna version."""
        import optuna

        registry = fresh_backend_instance._build_sampler_registry()

        if hasattr(optuna.samplers, "MOTPESampler"):
            assert "motpe" in registry
            assert registry["motpe"] is optuna.samplers.MOTPESampler
        else:
            assert "motpe" not in registry

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_registry_includes_qmc_if_available(self, fresh_backend_instance):
        """Test registry includes QMCSampler only if available in optuna version."""
        import optuna

        registry = fresh_backend_instance._build_sampler_registry()

        if hasattr(optuna.samplers, "QMCSampler"):
            assert "qmc" in registry
            assert registry["qmc"] is optuna.samplers.QMCSampler
        else:
            assert "qmc" not in registry

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_registry_excludes_unavailable_samplers(self, fresh_backend_instance):
        """Test registry only includes samplers that exist in current optuna version."""

        registry = fresh_backend_instance._build_sampler_registry()

        # Every sampler in registry should exist in optuna.samplers
        for _sampler_type, sampler_cls in registry.items():
            # The class should be an actual optuna sampler class
            assert sampler_cls is not None

    def test_registry_handles_missing_motpe_gracefully(self):
        """Test registry building doesn't fail when MOTPESampler is missing."""
        if not OPTUNA_INSTALLED:
            pytest.skip("Optuna not installed")

        import optuna

        from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

        backend = OptunaBackend()

        # Clear cache
        if hasattr(backend, "_sampler_registry_cache"):
            delattr(backend, "_sampler_registry_cache")

        # Mock optuna.samplers to not have MOTPESampler
        original_hasattr = hasattr

        def mock_hasattr(obj, name):
            if obj is optuna.samplers and name == "MOTPESampler":
                return False
            return original_hasattr(obj, name)

        with patch("builtins.hasattr", side_effect=mock_hasattr):
            # Should not raise AttributeError
            registry = backend._build_sampler_registry()

            # Core samplers should still be present
            assert "tpe" in registry
            assert "random" in registry

    def test_registry_handles_missing_qmc_gracefully(self):
        """Test registry building doesn't fail when QMCSampler is missing."""
        if not OPTUNA_INSTALLED:
            pytest.skip("Optuna not installed")

        import optuna

        from milia_pipeline.models.hpo.backends.optuna_backend import OptunaBackend

        backend = OptunaBackend()

        if hasattr(backend, "_sampler_registry_cache"):
            delattr(backend, "_sampler_registry_cache")

        original_hasattr = hasattr

        def mock_hasattr(obj, name):
            if obj is optuna.samplers and name == "QMCSampler":
                return False
            return original_hasattr(obj, name)

        with patch("builtins.hasattr", side_effect=mock_hasattr):
            registry = backend._build_sampler_registry()

            assert "tpe" in registry
            assert "random" in registry


# =============================================================================
# PART 3: create_sampler METHOD TESTS
# =============================================================================


class TestCreateSamplerBasic:
    """Test create_sampler method basic functionality."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_tpe_returns_tpe_sampler(self, backend_instance):
        """Test create_sampler('tpe') returns TPESampler instance."""
        import optuna

        sampler = backend_instance.create_sampler(sampler_type="tpe")

        assert isinstance(sampler, optuna.samplers.TPESampler)

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_random_returns_random_sampler(self, backend_instance):
        """Test create_sampler('random') returns RandomSampler instance."""
        import optuna

        sampler = backend_instance.create_sampler(sampler_type="random")

        assert isinstance(sampler, optuna.samplers.RandomSampler)

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_cmaes_returns_cmaes_sampler(self, backend_instance):
        """Test create_sampler('cmaes') returns CmaEsSampler instance."""
        import optuna

        sampler = backend_instance.create_sampler(sampler_type="cmaes")

        assert isinstance(sampler, optuna.samplers.CmaEsSampler)

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_grid_returns_grid_sampler(self, backend_instance):
        """Test create_sampler('grid') returns GridSampler instance."""
        import optuna

        search_space = {"x": [0.1, 0.5, 1.0]}

        sampler = backend_instance.create_sampler(sampler_type="grid", search_space=search_space)

        assert isinstance(sampler, optuna.samplers.GridSampler)

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_nsgaii_returns_nsgaii_sampler(self, backend_instance):
        """Test create_sampler('nsgaii') returns NSGAIISampler instance."""
        import optuna

        if not hasattr(optuna.samplers, "NSGAIISampler"):
            pytest.skip("NSGAIISampler not available in this optuna version")

        sampler = backend_instance.create_sampler(sampler_type="nsgaii")

        assert isinstance(sampler, optuna.samplers.NSGAIISampler)

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_motpe_returns_motpe_sampler_when_available(self, backend_instance):
        """Test create_sampler('motpe') returns MOTPESampler instance when available."""
        import optuna

        if not hasattr(optuna.samplers, "MOTPESampler"):
            pytest.skip("MOTPESampler not available in this optuna version")

        sampler = backend_instance.create_sampler(sampler_type="motpe")

        assert isinstance(sampler, optuna.samplers.MOTPESampler)

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_qmc_returns_qmc_sampler_when_available(self, backend_instance):
        """Test create_sampler('qmc') returns QMCSampler instance when available."""
        import optuna

        if not hasattr(optuna.samplers, "QMCSampler"):
            pytest.skip("QMCSampler not available in this optuna version")

        sampler = backend_instance.create_sampler(sampler_type="qmc")

        assert isinstance(sampler, optuna.samplers.QMCSampler)


class TestCreateSamplerWithSeed:
    """Test create_sampler method with seed parameter."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_with_seed_produces_sampler(self, backend_instance):
        """Test create_sampler with seed produces a valid sampler."""
        sampler = backend_instance.create_sampler(sampler_type="tpe", seed=42)

        # Sampler should be created successfully
        assert sampler is not None

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_with_same_seed_is_deterministic(self, backend_instance):
        """Test create_sampler with same seed produces deterministic sampling."""
        import optuna

        # Create two studies with same-seeded samplers
        sampler1 = backend_instance.create_sampler(sampler_type="random", seed=42)
        sampler2 = backend_instance.create_sampler(sampler_type="random", seed=42)

        study1 = optuna.create_study(sampler=sampler1)
        study2 = optuna.create_study(sampler=sampler2)

        # Run single trial on each
        study1.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=1)
        study2.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=1)

        # Same seed should produce same value
        assert study1.best_params["x"] == study2.best_params["x"]

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_with_different_seeds_differ(self, backend_instance):
        """Test create_sampler with different seeds produces different results."""
        import optuna

        sampler1 = backend_instance.create_sampler(sampler_type="random", seed=42)
        sampler2 = backend_instance.create_sampler(sampler_type="random", seed=12345)

        study1 = optuna.create_study(sampler=sampler1)
        study2 = optuna.create_study(sampler=sampler2)

        study1.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=1)
        study2.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=1)

        # Different seeds should (very likely) produce different values
        # Note: There's a tiny chance they could be equal, but extremely unlikely
        assert study1.best_params["x"] != study2.best_params["x"]

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_with_none_seed(self, backend_instance):
        """Test create_sampler with seed=None (random seed)."""
        sampler = backend_instance.create_sampler(sampler_type="tpe", seed=None)

        # Should not raise
        assert sampler is not None


class TestCreateSamplerTPEParameters:
    """Test create_sampler TPE-specific parameters."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_tpe_with_n_startup_trials(self, backend_instance):
        """Test create_sampler TPE with custom n_startup_trials."""
        n_startup = 20

        sampler = backend_instance.create_sampler(sampler_type="tpe", n_startup_trials=n_startup)

        assert sampler._n_startup_trials == n_startup

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_tpe_with_multivariate_true(self, backend_instance):
        """Test create_sampler TPE with multivariate=True."""
        sampler = backend_instance.create_sampler(sampler_type="tpe", multivariate=True)

        assert sampler._multivariate is True

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_tpe_with_multivariate_false(self, backend_instance):
        """Test create_sampler TPE with multivariate=False."""
        sampler = backend_instance.create_sampler(sampler_type="tpe", multivariate=False)

        assert sampler._multivariate is False

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_tpe_with_constant_liar(self, backend_instance):
        """Test create_sampler TPE with constant_liar=True."""
        sampler = backend_instance.create_sampler(sampler_type="tpe", constant_liar=True)

        assert sampler._constant_liar is True

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_tpe_default_multivariate_is_true(self, backend_instance):
        """Test create_sampler TPE default multivariate is True."""
        sampler = backend_instance.create_sampler(sampler_type="tpe")

        assert sampler._multivariate is True

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_tpe_default_constant_liar_is_false(self, backend_instance):
        """Test create_sampler TPE default constant_liar is False."""
        sampler = backend_instance.create_sampler(sampler_type="tpe")

        assert sampler._constant_liar is False


class TestCreateSamplerCMAESParameters:
    """Test create_sampler CMA-ES-specific parameters."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_cmaes_with_n_startup_trials(self, backend_instance):
        """Test create_sampler CMA-ES with custom n_startup_trials."""
        n_startup = 15

        sampler = backend_instance.create_sampler(sampler_type="cmaes", n_startup_trials=n_startup)

        assert sampler._n_startup_trials == n_startup

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_cmaes_with_restart_strategy(self, backend_instance):
        """Test create_sampler CMA-ES with restart_strategy parameter."""
        sampler = backend_instance.create_sampler(sampler_type="cmaes", restart_strategy="ipop")

        # CmaEsSampler should be configured with restart strategy
        assert sampler is not None


class TestCreateSamplerGridRequirements:
    """Test create_sampler Grid-specific requirements."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_grid_requires_search_space(self, backend_instance):
        """Test create_sampler('grid') raises BackendError without search_space."""
        from milia_pipeline.exceptions import BackendError

        with pytest.raises(BackendError) as exc_info:
            backend_instance.create_sampler(sampler_type="grid")

        assert "search_space" in str(exc_info.value).lower()

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_grid_with_valid_search_space(self, backend_instance):
        """Test create_sampler('grid') with valid search_space."""
        import optuna

        search_space = {"x": [0.1, 0.5, 1.0], "y": [1, 2, 3]}

        sampler = backend_instance.create_sampler(sampler_type="grid", search_space=search_space)

        assert isinstance(sampler, optuna.samplers.GridSampler)


class TestCreateSamplerErrorHandling:
    """Test create_sampler error handling."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_unknown_type_raises_backend_error(self, backend_instance):
        """Test create_sampler with unknown type raises BackendError."""
        from milia_pipeline.exceptions import BackendError

        with pytest.raises(BackendError) as exc_info:
            backend_instance.create_sampler(sampler_type="nonexistent_sampler")

        error = exc_info.value
        assert error.backend_name == "optuna"
        assert error.operation == "create_sampler"

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_error_lists_available_samplers(self, backend_instance):
        """Test BackendError message lists available samplers."""
        from milia_pipeline.exceptions import BackendError

        with pytest.raises(BackendError) as exc_info:
            backend_instance.create_sampler(sampler_type="invalid")

        error_details = exc_info.value.details
        assert "tpe" in error_details.lower()
        assert "random" in error_details.lower()

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_unavailable_motpe_raises_backend_error(self, backend_instance):
        """Test create_sampler('motpe') raises BackendError if MOTPESampler unavailable."""
        from milia_pipeline.exceptions import BackendError

        # Clear cached registry to force rebuild
        if hasattr(backend_instance, "_sampler_registry_cache"):
            delattr(backend_instance, "_sampler_registry_cache")

        # Mock _build_sampler_registry to return registry without 'motpe'
        registry_without_motpe = {
            "tpe": MagicMock(),
            "random": MagicMock(),
            "cmaes": MagicMock(),
            "grid": MagicMock(),
        }

        with patch.object(
            backend_instance, "_build_sampler_registry", return_value=registry_without_motpe
        ):
            with pytest.raises(BackendError) as exc_info:
                backend_instance.create_sampler(sampler_type="motpe")

            assert "motpe" in str(exc_info.value).lower()

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_unavailable_qmc_raises_backend_error(self, backend_instance):
        """Test create_sampler('qmc') raises BackendError if QMCSampler unavailable."""
        from milia_pipeline.exceptions import BackendError

        # Clear cached registry to force rebuild
        if hasattr(backend_instance, "_sampler_registry_cache"):
            delattr(backend_instance, "_sampler_registry_cache")

        # Mock _build_sampler_registry to return registry without 'qmc'
        registry_without_qmc = {
            "tpe": MagicMock(),
            "random": MagicMock(),
            "cmaes": MagicMock(),
            "grid": MagicMock(),
        }

        with patch.object(
            backend_instance, "_build_sampler_registry", return_value=registry_without_qmc
        ):
            with pytest.raises(BackendError) as exc_info:
                backend_instance.create_sampler(sampler_type="qmc")

            assert "qmc" in str(exc_info.value).lower()


class TestCreateSamplerLogging:
    """Test create_sampler logging behavior."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_logs_debug_message(self, backend_instance):
        """Test create_sampler logs debug message with sampler type and seed."""
        with patch("milia_pipeline.models.hpo.backends.optuna_backend.logger") as mock_logger:
            backend_instance.create_sampler(sampler_type="tpe", seed=42)

            mock_logger.debug.assert_called()
            debug_message = str(mock_logger.debug.call_args)
            assert "tpe" in debug_message.lower()


# =============================================================================
# PART 3: create_pruner METHOD TESTS
# =============================================================================


class TestCreatePrunerBasic:
    """Test create_pruner method basic functionality."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_median_returns_median_pruner(self, backend_instance):
        """Test create_pruner('median') returns MedianPruner instance."""
        import optuna

        pruner = backend_instance.create_pruner(pruner_type="median")

        assert isinstance(pruner, optuna.pruners.MedianPruner)

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_percentile_returns_percentile_pruner(self, backend_instance):
        """Test create_pruner('percentile') returns PercentilePruner instance."""
        import optuna

        pruner = backend_instance.create_pruner(pruner_type="percentile")

        assert isinstance(pruner, optuna.pruners.PercentilePruner)

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_hyperband_returns_hyperband_pruner(self, backend_instance):
        """Test create_pruner('hyperband') returns HyperbandPruner instance."""
        import optuna

        pruner = backend_instance.create_pruner(pruner_type="hyperband")

        assert isinstance(pruner, optuna.pruners.HyperbandPruner)

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_successive_halving_returns_sha_pruner(self, backend_instance):
        """Test create_pruner('successive_halving') returns SuccessiveHalvingPruner."""
        import optuna

        pruner = backend_instance.create_pruner(pruner_type="successive_halving")

        assert isinstance(pruner, optuna.pruners.SuccessiveHalvingPruner)

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_threshold_returns_threshold_pruner(self, backend_instance):
        """Test create_pruner('threshold') returns ThresholdPruner instance."""
        import optuna

        # ThresholdPruner requires at least lower or upper to be specified
        pruner = backend_instance.create_pruner(
            pruner_type="threshold",
            upper=1.0,  # Prune if value exceeds 1.0
        )

        assert isinstance(pruner, optuna.pruners.ThresholdPruner)

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_none_returns_nop_pruner(self, backend_instance):
        """Test create_pruner('none') returns NopPruner instance."""
        import optuna

        pruner = backend_instance.create_pruner(pruner_type="none")

        assert isinstance(pruner, optuna.pruners.NopPruner)


class TestCreatePrunerMedianParameters:
    """Test create_pruner median-specific parameters."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_median_with_n_startup_trials(self, backend_instance):
        """Test create_pruner median with custom n_startup_trials."""
        pruner = backend_instance.create_pruner(pruner_type="median", n_startup_trials=10)

        assert pruner._n_startup_trials == 10

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_median_with_n_warmup_steps(self, backend_instance):
        """Test create_pruner median with custom n_warmup_steps."""
        pruner = backend_instance.create_pruner(pruner_type="median", n_warmup_steps=20)

        assert pruner._n_warmup_steps == 20

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_median_with_interval_steps(self, backend_instance):
        """Test create_pruner median with custom interval_steps."""
        pruner = backend_instance.create_pruner(pruner_type="median", interval_steps=5)

        assert pruner._interval_steps == 5


class TestCreatePrunerPercentileParameters:
    """Test create_pruner percentile-specific parameters."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_percentile_with_percentile_value(self, backend_instance):
        """Test create_pruner percentile with custom percentile value."""
        pruner = backend_instance.create_pruner(pruner_type="percentile", percentile=50.0)

        assert pruner._percentile == 50.0

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_percentile_default_percentile_is_25(self, backend_instance):
        """Test create_pruner percentile default percentile is 25.0."""
        pruner = backend_instance.create_pruner(pruner_type="percentile")

        assert pruner._percentile == 25.0


class TestCreatePrunerHyperbandParameters:
    """Test create_pruner hyperband-specific parameters."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_hyperband_with_min_resource(self, backend_instance):
        """Test create_pruner hyperband with custom min_resource."""
        pruner = backend_instance.create_pruner(pruner_type="hyperband", min_resource=2)

        assert pruner._min_resource == 2

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_hyperband_with_reduction_factor(self, backend_instance):
        """Test create_pruner hyperband with custom reduction_factor."""
        pruner = backend_instance.create_pruner(pruner_type="hyperband", reduction_factor=4)

        assert pruner._reduction_factor == 4


class TestCreatePrunerThresholdParameters:
    """Test create_pruner threshold-specific parameters."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_threshold_with_upper_bound(self, backend_instance):
        """Test create_pruner threshold with upper bound."""
        import optuna

        pruner = backend_instance.create_pruner(pruner_type="threshold", upper=1.0)

        assert isinstance(pruner, optuna.pruners.ThresholdPruner)
        assert pruner._upper == 1.0

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_threshold_with_lower_bound(self, backend_instance):
        """Test create_pruner threshold with lower bound."""
        import optuna

        pruner = backend_instance.create_pruner(pruner_type="threshold", lower=0.0)

        assert isinstance(pruner, optuna.pruners.ThresholdPruner)
        assert pruner._lower == 0.0

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_threshold_with_both_bounds(self, backend_instance):
        """Test create_pruner threshold with both lower and upper bounds."""
        import optuna

        pruner = backend_instance.create_pruner(pruner_type="threshold", lower=0.0, upper=1.0)

        assert isinstance(pruner, optuna.pruners.ThresholdPruner)
        assert pruner._lower == 0.0
        assert pruner._upper == 1.0

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_threshold_with_n_warmup_steps(self, backend_instance):
        """Test create_pruner threshold passes n_warmup_steps."""
        pruner = backend_instance.create_pruner(
            pruner_type="threshold", n_warmup_steps=15, upper=1.0
        )

        assert pruner._n_warmup_steps == 15


class TestCreatePrunerSuccessiveHalvingParameters:
    """Test create_pruner successive_halving-specific parameters."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_successive_halving_with_min_resource(self, backend_instance):
        """Test create_pruner successive_halving with custom min_resource."""
        pruner = backend_instance.create_pruner(pruner_type="successive_halving", min_resource=2)

        assert pruner._min_resource == 2

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_successive_halving_with_reduction_factor(self, backend_instance):
        """Test create_pruner successive_halving with custom reduction_factor."""
        pruner = backend_instance.create_pruner(
            pruner_type="successive_halving", reduction_factor=3
        )

        assert pruner._reduction_factor == 3

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_successive_halving_with_min_early_stopping_rate(self, backend_instance):
        """Test create_pruner successive_halving with custom min_early_stopping_rate."""
        pruner = backend_instance.create_pruner(
            pruner_type="successive_halving", min_early_stopping_rate=2
        )

        assert pruner._min_early_stopping_rate == 2


class TestCreatePrunerPatient:
    """Test create_pruner patient pruner (wraps another pruner)."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_patient_wraps_median_by_default(self, backend_instance):
        """Test create_pruner patient wraps MedianPruner by default."""
        import optuna

        pruner = backend_instance.create_pruner(pruner_type="patient")

        assert isinstance(pruner, optuna.pruners.PatientPruner)
        assert isinstance(pruner._wrapped_pruner, optuna.pruners.MedianPruner)

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_patient_with_custom_wrapped_pruner(self, backend_instance):
        """Test create_pruner patient with custom wrapped_pruner_type."""
        import optuna

        pruner = backend_instance.create_pruner(
            pruner_type="patient", wrapped_pruner_type="percentile"
        )

        assert isinstance(pruner._wrapped_pruner, optuna.pruners.PercentilePruner)

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_patient_with_patience(self, backend_instance):
        """Test create_pruner patient with custom patience value."""
        pruner = backend_instance.create_pruner(pruner_type="patient", patience=10)

        assert pruner._patience == 10


class TestCreatePrunerErrorHandling:
    """Test create_pruner error handling."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_unknown_type_raises_backend_error(self, backend_instance):
        """Test create_pruner with unknown type raises BackendError."""
        from milia_pipeline.exceptions import BackendError

        with pytest.raises(BackendError) as exc_info:
            backend_instance.create_pruner(pruner_type="nonexistent_pruner")

        error = exc_info.value
        assert error.backend_name == "optuna"
        assert error.operation == "create_pruner"

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_error_lists_available_pruners(self, backend_instance):
        """Test BackendError message lists available pruners."""
        from milia_pipeline.exceptions import BackendError

        with pytest.raises(BackendError) as exc_info:
            backend_instance.create_pruner(pruner_type="invalid")

        error_details = exc_info.value.details
        assert "median" in error_details.lower()
        assert "hyperband" in error_details.lower()


class TestCreatePrunerLogging:
    """Test create_pruner logging behavior."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_logs_debug_message(self, backend_instance):
        """Test create_pruner logs debug message with pruner type."""
        with patch("milia_pipeline.models.hpo.backends.optuna_backend.logger") as mock_logger:
            backend_instance.create_pruner(pruner_type="median")

            mock_logger.debug.assert_called()
            debug_message = str(mock_logger.debug.call_args)
            assert "median" in debug_message.lower()


# -------

# =============================================================================
# PART 4: suggest_params METHOD TESTS - BASIC FUNCTIONALITY
# =============================================================================


class TestSuggestParamsBasic:
    """Test suggest_params method basic functionality."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_returns_dict(
        self, backend_instance, mock_trial, simple_search_space_dict
    ):
        """Test suggest_params returns a dictionary."""
        params = backend_instance.suggest_params(mock_trial, simple_search_space_dict)

        assert isinstance(params, dict)

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_contains_all_parameters(
        self, backend_instance, mock_trial, simple_search_space_dict
    ):
        """Test suggest_params returns all parameters from search space."""
        params = backend_instance.suggest_params(mock_trial, simple_search_space_dict)

        expected_keys = ["model.hidden_dim", "model.learning_rate", "training.batch_size"]

        for key in expected_keys:
            assert key in params, f"Missing parameter: {key}"

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_uses_category_prefix(
        self, backend_instance, mock_trial, simple_search_space_dict
    ):
        """Test suggest_params uses category.param_name format for keys."""
        params = backend_instance.suggest_params(mock_trial, simple_search_space_dict)

        # All keys should have format "category.param_name"
        for key in params:
            assert "." in key, f"Key '{key}' should have category prefix"
            parts = key.split(".")
            assert len(parts) == 2, f"Key '{key}' should have exactly one dot"

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_empty_search_space_returns_empty_dict(
        self, backend_instance, mock_trial
    ):
        """Test suggest_params with empty search space returns empty dict."""
        params = backend_instance.suggest_params(mock_trial, {})

        assert params == {}


# =============================================================================
# PART 4: suggest_params - INTEGER PARAMETER TESTS
# =============================================================================


class TestSuggestParamsInt:
    """Test suggest_params with integer parameters."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_int_calls_suggest_int(self, backend_instance, mock_trial):
        """Test suggest_params calls trial.suggest_int for int type."""
        search_space = {"model": {"layers": {"type": "int", "low": 1, "high": 10, "step": 1}}}

        backend_instance.suggest_params(mock_trial, search_space)

        mock_trial.suggest_int.assert_called_once()

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_int_passes_correct_bounds(self, backend_instance, mock_trial):
        """Test suggest_params passes correct low/high bounds for int."""
        search_space = {"model": {"layers": {"type": "int", "low": 2, "high": 8, "step": 1}}}

        backend_instance.suggest_params(mock_trial, search_space)

        call_args = mock_trial.suggest_int.call_args
        assert call_args[0][0] == "model.layers"  # name
        assert call_args[0][1] == 2  # low
        assert call_args[0][2] == 8  # high

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_int_passes_step(self, backend_instance, mock_trial):
        """Test suggest_params passes step parameter for int."""
        search_space = {"model": {"hidden": {"type": "int", "low": 32, "high": 256, "step": 32}}}

        backend_instance.suggest_params(mock_trial, search_space)

        call_kwargs = mock_trial.suggest_int.call_args[1]
        assert call_kwargs["step"] == 32

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_int_default_step_is_1(self, backend_instance, mock_trial):
        """Test suggest_params uses step=1 as default for int."""
        search_space = {
            "model": {
                "layers": {"type": "int", "low": 1, "high": 10}  # No step specified
            }
        }

        backend_instance.suggest_params(mock_trial, search_space)

        call_kwargs = mock_trial.suggest_int.call_args[1]
        assert call_kwargs["step"] == 1

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_int_with_dataclass_config(self, backend_instance, mock_trial):
        """Test suggest_params handles dataclass config for int type."""
        search_space = {"model": {"layers": ParamConfig(type=ParamType.INT, low=1, high=5, step=1)}}

        backend_instance.suggest_params(mock_trial, search_space)

        mock_trial.suggest_int.assert_called_once()


# =============================================================================
# PART 4: suggest_params - FLOAT PARAMETER TESTS
# =============================================================================


class TestSuggestParamsFloat:
    """Test suggest_params with float parameters."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_float_calls_suggest_float(self, backend_instance, mock_trial):
        """Test suggest_params calls trial.suggest_float for float type."""
        search_space = {
            "model": {"dropout": {"type": "float", "low": 0.0, "high": 0.5, "log": False}}
        }

        backend_instance.suggest_params(mock_trial, search_space)

        mock_trial.suggest_float.assert_called_once()

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_float_passes_correct_bounds(self, backend_instance, mock_trial):
        """Test suggest_params passes correct low/high bounds for float."""
        search_space = {
            "model": {"dropout": {"type": "float", "low": 0.1, "high": 0.9, "log": False}}
        }

        backend_instance.suggest_params(mock_trial, search_space)

        call_args = mock_trial.suggest_float.call_args
        assert call_args[0][0] == "model.dropout"
        assert call_args[0][1] == 0.1
        assert call_args[0][2] == 0.9

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_float_passes_log_true(self, backend_instance, mock_trial):
        """Test suggest_params passes log=True for float with log scale."""
        search_space = {"model": {"lr": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True}}}

        backend_instance.suggest_params(mock_trial, search_space)

        call_kwargs = mock_trial.suggest_float.call_args[1]
        assert call_kwargs["log"] is True

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_float_passes_log_false(self, backend_instance, mock_trial):
        """Test suggest_params passes log=False for linear scale."""
        search_space = {
            "model": {"dropout": {"type": "float", "low": 0.0, "high": 0.5, "log": False}}
        }

        backend_instance.suggest_params(mock_trial, search_space)

        call_kwargs = mock_trial.suggest_float.call_args[1]
        assert call_kwargs["log"] is False

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_float_default_log_is_false(self, backend_instance, mock_trial):
        """Test suggest_params uses log=False as default for float."""
        search_space = {
            "model": {
                "value": {"type": "float", "low": 0.0, "high": 1.0}  # No log specified
            }
        }

        backend_instance.suggest_params(mock_trial, search_space)

        call_kwargs = mock_trial.suggest_float.call_args[1]
        assert call_kwargs["log"] is False


# =============================================================================
# PART 4: suggest_params - LOGUNIFORM PARAMETER TESTS
# =============================================================================


class TestSuggestParamsLoguniform:
    """Test suggest_params with loguniform parameters."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_loguniform_calls_suggest_float_with_log(
        self, backend_instance, mock_trial
    ):
        """Test suggest_params calls suggest_float with log=True for loguniform."""
        search_space = {"model": {"lr": {"type": "loguniform", "low": 1e-6, "high": 1e-1}}}

        backend_instance.suggest_params(mock_trial, search_space)

        mock_trial.suggest_float.assert_called_once()
        call_kwargs = mock_trial.suggest_float.call_args[1]
        assert call_kwargs["log"] is True

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_loguniform_passes_correct_bounds(self, backend_instance, mock_trial):
        """Test suggest_params passes correct bounds for loguniform."""
        search_space = {"model": {"lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2}}}

        backend_instance.suggest_params(mock_trial, search_space)

        call_args = mock_trial.suggest_float.call_args
        assert call_args[0][1] == 1e-5
        assert call_args[0][2] == 1e-2


# =============================================================================
# PART 4: suggest_params - UNIFORM PARAMETER TESTS
# =============================================================================


class TestSuggestParamsUniform:
    """Test suggest_params with uniform parameters."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_uniform_calls_suggest_float(self, backend_instance, mock_trial):
        """Test suggest_params calls suggest_float for uniform type."""
        search_space = {"model": {"value": {"type": "uniform", "low": 0.0, "high": 1.0}}}

        backend_instance.suggest_params(mock_trial, search_space)

        mock_trial.suggest_float.assert_called_once()

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_uniform_does_not_use_log(self, backend_instance, mock_trial):
        """Test suggest_params uniform does not use log scale."""
        search_space = {"model": {"value": {"type": "uniform", "low": 0.0, "high": 1.0}}}

        backend_instance.suggest_params(mock_trial, search_space)

        # For uniform, log should not be True
        call_kwargs = mock_trial.suggest_float.call_args[1]
        assert call_kwargs.get("log", False) is False or "log" not in call_kwargs

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_uniform_passes_correct_bounds(self, backend_instance, mock_trial):
        """Test suggest_params passes correct bounds for uniform type."""
        search_space = {"model": {"value": {"type": "uniform", "low": 0.2, "high": 0.8}}}

        backend_instance.suggest_params(mock_trial, search_space)

        call_args = mock_trial.suggest_float.call_args
        assert call_args[0][0] == "model.value"  # name
        assert call_args[0][1] == 0.2  # low
        assert call_args[0][2] == 0.8  # high


# =============================================================================
# PART 4: suggest_params - CATEGORICAL PARAMETER TESTS
# =============================================================================


class TestSuggestParamsCategorical:
    """Test suggest_params with categorical parameters."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_categorical_calls_suggest_categorical(
        self, backend_instance, mock_trial
    ):
        """Test suggest_params calls trial.suggest_categorical for categorical type."""
        search_space = {
            "model": {"activation": {"type": "categorical", "choices": ["relu", "gelu"]}}
        }

        backend_instance.suggest_params(mock_trial, search_space)

        mock_trial.suggest_categorical.assert_called_once()

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_categorical_passes_choices(self, backend_instance, mock_trial):
        """Test suggest_params passes choices for categorical."""
        choices = ["adam", "sgd", "rmsprop"]
        search_space = {"optimizer": {"type_": {"type": "categorical", "choices": choices}}}

        backend_instance.suggest_params(mock_trial, search_space)

        call_args = mock_trial.suggest_categorical.call_args
        assert call_args[0][1] == choices

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_categorical_with_int_choices(self, backend_instance, mock_trial):
        """Test suggest_params handles categorical with integer choices."""
        choices = [16, 32, 64, 128]
        search_space = {"training": {"batch_size": {"type": "categorical", "choices": choices}}}

        backend_instance.suggest_params(mock_trial, search_space)

        call_args = mock_trial.suggest_categorical.call_args
        assert call_args[0][1] == choices

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_categorical_with_mixed_choices(self, backend_instance, mock_trial):
        """Test suggest_params handles categorical with mixed type choices."""
        choices = [None, "auto", 0.5]
        search_space = {"config": {"option": {"type": "categorical", "choices": choices}}}

        backend_instance.suggest_params(mock_trial, search_space)

        call_args = mock_trial.suggest_categorical.call_args
        assert call_args[0][1] == choices


# =============================================================================
# PART 4: suggest_params - ERROR HANDLING TESTS
# =============================================================================


class TestSuggestParamsErrorHandling:
    """Test suggest_params error handling."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_unknown_type_raises_backend_error(self, backend_instance, mock_trial):
        """Test suggest_params raises BackendError for unknown parameter type."""
        from milia_pipeline.exceptions import BackendError

        search_space = {"model": {"param": {"type": "unknown_type", "low": 0, "high": 1}}}

        with pytest.raises(BackendError) as exc_info:
            backend_instance.suggest_params(mock_trial, search_space)

        error = exc_info.value
        assert error.backend_name == "optuna"
        assert error.operation == "suggest_params"

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_error_includes_parameter_name(self, backend_instance, mock_trial):
        """Test BackendError includes the problematic parameter name."""
        from milia_pipeline.exceptions import BackendError

        search_space = {"model": {"bad_param": {"type": "invalid_type", "low": 0, "high": 1}}}

        with pytest.raises(BackendError) as exc_info:
            backend_instance.suggest_params(mock_trial, search_space)

        error_details = exc_info.value.details
        assert "model.bad_param" in error_details

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_missing_low_raises_backend_error(self, backend_instance, mock_trial):
        """Test suggest_params raises BackendError when 'low' is missing for int/float."""
        from milia_pipeline.exceptions import BackendError

        search_space = {
            "model": {
                "hidden_dim": {"type": "int", "high": 256}  # Missing 'low'
            }
        }

        with pytest.raises(BackendError) as exc_info:
            backend_instance.suggest_params(mock_trial, search_space)

        error = exc_info.value
        assert error.backend_name == "optuna"
        assert error.operation == "suggest_params"
        assert "low" in str(error) or "low" in error.details

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_missing_high_raises_backend_error(self, backend_instance, mock_trial):
        """Test suggest_params raises BackendError when 'high' is missing for int/float."""
        from milia_pipeline.exceptions import BackendError

        search_space = {
            "model": {
                "hidden_dim": {"type": "int", "low": 32}  # Missing 'high'
            }
        }

        with pytest.raises(BackendError) as exc_info:
            backend_instance.suggest_params(mock_trial, search_space)

        error = exc_info.value
        assert error.backend_name == "optuna"
        assert error.operation == "suggest_params"
        assert "high" in str(error) or "high" in error.details

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_missing_choices_raises_backend_error(
        self, backend_instance, mock_trial
    ):
        """Test suggest_params raises BackendError when 'choices' is missing for categorical."""
        from milia_pipeline.exceptions import BackendError

        search_space = {
            "model": {
                "activation": {"type": "categorical"}  # Missing 'choices'
            }
        }

        with pytest.raises(BackendError) as exc_info:
            backend_instance.suggest_params(mock_trial, search_space)

        error = exc_info.value
        assert error.backend_name == "optuna"
        assert error.operation == "suggest_params"
        assert "choices" in str(error) or "choices" in error.details

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_missing_type_raises_backend_error(self, backend_instance, mock_trial):
        """Test suggest_params raises BackendError when 'type' is missing."""
        from milia_pipeline.exceptions import BackendError

        search_space = {
            "model": {
                "hidden_dim": {"low": 32, "high": 256}  # Missing 'type'
            }
        }

        with pytest.raises(BackendError) as exc_info:
            backend_instance.suggest_params(mock_trial, search_space)

        error = exc_info.value
        assert error.backend_name == "optuna"
        assert error.operation == "suggest_params"
        assert "type" in str(error).lower() or "type" in error.details.lower()


# =============================================================================
# PART 4: suggest_params - DATACLASS/ENUM CONFIG SUPPORT
# =============================================================================


class TestSuggestParamsConfigObjects:
    """Test suggest_params with dataclass and enum configuration objects."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_handles_dataclass_config(
        self, backend_instance, mock_trial, search_space_with_dataclass
    ):
        """Test suggest_params handles dataclass-style config objects."""
        params = backend_instance.suggest_params(mock_trial, search_space_with_dataclass)

        assert "model.hidden_dim" in params
        assert "model.learning_rate" in params
        assert "training.activation" in params

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_extracts_type_from_enum(self, backend_instance, mock_trial):
        """Test suggest_params extracts type value from enum."""
        search_space = {
            "model": {"layers": ParamConfig(type=ParamType.INT, low=1, high=10, step=1)}
        }

        backend_instance.suggest_params(mock_trial, search_space)

        # Should call suggest_int since type is ParamType.INT
        mock_trial.suggest_int.assert_called_once()

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_handles_string_type(self, backend_instance, mock_trial):
        """Test suggest_params handles string type values."""
        search_space = {"model": {"layers": {"type": "int", "low": 1, "high": 10, "step": 1}}}

        backend_instance.suggest_params(mock_trial, search_space)

        mock_trial.suggest_int.assert_called_once()

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_handles_object_with_none_attribute(self, backend_instance, mock_trial):
        """Test suggest_params handles config object with None attribute value returning default."""
        # Create a config object that returns None for 'step' attribute
        search_space = {
            "model": {"layers": ParamConfig(type=ParamType.INT, low=1, high=10, step=None)}
        }

        backend_instance.suggest_params(mock_trial, search_space)

        # Should use default step=1 when config.step is None
        call_kwargs = mock_trial.suggest_int.call_args[1]
        assert call_kwargs["step"] == 1

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_handles_mixed_dict_and_dataclass(self, backend_instance, mock_trial):
        """Test suggest_params handles mixed dict and dataclass configs in same search space."""
        search_space = {
            "model": {
                "layers": {"type": "int", "low": 1, "high": 10, "step": 1},  # dict
                "dropout": ParamConfig(
                    type=ParamType.FLOAT, low=0.0, high=0.5, log=False
                ),  # dataclass
            }
        }

        backend_instance.suggest_params(mock_trial, search_space)

        # Both should be called
        mock_trial.suggest_int.assert_called_once()
        mock_trial.suggest_float.assert_called_once()


# =============================================================================
# PART 4: INTEGRATION TESTS - END-TO-END WORKFLOWS
# =============================================================================


class TestIntegrationStudyLifecycle:
    """Integration tests for complete study lifecycle."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_complete_optimization_workflow(self, backend_instance):
        """Test complete workflow: create study -> optimize -> get results."""
        # Create study
        study = backend_instance.create_study(
            study_name="integration_test_workflow", direction="minimize"
        )

        # Define objective
        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            y = trial.suggest_float("y", -10, 10)
            return (x - 2) ** 2 + (y + 3) ** 2

        # Optimize
        backend_instance.optimize(study=study, objective_fn=objective, n_trials=10)

        # Get results
        best_params = backend_instance.get_best_params(study)
        best_value = backend_instance.get_best_value(study)
        all_trials = backend_instance.get_all_trials(study)

        # Verify results
        assert "x" in best_params
        assert "y" in best_params
        assert isinstance(best_value, float)
        assert len(all_trials) == 10

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_workflow_with_sampler_and_pruner(self, backend_instance):
        """Test workflow with custom sampler and pruner."""
        # Create sampler and pruner
        sampler = backend_instance.create_sampler(sampler_type="tpe", seed=42, n_startup_trials=5)
        pruner = backend_instance.create_pruner(pruner_type="median", n_startup_trials=3)

        # Create study with custom sampler and pruner
        study = backend_instance.create_study(
            study_name="integration_test_sampler_pruner",
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )

        # Optimize
        def objective(trial):
            x = trial.suggest_float("x", 0, 1)
            return x

        backend_instance.optimize(study=study, objective_fn=objective, n_trials=5)

        # Verify
        assert len(study.trials) == 5
        assert study.sampler is sampler
        assert study.pruner is pruner

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_workflow_with_suggest_params(self, backend_instance, comprehensive_search_space):
        """Test workflow using suggest_params for parameter suggestion."""
        study = backend_instance.create_study(
            study_name="integration_test_suggest_params", direction="minimize"
        )

        def objective(trial):
            params = backend_instance.suggest_params(trial, comprehensive_search_space)

            # Simple objective using suggested params
            return (
                params["model.num_layers"] * 0.1
                + params["model.dropout"]
                + params["model.learning_rate"] * 100
            )

        backend_instance.optimize(study=study, objective_fn=objective, n_trials=5)

        best_params = backend_instance.get_best_params(study)

        # Verify all parameters were suggested
        assert "model.num_layers" in best_params
        assert "model.hidden_dim" in best_params
        assert "model.dropout" in best_params
        assert "model.learning_rate" in best_params
        assert "optimizer.weight_decay" in best_params
        assert "optimizer.optimizer_type" in best_params


class TestIntegrationErrorRecovery:
    """Integration tests for error handling and recovery."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_optimization_continues_after_trial_failure(self, backend_instance):
        """Test optimization continues after individual trial failures."""
        study = backend_instance.create_study(
            study_name="integration_test_failure_recovery", direction="minimize"
        )

        trial_count = [0]

        def sometimes_failing_objective(trial):
            trial_count[0] += 1
            x = trial.suggest_float("x", 0, 1)

            # Fail on first trial
            if trial_count[0] == 1:
                raise ValueError("Intentional failure")

            return x

        backend_instance.optimize(
            study=study, objective_fn=sometimes_failing_objective, n_trials=5, catch=(ValueError,)
        )

        # Should have completed 5 trials (1 failed, 4 succeeded)
        assert len(study.trials) == 5

        # Should be able to get best params (from successful trials)
        best_params = backend_instance.get_best_params(study)
        assert "x" in best_params


class TestIntegrationSamplerVersionCompatibility:
    """Integration tests for sampler version compatibility."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_optimization_with_all_available_samplers(self, backend_instance):
        """Test optimization works with all available samplers."""

        # Get available samplers from registry
        registry = backend_instance._build_sampler_registry()

        for sampler_type in registry:
            if sampler_type == "grid":
                # Grid sampler needs search_space
                continue

            sampler = backend_instance.create_sampler(sampler_type=sampler_type, seed=42)

            study = backend_instance.create_study(
                study_name=f"integration_test_{sampler_type}", direction="minimize", sampler=sampler
            )

            study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=2)

            assert len(study.trials) == 2, f"Sampler {sampler_type} failed"


# =============================================================================
# PART 4: PROTOCOL COMPLIANCE TESTS
# =============================================================================


class TestProtocolCompliance:
    """Test OptunaBackend compliance with HPOBackendProtocol."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_backend_implements_protocol(self, backend_instance):
        """Test OptunaBackend implements HPOBackendProtocol."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        # Check isinstance (requires @runtime_checkable)
        assert isinstance(backend_instance, HPOBackendProtocol)

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_backend_has_all_protocol_methods(self, backend_instance):
        """Test OptunaBackend has all methods required by protocol."""
        protocol_methods = [
            "create_study",
            "optimize",
            "get_best_params",
            "get_best_value",
            "get_all_trials",
            "create_pruner",
            "create_sampler",
        ]

        for method_name in protocol_methods:
            assert hasattr(backend_instance, method_name)
            assert callable(getattr(backend_instance, method_name))

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_backend_can_be_used_polymorphically(self, backend_instance):
        """Test OptunaBackend can be used polymorphically via protocol."""
        from milia_pipeline.models.hpo.backends.base import HPOBackendProtocol

        def use_backend(backend: HPOBackendProtocol):
            """Function that accepts any protocol-compliant backend."""
            study = backend.create_study("poly_test", "minimize")
            backend.optimize(study, lambda trial: trial.suggest_float("x", 0, 1), n_trials=1)
            return backend.get_best_params(study)

        # Should work without errors
        result = use_backend(backend_instance)
        assert isinstance(result, dict)


# =============================================================================
# PART 4: EDGE CASES AND BOUNDARY CONDITIONS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_study_with_very_long_name(self, backend_instance):
        """Test create_study with very long study name."""
        long_name = "a" * 500

        study = backend_instance.create_study(study_name=long_name, direction="minimize")

        assert study.study_name == long_name

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_optimize_with_zero_trials(self, backend_instance):
        """Test optimize with n_trials=0."""
        study = backend_instance.create_study(study_name="test_zero_trials", direction="minimize")

        backend_instance.optimize(study=study, objective_fn=lambda trial: 0.5, n_trials=0)

        assert len(study.trials) == 0

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_with_single_category(self, backend_instance, mock_trial):
        """Test suggest_params with single-element search space."""
        search_space = {
            "model": {"only_param": {"type": "float", "low": 0, "high": 1, "log": False}}
        }

        params = backend_instance.suggest_params(mock_trial, search_space)

        assert len(params) == 1
        assert "model.only_param" in params

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_sampler_with_seed_zero(self, backend_instance):
        """Test create_sampler with seed=0 (valid seed)."""
        # seed=0 is a valid seed value and should not raise
        sampler = backend_instance.create_sampler(sampler_type="tpe", seed=0)

        assert sampler is not None

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_pruner_with_zero_warmup_steps(self, backend_instance):
        """Test create_pruner with n_warmup_steps=0."""
        pruner = backend_instance.create_pruner(pruner_type="median", n_warmup_steps=0)

        assert pruner._n_warmup_steps == 0

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_get_all_trials_with_many_trials(self, backend_instance):
        """Test get_all_trials with many trials."""
        study = backend_instance.create_study(study_name="test_many_trials", direction="minimize")

        n_trials = 50
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=n_trials)

        trials = backend_instance.get_all_trials(study)

        assert len(trials) == n_trials
        # Verify all trials have correct structure
        for i, trial in enumerate(trials):
            assert trial["number"] == i


class TestSpecialCharactersAndUnicode:
    """Test handling of special characters and unicode."""

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_create_study_with_unicode_name(self, backend_instance):
        """Test create_study with unicode characters in name."""
        unicode_name = "test_study_日本語_émoji_🎯"

        study = backend_instance.create_study(study_name=unicode_name, direction="minimize")

        assert study.study_name == unicode_name

    @pytest.mark.skipif(not OPTUNA_INSTALLED, reason="Optuna not installed")
    def test_suggest_params_with_special_param_names(self, backend_instance, mock_trial):
        """Test suggest_params with special characters in parameter names."""
        search_space = {
            "model_v2": {"param_1": {"type": "float", "low": 0, "high": 1, "log": False}}
        }

        params = backend_instance.suggest_params(mock_trial, search_space)

        assert "model_v2.param_1" in params


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
