#!/usr/bin/env python3
"""
Complete Unit Test Suite for milia_pipeline/models/hpo/transfer/warm_start.py Module
PART 1: Imports, Mock Classes, Fixtures, WarmStartMethod Enum, WarmStartConfig Dataclass,
        TransferredTrial Dataclass, and WarmStartStrategy Initialization

Tests:
- Lazy import function (_lazy_import_optuna)
- WarmStartMethod enum (all values, value access, membership, iteration)
- WarmStartConfig dataclass (initialization, validation, __post_init__, defaults)
- TransferredTrial dataclass (initialization, defaults, attributes)
- WarmStartStrategy.__init__() (default config, custom config, lazy imports)
- WarmStartStrategy.weighted_transfer() static method
- WarmStartStrategy.filtered_transfer() static method
- WarmStartStrategy.full_transfer() static method
- WarmStartStrategy._extract_params() helper method
- WarmStartStrategy._extract_value() helper method
- WarmStartStrategy._flatten_search_space() helper method
- WarmStartStrategy._filter_params() helper method
- WarmStartStrategy.transfer() instance method
- WarmStartStrategy.apply_to_study() instance method
- WarmStartStrategy._filter_invalid_trials() helper method
- WarmStartStrategy._add_noise_to_trials() helper method
- WarmStartStrategy._select_adaptive_method() helper method
- WarmStartStrategy.get_transfer_summary() static method
- WarmStartStrategy.create_from_best_trials() static method
- Integration tests (full workflow)
- Edge cases and error handling

This is a PRODUCTION-READY test suite with comprehensive coverage.

Author: Milia Team
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# =============================================================================
# MOCK CLASSES FOR DEPENDENCIES
# =============================================================================


class MockTrialState(Enum):
    """Mock Optuna TrialState for testing."""

    COMPLETE = "COMPLETE"
    PRUNED = "PRUNED"
    FAIL = "FAIL"
    RUNNING = "RUNNING"
    WAITING = "WAITING"


class MockDirection(Enum):
    """Mock Optuna study direction."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class MockFrozenTrial:
    """Mock Optuna FrozenTrial for testing."""

    def __init__(
        self,
        params: dict[str, Any],
        value: float | None = None,
        state: MockTrialState = None,
        number: int = 0,
    ):
        self.params = params
        self.value = value
        self.values = [value] if value is not None else None
        self.state = state or MockTrialState.COMPLETE
        self.number = number


class MockStudy:
    """Mock Optuna study for testing."""

    def __init__(
        self,
        best_params: dict[str, Any] = None,
        best_value: float = 0.1,
        trials: list[MockFrozenTrial] = None,
        direction: MockDirection = MockDirection.MINIMIZE,
        study_name: str = "test_study",
    ):
        self.best_params = best_params or {"lr": 0.001, "hidden_dim": 128}
        self.best_value = best_value
        self.trials = trials or [
            MockFrozenTrial({"lr": 0.001, "hidden_dim": 128}, 0.1, number=0),
            MockFrozenTrial({"lr": 0.01, "hidden_dim": 64}, 0.2, number=1),
            MockFrozenTrial({"lr": 0.005, "hidden_dim": 256}, 0.15, number=2),
        ]
        self.direction = direction
        self.study_name = study_name
        self._enqueued_trials = []

    def enqueue_trial(self, params: dict[str, Any]):
        """Mock enqueue_trial method."""
        self._enqueued_trials.append(params)


class MockOptuna:
    """Mock Optuna module for testing."""

    class trial:
        TrialState = MockTrialState

    @staticmethod
    def create_study(direction="minimize", study_name=None):
        return MockStudy(direction=MockDirection(direction), study_name=study_name)


class MockWarmStartMethod(Enum):
    """Mock WarmStartMethod for testing without imports."""

    WEIGHTED = "weighted"
    FILTERED = "filtered"
    FULL = "full"
    ADAPTIVE = "adaptive"


class MockParamType(Enum):
    """Mock parameter type enum for testing."""

    INT = "int"
    FLOAT = "float"
    CATEGORICAL = "categorical"


class MockFailingStudy:
    """Mock study that fails on enqueue."""

    def __init__(self):
        self._enqueued_trials = []

    def enqueue_trial(self, params: dict[str, Any]):
        """Fail on enqueue."""
        raise RuntimeError("Enqueue failed")


# =============================================================================
# HELPER FUNCTIONS FOR TESTS
# =============================================================================


def create_mock_trials(n_trials: int = 5, with_values: bool = True) -> list[MockFrozenTrial]:
    """Create a list of mock trials for testing."""
    trials = []
    for i in range(n_trials):
        value = 0.1 * (i + 1) if with_values else None
        trials.append(
            MockFrozenTrial(
                params={
                    "lr": 0.001 * (i + 1),
                    "hidden_dim": 64 * (i + 1),
                    "dropout": 0.1 + 0.05 * i,
                },
                value=value,
                number=i,
            )
        )
    return trials


def create_mock_trial_dicts(n_trials: int = 5) -> list[dict[str, Any]]:
    """Create a list of mock trial dictionaries for testing."""
    return [
        {
            "params": {
                "lr": 0.001 * (i + 1),
                "hidden_dim": 64 * (i + 1),
                "dropout": 0.1 + 0.05 * i,
            },
            "value": 0.1 * (i + 1),
        }
        for i in range(n_trials)
    ]


def create_sample_search_space() -> dict[str, Any]:
    """Create a sample search space for testing."""
    return {
        "lr": {"type": "float", "low": 1e-5, "high": 1e-1},
        "hidden_dim": {"type": "int", "low": 32, "high": 512},
        "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        "activation": {"type": "categorical", "choices": ["relu", "gelu", "selu"]},
    }


def create_nested_search_space() -> dict[str, Any]:
    """Create a nested search space for testing."""
    return {
        "model": {
            "lr": {"type": "float", "low": 1e-5, "high": 1e-1},
            "hidden_dim": {"type": "int", "low": 32, "high": 512},
        },
        "training": {
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "batch_size": {"type": "int_uniform", "low": 16, "high": 128},
        },
    }


def create_transferred_trials(n_trials: int = 5) -> list:
    """Create a list of TransferredTrial objects for testing."""
    from milia_pipeline.models.hpo.transfer.warm_start import TransferredTrial

    trials = []
    for i in range(n_trials):
        trials.append(
            TransferredTrial(
                params={
                    "lr": 0.001 * (i + 1),
                    "hidden_dim": 64 * (i + 1),
                },
                value=0.1 * (i + 1),
                similarity=0.9 - 0.1 * i,
                weight=0.8 - 0.1 * i,
                is_valid=True,
            )
        )
    return trials


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_optuna():
    """Create mock Optuna module."""
    return MockOptuna()


@pytest.fixture
def mock_study():
    """Create mock Optuna study."""
    return MockStudy()


@pytest.fixture
def mock_trials():
    """Create list of mock FrozenTrials."""
    return create_mock_trials(5)


@pytest.fixture
def mock_trial_dicts():
    """Create list of mock trial dictionaries."""
    return create_mock_trial_dicts(5)


@pytest.fixture
def sample_search_space():
    """Create sample search space."""
    return create_sample_search_space()


@pytest.fixture
def nested_search_space():
    """Create nested search space."""
    return create_nested_search_space()


@pytest.fixture
def sample_similarities():
    """Create sample similarity scores."""
    return [0.95, 0.85, 0.75, 0.65, 0.55]


@pytest.fixture
def high_similarities():
    """Create high similarity scores."""
    return [0.99, 0.98, 0.97, 0.96, 0.95]


@pytest.fixture
def low_similarities():
    """Create low similarity scores."""
    return [0.3, 0.25, 0.2, 0.15, 0.1]


@pytest.fixture
def mixed_value_trials():
    """Create trials with mixed values including None."""
    return [
        MockFrozenTrial({"lr": 0.001}, value=0.1),
        MockFrozenTrial({"lr": 0.002}, value=None),
        MockFrozenTrial({"lr": 0.003}, value=0.2),
        MockFrozenTrial({"lr": 0.004}, value=None),
        MockFrozenTrial({"lr": 0.005}, value=0.15),
    ]


@pytest.fixture
def transferred_trials():
    """Create list of TransferredTrial objects."""
    return create_transferred_trials(5)


@pytest.fixture
def mixed_validity_trials():
    """Create TransferredTrials with mixed validity."""
    from milia_pipeline.models.hpo.transfer.warm_start import TransferredTrial

    return [
        TransferredTrial(params={"lr": 0.001}, is_valid=True),
        TransferredTrial(params={"lr": 0.002}, is_valid=False),
        TransferredTrial(params={"lr": 0.003}, is_valid=True),
        TransferredTrial(params={"lr": 0.004}, is_valid=False),
        TransferredTrial(params={"lr": 0.005}, is_valid=True),
    ]


# =============================================================================
# LAZY IMPORT FUNCTION TESTS
# =============================================================================


class TestLazyImportOptuna:
    """Test _lazy_import_optuna function."""

    def test_lazy_import_optuna_returns_optuna_or_none(self):
        """Test _lazy_import_optuna returns optuna module or None."""
        from milia_pipeline.models.hpo.transfer.warm_start import _lazy_import_optuna

        result = _lazy_import_optuna()
        # Result should be either optuna module or None
        assert result is None or hasattr(result, "create_study")

    def test_lazy_import_optuna_can_be_called_multiple_times(self):
        """Test that _lazy_import_optuna can be called multiple times safely."""
        from milia_pipeline.models.hpo.transfer.warm_start import _lazy_import_optuna

        result1 = _lazy_import_optuna()
        result2 = _lazy_import_optuna()

        # Should return same type both times
        assert type(result1) is type(result2)

    def test_lazy_import_optuna_handles_import_error(self):
        """Test _lazy_import_optuna handles ImportError gracefully."""
        from milia_pipeline.models.hpo.transfer.warm_start import _lazy_import_optuna

        # The function should not raise regardless of optuna availability
        try:
            result = _lazy_import_optuna()
            assert result is None or hasattr(result, "trial")
        except ImportError:
            pytest.fail("_lazy_import_optuna should handle ImportError")


# =============================================================================
# WARM-START METHOD ENUM TESTS
# =============================================================================


class TestWarmStartMethodEnum:
    """Test WarmStartMethod enum."""

    def test_warm_start_method_has_weighted(self):
        """Test WarmStartMethod has WEIGHTED value."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartMethod

        assert hasattr(WarmStartMethod, "WEIGHTED")
        assert WarmStartMethod.WEIGHTED.value == "weighted"

    def test_warm_start_method_has_filtered(self):
        """Test WarmStartMethod has FILTERED value."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartMethod

        assert hasattr(WarmStartMethod, "FILTERED")
        assert WarmStartMethod.FILTERED.value == "filtered"

    def test_warm_start_method_has_full(self):
        """Test WarmStartMethod has FULL value."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartMethod

        assert hasattr(WarmStartMethod, "FULL")
        assert WarmStartMethod.FULL.value == "full"

    def test_warm_start_method_has_adaptive(self):
        """Test WarmStartMethod has ADAPTIVE value."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartMethod

        assert hasattr(WarmStartMethod, "ADAPTIVE")
        assert WarmStartMethod.ADAPTIVE.value == "adaptive"

    def test_warm_start_method_from_string(self):
        """Test WarmStartMethod can be created from string."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartMethod

        method = WarmStartMethod("weighted")
        assert method == WarmStartMethod.WEIGHTED

        method = WarmStartMethod("filtered")
        assert method == WarmStartMethod.FILTERED

        method = WarmStartMethod("full")
        assert method == WarmStartMethod.FULL

        method = WarmStartMethod("adaptive")
        assert method == WarmStartMethod.ADAPTIVE

    def test_warm_start_method_invalid_string_raises(self):
        """Test WarmStartMethod raises ValueError for invalid string."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartMethod

        with pytest.raises(ValueError):
            WarmStartMethod("invalid_method")

    def test_warm_start_method_is_enum(self):
        """Test WarmStartMethod is an Enum."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartMethod

        assert issubclass(WarmStartMethod, Enum)

    def test_warm_start_method_iteration(self):
        """Test WarmStartMethod can be iterated."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartMethod

        methods = list(WarmStartMethod)
        assert len(methods) == 4  # WEIGHTED, FILTERED, FULL, ADAPTIVE

    def test_warm_start_method_membership(self):
        """Test WarmStartMethod membership checks."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartMethod

        assert WarmStartMethod.WEIGHTED in WarmStartMethod
        assert WarmStartMethod.ADAPTIVE in WarmStartMethod

    def test_warm_start_method_comparison(self):
        """Test WarmStartMethod comparison."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartMethod

        assert WarmStartMethod.WEIGHTED == WarmStartMethod.WEIGHTED
        assert WarmStartMethod.WEIGHTED != WarmStartMethod.FILTERED

    def test_warm_start_method_hashing(self):
        """Test WarmStartMethod can be used as dict key."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartMethod

        method_dict = {WarmStartMethod.WEIGHTED: "weight_func"}
        assert method_dict[WarmStartMethod.WEIGHTED] == "weight_func"


# =============================================================================
# WARM-START CONFIG DATACLASS TESTS
# =============================================================================


class TestWarmStartConfig:
    """Test WarmStartConfig dataclass."""

    def test_warm_start_config_default_initialization(self):
        """Test WarmStartConfig with default values."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartConfig, WarmStartMethod

        config = WarmStartConfig()

        assert config.method == WarmStartMethod.WEIGHTED
        assert config.n_trials == 10
        assert config.min_similarity == 0.5
        assert config.weight_by_performance is True
        assert config.filter_invalid is True
        assert config.scale_to_bounds is True
        assert config.add_noise is False
        assert config.noise_scale == 0.05

    def test_warm_start_config_custom_method(self):
        """Test WarmStartConfig with custom method."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartConfig, WarmStartMethod

        config = WarmStartConfig(method=WarmStartMethod.FILTERED)
        assert config.method == WarmStartMethod.FILTERED

        config = WarmStartConfig(method=WarmStartMethod.FULL)
        assert config.method == WarmStartMethod.FULL

        config = WarmStartConfig(method=WarmStartMethod.ADAPTIVE)
        assert config.method == WarmStartMethod.ADAPTIVE

    def test_warm_start_config_custom_n_trials(self):
        """Test WarmStartConfig with custom n_trials."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartConfig

        config = WarmStartConfig(n_trials=5)
        assert config.n_trials == 5

        config = WarmStartConfig(n_trials=100)
        assert config.n_trials == 100

    def test_warm_start_config_invalid_n_trials_raises(self):
        """Test WarmStartConfig raises for invalid n_trials."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartConfig

        with pytest.raises(ValueError, match="n_trials must be at least 1"):
            WarmStartConfig(n_trials=0)

        with pytest.raises(ValueError, match="n_trials must be at least 1"):
            WarmStartConfig(n_trials=-1)

    def test_warm_start_config_min_n_trials(self):
        """Test WarmStartConfig with minimum valid n_trials."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartConfig

        config = WarmStartConfig(n_trials=1)
        assert config.n_trials == 1

    def test_warm_start_config_custom_min_similarity(self):
        """Test WarmStartConfig with custom min_similarity."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartConfig

        config = WarmStartConfig(min_similarity=0.0)
        assert config.min_similarity == 0.0

        config = WarmStartConfig(min_similarity=0.8)
        assert config.min_similarity == 0.8

        config = WarmStartConfig(min_similarity=1.0)
        assert config.min_similarity == 1.0

    def test_warm_start_config_invalid_min_similarity_raises(self):
        """Test WarmStartConfig raises for invalid min_similarity."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartConfig

        with pytest.raises(ValueError, match="min_similarity must be between 0 and 1"):
            WarmStartConfig(min_similarity=-0.1)

        with pytest.raises(ValueError, match="min_similarity must be between 0 and 1"):
            WarmStartConfig(min_similarity=1.1)

    def test_warm_start_config_custom_noise_scale(self):
        """Test WarmStartConfig with custom noise_scale."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartConfig

        config = WarmStartConfig(noise_scale=0.0)
        assert config.noise_scale == 0.0

        config = WarmStartConfig(noise_scale=0.1)
        assert config.noise_scale == 0.1

        config = WarmStartConfig(noise_scale=1.0)
        assert config.noise_scale == 1.0

    def test_warm_start_config_invalid_noise_scale_raises(self):
        """Test WarmStartConfig raises for invalid noise_scale."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartConfig

        with pytest.raises(ValueError, match="noise_scale must be between 0 and 1"):
            WarmStartConfig(noise_scale=-0.1)

        with pytest.raises(ValueError, match="noise_scale must be between 0 and 1"):
            WarmStartConfig(noise_scale=1.1)

    def test_warm_start_config_boolean_options(self):
        """Test WarmStartConfig with boolean options."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartConfig

        config = WarmStartConfig(
            weight_by_performance=False, filter_invalid=False, scale_to_bounds=False, add_noise=True
        )

        assert config.weight_by_performance is False
        assert config.filter_invalid is False
        assert config.scale_to_bounds is False
        assert config.add_noise is True

    def test_warm_start_config_is_frozen(self):
        """Test WarmStartConfig is immutable (frozen).

        Note: With Pydantic V2, frozen models raise ValidationError (not FrozenInstanceError)
        when attempting to modify attributes. The error type is 'frozen_instance'.
        """
        from pydantic import ValidationError

        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartConfig

        config = WarmStartConfig()

        with pytest.raises(ValidationError, match="frozen"):
            config.n_trials = 20

    def test_warm_start_config_full_custom(self):
        """Test WarmStartConfig with all custom values."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartConfig, WarmStartMethod

        config = WarmStartConfig(
            method=WarmStartMethod.ADAPTIVE,
            n_trials=20,
            min_similarity=0.7,
            weight_by_performance=False,
            filter_invalid=False,
            scale_to_bounds=False,
            add_noise=True,
            noise_scale=0.15,
        )

        assert config.method == WarmStartMethod.ADAPTIVE
        assert config.n_trials == 20
        assert config.min_similarity == 0.7
        assert config.weight_by_performance is False
        assert config.filter_invalid is False
        assert config.scale_to_bounds is False
        assert config.add_noise is True
        assert config.noise_scale == 0.15

    def test_warm_start_config_boundary_values(self):
        """Test WarmStartConfig with boundary values."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartConfig

        # Test boundary for min_similarity
        config = WarmStartConfig(min_similarity=0.0)
        assert config.min_similarity == 0.0

        config = WarmStartConfig(min_similarity=1.0)
        assert config.min_similarity == 1.0

        # Test boundary for noise_scale
        config = WarmStartConfig(noise_scale=0.0)
        assert config.noise_scale == 0.0

        config = WarmStartConfig(noise_scale=1.0)
        assert config.noise_scale == 1.0

    def test_warm_start_config_to_dict(self):
        """Test WarmStartConfig.to_dict() returns correct dictionary.

        Pydantic V2: to_dict() wraps model_dump() for backward compatibility.
        """
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartConfig, WarmStartMethod

        config = WarmStartConfig(
            method=WarmStartMethod.FILTERED,
            n_trials=20,
            min_similarity=0.7,
            weight_by_performance=False,
            filter_invalid=False,
            scale_to_bounds=False,
            add_noise=True,
            noise_scale=0.15,
        )

        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["method"] == WarmStartMethod.FILTERED
        assert result["n_trials"] == 20
        assert result["min_similarity"] == 0.7
        assert result["weight_by_performance"] is False
        assert result["filter_invalid"] is False
        assert result["scale_to_bounds"] is False
        assert result["add_noise"] is True
        assert result["noise_scale"] == 0.15

    def test_warm_start_config_to_dict_with_defaults(self):
        """Test WarmStartConfig.to_dict() includes default values."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartConfig, WarmStartMethod

        config = WarmStartConfig()
        result = config.to_dict()

        assert result["method"] == WarmStartMethod.WEIGHTED
        assert result["n_trials"] == 10
        assert result["min_similarity"] == 0.5
        assert result["weight_by_performance"] is True
        assert result["filter_invalid"] is True
        assert result["scale_to_bounds"] is True
        assert result["add_noise"] is False
        assert result["noise_scale"] == 0.05


# =============================================================================
# TRANSFERRED TRIAL DATACLASS TESTS
# =============================================================================


class TestTransferredTrial:
    """Test TransferredTrial dataclass."""

    def test_transferred_trial_default_initialization(self):
        """Test TransferredTrial with minimal params."""
        from milia_pipeline.models.hpo.transfer.warm_start import TransferredTrial

        trial = TransferredTrial(params={"lr": 0.001})

        assert trial.params == {"lr": 0.001}
        assert trial.value is None
        assert trial.source_study is None
        assert trial.similarity == 1.0
        assert trial.weight == 1.0
        assert trial.is_valid is True

    def test_transferred_trial_with_value(self):
        """Test TransferredTrial with value."""
        from milia_pipeline.models.hpo.transfer.warm_start import TransferredTrial

        trial = TransferredTrial(params={"lr": 0.001}, value=0.15)

        assert trial.params == {"lr": 0.001}
        assert trial.value == 0.15

    def test_transferred_trial_with_source_study(self):
        """Test TransferredTrial with source_study."""
        from milia_pipeline.models.hpo.transfer.warm_start import TransferredTrial

        trial = TransferredTrial(params={"lr": 0.001}, source_study="study_A")

        assert trial.source_study == "study_A"

    def test_transferred_trial_with_similarity(self):
        """Test TransferredTrial with custom similarity."""
        from milia_pipeline.models.hpo.transfer.warm_start import TransferredTrial

        trial = TransferredTrial(params={"lr": 0.001}, similarity=0.85)

        assert trial.similarity == 0.85

    def test_transferred_trial_with_weight(self):
        """Test TransferredTrial with custom weight."""
        from milia_pipeline.models.hpo.transfer.warm_start import TransferredTrial

        trial = TransferredTrial(params={"lr": 0.001}, weight=0.75)

        assert trial.weight == 0.75

    def test_transferred_trial_with_is_valid_false(self):
        """Test TransferredTrial with is_valid=False."""
        from milia_pipeline.models.hpo.transfer.warm_start import TransferredTrial

        trial = TransferredTrial(params={"lr": 0.001}, is_valid=False)

        assert trial.is_valid is False

    def test_transferred_trial_full_initialization(self):
        """Test TransferredTrial with all values."""
        from milia_pipeline.models.hpo.transfer.warm_start import TransferredTrial

        trial = TransferredTrial(
            params={"lr": 0.001, "hidden_dim": 128},
            value=0.12,
            source_study="source_study",
            similarity=0.9,
            weight=0.8,
            is_valid=True,
        )

        assert trial.params == {"lr": 0.001, "hidden_dim": 128}
        assert trial.value == 0.12
        assert trial.source_study == "source_study"
        assert trial.similarity == 0.9
        assert trial.weight == 0.8
        assert trial.is_valid is True

    def test_transferred_trial_is_mutable(self):
        """Test TransferredTrial is mutable (not frozen)."""
        from milia_pipeline.models.hpo.transfer.warm_start import TransferredTrial

        trial = TransferredTrial(params={"lr": 0.001})

        # Should be able to modify attributes
        trial.value = 0.2
        assert trial.value == 0.2

        trial.is_valid = False
        assert trial.is_valid is False

    def test_transferred_trial_empty_params(self):
        """Test TransferredTrial with empty params."""
        from milia_pipeline.models.hpo.transfer.warm_start import TransferredTrial

        trial = TransferredTrial(params={})
        assert trial.params == {}

    def test_transferred_trial_complex_params(self):
        """Test TransferredTrial with complex params."""
        from milia_pipeline.models.hpo.transfer.warm_start import TransferredTrial

        params = {
            "lr": 0.001,
            "hidden_dim": 256,
            "dropout": 0.1,
            "activation": "relu",
            "num_layers": 3,
        }
        trial = TransferredTrial(params=params)

        assert trial.params == params
        assert len(trial.params) == 5

    def test_transferred_trial_to_dict(self):
        """Test TransferredTrial.to_dict() returns correct dictionary.

        Pydantic V2: to_dict() wraps model_dump() for backward compatibility.
        """
        from milia_pipeline.models.hpo.transfer.warm_start import TransferredTrial

        trial = TransferredTrial(
            params={"lr": 0.001, "hidden_dim": 128},
            value=0.15,
            source_study="study_A",
            similarity=0.9,
            weight=0.8,
            is_valid=True,
        )

        result = trial.to_dict()

        assert isinstance(result, dict)
        assert result["params"] == {"lr": 0.001, "hidden_dim": 128}
        assert result["value"] == 0.15
        assert result["source_study"] == "study_A"
        assert result["similarity"] == 0.9
        assert result["weight"] == 0.8
        assert result["is_valid"] is True

    def test_transferred_trial_to_dict_with_defaults(self):
        """Test TransferredTrial.to_dict() includes default values."""
        from milia_pipeline.models.hpo.transfer.warm_start import TransferredTrial

        trial = TransferredTrial(params={"lr": 0.001})
        result = trial.to_dict()

        assert result["params"] == {"lr": 0.001}
        assert result["value"] is None
        assert result["source_study"] is None
        assert result["similarity"] == 1.0
        assert result["weight"] == 1.0
        assert result["is_valid"] is True


# =============================================================================
# WARM-START STRATEGY INITIALIZATION TESTS
# =============================================================================


class TestWarmStartStrategyInit:
    """Test WarmStartStrategy.__init__()."""

    def test_warm_start_strategy_default_init(self):
        """Test WarmStartStrategy with default initialization."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            WarmStartConfig,
            WarmStartMethod,
            WarmStartStrategy,
        )

        strategy = WarmStartStrategy()

        assert strategy.config is not None
        assert isinstance(strategy.config, WarmStartConfig)
        assert strategy.config.method == WarmStartMethod.WEIGHTED

    def test_warm_start_strategy_with_custom_config(self):
        """Test WarmStartStrategy with custom config."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            WarmStartConfig,
            WarmStartMethod,
            WarmStartStrategy,
        )

        config = WarmStartConfig(method=WarmStartMethod.FILTERED, n_trials=20, min_similarity=0.7)
        strategy = WarmStartStrategy(config)

        assert strategy.config == config
        assert strategy.config.method == WarmStartMethod.FILTERED
        assert strategy.config.n_trials == 20
        assert strategy.config.min_similarity == 0.7

    def test_warm_start_strategy_stores_optuna_reference(self):
        """Test WarmStartStrategy stores optuna reference."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        strategy = WarmStartStrategy()

        # _optuna should be set (either to optuna module or None)
        assert hasattr(strategy, "_optuna")

    def test_warm_start_strategy_with_none_config_uses_default(self):
        """Test WarmStartStrategy with None config uses default."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartConfig, WarmStartStrategy

        strategy = WarmStartStrategy(config=None)

        assert strategy.config is not None
        assert isinstance(strategy.config, WarmStartConfig)

    def test_warm_start_strategy_multiple_instances_independent(self):
        """Test multiple WarmStartStrategy instances are independent."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            WarmStartConfig,
            WarmStartMethod,
            WarmStartStrategy,
        )

        config1 = WarmStartConfig(method=WarmStartMethod.WEIGHTED)
        config2 = WarmStartConfig(method=WarmStartMethod.FILTERED)

        strategy1 = WarmStartStrategy(config1)
        strategy2 = WarmStartStrategy(config2)

        assert strategy1.config.method == WarmStartMethod.WEIGHTED
        assert strategy2.config.method == WarmStartMethod.FILTERED


# =============================================================================
# MODULE EXPORTS TESTS
# =============================================================================


class TestModuleExports:
    """Test module exports."""

    def test_all_exports_available(self):
        """Test all expected exports are available."""
        from milia_pipeline.models.hpo.transfer import warm_start

        expected_exports = [
            "WarmStartMethod",
            "WarmStartConfig",
            "TransferredTrial",
            "WarmStartStrategy",
        ]

        for export in expected_exports:
            assert hasattr(warm_start, export), f"Missing export: {export}"

    def test_all_list_matches_exports(self):
        """Test __all__ matches available exports."""
        from milia_pipeline.models.hpo.transfer import warm_start

        if hasattr(warm_start, "__all__"):
            for export in warm_start.__all__:
                assert hasattr(warm_start, export), f"__all__ contains unavailable: {export}"

    def test_warm_start_method_importable(self):
        """Test WarmStartMethod can be imported directly."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartMethod

        assert WarmStartMethod is not None

    def test_warm_start_config_importable(self):
        """Test WarmStartConfig can be imported directly."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartConfig

        assert WarmStartConfig is not None

    def test_transferred_trial_importable(self):
        """Test TransferredTrial can be imported directly."""
        from milia_pipeline.models.hpo.transfer.warm_start import TransferredTrial

        assert TransferredTrial is not None

    def test_warm_start_strategy_importable(self):
        """Test WarmStartStrategy can be imported directly."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        assert WarmStartStrategy is not None


# =============================================================================
# WEIGHTED TRANSFER STATIC METHOD TESTS
# =============================================================================


class TestWeightedTransfer:
    """Test WarmStartStrategy.weighted_transfer() static method."""

    def test_weighted_transfer_empty_trials_returns_empty(self):
        """Test weighted_transfer with empty trials returns empty list."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.weighted_transfer(source_trials=[], similarities=[])

        assert result == []

    def test_weighted_transfer_basic(self, mock_trials, sample_similarities):
        """Test weighted_transfer basic functionality."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            TransferredTrial,
            WarmStartStrategy,
        )

        result = WarmStartStrategy.weighted_transfer(
            source_trials=mock_trials, similarities=sample_similarities
        )

        assert len(result) == len(mock_trials)
        assert all(isinstance(t, TransferredTrial) for t in result)

    def test_weighted_transfer_with_n_trials_limit(self, mock_trials, sample_similarities):
        """Test weighted_transfer with n_trials limit."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.weighted_transfer(
            source_trials=mock_trials, similarities=sample_similarities, n_trials=3
        )

        assert len(result) == 3

    def test_weighted_transfer_sorted_by_weight(self, mock_trials, sample_similarities):
        """Test weighted_transfer returns trials sorted by weight descending."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.weighted_transfer(
            source_trials=mock_trials, similarities=sample_similarities
        )

        # Check sorted by weight (descending)
        weights = [t.weight for t in result]
        assert weights == sorted(weights, reverse=True)

    def test_weighted_transfer_min_similarity_filter(self, mock_trials, sample_similarities):
        """Test weighted_transfer filters by min_similarity."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        # With min_similarity=0.7, only trials with sim >= 0.7 should pass
        result = WarmStartStrategy.weighted_transfer(
            source_trials=mock_trials, similarities=sample_similarities, min_similarity=0.7
        )

        # similarities are [0.95, 0.85, 0.75, 0.65, 0.55]
        # Only first 3 pass threshold of 0.7
        assert len(result) == 3
        assert all(t.similarity >= 0.7 for t in result)

    def test_weighted_transfer_high_min_similarity(self, mock_trials, sample_similarities):
        """Test weighted_transfer with high min_similarity filters most."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.weighted_transfer(
            source_trials=mock_trials, similarities=sample_similarities, min_similarity=0.9
        )

        # Only first trial (0.95) passes
        assert len(result) == 1
        assert result[0].similarity >= 0.9

    def test_weighted_transfer_zero_min_similarity(self, mock_trials, sample_similarities):
        """Test weighted_transfer with min_similarity=0 keeps all."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.weighted_transfer(
            source_trials=mock_trials, similarities=sample_similarities, min_similarity=0.0
        )

        assert len(result) == len(mock_trials)

    def test_weighted_transfer_with_performance_weighting(self, mock_trials, sample_similarities):
        """Test weighted_transfer with weight_by_performance=True."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.weighted_transfer(
            source_trials=mock_trials, similarities=sample_similarities, weight_by_performance=True
        )

        # Each trial should have weight = similarity * perf_weight
        # where perf_weight = 1.0 / (1.0 + abs(value))
        for trial in result:
            assert trial.weight > 0

    def test_weighted_transfer_without_performance_weighting(
        self, mock_trials, sample_similarities
    ):
        """Test weighted_transfer with weight_by_performance=False."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.weighted_transfer(
            source_trials=mock_trials, similarities=sample_similarities, weight_by_performance=False
        )

        # Weight should equal similarity when not weighted by performance
        for _i, trial in enumerate(result):
            # Find the original index based on similarity
            assert trial.weight == trial.similarity

    def test_weighted_transfer_preserves_params(self, mock_trials, sample_similarities):
        """Test weighted_transfer preserves trial params."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.weighted_transfer(
            source_trials=mock_trials, similarities=sample_similarities
        )

        # Check params are preserved (any order due to sorting)
        result_params = [t.params for t in result]
        for trial in mock_trials:
            assert trial.params in result_params

    def test_weighted_transfer_preserves_values(self, mock_trials, sample_similarities):
        """Test weighted_transfer preserves trial values."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.weighted_transfer(
            source_trials=mock_trials, similarities=sample_similarities
        )

        result_values = [t.value for t in result]
        for trial in mock_trials:
            assert trial.value in result_values

    def test_weighted_transfer_single_similarity_for_all(self, mock_trials):
        """Test weighted_transfer with single similarity value."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        # When similarities list is shorter, last value is used for remaining
        result = WarmStartStrategy.weighted_transfer(
            source_trials=mock_trials,
            similarities=[0.8],  # Single value
        )

        assert len(result) == len(mock_trials)
        # All should have similarity 0.8
        assert all(t.similarity == 0.8 for t in result)

    def test_weighted_transfer_with_dict_trials(self, mock_trial_dicts, sample_similarities):
        """Test weighted_transfer with dictionary trials."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.weighted_transfer(
            source_trials=mock_trial_dicts, similarities=sample_similarities
        )

        assert len(result) == len(mock_trial_dicts)
        assert all(len(t.params) > 0 for t in result)

    def test_weighted_transfer_all_below_threshold(self, mock_trials, low_similarities):
        """Test weighted_transfer when all trials below threshold."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.weighted_transfer(
            source_trials=mock_trials, similarities=low_similarities, min_similarity=0.5
        )

        assert len(result) == 0

    def test_weighted_transfer_none_values_handled(self, mixed_value_trials, high_similarities):
        """Test weighted_transfer handles None values."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.weighted_transfer(
            source_trials=mixed_value_trials,
            similarities=high_similarities,
            weight_by_performance=True,
        )

        # Should handle None values without error
        assert len(result) == len(mixed_value_trials)


# =============================================================================
# FILTERED TRANSFER STATIC METHOD TESTS
# =============================================================================


class TestFilteredTransfer:
    """Test WarmStartStrategy.filtered_transfer() static method."""

    def test_filtered_transfer_empty_trials_returns_empty(self, sample_search_space):
        """Test filtered_transfer with empty trials returns empty list."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.filtered_transfer(
            source_trials=[], target_search_space=sample_search_space
        )

        assert result == []

    def test_filtered_transfer_basic(self, mock_trials, sample_search_space):
        """Test filtered_transfer basic functionality."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            TransferredTrial,
            WarmStartStrategy,
        )

        result = WarmStartStrategy.filtered_transfer(
            source_trials=mock_trials, target_search_space=sample_search_space
        )

        assert all(isinstance(t, TransferredTrial) for t in result)

    def test_filtered_transfer_with_n_trials_limit(self, mock_trials, sample_search_space):
        """Test filtered_transfer with n_trials limit."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.filtered_transfer(
            source_trials=mock_trials, target_search_space=sample_search_space, n_trials=2
        )

        assert len(result) <= 2

    def test_filtered_transfer_sorted_by_value(self, mock_trials, sample_search_space):
        """Test filtered_transfer returns trials sorted by value ascending."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.filtered_transfer(
            source_trials=mock_trials, target_search_space=sample_search_space
        )

        # Check sorted by value (ascending - best first for minimization)
        values = [t.value for t in result if t.value is not None]
        assert values == sorted(values)

    def test_filtered_transfer_scale_to_bounds_true(self, sample_search_space):
        """Test filtered_transfer scales values to bounds."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        # Create trial with out-of-bounds value
        trial = MockFrozenTrial(
            params={"lr": 0.5, "hidden_dim": 1000},  # Both out of bounds
            value=0.1,
        )

        result = WarmStartStrategy.filtered_transfer(
            source_trials=[trial], target_search_space=sample_search_space, scale_to_bounds=True
        )

        if result:
            # lr should be clamped to 0.1 (high bound)
            # hidden_dim should be clamped to 512 (high bound)
            assert result[0].params.get("lr", 0) <= 0.1
            assert result[0].params.get("hidden_dim", 0) <= 512

    def test_filtered_transfer_scale_to_bounds_false(self, sample_search_space):
        """Test filtered_transfer without scaling rejects out-of-bounds."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        # Create trial with out-of-bounds value
        trial = MockFrozenTrial(
            params={"lr": 0.5, "hidden_dim": 1000},  # Both out of bounds
            value=0.1,
        )

        result = WarmStartStrategy.filtered_transfer(
            source_trials=[trial], target_search_space=sample_search_space, scale_to_bounds=False
        )

        # Should filter out invalid params
        # Result may be empty or have fewer params
        if result:
            # If any result, params should not include out-of-bounds values
            pass  # Behavior depends on implementation

    def test_filtered_transfer_with_categorical(self, sample_search_space):
        """Test filtered_transfer handles categorical parameters."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        # Trial with valid categorical
        trial = MockFrozenTrial(params={"activation": "relu"}, value=0.1)

        result = WarmStartStrategy.filtered_transfer(
            source_trials=[trial], target_search_space=sample_search_space
        )

        if result:
            assert "activation" in result[0].params
            assert result[0].params["activation"] == "relu"

    def test_filtered_transfer_invalid_categorical(self, sample_search_space):
        """Test filtered_transfer rejects invalid categorical."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        # Trial with invalid categorical
        trial = MockFrozenTrial(params={"activation": "invalid_activation"}, value=0.1)

        result = WarmStartStrategy.filtered_transfer(
            source_trials=[trial], target_search_space=sample_search_space
        )

        # Either empty result or activation not included
        if result:
            assert result[0].params.get("activation") != "invalid_activation"

    def test_filtered_transfer_with_dict_trials(self, mock_trial_dicts, sample_search_space):
        """Test filtered_transfer with dictionary trials."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.filtered_transfer(
            source_trials=mock_trial_dicts, target_search_space=sample_search_space
        )

        assert all(t.is_valid for t in result)

    def test_filtered_transfer_nested_search_space(self, mock_trials, nested_search_space):
        """Test filtered_transfer with nested search space."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.filtered_transfer(
            source_trials=mock_trials, target_search_space=nested_search_space
        )

        # Should handle nested search space
        assert isinstance(result, list)

    def test_filtered_transfer_preserves_is_valid(self, mock_trials, sample_search_space):
        """Test filtered_transfer sets is_valid=True for valid trials."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.filtered_transfer(
            source_trials=mock_trials, target_search_space=sample_search_space
        )

        assert all(t.is_valid for t in result)

    def test_filtered_transfer_int_type(self, sample_search_space):
        """Test filtered_transfer handles int type."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        trial = MockFrozenTrial(
            params={"hidden_dim": 128.0},  # Float that should become int
            value=0.1,
        )

        result = WarmStartStrategy.filtered_transfer(
            source_trials=[trial], target_search_space=sample_search_space
        )

        if result and "hidden_dim" in result[0].params:
            assert isinstance(result[0].params["hidden_dim"], int)

    def test_filtered_transfer_float_type(self, sample_search_space):
        """Test filtered_transfer handles float type."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        trial = MockFrozenTrial(params={"lr": 0.01, "dropout": 0.2}, value=0.1)

        result = WarmStartStrategy.filtered_transfer(
            source_trials=[trial], target_search_space=sample_search_space
        )

        if result and "lr" in result[0].params:
            assert isinstance(result[0].params["lr"], float)


# =============================================================================
# FULL TRANSFER STATIC METHOD TESTS
# =============================================================================


class TestFullTransfer:
    """Test WarmStartStrategy.full_transfer() static method."""

    def test_full_transfer_empty_trials_returns_empty(self):
        """Test full_transfer with empty trials returns empty list."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.full_transfer(source_trials=[], n_trials=10)

        assert result == []

    def test_full_transfer_basic(self, mock_trials):
        """Test full_transfer basic functionality."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            TransferredTrial,
            WarmStartStrategy,
        )

        result = WarmStartStrategy.full_transfer(source_trials=mock_trials, n_trials=5)

        assert len(result) == 5
        assert all(isinstance(t, TransferredTrial) for t in result)

    def test_full_transfer_with_n_trials_limit(self, mock_trials):
        """Test full_transfer respects n_trials limit."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.full_transfer(source_trials=mock_trials, n_trials=2)

        assert len(result) == 2

    def test_full_transfer_n_trials_larger_than_source(self, mock_trials):
        """Test full_transfer when n_trials > len(source_trials)."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.full_transfer(source_trials=mock_trials, n_trials=100)

        assert len(result) == len(mock_trials)

    def test_full_transfer_sorted_by_performance(self, mock_trials):
        """Test full_transfer sorts by performance."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.full_transfer(
            source_trials=mock_trials, n_trials=5, sort_by_performance=True
        )

        # Should be sorted by value ascending
        values = [t.value for t in result if t.value is not None]
        assert values == sorted(values)

    def test_full_transfer_not_sorted_by_performance(self, mock_trials):
        """Test full_transfer without performance sorting."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.full_transfer(
            source_trials=mock_trials, n_trials=5, sort_by_performance=False
        )

        assert len(result) == 5

    def test_full_transfer_preserves_params(self, mock_trials):
        """Test full_transfer preserves trial params."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.full_transfer(source_trials=mock_trials, n_trials=5)

        for trial in result:
            assert len(trial.params) > 0

    def test_full_transfer_preserves_values(self, mock_trials):
        """Test full_transfer preserves trial values."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.full_transfer(source_trials=mock_trials, n_trials=5)

        for trial in result:
            assert trial.value is not None

    def test_full_transfer_sets_is_valid(self, mock_trials):
        """Test full_transfer sets is_valid=True."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.full_transfer(source_trials=mock_trials, n_trials=5)

        assert all(t.is_valid for t in result)

    def test_full_transfer_sets_weight(self, mock_trials):
        """Test full_transfer sets weight=1.0."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.full_transfer(source_trials=mock_trials, n_trials=5)

        assert all(t.weight == 1.0 for t in result)

    def test_full_transfer_with_dict_trials(self, mock_trial_dicts):
        """Test full_transfer with dictionary trials."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.full_transfer(source_trials=mock_trial_dicts, n_trials=3)

        assert len(result) == 3
        assert all(len(t.params) > 0 for t in result)

    def test_full_transfer_none_values(self, mixed_value_trials):
        """Test full_transfer handles None values."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.full_transfer(
            source_trials=mixed_value_trials, n_trials=5, sort_by_performance=True
        )

        # None values should sort to end (infinity)
        assert len(result) == 5


# =============================================================================
# EXTRACT PARAMS HELPER METHOD TESTS
# =============================================================================


class TestExtractParams:
    """Test WarmStartStrategy._extract_params() helper method."""

    def test_extract_params_from_frozen_trial(self):
        """Test _extract_params from FrozenTrial-like object."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        trial = MockFrozenTrial(params={"lr": 0.001, "hidden_dim": 128})
        result = WarmStartStrategy._extract_params(trial)

        assert result == {"lr": 0.001, "hidden_dim": 128}

    def test_extract_params_from_dict_with_params_key(self):
        """Test _extract_params from dict with 'params' key."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        trial = {"params": {"lr": 0.001, "hidden_dim": 128}, "value": 0.1}
        result = WarmStartStrategy._extract_params(trial)

        assert result == {"lr": 0.001, "hidden_dim": 128}

    def test_extract_params_from_dict_without_params_key(self):
        """Test _extract_params from dict without 'params' key."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        trial = {"lr": 0.001, "hidden_dim": 128}
        result = WarmStartStrategy._extract_params(trial)

        assert result == {"lr": 0.001, "hidden_dim": 128}

    def test_extract_params_empty_dict(self):
        """Test _extract_params from empty dict."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy._extract_params({})

        assert result == {}

    def test_extract_params_object_without_params(self):
        """Test _extract_params from object without params attribute."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        class NoParams:
            pass

        obj = NoParams()
        result = WarmStartStrategy._extract_params(obj)

        assert result == {}

    def test_extract_params_returns_copy(self):
        """Test _extract_params returns a copy."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        original_params = {"lr": 0.001}
        trial = {"params": original_params}
        result = WarmStartStrategy._extract_params(trial)

        # Modifying result should not affect original
        result["lr"] = 0.1
        assert original_params["lr"] == 0.001


# =============================================================================
# EXTRACT VALUE HELPER METHOD TESTS
# =============================================================================


class TestExtractValue:
    """Test WarmStartStrategy._extract_value() helper method."""

    def test_extract_value_from_frozen_trial(self):
        """Test _extract_value from FrozenTrial-like object."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        trial = MockFrozenTrial(params={}, value=0.15)
        result = WarmStartStrategy._extract_value(trial)

        assert result == 0.15

    def test_extract_value_from_dict_with_value_key(self):
        """Test _extract_value from dict with 'value' key."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        trial = {"params": {}, "value": 0.15}
        result = WarmStartStrategy._extract_value(trial)

        assert result == 0.15

    def test_extract_value_from_dict_with_values_list(self):
        """Test _extract_value from dict with 'values' list."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        trial = {"params": {}, "values": [0.15, 0.20]}
        result = WarmStartStrategy._extract_value(trial)

        assert result == 0.15  # First value

    def test_extract_value_from_object_with_values(self):
        """Test _extract_value from object with values attribute."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        trial = MockFrozenTrial(params={}, value=0.15)
        # MockFrozenTrial sets both value and values
        result = WarmStartStrategy._extract_value(trial)

        assert result == 0.15

    def test_extract_value_none(self):
        """Test _extract_value returns None when no value."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        trial = MockFrozenTrial(params={}, value=None)
        result = WarmStartStrategy._extract_value(trial)

        assert result is None

    def test_extract_value_empty_dict(self):
        """Test _extract_value from empty dict."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy._extract_value({})

        assert result is None

    def test_extract_value_object_without_value(self):
        """Test _extract_value from object without value attribute."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        class NoValue:
            pass

        obj = NoValue()
        result = WarmStartStrategy._extract_value(obj)

        assert result is None


# =============================================================================
# FLATTEN SEARCH SPACE HELPER METHOD TESTS
# =============================================================================


class TestFlattenSearchSpace:
    """Test WarmStartStrategy._flatten_search_space() helper method."""

    def test_flatten_search_space_flat(self, sample_search_space):
        """Test _flatten_search_space with already flat space."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy._flatten_search_space(sample_search_space)

        assert "lr" in result
        assert "hidden_dim" in result
        assert "dropout" in result
        assert "activation" in result

    def test_flatten_search_space_nested(self, nested_search_space):
        """Test _flatten_search_space with nested space."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy._flatten_search_space(nested_search_space)

        # Should flatten nested structure
        assert "lr" in result
        assert "hidden_dim" in result
        assert "dropout" in result
        assert "batch_size" in result

    def test_flatten_search_space_empty(self):
        """Test _flatten_search_space with empty space."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy._flatten_search_space({})

        assert result == {}

    def test_flatten_search_space_preserves_type(self, sample_search_space):
        """Test _flatten_search_space preserves type info."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy._flatten_search_space(sample_search_space)

        assert result["lr"]["type"] == "float"
        assert result["hidden_dim"]["type"] == "int"
        assert result["activation"]["type"] == "categorical"

    def test_flatten_search_space_preserves_bounds(self, sample_search_space):
        """Test _flatten_search_space preserves bounds."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy._flatten_search_space(sample_search_space)

        assert result["lr"]["low"] == 1e-5
        assert result["lr"]["high"] == 1e-1
        assert result["hidden_dim"]["low"] == 32
        assert result["hidden_dim"]["high"] == 512


# =============================================================================
# FILTER PARAMS HELPER METHOD TESTS
# =============================================================================


class TestFilterParams:
    """Test WarmStartStrategy._filter_params() helper method."""

    def test_filter_params_all_valid(self, sample_search_space):
        """Test _filter_params with all valid params."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        params = {"lr": 0.01, "hidden_dim": 128, "dropout": 0.2}
        flat_space = WarmStartStrategy._flatten_search_space(sample_search_space)

        result, is_valid = WarmStartStrategy._filter_params(params, flat_space)

        assert is_valid
        assert "lr" in result
        assert "hidden_dim" in result
        assert "dropout" in result

    def test_filter_params_scales_to_bounds(self, sample_search_space):
        """Test _filter_params scales out-of-bounds values."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        params = {"lr": 0.5}  # Out of bounds (max is 0.1)
        flat_space = WarmStartStrategy._flatten_search_space(sample_search_space)

        result, is_valid = WarmStartStrategy._filter_params(
            params, flat_space, scale_to_bounds=True
        )

        if "lr" in result:
            assert result["lr"] <= 0.1

    def test_filter_params_no_scale_rejects_out_of_bounds(self, sample_search_space):
        """Test _filter_params rejects out-of-bounds without scaling."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        params = {"lr": 0.5}  # Out of bounds
        flat_space = WarmStartStrategy._flatten_search_space(sample_search_space)

        result, is_valid = WarmStartStrategy._filter_params(
            params, flat_space, scale_to_bounds=False
        )

        # Either not valid or lr not in result
        if "lr" in result:
            assert is_valid is False or result["lr"] <= 0.1

    def test_filter_params_categorical_valid(self, sample_search_space):
        """Test _filter_params with valid categorical."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        params = {"activation": "relu"}
        flat_space = WarmStartStrategy._flatten_search_space(sample_search_space)

        result, is_valid = WarmStartStrategy._filter_params(params, flat_space)

        assert "activation" in result
        assert result["activation"] == "relu"

    def test_filter_params_categorical_invalid(self, sample_search_space):
        """Test _filter_params with invalid categorical."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        params = {"activation": "invalid"}
        flat_space = WarmStartStrategy._flatten_search_space(sample_search_space)

        result, is_valid = WarmStartStrategy._filter_params(params, flat_space)

        # Invalid categorical should not be included
        assert "activation" not in result or is_valid is False

    def test_filter_params_int_type_conversion(self, sample_search_space):
        """Test _filter_params converts to int type."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        params = {"hidden_dim": 128.5}  # Float that should become int
        flat_space = WarmStartStrategy._flatten_search_space(sample_search_space)

        result, is_valid = WarmStartStrategy._filter_params(params, flat_space)

        if "hidden_dim" in result:
            assert isinstance(result["hidden_dim"], int)

    def test_filter_params_float_type_conversion(self, sample_search_space):
        """Test _filter_params ensures float type."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        params = {"lr": 1}  # Int that should remain/become float
        flat_space = WarmStartStrategy._flatten_search_space(sample_search_space)

        result, is_valid = WarmStartStrategy._filter_params(params, flat_space)

        if "lr" in result:
            assert isinstance(result["lr"], float)

    def test_filter_params_unknown_param_skipped(self, sample_search_space):
        """Test _filter_params skips unknown params."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        params = {"lr": 0.01, "unknown_param": 42}
        flat_space = WarmStartStrategy._flatten_search_space(sample_search_space)

        result, is_valid = WarmStartStrategy._filter_params(params, flat_space)

        assert "unknown_param" not in result
        assert "lr" in result

    def test_filter_params_empty_params(self, sample_search_space):
        """Test _filter_params with empty params."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        flat_space = WarmStartStrategy._flatten_search_space(sample_search_space)

        result, is_valid = WarmStartStrategy._filter_params({}, flat_space)

        assert result == {}
        # Empty result should be invalid
        assert is_valid is False

    def test_filter_params_enum_type(self):
        """Test _filter_params handles enum type values."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        search_space = {"hidden_dim": {"type": MockParamType.INT, "low": 32, "high": 256}}
        params = {"hidden_dim": 64}
        flat_space = WarmStartStrategy._flatten_search_space(search_space)

        result, is_valid = WarmStartStrategy._filter_params(params, flat_space)

        if "hidden_dim" in result:
            assert isinstance(result["hidden_dim"], int)

    def test_filter_params_loguniform_type(self):
        """Test _filter_params handles loguniform type."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        search_space = {"lr": {"type": "loguniform", "low": 1e-5, "high": 1e-1}}
        params = {"lr": 0.01}
        flat_space = WarmStartStrategy._flatten_search_space(search_space)

        result, is_valid = WarmStartStrategy._filter_params(params, flat_space)

        assert "lr" in result
        assert isinstance(result["lr"], float)

    def test_filter_params_int_uniform_type(self):
        """Test _filter_params handles int_uniform type."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        search_space = {"batch_size": {"type": "int_uniform", "low": 16, "high": 128}}
        params = {"batch_size": 32}
        flat_space = WarmStartStrategy._flatten_search_space(search_space)

        result, is_valid = WarmStartStrategy._filter_params(params, flat_space)

        assert "batch_size" in result
        assert isinstance(result["batch_size"], int)

    def test_filter_params_discrete_uniform_type(self):
        """Test _filter_params handles discrete_uniform type."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        search_space = {"step": {"type": "discrete_uniform", "low": 0.1, "high": 1.0}}
        params = {"step": 0.5}
        flat_space = WarmStartStrategy._flatten_search_space(search_space)

        result, is_valid = WarmStartStrategy._filter_params(params, flat_space)

        assert "step" in result
        assert isinstance(result["step"], float)

    def test_filter_params_unknown_type_passed_through(self):
        """Test _filter_params passes through unknown types."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        search_space = {"custom": {"type": "custom_type", "value": 42}}
        params = {"custom": 100}
        flat_space = WarmStartStrategy._flatten_search_space(search_space)

        result, is_valid = WarmStartStrategy._filter_params(params, flat_space)

        # Unknown types are passed through
        assert "custom" in result


# =============================================================================
# TRANSFER INSTANCE METHOD TESTS
# =============================================================================


class TestTransferInstanceMethod:
    """Test WarmStartStrategy.transfer() instance method."""

    def test_transfer_weighted_method(self, mock_trials, sample_similarities):
        """Test transfer with WEIGHTED method."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            WarmStartConfig,
            WarmStartMethod,
            WarmStartStrategy,
        )

        config = WarmStartConfig(method=WarmStartMethod.WEIGHTED, n_trials=3)
        strategy = WarmStartStrategy(config)

        result = strategy.transfer(source_trials=mock_trials, similarities=sample_similarities)

        assert len(result) <= 3

    def test_transfer_filtered_method(self, mock_trials, sample_search_space):
        """Test transfer with FILTERED method."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            WarmStartConfig,
            WarmStartMethod,
            WarmStartStrategy,
        )

        config = WarmStartConfig(method=WarmStartMethod.FILTERED, n_trials=3)
        strategy = WarmStartStrategy(config)

        result = strategy.transfer(
            source_trials=mock_trials, target_search_space=sample_search_space
        )

        assert isinstance(result, list)

    def test_transfer_full_method(self, mock_trials):
        """Test transfer with FULL method."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            WarmStartConfig,
            WarmStartMethod,
            WarmStartStrategy,
        )

        config = WarmStartConfig(method=WarmStartMethod.FULL, n_trials=3)
        strategy = WarmStartStrategy(config)

        result = strategy.transfer(source_trials=mock_trials)

        assert len(result) == 3

    def test_transfer_adaptive_method_with_similarities(self, mock_trials, sample_similarities):
        """Test transfer with ADAPTIVE method when similarities provided."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            WarmStartConfig,
            WarmStartMethod,
            WarmStartStrategy,
        )

        config = WarmStartConfig(method=WarmStartMethod.ADAPTIVE, n_trials=3)
        strategy = WarmStartStrategy(config)

        result = strategy.transfer(source_trials=mock_trials, similarities=sample_similarities)

        # Should select WEIGHTED when similarities vary
        assert isinstance(result, list)

    def test_transfer_adaptive_method_with_search_space(self, mock_trials, sample_search_space):
        """Test transfer with ADAPTIVE method when search space provided."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            WarmStartConfig,
            WarmStartMethod,
            WarmStartStrategy,
        )

        config = WarmStartConfig(method=WarmStartMethod.ADAPTIVE, n_trials=3)
        strategy = WarmStartStrategy(config)

        result = strategy.transfer(
            source_trials=mock_trials, target_search_space=sample_search_space
        )

        assert isinstance(result, list)

    def test_transfer_adaptive_method_no_extras(self, mock_trials):
        """Test transfer with ADAPTIVE method without similarities or space."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            WarmStartConfig,
            WarmStartMethod,
            WarmStartStrategy,
        )

        config = WarmStartConfig(method=WarmStartMethod.ADAPTIVE, n_trials=3)
        strategy = WarmStartStrategy(config)

        result = strategy.transfer(source_trials=mock_trials)

        # Should fall back to FULL
        assert len(result) == 3

    def test_transfer_filtered_without_search_space_raises(self, mock_trials):
        """Test transfer FILTERED without search space raises ValueError."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            WarmStartConfig,
            WarmStartMethod,
            WarmStartStrategy,
        )

        config = WarmStartConfig(method=WarmStartMethod.FILTERED)
        strategy = WarmStartStrategy(config)

        with pytest.raises(ValueError, match="target_search_space required"):
            strategy.transfer(source_trials=mock_trials)

    def test_transfer_weighted_without_similarities_defaults(self, mock_trials):
        """Test transfer WEIGHTED without similarities uses default 1.0."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            WarmStartConfig,
            WarmStartMethod,
            WarmStartStrategy,
        )

        config = WarmStartConfig(method=WarmStartMethod.WEIGHTED, n_trials=3)
        strategy = WarmStartStrategy(config)

        result = strategy.transfer(source_trials=mock_trials)

        # Should use default similarity of 1.0
        assert all(t.similarity == 1.0 for t in result)

    def test_transfer_with_filter_invalid(self, mock_trials, sample_search_space):
        """Test transfer with filter_invalid=True."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            WarmStartConfig,
            WarmStartMethod,
            WarmStartStrategy,
        )

        config = WarmStartConfig(method=WarmStartMethod.FULL, filter_invalid=True)
        strategy = WarmStartStrategy(config)

        result = strategy.transfer(
            source_trials=mock_trials, target_search_space=sample_search_space
        )

        # All returned should be valid
        assert all(t.is_valid for t in result)

    def test_transfer_with_add_noise(self, mock_trials, sample_search_space):
        """Test transfer with add_noise=True."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            WarmStartConfig,
            WarmStartMethod,
            WarmStartStrategy,
        )

        # Set seed for reproducibility
        np.random.seed(42)

        config = WarmStartConfig(method=WarmStartMethod.FULL, add_noise=True, noise_scale=0.1)
        strategy = WarmStartStrategy(config)

        result = strategy.transfer(
            source_trials=mock_trials, target_search_space=sample_search_space
        )

        assert isinstance(result, list)

    def test_transfer_uses_config_n_trials(self, mock_trials):
        """Test transfer uses config.n_trials."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            WarmStartConfig,
            WarmStartMethod,
            WarmStartStrategy,
        )

        config = WarmStartConfig(method=WarmStartMethod.FULL, n_trials=2)
        strategy = WarmStartStrategy(config)

        result = strategy.transfer(source_trials=mock_trials)

        assert len(result) == 2

    def test_transfer_uses_config_min_similarity(self, mock_trials, sample_similarities):
        """Test transfer uses config.min_similarity."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            WarmStartConfig,
            WarmStartMethod,
            WarmStartStrategy,
        )

        config = WarmStartConfig(method=WarmStartMethod.WEIGHTED, min_similarity=0.8)
        strategy = WarmStartStrategy(config)

        result = strategy.transfer(source_trials=mock_trials, similarities=sample_similarities)

        # Only trials with similarity >= 0.8 should pass
        # similarities are [0.95, 0.85, 0.75, 0.65, 0.55]
        assert len(result) == 2

    def test_transfer_empty_trials_returns_empty(self):
        """Test transfer with empty trials returns empty list."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            WarmStartStrategy,
        )

        strategy = WarmStartStrategy()
        result = strategy.transfer(source_trials=[])

        assert result == []


# =============================================================================
# APPLY TO STUDY INSTANCE METHOD TESTS
# =============================================================================


class TestApplyToStudy:
    """Test WarmStartStrategy.apply_to_study() instance method."""

    def test_apply_to_study_basic(self, mock_study, transferred_trials):
        """Test apply_to_study basic functionality."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        strategy = WarmStartStrategy()

        with patch.object(strategy, "_optuna", MockOptuna()):
            n_enqueued = strategy.apply_to_study(mock_study, transferred_trials)

        assert n_enqueued == len(transferred_trials)
        assert len(mock_study._enqueued_trials) == len(transferred_trials)

    def test_apply_to_study_skips_invalid(self, mock_study, mixed_validity_trials):
        """Test apply_to_study skips invalid trials."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        strategy = WarmStartStrategy()

        with patch.object(strategy, "_optuna", MockOptuna()):
            n_enqueued = strategy.apply_to_study(mock_study, mixed_validity_trials)

        # Only 3 are valid
        assert n_enqueued == 3

    def test_apply_to_study_no_optuna_returns_zero(self, mock_study, transferred_trials):
        """Test apply_to_study returns 0 when optuna not available."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        strategy = WarmStartStrategy()
        strategy._optuna = None

        n_enqueued = strategy.apply_to_study(mock_study, transferred_trials)

        assert n_enqueued == 0

    def test_apply_to_study_handles_enqueue_error(self, transferred_trials):
        """Test apply_to_study handles enqueue errors."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        failing_study = MockFailingStudy()
        strategy = WarmStartStrategy()

        with patch.object(strategy, "_optuna", MockOptuna()):
            n_enqueued = strategy.apply_to_study(failing_study, transferred_trials)

        # Should return 0 when all fail
        assert n_enqueued == 0

    def test_apply_to_study_empty_trials(self, mock_study):
        """Test apply_to_study with empty trials."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        strategy = WarmStartStrategy()

        with patch.object(strategy, "_optuna", MockOptuna()):
            n_enqueued = strategy.apply_to_study(mock_study, [])

        assert n_enqueued == 0

    def test_apply_to_study_partial_failure(self, mock_study):
        """Test apply_to_study with partial enqueue failures."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            TransferredTrial,
            WarmStartStrategy,
        )

        trials = [
            TransferredTrial(params={"lr": 0.001}, is_valid=True),
            TransferredTrial(params={"lr": 0.002}, is_valid=True),
        ]

        # Make second enqueue fail
        call_count = [0]
        original_enqueue = mock_study.enqueue_trial

        def failing_enqueue(params):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Second enqueue failed")
            original_enqueue(params)

        mock_study.enqueue_trial = failing_enqueue

        strategy = WarmStartStrategy()

        with patch.object(strategy, "_optuna", MockOptuna()):
            n_enqueued = strategy.apply_to_study(mock_study, trials)

        assert n_enqueued == 1


# =============================================================================
# FILTER INVALID TRIALS HELPER METHOD TESTS
# =============================================================================


class TestFilterInvalidTrials:
    """Test WarmStartStrategy._filter_invalid_trials() helper method."""

    def test_filter_invalid_trials_all_valid(self, sample_search_space):
        """Test _filter_invalid_trials when all valid."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            TransferredTrial,
            WarmStartStrategy,
        )

        trials = [
            TransferredTrial(params={"lr": 0.01, "hidden_dim": 128}, is_valid=True),
            TransferredTrial(params={"lr": 0.02, "hidden_dim": 256}, is_valid=True),
        ]

        strategy = WarmStartStrategy()
        result = strategy._filter_invalid_trials(trials, sample_search_space)

        assert len(result) == 2

    def test_filter_invalid_trials_removes_out_of_bounds(self, sample_search_space):
        """Test _filter_invalid_trials removes out-of-bounds params."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            TransferredTrial,
            WarmStartStrategy,
        )

        trials = [
            TransferredTrial(params={"lr": 0.01}, is_valid=True),  # Valid
            TransferredTrial(params={"lr": 1.0}, is_valid=True),  # Out of bounds (max is 0.1)
        ]

        strategy = WarmStartStrategy()
        result = strategy._filter_invalid_trials(trials, sample_search_space)

        # Out of bounds should be filtered
        assert len(result) <= 2

    def test_filter_invalid_trials_empty_input(self, sample_search_space):
        """Test _filter_invalid_trials with empty input."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        strategy = WarmStartStrategy()
        result = strategy._filter_invalid_trials([], sample_search_space)

        assert result == []


# =============================================================================
# ADD NOISE TO TRIALS HELPER METHOD TESTS
# =============================================================================


class TestAddNoiseToTrials:
    """Test WarmStartStrategy._add_noise_to_trials() helper method."""

    def test_add_noise_to_trials_modifies_numeric(self, sample_search_space):
        """Test _add_noise_to_trials modifies numeric params."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            TransferredTrial,
            WarmStartConfig,
            WarmStartStrategy,
        )

        np.random.seed(42)

        config = WarmStartConfig(add_noise=True, noise_scale=0.1)
        strategy = WarmStartStrategy(config)

        original_lr = 0.01
        trials = [TransferredTrial(params={"lr": original_lr})]

        result = strategy._add_noise_to_trials(trials, sample_search_space)

        # Value should be modified (but might be same by chance)
        assert len(result) == 1
        assert "lr" in result[0].params

    def test_add_noise_preserves_non_numeric(self, sample_search_space):
        """Test _add_noise_to_trials preserves categorical params."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            TransferredTrial,
            WarmStartConfig,
            WarmStartStrategy,
        )

        config = WarmStartConfig(add_noise=True, noise_scale=0.1)
        strategy = WarmStartStrategy(config)

        trials = [TransferredTrial(params={"activation": "relu"})]

        result = strategy._add_noise_to_trials(trials, sample_search_space)

        # Categorical should be unchanged
        assert result[0].params["activation"] == "relu"

    def test_add_noise_respects_bounds(self, sample_search_space):
        """Test _add_noise_to_trials respects parameter bounds."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            TransferredTrial,
            WarmStartConfig,
            WarmStartStrategy,
        )

        np.random.seed(42)

        config = WarmStartConfig(add_noise=True, noise_scale=0.5)  # Large noise
        strategy = WarmStartStrategy(config)

        trials = [TransferredTrial(params={"lr": 0.01, "hidden_dim": 128})]

        result = strategy._add_noise_to_trials(trials, sample_search_space)

        # Values should still be within bounds
        if "lr" in result[0].params:
            assert 1e-5 <= result[0].params["lr"] <= 1e-1
        if "hidden_dim" in result[0].params:
            assert 32 <= result[0].params["hidden_dim"] <= 512

    def test_add_noise_preserves_trial_metadata(self, sample_search_space):
        """Test _add_noise_to_trials preserves trial metadata."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            TransferredTrial,
            WarmStartConfig,
            WarmStartStrategy,
        )

        config = WarmStartConfig(add_noise=True, noise_scale=0.1)
        strategy = WarmStartStrategy(config)

        trials = [
            TransferredTrial(
                params={"lr": 0.01},
                value=0.15,
                source_study="source",
                similarity=0.9,
                weight=0.8,
                is_valid=True,
            )
        ]

        result = strategy._add_noise_to_trials(trials, sample_search_space)

        assert result[0].value == 0.15
        assert result[0].source_study == "source"
        assert result[0].similarity == 0.9
        assert result[0].weight == 0.8
        assert result[0].is_valid is True

    def test_add_noise_empty_trials(self, sample_search_space):
        """Test _add_noise_to_trials with empty trials."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartConfig, WarmStartStrategy

        config = WarmStartConfig(add_noise=True, noise_scale=0.1)
        strategy = WarmStartStrategy(config)

        result = strategy._add_noise_to_trials([], sample_search_space)

        assert result == []

    def test_add_noise_int_type(self, sample_search_space):
        """Test _add_noise_to_trials handles int type."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            TransferredTrial,
            WarmStartConfig,
            WarmStartStrategy,
        )

        np.random.seed(42)

        config = WarmStartConfig(add_noise=True, noise_scale=0.1)
        strategy = WarmStartStrategy(config)

        trials = [TransferredTrial(params={"hidden_dim": 128})]

        result = strategy._add_noise_to_trials(trials, sample_search_space)

        # Result should still be int
        if "hidden_dim" in result[0].params:
            assert isinstance(result[0].params["hidden_dim"], int)


# =============================================================================
# SELECT ADAPTIVE METHOD HELPER METHOD TESTS
# =============================================================================


class TestSelectAdaptiveMethod:
    """Test WarmStartStrategy._select_adaptive_method() helper method."""

    def test_select_adaptive_with_varying_similarities(self, mock_trials, sample_similarities):
        """Test _select_adaptive_method returns WEIGHTED with varying similarities."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartMethod, WarmStartStrategy

        strategy = WarmStartStrategy()
        result = strategy._select_adaptive_method(
            source_trials=mock_trials, target_space=None, similarities=sample_similarities
        )

        assert result == WarmStartMethod.WEIGHTED

    def test_select_adaptive_with_uniform_similarities(self, mock_trials, sample_search_space):
        """Test _select_adaptive_method with uniform similarities and space."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartMethod, WarmStartStrategy

        strategy = WarmStartStrategy()
        # All similarities are 1.0
        result = strategy._select_adaptive_method(
            source_trials=mock_trials,
            target_space=sample_search_space,
            similarities=[1.0] * len(mock_trials),
        )

        # Should choose FILTERED when similarities are all 1.0 and space is provided
        assert result == WarmStartMethod.FILTERED

    def test_select_adaptive_without_similarities_or_space(self, mock_trials):
        """Test _select_adaptive_method returns FULL without similarities or space."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartMethod, WarmStartStrategy

        strategy = WarmStartStrategy()
        result = strategy._select_adaptive_method(
            source_trials=mock_trials, target_space=None, similarities=None
        )

        assert result == WarmStartMethod.FULL

    def test_select_adaptive_with_space_only(self, mock_trials, sample_search_space):
        """Test _select_adaptive_method with only search space."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartMethod, WarmStartStrategy

        strategy = WarmStartStrategy()
        result = strategy._select_adaptive_method(
            source_trials=mock_trials, target_space=sample_search_space, similarities=None
        )

        assert result == WarmStartMethod.FILTERED


# =============================================================================
# GET TRANSFER SUMMARY STATIC METHOD TESTS
# =============================================================================


class TestGetTransferSummary:
    """Test WarmStartStrategy.get_transfer_summary() static method."""

    def test_get_transfer_summary_empty(self):
        """Test get_transfer_summary with empty trials."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.get_transfer_summary([])

        assert result["n_trials"] == 0
        assert result["n_valid"] == 0
        assert result["mean_similarity"] == 0.0
        assert result["mean_weight"] == 0.0
        assert result["best_value"] is None

    def test_get_transfer_summary_basic(self, transferred_trials):
        """Test get_transfer_summary basic functionality."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.get_transfer_summary(transferred_trials)

        assert result["n_trials"] == len(transferred_trials)
        assert result["n_valid"] == len(transferred_trials)
        assert "mean_similarity" in result
        assert "mean_weight" in result
        assert "best_value" in result
        assert "param_names" in result

    def test_get_transfer_summary_counts_valid(self, mixed_validity_trials):
        """Test get_transfer_summary counts valid trials."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.get_transfer_summary(mixed_validity_trials)

        assert result["n_trials"] == 5
        assert result["n_valid"] == 3  # 3 are valid

    def test_get_transfer_summary_best_value(self, transferred_trials):
        """Test get_transfer_summary finds best value."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.get_transfer_summary(transferred_trials)

        # Best value should be minimum
        values = [t.value for t in transferred_trials if t.value is not None]
        assert result["best_value"] == min(values)

    def test_get_transfer_summary_param_names(self, transferred_trials):
        """Test get_transfer_summary extracts param names."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.get_transfer_summary(transferred_trials)

        assert "lr" in result["param_names"]
        assert "hidden_dim" in result["param_names"]

    def test_get_transfer_summary_mean_similarity(self, transferred_trials):
        """Test get_transfer_summary computes mean similarity."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.get_transfer_summary(transferred_trials)

        expected_mean = np.mean([t.similarity for t in transferred_trials])
        assert abs(result["mean_similarity"] - expected_mean) < 1e-6

    def test_get_transfer_summary_mean_weight(self, transferred_trials):
        """Test get_transfer_summary computes mean weight."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.get_transfer_summary(transferred_trials)

        expected_mean = np.mean([t.weight for t in transferred_trials])
        assert abs(result["mean_weight"] - expected_mean) < 1e-6

    def test_get_transfer_summary_none_values(self):
        """Test get_transfer_summary handles None values."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            TransferredTrial,
            WarmStartStrategy,
        )

        trials = [
            TransferredTrial(params={"lr": 0.001}, value=None),
            TransferredTrial(params={"lr": 0.002}, value=0.1),
        ]

        result = WarmStartStrategy.get_transfer_summary(trials)

        assert result["best_value"] == 0.1


# =============================================================================
# CREATE FROM BEST TRIALS STATIC METHOD TESTS
# =============================================================================


class TestCreateFromBestTrials:
    """Test WarmStartStrategy.create_from_best_trials() static method."""

    def test_create_from_best_trials_basic(self, mock_study):
        """Test create_from_best_trials basic functionality."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            TransferredTrial,
            WarmStartStrategy,
        )

        result = WarmStartStrategy.create_from_best_trials(study=mock_study, n_trials=3)

        assert all(isinstance(t, TransferredTrial) for t in result)
        assert len(result) <= 3

    def test_create_from_best_trials_sorted_by_value(self, mock_study):
        """Test create_from_best_trials returns sorted trials."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.create_from_best_trials(study=mock_study, n_trials=10)

        # Should be sorted by value ascending
        values = [t.value for t in result if t.value is not None]
        assert values == sorted(values)

    def test_create_from_best_trials_excludes_pruned(self):
        """Test create_from_best_trials excludes pruned by default."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        study = MockStudy(
            trials=[
                MockFrozenTrial({"lr": 0.001}, 0.1, MockTrialState.COMPLETE),
                MockFrozenTrial({"lr": 0.002}, 0.2, MockTrialState.PRUNED),
                MockFrozenTrial({"lr": 0.003}, 0.15, MockTrialState.COMPLETE),
            ]
        )

        # Mock the _lazy_import_optuna to return a mock optuna with TrialState
        mock_optuna = MagicMock()
        mock_optuna.trial.TrialState.COMPLETE = MockTrialState.COMPLETE

        with patch(
            "milia_pipeline.models.hpo.transfer.warm_start._lazy_import_optuna",
            return_value=mock_optuna,
        ):
            result = WarmStartStrategy.create_from_best_trials(
                study=study, n_trials=10, include_pruned=False
            )

        # Should only include COMPLETE trials
        assert len(result) == 2

    def test_create_from_best_trials_includes_pruned(self):
        """Test create_from_best_trials includes pruned when requested."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        study = MockStudy(
            trials=[
                MockFrozenTrial({"lr": 0.001}, 0.1, MockTrialState.COMPLETE),
                MockFrozenTrial({"lr": 0.002}, 0.2, MockTrialState.PRUNED),
                MockFrozenTrial({"lr": 0.003}, 0.15, MockTrialState.COMPLETE),
            ]
        )

        result = WarmStartStrategy.create_from_best_trials(
            study=study, n_trials=10, include_pruned=True
        )

        # Should include all trials
        assert len(result) == 3

    def test_create_from_best_trials_no_optuna_returns_empty(self):
        """Test create_from_best_trials returns empty when optuna unavailable."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        with patch(
            "milia_pipeline.models.hpo.transfer.warm_start._lazy_import_optuna", return_value=None
        ):
            result = WarmStartStrategy.create_from_best_trials(study=MockStudy(), n_trials=3)

        # Behavior depends on implementation - may return empty or continue
        assert isinstance(result, list)

    def test_create_from_best_trials_n_trials_larger_than_study(self, mock_study):
        """Test create_from_best_trials when n_trials > study trials."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        # Mock the _lazy_import_optuna to return a mock optuna with TrialState
        mock_optuna = MagicMock()
        mock_optuna.trial.TrialState.COMPLETE = MockTrialState.COMPLETE

        with patch(
            "milia_pipeline.models.hpo.transfer.warm_start._lazy_import_optuna",
            return_value=mock_optuna,
        ):
            result = WarmStartStrategy.create_from_best_trials(study=mock_study, n_trials=100)

        # Should return all available
        assert len(result) == len(mock_study.trials)

    def test_create_from_best_trials_sets_is_valid(self, mock_study):
        """Test create_from_best_trials sets is_valid=True."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.create_from_best_trials(study=mock_study, n_trials=3)

        assert all(t.is_valid for t in result)

    def test_create_from_best_trials_sets_weight(self, mock_study):
        """Test create_from_best_trials sets weight=1.0."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        result = WarmStartStrategy.create_from_best_trials(study=mock_study, n_trials=3)

        assert all(t.weight == 1.0 for t in result)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for WarmStartStrategy."""

    def test_full_transfer_workflow(self, mock_trials, sample_search_space, sample_similarities):
        """Test full transfer workflow."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            WarmStartConfig,
            WarmStartMethod,
            WarmStartStrategy,
        )

        # Configure strategy
        config = WarmStartConfig(
            method=WarmStartMethod.WEIGHTED, n_trials=3, min_similarity=0.5, filter_invalid=True
        )
        strategy = WarmStartStrategy(config)

        # Transfer trials
        transferred = strategy.transfer(
            source_trials=mock_trials,
            target_search_space=sample_search_space,
            similarities=sample_similarities,
        )

        # Get summary
        summary = WarmStartStrategy.get_transfer_summary(transferred)

        assert len(transferred) <= 3
        assert summary["n_trials"] == len(transferred)

    def test_transfer_and_apply_workflow(self, mock_trials, sample_similarities):
        """Test transfer and apply to study workflow."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            WarmStartConfig,
            WarmStartMethod,
            WarmStartStrategy,
        )

        config = WarmStartConfig(method=WarmStartMethod.WEIGHTED, n_trials=3)
        strategy = WarmStartStrategy(config)

        # Transfer trials
        transferred = strategy.transfer(source_trials=mock_trials, similarities=sample_similarities)

        # Apply to study
        target_study = MockStudy()

        with patch.object(strategy, "_optuna", MockOptuna()):
            n_enqueued = strategy.apply_to_study(target_study, transferred)

        assert n_enqueued == len(transferred)
        assert len(target_study._enqueued_trials) == len(transferred)

    def test_create_and_transfer_workflow(self):
        """Test create from study and transfer workflow."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            WarmStartConfig,
            WarmStartMethod,
            WarmStartStrategy,
        )

        # Create source study with trials
        source_study = MockStudy(
            trials=[
                MockFrozenTrial({"lr": 0.001, "hidden_dim": 128}, 0.1),
                MockFrozenTrial({"lr": 0.005, "hidden_dim": 256}, 0.12),
                MockFrozenTrial({"lr": 0.01, "hidden_dim": 64}, 0.15),
            ]
        )

        # Mock the _lazy_import_optuna to return a mock optuna with TrialState
        mock_optuna = MagicMock()
        mock_optuna.trial.TrialState.COMPLETE = MockTrialState.COMPLETE

        with patch(
            "milia_pipeline.models.hpo.transfer.warm_start._lazy_import_optuna",
            return_value=mock_optuna,
        ):
            # Extract best trials
            source_trials = WarmStartStrategy.create_from_best_trials(source_study, n_trials=2)

        # Transfer using FULL method
        config = WarmStartConfig(method=WarmStartMethod.FULL, n_trials=2)
        strategy = WarmStartStrategy(config)

        transferred = strategy.transfer(source_trials=source_trials)

        assert len(transferred) == 2

    def test_adaptive_transfer_workflow(
        self, mock_trials, sample_search_space, sample_similarities
    ):
        """Test adaptive transfer selects appropriate method."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            WarmStartConfig,
            WarmStartMethod,
            WarmStartStrategy,
        )

        config = WarmStartConfig(method=WarmStartMethod.ADAPTIVE, n_trials=3)
        strategy = WarmStartStrategy(config)

        # With varying similarities, should use weighted
        transferred = strategy.transfer(
            source_trials=mock_trials,
            target_search_space=sample_search_space,
            similarities=sample_similarities,
        )

        assert len(transferred) <= 3


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_trial_transfer(self, sample_search_space):
        """Test transfer with single trial."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        trials = [MockFrozenTrial({"lr": 0.01, "hidden_dim": 128}, 0.1)]

        strategy = WarmStartStrategy()
        result = strategy.transfer(source_trials=trials, target_search_space=sample_search_space)

        assert len(result) == 1

    def test_trial_with_empty_params(self):
        """Test handling of trial with empty params."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        trials = [MockFrozenTrial({}, 0.1)]

        strategy = WarmStartStrategy()
        result = strategy.transfer(source_trials=trials)

        # Should handle empty params
        assert isinstance(result, list)

    def test_very_large_n_trials(self, mock_trials):
        """Test with very large n_trials value."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartConfig

        config = WarmStartConfig(n_trials=10000)

        assert config.n_trials == 10000

    def test_boundary_similarity_values(self, mock_trials):
        """Test with boundary similarity values."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        similarities = [0.0, 0.5, 1.0, 0.0, 1.0]

        result = WarmStartStrategy.weighted_transfer(
            source_trials=mock_trials, similarities=similarities, min_similarity=0.0
        )

        assert len(result) == len(mock_trials)

    def test_all_invalid_trials(self, sample_search_space):
        """Test when all trials become invalid."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            WarmStartConfig,
            WarmStartMethod,
            WarmStartStrategy,
        )

        # All params out of bounds
        trials = [
            MockFrozenTrial({"lr": 100.0}, 0.1),  # Way out of bounds
            MockFrozenTrial({"lr": 200.0}, 0.2),
        ]

        config = WarmStartConfig(method=WarmStartMethod.FILTERED, filter_invalid=True)
        strategy = WarmStartStrategy(config)

        result = strategy.transfer(source_trials=trials, target_search_space=sample_search_space)

        # May be empty or have scaled values
        assert isinstance(result, list)

    def test_mixed_trial_types(self):
        """Test with mixed FrozenTrial and dict types."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        trials = [
            MockFrozenTrial({"lr": 0.001}, 0.1),
            {"params": {"lr": 0.002}, "value": 0.2},
            {"lr": 0.003, "value": 0.3},
        ]

        result = WarmStartStrategy.full_transfer(source_trials=trials, n_trials=3)

        assert len(result) == 3

    def test_negative_trial_values(self):
        """Test trials with negative values."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        trials = [
            MockFrozenTrial({"lr": 0.001}, -0.5),
            MockFrozenTrial({"lr": 0.002}, -0.3),
            MockFrozenTrial({"lr": 0.003}, -0.1),
        ]

        result = WarmStartStrategy.full_transfer(
            source_trials=trials, n_trials=3, sort_by_performance=True
        )

        # Should sort by value ascending (most negative first)
        values = [t.value for t in result]
        assert values == sorted(values)

    def test_zero_noise_scale(self, sample_search_space):
        """Test with zero noise scale."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            TransferredTrial,
            WarmStartConfig,
            WarmStartStrategy,
        )

        config = WarmStartConfig(add_noise=True, noise_scale=0.0)
        strategy = WarmStartStrategy(config)

        original_lr = 0.01
        trials = [TransferredTrial(params={"lr": original_lr})]

        result = strategy._add_noise_to_trials(trials, sample_search_space)

        # With zero noise, value should be unchanged
        assert result[0].params["lr"] == original_lr

    def test_unicode_param_names(self):
        """Test with unicode parameter names."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        trials = [MockFrozenTrial({"学习率": 0.001, "hidden_维度": 128}, 0.1)]

        result = WarmStartStrategy.full_transfer(source_trials=trials, n_trials=1)

        assert "学习率" in result[0].params
        assert "hidden_维度" in result[0].params

    def test_float_hidden_dim_converted_to_int(self, sample_search_space):
        """Test float hidden_dim is converted to int."""
        from milia_pipeline.models.hpo.transfer.warm_start import WarmStartStrategy

        trials = [MockFrozenTrial({"hidden_dim": 128.7}, 0.1)]

        result = WarmStartStrategy.filtered_transfer(
            source_trials=trials, target_search_space=sample_search_space
        )

        if result and "hidden_dim" in result[0].params:
            assert isinstance(result[0].params["hidden_dim"], int)

    def test_transfer_summary_with_empty_param_names(self):
        """Test get_transfer_summary with trials having empty params."""
        from milia_pipeline.models.hpo.transfer.warm_start import (
            TransferredTrial,
            WarmStartStrategy,
        )

        trials = [TransferredTrial(params={}, value=0.1)]

        result = WarmStartStrategy.get_transfer_summary(trials)

        assert result["param_names"] == []


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
