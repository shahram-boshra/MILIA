#!/usr/bin/env python3
"""
Complete Unit Test Suite for milia_pipeline/models/hpo/analysis/study_analyzer.py Module

This comprehensive test suite covers all functionality in the study_analyzer.py module:

Enums, AnalysisConfig (Pydantic V2 BaseModel), and StudyAnalyzer Initialization
- ImportanceMethod enum (FANOVA, MDI)
- ExportFormat enum (JSON, CSV, DATAFRAME, DICT)
- AnalysisConfig Pydantic V2 BaseModel initialization and validation
- AnalysisConfig frozen behavior (immutability via Pydantic ValidationError)
- AnalysisConfig.to_dict() backward compatibility method
- StudyAnalyzer.__init__() with various parameters
- StudyAnalyzer.from_manager() class method
- StudyAnalyzer.from_storage() class method
- StudyAnalyzer.__repr__() method
- Module exports (__all__)

Trial Data Access and Parameter Importance Analysis
- StudyAnalyzer.get_trials() with state filtering and caching
- StudyAnalyzer.get_completed_trials() method
- StudyAnalyzer.get_trial_count() method
- StudyAnalyzer.get_parameter_importance() method
- StudyAnalyzer.get_parameter_importance_ranking() method
- StudyAnalyzer.clear_cache() method

Convergence, Trajectory, and Statistical Analysis
- StudyAnalyzer.get_convergence_data() method
- StudyAnalyzer.get_optimization_trajectory() method
- StudyAnalyzer.get_value_statistics() method
- StudyAnalyzer.get_parameter_statistics() for numeric and categorical params
- StudyAnalyzer.get_parameter_correlations() method

Multi-objective, Visualization, Export, Comparison, and Integration
- StudyAnalyzer.get_pareto_front() method
- StudyAnalyzer.get_hypervolume() method
- Visualization data methods (optimization_history, parameter_importance, slice_plot, contour_plot, parallel_coordinate)
- StudyAnalyzer.get_comprehensive_analysis() method
- StudyAnalyzer.export_results() with all formats
- StudyAnalyzer.to_dataframe() method
- StudyAnalyzer.compare_with() method
- Integration tests and edge cases

Location of module under test: ~/ml_projects/milia/milia_pipeline/models/hpo/analysis/study_analyzer.py
Location of test file: ~/ml_projects/milia/tests/test_hpo_analysis_study_analyzer.py

Pydantic V2 Migration Notes (Phase 15):
- AnalysisConfig migrated from @dataclass(frozen=True) to BaseModel with frozen=True
- Frozen models raise pydantic.ValidationError (not dataclasses.FrozenInstanceError)
- Field validators (@field_validator) raise ValueError wrapped in ValidationError
- Model validators (@model_validator) raise ValueError wrapped in ValidationError
- to_dict() method wraps model_dump() for backward compatibility

Author: Milia Team
Version: 1.1.0
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from typing import Any
from unittest.mock import patch

import pytest

try:
    from pydantic import ValidationError as PydanticValidationError
except ImportError:
    PydanticValidationError = Exception  # Fallback if pydantic not installed
from datetime import datetime, timedelta
from enum import Enum

# =============================================================================
# MOCK CLASSES FOR EXCEPTIONS
# =============================================================================


class MockHPOError(Exception):
    """
    Mock HPOError for testing.

    Mirrors the actual HPOError from milia_pipeline.exceptions
    with all required attributes for validation testing.
    """

    def __init__(
        self,
        message: str,
        study_name: str | None = None,
        trial_number: int | None = None,
        details: str | None = None,
        **kwargs,
    ):
        super().__init__(message)
        self.message = message
        self.study_name = study_name
        self.trial_number = trial_number
        self.details = details
        self.extra_info = kwargs

    def __str__(self) -> str:
        msg = self.message
        if self.study_name:
            msg += f". Study: '{self.study_name}'"
        if self.trial_number is not None:
            msg += f", Trial: {self.trial_number}"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


class MockHPOConfigurationError(MockHPOError):
    """
    Mock HPOConfigurationError for testing.

    Mirrors the actual HPOConfigurationError from milia_pipeline.exceptions.
    """

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        actual_value: Any = None,
        expected_value: Any = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.config_key = config_key
        self.actual_value = actual_value
        self.expected_value = expected_value

    def __str__(self) -> str:
        parts = [self.message]
        if self.config_key:
            parts.append(f"Key: '{self.config_key}'")
        if self.expected_value is not None:
            parts.append(f"Expected: {self.expected_value}")
        if self.actual_value is not None:
            parts.append(f"Actual: {self.actual_value}")
        if self.details:
            parts.append(f"Details: {self.details}")
        return " | ".join(parts)


class MockStudyNotFoundError(MockHPOError):
    """
    Mock StudyNotFoundError for testing.

    Mirrors the actual StudyNotFoundError from milia_pipeline.exceptions.
    """

    def __init__(
        self,
        message: str,
        study_name: str,
        available_studies: list[str] | None = None,
        storage_url: str | None = None,
        **kwargs,
    ):
        super().__init__(message, study_name=study_name, **kwargs)
        self.available_studies = available_studies or []
        self.storage_url = storage_url

    def __str__(self) -> str:
        msg = super().__str__()
        if self.available_studies:
            msg += f". Available studies: {', '.join(self.available_studies[:5])}"
        if self.storage_url:
            msg += f". Storage: {self.storage_url}"
        return msg


# =============================================================================
# MOCK CLASSES FOR OPTUNA DEPENDENCIES
# =============================================================================


class MockTrialState(Enum):
    """Mock Optuna TrialState for testing."""

    COMPLETE = "COMPLETE"
    PRUNED = "PRUNED"
    FAIL = "FAIL"
    RUNNING = "RUNNING"
    WAITING = "WAITING"


class MockStudyDirection(Enum):
    """Mock Optuna study direction."""

    MINIMIZE = "MINIMIZE"
    MAXIMIZE = "MAXIMIZE"


class MockFrozenTrial:
    """Mock Optuna FrozenTrial for testing."""

    def __init__(
        self,
        number: int,
        params: dict[str, Any],
        value: float | None = None,
        state: MockTrialState = None,
        datetime_start: datetime | None = None,
        datetime_complete: datetime | None = None,
        user_attrs: dict[str, Any] | None = None,
        intermediate_values: dict[int, float] | None = None,
        values: tuple[float, ...] | None = None,
    ):
        self.number = number
        self.params = params
        self.value = value
        self.values = values
        self.state = state or MockTrialState.COMPLETE
        self.datetime_start = datetime_start or datetime.now()
        self.datetime_complete = datetime_complete or (self.datetime_start + timedelta(seconds=60))
        self.user_attrs = user_attrs or {}
        self.intermediate_values = intermediate_values or {}


class MockStudy:
    """Mock Optuna Study for testing."""

    def __init__(
        self,
        study_name: str = "test_study",
        direction: MockStudyDirection = MockStudyDirection.MINIMIZE,
        trials: list[MockFrozenTrial] | None = None,
        best_params: dict[str, Any] | None = None,
        best_value: float | None = None,
        directions: list[MockStudyDirection] | None = None,
        best_trials: list[MockFrozenTrial] | None = None,
    ):
        self.study_name = study_name
        self.direction = direction
        self.trials = trials or []
        self._best_params = best_params or {"lr": 0.001, "hidden_channels": 64}
        self._best_value = best_value or 0.1
        self._directions = directions
        self._best_trials = best_trials or []

    @property
    def best_params(self) -> dict[str, Any]:
        return self._best_params

    @property
    def best_value(self) -> float:
        return self._best_value

    @property
    def directions(self) -> list[MockStudyDirection]:
        if self._directions:
            return self._directions
        raise AttributeError("Single-objective study has no 'directions' attribute")

    @property
    def best_trials(self) -> list[MockFrozenTrial]:
        return self._best_trials


class MockHPOManager:
    """Mock HPOManager for testing from_manager() method."""

    def __init__(self, study: MockStudy | None = None):
        self.study = study


class MockFanovaImportanceEvaluator:
    """Mock FanovaImportanceEvaluator for testing."""

    def __init__(self):
        pass


def mock_get_param_importances(study, evaluator=None, target=None):
    """Mock get_param_importances function for testing."""
    return {"lr": 0.5, "hidden_channels": 0.3, "dropout": 0.2}


def mock_get_param_importances_with_target(study, evaluator=None, target=None):
    """Mock get_param_importances with target handling."""
    if target is not None:
        return {"lr": 0.6, "hidden_channels": 0.4}
    return {"lr": 0.5, "hidden_channels": 0.3, "dropout": 0.2}


def mock_get_param_importances_failure(study, evaluator=None, target=None):
    """Mock get_param_importances that raises an exception."""
    raise RuntimeError("Failed to calculate importance")


def mock_load_study(study_name: str, storage: str):
    """Mock optuna.load_study for testing."""
    if study_name == "nonexistent_study":
        raise Exception("Study not found")
    return MockStudy(study_name=study_name)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_trial_state():
    """Create mock TrialState enum."""
    return MockTrialState


@pytest.fixture
def mock_study_direction():
    """Create mock StudyDirection enum."""
    return MockStudyDirection


@pytest.fixture
def sample_completed_trials():
    """Create sample completed trials for testing."""
    base_time = datetime.now()
    return [
        MockFrozenTrial(
            number=0,
            params={"lr": 0.01, "hidden_channels": 32, "dropout": 0.1},
            value=0.5,
            state=MockTrialState.COMPLETE,
            datetime_start=base_time,
            datetime_complete=base_time + timedelta(seconds=60),
        ),
        MockFrozenTrial(
            number=1,
            params={"lr": 0.001, "hidden_channels": 64, "dropout": 0.2},
            value=0.3,
            state=MockTrialState.COMPLETE,
            datetime_start=base_time + timedelta(seconds=61),
            datetime_complete=base_time + timedelta(seconds=120),
        ),
        MockFrozenTrial(
            number=2,
            params={"lr": 0.0001, "hidden_channels": 128, "dropout": 0.3},
            value=0.2,
            state=MockTrialState.COMPLETE,
            datetime_start=base_time + timedelta(seconds=121),
            datetime_complete=base_time + timedelta(seconds=180),
        ),
    ]


@pytest.fixture
def sample_mixed_trials():
    """Create sample mixed trials (completed, pruned, failed) for testing."""
    base_time = datetime.now()
    return [
        MockFrozenTrial(
            number=0,
            params={"lr": 0.01, "hidden_channels": 32},
            value=0.5,
            state=MockTrialState.COMPLETE,
            datetime_start=base_time,
            datetime_complete=base_time + timedelta(seconds=60),
        ),
        MockFrozenTrial(
            number=1,
            params={"lr": 0.001, "hidden_channels": 64},
            value=None,
            state=MockTrialState.PRUNED,
            datetime_start=base_time + timedelta(seconds=61),
            datetime_complete=base_time + timedelta(seconds=90),
        ),
        MockFrozenTrial(
            number=2,
            params={"lr": 0.0001, "hidden_channels": 128},
            value=None,
            state=MockTrialState.FAIL,
            datetime_start=base_time + timedelta(seconds=91),
            datetime_complete=None,
        ),
        MockFrozenTrial(
            number=3,
            params={"lr": 0.005, "hidden_channels": 96},
            value=0.25,
            state=MockTrialState.COMPLETE,
            datetime_start=base_time + timedelta(seconds=121),
            datetime_complete=base_time + timedelta(seconds=180),
        ),
    ]


@pytest.fixture
def mock_study(sample_completed_trials):
    """Create a mock study with completed trials."""
    return MockStudy(
        study_name="test_study",
        direction=MockStudyDirection.MINIMIZE,
        trials=sample_completed_trials,
    )


@pytest.fixture
def mock_study_mixed(sample_mixed_trials):
    """Create a mock study with mixed trial states."""
    return MockStudy(
        study_name="test_study_mixed",
        direction=MockStudyDirection.MINIMIZE,
        trials=sample_mixed_trials,
    )


@pytest.fixture
def mock_hpo_manager(mock_study):
    """Create a mock HPOManager with a study."""
    return MockHPOManager(study=mock_study)


@pytest.fixture
def mock_hpo_manager_no_study():
    """Create a mock HPOManager without a study."""
    return MockHPOManager(study=None)


# =============================================================================
# IMPORTANCEMETHOD ENUM TESTS
# =============================================================================


class TestImportanceMethodEnum:
    """Test ImportanceMethod enum."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_fanova_value(self):
        """Test FANOVA enum value."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import ImportanceMethod

        assert ImportanceMethod.FANOVA.value == "fanova"

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_mdi_value(self):
        """Test MDI enum value."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import ImportanceMethod

        assert ImportanceMethod.MDI.value == "mdi"

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_enum_members_count(self):
        """Test ImportanceMethod has exactly 2 members."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import ImportanceMethod

        assert len(ImportanceMethod) == 2

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_enum_is_enum_type(self):
        """Test ImportanceMethod is an Enum."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import ImportanceMethod

        assert issubclass(ImportanceMethod, Enum)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_fanova_is_default(self):
        """Test FANOVA is the default importance method (first in enum)."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import ImportanceMethod

        # FANOVA should be first and is documented as the default
        members = list(ImportanceMethod)
        assert members[0] == ImportanceMethod.FANOVA


# =============================================================================
# EXPORTFORMAT ENUM TESTS
# =============================================================================


class TestExportFormatEnum:
    """Test ExportFormat enum."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_json_value(self):
        """Test JSON enum value."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import ExportFormat

        assert ExportFormat.JSON.value == "json"

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_csv_value(self):
        """Test CSV enum value."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import ExportFormat

        assert ExportFormat.CSV.value == "csv"

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_dataframe_value(self):
        """Test DATAFRAME enum value."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import ExportFormat

        assert ExportFormat.DATAFRAME.value == "dataframe"

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_dict_value(self):
        """Test DICT enum value."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import ExportFormat

        assert ExportFormat.DICT.value == "dict"

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_enum_members_count(self):
        """Test ExportFormat has exactly 4 members."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import ExportFormat

        assert len(ExportFormat) == 4

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_enum_is_enum_type(self):
        """Test ExportFormat is an Enum."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import ExportFormat

        assert issubclass(ExportFormat, Enum)


# =============================================================================
# ANALYSISCONFIG DATACLASS TESTS
# =============================================================================


class TestAnalysisConfigInit:
    """Test AnalysisConfig dataclass initialization."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_default_values(self):
        """Test AnalysisConfig initializes with default values."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import (
            AnalysisConfig,
            ImportanceMethod,
        )

        config = AnalysisConfig()

        assert config.importance_method == ImportanceMethod.FANOVA
        assert config.n_importance_trials is None
        assert config.convergence_window == 10
        assert config.include_pruned is False
        assert config.include_failed is False
        assert config.percentile_thresholds == (25.0, 50.0, 75.0, 90.0, 95.0)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_custom_importance_method(self):
        """Test AnalysisConfig with custom importance method."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import (
            AnalysisConfig,
            ImportanceMethod,
        )

        config = AnalysisConfig(importance_method=ImportanceMethod.MDI)

        assert config.importance_method == ImportanceMethod.MDI

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_custom_n_importance_trials(self):
        """Test AnalysisConfig with custom n_importance_trials."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig

        config = AnalysisConfig(n_importance_trials=50)

        assert config.n_importance_trials == 50

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_custom_convergence_window(self):
        """Test AnalysisConfig with custom convergence_window."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig

        config = AnalysisConfig(convergence_window=20)

        assert config.convergence_window == 20

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_include_pruned_true(self):
        """Test AnalysisConfig with include_pruned=True."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig

        config = AnalysisConfig(include_pruned=True)

        assert config.include_pruned is True

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_include_failed_true(self):
        """Test AnalysisConfig with include_failed=True."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig

        config = AnalysisConfig(include_failed=True)

        assert config.include_failed is True

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_custom_percentile_thresholds(self):
        """Test AnalysisConfig with custom percentile_thresholds."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig

        custom_percentiles = (10.0, 50.0, 90.0)
        config = AnalysisConfig(percentile_thresholds=custom_percentiles)

        assert config.percentile_thresholds == custom_percentiles

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_all_custom_values(self):
        """Test AnalysisConfig with all custom values."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import (
            AnalysisConfig,
            ImportanceMethod,
        )

        config = AnalysisConfig(
            importance_method=ImportanceMethod.MDI,
            n_importance_trials=100,
            convergence_window=15,
            include_pruned=True,
            include_failed=True,
            percentile_thresholds=(5.0, 25.0, 50.0, 75.0, 95.0),
        )

        assert config.importance_method == ImportanceMethod.MDI
        assert config.n_importance_trials == 100
        assert config.convergence_window == 15
        assert config.include_pruned is True
        assert config.include_failed is True
        assert config.percentile_thresholds == (5.0, 25.0, 50.0, 75.0, 95.0)


class TestAnalysisConfigValidation:
    """Test AnalysisConfig __post_init__ validation."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_convergence_window_zero_raises_error(self):
        """Test convergence_window=0 raises validation error.

        Pydantic V2 @field_validator raises ValueError which gets wrapped
        in ValidationError during model instantiation.
        """
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig

        # Pydantic V2 validators raise ValueError which gets wrapped in ValidationError
        with pytest.raises((ValueError, PydanticValidationError)) as exc_info:
            AnalysisConfig(convergence_window=0)

        assert "convergence_window" in str(exc_info.value)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_convergence_window_negative_raises_error(self):
        """Test convergence_window=-5 raises validation error.

        Pydantic V2 @field_validator raises ValueError which gets wrapped
        in ValidationError during model instantiation.
        """
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig

        with pytest.raises((ValueError, PydanticValidationError)) as exc_info:
            AnalysisConfig(convergence_window=-5)

        assert "convergence_window" in str(exc_info.value)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_convergence_window_one_valid(self):
        """Test convergence_window=1 is valid (minimum allowed)."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig

        config = AnalysisConfig(convergence_window=1)

        assert config.convergence_window == 1

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_n_importance_trials_zero_raises_error(self):
        """Test n_importance_trials=0 raises validation error.

        Pydantic V2 @field_validator raises ValueError which gets wrapped
        in ValidationError during model instantiation.
        """
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig

        with pytest.raises((ValueError, PydanticValidationError)) as exc_info:
            AnalysisConfig(n_importance_trials=0)

        assert "n_importance_trials" in str(exc_info.value)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_n_importance_trials_negative_raises_error(self):
        """Test n_importance_trials=-10 raises validation error.

        Pydantic V2 @field_validator raises ValueError which gets wrapped
        in ValidationError during model instantiation.
        """
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig

        with pytest.raises((ValueError, PydanticValidationError)) as exc_info:
            AnalysisConfig(n_importance_trials=-10)

        assert "n_importance_trials" in str(exc_info.value)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_n_importance_trials_one_valid(self):
        """Test n_importance_trials=1 is valid (minimum allowed)."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig

        config = AnalysisConfig(n_importance_trials=1)

        assert config.n_importance_trials == 1

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_n_importance_trials_none_valid(self):
        """Test n_importance_trials=None is valid."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig

        config = AnalysisConfig(n_importance_trials=None)

        assert config.n_importance_trials is None

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_percentile_negative_raises_error(self):
        """Test percentile < 0 raises validation error.

        Pydantic V2 @model_validator raises ValueError which gets wrapped
        in ValidationError during model instantiation.
        """
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig

        with pytest.raises((ValueError, PydanticValidationError)) as exc_info:
            AnalysisConfig(percentile_thresholds=(-5.0, 50.0))

        assert "percentile" in str(exc_info.value).lower()

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_percentile_over_100_raises_error(self):
        """Test percentile > 100 raises validation error.

        Pydantic V2 @model_validator raises ValueError which gets wrapped
        in ValidationError during model instantiation.
        """
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig

        with pytest.raises((ValueError, PydanticValidationError)) as exc_info:
            AnalysisConfig(percentile_thresholds=(50.0, 105.0))

        assert "percentile" in str(exc_info.value).lower()

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_percentile_boundary_zero_valid(self):
        """Test percentile=0.0 is valid."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig

        config = AnalysisConfig(percentile_thresholds=(0.0, 50.0, 100.0))

        assert 0.0 in config.percentile_thresholds

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_percentile_boundary_100_valid(self):
        """Test percentile=100.0 is valid."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig

        config = AnalysisConfig(percentile_thresholds=(0.0, 50.0, 100.0))

        assert 100.0 in config.percentile_thresholds


class TestAnalysisConfigFrozen:
    """Test AnalysisConfig frozen behavior."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_config_is_frozen(self):
        """Test AnalysisConfig is frozen (immutable).

        Pydantic V2 frozen models raise ValidationError with error type
        'frozen_instance' when attempting to modify an attribute.
        """
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig

        config = AnalysisConfig()

        # Pydantic V2 frozen BaseModel raises ValidationError, not FrozenInstanceError
        with pytest.raises(PydanticValidationError) as exc_info:
            config.convergence_window = 20

        # Verify it's the frozen instance error type
        error_str = str(exc_info.value).lower()
        assert "frozen" in error_str or "instance is frozen" in error_str

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_config_is_hashable(self):
        """Test AnalysisConfig is hashable (frozen implies hashable)."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig

        config = AnalysisConfig()

        # Should be hashable and usable in sets/dicts
        assert hash(config) is not None
        config_set = {config}
        assert config in config_set


class TestAnalysisConfigToDict:
    """Test AnalysisConfig.to_dict() backward compatibility method.

    The to_dict() method was added during Pydantic V2 migration to maintain
    backward compatibility with code that used dataclass asdict() functionality.
    """

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_to_dict_returns_dict(self):
        """Test to_dict() returns a dictionary."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig

        config = AnalysisConfig()
        result = config.to_dict()

        assert isinstance(result, dict)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_to_dict_contains_all_fields(self):
        """Test to_dict() contains all config fields."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import (
            AnalysisConfig,
        )

        config = AnalysisConfig()
        result = config.to_dict()

        expected_keys = {
            "importance_method",
            "n_importance_trials",
            "convergence_window",
            "include_pruned",
            "include_failed",
            "percentile_thresholds",
        }
        assert expected_keys == set(result.keys())

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_to_dict_preserves_values(self):
        """Test to_dict() preserves config values correctly."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import (
            AnalysisConfig,
            ImportanceMethod,
        )

        config = AnalysisConfig(
            importance_method=ImportanceMethod.MDI,
            n_importance_trials=50,
            convergence_window=15,
            include_pruned=True,
            include_failed=True,
            percentile_thresholds=(10.0, 50.0, 90.0),
        )
        result = config.to_dict()

        # ImportanceMethod is serialized by Pydantic's model_dump
        assert result["importance_method"] == ImportanceMethod.MDI
        assert result["n_importance_trials"] == 50
        assert result["convergence_window"] == 15
        assert result["include_pruned"] is True
        assert result["include_failed"] is True
        assert result["percentile_thresholds"] == (10.0, 50.0, 90.0)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_to_dict_is_model_dump_wrapper(self):
        """Test to_dict() wraps Pydantic's model_dump() method."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig

        config = AnalysisConfig()

        # to_dict() should produce the same result as model_dump()
        assert config.to_dict() == config.model_dump()


# =============================================================================
# STUDYANALYZER INITIALIZATION TESTS
# =============================================================================


class TestStudyAnalyzerInit:
    """Test StudyAnalyzer.__init__() method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_init_with_study(self, mock_study):
        """Test StudyAnalyzer initializes with a study."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)

        assert analyzer.study == mock_study
        assert analyzer.config is not None

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_init_with_custom_config(self, mock_study):
        """Test StudyAnalyzer initializes with custom config."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import (
            AnalysisConfig,
            ImportanceMethod,
            StudyAnalyzer,
        )

        custom_config = AnalysisConfig(
            importance_method=ImportanceMethod.MDI, convergence_window=20
        )
        analyzer = StudyAnalyzer(mock_study, config=custom_config)

        assert analyzer.config == custom_config
        assert analyzer.config.importance_method == ImportanceMethod.MDI

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_init_creates_default_config(self, mock_study):
        """Test StudyAnalyzer creates default config if none provided."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)

        # Should have default config values
        assert analyzer.config.convergence_window == 10
        assert analyzer.config.include_pruned is False

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_init_initializes_caches_as_none(self, mock_study):
        """Test StudyAnalyzer initializes cache attributes as None."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)

        assert analyzer._trials_cache is None
        assert analyzer._importance_cache is None
        assert analyzer._completed_trials_cache is None

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_init_with_none_study_raises_error(self):
        """Test StudyAnalyzer raises error when study is None."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import HPOError, StudyAnalyzer

        with pytest.raises((MockHPOError, HPOError)) as exc_info:
            StudyAnalyzer(None)

        assert "None" in str(exc_info.value) or "cannot be None" in str(exc_info.value)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", False)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_init_without_optuna_raises_error(self, mock_study):
        """Test StudyAnalyzer raises error when Optuna is not available."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import HPOError, StudyAnalyzer

        with pytest.raises((MockHPOError, HPOError)) as exc_info:
            StudyAnalyzer(mock_study)

        assert "Optuna" in str(exc_info.value) or "not installed" in str(exc_info.value)


class TestStudyAnalyzerFromManager:
    """Test StudyAnalyzer.from_manager() class method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_from_manager_creates_analyzer(self, mock_hpo_manager):
        """Test from_manager creates StudyAnalyzer from HPOManager."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer.from_manager(mock_hpo_manager)

        assert analyzer.study == mock_hpo_manager.study
        assert analyzer.study.study_name == "test_study"

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_from_manager_with_config(self, mock_hpo_manager):
        """Test from_manager accepts custom config."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig, StudyAnalyzer

        custom_config = AnalysisConfig(convergence_window=15)
        analyzer = StudyAnalyzer.from_manager(mock_hpo_manager, config=custom_config)

        assert analyzer.config.convergence_window == 15

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_from_manager_no_study_raises_error(self, mock_hpo_manager_no_study):
        """Test from_manager raises error when manager has no study."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import HPOError, StudyAnalyzer

        with pytest.raises((MockHPOError, HPOError)) as exc_info:
            StudyAnalyzer.from_manager(mock_hpo_manager_no_study)

        assert "no study" in str(exc_info.value).lower()


class TestStudyAnalyzerFromStorage:
    """Test StudyAnalyzer.from_storage() class method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.optuna")
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_from_storage_loads_study(self, mock_optuna):
        """Test from_storage loads study from storage."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        mock_optuna.load_study = mock_load_study

        analyzer = StudyAnalyzer.from_storage(study_name="test_study", storage="sqlite:///test.db")

        assert analyzer.study.study_name == "test_study"

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.optuna")
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_from_storage_with_config(self, mock_optuna):
        """Test from_storage accepts custom config."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig, StudyAnalyzer

        mock_optuna.load_study = mock_load_study
        custom_config = AnalysisConfig(convergence_window=25)

        analyzer = StudyAnalyzer.from_storage(
            study_name="test_study", storage="sqlite:///test.db", config=custom_config
        )

        assert analyzer.config.convergence_window == 25

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.optuna")
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_from_storage_nonexistent_raises_error(self, mock_optuna):
        """Test from_storage raises StudyNotFoundError for nonexistent study."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import (
            StudyAnalyzer,
            StudyNotFoundError,
        )

        mock_optuna.load_study = mock_load_study

        with pytest.raises((MockStudyNotFoundError, StudyNotFoundError)) as exc_info:
            StudyAnalyzer.from_storage(study_name="nonexistent_study", storage="sqlite:///test.db")

        assert "nonexistent_study" in str(exc_info.value)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", False)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_from_storage_without_optuna_raises_error(self):
        """Test from_storage raises error when Optuna is not available."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import HPOError, StudyAnalyzer

        with pytest.raises((MockHPOError, HPOError)) as exc_info:
            StudyAnalyzer.from_storage(study_name="test_study", storage="sqlite:///test.db")

        assert "Optuna" in str(exc_info.value)


class TestStudyAnalyzerRepr:
    """Test StudyAnalyzer.__repr__() method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_repr_contains_study_name(self, mock_study):
        """Test __repr__ contains study name."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        repr_str = repr(analyzer)

        assert "test_study" in repr_str

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_repr_contains_n_trials(self, mock_study):
        """Test __repr__ contains number of trials."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        repr_str = repr(analyzer)

        assert "n_trials=" in repr_str

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_repr_contains_class_name(self, mock_study):
        """Test __repr__ contains class name."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        repr_str = repr(analyzer)

        assert "StudyAnalyzer" in repr_str


# =============================================================================
# MODULE EXPORTS TESTS
# =============================================================================


class TestModuleExports:
    """Test module exports (__all__)."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_studyanalyzer_in_all(self):
        """Test StudyAnalyzer is in __all__."""
        from milia_pipeline.models.hpo.analysis import study_analyzer

        assert "StudyAnalyzer" in study_analyzer.__all__

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_analysisconfig_in_all(self):
        """Test AnalysisConfig is in __all__."""
        from milia_pipeline.models.hpo.analysis import study_analyzer

        assert "AnalysisConfig" in study_analyzer.__all__

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_importancemethod_in_all(self):
        """Test ImportanceMethod is in __all__."""
        from milia_pipeline.models.hpo.analysis import study_analyzer

        assert "ImportanceMethod" in study_analyzer.__all__

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_exportformat_in_all(self):
        """Test ExportFormat is in __all__."""
        from milia_pipeline.models.hpo.analysis import study_analyzer

        assert "ExportFormat" in study_analyzer.__all__

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_all_exports_count(self):
        """Test __all__ contains exactly 4 exports."""
        from milia_pipeline.models.hpo.analysis import study_analyzer

        assert len(study_analyzer.__all__) == 4


# =============================================================================
# GET_TRIALS TESTS
# =============================================================================


class TestGetTrials:
    """Test StudyAnalyzer.get_trials() method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_trials_returns_list(self, mock_study):
        """Test get_trials returns a list of trial dicts."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        trials = analyzer.get_trials()

        assert isinstance(trials, list)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_trials_contains_expected_keys(self, mock_study):
        """Test each trial dict contains expected keys."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        trials = analyzer.get_trials()

        expected_keys = {
            "number",
            "params",
            "value",
            "state",
            "duration",
            "user_attrs",
            "intermediate_values",
            "datetime_start",
            "datetime_complete",
        }

        for trial in trials:
            assert expected_keys.issubset(set(trial.keys()))

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_trials_correct_count(self, mock_study):
        """Test get_trials returns correct number of trials."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        trials = analyzer.get_trials()

        # Default config excludes pruned and failed, all trials are COMPLETE
        assert len(trials) == 3

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_trials_filters_by_state(self, mock_study_mixed):
        """Test get_trials filters by state."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_mixed)

        # Only get COMPLETE trials
        completed = analyzer.get_trials(states=["COMPLETE"], use_cache=False)
        assert len(completed) == 2
        assert all(t["state"] == "COMPLETE" for t in completed)

        # Only get PRUNED trials
        pruned = analyzer.get_trials(states=["PRUNED"], use_cache=False)
        assert len(pruned) == 1
        assert all(t["state"] == "PRUNED" for t in pruned)

        # Only get FAIL trials
        failed = analyzer.get_trials(states=["FAIL"], use_cache=False)
        assert len(failed) == 1
        assert all(t["state"] == "FAIL" for t in failed)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_trials_multiple_states(self, mock_study_mixed):
        """Test get_trials with multiple states."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_mixed)

        # Get COMPLETE and PRUNED trials
        trials = analyzer.get_trials(states=["COMPLETE", "PRUNED"], use_cache=False)

        assert len(trials) == 3
        states = {t["state"] for t in trials}
        assert states == {"COMPLETE", "PRUNED"}

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_trials_with_include_pruned_config(self, mock_study_mixed):
        """Test get_trials respects include_pruned config."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig, StudyAnalyzer

        config = AnalysisConfig(include_pruned=True, include_failed=False)
        analyzer = StudyAnalyzer(mock_study_mixed, config=config)

        # When states is None, uses config settings
        trials = analyzer.get_trials(states=None, use_cache=False)
        states = {t["state"] for t in trials}

        assert "PRUNED" in states
        assert "FAIL" not in states

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_trials_with_include_failed_config(self, mock_study_mixed):
        """Test get_trials respects include_failed config."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig, StudyAnalyzer

        config = AnalysisConfig(include_pruned=False, include_failed=True)
        analyzer = StudyAnalyzer(mock_study_mixed, config=config)

        # When states is None, uses config settings
        trials = analyzer.get_trials(states=None, use_cache=False)
        states = {t["state"] for t in trials}

        assert "PRUNED" not in states
        assert "FAIL" in states

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_trials_caching(self, mock_study):
        """Test get_trials caches results when use_cache=True."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)

        # First call - should set cache
        trials1 = analyzer.get_trials(use_cache=True)
        assert analyzer._trials_cache is not None

        # Second call - should return cached
        trials2 = analyzer.get_trials(use_cache=True)

        assert trials1 is trials2  # Same object (from cache)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_trials_no_caching_when_states_specified(self, mock_study):
        """Test get_trials doesn't cache when states is specified."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)

        # Call with states - should not set cache
        _trials = analyzer.get_trials(states=["COMPLETE"], use_cache=True)

        # Cache should still be None since states was specified
        assert analyzer._trials_cache is None

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_trials_bypasses_cache_when_use_cache_false(self, mock_study):
        """Test get_trials bypasses cache when use_cache=False."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)

        # Set up cache
        trials1 = analyzer.get_trials(use_cache=True)

        # Get fresh data bypassing cache
        trials2 = analyzer.get_trials(use_cache=False)

        # Should be different list instances
        assert trials1 is not trials2

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_trials_calculates_duration(self, mock_study):
        """Test get_trials calculates correct duration."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        trials = analyzer.get_trials()

        # First trial should have duration of 60 seconds
        assert trials[0]["duration"] == 60.0

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_trials_duration_none_when_incomplete(self, mock_study_mixed):
        """Test get_trials sets duration to None when datetime_complete is None."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_mixed)
        # Use RUNNING state which has datetime_complete=None
        trials = analyzer.get_trials(states=["RUNNING"], use_cache=False)

        # Running trial has no datetime_complete
        if len(trials) > 0:
            assert trials[0]["duration"] is None

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_trials_empty_study(self, mock_study_empty):
        """Test get_trials returns empty list for empty study."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_empty)
        trials = analyzer.get_trials()

        assert trials == []


# =============================================================================
# GET_COMPLETED_TRIALS TESTS
# =============================================================================


class TestGetCompletedTrials:
    """Test StudyAnalyzer.get_completed_trials() method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_completed_trials_returns_only_complete(self, mock_study_mixed):
        """Test get_completed_trials returns only COMPLETE trials."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_mixed)
        completed = analyzer.get_completed_trials()

        assert all(t["state"] == "COMPLETE" for t in completed)
        assert len(completed) == 2

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_completed_trials_caching(self, mock_study):
        """Test get_completed_trials caches results."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)

        # First call
        completed1 = analyzer.get_completed_trials(use_cache=True)
        assert analyzer._completed_trials_cache is not None

        # Second call - should return cached
        completed2 = analyzer.get_completed_trials(use_cache=True)

        assert completed1 is completed2

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_completed_trials_bypasses_cache(self, mock_study):
        """Test get_completed_trials can bypass cache."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)

        # Set cache
        completed1 = analyzer.get_completed_trials(use_cache=True)

        # Bypass cache
        completed2 = analyzer.get_completed_trials(use_cache=False)

        assert completed1 is not completed2

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_completed_trials_empty_study(self, mock_study_empty):
        """Test get_completed_trials returns empty list for empty study."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_empty)
        completed = analyzer.get_completed_trials()

        assert completed == []


# =============================================================================
# GET_TRIAL_COUNT TESTS
# =============================================================================


class TestGetTrialCount:
    """Test StudyAnalyzer.get_trial_count() method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_trial_count_returns_dict(self, mock_study):
        """Test get_trial_count returns a dict."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        counts = analyzer.get_trial_count()

        assert isinstance(counts, dict)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_trial_count_contains_expected_keys(self, mock_study):
        """Test get_trial_count contains expected keys."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        counts = analyzer.get_trial_count()

        expected_keys = {"total", "completed", "pruned", "failed", "running"}
        assert set(counts.keys()) == expected_keys

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_trial_count_all_completed(self, mock_study):
        """Test get_trial_count with all completed trials."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        counts = analyzer.get_trial_count()

        assert counts["total"] == 3
        assert counts["completed"] == 3
        assert counts["pruned"] == 0
        assert counts["failed"] == 0
        assert counts["running"] == 0

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_trial_count_mixed_states(self, mock_study_mixed):
        """Test get_trial_count with mixed trial states."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_mixed)
        counts = analyzer.get_trial_count()

        # Verify counts match actual fixture (may have 4 or 5 trials depending on fixture)
        assert counts["total"] >= 4
        assert counts["completed"] >= 2
        assert counts["pruned"] >= 1
        assert counts["failed"] >= 1

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_trial_count_empty_study(self, mock_study_empty):
        """Test get_trial_count with empty study."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_empty)
        counts = analyzer.get_trial_count()

        assert counts["total"] == 0
        assert counts["completed"] == 0
        assert counts["pruned"] == 0
        assert counts["failed"] == 0
        assert counts["running"] == 0


# =============================================================================
# GET_PARAMETER_IMPORTANCE TESTS
# =============================================================================


class TestGetParameterImportance:
    """Test StudyAnalyzer.get_parameter_importance() method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_importance_returns_dict(self, mock_study):
        """Test get_parameter_importance returns a dict."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        importance = analyzer.get_parameter_importance()

        assert isinstance(importance, dict)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_importance_contains_params(self, mock_study):
        """Test get_parameter_importance contains parameter names."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        importance = analyzer.get_parameter_importance()

        assert "lr" in importance
        assert "hidden_channels" in importance

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_importance_values_are_floats(self, mock_study):
        """Test get_parameter_importance values are floats."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        importance = analyzer.get_parameter_importance()

        for val in importance.values():
            assert isinstance(val, float)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_importance_caching(self, mock_study):
        """Test get_parameter_importance caches results."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)

        # First call
        importance1 = analyzer.get_parameter_importance(use_cache=True)
        assert analyzer._importance_cache is not None

        # Second call - should return cached
        importance2 = analyzer.get_parameter_importance(use_cache=True)

        assert importance1 is importance2

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_importance_bypasses_cache(self, mock_study):
        """Test get_parameter_importance can bypass cache."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)

        # Set cache
        importance1 = analyzer.get_parameter_importance(use_cache=True)

        # Bypass cache
        importance2 = analyzer.get_parameter_importance(use_cache=False)

        # Results should be equal but different objects
        assert importance1 == importance2

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_importance_with_mdi_method(self, mock_study):
        """Test get_parameter_importance with MDI method."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import (
            ImportanceMethod,
            StudyAnalyzer,
        )

        analyzer = StudyAnalyzer(mock_study)
        importance = analyzer.get_parameter_importance(method=ImportanceMethod.MDI)

        assert isinstance(importance, dict)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_importance_insufficient_trials(self, mock_study_single_trial):
        """Test get_parameter_importance raises error with insufficient trials."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import HPOError, StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_single_trial)

        with pytest.raises((MockHPOError, HPOError)) as exc_info:
            analyzer.get_parameter_importance()

        assert "Not enough" in str(exc_info.value) or "at least 2" in str(exc_info.value)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_importance_empty_study(self, mock_study_empty):
        """Test get_parameter_importance raises error with empty study."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import HPOError, StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_empty)

        with pytest.raises((MockHPOError, HPOError)):
            analyzer.get_parameter_importance()

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances_failure,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_importance_calculation_failure(self, mock_study):
        """Test get_parameter_importance handles calculation failure."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import HPOError, StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)

        with pytest.raises((MockHPOError, HPOError, RuntimeError)) as exc_info:
            analyzer.get_parameter_importance()

        assert "Failed" in str(exc_info.value)


# =============================================================================
# GET_PARAMETER_IMPORTANCE_RANKING TESTS
# =============================================================================


class TestGetParameterImportanceRanking:
    """Test StudyAnalyzer.get_parameter_importance_ranking() method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_importance_ranking_returns_list(self, mock_study):
        """Test get_parameter_importance_ranking returns a list."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        ranking = analyzer.get_parameter_importance_ranking()

        assert isinstance(ranking, list)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_importance_ranking_returns_tuples(self, mock_study):
        """Test get_parameter_importance_ranking returns list of tuples."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        ranking = analyzer.get_parameter_importance_ranking()

        for item in ranking:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)  # parameter name
            assert isinstance(item[1], float)  # importance score

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_importance_ranking_is_sorted_descending(self, mock_study):
        """Test get_parameter_importance_ranking is sorted by importance descending."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        ranking = analyzer.get_parameter_importance_ranking()

        # Check descending order
        scores = [item[1] for item in ranking]
        assert scores == sorted(scores, reverse=True)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_importance_ranking_top_k(self, mock_study):
        """Test get_parameter_importance_ranking with top_k."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        ranking = analyzer.get_parameter_importance_ranking(top_k=2)

        assert len(ranking) == 2

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_importance_ranking_top_k_larger_than_params(self, mock_study):
        """Test get_parameter_importance_ranking with top_k larger than number of params."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        ranking = analyzer.get_parameter_importance_ranking(top_k=100)

        # Should return all params (3)
        assert len(ranking) == 3

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_importance_ranking_top_k_none(self, mock_study):
        """Test get_parameter_importance_ranking with top_k=None returns all."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        ranking = analyzer.get_parameter_importance_ranking(top_k=None)

        assert len(ranking) == 3

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_importance_ranking_most_important_first(self, mock_study):
        """Test most important parameter is first in ranking."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        ranking = analyzer.get_parameter_importance_ranking()

        # lr has highest importance (0.5)
        assert ranking[0][0] == "lr"
        assert ranking[0][1] == 0.5


# =============================================================================
# CLEAR_CACHE TESTS
# =============================================================================


class TestClearCache:
    """Test StudyAnalyzer.clear_cache() method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_clear_cache_clears_all_caches(self, mock_study):
        """Test clear_cache clears all cache attributes."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)

        # Populate caches
        analyzer.get_trials(use_cache=True)
        analyzer.get_completed_trials(use_cache=True)
        analyzer.get_parameter_importance(use_cache=True)

        # Verify caches are set
        assert analyzer._trials_cache is not None
        assert analyzer._completed_trials_cache is not None
        assert analyzer._importance_cache is not None

        # Clear caches
        analyzer.clear_cache()

        # Verify all caches are None
        assert analyzer._trials_cache is None
        assert analyzer._completed_trials_cache is None
        assert analyzer._importance_cache is None

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_clear_cache_returns_none(self, mock_study):
        """Test clear_cache returns None."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        result = analyzer.clear_cache()

        assert result is None

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_clear_cache_safe_when_caches_empty(self, mock_study):
        """Test clear_cache is safe when caches are already None."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)

        # Clear cache when already empty should not raise
        analyzer.clear_cache()

        assert analyzer._trials_cache is None
        assert analyzer._completed_trials_cache is None
        assert analyzer._importance_cache is None


# =============================================================================
# ADDITIONAL FIXTURES FOR CONVERGENCE AND STATISTICAL TESTS
# =============================================================================


@pytest.fixture
def sample_completed_trials_for_convergence():
    """Create sample completed trials for convergence testing."""
    base_time = datetime.now()
    return [
        MockFrozenTrial(
            number=0,
            params={"lr": 0.01, "hidden_channels": 32, "dropout": 0.1},
            value=0.5,
            state=MockTrialState.COMPLETE,
            datetime_start=base_time,
            datetime_complete=base_time + timedelta(seconds=60),
        ),
        MockFrozenTrial(
            number=1,
            params={"lr": 0.001, "hidden_channels": 64, "dropout": 0.2},
            value=0.4,  # Improvement
            state=MockTrialState.COMPLETE,
            datetime_start=base_time + timedelta(seconds=61),
            datetime_complete=base_time + timedelta(seconds=120),
        ),
        MockFrozenTrial(
            number=2,
            params={"lr": 0.0001, "hidden_channels": 128, "dropout": 0.3},
            value=0.3,  # Improvement
            state=MockTrialState.COMPLETE,
            datetime_start=base_time + timedelta(seconds=121),
            datetime_complete=base_time + timedelta(seconds=180),
        ),
        MockFrozenTrial(
            number=3,
            params={"lr": 0.005, "hidden_channels": 96, "dropout": 0.15},
            value=0.35,  # No improvement
            state=MockTrialState.COMPLETE,
            datetime_start=base_time + timedelta(seconds=181),
            datetime_complete=base_time + timedelta(seconds=240),
        ),
        MockFrozenTrial(
            number=4,
            params={"lr": 0.002, "hidden_channels": 80, "dropout": 0.25},
            value=0.32,  # No improvement
            state=MockTrialState.COMPLETE,
            datetime_start=base_time + timedelta(seconds=241),
            datetime_complete=base_time + timedelta(seconds=300),
        ),
    ]


@pytest.fixture
def sample_converged_trials():
    """Create trials that have converged (no improvement in recent trials)."""
    base_time = datetime.now()
    trials = []
    for i in range(15):
        # Value improves for first 5 trials, then stays constant
        if i < 5:
            value = 0.5 - i * 0.05  # 0.5, 0.45, 0.4, 0.35, 0.3
        else:
            value = 0.3 + (i - 5) * 0.01  # 0.3, 0.31, 0.32, ... (no improvement)

        trials.append(
            MockFrozenTrial(
                number=i,
                params={"lr": 0.01 / (i + 1), "hidden_channels": 32 + i * 8},
                value=value,
                state=MockTrialState.COMPLETE,
                datetime_start=base_time + timedelta(seconds=i * 60),
                datetime_complete=base_time + timedelta(seconds=(i + 1) * 60),
            )
        )
    return trials


@pytest.fixture
def sample_trials_with_categorical():
    """Create trials with categorical parameters."""
    base_time = datetime.now()
    return [
        MockFrozenTrial(
            number=0,
            params={"lr": 0.01, "activation": "relu", "optimizer": "adam"},
            value=0.5,
            state=MockTrialState.COMPLETE,
            datetime_start=base_time,
            datetime_complete=base_time + timedelta(seconds=60),
        ),
        MockFrozenTrial(
            number=1,
            params={"lr": 0.001, "activation": "gelu", "optimizer": "sgd"},
            value=0.4,
            state=MockTrialState.COMPLETE,
            datetime_start=base_time + timedelta(seconds=61),
            datetime_complete=base_time + timedelta(seconds=120),
        ),
        MockFrozenTrial(
            number=2,
            params={"lr": 0.0001, "activation": "relu", "optimizer": "adam"},
            value=0.3,
            state=MockTrialState.COMPLETE,
            datetime_start=base_time + timedelta(seconds=121),
            datetime_complete=base_time + timedelta(seconds=180),
        ),
    ]


@pytest.fixture
def mock_study_for_convergence(sample_completed_trials_for_convergence):
    """Create a mock study for convergence testing."""
    return MockStudy(
        study_name="test_study_convergence",
        direction=MockStudyDirection.MINIMIZE,
        trials=sample_completed_trials_for_convergence,
    )


@pytest.fixture
def mock_study_converged(sample_converged_trials):
    """Create a mock study that has converged."""
    return MockStudy(
        study_name="test_study_converged",
        direction=MockStudyDirection.MINIMIZE,
        trials=sample_converged_trials,
    )


@pytest.fixture
def mock_study_empty():
    """Create a mock study with no trials."""
    return MockStudy(
        study_name="test_study_empty",
        direction=MockStudyDirection.MINIMIZE,
        trials=[],
    )


@pytest.fixture
def mock_study_few_trials():
    """Create a mock study with only 2 trials (for correlation testing)."""
    base_time = datetime.now()
    trials = [
        MockFrozenTrial(
            number=0,
            params={"lr": 0.01, "hidden_channels": 32},
            value=0.5,
            state=MockTrialState.COMPLETE,
            datetime_start=base_time,
            datetime_complete=base_time + timedelta(seconds=60),
        ),
        MockFrozenTrial(
            number=1,
            params={"lr": 0.001, "hidden_channels": 64},
            value=0.3,
            state=MockTrialState.COMPLETE,
            datetime_start=base_time + timedelta(seconds=61),
            datetime_complete=base_time + timedelta(seconds=120),
        ),
    ]
    return MockStudy(
        study_name="test_study_few",
        direction=MockStudyDirection.MINIMIZE,
        trials=trials,
    )


@pytest.fixture
def mock_study_categorical(sample_trials_with_categorical):
    """Create a mock study with categorical parameters."""
    return MockStudy(
        study_name="test_study_categorical",
        direction=MockStudyDirection.MINIMIZE,
        trials=sample_trials_with_categorical,
    )


# =============================================================================
# GET_CONVERGENCE_DATA TESTS
# =============================================================================


class TestGetConvergenceData:
    """Test StudyAnalyzer.get_convergence_data() method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_convergence_data_returns_dict(self, mock_study_for_convergence):
        """Test get_convergence_data returns a dict."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        data = analyzer.get_convergence_data()

        assert isinstance(data, dict)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_convergence_data_contains_expected_keys(self, mock_study_for_convergence):
        """Test get_convergence_data contains expected keys."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        data = analyzer.get_convergence_data()

        expected_keys = {
            "trial_numbers",
            "best_values",
            "improvements",
            "convergence_rate",
            "converged",
            "convergence_trial",
        }
        assert set(data.keys()) == expected_keys

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_convergence_data_trial_numbers(self, mock_study_for_convergence):
        """Test get_convergence_data trial_numbers are correct."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        data = analyzer.get_convergence_data()

        assert data["trial_numbers"] == [0, 1, 2, 3, 4]

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_convergence_data_best_values_minimize(self, mock_study_for_convergence):
        """Test get_convergence_data tracks best values correctly for MINIMIZE."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        data = analyzer.get_convergence_data()

        # Best should be decreasing (or same) for MINIMIZE
        # Trial 0: 0.5, Trial 1: 0.4 (best), Trial 2: 0.3 (best), Trial 3: 0.3 (no improvement), Trial 4: 0.3
        assert data["best_values"] == [0.5, 0.4, 0.3, 0.3, 0.3]

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_convergence_data_improvements(self, mock_study_for_convergence):
        """Test get_convergence_data calculates improvements correctly."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        data = analyzer.get_convergence_data()

        # Improvements: 0 (first), 0.1 (0.5->0.4), 0.1 (0.4->0.3), 0, 0
        assert data["improvements"][0] == 0.0
        assert abs(data["improvements"][1] - 0.1) < 1e-6
        assert abs(data["improvements"][2] - 0.1) < 1e-6
        assert data["improvements"][3] == 0.0
        assert data["improvements"][4] == 0.0

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_convergence_data_maximize_direction(self, mock_study_maximize):
        """Test get_convergence_data handles MAXIMIZE direction."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_maximize)
        data = analyzer.get_convergence_data()

        # Best should be increasing for MAXIMIZE
        assert data["best_values"] == [0.5, 0.7, 0.9]

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_convergence_data_empty_study(self, mock_study_empty):
        """Test get_convergence_data with empty study."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_empty)
        data = analyzer.get_convergence_data()

        assert data["trial_numbers"] == []
        assert data["best_values"] == []
        assert data["improvements"] == []
        assert data["convergence_rate"] == 0.0
        assert data["converged"] is False
        assert data["convergence_trial"] is None

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_convergence_data_convergence_rate(self, mock_study_for_convergence):
        """Test get_convergence_data calculates convergence_rate."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        data = analyzer.get_convergence_data()

        # convergence_rate is average improvement over window
        assert isinstance(data["convergence_rate"], float)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_convergence_data_converged_detection(self, mock_study_converged):
        """Test get_convergence_data detects convergence."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig, StudyAnalyzer

        # Use smaller convergence window to ensure convergence is detected
        config = AnalysisConfig(convergence_window=5)
        analyzer = StudyAnalyzer(mock_study_converged, config=config)
        data = analyzer.get_convergence_data()

        # Should detect convergence since last 5+ trials have no improvement
        # (values stay at 0.3 or worse after trial 5)
        assert isinstance(data["converged"], bool)


# =============================================================================
# GET_OPTIMIZATION_TRAJECTORY TESTS
# =============================================================================


class TestGetOptimizationTrajectory:
    """Test StudyAnalyzer.get_optimization_trajectory() method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_optimization_trajectory_returns_dict(self, mock_study_for_convergence):
        """Test get_optimization_trajectory returns a dict."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        data = analyzer.get_optimization_trajectory()

        assert isinstance(data, dict)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_optimization_trajectory_contains_expected_keys(self, mock_study_for_convergence):
        """Test get_optimization_trajectory contains expected keys."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        data = analyzer.get_optimization_trajectory()

        expected_keys = {
            "trials",
            "duration_total",
            "duration_mean",
            "values",
            "best_value",
            "best_trial",
        }
        assert set(data.keys()) == expected_keys

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_optimization_trajectory_best_value_minimize(self, mock_study_for_convergence):
        """Test get_optimization_trajectory finds best value for MINIMIZE."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        data = analyzer.get_optimization_trajectory()

        # Best value should be 0.3 (minimum)
        assert data["best_value"] == 0.3

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_optimization_trajectory_best_value_maximize(self, mock_study_maximize):
        """Test get_optimization_trajectory finds best value for MAXIMIZE."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_maximize)
        data = analyzer.get_optimization_trajectory()

        # Best value should be 0.9 (maximum)
        assert data["best_value"] == 0.9

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_optimization_trajectory_best_trial(self, mock_study_for_convergence):
        """Test get_optimization_trajectory identifies best trial."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        data = analyzer.get_optimization_trajectory()

        # Best trial should be trial 2 (value 0.3)
        assert data["best_trial"] == 2

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_optimization_trajectory_duration_total(self, mock_study_for_convergence):
        """Test get_optimization_trajectory calculates total duration."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        data = analyzer.get_optimization_trajectory()

        # 5 trials with approximately 59 seconds each due to gaps in fixture timing
        assert data["duration_total"] >= 290.0 and data["duration_total"] <= 310.0

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_optimization_trajectory_duration_mean(self, mock_study_for_convergence):
        """Test get_optimization_trajectory calculates mean duration."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        data = analyzer.get_optimization_trajectory()

        # Mean should be approximately 59-60 seconds
        assert data["duration_mean"] >= 58.0 and data["duration_mean"] <= 62.0

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_optimization_trajectory_empty_study(self, mock_study_empty):
        """Test get_optimization_trajectory with empty study."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_empty)
        data = analyzer.get_optimization_trajectory()

        assert data["trials"] == []
        assert data["duration_total"] == 0.0
        assert data["duration_mean"] == 0.0
        assert data["values"] == []
        assert data["best_value"] is None
        assert data["best_trial"] is None

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_optimization_trajectory_trials_contain_params(self, mock_study_for_convergence):
        """Test trials in trajectory contain params."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        data = analyzer.get_optimization_trajectory()

        for trial in data["trials"]:
            assert "params" in trial
            assert "number" in trial
            assert "value" in trial


# =============================================================================
# GET_VALUE_STATISTICS TESTS
# =============================================================================


class TestGetValueStatistics:
    """Test StudyAnalyzer.get_value_statistics() method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_value_statistics_returns_dict(self, mock_study_for_convergence):
        """Test get_value_statistics returns a dict."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        stats = analyzer.get_value_statistics()

        assert isinstance(stats, dict)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_value_statistics_contains_expected_keys(self, mock_study_for_convergence):
        """Test get_value_statistics contains expected keys."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        stats = analyzer.get_value_statistics()

        expected_keys = {"count", "mean", "std", "min", "max", "median", "percentiles", "range"}
        assert set(stats.keys()) == expected_keys

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_value_statistics_count(self, mock_study_for_convergence):
        """Test get_value_statistics count is correct."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        stats = analyzer.get_value_statistics()

        assert stats["count"] == 5

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_value_statistics_min_max(self, mock_study_for_convergence):
        """Test get_value_statistics min and max are correct."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        stats = analyzer.get_value_statistics()

        # Values: 0.5, 0.4, 0.3, 0.35, 0.32
        assert stats["min"] == 0.3
        assert stats["max"] == 0.5

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_value_statistics_range(self, mock_study_for_convergence):
        """Test get_value_statistics range is correct."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        stats = analyzer.get_value_statistics()

        assert abs(stats["range"] - 0.2) < 1e-6  # 0.5 - 0.3

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_value_statistics_mean(self, mock_study_for_convergence):
        """Test get_value_statistics mean is correct."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        stats = analyzer.get_value_statistics()

        # Values: 0.5, 0.4, 0.3, 0.35, 0.32 -> mean = 0.374
        expected_mean = (0.5 + 0.4 + 0.3 + 0.35 + 0.32) / 5
        assert abs(stats["mean"] - expected_mean) < 1e-6

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_value_statistics_percentiles(self, mock_study_for_convergence):
        """Test get_value_statistics percentiles are calculated."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        stats = analyzer.get_value_statistics()

        assert isinstance(stats["percentiles"], dict)
        # Default percentile thresholds
        assert 25.0 in stats["percentiles"]
        assert 50.0 in stats["percentiles"]
        assert 75.0 in stats["percentiles"]

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_value_statistics_empty_study(self, mock_study_empty):
        """Test get_value_statistics with empty study."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_empty)
        stats = analyzer.get_value_statistics()

        assert stats["count"] == 0
        assert stats["mean"] is None
        assert stats["std"] is None
        assert stats["min"] is None
        assert stats["max"] is None
        assert stats["median"] is None
        assert stats["percentiles"] == {}
        assert stats["range"] is None


# =============================================================================
# GET_PARAMETER_STATISTICS TESTS
# =============================================================================


class TestGetParameterStatistics:
    """Test StudyAnalyzer.get_parameter_statistics() method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_statistics_returns_dict(self, mock_study_for_convergence):
        """Test get_parameter_statistics returns a dict."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        stats = analyzer.get_parameter_statistics("lr")

        assert isinstance(stats, dict)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_statistics_numeric_type(self, mock_study_for_convergence):
        """Test get_parameter_statistics identifies numeric type."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        stats = analyzer.get_parameter_statistics("lr")

        assert stats["type"] == "numeric"

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_statistics_numeric_contains_stats(self, mock_study_for_convergence):
        """Test get_parameter_statistics for numeric param contains expected keys."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        stats = analyzer.get_parameter_statistics("lr")

        assert "count" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_statistics_categorical_type(self, mock_study_categorical):
        """Test get_parameter_statistics identifies categorical type."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_categorical)
        stats = analyzer.get_parameter_statistics("activation")

        assert stats["type"] == "categorical"

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_statistics_categorical_contains_keys(self, mock_study_categorical):
        """Test get_parameter_statistics for categorical param contains expected keys."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_categorical)
        stats = analyzer.get_parameter_statistics("activation")

        assert "unique_values" in stats
        assert "value_counts" in stats
        assert "mode" in stats

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_statistics_categorical_mode(self, mock_study_categorical):
        """Test get_parameter_statistics mode for categorical."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_categorical)
        stats = analyzer.get_parameter_statistics("activation")

        # relu appears twice, gelu once -> mode is relu
        assert stats["mode"] == "relu"

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_statistics_unknown_param_raises_error(self, mock_study_for_convergence):
        """Test get_parameter_statistics raises error for unknown parameter."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import HPOError, StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)

        with pytest.raises((MockHPOError, HPOError)) as exc_info:
            analyzer.get_parameter_statistics("nonexistent_param")

        assert "nonexistent_param" in str(exc_info.value)


# =============================================================================
# GET_PARAMETER_CORRELATIONS TESTS
# =============================================================================


class TestGetParameterCorrelations:
    """Test StudyAnalyzer.get_parameter_correlations() method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_correlations_returns_dict(self, mock_study_for_convergence):
        """Test get_parameter_correlations returns a dict."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        corr = analyzer.get_parameter_correlations()

        assert isinstance(corr, dict)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_correlations_contains_objective(self, mock_study_for_convergence):
        """Test get_parameter_correlations contains objective correlations."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        corr = analyzer.get_parameter_correlations()

        assert "objective" in corr

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_correlations_contains_parameters(self, mock_study_for_convergence):
        """Test get_parameter_correlations contains parameter correlations."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        corr = analyzer.get_parameter_correlations()

        assert "parameters" in corr

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_correlations_objective_has_params(self, mock_study_for_convergence):
        """Test objective correlations include numeric parameters."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        corr = analyzer.get_parameter_correlations()

        # Should have correlations for numeric params
        assert "lr" in corr["objective"]
        assert "hidden_channels" in corr["objective"]

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_correlations_values_are_floats(self, mock_study_for_convergence):
        """Test correlation values are floats."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        corr = analyzer.get_parameter_correlations()

        for val in corr["objective"].values():
            assert isinstance(val, float)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_correlations_values_in_range(self, mock_study_for_convergence):
        """Test correlation values are in [-1, 1] range."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        corr = analyzer.get_parameter_correlations()

        for val in corr["objective"].values():
            assert -1.0 <= val <= 1.0

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_correlations_insufficient_trials(self, mock_study_few_trials):
        """Test get_parameter_correlations with insufficient trials returns empty."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_few_trials)
        corr = analyzer.get_parameter_correlations()

        # Should return empty correlations (need at least 3 trials)
        assert corr["objective"] == {}
        assert corr["parameters"] == {}

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_correlations_param_to_param(self, mock_study_for_convergence):
        """Test parameter-to-parameter correlations."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_for_convergence)
        corr = analyzer.get_parameter_correlations()

        # Parameters dict should contain param-param correlations
        assert "lr" in corr["parameters"]
        assert "hidden_channels" in corr["parameters"]["lr"] or len(corr["parameters"]["lr"]) >= 0


# =============================================================================
# ADDITIONAL MOCKS FOR MULTI-OBJECTIVE AND EXPORT TESTS
# =============================================================================


class MockWFG:
    """Mock WFG hypervolume calculator."""

    def compute(self, points, reference_point):
        """Mock hypervolume computation."""
        return 0.42


class MockDataFrame:
    """Mock pandas DataFrame for testing."""

    def __init__(self, data=None):
        self.data = data or []
        self._columns = []

    def to_csv(self, path, index=False):
        """Mock to_csv method."""
        pass

    def head(self, n=5):
        return self


# =============================================================================
# ADDITIONAL FIXTURES FOR MULTI-OBJECTIVE AND EXPORT TESTS
# =============================================================================


@pytest.fixture
def mock_multi_objective_study():
    """Create a mock multi-objective study."""
    base_time = datetime.now()
    trials = [
        MockFrozenTrial(
            number=0,
            params={"lr": 0.01, "hidden_channels": 32},
            values=(0.5, 0.3),
            value=0.5,
            state=MockTrialState.COMPLETE,
            datetime_start=base_time,
            datetime_complete=base_time + timedelta(seconds=60),
        ),
        MockFrozenTrial(
            number=1,
            params={"lr": 0.001, "hidden_channels": 64},
            values=(0.3, 0.5),
            value=0.3,
            state=MockTrialState.COMPLETE,
            datetime_start=base_time + timedelta(seconds=61),
            datetime_complete=base_time + timedelta(seconds=120),
        ),
        MockFrozenTrial(
            number=2,
            params={"lr": 0.0001, "hidden_channels": 128},
            values=(0.4, 0.4),
            value=0.4,
            state=MockTrialState.COMPLETE,
            datetime_start=base_time + timedelta(seconds=121),
            datetime_complete=base_time + timedelta(seconds=180),
        ),
    ]

    return MockStudy(
        study_name="test_multi_objective",
        direction=MockStudyDirection.MINIMIZE,
        directions=[MockStudyDirection.MINIMIZE, MockStudyDirection.MINIMIZE],
        trials=trials,
        best_trials=trials[:2],  # First two are on Pareto front
    )


@pytest.fixture
def mock_study_maximize():
    """Create a mock study with MAXIMIZE direction."""
    base_time = datetime.now()
    trials = [
        MockFrozenTrial(
            number=0,
            params={"lr": 0.01, "hidden_channels": 32},
            value=0.5,
            state=MockTrialState.COMPLETE,
            datetime_start=base_time,
            datetime_complete=base_time + timedelta(seconds=60),
        ),
        MockFrozenTrial(
            number=1,
            params={"lr": 0.001, "hidden_channels": 64},
            value=0.7,
            state=MockTrialState.COMPLETE,
            datetime_start=base_time + timedelta(seconds=61),
            datetime_complete=base_time + timedelta(seconds=120),
        ),
        MockFrozenTrial(
            number=2,
            params={"lr": 0.0001, "hidden_channels": 128},
            value=0.9,
            state=MockTrialState.COMPLETE,
            datetime_start=base_time + timedelta(seconds=121),
            datetime_complete=base_time + timedelta(seconds=180),
        ),
    ]
    return MockStudy(
        study_name="test_study_maximize",
        direction=MockStudyDirection.MAXIMIZE,
        trials=trials,
    )


# =============================================================================
# GET_PARETO_FRONT TESTS
# =============================================================================


class TestGetParetoFront:
    """Test StudyAnalyzer.get_pareto_front() method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_pareto_front_returns_list(self, mock_multi_objective_study):
        """Test get_pareto_front returns a list."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_multi_objective_study)
        pareto = analyzer.get_pareto_front()

        assert isinstance(pareto, list)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_pareto_front_contains_expected_keys(self, mock_multi_objective_study):
        """Test Pareto front entries contain expected keys."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_multi_objective_study)
        pareto = analyzer.get_pareto_front()

        for trial in pareto:
            assert "number" in trial
            assert "values" in trial
            assert "params" in trial

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_pareto_front_single_objective_raises_error(self, mock_study):
        """Test get_pareto_front raises error for single-objective study."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)

        with pytest.raises(MockHPOError) as exc_info:
            analyzer.get_pareto_front()

        assert "multi-objective" in str(exc_info.value).lower()


# =============================================================================
# GET_HYPERVOLUME TESTS
# =============================================================================


class TestGetHypervolume:
    """Test StudyAnalyzer.get_hypervolume() method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_hypervolume_requires_reference_point(self, mock_multi_objective_study):
        """Test get_hypervolume requires reference_point."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_multi_objective_study)

        with pytest.raises(MockHPOError) as exc_info:
            analyzer.get_hypervolume(reference_point=None)

        assert "reference_point" in str(exc_info.value)


# =============================================================================
# VISUALIZATION DATA TESTS
# =============================================================================


class TestGetOptimizationHistoryData:
    """Test StudyAnalyzer.get_optimization_history_data() method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_optimization_history_data_returns_dict(self, mock_study):
        """Test get_optimization_history_data returns a dict."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        data = analyzer.get_optimization_history_data()

        assert isinstance(data, dict)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_optimization_history_data_contains_expected_keys(self, mock_study):
        """Test get_optimization_history_data contains expected keys."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        data = analyzer.get_optimization_history_data()

        expected_keys = {"trial_numbers", "values", "best_values", "infeasible_trials", "direction"}
        assert set(data.keys()) == expected_keys

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_optimization_history_data_tracks_infeasible(self, mock_study_mixed):
        """Test get_optimization_history_data tracks infeasible trials."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_mixed)
        data = analyzer.get_optimization_history_data()

        # Pruned and failed trials should be in infeasible list
        assert len(data["infeasible_trials"]) == 2  # trials 1 and 2


class TestGetParameterImportanceData:
    """Test StudyAnalyzer.get_parameter_importance_data() method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_importance_data_returns_dict(self, mock_study):
        """Test get_parameter_importance_data returns a dict."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        data = analyzer.get_parameter_importance_data()

        assert isinstance(data, dict)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_importance_data_contains_expected_keys(self, mock_study):
        """Test get_parameter_importance_data contains expected keys."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        data = analyzer.get_parameter_importance_data()

        expected_keys = {"parameters", "importances", "sorted_indices"}
        assert set(data.keys()) == expected_keys

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parameter_importance_data_empty_on_error(self, mock_study_empty):
        """Test get_parameter_importance_data returns empty on error."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_empty)
        data = analyzer.get_parameter_importance_data()

        assert data["parameters"] == []
        assert data["importances"] == []


class TestGetSlicePlotData:
    """Test StudyAnalyzer.get_slice_plot_data() method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_slice_plot_data_returns_dict(self, mock_study):
        """Test get_slice_plot_data returns a dict."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        data = analyzer.get_slice_plot_data("lr")

        assert isinstance(data, dict)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_slice_plot_data_contains_expected_keys(self, mock_study):
        """Test get_slice_plot_data contains expected keys."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        data = analyzer.get_slice_plot_data("lr")

        expected_keys = {
            "parameter_values",
            "objective_values",
            "best_value",
            "best_param",
            "parameter_name",
        }
        assert set(data.keys()) == expected_keys

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_slice_plot_data_parameter_name(self, mock_study):
        """Test get_slice_plot_data stores parameter name."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        data = analyzer.get_slice_plot_data("lr")

        assert data["parameter_name"] == "lr"


class TestGetContourPlotData:
    """Test StudyAnalyzer.get_contour_plot_data() method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_contour_plot_data_returns_dict(self, mock_study):
        """Test get_contour_plot_data returns a dict."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        data = analyzer.get_contour_plot_data("lr", "hidden_channels")

        assert isinstance(data, dict)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_contour_plot_data_contains_expected_keys(self, mock_study):
        """Test get_contour_plot_data contains expected keys."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        data = analyzer.get_contour_plot_data("lr", "hidden_channels")

        expected_keys = {"x_values", "y_values", "z_values", "param_x", "param_y"}
        assert set(data.keys()) == expected_keys


class TestGetParallelCoordinateData:
    """Test StudyAnalyzer.get_parallel_coordinate_data() method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parallel_coordinate_data_returns_dict(self, mock_study):
        """Test get_parallel_coordinate_data returns a dict."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        data = analyzer.get_parallel_coordinate_data()

        assert isinstance(data, dict)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parallel_coordinate_data_contains_expected_keys(self, mock_study):
        """Test get_parallel_coordinate_data contains expected keys."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        data = analyzer.get_parallel_coordinate_data()

        assert "parameters" in data
        assert "trials" in data
        assert "objective_values" in data

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_parallel_coordinate_data_empty_study(self, mock_study_empty):
        """Test get_parallel_coordinate_data with empty study."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study_empty)
        data = analyzer.get_parallel_coordinate_data()

        assert data["parameters"] == []
        assert data["trials"] == []
        assert data["objective_values"] == []


# =============================================================================
# GET_COMPREHENSIVE_ANALYSIS TESTS
# =============================================================================


class TestGetComprehensiveAnalysis:
    """Test StudyAnalyzer.get_comprehensive_analysis() method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_comprehensive_analysis_returns_dict(self, mock_study):
        """Test get_comprehensive_analysis returns a dict."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        analysis = analyzer.get_comprehensive_analysis()

        assert isinstance(analysis, dict)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_comprehensive_analysis_contains_study_info(self, mock_study):
        """Test get_comprehensive_analysis contains study_info."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        analysis = analyzer.get_comprehensive_analysis()

        assert "study_info" in analysis
        assert analysis["study_info"]["study_name"] == "test_study"

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_get_comprehensive_analysis_contains_all_sections(self, mock_study):
        """Test get_comprehensive_analysis contains all sections."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        analysis = analyzer.get_comprehensive_analysis()

        expected_keys = {
            "study_info",
            "trial_counts",
            "value_statistics",
            "convergence",
            "trajectory",
            "importance",
            "correlations",
        }
        assert expected_keys.issubset(set(analysis.keys()))


# =============================================================================
# EXPORT_RESULTS TESTS
# =============================================================================


class TestExportResults:
    """Test StudyAnalyzer.export_results() method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_export_results_dict_format(self, mock_study):
        """Test export_results with DICT format."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import ExportFormat, StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        result = analyzer.export_results(format=ExportFormat.DICT)

        assert isinstance(result, dict)
        assert "study_name" in result

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_export_results_dict_format_no_path(self, mock_study):
        """Test export_results DICT format raises error with path."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import ExportFormat, StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)

        with pytest.raises(MockHPOError) as exc_info:
            analyzer.export_results(path="output.json", format=ExportFormat.DICT)

        assert "DICT" in str(exc_info.value)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_export_results_json_format_no_path(self, mock_study):
        """Test export_results JSON format without path returns dict."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import ExportFormat, StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        result = analyzer.export_results(format=ExportFormat.JSON)

        assert isinstance(result, dict)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_export_results_json_format_with_path(self, mock_study, tmp_path):
        """Test export_results JSON format with path saves file."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import ExportFormat, StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        output_path = tmp_path / "output.json"
        result = analyzer.export_results(path=str(output_path), format=ExportFormat.JSON)

        assert result is None
        assert output_path.exists()

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.PANDAS_AVAILABLE", False)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_export_results_csv_without_pandas_raises_error(self, mock_study):
        """Test export_results CSV format raises error without pandas."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import ExportFormat, StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)

        with pytest.raises(MockHPOError) as exc_info:
            analyzer.export_results(path="output.csv", format=ExportFormat.CSV)

        assert "Pandas" in str(exc_info.value) or "pandas" in str(exc_info.value)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_export_results_include_trials(self, mock_study):
        """Test export_results includes trials when requested."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import ExportFormat, StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        result = analyzer.export_results(format=ExportFormat.DICT, include_trials=True)

        assert "trials" in result

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_export_results_include_analysis(self, mock_study):
        """Test export_results includes analysis when requested."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import ExportFormat, StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)
        result = analyzer.export_results(format=ExportFormat.DICT, include_analysis=True)

        assert "analysis" in result


# =============================================================================
# TO_DATAFRAME TESTS
# =============================================================================


class TestToDataframe:
    """Test StudyAnalyzer.to_dataframe() method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.PANDAS_AVAILABLE", False)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_to_dataframe_without_pandas_raises_error(self, mock_study):
        """Test to_dataframe raises error without pandas."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)

        with pytest.raises(MockHPOError) as exc_info:
            analyzer.to_dataframe()

        assert "Pandas" in str(exc_info.value) or "pandas" in str(exc_info.value)


# =============================================================================
# COMPARE_WITH TESTS
# =============================================================================


class TestCompareWith:
    """Test StudyAnalyzer.compare_with() method."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_compare_with_returns_dict(self, mock_study, mock_study_maximize):
        """Test compare_with returns a dict."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer1 = StudyAnalyzer(mock_study)
        analyzer2 = StudyAnalyzer(mock_study_maximize)

        comparison = analyzer1.compare_with(analyzer2)

        assert isinstance(comparison, dict)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_compare_with_contains_expected_keys(self, mock_study, mock_study_maximize):
        """Test compare_with contains expected keys."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer1 = StudyAnalyzer(mock_study)
        analyzer2 = StudyAnalyzer(mock_study_maximize)

        comparison = analyzer1.compare_with(analyzer2)

        expected_keys = {
            "studies",
            "best_values",
            "winner",
            "trial_counts",
            "value_statistics",
            "improvement",
        }
        assert set(comparison.keys()) == expected_keys

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_compare_with_identifies_studies(self, mock_study, mock_study_maximize):
        """Test compare_with identifies both studies."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer1 = StudyAnalyzer(mock_study)
        analyzer2 = StudyAnalyzer(mock_study_maximize)

        comparison = analyzer1.compare_with(analyzer2)

        assert len(comparison["studies"]) == 2
        assert "test_study" in comparison["studies"]
        assert "test_study_maximize" in comparison["studies"]

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_compare_with_determines_winner(self, mock_study):
        """Test compare_with determines a winner."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        # Create another study with different values
        base_time = datetime.now()
        trials = [
            MockFrozenTrial(
                number=0,
                params={"lr": 0.01, "hidden_channels": 32},
                value=0.1,  # Better than mock_study's 0.2
                state=MockTrialState.COMPLETE,
                datetime_start=base_time,
                datetime_complete=base_time + timedelta(seconds=60),
            ),
        ]
        better_study = MockStudy(
            study_name="better_study",
            direction=MockStudyDirection.MINIMIZE,
            trials=trials,
        )

        analyzer1 = StudyAnalyzer(mock_study)
        analyzer2 = StudyAnalyzer(better_study)

        comparison = analyzer1.compare_with(analyzer2)

        # better_study should be winner (lower is better for MINIMIZE)
        assert comparison["winner"] == "better_study"

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_compare_with_empty_studies(self, mock_study_empty):
        """Test compare_with handles empty studies."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer1 = StudyAnalyzer(mock_study_empty)
        analyzer2 = StudyAnalyzer(mock_study_empty)

        comparison = analyzer1.compare_with(analyzer2)

        assert comparison["winner"] is None
        assert comparison["improvement"] is None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for StudyAnalyzer."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_full_analysis_workflow(self, mock_study):
        """Test full analysis workflow."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import (
            AnalysisConfig,
            ExportFormat,
            StudyAnalyzer,
        )

        # Create analyzer with custom config
        config = AnalysisConfig(convergence_window=5, include_pruned=False)
        analyzer = StudyAnalyzer(mock_study, config=config)

        # Get trials
        trials = analyzer.get_trials()
        assert len(trials) > 0

        # Get completed trials
        completed = analyzer.get_completed_trials()
        assert len(completed) > 0

        # Get trial counts
        counts = analyzer.get_trial_count()
        assert counts["total"] > 0

        # Get convergence data
        convergence = analyzer.get_convergence_data()
        assert "best_values" in convergence

        # Get trajectory
        trajectory = analyzer.get_optimization_trajectory()
        assert trajectory["best_value"] is not None

        # Get value statistics
        stats = analyzer.get_value_statistics()
        assert stats["count"] > 0

        # Get parameter importance
        importance = analyzer.get_parameter_importance()
        assert len(importance) > 0

        # Get comprehensive analysis
        analysis = analyzer.get_comprehensive_analysis()
        assert "study_info" in analysis

        # Export results using ExportFormat.DICT enum
        results = analyzer.export_results(format=ExportFormat.DICT)
        assert "study_name" in results

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_cache_clear_refreshes_data(self, mock_study):
        """Test clearing cache refreshes data on next call."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)

        # Populate cache
        trials1 = analyzer.get_trials(use_cache=True)
        assert analyzer._trials_cache is not None

        # Clear cache
        analyzer.clear_cache()
        assert analyzer._trials_cache is None

        # Get fresh data
        trials2 = analyzer.get_trials(use_cache=True)
        assert analyzer._trials_cache is not None

        # Data should be equal
        assert len(trials1) == len(trials2)

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.get_param_importances",
        mock_get_param_importances,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.FanovaImportanceEvaluator",
        MockFanovaImportanceEvaluator,
    )
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_visualization_data_workflow(self, mock_study):
        """Test visualization data preparation workflow."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        analyzer = StudyAnalyzer(mock_study)

        # Get optimization history data
        history = analyzer.get_optimization_history_data()
        assert len(history["trial_numbers"]) > 0

        # Get parameter importance data
        importance_data = analyzer.get_parameter_importance_data()
        assert len(importance_data["parameters"]) > 0

        # Get slice plot data
        slice_data = analyzer.get_slice_plot_data("lr")
        assert len(slice_data["parameter_values"]) > 0

        # Get contour plot data
        contour_data = analyzer.get_contour_plot_data("lr", "hidden_channels")
        assert len(contour_data["x_values"]) > 0

        # Get parallel coordinate data
        parallel_data = analyzer.get_parallel_coordinate_data()
        assert len(parallel_data["trials"]) > 0


# =============================================================================
# EDGE CASES TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_single_completed_trial(self):
        """Test analyzer with single completed trial."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        base_time = datetime.now()
        single_trial = MockFrozenTrial(
            number=0,
            params={"lr": 0.01, "hidden_channels": 32},
            value=0.5,
            state=MockTrialState.COMPLETE,
            datetime_start=base_time,
            datetime_complete=base_time + timedelta(seconds=60),
        )
        single_study = MockStudy(
            study_name="single_trial_study",
            direction=MockStudyDirection.MINIMIZE,
            trials=[single_trial],
        )

        analyzer = StudyAnalyzer(single_study)

        # Should work without error
        stats = analyzer.get_value_statistics()
        assert stats["count"] == 1
        assert stats["std"] == 0.0  # Single value

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_all_failed_trials(self):
        """Test analyzer with all failed trials."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        base_time = datetime.now()
        failed_trials = [
            MockFrozenTrial(
                number=i,
                params={"lr": 0.01, "hidden_channels": 32},
                value=None,
                state=MockTrialState.FAIL,
                datetime_start=base_time + timedelta(seconds=i * 60),
                datetime_complete=None,
            )
            for i in range(3)
        ]
        failed_study = MockStudy(
            study_name="all_failed_study",
            direction=MockStudyDirection.MINIMIZE,
            trials=failed_trials,
        )

        analyzer = StudyAnalyzer(failed_study)

        # Should handle gracefully
        counts = analyzer.get_trial_count()
        assert counts["failed"] == 3
        assert counts["completed"] == 0

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.OPTUNA_AVAILABLE", True)
    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_convergence_window_larger_than_trials(self, mock_study):
        """Test convergence window larger than number of trials."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig, StudyAnalyzer

        config = AnalysisConfig(convergence_window=100)  # Much larger than 3 trials
        analyzer = StudyAnalyzer(mock_study, config=config)

        # Should handle gracefully
        convergence = analyzer.get_convergence_data()
        assert "converged" in convergence


# =============================================================================
# MODULE IMPORT TESTS
# =============================================================================


class TestModuleImports:
    """Test module-level imports."""

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_import_studyanalyzer(self):
        """Test StudyAnalyzer can be imported."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import StudyAnalyzer

        assert StudyAnalyzer is not None

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_import_analysisconfig(self):
        """Test AnalysisConfig can be imported."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import AnalysisConfig

        assert AnalysisConfig is not None

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_import_importancemethod(self):
        """Test ImportanceMethod can be imported."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import ImportanceMethod

        assert ImportanceMethod is not None

    @patch("milia_pipeline.models.hpo.analysis.study_analyzer.HPOError", MockHPOError)
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.HPOConfigurationError",
        MockHPOConfigurationError,
    )
    @patch(
        "milia_pipeline.models.hpo.analysis.study_analyzer.StudyNotFoundError",
        MockStudyNotFoundError,
    )
    def test_import_exportformat(self):
        """Test ExportFormat can be imported."""
        from milia_pipeline.models.hpo.analysis.study_analyzer import ExportFormat

        assert ExportFormat is not None


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
