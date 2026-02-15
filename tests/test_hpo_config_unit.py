#!/usr/bin/env python3
"""
Complete Unit Test Suite for hpo_config.py Module

Tests HPO configuration system including:
- ParamType enum validation
- SearchSpaceParamConfig (Pydantic V2 frozen BaseModel)
  - Type-based validation (INT, FLOAT, CATEGORICAL, LOGUNIFORM, etc.)
  - Low/high bounds validation
  - Choices validation for categorical
  - Step validation for discrete uniform
- PrunerType enum validation
- PrunerConfig (Pydantic V2 frozen BaseModel)
  - Default values
  - Non-negative validation for startup trials, warmup steps
  - Positive validation for interval steps
  - Percentile range validation
  - Hyperband n_brackets validation
- SamplerType enum validation
- SamplerConfig (Pydantic V2 frozen BaseModel)
  - Default values
  - Non-negative validation for startup trials
  - Seed validation
- OptimizationDirection enum validation
- StudyConfig (Pydantic V2 frozen BaseModel)
  - Default values
  - Empty metric validation
  - Empty study_name validation
  - Storage type validation
  - is_multi_objective property
- MultiObjectiveStudyConfig (Pydantic V2 frozen BaseModel)
  - Directions/metrics length matching
  - Valid directions validation
  - Minimum 2 metrics validation
  - Empty metrics validation
  - Reference point validation
  - is_multi_objective property
- HPOConfig (Pydantic V2 frozen BaseModel)
  - Default values
  - Backend validation
  - n_trials validation
  - timeout validation
  - n_jobs validation
  - cv_folds validation
  - cv_metric_aggregation validation
  - from_dict() factory method
- Edge cases and error conditions
- Frozen model immutability (Pydantic V2 ValidationError)

Note: This module was migrated from Python dataclasses to Pydantic V2 BaseModel
with frozen=True. Immutability violations now raise pydantic.ValidationError
instead of dataclasses.FrozenInstanceError.

This is a PRODUCTION-READY test suite with comprehensive coverage.

Author: Milia Team
Version: 2.0.0
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))


import pytest
from pydantic import ValidationError

# Import the module under test
from milia_pipeline.models.hpo.hpo_config import (
    HPOConfig,
    MultiObjectiveStudyConfig,
    OptimizationDirection,
    # Enums
    ParamType,
    PrunerConfig,
    PrunerType,
    SamplerConfig,
    SamplerType,
    # Config classes (Pydantic V2 BaseModel with frozen=True)
    SearchSpaceParamConfig,
    StudyConfig,
)

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def valid_int_param_config():
    """Create a valid integer parameter config."""
    return SearchSpaceParamConfig(type=ParamType.INT, low=32, high=256, step=32)


@pytest.fixture
def valid_float_param_config():
    """Create a valid float parameter config."""
    return SearchSpaceParamConfig(type=ParamType.FLOAT, low=0.0, high=1.0)


@pytest.fixture
def valid_categorical_param_config():
    """Create a valid categorical parameter config."""
    return SearchSpaceParamConfig(type=ParamType.CATEGORICAL, choices=["relu", "gelu", "silu"])


@pytest.fixture
def valid_loguniform_param_config():
    """Create a valid loguniform parameter config."""
    return SearchSpaceParamConfig(type=ParamType.LOGUNIFORM, low=1e-5, high=1e-2)


@pytest.fixture
def valid_pruner_config():
    """Create a valid pruner config with defaults."""
    return PrunerConfig()


@pytest.fixture
def valid_sampler_config():
    """Create a valid sampler config with defaults."""
    return SamplerConfig()


@pytest.fixture
def valid_study_config():
    """Create a valid study config with defaults."""
    return StudyConfig()


@pytest.fixture
def valid_multi_objective_config():
    """Create a valid multi-objective study config."""
    return MultiObjectiveStudyConfig(
        directions=("minimize", "maximize"), metrics=("val_loss", "val_accuracy")
    )


@pytest.fixture
def valid_hpo_config():
    """Create a valid HPO config with defaults."""
    return HPOConfig()


@pytest.fixture
def sample_config_dict():
    """Create a sample config dictionary for from_dict testing."""
    return {
        "enabled": True,
        "n_trials": 100,
        "backend": "optuna",
        "timeout": 3600,
        "n_jobs": 4,
        "cv_folds": 5,
        "cv_metric_aggregation": "mean",
        "pruner": {"type": "median", "n_startup_trials": 10, "n_warmup_steps": 5},
        "sampler": {"type": "tpe", "n_startup_trials": 15, "seed": 42},
        "study": {"direction": "minimize", "metric": "val_loss", "study_name": "test_study"},
        "search_space": {
            "model": {
                "hidden_channels": {"type": "int", "low": 32, "high": 256},
                "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            },
            "optimizer": {"learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-2}},
        },
    }


# =============================================================================
# PARAMTYPE ENUM TESTS
# =============================================================================


class TestParamTypeEnum:
    """Test ParamType enum values and behavior."""

    def test_param_type_int_value(self):
        """Test ParamType.INT has correct value."""
        assert ParamType.INT.value == "int"

    def test_param_type_float_value(self):
        """Test ParamType.FLOAT has correct value."""
        assert ParamType.FLOAT.value == "float"

    def test_param_type_categorical_value(self):
        """Test ParamType.CATEGORICAL has correct value."""
        assert ParamType.CATEGORICAL.value == "categorical"

    def test_param_type_loguniform_value(self):
        """Test ParamType.LOGUNIFORM has correct value."""
        assert ParamType.LOGUNIFORM.value == "loguniform"

    def test_param_type_uniform_value(self):
        """Test ParamType.UNIFORM has correct value."""
        assert ParamType.UNIFORM.value == "uniform"

    def test_param_type_int_uniform_value(self):
        """Test ParamType.INT_UNIFORM has correct value."""
        assert ParamType.INT_UNIFORM.value == "int_uniform"

    def test_param_type_discrete_uniform_value(self):
        """Test ParamType.DISCRETE_UNIFORM has correct value."""
        assert ParamType.DISCRETE_UNIFORM.value == "discrete_uniform"

    def test_param_type_from_string(self):
        """Test creating ParamType from string value."""
        assert ParamType("int") == ParamType.INT
        assert ParamType("float") == ParamType.FLOAT
        assert ParamType("categorical") == ParamType.CATEGORICAL

    def test_param_type_invalid_string_raises_error(self):
        """Test invalid string raises ValueError."""
        with pytest.raises(ValueError):
            ParamType("invalid_type")

    def test_all_param_types_defined(self):
        """Test all expected param types are defined."""
        expected_types = {
            "INT",
            "FLOAT",
            "CATEGORICAL",
            "LOGUNIFORM",
            "UNIFORM",
            "INT_UNIFORM",
            "DISCRETE_UNIFORM",
        }
        actual_types = {pt.name for pt in ParamType}
        assert actual_types == expected_types


# =============================================================================
# SEARCHSPACEPARAMCONFIG TESTS
# =============================================================================


class TestSearchSpaceParamConfigValidCreation:
    """Test valid SearchSpaceParamConfig creation."""

    def test_valid_int_param_creation(self, valid_int_param_config):
        """Test creating valid INT parameter config."""
        assert valid_int_param_config.type == ParamType.INT
        assert valid_int_param_config.low == 32
        assert valid_int_param_config.high == 256
        assert valid_int_param_config.step == 32

    def test_valid_float_param_creation(self, valid_float_param_config):
        """Test creating valid FLOAT parameter config."""
        assert valid_float_param_config.type == ParamType.FLOAT
        assert valid_float_param_config.low == 0.0
        assert valid_float_param_config.high == 1.0

    def test_valid_categorical_param_creation(self, valid_categorical_param_config):
        """Test creating valid CATEGORICAL parameter config."""
        assert valid_categorical_param_config.type == ParamType.CATEGORICAL
        assert valid_categorical_param_config.choices == ["relu", "gelu", "silu"]

    def test_valid_loguniform_param_creation(self, valid_loguniform_param_config):
        """Test creating valid LOGUNIFORM parameter config."""
        assert valid_loguniform_param_config.type == ParamType.LOGUNIFORM
        assert valid_loguniform_param_config.low == 1e-5
        assert valid_loguniform_param_config.high == 1e-2

    def test_valid_uniform_param_creation(self):
        """Test creating valid UNIFORM parameter config."""
        config = SearchSpaceParamConfig(type=ParamType.UNIFORM, low=0.0, high=10.0)
        assert config.type == ParamType.UNIFORM
        assert config.low == 0.0
        assert config.high == 10.0

    def test_valid_int_uniform_param_creation(self):
        """Test creating valid INT_UNIFORM parameter config."""
        config = SearchSpaceParamConfig(type=ParamType.INT_UNIFORM, low=1, high=100)
        assert config.type == ParamType.INT_UNIFORM
        assert config.low == 1
        assert config.high == 100

    def test_valid_discrete_uniform_param_creation(self):
        """Test creating valid DISCRETE_UNIFORM parameter config."""
        config = SearchSpaceParamConfig(type=ParamType.DISCRETE_UNIFORM, low=0, high=10, step=1)
        assert config.type == ParamType.DISCRETE_UNIFORM
        assert config.low == 0
        assert config.high == 10
        assert config.step == 1

    def test_float_param_with_log_flag(self):
        """Test float parameter with log flag."""
        config = SearchSpaceParamConfig(type=ParamType.FLOAT, low=1e-5, high=1e-2, log=True)
        assert config.log is True

    def test_default_log_is_false(self, valid_float_param_config):
        """Test default log flag is False."""
        assert valid_float_param_config.log is False


class TestSearchSpaceParamConfigValidationErrors:
    """Test SearchSpaceParamConfig validation errors (Pydantic V2)."""

    def test_int_param_missing_low_raises_error(self):
        """Test INT param without low raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SearchSpaceParamConfig(type=ParamType.INT, high=100)
        error_str = str(exc_info.value)
        assert "low" in error_str.lower() or "high" in error_str.lower()

    def test_int_param_missing_high_raises_error(self):
        """Test INT param without high raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SearchSpaceParamConfig(type=ParamType.INT, low=1)
        error_str = str(exc_info.value)
        assert "low" in error_str.lower() or "high" in error_str.lower()

    def test_float_param_missing_low_raises_error(self):
        """Test FLOAT param without low raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SearchSpaceParamConfig(type=ParamType.FLOAT, high=1.0)
        error_str = str(exc_info.value)
        assert "low" in error_str.lower() or "high" in error_str.lower()

    def test_float_param_missing_high_raises_error(self):
        """Test FLOAT param without high raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SearchSpaceParamConfig(type=ParamType.FLOAT, low=0.0)
        error_str = str(exc_info.value)
        assert "low" in error_str.lower() or "high" in error_str.lower()

    def test_loguniform_param_missing_bounds_raises_error(self):
        """Test LOGUNIFORM param without bounds raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SearchSpaceParamConfig(type=ParamType.LOGUNIFORM)
        error_str = str(exc_info.value)
        assert "low" in error_str.lower() or "high" in error_str.lower()

    def test_uniform_param_missing_bounds_raises_error(self):
        """Test UNIFORM param without bounds raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SearchSpaceParamConfig(type=ParamType.UNIFORM)
        error_str = str(exc_info.value)
        assert "low" in error_str.lower() or "high" in error_str.lower()

    def test_int_uniform_param_missing_bounds_raises_error(self):
        """Test INT_UNIFORM param without bounds raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SearchSpaceParamConfig(type=ParamType.INT_UNIFORM, low=1)
        error_str = str(exc_info.value)
        assert "low" in error_str.lower() or "high" in error_str.lower()

    def test_low_equals_high_raises_error(self):
        """Test low == high raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SearchSpaceParamConfig(type=ParamType.INT, low=100, high=100)
        error_str = str(exc_info.value)
        assert "low" in error_str.lower() or "less" in error_str.lower()

    def test_low_greater_than_high_raises_error(self):
        """Test low > high raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SearchSpaceParamConfig(type=ParamType.FLOAT, low=100.0, high=10.0)
        error_str = str(exc_info.value)
        assert "low" in error_str.lower() or "less" in error_str.lower()

    def test_categorical_missing_choices_raises_error(self):
        """Test CATEGORICAL param without choices raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SearchSpaceParamConfig(type=ParamType.CATEGORICAL)
        error_str = str(exc_info.value)
        assert "choices" in error_str.lower()

    def test_categorical_empty_choices_raises_error(self):
        """Test CATEGORICAL param with empty choices raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SearchSpaceParamConfig(type=ParamType.CATEGORICAL, choices=[])
        error_str = str(exc_info.value)
        assert "choices" in error_str.lower() or "empty" in error_str.lower()

    def test_discrete_uniform_without_step_uses_default(self):
        """Test DISCRETE_UNIFORM without step uses default value."""
        # DISCRETE_UNIFORM does not require step - it has a default
        config = SearchSpaceParamConfig(type=ParamType.DISCRETE_UNIFORM, low=0.0, high=1.0)
        assert config.type == ParamType.DISCRETE_UNIFORM
        assert config.low == 0.0
        assert config.high == 1.0

    def test_discrete_uniform_missing_bounds_raises_error(self):
        """Test DISCRETE_UNIFORM without bounds raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SearchSpaceParamConfig(type=ParamType.DISCRETE_UNIFORM, step=0.1)
        error_str = str(exc_info.value)
        assert "low" in error_str.lower() or "high" in error_str.lower()


class TestSearchSpaceParamConfigFrozen:
    """Test SearchSpaceParamConfig is frozen (immutable via Pydantic V2)."""

    def test_cannot_modify_type_after_creation(self, valid_int_param_config):
        """Test type cannot be modified after creation."""
        with pytest.raises(ValidationError, match="frozen"):
            valid_int_param_config.type = ParamType.FLOAT

    def test_cannot_modify_low_after_creation(self, valid_int_param_config):
        """Test low cannot be modified after creation."""
        with pytest.raises(ValidationError, match="frozen"):
            valid_int_param_config.low = 0

    def test_cannot_modify_high_after_creation(self, valid_int_param_config):
        """Test high cannot be modified after creation."""
        with pytest.raises(ValidationError, match="frozen"):
            valid_int_param_config.high = 512

    def test_cannot_modify_choices_after_creation(self, valid_categorical_param_config):
        """Test choices cannot be modified after creation."""
        with pytest.raises(ValidationError, match="frozen"):
            valid_categorical_param_config.choices = ["new_choice"]


# =============================================================================
# PRUNERTYPE ENUM TESTS
# =============================================================================


class TestPrunerTypeEnum:
    """Test PrunerType enum values and behavior."""

    def test_pruner_type_median_value(self):
        """Test PrunerType.MEDIAN has correct value."""
        assert PrunerType.MEDIAN.value == "median"

    def test_pruner_type_percentile_value(self):
        """Test PrunerType.PERCENTILE has correct value."""
        assert PrunerType.PERCENTILE.value == "percentile"

    def test_pruner_type_hyperband_value(self):
        """Test PrunerType.HYPERBAND has correct value."""
        assert PrunerType.HYPERBAND.value == "hyperband"

    def test_pruner_type_successive_halving_value(self):
        """Test PrunerType.SUCCESSIVE_HALVING has correct value."""
        assert PrunerType.SUCCESSIVE_HALVING.value == "successive_halving"

    def test_pruner_type_threshold_value(self):
        """Test PrunerType.THRESHOLD has correct value."""
        assert PrunerType.THRESHOLD.value == "threshold"

    def test_pruner_type_patient_value(self):
        """Test PrunerType.PATIENT has correct value."""
        assert PrunerType.PATIENT.value == "patient"

    def test_pruner_type_none_value(self):
        """Test PrunerType.NONE has correct value."""
        assert PrunerType.NONE.value == "none"

    def test_pruner_type_from_string(self):
        """Test creating PrunerType from string value."""
        assert PrunerType("median") == PrunerType.MEDIAN
        assert PrunerType("hyperband") == PrunerType.HYPERBAND
        assert PrunerType("none") == PrunerType.NONE

    def test_all_pruner_types_defined(self):
        """Test all expected pruner types are defined."""
        expected_types = {
            "MEDIAN",
            "PERCENTILE",
            "HYPERBAND",
            "SUCCESSIVE_HALVING",
            "THRESHOLD",
            "PATIENT",
            "NONE",
        }
        actual_types = {pt.name for pt in PrunerType}
        assert actual_types == expected_types


# =============================================================================
# PRUNERCONFIG TESTS
# =============================================================================


class TestPrunerConfigValidCreation:
    """Test valid PrunerConfig creation."""

    def test_default_pruner_config_values(self, valid_pruner_config):
        """Test default PrunerConfig values."""
        assert valid_pruner_config.type == PrunerType.MEDIAN
        assert valid_pruner_config.n_startup_trials == 5
        assert valid_pruner_config.n_warmup_steps == 10
        assert valid_pruner_config.interval_steps == 1
        assert valid_pruner_config.percentile == 25.0
        assert valid_pruner_config.n_brackets == 4

    def test_custom_median_pruner(self):
        """Test custom median pruner config."""
        config = PrunerConfig(type=PrunerType.MEDIAN, n_startup_trials=10, n_warmup_steps=20)
        assert config.type == PrunerType.MEDIAN
        assert config.n_startup_trials == 10
        assert config.n_warmup_steps == 20

    def test_percentile_pruner(self):
        """Test percentile pruner config."""
        config = PrunerConfig(type=PrunerType.PERCENTILE, percentile=50.0)
        assert config.type == PrunerType.PERCENTILE
        assert config.percentile == 50.0

    def test_hyperband_pruner(self):
        """Test hyperband pruner config."""
        config = PrunerConfig(type=PrunerType.HYPERBAND, n_brackets=5)
        assert config.type == PrunerType.HYPERBAND
        assert config.n_brackets == 5

    def test_none_pruner(self):
        """Test none pruner config (no pruning)."""
        config = PrunerConfig(type=PrunerType.NONE)
        assert config.type == PrunerType.NONE

    def test_zero_startup_trials_allowed(self):
        """Test zero startup trials is allowed."""
        config = PrunerConfig(n_startup_trials=0)
        assert config.n_startup_trials == 0

    def test_zero_warmup_steps_allowed(self):
        """Test zero warmup steps is allowed."""
        config = PrunerConfig(n_warmup_steps=0)
        assert config.n_warmup_steps == 0


class TestPrunerConfigValidationErrors:
    """Test PrunerConfig validation errors (Pydantic V2)."""

    def test_negative_startup_trials_raises_error(self):
        """Test negative n_startup_trials raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PrunerConfig(n_startup_trials=-1)
        error_str = str(exc_info.value)
        assert "n_startup_trials" in error_str or "non-negative" in error_str.lower()

    def test_negative_warmup_steps_raises_error(self):
        """Test negative n_warmup_steps raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PrunerConfig(n_warmup_steps=-5)
        error_str = str(exc_info.value)
        assert "n_warmup_steps" in error_str or "non-negative" in error_str.lower()

    def test_zero_interval_steps_raises_error(self):
        """Test zero interval_steps raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PrunerConfig(interval_steps=0)
        error_str = str(exc_info.value)
        assert "interval_steps" in error_str or "at least 1" in error_str.lower()

    def test_negative_interval_steps_raises_error(self):
        """Test negative interval_steps raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PrunerConfig(interval_steps=-1)
        error_str = str(exc_info.value)
        assert "interval_steps" in error_str or "at least 1" in error_str.lower()

    def test_percentile_zero_raises_error(self):
        """Test percentile == 0 raises ValidationError for percentile pruner."""
        with pytest.raises(ValidationError) as exc_info:
            PrunerConfig(type=PrunerType.PERCENTILE, percentile=0.0)
        error_str = str(exc_info.value)
        assert "percentile" in error_str.lower()

    def test_percentile_100_raises_error(self):
        """Test percentile == 100 raises ValidationError for percentile pruner."""
        with pytest.raises(ValidationError) as exc_info:
            PrunerConfig(type=PrunerType.PERCENTILE, percentile=100.0)
        error_str = str(exc_info.value)
        assert "percentile" in error_str.lower()

    def test_percentile_negative_raises_error(self):
        """Test negative percentile raises ValidationError for percentile pruner."""
        with pytest.raises(ValidationError) as exc_info:
            PrunerConfig(type=PrunerType.PERCENTILE, percentile=-10.0)
        error_str = str(exc_info.value)
        assert "percentile" in error_str.lower()

    def test_percentile_over_100_raises_error(self):
        """Test percentile > 100 raises ValidationError for percentile pruner."""
        with pytest.raises(ValidationError) as exc_info:
            PrunerConfig(type=PrunerType.PERCENTILE, percentile=150.0)
        error_str = str(exc_info.value)
        assert "percentile" in error_str.lower()

    def test_hyperband_zero_brackets_raises_error(self):
        """Test n_brackets == 0 raises ValidationError for hyperband pruner."""
        with pytest.raises(ValidationError) as exc_info:
            PrunerConfig(type=PrunerType.HYPERBAND, n_brackets=0)
        error_str = str(exc_info.value)
        assert "n_brackets" in error_str or "at least 1" in error_str.lower()

    def test_hyperband_negative_brackets_raises_error(self):
        """Test negative n_brackets raises ValidationError for hyperband pruner."""
        with pytest.raises(ValidationError) as exc_info:
            PrunerConfig(type=PrunerType.HYPERBAND, n_brackets=-2)
        error_str = str(exc_info.value)
        assert "n_brackets" in error_str or "at least 1" in error_str.lower()


class TestPrunerConfigFrozen:
    """Test PrunerConfig is frozen (immutable via Pydantic V2)."""

    def test_cannot_modify_type_after_creation(self, valid_pruner_config):
        """Test type cannot be modified after creation."""
        with pytest.raises(ValidationError, match="frozen"):
            valid_pruner_config.type = PrunerType.HYPERBAND

    def test_cannot_modify_n_startup_trials_after_creation(self, valid_pruner_config):
        """Test n_startup_trials cannot be modified after creation."""
        with pytest.raises(ValidationError, match="frozen"):
            valid_pruner_config.n_startup_trials = 100


# =============================================================================
# SAMPLERTYPE ENUM TESTS
# =============================================================================


class TestSamplerTypeEnum:
    """Test SamplerType enum values and behavior."""

    def test_sampler_type_tpe_value(self):
        """Test SamplerType.TPE has correct value."""
        assert SamplerType.TPE.value == "tpe"

    def test_sampler_type_random_value(self):
        """Test SamplerType.RANDOM has correct value."""
        assert SamplerType.RANDOM.value == "random"

    def test_sampler_type_cmaes_value(self):
        """Test SamplerType.CMAES has correct value."""
        assert SamplerType.CMAES.value == "cmaes"

    def test_sampler_type_grid_value(self):
        """Test SamplerType.GRID has correct value."""
        assert SamplerType.GRID.value == "grid"

    def test_sampler_type_nsgaii_value(self):
        """Test SamplerType.NSGAII has correct value."""
        assert SamplerType.NSGAII.value == "nsgaii"

    def test_sampler_type_motpe_value(self):
        """Test SamplerType.MOTPE has correct value."""
        assert SamplerType.MOTPE.value == "motpe"

    def test_sampler_type_qmcsampler_value(self):
        """Test SamplerType.QMCSAMPLER has correct value."""
        assert SamplerType.QMCSAMPLER.value == "qmc"

    def test_sampler_type_from_string(self):
        """Test creating SamplerType from string value."""
        assert SamplerType("tpe") == SamplerType.TPE
        assert SamplerType("random") == SamplerType.RANDOM
        assert SamplerType("qmc") == SamplerType.QMCSAMPLER

    def test_all_sampler_types_defined(self):
        """Test all expected sampler types are defined."""
        expected_types = {"TPE", "RANDOM", "CMAES", "GRID", "NSGAII", "MOTPE", "QMCSAMPLER"}
        actual_types = {st.name for st in SamplerType}
        assert actual_types == expected_types


# =============================================================================
# SAMPLERCONFIG TESTS
# =============================================================================


class TestSamplerConfigValidCreation:
    """Test valid SamplerConfig creation."""

    def test_default_sampler_config_values(self, valid_sampler_config):
        """Test default SamplerConfig values."""
        assert valid_sampler_config.type == SamplerType.TPE
        assert valid_sampler_config.n_startup_trials == 10
        assert valid_sampler_config.seed is None
        assert valid_sampler_config.multivariate is True
        assert valid_sampler_config.constant_liar is False

    def test_custom_tpe_sampler(self):
        """Test custom TPE sampler config."""
        config = SamplerConfig(type=SamplerType.TPE, n_startup_trials=20, multivariate=False)
        assert config.type == SamplerType.TPE
        assert config.n_startup_trials == 20
        assert config.multivariate is False

    def test_random_sampler_with_seed(self):
        """Test random sampler with seed."""
        config = SamplerConfig(type=SamplerType.RANDOM, seed=42)
        assert config.type == SamplerType.RANDOM
        assert config.seed == 42

    def test_tpe_sampler_with_constant_liar(self):
        """Test TPE sampler with constant_liar for parallel optimization."""
        config = SamplerConfig(type=SamplerType.TPE, constant_liar=True)
        assert config.constant_liar is True

    def test_zero_startup_trials_allowed(self):
        """Test zero startup trials is allowed."""
        config = SamplerConfig(n_startup_trials=0)
        assert config.n_startup_trials == 0

    def test_zero_seed_allowed(self):
        """Test zero seed is allowed."""
        config = SamplerConfig(seed=0)
        assert config.seed == 0


class TestSamplerConfigValidationErrors:
    """Test SamplerConfig validation errors (Pydantic V2)."""

    def test_negative_startup_trials_raises_error(self):
        """Test negative n_startup_trials raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SamplerConfig(n_startup_trials=-1)
        error_str = str(exc_info.value)
        assert "n_startup_trials" in error_str or "non-negative" in error_str.lower()

    def test_negative_seed_raises_error(self):
        """Test negative seed raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SamplerConfig(seed=-42)
        error_str = str(exc_info.value)
        assert "seed" in error_str or "non-negative" in error_str.lower()


class TestSamplerConfigFrozen:
    """Test SamplerConfig is frozen (immutable via Pydantic V2)."""

    def test_cannot_modify_type_after_creation(self, valid_sampler_config):
        """Test type cannot be modified after creation."""
        with pytest.raises(ValidationError, match="frozen"):
            valid_sampler_config.type = SamplerType.RANDOM

    def test_cannot_modify_seed_after_creation(self, valid_sampler_config):
        """Test seed cannot be modified after creation."""
        with pytest.raises(ValidationError, match="frozen"):
            valid_sampler_config.seed = 42


# =============================================================================
# OPTIMIZATIONDIRECTION ENUM TESTS
# =============================================================================


class TestOptimizationDirectionEnum:
    """Test OptimizationDirection enum values and behavior."""

    def test_optimization_direction_minimize_value(self):
        """Test OptimizationDirection.MINIMIZE has correct value."""
        assert OptimizationDirection.MINIMIZE.value == "minimize"

    def test_optimization_direction_maximize_value(self):
        """Test OptimizationDirection.MAXIMIZE has correct value."""
        assert OptimizationDirection.MAXIMIZE.value == "maximize"

    def test_optimization_direction_from_string(self):
        """Test creating OptimizationDirection from string value."""
        assert OptimizationDirection("minimize") == OptimizationDirection.MINIMIZE
        assert OptimizationDirection("maximize") == OptimizationDirection.MAXIMIZE

    def test_all_optimization_directions_defined(self):
        """Test all expected optimization directions are defined."""
        expected_directions = {"MINIMIZE", "MAXIMIZE"}
        actual_directions = {od.name for od in OptimizationDirection}
        assert actual_directions == expected_directions


# =============================================================================
# STUDYCONFIG TESTS
# =============================================================================


class TestStudyConfigValidCreation:
    """Test valid StudyConfig creation."""

    def test_default_study_config_values(self, valid_study_config):
        """Test default StudyConfig values."""
        assert valid_study_config.direction == OptimizationDirection.MINIMIZE
        assert valid_study_config.metric == "val_loss"
        assert valid_study_config.study_name == "milia_hpo"
        assert valid_study_config.storage is None
        assert valid_study_config.load_if_exists is True

    def test_custom_study_config(self):
        """Test custom study config."""
        config = StudyConfig(
            direction=OptimizationDirection.MAXIMIZE,
            metric="val_accuracy",
            study_name="accuracy_study",
            storage="sqlite:///study.db",
            load_if_exists=False,
        )
        assert config.direction == OptimizationDirection.MAXIMIZE
        assert config.metric == "val_accuracy"
        assert config.study_name == "accuracy_study"
        assert config.storage == "sqlite:///study.db"
        assert config.load_if_exists is False

    def test_is_multi_objective_property_false(self, valid_study_config):
        """Test is_multi_objective property returns False for StudyConfig."""
        assert valid_study_config.is_multi_objective is False

    def test_storage_with_mysql_url(self):
        """Test storage with MySQL URL."""
        config = StudyConfig(storage="mysql://user:pass@localhost/db")  # pragma: allowlist secret
        assert config.storage == "mysql://user:pass@localhost/db"  # pragma: allowlist secret

    def test_storage_with_postgresql_url(self):
        """Test storage with PostgreSQL URL."""
        config = StudyConfig(
            storage="postgresql://user:pass@localhost/db"  # pragma: allowlist secret
        )
        assert config.storage == "postgresql://user:pass@localhost/db"  # pragma: allowlist secret


class TestStudyConfigValidationErrors:
    """Test StudyConfig validation errors (Pydantic V2)."""

    def test_empty_metric_raises_error(self):
        """Test empty metric raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            StudyConfig(metric="")
        error_str = str(exc_info.value)
        assert "metric" in error_str.lower() or "empty" in error_str.lower()

    def test_empty_study_name_raises_error(self):
        """Test empty study_name raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            StudyConfig(study_name="")
        error_str = str(exc_info.value)
        assert "study_name" in error_str or "empty" in error_str.lower()

    def test_non_string_storage_raises_error(self):
        """Test non-string storage raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            StudyConfig(storage=123)
        error_str = str(exc_info.value)
        assert "storage" in error_str.lower() or "string" in error_str.lower()

    def test_non_string_storage_dict_raises_error(self):
        """Test dict storage raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            StudyConfig(storage={"url": "sqlite:///study.db"})
        error_str = str(exc_info.value)
        assert "storage" in error_str.lower() or "string" in error_str.lower()


class TestStudyConfigFrozen:
    """Test StudyConfig is frozen (immutable via Pydantic V2)."""

    def test_cannot_modify_direction_after_creation(self, valid_study_config):
        """Test direction cannot be modified after creation."""
        with pytest.raises(ValidationError, match="frozen"):
            valid_study_config.direction = OptimizationDirection.MAXIMIZE

    def test_cannot_modify_metric_after_creation(self, valid_study_config):
        """Test metric cannot be modified after creation."""
        with pytest.raises(ValidationError, match="frozen"):
            valid_study_config.metric = "new_metric"


# =============================================================================
# MULTIOBJECTIVESTUDYCONFIG TESTS
# =============================================================================


class TestMultiObjectiveStudyConfigValidCreation:
    """Test valid MultiObjectiveStudyConfig creation."""

    def test_valid_multi_objective_config(self, valid_multi_objective_config):
        """Test valid multi-objective study config."""
        assert valid_multi_objective_config.directions == ("minimize", "maximize")
        assert valid_multi_objective_config.metrics == ("val_loss", "val_accuracy")
        assert valid_multi_objective_config.study_name == "milia_hpo_multi"
        assert valid_multi_objective_config.storage is None
        assert valid_multi_objective_config.load_if_exists is True
        assert valid_multi_objective_config.reference_point is None

    def test_is_multi_objective_property_true(self, valid_multi_objective_config):
        """Test is_multi_objective property returns True."""
        assert valid_multi_objective_config.is_multi_objective is True

    def test_multi_objective_with_reference_point(self):
        """Test multi-objective config with reference point."""
        config = MultiObjectiveStudyConfig(
            directions=("minimize", "minimize"),
            metrics=("val_loss", "training_time"),
            reference_point=(1.0, 3600.0),
        )
        assert config.reference_point == (1.0, 3600.0)

    def test_three_objectives(self):
        """Test config with three objectives."""
        config = MultiObjectiveStudyConfig(
            directions=("minimize", "maximize", "minimize"),
            metrics=("loss", "accuracy", "inference_time"),
        )
        assert len(config.directions) == 3
        assert len(config.metrics) == 3

    def test_all_minimize_directions(self):
        """Test config with all minimize directions."""
        config = MultiObjectiveStudyConfig(
            directions=("minimize", "minimize"), metrics=("loss1", "loss2")
        )
        assert all(d == "minimize" for d in config.directions)

    def test_all_maximize_directions(self):
        """Test config with all maximize directions."""
        config = MultiObjectiveStudyConfig(
            directions=("maximize", "maximize"), metrics=("acc1", "acc2")
        )
        assert all(d == "maximize" for d in config.directions)


class TestMultiObjectiveStudyConfigValidationErrors:
    """Test MultiObjectiveStudyConfig validation errors (Pydantic V2)."""

    def test_directions_metrics_length_mismatch_raises_error(self):
        """Test directions/metrics length mismatch raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MultiObjectiveStudyConfig(
                directions=("minimize", "maximize"),
                metrics=("val_loss",),  # Only one metric
            )
        error_str = str(exc_info.value)
        assert "length" in error_str.lower() or "directions" in error_str.lower()

    def test_directions_metrics_length_mismatch_more_metrics(self):
        """Test more metrics than directions raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MultiObjectiveStudyConfig(
                directions=("minimize",), metrics=("val_loss", "val_accuracy")
            )
        error_str = str(exc_info.value)
        assert "length" in error_str.lower() or "directions" in error_str.lower()

    def test_invalid_direction_raises_error(self):
        """Test invalid direction raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MultiObjectiveStudyConfig(
                directions=("minimize", "invalid_direction"), metrics=("val_loss", "val_accuracy")
            )
        error_str = str(exc_info.value)
        assert "direction" in error_str.lower() or "invalid" in error_str.lower()

    def test_single_metric_raises_error(self):
        """Test single metric raises ValidationError (need at least 2)."""
        with pytest.raises(ValidationError) as exc_info:
            MultiObjectiveStudyConfig(directions=("minimize",), metrics=("val_loss",))
        error_str = str(exc_info.value)
        assert "metric" in error_str.lower() or "at least 2" in error_str.lower()

    def test_empty_metric_in_list_raises_error(self):
        """Test empty metric string in list raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MultiObjectiveStudyConfig(directions=("minimize", "maximize"), metrics=("val_loss", ""))
        error_str = str(exc_info.value)
        assert "metric" in error_str.lower() or "empty" in error_str.lower()

    def test_reference_point_length_mismatch_raises_error(self):
        """Test reference_point length mismatch raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MultiObjectiveStudyConfig(
                directions=("minimize", "maximize"),
                metrics=("val_loss", "val_accuracy"),
                reference_point=(1.0,),  # Only one value
            )
        error_str = str(exc_info.value)
        assert "reference_point" in error_str.lower() or "length" in error_str.lower()

    def test_empty_study_name_raises_error(self):
        """Test empty study_name raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MultiObjectiveStudyConfig(
                directions=("minimize", "maximize"),
                metrics=("val_loss", "val_accuracy"),
                study_name="",
            )
        error_str = str(exc_info.value)
        assert "study_name" in error_str.lower() or "empty" in error_str.lower()


class TestMultiObjectiveStudyConfigFrozen:
    """Test MultiObjectiveStudyConfig is frozen (immutable via Pydantic V2)."""

    def test_cannot_modify_directions_after_creation(self, valid_multi_objective_config):
        """Test directions cannot be modified after creation."""
        with pytest.raises(ValidationError, match="frozen"):
            valid_multi_objective_config.directions = ("maximize", "minimize")

    def test_cannot_modify_metrics_after_creation(self, valid_multi_objective_config):
        """Test metrics cannot be modified after creation."""
        with pytest.raises(ValidationError, match="frozen"):
            valid_multi_objective_config.metrics = ("new_metric1", "new_metric2")


# =============================================================================
# HPOCONFIG TESTS - VALID CREATION
# =============================================================================


class TestHPOConfigValidCreation:
    """Test valid HPOConfig creation."""

    def test_default_hpo_config_values(self, valid_hpo_config):
        """Test default HPOConfig values."""
        assert valid_hpo_config.enabled is False
        assert valid_hpo_config.backend == "optuna"
        assert valid_hpo_config.n_trials == 100
        assert valid_hpo_config.timeout is None
        assert valid_hpo_config.n_jobs == 1
        assert valid_hpo_config.search_space == {}
        assert isinstance(valid_hpo_config.pruner, PrunerConfig)
        assert isinstance(valid_hpo_config.sampler, SamplerConfig)
        assert isinstance(valid_hpo_config.study, StudyConfig)
        assert valid_hpo_config.cv_folds == 0
        assert valid_hpo_config.cv_metric_aggregation == "mean"
        assert valid_hpo_config.task_type is None

    def test_enabled_hpo_config(self):
        """Test HPOConfig with enabled=True."""
        config = HPOConfig(enabled=True)
        assert config.enabled is True

    def test_custom_n_trials(self):
        """Test HPOConfig with custom n_trials."""
        config = HPOConfig(n_trials=500)
        assert config.n_trials == 500

    def test_custom_timeout(self):
        """Test HPOConfig with custom timeout."""
        config = HPOConfig(timeout=3600)
        assert config.timeout == 3600

    def test_parallel_jobs(self):
        """Test HPOConfig with multiple jobs."""
        config = HPOConfig(n_jobs=4)
        assert config.n_jobs == 4

    def test_ray_tune_backend(self):
        """Test HPOConfig with ray_tune backend."""
        config = HPOConfig(backend="ray_tune")
        assert config.backend == "ray_tune"

    def test_cv_folds_configuration(self):
        """Test HPOConfig with cross-validation folds."""
        config = HPOConfig(cv_folds=5)
        assert config.cv_folds == 5

    def test_cv_metric_aggregation_median(self):
        """Test HPOConfig with median aggregation."""
        config = HPOConfig(cv_metric_aggregation="median")
        assert config.cv_metric_aggregation == "median"

    def test_cv_metric_aggregation_min(self):
        """Test HPOConfig with min aggregation."""
        config = HPOConfig(cv_metric_aggregation="min")
        assert config.cv_metric_aggregation == "min"

    def test_cv_metric_aggregation_max(self):
        """Test HPOConfig with max aggregation."""
        config = HPOConfig(cv_metric_aggregation="max")
        assert config.cv_metric_aggregation == "max"

    def test_zero_cv_folds_means_no_cv(self):
        """Test cv_folds=0 means no cross-validation."""
        config = HPOConfig(cv_folds=0)
        assert config.cv_folds == 0

    def test_default_task_type_is_none(self):
        """Test default task_type is None."""
        config = HPOConfig()
        assert config.task_type is None

    def test_custom_task_type_graph_regression(self):
        """Test HPOConfig with task_type='graph_regression'."""
        config = HPOConfig(task_type="graph_regression")
        assert config.task_type == "graph_regression"

    def test_custom_task_type_node_classification(self):
        """Test HPOConfig with task_type='node_classification'."""
        config = HPOConfig(task_type="node_classification")
        assert config.task_type == "node_classification"

    def test_custom_task_type_edge_prediction(self):
        """Test HPOConfig with task_type='edge_prediction'."""
        config = HPOConfig(task_type="edge_prediction")
        assert config.task_type == "edge_prediction"


# =============================================================================
# HPOCONFIG TESTS - VALIDATION ERRORS
# =============================================================================


class TestHPOConfigValidationErrors:
    """Test HPOConfig validation errors (Pydantic V2)."""

    def test_invalid_backend_raises_error(self):
        """Test invalid backend raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            HPOConfig(backend="invalid_backend")
        error_str = str(exc_info.value)
        assert "backend" in error_str.lower() or "optuna" in error_str.lower()

    def test_zero_n_trials_raises_error(self):
        """Test n_trials=0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            HPOConfig(n_trials=0)
        error_str = str(exc_info.value)
        assert "n_trials" in error_str or "at least 1" in error_str.lower()

    def test_negative_n_trials_raises_error(self):
        """Test negative n_trials raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            HPOConfig(n_trials=-10)
        error_str = str(exc_info.value)
        assert "n_trials" in error_str or "at least 1" in error_str.lower()

    def test_zero_timeout_raises_error(self):
        """Test timeout=0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            HPOConfig(timeout=0)
        error_str = str(exc_info.value)
        assert "timeout" in error_str.lower() or "positive" in error_str.lower()

    def test_negative_timeout_raises_error(self):
        """Test negative timeout raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            HPOConfig(timeout=-100)
        error_str = str(exc_info.value)
        assert "timeout" in error_str.lower() or "positive" in error_str.lower()

    def test_zero_n_jobs_raises_error(self):
        """Test n_jobs=0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            HPOConfig(n_jobs=0)
        error_str = str(exc_info.value)
        assert "n_jobs" in error_str or "at least 1" in error_str.lower()

    def test_negative_n_jobs_raises_error(self):
        """Test negative n_jobs raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            HPOConfig(n_jobs=-1)
        error_str = str(exc_info.value)
        assert "n_jobs" in error_str or "at least 1" in error_str.lower()

    def test_negative_cv_folds_raises_error(self):
        """Test negative cv_folds raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            HPOConfig(cv_folds=-1)
        error_str = str(exc_info.value)
        assert "cv_folds" in error_str or "non-negative" in error_str.lower()

    def test_invalid_cv_metric_aggregation_raises_error(self):
        """Test invalid cv_metric_aggregation raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            HPOConfig(cv_metric_aggregation="invalid_agg")
        error_str = str(exc_info.value)
        assert "cv_metric_aggregation" in error_str or "mean" in error_str.lower()


class TestHPOConfigFrozen:
    """Test HPOConfig is frozen (immutable via Pydantic V2)."""

    def test_cannot_modify_enabled_after_creation(self, valid_hpo_config):
        """Test enabled cannot be modified after creation."""
        with pytest.raises(ValidationError, match="frozen"):
            valid_hpo_config.enabled = True

    def test_cannot_modify_backend_after_creation(self, valid_hpo_config):
        """Test backend cannot be modified after creation."""
        with pytest.raises(ValidationError, match="frozen"):
            valid_hpo_config.backend = "ray_tune"

    def test_cannot_modify_n_trials_after_creation(self, valid_hpo_config):
        """Test n_trials cannot be modified after creation."""
        with pytest.raises(ValidationError, match="frozen"):
            valid_hpo_config.n_trials = 500

    def test_cannot_modify_task_type_after_creation(self):
        """Test task_type cannot be modified after creation."""
        config = HPOConfig(task_type="graph_regression")
        with pytest.raises(ValidationError, match="frozen"):
            config.task_type = "node_classification"


# =============================================================================
# HPOCONFIG FROM_DICT TESTS
# =============================================================================


class TestHPOConfigFromDict:
    """Test HPOConfig.from_dict() factory method."""

    def test_from_dict_basic(self, sample_config_dict):
        """Test creating HPOConfig from basic dictionary."""
        config = HPOConfig.from_dict(sample_config_dict)

        assert config.enabled is True
        assert config.n_trials == 100
        assert config.backend == "optuna"
        assert config.timeout == 3600
        assert config.n_jobs == 4
        assert config.cv_folds == 5
        assert config.cv_metric_aggregation == "mean"

    def test_from_dict_pruner_parsing(self, sample_config_dict):
        """Test from_dict correctly parses pruner configuration."""
        config = HPOConfig.from_dict(sample_config_dict)

        assert config.pruner.type == PrunerType.MEDIAN
        assert config.pruner.n_startup_trials == 10
        assert config.pruner.n_warmup_steps == 5

    def test_from_dict_sampler_parsing(self, sample_config_dict):
        """Test from_dict correctly parses sampler configuration."""
        config = HPOConfig.from_dict(sample_config_dict)

        assert config.sampler.type == SamplerType.TPE
        assert config.sampler.n_startup_trials == 15
        assert config.sampler.seed == 42

    def test_from_dict_study_parsing(self, sample_config_dict):
        """Test from_dict correctly parses study configuration."""
        config = HPOConfig.from_dict(sample_config_dict)

        assert config.study.direction == OptimizationDirection.MINIMIZE
        assert config.study.metric == "val_loss"
        assert config.study.study_name == "test_study"

    def test_from_dict_search_space_parsing(self, sample_config_dict):
        """Test from_dict correctly parses search space configuration."""
        config = HPOConfig.from_dict(sample_config_dict)

        assert "model" in config.search_space
        assert "optimizer" in config.search_space

        # Check model parameters
        assert "hidden_channels" in config.search_space["model"]
        assert config.search_space["model"]["hidden_channels"].type == ParamType.INT
        assert config.search_space["model"]["hidden_channels"].low == 32
        assert config.search_space["model"]["hidden_channels"].high == 256

        assert "dropout" in config.search_space["model"]
        assert config.search_space["model"]["dropout"].type == ParamType.FLOAT

        # Check optimizer parameters
        assert "learning_rate" in config.search_space["optimizer"]
        assert config.search_space["optimizer"]["learning_rate"].type == ParamType.LOGUNIFORM

    def test_from_dict_empty_dict_uses_defaults(self):
        """Test from_dict with empty dict uses default values."""
        config = HPOConfig.from_dict({})

        assert config.enabled is False
        assert config.backend == "optuna"
        assert config.n_trials == 100
        assert config.timeout is None
        assert config.n_jobs == 1
        assert config.task_type is None

    def test_from_dict_partial_config(self):
        """Test from_dict with partial configuration."""
        config_dict = {"enabled": True, "n_trials": 50}
        config = HPOConfig.from_dict(config_dict)

        assert config.enabled is True
        assert config.n_trials == 50
        assert config.backend == "optuna"  # Default

    def test_from_dict_empty_pruner_uses_defaults(self):
        """Test from_dict with empty pruner dict uses defaults."""
        config_dict = {"pruner": {}}
        config = HPOConfig.from_dict(config_dict)

        assert config.pruner.type == PrunerType.MEDIAN
        assert config.pruner.n_startup_trials == 5

    def test_from_dict_empty_sampler_uses_defaults(self):
        """Test from_dict with empty sampler dict uses defaults."""
        config_dict = {"sampler": {}}
        config = HPOConfig.from_dict(config_dict)

        assert config.sampler.type == SamplerType.TPE
        assert config.sampler.n_startup_trials == 10

    def test_from_dict_empty_study_uses_defaults(self):
        """Test from_dict with empty study dict uses defaults."""
        config_dict = {"study": {}}
        config = HPOConfig.from_dict(config_dict)

        assert config.study.direction == OptimizationDirection.MINIMIZE
        assert config.study.metric == "val_loss"

    def test_from_dict_empty_search_space(self):
        """Test from_dict with empty search_space."""
        config_dict = {"search_space": {}}
        config = HPOConfig.from_dict(config_dict)

        assert config.search_space == {}

    def test_from_dict_hyperband_pruner(self):
        """Test from_dict with hyperband pruner."""
        config_dict = {"pruner": {"type": "hyperband", "n_brackets": 5}}
        config = HPOConfig.from_dict(config_dict)

        assert config.pruner.type == PrunerType.HYPERBAND
        assert config.pruner.n_brackets == 5

    def test_from_dict_random_sampler(self):
        """Test from_dict with random sampler."""
        config_dict = {"sampler": {"type": "random", "seed": 123}}
        config = HPOConfig.from_dict(config_dict)

        assert config.sampler.type == SamplerType.RANDOM
        assert config.sampler.seed == 123

    def test_from_dict_maximize_direction(self):
        """Test from_dict with maximize direction."""
        config_dict = {"study": {"direction": "maximize", "metric": "val_accuracy"}}
        config = HPOConfig.from_dict(config_dict)

        assert config.study.direction == OptimizationDirection.MAXIMIZE
        assert config.study.metric == "val_accuracy"

    def test_from_dict_categorical_search_space(self):
        """Test from_dict with categorical search space parameter."""
        config_dict = {
            "search_space": {
                "model": {
                    "activation": {"type": "categorical", "choices": ["relu", "gelu", "silu"]}
                }
            }
        }
        config = HPOConfig.from_dict(config_dict)

        assert config.search_space["model"]["activation"].type == ParamType.CATEGORICAL
        assert config.search_space["model"]["activation"].choices == ["relu", "gelu", "silu"]

    def test_from_dict_discrete_uniform_search_space(self):
        """Test from_dict with discrete_uniform search space parameter."""
        config_dict = {
            "search_space": {
                "model": {
                    "num_layers": {"type": "discrete_uniform", "low": 1, "high": 10, "step": 2}
                }
            }
        }
        config = HPOConfig.from_dict(config_dict)

        assert config.search_space["model"]["num_layers"].type == ParamType.DISCRETE_UNIFORM
        assert config.search_space["model"]["num_layers"].step == 2

    def test_from_dict_task_type_parsing(self):
        """Test from_dict correctly parses task_type."""
        config_dict = {"task_type": "graph_regression"}
        config = HPOConfig.from_dict(config_dict)

        assert config.task_type == "graph_regression"

    def test_from_dict_task_type_node_classification(self):
        """Test from_dict with node_classification task_type."""
        config_dict = {"enabled": True, "task_type": "node_classification"}
        config = HPOConfig.from_dict(config_dict)

        assert config.task_type == "node_classification"
        assert config.enabled is True

    def test_from_dict_task_type_none_when_not_specified(self):
        """Test from_dict returns None task_type when not specified."""
        config_dict = {"enabled": True, "n_trials": 50}
        config = HPOConfig.from_dict(config_dict)

        assert config.task_type is None


class TestHPOConfigFromDictValidationErrors:
    """Test HPOConfig.from_dict() validation errors (Pydantic V2)."""

    def test_from_dict_invalid_param_type_raises_error(self):
        """Test from_dict with invalid param type raises error."""
        config_dict = {
            "search_space": {"model": {"param": {"type": "invalid_type", "low": 0, "high": 100}}}
        }
        with pytest.raises(ValueError):
            HPOConfig.from_dict(config_dict)

    def test_from_dict_invalid_pruner_type_raises_error(self):
        """Test from_dict with invalid pruner type raises error."""
        config_dict = {"pruner": {"type": "invalid_pruner"}}
        with pytest.raises(ValueError):
            HPOConfig.from_dict(config_dict)

    def test_from_dict_invalid_sampler_type_raises_error(self):
        """Test from_dict with invalid sampler type raises error."""
        config_dict = {"sampler": {"type": "invalid_sampler"}}
        with pytest.raises(ValueError):
            HPOConfig.from_dict(config_dict)

    def test_from_dict_invalid_direction_raises_error(self):
        """Test from_dict with invalid direction raises error."""
        config_dict = {"study": {"direction": "invalid_direction"}}
        with pytest.raises(ValueError):
            HPOConfig.from_dict(config_dict)

    def test_from_dict_missing_bounds_for_int_param(self):
        """Test from_dict with missing bounds for int param raises error."""
        config_dict = {
            "search_space": {
                "model": {
                    "param": {
                        "type": "int"
                        # Missing low and high
                    }
                }
            }
        }
        with pytest.raises(ValidationError):
            HPOConfig.from_dict(config_dict)

    def test_from_dict_missing_choices_for_categorical(self):
        """Test from_dict with missing choices for categorical raises error."""
        config_dict = {
            "search_space": {
                "model": {
                    "param": {
                        "type": "categorical"
                        # Missing choices
                    }
                }
            }
        }
        with pytest.raises(ValidationError):
            HPOConfig.from_dict(config_dict)


# =============================================================================
# EDGE CASES AND INTEGRATION TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_float_bounds(self):
        """Test very small float bounds."""
        config = SearchSpaceParamConfig(type=ParamType.FLOAT, low=1e-10, high=1e-9)
        assert config.low == 1e-10
        assert config.high == 1e-9

    def test_very_large_int_bounds(self):
        """Test very large integer bounds."""
        config = SearchSpaceParamConfig(type=ParamType.INT, low=1, high=10**9)
        assert config.high == 10**9

    def test_single_choice_categorical(self):
        """Test categorical with single choice."""
        config = SearchSpaceParamConfig(type=ParamType.CATEGORICAL, choices=["only_option"])
        assert len(config.choices) == 1

    def test_categorical_with_none_choice(self):
        """Test categorical with None as a choice."""
        config = SearchSpaceParamConfig(
            type=ParamType.CATEGORICAL, choices=[None, "option1", "option2"]
        )
        assert None in config.choices

    def test_categorical_with_numeric_choices(self):
        """Test categorical with numeric choices."""
        config = SearchSpaceParamConfig(type=ParamType.CATEGORICAL, choices=[1, 2, 4, 8, 16])
        assert config.choices == [1, 2, 4, 8, 16]

    def test_categorical_with_mixed_type_choices(self):
        """Test categorical with mixed type choices."""
        config = SearchSpaceParamConfig(
            type=ParamType.CATEGORICAL, choices=["string", 123, 0.5, True, None]
        )
        assert len(config.choices) == 5

    def test_very_large_percentile_below_100(self):
        """Test percentile just below 100."""
        config = PrunerConfig(type=PrunerType.PERCENTILE, percentile=99.99)
        assert config.percentile == 99.99

    def test_very_small_percentile_above_0(self):
        """Test percentile just above 0."""
        config = PrunerConfig(type=PrunerType.PERCENTILE, percentile=0.01)
        assert config.percentile == 0.01

    def test_large_n_brackets(self):
        """Test hyperband with large n_brackets."""
        config = PrunerConfig(type=PrunerType.HYPERBAND, n_brackets=100)
        assert config.n_brackets == 100

    def test_minimum_valid_n_trials(self):
        """Test HPOConfig with minimum valid n_trials (1)."""
        config = HPOConfig(n_trials=1)
        assert config.n_trials == 1

    def test_minimum_valid_timeout(self):
        """Test HPOConfig with minimum valid timeout (1)."""
        config = HPOConfig(timeout=1)
        assert config.timeout == 1

    def test_large_cv_folds(self):
        """Test HPOConfig with large cv_folds."""
        config = HPOConfig(cv_folds=100)
        assert config.cv_folds == 100

    def test_float_as_integer_bounds(self):
        """Test integer parameter with float bounds (should work as validation checks numeric)."""
        config = SearchSpaceParamConfig(type=ParamType.INT, low=32.0, high=256.0)
        assert config.low == 32.0
        assert config.high == 256.0


class TestComplexConfigurations:
    """Test complex configuration scenarios."""

    def test_full_hpo_config_with_all_components(self):
        """Test creating a fully configured HPOConfig."""
        search_space = {
            "model": {
                "hidden_channels": SearchSpaceParamConfig(type=ParamType.INT, low=32, high=256),
                "num_layers": SearchSpaceParamConfig(type=ParamType.INT, low=1, high=5),
                "dropout": SearchSpaceParamConfig(type=ParamType.FLOAT, low=0.0, high=0.5),
                "activation": SearchSpaceParamConfig(
                    type=ParamType.CATEGORICAL, choices=["relu", "gelu"]
                ),
            },
            "optimizer": {
                "lr": SearchSpaceParamConfig(type=ParamType.LOGUNIFORM, low=1e-5, high=1e-2)
            },
        }

        pruner = PrunerConfig(type=PrunerType.HYPERBAND, n_startup_trials=10, n_brackets=4)

        sampler = SamplerConfig(
            type=SamplerType.TPE, n_startup_trials=20, seed=42, multivariate=True
        )

        study = StudyConfig(
            direction=OptimizationDirection.MINIMIZE,
            metric="val_mae",
            study_name="full_test_study",
            storage="sqlite:///study.db",
        )

        config = HPOConfig(
            enabled=True,
            backend="optuna",
            n_trials=200,
            timeout=7200,
            n_jobs=8,
            search_space=search_space,
            pruner=pruner,
            sampler=sampler,
            study=study,
            cv_folds=5,
            cv_metric_aggregation="mean",
        )

        assert config.enabled is True
        assert len(config.search_space) == 2
        assert len(config.search_space["model"]) == 4
        assert config.pruner.type == PrunerType.HYPERBAND
        assert config.sampler.seed == 42
        assert config.study.storage == "sqlite:///study.db"

    def test_nested_search_space_categories(self):
        """Test search space with multiple nested categories."""
        search_space = {
            "encoder": {"hidden": SearchSpaceParamConfig(type=ParamType.INT, low=32, high=128)},
            "decoder": {"hidden": SearchSpaceParamConfig(type=ParamType.INT, low=32, high=128)},
            "regularization": {
                "dropout": SearchSpaceParamConfig(type=ParamType.FLOAT, low=0.0, high=0.5),
                "weight_decay": SearchSpaceParamConfig(
                    type=ParamType.LOGUNIFORM, low=1e-6, high=1e-3
                ),
            },
        }

        config = HPOConfig(search_space=search_space)

        assert "encoder" in config.search_space
        assert "decoder" in config.search_space
        assert "regularization" in config.search_space


class TestErrorContextInformation:
    """Test that error messages contain helpful context (Pydantic V2)."""

    def test_validation_error_contains_field_name(self):
        """Test ValidationError includes field name in message."""
        with pytest.raises(ValidationError) as exc_info:
            HPOConfig(backend="invalid")

        error_str = str(exc_info.value)
        # Pydantic V2 includes field context in error
        assert "backend" in error_str.lower()

    def test_validation_error_contains_value_info(self):
        """Test ValidationError includes value information."""
        with pytest.raises(ValidationError) as exc_info:
            HPOConfig(backend="invalid")

        error_str = str(exc_info.value)
        # Check error contains relevant context
        assert "invalid" in error_str.lower() or "backend" in error_str.lower()

    def test_validation_error_contains_constraint_info(self):
        """Test ValidationError includes constraint information."""
        with pytest.raises(ValidationError) as exc_info:
            HPOConfig(backend="invalid")

        error_str = str(exc_info.value)
        # Should mention valid options or constraints
        assert (
            "optuna" in error_str.lower()
            or "ray_tune" in error_str.lower()
            or "backend" in error_str.lower()
        )

    def test_search_space_error_includes_type_info(self):
        """Test search space error includes type information."""
        with pytest.raises(ValidationError) as exc_info:
            SearchSpaceParamConfig(type=ParamType.INT)

        error_str = str(exc_info.value)
        # Should indicate missing bounds or type requirement
        assert (
            "low" in error_str.lower() or "high" in error_str.lower() or "int" in error_str.lower()
        )


class TestBaseModelEquality:
    """Test Pydantic V2 BaseModel equality comparisons."""

    def test_identical_search_space_configs_equal(self):
        """Test identical SearchSpaceParamConfig instances are equal."""
        config1 = SearchSpaceParamConfig(type=ParamType.INT, low=1, high=10)
        config2 = SearchSpaceParamConfig(type=ParamType.INT, low=1, high=10)
        assert config1 == config2

    def test_different_search_space_configs_not_equal(self):
        """Test different SearchSpaceParamConfig instances are not equal."""
        config1 = SearchSpaceParamConfig(type=ParamType.INT, low=1, high=10)
        config2 = SearchSpaceParamConfig(type=ParamType.INT, low=1, high=20)
        assert config1 != config2

    def test_identical_pruner_configs_equal(self):
        """Test identical PrunerConfig instances are equal."""
        config1 = PrunerConfig()
        config2 = PrunerConfig()
        assert config1 == config2

    def test_identical_hpo_configs_equal(self):
        """Test identical HPOConfig instances are equal."""
        config1 = HPOConfig()
        config2 = HPOConfig()
        assert config1 == config2


class TestBaseModelHashing:
    """Test Pydantic V2 frozen BaseModel hashing (required for frozen=True)."""

    def test_search_space_config_hashable(self, valid_int_param_config):
        """Test SearchSpaceParamConfig is hashable."""
        hash_value = hash(valid_int_param_config)
        assert isinstance(hash_value, int)

    def test_pruner_config_hashable(self, valid_pruner_config):
        """Test PrunerConfig is hashable."""
        hash_value = hash(valid_pruner_config)
        assert isinstance(hash_value, int)

    def test_sampler_config_hashable(self, valid_sampler_config):
        """Test SamplerConfig is hashable."""
        hash_value = hash(valid_sampler_config)
        assert isinstance(hash_value, int)

    def test_study_config_hashable(self, valid_study_config):
        """Test StudyConfig is hashable."""
        hash_value = hash(valid_study_config)
        assert isinstance(hash_value, int)

    def test_configs_can_be_used_in_sets(self, valid_int_param_config):
        """Test configs can be used as set members."""
        config_set = {valid_int_param_config}
        assert len(config_set) == 1

        # Adding identical config shouldn't increase size
        identical = SearchSpaceParamConfig(type=ParamType.INT, low=32, high=256, step=32)
        config_set.add(identical)
        assert len(config_set) == 1


# =============================================================================
# MODULE EXPORTS TESTS
# =============================================================================


class TestModuleExports:
    """Test module __all__ exports."""

    def test_param_type_exported(self):
        """Test ParamType is properly exported."""
        from milia_pipeline.models.hpo.hpo_config import ParamType

        assert ParamType is not None

    def test_search_space_param_config_exported(self):
        """Test SearchSpaceParamConfig is properly exported."""
        from milia_pipeline.models.hpo.hpo_config import SearchSpaceParamConfig

        assert SearchSpaceParamConfig is not None

    def test_pruner_type_exported(self):
        """Test PrunerType is properly exported."""
        from milia_pipeline.models.hpo.hpo_config import PrunerType

        assert PrunerType is not None

    def test_pruner_config_exported(self):
        """Test PrunerConfig is properly exported."""
        from milia_pipeline.models.hpo.hpo_config import PrunerConfig

        assert PrunerConfig is not None

    def test_sampler_type_exported(self):
        """Test SamplerType is properly exported."""
        from milia_pipeline.models.hpo.hpo_config import SamplerType

        assert SamplerType is not None

    def test_sampler_config_exported(self):
        """Test SamplerConfig is properly exported."""
        from milia_pipeline.models.hpo.hpo_config import SamplerConfig

        assert SamplerConfig is not None

    def test_optimization_direction_exported(self):
        """Test OptimizationDirection is properly exported."""
        from milia_pipeline.models.hpo.hpo_config import OptimizationDirection

        assert OptimizationDirection is not None

    def test_study_config_exported(self):
        """Test StudyConfig is properly exported."""
        from milia_pipeline.models.hpo.hpo_config import StudyConfig

        assert StudyConfig is not None

    def test_multi_objective_study_config_exported(self):
        """Test MultiObjectiveStudyConfig is properly exported."""
        from milia_pipeline.models.hpo.hpo_config import MultiObjectiveStudyConfig

        assert MultiObjectiveStudyConfig is not None

    def test_hpo_config_exported(self):
        """Test HPOConfig is properly exported."""
        from milia_pipeline.models.hpo.hpo_config import HPOConfig

        assert HPOConfig is not None


# =============================================================================
# TO_DICT BACKWARD COMPATIBILITY TESTS (Pydantic V2 model_dump)
# =============================================================================


class TestToDictBackwardCompatibility:
    """Test to_dict() methods provide backward compatibility with model_dump()."""

    def test_pruner_config_to_dict(self, valid_pruner_config):
        """Test PrunerConfig.to_dict() returns dict."""
        result = valid_pruner_config.to_dict()
        assert isinstance(result, dict)
        assert "type" in result
        assert "n_startup_trials" in result
        assert "n_warmup_steps" in result
        assert "interval_steps" in result

    def test_sampler_config_to_dict(self, valid_sampler_config):
        """Test SamplerConfig.to_dict() returns dict."""
        result = valid_sampler_config.to_dict()
        assert isinstance(result, dict)
        assert "type" in result
        assert "n_startup_trials" in result
        assert "multivariate" in result

    def test_study_config_to_dict(self, valid_study_config):
        """Test StudyConfig.to_dict() returns dict."""
        result = valid_study_config.to_dict()
        assert isinstance(result, dict)
        assert "direction" in result
        assert "metric" in result
        assert "study_name" in result

    def test_multi_objective_study_config_to_dict(self, valid_multi_objective_config):
        """Test MultiObjectiveStudyConfig.to_dict() returns dict."""
        result = valid_multi_objective_config.to_dict()
        assert isinstance(result, dict)
        assert "directions" in result
        assert "metrics" in result
        assert "study_name" in result

    def test_hpo_config_to_dict(self, valid_hpo_config):
        """Test HPOConfig.to_dict() returns dict."""
        result = valid_hpo_config.to_dict()
        assert isinstance(result, dict)
        assert "enabled" in result
        assert "backend" in result
        assert "n_trials" in result
        assert "pruner" in result
        assert "sampler" in result
        assert "study" in result

    def test_to_dict_matches_model_dump(self, valid_hpo_config):
        """Test to_dict() produces same result as model_dump()."""
        to_dict_result = valid_hpo_config.to_dict()
        model_dump_result = valid_hpo_config.model_dump()
        assert to_dict_result == model_dump_result


# =============================================================================
# PYDANTIC V2 SPECIFIC TESTS
# =============================================================================


class TestPydanticV2Features:
    """Test Pydantic V2 specific features and behaviors."""

    def test_base_model_inheritance(self):
        """Test configs inherit from Pydantic BaseModel."""
        from pydantic import BaseModel

        assert issubclass(PrunerConfig, BaseModel)
        assert issubclass(SamplerConfig, BaseModel)
        assert issubclass(StudyConfig, BaseModel)
        assert issubclass(MultiObjectiveStudyConfig, BaseModel)
        assert issubclass(HPOConfig, BaseModel)

    def test_model_fields_attribute(self, valid_hpo_config):
        """Test model_fields attribute exists (Pydantic V2)."""
        assert hasattr(HPOConfig, "model_fields")
        assert "enabled" in HPOConfig.model_fields
        assert "backend" in HPOConfig.model_fields
        assert "n_trials" in HPOConfig.model_fields

    def test_model_validate_method(self):
        """Test model_validate class method works (Pydantic V2)."""
        config = HPOConfig.model_validate({"enabled": True, "n_trials": 50})
        assert config.enabled is True
        assert config.n_trials == 50

    def test_model_dump_json(self, valid_hpo_config):
        """Test model_dump_json works (Pydantic V2)."""
        json_str = valid_hpo_config.model_dump_json()
        assert isinstance(json_str, str)
        assert "enabled" in json_str
        assert "backend" in json_str

    def test_frozen_config_is_hashable(self, valid_hpo_config):
        """Test frozen config hashability depends on field types.

        Note: HPOConfig contains search_space (Dict) which is mutable/unhashable,
        so HPOConfig itself cannot be hashed. However, configs without mutable
        container fields (like PrunerConfig, SamplerConfig) can be hashed.
        """
        # PrunerConfig has no mutable containers - should be hashable
        pruner = PrunerConfig()
        hash_value = hash(pruner)
        assert isinstance(hash_value, int)

        # SamplerConfig has no mutable containers - should be hashable
        sampler = SamplerConfig()
        hash_value = hash(sampler)
        assert isinstance(hash_value, int)

        # StudyConfig has no mutable containers - should be hashable
        study = StudyConfig()
        hash_value = hash(study)
        assert isinstance(hash_value, int)

        # HPOConfig has search_space Dict - cannot be hashed
        # This is expected Pydantic V2 behavior for frozen models with unhashable fields
        with pytest.raises(TypeError, match="unhashable"):
            hash(valid_hpo_config)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
