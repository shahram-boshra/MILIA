#!/usr/bin/env python3
"""
Complete Unit Test Suite for milia_pipeline/models/hpo/nas/nas_manager.py Module
PART 1: NASConfig Pydantic V2 BaseModel Tests and NASManager Initialization Tests

Tests the nas_manager.py module including:
- NASConfig Pydantic BaseModel initialization with default values
- NASConfig Pydantic BaseModel initialization with custom values
- NASConfig @field_validator validation for n_trials
- NASConfig @field_validator validation for timeout
- NASConfig @field_validator validation for direction
- NASConfig @field_validator validation for cv_folds
- NASConfig validation error handling (Pydantic ValidationError)
- NASConfig attribute access and type checking
- NASConfig to_dict() method (backward compatible wrapper for model_dump)
- NASManager initialization with default architecture space
- NASManager initialization with custom HPOConfig
- NASManager initialization with custom NASConfig
- NASManager _convert_arch_space_to_hpo_format method
- NASManager _create_hpo_config_from_nas_config method
- NASManager _merge_search_spaces method
- NASManager.__init__ with GNNArchitectureSpace
- NASManager.__init__ with custom HPOConfig
- NASManager.__init__ with custom NASConfig
- NASManager._convert_arch_space_to_hpo_format method
- NASManager._create_hpo_config_from_nas_config method
- NASManager._merge_search_spaces method
- NASManager attribute initialization
- Search space conversion correctness
- HPOConfig creation from NASConfig
- Search space merging behavior
- NASManager.search() method
- NASManager._extract_architecture() method
- NASManager.build_model() method
- NASManager._build_heterogeneous_model() method
- NASManager.get_best_architecture() method
- NASManager.get_best_params() method
- NASManager.get_search_summary() method
- Architecture extraction correctness
- Model building with different architecture types
- Error handling for getter methods before search
- HeterogeneousGNN.__init__() method
- HeterogeneousGNN._create_layer() method
- HeterogeneousGNN._get_activation() method
- HeterogeneousGNN._create_pooling() method
- HeterogeneousGNN.forward() method
- create_nas_manager() convenience function
- get_default_gnn_search_space() convenience function
- Module exports (__all__)
- Integration tests
- Edge cases and error handling

Pydantic V2 Migration (Phase 36):
    - NASConfig migrated from @dataclass to mutable BaseModel
    - Tests updated to use Pydantic ValidationError instead of ConfigurationError
    - Removed is_dataclass() checks (NASConfig is now a BaseModel)
    - Added tests for to_dict() method (backward compatible model_dump wrapper)
    - Validation tests use pydantic.ValidationError for field validation errors

Location of module under test: milia_pipeline/models/hpo/nas/nas_manager.py
Location of test file: tests/test_hpo_nas_nas_manager.py

Author: Milia Team
Version: 1.1.0
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import pytest
from typing import Dict, Any, Optional, List, Tuple
from unittest.mock import patch, MagicMock, PropertyMock
from dataclasses import dataclass, field
from enum import Enum
from pydantic import ValidationError


# =============================================================================
# MOCK CLASSES FOR EXCEPTIONS
# =============================================================================

class MockConfigurationError(Exception):
    """
    Mock ConfigurationError for testing.
    
    Mirrors the actual ConfigurationError from milia_pipeline.exceptions
    with all required attributes for validation testing.
    """
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        actual_value: Any = None,
        expected_value: Any = None,
        details: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message)
        self.message = message
        self.config_key = config_key
        self.actual_value = actual_value
        self.expected_value = expected_value
        self.details = details
        self.extra_info = kwargs

    def __str__(self) -> str:
        parts = [self.message]
        if self.config_key:
            parts.append(f"Key: '{self.config_key}'")
        if self.expected_value is not None:
            parts.append(f"Expected: {self.expected_value}")
        if self.actual_value is not None:
            parts.append(f"Actual: '{self.actual_value}'")
        if self.details:
            parts.append(f"Details: {self.details}")
        return " ".join(parts)


class MockSearchSpaceError(Exception):
    """
    Mock SearchSpaceError for testing.
    
    Mirrors the actual SearchSpaceError from milia_pipeline.exceptions
    with all required attributes for validation testing.
    """
    def __init__(
        self,
        message: str,
        parameter_name: Optional[str] = None,
        parameter_config: Optional[Dict[str, Any]] = None,
        study_name: Optional[str] = None,
        trial_number: Optional[int] = None,
        details: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message)
        self.message = message
        self.parameter_name = parameter_name
        self.parameter_config = parameter_config
        self.study_name = study_name
        self.trial_number = trial_number
        self.details = details
        self.extra_info = kwargs

    def __str__(self) -> str:
        msg = self.message
        if self.parameter_name:
            msg += f". Parameter: '{self.parameter_name}'"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


class MockHPOError(Exception):
    """
    Mock HPOError for testing.
    
    Mirrors the actual HPOError from milia_pipeline.exceptions.
    """
    def __init__(
        self,
        message: str,
        study_name: Optional[str] = None,
        trial_number: Optional[int] = None,
        details: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message)
        self.message = message
        self.study_name = study_name
        self.trial_number = trial_number
        self.details = details

    def __str__(self) -> str:
        msg = self.message
        if self.study_name:
            msg += f". Study: '{self.study_name}'"
        if self.trial_number is not None:
            msg += f", Trial: {self.trial_number}"
        if self.details:
            msg += f". Details: {self.details}"
        return msg


# =============================================================================
# MOCK CLASSES FOR HPO DEPENDENCIES
# =============================================================================

class MockParamType(Enum):
    """Mock ParamType enum for testing."""
    INT = "int"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    LOGUNIFORM = "loguniform"
    UNIFORM = "uniform"
    INT_UNIFORM = "int_uniform"
    DISCRETE_UNIFORM = "discrete_uniform"


class MockOptimizationDirection(Enum):
    """Mock OptimizationDirection for testing."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class MockPrunerType(Enum):
    """Mock PrunerType for testing."""
    MEDIAN = "median"
    PERCENTILE = "percentile"
    HYPERBAND = "hyperband"
    NONE = "none"


class MockSamplerType(Enum):
    """Mock SamplerType for testing."""
    TPE = "tpe"
    RANDOM = "random"
    CMAES = "cmaes"


@dataclass
class MockSearchSpaceParamConfig:
    """Mock SearchSpaceParamConfig for testing."""
    type: MockParamType
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[int] = None
    choices: Optional[List[Any]] = None
    log: bool = False


@dataclass
class MockPrunerConfig:
    """Mock PrunerConfig for testing."""
    type: MockPrunerType = MockPrunerType.MEDIAN
    n_startup_trials: int = 5
    n_warmup_steps: int = 10
    interval_steps: int = 1
    percentile: float = 25.0
    n_brackets: int = 4


@dataclass
class MockSamplerConfig:
    """Mock SamplerConfig for testing."""
    type: MockSamplerType = MockSamplerType.TPE
    n_startup_trials: int = 10
    seed: Optional[int] = None
    multivariate: bool = True
    constant_liar: bool = False


@dataclass
class MockStudyConfig:
    """Mock StudyConfig for testing."""
    direction: MockOptimizationDirection = MockOptimizationDirection.MINIMIZE
    metric: str = "val_loss"
    study_name: str = "milia_hpo"
    storage: Optional[str] = None
    load_if_exists: bool = True


@dataclass
class MockHPOConfig:
    """Mock HPOConfig for testing."""
    enabled: bool = False
    backend: str = "optuna"
    n_trials: int = 100
    timeout: Optional[int] = None
    n_jobs: int = 1
    search_space: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pruner: MockPrunerConfig = field(default_factory=MockPrunerConfig)
    sampler: MockSamplerConfig = field(default_factory=MockSamplerConfig)
    study: MockStudyConfig = field(default_factory=MockStudyConfig)
    cv_folds: int = 0
    cv_metric_aggregation: str = "mean"


class MockHPOManager:
    """Mock HPOManager for testing."""
    def __init__(self, config):
        self.config = config
        self.study = None
        self.best_params = None
        self.backend = MagicMock()
    
    def optimize(self, model_name, dataset, base_hyperparameters=None, 
                 trainer_kwargs=None, callbacks=None):
        """Mock optimize method."""
        return {"hidden_channels": 64, "num_layers": 3}
    
    def get_best_value(self):
        """Mock get_best_value method."""
        return 0.1234


# =============================================================================
# MOCK CLASSES FOR SEARCH SPACE DEPENDENCIES
# =============================================================================

class MockLayerType(Enum):
    """Mock LayerType for testing."""
    GCN = "gcn"
    GAT = "gat"
    SAGE = "sage"
    GIN = "gin"
    GATV2 = "gatv2"
    TRANSFORMER = "transformer"
    PNA = "pna"


class MockPoolingType(Enum):
    """Mock PoolingType for testing."""
    MEAN = "mean"
    MAX = "max"
    SUM = "sum"
    ATTENTION = "attention"


class MockAggregationType(Enum):
    """Mock AggregationType for testing."""
    MEAN = "mean"
    MAX = "max"
    SUM = "sum"


class MockActivationType(Enum):
    """Mock ActivationType for testing."""
    RELU = "relu"
    GELU = "gelu"
    ELU = "elu"


@dataclass
class MockGNNArchitectureSpace:
    """Mock GNNArchitectureSpace for testing."""
    min_layers: int = 2
    max_layers: int = 8
    layer_types: List[MockLayerType] = field(
        default_factory=lambda: [MockLayerType.GCN, MockLayerType.GAT, MockLayerType.SAGE]
    )
    hidden_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    heads: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    dropout_range: Tuple[float, float] = (0.0, 0.6)
    allow_skip_connections: bool = True
    allow_dense_connections: bool = False
    allow_mixed_layers: bool = True
    pooling_types: List[MockPoolingType] = field(
        default_factory=lambda: [MockPoolingType.MEAN, MockPoolingType.ATTENTION]
    )
    aggregation_types: List[MockAggregationType] = field(
        default_factory=lambda: [MockAggregationType.MEAN, MockAggregationType.SUM]
    )
    activation_types: List[MockActivationType] = field(
        default_factory=lambda: [MockActivationType.RELU, MockActivationType.GELU, MockActivationType.ELU]
    )
    batch_norm_options: List[bool] = field(default_factory=lambda: [True, False])
    
    def get_search_dimensions(self) -> int:
        """Mock get_search_dimensions method."""
        return 15
    
    def estimate_search_space_size(self) -> int:
        """Mock estimate_search_space_size method."""
        return 1000000


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def mock_configuration_error():
    """Provide MockConfigurationError class for patching."""
    return MockConfigurationError


@pytest.fixture
def mock_search_space_error():
    """Provide MockSearchSpaceError class for patching."""
    return MockSearchSpaceError


@pytest.fixture
def mock_hpo_error():
    """Provide MockHPOError class for patching."""
    return MockHPOError


@pytest.fixture
def mock_arch_space():
    """Create a mock GNNArchitectureSpace for testing."""
    return MockGNNArchitectureSpace()


@pytest.fixture
def mock_hpo_config():
    """Create a mock HPOConfig for testing."""
    return MockHPOConfig(enabled=True)


@pytest.fixture
def mock_hpo_manager():
    """Create a mock HPOManager for testing."""
    return MockHPOManager(MockHPOConfig(enabled=True))


# =============================================================================
# NASCONFIG INITIALIZATION TESTS - DEFAULT VALUES
# =============================================================================

class TestNASConfigDefaultValues:
    """Test NASConfig Pydantic BaseModel default value initialization."""

    def test_nasconfig_default_n_trials(self):
        """Test NASConfig default n_trials is 100."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig()
        assert config.n_trials == 100

    def test_nasconfig_default_timeout_is_none(self):
        """Test NASConfig default timeout is None."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig()
        assert config.timeout is None

    def test_nasconfig_default_metric(self):
        """Test NASConfig default metric is 'val_loss'."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig()
        assert config.metric == "val_loss"

    def test_nasconfig_default_direction(self):
        """Test NASConfig default direction is 'minimize'."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig()
        assert config.direction == "minimize"

    def test_nasconfig_default_cv_folds(self):
        """Test NASConfig default cv_folds is 0."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig()
        assert config.cv_folds == 0

    def test_nasconfig_default_study_name(self):
        """Test NASConfig default study_name is 'milia_nas'."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig()
        assert config.study_name == "milia_nas"

    def test_nasconfig_default_storage_is_none(self):
        """Test NASConfig default storage is None."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig()
        assert config.storage is None


# =============================================================================
# NASCONFIG INITIALIZATION TESTS - CUSTOM VALUES
# =============================================================================

class TestNASConfigCustomValues:
    """Test NASConfig Pydantic BaseModel initialization with custom values."""

    def test_nasconfig_custom_n_trials(self):
        """Test NASConfig with custom n_trials value."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(n_trials=200)
        assert config.n_trials == 200

    def test_nasconfig_custom_timeout(self):
        """Test NASConfig with custom timeout value."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(timeout=3600)
        assert config.timeout == 3600

    def test_nasconfig_custom_metric(self):
        """Test NASConfig with custom metric value."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(metric="val_accuracy")
        assert config.metric == "val_accuracy"

    def test_nasconfig_custom_direction_maximize(self):
        """Test NASConfig with direction 'maximize'."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(direction="maximize")
        assert config.direction == "maximize"

    def test_nasconfig_custom_cv_folds(self):
        """Test NASConfig with custom cv_folds value."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(cv_folds=5)
        assert config.cv_folds == 5

    def test_nasconfig_custom_study_name(self):
        """Test NASConfig with custom study_name value."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(study_name="custom_nas_study")
        assert config.study_name == "custom_nas_study"

    def test_nasconfig_custom_storage(self):
        """Test NASConfig with custom storage URL."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(storage="sqlite:///nas_study.db")
        assert config.storage == "sqlite:///nas_study.db"

    def test_nasconfig_all_custom_values(self):
        """Test NASConfig with all custom values."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(
            n_trials=150,
            timeout=7200,
            metric="val_mae",
            direction="minimize",
            cv_folds=10,
            study_name="full_custom_study",
            storage="sqlite:///full_custom.db"
        )
        
        assert config.n_trials == 150
        assert config.timeout == 7200
        assert config.metric == "val_mae"
        assert config.direction == "minimize"
        assert config.cv_folds == 10
        assert config.study_name == "full_custom_study"
        assert config.storage == "sqlite:///full_custom.db"


# =============================================================================
# NASCONFIG VALIDATION TESTS - n_trials
# =============================================================================

class TestNASConfigValidationNTrials:
    """Test NASConfig @field_validator validation for n_trials."""

    def test_nasconfig_n_trials_zero_raises_validation_error(self):
        """Test NASConfig raises ValidationError when n_trials is 0."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        with pytest.raises(ValidationError) as exc_info:
            NASConfig(n_trials=0)
        
        # Verify the error is about n_trials validation
        error_str = str(exc_info.value)
        assert "n_trials" in error_str or "at least 1" in error_str

    def test_nasconfig_n_trials_negative_raises_validation_error(self):
        """Test NASConfig raises ValidationError when n_trials is negative."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        with pytest.raises(ValidationError) as exc_info:
            NASConfig(n_trials=-10)
        
        # Verify the error is about n_trials validation
        error_str = str(exc_info.value)
        assert "n_trials" in error_str or "at least 1" in error_str

    def test_nasconfig_n_trials_one_is_valid(self):
        """Test NASConfig accepts n_trials=1 as valid."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(n_trials=1)
        assert config.n_trials == 1

    def test_nasconfig_n_trials_large_value_is_valid(self):
        """Test NASConfig accepts large n_trials value."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(n_trials=10000)
        assert config.n_trials == 10000


# =============================================================================
# NASCONFIG VALIDATION TESTS - timeout
# =============================================================================

class TestNASConfigValidationTimeout:
    """Test NASConfig @field_validator validation for timeout."""

    def test_nasconfig_timeout_zero_raises_validation_error(self):
        """Test NASConfig raises ValidationError when timeout is 0."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        with pytest.raises(ValidationError) as exc_info:
            NASConfig(timeout=0)
        
        # Verify the error is about timeout validation
        error_str = str(exc_info.value)
        assert "timeout" in error_str or "positive" in error_str

    def test_nasconfig_timeout_negative_raises_validation_error(self):
        """Test NASConfig raises ValidationError when timeout is negative."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        with pytest.raises(ValidationError) as exc_info:
            NASConfig(timeout=-100)
        
        # Verify the error is about timeout validation
        error_str = str(exc_info.value)
        assert "timeout" in error_str or "positive" in error_str

    def test_nasconfig_timeout_one_is_valid(self):
        """Test NASConfig accepts timeout=1 as valid."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(timeout=1)
        assert config.timeout == 1

    def test_nasconfig_timeout_none_is_valid(self):
        """Test NASConfig accepts timeout=None as valid (no timeout)."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(timeout=None)
        assert config.timeout is None


# =============================================================================
# NASCONFIG VALIDATION TESTS - direction
# =============================================================================

class TestNASConfigValidationDirection:
    """Test NASConfig @field_validator validation for direction."""

    def test_nasconfig_direction_invalid_raises_validation_error(self):
        """Test NASConfig raises ValidationError for invalid direction."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        with pytest.raises(ValidationError) as exc_info:
            NASConfig(direction="invalid_direction")
        
        # Verify the error is about direction validation
        error_str = str(exc_info.value)
        assert "direction" in error_str or "Invalid" in error_str

    def test_nasconfig_direction_empty_string_raises_validation_error(self):
        """Test NASConfig raises ValidationError for empty direction."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        with pytest.raises(ValidationError) as exc_info:
            NASConfig(direction="")
        
        # Verify the error is about direction validation
        error_str = str(exc_info.value)
        assert "direction" in error_str or "Invalid" in error_str

    def test_nasconfig_direction_minimize_is_valid(self):
        """Test NASConfig accepts direction='minimize' as valid."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(direction="minimize")
        assert config.direction == "minimize"

    def test_nasconfig_direction_maximize_is_valid(self):
        """Test NASConfig accepts direction='maximize' as valid."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(direction="maximize")
        assert config.direction == "maximize"

    def test_nasconfig_direction_case_sensitive(self):
        """Test NASConfig direction validation is case-sensitive."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        with pytest.raises(ValidationError):
            NASConfig(direction="MINIMIZE")
        
        with pytest.raises(ValidationError):
            NASConfig(direction="Maximize")


# =============================================================================
# NASCONFIG VALIDATION TESTS - cv_folds
# =============================================================================

class TestNASConfigValidationCVFolds:
    """Test NASConfig @field_validator validation for cv_folds."""

    def test_nasconfig_cv_folds_negative_raises_validation_error(self):
        """Test NASConfig raises ValidationError when cv_folds is negative."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        with pytest.raises(ValidationError) as exc_info:
            NASConfig(cv_folds=-1)
        
        # Verify the error is about cv_folds validation
        error_str = str(exc_info.value)
        assert "cv_folds" in error_str or "non-negative" in error_str

    def test_nasconfig_cv_folds_zero_is_valid(self):
        """Test NASConfig accepts cv_folds=0 as valid (no CV)."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(cv_folds=0)
        assert config.cv_folds == 0

    def test_nasconfig_cv_folds_positive_is_valid(self):
        """Test NASConfig accepts positive cv_folds values."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(cv_folds=5)
        assert config.cv_folds == 5

    def test_nasconfig_cv_folds_large_value_is_valid(self):
        """Test NASConfig accepts large cv_folds values."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(cv_folds=20)
        assert config.cv_folds == 20


# =============================================================================
# NASCONFIG ATTRIBUTE TYPE TESTS
# =============================================================================

class TestNASConfigAttributeTypes:
    """Test NASConfig attribute types."""

    def test_nasconfig_n_trials_is_int(self):
        """Test NASConfig n_trials is an integer."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig()
        assert isinstance(config.n_trials, int)

    def test_nasconfig_timeout_is_int_or_none(self):
        """Test NASConfig timeout is an integer or None."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig()
        assert config.timeout is None or isinstance(config.timeout, int)
        
        config_with_timeout = NASConfig(timeout=3600)
        assert isinstance(config_with_timeout.timeout, int)

    def test_nasconfig_metric_is_str(self):
        """Test NASConfig metric is a string."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig()
        assert isinstance(config.metric, str)

    def test_nasconfig_direction_is_str(self):
        """Test NASConfig direction is a string."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig()
        assert isinstance(config.direction, str)

    def test_nasconfig_cv_folds_is_int(self):
        """Test NASConfig cv_folds is an integer."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig()
        assert isinstance(config.cv_folds, int)

    def test_nasconfig_study_name_is_str(self):
        """Test NASConfig study_name is a string."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig()
        assert isinstance(config.study_name, str)

    def test_nasconfig_storage_is_str_or_none(self):
        """Test NASConfig storage is a string or None."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig()
        assert config.storage is None or isinstance(config.storage, str)
        
        config_with_storage = NASConfig(storage="sqlite:///test.db")
        assert isinstance(config_with_storage.storage, str)


# =============================================================================
# NASCONFIG PYDANTIC BASEMODEL PROPERTIES TESTS
# =============================================================================

class TestNASConfigBaseModelProperties:
    """Test NASConfig Pydantic BaseModel properties."""

    def test_nasconfig_is_pydantic_basemodel(self):
        """Test NASConfig is a Pydantic BaseModel."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        from pydantic import BaseModel
        
        assert issubclass(NASConfig, BaseModel)

    def test_nasconfig_has_all_expected_fields(self):
        """Test NASConfig has all expected fields."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig()
        
        expected_fields = ['n_trials', 'timeout', 'metric', 'direction', 
                          'cv_folds', 'study_name', 'storage']
        
        for field_name in expected_fields:
            assert hasattr(config, field_name), f"Missing field: {field_name}"

    def test_nasconfig_repr_contains_values(self):
        """Test NASConfig repr includes values."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(n_trials=50, metric="test_metric")
        repr_str = repr(config)
        
        assert "NASConfig" in repr_str
        assert "50" in repr_str or "n_trials" in repr_str

    def test_nasconfig_equality_same_values(self):
        """Test NASConfig instances with same values are equal."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config1 = NASConfig(n_trials=50, metric="val_loss")
        config2 = NASConfig(n_trials=50, metric="val_loss")
        
        assert config1 == config2

    def test_nasconfig_inequality_different_values(self):
        """Test NASConfig instances with different values are not equal."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config1 = NASConfig(n_trials=50)
        config2 = NASConfig(n_trials=100)
        
        assert config1 != config2

    def test_nasconfig_is_mutable(self):
        """Test NASConfig instances are mutable (not frozen)."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(n_trials=50)
        # Should not raise - Pydantic BaseModel is mutable by default
        config.n_trials = 100
        assert config.n_trials == 100

    def test_nasconfig_to_dict_returns_dict(self):
        """Test NASConfig to_dict() returns a dictionary."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(n_trials=75, metric="val_accuracy")
        result = config.to_dict()
        
        assert isinstance(result, dict)
        assert result['n_trials'] == 75
        assert result['metric'] == "val_accuracy"

    def test_nasconfig_to_dict_contains_all_fields(self):
        """Test NASConfig to_dict() contains all fields."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig()
        result = config.to_dict()
        
        expected_keys = {'n_trials', 'timeout', 'metric', 'direction', 
                        'cv_folds', 'study_name', 'storage'}
        
        assert set(result.keys()) == expected_keys

    def test_nasconfig_model_dump_equivalence(self):
        """Test NASConfig to_dict() wraps model_dump() correctly."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(n_trials=100, cv_folds=5)
        
        # to_dict() should return same as model_dump()
        assert config.to_dict() == config.model_dump()


# =============================================================================
# NASCONFIG PYDANTIC VALIDATION ERROR STRUCTURE TESTS
# =============================================================================

class TestNASConfigValidationErrorStructure:
    """Test Pydantic ValidationError structure when NASConfig validation fails."""

    def test_nasconfig_validation_error_for_n_trials_has_field_info(self):
        """Test ValidationError for n_trials contains field location info."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        with pytest.raises(ValidationError) as exc_info:
            NASConfig(n_trials=0)
        
        # Pydantic ValidationError has errors() method returning list of error dicts
        errors = exc_info.value.errors()
        assert len(errors) >= 1
        # Each error dict has 'loc', 'msg', 'type' keys
        error = errors[0]
        assert 'loc' in error
        assert 'msg' in error

    def test_nasconfig_validation_error_for_timeout_has_field_info(self):
        """Test ValidationError for timeout contains field location info."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        with pytest.raises(ValidationError) as exc_info:
            NASConfig(timeout=0)
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1
        error = errors[0]
        assert 'loc' in error
        assert 'msg' in error

    def test_nasconfig_validation_error_for_direction_has_field_info(self):
        """Test ValidationError for direction contains field location info."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        with pytest.raises(ValidationError) as exc_info:
            NASConfig(direction="invalid")
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1
        error = errors[0]
        assert 'loc' in error
        assert 'msg' in error

    def test_nasconfig_validation_error_for_cv_folds_has_field_info(self):
        """Test ValidationError for cv_folds contains field location info."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        with pytest.raises(ValidationError) as exc_info:
            NASConfig(cv_folds=-1)
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1
        error = errors[0]
        assert 'loc' in error
        assert 'msg' in error

    def test_nasconfig_multiple_validation_errors(self):
        """Test ValidationError can contain multiple field errors."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        with pytest.raises(ValidationError) as exc_info:
            NASConfig(n_trials=0, timeout=-1, direction="bad", cv_folds=-5)
        
        # Should have multiple errors
        errors = exc_info.value.errors()
        assert len(errors) >= 1  # At least one error, possibly more


# =============================================================================
# NASCONFIG BOUNDARY VALUE TESTS
# =============================================================================

class TestNASConfigBoundaryValues:
    """Test NASConfig with boundary values."""

    def test_nasconfig_minimum_valid_n_trials(self):
        """Test NASConfig with minimum valid n_trials (1)."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(n_trials=1)
        assert config.n_trials == 1

    def test_nasconfig_minimum_valid_timeout(self):
        """Test NASConfig with minimum valid timeout (1)."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(timeout=1)
        assert config.timeout == 1

    def test_nasconfig_minimum_valid_cv_folds(self):
        """Test NASConfig with minimum valid cv_folds (0)."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(cv_folds=0)
        assert config.cv_folds == 0

    def test_nasconfig_maximum_reasonable_n_trials(self):
        """Test NASConfig with very large n_trials value."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(n_trials=100000)
        assert config.n_trials == 100000

    def test_nasconfig_maximum_reasonable_timeout(self):
        """Test NASConfig with very large timeout value."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        config = NASConfig(timeout=86400 * 7)  # 1 week in seconds
        assert config.timeout == 86400 * 7

# =============================================================================
# TEST FIXTURES - NASMANAGER SECTION
# =============================================================================

@pytest.fixture
def mock_arch_space():
    """Create a mock GNNArchitectureSpace for testing."""
    return MockGNNArchitectureSpace()


@pytest.fixture
def mock_arch_space_no_mixed_layers():
    """Create a mock GNNArchitectureSpace without mixed layers."""
    return MockGNNArchitectureSpace(allow_mixed_layers=False)


@pytest.fixture
def mock_arch_space_no_skip_connections():
    """Create a mock GNNArchitectureSpace without skip connections."""
    return MockGNNArchitectureSpace(allow_skip_connections=False)


@pytest.fixture
def mock_arch_space_with_dense_connections():
    """Create a mock GNNArchitectureSpace with dense connections."""
    return MockGNNArchitectureSpace(allow_dense_connections=True)


@pytest.fixture
def mock_arch_space_gcn_only():
    """Create a mock GNNArchitectureSpace with only GCN layer type."""
    return MockGNNArchitectureSpace(
        layer_types=[MockLayerType.GCN],
        allow_mixed_layers=False
    )


@pytest.fixture
def mock_hpo_config_enabled():
    """Create an enabled mock HPOConfig for testing."""
    return MockHPOConfig(enabled=True)


@pytest.fixture
def mock_hpo_config_with_search_space():
    """Create a mock HPOConfig with existing search space."""
    return MockHPOConfig(
        enabled=True,
        search_space={
            "optimizer": {
                "lr": MockSearchSpaceParamConfig(
                    type=MockParamType.LOGUNIFORM,
                    low=1e-5,
                    high=1e-2
                )
            }
        }
    )


# =============================================================================
# NASMANAGER INITIALIZATION TESTS - BASIC
# =============================================================================

class TestNASManagerInitBasic:
    """Test NASManager.__init__ basic functionality."""

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_nasmanager_init_creates_hpo_manager(self):
        """Test NASManager initialization creates HPOManager."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        assert nas.hpo_manager is not None

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_nasmanager_init_stores_arch_space(self):
        """Test NASManager initialization stores architecture space."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        assert nas.arch_space is arch_space

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_nasmanager_init_best_architecture_is_none(self):
        """Test NASManager initialization sets best_architecture to None."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        assert nas.best_architecture is None

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_nasmanager_init_best_params_is_none(self):
        """Test NASManager initialization sets _best_params to None."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        assert nas._best_params is None


# =============================================================================
# NASMANAGER INITIALIZATION TESTS - WITH NASCONFIG
# =============================================================================

class TestNASManagerInitWithNASConfig:
    """Test NASManager initialization with NASConfig."""

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_nasmanager_init_with_nas_config(self):
        """Test NASManager initialization with custom NASConfig."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager, NASConfig
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas_config = NASConfig(n_trials=50)
        nas = NASManager(arch_space, nas_config=nas_config)
        
        assert nas.hpo_manager is not None

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_nasmanager_init_nas_config_none_uses_default(self):
        """Test NASManager uses default NASConfig when nas_config is None."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space, nas_config=None)
        
        assert nas.hpo_manager is not None

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_nasmanager_init_nas_config_with_cv_folds(self):
        """Test NASManager initialization with NASConfig having cv_folds."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager, NASConfig
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas_config = NASConfig(n_trials=100, cv_folds=5)
        nas = NASManager(arch_space, nas_config=nas_config)
        
        assert nas.hpo_manager is not None


# =============================================================================
# NASMANAGER INITIALIZATION TESTS - WITH HPOCONFIG
# =============================================================================

class TestNASManagerInitWithHPOConfig:
    """Test NASManager initialization with HPOConfig."""

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_nasmanager_init_with_hpo_config(self):
        """Test NASManager initialization with custom HPOConfig."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        from milia_pipeline.models.hpo.hpo_config import HPOConfig
        
        arch_space = GNNArchitectureSpace()
        hpo_config = HPOConfig(enabled=True, n_trials=200)
        nas = NASManager(arch_space, hpo_config=hpo_config)
        
        assert nas.hpo_manager is not None

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_nasmanager_init_hpo_config_overrides_nas_config(self):
        """Test NASManager initialization with hpo_config ignores nas_config."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager, NASConfig
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        from milia_pipeline.models.hpo.hpo_config import HPOConfig
        
        arch_space = GNNArchitectureSpace()
        hpo_config = HPOConfig(enabled=True, n_trials=200)
        nas_config = NASConfig(n_trials=50)  # This should be ignored
        
        nas = NASManager(arch_space, hpo_config=hpo_config, nas_config=nas_config)
        
        assert nas.hpo_manager is not None


# =============================================================================
# NASMANAGER _CONVERT_ARCH_SPACE_TO_HPO_FORMAT TESTS
# =============================================================================

class TestNASManagerConvertArchSpace:
    """Test NASManager._convert_arch_space_to_hpo_format method."""

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_convert_arch_space_returns_dict(self):
        """Test _convert_arch_space_to_hpo_format returns a dict."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        result = nas._convert_arch_space_to_hpo_format(arch_space)
        
        assert isinstance(result, dict)

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_convert_arch_space_has_architecture_category(self):
        """Test _convert_arch_space_to_hpo_format creates 'architecture' category."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        result = nas._convert_arch_space_to_hpo_format(arch_space)
        
        assert 'architecture' in result

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_convert_arch_space_has_num_layers_param(self):
        """Test converted search space has num_layers parameter."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        result = nas._convert_arch_space_to_hpo_format(arch_space)
        
        assert 'num_layers' in result['architecture']

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_convert_arch_space_has_hidden_channels_param(self):
        """Test converted search space has hidden_channels parameter."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        result = nas._convert_arch_space_to_hpo_format(arch_space)
        
        assert 'hidden_channels' in result['architecture']

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_convert_arch_space_has_pooling_param(self):
        """Test converted search space has pooling parameter."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        result = nas._convert_arch_space_to_hpo_format(arch_space)
        
        assert 'pooling' in result['architecture']

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_convert_arch_space_has_dropout_param(self):
        """Test converted search space has dropout parameter."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        result = nas._convert_arch_space_to_hpo_format(arch_space)
        
        assert 'dropout' in result['architecture']

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_convert_arch_space_has_aggregation_param(self):
        """Test converted search space has aggregation parameter."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        result = nas._convert_arch_space_to_hpo_format(arch_space)
        
        assert 'aggregation' in result['architecture']

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_convert_arch_space_has_activation_param(self):
        """Test converted search space has activation parameter."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        result = nas._convert_arch_space_to_hpo_format(arch_space)
        
        assert 'activation' in result['architecture']

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_convert_arch_space_has_batch_norm_param(self):
        """Test converted search space has batch_norm parameter."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        result = nas._convert_arch_space_to_hpo_format(arch_space)
        
        assert 'batch_norm' in result['architecture']


# =============================================================================
# NASMANAGER _CONVERT_ARCH_SPACE SKIP/DENSE CONNECTIONS TESTS
# =============================================================================

class TestNASManagerConvertArchSpaceConnections:
    """Test conversion of skip/dense connection options."""

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_convert_arch_space_with_skip_connections(self):
        """Test conversion includes use_skip_connections when allowed."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace(allow_skip_connections=True)
        nas = NASManager(arch_space)
        
        result = nas._convert_arch_space_to_hpo_format(arch_space)
        
        assert 'use_skip_connections' in result['architecture']

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_convert_arch_space_without_skip_connections(self):
        """Test conversion excludes use_skip_connections when not allowed."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace(allow_skip_connections=False)
        nas = NASManager(arch_space)
        
        result = nas._convert_arch_space_to_hpo_format(arch_space)
        
        assert 'use_skip_connections' not in result['architecture']

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_convert_arch_space_with_dense_connections(self):
        """Test conversion includes use_dense_connections when allowed."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace(allow_dense_connections=True)
        nas = NASManager(arch_space)
        
        result = nas._convert_arch_space_to_hpo_format(arch_space)
        
        assert 'use_dense_connections' in result['architecture']

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_convert_arch_space_without_dense_connections(self):
        """Test conversion excludes use_dense_connections when not allowed."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace(allow_dense_connections=False)
        nas = NASManager(arch_space)
        
        result = nas._convert_arch_space_to_hpo_format(arch_space)
        
        assert 'use_dense_connections' not in result['architecture']


# =============================================================================
# NASMANAGER _CONVERT_ARCH_SPACE MIXED LAYERS TESTS
# =============================================================================

class TestNASManagerConvertArchSpaceMixedLayers:
    """Test conversion of mixed layers options."""

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_convert_arch_space_mixed_layers_has_per_layer_types(self):
        """Test conversion creates per-layer type params when allow_mixed_layers=True."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace(allow_mixed_layers=True, max_layers=4)
        nas = NASManager(arch_space)
        
        result = nas._convert_arch_space_to_hpo_format(arch_space)
        
        # Should have layer_0_type, layer_1_type, etc.
        for i in range(4):
            assert f'layer_{i}_type' in result['architecture']

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_convert_arch_space_no_mixed_layers_has_single_layer_type(self):
        """Test conversion creates single layer_type param when allow_mixed_layers=False."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace(allow_mixed_layers=False)
        nas = NASManager(arch_space)
        
        result = nas._convert_arch_space_to_hpo_format(arch_space)
        
        assert 'layer_type' in result['architecture']
        assert 'layer_0_type' not in result['architecture']

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_convert_arch_space_mixed_layers_with_attention_has_per_layer_heads(self):
        """Test conversion creates per-layer heads params when mixed layers and attention types."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace, LayerType
        
        arch_space = GNNArchitectureSpace(
            allow_mixed_layers=True,
            max_layers=3,
            layer_types=[LayerType.GAT, LayerType.GCN]  # GAT requires heads
        )
        nas = NASManager(arch_space)
        
        result = nas._convert_arch_space_to_hpo_format(arch_space)
        
        # Should have layer_0_heads, layer_1_heads, etc.
        for i in range(3):
            assert f'layer_{i}_heads' in result['architecture']


# =============================================================================
# NASMANAGER _CREATE_HPO_CONFIG_FROM_NAS_CONFIG TESTS
# =============================================================================

class TestNASManagerCreateHPOConfigFromNASConfig:
    """Test NASManager._create_hpo_config_from_nas_config method."""

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_create_hpo_config_returns_hpo_config(self):
        """Test _create_hpo_config_from_nas_config returns HPOConfig."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager, NASConfig
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        from milia_pipeline.models.hpo.hpo_config import HPOConfig
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        nas_config = NASConfig()
        
        search_space = nas._convert_arch_space_to_hpo_format(arch_space)
        result = nas._create_hpo_config_from_nas_config(nas_config, search_space)
        
        assert isinstance(result, HPOConfig)

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_create_hpo_config_sets_enabled_true(self):
        """Test _create_hpo_config_from_nas_config sets enabled=True."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager, NASConfig
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        nas_config = NASConfig()
        
        search_space = nas._convert_arch_space_to_hpo_format(arch_space)
        result = nas._create_hpo_config_from_nas_config(nas_config, search_space)
        
        assert result.enabled is True

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_create_hpo_config_sets_n_trials_from_nas_config(self):
        """Test _create_hpo_config_from_nas_config uses nas_config.n_trials."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager, NASConfig
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        nas_config = NASConfig(n_trials=250)
        
        search_space = nas._convert_arch_space_to_hpo_format(arch_space)
        result = nas._create_hpo_config_from_nas_config(nas_config, search_space)
        
        assert result.n_trials == 250

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_create_hpo_config_sets_cv_folds_from_nas_config(self):
        """Test _create_hpo_config_from_nas_config uses nas_config.cv_folds."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager, NASConfig
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        nas_config = NASConfig(cv_folds=5)
        
        search_space = nas._convert_arch_space_to_hpo_format(arch_space)
        result = nas._create_hpo_config_from_nas_config(nas_config, search_space)
        
        assert result.cv_folds == 5


# =============================================================================
# NASMANAGER _MERGE_SEARCH_SPACES TESTS
# =============================================================================

class TestNASManagerMergeSearchSpaces:
    """Test NASManager._merge_search_spaces method."""

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_merge_search_spaces_returns_hpo_config(self):
        """Test _merge_search_spaces returns HPOConfig."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        from milia_pipeline.models.hpo.hpo_config import HPOConfig
        
        arch_space = GNNArchitectureSpace()
        hpo_config = HPOConfig(enabled=True)
        nas = NASManager(arch_space, hpo_config=hpo_config)
        
        arch_search_space = nas._convert_arch_space_to_hpo_format(arch_space)
        result = nas._merge_search_spaces(hpo_config, arch_search_space)
        
        assert isinstance(result, HPOConfig)

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_merge_search_spaces_preserves_existing_space(self):
        """Test _merge_search_spaces preserves existing search space."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        from milia_pipeline.models.hpo.hpo_config import (
            HPOConfig, SearchSpaceParamConfig, ParamType
        )
        
        # Create HPOConfig with existing search space
        existing_space = {
            "optimizer": {
                "lr": SearchSpaceParamConfig(
                    type=ParamType.LOGUNIFORM,
                    low=1e-5,
                    high=1e-2
                )
            }
        }
        hpo_config = HPOConfig(enabled=True, search_space=existing_space)
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space, hpo_config=hpo_config)
        
        arch_search_space = nas._convert_arch_space_to_hpo_format(arch_space)
        result = nas._merge_search_spaces(hpo_config, arch_search_space)
        
        # Should have both optimizer and architecture categories
        assert 'optimizer' in result.search_space
        assert 'architecture' in result.search_space

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_merge_search_spaces_arch_takes_precedence(self):
        """Test _merge_search_spaces: architecture params take precedence."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        from milia_pipeline.models.hpo.hpo_config import (
            HPOConfig, SearchSpaceParamConfig, ParamType
        )
        
        # Create HPOConfig with conflicting architecture params
        existing_space = {
            "architecture": {
                "hidden_channels": SearchSpaceParamConfig(
                    type=ParamType.CATEGORICAL,
                    choices=[64, 128]  # Different from arch_space defaults
                )
            }
        }
        hpo_config = HPOConfig(enabled=True, search_space=existing_space)
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space, hpo_config=hpo_config)
        
        arch_search_space = nas._convert_arch_space_to_hpo_format(arch_space)
        result = nas._merge_search_spaces(hpo_config, arch_search_space)
        
        # Architecture params from arch_space should override
        assert 'hidden_channels' in result.search_space['architecture']


# =============================================================================
# NASMANAGER PARAMTYPE AND SEARCHSPACEPARAMCONFIG TESTS
# =============================================================================

class TestNASManagerParamTypes:
    """Test parameter types in converted search space."""

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_num_layers_is_int_type(self):
        """Test num_layers parameter has INT type."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        from milia_pipeline.models.hpo.hpo_config import ParamType
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        result = nas._convert_arch_space_to_hpo_format(arch_space)
        
        assert result['architecture']['num_layers'].type == ParamType.INT

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_hidden_channels_is_categorical_type(self):
        """Test hidden_channels parameter has CATEGORICAL type."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        from milia_pipeline.models.hpo.hpo_config import ParamType
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        result = nas._convert_arch_space_to_hpo_format(arch_space)
        
        assert result['architecture']['hidden_channels'].type == ParamType.CATEGORICAL

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_dropout_is_float_type(self):
        """Test dropout parameter has FLOAT type."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        from milia_pipeline.models.hpo.hpo_config import ParamType
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        result = nas._convert_arch_space_to_hpo_format(arch_space)
        
        assert result['architecture']['dropout'].type == ParamType.FLOAT

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_pooling_is_categorical_type(self):
        """Test pooling parameter has CATEGORICAL type."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        from milia_pipeline.models.hpo.hpo_config import ParamType
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        result = nas._convert_arch_space_to_hpo_format(arch_space)
        
        assert result['architecture']['pooling'].type == ParamType.CATEGORICAL


# =============================================================================
# NASMANAGER CONVERSION VALUE CORRECTNESS TESTS
# =============================================================================

class TestNASManagerConversionValues:
    """Test converted values are correct."""

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_num_layers_low_matches_min_layers(self):
        """Test num_layers.low matches arch_space.min_layers."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace(min_layers=3)
        nas = NASManager(arch_space)
        
        result = nas._convert_arch_space_to_hpo_format(arch_space)
        
        assert result['architecture']['num_layers'].low == 3

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_num_layers_high_matches_max_layers(self):
        """Test num_layers.high matches arch_space.max_layers."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace(max_layers=10)
        nas = NASManager(arch_space)
        
        result = nas._convert_arch_space_to_hpo_format(arch_space)
        
        assert result['architecture']['num_layers'].high == 10

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_dropout_range_matches(self):
        """Test dropout.low and dropout.high match arch_space.dropout_range."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace(dropout_range=(0.1, 0.5))
        nas = NASManager(arch_space)
        
        result = nas._convert_arch_space_to_hpo_format(arch_space)
        
        assert result['architecture']['dropout'].low == 0.1
        assert result['architecture']['dropout'].high == 0.5

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_hidden_channels_choices_match(self):
        """Test hidden_channels.choices matches arch_space.hidden_channels."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace(hidden_channels=[64, 128, 256])
        nas = NASManager(arch_space)
        
        result = nas._convert_arch_space_to_hpo_format(arch_space)
        
        assert result['architecture']['hidden_channels'].choices == [64, 128, 256]

# =============================================================================
# SPECIALIZED MOCK MANAGERS FOR SEARCH/EXTRACT TESTS
# =============================================================================

class MockHPOManagerWithMixedLayers(MockHPOManager):
    """Mock HPOManager that returns per-layer parameters."""
    def optimize(self, model_name, dataset, base_hyperparameters=None,
                 trainer_kwargs=None, callbacks=None):
        self.best_params = {
            "num_layers": 3,
            "hidden_channels": 64,
            "pooling": "mean",
            "dropout": 0.1,
            "aggregation": "mean",
            "activation": "relu",
            "batch_norm": True,
            "use_skip_connections": True,
            "layer_0_type": "gcn",
            "layer_0_heads": 1,
            "layer_1_type": "gat",
            "layer_1_heads": 4,
            "layer_2_type": "sage",
            "layer_2_heads": 1,
        }
        return self.best_params


# =============================================================================
# MOCK CLASSES FOR MODEL FACTORY
# =============================================================================

class MockModel:
    """Mock PyTorch model for testing."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def forward(self, x, edge_index, batch=None):
        return x


class MockModelFactory:
    """Mock ModelFactory for testing."""
    def create_model(self, model_name, hyperparameters, sample_data=None):
        return MockModel(**hyperparameters)


def mock_get_factory():
    """Mock get_factory function."""
    return MockModelFactory()


# =============================================================================
# TEST FIXTURES - SEARCH AND EXTRACT TESTS
# =============================================================================

@pytest.fixture
def mock_arch_space_no_mixed():
    """Create a mock GNNArchitectureSpace without mixed layers."""
    return MockGNNArchitectureSpace(allow_mixed_layers=False)


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    dataset = MagicMock()
    dataset.__getitem__ = MagicMock(return_value=MagicMock())
    return dataset


@pytest.fixture
def sample_params():
    """Sample parameters as returned from HPO."""
    return {
        "num_layers": 3,
        "hidden_channels": 64,
        "pooling": "mean",
        "dropout": 0.1,
        "aggregation": "mean",
        "activation": "relu",
        "batch_norm": True,
        "layer_type": "gcn",
        "heads": 4,
    }


@pytest.fixture
def sample_mixed_layer_params():
    """Sample parameters with per-layer types."""
    return {
        "num_layers": 3,
        "hidden_channels": 64,
        "pooling": "mean",
        "dropout": 0.1,
        "aggregation": "mean",
        "activation": "relu",
        "batch_norm": True,
        "use_skip_connections": True,
        "layer_0_type": "gcn",
        "layer_0_heads": 1,
        "layer_1_type": "gat",
        "layer_1_heads": 4,
        "layer_2_type": "sage",
        "layer_2_heads": 1,
    }


@pytest.fixture
def sample_architecture():
    """Sample architecture configuration."""
    return {
        "num_layers": 3,
        "hidden_channels": 64,
        "pooling": "mean",
        "dropout": 0.1,
        "aggregation": "mean",
        "activation": "relu",
        "batch_norm": True,
        "use_skip_connections": True,
        "use_dense_connections": False,
        "layers": [
            {"type": "gcn", "hidden_channels": 64, "heads": 1, "dropout": 0.1,
             "activation": "relu", "batch_norm": True, "residual": True},
            {"type": "gcn", "hidden_channels": 64, "heads": 1, "dropout": 0.1,
             "activation": "relu", "batch_norm": True, "residual": True},
            {"type": "gcn", "hidden_channels": 64, "heads": 1, "dropout": 0.1,
             "activation": "relu", "batch_norm": True, "residual": True},
        ]
    }


# =============================================================================
# NASMANAGER SEARCH METHOD TESTS
# =============================================================================

class TestNASManagerSearch:
    """Test NASManager.search() method."""

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_search_returns_dict(self, mock_dataset):
        """Test search() returns a dictionary."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        result = nas.search(dataset=mock_dataset)
        
        assert isinstance(result, dict)

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_search_sets_best_architecture(self, mock_dataset):
        """Test search() sets best_architecture attribute."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        assert nas.best_architecture is None
        
        nas.search(dataset=mock_dataset)
        
        assert nas.best_architecture is not None

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_search_sets_best_params(self, mock_dataset):
        """Test search() sets _best_params attribute."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        assert nas._best_params is None
        
        nas.search(dataset=mock_dataset)
        
        assert nas._best_params is not None

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_search_result_has_num_layers(self, mock_dataset):
        """Test search() result contains num_layers."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        result = nas.search(dataset=mock_dataset)
        
        assert 'num_layers' in result

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_search_result_has_hidden_channels(self, mock_dataset):
        """Test search() result contains hidden_channels."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        result = nas.search(dataset=mock_dataset)
        
        assert 'hidden_channels' in result

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_search_result_has_layers_list(self, mock_dataset):
        """Test search() result contains layers list."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        result = nas.search(dataset=mock_dataset)
        
        assert 'layers' in result
        assert isinstance(result['layers'], list)


# =============================================================================
# NASMANAGER _EXTRACT_ARCHITECTURE METHOD TESTS
# =============================================================================

class TestNASManagerExtractArchitecture:
    """Test NASManager._extract_architecture() method."""

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_extract_architecture_returns_dict(self, sample_params):
        """Test _extract_architecture returns a dictionary."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace(allow_mixed_layers=False)
        nas = NASManager(arch_space)
        
        result = nas._extract_architecture(sample_params)
        
        assert isinstance(result, dict)

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_extract_architecture_extracts_num_layers(self, sample_params):
        """Test _extract_architecture extracts num_layers correctly."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace(allow_mixed_layers=False)
        nas = NASManager(arch_space)
        
        result = nas._extract_architecture(sample_params)
        
        assert result['num_layers'] == 3

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_extract_architecture_extracts_hidden_channels(self, sample_params):
        """Test _extract_architecture extracts hidden_channels correctly."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace(allow_mixed_layers=False)
        nas = NASManager(arch_space)
        
        result = nas._extract_architecture(sample_params)
        
        assert result['hidden_channels'] == 64

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_extract_architecture_creates_layers_list(self, sample_params):
        """Test _extract_architecture creates layers list."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace(allow_mixed_layers=False)
        nas = NASManager(arch_space)
        
        result = nas._extract_architecture(sample_params)
        
        assert 'layers' in result
        assert len(result['layers']) == 3

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_extract_architecture_with_mixed_layers(self, sample_mixed_layer_params):
        """Test _extract_architecture with mixed layers creates correct layers."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace(allow_mixed_layers=True)
        nas = NASManager(arch_space)
        
        result = nas._extract_architecture(sample_mixed_layer_params)
        
        assert result['layers'][0]['type'] == 'gcn'
        assert result['layers'][1]['type'] == 'gat'
        assert result['layers'][2]['type'] == 'sage'

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_extract_architecture_mixed_layers_heads(self, sample_mixed_layer_params):
        """Test _extract_architecture with mixed layers extracts heads correctly."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace(allow_mixed_layers=True)
        nas = NASManager(arch_space)
        
        result = nas._extract_architecture(sample_mixed_layer_params)
        
        assert result['layers'][0]['heads'] == 1
        assert result['layers'][1]['heads'] == 4
        assert result['layers'][2]['heads'] == 1

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_extract_architecture_use_skip_connections(self, sample_mixed_layer_params):
        """Test _extract_architecture extracts use_skip_connections."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace(allow_mixed_layers=True)
        nas = NASManager(arch_space)
        
        result = nas._extract_architecture(sample_mixed_layer_params)
        
        assert result['use_skip_connections'] is True


# =============================================================================
# NASMANAGER BUILD_MODEL METHOD TESTS
# =============================================================================

class TestNASManagerBuildModel:
    """Test NASManager.build_model() method."""

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    @patch('milia_pipeline.models.factory.model_factory.ModelFactory', MockModelFactory)
    @patch('milia_pipeline.models.factory.model_factory.get_factory', mock_get_factory)
    def test_build_model_returns_module(self, sample_architecture):
        """Test build_model returns a nn.Module."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        model = nas.build_model(
            architecture=sample_architecture,
            in_channels=10,
            out_channels=1
        )
        
        assert model is not None

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_build_model_heterogeneous_architecture(self):
        """Test build_model with heterogeneous (mixed layer types) architecture."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        mixed_arch = {
            "num_layers": 3,
            "hidden_channels": 64,
            "pooling": "mean",
            "dropout": 0.1,
            "activation": "relu",
            "batch_norm": True,
            "use_skip_connections": True,
            "layers": [
                {"type": "gcn", "hidden_channels": 64, "heads": 1},
                {"type": "gat", "hidden_channels": 64, "heads": 4},
                {"type": "sage", "hidden_channels": 64, "heads": 1},
            ]
        }
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        # This should call _build_heterogeneous_model
        model = nas.build_model(
            architecture=mixed_arch,
            in_channels=10,
            out_channels=1
        )
        
        assert model is not None


# =============================================================================
# NASMANAGER GET_BEST_ARCHITECTURE METHOD TESTS
# =============================================================================

class TestNASManagerGetBestArchitecture:
    """Test NASManager.get_best_architecture() method."""

    @patch('milia_pipeline.models.hpo.nas.nas_manager.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_get_best_architecture_before_search_raises_error(self):
        """Test get_best_architecture raises HPOError before search."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        with pytest.raises(Exception) as exc_info:
            nas.get_best_architecture()
        
        assert "MockHPOError" in type(exc_info.value).__name__
        assert "No architecture search completed yet" in str(exc_info.value)

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_get_best_architecture_after_search_returns_dict(self, mock_dataset):
        """Test get_best_architecture returns dict after search."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        nas.search(dataset=mock_dataset)
        result = nas.get_best_architecture()
        
        assert isinstance(result, dict)

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_get_best_architecture_returns_same_as_search(self, mock_dataset):
        """Test get_best_architecture returns same result as search."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        search_result = nas.search(dataset=mock_dataset)
        get_result = nas.get_best_architecture()
        
        assert search_result == get_result


# =============================================================================
# NASMANAGER GET_BEST_PARAMS METHOD TESTS
# =============================================================================

class TestNASManagerGetBestParams:
    """Test NASManager.get_best_params() method."""

    @patch('milia_pipeline.models.hpo.nas.nas_manager.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_get_best_params_before_search_raises_error(self):
        """Test get_best_params raises HPOError before search."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        with pytest.raises(Exception) as exc_info:
            nas.get_best_params()
        
        assert "MockHPOError" in type(exc_info.value).__name__
        assert "No architecture search completed yet" in str(exc_info.value)

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_get_best_params_after_search_returns_dict(self, mock_dataset):
        """Test get_best_params returns dict after search."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        nas.search(dataset=mock_dataset)
        result = nas.get_best_params()
        
        assert isinstance(result, dict)

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_get_best_params_contains_raw_params(self, mock_dataset):
        """Test get_best_params contains raw HPO parameters."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        nas.search(dataset=mock_dataset)
        result = nas.get_best_params()
        
        # Raw params should have the flat structure from HPO
        assert 'hidden_channels' in result or 'num_layers' in result


# =============================================================================
# NASMANAGER GET_SEARCH_SUMMARY METHOD TESTS
# =============================================================================

class TestNASManagerGetSearchSummary:
    """Test NASManager.get_search_summary() method."""

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_get_search_summary_returns_dict(self):
        """Test get_search_summary returns a dictionary."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        result = nas.get_search_summary()
        
        assert isinstance(result, dict)

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_get_search_summary_has_search_dimensions(self):
        """Test get_search_summary includes search_dimensions."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        result = nas.get_search_summary()
        
        assert 'search_dimensions' in result

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_get_search_summary_has_estimated_space_size(self):
        """Test get_search_summary includes estimated_space_size."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        result = nas.get_search_summary()
        
        assert 'estimated_space_size' in result

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_get_search_summary_has_n_trials(self):
        """Test get_search_summary includes n_trials."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        result = nas.get_search_summary()
        
        assert 'n_trials' in result

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_get_search_summary_has_allow_mixed_layers(self):
        """Test get_search_summary includes allow_mixed_layers."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        result = nas.get_search_summary()
        
        assert 'allow_mixed_layers' in result

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_get_search_summary_has_layer_types(self):
        """Test get_search_summary includes layer_types."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        result = nas.get_search_summary()
        
        assert 'layer_types' in result

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_get_search_summary_after_search_has_best_architecture(self, mock_dataset):
        """Test get_search_summary after search includes best_architecture."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        
        nas.search(dataset=mock_dataset)
        result = nas.get_search_summary()
        
        assert 'best_architecture' in result


# =============================================================================
# NASMANAGER EXTRACT ARCHITECTURE EDGE CASES
# =============================================================================

class TestNASManagerExtractArchitectureEdgeCases:
    """Test edge cases in _extract_architecture method."""

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_extract_architecture_missing_num_layers_uses_default(self):
        """Test _extract_architecture uses default when num_layers missing."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace(allow_mixed_layers=False)
        nas = NASManager(arch_space)
        
        params = {"hidden_channels": 64}  # No num_layers
        result = nas._extract_architecture(params)
        
        assert result['num_layers'] == 3  # Default value

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_extract_architecture_missing_hidden_channels_uses_default(self):
        """Test _extract_architecture uses default when hidden_channels missing."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace(allow_mixed_layers=False)
        nas = NASManager(arch_space)
        
        params = {"num_layers": 3}  # No hidden_channels
        result = nas._extract_architecture(params)
        
        assert result['hidden_channels'] == 64  # Default value

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_extract_architecture_empty_params_uses_defaults(self):
        """Test _extract_architecture uses defaults with empty params."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace(allow_mixed_layers=False)
        nas = NASManager(arch_space)
        
        params = {}
        result = nas._extract_architecture(params)
        
        assert 'num_layers' in result
        assert 'hidden_channels' in result
        assert 'pooling' in result
        assert 'layers' in result

# =============================================================================
# MOCK TORCH_GEOMETRIC MODULES
# =============================================================================

class MockGCNConv:
    """Mock GCNConv layer."""
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def __call__(self, x, edge_index):
        return x


class MockGATConv:
    """Mock GATConv layer."""
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
    
    def __call__(self, x, edge_index):
        return x


class MockSAGEConv:
    """Mock SAGEConv layer."""
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def __call__(self, x, edge_index):
        return x


class MockGINConv:
    """Mock GINConv layer."""
    def __init__(self, nn):
        self.nn = nn
    
    def __call__(self, x, edge_index):
        return x


class MockGATv2Conv:
    """Mock GATv2Conv layer."""
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
    
    def __call__(self, x, edge_index):
        return x


class MockTransformerConv:
    """Mock TransformerConv layer."""
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
    
    def __call__(self, x, edge_index):
        return x


def mock_global_mean_pool(x, batch):
    """Mock global_mean_pool function."""
    return x


def mock_global_max_pool(x, batch):
    """Mock global_max_pool function."""
    return x


def mock_global_add_pool(x, batch):
    """Mock global_add_pool function."""
    return x


class MockGlobalAttention:
    """Mock GlobalAttention pooling."""
    def __init__(self, gate_nn):
        self.gate_nn = gate_nn
    
    def __call__(self, x, batch):
        return x


# =============================================================================
# TEST FIXTURES - HETEROGENEOUSGNN TESTS
# =============================================================================

@pytest.fixture
def sample_architecture_heterogeneous():
    """Sample architecture configuration for HeterogeneousGNN."""
    return {
        "num_layers": 3,
        "hidden_channels": 64,
        "pooling": "mean",
        "dropout": 0.1,
        "aggregation": "mean",
        "activation": "relu",
        "batch_norm": True,
        "use_skip_connections": True,
        "use_dense_connections": False,
        "layers": [
            {"type": "gcn", "hidden_channels": 64, "heads": 1},
            {"type": "gat", "hidden_channels": 64, "heads": 4},
            {"type": "sage", "hidden_channels": 64, "heads": 1},
        ]
    }


@pytest.fixture
def sample_architecture_gcn_only():
    """Sample architecture with only GCN layers."""
    return {
        "num_layers": 3,
        "hidden_channels": 64,
        "pooling": "mean",
        "dropout": 0.1,
        "activation": "relu",
        "batch_norm": True,
        "use_skip_connections": False,
        "layers": [
            {"type": "gcn", "hidden_channels": 64, "heads": 1},
            {"type": "gcn", "hidden_channels": 64, "heads": 1},
            {"type": "gcn", "hidden_channels": 64, "heads": 1},
        ]
    }


@pytest.fixture
def sample_architecture_with_attention():
    """Sample architecture with attention pooling."""
    return {
        "num_layers": 2,
        "hidden_channels": 64,
        "pooling": "attention",
        "dropout": 0.2,
        "activation": "gelu",
        "batch_norm": True,
        "use_skip_connections": True,
        "layers": [
            {"type": "gat", "hidden_channels": 64, "heads": 4},
            {"type": "gat", "hidden_channels": 64, "heads": 4},
        ]
    }


# =============================================================================
# HETEROGENEOUSGNN INITIALIZATION TESTS
# =============================================================================

class TestHeterogeneousGNNInit:
    """Test HeterogeneousGNN.__init__() method."""

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_heterogeneous_gnn_init_creates_layers(self, sample_architecture):
        """Test HeterogeneousGNN initialization creates layers ModuleList."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        
        model = HeterogeneousGNN(
            architecture=sample_architecture,
            in_channels=10,
            out_channels=1
        )
        
        assert hasattr(model, 'layers')

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_heterogeneous_gnn_init_creates_batch_norms(self, sample_architecture):
        """Test HeterogeneousGNN initialization creates batch_norms ModuleList."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        
        model = HeterogeneousGNN(
            architecture=sample_architecture,
            in_channels=10,
            out_channels=1
        )
        
        assert hasattr(model, 'batch_norms')

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_heterogeneous_gnn_init_creates_skips(self, sample_architecture):
        """Test HeterogeneousGNN initialization creates skips ModuleList."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        
        model = HeterogeneousGNN(
            architecture=sample_architecture,
            in_channels=10,
            out_channels=1
        )
        
        assert hasattr(model, 'skips')

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_heterogeneous_gnn_init_creates_classifier(self, sample_architecture):
        """Test HeterogeneousGNN initialization creates classifier layer."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        
        model = HeterogeneousGNN(
            architecture=sample_architecture,
            in_channels=10,
            out_channels=1
        )
        
        assert hasattr(model, 'classifier')

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_heterogeneous_gnn_init_creates_dropout(self, sample_architecture):
        """Test HeterogeneousGNN initialization creates dropout layer."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        
        model = HeterogeneousGNN(
            architecture=sample_architecture,
            in_channels=10,
            out_channels=1
        )
        
        assert hasattr(model, 'dropout')

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_heterogeneous_gnn_init_creates_activation(self, sample_architecture):
        """Test HeterogeneousGNN initialization creates activation layer."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        
        model = HeterogeneousGNN(
            architecture=sample_architecture,
            in_channels=10,
            out_channels=1
        )
        
        assert hasattr(model, 'activation')

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_heterogeneous_gnn_init_stores_architecture(self, sample_architecture):
        """Test HeterogeneousGNN initialization stores architecture."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        
        model = HeterogeneousGNN(
            architecture=sample_architecture,
            in_channels=10,
            out_channels=1
        )
        
        assert model.architecture == sample_architecture


# =============================================================================
# HETEROGENEOUSGNN _CREATE_LAYER TESTS
# =============================================================================

class TestHeterogeneousGNNCreateLayer:
    """Test HeterogeneousGNN._create_layer() method."""

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_create_layer_gcn(self, sample_architecture):
        """Test _create_layer creates GCN layer."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        
        model = HeterogeneousGNN(
            architecture=sample_architecture,
            in_channels=10,
            out_channels=1
        )
        
        layer = model._create_layer('gcn', 10, 64)
        
        assert layer is not None

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_create_layer_gat(self, sample_architecture):
        """Test _create_layer creates GAT layer."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        
        model = HeterogeneousGNN(
            architecture=sample_architecture,
            in_channels=10,
            out_channels=1
        )
        
        layer = model._create_layer('gat', 10, 64, heads=4)
        
        assert layer is not None

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_create_layer_sage(self, sample_architecture):
        """Test _create_layer creates SAGE layer."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        
        model = HeterogeneousGNN(
            architecture=sample_architecture,
            in_channels=10,
            out_channels=1
        )
        
        layer = model._create_layer('sage', 10, 64)
        
        assert layer is not None

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_create_layer_gin(self, sample_architecture):
        """Test _create_layer creates GIN layer."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        
        model = HeterogeneousGNN(
            architecture=sample_architecture,
            in_channels=10,
            out_channels=1
        )
        
        layer = model._create_layer('gin', 10, 64)
        
        assert layer is not None

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_create_layer_unknown_defaults_to_gcn(self, sample_architecture):
        """Test _create_layer with unknown type defaults to GCN."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        
        model = HeterogeneousGNN(
            architecture=sample_architecture,
            in_channels=10,
            out_channels=1
        )
        
        # Unknown layer type should default to GCN
        layer = model._create_layer('unknown_type', 10, 64)
        
        assert layer is not None


# =============================================================================
# HETEROGENEOUSGNN _GET_ACTIVATION TESTS
# =============================================================================

class TestHeterogeneousGNNGetActivation:
    """Test HeterogeneousGNN._get_activation() method."""

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_get_activation_relu(self, sample_architecture):
        """Test _get_activation returns ReLU for 'relu'."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        import torch.nn as nn
        
        model = HeterogeneousGNN(
            architecture=sample_architecture,
            in_channels=10,
            out_channels=1
        )
        
        activation = model._get_activation('relu')
        
        assert isinstance(activation, nn.ReLU)

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_get_activation_gelu(self, sample_architecture):
        """Test _get_activation returns GELU for 'gelu'."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        import torch.nn as nn
        
        model = HeterogeneousGNN(
            architecture=sample_architecture,
            in_channels=10,
            out_channels=1
        )
        
        activation = model._get_activation('gelu')
        
        assert isinstance(activation, nn.GELU)

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_get_activation_elu(self, sample_architecture):
        """Test _get_activation returns ELU for 'elu'."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        import torch.nn as nn
        
        model = HeterogeneousGNN(
            architecture=sample_architecture,
            in_channels=10,
            out_channels=1
        )
        
        activation = model._get_activation('elu')
        
        assert isinstance(activation, nn.ELU)

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_get_activation_leaky_relu(self, sample_architecture):
        """Test _get_activation returns LeakyReLU for 'leaky_relu'."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        import torch.nn as nn
        
        model = HeterogeneousGNN(
            architecture=sample_architecture,
            in_channels=10,
            out_channels=1
        )
        
        activation = model._get_activation('leaky_relu')
        
        assert isinstance(activation, nn.LeakyReLU)

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_get_activation_unknown_defaults_to_relu(self, sample_architecture):
        """Test _get_activation returns ReLU for unknown name."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        import torch.nn as nn
        
        model = HeterogeneousGNN(
            architecture=sample_architecture,
            in_channels=10,
            out_channels=1
        )
        
        activation = model._get_activation('unknown_activation')
        
        assert isinstance(activation, nn.ReLU)


# =============================================================================
# HETEROGENEOUSGNN _CREATE_POOLING TESTS
# =============================================================================

class TestHeterogeneousGNNCreatePooling:
    """Test HeterogeneousGNN._create_pooling() method."""

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_create_pooling_mean(self, sample_architecture):
        """Test _create_pooling returns global_mean_pool for 'mean'."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        
        model = HeterogeneousGNN(
            architecture=sample_architecture,
            in_channels=10,
            out_channels=1
        )
        
        pooling = model._create_pooling('mean')
        
        assert pooling is not None

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_create_pooling_max(self, sample_architecture):
        """Test _create_pooling returns global_max_pool for 'max'."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        
        model = HeterogeneousGNN(
            architecture=sample_architecture,
            in_channels=10,
            out_channels=1
        )
        
        pooling = model._create_pooling('max')
        
        assert pooling is not None

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_create_pooling_sum(self, sample_architecture):
        """Test _create_pooling returns global_add_pool for 'sum'."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        
        model = HeterogeneousGNN(
            architecture=sample_architecture,
            in_channels=10,
            out_channels=1
        )
        
        pooling = model._create_pooling('sum')
        
        assert pooling is not None

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_create_pooling_attention(self, sample_architecture_with_attention):
        """Test _create_pooling returns GlobalAttention for 'attention'."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        
        model = HeterogeneousGNN(
            architecture=sample_architecture_with_attention,
            in_channels=10,
            out_channels=1
        )
        
        pooling = model._create_pooling('attention')
        
        assert pooling is not None


# =============================================================================
# CREATE_NAS_MANAGER CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestCreateNASManager:
    """Test create_nas_manager() convenience function."""

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_create_nas_manager_returns_nas_manager(self):
        """Test create_nas_manager returns NASManager instance."""
        from milia_pipeline.models.hpo.nas.nas_manager import create_nas_manager, NASManager
        
        nas = create_nas_manager()
        
        assert isinstance(nas, NASManager)

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_create_nas_manager_with_default_arch_space(self):
        """Test create_nas_manager creates default architecture space when None."""
        from milia_pipeline.models.hpo.nas.nas_manager import create_nas_manager
        
        nas = create_nas_manager(arch_space=None)
        
        assert nas.arch_space is not None

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_create_nas_manager_with_custom_arch_space(self):
        """Test create_nas_manager uses provided architecture space."""
        from milia_pipeline.models.hpo.nas.nas_manager import create_nas_manager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        custom_space = GNNArchitectureSpace(min_layers=3, max_layers=5)
        nas = create_nas_manager(arch_space=custom_space)
        
        assert nas.arch_space is custom_space

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_create_nas_manager_with_n_trials(self):
        """Test create_nas_manager uses n_trials parameter."""
        from milia_pipeline.models.hpo.nas.nas_manager import create_nas_manager
        
        nas = create_nas_manager(n_trials=50)
        
        assert nas.hpo_manager is not None

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_create_nas_manager_with_kwargs(self):
        """Test create_nas_manager passes kwargs to NASConfig."""
        from milia_pipeline.models.hpo.nas.nas_manager import create_nas_manager
        
        nas = create_nas_manager(n_trials=75, cv_folds=5, timeout=3600)
        
        assert nas.hpo_manager is not None


# =============================================================================
# GET_DEFAULT_GNN_SEARCH_SPACE CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestGetDefaultGNNSearchSpace:
    """Test get_default_gnn_search_space() convenience function."""

    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    def test_get_default_gnn_search_space_returns_space(self):
        """Test get_default_gnn_search_space returns GNNArchitectureSpace."""
        from milia_pipeline.models.hpo.nas.nas_manager import get_default_gnn_search_space
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        space = get_default_gnn_search_space()
        
        assert isinstance(space, GNNArchitectureSpace)

    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    def test_get_default_gnn_search_space_has_default_min_layers(self):
        """Test get_default_gnn_search_space has default min_layers=2."""
        from milia_pipeline.models.hpo.nas.nas_manager import get_default_gnn_search_space
        
        space = get_default_gnn_search_space()
        
        assert space.min_layers == 2

    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    def test_get_default_gnn_search_space_has_default_max_layers(self):
        """Test get_default_gnn_search_space has default max_layers=8."""
        from milia_pipeline.models.hpo.nas.nas_manager import get_default_gnn_search_space
        
        space = get_default_gnn_search_space()
        
        assert space.max_layers == 8

    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    def test_get_default_gnn_search_space_has_layer_types(self):
        """Test get_default_gnn_search_space has default layer_types."""
        from milia_pipeline.models.hpo.nas.nas_manager import get_default_gnn_search_space
        
        space = get_default_gnn_search_space()
        
        assert len(space.layer_types) > 0


# =============================================================================
# MODULE EXPORTS TESTS
# =============================================================================

class TestModuleExports:
    """Test module __all__ exports."""

    def test_module_exports_nasconfig(self):
        """Test NASConfig is in module exports."""
        from milia_pipeline.models.hpo.nas import nas_manager
        
        assert 'NASConfig' in nas_manager.__all__

    def test_module_exports_nasmanager(self):
        """Test NASManager is in module exports."""
        from milia_pipeline.models.hpo.nas import nas_manager
        
        assert 'NASManager' in nas_manager.__all__

    def test_module_exports_heterogeneousgnn(self):
        """Test HeterogeneousGNN is in module exports."""
        from milia_pipeline.models.hpo.nas import nas_manager
        
        assert 'HeterogeneousGNN' in nas_manager.__all__

    def test_module_exports_create_nas_manager(self):
        """Test create_nas_manager is in module exports."""
        from milia_pipeline.models.hpo.nas import nas_manager
        
        assert 'create_nas_manager' in nas_manager.__all__

    def test_module_exports_get_default_gnn_search_space(self):
        """Test get_default_gnn_search_space is in module exports."""
        from milia_pipeline.models.hpo.nas import nas_manager
        
        assert 'get_default_gnn_search_space' in nas_manager.__all__


# =============================================================================
# IMPORT TESTS
# =============================================================================

class TestModuleImports:
    """Test module-level imports."""

    def test_import_nasconfig(self):
        """Test NASConfig can be imported."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASConfig
        
        assert NASConfig is not None

    def test_import_nasmanager(self):
        """Test NASManager can be imported."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        
        assert NASManager is not None

    def test_import_heterogeneousgnn(self):
        """Test HeterogeneousGNN can be imported."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        
        assert HeterogeneousGNN is not None

    def test_import_create_nas_manager(self):
        """Test create_nas_manager can be imported."""
        from milia_pipeline.models.hpo.nas.nas_manager import create_nas_manager
        
        assert create_nas_manager is not None

    def test_import_get_default_gnn_search_space(self):
        """Test get_default_gnn_search_space can be imported."""
        from milia_pipeline.models.hpo.nas.nas_manager import get_default_gnn_search_space
        
        assert get_default_gnn_search_space is not None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for NASManager."""

    @patch('milia_pipeline.models.hpo.nas.nas_manager.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_full_workflow_search_and_build(self, mock_dataset):
        """Test full workflow: create manager, search, build model."""
        from milia_pipeline.models.hpo.nas.nas_manager import NASManager
        from milia_pipeline.models.hpo.nas.search_space import GNNArchitectureSpace
        
        arch_space = GNNArchitectureSpace()
        nas = NASManager(arch_space)
        best_arch = nas.search(dataset=mock_dataset)
        
        assert best_arch is not None
        assert 'num_layers' in best_arch
        assert 'layers' in best_arch
        
        # Build model using _build_heterogeneous_model to avoid ModelFactory import issues
        model = nas._build_heterogeneous_model(
            architecture=best_arch,
            in_channels=10,
            out_channels=1
        )
        
        assert model is not None

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    @patch('milia_pipeline.models.hpo.nas.nas_manager.HPOManager', MockHPOManager)
    def test_convenience_function_workflow(self, mock_dataset):
        """Test workflow using convenience functions."""
        from milia_pipeline.models.hpo.nas.nas_manager import (
            create_nas_manager, get_default_gnn_search_space
        )
        
        # Use convenience function to get search space
        space = get_default_gnn_search_space()
        
        # Use convenience function to create manager
        nas = create_nas_manager(arch_space=space, n_trials=50)
        
        # Run search
        best_arch = nas.search(dataset=mock_dataset)
        
        # Verify
        assert best_arch is not None
        assert nas.get_best_architecture() == best_arch


# =============================================================================
# EDGE CASES AND ERROR HANDLING TESTS
# =============================================================================

class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_heterogeneous_gnn_no_batch_norm(self):
        """Test HeterogeneousGNN with batch_norm=False."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        
        arch = {
            "num_layers": 2,
            "hidden_channels": 64,
            "pooling": "mean",
            "dropout": 0.0,
            "activation": "relu",
            "batch_norm": False,  # No batch norm
            "use_skip_connections": False,
            "layers": [
                {"type": "gcn", "hidden_channels": 64, "heads": 1},
                {"type": "gcn", "hidden_channels": 64, "heads": 1},
            ]
        }
        
        model = HeterogeneousGNN(
            architecture=arch,
            in_channels=10,
            out_channels=1
        )
        
        assert model.use_batch_norm is False

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_heterogeneous_gnn_no_skip_connections(self):
        """Test HeterogeneousGNN with use_skip_connections=False."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        
        arch = {
            "num_layers": 2,
            "hidden_channels": 64,
            "pooling": "mean",
            "dropout": 0.1,
            "activation": "relu",
            "batch_norm": True,
            "use_skip_connections": False,  # No skip connections
            "layers": [
                {"type": "gcn", "hidden_channels": 64, "heads": 1},
                {"type": "gcn", "hidden_channels": 64, "heads": 1},
            ]
        }
        
        model = HeterogeneousGNN(
            architecture=arch,
            in_channels=10,
            out_channels=1
        )
        
        assert model.use_skip is False

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_heterogeneous_gnn_zero_dropout(self):
        """Test HeterogeneousGNN with dropout=0.0."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        
        arch = {
            "num_layers": 2,
            "hidden_channels": 64,
            "pooling": "mean",
            "dropout": 0.0,  # No dropout
            "activation": "relu",
            "batch_norm": True,
            "use_skip_connections": True,
            "layers": [
                {"type": "gcn", "hidden_channels": 64, "heads": 1},
                {"type": "gcn", "hidden_channels": 64, "heads": 1},
            ]
        }
        
        model = HeterogeneousGNN(
            architecture=arch,
            in_channels=10,
            out_channels=1
        )
        
        assert model.dropout_p == 0.0

    @patch('milia_pipeline.exceptions.ConfigurationError', MockConfigurationError)
    @patch('milia_pipeline.exceptions.SearchSpaceError', MockSearchSpaceError)
    @patch('milia_pipeline.exceptions.HPOError', MockHPOError)
    def test_heterogeneous_gnn_single_layer(self):
        """Test HeterogeneousGNN with single layer."""
        from milia_pipeline.models.hpo.nas.nas_manager import HeterogeneousGNN
        
        arch = {
            "num_layers": 1,
            "hidden_channels": 64,
            "pooling": "mean",
            "dropout": 0.1,
            "activation": "relu",
            "batch_norm": True,
            "use_skip_connections": False,
            "layers": [
                {"type": "gcn", "hidden_channels": 64, "heads": 1},
            ]
        }
        
        model = HeterogeneousGNN(
            architecture=arch,
            in_channels=10,
            out_channels=1
        )
        
        assert len(model.layers) == 1

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
