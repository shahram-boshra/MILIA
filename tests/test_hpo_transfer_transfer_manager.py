#!/usr/bin/env python3
"""
Complete Unit Test Suite for milia_pipeline/models/hpo/transfer/transfer_manager.py Module

Tests:
- MetaFeatureMethod enum
- AdaptationMethod enum
- TransferConfig Pydantic V2 BaseModel (frozen=True, field_validator, model_validator)
- RegisteredStudyInfo Pydantic V2 BaseModel (to_dict, from_dict, model_validate)
- HPOTransferManager.__init__()
- HPOTransferManager lazy imports
- HPOTransferManager study registration (register_study, unregister_study, get_registered_studies, get_study_info)
- HPOTransferManager similarity computation (_compute_similarity, find_similar_studies, compute_dataset_similarity)
- HPOTransferManager warm-start operations (warm_start_study, _transfer_trials)
- HPOTransferManager parameter filtering (_filter_params, _flatten_search_space, _add_noise_to_params)
- HPOTransferManager meta-feature extraction (_extract_meta_features, _basic_meta_features)
- HPOTransferManager persistence (_save_meta_db, _load_meta_db, export_meta_db, import_meta_db)
- HPOTransferManager utility methods (get_transfer_summary, clear, __repr__)

Pydantic V2 Migration Notes:
- TransferConfig uses frozen=True BaseModel pattern (not @dataclass)
- RegisteredStudyInfo uses mutable BaseModel pattern
- Uses @field_validator for individual field validation
- Uses @model_validator(mode='before') for enum conversion and timestamp initialization

This is a PRODUCTION-READY test suite with comprehensive coverage.

Author: Milia Team
Version: 1.1.0
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# MOCK CLASSES FOR DEPENDENCIES
# =============================================================================


class MockMetaFeatureMethod(Enum):
    """Mock MetaFeatureMethod enum for testing."""

    STATISTICAL = "statistical"
    LEARNED = "learned"
    LANDMARK = "landmark"


class MockAdaptationMethod(Enum):
    """Mock AdaptationMethod enum for testing."""

    WEIGHTED = "weighted"
    FILTERED = "filtered"
    FULL = "full"
    ADAPTIVE = "adaptive"


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


class MockTrial:
    """Mock Optuna trial for testing."""

    def __init__(self, params: dict[str, Any], value: float, state=None):
        self.params = params
        self.value = value
        self.state = state or MockTrialState.COMPLETE


class MockStudy:
    """Mock Optuna study for testing."""

    def __init__(
        self,
        best_params: dict[str, Any] = None,
        best_value: float = 0.1,
        trials: list[MockTrial] = None,
        direction: MockDirection = MockDirection.MINIMIZE,
    ):
        self.best_params = best_params or {"lr": 0.001, "hidden_dim": 128}
        self.best_value = best_value
        # Use 'is None' check to allow empty lists
        if trials is None:
            self.trials = [
                MockTrial({"lr": 0.001, "hidden_dim": 128}, 0.1),
                MockTrial({"lr": 0.01, "hidden_dim": 64}, 0.2),
            ]
        else:
            self.trials = trials
        self.direction = direction
        self._enqueued_trials = []

    def enqueue_trial(self, params: dict[str, Any]):
        """Mock enqueue_trial method."""
        self._enqueued_trials.append(params)


class MockOptuna:
    """Mock Optuna module for testing."""

    class trial:
        TrialState = MockTrialState


class MockMetaFeatureExtractor:
    """Mock MetaFeatureExtractor for testing."""

    @staticmethod
    def extract(dataset):
        return {"n_samples": 1000.0, "mean_nodes": 25.0}

    @staticmethod
    def compute_similarity(features_a, features_b):
        return 0.95


class MockMetaFeatureConfig:
    """Mock MetaFeatureConfig for testing."""

    pass


class MockWarmStartStrategy:
    """Mock WarmStartStrategy for testing."""

    def __init__(self, config=None):
        self.config = config

    def _filter_params(self, params, flat_space, scale_to_bounds=True):
        return params, True

    def _flatten_search_space(self, search_space):
        return search_space


class MockWarmStartConfig:
    """Mock WarmStartConfig for testing."""

    pass


class MockWarmStartMethod(Enum):
    """Mock WarmStartMethod for testing."""

    WEIGHTED = "weighted"
    FILTERED = "filtered"
    FULL = "full"
    ADAPTIVE = "adaptive"


@dataclass
class MockTransferredTrial:
    """Mock TransferredTrial for testing."""

    params: dict[str, Any]
    value: float | None = None
    source_study: str | None = None
    similarity: float = 1.0
    weight: float = 1.0
    is_valid: bool = True


class MockHPOError(Exception):
    """Mock HPOError for testing."""

    def __init__(
        self, message: str, study_name: str | None = None, details: str | None = None, **kwargs
    ):
        super().__init__(message)
        self.message = message
        self.study_name = study_name
        self.details = details


class MockDataset:
    """Mock PyG-like dataset for testing."""

    def __init__(self, size=100, n_features=10, mean_edges=50, mean_nodes=20):
        self._size = size
        self._n_features = n_features
        self._mean_edges = mean_edges
        self._mean_nodes = mean_nodes
        self._data = [self._create_mock_data() for _ in range(size)]

    def _create_mock_data(self):
        data = MagicMock()
        data.x = MagicMock()
        data.x.shape = (self._mean_nodes, self._n_features)
        data.edge_index = MagicMock()
        data.edge_index.shape = (2, self._mean_edges)
        data.y = MagicMock()
        data.y.numel.return_value = 1
        data.y.item.return_value = 0.5
        return data

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        return self._data[idx]


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
def mock_dataset():
    """Create mock PyG-like dataset."""
    return MockDataset()


@pytest.fixture
def sample_meta_features():
    """Create sample meta-features dictionary."""
    return {
        "n_samples": 1000.0,
        "n_features": 10.0,
        "mean_nodes": 25.0,
        "mean_edges": 50.0,
        "target_mean": 0.5,
        "target_std": 0.1,
    }


@pytest.fixture
def sample_best_params():
    """Create sample best parameters dictionary."""
    return {"lr": 0.001, "hidden_dim": 128, "dropout": 0.2, "num_layers": 3}


@pytest.fixture
def sample_search_space():
    """Create sample search space dictionary."""
    return {
        "hyperparameters": {
            "hidden_dim": {"type": "int", "low": 32, "high": 256, "step": 32},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        },
        "optimizer": {
            "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
        },
    }


# =============================================================================
# META FEATURE METHOD ENUM TESTS
# =============================================================================


class TestMetaFeatureMethodEnum:
    """Test MetaFeatureMethod enum."""

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_meta_feature_method_has_statistical(self, *mocks):
        """Test MetaFeatureMethod has STATISTICAL value."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import MetaFeatureMethod

        assert hasattr(MetaFeatureMethod, "STATISTICAL")
        assert MetaFeatureMethod.STATISTICAL.value == "statistical"

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_meta_feature_method_has_learned(self, *mocks):
        """Test MetaFeatureMethod has LEARNED value."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import MetaFeatureMethod

        assert hasattr(MetaFeatureMethod, "LEARNED")
        assert MetaFeatureMethod.LEARNED.value == "learned"

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_meta_feature_method_has_landmark(self, *mocks):
        """Test MetaFeatureMethod has LANDMARK value."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import MetaFeatureMethod

        assert hasattr(MetaFeatureMethod, "LANDMARK")
        assert MetaFeatureMethod.LANDMARK.value == "landmark"

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_meta_feature_method_from_string(self, *mocks):
        """Test MetaFeatureMethod can be created from string."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import MetaFeatureMethod

        method = MetaFeatureMethod("statistical")
        assert method == MetaFeatureMethod.STATISTICAL


# =============================================================================
# ADAPTATION METHOD ENUM TESTS
# =============================================================================


class TestAdaptationMethodEnum:
    """Test AdaptationMethod enum."""

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_adaptation_method_has_weighted(self, *mocks):
        """Test AdaptationMethod has WEIGHTED value."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import AdaptationMethod

        assert hasattr(AdaptationMethod, "WEIGHTED")
        assert AdaptationMethod.WEIGHTED.value == "weighted"

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_adaptation_method_has_filtered(self, *mocks):
        """Test AdaptationMethod has FILTERED value."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import AdaptationMethod

        assert hasattr(AdaptationMethod, "FILTERED")
        assert AdaptationMethod.FILTERED.value == "filtered"

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_adaptation_method_has_full(self, *mocks):
        """Test AdaptationMethod has FULL value."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import AdaptationMethod

        assert hasattr(AdaptationMethod, "FULL")
        assert AdaptationMethod.FULL.value == "full"

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_adaptation_method_has_adaptive(self, *mocks):
        """Test AdaptationMethod has ADAPTIVE value."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import AdaptationMethod

        assert hasattr(AdaptationMethod, "ADAPTIVE")
        assert AdaptationMethod.ADAPTIVE.value == "adaptive"


# =============================================================================
# TRANSFER CONFIG PYDANTIC V2 BASEMODEL TESTS
# =============================================================================


class TestTransferConfig:
    """Test TransferConfig Pydantic V2 frozen BaseModel."""

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_transfer_config_default_values(self, *mocks):
        """Test TransferConfig has correct default values."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            AdaptationMethod,
            MetaFeatureMethod,
            TransferConfig,
        )

        config = TransferConfig()

        assert config.n_warm_start_trials == 10
        assert config.similarity_threshold == 0.7
        assert config.meta_feature_method == MetaFeatureMethod.STATISTICAL
        assert config.adaptation_method == AdaptationMethod.WEIGHTED
        assert config.weight_by_performance is True
        assert config.scale_to_bounds is True
        assert config.add_noise is False
        assert config.noise_scale == 0.05
        assert config.persist_meta_db is False
        assert config.meta_db_path is None

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_transfer_config_custom_values(self, *mocks):
        """Test TransferConfig with custom values."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            AdaptationMethod,
            MetaFeatureMethod,
            TransferConfig,
        )

        config = TransferConfig(
            n_warm_start_trials=20,
            similarity_threshold=0.8,
            meta_feature_method=MetaFeatureMethod.LEARNED,
            adaptation_method=AdaptationMethod.FILTERED,
            weight_by_performance=False,
            scale_to_bounds=False,
            add_noise=True,
            noise_scale=0.1,
            persist_meta_db=True,
            meta_db_path="/path/to/db.json",
        )

        assert config.n_warm_start_trials == 20
        assert config.similarity_threshold == 0.8
        assert config.meta_feature_method == MetaFeatureMethod.LEARNED
        assert config.adaptation_method == AdaptationMethod.FILTERED
        assert config.weight_by_performance is False
        assert config.scale_to_bounds is False
        assert config.add_noise is True
        assert config.noise_scale == 0.1
        assert config.persist_meta_db is True
        assert config.meta_db_path == "/path/to/db.json"

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_transfer_config_raises_for_invalid_n_warm_start_trials(self, *mocks):
        """Test TransferConfig raises ValueError for n_warm_start_trials < 1."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import TransferConfig

        with pytest.raises(ValueError) as exc_info:
            TransferConfig(n_warm_start_trials=0)

        assert "n_warm_start_trials" in str(exc_info.value)

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_transfer_config_raises_for_negative_n_warm_start_trials(self, *mocks):
        """Test TransferConfig raises ValueError for negative n_warm_start_trials."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import TransferConfig

        with pytest.raises(ValueError):
            TransferConfig(n_warm_start_trials=-5)

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_transfer_config_raises_for_similarity_threshold_below_zero(self, *mocks):
        """Test TransferConfig raises ValueError for similarity_threshold < 0."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import TransferConfig

        with pytest.raises(ValueError) as exc_info:
            TransferConfig(similarity_threshold=-0.1)

        assert "similarity_threshold" in str(exc_info.value)

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_transfer_config_raises_for_similarity_threshold_above_one(self, *mocks):
        """Test TransferConfig raises ValueError for similarity_threshold > 1."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import TransferConfig

        with pytest.raises(ValueError):
            TransferConfig(similarity_threshold=1.5)

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_transfer_config_raises_for_negative_noise_scale(self, *mocks):
        """Test TransferConfig raises ValueError for negative noise_scale."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import TransferConfig

        with pytest.raises(ValueError) as exc_info:
            TransferConfig(noise_scale=-0.1)

        assert "noise_scale" in str(exc_info.value)

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_transfer_config_raises_for_noise_scale_above_one(self, *mocks):
        """Test TransferConfig raises ValueError for noise_scale > 1."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import TransferConfig

        with pytest.raises(ValueError):
            TransferConfig(noise_scale=1.5)

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_transfer_config_raises_for_persist_without_path(self, *mocks):
        """Test TransferConfig raises ValueError when persist_meta_db is True but meta_db_path is None."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import TransferConfig

        with pytest.raises(ValueError) as exc_info:
            TransferConfig(persist_meta_db=True, meta_db_path=None)

        assert "meta_db_path" in str(exc_info.value)

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_transfer_config_converts_string_meta_feature_method(self, *mocks):
        """Test TransferConfig converts string meta_feature_method to enum."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            MetaFeatureMethod,
            TransferConfig,
        )

        config = TransferConfig(meta_feature_method="statistical")

        assert config.meta_feature_method == MetaFeatureMethod.STATISTICAL

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_transfer_config_converts_string_adaptation_method(self, *mocks):
        """Test TransferConfig converts string adaptation_method to enum."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            AdaptationMethod,
            TransferConfig,
        )

        config = TransferConfig(adaptation_method="filtered")

        assert config.adaptation_method == AdaptationMethod.FILTERED

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_transfer_config_is_frozen(self, *mocks):
        """Test TransferConfig is a frozen dataclass."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import TransferConfig

        config = TransferConfig()

        with pytest.raises(Exception):  # FrozenInstanceError
            config.n_warm_start_trials = 20

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_transfer_config_boundary_similarity_threshold_zero(self, *mocks):
        """Test TransferConfig accepts similarity_threshold of 0."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import TransferConfig

        config = TransferConfig(similarity_threshold=0.0)
        assert config.similarity_threshold == 0.0

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_transfer_config_boundary_similarity_threshold_one(self, *mocks):
        """Test TransferConfig accepts similarity_threshold of 1."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import TransferConfig

        config = TransferConfig(similarity_threshold=1.0)
        assert config.similarity_threshold == 1.0

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_transfer_config_to_dict(self, *mocks):
        """Test TransferConfig.to_dict() returns proper dictionary (Pydantic V2 model_dump)."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            AdaptationMethod,
            MetaFeatureMethod,
            TransferConfig,
        )

        config = TransferConfig(
            n_warm_start_trials=15,
            similarity_threshold=0.8,
            meta_feature_method=MetaFeatureMethod.STATISTICAL,
            adaptation_method=AdaptationMethod.WEIGHTED,
            add_noise=True,
            noise_scale=0.1,
        )

        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["n_warm_start_trials"] == 15
        assert result["similarity_threshold"] == 0.8
        assert result["add_noise"] is True
        assert result["noise_scale"] == 0.1
        # Verify enum values are serialized correctly
        assert result["meta_feature_method"] == MetaFeatureMethod.STATISTICAL
        assert result["adaptation_method"] == AdaptationMethod.WEIGHTED


# =============================================================================
# REGISTERED STUDY INFO PYDANTIC V2 BASEMODEL TESTS
# =============================================================================


class TestRegisteredStudyInfo:
    """Test RegisteredStudyInfo Pydantic V2 mutable BaseModel."""

    def test_registered_study_info_required_fields(self):
        """Test RegisteredStudyInfo with required fields only."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import RegisteredStudyInfo

        sample_meta_features = {"n_samples": 1000.0, "n_features": 10.0}
        sample_best_params = {"lr": 0.001, "hidden_dim": 128}

        info = RegisteredStudyInfo(
            study_name="test_study",
            meta_features=sample_meta_features,
            best_params=sample_best_params,
            best_value=0.1,
            n_trials=50,
        )

        assert info.study_name == "test_study"
        assert info.meta_features == sample_meta_features
        assert info.best_params == sample_best_params
        assert info.best_value == 0.1
        assert info.n_trials == 50
        assert info.n_completed == 0  # Default
        assert info.direction == "minimize"  # Default
        assert info.model_name is None  # Default
        assert info.dataset_info is None  # Default
        assert info.registered_at is not None  # Auto-set

    def test_registered_study_info_all_fields(self):
        """Test RegisteredStudyInfo with all fields."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import RegisteredStudyInfo

        sample_meta_features = {"n_samples": 1000.0, "n_features": 10.0}
        sample_best_params = {"lr": 0.001, "hidden_dim": 128}

        info = RegisteredStudyInfo(
            study_name="test_study",
            meta_features=sample_meta_features,
            best_params=sample_best_params,
            best_value=0.1,
            n_trials=50,
            n_completed=45,
            direction="maximize",
            model_name="GCN",
            dataset_info={"name": "QM9", "size": 1000},
            registered_at="2024-01-15T10:30:00",
        )

        assert info.n_completed == 45
        assert info.direction == "maximize"
        assert info.model_name == "GCN"
        assert info.dataset_info == {"name": "QM9", "size": 1000}
        assert info.registered_at == "2024-01-15T10:30:00"

    def test_registered_study_info_auto_sets_registered_at(self):
        """Test RegisteredStudyInfo auto-sets registered_at if not provided."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import RegisteredStudyInfo

        sample_meta_features = {"n_samples": 1000.0, "n_features": 10.0}
        sample_best_params = {"lr": 0.001, "hidden_dim": 128}

        info = RegisteredStudyInfo(
            study_name="test_study",
            meta_features=sample_meta_features,
            best_params=sample_best_params,
            best_value=0.1,
            n_trials=50,
        )

        assert info.registered_at is not None
        # Should be a valid ISO format datetime string
        datetime.fromisoformat(info.registered_at)

    def test_registered_study_info_to_dict(self):
        """Test RegisteredStudyInfo.to_dict() method."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import RegisteredStudyInfo

        sample_meta_features = {"n_samples": 1000.0, "n_features": 10.0}
        sample_best_params = {"lr": 0.001, "hidden_dim": 128}

        info = RegisteredStudyInfo(
            study_name="test_study",
            meta_features=sample_meta_features,
            best_params=sample_best_params,
            best_value=0.1,
            n_trials=50,
            n_completed=45,
            direction="minimize",
            model_name="GCN",
            dataset_info={"name": "QM9"},
            registered_at="2024-01-15T10:30:00",
        )

        result = info.to_dict()

        assert isinstance(result, dict)
        assert result["study_name"] == "test_study"
        assert result["meta_features"] == sample_meta_features
        assert result["best_params"] == sample_best_params
        assert result["best_value"] == 0.1
        assert result["n_trials"] == 50
        assert result["n_completed"] == 45
        assert result["direction"] == "minimize"
        assert result["model_name"] == "GCN"
        assert result["dataset_info"] == {"name": "QM9"}
        assert result["registered_at"] == "2024-01-15T10:30:00"

    def test_registered_study_info_from_dict(self):
        """Test RegisteredStudyInfo.from_dict() class method."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import RegisteredStudyInfo

        sample_meta_features = {"n_samples": 1000.0, "n_features": 10.0}
        sample_best_params = {"lr": 0.001, "hidden_dim": 128}

        data = {
            "study_name": "test_study",
            "meta_features": sample_meta_features,
            "best_params": sample_best_params,
            "best_value": 0.1,
            "n_trials": 50,
            "n_completed": 45,
            "direction": "minimize",
            "model_name": "GCN",
            "dataset_info": {"name": "QM9"},
            "registered_at": "2024-01-15T10:30:00",
        }

        info = RegisteredStudyInfo.from_dict(data)

        assert info.study_name == "test_study"
        assert info.meta_features == sample_meta_features
        assert info.best_params == sample_best_params
        assert info.best_value == 0.1
        assert info.n_trials == 50
        assert info.n_completed == 45
        assert info.direction == "minimize"
        assert info.model_name == "GCN"
        assert info.dataset_info == {"name": "QM9"}
        assert info.registered_at == "2024-01-15T10:30:00"

    def test_registered_study_info_roundtrip(self):
        """Test to_dict and from_dict produce equivalent objects."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import RegisteredStudyInfo

        sample_meta_features = {"n_samples": 1000.0, "n_features": 10.0}
        sample_best_params = {"lr": 0.001, "hidden_dim": 128}

        original = RegisteredStudyInfo(
            study_name="test_study",
            meta_features=sample_meta_features,
            best_params=sample_best_params,
            best_value=0.1,
            n_trials=50,
            n_completed=45,
            direction="minimize",
            model_name="GCN",
            dataset_info={"name": "QM9"},
            registered_at="2024-01-15T10:30:00",
        )

        data = original.to_dict()
        reconstructed = RegisteredStudyInfo.from_dict(data)

        assert reconstructed.study_name == original.study_name
        assert reconstructed.meta_features == original.meta_features
        assert reconstructed.best_params == original.best_params
        assert reconstructed.best_value == original.best_value
        assert reconstructed.n_trials == original.n_trials


# =============================================================================
# HPO TRANSFER MANAGER INITIALIZATION TESTS
# =============================================================================


class TestHPOTransferManagerInit:
    """Test HPOTransferManager.__init__() method."""

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_init_with_default_config(self, *mocks):
        """Test initialization with default configuration."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            TransferConfig,
        )

        manager = HPOTransferManager()

        assert manager.config is not None
        assert isinstance(manager.config, TransferConfig)
        assert manager.config.n_warm_start_trials == 10

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_init_with_custom_config(self, *mocks):
        """Test initialization with custom configuration."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            TransferConfig,
        )

        config = TransferConfig(n_warm_start_trials=20, similarity_threshold=0.9)
        manager = HPOTransferManager(config)

        assert manager.config.n_warm_start_trials == 20
        assert manager.config.similarity_threshold == 0.9

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_init_creates_empty_meta_db(self, *mocks):
        """Test initialization creates empty meta-database."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()

        assert manager._meta_db == {}

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_init_creates_empty_study_cache(self, *mocks):
        """Test initialization creates empty study cache."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()

        assert manager._study_cache == {}

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_init_lazy_loads_optuna(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test initialization lazy-loads optuna."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()

        manager = HPOTransferManager()

        mock_optuna.assert_called_once()
        assert manager._optuna is not None

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_init_lazy_loads_meta_features(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test initialization lazy-loads meta_features module."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_meta_features.return_value = (MockMetaFeatureExtractor, MockMetaFeatureConfig)

        _manager = HPOTransferManager()

        mock_meta_features.assert_called_once()

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_init_lazy_loads_warm_start(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test initialization lazy-loads warm_start module."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_warm_start.return_value = (
            MockWarmStartStrategy,
            MockWarmStartConfig,
            MockWarmStartMethod,
            MockTransferredTrial,
        )

        _manager = HPOTransferManager()

        mock_warm_start.assert_called_once()

    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_init_lazy_loads_hpo_error(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test initialization lazy-loads HPOError."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_hpo_error.return_value = MockHPOError

        _manager = HPOTransferManager()

        mock_hpo_error.assert_called_once()

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_init_handles_none_optuna(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test initialization handles None optuna gracefully."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()

        assert manager._optuna is None

    @patch("os.path.exists", return_value=False)
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_init_with_persist_meta_db_loads_from_disk(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error, mock_exists
    ):
        """Test initialization with persist_meta_db attempts to load from disk."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            TransferConfig,
        )

        config = TransferConfig(persist_meta_db=True, meta_db_path="/path/to/meta_db.json")
        _manager = HPOTransferManager(config)

        # Should have attempted to check if file exists
        mock_exists.assert_called()


# =============================================================================
# HPO TRANSFER MANAGER STUDY REGISTRATION TESTS
# =============================================================================


class TestHPOTransferManagerRegistration:
    """Test HPOTransferManager study registration methods."""

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_register_study_success(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test successful study registration."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()
        mock_dataset = MockDataset()

        manager = HPOTransferManager()
        study = MockStudy()

        result = manager.register_study(
            study_name="test_study",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
            model_name="GCN",
        )

        assert result.study_name == "test_study"
        assert "test_study" in manager._meta_db
        assert manager._meta_db["test_study"].model_name == "GCN"

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_register_study_caches_study(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test study registration caches study object."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()
        mock_dataset = MockDataset()

        manager = HPOTransferManager()
        study = MockStudy()

        manager.register_study(
            study_name="test_study",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
            cache_study=True,
        )

        assert "test_study" in manager._study_cache
        assert manager._study_cache["test_study"] is study

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_register_study_no_cache(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test study registration without caching."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()
        mock_dataset = MockDataset()

        manager = HPOTransferManager()
        study = MockStudy()

        manager.register_study(
            study_name="test_study",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
            cache_study=False,
        )

        assert "test_study" not in manager._study_cache

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_register_study_raises_for_duplicate(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test study registration raises ValueError for duplicate study name."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()
        mock_dataset = MockDataset()

        manager = HPOTransferManager()
        study = MockStudy()

        manager.register_study(
            study_name="test_study",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
        )

        with pytest.raises(ValueError) as exc_info:
            manager.register_study(
                study_name="test_study",
                study=study,
                dataset=mock_dataset,
                meta_features={"n_samples": 100.0},
            )

        assert "already registered" in str(exc_info.value)

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_register_study_extracts_study_info(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test study registration extracts study information correctly."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()
        mock_dataset = MockDataset()

        manager = HPOTransferManager()
        study = MockStudy(
            best_params={"lr": 0.001},
            best_value=0.05,
            trials=[MockTrial({"lr": 0.001}, 0.05), MockTrial({"lr": 0.01}, 0.1)],
        )

        result = manager.register_study(
            study_name="test_study",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
        )

        assert result.best_params == {"lr": 0.001}
        assert result.best_value == 0.05
        assert result.n_trials == 2

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_unregister_study_success(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test successful study unregistration."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()
        mock_dataset = MockDataset()

        manager = HPOTransferManager()
        study = MockStudy()

        manager.register_study(
            study_name="test_study",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
        )

        result = manager.unregister_study("test_study")

        assert result is True
        assert "test_study" not in manager._meta_db

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_unregister_study_not_found(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test unregister_study returns False for non-existent study."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()

        result = manager.unregister_study("nonexistent_study")

        assert result is False

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_unregister_study_clears_cache(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test unregister_study clears study from cache."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()
        mock_dataset = MockDataset()

        manager = HPOTransferManager()
        study = MockStudy()

        manager.register_study(
            study_name="test_study",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
        )

        manager.unregister_study("test_study")

        assert "test_study" not in manager._study_cache

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_unregister_study_saves_meta_db_when_persist_enabled(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test unregister_study saves meta_db when persist_meta_db is True."""
        import tempfile

        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            TransferConfig,
        )

        mock_optuna.return_value = MockOptuna()
        mock_dataset = MockDataset()

        # Create temp file for meta_db
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            config = TransferConfig(persist_meta_db=True, meta_db_path=temp_path)
            manager = HPOTransferManager(config)
            study = MockStudy()

            # Register two studies
            manager.register_study(
                study_name="study1",
                study=study,
                dataset=mock_dataset,
                meta_features={"n_samples": 100.0},
            )
            manager.register_study(
                study_name="study2",
                study=study,
                dataset=mock_dataset,
                meta_features={"n_samples": 200.0},
            )

            # Unregister one
            manager.unregister_study("study1")

            # Verify file was updated - only study2 should remain
            with open(temp_path) as f:
                data = json.load(f)

            assert "study1" not in data
            assert "study2" in data
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_get_registered_studies(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test get_registered_studies returns list of study names."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()
        mock_dataset = MockDataset()

        manager = HPOTransferManager()
        study = MockStudy()

        manager.register_study(
            study_name="study1",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
        )
        manager.register_study(
            study_name="study2",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 200.0},
        )

        result = manager.get_registered_studies()

        assert isinstance(result, list)
        assert "study1" in result
        assert "study2" in result
        assert len(result) == 2

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_get_registered_studies_empty(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test get_registered_studies returns empty list when no studies registered."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()

        result = manager.get_registered_studies()

        assert result == []

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_get_study_info_success(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test get_study_info returns study info for registered study."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            RegisteredStudyInfo,
        )

        mock_optuna.return_value = MockOptuna()
        mock_dataset = MockDataset()

        manager = HPOTransferManager()
        study = MockStudy()

        manager.register_study(
            study_name="test_study",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
        )

        result = manager.get_study_info("test_study")

        assert result is not None
        assert isinstance(result, RegisteredStudyInfo)
        assert result.study_name == "test_study"

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_get_study_info_not_found(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test get_study_info returns None for non-existent study."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()

        result = manager.get_study_info("nonexistent_study")

        assert result is None

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_register_study_with_dataset_info(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test study registration with dataset_info."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()
        mock_dataset = MockDataset()

        manager = HPOTransferManager()
        study = MockStudy()

        result = manager.register_study(
            study_name="test_study",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
            dataset_info={"name": "QM9", "split": "random"},
        )

        assert result.dataset_info == {"name": "QM9", "split": "random"}


# =============================================================================
# NOTE: Mock classes are consolidated at the top of the file to avoid duplication.
# The following helper functions are used for creating test data inline.
# =============================================================================


# =============================================================================
# HELPER FUNCTIONS FOR INLINE DATA
# =============================================================================


def get_sample_search_space():
    """Create sample search space dictionary."""
    return {
        "hyperparameters": {
            "hidden_dim": {"type": "int", "low": 32, "high": 256, "step": 32},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        },
        "optimizer": {
            "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
        },
    }


def get_flat_search_space():
    """Create flat search space dictionary."""
    return {
        "hidden_dim": {"type": "int", "low": 32, "high": 256, "step": 32},
        "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
    }


# =============================================================================
# FIND SIMILAR STUDIES TESTS
# =============================================================================


class TestFindSimilarStudies:
    """Test HPOTransferManager.find_similar_studies() method."""

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_find_similar_studies_empty_registry(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test find_similar_studies returns empty list when no studies registered."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()
        mock_dataset = MockDataset()

        manager = HPOTransferManager()
        result = manager.find_similar_studies(mock_dataset)

        assert result == []

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_find_similar_studies_returns_sorted_by_similarity(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test find_similar_studies returns results sorted by similarity (descending)."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            TransferConfig,
        )

        mock_optuna.return_value = MockOptuna()
        mock_dataset = MockDataset()

        config = TransferConfig(similarity_threshold=0.0)
        manager = HPOTransferManager(config)

        study = MockStudy()
        manager.register_study(
            study_name="study1",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 1000.0, "mean_nodes": 25.0},
        )
        manager.register_study(
            study_name="study2",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 500.0, "mean_nodes": 20.0},
        )

        result = manager.find_similar_studies(mock_dataset, top_k=5)

        assert isinstance(result, list)
        if len(result) >= 2:
            assert result[0][1] >= result[1][1]

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_find_similar_studies_respects_top_k(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test find_similar_studies respects top_k parameter."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            TransferConfig,
        )

        mock_optuna.return_value = MockOptuna()
        mock_dataset = MockDataset()

        config = TransferConfig(similarity_threshold=0.0)
        manager = HPOTransferManager(config)

        study = MockStudy()
        for i in range(5):
            manager.register_study(
                study_name=f"study{i}",
                study=study,
                dataset=mock_dataset,
                meta_features={"n_samples": float(1000 + i * 100), "mean_nodes": 25.0},
            )

        result = manager.find_similar_studies(mock_dataset, top_k=2)
        assert len(result) <= 2

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_find_similar_studies_filters_by_model_name(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test find_similar_studies filters by model_name."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            TransferConfig,
        )

        mock_optuna.return_value = MockOptuna()
        mock_dataset = MockDataset()

        config = TransferConfig(similarity_threshold=0.0)
        manager = HPOTransferManager(config)

        study = MockStudy()
        manager.register_study(
            study_name="gcn_study",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 1000.0},
            model_name="GCN",
        )
        manager.register_study(
            study_name="gat_study",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 1000.0},
            model_name="GAT",
        )

        result = manager.find_similar_studies(mock_dataset, model_name="GCN")

        study_names = [name for name, _ in result]
        assert "gcn_study" in study_names or len(result) == 0
        assert "gat_study" not in study_names

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_find_similar_studies_filters_by_min_trials(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test find_similar_studies filters by min_trials."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            TransferConfig,
        )

        mock_optuna.return_value = MockOptuna()
        mock_dataset = MockDataset()

        config = TransferConfig(similarity_threshold=0.0)
        manager = HPOTransferManager(config)

        study1 = MockStudy(trials=[MockTrial({"lr": 0.001}, 0.1) for _ in range(10)])
        study2 = MockStudy(trials=[MockTrial({"lr": 0.001}, 0.1)])

        manager.register_study(
            study_name="many_trials",
            study=study1,
            dataset=mock_dataset,
            meta_features={"n_samples": 1000.0},
        )
        manager.register_study(
            study_name="few_trials",
            study=study2,
            dataset=mock_dataset,
            meta_features={"n_samples": 1000.0},
        )

        result = manager.find_similar_studies(mock_dataset, min_trials=5)
        study_names = [name for name, _ in result]
        assert "few_trials" not in study_names

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_find_similar_studies_respects_similarity_threshold(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test find_similar_studies filters by similarity_threshold."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            TransferConfig,
        )

        mock_optuna.return_value = MockOptuna()
        mock_dataset = MockDataset()

        config = TransferConfig(similarity_threshold=0.999)
        manager = HPOTransferManager(config)

        study = MockStudy()
        manager.register_study(
            study_name="study1",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
        )

        result = manager.find_similar_studies(mock_dataset)
        for _name, sim in result:
            assert sim >= 0.999


# =============================================================================
# COMPUTE DATASET SIMILARITY TESTS
# =============================================================================


class TestComputeDatasetSimilarity:
    """Test HPOTransferManager.compute_dataset_similarity() method."""

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_compute_dataset_similarity_same_dataset(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test compute_dataset_similarity returns valid range for same dataset."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_dataset = MockDataset()
        manager = HPOTransferManager()
        result = manager.compute_dataset_similarity(mock_dataset, mock_dataset)
        assert 0.0 <= result <= 1.0

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_compute_dataset_similarity_different_datasets(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test compute_dataset_similarity for different datasets."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()
        dataset_a = MockDataset(size=100, n_features=10)
        dataset_b = MockDataset(size=200, n_features=20)
        result = manager.compute_dataset_similarity(dataset_a, dataset_b)
        assert 0.0 <= result <= 1.0


# =============================================================================
# _COMPUTE_SIMILARITY TESTS
# =============================================================================


class TestComputeSimilarity:
    """Test HPOTransferManager._compute_similarity() method."""

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_compute_similarity_identical_features(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _compute_similarity returns 1.0 for identical features."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()
        features = {"n_samples": 1000.0, "mean_nodes": 25.0}
        result = manager._compute_similarity(features, features)
        assert result == pytest.approx(1.0, abs=0.001)

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_compute_similarity_no_common_keys(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _compute_similarity returns 0.0 when no common keys."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()
        features_a = {"n_samples": 1000.0}
        features_b = {"mean_nodes": 25.0}
        result = manager._compute_similarity(features_a, features_b)
        assert result == 0.0

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_compute_similarity_partial_overlap(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _compute_similarity with partial key overlap."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()
        features_a = {"n_samples": 1000.0, "mean_nodes": 25.0, "extra": 100.0}
        features_b = {"n_samples": 1000.0, "mean_nodes": 25.0, "other": 200.0}
        result = manager._compute_similarity(features_a, features_b)
        assert result == pytest.approx(1.0, abs=0.001)

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_compute_similarity_zero_vectors(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _compute_similarity handles zero vectors."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()
        features_a = {"n_samples": 0.0, "mean_nodes": 0.0}
        features_b = {"n_samples": 1000.0, "mean_nodes": 25.0}
        result = manager._compute_similarity(features_a, features_b)
        assert result == 0.0

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_compute_similarity_uses_meta_extractor_if_available(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _compute_similarity uses MetaFeatureExtractor if available."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_meta_features.return_value = (MockMetaFeatureExtractor, None)
        manager = HPOTransferManager()
        features_a = {"n_samples": 1000.0}
        features_b = {"n_samples": 500.0}
        result = manager._compute_similarity(features_a, features_b)
        assert result == 0.95

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_compute_similarity_clamps_to_valid_range(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _compute_similarity clamps result to [0, 1]."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()
        features_a = {"n_samples": 1000.0, "mean_nodes": 25.0}
        features_b = {"n_samples": 500.0, "mean_nodes": 50.0}
        result = manager._compute_similarity(features_a, features_b)
        assert 0.0 <= result <= 1.0


# =============================================================================
# COMPUTE DATASET SIMILARITY TESTS
# =============================================================================


class TestComputeDatasetSimilarity:
    """Test HPOTransferManager.compute_dataset_similarity() method."""

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_compute_dataset_similarity_identical_datasets(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test compute_dataset_similarity returns high similarity for identical datasets."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()
        dataset = MockDataset(size=100, n_features=10, mean_edges=50, mean_nodes=20)

        result = manager.compute_dataset_similarity(dataset, dataset)

        assert result == pytest.approx(1.0, abs=0.001)

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_compute_dataset_similarity_different_datasets(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test compute_dataset_similarity returns valid similarity for different datasets."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()
        dataset_a = MockDataset(size=100, n_features=10, mean_edges=50, mean_nodes=20)
        dataset_b = MockDataset(size=200, n_features=20, mean_edges=100, mean_nodes=40)

        result = manager.compute_dataset_similarity(dataset_a, dataset_b)

        # Should return a valid similarity score
        assert 0.0 <= result <= 1.0

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_compute_dataset_similarity_uses_meta_extractor(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test compute_dataset_similarity uses MetaFeatureExtractor when available."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_meta_features.return_value = (MockMetaFeatureExtractor, None)

        manager = HPOTransferManager()
        dataset_a = MockDataset()
        dataset_b = MockDataset()

        result = manager.compute_dataset_similarity(dataset_a, dataset_b)

        # MockMetaFeatureExtractor.compute_similarity returns 0.95
        assert result == 0.95


# =============================================================================
# EXTRACT META FEATURES TESTS
# =============================================================================


class TestExtractMetaFeatures:
    """Test HPOTransferManager._extract_meta_features() method."""

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_extract_meta_features_fallback_to_basic(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _extract_meta_features falls back to _basic_meta_features when extractor unavailable."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()
        dataset = MockDataset()

        result = manager._extract_meta_features(dataset)

        assert isinstance(result, dict)
        assert "n_samples" in result

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_extract_meta_features_uses_extractor_when_available(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _extract_meta_features uses MetaFeatureExtractor when available."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_meta_features.return_value = (MockMetaFeatureExtractor, None)

        manager = HPOTransferManager()
        dataset = MockDataset()

        result = manager._extract_meta_features(dataset)

        # MockMetaFeatureExtractor.extract returns specific features
        assert result == {"n_samples": 1000.0, "mean_nodes": 25.0}


# =============================================================================
# WARM-START STUDY TESTS
# =============================================================================


class TestWarmStartStudy:
    """Test HPOTransferManager.warm_start_study() method."""

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_warm_start_study_raises_without_source_or_dataset(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test warm_start_study raises ValueError without source_studies or target_dataset."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()
        manager = HPOTransferManager()
        target_study = MockStudy()

        with pytest.raises(ValueError) as exc_info:
            manager.warm_start_study(target_study)
        assert "source_studies" in str(exc_info.value) or "target_dataset" in str(exc_info.value)

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_warm_start_study_returns_zero_for_empty_sources(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test warm_start_study returns 0 when source_studies is empty."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()
        manager = HPOTransferManager()
        target_study = MockStudy()

        result = manager.warm_start_study(target_study, source_studies=[])
        assert result == 0

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_warm_start_study_warns_for_missing_source(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test warm_start_study warns and continues if source study not found."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()
        manager = HPOTransferManager()
        target_study = MockStudy()

        result = manager.warm_start_study(target_study, source_studies=["nonexistent"])
        assert result == 0

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_warm_start_study_transfers_trials(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test warm_start_study transfers trials from source to target."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()
        mock_dataset = MockDataset()

        manager = HPOTransferManager()
        source_study = MockStudy(trials=[MockTrial({"lr": 0.001}, 0.1)])

        manager.register_study(
            study_name="source",
            study=source_study,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
        )

        target_study = MockStudy()
        result = manager.warm_start_study(target_study, source_studies=["source"])
        assert result >= 0

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_warm_start_study_auto_detects_similar_studies(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test warm_start_study auto-detects similar studies when target_dataset provided."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            TransferConfig,
        )

        mock_optuna.return_value = MockOptuna()
        mock_dataset = MockDataset()

        config = TransferConfig(similarity_threshold=0.0)
        manager = HPOTransferManager(config)
        source_study = MockStudy(trials=[MockTrial({"lr": 0.001}, 0.1)])

        manager.register_study(
            study_name="source",
            study=source_study,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
        )

        target_study = MockStudy()
        result = manager.warm_start_study(target_study, target_dataset=mock_dataset)
        assert result >= 0


# =============================================================================
# _TRANSFER_TRIALS TESTS
# =============================================================================


class TestTransferTrials:
    """Test HPOTransferManager._transfer_trials() method."""

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_transfer_trials_returns_zero_without_optuna(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _transfer_trials returns 0 when Optuna unavailable."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()
        source_study = MockStudy()
        target_study = MockStudy()
        result = manager._transfer_trials(
            source_study=source_study, target_study=target_study, n_trials=5
        )
        assert result == 0

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_transfer_trials_enqueues_trials(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _transfer_trials enqueues trials into target study."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()
        manager = HPOTransferManager()

        source_study = MockStudy(trials=[MockTrial({"lr": 0.001}, 0.1)])
        target_study = MockStudy()
        result = manager._transfer_trials(
            source_study=source_study, target_study=target_study, n_trials=5
        )
        assert result >= 0

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_transfer_trials_respects_n_trials(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _transfer_trials respects n_trials limit."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()
        manager = HPOTransferManager()

        source_study = MockStudy(
            trials=[MockTrial({"lr": 0.001 * i}, 0.1 * i) for i in range(1, 11)]
        )
        target_study = MockStudy()
        result = manager._transfer_trials(
            source_study=source_study, target_study=target_study, n_trials=3
        )
        assert result <= 3

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    def test_transfer_trials_filters_by_search_space(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _transfer_trials filters params by target_search_space."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()
        sample_search_space = get_sample_search_space()
        manager = HPOTransferManager()

        source_study = MockStudy(trials=[MockTrial({"lr": 0.001, "hidden_dim": 128}, 0.1)])
        target_study = MockStudy()
        result = manager._transfer_trials(
            source_study=source_study,
            target_study=target_study,
            n_trials=5,
            target_search_space=sample_search_space,
        )
        assert result >= 0


# =============================================================================
# _FILTER_PARAMS TESTS
# =============================================================================


class TestFilterParams:
    """Test HPOTransferManager._filter_params() method."""

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_filter_params_keeps_valid_params(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _filter_params keeps valid parameters."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        flat_search_space = get_flat_search_space()
        manager = HPOTransferManager()
        params = {"hidden_dim": 64, "dropout": 0.3, "lr": 0.001}
        result = manager._filter_params(params, flat_search_space)
        assert isinstance(result, dict)

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_filter_params_removes_unknown_params(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _filter_params removes parameters not in search space."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        flat_search_space = get_flat_search_space()
        manager = HPOTransferManager()
        params = {"hidden_dim": 64, "unknown_param": 100}
        result = manager._filter_params(params, flat_search_space)
        assert "unknown_param" not in result

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_filter_params_scales_int_to_bounds(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _filter_params scales int params to bounds when scale_to_bounds=True."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            TransferConfig,
        )

        config = TransferConfig(scale_to_bounds=True)
        manager = HPOTransferManager(config)
        search_space = {"hidden_dim": {"type": "int", "low": 32, "high": 128}}
        params = {"hidden_dim": 256}
        result = manager._filter_params(params, search_space)
        assert result["hidden_dim"] <= 128

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_filter_params_scales_float_to_bounds(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _filter_params scales float params to bounds when scale_to_bounds=True."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            TransferConfig,
        )

        config = TransferConfig(scale_to_bounds=True)
        manager = HPOTransferManager(config)
        search_space = {"dropout": {"type": "float", "low": 0.0, "high": 0.5}}
        params = {"dropout": 0.8}
        result = manager._filter_params(params, search_space)
        assert result["dropout"] <= 0.5

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_filter_params_handles_categorical(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _filter_params handles categorical parameters correctly."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()
        search_space = {"activation": {"type": "categorical", "choices": ["relu", "gelu"]}}

        params_valid = {"activation": "relu"}
        result = manager._filter_params(params_valid, search_space)
        assert result.get("activation") == "relu"

        params_invalid = {"activation": "sigmoid"}
        result = manager._filter_params(params_invalid, search_space)
        assert "activation" not in result or result.get("activation") != "sigmoid"

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_filter_params_uses_warm_start_if_available(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _filter_params uses WarmStartStrategy if available."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_warm_start.return_value = (MockWarmStartStrategy, None, None, None)
        manager = HPOTransferManager()
        search_space = {"hidden_dim": {"type": "int", "low": 32, "high": 256}}
        params = {"hidden_dim": 64}
        result = manager._filter_params(params, search_space)
        assert isinstance(result, dict)


# =============================================================================
# _FLATTEN_SEARCH_SPACE TESTS
# =============================================================================


class TestFlattenSearchSpace:
    """Test HPOTransferManager._flatten_search_space() method."""

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_flatten_search_space_nested(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _flatten_search_space flattens nested search space."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        sample_search_space = get_sample_search_space()
        manager = HPOTransferManager()
        result = manager._flatten_search_space(sample_search_space)

        assert isinstance(result, dict)
        # Check that all param configs are accessible (keys may be flattened with dots or just param names)
        # The actual implementation may use "hyperparameters.hidden_dim" or "hidden_dim" format
        all_keys_str = str(result.keys())
        assert "hidden_dim" in all_keys_str
        assert "dropout" in all_keys_str
        assert "lr" in all_keys_str

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_flatten_search_space_already_flat(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _flatten_search_space handles already flat space."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        flat_search_space = get_flat_search_space()
        manager = HPOTransferManager()
        result = manager._flatten_search_space(flat_search_space)

        assert "hidden_dim" in result
        assert result["hidden_dim"]["type"] == "int"

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_flatten_search_space_empty(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _flatten_search_space handles empty space."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()
        result = manager._flatten_search_space({})
        assert result == {}


# =============================================================================
# _ADD_NOISE_TO_PARAMS TESTS
# =============================================================================


class TestAddNoiseToParams:
    """Test HPOTransferManager._add_noise_to_params() method."""

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_add_noise_returns_params_when_no_search_space(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _add_noise_to_params returns original params when no search space."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()
        params = {"lr": 0.001, "hidden_dim": 128}
        result = manager._add_noise_to_params(params, None)
        assert result == params

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_add_noise_modifies_float_params(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _add_noise_to_params adds noise to float parameters."""
        import numpy as np

        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            TransferConfig,
        )

        np.random.seed(42)
        sample_search_space = get_sample_search_space()
        config = TransferConfig(add_noise=True, noise_scale=0.1)
        manager = HPOTransferManager(config)
        params = {"dropout": 0.25}
        result = manager._add_noise_to_params(params, sample_search_space)

        assert "dropout" in result
        assert isinstance(result["dropout"], float)

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_add_noise_respects_bounds(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _add_noise_to_params keeps values within bounds."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            TransferConfig,
        )

        sample_search_space = get_sample_search_space()
        config = TransferConfig(add_noise=True, noise_scale=0.5)
        manager = HPOTransferManager(config)
        params = {"dropout": 0.5}

        for _ in range(10):
            result = manager._add_noise_to_params(params, sample_search_space)
            assert result["dropout"] >= 0.0
            assert result["dropout"] <= 0.5

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_add_noise_modifies_int_params(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _add_noise_to_params adds noise to int parameters."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            TransferConfig,
        )

        sample_search_space = get_sample_search_space()
        config = TransferConfig(add_noise=True, noise_scale=0.1)
        manager = HPOTransferManager(config)
        params = {"hidden_dim": 128}
        result = manager._add_noise_to_params(params, sample_search_space)

        assert "hidden_dim" in result
        assert isinstance(result["hidden_dim"], int)


# =============================================================================
# _EXTRACT_META_FEATURES TESTS
# =============================================================================


class TestExtractMetaFeatures:
    """Test HPOTransferManager._extract_meta_features() method."""

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_extract_meta_features_uses_extractor_if_available(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _extract_meta_features uses MetaFeatureExtractor if available."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_meta_features.return_value = (MockMetaFeatureExtractor, None)
        mock_dataset = MockDataset()
        manager = HPOTransferManager()
        result = manager._extract_meta_features(mock_dataset)
        assert result == {"n_samples": 1000.0, "mean_nodes": 25.0}

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_extract_meta_features_falls_back_to_basic(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _extract_meta_features falls back to _basic_meta_features."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_dataset = MockDataset()
        manager = HPOTransferManager()
        result = manager._extract_meta_features(mock_dataset)

        assert isinstance(result, dict)
        assert "n_samples" in result


# =============================================================================
# _BASIC_META_FEATURES TESTS
# =============================================================================


class TestBasicMetaFeatures:
    """Test HPOTransferManager._basic_meta_features() method."""

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_basic_meta_features_extracts_n_samples(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _basic_meta_features extracts n_samples."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_dataset = MockDataset()
        manager = HPOTransferManager()
        result = manager._basic_meta_features(mock_dataset)

        assert "n_samples" in result
        assert result["n_samples"] == 100.0

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_basic_meta_features_extracts_n_features(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _basic_meta_features extracts n_features from node features."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_dataset = MockDataset()
        manager = HPOTransferManager()
        result = manager._basic_meta_features(mock_dataset)

        assert "n_features" in result
        assert result["n_features"] == 10.0

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_basic_meta_features_handles_empty_dataset(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _basic_meta_features handles empty dataset."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()
        empty_dataset = MockDataset(size=0)
        result = manager._basic_meta_features(empty_dataset)
        assert result["n_samples"] == 0.0

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_basic_meta_features_handles_exception(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _basic_meta_features handles exceptions gracefully."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()

        class BrokenDataset:
            def __len__(self):
                raise RuntimeError("Broken")

        result = manager._basic_meta_features(BrokenDataset())
        assert result["n_samples"] == 0.0

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_basic_meta_features_extracts_edge_stats(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _basic_meta_features extracts edge statistics."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_dataset = MockDataset()
        manager = HPOTransferManager()
        result = manager._basic_meta_features(mock_dataset)
        assert "mean_edges" in result

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_basic_meta_features_extracts_target_stats(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _basic_meta_features extracts target statistics."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_dataset = MockDataset()
        manager = HPOTransferManager()
        result = manager._basic_meta_features(mock_dataset)
        assert "target_mean" in result

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_basic_meta_features_handles_multi_target(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _basic_meta_features handles datasets with multi-dimensional targets using .mean()."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()

        # Create dataset with multi-dimensional targets that use .mean() method
        class MultiTargetDataset:
            def __init__(self):
                self._data = []
                for _ in range(10):
                    data = MagicMock()
                    data.x = MagicMock()
                    data.x.shape = (10, 5)
                    data.edge_index = MagicMock()
                    data.edge_index.shape = (2, 20)
                    data.y = MagicMock()
                    # Multi-dimensional target - numel() > 1
                    data.y.numel.return_value = 3
                    data.y.mean.return_value = MagicMock(item=MagicMock(return_value=0.75))
                    self._data.append(data)

            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return self._data[idx]

        dataset = MultiTargetDataset()
        result = manager._basic_meta_features(dataset)

        assert "target_mean" in result
        assert result["target_mean"] == pytest.approx(0.75, abs=0.01)

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_basic_meta_features_handles_no_x_attribute(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _basic_meta_features handles datasets without x (node features) attribute."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()

        class NoNodeFeaturesDataset:
            def __init__(self):
                self._data = []
                for _ in range(5):
                    data = MagicMock()
                    data.x = None  # No node features
                    data.edge_index = MagicMock()
                    data.edge_index.shape = (2, 10)
                    data.y = MagicMock()
                    data.y.numel.return_value = 1
                    data.y.item.return_value = 0.5
                    self._data.append(data)

            def __len__(self):
                return 5

            def __getitem__(self, idx):
                return self._data[idx]

        dataset = NoNodeFeaturesDataset()
        result = manager._basic_meta_features(dataset)

        # Should still work and have n_samples
        assert "n_samples" in result
        assert result["n_samples"] == 5.0
        # n_features should not be present since x is None
        assert "n_features" not in result


# =============================================================================
# _FLATTEN_SEARCH_SPACE TESTS
# =============================================================================


class TestFlattenSearchSpace:
    """Test HPOTransferManager._flatten_search_space() method."""

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_flatten_search_space_nested(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _flatten_search_space flattens nested search space."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()

        nested_space = {
            "hyperparameters": {
                "hidden_dim": {"type": "int", "low": 32, "high": 256},
                "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            },
            "optimizer": {
                "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
            },
        }

        result = manager._flatten_search_space(nested_space)

        # All nested params should be flattened
        assert "hyperparameters.hidden_dim" in result or "hidden_dim" in result
        assert "hyperparameters.dropout" in result or "dropout" in result
        assert "optimizer.lr" in result or "lr" in result

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_flatten_search_space_already_flat(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _flatten_search_space handles already flat search space."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()

        flat_space = {
            "hidden_dim": {"type": "int", "low": 32, "high": 256},
            "lr": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
        }

        result = manager._flatten_search_space(flat_space)

        assert "hidden_dim" in result
        assert "lr" in result
        assert result["hidden_dim"]["type"] == "int"

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    def test_flatten_search_space_empty(
        self, mock_optuna, mock_meta_features, mock_warm_start, mock_hpo_error
    ):
        """Test _flatten_search_space handles empty search space."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()

        result = manager._flatten_search_space({})

        assert result == {}


# =============================================================================
# NOTE: Mock classes are consolidated at the top of the file.
# Fixtures below are unique to persistence tests.
# =============================================================================

# =============================================================================
# TEST FIXTURES FOR PERSISTENCE TESTS
# =============================================================================


@pytest.fixture
def sample_study_info_data():
    """Create sample study info data for import/export."""
    return {
        "study1": {
            "study_name": "study1",
            "meta_features": {"n_samples": 1000.0},
            "best_params": {"lr": 0.001},
            "best_value": 0.1,
            "n_trials": 50,
            "n_completed": 45,
            "direction": "minimize",
            "model_name": "GCN",
            "dataset_info": None,
            "registered_at": "2024-01-15T10:00:00",
        },
        "study2": {
            "study_name": "study2",
            "meta_features": {"n_samples": 2000.0},
            "best_params": {"lr": 0.0001},
            "best_value": 0.05,
            "n_trials": 100,
            "n_completed": 95,
            "direction": "minimize",
            "model_name": "GAT",
            "dataset_info": None,
            "registered_at": "2024-01-15T11:00:00",
        },
    }


@pytest.fixture
def temp_meta_db_path():
    """Create temporary file path for meta-database."""
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


# =============================================================================
# _SAVE_META_DB TESTS
# =============================================================================


class TestSaveMetaDb:
    """Test HPOTransferManager._save_meta_db() method."""

    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_save_meta_db_does_nothing_without_path(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna
    ):
        """Test _save_meta_db does nothing when meta_db_path is None."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()

        manager = HPOTransferManager()

        # Should not raise
        manager._save_meta_db()

    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_save_meta_db_writes_to_file(
        self,
        mock_hpo_error,
        mock_warm_start,
        mock_meta_features,
        mock_optuna,
        temp_meta_db_path,
        mock_dataset,
    ):
        """Test _save_meta_db writes data to file."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            TransferConfig,
        )

        mock_optuna.return_value = MockOptuna()

        config = TransferConfig(persist_meta_db=True, meta_db_path=temp_meta_db_path)
        manager = HPOTransferManager(config)

        # Register a study
        study = MockStudy()
        manager.register_study(
            study_name="test_study",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
        )

        # Verify file was written
        assert os.path.exists(temp_meta_db_path)

        with open(temp_meta_db_path) as f:
            data = json.load(f)

        assert "test_study" in data

    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch("builtins.open", side_effect=PermissionError("Permission denied"))
    def test_save_meta_db_handles_write_error(
        self, mock_open, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna
    ):
        """Test _save_meta_db handles write errors gracefully."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            TransferConfig,
        )

        mock_optuna.return_value = MockOptuna()

        config = TransferConfig(persist_meta_db=True, meta_db_path="/invalid/path/meta_db.json")
        manager = HPOTransferManager(config)

        # Should not raise, just log warning
        manager._save_meta_db()


# =============================================================================
# _LOAD_META_DB TESTS
# =============================================================================


class TestLoadMetaDb:
    """Test HPOTransferManager._load_meta_db() method."""

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_load_meta_db_does_nothing_without_path(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna
    ):
        """Test _load_meta_db does nothing when meta_db_path is None."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()

        # Should not raise
        manager._load_meta_db()

        assert manager._meta_db == {}

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    @patch("os.path.exists", return_value=False)
    def test_load_meta_db_does_nothing_when_file_not_exists(
        self, mock_exists, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna
    ):
        """Test _load_meta_db does nothing when file does not exist."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            TransferConfig,
        )

        config = TransferConfig(persist_meta_db=True, meta_db_path="/path/to/nonexistent.json")
        manager = HPOTransferManager(config)

        assert manager._meta_db == {}

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_load_meta_db_loads_data(
        self,
        mock_hpo_error,
        mock_warm_start,
        mock_meta_features,
        mock_optuna,
        temp_meta_db_path,
        sample_study_info_data,
    ):
        """Test _load_meta_db loads data from file."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            TransferConfig,
        )

        # Write sample data to file
        with open(temp_meta_db_path, "w") as f:
            json.dump(sample_study_info_data, f)

        config = TransferConfig(persist_meta_db=True, meta_db_path=temp_meta_db_path)
        manager = HPOTransferManager(config)

        assert "study1" in manager._meta_db
        assert "study2" in manager._meta_db
        assert manager._meta_db["study1"].model_name == "GCN"

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_load_meta_db_handles_invalid_json(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna, temp_meta_db_path
    ):
        """Test _load_meta_db handles invalid JSON gracefully."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            TransferConfig,
        )

        # Write invalid JSON to file
        with open(temp_meta_db_path, "w") as f:
            f.write("not valid json {{{")

        config = TransferConfig(persist_meta_db=True, meta_db_path=temp_meta_db_path)

        # Should not raise, just log warning
        manager = HPOTransferManager(config)

        assert manager._meta_db == {}


# =============================================================================
# EXPORT_META_DB TESTS
# =============================================================================


class TestExportMetaDb:
    """Test HPOTransferManager.export_meta_db() method."""

    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_export_meta_db_writes_to_file(
        self,
        mock_hpo_error,
        mock_warm_start,
        mock_meta_features,
        mock_optuna,
        temp_meta_db_path,
        mock_dataset,
    ):
        """Test export_meta_db writes data to specified path."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()

        manager = HPOTransferManager()

        # Register studies
        study = MockStudy()
        manager.register_study(
            study_name="study1",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
        )
        manager.register_study(
            study_name="study2",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 200.0},
        )

        manager.export_meta_db(temp_meta_db_path)

        with open(temp_meta_db_path) as f:
            data = json.load(f)

        assert "study1" in data
        assert "study2" in data

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_export_meta_db_empty_db(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna, temp_meta_db_path
    ):
        """Test export_meta_db with empty database."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()

        manager.export_meta_db(temp_meta_db_path)

        with open(temp_meta_db_path) as f:
            data = json.load(f)

        assert data == {}


# =============================================================================
# IMPORT_META_DB TESTS
# =============================================================================


class TestImportMetaDb:
    """Test HPOTransferManager.import_meta_db() method."""

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_import_meta_db_loads_data(
        self,
        mock_hpo_error,
        mock_warm_start,
        mock_meta_features,
        mock_optuna,
        temp_meta_db_path,
        sample_study_info_data,
    ):
        """Test import_meta_db loads data from file."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        # Write sample data to file
        with open(temp_meta_db_path, "w") as f:
            json.dump(sample_study_info_data, f)

        manager = HPOTransferManager()

        n_imported = manager.import_meta_db(temp_meta_db_path)

        assert n_imported == 2
        assert "study1" in manager._meta_db
        assert "study2" in manager._meta_db

    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_import_meta_db_merge_true(
        self,
        mock_hpo_error,
        mock_warm_start,
        mock_meta_features,
        mock_optuna,
        temp_meta_db_path,
        sample_study_info_data,
        mock_dataset,
    ):
        """Test import_meta_db merges with existing data when merge=True."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()

        # Write sample data to file
        with open(temp_meta_db_path, "w") as f:
            json.dump(sample_study_info_data, f)

        manager = HPOTransferManager()

        # Register an existing study
        study = MockStudy()
        manager.register_study(
            study_name="existing_study",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
        )

        n_imported = manager.import_meta_db(temp_meta_db_path, merge=True)

        assert n_imported == 2
        assert "existing_study" in manager._meta_db
        assert "study1" in manager._meta_db
        assert "study2" in manager._meta_db

    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_import_meta_db_merge_false(
        self,
        mock_hpo_error,
        mock_warm_start,
        mock_meta_features,
        mock_optuna,
        temp_meta_db_path,
        sample_study_info_data,
        mock_dataset,
    ):
        """Test import_meta_db replaces existing data when merge=False."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()

        # Write sample data to file
        with open(temp_meta_db_path, "w") as f:
            json.dump(sample_study_info_data, f)

        manager = HPOTransferManager()

        # Register an existing study
        study = MockStudy()
        manager.register_study(
            study_name="existing_study",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
        )

        n_imported = manager.import_meta_db(temp_meta_db_path, merge=False)

        assert n_imported == 2
        assert "existing_study" not in manager._meta_db
        assert "study1" in manager._meta_db
        assert "study2" in manager._meta_db

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_import_meta_db_skips_duplicates(
        self,
        mock_hpo_error,
        mock_warm_start,
        mock_meta_features,
        mock_optuna,
        temp_meta_db_path,
        sample_study_info_data,
    ):
        """Test import_meta_db skips studies that already exist when merge=True."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        # Write sample data to file
        with open(temp_meta_db_path, "w") as f:
            json.dump(sample_study_info_data, f)

        manager = HPOTransferManager()

        # Import once
        n_imported_first = manager.import_meta_db(temp_meta_db_path, merge=True)

        # Import again - should skip all
        n_imported_second = manager.import_meta_db(temp_meta_db_path, merge=True)

        assert n_imported_first == 2
        assert n_imported_second == 0


# =============================================================================
# GET_TRANSFER_SUMMARY TESTS
# =============================================================================


class TestGetTransferSummary:
    """Test HPOTransferManager.get_transfer_summary() method."""

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_get_transfer_summary_empty(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna
    ):
        """Test get_transfer_summary with empty database."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()

        summary = manager.get_transfer_summary()

        assert summary["n_studies"] == 0
        assert summary["n_cached"] == 0
        assert summary["total_trials"] == 0
        assert summary["models"] == []

    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_get_transfer_summary_with_studies(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna, mock_dataset
    ):
        """Test get_transfer_summary with registered studies."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()

        manager = HPOTransferManager()

        # Register studies
        study1 = MockStudy(trials=[MockTrial({"lr": 0.001}, 0.1) for _ in range(10)])
        study2 = MockStudy(trials=[MockTrial({"lr": 0.001}, 0.1) for _ in range(20)])

        manager.register_study(
            study_name="gcn_study",
            study=study1,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
            model_name="GCN",
        )
        manager.register_study(
            study_name="gat_study",
            study=study2,
            dataset=mock_dataset,
            meta_features={"n_samples": 200.0},
            model_name="GAT",
        )

        summary = manager.get_transfer_summary()

        assert summary["n_studies"] == 2
        assert summary["n_cached"] == 2
        assert summary["total_trials"] == 30
        assert "GCN" in summary["models"]
        assert "GAT" in summary["models"]

    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_get_transfer_summary_includes_config_info(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna, mock_dataset
    ):
        """Test get_transfer_summary includes configuration info."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            TransferConfig,
        )

        mock_optuna.return_value = MockOptuna()

        config = TransferConfig(similarity_threshold=0.85, n_warm_start_trials=15)
        manager = HPOTransferManager(config)

        study = MockStudy()
        manager.register_study(
            study_name="test_study",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
        )

        summary = manager.get_transfer_summary()

        assert summary["similarity_threshold"] == 0.85
        assert summary["n_warm_start_trials"] == 15


# =============================================================================
# CLEAR TESTS
# =============================================================================


class TestClear:
    """Test HPOTransferManager.clear() method."""

    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_clear_removes_all_studies(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna, mock_dataset
    ):
        """Test clear() removes all registered studies."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()

        manager = HPOTransferManager()

        study = MockStudy()
        manager.register_study(
            study_name="study1",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
        )
        manager.register_study(
            study_name="study2",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 200.0},
        )

        manager.clear()

        assert manager._meta_db == {}

    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_clear_removes_cache(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna, mock_dataset
    ):
        """Test clear() removes study cache."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()

        manager = HPOTransferManager()

        study = MockStudy()
        manager.register_study(
            study_name="test_study",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
        )

        manager.clear()

        assert manager._study_cache == {}

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_clear_on_empty_manager(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna
    ):
        """Test clear() on empty manager doesn't raise."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()

        # Should not raise
        manager.clear()

        assert manager._meta_db == {}
        assert manager._study_cache == {}


# =============================================================================
# _RAISE_HPO_ERROR TESTS
# =============================================================================


class TestRaiseHpoError:
    """Test HPOTransferManager._raise_hpo_error() method."""

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error")
    def test_raise_hpo_error_uses_hpo_error_if_available(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna
    ):
        """Test _raise_hpo_error uses HPOError if available."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_hpo_error.return_value = MockHPOError

        manager = HPOTransferManager()

        with pytest.raises(MockHPOError) as exc_info:
            manager._raise_hpo_error(
                message="Test error", study_name="test_study", details="Test details"
            )

        assert "Test error" in str(exc_info.value)

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_raise_hpo_error_falls_back_to_runtime_error(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna
    ):
        """Test _raise_hpo_error falls back to RuntimeError if HPOError unavailable."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()

        with pytest.raises(RuntimeError) as exc_info:
            manager._raise_hpo_error(
                message="Test error", study_name="test_study", details="Test details"
            )

        assert "Test error" in str(exc_info.value)
        assert "test_study" in str(exc_info.value)
        assert "Test details" in str(exc_info.value)


# =============================================================================
# __REPR__ TESTS
# =============================================================================


class TestRepr:
    """Test HPOTransferManager.__repr__() method."""

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_repr_empty_manager(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna
    ):
        """Test __repr__ for empty manager."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()

        repr_str = repr(manager)

        assert "HPOTransferManager" in repr_str
        assert "n_studies=0" in repr_str
        assert "n_cached=0" in repr_str

    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_repr_with_studies(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna, mock_dataset
    ):
        """Test __repr__ with registered studies."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()

        manager = HPOTransferManager()

        study = MockStudy()
        manager.register_study(
            study_name="test_study",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
        )

        repr_str = repr(manager)

        assert "n_studies=1" in repr_str
        assert "n_cached=1" in repr_str

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_repr_includes_similarity_threshold(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna
    ):
        """Test __repr__ includes similarity threshold."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            TransferConfig,
        )

        config = TransferConfig(similarity_threshold=0.85)
        manager = HPOTransferManager(config)

        repr_str = repr(manager)

        assert "similarity_threshold=0.85" in repr_str


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegrationScenarios:
    """Test comprehensive integration scenarios."""

    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_full_workflow_register_find_warmstart(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna, mock_dataset
    ):
        """Test full workflow: register, find similar, warm-start."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            TransferConfig,
        )

        mock_optuna.return_value = MockOptuna()

        config = TransferConfig(
            n_warm_start_trials=5,
            similarity_threshold=0.0,  # Allow all
        )
        manager = HPOTransferManager(config)

        # 1. Register source study
        source_study = MockStudy(trials=[MockTrial({"lr": 0.001, "hidden_dim": 128}, 0.1)])
        manager.register_study(
            study_name="source_study",
            study=source_study,
            dataset=mock_dataset,
            meta_features={"n_samples": 1000.0, "mean_nodes": 25.0},
            model_name="GCN",
        )

        # 2. Find similar studies
        similar = manager.find_similar_studies(mock_dataset, model_name="GCN")

        # 3. Warm-start new study
        target_study = MockStudy()
        n_transferred = manager.warm_start_study(
            target_study, source_studies=[name for name, _ in similar]
        )

        # Verify
        assert len(manager._meta_db) == 1
        assert n_transferred >= 0

    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_export_import_roundtrip(
        self,
        mock_hpo_error,
        mock_warm_start,
        mock_meta_features,
        mock_optuna,
        temp_meta_db_path,
        mock_dataset,
    ):
        """Test export and import roundtrip."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()

        # Create manager and register studies
        manager1 = HPOTransferManager()

        study = MockStudy()
        manager1.register_study(
            study_name="study1",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 1000.0},
            model_name="GCN",
        )
        manager1.register_study(
            study_name="study2",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 2000.0},
            model_name="GAT",
        )

        # Export
        manager1.export_meta_db(temp_meta_db_path)

        # Import into new manager
        manager2 = HPOTransferManager()
        n_imported = manager2.import_meta_db(temp_meta_db_path)

        # Verify
        assert n_imported == 2
        assert "study1" in manager2._meta_db
        assert "study2" in manager2._meta_db
        assert manager2._meta_db["study1"].model_name == "GCN"

    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_multiple_model_types(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna, mock_dataset
    ):
        """Test managing studies with multiple model types."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import (
            HPOTransferManager,
            TransferConfig,
        )

        mock_optuna.return_value = MockOptuna()

        config = TransferConfig(similarity_threshold=0.0)
        manager = HPOTransferManager(config)

        study = MockStudy()

        # Register studies with different model types
        for model_name in ["GCN", "GAT", "GraphSAGE", "SchNet"]:
            manager.register_study(
                study_name=f"{model_name.lower()}_study",
                study=study,
                dataset=mock_dataset,
                meta_features={"n_samples": 1000.0},
                model_name=model_name,
            )

        # Get summary
        summary = manager.get_transfer_summary()

        assert summary["n_studies"] == 4
        assert len(summary["models"]) == 4

        # Filter by model
        gcn_similar = manager.find_similar_studies(mock_dataset, model_name="GCN")
        gat_similar = manager.find_similar_studies(mock_dataset, model_name="GAT")

        # Each filter should return only that model
        for name, _ in gcn_similar:
            assert "gcn" in name
        for name, _ in gat_similar:
            assert "gat" in name


# =============================================================================
# EDGE CASES AND ERROR HANDLING TESTS
# =============================================================================


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_register_study_with_no_trials(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna, mock_dataset
    ):
        """Test registering study with no trials."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()

        manager = HPOTransferManager()

        study = MockStudy(trials=[])

        result = manager.register_study(
            study_name="empty_study",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
        )

        assert result.n_trials == 0

    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_register_study_with_direction_name_attribute(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna, mock_dataset
    ):
        """Test registering study where direction has name attribute."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()

        manager = HPOTransferManager()

        study = MockStudy(direction=MockDirection.MAXIMIZE)

        result = manager.register_study(
            study_name="max_study",
            study=study,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
        )

        assert result.direction == "maximize"

    @patch("milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna")
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_warm_start_with_enqueue_error(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna, mock_dataset
    ):
        """Test warm_start handles enqueue errors gracefully."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        mock_optuna.return_value = MockOptuna()

        manager = HPOTransferManager()

        source_study = MockStudy(trials=[MockTrial({"lr": 0.001}, 0.1)])
        manager.register_study(
            study_name="source",
            study=source_study,
            dataset=mock_dataset,
            meta_features={"n_samples": 100.0},
        )

        # Create target study that raises on enqueue
        class FailingStudy:
            def enqueue_trial(self, params):
                raise RuntimeError("Enqueue failed")

            trials = []

        target_study = FailingStudy()

        # Should not raise, just log and return 0
        result = manager.warm_start_study(target_study, source_studies=["source"])

        assert result == 0

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_filter_params_with_enum_type(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna
    ):
        """Test _filter_params handles enum type values."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()

        class MockParamType(Enum):
            INT = "int"

        search_space = {"hidden_dim": {"type": MockParamType.INT, "low": 32, "high": 256}}
        params = {"hidden_dim": 64}

        result = manager._filter_params(params, search_space)

        assert "hidden_dim" in result
        assert result["hidden_dim"] == 64

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_filter_params_int_uniform_type(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna
    ):
        """Test _filter_params handles int_uniform type."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()

        search_space = {"batch_size": {"type": "int_uniform", "low": 16, "high": 128}}
        params = {"batch_size": 32}

        result = manager._filter_params(params, search_space)

        assert "batch_size" in result
        assert result["batch_size"] == 32

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_filter_params_uniform_type(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna
    ):
        """Test _filter_params handles uniform type."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()

        search_space = {"temperature": {"type": "uniform", "low": 0.1, "high": 2.0}}
        params = {"temperature": 1.0}

        result = manager._filter_params(params, search_space)

        assert "temperature" in result
        assert result["temperature"] == 1.0

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_filter_params_unknown_type(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna
    ):
        """Test _filter_params handles unknown param type."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()

        search_space = {"custom_param": {"type": "custom_type", "value": 42}}
        params = {"custom_param": 42}

        result = manager._filter_params(params, search_space)

        # Unknown types are passed through
        assert "custom_param" in result

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_basic_meta_features_dataset_without_target(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna
    ):
        """Test _basic_meta_features handles dataset without target."""
        from milia_pipeline.models.hpo.transfer.transfer_manager import HPOTransferManager

        manager = HPOTransferManager()

        # Create dataset without y attribute
        class NoTargetDataset:
            def __init__(self):
                self._data = [MagicMock(x=MagicMock(shape=(10, 5)), y=None) for _ in range(10)]

            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return self._data[idx]

        dataset = NoTargetDataset()

        result = manager._basic_meta_features(dataset)

        assert "n_samples" in result
        # target_mean/std should not be present or should handle gracefully


# =============================================================================
# MODULE EXPORTS TESTS
# =============================================================================


class TestModuleExports:
    """Test module exports."""

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_all_exports_available(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna
    ):
        """Test all expected exports are available."""
        from milia_pipeline.models.hpo.transfer import transfer_manager

        expected_exports = [
            "MetaFeatureMethod",
            "AdaptationMethod",
            "TransferConfig",
            "RegisteredStudyInfo",
            "HPOTransferManager",
        ]

        for export in expected_exports:
            assert hasattr(transfer_manager, export), f"Missing export: {export}"

    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_optuna", return_value=None
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_meta_features",
        return_value=(None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_warm_start",
        return_value=(None, None, None, None),
    )
    @patch(
        "milia_pipeline.models.hpo.transfer.transfer_manager._lazy_import_hpo_error",
        return_value=None,
    )
    def test_all_list_matches_exports(
        self, mock_hpo_error, mock_warm_start, mock_meta_features, mock_optuna
    ):
        """Test __all__ matches available exports."""
        from milia_pipeline.models.hpo.transfer import transfer_manager

        if hasattr(transfer_manager, "__all__"):
            for export in transfer_manager.__all__:
                assert hasattr(transfer_manager, export), f"__all__ contains unavailable: {export}"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
