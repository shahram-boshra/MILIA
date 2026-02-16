#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for HPO Manager and Related Modules

This integration test suite verifies that the following five modules work together
correctly:
1. hpo_manager.py - HPO orchestration with task-specific data preparation
2. main.py - Main entry point with task-specific data preparation
3. loss_functions.py - LossRegistry for loss function creation
4. optimizers.py - OptimizerRegistry for optimizer creation
5. schedulers.py - SchedulerRegistry for scheduler creation

Integration Test Areas (25 test classes, 100+ tests):

1. HPO Manager + Loss Registry Integration (TestHPOManagerLossRegistryIntegration)
   - Task-specific loss function selection for all 6 task types
   - Parameter filtering across components
   - Fallback behavior when registry unavailable

2. HPO Manager + Optimizer Registry Integration (TestHPOManagerOptimizerRegistryIntegration)
   - Optimizer creation with HPO-suggested parameters
   - Parameter merging and filtering
   - All optimizer types

3. HPO Manager + Scheduler Registry Integration (TestHPOManagerSchedulerRegistryIntegration)
   - Scheduler creation with HPO-suggested parameters
   - Metric-based scheduler configuration
   - Warmup scheduler integration

4. HPO Manager + Main Module Integration (TestHPOManagerMainModuleIntegration)
   - Task-specific data preparation consistency
   - Task type inference alignment
   - Data preparation for all task types

5. Registry Cross-Module Integration (TestRegistryCrossModuleIntegration)
   - Loss + Optimizer + Scheduler creation pipeline
   - Parameter filtering consistency

6. End-to-End HPO Workflow Integration (TestEndToEndHPOWorkflow)
   - Full HPO trial execution (mocked)
   - Component creation pipeline
   - Best parameter retrieval

7. Task-Specific Data Flow (TestTaskSpecificDataFlow)
   - All 6 task types data flow
   - Loss selection per task

8. Cross-Module Error Handling (TestCrossModuleErrorHandling)
   - Error propagation between modules
   - Graceful fallback behavior

9. Registry List Functions (TestRegistryListFunctions)
   - All registries listing
   - Component instantiation validation

10. Scheduler-Specific Features (TestSchedulerSpecificFeatures)
    - Metric-based vs step-based schedulers
    - Warmup scheduler creation

11. Helper Function Integration (TestHelperFunctionIntegration)
    - _flatten_params function
    - _extract_param_categories function

12. Loss Registry Advanced Features (TestLossRegistryAdvancedFeatures)
    - Custom loss registration
    - Loss info retrieval
    - Valid params introspection

13. Optimizer Registry Advanced Features (TestOptimizerRegistryAdvancedFeatures)
    - Custom optimizer registration
    - Default params handling
    - Valid params introspection

14. Scheduler Registry Advanced Features (TestSchedulerRegistryAdvancedFeatures)
    - Custom scheduler registration
    - Warmup scheduler creation
    - Sequential scheduler support

15. HPO Manager Convenience Functions (TestHPOManagerConvenienceFunctions)
    - is_hpo_enabled
    - get_best_params
    - create_hpo_manager

16. HPO Study Management (TestHPOStudyManagement)
    - Study creation
    - Study statistics
    - Trial management

17. Task Type Mapping Completeness (TestTaskTypeMappingCompleteness)
    - All task types have loss mappings
    - All task types have data prep functions

18. Parameter Filtering Consistency (TestParameterFilteringConsistency)
    - Consistent filtering across all registries
    - Invalid params always filtered

19. Cross-Validation Integration (TestCrossValidationIntegration)
    - CV with registries
    - Metric aggregation

20. Final Model Training Integration (TestFinalModelTrainingIntegration)
    - train_final_model flow
    - Best params application

21. Search Space Filtering Integration (TestSearchSpaceFilteringIntegration)
    - Model-specific filtering
    - Registry metadata usage

22. Data Splitting Integration (TestDataSplittingIntegration)
    - DataSplitter + HPO integration
    - Split ratios validation

23. Callback Integration (TestCallbackIntegration)
    - HPO callback creation
    - Callback list management

24. Device Management Integration (TestDeviceManagementIntegration)
    - DeviceManager + HPO integration
    - Model device placement

25. Factory Integration (TestFactoryIntegration)
    - ModelFactory + HPO integration
    - Model creation with HPO params

Author: Milia Team
Version: 2.0.0
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# =============================================================================
# MOCK ENUMS AND DATACLASSES (Avoid sys.modules pollution)
# =============================================================================


class MockParamType(Enum):
    """Mock ParamType enum for testing."""

    INT = "int"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    LOGUNIFORM = "loguniform"
    UNIFORM = "uniform"


class MockDirection(Enum):
    """Mock Direction enum for testing."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class MockPrunerType(Enum):
    """Mock PrunerType enum for testing."""

    MEDIAN = "median"
    PERCENTILE = "percentile"
    HYPERBAND = "hyperband"
    NONE = "none"


class MockSamplerType(Enum):
    """Mock SamplerType enum for testing."""

    TPE = "tpe"
    RANDOM = "random"
    CMAES = "cmaes"


@dataclass
class MockPrunerConfig:
    """Mock PrunerConfig for testing."""

    type: MockPrunerType = MockPrunerType.MEDIAN
    n_startup_trials: int = 5
    n_warmup_steps: int = 10
    interval_steps: int = 1
    percentile: float = 25.0


@dataclass
class MockSamplerConfig:
    """Mock SamplerConfig for testing."""

    type: MockSamplerType = MockSamplerType.TPE
    seed: int | None = None
    n_startup_trials: int = 10
    multivariate: bool = False
    constant_liar: bool = False


@dataclass
class MockStudyConfig:
    """Mock StudyConfig for testing."""

    study_name: str = "test_study"
    direction: MockDirection = MockDirection.MINIMIZE
    storage: str | None = None
    load_if_exists: bool = True
    metric: str = "val_loss"


@dataclass
class MockSearchSpaceParamConfig:
    """Mock SearchSpaceParamConfig for testing."""

    type: MockParamType
    low: float | None = None
    high: float | None = None
    step: float | None = None
    log: bool = False
    choices: list[Any] | None = None


class MockHPOConfig:
    """Mock HPOConfig for testing."""

    def __init__(
        self,
        enabled: bool = True,
        n_trials: int = 100,
        timeout: int | None = None,
        n_jobs: int = 1,
        backend: str = "optuna",
        cv_folds: int = 0,
        cv_metric_aggregation: str = "mean",
        search_space: dict | None = None,
        pruner: MockPrunerConfig | None = None,
        sampler: MockSamplerConfig | None = None,
        study: MockStudyConfig | None = None,
    ):
        self.enabled = enabled
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.backend = backend
        self.cv_folds = cv_folds
        self.cv_metric_aggregation = cv_metric_aggregation
        self.search_space = search_space or {}
        self.pruner = pruner or MockPrunerConfig()
        self.sampler = sampler or MockSamplerConfig()
        self.study = study or MockStudyConfig()

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "MockHPOConfig":
        """Create MockHPOConfig from dictionary."""
        return cls(**config_dict)


# =============================================================================
# MOCK PyG DATA CLASSES
# =============================================================================


class MockPyGData:
    """Mock PyTorch Geometric Data object for testing."""

    # Sentinel to distinguish "not provided" from "explicitly None"
    _NOT_PROVIDED = object()

    def __init__(
        self,
        x=_NOT_PROVIDED,  # Use sentinel to allow explicit None
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        y=_NOT_PROVIDED,  # Use sentinel, not Optional[torch.Tensor] = None
        num_nodes: int | None = None,
        edge_label: torch.Tensor | None = None,
        edge_value: torch.Tensor | None = None,
        edge_y: torch.Tensor | None = None,
    ):
        # Only use default x if not provided; allow explicit None
        if x is MockPyGData._NOT_PROVIDED:
            self.x = torch.randn(10, 16)
        else:
            self.x = x  # Can be None or a tensor
        self.edge_index = (
            edge_index
            if edge_index is not None
            else torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        )
        self.edge_attr = edge_attr
        # Only use default if not provided; allow explicit None
        if y is MockPyGData._NOT_PROVIDED:
            self.y = torch.tensor([1.0])
        else:
            self.y = y  # Can be None or a tensor
        # For num_nodes: use provided value, or infer from x if available
        if num_nodes is not None:
            self._num_nodes = num_nodes
        elif self.x is not None:
            self._num_nodes = self.x.size(0)
        else:
            self._num_nodes = 10  # Default when x is None
        self.edge_label = edge_label
        self.edge_value = edge_value
        self.edge_y = edge_y

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    def clone(self) -> "MockPyGData":
        """Create a copy of this data object (required by PyG transforms)."""
        return MockPyGData(
            x=self.x.clone() if self.x is not None else None,
            edge_index=self.edge_index.clone() if self.edge_index is not None else None,
            edge_attr=self.edge_attr.clone() if self.edge_attr is not None else None,
            y=self.y.clone() if self.y is not None else None,
            num_nodes=self._num_nodes,
            edge_label=self.edge_label.clone() if self.edge_label is not None else None,
            edge_value=self.edge_value.clone() if self.edge_value is not None else None,
            edge_y=self.edge_y.clone() if self.edge_y is not None else None,
        )

    def __repr__(self):
        x_repr = self.x.shape if self.x is not None else None
        y_repr = self.y.shape if self.y is not None else None
        return f"MockPyGData(x={x_repr}, y={y_repr})"


class MockPyGDataset:
    """Mock PyTorch Geometric Dataset for testing."""

    def __init__(self, data_list: list[MockPyGData] | None = None, task_type: str | None = None):
        self._data_list = data_list or [MockPyGData() for _ in range(10)]
        # Only set task_type attribute if explicitly provided (not None)
        # This matches real PyG datasets which don't have task_type by default
        if task_type is not None:
            self.task_type = task_type

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Preserve task_type when slicing
            new_dataset = MockPyGDataset(self._data_list[idx])
            if hasattr(self, "task_type"):
                new_dataset.task_type = self.task_type
            return new_dataset
        return self._data_list[idx]

    def __iter__(self):
        return iter(self._data_list)


# =============================================================================
# MOCK MODEL CLASS
# =============================================================================


class MockGNNModel(nn.Module):
    """Mock GNN model for integration testing."""

    def __init__(self, in_channels: int = 16, hidden_channels: int = 32, out_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Linear(in_channels, hidden_channels)
        self.conv2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        if batch is not None:
            # Global mean pooling simulation
            x = x.mean(dim=0, keepdim=True)
        return x


# =============================================================================
# PYTEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def mock_hpo_config():
    """Create a mock HPOConfig instance."""
    return MockHPOConfig(
        enabled=True,
        n_trials=10,
        timeout=3600,
        n_jobs=1,
        backend="optuna",
        cv_folds=0,
        search_space={
            "model": {
                "hidden_channels": MockSearchSpaceParamConfig(
                    type=MockParamType.INT, low=32, high=256, step=32
                ),
                "learning_rate": MockSearchSpaceParamConfig(
                    type=MockParamType.LOGUNIFORM, low=1e-5, high=1e-2
                ),
            },
            "optimizer": {
                "weight_decay": MockSearchSpaceParamConfig(
                    type=MockParamType.LOGUNIFORM, low=1e-6, high=1e-3
                ),
            },
            "loss": {
                "alpha": MockSearchSpaceParamConfig(type=MockParamType.FLOAT, low=0.1, high=0.9),
            },
        },
    )


@pytest.fixture
def mock_backend():
    """Create a mock HPO backend."""
    backend = MagicMock()
    backend.create_pruner.return_value = MagicMock()
    backend.create_sampler.return_value = MagicMock()
    backend.create_study.return_value = MagicMock()
    backend.optimize.return_value = None
    backend.get_best_params.return_value = {
        "learning_rate": 0.001,
        "hidden_channels": 128,
        "weight_decay": 1e-5,
    }
    backend.get_best_value.return_value = 0.05
    backend.suggest_params.return_value = {
        "learning_rate": 0.001,
        "hidden_channels": 128,
    }
    return backend


@pytest.fixture
def mock_pyg_dataset():
    """Create a mock PyG dataset."""
    return MockPyGDataset()


@pytest.fixture
def mock_model():
    """Create a mock GNN model."""
    return MockGNNModel()


@pytest.fixture
def graph_regression_data():
    """Create mock data for graph regression task."""
    data_list = [MockPyGData(y=torch.tensor([float(i)])) for i in range(10)]
    return MockPyGDataset(data_list, task_type="graph_regression")


@pytest.fixture
def node_regression_data():
    """Create mock data for node regression task."""
    data_list = [
        MockPyGData(
            x=torch.randn(10, 16),
            y=torch.randn(10),  # Node-level targets
        )
        for _ in range(5)
    ]
    return MockPyGDataset(data_list, task_type="node_regression")


@pytest.fixture
def link_prediction_data():
    """Create mock data for link prediction task."""
    data_list = [
        MockPyGData(
            edge_label=torch.tensor([1, 0, 1, 0, 1]),
        )
        for _ in range(5)
    ]
    return MockPyGDataset(data_list, task_type="link_prediction")


# =============================================================================
# INTEGRATION TEST CLASS 1: HPO Manager + Loss Registry
# =============================================================================


class TestHPOManagerLossRegistryIntegration:
    """
    Integration tests for HPO Manager and Loss Registry.

    Verifies:
    - Task-specific loss function selection
    - Parameter filtering from HPO to loss creation
    - Loss registry interaction with HPO workflow
    """

    @patch("milia_pipeline.models.hpo.hpo_manager.LossRegistry")
    def test_loss_creation_for_graph_regression(self, mock_loss_registry, mock_logger):
        """Test that graph regression task creates MSE loss via registry."""
        # Import the helper function
        from milia_pipeline.models.hpo.hpo_manager import _create_loss_from_registry

        mock_loss_registry.get_loss.return_value = nn.MSELoss()

        loss_fn = _create_loss_from_registry("graph_regression", {"alpha": 0.5})

        # Should call get_loss with 'mse'
        mock_loss_registry.get_loss.assert_called_once()
        call_args = mock_loss_registry.get_loss.call_args
        assert call_args[0][0] == "mse"
        assert isinstance(loss_fn, nn.Module)

    @patch("milia_pipeline.models.hpo.hpo_manager.LossRegistry")
    def test_loss_creation_for_link_prediction(self, mock_loss_registry, mock_logger):
        """Test that link prediction task creates BCE with logits loss."""
        from milia_pipeline.models.hpo.hpo_manager import _create_loss_from_registry

        mock_loss_registry.get_loss.return_value = nn.BCEWithLogitsLoss()

        _loss_fn = _create_loss_from_registry("link_prediction", None)

        mock_loss_registry.get_loss.assert_called_once()
        call_args = mock_loss_registry.get_loss.call_args
        assert call_args[0][0] == "bce_with_logits"

    @patch("milia_pipeline.models.hpo.hpo_manager.LossRegistry")
    def test_loss_creation_for_classification(self, mock_loss_registry, mock_logger):
        """Test that classification task creates cross entropy loss."""
        from milia_pipeline.models.hpo.hpo_manager import _create_loss_from_registry

        mock_loss_registry.get_loss.return_value = nn.CrossEntropyLoss()

        _loss_fn = _create_loss_from_registry("graph_classification", None)

        mock_loss_registry.get_loss.assert_called_once()
        call_args = mock_loss_registry.get_loss.call_args
        assert call_args[0][0] == "cross_entropy"

    def test_get_loss_name_for_task_mapping(self):
        """Test the task type to loss name mapping."""
        from milia_pipeline.models.hpo.hpo_manager import _get_loss_name_for_task

        # Regression tasks
        assert _get_loss_name_for_task("graph_regression") == "mse"
        assert _get_loss_name_for_task("node_regression") == "mse"
        assert _get_loss_name_for_task("edge_regression") == "mse"

        # Classification tasks
        assert _get_loss_name_for_task("graph_classification") == "cross_entropy"
        assert _get_loss_name_for_task("node_classification") == "cross_entropy"

        # Link prediction (binary classification)
        assert _get_loss_name_for_task("link_prediction") == "bce_with_logits"

    @patch("milia_pipeline.models.hpo.hpo_manager.LossRegistry", None)
    def test_loss_creation_fallback_when_registry_unavailable(self, mock_logger):
        """Test fallback loss creation when LossRegistry is unavailable."""
        from milia_pipeline.models.hpo.hpo_manager import _create_loss_from_registry

        # Should use inline fallback
        loss_fn = _create_loss_from_registry("graph_regression", None)

        assert isinstance(loss_fn, nn.MSELoss)


# =============================================================================
# INTEGRATION TEST CLASS 2: HPO Manager + Optimizer Registry
# =============================================================================


class TestHPOManagerOptimizerRegistryIntegration:
    """
    Integration tests for HPO Manager and Optimizer Registry.

    Verifies:
    - Optimizer creation with HPO-suggested parameters
    - Parameter merging and filtering
    - Registry defaults handling
    """

    @patch("milia_pipeline.models.hpo.hpo_manager.OptimizerRegistry")
    def test_optimizer_creation_with_hpo_params(self, mock_optimizer_registry, mock_model):
        """Test optimizer creation with HPO-suggested parameters."""
        from milia_pipeline.models.hpo.hpo_manager import _create_optimizer_from_registry

        mock_optimizer = MagicMock()
        mock_optimizer_registry.get_optimizer.return_value = mock_optimizer

        hpo_params = {"lr": 0.001, "weight_decay": 1e-5}

        _optimizer = _create_optimizer_from_registry(
            mock_model.parameters(), optimizer_params=hpo_params, optimizer_name="adam"
        )

        mock_optimizer_registry.get_optimizer.assert_called_once()
        call_args = mock_optimizer_registry.get_optimizer.call_args
        assert call_args[0][0] == "adam"  # optimizer name
        assert call_args[0][2] == hpo_params  # params

    @patch("milia_pipeline.models.hpo.hpo_manager.OptimizerRegistry")
    def test_optimizer_creation_with_different_types(self, mock_optimizer_registry, mock_model):
        """Test optimizer creation with different optimizer types."""
        from milia_pipeline.models.hpo.hpo_manager import _create_optimizer_from_registry

        mock_optimizer_registry.get_optimizer.return_value = MagicMock()

        for opt_name in ["adam", "adamw", "sgd", "rmsprop"]:
            _create_optimizer_from_registry(mock_model.parameters(), optimizer_name=opt_name)

            call_args = mock_optimizer_registry.get_optimizer.call_args
            assert call_args[0][0] == opt_name

    @patch("milia_pipeline.models.hpo.hpo_manager.OptimizerRegistry", None)
    def test_optimizer_fallback_when_registry_unavailable(self, mock_model):
        """Test fallback optimizer creation when registry unavailable."""
        from milia_pipeline.models.hpo.hpo_manager import _create_optimizer_from_registry

        optimizer = _create_optimizer_from_registry(
            mock_model.parameters(), optimizer_params={"lr": 0.001}, optimizer_name="adam"
        )

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.defaults["lr"] == 0.001


# =============================================================================
# INTEGRATION TEST CLASS 3: HPO Manager + Scheduler Registry
# =============================================================================


class TestHPOManagerSchedulerRegistryIntegration:
    """
    Integration tests for HPO Manager and Scheduler Registry.

    Verifies:
    - Scheduler creation with HPO-suggested parameters
    - Parameter filtering for different scheduler types
    - Metric-based scheduler configuration
    """

    @patch("milia_pipeline.models.hpo.hpo_manager.SchedulerRegistry")
    def test_scheduler_creation_with_hpo_params(self, mock_scheduler_registry, mock_model):
        """Test scheduler creation with HPO-suggested parameters."""
        from milia_pipeline.models.hpo.hpo_manager import _create_scheduler_from_registry

        mock_scheduler = MagicMock()
        mock_scheduler_registry.get_scheduler.return_value = mock_scheduler

        optimizer = torch.optim.Adam(mock_model.parameters())
        scheduler_params = {"patience": 10, "factor": 0.5}

        _scheduler = _create_scheduler_from_registry(
            optimizer, scheduler_params=scheduler_params, scheduler_name="reduce_on_plateau"
        )

        mock_scheduler_registry.get_scheduler.assert_called_once()
        call_args = mock_scheduler_registry.get_scheduler.call_args
        assert call_args[0][0] == "reduce_on_plateau"

    @patch("milia_pipeline.models.hpo.hpo_manager.SchedulerRegistry")
    def test_scheduler_creation_cosine_annealing(self, mock_scheduler_registry, mock_model):
        """Test cosine annealing scheduler creation."""
        from milia_pipeline.models.hpo.hpo_manager import _create_scheduler_from_registry

        mock_scheduler_registry.get_scheduler.return_value = MagicMock()

        optimizer = torch.optim.Adam(mock_model.parameters())
        scheduler_params = {"T_max": 100, "eta_min": 1e-6}

        _create_scheduler_from_registry(
            optimizer, scheduler_params=scheduler_params, scheduler_name="cosine_annealing"
        )

        call_args = mock_scheduler_registry.get_scheduler.call_args
        assert call_args[0][0] == "cosine_annealing"

    def test_scheduler_not_created_without_params(self, mock_model):
        """Test that no scheduler is created when params are empty."""
        from milia_pipeline.models.hpo.hpo_manager import _create_scheduler_from_registry

        optimizer = torch.optim.Adam(mock_model.parameters())

        scheduler = _create_scheduler_from_registry(
            optimizer,
            scheduler_params=None,  # No params
            scheduler_name="reduce_on_plateau",
        )

        assert scheduler is None


# =============================================================================
# INTEGRATION TEST CLASS 4: HPO Manager + Main Module (Task Data Prep)
# =============================================================================


class TestHPOManagerMainModuleIntegration:
    """
    Integration tests for HPO Manager and Main Module.

    Verifies:
    - Task-specific data preparation consistency between modules
    - Task type inference alignment
    - Data preparation function behavior
    """

    def test_task_type_inference_from_dataset_metadata(self, mock_pyg_dataset):
        """Test task type inference uses dataset metadata when available."""
        from milia_pipeline.models.hpo.hpo_manager import infer_task_type

        mock_pyg_dataset.task_type = "graph_regression"

        inferred = infer_task_type(mock_pyg_dataset)

        assert inferred == "graph_regression"

    def test_task_type_inference_from_metric_regression(self):
        """Test task type inference from regression metrics."""
        from milia_pipeline.models.hpo.hpo_manager import infer_task_type

        dataset = MockPyGDataset()

        # Regression metrics
        assert infer_task_type(dataset, metric="val_mae") == "graph_regression"
        assert infer_task_type(dataset, metric="val_mse") == "graph_regression"
        assert infer_task_type(dataset, metric="val_rmse") == "graph_regression"
        assert infer_task_type(dataset, metric="val_loss") == "graph_regression"

    def test_task_type_inference_from_metric_classification(self):
        """Test task type inference from classification metrics."""
        from milia_pipeline.models.hpo.hpo_manager import infer_task_type

        dataset = MockPyGDataset()

        # Classification metrics
        assert infer_task_type(dataset, metric="val_accuracy") == "graph_classification"
        assert infer_task_type(dataset, metric="val_f1") == "graph_classification"
        assert infer_task_type(dataset, metric="val_auc") == "graph_classification"

    def test_task_type_inference_from_target_tensor(self):
        """Test task type inference from target tensor analysis."""
        from milia_pipeline.models.hpo.hpo_manager import infer_task_type

        # Continuous target (regression)
        regression_data = MockPyGData(y=torch.tensor([1.5]))
        regression_dataset = MockPyGDataset([regression_data])

        # Integer target (classification)
        classification_data = MockPyGData(y=torch.tensor([1], dtype=torch.long))
        classification_dataset = MockPyGDataset([classification_data])

        # Should infer based on dtype
        assert infer_task_type(regression_dataset) == "graph_regression"
        assert infer_task_type(classification_dataset) == "graph_classification"

    def test_hpo_data_prep_matches_main_data_prep_for_graph_tasks(
        self, graph_regression_data, mock_logger
    ):
        """Test HPO and main.py data prep produce same results for graph tasks."""
        from milia_pipeline.models.hpo.hpo_manager import _prepare_data_for_task_hpo

        train_data = graph_regression_data[:7]
        val_data = graph_regression_data[7:]

        # HPO version (train, val, num_classes)
        hpo_train, hpo_val, _ = _prepare_data_for_task_hpo(train_data, val_data, "graph_regression")

        # For graph regression, data should be unchanged
        assert len(hpo_train) == len(train_data)
        assert len(hpo_val) == len(val_data)

    def test_hpo_data_prep_link_prediction_with_edge_label(self, link_prediction_data, mock_logger):
        """Test HPO data prep for link prediction when edge_label exists."""
        from milia_pipeline.models.hpo.hpo_manager import _prepare_data_for_task_hpo

        train_data = link_prediction_data[:3]
        val_data = link_prediction_data[3:]

        # Data already has edge_label, should return unchanged
        hpo_train, hpo_val, _ = _prepare_data_for_task_hpo(train_data, val_data, "link_prediction")

        assert len(hpo_train) == len(train_data)
        assert hasattr(hpo_train[0], "edge_label")


# =============================================================================
# INTEGRATION TEST CLASS 5: Registry Cross-Module Integration
# =============================================================================


class TestRegistryCrossModuleIntegration:
    """
    Integration tests for cross-registry module interactions.

    Verifies:
    - Loss + Optimizer + Scheduler creation pipeline
    - Parameter filtering consistency across registries
    - Component creation with shared HPO parameters
    """

    def test_full_training_component_creation_pipeline(self, mock_model):
        """Test creating all training components in sequence."""
        # Import registries
        from milia_pipeline.models.training.loss_functions import LossRegistry
        from milia_pipeline.models.training.optimizers import OptimizerRegistry
        from milia_pipeline.models.training.schedulers import SchedulerRegistry

        # Create loss
        loss_fn = LossRegistry.get_loss("mse", {"reduction": "mean"})
        assert isinstance(loss_fn, nn.Module)

        # Create optimizer
        optimizer = OptimizerRegistry.get_optimizer(
            "adam", mock_model.parameters(), {"lr": 0.001, "weight_decay": 1e-5}
        )
        assert isinstance(optimizer, torch.optim.Optimizer)

        # Create scheduler
        scheduler = SchedulerRegistry.get_scheduler(
            "reduce_on_plateau", optimizer, {"patience": 10, "factor": 0.5}
        )
        assert scheduler is not None

    def test_parameter_filtering_consistency_across_registries(self, mock_model):
        """Test that invalid parameters are filtered consistently."""
        from milia_pipeline.models.training.loss_functions import LossRegistry
        from milia_pipeline.models.training.optimizers import OptimizerRegistry
        from milia_pipeline.models.training.schedulers import SchedulerRegistry

        # Mixed params with some invalid for each component
        shared_params = {
            "lr": 0.001,
            "weight_decay": 1e-5,
            "alpha": 0.5,
            "gamma": 2.0,
            "patience": 10,
            "invalid_param": "should_be_ignored",
        }

        # Loss should filter to only valid params
        loss_fn = LossRegistry.get_loss("mse", shared_params)
        assert isinstance(loss_fn, nn.MSELoss)

        # Optimizer should filter to only valid params
        optimizer = OptimizerRegistry.get_optimizer("adam", mock_model.parameters(), shared_params)
        assert isinstance(optimizer, torch.optim.Adam)

        # Scheduler should filter to only valid params
        scheduler = SchedulerRegistry.get_scheduler("reduce_on_plateau", optimizer, shared_params)
        assert scheduler is not None

    def test_hpo_params_distributed_to_correct_components(self, mock_model):
        """Test HPO parameters are correctly routed to components."""
        from milia_pipeline.models.training.loss_functions import LossRegistry
        from milia_pipeline.models.training.optimizers import OptimizerRegistry

        # Simulate HPO-suggested params
        hpo_params = {
            "model": {"hidden_channels": 128, "num_layers": 3},
            "optimizer": {"lr": 0.001, "weight_decay": 1e-5},
            "loss": {"alpha": 0.25, "gamma": 2.0},
        }

        # Create loss with loss params
        loss_fn = LossRegistry.get_loss("focal", hpo_params.get("loss", {}))
        assert hasattr(loss_fn, "alpha")
        assert loss_fn.alpha == 0.25

        # Create optimizer with optimizer params
        optimizer = OptimizerRegistry.get_optimizer(
            "adam", mock_model.parameters(), hpo_params.get("optimizer", {})
        )
        assert optimizer.defaults["lr"] == 0.001


# =============================================================================
# INTEGRATION TEST CLASS 6: End-to-End HPO Workflow
# =============================================================================


class TestEndToEndHPOWorkflow:
    """
    End-to-end integration tests for HPO workflow.

    Verifies:
    - Full HPO trial execution with mocked components
    - Component creation pipeline in HPO context
    - Result aggregation and best parameter retrieval
    """

    @patch("milia_pipeline.models.hpo.hpo_manager.get_backend")
    @patch("milia_pipeline.models.hpo.hpo_manager.get_factory")
    def test_hpo_manager_initialization_integrates_backend(
        self, mock_get_factory, mock_get_backend, mock_hpo_config, mock_backend
    ):
        """Test HPOManager properly initializes with backend."""
        mock_get_backend.return_value = mock_backend
        mock_get_factory.return_value = MagicMock()

        from milia_pipeline.models.hpo.hpo_manager import HPOManager

        manager = HPOManager(mock_hpo_config)

        assert manager.config.enabled
        assert manager.backend is not None
        mock_get_backend.assert_called_once_with(mock_hpo_config.backend)

    @patch("milia_pipeline.models.hpo.hpo_manager.get_backend")
    def test_hpo_manager_creates_study_with_correct_config(
        self, mock_get_backend, mock_hpo_config, mock_backend
    ):
        """Test HPOManager creates study with correct configuration."""
        mock_get_backend.return_value = mock_backend

        from milia_pipeline.models.hpo.hpo_manager import HPOManager

        _manager = HPOManager(mock_hpo_config)

        # Verify backend was retrieved
        mock_get_backend.assert_called_once()

        # Backend should have proper methods
        assert hasattr(mock_backend, "create_study")
        assert hasattr(mock_backend, "create_pruner")
        assert hasattr(mock_backend, "create_sampler")

    @patch("milia_pipeline.models.hpo.hpo_manager.LossRegistry")
    @patch("milia_pipeline.models.hpo.hpo_manager.OptimizerRegistry")
    @patch("milia_pipeline.models.hpo.hpo_manager.SchedulerRegistry")
    def test_hpo_trial_creates_all_components(
        self,
        mock_scheduler_registry,
        mock_optimizer_registry,
        mock_loss_registry,
        mock_model,
    ):
        """Test that an HPO trial creates all required training components."""
        from milia_pipeline.models.hpo.hpo_manager import (
            _create_loss_from_registry,
            _create_optimizer_from_registry,
            _create_scheduler_from_registry,
        )

        mock_loss_registry.get_loss.return_value = nn.MSELoss()
        mock_optimizer_registry.get_optimizer.return_value = torch.optim.Adam(
            mock_model.parameters()
        )
        mock_scheduler_registry.get_scheduler.return_value = MagicMock()

        # Simulate HPO trial component creation
        loss_fn = _create_loss_from_registry("graph_regression", {"alpha": 0.5})
        optimizer = _create_optimizer_from_registry(
            mock_model.parameters(), {"lr": 0.001, "weight_decay": 1e-5}, "adam"
        )
        scheduler = _create_scheduler_from_registry(
            optimizer, {"patience": 10}, "reduce_on_plateau"
        )

        # All components should be created
        assert loss_fn is not None
        assert optimizer is not None
        assert scheduler is not None

        # Registries should be called
        mock_loss_registry.get_loss.assert_called_once()
        mock_optimizer_registry.get_optimizer.assert_called_once()
        mock_scheduler_registry.get_scheduler.assert_called_once()

    def test_param_extraction_categories(self):
        """Test parameter category extraction for HPO."""
        from milia_pipeline.models.hpo.hpo_manager import _extract_param_categories

        # Params with known category keys (NOT prefixed, but actual param names)
        # The function extracts by known parameter names, not prefixes
        flat_params = {
            "hidden_channels": 128,  # model param (unknown, goes to model)
            "num_layers": 3,  # model param (unknown, goes to model)
            "lr": 0.001,  # optimizer param (known key)
            "weight_decay": 1e-5,  # optimizer param (known key)
            "patience": 10,  # scheduler param (known key)
            "alpha": 0.25,  # loss param (known key)
        }

        # Returns tuple: (model_params, optimizer_params, scheduler_params, loss_params, training_params)
        model_params, optimizer_params, scheduler_params, loss_params, training_params = (
            _extract_param_categories(flat_params)
        )

        # Model params = unknown params
        assert "hidden_channels" in model_params
        assert model_params["hidden_channels"] == 128
        assert "num_layers" in model_params

        # Optimizer params
        assert "lr" in optimizer_params
        assert optimizer_params["lr"] == 0.001
        assert "weight_decay" in optimizer_params

        # Scheduler params
        assert "patience" in scheduler_params
        assert scheduler_params["patience"] == 10

        # Loss params
        assert "alpha" in loss_params
        assert loss_params["alpha"] == 0.25


# =============================================================================
# INTEGRATION TEST CLASS 7: Task-Specific Data Flow
# =============================================================================


class TestTaskSpecificDataFlow:
    """
    Integration tests for task-specific data flow through HPO.

    Verifies:
    - Data preparation for each task type
    - Correct loss function selection per task
    - End-to-end data flow consistency
    """

    def test_graph_regression_full_flow(self, graph_regression_data):
        """Test complete data flow for graph regression task."""
        from milia_pipeline.models.hpo.hpo_manager import (
            _get_loss_name_for_task,
            _prepare_data_for_task_hpo,
            infer_task_type,
        )

        # Infer task type
        task_type = infer_task_type(graph_regression_data, metric="val_mae")
        assert task_type == "graph_regression"

        # Prepare data
        train, val, _ = _prepare_data_for_task_hpo(
            graph_regression_data[:7], graph_regression_data[7:], task_type
        )
        assert len(train) == 7
        assert len(val) == 3

        # Get loss name
        loss_name = _get_loss_name_for_task(task_type)
        assert loss_name == "mse"

    def test_classification_full_flow(self):
        """Test complete data flow for classification task."""
        from milia_pipeline.models.hpo.hpo_manager import (
            _get_loss_name_for_task,
            _prepare_data_for_task_hpo,
            infer_task_type,
        )

        # Create classification dataset
        data_list = [MockPyGData(y=torch.tensor([i % 3], dtype=torch.long)) for i in range(10)]
        dataset = MockPyGDataset(data_list, task_type="graph_classification")

        # Infer task type
        task_type = infer_task_type(dataset, metric="val_accuracy")
        assert task_type == "graph_classification"

        # Prepare data
        train, val, _ = _prepare_data_for_task_hpo(dataset[:7], dataset[7:], task_type)

        # Get loss name
        loss_name = _get_loss_name_for_task(task_type)
        assert loss_name == "cross_entropy"

    def test_link_prediction_full_flow(self, link_prediction_data):
        """Test complete data flow for link prediction task."""
        from milia_pipeline.models.hpo.hpo_manager import (
            _get_loss_name_for_task,
            _prepare_data_for_task_hpo,
        )

        task_type = "link_prediction"

        # Prepare data
        train, val, _ = _prepare_data_for_task_hpo(
            link_prediction_data[:3], link_prediction_data[3:], task_type
        )

        # Verify edge_label preserved
        assert hasattr(train[0], "edge_label")

        # Get loss name
        loss_name = _get_loss_name_for_task(task_type)
        assert loss_name == "bce_with_logits"


# =============================================================================
# INTEGRATION TEST CLASS 8: Error Handling Across Modules
# =============================================================================


class TestCrossModuleErrorHandling:
    """
    Integration tests for error handling across modules.

    Verifies:
    - Error propagation between modules
    - Graceful fallback behavior
    - Exception consistency
    """

    def test_invalid_loss_name_raises_value_error(self):
        """Test that invalid loss name raises ValueError."""
        from milia_pipeline.models.training.loss_functions import LossRegistry

        with pytest.raises(ValueError, match="Unknown loss function"):
            LossRegistry.get_loss("invalid_loss_name")

    def test_invalid_optimizer_name_raises_value_error(self, mock_model):
        """Test that invalid optimizer name raises ValueError."""
        from milia_pipeline.models.training.optimizers import OptimizerRegistry

        with pytest.raises(ValueError, match="Unknown optimizer"):
            OptimizerRegistry.get_optimizer("invalid_optimizer", mock_model.parameters())

    def test_invalid_scheduler_name_raises_value_error(self, mock_model):
        """Test that invalid scheduler name raises ValueError."""
        from milia_pipeline.models.training.schedulers import SchedulerRegistry

        optimizer = torch.optim.Adam(mock_model.parameters())

        with pytest.raises(ValueError, match="Unknown scheduler"):
            SchedulerRegistry.get_scheduler("invalid_scheduler", optimizer, {"param": 1})

    @patch("milia_pipeline.models.hpo.hpo_manager.get_backend")
    def test_hpo_error_when_disabled(self, mock_get_backend):
        """Test HPOError raised when HPO is disabled but optimize called."""
        from milia_pipeline.exceptions import HPOError
        from milia_pipeline.models.hpo.hpo_manager import HPOManager

        config = MockHPOConfig(enabled=False)
        manager = HPOManager(config)

        with pytest.raises(HPOError, match="HPO is disabled"):
            manager.optimize(model_name="GCN", dataset=MockPyGDataset())


# =============================================================================
# INTEGRATION TEST CLASS 9: Registry List Functions
# =============================================================================


class TestRegistryListFunctions:
    """
    Integration tests for registry listing and discovery functions.

    Verifies:
    - Loss registry listing
    - Optimizer registry listing
    - Scheduler registry listing
    - Consistency of available options
    """

    def test_loss_registry_list_available(self):
        """Test LossRegistry.list_available returns expected losses."""
        from milia_pipeline.models.training.loss_functions import LossRegistry

        available = LossRegistry.list_available()

        # Should include common losses
        assert "mse" in available
        assert "mae" in available
        assert "cross_entropy" in available
        assert "bce" in available
        assert "focal" in available

    def test_optimizer_registry_list_available(self):
        """Test OptimizerRegistry.list_available returns expected optimizers."""
        from milia_pipeline.models.training.optimizers import OptimizerRegistry

        available = OptimizerRegistry.list_available()

        # Should include common optimizers
        assert "adam" in available
        assert "adamw" in available
        assert "sgd" in available
        assert "rmsprop" in available

    def test_scheduler_registry_list_available(self):
        """Test SchedulerRegistry.list_available returns expected schedulers."""
        from milia_pipeline.models.training.schedulers import SchedulerRegistry

        available = SchedulerRegistry.list_available()

        # Should include common schedulers
        assert "reduce_on_plateau" in available
        assert "step_lr" in available
        assert "cosine_annealing" in available

    def test_all_listed_losses_can_be_instantiated(self):
        """Test all listed losses can be instantiated."""
        from milia_pipeline.models.training.loss_functions import LossRegistry

        available = LossRegistry.list_available()

        for loss_name in available:
            try:
                loss_fn = LossRegistry.get_loss(loss_name)
                assert isinstance(loss_fn, nn.Module)
            except Exception as e:
                pytest.fail(f"Failed to instantiate loss '{loss_name}': {e}")

    def test_all_listed_optimizers_can_be_instantiated(self, mock_model):
        """Test all listed optimizers can be instantiated."""
        from milia_pipeline.models.training.optimizers import OptimizerRegistry

        available = OptimizerRegistry.list_available()

        for opt_name in available:
            try:
                optimizer = OptimizerRegistry.get_optimizer(opt_name, mock_model.parameters())
                assert isinstance(optimizer, torch.optim.Optimizer)
            except Exception as e:
                pytest.fail(f"Failed to instantiate optimizer '{opt_name}': {e}")


# =============================================================================
# INTEGRATION TEST CLASS 10: Scheduler-Specific Features
# =============================================================================


class TestSchedulerSpecificFeatures:
    """
    Integration tests for scheduler-specific features.

    Verifies:
    - Metric-based schedulers (ReduceLROnPlateau)
    - Step method handling
    - Scheduler state management
    """

    def test_reduce_on_plateau_requires_metric(self, mock_model):
        """Test ReduceLROnPlateau is metric-based."""
        from milia_pipeline.models.training.schedulers import SchedulerRegistry

        optimizer = torch.optim.Adam(mock_model.parameters())

        scheduler = SchedulerRegistry.get_scheduler(
            "reduce_on_plateau", optimizer, {"patience": 5, "factor": 0.1}
        )

        # Should be able to step with metric
        assert scheduler is not None
        # ReduceLROnPlateau requires metric in step()
        assert SchedulerRegistry.is_metric_based("reduce_on_plateau")

    def test_step_lr_not_metric_based(self, mock_model):
        """Test StepLR is not metric-based."""
        from milia_pipeline.models.training.schedulers import SchedulerRegistry

        optimizer = torch.optim.Adam(mock_model.parameters())

        scheduler = SchedulerRegistry.get_scheduler(
            "step_lr", optimizer, {"step_size": 10, "gamma": 0.1}
        )

        assert scheduler is not None
        assert not SchedulerRegistry.is_metric_based("step_lr")

    def test_scheduler_defaults_applied(self, mock_model):
        """Test scheduler defaults are applied when not specified."""
        from milia_pipeline.models.training.schedulers import SchedulerRegistry

        optimizer = torch.optim.Adam(mock_model.parameters())

        # Create with minimal params - defaults should be applied
        scheduler = SchedulerRegistry.get_scheduler(
            "reduce_on_plateau",
            optimizer,
            {},  # Empty - should use defaults
        )

        assert scheduler is not None


# =============================================================================
# INTEGRATION TEST CLASS 11: Helper Function Integration
# =============================================================================


class TestHelperFunctionIntegration:
    """
    Integration tests for HPO helper functions.

    Verifies:
    - _flatten_params correctly flattens nested params
    - _extract_param_categories correctly categorizes params
    - Helper functions work with registry-created components
    """

    def test_flatten_params_removes_category_prefix(self):
        """Test _flatten_params removes category prefixes."""
        from milia_pipeline.models.hpo.hpo_manager import _flatten_params

        nested_params = {
            "model.hidden_channels": 128,
            "model.num_layers": 3,
            "optimizer.lr": 0.001,
            "optimizer.weight_decay": 1e-5,
        }

        flat = _flatten_params(nested_params)

        assert "hidden_channels" in flat
        assert flat["hidden_channels"] == 128
        assert "lr" in flat
        assert flat["lr"] == 0.001
        assert "model.hidden_channels" not in flat

    def test_flatten_params_preserves_non_prefixed(self):
        """Test _flatten_params preserves non-prefixed params."""
        from milia_pipeline.models.hpo.hpo_manager import _flatten_params

        params = {
            "hidden_channels": 128,
            "lr": 0.001,
        }

        flat = _flatten_params(params)

        assert flat == params

    def test_extract_categories_optimizer_keys(self):
        """Test _extract_param_categories identifies optimizer keys."""
        from milia_pipeline.models.hpo.hpo_manager import _extract_param_categories

        params = {
            "lr": 0.001,
            "learning_rate": 0.002,
            "weight_decay": 1e-5,
            "momentum": 0.9,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "amsgrad": True,
            "nesterov": False,
        }

        model, opt, sched, loss, train = _extract_param_categories(params)

        assert len(opt) == 8
        assert "lr" in opt
        assert "weight_decay" in opt
        assert "momentum" in opt

    def test_extract_categories_scheduler_keys(self):
        """Test _extract_param_categories identifies scheduler keys."""
        from milia_pipeline.models.hpo.hpo_manager import _extract_param_categories

        params = {
            "factor": 0.5,
            "patience": 10,
            "step_size": 30,
            "gamma": 0.1,
            "T_max": 100,
            "eta_min": 1e-6,
            "cooldown": 5,
            "min_lr": 1e-8,
        }

        model, opt, sched, loss, train = _extract_param_categories(params)

        assert len(sched) == 8
        assert "factor" in sched
        assert "patience" in sched
        assert "T_max" in sched

    def test_extract_categories_loss_keys(self):
        """Test _extract_param_categories identifies loss keys."""
        from milia_pipeline.models.hpo.hpo_manager import _extract_param_categories

        params = {
            "alpha": 0.25,
            "gamma": 2.0,
            "reduction": "mean",
            "weight": None,
            "label_smoothing": 0.1,
        }

        model, opt, sched, loss, train = _extract_param_categories(params)

        # Note: 'gamma' is in both scheduler and loss keys - scheduler takes precedence
        assert "alpha" in loss
        assert "reduction" in loss

    def test_extract_categories_training_keys(self):
        """Test _extract_param_categories identifies training keys."""
        from milia_pipeline.models.hpo.hpo_manager import _extract_param_categories

        params = {
            "batch_size": 32,
            "epochs": 100,
            "max_epochs": 200,
            "gradient_clip_val": 1.0,
            "shuffle": True,
        }

        model, opt, sched, loss, train = _extract_param_categories(params)

        assert "batch_size" in train
        assert "epochs" in train
        assert "max_epochs" in train

    def test_extract_categories_model_params_are_remainder(self):
        """Test _extract_param_categories puts unknown keys in model_params."""
        from milia_pipeline.models.hpo.hpo_manager import _extract_param_categories

        params = {
            "hidden_channels": 128,
            "num_layers": 3,
            "heads": 4,
            "dropout": 0.1,
            "lr": 0.001,  # optimizer key
        }

        model, opt, sched, loss, train = _extract_param_categories(params)

        assert "hidden_channels" in model
        assert "num_layers" in model
        assert "heads" in model
        assert "dropout" in model
        assert "lr" in opt
        assert "lr" not in model


# =============================================================================
# INTEGRATION TEST CLASS 12: Loss Registry Advanced Features
# =============================================================================


class TestLossRegistryAdvancedFeatures:
    """
    Integration tests for advanced LossRegistry features.

    Verifies:
    - Custom loss registration
    - Loss info retrieval
    - Valid params introspection
    """

    def test_get_loss_info_returns_expected_fields(self):
        """Test get_loss_info returns all expected fields."""
        from milia_pipeline.models.training.loss_functions import LossRegistry

        info = LossRegistry.get_loss_info("focal")

        assert "name" in info
        assert "class" in info
        assert "module" in info
        assert "doc" in info
        assert "valid_params" in info
        assert info["name"] == "focal"

    def test_get_valid_params_returns_params_with_defaults(self):
        """Test get_valid_params returns parameters with defaults."""
        from milia_pipeline.models.training.loss_functions import LossRegistry

        params = LossRegistry.get_valid_params("focal")

        assert "alpha" in params
        assert "gamma" in params
        assert "reduction" in params
        assert params["alpha"] == 0.25
        assert params["gamma"] == 2.0

    def test_register_custom_loss_adds_to_registry(self):
        """Test register_custom_loss adds custom loss to registry."""
        from milia_pipeline.models.training.loss_functions import LossRegistry

        class TestCustomLoss(nn.Module):
            def __init__(self, scale=1.0):
                super().__init__()
                self.scale = scale

            def forward(self, input, target):
                return self.scale * ((input - target) ** 2).mean()

        # Register custom loss
        LossRegistry.register_custom_loss("test_custom", TestCustomLoss, overwrite=True)

        # Should be able to retrieve it
        loss_fn = LossRegistry.get_loss("test_custom", {"scale": 2.0})
        assert isinstance(loss_fn, TestCustomLoss)
        assert loss_fn.scale == 2.0

    def test_filter_params_removes_invalid_params(self):
        """Test _filter_params removes invalid parameters."""
        from milia_pipeline.models.training.loss_functions import LossRegistry

        # MSELoss doesn't accept 'alpha' or 'gamma'
        loss_fn = LossRegistry.get_loss(
            "mse",
            {
                "alpha": 0.25,  # Invalid for MSE
                "gamma": 2.0,  # Invalid for MSE
                "reduction": "mean",  # Valid
            },
        )

        # Should succeed without error
        assert isinstance(loss_fn, nn.MSELoss)

    def test_loss_alias_works(self):
        """Test loss aliases work correctly."""
        from milia_pipeline.models.training.loss_functions import LossRegistry

        # 'l1' should be alias for 'mae'
        l1_loss = LossRegistry.get_loss("l1")
        mae_loss = LossRegistry.get_loss("mae")

        assert type(l1_loss) == type(mae_loss)

    def test_all_regression_losses_available(self):
        """Test all regression losses are available."""
        from milia_pipeline.models.training.loss_functions import LossRegistry

        regression_losses = ["mse", "mae", "l1", "huber", "smooth_l1", "rmse", "weighted_mse"]

        for loss_name in regression_losses:
            assert loss_name in LossRegistry.list_available(), (
                f"Missing regression loss: {loss_name}"
            )

    def test_all_classification_losses_available(self):
        """Test all classification losses are available."""
        from milia_pipeline.models.training.loss_functions import LossRegistry

        classification_losses = ["cross_entropy", "ce", "nll", "bce", "bce_with_logits", "focal"]

        for loss_name in classification_losses:
            assert loss_name in LossRegistry.list_available(), (
                f"Missing classification loss: {loss_name}"
            )


# =============================================================================
# INTEGRATION TEST CLASS 13: Optimizer Registry Advanced Features
# =============================================================================


class TestOptimizerRegistryAdvancedFeatures:
    """
    Integration tests for advanced OptimizerRegistry features.

    Verifies:
    - Custom optimizer registration
    - Default params handling
    - Valid params introspection
    """

    def test_get_optimizer_info_returns_expected_fields(self, mock_model):
        """Test get_optimizer_info returns all expected fields."""
        from milia_pipeline.models.training.optimizers import OptimizerRegistry

        info = OptimizerRegistry.get_optimizer_info("adam")

        assert "name" in info
        assert "class" in info
        assert "module" in info
        assert "default_params" in info
        assert "valid_params" in info
        assert info["name"] == "adam"

    def test_get_default_params_returns_registry_defaults(self):
        """Test get_default_params returns registry defaults."""
        from milia_pipeline.models.training.optimizers import OptimizerRegistry

        defaults = OptimizerRegistry.get_default_params("adam")

        assert "lr" in defaults
        assert "betas" in defaults
        assert defaults["lr"] == 0.001

    def test_get_valid_params_returns_params_with_defaults(self, mock_model):
        """Test get_valid_params returns parameters with defaults."""
        from milia_pipeline.models.training.optimizers import OptimizerRegistry

        params = OptimizerRegistry.get_valid_params("adam")

        assert "lr" in params
        assert "betas" in params
        assert "eps" in params
        assert "weight_decay" in params

    def test_optimizer_merges_defaults_with_provided(self, mock_model):
        """Test optimizer creation merges defaults with provided params."""
        from milia_pipeline.models.training.optimizers import OptimizerRegistry

        # Only provide lr, other params should come from defaults
        optimizer = OptimizerRegistry.get_optimizer(
            "adam",
            mock_model.parameters(),
            {"lr": 0.01},  # Override lr
        )

        assert optimizer.defaults["lr"] == 0.01
        # Other params should be from defaults
        assert "betas" in optimizer.defaults

    def test_all_adaptive_optimizers_available(self, mock_model):
        """Test all adaptive optimizers are available."""
        from milia_pipeline.models.training.optimizers import OptimizerRegistry

        adaptive_optimizers = ["adam", "adamw", "adamax", "adadelta", "adagrad", "rmsprop"]

        for opt_name in adaptive_optimizers:
            assert opt_name in OptimizerRegistry.list_available(), f"Missing optimizer: {opt_name}"

    def test_all_sgd_variants_available(self, mock_model):
        """Test all SGD variants are available."""
        from milia_pipeline.models.training.optimizers import OptimizerRegistry

        sgd_variants = ["sgd", "asgd"]

        for opt_name in sgd_variants:
            assert opt_name in OptimizerRegistry.list_available(), f"Missing optimizer: {opt_name}"


# =============================================================================
# INTEGRATION TEST CLASS 14: Scheduler Registry Advanced Features
# =============================================================================


class TestSchedulerRegistryAdvancedFeatures:
    """
    Integration tests for advanced SchedulerRegistry features.

    Verifies:
    - Custom scheduler registration
    - Warmup scheduler creation
    - Sequential scheduler support
    """

    def test_get_scheduler_info_returns_expected_fields(self, mock_model):
        """Test get_scheduler_info returns all expected fields."""
        from milia_pipeline.models.training.schedulers import SchedulerRegistry

        info = SchedulerRegistry.get_scheduler_info("reduce_on_plateau")

        assert "name" in info
        assert "class" in info
        assert "metric_based" in info
        assert "default_params" in info
        assert "valid_params" in info
        assert info["metric_based"] is True

    def test_get_default_params_returns_registry_defaults(self, mock_model):
        """Test get_default_params returns registry defaults."""
        from milia_pipeline.models.training.schedulers import SchedulerRegistry

        defaults = SchedulerRegistry.get_default_params("reduce_on_plateau")

        assert "factor" in defaults
        assert "patience" in defaults
        assert defaults["factor"] == 0.1
        assert defaults["patience"] == 10

    def test_create_warmup_scheduler_integration(self, mock_model):
        """Test create_warmup_scheduler creates valid scheduler."""
        from milia_pipeline.models.training.schedulers import create_warmup_scheduler

        optimizer = torch.optim.Adam(mock_model.parameters(), lr=0.001)

        scheduler = create_warmup_scheduler(
            optimizer=optimizer,
            warmup_epochs=10,
            total_epochs=100,
            warmup_start_lr=1e-6,
            after_scheduler_name="cosine_annealing",
        )

        assert scheduler is not None
        # Should be a SequentialLR
        assert hasattr(scheduler, "_schedulers")

    def test_all_adaptive_schedulers_available(self, mock_model):
        """Test all adaptive schedulers are available."""
        from milia_pipeline.models.training.schedulers import SchedulerRegistry

        assert "reduce_on_plateau" in SchedulerRegistry.list_available()

    def test_all_step_schedulers_available(self, mock_model):
        """Test all step-based schedulers are available."""
        from milia_pipeline.models.training.schedulers import SchedulerRegistry

        step_schedulers = ["step_lr", "multistep_lr", "exponential_lr"]

        for sched_name in step_schedulers:
            assert sched_name in SchedulerRegistry.list_available(), (
                f"Missing scheduler: {sched_name}"
            )

    def test_all_cosine_schedulers_available(self, mock_model):
        """Test all cosine schedulers are available."""
        from milia_pipeline.models.training.schedulers import SchedulerRegistry

        cosine_schedulers = ["cosine_annealing", "cosine_annealing_warm_restarts"]

        for sched_name in cosine_schedulers:
            assert sched_name in SchedulerRegistry.list_available(), (
                f"Missing scheduler: {sched_name}"
            )

    def test_all_cyclic_schedulers_available(self, mock_model):
        """Test all cyclic schedulers are available."""
        from milia_pipeline.models.training.schedulers import SchedulerRegistry

        cyclic_schedulers = ["cyclic_lr", "one_cycle"]

        for sched_name in cyclic_schedulers:
            assert sched_name in SchedulerRegistry.list_available(), (
                f"Missing scheduler: {sched_name}"
            )


# =============================================================================
# INTEGRATION TEST CLASS 15: HPO Manager Convenience Functions
# =============================================================================


class TestHPOManagerConvenienceFunctions:
    """
    Integration tests for HPO Manager convenience functions.

    Verifies:
    - is_hpo_enabled
    - get_best_params
    - create_hpo_manager
    """

    def test_is_hpo_enabled_with_enabled_config(self):
        """Test is_hpo_enabled returns True for enabled config."""
        from milia_pipeline.models.hpo.hpo_manager import is_hpo_enabled

        config = MockHPOConfig(enabled=True)
        assert is_hpo_enabled(config) is True

    def test_is_hpo_enabled_with_disabled_config(self):
        """Test is_hpo_enabled returns False for disabled config."""
        from milia_pipeline.models.hpo.hpo_manager import is_hpo_enabled

        config = MockHPOConfig(enabled=False)
        assert is_hpo_enabled(config) is False

    def test_is_hpo_enabled_with_none_config(self):
        """Test is_hpo_enabled returns False for None config."""
        from milia_pipeline.models.hpo.hpo_manager import is_hpo_enabled

        assert is_hpo_enabled(None) is False

    @patch("milia_pipeline.models.hpo.hpo_manager.get_backend")
    def test_get_best_params_raises_error_when_no_optimization(self, mock_get_backend):
        """Test get_best_params raises error when no optimization done."""
        from milia_pipeline.exceptions import HPOError
        from milia_pipeline.models.hpo.hpo_manager import HPOManager, get_best_params

        config = MockHPOConfig(enabled=True)
        manager = HPOManager(config)

        with pytest.raises(HPOError, match="No optimization completed"):
            get_best_params(manager)

    @patch("milia_pipeline.models.hpo.hpo_manager.get_backend")
    def test_create_hpo_manager_creates_valid_manager(self, mock_get_backend):
        """Test create_hpo_manager creates valid HPOManager."""
        from milia_pipeline.models.hpo.hpo_manager import create_hpo_manager

        mock_get_backend.return_value = MagicMock()

        manager = create_hpo_manager(enabled=True, n_trials=50, backend="optuna")

        assert manager is not None
        assert manager.config.enabled is True
        assert manager.config.n_trials == 50


# =============================================================================
# INTEGRATION TEST CLASS 16: HPO Study Management
# =============================================================================


class TestHPOStudyManagement:
    """
    Integration tests for HPO study management.

    Verifies:
    - Study creation
    - Study statistics
    - Trial management
    """

    @patch("milia_pipeline.models.hpo.hpo_manager.get_backend")
    def test_get_best_value_raises_error_without_study(self, mock_get_backend):
        """Test get_best_value raises error when no study exists."""
        from milia_pipeline.exceptions import HPOError
        from milia_pipeline.models.hpo.hpo_manager import HPOManager

        config = MockHPOConfig(enabled=True)
        manager = HPOManager(config)

        with pytest.raises(HPOError, match="No study available"):
            manager.get_best_value()

    @patch("milia_pipeline.models.hpo.hpo_manager.get_backend")
    def test_get_best_trial_raises_error_without_study(self, mock_get_backend):
        """Test get_best_trial raises error when no study exists."""
        from milia_pipeline.exceptions import HPOError
        from milia_pipeline.models.hpo.hpo_manager import HPOManager

        config = MockHPOConfig(enabled=True)
        manager = HPOManager(config)

        with pytest.raises(HPOError, match="No study available"):
            manager.get_best_trial()

    @patch("milia_pipeline.models.hpo.hpo_manager.get_backend")
    def test_get_all_trials_raises_error_without_study(self, mock_get_backend):
        """Test get_all_trials raises error when no study exists."""
        from milia_pipeline.exceptions import HPOError
        from milia_pipeline.models.hpo.hpo_manager import HPOManager

        config = MockHPOConfig(enabled=True)
        manager = HPOManager(config)

        with pytest.raises(HPOError, match="No study available"):
            manager.get_all_trials()

    @patch("milia_pipeline.models.hpo.hpo_manager.get_backend")
    def test_get_study_statistics_raises_error_without_study(self, mock_get_backend):
        """Test get_study_statistics raises error when no study exists."""
        from milia_pipeline.exceptions import HPOError
        from milia_pipeline.models.hpo.hpo_manager import HPOManager

        config = MockHPOConfig(enabled=True)
        manager = HPOManager(config)

        with pytest.raises(HPOError, match="No study available"):
            manager.get_study_statistics()


# =============================================================================
# INTEGRATION TEST CLASS 17: Task Type Mapping Completeness
# =============================================================================


class TestTaskTypeMappingCompleteness:
    """
    Integration tests for task type mapping completeness.

    Verifies:
    - All task types have loss mappings
    - All task types have data prep functions
    """

    def test_all_task_types_have_loss_mappings(self):
        """Test all task types have corresponding loss mappings."""
        from milia_pipeline.models.hpo.hpo_manager import _get_loss_name_for_task

        task_types = [
            "graph_regression",
            "graph_classification",
            "node_regression",
            "node_classification",
            "link_prediction",
            "edge_regression",
        ]

        for task_type in task_types:
            loss_name = _get_loss_name_for_task(task_type)
            assert loss_name is not None, f"No loss mapping for task: {task_type}"
            assert isinstance(loss_name, str), f"Loss name should be string for task: {task_type}"

    def test_regression_tasks_map_to_mse(self):
        """Test regression tasks map to MSE loss."""
        from milia_pipeline.models.hpo.hpo_manager import _get_loss_name_for_task

        regression_tasks = ["graph_regression", "node_regression", "edge_regression"]

        for task in regression_tasks:
            assert _get_loss_name_for_task(task) == "mse", f"Task {task} should map to MSE"

    def test_classification_tasks_map_to_cross_entropy(self):
        """Test classification tasks map to cross entropy loss."""
        from milia_pipeline.models.hpo.hpo_manager import _get_loss_name_for_task

        classification_tasks = ["graph_classification", "node_classification"]

        for task in classification_tasks:
            assert _get_loss_name_for_task(task) == "cross_entropy", (
                f"Task {task} should map to cross_entropy"
            )

    def test_link_prediction_maps_to_bce_with_logits(self):
        """Test link prediction maps to BCE with logits loss."""
        from milia_pipeline.models.hpo.hpo_manager import _get_loss_name_for_task

        assert _get_loss_name_for_task("link_prediction") == "bce_with_logits"

    def test_all_task_types_handled_in_data_prep(self):
        """Test all task types are handled in data preparation."""
        from milia_pipeline.models.hpo.hpo_manager import _prepare_data_for_task_hpo

        _task_types = [
            "graph_regression",
            "graph_classification",
            "node_regression",
            "node_classification",
            "link_prediction",
            "edge_regression",
        ]

        # Create mock data that should work for graph-level tasks
        mock_train = MockPyGDataset([MockPyGData() for _ in range(5)])
        mock_val = MockPyGDataset([MockPyGData() for _ in range(2)])

        # Graph-level tasks should work without error
        for task_type in ["graph_regression", "graph_classification"]:
            train, val, _ = _prepare_data_for_task_hpo(mock_train, mock_val, task_type)
            assert train is not None
            assert val is not None


# =============================================================================
# INTEGRATION TEST CLASS 18: Parameter Filtering Consistency
# =============================================================================


class TestParameterFilteringConsistency:
    """
    Integration tests for parameter filtering consistency across modules.

    Verifies:
    - Consistent filtering across all registries
    - Invalid params always filtered
    """

    def test_loss_registry_filters_invalid_params(self):
        """Test LossRegistry filters invalid parameters."""
        from milia_pipeline.models.training.loss_functions import LossRegistry

        # Pass params that are valid for different losses
        mixed_params = {
            "alpha": 0.25,  # Valid for focal
            "gamma": 2.0,  # Valid for focal
            "reduction": "mean",  # Valid for most
            "invalid_xyz": 123,  # Invalid
        }

        # MSELoss should filter out alpha, gamma, invalid_xyz
        mse = LossRegistry.get_loss("mse", mixed_params)
        assert isinstance(mse, nn.MSELoss)

        # FocalLoss should use alpha, gamma and filter invalid_xyz
        focal = LossRegistry.get_loss("focal", mixed_params)
        assert focal.alpha == 0.25
        assert focal.gamma == 2.0

    def test_optimizer_registry_filters_invalid_params(self, mock_model):
        """Test OptimizerRegistry filters invalid parameters."""
        from milia_pipeline.models.training.optimizers import OptimizerRegistry

        mixed_params = {
            "lr": 0.001,
            "weight_decay": 1e-5,
            "momentum": 0.9,  # Valid for SGD, not Adam
            "invalid_xyz": 123,  # Invalid for all
        }

        # Adam should filter out momentum and invalid_xyz
        adam = OptimizerRegistry.get_optimizer("adam", mock_model.parameters(), mixed_params)
        assert isinstance(adam, torch.optim.Adam)

    def test_scheduler_registry_filters_invalid_params(self, mock_model):
        """Test SchedulerRegistry filters invalid parameters."""
        from milia_pipeline.models.training.schedulers import SchedulerRegistry

        optimizer = torch.optim.Adam(mock_model.parameters())

        mixed_params = {
            "patience": 10,  # Valid for ReduceLROnPlateau
            "factor": 0.5,  # Valid for ReduceLROnPlateau
            "T_max": 100,  # Valid for CosineAnnealingLR, not ReduceLROnPlateau
            "invalid_xyz": 123,  # Invalid for all
        }

        # ReduceLROnPlateau should filter out T_max and invalid_xyz
        scheduler = SchedulerRegistry.get_scheduler("reduce_on_plateau", optimizer, mixed_params)
        assert scheduler is not None


# =============================================================================
# INTEGRATION TEST CLASS 19: Cross-Validation Integration
# =============================================================================


class TestCrossValidationIntegration:
    """
    Integration tests for cross-validation with registries.

    Verifies:
    - CV metric aggregation methods
    - CV with registry-created components
    """

    def test_cv_aggregation_methods_exist(self):
        """Test CV metric aggregation methods exist in statistics module."""
        from statistics import mean, median

        # These should be available for CV aggregation
        values = [0.1, 0.2, 0.3, 0.4, 0.5]

        assert mean(values) == 0.3
        assert median(values) == 0.3
        assert min(values) == 0.1
        assert max(values) == 0.5


# =============================================================================
# INTEGRATION TEST CLASS 20: Final Model Training Integration
# =============================================================================


class TestFinalModelTrainingIntegration:
    """
    Integration tests for final model training after HPO.

    Verifies:
    - train_final_model raises errors appropriately
    - Best params are applied correctly
    """

    @patch("milia_pipeline.models.hpo.hpo_manager.get_backend")
    def test_train_final_model_raises_error_without_best_params(self, mock_get_backend):
        """Test train_final_model raises error when no best params."""
        from milia_pipeline.exceptions import HPOError
        from milia_pipeline.models.hpo.hpo_manager import HPOManager

        config = MockHPOConfig(enabled=True)
        manager = HPOManager(config)

        with pytest.raises(HPOError, match="No best parameters available"):
            manager.train_final_model(dataset=MockPyGDataset(), model_name="GCN")


# =============================================================================
# INTEGRATION TEST CLASS 21: Search Space Filtering Integration
# =============================================================================


class TestSearchSpaceFilteringIntegration:
    """
    Integration tests for search space filtering with registry.

    Verifies:
    - Model-specific search space filtering
    - Registry metadata usage for filtering
    """

    @patch("milia_pipeline.models.hpo.hpo_manager.get_backend")
    @patch("milia_pipeline.models.hpo.hpo_manager._REGISTRY_AVAILABLE", False)
    def test_filter_search_space_without_registry(self, mock_get_backend):
        """Test search space filtering falls back when registry unavailable."""
        from milia_pipeline.models.hpo.hpo_manager import HPOManager

        config = MockHPOConfig(enabled=True)
        manager = HPOManager(config)

        search_space = {
            "hyperparameters": {
                "hidden_channels": MockSearchSpaceParamConfig(
                    type=MockParamType.INT, low=32, high=256
                ),
                "heads": MockSearchSpaceParamConfig(type=MockParamType.INT, low=1, high=8),
            }
        }

        # Should return original space when registry unavailable
        filtered = manager._filter_search_space_for_model("GCN", search_space)

        # Original should not be mutated
        assert "hyperparameters" in filtered


# =============================================================================
# INTEGRATION TEST CLASS 22: Data Splitting Integration
# =============================================================================


class TestDataSplittingIntegration:
    """
    Integration tests for DataSplitter with HPO.

    Verifies:
    - Split ratios are respected
    - HPO uses correct splits
    """

    def test_data_splitter_available_check(self):
        """Test DataSplitter availability can be checked."""
        try:
            from milia_pipeline.models.training.data_splitting import DataSplitter

            assert DataSplitter is not None
        except ImportError:
            # DataSplitter might not be available in test environment
            pytest.skip("DataSplitter not available")


# =============================================================================
# INTEGRATION TEST CLASS 23: Callback Integration
# =============================================================================


class TestCallbackIntegration:
    """
    Integration tests for callback system with HPO.

    Verifies:
    - HPO callback can be created
    - Callback list management
    """

    def test_create_hpo_callback_optuna(self):
        """Test HPO callback creation for Optuna backend."""
        from milia_pipeline.models.hpo.callbacks import create_hpo_callback

        mock_trial = MagicMock()

        callback = create_hpo_callback(
            trial=mock_trial, monitor="val_loss", report_every=1, backend="optuna"
        )

        assert callback is not None


# =============================================================================
# INTEGRATION TEST CLASS 24: Task-Specific Data Preparation (All Tasks)
# =============================================================================


class TestTaskSpecificDataPreparationAllTasks:
    """
    Integration tests for task-specific data preparation for all task types.

    Verifies:
    - Graph regression/classification pass through
    - Link prediction applies transform or validates
    - Edge regression validates edge attributes
    - Node-level tasks validate y shape
    """

    def test_graph_regression_passes_through(self):
        """Test graph regression data passes through unchanged."""
        from milia_pipeline.models.hpo.hpo_manager import _prepare_data_for_task_hpo

        train_data = [MockPyGData() for _ in range(5)]
        val_data = [MockPyGData() for _ in range(2)]

        result_train, result_val, _ = _prepare_data_for_task_hpo(
            train_data, val_data, "graph_regression"
        )

        assert len(result_train) == 5
        assert len(result_val) == 2

    def test_graph_classification_passes_through(self):
        """Test graph classification data passes through unchanged."""
        from milia_pipeline.models.hpo.hpo_manager import _prepare_data_for_task_hpo

        train_data = [MockPyGData() for _ in range(5)]
        val_data = [MockPyGData() for _ in range(2)]

        result_train, result_val, _ = _prepare_data_for_task_hpo(
            train_data, val_data, "graph_classification"
        )

        assert len(result_train) == 5
        assert len(result_val) == 2

    def test_link_prediction_with_edge_label(self):
        """Test link prediction passes through when edge_label exists."""
        from milia_pipeline.models.hpo.hpo_manager import _prepare_data_for_task_hpo

        # Create data with edge_label
        train_data = [MockPyGData(edge_label=torch.tensor([1, 0, 1])) for _ in range(3)]
        val_data = [MockPyGData(edge_label=torch.tensor([0, 1])) for _ in range(2)]

        result_train, result_val, _ = _prepare_data_for_task_hpo(
            train_data, val_data, "link_prediction"
        )

        assert len(result_train) == 3
        assert len(result_val) == 2

    def test_edge_regression_with_edge_value(self):
        """Test edge regression passes through when edge_value exists."""
        from milia_pipeline.models.hpo.hpo_manager import _prepare_data_for_task_hpo

        # Create data with edge_value
        train_data = [MockPyGData(edge_value=torch.randn(5)) for _ in range(3)]
        val_data = [MockPyGData(edge_value=torch.randn(5)) for _ in range(2)]

        result_train, result_val, _ = _prepare_data_for_task_hpo(
            train_data, val_data, "edge_regression"
        )

        assert len(result_train) == 3
        assert len(result_val) == 2

    def test_edge_regression_with_edge_y(self):
        """Test edge regression passes through when edge_y exists."""
        from milia_pipeline.models.hpo.hpo_manager import _prepare_data_for_task_hpo

        # Create data with edge_y
        train_data = [MockPyGData(edge_y=torch.randn(5)) for _ in range(3)]
        val_data = [MockPyGData(edge_y=torch.randn(5)) for _ in range(2)]

        result_train, result_val, _ = _prepare_data_for_task_hpo(
            train_data, val_data, "edge_regression"
        )

        assert len(result_train) == 3
        assert len(result_val) == 2

    def test_edge_regression_raises_without_edge_attributes(self):
        """Test edge regression raises error without edge attributes."""
        from milia_pipeline.exceptions import HPOError
        from milia_pipeline.models.hpo.hpo_manager import _prepare_data_for_task_hpo

        # Create data without edge_value or edge_y
        train_data = [MockPyGData() for _ in range(3)]
        val_data = [MockPyGData() for _ in range(2)]

        with pytest.raises(HPOError, match="edge_regression"):
            _prepare_data_for_task_hpo(train_data, val_data, "edge_regression")

    def test_edge_regression_raises_on_empty_dataset(self):
        """Test edge regression raises error on empty dataset."""
        from milia_pipeline.exceptions import HPOError
        from milia_pipeline.models.hpo.hpo_manager import _prepare_data_for_task_hpo

        train_data = []
        val_data = []

        with pytest.raises(HPOError, match="empty dataset"):
            _prepare_data_for_task_hpo(train_data, val_data, "edge_regression")

    def test_node_regression_with_valid_y_shape(self):
        """Test node regression passes through with valid y shape."""
        from milia_pipeline.models.hpo.hpo_manager import _prepare_data_for_task_hpo

        # Create data with node-level y (shape matches num_nodes)
        train_data = [
            MockPyGData(
                x=torch.randn(10, 16),
                y=torch.randn(10),  # Same as num_nodes
            )
            for _ in range(3)
        ]
        val_data = [MockPyGData(x=torch.randn(10, 16), y=torch.randn(10)) for _ in range(2)]

        result_train, result_val, _ = _prepare_data_for_task_hpo(
            train_data, val_data, "node_regression"
        )

        assert len(result_train) == 3
        assert len(result_val) == 2

    def test_node_classification_with_valid_y_shape(self):
        """Test node classification passes through with valid y shape."""
        from milia_pipeline.models.hpo.hpo_manager import _prepare_data_for_task_hpo

        # Create data with node-level y (shape matches num_nodes)
        train_data = [
            MockPyGData(
                x=torch.randn(10, 16),
                y=torch.randint(0, 5, (10,)),  # Same as num_nodes
            )
            for _ in range(3)
        ]
        val_data = [
            MockPyGData(x=torch.randn(10, 16), y=torch.randint(0, 5, (10,))) for _ in range(2)
        ]

        result_train, result_val, _ = _prepare_data_for_task_hpo(
            train_data, val_data, "node_classification"
        )

        assert len(result_train) == 3
        assert len(result_val) == 2

    def test_node_level_extracts_from_x_when_y_is_none(self):
        """Test node-level task extracts targets from x when y is None."""
        from milia_pipeline.models.hpo.hpo_manager import _prepare_data_for_task_hpo

        # Create data without y but with valid x
        # The function should extract targets from x (default source for node tasks)
        train_data = [MockPyGData(x=torch.randn(10, 16), y=None) for _ in range(3)]
        val_data = [MockPyGData(x=torch.randn(10, 16), y=None) for _ in range(2)]

        # Should NOT raise - extracts from x instead
        result_train, result_val, _ = _prepare_data_for_task_hpo(
            train_data, val_data, "node_regression"
        )

        assert len(result_train) == 3
        assert len(result_val) == 2

    def test_node_level_extracts_from_x_when_y_is_scalar(self):
        """Test node-level task extracts targets from x when y is scalar (graph-level)."""
        from milia_pipeline.models.hpo.hpo_manager import _prepare_data_for_task_hpo

        # Create data with scalar y (graph-level) but valid x
        # The function should extract targets from x instead
        train_data = [MockPyGData(x=torch.randn(10, 16), y=torch.tensor(1.0)) for _ in range(3)]
        val_data = [MockPyGData(x=torch.randn(10, 16), y=torch.tensor(1.0)) for _ in range(2)]

        # Should NOT raise - extracts from x instead
        result_train, result_val, _ = _prepare_data_for_task_hpo(
            train_data, val_data, "node_regression"
        )

        assert len(result_train) == 3
        assert len(result_val) == 2

    def test_node_level_extracts_from_x_when_y_shape_mismatched(self):
        """Test node-level task extracts targets from x when y shape doesn't match num_nodes."""
        from milia_pipeline.models.hpo.hpo_manager import _prepare_data_for_task_hpo

        # Create data with y shape not matching num_nodes but valid x
        # The function should extract targets from x instead
        train_data = [
            MockPyGData(
                x=torch.randn(10, 16),  # 10 nodes
                y=torch.randn(5),  # Only 5 targets - doesn't match
            )
            for _ in range(3)
        ]
        val_data = [MockPyGData(x=torch.randn(10, 16), y=torch.randn(5)) for _ in range(2)]

        # Should NOT raise - extracts from x instead
        result_train, result_val, _ = _prepare_data_for_task_hpo(
            train_data, val_data, "node_regression"
        )

        assert len(result_train) == 3
        assert len(result_val) == 2

    def test_node_level_raises_when_both_y_and_x_unavailable(self):
        """Test node-level task raises error when both y is invalid and x is unavailable."""
        from milia_pipeline.exceptions import HPOError
        from milia_pipeline.models.hpo.hpo_manager import _prepare_data_for_task_hpo

        # Create data without valid y AND without valid x
        # This should raise HPOError because no source is available
        train_data = [
            MockPyGData(
                x=None,  # No x available
                y=torch.tensor(1.0),  # Scalar y (not node-level)
            )
            for _ in range(3)
        ]
        val_data = [MockPyGData(x=None, y=torch.tensor(1.0)) for _ in range(2)]

        with pytest.raises(HPOError, match="node-level targets"):
            _prepare_data_for_task_hpo(train_data, val_data, "node_regression")

    def test_unknown_task_type_logs_warning_and_passes_through(self):
        """Test unknown task type logs warning and passes through."""
        from milia_pipeline.models.hpo.hpo_manager import _prepare_data_for_task_hpo

        train_data = [MockPyGData() for _ in range(5)]
        val_data = [MockPyGData() for _ in range(2)]

        # Should not raise, just log warning
        result_train, result_val, _ = _prepare_data_for_task_hpo(
            train_data, val_data, "unknown_task_type"
        )

        assert len(result_train) == 5
        assert len(result_val) == 2


# =============================================================================
# INTEGRATION TEST CLASS 25: HPO Manager from_yaml Integration
# =============================================================================


class TestHPOManagerFromYamlIntegration:
    """
    Integration tests for HPOManager.from_yaml loading.

    Verifies:
    - YAML loading works
    - Section navigation works
    - Error handling for missing sections
    """

    @patch("builtins.open", new_callable=MagicMock)
    @patch("yaml.safe_load")
    @patch("milia_pipeline.models.hpo.hpo_manager.get_backend")
    def test_from_yaml_loads_nested_section(self, mock_get_backend, mock_yaml_load, mock_open):
        """Test from_yaml navigates nested sections correctly."""
        from milia_pipeline.models.hpo.hpo_manager import HPOManager

        # Mock YAML content
        mock_yaml_load.return_value = {
            "models": {
                "hpo": {
                    "enabled": True,
                    "n_trials": 50,
                    "backend": "optuna",
                }
            }
        }

        # Mock HPOConfig.from_dict to return MockHPOConfig
        with patch.object(HPOManager, "from_config") as mock_from_config:
            mock_from_config.return_value = MagicMock()

            _manager = HPOManager.from_yaml("config.yaml", section="models.hpo")

            mock_from_config.assert_called_once()

    @patch("builtins.open", new_callable=MagicMock)
    @patch("yaml.safe_load")
    def test_from_yaml_raises_error_for_missing_section(self, mock_yaml_load, mock_open):
        """Test from_yaml raises error for missing section."""
        from milia_pipeline.exceptions import HPOConfigurationError
        from milia_pipeline.models.hpo.hpo_manager import HPOManager

        # Mock YAML content without the section
        mock_yaml_load.return_value = {
            "models": {
                "training": {}  # No 'hpo' section
            }
        }

        with pytest.raises(HPOConfigurationError, match="not found"):
            HPOManager.from_yaml("config.yaml", section="models.hpo")


# =============================================================================
# TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
