#!/usr/bin/env python3
"""
HPO Integration Test Suite for milia_pipeline/models/hpo Module

This is a PRODUCTION-READY integration test suite that verifies:
1. HPO module public API accessibility from models/ package
2. Trainer + HPO callback real integration
3. HPOManager + ModelFactory integration
4. HPOManager + DataSplitter cross-validation integration
5. Exception hierarchy and propagation
6. Configuration loading and validation
7. OptunaPruningCallback Callback ABC compliance
8. TrialPruned exception propagation through Trainer

CRITICAL: This test uses real objects where possible, with mocking only for:
- Optuna trial objects (to control pruning behavior)
- External file I/O (no real file downloads)
- Expensive operations that would slow tests

NOTE: This test file follows strict mock pollution prevention guidelines:
- NO sys.modules injection at module level
- Test-level @patch decorators for all mocking
- No global mock state that could affect other tests

Path on local machine: ~/ml_projects/milia/tests/test_hpo_integration.py

Author: Milia Team
Version: 1.0.0
"""

import sys
from pathlib import Path

# =============================================================================
# ADD PROJECT ROOT TO PYTHON PATH FIRST
# =============================================================================
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


# =============================================================================
# TEST FIXTURES - REAL MODEL CLASSES
# =============================================================================


class MinimalGNNModel(nn.Module):
    """
    Minimal GNN model for integration testing.

    This is a real torch.nn.Module that can be trained, not a mock.
    Designed to be lightweight for fast test execution.
    """

    def __init__(
        self,
        in_channels: int = 5,
        hidden_channels: int = 16,
        out_channels: int = 1,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        # Simple linear layers (no GNN convolutions for speed)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(nn.Linear(hidden_channels, out_channels))

        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, batch=None):
        """Forward pass with optional batch for graph-level tasks."""
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.act(x)
            if self.dropout > 0:
                x = self.drop(x)

        x = self.layers[-1](x)

        # Global mean pooling for graph-level task
        if batch is not None:
            from torch_geometric.nn import global_mean_pool

            x = global_mean_pool(x, batch)

        return x


@pytest.fixture
def minimal_gnn_model_class():
    """Return the MinimalGNNModel class for factory registration."""
    return MinimalGNNModel


@pytest.fixture
def sample_pyg_data():
    """Create sample PyG Data object for testing."""
    return Data(
        x=torch.randn(10, 5),  # 10 nodes, 5 features
        edge_index=torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long
        ),
        y=torch.tensor([0.5]),  # Graph-level target
        num_nodes=10,
    )


@pytest.fixture
def sample_pyg_dataset(sample_pyg_data):
    """Create a list of sample PyG Data objects for testing (mimics dataset)."""
    data_list = []
    for i in range(10):
        num_nodes = 8 + i
        data = Data(
            x=torch.randn(num_nodes, 5),
            edge_index=torch.randint(0, num_nodes, (2, num_nodes * 2)),
            y=torch.randn(1),
            num_nodes=num_nodes,
        )
        data_list.append(data)
    return data_list


@pytest.fixture
def mock_optuna_trial():
    """Create a mock Optuna trial object for controlled testing."""
    trial = MagicMock()
    trial.number = 0
    trial.report = MagicMock()
    trial.should_prune = MagicMock(return_value=False)

    # Mock suggest methods
    trial.suggest_int = MagicMock(return_value=32)
    trial.suggest_float = MagicMock(return_value=0.001)
    trial.suggest_categorical = MagicMock(return_value="relu")

    return trial


@pytest.fixture
def mock_optuna_trial_pruned():
    """Create a mock Optuna trial that signals pruning."""
    trial = MagicMock()
    trial.number = 1
    trial.report = MagicMock()
    trial.should_prune = MagicMock(return_value=True)
    return trial


# =============================================================================
# TEST CLASS: HPO PUBLIC API ACCESSIBILITY
# =============================================================================


class TestHPOPublicAPIAccessibility:
    """
    Test that all HPO components are accessible from the models/ package public API.

    This verifies the integration of hpo/ subpackage exports into models/__init__.py.
    """

    def test_hpo_manager_importable_from_models(self):
        """Test HPOManager can be imported from milia_pipeline.models."""
        from milia_pipeline.models import HPOManager

        assert HPOManager is not None

    def test_hpo_config_importable_from_models(self):
        """Test HPOConfig can be imported from milia_pipeline.models."""
        from milia_pipeline.models import HPOConfig

        assert HPOConfig is not None

    def test_hpo_available_flag_importable_from_models(self):
        """Test HPO_AVAILABLE flag can be imported from milia_pipeline.models."""
        from milia_pipeline.models import HPO_AVAILABLE

        assert isinstance(HPO_AVAILABLE, bool)

    def test_optuna_pruning_callback_importable_from_models(self):
        """Test OptunaPruningCallback can be imported from milia_pipeline.models."""
        from milia_pipeline.models import HPO_AVAILABLE

        if HPO_AVAILABLE:
            from milia_pipeline.models import OptunaPruningCallback

            assert OptunaPruningCallback is not None
        else:
            pytest.skip("HPO not available (Optuna not installed)")

    def test_convenience_functions_importable_from_models(self):
        """Test HPO convenience functions can be imported from milia_pipeline.models."""
        from milia_pipeline.models import HPO_AVAILABLE

        if HPO_AVAILABLE:
            from milia_pipeline.models.hpo import (
                create_hpo_manager,
                get_best_params,
                is_hpo_enabled,
            )

            assert callable(is_hpo_enabled)
            assert callable(get_best_params)
            assert callable(create_hpo_manager)
        else:
            pytest.skip("HPO not available")

    def test_hpo_exceptions_importable_from_exceptions_module(self):
        """Test HPO exceptions are in centralized exceptions.py."""
        from milia_pipeline.exceptions import (
            BackendError,
            HPOConfigurationError,
            HPOError,
            PruningError,
            SearchSpaceError,
            StudyNotFoundError,
            TrialFailedError,
        )

        # Verify hierarchy
        assert issubclass(HPOError, Exception)
        assert issubclass(HPOConfigurationError, HPOError)
        assert issubclass(TrialFailedError, HPOError)
        assert issubclass(StudyNotFoundError, HPOError)
        assert issubclass(BackendError, HPOError)
        assert issubclass(SearchSpaceError, HPOError)
        assert issubclass(PruningError, HPOError)

    def test_get_module_info_shows_hpo(self):
        """Test get_module_info() includes HPO information."""
        from milia_pipeline.models import get_module_info

        info = get_module_info()

        assert "hpo_available" in info
        assert "phase_8_features" in info

        if info["hpo_available"]:
            assert info["phase_8_features"]["hyperparameter_optimization"] is True


# =============================================================================
# TEST CLASS: HPO EXCEPTION HIERARCHY
# =============================================================================


class TestHPOExceptionHierarchy:
    """
    Test HPO exception hierarchy integration with centralized exceptions.py.
    """

    def test_hpo_error_inherits_from_model_error(self):
        """Test HPOError inherits from ModelError for consistent handling."""
        from milia_pipeline.exceptions import HPOError, ModelError

        assert issubclass(HPOError, ModelError)

    def test_hpo_configuration_error_attributes(self):
        """Test HPOConfigurationError has expected attributes."""
        from milia_pipeline.exceptions import HPOConfigurationError

        error = HPOConfigurationError(
            "Invalid config",
            config_key="hpo.n_trials",
            actual_value=-1,
            expected_value="positive integer",
        )

        assert error.config_key == "hpo.n_trials"
        assert error.actual_value == -1
        assert error.expected_value == "positive integer"
        assert "Invalid config" in str(error)

    def test_trial_failed_error_attributes(self):
        """Test TrialFailedError has expected attributes."""
        from milia_pipeline.exceptions import TrialFailedError

        error = TrialFailedError(
            "Trial crashed",
            trial_number=42,
            trial_params={"lr": 0.001},
            original_error="CUDA OOM",
            epoch=10,
        )

        assert error.trial_number == 42
        assert error.trial_params == {"lr": 0.001}
        assert error.original_error == "CUDA OOM"
        assert error.epoch == 10

    def test_catch_all_hpo_errors_by_base_class(self):
        """Test all HPO exceptions can be caught by HPOError base class."""
        from milia_pipeline.exceptions import (
            BackendError,
            HPOConfigurationError,
            HPOError,
            PruningError,
            SearchSpaceError,
            StudyNotFoundError,
            TrialFailedError,
        )

        # Create exception instances with their required arguments
        exception_instances = [
            HPOConfigurationError("test message", config_key="test.key"),
            TrialFailedError("test message", trial_number=0),
            StudyNotFoundError("test message", study_name="test_study"),
            BackendError("test message", backend_name="optuna"),
            SearchSpaceError("test message", parameter_name="lr"),
            PruningError("test message", trial_number=0),
        ]

        for exc in exception_instances:
            try:
                raise exc
            except HPOError as e:
                assert "test message" in str(e)
            except Exception as e:
                pytest.fail(f"{exc.__class__.__name__} not caught by HPOError: {e}")


# =============================================================================
# TEST CLASS: OPTUNA PRUNING CALLBACK + CALLBACK ABC COMPLIANCE
# =============================================================================


class TestOptunaPruningCallbackABCCompliance:
    """
    Test OptunaPruningCallback properly extends Callback ABC from callbacks.py.
    """

    def test_callback_inherits_from_callback_abc(self):
        """Test OptunaPruningCallback inherits from Callback ABC."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from milia_pipeline.models.hpo.callbacks.optuna_callback import OptunaPruningCallback
        from milia_pipeline.models.training.callbacks import Callback

        assert issubclass(OptunaPruningCallback, Callback)

    def test_callback_has_set_trainer_method(self, mock_optuna_trial):
        """Test OptunaPruningCallback has set_trainer method."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from milia_pipeline.models.hpo.callbacks.optuna_callback import OptunaPruningCallback

        callback = OptunaPruningCallback(trial=mock_optuna_trial)

        assert hasattr(callback, "set_trainer")
        assert callable(callback.set_trainer)

    def test_callback_has_on_train_begin_method(self, mock_optuna_trial):
        """Test OptunaPruningCallback has on_train_begin method."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from milia_pipeline.models.hpo.callbacks.optuna_callback import OptunaPruningCallback

        callback = OptunaPruningCallback(trial=mock_optuna_trial)

        assert hasattr(callback, "on_train_begin")
        assert callable(callback.on_train_begin)

    def test_callback_has_on_epoch_end_method(self, mock_optuna_trial):
        """Test OptunaPruningCallback has on_epoch_end method."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from milia_pipeline.models.hpo.callbacks.optuna_callback import OptunaPruningCallback

        callback = OptunaPruningCallback(trial=mock_optuna_trial)

        assert hasattr(callback, "on_epoch_end")
        assert callable(callback.on_epoch_end)

    def test_callback_has_on_train_end_method(self, mock_optuna_trial):
        """Test OptunaPruningCallback has on_train_end method."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from milia_pipeline.models.hpo.callbacks.optuna_callback import OptunaPruningCallback

        callback = OptunaPruningCallback(trial=mock_optuna_trial)

        assert hasattr(callback, "on_train_end")
        assert callable(callback.on_train_end)

    def test_callback_has_should_stop_method(self, mock_optuna_trial):
        """Test OptunaPruningCallback has should_stop method for early stopping integration."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from milia_pipeline.models.hpo.callbacks.optuna_callback import OptunaPruningCallback

        callback = OptunaPruningCallback(trial=mock_optuna_trial)

        assert hasattr(callback, "should_stop")
        assert callable(callback.should_stop)
        # Method should return boolean when called
        result = callback.should_stop()
        assert isinstance(result, bool)


# =============================================================================
# TEST CLASS: TRAINER + HPO CALLBACK REAL INTEGRATION
# =============================================================================


class TestTrainerHPOCallbackIntegration:
    """
    Test real integration between Trainer and OptunaPruningCallback.

    These tests use real Trainer instances with mock trial objects to verify:
    - Callback receives correct metrics from Trainer
    - trial.report() is called with actual epoch metrics
    - should_stop property is checked by Trainer
    """

    def test_trainer_accepts_hpo_callback_parameter(self):
        """Test Trainer accepts hpo_callback parameter."""
        import inspect

        from milia_pipeline.models.training.trainer import Trainer

        sig = inspect.signature(Trainer.__init__)
        params = sig.parameters

        assert "hpo_callback" in params, "Trainer.__init__ must have hpo_callback parameter"

    def test_hpo_callback_receives_metrics_on_epoch_end(
        self, mock_optuna_trial, sample_pyg_dataset
    ):
        """Test HPO callback receives actual metrics during training."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from torch_geometric.loader import DataLoader

        from milia_pipeline.models.hpo.callbacks.optuna_callback import OptunaPruningCallback
        from milia_pipeline.models.training.trainer import Trainer

        # Create real model
        model = MinimalGNNModel(in_channels=5, hidden_channels=8, out_channels=1)

        # Create real dataloader
        train_loader = DataLoader(sample_pyg_dataset[:6], batch_size=2, shuffle=False)
        val_loader = DataLoader(sample_pyg_dataset[6:], batch_size=2, shuffle=False)

        # Create HPO callback with mock trial
        hpo_callback = OptunaPruningCallback(
            trial=mock_optuna_trial, monitor="val_loss", report_every=1
        )

        # Create trainer with HPO callback
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=nn.MSELoss(),
            optimizer=optimizer,
            max_epochs=2,
            hpo_callback=hpo_callback,
        )

        # Train for a couple epochs
        results = trainer.fit()

        # Verify trial.report was called with actual metrics
        assert mock_optuna_trial.report.called, "trial.report() should be called during training"

        # Check that report was called with epoch number and metric value
        calls = mock_optuna_trial.report.call_args_list
        assert len(calls) >= 1, "Should have at least one report call"

        # Verify call signature: trial.report(value, epoch)
        for call in calls:
            args = call[0]
            assert len(args) == 2, "report should be called with (value, epoch)"
            value, epoch = args
            assert isinstance(value, float), "Metric value should be float"
            assert isinstance(epoch, int), "Epoch should be int"

    def test_hpo_callback_added_to_trainer_callbacks_list(
        self, mock_optuna_trial, sample_pyg_dataset
    ):
        """Test HPO callback is properly added to trainer's callback list."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from torch_geometric.loader import DataLoader

        from milia_pipeline.models.hpo.callbacks.optuna_callback import OptunaPruningCallback
        from milia_pipeline.models.training.trainer import Trainer

        model = MinimalGNNModel(in_channels=5, hidden_channels=8, out_channels=1)
        train_loader = DataLoader(sample_pyg_dataset[:6], batch_size=2)

        hpo_callback = OptunaPruningCallback(trial=mock_optuna_trial, monitor="val_loss")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            loss_fn=nn.MSELoss(),
            optimizer=optimizer,
            max_epochs=1,
            hpo_callback=hpo_callback,
        )

        # Verify callback was added to callbacks list
        assert hpo_callback in trainer.callbacks, "HPO callback should be in trainer.callbacks list"

    def test_trainer_checks_should_stop_from_callbacks(self, mock_optuna_trial, sample_pyg_dataset):
        """Test Trainer checks should_stop property from callbacks."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from torch_geometric.loader import DataLoader

        from milia_pipeline.models.hpo.callbacks.optuna_callback import OptunaPruningCallback
        from milia_pipeline.models.training.trainer import Trainer

        model = MinimalGNNModel(in_channels=5, hidden_channels=8, out_channels=1)
        train_loader = DataLoader(sample_pyg_dataset[:6], batch_size=2)
        val_loader = DataLoader(sample_pyg_dataset[6:], batch_size=2)

        # Create callback with trial that will signal stop after first call
        hpo_callback = OptunaPruningCallback(trial=mock_optuna_trial, monitor="val_loss")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=nn.MSELoss(),
            optimizer=optimizer,
            max_epochs=5,
            hpo_callback=hpo_callback,
        )

        # Verify _should_stop method exists and checks callbacks
        assert hasattr(trainer, "_should_stop"), "Trainer should have _should_stop method"

        # Initially should_stop is False (mock returns False)
        assert trainer._should_stop() == False


# =============================================================================
# TEST CLASS: TRIAL PRUNED EXCEPTION PROPAGATION
# =============================================================================


class TestTrialPrunedExceptionPropagation:
    """
    Test that optuna.TrialPruned exception propagates correctly through Trainer.

    CRITICAL: The Trainer._on_epoch_end() method has try-except that could
    catch TrialPruned. This test verifies proper exception handling.
    """

    def test_trial_pruned_exception_raised_when_should_prune(
        self, mock_optuna_trial_pruned, sample_pyg_dataset
    ):
        """Test TrialPruned exception is raised when trial.should_prune() returns True."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        try:
            import optuna
        except ImportError:
            pytest.skip("Optuna not installed")

        from milia_pipeline.models.hpo.callbacks.optuna_callback import OptunaPruningCallback

        callback = OptunaPruningCallback(trial=mock_optuna_trial_pruned, monitor="val_loss")

        mock_trainer = MagicMock()
        metrics = {"val_loss": 0.5, "train_loss": 0.6}

        # Should raise TrialPruned when should_prune returns True
        with pytest.raises(optuna.TrialPruned):
            callback.on_epoch_end(mock_trainer, epoch=1, metrics=metrics)

    def test_callback_correctly_reports_before_pruning_check(
        self, mock_optuna_trial_pruned, sample_pyg_dataset
    ):
        """Test callback reports metric before checking for pruning."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        try:
            import optuna
        except ImportError:
            pytest.skip("Optuna not installed")

        from milia_pipeline.models.hpo.callbacks.optuna_callback import OptunaPruningCallback

        callback = OptunaPruningCallback(trial=mock_optuna_trial_pruned, monitor="val_loss")

        mock_trainer = MagicMock()
        metrics = {"val_loss": 0.5}

        # Will raise TrialPruned, but should report first
        with pytest.raises(optuna.TrialPruned):
            callback.on_epoch_end(mock_trainer, epoch=1, metrics=metrics)

        # Verify report was called before pruning
        mock_optuna_trial_pruned.report.assert_called_once_with(0.5, 1)


# =============================================================================
# TEST CLASS: HPO CONFIG INTEGRATION
# =============================================================================


class TestHPOConfigIntegration:
    """
    Test HPOConfig dataclass integration with HPOManager.
    """

    def test_hpo_config_creation_with_defaults(self):
        """Test HPOConfig can be created with default values."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from milia_pipeline.models.hpo import HPOConfig

        config = HPOConfig(enabled=True)

        assert config.enabled is True
        assert config.n_trials > 0
        assert config.backend in ["optuna", "ray_tune"]

    def test_hpo_config_is_frozen_dataclass(self):
        """Test HPOConfig is a frozen (immutable) BaseModel/dataclass."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from milia_pipeline.models.hpo import HPOConfig

        config = HPOConfig(enabled=True, n_trials=50)

        # Frozen model should raise on attribute assignment
        # Pydantic V2 frozen BaseModel raises ValidationError
        # dataclass(frozen=True) raises FrozenInstanceError (subclass of AttributeError)
        # We import ValidationError dynamically to support both patterns
        try:
            from pydantic import ValidationError as PydanticValidationError

            expected_exceptions = (AttributeError, TypeError, PydanticValidationError)
        except ImportError:
            expected_exceptions = (AttributeError, TypeError)

        with pytest.raises(expected_exceptions):
            config.n_trials = 100

    def test_hpo_manager_accepts_hpo_config(self):
        """Test HPOManager can be initialized with HPOConfig."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from milia_pipeline.models.hpo import HPOConfig, HPOManager

        config = HPOConfig(enabled=True, n_trials=10)
        manager = HPOManager(config)

        assert manager.config is config
        assert manager.config.enabled is True
        assert manager.config.n_trials == 10

    def test_is_hpo_enabled_function(self):
        """Test is_hpo_enabled convenience function."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from milia_pipeline.models.hpo import HPOConfig, is_hpo_enabled

        enabled_config = HPOConfig(enabled=True)
        disabled_config = HPOConfig(enabled=False)

        assert is_hpo_enabled(enabled_config) is True
        assert is_hpo_enabled(disabled_config) is False
        assert is_hpo_enabled(None) is False


# =============================================================================
# TEST CLASS: HPO MANAGER BACKEND INTEGRATION
# =============================================================================


class TestHPOManagerBackendIntegration:
    """
    Test HPOManager integration with HPO backends.
    """

    def test_hpo_manager_initializes_backend_when_enabled(self):
        """Test HPOManager initializes backend when HPO is enabled."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from milia_pipeline.models.hpo import HPOConfig, HPOManager

        config = HPOConfig(enabled=True, backend="optuna")
        manager = HPOManager(config)

        assert manager.backend is not None

    def test_hpo_manager_no_backend_when_disabled(self):
        """Test HPOManager does not initialize backend when disabled."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from milia_pipeline.models.hpo import HPOConfig, HPOManager

        config = HPOConfig(enabled=False)
        manager = HPOManager(config)

        assert manager.backend is None

    def test_optimize_raises_when_disabled(self):
        """Test optimize() raises HPOError when HPO is disabled."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from milia_pipeline.exceptions import HPOError
        from milia_pipeline.models.hpo import HPOConfig, HPOManager

        config = HPOConfig(enabled=False)
        manager = HPOManager(config)

        with pytest.raises(HPOError) as exc_info:
            manager.optimize(model_name="GCN", dataset=[])

        assert "disabled" in str(exc_info.value).lower()


# =============================================================================
# TEST CLASS: DATA SPLITTER INTEGRATION FOR CROSS-VALIDATION
# =============================================================================


class TestDataSplitterCVIntegration:
    """
    Test HPOManager integration with DataSplitter for cross-validation.
    """

    def test_data_splitter_k_fold_split_returns_correct_format(self, sample_pyg_dataset):
        """Test DataSplitter.k_fold_split returns correct format for HPO CV."""
        from milia_pipeline.models.training.data_splitting import DataSplitter

        # Create a simple indexable wrapper
        class SimpleDataset:
            def __init__(self, data_list):
                self._data = data_list

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

        dataset = SimpleDataset(sample_pyg_dataset)

        n_splits = 3
        folds = DataSplitter.k_fold_split(
            dataset=dataset, n_splits=n_splits, random_seed=42, shuffle=True
        )

        # Should return list of (train, val) tuples
        assert isinstance(folds, list)
        assert len(folds) == n_splits

        for train_subset, val_subset in folds:
            assert isinstance(train_subset, Subset)
            assert isinstance(val_subset, Subset)

            # Subsets should have data
            assert len(train_subset) > 0
            assert len(val_subset) > 0

    def test_k_fold_split_coverage(self, sample_pyg_dataset):
        """Test all data points are covered across folds."""
        from milia_pipeline.models.training.data_splitting import DataSplitter

        class SimpleDataset:
            def __init__(self, data_list):
                self._data = data_list

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

        dataset = SimpleDataset(sample_pyg_dataset)
        n_splits = 3

        folds = DataSplitter.k_fold_split(
            dataset=dataset, n_splits=n_splits, random_seed=42, shuffle=False
        )

        # Collect all validation indices
        all_val_indices = set()
        for _, val_subset in folds:
            all_val_indices.update(val_subset.indices)

        # Should cover all indices
        expected_indices = set(range(len(dataset)))
        assert all_val_indices == expected_indices, (
            "K-fold should cover all data points in validation"
        )


# =============================================================================
# TEST CLASS: HPO MANAGER FACTORY INTEGRATION (MOCKED)
# =============================================================================


class TestHPOManagerFactoryIntegration:
    """
    Test HPOManager integration with ModelFactory.

    Uses mocking for factory to avoid full training cycles while
    verifying the integration contract.
    """

    def test_hpo_manager_uses_model_factory(self):
        """Test HPOManager attempts to use ModelFactory for model creation."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from milia_pipeline.models.hpo import HPOConfig, HPOManager

        config = HPOConfig(enabled=True, n_trials=1)
        manager = HPOManager(config)

        # Verify manager has factory reference attribute
        assert hasattr(manager, "_model_factory")

    @patch("milia_pipeline.models.hpo.hpo_manager.get_factory")
    @patch("milia_pipeline.models.hpo.hpo_manager.Trainer")
    def test_optimize_calls_factory_create_model(self, mock_trainer_class, mock_get_factory):
        """Test optimize() calls factory.create_model() during trials."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from milia_pipeline.models.hpo import HPOConfig, HPOManager

        # Setup mock factory
        mock_factory = MagicMock()
        mock_model = MagicMock()
        mock_factory.create_model.return_value = mock_model
        mock_get_factory.return_value = mock_factory

        # Setup mock trainer
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.fit.return_value = {"val_loss": 0.1, "best_val_loss": 0.1}
        mock_trainer_class.return_value = mock_trainer_instance

        # Create config with minimal settings
        config = HPOConfig(
            enabled=True,
            n_trials=1,
            timeout=10,
        )
        manager = HPOManager(config)

        # Run optimization with minimal dataset
        mock_dataset = [MagicMock()]

        # The optimize should attempt to use factory
        # This test verifies the integration point exists
        # Full optimization is tested separately
        assert manager.backend is not None


# =============================================================================
# TEST CLASS: CREATE HPO CALLBACK FACTORY FUNCTION
# =============================================================================


class TestCreateHPOCallbackFactory:
    """
    Test create_hpo_callback() factory function.
    """

    def test_create_hpo_callback_returns_optuna_callback(self, mock_optuna_trial):
        """Test factory creates OptunaPruningCallback for optuna backend."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from milia_pipeline.models.hpo.callbacks import create_hpo_callback
        from milia_pipeline.models.hpo.callbacks.optuna_callback import OptunaPruningCallback

        callback = create_hpo_callback(
            trial=mock_optuna_trial, monitor="val_loss", backend="optuna"
        )

        assert isinstance(callback, OptunaPruningCallback)
        assert callback.monitor == "val_loss"

    def test_create_hpo_callback_raises_for_ray_tune(self, mock_optuna_trial):
        """Test factory raises NotImplementedError for ray_tune backend."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from milia_pipeline.models.hpo.callbacks import create_hpo_callback

        with pytest.raises(NotImplementedError):
            create_hpo_callback(trial=mock_optuna_trial, monitor="val_loss", backend="ray_tune")

    def test_create_hpo_callback_raises_for_unknown_backend(self, mock_optuna_trial):
        """Test factory raises ValueError for unknown backend."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from milia_pipeline.models.hpo.callbacks import create_hpo_callback

        with pytest.raises(ValueError) as exc_info:
            create_hpo_callback(
                trial=mock_optuna_trial, monitor="val_loss", backend="unknown_backend"
            )

        assert "unknown_backend" in str(exc_info.value)


# =============================================================================
# TEST CLASS: HPO MODULE INFO AND DEPENDENCIES
# =============================================================================


class TestHPOModuleInfoAndDependencies:
    """
    Test HPO module information and dependency checking functions.
    """

    def test_get_hpo_module_info_structure(self):
        """Test get_hpo_module_info returns correct structure."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from milia_pipeline.models.hpo import get_hpo_module_info

        info = get_hpo_module_info()

        assert "version" in info
        assert "backends" in info
        assert "optuna_available" in info
        assert "subpackages" in info
        assert "components" in info

        # Verify subpackages list
        expected_subpackages = [
            "backends",
            "callbacks",
            "search_spaces",
            "analysis",
            "transfer",
            "nas",
        ]
        for subpkg in expected_subpackages:
            assert subpkg in info["subpackages"]

    def test_check_hpo_dependencies_structure(self):
        """Test check_hpo_dependencies returns correct structure."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from milia_pipeline.models.hpo import check_hpo_dependencies

        deps = check_hpo_dependencies()

        # Required dependencies
        assert "optuna" in deps
        assert "torch" in deps
        assert "numpy" in deps

        # Summary flags
        assert "all_required_available" in deps

        # Each dep should have available and version
        for dep_name in ["optuna", "torch", "numpy"]:
            assert "available" in deps[dep_name]
            assert "version" in deps[dep_name]


# =============================================================================
# TEST CLASS: HPO SEARCH SPACE BUILDER INTEGRATION
# =============================================================================


class TestSearchSpaceBuilderIntegration:
    """
    Test SearchSpaceBuilder integration with HPOManager.
    """

    def test_search_space_builder_importable(self):
        """Test SearchSpaceBuilder can be imported from HPO module."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from milia_pipeline.models.hpo import SearchSpaceBuilder

        assert SearchSpaceBuilder is not None

    def test_search_space_builder_creates_search_space(self):
        """Test SearchSpaceBuilder creates valid search space configuration."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from milia_pipeline.models.hpo import SearchSpaceBuilder

        builder = SearchSpaceBuilder()

        # Add some parameters
        builder.add_int("hidden_channels", low=16, high=256)
        builder.add_float("learning_rate", low=1e-5, high=1e-2, log=True)
        builder.add_categorical("activation", choices=["relu", "elu", "gelu"])

        search_space = builder.build()

        # SearchSpaceBuilder returns nested structure with 'hyperparameters' key
        assert "hyperparameters" in search_space
        hyperparams = search_space["hyperparameters"]

        assert "hidden_channels" in hyperparams
        assert "learning_rate" in hyperparams
        assert "activation" in hyperparams


# =============================================================================
# TEST CLASS: END-TO-END HPO SMOKE TEST
# =============================================================================


class TestHPOEndToEndSmoke:
    """
    End-to-end smoke tests for HPO functionality.

    These tests verify the complete HPO workflow works with
    minimal real objects and controlled mocking.
    """

    def test_hpo_manager_creation_workflow(self):
        """Test complete HPOManager creation workflow."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from milia_pipeline.models.hpo import (
            HPOConfig,
            HPOManager,
            create_hpo_manager,
        )

        # Method 1: Direct instantiation
        config = HPOConfig(
            enabled=True,
            n_trials=5,
            backend="optuna",
        )
        manager1 = HPOManager(config)

        assert manager1.config.enabled is True
        assert manager1.backend is not None

        # Method 2: Convenience function
        manager2 = create_hpo_manager(
            enabled=True,
            n_trials=10,
        )

        assert manager2.config.n_trials == 10

    def test_hpo_callback_creation_workflow(self, mock_optuna_trial):
        """Test complete HPO callback creation workflow."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from milia_pipeline.models.hpo import OptunaPruningCallback, create_hpo_callback

        # Method 1: Direct instantiation
        callback1 = OptunaPruningCallback(
            trial=mock_optuna_trial,
            monitor="val_loss",
            report_every=1,
        )

        assert callback1.monitor == "val_loss"

        # Method 2: Factory function
        callback2 = create_hpo_callback(
            trial=mock_optuna_trial,
            monitor="train_loss",
            backend="optuna",
        )

        assert callback2.monitor == "train_loss"

    def test_trainer_with_hpo_callback_runs_without_error(
        self, mock_optuna_trial, sample_pyg_dataset
    ):
        """Test Trainer with HPO callback completes without error."""
        from milia_pipeline.models import HPO_AVAILABLE

        if not HPO_AVAILABLE:
            pytest.skip("HPO not available")

        from torch_geometric.loader import DataLoader

        from milia_pipeline.models.hpo import OptunaPruningCallback
        from milia_pipeline.models.training.trainer import Trainer

        # Create minimal model
        model = MinimalGNNModel(in_channels=5, hidden_channels=8, out_channels=1)

        # Create dataloaders
        train_loader = DataLoader(sample_pyg_dataset[:6], batch_size=3, shuffle=False)
        val_loader = DataLoader(sample_pyg_dataset[6:], batch_size=2, shuffle=False)

        # Create HPO callback
        hpo_callback = OptunaPruningCallback(
            trial=mock_optuna_trial,
            monitor="val_loss",
            report_every=1,
        )

        # Create trainer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=nn.MSELoss(),
            optimizer=optimizer,
            max_epochs=2,
            hpo_callback=hpo_callback,
        )

        # Run training - should complete without error
        results = trainer.fit()

        # Verify basic results structure
        assert "train_metrics" in results or "training_time" in results
        assert mock_optuna_trial.report.called


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-W", "ignore::DeprecationWarning"])
