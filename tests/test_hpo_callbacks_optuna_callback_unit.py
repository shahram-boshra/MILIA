#!/usr/bin/env python3
"""
Complete Unit Test Suite for milia_pipeline/models/hpo/callbacks/optuna_callback.py Module

Tests OptunaPruningCallback class and create_hpo_callback factory function including:
- OPTUNA_AVAILABLE flag behavior
- OptunaPruningCallback.__init__()
  - Successful initialization with all parameters
  - Default parameter values
  - ImportError when optuna is not installed
  - Attribute storage (trial, monitor, report_every)
  - Logger debug message on initialization
  - _reported_steps initialized as empty set
- OptunaPruningCallback.set_trainer()
  - Stores trainer reference correctly
- OptunaPruningCallback.on_train_begin(trainer)
  - Logs debug message with trial number and monitor metric
  - Accepts trainer as required argument
- OptunaPruningCallback.on_epoch_end()
  - Reports metric to trial at correct frequency
  - Skips reporting based on report_every parameter
  - Handles missing monitor metric with alternative names (val_, validation_)
  - Logs warning when metric not found
  - Stores last reported value
  - Calls trial.report() with correct arguments
  - Raises optuna.TrialPruned when trial.should_prune() returns True
  - Logs info message when trial is pruned
  - Prevents duplicate step reporting via _reported_steps tracking
  - Logs skip message for duplicate steps
  - Still checks should_prune() for duplicate steps
  - Can prune even on duplicate step reports
- OptunaPruningCallback.on_train_end(trainer)
  - Logs debug message with final metric value
  - Uses _last_reported_value as fallback
  - Uses metrics_history when available (overrides fallback)
  - Handles trainer without metrics_history attribute
  - Handles empty metric history list
  - Only takes trainer argument (no metrics kwarg)
- OptunaPruningCallback.should_stop() method
  - Returns True when trial.should_prune() is True
  - Returns False when trial.should_prune() is False
  - Is a method (not a property) that delegates to trial.should_prune()
- create_hpo_callback() factory function
  - Creates OptunaPruningCallback for "optuna" backend
  - Raises NotImplementedError for "ray_tune" backend
  - Raises ValueError for unknown backend
  - Passes all parameters correctly to OptunaPruningCallback

This is a PRODUCTION-READY test suite with comprehensive coverage.

Author: Milia Team
Version: 1.3.0

Changelog:
- v1.3.0: Enhanced test coverage for production-readiness:
          - Added test for _reported_steps initialization as empty set
          - Added test for should_prune() still called on duplicate steps
          - Added test for pruning possible on duplicate step reports
          - Added test for on_train_end using metrics_history when available
          - Added test for on_train_end when trainer lacks metrics_history attr
          - Added test for on_train_end handling empty metric history list
- v1.2.0: Fixed tests to match actual implementation:
          - on_train_begin() requires trainer argument
          - on_train_end() only takes trainer (no metrics kwarg)
          - should_stop() is a method, not a property
- v1.1.0: Added tests for duplicate step prevention (_reported_steps feature)
- v1.0.0: Initial release
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_optuna():
    """Create a comprehensive mock for the optuna module."""
    mock = MagicMock()

    # Mock Trial class
    mock.Trial = MagicMock()

    # Mock TrialPruned exception
    mock.TrialPruned = type("TrialPruned", (Exception,), {})

    return mock


@pytest.fixture
def mock_trial():
    """Create a mock Optuna trial object."""
    trial = MagicMock()
    trial.number = 5
    trial.report = MagicMock()
    trial.should_prune = MagicMock(return_value=False)
    return trial


@pytest.fixture
def mock_trial_pruned():
    """Create a mock Optuna trial object that signals pruning."""
    trial = MagicMock()
    trial.number = 7
    trial.report = MagicMock()
    trial.should_prune = MagicMock(return_value=True)
    return trial


@pytest.fixture
def mock_trainer():
    """Create a mock Trainer object."""
    trainer = MagicMock()
    trainer.model = MagicMock()
    trainer.optimizer = MagicMock()
    return trainer


@pytest.fixture
def sample_metrics():
    """Create sample metrics dict."""
    return {
        "train_loss": 0.5,
        "train_mae": 0.1,
        "val_loss": 0.45,
        "val_mae": 0.09,
    }


@pytest.fixture
def sample_metrics_alternative_names():
    """Create sample metrics with alternative naming conventions."""
    return {
        "train_loss": 0.5,
        "validation_loss": 0.42,
        "loss": 0.4,
    }


@pytest.fixture
def sample_metrics_missing_monitor():
    """Create sample metrics without the monitored metric."""
    return {
        "train_loss": 0.5,
        "train_accuracy": 0.85,
        "other_metric": 0.7,
    }


# =============================================================================
# MOCK CALLBACK BASE CLASS
# =============================================================================


class MockCallback:
    """Mock Callback base class for testing when real Callback is unavailable.

    Note: These signatures must match the actual Callback ABC from callbacks.py
    to ensure tests accurately reflect the expected interface.
    """

    def set_trainer(self, trainer):
        pass

    def on_train_begin(self, trainer):
        pass

    def on_epoch_end(self, trainer, epoch, metrics):
        pass

    def on_train_end(self, trainer):
        pass


# =============================================================================
# MODULE IMPORT TESTS
# =============================================================================


class TestOptunaAvailableFlag:
    """Test OPTUNA_AVAILABLE module-level flag behavior."""

    def test_optuna_available_is_defined(self):
        """Test OPTUNA_AVAILABLE flag is defined in the module."""
        from milia_pipeline.models.hpo.callbacks.optuna_callback import OPTUNA_AVAILABLE

        assert isinstance(OPTUNA_AVAILABLE, bool)

    def test_optuna_available_true_when_optuna_installed(self):
        """Test OPTUNA_AVAILABLE is True when optuna is installed."""
        try:
            import optuna

            from milia_pipeline.models.hpo.callbacks.optuna_callback import OPTUNA_AVAILABLE

            assert OPTUNA_AVAILABLE is True
        except ImportError:
            pytest.skip("Optuna not installed")

    def test_module_defines_optuna_pruning_callback_class(self):
        """Test OptunaPruningCallback class is defined in module."""
        from milia_pipeline.models.hpo.callbacks.optuna_callback import OptunaPruningCallback

        assert OptunaPruningCallback is not None

    def test_module_defines_create_hpo_callback_function(self):
        """Test create_hpo_callback function is defined in module."""
        from milia_pipeline.models.hpo.callbacks.optuna_callback import create_hpo_callback

        assert callable(create_hpo_callback)


# =============================================================================
# OPTUNA PRUNING CALLBACK INITIALIZATION TESTS
# =============================================================================


class TestOptunaPruningCallbackInit:
    """Test OptunaPruningCallback.__init__() method."""

    def test_init_success_with_default_parameters(self, mock_optuna, mock_trial):
        """Test successful initialization with default parameters."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial)

            assert callback.trial is mock_trial
            assert callback.monitor == "val_loss"  # Default value
            assert callback.report_every == 1  # Default value

    def test_init_success_with_custom_parameters(self, mock_optuna, mock_trial):
        """Test successful initialization with custom parameters."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_mae", report_every=5)

            assert callback.trial is mock_trial
            assert callback.monitor == "val_mae"
            assert callback.report_every == 5

    def test_init_sets_trainer_to_none(self, mock_optuna, mock_trial):
        """Test initialization sets _trainer to None."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial)

            assert callback._trainer is None

    def test_init_sets_last_reported_value_to_none(self, mock_optuna, mock_trial):
        """Test initialization sets _last_reported_value to None."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial)

            assert callback._last_reported_value is None

    def test_init_sets_reported_steps_to_empty_set(self, mock_optuna, mock_trial):
        """Test initialization sets _reported_steps to empty set."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial)

            assert isinstance(callback._reported_steps, set)
            assert len(callback._reported_steps) == 0

    def test_init_raises_import_error_when_optuna_not_available(self, mock_trial):
        """Test ImportError raised when optuna is not installed."""
        with patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", False):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import OptunaPruningCallback

            with pytest.raises(ImportError) as exc_info:
                OptunaPruningCallback(trial=mock_trial)

            error = exc_info.value
            assert "Optuna is required" in str(error)
            assert "pip install optuna" in str(error)

    def test_init_logs_debug_message(self, mock_optuna, mock_trial):
        """Test initialization logs debug message."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.logger") as mock_logger,
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_mae", report_every=3)

            mock_logger.debug.assert_called()
            debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
            assert any("val_mae" in str(c) for c in debug_calls)
            assert any("3" in str(c) for c in debug_calls)

    def test_init_calls_super_init(self, mock_optuna, mock_trial):
        """Test initialization calls parent class __init__."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            # Should not raise - parent __init__ should be called
            callback = OptunaPruningCallback(trial=mock_trial)
            assert callback is not None


# =============================================================================
# SET_TRAINER TESTS
# =============================================================================


class TestSetTrainer:
    """Test OptunaPruningCallback.set_trainer() method."""

    def test_set_trainer_stores_trainer_reference(self, mock_optuna, mock_trial, mock_trainer):
        """Test set_trainer stores the trainer reference."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial)
            assert callback._trainer is None

            callback.set_trainer(mock_trainer)

            assert callback._trainer is mock_trainer

    def test_set_trainer_can_replace_existing_trainer(self, mock_optuna, mock_trial, mock_trainer):
        """Test set_trainer can replace an existing trainer reference."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial)

            trainer1 = MagicMock()
            trainer2 = MagicMock()

            callback.set_trainer(trainer1)
            assert callback._trainer is trainer1

            callback.set_trainer(trainer2)
            assert callback._trainer is trainer2

    def test_set_trainer_accepts_none(self, mock_optuna, mock_trial, mock_trainer):
        """Test set_trainer accepts None as trainer."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial)
            callback.set_trainer(mock_trainer)
            assert callback._trainer is mock_trainer

            callback.set_trainer(None)
            assert callback._trainer is None


# =============================================================================
# ON_TRAIN_BEGIN TESTS
# =============================================================================


class TestOnTrainBegin:
    """Test OptunaPruningCallback.on_train_begin() method."""

    def test_on_train_begin_logs_debug_message(self, mock_optuna, mock_trial, mock_trainer):
        """Test on_train_begin logs debug message with trial number."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.logger") as mock_logger,
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")
            mock_logger.reset_mock()  # Reset after __init__ logging

            callback.on_train_begin(mock_trainer)

            mock_logger.debug.assert_called()
            debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
            assert any("5" in str(c) for c in debug_calls)  # trial.number = 5
            assert any("val_loss" in str(c) for c in debug_calls)

    def test_on_train_begin_accepts_trainer_argument(self, mock_optuna, mock_trial, mock_trainer):
        """Test on_train_begin accepts trainer argument."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial)

            # Should not raise
            callback.on_train_begin(mock_trainer)

    def test_on_train_begin_returns_none(self, mock_optuna, mock_trial, mock_trainer):
        """Test on_train_begin returns None."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial)
            result = callback.on_train_begin(mock_trainer)

            assert result is None


# =============================================================================
# ON_EPOCH_END TESTS
# =============================================================================


class TestOnEpochEnd:
    """Test OptunaPruningCallback.on_epoch_end() method."""

    def test_on_epoch_end_reports_metric_to_trial(
        self, mock_optuna, mock_trial, mock_trainer, sample_metrics
    ):
        """Test on_epoch_end reports metric value to trial."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")

            callback.on_epoch_end(mock_trainer, epoch=1, metrics=sample_metrics)

            mock_trial.report.assert_called_once_with(0.45, 1)

    def test_on_epoch_end_reports_every_epoch_by_default(
        self, mock_optuna, mock_trial, mock_trainer, sample_metrics
    ):
        """Test on_epoch_end reports every epoch when report_every=1."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss", report_every=1)

            for epoch in range(5):
                callback.on_epoch_end(mock_trainer, epoch=epoch, metrics=sample_metrics)

            assert mock_trial.report.call_count == 5

    def test_on_epoch_end_respects_report_every_parameter(
        self, mock_optuna, mock_trial, mock_trainer, sample_metrics
    ):
        """Test on_epoch_end only reports at specified frequency."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss", report_every=3)

            # Epochs 0, 1, 2, 3, 4, 5 - should report at 0, 3
            for epoch in range(6):
                callback.on_epoch_end(mock_trainer, epoch=epoch, metrics=sample_metrics)

            # report_every=3 means epoch % 3 == 0 -> epochs 0, 3
            assert mock_trial.report.call_count == 2

    def test_on_epoch_end_skips_intermediate_epochs(
        self, mock_optuna, mock_trial, mock_trainer, sample_metrics
    ):
        """Test on_epoch_end skips epochs that don't match report_every."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss", report_every=5)

            # Epochs 1, 2, 3, 4 should be skipped
            for epoch in [1, 2, 3, 4]:
                callback.on_epoch_end(mock_trainer, epoch=epoch, metrics=sample_metrics)

            mock_trial.report.assert_not_called()

    def test_on_epoch_end_finds_metric_with_val_prefix(self, mock_optuna, mock_trial, mock_trainer):
        """Test on_epoch_end finds metric using val_ prefix when direct name not found."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="loss")

            metrics = {"val_loss": 0.42, "train_loss": 0.5}
            callback.on_epoch_end(mock_trainer, epoch=0, metrics=metrics)

            mock_trial.report.assert_called_once_with(0.42, 0)

    def test_on_epoch_end_finds_metric_with_validation_prefix(
        self, mock_optuna, mock_trial, mock_trainer, sample_metrics_alternative_names
    ):
        """Test on_epoch_end finds metric using validation_ prefix."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="loss")

            # metrics has validation_loss but no val_loss or loss direct
            metrics = {"validation_loss": 0.38, "train_loss": 0.5}
            callback.on_epoch_end(mock_trainer, epoch=0, metrics=metrics)

            mock_trial.report.assert_called_once_with(0.38, 0)

    def test_on_epoch_end_removes_val_prefix_as_fallback(
        self, mock_optuna, mock_trial, mock_trainer
    ):
        """Test on_epoch_end tries removing val_ prefix to find metric."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")

            # Only has 'loss' not 'val_loss'
            metrics = {"loss": 0.35, "train_loss": 0.5}
            callback.on_epoch_end(mock_trainer, epoch=0, metrics=metrics)

            mock_trial.report.assert_called_once_with(0.35, 0)

    def test_on_epoch_end_logs_warning_when_metric_not_found(
        self, mock_optuna, mock_trial, mock_trainer, sample_metrics_missing_monitor
    ):
        """Test on_epoch_end logs warning when monitor metric is not found."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.logger") as mock_logger,
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")
            mock_logger.reset_mock()

            callback.on_epoch_end(mock_trainer, epoch=5, metrics=sample_metrics_missing_monitor)

            mock_logger.warning.assert_called()
            warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
            assert any("val_loss" in str(c) for c in warning_calls)
            assert any("not found" in str(c) for c in warning_calls)

    def test_on_epoch_end_returns_early_when_metric_not_found(
        self, mock_optuna, mock_trial, mock_trainer, sample_metrics_missing_monitor
    ):
        """Test on_epoch_end returns early without reporting when metric not found."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")

            callback.on_epoch_end(mock_trainer, epoch=5, metrics=sample_metrics_missing_monitor)

            mock_trial.report.assert_not_called()
            mock_trial.should_prune.assert_not_called()

    def test_on_epoch_end_stores_last_reported_value(
        self, mock_optuna, mock_trial, mock_trainer, sample_metrics
    ):
        """Test on_epoch_end stores the last reported value."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")
            assert callback._last_reported_value is None

            callback.on_epoch_end(mock_trainer, epoch=0, metrics=sample_metrics)

            assert callback._last_reported_value == 0.45

    def test_on_epoch_end_updates_last_reported_value(self, mock_optuna, mock_trial, mock_trainer):
        """Test on_epoch_end updates last reported value on each report."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")

            metrics1 = {"val_loss": 0.5}
            callback.on_epoch_end(mock_trainer, epoch=0, metrics=metrics1)
            assert callback._last_reported_value == 0.5

            metrics2 = {"val_loss": 0.3}
            callback.on_epoch_end(mock_trainer, epoch=1, metrics=metrics2)
            assert callback._last_reported_value == 0.3

    def test_on_epoch_end_logs_debug_message(
        self, mock_optuna, mock_trial, mock_trainer, sample_metrics
    ):
        """Test on_epoch_end logs debug message with trial and metric info."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.logger") as mock_logger,
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")
            mock_logger.reset_mock()

            callback.on_epoch_end(mock_trainer, epoch=10, metrics=sample_metrics)

            mock_logger.debug.assert_called()
            debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
            assert any("Trial 5" in str(c) or "trial" in str(c).lower() for c in debug_calls)

    def test_on_epoch_end_checks_for_pruning(
        self, mock_optuna, mock_trial, mock_trainer, sample_metrics
    ):
        """Test on_epoch_end calls trial.should_prune()."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")

            callback.on_epoch_end(mock_trainer, epoch=0, metrics=sample_metrics)

            mock_trial.should_prune.assert_called_once()

    def test_on_epoch_end_raises_trial_pruned_when_should_prune(
        self, mock_optuna, mock_trial_pruned, mock_trainer, sample_metrics
    ):
        """Test on_epoch_end raises TrialPruned when trial.should_prune() returns True."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial_pruned, monitor="val_loss")

            with pytest.raises(mock_optuna.TrialPruned) as exc_info:
                callback.on_epoch_end(mock_trainer, epoch=3, metrics=sample_metrics)

            assert "epoch 3" in str(exc_info.value)
            assert "val_loss" in str(exc_info.value)

    def test_on_epoch_end_logs_info_when_pruned(
        self, mock_optuna, mock_trial_pruned, mock_trainer, sample_metrics
    ):
        """Test on_epoch_end logs info message when trial is pruned."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.logger") as mock_logger,
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial_pruned, monitor="val_loss")
            mock_logger.reset_mock()

            with pytest.raises(mock_optuna.TrialPruned):
                callback.on_epoch_end(mock_trainer, epoch=3, metrics=sample_metrics)

            mock_logger.info.assert_called()
            info_calls = [str(c) for c in mock_logger.info.call_args_list]
            assert any("pruned" in str(c).lower() for c in info_calls)

    def test_on_epoch_end_does_not_raise_when_not_pruned(
        self, mock_optuna, mock_trial, mock_trainer, sample_metrics
    ):
        """Test on_epoch_end does not raise when trial.should_prune() returns False."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")

            # Should not raise - trial.should_prune() returns False by default
            callback.on_epoch_end(mock_trainer, epoch=0, metrics=sample_metrics)

    def test_on_epoch_end_handles_empty_metrics_dict(self, mock_optuna, mock_trial, mock_trainer):
        """Test on_epoch_end handles empty metrics dictionary."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")

            # Should not raise, just return early
            callback.on_epoch_end(mock_trainer, epoch=0, metrics={})

            mock_trial.report.assert_not_called()

    def test_on_epoch_end_prevents_duplicate_step_reporting(
        self, mock_optuna, mock_trial, mock_trainer, sample_metrics
    ):
        """Test on_epoch_end prevents reporting the same step twice."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")

            # Call on_epoch_end twice with same epoch
            callback.on_epoch_end(mock_trainer, epoch=5, metrics=sample_metrics)
            callback.on_epoch_end(mock_trainer, epoch=5, metrics=sample_metrics)

            # Should only report once despite two calls
            assert mock_trial.report.call_count == 1

    def test_on_epoch_end_tracks_reported_steps(
        self, mock_optuna, mock_trial, mock_trainer, sample_metrics
    ):
        """Test on_epoch_end tracks which steps have been reported."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")

            # Initially no steps reported
            assert len(callback._reported_steps) == 0

            # Report epochs 0, 2, 4
            callback.on_epoch_end(mock_trainer, epoch=0, metrics=sample_metrics)
            callback.on_epoch_end(mock_trainer, epoch=2, metrics=sample_metrics)
            callback.on_epoch_end(mock_trainer, epoch=4, metrics=sample_metrics)

            # Tracked steps should include 0, 2, 4
            assert 0 in callback._reported_steps
            assert 2 in callback._reported_steps
            assert 4 in callback._reported_steps
            assert len(callback._reported_steps) == 3

    def test_on_epoch_end_logs_skip_message_for_duplicate_steps(
        self, mock_optuna, mock_trial, mock_trainer, sample_metrics
    ):
        """Test on_epoch_end logs debug message when skipping duplicate step."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.logger") as mock_logger,
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")

            # First call
            callback.on_epoch_end(mock_trainer, epoch=3, metrics=sample_metrics)
            mock_logger.reset_mock()

            # Second call to same epoch
            callback.on_epoch_end(mock_trainer, epoch=3, metrics=sample_metrics)

            # Should log about skipping duplicate
            debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
            assert any(
                "already reported" in str(c).lower() or "skipping" in str(c).lower()
                for c in debug_calls
            )

    def test_on_epoch_end_still_checks_pruning_for_duplicate_steps(
        self, mock_optuna, mock_trial, mock_trainer, sample_metrics
    ):
        """Test on_epoch_end still calls should_prune() even for duplicate step reports."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")

            # First call
            callback.on_epoch_end(mock_trainer, epoch=3, metrics=sample_metrics)
            mock_trial.should_prune.reset_mock()

            # Second call to same epoch - should still check pruning
            callback.on_epoch_end(mock_trainer, epoch=3, metrics=sample_metrics)

            # should_prune should still be called even for duplicate
            mock_trial.should_prune.assert_called_once()

    def test_on_epoch_end_can_prune_on_duplicate_step(
        self, mock_optuna, mock_trainer, sample_metrics
    ):
        """Test on_epoch_end can raise TrialPruned even on duplicate step report."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            # Create trial that will prune on second call to same epoch
            trial = MagicMock()
            trial.number = 5
            trial.report = MagicMock()
            trial.should_prune = MagicMock(side_effect=[False, True])

            callback = OptunaPruningCallback(trial=trial, monitor="val_loss")

            # First call - no pruning
            callback.on_epoch_end(mock_trainer, epoch=3, metrics=sample_metrics)

            # Second call to same epoch - should prune
            with pytest.raises(mock_optuna.TrialPruned):
                callback.on_epoch_end(mock_trainer, epoch=3, metrics=sample_metrics)

            # Report should only be called once (first call)
            assert trial.report.call_count == 1


# =============================================================================
# ON_TRAIN_END TESTS
# =============================================================================


class TestOnTrainEnd:
    """Test OptunaPruningCallback.on_train_end() method."""

    def test_on_train_end_logs_debug_message(self, mock_optuna, mock_trial, mock_trainer):
        """Test on_train_end logs debug message."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.logger") as mock_logger,
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")
            mock_logger.reset_mock()

            callback.on_train_end(mock_trainer)

            mock_logger.debug.assert_called()
            debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
            assert any("ended" in str(c).lower() or "end" in str(c).lower() for c in debug_calls)

    def test_on_train_end_logs_final_metric_value(self, mock_optuna, mock_trial, mock_trainer):
        """Test on_train_end logs the final metric value from _last_reported_value."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.logger") as mock_logger,
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")
            callback._last_reported_value = 0.45
            mock_logger.reset_mock()

            callback.on_train_end(mock_trainer)

            debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
            # Final val_loss is 0.45
            assert any("0.45" in str(c) or "val_loss" in str(c) for c in debug_calls)

    def test_on_train_end_uses_last_reported_value_as_fallback(
        self, mock_optuna, mock_trial, mock_trainer
    ):
        """Test on_train_end uses _last_reported_value as fallback when metrics_history is empty."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.logger") as mock_logger,
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")
            callback._last_reported_value = 0.33  # Set manually
            mock_logger.reset_mock()

            # Ensure trainer.metrics_history doesn't have our metric
            # so fallback to _last_reported_value is used
            mock_trainer.metrics_history = {}

            callback.on_train_end(mock_trainer)

            debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
            assert any("0.33" in str(c) for c in debug_calls)

    def test_on_train_end_uses_metrics_history_when_available(
        self, mock_optuna, mock_trial, mock_trainer
    ):
        """Test on_train_end uses metrics_history when it contains the monitor metric."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.logger") as mock_logger,
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")
            callback._last_reported_value = 0.33  # Should be overridden
            mock_logger.reset_mock()

            # metrics_history has our metric - should use last value from history
            mock_trainer.metrics_history = {"val_loss": [0.5, 0.4, 0.25]}

            callback.on_train_end(mock_trainer)

            debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
            # Should log 0.25 (last value from history), not 0.33
            assert any("0.25" in str(c) for c in debug_calls)

    def test_on_train_end_handles_trainer_without_metrics_history_attr(
        self, mock_optuna, mock_trial
    ):
        """Test on_train_end handles trainer that has no metrics_history attribute."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.logger") as mock_logger,
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")
            callback._last_reported_value = 0.42
            mock_logger.reset_mock()

            # Create trainer without metrics_history attribute
            trainer_no_history = MagicMock(spec=["model", "optimizer"])
            # Remove metrics_history from spec so hasattr returns False
            del trainer_no_history.metrics_history

            # Should not raise - falls back to _last_reported_value
            callback.on_train_end(trainer_no_history)

            debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
            assert any("0.42" in str(c) for c in debug_calls)

    def test_on_train_end_handles_empty_metric_history_list(
        self, mock_optuna, mock_trial, mock_trainer
    ):
        """Test on_train_end handles when metrics_history has monitor but list is empty."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.logger") as mock_logger,
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")
            callback._last_reported_value = 0.55
            mock_logger.reset_mock()

            # metrics_history exists but val_loss list is empty
            mock_trainer.metrics_history = {"val_loss": [], "train_loss": [0.3]}

            callback.on_train_end(mock_trainer)

            debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
            # Should fall back to _last_reported_value since list is empty
            assert any("0.55" in str(c) for c in debug_calls)

    def test_on_train_end_returns_none(self, mock_optuna, mock_trial, mock_trainer):
        """Test on_train_end returns None."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")
            result = callback.on_train_end(mock_trainer)

            assert result is None

    def test_on_train_end_logs_trial_number(self, mock_optuna, mock_trial, mock_trainer):
        """Test on_train_end includes trial number in log message."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.logger") as mock_logger,
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")
            mock_logger.reset_mock()

            callback.on_train_end(mock_trainer)

            debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
            # trial.number = 5
            assert any("5" in str(c) for c in debug_calls)


# =============================================================================
# SHOULD_STOP METHOD TESTS
# =============================================================================


class TestShouldStopMethod:
    """Test OptunaPruningCallback.should_stop() method."""

    def test_should_stop_returns_false_when_not_pruned(self, mock_optuna, mock_trial):
        """Test should_stop() returns False when trial.should_prune() returns False."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            mock_trial.should_prune.return_value = False
            callback = OptunaPruningCallback(trial=mock_trial)

            assert callback.should_stop() is False

    def test_should_stop_returns_true_when_pruned(self, mock_optuna, mock_trial_pruned):
        """Test should_stop() returns True when trial.should_prune() returns True."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial_pruned)

            assert callback.should_stop() is True

    def test_should_stop_calls_trial_should_prune(self, mock_optuna, mock_trial):
        """Test should_stop() delegates to trial.should_prune()."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial)
            mock_trial.should_prune.reset_mock()

            _ = callback.should_stop()

            mock_trial.should_prune.assert_called_once()

    def test_should_stop_is_method_returning_bool(self, mock_optuna, mock_trial):
        """Test should_stop is a method that returns a bool."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial)

            # Check it's callable
            assert callable(callback.should_stop)
            # Check it returns a bool when called
            result = callback.should_stop()
            assert isinstance(result, bool)


# =============================================================================
# CREATE_HPO_CALLBACK FACTORY FUNCTION TESTS
# =============================================================================


class TestCreateHpoCallback:
    """Test create_hpo_callback() factory function."""

    def test_creates_optuna_pruning_callback_for_optuna_backend(self, mock_optuna, mock_trial):
        """Test factory creates OptunaPruningCallback for 'optuna' backend."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
                create_hpo_callback,
            )

            callback = create_hpo_callback(trial=mock_trial, backend="optuna")

            assert isinstance(callback, OptunaPruningCallback)

    def test_passes_all_parameters_to_optuna_callback(self, mock_optuna, mock_trial):
        """Test factory passes all parameters to OptunaPruningCallback."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                create_hpo_callback,
            )

            callback = create_hpo_callback(
                trial=mock_trial, monitor="val_mae", report_every=10, backend="optuna"
            )

            assert callback.trial is mock_trial
            assert callback.monitor == "val_mae"
            assert callback.report_every == 10

    def test_uses_default_parameters(self, mock_optuna, mock_trial):
        """Test factory uses default parameter values."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                create_hpo_callback,
            )

            callback = create_hpo_callback(trial=mock_trial)

            assert callback.monitor == "val_loss"  # Default
            assert callback.report_every == 1  # Default

    def test_raises_not_implemented_error_for_ray_tune(self, mock_optuna, mock_trial):
        """Test factory raises NotImplementedError for 'ray_tune' backend."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                create_hpo_callback,
            )

            with pytest.raises(NotImplementedError) as exc_info:
                create_hpo_callback(trial=mock_trial, backend="ray_tune")

            assert "Ray Tune" in str(exc_info.value)
            assert "not yet implemented" in str(exc_info.value)

    def test_raises_value_error_for_unknown_backend(self, mock_optuna, mock_trial):
        """Test factory raises ValueError for unknown backend."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                create_hpo_callback,
            )

            with pytest.raises(ValueError) as exc_info:
                create_hpo_callback(trial=mock_trial, backend="unknown_backend")

            assert "Unknown backend" in str(exc_info.value)
            assert "unknown_backend" in str(exc_info.value)

    def test_accepts_optuna_backend_case_sensitive(self, mock_optuna, mock_trial):
        """Test factory is case-sensitive for backend name."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                create_hpo_callback,
            )

            # "optuna" works
            callback = create_hpo_callback(trial=mock_trial, backend="optuna")
            assert callback is not None

            # "Optuna" should fail
            with pytest.raises(ValueError):
                create_hpo_callback(trial=mock_trial, backend="Optuna")


# =============================================================================
# CALLBACK INTEGRATION TESTS
# =============================================================================


class TestCallbackIntegration:
    """Test OptunaPruningCallback integration with training workflow."""

    def test_full_training_workflow(self, mock_optuna, mock_trial, mock_trainer):
        """Test complete training workflow with callback."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")

            # Simulate training workflow
            callback.set_trainer(mock_trainer)
            callback.on_train_begin(mock_trainer)

            for epoch in range(5):
                metrics = {"val_loss": 0.5 - 0.05 * epoch}
                callback.on_epoch_end(mock_trainer, epoch=epoch, metrics=metrics)

            callback.on_train_end(mock_trainer)

            # Verify 5 reports were made
            assert mock_trial.report.call_count == 5
            # Last epoch is 4, so value is 0.5 - 0.05*4 = 0.3
            assert callback._last_reported_value == 0.3

    def test_training_workflow_with_pruning(self, mock_optuna, mock_trainer):
        """Test training workflow that gets pruned mid-training."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            # Create trial that will prune at epoch 3
            trial = MagicMock()
            trial.number = 10
            trial.report = MagicMock()
            trial.should_prune = MagicMock(side_effect=[False, False, False, True])

            callback = OptunaPruningCallback(trial=trial, monitor="val_loss")
            callback.set_trainer(mock_trainer)
            callback.on_train_begin(mock_trainer)

            # Train until pruning
            pruned_at_epoch = None
            for epoch in range(10):
                metrics = {"val_loss": 0.5 + 0.01 * epoch}  # Loss getting worse
                try:
                    callback.on_epoch_end(mock_trainer, epoch=epoch, metrics=metrics)
                except mock_optuna.TrialPruned:
                    pruned_at_epoch = epoch
                    break

            assert pruned_at_epoch == 3
            assert trial.report.call_count == 4  # Epochs 0, 1, 2, 3


# =============================================================================
# MODULE EXPORTS TESTS
# =============================================================================


class TestModuleExports:
    """Test module exports and public API."""

    def test_optuna_pruning_callback_class_exported(self):
        """Test OptunaPruningCallback class is exported."""
        from milia_pipeline.models.hpo.callbacks.optuna_callback import OptunaPruningCallback

        assert OptunaPruningCallback is not None

    def test_create_hpo_callback_function_exported(self):
        """Test create_hpo_callback function is exported."""
        from milia_pipeline.models.hpo.callbacks.optuna_callback import create_hpo_callback

        assert create_hpo_callback is not None
        assert callable(create_hpo_callback)

    def test_optuna_available_flag_exported(self):
        """Test OPTUNA_AVAILABLE flag is exported."""
        from milia_pipeline.models.hpo.callbacks.optuna_callback import OPTUNA_AVAILABLE

        assert isinstance(OPTUNA_AVAILABLE, bool)

    def test_module_has_docstring(self):
        """Test optuna_callback module has docstring."""
        from milia_pipeline.models.hpo.callbacks import optuna_callback

        assert optuna_callback.__doc__ is not None

    def test_class_has_docstring(self):
        """Test OptunaPruningCallback class has docstring."""
        from milia_pipeline.models.hpo.callbacks.optuna_callback import OptunaPruningCallback

        assert OptunaPruningCallback.__doc__ is not None

    def test_create_hpo_callback_has_docstring(self):
        """Test create_hpo_callback function has docstring."""
        from milia_pipeline.models.hpo.callbacks.optuna_callback import create_hpo_callback

        assert create_hpo_callback.__doc__ is not None


# =============================================================================
# EDGE CASES TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_report_every_equals_one(self, mock_optuna, mock_trial, mock_trainer):
        """Test report_every=1 reports every epoch."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss", report_every=1)
            metrics = {"val_loss": 0.5}

            for epoch in range(100):
                callback.on_epoch_end(mock_trainer, epoch=epoch, metrics=metrics)

            assert mock_trial.report.call_count == 100

    def test_report_every_large_value(self, mock_optuna, mock_trial, mock_trainer):
        """Test report_every with large value reports infrequently."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss", report_every=100)
            metrics = {"val_loss": 0.5}

            # 50 epochs, report_every=100, only epoch 0 should report
            for epoch in range(50):
                callback.on_epoch_end(mock_trainer, epoch=epoch, metrics=metrics)

            assert mock_trial.report.call_count == 1  # Only epoch 0

    def test_epoch_zero_always_reports(self, mock_optuna, mock_trial, mock_trainer):
        """Test epoch 0 always reports regardless of report_every."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(
                trial=mock_trial, monitor="val_loss", report_every=1000
            )
            metrics = {"val_loss": 0.5}

            callback.on_epoch_end(mock_trainer, epoch=0, metrics=metrics)

            mock_trial.report.assert_called_once()

    def test_metrics_with_float_values(self, mock_optuna, mock_trial, mock_trainer):
        """Test on_epoch_end handles various float values correctly."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_loss")

            test_values = [0.0, 1.0, -1.0, 1e-10, 1e10, 0.123456789]

            for i, val in enumerate(test_values):
                metrics = {"val_loss": val}
                callback.on_epoch_end(mock_trainer, epoch=i, metrics=metrics)
                assert mock_trial.report.call_args[0][0] == val

    def test_monitor_metric_with_special_characters(self, mock_optuna, mock_trial, mock_trainer):
        """Test monitoring metric with underscores and numbers."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="val_mae_2d")
            metrics = {"val_mae_2d": 0.123}

            callback.on_epoch_end(mock_trainer, epoch=0, metrics=metrics)

            mock_trial.report.assert_called_once_with(0.123, 0)

    def test_trial_with_high_number(self, mock_optuna, mock_trainer):
        """Test callback works with high trial numbers."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            trial = MagicMock()
            trial.number = 99999
            trial.should_prune.return_value = False

            callback = OptunaPruningCallback(trial=trial, monitor="val_loss")
            metrics = {"val_loss": 0.5}

            # Should not raise
            callback.on_epoch_end(mock_trainer, epoch=0, metrics=metrics)

    def test_callback_inheritance_from_base_callback(self, mock_optuna, mock_trial):
        """Test OptunaPruningCallback inherits from Callback."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                Callback,
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial)

            assert isinstance(callback, Callback)


# =============================================================================
# LOGGING TESTS
# =============================================================================


class TestLogging:
    """Test logging behavior throughout OptunaPruningCallback."""

    def test_logger_uses_correct_module_name(self):
        """Test logger is created with correct module name."""
        from milia_pipeline.models.hpo.callbacks.optuna_callback import logger

        expected_name = "milia_pipeline.models.hpo.callbacks.optuna_callback"
        assert logger.name == expected_name

    def test_init_logs_at_debug_level(self, mock_optuna, mock_trial):
        """Test __init__ logs at DEBUG level."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.logger") as mock_logger,
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial)

            mock_logger.debug.assert_called()
            mock_logger.info.assert_not_called()
            mock_logger.warning.assert_not_called()

    def test_on_train_begin_logs_at_debug_level(self, mock_optuna, mock_trial, mock_trainer):
        """Test on_train_begin logs at DEBUG level."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.logger") as mock_logger,
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial)
            mock_logger.reset_mock()

            callback.on_train_begin(mock_trainer)

            mock_logger.debug.assert_called()

    def test_metric_not_found_logs_at_warning_level(self, mock_optuna, mock_trial, mock_trainer):
        """Test missing metric logs at WARNING level."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.logger") as mock_logger,
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="nonexistent_metric")
            mock_logger.reset_mock()

            callback.on_epoch_end(mock_trainer, epoch=0, metrics={"other_metric": 0.5})

            mock_logger.warning.assert_called()

    def test_pruning_logs_at_info_level(self, mock_optuna, mock_trial_pruned, mock_trainer):
        """Test pruning event logs at INFO level."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.logger") as mock_logger,
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial_pruned, monitor="val_loss")
            mock_logger.reset_mock()

            with pytest.raises(mock_optuna.TrialPruned):
                callback.on_epoch_end(mock_trainer, epoch=0, metrics={"val_loss": 0.5})

            mock_logger.info.assert_called()

    def test_warning_log_includes_available_metrics(self, mock_optuna, mock_trial, mock_trainer):
        """Test warning log includes list of available metrics."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.logger") as mock_logger,
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial, monitor="missing")
            mock_logger.reset_mock()

            metrics = {"train_loss": 0.5, "val_mae": 0.1}
            callback.on_epoch_end(mock_trainer, epoch=0, metrics=metrics)

            warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
            # Check both metric names appear in warning
            assert any("train_loss" in str(c) for c in warning_calls)
            assert any("val_mae" in str(c) for c in warning_calls)


# =============================================================================
# PROTOCOL COMPLIANCE TESTS
# =============================================================================


class TestProtocolCompliance:
    """Test OptunaPruningCallback complies with Callback protocol."""

    def test_has_set_trainer_method(self, mock_optuna, mock_trial):
        """Test OptunaPruningCallback has set_trainer method."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial)

            assert hasattr(callback, "set_trainer")
            assert callable(callback.set_trainer)

    def test_has_on_train_begin_method(self, mock_optuna, mock_trial):
        """Test OptunaPruningCallback has on_train_begin method."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial)

            assert hasattr(callback, "on_train_begin")
            assert callable(callback.on_train_begin)

    def test_has_on_epoch_end_method(self, mock_optuna, mock_trial):
        """Test OptunaPruningCallback has on_epoch_end method."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial)

            assert hasattr(callback, "on_epoch_end")
            assert callable(callback.on_epoch_end)

    def test_has_on_train_end_method(self, mock_optuna, mock_trial):
        """Test OptunaPruningCallback has on_train_end method."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial)

            assert hasattr(callback, "on_train_end")
            assert callable(callback.on_train_end)

    def test_has_should_stop_method(self, mock_optuna, mock_trial):
        """Test OptunaPruningCallback has should_stop method."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial)

            assert hasattr(callback, "should_stop")
            # Verify it's callable (a method)
            assert callable(callback.should_stop)
            # Verify calling it returns a bool
            assert isinstance(callback.should_stop(), bool)


# =============================================================================
# EXCEPTION HANDLING TESTS
# =============================================================================


class TestExceptionHandling:
    """Test exception handling in OptunaPruningCallback."""

    def test_pruning_error_import(self):
        """Test PruningError can be imported from exceptions."""
        from milia_pipeline.exceptions import PruningError

        assert PruningError is not None

    def test_pruning_error_inherits_from_hpo_error(self):
        """Test PruningError inherits from HPOError."""
        from milia_pipeline.exceptions import HPOError, PruningError

        assert issubclass(PruningError, HPOError)

    def test_trial_pruned_exception_message_format(
        self, mock_optuna, mock_trial_pruned, mock_trainer
    ):
        """Test TrialPruned exception has correct message format."""
        with (
            patch.dict("sys.modules", {"optuna": mock_optuna}),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.OPTUNA_AVAILABLE", True),
            patch("milia_pipeline.models.hpo.callbacks.optuna_callback.optuna", mock_optuna),
        ):
            from milia_pipeline.models.hpo.callbacks.optuna_callback import (
                OptunaPruningCallback,
            )

            callback = OptunaPruningCallback(trial=mock_trial_pruned, monitor="val_loss")
            metrics = {"val_loss": 0.789}

            with pytest.raises(mock_optuna.TrialPruned) as exc_info:
                callback.on_epoch_end(mock_trainer, epoch=5, metrics=metrics)

            error_msg = str(exc_info.value)
            assert "epoch 5" in error_msg
            assert "val_loss" in error_msg
            assert "0.789" in error_msg


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
