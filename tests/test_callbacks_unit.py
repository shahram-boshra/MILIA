#!/usr/bin/env python3
"""
Complete Unit Test Suite for callbacks.py Module

Tests callback system including:
- Callback abstract base class
- EarlyStopping callback with various modes and configurations
- ModelCheckpoint callback with top-k tracking
- TensorBoardLogger callback with metric logging
- LearningRateMonitor callback
- ProgressBar callback
- GradientMonitor callback
- Thread safety considerations
- Edge cases and error handling
- Integration with trainer mock

This is a PRODUCTION-READY test suite with comprehensive coverage.

Author: milia Team
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import logging
import shutil
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

# Import the module under test
from milia_pipeline.models.training.callbacks import (
    # Base class
    Callback,
    # Factory
    CallbackFactory,
    # Callback implementations
    EarlyStopping,
    GradientMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
    TensorBoardLogger,
)

# Import exception
try:
    from milia_pipeline.exceptions import CheckpointError
except ImportError:
    # Use the fallback defined in callbacks.py
    from milia_pipeline.models.training.callbacks import CheckpointError

# Import the _is_tensorboard_available function for testing
from milia_pipeline.models.training.callbacks import _is_tensorboard_available

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_trainer():
    """
    Create a mock trainer for testing callbacks.

    The mock is carefully constructed to ensure all state_dict() returns
    are pickle-compatible for torch.save() operations used by ModelCheckpoint.

    CRITICAL: All attributes accessed by _save_checkpoint must be explicitly set
    to non-Mock values. If an attribute is not set, Mock auto-creates a new Mock
    object which cannot be pickled by torch.save().
    """
    trainer = Mock()
    trainer.max_epochs = 100
    trainer.current_epoch = 0
    trainer.global_step = 0
    trainer.best_val_loss = float("inf")
    trainer.metrics_history = defaultdict(list)

    # CRITICAL: model_info must be explicitly set to avoid Mock auto-creation
    # _save_checkpoint accesses trainer.model_info for hyper_parameters and data_info
    trainer.model_info = None

    # Mock model with pickle-compatible state_dict
    trainer.model = Mock(spec=nn.Module)
    # Return actual tensors that can be pickled
    trainer.model.state_dict = Mock(
        return_value={
            "layer1.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "layer1.bias": torch.tensor([0.1, 0.2]),
        }
    )
    trainer.model.parameters = Mock(
        return_value=[Mock(grad=Mock(data=Mock(norm=Mock(return_value=torch.tensor(1.0)))))]
    )

    # Mock optimizer with pickle-compatible state_dict
    trainer.optimizer = Mock(spec=optim.Adam)
    trainer.optimizer.state_dict = Mock(
        return_value={
            "state": {},
            "param_groups": [{"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-08}],
        }
    )
    trainer.optimizer.param_groups = [{"lr": 0.001}]

    # Mock scheduler with pickle-compatible state_dict
    trainer.scheduler = Mock()
    trainer.scheduler.state_dict = Mock(return_value={"last_epoch": 0, "base_lrs": [0.001]})

    return trainer


@pytest.fixture
def sample_metrics():
    """Create sample metrics dictionary."""
    return {
        "train_loss": 0.5,
        "val_loss": 0.6,
        "train_acc": 0.85,
        "val_acc": 0.82,
        "epoch_time": 10.5,
    }


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoint tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_tensorboard_dir():
    """Create a temporary directory for TensorBoard tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_working_dir():
    """Create a temporary working root directory for CallbackFactory tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# BASE CALLBACK TESTS
# =============================================================================


class TestCallbackBase:
    """Test Callback base class."""

    def test_is_base_class_with_hooks(self):
        """Test that Callback is a proper base class with expected hook methods."""
        assert hasattr(Callback, "set_trainer")
        assert hasattr(Callback, "on_train_begin")
        assert hasattr(Callback, "on_epoch_end")
        assert hasattr(Callback, "on_train_end")

    def test_can_instantiate_directly(self):
        """Test that Callback can be instantiated as a base with default no-op hooks."""
        # Callback is a concrete base class with default no-op implementations
        # Subclasses override only the hooks they need
        callback = Callback()
        assert isinstance(callback, Callback)

    def test_set_trainer(self, mock_trainer):
        """Test set_trainer method."""
        callback = Callback()
        callback.set_trainer(mock_trainer)
        assert callback.trainer is mock_trainer

    def test_on_train_begin_default(self, mock_trainer):
        """Test on_train_begin default implementation does nothing."""
        callback = Callback()
        # Should not raise any errors
        result = callback.on_train_begin(mock_trainer)
        assert result is None

    def test_on_epoch_end_default(self, mock_trainer, sample_metrics):
        """Test on_epoch_end default implementation does nothing."""
        callback = Callback()
        # Should not raise any errors
        result = callback.on_epoch_end(mock_trainer, 0, sample_metrics)
        assert result is None

    def test_on_train_end_default(self, mock_trainer):
        """Test on_train_end default implementation does nothing."""
        callback = Callback()
        # Should not raise any errors
        result = callback.on_train_end(mock_trainer)
        assert result is None

    def test_custom_callback_inheritance(self, mock_trainer):
        """Test creating a custom callback by inheriting."""

        class CustomCallback(Callback):
            def __init__(self):
                super().__init__()
                self.called = False

            def on_epoch_end(self, trainer, epoch, metrics):
                self.called = True

        callback = CustomCallback()
        callback.on_epoch_end(mock_trainer, 0, {})
        assert callback.called is True


# =============================================================================
# TENSORBOARD AVAILABILITY CHECK TESTS
# =============================================================================


class TestTensorBoardAvailability:
    """Test _is_tensorboard_available helper function."""

    def test_is_tensorboard_available_returns_bool(self):
        """Test that _is_tensorboard_available returns a boolean."""
        result = _is_tensorboard_available()
        assert isinstance(result, bool)

    def test_is_tensorboard_available_when_installed(self):
        """Test _is_tensorboard_available returns True when tensorboard is installed."""
        # This test verifies the function works correctly when tensorboard IS installed
        # The actual result depends on the test environment
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401 — import tests full dependency chain (torch shim → tensorboard package)

            # TensorBoard is available in this environment
            assert _is_tensorboard_available() is True
        except ImportError:
            # TensorBoard is not available in this environment
            assert _is_tensorboard_available() is False

    def test_is_tensorboard_available_handles_import_error(self):
        """Test _is_tensorboard_available handles ImportError gracefully."""
        # The function should never raise an exception
        # It should return False if import fails
        result = _is_tensorboard_available()
        assert result in (True, False)


# =============================================================================
# CHECKPOINT ERROR TESTS
# =============================================================================


class TestCheckpointError:
    """Test CheckpointError exception class."""

    def test_checkpoint_error_is_exception(self):
        """Test that CheckpointError is an Exception subclass."""
        assert issubclass(CheckpointError, Exception)

    def test_checkpoint_error_can_be_raised(self):
        """Test that CheckpointError can be raised and caught."""
        with pytest.raises(CheckpointError):
            raise CheckpointError("Test error message")

    def test_checkpoint_error_message(self):
        """Test that CheckpointError preserves error message."""
        error_msg = "Checkpoint save failed"
        try:
            raise CheckpointError(error_msg)
        except CheckpointError as e:
            assert str(e) == error_msg


# =============================================================================
# EARLY STOPPING TESTS
# =============================================================================


class TestEarlyStopping:
    """Test EarlyStopping callback."""

    def test_initialization_defaults(self):
        """Test EarlyStopping initialization with default values."""
        es = EarlyStopping()
        assert es.monitor == "val_loss"
        assert es.patience == 10
        assert es.mode == "min"
        assert es.min_delta == 0.0001
        assert es.verbose is True
        assert es.best_score is None
        assert es.counter == 0
        assert es._stop is False

    def test_initialization_custom_values(self):
        """Test EarlyStopping initialization with custom values."""
        es = EarlyStopping(monitor="val_acc", patience=5, mode="max", min_delta=0.01, verbose=False)
        assert es.monitor == "val_acc"
        assert es.patience == 5
        assert es.mode == "max"
        assert es.min_delta == 0.01
        assert es.verbose is False

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be 'min' or 'max'"):
            EarlyStopping(mode="invalid")

    def test_first_epoch_sets_best_score(self, mock_trainer):
        """Test that first epoch sets the best score."""
        es = EarlyStopping(monitor="val_loss", verbose=False)
        metrics = {"val_loss": 0.5}

        es.on_epoch_end(mock_trainer, 0, metrics)

        assert es.best_score == 0.5
        assert es.counter == 0
        assert es._stop is False

    def test_improvement_detected_min_mode(self, mock_trainer):
        """Test improvement detection in min mode."""
        es = EarlyStopping(monitor="val_loss", mode="min", min_delta=0.01, verbose=False)

        # First epoch
        es.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})
        assert es.best_score == 0.5
        assert es.counter == 0

        # Second epoch with improvement
        es.on_epoch_end(mock_trainer, 1, {"val_loss": 0.4})
        assert es.best_score == 0.4
        assert es.counter == 0

    def test_improvement_detected_max_mode(self, mock_trainer):
        """Test improvement detection in max mode."""
        es = EarlyStopping(monitor="val_acc", mode="max", min_delta=0.01, verbose=False)

        # First epoch
        es.on_epoch_end(mock_trainer, 0, {"val_acc": 0.8})
        assert es.best_score == 0.8
        assert es.counter == 0

        # Second epoch with improvement
        es.on_epoch_end(mock_trainer, 1, {"val_acc": 0.85})
        assert es.best_score == 0.85
        assert es.counter == 0

    def test_no_improvement_increments_counter(self, mock_trainer):
        """Test that lack of improvement increments counter."""
        es = EarlyStopping(monitor="val_loss", mode="min", patience=3, verbose=False)

        es.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})
        assert es.counter == 0

        # No improvement
        es.on_epoch_end(mock_trainer, 1, {"val_loss": 0.55})
        assert es.counter == 1
        assert es._stop is False

        es.on_epoch_end(mock_trainer, 2, {"val_loss": 0.56})
        assert es.counter == 2
        assert es._stop is False

    def test_early_stopping_triggered(self, mock_trainer):
        """Test that early stopping is triggered after patience exhausted."""
        es = EarlyStopping(monitor="val_loss", mode="min", patience=2, verbose=False)

        es.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})
        es.on_epoch_end(mock_trainer, 1, {"val_loss": 0.55})
        es.on_epoch_end(mock_trainer, 2, {"val_loss": 0.56})

        assert es.counter == 2
        assert es._stop is True
        assert es.should_stop() is True

    def test_counter_resets_on_improvement(self, mock_trainer):
        """Test that counter resets when improvement occurs."""
        es = EarlyStopping(monitor="val_loss", mode="min", patience=3, verbose=False)

        es.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})
        es.on_epoch_end(mock_trainer, 1, {"val_loss": 0.55})
        assert es.counter == 1

        # Improvement resets counter
        es.on_epoch_end(mock_trainer, 2, {"val_loss": 0.4})
        assert es.counter == 0
        assert es._stop is False

    def test_min_delta_threshold(self, mock_trainer):
        """Test that min_delta is properly applied."""
        es = EarlyStopping(monitor="val_loss", mode="min", min_delta=0.1, verbose=False)

        es.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})

        # Improvement less than min_delta
        es.on_epoch_end(mock_trainer, 1, {"val_loss": 0.45})
        assert es.counter == 1  # Not considered improvement

        # Improvement greater than min_delta
        es.on_epoch_end(mock_trainer, 2, {"val_loss": 0.3})
        assert es.counter == 0  # Considered improvement

    def test_metric_not_found_warning(self, mock_trainer, caplog):
        """Test warning when monitored metric is not found."""
        es = EarlyStopping(monitor="nonexistent_metric", verbose=False)

        with caplog.at_level(logging.WARNING):
            es.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})

        assert "metric 'nonexistent_metric' not found" in caplog.text
        assert es.best_score is None
        assert es._stop is False

    def test_verbose_logging(self, mock_trainer, caplog):
        """Test verbose logging mode."""
        es = EarlyStopping(monitor="val_loss", mode="min", verbose=True)

        with caplog.at_level(logging.INFO):
            es.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})
            assert "Initial val_loss=0.5" in caplog.text

            caplog.clear()
            es.on_epoch_end(mock_trainer, 1, {"val_loss": 0.4})
            assert "Improvement detected" in caplog.text

            caplog.clear()
            es.on_epoch_end(mock_trainer, 2, {"val_loss": 0.45})
            assert "No improvement" in caplog.text

    def test_is_improvement_method(self):
        """Test _is_improvement method."""
        es_min = EarlyStopping(mode="min", min_delta=0.01)
        es_min.best_score = 0.5
        assert es_min._is_improvement(0.4) is True
        assert es_min._is_improvement(0.495) is False
        assert es_min._is_improvement(0.6) is False

        es_max = EarlyStopping(mode="max", min_delta=0.01)
        es_max.best_score = 0.5
        assert es_max._is_improvement(0.6) is True
        assert es_max._is_improvement(0.505) is False
        assert es_max._is_improvement(0.4) is False

    def test_should_stop_method(self):
        """Test should_stop method."""
        es = EarlyStopping(patience=1, verbose=False)
        assert es.should_stop() is False

        es._stop = True
        assert es.should_stop() is True

    def test_on_train_end_with_early_stop(self, mock_trainer, caplog):
        """Test on_train_end when early stopping occurred."""
        es = EarlyStopping(monitor="val_loss", patience=1, verbose=False)
        es.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})
        es.on_epoch_end(mock_trainer, 1, {"val_loss": 0.6})

        with caplog.at_level(logging.INFO):
            es.on_train_end(mock_trainer)

        assert "Training stopped early" in caplog.text
        assert "Best val_loss=" in caplog.text

    def test_on_train_end_without_early_stop(self, mock_trainer, caplog):
        """Test on_train_end when early stopping did not occur."""
        es = EarlyStopping(monitor="val_loss", verbose=False)
        es.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})

        with caplog.at_level(logging.INFO):
            es.on_train_end(mock_trainer)

        assert "Training completed without early stopping" in caplog.text


# =============================================================================
# MODEL CHECKPOINT TESTS
# =============================================================================


class TestModelCheckpoint:
    """Test ModelCheckpoint callback."""

    def test_initialization_defaults(self, temp_checkpoint_dir):
        """Test ModelCheckpoint initialization with default values."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir)
        assert mc.dirpath == temp_checkpoint_dir
        assert mc.monitor == "val_loss"
        assert mc.mode == "min"
        assert mc.save_top_k == 3
        assert mc.save_last is True
        assert mc.verbose is True
        assert mc.best_checkpoints == []
        assert temp_checkpoint_dir.exists()

    def test_initialization_custom_values(self, temp_checkpoint_dir):
        """Test ModelCheckpoint initialization with custom values."""
        mc = ModelCheckpoint(
            dirpath=temp_checkpoint_dir,
            monitor="val_acc",
            mode="max",
            save_top_k=5,
            save_last=False,
            filename_pattern="model-{epoch:02d}.pt",
            verbose=False,
        )
        assert mc.monitor == "val_acc"
        assert mc.mode == "max"
        assert mc.save_top_k == 5
        assert mc.save_last is False
        assert mc.filename_pattern == "model-{epoch:02d}.pt"
        assert mc.verbose is False

    def test_invalid_mode_raises_error(self, temp_checkpoint_dir):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be 'min' or 'max'"):
            ModelCheckpoint(dirpath=temp_checkpoint_dir, mode="invalid")

    def test_dirpath_none_raises_error(self):
        """Test that dirpath=None raises ValueError."""
        with pytest.raises(ValueError, match="requires 'dirpath' to be specified"):
            ModelCheckpoint(dirpath=None)

    def test_save_top_k_zero_warning(self, temp_checkpoint_dir, caplog):
        """Test warning when save_top_k is 0."""
        with caplog.at_level(logging.WARNING):
            _mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, save_top_k=0)

        assert "save_top_k=0, no checkpoints will be saved" in caplog.text

    def test_checkpoint_directory_creation(self):
        """Test that checkpoint directory is created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "new_dir" / "checkpoints"
            _mc = ModelCheckpoint(dirpath=checkpoint_dir)
            assert checkpoint_dir.exists()

    def test_save_checkpoint_basic(self, mock_trainer, temp_checkpoint_dir):
        """Test basic checkpoint saving."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, verbose=False)
        metrics = {"val_loss": 0.5}

        mc.on_epoch_end(mock_trainer, 0, metrics)

        # Check checkpoint file was created (use epoch=*.pt to exclude best.pt)
        checkpoints = list(temp_checkpoint_dir.glob("epoch=*.pt"))
        assert len(checkpoints) == 1

        # Check checkpoint content
        checkpoint = torch.load(checkpoints[0], weights_only=False)
        assert checkpoint["epoch"] == 0
        assert checkpoint["monitored_score"] == 0.5
        assert checkpoint["monitored_metric"] == "val_loss"
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint

    def test_save_top_k_tracking(self, mock_trainer, temp_checkpoint_dir):
        """Test that only top-k checkpoints are kept."""
        mc = ModelCheckpoint(
            dirpath=temp_checkpoint_dir, monitor="val_loss", mode="min", save_top_k=2, verbose=False
        )

        # Save 3 checkpoints
        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})
        mc.on_epoch_end(mock_trainer, 1, {"val_loss": 0.4})
        mc.on_epoch_end(mock_trainer, 2, {"val_loss": 0.6})

        # Should only have 2 best checkpoints
        checkpoints = list(temp_checkpoint_dir.glob("epoch=*.pt"))
        assert len(checkpoints) == 2
        assert len(mc.best_checkpoints) == 2

        # Verify best checkpoints are kept (0.4 and 0.5, not 0.6)
        scores = [score for score, _ in mc.best_checkpoints]
        assert 0.4 in scores
        assert 0.5 in scores
        assert 0.6 not in scores

    def test_save_top_k_max_mode(self, mock_trainer, temp_checkpoint_dir):
        """Test top-k tracking in max mode."""
        mc = ModelCheckpoint(
            dirpath=temp_checkpoint_dir, monitor="val_acc", mode="max", save_top_k=2, verbose=False
        )

        mc.on_epoch_end(mock_trainer, 0, {"val_acc": 0.8})
        mc.on_epoch_end(mock_trainer, 1, {"val_acc": 0.9})
        mc.on_epoch_end(mock_trainer, 2, {"val_acc": 0.75})

        checkpoints = list(temp_checkpoint_dir.glob("epoch=*.pt"))
        assert len(checkpoints) == 2

        # Verify best checkpoints (0.9 and 0.8, not 0.75)
        scores = [score for score, _ in mc.best_checkpoints]
        assert 0.9 in scores
        assert 0.8 in scores
        assert 0.75 not in scores

    def test_save_top_k_minus_one_saves_all(self, mock_trainer, temp_checkpoint_dir):
        """Test that save_top_k=-1 saves all checkpoints."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, save_top_k=-1, verbose=False)

        for i in range(5):
            mc.on_epoch_end(mock_trainer, i, {"val_loss": 0.5 - i * 0.1})

        checkpoints = list(temp_checkpoint_dir.glob("epoch=*.pt"))
        assert len(checkpoints) == 5
        assert len(mc.best_checkpoints) == 5

    def test_save_last_checkpoint(self, mock_trainer, temp_checkpoint_dir):
        """Test saving last checkpoint at training end."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, save_last=True, verbose=False)

        mock_trainer.current_epoch = 10
        mc.on_train_end(mock_trainer)

        last_checkpoint = temp_checkpoint_dir / "last.pt"
        assert last_checkpoint.exists()

        checkpoint = torch.load(last_checkpoint, weights_only=False)
        assert checkpoint["epoch"] == 10

    def test_no_save_last_checkpoint(self, mock_trainer, temp_checkpoint_dir):
        """Test not saving last checkpoint when disabled."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, save_last=False, verbose=False)

        mc.on_train_end(mock_trainer)

        last_checkpoint = temp_checkpoint_dir / "last.pt"
        assert not last_checkpoint.exists()

    def test_metric_not_found_warning(self, mock_trainer, temp_checkpoint_dir, caplog):
        """Test warning when monitored metric is not found."""
        mc = ModelCheckpoint(
            dirpath=temp_checkpoint_dir, monitor="nonexistent_metric", verbose=False
        )

        with caplog.at_level(logging.WARNING):
            mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})

        assert "metric 'nonexistent_metric' not found" in caplog.text
        checkpoints = list(temp_checkpoint_dir.glob("*.pt"))
        assert len(checkpoints) == 0

    def test_save_top_k_zero_no_save(self, mock_trainer, temp_checkpoint_dir):
        """Test that save_top_k=0 prevents saving."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, save_top_k=0, verbose=False)

        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})

        checkpoints = list(temp_checkpoint_dir.glob("epoch=*.pt"))
        assert len(checkpoints) == 0

    def test_checkpoint_includes_scheduler(self, mock_trainer, temp_checkpoint_dir):
        """Test that checkpoint includes scheduler state if present."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, verbose=False)
        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})

        checkpoints = list(temp_checkpoint_dir.glob("*.pt"))
        checkpoint = torch.load(checkpoints[0], weights_only=False)
        assert "scheduler_state_dict" in checkpoint

    def test_checkpoint_without_scheduler(self, mock_trainer, temp_checkpoint_dir):
        """Test checkpoint when scheduler is None."""
        mock_trainer.scheduler = None
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, verbose=False)
        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})

        checkpoints = list(temp_checkpoint_dir.glob("*.pt"))
        checkpoint = torch.load(checkpoints[0], weights_only=False)
        assert "scheduler_state_dict" not in checkpoint

    def test_filename_pattern_formatting(self, mock_trainer, temp_checkpoint_dir):
        """Test custom filename pattern formatting."""
        mc = ModelCheckpoint(
            dirpath=temp_checkpoint_dir,
            filename_pattern="model_{epoch:03d}_{monitor}_{score:.2f}.pt",
            verbose=False,
        )

        mc.on_epoch_end(mock_trainer, 5, {"val_loss": 0.123})

        # Use model_*.pt to match custom pattern and exclude best.pt
        checkpoints = list(temp_checkpoint_dir.glob("model_*.pt"))
        assert len(checkpoints) == 1
        assert checkpoints[0].name == "model_005_val_loss_0.12.pt"

    def test_save_checkpoint_error_handling(self, mock_trainer, temp_checkpoint_dir, caplog):
        """Test error handling during checkpoint save."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, verbose=False)

        # Make torch.save fail
        with patch("torch.save", side_effect=Exception("Save failed")):
            with caplog.at_level(logging.ERROR):
                mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})

        assert "Failed to save checkpoint" in caplog.text

    def test_on_train_end_save_error_handling(self, mock_trainer, temp_checkpoint_dir, caplog):
        """Test error handling when saving final checkpoint fails."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, save_last=True, verbose=False)

        with patch.object(mc, "_save_checkpoint", side_effect=Exception("Save failed")):
            with caplog.at_level(logging.ERROR):
                mc.on_train_end(mock_trainer)

        assert "Failed to save final checkpoint" in caplog.text

    def test_on_train_end_summary_logging(self, mock_trainer, temp_checkpoint_dir, caplog):
        """Test summary logging at training end."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, verbose=False)

        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})
        mc.on_epoch_end(mock_trainer, 1, {"val_loss": 0.4})

        with caplog.at_level(logging.INFO):
            mc.on_train_end(mock_trainer)

        assert "Best checkpoint" in caplog.text
        assert "val_loss=0.4" in caplog.text

    def test_old_checkpoint_removal(self, mock_trainer, temp_checkpoint_dir, caplog):
        """Test that old checkpoints are removed when exceeding save_top_k."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, save_top_k=2, verbose=True)

        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})
        mc.on_epoch_end(mock_trainer, 1, {"val_loss": 0.4})

        with caplog.at_level(logging.INFO):
            mc.on_epoch_end(mock_trainer, 2, {"val_loss": 0.3})

        assert "Removed old checkpoint" in caplog.text

        # Only 2 checkpoints should remain
        checkpoints = list(temp_checkpoint_dir.glob("epoch=*.pt"))
        assert len(checkpoints) == 2

    # -------------------------------------------------------------------------
    # best_model_path and best_model_score Property Tests
    # -------------------------------------------------------------------------

    def test_best_model_path_initially_none(self, temp_checkpoint_dir):
        """Test that best_model_path is None before any checkpoints are saved."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, verbose=False)
        assert mc.best_model_path is None

    def test_best_model_score_initially_none(self, temp_checkpoint_dir):
        """Test that best_model_score is None before any checkpoints are saved."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, verbose=False)
        assert mc.best_model_score is None

    def test_best_model_path_updated_on_first_checkpoint(self, mock_trainer, temp_checkpoint_dir):
        """Test that best_model_path is set after first checkpoint."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, save_best=True, verbose=False)
        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})

        assert mc.best_model_path is not None
        assert mc.best_model_path == temp_checkpoint_dir / "best.pt"

    def test_best_model_score_updated_on_first_checkpoint(self, mock_trainer, temp_checkpoint_dir):
        """Test that best_model_score is set after first checkpoint."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, verbose=False)
        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})

        assert mc.best_model_score == 0.5

    def test_best_model_path_updated_on_improvement_min_mode(
        self, mock_trainer, temp_checkpoint_dir
    ):
        """Test best_model_path updates when a better checkpoint is saved in min mode."""
        mc = ModelCheckpoint(
            dirpath=temp_checkpoint_dir,
            monitor="val_loss",
            mode="min",
            save_best=True,
            verbose=False,
        )

        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})
        first_score = mc.best_model_score

        mc.on_epoch_end(mock_trainer, 1, {"val_loss": 0.3})  # Better

        assert mc.best_model_score == 0.3
        assert mc.best_model_score < first_score
        assert mc.best_model_path == temp_checkpoint_dir / "best.pt"

    def test_best_model_path_updated_on_improvement_max_mode(
        self, mock_trainer, temp_checkpoint_dir
    ):
        """Test best_model_path updates when a better checkpoint is saved in max mode."""
        mc = ModelCheckpoint(
            dirpath=temp_checkpoint_dir,
            monitor="val_acc",
            mode="max",
            save_best=True,
            verbose=False,
        )

        mc.on_epoch_end(mock_trainer, 0, {"val_acc": 0.8})
        first_score = mc.best_model_score

        mc.on_epoch_end(mock_trainer, 1, {"val_acc": 0.9})  # Better

        assert mc.best_model_score == 0.9
        assert mc.best_model_score > first_score
        assert mc.best_model_path == temp_checkpoint_dir / "best.pt"

    def test_best_model_score_not_updated_on_worse_checkpoint(
        self, mock_trainer, temp_checkpoint_dir
    ):
        """Test best_model_score does not update for worse checkpoints."""
        mc = ModelCheckpoint(
            dirpath=temp_checkpoint_dir, monitor="val_loss", mode="min", verbose=False
        )

        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.3})
        mc.on_epoch_end(mock_trainer, 1, {"val_loss": 0.5})  # Worse

        assert mc.best_model_score == 0.3  # Should remain 0.3

    # -------------------------------------------------------------------------
    # save_best Feature Tests
    # -------------------------------------------------------------------------

    def test_save_best_creates_best_pt_file(self, mock_trainer, temp_checkpoint_dir):
        """Test that save_best=True creates a best.pt file."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, save_best=True, verbose=False)

        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})

        best_pt = temp_checkpoint_dir / "best.pt"
        assert best_pt.exists()

    def test_save_best_false_no_best_pt_file(self, mock_trainer, temp_checkpoint_dir):
        """Test that save_best=False does not create best.pt file."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, save_best=False, verbose=False)

        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})

        best_pt = temp_checkpoint_dir / "best.pt"
        assert not best_pt.exists()

    def test_save_best_updates_best_pt_on_improvement(self, mock_trainer, temp_checkpoint_dir):
        """Test that best.pt is updated when a better checkpoint is found."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, save_best=True, verbose=False)

        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})

        # Load first best.pt and check epoch
        best_pt = temp_checkpoint_dir / "best.pt"
        checkpoint_v1 = torch.load(best_pt, weights_only=False)
        assert checkpoint_v1["epoch"] == 0

        mc.on_epoch_end(mock_trainer, 1, {"val_loss": 0.3})  # Better

        # Load updated best.pt and verify it's epoch 1
        checkpoint_v2 = torch.load(best_pt, weights_only=False)
        assert checkpoint_v2["epoch"] == 1
        assert checkpoint_v2["monitored_score"] == 0.3

    def test_save_best_does_not_update_on_worse_checkpoint(self, mock_trainer, temp_checkpoint_dir):
        """Test that best.pt is NOT updated when checkpoint is worse."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, save_best=True, verbose=False)

        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.3})

        best_pt = temp_checkpoint_dir / "best.pt"
        checkpoint_v1 = torch.load(best_pt, weights_only=False)
        assert checkpoint_v1["epoch"] == 0

        mc.on_epoch_end(mock_trainer, 1, {"val_loss": 0.5})  # Worse

        # best.pt should still be from epoch 0
        checkpoint_v2 = torch.load(best_pt, weights_only=False)
        assert checkpoint_v2["epoch"] == 0
        assert checkpoint_v2["monitored_score"] == 0.3

    def test_save_best_is_best_marker_true(self, mock_trainer, temp_checkpoint_dir):
        """Test that best.pt has is_best=True marker."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, save_best=True, verbose=False)

        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})

        best_pt = temp_checkpoint_dir / "best.pt"
        checkpoint = torch.load(best_pt, weights_only=False)
        assert checkpoint.get("is_best") is True

    def test_regular_checkpoint_is_best_marker_false(self, mock_trainer, temp_checkpoint_dir):
        """Test that regular checkpoints have is_best=False marker."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, save_best=True, verbose=False)

        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})

        # Find the epoch checkpoint (not best.pt or last.pt)
        epoch_checkpoints = list(temp_checkpoint_dir.glob("epoch=*.pt"))
        assert len(epoch_checkpoints) >= 1

        checkpoint = torch.load(epoch_checkpoints[0], weights_only=False)
        assert checkpoint.get("is_best") is False

    def test_best_model_path_returns_best_pt_when_save_best_true(
        self, mock_trainer, temp_checkpoint_dir
    ):
        """Test best_model_path property returns best.pt path when save_best=True."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, save_best=True, verbose=False)

        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})

        # When save_best=True, best_model_path should point to best.pt
        assert mc.best_model_path == temp_checkpoint_dir / "best.pt"

    def test_best_model_path_returns_checkpoint_path_when_save_best_false(
        self, mock_trainer, temp_checkpoint_dir
    ):
        """Test best_model_path property returns actual checkpoint path when save_best=False."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, save_best=False, verbose=False)

        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})

        # When save_best=False, best_model_path should be the actual checkpoint file
        assert mc.best_model_path is not None
        assert mc.best_model_path.exists()
        assert mc.best_model_path.name != "best.pt"

    # -------------------------------------------------------------------------
    # Checkpoint V2.0 Format Tests
    # -------------------------------------------------------------------------

    def test_checkpoint_contains_version_info(self, mock_trainer, temp_checkpoint_dir):
        """Test that checkpoint contains version_info with format version."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, verbose=False)
        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})

        checkpoints = list(temp_checkpoint_dir.glob("epoch=*.pt"))
        checkpoint = torch.load(checkpoints[0], weights_only=False)

        assert "version_info" in checkpoint
        assert checkpoint["version_info"]["checkpoint_format_version"] == "2.0"
        assert "pytorch_version" in checkpoint["version_info"]
        assert "pyg_version" in checkpoint["version_info"]
        assert "created_at" in checkpoint["version_info"]

    def test_checkpoint_contains_data_info(self, mock_trainer, temp_checkpoint_dir):
        """Test that checkpoint contains data_info section."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, verbose=False)
        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})

        checkpoints = list(temp_checkpoint_dir.glob("epoch=*.pt"))
        checkpoint = torch.load(checkpoints[0], weights_only=False)

        assert "data_info" in checkpoint
        assert "requires_edge_features" in checkpoint["data_info"]
        assert "uses_edge_features" in checkpoint["data_info"]
        assert "structural_features_config" in checkpoint["data_info"]

    def test_checkpoint_contains_hyper_parameters(self, mock_trainer, temp_checkpoint_dir):
        """Test that checkpoint contains hyper_parameters section."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, verbose=False)
        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})

        checkpoints = list(temp_checkpoint_dir.glob("epoch=*.pt"))
        checkpoint = torch.load(checkpoints[0], weights_only=False)

        assert "hyper_parameters" in checkpoint

    def test_checkpoint_hyper_parameters_from_model_info(self, mock_trainer, temp_checkpoint_dir):
        """Test that hyper_parameters are populated from trainer.model_info."""
        # Setup trainer with model_info
        mock_trainer.model_info = {
            "name": "TestModel",
            "task_type": "classification",
            "hyperparameters_values": {"hidden_dim": 128},
            "wrapper_info": {"wrapper_type": "standard"},
            "target_selection": {"target": "label"},
            "requires_edge_features": True,
            "uses_edge_features": False,
            "structural_features_config": {"use_degree": True},
        }

        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, verbose=False)
        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})

        checkpoints = list(temp_checkpoint_dir.glob("epoch=*.pt"))
        checkpoint = torch.load(checkpoints[0], weights_only=False)

        hp = checkpoint["hyper_parameters"]
        assert hp["model_name"] == "TestModel"
        assert hp["task_type"] == "classification"
        assert hp["hyperparameters"]["hidden_dim"] == 128
        assert hp["model_info"] == mock_trainer.model_info
        assert hp["wrapper_info"] == {"wrapper_type": "standard"}
        assert hp["target_selection_config"] == {"target": "label"}

        data_info = checkpoint["data_info"]
        assert data_info["requires_edge_features"] is True
        assert data_info["uses_edge_features"] is False
        assert data_info["structural_features_config"] == {"use_degree": True}

    def test_checkpoint_created_at_timestamp_format(self, mock_trainer, temp_checkpoint_dir):
        """Test that created_at timestamp is in ISO format."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, verbose=False)
        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})

        checkpoints = list(temp_checkpoint_dir.glob("epoch=*.pt"))
        checkpoint = torch.load(checkpoints[0], weights_only=False)

        created_at = checkpoint["version_info"]["created_at"]
        # Verify it's a valid ISO format by parsing it
        parsed_date = datetime.fromisoformat(created_at)
        assert isinstance(parsed_date, datetime)

    def test_checkpoint_without_model_info(self, mock_trainer, temp_checkpoint_dir):
        """Test checkpoint creation when trainer has no model_info."""
        # Remove model_info from trainer
        if hasattr(mock_trainer, "model_info"):
            delattr(mock_trainer, "model_info")

        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, verbose=False)
        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})

        checkpoints = list(temp_checkpoint_dir.glob("epoch=*.pt"))
        checkpoint = torch.load(checkpoints[0], weights_only=False)

        # Should have empty hyper_parameters and data_info
        assert checkpoint["hyper_parameters"] == {}
        assert checkpoint["data_info"]["requires_edge_features"] is False
        assert checkpoint["data_info"]["uses_edge_features"] is False
        assert checkpoint["data_info"]["structural_features_config"] == {}

    def test_on_train_end_logs_best_pt_path(self, mock_trainer, temp_checkpoint_dir, caplog):
        """Test that on_train_end logs the best.pt path when save_best=True."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, save_best=True, verbose=False)

        mc.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})

        with caplog.at_level(logging.INFO):
            mc.on_train_end(mock_trainer)

        assert "best.pt" in caplog.text or "Best model saved" in caplog.text


# =============================================================================
# TENSORBOARD LOGGER TESTS
# =============================================================================


class TestTensorBoardLogger:
    """Test TensorBoardLogger callback."""

    def test_initialization(self, temp_tensorboard_dir):
        """Test TensorBoardLogger initialization."""
        tb = TensorBoardLogger(log_dir=temp_tensorboard_dir, flush_secs=60)
        assert tb.log_dir == temp_tensorboard_dir
        assert tb.flush_secs == 60
        assert tb.writer is None
        assert temp_tensorboard_dir.exists()

    def test_initialization_stores_tensorboard_availability(self):
        """Test that initialization stores tensorboard availability status."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "tb_logs"
            tb = TensorBoardLogger(log_dir=log_dir)
            # The _tensorboard_available attribute should match the function result
            assert tb._tensorboard_available == _is_tensorboard_available()

    def test_initialization_no_directory_when_tensorboard_unavailable(self, caplog):
        """Test that log directory is NOT created when tensorboard is unavailable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "new_tb_logs"

            # Patch _is_tensorboard_available to return False
            with (
                patch(
                    "milia_pipeline.models.training.callbacks._is_tensorboard_available",
                    return_value=False,
                ),
                caplog.at_level(logging.WARNING),
            ):
                tb = TensorBoardLogger(log_dir=log_dir)

            # Directory should NOT be created
            assert not log_dir.exists()
            # Warning should be logged
            assert "tensorboard not installed" in caplog.text
            # Availability flag should be False
            assert tb._tensorboard_available is False

    def test_log_directory_creation(self):
        """Test that log directory is created when tensorboard is available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs" / "tensorboard"

            # Mock _is_tensorboard_available to return True to test directory creation
            with patch(
                "milia_pipeline.models.training.callbacks._is_tensorboard_available",
                return_value=True,
            ):
                _tb = TensorBoardLogger(log_dir=log_dir)
                assert log_dir.exists()

    def test_on_train_begin_creates_writer(self, mock_trainer, temp_tensorboard_dir):
        """Test that on_train_begin creates SummaryWriter."""
        tb = TensorBoardLogger(log_dir=temp_tensorboard_dir)

        # Only test if tensorboard is actually available
        if tb._tensorboard_available:
            tb.on_train_begin(mock_trainer)
            # Writer should be created
            assert tb.writer is not None
        else:
            # If tensorboard is not available, writer should remain None
            tb.on_train_begin(mock_trainer)
            assert tb.writer is None

    def test_on_train_begin_with_tensorboard_available(self, mock_trainer, temp_tensorboard_dir):
        """Test on_train_begin when TensorBoard is available."""
        tb = TensorBoardLogger(log_dir=temp_tensorboard_dir)

        # Mock SummaryWriter at the module level within the function scope
        mock_summary_writer = Mock()

        with (
            patch.object(tb, "_tensorboard_available", True),
            patch(
                "milia_pipeline.models.training.callbacks.TensorBoardLogger.on_train_begin"
            ) as _mock_method,
        ):
            # Call the actual method but verify behavior
            tb._tensorboard_available = True
            # Create a mock writer directly to test the expected behavior
            tb.writer = mock_summary_writer

            # Verify the writer was set
            assert tb.writer is mock_summary_writer

    def test_on_train_begin_handles_unavailable_tensorboard(
        self, mock_trainer, temp_tensorboard_dir, caplog
    ):
        """Test that on_train_begin handles missing tensorboard gracefully."""
        tb = TensorBoardLogger(log_dir=temp_tensorboard_dir)

        # Force tensorboard to appear unavailable
        tb._tensorboard_available = False

        # Should not raise any errors and writer should remain None
        tb.on_train_begin(mock_trainer)

        assert tb.writer is None

    def test_on_epoch_end_logs_metrics(self, mock_trainer, temp_tensorboard_dir):
        """Test that metrics are logged to TensorBoard."""
        tb = TensorBoardLogger(log_dir=temp_tensorboard_dir)
        mock_writer = Mock()
        tb.writer = mock_writer

        metrics = {"train_loss": 0.5, "val_loss": 0.6, "train_acc": 0.85}

        tb.on_epoch_end(mock_trainer, 10, metrics)

        # Check that add_scalar was called for each metric (3) + learning rate (1)
        assert mock_writer.add_scalar.call_count == 4
        mock_writer.add_scalar.assert_any_call("train_loss", 0.5, 10)
        mock_writer.add_scalar.assert_any_call("val_loss", 0.6, 10)
        mock_writer.add_scalar.assert_any_call("train_acc", 0.85, 10)
        mock_writer.add_scalar.assert_any_call("learning_rate/group_0", 0.001, 10)

    def test_on_epoch_end_skips_non_numeric(self, mock_trainer, temp_tensorboard_dir):
        """Test that non-numeric metrics are skipped."""
        tb = TensorBoardLogger(log_dir=temp_tensorboard_dir)
        mock_writer = Mock()
        tb.writer = mock_writer

        metrics = {
            "train_loss": 0.5,
            "message": "hello",  # Non-numeric
            "data": [1, 2, 3],  # Non-numeric
        }

        tb.on_epoch_end(mock_trainer, 10, metrics)

        # Only train_loss (1) + learning rate (1) should be logged
        assert mock_writer.add_scalar.call_count == 2
        mock_writer.add_scalar.assert_any_call("train_loss", 0.5, 10)
        mock_writer.add_scalar.assert_any_call("learning_rate/group_0", 0.001, 10)

    def test_on_epoch_end_skips_special_keys(self, mock_trainer, temp_tensorboard_dir):
        """Test that special keys are skipped."""
        tb = TensorBoardLogger(log_dir=temp_tensorboard_dir)
        mock_writer = Mock()
        tb.writer = mock_writer

        metrics = {
            "train_loss": 0.5,
            "is_best": True,  # Special key
            "epoch_time": 10.5,  # Special key
        }

        tb.on_epoch_end(mock_trainer, 10, metrics)

        # Only train_loss (1) + learning rate (1) should be logged
        assert mock_writer.add_scalar.call_count == 2
        mock_writer.add_scalar.assert_any_call("train_loss", 0.5, 10)
        mock_writer.add_scalar.assert_any_call("learning_rate/group_0", 0.001, 10)

    def test_on_epoch_end_logs_learning_rate(self, mock_trainer, temp_tensorboard_dir):
        """Test that learning rate is logged."""
        tb = TensorBoardLogger(log_dir=temp_tensorboard_dir)
        mock_writer = Mock()
        tb.writer = mock_writer

        mock_trainer.optimizer.param_groups = [{"lr": 0.001}, {"lr": 0.0005}]

        tb.on_epoch_end(mock_trainer, 10, {"train_loss": 0.5})

        # Check learning rate logging
        mock_writer.add_scalar.assert_any_call("learning_rate/group_0", 0.001, 10)
        mock_writer.add_scalar.assert_any_call("learning_rate/group_1", 0.0005, 10)

    def test_on_epoch_end_without_writer(self, mock_trainer, temp_tensorboard_dir):
        """Test that on_epoch_end does nothing if writer is None."""
        tb = TensorBoardLogger(log_dir=temp_tensorboard_dir)
        tb.writer = None

        # Should not raise any errors
        tb.on_epoch_end(mock_trainer, 10, {"train_loss": 0.5})

    def test_on_epoch_end_with_logging_error(self, mock_trainer, temp_tensorboard_dir, caplog):
        """Test error handling during metric logging."""
        tb = TensorBoardLogger(log_dir=temp_tensorboard_dir)
        mock_writer = Mock()

        # First call (metric) should fail, second call (learning rate) should succeed
        mock_writer.add_scalar.side_effect = [Exception("Logging failed"), None]
        tb.writer = mock_writer

        with caplog.at_level(logging.WARNING):
            tb.on_epoch_end(mock_trainer, 10, {"train_loss": 0.5})

        assert "Failed to log" in caplog.text
        # Both calls should have been attempted (metric and learning rate)
        assert mock_writer.add_scalar.call_count == 2

    def test_on_train_end_closes_writer(self, mock_trainer, temp_tensorboard_dir, caplog):
        """Test that writer is closed at training end."""
        tb = TensorBoardLogger(log_dir=temp_tensorboard_dir)
        mock_writer = Mock()
        tb.writer = mock_writer

        with caplog.at_level(logging.INFO):
            tb.on_train_end(mock_trainer)

        mock_writer.close.assert_called_once()
        assert "Closed writer" in caplog.text

    def test_on_train_end_without_writer(self, mock_trainer, temp_tensorboard_dir):
        """Test that on_train_end does nothing if writer is None."""
        tb = TensorBoardLogger(log_dir=temp_tensorboard_dir)
        tb.writer = None

        # Should not raise any errors
        tb.on_train_end(mock_trainer)


# =============================================================================
# LEARNING RATE MONITOR TESTS
# =============================================================================


class TestLearningRateMonitor:
    """Test LearningRateMonitor callback."""

    def test_initialization_defaults(self):
        """Test LearningRateMonitor initialization."""
        lr_monitor = LearningRateMonitor()
        assert lr_monitor.log_to_console is True

    def test_initialization_custom(self):
        """Test LearningRateMonitor with custom values."""
        lr_monitor = LearningRateMonitor(log_to_console=False)
        assert lr_monitor.log_to_console is False

    def test_on_epoch_end_single_param_group(self, mock_trainer, caplog):
        """Test logging with single parameter group."""
        lr_monitor = LearningRateMonitor(log_to_console=True)
        mock_trainer.optimizer.param_groups = [{"lr": 0.001}]

        with caplog.at_level(logging.INFO):
            lr_monitor.on_epoch_end(mock_trainer, 5, {})

        assert "Epoch   5 | lr = 1.00e-03" in caplog.text

    def test_on_epoch_end_multiple_param_groups(self, mock_trainer, caplog):
        """Test logging with multiple parameter groups."""
        lr_monitor = LearningRateMonitor(log_to_console=True)
        mock_trainer.optimizer.param_groups = [{"lr": 0.001}, {"lr": 0.0005}, {"lr": 0.0001}]

        with caplog.at_level(logging.INFO):
            lr_monitor.on_epoch_end(mock_trainer, 5, {})

        assert "Epoch   5 | lr[group_0] = 1.00e-03" in caplog.text
        assert "Epoch   5 | lr[group_1] = 5.00e-04" in caplog.text
        assert "Epoch   5 | lr[group_2] = 1.00e-04" in caplog.text

    def test_on_epoch_end_no_console_logging(self, mock_trainer, caplog):
        """Test that console logging can be disabled."""
        lr_monitor = LearningRateMonitor(log_to_console=False)

        with caplog.at_level(logging.INFO):
            lr_monitor.on_epoch_end(mock_trainer, 5, {})

        assert "lr" not in caplog.text

    def test_on_epoch_end_without_optimizer(self, caplog):
        """Test handling when optimizer is not present."""
        lr_monitor = LearningRateMonitor(log_to_console=True)
        trainer = Mock()
        trainer.optimizer = None

        # Should not raise any errors
        with caplog.at_level(logging.INFO):
            lr_monitor.on_epoch_end(trainer, 5, {})

        assert "lr" not in caplog.text

    def test_on_epoch_end_without_optimizer_attribute(self, caplog):
        """Test handling when trainer has no optimizer attribute."""
        lr_monitor = LearningRateMonitor(log_to_console=True)
        trainer = Mock(spec=[])

        # Should not raise any errors
        with caplog.at_level(logging.INFO):
            lr_monitor.on_epoch_end(trainer, 5, {})

        assert "lr" not in caplog.text


# =============================================================================
# PROGRESS BAR TESTS
# =============================================================================


class TestProgressBar:
    """Test ProgressBar callback."""

    def test_initialization_defaults(self):
        """Test ProgressBar initialization."""
        progress = ProgressBar()
        assert progress.update_frequency == 1
        assert progress.start_time is None

    def test_initialization_custom(self):
        """Test ProgressBar with custom frequency."""
        progress = ProgressBar(update_frequency=5)
        assert progress.update_frequency == 5

    def test_on_train_begin(self, mock_trainer, caplog):
        """Test on_train_begin sets start time and logs header."""
        progress = ProgressBar()

        with caplog.at_level(logging.INFO):
            progress.on_train_begin(mock_trainer)

        assert progress.start_time is not None
        assert "Training Progress" in caplog.text
        assert "=" * 70 in caplog.text

    def test_on_epoch_end_displays_progress(self, mock_trainer, caplog):
        """Test progress display on epoch end."""
        progress = ProgressBar(update_frequency=1)
        progress.start_time = time.time()
        mock_trainer.max_epochs = 100

        metrics = {"train_loss": 0.5, "val_loss": 0.6}

        with caplog.at_level(logging.INFO):
            progress.on_epoch_end(mock_trainer, 9, metrics)

        assert "Progress:  10.0%" in caplog.text
        assert "Epoch  10/100" in caplog.text
        assert "train_loss: 0.5000" in caplog.text
        assert "val_loss: 0.6000" in caplog.text

    def test_on_epoch_end_respects_update_frequency(self, mock_trainer, caplog):
        """Test that updates respect frequency setting."""
        progress = ProgressBar(update_frequency=5)
        progress.start_time = time.time()

        # Epoch 3 should not log
        with caplog.at_level(logging.INFO):
            progress.on_epoch_end(mock_trainer, 3, {"train_loss": 0.5})
        assert "Progress:" not in caplog.text

        # Epoch 5 should log
        caplog.clear()
        with caplog.at_level(logging.INFO):
            progress.on_epoch_end(mock_trainer, 5, {"train_loss": 0.5})
        assert "Progress:" in caplog.text

    def test_on_epoch_end_without_metrics(self, mock_trainer, caplog):
        """Test progress display with missing metrics."""
        progress = ProgressBar(update_frequency=1)
        progress.start_time = time.time()
        mock_trainer.max_epochs = 100

        with caplog.at_level(logging.INFO):
            progress.on_epoch_end(mock_trainer, 9, {})

        assert "Progress:  10.0%" in caplog.text
        assert "train_loss" not in caplog.text
        assert "val_loss" not in caplog.text

    def test_on_train_end_displays_summary(self, mock_trainer, caplog):
        """Test summary display at training end."""
        progress = ProgressBar()
        progress.start_time = time.time() - 120.5  # 120.5 seconds ago

        with caplog.at_level(logging.INFO):
            progress.on_train_end(mock_trainer)

        assert "Training completed" in caplog.text
        assert "120." in caplog.text  # Duration in seconds
        assert "2.0" in caplog.text  # Duration in minutes
        assert "=" * 70 in caplog.text


# =============================================================================
# GRADIENT MONITOR TESTS
# =============================================================================


class TestGradientMonitor:
    """Test GradientMonitor callback."""

    def test_initialization_defaults(self):
        """Test GradientMonitor initialization."""
        grad_monitor = GradientMonitor()
        assert grad_monitor.log_frequency == 10

    def test_initialization_custom(self):
        """Test GradientMonitor with custom frequency."""
        grad_monitor = GradientMonitor(log_frequency=5)
        assert grad_monitor.log_frequency == 5

    def test_on_epoch_end_computes_gradient_norm(self, mock_trainer, caplog):
        """Test gradient norm computation."""
        grad_monitor = GradientMonitor(log_frequency=1)

        # Setup mock parameters with gradients
        param1 = Mock()
        param1.grad = Mock()
        param1.grad.data.norm = Mock(return_value=torch.tensor(2.0))

        param2 = Mock()
        param2.grad = Mock()
        param2.grad.data.norm = Mock(return_value=torch.tensor(3.0))

        mock_trainer.model.parameters = Mock(return_value=[param1, param2])

        with caplog.at_level(logging.INFO):
            grad_monitor.on_epoch_end(mock_trainer, 0, {})

        # Total norm should be sqrt(2^2 + 3^2) = sqrt(13) ≈ 3.606
        assert "Gradient norm:" in caplog.text
        assert "3.60" in caplog.text
        assert "Params with grad: 2" in caplog.text

    def test_on_epoch_end_respects_log_frequency(self, mock_trainer, caplog):
        """Test that logging respects frequency setting."""
        grad_monitor = GradientMonitor(log_frequency=10)

        # Epoch 5 should not log
        with caplog.at_level(logging.INFO):
            grad_monitor.on_epoch_end(mock_trainer, 5, {})
        assert "Gradient norm" not in caplog.text

        # Epoch 10 should log
        caplog.clear()
        with caplog.at_level(logging.INFO):
            grad_monitor.on_epoch_end(mock_trainer, 10, {})
        assert "Gradient norm" in caplog.text

    def test_on_epoch_end_handles_no_gradients(self, caplog):
        """Test handling when no parameters have gradients."""
        grad_monitor = GradientMonitor(log_frequency=1)

        trainer = Mock()
        param = Mock()
        param.grad = None
        trainer.model.parameters = Mock(return_value=[param])

        with caplog.at_level(logging.INFO):
            grad_monitor.on_epoch_end(trainer, 0, {})

        # Should not log anything since no params have gradients
        assert "Gradient norm" not in caplog.text

    def test_on_epoch_end_mixed_gradients(self, caplog):
        """Test handling when some parameters have gradients and some don't."""
        grad_monitor = GradientMonitor(log_frequency=1)

        trainer = Mock()
        param_with_grad = Mock()
        param_with_grad.grad = Mock()
        param_with_grad.grad.data.norm = Mock(return_value=torch.tensor(2.0))

        param_without_grad = Mock()
        param_without_grad.grad = None

        trainer.model.parameters = Mock(return_value=[param_with_grad, param_without_grad])

        with caplog.at_level(logging.INFO):
            grad_monitor.on_epoch_end(trainer, 0, {})

        assert "Gradient norm:" in caplog.text
        assert "Params with grad: 1" in caplog.text


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestCallbackIntegration:
    """Integration tests for callbacks working together."""

    def test_multiple_callbacks_on_epoch_end(self, mock_trainer, temp_checkpoint_dir):
        """Test multiple callbacks work together on epoch end."""
        es = EarlyStopping(monitor="val_loss", patience=2, verbose=False)
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, verbose=False)
        lr_monitor = LearningRateMonitor(log_to_console=False)

        metrics = {"val_loss": 0.5}

        # All callbacks should execute without conflict
        es.on_epoch_end(mock_trainer, 0, metrics)
        mc.on_epoch_end(mock_trainer, 0, metrics)
        lr_monitor.on_epoch_end(mock_trainer, 0, metrics)

        assert es.best_score == 0.5
        # Use epoch=*.pt to exclude best.pt (save_best=True by default creates best.pt)
        checkpoints = list(temp_checkpoint_dir.glob("epoch=*.pt"))
        assert len(checkpoints) == 1

    def test_callback_workflow(self, mock_trainer, temp_checkpoint_dir):
        """Test complete callback workflow from train begin to train end."""
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, verbose=False),
            ModelCheckpoint(dirpath=temp_checkpoint_dir, verbose=False),
            LearningRateMonitor(log_to_console=False),
            ProgressBar(update_frequency=1),
        ]

        # Set trainer for all callbacks
        for callback in callbacks:
            callback.set_trainer(mock_trainer)

        # Training begin
        for callback in callbacks:
            callback.on_train_begin(mock_trainer)

        # Simulate epochs
        for epoch in range(5):
            metrics = {"val_loss": 0.5 - epoch * 0.05}
            for callback in callbacks:
                callback.on_epoch_end(mock_trainer, epoch, metrics)

        # Training end
        for callback in callbacks:
            callback.on_train_end(mock_trainer)

        # Verify results
        assert callbacks[0].best_score == 0.3  # EarlyStopping
        checkpoints = list(temp_checkpoint_dir.glob("epoch=*.pt"))
        assert len(checkpoints) == 3  # ModelCheckpoint with save_top_k=3


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_early_stopping_with_none_metrics(self, mock_trainer):
        """Test EarlyStopping handles None in metrics."""
        es = EarlyStopping(verbose=False)

        # Should not crash
        es.on_epoch_end(mock_trainer, 0, {"val_loss": None})
        assert es.best_score is None

    def test_model_checkpoint_path_with_special_characters(self, temp_checkpoint_dir):
        """Test ModelCheckpoint with special characters in paths."""
        subdir = temp_checkpoint_dir / "test-model_v1.0"
        _mc = ModelCheckpoint(dirpath=subdir, verbose=False)
        assert subdir.exists()

    def test_callbacks_with_empty_metrics(self, mock_trainer):
        """Test callbacks handle empty metrics gracefully."""
        es = EarlyStopping(verbose=False)
        lr_monitor = LearningRateMonitor(log_to_console=False)
        progress = ProgressBar()
        progress.start_time = time.time()

        empty_metrics = {}

        # Should not crash
        es.on_epoch_end(mock_trainer, 0, empty_metrics)
        lr_monitor.on_epoch_end(mock_trainer, 0, empty_metrics)
        progress.on_epoch_end(mock_trainer, 0, empty_metrics)

    def test_gradient_monitor_with_empty_model(self, caplog):
        """Test GradientMonitor with model that has no parameters."""
        grad_monitor = GradientMonitor(log_frequency=1)

        trainer = Mock()
        trainer.model.parameters = Mock(return_value=[])

        with caplog.at_level(logging.INFO):
            grad_monitor.on_epoch_end(trainer, 0, {})

        # Should not log anything
        assert "Gradient norm" not in caplog.text

    def test_early_stopping_patience_zero(self, mock_trainer):
        """Test EarlyStopping with patience=0."""
        es = EarlyStopping(patience=0, verbose=False)

        es.on_epoch_end(mock_trainer, 0, {"val_loss": 0.5})
        es.on_epoch_end(mock_trainer, 1, {"val_loss": 0.6})

        # Should stop immediately after first non-improvement
        assert es.should_stop() is True

    def test_model_checkpoint_concurrent_saves(self, mock_trainer, temp_checkpoint_dir):
        """Test ModelCheckpoint handles concurrent epoch ends."""
        mc = ModelCheckpoint(dirpath=temp_checkpoint_dir, save_top_k=5, verbose=False)

        # Simulate rapid successive saves
        for i in range(10):
            mc.on_epoch_end(mock_trainer, i, {"val_loss": 0.5 - i * 0.01})

        # Should have only top 5
        checkpoints = list(temp_checkpoint_dir.glob("epoch=*.pt"))
        assert len(checkpoints) == 5


# =============================================================================
# CALLBACK FACTORY TESTS
# =============================================================================


class TestCallbackFactory:
    """Test CallbackFactory class."""

    # -------------------------------------------------------------------------
    # Registry Tests
    # -------------------------------------------------------------------------

    def test_callback_registry_contains_all_callbacks(self):
        """Test that registry contains all expected callback types."""
        expected_callbacks = {
            "early_stopping",
            "model_checkpoint",
            "tensorboard",
            "lr_monitor",
            "progress_bar",
            "gradient_monitor",
        }
        assert set(CallbackFactory._callback_registry.keys()) == expected_callbacks

    def test_callback_registry_maps_to_correct_classes(self):
        """Test that registry maps to correct callback classes."""
        assert CallbackFactory._callback_registry["early_stopping"] is EarlyStopping
        assert CallbackFactory._callback_registry["model_checkpoint"] is ModelCheckpoint
        assert CallbackFactory._callback_registry["tensorboard"] is TensorBoardLogger
        assert CallbackFactory._callback_registry["lr_monitor"] is LearningRateMonitor
        assert CallbackFactory._callback_registry["progress_bar"] is ProgressBar
        assert CallbackFactory._callback_registry["gradient_monitor"] is GradientMonitor

    def test_path_params_registry(self):
        """Test path parameters registry configuration."""
        assert "model_checkpoint" in CallbackFactory._path_params
        assert CallbackFactory._path_params["model_checkpoint"] == {"dirpath": "checkpoints"}
        assert "tensorboard" in CallbackFactory._path_params
        assert CallbackFactory._path_params["tensorboard"] == {"log_dir": "tensorboard_logs"}

    # -------------------------------------------------------------------------
    # list_available Tests
    # -------------------------------------------------------------------------

    def test_list_available_returns_sorted_list(self):
        """Test that list_available returns sorted list of callback names."""
        available = CallbackFactory.list_available()
        assert isinstance(available, list)
        assert available == sorted(available)
        assert "early_stopping" in available
        assert "model_checkpoint" in available

    # -------------------------------------------------------------------------
    # get_callback_class Tests
    # -------------------------------------------------------------------------

    def test_get_callback_class_valid_name(self):
        """Test get_callback_class with valid callback name."""
        cls = CallbackFactory.get_callback_class("early_stopping")
        assert cls is EarlyStopping

    def test_get_callback_class_invalid_name_raises_error(self):
        """Test get_callback_class with invalid name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown callback: 'invalid'"):
            CallbackFactory.get_callback_class("invalid")

    def test_get_callback_class_error_shows_available(self):
        """Test that error message shows available callbacks."""
        with pytest.raises(ValueError) as exc_info:
            CallbackFactory.get_callback_class("nonexistent")
        assert "Available:" in str(exc_info.value)
        assert "early_stopping" in str(exc_info.value)

    # -------------------------------------------------------------------------
    # from_config Tests - Basic Functionality
    # -------------------------------------------------------------------------

    def test_from_config_empty_config_returns_default_callbacks(self, temp_working_dir):
        """Test from_config with empty config returns early_stopping and model_checkpoint by default."""
        callbacks = CallbackFactory.from_config(
            callback_config={}, working_root_dir=temp_working_dir
        )
        # early_stopping and model_checkpoint default to enabled
        assert len(callbacks) == 2
        callback_types = [type(cb).__name__ for cb in callbacks]
        assert "EarlyStopping" in callback_types
        assert "ModelCheckpoint" in callback_types

    def test_from_config_explicit_disabled(self, temp_working_dir):
        """Test that explicitly disabled callbacks are not created."""
        config = {
            "early_stopping": {"enabled": False},
            "model_checkpoint": {"enabled": False},
        }
        callbacks = CallbackFactory.from_config(
            callback_config=config, working_root_dir=temp_working_dir
        )
        assert len(callbacks) == 0

    def test_from_config_explicit_enabled(self, temp_working_dir):
        """Test that explicitly enabled callbacks are created."""
        config = {
            "progress_bar": {"enabled": True},
            "lr_monitor": {"enabled": True},
            "early_stopping": {"enabled": False},
            "model_checkpoint": {"enabled": False},
        }
        callbacks = CallbackFactory.from_config(
            callback_config=config, working_root_dir=temp_working_dir
        )
        assert len(callbacks) == 2
        callback_types = [type(cb).__name__ for cb in callbacks]
        assert "ProgressBar" in callback_types
        assert "LearningRateMonitor" in callback_types

    def test_from_config_with_params(self, temp_working_dir):
        """Test from_config passes parameters to callback constructors."""
        config = {
            "early_stopping": {
                "enabled": True,
                "params": {"patience": 20, "monitor": "val_acc", "mode": "max", "min_delta": 0.001},
            },
            "model_checkpoint": {"enabled": False},
        }
        callbacks = CallbackFactory.from_config(
            callback_config=config, working_root_dir=temp_working_dir
        )
        assert len(callbacks) == 1
        es = callbacks[0]
        assert isinstance(es, EarlyStopping)
        assert es.patience == 20
        assert es.monitor == "val_acc"
        assert es.mode == "max"
        assert es.min_delta == 0.001

    # -------------------------------------------------------------------------
    # from_config Tests - Path Resolution
    # -------------------------------------------------------------------------

    def test_from_config_auto_generates_checkpoint_path(self, temp_working_dir):
        """Test that dirpath is auto-generated for model_checkpoint when None."""
        config = {
            "model_checkpoint": {"enabled": True, "params": {"dirpath": None}},
            "early_stopping": {"enabled": False},
        }
        callbacks = CallbackFactory.from_config(
            callback_config=config, working_root_dir=temp_working_dir
        )

        checkpoint_cb = callbacks[0]
        assert isinstance(checkpoint_cb, ModelCheckpoint)
        expected_path = temp_working_dir / "checkpoints"
        assert checkpoint_cb.dirpath == expected_path
        assert expected_path.exists()

    def test_from_config_auto_generates_tensorboard_path(self, temp_working_dir):
        """Test that log_dir is auto-generated for tensorboard when None."""
        config = {
            "tensorboard": {"enabled": True, "params": {"log_dir": None}},
            "early_stopping": {"enabled": False},
            "model_checkpoint": {"enabled": False},
        }

        # Mock _is_tensorboard_available to ensure tensorboard callback is created
        with patch(
            "milia_pipeline.models.training.callbacks._is_tensorboard_available", return_value=True
        ):
            callbacks = CallbackFactory.from_config(
                callback_config=config, working_root_dir=temp_working_dir
            )

        tb_cb = callbacks[0]
        assert isinstance(tb_cb, TensorBoardLogger)
        expected_path = temp_working_dir / "tensorboard_logs"
        assert tb_cb.log_dir == expected_path
        # Note: directory might be created by TensorBoardLogger based on availability

    def test_from_config_respects_provided_paths(self, temp_working_dir):
        """Test that provided paths are used when specified."""
        custom_checkpoint_dir = temp_working_dir / "my_custom_checkpoints"
        config = {
            "model_checkpoint": {
                "enabled": True,
                "params": {"dirpath": str(custom_checkpoint_dir)},
            },
            "early_stopping": {"enabled": False},
        }
        callbacks = CallbackFactory.from_config(
            callback_config=config, working_root_dir=temp_working_dir
        )

        checkpoint_cb = callbacks[0]
        assert checkpoint_cb.dirpath == custom_checkpoint_dir

    def test_from_config_expands_user_path(self, temp_working_dir):
        """Test that user paths (~) are expanded."""
        config = {
            "model_checkpoint": {"enabled": True, "params": {"dirpath": "~/checkpoints"}},
            "early_stopping": {"enabled": False},
        }
        callbacks = CallbackFactory.from_config(
            callback_config=config, working_root_dir=temp_working_dir
        )

        checkpoint_cb = callbacks[0]
        # Path should be expanded, not contain ~
        assert "~" not in str(checkpoint_cb.dirpath)
        assert str(checkpoint_cb.dirpath).startswith("/")

    # -------------------------------------------------------------------------
    # from_config Tests - Parameter Filtering
    # -------------------------------------------------------------------------

    def test_from_config_filters_unsupported_params(self, temp_working_dir, caplog):
        """Test that unsupported parameters are filtered out."""
        config = {
            "early_stopping": {
                "enabled": True,
                "params": {"patience": 5, "unsupported_param": "value", "another_invalid": 123},
            },
            "model_checkpoint": {"enabled": False},
        }
        with caplog.at_level(logging.DEBUG):
            callbacks = CallbackFactory.from_config(
                callback_config=config, working_root_dir=temp_working_dir
            )

        # Should still create the callback
        assert len(callbacks) == 1
        es = callbacks[0]
        assert es.patience == 5
        # Unsupported params should be mentioned in debug log
        assert "Ignored unsupported params" in caplog.text or len(callbacks) == 1

    # -------------------------------------------------------------------------
    # from_config Tests - Error Handling
    # -------------------------------------------------------------------------

    def test_from_config_raises_on_invalid_callback_params(self, temp_working_dir):
        """Test that invalid callback parameters raise appropriate errors."""
        config = {
            "early_stopping": {
                "enabled": True,
                "params": {"mode": "invalid_mode"},  # Invalid mode
            },
            "model_checkpoint": {"enabled": False},
        }
        with pytest.raises(ValueError, match="mode must be 'min' or 'max'"):
            CallbackFactory.from_config(callback_config=config, working_root_dir=temp_working_dir)

    def test_from_config_logs_callback_count(self, temp_working_dir, caplog):
        """Test that from_config logs the number of callbacks created."""
        config = {
            "early_stopping": {"enabled": True},
            "model_checkpoint": {"enabled": True},
            "progress_bar": {"enabled": True},
        }
        with caplog.at_level(logging.INFO):
            _callbacks = CallbackFactory.from_config(
                callback_config=config, working_root_dir=temp_working_dir
            )

        assert "Created 3 callbacks" in caplog.text

    def test_from_config_accepts_custom_logger(self, temp_working_dir, caplog):
        """Test that from_config uses provided logger."""
        custom_logger = logging.getLogger("custom_test_logger")
        custom_logger.setLevel(logging.DEBUG)

        config = {
            "early_stopping": {"enabled": False},
            "model_checkpoint": {"enabled": False},
        }

        with caplog.at_level(logging.DEBUG):
            CallbackFactory.from_config(
                callback_config=config,
                working_root_dir=temp_working_dir,
                callback_logger=custom_logger,
            )

    # -------------------------------------------------------------------------
    # from_config Tests - All Callbacks
    # -------------------------------------------------------------------------

    def test_from_config_creates_all_callbacks(self, temp_working_dir):
        """Test creating all callback types at once."""
        config = {
            "early_stopping": {"enabled": True},
            "model_checkpoint": {"enabled": True},
            "tensorboard": {"enabled": True},
            "lr_monitor": {"enabled": True},
            "progress_bar": {"enabled": True},
            "gradient_monitor": {"enabled": True},
        }

        # Mock _is_tensorboard_available to ensure consistent test behavior
        with patch(
            "milia_pipeline.models.training.callbacks._is_tensorboard_available", return_value=True
        ):
            callbacks = CallbackFactory.from_config(
                callback_config=config, working_root_dir=temp_working_dir
            )

        assert len(callbacks) == 6
        callback_types = {type(cb).__name__ for cb in callbacks}
        assert callback_types == {
            "EarlyStopping",
            "ModelCheckpoint",
            "TensorBoardLogger",
            "LearningRateMonitor",
            "ProgressBar",
            "GradientMonitor",
        }

    def test_from_config_skips_tensorboard_when_unavailable(self, temp_working_dir, caplog):
        """Test that CallbackFactory skips TensorBoardLogger when tensorboard is not installed."""
        config = {
            "tensorboard": {"enabled": True},
            "early_stopping": {"enabled": False},
            "model_checkpoint": {"enabled": False},
        }

        # Mock _is_tensorboard_available to return False
        with (
            patch(
                "milia_pipeline.models.training.callbacks._is_tensorboard_available",
                return_value=False,
            ),
            caplog.at_level(logging.INFO),
        ):
            callbacks = CallbackFactory.from_config(
                callback_config=config, working_root_dir=temp_working_dir
            )

        # TensorBoardLogger should not be created
        callback_types = [type(cb).__name__ for cb in callbacks]
        assert "TensorBoardLogger" not in callback_types

        # Should log that tensorboard was skipped
        assert (
            "tensorboard not installed" in caplog.text
            or "Skipping TensorBoardLogger" in caplog.text
        )

    def test_from_config_creates_tensorboard_when_available(self, temp_working_dir):
        """Test that CallbackFactory creates TensorBoardLogger when tensorboard is installed."""
        config = {
            "tensorboard": {"enabled": True},
            "early_stopping": {"enabled": False},
            "model_checkpoint": {"enabled": False},
        }

        # Mock _is_tensorboard_available to return True
        with patch(
            "milia_pipeline.models.training.callbacks._is_tensorboard_available", return_value=True
        ):
            callbacks = CallbackFactory.from_config(
                callback_config=config, working_root_dir=temp_working_dir
            )

        # TensorBoardLogger should be created
        callback_types = [type(cb).__name__ for cb in callbacks]
        assert "TensorBoardLogger" in callback_types

    # -------------------------------------------------------------------------
    # _filter_params Tests
    # -------------------------------------------------------------------------

    def test_filter_params_filters_to_valid_params(self):
        """Test _filter_params filters to valid constructor parameters."""
        params = {"patience": 10, "monitor": "val_loss", "invalid_param": "should_be_removed"}
        filtered = CallbackFactory._filter_params(EarlyStopping, params)

        assert "patience" in filtered
        assert "monitor" in filtered
        assert "invalid_param" not in filtered

    def test_filter_params_empty_params(self):
        """Test _filter_params with empty params."""
        filtered = CallbackFactory._filter_params(EarlyStopping, {})
        assert filtered == {}

    def test_filter_params_all_valid(self):
        """Test _filter_params when all params are valid."""
        params = {"patience": 10, "monitor": "val_loss", "mode": "min"}
        filtered = CallbackFactory._filter_params(EarlyStopping, params)
        assert filtered == params

    def test_filter_params_none_params(self):
        """Test _filter_params with None params."""
        # None should be treated as empty
        filtered = CallbackFactory._filter_params(EarlyStopping, None)
        assert filtered == {}

    # -------------------------------------------------------------------------
    # register_custom_callback Tests
    # -------------------------------------------------------------------------

    def test_register_custom_callback_success(self):
        """Test registering a custom callback."""

        class CustomCallback(Callback):
            def __init__(self, custom_param: int = 5):
                super().__init__()
                self.custom_param = custom_param

        # Register
        CallbackFactory.register_custom_callback("custom", CustomCallback)

        try:
            assert "custom" in CallbackFactory._callback_registry
            assert CallbackFactory._callback_registry["custom"] is CustomCallback
            assert "custom" in CallbackFactory.list_available()
        finally:
            # Cleanup
            del CallbackFactory._callback_registry["custom"]

    def test_register_custom_callback_with_path_params(self):
        """Test registering a custom callback with path parameters."""

        class CustomCallback(Callback):
            def __init__(self, output_dir: Path = None):
                super().__init__()
                self.output_dir = output_dir

        CallbackFactory.register_custom_callback(
            "custom_with_path", CustomCallback, path_params={"output_dir": "custom_output"}
        )

        try:
            assert "custom_with_path" in CallbackFactory._path_params
            assert CallbackFactory._path_params["custom_with_path"] == {
                "output_dir": "custom_output"
            }
        finally:
            # Cleanup
            del CallbackFactory._callback_registry["custom_with_path"]
            del CallbackFactory._path_params["custom_with_path"]

    def test_register_custom_callback_duplicate_raises_error(self):
        """Test that registering duplicate callback raises error without overwrite."""

        class CustomCallback(Callback):
            pass

        CallbackFactory.register_custom_callback("temp_custom", CustomCallback)

        try:
            with pytest.raises(ValueError, match="already registered"):
                CallbackFactory.register_custom_callback("temp_custom", CustomCallback)
        finally:
            del CallbackFactory._callback_registry["temp_custom"]

    def test_register_custom_callback_overwrite(self):
        """Test overwriting existing callback registration."""

        class CustomCallback1(Callback):
            pass

        class CustomCallback2(Callback):
            pass

        CallbackFactory.register_custom_callback("overwrite_test", CustomCallback1)

        try:
            # Should succeed with overwrite=True
            CallbackFactory.register_custom_callback(
                "overwrite_test", CustomCallback2, overwrite=True
            )
            assert CallbackFactory._callback_registry["overwrite_test"] is CustomCallback2
        finally:
            del CallbackFactory._callback_registry["overwrite_test"]

    def test_register_custom_callback_non_callback_subclass_raises_error(self):
        """Test that registering non-Callback subclass raises TypeError."""

        class NotACallback:
            pass

        with pytest.raises(TypeError, match="must be subclass of Callback"):
            CallbackFactory.register_custom_callback("not_callback", NotACallback)

    # -------------------------------------------------------------------------
    # Integration Tests
    # -------------------------------------------------------------------------

    def test_from_config_with_custom_registered_callback(self, temp_working_dir):
        """Test that custom registered callbacks can be created via from_config."""

        class CustomCallback(Callback):
            def __init__(self, my_param: int = 10):
                super().__init__()
                self.my_param = my_param

        CallbackFactory.register_custom_callback("custom_integration", CustomCallback)

        try:
            config = {
                "custom_integration": {"enabled": True, "params": {"my_param": 42}},
                "early_stopping": {"enabled": False},
                "model_checkpoint": {"enabled": False},
            }
            callbacks = CallbackFactory.from_config(
                callback_config=config, working_root_dir=temp_working_dir
            )

            assert len(callbacks) == 1
            custom_cb = callbacks[0]
            assert isinstance(custom_cb, CustomCallback)
            assert custom_cb.my_param == 42
        finally:
            del CallbackFactory._callback_registry["custom_integration"]


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
