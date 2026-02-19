"""
Callbacks Module

Extensible callback system for training monitoring and control.
Includes common callbacks: EarlyStopping, ModelCheckpoint, TensorBoard, etc.

Author: milia Team
Version: 1.0.0
"""

import logging
from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

# Import exceptions with fallback
try:
    from milia_pipeline.exceptions import CheckpointError
except ImportError:

    class CheckpointError(Exception):
        """Exception raised for checkpoint issues."""

        pass


logger = logging.getLogger(__name__)


# =============================================================================
# OPTIONAL DEPENDENCY CHECKS
# =============================================================================


def _is_tensorboard_available() -> bool:
    """
    Check if TensorBoard is installed and available.

    Returns:
        True if torch.utils.tensorboard.SummaryWriter can be imported, False otherwise.

    Note:
        This check is used to avoid creating empty directories when TensorBoard
        is not installed. TensorBoard is an OPTIONAL dependency of MILIA.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401 — import tests full dependency chain (torch shim → tensorboard package)

        return True
    except ImportError:
        return False


# =============================================================================
# BASE CALLBACK
# =============================================================================


class Callback(ABC):
    """
    Abstract base class for training callbacks.

    Callbacks provide hooks into the training process to:
    - Monitor metrics
    - Control training flow (e.g., early stopping)
    - Save checkpoints
    - Log to external systems
    - Implement custom behaviors

    Methods to override:
    - set_trainer(trainer): Called when callback is attached to trainer
    - on_train_begin(trainer): Called at the start of training
    - on_epoch_end(trainer, epoch, metrics): Called at the end of each epoch
    - on_train_end(trainer): Called at the end of training

    Usage:
        >>> class CustomCallback(Callback):
        ...     def on_epoch_end(self, trainer, epoch, metrics):
        ...         print(f"Epoch {epoch}: {metrics}")
        >>>
        >>> trainer = Trainer(..., callbacks=[CustomCallback()])
    """

    def set_trainer(self, trainer):
        """
        Attach callback to trainer.

        Args:
            trainer: Trainer instance
        """
        self.trainer = trainer

    def on_train_begin(self, trainer):
        """
        Called at the beginning of training.

        Args:
            trainer: Trainer instance
        """
        pass

    def on_epoch_end(self, trainer, epoch: int, metrics: dict[str, float]):
        """
        Called at the end of each epoch.

        Args:
            trainer: Trainer instance
            epoch: Current epoch number
            metrics: Dictionary of metrics for this epoch
        """
        pass

    def on_train_end(self, trainer):
        """
        Called at the end of training.

        Args:
            trainer: Trainer instance
        """
        pass


# =============================================================================
# EARLY STOPPING
# =============================================================================


class EarlyStopping(Callback):
    """
    Stop training when monitored metric stops improving.

    Monitors a metric (e.g., validation loss) and stops training if it
    doesn't improve for a specified number of epochs (patience).

    Args:
        monitor: Metric to monitor (default: "val_loss")
        patience: Number of epochs with no improvement before stopping
        mode: "min" for metrics that should decrease, "max" for metrics that should increase
        min_delta: Minimum change to qualify as improvement
        verbose: Whether to log when improvement occurs

    Attributes:
        best_score: Best score observed so far
        counter: Number of epochs since last improvement

    Usage:
        >>> early_stop = EarlyStopping(
        ...     monitor="val_loss",
        ...     patience=10,
        ...     mode="min",
        ...     min_delta=0.0001
        ... )
        >>> trainer = Trainer(..., callbacks=[early_stop])
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.0001,
        verbose: bool = True,
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose

        # State
        self.best_score = None
        self.counter = 0
        self._stop = False

        # Validation
        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        logger.info(
            f"EarlyStopping initialized: monitor='{monitor}', "
            f"patience={patience}, mode='{mode}', min_delta={min_delta}"
        )

    def on_epoch_end(self, trainer, epoch: int, metrics: dict[str, float]):
        """Check for improvement and update early stopping state."""
        score = metrics.get(self.monitor)

        if score is None:
            logger.warning(
                f"EarlyStopping: metric '{self.monitor}' not found in metrics. "
                f"Available metrics: {list(metrics.keys())}"
            )
            return

        # First epoch
        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                logger.info(f"EarlyStopping: Initial {self.monitor}={score:.6f}")
            return

        # Check for improvement
        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            if self.verbose:
                logger.info(
                    f"EarlyStopping: Improvement detected! "
                    f"{self.monitor}={score:.6f} (counter reset)"
                )
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping: No improvement for {self.counter} epoch(s). "
                    f"{self.monitor}={score:.6f} vs best={self.best_score:.6f}"
                )

            # Trigger early stopping
            if self.counter >= self.patience:
                logger.info(
                    f"EarlyStopping: Stopping training at epoch {epoch}. "
                    f"No improvement for {self.patience} epochs. "
                    f"Best {self.monitor}={self.best_score:.6f}"
                )
                self._stop = True

    def _is_improvement(self, score: float) -> bool:
        """Check if score represents an improvement."""
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:  # mode == "max"
            return score > self.best_score + self.min_delta

    def should_stop(self) -> bool:
        """Check if training should stop."""
        return self._stop

    def on_train_end(self, trainer):
        """Log final early stopping state."""
        if self._stop:
            logger.info(
                f"EarlyStopping: Training stopped early. Best {self.monitor}={self.best_score:.6f}"
            )
        else:
            logger.info(
                f"EarlyStopping: Training completed without early stopping. "
                f"Best {self.monitor}={self.best_score:.6f}"
            )


# =============================================================================
# MODEL CHECKPOINT
# =============================================================================


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.

    Automatically saves the best K checkpoints based on a monitored metric.
    Also optionally saves the last checkpoint and a dedicated best.pt file.

    DYNAMIC: Automatically tracks best model path and score during training.
    PRODUCTION-READY: Saves dedicated best.pt for easy post-training access.
    FUTURE-PROOF: Matches PyTorch Lightning ModelCheckpoint API pattern.

    Args:
        dirpath: Directory to save checkpoints (REQUIRED - must be explicitly provided,
                 typically derived from config['global_paths']['working_root_dir'])
        monitor: Metric to monitor for determining best checkpoints
        mode: "min" or "max" - direction of improvement
        save_top_k: Number of best checkpoints to keep (-1 for all)
        save_last: Whether to save a "last.pt" checkpoint at the end
        save_best: Whether to save a dedicated "best.pt" checkpoint (default: True)
        filename_pattern: Pattern for checkpoint filenames
        verbose: Whether to log checkpoint operations

    Attributes:
        best_model_path: Path to the best checkpoint file (updated automatically)
        best_model_score: Score of the best checkpoint (updated automatically)

    Raises:
        ValueError: If dirpath is None (must be explicitly specified)

    Usage:
        >>> # dirpath should come from config['global_paths']['working_root_dir']
        >>> checkpoint = ModelCheckpoint(
        ...     dirpath=Path(working_root_dir) / "checkpoints",
        ...     monitor="val_loss",
        ...     mode="min",
        ...     save_top_k=3,
        ...     save_last=True,
        ...     save_best=True
        ... )
        >>> trainer = Trainer(..., callbacks=[checkpoint])
        >>> trainer.fit()
        >>> # Access best checkpoint after training
        >>> print(checkpoint.best_model_path)  # Path to best.pt
        >>> print(checkpoint.best_model_score)  # Best validation loss
    """

    def __init__(
        self,
        dirpath: Path | None = None,
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 3,
        save_last: bool = True,
        save_best: bool = True,
        filename_pattern: str = "epoch={epoch:03d}-{monitor}={score:.4f}.pt",
        verbose: bool = True,
    ):
        super().__init__()
        # Require explicit dirpath - no hardcoded fallback to './checkpoints'
        # The caller (main.py _create_callbacks) is responsible for providing a proper
        # dirpath based on config['global_paths']['working_root_dir'].
        # This ensures checkpoints are never created in the project root directory.
        if dirpath is None:
            raise ValueError(
                "ModelCheckpoint requires 'dirpath' to be specified. "
                "The checkpoint directory should be derived from config['global_paths']['working_root_dir']. "
                "If using main.py, ensure _create_callbacks() is called with a valid config."
            )
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.save_best = save_best
        self.filename_pattern = filename_pattern
        self.verbose = verbose

        # Create directory
        self.dirpath.mkdir(parents=True, exist_ok=True)

        # State: list of (score, path) tuples
        self.best_checkpoints: list[tuple] = []

        # =====================================================================
        # BEST MODEL TRACKING (Following PyTorch Lightning pattern)
        # =====================================================================
        # DYNAMIC: Updated whenever a new best checkpoint is found
        # PRODUCTION-READY: Provides direct access to best model for post-training
        # FUTURE-PROOF: Standard interface matching PyTorch Lightning
        # =====================================================================
        self._best_model_path: Path | None = None
        self._best_model_score: float | None = None

        # Validation
        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        if save_top_k == 0:
            logger.warning("ModelCheckpoint: save_top_k=0, no checkpoints will be saved")

        logger.info(
            f"ModelCheckpoint initialized: dirpath='{dirpath}', "
            f"monitor='{monitor}', mode='{mode}', save_top_k={save_top_k}"
        )

    # =========================================================================
    # BEST MODEL PROPERTIES (PyTorch Lightning compatible API)
    # =========================================================================

    @property
    def best_model_path(self) -> Path | None:
        """
        Path to the best model checkpoint.

        DYNAMIC: Updated automatically when a new best checkpoint is saved.
        PRODUCTION-READY: Returns None if no checkpoints saved yet.
        FUTURE-PROOF: Matches PyTorch Lightning ModelCheckpoint API.

        Returns:
            Path to best checkpoint file (best.pt), or None if no checkpoints exist.

        Example:
            >>> checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/', monitor='val_loss')
            >>> trainer.fit(model)
            >>> print(checkpoint_callback.best_model_path)
            PosixPath('checkpoints/best.pt')
        """
        if self.save_best and self._best_model_path is not None:
            return self.dirpath / "best.pt"
        return self._best_model_path

    @property
    def best_model_score(self) -> float | None:
        """
        Score of the best model checkpoint.

        DYNAMIC: Updated automatically when a new best checkpoint is saved.
        PRODUCTION-READY: Returns None if no checkpoints saved yet.
        FUTURE-PROOF: Matches PyTorch Lightning ModelCheckpoint API.

        Returns:
            Best score value (float), or None if no checkpoints exist.

        Example:
            >>> checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/', monitor='val_loss')
            >>> trainer.fit(model)
            >>> print(checkpoint_callback.best_model_score)
            0.0234
        """
        return self._best_model_score

    def on_epoch_end(self, trainer, epoch: int, metrics: dict[str, float]):
        """Save checkpoint if it's among the best."""
        score = metrics.get(self.monitor)

        if score is None:
            logger.warning(
                f"ModelCheckpoint: metric '{self.monitor}' not found. "
                f"Available: {list(metrics.keys())}"
            )
            return

        # Don't save if save_top_k is 0
        if self.save_top_k == 0:
            return

        # Generate filename
        filename = self.filename_pattern.format(epoch=epoch, monitor=self.monitor, score=score)
        checkpoint_path = self.dirpath / filename

        # Save checkpoint
        try:
            self._save_checkpoint(trainer, checkpoint_path, epoch, score)

            # Track this checkpoint
            self.best_checkpoints.append((score, checkpoint_path))

            # Sort checkpoints (best first)
            reverse = self.mode == "max"
            self.best_checkpoints.sort(key=lambda x: x[0], reverse=reverse)

            # =================================================================
            # UPDATE BEST MODEL TRACKING
            # =================================================================
            # DYNAMIC: Automatically updates when a new best is found
            # PRODUCTION-READY: Saves dedicated best.pt for easy post-training access
            # FUTURE-PROOF: Matches PyTorch Lightning pattern
            # =================================================================
            current_best_score, current_best_path = self.best_checkpoints[0]
            is_new_best = (
                self._best_model_score is None
                or (self.mode == "min" and current_best_score < self._best_model_score)
                or (self.mode == "max" and current_best_score > self._best_model_score)
            )

            if is_new_best:
                self._best_model_score = current_best_score
                self._best_model_path = current_best_path

                # Save dedicated best.pt file for easy access
                if self.save_best:
                    best_path = self.dirpath / "best.pt"
                    try:
                        self._save_checkpoint(
                            trainer, best_path, epoch, current_best_score, is_best=True
                        )
                        if self.verbose:
                            logger.info(
                                f"ModelCheckpoint: New best model! "
                                f"{self.monitor}={current_best_score:.6f} -> best.pt"
                            )
                    except Exception as e:
                        logger.error(f"ModelCheckpoint: Failed to save best.pt: {e}")

            # Remove old checkpoints if exceeding save_top_k
            if self.save_top_k > 0 and len(self.best_checkpoints) > self.save_top_k:
                # Remove worst checkpoint
                _, old_path = self.best_checkpoints.pop()
                if old_path.exists():
                    old_path.unlink()
                    if self.verbose:
                        logger.info(f"ModelCheckpoint: Removed old checkpoint: {old_path.name}")

        except Exception as e:
            logger.error(f"ModelCheckpoint: Failed to save checkpoint: {e}")

    def on_train_end(self, trainer):
        """Save final checkpoint if save_last is True."""
        if self.save_last:
            last_path = self.dirpath / "last.pt"
            try:
                self._save_checkpoint(trainer, last_path, trainer.current_epoch, None)
                if self.verbose:
                    logger.info(f"ModelCheckpoint: Saved final checkpoint: {last_path}")
            except Exception as e:
                logger.error(f"ModelCheckpoint: Failed to save final checkpoint: {e}")

        # Log summary with best model information
        if self._best_model_path is not None:
            logger.info(
                f"ModelCheckpoint: Best checkpoint - "
                f"{self.monitor}={self._best_model_score:.6f} at {self._best_model_path.name}"
            )
            if self.save_best:
                best_pt_path = self.dirpath / "best.pt"
                logger.info(f"ModelCheckpoint: Best model saved to: {best_pt_path}")
        elif self.best_checkpoints:
            # Fallback to best_checkpoints list if properties not set
            best_score, best_path = self.best_checkpoints[0]
            logger.info(
                f"ModelCheckpoint: Best checkpoint - "
                f"{self.monitor}={best_score:.6f} at {best_path.name}"
            )

    def _save_checkpoint(
        self, trainer, path: Path, epoch: int, score: float | None, is_best: bool = False
    ):
        """
        Save checkpoint to disk.

        Args:
            trainer: Trainer instance
            path: Path to save checkpoint
            epoch: Current epoch number
            score: Monitored metric score (optional)
            is_best: Whether this is the best checkpoint (for best.pt)
        """
        checkpoint = {
            "epoch": epoch,
            "global_step": trainer.global_step,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "metrics_history": dict(trainer.metrics_history),
            "best_val_loss": trainer.best_val_loss,
        }

        # Add scheduler state if present
        if trainer.scheduler is not None:
            checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()

        # Add monitored score
        if score is not None:
            checkpoint["monitored_score"] = score
            checkpoint["monitored_metric"] = self.monitor

        # =====================================================================
        # BEST CHECKPOINT MARKER
        # =====================================================================
        # DYNAMIC: is_best flag indicates this is THE best checkpoint
        # PRODUCTION-READY: Post-training can verify checkpoint is best
        # FUTURE-PROOF: Standard marker for checkpoint identification
        # =====================================================================
        checkpoint["is_best"] = is_best

        # =====================================================================
        # V2.0 CHECKPOINT FORMAT: Add model recreation metadata
        # =====================================================================
        # DYNAMIC: Uses whatever is in trainer.model_info
        # PRODUCTION-READY: Enables post_training.load_model() to work
        # FUTURE-PROOF: Format version enables backward compatibility
        # =====================================================================
        if hasattr(trainer, "model_info") and trainer.model_info:
            checkpoint["hyper_parameters"] = {
                "model_name": trainer.model_info.get("name"),
                "task_type": trainer.model_info.get("task_type"),
                "hyperparameters": trainer.model_info.get("hyperparameters_values", {}),
                "model_info": trainer.model_info,
                "wrapper_info": trainer.model_info.get("wrapper_info", {}),
                "target_selection_config": trainer.model_info.get("target_selection"),
            }
        else:
            checkpoint["hyper_parameters"] = {}

        # =====================================================================
        # FIX 17: SAVE FEATURIZATION CONFIG IN CHECKPOINT
        # =====================================================================
        # DYNAMIC: Saves whatever structural_features_config is in model_info
        # PRODUCTION-READY: Follows existing pattern for edge feature flags
        # FUTURE-PROOF: Works with any featurization config structure
        # =====================================================================
        checkpoint["data_info"] = {
            "requires_edge_features": trainer.model_info.get("requires_edge_features", False)
            if hasattr(trainer, "model_info") and trainer.model_info
            else False,
            "uses_edge_features": trainer.model_info.get("uses_edge_features", False)
            if hasattr(trainer, "model_info") and trainer.model_info
            else False,
            "structural_features_config": trainer.model_info.get("structural_features_config", {})
            if hasattr(trainer, "model_info") and trainer.model_info
            else {},
        }

        # Version info for format detection
        try:
            import torch_geometric

            pyg_version = torch_geometric.__version__
        except (ImportError, AttributeError):
            pyg_version = "unknown"

        checkpoint["version_info"] = {
            "checkpoint_format_version": "2.0",
            "pytorch_version": torch.__version__,
            "pyg_version": pyg_version,
            "created_at": datetime.now().isoformat(),
        }

        torch.save(checkpoint, path)

        if self.verbose and not is_best:  # Don't double-log for best.pt
            logger.info(f"ModelCheckpoint: Saved checkpoint to {path.name}")


# =============================================================================
# TENSORBOARD LOGGER
# =============================================================================


class TensorBoardLogger(Callback):
    """
    Log metrics to TensorBoard.

    Automatically logs all metrics to TensorBoard for visualization.

    Args:
        log_dir: Directory for TensorBoard logs
        flush_secs: How often to flush logs (seconds)

    Usage:
        >>> tb_logger = TensorBoardLogger(log_dir="logs/tensorboard")
        >>> trainer = Trainer(..., callbacks=[tb_logger])
        >>> # View with: tensorboard --logdir=logs/tensorboard
    """

    def __init__(self, log_dir: Path, flush_secs: int = 120):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.flush_secs = flush_secs
        self.writer = None
        self._tensorboard_available = _is_tensorboard_available()

        # Only create log directory if TensorBoard is available
        # This prevents empty directories when TensorBoard is not installed
        if self._tensorboard_available:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        else:
            logger.warning(
                "TensorBoardLogger: tensorboard not installed. "
                "Directory will not be created. Install with: pip install tensorboard"
            )

    def on_train_begin(self, trainer):
        """Initialize TensorBoard writer."""
        if not self._tensorboard_available:
            # Already warned in __init__, just return silently
            return

        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=self.flush_secs)
            logger.info(f"TensorBoardLogger: Logging to {self.log_dir}")
        except ImportError:
            # Should not happen since we checked in __init__, but handle gracefully
            logger.warning(
                "TensorBoardLogger: tensorboard not installed. "
                "Install with: pip install tensorboard"
            )
            self.writer = None

    def on_epoch_end(self, trainer, epoch: int, metrics: dict[str, float]):
        """Log metrics to TensorBoard."""
        if self.writer is None:
            return

        for key, value in metrics.items():
            # Skip non-numeric values
            if not isinstance(value, (int, float)):
                continue

            # Skip special keys
            if key in ["is_best", "epoch_time"]:
                continue

            try:
                self.writer.add_scalar(key, value, epoch)
            except Exception as e:
                logger.warning(f"TensorBoardLogger: Failed to log {key}: {e}")

        # Log learning rate
        if hasattr(trainer, "optimizer") and trainer.optimizer:
            for i, param_group in enumerate(trainer.optimizer.param_groups):
                lr = param_group["lr"]
                self.writer.add_scalar(f"learning_rate/group_{i}", lr, epoch)

    def on_train_end(self, trainer):
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
            logger.info("TensorBoardLogger: Closed writer")


# =============================================================================
# LEARNING RATE MONITOR
# =============================================================================


class LearningRateMonitor(Callback):
    """
    Monitor and log learning rate.

    Logs the current learning rate at each epoch.

    Args:
        log_to_console: Whether to log to console

    Usage:
        >>> lr_monitor = LearningRateMonitor()
        >>> trainer = Trainer(..., callbacks=[lr_monitor])
    """

    def __init__(self, log_to_console: bool = True):
        super().__init__()
        self.log_to_console = log_to_console

    def on_epoch_end(self, trainer, epoch: int, metrics: dict[str, float]):
        """Log learning rate."""
        if not hasattr(trainer, "optimizer") or trainer.optimizer is None:
            return

        lrs = []
        for i, param_group in enumerate(trainer.optimizer.param_groups):
            lr = param_group["lr"]
            lrs.append(lr)

            if self.log_to_console:
                if len(trainer.optimizer.param_groups) > 1:
                    logger.info(f"Epoch {epoch:3d} | lr[group_{i}] = {lr:.2e}")
                else:
                    logger.info(f"Epoch {epoch:3d} | lr = {lr:.2e}")


# =============================================================================
# PROGRESS BAR CALLBACK
# =============================================================================


class ProgressBar(Callback):
    """
    Display training progress with metrics.

    Simple progress tracking without external dependencies.

    Args:
        update_frequency: How often to update (in epochs)

    Usage:
        >>> progress = ProgressBar()
        >>> trainer = Trainer(..., callbacks=[progress])
    """

    def __init__(self, update_frequency: int = 1):
        super().__init__()
        self.update_frequency = update_frequency
        self.start_time = None

    def on_train_begin(self, trainer):
        """Record start time."""
        import time

        self.start_time = time.time()
        logger.info("=" * 70)
        logger.info("Training Progress")
        logger.info("=" * 70)

    def on_epoch_end(self, trainer, epoch: int, metrics: dict[str, float]):
        """Display progress."""
        if epoch % self.update_frequency != 0:
            return

        import time

        elapsed = time.time() - self.start_time
        progress = (epoch + 1) / trainer.max_epochs * 100

        # Build metrics string
        metrics_str = []
        if "train_loss" in metrics:
            metrics_str.append(f"train_loss: {metrics['train_loss']:.4f}")
        if "val_loss" in metrics:
            metrics_str.append(f"val_loss: {metrics['val_loss']:.4f}")

        logger.info(
            f"Progress: {progress:5.1f}% | "
            f"Epoch {epoch + 1:3d}/{trainer.max_epochs} | "
            f"{' | '.join(metrics_str)} | "
            f"Time: {elapsed:.1f}s"
        )

    def on_train_end(self, trainer):
        """Display completion message."""
        import time

        elapsed = time.time() - self.start_time
        logger.info("=" * 70)
        logger.info(f"Training completed in {elapsed:.2f}s ({elapsed / 60:.2f}min)")
        logger.info("=" * 70)


# =============================================================================
# GRADIENT MONITOR
# =============================================================================


class GradientMonitor(Callback):
    """
    Monitor gradient statistics.

    Tracks gradient norms to detect vanishing/exploding gradients.

    Args:
        log_frequency: Log every N epochs

    Usage:
        >>> grad_monitor = GradientMonitor(log_frequency=10)
        >>> trainer = Trainer(..., callbacks=[grad_monitor])
    """

    def __init__(self, log_frequency: int = 10):
        super().__init__()
        self.log_frequency = log_frequency

    def on_epoch_end(self, trainer, epoch: int, metrics: dict[str, float]):
        """Log gradient statistics."""
        if epoch % self.log_frequency != 0:
            return

        total_norm = 0.0
        num_params = 0

        for param in trainer.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                num_params += 1

        if num_params > 0:
            total_norm = total_norm**0.5
            logger.info(
                f"Epoch {epoch:3d} | "
                f"Gradient norm: {total_norm:.4f} | "
                f"Params with grad: {num_params}"
            )


# =============================================================================
# CALLBACK FACTORY
# =============================================================================


class CallbackFactory:
    """
    Factory for creating callbacks from configuration.

    Provides a centralized, DYNAMIC way to create callbacks from config dictionaries.
    Supports all callback types defined in this module with automatic parameter filtering.

    DYNAMIC: Supports all callback types via registry lookup
    PRODUCTION-READY: Handles path resolution, enabled flags, parameter validation
    FUTURE-PROOF: New callbacks auto-available when added to _callback_registry

    Usage:
        >>> from milia_pipeline.models.training import CallbackFactory
        >>> callbacks = CallbackFactory.from_config(
        ...     callback_config=training_config.get('callbacks', {}),
        ...     working_root_dir=Path('/path/to/working/dir')
        ... )
    """

    # Registry mapping config keys to callback classes
    _callback_registry = {
        "early_stopping": EarlyStopping,
        "model_checkpoint": ModelCheckpoint,
        "tensorboard": TensorBoardLogger,
        "lr_monitor": LearningRateMonitor,
        "progress_bar": ProgressBar,
        "gradient_monitor": GradientMonitor,
    }

    # Parameters that need path resolution (value is the default subdirectory)
    _path_params = {
        "model_checkpoint": {"dirpath": "checkpoints"},
        "tensorboard": {"log_dir": "tensorboard_logs"},
    }

    @classmethod
    def from_config(
        cls,
        callback_config: dict[str, Any],
        working_root_dir: Path,
        callback_logger: logging.Logger | None = None,
    ) -> list[Callback]:
        """
        Create callbacks from configuration dictionary.

        Args:
            callback_config: Dictionary with callback configurations.
                Expected structure:
                {
                    'early_stopping': {'enabled': True, 'params': {...}},
                    'model_checkpoint': {'enabled': True, 'params': {...}},
                    ...
                }
            working_root_dir: Base directory for auto-generated paths
            callback_logger: Optional logger for debug messages

        Returns:
            List of instantiated callback objects

        Example:
            >>> callbacks = CallbackFactory.from_config(
            ...     callback_config={
            ...         'early_stopping': {
            ...             'enabled': True,
            ...             'params': {'patience': 10, 'monitor': 'val_loss'}
            ...         },
            ...         'model_checkpoint': {
            ...             'enabled': True,
            ...             'params': {'save_top_k': 3, 'dirpath': None}
            ...         }
            ...     },
            ...     working_root_dir=Path('/data/experiment')
            ... )
        """
        log = callback_logger or logger
        callbacks = []
        working_root_dir = Path(working_root_dir)

        for callback_name, callback_class in cls._callback_registry.items():
            config = callback_config.get(callback_name, {})

            # Check if callback is enabled (default: False for safety)
            # Exception: early_stopping and model_checkpoint default to True for backward compat
            default_enabled = callback_name in ("early_stopping", "model_checkpoint")
            if not config.get("enabled", default_enabled):
                log.debug(f"CallbackFactory: {callback_name} is disabled")
                continue

            # Skip TensorBoardLogger if tensorboard is not installed
            # This prevents creating empty directories for unavailable optional dependencies
            if callback_name == "tensorboard" and not _is_tensorboard_available():
                log.info(
                    "CallbackFactory: Skipping TensorBoardLogger - tensorboard not installed. "
                    "Install with: pip install tensorboard"
                )
                continue

            # Get parameters
            params = config.get("params", {}).copy()

            # Resolve path parameters
            if callback_name in cls._path_params:
                for param_name, default_subdir in cls._path_params[callback_name].items():
                    param_value = params.get(param_name)
                    if param_value is None:
                        # Auto-generate path under working_root_dir
                        resolved_path = working_root_dir / default_subdir
                        # Only create directory here for non-optional callbacks
                        # TensorBoardLogger handles its own directory creation
                        # after verifying tensorboard availability
                        if callback_name != "tensorboard":
                            resolved_path.mkdir(parents=True, exist_ok=True)
                        params[param_name] = resolved_path
                        log.debug(
                            f"CallbackFactory: Auto-generated {param_name} for {callback_name}: "
                            f"{resolved_path}"
                        )
                    else:
                        # Use provided path, expand user (~)
                        params[param_name] = Path(param_value).expanduser()

            # Filter parameters to only those accepted by the callback class
            filtered_params = cls._filter_params(callback_class, params)

            try:
                callback = callback_class(**filtered_params)
                callbacks.append(callback)
                log.debug(
                    f"CallbackFactory: Created {callback_name} with params: {filtered_params}"
                )
            except Exception as e:
                log.error(f"CallbackFactory: Failed to create {callback_name}: {e}")
                raise

        log.info(f"CallbackFactory: Created {len(callbacks)} callbacks")
        return callbacks

    @classmethod
    def _filter_params(cls, callback_class: type, params: dict[str, Any]) -> dict[str, Any]:
        """
        Filter parameters to only those accepted by the callback constructor.

        Uses inspect.signature() for dynamic introspection.
        """
        import inspect

        if not params:
            return {}

        try:
            sig = inspect.signature(callback_class.__init__)
            valid_param_names = set(sig.parameters.keys()) - {"self"}
            filtered = {k: v for k, v in params.items() if k in valid_param_names}

            # Log ignored params
            ignored = set(params.keys()) - set(filtered.keys())
            if ignored:
                logger.debug(
                    f"CallbackFactory: Ignored unsupported params for {callback_class.__name__}: {ignored}"
                )

            return filtered
        except (ValueError, TypeError):
            return params

    @classmethod
    def list_available(cls) -> list[str]:
        """List all available callback types."""
        return sorted(cls._callback_registry.keys())

    @classmethod
    def get_callback_class(cls, name: str) -> type:
        """Get callback class by name."""
        if name not in cls._callback_registry:
            available = ", ".join(cls.list_available())
            raise ValueError(f"Unknown callback: '{name}'. Available: {available}")
        return cls._callback_registry[name]

    @classmethod
    def register_custom_callback(
        cls,
        name: str,
        callback_class: type,
        path_params: dict[str, str] | None = None,
        overwrite: bool = False,
    ):
        """
        Register a custom callback type.

        Args:
            name: Config key name for the callback
            callback_class: Callback class (must be Callback subclass)
            path_params: Dict mapping param names to default subdirectories
            overwrite: Whether to overwrite existing callback with same name
        """
        if not issubclass(callback_class, Callback):
            raise TypeError(
                f"callback_class must be subclass of Callback, got {type(callback_class)}"
            )

        if name in cls._callback_registry and not overwrite:
            raise ValueError(
                f"Callback '{name}' already registered. Use overwrite=True to replace."
            )

        cls._callback_registry[name] = callback_class
        if path_params:
            cls._path_params[name] = path_params

        logger.info(f"CallbackFactory: Registered custom callback '{name}'")


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info(
    f"callbacks module loaded - {len(CallbackFactory._callback_registry)} callback types available"
)
