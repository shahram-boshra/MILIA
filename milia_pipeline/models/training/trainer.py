"""
Trainer Module

Main training engine with comprehensive features:
- Training/validation/test loops
- Callback system integration
- Metric tracking and logging
- Checkpoint management with v2.0 format (Phase 1 Enhancement)
- Early stopping support
- Learning rate scheduling
- HPO (Hyperparameter Optimization) callback integration

Phase 1 Enhancement (v1.2.0):
- Enhanced checkpoint format with hyper_parameters for model recreation
- Data compatibility info (data_info) for inference validation
- Version info for checkpoint format versioning
- Backward compatible with v1.0 checkpoints

Author: milia Team
Version: 1.2.0
"""

from __future__ import annotations

import inspect
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

if TYPE_CHECKING:
    from .callbacks import Callback

# Import exceptions with fallback
try:
    from milia_pipeline.exceptions import CheckpointError, TrainingError
except ImportError:

    class TrainingError(Exception):
        """Exception raised during training."""

        pass

    class CheckpointError(Exception):
        """Exception raised for checkpoint issues."""

        pass


logger = logging.getLogger(__name__)


# =============================================================================
# TRAINER CLASS
# =============================================================================


class Trainer:
    """
    Main training engine for graph neural network models.

    Features:
    - Flexible training/validation/test loops
    - Callback system for extensibility
    - Comprehensive metric tracking
    - Automatic checkpoint saving
    - Progress logging
    - Early stopping support
    - Learning rate scheduling
    - Device management
    - HPO callback integration for hyperparameter optimization

    Usage:
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     loss_fn=nn.MSELoss(),
        ...     optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        ...     device=torch.device("cuda"),
        ...     max_epochs=100
        ... )
        >>> results = trainer.fit()

    HPO Usage:
        >>> from milia_pipeline.models.hpo.callbacks import OptunaPruningCallback
        >>> hpo_callback = OptunaPruningCallback(trial=optuna_trial, monitor="val_loss")
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     hpo_callback=hpo_callback,
        ...     ...
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
        loss_fn: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        device: torch.device | None = None,
        callbacks: list[Callback] | None = None,
        max_epochs: int = 100,
        log_every_n_steps: int = 50,
        checkpoint_dir: Path | None = None,
        gradient_clip_val: float | None = None,
        accumulate_grad_batches: int = 1,
        hpo_callback: Callback | None = None,
        model_info: dict[str, Any] | None = None,
        # NEW: Metrics for evaluation
        metrics: dict[str, nn.Module] | None = None,
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            test_loader: Test data loader (optional)
            loss_fn: Loss function
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler (optional)
            device: Device to train on (default: auto-detect)
            callbacks: List of callback objects (optional)
            max_epochs: Maximum number of training epochs
            log_every_n_steps: Log frequency
            checkpoint_dir: Directory for saving checkpoints (optional)
            gradient_clip_val: Gradient clipping value (optional)
            accumulate_grad_batches: Number of batches to accumulate gradients
            hpo_callback: HPO-specific callback for hyperparameter optimization (optional).
                         When provided by HPOManager, enables trial pruning and metric
                         reporting during optimization. This callback is automatically
                         appended to the callbacks list.
            model_info: Model metadata dictionary (optional). When provided, contains
                       model capability information used for intelligent forward pass:
                       - uses_edge_features: Whether model is configured to use edge features
                       - requires_edge_features: Whether model requires edge features
                       - detected_edge_params: Edge dimension parameters found in schema
                       If not provided, Trainer will try forward without edge_attr first.
            metrics: Dictionary mapping metric names to metric modules (optional).
                    When provided via MetricsRegistry.get_metrics_for_task(), enables
                    automatic computation of evaluation metrics (MSE, MAE, R2, etc.)
                    during validation and test. Metrics are moved to device automatically.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn or nn.MSELoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        self.max_epochs = max_epochs
        self.log_every_n_steps = log_every_n_steps
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches

        # NEW: Store metrics for evaluation
        # DYNAMIC: Accepts any dict of metric modules
        # PRODUCTION-READY: Handles None gracefully
        # FUTURE-PROOF: Metrics moved to device with model
        self.metrics = metrics or {}

        # Store HPO callback reference (set by HPOManager)
        self.hpo_callback = hpo_callback

        # Store model metadata for intelligent forward pass
        self.model_info = model_info or {}
        self._uses_edge_features = self.model_info.get("uses_edge_features", None)

        # Store task type for intelligent target handling (Phase 2: edge-level tasks)
        self._task_type = self.model_info.get("task_type", None)

        # Store classification flag for target reshape logic
        # Classification tasks don't need reshape - target is [batch] of class indices
        self._is_classification_task = self.model_info.get("is_classification", False)

        # Store out_channels for dynamic target reshaping (graph-level multi-target tasks)
        # When PyG batches graphs with multi-target y, it flattens [batch, targets] to [batch*targets]
        # This value is used in _get_target() to reshape back to [batch, targets]
        self._out_channels = self.model_info.get("out_channels", 1)

        # ====================================================================
        # TARGET SELECTION INITIALIZATION
        # ====================================================================
        # DYNAMIC: Works with any selection from model_info
        # PRODUCTION-READY: Handles None gracefully
        # FUTURE-PROOF: Selection info propagated via model_info
        # ====================================================================
        self._target_selection = self.model_info.get("target_selection", None)
        self._target_indices = None
        self._original_out_channels = self.model_info.get("original_out_channels", None)

        if self._target_selection is not None:
            self._target_indices = self._target_selection.get("resolved_indices", None)
            logger.info(
                f"Trainer initialized with target selection: "
                f"indices={self._target_indices}, "
                f"names={self._target_selection.get('resolved_names', 'N/A')}, "
                f"out_channels={self._out_channels}"
            )

        # If HPO callback provided, add to callbacks list
        if self.hpo_callback is not None:
            self.callbacks.append(self.hpo_callback)
            logger.debug(f"HPO callback added: {self.hpo_callback.__class__.__name__}")

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Move model to device
        self.model = self.model.to(self.device)
        logger.info(f"Model moved to device: {self.device}")

        # NEW: Move metrics to device
        for name, metric in self.metrics.items():
            if hasattr(metric, "to"):
                self.metrics[name] = metric.to(self.device)
        if self.metrics:
            logger.info(f"Metrics moved to device: {list(self.metrics.keys())}")

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.metrics_history = defaultdict(list)
        self.training_time = 0.0

        # Validate configuration
        self._validate_configuration()

        # Initialize callbacks
        for callback in self.callbacks:
            callback.set_trainer(self)

        # Create checkpoint directory if specified
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Checkpoint directory: {self.checkpoint_dir}")

        logger.info(
            f"Trainer initialized - "
            f"Epochs: {self.max_epochs}, Device: {self.device}, "
            f"Callbacks: {len(self.callbacks)}"
            f"{', HPO: enabled' if self.hpo_callback else ''}"
        )

    def _validate_configuration(self):
        """Validate trainer configuration."""
        if self.optimizer is None:
            raise TrainingError(
                "Optimizer is required for training. Please provide optimizer argument."
            )

        if self.train_loader is None:
            raise TrainingError("Training data loader is required.")

        if self.accumulate_grad_batches < 1:
            raise TrainingError(
                f"accumulate_grad_batches must be >= 1, got {self.accumulate_grad_batches}"
            )

    def fit(self) -> dict[str, Any]:
        """
        Main training loop.

        Executes the complete training process:
        1. Calls on_train_begin callbacks
        2. Loops through epochs
        3. Trains for one epoch
        4. Validates (if validation loader provided)
        5. Updates learning rate scheduler
        6. Calls on_epoch_end callbacks
        7. Checks for early stopping
        8. Calls on_train_end callbacks
        9. Evaluates on test set (if test loader provided)

        Returns:
            Dictionary containing:
                - 'train_metrics': Training metrics history
                - 'val_metrics': Validation metrics (if val_loader provided)
                - 'test_metrics': Test metrics (if test_loader provided)
                - 'training_time': Total training time in seconds
                - 'best_epoch': Epoch with best validation loss

        Raises:
            TrainingError: If training fails

        Example:
            >>> results = trainer.fit()
            >>> print(f"Best val loss: {min(results['train_metrics']['val_loss'])}")
        """
        logger.info("=" * 70)
        logger.info("Starting training")
        logger.info("=" * 70)

        start_time = datetime.now()

        try:
            self._on_train_begin()

            for epoch in range(self.max_epochs):
                self.current_epoch = epoch

                epoch_start = datetime.now()

                # Training epoch
                train_metrics = self._train_epoch()

                # Validation epoch
                val_metrics = {}
                if self.val_loader is not None:
                    val_metrics = self._validate_epoch()

                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}

                # Track best validation loss
                if "val_loss" in val_metrics:
                    if val_metrics["val_loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["val_loss"]
                        epoch_metrics["is_best"] = True
                    else:
                        epoch_metrics["is_best"] = False

                # Update metrics history
                for key, value in epoch_metrics.items():
                    if key != "is_best":
                        self.metrics_history[key].append(value)

                # Update learning rate
                if self.scheduler is not None:
                    self._update_scheduler(val_metrics, train_metrics)

                # Epoch time
                epoch_time = (datetime.now() - epoch_start).total_seconds()
                epoch_metrics["epoch_time"] = epoch_time

                # Log epoch summary
                self._log_epoch_summary(epoch, epoch_metrics)

                # Callbacks
                self._on_epoch_end(epoch_metrics)

                # Check early stopping
                if self._should_stop():
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

            self._on_train_end()

            # Calculate total training time
            self.training_time = (datetime.now() - start_time).total_seconds()

            # Test evaluation
            test_metrics = {}
            if self.test_loader is not None:
                logger.info("Evaluating on test set...")
                test_metrics = self.test()

            # Find best epoch
            best_epoch = None
            if "val_loss" in self.metrics_history:
                best_epoch = int(torch.argmin(torch.tensor(self.metrics_history["val_loss"])))

            logger.info("=" * 70)
            logger.info(f"Training completed in {self.training_time:.2f}s")
            if best_epoch is not None:
                logger.info(f"Best validation loss at epoch {best_epoch}: {self.best_val_loss:.6f}")
            logger.info("=" * 70)

            return {
                "train_metrics": dict(self.metrics_history),
                "test_metrics": test_metrics,
                "training_time": self.training_time,
                "best_epoch": best_epoch,
                "best_val_loss": self.best_val_loss if best_epoch is not None else None,
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            self._on_train_end()
            raise TrainingError(f"Training failed: {e}") from e

    def _train_epoch(self) -> dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary with training metrics (train_loss, etc.)
        """
        logger.debug("[DIAGNOSTIC] _train_epoch: Starting")
        logger.debug("[DIAGNOSTIC] _train_epoch: Calling model.train()")
        self.model.train()
        logger.debug("[DIAGNOSTIC] _train_epoch: model.train() completed")
        epoch_loss = 0.0
        num_batches = 0

        # Zero gradients at start
        logger.debug("[DIAGNOSTIC] _train_epoch: Calling optimizer.zero_grad()")
        self.optimizer.zero_grad()
        logger.debug("[DIAGNOSTIC] _train_epoch: optimizer.zero_grad() completed")

        logger.debug("[DIAGNOSTIC] _train_epoch: Starting DataLoader iteration")
        try:
            for batch_idx, batch in enumerate(self.train_loader):
                logger.debug(f"[DIAGNOSTIC] _train_epoch: Processing batch {batch_idx}")
            try:
                # Move batch to device
                logger.debug("[DIAGNOSTIC] _train_epoch: Moving batch to device")
                batch = batch.to(self.device)
                logger.debug(
                    "[DIAGNOSTIC] _train_epoch: Batch moved to device, calling _forward_pass"
                )

                # Forward pass
                out = self._forward_pass(batch)
                # Get appropriate target based on task type
                target = self._get_target(batch)
                loss = self.loss_fn(out, target)

                # Scale loss for gradient accumulation
                loss = loss / self.accumulate_grad_batches

                # Backward pass
                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                    # Gradient clipping
                    if self.gradient_clip_val is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip_val
                        )

                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Track metrics
                epoch_loss += loss.item() * self.accumulate_grad_batches
                num_batches += 1
                self.global_step += 1

                # Logging
                if batch_idx % self.log_every_n_steps == 0:
                    logger.debug(
                        f"Epoch {self.current_epoch} | "
                        f"Batch {batch_idx}/{len(self.train_loader)} | "
                        f"Loss: {loss.item() * self.accumulate_grad_batches:.6f}"
                    )

            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                raise TrainingError(f"Training batch failed: {e}") from e
        except Exception as dataloader_error:
            # Catch errors that happen during DataLoader iteration (collation, etc.)
            import traceback

            logger.error(f"DataLoader iteration error: {dataloader_error}")
            logger.debug(f"[DIAGNOSTIC] Full traceback:\n{traceback.format_exc()}")
            raise

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        return {"train_loss": avg_loss}

    @torch.no_grad()
    def _validate_epoch(self) -> dict[str, float]:
        """
        Validate for one epoch.

        UPDATED: Now computes configured metrics alongside loss.

        DYNAMIC: Accumulates predictions/targets for proper metric computation
        PRODUCTION-READY: Uses TorchMetrics accumulation for correct epoch-level metrics
        FUTURE-PROOF: Metrics computed via MetricsRegistry-provided modules

        Returns:
            Dictionary with validation metrics (val_loss, val_mse, val_mae, etc.)
        """
        self.model.eval()
        epoch_loss = 0.0
        num_batches = 0

        # Collect all predictions and targets for metric computation
        all_preds = []
        all_targets = []

        # Reset metrics at start of epoch
        for metric in self.metrics.values():
            if hasattr(metric, "reset"):
                metric.reset()

        for batch in self.val_loader:
            try:
                batch = batch.to(self.device)
                out = self._forward_pass(batch)
                # Get appropriate target based on task type
                target = self._get_target(batch)
                loss = self.loss_fn(out, target)
                epoch_loss += loss.item()
                num_batches += 1

                # Collect predictions and targets for metrics
                all_preds.append(out.detach())
                all_targets.append(target.detach())

                # Update metrics incrementally (TorchMetrics pattern)
                # Convert target to appropriate dtype for classification metrics
                metric_target = self._prepare_target_for_metrics(target)
                for metric in self.metrics.values():
                    if hasattr(metric, "update"):
                        metric.update(out, metric_target)

            except Exception as e:
                logger.warning(f"Error in validation batch: {e}")
                continue

        avg_loss = epoch_loss / num_batches if num_batches > 0 else float("inf")
        results = {"val_loss": avg_loss}

        # Compute final metric values
        for name, metric in self.metrics.items():
            try:
                if hasattr(metric, "compute"):
                    # TorchMetrics pattern
                    value = metric.compute()
                    if hasattr(value, "item"):
                        value = value.item()
                    results[f"val_{name}"] = value
                elif all_preds and all_targets:
                    # Fallback: compute on concatenated tensors
                    preds = torch.cat(all_preds, dim=0)
                    targets = torch.cat(all_targets, dim=0)
                    # Convert target dtype for classification metrics
                    targets = self._prepare_target_for_metrics(targets)
                    value = metric(preds, targets)
                    if hasattr(value, "item"):
                        value = value.item()
                    results[f"val_{name}"] = value
            except Exception as e:
                logger.warning(f"Error computing metric '{name}': {e}")
                continue

        return results

    @torch.no_grad()
    def test(self) -> dict[str, float]:
        """
        Evaluate on test set.

        UPDATED: Now computes configured metrics alongside loss.

        DYNAMIC: Works with any metrics from MetricsRegistry
        PRODUCTION-READY: Proper accumulation and logging
        FUTURE-PROOF: Metrics automatically included in results

        Returns:
            Dictionary with test metrics (test_loss, test_mse, test_mae, etc.)

        Example:
            >>> test_metrics = trainer.test()
            >>> print(f"Test loss: {test_metrics['test_loss']:.6f}")
            >>> print(f"Test MAE: {test_metrics['test_mae']:.6f}")
        """
        if self.test_loader is None:
            logger.warning("No test loader provided, skipping test evaluation")
            return {}

        self.model.eval()
        test_loss = 0.0
        num_batches = 0

        # Collect all predictions and targets for metric computation
        all_preds = []
        all_targets = []

        # Reset metrics at start of test
        for metric in self.metrics.values():
            if hasattr(metric, "reset"):
                metric.reset()

        for batch in self.test_loader:
            try:
                batch = batch.to(self.device)
                out = self._forward_pass(batch)
                # Get appropriate target based on task type
                target = self._get_target(batch)
                loss = self.loss_fn(out, target)
                test_loss += loss.item()
                num_batches += 1

                # Collect predictions and targets for metrics
                all_preds.append(out.detach())
                all_targets.append(target.detach())

                # Update metrics incrementally (TorchMetrics pattern)
                # Convert target to appropriate dtype for classification metrics
                metric_target = self._prepare_target_for_metrics(target)
                for metric in self.metrics.values():
                    if hasattr(metric, "update"):
                        metric.update(out, metric_target)

            except Exception as e:
                logger.warning(f"Error in test batch: {e}")
                continue

        avg_loss = test_loss / num_batches if num_batches > 0 else float("inf")
        results = {"test_loss": avg_loss}

        # Compute final metric values
        for name, metric in self.metrics.items():
            try:
                if hasattr(metric, "compute"):
                    # TorchMetrics pattern
                    value = metric.compute()
                    if hasattr(value, "item"):
                        value = value.item()
                    results[f"test_{name}"] = value
                elif all_preds and all_targets:
                    # Fallback: compute on concatenated tensors
                    preds = torch.cat(all_preds, dim=0)
                    targets = torch.cat(all_targets, dim=0)
                    # Convert target dtype for classification metrics
                    targets = self._prepare_target_for_metrics(targets)
                    value = metric(preds, targets)
                    if hasattr(value, "item"):
                        value = value.item()
                    results[f"test_{name}"] = value
            except Exception as e:
                logger.warning(f"Error computing metric '{name}': {e}")
                continue

        # Log all results
        logger.info(f"Test results: {results}")

        return results

    # =========================================================================
    # DYNAMIC FORWARD SIGNATURE INTROSPECTION
    # =========================================================================
    # DYNAMIC: Introspects model.forward() at runtime to determine parameters
    # PRODUCTION-READY: Caches signature for performance, handles edge cases
    # FUTURE-PROOF: Works with ANY PyG model regardless of forward signature
    # =========================================================================

    def _get_forward_signature_params(self) -> list[str]:
        """
        Get the parameter names of the model's forward method.

        Uses Python's inspect.signature to dynamically introspect the forward
        method. This enables support for models with non-standard signatures
        like SchNet(z, pos, batch) or DimeNet(z, pos, batch).

        Results are cached in self._forward_params for performance.

        Returns:
            List of parameter names (excluding 'self')

        Example:
            >>> # For SchNet: forward(self, z, pos, batch=None)
            >>> params = trainer._get_forward_signature_params()
            >>> print(params)
            ['z', 'pos', 'batch']

            >>> # For GCN: forward(self, x, edge_index, edge_weight=None, batch=None)
            >>> params = trainer._get_forward_signature_params()
            >>> print(params)
            ['x', 'edge_index', 'edge_weight', 'batch']
        """
        # Return cached result if available
        if hasattr(self, "_forward_params") and self._forward_params is not None:
            return self._forward_params

        try:
            # Get the actual model (unwrap wrapper if needed)
            model = self.model

            # Handle wrapper models (GraphLevelModelWrapper, EdgeLevelModelWrapper)
            if hasattr(model, "model"):
                inner_model = model.model
                logger.debug(f"Unwrapped model: {type(inner_model).__name__}")
            else:
                inner_model = model
                logger.debug(f"Using model directly: {type(inner_model).__name__}")

            # Get forward method signature
            sig = inspect.signature(inner_model.forward)
            params = [name for name in sig.parameters if name != "self"]

            # Cache the result
            self._forward_params = params

            # Log at INFO level for visibility during debugging
            if params:
                logger.info(
                    f"Introspected forward signature for {type(inner_model).__name__}: {params}"
                )
            else:
                logger.warning(
                    f"Empty forward signature introspected for {type(inner_model).__name__}"
                )

            return params

        except Exception as e:
            logger.warning(f"Could not introspect forward signature: {e}")
            self._forward_params = None
            return []

    def _model_accepts_3d_params(self) -> bool:
        """
        Check if the model can accept 3D molecular parameters (z, pos).

        Only certain models can handle z (atomic numbers) and pos (positions):
        - ParallelEnsemble: Has signature-aware forwarding to inner models
        - 3D models (SchNet, DimeNet): Have z and pos as explicit parameters

        Standard GNN models (GCN, GAT, GraphSAGE) wrapped in GraphLevelModelWrapper
        do NOT accept z/pos and will raise TypeError if passed.

        CRITICAL: GraphLevelModelWrapper has **kwargs but just passes them to
        the inner model. If the inner model doesn't accept z/pos, the call fails.

        DYNAMIC: Checks model signature and type at runtime
        PRODUCTION-READY: Caches result for performance
        FUTURE-PROOF: Works with any model that has z/pos in signature

        Returns:
            True if model can accept z and pos parameters
        """
        # Return cached result if available
        if hasattr(self, "_accepts_3d_params"):
            return self._accepts_3d_params

        try:
            forward_params = self._get_forward_signature_params()

            # Check if inner model explicitly has z and pos parameters
            # This is the ONLY reliable check - if the innermost model has z/pos,
            # then we can pass them. Otherwise, we cannot.
            has_explicit_3d = "z" in forward_params and "pos" in forward_params

            # Check if model is an ensemble (ParallelEnsemble, StackingEnsemble, etc.)
            # or composition (HierarchicalComposition, SequentialStack)
            # These have special handling in their forward() to dispatch z/pos
            # to inner models that need them via signature-aware calling
            is_ensemble_or_composition = False
            model = self.model
            model_class_name = type(model).__name__

            # Check for ensemble/composition patterns in class name
            # Covers: ParallelEnsemble, StackingEnsemble, HierarchicalComposition, SequentialStack
            ensemble_patterns = ("Ensemble", "Composition", "Stack")

            # Check outer model first
            if any(pattern in model_class_name for pattern in ensemble_patterns):
                is_ensemble_or_composition = True
            # Check inner model if wrapper
            elif hasattr(model, "model"):
                inner_model = model.model
                inner_class_name = type(inner_model).__name__
                if any(pattern in inner_class_name for pattern in ensemble_patterns):
                    is_ensemble_or_composition = True

            # Model accepts 3D params if:
            # 1. Inner model explicitly has z and pos in signature (SchNet, DimeNet), OR
            # 2. Model is an ensemble/composition (has signature-aware dispatching)
            self._accepts_3d_params = has_explicit_3d or is_ensemble_or_composition

            logger.debug(
                f"Model 3D param check: explicit_3d={has_explicit_3d}, "
                f"is_ensemble_or_composition={is_ensemble_or_composition}, result={self._accepts_3d_params}"
            )

            return self._accepts_3d_params

        except Exception as e:
            logger.debug(f"Could not check 3D param acceptance: {e}")
            # Default to False (safer - don't pass unknown params)
            self._accepts_3d_params = False
            return False

    def _forward_with_dynamic_signature(self, batch: Data | Batch) -> torch.Tensor | None:
        """
        Execute forward pass using dynamically introspected signature.

        Maps batch attributes to forward parameters based on introspected
        signature. This enables support for models with non-standard signatures
        like SchNet(z, pos, batch) or DimeNet(z, pos, batch).

        DYNAMIC: Uses inspect.signature to determine required parameters
        PRODUCTION-READY: Gracefully returns None if dynamic call fails
        FUTURE-PROOF: Works with any model forward signature

        Args:
            batch: PyG Data or Batch object

        Returns:
            Model output tensor, or None if dynamic forward fails

        Example:
            >>> # For SchNet with batch containing z, pos, batch attributes
            >>> out = trainer._forward_with_dynamic_signature(batch)
            >>> # Calls: model(z=batch.z, pos=batch.pos, batch=batch.batch)
        """
        forward_params = self._get_forward_signature_params()

        if not forward_params:
            logger.debug(
                "Dynamic forward: No forward parameters introspected, skipping dynamic approach"
            )
            return None

        # Build kwargs by matching batch attributes to forward parameters
        kwargs = {}

        # Map common parameter name variations
        # Key: forward param name -> Value: list of batch attribute names to try
        param_mappings = {
            "z": ["z", "atomic_numbers"],
            "pos": ["pos", "positions", "coords"],
            "batch": ["batch"],
            "x": ["x", "node_features"],
            "edge_index": ["edge_index"],
            "edge_attr": ["edge_attr", "edge_features"],
            "edge_weight": ["edge_weight", "edge_attr"],
            "edge_label_index": ["edge_label_index"],
        }

        # Determine if edge_attr should be included based on model_info
        # If uses_edge_features is explicitly False, skip edge_attr even if
        # the model signature accepts it (avoids passing edge_attr to models
        # like ensembles whose inner models don't support it)
        skip_edge_attr = self._uses_edge_features is False

        for param in forward_params:
            # Skip edge_attr if model doesn't use edge features
            if skip_edge_attr and param in ("edge_attr", "edge_weight"):
                continue

            # Check if parameter has known mappings
            if param in param_mappings:
                for attr_name in param_mappings[param]:
                    if hasattr(batch, attr_name):
                        value = getattr(batch, attr_name)
                        if value is not None:
                            kwargs[param] = value
                            break
            else:
                # Direct attribute match
                if hasattr(batch, param):
                    value = getattr(batch, param)
                    if value is not None:
                        kwargs[param] = value

        # CRITICAL: Only pass z and pos if the model can actually accept them.
        # This enables heterogeneous ensembles where the outer model (ParallelEnsemble)
        # has signature (x, edge_index, ...) but inner models (SchNet) need (z, pos, batch).
        # Standard GNN models (GCN, GAT, GraphSAGE) do NOT accept z/pos and will error.
        #
        # FIX 20: Use _model_accepts_3d_params() to check before adding z/pos
        # BEFORE: Unconditionally added z/pos, causing errors for non-3D models
        # AFTER: Only add z/pos if model is ensemble OR has z/pos in signature
        if self._model_accepts_3d_params():
            if "z" not in kwargs and hasattr(batch, "z") and batch.z is not None:
                kwargs["z"] = batch.z
            if "pos" not in kwargs and hasattr(batch, "pos") and batch.pos is not None:
                kwargs["pos"] = batch.pos

        # Verify we have all required parameters
        try:
            sig = inspect.signature(
                self.model.forward if not hasattr(self.model, "model") else self.model.model.forward
            )
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                # Skip variadic parameters (*args, **kwargs) - they are not required
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                # Skip edge_attr check if we intentionally excluded it
                if skip_edge_attr and param_name in ("edge_attr", "edge_weight"):
                    continue
                # Check if required parameter is missing
                if param.default is inspect.Parameter.empty and param_name not in kwargs:
                    logger.debug(
                        f"Dynamic forward: Required parameter '{param_name}' not found in batch. "
                        f"kwargs has: {list(kwargs.keys())}"
                    )
                    return None
        except Exception as e:
            logger.warning(f"Dynamic forward: Signature validation failed: {e}")

        # Try the dynamic forward call
        try:
            # Log kwargs being passed to model (DEBUG level for production)
            logger.debug(f"[DIAGNOSTIC] Dynamic forward kwargs: {list(kwargs.keys())}")
            if "edge_attr" in kwargs:
                ea = kwargs["edge_attr"]
                logger.debug(
                    f"[DIAGNOSTIC] edge_attr in kwargs: shape={list(ea.shape) if ea is not None else None}"
                )
            else:
                logger.debug("[DIAGNOSTIC] edge_attr NOT in kwargs (correct for ensembles)")

            return self.model(**kwargs)
        except Exception as e:
            # =====================================================================
            # FIX 28: Dynamic Forward Fallback Logging
            # =====================================================================
            # ISSUE: Dynamic forward failure was logged at ERROR level for EVERY
            # batch, creating hundreds of misleading error logs when training
            # actually works fine via the fallback path.
            #
            # ROOT CAUSE: Models with variadic signatures (*args, **kwargs) like
            # GAE, VGAE, and wrappers (EdgeLevelModelWrapper) cannot have their
            # parameters matched by the dynamic introspection approach. The
            # introspected signature shows ['args', 'kwargs'] which doesn't map
            # to concrete batch attributes. The fallback to standard forward
            # strategies (positional arguments) handles these cases correctly.
            #
            # FIX: Log at DEBUG level (not ERROR) since this is expected fallback
            # behavior for models with variadic signatures. Only log once per
            # training session to reduce noise.
            #
            # DYNAMIC: Uses instance flag to track if already logged
            # PRODUCTION-READY: Reduces log noise while preserving diagnostics
            # FUTURE-PROOF: Works with any model signature pattern
            # =====================================================================
            if not getattr(self, "_dynamic_forward_fallback_logged", False):
                logger.debug(
                    f"Dynamic forward not applicable for {type(self.model).__name__}, "
                    f"using standard forward strategies. Reason: {e}"
                )
                self._dynamic_forward_fallback_logged = True
            return None

    def _forward_pass(self, batch: Data | Batch) -> torch.Tensor:
        """
        Execute forward pass through model.

        DYNAMIC: First tries introspected signature, then falls back to
        standard signatures. This enables support for ANY PyG model.

        Strategy order:
        1. Dynamic signature-based call (introspects model.forward())
        2. Standard GNN signature: model(x, edge_index, batch=batch)
        3. Edge feature models: model(x, edge_index, edge_attr, batch=batch)
        4. Batch object: model(batch)

        This approach handles:
        - Standard GNN models (GCN, GraphSAGE, GIN) - strategy 2
        - Edge feature models (NNConv, AttentiveFP) - strategy 3
        - 3D molecular models (SchNet, DimeNet) - strategy 1 (z, pos, batch)
        - Any model with non-standard signature - strategy 1
        - Link prediction tasks: passes edge_label_index when needed

        Args:
            batch: PyG Data or Batch object

        Returns:
            Model output tensor

        Raises:
            TrainingError: If all forward signature attempts fail
        """
        # =====================================================================
        # STRATEGY 0: DYNAMIC SIGNATURE-BASED FORWARD (PRIORITY)
        # =====================================================================
        # Try dynamic introspection first - handles SchNet, DimeNet, etc.
        # This is the most future-proof approach as it works with any signature
        # =====================================================================
        result = self._forward_with_dynamic_signature(batch)
        if result is not None:
            return result

        # =====================================================================
        # FALLBACK: STANDARD FORWARD STRATEGIES
        # =====================================================================
        # If dynamic signature fails, fall back to standard approaches
        # =====================================================================
        has_batch = hasattr(batch, "batch") and batch.batch is not None
        has_edge_attr = hasattr(batch, "edge_attr") and batch.edge_attr is not None
        has_edge_label_index = (
            hasattr(batch, "edge_label_index") and batch.edge_label_index is not None
        )

        # Determine whether to use edge features based on model configuration
        use_edge_features = self._should_use_edge_features(has_edge_attr)

        # Try the appropriate forward signature
        if use_edge_features and has_edge_attr:
            # Model is configured to use edge features
            return self._forward_with_edge_features(batch, has_batch, has_edge_label_index)
        else:
            # Model doesn't use edge features, or no edge_attr in data
            return self._forward_without_edge_features(
                batch, has_batch, has_edge_attr, has_edge_label_index
            )

    def _should_use_edge_features(self, has_edge_attr: bool) -> bool:
        """
        Determine whether to pass edge_attr to the model.

        Decision logic:
        1. If model_info says uses_edge_features=True, use them (if available)
        2. If model_info says uses_edge_features=False, don't use them
        3. If unknown (None), default to False (safer for most models)

        Args:
            has_edge_attr: Whether batch has edge_attr attribute

        Returns:
            True if edge features should be passed to model
        """
        if not has_edge_attr:
            # No edge features in data, can't use them
            return False

        if self._uses_edge_features is True:
            # Model is configured to use edge features
            return True
        elif self._uses_edge_features is False:
            # Model explicitly does not use edge features
            return False
        else:
            # Unknown (None) - default to not using edge features
            # This is the safer default as most models (GCN, GraphSAGE)
            # don't support multi-dimensional edge_attr
            logger.debug("No model_info provided, defaulting to not using edge features")
            return False

    def _forward_with_edge_features(
        self, batch: Data | Batch, has_batch: bool, has_edge_label_index: bool = False
    ) -> torch.Tensor:
        """
        Execute forward pass WITH edge features.

        Tries multiple signatures for models that use edge features:
        1. Positional: model(x, edge_index, edge_attr, batch=batch)
        2. Keyword: model(x, edge_index, edge_attr=edge_attr, batch=batch)
        3. Fallback without edge_attr (in case of misconfiguration)
        4. Final fallback: model(batch)

        For link prediction tasks (when edge_label_index exists):
        - Passes edge_label_index to model for correct edge decoding
        - EdgeLevelModelWrapper uses this to decode only labeled edges

        For heterogeneous ensembles (containing 3D models like SchNet):
        - Passes z and pos in kwargs so ensemble can forward to 3D models

        Args:
            batch: PyG Data or Batch object
            has_batch: Whether batch object has batch assignment
            has_edge_label_index: Whether batch has edge_label_index (link prediction)

        Returns:
            Model output tensor
        """
        # Build kwargs for optional parameters
        extra_kwargs = {}

        # Link prediction support
        if has_edge_label_index:
            extra_kwargs["edge_label_index"] = batch.edge_label_index

        # 3D model support for heterogeneous ensembles
        # Only pass z and pos to models that can accept them:
        # - ParallelEnsemble (has **kwargs in forward signature)
        # - 3D models like SchNet (have z and pos in forward signature)
        # Standard GNN models (GCN, GAT, etc.) do NOT accept z/pos and will error
        model_accepts_3d_params = self._model_accepts_3d_params()
        if model_accepts_3d_params:
            if hasattr(batch, "z") and batch.z is not None:
                extra_kwargs["z"] = batch.z
            if hasattr(batch, "pos") and batch.pos is not None:
                extra_kwargs["pos"] = batch.pos

        # Strategy 1: Positional edge_attr
        try:
            if has_batch:
                return self.model(
                    batch.x, batch.edge_index, batch.edge_attr, batch=batch.batch, **extra_kwargs
                )
            else:
                return self.model(batch.x, batch.edge_index, batch.edge_attr, **extra_kwargs)
        except Exception as e1:
            logger.debug(f"Forward with positional edge_attr failed: {e1}")

        # Strategy 2: Keyword edge_attr
        try:
            if has_batch:
                return self.model(
                    batch.x,
                    batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch,
                    **extra_kwargs,
                )
            else:
                return self.model(
                    batch.x, batch.edge_index, edge_attr=batch.edge_attr, **extra_kwargs
                )
        except Exception as e2:
            logger.debug(f"Forward with keyword edge_attr failed: {e2}")

        # Strategy 3: Fallback without edge_attr (misconfiguration recovery)
        try:
            if has_batch:
                return self.model(batch.x, batch.edge_index, batch=batch.batch, **extra_kwargs)
            else:
                return self.model(batch.x, batch.edge_index, **extra_kwargs)
        except Exception as e3:
            logger.debug(f"Forward without edge_attr fallback failed: {e3}")

        # Strategy 4: Final fallback - batch object
        try:
            return self.model(batch)
        except Exception as e4:
            raise TrainingError(
                f"Forward pass failed (model configured for edge features). "
                f"Tried: (1) positional edge_attr, (2) keyword edge_attr, "
                f"(3) no edge_attr, (4) batch object. "
                f"Final error: {e4}"
            ) from e4

    def _forward_without_edge_features(
        self,
        batch: Data | Batch,
        has_batch: bool,
        has_edge_attr: bool,
        has_edge_label_index: bool = False,
    ) -> torch.Tensor:
        """
        Execute forward pass WITHOUT edge features.

        Primary path for most GNN models (GCN, GraphSAGE, GIN, etc.)

        Tries:
        1. Basic: model(x, edge_index, batch=batch)
        2. With edge_attr as fallback (if model actually needs it)
        3. Final fallback: model(batch)

        For link prediction tasks (when edge_label_index exists):
        - Passes edge_label_index to model for correct edge decoding
        - EdgeLevelModelWrapper uses this to decode only labeled edges

        For heterogeneous ensembles (containing 3D models like SchNet):
        - Passes z and pos in kwargs so ensemble can forward to 3D models

        Args:
            batch: PyG Data or Batch object
            has_batch: Whether batch object has batch assignment
            has_edge_attr: Whether edge_attr exists in batch
            has_edge_label_index: Whether batch has edge_label_index (link prediction)

        Returns:
            Model output tensor
        """
        # Build kwargs for optional parameters
        extra_kwargs = {}

        # Link prediction support
        if has_edge_label_index:
            extra_kwargs["edge_label_index"] = batch.edge_label_index

        # 3D model support for heterogeneous ensembles
        # Only pass z and pos to models that can accept them:
        # - ParallelEnsemble (has **kwargs in forward signature)
        # - 3D models like SchNet (have z and pos in forward signature)
        # Standard GNN models (GCN, GAT, etc.) do NOT accept z/pos and will error
        model_accepts_3d_params = self._model_accepts_3d_params()
        if model_accepts_3d_params:
            if hasattr(batch, "z") and batch.z is not None:
                extra_kwargs["z"] = batch.z
            if hasattr(batch, "pos") and batch.pos is not None:
                extra_kwargs["pos"] = batch.pos

        # Strategy 1: Basic signature without edge_attr
        try:
            if has_batch:
                return self.model(batch.x, batch.edge_index, batch=batch.batch, **extra_kwargs)
            else:
                return self.model(batch.x, batch.edge_index, **extra_kwargs)
        except Exception as e1:
            logger.debug(f"Basic forward without edge_attr failed: {e1}")

        # Strategy 2: Try with edge_attr as fallback (model might need it despite metadata)
        if has_edge_attr:
            try:
                if has_batch:
                    return self.model(
                        batch.x,
                        batch.edge_index,
                        batch.edge_attr,
                        batch=batch.batch,
                        **extra_kwargs,
                    )
                else:
                    return self.model(batch.x, batch.edge_index, batch.edge_attr, **extra_kwargs)
            except Exception as e2:
                logger.debug(f"Forward with edge_attr fallback failed: {e2}")

        # Strategy 3: Final fallback - batch object
        try:
            return self.model(batch)
        except Exception as e3:
            raise TrainingError(
                f"Forward pass failed (model not using edge features). "
                f"Tried: (1) basic signature, "
                f"{'(2) with edge_attr fallback, ' if has_edge_attr else ''}"
                f"({'3' if has_edge_attr else '2'}) batch object. "
                f"Final error: {e3}"
            ) from e3

    def _is_edge_level_task(self) -> bool:
        """
        Determine if current task is edge-level.

        Edge-level tasks predict properties of edges rather than nodes or graphs.
        These tasks require special target handling:
        - link_prediction: Uses batch.edge_label
        - edge_regression: Uses batch.edge_value or batch.edge_y

        Returns:
            True if task_type indicates an edge-level task

        Example:
            >>> trainer._task_type = 'link_prediction'
            >>> trainer._is_edge_level_task()
            True

            >>> trainer._task_type = 'graph_regression'
            >>> trainer._is_edge_level_task()
            False
        """
        if self._task_type is None:
            return False

        task_lower = self._task_type.lower()

        # Explicit edge-level task types
        edge_level_tasks = ["link_prediction", "edge_regression"]

        if task_lower in edge_level_tasks:
            return True

        # Future-proof: any task starting with 'link_' or 'edge_'
        return task_lower.startswith("link_") or task_lower.startswith("edge_")

    def _is_graph_level_task(self) -> bool:
        """
        Determine if current task is graph-level.

        Graph-level tasks predict properties of entire graphs rather than
        individual nodes or edges. These tasks may require special target
        handling when multi-target y is flattened by PyG batching.

        Returns:
            True if task_type indicates a graph-level task

        Example:
            >>> trainer._task_type = 'graph_regression'
            >>> trainer._is_graph_level_task()
            True

            >>> trainer._task_type = 'node_classification'
            >>> trainer._is_graph_level_task()
            False
        """
        if self._task_type is None:
            return False

        task_lower = self._task_type.lower()

        # Explicit graph-level task types
        graph_level_tasks = ["graph_regression", "graph_classification"]

        if task_lower in graph_level_tasks:
            return True

        # Future-proof: any task starting with 'graph_'
        return task_lower.startswith("graph_")

    def _get_target(self, batch: Data | Batch) -> torch.Tensor:
        """
        Get appropriate target tensor based on task type.

        Intelligently selects the correct target attribute from the batch:
        - Node-level tasks: batch.y (node labels/values)
        - Graph-level tasks: batch.y (graph labels/values)
        - Link prediction: batch.edge_label (binary edge labels)
        - Edge regression: batch.edge_value or batch.edge_y (edge values)

        Args:
            batch: PyG Data or Batch object containing graph data

        Returns:
            Target tensor appropriate for the current task type

        Example:
            >>> # For link prediction
            >>> trainer._task_type = 'link_prediction'
            >>> target = trainer._get_target(batch)  # Returns batch.edge_label

            >>> # For graph regression
            >>> trainer._task_type = 'graph_regression'
            >>> target = trainer._get_target(batch)  # Returns batch.y
        """

        if not self._is_edge_level_task():
            # Node-level and graph-level tasks use batch.y
            target = batch.y

            # ================================================================
            # TARGET SELECTION: Apply selection BEFORE reshaping
            # ================================================================
            # DYNAMIC: Works with any selection indices
            # PRODUCTION-READY: Handles all tensor shapes
            # FUTURE-PROOF: Selection logic isolated in helper method
            # ================================================================
            if self._target_indices is not None and target is not None:
                target = self._apply_target_selection(target, batch)

            # Handle graph-level multi-target: reshape from [batch*targets] to [batch, targets]
            #
            # WHY THIS IS NEEDED (REGRESSION ONLY):
            # PyG concatenates graph-level y along dim 0 during batching, so for 32 graphs
            # with 8 targets each:
            #   - Each graph has y.shape = [8] (1D tensor with 8 values)
            #   - PyG batching concatenates: batch.y.shape = [32*8] = [256]
            #   - Model outputs: [32, 8]
            #   - Need to reshape target: [256] -> [32, 8] to match model output
            #
            # CLASSIFICATION BEHAVIOR:
            # For classification tasks with discretized targets:
            #   - Each graph has y = scalar (class index, 0-dim tensor)
            #   - PyG batching: scalars are unsqueezed then concatenated -> [batch_size]
            #   - Model outputs: [batch_size, num_classes] (logits)
            #   - CrossEntropyLoss expects: input [N, C], target [N]
            #   - NO RESHAPE NEEDED - target [batch_size] is already correct
            #
            # NOTE: After target selection, num_targets = len(selected_indices)
            #
            # SAFETY CONDITIONS (ALL must be true to reshape):
            # 1. Task is graph-level (not node-level or edge-level)
            # 2. NOT a classification task (classification targets are already [batch])
            # 3. out_channels > 1 (multi-target, not single-target)
            # 4. target exists and is 1D (flattened by PyG batching)
            # 5. batch.num_graphs is available (batched data)
            # 6. target size matches expected size (num_graphs * num_targets)
            if (
                self._is_graph_level_task()
                and not self._is_classification_task  # Don't reshape classification targets
                and self._out_channels is not None
                and self._out_channels > 1
                and target is not None
                and target.dim() == 1
                and hasattr(batch, "num_graphs")
                and batch.num_graphs is not None
            ):
                num_graphs = batch.num_graphs
                num_targets = self._out_channels  # This is now the SELECTED count
                expected_size = num_graphs * num_targets

                # Only reshape if sizes match exactly (safety check)
                if target.size(0) == expected_size:
                    target = target.view(num_graphs, num_targets)
                    logger.debug(
                        f"Reshaped graph-level multi-target: [{expected_size}] -> "
                        f"[{num_graphs}, {num_targets}]"
                    )
                else:
                    # Size mismatch - don't reshape, log warning for debugging
                    logger.warning(
                        f"Graph-level multi-target reshape skipped: "
                        f"target.size(0)={target.size(0)} != expected {expected_size} "
                        f"(num_graphs={num_graphs}, out_channels={num_targets}). "
                        f"This may indicate a data/config mismatch."
                    )

            # For classification: target should already be [batch_size] with class indices
            # Log for debugging if classification task detected
            if self._is_classification_task and self._is_graph_level_task():
                if target is not None and hasattr(batch, "num_graphs"):
                    logger.debug(
                        f"Classification task: target shape [{target.size(0)}] "
                        f"(class indices for {batch.num_graphs} graphs, no reshape needed)"
                    )

            return target

        task_lower = self._task_type.lower()

        if task_lower == "link_prediction":
            # Link prediction uses edge_label (binary: 0 or 1)
            if hasattr(batch, "edge_label") and batch.edge_label is not None:
                return batch.edge_label.float()  # BCEWithLogitsLoss expects float
            else:
                logger.warning(
                    "link_prediction task but batch.edge_label not found. Falling back to batch.y"
                )
                return batch.y

        elif task_lower == "edge_regression":
            # Edge regression uses edge_value or edge_y
            if hasattr(batch, "edge_value") and batch.edge_value is not None:
                return batch.edge_value
            elif hasattr(batch, "edge_y") and batch.edge_y is not None:
                return batch.edge_y
            else:
                logger.warning(
                    "edge_regression task but batch.edge_value/edge_y not found. "
                    "Falling back to batch.y"
                )
                return batch.y

        elif task_lower == "edge_classification":
            # Edge classification uses edge_y (class indices from DiscretizeTargets)
            # or edge_label (if already integer labels)
            if hasattr(batch, "edge_y") and batch.edge_y is not None:
                target = batch.edge_y
                # Flatten if needed - edge_y should be [num_edges] for cross_entropy
                if target.dim() > 1:
                    target = target.squeeze(-1)

                # Validate edge_y size matches edge_index size
                # This is critical for edge-level predictions
                num_target_edges = target.size(0)
                num_batch_edges = batch.edge_index.size(1) if hasattr(batch, "edge_index") else None

                if num_batch_edges is not None and num_target_edges != num_batch_edges:
                    logger.warning(
                        f"edge_classification: edge_y has {num_target_edges} values but "
                        f"edge_index has {num_batch_edges} edges. This indicates a data "
                        f"inconsistency - edge_y and edge_index should have matching counts. "
                        f"The model will predict on edge_index edges, but targets come from edge_y. "
                        f"Ensure edge_y was extracted from the same edge set as edge_index."
                    )

                return target.long()  # CrossEntropyLoss expects LongTensor
            elif hasattr(batch, "edge_label") and batch.edge_label is not None:
                return batch.edge_label.long()
            else:
                logger.warning(
                    "edge_classification task but batch.edge_y/edge_label not found. "
                    "Falling back to batch.y"
                )
                return batch.y
                return batch.y

        # Fallback for any other edge-level task (future-proofing)
        if hasattr(batch, "edge_label") and batch.edge_label is not None:
            return batch.edge_label.float()
        elif hasattr(batch, "edge_value") and batch.edge_value is not None:
            return batch.edge_value
        else:
            return batch.y

    def _apply_target_selection(self, target: torch.Tensor, batch: Data | Batch) -> torch.Tensor:
        """
        Apply target selection to extract specified target columns.

        DYNAMIC: Handles both 1D (flattened) and 2D tensors
        PRODUCTION-READY: Comprehensive shape handling with logging
        FUTURE-PROOF: Isolated selection logic, easy to extend

        Args:
            target: Original target tensor (batch.y)
            batch: Batch object for metadata access

        Returns:
            Selected target tensor with only specified columns
        """
        indices = self._target_indices

        if target is None or indices is None:
            return target

        # ====================================================================
        # CLASSIFICATION BYPASS: Skip target selection for classification tasks
        # ====================================================================
        # For classification tasks (graph_classification, node_classification):
        # - Targets are class indices (scalars), NOT multi-target arrays
        # - Target selection was already done during discretization via target_column
        # - Shape is [batch_size] with class indices, not [batch_size * num_targets]
        # - No selection needed here - return target as-is
        # ====================================================================
        if self._is_classification_task:
            logger.debug(
                f"Target selection skipped for classification task: "
                f"target shape {list(target.shape)} (class indices, no selection needed)"
            )
            return target

        # ====================================================================
        # ALREADY-EXTRACTED CHECK: Skip if data preparation already extracted
        # ====================================================================
        # For node/edge level tasks, data preparation (_prepare_node_level_data,
        # _prepare_edge_regression_data) extracts targets from source tensor and
        # assigns to y. In this case, y already has the correct shape matching
        # out_channels, and re-applying selection would fail (indices refer to
        # original source tensor columns, not the extracted y).
        #
        # Detection: If target's last dimension equals out_channels (selected count),
        # selection was already applied during data preparation.
        # ====================================================================
        num_selected = len(indices)
        target_last_dim = target.size(-1) if target.dim() > 1 else target.size(0)

        # Check if target already has the selected dimensions
        if target_last_dim == num_selected:
            # Additional check: if original had more columns, selection was applied
            original_total = self._original_out_channels or self._target_selection.get(
                "total_available"
            )
            if original_total is not None and original_total > num_selected:
                logger.debug(
                    f"Target selection skipped (already extracted by data preparation): "
                    f"target shape {list(target.shape)} matches out_channels={num_selected}"
                )
                return target

        # Get total number of original targets from selection info
        total_targets = self._original_out_channels or self._target_selection.get(
            "total_available", len(indices)
        )

        # Case 1: 2D tensor [batch, num_targets] - simple column selection
        if target.dim() == 2:
            selected = target[:, indices]
            logger.debug(
                f"Target selection (2D): {list(target.shape)} -> {list(selected.shape)} "
                f"[indices: {indices}]"
            )
            return selected

        # Case 2: 1D tensor [batch * num_targets] - flattened graph-level targets
        # PyG concatenates graph-level y during batching
        if target.dim() == 1 and hasattr(batch, "num_graphs") and batch.num_graphs is not None:
            num_graphs = batch.num_graphs
            expected_flat_size = num_graphs * total_targets

            if target.size(0) == expected_flat_size:
                # Reshape to [num_graphs, total_targets], select columns, then flatten
                reshaped = target.view(num_graphs, total_targets)
                selected = reshaped[:, indices]
                # Return flattened to match expected input for reshape logic
                result = selected.contiguous().view(-1)
                logger.debug(
                    f"Target selection (1D flattened): [{target.size(0)}] -> "
                    f"reshape [{num_graphs}, {total_targets}] -> "
                    f"select {indices} -> flatten [{result.size(0)}]"
                )
                return result
            else:
                # Size doesn't match expected - log warning
                logger.warning(
                    f"Target selection: 1D target size {target.size(0)} != "
                    f"expected {expected_flat_size} (num_graphs={num_graphs}, "
                    f"total_targets={total_targets}). Returning original."
                )
                return target

        # Case 3: 1D tensor without num_graphs - likely single sample
        if target.dim() == 1 and target.size(0) == total_targets:
            selected = target[indices]
            logger.debug(
                f"Target selection (1D single): [{target.size(0)}] -> [{selected.size(0)}]"
            )
            return selected

        # Fallback: return original if structure not recognized
        logger.warning(
            f"Target selection: unrecognized tensor structure "
            f"(shape={list(target.shape)}, dim={target.dim()}). Returning original."
        )
        return target

    def _prepare_target_for_metrics(self, target: torch.Tensor) -> torch.Tensor:
        """
        Prepare target tensor for metric computation with appropriate dtype.

        DYNAMIC: Automatically detects classification tasks requiring integer targets
        PRODUCTION-READY: Handles BCEWithLogitsLoss float targets vs TorchMetrics int targets
        FUTURE-PROOF: Task-aware conversion based on self._task_type

        Background:
            For binary classification tasks like link_prediction:
            - BCEWithLogitsLoss requires float targets (0.0 or 1.0)
            - TorchMetrics classification metrics (AUROC, Accuracy, etc.) require
              integer targets (torch.long with values 0 or 1)

            This method bridges the gap by converting float targets to long for metrics
            while keeping the original float targets for loss computation.

        Args:
            target: Target tensor (may be float for classification tasks)

        Returns:
            Target tensor with appropriate dtype for metrics:
            - For classification tasks: converted to torch.long
            - For regression tasks: unchanged (float)

        Example:
            >>> # link_prediction with float target [0., 1., 1., 0.]
            >>> metric_target = self._prepare_target_for_metrics(target)
            >>> # Returns tensor([0, 1, 1, 0], dtype=torch.int64)
        """
        if target is None:
            return target

        # Determine if this is a classification task requiring integer targets
        # Classification tasks include: link_prediction, *_classification
        is_classification = False

        if self._task_type is not None:
            task_lower = self._task_type.lower()
            is_classification = task_lower == "link_prediction" or "classification" in task_lower

        # Also check the explicit classification flag from model_info
        if self._is_classification_task:
            is_classification = True

        if is_classification and target.dtype in (torch.float32, torch.float64, torch.float16):
            # Convert float targets to long for classification metrics
            # This handles edge_label from RandomLinkSplit which is float
            return target.long()

        return target

    def _update_scheduler(self, val_metrics: dict[str, float], train_metrics: dict[str, float]):
        """Update learning rate scheduler."""
        try:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # ReduceLROnPlateau needs a metric
                metric = val_metrics.get("val_loss", train_metrics.get("train_loss"))
                if metric is not None:
                    self.scheduler.step(metric)
            else:
                # Other schedulers
                self.scheduler.step()

        except Exception as e:
            logger.warning(f"Scheduler update failed: {e}")

    def _log_epoch_summary(self, epoch: int, metrics: dict[str, Any]):
        """Log epoch summary."""
        summary_parts = [f"Epoch {epoch:3d}/{self.max_epochs}"]

        if "train_loss" in metrics:
            summary_parts.append(f"train_loss: {metrics['train_loss']:.6f}")

        if "val_loss" in metrics:
            summary_parts.append(f"val_loss: {metrics['val_loss']:.6f}")
            if metrics.get("is_best", False):
                summary_parts.append("(BEST)")

        if "epoch_time" in metrics:
            summary_parts.append(f"time: {metrics['epoch_time']:.2f}s")

        # Get current learning rate
        if self.optimizer:
            lr = self.optimizer.param_groups[0]["lr"]
            summary_parts.append(f"lr: {lr:.2e}")

        logger.info(" | ".join(summary_parts))

    def save_checkpoint(
        self,
        filepath: Path,
        hyper_parameters: dict[str, Any] | None = None,
        data_info: dict[str, Any] | None = None,
        **extra_data,
    ):
        """
        Save training checkpoint with model recreation metadata.

        Phase 1 Enhancement: Adds hyper_parameters and data_info for model
        recreation during inference, following PyTorch Lightning/Chemprop patterns.

        DYNAMIC: Saves COMPLETE hyper_parameters dict as-is. No hard-coded
                 parameter lists - ANY model parameters work without code changes.

        PRODUCTION-READY: Saves COMPLETE model_info from create_model_with_info()
                         which contains ALL runtime-computed information.

        FUTURE-PROOF: Uses version_info for checkpoint format versioning.
                      Backward compatible with v1.0 checkpoints (no hyper_parameters).

        Args:
            filepath: Path to save checkpoint
            hyper_parameters: Model recreation parameters (optional). When provided,
                enables loading the model for inference without external config.
                Should contain:
                - model_name: Registry name (e.g., "GCN", "GAT", "SchNet")
                - task_type: Task type (e.g., "graph_regression")
                - hyperparameters: COMPLETE hyperparameters dict passed to create_model()
                - model_info: COMPLETE model_info dict from create_model_with_info()
                If not provided, falls back to self.model_info if available.
            data_info: Data compatibility information (optional). Contains:
                - num_node_features: Number of node features model was trained with
                - num_edge_features: Number of edge features (if applicable)
                - requires_edge_features: Whether model requires edge features
                - requires_pos: Whether model requires 3D positions (e.g., SchNet)
            **extra_data: Additional data to save

        Example:
            >>> # Basic save (backward compatible)
            >>> trainer.save_checkpoint(Path("checkpoint.pt"))
            >>>
            >>> # Save with full model recreation metadata (recommended)
            >>> trainer.save_checkpoint(
            ...     Path("checkpoint.pt"),
            ...     hyper_parameters={
            ...         'model_name': 'GCN',
            ...         'task_type': 'graph_regression',
            ...         'hyperparameters': {'hidden_channels': 64, 'num_layers': 3},
            ...         'model_info': model_info,  # From create_model_with_info()
            ...     },
            ...     data_info={
            ...         'num_node_features': 16,
            ...         'num_edge_features': 4,
            ...     }
            ... )
        """
        try:
            # =================================================================
            # TRAINING STATE (Existing fields - unchanged for backward compat)
            # =================================================================
            checkpoint = {
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics_history": dict(self.metrics_history),
                "best_val_loss": self.best_val_loss,
            }

            if self.scheduler is not None:
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

            # =================================================================
            # MODEL RECREATION METADATA (Phase 1 Enhancement)
            # =================================================================
            # DYNAMIC: If hyper_parameters not explicitly provided, construct
            # from self.model_info (which contains complete model metadata)
            # This ensures checkpoints are self-contained for inference
            # =================================================================
            # CRITICAL FIX: Save hyperparameters at BOTH locations:
            # 1. 'hyperparameters' - Direct location (expected by model_loader)
            # 2. 'model_info.hyperparameters_values' - Existing location (backward compat)
            # =================================================================
            if hyper_parameters is not None:
                # Use explicitly provided hyper_parameters
                checkpoint["hyper_parameters"] = hyper_parameters
            elif self.model_info:
                # Construct from stored model_info (DYNAMIC - uses whatever is stored)
                # Extract actual hyperparameters values for model recreation
                hyperparams_values = self.model_info.get("hyperparameters_values", {})

                checkpoint["hyper_parameters"] = {
                    "model_name": self.model_info.get("name"),
                    "task_type": self.model_info.get("task_type"),
                    # CRITICAL: Save hyperparameters at expected location for model_loader
                    "hyperparameters": hyperparams_values,
                    # Store COMPLETE model_info for all runtime-computed values
                    "model_info": self.model_info,
                    # Store target_selection_config if applicable
                    "target_selection_config": self.model_info.get("target_selection"),
                }
            else:
                # No model_info available - v1.0 compatible checkpoint
                checkpoint["hyper_parameters"] = {}

            # =================================================================
            # DATA COMPATIBILITY INFO (Phase 1 Enhancement)
            # =================================================================
            # PRODUCTION-READY: Enables validation before loading for inference
            # =================================================================
            if data_info is not None:
                checkpoint["data_info"] = data_info
            else:
                # Construct minimal data_info from model_info if available
                checkpoint["data_info"] = {
                    "requires_edge_features": self.model_info.get("requires_edge_features", False)
                    if self.model_info
                    else False,
                    "uses_edge_features": self.model_info.get("uses_edge_features", False)
                    if self.model_info
                    else False,
                }

            # =================================================================
            # VERSION INFO (Phase 1 Enhancement)
            # =================================================================
            # FUTURE-PROOF: Enables format versioning for backward compatibility
            # =================================================================
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

            # =================================================================
            # EXTRA DATA (Existing - unchanged)
            # =================================================================
            checkpoint.update(extra_data)

            # Save checkpoint
            torch.save(checkpoint, filepath)
            logger.info(f"Checkpoint saved: {filepath} (format v2.0)")

        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint: {e}") from e

    def load_checkpoint(self, filepath: Path) -> dict[str, Any]:
        """
        Load training checkpoint with v2.0 format support.

        Phase 1 Enhancement: Handles both v1.0 (legacy) and v2.0 (enhanced)
        checkpoint formats with full backward compatibility.

        DYNAMIC: Automatically detects checkpoint format version.
        PRODUCTION-READY: Logs format version and provides clear warnings for legacy checkpoints.
        FUTURE-PROOF: Version-aware loading enables smooth upgrades.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Dictionary with extra data from checkpoint, including:
            - hyper_parameters: Model recreation parameters (v2.0 only)
            - data_info: Data compatibility information (v2.0 only)
            - version_info: Checkpoint format version info (v2.0 only)
            - Any other extra data saved with the checkpoint

        Example:
            >>> extra = trainer.load_checkpoint(Path("checkpoint.pt"))
            >>> if 'hyper_parameters' in extra:
            ...     print(f"Model: {extra['hyper_parameters'].get('model_name')}")
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)

            # =================================================================
            # DETECT CHECKPOINT FORMAT VERSION
            # =================================================================
            version_info = checkpoint.get("version_info", {})
            format_version = version_info.get("checkpoint_format_version", "1.0")

            if format_version == "1.0":
                logger.warning(
                    f"Loading v1.0 checkpoint from {filepath}. "
                    "This checkpoint does not contain model recreation metadata. "
                    "For inference without external config, re-save with v2.0 format."
                )
            else:
                logger.debug(f"Loading v{format_version} checkpoint from {filepath}")

            # =================================================================
            # LOAD TRAINING STATE (Unchanged - backward compatible)
            # =================================================================
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            self.current_epoch = checkpoint.get("epoch", 0)
            self.global_step = checkpoint.get("global_step", 0)
            self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            self.metrics_history = defaultdict(list, checkpoint.get("metrics_history", {}))

            # =================================================================
            # RESTORE MODEL_INFO FROM CHECKPOINT (v2.0 Enhancement)
            # =================================================================
            # DYNAMIC: If checkpoint has hyper_parameters with model_info,
            # update self.model_info to maintain consistency
            # =================================================================
            hyper_params = checkpoint.get("hyper_parameters", {})
            if "model_info" in hyper_params and hyper_params["model_info"]:
                # Merge with existing model_info (checkpoint takes precedence)
                if self.model_info:
                    self.model_info.update(hyper_params["model_info"])
                else:
                    self.model_info = hyper_params["model_info"]
                logger.debug("Restored model_info from checkpoint")

            logger.info(f"Checkpoint loaded: {filepath} (format v{format_version})")

            # =================================================================
            # RETURN EXTRA DATA (Including new v2.0 fields)
            # =================================================================
            # For v2.0: includes hyper_parameters, data_info, version_info
            # For v1.0: only includes any custom extra_data that was saved
            # =================================================================
            core_keys = {
                "epoch",
                "global_step",
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "metrics_history",
                "best_val_loss",
            }
            extra_keys = set(checkpoint.keys()) - core_keys
            return {k: checkpoint[k] for k in extra_keys}

        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint: {e}") from e

    # =========================================================================
    # CHECKPOINT UTILITIES (Phase 1 Enhancement)
    # =========================================================================

    @staticmethod
    def get_checkpoint_info(filepath: Path) -> dict[str, Any]:
        """
        Get information about a checkpoint without loading the model.

        DYNAMIC: Returns whatever metadata is in the checkpoint.
        PRODUCTION-READY: Safe to call without model/optimizer instances.
        FUTURE-PROOF: Works with any checkpoint format version.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Dictionary with checkpoint information:
            - format_version: Checkpoint format version ('1.0' or '2.0')
            - is_v2: Whether checkpoint has v2.0 metadata
            - epoch: Training epoch when checkpoint was saved
            - best_val_loss: Best validation loss at checkpoint time
            - model_name: Model name (v2.0 only)
            - task_type: Task type (v2.0 only)
            - hyper_parameters: Full hyper_parameters dict (v2.0 only)
            - data_info: Data compatibility info (v2.0 only)
            - version_info: Version information (v2.0 only)

        Example:
            >>> info = Trainer.get_checkpoint_info(Path("model.pt"))
            >>> print(f"Format: v{info['format_version']}")
            >>> if info['is_v2']:
            ...     print(f"Model: {info['model_name']}")
        """
        checkpoint = torch.load(filepath, map_location="cpu")

        version_info = checkpoint.get("version_info", {})
        format_version = version_info.get("checkpoint_format_version", "1.0")
        hyper_params = checkpoint.get("hyper_parameters", {})

        return {
            "format_version": format_version,
            "is_v2": format_version >= "2.0",
            "epoch": checkpoint.get("epoch", 0),
            "best_val_loss": checkpoint.get("best_val_loss", None),
            "model_name": hyper_params.get("model_name")
            or hyper_params.get("model_info", {}).get("name"),
            "task_type": hyper_params.get("task_type")
            or hyper_params.get("model_info", {}).get("task_type"),
            "hyper_parameters": hyper_params,
            "data_info": checkpoint.get("data_info", {}),
            "version_info": version_info,
        }

    @staticmethod
    def is_v2_checkpoint(filepath: Path) -> bool:
        """
        Check if a checkpoint uses v2.0 format.

        Args:
            filepath: Path to checkpoint file

        Returns:
            True if checkpoint has v2.0 metadata, False otherwise

        Example:
            >>> if Trainer.is_v2_checkpoint(Path("model.pt")):
            ...     print("Checkpoint can be loaded for inference without config")
        """
        try:
            checkpoint = torch.load(filepath, map_location="cpu")
            version_info = checkpoint.get("version_info", {})
            format_version = version_info.get("checkpoint_format_version", "1.0")
            return format_version >= "2.0"
        except Exception:
            return False

    # =========================================================================
    # RESULTS SAVING (Phase 5 Refactor)
    # =========================================================================

    def save_results(
        self,
        output_dir: str | Path,
        results: dict[str, Any],
        save_checkpoint: bool = True,
        checkpoint_filename: str = "final_model.pt",
        results_filename: str = "training_results.json",
    ) -> dict[str, Path]:
        """
        Save training results and optionally final checkpoint.

        DYNAMIC: Handles any results dictionary structure
        PRODUCTION-READY: Comprehensive error handling, JSON serialization
        FUTURE-PROOF: Extensible via additional kwargs, returns saved paths

        Args:
            output_dir: Directory to save results (created if doesn't exist)
            results: Training results dictionary from fit()
            save_checkpoint: Whether to save final model checkpoint (default: True)
            checkpoint_filename: Filename for checkpoint (default: 'final_model.pt')
            results_filename: Filename for results JSON (default: 'training_results.json')

        Returns:
            Dict with paths to saved files:
            - 'results_path': Path to training_results.json
            - 'checkpoint_path': Path to checkpoint (if save_checkpoint=True)

        Raises:
            CheckpointError: If checkpoint saving fails
            IOError: If results file cannot be written

        Example:
            >>> results = trainer.fit()
            >>> saved_paths = trainer.save_results(
            ...     output_dir='./training_output',
            ...     results=results
            ... )
            >>> print(f"Results saved to: {saved_paths['results_path']}")
        """
        import json

        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        # Save checkpoint if requested
        if save_checkpoint:
            checkpoint_path = output_path / checkpoint_filename
            self.save_checkpoint(checkpoint_path)
            saved_paths["checkpoint_path"] = checkpoint_path
            logger.info(f"Final checkpoint saved: {checkpoint_path}")

        # Convert results to JSON-serializable format
        serializable_results = self._make_json_serializable(results)

        # Add metadata
        serializable_results["_metadata"] = {
            "saved_at": datetime.now().isoformat(),
            "epochs_completed": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "device": str(self.device),
        }

        # Save results JSON
        results_path = output_path / results_filename
        try:
            with open(results_path, "w") as f:
                json.dump(serializable_results, f, indent=2)
            saved_paths["results_path"] = results_path
            logger.info(f"Training results saved: {results_path}")
        except Exception as e:
            raise OSError(f"Failed to save training results: {e}") from e

        return saved_paths

    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Recursively convert object to JSON-serializable format.

        DYNAMIC: Handles nested dicts, lists, and various types
        PRODUCTION-READY: Safe fallback to string for unknown types
        FUTURE-PROOF: Extensible type handling

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable version of object
        """
        if obj is None:
            return None
        elif isinstance(obj, (bool, int, float, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(k): self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, "item"):
            # Handle numpy/torch scalars
            return obj.item()
        elif hasattr(obj, "tolist"):
            # Handle numpy/torch arrays
            return obj.tolist()
        else:
            # Fallback: convert to string
            return str(obj)

    # =========================================================================
    # CALLBACK HOOKS
    # =========================================================================

    def _on_train_begin(self):
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            try:
                callback.on_train_begin(self)
            except Exception as e:
                logger.warning(f"Callback {callback.__class__.__name__}.on_train_begin failed: {e}")

    def _on_epoch_end(self, metrics: dict[str, float]):
        for callback in self.callbacks:
            try:
                callback.on_epoch_end(self, self.current_epoch, metrics)
            except Exception as e:
                # Don't catch optuna.TrialPruned - let it propagate!
                try:
                    import optuna

                    if isinstance(e, optuna.TrialPruned):
                        raise
                except ImportError:
                    pass
                logger.warning(f"Callback {callback.__class__.__name__}.on_epoch_end failed: {e}")

    def _on_train_end(self):
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            try:
                callback.on_train_end(self)
            except Exception as e:
                logger.warning(f"Callback {callback.__class__.__name__}.on_train_end failed: {e}")

    def _should_stop(self) -> bool:
        """
        Check if training should stop (from callbacks).

        Handles both method-based should_stop (e.g., EarlyStopping)
        and property-based should_stop (for compatibility).
        """
        for callback in self.callbacks:
            try:
                if hasattr(callback, "should_stop"):
                    # Get the should_stop attribute
                    should_stop_attr = callback.should_stop

                    # Handle both methods and properties
                    if callable(should_stop_attr):
                        # It's a method - call it
                        if should_stop_attr():
                            return True
                    else:
                        # It's a property (already evaluated to bool) - use directly
                        if should_stop_attr:
                            return True
            except Exception as e:
                logger.warning(f"Callback {callback.__class__.__name__}.should_stop failed: {e}")
        return False


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info("trainer module loaded")
