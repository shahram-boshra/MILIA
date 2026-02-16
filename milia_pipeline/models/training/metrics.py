"""
Metrics Registry Module

Centralized registry for evaluation metrics with:
- TorchMetrics-based implementations for GPU/distributed support
- Task-aware metric selection (regression vs classification)
- Automatic fallback to default metrics on incompatibility
- Custom metric registration support
- Full parameter introspection via inspect.signature()

Author: milia Team
Version: 1.0.0
"""

import inspect
import logging
from typing import Any

import torch
import torch.nn as nn

# TorchMetrics imports - MUST be available (added to environment.yml)
try:
    import torchmetrics
    from torchmetrics import Metric, MetricCollection
    from torchmetrics.classification import (
        AUROC,
        Accuracy,
        AveragePrecision,
        F1Score,
        Precision,
        Recall,
    )
    from torchmetrics.regression import (
        ExplainedVariance,
        MeanAbsoluteError,
        MeanAbsolutePercentageError,
        MeanSquaredError,
        R2Score,
    )

    TORCHMETRICS_AVAILABLE = True
except ImportError as e:
    TORCHMETRICS_AVAILABLE = False
    TORCHMETRICS_IMPORT_ERROR = str(e)


logger = logging.getLogger(__name__)


# =============================================================================
# CUSTOM METRIC WRAPPERS (For compatibility with existing loss function style)
# =============================================================================


class RMSEMetric(nn.Module):
    """
    Root Mean Squared Error Metric wrapper.

    Wraps TorchMetrics MeanSquaredError with squared=False for RMSE.
    Provides nn.Module interface consistent with loss functions.
    """

    def __init__(self):
        super().__init__()
        if TORCHMETRICS_AVAILABLE:
            self._metric = MeanSquaredError(squared=False)
        else:
            self._metric = None

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute RMSE."""
        if self._metric is not None:
            return self._metric(preds, target)
        # Fallback implementation
        return torch.sqrt(torch.mean((preds - target) ** 2))

    def to(self, device):
        """Move metric to device."""
        super().to(device)
        if self._metric is not None:
            self._metric = self._metric.to(device)
        return self


# =============================================================================
# METRICS REGISTRY
# =============================================================================


class MetricsRegistry:
    """
    Centralized registry for evaluation metrics.

    DYNAMIC: Uses TorchMetrics as backend with automatic device placement
    PRODUCTION-READY: Supports DDP/distributed training, proper accumulation
    FUTURE-PROOF: Extensible via register_custom_metric(), follows LossRegistry pattern

    Usage:
        >>> from milia_pipeline.models.training import MetricsRegistry
        >>> metric = MetricsRegistry.get_metric("mse")
        >>> metrics = MetricsRegistry.get_metrics_for_task("graph_regression")
        >>> available = MetricsRegistry.list_available()
    """

    # =========================================================================
    # METRIC REGISTRY - Maps string names to TorchMetrics classes
    # =========================================================================
    _metrics: dict[str, type] = {
        # =====================================================================
        # REGRESSION METRICS
        # =====================================================================
        "mse": MeanSquaredError if TORCHMETRICS_AVAILABLE else None,
        "mae": MeanAbsoluteError if TORCHMETRICS_AVAILABLE else None,
        "rmse": RMSEMetric,  # Custom wrapper
        "r2": R2Score if TORCHMETRICS_AVAILABLE else None,
        "mape": MeanAbsolutePercentageError if TORCHMETRICS_AVAILABLE else None,
        "explained_variance": ExplainedVariance if TORCHMETRICS_AVAILABLE else None,
        # =====================================================================
        # CLASSIFICATION METRICS
        # =====================================================================
        "accuracy": Accuracy if TORCHMETRICS_AVAILABLE else None,
        "precision": Precision if TORCHMETRICS_AVAILABLE else None,
        "recall": Recall if TORCHMETRICS_AVAILABLE else None,
        "f1": F1Score if TORCHMETRICS_AVAILABLE else None,
        "auroc": AUROC if TORCHMETRICS_AVAILABLE else None,
        "auprc": AveragePrecision if TORCHMETRICS_AVAILABLE else None,
    }

    # =========================================================================
    # TASK-AWARE METRIC SELECTION (Mirrors LossRegistry pattern)
    # =========================================================================

    # Classification vs Regression metric categories
    _classification_metrics: set[str] = {"accuracy", "precision", "recall", "f1", "auroc", "auprc"}
    _regression_metrics: set[str] = {"mse", "mae", "rmse", "r2", "mape", "explained_variance"}

    # Task type to default metrics mapping
    # CRITICAL: Defines fallback metrics when config metrics are incompatible
    _task_to_default_metrics: dict[str, list[str]] = {
        # Regression tasks → Regression metrics
        "graph_regression": ["mae", "mse", "rmse", "r2"],
        "node_regression": ["mae", "mse", "rmse", "r2"],
        "edge_regression": ["mae", "mse", "rmse", "r2"],
        # Classification tasks → Classification metrics
        "graph_classification": ["accuracy", "f1", "precision", "recall", "auroc"],
        "node_classification": ["accuracy", "f1", "precision", "recall"],
        "edge_classification": ["accuracy", "f1", "precision", "recall"],
        # Link prediction → Binary classification metrics
        "link_prediction": ["auroc", "auprc", "accuracy"],
    }

    @classmethod
    def get_metric(
        cls, name: str, params: dict[str, Any] | None = None, device: torch.device | None = None
    ) -> nn.Module:
        """
        Get metric by name.

        DYNAMIC: Filters parameters via inspect.signature() introspection
        PRODUCTION-READY: Handles device placement, validates inputs
        FUTURE-PROOF: Works with any TorchMetrics-compatible class

        Args:
            name: Metric name (e.g., "mse", "accuracy", "f1")
            params: Optional parameters to pass to metric constructor
            device: Optional device to place metric on

        Returns:
            Instantiated metric module

        Raises:
            ValueError: If metric name not found
            RuntimeError: If TorchMetrics not available
        """
        name_lower = name.lower()

        if name_lower not in cls._metrics:
            available = ", ".join(sorted(cls._metrics.keys()))
            raise ValueError(f"Unknown metric: '{name}'. Available metrics: {available}")

        metric_cls = cls._metrics[name_lower]

        if metric_cls is None:
            raise RuntimeError(
                f"Metric '{name}' requires TorchMetrics but it's not installed. "
                f"Install with: pip install torchmetrics>=1.0.0"
            )

        params = params or {}

        # Filter parameters to only those accepted by the metric class
        filtered_params = cls._filter_params(metric_cls, params)

        try:
            metric = metric_cls(**filtered_params)

            if filtered_params:
                logger.debug(f"Initialized {name} metric with params: {filtered_params}")
            else:
                logger.debug(f"Initialized {name} metric with default params")

            # Log filtered out params at debug level
            ignored = set(params.keys()) - set(filtered_params.keys())
            if ignored:
                logger.debug(f"Metric '{name}': ignored unsupported params {ignored}")

            # Move to device if specified
            if device is not None and hasattr(metric, "to"):
                metric = metric.to(device)

            return metric

        except TypeError as e:
            raise ValueError(
                f"Invalid parameters for metric '{name}': {filtered_params}. Error: {e}"
            ) from e

    @classmethod
    def _filter_params(cls, target_cls: type, params: dict[str, Any]) -> dict[str, Any]:
        """
        Filter parameters to only those accepted by the target class constructor.

        DYNAMIC: Uses inspect.signature() for runtime introspection
        PRODUCTION-READY: Handles edge cases (C extensions, __new__ vs __init__)
        FUTURE-PROOF: Works with any class without hardcoded param lists

        Note: TorchMetrics v1.0+ classification metrics use __new__ for task routing,
        so we must inspect BOTH __new__ and __init__ to capture all valid parameters.

        Args:
            target_cls: The class whose constructor parameters to check
            params: Dictionary of parameters to filter

        Returns:
            Filtered dictionary containing only valid parameters
        """
        if not params:
            return {}

        valid_param_names = set()

        # Check __init__ parameters
        try:
            init_sig = inspect.signature(target_cls.__init__)
            valid_param_names.update(init_sig.parameters.keys())
        except (ValueError, TypeError):
            pass

        # Check __new__ parameters (TorchMetrics v1.0+ uses __new__ for task routing)
        # This is critical for AUROC, Accuracy, Precision, Recall, F1, AveragePrecision
        try:
            if hasattr(target_cls, "__new__"):
                new_sig = inspect.signature(target_cls.__new__)
                valid_param_names.update(new_sig.parameters.keys())
        except (ValueError, TypeError):
            pass

        # Remove 'self' and 'cls' from valid params
        valid_param_names -= {"self", "cls"}

        # If we found valid parameters, filter; otherwise return original params
        if valid_param_names:
            filtered = {k: v for k, v in params.items() if k in valid_param_names}
            return filtered
        else:
            # Fallback: return original params if introspection completely failed
            return params

    @classmethod
    def list_available(cls) -> list[str]:
        """
        List all available metric names.

        Returns:
            Sorted list of metric names
        """
        return sorted(cls._metrics.keys())

    @classmethod
    def get_metric_info(cls, name: str) -> dict[str, Any]:
        """
        Get information about a metric.

        Args:
            name: Metric name

        Returns:
            Dictionary with metric information including valid parameters
        """
        name_lower = name.lower()

        if name_lower not in cls._metrics:
            raise ValueError(f"Unknown metric: '{name}'")

        metric_cls = cls._metrics[name_lower]

        if metric_cls is None:
            return {
                "name": name_lower,
                "class": None,
                "available": False,
                "reason": "TorchMetrics not installed",
            }

        return {
            "name": name_lower,
            "class": metric_cls.__name__,
            "module": metric_cls.__module__,
            "doc": metric_cls.__doc__,
            "valid_params": cls.get_valid_params(name_lower),
            "is_classification": name_lower in cls._classification_metrics,
            "is_regression": name_lower in cls._regression_metrics,
        }

    @classmethod
    def get_valid_params(cls, name: str) -> dict[str, Any]:
        """
        Get valid parameters for a metric using introspection.

        Args:
            name: Metric name

        Returns:
            Dictionary mapping parameter names to their default values
        """
        name_lower = name.lower()

        if name_lower not in cls._metrics:
            raise ValueError(f"Unknown metric: '{name}'")

        metric_cls = cls._metrics[name_lower]

        if metric_cls is None:
            return {}

        try:
            sig = inspect.signature(metric_cls.__init__)
            params = {}
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                if param.default is not inspect.Parameter.empty:
                    params[param_name] = param.default
                else:
                    params[param_name] = None  # Required param, no default
            return params
        except (ValueError, TypeError):
            return {}

    @classmethod
    def register_custom_metric(
        cls,
        name: str,
        metric_class: type,
        is_classification: bool = False,
        is_regression: bool = True,
        overwrite: bool = False,
    ):
        """
        Register a custom metric.

        Args:
            name: Name to register metric under
            metric_class: Metric class (must be nn.Module or torchmetrics.Metric subclass)
            is_classification: Whether this is a classification metric
            is_regression: Whether this is a regression metric
            overwrite: Whether to overwrite existing metric with same name

        Raises:
            ValueError: If name exists and overwrite=False
            TypeError: If metric_class is not a valid metric type
        """
        if not (
            isinstance(metric_class, type)
            and (
                issubclass(metric_class, nn.Module)
                or (TORCHMETRICS_AVAILABLE and issubclass(metric_class, Metric))
            )
        ):
            raise TypeError(
                f"metric_class must be a subclass of nn.Module or torchmetrics.Metric, "
                f"got {type(metric_class)}"
            )

        name_lower = name.lower()

        if name_lower in cls._metrics and not overwrite:
            raise ValueError(f"Metric '{name}' already registered. Use overwrite=True to replace.")

        cls._metrics[name_lower] = metric_class

        if is_classification:
            cls._classification_metrics.add(name_lower)
        if is_regression:
            cls._regression_metrics.add(name_lower)

        logger.info(f"Registered custom metric: '{name}'")

    # =========================================================================
    # TASK-AWARE METRIC SELECTION
    # =========================================================================

    @classmethod
    def get_metrics_for_task(
        cls,
        task_type: str,
        metric_names: list[str] | None = None,
        params: dict[str, Any] | None = None,
        device: torch.device | None = None,
        num_classes: int | None = None,
    ) -> dict[str, nn.Module]:
        """
        Get metrics with task-aware automatic selection.

        DYNAMIC: Validates config metrics against task type, falls back to defaults
        PRODUCTION-READY: Logs warnings on incompatibility, never fails silently
        FUTURE-PROOF: Extensible via _task_to_default_metrics

        Metric Selection Strategy:
        1. If metric_names provided and ALL compatible with task → use them
        2. If metric_names provided but ANY incompatible → warn and use defaults
        3. If metric_names is None/empty → auto-select based on task_type

        Args:
            task_type: Task type string (e.g., 'graph_regression', 'graph_classification')
            metric_names: Optional list of metric names from config
            params: Optional parameters to pass to metrics
            device: Optional device to place metrics on
            num_classes: Number of classes (required for classification metrics)

        Returns:
            Dictionary mapping metric names to instantiated metric modules

        Example:
            >>> # Auto-select for regression task
            >>> metrics = MetricsRegistry.get_metrics_for_task('graph_regression')
            >>> # Returns {'mae': MAE(), 'mse': MSE(), 'rmse': RMSE(), 'r2': R2Score()}

            >>> # Config metrics incompatible with task → fallback with warning
            >>> metrics = MetricsRegistry.get_metrics_for_task(
            ...     'graph_regression',
            ...     metric_names=['accuracy', 'f1']  # WRONG for regression!
            ... )
            >>> # Warning logged, returns default regression metrics
        """
        task_lower = task_type.lower() if task_type else ""
        params = params or {}

        # Determine task category
        is_classification = "classification" in task_lower or task_lower == "link_prediction"

        # Get default metrics for this task type
        default_metrics = cls._task_to_default_metrics.get(task_lower, ["mse", "mae"])

        # Determine which metrics to use
        if metric_names:
            # Validate ALL config metrics against task type
            incompatible = []
            compatible = []

            for name in metric_names:
                name_lower = name.lower()
                if cls.is_metric_compatible_with_task(name_lower, task_type):
                    compatible.append(name_lower)
                else:
                    incompatible.append(name_lower)

            if incompatible:
                # MISMATCH DETECTED: Log warning and fall back to defaults
                logger.warning(
                    f"Config metrics {incompatible} incompatible with task '{task_type}'. "
                    f"Falling back to default metrics: {default_metrics}. "
                    f"Compatible metrics in config: {compatible if compatible else 'none'}"
                )
                final_metrics = default_metrics
            else:
                # All config metrics are compatible
                final_metrics = compatible
        else:
            # No metrics specified → auto-select based on task type
            final_metrics = default_metrics
            logger.info(f"Auto-selected metrics for task '{task_type}': {final_metrics}")

        # Build metric params (add task and num_classes for classification metrics)
        # CRITICAL FIX: TorchMetrics v1.0+ requires 'task' parameter for classification metrics
        # The 'task' must be set regardless of whether num_classes is provided
        metric_params = params.copy()
        if is_classification:
            # Determine task type based on task_type and num_classes
            # link_prediction is always binary (predicting edge existence: yes/no)
            # Other classification tasks depend on num_classes
            if task_lower == "link_prediction":
                # Link prediction is inherently binary (edge exists or not)
                metric_params["task"] = "binary"
                logger.debug("link_prediction: Setting task='binary' for classification metrics")
            elif num_classes is not None:
                metric_params["num_classes"] = num_classes
                if num_classes == 2:
                    metric_params["task"] = "binary"
                else:
                    metric_params["task"] = "multiclass"
            else:
                # Default to binary when num_classes not specified
                # This is a safe default that works for most cases
                # For multiclass, user should provide num_classes in config
                metric_params["task"] = "binary"
                logger.debug(
                    f"Classification task '{task_type}' without num_classes: "
                    f"defaulting to task='binary'. Provide num_classes for multiclass."
                )

        # Instantiate metrics
        metrics = {}
        for name in final_metrics:
            try:
                metrics[name] = cls.get_metric(name, metric_params, device)
            except Exception as e:
                logger.warning(f"Failed to create metric '{name}': {e}")
                continue

        if not metrics:
            raise RuntimeError(
                f"No metrics could be created for task '{task_type}'. Tried: {final_metrics}"
            )

        return metrics

    @classmethod
    def get_default_metrics_for_task(cls, task_type: str) -> list[str]:
        """
        Get the default metric names for a task type.

        Args:
            task_type: Task type string

        Returns:
            List of default metric names for the task type
        """
        task_lower = task_type.lower() if task_type else ""
        return cls._task_to_default_metrics.get(task_lower, ["mse", "mae"])

    @classmethod
    def is_metric_compatible_with_task(cls, metric_name: str, task_type: str) -> bool:
        """
        Check if a metric is compatible with a task type.

        Args:
            metric_name: Metric name
            task_type: Task type string

        Returns:
            True if compatible, False otherwise
        """
        name_lower = metric_name.lower() if metric_name else ""
        task_lower = task_type.lower() if task_type else ""

        is_classification_task = "classification" in task_lower or task_lower == "link_prediction"
        is_regression_task = "regression" in task_lower

        if is_classification_task and name_lower in cls._regression_metrics:
            return False
        return not (is_regression_task and name_lower in cls._classification_metrics)

    @classmethod
    def create_metric_collection(
        cls,
        task_type: str,
        metric_names: list[str] | None = None,
        params: dict[str, Any] | None = None,
        device: torch.device | None = None,
        num_classes: int | None = None,
        prefix: str = "",
    ) -> "MetricCollection":
        """
        Create a TorchMetrics MetricCollection for efficient batch computation.

        DYNAMIC: Uses get_metrics_for_task() for task-aware selection
        PRODUCTION-READY: MetricCollection provides compute_groups optimization
        FUTURE-PROOF: MetricCollection is the recommended pattern for multiple metrics

        Args:
            task_type: Task type string
            metric_names: Optional list of metric names from config
            params: Optional parameters for metrics
            device: Device to place metrics on
            num_classes: Number of classes for classification
            prefix: Prefix for metric names (e.g., "val_", "test_")

        Returns:
            MetricCollection instance
        """
        if not TORCHMETRICS_AVAILABLE:
            raise RuntimeError(
                "MetricCollection requires TorchMetrics. "
                "Install with: pip install torchmetrics>=1.0.0"
            )

        metrics = cls.get_metrics_for_task(
            task_type=task_type,
            metric_names=metric_names,
            params=params,
            device=device,
            num_classes=num_classes,
        )

        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        collection = MetricCollection(metrics)

        if device is not None:
            collection = collection.to(device)

        return collection


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_metric(
    name: str, params: dict[str, Any] | None = None, device: torch.device | None = None
) -> nn.Module:
    """
    Convenience function to get a metric by name.

    Example:
        >>> from milia_pipeline.models.training import get_metric
        >>> mse = get_metric("mse")
    """
    return MetricsRegistry.get_metric(name, params, device)


def get_metrics_for_task(
    task_type: str,
    metric_names: list[str] | None = None,
    params: dict[str, Any] | None = None,
    device: torch.device | None = None,
    num_classes: int | None = None,
) -> dict[str, nn.Module]:
    """
    Convenience function to get task-aware metrics.

    DYNAMIC: Automatically selects appropriate metrics based on task_type
    PRODUCTION-READY: Validates config metrics, falls back to defaults
    FUTURE-PROOF: Extensible via MetricsRegistry class attributes

    Example:
        >>> from milia_pipeline.models.training import get_metrics_for_task
        >>> # Auto-select for regression
        >>> metrics = get_metrics_for_task('graph_regression')
        >>>
        >>> # With explicit config (will warn if incompatible)
        >>> metrics = get_metrics_for_task('graph_classification', ['accuracy', 'f1'])
    """
    return MetricsRegistry.get_metrics_for_task(
        task_type, metric_names, params, device, num_classes
    )


def list_metrics() -> list[str]:
    """
    Convenience function to list available metrics.

    Example:
        >>> from milia_pipeline.models.training import list_metrics
        >>> print(list_metrics())
    """
    return MetricsRegistry.list_available()


def get_default_metrics_for_task(task_type: str) -> list[str]:
    """
    Convenience function to get default metric names for a task type.

    Example:
        >>> from milia_pipeline.models.training import get_default_metrics_for_task
        >>> get_default_metrics_for_task('graph_classification')
        ['accuracy', 'f1', 'precision', 'recall', 'auroc']
    """
    return MetricsRegistry.get_default_metrics_for_task(task_type)


def is_metric_compatible_with_task(metric_name: str, task_type: str) -> bool:
    """
    Convenience function to check metric-task compatibility.

    Example:
        >>> from milia_pipeline.models.training import is_metric_compatible_with_task
        >>> is_metric_compatible_with_task('mse', 'graph_classification')
        False
    """
    return MetricsRegistry.is_metric_compatible_with_task(metric_name, task_type)


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

if TORCHMETRICS_AVAILABLE:
    logger.info(
        f"metrics module loaded - {len(MetricsRegistry._metrics)} metrics available (TorchMetrics backend)"
    )
else:
    logger.warning(
        "metrics module loaded - TorchMetrics not available. "
        "Install with: pip install torchmetrics>=1.0.0"
    )
