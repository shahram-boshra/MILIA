"""
Loss Functions Registry

Centralized registry for loss functions with:
- Common PyTorch losses (MSE, CrossEntropy, etc.)
- Custom graph-specific losses (Focal, etc.)
- Easy instantiation via string names

Author: milia Team
Version: 1.0.0
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# CUSTOM LOSS FUNCTIONS
# =============================================================================


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Focal Loss = -α(1-p)^γ * log(p)

    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples
        gamma: Focusing parameter for hard examples (default: 2.0)
        reduction: Specifies the reduction to apply ('none', 'mean', 'sum')

    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
        https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (logits) of shape (N, C) or (N,)
            targets: Ground truth labels of shape (N,)

        Returns:
            Focal loss value
        """
        # Convert to probabilities
        p = torch.sigmoid(inputs)

        # Calculate focal loss
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class WeightedMSELoss(nn.Module):
    """
    Weighted Mean Squared Error Loss.

    Allows per-sample or per-feature weighting of MSE loss.

    Args:
        reduction: Specifies the reduction to apply ('none', 'mean', 'sum')
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            inputs: Predictions
            targets: Ground truth
            weights: Optional weights of same shape as inputs/targets

        Returns:
            Weighted MSE loss
        """
        mse = (inputs - targets) ** 2

        if weights is not None:
            mse = mse * weights

        if self.reduction == "mean":
            return mse.mean()
        elif self.reduction == "sum":
            return mse.sum()
        else:
            return mse


class RMSELoss(nn.Module):
    """
    Root Mean Squared Error Loss.

    RMSE = sqrt(MSE)
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(inputs, targets))


# =============================================================================
# LOSS REGISTRY
# =============================================================================


class LossRegistry:
    """
    Centralized registry for loss functions.

    Provides easy access to common PyTorch losses and custom implementations
    via string names.

    Usage:
        >>> from milia_pipeline.models import LossRegistry
        >>> loss_fn = LossRegistry.get_loss("mse")
        >>> loss_fn = LossRegistry.get_loss("focal", {"alpha": 0.5, "gamma": 2.0})
        >>> available = LossRegistry.list_available()
    """

    _losses = {
        # =====================================================================
        # REGRESSION LOSSES
        # =====================================================================
        "mse": nn.MSELoss,
        "mae": nn.L1Loss,
        "l1": nn.L1Loss,  # Alias for MAE
        "huber": nn.HuberLoss,
        "smooth_l1": nn.SmoothL1Loss,
        "rmse": RMSELoss,
        "weighted_mse": WeightedMSELoss,
        # =====================================================================
        # CLASSIFICATION LOSSES
        # =====================================================================
        "cross_entropy": nn.CrossEntropyLoss,
        "ce": nn.CrossEntropyLoss,  # Alias
        "nll": nn.NLLLoss,
        "bce": nn.BCELoss,
        "bce_with_logits": nn.BCEWithLogitsLoss,
        "focal": FocalLoss,
        # =====================================================================
        # MULTI-LABEL LOSSES
        # =====================================================================
        "multilabel_soft_margin": nn.MultiLabelSoftMarginLoss,
        # =====================================================================
        # RANKING LOSSES
        # =====================================================================
        "margin_ranking": nn.MarginRankingLoss,
        "triplet_margin": nn.TripletMarginLoss,
        # =====================================================================
        # OTHER LOSSES
        # =====================================================================
        "kl_div": nn.KLDivLoss,
        "poisson_nll": nn.PoissonNLLLoss,
        "cosine_embedding": nn.CosineEmbeddingLoss,
    }

    @classmethod
    def get_loss(cls, name: str, params: dict[str, Any] | None = None) -> nn.Module:
        """
        Get loss function by name.

        Dynamically filters parameters to only those accepted by the loss class,
        preventing errors when unsupported parameters are passed (e.g., 'alpha'
        parameter intended for FocalLoss being passed to MSELoss).

        Args:
            name: Loss function name (e.g., "mse", "cross_entropy", "focal")
            params: Optional dictionary of parameters to pass to loss constructor.
                   Invalid parameters are automatically filtered out.

        Returns:
            Instantiated loss function

        Raises:
            ValueError: If loss name not found in registry

        Example:
            >>> # Simple usage
            >>> loss_fn = LossRegistry.get_loss("mse")
            >>>
            >>> # With parameters
            >>> loss_fn = LossRegistry.get_loss("focal", {
            ...     "alpha": 0.25,
            ...     "gamma": 2.0,
            ...     "reduction": "mean"
            ... })
            >>>
            >>> # Safe usage - invalid params are filtered
            >>> loss_fn = LossRegistry.get_loss("mse", {"alpha": 0.5})
            >>> # Works! 'alpha' is filtered out since MSELoss doesn't accept it
        """
        if name not in cls._losses:
            available = ", ".join(sorted(cls._losses.keys()))
            raise ValueError(f"Unknown loss function: '{name}'. Available losses: {available}")

        loss_cls = cls._losses[name]
        params = params or {}

        # Filter parameters to only those accepted by the loss class
        filtered_params = cls._filter_params(loss_cls, params)

        try:
            loss_fn = loss_cls(**filtered_params)

            if filtered_params:
                logger.debug(f"Initialized {name} loss with params: {filtered_params}")
            else:
                logger.debug(f"Initialized {name} loss with default params")

            # Log filtered out params at debug level
            ignored = set(params.keys()) - set(filtered_params.keys())
            if ignored:
                logger.debug(f"Loss '{name}': ignored unsupported params {ignored}")

            return loss_fn

        except TypeError as e:
            raise ValueError(
                f"Invalid parameters for loss '{name}': {filtered_params}. Error: {e}"
            ) from e

    @classmethod
    def _filter_params(cls, target_cls: type, params: dict[str, Any]) -> dict[str, Any]:
        """
        Filter parameters to only those accepted by the target class constructor.

        Uses inspect.signature() for dynamic introspection, ensuring only valid
        parameters are passed to the constructor. This is DYNAMIC (works with any
        class), PRODUCTION-READY (handles edge cases), and FUTURE-PROOF (no
        hardcoded parameter lists).

        Args:
            target_cls: The class whose constructor parameters to check
            params: Dictionary of parameters to filter

        Returns:
            Filtered dictionary containing only valid parameters
        """
        import inspect

        if not params:
            return {}

        try:
            sig = inspect.signature(target_cls.__init__)
            valid_param_names = set(sig.parameters.keys()) - {"self"}
            filtered = {k: v for k, v in params.items() if k in valid_param_names}
            return filtered
        except (ValueError, TypeError):
            # Fallback: return original params if introspection fails
            # (e.g., for built-in types or C extensions without signatures)
            return params

    @classmethod
    def list_available(cls) -> list[str]:
        """
        List all available loss function names.

        Returns:
            List of loss function names

        Example:
            >>> losses = LossRegistry.list_available()
            >>> print(f"Available losses: {', '.join(losses)}")
        """
        return sorted(cls._losses.keys())

    @classmethod
    def get_loss_info(cls, name: str) -> dict[str, Any]:
        """
        Get information about a loss function.

        Args:
            name: Loss function name

        Returns:
            Dictionary with loss information including valid parameters

        Example:
            >>> info = LossRegistry.get_loss_info("focal")
            >>> print(info['valid_params'])
        """
        if name not in cls._losses:
            raise ValueError(f"Unknown loss function: '{name}'")

        loss_cls = cls._losses[name]

        return {
            "name": name,
            "class": loss_cls.__name__,
            "module": loss_cls.__module__,
            "doc": loss_cls.__doc__,
            "valid_params": cls.get_valid_params(name),
        }

    @classmethod
    def get_valid_params(cls, name: str) -> dict[str, Any]:
        """
        Get valid parameters for a loss function using introspection.

        Uses inspect.signature() to dynamically discover what parameters
        the loss function's constructor accepts, along with their default values.

        Args:
            name: Loss function name

        Returns:
            Dictionary mapping parameter names to their default values
            (inspect.Parameter.empty if no default)

        Example:
            >>> params = LossRegistry.get_valid_params("focal")
            >>> print(params)
            {'alpha': 0.25, 'gamma': 2.0, 'reduction': 'mean'}

            >>> params = LossRegistry.get_valid_params("mse")
            >>> print(params)
            {'size_average': None, 'reduce': None, 'reduction': 'mean'}
        """
        import inspect

        if name not in cls._losses:
            raise ValueError(f"Unknown loss function: '{name}'")

        loss_cls = cls._losses[name]

        try:
            sig = inspect.signature(loss_cls.__init__)
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
            # Cannot introspect (e.g., built-in C extension)
            return {}

    @classmethod
    def register_custom_loss(cls, name: str, loss_class: type, overwrite: bool = False):
        """
        Register a custom loss function.

        Args:
            name: Name to register loss under
            loss_class: Loss class (must be nn.Module subclass)
            overwrite: Whether to overwrite existing loss with same name

        Raises:
            ValueError: If name exists and overwrite=False
            TypeError: If loss_class is not nn.Module subclass

        Example:
            >>> class MyCustomLoss(nn.Module):
            ...     def forward(self, input, target):
            ...         return ((input - target) ** 2).mean()
            >>>
            >>> LossRegistry.register_custom_loss("my_loss", MyCustomLoss)
            >>> loss_fn = LossRegistry.get_loss("my_loss")
        """
        if not issubclass(loss_class, nn.Module):
            raise TypeError(f"loss_class must be a subclass of nn.Module, got {type(loss_class)}")

        if name in cls._losses and not overwrite:
            raise ValueError(f"Loss '{name}' already registered. Use overwrite=True to replace.")

        cls._losses[name] = loss_class
        logger.info(f"Registered custom loss: '{name}'")

    # =========================================================================
    # TASK-AWARE LOSS SELECTION (DYNAMIC, PRODUCTION-READY, FUTURE-PROOF)
    # =========================================================================

    # Classification vs Regression loss categories
    _classification_losses = {"cross_entropy", "ce", "nll", "bce", "bce_with_logits", "focal"}
    _regression_losses = {"mse", "mae", "l1", "huber", "smooth_l1", "rmse", "weighted_mse"}

    # Task type to default loss mapping
    _task_to_default_loss = {
        # Classification tasks → CrossEntropyLoss (expects Long targets)
        "graph_classification": "cross_entropy",
        "node_classification": "cross_entropy",
        "edge_classification": "cross_entropy",
        # Regression tasks → MSELoss (expects Float targets)
        "graph_regression": "mse",
        "node_regression": "mse",
        "edge_regression": "mse",
        # Link prediction → BCEWithLogitsLoss (binary classification)
        "link_prediction": "bce_with_logits",
    }

    @classmethod
    def get_loss_for_task(
        cls, task_type: str, name: str | None = None, params: dict[str, Any] | None = None
    ) -> nn.Module:
        """
        Get loss function with task-aware automatic selection.

        DYNAMIC: Automatically selects appropriate loss based on task_type when
                 name is not provided or is incompatible with the task type.
        PRODUCTION-READY: Handles all task types with appropriate loss functions,
                          provides clear logging for transparency, prevents dtype
                          mismatch errors between loss functions and targets.
        FUTURE-PROOF: Extensible via _task_to_default_loss and _classification_losses/
                      _regression_losses class attributes. Easy to add new task types
                      or loss functions without modifying this method.

        Loss Selection Strategy:
        1. If name is provided and compatible with task → use it
        2. If name is provided but incompatible → override with warning
        3. If name is None → auto-select based on task_type

        This prevents the common dtype mismatch errors:
        - "Found dtype Long but expected Float" (regression loss with classification targets)
        - "Expected Long but got Float" (classification loss with regression targets)

        Args:
            task_type: Task type string (e.g., 'graph_classification', 'graph_regression')
            name: Optional loss function name. If None, auto-selects based on task_type.
            params: Optional dictionary of parameters to pass to loss constructor.

        Returns:
            Instantiated loss function appropriate for the task type

        Example:
            >>> # Auto-select for classification task
            >>> loss_fn = LossRegistry.get_loss_for_task('graph_classification')
            >>> # Returns CrossEntropyLoss

            >>> # Override with compatible loss
            >>> loss_fn = LossRegistry.get_loss_for_task('graph_classification', 'nll')
            >>> # Returns NLLLoss (compatible with classification)

            >>> # Incompatible loss gets overridden with warning
            >>> loss_fn = LossRegistry.get_loss_for_task('graph_classification', 'mse')
            >>> # Warning logged, returns CrossEntropyLoss (auto-corrected)
        """
        task_lower = task_type.lower() if task_type else ""
        name_lower = name.lower() if name else ""

        # Determine task category
        is_classification = "classification" in task_lower
        is_regression = "regression" in task_lower

        final_loss_name = None

        if name_lower:
            # User specified a loss - check compatibility
            if is_classification and name_lower in cls._regression_losses:
                # INCOMPATIBLE: regression loss for classification task
                default = cls._task_to_default_loss.get(task_lower, "cross_entropy")
                logger.warning(
                    f"Task '{task_type}' is classification but loss '{name}' is for regression. "
                    f"Classification targets are Long (class indices) but '{name}' expects Float. "
                    f"Auto-selecting '{default}' to prevent dtype mismatch error."
                )
                final_loss_name = default
            elif is_regression and name_lower in cls._classification_losses:
                # INCOMPATIBLE: classification loss for regression task
                default = cls._task_to_default_loss.get(task_lower, "mse")
                logger.warning(
                    f"Task '{task_type}' is regression but loss '{name}' is for classification. "
                    f"Regression targets are Float but '{name}' expects Long class indices. "
                    f"Auto-selecting '{default}' to prevent dtype mismatch error."
                )
                final_loss_name = default
            else:
                # Compatible - use user's choice
                final_loss_name = name_lower
        else:
            # No loss specified - auto-select based on task type
            final_loss_name = cls._task_to_default_loss.get(task_lower, "mse")
            logger.info(f"Auto-selected loss '{final_loss_name}' for task type '{task_type}'")

        return cls.get_loss(final_loss_name, params)

    @classmethod
    def get_default_loss_for_task(cls, task_type: str) -> str:
        """
        Get the default loss function name for a task type.

        Args:
            task_type: Task type string

        Returns:
            Default loss function name for the task type

        Example:
            >>> LossRegistry.get_default_loss_for_task('graph_classification')
            'cross_entropy'
            >>> LossRegistry.get_default_loss_for_task('graph_regression')
            'mse'
        """
        task_lower = task_type.lower() if task_type else ""
        return cls._task_to_default_loss.get(task_lower, "mse")

    @classmethod
    def is_loss_compatible_with_task(cls, loss_name: str, task_type: str) -> bool:
        """
        Check if a loss function is compatible with a task type.

        Args:
            loss_name: Loss function name
            task_type: Task type string

        Returns:
            True if compatible, False otherwise

        Example:
            >>> LossRegistry.is_loss_compatible_with_task('mse', 'graph_regression')
            True
            >>> LossRegistry.is_loss_compatible_with_task('mse', 'graph_classification')
            False
        """
        name_lower = loss_name.lower() if loss_name else ""
        task_lower = task_type.lower() if task_type else ""

        is_classification = "classification" in task_lower or task_lower == "link_prediction"
        is_regression = "regression" in task_lower

        if is_classification and name_lower in cls._regression_losses:
            return False
        return not (is_regression and name_lower in cls._classification_losses)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_loss(name: str, params: dict[str, Any] | None = None) -> nn.Module:
    """
    Convenience function to get loss function.

    Example:
        >>> from milia_pipeline.models import get_loss
        >>> loss_fn = get_loss("mse")
    """
    return LossRegistry.get_loss(name, params)


def get_loss_for_task(
    task_type: str, name: str | None = None, params: dict[str, Any] | None = None
) -> nn.Module:
    """
    Convenience function to get task-aware loss function.

    DYNAMIC: Automatically selects appropriate loss based on task_type when
             name is not provided or is incompatible with the task type.
    PRODUCTION-READY: Handles all task types, prevents dtype mismatch errors.
    FUTURE-PROOF: Extensible via LossRegistry class attributes.

    Example:
        >>> from milia_pipeline.models import get_loss_for_task
        >>> # Auto-select for classification
        >>> loss_fn = get_loss_for_task('graph_classification')
        >>> # Returns CrossEntropyLoss

        >>> # Override with specific loss
        >>> loss_fn = get_loss_for_task('graph_regression', 'huber')
        >>> # Returns HuberLoss
    """
    return LossRegistry.get_loss_for_task(task_type, name, params)


def list_losses() -> list[str]:
    """
    Convenience function to list available losses.

    Example:
        >>> from milia_pipeline.models import list_losses
        >>> print(list_losses())
    """
    return LossRegistry.list_available()


def get_default_loss_for_task(task_type: str) -> str:
    """
    Convenience function to get default loss name for a task type.

    Example:
        >>> from milia_pipeline.models import get_default_loss_for_task
        >>> get_default_loss_for_task('graph_classification')
        'cross_entropy'
    """
    return LossRegistry.get_default_loss_for_task(task_type)


def is_loss_compatible_with_task(loss_name: str, task_type: str) -> bool:
    """
    Convenience function to check loss-task compatibility.

    Example:
        >>> from milia_pipeline.models import is_loss_compatible_with_task
        >>> is_loss_compatible_with_task('mse', 'graph_classification')
        False
    """
    return LossRegistry.is_loss_compatible_with_task(loss_name, task_type)


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info(f"loss_functions module loaded - {len(LossRegistry._losses)} losses available")
