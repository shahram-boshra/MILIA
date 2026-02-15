"""
Schedulers Registry

Centralized registry for learning rate schedulers with:
- Common PyTorch schedulers
- Easy instantiation via string names
- Integration with Trainer

Author: milia Team
Version: 1.0.0
"""

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


# =============================================================================
# SCHEDULER REGISTRY
# =============================================================================


class SchedulerRegistry:
    """
    Centralized registry for learning rate schedulers.

    Provides easy access to common PyTorch LR schedulers via string names.

    Usage:
        >>> from milia_pipeline.models import SchedulerRegistry
        >>> scheduler = SchedulerRegistry.get_scheduler(
        ...     "cosine_annealing",
        ...     optimizer,
        ...     {"T_max": 100}
        ... )
    """

    _schedulers = {
        # =====================================================================
        # ADAPTIVE SCHEDULERS (metric-based)
        # =====================================================================
        "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        # =====================================================================
        # STEP-BASED SCHEDULERS
        # =====================================================================
        "step_lr": torch.optim.lr_scheduler.StepLR,
        "multistep_lr": torch.optim.lr_scheduler.MultiStepLR,
        "exponential_lr": torch.optim.lr_scheduler.ExponentialLR,
        # =====================================================================
        # COSINE ANNEALING SCHEDULERS
        # =====================================================================
        "cosine_annealing": torch.optim.lr_scheduler.CosineAnnealingLR,
        "cosine_annealing_warm_restarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        # =====================================================================
        # CYCLIC SCHEDULERS
        # =====================================================================
        "cyclic_lr": torch.optim.lr_scheduler.CyclicLR,
        "one_cycle": torch.optim.lr_scheduler.OneCycleLR,
        # =====================================================================
        # POLYNOMIAL SCHEDULERS
        # =====================================================================
        "polynomial_lr": torch.optim.lr_scheduler.PolynomialLR,
        # =====================================================================
        # LINEAR SCHEDULERS
        # =====================================================================
        "linear_lr": torch.optim.lr_scheduler.LinearLR,
        # =====================================================================
        # CHAINED/SEQUENTIAL SCHEDULERS
        # =====================================================================
        "chained": torch.optim.lr_scheduler.ChainedScheduler,
        "sequential": torch.optim.lr_scheduler.SequentialLR,
        # =====================================================================
        # CONSTANT SCHEDULERS
        # =====================================================================
        "constant_lr": torch.optim.lr_scheduler.ConstantLR,
    }

    # Default parameters for common schedulers
    _defaults = {
        "reduce_on_plateau": {
            "mode": "min",
            "factor": 0.1,
            "patience": 10,
            "threshold": 1e-4,
            "threshold_mode": "rel",
            "cooldown": 0,
            "min_lr": 0,
            "eps": 1e-8,
        },
        "step_lr": {"step_size": 30, "gamma": 0.1},
        "exponential_lr": {"gamma": 0.95},
        "cosine_annealing": {"T_max": 100, "eta_min": 0},
        "cyclic_lr": {"base_lr": 0.001, "max_lr": 0.01, "step_size_up": 2000, "mode": "triangular"},
    }

    # Schedulers that require metric (e.g., ReduceLROnPlateau)
    _metric_based = {"reduce_on_plateau"}

    @classmethod
    def get_scheduler(
        cls, name: str, optimizer: torch.optim.Optimizer, params: dict[str, Any] | None = None
    ):
        """
        Get learning rate scheduler by name.

        Merges provided parameters with registry defaults and dynamically filters
        to only those accepted by the scheduler class. This prevents errors when
        unsupported parameters are passed.

        Args:
            name: Scheduler name (e.g., "cosine_annealing", "reduce_on_plateau")
            optimizer: PyTorch optimizer instance
            params: Optional dictionary of scheduler parameters.
                   These override registry defaults. Invalid parameters
                   are automatically filtered out.

        Returns:
            Instantiated scheduler

        Raises:
            ValueError: If scheduler name not found in registry

        Example:
            >>> # Simple usage with registry defaults
            >>> scheduler = SchedulerRegistry.get_scheduler(
            ...     "step_lr",
            ...     optimizer
            ... )
            >>>
            >>> # With custom parameters (merged with defaults)
            >>> scheduler = SchedulerRegistry.get_scheduler(
            ...     "cosine_annealing",
            ...     optimizer,
            ...     {"T_max": 100, "eta_min": 1e-6}
            ... )
            >>>
            >>> # Safe usage - invalid params are filtered
            >>> scheduler = SchedulerRegistry.get_scheduler(
            ...     "step_lr",
            ...     optimizer,
            ...     {"step_size": 10, "invalid_param": 123}
            ... )
            >>> # Works! 'invalid_param' is filtered out
            >>>
            >>> # Check if scheduler needs metric in step()
            >>> if SchedulerRegistry.is_metric_based(name):
            ...     scheduler.step(val_loss)
            ... else:
            ...     scheduler.step()
        """
        if name not in cls._schedulers:
            available = ", ".join(sorted(cls._schedulers.keys()))
            raise ValueError(f"Unknown scheduler: '{name}'. Available schedulers: {available}")

        sched_cls = cls._schedulers[name]

        # Merge defaults with provided params (provided params take precedence)
        merged_params = {**cls._defaults.get(name, {}), **(params or {})}

        # Filter parameters to only those accepted by the scheduler class
        filtered_params = cls._filter_params(sched_cls, merged_params)

        try:
            scheduler = sched_cls(optimizer, **filtered_params)

            if filtered_params:
                logger.debug(f"Initialized {name} scheduler with params: {filtered_params}")
            else:
                logger.debug(f"Initialized {name} scheduler with default params")

            # Log filtered out params at debug level
            ignored = set(merged_params.keys()) - set(filtered_params.keys())
            if ignored:
                logger.debug(f"Scheduler '{name}': ignored unsupported params {ignored}")

            return scheduler

        except TypeError as e:
            raise ValueError(
                f"Invalid parameters for scheduler '{name}': {filtered_params}. Error: {e}"
            ) from e

    @classmethod
    def _filter_params(cls, target_cls: type, params: dict[str, Any]) -> dict[str, Any]:
        """
        Filter parameters to only those accepted by the target scheduler constructor.

        Uses inspect.signature() for dynamic introspection, ensuring only valid
        parameters are passed to the constructor. This is DYNAMIC (works with any
        scheduler), PRODUCTION-READY (handles edge cases), and FUTURE-PROOF (no
        hardcoded parameter lists).

        Args:
            target_cls: The scheduler class whose constructor parameters to check
            params: Dictionary of parameters to filter

        Returns:
            Filtered dictionary containing only valid parameters
        """
        import inspect

        if not params:
            return {}

        try:
            sig = inspect.signature(target_cls.__init__)
            # Exclude 'self' and 'optimizer' (the optimizer argument)
            valid_param_names = set(sig.parameters.keys()) - {"self", "optimizer"}
            filtered = {k: v for k, v in params.items() if k in valid_param_names}
            return filtered
        except (ValueError, TypeError):
            # Fallback: return original params if introspection fails
            return params

    @classmethod
    def list_available(cls) -> list[str]:
        """
        List all available scheduler names.

        Returns:
            List of scheduler names

        Example:
            >>> schedulers = SchedulerRegistry.list_available()
            >>> print(f"Available schedulers: {', '.join(schedulers)}")
        """
        return sorted(cls._schedulers.keys())

    @classmethod
    def is_metric_based(cls, name: str) -> bool:
        """
        Check if scheduler requires a metric (e.g., ReduceLROnPlateau).

        Args:
            name: Scheduler name

        Returns:
            True if scheduler requires metric in step() call

        Example:
            >>> if SchedulerRegistry.is_metric_based("reduce_on_plateau"):
            ...     scheduler.step(val_loss)
            ... else:
            ...     scheduler.step()
        """
        return name in cls._metric_based

    @classmethod
    def get_scheduler_info(cls, name: str) -> dict[str, Any]:
        """
        Get information about a scheduler.

        Args:
            name: Scheduler name

        Returns:
            Dictionary with scheduler information including valid parameters

        Example:
            >>> info = SchedulerRegistry.get_scheduler_info("cosine_annealing")
            >>> print(info['valid_params'])
        """
        if name not in cls._schedulers:
            raise ValueError(f"Unknown scheduler: '{name}'")

        sched_cls = cls._schedulers[name]

        return {
            "name": name,
            "class": sched_cls.__name__,
            "module": sched_cls.__module__,
            "metric_based": name in cls._metric_based,
            "default_params": cls._defaults.get(name, {}),
            "valid_params": cls.get_valid_params(name),
            "doc": sched_cls.__doc__,
        }

    @classmethod
    def get_default_params(cls, name: str) -> dict[str, Any]:
        """
        Get default parameters for a scheduler from registry defaults.

        Note: These are registry-defined defaults, not introspected from the class.
        Use get_valid_params() to see all valid parameters with their class defaults.

        Args:
            name: Scheduler name

        Returns:
            Dictionary of registry default parameters

        Example:
            >>> defaults = SchedulerRegistry.get_default_params("step_lr")
            >>> print(defaults)
            {'step_size': 30, 'gamma': 0.1}
        """
        if name not in cls._schedulers:
            raise ValueError(f"Unknown scheduler: '{name}'")

        return cls._defaults.get(name, {}).copy()

    @classmethod
    def get_valid_params(cls, name: str) -> dict[str, Any]:
        """
        Get valid parameters for a scheduler using introspection.

        Uses inspect.signature() to dynamically discover what parameters
        the scheduler's constructor accepts, along with their default values.

        Args:
            name: Scheduler name

        Returns:
            Dictionary mapping parameter names to their default values
            (None if no default / required parameter)

        Example:
            >>> params = SchedulerRegistry.get_valid_params("reduce_on_plateau")
            >>> print(params)
            {'mode': 'min', 'factor': 0.1, 'patience': 10, ...}
        """
        import inspect

        if name not in cls._schedulers:
            raise ValueError(f"Unknown scheduler: '{name}'")

        sched_cls = cls._schedulers[name]

        try:
            sig = inspect.signature(sched_cls.__init__)
            params = {}
            for param_name, param in sig.parameters.items():
                if param_name in ("self", "optimizer"):
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
    def register_custom_scheduler(
        cls,
        name: str,
        scheduler_class: type,
        default_params: dict[str, Any] | None = None,
        metric_based: bool = False,
        overwrite: bool = False,
    ):
        """
        Register a custom scheduler.

        Args:
            name: Name to register scheduler under
            scheduler_class: Scheduler class (must be LRScheduler subclass)
            default_params: Optional default parameters
            metric_based: Whether scheduler requires metric in step()
            overwrite: Whether to overwrite existing scheduler with same name

        Raises:
            ValueError: If name exists and overwrite=False

        Example:
            >>> class MyScheduler(torch.optim.lr_scheduler._LRScheduler):
            ...     def __init__(self, optimizer, my_param=0.5, last_epoch=-1):
            ...         self.my_param = my_param
            ...         super().__init__(optimizer, last_epoch)
            ...
            ...     def get_lr(self):
            ...         return [base_lr * self.my_param for base_lr in self.base_lrs]
            >>>
            >>> SchedulerRegistry.register_custom_scheduler(
            ...     "my_scheduler",
            ...     MyScheduler,
            ...     {"my_param": 0.5}
            ... )
        """
        if name in cls._schedulers and not overwrite:
            raise ValueError(
                f"Scheduler '{name}' already registered. Use overwrite=True to replace."
            )

        cls._schedulers[name] = scheduler_class

        if default_params:
            cls._defaults[name] = default_params

        if metric_based:
            cls._metric_based.add(name)

        logger.info(f"Registered custom scheduler: '{name}'")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_scheduler(
    name: str, optimizer: torch.optim.Optimizer, params: dict[str, Any] | None = None
):
    """
    Convenience function to get scheduler.

    Example:
        >>> from milia_pipeline.models import get_scheduler
        >>> scheduler = get_scheduler("cosine_annealing", optimizer, {"T_max": 100})
    """
    return SchedulerRegistry.get_scheduler(name, optimizer, params)


def list_schedulers() -> list[str]:
    """
    Convenience function to list available schedulers.

    Example:
        >>> from milia_pipeline.models import list_schedulers
        >>> print(list_schedulers())
    """
    return SchedulerRegistry.list_available()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    warmup_start_lr: float = 1e-6,
    after_scheduler_name: str = "cosine_annealing",
    after_scheduler_params: dict[str, Any] | None = None,
) -> torch.optim.lr_scheduler.SequentialLR:
    """
    Create a scheduler with warmup followed by another scheduler.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for warmup
        total_epochs: Total number of epochs
        warmup_start_lr: Starting LR for warmup
        after_scheduler_name: Scheduler to use after warmup
        after_scheduler_params: Parameters for after_scheduler

    Returns:
        SequentialLR combining warmup and main scheduler

    Example:
        >>> scheduler = create_warmup_scheduler(
        ...     optimizer,
        ...     warmup_epochs=10,
        ...     total_epochs=100,
        ...     after_scheduler_name="cosine_annealing"
        ... )
    """
    # Warmup scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_lr / optimizer.param_groups[0]["lr"],
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    # Main scheduler
    after_params = after_scheduler_params or {}
    if after_scheduler_name == "cosine_annealing" and "T_max" not in after_params:
        after_params["T_max"] = total_epochs - warmup_epochs

    main_scheduler = SchedulerRegistry.get_scheduler(after_scheduler_name, optimizer, after_params)

    # Combine
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
    )

    logger.info(
        f"Created warmup scheduler: {warmup_epochs} epochs warmup, then {after_scheduler_name}"
    )

    return scheduler


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info(f"schedulers module loaded - {len(SchedulerRegistry._schedulers)} schedulers available")
