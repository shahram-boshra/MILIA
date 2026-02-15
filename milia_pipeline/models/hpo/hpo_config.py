"""
HPO Configuration Module

Provides configuration classes for the Hyperparameter Optimization (HPO) system.

This module defines frozen BaseModel classes for configuring all aspects of HPO including:
- Search space parameter definitions
- Pruner configurations for early trial termination
- Sampler configurations for hyperparameter suggestion
- Study configurations for single and multi-objective optimization
- Master HPO configuration with the MASTER SWITCH

Pydantic V2 Migration (Phase 6b):
    - Migrated from @dataclass(frozen=True) to BaseModel with frozen=True
    - Uses @field_validator for individual field validation
    - Uses @model_validator(mode='before') for cross-field validation on frozen models
    - NON-BREAKING: Same constructor API and attribute access
    - Follows established pattern from config_containers.py

Author: Milia Team
Version: 1.1.0
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# =============================================================================
# PARAMETER TYPE IMPORTS (Single Source of Truth)
# =============================================================================
# Import ParamType from canonical location to ensure consistency across HPO module
from .search_spaces.param_types import ParamType, SearchSpaceParamConfig

# =============================================================================
# PRUNER CONFIGURATION
# =============================================================================


class PrunerType(Enum):
    """
    Supported pruner types for early trial termination.

    Pruners allow early stopping of unpromising trials to save computational
    resources. Different pruners use different strategies.

    Attributes:
        MEDIAN: Prune if intermediate value is worse than median of previous trials
        PERCENTILE: Prune if intermediate value is in bottom percentile
        HYPERBAND: Successive halving with multiple brackets
        SUCCESSIVE_HALVING: Aggressively prune poor performers
        THRESHOLD: Prune if value exceeds a fixed threshold
        PATIENT: Prune only after several consecutive poor reports
        NONE: No pruning (run all trials to completion)
    """

    MEDIAN = "median"
    PERCENTILE = "percentile"
    HYPERBAND = "hyperband"
    SUCCESSIVE_HALVING = "successive_halving"
    THRESHOLD = "threshold"
    PATIENT = "patient"
    NONE = "none"


class PrunerConfig(BaseModel, frozen=True):
    """
    Pruner configuration for early trial termination.

    Pattern: Follows frozen BaseModel pattern from config_containers.py (Pydantic V2)

    Pruners monitor intermediate training values and terminate trials
    that are unlikely to produce good results, saving computational time.

    Attributes:
        type: Pruner type (median, hyperband, percentile, etc.)
        n_startup_trials: Trials before pruning begins (warm-up period)
        n_warmup_steps: Epochs before pruning within a trial
        interval_steps: Check pruning every N steps
        percentile: For percentile pruner - bottom percentile to prune (default: 25.0)
        n_brackets: For Hyperband pruner - number of brackets (default: 4)

    Examples:
        >>> # Median pruner with defaults
        >>> PrunerConfig(type=PrunerType.MEDIAN)

        >>> # Hyperband pruner
        >>> PrunerConfig(type=PrunerType.HYPERBAND, n_brackets=3)

        >>> # No pruning
        >>> PrunerConfig(type=PrunerType.NONE)
    """

    type: PrunerType = PrunerType.MEDIAN
    n_startup_trials: int = 5
    n_warmup_steps: int = 10
    interval_steps: int = 1
    percentile: float = 25.0
    n_brackets: int = 4

    @field_validator("n_startup_trials")
    @classmethod
    def validate_n_startup_trials(cls, v: int) -> int:
        """Validate n_startup_trials is non-negative."""
        if v < 0:
            raise ValueError(f"n_startup_trials must be non-negative, got {v}")
        return v

    @field_validator("n_warmup_steps")
    @classmethod
    def validate_n_warmup_steps(cls, v: int) -> int:
        """Validate n_warmup_steps is non-negative."""
        if v < 0:
            raise ValueError(f"n_warmup_steps must be non-negative, got {v}")
        return v

    @field_validator("interval_steps")
    @classmethod
    def validate_interval_steps(cls, v: int) -> int:
        """Validate interval_steps is at least 1."""
        if v < 1:
            raise ValueError(f"interval_steps must be at least 1, got {v}")
        return v

    @model_validator(mode="before")
    @classmethod
    def validate_type_specific_fields(cls, data: Any) -> Any:
        """Validate fields based on pruner type."""
        if isinstance(data, dict):
            pruner_type = data.get("type", PrunerType.MEDIAN)

            # Convert string to enum if needed
            if isinstance(pruner_type, str):
                try:
                    pruner_type = PrunerType(pruner_type)
                except ValueError:
                    return data  # Let Pydantic handle the error

            # Percentile pruner validation
            if pruner_type == PrunerType.PERCENTILE:
                percentile = data.get("percentile", 25.0)
                if not (0 < percentile < 100):
                    raise ValueError(
                        f"percentile must be between 0 and 100 (exclusive), got {percentile}"
                    )

            # Hyperband pruner validation
            if pruner_type == PrunerType.HYPERBAND:
                n_brackets = data.get("n_brackets", 4)
                if n_brackets < 1:
                    raise ValueError(
                        f"n_brackets must be at least 1 for Hyperband pruner, got {n_brackets}"
                    )

        return data

    def to_dict(self) -> dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()


# =============================================================================
# SAMPLER CONFIGURATION
# =============================================================================


class SamplerType(Enum):
    """
    Supported sampler types for hyperparameter suggestion.

    Samplers determine how hyperparameters are suggested during optimization.
    Different samplers use different strategies from random to Bayesian.

    Attributes:
        TPE: Tree-structured Parzen Estimator (Bayesian optimization)
        RANDOM: Random sampling
        CMAES: Covariance Matrix Adaptation Evolution Strategy
        GRID: Grid search over discrete space
        NSGAII: Non-dominated Sorting Genetic Algorithm II (multi-objective)
        MOTPE: Multi-objective TPE
        QMCSAMPLER: Quasi-Monte Carlo sampling
    """

    TPE = "tpe"
    RANDOM = "random"
    CMAES = "cmaes"
    GRID = "grid"
    NSGAII = "nsgaii"
    MOTPE = "motpe"
    QMCSAMPLER = "qmc"


class SamplerConfig(BaseModel, frozen=True):
    """
    Sampler configuration for hyperparameter suggestion.

    Pattern: Follows frozen BaseModel pattern from config_containers.py (Pydantic V2)

    Controls how hyperparameters are sampled during optimization.
    TPE (Tree-structured Parzen Estimator) is the recommended default
    for most use cases.

    Attributes:
        type: Sampler type (tpe, random, cmaes, grid)
        n_startup_trials: Random trials before Bayesian optimization begins
        seed: Random seed for reproducibility
        multivariate: Whether to use multivariate TPE (considers parameter correlations)
        constant_liar: For parallel optimization - impute running trial values

    Examples:
        >>> # TPE with defaults
        >>> SamplerConfig(type=SamplerType.TPE)

        >>> # Random sampling with seed
        >>> SamplerConfig(type=SamplerType.RANDOM, seed=42)

        >>> # TPE for parallel optimization
        >>> SamplerConfig(type=SamplerType.TPE, constant_liar=True)
    """

    type: SamplerType = SamplerType.TPE
    n_startup_trials: int = 10
    seed: int | None = None
    multivariate: bool = True
    constant_liar: bool = False

    @field_validator("n_startup_trials")
    @classmethod
    def validate_n_startup_trials(cls, v: int) -> int:
        """Validate n_startup_trials is non-negative."""
        if v < 0:
            raise ValueError(f"n_startup_trials must be non-negative, got {v}")
        return v

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, v: int | None) -> int | None:
        """Validate seed is non-negative if provided."""
        if v is not None and v < 0:
            raise ValueError(f"seed must be non-negative, got {v}")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()


# =============================================================================
# STUDY CONFIGURATION
# =============================================================================


class OptimizationDirection(Enum):
    """
    Optimization direction for single-objective optimization.

    Attributes:
        MINIMIZE: Minimize the objective value (e.g., loss)
        MAXIMIZE: Maximize the objective value (e.g., accuracy)
    """

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class StudyConfig(BaseModel, frozen=True):
    """
    Optuna study configuration for single-objective optimization.

    Pattern: Follows frozen BaseModel pattern from config_containers.py (Pydantic V2)

    Defines the study settings including optimization direction,
    metric to optimize, persistence settings, and study naming.

    Attributes:
        direction: Optimization direction (minimize/maximize)
        metric: Metric name to optimize (must match Trainer output keys)
        study_name: Name for the study (used for persistence and identification)
        storage: Storage URL (None for in-memory, "sqlite:///file.db" for persistence)
        load_if_exists: Whether to resume existing study with same name

    Examples:
        >>> # Minimize validation loss (default)
        >>> StudyConfig(direction=OptimizationDirection.MINIMIZE, metric="val_loss")

        >>> # Maximize accuracy with persistence
        >>> StudyConfig(
        ...     direction=OptimizationDirection.MAXIMIZE,
        ...     metric="val_accuracy",
        ...     storage="sqlite:///hpo_study.db"
        ... )
    """

    direction: OptimizationDirection = OptimizationDirection.MINIMIZE
    metric: str = "val_loss"
    study_name: str = "milia_hpo"
    storage: str | None = None
    load_if_exists: bool = True

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        """Validate metric is not empty."""
        if not v:
            raise ValueError("metric cannot be empty")
        return v

    @field_validator("study_name")
    @classmethod
    def validate_study_name(cls, v: str) -> str:
        """Validate study_name is not empty."""
        if not v:
            raise ValueError("study_name cannot be empty")
        return v

    @field_validator("storage")
    @classmethod
    def validate_storage(cls, v: str | None) -> str | None:
        """Validate storage is a string or None."""
        if v is not None and not isinstance(v, str):
            raise ValueError(f"storage must be a string URL or None, got {type(v).__name__}")
        return v

    @property
    def is_multi_objective(self) -> bool:
        """Check if this is multi-objective optimization."""
        return False

    def to_dict(self) -> dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()


# =============================================================================
# MULTI-OBJECTIVE STUDY CONFIGURATION
# =============================================================================


class MultiObjectiveStudyConfig(BaseModel, frozen=True):
    """
    Configuration for multi-objective optimization.

    Pattern: Follows frozen BaseModel pattern from config_containers.py (Pydantic V2)

    Supports Pareto optimization for competing objectives
    (e.g., accuracy vs speed, MAE vs model size). Multi-objective
    optimization finds a set of Pareto-optimal solutions.

    Attributes:
        directions: Optimization direction per objective (tuple of "minimize"/"maximize")
        metrics: Metric names to optimize (must match Trainer output keys)
        study_name: Name for the study
        storage: Storage URL (None for in-memory)
        load_if_exists: Whether to resume existing study
        reference_point: Reference point for hypervolume calculation (optional)

    Examples:
        >>> # Minimize loss and training time
        >>> MultiObjectiveStudyConfig(
        ...     directions=("minimize", "minimize"),
        ...     metrics=("val_loss", "training_time")
        ... )

        >>> # Maximize accuracy, minimize model size
        >>> MultiObjectiveStudyConfig(
        ...     directions=("maximize", "minimize"),
        ...     metrics=("val_accuracy", "n_parameters"),
        ...     reference_point=(0.0, 1e8)
        ... )
    """

    directions: tuple[str, ...] = ("minimize",)
    metrics: tuple[str, ...] = ("val_loss",)
    study_name: str = "milia_hpo_multi"
    storage: str | None = None
    load_if_exists: bool = True
    reference_point: tuple[float, ...] | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_multi_objective_config(cls, data: Any) -> Any:
        """Validate multi-objective configuration consistency."""
        if isinstance(data, dict):
            directions = data.get("directions", ("minimize",))
            metrics = data.get("metrics", ("val_loss",))
            reference_point = data.get("reference_point")

            # Check directions and metrics have same length
            if len(directions) != len(metrics):
                raise ValueError(
                    f"directions and metrics must have same length. "
                    f"Got directions length {len(directions)}, metrics length {len(metrics)}"
                )

            # Validate each direction
            valid_directions = ("minimize", "maximize")
            for d in directions:
                if d not in valid_directions:
                    raise ValueError(f"Invalid direction: {d}. Must be 'minimize' or 'maximize'")

            # Multi-objective requires at least 2 metrics
            if len(metrics) < 2:
                raise ValueError(f"Multi-objective requires at least 2 metrics, got {len(metrics)}")

            # Validate metrics are non-empty
            for metric in metrics:
                if not metric:
                    raise ValueError("metrics cannot contain empty strings")

            # Validate reference_point length matches metrics
            if reference_point is not None:
                if len(reference_point) != len(metrics):
                    raise ValueError(
                        f"reference_point must match number of metrics. "
                        f"Got reference_point length {len(reference_point)}, metrics length {len(metrics)}"
                    )

            # Validate study_name
            study_name = data.get("study_name", "milia_hpo_multi")
            if not study_name:
                raise ValueError("study_name cannot be empty")

        return data

    @property
    def is_multi_objective(self) -> bool:
        """Check if this is multi-objective optimization."""
        return True

    def to_dict(self) -> dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()


# =============================================================================
# MASTER HPO CONFIGURATION
# =============================================================================


class HPOConfig(BaseModel, frozen=True):
    """
    Master HPO configuration.

    Pattern: Follows frozen BaseModel pattern from config_containers.py (Pydantic V2)

    This is the MASTER SWITCH configuration that enables/disables HPO
    and configures all aspects of hyperparameter optimization including
    backend selection, trial settings, search space, pruning, and sampling.

    Attributes:
        enabled: MASTER SWITCH - enables HPO when True
        backend: HPO backend ("optuna" or "ray_tune")
        n_trials: Number of trials to run
        timeout: Maximum time in seconds (None for no limit)
        n_jobs: Number of parallel jobs (1 for sequential)
        search_space: Hyperparameter search space configuration
        pruner: Pruner configuration for early trial termination
        sampler: Sampler configuration for hyperparameter suggestion
        study: Study configuration (single-objective)
        cv_folds: Number of cross-validation folds (0 for no CV)
        cv_metric_aggregation: How to aggregate CV metrics ("mean", "median", "min", "max")

    Examples:
        >>> # Disabled HPO (default)
        >>> config = HPOConfig(enabled=False)

        >>> # Basic HPO with 50 trials
        >>> config = HPOConfig(enabled=True, n_trials=50)

        >>> # HPO with cross-validation
        >>> config = HPOConfig(enabled=True, n_trials=100, cv_folds=5)

        >>> # Load from dictionary
        >>> config = HPOConfig.from_dict({'enabled': True, 'n_trials': 100})
    """

    enabled: bool = False
    backend: str = "optuna"
    n_trials: int = 100
    timeout: int | None = None
    n_jobs: int = 1
    search_space: dict[str, dict[str, SearchSpaceParamConfig]] = Field(default_factory=dict)
    pruner: PrunerConfig = Field(default_factory=PrunerConfig)
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)
    study: StudyConfig = Field(default_factory=StudyConfig)
    cv_folds: int = 0
    cv_metric_aggregation: str = "mean"
    task_type: str | None = None

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        """Validate backend is a supported HPO backend."""
        valid_backends = ("optuna", "ray_tune")
        if v not in valid_backends:
            raise ValueError(f"Unknown HPO backend: '{v}'. Must be 'optuna' or 'ray_tune'")
        return v

    @field_validator("n_trials")
    @classmethod
    def validate_n_trials(cls, v: int) -> int:
        """Validate n_trials is at least 1."""
        if v < 1:
            raise ValueError(f"n_trials must be at least 1, got {v}")
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int | None) -> int | None:
        """Validate timeout is positive or None."""
        if v is not None and v < 1:
            raise ValueError(f"timeout must be positive or None, got {v}")
        return v

    @field_validator("n_jobs")
    @classmethod
    def validate_n_jobs(cls, v: int) -> int:
        """Validate n_jobs is at least 1."""
        if v < 1:
            raise ValueError(f"n_jobs must be at least 1, got {v}")
        return v

    @field_validator("cv_folds")
    @classmethod
    def validate_cv_folds(cls, v: int) -> int:
        """Validate cv_folds is non-negative."""
        if v < 0:
            raise ValueError(f"cv_folds must be non-negative, got {v}")
        return v

    @field_validator("cv_metric_aggregation")
    @classmethod
    def validate_cv_metric_aggregation(cls, v: str) -> str:
        """Validate cv_metric_aggregation is a valid value."""
        valid_aggregations = ("mean", "median", "min", "max")
        if v not in valid_aggregations:
            raise ValueError(
                f"Invalid cv_metric_aggregation: '{v}'. Must be 'mean', 'median', 'min', or 'max'"
            )
        return v

    def to_dict(self) -> dict[str, Any]:
        """Backward compatible dict conversion."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "HPOConfig":
        """
        Create HPOConfig from dictionary.

        Pattern: Follows ModelConfig.from_yaml() (config_bridge.py:967-993)

        This method parses a configuration dictionary (typically from YAML)
        and creates a fully validated HPOConfig instance with all nested
        configurations properly instantiated.

        Args:
            config_dict: Dictionary containing HPO configuration

        Returns:
            HPOConfig: Validated configuration instance

        Raises:
            ValueError: If configuration is invalid

        Examples:
            >>> config_dict = {
            ...     'enabled': True,
            ...     'n_trials': 100,
            ...     'pruner': {'type': 'median', 'n_startup_trials': 10},
            ...     'sampler': {'type': 'tpe'},
            ...     'search_space': {
            ...         'model': {
            ...             'hidden_channels': {'type': 'int', 'low': 32, 'high': 256}
            ...         }
            ...     }
            ... }
            >>> config = HPOConfig.from_dict(config_dict)
        """
        pruner_dict = config_dict.get("pruner", {})
        sampler_dict = config_dict.get("sampler", {})
        study_dict = config_dict.get("study", {})

        if "type" in pruner_dict:
            pruner_dict = pruner_dict.copy()
            pruner_dict["type"] = PrunerType(pruner_dict["type"])
        if "type" in sampler_dict:
            sampler_dict = sampler_dict.copy()
            sampler_dict["type"] = SamplerType(sampler_dict["type"])
        if "direction" in study_dict:
            study_dict = study_dict.copy()
            study_dict["direction"] = OptimizationDirection(study_dict["direction"])

        search_space: dict[str, dict[str, SearchSpaceParamConfig]] = {}
        raw_search_space = config_dict.get("search_space", {})
        for category, params in raw_search_space.items():
            search_space[category] = {}
            for param_name, param_config in params.items():
                param_config_copy = param_config.copy()
                param_config_copy["type"] = ParamType(param_config_copy["type"])
                search_space[category][param_name] = SearchSpaceParamConfig(**param_config_copy)

        return cls(
            enabled=config_dict.get("enabled", False),
            backend=config_dict.get("backend", "optuna"),
            n_trials=config_dict.get("n_trials", 100),
            timeout=config_dict.get("timeout"),
            n_jobs=config_dict.get("n_jobs", 1),
            search_space=search_space,
            pruner=PrunerConfig(**pruner_dict) if pruner_dict else PrunerConfig(),
            sampler=SamplerConfig(**sampler_dict) if sampler_dict else SamplerConfig(),
            study=StudyConfig(**study_dict) if study_dict else StudyConfig(),
            cv_folds=config_dict.get("cv_folds", 0),
            cv_metric_aggregation=config_dict.get("cv_metric_aggregation", "mean"),
            task_type=config_dict.get("task_type"),
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "ParamType",
    "SearchSpaceParamConfig",
    "PrunerType",
    "PrunerConfig",
    "SamplerType",
    "SamplerConfig",
    "OptimizationDirection",
    "StudyConfig",
    "MultiObjectiveStudyConfig",
    "HPOConfig",
]
