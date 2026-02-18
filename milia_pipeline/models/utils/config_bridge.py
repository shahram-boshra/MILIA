"""
Configuration Bridge for Models Module

Provides configuration loading and access for the models module,
integrating with the milia pipeline configuration system.

This module follows the established pattern from config_accessors.py
and provides structured access to model configuration from config.yaml.

Features:
- Load and parse models configuration section
- Structured configuration containers
- Validation of required fields
- Safe defaults for optional settings
- Integration with Phase 1-3 components
- HPO (Hyperparameter Optimization) configuration bridge (Phase 8)

Author: Milia Team
Version: 1.1.0
"""

import logging
import warnings
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# Import config loader
try:
    from milia_pipeline.config.config_loader import load_config

    CONFIG_LOADER_AVAILABLE = True
except ImportError:
    CONFIG_LOADER_AVAILABLE = False
    warnings.warn("config_loader not available - using fallback YAML loading", UserWarning, stacklevel=2)
    import yaml

    def load_config(config_path: Path | None = None) -> dict[str, Any]:
        """Fallback config loader."""
        if config_path is None:
            config_path = Path("config.yaml")
        with open(config_path) as f:
            return yaml.safe_load(f)


# Import exceptions with fallback
try:
    from milia_pipeline.exceptions import ConfigurationError, ModelError, ValidationError
except ImportError:

    class ConfigurationError(Exception):
        """Configuration error."""

        pass

    class ModelError(Exception):
        """Model error."""

        pass

    class ValidationError(Exception):
        """Validation error."""

        pass


logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION ENUMS
# =============================================================================


class TaskType(Enum):
    """Supported task types."""

    NODE_REGRESSION = "node_regression"
    NODE_CLASSIFICATION = "node_classification"
    GRAPH_REGRESSION = "graph_regression"
    GRAPH_CLASSIFICATION = "graph_classification"
    LINK_PREDICTION = "link_prediction"
    EDGE_REGRESSION = "edge_regression"


class DataSplitMethod(Enum):
    """Data splitting methods."""

    RANDOM = "random"
    STRATIFIED = "stratified"
    TEMPORAL = "temporal"
    SCAFFOLD = "scaffold"


class LossFunction(Enum):
    """Supported loss functions."""

    # Regression
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    SMOOTH_L1 = "smooth_l1"
    RMSE = "rmse"
    # Classification
    CROSS_ENTROPY = "cross_entropy"
    BCE = "bce"
    FOCAL = "focal"
    BCE_WITH_LOGITS = "bce_with_logits"


class OptimizerType(Enum):
    """Supported optimizers."""

    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"


class SchedulerType(Enum):
    """Supported learning rate schedulers."""

    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    COSINE_ANNEALING = "cosine_annealing"
    STEP_LR = "step_lr"
    EXPONENTIAL_LR = "exponential_lr"
    CYCLIC_LR = "cyclic_lr"
    ONE_CYCLE_LR = "one_cycle_lr"


class DeviceType(Enum):
    """Supported device types."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    TPU = "tpu"


class DistributedStrategy(Enum):
    """Distributed training strategies."""

    DDP = "ddp"
    FSDP = "fsdp"
    DEEPSPEED = "deepspeed"
    HOROVOD = "horovod"
    NONE = "none"


class MixedPrecision(Enum):
    """Mixed precision modes."""

    NO = "no"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"


class DeploymentStrategy(Enum):
    """Deployment strategies."""

    LOCAL = "local"
    CLOUD = "cloud"
    EDGE = "edge"
    FEDERATED = "federated"
    SERVERLESS = "serverless"


# =============================================================================
# HPO CONFIGURATION ENUMS (Phase 8)
# =============================================================================


class HPOParamType(Enum):
    """HPO parameter types for config bridge."""

    INT = "int"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    LOGUNIFORM = "loguniform"


class HPOPrunerType(Enum):
    """HPO pruner types for config bridge."""

    MEDIAN = "median"
    HYPERBAND = "hyperband"
    PERCENTILE = "percentile"
    NONE = "none"


class HPOSamplerType(Enum):
    """HPO sampler types for config bridge."""

    TPE = "tpe"
    RANDOM = "random"
    CMAES = "cmaes"
    GRID = "grid"


class HPODirection(Enum):
    """HPO optimization direction."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


# =============================================================================
# CONFIGURATION CONTAINERS
# =============================================================================


class ModelSelectionConfig(BaseModel):
    """Model selection configuration."""

    task_type: str
    model_name: str
    baseline_model: str | None = None

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, v: str) -> str:
        """Validate task_type is a valid TaskType enum value."""
        if not v:
            raise ValueError("task_type is required")
        try:
            TaskType(v)
        except ValueError:
            valid_tasks = [t.value for t in TaskType]
            raise ValueError(f"Invalid task_type '{v}'. Must be one of: {valid_tasks}") from None
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model_name is provided."""
        if not v:
            raise ValueError("model_name is required")
        return v

    def validate(self):
        """Backward compatible validate method (validation happens on construction)."""
        pass


class DataSplitConfig(BaseModel):
    """Data splitting configuration."""

    method: str = "random"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int | None = 42
    shuffle: bool = True
    stratify_by: str | None = None

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate method is a valid DataSplitMethod enum value."""
        try:
            DataSplitMethod(v)
        except ValueError:
            valid_methods = [m.value for m in DataSplitMethod]
            raise ValueError(f"Invalid split method '{v}'. Must be one of: {valid_methods}") from None
        return v

    @model_validator(mode="after")
    def validate_ratios(self) -> "DataSplitConfig":
        """Validate that split ratios sum to 1.0 and are in valid range."""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= total_ratio <= 1.01):  # Allow for floating point errors
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

        if not all(0 <= r <= 1 for r in [self.train_ratio, self.val_ratio, self.test_ratio]):
            raise ValueError("All split ratios must be between 0 and 1")
        return self

    def validate(self):
        """Backward compatible validate method (validation happens on construction)."""
        pass


class LossConfig(BaseModel):
    """Loss function configuration."""

    name: str = "mse"
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is a valid LossFunction enum value."""
        try:
            LossFunction(v)
        except ValueError:
            valid_losses = [loss.value for loss in LossFunction]
            raise ValueError(f"Invalid loss function '{v}'. Must be one of: {valid_losses}") from None
        return v

    def validate(self):
        """Backward compatible validate method (validation happens on construction)."""
        pass


class OptimizerConfig(BaseModel):
    """Optimizer configuration."""

    name: str = "adam"
    params: dict[str, Any] = Field(default_factory=lambda: {"lr": 0.001, "weight_decay": 0.0001})

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is a valid OptimizerType enum value."""
        try:
            OptimizerType(v)
        except ValueError:
            valid_optimizers = [o.value for o in OptimizerType]
            raise ValueError(f"Invalid optimizer '{v}'. Must be one of: {valid_optimizers}") from None
        return v

    @model_validator(mode="after")
    def check_learning_rate(self) -> "OptimizerConfig":
        """Check if learning rate is specified."""
        if "lr" not in self.params and "learning_rate" not in self.params:
            logger.warning("No learning rate specified in optimizer params, using default")
        return self

    def validate(self):
        """Backward compatible validate method (validation happens on construction)."""
        pass


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""

    enabled: bool = True
    name: str = "reduce_on_plateau"
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_scheduler_name(self) -> "SchedulerConfig":
        """Validate scheduler name if enabled."""
        if self.enabled:
            try:
                SchedulerType(self.name)
            except ValueError:
                valid_schedulers = [s.value for s in SchedulerType]
                raise ValueError(
                    f"Invalid scheduler '{self.name}'. Must be one of: {valid_schedulers}"
                ) from None
        return self

    def validate(self):
        """Backward compatible validate method (validation happens on construction)."""
        pass


class CallbackConfig(BaseModel):
    """Individual callback configuration."""

    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class CallbacksConfig(BaseModel):
    """Callbacks configuration."""

    early_stopping: CallbackConfig = Field(
        default_factory=lambda: CallbackConfig(
            enabled=True,
            params={"monitor": "val_loss", "patience": 20, "mode": "min", "min_delta": 0.0001},
        )
    )
    model_checkpoint: CallbackConfig = Field(
        default_factory=lambda: CallbackConfig(
            enabled=True,
            params={"monitor": "val_loss", "save_top_k": 3, "mode": "min", "save_last": True},
        )
    )
    tensorboard: CallbackConfig = Field(
        default_factory=lambda: CallbackConfig(enabled=True, params={"log_dir": None})
    )
    lr_monitor: CallbackConfig = Field(
        default_factory=lambda: CallbackConfig(enabled=True, params={"logging_interval": "epoch"})
    )
    progress_bar: CallbackConfig = Field(
        default_factory=lambda: CallbackConfig(enabled=True, params={"refresh_rate": 1})
    )


class ValidationConfig(BaseModel):
    """Validation configuration."""

    check_val_every_n_epoch: int = 1
    val_check_interval: int | None = None


class LoggingConfig(BaseModel):
    """Training logging configuration."""

    log_every_n_steps: int = 50
    log_metrics: bool = True
    log_gradients: bool = False
    log_weights: bool = False


class TrainingConfig(BaseModel):
    """Training configuration."""

    data_split: DataSplitConfig = Field(default_factory=DataSplitConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def validate(self):
        """Backward compatible validate method (validation happens on construction)."""
        # Nested models are validated automatically by Pydantic
        pass


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    metrics: list[str] = Field(default_factory=lambda: ["mse", "mae", "r2"])
    test_after_training: bool = True
    save_predictions: bool = True
    predictions_dir: str | None = None


class DeviceConfig(BaseModel):
    """Device configuration."""

    type: str = "auto"
    gpu_ids: list[int] = Field(default_factory=lambda: [0])
    allow_fallback: bool = True

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate type is a valid DeviceType enum value."""
        try:
            DeviceType(v)
        except ValueError:
            valid_devices = [d.value for d in DeviceType]
            raise ValueError(f"Invalid device type '{v}'. Must be one of: {valid_devices}") from None
        return v

    def validate(self):
        """Backward compatible validate method (validation happens on construction)."""
        pass


class DDPConfig(BaseModel):
    """DistributedDataParallel configuration."""

    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True


class FSDPConfig(BaseModel):
    """Fully Sharded Data Parallel configuration."""

    sharding_strategy: str = "full_shard"
    cpu_offload: bool = False
    backward_prefetch: bool = True


class DeepSpeedConfig(BaseModel):
    """DeepSpeed configuration."""

    enabled: bool = False
    config_path: str | None = None
    zero_stage: int = 2
    offload_optimizer: bool = False
    offload_param: bool = False


class DistributedConfig(BaseModel):
    """Distributed training configuration."""

    enabled: bool = False
    strategy: str = "ddp"
    ddp: DDPConfig = Field(default_factory=DDPConfig)
    fsdp: FSDPConfig = Field(default_factory=FSDPConfig)
    deepspeed: DeepSpeedConfig = Field(default_factory=DeepSpeedConfig)
    num_nodes: int = 1
    world_size: int = 1
    node_rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 12355

    @model_validator(mode="after")
    def validate_strategy(self) -> "DistributedConfig":
        """Validate strategy if distributed is enabled."""
        if self.enabled:
            try:
                DistributedStrategy(self.strategy)
            except ValueError:
                valid_strategies = [s.value for s in DistributedStrategy]
                raise ValueError(
                    f"Invalid distributed strategy '{self.strategy}'. "
                    f"Must be one of: {valid_strategies}"
                ) from None
        return self

    def validate(self):
        """Backward compatible validate method (validation happens on construction)."""
        pass


class MemoryConfig(BaseModel):
    """Memory optimization configuration."""

    mixed_precision: str = "no"
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1
    max_memory_per_gpu: float | None = None
    empty_cache_interval: int = 0

    @field_validator("mixed_precision")
    @classmethod
    def validate_mixed_precision(cls, v: str) -> str:
        """Validate mixed_precision is a valid MixedPrecision enum value."""
        try:
            MixedPrecision(v)
        except ValueError:
            valid_precision = [m.value for m in MixedPrecision]
            raise ValueError(f"Invalid mixed precision '{v}'. Must be one of: {valid_precision}") from None
        return v

    def validate(self):
        """Backward compatible validate method (validation happens on construction)."""
        pass


class DataLoaderConfig(BaseModel):
    """DataLoader optimization configuration."""

    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = False


class ComputationConfig(BaseModel):
    """Computation optimization configuration."""

    compile_model: bool = False
    compile_mode: str = "default"
    use_cudnn_benchmark: bool = True
    enable_tf32: bool = True
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)


class AccelerationConfig(BaseModel):
    """Hardware acceleration configuration."""

    enabled: bool = False
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    computation: ComputationConfig = Field(default_factory=ComputationConfig)

    def validate(self):
        """Backward compatible validate method (validation happens on construction)."""
        # Nested models are validated automatically by Pydantic
        pass


class QuantizationConfig(BaseModel):
    """Quantization configuration."""

    enabled: bool = False
    method: str = "dynamic"
    backend: str = "fbgemm"
    dtype: str = "qint8"


class PruningConfig(BaseModel):
    """Pruning configuration."""

    enabled: bool = False
    method: str = "magnitude"
    amount: float = 0.3
    iterative: bool = False
    iterations: int = 5


class DistillationConfig(BaseModel):
    """Knowledge distillation configuration."""

    enabled: bool = False
    teacher_checkpoint: str | None = None
    temperature: float = 3.0
    alpha: float = 0.5


class OptimizationConfig(BaseModel):
    """Model optimization configuration."""

    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    pruning: PruningConfig = Field(default_factory=PruningConfig)
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)


class EdgeDeploymentConfig(BaseModel):
    """Edge deployment configuration."""

    target_device: str = "jetson_nano"
    optimization_level: str = "balanced"


class CloudDeploymentConfig(BaseModel):
    """Cloud deployment configuration."""

    provider: str | None = None
    instance_type: str | None = None
    accelerator: str = "gpu"
    auto_scaling: bool = False
    min_instances: int = 1
    max_instances: int = 10


class FederatedConfig(BaseModel):
    """Federated learning configuration."""

    num_clients: int = 10
    rounds: int = 50
    aggregation: str = "fedavg"
    client_selection: str = "random"


class DriftDetectionConfig(BaseModel):
    """Drift detection configuration."""

    enabled: bool = True
    method: str = "statistical"
    threshold: float = 0.1


class RetrainingConfig(BaseModel):
    """Retraining configuration."""

    enabled: bool = False
    trigger: str = "manual"
    schedule: str | None = None
    performance_threshold: float = 0.05


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""

    enabled: bool = True
    metrics: list[str] = Field(
        default_factory=lambda: [
            "inference_latency",
            "throughput",
            "memory_usage",
            "prediction_accuracy",
        ]
    )
    drift_detection: DriftDetectionConfig = Field(default_factory=DriftDetectionConfig)
    retraining: RetrainingConfig = Field(default_factory=RetrainingConfig)


class DeploymentConfig(BaseModel):
    """Deployment configuration (Phase 3)."""

    enabled: bool = False
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    strategy: str = "cloud"
    edge: EdgeDeploymentConfig = Field(default_factory=EdgeDeploymentConfig)
    cloud: CloudDeploymentConfig = Field(default_factory=CloudDeploymentConfig)
    federated: FederatedConfig = Field(default_factory=FederatedConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @model_validator(mode="after")
    def validate_strategy(self) -> "DeploymentConfig":
        """Validate strategy if deployment is enabled."""
        if self.enabled:
            try:
                DeploymentStrategy(self.strategy)
            except ValueError:
                valid_strategies = [s.value for s in DeploymentStrategy]
                raise ValueError(
                    f"Invalid deployment strategy '{self.strategy}'. "
                    f"Must be one of: {valid_strategies}"
                ) from None
        return self

    def validate(self):
        """Backward compatible validate method (validation happens on construction)."""
        pass


class PluginsConfig(BaseModel):
    """Plugins configuration."""

    enabled: bool = True
    plugin_paths: list[str] = Field(default_factory=lambda: ["./plugins/models"])
    auto_discover: bool = True
    auto_validate: bool = True
    validation_level: str = "standard"


# =============================================================================
# HPO CONFIGURATION CLASSES (Phase 8)
# =============================================================================
# Note: Full implementation in milia_pipeline/models/hpo/hpo_config.py
# These are bridge classes for integration with config system


class HPOSearchSpaceParamBridge(BaseModel):
    """
    Bridge class for HPO search space parameter.

    Defines a single hyperparameter's search space configuration.

    Attributes:
        type: Parameter type (int, float, categorical, loguniform)
        low: Lower bound for numeric types
        high: Upper bound for numeric types
        step: Step size for int types (optional)
        choices: List of choices for categorical type
        log: Whether to use log scale (for float type)
    """

    type: HPOParamType
    low: float | None = None
    high: float | None = None
    step: int | None = None
    choices: list[Any] | None = None
    log: bool = False


class HPOPrunerConfigBridge(BaseModel):
    """
    Bridge class for HPO pruner configuration.

    Configures early trial termination strategy.

    Attributes:
        type: Pruner type (median, hyperband, percentile, none)
        n_startup_trials: Trials before pruning begins
        n_warmup_steps: Epochs before pruning within a trial
        interval_steps: Check pruning every N steps
        percentile: For percentile pruner (default: 25.0)
    """

    type: HPOPrunerType = HPOPrunerType.MEDIAN
    n_startup_trials: int = 5
    n_warmup_steps: int = 10
    interval_steps: int = 1
    percentile: float = 25.0


class HPOSamplerConfigBridge(BaseModel):
    """
    Bridge class for HPO sampler configuration.

    Configures hyperparameter suggestion strategy.

    Attributes:
        type: Sampler type (tpe, random, cmaes, grid)
        n_startup_trials: Random trials before Bayesian optimization
        seed: Random seed for reproducibility
        multivariate: Whether to use multivariate TPE
        constant_liar: For parallel optimization
    """

    type: HPOSamplerType = HPOSamplerType.TPE
    n_startup_trials: int = 10
    seed: int | None = None
    multivariate: bool = True
    constant_liar: bool = False


class HPOStudyConfigBridge(BaseModel):
    """
    Bridge class for HPO study configuration.

    Configures the optimization study settings.

    Attributes:
        direction: Optimization direction (minimize/maximize)
        metric: Metric name to optimize (must match Trainer output)
        study_name: Name for the study (for persistence)
        storage: Storage URL (None for in-memory, "sqlite:///file.db" for persistence)
        load_if_exists: Whether to resume existing study
    """

    direction: HPODirection = HPODirection.MINIMIZE
    metric: str = "val_loss"
    study_name: str = "milia_hpo"
    storage: str | None = None
    load_if_exists: bool = True


class HPOConfigBridge(BaseModel):
    """
    Bridge class for HPO configuration.

    Integrates HPO settings with the config bridge system.
    Full implementation in milia_pipeline/models/hpo/hpo_config.py

    This is the MASTER SWITCH configuration that enables/disables HPO
    and configures all aspects of hyperparameter optimization.

    Attributes:
        enabled: MASTER SWITCH - enables HPO when True
        backend: HPO backend ("optuna" or "ray_tune")
        n_trials: Number of trials to run
        timeout: Maximum time in seconds (None for no limit)
        n_jobs: Number of parallel jobs (1 for sequential)
        search_space: Hyperparameter search space configuration
        pruner: Pruner configuration
        sampler: Sampler configuration
        study: Study configuration
        cv_folds: Number of cross-validation folds (0 for no CV)
        cv_metric_aggregation: How to aggregate CV metrics ("mean", "median", "min", "max")
    """

    enabled: bool = False
    backend: str = "optuna"
    n_trials: int = 100
    timeout: int | None = None
    n_jobs: int = 1
    search_space: dict[str, dict[str, HPOSearchSpaceParamBridge]] = Field(default_factory=dict)
    pruner: HPOPrunerConfigBridge = Field(default_factory=HPOPrunerConfigBridge)
    sampler: HPOSamplerConfigBridge = Field(default_factory=HPOSamplerConfigBridge)
    study: HPOStudyConfigBridge = Field(default_factory=HPOStudyConfigBridge)
    cv_folds: int = 0
    cv_metric_aggregation: str = "mean"

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        """Validate backend is a supported HPO backend."""
        if v not in ("optuna", "ray_tune"):
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
        valid_values = ("mean", "median", "min", "max")
        if v not in valid_values:
            raise ValueError(
                f"Invalid cv_metric_aggregation: '{v}'. Must be one of: {valid_values}"
            )
        return v

    def validate(self):
        """Backward compatible validate method (validation happens on construction)."""
        pass


# =============================================================================
# MAIN MODEL CONFIGURATION
# =============================================================================


class ModelConfig(BaseModel):
    """
    Complete models module configuration.

    This is the main configuration container that holds all model-related
    settings across all phases (Core, Acceleration, Deployment, Plugins, HPO).

    Attributes:
        enabled: Whether models module is enabled
        selection: Model selection configuration
        hyperparameters: Model hyperparameters
        training: Training configuration
        evaluation: Evaluation configuration
        acceleration: Hardware acceleration configuration (Phase 2)
        deployment: Deployment configuration (Phase 3)
        plugins: Plugin system configuration
        hpo: HPO configuration (Phase 8)
        raw: Raw configuration dictionary
    """

    enabled: bool = False
    selection: ModelSelectionConfig = Field(
        default_factory=lambda: ModelSelectionConfig(task_type="graph_regression", model_name="GCN")
    )
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    acceleration: AccelerationConfig = Field(default_factory=AccelerationConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    hpo: HPOConfigBridge = Field(default_factory=HPOConfigBridge)
    raw: dict[str, Any] = Field(default_factory=dict)

    def validate(self):
        """
        Backward compatible validate method (validation happens on construction).

        Note: Pydantic validates all nested models automatically on construction.
        This method is kept for backward compatibility with existing code that
        calls config.validate() explicitly.
        """
        # All validation happens automatically on construction
        pass

    def is_phase_enabled(self, phase: str) -> bool:
        """
        Check if a specific phase is enabled.

        Args:
            phase: Phase name ('core', 'acceleration', 'deployment', 'plugins', 'hpo')

        Returns:
            True if phase is enabled
        """
        phase_map = {
            "core": self.enabled,
            "acceleration": self.enabled and self.acceleration.enabled,
            "deployment": self.enabled and self.deployment.enabled,
            "plugins": self.enabled and self.plugins.enabled,
            "hpo": self.enabled and self.hpo.enabled,
        }
        return phase_map.get(phase.lower(), False)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ModelConfig":
        """
        Create ModelConfig from dictionary.

        Args:
            config_dict: Configuration dictionary from YAML

        Returns:
            ModelConfig instance
        """

        # Helper function to safely get nested config
        def get_config(key: str, default_factory):
            return config_dict.get(
                key, default_factory() if callable(default_factory) else default_factory
            )

        # Parse selection
        selection_dict = get_config("selection", {})
        selection = ModelSelectionConfig(
            task_type=selection_dict.get("task_type", "graph_regression"),
            model_name=selection_dict.get("model_name", "GCN"),
            baseline_model=selection_dict.get("baseline_model"),
        )

        # Parse hyperparameters
        hyperparameters = get_config("hyperparameters", {})

        # Parse training config
        training_dict = get_config("training", {})

        # Data split
        split_dict = training_dict.get("data_split", {})
        data_split = DataSplitConfig(
            method=split_dict.get("method", "random"),
            train_ratio=split_dict.get("train_ratio", 0.8),
            val_ratio=split_dict.get("val_ratio", 0.1),
            test_ratio=split_dict.get("test_ratio", 0.1),
            random_seed=split_dict.get("random_seed", 42),
            shuffle=split_dict.get("shuffle", True),
            stratify_by=split_dict.get("stratify_by"),
        )

        # Loss config
        loss_dict = training_dict.get("loss", {})
        loss = LossConfig(name=loss_dict.get("name", "mse"), params=loss_dict.get("params", {}))

        # Optimizer config
        opt_dict = training_dict.get("optimizer", {})
        optimizer = OptimizerConfig(
            name=opt_dict.get("name", "adam"),
            params=opt_dict.get("params", {"lr": 0.001, "weight_decay": 0.0001}),
        )

        # Scheduler config
        sched_dict = training_dict.get("scheduler", {})
        scheduler = SchedulerConfig(
            enabled=sched_dict.get("enabled", True),
            name=sched_dict.get("name", "reduce_on_plateau"),
            params=sched_dict.get("params", {}),
        )

        # Callbacks config
        callbacks_dict = training_dict.get("callbacks", {})
        callbacks = CallbacksConfig(
            early_stopping=CallbackConfig(**callbacks_dict.get("early_stopping", {}))
            if "early_stopping" in callbacks_dict
            else CallbackConfig(),
            model_checkpoint=CallbackConfig(**callbacks_dict.get("model_checkpoint", {}))
            if "model_checkpoint" in callbacks_dict
            else CallbackConfig(),
            tensorboard=CallbackConfig(**callbacks_dict.get("tensorboard", {}))
            if "tensorboard" in callbacks_dict
            else CallbackConfig(),
            lr_monitor=CallbackConfig(**callbacks_dict.get("lr_monitor", {}))
            if "lr_monitor" in callbacks_dict
            else CallbackConfig(),
            progress_bar=CallbackConfig(**callbacks_dict.get("progress_bar", {}))
            if "progress_bar" in callbacks_dict
            else CallbackConfig(),
        )

        # Validation config
        val_dict = training_dict.get("validation", {})
        validation = ValidationConfig(
            check_val_every_n_epoch=val_dict.get("check_val_every_n_epoch", 1),
            val_check_interval=val_dict.get("val_check_interval"),
        )

        # Logging config
        log_dict = training_dict.get("logging", {})
        logging_config = LoggingConfig(
            log_every_n_steps=log_dict.get("log_every_n_steps", 50),
            log_metrics=log_dict.get("log_metrics", True),
            log_gradients=log_dict.get("log_gradients", False),
            log_weights=log_dict.get("log_weights", False),
        )

        training = TrainingConfig(
            data_split=data_split,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=callbacks,
            validation=validation,
            logging=logging_config,
        )

        # Parse evaluation config
        eval_dict = get_config("evaluation", {})
        evaluation = EvaluationConfig(
            metrics=eval_dict.get("metrics", ["mse", "mae", "r2"]),
            test_after_training=eval_dict.get("test_after_training", True),
            save_predictions=eval_dict.get("save_predictions", True),
            predictions_dir=eval_dict.get("predictions_dir"),
        )

        # Parse acceleration config (Phase 2)
        accel_dict = get_config("acceleration", {})
        acceleration = cls._parse_acceleration_config(accel_dict)

        # Parse deployment config (Phase 3)
        deploy_dict = get_config("deployment", {})
        deployment = cls._parse_deployment_config(deploy_dict)

        # Parse plugins config
        plugins_dict = get_config("plugins", {})
        plugins = PluginsConfig(
            enabled=plugins_dict.get("enabled", True),
            plugin_paths=plugins_dict.get("plugin_paths", ["./plugins/models"]),
            auto_discover=plugins_dict.get("auto_discover", True),
            auto_validate=plugins_dict.get("auto_validate", True),
            validation_level=plugins_dict.get("validation_level", "standard"),
        )

        # Parse HPO config (Phase 8)
        hpo_dict = get_config("hpo", {})
        hpo = cls._parse_hpo_config(hpo_dict)

        return cls(
            enabled=config_dict.get("enabled", False),
            selection=selection,
            hyperparameters=hyperparameters,
            training=training,
            evaluation=evaluation,
            acceleration=acceleration,
            deployment=deployment,
            plugins=plugins,
            hpo=hpo,
            raw=config_dict,
        )

    @staticmethod
    def _parse_acceleration_config(accel_dict: dict[str, Any]) -> AccelerationConfig:
        """Parse acceleration configuration."""
        # Device config
        device_dict = accel_dict.get("device", {})
        device = DeviceConfig(
            type=device_dict.get("type", "auto"),
            gpu_ids=device_dict.get("gpu_ids", [0]),
            allow_fallback=device_dict.get("allow_fallback", True),
        )

        # Distributed config
        dist_dict = accel_dict.get("distributed", {})
        distributed = DistributedConfig(
            enabled=dist_dict.get("enabled", False),
            strategy=dist_dict.get("strategy", "ddp"),
            ddp=DDPConfig(**dist_dict.get("ddp", {})),
            fsdp=FSDPConfig(**dist_dict.get("fsdp", {})),
            deepspeed=DeepSpeedConfig(**dist_dict.get("deepspeed", {})),
            num_nodes=dist_dict.get("num_nodes", 1),
            world_size=dist_dict.get("world_size", 1),
            node_rank=dist_dict.get("node_rank", 0),
            master_addr=dist_dict.get("master_addr", "localhost"),
            master_port=dist_dict.get("master_port", 12355),
        )

        # Memory config
        mem_dict = accel_dict.get("memory", {})
        memory = MemoryConfig(
            mixed_precision=mem_dict.get("mixed_precision", "no"),
            gradient_checkpointing=mem_dict.get("gradient_checkpointing", False),
            gradient_accumulation_steps=mem_dict.get("gradient_accumulation_steps", 1),
            max_memory_per_gpu=mem_dict.get("max_memory_per_gpu"),
            empty_cache_interval=mem_dict.get("empty_cache_interval", 0),
        )

        # Computation config
        comp_dict = accel_dict.get("computation", {})
        dataloader_dict = comp_dict.get("dataloader", {})
        computation = ComputationConfig(
            compile_model=comp_dict.get("compile_model", False),
            compile_mode=comp_dict.get("compile_mode", "default"),
            use_cudnn_benchmark=comp_dict.get("use_cudnn_benchmark", True),
            enable_tf32=comp_dict.get("enable_tf32", True),
            dataloader=DataLoaderConfig(**dataloader_dict),
        )

        return AccelerationConfig(
            enabled=accel_dict.get("enabled", False),
            device=device,
            distributed=distributed,
            memory=memory,
            computation=computation,
        )

    @staticmethod
    def _parse_deployment_config(deploy_dict: dict[str, Any]) -> DeploymentConfig:
        """Parse deployment configuration."""
        # Optimization config
        opt_dict = deploy_dict.get("optimization", {})
        optimization = OptimizationConfig(
            quantization=QuantizationConfig(**opt_dict.get("quantization", {})),
            pruning=PruningConfig(**opt_dict.get("pruning", {})),
            distillation=DistillationConfig(**opt_dict.get("distillation", {})),
        )

        # Edge config
        edge_dict = deploy_dict.get("edge", {})
        edge = EdgeDeploymentConfig(
            target_device=edge_dict.get("target_device", "jetson_nano"),
            optimization_level=edge_dict.get("optimization_level", "balanced"),
        )

        # Cloud config
        cloud_dict = deploy_dict.get("cloud", {})
        cloud = CloudDeploymentConfig(
            provider=cloud_dict.get("provider"),
            instance_type=cloud_dict.get("instance_type"),
            accelerator=cloud_dict.get("accelerator", "gpu"),
            auto_scaling=cloud_dict.get("auto_scaling", False),
            min_instances=cloud_dict.get("min_instances", 1),
            max_instances=cloud_dict.get("max_instances", 10),
        )

        # Federated config
        fed_dict = deploy_dict.get("federated", {})
        federated = FederatedConfig(
            num_clients=fed_dict.get("num_clients", 10),
            rounds=fed_dict.get("rounds", 50),
            aggregation=fed_dict.get("aggregation", "fedavg"),
            client_selection=fed_dict.get("client_selection", "random"),
        )

        # Monitoring config
        mon_dict = deploy_dict.get("monitoring", {})
        drift_dict = mon_dict.get("drift_detection", {})
        retrain_dict = mon_dict.get("retraining", {})
        monitoring = MonitoringConfig(
            enabled=mon_dict.get("enabled", True),
            metrics=mon_dict.get(
                "metrics",
                ["inference_latency", "throughput", "memory_usage", "prediction_accuracy"],
            ),
            drift_detection=DriftDetectionConfig(**drift_dict),
            retraining=RetrainingConfig(**retrain_dict),
        )

        return DeploymentConfig(
            enabled=deploy_dict.get("enabled", False),
            optimization=optimization,
            strategy=deploy_dict.get("strategy", "cloud"),
            edge=edge,
            cloud=cloud,
            federated=federated,
            monitoring=monitoring,
        )

    @staticmethod
    def _parse_hpo_config(hpo_dict: dict[str, Any]) -> HPOConfigBridge:
        """
        Parse HPO configuration.

        Args:
            hpo_dict: HPO configuration dictionary from YAML

        Returns:
            HPOConfigBridge instance
        """
        if not hpo_dict:
            return HPOConfigBridge()

        # Parse pruner config
        pruner_dict = hpo_dict.get("pruner", {})
        pruner_type_str = pruner_dict.get("type", "median")
        try:
            pruner_type = HPOPrunerType(pruner_type_str)
        except ValueError:
            pruner_type = HPOPrunerType.MEDIAN

        pruner = HPOPrunerConfigBridge(
            type=pruner_type,
            n_startup_trials=pruner_dict.get("n_startup_trials", 5),
            n_warmup_steps=pruner_dict.get("n_warmup_steps", 10),
            interval_steps=pruner_dict.get("interval_steps", 1),
            percentile=pruner_dict.get("percentile", 25.0),
        )

        # Parse sampler config
        sampler_dict = hpo_dict.get("sampler", {})
        sampler_type_str = sampler_dict.get("type", "tpe")
        try:
            sampler_type = HPOSamplerType(sampler_type_str)
        except ValueError:
            sampler_type = HPOSamplerType.TPE

        sampler = HPOSamplerConfigBridge(
            type=sampler_type,
            n_startup_trials=sampler_dict.get("n_startup_trials", 10),
            seed=sampler_dict.get("seed"),
            multivariate=sampler_dict.get("multivariate", True),
            constant_liar=sampler_dict.get("constant_liar", False),
        )

        # Parse study config
        study_dict = hpo_dict.get("study", {})
        direction_str = study_dict.get("direction", "minimize")
        try:
            direction = HPODirection(direction_str)
        except ValueError:
            direction = HPODirection.MINIMIZE

        study = HPOStudyConfigBridge(
            direction=direction,
            metric=study_dict.get("metric", "val_loss"),
            study_name=study_dict.get("study_name", "milia_hpo"),
            storage=study_dict.get("storage"),
            load_if_exists=study_dict.get("load_if_exists", True),
        )

        # Parse search space
        search_space: dict[str, dict[str, HPOSearchSpaceParamBridge]] = {}
        raw_search_space = hpo_dict.get("search_space", {})

        for category, params in raw_search_space.items():
            search_space[category] = {}
            for param_name, param_config in params.items():
                param_type_str = param_config.get("type", "float")
                try:
                    param_type = HPOParamType(param_type_str)
                except ValueError:
                    param_type = HPOParamType.FLOAT

                search_space[category][param_name] = HPOSearchSpaceParamBridge(
                    type=param_type,
                    low=param_config.get("low"),
                    high=param_config.get("high"),
                    step=param_config.get("step"),
                    choices=param_config.get("choices"),
                    log=param_config.get("log", False),
                )

        return HPOConfigBridge(
            enabled=hpo_dict.get("enabled", False),
            backend=hpo_dict.get("backend", "optuna"),
            n_trials=hpo_dict.get("n_trials", 100),
            timeout=hpo_dict.get("timeout"),
            n_jobs=hpo_dict.get("n_jobs", 1),
            search_space=search_space,
            pruner=pruner,
            sampler=sampler,
            study=study,
            cv_folds=hpo_dict.get("cv_folds", 0),
            cv_metric_aggregation=hpo_dict.get("cv_metric_aggregation", "mean"),
        )

    @classmethod
    def from_yaml(cls, config_path: Path | None = None) -> "ModelConfig":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config.yaml (default: uses config loader)

        Returns:
            ModelConfig instance

        Example:
            >>> config = ModelConfig.from_yaml()
            >>> print(config.selection.model_name)
            'GCN'
        """
        try:
            full_config = load_config(config_path)
        except Exception as e:
            raise ConfigurationError(f"Failed to load config: {e}") from e

        models_config = full_config.get("models", {})

        if not models_config:
            logger.warning("No 'models' section found in config.yaml - using defaults")
            models_config = {}

        return cls.from_dict(models_config)


# =============================================================================
# CONFIGURATION ACCESSOR FUNCTIONS
# =============================================================================


def get_models_config(config_path: Path | None = None) -> ModelConfig:
    """
    Get models module configuration.

    Args:
        config_path: Optional path to config.yaml

    Returns:
        ModelConfig instance

    Example:
        >>> config = get_models_config()
        >>> print(f"Model: {config.selection.model_name}")
        >>> print(f"Task: {config.selection.task_type}")
    """
    return ModelConfig.from_yaml(config_path)


def is_models_enabled(config_path: Path | None = None) -> bool:
    """
    Check if models module is enabled.

    Args:
        config_path: Optional path to config.yaml

    Returns:
        True if models module is enabled
    """
    config = get_models_config(config_path)
    return config.enabled


def get_model_selection(config_path: Path | None = None) -> ModelSelectionConfig:
    """
    Get model selection configuration.

    Args:
        config_path: Optional path to config.yaml

    Returns:
        ModelSelectionConfig instance
    """
    config = get_models_config(config_path)
    return config.selection


def get_training_config(config_path: Path | None = None) -> TrainingConfig:
    """
    Get training configuration.

    Args:
        config_path: Optional path to config.yaml

    Returns:
        TrainingConfig instance
    """
    config = get_models_config(config_path)
    return config.training


def get_acceleration_config(config_path: Path | None = None) -> AccelerationConfig:
    """
    Get acceleration configuration.

    Args:
        config_path: Optional path to config.yaml

    Returns:
        AccelerationConfig instance
    """
    config = get_models_config(config_path)
    return config.acceleration


def get_deployment_config(config_path: Path | None = None) -> DeploymentConfig:
    """
    Get deployment configuration.

    Args:
        config_path: Optional path to config.yaml

    Returns:
        DeploymentConfig instance
    """
    config = get_models_config(config_path)
    return config.deployment


def get_plugins_config(config_path: Path | None = None) -> PluginsConfig:
    """
    Get plugins configuration.

    Args:
        config_path: Optional path to config.yaml

    Returns:
        PluginsConfig instance
    """
    config = get_models_config(config_path)
    return config.plugins


def get_hpo_config(config_path: Path | None = None) -> HPOConfigBridge:
    """
    Get HPO configuration.

    Args:
        config_path: Optional path to config.yaml

    Returns:
        HPOConfigBridge instance

    Example:
        >>> hpo_config = get_hpo_config()
        >>> if hpo_config.enabled:
        ...     print(f"HPO backend: {hpo_config.backend}")
        ...     print(f"N trials: {hpo_config.n_trials}")
    """
    config = get_models_config(config_path)
    return config.hpo


def is_hpo_enabled(config_path: Path | None = None) -> bool:
    """
    Check if HPO is enabled.

    Args:
        config_path: Optional path to config.yaml

    Returns:
        True if HPO is enabled

    Example:
        >>> if is_hpo_enabled():
        ...     print("HPO is enabled - will run hyperparameter optimization")
    """
    config = get_models_config(config_path)
    return config.hpo.enabled


def validate_models_config(config_path: Path | None = None) -> bool:
    """
    Validate models configuration.

    Args:
        config_path: Optional path to config.yaml

    Returns:
        True if valid

    Raises:
        ConfigurationError: If validation fails
    """
    config = get_models_config(config_path)
    config.validate()
    return True


# =============================================================================
# MODULE METADATA
# =============================================================================

__all__ = [
    # Main config class
    "ModelConfig",
    # Config containers
    "ModelSelectionConfig",
    "DataSplitConfig",
    "LossConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "CallbacksConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "AccelerationConfig",
    "DeploymentConfig",
    "PluginsConfig",
    # HPO config classes (Phase 8)
    "HPOConfigBridge",
    "HPOSearchSpaceParamBridge",
    "HPOPrunerConfigBridge",
    "HPOSamplerConfigBridge",
    "HPOStudyConfigBridge",
    # Accessor functions
    "get_models_config",
    "is_models_enabled",
    "get_model_selection",
    "get_training_config",
    "get_acceleration_config",
    "get_deployment_config",
    "get_plugins_config",
    "get_hpo_config",
    "is_hpo_enabled",
    "validate_models_config",
    # Enums
    "TaskType",
    "DataSplitMethod",
    "LossFunction",
    "OptimizerType",
    "SchedulerType",
    "DeviceType",
    "DistributedStrategy",
    "MixedPrecision",
    "DeploymentStrategy",
    # HPO Enums (Phase 8)
    "HPOParamType",
    "HPOPrunerType",
    "HPOSamplerType",
    "HPODirection",
]
