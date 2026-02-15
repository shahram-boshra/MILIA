# models/utils/__init__
"""
Models Utilities Module

Provides utility functions and classes for the models module, including:
- Configuration management and bridge patterns
- PyTorch Geometric integration utilities
- Data validation and processing helpers
- HPO (Hyperparameter Optimization) configuration bridge (Phase 8)

This module consolidates utilities from:
- config_bridge: Configuration bridge for models module integration
- pyg_integration: PyTorch Geometric data handling and validation

Author: milia Team
Version: 1.1.0
"""

import logging
from typing import TYPE_CHECKING

# Configure module logger
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION BRIDGE IMPORTS
# =============================================================================

# Import all configuration-related components from config_bridge
from .config_bridge import (
    AccelerationConfig,
    # Configuration containers - Nested/Sub-configs
    # Callback sub-config
    CallbackConfig,
    CallbacksConfig,
    CloudDeploymentConfig,
    ComputationConfig,
    DataLoaderConfig,
    DataSplitConfig,
    DataSplitMethod,
    DDPConfig,
    DeepSpeedConfig,
    DeploymentConfig,
    DeploymentStrategy,
    # Device and Distributed sub-configs
    DeviceConfig,
    DeviceType,
    DistillationConfig,
    DistributedConfig,
    DistributedStrategy,
    # Monitoring sub-configs
    DriftDetectionConfig,
    # Deployment strategy sub-configs
    EdgeDeploymentConfig,
    EvaluationConfig,
    FederatedConfig,
    FSDPConfig,
    # HPO configuration classes (Phase 8)
    HPOConfigBridge,
    HPODirection,
    # Enums - HPO (Phase 8)
    HPOParamType,
    HPOPrunerConfigBridge,
    HPOPrunerType,
    HPOSamplerConfigBridge,
    HPOSamplerType,
    HPOSearchSpaceParamBridge,
    HPOStudyConfigBridge,
    LoggingConfig,
    LossConfig,
    LossFunction,
    # Memory and Computation sub-configs
    MemoryConfig,
    MixedPrecision,
    # Main configuration class
    ModelConfig,
    # Configuration containers - Primary
    ModelSelectionConfig,
    MonitoringConfig,
    OptimizationConfig,
    OptimizerConfig,
    OptimizerType,
    PluginsConfig,
    PruningConfig,
    # Deployment optimization sub-configs
    QuantizationConfig,
    RetrainingConfig,
    SchedulerConfig,
    SchedulerType,
    # Enums - Core
    TaskType,
    TrainingConfig,
    # Training sub-configs
    ValidationConfig,
    get_acceleration_config,
    get_deployment_config,
    get_hpo_config,
    get_model_selection,
    # Accessor functions
    get_models_config,
    get_plugins_config,
    get_training_config,
    is_hpo_enabled,
    is_models_enabled,
    validate_models_config,
)

# =============================================================================
# PYTORCH GEOMETRIC INTEGRATION IMPORTS
# =============================================================================
# Import all PyG integration utilities from pyg_integration
from .pyg_integration import (
    check_data_compatibility,
    clone_data,
    # Statistics functions
    compute_dataset_statistics,
    compute_graph_statistics,
    # Batch processing utilities
    create_dataloader,
    detach_data,
    get_batch_info,
    # Feature inference
    infer_num_features,
    infer_out_channels,
    print_dataset_summary,
    # Data utilities
    to_device,
    # Validation functions
    validate_pyg_data,
)

# =============================================================================
# MODULE METADATA
# =============================================================================

__version__ = "1.1.0"
__author__ = "milia Team"

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # =========================================================================
    # Configuration Bridge API
    # =========================================================================
    # Main configuration class
    "ModelConfig",
    # Configuration containers - Primary
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
    # Configuration containers - Nested/Sub-configs
    # Callback sub-config
    "CallbackConfig",
    # Training sub-configs
    "ValidationConfig",
    "LoggingConfig",
    # Device and Distributed sub-configs
    "DeviceConfig",
    "DDPConfig",
    "FSDPConfig",
    "DeepSpeedConfig",
    "DistributedConfig",
    # Memory and Computation sub-configs
    "MemoryConfig",
    "DataLoaderConfig",
    "ComputationConfig",
    # Deployment optimization sub-configs
    "QuantizationConfig",
    "PruningConfig",
    "DistillationConfig",
    "OptimizationConfig",
    # Deployment strategy sub-configs
    "EdgeDeploymentConfig",
    "CloudDeploymentConfig",
    "FederatedConfig",
    # Monitoring sub-configs
    "DriftDetectionConfig",
    "RetrainingConfig",
    "MonitoringConfig",
    # HPO configuration classes (Phase 8)
    "HPOConfigBridge",
    "HPOSearchSpaceParamBridge",
    "HPOPrunerConfigBridge",
    "HPOSamplerConfigBridge",
    "HPOStudyConfigBridge",
    # Configuration accessor functions
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
    # Configuration enums - Core
    "TaskType",
    "DataSplitMethod",
    "LossFunction",
    "OptimizerType",
    "SchedulerType",
    "DeviceType",
    "DistributedStrategy",
    "MixedPrecision",
    "DeploymentStrategy",
    # Configuration enums - HPO (Phase 8)
    "HPOParamType",
    "HPOPrunerType",
    "HPOSamplerType",
    "HPODirection",
    # =========================================================================
    # PyTorch Geometric Integration API
    # =========================================================================
    # Validation
    "validate_pyg_data",
    "check_data_compatibility",
    # Feature inference
    "infer_num_features",
    "infer_out_channels",
    # Statistics
    "compute_dataset_statistics",
    "print_dataset_summary",
    "compute_graph_statistics",
    # Batch processing
    "create_dataloader",
    "get_batch_info",
    # Data utilities
    "to_device",
    "detach_data",
    "clone_data",
]


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================


def _initialize_module():
    """
    Initialize the models utils module.

    Performs any necessary setup tasks such as:
    - Logging configuration validation
    - Import verification
    - Deprecation warnings if needed
    """
    logger.debug("Initializing models.utils module v%s", __version__)

    # Verify critical imports are available
    try:
        import torch

        logger.debug("PyTorch available: %s", torch.__version__)
    except ImportError:
        logger.warning("PyTorch not available - some features may be limited")

    try:
        import torch_geometric

        logger.debug("PyTorch Geometric available: %s", torch_geometric.__version__)
    except ImportError:
        logger.warning("PyTorch Geometric not available - PyG features disabled")

    # Check for Optuna availability (HPO Phase 8)
    try:
        import optuna

        logger.debug("Optuna available: %s", optuna.__version__)
    except ImportError:
        logger.debug("Optuna not available - HPO features require optuna installation")


# Initialize module on import
_initialize_module()


# =============================================================================
# USAGE EXAMPLES AND DOCUMENTATION
# =============================================================================

"""
Usage Examples
==============

1. Configuration Management:
   ---------------------------

   Basic configuration loading:
   >>> from milia_pipeline.models.utils import get_models_config
   >>> config = get_models_config()
   >>> print(f"Model: {config.selection.model_name}")
   >>> print(f"Task: {config.selection.task_type}")

   Check if models module is enabled:
   >>> from milia_pipeline.models.utils import is_models_enabled
   >>> if is_models_enabled():
   ...     print("Models module is enabled")

   Get specific configuration sections:
   >>> from milia_pipeline.models.utils import (
   ...     get_training_config,
   ...     get_acceleration_config
   ... )
   >>> training_cfg = get_training_config()
   >>> accel_cfg = get_acceleration_config()

   Validate configuration:
   >>> from milia_pipeline.models.utils import validate_models_config
   >>> try:
   ...     validate_models_config()
   ...     print("Configuration is valid")
   ... except ConfigurationError as e:
   ...     print(f"Configuration error: {e}")

   Using enums for type-safe configuration:
   >>> from milia_pipeline.models.utils import TaskType, OptimizerType
   >>> task = TaskType.GRAPH_REGRESSION
   >>> optimizer = OptimizerType.ADAM


2. HPO Configuration (Phase 8):
   -----------------------------

   Check if HPO is enabled:
   >>> from milia_pipeline.models.utils import is_hpo_enabled
   >>> if is_hpo_enabled():
   ...     print("HPO is enabled - will run hyperparameter optimization")

   Get HPO configuration:
   >>> from milia_pipeline.models.utils import get_hpo_config
   >>> hpo_config = get_hpo_config()
   >>> if hpo_config.enabled:
   ...     print(f"HPO backend: {hpo_config.backend}")
   ...     print(f"N trials: {hpo_config.n_trials}")
   ...     print(f"Sampler: {hpo_config.sampler.type}")
   ...     print(f"Pruner: {hpo_config.pruner.type}")

   Using HPO enums for type-safe configuration:
   >>> from milia_pipeline.models.utils import (
   ...     HPOParamType,
   ...     HPOPrunerType,
   ...     HPOSamplerType,
   ...     HPODirection
   ... )
   >>> param_type = HPOParamType.FLOAT
   >>> pruner = HPOPrunerType.MEDIAN
   >>> sampler = HPOSamplerType.TPE
   >>> direction = HPODirection.MINIMIZE

   Accessing HPO sub-configurations:
   >>> from milia_pipeline.models.utils import (
   ...     HPOConfigBridge,
   ...     HPOSearchSpaceParamBridge,
   ...     HPOPrunerConfigBridge,
   ...     HPOSamplerConfigBridge,
   ...     HPOStudyConfigBridge
   ... )
   >>> # These are bridge classes for config.yaml integration
   >>> # Full HPO implementation in milia_pipeline/models/hpo/


3. PyTorch Geometric Data Validation:
   ------------------------------------

   Validate PyG Data object:
   >>> from milia_pipeline.models.utils import validate_pyg_data
   >>> from torch_geometric.data import Data
   >>> import torch
   >>>
   >>> data = Data(
   ...     x=torch.randn(5, 10),
   ...     edge_index=torch.randint(0, 5, (2, 8))
   ... )
   >>> result = validate_pyg_data(data)
   >>> if result['valid']:
   ...     print("Data is valid!")
   ...     print(f"Number of nodes: {result['info']['num_nodes']}")
   ...     print(f"Number of edges: {result['info']['num_edges']}")

   Check data compatibility with model requirements:
   >>> from milia_pipeline.models.utils import check_data_compatibility
   >>> compatible, missing = check_data_compatibility(
   ...     data,
   ...     requires_edge_features=True,
   ...     requires_node_features=True
   ... )
   >>> if not compatible:
   ...     print(f"Missing requirements: {missing}")


4. Dataset Statistics and Inspection:
   ------------------------------------

   Compute dataset statistics:
   >>> from milia_pipeline.models.utils import (
   ...     compute_dataset_statistics,
   ...     print_dataset_summary
   ... )
   >>> stats = compute_dataset_statistics(train_dataset)
   >>> print(f"Number of graphs: {stats['num_graphs']}")
   >>> print(f"Average nodes: {stats['avg_num_nodes']:.2f}")
   >>>
   >>> # Or print a formatted summary
   >>> print_dataset_summary(train_dataset, "Training Set")

   Infer feature dimensions:
   >>> from milia_pipeline.models.utils import infer_num_features
   >>> num_node_features = infer_num_features(dataset)
   >>> print(f"Node features: {num_node_features}")


5. Batch Processing:
   -------------------

   Create DataLoader:
   >>> from milia_pipeline.models.utils import create_dataloader
   >>> train_loader = create_dataloader(
   ...     train_dataset,
   ...     batch_size=32,
   ...     shuffle=True,
   ...     num_workers=4
   ... )

   Get batch information:
   >>> from milia_pipeline.models.utils import get_batch_info
   >>> for batch in train_loader:
   ...     info = get_batch_info(batch)
   ...     print(f"Batch size: {info['batch_size']}")
   ...     print(f"Total nodes: {info['total_nodes']}")
   ...     break


6. Data Utilities:
   ----------------

   Move data to device:
   >>> from milia_pipeline.models.utils import to_device
   >>> import torch
   >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   >>> data = to_device(data, device)

   Detach data from computation graph:
   >>> from milia_pipeline.models.utils import detach_data
   >>> data = detach_data(data)

   Clone data:
   >>> from milia_pipeline.models.utils import clone_data
   >>> data_copy = clone_data(data)


7. Graph Statistics:
   ------------------

   Compute individual graph statistics:
   >>> from milia_pipeline.models.utils import compute_graph_statistics
   >>> stats = compute_graph_statistics(data)
   >>> print(f"Graph density: {stats['density']:.4f}")
   >>> print(f"Average degree: {stats['avg_degree']:.2f}")
   >>> print(f"Degree range: [{stats['min_degree']}, {stats['max_degree']}]")


8. Complete Training Setup Example with HPO:
   -------------------------------------------

   >>> from milia_pipeline.models.utils import (
   ...     get_models_config,
   ...     get_training_config,
   ...     get_hpo_config,
   ...     is_hpo_enabled,
   ...     validate_pyg_data,
   ...     create_dataloader,
   ...     to_device
   ... )
   >>>
   >>> # Load configuration
   >>> config = get_models_config()
   >>> training_cfg = get_training_config()
   >>>
   >>> # Check HPO status
   >>> if is_hpo_enabled():
   ...     hpo_cfg = get_hpo_config()
   ...     print(f"HPO enabled with {hpo_cfg.n_trials} trials")
   ...     print(f"Using {hpo_cfg.backend} backend")
   >>>
   >>> # Validate data
   >>> for data in dataset:
   ...     result = validate_pyg_data(data, strict=True)
   ...     if not result['valid']:
   ...         raise ValueError(f"Invalid data: {result['errors']}")
   >>>
   >>> # Create dataloaders
   >>> train_loader = create_dataloader(
   ...     train_dataset,
   ...     batch_size=training_cfg.batch_size,
   ...     shuffle=True
   ... )
   >>>
   >>> # Setup device
   >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   >>>
   >>> # Training loop
   >>> for batch in train_loader:
   ...     batch = to_device(batch, device)
   ...     # ... training code ...


9. Accessing Nested Configuration Classes:
   -----------------------------------------

   >>> from milia_pipeline.models.utils import (
   ...     # Callback configuration
   ...     CallbackConfig,
   ...     CallbacksConfig,
   ...
   ...     # Training sub-configs
   ...     ValidationConfig,
   ...     LoggingConfig,
   ...
   ...     # Distributed training configs
   ...     DDPConfig,
   ...     FSDPConfig,
   ...     DeepSpeedConfig,
   ...     DistributedConfig,
   ...
   ...     # Memory optimization
   ...     MemoryConfig,
   ...     DataLoaderConfig,
   ...     ComputationConfig,
   ...
   ...     # Deployment optimization
   ...     QuantizationConfig,
   ...     PruningConfig,
   ...     DistillationConfig,
   ...     OptimizationConfig,
   ...
   ...     # Deployment strategies
   ...     EdgeDeploymentConfig,
   ...     CloudDeploymentConfig,
   ...     FederatedConfig,
   ...
   ...     # Monitoring
   ...     DriftDetectionConfig,
   ...     RetrainingConfig,
   ...     MonitoringConfig,
   ... )
   >>>
   >>> # Example: Create custom distributed config
   >>> ddp_config = DDPConfig(
   ...     find_unused_parameters=True,
   ...     gradient_as_bucket_view=True
   ... )


Integration Notes
=================

Configuration Bridge Pattern:
- The config_bridge module follows the established pattern from config_accessors.py
- Provides structured access to model configuration from config.yaml
- Uses dataclasses for type-safe configuration containers
- Includes comprehensive validation for all configuration options

HPO Integration (Phase 8):
- HPOConfigBridge provides the MASTER SWITCH for HPO via `enabled` flag
- Supports Optuna as primary backend, Ray Tune as optional scale-out
- Search space configuration via HPOSearchSpaceParamBridge
- Pruner configuration via HPOPrunerConfigBridge (median, hyperband, percentile)
- Sampler configuration via HPOSamplerConfigBridge (TPE, random, CMA-ES, grid)
- Study configuration via HPOStudyConfigBridge (persistence, direction, metrics)
- Full implementation in milia_pipeline/models/hpo/

PyTorch Geometric Integration:
- All functions designed to work seamlessly with PyG Data and Dataset objects
- Provides validation, statistics, and batch processing utilities
- Compatible with PyG DataLoader and standard PyTorch workflows
- Supports both single graphs and batched graph data

Thread Safety:
- Configuration loading is thread-safe when using accessor functions
- PyG data operations are thread-safe for read operations
- Modifications to PyG Data objects should be externally synchronized

Performance Considerations:
- Dataset statistics computation may be expensive for large datasets
- Use caching where appropriate for repeated statistics calculations
- DataLoader num_workers should be tuned based on system resources
- Device transfers should be minimized in training loops
"""
