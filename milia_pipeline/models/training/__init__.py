# models/training/__init__
"""
Training Module

Comprehensive training infrastructure for graph neural networks with:
- Main Trainer class for training/validation/test loops
- Extensive callback system for monitoring and control
- Loss function registry with custom implementations
- Optimizer and scheduler registries
- Data splitting strategies for various scenarios
- Task-specific data preparation (Phase 4)
- Complete metric tracking and checkpoint management
- Enhanced checkpoint format v2.0 with model recreation metadata (Phase 1 Post-Training)

This module provides all necessary components for training graph neural networks
in the milia Pipeline, supporting research experiments and production deployments.

Author: milia Team
Version: 1.1.0

Phase 1 Post-Training Enhancement (v1.1.0):
    The Trainer class now supports enhanced checkpoint format v2.0 which includes:
    - hyper_parameters: Complete model recreation metadata
    - data_info: Data compatibility information
    - version_info: Checkpoint format versioning

    New static methods for checkpoint inspection:
    - Trainer.get_checkpoint_info(filepath): Get checkpoint info without loading model
    - Trainer.is_v2_checkpoint(filepath): Check if checkpoint uses v2.0 format

Quick Start:
    >>> from milia_pipeline.models.training import (
    ...     Trainer,
    ...     get_loss,
    ...     get_optimizer,
    ...     get_scheduler,
    ...     EarlyStopping,
    ...     ModelCheckpoint,
    ...     random_split,
    ...     temporal_split,
    ...     scaffold_split,
    ...     k_fold_split,
    ...     TaskDataPreparer,  # Phase 4
    ...     prepare_data_for_task,  # Phase 4
    ... )
    >>>
    >>> # Setup training
    >>> loss_fn = get_loss("mse")
    >>> optimizer = get_optimizer("adam", model.parameters(), {"lr": 0.001})
    >>> scheduler = get_scheduler("cosine_annealing", optimizer, {"T_max": 100})
    >>>
    >>> # Split data
    >>> train_data, val_data, test_data = random_split(dataset)
    >>>
    >>> # Prepare for task (Phase 4)
    >>> train_data, val_data, test_data, num_classes = prepare_data_for_task(
    ...     train_data, val_data, test_data,
    ...     task_type='graph_classification'
    ... )
    >>>
    >>> # Create trainer with callbacks
    >>> trainer = Trainer(
    ...     model=model,
    ...     train_loader=train_loader,
    ...     val_loader=val_loader,
    ...     loss_fn=loss_fn,
    ...     optimizer=optimizer,
    ...     scheduler=scheduler,
    ...     callbacks=[
    ...         EarlyStopping(patience=10),
    ...         ModelCheckpoint(dirpath="checkpoints/", save_top_k=3)
    ...     ],
    ...     max_epochs=100
    ... )
    >>>
    >>> # Train
    >>> results = trainer.fit()
    >>>
    >>> # Save checkpoint with v2.0 format (auto-includes model_info)
    >>> trainer.save_checkpoint(Path("model.pt"))
    >>>
    >>> # Inspect checkpoint without loading model (Phase 1 Post-Training)
    >>> info = Trainer.get_checkpoint_info(Path("model.pt"))
    >>> print(f"Format: v{info['format_version']}, Model: {info['model_name']}")

Module Structure:
    - trainer: Main training engine (Trainer class) with v2.0 checkpoint support
    - callbacks: Training callbacks and monitoring
    - loss_functions: Loss function registry and custom losses
    - optimizers: Optimizer registry
    - schedulers: Learning rate scheduler registry
    - data_splitting: Data splitting utilities
    - data_preparation: Task-specific data preparation (Phase 4)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# VERSION INFORMATION
# =============================================================================

__version__ = "1.1.0"  # Phase 1 Post-Training: Enhanced checkpoint format v2.0
__author__ = "milia Team"


# =============================================================================
# CORE TRAINING COMPONENTS
# =============================================================================

# Main Trainer (Phase 1 Post-Training: Enhanced with v2.0 checkpoint support)
# Trainer class now includes:
#   - save_checkpoint(filepath, hyper_parameters, data_info, **extra_data)
#   - load_checkpoint(filepath) -> Dict with v2.0 format detection
#   - get_checkpoint_info(filepath) [STATIC] -> Dict with checkpoint metadata
#   - is_v2_checkpoint(filepath) [STATIC] -> bool
# Callback System
from .callbacks import (
    Callback,
    # Phase 3 Refactor: Add CallbackFactory for config-based callback creation
    CallbackFactory,
    EarlyStopping,
    GradientMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
    TensorBoardLogger,
)

# Phase 4 Refactor: Data Preparation for Tasks
from .data_preparation import (
    # Main Class
    TaskDataPreparer,
    list_supported_tasks,
    # Convenience Functions
    prepare_data_for_task,
)

# Data Splitting
from .data_splitting import (
    # Main Class
    DataSplitter,
    k_fold_split,
    # Convenience Functions
    random_split,
    scaffold_split,
    stratified_split,
    temporal_split,
)

# Loss Functions
from .loss_functions import (
    # Custom Losses
    FocalLoss,
    # Registry
    LossRegistry,
    RMSELoss,
    WeightedMSELoss,
    get_default_loss_for_task,
    # Convenience Functions
    get_loss,
    get_loss_for_task,
    is_loss_compatible_with_task,
    list_losses,
)

# Metrics (NEW - MetricsRegistry Implementation)
# Exports verified from metrics.py Lines 594-668
from .metrics import (
    # Registry
    MetricsRegistry,
    # Custom Metrics
    RMSEMetric,
    get_default_metrics_for_task,
    # Convenience Functions
    get_metric,
    get_metrics_for_task,
    is_metric_compatible_with_task,
    list_metrics,
)

# Optimizers
from .optimizers import (
    # Registry
    OptimizerRegistry,
    # Convenience Functions
    get_optimizer,
    list_optimizers,
)

# Schedulers
from .schedulers import (
    # Registry
    SchedulerRegistry,
    create_warmup_scheduler,
    # Convenience Functions
    get_scheduler,
    list_schedulers,
)
from .trainer import (
    Trainer,
)

# Visualization (NEW - TrainingVisualizer Implementation)
# Exports verified from visualization.py Lines 511-543
from .visualization import (
    # Main Class
    TrainingVisualizer,
    create_visualizer,
    # Convenience Functions
    plot_training_summary,
)

# =============================================================================
# PUBLIC API - ORGANIZED BY CATEGORY
# =============================================================================

# Core training class (Phase 1 Post-Training: v2.0 checkpoint support)
# Note: Static methods Trainer.get_checkpoint_info() and Trainer.is_v2_checkpoint()
#       are accessible via the Trainer class
__all_trainer__ = [
    "Trainer",
]

# Callback classes
__all_callbacks__ = [
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "TensorBoardLogger",
    "LearningRateMonitor",
    "ProgressBar",
    "GradientMonitor",
    "CallbackFactory",
]

# Loss function components
__all_losses__ = [
    "LossRegistry",
    "FocalLoss",
    "WeightedMSELoss",
    "RMSELoss",
    "get_loss",
    "get_loss_for_task",
    "list_losses",
    "get_default_loss_for_task",
    "is_loss_compatible_with_task",
]

# Metrics components (NEW - verified from metrics.py)
__all_metrics__ = [
    "MetricsRegistry",
    "RMSEMetric",
    "get_metric",
    "get_metrics_for_task",
    "list_metrics",
    "get_default_metrics_for_task",
    "is_metric_compatible_with_task",
]

# Visualization components (NEW - verified from visualization.py)
__all_visualization__ = [
    "TrainingVisualizer",
    "plot_training_summary",
    "create_visualizer",
]

# Optimizer components
__all_optimizers__ = [
    "OptimizerRegistry",
    "get_optimizer",
    "list_optimizers",
]

# Scheduler components
__all_schedulers__ = [
    "SchedulerRegistry",
    "get_scheduler",
    "list_schedulers",
    "create_warmup_scheduler",
]

# Data splitting components
__all_data_splitting__ = [
    "DataSplitter",
    "random_split",
    "stratified_split",
    "temporal_split",
    "scaffold_split",
    "k_fold_split",
]

# Data preparation components (Phase 4)
__all_data_preparation__ = [
    "TaskDataPreparer",
    "prepare_data_for_task",
    "list_supported_tasks",
]

# Complete public API
__all__ = (
    __all_trainer__
    + __all_callbacks__
    + __all_losses__
    + __all_metrics__  # NEW
    + __all_visualization__  # NEW
    + __all_optimizers__
    + __all_schedulers__
    + __all_data_splitting__
    + __all_data_preparation__
    + [
        # Module-level convenience functions
        "get_available_components",
        "create_training_pipeline",
        "quick_train",
        "print_available_components",
        "get_training_recipe",
    ]
)


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================


def get_available_components() -> dict[str, list[str]]:
    """
    Get all available training components.

    Returns:
        Dictionary mapping component types to available options

    Example:
        >>> from milia_pipeline.models.training import get_available_components
        >>> components = get_available_components()
        >>> print(f"Available losses: {components['losses']}")
        >>> print(f"Available metrics: {components['metrics']}")  # NEW
        >>> print(f"Available optimizers: {components['optimizers']}")
        >>> print(f"Available schedulers: {components['schedulers']}")
        >>> print(f"Available callbacks: {components['callbacks']}")
        >>> print(f"Supported task types: {components['task_types']}")
    """
    return {
        "losses": list_losses(),
        "metrics": list_metrics(),  # NEW - from metrics.py Line 636
        "optimizers": list_optimizers(),
        "schedulers": list_schedulers(),
        "callbacks": [
            "Callback",
            "EarlyStopping",
            "ModelCheckpoint",
            "TensorBoardLogger",
            "LearningRateMonitor",
            "ProgressBar",
            "GradientMonitor",
            "CallbackFactory",
        ],
        "data_splitting": [
            "random_split",
            "stratified_split",
            "temporal_split",
            "scaffold_split",
            "k_fold_split",
        ],
        # Phase 4: Add supported task types
        "task_types": list_supported_tasks(),
    }


def create_training_pipeline(
    model,
    dataset,
    loss_name: str = "mse",
    optimizer_name: str = "adam",
    optimizer_params: dict[str, Any] | None = None,
    scheduler_name: str | None = None,
    scheduler_params: dict[str, Any] | None = None,
    batch_size: int = 32,
    max_epochs: int = 100,
    split_strategy: str = "random",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    enable_early_stopping: bool = True,
    early_stopping_patience: int = 10,
    enable_checkpointing: bool = True,
    checkpoint_dir: str | None = None,
    enable_tensorboard: bool = False,
    tensorboard_log_dir: str | None = None,
    device: str | None = None,
) -> tuple["Trainer", dict[str, Any]]:
    """
    Create a complete training pipeline with sensible defaults.

    This is a high-level convenience function that sets up all components
    needed for training in a single call.

    Args:
        model: PyTorch model to train
        dataset: PyTorch Geometric dataset
        loss_name: Loss function name (default: "mse")
        optimizer_name: Optimizer name (default: "adam")
        optimizer_params: Optimizer parameters (default: {"lr": 0.001})
        scheduler_name: Scheduler name (default: None)
        scheduler_params: Scheduler parameters (default: None)
        batch_size: Batch size (default: 32)
        max_epochs: Maximum epochs (default: 100)
        split_strategy: Data split strategy (default: "random")
        train_ratio: Training set ratio (default: 0.8)
        val_ratio: Validation set ratio (default: 0.1)
        test_ratio: Test set ratio (default: 0.1)
        enable_early_stopping: Enable early stopping (default: True)
        early_stopping_patience: Early stopping patience (default: 10)
        enable_checkpointing: Enable model checkpointing (default: True)
        checkpoint_dir: Checkpoint directory (default: None)
        enable_tensorboard: Enable TensorBoard logging (default: False)
        tensorboard_log_dir: TensorBoard log directory (default: None)
        device: Device to use (default: auto-detect)

    Returns:
        Tuple of (trainer, info_dict) where info_dict contains component details

    Example:
        >>> from milia_pipeline.models.training import create_training_pipeline
        >>> trainer, info = create_training_pipeline(
        ...     model=my_model,
        ...     dataset=my_dataset,
        ...     loss_name="mse",
        ...     optimizer_name="adamw",
        ...     max_epochs=100
        ... )
        >>> results = trainer.fit()
    """
    from pathlib import Path

    import torch
    from torch_geometric.loader import DataLoader

    # Set defaults
    if optimizer_params is None:
        optimizer_params = {"lr": 0.001}

    info = {}

    # Setup loss
    loss_fn = get_loss(loss_name)
    info["loss"] = loss_name

    # Setup optimizer
    optimizer = get_optimizer(optimizer_name, model.parameters(), optimizer_params)
    info["optimizer"] = optimizer_name
    info["optimizer_params"] = optimizer_params

    # Setup scheduler
    scheduler = None
    if scheduler_name:
        scheduler = get_scheduler(scheduler_name, optimizer, scheduler_params or {})
        info["scheduler"] = scheduler_name
        info["scheduler_params"] = scheduler_params

    # Split data
    split_funcs = {
        "random": random_split,
        "stratified": stratified_split,
        "temporal": temporal_split,
        "scaffold": scaffold_split,
    }

    if split_strategy not in split_funcs:
        raise ValueError(
            f"Unknown split strategy: {split_strategy}. Available: {list(split_funcs.keys())}"
        )

    split_func = split_funcs[split_strategy]
    train_subset, val_subset, test_subset = split_func(dataset, train_ratio, val_ratio, test_ratio)

    info["split_strategy"] = split_strategy
    info["train_size"] = len(train_subset)
    info["val_size"] = len(val_subset)
    info["test_size"] = len(test_subset)

    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    # Setup callbacks
    callbacks = []

    if enable_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss", patience=early_stopping_patience, mode="min", verbose=True
            )
        )

    if enable_checkpointing:
        checkpoint_path = Path(checkpoint_dir or "./checkpoints")
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_path,
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
                verbose=True,
            )
        )

    if enable_tensorboard:
        tb_log_path = Path(tensorboard_log_dir or "./logs/tensorboard")
        callbacks.append(TensorBoardLogger(log_dir=tb_log_path))

    # Always add progress bar
    callbacks.append(ProgressBar())

    info["callbacks"] = callbacks

    # Determine device
    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device_obj,
        callbacks=callbacks,
        max_epochs=max_epochs,
    )

    logger.info(
        f"Training pipeline created: "
        f"loss={loss_name}, optimizer={optimizer_name}, "
        f"scheduler={scheduler_name}, epochs={max_epochs}"
    )

    return trainer, info


def quick_train(
    model,
    dataset,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str | None = None,
) -> dict[str, Any]:
    """
    Quick training with minimal configuration.

    Simplified interface for rapid experimentation. Uses sensible defaults
    for all training components.

    Args:
        model: PyTorch model to train
        dataset: PyTorch Geometric dataset
        epochs: Number of training epochs (default: 100)
        batch_size: Batch size (default: 32)
        learning_rate: Learning rate (default: 0.001)
        device: Device to use (default: auto-detect)

    Returns:
        Dictionary with training results

    Example:
        >>> from milia_pipeline.models.training import quick_train
        >>> results = quick_train(model, dataset, epochs=50, learning_rate=0.0001)
        >>> print(f"Final validation loss: {results['val_metrics']['val_loss'][-1]}")
    """
    trainer, info = create_training_pipeline(
        model=model,
        dataset=dataset,
        optimizer_params={"lr": learning_rate},
        batch_size=batch_size,
        max_epochs=epochs,
        enable_early_stopping=True,
        enable_checkpointing=False,
        enable_tensorboard=False,
        device=device,
    )

    results = trainer.fit()
    return results


# =============================================================================
# REGISTRY INFORMATION
# =============================================================================


def print_available_components():
    """
    Print all available training components to console.

    Useful for exploring available options during development.

    Example:
        >>> from milia_pipeline.models.training import print_available_components
        >>> print_available_components()
    """
    components = get_available_components()

    print("=" * 70)
    print("milia Pipeline - Available Training Components")
    print("=" * 70)

    print(f"\n📊 Loss Functions ({len(components['losses'])} available):")
    for i, name in enumerate(components["losses"], 1):
        print(f"  {i:2d}. {name}")

    # NEW: Metrics section
    print(f"\n📏 Metrics ({len(components['metrics'])} available):")
    for i, name in enumerate(components["metrics"], 1):
        print(f"  {i:2d}. {name}")

    print(f"\n🔧 Optimizers ({len(components['optimizers'])} available):")
    for i, name in enumerate(components["optimizers"], 1):
        print(f"  {i:2d}. {name}")

    print(f"\n📈 Schedulers ({len(components['schedulers'])} available):")
    for i, name in enumerate(components["schedulers"], 1):
        print(f"  {i:2d}. {name}")

    print(f"\n🔔 Callbacks ({len(components['callbacks'])} available):")
    for i, name in enumerate(components["callbacks"], 1):
        print(f"  {i:2d}. {name}")

    print(f"\n✂️  Data Splitting Strategies ({len(components['data_splitting'])} available):")
    for i, name in enumerate(components["data_splitting"], 1):
        print(f"  {i:2d}. {name}")

    # Phase 4: Add task types
    print(f"\n🎯 Supported Task Types ({len(components['task_types'])} available):")
    for i, name in enumerate(components["task_types"], 1):
        print(f"  {i:2d}. {name}")

    print("\n" + "=" * 70)
    print(f"Training Module v{__version__}")
    print("=" * 70)


# =============================================================================
# USAGE EXAMPLES AND RECIPES
# =============================================================================


def get_training_recipe(recipe_name: str) -> dict[str, Any]:
    """
    Get pre-configured training recipes for common scenarios.

    Provides battle-tested configurations for different training scenarios.

    Args:
        recipe_name: Name of recipe
            - "fast_prototyping": Quick experiments with minimal overhead
            - "production_training": Full monitoring and checkpointing
            - "research_experiment": Comprehensive logging and analysis
            - "fine_tuning": Configuration for fine-tuning pre-trained models

    Returns:
        Dictionary of configuration parameters

    Raises:
        ValueError: If recipe_name is not recognized

    Example:
        >>> from milia_pipeline.models.training import get_training_recipe
        >>> config = get_training_recipe("production_training")
        >>> trainer, info = create_training_pipeline(model, dataset, **config)
        >>> results = trainer.fit()
    """
    recipes = {
        "fast_prototyping": {
            "loss_name": "mse",
            "optimizer_name": "adam",
            "optimizer_params": {"lr": 0.001},
            "max_epochs": 50,
            "batch_size": 64,
            "enable_early_stopping": True,
            "early_stopping_patience": 5,
            "enable_checkpointing": False,
            "enable_tensorboard": False,
        },
        "production_training": {
            "loss_name": "mse",
            "optimizer_name": "adamw",
            "optimizer_params": {"lr": 0.001, "weight_decay": 0.01},
            "scheduler_name": "cosine_annealing",
            "scheduler_params": {"T_max": 100, "eta_min": 1e-6},
            "max_epochs": 100,
            "batch_size": 32,
            "enable_early_stopping": True,
            "early_stopping_patience": 15,
            "enable_checkpointing": True,
            "checkpoint_dir": "./checkpoints",
            "enable_tensorboard": True,
            "tensorboard_log_dir": "./logs/tensorboard",
        },
        "research_experiment": {
            "loss_name": "mse",
            "optimizer_name": "adamw",
            "optimizer_params": {"lr": 0.001, "weight_decay": 0.01},
            "scheduler_name": "reduce_on_plateau",
            "scheduler_params": {"mode": "min", "factor": 0.5, "patience": 10, "min_lr": 1e-7},
            "max_epochs": 200,
            "batch_size": 32,
            "split_strategy": "random",
            "enable_early_stopping": True,
            "early_stopping_patience": 20,
            "enable_checkpointing": True,
            "checkpoint_dir": "./experiments/checkpoints",
            "enable_tensorboard": True,
            "tensorboard_log_dir": "./experiments/logs",
        },
        "fine_tuning": {
            "loss_name": "mse",
            "optimizer_name": "adam",
            "optimizer_params": {"lr": 0.0001},  # Lower LR for fine-tuning
            "scheduler_name": "cosine_annealing",
            "scheduler_params": {"T_max": 50, "eta_min": 1e-7},
            "max_epochs": 50,
            "batch_size": 16,  # Smaller batch for stability
            "enable_early_stopping": True,
            "early_stopping_patience": 10,
            "enable_checkpointing": True,
            "enable_tensorboard": True,
        },
    }

    if recipe_name not in recipes:
        available = ", ".join(recipes.keys())
        raise ValueError(f"Unknown recipe: '{recipe_name}'. Available recipes: {available}")

    return recipes[recipe_name].copy()


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info(f"training module loaded - v{__version__} - {len(__all__)} public components")

# Log component counts
_components = get_available_components()
logger.info(
    f"Available components: "
    f"losses={len(_components['losses'])}, "
    f"metrics={len(_components['metrics'])}, "  # NEW
    f"optimizers={len(_components['optimizers'])}, "
    f"schedulers={len(_components['schedulers'])}, "
    f"callbacks={len(_components['callbacks'])}, "
    f"splitting_strategies={len(_components['data_splitting'])}, "
    f"task_types={len(_components['task_types'])}"
)
