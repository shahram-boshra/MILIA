# models/deployment/__init__
"""
Model Deployment Module

Comprehensive deployment infrastructure for production environments.
Provides unified interface for model optimization, deployment strategies,
and production monitoring.

This module integrates:
- Deployment strategies (cloud, edge, container, local)
- Model optimization (quantization, pruning, distillation)
- Production monitoring (performance, drift detection, health checks)

Features:
- Multi-platform deployment (AWS, GCP, Azure, Edge, Container, Local)
- Model optimization for inference (quantization, pruning, ONNX export)
- Real-time monitoring and alerting
- Health checks and drift detection
- A/B testing and canary deployment support
- Thread-safe operations
- Comprehensive error handling

Usage Examples:
    >>> # Quick local deployment
    >>> from milia_pipeline.models.deployment import deploy_model_locally
    >>> manager = deploy_model_locally(model, save_path="./models")
    >>> result = manager.predict(input_data)

    >>> # Cloud deployment with optimization
    >>> from milia_pipeline.models.deployment import (
    ...     DeploymentManager,
    ...     DeploymentConfig,
    ...     optimize_model_for_deployment
    ... )
    >>>
    >>> # Optimize model
    >>> optimized_model = optimize_model_for_deployment(
    ...     model,
    ...     quantize=True,
    ...     prune_amount=0.3
    ... )
    >>>
    >>> # Deploy to AWS
    >>> config = DeploymentConfig(target="aws", instance_type="ml.m5.large")
    >>> manager = DeploymentManager(config)
    >>> model_path = manager.prepare_model(optimized_model, "./prepared")
    >>> deployment_info = manager.deploy(model_path)

    >>> # Production monitoring
    >>> from milia_pipeline.models.deployment import (
    ...     create_production_monitor,
    ...     MonitoringConfig
    ... )
    >>>
    >>> monitor = create_production_monitor(model_name="my_model")
    >>>
    >>> # Log predictions
    >>> start = time.time()
    >>> output = model(input_data)
    >>> latency = time.time() - start
    >>>
    >>> monitor.log_prediction(
    ...     input_data=input_data,
    ...     output=output,
    ...     latency=latency,
    ...     ground_truth=labels
    ... )
    >>>
    >>> # Get monitoring summary
    >>> metrics = monitor.get_metrics_summary()
    >>> health = monitor.health_check()

Author: milia Team
Version: 1.0.0
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# =============================================================================
# EXCEPTION HANDLING WITH FALLBACK
# =============================================================================

try:
    from milia_pipeline.exceptions import (
        AlertError,
        ConfigurationError,
        DeploymentError,
        ExportError,
        ModelError,
        MonitoringError,
        OptimizationError,
    )
except ImportError:
    # Fallback exception definitions
    class ModelError(Exception):
        """Base exception for model-related errors."""

        pass

    class DeploymentError(ModelError):
        """Exception raised for deployment-related errors."""

        pass

    class OptimizationError(ModelError):
        """Exception raised for optimization-related errors."""

        pass

    class MonitoringError(ModelError):
        """Exception raised for monitoring-related errors."""

        pass

    class ConfigurationError(ModelError):
        """Exception raised for configuration errors."""

        pass

    class ExportError(ModelError):
        """Exception raised for model export errors."""

        pass

    class AlertError(MonitoringError):
        """Exception raised for alert-related errors."""

        pass


logger = logging.getLogger(__name__)


# =============================================================================
# CORE IMPORTS
# =============================================================================

# Deployment Strategies
from .deployment_strategies import (
    # Strategy implementations
    AWSDeploymentStrategy,
    AzureDeploymentStrategy,
    ContainerDeploymentStrategy,
    # Configuration
    DeploymentConfig,
    # Manager
    DeploymentManager,
    # Base classes
    DeploymentStrategy,
    # Enums
    DeploymentTarget,
    EdgeDeploymentStrategy,
    GCPDeploymentStrategy,
    LocalDeploymentStrategy,
    ServingMode,
    # Convenience functions
    deploy_locally,
    list_deployment_targets,
)

# Model Optimization
from .model_optimization import (
    # Main optimizer
    ModelOptimizer,
    # Configuration
    OptimizationConfig,
    PruningType,
    # Enums
    QuantizationType,
    prune_for_deployment,
    # Convenience functions
    quantize_for_inference,
)

# Monitoring
from .monitoring import (
    Alert,
    AlertSeverity,
    DriftType,
    # Enums
    MetricType,
    # Main monitor
    ModelMonitor,
    # Configuration and data classes
    MonitoringConfig,
    # Convenience functions
    create_monitor,
)

# =============================================================================
# MODULE METADATA
# =============================================================================

__version__ = "1.0.0"
__author__ = "Milia Team"
__all__ = [
    # Exceptions
    "ModelError",
    "DeploymentError",
    "OptimizationError",
    "MonitoringError",
    "ConfigurationError",
    "ExportError",
    "AlertError",
    # Deployment - Base and Config
    "DeploymentStrategy",
    "DeploymentConfig",
    "DeploymentTarget",
    "ServingMode",
    # Deployment - Strategies
    "AWSDeploymentStrategy",
    "GCPDeploymentStrategy",
    "AzureDeploymentStrategy",
    "EdgeDeploymentStrategy",
    "ContainerDeploymentStrategy",
    "LocalDeploymentStrategy",
    "DeploymentManager",
    # Optimization
    "ModelOptimizer",
    "OptimizationConfig",
    "QuantizationType",
    "PruningType",
    # Monitoring
    "ModelMonitor",
    "MonitoringConfig",
    "Alert",
    "MetricType",
    "AlertSeverity",
    "DriftType",
    # High-level convenience functions (defined in __init__.py)
    "deploy_model_locally",
    "deploy_model_to_cloud",
    "optimize_model_for_deployment",
    "create_production_monitor",
    "create_deployment_pipeline",
    # Utility functions (defined in __init__.py)
    "list_deployment_targets",
    "get_deployment_info",
    "validate_deployment_config",
    # Module-level convenience functions (imported from submodules)
    "deploy_locally",
    "quantize_for_inference",
    "prune_for_deployment",
    "create_monitor",
]


# =============================================================================
# HIGH-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================


def deploy_model_locally(
    model: nn.Module, save_path: str | Path, verbose: bool = True
) -> DeploymentManager:
    """
    Quick local deployment for development and testing.

    This is the simplest way to deploy a model locally. The model is
    prepared, saved, and loaded into memory for immediate inference.

    Args:
        model: PyTorch model to deploy
        save_path: Path to save the model
        verbose: Whether to log deployment information

    Returns:
        DeploymentManager instance ready for inference

    Example:
        >>> manager = deploy_model_locally(model, "./my_model")
        >>> result = manager.predict(input_tensor)
        >>> manager.teardown()
    """
    try:
        config = DeploymentConfig(target="local")
        manager = DeploymentManager(config=config, verbose=verbose)

        model_path = manager.prepare_model(model, save_path)
        manager.deploy(model_path)

        if verbose:
            logger.info(f"Model deployed locally at {model_path}")

        return manager

    except Exception as e:
        raise DeploymentError(f"Local deployment failed: {e}") from e


def deploy_model_to_cloud(
    model: nn.Module,
    save_path: str | Path,
    target: str = "aws",
    instance_type: str | None = None,
    num_instances: int = 1,
    auto_scaling: bool = False,
    verbose: bool = True,
) -> DeploymentManager:
    """
    Deploy model to cloud platform (AWS, GCP, or Azure).

    Prepares the model for cloud deployment with proper packaging,
    dependency management, and deployment scripts.

    Args:
        model: PyTorch model to deploy
        save_path: Path to save prepared model
        target: Cloud platform ('aws', 'gcp', 'azure')
        instance_type: VM/instance type for deployment
        num_instances: Number of instances for scaling
        auto_scaling: Enable auto-scaling
        verbose: Whether to log deployment information

    Returns:
        DeploymentManager instance with deployment info

    Example:
        >>> manager = deploy_model_to_cloud(
        ...     model,
        ...     "./model",
        ...     target="aws",
        ...     instance_type="ml.m5.large",
        ...     auto_scaling=True
        ... )
        >>> print(manager.get_deployment_info())

    Raises:
        DeploymentError: If deployment fails
        ConfigurationError: If invalid target specified
    """
    try:
        if target not in ["aws", "gcp", "azure"]:
            raise ConfigurationError(
                f"Invalid cloud target: {target}. Available: ['aws', 'gcp', 'azure']"
            )

        config = DeploymentConfig(
            target=target,
            instance_type=instance_type,
            num_instances=num_instances,
            auto_scaling=auto_scaling,
        )

        manager = DeploymentManager(config=config, verbose=verbose)

        model_path = manager.prepare_model(model, save_path)
        deployment_info = manager.deploy(model_path)

        if verbose:
            logger.info(f"Model prepared for {target} deployment at {model_path}")
            logger.info(f"Deployment info: {deployment_info}")

        return manager

    except Exception as e:
        raise DeploymentError(f"Cloud deployment failed: {e}") from e


def optimize_model_for_deployment(
    model: nn.Module,
    quantize: bool = False,
    quantization_type: str = "dynamic",
    prune: bool = False,
    prune_amount: float = 0.3,
    export_onnx: bool = False,
    onnx_path: str | Path | None = None,
    dummy_input: torch.Tensor | None = None,
    optimize_for_mobile: bool = False,
    verbose: bool = True,
) -> nn.Module:
    """
    Optimize model for production deployment.

    Applies various optimization techniques including quantization,
    pruning, and model export to reduce model size and improve
    inference speed.

    Args:
        model: PyTorch model to optimize
        quantize: Enable quantization
        quantization_type: Type of quantization ('dynamic', 'static', 'fp16')
        prune: Enable pruning
        prune_amount: Fraction of weights to prune (0.0-1.0)
        export_onnx: Export to ONNX format
        onnx_path: Path for ONNX export (required if export_onnx=True)
        dummy_input: Example input for ONNX export
        optimize_for_mobile: Optimize for mobile deployment
        verbose: Whether to log optimization information

    Returns:
        Optimized PyTorch model

    Example:
        >>> # Quantize and prune
        >>> optimized = optimize_model_for_deployment(
        ...     model,
        ...     quantize=True,
        ...     prune=True,
        ...     prune_amount=0.3
        ... )
        >>>
        >>> # Export to ONNX
        >>> optimized = optimize_model_for_deployment(
        ...     model,
        ...     export_onnx=True,
        ...     onnx_path="model.onnx",
        ...     dummy_input=torch.randn(1, 3, 224, 224)
        ... )

    Raises:
        OptimizationError: If optimization fails
        ExportError: If ONNX export fails
    """
    try:
        optimizer = ModelOptimizer(
            quantization_enabled=quantize,
            quantization_type=quantization_type,
            pruning_enabled=prune,
            pruning_amount=prune_amount,
            export_onnx=export_onnx,
            optimize_for_mobile=optimize_for_mobile,
            verbose=verbose,
        )

        # Get original model size
        if verbose:
            orig_size = optimizer.get_model_size(model)
            logger.info(f"Original model size: {orig_size['total_mb']:.2f} MB")

        # Apply optimizations
        optimized_model = model

        if quantize:
            optimized_model = optimizer.quantize_model(optimized_model)
            if verbose:
                logger.info(f"Applied {quantization_type} quantization")

        if prune:
            optimized_model = optimizer.prune_model(optimized_model)
            if verbose:
                logger.info(f"Pruned {prune_amount * 100:.1f}% of weights")

        # Compare models
        if verbose and (quantize or prune):
            comparison = optimizer.compare_models(model, optimized_model)
            logger.info(
                f"Optimized model size: {comparison['optimized_size_mb']:.2f} MB "
                f"({comparison['size_reduction'] * 100:.1f}% reduction)"
            )

        # Export to ONNX
        if export_onnx:
            if onnx_path is None:
                raise ConfigurationError("onnx_path required when export_onnx=True")
            if dummy_input is None:
                raise ConfigurationError("dummy_input required for ONNX export")

            optimizer.export_to_onnx(optimized_model, onnx_path, dummy_input)

        return optimized_model

    except Exception as e:
        raise OptimizationError(f"Model optimization failed: {e}") from e


def create_production_monitor(
    model_name: str = "model",
    enable_performance_tracking: bool = True,
    enable_drift_detection: bool = True,
    enable_health_checks: bool = True,
    enable_alerting: bool = True,
    drift_threshold: float = 0.05,
    alert_threshold: float = 0.1,
    metrics_window_size: int = 1000,
    verbose: bool = True,
) -> ModelMonitor:
    """
    Create production-ready model monitor.

    Sets up comprehensive monitoring for deployed models including
    performance tracking, drift detection, health checks, and alerting.

    Args:
        model_name: Name of the model being monitored
        enable_performance_tracking: Track latency, throughput, errors
        enable_drift_detection: Detect data and model drift
        enable_health_checks: Perform periodic health checks
        enable_alerting: Send alerts on anomalies
        drift_threshold: Threshold for drift detection (0.0-1.0)
        alert_threshold: Threshold for triggering alerts
        metrics_window_size: Number of recent metrics to keep
        verbose: Whether to log monitoring information

    Returns:
        ModelMonitor instance ready for production monitoring

    Example:
        >>> monitor = create_production_monitor(
        ...     model_name="molecular_property_predictor",
        ...     drift_threshold=0.05
        ... )
        >>>
        >>> # Register alert callback
        >>> def alert_handler(alert):
        ...     print(f"ALERT: {alert.message}")
        >>> monitor.register_alert_callback(alert_handler)
        >>>
        >>> # Log predictions
        >>> monitor.log_prediction(
        ...     input_data=data,
        ...     output=prediction,
        ...     latency=0.05,
        ...     ground_truth=label
        ... )
        >>>
        >>> # Check health
        >>> health = monitor.health_check()
        >>> print(f"Status: {health['status']}")
        >>>
        >>> # Get metrics
        >>> metrics = monitor.get_metrics_summary()
    """
    try:
        config = MonitoringConfig(
            enable_performance_tracking=enable_performance_tracking,
            enable_drift_detection=enable_drift_detection,
            enable_health_checks=enable_health_checks,
            enable_alerting=enable_alerting,
            drift_threshold=drift_threshold,
            alert_threshold=alert_threshold,
            metrics_window_size=metrics_window_size,
        )

        monitor = ModelMonitor(config=config, model_name=model_name, verbose=verbose)

        if verbose:
            logger.info(
                f"Production monitor created for '{model_name}' - All monitoring features enabled"
            )

        return monitor

    except Exception as e:
        raise MonitoringError(f"Monitor creation failed: {e}") from e


def create_deployment_pipeline(
    model: nn.Module,
    save_path: str | Path,
    target: str = "local",
    optimize: bool = True,
    quantize: bool = False,
    prune: bool = False,
    prune_amount: float = 0.3,
    enable_monitoring: bool = True,
    model_name: str = "model",
    instance_type: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Create complete end-to-end deployment pipeline.

    This function combines model optimization, deployment, and monitoring
    into a single workflow for streamlined production deployment.

    Args:
        model: PyTorch model to deploy
        save_path: Path to save prepared model
        target: Deployment target ('local', 'aws', 'gcp', 'azure', etc.)
        optimize: Apply optimization techniques
        quantize: Enable quantization
        prune: Enable pruning
        prune_amount: Fraction of weights to prune
        enable_monitoring: Set up production monitoring
        model_name: Name for monitoring
        instance_type: Instance type for cloud deployment
        verbose: Whether to log pipeline information

    Returns:
        Dictionary containing:
            - 'manager': DeploymentManager instance
            - 'monitor': ModelMonitor instance (if enabled)
            - 'deployment_info': Deployment information
            - 'optimization_summary': Optimization results (if applied)

    Example:
        >>> # Complete pipeline with optimization and monitoring
        >>> pipeline = create_deployment_pipeline(
        ...     model=my_model,
        ...     save_path="./deployment",
        ...     target="local",
        ...     optimize=True,
        ...     quantize=True,
        ...     prune=True,
        ...     enable_monitoring=True,
        ...     model_name="my_production_model"
        ... )
        >>>
        >>> # Use deployed model
        >>> result = pipeline['manager'].predict(input_data)
        >>>
        >>> # Monitor performance
        >>> pipeline['monitor'].log_prediction(
        ...     input_data=input_data,
        ...     output=result,
        ...     latency=0.05
        ... )
        >>>
        >>> # Get status
        >>> metrics = pipeline['monitor'].get_metrics_summary()
        >>> deployment_info = pipeline['manager'].get_deployment_info()

    Raises:
        DeploymentError: If deployment pipeline fails
    """
    try:
        pipeline_result = {}

        if verbose:
            logger.info(
                f"Creating deployment pipeline - "
                f"Target: {target}, Optimize: {optimize}, Monitor: {enable_monitoring}"
            )

        # Step 1: Optimization
        deployment_model = model
        if optimize:
            if verbose:
                logger.info("Step 1/3: Optimizing model...")

            deployment_model = optimize_model_for_deployment(
                model=model,
                quantize=quantize,
                prune=prune,
                prune_amount=prune_amount,
                verbose=verbose,
            )

            # Get optimization summary
            optimizer = ModelOptimizer(verbose=False)
            pipeline_result["optimization_summary"] = optimizer.compare_models(
                model, deployment_model
            )

        # Step 2: Deployment
        if verbose:
            logger.info(f"Step 2/3: Deploying to {target}...")

        if target == "local":
            manager = deploy_model_locally(deployment_model, save_path, verbose=verbose)
        elif target in ["aws", "gcp", "azure"]:
            manager = deploy_model_to_cloud(
                deployment_model,
                save_path,
                target=target,
                instance_type=instance_type,
                verbose=verbose,
            )
        else:
            # Use generic deployment manager
            config = DeploymentConfig(target=target, instance_type=instance_type)
            manager = DeploymentManager(config=config, verbose=verbose)
            model_path = manager.prepare_model(deployment_model, save_path)
            manager.deploy(model_path)

        pipeline_result["manager"] = manager
        pipeline_result["deployment_info"] = manager.get_deployment_info()

        # Step 3: Monitoring
        if enable_monitoring:
            if verbose:
                logger.info("Step 3/3: Setting up monitoring...")

            monitor = create_production_monitor(model_name=model_name, verbose=verbose)
            pipeline_result["monitor"] = monitor

        if verbose:
            logger.info("Deployment pipeline created successfully")
            logger.info(f"Target: {target}")
            if optimize:
                opt_summary = pipeline_result.get("optimization_summary", {})
                logger.info(
                    f"Optimization: {opt_summary.get('size_reduction', 0) * 100:.1f}% "
                    f"size reduction"
                )
            logger.info(f"Monitoring: {'Enabled' if enable_monitoring else 'Disabled'}")

        return pipeline_result

    except Exception as e:
        raise DeploymentError(f"Deployment pipeline failed: {e}") from e


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_deployment_info(manager: DeploymentManager) -> dict[str, Any]:
    """
    Get comprehensive deployment information.

    Args:
        manager: DeploymentManager instance

    Returns:
        Dictionary with deployment details
    """
    return manager.get_deployment_info()


def validate_deployment_config(config: DeploymentConfig) -> bool:
    """
    Validate deployment configuration.

    Args:
        config: DeploymentConfig to validate

    Returns:
        True if valid

    Raises:
        ConfigurationError: If configuration is invalid
    """
    valid_targets = list_deployment_targets()

    if config.target not in valid_targets:
        raise ConfigurationError(
            f"Invalid deployment target: {config.target}. Available: {valid_targets}"
        )

    if config.num_instances < 1:
        raise ConfigurationError("num_instances must be >= 1")

    if config.auto_scaling:
        if config.min_instances < 1:
            raise ConfigurationError("min_instances must be >= 1")
        if config.max_instances < config.min_instances:
            raise ConfigurationError("max_instances must be >= min_instances")

    if config.max_batch_size < 1:
        raise ConfigurationError("max_batch_size must be >= 1")

    if config.timeout_seconds < 1:
        raise ConfigurationError("timeout_seconds must be >= 1")

    return True


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

# Log module initialization
logger.info(
    f"Deployment module loaded (v{__version__}) - Available targets: {list_deployment_targets()}"
)

# Verify critical components are available
_critical_components = ["DeploymentManager", "ModelOptimizer", "ModelMonitor"]

for component in _critical_components:
    if component not in globals():
        logger.error(f"Critical component '{component}' not loaded")
    else:
        logger.debug(f"Component '{component}' loaded successfully")
