# models/acceleration/__init__
"""
Acceleration Module

Comprehensive hardware acceleration and optimization for model training and inference.
Provides device management, memory optimization, computation optimization, and distributed
training strategies for efficient scaling across CPU, GPU, TPU, and multi-node setups.

This module integrates multiple acceleration strategies:
- Device Management: Auto-detection, multi-GPU support, CPU/CUDA/MPS/TPU
- Memory Optimization: Mixed precision, gradient checkpointing, memory profiling
- Computation Optimization: torch.compile, cuDNN, JIT compilation, kernel fusion
- Distributed Training: DataParallel, DistributedDataParallel, FSDP, DeepSpeed, Horovod

Key Features:
- Automatic device detection with priority ordering
- Mixed precision training (FP16, BF16, FP8)
- Gradient checkpointing for memory efficiency
- torch.compile support (PyTorch 2.0+)
- Multi-GPU and multi-node distributed training
- Memory profiling and leak detection
- Performance benchmarking utilities
- Production-ready optimization presets

Usage Examples:

    Basic Device Setup:
        >>> from milia_pipeline.models.acceleration import DeviceManager
        >>> manager = DeviceManager()  # Auto-detect best device
        >>> device = manager.get_device()
        >>> model = model.to(device)

    Memory Optimization:
        >>> from milia_pipeline.models.acceleration import MemoryOptimizer
        >>> optimizer = MemoryOptimizer(mixed_precision=True, gradient_checkpointing=True)
        >>> model = optimizer.enable_gradient_checkpointing(model)
        >>> 
        >>> # Training with mixed precision
        >>> with optimizer.autocast():
        ...     output = model(input)
        ...     loss = criterion(output, target)

    Computation Optimization:
        >>> from milia_pipeline.models.acceleration import ComputationOptimizer
        >>> optimizer = ComputationOptimizer(compile_model=True, cudnn_benchmark=True)
        >>> model = optimizer.optimize_model(model)

    Distributed Training:
        >>> from milia_pipeline.models.acceleration import DistributedManager
        >>> manager = DistributedManager(strategy="ddp")
        >>> manager.setup()
        >>> model = manager.wrap_model(model)
        >>> # Training...
        >>> manager.cleanup()

    Quick Setup (Auto-optimization):
        >>> from milia_pipeline.models.acceleration import auto_optimize_for_training
        >>> device, model, optimizer_instance = auto_optimize_for_training(
        ...     model,
        ...     model_size="large",
        ...     enable_distributed=True
        ... )

Module Structure:
- device_manager: Device detection, selection, and monitoring
- memory_optimization: Memory efficiency and mixed precision
- computation_optimization: Computation speed and optimization
- distributed_strategies: Multi-GPU and multi-node training

Author: milia Team
Version: 1.0.0
"""

import logging
from typing import Optional, Dict, Any, List, Union, Tuple

import torch
import torch.nn as nn

# =============================================================================
# VERSION AND METADATA
# =============================================================================

__version__ = "1.0.0"
__author__ = "milia Team"
__description__ = "Hardware acceleration and optimization for milia Pipeline"


# =============================================================================
# LOGGING SETUP
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# CORE IMPORTS
# =============================================================================

# Device Management
from .device_manager import (
    DeviceManager,
    DeviceInfo,
    DeviceType,
    get_default_device,
    list_available_devices,
    get_device_capabilities
)

# Memory Optimization
from .memory_optimization import (
    MemoryOptimizer,
    MemoryConfig,
    get_memory_efficient_settings,
    estimate_model_memory
)

# Computation Optimization
from .computation_optimization import (
    ComputationOptimizer,
    ComputationConfig,
    get_optimal_settings,
    auto_optimize_model,
    optimize_inference
)

# Distributed Strategies
from .distributed_strategies import (
    DistributedManager,
    DistributedConfig,
    DistributedStrategy,
    DistributedBackend,
    is_distributed_available,
    get_world_size,
    get_rank,
    is_main_process
)


# =============================================================================
# EXCEPTION IMPORTS (with fallback)
# =============================================================================

try:
    from milia_pipeline.exceptions import (
        ModelError,
        HardwareError,
        DeviceNotAvailableError,
        MemoryError as VQMMemoryError,
        OptimizationError,
        DistributedError
    )
except ImportError:
    # Fallback exception definitions
    class ModelError(Exception):
        """Base exception for model-related errors."""
        pass
    
    class HardwareError(ModelError):
        """Exception raised for hardware-related errors."""
        pass
    
    class DeviceNotAvailableError(HardwareError):
        """Exception raised when requested device is not available."""
        pass
    
    class VQMMemoryError(HardwareError):
        """Exception raised for memory-related errors."""
        pass
    
    class OptimizationError(HardwareError):
        """Exception raised for optimization-related errors."""
        pass
    
    class DistributedError(HardwareError):
        """Exception raised for distributed training errors."""
        pass


# =============================================================================
# UNIFIED ACCELERATION MANAGER
# =============================================================================

class AccelerationManager:
    """
    Unified manager for all acceleration strategies.
    
    Provides a single interface for device management, memory optimization,
    computation optimization, and distributed training setup.
    
    Usage:
        >>> # Complete setup with all optimizations
        >>> manager = AccelerationManager(
        ...     device="auto",
        ...     mixed_precision=True,
        ...     gradient_checkpointing=True,
        ...     compile_model=True,
        ...     distributed_strategy="ddp"
        ... )
        >>> 
        >>> # Initialize and optimize model
        >>> manager.setup()
        >>> device = manager.get_device()
        >>> model = manager.optimize_model(model)
        >>> 
        >>> # Training with optimizations
        >>> with manager.autocast():
        ...     output = model(input)
        ...     loss = criterion(output, target)
        >>> 
        >>> # Cleanup
        >>> manager.cleanup()
    """
    
    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        mixed_precision: bool = False,
        precision: str = "fp16",
        gradient_checkpointing: bool = False,
        compile_model: bool = False,
        compile_mode: str = "default",
        cudnn_benchmark: bool = True,
        distributed_strategy: Union[str, DistributedStrategy] = "none",
        distributed_backend: Union[str, DistributedBackend] = "auto",
        verbose: bool = True
    ):
        """
        Initialize unified acceleration manager.
        
        Args:
            device: Device specification (auto, cuda, cuda:0, mps, cpu)
            mixed_precision: Enable mixed precision training
            precision: Precision type (fp16, bf16, fp32)
            gradient_checkpointing: Enable gradient checkpointing
            compile_model: Enable torch.compile
            compile_mode: Compilation mode (default, reduce-overhead, max-autotune)
            cudnn_benchmark: Enable cuDNN benchmark mode
            distributed_strategy: Distributed strategy (none, dp, ddp, fsdp)
            distributed_backend: Communication backend (auto, nccl, gloo, mpi)
            verbose: Whether to log information
        """
        self.verbose = verbose
        self._is_setup = False
        
        # Initialize device manager
        self.device_manager = DeviceManager(
            device=device,
            verbose=verbose
        )
        
        # Initialize memory optimizer
        self.memory_optimizer = MemoryOptimizer(
            mixed_precision=mixed_precision,
            precision=precision,
            gradient_checkpointing=gradient_checkpointing,
            device=self.device_manager.get_device(),
            verbose=verbose
        )
        
        # Initialize computation optimizer
        self.computation_optimizer = ComputationOptimizer(
            compile_model=compile_model,
            compile_mode=compile_mode,
            cudnn_benchmark=cudnn_benchmark,
            device=self.device_manager.get_device(),
            verbose=verbose
        )
        
        # Initialize distributed manager
        self.distributed_manager = DistributedManager(
            strategy=distributed_strategy,
            backend=distributed_backend,
            mixed_precision=mixed_precision,
            verbose=verbose
        )
        
        if self.verbose:
            logger.info(
                f"AccelerationManager initialized - "
                f"Device: {self.device_manager.get_device()}, "
                f"Mixed Precision: {mixed_precision}, "
                f"Compile: {compile_model}, "
                f"Distributed: {distributed_strategy}"
            )
    
    def setup(self):
        """
        Setup acceleration components.
        
        Must be called before training, especially for distributed training.
        """
        if self._is_setup:
            logger.warning("AccelerationManager already setup")
            return
        
        # Setup distributed training if needed
        if self.distributed_manager.config.strategy != DistributedStrategy.NONE:
            self.distributed_manager.setup()
        
        self._is_setup = True
        
        if self.verbose:
            logger.info("AccelerationManager setup complete")
    
    def optimize_model(
        self,
        model: nn.Module,
        move_to_device: bool = True
    ) -> nn.Module:
        """
        Apply all optimization strategies to model.
        
        Args:
            model: Model to optimize
            move_to_device: Whether to move model to device
            
        Returns:
            Optimized model
        """
        # Move to device first
        if move_to_device:
            model = self.device_manager.move_to_device(model)
        
        # Apply gradient checkpointing
        if self.memory_optimizer.config.gradient_checkpointing:
            model = self.memory_optimizer.enable_gradient_checkpointing(model)
        
        # Apply computation optimizations
        model = self.computation_optimizer.optimize_model(model)
        
        # Wrap for distributed training
        if self.distributed_manager.config.strategy != DistributedStrategy.NONE:
            model = self.distributed_manager.wrap_model(model)
        
        if self.verbose:
            logger.info("Model optimization complete")
        
        return model
    
    def get_device(self) -> torch.device:
        """Get the configured device."""
        return self.device_manager.get_device()
    
    def get_grad_scaler(self):
        """Get gradient scaler for mixed precision training."""
        return self.memory_optimizer.get_grad_scaler()
    
    def autocast(self):
        """Context manager for automatic mixed precision."""
        return self.memory_optimizer.autocast()
    
    def step(self):
        """
        Perform step operations (cache clearing, GC, etc.).
        
        Should be called after each training step.
        """
        self.memory_optimizer.step()
    
    def is_main_process(self) -> bool:
        """Check if current process is main process."""
        return self.distributed_manager.is_main_process()
    
    def barrier(self):
        """Synchronize all processes."""
        self.distributed_manager.barrier()
    
    def cleanup(self):
        """Cleanup acceleration components."""
        if self.distributed_manager.config.strategy != DistributedStrategy.NONE:
            self.distributed_manager.cleanup()
        
        self._is_setup = False
        
        if self.verbose:
            logger.info("AccelerationManager cleanup complete")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        return self.memory_optimizer.get_memory_stats()
    
    def get_device_info(self) -> DeviceInfo:
        """Get device information."""
        return self.device_manager.get_device_info()
    
    def print_summary(self):
        """Print comprehensive acceleration summary."""
        print("=" * 70)
        print("Acceleration Configuration Summary")
        print("=" * 70)
        
        # Device info
        device_info = self.device_manager.get_device_info()
        print(f"\nDevice: {device_info.device_type.upper()}")
        if device_info.name:
            print(f"  Name: {device_info.name}")
        if device_info.total_memory:
            print(f"  Memory: {device_info.memory_summary()}")
        
        # Memory optimization
        print(f"\nMemory Optimization:")
        print(f"  Mixed Precision: {self.memory_optimizer.config.mixed_precision}")
        print(f"  Precision: {self.memory_optimizer.config.precision}")
        print(f"  Gradient Checkpointing: {self.memory_optimizer.config.gradient_checkpointing}")
        
        # Computation optimization
        print(f"\nComputation Optimization:")
        print(f"  torch.compile: {self.computation_optimizer.config.compile_model}")
        if self.computation_optimizer.config.compile_model:
            print(f"  Compile Mode: {self.computation_optimizer.config.compile_mode}")
        print(f"  cuDNN Benchmark: {self.computation_optimizer.config.cudnn_benchmark}")
        
        # Distributed training
        print(f"\nDistributed Training:")
        print(f"  Strategy: {self.distributed_manager.config.strategy.value.upper()}")
        if self.distributed_manager.config.strategy != DistributedStrategy.NONE:
            print(f"  Backend: {self.distributed_manager.config.backend.value}")
            print(f"  World Size: {self.distributed_manager.config.world_size}")
            print(f"  Rank: {self.distributed_manager.config.rank}")
        
        print("=" * 70)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def auto_optimize_for_training(
    model: nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    model_size: str = "medium",
    enable_distributed: bool = False,
    distributed_strategy: str = "ddp",
    verbose: bool = True
) -> Tuple[torch.device, nn.Module, AccelerationManager]:
    """
    Automatically optimize model for training with recommended settings.
    
    Applies optimal settings based on model size and available hardware.
    
    Args:
        model: Model to optimize
        device: Target device (None=auto-detect)
        model_size: Model size category (small, medium, large, xlarge)
        enable_distributed: Enable distributed training
        distributed_strategy: Distributed strategy (dp, ddp, fsdp)
        verbose: Whether to log information
        
    Returns:
        Tuple of (device, optimized_model, acceleration_manager)
        
    Example:
        >>> device, model, manager = auto_optimize_for_training(
        ...     model,
        ...     model_size="large",
        ...     enable_distributed=True
        ... )
        >>> 
        >>> # Training loop
        >>> for batch in dataloader:
        ...     with manager.autocast():
        ...         output = model(batch)
        ...         loss = criterion(output, target)
        ...     manager.step()
        >>> 
        >>> manager.cleanup()
    """
    # Get memory-efficient settings
    memory_settings = get_memory_efficient_settings(
        torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        model_size=model_size
    )
    
    # Determine if we should compile
    compile_model = model_size in ["small", "medium"]
    
    # Create manager with optimal settings
    manager = AccelerationManager(
        device=device,
        mixed_precision=memory_settings['mixed_precision'],
        gradient_checkpointing=memory_settings['gradient_checkpointing'],
        compile_model=compile_model,
        compile_mode="default" if model_size in ["small", "medium"] else "reduce-overhead",
        cudnn_benchmark=True,
        distributed_strategy=distributed_strategy if enable_distributed else "none",
        verbose=verbose
    )
    
    # Setup and optimize
    manager.setup()
    optimized_model = manager.optimize_model(model)
    device = manager.get_device()
    
    if verbose:
        logger.info(
            f"Auto-optimization complete - "
            f"Model Size: {model_size}, "
            f"Device: {device}, "
            f"Distributed: {enable_distributed}"
        )
    
    return device, optimized_model, manager


def auto_optimize_for_inference(
    model: nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    optimize_for_latency: bool = True,
    verbose: bool = True
) -> Tuple[torch.device, nn.Module, AccelerationManager]:
    """
    Automatically optimize model for inference with recommended settings.
    
    Args:
        model: Model to optimize
        device: Target device (None=auto-detect)
        optimize_for_latency: Optimize for latency vs throughput
        verbose: Whether to log information
        
    Returns:
        Tuple of (device, optimized_model, acceleration_manager)
        
    Example:
        >>> device, model, manager = auto_optimize_for_inference(model)
        >>> 
        >>> # Inference
        >>> with torch.no_grad():
        ...     with manager.autocast():
        ...         output = model(input)
    """
    manager = AccelerationManager(
        device=device,
        mixed_precision=True,
        precision="fp16",
        gradient_checkpointing=False,
        compile_model=True,
        compile_mode="max-autotune" if optimize_for_latency else "default",
        cudnn_benchmark=True,
        distributed_strategy="none",
        verbose=verbose
    )
    
    manager.setup()
    optimized_model = manager.optimize_model(model)
    device = manager.get_device()
    
    if verbose:
        logger.info(
            f"Inference optimization complete - "
            f"Device: {device}, "
            f"Latency Optimized: {optimize_for_latency}"
        )
    
    return device, optimized_model, manager


def get_recommended_settings(
    model: nn.Module,
    device: Optional[torch.device] = None,
    task_type: str = "training"
) -> Dict[str, Any]:
    """
    Get recommended acceleration settings for model and task.
    
    Args:
        model: Model to analyze
        device: Target device
        task_type: Task type (training, inference)
        
    Returns:
        Dictionary with recommended settings
        
    Example:
        >>> settings = get_recommended_settings(model, task_type="training")
        >>> manager = AccelerationManager(**settings)
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Estimate model size
    param_count = sum(p.numel() for p in model.parameters())
    
    if param_count < 10_000_000:  # < 10M parameters
        model_size = "small"
    elif param_count < 100_000_000:  # < 100M parameters
        model_size = "medium"
    elif param_count < 500_000_000:  # < 500M parameters
        model_size = "large"
    else:
        model_size = "xlarge"
    
    # Get memory settings
    memory_settings = get_memory_efficient_settings(device, model_size)
    
    # Get computation settings
    computation_settings = get_optimal_settings(device, task_type)
    
    # Combine settings
    settings = {
        'device': device,
        'mixed_precision': memory_settings['mixed_precision'],
        'gradient_checkpointing': memory_settings['gradient_checkpointing'],
        'compile_model': computation_settings['compile_model'],
        'compile_mode': computation_settings['compile_mode'],
        'cudnn_benchmark': computation_settings['cudnn_benchmark'],
        'model_size': model_size,
        'task_type': task_type
    }
    
    return settings


def create_acceleration_manager(
    config: Dict[str, Any],
    verbose: bool = True
) -> AccelerationManager:
    """
    Create acceleration manager from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        verbose: Whether to log information
        
    Returns:
        Configured AccelerationManager instance
        
    Example:
        >>> config = {
        ...     'device': 'cuda',
        ...     'mixed_precision': True,
        ...     'compile_model': True,
        ...     'distributed_strategy': 'ddp'
        ... }
        >>> manager = create_acceleration_manager(config)
    """
    return AccelerationManager(**config, verbose=verbose)


def benchmark_accelerations(
    model: nn.Module,
    input_data: torch.Tensor,
    configurations: List[Dict[str, Any]],
    num_iterations: int = 100
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different acceleration configurations.
    
    Args:
        model: Model to benchmark
        input_data: Sample input tensor
        configurations: List of configuration dictionaries
        num_iterations: Number of iterations per config
        
    Returns:
        Dictionary mapping config names to performance metrics
        
    Example:
        >>> configs = [
        ...     {'name': 'baseline', 'compile_model': False},
        ...     {'name': 'compiled', 'compile_model': True},
        ...     {'name': 'mixed_precision', 'mixed_precision': True}
        ... ]
        >>> results = benchmark_accelerations(model, input_data, configs)
        >>> for name, metrics in results.items():
        ...     print(f"{name}: {metrics['avg_time_ms']:.2f}ms")
    """
    results = {}
    
    for config in configurations:
        config_name = config.pop('name', 'unnamed')
        
        # Create manager
        manager = AccelerationManager(**config, verbose=False)
        manager.setup()
        
        # Optimize model
        optimized_model = manager.optimize_model(model)
        device = manager.get_device()
        input_on_device = input_data.to(device)
        
        # Benchmark
        comp_optimizer = ComputationOptimizer(verbose=False)
        bench_results = comp_optimizer.benchmark_model(
            optimized_model,
            input_on_device,
            num_iterations=num_iterations
        )
        
        results[config_name] = bench_results
        
        # Cleanup
        manager.cleanup()
    
    return results


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Version
    '__version__',
    '__author__',
    '__description__',
    
    # Core Managers
    'AccelerationManager',
    'DeviceManager',
    'MemoryOptimizer',
    'ComputationOptimizer',
    'DistributedManager',
    
    # Configuration Classes
    'DeviceInfo',
    'DeviceType',
    'MemoryConfig',
    'ComputationConfig',
    'DistributedConfig',
    'DistributedStrategy',
    'DistributedBackend',
    
    # Device Management Functions
    'get_default_device',
    'list_available_devices',
    'get_device_capabilities',
    
    # Memory Optimization Functions
    'get_memory_efficient_settings',
    'estimate_model_memory',
    
    # Computation Optimization Functions
    'get_optimal_settings',
    'auto_optimize_model',
    'optimize_inference',
    
    # Distributed Training Functions
    'is_distributed_available',
    'get_world_size',
    'get_rank',
    'is_main_process',
    
    # Convenience Functions
    'auto_optimize_for_training',
    'auto_optimize_for_inference',
    'get_recommended_settings',
    'create_acceleration_manager',
    'benchmark_accelerations',
    
    # Exceptions
    'ModelError',
    'HardwareError',
    'DeviceNotAvailableError',
    'VQMMemoryError',
    'OptimizationError',
    'DistributedError',
]


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info(f"acceleration module v{__version__} loaded")
logger.info(
    f"Available acceleration features: "
    f"Device Management, Memory Optimization, Computation Optimization, Distributed Training"
)

# Check for available hardware acceleration
_capabilities = get_device_capabilities()
if _capabilities['cuda_available']:
    logger.info(f"CUDA available: {_capabilities['cuda_device_count']} device(s)")
if _capabilities['mps_available']:
    logger.info("MPS (Apple Silicon) available")
if _capabilities['tpu_available']:
    logger.info("TPU available")
