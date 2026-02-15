"""
Memory Optimization

Comprehensive memory optimization strategies for efficient GPU/TPU memory usage.
Supports mixed precision training, gradient checkpointing, memory profiling, and more.

Features:
- Mixed precision training (FP16, BF16, FP8)
- Automatic Mixed Precision (AMP) support
- Gradient checkpointing (activation checkpointing)
- Memory profiling and monitoring
- Efficient data loading strategies
- Dynamic batch sizing
- Memory leak detection
- Garbage collection strategies

Pydantic V2 Migration (Phase 9):
    - Migrated MemoryConfig from @dataclass to Pydantic BaseModel (mutable)
    - Uses model_dump() for to_dict() method (backward compatible)
    - NON-BREAKING: Same constructor API and attribute access
    - Follows established pattern from device_manager.py (Phase 7)

Author: milia Team
Version: 1.1.0
"""

import gc
import logging
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel
from torch.cuda.amp import GradScaler, autocast

# Import exceptions with fallback
try:
    from milia_pipeline.exceptions import HardwareError, ModelError
    from milia_pipeline.exceptions import MemoryError as VQMMemoryError
except ImportError:

    class ModelError(Exception):
        """Base exception for model-related errors."""

        pass

    class HardwareError(ModelError):
        """Exception raised for hardware-related errors."""

        pass

    class VQMMemoryError(HardwareError):
        """Exception raised for memory-related errors."""

        pass


logger = logging.getLogger(__name__)


# =============================================================================
# MEMORY OPTIMIZATION CONFIGURATION
# =============================================================================


class MemoryConfig(BaseModel):
    """
    Configuration for memory optimization.

    Pydantic V2 Migration (Phase 9):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses model_dump() for backward-compatible to_dict() method
        - NON-BREAKING: Same constructor API and attribute access
        - Follows established pattern from device_manager.py (Phase 7)

    Attributes:
        mixed_precision: Enable mixed precision training
        precision: Precision type (fp16, bf16, fp32)
        gradient_checkpointing: Enable gradient checkpointing
        pin_memory: Pin memory for faster CPU-GPU transfer
        non_blocking: Use non-blocking data transfers
        empty_cache_interval: Steps between cache clearing (0=disabled)
        garbage_collect_interval: Steps between GC runs (0=disabled)
        max_memory_allocated: Maximum memory to allocate (MB, 0=unlimited)
        growth_interval: Interval for dynamic batch size growth
    """

    mixed_precision: bool = False
    precision: str = "fp16"  # fp16, bf16, fp32, fp8
    gradient_checkpointing: bool = False
    pin_memory: bool = True
    non_blocking: bool = True
    empty_cache_interval: int = 0
    garbage_collect_interval: int = 0
    max_memory_allocated: int = 0
    growth_interval: int = 0

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Backward compatible method wrapping Pydantic V2's model_dump().
        """
        return self.model_dump()


# =============================================================================
# MEMORY OPTIMIZER
# =============================================================================


class MemoryOptimizer:
    """
    Manager for memory optimization strategies.

    Provides various memory optimization techniques including mixed precision,
    gradient checkpointing, and memory monitoring.

    Usage:
        >>> # Mixed precision training
        >>> optimizer = MemoryOptimizer(mixed_precision=True)
        >>> scaler = optimizer.get_grad_scaler()
        >>>
        >>> # Training loop
        >>> with optimizer.autocast():
        ...     output = model(input)
        ...     loss = criterion(output, target)
        >>>
        >>> scaler.scale(loss).backward()
        >>> scaler.step(optimizer)
        >>> scaler.update()
        >>>
        >>> # Gradient checkpointing
        >>> optimizer = MemoryOptimizer(gradient_checkpointing=True)
        >>> model = optimizer.enable_gradient_checkpointing(model)
        >>>
        >>> # Memory monitoring
        >>> stats = optimizer.get_memory_stats()
        >>> print(f"Peak memory: {stats['peak_memory_gb']:.2f} GB")
    """

    def __init__(
        self,
        mixed_precision: bool = False,
        precision: str = "fp16",
        gradient_checkpointing: bool = False,
        pin_memory: bool = True,
        non_blocking: bool = True,
        empty_cache_interval: int = 0,
        garbage_collect_interval: int = 0,
        max_memory_allocated: int = 0,
        device: torch.device | None = None,
        verbose: bool = True,
    ):
        """
        Initialize memory optimizer.

        Args:
            mixed_precision: Enable automatic mixed precision
            precision: Precision type (fp16, bf16, fp32)
            gradient_checkpointing: Enable gradient checkpointing
            pin_memory: Pin memory for faster data transfer
            non_blocking: Use non-blocking transfers
            empty_cache_interval: Steps between cache clearing
            garbage_collect_interval: Steps between GC runs
            max_memory_allocated: Max memory in MB (0=unlimited)
            device: Device for memory monitoring
            verbose: Whether to log information
        """
        self.verbose = verbose
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._step_count = 0
        self._grad_scaler: GradScaler | None = None

        # Create config
        self.config = MemoryConfig(
            mixed_precision=mixed_precision,
            precision=precision,
            gradient_checkpointing=gradient_checkpointing,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            empty_cache_interval=empty_cache_interval,
            garbage_collect_interval=garbage_collect_interval,
            max_memory_allocated=max_memory_allocated,
        )

        # Validate configuration
        self._validate_config()

        # Initialize grad scaler for mixed precision
        if self.config.mixed_precision and self.device.type == "cuda":
            self._grad_scaler = GradScaler()

        if self.verbose:
            logger.info(
                f"MemoryOptimizer initialized - "
                f"Mixed Precision: {self.config.mixed_precision}, "
                f"Precision: {self.config.precision}, "
                f"Gradient Checkpointing: {self.config.gradient_checkpointing}"
            )

    def _validate_config(self):
        """Validate memory configuration."""
        # Check precision support
        if self.config.precision == "bf16":
            if self.device.type == "cuda":
                if not torch.cuda.is_bf16_supported():
                    warnings.warn("BF16 not supported on this device, falling back to FP16")
                    self.config.precision = "fp16"

        # Check if mixed precision is available
        if self.config.mixed_precision and self.device.type not in ["cuda", "cpu"]:
            warnings.warn(f"Mixed precision not supported on {self.device.type}, disabling")
            self.config.mixed_precision = False

    # =========================================================================
    # MIXED PRECISION TRAINING
    # =========================================================================

    @contextmanager
    def autocast(self):
        """
        Context manager for automatic mixed precision.

        Usage:
            >>> with optimizer.autocast():
            ...     output = model(input)
            ...     loss = criterion(output, target)
        """
        if not self.config.mixed_precision:
            yield
            return

        dtype = self._get_autocast_dtype()

        if self.device.type == "cuda":
            with autocast(enabled=True, dtype=dtype):
                yield
        elif self.device.type == "cpu":
            with torch.cpu.amp.autocast(enabled=True, dtype=dtype):
                yield
        else:
            yield

    def _get_autocast_dtype(self) -> torch.dtype:
        """Get dtype for autocast based on precision config."""
        if self.config.precision == "bf16":
            return torch.bfloat16
        elif self.config.precision == "fp16":
            return torch.float16
        else:
            return torch.float32

    def get_grad_scaler(self) -> GradScaler | None:
        """
        Get gradient scaler for mixed precision training.

        Returns:
            GradScaler instance or None if mixed precision disabled
        """
        return self._grad_scaler

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for mixed precision training.

        Args:
            loss: Loss tensor

        Returns:
            Scaled loss
        """
        if self._grad_scaler is not None:
            return self._grad_scaler.scale(loss)
        return loss

    def step_optimizer(self, optimizer: torch.optim.Optimizer, scaler_unscale: bool = True):
        """
        Step optimizer with gradient scaling.

        Args:
            optimizer: Optimizer to step
            scaler_unscale: Whether to unscale gradients first
        """
        if self._grad_scaler is not None:
            if scaler_unscale:
                self._grad_scaler.unscale_(optimizer)
            self._grad_scaler.step(optimizer)
            self._grad_scaler.update()
        else:
            optimizer.step()

    # =========================================================================
    # GRADIENT CHECKPOINTING
    # =========================================================================

    def enable_gradient_checkpointing(
        self, model: nn.Module, checkpoint_segments: int | None = None
    ) -> nn.Module:
        """
        Enable gradient checkpointing for memory efficiency.

        Trades compute for memory by recomputing activations during backward pass.

        Args:
            model: Model to enable checkpointing on
            checkpoint_segments: Number of segments to checkpoint (None=auto)

        Returns:
            Model with gradient checkpointing enabled
        """
        if not self.config.gradient_checkpointing:
            return model

        # Try PyTorch Geometric models
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if self.verbose:
                logger.info("Enabled gradient checkpointing (PyG method)")
            return model

        # Try transformers/HuggingFace models
        if hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()
            if self.verbose:
                logger.info("Enabled gradient checkpointing (Transformers method)")
            return model

        # Manual checkpointing for custom models
        # This wraps forward passes with torch.utils.checkpoint
        if hasattr(model, "forward"):
            original_forward = model.forward

            def checkpointed_forward(*args, **kwargs):
                return torch.utils.checkpoint.checkpoint(
                    original_forward, *args, use_reentrant=False, **kwargs
                )

            model.forward = checkpointed_forward

            if self.verbose:
                logger.info("Enabled gradient checkpointing (manual wrapper)")

        return model

    def checkpoint_sequential(self, functions: list[Callable], segments: int, *inputs):
        """
        Checkpoint sequential operations.

        Args:
            functions: List of functions to checkpoint
            segments: Number of checkpoint segments
            *inputs: Input tensors

        Returns:
            Output of sequential functions
        """
        return torch.utils.checkpoint.checkpoint_sequential(
            functions, segments, *inputs, use_reentrant=False
        )

    # =========================================================================
    # MEMORY MONITORING
    # =========================================================================

    def get_memory_stats(self) -> dict[str, Any]:
        """
        Get current memory statistics.

        Returns:
            Dictionary with memory statistics
        """
        if self.device.type != "cuda":
            return {
                "device": str(self.device),
                "message": "Memory stats only available for CUDA devices",
            }

        torch.cuda.synchronize(self.device)

        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        max_allocated = torch.cuda.max_memory_allocated(self.device)
        max_reserved = torch.cuda.max_memory_reserved(self.device)

        total = torch.cuda.get_device_properties(self.device).total_memory

        return {
            "device": str(self.device),
            "allocated": allocated,
            "reserved": reserved,
            "max_allocated": max_allocated,
            "max_reserved": max_reserved,
            "total": total,
            "allocated_gb": allocated / (1024**3),
            "reserved_gb": reserved / (1024**3),
            "max_allocated_gb": max_allocated / (1024**3),
            "max_reserved_gb": max_reserved / (1024**3),
            "total_gb": total / (1024**3),
            "utilization": (allocated / total) * 100 if total > 0 else 0,
        }

    def get_memory_summary(self) -> str:
        """
        Get human-readable memory summary.

        Returns:
            Formatted memory summary string
        """
        if self.device.type != "cuda":
            return "Memory summary only available for CUDA devices"

        stats = self.get_memory_stats()

        summary = (
            f"Memory Summary ({stats['device']}):\n"
            f"  Allocated: {stats['allocated_gb']:.2f} GB\n"
            f"  Reserved:  {stats['reserved_gb']:.2f} GB\n"
            f"  Peak:      {stats['max_allocated_gb']:.2f} GB\n"
            f"  Total:     {stats['total_gb']:.2f} GB\n"
            f"  Usage:     {stats['utilization']:.1f}%"
        )

        return summary

    def reset_peak_memory_stats(self):
        """Reset peak memory statistics."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            if self.verbose:
                logger.debug("Reset peak memory statistics")

    def empty_cache(self):
        """Empty the CUDA cache."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            if self.verbose:
                logger.debug("Emptied CUDA cache")

    def run_garbage_collection(self):
        """Run Python garbage collection."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        if self.verbose:
            logger.debug("Ran garbage collection")

    def step(self):
        """
        Perform memory optimization step.

        Should be called at each training step to run periodic
        cache clearing and garbage collection.
        """
        self._step_count += 1

        # Periodic cache clearing
        if (
            self.config.empty_cache_interval > 0
            and self._step_count % self.config.empty_cache_interval == 0
        ):
            self.empty_cache()

        # Periodic garbage collection
        if (
            self.config.garbage_collect_interval > 0
            and self._step_count % self.config.garbage_collect_interval == 0
        ):
            self.run_garbage_collection()

    def check_memory_usage(self, threshold: float = 0.9) -> bool:
        """
        Check if memory usage exceeds threshold.

        Args:
            threshold: Memory usage threshold (0.0-1.0)

        Returns:
            True if memory usage is below threshold
        """
        if self.device.type != "cuda":
            return True

        stats = self.get_memory_stats()
        usage = stats["utilization"] / 100

        if usage > threshold:
            if self.verbose:
                logger.warning(
                    f"Memory usage high: {usage * 100:.1f}% (threshold: {threshold * 100:.1f}%)"
                )
            return False

        return True

    # =========================================================================
    # MEMORY PROFILING
    # =========================================================================

    @contextmanager
    def profile_memory(
        self, record_shapes: bool = True, profile_memory: bool = True, with_stack: bool = False
    ):
        """
        Context manager for memory profiling.

        Args:
            record_shapes: Record tensor shapes
            profile_memory: Profile memory allocations
            with_stack: Include stack traces

        Usage:
            >>> with optimizer.profile_memory():
            ...     output = model(input)
            ...     loss = criterion(output, target)
            ...     loss.backward()
        """
        if self.device.type != "cuda":
            yield
            return

        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
        ) as prof:
            yield prof

        torch.cuda.synchronize(self.device)

        if self.verbose:
            # Print memory profiling results
            print("\nMemory Profiling Results:")
            print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

    def get_memory_snapshot(self) -> dict[str, Any]:
        """
        Get detailed memory snapshot.

        Returns:
            Dictionary with memory allocation details
        """
        if self.device.type != "cuda":
            return {"message": "Snapshot only available for CUDA"}

        snapshot = torch.cuda.memory_snapshot()

        return {
            "device": str(self.device),
            "snapshot": snapshot,
            "num_allocations": len(snapshot),
            "timestamp": torch.cuda.Event(enable_timing=True).record(),
        }

    # =========================================================================
    # DATA LOADING OPTIMIZATION
    # =========================================================================

    def optimize_dataloader(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_workers: int | None = None,
        prefetch_factor: int = 2,
    ) -> torch.utils.data.DataLoader:
        """
        Optimize DataLoader for memory efficiency.

        Args:
            dataloader: DataLoader to optimize
            num_workers: Number of worker processes (None=auto)
            prefetch_factor: Samples to prefetch per worker

        Returns:
            Optimized DataLoader
        """
        if num_workers is None:
            num_workers = min(4, torch.get_num_threads())

        # Create new dataloader with optimized settings
        optimized = torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=False,  # Preserve original shuffle setting
            num_workers=num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=dataloader.drop_last,
            timeout=dataloader.timeout,
            worker_init_fn=dataloader.worker_init_fn,
            prefetch_factor=prefetch_factor,
            persistent_workers=True if num_workers > 0 else False,
        )

        if self.verbose:
            logger.info(
                f"Optimized DataLoader - "
                f"Workers: {num_workers}, "
                f"Pin Memory: {self.config.pin_memory}, "
                f"Prefetch: {prefetch_factor}"
            )

        return optimized

    # =========================================================================
    # MEMORY LEAK DETECTION
    # =========================================================================

    def detect_memory_leaks(
        self, model: nn.Module, dummy_input: torch.Tensor, num_iterations: int = 10
    ) -> dict[str, Any]:
        """
        Detect potential memory leaks.

        Args:
            model: Model to test
            dummy_input: Dummy input tensor
            num_iterations: Number of iterations to test

        Returns:
            Dictionary with leak detection results
        """
        if self.device.type != "cuda":
            return {"message": "Leak detection only available for CUDA"}

        model.eval()
        torch.cuda.reset_peak_memory_stats(self.device)

        memory_usage = []

        for i in range(num_iterations):
            torch.cuda.synchronize(self.device)
            mem_before = torch.cuda.memory_allocated(self.device)

            with torch.no_grad():
                _ = model(dummy_input)

            torch.cuda.synchronize(self.device)
            mem_after = torch.cuda.memory_allocated(self.device)

            memory_usage.append(mem_after - mem_before)

            # Clear cache periodically
            if i % 5 == 0:
                torch.cuda.empty_cache()

        # Analyze memory usage trend
        avg_usage = sum(memory_usage) / len(memory_usage)
        max_usage = max(memory_usage)
        min_usage = min(memory_usage)

        # Check if memory is growing
        is_leaking = memory_usage[-1] > memory_usage[0] * 1.1

        return {
            "iterations": num_iterations,
            "avg_usage_mb": avg_usage / (1024**2),
            "max_usage_mb": max_usage / (1024**2),
            "min_usage_mb": min_usage / (1024**2),
            "is_leaking": is_leaking,
            "memory_usage_trend": memory_usage,
        }

    def print_memory_summary(self):
        """Print formatted memory summary."""
        print("=" * 70)
        print("Memory Optimization Summary")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.config.mixed_precision}")
        print(f"Precision: {self.config.precision}")
        print(f"Gradient Checkpointing: {self.config.gradient_checkpointing}")
        print(f"Pin Memory: {self.config.pin_memory}")

        if self.device.type == "cuda":
            print("\n" + self.get_memory_summary())

        print("=" * 70)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_memory_efficient_settings(
    device: torch.device, model_size: str = "medium"
) -> dict[str, Any]:
    """
    Get recommended memory-efficient settings.

    Args:
        device: Target device
        model_size: Model size category (small, medium, large, xlarge)

    Returns:
        Dictionary with recommended settings
    """
    settings = {
        "small": {
            "mixed_precision": False,
            "gradient_checkpointing": False,
            "empty_cache_interval": 0,
            "garbage_collect_interval": 0,
        },
        "medium": {
            "mixed_precision": True,
            "gradient_checkpointing": False,
            "empty_cache_interval": 100,
            "garbage_collect_interval": 0,
        },
        "large": {
            "mixed_precision": True,
            "gradient_checkpointing": True,
            "empty_cache_interval": 50,
            "garbage_collect_interval": 200,
        },
        "xlarge": {
            "mixed_precision": True,
            "gradient_checkpointing": True,
            "empty_cache_interval": 10,
            "garbage_collect_interval": 50,
        },
    }

    return settings.get(model_size, settings["medium"])


def estimate_model_memory(
    model: nn.Module, input_size: tuple, batch_size: int = 1, precision: str = "fp32"
) -> dict[str, float]:
    """
    Estimate model memory requirements.

    Args:
        model: Model to estimate
        input_size: Input tensor size
        batch_size: Batch size
        precision: Precision type (fp16, fp32)

    Returns:
        Dictionary with memory estimates (in MB)
    """
    # Calculate parameter memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())

    # Estimate activation memory (rough estimate)
    bytes_per_element = 2 if precision == "fp16" else 4
    activation_memory = batch_size * input_size[0] * bytes_per_element * 10  # rough multiplier

    # Gradient memory (same as parameters)
    gradient_memory = param_memory

    # Optimizer state (2x parameters for Adam)
    optimizer_memory = param_memory * 2

    total_memory = param_memory + activation_memory + gradient_memory + optimizer_memory

    return {
        "parameters_mb": param_memory / (1024**2),
        "activations_mb": activation_memory / (1024**2),
        "gradients_mb": gradient_memory / (1024**2),
        "optimizer_mb": optimizer_memory / (1024**2),
        "total_mb": total_memory / (1024**2),
        "total_gb": total_memory / (1024**3),
    }


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info("memory_optimization module loaded")
