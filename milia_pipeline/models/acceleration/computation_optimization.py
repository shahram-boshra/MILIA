"""
Computation Optimization

Comprehensive computation optimization strategies for faster training and inference.
Supports torch.compile, cuDNN optimization, graph optimization, and performance profiling.

Features:
- torch.compile support (PyTorch 2.0+)
- cuDNN benchmark mode and deterministic operations
- Graph optimization and fusion
- JIT (Just-In-Time) compilation
- Operator fusion strategies
- Performance profiling and benchmarking
- TensorCore optimization
- Channel-last memory format

Pydantic V2 Migration (Phase 10):
    - Migrated ComputationConfig from @dataclass to Pydantic BaseModel (mutable)
    - Uses model_dump() for to_dict() method (backward compatible)
    - NON-BREAKING: Same constructor API and attribute access
    - Follows established pattern from memory_optimization.py (Phase 9)

Author: milia Team
Version: 1.1.0
"""

import logging
import time
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel

# Import exceptions with fallback
try:
    from milia_pipeline.exceptions import HardwareError, ModelError, OptimizationError
except ImportError:

    class ModelError(Exception):
        """Base exception for model-related errors."""

        pass

    class HardwareError(ModelError):
        """Exception raised for hardware-related errors."""

        pass

    class OptimizationError(HardwareError):
        """Exception raised for optimization-related errors."""

        pass


logger = logging.getLogger(__name__)


# =============================================================================
# COMPUTATION OPTIMIZATION CONFIGURATION
# =============================================================================


class ComputationConfig(BaseModel):
    """
    Configuration for computation optimization.

    Pydantic V2 Migration (Phase 10):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses model_dump() for backward-compatible to_dict() method
        - NON-BREAKING: Same constructor API and attribute access
        - Follows established pattern from memory_optimization.py (Phase 9)

    Attributes:
        compile_model: Enable torch.compile (PyTorch 2.0+)
        compile_mode: Compilation mode (default, reduce-overhead, max-autotune)
        compile_dynamic: Enable dynamic shapes in compilation
        cudnn_benchmark: Enable cuDNN benchmark mode
        cudnn_deterministic: Enable deterministic operations
        use_tf32: Enable TensorFloat-32 on Ampere+ GPUs
        channels_last: Use channels-last memory format
        fusion_strategy: Kernel fusion strategy (none, default, aggressive)
        jit_compile: Enable JIT compilation
        operator_fusion: Enable operator fusion
    """

    compile_model: bool = False
    compile_mode: str = "default"  # default, reduce-overhead, max-autotune
    compile_dynamic: bool = False
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    use_tf32: bool = True
    channels_last: bool = False
    fusion_strategy: str = "default"  # none, default, aggressive
    jit_compile: bool = False
    operator_fusion: bool = True

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Backward compatible method wrapping Pydantic V2's model_dump().
        """
        return self.model_dump()


# =============================================================================
# COMPUTATION OPTIMIZER
# =============================================================================


class ComputationOptimizer:
    """
    Manager for computation optimization strategies.

    Provides various computation optimization techniques including torch.compile,
    cuDNN optimization, and performance profiling.

    Usage:
        >>> # Basic optimization
        >>> optimizer = ComputationOptimizer(
        ...     compile_model=True,
        ...     cudnn_benchmark=True
        ... )
        >>> model = optimizer.optimize_model(model)
        >>>
        >>> # Advanced: torch.compile with max-autotune
        >>> optimizer = ComputationOptimizer(
        ...     compile_model=True,
        ...     compile_mode="max-autotune"
        ... )
        >>> model = optimizer.optimize_model(model)
        >>>
        >>> # Profile performance
        >>> with optimizer.profile_performance():
        ...     output = model(input)
    """

    def __init__(
        self,
        compile_model: bool = False,
        compile_mode: str = "default",
        compile_dynamic: bool = False,
        cudnn_benchmark: bool = True,
        cudnn_deterministic: bool = False,
        use_tf32: bool = True,
        channels_last: bool = False,
        fusion_strategy: str = "default",
        jit_compile: bool = False,
        operator_fusion: bool = True,
        device: torch.device | None = None,
        verbose: bool = True,
    ):
        """
        Initialize computation optimizer.

        Args:
            compile_model: Enable torch.compile
            compile_mode: Compilation mode
            compile_dynamic: Enable dynamic shapes
            cudnn_benchmark: Enable cuDNN benchmark
            cudnn_deterministic: Enable deterministic ops
            use_tf32: Enable TF32 on Ampere GPUs
            channels_last: Use channels-last format
            fusion_strategy: Kernel fusion strategy
            jit_compile: Enable JIT compilation
            operator_fusion: Enable operator fusion
            device: Target device
            verbose: Whether to log information
        """
        self.verbose = verbose
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._compiled_models = {}

        # Create config
        self.config = ComputationConfig(
            compile_model=compile_model,
            compile_mode=compile_mode,
            compile_dynamic=compile_dynamic,
            cudnn_benchmark=cudnn_benchmark,
            cudnn_deterministic=cudnn_deterministic,
            use_tf32=use_tf32,
            channels_last=channels_last,
            fusion_strategy=fusion_strategy,
            jit_compile=jit_compile,
            operator_fusion=operator_fusion,
        )

        # Apply global optimizations
        self._apply_global_optimizations()

        if self.verbose:
            logger.info(
                f"ComputationOptimizer initialized - "
                f"Compile: {self.config.compile_model}, "
                f"cuDNN Benchmark: {self.config.cudnn_benchmark}, "
                f"TF32: {self.config.use_tf32}"
            )

    def _apply_global_optimizations(self):
        """Apply global PyTorch optimization settings."""
        # cuDNN optimizations
        if self.device.type == "cuda":
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = self.config.cudnn_benchmark
                torch.backends.cudnn.deterministic = self.config.cudnn_deterministic

                if self.verbose:
                    logger.info(
                        f"cuDNN settings - "
                        f"Benchmark: {self.config.cudnn_benchmark}, "
                        f"Deterministic: {self.config.cudnn_deterministic}"
                    )

            # TF32 optimization (Ampere+ GPUs)
            if hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = self.config.use_tf32
                torch.backends.cudnn.allow_tf32 = self.config.use_tf32

                if self.verbose:
                    logger.info(f"TF32 enabled: {self.config.use_tf32}")

        # Operator fusion
        if self.config.operator_fusion:
            torch._C._jit_set_profiling_executor(True)
            torch._C._jit_set_profiling_mode(True)
            if self.verbose:
                logger.info("Operator fusion enabled")

    # =========================================================================
    # MODEL COMPILATION (torch.compile)
    # =========================================================================

    def compile_model(
        self,
        model: nn.Module,
        mode: str | None = None,
        dynamic: bool | None = None,
        fullgraph: bool = False,
        backend: str = "inductor",
    ) -> nn.Module:
        """
        Compile model using torch.compile (PyTorch 2.0+).

        Args:
            model: Model to compile
            mode: Compilation mode (default, reduce-overhead, max-autotune)
            dynamic: Enable dynamic shapes
            fullgraph: Require full graph compilation
            backend: Compilation backend (inductor, aot_eager, cudagraphs)

        Returns:
            Compiled model

        Raises:
            OptimizationError: If compilation fails
        """
        if not self.config.compile_model:
            return model

        # Check PyTorch version
        if not hasattr(torch, "compile"):
            warnings.warn("torch.compile requires PyTorch 2.0+. Skipping compilation.")
            return model

        mode = mode or self.config.compile_mode
        dynamic = dynamic if dynamic is not None else self.config.compile_dynamic

        try:
            compiled_model = torch.compile(
                model, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend
            )

            if self.verbose:
                logger.info(
                    f"Model compiled - Mode: {mode}, Dynamic: {dynamic}, Backend: {backend}"
                )

            # Cache compiled model
            model_id = id(model)
            self._compiled_models[model_id] = compiled_model

            return compiled_model

        except Exception as e:
            raise OptimizationError(f"Failed to compile model: {e}") from e

    # =========================================================================
    # JIT COMPILATION
    # =========================================================================

    def jit_script_model(
        self, model: nn.Module, example_inputs: tuple | None = None
    ) -> torch.jit.ScriptModule:
        """
        JIT script model for optimization.

        Args:
            model: Model to script
            example_inputs: Example inputs for tracing (optional)

        Returns:
            JIT scripted model
        """
        if not self.config.jit_compile:
            return model

        try:
            if example_inputs is not None:
                # Trace model
                scripted = torch.jit.trace(model, example_inputs)
                if self.verbose:
                    logger.info("Model JIT traced")
            else:
                # Script model
                scripted = torch.jit.script(model)
                if self.verbose:
                    logger.info("Model JIT scripted")

            # Optimize
            scripted = torch.jit.optimize_for_inference(scripted)

            return scripted

        except Exception as e:
            logger.warning(f"JIT compilation failed: {e}. Using original model.")
            return model

    def jit_freeze_model(self, model: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
        """
        Freeze JIT model for inference optimization.

        Args:
            model: JIT scripted model

        Returns:
            Frozen model
        """
        try:
            frozen = torch.jit.freeze(model)
            if self.verbose:
                logger.info("Model frozen for inference")
            return frozen
        except Exception as e:
            logger.warning(f"Model freezing failed: {e}")
            return model

    # =========================================================================
    # MEMORY FORMAT OPTIMIZATION
    # =========================================================================

    def convert_to_channels_last(self, model: nn.Module) -> nn.Module:
        """
        Convert model to channels-last memory format for performance.

        Beneficial for convolution-heavy models on modern GPUs.

        Args:
            model: Model to convert

        Returns:
            Converted model
        """
        if not self.config.channels_last:
            return model

        try:
            model = model.to(memory_format=torch.channels_last)
            if self.verbose:
                logger.info("Converted model to channels-last format")
        except Exception as e:
            logger.warning(f"Channels-last conversion failed: {e}")

        return model

    # =========================================================================
    # MAIN OPTIMIZATION ENTRY POINT
    # =========================================================================

    def optimize_model(self, model: nn.Module, example_inputs: tuple | None = None) -> nn.Module:
        """
        Apply all enabled optimizations to model.

        Args:
            model: Model to optimize
            example_inputs: Example inputs for tracing/profiling

        Returns:
            Optimized model
        """
        if self.verbose:
            logger.info("Starting model optimization...")

        # Move to device
        model = model.to(self.device)

        # Channels-last format
        if self.config.channels_last:
            model = self.convert_to_channels_last(model)

        # JIT compilation
        if self.config.jit_compile:
            model = self.jit_script_model(model, example_inputs)
            if isinstance(model, torch.jit.ScriptModule):
                model = self.jit_freeze_model(model)

        # torch.compile (PyTorch 2.0+)
        if self.config.compile_model and not isinstance(model, torch.jit.ScriptModule):
            model = self.compile_model(model)

        if self.verbose:
            logger.info("Model optimization complete")

        return model

    # =========================================================================
    # PERFORMANCE PROFILING
    # =========================================================================

    @contextmanager
    def profile_performance(
        self, record_shapes: bool = True, with_stack: bool = False, with_flops: bool = False
    ):
        """
        Context manager for performance profiling.

        Args:
            record_shapes: Record tensor shapes
            with_stack: Include stack traces
            with_flops: Profile FLOPs

        Usage:
            >>> with optimizer.profile_performance():
            ...     output = model(input)
        """
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.profiler.profile(
            activities=activities,
            record_shapes=record_shapes,
            with_stack=with_stack,
            with_flops=with_flops,
        ) as prof:
            yield prof

        if self.verbose:
            # Print profiling results
            print("\nPerformance Profiling Results:")
            print(
                prof.key_averages().table(
                    sort_by="cuda_time_total" if self.device.type == "cuda" else "cpu_time_total",
                    row_limit=10,
                )
            )

    def benchmark_model(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> dict[str, float]:
        """
        Benchmark model performance.

        Args:
            model: Model to benchmark
            input_data: Input tensor
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations

        Returns:
            Dictionary with benchmark results
        """
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(input_data)

        # Synchronize
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        # Benchmark
        timings = []

        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()

                _ = model(input_data)

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)

                end = time.perf_counter()
                timings.append(end - start)

        # Calculate statistics
        avg_time = sum(timings) / len(timings)
        min_time = min(timings)
        max_time = max(timings)
        throughput = 1.0 / avg_time if avg_time > 0 else 0

        results = {
            "iterations": num_iterations,
            "avg_time_ms": avg_time * 1000,
            "min_time_ms": min_time * 1000,
            "max_time_ms": max_time * 1000,
            "throughput_fps": throughput,
            "std_time_ms": (sum((t - avg_time) ** 2 for t in timings) / len(timings)) ** 0.5 * 1000,
        }

        if self.verbose:
            logger.info(
                f"Benchmark results - "
                f"Avg: {results['avg_time_ms']:.2f}ms, "
                f"Throughput: {results['throughput_fps']:.1f} samples/sec"
            )

        return results

    def compare_optimizations(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        optimization_configs: list[dict[str, Any]],
        num_iterations: int = 100,
    ) -> dict[str, dict[str, float]]:
        """
        Compare different optimization configurations.

        Args:
            model: Model to benchmark
            input_data: Input tensor
            optimization_configs: List of optimization configs to compare
            num_iterations: Number of iterations per config

        Returns:
            Dictionary mapping config names to benchmark results
        """
        results = {}

        for i, config in enumerate(optimization_configs):
            config_name = config.pop("name", f"config_{i}")

            # Create optimizer with config
            optimizer = ComputationOptimizer(**config, verbose=False)

            # Optimize model
            optimized_model = optimizer.optimize_model(
                model.clone() if hasattr(model, "clone") else model
            )

            # Benchmark
            bench_results = self.benchmark_model(
                optimized_model, input_data, num_iterations=num_iterations, warmup_iterations=10
            )

            results[config_name] = bench_results

            if self.verbose:
                logger.info(f"{config_name}: {bench_results['avg_time_ms']:.2f}ms")

        return results

    # =========================================================================
    # KERNEL FUSION
    # =========================================================================

    def enable_fusion(self, aggressive: bool = False):
        """
        Enable kernel fusion optimizations.

        Args:
            aggressive: Use aggressive fusion strategy
        """
        if aggressive:
            torch._C._jit_set_texpr_fuser_enabled(True)
            torch._C._jit_set_nvfuser_enabled(True)
            if self.verbose:
                logger.info("Enabled aggressive kernel fusion")
        else:
            torch._C._jit_set_profiling_executor(True)
            if self.verbose:
                logger.info("Enabled default kernel fusion")

    def disable_fusion(self):
        """Disable kernel fusion optimizations."""
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(False)
        torch._C._jit_set_profiling_executor(False)
        if self.verbose:
            logger.info("Disabled kernel fusion")

    # =========================================================================
    # GRAPH OPTIMIZATION
    # =========================================================================

    def optimize_graph(
        self, model: torch.jit.ScriptModule, optimization_level: int = 1
    ) -> torch.jit.ScriptModule:
        """
        Optimize JIT graph.

        Args:
            model: JIT scripted model
            optimization_level: Optimization level (0-3)

        Returns:
            Optimized model
        """
        try:
            # Apply graph optimizations
            torch._C._jit_pass_inline(model.graph)
            torch._C._jit_pass_constant_propagation(model.graph)
            torch._C._jit_pass_peephole(model.graph, addmm_fusion_enabled=True)

            if optimization_level >= 2:
                torch._C._jit_pass_fuse(model.graph)

            if optimization_level >= 3:
                torch._C._jit_pass_remove_mutation(model.graph)

            if self.verbose:
                logger.info(f"Applied graph optimizations (level {optimization_level})")

        except Exception as e:
            logger.warning(f"Graph optimization failed: {e}")

        return model

    def print_optimization_summary(self):
        """Print formatted optimization summary."""
        print("=" * 70)
        print("Computation Optimization Summary")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"torch.compile: {self.config.compile_model}")
        if self.config.compile_model:
            print(f"  Mode: {self.config.compile_mode}")
            print(f"  Dynamic: {self.config.compile_dynamic}")

        if self.device.type == "cuda":
            print(f"cuDNN Benchmark: {self.config.cudnn_benchmark}")
            print(f"cuDNN Deterministic: {self.config.cudnn_deterministic}")
            print(f"TF32: {self.config.use_tf32}")

        print(f"Channels Last: {self.config.channels_last}")
        print(f"JIT Compile: {self.config.jit_compile}")
        print(f"Operator Fusion: {self.config.operator_fusion}")
        print(f"Fusion Strategy: {self.config.fusion_strategy}")
        print("=" * 70)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_optimal_settings(device: torch.device, task_type: str = "training") -> dict[str, Any]:
    """
    Get optimal computation settings for task.

    Args:
        device: Target device
        task_type: Task type (training, inference)

    Returns:
        Dictionary with optimal settings
    """
    if task_type == "training":
        return {
            "compile_model": True,
            "compile_mode": "default",
            "cudnn_benchmark": True,
            "cudnn_deterministic": False,
            "use_tf32": True,
            "channels_last": False,
            "jit_compile": False,
            "operator_fusion": True,
        }
    else:  # inference
        return {
            "compile_model": True,
            "compile_mode": "max-autotune",
            "cudnn_benchmark": True,
            "cudnn_deterministic": False,
            "use_tf32": True,
            "channels_last": True,
            "jit_compile": True,
            "operator_fusion": True,
        }


def auto_optimize_model(
    model: nn.Module, task_type: str = "training", device: torch.device | None = None
) -> nn.Module:
    """
    Auto-optimize model with best settings.

    Args:
        model: Model to optimize
        task_type: Task type (training, inference)
        device: Target device

    Returns:
        Optimized model
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = get_optimal_settings(device, task_type)

    optimizer = ComputationOptimizer(**settings, device=device)
    return optimizer.optimize_model(model)


# =============================================================================
# DECORATORS
# =============================================================================


def optimize_inference(func: Callable) -> Callable:
    """
    Decorator to optimize function for inference.

    Usage:
        >>> @optimize_inference
        ... def predict(model, input_data):
        ...     return model(input_data)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad(), torch.inference_mode():
            return func(*args, **kwargs)

    return wrapper


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info("computation_optimization module loaded")
