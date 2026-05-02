"""
Distributed Training Strategies

Comprehensive distributed training support for multi-GPU and multi-node setups.
Supports DataParallel (DP), DistributedDataParallel (DDP), FSDP, DeepSpeed, and Horovod.

Features:
- Multiple distributed backends (gloo, nccl, mpi)
- Data parallel training (DP, DDP)
- Fully Sharded Data Parallel (FSDP)
- DeepSpeed integration (ZeRO stages 1-3)
- Horovod support
- Gradient synchronization strategies
- Communication optimization
- Fault tolerance and checkpointing

Pydantic V2 Migration (Phase 8):
    - Migrated DistributedConfig from @dataclass to Pydantic BaseModel (mutable)
    - Uses model_dump(mode='json') for to_dict() method (automatic enum serialization)
    - NON-BREAKING: Same constructor API and attribute access
    - Follows established pattern from device_manager.py (Phase 7)

Author: milia Team
Version: 1.1.0
"""

import importlib.util
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from pydantic import BaseModel
from torch.nn.parallel import DataParallel, DistributedDataParallel

# Import exceptions with fallback
try:
    from milia_pipeline.exceptions import DistributedError, HardwareError, ModelError
except ImportError:

    class ModelError(Exception):
        """Base exception for model-related errors."""

        pass

    class HardwareError(ModelError):
        """Exception raised for hardware-related errors."""

        pass

    class DistributedError(HardwareError):
        """Exception raised for distributed training errors."""

        pass


# Import the canonical PyTorch>=2.6 safe-load helper from trainer.py.
#
# `_safe_torch_load` invokes torch.load() with weights_only=True first (the
# secure, future-aligned path) and falls back to weights_only=False with a
# logged warning only on diagnostic-discriminated weights_only blocks. This
# eliminates the FutureWarning emitted by torch>=2.4 and prepares this
# module for the torch>=2.6 default flip without weakening security for
# legitimate MILIA-authored checkpoints.
#
# The same retry pattern is implemented in CheckpointManager.load() in
# milia_pipeline/models/post_training/checkpoint/checkpoint_manager.py.
# Reusing the helper here keeps the policy centralized in one place.
#
# Fallback rationale: if trainer.py is unavailable (e.g., partial install
# where the training subpackage was excluded), this module degrades to
# legacy torch.load() behavior rather than failing at import time. That
# preserves the historical contract for callers that never exercised
# load_checkpoint().
try:
    from milia_pipeline.models.training.trainer import _safe_torch_load
except ImportError:

    def _safe_torch_load(filepath, map_location=None):
        """Fallback when trainer module is unavailable; matches legacy semantics."""
        return torch.load(filepath, map_location=map_location)


logger = logging.getLogger(__name__)


# =============================================================================
# DISTRIBUTED STRATEGY TYPES
# =============================================================================


class DistributedStrategy(Enum):
    """Enumeration of distributed training strategies."""

    NONE = "none"  # Single device training
    DP = "dp"  # DataParallel (single-node, multi-GPU)
    DDP = "ddp"  # DistributedDataParallel (multi-node, multi-GPU)
    FSDP = "fsdp"  # Fully Sharded Data Parallel
    DEEPSPEED = "deepspeed"  # DeepSpeed (ZeRO)
    HOROVOD = "horovod"  # Horovod


class DistributedBackend(Enum):
    """Distributed communication backends."""

    GLOO = "gloo"  # CPU-based, works on all platforms
    NCCL = "nccl"  # NVIDIA GPUs, fastest for GPU
    MPI = "mpi"  # Message Passing Interface
    AUTO = "auto"  # Auto-detect best backend


# =============================================================================
# DISTRIBUTED CONFIGURATION
# =============================================================================


class DistributedConfig(BaseModel):
    """
    Configuration for distributed training.

    Pydantic V2 Migration (Phase 8):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses model_dump(mode='json') for automatic enum value serialization
        - NON-BREAKING: Same constructor API and attribute access
        - Follows established pattern from device_manager.py (Phase 7)

    Attributes:
        strategy: Distributed strategy to use
        backend: Communication backend
        world_size: Total number of processes
        rank: Current process rank
        local_rank: Local process rank (GPU ID on node)
        master_addr: Master node address
        master_port: Master node port
        find_unused_parameters: Find unused parameters (DDP)
        gradient_as_bucket_view: Use gradient buckets (DDP)
        static_graph: Enable static graph optimization (DDP)
        cpu_offload: Enable CPU offloading (FSDP)
        mixed_precision: Enable mixed precision training
    """

    strategy: DistributedStrategy = DistributedStrategy.NONE
    backend: DistributedBackend = DistributedBackend.AUTO
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    static_graph: bool = False
    cpu_offload: bool = False
    mixed_precision: bool = False

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Backward compatible method wrapping Pydantic V2's model_dump().
        Uses mode='json' for automatic enum value serialization.
        """
        return self.model_dump(mode="json")


# =============================================================================
# DISTRIBUTED MANAGER
# =============================================================================


class DistributedManager:
    """
    Manager for distributed training setup and coordination.

    Handles initialization, model wrapping, and cleanup for various
    distributed training strategies.

    Usage:
        >>> # DataParallel (simple multi-GPU)
        >>> manager = DistributedManager(strategy="dp")
        >>> model = manager.wrap_model(model)
        >>>
        >>> # DistributedDataParallel (recommended)
        >>> manager = DistributedManager(strategy="ddp")
        >>> manager.setup()
        >>> model = manager.wrap_model(model)
        >>> # Training...
        >>> manager.cleanup()
        >>>
        >>> # FSDP (memory efficient)
        >>> manager = DistributedManager(strategy="fsdp", cpu_offload=True)
        >>> manager.setup()
        >>> model = manager.wrap_model(model)
    """

    def __init__(
        self,
        strategy: str | DistributedStrategy = "none",
        backend: str | DistributedBackend = "auto",
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = True,
        static_graph: bool = False,
        cpu_offload: bool = False,
        mixed_precision: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize distributed manager.

        Args:
            strategy: Distributed strategy (none, dp, ddp, fsdp, deepspeed, horovod)
            backend: Communication backend (auto, nccl, gloo, mpi)
            find_unused_parameters: Find unused parameters in backward (DDP)
            gradient_as_bucket_view: Use gradient buckets for efficiency (DDP)
            static_graph: Enable static graph optimization (DDP)
            cpu_offload: Enable CPU offloading to reduce GPU memory (FSDP)
            mixed_precision: Enable mixed precision training
            verbose: Whether to log information
        """
        # Convert string to enum
        if isinstance(strategy, str):
            strategy = DistributedStrategy(strategy.lower())
        if isinstance(backend, str):
            backend = DistributedBackend(backend.lower())

        self.verbose = verbose
        self._is_initialized = False
        self._original_model = None

        # Create config
        self.config = DistributedConfig(
            strategy=strategy,
            backend=backend,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
            cpu_offload=cpu_offload,
            mixed_precision=mixed_precision,
        )

        # Load environment variables (for DDP/FSDP)
        self._load_env_variables()

        if self.verbose:
            logger.info(
                f"DistributedManager initialized - "
                f"Strategy: {self.config.strategy.value}, "
                f"Backend: {self.config.backend.value}"
            )

    def _load_env_variables(self):
        """Load distributed training environment variables."""
        # World size and rank
        self.config.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.config.rank = int(os.environ.get("RANK", 0))
        self.config.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Master address and port
        self.config.master_addr = os.environ.get("MASTER_ADDR", "localhost")
        self.config.master_port = os.environ.get("MASTER_PORT", "12355")

    def _get_backend(self) -> str:
        """
        Determine the best communication backend.

        Returns:
            Backend name (gloo, nccl, mpi)
        """
        if self.config.backend != DistributedBackend.AUTO:
            return self.config.backend.value

        # Auto-detection
        if torch.cuda.is_available():
            return "nccl"  # NCCL is fastest for CUDA
        else:
            return "gloo"  # Gloo works on CPU

    def setup(self):
        """
        Initialize distributed training environment.

        Must be called before wrapping model for DDP/FSDP/DeepSpeed/Horovod.

        Raises:
            DistributedError: If initialization fails
        """
        if self._is_initialized:
            logger.warning("Distributed environment already initialized")
            return

        strategy = self.config.strategy

        if strategy == DistributedStrategy.NONE:
            # No distributed training
            self._is_initialized = True
            return

        elif strategy == DistributedStrategy.DP:
            # DataParallel doesn't require init
            self._is_initialized = True
            if self.verbose:
                logger.info("DataParallel mode - no initialization needed")
            return

        elif strategy in [DistributedStrategy.DDP, DistributedStrategy.FSDP]:
            # Initialize process group for DDP/FSDP
            if not dist.is_initialized():
                backend = self._get_backend()

                # Set environment variables
                os.environ["MASTER_ADDR"] = self.config.master_addr
                os.environ["MASTER_PORT"] = self.config.master_port

                try:
                    dist.init_process_group(
                        backend=backend, rank=self.config.rank, world_size=self.config.world_size
                    )

                    if self.verbose and self.is_main_process():
                        logger.info(
                            f"Initialized {strategy.value.upper()} - "
                            f"Backend: {backend}, "
                            f"World Size: {self.config.world_size}, "
                            f"Rank: {self.config.rank}"
                        )

                    self._is_initialized = True

                except Exception as e:
                    raise DistributedError(
                        f"Failed to initialize {strategy.value.upper()}: {e}"
                    ) from e
            else:
                self._is_initialized = True
                if self.verbose:
                    logger.info("Process group already initialized")

        elif strategy == DistributedStrategy.DEEPSPEED:
            # DeepSpeed initialization happens in wrap_model
            self._is_initialized = True
            if self.verbose:
                logger.info("DeepSpeed mode - initialization deferred to wrap_model")

        elif strategy == DistributedStrategy.HOROVOD:
            # Initialize Horovod
            try:
                import horovod.torch as hvd

                hvd.init()

                self.config.rank = hvd.rank()
                self.config.world_size = hvd.size()
                self.config.local_rank = hvd.local_rank()

                if self.verbose and self.is_main_process():
                    logger.info(
                        f"Initialized Horovod - "
                        f"World Size: {self.config.world_size}, "
                        f"Rank: {self.config.rank}"
                    )

                self._is_initialized = True

            except ImportError:
                raise DistributedError(
                    "Horovod not installed. Install with: pip install horovod"
                ) from None
            except Exception as e:
                raise DistributedError(f"Failed to initialize Horovod: {e}") from e

    def wrap_model(self, model: nn.Module, device_ids: list[int] | None = None) -> nn.Module:
        """
        Wrap model for distributed training.

        Args:
            model: PyTorch model to wrap
            device_ids: List of device IDs (for DP/DDP)

        Returns:
            Wrapped model

        Raises:
            DistributedError: If wrapping fails
        """
        if not self._is_initialized:
            raise DistributedError(
                "Must call setup() before wrapping model. Call manager.setup() first."
            )

        self._original_model = model
        strategy = self.config.strategy

        if strategy == DistributedStrategy.NONE:
            # No wrapping needed
            return model

        elif strategy == DistributedStrategy.DP:
            # DataParallel (simple multi-GPU)
            if not torch.cuda.is_available():
                raise DistributedError("DataParallel requires CUDA. No CUDA devices found.")

            if device_ids is None:
                device_ids = list(range(torch.cuda.device_count()))

            wrapped_model = DataParallel(model, device_ids=device_ids)

            if self.verbose:
                logger.info(f"Wrapped model with DataParallel - Devices: {device_ids}")

            return wrapped_model

        elif strategy == DistributedStrategy.DDP:
            # DistributedDataParallel
            device = torch.device("cuda", self.config.local_rank)
            model = model.to(device)

            wrapped_model = DistributedDataParallel(
                model,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
                find_unused_parameters=self.config.find_unused_parameters,
                gradient_as_bucket_view=self.config.gradient_as_bucket_view,
                static_graph=self.config.static_graph,
            )

            if self.verbose and self.is_main_process():
                logger.info(f"Wrapped model with DDP - Device: cuda:{self.config.local_rank}")

            return wrapped_model

        elif strategy == DistributedStrategy.FSDP:
            # Fully Sharded Data Parallel
            try:
                from torch.distributed.fsdp import CPUOffload, MixedPrecision
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

                # Auto-wrap policy for large models
                auto_wrap_policy = size_based_auto_wrap_policy

                # CPU offload configuration
                cpu_offload = CPUOffload(offload_params=True) if self.config.cpu_offload else None

                # Mixed precision configuration
                mixed_precision = None
                if self.config.mixed_precision:
                    mixed_precision = MixedPrecision(
                        param_dtype=torch.float16,
                        reduce_dtype=torch.float16,
                        buffer_dtype=torch.float16,
                    )

                wrapped_model = FSDP(
                    model,
                    auto_wrap_policy=auto_wrap_policy,
                    cpu_offload=cpu_offload,
                    mixed_precision=mixed_precision,
                    device_id=torch.cuda.current_device(),
                )

                if self.verbose and self.is_main_process():
                    logger.info(
                        f"Wrapped model with FSDP - "
                        f"CPU Offload: {self.config.cpu_offload}, "
                        f"Mixed Precision: {self.config.mixed_precision}"
                    )

                return wrapped_model

            except ImportError:
                raise DistributedError(
                    "FSDP requires PyTorch 1.12+. Upgrade PyTorch or use DDP instead."
                ) from None

        elif strategy == DistributedStrategy.DEEPSPEED:
            # DeepSpeed (requires separate initialization)
            try:
                _deepspeed_available = importlib.util.find_spec("deepspeed") is not None
            except ValueError:
                # find_spec raises ValueError if deepspeed is in sys.modules
                # but __spec__ is not set or is None (documented CPython behavior)
                _deepspeed_available = True

            if _deepspeed_available:
                # DeepSpeed requires config file and separate initialization
                # This is a placeholder - actual implementation needs config
                raise DistributedError(
                    "DeepSpeed requires additional configuration. "
                    "Use deepspeed.initialize() directly with config file. "
                    "See: https://www.deepspeed.ai/getting-started/"
                )
            else:
                raise DistributedError(
                    "DeepSpeed not installed. Install with: pip install deepspeed"
                )

        elif strategy == DistributedStrategy.HOROVOD:
            # Horovod doesn't wrap model, but broadcasts state
            try:
                import horovod.torch as hvd

                # Broadcast parameters from rank 0
                hvd.broadcast_parameters(model.state_dict(), root_rank=0)
                hvd.broadcast_optimizer_state(
                    model.parameters() if hasattr(model, "parameters") else [], root_rank=0
                )

                if self.verbose and self.is_main_process():
                    logger.info("Broadcasted model parameters via Horovod")

                return model

            except ImportError:
                raise DistributedError(
                    "Horovod not installed. Install with: pip install horovod"
                ) from None

        else:
            raise DistributedError(f"Unknown strategy: {strategy}")

    def cleanup(self):
        """
        Cleanup distributed training environment.

        Should be called at end of training.
        """
        if not self._is_initialized:
            return

        strategy = self.config.strategy

        if strategy in [DistributedStrategy.DDP, DistributedStrategy.FSDP]:
            if dist.is_initialized():
                dist.destroy_process_group()
                if self.verbose and self.is_main_process():
                    logger.info(f"Destroyed {strategy.value.upper()} process group")

        elif strategy == DistributedStrategy.HOROVOD:
            try:
                import horovod.torch as hvd

                hvd.shutdown()
                if self.verbose and self.is_main_process():
                    logger.info("Shutdown Horovod")
            except ImportError:
                pass

        self._is_initialized = False

    def is_main_process(self) -> bool:
        """
        Check if current process is the main process.

        Returns:
            True if main process (rank 0)
        """
        return self.config.rank == 0

    def barrier(self):
        """Synchronize all processes (wait for all to reach this point)."""
        if self.config.strategy in [DistributedStrategy.DDP, DistributedStrategy.FSDP]:
            if dist.is_initialized():
                dist.barrier()
        elif self.config.strategy == DistributedStrategy.HOROVOD:
            try:
                import horovod.torch as hvd

                hvd.allreduce(torch.tensor([0]), name="barrier")
            except ImportError:
                pass

    def get_world_size(self) -> int:
        """Get total number of processes."""
        return self.config.world_size

    def get_rank(self) -> int:
        """Get current process rank."""
        return self.config.rank

    def get_local_rank(self) -> int:
        """Get local process rank (GPU ID on current node)."""
        return self.config.local_rank

    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """
        All-reduce operation across all processes.

        Args:
            tensor: Tensor to reduce
            op: Reduction operation (sum, avg, min, max)

        Returns:
            Reduced tensor
        """
        if self.config.strategy in [DistributedStrategy.DDP, DistributedStrategy.FSDP]:
            if dist.is_initialized():
                op_map = {
                    "sum": dist.ReduceOp.SUM,
                    "avg": dist.ReduceOp.AVG
                    if hasattr(dist.ReduceOp, "AVG")
                    else dist.ReduceOp.SUM,
                    "min": dist.ReduceOp.MIN,
                    "max": dist.ReduceOp.MAX,
                }
                dist.all_reduce(tensor, op=op_map.get(op, dist.ReduceOp.SUM))
                if op == "avg" and not hasattr(dist.ReduceOp, "AVG"):
                    tensor /= self.config.world_size

        elif self.config.strategy == DistributedStrategy.HOROVOD:
            try:
                import horovod.torch as hvd

                tensor = hvd.allreduce(tensor, op=hvd.Sum if op == "sum" else hvd.Average)
            except ImportError:
                pass

        return tensor

    def all_gather(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        """
        All-gather operation across all processes.

        Args:
            tensor: Tensor to gather

        Returns:
            List of tensors from all processes
        """
        if self.config.strategy in [DistributedStrategy.DDP, DistributedStrategy.FSDP]:
            if dist.is_initialized():
                tensor_list = [torch.zeros_like(tensor) for _ in range(self.config.world_size)]
                dist.all_gather(tensor_list, tensor)
                return tensor_list

        elif self.config.strategy == DistributedStrategy.HOROVOD:
            try:
                import horovod.torch as hvd

                return hvd.allgather(tensor)
            except ImportError:
                pass

        return [tensor]

    def save_checkpoint(
        self,
        model: nn.Module,
        filepath: str | Path,
        include_optimizer: bool = False,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        """
        Save model checkpoint (only on main process).

        Args:
            model: Model to save
            filepath: Path to save checkpoint
            include_optimizer: Whether to include optimizer state
            optimizer: Optimizer instance (required if include_optimizer=True)
        """
        if not self.is_main_process():
            return

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Get model state dict
        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

        checkpoint = {"model_state_dict": state_dict, "distributed_config": self.config.to_dict()}

        if include_optimizer and optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, filepath)

        if self.verbose:
            logger.info(f"Saved checkpoint to {filepath}")

    def load_checkpoint(
        self, model: nn.Module, filepath: str | Path, optimizer: torch.optim.Optimizer | None = None
    ) -> dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            model: Model to load into
            filepath: Path to checkpoint
            optimizer: Optimizer instance (optional)

        Returns:
            Checkpoint dictionary
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = _safe_torch_load(filepath, map_location="cpu")

        # Load model state
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.verbose and self.is_main_process():
            logger.info(f"Loaded checkpoint from {filepath}")

        return checkpoint

    def print_distributed_summary(self):
        """Print distributed training configuration summary."""
        if not self.is_main_process():
            return

        print("=" * 70)
        print("Distributed Training Configuration")
        print("=" * 70)
        print(f"Strategy: {self.config.strategy.value.upper()}")
        print(f"Backend: {self.config.backend.value}")
        print(f"World Size: {self.config.world_size}")
        print(f"Rank: {self.config.rank}")
        print(f"Local Rank: {self.config.local_rank}")
        print(f"Master Address: {self.config.master_addr}:{self.config.master_port}")

        if self.config.strategy == DistributedStrategy.DDP:
            print(f"Find Unused Parameters: {self.config.find_unused_parameters}")
            print(f"Gradient As Bucket View: {self.config.gradient_as_bucket_view}")
            print(f"Static Graph: {self.config.static_graph}")

        if self.config.strategy == DistributedStrategy.FSDP:
            print(f"CPU Offload: {self.config.cpu_offload}")
            print(f"Mixed Precision: {self.config.mixed_precision}")

        print("=" * 70)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def is_distributed_available() -> bool:
    """
    Check if distributed training is available.

    Returns:
        True if distributed training is supported
    """
    return dist.is_available()


def get_world_size() -> int:
    """
    Get world size (number of processes).

    Returns:
        World size (1 if not distributed)
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """
    Get current process rank.

    Returns:
        Rank (0 if not distributed)
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """
    Check if current process is the main process.

    Returns:
        True if main process (rank 0)
    """
    return get_rank() == 0


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info("distributed_strategies module loaded")
