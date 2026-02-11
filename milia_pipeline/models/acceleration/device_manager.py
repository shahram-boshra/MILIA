"""
Device Manager

Comprehensive device detection, selection, and management for hardware acceleration.
Supports CPU, CUDA (single/multi-GPU), MPS (Apple Silicon), and TPU.

Features:
- Automatic device detection with priority ordering
- Manual device selection and validation
- Device capability querying
- Memory monitoring and optimization
- Multi-GPU device management
- Device context managers
- Hardware compatibility checks

Pydantic V2 Migration (Phase 7):
    - Migrated DeviceInfo from @dataclass to Pydantic BaseModel (mutable)
    - Uses model_dump() for to_dict() method (backward compatible)
    - NON-BREAKING: Same constructor API and attribute access
    - Follows established pattern from config_bridge.py (Phase 3)

Author: milia Team
Version: 1.1.0
"""

import logging
import warnings
from typing import Optional, List, Dict, Any, Union, Tuple
from pydantic import BaseModel
from contextlib import contextmanager
from enum import Enum

import torch
import torch.nn as nn

# Import exceptions with fallback
try:
    from milia_pipeline.exceptions import (
        ModelError,
        HardwareError,
        DeviceNotAvailableError
    )
except ImportError:
    class ModelError(Exception):
        """Base exception for model-related errors."""
        pass
    
    class HardwareError(ModelError):
        """Exception raised for hardware-related errors."""
        pass
    
    class DeviceNotAvailableError(HardwareError):
        """Exception raised when requested device is not available."""
        pass


logger = logging.getLogger(__name__)


# =============================================================================
# DEVICE TYPES
# =============================================================================

class DeviceType(Enum):
    """Enumeration of supported device types."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    TPU = "tpu"
    AUTO = "auto"


# =============================================================================
# DEVICE INFORMATION
# =============================================================================

class DeviceInfo(BaseModel):
    """
    Information about a compute device.
    
    Pattern: Follows mutable BaseModel pattern from config_bridge.py (Pydantic V2)
    
    Attributes:
        device_type: Type of device (cpu, cuda, mps, tpu)
        device_id: Device ID (for multi-GPU systems)
        name: Device name
        total_memory: Total memory in bytes (None for CPU)
        available_memory: Available memory in bytes (None for CPU)
        compute_capability: CUDA compute capability (major, minor) or None
        is_available: Whether device is available
        is_default: Whether this is the default device
    """
    device_type: str
    device_id: Optional[int] = None
    name: Optional[str] = None
    total_memory: Optional[int] = None
    available_memory: Optional[int] = None
    compute_capability: Optional[Tuple[int, int]] = None
    is_available: bool = True
    is_default: bool = False
    
    # Allow arbitrary types for compute_capability tuple compatibility
    model_config = {'arbitrary_types_allowed': True}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Backward compatible method wrapping Pydantic V2's model_dump().
        """
        return self.model_dump()
    
    def memory_summary(self) -> str:
        """Get human-readable memory summary."""
        if self.total_memory is None:
            return "N/A"
        
        total_gb = self.total_memory / (1024**3)
        if self.available_memory:
            avail_gb = self.available_memory / (1024**3)
            used_gb = total_gb - avail_gb
            return f"{used_gb:.2f}GB / {total_gb:.2f}GB used"
        else:
            return f"{total_gb:.2f}GB total"


# =============================================================================
# DEVICE MANAGER
# =============================================================================

class DeviceManager:
    """
    Manager for device detection, selection, and monitoring.
    
    Provides comprehensive device management including:
    - Automatic device detection with priority ordering
    - Manual device selection and validation
    - Device capability querying
    - Memory monitoring
    - Multi-GPU management
    
    Usage:
        >>> # Auto-detect best device
        >>> manager = DeviceManager()
        >>> device = manager.get_device()
        >>> model = model.to(device)
        >>> 
        >>> # Manual device selection
        >>> manager = DeviceManager(device="cuda:0")
        >>> device = manager.get_device()
        >>> 
        >>> # Check device info
        >>> info = manager.get_device_info()
        >>> print(f"Using {info.name} with {info.memory_summary()}")
        >>> 
        >>> # Multi-GPU setup
        >>> devices = manager.get_available_devices("cuda")
        >>> print(f"Found {len(devices)} CUDA devices")
    """
    
    # Device priority for auto-detection (highest to lowest)
    _DEVICE_PRIORITY = [DeviceType.CUDA, DeviceType.MPS, DeviceType.TPU, DeviceType.CPU]
    
    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        allow_fallback: bool = True,
        verbose: bool = True
    ):
        """
        Initialize device manager.
        
        Args:
            device: Specific device to use (e.g., "cuda", "cuda:0", "mps", "cpu")
                   If None or "auto", will auto-detect best available device
            allow_fallback: Whether to fallback to CPU if requested device unavailable
            verbose: Whether to log device information
            
        Raises:
            DeviceNotAvailableError: If requested device is not available and fallback disabled
        """
        self.allow_fallback = allow_fallback
        self.verbose = verbose
        self._device: Optional[torch.device] = None
        self._device_info: Optional[DeviceInfo] = None
        
        # Set device
        if device is None or device == "auto":
            self._device = self._auto_detect_device()
        else:
            self._device = self._validate_and_set_device(device)
        
        # Cache device info
        self._device_info = self._get_device_info(self._device)
        
        if self.verbose:
            logger.info(
                f"DeviceManager initialized - "
                f"Device: {self._device}, "
                f"Type: {self._device_info.device_type}, "
                f"Name: {self._device_info.name}"
            )
    
    def _auto_detect_device(self) -> torch.device:
        """
        Auto-detect best available device based on priority.
        
        Returns:
            torch.device: Best available device
        """
        for device_type in self._DEVICE_PRIORITY:
            if device_type == DeviceType.CUDA and torch.cuda.is_available():
                device = torch.device('cuda')
                if self.verbose:
                    logger.info(
                        f"Auto-detected CUDA device: {torch.cuda.get_device_name(0)}"
                    )
                return device
            
            elif device_type == DeviceType.MPS and self._is_mps_available():
                device = torch.device('mps')
                if self.verbose:
                    logger.info("Auto-detected MPS device (Apple Silicon)")
                return device
            
            elif device_type == DeviceType.TPU and self._is_tpu_available():
                device = self._get_tpu_device()
                if self.verbose:
                    logger.info("Auto-detected TPU device")
                return device
        
        # Fallback to CPU
        device = torch.device('cpu')
        if self.verbose:
            logger.info("No accelerator detected, using CPU")
        return device
    
    def _validate_and_set_device(
        self,
        device: Union[str, torch.device]
    ) -> torch.device:
        """
        Validate and set specific device.
        
        Args:
            device: Device specification (str or torch.device)
            
        Returns:
            torch.device: Validated device
            
        Raises:
            DeviceNotAvailableError: If device not available
        """
        # Convert to torch.device
        if isinstance(device, str):
            device = torch.device(device)
        
        # Validate device availability
        if device.type == 'cuda':
            if not torch.cuda.is_available():
                if self.allow_fallback:
                    logger.warning(
                        f"CUDA not available, falling back to CPU"
                    )
                    return torch.device('cpu')
                else:
                    raise DeviceNotAvailableError(
                        "CUDA requested but not available. "
                        "Install CUDA-enabled PyTorch or set allow_fallback=True."
                    )
            
            # Validate specific GPU ID
            if device.index is not None:
                if device.index >= torch.cuda.device_count():
                    if self.allow_fallback:
                        logger.warning(
                            f"CUDA device {device.index} not available, "
                            f"using cuda:0"
                        )
                        return torch.device('cuda:0')
                    else:
                        raise DeviceNotAvailableError(
                            f"CUDA device {device.index} not available. "
                            f"Available devices: 0-{torch.cuda.device_count()-1}"
                        )
        
        elif device.type == 'mps':
            if not self._is_mps_available():
                if self.allow_fallback:
                    logger.warning(
                        f"MPS not available, falling back to CPU"
                    )
                    return torch.device('cpu')
                else:
                    raise DeviceNotAvailableError(
                        "MPS requested but not available. "
                        "MPS requires PyTorch 1.12+ on Apple Silicon."
                    )
        
        elif device.type == 'tpu':
            if not self._is_tpu_available():
                if self.allow_fallback:
                    logger.warning(
                        f"TPU not available, falling back to CPU"
                    )
                    return torch.device('cpu')
                else:
                    raise DeviceNotAvailableError(
                        "TPU requested but not available."
                    )
        
        return device
    
    def _is_mps_available(self) -> bool:
        """Check if MPS (Metal Performance Shaders) is available."""
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    def _is_tpu_available(self) -> bool:
        """Check if TPU is available."""
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            return True
        except ImportError:
            return False
    
    def _get_tpu_device(self) -> torch.device:
        """Get TPU device."""
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        except ImportError:
            raise DeviceNotAvailableError(
                "TPU support requires torch_xla package. "
                "Install with: pip install torch_xla"
            )
    
    def _get_device_info(self, device: torch.device) -> DeviceInfo:
        """
        Get detailed information about a device.
        
        Args:
            device: Device to query
            
        Returns:
            DeviceInfo: Device information
        """
        if device.type == 'cuda':
            device_id = device.index if device.index is not None else 0
            props = torch.cuda.get_device_properties(device_id)
            
            return DeviceInfo(
                device_type='cuda',
                device_id=device_id,
                name=props.name,
                total_memory=props.total_memory,
                available_memory=self._get_cuda_available_memory(device_id),
                compute_capability=(props.major, props.minor),
                is_available=True,
                is_default=(device_id == 0)
            )
        
        elif device.type == 'mps':
            return DeviceInfo(
                device_type='mps',
                name='Apple MPS',
                is_available=True,
                is_default=True
            )
        
        elif device.type == 'tpu':
            return DeviceInfo(
                device_type='tpu',
                name='Google TPU',
                is_available=True,
                is_default=True
            )
        
        else:  # CPU
            return DeviceInfo(
                device_type='cpu',
                name='CPU',
                is_available=True,
                is_default=True
            )
    
    def _get_cuda_available_memory(self, device_id: int = 0) -> int:
        """
        Get available CUDA memory.
        
        Args:
            device_id: CUDA device ID
            
        Returns:
            Available memory in bytes
        """
        if not torch.cuda.is_available():
            return 0
        
        try:
            torch.cuda.set_device(device_id)
            total = torch.cuda.get_device_properties(device_id).total_memory
            allocated = torch.cuda.memory_allocated(device_id)
            return total - allocated
        except Exception as e:
            logger.debug(f"Could not get CUDA memory info: {e}")
            return 0
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def get_device(self) -> torch.device:
        """
        Get the currently configured device.
        
        Returns:
            torch.device: Current device
        """
        return self._device
    
    def get_device_info(self) -> DeviceInfo:
        """
        Get detailed information about current device.
        
        Returns:
            DeviceInfo: Device information
        """
        return self._device_info
    
    def get_available_devices(
        self,
        device_type: Optional[str] = None
    ) -> List[DeviceInfo]:
        """
        Get list of available devices.
        
        Args:
            device_type: Filter by device type (cuda, mps, cpu, tpu)
                        If None, returns all available devices
        
        Returns:
            List of DeviceInfo objects
        """
        devices = []
        
        # CPU is always available
        if device_type is None or device_type == 'cpu':
            devices.append(DeviceInfo(
                device_type='cpu',
                name='CPU',
                is_available=True
            ))
        
        # CUDA devices
        if (device_type is None or device_type == 'cuda') and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append(DeviceInfo(
                    device_type='cuda',
                    device_id=i,
                    name=props.name,
                    total_memory=props.total_memory,
                    available_memory=self._get_cuda_available_memory(i),
                    compute_capability=(props.major, props.minor),
                    is_available=True,
                    is_default=(i == 0)
                ))
        
        # MPS device
        if (device_type is None or device_type == 'mps') and self._is_mps_available():
            devices.append(DeviceInfo(
                device_type='mps',
                name='Apple MPS',
                is_available=True
            ))
        
        # TPU device
        if (device_type is None or device_type == 'tpu') and self._is_tpu_available():
            devices.append(DeviceInfo(
                device_type='tpu',
                name='Google TPU',
                is_available=True
            ))
        
        return devices
    
    def move_to_device(
        self,
        model: nn.Module,
        non_blocking: bool = False
    ) -> nn.Module:
        """
        Move model to the configured device.
        
        Args:
            model: PyTorch model to move
            non_blocking: Whether to use non-blocking transfer
            
        Returns:
            Model on device
        """
        model = model.to(self._device, non_blocking=non_blocking)
        
        if self.verbose:
            logger.info(f"Model moved to {self._device}")
        
        return model
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get current memory usage information.
        
        Returns:
            Dictionary with memory information:
            - device: Device name
            - total_memory: Total memory (bytes)
            - allocated_memory: Currently allocated memory (bytes)
            - reserved_memory: Reserved memory (bytes)
            - free_memory: Free memory (bytes)
        """
        if self._device.type != 'cuda':
            return {
                'device': str(self._device),
                'message': 'Memory info only available for CUDA devices'
            }
        
        device_id = self._device.index if self._device.index is not None else 0
        
        total = torch.cuda.get_device_properties(device_id).total_memory
        allocated = torch.cuda.memory_allocated(device_id)
        reserved = torch.cuda.memory_reserved(device_id)
        free = total - reserved
        
        return {
            'device': str(self._device),
            'total_memory': total,
            'allocated_memory': allocated,
            'reserved_memory': reserved,
            'free_memory': free,
            'total_memory_gb': total / (1024**3),
            'allocated_memory_gb': allocated / (1024**3),
            'reserved_memory_gb': reserved / (1024**3),
            'free_memory_gb': free / (1024**3)
        }
    
    def reset_peak_memory_stats(self):
        """Reset peak memory statistics (CUDA only)."""
        if self._device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self._device)
            if self.verbose:
                logger.info("Reset peak memory statistics")
    
    def empty_cache(self):
        """Empty the cache (CUDA only)."""
        if self._device.type == 'cuda':
            torch.cuda.empty_cache()
            if self.verbose:
                logger.info("Emptied CUDA cache")
    
    def synchronize(self):
        """Synchronize device (wait for all operations to complete)."""
        if self._device.type == 'cuda':
            torch.cuda.synchronize(self._device)
        elif self._device.type == 'tpu':
            try:
                import torch_xla.core.xla_model as xm
                xm.mark_step()
            except ImportError:
                pass
    
    @contextmanager
    def device_context(self, device: Optional[Union[str, torch.device]] = None):
        """
        Context manager for temporary device switching.
        
        Args:
            device: Device to switch to (uses current device if None)
            
        Usage:
            >>> manager = DeviceManager()
            >>> with manager.device_context("cuda:1"):
            ...     # Operations on cuda:1
            ...     output = model(input)
            >>> # Back to original device
        """
        if device is None:
            device = self._device
        elif isinstance(device, str):
            device = torch.device(device)
        
        # Save original device
        original_device = self._device
        
        try:
            # Switch to new device
            self._device = device
            yield device
        finally:
            # Restore original device
            self._device = original_device
    
    def print_device_summary(self):
        """Print a formatted summary of available devices."""
        print("=" * 70)
        print("Device Summary")
        print("=" * 70)
        
        devices = self.get_available_devices()
        
        for device in devices:
            print(f"\n{device.device_type.upper()}", end="")
            if device.device_id is not None:
                print(f":{device.device_id}", end="")
            print(f" - {device.name}")
            
            if device.total_memory:
                print(f"  Memory: {device.memory_summary()}")
            
            if device.compute_capability:
                print(f"  Compute Capability: {device.compute_capability[0]}.{device.compute_capability[1]}")
            
            if device.is_default:
                print("  [DEFAULT]")
        
        print("\n" + "=" * 70)
        print(f"Current Device: {self._device}")
        print("=" * 70)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_default_device(
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = False
) -> torch.device:
    """
    Get default device (convenience function).
    
    Args:
        device: Specific device or "auto" for auto-detection
        verbose: Whether to log device info
        
    Returns:
        torch.device: Selected device
        
    Example:
        >>> device = get_default_device()
        >>> model = model.to(device)
        >>> 
        >>> # Or with specific device
        >>> device = get_default_device("cuda:0")
    """
    manager = DeviceManager(device=device, verbose=verbose)
    return manager.get_device()


def list_available_devices() -> List[DeviceInfo]:
    """
    List all available devices (convenience function).
    
    Returns:
        List of DeviceInfo objects
        
    Example:
        >>> devices = list_available_devices()
        >>> for device in devices:
        ...     print(f"{device.device_type}: {device.name}")
    """
    manager = DeviceManager(verbose=False)
    return manager.get_available_devices()


def get_device_capabilities() -> Dict[str, bool]:
    """
    Get capabilities of available hardware.
    
    Returns:
        Dictionary with capability flags:
        - cuda_available: Whether CUDA is available
        - cuda_device_count: Number of CUDA devices
        - mps_available: Whether MPS is available
        - tpu_available: Whether TPU is available
        - cudnn_available: Whether cuDNN is available
        - cudnn_enabled: Whether cuDNN is enabled
        
    Example:
        >>> caps = get_device_capabilities()
        >>> if caps['cuda_available']:
        ...     print(f"Found {caps['cuda_device_count']} CUDA devices")
    """
    capabilities = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'tpu_available': False,
        'cudnn_available': torch.backends.cudnn.is_available() if torch.cuda.is_available() else False,
        'cudnn_enabled': torch.backends.cudnn.enabled if torch.cuda.is_available() else False
    }
    
    # Check TPU
    try:
        import torch_xla
        capabilities['tpu_available'] = True
    except ImportError:
        pass
    
    return capabilities


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info("device_manager module loaded")
