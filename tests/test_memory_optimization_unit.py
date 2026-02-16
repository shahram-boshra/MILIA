#!/usr/bin/env python3
"""
Extended Production-Ready Unit Test Suite for memory_optimization.py Module

Comprehensive test coverage including:
- MemoryConfig Pydantic BaseModel (initialization, to_dict via model_dump(), all fields)
- MemoryOptimizer initialization and configuration
- Configuration validation (_validate_config)
- Mixed precision training (autocast, _get_autocast_dtype)
- Gradient scaling (get_grad_scaler, scale_loss, step_optimizer)
- Gradient checkpointing (enable_gradient_checkpointing, checkpoint_sequential)
- Memory monitoring (get_memory_stats, get_memory_summary)
- Cache management (reset_peak_memory_stats, empty_cache, run_garbage_collection)
- Step-based optimization (step, check_memory_usage)
- Memory profiling (profile_memory, get_memory_snapshot)
- DataLoader optimization (optimize_dataloader)
- Memory leak detection (detect_memory_leaks)
- Memory summary printing (print_memory_summary)
- Convenience functions (get_memory_efficient_settings, estimate_model_memory)
- Exception handling (ModelError, HardwareError, VQMMemoryError)
- Edge cases and error scenarios

Author: milia Team
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import logging
import warnings
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn

# Backward-compatible GradScaler import: torch.amp.GradScaler is preferred in PyTorch 2.4+,
# but torch.cuda.amp.GradScaler remains available for backward compatibility.
# We import from torch.cuda.amp to match the module under test's import pattern.
from torch.cuda.amp import GradScaler

# Import the module under test
from milia_pipeline.models.acceleration.memory_optimization import (
    HardwareError,
    MemoryConfig,
    MemoryOptimizer,
    ModelError,
    VQMMemoryError,
    estimate_model_memory,
    get_memory_efficient_settings,
)

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_model():
    """Create a mock PyTorch model."""
    model = Mock(spec=nn.Module)
    model.train = Mock(return_value=model)
    model.eval = Mock(return_value=model)
    model.to = Mock(return_value=model)
    model.state_dict = Mock(return_value={"param1": torch.tensor([1.0])})
    model.load_state_dict = Mock()
    model.parameters = Mock(return_value=[torch.nn.Parameter(torch.randn(3, 3))])

    def mock_forward(*args, **kwargs):
        batch_size = 4
        return torch.randn(batch_size, 1, requires_grad=True)

    model.__call__ = Mock(side_effect=mock_forward)
    model.forward = Mock(side_effect=mock_forward)
    return model


@pytest.fixture
def simple_model():
    """Create a simple real PyTorch model for testing."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
            self.relu = nn.ReLU()
            self.output = nn.Linear(5, 1)

        def forward(self, x):
            x = self.linear(x)
            x = self.relu(x)
            return self.output(x)

    return SimpleModel()


@pytest.fixture
def sample_input():
    """Create sample input tensor."""
    return torch.randn(4, 10)


@pytest.fixture
def default_memory_config():
    """Create default MemoryConfig."""
    return MemoryConfig()


@pytest.fixture
def mixed_precision_config():
    """Create MemoryConfig with mixed precision enabled."""
    return MemoryConfig(
        mixed_precision=True,
        precision="fp16",
        gradient_checkpointing=False,
        pin_memory=True,
        non_blocking=True,
    )


@pytest.fixture
def bf16_memory_config():
    """Create MemoryConfig for BF16 precision."""
    return MemoryConfig(mixed_precision=True, precision="bf16", gradient_checkpointing=True)


@pytest.fixture
def full_memory_config():
    """Create MemoryConfig with all options set."""
    return MemoryConfig(
        mixed_precision=True,
        precision="fp16",
        gradient_checkpointing=True,
        pin_memory=True,
        non_blocking=True,
        empty_cache_interval=50,
        garbage_collect_interval=100,
        max_memory_allocated=8192,
        growth_interval=10,
    )


@pytest.fixture
def mock_cuda_available():
    """Mock CUDA as available."""
    with patch("torch.cuda.is_available", return_value=True):
        yield


@pytest.fixture
def mock_cuda_unavailable():
    """Mock CUDA as unavailable."""
    with patch("torch.cuda.is_available", return_value=False):
        yield


@pytest.fixture
def mock_cuda_device():
    """Create mock CUDA device with all required methods."""
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.is_bf16_supported", return_value=True):
            with patch("torch.cuda.synchronize"):
                with patch("torch.cuda.memory_allocated", return_value=1024**3):
                    with patch("torch.cuda.memory_reserved", return_value=2 * 1024**3):
                        with patch("torch.cuda.max_memory_allocated", return_value=1.5 * 1024**3):
                            with patch(
                                "torch.cuda.max_memory_reserved", return_value=2.5 * 1024**3
                            ):
                                mock_props = MagicMock()
                                mock_props.total_memory = 8 * 1024**3
                                with patch(
                                    "torch.cuda.get_device_properties", return_value=mock_props
                                ):
                                    yield


@pytest.fixture
def mock_dataloader():
    """Create a mock DataLoader."""
    mock_dataset = MagicMock()
    mock_dataset.__len__ = Mock(return_value=100)

    dataloader = MagicMock(spec=torch.utils.data.DataLoader)
    dataloader.dataset = mock_dataset
    dataloader.batch_size = 32
    dataloader.drop_last = False
    dataloader.timeout = 0
    dataloader.worker_init_fn = None
    return dataloader


# =============================================================================
# MEMORY CONFIG PYDANTIC MODEL TESTS
# =============================================================================


class TestMemoryConfig:
    """Test MemoryConfig Pydantic BaseModel (migrated from dataclass in Phase 9)."""

    def test_default_initialization(self):
        """Test MemoryConfig with default values."""
        config = MemoryConfig()

        assert config.mixed_precision is False
        assert config.precision == "fp16"
        assert config.gradient_checkpointing is False
        assert config.pin_memory is True
        assert config.non_blocking is True
        assert config.empty_cache_interval == 0
        assert config.garbage_collect_interval == 0
        assert config.max_memory_allocated == 0
        assert config.growth_interval == 0

    def test_mixed_precision_initialization(self, mixed_precision_config):
        """Test MemoryConfig with mixed precision settings."""
        config = mixed_precision_config

        assert config.mixed_precision is True
        assert config.precision == "fp16"
        assert config.gradient_checkpointing is False
        assert config.pin_memory is True
        assert config.non_blocking is True

    def test_bf16_precision_initialization(self, bf16_memory_config):
        """Test MemoryConfig with BF16 precision."""
        config = bf16_memory_config

        assert config.mixed_precision is True
        assert config.precision == "bf16"
        assert config.gradient_checkpointing is True

    def test_full_initialization(self, full_memory_config):
        """Test MemoryConfig with all fields set."""
        config = full_memory_config

        assert config.mixed_precision is True
        assert config.precision == "fp16"
        assert config.gradient_checkpointing is True
        assert config.pin_memory is True
        assert config.non_blocking is True
        assert config.empty_cache_interval == 50
        assert config.garbage_collect_interval == 100
        assert config.max_memory_allocated == 8192
        assert config.growth_interval == 10

    def test_to_dict_default(self):
        """Test to_dict with default values."""
        config = MemoryConfig()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["mixed_precision"] is False
        assert result["precision"] == "fp16"
        assert result["gradient_checkpointing"] is False
        assert result["pin_memory"] is True
        assert result["non_blocking"] is True
        assert result["empty_cache_interval"] == 0
        assert result["garbage_collect_interval"] == 0
        assert result["max_memory_allocated"] == 0
        assert result["growth_interval"] == 0

    def test_to_dict_with_mixed_precision(self, mixed_precision_config):
        """Test to_dict with mixed precision configuration."""
        result = mixed_precision_config.to_dict()

        assert result["mixed_precision"] is True
        assert result["precision"] == "fp16"

    def test_to_dict_with_bf16(self, bf16_memory_config):
        """Test to_dict with BF16 configuration."""
        result = bf16_memory_config.to_dict()

        assert result["precision"] == "bf16"
        assert result["gradient_checkpointing"] is True

    def test_to_dict_includes_all_fields(self):
        """Test to_dict includes all expected fields."""
        config = MemoryConfig()
        result = config.to_dict()

        expected_keys = [
            "mixed_precision",
            "precision",
            "gradient_checkpointing",
            "pin_memory",
            "non_blocking",
            "empty_cache_interval",
            "garbage_collect_interval",
            "max_memory_allocated",
            "growth_interval",
        ]

        for key in expected_keys:
            assert key in result

        assert len(result) == len(expected_keys)

    def test_to_dict_returns_new_dict(self):
        """Test to_dict returns a new dictionary each time."""
        config = MemoryConfig()

        dict1 = config.to_dict()
        dict2 = config.to_dict()

        assert dict1 is not dict2
        assert dict1 == dict2

    def test_config_precision_fp32(self):
        """Test MemoryConfig with FP32 precision."""
        config = MemoryConfig(precision="fp32")

        assert config.precision == "fp32"
        result = config.to_dict()
        assert result["precision"] == "fp32"

    def test_config_precision_fp8(self):
        """Test MemoryConfig with FP8 precision."""
        config = MemoryConfig(precision="fp8")

        assert config.precision == "fp8"

    def test_config_cache_intervals(self):
        """Test MemoryConfig with cache interval settings."""
        config = MemoryConfig(empty_cache_interval=25, garbage_collect_interval=50)

        assert config.empty_cache_interval == 25
        assert config.garbage_collect_interval == 50

    def test_config_max_memory_allocated(self):
        """Test MemoryConfig max_memory_allocated field."""
        config = MemoryConfig(max_memory_allocated=16384)

        assert config.max_memory_allocated == 16384

    def test_pydantic_model_dump_equivalence(self):
        """Test that to_dict() correctly wraps Pydantic's model_dump()."""
        config = MemoryConfig(mixed_precision=True, precision="bf16", gradient_checkpointing=True)

        # to_dict() should return same result as model_dump()
        to_dict_result = config.to_dict()
        model_dump_result = config.model_dump()

        assert to_dict_result == model_dump_result

    def test_pydantic_model_is_mutable(self):
        """Test that MemoryConfig is mutable as expected for Pydantic BaseModel."""
        config = MemoryConfig()

        # Pydantic BaseModel allows attribute mutation
        config.mixed_precision = True
        config.precision = "bf16"

        assert config.mixed_precision is True
        assert config.precision == "bf16"

    def test_pydantic_model_validation_types(self):
        """Test MemoryConfig Pydantic type coercion behavior."""
        # Pydantic should coerce compatible types
        config = MemoryConfig(
            empty_cache_interval=50,  # int
            max_memory_allocated=8192,  # int
        )

        assert isinstance(config.empty_cache_interval, int)
        assert isinstance(config.max_memory_allocated, int)

    def test_pydantic_model_field_names(self):
        """Test that Pydantic model has expected field names."""
        config = MemoryConfig()

        # Pydantic V2 provides model_fields attribute
        expected_fields = {
            "mixed_precision",
            "precision",
            "gradient_checkpointing",
            "pin_memory",
            "non_blocking",
            "empty_cache_interval",
            "garbage_collect_interval",
            "max_memory_allocated",
            "growth_interval",
        }

        actual_fields = set(config.model_fields.keys())
        assert actual_fields == expected_fields


# =============================================================================
# MEMORY OPTIMIZER INITIALIZATION TESTS
# =============================================================================


class TestMemoryOptimizerInitialization:
    """Test MemoryOptimizer initialization and configuration."""

    def test_minimal_initialization(self):
        """Test MemoryOptimizer with default parameters."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(verbose=False)

        assert optimizer.config.mixed_precision is False
        assert optimizer.config.precision == "fp16"
        assert optimizer.config.gradient_checkpointing is False
        assert optimizer.verbose is False
        assert optimizer._step_count == 0
        assert optimizer._grad_scaler is None

    def test_initialization_with_mixed_precision_cuda(self):
        """Test MemoryOptimizer with mixed precision on CUDA."""
        with patch("torch.cuda.is_available", return_value=True):
            optimizer = MemoryOptimizer(mixed_precision=True, verbose=False)

        assert optimizer.config.mixed_precision is True
        assert optimizer._grad_scaler is not None
        assert isinstance(optimizer._grad_scaler, GradScaler)

    def test_initialization_with_mixed_precision_cpu(self):
        """Test MemoryOptimizer with mixed precision on CPU."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(
                mixed_precision=True, device=torch.device("cpu"), verbose=False
            )

        assert optimizer.config.mixed_precision is True
        assert optimizer._grad_scaler is None

    def test_initialization_with_precision_fp16(self):
        """Test MemoryOptimizer with FP16 precision."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(precision="fp16", verbose=False)

        assert optimizer.config.precision == "fp16"

    def test_initialization_with_precision_fp32(self):
        """Test MemoryOptimizer with FP32 precision."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(precision="fp32", verbose=False)

        assert optimizer.config.precision == "fp32"

    def test_initialization_with_gradient_checkpointing(self):
        """Test MemoryOptimizer with gradient checkpointing enabled."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(gradient_checkpointing=True, verbose=False)

        assert optimizer.config.gradient_checkpointing is True

    def test_initialization_with_pin_memory_false(self):
        """Test MemoryOptimizer with pin_memory disabled."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(pin_memory=False, verbose=False)

        assert optimizer.config.pin_memory is False

    def test_initialization_with_non_blocking_false(self):
        """Test MemoryOptimizer with non_blocking disabled."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(non_blocking=False, verbose=False)

        assert optimizer.config.non_blocking is False

    def test_initialization_with_cache_intervals(self):
        """Test MemoryOptimizer with cache interval settings."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(
                empty_cache_interval=100, garbage_collect_interval=200, verbose=False
            )

        assert optimizer.config.empty_cache_interval == 100
        assert optimizer.config.garbage_collect_interval == 200

    def test_initialization_with_max_memory_allocated(self):
        """Test MemoryOptimizer with max_memory_allocated."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(max_memory_allocated=8192, verbose=False)

        assert optimizer.config.max_memory_allocated == 8192

    def test_initialization_with_custom_device(self):
        """Test MemoryOptimizer with custom device."""
        custom_device = torch.device("cpu")

        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(device=custom_device, verbose=False)

        assert optimizer.device == custom_device

    def test_initialization_verbose_true(self):
        """Test MemoryOptimizer with verbose=True."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(verbose=True)

        assert optimizer.verbose is True

    def test_initialization_verbose_false(self):
        """Test MemoryOptimizer with verbose=False."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(verbose=False)

        assert optimizer.verbose is False

    def test_initialization_default_device_cuda(self):
        """Test MemoryOptimizer default device is CUDA when available."""
        with patch("torch.cuda.is_available", return_value=True):
            optimizer = MemoryOptimizer(verbose=False)

        assert optimizer.device.type == "cuda"

    def test_initialization_default_device_cpu(self):
        """Test MemoryOptimizer default device is CPU when CUDA unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(verbose=False)

        assert optimizer.device.type == "cpu"

    def test_initialization_step_count_starts_at_zero(self):
        """Test MemoryOptimizer step count starts at zero."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(verbose=False)

        assert optimizer._step_count == 0

    def test_verbose_logging_on_init(self, caplog):
        """Test verbose=True produces logging during initialization."""
        with patch("torch.cuda.is_available", return_value=False):
            with caplog.at_level(logging.INFO):
                _optimizer = MemoryOptimizer(verbose=True)

        assert "MemoryOptimizer initialized" in caplog.text

    def test_silent_initialization(self, caplog):
        """Test silent initialization when verbose=False."""
        with caplog.at_level(logging.INFO):
            caplog.clear()
            with patch("torch.cuda.is_available", return_value=False):
                _optimizer = MemoryOptimizer(verbose=False)

        init_logs = [r for r in caplog.records if "MemoryOptimizer initialized" in r.message]
        assert len(init_logs) == 0


# =============================================================================
# CONFIGURATION VALIDATION TESTS
# =============================================================================


class TestConfigValidation:
    """Test _validate_config method."""

    def test_validate_bf16_supported(self):
        """Test BF16 validation when supported."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.is_bf16_supported", return_value=True):
                optimizer = MemoryOptimizer(precision="bf16", verbose=False)

        assert optimizer.config.precision == "bf16"

    def test_validate_bf16_not_supported_fallback(self):
        """Test BF16 validation falls back to FP16 when not supported."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.is_bf16_supported", return_value=False):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    optimizer = MemoryOptimizer(precision="bf16", verbose=False)

                    assert optimizer.config.precision == "fp16"
                    assert len(w) >= 1
                    assert "BF16 not supported" in str(w[0].message)

    def test_validate_mixed_precision_unsupported_device(self):
        """Test mixed precision disabled for unsupported device types."""
        # Create a non-CUDA, non-CPU device (mocked)
        mock_device = MagicMock()
        mock_device.type = "xla"

        with patch("torch.cuda.is_available", return_value=False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                optimizer = MemoryOptimizer(mixed_precision=True, device=mock_device, verbose=False)

                assert optimizer.config.mixed_precision is False
                assert len(w) >= 1
                assert "Mixed precision not supported" in str(w[0].message)

    def test_validate_fp16_on_cuda(self):
        """Test FP16 validation on CUDA device."""
        with patch("torch.cuda.is_available", return_value=True):
            optimizer = MemoryOptimizer(precision="fp16", verbose=False)

        assert optimizer.config.precision == "fp16"

    def test_validate_fp32_on_any_device(self):
        """Test FP32 validation on any device."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(precision="fp32", verbose=False)

        assert optimizer.config.precision == "fp32"


# =============================================================================
# MIXED PRECISION TRAINING TESTS
# =============================================================================


class TestMixedPrecisionTraining:
    """Test mixed precision training functionality."""

    def test_autocast_disabled(self):
        """Test autocast context manager when mixed precision disabled."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(mixed_precision=False, verbose=False)

        with optimizer.autocast():
            tensor = torch.randn(10)

        assert tensor is not None

    def test_autocast_enabled_cuda(self):
        """Test autocast context manager with CUDA and mixed precision enabled."""
        with patch("torch.cuda.is_available", return_value=True):
            optimizer = MemoryOptimizer(mixed_precision=True, verbose=False)

        with patch("torch.cuda.amp.autocast") as mock_autocast:
            _mock_context = MagicMock()
            mock_autocast.return_value.__enter__ = Mock(return_value=None)
            mock_autocast.return_value.__exit__ = Mock(return_value=None)

            with optimizer.autocast():
                pass

    def test_autocast_enabled_cpu(self):
        """Test autocast context manager with CPU and mixed precision enabled."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(
                mixed_precision=True, device=torch.device("cpu"), verbose=False
            )

        with optimizer.autocast():
            tensor = torch.randn(10)

        assert tensor is not None

    def test_get_autocast_dtype_fp16(self):
        """Test _get_autocast_dtype returns float16 for fp16."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(precision="fp16", verbose=False)

        dtype = optimizer._get_autocast_dtype()
        assert dtype == torch.float16

    def test_get_autocast_dtype_bf16(self):
        """Test _get_autocast_dtype returns bfloat16 for bf16."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.is_bf16_supported", return_value=True):
                optimizer = MemoryOptimizer(precision="bf16", verbose=False)

        dtype = optimizer._get_autocast_dtype()
        assert dtype == torch.bfloat16

    def test_get_autocast_dtype_fp32(self):
        """Test _get_autocast_dtype returns float32 for fp32."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(precision="fp32", verbose=False)

        dtype = optimizer._get_autocast_dtype()
        assert dtype == torch.float32

    def test_get_autocast_dtype_unknown_defaults_to_fp32(self):
        """Test _get_autocast_dtype defaults to float32 for unknown precision."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(precision="unknown", verbose=False)

        dtype = optimizer._get_autocast_dtype()
        assert dtype == torch.float32


# =============================================================================
# GRADIENT SCALING TESTS
# =============================================================================


class TestGradientScaling:
    """Test gradient scaling functionality."""

    def test_get_grad_scaler_with_mixed_precision_cuda(self):
        """Test get_grad_scaler returns GradScaler when mixed precision enabled on CUDA."""
        with patch("torch.cuda.is_available", return_value=True):
            optimizer = MemoryOptimizer(mixed_precision=True, verbose=False)

        scaler = optimizer.get_grad_scaler()
        assert scaler is not None
        assert isinstance(scaler, GradScaler)

    def test_get_grad_scaler_without_mixed_precision(self):
        """Test get_grad_scaler returns None when mixed precision disabled."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(mixed_precision=False, verbose=False)

        scaler = optimizer.get_grad_scaler()
        assert scaler is None

    def test_get_grad_scaler_cpu_device(self):
        """Test get_grad_scaler returns None for CPU device."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(
                mixed_precision=True, device=torch.device("cpu"), verbose=False
            )

        scaler = optimizer.get_grad_scaler()
        assert scaler is None

    def test_scale_loss_with_scaler(self):
        """Test scale_loss scales loss when scaler is available."""
        with patch("torch.cuda.is_available", return_value=True):
            optimizer = MemoryOptimizer(mixed_precision=True, verbose=False)

        loss = torch.tensor(1.0, requires_grad=True)

        with patch.object(optimizer._grad_scaler, "scale", return_value=loss * 2) as mock_scale:
            _scaled_loss = optimizer.scale_loss(loss)
            mock_scale.assert_called_once_with(loss)

    def test_scale_loss_without_scaler(self):
        """Test scale_loss returns original loss when scaler unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(mixed_precision=False, verbose=False)

        loss = torch.tensor(1.0, requires_grad=True)
        scaled_loss = optimizer.scale_loss(loss)

        assert torch.equal(scaled_loss, loss)

    def test_step_optimizer_with_scaler(self):
        """Test step_optimizer with gradient scaler."""
        with patch("torch.cuda.is_available", return_value=True):
            optimizer = MemoryOptimizer(mixed_precision=True, verbose=False)

        mock_optimizer = MagicMock()

        with patch.object(optimizer._grad_scaler, "unscale_") as mock_unscale:
            with patch.object(optimizer._grad_scaler, "step") as mock_step:
                with patch.object(optimizer._grad_scaler, "update") as mock_update:
                    optimizer.step_optimizer(mock_optimizer, scaler_unscale=True)

                    mock_unscale.assert_called_once_with(mock_optimizer)
                    mock_step.assert_called_once_with(mock_optimizer)
                    mock_update.assert_called_once()

    def test_step_optimizer_with_scaler_no_unscale(self):
        """Test step_optimizer with gradient scaler without unscaling."""
        with patch("torch.cuda.is_available", return_value=True):
            optimizer = MemoryOptimizer(mixed_precision=True, verbose=False)

        mock_optimizer = MagicMock()

        with patch.object(optimizer._grad_scaler, "unscale_") as mock_unscale:
            with patch.object(optimizer._grad_scaler, "step") as mock_step:
                with patch.object(optimizer._grad_scaler, "update") as mock_update:
                    optimizer.step_optimizer(mock_optimizer, scaler_unscale=False)

                    mock_unscale.assert_not_called()
                    mock_step.assert_called_once_with(mock_optimizer)
                    mock_update.assert_called_once()

    def test_step_optimizer_without_scaler(self):
        """Test step_optimizer without gradient scaler."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(mixed_precision=False, verbose=False)

        mock_optimizer = MagicMock()
        optimizer.step_optimizer(mock_optimizer)

        mock_optimizer.step.assert_called_once()


# =============================================================================
# GRADIENT CHECKPOINTING TESTS
# =============================================================================


class TestGradientCheckpointing:
    """Test gradient checkpointing functionality."""

    def test_enable_gradient_checkpointing_disabled(self, simple_model):
        """Test enable_gradient_checkpointing returns model unchanged when disabled."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(gradient_checkpointing=False, verbose=False)

        result = optimizer.enable_gradient_checkpointing(simple_model)
        assert result is simple_model

    def test_enable_gradient_checkpointing_pyg_method(self):
        """Test enable_gradient_checkpointing uses PyG method when available."""
        mock_model = MagicMock()
        mock_model.gradient_checkpointing_enable = MagicMock()

        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(gradient_checkpointing=True, verbose=False)

        result = optimizer.enable_gradient_checkpointing(mock_model)

        mock_model.gradient_checkpointing_enable.assert_called_once()
        assert result is mock_model

    def test_enable_gradient_checkpointing_transformers_method(self):
        """Test enable_gradient_checkpointing uses Transformers method when available."""
        mock_model = MagicMock()
        del mock_model.gradient_checkpointing_enable
        mock_model.enable_gradient_checkpointing = MagicMock()

        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(gradient_checkpointing=True, verbose=False)

        result = optimizer.enable_gradient_checkpointing(mock_model)

        mock_model.enable_gradient_checkpointing.assert_called_once()
        assert result is mock_model

    def test_enable_gradient_checkpointing_manual_wrapper(self, simple_model):
        """Test enable_gradient_checkpointing uses manual wrapper as fallback."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(gradient_checkpointing=True, verbose=False)

        original_forward = simple_model.forward
        result = optimizer.enable_gradient_checkpointing(simple_model)

        # Forward should be wrapped
        assert result.forward != original_forward

    def test_enable_gradient_checkpointing_logs_pyg(self, caplog):
        """Test enable_gradient_checkpointing logs PyG method usage."""
        mock_model = MagicMock()
        mock_model.gradient_checkpointing_enable = MagicMock()

        with patch("torch.cuda.is_available", return_value=False):
            with caplog.at_level(logging.INFO):
                optimizer = MemoryOptimizer(gradient_checkpointing=True, verbose=True)
                optimizer.enable_gradient_checkpointing(mock_model)

        assert "PyG method" in caplog.text

    def test_enable_gradient_checkpointing_logs_transformers(self, caplog):
        """Test enable_gradient_checkpointing logs Transformers method usage."""
        mock_model = MagicMock()
        del mock_model.gradient_checkpointing_enable
        mock_model.enable_gradient_checkpointing = MagicMock()

        with patch("torch.cuda.is_available", return_value=False):
            with caplog.at_level(logging.INFO):
                optimizer = MemoryOptimizer(gradient_checkpointing=True, verbose=True)
                optimizer.enable_gradient_checkpointing(mock_model)

        assert "Transformers method" in caplog.text

    def test_enable_gradient_checkpointing_logs_manual(self, caplog, simple_model):
        """Test enable_gradient_checkpointing logs manual wrapper usage."""
        with patch("torch.cuda.is_available", return_value=False):
            with caplog.at_level(logging.INFO):
                optimizer = MemoryOptimizer(gradient_checkpointing=True, verbose=True)
                optimizer.enable_gradient_checkpointing(simple_model)

        assert "manual wrapper" in caplog.text

    def test_enable_gradient_checkpointing_with_segments(self):
        """Test enable_gradient_checkpointing with checkpoint_segments parameter."""
        mock_model = MagicMock()
        mock_model.gradient_checkpointing_enable = MagicMock()

        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(gradient_checkpointing=True, verbose=False)

        _result = optimizer.enable_gradient_checkpointing(mock_model, checkpoint_segments=4)

        mock_model.gradient_checkpointing_enable.assert_called_once()

    def test_checkpoint_sequential(self):
        """Test checkpoint_sequential method."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(verbose=False)

        # Create simple functions
        def func1(x):
            return x + 1

        def func2(x):
            return x * 2

        functions = nn.Sequential(nn.Linear(10, 10), nn.ReLU())

        with patch("torch.utils.checkpoint.checkpoint_sequential") as mock_checkpoint:
            mock_checkpoint.return_value = torch.randn(4, 10)

            input_tensor = torch.randn(4, 10, requires_grad=True)
            _result = optimizer.checkpoint_sequential(functions, 2, input_tensor)

            mock_checkpoint.assert_called_once()


# =============================================================================
# MEMORY MONITORING TESTS
# =============================================================================


class TestMemoryMonitoring:
    """Test memory monitoring functionality."""

    def test_get_memory_stats_cpu(self):
        """Test get_memory_stats returns message for CPU device."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(device=torch.device("cpu"), verbose=False)

        stats = optimizer.get_memory_stats()

        assert "device" in stats
        assert "message" in stats
        assert "only available for CUDA" in stats["message"]

    def test_get_memory_stats_cuda(self, mock_cuda_device):
        """Test get_memory_stats returns proper stats for CUDA device."""
        optimizer = MemoryOptimizer(verbose=False)
        stats = optimizer.get_memory_stats()

        assert "device" in stats
        assert "allocated" in stats
        assert "reserved" in stats
        assert "max_allocated" in stats
        assert "max_reserved" in stats
        assert "total" in stats
        assert "allocated_gb" in stats
        assert "reserved_gb" in stats
        assert "utilization" in stats

    def test_get_memory_stats_values(self, mock_cuda_device):
        """Test get_memory_stats returns expected values."""
        optimizer = MemoryOptimizer(verbose=False)
        stats = optimizer.get_memory_stats()

        assert stats["allocated"] == 1024**3
        assert stats["reserved"] == 2 * 1024**3
        assert stats["total"] == 8 * 1024**3
        assert stats["allocated_gb"] == 1.0
        assert stats["reserved_gb"] == 2.0
        assert stats["total_gb"] == 8.0

    def test_get_memory_stats_utilization(self, mock_cuda_device):
        """Test get_memory_stats calculates utilization correctly."""
        optimizer = MemoryOptimizer(verbose=False)
        stats = optimizer.get_memory_stats()

        # allocated / total * 100 = 1GB / 8GB * 100 = 12.5%
        assert stats["utilization"] == 12.5

    def test_get_memory_summary_cpu(self):
        """Test get_memory_summary returns message for CPU device."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(device=torch.device("cpu"), verbose=False)

        summary = optimizer.get_memory_summary()

        assert "only available for CUDA" in summary

    def test_get_memory_summary_cuda(self, mock_cuda_device):
        """Test get_memory_summary returns formatted string for CUDA."""
        optimizer = MemoryOptimizer(verbose=False)
        summary = optimizer.get_memory_summary()

        assert isinstance(summary, str)
        assert "Memory Summary" in summary
        assert "Allocated:" in summary
        assert "Reserved:" in summary
        assert "Peak:" in summary
        assert "Total:" in summary
        assert "Usage:" in summary

    def test_reset_peak_memory_stats_cuda(self):
        """Test reset_peak_memory_stats calls CUDA reset."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.reset_peak_memory_stats") as mock_reset:
                optimizer = MemoryOptimizer(verbose=False)
                optimizer.reset_peak_memory_stats()

                mock_reset.assert_called_once()

    def test_reset_peak_memory_stats_cpu(self):
        """Test reset_peak_memory_stats is no-op for CPU."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(device=torch.device("cpu"), verbose=False)

        # Should not raise
        optimizer.reset_peak_memory_stats()

    def test_reset_peak_memory_stats_logs(self, caplog):
        """Test reset_peak_memory_stats logs when verbose."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.reset_peak_memory_stats"):
                with caplog.at_level(logging.DEBUG):
                    optimizer = MemoryOptimizer(verbose=True)
                    optimizer.reset_peak_memory_stats()

        assert "Reset peak memory statistics" in caplog.text


# =============================================================================
# CACHE MANAGEMENT TESTS
# =============================================================================


class TestCacheManagement:
    """Test cache management functionality."""

    def test_empty_cache_cuda(self):
        """Test empty_cache calls CUDA empty_cache."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.empty_cache") as mock_empty:
                optimizer = MemoryOptimizer(verbose=False)
                optimizer.empty_cache()

                mock_empty.assert_called_once()

    def test_empty_cache_cpu(self):
        """Test empty_cache is no-op for CPU."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(device=torch.device("cpu"), verbose=False)

        # Should not raise
        optimizer.empty_cache()

    def test_empty_cache_logs(self, caplog):
        """Test empty_cache logs when verbose."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.empty_cache"):
                with caplog.at_level(logging.DEBUG):
                    optimizer = MemoryOptimizer(verbose=True)
                    optimizer.empty_cache()

        assert "Emptied CUDA cache" in caplog.text

    def test_run_garbage_collection(self):
        """Test run_garbage_collection calls gc.collect."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(device=torch.device("cpu"), verbose=False)

        with patch("gc.collect") as mock_gc:
            optimizer.run_garbage_collection()
            mock_gc.assert_called_once()

    def test_run_garbage_collection_cuda(self):
        """Test run_garbage_collection also empties CUDA cache."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.empty_cache") as mock_empty:
                with patch("gc.collect"):
                    optimizer = MemoryOptimizer(verbose=False)
                    optimizer.run_garbage_collection()

                    mock_empty.assert_called_once()

    def test_run_garbage_collection_logs(self, caplog):
        """Test run_garbage_collection logs when verbose."""
        with patch("torch.cuda.is_available", return_value=False), patch("gc.collect"):
            with caplog.at_level(logging.DEBUG):
                optimizer = MemoryOptimizer(device=torch.device("cpu"), verbose=True)
                optimizer.run_garbage_collection()

        assert "Ran garbage collection" in caplog.text


# =============================================================================
# STEP-BASED OPTIMIZATION TESTS
# =============================================================================


class TestStepOptimization:
    """Test step-based optimization functionality."""

    def test_step_increments_counter(self):
        """Test step method increments step counter."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(verbose=False)

        assert optimizer._step_count == 0
        optimizer.step()
        assert optimizer._step_count == 1
        optimizer.step()
        assert optimizer._step_count == 2

    def test_step_empty_cache_interval(self):
        """Test step method triggers cache clearing at interval."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.empty_cache") as mock_empty:
                optimizer = MemoryOptimizer(empty_cache_interval=5, verbose=False)

                mock_empty.reset_mock()

                for _i in range(10):
                    optimizer.step()

                # Should be called at step 5 and 10
                assert mock_empty.call_count == 2

    def test_step_garbage_collect_interval(self):
        """Test step method triggers garbage collection at interval."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(
                garbage_collect_interval=3, device=torch.device("cpu"), verbose=False
            )

        with patch.object(optimizer, "run_garbage_collection") as mock_gc:
            for _i in range(9):
                optimizer.step()

            # Should be called at step 3, 6, 9
            assert mock_gc.call_count == 3

    def test_step_no_cache_clear_when_interval_zero(self):
        """Test step method doesn't clear cache when interval is 0."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.empty_cache") as mock_empty:
                optimizer = MemoryOptimizer(empty_cache_interval=0, verbose=False)

                mock_empty.reset_mock()

                for _i in range(10):
                    optimizer.step()

                mock_empty.assert_not_called()

    def test_step_no_gc_when_interval_zero(self):
        """Test step method doesn't run GC when interval is 0."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(
                garbage_collect_interval=0, device=torch.device("cpu"), verbose=False
            )

        with patch.object(optimizer, "run_garbage_collection") as mock_gc:
            for _i in range(10):
                optimizer.step()

            mock_gc.assert_not_called()

    def test_check_memory_usage_cpu(self):
        """Test check_memory_usage returns True for CPU."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(device=torch.device("cpu"), verbose=False)

        result = optimizer.check_memory_usage(threshold=0.9)
        assert result is True

    def test_check_memory_usage_below_threshold(self, mock_cuda_device):
        """Test check_memory_usage returns True when below threshold."""
        optimizer = MemoryOptimizer(verbose=False)

        # Utilization is 12.5%, threshold is 90%
        result = optimizer.check_memory_usage(threshold=0.9)
        assert result is True

    def test_check_memory_usage_above_threshold(self):
        """Test check_memory_usage returns False when above threshold."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.synchronize"):
                with patch("torch.cuda.memory_allocated", return_value=9 * 1024**3):
                    with patch("torch.cuda.memory_reserved", return_value=9 * 1024**3):
                        with patch("torch.cuda.max_memory_allocated", return_value=9 * 1024**3):
                            with patch("torch.cuda.max_memory_reserved", return_value=9 * 1024**3):
                                mock_props = MagicMock()
                                mock_props.total_memory = 10 * 1024**3
                                with patch(
                                    "torch.cuda.get_device_properties", return_value=mock_props
                                ):
                                    optimizer = MemoryOptimizer(verbose=False)

                                    # Utilization is 90%, threshold is 80%
                                    result = optimizer.check_memory_usage(threshold=0.8)
                                    assert result is False

    def test_check_memory_usage_logs_warning(self, caplog):
        """Test check_memory_usage logs warning when above threshold."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.synchronize"):
                with patch("torch.cuda.memory_allocated", return_value=9 * 1024**3):
                    with patch("torch.cuda.memory_reserved", return_value=9 * 1024**3):
                        with patch("torch.cuda.max_memory_allocated", return_value=9 * 1024**3):
                            with patch("torch.cuda.max_memory_reserved", return_value=9 * 1024**3):
                                mock_props = MagicMock()
                                mock_props.total_memory = 10 * 1024**3
                                with (
                                    patch(
                                        "torch.cuda.get_device_properties", return_value=mock_props
                                    ),
                                    caplog.at_level(logging.WARNING),
                                ):
                                    optimizer = MemoryOptimizer(verbose=True)
                                    optimizer.check_memory_usage(threshold=0.8)

        assert "Memory usage high" in caplog.text


# =============================================================================
# MEMORY PROFILING TESTS
# =============================================================================


class TestMemoryProfiling:
    """Test memory profiling functionality."""

    def test_profile_memory_cpu(self):
        """Test profile_memory yields immediately for CPU."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(device=torch.device("cpu"), verbose=False)

        with optimizer.profile_memory():
            tensor = torch.randn(10)

        assert tensor is not None

    def test_profile_memory_cuda(self):
        """Test profile_memory uses profiler for CUDA."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.synchronize"):
                with patch("torch.cuda.reset_peak_memory_stats"):
                    with patch("torch.profiler.profile") as mock_profiler:
                        mock_context = MagicMock()
                        mock_profiler.return_value.__enter__ = Mock(return_value=mock_context)
                        mock_profiler.return_value.__exit__ = Mock(return_value=None)

                        optimizer = MemoryOptimizer(verbose=False)

                        with optimizer.profile_memory():
                            pass

                        mock_profiler.assert_called_once()

    def test_profile_memory_record_shapes(self):
        """Test profile_memory with record_shapes parameter."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.synchronize"):
                with patch("torch.cuda.reset_peak_memory_stats"):
                    with patch("torch.profiler.profile") as mock_profiler:
                        mock_context = MagicMock()
                        mock_profiler.return_value.__enter__ = Mock(return_value=mock_context)
                        mock_profiler.return_value.__exit__ = Mock(return_value=None)

                        optimizer = MemoryOptimizer(verbose=False)

                        with optimizer.profile_memory(record_shapes=True):
                            pass

                        call_kwargs = mock_profiler.call_args[1]
                        assert call_kwargs["record_shapes"] is True

    def test_profile_memory_with_stack(self):
        """Test profile_memory with with_stack parameter."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.synchronize"):
                with patch("torch.cuda.reset_peak_memory_stats"):
                    with patch("torch.profiler.profile") as mock_profiler:
                        mock_context = MagicMock()
                        mock_profiler.return_value.__enter__ = Mock(return_value=mock_context)
                        mock_profiler.return_value.__exit__ = Mock(return_value=None)

                        optimizer = MemoryOptimizer(verbose=False)

                        with optimizer.profile_memory(with_stack=True):
                            pass

                        call_kwargs = mock_profiler.call_args[1]
                        assert call_kwargs["with_stack"] is True

    def test_get_memory_snapshot_cpu(self):
        """Test get_memory_snapshot returns message for CPU."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(device=torch.device("cpu"), verbose=False)

        snapshot = optimizer.get_memory_snapshot()

        assert "message" in snapshot
        assert "only available for CUDA" in snapshot["message"]

    def test_get_memory_snapshot_cuda(self):
        """Test get_memory_snapshot returns snapshot for CUDA."""
        with patch("torch.cuda.is_available", return_value=True):
            mock_snapshot = [{"address": 0x1000, "size": 1024}]
            with patch("torch.cuda.memory_snapshot", return_value=mock_snapshot):
                mock_event = MagicMock()
                with patch("torch.cuda.Event", return_value=mock_event):
                    optimizer = MemoryOptimizer(verbose=False)
                    snapshot = optimizer.get_memory_snapshot()

        assert "device" in snapshot
        assert "snapshot" in snapshot
        assert "num_allocations" in snapshot
        assert "timestamp" in snapshot
        assert snapshot["num_allocations"] == 1


# =============================================================================
# DATALOADER OPTIMIZATION TESTS
# =============================================================================


class TestDataLoaderOptimization:
    """Test DataLoader optimization functionality."""

    def test_optimize_dataloader_basic(self, mock_dataloader):
        """Test optimize_dataloader creates new optimized DataLoader."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(verbose=False)

        with patch("torch.utils.data.DataLoader") as mock_dl_class:
            mock_dl_class.return_value = MagicMock()
            _result = optimizer.optimize_dataloader(mock_dataloader)

            mock_dl_class.assert_called_once()

    def test_optimize_dataloader_with_num_workers(self, mock_dataloader):
        """Test optimize_dataloader with custom num_workers."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(verbose=False)

        with patch("torch.utils.data.DataLoader") as mock_dl_class:
            mock_dl_class.return_value = MagicMock()
            _result = optimizer.optimize_dataloader(mock_dataloader, num_workers=8)

            call_kwargs = mock_dl_class.call_args[1]
            assert call_kwargs["num_workers"] == 8

    def test_optimize_dataloader_auto_num_workers(self, mock_dataloader):
        """Test optimize_dataloader auto-selects num_workers."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(verbose=False)

        with patch("torch.get_num_threads", return_value=8):
            with patch("torch.utils.data.DataLoader") as mock_dl_class:
                mock_dl_class.return_value = MagicMock()
                _result = optimizer.optimize_dataloader(mock_dataloader)

                call_kwargs = mock_dl_class.call_args[1]
                # min(4, 8) = 4
                assert call_kwargs["num_workers"] == 4

    def test_optimize_dataloader_prefetch_factor(self, mock_dataloader):
        """Test optimize_dataloader with custom prefetch_factor."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(verbose=False)

        with patch("torch.utils.data.DataLoader") as mock_dl_class:
            mock_dl_class.return_value = MagicMock()
            _result = optimizer.optimize_dataloader(mock_dataloader, prefetch_factor=4)

            call_kwargs = mock_dl_class.call_args[1]
            assert call_kwargs["prefetch_factor"] == 4

    def test_optimize_dataloader_pin_memory(self, mock_dataloader):
        """Test optimize_dataloader uses config pin_memory."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(pin_memory=True, verbose=False)

        with patch("torch.utils.data.DataLoader") as mock_dl_class:
            mock_dl_class.return_value = MagicMock()
            _result = optimizer.optimize_dataloader(mock_dataloader)

            call_kwargs = mock_dl_class.call_args[1]
            assert call_kwargs["pin_memory"] is True

    def test_optimize_dataloader_preserves_dataset(self, mock_dataloader):
        """Test optimize_dataloader preserves original dataset."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(verbose=False)

        with patch("torch.utils.data.DataLoader") as mock_dl_class:
            mock_dl_class.return_value = MagicMock()
            _result = optimizer.optimize_dataloader(mock_dataloader)

            call_args = mock_dl_class.call_args[0]
            assert call_args[0] is mock_dataloader.dataset

    def test_optimize_dataloader_preserves_batch_size(self, mock_dataloader):
        """Test optimize_dataloader preserves original batch_size."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(verbose=False)

        with patch("torch.utils.data.DataLoader") as mock_dl_class:
            mock_dl_class.return_value = MagicMock()
            _result = optimizer.optimize_dataloader(mock_dataloader)

            call_kwargs = mock_dl_class.call_args[1]
            assert call_kwargs["batch_size"] == 32

    def test_optimize_dataloader_persistent_workers(self, mock_dataloader):
        """Test optimize_dataloader sets persistent_workers correctly."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(verbose=False)

        with patch("torch.utils.data.DataLoader") as mock_dl_class:
            mock_dl_class.return_value = MagicMock()

            # With num_workers > 0
            _result = optimizer.optimize_dataloader(mock_dataloader, num_workers=4)
            call_kwargs = mock_dl_class.call_args[1]
            assert call_kwargs["persistent_workers"] is True

            # With num_workers = 0
            _result = optimizer.optimize_dataloader(mock_dataloader, num_workers=0)
            call_kwargs = mock_dl_class.call_args[1]
            assert call_kwargs["persistent_workers"] is False

    def test_optimize_dataloader_logs(self, caplog, mock_dataloader):
        """Test optimize_dataloader logs when verbose."""
        with patch("torch.cuda.is_available", return_value=False):
            with caplog.at_level(logging.INFO):
                optimizer = MemoryOptimizer(verbose=True)

        with patch("torch.utils.data.DataLoader") as mock_dl_class:
            mock_dl_class.return_value = MagicMock()
            with caplog.at_level(logging.INFO):
                _result = optimizer.optimize_dataloader(mock_dataloader)

        assert "Optimized DataLoader" in caplog.text


# =============================================================================
# MEMORY LEAK DETECTION TESTS
# =============================================================================


class TestMemoryLeakDetection:
    """Test memory leak detection functionality."""

    def test_detect_memory_leaks_cpu(self, simple_model, sample_input):
        """Test detect_memory_leaks returns message for CPU."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(device=torch.device("cpu"), verbose=False)

        result = optimizer.detect_memory_leaks(simple_model, sample_input)

        assert "message" in result
        assert "only available for CUDA" in result["message"]

    def test_detect_memory_leaks_cuda(self):
        """Test detect_memory_leaks performs leak detection for CUDA."""
        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval = Mock(return_value=mock_model)
        mock_model.__call__ = Mock(return_value=torch.randn(4, 1))

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.synchronize"):
                with patch("torch.cuda.reset_peak_memory_stats"):
                    with patch("torch.cuda.memory_allocated", return_value=1024**3):
                        with patch("torch.cuda.empty_cache"):
                            optimizer = MemoryOptimizer(verbose=False)

                            dummy_input = torch.randn(4, 10)
                            result = optimizer.detect_memory_leaks(
                                mock_model, dummy_input, num_iterations=5
                            )

        assert "iterations" in result
        assert "avg_usage_mb" in result
        assert "max_usage_mb" in result
        assert "min_usage_mb" in result
        assert "is_leaking" in result
        assert "memory_usage_trend" in result
        assert result["iterations"] == 5

    def test_detect_memory_leaks_sets_eval_mode(self):
        """Test detect_memory_leaks sets model to eval mode."""
        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval = Mock(return_value=mock_model)
        mock_model.__call__ = Mock(return_value=torch.randn(4, 1))

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.synchronize"):
                with patch("torch.cuda.reset_peak_memory_stats"):
                    with patch("torch.cuda.memory_allocated", return_value=1024**3):
                        with patch("torch.cuda.empty_cache"):
                            optimizer = MemoryOptimizer(verbose=False)

                            dummy_input = torch.randn(4, 10)
                            optimizer.detect_memory_leaks(mock_model, dummy_input, num_iterations=3)

        mock_model.eval.assert_called_once()

    def test_detect_memory_leaks_no_leak(self):
        """Test detect_memory_leaks detects no leak when memory stable."""
        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval = Mock(return_value=mock_model)
        mock_model.__call__ = Mock(return_value=torch.randn(4, 1))

        # Return same memory allocation each time (no leak)
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.synchronize"):
                with patch("torch.cuda.reset_peak_memory_stats"):
                    with patch("torch.cuda.memory_allocated", return_value=1024**3):
                        with patch("torch.cuda.empty_cache"):
                            optimizer = MemoryOptimizer(verbose=False)

                            dummy_input = torch.randn(4, 10)
                            result = optimizer.detect_memory_leaks(
                                mock_model, dummy_input, num_iterations=5
                            )

        # When memory is stable, is_leaking should be False
        # (0 - 0) * 1.1 = 0, so not leaking
        assert result["is_leaking"] is False


# =============================================================================
# PRINT MEMORY SUMMARY TESTS
# =============================================================================


class TestPrintMemorySummary:
    """Test print_memory_summary method."""

    def test_print_memory_summary_cpu(self, capsys):
        """Test print_memory_summary for CPU device."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(device=torch.device("cpu"), verbose=False)

        optimizer.print_memory_summary()

        captured = capsys.readouterr()
        assert "Memory Optimization Summary" in captured.out
        assert "Device:" in captured.out
        assert "Mixed Precision:" in captured.out

    def test_print_memory_summary_cuda(self, capsys, mock_cuda_device):
        """Test print_memory_summary for CUDA device."""
        optimizer = MemoryOptimizer(verbose=False)

        optimizer.print_memory_summary()

        captured = capsys.readouterr()
        assert "Memory Optimization Summary" in captured.out
        assert "Memory Summary" in captured.out

    def test_print_memory_summary_shows_precision(self, capsys):
        """Test print_memory_summary shows precision."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(precision="fp16", device=torch.device("cpu"), verbose=False)

        optimizer.print_memory_summary()

        captured = capsys.readouterr()
        assert "Precision: fp16" in captured.out

    def test_print_memory_summary_shows_gradient_checkpointing(self, capsys):
        """Test print_memory_summary shows gradient checkpointing status."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(
                gradient_checkpointing=True, device=torch.device("cpu"), verbose=False
            )

        optimizer.print_memory_summary()

        captured = capsys.readouterr()
        assert "Gradient Checkpointing: True" in captured.out

    def test_print_memory_summary_shows_pin_memory(self, capsys):
        """Test print_memory_summary shows pin_memory status."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(pin_memory=True, device=torch.device("cpu"), verbose=False)

        optimizer.print_memory_summary()

        captured = capsys.readouterr()
        assert "Pin Memory: True" in captured.out


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestGetMemoryEfficientSettings:
    """Test get_memory_efficient_settings convenience function."""

    def test_get_settings_small(self):
        """Test get_memory_efficient_settings for small model."""
        device = torch.device("cpu")
        settings = get_memory_efficient_settings(device, model_size="small")

        assert settings["mixed_precision"] is False
        assert settings["gradient_checkpointing"] is False
        assert settings["empty_cache_interval"] == 0
        assert settings["garbage_collect_interval"] == 0

    def test_get_settings_medium(self):
        """Test get_memory_efficient_settings for medium model."""
        device = torch.device("cpu")
        settings = get_memory_efficient_settings(device, model_size="medium")

        assert settings["mixed_precision"] is True
        assert settings["gradient_checkpointing"] is False
        assert settings["empty_cache_interval"] == 100
        assert settings["garbage_collect_interval"] == 0

    def test_get_settings_large(self):
        """Test get_memory_efficient_settings for large model."""
        device = torch.device("cpu")
        settings = get_memory_efficient_settings(device, model_size="large")

        assert settings["mixed_precision"] is True
        assert settings["gradient_checkpointing"] is True
        assert settings["empty_cache_interval"] == 50
        assert settings["garbage_collect_interval"] == 200

    def test_get_settings_xlarge(self):
        """Test get_memory_efficient_settings for xlarge model."""
        device = torch.device("cpu")
        settings = get_memory_efficient_settings(device, model_size="xlarge")

        assert settings["mixed_precision"] is True
        assert settings["gradient_checkpointing"] is True
        assert settings["empty_cache_interval"] == 10
        assert settings["garbage_collect_interval"] == 50

    def test_get_settings_unknown_defaults_to_medium(self):
        """Test get_memory_efficient_settings defaults to medium for unknown size."""
        device = torch.device("cpu")
        settings = get_memory_efficient_settings(device, model_size="unknown")

        # Should return medium settings
        assert settings["mixed_precision"] is True
        assert settings["gradient_checkpointing"] is False

    def test_get_settings_returns_dict(self):
        """Test get_memory_efficient_settings returns dictionary."""
        device = torch.device("cpu")
        settings = get_memory_efficient_settings(device)

        assert isinstance(settings, dict)

    def test_get_settings_default_is_medium(self):
        """Test get_memory_efficient_settings default is medium."""
        device = torch.device("cpu")
        settings = get_memory_efficient_settings(device)

        assert settings["mixed_precision"] is True
        assert settings["empty_cache_interval"] == 100


class TestEstimateModelMemory:
    """Test estimate_model_memory convenience function."""

    def test_estimate_memory_simple_model(self, simple_model):
        """Test estimate_model_memory with simple model."""
        estimates = estimate_model_memory(simple_model, input_size=(10,), batch_size=1)

        assert "parameters_mb" in estimates
        assert "activations_mb" in estimates
        assert "gradients_mb" in estimates
        assert "optimizer_mb" in estimates
        assert "total_mb" in estimates
        assert "total_gb" in estimates

    def test_estimate_memory_fp16(self, simple_model):
        """Test estimate_model_memory with FP16 precision."""
        estimates_fp16 = estimate_model_memory(
            simple_model, input_size=(10,), batch_size=1, precision="fp16"
        )

        estimates_fp32 = estimate_model_memory(
            simple_model, input_size=(10,), batch_size=1, precision="fp32"
        )

        # FP16 activations should use less memory
        assert estimates_fp16["activations_mb"] < estimates_fp32["activations_mb"]

    def test_estimate_memory_batch_size_scaling(self, simple_model):
        """Test estimate_model_memory scales with batch size."""
        estimates_batch_1 = estimate_model_memory(simple_model, input_size=(10,), batch_size=1)

        estimates_batch_8 = estimate_model_memory(simple_model, input_size=(10,), batch_size=8)

        # Activation memory should scale with batch size
        assert estimates_batch_8["activations_mb"] > estimates_batch_1["activations_mb"]

    def test_estimate_memory_positive_values(self, simple_model):
        """Test estimate_model_memory returns positive values."""
        estimates = estimate_model_memory(simple_model, input_size=(10,), batch_size=4)

        assert estimates["parameters_mb"] >= 0
        assert estimates["activations_mb"] >= 0
        assert estimates["gradients_mb"] >= 0
        assert estimates["optimizer_mb"] >= 0
        assert estimates["total_mb"] >= 0
        assert estimates["total_gb"] >= 0

    def test_estimate_memory_total_is_sum(self, simple_model):
        """Test estimate_model_memory total is sum of components."""
        estimates = estimate_model_memory(simple_model, input_size=(10,), batch_size=4)

        # Convert to bytes for comparison
        expected_total = (
            estimates["parameters_mb"]
            + estimates["activations_mb"]
            + estimates["gradients_mb"]
            + estimates["optimizer_mb"]
        )

        assert abs(estimates["total_mb"] - expected_total) < 0.001

    def test_estimate_memory_gb_conversion(self, simple_model):
        """Test estimate_model_memory GB conversion is correct."""
        estimates = estimate_model_memory(simple_model, input_size=(10,), batch_size=4)

        expected_gb = estimates["total_mb"] / 1024
        assert abs(estimates["total_gb"] - expected_gb) < 0.001


# =============================================================================
# EXCEPTION TESTS
# =============================================================================


class TestExceptions:
    """Test exception classes."""

    def test_model_error_is_exception(self):
        """Test ModelError is an Exception."""
        assert issubclass(ModelError, Exception)

    def test_hardware_error_is_model_error(self):
        """Test HardwareError is a ModelError."""
        assert issubclass(HardwareError, ModelError)

    def test_vqm_memory_error_is_hardware_error(self):
        """Test VQMMemoryError is a HardwareError."""
        assert issubclass(VQMMemoryError, HardwareError)

    def test_model_error_can_be_raised(self):
        """Test ModelError can be raised with message."""
        with pytest.raises(ModelError) as exc_info:
            raise ModelError("Test error message")

        assert "Test error message" in str(exc_info.value)

    def test_hardware_error_can_be_raised(self):
        """Test HardwareError can be raised with message."""
        with pytest.raises(HardwareError) as exc_info:
            raise HardwareError("Hardware failure")

        assert "Hardware failure" in str(exc_info.value)

    def test_vqm_memory_error_can_be_raised(self):
        """Test VQMMemoryError can be raised with message."""
        with pytest.raises(VQMMemoryError) as exc_info:
            raise VQMMemoryError("Out of memory")

        assert "Out of memory" in str(exc_info.value)

    def test_hardware_error_caught_as_model_error(self):
        """Test HardwareError can be caught as ModelError."""
        try:
            raise HardwareError("Test")
        except ModelError:
            pass  # Should be caught

    def test_vqm_memory_error_caught_as_hardware_error(self):
        """Test VQMMemoryError can be caught as HardwareError."""
        try:
            raise VQMMemoryError("Test")
        except HardwareError:
            pass  # Should be caught

    def test_vqm_memory_error_caught_as_model_error(self):
        """Test VQMMemoryError can be caught as ModelError."""
        try:
            raise VQMMemoryError("Test")
        except ModelError:
            pass  # Should be caught


# =============================================================================
# EDGE CASES AND ERROR SCENARIOS
# =============================================================================


class TestEdgeCasesAndErrors:
    """Test edge cases and error scenarios."""

    def test_optimizer_with_all_parameters(self):
        """Test MemoryOptimizer with all parameters set."""
        with patch("torch.cuda.is_available", return_value=True):
            optimizer = MemoryOptimizer(
                mixed_precision=True,
                precision="fp16",
                gradient_checkpointing=True,
                pin_memory=True,
                non_blocking=True,
                empty_cache_interval=50,
                garbage_collect_interval=100,
                max_memory_allocated=8192,
                device=torch.device("cuda"),
                verbose=True,
            )

        assert optimizer.config.mixed_precision is True
        assert optimizer.config.precision == "fp16"
        assert optimizer.config.gradient_checkpointing is True
        assert optimizer.config.pin_memory is True
        assert optimizer.config.non_blocking is True
        assert optimizer.config.empty_cache_interval == 50
        assert optimizer.config.garbage_collect_interval == 100
        assert optimizer.config.max_memory_allocated == 8192
        assert optimizer.verbose is True

    def test_config_to_dict_returns_new_dict_each_time(self):
        """Test to_dict returns a new dictionary each time."""
        config = MemoryConfig()

        dict1 = config.to_dict()
        dict2 = config.to_dict()

        assert dict1 is not dict2
        assert dict1 == dict2

    def test_autocast_non_cuda_non_cpu_device(self):
        """Test autocast with non-CUDA non-CPU device."""
        mock_device = MagicMock()
        mock_device.type = "xla"

        with patch("torch.cuda.is_available", return_value=False):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                optimizer = MemoryOptimizer(
                    mixed_precision=False,  # Disabled to avoid warning
                    device=mock_device,
                    verbose=False,
                )

        # Should just yield without any autocast
        with optimizer.autocast():
            tensor = torch.randn(10)

        assert tensor is not None

    def test_step_count_overflow_handling(self):
        """Test step count handles large numbers."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(verbose=False)

        optimizer._step_count = 1000000
        optimizer.step()

        assert optimizer._step_count == 1000001

    def test_empty_functions_list_checkpoint_sequential(self):
        """Test checkpoint_sequential with empty functions list."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(verbose=False)

        functions = nn.Sequential()

        with patch("torch.utils.checkpoint.checkpoint_sequential") as mock_checkpoint:
            mock_checkpoint.return_value = torch.randn(4, 10)

            input_tensor = torch.randn(4, 10, requires_grad=True)
            # This should handle empty sequential
            _result = optimizer.checkpoint_sequential(functions, 0, input_tensor)

    def test_memory_stats_zero_total_memory(self):
        """Test get_memory_stats handles zero total memory gracefully."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.synchronize"):
                with patch("torch.cuda.memory_allocated", return_value=0):
                    with patch("torch.cuda.memory_reserved", return_value=0):
                        with patch("torch.cuda.max_memory_allocated", return_value=0):
                            with patch("torch.cuda.max_memory_reserved", return_value=0):
                                mock_props = MagicMock()
                                mock_props.total_memory = 0
                                with patch(
                                    "torch.cuda.get_device_properties", return_value=mock_props
                                ):
                                    optimizer = MemoryOptimizer(verbose=False)
                                    stats = optimizer.get_memory_stats()

        # Should handle zero total gracefully
        assert stats["utilization"] == 0

    def test_gradient_checkpointing_model_without_forward(self):
        """Test gradient checkpointing with model that has no forward attribute."""
        mock_model = MagicMock()
        del mock_model.gradient_checkpointing_enable
        del mock_model.enable_gradient_checkpointing
        del mock_model.forward

        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(gradient_checkpointing=True, verbose=False)

        # Should return model unchanged
        result = optimizer.enable_gradient_checkpointing(mock_model)
        assert result is mock_model

    def test_scale_loss_with_zero_loss(self):
        """Test scale_loss with zero loss value."""
        with patch("torch.cuda.is_available", return_value=True):
            optimizer = MemoryOptimizer(mixed_precision=True, verbose=False)

        loss = torch.tensor(0.0, requires_grad=True)
        scaled_loss = optimizer.scale_loss(loss)

        assert scaled_loss is not None

    def test_check_memory_usage_default_threshold(self, mock_cuda_device):
        """Test check_memory_usage with default threshold."""
        optimizer = MemoryOptimizer(verbose=False)

        # Default threshold is 0.9
        result = optimizer.check_memory_usage()
        assert result is True  # 12.5% < 90%


# =============================================================================
# LOGGING TESTS
# =============================================================================


class TestLogging:
    """Test logging behavior."""

    def test_logger_name(self):
        """Test logger has correct name."""
        from milia_pipeline.models.acceleration import memory_optimization

        assert memory_optimization.logger.name == memory_optimization.__name__

    def test_module_loaded_log(self):
        """Test module loaded log exists."""
        from milia_pipeline.models.acceleration import memory_optimization

        assert memory_optimization.logger is not None

    def test_optimizer_init_logs_on_verbose(self, caplog):
        """Test MemoryOptimizer init logs on verbose."""
        with patch("torch.cuda.is_available", return_value=False):
            with caplog.at_level(logging.INFO):
                _optimizer = MemoryOptimizer(verbose=True)

        assert any("MemoryOptimizer initialized" in record.message for record in caplog.records)

    def test_gradient_checkpointing_logs(self, caplog, simple_model):
        """Test gradient checkpointing logs on verbose."""
        with patch("torch.cuda.is_available", return_value=False):
            with caplog.at_level(logging.INFO):
                optimizer = MemoryOptimizer(gradient_checkpointing=True, verbose=True)
                optimizer.enable_gradient_checkpointing(simple_model)

        assert any("gradient checkpointing" in record.message.lower() for record in caplog.records)

    def test_dataloader_optimization_logs(self, caplog, mock_dataloader):
        """Test dataloader optimization logs on verbose."""
        with patch("torch.cuda.is_available", return_value=False):
            with caplog.at_level(logging.INFO):
                optimizer = MemoryOptimizer(verbose=True)

        with patch("torch.utils.data.DataLoader") as mock_dl_class:
            mock_dl_class.return_value = MagicMock()
            with caplog.at_level(logging.INFO):
                optimizer.optimize_dataloader(mock_dataloader)

        assert "Optimized DataLoader" in caplog.text


# =============================================================================
# INTEGRATION-LIKE TESTS
# =============================================================================


class TestIntegrationScenarios:
    """Test integration-like scenarios."""

    def test_full_training_workflow_mocked(self, simple_model):
        """Test full training workflow with mocks."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(
                mixed_precision=False,
                gradient_checkpointing=True,
                empty_cache_interval=5,
                garbage_collect_interval=10,
                verbose=False,
            )

        # Enable gradient checkpointing
        _model = optimizer.enable_gradient_checkpointing(simple_model)

        # Simulate training steps
        for _i in range(15):
            with optimizer.autocast():
                pass  # Forward pass would go here
            optimizer.step()

        assert optimizer._step_count == 15

    def test_memory_config_roundtrip(self):
        """Test MemoryConfig to_dict and back."""
        original = MemoryConfig(
            mixed_precision=True,
            precision="bf16",
            gradient_checkpointing=True,
            pin_memory=True,
            non_blocking=False,
            empty_cache_interval=25,
            garbage_collect_interval=50,
            max_memory_allocated=4096,
            growth_interval=5,
        )

        config_dict = original.to_dict()

        restored = MemoryConfig(**config_dict)

        assert restored.mixed_precision == original.mixed_precision
        assert restored.precision == original.precision
        assert restored.gradient_checkpointing == original.gradient_checkpointing
        assert restored.pin_memory == original.pin_memory
        assert restored.non_blocking == original.non_blocking
        assert restored.empty_cache_interval == original.empty_cache_interval
        assert restored.garbage_collect_interval == original.garbage_collect_interval
        assert restored.max_memory_allocated == original.max_memory_allocated
        assert restored.growth_interval == original.growth_interval

    def test_optimizer_config_matches_init_params(self):
        """Test MemoryOptimizer config matches init parameters."""
        with patch("torch.cuda.is_available", return_value=False):
            optimizer = MemoryOptimizer(
                mixed_precision=True,
                precision="fp32",
                gradient_checkpointing=True,
                pin_memory=False,
                non_blocking=False,
                empty_cache_interval=100,
                garbage_collect_interval=200,
                max_memory_allocated=16384,
                verbose=False,
            )

        config_dict = optimizer.config.to_dict()

        # Note: mixed_precision may be disabled due to validation
        assert config_dict["precision"] == "fp32"
        assert config_dict["gradient_checkpointing"] is True
        assert config_dict["pin_memory"] is False
        assert config_dict["non_blocking"] is False
        assert config_dict["empty_cache_interval"] == 100
        assert config_dict["garbage_collect_interval"] == 200
        assert config_dict["max_memory_allocated"] == 16384

    def test_multiple_optimizers_independent(self):
        """Test multiple MemoryOptimizers are independent."""
        with patch("torch.cuda.is_available", return_value=False):
            opt1 = MemoryOptimizer(precision="fp16", empty_cache_interval=10, verbose=False)

            opt2 = MemoryOptimizer(precision="fp32", empty_cache_interval=20, verbose=False)

        opt1.step()
        opt1.step()
        opt1.step()

        assert opt1._step_count == 3
        assert opt2._step_count == 0
        assert opt1.config.precision == "fp16"
        assert opt2.config.precision == "fp32"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
