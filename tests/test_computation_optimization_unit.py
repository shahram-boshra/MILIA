#!/usr/bin/env python3
"""
Extended Production-Ready Unit Test Suite for computation_optimization.py Module

Comprehensive test coverage including:
- ComputationConfig Pydantic BaseModel (initialization, to_dict/model_dump, validation)
- Pydantic V2 migration compliance (model_dump, model_validate, mutability)
- ComputationOptimizer initialization and configuration validation
- Global optimization application (cuDNN, TF32, operator fusion)
- Model compilation with torch.compile (PyTorch 2.0+)
- JIT compilation and scripting (tracing, scripting, freezing)
- Memory format optimization (channels-last conversion)
- Main optimization entry point (optimize_model)
- Performance profiling (context manager)
- Benchmarking (model benchmarking, comparison)
- Kernel fusion (enable/disable)
- Graph optimization
- Convenience functions (get_optimal_settings, auto_optimize_model)
- Decorators (optimize_inference)
- Error handling and edge cases
- Device management (CPU/CUDA scenarios)
- Exception handling (OptimizationError)
- Parametrized tests for compile modes and backends
- Thread safety and concurrent access patterns

This is an EXTENDED PRODUCTION-READY test suite with comprehensive coverage
for enterprise-grade deployment.

Pydantic V2 Migration Notes:
    - Tests verify model_dump() backward compatibility via to_dict()
    - Tests verify BaseModel mutable behavior
    - Tests verify field validation and type coercion
    - NO sys.modules pollution - all mocks use test-level @patch decorators

Author: milia Team
Version: 1.1.0
"""
import sys
from pathlib import Path

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import pytest
import logging
import time
import warnings
from unittest.mock import Mock, patch, MagicMock, call, PropertyMock
from typing import Dict, Any, List
from contextlib import nullcontext
from functools import wraps
import gc

import torch
import torch.nn as nn
import torch.optim as optim


# Import the module under test
from milia_pipeline.models.acceleration.computation_optimization import (
    ComputationConfig,
    ComputationOptimizer,
    get_optimal_settings,
    auto_optimize_model,
    optimize_inference,
    OptimizationError,
    ModelError,
    HardwareError,
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
    model.state_dict = Mock(return_value={'param1': torch.tensor([1.0])})
    model.load_state_dict = Mock()
    model.parameters = Mock(return_value=[torch.nn.Parameter(torch.randn(3, 3))])
    
    # Mock forward pass
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
def conv_model():
    """Create a simple convolutional model for channels-last testing."""
    class ConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 1)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    return ConvModel()


@pytest.fixture
def sample_input():
    """Create sample input tensor."""
    return torch.randn(4, 10)


@pytest.fixture
def sample_conv_input():
    """Create sample input tensor for convolutional model."""
    return torch.randn(2, 3, 32, 32)


@pytest.fixture
def cpu_device():
    """Return CPU device."""
    return torch.device('cpu')


@pytest.fixture
def cuda_device():
    """Return CUDA device (mocked if unavailable)."""
    return torch.device('cuda')


@pytest.fixture
def default_config():
    """Create default ComputationConfig."""
    return ComputationConfig()


@pytest.fixture
def full_config():
    """Create fully configured ComputationConfig."""
    return ComputationConfig(
        compile_model=True,
        compile_mode="max-autotune",
        compile_dynamic=True,
        cudnn_benchmark=True,
        cudnn_deterministic=False,
        use_tf32=True,
        channels_last=True,
        fusion_strategy="aggressive",
        jit_compile=True,
        operator_fusion=True
    )


@pytest.fixture
def mock_jit_script_module():
    """Create a mock JIT ScriptModule."""
    mock_module = MagicMock(spec=torch.jit.ScriptModule)
    mock_module.graph = MagicMock()
    return mock_module


@pytest.fixture
def mock_profiler():
    """Create a mock profiler."""
    profiler = MagicMock()
    profiler.key_averages = Mock(return_value=MagicMock())
    profiler.key_averages.return_value.table = Mock(return_value="Profiling Table")
    return profiler


# =============================================================================
# COMPUTATION CONFIG TESTS
# =============================================================================

class TestComputationConfig:
    """
    Test ComputationConfig Pydantic BaseModel.
    
    Pydantic V2 Migration (Phase 10):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Tests verify model_dump() via backward-compatible to_dict()
        - Tests verify BaseModel behaviors and validation
    """
    
    def test_default_initialization(self):
        """Test ComputationConfig default values."""
        config = ComputationConfig()
        
        assert config.compile_model is False
        assert config.compile_mode == "default"
        assert config.compile_dynamic is False
        assert config.cudnn_benchmark is True
        assert config.cudnn_deterministic is False
        assert config.use_tf32 is True
        assert config.channels_last is False
        assert config.fusion_strategy == "default"
        assert config.jit_compile is False
        assert config.operator_fusion is True
    
    def test_custom_initialization(self):
        """Test ComputationConfig with custom values."""
        config = ComputationConfig(
            compile_model=True,
            compile_mode="max-autotune",
            compile_dynamic=True,
            cudnn_benchmark=False,
            cudnn_deterministic=True,
            use_tf32=False,
            channels_last=True,
            fusion_strategy="aggressive",
            jit_compile=True,
            operator_fusion=False
        )
        
        assert config.compile_model is True
        assert config.compile_mode == "max-autotune"
        assert config.compile_dynamic is True
        assert config.cudnn_benchmark is False
        assert config.cudnn_deterministic is True
        assert config.use_tf32 is False
        assert config.channels_last is True
        assert config.fusion_strategy == "aggressive"
        assert config.jit_compile is True
        assert config.operator_fusion is False
    
    def test_to_dict(self):
        """Test ComputationConfig to_dict method."""
        config = ComputationConfig(
            compile_model=True,
            compile_mode="reduce-overhead"
        )
        
        result = config.to_dict()
        
        assert isinstance(result, dict)
        assert result['compile_model'] is True
        assert result['compile_mode'] == "reduce-overhead"
        assert result['compile_dynamic'] is False
        assert result['cudnn_benchmark'] is True
        assert result['cudnn_deterministic'] is False
        assert result['use_tf32'] is True
        assert result['channels_last'] is False
        assert result['fusion_strategy'] == "default"
        assert result['jit_compile'] is False
        assert result['operator_fusion'] is True
    
    def test_to_dict_all_fields(self, full_config):
        """Test to_dict includes all fields."""
        result = full_config.to_dict()
        
        expected_keys = [
            'compile_model', 'compile_mode', 'compile_dynamic',
            'cudnn_benchmark', 'cudnn_deterministic', 'use_tf32',
            'channels_last', 'fusion_strategy', 'jit_compile', 'operator_fusion'
        ]
        
        for key in expected_keys:
            assert key in result
        
        assert len(result) == len(expected_keys)
    
    def test_compile_mode_values(self):
        """Test various compile_mode values."""
        modes = ["default", "reduce-overhead", "max-autotune"]
        
        for mode in modes:
            config = ComputationConfig(compile_mode=mode)
            assert config.compile_mode == mode
    
    def test_fusion_strategy_values(self):
        """Test various fusion_strategy values."""
        strategies = ["none", "default", "aggressive"]
        
        for strategy in strategies:
            config = ComputationConfig(fusion_strategy=strategy)
            assert config.fusion_strategy == strategy
    
    def test_pydantic_model_dump_method(self):
        """Test Pydantic V2 model_dump method is available and works."""
        config = ComputationConfig(
            compile_model=True,
            compile_mode="max-autotune"
        )
        
        # Verify model_dump exists and returns dict
        assert hasattr(config, 'model_dump')
        result = config.model_dump()
        assert isinstance(result, dict)
        assert result['compile_model'] is True
        assert result['compile_mode'] == "max-autotune"
    
    def test_to_dict_uses_model_dump(self):
        """Test that to_dict() is a wrapper around model_dump()."""
        config = ComputationConfig(
            compile_model=True,
            cudnn_benchmark=False
        )
        
        # Both methods should return equivalent results
        to_dict_result = config.to_dict()
        model_dump_result = config.model_dump()
        
        assert to_dict_result == model_dump_result
    
    def test_pydantic_basemodel_mutability(self):
        """Test that ComputationConfig is mutable (not frozen)."""
        config = ComputationConfig(compile_model=False)
        
        # Should be able to modify attributes
        config.compile_model = True
        assert config.compile_model is True
        
        config.compile_mode = "reduce-overhead"
        assert config.compile_mode == "reduce-overhead"
    
    def test_pydantic_model_validate_from_dict(self):
        """Test Pydantic V2 model_validate method for dict parsing."""
        data = {
            'compile_model': True,
            'compile_mode': 'max-autotune',
            'compile_dynamic': True,
            'cudnn_benchmark': False,
            'cudnn_deterministic': True,
            'use_tf32': False,
            'channels_last': True,
            'fusion_strategy': 'aggressive',
            'jit_compile': True,
            'operator_fusion': False
        }
        
        # Verify model_validate exists and works
        assert hasattr(ComputationConfig, 'model_validate')
        config = ComputationConfig.model_validate(data)
        
        assert config.compile_model is True
        assert config.compile_mode == 'max-autotune'
        assert config.fusion_strategy == 'aggressive'
    
    def test_pydantic_type_coercion(self):
        """Test Pydantic type coercion for boolean fields."""
        # Pydantic should coerce truthy/falsy values
        config = ComputationConfig(
            compile_model=1,  # Should coerce to True
            cudnn_benchmark=0  # Should coerce to False
        )
        
        assert config.compile_model is True
        assert config.cudnn_benchmark is False
    
    def test_config_copy_independence(self):
        """Test that model_dump creates independent copy."""
        config = ComputationConfig(compile_model=True)
        dict_copy = config.to_dict()
        
        # Modify the copy
        dict_copy['compile_model'] = False
        
        # Original should be unchanged
        assert config.compile_model is True
    
    def test_config_field_count(self):
        """Test that config has expected number of fields."""
        config = ComputationConfig()
        result = config.to_dict()
        
        expected_field_count = 10
        assert len(result) == expected_field_count, (
            f"Expected {expected_field_count} fields, got {len(result)}"
        )
    
    def test_config_partial_initialization(self):
        """Test config with partial field specification uses defaults."""
        config = ComputationConfig(compile_model=True)
        
        # Specified field
        assert config.compile_model is True
        
        # Default fields should have their default values
        assert config.compile_mode == "default"
        assert config.compile_dynamic is False
        assert config.cudnn_benchmark is True
        assert config.cudnn_deterministic is False
        assert config.use_tf32 is True
        assert config.channels_last is False
        assert config.fusion_strategy == "default"
        assert config.jit_compile is False
        assert config.operator_fusion is True


# =============================================================================
# PYDANTIC V2 VALIDATION TESTS
# =============================================================================

class TestPydanticV2Validation:
    """
    Test Pydantic V2 specific validation behaviors.
    
    These tests verify that the migration from dataclass to Pydantic BaseModel
    maintains expected behavior and adds proper validation.
    """
    
    def test_model_fields_attribute_exists(self):
        """Test Pydantic V2 model_fields attribute exists."""
        assert hasattr(ComputationConfig, 'model_fields')
        
        # Verify all expected fields are present
        expected_fields = {
            'compile_model', 'compile_mode', 'compile_dynamic',
            'cudnn_benchmark', 'cudnn_deterministic', 'use_tf32',
            'channels_last', 'fusion_strategy', 'jit_compile', 'operator_fusion'
        }
        
        assert set(ComputationConfig.model_fields.keys()) == expected_fields
    
    def test_model_json_schema_generation(self):
        """Test Pydantic V2 can generate JSON schema."""
        # Pydantic V2 models can generate JSON schema
        assert hasattr(ComputationConfig, 'model_json_schema')
        
        schema = ComputationConfig.model_json_schema()
        
        assert isinstance(schema, dict)
        assert 'properties' in schema
        assert 'compile_model' in schema['properties']
    
    def test_model_copy_method(self):
        """Test Pydantic V2 model_copy method."""
        original = ComputationConfig(compile_model=True, compile_mode="default")
        
        # Pydantic V2 uses model_copy instead of copy
        assert hasattr(original, 'model_copy')
        
        copied = original.model_copy()
        
        assert copied.compile_model == original.compile_model
        assert copied.compile_mode == original.compile_mode
        assert copied is not original
    
    def test_model_copy_with_update(self):
        """Test Pydantic V2 model_copy with update parameter."""
        original = ComputationConfig(compile_model=True, compile_mode="default")
        
        copied = original.model_copy(update={'compile_mode': 'max-autotune'})
        
        # Original unchanged
        assert original.compile_mode == "default"
        
        # Copy has updated value
        assert copied.compile_mode == "max-autotune"
        assert copied.compile_model is True  # Other fields preserved
    
    def test_model_dump_exclude(self):
        """Test Pydantic V2 model_dump with exclude parameter."""
        config = ComputationConfig(compile_model=True, compile_mode="default")
        
        result = config.model_dump(exclude={'compile_model'})
        
        assert 'compile_model' not in result
        assert 'compile_mode' in result
    
    def test_model_dump_include(self):
        """Test Pydantic V2 model_dump with include parameter."""
        config = ComputationConfig(compile_model=True, compile_mode="default")
        
        result = config.model_dump(include={'compile_model', 'compile_mode'})
        
        assert 'compile_model' in result
        assert 'compile_mode' in result
        assert len(result) == 2
    
    def test_string_field_accepts_string(self):
        """Test string fields accept string values properly."""
        config = ComputationConfig(
            compile_mode="reduce-overhead",
            fusion_strategy="aggressive"
        )
        
        assert config.compile_mode == "reduce-overhead"
        assert config.fusion_strategy == "aggressive"
    
    def test_config_repr(self):
        """Test config has a readable string representation."""
        config = ComputationConfig(compile_model=True)
        
        repr_str = repr(config)
        
        # Pydantic models have repr
        assert 'ComputationConfig' in repr_str
        assert 'compile_model=True' in repr_str
    
    def test_config_equality(self):
        """Test config equality comparison."""
        config1 = ComputationConfig(compile_model=True, compile_mode="default")
        config2 = ComputationConfig(compile_model=True, compile_mode="default")
        config3 = ComputationConfig(compile_model=False, compile_mode="default")
        
        assert config1 == config2
        assert config1 != config3
    
    def test_config_hash_not_supported_when_mutable(self):
        """Test that mutable Pydantic models are not hashable by default."""
        config = ComputationConfig()
        
        # Mutable Pydantic models should not be hashable
        # This is expected behavior for mutable objects
        with pytest.raises(TypeError):
            hash(config)


# =============================================================================
# COMPUTATION OPTIMIZER INITIALIZATION TESTS
# =============================================================================

class TestComputationOptimizerInitialization:
    """Test ComputationOptimizer initialization and configuration."""
    
    def test_minimal_initialization(self):
        """Test ComputationOptimizer initialization with minimal parameters."""
        with patch('torch.cuda.is_available', return_value=False):
            optimizer = ComputationOptimizer()
        
        assert optimizer.verbose is True
        assert optimizer.device == torch.device('cpu')
        assert optimizer._compiled_models == {}
        assert isinstance(optimizer.config, ComputationConfig)
    
    def test_full_initialization(self):
        """Test ComputationOptimizer initialization with all parameters."""
        device = torch.device('cpu')
        
        optimizer = ComputationOptimizer(
            compile_model=True,
            compile_mode="max-autotune",
            compile_dynamic=True,
            cudnn_benchmark=False,
            cudnn_deterministic=True,
            use_tf32=False,
            channels_last=True,
            fusion_strategy="aggressive",
            jit_compile=True,
            operator_fusion=False,
            device=device,
            verbose=False
        )
        
        assert optimizer.device == device
        assert optimizer.verbose is False
        assert optimizer.config.compile_model is True
        assert optimizer.config.compile_mode == "max-autotune"
        assert optimizer.config.compile_dynamic is True
        assert optimizer.config.cudnn_benchmark is False
        assert optimizer.config.cudnn_deterministic is True
        assert optimizer.config.use_tf32 is False
        assert optimizer.config.channels_last is True
        assert optimizer.config.fusion_strategy == "aggressive"
        assert optimizer.config.jit_compile is True
        assert optimizer.config.operator_fusion is False
    
    def test_auto_device_detection_cpu(self):
        """Test automatic device detection when CUDA is unavailable."""
        with patch('torch.cuda.is_available', return_value=False):
            optimizer = ComputationOptimizer(verbose=False)
        
        assert optimizer.device == torch.device('cpu')
    
    def test_auto_device_detection_cuda(self):
        """Test automatic device detection when CUDA is available."""
        with patch('torch.cuda.is_available', return_value=True):
            optimizer = ComputationOptimizer(verbose=False)
        
        assert optimizer.device == torch.device('cuda')
    
    def test_explicit_device_override(self):
        """Test explicit device override ignores auto-detection."""
        explicit_device = torch.device('cpu')
        
        with patch('torch.cuda.is_available', return_value=True):
            optimizer = ComputationOptimizer(device=explicit_device, verbose=False)
        
        assert optimizer.device == explicit_device
    
    def test_config_passed_correctly(self):
        """Test that config is constructed correctly from init params."""
        optimizer = ComputationOptimizer(
            compile_model=True,
            compile_mode="reduce-overhead",
            verbose=False,
            device=torch.device('cpu')
        )
        
        assert optimizer.config.compile_model is True
        assert optimizer.config.compile_mode == "reduce-overhead"
    
    def test_compiled_models_cache_initialized(self):
        """Test that compiled models cache is initialized empty."""
        with patch('torch.cuda.is_available', return_value=False):
            optimizer = ComputationOptimizer(verbose=False)
        
        assert optimizer._compiled_models == {}
        assert isinstance(optimizer._compiled_models, dict)
    
    def test_verbose_logging(self, caplog):
        """Test verbose logging during initialization."""
        with caplog.at_level(logging.INFO):
            with patch('torch.cuda.is_available', return_value=False):
                optimizer = ComputationOptimizer(verbose=True)
        
        assert "ComputationOptimizer initialized" in caplog.text
    
    def test_silent_initialization(self, caplog):
        """Test silent initialization when verbose=False."""
        with caplog.at_level(logging.INFO):
            with patch('torch.cuda.is_available', return_value=False):
                # Clear any existing logs
                caplog.clear()
                optimizer = ComputationOptimizer(verbose=False)
        
        # Check that no INFO logs about initialization were captured after clearing
        optimizer_init_logs = [r for r in caplog.records 
                               if "ComputationOptimizer initialized" in r.message]
        assert len(optimizer_init_logs) == 0


# =============================================================================
# GLOBAL OPTIMIZATIONS TESTS
# =============================================================================

class TestGlobalOptimizations:
    """Test global PyTorch optimization settings."""
    
    def test_apply_global_optimizations_cpu(self):
        """Test global optimizations on CPU device (skips CUDA-specific)."""
        device = torch.device('cpu')
        
        with patch.object(torch._C, '_jit_set_profiling_executor') as mock_exec:
            with patch.object(torch._C, '_jit_set_profiling_mode') as mock_mode:
                optimizer = ComputationOptimizer(
                    device=device,
                    operator_fusion=True,
                    verbose=False
                )
        
        mock_exec.assert_called_once_with(True)
        mock_mode.assert_called_once_with(True)
    
    def test_apply_global_optimizations_cuda_cudnn_benchmark(self):
        """Test cuDNN benchmark setting on CUDA device."""
        device = torch.device('cuda')
        
        # Store original value to restore later
        original_benchmark = torch.backends.cudnn.benchmark
        
        try:
            with patch.object(torch.backends.cudnn, 'is_available', return_value=True):
                optimizer = ComputationOptimizer(
                    device=device,
                    cudnn_benchmark=True,
                    verbose=False
                )
            
            # Verify the config has correct value
            assert optimizer.config.cudnn_benchmark is True
            # Verify the actual torch backend was set (when cudnn available)
            if torch.backends.cudnn.is_available():
                assert torch.backends.cudnn.benchmark is True
        finally:
            # Restore original value
            torch.backends.cudnn.benchmark = original_benchmark
    
    def test_apply_global_optimizations_cuda_deterministic(self):
        """Test cuDNN deterministic setting on CUDA device."""
        device = torch.device('cuda')
        
        with patch.object(torch.backends.cudnn, 'is_available', return_value=True):
            optimizer = ComputationOptimizer(
                device=device,
                cudnn_deterministic=True,
                verbose=False
            )
        
        assert optimizer.config.cudnn_deterministic is True
    
    def test_apply_global_optimizations_tf32(self):
        """Test TF32 settings on CUDA device."""
        device = torch.device('cuda')
        
        with patch.object(torch.backends.cudnn, 'is_available', return_value=True):
            # Check if the matmul attribute exists
            if hasattr(torch.backends.cuda, 'matmul'):
                optimizer = ComputationOptimizer(
                    device=device,
                    use_tf32=True,
                    verbose=False
                )
                assert optimizer.config.use_tf32 is True
    
    def test_operator_fusion_enabled(self):
        """Test operator fusion is enabled."""
        with patch.object(torch._C, '_jit_set_profiling_executor') as mock_exec:
            with patch.object(torch._C, '_jit_set_profiling_mode') as mock_mode:
                optimizer = ComputationOptimizer(
                    operator_fusion=True,
                    verbose=False,
                    device=torch.device('cpu')
                )
        
        mock_exec.assert_called_with(True)
        mock_mode.assert_called_with(True)
    
    def test_operator_fusion_disabled(self):
        """Test operator fusion is not called when disabled."""
        with patch.object(torch._C, '_jit_set_profiling_executor') as mock_exec:
            with patch.object(torch._C, '_jit_set_profiling_mode') as mock_mode:
                optimizer = ComputationOptimizer(
                    operator_fusion=False,
                    verbose=False,
                    device=torch.device('cpu')
                )
        
        mock_exec.assert_not_called()
        mock_mode.assert_not_called()


# =============================================================================
# MODEL COMPILATION TESTS (torch.compile)
# =============================================================================

class TestModelCompilation:
    """Test torch.compile functionality."""
    
    def test_compile_model_disabled_returns_original(self, simple_model):
        """Test that compile_model returns original when disabled."""
        optimizer = ComputationOptimizer(
            compile_model=False,
            verbose=False,
            device=torch.device('cpu')
        )
        
        result = optimizer.compile_model(simple_model)
        
        assert result is simple_model
    
    def test_compile_model_pytorch_version_check(self, simple_model):
        """Test warning when torch.compile not available."""
        optimizer = ComputationOptimizer(
            compile_model=True,
            verbose=False,
            device=torch.device('cpu')
        )
        
        # Mock torch.compile not existing
        with patch.object(torch, 'compile', None, create=True):
            if hasattr(torch, 'compile'):
                delattr(torch, 'compile')
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # Need to mock hasattr to return False for 'compile'
                original_hasattr = hasattr
                
                def mock_hasattr(obj, name):
                    if obj is torch and name == 'compile':
                        return False
                    return original_hasattr(obj, name)
                
                with patch('builtins.hasattr', mock_hasattr):
                    result = optimizer.compile_model(simple_model)
        
        assert result is simple_model
    
    @pytest.mark.skipif(
        not hasattr(torch, 'compile'),
        reason="torch.compile requires PyTorch 2.0+"
    )
    def test_compile_model_success(self, simple_model):
        """Test successful model compilation."""
        optimizer = ComputationOptimizer(
            compile_model=True,
            compile_mode="default",
            verbose=False,
            device=torch.device('cpu')
        )
        
        with patch.object(torch, 'compile', return_value=simple_model) as mock_compile:
            result = optimizer.compile_model(simple_model)
        
        mock_compile.assert_called_once()
        assert result is not None
    
    @pytest.mark.skipif(
        not hasattr(torch, 'compile'),
        reason="torch.compile requires PyTorch 2.0+"
    )
    def test_compile_model_with_mode_override(self, simple_model):
        """Test model compilation with mode override."""
        optimizer = ComputationOptimizer(
            compile_model=True,
            compile_mode="default",
            verbose=False,
            device=torch.device('cpu')
        )
        
        with patch.object(torch, 'compile', return_value=simple_model) as mock_compile:
            result = optimizer.compile_model(simple_model, mode="max-autotune")
        
        call_kwargs = mock_compile.call_args[1]
        assert call_kwargs['mode'] == "max-autotune"
    
    @pytest.mark.skipif(
        not hasattr(torch, 'compile'),
        reason="torch.compile requires PyTorch 2.0+"
    )
    def test_compile_model_with_dynamic_override(self, simple_model):
        """Test model compilation with dynamic shapes override."""
        optimizer = ComputationOptimizer(
            compile_model=True,
            compile_dynamic=False,
            verbose=False,
            device=torch.device('cpu')
        )
        
        with patch.object(torch, 'compile', return_value=simple_model) as mock_compile:
            result = optimizer.compile_model(simple_model, dynamic=True)
        
        call_kwargs = mock_compile.call_args[1]
        assert call_kwargs['dynamic'] is True
    
    @pytest.mark.skipif(
        not hasattr(torch, 'compile'),
        reason="torch.compile requires PyTorch 2.0+"
    )
    def test_compile_model_fullgraph(self, simple_model):
        """Test model compilation with fullgraph option."""
        optimizer = ComputationOptimizer(
            compile_model=True,
            verbose=False,
            device=torch.device('cpu')
        )
        
        with patch.object(torch, 'compile', return_value=simple_model) as mock_compile:
            result = optimizer.compile_model(simple_model, fullgraph=True)
        
        call_kwargs = mock_compile.call_args[1]
        assert call_kwargs['fullgraph'] is True
    
    @pytest.mark.skipif(
        not hasattr(torch, 'compile'),
        reason="torch.compile requires PyTorch 2.0+"
    )
    def test_compile_model_backend(self, simple_model):
        """Test model compilation with different backends."""
        optimizer = ComputationOptimizer(
            compile_model=True,
            verbose=False,
            device=torch.device('cpu')
        )
        
        backends = ["inductor", "aot_eager", "cudagraphs"]
        
        for backend in backends:
            with patch.object(torch, 'compile', return_value=simple_model) as mock_compile:
                result = optimizer.compile_model(simple_model, backend=backend)
            
            call_kwargs = mock_compile.call_args[1]
            assert call_kwargs['backend'] == backend
    
    @pytest.mark.skipif(
        not hasattr(torch, 'compile'),
        reason="torch.compile requires PyTorch 2.0+"
    )
    def test_compile_model_caches_result(self, simple_model):
        """Test that compiled model is cached."""
        optimizer = ComputationOptimizer(
            compile_model=True,
            verbose=False,
            device=torch.device('cpu')
        )
        
        compiled_mock = Mock()
        
        with patch.object(torch, 'compile', return_value=compiled_mock):
            result = optimizer.compile_model(simple_model)
        
        model_id = id(simple_model)
        assert model_id in optimizer._compiled_models
        assert optimizer._compiled_models[model_id] is compiled_mock
    
    @pytest.mark.skipif(
        not hasattr(torch, 'compile'),
        reason="torch.compile requires PyTorch 2.0+"
    )
    def test_compile_model_failure_raises_optimization_error(self, simple_model):
        """Test that compilation failure raises OptimizationError."""
        optimizer = ComputationOptimizer(
            compile_model=True,
            verbose=False,
            device=torch.device('cpu')
        )
        
        with patch.object(torch, 'compile', side_effect=Exception("Compile failed")):
            with pytest.raises(OptimizationError) as exc_info:
                optimizer.compile_model(simple_model)
        
        assert "Failed to compile model" in str(exc_info.value)
    
    def test_compile_model_logging(self, simple_model, caplog):
        """Test compile_model logging when verbose."""
        optimizer = ComputationOptimizer(
            compile_model=True,
            verbose=True,
            device=torch.device('cpu')
        )
        
        if hasattr(torch, 'compile'):
            with caplog.at_level(logging.INFO):
                with patch.object(torch, 'compile', return_value=simple_model):
                    optimizer.compile_model(simple_model)
            
            assert "Model compiled" in caplog.text
    
    @pytest.mark.parametrize("compile_mode", ["default", "reduce-overhead", "max-autotune"])
    @pytest.mark.skipif(
        not hasattr(torch, 'compile'),
        reason="torch.compile requires PyTorch 2.0+"
    )
    def test_compile_model_all_modes_parametrized(self, simple_model, compile_mode):
        """Test model compilation with all supported modes (parametrized)."""
        optimizer = ComputationOptimizer(
            compile_model=True,
            compile_mode=compile_mode,
            verbose=False,
            device=torch.device('cpu')
        )
        
        with patch.object(torch, 'compile', return_value=simple_model) as mock_compile:
            result = optimizer.compile_model(simple_model)
        
        call_kwargs = mock_compile.call_args[1]
        assert call_kwargs['mode'] == compile_mode
    
    @pytest.mark.parametrize("backend", ["inductor", "aot_eager", "cudagraphs"])
    @pytest.mark.skipif(
        not hasattr(torch, 'compile'),
        reason="torch.compile requires PyTorch 2.0+"
    )
    def test_compile_model_all_backends_parametrized(self, simple_model, backend):
        """Test model compilation with all supported backends (parametrized)."""
        optimizer = ComputationOptimizer(
            compile_model=True,
            verbose=False,
            device=torch.device('cpu')
        )
        
        with patch.object(torch, 'compile', return_value=simple_model) as mock_compile:
            result = optimizer.compile_model(simple_model, backend=backend)
        
        call_kwargs = mock_compile.call_args[1]
        assert call_kwargs['backend'] == backend
    
    @pytest.mark.parametrize("dynamic,fullgraph", [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ])
    @pytest.mark.skipif(
        not hasattr(torch, 'compile'),
        reason="torch.compile requires PyTorch 2.0+"
    )
    def test_compile_model_dynamic_fullgraph_combinations(self, simple_model, dynamic, fullgraph):
        """Test model compilation with dynamic and fullgraph combinations."""
        optimizer = ComputationOptimizer(
            compile_model=True,
            compile_dynamic=dynamic,
            verbose=False,
            device=torch.device('cpu')
        )
        
        with patch.object(torch, 'compile', return_value=simple_model) as mock_compile:
            result = optimizer.compile_model(simple_model, fullgraph=fullgraph)
        
        call_kwargs = mock_compile.call_args[1]
        assert call_kwargs['dynamic'] == dynamic
        assert call_kwargs['fullgraph'] == fullgraph


# =============================================================================
# JIT COMPILATION TESTS
# =============================================================================

class TestJITCompilation:
    """Test JIT compilation functionality."""
    
    def test_jit_script_model_disabled_returns_original(self, simple_model):
        """Test jit_script_model returns original when disabled."""
        optimizer = ComputationOptimizer(
            jit_compile=False,
            verbose=False,
            device=torch.device('cpu')
        )
        
        result = optimizer.jit_script_model(simple_model)
        
        assert result is simple_model
    
    def test_jit_script_model_without_example_inputs(self, simple_model):
        """Test JIT scripting without example inputs."""
        optimizer = ComputationOptimizer(
            jit_compile=True,
            verbose=False,
            device=torch.device('cpu')
        )
        
        mock_scripted = Mock(spec=torch.jit.ScriptModule)
        
        with patch.object(torch.jit, 'script', return_value=mock_scripted) as mock_script:
            with patch.object(
                torch.jit, 'optimize_for_inference', 
                return_value=mock_scripted
            ) as mock_optimize:
                result = optimizer.jit_script_model(simple_model)
        
        mock_script.assert_called_once_with(simple_model)
        mock_optimize.assert_called_once()
    
    def test_jit_script_model_with_example_inputs(self, simple_model, sample_input):
        """Test JIT tracing with example inputs."""
        optimizer = ComputationOptimizer(
            jit_compile=True,
            verbose=False,
            device=torch.device('cpu')
        )
        
        mock_traced = Mock(spec=torch.jit.ScriptModule)
        
        with patch.object(torch.jit, 'trace', return_value=mock_traced) as mock_trace:
            with patch.object(
                torch.jit, 'optimize_for_inference', 
                return_value=mock_traced
            ) as mock_optimize:
                result = optimizer.jit_script_model(
                    simple_model, 
                    example_inputs=(sample_input,)
                )
        
        mock_trace.assert_called_once()
        mock_optimize.assert_called_once()
    
    def test_jit_script_model_failure_returns_original(self, simple_model):
        """Test JIT scripting failure returns original model."""
        optimizer = ComputationOptimizer(
            jit_compile=True,
            verbose=False,
            device=torch.device('cpu')
        )
        
        with patch.object(torch.jit, 'script', side_effect=Exception("JIT failed")):
            result = optimizer.jit_script_model(simple_model)
        
        assert result is simple_model
    
    def test_jit_script_model_failure_logs_warning(self, simple_model, caplog):
        """Test JIT scripting failure logs warning."""
        optimizer = ComputationOptimizer(
            jit_compile=True,
            verbose=True,
            device=torch.device('cpu')
        )
        
        with caplog.at_level(logging.WARNING):
            with patch.object(torch.jit, 'script', side_effect=Exception("JIT failed")):
                result = optimizer.jit_script_model(simple_model)
        
        assert "JIT compilation failed" in caplog.text
    
    def test_jit_freeze_model_success(self, mock_jit_script_module):
        """Test successful model freezing."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        frozen_mock = Mock(spec=torch.jit.ScriptModule)
        
        with patch.object(torch.jit, 'freeze', return_value=frozen_mock) as mock_freeze:
            result = optimizer.jit_freeze_model(mock_jit_script_module)
        
        mock_freeze.assert_called_once_with(mock_jit_script_module)
        assert result is frozen_mock
    
    def test_jit_freeze_model_failure_returns_original(self, mock_jit_script_module):
        """Test model freezing failure returns original."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        with patch.object(torch.jit, 'freeze', side_effect=Exception("Freeze failed")):
            result = optimizer.jit_freeze_model(mock_jit_script_module)
        
        assert result is mock_jit_script_module
    
    def test_jit_freeze_model_logging(self, mock_jit_script_module, caplog):
        """Test model freezing logging."""
        optimizer = ComputationOptimizer(
            verbose=True,
            device=torch.device('cpu')
        )
        
        frozen_mock = Mock(spec=torch.jit.ScriptModule)
        
        with caplog.at_level(logging.INFO):
            with patch.object(torch.jit, 'freeze', return_value=frozen_mock):
                optimizer.jit_freeze_model(mock_jit_script_module)
        
        assert "Model frozen" in caplog.text
    
    def test_jit_script_model_trace_logging(self, simple_model, sample_input, caplog):
        """Test JIT tracing logs correctly when verbose."""
        optimizer = ComputationOptimizer(
            jit_compile=True,
            verbose=True,
            device=torch.device('cpu')
        )
        
        mock_traced = Mock(spec=torch.jit.ScriptModule)
        
        with caplog.at_level(logging.INFO):
            with patch.object(torch.jit, 'trace', return_value=mock_traced):
                with patch.object(torch.jit, 'optimize_for_inference', return_value=mock_traced):
                    optimizer.jit_script_model(simple_model, example_inputs=(sample_input,))
        
        assert "JIT traced" in caplog.text
    
    def test_jit_script_model_script_logging(self, simple_model, caplog):
        """Test JIT scripting logs correctly when verbose."""
        optimizer = ComputationOptimizer(
            jit_compile=True,
            verbose=True,
            device=torch.device('cpu')
        )
        
        mock_scripted = Mock(spec=torch.jit.ScriptModule)
        
        with caplog.at_level(logging.INFO):
            with patch.object(torch.jit, 'script', return_value=mock_scripted):
                with patch.object(torch.jit, 'optimize_for_inference', return_value=mock_scripted):
                    optimizer.jit_script_model(simple_model)
        
        assert "JIT scripted" in caplog.text
    
    def test_jit_freeze_model_failure_logging(self, mock_jit_script_module, caplog):
        """Test model freezing failure logs warning."""
        optimizer = ComputationOptimizer(
            verbose=True,
            device=torch.device('cpu')
        )
        
        with caplog.at_level(logging.WARNING):
            with patch.object(torch.jit, 'freeze', side_effect=Exception("Freeze failed")):
                optimizer.jit_freeze_model(mock_jit_script_module)
        
        assert "Model freezing failed" in caplog.text


# =============================================================================
# MEMORY FORMAT OPTIMIZATION TESTS
# =============================================================================

class TestMemoryFormatOptimization:
    """Test memory format (channels-last) optimization."""
    
    def test_convert_to_channels_last_disabled_returns_original(self, conv_model):
        """Test channels-last conversion returns original when disabled."""
        optimizer = ComputationOptimizer(
            channels_last=False,
            verbose=False,
            device=torch.device('cpu')
        )
        
        result = optimizer.convert_to_channels_last(conv_model)
        
        assert result is conv_model
    
    def test_convert_to_channels_last_enabled(self, conv_model):
        """Test channels-last conversion when enabled."""
        optimizer = ComputationOptimizer(
            channels_last=True,
            verbose=False,
            device=torch.device('cpu')
        )
        
        result = optimizer.convert_to_channels_last(conv_model)
        
        # The model should be converted
        assert result is not None
    
    def test_convert_to_channels_last_failure_returns_original(self):
        """Test channels-last conversion failure returns original."""
        optimizer = ComputationOptimizer(
            channels_last=True,
            verbose=False,
            device=torch.device('cpu')
        )
        
        # Create a mock model that raises on to()
        mock_model = Mock(spec=nn.Module)
        mock_model.to = Mock(side_effect=Exception("Conversion failed"))
        
        result = optimizer.convert_to_channels_last(mock_model)
        
        assert result is mock_model
    
    def test_convert_to_channels_last_logging(self, conv_model, caplog):
        """Test channels-last conversion logging."""
        optimizer = ComputationOptimizer(
            channels_last=True,
            verbose=True,
            device=torch.device('cpu')
        )
        
        with caplog.at_level(logging.INFO):
            optimizer.convert_to_channels_last(conv_model)
        
        assert "channels-last" in caplog.text.lower()


# =============================================================================
# MAIN OPTIMIZATION ENTRY POINT TESTS
# =============================================================================

class TestOptimizeModel:
    """Test main optimize_model entry point."""
    
    def test_optimize_model_moves_to_device(self, simple_model):
        """Test optimize_model moves model to device."""
        device = torch.device('cpu')
        optimizer = ComputationOptimizer(
            verbose=False,
            device=device
        )
        
        result = optimizer.optimize_model(simple_model)
        
        # Model should be on correct device
        for param in result.parameters():
            assert param.device == device
    
    def test_optimize_model_applies_channels_last(self, conv_model):
        """Test optimize_model applies channels-last when enabled."""
        optimizer = ComputationOptimizer(
            channels_last=True,
            verbose=False,
            device=torch.device('cpu')
        )
        
        with patch.object(
            optimizer, 'convert_to_channels_last', 
            return_value=conv_model
        ) as mock_convert:
            result = optimizer.optimize_model(conv_model)
        
        mock_convert.assert_called_once()
    
    def test_optimize_model_applies_jit_compile(self, simple_model):
        """Test optimize_model applies JIT compilation when enabled."""
        optimizer = ComputationOptimizer(
            jit_compile=True,
            verbose=False,
            device=torch.device('cpu')
        )
        
        mock_scripted = Mock(spec=torch.jit.ScriptModule)
        mock_frozen = Mock(spec=torch.jit.ScriptModule)
        
        with patch.object(
            optimizer, 'jit_script_model', 
            return_value=mock_scripted
        ) as mock_script:
            with patch.object(
                optimizer, 'jit_freeze_model',
                return_value=mock_frozen
            ) as mock_freeze:
                result = optimizer.optimize_model(simple_model)
        
        mock_script.assert_called_once()
        mock_freeze.assert_called_once()
    
    def test_optimize_model_applies_torch_compile(self, simple_model):
        """Test optimize_model applies torch.compile when enabled."""
        optimizer = ComputationOptimizer(
            compile_model=True,
            jit_compile=False,
            verbose=False,
            device=torch.device('cpu')
        )
        
        with patch.object(
            optimizer, 'compile_model', 
            return_value=simple_model
        ) as mock_compile:
            result = optimizer.optimize_model(simple_model)
        
        mock_compile.assert_called_once()
    
    def test_optimize_model_skips_compile_for_jit(self, simple_model):
        """Test optimize_model skips torch.compile for JIT scripted models."""
        optimizer = ComputationOptimizer(
            compile_model=True,
            jit_compile=True,
            verbose=False,
            device=torch.device('cpu')
        )
        
        # Create a real JIT scripted module to test the isinstance check
        class SimpleModule(nn.Module):
            def forward(self, x):
                return x * 2
        
        real_scripted = torch.jit.script(SimpleModule())
        
        with patch.object(
            optimizer, 'jit_script_model', 
            return_value=real_scripted
        ):
            with patch.object(
                optimizer, 'jit_freeze_model',
                return_value=real_scripted
            ):
                with patch.object(
                    optimizer, 'compile_model', 
                    return_value=real_scripted
                ) as mock_compile:
                    result = optimizer.optimize_model(simple_model)
        
        # compile_model should not be called for JIT scripted models
        mock_compile.assert_not_called()
    
    def test_optimize_model_with_example_inputs(self, simple_model, sample_input):
        """Test optimize_model with example inputs."""
        optimizer = ComputationOptimizer(
            jit_compile=True,
            verbose=False,
            device=torch.device('cpu')
        )
        
        mock_scripted = Mock(spec=torch.jit.ScriptModule)
        
        with patch.object(
            optimizer, 'jit_script_model', 
            return_value=mock_scripted
        ) as mock_script:
            with patch.object(
                optimizer, 'jit_freeze_model',
                return_value=mock_scripted
            ):
                result = optimizer.optimize_model(
                    simple_model, 
                    example_inputs=(sample_input,)
                )
        
        # Verify example_inputs was passed as second positional argument
        # jit_script_model(model, example_inputs) - positional args
        call_args = mock_script.call_args
        # call_args[0] contains positional arguments: (model, example_inputs)
        assert len(call_args[0]) == 2
        assert call_args[0][1] == (sample_input,)
    
    def test_optimize_model_logging_start(self, simple_model, caplog):
        """Test optimize_model logs start message."""
        optimizer = ComputationOptimizer(
            verbose=True,
            device=torch.device('cpu')
        )
        
        with caplog.at_level(logging.INFO):
            optimizer.optimize_model(simple_model)
        
        assert "Starting model optimization" in caplog.text
    
    def test_optimize_model_logging_complete(self, simple_model, caplog):
        """Test optimize_model logs completion message."""
        optimizer = ComputationOptimizer(
            verbose=True,
            device=torch.device('cpu')
        )
        
        with caplog.at_level(logging.INFO):
            optimizer.optimize_model(simple_model)
        
        assert "Model optimization complete" in caplog.text


# =============================================================================
# PERFORMANCE PROFILING TESTS
# =============================================================================

class TestPerformanceProfiling:
    """Test performance profiling functionality."""
    
    def test_profile_performance_context_manager_cpu(self, simple_model, sample_input):
        """Test profile_performance context manager on CPU."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        simple_model.to(torch.device('cpu'))
        
        with patch.object(torch.profiler, 'profile') as mock_profile:
            mock_context = MagicMock()
            mock_profile.return_value.__enter__ = Mock(return_value=mock_context)
            mock_profile.return_value.__exit__ = Mock(return_value=False)
            
            with optimizer.profile_performance():
                _ = simple_model(sample_input)
        
        mock_profile.assert_called_once()
    
    def test_profile_performance_includes_cuda_activity(self):
        """Test profile_performance includes CUDA activity when on CUDA."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cuda')
        )
        
        with patch.object(torch.profiler, 'profile') as mock_profile:
            mock_context = MagicMock()
            mock_context.key_averages = Mock(return_value=MagicMock())
            mock_context.key_averages.return_value.table = Mock(return_value="table")
            mock_profile.return_value.__enter__ = Mock(return_value=mock_context)
            mock_profile.return_value.__exit__ = Mock(return_value=False)
            
            with optimizer.profile_performance():
                pass
        
        call_kwargs = mock_profile.call_args[1]
        activities = call_kwargs.get('activities', [])
        # Should include CUDA activity
        assert torch.profiler.ProfilerActivity.CUDA in activities
    
    def test_profile_performance_record_shapes(self, simple_model, sample_input):
        """Test profile_performance with record_shapes option."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        with patch.object(torch.profiler, 'profile') as mock_profile:
            mock_context = MagicMock()
            mock_profile.return_value.__enter__ = Mock(return_value=mock_context)
            mock_profile.return_value.__exit__ = Mock(return_value=False)
            
            with optimizer.profile_performance(record_shapes=True):
                pass
        
        call_kwargs = mock_profile.call_args[1]
        assert call_kwargs.get('record_shapes') is True
    
    def test_profile_performance_with_stack(self, simple_model, sample_input):
        """Test profile_performance with stack traces."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        with patch.object(torch.profiler, 'profile') as mock_profile:
            mock_context = MagicMock()
            mock_profile.return_value.__enter__ = Mock(return_value=mock_context)
            mock_profile.return_value.__exit__ = Mock(return_value=False)
            
            with optimizer.profile_performance(with_stack=True):
                pass
        
        call_kwargs = mock_profile.call_args[1]
        assert call_kwargs.get('with_stack') is True
    
    def test_profile_performance_with_flops(self, simple_model, sample_input):
        """Test profile_performance with FLOPs profiling."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        with patch.object(torch.profiler, 'profile') as mock_profile:
            mock_context = MagicMock()
            mock_profile.return_value.__enter__ = Mock(return_value=mock_context)
            mock_profile.return_value.__exit__ = Mock(return_value=False)
            
            with optimizer.profile_performance(with_flops=True):
                pass
        
        call_kwargs = mock_profile.call_args[1]
        assert call_kwargs.get('with_flops') is True
    
    def test_profile_performance_yields_profiler(self):
        """Test profile_performance yields profiler object."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        mock_profiler = MagicMock()
        
        with patch.object(torch.profiler, 'profile') as mock_profile:
            mock_profile.return_value.__enter__ = Mock(return_value=mock_profiler)
            mock_profile.return_value.__exit__ = Mock(return_value=False)
            
            with optimizer.profile_performance() as prof:
                assert prof is mock_profiler
    
    def test_profile_performance_verbose_prints_table(self, simple_model, sample_input, capsys):
        """Test profile_performance prints table when verbose."""
        optimizer = ComputationOptimizer(
            verbose=True,
            device=torch.device('cpu')
        )
        
        simple_model.to(torch.device('cpu'))
        
        mock_profiler = MagicMock()
        mock_key_averages = MagicMock()
        mock_key_averages.table.return_value = "Mocked Profiling Table"
        mock_profiler.key_averages.return_value = mock_key_averages
        
        with patch.object(torch.profiler, 'profile') as mock_profile:
            mock_profile.return_value.__enter__ = Mock(return_value=mock_profiler)
            mock_profile.return_value.__exit__ = Mock(return_value=False)
            
            with optimizer.profile_performance():
                _ = simple_model(sample_input)
        
        captured = capsys.readouterr()
        assert "Performance Profiling Results" in captured.out
    
    def test_profile_performance_cpu_only_activities(self):
        """Test profile_performance includes only CPU activity when on CPU."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        with patch.object(torch.profiler, 'profile') as mock_profile:
            mock_context = MagicMock()
            mock_profile.return_value.__enter__ = Mock(return_value=mock_context)
            mock_profile.return_value.__exit__ = Mock(return_value=False)
            
            with optimizer.profile_performance():
                pass
        
        call_kwargs = mock_profile.call_args[1]
        activities = call_kwargs.get('activities', [])
        assert torch.profiler.ProfilerActivity.CPU in activities
        assert torch.profiler.ProfilerActivity.CUDA not in activities
    
    def test_profile_performance_all_options_combined(self):
        """Test profile_performance with all options enabled."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        with patch.object(torch.profiler, 'profile') as mock_profile:
            mock_context = MagicMock()
            mock_profile.return_value.__enter__ = Mock(return_value=mock_context)
            mock_profile.return_value.__exit__ = Mock(return_value=False)
            
            with optimizer.profile_performance(
                record_shapes=True,
                with_stack=True,
                with_flops=True
            ):
                pass
        
        call_kwargs = mock_profile.call_args[1]
        assert call_kwargs.get('record_shapes') is True
        assert call_kwargs.get('with_stack') is True
        assert call_kwargs.get('with_flops') is True


# =============================================================================
# BENCHMARKING TESTS
# =============================================================================

class TestBenchmarking:
    """Test model benchmarking functionality."""
    
    def test_benchmark_model_basic(self, simple_model, sample_input):
        """Test basic model benchmarking."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        simple_model.to(torch.device('cpu'))
        
        results = optimizer.benchmark_model(
            simple_model,
            sample_input,
            num_iterations=10,
            warmup_iterations=2
        )
        
        assert 'iterations' in results
        assert 'avg_time_ms' in results
        assert 'min_time_ms' in results
        assert 'max_time_ms' in results
        assert 'throughput_fps' in results
        assert 'std_time_ms' in results
        
        assert results['iterations'] == 10
        assert results['avg_time_ms'] >= 0
        assert results['min_time_ms'] >= 0
        assert results['max_time_ms'] >= 0
        assert results['throughput_fps'] >= 0
        assert results['std_time_ms'] >= 0
    
    def test_benchmark_model_sets_eval_mode(self, simple_model, sample_input):
        """Test benchmark_model sets model to eval mode."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        simple_model.to(torch.device('cpu'))
        simple_model.train()  # Ensure in train mode first
        
        optimizer.benchmark_model(
            simple_model,
            sample_input,
            num_iterations=5,
            warmup_iterations=2
        )
        
        # Model should be in eval mode after benchmark
        assert not simple_model.training
    
    def test_benchmark_model_warmup_iterations(self, simple_model, sample_input):
        """Test benchmark_model performs warmup iterations."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        simple_model.to(torch.device('cpu'))
        
        call_count = [0]
        original_forward = simple_model.forward
        
        def counting_forward(*args, **kwargs):
            call_count[0] += 1
            return original_forward(*args, **kwargs)
        
        simple_model.forward = counting_forward
        
        optimizer.benchmark_model(
            simple_model,
            sample_input,
            num_iterations=10,
            warmup_iterations=5
        )
        
        # Should have warmup + benchmark iterations
        assert call_count[0] == 15
    
    def test_benchmark_model_timing_consistency(self, simple_model, sample_input):
        """Test benchmark_model timing is consistent."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        simple_model.to(torch.device('cpu'))
        
        results = optimizer.benchmark_model(
            simple_model,
            sample_input,
            num_iterations=50,
            warmup_iterations=10
        )
        
        # Min should be <= avg <= max
        assert results['min_time_ms'] <= results['avg_time_ms']
        assert results['avg_time_ms'] <= results['max_time_ms']
    
    def test_benchmark_model_throughput_calculation(self, simple_model, sample_input):
        """Test benchmark_model throughput calculation."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        simple_model.to(torch.device('cpu'))
        
        results = optimizer.benchmark_model(
            simple_model,
            sample_input,
            num_iterations=10,
            warmup_iterations=2
        )
        
        # Throughput should be approximately 1/avg_time
        expected_throughput = 1000.0 / results['avg_time_ms']
        assert abs(results['throughput_fps'] - expected_throughput) < 1.0
    
    def test_benchmark_model_logging(self, simple_model, sample_input, caplog):
        """Test benchmark_model logging when verbose."""
        optimizer = ComputationOptimizer(
            verbose=True,
            device=torch.device('cpu')
        )
        
        simple_model.to(torch.device('cpu'))
        
        with caplog.at_level(logging.INFO):
            optimizer.benchmark_model(
                simple_model,
                sample_input,
                num_iterations=5,
                warmup_iterations=2
            )
        
        assert "Benchmark results" in caplog.text
    
    def test_benchmark_model_cuda_sync(self, simple_model, sample_input):
        """Test benchmark_model calls CUDA sync on CUDA device."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cuda')
        )
        
        with patch.object(torch.cuda, 'synchronize') as mock_sync:
            with patch.object(simple_model, 'eval', return_value=simple_model):
                with patch.object(simple_model, '__call__', return_value=torch.randn(4, 1)):
                    optimizer.benchmark_model(
                        simple_model,
                        sample_input,
                        num_iterations=5,
                        warmup_iterations=2
                    )
        
        # Should call synchronize multiple times
        assert mock_sync.call_count > 0


# =============================================================================
# COMPARE OPTIMIZATIONS TESTS
# =============================================================================

class TestCompareOptimizations:
    """Test optimization comparison functionality."""
    
    def test_compare_optimizations_basic(self, simple_model, sample_input):
        """Test basic optimization comparison."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        simple_model.to(torch.device('cpu'))
        
        configs = [
            {'name': 'baseline', 'compile_model': False, 'device': torch.device('cpu')},
            {'name': 'compiled', 'compile_model': False, 'jit_compile': False, 'device': torch.device('cpu')},
        ]
        
        results = optimizer.compare_optimizations(
            simple_model,
            sample_input,
            configs,
            num_iterations=5
        )
        
        assert 'baseline' in results
        assert 'compiled' in results
        assert 'avg_time_ms' in results['baseline']
        assert 'avg_time_ms' in results['compiled']
    
    def test_compare_optimizations_auto_naming(self, simple_model, sample_input):
        """Test optimization comparison auto-generates names."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        simple_model.to(torch.device('cpu'))
        
        configs = [
            {'compile_model': False, 'device': torch.device('cpu')},
            {'compile_model': False, 'jit_compile': False, 'device': torch.device('cpu')},
        ]
        
        results = optimizer.compare_optimizations(
            simple_model,
            sample_input,
            configs,
            num_iterations=5
        )
        
        assert 'config_0' in results
        assert 'config_1' in results
    
    def test_compare_optimizations_single_config(self, simple_model, sample_input):
        """Test optimization comparison with single config."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        simple_model.to(torch.device('cpu'))
        
        configs = [
            {'name': 'single', 'compile_model': False, 'device': torch.device('cpu')},
        ]
        
        results = optimizer.compare_optimizations(
            simple_model,
            sample_input,
            configs,
            num_iterations=5
        )
        
        assert len(results) == 1
        assert 'single' in results
    
    def test_compare_optimizations_empty_configs(self, simple_model, sample_input):
        """Test optimization comparison with empty configs list."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        simple_model.to(torch.device('cpu'))
        
        results = optimizer.compare_optimizations(
            simple_model,
            sample_input,
            [],
            num_iterations=5
        )
        
        assert results == {}
    
    def test_compare_optimizations_result_structure(self, simple_model, sample_input):
        """Test optimization comparison result structure completeness."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        simple_model.to(torch.device('cpu'))
        
        configs = [
            {'name': 'test', 'compile_model': False, 'device': torch.device('cpu')},
        ]
        
        results = optimizer.compare_optimizations(
            simple_model,
            sample_input,
            configs,
            num_iterations=5
        )
        
        expected_keys = ['iterations', 'avg_time_ms', 'min_time_ms', 'max_time_ms', 
                         'throughput_fps', 'std_time_ms']
        for key in expected_keys:
            assert key in results['test'], f"Missing key: {key}"
    
    def test_compare_optimizations_model_without_clone(self, simple_model, sample_input):
        """Test optimization comparison handles model without clone method."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        simple_model.to(torch.device('cpu'))
        
        # Ensure model doesn't have clone (it shouldn't by default for nn.Module)
        assert not hasattr(simple_model, 'clone')
        
        configs = [
            {'name': 'test', 'compile_model': False, 'device': torch.device('cpu')},
        ]
        
        # Should not raise even without clone method
        results = optimizer.compare_optimizations(
            simple_model,
            sample_input,
            configs,
            num_iterations=5
        )
        
        assert 'test' in results
        assert len(results) == 1
    
    def test_compare_optimizations_logging(self, simple_model, sample_input, caplog):
        """Test optimization comparison logging."""
        optimizer = ComputationOptimizer(
            verbose=True,
            device=torch.device('cpu')
        )
        
        simple_model.to(torch.device('cpu'))
        
        configs = [
            {'name': 'test_config', 'compile_model': False, 'device': torch.device('cpu')},
        ]
        
        with caplog.at_level(logging.INFO):
            results = optimizer.compare_optimizations(
                simple_model,
                sample_input,
                configs,
                num_iterations=5
            )
        
        assert "test_config" in caplog.text


# =============================================================================
# KERNEL FUSION TESTS
# =============================================================================

class TestKernelFusion:
    """Test kernel fusion functionality."""
    
    def test_enable_fusion_default(self):
        """Test enabling default kernel fusion."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        with patch.object(torch._C, '_jit_set_profiling_executor') as mock_exec:
            optimizer.enable_fusion(aggressive=False)
        
        mock_exec.assert_called_once_with(True)
    
    def test_enable_fusion_aggressive(self):
        """Test enabling aggressive kernel fusion."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        with patch.object(torch._C, '_jit_set_texpr_fuser_enabled') as mock_texpr:
            with patch.object(torch._C, '_jit_set_nvfuser_enabled') as mock_nvfuser:
                optimizer.enable_fusion(aggressive=True)
        
        mock_texpr.assert_called_once_with(True)
        mock_nvfuser.assert_called_once_with(True)
    
    def test_enable_fusion_logging(self, caplog):
        """Test enable_fusion logging."""
        optimizer = ComputationOptimizer(
            verbose=True,
            device=torch.device('cpu')
        )
        
        with caplog.at_level(logging.INFO):
            with patch.object(torch._C, '_jit_set_profiling_executor'):
                optimizer.enable_fusion(aggressive=False)
        
        assert "kernel fusion" in caplog.text.lower()
    
    def test_enable_fusion_aggressive_logging(self, caplog):
        """Test aggressive enable_fusion logging."""
        optimizer = ComputationOptimizer(
            verbose=True,
            device=torch.device('cpu')
        )
        
        with caplog.at_level(logging.INFO):
            with patch.object(torch._C, '_jit_set_texpr_fuser_enabled'):
                with patch.object(torch._C, '_jit_set_nvfuser_enabled'):
                    optimizer.enable_fusion(aggressive=True)
        
        assert "aggressive" in caplog.text.lower()
    
    def test_disable_fusion(self):
        """Test disabling kernel fusion."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        with patch.object(torch._C, '_jit_set_texpr_fuser_enabled') as mock_texpr:
            with patch.object(torch._C, '_jit_set_nvfuser_enabled') as mock_nvfuser:
                with patch.object(torch._C, '_jit_set_profiling_executor') as mock_exec:
                    optimizer.disable_fusion()
        
        mock_texpr.assert_called_once_with(False)
        mock_nvfuser.assert_called_once_with(False)
        mock_exec.assert_called_once_with(False)
    
    def test_disable_fusion_logging(self, caplog):
        """Test disable_fusion logging."""
        optimizer = ComputationOptimizer(
            verbose=True,
            device=torch.device('cpu')
        )
        
        with caplog.at_level(logging.INFO):
            with patch.object(torch._C, '_jit_set_texpr_fuser_enabled'):
                with patch.object(torch._C, '_jit_set_nvfuser_enabled'):
                    with patch.object(torch._C, '_jit_set_profiling_executor'):
                        optimizer.disable_fusion()
        
        assert "Disabled kernel fusion" in caplog.text


# =============================================================================
# GRAPH OPTIMIZATION TESTS
# =============================================================================

class TestGraphOptimization:
    """Test graph optimization functionality."""
    
    def test_optimize_graph_level_1(self, mock_jit_script_module):
        """Test graph optimization at level 1."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        with patch.object(torch._C, '_jit_pass_inline') as mock_inline:
            with patch.object(torch._C, '_jit_pass_constant_propagation') as mock_const:
                with patch.object(torch._C, '_jit_pass_peephole') as mock_peep:
                    result = optimizer.optimize_graph(
                        mock_jit_script_module,
                        optimization_level=1
                    )
        
        mock_inline.assert_called_once()
        mock_const.assert_called_once()
        mock_peep.assert_called_once()
    
    def test_optimize_graph_level_2(self, mock_jit_script_module):
        """Test graph optimization at level 2 includes fusion."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        with patch.object(torch._C, '_jit_pass_inline'):
            with patch.object(torch._C, '_jit_pass_constant_propagation'):
                with patch.object(torch._C, '_jit_pass_peephole'):
                    with patch.object(torch._C, '_jit_pass_fuse') as mock_fuse:
                        result = optimizer.optimize_graph(
                            mock_jit_script_module,
                            optimization_level=2
                        )
        
        mock_fuse.assert_called_once()
    
    def test_optimize_graph_level_3(self, mock_jit_script_module):
        """Test graph optimization at level 3 includes mutation removal."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        with patch.object(torch._C, '_jit_pass_inline'):
            with patch.object(torch._C, '_jit_pass_constant_propagation'):
                with patch.object(torch._C, '_jit_pass_peephole'):
                    with patch.object(torch._C, '_jit_pass_fuse'):
                        with patch.object(
                            torch._C, '_jit_pass_remove_mutation'
                        ) as mock_remove:
                            result = optimizer.optimize_graph(
                                mock_jit_script_module,
                                optimization_level=3
                            )
        
        mock_remove.assert_called_once()
    
    def test_optimize_graph_failure_returns_model(self, mock_jit_script_module):
        """Test graph optimization failure returns original model."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        with patch.object(
            torch._C, '_jit_pass_inline', 
            side_effect=Exception("Optimization failed")
        ):
            result = optimizer.optimize_graph(
                mock_jit_script_module,
                optimization_level=1
            )
        
        assert result is mock_jit_script_module
    
    def test_optimize_graph_failure_logs_warning(self, mock_jit_script_module, caplog):
        """Test graph optimization failure logs warning."""
        optimizer = ComputationOptimizer(
            verbose=True,
            device=torch.device('cpu')
        )
        
        with caplog.at_level(logging.WARNING):
            with patch.object(
                torch._C, '_jit_pass_inline', 
                side_effect=Exception("Optimization failed")
            ):
                optimizer.optimize_graph(
                    mock_jit_script_module,
                    optimization_level=1
                )
        
        assert "Graph optimization failed" in caplog.text
    
    def test_optimize_graph_logging_success(self, mock_jit_script_module, caplog):
        """Test graph optimization logs success."""
        optimizer = ComputationOptimizer(
            verbose=True,
            device=torch.device('cpu')
        )
        
        with caplog.at_level(logging.INFO):
            with patch.object(torch._C, '_jit_pass_inline'):
                with patch.object(torch._C, '_jit_pass_constant_propagation'):
                    with patch.object(torch._C, '_jit_pass_peephole'):
                        optimizer.optimize_graph(
                            mock_jit_script_module,
                            optimization_level=1
                        )
        
        assert "Applied graph optimizations" in caplog.text


# =============================================================================
# PRINT OPTIMIZATION SUMMARY TESTS
# =============================================================================

class TestPrintOptimizationSummary:
    """Test optimization summary printing."""
    
    def test_print_optimization_summary_cpu(self, capsys):
        """Test print_optimization_summary on CPU."""
        optimizer = ComputationOptimizer(
            compile_model=True,
            compile_mode="default",
            verbose=False,
            device=torch.device('cpu')
        )
        
        optimizer.print_optimization_summary()
        
        captured = capsys.readouterr()
        assert "Computation Optimization Summary" in captured.out
        assert "Device: cpu" in captured.out
        assert "torch.compile: True" in captured.out
        assert "Mode: default" in captured.out
    
    def test_print_optimization_summary_cuda(self, capsys):
        """Test print_optimization_summary on CUDA."""
        optimizer = ComputationOptimizer(
            cudnn_benchmark=True,
            use_tf32=True,
            verbose=False,
            device=torch.device('cuda')
        )
        
        optimizer.print_optimization_summary()
        
        captured = capsys.readouterr()
        assert "Device: cuda" in captured.out
        assert "cuDNN Benchmark" in captured.out
        assert "TF32" in captured.out
    
    def test_print_optimization_summary_all_options(self, capsys):
        """Test print_optimization_summary with all options."""
        optimizer = ComputationOptimizer(
            compile_model=True,
            compile_mode="max-autotune",
            compile_dynamic=True,
            channels_last=True,
            jit_compile=True,
            operator_fusion=True,
            fusion_strategy="aggressive",
            verbose=False,
            device=torch.device('cpu')
        )
        
        optimizer.print_optimization_summary()
        
        captured = capsys.readouterr()
        assert "Channels Last: True" in captured.out
        assert "JIT Compile: True" in captured.out
        assert "Operator Fusion: True" in captured.out
        assert "Fusion Strategy: aggressive" in captured.out
        assert "Dynamic: True" in captured.out
    
    def test_print_optimization_summary_compile_disabled(self, capsys):
        """Test print_optimization_summary with compile disabled."""
        optimizer = ComputationOptimizer(
            compile_model=False,
            verbose=False,
            device=torch.device('cpu')
        )
        
        optimizer.print_optimization_summary()
        
        captured = capsys.readouterr()
        assert "torch.compile: False" in captured.out
        # Mode should not be printed when compile is disabled
        assert "Mode:" not in captured.out or captured.out.count("Mode:") == 0


# =============================================================================
# CONVENIENCE FUNCTIONS TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_get_optimal_settings_training(self):
        """Test get_optimal_settings for training."""
        device = torch.device('cpu')
        
        settings = get_optimal_settings(device, task_type="training")
        
        assert settings['compile_model'] is True
        assert settings['compile_mode'] == 'default'
        assert settings['cudnn_benchmark'] is True
        assert settings['cudnn_deterministic'] is False
        assert settings['use_tf32'] is True
        assert settings['channels_last'] is False
        assert settings['jit_compile'] is False
        assert settings['operator_fusion'] is True
    
    def test_get_optimal_settings_inference(self):
        """Test get_optimal_settings for inference."""
        device = torch.device('cpu')
        
        settings = get_optimal_settings(device, task_type="inference")
        
        assert settings['compile_model'] is True
        assert settings['compile_mode'] == 'max-autotune'
        assert settings['cudnn_benchmark'] is True
        assert settings['cudnn_deterministic'] is False
        assert settings['use_tf32'] is True
        assert settings['channels_last'] is True
        assert settings['jit_compile'] is True
        assert settings['operator_fusion'] is True
    
    def test_get_optimal_settings_unknown_task_defaults_to_inference(self):
        """Test get_optimal_settings defaults to inference for unknown task."""
        device = torch.device('cpu')
        
        settings = get_optimal_settings(device, task_type="unknown")
        
        # Should match inference settings
        assert settings['compile_mode'] == 'max-autotune'
        assert settings['jit_compile'] is True
    
    def test_auto_optimize_model_training(self, simple_model):
        """Test auto_optimize_model for training."""
        device = torch.device('cpu')
        
        with patch('milia_pipeline.models.acceleration.computation_optimization.ComputationOptimizer') as MockOptimizer:
            mock_instance = MagicMock()
            mock_instance.optimize_model.return_value = simple_model
            MockOptimizer.return_value = mock_instance
            
            result = auto_optimize_model(
                simple_model,
                task_type="training",
                device=device
            )
        
        MockOptimizer.assert_called_once()
        mock_instance.optimize_model.assert_called_once_with(simple_model)
    
    def test_auto_optimize_model_inference(self, simple_model):
        """Test auto_optimize_model for inference."""
        device = torch.device('cpu')
        
        with patch('milia_pipeline.models.acceleration.computation_optimization.ComputationOptimizer') as MockOptimizer:
            mock_instance = MagicMock()
            mock_instance.optimize_model.return_value = simple_model
            MockOptimizer.return_value = mock_instance
            
            result = auto_optimize_model(
                simple_model,
                task_type="inference",
                device=device
            )
        
        MockOptimizer.assert_called_once()
    
    def test_auto_optimize_model_auto_device(self, simple_model):
        """Test auto_optimize_model with automatic device detection."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('milia_pipeline.models.acceleration.computation_optimization.ComputationOptimizer') as MockOptimizer:
                mock_instance = MagicMock()
                mock_instance.optimize_model.return_value = simple_model
                MockOptimizer.return_value = mock_instance
                
                result = auto_optimize_model(simple_model)
        
        # Should use CPU device
        call_kwargs = MockOptimizer.call_args[1]
        assert call_kwargs['device'] == torch.device('cpu')


# =============================================================================
# DECORATOR TESTS
# =============================================================================

class TestDecorators:
    """Test decorator functions."""
    
    def test_optimize_inference_decorator(self, simple_model, sample_input):
        """Test optimize_inference decorator."""
        simple_model.to(torch.device('cpu'))
        
        @optimize_inference
        def predict(model, data):
            return model(data)
        
        # The function should work with no_grad and inference_mode
        result = predict(simple_model, sample_input)
        
        assert result is not None
        assert result.shape[0] == sample_input.shape[0]
    
    def test_optimize_inference_no_grad(self, simple_model, sample_input):
        """Test optimize_inference uses no_grad context."""
        simple_model.to(torch.device('cpu'))
        
        @optimize_inference
        def predict(model, data):
            # Check that gradients are disabled
            return model(data), torch.is_grad_enabled()
        
        result, grad_enabled = predict(simple_model, sample_input)
        
        assert grad_enabled is False
    
    def test_optimize_inference_preserves_function_name(self):
        """Test optimize_inference preserves function metadata."""
        @optimize_inference
        def my_prediction_function(model, data):
            """My docstring."""
            return model(data)
        
        assert my_prediction_function.__name__ == "my_prediction_function"
        assert "My docstring" in my_prediction_function.__doc__
    
    def test_optimize_inference_with_kwargs(self, simple_model, sample_input):
        """Test optimize_inference with keyword arguments."""
        simple_model.to(torch.device('cpu'))
        
        @optimize_inference
        def predict_with_options(model, data, multiply=1.0):
            return model(data) * multiply
        
        result = predict_with_options(simple_model, sample_input, multiply=2.0)
        
        assert result is not None
    
    def test_optimize_inference_inference_mode(self, simple_model, sample_input):
        """Test optimize_inference uses inference_mode context."""
        simple_model.to(torch.device('cpu'))
        
        inference_mode_active = [None]
        
        @optimize_inference
        def predict(model, data):
            # Check inference mode is active
            inference_mode_active[0] = torch.is_inference_mode_enabled()
            return model(data)
        
        result = predict(simple_model, sample_input)
        
        assert inference_mode_active[0] is True
    
    def test_optimize_inference_nested_decorators(self, simple_model, sample_input):
        """Test optimize_inference works with nested decorators."""
        simple_model.to(torch.device('cpu'))
        
        def logging_decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        
        @logging_decorator
        @optimize_inference
        def predict(model, data):
            return model(data)
        
        result = predict(simple_model, sample_input)
        assert result is not None
    
    def test_optimize_inference_return_value_passthrough(self, simple_model, sample_input):
        """Test optimize_inference passes through return value correctly."""
        simple_model.to(torch.device('cpu'))
        
        @optimize_inference
        def predict_with_metadata(model, data):
            output = model(data)
            return {'output': output, 'shape': output.shape}
        
        result = predict_with_metadata(simple_model, sample_input)
        
        assert isinstance(result, dict)
        assert 'output' in result
        assert 'shape' in result


# =============================================================================
# EXCEPTION HANDLING TESTS
# =============================================================================

class TestExceptionHandling:
    """Test exception handling and error scenarios."""
    
    def test_optimization_error_inherits_from_hardware_error(self):
        """Test OptimizationError inherits from HardwareError."""
        assert issubclass(OptimizationError, HardwareError)
    
    def test_hardware_error_inherits_from_model_error(self):
        """Test HardwareError inherits from ModelError."""
        assert issubclass(HardwareError, ModelError)
    
    def test_model_error_inherits_from_exception(self):
        """Test ModelError inherits from Exception."""
        assert issubclass(ModelError, Exception)
    
    def test_optimization_error_message(self):
        """Test OptimizationError message."""
        error = OptimizationError("Test error message")
        
        assert str(error) == "Test error message"
    
    def test_optimization_error_raise_and_catch(self):
        """Test raising and catching OptimizationError."""
        with pytest.raises(OptimizationError) as exc_info:
            raise OptimizationError("Compilation failed")
        
        assert "Compilation failed" in str(exc_info.value)
    
    def test_catch_optimization_error_as_hardware_error(self):
        """Test OptimizationError can be caught as HardwareError."""
        with pytest.raises(HardwareError):
            raise OptimizationError("Test error")
    
    def test_catch_optimization_error_as_model_error(self):
        """Test OptimizationError can be caught as ModelError."""
        with pytest.raises(ModelError):
            raise OptimizationError("Test error")


# =============================================================================
# EDGE CASES AND ERROR SCENARIOS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error scenarios."""
    
    def test_empty_compiled_models_cache(self):
        """Test empty compiled models cache."""
        with patch('torch.cuda.is_available', return_value=False):
            optimizer = ComputationOptimizer(verbose=False)
        
        assert len(optimizer._compiled_models) == 0
    
    def test_optimize_model_with_none_example_inputs(self, simple_model):
        """Test optimize_model with None example_inputs."""
        optimizer = ComputationOptimizer(
            jit_compile=True,
            verbose=False,
            device=torch.device('cpu')
        )
        
        # Mock the JIT operations
        mock_scripted = Mock(spec=torch.jit.ScriptModule)
        
        with patch.object(optimizer, 'jit_script_model', return_value=mock_scripted):
            with patch.object(optimizer, 'jit_freeze_model', return_value=mock_scripted):
                result = optimizer.optimize_model(simple_model, example_inputs=None)
        
        # Should still work with None example_inputs
        assert result is not None
    
    def test_benchmark_with_zero_iterations(self, simple_model, sample_input):
        """Test benchmark with edge case iteration counts."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        simple_model.to(torch.device('cpu'))
        
        # Test with minimum iterations
        results = optimizer.benchmark_model(
            simple_model,
            sample_input,
            num_iterations=1,
            warmup_iterations=0
        )
        
        assert results['iterations'] == 1
        assert results['avg_time_ms'] >= 0
    
    def test_config_immutability_concept(self):
        """Test that config values are properly stored."""
        config = ComputationConfig(
            compile_model=True,
            compile_mode="max-autotune"
        )
        
        # Get dict representation
        dict1 = config.to_dict()
        
        # Modify the dict
        dict1['compile_model'] = False
        
        # Original config should be unchanged
        assert config.compile_model is True
    
    def test_multiple_optimizer_instances(self):
        """Test multiple optimizer instances don't interfere."""
        with patch('torch.cuda.is_available', return_value=False):
            optimizer1 = ComputationOptimizer(
                compile_model=True,
                verbose=False
            )
            optimizer2 = ComputationOptimizer(
                compile_model=False,
                verbose=False
            )
        
        assert optimizer1.config.compile_model is True
        assert optimizer2.config.compile_model is False
        
        # Their caches should be separate
        assert optimizer1._compiled_models is not optimizer2._compiled_models
    
    def test_device_type_string_handling(self):
        """Test device can be torch.device object."""
        device = torch.device('cpu')
        
        optimizer = ComputationOptimizer(
            device=device,
            verbose=False
        )
        
        assert optimizer.device == device
        assert optimizer.device.type == 'cpu'
    
    def test_large_warmup_iterations(self, simple_model, sample_input):
        """Test benchmark with large warmup iterations."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        simple_model.to(torch.device('cpu'))
        
        results = optimizer.benchmark_model(
            simple_model,
            sample_input,
            num_iterations=5,
            warmup_iterations=20
        )
        
        assert results['iterations'] == 5
    
    def test_benchmark_std_calculation(self, simple_model, sample_input):
        """Test benchmark standard deviation calculation is correct."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        simple_model.to(torch.device('cpu'))
        
        results = optimizer.benchmark_model(
            simple_model,
            sample_input,
            num_iterations=50,
            warmup_iterations=5
        )
        
        # Standard deviation should be non-negative
        assert results['std_time_ms'] >= 0
        
        # With enough iterations, std should be reasonable relative to avg
        # (just a sanity check, not a strict bound)
        assert results['std_time_ms'] <= results['avg_time_ms'] * 10
    
    def test_optimizer_config_reference_integrity(self):
        """Test that optimizer's config is properly referenced."""
        with patch('torch.cuda.is_available', return_value=False):
            optimizer = ComputationOptimizer(
                compile_model=True,
                compile_mode="max-autotune",
                verbose=False
            )
        
        # Config should be the same object
        config_ref1 = optimizer.config
        config_ref2 = optimizer.config
        assert config_ref1 is config_ref2
    
    def test_compiled_models_cache_multiple_models(self, simple_model):
        """Test compiled models cache handles multiple different models."""
        optimizer = ComputationOptimizer(
            compile_model=True,
            verbose=False,
            device=torch.device('cpu')
        )
        
        if hasattr(torch, 'compile'):
            # Create another simple model
            class AnotherModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(5, 3)
                
                def forward(self, x):
                    return self.linear(x)
            
            another_model = AnotherModel()
            
            with patch.object(torch, 'compile', side_effect=lambda m, **kw: m):
                optimizer.compile_model(simple_model)
                optimizer.compile_model(another_model)
            
            # Both models should be cached with different keys
            assert len(optimizer._compiled_models) == 2
            assert id(simple_model) in optimizer._compiled_models
            assert id(another_model) in optimizer._compiled_models
    
    def test_device_cuda_index(self):
        """Test optimizer handles CUDA device with index."""
        device = torch.device('cuda:0')
        
        optimizer = ComputationOptimizer(
            device=device,
            verbose=False
        )
        
        assert optimizer.device == device
        assert optimizer.device.type == 'cuda'
        assert optimizer.device.index == 0
    
    def test_config_all_false_booleans(self):
        """Test config with all boolean fields set to False."""
        config = ComputationConfig(
            compile_model=False,
            compile_dynamic=False,
            cudnn_benchmark=False,
            cudnn_deterministic=False,
            use_tf32=False,
            channels_last=False,
            jit_compile=False,
            operator_fusion=False
        )
        
        result = config.to_dict()
        
        assert result['compile_model'] is False
        assert result['compile_dynamic'] is False
        assert result['cudnn_benchmark'] is False
        assert result['cudnn_deterministic'] is False
        assert result['use_tf32'] is False
        assert result['channels_last'] is False
        assert result['jit_compile'] is False
        assert result['operator_fusion'] is False
    
    def test_config_all_true_booleans(self):
        """Test config with all boolean fields set to True."""
        config = ComputationConfig(
            compile_model=True,
            compile_dynamic=True,
            cudnn_benchmark=True,
            cudnn_deterministic=True,
            use_tf32=True,
            channels_last=True,
            jit_compile=True,
            operator_fusion=True
        )
        
        result = config.to_dict()
        
        assert result['compile_model'] is True
        assert result['compile_dynamic'] is True
        assert result['cudnn_benchmark'] is True
        assert result['cudnn_deterministic'] is True
        assert result['use_tf32'] is True
        assert result['channels_last'] is True
        assert result['jit_compile'] is True
        assert result['operator_fusion'] is True


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple functionalities."""
    
    def test_full_optimization_pipeline_cpu(self, simple_model, sample_input):
        """Test full optimization pipeline on CPU."""
        optimizer = ComputationOptimizer(
            compile_model=False,  # Disable for simple test
            jit_compile=False,
            channels_last=False,
            verbose=False,
            device=torch.device('cpu')
        )
        
        # Optimize model
        optimized = optimizer.optimize_model(simple_model)
        
        # Benchmark
        results = optimizer.benchmark_model(
            optimized,
            sample_input,
            num_iterations=5,
            warmup_iterations=2
        )
        
        assert results['avg_time_ms'] > 0
        assert results['throughput_fps'] > 0
    
    def test_config_to_optimizer_consistency(self):
        """Test config values match optimizer initialization."""
        config = ComputationConfig(
            compile_model=True,
            compile_mode="max-autotune",
            cudnn_benchmark=False
        )
        
        optimizer = ComputationOptimizer(
            compile_model=config.compile_model,
            compile_mode=config.compile_mode,
            cudnn_benchmark=config.cudnn_benchmark,
            verbose=False,
            device=torch.device('cpu')
        )
        
        assert optimizer.config.compile_model == config.compile_model
        assert optimizer.config.compile_mode == config.compile_mode
        assert optimizer.config.cudnn_benchmark == config.cudnn_benchmark
    
    def test_sequential_optimizations(self, simple_model, sample_input):
        """Test sequential optimization operations."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        simple_model.to(torch.device('cpu'))
        
        # First enable fusion
        with patch.object(torch._C, '_jit_set_profiling_executor'):
            optimizer.enable_fusion(aggressive=False)
        
        # Then benchmark
        results1 = optimizer.benchmark_model(
            simple_model,
            sample_input,
            num_iterations=3,
            warmup_iterations=1
        )
        
        # Disable fusion
        with patch.object(torch._C, '_jit_set_texpr_fuser_enabled'):
            with patch.object(torch._C, '_jit_set_nvfuser_enabled'):
                with patch.object(torch._C, '_jit_set_profiling_executor'):
                    optimizer.disable_fusion()
        
        # Benchmark again
        results2 = optimizer.benchmark_model(
            simple_model,
            sample_input,
            num_iterations=3,
            warmup_iterations=1
        )
        
        # Both should complete successfully
        assert results1['iterations'] == 3
        assert results2['iterations'] == 3


# =============================================================================
# CUDA-SPECIFIC TESTS (MOCKED)
# =============================================================================

class TestCUDASpecific:
    """Test CUDA-specific functionality (mocked)."""
    
    def test_cuda_device_sync_in_benchmark(self):
        """Test CUDA synchronization in benchmark."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cuda')
        )
        
        mock_model = Mock(spec=nn.Module)
        mock_model.eval = Mock(return_value=mock_model)
        mock_model.__call__ = Mock(return_value=torch.randn(4, 1))
        
        input_data = torch.randn(4, 10)
        
        with patch.object(torch.cuda, 'synchronize') as mock_sync:
            optimizer.benchmark_model(
                mock_model,
                input_data,
                num_iterations=3,
                warmup_iterations=1
            )
        
        # synchronize should be called for warmup sync + each iteration
        assert mock_sync.call_count >= 3
    
    def test_cuda_profiler_activities(self):
        """Test CUDA profiler activities are included."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cuda')
        )
        
        with patch.object(torch.profiler, 'profile') as mock_profile:
            mock_context = MagicMock()
            mock_context.key_averages = Mock(return_value=MagicMock())
            mock_context.key_averages.return_value.table = Mock(return_value="table")
            mock_profile.return_value.__enter__ = Mock(return_value=mock_context)
            mock_profile.return_value.__exit__ = Mock(return_value=False)
            
            with optimizer.profile_performance():
                pass
        
        # Verify CUDA activity is included
        call_kwargs = mock_profile.call_args[1]
        activities = call_kwargs.get('activities', [])
        assert torch.profiler.ProfilerActivity.CUDA in activities
    
    def test_tf32_settings_on_cuda(self):
        """Test TF32 settings are applied on CUDA device."""
        with patch.object(torch.backends.cudnn, 'is_available', return_value=True):
            optimizer = ComputationOptimizer(
                use_tf32=True,
                verbose=False,
                device=torch.device('cuda')
            )
        
        assert optimizer.config.use_tf32 is True
    
    def test_cudnn_settings_on_cuda(self):
        """Test cuDNN settings are applied on CUDA device."""
        with patch.object(torch.backends.cudnn, 'is_available', return_value=True):
            optimizer = ComputationOptimizer(
                cudnn_benchmark=True,
                cudnn_deterministic=True,
                verbose=False,
                device=torch.device('cuda')
            )
        
        assert optimizer.config.cudnn_benchmark is True
        assert optimizer.config.cudnn_deterministic is True


# =============================================================================
# MEMORY AND RESOURCE TESTS
# =============================================================================

class TestMemoryAndResources:
    """Test memory and resource management."""
    
    def test_compiled_models_cache_growth(self, simple_model):
        """Test compiled models cache grows with compiled models."""
        optimizer = ComputationOptimizer(
            compile_model=True,
            verbose=False,
            device=torch.device('cpu')
        )
        
        if hasattr(torch, 'compile'):
            initial_size = len(optimizer._compiled_models)
            
            with patch.object(torch, 'compile', return_value=simple_model):
                optimizer.compile_model(simple_model)
            
            assert len(optimizer._compiled_models) == initial_size + 1
    
    def test_garbage_collection_after_benchmark(self, simple_model, sample_input):
        """Test garbage collection can clean up after benchmark."""
        optimizer = ComputationOptimizer(
            verbose=False,
            device=torch.device('cpu')
        )
        
        simple_model.to(torch.device('cpu'))
        
        # Run benchmark
        optimizer.benchmark_model(
            simple_model,
            sample_input,
            num_iterations=5,
            warmup_iterations=2
        )
        
        # Force garbage collection
        gc.collect()
        
        # Should complete without errors


# =============================================================================
# THREAD SAFETY AND CONCURRENT ACCESS TESTS
# =============================================================================

class TestThreadSafety:
    """Test thread safety and concurrent access patterns."""
    
    def test_multiple_optimizers_concurrent_config_access(self):
        """Test multiple optimizers can be accessed concurrently."""
        import threading
        
        results = []
        errors = []
        
        def create_and_access_optimizer(compile_mode, results_list, errors_list):
            try:
                with patch('torch.cuda.is_available', return_value=False):
                    optimizer = ComputationOptimizer(
                        compile_model=True,
                        compile_mode=compile_mode,
                        verbose=False
                    )
                    # Access config multiple times
                    for _ in range(10):
                        config_dict = optimizer.config.to_dict()
                        assert config_dict['compile_mode'] == compile_mode
                    results_list.append(compile_mode)
            except Exception as e:
                errors_list.append(str(e))
        
        threads = []
        modes = ["default", "reduce-overhead", "max-autotune"]
        
        for mode in modes:
            t = threading.Thread(
                target=create_and_access_optimizer,
                args=(mode, results, errors)
            )
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 3
    
    def test_config_to_dict_thread_safety(self):
        """Test that to_dict() is safe to call from multiple threads."""
        import threading
        
        config = ComputationConfig(
            compile_model=True,
            compile_mode="max-autotune"
        )
        
        results = []
        errors = []
        
        def call_to_dict(config_obj, results_list, errors_list):
            try:
                for _ in range(100):
                    d = config_obj.to_dict()
                    assert d['compile_model'] is True
                    assert d['compile_mode'] == "max-autotune"
                results_list.append(True)
            except Exception as e:
                errors_list.append(str(e))
        
        threads = []
        for _ in range(5):
            t = threading.Thread(target=call_to_dict, args=(config, results, errors))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
    
    def test_optimizer_instance_isolation(self):
        """Test that optimizer instances are properly isolated."""
        with patch('torch.cuda.is_available', return_value=False):
            opt1 = ComputationOptimizer(compile_model=True, verbose=False)
            opt2 = ComputationOptimizer(compile_model=False, verbose=False)
        
        # Modify opt1's config
        opt1.config.compile_model = False
        
        # opt2 should not be affected (they have separate config instances)
        # Note: Since we modified opt1's config after creation,
        # we're testing that the configs are separate objects
        assert opt1.config is not opt2.config


# =============================================================================
# LOGGING CONFIGURATION TESTS
# =============================================================================

class TestLoggingConfiguration:
    """Test logging configuration and behavior."""
    
    def test_logger_name(self):
        """Test logger has correct name."""
        from milia_pipeline.models.acceleration import computation_optimization
        
        assert computation_optimization.logger.name == computation_optimization.__name__
    
    def test_verbose_true_logs_info(self, caplog):
        """Test verbose=True produces INFO logs."""
        with caplog.at_level(logging.INFO):
            with patch('torch.cuda.is_available', return_value=False):
                optimizer = ComputationOptimizer(verbose=True)
        
        # Should have logged initialization info
        assert len(caplog.records) > 0
    
    def test_verbose_false_reduces_logging(self, caplog):
        """Test verbose=False reduces logging."""
        with caplog.at_level(logging.INFO):
            caplog.clear()
            with patch('torch.cuda.is_available', return_value=False):
                optimizer = ComputationOptimizer(verbose=False)
        
        # Check for absence of specific verbose logs
        init_logs = [r for r in caplog.records 
                     if "ComputationOptimizer initialized" in r.message]
        assert len(init_logs) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
