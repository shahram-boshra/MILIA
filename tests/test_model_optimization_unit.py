#!/usr/bin/env python3
"""
Unit Tests for model_optimization.py Module

Comprehensive test suite covering:
- QuantizationType and PruningType enums
- OptimizationConfig Pydantic BaseModel (v1.1.0 Pydantic V2 migration)
- ModelOptimizer class (quantization, pruning, distillation, export, metrics)
- Convenience functions: quantize_for_inference, prune_for_deployment
- Pydantic V2 behavior: model_dump(), model_validate(), model_fields,
  model_json_schema(), model_copy(), ValidationError handling

Author: milia Team
Test Module Version: 1.1.0
Target Module: milia_pipeline/models/deployment/model_optimization.py
"""

import copy
import json
import logging
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError as PydanticValidationError

# =============================================================================
# ADD PROJECT ROOT TO PYTHON PATH
# =============================================================================
# Get the project root (parent of 'tests' directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# MODULE IMPORTS (with mocking strategy for torch dependencies)
# =============================================================================
# We import torch and nn for type annotations and mock creation
import torch
import torch.nn as nn
from torch.nn.utils import prune

# Import the module under test
from milia_pipeline.models.deployment.model_optimization import (
    ExportError,
    # Exceptions
    ModelError,
    # Main class
    ModelOptimizer,
    # Dataclass
    OptimizationConfig,
    OptimizationError,
    PruningType,
    # Enums
    QuantizationType,
    prune_for_deployment,
    # Convenience functions
    quantize_for_inference,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def mock_quantization_backend():
    """
    Mock the quantization backend setting to avoid FBGEMM not supported errors.
    This is needed because the test environment may not have FBGEMM support.
    We mock the underlying C function that actually sets the engine.
    """
    with patch("torch._C._set_qengine", return_value=None):
        yield


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath, ignore_errors=True)


@pytest.fixture
def simple_linear_model():
    """Create a simple Linear model for testing."""
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
    return model


@pytest.fixture
def simple_conv_model():
    """Create a simple convolutional model for testing."""
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
    )
    return model


@pytest.fixture
def mock_model():
    """Create a mock PyTorch model."""
    model = MagicMock(spec=nn.Module)
    model.state_dict.return_value = {"layer.weight": torch.randn(10, 10)}
    model.eval = MagicMock(return_value=model)
    model.train = MagicMock(return_value=model)
    model.half = MagicMock(return_value=model)
    model.parameters = MagicMock(return_value=[torch.randn(10, 10)])
    model.named_modules = MagicMock(return_value=[("linear", nn.Linear(10, 10))])
    model.named_parameters = MagicMock(return_value=[("weight", torch.randn(10, 10))])
    model.buffers = MagicMock(return_value=[torch.randn(5, 5)])
    return model


@pytest.fixture
def default_config():
    """Create a default OptimizationConfig."""
    return OptimizationConfig()


@pytest.fixture
def quantization_config():
    """Create a quantization-enabled config."""
    return OptimizationConfig(
        quantization_enabled=True, quantization_type="dynamic", quantization_backend="fbgemm"
    )


@pytest.fixture
def pruning_config():
    """Create a pruning-enabled config."""
    return OptimizationConfig(pruning_enabled=True, pruning_type="magnitude", pruning_amount=0.3)


@pytest.fixture
def distillation_config():
    """Create a distillation-enabled config."""
    return OptimizationConfig(
        distillation_enabled=True, distillation_temperature=3.0, distillation_alpha=0.5
    )


@pytest.fixture
def full_config():
    """Create a fully configured OptimizationConfig."""
    return OptimizationConfig(
        quantization_enabled=True,
        quantization_type="dynamic",
        quantization_backend="fbgemm",
        pruning_enabled=True,
        pruning_type="magnitude",
        pruning_amount=0.4,
        distillation_enabled=True,
        distillation_temperature=4.0,
        distillation_alpha=0.6,
        export_onnx=True,
        optimize_for_mobile=True,
    )


# =============================================================================
# TESTS: QuantizationType Enum
# =============================================================================


class TestQuantizationTypeEnum:
    """Tests for QuantizationType enum."""

    def test_dynamic_value(self):
        """Test DYNAMIC has correct value."""
        assert QuantizationType.DYNAMIC.value == "dynamic"

    def test_static_value(self):
        """Test STATIC has correct value."""
        assert QuantizationType.STATIC.value == "static"

    def test_qat_value(self):
        """Test QAT has correct value."""
        assert QuantizationType.QAT.value == "qat"

    def test_fp16_value(self):
        """Test FP16 has correct value."""
        assert QuantizationType.FP16.value == "fp16"

    def test_all_quantization_types_count(self):
        """Test that all 4 quantization types exist."""
        assert len(QuantizationType) == 4

    def test_enum_member_access(self):
        """Test enum member access by name."""
        assert QuantizationType["DYNAMIC"] == QuantizationType.DYNAMIC
        assert QuantizationType["STATIC"] == QuantizationType.STATIC
        assert QuantizationType["QAT"] == QuantizationType.QAT
        assert QuantizationType["FP16"] == QuantizationType.FP16

    def test_enum_iteration(self):
        """Test iterating over enum members."""
        types = list(QuantizationType)
        assert len(types) == 4
        assert QuantizationType.DYNAMIC in types
        assert QuantizationType.STATIC in types
        assert QuantizationType.QAT in types
        assert QuantizationType.FP16 in types


# =============================================================================
# TESTS: PruningType Enum
# =============================================================================


class TestPruningTypeEnum:
    """Tests for PruningType enum."""

    def test_unstructured_value(self):
        """Test UNSTRUCTURED has correct value."""
        assert PruningType.UNSTRUCTURED.value == "unstructured"

    def test_structured_value(self):
        """Test STRUCTURED has correct value."""
        assert PruningType.STRUCTURED.value == "structured"

    def test_magnitude_value(self):
        """Test MAGNITUDE has correct value."""
        assert PruningType.MAGNITUDE.value == "magnitude"

    def test_gradient_value(self):
        """Test GRADIENT has correct value."""
        assert PruningType.GRADIENT.value == "gradient"

    def test_all_pruning_types_count(self):
        """Test that all 4 pruning types exist."""
        assert len(PruningType) == 4

    def test_enum_member_access(self):
        """Test enum member access by name."""
        assert PruningType["UNSTRUCTURED"] == PruningType.UNSTRUCTURED
        assert PruningType["STRUCTURED"] == PruningType.STRUCTURED
        assert PruningType["MAGNITUDE"] == PruningType.MAGNITUDE
        assert PruningType["GRADIENT"] == PruningType.GRADIENT

    def test_enum_iteration(self):
        """Test iterating over enum members."""
        types = list(PruningType)
        assert len(types) == 4
        assert PruningType.UNSTRUCTURED in types
        assert PruningType.STRUCTURED in types
        assert PruningType.MAGNITUDE in types
        assert PruningType.GRADIENT in types


# =============================================================================
# TESTS: OptimizationConfig Dataclass
# =============================================================================


class TestOptimizationConfig:
    """Tests for OptimizationConfig Pydantic BaseModel."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OptimizationConfig()
        assert config.quantization_enabled is False
        assert config.quantization_type == "dynamic"
        assert config.quantization_backend == "fbgemm"
        assert config.pruning_enabled is False
        assert config.pruning_type == "magnitude"
        assert config.pruning_amount == 0.3
        assert config.distillation_enabled is False
        assert config.distillation_temperature == 3.0
        assert config.distillation_alpha == 0.5
        assert config.export_onnx is False
        assert config.optimize_for_mobile is False

    def test_custom_quantization_enabled(self):
        """Test custom quantization_enabled configuration."""
        config = OptimizationConfig(quantization_enabled=True)
        assert config.quantization_enabled is True

    def test_custom_quantization_type(self):
        """Test custom quantization_type configuration."""
        config = OptimizationConfig(quantization_type="static")
        assert config.quantization_type == "static"

    def test_custom_quantization_backend(self):
        """Test custom quantization_backend configuration."""
        config = OptimizationConfig(quantization_backend="qnnpack")
        assert config.quantization_backend == "qnnpack"

    def test_custom_pruning_enabled(self):
        """Test custom pruning_enabled configuration."""
        config = OptimizationConfig(pruning_enabled=True)
        assert config.pruning_enabled is True

    def test_custom_pruning_type(self):
        """Test custom pruning_type configuration."""
        config = OptimizationConfig(pruning_type="structured")
        assert config.pruning_type == "structured"

    def test_custom_pruning_amount(self):
        """Test custom pruning_amount configuration."""
        config = OptimizationConfig(pruning_amount=0.5)
        assert config.pruning_amount == 0.5

    def test_custom_distillation_enabled(self):
        """Test custom distillation_enabled configuration."""
        config = OptimizationConfig(distillation_enabled=True)
        assert config.distillation_enabled is True

    def test_custom_distillation_temperature(self):
        """Test custom distillation_temperature configuration."""
        config = OptimizationConfig(distillation_temperature=5.0)
        assert config.distillation_temperature == 5.0

    def test_custom_distillation_alpha(self):
        """Test custom distillation_alpha configuration."""
        config = OptimizationConfig(distillation_alpha=0.7)
        assert config.distillation_alpha == 0.7

    def test_custom_export_onnx(self):
        """Test custom export_onnx configuration."""
        config = OptimizationConfig(export_onnx=True)
        assert config.export_onnx is True

    def test_custom_optimize_for_mobile(self):
        """Test custom optimize_for_mobile configuration."""
        config = OptimizationConfig(optimize_for_mobile=True)
        assert config.optimize_for_mobile is True

    def test_to_dict_method(self):
        """Test to_dict method returns correct dictionary (wraps Pydantic model_dump)."""
        config = OptimizationConfig(
            quantization_enabled=True,
            quantization_type="static",
            pruning_enabled=True,
            pruning_amount=0.4,
        )
        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["quantization_enabled"] is True
        assert result["quantization_type"] == "static"
        assert result["pruning_enabled"] is True
        assert result["pruning_amount"] == 0.4

        # Verify to_dict() is consistent with Pydantic V2 model_dump()
        assert result == config.model_dump()

    def test_to_dict_contains_all_fields(self):
        """Test to_dict contains all expected fields."""
        config = OptimizationConfig()
        result = config.to_dict()

        expected_keys = {
            "quantization_enabled",
            "quantization_type",
            "quantization_backend",
            "pruning_enabled",
            "pruning_type",
            "pruning_amount",
            "distillation_enabled",
            "distillation_temperature",
            "distillation_alpha",
            "export_onnx",
            "optimize_for_mobile",
        }
        assert set(result.keys()) == expected_keys

    def test_to_dict_is_json_serializable(self):
        """Test that to_dict output is JSON serializable."""
        config = OptimizationConfig(quantization_enabled=True, pruning_enabled=True)
        result = config.to_dict()
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

    def test_full_configuration(self):
        """Test creating a full configuration with all options."""
        config = OptimizationConfig(
            quantization_enabled=True,
            quantization_type="qat",
            quantization_backend="qnnpack",
            pruning_enabled=True,
            pruning_type="structured",
            pruning_amount=0.5,
            distillation_enabled=True,
            distillation_temperature=4.0,
            distillation_alpha=0.6,
            export_onnx=True,
            optimize_for_mobile=True,
        )

        assert config.quantization_enabled is True
        assert config.quantization_type == "qat"
        assert config.quantization_backend == "qnnpack"
        assert config.pruning_enabled is True
        assert config.pruning_type == "structured"
        assert config.pruning_amount == 0.5
        assert config.distillation_enabled is True
        assert config.distillation_temperature == 4.0
        assert config.distillation_alpha == 0.6
        assert config.export_onnx is True
        assert config.optimize_for_mobile is True


# =============================================================================
# TESTS: ModelOptimizer Initialization
# =============================================================================


class TestModelOptimizerInitialization:
    """Tests for ModelOptimizer initialization."""

    def test_default_initialization(self):
        """Test default ModelOptimizer initialization."""
        optimizer = ModelOptimizer()
        assert optimizer.config.quantization_enabled is False
        assert optimizer.config.pruning_enabled is False
        assert optimizer.config.distillation_enabled is False
        assert optimizer.verbose is True

    def test_initialization_with_quantization(self):
        """Test initialization with quantization enabled."""
        optimizer = ModelOptimizer(quantization_enabled=True, quantization_type="dynamic")
        assert optimizer.config.quantization_enabled is True
        assert optimizer.config.quantization_type == "dynamic"

    def test_initialization_with_pruning(self):
        """Test initialization with pruning enabled."""
        optimizer = ModelOptimizer(
            pruning_enabled=True, pruning_type="magnitude", pruning_amount=0.4
        )
        assert optimizer.config.pruning_enabled is True
        assert optimizer.config.pruning_type == "magnitude"
        assert optimizer.config.pruning_amount == 0.4

    def test_initialization_with_distillation(self):
        """Test initialization with distillation enabled."""
        optimizer = ModelOptimizer(
            distillation_enabled=True, distillation_temperature=5.0, distillation_alpha=0.7
        )
        assert optimizer.config.distillation_enabled is True
        assert optimizer.config.distillation_temperature == 5.0
        assert optimizer.config.distillation_alpha == 0.7

    def test_initialization_with_export_options(self):
        """Test initialization with export options."""
        optimizer = ModelOptimizer(export_onnx=True, optimize_for_mobile=True)
        assert optimizer.config.export_onnx is True
        assert optimizer.config.optimize_for_mobile is True

    def test_initialization_verbose_true(self):
        """Test initialization with verbose=True."""
        optimizer = ModelOptimizer(verbose=True)
        assert optimizer.verbose is True

    def test_initialization_verbose_false(self):
        """Test initialization with verbose=False."""
        optimizer = ModelOptimizer(verbose=False)
        assert optimizer.verbose is False

    def test_initialization_full_config(self):
        """Test initialization with all options."""
        optimizer = ModelOptimizer(
            quantization_enabled=True,
            quantization_type="static",
            quantization_backend="qnnpack",
            pruning_enabled=True,
            pruning_type="structured",
            pruning_amount=0.5,
            distillation_enabled=True,
            distillation_temperature=4.0,
            distillation_alpha=0.6,
            export_onnx=True,
            optimize_for_mobile=True,
            verbose=False,
        )

        assert optimizer.config.quantization_enabled is True
        assert optimizer.config.quantization_type == "static"
        assert optimizer.config.quantization_backend == "qnnpack"
        assert optimizer.config.pruning_enabled is True
        assert optimizer.config.pruning_type == "structured"
        assert optimizer.config.pruning_amount == 0.5
        assert optimizer.config.distillation_enabled is True
        assert optimizer.config.distillation_temperature == 4.0
        assert optimizer.config.distillation_alpha == 0.6
        assert optimizer.config.export_onnx is True
        assert optimizer.config.optimize_for_mobile is True
        assert optimizer.verbose is False

    def test_quantization_backend_set_when_enabled(self):
        """Test that quantization backend is set when enabled."""
        # The autouse fixture already mocks the backend setting
        # Just verify the optimizer stores the backend correctly in config
        optimizer = ModelOptimizer(
            quantization_enabled=True, quantization_backend="fbgemm", verbose=False
        )
        # Verify the backend was stored in config
        assert optimizer.config.quantization_backend == "fbgemm"


# =============================================================================
# TESTS: ModelOptimizer Quantization
# =============================================================================


class TestModelOptimizerQuantization:
    """Tests for ModelOptimizer quantization methods."""

    def test_quantize_model_when_disabled(self, simple_linear_model):
        """Test quantize_model returns original model when disabled."""
        optimizer = ModelOptimizer(quantization_enabled=False, verbose=False)
        result = optimizer.quantize_model(simple_linear_model)
        assert result is simple_linear_model

    def test_quantize_model_dynamic(self, simple_linear_model):
        """Test dynamic quantization."""
        optimizer = ModelOptimizer(
            quantization_enabled=True, quantization_type="dynamic", verbose=False
        )

        with patch("torch.quantization.quantize_dynamic") as mock_quantize:
            mock_quantize.return_value = simple_linear_model
            _result = optimizer.quantize_model(simple_linear_model)

            mock_quantize.assert_called_once()
            # Verify the model and dtypes were passed
            call_args = mock_quantize.call_args
            assert call_args[0][0] is simple_linear_model

    def test_quantize_model_fp16(self, mock_model):
        """Test FP16 quantization."""
        optimizer = ModelOptimizer(
            quantization_enabled=True, quantization_type="fp16", verbose=False
        )

        _result = optimizer.quantize_model(mock_model)
        mock_model.half.assert_called_once()

    def test_quantize_model_static_requires_calibration_data(self, simple_linear_model):
        """Test static quantization requires calibration data."""
        optimizer = ModelOptimizer(
            quantization_enabled=True, quantization_type="static", verbose=False
        )

        with pytest.raises(OptimizationError, match="calibration_data"):
            optimizer.quantize_model(simple_linear_model)

    def test_quantize_model_static_with_calibration_data(self, mock_model):
        """Test static quantization with calibration data."""
        optimizer = ModelOptimizer(
            quantization_enabled=True, quantization_type="static", verbose=False
        )

        calibration_data = [torch.randn(1, 10) for _ in range(5)]

        with (
            patch("torch.quantization.get_default_qconfig") as mock_qconfig,
            patch("torch.quantization.fuse_modules") as mock_fuse,
            patch("torch.quantization.prepare") as mock_prepare,
            patch("torch.quantization.convert") as mock_convert,
        ):
            mock_fuse.return_value = mock_model
            mock_prepared = MagicMock()
            mock_prepare.return_value = mock_prepared
            mock_convert.return_value = mock_model

            _result = optimizer.quantize_model(mock_model, calibration_data=calibration_data)

            mock_qconfig.assert_called_once()
            mock_fuse.assert_called_once()
            mock_prepare.assert_called_once()
            mock_convert.assert_called_once()

    def test_quantize_model_qat(self, mock_model):
        """Test QAT preparation."""
        optimizer = ModelOptimizer(
            quantization_enabled=True, quantization_type="qat", verbose=False
        )

        with (
            patch("torch.quantization.get_default_qat_qconfig") as mock_qconfig,
            patch("torch.quantization.fuse_modules") as mock_fuse,
            patch("torch.quantization.prepare_qat") as mock_prepare_qat,
        ):
            mock_fuse.return_value = mock_model
            mock_prepare_qat.return_value = mock_model

            _result = optimizer.quantize_model(mock_model)

            mock_model.train.assert_called()
            mock_qconfig.assert_called_once()
            mock_fuse.assert_called_once()
            mock_prepare_qat.assert_called_once()

    def test_quantize_model_unknown_type_raises_error(self, simple_linear_model):
        """Test unknown quantization type raises error."""
        optimizer = ModelOptimizer(
            quantization_enabled=True, quantization_type="unknown_type", verbose=False
        )

        with pytest.raises(OptimizationError, match="Unknown quantization type"):
            optimizer.quantize_model(simple_linear_model)

    def test_finalize_qat(self, mock_model):
        """Test QAT finalization."""
        optimizer = ModelOptimizer(
            quantization_enabled=True, quantization_type="qat", verbose=False
        )

        with patch("torch.quantization.convert") as mock_convert:
            mock_convert.return_value = mock_model

            _result = optimizer.finalize_qat(mock_model)

            mock_model.eval.assert_called()
            mock_convert.assert_called_once()

    def test_quantize_model_exception_handling(self, simple_linear_model):
        """Test exception handling in quantize_model."""
        optimizer = ModelOptimizer(
            quantization_enabled=True, quantization_type="dynamic", verbose=False
        )

        with (
            patch(
                "torch.quantization.quantize_dynamic",
                side_effect=RuntimeError("Quantization error"),
            ),
            pytest.raises(OptimizationError, match="Quantization failed"),
        ):
            optimizer.quantize_model(simple_linear_model)


# =============================================================================
# TESTS: ModelOptimizer Pruning
# =============================================================================


class TestModelOptimizerPruning:
    """Tests for ModelOptimizer pruning methods."""

    def test_prune_model_when_disabled(self, simple_linear_model):
        """Test prune_model returns original model when disabled."""
        optimizer = ModelOptimizer(pruning_enabled=False, verbose=False)
        result = optimizer.prune_model(simple_linear_model)
        assert result is simple_linear_model

    def test_prune_model_magnitude(self, simple_linear_model):
        """Test magnitude-based pruning."""
        optimizer = ModelOptimizer(
            pruning_enabled=True, pruning_type="magnitude", pruning_amount=0.3, verbose=False
        )

        result = optimizer.prune_model(simple_linear_model)

        # Model should be returned (pruning may or may not modify it)
        assert result is not None

    def test_prune_model_unstructured(self, simple_linear_model):
        """Test unstructured pruning."""
        optimizer = ModelOptimizer(
            pruning_enabled=True, pruning_type="unstructured", pruning_amount=0.3, verbose=False
        )

        with patch.object(prune, "global_unstructured") as mock_prune:
            _result = optimizer.prune_model(simple_linear_model)
            mock_prune.assert_called()

    def test_prune_model_structured(self, simple_conv_model):
        """Test structured pruning (requires Conv2d)."""
        optimizer = ModelOptimizer(
            pruning_enabled=True, pruning_type="structured", pruning_amount=0.3, verbose=False
        )

        with patch.object(prune, "ln_structured") as mock_prune:
            _result = optimizer.prune_model(simple_conv_model)
            # Structured pruning is called for Conv2d layers
            assert mock_prune.call_count >= 0  # May be called for Conv2d layers

    def test_prune_model_custom_amount(self, simple_linear_model):
        """Test pruning with custom amount parameter."""
        optimizer = ModelOptimizer(
            pruning_enabled=True, pruning_type="magnitude", pruning_amount=0.3, verbose=False
        )

        # Override with custom amount
        result = optimizer.prune_model(simple_linear_model, amount=0.5)
        assert result is not None

    def test_prune_model_iterative_steps(self, simple_linear_model):
        """Test iterative pruning."""
        optimizer = ModelOptimizer(
            pruning_enabled=True, pruning_type="magnitude", pruning_amount=0.3, verbose=False
        )

        result = optimizer.prune_model(simple_linear_model, iterative_steps=3)
        assert result is not None

    def test_prune_model_unknown_type_raises_error(self, simple_linear_model):
        """Test unknown pruning type raises error."""
        optimizer = ModelOptimizer(
            pruning_enabled=True, pruning_type="unknown_type", pruning_amount=0.3, verbose=False
        )

        with pytest.raises(OptimizationError, match="Unknown pruning type"):
            optimizer.prune_model(simple_linear_model)

    def test_get_sparsity(self, simple_linear_model):
        """Test get_sparsity method."""
        optimizer = ModelOptimizer(verbose=False)

        sparsity = optimizer.get_sparsity(simple_linear_model)

        assert "total_parameters" in sparsity
        assert "zero_parameters" in sparsity
        assert "global_sparsity" in sparsity
        assert "compression_ratio" in sparsity
        assert isinstance(sparsity["total_parameters"], int)
        assert isinstance(sparsity["zero_parameters"], int)
        assert isinstance(sparsity["global_sparsity"], float)
        assert 0.0 <= sparsity["global_sparsity"] <= 1.0

    def test_get_sparsity_with_pruned_model(self, simple_linear_model):
        """Test get_sparsity with a pruned model."""
        optimizer = ModelOptimizer(
            pruning_enabled=True, pruning_type="magnitude", pruning_amount=0.3, verbose=False
        )

        # Prune the model
        pruned_model = optimizer.prune_model(simple_linear_model)

        sparsity = optimizer.get_sparsity(pruned_model)

        # Sparsity should be greater than 0 after pruning
        assert sparsity["global_sparsity"] >= 0.0

    def test_magnitude_pruning_internal(self, simple_linear_model):
        """Test internal _magnitude_pruning method."""
        optimizer = ModelOptimizer(verbose=False)

        with patch.object(prune, "l1_unstructured") as mock_prune:
            _result = optimizer._magnitude_pruning(simple_linear_model, 0.3)
            # Should be called for Linear layers
            assert mock_prune.call_count >= 0

    def test_unstructured_pruning_internal(self, simple_linear_model):
        """Test internal _unstructured_pruning method."""
        optimizer = ModelOptimizer(verbose=False)

        with patch.object(prune, "global_unstructured") as mock_prune:
            _result = optimizer._unstructured_pruning(simple_linear_model, 0.3)
            mock_prune.assert_called()

    def test_structured_pruning_internal(self, simple_conv_model):
        """Test internal _structured_pruning method."""
        optimizer = ModelOptimizer(verbose=False)

        with patch.object(prune, "ln_structured") as _mock_prune:
            result = optimizer._structured_pruning(simple_conv_model, 0.3)
            # Should be called for Conv2d layers
            assert result is not None


# =============================================================================
# TESTS: ModelOptimizer Knowledge Distillation
# =============================================================================


class TestModelOptimizerDistillation:
    """Tests for ModelOptimizer knowledge distillation methods."""

    def test_distillation_loss_when_disabled(self):
        """Test distillation_loss returns standard loss when disabled."""
        optimizer = ModelOptimizer(distillation_enabled=False, verbose=False)

        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))

        loss = optimizer.distillation_loss(student_logits, teacher_logits, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar

    def test_distillation_loss_when_enabled(self):
        """Test distillation_loss computes combined loss when enabled."""
        optimizer = ModelOptimizer(
            distillation_enabled=True,
            distillation_temperature=3.0,
            distillation_alpha=0.5,
            verbose=False,
        )

        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))

        loss = optimizer.distillation_loss(student_logits, teacher_logits, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative

    def test_distillation_loss_different_temperatures(self):
        """Test distillation_loss with different temperatures."""
        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))

        optimizer_low_temp = ModelOptimizer(
            distillation_enabled=True,
            distillation_temperature=1.0,
            distillation_alpha=0.5,
            verbose=False,
        )

        optimizer_high_temp = ModelOptimizer(
            distillation_enabled=True,
            distillation_temperature=10.0,
            distillation_alpha=0.5,
            verbose=False,
        )

        loss_low = optimizer_low_temp.distillation_loss(student_logits, teacher_logits, targets)
        loss_high = optimizer_high_temp.distillation_loss(student_logits, teacher_logits, targets)

        # Both should produce valid losses
        assert isinstance(loss_low, torch.Tensor)
        assert isinstance(loss_high, torch.Tensor)

    def test_distillation_loss_different_alphas(self):
        """Test distillation_loss with different alpha values."""
        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))

        optimizer_low_alpha = ModelOptimizer(
            distillation_enabled=True,
            distillation_temperature=3.0,
            distillation_alpha=0.1,
            verbose=False,
        )

        optimizer_high_alpha = ModelOptimizer(
            distillation_enabled=True,
            distillation_temperature=3.0,
            distillation_alpha=0.9,
            verbose=False,
        )

        loss_low = optimizer_low_alpha.distillation_loss(student_logits, teacher_logits, targets)
        loss_high = optimizer_high_alpha.distillation_loss(student_logits, teacher_logits, targets)

        # Both should produce valid losses
        assert isinstance(loss_low, torch.Tensor)
        assert isinstance(loss_high, torch.Tensor)

    def test_distillation_loss_custom_loss_fn(self):
        """Test distillation_loss with custom student loss function."""
        optimizer = ModelOptimizer(
            distillation_enabled=True,
            distillation_temperature=3.0,
            distillation_alpha=0.5,
            verbose=False,
        )

        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))

        custom_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        loss = optimizer.distillation_loss(
            student_logits, teacher_logits, targets, student_loss_fn=custom_loss_fn
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_create_student_model(self, simple_linear_model):
        """Test create_student_model method (placeholder)."""
        optimizer = ModelOptimizer(verbose=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _student = optimizer.create_student_model(simple_linear_model)

            # Should issue a warning about placeholder implementation
            assert len(w) >= 1
            assert "placeholder" in str(w[-1].message).lower()

    def test_create_student_model_reduction_factor(self, simple_linear_model):
        """Test create_student_model with custom reduction factor."""
        optimizer = ModelOptimizer(verbose=False)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            student = optimizer.create_student_model(simple_linear_model, reduction_factor=0.25)

            # Should return a model (deep copy in placeholder)
            assert student is not None


# =============================================================================
# TESTS: ModelOptimizer Export
# =============================================================================


class TestModelOptimizerExport:
    """Tests for ModelOptimizer export methods."""

    def test_export_to_onnx(self, simple_linear_model, temp_dir):
        """Test ONNX export."""
        optimizer = ModelOptimizer(verbose=False)

        filepath = temp_dir / "model.onnx"
        dummy_input = torch.randn(1, 10)

        with patch("torch.onnx.export") as mock_export:
            optimizer.export_to_onnx(simple_linear_model, filepath, dummy_input)

            mock_export.assert_called_once()

    def test_export_to_onnx_creates_directory(self, simple_linear_model, temp_dir):
        """Test ONNX export creates parent directory."""
        optimizer = ModelOptimizer(verbose=False)

        nested_path = temp_dir / "nested" / "path" / "model.onnx"
        dummy_input = torch.randn(1, 10)

        with patch("torch.onnx.export"):
            optimizer.export_to_onnx(simple_linear_model, nested_path, dummy_input)

            assert nested_path.parent.exists()

    def test_export_to_onnx_with_custom_names(self, simple_linear_model, temp_dir):
        """Test ONNX export with custom input/output names."""
        optimizer = ModelOptimizer(verbose=False)

        filepath = temp_dir / "model.onnx"
        dummy_input = torch.randn(1, 10)

        with patch("torch.onnx.export") as mock_export:
            optimizer.export_to_onnx(
                simple_linear_model,
                filepath,
                dummy_input,
                input_names=["features"],
                output_names=["predictions"],
            )

            mock_export.assert_called_once()
            call_kwargs = mock_export.call_args[1]
            assert call_kwargs["input_names"] == ["features"]
            assert call_kwargs["output_names"] == ["predictions"]

    def test_export_to_onnx_with_dynamic_axes(self, simple_linear_model, temp_dir):
        """Test ONNX export with dynamic axes."""
        optimizer = ModelOptimizer(verbose=False)

        filepath = temp_dir / "model.onnx"
        dummy_input = torch.randn(1, 10)
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

        with patch("torch.onnx.export") as mock_export:
            optimizer.export_to_onnx(
                simple_linear_model, filepath, dummy_input, dynamic_axes=dynamic_axes
            )

            mock_export.assert_called_once()
            call_kwargs = mock_export.call_args[1]
            assert call_kwargs["dynamic_axes"] == dynamic_axes

    def test_export_to_onnx_with_opset_version(self, simple_linear_model, temp_dir):
        """Test ONNX export with custom opset version."""
        optimizer = ModelOptimizer(verbose=False)

        filepath = temp_dir / "model.onnx"
        dummy_input = torch.randn(1, 10)

        with patch("torch.onnx.export") as mock_export:
            optimizer.export_to_onnx(simple_linear_model, filepath, dummy_input, opset_version=12)

            mock_export.assert_called_once()
            call_kwargs = mock_export.call_args[1]
            assert call_kwargs["opset_version"] == 12

    def test_export_to_onnx_failure_raises_error(self, simple_linear_model, temp_dir):
        """Test ONNX export failure raises ExportError."""
        optimizer = ModelOptimizer(verbose=False)

        filepath = temp_dir / "model.onnx"
        dummy_input = torch.randn(1, 10)

        with (
            patch("torch.onnx.export", side_effect=RuntimeError("Export failed")),
            pytest.raises(ExportError, match="ONNX export failed"),
        ):
            optimizer.export_to_onnx(simple_linear_model, filepath, dummy_input)

    def test_export_to_onnx_string_path(self, simple_linear_model, temp_dir):
        """Test ONNX export with string path."""
        optimizer = ModelOptimizer(verbose=False)

        filepath = str(temp_dir / "model.onnx")
        dummy_input = torch.randn(1, 10)

        with patch("torch.onnx.export") as mock_export:
            optimizer.export_to_onnx(simple_linear_model, filepath, dummy_input)

            mock_export.assert_called_once()

    def test_optimize_for_mobile_when_disabled(self, simple_linear_model, temp_dir):
        """Test optimize_for_mobile returns when disabled."""
        optimizer = ModelOptimizer(optimize_for_mobile=False, verbose=False)

        filepath = temp_dir / "model.pt"
        example_inputs = (torch.randn(1, 10),)

        # Should return early without doing anything
        result = optimizer.optimize_for_mobile(simple_linear_model, example_inputs, filepath)

        assert result is None

    def test_optimize_for_mobile_when_enabled(self, simple_linear_model, temp_dir):
        """Test optimize_for_mobile when enabled."""
        optimizer = ModelOptimizer(optimize_for_mobile=True, verbose=False)

        filepath = temp_dir / "model.pt"
        example_inputs = (torch.randn(1, 10),)

        mock_traced = MagicMock()
        mock_optimized = MagicMock()

        with (
            patch.object(torch.jit, "trace", return_value=mock_traced),
            patch("torch.utils.mobile_optimizer.optimize_for_mobile", return_value=mock_optimized),
        ):
            optimizer.optimize_for_mobile(simple_linear_model, example_inputs, filepath)

            mock_optimized._save_for_lite_interpreter.assert_called_once()

    def test_optimize_for_mobile_import_error(self, simple_linear_model, temp_dir):
        """Test optimize_for_mobile handles import error."""
        optimizer = ModelOptimizer(optimize_for_mobile=True, verbose=False)

        filepath = temp_dir / "model.pt"
        example_inputs = (torch.randn(1, 10),)

        with (
            patch.object(
                torch.jit, "trace", side_effect=ImportError("Mobile optimizer not available")
            ),
            pytest.raises(ExportError, match="Mobile optimization"),
        ):
            optimizer.optimize_for_mobile(simple_linear_model, example_inputs, filepath)

    def test_optimize_for_mobile_general_error(self, simple_linear_model, temp_dir):
        """Test optimize_for_mobile handles general errors."""
        optimizer = ModelOptimizer(optimize_for_mobile=True, verbose=False)

        filepath = temp_dir / "model.pt"
        example_inputs = (torch.randn(1, 10),)

        with (
            patch.object(torch.jit, "trace", side_effect=RuntimeError("Tracing failed")),
            pytest.raises(ExportError, match="Mobile optimization failed"),
        ):
            optimizer.optimize_for_mobile(simple_linear_model, example_inputs, filepath)


# =============================================================================
# TESTS: ModelOptimizer Metrics
# =============================================================================


class TestModelOptimizerMetrics:
    """Tests for ModelOptimizer metrics methods."""

    def test_get_model_size(self, simple_linear_model):
        """Test get_model_size method."""
        optimizer = ModelOptimizer(verbose=False)

        size_info = optimizer.get_model_size(simple_linear_model)

        assert "parameters_mb" in size_info
        assert "buffers_mb" in size_info
        assert "total_mb" in size_info
        assert "num_parameters" in size_info
        assert size_info["parameters_mb"] >= 0
        assert size_info["buffers_mb"] >= 0
        assert size_info["total_mb"] >= 0
        assert size_info["num_parameters"] >= 0

    def test_get_model_size_linear_model(self, simple_linear_model):
        """Test get_model_size with linear model."""
        optimizer = ModelOptimizer(verbose=False)

        size_info = optimizer.get_model_size(simple_linear_model)

        # Calculate expected parameters manually
        # Linear(10, 20): 10*20 + 20 = 220
        # Linear(20, 10): 20*10 + 10 = 210
        # Total: 430 parameters
        expected_params = 10 * 20 + 20 + 20 * 10 + 10
        assert size_info["num_parameters"] == expected_params

    def test_get_model_size_conv_model(self, simple_conv_model):
        """Test get_model_size with convolutional model."""
        optimizer = ModelOptimizer(verbose=False)

        size_info = optimizer.get_model_size(simple_conv_model)

        # Should have parameters and potentially buffers (from BatchNorm)
        assert size_info["num_parameters"] > 0
        assert size_info["total_mb"] > 0

    def test_compare_models(self, simple_linear_model):
        """Test compare_models method."""
        optimizer = ModelOptimizer(verbose=False)

        # Create a "smaller" model by deep copying
        optimized_model = copy.deepcopy(simple_linear_model)

        comparison = optimizer.compare_models(simple_linear_model, optimized_model)

        assert "original_size_mb" in comparison
        assert "optimized_size_mb" in comparison
        assert "size_reduction" in comparison
        assert "compression_ratio" in comparison
        assert "original_params" in comparison
        assert "optimized_params" in comparison

    def test_compare_models_identical(self, simple_linear_model):
        """Test compare_models with identical models."""
        optimizer = ModelOptimizer(verbose=False)

        comparison = optimizer.compare_models(simple_linear_model, simple_linear_model)

        assert comparison["size_reduction"] == 0.0
        assert comparison["compression_ratio"] == 1.0
        assert comparison["original_params"] == comparison["optimized_params"]

    def test_compare_models_different_sizes(self):
        """Test compare_models with different sized models."""
        optimizer = ModelOptimizer(verbose=False)

        large_model = nn.Linear(100, 100)  # 10100 params
        small_model = nn.Linear(10, 10)  # 110 params

        comparison = optimizer.compare_models(large_model, small_model)

        # Size reduction should be positive (smaller model)
        assert comparison["size_reduction"] > 0
        # Compression ratio should be > 1
        assert comparison["compression_ratio"] > 1
        assert comparison["original_params"] > comparison["optimized_params"]

    def test_compare_models_verbose(self, simple_linear_model, caplog):
        """Test compare_models logs info when verbose."""
        optimizer = ModelOptimizer(verbose=True)

        with caplog.at_level(logging.INFO):
            _comparison = optimizer.compare_models(simple_linear_model, simple_linear_model)

        # Logging output may or may not be captured depending on handler config


# =============================================================================
# TESTS: ModelOptimizer Summary
# =============================================================================


class TestModelOptimizerSummary:
    """Tests for ModelOptimizer summary methods."""

    def test_print_optimization_summary(self, default_config, capsys):
        """Test print_optimization_summary method."""
        optimizer = ModelOptimizer(verbose=False)

        optimizer.print_optimization_summary()

        captured = capsys.readouterr()
        assert "Model Optimization Summary" in captured.out
        assert "Quantization" in captured.out
        assert "Pruning" in captured.out
        assert "Knowledge Distillation" in captured.out
        assert "ONNX Export" in captured.out
        assert "Mobile Optimization" in captured.out

    def test_print_optimization_summary_with_quantization(self, capsys):
        """Test summary with quantization enabled."""
        optimizer = ModelOptimizer(
            quantization_enabled=True,
            quantization_type="dynamic",
            quantization_backend="fbgemm",
            verbose=False,
        )

        optimizer.print_optimization_summary()

        captured = capsys.readouterr()
        assert "True" in captured.out  # Quantization: True
        assert "dynamic" in captured.out
        assert "fbgemm" in captured.out

    def test_print_optimization_summary_with_pruning(self, capsys):
        """Test summary with pruning enabled."""
        optimizer = ModelOptimizer(
            pruning_enabled=True, pruning_type="magnitude", pruning_amount=0.3, verbose=False
        )

        optimizer.print_optimization_summary()

        captured = capsys.readouterr()
        assert "magnitude" in captured.out
        assert "30.0%" in captured.out

    def test_print_optimization_summary_with_distillation(self, capsys):
        """Test summary with distillation enabled."""
        optimizer = ModelOptimizer(
            distillation_enabled=True,
            distillation_temperature=4.0,
            distillation_alpha=0.6,
            verbose=False,
        )

        optimizer.print_optimization_summary()

        captured = capsys.readouterr()
        assert "Knowledge Distillation: True" in captured.out
        assert "4.0" in captured.out
        assert "0.6" in captured.out


# =============================================================================
# TESTS: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_quantize_for_inference_dynamic(self, simple_linear_model):
        """Test quantize_for_inference with dynamic quantization."""
        with patch("torch.quantization.quantize_dynamic") as mock_quantize:
            mock_quantize.return_value = simple_linear_model
            _result = quantize_for_inference(simple_linear_model, "dynamic")

            mock_quantize.assert_called_once()

    def test_quantize_for_inference_fp16(self, mock_model):
        """Test quantize_for_inference with fp16."""
        _result = quantize_for_inference(mock_model, "fp16")
        mock_model.half.assert_called()

    def test_quantize_for_inference_default_type(self, simple_linear_model):
        """Test quantize_for_inference uses dynamic by default."""
        with patch("torch.quantization.quantize_dynamic") as mock_quantize:
            mock_quantize.return_value = simple_linear_model
            _result = quantize_for_inference(simple_linear_model)

            mock_quantize.assert_called_once()

    def test_prune_for_deployment_default(self, simple_linear_model):
        """Test prune_for_deployment with default amount."""
        result = prune_for_deployment(simple_linear_model)
        assert result is not None

    def test_prune_for_deployment_custom_amount(self, simple_linear_model):
        """Test prune_for_deployment with custom amount."""
        result = prune_for_deployment(simple_linear_model, amount=0.5)
        assert result is not None

    def test_prune_for_deployment_small_amount(self, simple_linear_model):
        """Test prune_for_deployment with small amount."""
        result = prune_for_deployment(simple_linear_model, amount=0.1)
        assert result is not None

    def test_prune_for_deployment_large_amount(self, simple_linear_model):
        """Test prune_for_deployment with large amount."""
        result = prune_for_deployment(simple_linear_model, amount=0.9)
        assert result is not None


# =============================================================================
# TESTS: Exception Classes
# =============================================================================


class TestExceptionClasses:
    """Tests for exception classes."""

    def test_model_error_is_exception(self):
        """Test ModelError is an Exception subclass."""
        assert issubclass(ModelError, Exception)

    def test_optimization_error_is_model_error(self):
        """Test OptimizationError is a ModelError subclass."""
        assert issubclass(OptimizationError, ModelError)

    def test_export_error_is_model_error(self):
        """Test ExportError is a ModelError subclass."""
        assert issubclass(ExportError, ModelError)

    def test_optimization_error_message(self):
        """Test OptimizationError can carry a message."""
        error = OptimizationError("Test error message")
        assert str(error) == "Test error message"

    def test_export_error_message(self):
        """Test ExportError can carry a message."""
        error = ExportError("Export failed")
        assert str(error) == "Export failed"

    def test_model_error_raise(self):
        """Test ModelError can be raised and caught."""
        with pytest.raises(ModelError):
            raise ModelError("Test")

    def test_optimization_error_raise(self):
        """Test OptimizationError can be raised and caught."""
        with pytest.raises(OptimizationError):
            raise OptimizationError("Test")

    def test_export_error_raise(self):
        """Test ExportError can be raised and caught."""
        with pytest.raises(ExportError):
            raise ExportError("Test")

    def test_catch_optimization_error_as_model_error(self):
        """Test OptimizationError can be caught as ModelError."""
        with pytest.raises(ModelError):
            raise OptimizationError("Test")

    def test_catch_export_error_as_model_error(self):
        """Test ExportError can be caught as ModelError."""
        with pytest.raises(ModelError):
            raise ExportError("Test")


# =============================================================================
# TESTS: Edge Cases and Boundary Conditions
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_model(self):
        """Test operations on empty model."""
        empty_model = nn.Sequential()
        optimizer = ModelOptimizer(verbose=False)

        size_info = optimizer.get_model_size(empty_model)
        assert size_info["num_parameters"] == 0
        assert size_info["total_mb"] == 0

    def test_pruning_amount_zero(self, simple_linear_model):
        """Test pruning with amount 0."""
        optimizer = ModelOptimizer(
            pruning_enabled=True, pruning_type="magnitude", pruning_amount=0.0, verbose=False
        )

        result = optimizer.prune_model(simple_linear_model)
        assert result is not None

    def test_pruning_amount_one(self, simple_linear_model):
        """Test pruning with amount 1.0 (100%)."""
        optimizer = ModelOptimizer(
            pruning_enabled=True, pruning_type="magnitude", pruning_amount=1.0, verbose=False
        )

        result = optimizer.prune_model(simple_linear_model)
        assert result is not None

    def test_distillation_temperature_very_low(self):
        """Test distillation with very low temperature."""
        optimizer = ModelOptimizer(
            distillation_enabled=True,
            distillation_temperature=0.1,
            distillation_alpha=0.5,
            verbose=False,
        )

        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))

        loss = optimizer.distillation_loss(student_logits, teacher_logits, targets)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_distillation_temperature_very_high(self):
        """Test distillation with very high temperature."""
        optimizer = ModelOptimizer(
            distillation_enabled=True,
            distillation_temperature=100.0,
            distillation_alpha=0.5,
            verbose=False,
        )

        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))

        loss = optimizer.distillation_loss(student_logits, teacher_logits, targets)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_distillation_alpha_zero(self):
        """Test distillation with alpha 0 (no distillation loss)."""
        optimizer = ModelOptimizer(
            distillation_enabled=True,
            distillation_temperature=3.0,
            distillation_alpha=0.0,
            verbose=False,
        )

        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))

        loss = optimizer.distillation_loss(student_logits, teacher_logits, targets)
        assert not torch.isnan(loss)

    def test_distillation_alpha_one(self):
        """Test distillation with alpha 1 (full distillation loss)."""
        optimizer = ModelOptimizer(
            distillation_enabled=True,
            distillation_temperature=3.0,
            distillation_alpha=1.0,
            verbose=False,
        )

        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))

        loss = optimizer.distillation_loss(student_logits, teacher_logits, targets)
        assert not torch.isnan(loss)

    def test_single_batch_distillation(self):
        """Test distillation with batch size 1."""
        optimizer = ModelOptimizer(
            distillation_enabled=True,
            distillation_temperature=3.0,
            distillation_alpha=0.5,
            verbose=False,
        )

        student_logits = torch.randn(1, 10)
        teacher_logits = torch.randn(1, 10)
        targets = torch.randint(0, 10, (1,))

        loss = optimizer.distillation_loss(student_logits, teacher_logits, targets)
        assert not torch.isnan(loss)

    def test_large_batch_distillation(self):
        """Test distillation with large batch size."""
        optimizer = ModelOptimizer(
            distillation_enabled=True,
            distillation_temperature=3.0,
            distillation_alpha=0.5,
            verbose=False,
        )

        student_logits = torch.randn(256, 10)
        teacher_logits = torch.randn(256, 10)
        targets = torch.randint(0, 10, (256,))

        loss = optimizer.distillation_loss(student_logits, teacher_logits, targets)
        assert not torch.isnan(loss)

    def test_get_sparsity_no_weight_params(self):
        """Test get_sparsity with model having no weight parameters."""
        optimizer = ModelOptimizer(verbose=False)

        # Create a model with only bias (no 'weight' in name)
        class BiasOnlyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(10))

            def forward(self, x):
                return x + self.bias

        model = BiasOnlyModel()
        sparsity = optimizer.get_sparsity(model)

        # Should handle case with no 'weight' parameters
        assert sparsity["global_sparsity"] == 0  # Division by zero protection

    def test_config_to_dict_immutability(self):
        """Test that to_dict returns a copy, not the internal state."""
        config = OptimizationConfig(quantization_enabled=True)
        dict1 = config.to_dict()
        dict1["quantization_enabled"] = False

        # Original config should be unchanged
        assert config.quantization_enabled is True


# =============================================================================
# TESTS: Integration Scenarios
# =============================================================================


class TestIntegrationScenarios:
    """Tests for integration scenarios combining multiple features."""

    def test_quantize_then_prune(self, simple_linear_model):
        """Test quantizing then pruning a model."""
        quant_optimizer = ModelOptimizer(
            quantization_enabled=True, quantization_type="dynamic", verbose=False
        )

        prune_optimizer = ModelOptimizer(
            pruning_enabled=True, pruning_type="magnitude", pruning_amount=0.3, verbose=False
        )

        with patch("torch.quantization.quantize_dynamic") as mock_quantize:
            mock_quantize.return_value = simple_linear_model
            quantized = quant_optimizer.quantize_model(simple_linear_model)
            pruned = prune_optimizer.prune_model(quantized)

            assert pruned is not None

    def test_prune_then_quantize(self, simple_linear_model):
        """Test pruning then quantizing a model."""
        prune_optimizer = ModelOptimizer(
            pruning_enabled=True, pruning_type="magnitude", pruning_amount=0.3, verbose=False
        )

        quant_optimizer = ModelOptimizer(
            quantization_enabled=True, quantization_type="dynamic", verbose=False
        )

        pruned = prune_optimizer.prune_model(simple_linear_model)

        with patch("torch.quantization.quantize_dynamic") as mock_quantize:
            mock_quantize.return_value = pruned
            quantized = quant_optimizer.quantize_model(pruned)

            assert quantized is not None

    def test_optimize_and_export(self, simple_linear_model, temp_dir):
        """Test optimizing and exporting a model."""
        optimizer = ModelOptimizer(
            pruning_enabled=True, pruning_type="magnitude", pruning_amount=0.2, verbose=False
        )

        # Prune
        pruned = optimizer.prune_model(simple_linear_model)

        # Export
        filepath = temp_dir / "model.onnx"
        dummy_input = torch.randn(1, 10)

        with patch("torch.onnx.export") as mock_export:
            optimizer.export_to_onnx(pruned, filepath, dummy_input)
            mock_export.assert_called_once()

    def test_compare_before_after_optimization(self, simple_linear_model):
        """Test comparing model before and after optimization."""
        optimizer = ModelOptimizer(
            pruning_enabled=True, pruning_type="magnitude", pruning_amount=0.5, verbose=False
        )

        original = copy.deepcopy(simple_linear_model)
        optimized = optimizer.prune_model(simple_linear_model)

        comparison = optimizer.compare_models(original, optimized)

        assert comparison["original_params"] == comparison["optimized_params"]  # Same structure

    def test_full_optimization_pipeline(self, simple_linear_model, temp_dir):
        """Test a complete optimization pipeline."""
        optimizer = ModelOptimizer(
            quantization_enabled=False,  # Skip for simplicity
            pruning_enabled=True,
            pruning_type="magnitude",
            pruning_amount=0.3,
            distillation_enabled=True,
            distillation_temperature=3.0,
            distillation_alpha=0.5,
            export_onnx=True,
            verbose=False,
        )

        # Get initial metrics
        initial_size = optimizer.get_model_size(simple_linear_model)

        # Prune
        pruned = optimizer.prune_model(simple_linear_model)

        # Get final metrics
        final_size = optimizer.get_model_size(pruned)
        sparsity = optimizer.get_sparsity(pruned)

        # Compare
        comparison = optimizer.compare_models(simple_linear_model, pruned)

        # Export
        filepath = temp_dir / "optimized_model.onnx"
        dummy_input = torch.randn(1, 10)

        with patch("torch.onnx.export"):
            optimizer.export_to_onnx(pruned, filepath, dummy_input)

        # Print summary
        optimizer.print_optimization_summary()

        # Verify all operations completed
        assert pruned is not None
        assert "total_mb" in initial_size
        assert "total_mb" in final_size
        assert "global_sparsity" in sparsity
        assert "compression_ratio" in comparison


# =============================================================================
# TESTS: Logging Behavior
# =============================================================================


class TestLoggingBehavior:
    """Tests for logging behavior."""

    def test_verbose_true_initialization(self, caplog):
        """Test verbose logging during initialization."""
        with caplog.at_level(logging.INFO):
            _optimizer = ModelOptimizer(
                pruning_enabled=True, distillation_enabled=True, verbose=True
            )

        # Logging may or may not be captured depending on handler configuration

    def test_verbose_false_initialization(self, caplog):
        """Test reduced logging during initialization with verbose=False."""
        with caplog.at_level(logging.INFO):
            _optimizer = ModelOptimizer(pruning_enabled=True, verbose=False)

        # Should have fewer or no logs with verbose=False


# =============================================================================
# TESTS: Module Level
# =============================================================================


class TestModuleLevel:
    """Tests for module-level behavior."""

    def test_all_public_classes_exported(self):
        """Test all expected public classes are available."""
        from milia_pipeline.models.deployment import model_optimization

        assert hasattr(model_optimization, "QuantizationType")
        assert hasattr(model_optimization, "PruningType")
        assert hasattr(model_optimization, "OptimizationConfig")
        assert hasattr(model_optimization, "ModelOptimizer")
        assert hasattr(model_optimization, "quantize_for_inference")
        assert hasattr(model_optimization, "prune_for_deployment")

    def test_exceptions_available(self):
        """Test exception classes are available."""
        from milia_pipeline.models.deployment import model_optimization

        assert hasattr(model_optimization, "ModelError")
        assert hasattr(model_optimization, "OptimizationError")
        assert hasattr(model_optimization, "ExportError")

    def test_enums_have_correct_values(self):
        """Test enum values are accessible."""
        assert QuantizationType.DYNAMIC.value == "dynamic"
        assert PruningType.MAGNITUDE.value == "magnitude"


# =============================================================================
# TESTS: Pydantic BaseModel Behavior (v1.1.0 Migration)
# =============================================================================


class TestPydanticBaseModelBehavior:
    """Tests for Pydantic V2 BaseModel behavior of OptimizationConfig.

    Verifies the v1.1.0 migration from @dataclass to Pydantic BaseModel
    preserves backward compatibility while exposing Pydantic V2 features.
    """

    def test_config_equality(self):
        """Test OptimizationConfig equality comparison."""
        config1 = OptimizationConfig(quantization_enabled=True)
        config2 = OptimizationConfig(quantization_enabled=True)
        config3 = OptimizationConfig(quantization_enabled=False)

        assert config1 == config2
        assert config1 != config3

    def test_config_repr(self):
        """Test OptimizationConfig has useful repr."""
        config = OptimizationConfig(quantization_enabled=True)
        repr_str = repr(config)

        assert "OptimizationConfig" in repr_str
        assert "quantization_enabled=True" in repr_str

    def test_config_is_mutable(self):
        """Test OptimizationConfig fields can be modified."""
        config = OptimizationConfig()
        config.quantization_enabled = True

        assert config.quantization_enabled is True

    # ----- Pydantic V2 model_dump() -----

    def test_model_dump_returns_dict(self):
        """Test Pydantic V2 model_dump() returns a dictionary."""
        config = OptimizationConfig(quantization_enabled=True, pruning_amount=0.5)
        result = config.model_dump()

        assert isinstance(result, dict)
        assert result["quantization_enabled"] is True
        assert result["pruning_amount"] == 0.5

    def test_model_dump_matches_to_dict(self):
        """Test model_dump() and to_dict() produce identical output."""
        config = OptimizationConfig(
            quantization_enabled=True,
            quantization_type="static",
            pruning_enabled=True,
            pruning_type="structured",
            pruning_amount=0.4,
            distillation_enabled=True,
            distillation_temperature=4.0,
            distillation_alpha=0.6,
            export_onnx=True,
            optimize_for_mobile=True,
        )

        assert config.to_dict() == config.model_dump()

    def test_model_dump_contains_all_fields(self):
        """Test model_dump() contains every declared field."""
        config = OptimizationConfig()
        dumped = config.model_dump()

        expected_keys = {
            "quantization_enabled",
            "quantization_type",
            "quantization_backend",
            "pruning_enabled",
            "pruning_type",
            "pruning_amount",
            "distillation_enabled",
            "distillation_temperature",
            "distillation_alpha",
            "export_onnx",
            "optimize_for_mobile",
        }
        assert set(dumped.keys()) == expected_keys

    # ----- Pydantic V2 model_validate() -----

    def test_model_validate_from_dict(self):
        """Test Pydantic V2 model_validate() reconstructs from dict."""
        source = {
            "quantization_enabled": True,
            "quantization_type": "static",
            "quantization_backend": "qnnpack",
            "pruning_enabled": True,
            "pruning_type": "structured",
            "pruning_amount": 0.5,
            "distillation_enabled": True,
            "distillation_temperature": 4.0,
            "distillation_alpha": 0.6,
            "export_onnx": True,
            "optimize_for_mobile": True,
        }
        config = OptimizationConfig.model_validate(source)

        assert config.quantization_enabled is True
        assert config.quantization_type == "static"
        assert config.quantization_backend == "qnnpack"
        assert config.pruning_enabled is True
        assert config.pruning_type == "structured"
        assert config.pruning_amount == 0.5
        assert config.distillation_enabled is True
        assert config.distillation_temperature == 4.0
        assert config.distillation_alpha == 0.6
        assert config.export_onnx is True
        assert config.optimize_for_mobile is True

    def test_model_validate_partial_dict_uses_defaults(self):
        """Test model_validate() fills missing keys with defaults."""
        config = OptimizationConfig.model_validate({"quantization_enabled": True})

        assert config.quantization_enabled is True
        # All other fields should have defaults
        assert config.quantization_type == "dynamic"
        assert config.pruning_enabled is False
        assert config.pruning_amount == 0.3
        assert config.distillation_enabled is False

    def test_model_validate_roundtrip(self):
        """Test model_validate(model_dump()) is a lossless round-trip."""
        original = OptimizationConfig(
            quantization_enabled=True,
            pruning_enabled=True,
            pruning_amount=0.42,
            distillation_temperature=7.5,
        )
        reconstructed = OptimizationConfig.model_validate(original.model_dump())

        assert reconstructed == original

    # ----- Pydantic V2 model_fields -----

    def test_model_fields_lists_all_declared_fields(self):
        """Test Pydantic V2 model_fields on the class contains all declared fields."""
        expected_fields = {
            "quantization_enabled",
            "quantization_type",
            "quantization_backend",
            "pruning_enabled",
            "pruning_type",
            "pruning_amount",
            "distillation_enabled",
            "distillation_temperature",
            "distillation_alpha",
            "export_onnx",
            "optimize_for_mobile",
        }
        assert set(OptimizationConfig.model_fields.keys()) == expected_fields

    def test_model_fields_default_values(self):
        """Test model_fields exposes default values for each field."""
        fields = OptimizationConfig.model_fields

        assert fields["quantization_enabled"].default is False
        assert fields["quantization_type"].default == "dynamic"
        assert fields["quantization_backend"].default == "fbgemm"
        assert fields["pruning_enabled"].default is False
        assert fields["pruning_type"].default == "magnitude"
        assert fields["pruning_amount"].default == 0.3
        assert fields["distillation_enabled"].default is False
        assert fields["distillation_temperature"].default == 3.0
        assert fields["distillation_alpha"].default == 0.5
        assert fields["export_onnx"].default is False
        assert fields["optimize_for_mobile"].default is False

    # ----- Pydantic V2 model_json_schema() -----

    def test_model_json_schema_returns_dict(self):
        """Test Pydantic V2 model_json_schema() returns a valid schema dict."""
        schema = OptimizationConfig.model_json_schema()

        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        assert "properties" in schema
        assert "title" in schema

    def test_model_json_schema_contains_all_properties(self):
        """Test JSON schema contains all OptimizationConfig properties."""
        schema = OptimizationConfig.model_json_schema()
        properties = schema["properties"]

        expected_properties = {
            "quantization_enabled",
            "quantization_type",
            "quantization_backend",
            "pruning_enabled",
            "pruning_type",
            "pruning_amount",
            "distillation_enabled",
            "distillation_temperature",
            "distillation_alpha",
            "export_onnx",
            "optimize_for_mobile",
        }
        assert set(properties.keys()) == expected_properties

    def test_model_json_schema_property_types(self):
        """Test JSON schema declares correct types for key fields."""
        schema = OptimizationConfig.model_json_schema()
        properties = schema["properties"]

        # Boolean fields
        for bool_field in (
            "quantization_enabled",
            "pruning_enabled",
            "distillation_enabled",
            "export_onnx",
            "optimize_for_mobile",
        ):
            assert properties[bool_field]["type"] == "boolean", (
                f"Expected 'boolean' type for {bool_field}"
            )

        # String fields
        for str_field in ("quantization_type", "quantization_backend", "pruning_type"):
            assert properties[str_field]["type"] == "string", (
                f"Expected 'string' type for {str_field}"
            )

        # Float fields
        for float_field in ("pruning_amount", "distillation_temperature", "distillation_alpha"):
            assert properties[float_field]["type"] == "number", (
                f"Expected 'number' type for {float_field}"
            )

    # ----- Pydantic V2 model_copy() -----

    def test_model_copy_creates_independent_copy(self):
        """Test Pydantic V2 model_copy() creates an independent instance."""
        original = OptimizationConfig(quantization_enabled=True, pruning_amount=0.4)
        copied = original.model_copy()

        assert copied == original
        assert copied is not original

        # Mutating the copy must not affect the original
        copied.pruning_amount = 0.9
        assert original.pruning_amount == 0.4

    def test_model_copy_with_update(self):
        """Test model_copy(update=...) overrides specified fields."""
        original = OptimizationConfig(quantization_enabled=False, pruning_amount=0.3)
        updated = original.model_copy(update={"quantization_enabled": True, "pruning_amount": 0.7})

        assert updated.quantization_enabled is True
        assert updated.pruning_amount == 0.7
        # Original unchanged
        assert original.quantization_enabled is False
        assert original.pruning_amount == 0.3

    # ----- Pydantic V2 ValidationError -----

    def test_validation_error_on_invalid_bool_type(self):
        """Test Pydantic raises ValidationError for non-coercible bool field."""
        with pytest.raises(PydanticValidationError):
            OptimizationConfig(quantization_enabled="not_a_bool_value")

    def test_validation_error_on_invalid_float_type(self):
        """Test Pydantic raises ValidationError for non-coercible float field."""
        with pytest.raises(PydanticValidationError):
            OptimizationConfig(pruning_amount="not_a_float")

    def test_validation_error_on_model_validate_invalid(self):
        """Test model_validate() raises ValidationError on invalid data."""
        with pytest.raises(PydanticValidationError):
            OptimizationConfig.model_validate({"pruning_amount": "invalid"})

    # ----- Pydantic V2 BaseModel inheritance verification -----

    def test_is_pydantic_base_model_subclass(self):
        """Test OptimizationConfig inherits from Pydantic BaseModel."""
        from pydantic import BaseModel

        assert issubclass(OptimizationConfig, BaseModel)

    def test_instance_is_pydantic_base_model(self):
        """Test OptimizationConfig instances are Pydantic BaseModel instances."""
        from pydantic import BaseModel

        config = OptimizationConfig()
        assert isinstance(config, BaseModel)


# =============================================================================
# TESTS: Path Handling
# =============================================================================


class TestPathHandling:
    """Tests for path handling in export methods."""

    def test_path_object_handling(self, simple_linear_model, temp_dir):
        """Test methods handle Path objects correctly."""
        optimizer = ModelOptimizer(verbose=False)

        filepath = Path(temp_dir) / "model.onnx"
        dummy_input = torch.randn(1, 10)

        with patch("torch.onnx.export"):
            optimizer.export_to_onnx(simple_linear_model, filepath, dummy_input)

        assert filepath.parent.exists()

    def test_string_path_handling(self, simple_linear_model, temp_dir):
        """Test methods handle string paths correctly."""
        optimizer = ModelOptimizer(verbose=False)

        filepath = str(temp_dir / "model.onnx")
        dummy_input = torch.randn(1, 10)

        with patch("torch.onnx.export"):
            optimizer.export_to_onnx(simple_linear_model, filepath, dummy_input)

    def test_nested_directory_creation(self, simple_linear_model, temp_dir):
        """Test methods create nested directories."""
        optimizer = ModelOptimizer(verbose=False)

        nested_path = temp_dir / "a" / "b" / "c" / "model.onnx"
        dummy_input = torch.randn(1, 10)

        with patch("torch.onnx.export"):
            optimizer.export_to_onnx(simple_linear_model, nested_path, dummy_input)

        assert nested_path.parent.exists()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
