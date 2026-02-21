"""
Model Optimization

Comprehensive model optimization strategies for production deployment.
Supports quantization, pruning, knowledge distillation, and model export.

Features:
- Dynamic and static quantization
- Post-training quantization (PTQ)
- Quantization-aware training (QAT)
- Structured and unstructured pruning
- Magnitude-based and gradient-based pruning
- Knowledge distillation (teacher-student)
- ONNX export for cross-platform deployment
- Model compression metrics
- Optimization profiling

Pydantic V2 Migration (Phase 22):
    - Migrated OptimizationConfig from @dataclass to Pydantic BaseModel (mutable)
    - Uses model_dump() for OptimizationConfig.to_dict() (backward compatible)
    - NON-BREAKING: Same constructor API and attribute access
    - Follows established pattern from monitoring.py (Phase 21)

Author: milia Team
Version: 1.1.0
"""

import contextlib
import copy
import logging
import warnings
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from torch.nn.utils import prune

# Import exceptions with fallback
try:
    from milia_pipeline.exceptions import ExportError, ModelError, OptimizationError
except ImportError:

    class ModelError(Exception):
        """Base exception for model-related errors."""

        pass

    class OptimizationError(ModelError):
        """Exception raised for optimization-related errors."""

        pass

    class ExportError(ModelError):
        """Exception raised for model export errors."""

        pass


logger = logging.getLogger(__name__)


# =============================================================================
# OPTIMIZATION TYPES
# =============================================================================


class QuantizationType(Enum):
    """Quantization types."""

    DYNAMIC = "dynamic"  # Dynamic quantization (weights + activations)
    STATIC = "static"  # Static quantization (calibration required)
    QAT = "qat"  # Quantization-aware training
    FP16 = "fp16"  # Half precision


class PruningType(Enum):
    """Pruning types."""

    UNSTRUCTURED = "unstructured"  # Remove individual weights
    STRUCTURED = "structured"  # Remove entire channels/neurons
    MAGNITUDE = "magnitude"  # Based on weight magnitude
    GRADIENT = "gradient"  # Based on gradient information


# =============================================================================
# OPTIMIZATION CONFIGURATION
# =============================================================================


class OptimizationConfig(BaseModel):
    """
    Configuration for model optimization.

    Pydantic V2 Migration (Phase 22):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses model_dump() for to_dict() method (backward compatible)
        - NON-BREAKING: Same constructor API and attribute access

    Attributes:
        quantization_enabled: Enable quantization
        quantization_type: Type of quantization
        quantization_backend: Backend for quantization (fbgemm, qnnpack)
        pruning_enabled: Enable pruning
        pruning_type: Type of pruning
        pruning_amount: Amount to prune (0.0-1.0)
        distillation_enabled: Enable knowledge distillation
        distillation_temperature: Temperature for distillation
        distillation_alpha: Weight for distillation loss
        export_onnx: Enable ONNX export
        optimize_for_mobile: Optimize for mobile deployment
    """

    quantization_enabled: bool = False
    quantization_type: str = "dynamic"
    quantization_backend: str = "fbgemm"
    pruning_enabled: bool = False
    pruning_type: str = "magnitude"
    pruning_amount: float = 0.3
    distillation_enabled: bool = False
    distillation_temperature: float = 3.0
    distillation_alpha: float = 0.5
    export_onnx: bool = False
    optimize_for_mobile: bool = False

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Backward compatible method wrapping Pydantic V2's model_dump().
        """
        return self.model_dump()


# =============================================================================
# MODEL OPTIMIZER
# =============================================================================


class ModelOptimizer:
    """
    Manager for model optimization strategies.

    Provides comprehensive model optimization including quantization,
    pruning, distillation, and export.

    Usage:
        >>> # Quantization
        >>> optimizer = ModelOptimizer(quantization_enabled=True)
        >>> quantized_model = optimizer.quantize_model(model)
        >>>
        >>> # Pruning
        >>> optimizer = ModelOptimizer(
        ...     pruning_enabled=True,
        ...     pruning_amount=0.3
        ... )
        >>> pruned_model = optimizer.prune_model(model)
        >>>
        >>> # Knowledge distillation
        >>> optimizer = ModelOptimizer(distillation_enabled=True)
        >>> loss = optimizer.distillation_loss(
        ...     student_output, teacher_output, targets
        ... )
        >>>
        >>> # ONNX export
        >>> optimizer.export_to_onnx(model, "model.onnx", dummy_input)
    """

    def __init__(
        self,
        quantization_enabled: bool = False,
        quantization_type: str = "dynamic",
        quantization_backend: str = "fbgemm",
        pruning_enabled: bool = False,
        pruning_type: str = "magnitude",
        pruning_amount: float = 0.3,
        distillation_enabled: bool = False,
        distillation_temperature: float = 3.0,
        distillation_alpha: float = 0.5,
        export_onnx: bool = False,
        optimize_for_mobile: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize model optimizer.

        Args:
            quantization_enabled: Enable quantization
            quantization_type: Quantization type (dynamic, static, qat, fp16)
            quantization_backend: Backend (fbgemm for x86, qnnpack for ARM)
            pruning_enabled: Enable pruning
            pruning_type: Pruning type (unstructured, structured, magnitude)
            pruning_amount: Fraction of weights to prune (0.0-1.0)
            distillation_enabled: Enable knowledge distillation
            distillation_temperature: Softmax temperature for distillation
            distillation_alpha: Weight for distillation loss
            export_onnx: Enable ONNX export
            optimize_for_mobile: Optimize for mobile deployment
            verbose: Whether to log information
        """
        self.verbose = verbose

        # Create config
        self.config = OptimizationConfig(
            quantization_enabled=quantization_enabled,
            quantization_type=quantization_type,
            quantization_backend=quantization_backend,
            pruning_enabled=pruning_enabled,
            pruning_type=pruning_type,
            pruning_amount=pruning_amount,
            distillation_enabled=distillation_enabled,
            distillation_temperature=distillation_temperature,
            distillation_alpha=distillation_alpha,
            export_onnx=export_onnx,
            optimize_for_mobile=optimize_for_mobile,
        )

        # Set quantization backend
        if quantization_enabled:
            torch.backends.quantized.engine = quantization_backend

        if self.verbose:
            logger.info(
                f"ModelOptimizer initialized - "
                f"Quantization: {quantization_enabled}, "
                f"Pruning: {pruning_enabled}, "
                f"Distillation: {distillation_enabled}"
            )

    # =========================================================================
    # QUANTIZATION
    # =========================================================================

    def quantize_model(
        self,
        model: nn.Module,
        example_inputs: torch.Tensor | None = None,
        calibration_data: list[torch.Tensor] | None = None,
    ) -> nn.Module:
        """
        Quantize model for faster inference.

        Args:
            model: Model to quantize
            example_inputs: Example inputs for tracing
            calibration_data: Calibration data for static quantization

        Returns:
            Quantized model

        Raises:
            OptimizationError: If quantization fails
        """
        if not self.config.quantization_enabled:
            return model

        quant_type = self.config.quantization_type

        try:
            if quant_type == "dynamic":
                quantized = self._dynamic_quantization(model)
            elif quant_type == "static":
                if calibration_data is None:
                    raise OptimizationError("Static quantization requires calibration_data")
                quantized = self._static_quantization(model, calibration_data, example_inputs)
            elif quant_type == "qat":
                # QAT requires training, return prepared model
                quantized = self._prepare_qat(model)
                if self.verbose:
                    logger.info("Model prepared for QAT. Train the model, then call finalize_qat()")
            elif quant_type == "fp16":
                quantized = model.half()
                if self.verbose:
                    logger.info("Converted model to FP16")
            else:
                raise OptimizationError(f"Unknown quantization type: {quant_type}")

            return quantized

        except Exception as e:
            raise OptimizationError(f"Quantization failed: {e}") from e

    def _dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization."""
        # Quantize linear and LSTM layers
        quantized = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=torch.qint8
        )

        if self.verbose:
            logger.info("Applied dynamic quantization")

        return quantized

    def _static_quantization(
        self,
        model: nn.Module,
        calibration_data: list[torch.Tensor],
        example_inputs: torch.Tensor | None = None,
    ) -> nn.Module:
        """Apply static quantization with calibration."""
        # Prepare model for static quantization
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig(self.config.quantization_backend)

        # Fuse modules (Conv+BN+ReLU, etc.)
        model_fused = torch.quantization.fuse_modules(
            model,
            [["conv", "bn", "relu"]],  # Common fusion pattern
        )

        # Prepare for calibration
        model_prepared = torch.quantization.prepare(model_fused)

        # Calibrate with sample data
        with torch.no_grad():
            for data in calibration_data:
                model_prepared(data)

        # Convert to quantized model
        quantized = torch.quantization.convert(model_prepared)

        if self.verbose:
            logger.info("Applied static quantization with calibration")

        return quantized

    def _prepare_qat(self, model: nn.Module) -> nn.Module:
        """Prepare model for quantization-aware training."""
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig(self.config.quantization_backend)

        # Fuse modules
        model_fused = torch.quantization.fuse_modules(model, [["conv", "bn", "relu"]])

        # Prepare for QAT
        model_prepared = torch.quantization.prepare_qat(model_fused)

        if self.verbose:
            logger.info("Prepared model for QAT")

        return model_prepared

    def finalize_qat(self, model: nn.Module) -> nn.Module:
        """
        Finalize quantization-aware training.

        Call this after training a QAT-prepared model.

        Args:
            model: QAT-trained model

        Returns:
            Quantized model
        """
        model.eval()
        quantized = torch.quantization.convert(model)

        if self.verbose:
            logger.info("Finalized QAT - model quantized")

        return quantized

    # =========================================================================
    # PRUNING
    # =========================================================================

    def prune_model(
        self, model: nn.Module, amount: float | None = None, iterative_steps: int = 1
    ) -> nn.Module:
        """
        Prune model to reduce size and computation.

        Args:
            model: Model to prune
            amount: Fraction to prune (uses config if None)
            iterative_steps: Number of iterative pruning steps

        Returns:
            Pruned model
        """
        if not self.config.pruning_enabled:
            return model

        amount = amount or self.config.pruning_amount
        prune_type = self.config.pruning_type

        # Iterative pruning
        for step in range(iterative_steps):
            step_amount = amount / iterative_steps

            if prune_type == "magnitude":
                model = self._magnitude_pruning(model, step_amount)
            elif prune_type == "unstructured":
                model = self._unstructured_pruning(model, step_amount)
            elif prune_type == "structured":
                model = self._structured_pruning(model, step_amount)
            else:
                raise OptimizationError(f"Unknown pruning type: {prune_type}")

            if self.verbose:
                logger.info(f"Pruning step {step + 1}/{iterative_steps} complete")

        # Make pruning permanent
        model = self._remove_pruning_reparameterization(model)

        if self.verbose:
            logger.info(f"Pruned {amount * 100:.1f}% of weights")

        return model

    def _magnitude_pruning(self, model: nn.Module, amount: float) -> nn.Module:
        """Apply magnitude-based pruning."""
        for _name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.l1_unstructured(module, name="weight", amount=amount)

        return model

    def _unstructured_pruning(self, model: nn.Module, amount: float) -> nn.Module:
        """Apply unstructured pruning."""
        parameters_to_prune = []

        for _name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, "weight"))

        prune.global_unstructured(
            parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount
        )

        return model

    def _structured_pruning(self, model: nn.Module, amount: float) -> nn.Module:
        """Apply structured pruning (remove entire channels)."""
        for _name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Prune output channels
                prune.ln_structured(
                    module,
                    name="weight",
                    amount=amount,
                    n=2,
                    dim=0,  # Output channels
                )

        return model

    def _remove_pruning_reparameterization(self, model: nn.Module) -> nn.Module:
        """Make pruning permanent by removing reparameterization."""
        for _name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                with contextlib.suppress(ValueError):  # No pruning on this module
                    prune.remove(module, "weight")

        return model

    def get_sparsity(self, model: nn.Module) -> dict[str, float]:
        """
        Calculate model sparsity.

        Args:
            model: Model to analyze

        Returns:
            Dictionary with sparsity statistics
        """
        total_params = 0
        zero_params = 0

        for name, param in model.named_parameters():
            if "weight" in name:
                total_params += param.numel()
                zero_params += (param == 0).sum().item()

        global_sparsity = zero_params / total_params if total_params > 0 else 0

        return {
            "total_parameters": total_params,
            "zero_parameters": zero_params,
            "global_sparsity": global_sparsity,
            "compression_ratio": 1.0 / (1.0 - global_sparsity)
            if global_sparsity < 1.0
            else float("inf"),
        }

    # =========================================================================
    # KNOWLEDGE DISTILLATION
    # =========================================================================

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        student_loss_fn: nn.Module | None = None,
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss.

        Args:
            student_logits: Student model output logits
            teacher_logits: Teacher model output logits
            targets: Ground truth targets
            student_loss_fn: Loss function for student (default: CrossEntropy)

        Returns:
            Combined distillation loss
        """
        if not self.config.distillation_enabled:
            # Return standard loss if distillation disabled
            loss_fn = student_loss_fn or nn.CrossEntropyLoss()
            return loss_fn(student_logits, targets)

        temperature = self.config.distillation_temperature
        alpha = self.config.distillation_alpha

        # Distillation loss (KL divergence between teacher and student)
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature**2)

        # Student loss (standard cross-entropy)
        loss_fn = student_loss_fn or nn.CrossEntropyLoss()
        student_loss = loss_fn(student_logits, targets)

        # Combined loss
        total_loss = alpha * distillation_loss + (1 - alpha) * student_loss

        return total_loss

    def create_student_model(
        self, teacher_model: nn.Module, reduction_factor: float = 0.5
    ) -> nn.Module:
        """
        Create a student model from teacher (simplified architecture).

        Args:
            teacher_model: Teacher model
            reduction_factor: Factor to reduce model size

        Returns:
            Student model (placeholder - needs custom implementation)
        """
        # This is a placeholder - actual implementation depends on model architecture
        # Users should implement their own student model creation
        warnings.warn(
            "create_student_model is a placeholder. "
            "Implement custom student model based on your architecture.", stacklevel=2
        )

        # Return a copy for now
        return copy.deepcopy(teacher_model)

    # =========================================================================
    # MODEL EXPORT
    # =========================================================================

    def export_to_onnx(
        self,
        model: nn.Module,
        filepath: str | Path,
        dummy_input: torch.Tensor,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        opset_version: int = 14,
    ):
        """
        Export model to ONNX format.

        Args:
            model: Model to export
            filepath: Output file path
            dummy_input: Example input tensor
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dynamic axes for variable-size inputs
            opset_version: ONNX opset version

        Raises:
            ExportError: If export fails
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        input_names = input_names or ["input"]
        output_names = output_names or ["output"]

        try:
            model.eval()

            torch.onnx.export(
                model,
                dummy_input,
                str(filepath),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                export_params=True,
            )

            if self.verbose:
                logger.info(f"Exported model to ONNX: {filepath}")

        except Exception as e:
            raise ExportError(f"ONNX export failed: {e}") from e

    def optimize_for_mobile(self, model: nn.Module, example_inputs: tuple, filepath: str | Path):
        """
        Optimize model for mobile deployment.

        Args:
            model: Model to optimize
            example_inputs: Example inputs for tracing
            filepath: Output file path
        """
        if not self.config.optimize_for_mobile:
            return

        try:
            # Trace model
            traced = torch.jit.trace(model, example_inputs)

            # Optimize for mobile
            from torch.utils.mobile_optimizer import optimize_for_mobile

            optimized = optimize_for_mobile(traced)

            # Save
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            optimized._save_for_lite_interpreter(str(filepath))

            if self.verbose:
                logger.info(f"Optimized model for mobile: {filepath}")

        except ImportError:
            raise ExportError("Mobile optimization requires PyTorch with mobile support") from None
        except Exception as e:
            raise ExportError(f"Mobile optimization failed: {e}") from e

    # =========================================================================
    # OPTIMIZATION METRICS
    # =========================================================================

    def get_model_size(self, model: nn.Module) -> dict[str, float]:
        """
        Get model size in MB.

        Args:
            model: Model to analyze

        Returns:
            Dictionary with size metrics
        """
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = param_size + buffer_size

        return {
            "parameters_mb": param_size / (1024**2),
            "buffers_mb": buffer_size / (1024**2),
            "total_mb": total_size / (1024**2),
            "num_parameters": sum(p.numel() for p in model.parameters()),
        }

    def compare_models(
        self, original_model: nn.Module, optimized_model: nn.Module
    ) -> dict[str, Any]:
        """
        Compare original and optimized models.

        Args:
            original_model: Original model
            optimized_model: Optimized model

        Returns:
            Dictionary with comparison metrics
        """
        orig_size = self.get_model_size(original_model)
        opt_size = self.get_model_size(optimized_model)

        size_reduction = 1.0 - (opt_size["total_mb"] / orig_size["total_mb"])

        comparison = {
            "original_size_mb": orig_size["total_mb"],
            "optimized_size_mb": opt_size["total_mb"],
            "size_reduction": size_reduction,
            "compression_ratio": orig_size["total_mb"] / opt_size["total_mb"],
            "original_params": orig_size["num_parameters"],
            "optimized_params": opt_size["num_parameters"],
        }

        if self.verbose:
            logger.info(
                f"Model comparison - "
                f"Size reduction: {size_reduction * 100:.1f}%, "
                f"Compression: {comparison['compression_ratio']:.2f}x"
            )

        return comparison

    def print_optimization_summary(self):
        """Print formatted optimization summary."""
        print("=" * 70)
        print("Model Optimization Summary")
        print("=" * 70)
        print(f"Quantization: {self.config.quantization_enabled}")
        if self.config.quantization_enabled:
            print(f"  Type: {self.config.quantization_type}")
            print(f"  Backend: {self.config.quantization_backend}")

        print(f"Pruning: {self.config.pruning_enabled}")
        if self.config.pruning_enabled:
            print(f"  Type: {self.config.pruning_type}")
            print(f"  Amount: {self.config.pruning_amount * 100:.1f}%")

        print(f"Knowledge Distillation: {self.config.distillation_enabled}")
        if self.config.distillation_enabled:
            print(f"  Temperature: {self.config.distillation_temperature}")
            print(f"  Alpha: {self.config.distillation_alpha}")

        print(f"ONNX Export: {self.config.export_onnx}")
        print(f"Mobile Optimization: {self.config.optimize_for_mobile}")
        print("=" * 70)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def quantize_for_inference(model: nn.Module, quantization_type: str = "dynamic") -> nn.Module:
    """
    Quick quantization for inference.

    Args:
        model: Model to quantize
        quantization_type: Quantization type

    Returns:
        Quantized model
    """
    optimizer = ModelOptimizer(
        quantization_enabled=True, quantization_type=quantization_type, verbose=False
    )
    return optimizer.quantize_model(model)


def prune_for_deployment(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """
    Quick pruning for deployment.

    Args:
        model: Model to prune
        amount: Pruning amount

    Returns:
        Pruned model
    """
    optimizer = ModelOptimizer(pruning_enabled=True, pruning_amount=amount, verbose=False)
    return optimizer.prune_model(model)


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info("model_optimization module loaded")
