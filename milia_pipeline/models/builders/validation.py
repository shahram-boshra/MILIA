"""
Architecture Validation

Comprehensive validation for custom architectures.
Checks channel compatibility, task requirements, data compatibility.

Author: Milia Team
Version: 1.0.0
"""

import logging
from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .architecture_builder import LayerConfig

# Import layer registry and builder
from .layer_registry import LayerCategory, LayerRegistry
from .layer_registry import registry as layer_registry

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATOR CLASS
# =============================================================================


class ArchitectureValidator:
    """
    Validator for custom GNN architectures.

    Performs comprehensive validation:
    - Channel flow validation
    - Task compatibility checking
    - Data requirement validation
    - Layer ordering validation
    - Output dimension validation

    Usage:
        >>> validator = ArchitectureValidator()
        >>> result = validator.validate(layers, task_type, in_channels, out_channels)
        >>> if not result['valid']:
        ...     print(result['errors'])
    """

    def __init__(self, registry: LayerRegistry | None = None):
        """
        Initialize validator.

        Args:
            registry: Layer registry (uses global if None)
        """
        self.registry = registry or layer_registry

    # =========================================================================
    # MAIN VALIDATION
    # =========================================================================

    def validate(
        self, layers: list[LayerConfig], task_type: str, in_channels: int, out_channels: int
    ) -> dict[str, Any]:
        """
        Validate architecture.

        Args:
            layers: List of layer configurations
            task_type: Task type
            in_channels: Input channels
            out_channels: Expected output channels

        Returns:
            Validation result dictionary with:
            - valid: bool
            - errors: List[str]
            - warnings: List[str]
            - suggestions: List[str]
        """
        errors = []
        warnings = []
        suggestions = []

        # Check if empty
        if not layers:
            errors.append("Architecture has no layers")
            suggestions.append("Add at least one layer")
            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "suggestions": suggestions,
            }

        # Validate channel flow
        channel_result = self.validate_channel_flow(layers, in_channels, out_channels)
        errors.extend(channel_result["errors"])
        warnings.extend(channel_result["warnings"])
        suggestions.extend(channel_result["suggestions"])

        # Validate task compatibility
        task_result = self.validate_task_compatibility(layers, task_type)
        errors.extend(task_result["errors"])
        warnings.extend(task_result["warnings"])
        suggestions.extend(task_result["suggestions"])

        # Validate layer ordering
        order_result = self.validate_layer_ordering(layers)
        errors.extend(order_result["errors"])
        warnings.extend(order_result["warnings"])
        suggestions.extend(order_result["suggestions"])

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
        }

    # =========================================================================
    # SPECIFIC VALIDATIONS
    # =========================================================================

    def validate_channel_flow(
        self, layers: list[LayerConfig], in_channels: int, out_channels: int
    ) -> dict[str, Any]:
        """
        Validate channel dimensions flow correctly through layers.

        Args:
            layers: List of layer configurations
            in_channels: Input channels
            out_channels: Expected output channels

        Returns:
            Validation result
        """
        errors = []
        warnings = []
        suggestions = []

        current_channels = in_channels

        for i, layer_config in enumerate(layers):
            metadata = self.registry.get_layer_metadata(layer_config.type)

            # Check input channels
            if metadata.has_in_channels:
                expected_in = current_channels
                if layer_config.in_channels and layer_config.in_channels != expected_in:
                    errors.append(
                        f"Layer {i} ({layer_config.type}): Expected in_channels={expected_in}, "
                        f"got {layer_config.in_channels}"
                    )
                    suggestions.append(
                        f"Remove in_channels parameter from layer {i}, "
                        f"it will be inferred automatically"
                    )

            # Update current channels
            if metadata.has_out_channels:
                if layer_config.out_channels:
                    out_ch = layer_config.out_channels
                elif "out_channels" in layer_config.params:
                    out_ch = layer_config.params["out_channels"]
                elif "out_features" in layer_config.params:
                    out_ch = layer_config.params["out_features"]
                else:
                    warnings.append(
                        f"Layer {i} ({layer_config.type}): out_channels not specified, "
                        f"using {current_channels}"
                    )
                    out_ch = current_channels

                # Handle multi-head attention
                if "heads" in layer_config.params:
                    heads = layer_config.params["heads"]
                    concat = layer_config.params.get("concat", True)
                    if concat:
                        current_channels = out_ch * heads
                    else:
                        current_channels = out_ch
                else:
                    current_channels = out_ch
            # else: layer doesn't change channels

        # Check final output
        if current_channels != out_channels:
            warnings.append(
                f"Final output channels ({current_channels}) != expected ({out_channels})"
            )
            suggestions.append(
                f"Add Linear layer at end: .add_layer('Linear', out_features={out_channels})"
            )

        return {"errors": errors, "warnings": warnings, "suggestions": suggestions}

    def validate_task_compatibility(
        self, layers: list[LayerConfig], task_type: str
    ) -> dict[str, Any]:
        """
        Validate architecture is compatible with task type.

        Args:
            layers: List of layer configurations
            task_type: Task type (e.g., "graph_regression")

        Returns:
            Validation result
        """
        errors = []
        warnings = []
        suggestions = []

        # Check graph-level task requirements
        if "graph" in task_type.lower():
            # Must have pooling layer
            has_pooling = any(
                self.registry.get_layer_metadata(l.type).category == LayerCategory.POOLING
                for l in layers
            )

            if not has_pooling:
                errors.append(f"Task '{task_type}' requires pooling layer for graph-level output")
                suggestions.append(
                    "Add pooling layer before final layer: .insert_layer(-1, 'global_mean_pool')"
                )

        # Check node-level task requirements
        elif "node" in task_type.lower():
            # Should not have pooling
            pooling_layers = [
                (i, l.type)
                for i, l in enumerate(layers)
                if self.registry.get_layer_metadata(l.type).category == LayerCategory.POOLING
            ]

            if pooling_layers:
                warnings.append(
                    f"Task '{task_type}' is node-level, but architecture has pooling layers: "
                    f"{[name for _, name in pooling_layers]}"
                )
                suggestions.append("Remove pooling layers for node-level tasks")

        return {"errors": errors, "warnings": warnings, "suggestions": suggestions}

    def validate_layer_ordering(self, layers: list[LayerConfig]) -> dict[str, Any]:
        """
        Validate layer ordering is sensible.

        Args:
            layers: List of layer configurations

        Returns:
            Validation result
        """
        errors = []
        warnings = []
        suggestions = []

        for i in range(len(layers) - 1):
            current = layers[i]
            next_layer = layers[i + 1]

            current_meta = self.registry.get_layer_metadata(current.type)
            next_meta = self.registry.get_layer_metadata(next_layer.type)

            # Pooling should come after convolutional layers
            if current_meta.category == LayerCategory.POOLING:
                if next_meta.category == LayerCategory.CONVOLUTIONAL:
                    warnings.append(
                        f"Unusual ordering: Pooling layer {i} ({current.type}) "
                        f"followed by convolutional layer {i + 1} ({next_layer.type})"
                    )
                    suggestions.append("Consider moving pooling layer later in architecture")

            # Activation should come after convolutional/linear layers
            if current_meta.category == LayerCategory.ACTIVATION:
                if next_meta.category == LayerCategory.ACTIVATION:
                    warnings.append(
                        f"Multiple consecutive activation layers: "
                        f"{i} ({current.type}), {i + 1} ({next_layer.type})"
                    )

        return {"errors": errors, "warnings": warnings, "suggestions": suggestions}

    def validate_data_compatibility(
        self, architecture: nn.Module, sample_data: Data
    ) -> dict[str, Any]:
        """
        Validate architecture is compatible with data.

        Args:
            architecture: Built architecture
            sample_data: Sample data

        Returns:
            Validation result
        """
        errors = []
        warnings = []
        suggestions = []

        # Check required data fields
        if not hasattr(sample_data, "x") or sample_data.x is None:
            errors.append("Sample data missing node features (x)")

        if not hasattr(sample_data, "edge_index") or sample_data.edge_index is None:
            warnings.append("Sample data missing edge_index")

        # Test forward pass
        try:
            with torch.no_grad():
                if hasattr(sample_data, "batch"):
                    out = architecture(
                        sample_data.x,
                        sample_data.edge_index,
                        getattr(sample_data, "edge_attr", None),
                        sample_data.batch,
                    )
                else:
                    out = architecture(
                        sample_data.x,
                        sample_data.edge_index,
                        getattr(sample_data, "edge_attr", None),
                    )
        except Exception as e:
            errors.append(f"Forward pass failed: {e}")
            suggestions.append("Check layer parameters and data compatibility")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
        }

    # =========================================================================
    # SUGGESTION GENERATION
    # =========================================================================

    def suggest_fixes(self, validation_result: dict[str, Any]) -> list[str]:
        """
        Generate actionable fix suggestions.

        Args:
            validation_result: Validation result from validate()

        Returns:
            List of fix suggestions
        """
        fixes = []

        # Already included in validation_result['suggestions']
        if validation_result.get("suggestions"):
            fixes.extend(validation_result["suggestions"])

        # Generate additional fixes based on error patterns
        for error in validation_result.get("errors", []):
            if "channel mismatch" in error.lower():
                fixes.append("Run builder._infer_channels() before building")
            elif "pooling" in error.lower():
                fixes.append("Add appropriate pooling layer for task type")

        return fixes


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def validate_architecture(
    layers: list[LayerConfig], task_type: str, in_channels: int, out_channels: int
) -> dict[str, Any]:
    """
    Convenience function to validate architecture.

    Args:
        layers: List of layer configurations
        task_type: Task type
        in_channels: Input channels
        out_channels: Output channels

    Returns:
        Validation result
    """
    validator = ArchitectureValidator()
    return validator.validate(layers, task_type, in_channels, out_channels)


def validate_data_compatibility(architecture: nn.Module, sample_data: Data) -> dict[str, Any]:
    """
    Convenience function to validate data compatibility.

    Args:
        architecture: Built architecture
        sample_data: Sample data

    Returns:
        Validation result
    """
    validator = ArchitectureValidator()
    return validator.validate_data_compatibility(architecture, sample_data)


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info("validation module loaded")
