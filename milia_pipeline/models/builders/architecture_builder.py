"""
Architecture Builder

Dynamic composition and modification of GNN architectures.
Supports sequential layer composition, residual connections, and branching.
Enhanced to handle both class-based and functional layers seamlessly.

Pydantic V2 Migration (Phase 33):
    - Migrated from @dataclass to Pydantic BaseModel (mutable)
    - Uses Field(default_factory=...) for mutable defaults
    - Preserves custom to_dict() methods for nested serialization (LayerConfig, ResidualConnection)
    - Preserves custom from_dict() classmethods for nested deserialization
    - NON-BREAKING: Same constructor API and attribute access preserved

Author: milia Team
Version: 1.1.0
"""

import logging
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from torch_geometric.data import Data

# Import layer registry
from .layer_registry import (
    FunctionalLayerWrapper,
    LayerCategory,
    LayerNotFoundError,
    LayerRegistry,
)
from .layer_registry import registry as layer_registry

# Import exceptions
try:
    from milia_pipeline.exceptions import ModelError
except ImportError:

    class ModelError(Exception):
        pass


logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class LayerConfig(BaseModel):
    """
    Configuration for a single layer in architecture.

    Pydantic V2 Migration (Phase 33):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses Field(default_factory=...) for mutable defaults
        - Preserves custom to_dict() for backward compatibility
        - NON-BREAKING: Same constructor API and attribute access

    Attributes:
        type: Layer type name
        params: Layer parameters
        position: Position in architecture
        in_channels: Input channels (inferred if None)
        out_channels: Output channels (from params or inferred)
        input_from: List of layer indices to receive input from
    """

    type: str
    params: dict[str, Any]
    position: int
    in_channels: int | None = None
    out_channels: int | None = None
    input_from: list[int] = Field(default_factory=lambda: [-1])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "params": self.params,
            "position": self.position,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "input_from": self.input_from,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayerConfig":
        """Create from dictionary."""
        return cls(
            type=data["type"],
            params=data.get("params", {}),
            position=data["position"],
            in_channels=data.get("in_channels"),
            out_channels=data.get("out_channels"),
            input_from=data.get("input_from", [-1]),
        )


class ResidualConnection(BaseModel):
    """
    Residual/skip connection specification.

    Pydantic V2 Migration (Phase 33):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Preserves custom to_dict() for backward compatibility
        - NON-BREAKING: Same constructor API and attribute access

    Attributes:
        start_layer: Index of start layer
        end_layer: Index of end layer
        connection_type: Type of connection ("add" or "concat")
    """

    start_layer: int
    end_layer: int
    connection_type: str = "add"  # "add" or "concat"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "connection_type": self.connection_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResidualConnection":
        """Create from dictionary."""
        return cls(
            start_layer=data["start_layer"],
            end_layer=data["end_layer"],
            connection_type=data.get("connection_type", "add"),
        )


class ArchitectureConfig(BaseModel):
    """
    Complete architecture configuration.

    Pydantic V2 Migration (Phase 33):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses Field(default_factory=list) for mutable defaults
        - Preserves custom to_dict() for nested serialization (calls layer.to_dict())
        - Preserves custom from_dict() for nested deserialization
        - NON-BREAKING: Same constructor API and attribute access

    Attributes:
        name: Architecture name
        task_type: Task type (node_regression, graph_classification, etc.)
        in_channels: Input feature dimension
        out_channels: Output dimension
        layers: List of layer configurations
        residual_connections: List of residual connections
    """

    name: str
    task_type: str
    in_channels: int
    out_channels: int
    layers: list[LayerConfig] = Field(default_factory=list)
    residual_connections: list[ResidualConnection] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "task_type": self.task_type,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "layers": [layer.to_dict() for layer in self.layers],
            "residual_connections": [rc.to_dict() for rc in self.residual_connections],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArchitectureConfig":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            task_type=data["task_type"],
            in_channels=data["in_channels"],
            out_channels=data["out_channels"],
            layers=[LayerConfig.from_dict(l) for l in data.get("layers", [])],
            residual_connections=[
                ResidualConnection.from_dict(rc) for rc in data.get("residual_connections", [])
            ],
        )


# =============================================================================
# EXCEPTIONS
# =============================================================================


class ArchitectureError(ModelError):
    """Base exception for architecture building errors."""

    pass


class ChannelMismatchError(ArchitectureError):
    """Exception for channel dimension mismatches."""

    def __init__(
        self,
        layer1_name: str,
        layer1_pos: int,
        out_channels: int,
        layer2_name: str,
        layer2_pos: int,
        in_channels: int,
    ):
        self.layer1_name = layer1_name
        self.layer1_pos = layer1_pos
        self.out_channels = out_channels
        self.layer2_name = layer2_name
        self.layer2_pos = layer2_pos
        self.in_channels = in_channels

        message = (
            f"Channel mismatch between layers:\n"
            f"  Layer {layer1_pos} ({layer1_name}) outputs {out_channels} channels\n"
            f"  Layer {layer2_pos} ({layer2_name}) expects {in_channels} channels\n"
            f"Suggestion: Adjust out_channels of layer {layer1_pos} or in_channels of layer {layer2_pos}"
        )
        super().__init__(message)


# =============================================================================
# ARCHITECTURE BUILDER
# =============================================================================


class ArchitectureBuilder:
    """
    Builder for custom GNN architectures.

    Supports:
    - Sequential layer composition
    - Dynamic add/remove/insert/replace operations
    - Automatic channel inference
    - Residual connections
    - Architecture validation
    - Automatic handling of both class-based and functional layers

    Usage:
        >>> builder = ArchitectureBuilder("graph_regression", in_channels=16, out_channels=1)
        >>> builder.add_layer("GCNConv", out_channels=64)
        >>> builder.add_layer("ReLU")
        >>> builder.add_layer("global_mean_pool")  # Functional layer, automatically wrapped
        >>> builder.add_layer("Linear", out_features=1)
        >>> model = builder.build()
    """

    def __init__(
        self, task_type: str, in_channels: int, out_channels: int, name: str = "CustomArchitecture"
    ):
        """
        Initialize architecture builder.

        Args:
            task_type: Task type (e.g., "graph_regression", "node_classification")
            in_channels: Input feature dimension
            out_channels: Output dimension
            name: Architecture name
        """
        self.task_type = task_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.name = name

        self.layers: list[LayerConfig] = []
        self.residual_connections: list[ResidualConnection] = []

        self.registry = layer_registry

        logger.debug(
            f"Initialized ArchitectureBuilder: {task_type}, in={in_channels}, out={out_channels}"
        )

    # =========================================================================
    # LAYER MANIPULATION
    # =========================================================================

    def add_layer(self, layer_type: str, position: int = -1, **params) -> "ArchitectureBuilder":
        """
        Add a layer to the architecture.

        Args:
            layer_type: Layer type name (e.g., "GCNConv", "ReLU", "global_mean_pool")
            position: Position to insert (-1 = append to end)
            **params: Layer parameters

        Returns:
            Self for method chaining

        Example:
            >>> builder.add_layer("GCNConv", out_channels=64)
            >>> builder.add_layer("ReLU")
            >>> builder.add_layer("global_mean_pool")  # Functional layer
        """
        # Validate layer exists
        if not self.registry.has_layer(layer_type):
            raise LayerNotFoundError(layer_type, self.registry.list_layers())

        # Determine position
        if position == -1:
            position = len(self.layers)

        # Create layer config
        layer_config = LayerConfig(type=layer_type, params=params, position=position)

        # Insert layer
        if position >= len(self.layers):
            self.layers.append(layer_config)
        else:
            self.layers.insert(position, layer_config)
            # Update positions of subsequent layers
            for i in range(position + 1, len(self.layers)):
                self.layers[i].position = i

        logger.debug(f"Added layer {layer_type} at position {position}")
        return self

    def remove_layer(self, position: int) -> "ArchitectureBuilder":
        """
        Remove a layer from the architecture.

        Args:
            position: Position of layer to remove

        Returns:
            Self for method chaining
        """
        if position < 0 or position >= len(self.layers):
            raise ValueError(f"Invalid position: {position}")

        removed = self.layers.pop(position)

        # Update positions
        for i in range(position, len(self.layers)):
            self.layers[i].position = i

        logger.debug(f"Removed layer {removed.type} from position {position}")
        return self

    def insert_layer(self, position: int, layer_type: str, **params) -> "ArchitectureBuilder":
        """
        Insert a layer at specific position.

        Args:
            position: Position to insert
            layer_type: Layer type name
            **params: Layer parameters

        Returns:
            Self for method chaining
        """
        return self.add_layer(layer_type, position=position, **params)

    def replace_layer(self, position: int, new_layer_type: str, **params) -> "ArchitectureBuilder":
        """
        Replace a layer at specific position.

        Args:
            position: Position of layer to replace
            new_layer_type: New layer type
            **params: New layer parameters

        Returns:
            Self for method chaining
        """
        if position < 0 or position >= len(self.layers):
            raise ValueError(f"Invalid position: {position}")

        old_type = self.layers[position].type

        # Create new layer config
        new_config = LayerConfig(type=new_layer_type, params=params, position=position)

        self.layers[position] = new_config

        logger.debug(f"Replaced {old_type} with {new_layer_type} at position {position}")
        return self

    def swap_layers(self, pos1: int, pos2: int) -> "ArchitectureBuilder":
        """
        Swap two layers.

        Args:
            pos1: Position of first layer
            pos2: Position of second layer

        Returns:
            Self for method chaining
        """
        if pos1 < 0 or pos1 >= len(self.layers):
            raise ValueError(f"Invalid position: {pos1}")
        if pos2 < 0 or pos2 >= len(self.layers):
            raise ValueError(f"Invalid position: {pos2}")

        # Swap layers
        self.layers[pos1], self.layers[pos2] = self.layers[pos2], self.layers[pos1]

        # Update positions
        self.layers[pos1].position = pos1
        self.layers[pos2].position = pos2

        logger.debug(f"Swapped layers at positions {pos1} and {pos2}")
        return self

    # =========================================================================
    # RESIDUAL CONNECTIONS
    # =========================================================================

    def add_residual_connection(
        self, start: int, end: int, connection_type: str = "add"
    ) -> "ArchitectureBuilder":
        """
        Add a residual/skip connection.

        Args:
            start: Start layer index
            end: End layer index
            connection_type: "add" or "concat"

        Returns:
            Self for method chaining
        """
        if start < 0 or start >= len(self.layers):
            raise ValueError(f"Invalid start position: {start}")
        if end < 0 or end >= len(self.layers):
            raise ValueError(f"Invalid end position: {end}")
        if start >= end:
            raise ValueError("start must be < end")

        residual = ResidualConnection(
            start_layer=start, end_layer=end, connection_type=connection_type
        )
        self.residual_connections.append(residual)

        logger.debug(f"Added residual connection from {start} to {end}")
        return self

    # =========================================================================
    # CHANNEL INFERENCE
    # =========================================================================

    def _infer_channels(self):
        """Infer and set channel dimensions for all layers."""
        if not self.layers:
            return

        # Track current channel dimension
        current_channels = self.in_channels

        for _i, layer_config in enumerate(self.layers):
            metadata = self.registry.get_layer_metadata(layer_config.type)

            # Set in_channels (or in_features for Linear layers)
            if metadata.has_in_channels:
                layer_config.in_channels = current_channels

                # Linear layer uses in_features, not in_channels
                if layer_config.type == "Linear":
                    if "in_features" not in layer_config.params:
                        layer_config.params["in_features"] = current_channels
                else:
                    if "in_channels" not in layer_config.params:
                        layer_config.params["in_channels"] = current_channels

            # Get or infer out_channels
            if metadata.has_out_channels:
                if "out_channels" in layer_config.params:
                    out_ch = layer_config.params["out_channels"]
                elif "out_features" in layer_config.params:  # For Linear layers
                    out_ch = layer_config.params["out_features"]
                else:
                    # Use in_channels as default
                    out_ch = current_channels

                    # Set appropriate parameter name
                    if layer_config.type == "Linear":
                        layer_config.params["out_features"] = out_ch
                    else:
                        layer_config.params["out_channels"] = out_ch

                layer_config.out_channels = out_ch

                # Handle multi-head attention (e.g., GAT)
                if "heads" in layer_config.params:
                    heads = layer_config.params["heads"]
                    # Check if concatenating heads
                    concat = layer_config.params.get("concat", True)
                    current_channels = out_ch * heads if concat else out_ch
                else:
                    current_channels = out_ch
            else:
                # Layer doesn't change channels
                layer_config.out_channels = current_channels

        logger.debug("Channel inference complete")

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate_architecture(self) -> dict[str, Any]:
        """
        Validate architecture.

        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'errors': List[str],
                'warnings': List[str],
                'suggestions': List[str]
            }
        """
        errors = []
        warnings = []
        suggestions = []

        # Check if architecture is empty
        if not self.layers:
            errors.append("Architecture has no layers")
            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "suggestions": ["Add at least one layer to the architecture"],
            }

        # Infer channels first
        self._infer_channels()

        # Validate channel flow
        for i in range(len(self.layers) - 1):
            current = self.layers[i]
            next_layer = self.layers[i + 1]

            current_meta = self.registry.get_layer_metadata(current.type)
            next_meta = self.registry.get_layer_metadata(next_layer.type)

            if current_meta.has_out_channels and next_meta.has_in_channels:
                out_ch = current.out_channels
                in_ch = next_layer.in_channels

                if out_ch != in_ch:
                    errors.append(
                        f"Channel mismatch: Layer {i} ({current.type}) outputs "
                        f"{out_ch} channels, Layer {i + 1} ({next_layer.type}) "
                        f"expects {in_ch} channels"
                    )
                    suggestions.append(
                        f"Adjust out_channels of {current.type} to {in_ch} or "
                        f"in_channels of {next_layer.type} to {out_ch}"
                    )

        # Check task-specific requirements
        if "graph" in self.task_type:
            # Graph-level tasks need pooling
            has_pooling = any(
                self.registry.get_layer_metadata(l.type).category == LayerCategory.POOLING
                for l in self.layers
            )
            if not has_pooling:
                errors.append(
                    f"Task '{self.task_type}' requires pooling layer (e.g., global_mean_pool)"
                )
                suggestions.append(
                    "Add pooling layer before final output: "
                    "builder.insert_layer(-1, 'global_mean_pool')"
                )

        # Check final output dimension
        if self.layers:
            last_layer = self.layers[-1]
            if last_layer.out_channels != self.out_channels:
                warnings.append(
                    f"Final layer outputs {last_layer.out_channels} channels, "
                    f"but target is {self.out_channels}"
                )
                suggestions.append(
                    f"Add Linear layer: "
                    f"builder.add_layer('Linear', out_features={self.out_channels})"
                )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
        }

    # =========================================================================
    # BUILDING
    # =========================================================================

    def build(self) -> nn.Module:
        """
        Build the architecture into a nn.Module.

        Returns:
            Built model

        Raises:
            ArchitectureError: If architecture is invalid
        """
        # Validate first
        validation = self.validate_architecture()
        if not validation["valid"]:
            error_msg = "Architecture validation failed:\n"
            for error in validation["errors"]:
                error_msg += f"  - {error}\n"
            if validation["suggestions"]:
                error_msg += "\nSuggestions:\n"
                for suggestion in validation["suggestions"]:
                    error_msg += f"  - {suggestion}\n"
            raise ArchitectureError(error_msg)

        # Build model
        model = CustomArchitecture(
            layers=self.layers,
            residual_connections=self.residual_connections,
            registry=self.registry,
            name=self.name,
        )

        logger.info(f"Built architecture '{self.name}' with {len(self.layers)} layers")
        return model

    # =========================================================================
    # CONFIG IMPORT/EXPORT
    # =========================================================================

    def to_config(self) -> ArchitectureConfig:
        """Export architecture configuration."""
        return ArchitectureConfig(
            name=self.name,
            task_type=self.task_type,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            layers=deepcopy(self.layers),
            residual_connections=deepcopy(self.residual_connections),
        )

    @classmethod
    def from_config(
        cls,
        config: ArchitectureConfig | dict[str, Any],
        task_type: str | None = None,
        sample_data: Data | None = None,
    ) -> "ArchitectureBuilder":
        """
        Create builder from configuration.

        Args:
            config: ArchitectureConfig or dictionary
            task_type: Override task type
            sample_data: Sample data for inference (optional)

        Returns:
            ArchitectureBuilder instance
        """
        # Convert dict to ArchitectureConfig if needed
        if isinstance(config, dict):
            config = ArchitectureConfig.from_dict(config)

        # Create builder
        builder = cls(
            task_type=task_type or config.task_type,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            name=config.name,
        )

        # Add layers
        for layer_config in config.layers:
            builder.add_layer(layer_config.type, **layer_config.params)

        # Add residual connections
        for rc in config.residual_connections:
            builder.add_residual_connection(rc.start_layer, rc.end_layer, rc.connection_type)

        return builder

    # =========================================================================
    # UTILITY
    # =========================================================================

    def __len__(self) -> int:
        """Return number of layers."""
        return len(self.layers)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ArchitectureBuilder(name='{self.name}', "
            f"task='{self.task_type}', layers={len(self.layers)})"
        )


# =============================================================================
# CUSTOM ARCHITECTURE MODULE
# =============================================================================


class CustomArchitecture(nn.Module):
    """
    Custom architecture built from layer configurations.

    This is the actual nn.Module that gets created by ArchitectureBuilder.build().
    Handles both class-based and functional (wrapped) layers seamlessly.
    Automatically handles dimension mismatches in residual connections via projection layers.
    """

    def __init__(
        self,
        layers: list[LayerConfig],
        residual_connections: list[ResidualConnection],
        registry: LayerRegistry,
        name: str = "CustomArchitecture",
    ):
        """
        Initialize custom architecture.

        Args:
            layers: List of layer configurations
            residual_connections: List of residual connections
            registry: Layer registry
            name: Architecture name
        """
        super().__init__()
        self.name = name
        self.layer_configs = layers
        self.residual_connections = residual_connections
        self.registry = registry

        # Build layers
        self._build_layers()

        # Projection layers will be created dynamically on first forward pass
        # This is truly dynamic - no assumptions about channels
        self.projections = nn.ModuleDict()
        self._projections_initialized = False

    def _build_layers(self):
        """Build all layers from configurations."""
        self.layers_list = nn.ModuleList()

        for layer_config in self.layer_configs:
            # Get layer class
            layer_class = self.registry.get_layer(layer_config.type)

            # Get metadata for better error messages
            metadata = self.registry.get_layer_metadata(layer_config.type)

            # Instantiate layer
            try:
                layer = layer_class(**layer_config.params)
            except TypeError as e:
                # Enhanced error message
                error_msg = (
                    f"Failed to instantiate {layer_config.type} with params "
                    f"{layer_config.params}:\n"
                    f"  Error: {e}\n"
                )

                if metadata.is_functional:
                    error_msg += (
                        f"  Note: {layer_config.type} is a functional layer (wrapped automatically).\n"
                        f"  It should not require instantiation parameters.\n"
                    )
                else:
                    error_msg += "  Hint: Check the layer documentation for required parameters.\n"

                raise ArchitectureError(error_msg)

            self.layers_list.append(layer)
            logger.debug(f"Built layer {layer_config.type} at position {layer_config.position}")

    def _create_projection_if_needed(
        self, rc_idx: int, start_shape: torch.Size, end_shape: torch.Size, device: torch.device
    ) -> nn.Module | None:
        """
        Dynamically create projection layer if dimensions don't match.

        This is called during first forward pass with actual tensor shapes.
        Production-ready: handles any dimension mismatch dynamically.

        Args:
            rc_idx: Residual connection index
            start_shape: Shape of start layer output
            end_shape: Shape of end layer output
            device: Device to create projection on

        Returns:
            Projection layer or None if shapes match
        """
        projection_key = f"rc_{rc_idx}"

        # Check if already created
        if projection_key in self.projections:
            return self.projections[projection_key]

        # Check if projection needed
        if start_shape == end_shape:
            return None

        # Handle different cases
        start_channels = start_shape[-1] if len(start_shape) > 1 else start_shape[0]
        end_channels = end_shape[-1] if len(end_shape) > 1 else end_shape[0]

        # Create projection
        projection = nn.Linear(start_channels, end_channels, bias=False).to(device)

        # Register in ModuleDict for proper parameter tracking
        self.projections[projection_key] = projection

        logger.info(
            f"Dynamically created projection for residual connection {rc_idx}: "
            f"{start_channels} -> {end_channels} channels"
        )

        return projection

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through architecture.

        PRODUCTION-READY DYNAMIC SOLUTION:
        - Detects actual tensor shapes at runtime (first forward pass)
        - Creates projection layers only when needed, based on real dimensions
        - No hardcoded assumptions about layer output channels
        - Handles any custom layer or edge case automatically

        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge attributes
            batch: Batch assignment

        Returns:
            Output tensor
        """
        # Store intermediate outputs for residual connections
        # outputs[i] = output of layer i (after applying the layer)
        outputs = []

        # Forward through layers
        current = x
        for i, (layer, config) in enumerate(zip(self.layers_list, self.layer_configs, strict=False)):
            metadata = self.registry.get_layer_metadata(config.type)

            # Apply layer based on its requirements
            try:
                # Handle functional layers (pooling, aggregation, etc.) - pass all available arguments
                # These layers have their own logic to handle which arguments they need
                if metadata.is_functional or isinstance(layer, FunctionalLayerWrapper):
                    current = layer(
                        current, edge_index=edge_index, edge_attr=edge_attr, batch=batch
                    )
                # Handle layers that need edge_index (GNN convolutional layers)
                elif metadata.requires_edge_index:
                    if edge_index is None:
                        raise ValueError(f"{config.type} requires edge_index but none provided")
                    if metadata.requires_edge_attr and edge_attr is not None:
                        current = layer(current, edge_index, edge_attr)
                    else:
                        current = layer(current, edge_index)
                # Handle layers that need batch but not edge_index (rare case)
                elif metadata.requires_batch:
                    if batch is None:
                        raise ValueError(f"{config.type} requires batch but none provided")
                    current = layer(current, batch)
                # Simple layers (activations, normalization, dropout, etc.)
                else:
                    current = layer(current)
            except Exception as e:
                raise ArchitectureError(
                    f"Error in forward pass at layer {i} ({config.type}):\n"
                    f"  {str(e)}\n"
                    f"  Layer metadata: requires_edge_index={metadata.requires_edge_index}, "
                    f"requires_batch={metadata.requires_batch}, "
                    f"is_functional={metadata.is_functional}"
                )

            # Store output BEFORE applying residual connections
            # This is the "clean" output of this layer
            layer_output = current

            # Apply residual connections with DYNAMIC projection
            for rc_idx, rc in enumerate(self.residual_connections):
                if rc.end_layer == i:
                    # Get output from the start layer
                    start_output = outputs[rc.start_layer]

                    if rc.connection_type == "add":
                        # DYNAMIC PROJECTION: Create projection based on actual tensor shapes
                        if start_output.shape != current.shape:
                            # Create projection dynamically if not already exists
                            projection = self._create_projection_if_needed(
                                rc_idx=rc_idx,
                                start_shape=start_output.shape,
                                end_shape=current.shape,
                                device=current.device,
                            )

                            if projection is not None:
                                start_output = projection(start_output)

                        # Final dimension check
                        if current.shape != start_output.shape:
                            raise ArchitectureError(
                                f"Residual connection {rc_idx} (layers {rc.start_layer}->{rc.end_layer}): "
                                f"Shape mismatch - current={current.shape}, residual={start_output.shape}. "
                                f"This should not happen with dynamic projection. Please report this issue."
                            )

                        current = current + start_output

                    elif rc.connection_type == "concat":
                        # Concatenate along feature dimension (last dimension)
                        current = torch.cat([current, start_output], dim=-1)

            # Store the layer output (before residual) for use in future residual connections
            outputs.append(layer_output)

        return current

    def __repr__(self) -> str:
        """String representation."""
        layer_str = "\n".join(
            [
                f"  ({i}) {config.type}: {config.params}"
                for i, config in enumerate(self.layer_configs)
            ]
        )
        return f"{self.name}(\n{layer_str}\n)"


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info("architecture_builder module loaded")
