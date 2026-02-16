"""
Layer Registry

Thread-safe singleton registry for individual GNN layers.
Manages convolutional layers, pooling operations, normalizations, activations.
Includes automatic wrapping of functional operations into nn.Module.

Pydantic V2 Migration (Phase 35):
    - Migrated LayerMetadata from @dataclass to Pydantic BaseModel (mutable)
    - Uses Field(default_factory=...) for mutable defaults
    - Preserves custom to_dict() with enum .value serialization
    - NON-BREAKING: Same constructor API and attribute access preserved

Author: milia Team
Version: 1.1.0
"""

import logging
import threading
from collections.abc import Callable
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from pydantic import BaseModel, Field

# Import exceptions
try:
    from milia_pipeline.exceptions import ModelError
except ImportError:

    class ModelError(Exception):
        pass


logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class LayerCategory(Enum):
    """Categories of GNN layers."""

    CONVOLUTIONAL = "convolutional"
    POOLING = "pooling"
    NORMALIZATION = "normalization"
    ACTIVATION = "activation"
    AGGREGATION = "aggregation"
    LINEAR = "linear"
    DROPOUT = "dropout"
    CUSTOM = "custom"


# =============================================================================
# FUNCTIONAL LAYER WRAPPER
# =============================================================================


class FunctionalLayerWrapper(nn.Module):
    """
    Wraps functional operations (like global_mean_pool) into nn.Module.

    This allows functional operations to be treated uniformly with class-based
    layers in the architecture builder system.

    Attributes:
        func: The underlying functional operation
        func_name: Name of the function for debugging
        requires_batch: Whether function requires batch argument
        requires_edge_index: Whether function requires edge_index argument
        requires_edge_attr: Whether function requires edge_attr argument
    """

    def __init__(
        self,
        func: Callable,
        func_name: str,
        requires_batch: bool = False,
        requires_edge_index: bool = False,
        requires_edge_attr: bool = False,
    ):
        """
        Initialize functional layer wrapper.

        Args:
            func: The functional operation to wrap
            func_name: Name of the function
            requires_batch: Whether the function needs batch argument
            requires_edge_index: Whether the function needs edge_index
            requires_edge_attr: Whether the function needs edge_attr
        """
        super().__init__()
        self.func = func
        self.func_name = func_name
        self.requires_batch = requires_batch
        self.requires_edge_index = requires_edge_index
        self.requires_edge_attr = requires_edge_attr

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the functional operation.

        Args:
            x: Input tensor
            edge_index: Edge indices (optional)
            edge_attr: Edge attributes (optional)
            batch: Batch assignment (optional)

        Returns:
            Output tensor
        """
        # Build arguments based on requirements
        args = [x]

        if self.requires_edge_index:
            if edge_index is None:
                raise ValueError(f"{self.func_name} requires edge_index but none provided")
            args.append(edge_index)

            if self.requires_edge_attr and edge_attr is not None:
                args.append(edge_attr)

        if self.requires_batch:
            if batch is None:
                raise ValueError(f"{self.func_name} requires batch but none provided")
            args.append(batch)

        return self.func(*args)

    def __repr__(self) -> str:
        """String representation."""
        return f"FunctionalLayerWrapper({self.func_name})"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class LayerMetadata(BaseModel):
    """
    Metadata for a GNN layer.

    Pydantic V2 Migration (Phase 35):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses Field(default_factory=...) for mutable defaults
        - Preserves custom to_dict() with enum .value serialization
        - NON-BREAKING: Same constructor API and attribute access

    Attributes:
        name: Layer name (e.g., "GCNConv")
        category: Layer category
        class_path: Full import path
        description: Brief description
        requires_edge_index: Whether layer needs edge_index
        requires_edge_attr: Whether layer needs edge_attr
        requires_batch: Whether layer needs batch assignment
        has_in_channels: Whether layer has in_channels parameter
        has_out_channels: Whether layer has out_channels parameter
        modifies_graph_structure: Whether layer changes graph (e.g., pooling)
        supported_task_levels: Task levels supported (node, edge, graph)
        is_functional: Whether this was originally a functional operation
    """

    name: str
    category: LayerCategory
    class_path: str
    description: str
    requires_edge_index: bool = True
    requires_edge_attr: bool = False
    requires_batch: bool = False
    has_in_channels: bool = True
    has_out_channels: bool = True
    modifies_graph_structure: bool = False
    supported_task_levels: list[str] = Field(default_factory=lambda: ["node", "edge", "graph"])
    is_functional: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "category": self.category.value,
            "class_path": self.class_path,
            "description": self.description,
            "requires_edge_index": self.requires_edge_index,
            "requires_edge_attr": self.requires_edge_attr,
            "requires_batch": self.requires_batch,
            "has_in_channels": self.has_in_channels,
            "has_out_channels": self.has_out_channels,
            "modifies_graph_structure": self.modifies_graph_structure,
            "supported_task_levels": self.supported_task_levels,
            "is_functional": self.is_functional,
        }


# =============================================================================
# EXCEPTIONS
# =============================================================================


class LayerNotFoundError(ModelError):
    """Exception raised when layer is not found in registry."""

    def __init__(self, layer_name: str, available_layers: list[str] | None = None):
        self.layer_name = layer_name
        self.available_layers = available_layers or []
        message = f"Layer '{layer_name}' not found in registry."
        if available_layers:
            message += f"\nAvailable layers: {', '.join(sorted(available_layers[:10]))}"
            if len(available_layers) > 10:
                message += f" ... and {len(available_layers) - 10} more"
        super().__init__(message)


# =============================================================================
# LAYER REGISTRY (SINGLETON)
# =============================================================================


class LayerRegistry:
    """
    Thread-safe singleton registry for GNN layers.

    Manages a catalog of individual layers that can be used in
    custom architecture building. Automatically wraps functional
    operations (like global_mean_pool) into nn.Module for uniform handling.

    Usage:
        >>> registry = LayerRegistry()
        >>> layer_class = registry.get_layer("GCNConv")
        >>> layer = layer_class(in_channels=16, out_channels=64)
        >>>
        >>> # Functional layers are automatically wrapped
        >>> pool_class = registry.get_layer("global_mean_pool")
        >>> pool = pool_class()  # No arguments needed
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize registry (only once due to singleton)."""
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self._layers: dict[str, type[nn.Module]] = {}
        self._metadata: dict[str, LayerMetadata] = {}
        self._by_category: dict[LayerCategory, set[str]] = {}
        self._lock = threading.RLock()

        logger.info("LayerRegistry initialized")

        # Auto-register PyG layers
        self._register_builtin_layers()

    # =========================================================================
    # REGISTRATION
    # =========================================================================

    def _register_builtin_layers(self):
        """Register all built-in PyG layers."""
        logger.info("Registering built-in PyG layers...")

        # Convolutional layers
        self._register_convolutional_layers()

        # Pooling layers
        self._register_pooling_layers()

        # Normalization layers
        self._register_normalization_layers()

        # Activation layers
        self._register_activation_layers()

        # Aggregation layers
        self._register_aggregation_layers()

        # Linear/Dropout layers
        self._register_standard_layers()

        total = len(self._layers)
        logger.info(f"Registered {total} built-in layers")

    def _register_convolutional_layers(self):
        """Register convolutional layers."""
        conv_layers = {
            "GCNConv": (pyg_nn.GCNConv, "Graph Convolutional Network layer"),
            "GATConv": (pyg_nn.GATConv, "Graph Attention Network layer"),
            "SAGEConv": (pyg_nn.SAGEConv, "GraphSAGE layer"),
            "GINConv": (pyg_nn.GINConv, "Graph Isomorphism Network layer"),
            "ChebConv": (pyg_nn.ChebConv, "Chebyshev spectral convolution"),
            "GraphConv": (pyg_nn.GraphConv, "Graph convolution operator"),
            "GatedGraphConv": (pyg_nn.GatedGraphConv, "Gated graph convolution"),
            "EdgeConv": (pyg_nn.EdgeConv, "Edge convolution layer"),
            "TAGConv": (pyg_nn.TAGConv, "Topology Adaptive Graph Convolution"),
            "ARMAConv": (pyg_nn.ARMAConv, "ARMA graph convolution"),
            "SGConv": (pyg_nn.SGConv, "Simple Graph Convolution"),
            "APPNP": (pyg_nn.APPNP, "Approximate Personalized Propagation"),
            "MFConv": (pyg_nn.MFConv, "Max-Relative Graph Convolution"),
            "RGCNConv": (pyg_nn.RGCNConv, "Relational Graph Convolutional layer"),
            "SignedConv": (pyg_nn.SignedConv, "Signed graph convolution"),
            "DNAConv": (pyg_nn.DNAConv, "Dynamic Neighborhood Aggregation"),
            "PANConv": (pyg_nn.PANConv, "Path Integral Based Convolution"),
            "PointNetConv": (pyg_nn.PointNetConv, "PointNet convolution layer"),
            "GMMConv": (pyg_nn.GMMConv, "Gaussian Mixture Model convolution"),
            "SplineConv": (pyg_nn.SplineConv, "Spline-based convolution"),
            "NNConv": (pyg_nn.NNConv, "Neural network convolution"),
            "CGConv": (pyg_nn.CGConv, "Crystal Graph Convolutional layer"),
            "TransformerConv": (pyg_nn.TransformerConv, "Transformer-style convolution"),
            "GATv2Conv": (pyg_nn.GATv2Conv, "Graph Attention Network v2"),
            "SuperGATConv": (pyg_nn.SuperGATConv, "SuperGAT convolution"),
            "FiLMConv": (pyg_nn.FiLMConv, "Feature-wise Linear Modulation"),
            "GeneralConv": (pyg_nn.GeneralConv, "General graph convolution"),
            "HGTConv": (pyg_nn.HGTConv, "Heterogeneous Graph Transformer"),
            "HEATConv": (pyg_nn.HEATConv, "Heterogeneous Edge-Enhanced Attention"),
            "LEConv": (pyg_nn.LEConv, "Local Extremum Convolution"),
            "ClusterGCNConv": (pyg_nn.conv.ClusterGCNConv, "ClusterGCN convolution"),
            "GENConv": (pyg_nn.GENConv, "Generalized Graph Convolution"),
        }

        for name, (layer_class, description) in conv_layers.items():
            metadata = LayerMetadata(
                name=name,
                category=LayerCategory.CONVOLUTIONAL,
                class_path=f"{layer_class.__module__}.{layer_class.__name__}",
                description=description,
                requires_edge_index=True,
                has_in_channels=True,
                has_out_channels=True,
            )
            self._register_layer(name, layer_class, metadata)

    def _register_pooling_layers(self):
        """Register pooling layers with automatic functional wrapping."""
        # Global pooling functions - these need wrapping
        global_pooling_funcs = {
            "global_mean_pool": (
                pyg_nn.global_mean_pool,
                "Global mean pooling",
                True,  # requires_batch
                False,  # requires_edge_index
                False,  # requires_edge_attr
            ),
            "global_max_pool": (pyg_nn.global_max_pool, "Global max pooling", True, False, False),
            "global_add_pool": (pyg_nn.global_add_pool, "Global sum pooling", True, False, False),
        }

        # Register functional pooling layers with wrapping
        for name, (
            func,
            description,
            req_batch,
            req_edge_idx,
            req_edge_attr,
        ) in global_pooling_funcs.items():
            # Wrap the function
            wrapped_class = type(
                f"Wrapped_{name}",
                (FunctionalLayerWrapper,),
                {
                    "__init__": lambda self, f=func, n=name, rb=req_batch, rei=req_edge_idx, rea=req_edge_attr: (
                        FunctionalLayerWrapper.__init__(self, f, n, rb, rei, rea)
                    )
                },
            )

            metadata = LayerMetadata(
                name=name,
                category=LayerCategory.POOLING,
                class_path=f"torch_geometric.nn.{name}",
                description=description,
                requires_edge_index=req_edge_idx,
                requires_batch=req_batch,
                requires_edge_attr=req_edge_attr,
                has_in_channels=False,
                has_out_channels=False,
                modifies_graph_structure=True,
                supported_task_levels=["graph"],
                is_functional=True,
            )
            self._register_layer(name, wrapped_class, metadata)

        # Class-based pooling layers - register normally
        class_pooling_layers = {
            "TopKPooling": (pyg_nn.TopKPooling, "Top-K pooling", True, True),
            "SAGPooling": (pyg_nn.SAGPooling, "Self-Attention Graph Pooling", True, True),
            "EdgePooling": (pyg_nn.EdgePooling, "Edge pooling", True, True),
            "ASAPooling": (pyg_nn.ASAPooling, "Adaptive Structure Aware Pooling", True, True),
            "PANPooling": (pyg_nn.PANPooling, "Path Integral Pooling", True, True),
            "MemPooling": (pyg_nn.MemPooling, "Memory-based pooling", True, True),
        }

        for name, (layer_class, description, has_in_ch, has_out_ch) in class_pooling_layers.items():
            metadata = LayerMetadata(
                name=name,
                category=LayerCategory.POOLING,
                class_path=f"{layer_class.__module__}.{layer_class.__name__}",
                description=description,
                requires_edge_index=True,
                requires_batch=True,
                has_in_channels=has_in_ch,
                has_out_channels=has_out_ch,
                modifies_graph_structure=True,
                supported_task_levels=["graph"],
            )
            self._register_layer(name, layer_class, metadata)

    def _register_normalization_layers(self):
        """Register normalization layers."""
        norm_layers = {
            "BatchNorm": (pyg_nn.BatchNorm, "Batch normalization for graphs"),
            "LayerNorm": (pyg_nn.LayerNorm, "Layer normalization"),
            "InstanceNorm": (pyg_nn.InstanceNorm, "Instance normalization"),
            "GraphNorm": (pyg_nn.GraphNorm, "Graph normalization"),
            "PairNorm": (pyg_nn.PairNorm, "Pair normalization"),
            "MeanSubtractionNorm": (pyg_nn.MeanSubtractionNorm, "Mean subtraction normalization"),
            "DiffGroupNorm": (pyg_nn.DiffGroupNorm, "Differentiable Group Normalization"),
        }

        for name, (layer_class, description) in norm_layers.items():
            metadata = LayerMetadata(
                name=name,
                category=LayerCategory.NORMALIZATION,
                class_path=f"{layer_class.__module__}.{layer_class.__name__}",
                description=description,
                requires_edge_index=False,
                has_in_channels=True,
                has_out_channels=False,
            )
            self._register_layer(name, layer_class, metadata)

    def _register_activation_layers(self):
        """Register activation layers."""
        activation_layers = {
            "ReLU": (nn.ReLU, "Rectified Linear Unit"),
            "LeakyReLU": (nn.LeakyReLU, "Leaky ReLU"),
            "ELU": (nn.ELU, "Exponential Linear Unit"),
            "PReLU": (nn.PReLU, "Parametric ReLU"),
            "GELU": (nn.GELU, "Gaussian Error Linear Unit"),
            "Tanh": (nn.Tanh, "Hyperbolic tangent"),
            "Sigmoid": (nn.Sigmoid, "Sigmoid activation"),
            "Softplus": (nn.Softplus, "Softplus activation"),
            "SiLU": (nn.SiLU, "Sigmoid Linear Unit (Swish)"),
        }

        for name, (layer_class, description) in activation_layers.items():
            metadata = LayerMetadata(
                name=name,
                category=LayerCategory.ACTIVATION,
                class_path=f"{layer_class.__module__}.{layer_class.__name__}",
                description=description,
                requires_edge_index=False,
                has_in_channels=False,
                has_out_channels=False,
            )
            self._register_layer(name, layer_class, metadata)

    def _register_aggregation_layers(self):
        """Register aggregation layers (already wrapped in original implementation)."""

        # These are custom wrappers that were already in the original code
        class MeanAggregation(nn.Module):
            def forward(self, x, index):
                return pyg_nn.global_mean_pool(x, index)

        class MaxAggregation(nn.Module):
            def forward(self, x, index):
                return pyg_nn.global_max_pool(x, index)

        class SumAggregation(nn.Module):
            def forward(self, x, index):
                return pyg_nn.global_add_pool(x, index)

        agg_layers = {
            "MeanAggregation": (MeanAggregation, "Mean aggregation"),
            "MaxAggregation": (MaxAggregation, "Max aggregation"),
            "SumAggregation": (SumAggregation, "Sum aggregation"),
        }

        for name, (layer_class, description) in agg_layers.items():
            metadata = LayerMetadata(
                name=name,
                category=LayerCategory.AGGREGATION,
                class_path=f"milia_pipeline.models.builders.layer_registry.{name}",
                description=description,
                requires_edge_index=False,
                requires_batch=True,
                has_in_channels=False,
                has_out_channels=False,
            )
            self._register_layer(name, layer_class, metadata)

    def _register_standard_layers(self):
        """Register standard PyTorch layers."""
        standard_layers = {
            "Linear": (nn.Linear, LayerCategory.LINEAR, "Fully connected layer"),
            "Dropout": (nn.Dropout, LayerCategory.DROPOUT, "Dropout layer"),
        }

        for name, (layer_class, category, description) in standard_layers.items():
            metadata = LayerMetadata(
                name=name,
                category=category,
                class_path=f"{layer_class.__module__}.{layer_class.__name__}",
                description=description,
                requires_edge_index=False,
                has_in_channels=(name == "Linear"),
                has_out_channels=(name == "Linear"),
            )
            self._register_layer(name, layer_class, metadata)

    def _is_functional(self, obj: Any) -> bool:
        """
        Check if an object is a function rather than a class.

        Args:
            obj: Object to check

        Returns:
            True if object is a function, False if it's a class
        """
        return callable(obj) and not isinstance(obj, type)

    def _create_functional_wrapper(
        self, name: str, func: Callable, metadata: LayerMetadata
    ) -> type[nn.Module]:
        """
        Create a wrapper class for a functional operation.

        Args:
            name: Name of the function
            func: The functional operation
            metadata: Layer metadata with requirements

        Returns:
            Wrapper class that can be instantiated
        """
        # Create a custom wrapper class
        wrapped_class = type(
            f"Wrapped_{name}",
            (FunctionalLayerWrapper,),
            {
                "__init__": lambda self: FunctionalLayerWrapper.__init__(
                    self,
                    func,
                    name,
                    metadata.requires_batch,
                    metadata.requires_edge_index,
                    metadata.requires_edge_attr,
                )
            },
        )

        logger.debug(f"Created functional wrapper for {name}")
        return wrapped_class

    def _register_layer(self, name: str, layer_class: type[nn.Module], metadata: LayerMetadata):
        """
        Internal method to register a layer.
        Automatically wraps functional operations.
        """
        with self._lock:
            # Check if this is a functional operation that needs wrapping
            if self._is_functional(layer_class) and not metadata.is_functional:
                # This is a function but not yet marked as wrapped
                logger.debug(f"Auto-wrapping functional layer: {name}")
                layer_class = self._create_functional_wrapper(name, layer_class, metadata)
                metadata.is_functional = True

            self._layers[name] = layer_class
            self._metadata[name] = metadata

            # Add to category index
            if metadata.category not in self._by_category:
                self._by_category[metadata.category] = set()
            self._by_category[metadata.category].add(name)

            logger.debug(f"Registered layer: {name}")

    def register_custom_layer(
        self,
        name: str,
        layer_class: type[nn.Module],
        metadata: LayerMetadata | None = None,
        category: LayerCategory = LayerCategory.CUSTOM,
        overwrite: bool = False,
    ):
        """
        Register a custom layer.
        Automatically wraps functional operations if needed.

        Args:
            name: Layer name
            layer_class: Layer class or function (will be auto-wrapped if function)
            metadata: Layer metadata (optional, will be auto-generated if None)
            category: Layer category
            overwrite: Whether to overwrite existing layer

        Raises:
            ValueError: If layer already exists and overwrite=False
            TypeError: If layer_class is not callable

        Example:
            >>> class MyCustomLayer(nn.Module):
            ...     def __init__(self, in_channels, out_channels):
            ...         super().__init__()
            ...         self.linear = nn.Linear(in_channels, out_channels)
            ...
            ...     def forward(self, x, edge_index):
            ...         return self.linear(x)
            >>>
            >>> registry = LayerRegistry()
            >>> registry.register_custom_layer("MyCustomLayer", MyCustomLayer)
            >>>
            >>> # Can also register functions
            >>> def my_custom_pool(x, batch):
            ...     return x.mean(dim=0)
            >>>
            >>> registry.register_custom_layer(
            ...     "my_custom_pool",
            ...     my_custom_pool,
            ...     metadata=LayerMetadata(
            ...         name="my_custom_pool",
            ...         category=LayerCategory.POOLING,
            ...         class_path="custom",
            ...         description="Custom pooling",
            ...         requires_batch=True,
            ...         requires_edge_index=False
            ...     )
            ... )
        """
        if not callable(layer_class):
            raise TypeError(f"layer_class must be callable, got {type(layer_class)}")

        with self._lock:
            if name in self._layers and not overwrite:
                raise ValueError(
                    f"Layer '{name}' already registered. Use overwrite=True to replace."
                )

            # Auto-generate metadata if not provided
            if metadata is None:
                is_func = self._is_functional(layer_class)

                metadata = LayerMetadata(
                    name=name,
                    category=category,
                    class_path=f"{layer_class.__module__}.{layer_class.__name__}"
                    if hasattr(layer_class, "__module__")
                    else "custom",
                    description=f"Custom {'function' if is_func else 'layer'}: {name}",
                    is_functional=is_func,
                )

            self._register_layer(name, layer_class, metadata)
            logger.info(f"Registered custom layer: {name}")

    # =========================================================================
    # RETRIEVAL
    # =========================================================================

    def get_layer(self, name: str) -> type[nn.Module]:
        """
        Get layer class by name.

        Args:
            name: Layer name

        Returns:
            Layer class (or wrapped functional layer)

        Raises:
            LayerNotFoundError: If layer not found

        Example:
            >>> registry = LayerRegistry()
            >>> GCNConv = registry.get_layer("GCNConv")
            >>> layer = GCNConv(in_channels=16, out_channels=64)
            >>>
            >>> # Functional layers return wrapped version
            >>> GlobalMeanPool = registry.get_layer("global_mean_pool")
            >>> pool = GlobalMeanPool()  # No arguments needed
        """
        with self._lock:
            if name not in self._layers:
                raise LayerNotFoundError(name, list(self._layers.keys()))
            return self._layers[name]

    def get_layer_metadata(self, name: str) -> LayerMetadata:
        """
        Get layer metadata.

        Args:
            name: Layer name

        Returns:
            LayerMetadata instance

        Raises:
            LayerNotFoundError: If layer not found
        """
        with self._lock:
            if name not in self._metadata:
                raise LayerNotFoundError(name, list(self._layers.keys()))
            return self._metadata[name]

    def has_layer(self, name: str) -> bool:
        """Check if layer exists in registry."""
        with self._lock:
            return name in self._layers

    def list_layers(self, category: LayerCategory | None = None) -> list[str]:
        """
        List available layers.

        Args:
            category: Optional category filter

        Returns:
            List of layer names

        Example:
            >>> registry = LayerRegistry()
            >>> conv_layers = registry.list_layers(LayerCategory.CONVOLUTIONAL)
            >>> all_layers = registry.list_layers()
        """
        with self._lock:
            if category is None:
                return sorted(self._layers.keys())
            else:
                return sorted(self._by_category.get(category, set()))

    def list_categories(self) -> list[LayerCategory]:
        """List available layer categories."""
        with self._lock:
            return list(self._by_category.keys())

    # =========================================================================
    # UTILITY
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            by_category = {cat.value: len(names) for cat, names in self._by_category.items()}

            functional_count = sum(
                1 for metadata in self._metadata.values() if metadata.is_functional
            )

            return {
                "total_layers": len(self._layers),
                "by_category": by_category,
                "categories": [cat.value for cat in self._by_category],
                "functional_layers": functional_count,
                "class_layers": len(self._layers) - functional_count,
            }

    def __len__(self) -> int:
        """Return number of registered layers."""
        with self._lock:
            return len(self._layers)

    def __contains__(self, name: str) -> bool:
        """Check if layer exists."""
        return self.has_layer(name)

    def __repr__(self) -> str:
        """String representation."""
        with self._lock:
            stats = self.get_statistics()
            return (
                f"LayerRegistry(total={len(self._layers)}, "
                f"functional={stats['functional_layers']}, "
                f"class={stats['class_layers']})"
            )


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

# Global registry instance
registry = LayerRegistry()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_layer(name: str) -> type[nn.Module]:
    """Get layer class from global registry."""
    return registry.get_layer(name)


def list_layers(category: LayerCategory | None = None) -> list[str]:
    """List available layers from global registry."""
    return registry.list_layers(category)


def get_layer_metadata(name: str) -> LayerMetadata:
    """Get layer metadata from global registry."""
    return registry.get_layer_metadata(name)


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info(f"layer_registry module loaded - {len(registry)} layers registered")
