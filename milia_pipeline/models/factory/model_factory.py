"""
Model Factory

Factory pattern for creating model instances with comprehensive validation.
Handles hyperparameter processing, channel inference, and device placement.

PHASE 7 EXTENSION: Added support for custom architectures and ensembles.
PHASE 4 MIGRATION: Updated to use dynamic introspection from pyg_introspector.

Author: milia Team
Version: 1.2.0
"""

import contextlib
import logging
from datetime import datetime
from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from ..registry.model_registry import ModelRegistry

# Dynamic introspection replaces static model_categories (Phase 4 Migration)
from ..registry.pyg_introspector import (
    ModelMetadata,  # Alias for DynamicModelMetadata
    get_model_conv_kwargs,  # NEW (Fix 19): Model-specific conv kwargs filtering
    get_model_metadata,
    validate_params_against_signature,
)

# Import exceptions with fallback
try:
    from milia_pipeline.exceptions import (
        DataCompatibilityError,
        HyperparameterError,
        ModelError,
        ModelInstantiationError,
        ModelNotFoundError,
        ModelValidationError,
    )
except ImportError:
    # Fallback exceptions if main exceptions module not available
    class ModelError(Exception):
        """Base exception for model-related errors."""

        pass

    class ModelValidationError(ModelError):
        """Exception raised when model validation fails."""

        pass

    class ModelInstantiationError(ModelError):
        """Exception raised when model instantiation fails."""

        pass

    class ModelNotFoundError(ModelError):
        """Exception raised when model is not found in registry."""

        pass

    class HyperparameterError(ModelError):
        """Exception raised for hyperparameter issues."""

        pass

    class DataCompatibilityError(ModelError):
        """Exception raised when data is incompatible with model."""

        pass


# Import builders module (conditional for backward compatibility)
try:
    from ..builders.config_parser import parse_custom_architecture, parse_ensemble

    _BUILDERS_AVAILABLE = True
except ImportError:
    _BUILDERS_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.debug("Builders module not available, custom/ensemble models not supported")

# Import pyg_integration utilities (conditional for backward compatibility)
try:
    from ..utils.pyg_integration import infer_out_channels

    _PYG_INTEGRATION_AVAILABLE = True
except ImportError:
    infer_out_channels = None
    _PYG_INTEGRATION_AVAILABLE = False


logger = logging.getLogger(__name__)


# =============================================================================
# DYNAMIC PYG OPTIONAL DEPENDENCY VALIDATION
# =============================================================================
# DYNAMIC: Checks dependencies at runtime by inspecting model source
# PRODUCTION-READY: Provides clear, actionable error messages
# FUTURE-PROOF: Works with any PyG model without hardcoded model names
# =============================================================================


def _check_pyg_optional_dependency(package_name: str) -> bool:
    """
    Check if a PyG optional dependency is available.

    Args:
        package_name: Name of the package to check (e.g., 'torch_cluster')

    Returns:
        True if package is available and functional, False otherwise
    """
    try:
        import importlib

        module = importlib.import_module(package_name)
        return module is not None
    except ImportError:
        return False
    except Exception:
        return False


def _get_pyg_function_dependencies() -> dict[str, str]:
    """
    Map PyG functions to their required optional packages.

    Returns:
        Dict mapping function names to package names

    Note:
        This mapping is based on PyG's architecture where certain functions
        are only available when optional packages are installed.
    """
    return {
        # torch_cluster functions
        "radius_graph": "torch_cluster",
        "radius": "torch_cluster",
        "knn_graph": "torch_cluster",
        "knn": "torch_cluster",
        "nearest": "torch_cluster",
        "graclus_cluster": "torch_cluster",
        "grid_cluster": "torch_cluster",
        "fps": "torch_cluster",  # Farthest point sampling
        "random_walk": "torch_cluster",
        # torch_sparse functions
        "SparseTensor": "torch_sparse",
        "spmm": "torch_sparse",
        "spspmm": "torch_sparse",
        # torch_scatter functions
        "scatter": "torch_scatter",
        "scatter_add": "torch_scatter",
        "scatter_mean": "torch_scatter",
        "scatter_max": "torch_scatter",
        "scatter_min": "torch_scatter",
        # torch_spline_conv functions
        "SplineConv": "torch_spline_conv",
    }


def _detect_model_dependencies(model_class: type) -> dict[str, str]:
    """
    Dynamically detect what PyG optional dependencies a model requires.

    Uses source code inspection to find imports and function calls that
    require optional packages.

    DYNAMIC: Inspects actual model module source code at runtime
    PRODUCTION-READY: Handles missing source gracefully
    FUTURE-PROOF: Works with any model without hardcoding

    Args:
        model_class: The model class to inspect

    Returns:
        Dict mapping dependency functions found to their required packages

    Example:
        >>> from torch_geometric.nn.models import SchNet
        >>> deps = _detect_model_dependencies(SchNet)
        >>> print(deps)
        {'radius_graph': 'torch_cluster'}
    """
    import inspect

    required_deps = {}
    function_to_package = _get_pyg_function_dependencies()

    # Primary approach: Check the MODULE source where imports live
    # This is important because PyG models import functions like radius_graph
    # at module level, not within the class definition
    try:
        module = inspect.getmodule(model_class)
        if module is not None:
            module_source = inspect.getsource(module)
            for func_name, package_name in function_to_package.items():
                if func_name in module_source:
                    required_deps[func_name] = package_name
    except (OSError, TypeError):
        # Module source not available, fall back to class source
        try:
            source = inspect.getsource(model_class)
            for func_name, package_name in function_to_package.items():
                if func_name in source:
                    required_deps[func_name] = package_name
        except (OSError, TypeError):
            pass  # Cannot inspect, will rely on runtime error

    return required_deps


def validate_model_dependencies(model_class: type, model_name: str) -> None:
    """
    Validate that all required optional dependencies for a model are available.

    DYNAMIC: Detects dependencies from model source code
    PRODUCTION-READY: Provides clear error message about missing packages
    FUTURE-PROOF: Works with any PyG model

    Args:
        model_class: The model class to validate
        model_name: Name of the model (for error messages)

    Raises:
        ModelInstantiationError: If required dependencies are missing

    Example:
        >>> from torch_geometric.nn.models import SchNet
        >>> validate_model_dependencies(SchNet, 'SchNet')
        ModelInstantiationError: Model 'SchNet' requires 'torch_cluster' package...
    """
    # Detect what dependencies this model needs
    required_deps = _detect_model_dependencies(model_class)

    if not required_deps:
        return  # No special dependencies detected

    # Check which required packages are missing
    missing_packages = set()
    missing_functions = []

    for func_name, package_name in required_deps.items():
        if not _check_pyg_optional_dependency(package_name):
            missing_packages.add(package_name)
            missing_functions.append(f"'{func_name}' (requires {package_name})")

    if missing_packages:
        packages_str = ", ".join(sorted(missing_packages))
        functions_str = ", ".join(missing_functions)

        raise ModelInstantiationError(
            f"Model '{model_name}' requires optional PyG package(s): {packages_str}. "
            f"Functions used: {functions_str}.",
            model_name=model_name,
        )


# =============================================================================
# GRAPH-LEVEL MODEL WRAPPER
# =============================================================================


class GraphLevelModelWrapper(torch.nn.Module):
    """
    Wrapper for single models to handle graph-level tasks.

    PyG's BasicGNN models (GCN, GAT, GraphSAGE, etc.) output node-level
    predictions [num_nodes, out_channels]. For graph-level tasks, we need
    to apply global pooling to get [num_graphs, out_channels].

    Some 3D models (SchNet, DimeNet) have fixed output dimensions (typically 1)
    regardless of the target dimensionality. This wrapper can add an output
    projection layer to match the required out_channels.

    This wrapper:
    1. Passes input through the wrapped model
    2. Detects if output needs global pooling (based on task_type)
    3. Applies appropriate pooling for graph-level tasks
    4. Applies output projection if model output doesn't match out_channels
    5. Passes through unchanged for node-level tasks

    Features:
    - Dynamic task detection (any task starting with 'graph_')
    - Configurable pooling method (mean, max, add)
    - Automatic output shape detection
    - Output projection for fixed-output models (SchNet, DimeNet)
    - Backward compatible (node-level tasks unchanged)
    - CUDA compatible (uses PyG native pooling)

    Usage:
        >>> model = GCN(in_channels=16, hidden_channels=64, out_channels=1)
        >>> wrapped = GraphLevelModelWrapper(model, task_type='graph_regression')
        >>> out = wrapped(x, edge_index, batch=batch)  # [num_graphs, 1]

        >>> # For SchNet with multi-target output:
        >>> schnet = SchNet(hidden_channels=128)
        >>> wrapped = GraphLevelModelWrapper(schnet, task_type='graph_regression', out_channels=8)
        >>> out = wrapped(z=z, pos=pos, batch=batch)  # [num_graphs, 8]
    """

    def __init__(
        self,
        model: torch.nn.Module,
        task_type: str,
        pooling_method: str = "mean",
        out_channels: int | None = None,
    ):
        """
        Initialize wrapper.

        Args:
            model: The model to wrap (e.g., GCN, GAT, GraphSAGE, SchNet)
            task_type: Task type string (e.g., 'graph_regression', 'node_classification')
            pooling_method: Pooling method for graph-level tasks ('mean', 'max', 'add')
            out_channels: Target output channels. If provided and model's output
                         doesn't match, an output projection layer is added.
                         This is essential for models like SchNet that have
                         fixed output dimensions (1) regardless of task.
        """
        super().__init__()
        self.model = model
        self.task_type = task_type
        self.pooling_method = pooling_method
        self.out_channels = out_channels

        # Output projection layer for models with fixed output dimensions
        # This is dynamically created on first forward pass when we know the model's output dim
        self.output_projection = None
        self._model_out_dim = None  # Will be set on first forward

        logger.debug(
            f"GraphLevelModelWrapper initialized: task_type={task_type}, "
            f"pooling_method={pooling_method}, out_channels={out_channels}"
        )

    def _is_graph_level_task(self) -> bool:
        """
        Determine if current task is graph-level.

        Returns:
            True if task_type starts with 'graph_' (case-insensitive)
        """
        if self.task_type is None:
            return False
        return self.task_type.lower().startswith("graph_")

    def _apply_global_pooling(
        self, x: torch.Tensor, batch: torch.Tensor | None, pooling_method: str = "mean"
    ) -> torch.Tensor:
        """
        Apply global pooling to convert node features to graph features.

        Args:
            x: Node features [num_nodes, features]
            batch: Batch assignment vector [num_nodes]
            pooling_method: One of 'mean', 'max', 'add'

        Returns:
            Graph features [num_graphs, features]
        """
        # Handle single graph case (no batch vector)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Apply pooling based on method
        if pooling_method == "mean":
            return global_mean_pool(x, batch)
        elif pooling_method == "max":
            return global_max_pool(x, batch)
        elif pooling_method == "add":
            return global_add_pool(x, batch)
        else:
            logger.warning(f"Unknown pooling method '{pooling_method}', using 'mean'")
            return global_mean_pool(x, batch)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass with automatic global pooling for graph-level tasks.

        Supports multiple calling conventions:
        - forward(x, edge_index, batch=batch)
        - forward(x, edge_index, edge_attr, batch=batch)
        - forward(z, pos, batch=batch) for 3D models like SchNet
        - forward(data) where data is a PyG Data/Batch object

        Returns:
            Model output, with global pooling applied for graph-level tasks,
            and output projection applied if out_channels doesn't match model output
        """
        # Extract batch from kwargs or args
        batch = kwargs.get("batch")

        # Check if first argument is a Data object
        if len(args) > 0 and hasattr(args[0], "batch"):
            batch = args[0].batch

        # Forward through wrapped model
        out = self.model(*args, **kwargs)

        # Apply global pooling for graph-level tasks
        if self._is_graph_level_task():
            # Check if output is node-level (needs pooling)
            # Node-level: [num_nodes, features], Graph-level: [num_graphs, features]
            if batch is not None:
                num_graphs = batch.max().item() + 1
                # If output size doesn't match num_graphs, apply pooling
                if out.size(0) != num_graphs:
                    out = self._apply_global_pooling(out, batch, self.pooling_method)
                    logger.debug(
                        f"Applied {self.pooling_method} pooling: "
                        f"[{out.size(0)}, {out.size(1) if out.dim() > 1 else 1}]"
                    )
            elif out.size(0) > 1:
                # Single graph case - pool all nodes
                out = self._apply_global_pooling(out, batch, self.pooling_method)

        # Apply output projection if out_channels is specified and doesn't match
        # This handles models like SchNet that have fixed output dimensions (1)
        if self.out_channels is not None:
            model_out_dim = out.size(-1) if out.dim() > 1 else 1

            # Create projection layer if needed (lazy initialization)
            if model_out_dim != self.out_channels:
                if self.output_projection is None or self._model_out_dim != model_out_dim:
                    self._model_out_dim = model_out_dim
                    self.output_projection = torch.nn.Linear(model_out_dim, self.out_channels).to(
                        out.device
                    )
                    logger.info(
                        f"Created output projection: {model_out_dim} -> {self.out_channels} "
                        f"for {type(self.model).__name__}"
                    )

                # Handle 1D output (scalar per graph)
                if out.dim() == 1:
                    out = out.unsqueeze(-1)

                out = self.output_projection(out)

        return out

    def __getattr__(self, name: str):
        """
        Delegate attribute access to wrapped model.

        This implementation safely handles the nn.Module initialization phase
        and correctly accesses the wrapped model from _modules registry.

        Pattern aligned with PyTorch's wrapper modules.
        """
        # First, try the standard nn.Module attribute lookup
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass

        # Safely access 'model' from _modules (where nn.Module stores submodules)
        try:
            _modules = object.__getattribute__(self, "_modules")
            if "model" in _modules:
                model = _modules["model"]
                return getattr(model, name)
        except (AttributeError, KeyError):
            pass

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


# =============================================================================
# EDGE-LEVEL MODEL WRAPPER
# =============================================================================


class EdgeLevelModelWrapper(torch.nn.Module):
    """
    Wrapper for models to handle edge-level tasks (link_prediction, edge_regression, edge_classification).

    PyG's BasicGNN models (GCN, GAT, GraphSAGE, etc.) output node-level
    embeddings [num_nodes, out_channels]. For edge-level tasks, we need
    to compute edge scores from node embeddings using a decoder.

    This wrapper:
    1. Passes input through the wrapped model to get node embeddings
    2. Applies edge decoder to compute edge scores from node pairs
    3. Returns edge-level predictions [num_edges] or [num_edges, out_channels]

    Features:
    - Dynamic task detection (link_prediction, edge_regression, edge_classification)
    - Configurable decoder method (dot_product, concat_mlp, etc.)
    - Supports edge_label_index for link prediction
    - Supports multi-dimensional output for edge_regression and edge_classification
    - Backward compatible (non-edge tasks pass through unchanged)
    - CUDA compatible

    Usage:
        >>> model = GCN(in_channels=16, hidden_channels=64, out_channels=64)
        >>> wrapped = EdgeLevelModelWrapper(model, task_type='link_prediction')
        >>> # For link prediction: pass edge_label_index
        >>> out = wrapped(x, edge_index, edge_label_index=edge_label_index, batch=batch)
        >>> # Returns: [num_edge_pairs] edge scores

        >>> # For edge_regression with multi-dimensional output:
        >>> wrapped = EdgeLevelModelWrapper(model, task_type='edge_regression',
        ...                                  edge_out_channels=21)
        >>> out = wrapped(x, edge_index)
        >>> # Returns: [num_edges, 21] edge predictions

        >>> # For edge_classification with num_classes output:
        >>> wrapped = EdgeLevelModelWrapper(model, task_type='edge_classification',
        ...                                  edge_out_channels=10)  # 10 classes
        >>> out = wrapped(x, edge_index)
        >>> # Returns: [num_edges, 10] class logits
    """

    def __init__(
        self,
        model: torch.nn.Module,
        task_type: str,
        decoder_method: str = "dot_product",
        edge_out_channels: int | None = None,
        model_out_channels: int | None = None,
    ):
        """
        Initialize wrapper.

        Args:
            model: The model to wrap (e.g., GCN, GAT, GraphSAGE, ParallelEnsemble)
            task_type: Task type string ('link_prediction', 'edge_regression', 'edge_classification')
            decoder_method: Decoder method for computing edge scores
                           - 'dot_product': (z[src] * z[dst]).sum(-1) -> scalar per edge
                           - 'concat_mlp': MLP(concat(z[src], z[dst])) -> multi-dim output
                           - 'hadamard_mlp': MLP(z[src] * z[dst]) -> multi-dim output
            edge_out_channels: Output channels for edge_regression/edge_classification.
                              For classification, this is num_classes.
                              If None, uses scalar output.
            model_out_channels: Explicit output embedding dimension of wrapped model.
                               If provided, overrides auto-inference. Use this when
                               wrapping ensemble models where auto-inference fails.
        """
        super().__init__()
        self.model = model
        self.task_type = task_type
        self.decoder_method = decoder_method
        self.edge_out_channels = edge_out_channels

        # Get model's output embedding dimension for MLP decoder
        # Use explicit value if provided, otherwise infer from model
        if model_out_channels is not None:
            self._model_out_channels = model_out_channels
        else:
            self._model_out_channels = self._infer_model_out_channels()

        # Create MLP decoder for multi-dimensional edge_regression/edge_classification
        self.edge_mlp = None
        if self._needs_mlp_decoder():
            self._create_mlp_decoder()

        logger.debug(
            f"EdgeLevelModelWrapper initialized: task_type={task_type}, "
            f"decoder_method={decoder_method}, edge_out_channels={edge_out_channels}, "
            f"model_out_channels={self._model_out_channels}"
        )

    def _infer_model_out_channels(self) -> int:
        """Infer the output embedding dimension from wrapped model."""
        # Try common attribute names
        for attr in ["out_channels", "hidden_channels", "embed_dim", "output_dim"]:
            if hasattr(self.model, attr):
                val = getattr(self.model, attr)
                if isinstance(val, int) and val > 0:
                    return val
        # Default fallback
        return 64

    def _needs_mlp_decoder(self) -> bool:
        """Check if we need an MLP decoder for multi-dimensional output."""
        if self.task_type is None:
            return False
        task_lower = self.task_type.lower()
        # edge_regression with multi-dimensional output needs MLP
        if task_lower == "edge_regression" and self.edge_out_channels is not None:
            return True
        # edge_classification needs MLP to output class logits
        if task_lower == "edge_classification" and self.edge_out_channels is not None:
            return True
        # Use MLP for specific decoder methods
        return self.decoder_method in ["concat_mlp", "hadamard_mlp"]

    def _create_mlp_decoder(self):
        """Create MLP decoder for edge predictions."""
        out_dim = self.edge_out_channels if self.edge_out_channels is not None else 1

        if self.decoder_method == "concat_mlp":
            # Input: concatenation of src and dst embeddings
            in_dim = self._model_out_channels * 2
        else:
            # Input: element-wise product (hadamard) of embeddings
            in_dim = self._model_out_channels

        # Simple 2-layer MLP
        hidden_dim = max(in_dim // 2, out_dim * 2, 32)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )
        logger.debug(
            f"Created MLP decoder: in_dim={in_dim}, hidden_dim={hidden_dim}, out_dim={out_dim}"
        )

    def _is_edge_level_task(self) -> bool:
        """
        Determine if current task is edge-level.

        Returns:
            True if task_type is an edge-level task
        """
        if self.task_type is None:
            return False
        task_lower = self.task_type.lower()
        return (
            task_lower in ["link_prediction", "edge_regression"]
            or task_lower.startswith("link_")
            or task_lower.startswith("edge_")
        )

    def _decode_edges(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute edge scores from node embeddings.

        Args:
            z: Node embeddings [num_nodes, embedding_dim]
            edge_index: Edge indices [2, num_edges] - pairs to score

        Returns:
            Edge scores [num_edges] for link_prediction
            Edge predictions [num_edges, edge_out_channels] for edge_regression
        """
        src, dst = edge_index[0], edge_index[1]
        z_src, z_dst = z[src], z[dst]

        if self.decoder_method == "dot_product":
            # Dot product decoder: element-wise product then sum -> scalar per edge
            return (z_src * z_dst).sum(dim=-1)

        elif self.decoder_method == "concat_mlp":
            # Concatenate src and dst embeddings, pass through MLP
            edge_features = torch.cat([z_src, z_dst], dim=-1)
            return self.edge_mlp(edge_features)

        elif self.decoder_method == "hadamard_mlp":
            # Element-wise product (Hadamard), pass through MLP
            edge_features = z_src * z_dst
            return self.edge_mlp(edge_features)

        else:
            # Default to dot product with warning
            logger.warning(f"Unknown decoder method '{self.decoder_method}', using 'dot_product'")
            return (z_src * z_dst).sum(dim=-1)

    def forward(
        self,
        x: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_label_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with edge decoding for edge-level tasks.

        Supports multiple calling conventions:
        - forward(x, edge_index, batch=batch)
        - forward(x, edge_index, edge_attr=edge_attr, batch=batch)
        - forward(data) where data is a PyG Data/Batch object

        For edge-level tasks:
        1. Encode: Get node embeddings from wrapped model
        2. Decode: Compute edge scores from node pairs

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph structure [2, num_edges] (used for message passing)
            edge_label_index: Edge pairs to predict [2, num_edge_pairs] (for link prediction)
                             If None, uses edge_index
            edge_attr: Edge attributes (optional, passed to model if supported)
            batch: Batch assignment [num_nodes] (optional)
            **kwargs: Additional arguments passed to model

        Returns:
            For edge-level tasks: Edge scores [num_edges] or [num_edge_pairs]
            For other tasks: Model output unchanged
        """
        # Handle case where first arg is a Data/Batch object
        # This happens when trainer calls model(batch)
        if x is not None and hasattr(x, "x") and hasattr(x, "edge_index"):
            # x is actually a Data/Batch object
            data = x
            x = data.x
            edge_index = data.edge_index
            edge_attr = getattr(data, "edge_attr", None)
            batch = getattr(data, "batch", None)
            edge_label_index = getattr(data, "edge_label_index", None)

        # Validate required inputs
        if x is None or edge_index is None:
            raise ValueError(
                "EdgeLevelModelWrapper.forward() requires x and edge_index. "
                "Pass either (x, edge_index, ...) or a PyG Data/Batch object."
            )

        # Build forward arguments for wrapped model
        # Try different signatures for compatibility with various models
        # =================================================================
        # FIX 29b: VGAE/Variational Autoencoder Encoding Support
        # =================================================================
        # ISSUE: VGAE's forward() inherits from GAE and returns the raw
        # encoder output. For variational encoders, this is a tuple
        # (mu, logstd), not a tensor. EdgeLevelModelWrapper then crashes
        # when trying to access z.shape on a tuple.
        #
        # ROOT CAUSE: PyG's VGAE.forward() returns self.encoder(*args)
        # directly, while VGAE.encode() performs reparameterization to
        # return a proper tensor z. The standard usage pattern for VGAE
        # is to call model.encode(), not model.forward().
        #
        # FIX: Check if the wrapped model has an encode() method (like
        # GAE/VGAE) and use that instead of direct __call__. This ensures
        # proper reparameterization for variational autoencoders.
        #
        # DYNAMIC: Uses hasattr() to detect encode() at runtime
        # PRODUCTION-READY: Works with both GAE (tensor) and VGAE (tuple)
        # FUTURE-PROOF: Works with any autoencoder following PyG pattern
        # =================================================================

        # Determine whether to use encode() method (for autoencoders)
        # GAE/VGAE models have encode() which handles variational reparameterization
        use_encode_method = hasattr(self.model, "encode") and callable(self.model.encode)

        try:
            if use_encode_method:
                # Use encode() for autoencoder models (handles reparameterization)
                if edge_attr is not None and batch is not None:
                    z = self.model.encode(x, edge_index, edge_attr, batch=batch, **kwargs)
                elif batch is not None:
                    z = self.model.encode(x, edge_index, batch=batch, **kwargs)
                elif edge_attr is not None:
                    z = self.model.encode(x, edge_index, edge_attr, **kwargs)
                else:
                    z = self.model.encode(x, edge_index, **kwargs)
            else:
                # Standard forward call for non-autoencoder models
                if edge_attr is not None and batch is not None:
                    z = self.model(x, edge_index, edge_attr, batch=batch, **kwargs)
                elif batch is not None:
                    z = self.model(x, edge_index, batch=batch, **kwargs)
                elif edge_attr is not None:
                    z = self.model(x, edge_index, edge_attr, **kwargs)
                else:
                    z = self.model(x, edge_index, **kwargs)
        except TypeError:
            # Fallback to simple signature
            try:
                if use_encode_method:
                    z = self.model.encode(x, edge_index)
                else:
                    z = self.model(x, edge_index)
            except TypeError:
                z = self.model.encode(x) if use_encode_method else self.model(x)

        # Handle tuple output from variational encoders that don't use encode()
        # This is a safety net for edge cases
        if isinstance(z, tuple):
            # For variational autoencoders, z is (mu, logstd)
            # Use mu directly (equivalent to eval mode reparameterization)
            logger.debug(
                "EdgeLevelModelWrapper: Received tuple output (likely variational encoder), "
                "using first element (mu) for decoding"
            )
            z = z[0]

        # For edge-level tasks, decode node embeddings to edge scores
        if self._is_edge_level_task():
            # Use edge_label_index if provided (for link prediction)
            # Otherwise use edge_index (for edge regression on existing edges)
            target_edges = edge_label_index if edge_label_index is not None else edge_index
            logger.debug(
                f"EdgeLevelModelWrapper: task_type={self.task_type}, "
                f"_is_edge_level_task=True, z.shape={list(z.shape)}, "
                f"target_edges.shape={list(target_edges.shape)}, decoder={self.decoder_method}"
            )
            decoded = self._decode_edges(z, target_edges)
            logger.debug(f"EdgeLevelModelWrapper: decoded.shape={list(decoded.shape)}")
            return decoded

        # Non-edge tasks: pass through unchanged
        logger.debug(
            f"EdgeLevelModelWrapper: task_type={self.task_type}, _is_edge_level_task=False, returning z unchanged"
        )
        return z

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Encode nodes to embeddings (without decoding).

        Useful when you need node embeddings separately.

        For autoencoder models (GAE/VGAE), this calls model.encode() to ensure
        proper reparameterization for variational models.

        Args:
            x: Node features
            edge_index: Graph structure
            edge_attr: Edge attributes (optional)
            batch: Batch assignment (optional)

        Returns:
            Node embeddings [num_nodes, embedding_dim]
        """
        # Use encode() method if available (for autoencoder models)
        use_encode_method = hasattr(self.model, "encode") and callable(self.model.encode)

        try:
            if use_encode_method:
                if edge_attr is not None and batch is not None:
                    z = self.model.encode(x, edge_index, edge_attr, batch=batch, **kwargs)
                elif batch is not None:
                    z = self.model.encode(x, edge_index, batch=batch, **kwargs)
                elif edge_attr is not None:
                    z = self.model.encode(x, edge_index, edge_attr, **kwargs)
                else:
                    z = self.model.encode(x, edge_index, **kwargs)
            else:
                if edge_attr is not None and batch is not None:
                    z = self.model(x, edge_index, edge_attr, batch=batch, **kwargs)
                elif batch is not None:
                    z = self.model(x, edge_index, batch=batch, **kwargs)
                elif edge_attr is not None:
                    z = self.model(x, edge_index, edge_attr, **kwargs)
                else:
                    z = self.model(x, edge_index, **kwargs)
        except TypeError:
            try:
                if use_encode_method:
                    z = self.model.encode(x, edge_index)
                else:
                    z = self.model(x, edge_index)
            except TypeError:
                z = self.model.encode(x) if use_encode_method else self.model(x)

        # Handle tuple output from variational encoders
        if isinstance(z, tuple):
            z = z[0]

        return z

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Decode edge scores from node embeddings.

        Args:
            z: Node embeddings [num_nodes, embedding_dim]
            edge_index: Edge indices to decode [2, num_edges]

        Returns:
            Edge scores [num_edges]
        """
        return self._decode_edges(z, edge_index)

    def __getattr__(self, name: str):
        """
        Delegate attribute access to wrapped model.

        This implementation safely handles the nn.Module initialization phase
        and prevents infinite recursion by using object.__getattribute__ to
        access internal attributes.

        Pattern used by PyTorch's DataParallel and DistributedDataParallel.
        """
        # First, try the standard nn.Module attribute lookup
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass

        # Safely access 'model' without triggering __getattr__ recursion
        # This pattern is used by PyTorch's own wrapper modules
        try:
            model = object.__getattribute__(self, "model")
            return getattr(model, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from None


# =============================================================================
# MODEL VALIDATOR
# =============================================================================


class ModelValidator:
    """
    Validator for model hyperparameters and data compatibility.

    Validates:
    - Hyperparameter types and ranges
    - Required vs optional parameters
    - Data compatibility with model requirements

    Usage:
        >>> validator = ModelValidator()
        >>> validator.validate_hyperparameters(hparams, schema)
        >>> validator.validate_data_compatibility(data, metadata)
    """

    def validate_hyperparameters(
        self, hyperparameters: dict[str, Any], schema: dict[str, Any]
    ) -> None:
        """
        Validate hyperparameters against schema.

        Schema format (JSON Schema style):
        {
            "param_name": {
                "type": "integer" | "float" | "string" | "boolean" | "array" | "module",
                "required": True | False,
                "default": value,
                "min": min_value,
                "max": max_value,
                "options": [allowed_values],
                "description": "Parameter description"
            }
        }

        Args:
            hyperparameters: Dictionary of hyperparameter values
            schema: Hyperparameter schema from ModelMetadata

        Raises:
            HyperparameterError: If validation fails

        Example:
            >>> schema = {
            ...     "hidden_channels": {"type": "integer", "required": True, "min": 1},
            ...     "dropout": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0}
            ... }
            >>> validator.validate_hyperparameters({"hidden_channels": 64}, schema)
        """
        errors = []

        # Check required parameters
        for param, spec in schema.items():
            if (
                spec.get("required", False)
                and param not in hyperparameters
                and "default" not in spec
            ):
                errors.append(
                    f"Required parameter '{param}' is missing and has no default value"
                )

        # Validate provided parameters
        for param, value in hyperparameters.items():
            if param not in schema:
                # Allow extra parameters not in schema (for flexibility)
                logger.debug(f"Parameter '{param}' not in schema, allowing anyway")
                continue

            spec = schema[param]

            # Skip validation for None values if parameter is optional or None is in options
            if value is None:
                # Check if None is explicitly allowed
                options = spec.get("options", [])
                is_optional = not spec.get("required", False)
                if None in options or is_optional:
                    logger.debug(f"Parameter '{param}' is None (allowed for optional parameter)")
                    continue

            # Type validation
            expected_type = spec.get("type")
            if expected_type and not self._validate_type(value, expected_type, param):
                errors.append(
                    f"Parameter '{param}' has invalid type. "
                    f"Expected {expected_type}, got {type(value).__name__}"
                )
                continue

            # Range validation (for numeric types)
            if expected_type in ["integer", "float"]:
                if "min" in spec and value < spec["min"]:
                    errors.append(f"Parameter '{param}' must be >= {spec['min']}, got {value}")
                if "max" in spec and value > spec["max"]:
                    errors.append(f"Parameter '{param}' must be <= {spec['max']}, got {value}")

            # Options validation (enum-like)
            if "options" in spec:
                options = spec["options"]
                # Check if value is in allowed options
                if value not in options:
                    errors.append(f"Parameter '{param}' must be one of {options}, got '{value}'")

        if errors:
            error_msg = "Hyperparameter validation failed:\n  - " + "\n  - ".join(errors)
            raise HyperparameterError(error_msg)

        logger.debug("Hyperparameter validation passed")

    def _validate_type(self, value: Any, expected_type: str, param_name: str) -> bool:
        """
        Validate value type against expected type.

        Args:
            value: Value to validate
            expected_type: Expected type string
            param_name: Parameter name (for logging)

        Returns:
            True if type is valid, False otherwise
        """
        if expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        elif expected_type == "float":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        elif expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "array":
            return isinstance(value, (list, tuple))
        elif expected_type == "module":
            # For module types (like conv, norm in DeepGCN)
            return True  # Skip validation for module types
        else:
            logger.warning(f"Unknown type '{expected_type}' for parameter '{param_name}'")
            return True  # Allow unknown types

    def validate_data_compatibility(self, data: Data, metadata: ModelMetadata) -> None:
        """
        Validate data compatibility with model requirements.

        DYNAMIC: Uses introspected forward signature to determine requirements
        PRODUCTION-READY: Provides clear error messages for incompatibility
        FUTURE-PROOF: Works with any model (SchNet, DimeNet, GCN, etc.)

        Checks:
        - Required graph structure (edge_index)
        - Edge features if required
        - Edge weights if required
        - Heterogeneous graph support
        - Forward signature requirements (z, pos for SchNet/DimeNet, etc.)

        Args:
            data: PyG Data object
            metadata: Model metadata with requirements

        Raises:
            DataCompatibilityError: If data is incompatible

        Example:
            >>> from torch_geometric.data import Data
            >>> data = Data(x=torch.randn(10, 16), edge_index=torch.randint(0, 10, (2, 20)))
            >>> validator.validate_data_compatibility(data, model_metadata)

            >>> # For SchNet - will fail if data lacks z and pos:
            >>> validator.validate_data_compatibility(data, schnet_metadata)
            DataCompatibilityError: Model requires 'z' (atomic numbers) but data lacks this attribute
        """
        errors = []

        # =====================================================================
        # CHECK FORWARD SIGNATURE REQUIREMENTS (DYNAMIC)
        # =====================================================================
        # This is the key validation for models like SchNet, DimeNet that require
        # specific data attributes (z, pos) that standard GNNs don't need.
        # =====================================================================
        if hasattr(metadata, "required_data_attributes") and metadata.required_data_attributes:
            required_attrs = metadata.required_data_attributes

            # Map attribute names to human-readable descriptions
            attr_descriptions = {
                "z": "atomic numbers",
                "pos": "atomic positions/coordinates",
                "x": "node features",
                "edge_index": "graph connectivity",
                "edge_attr": "edge features",
                "edge_weight": "edge weights",
            }

            for attr in required_attrs:
                if not hasattr(data, attr) or getattr(data, attr) is None:
                    desc = attr_descriptions.get(attr, attr)
                    errors.append(
                        f"Model requires '{attr}' ({desc}) but data lacks this attribute. "
                        f"This model may require specialized molecular data."
                    )

        # =====================================================================
        # STANDARD CHECKS (BACKWARD COMPATIBLE)
        # =====================================================================

        # Check edge_index requirement
        if not hasattr(data, "edge_index") or data.edge_index is None:
            errors.append("Data missing edge_index (required for graph structure)")

        # Check edge features requirement
        if metadata.requires_edge_features and (
            not hasattr(data, "edge_attr") or data.edge_attr is None
        ):
            errors.append("Model requires edge features, but data has no edge_attr")

        # Check edge weights requirement
        if metadata.requires_edge_weights and (
            not hasattr(data, "edge_weight") or data.edge_weight is None
        ):
            errors.append("Model requires edge weights, but data has no edge_weight")

        # Check heterogeneous graph support
        if (
            (hasattr(data, "node_type") or hasattr(data, "edge_type"))
            and not metadata.supports_heterogeneous
        ):
            # This is a heterogeneous graph
            errors.append(
                "Model does not support heterogeneous graphs, "
                "but data appears to be heterogeneous"
            )

        if errors:
            error_msg = "Data compatibility validation failed:\n  - " + "\n  - ".join(errors)
            raise DataCompatibilityError(error_msg)

        logger.debug("Data compatibility validation passed")


# =============================================================================
# MODEL FACTORY
# =============================================================================


class ModelFactory:
    """
    Factory for creating model instances with validation.

    Supports:
    - Standard models from registry (GCN, GAT, etc.)
    - Custom architectures via ArchitectureBuilder
    - Ensemble models via ModelComposer

    Features:
    - Automatic channel inference
    - Hyperparameter validation
    - Data compatibility checking
    - Device placement

    Usage:
        >>> factory = ModelFactory()
        >>> model = factory.create_model(
        ...     name="GCN",
        ...     hyperparameters={"hidden_channels": 64, "num_layers": 3},
        ...     task_type="graph_regression",
        ...     sample_data=sample_data
        ... )
    """

    def __init__(self):
        """Initialize factory with validator and registry."""
        self.validator = ModelValidator()
        self.registry = ModelRegistry()
        logger.debug("ModelFactory initialized")

    def create_model(
        self,
        name: str,
        hyperparameters: dict[str, Any],
        task_type: str,
        sample_data: Data | None = None,
        device: torch.device | None = None,
        _skip_wrapper: bool = False,
    ) -> torch.nn.Module:
        """
        Create a model instance.

        PHASE 7 EXTENSION: Now supports:
        - name="custom" for custom architectures
        - name="ensemble" for ensemble models
        - All standard registry models (backward compatible)

        Args:
            name: Model name ("GCN", "GAT", "custom", "ensemble", etc.)
            hyperparameters: Hyperparameter dictionary
                For standard models: {"hidden_channels": 64, ...}
                For custom: {"architecture_config": {...}}
                For ensemble: {"ensemble_config": {...}}
            task_type: Task type (e.g., "graph_regression")
            sample_data: Sample data for channel inference
            device: Target device (CPU/CUDA/MPS)
            _skip_wrapper: Internal flag to skip graph/edge-level wrappers.
                          Used when creating models for ensembles where the
                          ensemble handles task-level wrapping/pooling.

        Returns:
            Instantiated model

        Raises:
            ModelNotFoundError: If model not found in registry
            ModelInstantiationError: If model creation fails
            HyperparameterError: If hyperparameters invalid

        Example:
            >>> # Standard model
            >>> model = factory.create_model("GCN", {...}, "graph_regression")
            >>>
            >>> # Custom architecture
            >>> model = factory.create_model(
            ...     name="custom",
            ...     hyperparameters={"architecture_config": config},
            ...     task_type="graph_regression"
            ... )
            >>>
            >>> # Ensemble
            >>> model = factory.create_model(
            ...     name="ensemble",
            ...     hyperparameters={"ensemble_config": config},
            ...     task_type="graph_regression"
            ... )
        """
        start_time = datetime.now()
        logger.info(f"Creating model: {name} for task: {task_type}")

        # =====================================================================
        # PHASE 7: HANDLE CUSTOM ARCHITECTURES
        # =====================================================================
        if name.lower() == "custom":
            if not _BUILDERS_AVAILABLE:
                raise ModelInstantiationError(
                    "Custom architectures not available. "
                    "Builders module not found. "
                    "Please ensure milia_pipeline.models.builders is installed.",
                    model_name="custom",
                )

            return self._create_custom_model(hyperparameters, task_type, sample_data, device)

        # =====================================================================
        # PHASE 7: HANDLE ENSEMBLE MODELS
        # =====================================================================
        if name.lower() == "ensemble":
            if not _BUILDERS_AVAILABLE:
                raise ModelInstantiationError(
                    "Ensemble models not available. "
                    "Builders module not found. "
                    "Please ensure milia_pipeline.models.builders is installed.",
                    model_name="ensemble",
                )

            # For classification ensembles, extract num_classes_override from
            # hyperparameters (injected by create_model_with_info) and pass to
            # _create_ensemble_model for propagation to individual models.
            #
            # For node/graph classification: uses 'out_channels' (models output class logits)
            # For edge_classification: uses 'num_classes' (models output embeddings,
            #                          wrapper decoder outputs num_classes)
            num_classes_override = hyperparameters.get("out_channels") or hyperparameters.get(
                "num_classes"
            )

            return self._create_ensemble_model(
                hyperparameters,
                task_type,
                sample_data,
                device,
                num_classes_override=num_classes_override,
            )

        # =====================================================================
        # STANDARD MODEL PATH (UNCHANGED - BACKWARD COMPATIBLE)
        # =====================================================================

        # Check if model exists in registry
        if not self.registry.has_model(name):
            available = self.registry.list_available_models()
            raise ModelNotFoundError(
                f"Model '{name}' not found in registry",
                model_name=name,
                available_models=available[:10],  # Show first 10
            )

        # Get model metadata
        try:
            metadata = get_model_metadata(name)
        except Exception as e:
            raise ModelInstantiationError(
                f"Failed to get metadata for model '{name}': {e}", model_name=name
            ) from e

        # Validate task compatibility
        if task_type not in metadata.supported_tasks:
            raise ModelValidationError(
                f"Model '{name}' does not support task '{task_type}'. "
                f"Supported tasks: {metadata.supported_tasks}",
                model_name=name,
            )

        # Process hyperparameters (inference + defaults)
        processed_params = self._process_hyperparameters(
            hyperparameters, metadata, sample_data, task_type
        )

        # Validate hyperparameters
        if metadata.hyperparameters:
            try:
                self.validator.validate_hyperparameters(processed_params, metadata.hyperparameters)
            except HyperparameterError as e:
                raise HyperparameterError(
                    f"Hyperparameter validation failed for model '{name}': {e}", model_name=name
                ) from e

        # Phase 4: Additional validation against actual model signature
        # This catches parameters that may have been introspected incorrectly
        # or that don't match the actual model constructor
        model_class = self.registry.get_model(name)
        if model_class is not None:
            try:
                is_valid, errors = validate_params_against_signature(model_class, processed_params)
                if not is_valid:
                    # Log as warning rather than error - introspected schema may be permissive
                    logger.warning(f"Signature validation warnings for '{name}': {errors}")
            except Exception as e:
                # Don't fail if signature validation itself fails
                logger.debug(f"Could not perform signature validation for '{name}': {e}")

        # Validate data compatibility if sample data provided
        if sample_data is not None:
            try:
                self.validator.validate_data_compatibility(sample_data, metadata)
            except DataCompatibilityError as e:
                logger.warning(f"Data compatibility warning for model '{name}': {e}")

        # Get model class from registry
        try:
            model_class = self.registry.get_model(name)
        except Exception as e:
            raise ModelInstantiationError(
                f"Failed to get model class for '{name}': {e}", model_name=name
            ) from e

        # =====================================================================
        # VALIDATE OPTIONAL DEPENDENCIES (DYNAMIC, PRODUCTION-READY)
        # =====================================================================
        # Check if model requires optional PyG packages (torch_cluster, etc.)
        # Raises clear error with installation instructions if missing
        # NOTE: Does NOT automatically install - user must run command manually
        # =====================================================================
        try:
            validate_model_dependencies(model_class, name)
        except ModelInstantiationError:
            # Re-raise dependency errors directly (they have detailed instructions)
            raise
        except Exception as e:
            # Log but don't fail for unexpected validation errors
            logger.debug(f"Could not validate dependencies for '{name}': {e}")

        # Instantiate model
        try:
            model = model_class(**processed_params)
        except Exception as e:
            raise ModelInstantiationError(
                f"Failed to instantiate model '{name}': {e}",
                model_name=name,
                hyperparameters=processed_params,
            ) from e

        # Wrap model for graph-level tasks
        # PyG BasicGNN models output node-level predictions [num_nodes, out_channels]
        # Graph-level tasks need [num_graphs, out_channels] via global pooling
        # Some models (SchNet, DimeNet) have fixed output dimensions and need projection
        #
        # CRITICAL: Skip wrapping when _skip_wrapper=True (used for ensemble models)
        # Ensembles handle graph-level pooling at the composition level, not per-model
        if self._is_graph_level_task(task_type) and not _skip_wrapper:
            pooling_method = processed_params.get("pooling_method", "mean")

            # Infer required out_channels for models that need output projection
            # This is needed for 3D models like SchNet that have fixed output (1 scalar)
            wrapper_out_channels = None

            # =================================================================
            # FIX 26: PRESERVE out_channels FOR WRAPPER FROM ORIGINAL HYPERPARAMETERS
            # =================================================================
            # PROBLEM: Models like SchNet don't have 'out_channels' in their schema
            #          (they output a fixed scalar), so out_channels gets filtered
            #          out during parameter processing. But the WRAPPER needs
            #          out_channels to create the output projection layer.
            #
            # SOLUTION: Check BOTH processed_params (for models that accept it)
            #           AND the original hyperparameters (for models that don't).
            #           The out_channels in hyperparameters represents the user's
            #           intended output dimensionality for multi-target prediction.
            #
            # DYNAMIC: Works for any model regardless of its parameter schema
            # PRODUCTION-READY: Preserves user-specified out_channels through wrapper
            # FUTURE-PROOF: Works for any future 3D model with fixed output dimensions
            # =================================================================

            # Try to get out_channels from processed_params (for models that accept it)
            if "out_channels" in processed_params:
                wrapper_out_channels = processed_params["out_channels"]
            # FIX 26: Check original hyperparameters for models that don't accept out_channels
            # (e.g., SchNet, DimeNet which have fixed scalar output)
            elif "out_channels" in hyperparameters:
                wrapper_out_channels = hyperparameters["out_channels"]
                logger.debug(
                    f"Using out_channels={wrapper_out_channels} from hyperparameters for wrapper "
                    f"(model '{metadata.name}' has fixed output dimensions)"
                )
            # Infer from sample_data for multi-target regression
            elif (
                sample_data is not None and hasattr(sample_data, "y") and sample_data.y is not None
            ):
                y = sample_data.y
                if y.dim() == 0:
                    wrapper_out_channels = 1
                elif y.dim() == 1:
                    wrapper_out_channels = y.size(0)
                else:
                    wrapper_out_channels = y.size(-1)

            model = GraphLevelModelWrapper(
                model, task_type, pooling_method, out_channels=wrapper_out_channels
            )
            logger.info(f"Wrapped model for graph-level task: {task_type}")

        # Wrap model for edge-level tasks
        # Edge-level tasks (link_prediction, edge_regression, edge_classification) need an edge decoder
        # to compute edge scores from node embeddings
        #
        # CRITICAL: Skip wrapping when _skip_wrapper=True (used for ensemble models)
        elif self._is_edge_level_task(task_type) and not _skip_wrapper:
            # Determine decoder method and output channels for edge_regression/edge_classification
            decoder_method = processed_params.get("decoder_method", "dot_product")
            edge_out_channels = None

            # For edge_regression, use MLP decoder with multi-dimensional output
            if task_type.lower() == "edge_regression":
                # Get edge_out_channels from hyperparameters or infer from sample_data
                edge_out_channels = processed_params.get("edge_out_channels")
                if (
                    edge_out_channels is None
                    and sample_data is not None
                    and hasattr(sample_data, "edge_attr")
                    and sample_data.edge_attr is not None
                ):
                    edge_out_channels = sample_data.edge_attr.shape[-1]
                    logger.info(
                        f"Inferred edge_out_channels={edge_out_channels} from sample_data.edge_attr"
                    )

                # Use MLP decoder for multi-dimensional edge regression
                if edge_out_channels is not None and edge_out_channels > 1:
                    decoder_method = processed_params.get("decoder_method", "hadamard_mlp")
                    logger.info(
                        f"Using {decoder_method} decoder for edge_regression with {edge_out_channels} output channels"
                    )

            # For edge_classification, use MLP decoder with num_classes output channels
            elif task_type.lower() == "edge_classification":
                # Get num_classes from hyperparameters (set by main.py after DiscretizeTargets)
                edge_out_channels = processed_params.get("out_channels") or processed_params.get(
                    "num_classes"
                )
                if edge_out_channels is None and num_classes_override is not None:
                    edge_out_channels = num_classes_override
                if edge_out_channels is None:
                    # Default to 10 classes if not specified (DiscretizeTargets default)
                    edge_out_channels = 10
                    logger.warning(
                        f"edge_classification: num_classes not specified, defaulting to {edge_out_channels}"
                    )

                # Use MLP decoder for classification (outputs class logits)
                decoder_method = processed_params.get("decoder_method", "hadamard_mlp")
                logger.info(
                    f"Using {decoder_method} decoder for edge_classification with {edge_out_channels} output classes"
                )

            model = EdgeLevelModelWrapper(model, task_type, decoder_method, edge_out_channels)
            logger.info(f"Wrapped model for edge-level task: {task_type}")

        # Move to device if specified
        if device is not None:
            try:
                model = model.to(device)
                logger.debug(f"Moved model to device: {device}")
            except Exception as e:
                logger.warning(f"Failed to move model to device {device}: {e}")

        # Log success
        elapsed = (datetime.now() - start_time).total_seconds()
        param_count = self._count_parameters(model)
        logger.info(
            f"Model created successfully: {name} ({param_count:,} parameters) in {elapsed:.2f}s"
        )

        return model

    def create_model_with_info(
        self,
        name: str,
        hyperparameters: dict[str, Any],
        task_type: str,
        sample_data: Data | None = None,
        device: torch.device | None = None,
        target_selection_config: dict[str, Any] | None = None,  # For regression target selection
        num_classes_override: int | None = None,  # For classification with discretized targets
    ) -> tuple[torch.nn.Module, dict[str, Any]]:
        """
        Create model and return it along with its metadata and configuration info.

        This method provides both the model and information about its capabilities,
        enabling intelligent forward pass handling based on model requirements.

        The returned model_info includes:
        - requires_edge_features: From metadata - model REQUIRES edge features
        - uses_edge_features: Computed - model is CONFIGURED to use edge features
          (True if requires_edge_features=True OR any edge dimension parameter is set)

        Edge dimension parameters are detected dynamically from the hyperparameter
        schema by looking for integer parameters with 'edge' and dimension-related
        keywords (dim, channels, features) in their names.

        Args:
            name: Model name (e.g., "GCN", "GAT")
            hyperparameters: Model hyperparameters
            task_type: Task type string
            sample_data: Sample data for inference (optional)
            device: Target device (optional)
            target_selection_config: Config for selecting subset of regression targets (optional)
            num_classes_override: Override out_channels for classification tasks (optional)
                When provided for classification tasks, this value is used as out_channels
                instead of inferring from sample_data. Used when targets are discretized
                and the number of classes is known from the discretization config.

        Returns:
            Tuple of (model, model_info) where model_info contains:
            - requires_edge_features: Whether model requires edge features (from metadata)
            - uses_edge_features: Whether model is configured to use edge features
            - detected_edge_params: List of detected edge dimension parameters
            - supported_tasks: List of supported task types
            - Other metadata fields

        Example:
            >>> factory = ModelFactory()
            >>> model, info = factory.create_model_with_info(
            ...     name="GCN",
            ...     hyperparameters={"hidden_channels": 64},
            ...     task_type="graph_regression"
            ... )
            >>> print(f"Model uses edge features: {info['uses_edge_features']}")
            False
            >>>
            >>> # GAT with edge_dim configured
            >>> model, info = factory.create_model_with_info(
            ...     name="GAT",
            ...     hyperparameters={"hidden_channels": 64, "edge_dim": 16},
            ...     task_type="graph_regression"
            ... )
            >>> print(f"Model uses edge features: {info['uses_edge_features']}")
            True
            >>>
            >>> # GeneralConv with in_edge_channels configured
            >>> model, info = factory.create_model_with_info(
            ...     name="GeneralConv",
            ...     hyperparameters={"in_channels": 64, "out_channels": 32, "in_edge_channels": 8},
            ...     task_type="node_regression"
            ... )
            >>> print(f"Model uses edge features: {info['uses_edge_features']}")
            True
            >>>
            >>> # Classification with discretized targets (num_classes_override)
            >>> model, info = factory.create_model_with_info(
            ...     name="GCN",
            ...     hyperparameters={"hidden_channels": 64},
            ...     task_type="graph_classification",
            ...     num_classes_override=10  # From DiscretizeTargets n_bins
            ... )
            >>> print(f"out_channels: {info['out_channels']}")
            10
        """
        # =====================================================================
        # CLASSIFICATION FIX: Inject num_classes_override into hyperparameters
        # BEFORE model creation so the model is built with correct output dim
        # =====================================================================
        is_classification = "classification" in task_type.lower()

        if num_classes_override is not None and is_classification:
            # Create a copy to avoid modifying the original
            hyperparameters = dict(hyperparameters)
            hyperparameters["out_channels"] = num_classes_override
            logger.info(
                f"Classification: Injecting out_channels={num_classes_override} "
                f"from num_classes_override into hyperparameters"
            )

        # =====================================================================
        # REGRESSION TARGET SELECTION FIX: Pre-resolve and inject out_channels
        # BEFORE model creation so ensembles are built with correct output dim
        # =====================================================================
        # For regression with target_selection, the selected out_channels must
        # be known BEFORE model creation. Otherwise, ensembles will be built
        # with the full target dimension (e.g., 8) instead of selected (e.g., 1).
        #
        # DYNAMIC: Works with any selection specification (indices or names)
        # PRODUCTION-READY: Validates against actual dataset
        # FUTURE-PROOF: Same pattern as classification num_classes_override
        # =====================================================================
        pre_resolved_target_selection = None
        selected_out_channels = None
        original_out_channels_for_ts = None

        if target_selection_config is not None and not is_classification:
            from .target_selection_config import TargetSelectionConfig

            # Parse target selection config
            ts_config = TargetSelectionConfig.from_config(target_selection_config)

            if ts_config.resolved_indices is not None:
                # Already resolved by data preparation
                selected_out_channels = len(ts_config.resolved_indices)
                original_out_channels_for_ts = ts_config.total_available
                pre_resolved_target_selection = ts_config.to_dict()
                logger.debug(
                    f"Target selection pre-resolved (by data preparation): "
                    f"out_channels={selected_out_channels}, original={original_out_channels_for_ts}"
                )
            else:
                # Pre-resolve now so we know out_channels before model creation
                available_names = None
                total_count = None

                if sample_data is not None:
                    # ================================================================
                    # FIX 21: Use correct source attribute based on target_source config
                    # ================================================================
                    # BEFORE: Always used y.shape regardless of target_source
                    # AFTER: Use shape of the configured source attribute
                    #
                    # This fixes node_regression with target_source: "x" where the
                    # indices should be validated against x.shape, not y.shape
                    # ================================================================
                    config_source = (
                        ts_config.config_source.lower() if ts_config.config_source else "auto"
                    )

                    # Determine which attribute to get property names from
                    if (
                        config_source == "x"
                        and hasattr(sample_data, "x_property_names")
                        and sample_data.x_property_names is not None
                    ):
                        available_names = list(sample_data.x_property_names)
                        logger.debug(f"Found x_property_names in sample_data: {available_names}")
                    elif (
                        config_source == "edge_attr"
                        and hasattr(sample_data, "edge_attr_property_names")
                        and sample_data.edge_attr_property_names is not None
                    ):
                        available_names = list(sample_data.edge_attr_property_names)
                        logger.debug(
                            f"Found edge_attr_property_names in sample_data: {available_names}"
                        )
                    elif (
                        hasattr(sample_data, "y_property_names")
                        and sample_data.y_property_names is not None
                    ):
                        available_names = list(sample_data.y_property_names)
                        logger.debug(f"Found y_property_names in sample_data: {available_names}")

                    # Determine total_count from the correct source attribute
                    if config_source == "x":
                        if hasattr(sample_data, "x") and sample_data.x is not None:
                            x_shape = sample_data.x.shape
                            if len(x_shape) > 0:
                                total_count = x_shape[-1] if len(x_shape) > 1 else x_shape[0]
                                logger.debug(
                                    f"Using x.shape for total_count: {total_count} (config_source='x')"
                                )
                    elif config_source == "edge_attr":
                        if hasattr(sample_data, "edge_attr") and sample_data.edge_attr is not None:
                            ea_shape = sample_data.edge_attr.shape
                            if len(ea_shape) > 0:
                                total_count = ea_shape[-1] if len(ea_shape) > 1 else ea_shape[0]
                                logger.debug(
                                    f"Using edge_attr.shape for total_count: {total_count} (config_source='edge_attr')"
                                )
                    else:
                        # Default: use y (standard PyG convention)
                        if hasattr(sample_data, "y") and sample_data.y is not None:
                            y_shape = sample_data.y.shape
                            if len(y_shape) > 0:
                                total_count = y_shape[-1] if len(y_shape) > 1 else y_shape[0]
                                logger.debug(f"Using y.shape for total_count: {total_count}")

                if total_count is not None:
                    ts_config.resolve(available_names, total_count)
                    selected_out_channels = len(ts_config.resolved_indices)
                    original_out_channels_for_ts = total_count
                    pre_resolved_target_selection = ts_config.to_dict()
                    logger.info(
                        f"Target selection pre-resolved: out_channels {original_out_channels_for_ts} -> {selected_out_channels} "
                        f"[selected: {ts_config.resolved_names or ts_config.resolved_indices}]"
                    )

            # Inject selected out_channels into hyperparameters for model creation
            if selected_out_channels is not None:
                hyperparameters = dict(hyperparameters)
                hyperparameters["out_channels"] = selected_out_channels
                logger.debug(
                    f"Regression target selection: Injecting out_channels={selected_out_channels} "
                    f"into hyperparameters for model creation"
                )

        # Create the model (now with correct out_channels for classification OR target selection)
        model = self.create_model(
            name=name,
            hyperparameters=hyperparameters,
            task_type=task_type,
            sample_data=sample_data,
            device=device,
        )

        # Get model info/metadata
        model_info = self.get_model_info(name)

        # If model_info is None (shouldn't happen for registered models),
        # provide safe defaults
        if model_info is None:
            logger.warning(f"No metadata found for model '{name}', using defaults")
            model_info = {
                "name": name,
                "requires_edge_features": False,
                "requires_edge_weights": False,
                "supported_tasks": [],
                "hyperparameters": {},
            }

        # Determine if model is configured to use edge features
        requires_edge_features = model_info.get("requires_edge_features", False)

        # Dynamically detect edge dimension parameters from schema
        schema = model_info.get("hyperparameters", {})
        detected_edge_params = self._detect_edge_feature_params(schema)

        # Check if any edge dimension parameter was configured (non-None value)
        edge_dim_configured = any(
            hyperparameters.get(param) is not None for param in detected_edge_params
        )

        # Model uses edge features if it requires them OR if edge dims are configured
        model_info["uses_edge_features"] = requires_edge_features or edge_dim_configured
        model_info["detected_edge_params"] = detected_edge_params

        if model_info["uses_edge_features"]:
            configured_params = [
                f"{p}={hyperparameters.get(p)}"
                for p in detected_edge_params
                if hyperparameters.get(p) is not None
            ]
            logger.debug(
                f"Model '{name}' configured to use edge features "
                f"(requires={requires_edge_features}, configured=[{', '.join(configured_params)}])"
            )

        # Add task_type to model_info for Trainer to use
        # This enables intelligent target selection (batch.y vs edge_label vs edge_value)
        model_info["task_type"] = task_type

        # =====================================================================
        # Add out_channels to model_info for Trainer to use
        # This enables dynamic target reshaping for graph-level multi-target tasks
        # where PyG batching flattens y from [batch, targets] to [batch*targets]
        #
        # Strategy (in order of preference):
        # 0. CLASSIFICATION OVERRIDE: Use num_classes_override if provided for classification
        # 1. Read from model itself (single source of truth - model was just built with this value)
        # 2. Read from user's explicit hyperparameters
        # 3. Infer from sample_data (same logic as model construction)
        # 4. Default to 1 (safe - single target is most common, and no reshape needed)
        #
        # Note: is_classification was already determined above before model creation
        # =====================================================================
        out_channels = None

        # Strategy 0: CLASSIFICATION OVERRIDE - Use num_classes when targets are discretized
        # This takes highest priority for classification tasks because:
        # - The original data shape (e.g., y=[8] for 8 regression targets) is misleading
        # - After discretization, we need out_channels = n_bins (num_classes)
        # - This value comes from DiscretizeTargets config and is known before model creation
        if num_classes_override is not None and is_classification:
            out_channels = num_classes_override
            logger.debug(
                f"Classification: Using num_classes_override={num_classes_override} as out_channels"
            )

        # Strategy 1: Try to read from the model itself (most reliable for non-override cases)
        if out_channels is None:
            actual_model = model.model if hasattr(model, "model") else model
            if hasattr(actual_model, "out_channels") and actual_model.out_channels is not None:
                out_channels = actual_model.out_channels
                logger.debug(f"Read out_channels={out_channels} from model attribute")

        # Strategy 2: Check user's explicit hyperparameters
        if out_channels is None:
            explicit_out = hyperparameters.get("out_channels")
            if explicit_out is not None:
                out_channels = explicit_out
                logger.debug(f"Using explicit out_channels={out_channels} from hyperparameters")

        # Strategy 3: Infer from sample_data
        if out_channels is None and sample_data is not None and infer_out_channels is not None:
            inferred = infer_out_channels(data=sample_data, task_type=task_type, default=None)
            if inferred is not None:
                out_channels = inferred
                logger.debug(f"Inferred out_channels={out_channels} from sample_data")

        # Strategy 4: Safe default
        if out_channels is None:
            out_channels = 1
            logger.debug("Using default out_channels=1")

        # ====================================================================
        # TARGET SELECTION RESOLUTION
        # ====================================================================
        # DYNAMIC: Works with any selection specification
        # PRODUCTION-READY: Validates against actual dataset
        # FUTURE-PROOF: Extensible selection modes via TargetSelectionConfig
        # ====================================================================
        target_selection = None
        original_out_channels = out_channels

        # Use pre-resolved target selection if available (from pre-model-creation resolution)
        # This ensures consistency between model out_channels and model_info
        if pre_resolved_target_selection is not None:
            target_selection = pre_resolved_target_selection
            out_channels = selected_out_channels
            original_out_channels = original_out_channels_for_ts or out_channels
            logger.debug(
                f"Using pre-resolved target selection: out_channels={out_channels}, "
                f"original={original_out_channels}"
            )
        elif target_selection_config is not None and not is_classification:
            # ================================================================
            # SKIP for classification tasks: target_selection applies to
            # regression targets (multi-column), not class indices
            # Classification uses num_classes_override from discretization
            # ================================================================
            from .target_selection_config import TargetSelectionConfig

            # Parse target selection config
            ts_config = TargetSelectionConfig.from_config(target_selection_config)

            # ================================================================
            # CRITICAL: Skip resolution if already resolved by data preparation
            # ================================================================
            # Data preparation functions (e.g., _prepare_node_regression_data_hpo)
            # may have already extracted targets using the same indices.
            # In that case, resolved_indices is already populated and the new y
            # tensor has the reduced shape. Re-resolving would fail because the
            # original indices (e.g., [5, 6]) are invalid for the new shape.
            # ================================================================
            if ts_config.resolved_indices is not None:
                # Already resolved by data preparation - use existing resolution
                out_channels = len(ts_config.resolved_indices)
                target_selection = ts_config.to_dict()

                # Use total_available from config as original_out_channels
                # This is the original source tensor column count before extraction
                original_out_channels = ts_config.total_available or original_out_channels

                logger.info(
                    f"Target selection already resolved (by data preparation): "
                    f"out_channels={out_channels}, original={original_out_channels} "
                    f"[indices: {ts_config.resolved_indices}]"
                )
            else:
                # Not yet resolved - resolve against actual data
                # Get property names from sample data if available
                available_names = None
                total_count = out_channels  # Default to inferred out_channels

                if sample_data is not None:
                    # ================================================================
                    # FIX 21b: Use correct source attribute based on target_source config
                    # (Same logic as Fix 21 in pre-resolution block)
                    # ================================================================
                    config_source = (
                        ts_config.config_source.lower() if ts_config.config_source else "auto"
                    )

                    # Determine which attribute to get property names from
                    if (
                        config_source == "x"
                        and hasattr(sample_data, "x_property_names")
                        and sample_data.x_property_names is not None
                    ):
                        available_names = list(sample_data.x_property_names)
                        logger.debug(f"Found x_property_names in sample_data: {available_names}")
                    elif (
                        config_source == "edge_attr"
                        and hasattr(sample_data, "edge_attr_property_names")
                        and sample_data.edge_attr_property_names is not None
                    ):
                        available_names = list(sample_data.edge_attr_property_names)
                        logger.debug(
                            f"Found edge_attr_property_names in sample_data: {available_names}"
                        )
                    elif (
                        hasattr(sample_data, "y_property_names")
                        and sample_data.y_property_names is not None
                    ):
                        available_names = list(sample_data.y_property_names)
                        logger.debug(f"Found y_property_names in sample_data: {available_names}")

                    # Determine total_count from the correct source attribute
                    if config_source == "x":
                        if hasattr(sample_data, "x") and sample_data.x is not None:
                            x_shape = sample_data.x.shape
                            if len(x_shape) > 0:
                                total_count = x_shape[-1] if len(x_shape) > 1 else x_shape[0]
                                logger.debug(
                                    f"Using x.shape for total_count: {total_count} (config_source='x')"
                                )
                    elif config_source == "edge_attr":
                        if hasattr(sample_data, "edge_attr") and sample_data.edge_attr is not None:
                            ea_shape = sample_data.edge_attr.shape
                            if len(ea_shape) > 0:
                                total_count = ea_shape[-1] if len(ea_shape) > 1 else ea_shape[0]
                                logger.debug(
                                    f"Using edge_attr.shape for total_count: {total_count} (config_source='edge_attr')"
                                )
                    else:
                        # Default: use y (standard PyG convention)
                        if hasattr(sample_data, "y") and sample_data.y is not None:
                            y_shape = sample_data.y.shape
                            if len(y_shape) > 0:
                                total_count = y_shape[-1] if len(y_shape) > 1 else y_shape[0]
                                logger.debug(f"Using y.shape for total_count: {total_count}")

                # Resolve selection against actual data
                ts_config.resolve(available_names, total_count)

                # Adjust out_channels based on selection
                out_channels = len(ts_config.resolved_indices)
                target_selection = ts_config.to_dict()

                logger.info(
                    f"Target selection applied: out_channels {original_out_channels} -> {out_channels} "
                    f"[selected: {ts_config.resolved_names or ts_config.resolved_indices}]"
                )

        model_info["out_channels"] = out_channels
        model_info["target_selection"] = target_selection  # NEW: Add to model_info
        model_info["original_out_channels"] = (
            original_out_channels if target_selection else None
        )  # NEW
        model_info["num_classes_override"] = (
            num_classes_override  # For classification with discretization
        )
        model_info["is_classification"] = is_classification

        # =====================================================================
        # CHECKPOINT SUPPORT: Store actual hyperparameter VALUES
        # =====================================================================
        # CRITICAL: model_info['hyperparameters'] already contains the SCHEMA
        # from get_model_info(). We store actual VALUES under a different key
        # to avoid breaking code that expects the schema.
        #
        # DYNAMIC: Stores hyperparameters dict PLUS inferred values from model
        # PRODUCTION-READY: Enables model recreation from checkpoints
        # FUTURE-PROOF: Works with any model and any hyperparameter set
        # =====================================================================
        # CRITICAL FIX: Extract PROCESSED hyperparameters including inferred values
        # The original hyperparameters dict may not contain in_channels/out_channels
        # which are often inferred from sample_data during model creation.
        # We need to extract these from the created model to ensure checkpoint
        # can recreate the model with correct dimensions.
        # =====================================================================
        processed_hyperparams = dict(hyperparameters)  # Start with original

        # =====================================================================
        # CRITICAL: Extract hyperparameters from created model for checkpoint
        # =====================================================================
        # The model may have inferred values (in_channels, out_channels) from
        # sample_data that are NOT in the original hyperparameters dict.
        # We MUST extract these for checkpoint to recreate model correctly.
        #
        # DYNAMIC: Works with ANY model - wrapped or unwrapped, standard or custom
        # PRODUCTION-READY: Falls back gracefully if attributes not available
        # FUTURE-PROOF: Uses hasattr() pattern, compatible with any PyG model
        # =====================================================================

        # Unwrap model to get to the actual model with hyperparameters
        # Handle multiple levels of wrapping (GraphLevelModelWrapper, etc.)
        actual_model = model
        unwrap_depth = 0
        while hasattr(actual_model, "model") and unwrap_depth < 10:
            actual_model = actual_model.model
            unwrap_depth += 1

        if unwrap_depth > 0:
            logger.debug(f"Unwrapped model {unwrap_depth} level(s) to access hyperparameters")

        # Track what we extracted for logging
        extracted_params = {}

        # Extract in_channels from created model (CRITICAL for checkpoint recreation)
        if (
            hasattr(actual_model, "in_channels")
            and actual_model.in_channels is not None
            and (
                "in_channels" not in processed_hyperparams
                or processed_hyperparams.get("in_channels") != actual_model.in_channels
            )
        ):
            # Always store in_channels if available, even if already in hyperparams
            # This ensures we capture the ACTUAL value used in the model
            processed_hyperparams["in_channels"] = actual_model.in_channels
            extracted_params["in_channels"] = actual_model.in_channels

        # Extract out_channels from created model
        if (
            hasattr(actual_model, "out_channels")
            and actual_model.out_channels is not None
            and (
                "out_channels" not in processed_hyperparams
                or processed_hyperparams.get("out_channels") != actual_model.out_channels
            )
        ):
            processed_hyperparams["out_channels"] = actual_model.out_channels
            extracted_params["out_channels"] = actual_model.out_channels

        # Extract hidden_channels if available
        if (
            hasattr(actual_model, "hidden_channels")
            and actual_model.hidden_channels is not None
            and (
                "hidden_channels" not in processed_hyperparams
                or processed_hyperparams.get("hidden_channels")
                != actual_model.hidden_channels
            )
        ):
            processed_hyperparams["hidden_channels"] = actual_model.hidden_channels
            extracted_params["hidden_channels"] = actual_model.hidden_channels

        # Extract num_layers if available
        if hasattr(actual_model, "num_layers") and actual_model.num_layers is not None and (
            "num_layers" not in processed_hyperparams
            or processed_hyperparams.get("num_layers") != actual_model.num_layers
        ):
            processed_hyperparams["num_layers"] = actual_model.num_layers
            extracted_params["num_layers"] = actual_model.num_layers

        # =====================================================================
        # FALLBACK: If in_channels still missing, try to get from sample_data
        # This handles cases where model doesn't expose in_channels attribute
        # (e.g., custom models, ensembles, or models using lazy initialization)
        # =====================================================================
        if (
            "in_channels" not in processed_hyperparams
            and sample_data is not None
            and hasattr(sample_data, "x")
            and sample_data.x is not None
        ):
            inferred_in_channels = sample_data.x.size(-1)
            processed_hyperparams["in_channels"] = inferred_in_channels
            extracted_params["in_channels"] = f"{inferred_in_channels} (from sample_data)"
            logger.info(
                f"Inferred in_channels={inferred_in_channels} from sample_data "
                f"(model does not expose in_channels attribute)"
            )

        # Log extracted parameters at INFO level for visibility
        if extracted_params:
            param_str = ", ".join(f"{k}={v}" for k, v in extracted_params.items())
            logger.info(f"Extracted model hyperparameters for checkpoint: {param_str}")

        # Warn if critical parameters are still missing
        if "in_channels" not in processed_hyperparams:
            logger.warning(
                "CHECKPOINT WARNING: 'in_channels' not available in hyperparameters. "
                "Model may not recreate correctly from checkpoint. "
                "Ensure sample_data is provided during model creation."
            )

        model_info["hyperparameters_values"] = processed_hyperparams

        return model, model_info

    @staticmethod
    def _detect_edge_feature_params(hyperparameter_schema: dict[str, Any]) -> list[str]:
        """
        Dynamically detect edge feature dimension parameters from schema.

        Scans the hyperparameter schema to find parameters that control edge
        feature dimensions. This is done by pattern matching on parameter names
        rather than hardcoding specific names, making it future-proof for new models.

        Detection criteria:
        - Parameter name contains 'edge' AND contains dimension keywords
          (dim, channels, features, size)
        - Parameter type is 'integer' (dimension parameters are integers)

        Args:
            hyperparameter_schema: Model's hyperparameter schema from metadata

        Returns:
            List of parameter names that control edge feature dimensions

        Examples:
            Detected parameters include:
            - edge_dim (GATConv, TransformerConv)
            - in_edge_channels (GeneralConv)
            - edge_feature_dim (future models)
            - num_edge_features (future models)
        """
        edge_dim_params = []

        # Keywords that indicate dimension/size parameters
        dimension_keywords = ("dim", "channels", "features", "size")

        for param_name, param_spec in hyperparameter_schema.items():
            param_lower = param_name.lower()

            # Must contain 'edge' to be edge-related
            if "edge" not in param_lower:
                continue

            # Must contain a dimension-related keyword
            has_dimension_keyword = any(kw in param_lower for kw in dimension_keywords)
            if not has_dimension_keyword:
                continue

            # Must be integer type (dimensions are integers)
            if param_spec.get("type") != "integer":
                continue

            edge_dim_params.append(param_name)

        return edge_dim_params

    # =========================================================================
    # PHASE 7: CUSTOM ARCHITECTURE CREATION
    # =========================================================================

    def _create_custom_model(
        self,
        hyperparameters: dict[str, Any],
        task_type: str,
        sample_data: Data | None,
        device: torch.device | None,
    ) -> torch.nn.Module:
        """
        Create custom architecture using ArchitectureBuilder.

        Args:
            hyperparameters: Must contain "architecture_config"
            task_type: Task type
            sample_data: Sample data for channel inference
            device: Target device

        Returns:
            Built custom architecture

        Raises:
            ModelInstantiationError: If creation fails
        """
        logger.info("Creating custom architecture")

        # Extract architecture config
        if "architecture_config" not in hyperparameters:
            raise ModelInstantiationError(
                "Custom architecture requires 'architecture_config' in hyperparameters",
                model_name="custom",
                details="Provide config dictionary with 'layers' field",
            )

        config = hyperparameters["architecture_config"]

        # Infer channels from sample data if not in config
        if isinstance(config, dict):
            if (
                "in_channels" not in config
                and sample_data is not None
                and hasattr(sample_data, "x")
                and sample_data.x is not None
            ):
                config["in_channels"] = sample_data.x.size(-1)
                logger.debug(f"Inferred in_channels={config['in_channels']} from sample data")

            if "out_channels" not in config:
                if infer_out_channels is not None and sample_data is not None:
                    out_channels = infer_out_channels(
                        data=sample_data, task_type=task_type, default=None
                    )
                    if out_channels is not None:
                        config["out_channels"] = out_channels
                        logger.debug(f"Inferred out_channels={out_channels} for custom model")
                else:
                    # Fallback if infer_out_channels not available
                    if sample_data is not None:
                        if "regression" in task_type.lower():
                            config["out_channels"] = 1
                        elif "classification" in task_type.lower() and hasattr(sample_data, "y"):
                            y = sample_data.y
                            if y.dim() > 1:
                                config["out_channels"] = y.size(-1)
                            else:
                                config["out_channels"] = len(torch.unique(y))

            if "task_type" not in config:
                config["task_type"] = task_type

        # Parse config - this returns an ArchitectureBuilder with layers already added
        try:
            builder = parse_custom_architecture(config, task_type=task_type, validate=True)
        except Exception as e:
            raise ModelInstantiationError(
                f"Failed to parse architecture config: {e}", model_name="custom"
            ) from e

        # Build model
        try:
            model = builder.build()
        except Exception as e:
            raise ModelInstantiationError(
                f"Failed to build custom architecture: {e}", model_name="custom"
            ) from e

        # Move to device
        if device is not None:
            model = model.to(device)

        param_count = self._count_parameters(model)
        logger.info(f"Custom architecture created ({param_count:,} parameters)")

        return model

    # =========================================================================
    # PHASE 7: ENSEMBLE MODEL CREATION
    # =========================================================================

    def _create_ensemble_model(
        self,
        hyperparameters: dict[str, Any],
        task_type: str,
        sample_data: Data | None,
        device: torch.device | None,
        num_classes_override: int | None = None,
    ) -> torch.nn.Module:
        """
        Create ensemble using ModelComposer.

        Args:
            hyperparameters: Must contain "ensemble_config"
            task_type: Task type
            sample_data: Sample data for channel inference
            device: Target device
            num_classes_override: Override out_channels for classification tasks.
                When provided for classification, this value is propagated to
                all individual models in the ensemble to ensure correct output
                dimensions (num_classes instead of 1).

        Returns:
            Built ensemble model

        Raises:
            ModelInstantiationError: If creation fails
        """
        logger.info("Creating ensemble model")

        # Extract ensemble config
        if "ensemble_config" not in hyperparameters:
            raise ModelInstantiationError(
                "Ensemble requires 'ensemble_config' in hyperparameters",
                model_name="ensemble",
                details="Provide config dictionary with 'models' field",
            )

        config = hyperparameters["ensemble_config"]

        # Parse config - returns ModelComposer with strategy/fusion set
        try:
            composer = parse_ensemble(config, task_type=task_type, validate=True)
        except Exception as e:
            raise ModelInstantiationError(
                f"Failed to parse ensemble config: {e}", model_name="ensemble"
            ) from e

        # Add models to ensemble from config
        if "models" in config and isinstance(config, dict):
            try:
                # =====================================================================
                # HIERARCHICAL AND SEQUENTIAL DIMENSION CHAINING
                # =====================================================================
                # For hierarchical ensembles, models at level > 0 receive the OUTPUT
                # of level 0 models as their INPUT.
                #
                # For sequential ensembles, models are chained: output of model N
                # becomes input of model N+1. Each model after the first needs
                # in_channels = previous model's out_channels.
                #
                # - Level 0 / First model: in_channels = sample_data.x features
                # - Level N / Model N>0: in_channels = out_channels of previous
                #
                # DYNAMIC: Automatically detects strategy and chains dimensions
                # PRODUCTION-READY: Handles any number of levels/models
                # FUTURE-PROOF: Works with any model that has in_channels/out_channels
                # =====================================================================

                strategy = config.get("strategy", "parallel")
                is_hierarchical = strategy.lower() == "hierarchical"
                is_sequential = strategy.lower() == "sequential"
                needs_dimension_chaining = is_hierarchical or is_sequential

                # Group models by level for hierarchical processing
                # For sequential, all models are at level 0 but processed in order
                models_by_level = {}
                for model_spec in config["models"]:
                    level = model_spec.get("level", 0)
                    if level not in models_by_level:
                        models_by_level[level] = []
                    models_by_level[level].append(model_spec)

                # Track output dimension of previous model/level (for dimension chaining)
                prev_out_channels = None

                # ================================================================
                # FIX 23: PROPAGATE in_channels/out_channels FROM CHECKPOINT
                # ================================================================
                # PROBLEM: When loading ensemble from checkpoint, top-level hyperparameters
                #          contain in_channels/out_channels from training, but individual
                #          model specs have empty hyperparameters {}. Without sample_data,
                #          create_model() cannot infer dimensions, causing shape mismatch.
                #
                # SOLUTION: Extract in_channels/out_channels from top-level hyperparameters
                #           and propagate to individual models when not already specified.
                #
                # DYNAMIC: Only propagates if values exist and models don't override
                # PRODUCTION-READY: Enables checkpoint loading for any ensemble
                # FUTURE-PROOF: Works with any model that accepts in_channels/out_channels
                # ================================================================
                ensemble_in_channels = hyperparameters.get("in_channels")
                ensemble_out_channels = hyperparameters.get("out_channels")

                if ensemble_in_channels is not None:
                    logger.debug(
                        f"Ensemble checkpoint: propagating in_channels={ensemble_in_channels} "
                        f"to individual models"
                    )
                if ensemble_out_channels is not None:
                    logger.debug(
                        f"Ensemble checkpoint: propagating out_channels={ensemble_out_channels} "
                        f"to individual models"
                    )

                # Process levels in order
                for level in sorted(models_by_level.keys()):
                    for model_spec in models_by_level[level]:
                        model_name = model_spec.get("name")
                        model_hparams = model_spec.get("hyperparameters", {}).copy()
                        weight = model_spec.get("weight", 1.0)

                        # ================================================================
                        # FIX 23 (Part 2): PROPAGATE CHECKPOINT DIMENSIONS TO INDIVIDUAL MODELS
                        # ================================================================
                        # For PARALLEL ensembles (level 0, no chaining), propagate in_channels
                        # and out_channels from top-level hyperparameters to individual models
                        # when the model spec doesn't already specify them.
                        #
                        # This is CRITICAL for checkpoint loading where:
                        # - Top-level hyperparameters have in_channels=18, out_channels=8
                        # - Model specs have empty hyperparameters {}
                        # - sample_data is None (no data to infer from)
                        #
                        # DYNAMIC: Only applies when values exist and not already specified
                        # PRODUCTION-READY: Enables ensemble checkpoint loading
                        # FUTURE-PROOF: Works with any ensemble strategy
                        # ================================================================

                        # Propagate in_channels from checkpoint to level 0 models (parallel input)
                        # For hierarchical/sequential, dimension chaining handles this below
                        if (
                            level == 0
                            and ensemble_in_channels is not None
                            and "in_channels" not in model_hparams
                        ):
                            model_hparams["in_channels"] = ensemble_in_channels
                            logger.debug(
                                f"Checkpoint propagation: {model_name} "
                                f"in_channels set to {ensemble_in_channels}"
                            )

                        # Propagate out_channels from checkpoint to ALL models
                        # (unless classification override or model spec specifies it)
                        # This ensures all parallel models produce the same output dimension
                        if (
                            ensemble_out_channels is not None
                            and "out_channels" not in model_hparams
                        ):
                            model_hparams["out_channels"] = ensemble_out_channels
                            logger.debug(
                                f"Checkpoint propagation: {model_name} "
                                f"out_channels set to {ensemble_out_channels}"
                            )

                        # ================================================================
                        # CLASSIFICATION FIX: Inject num_classes_override into model hparams
                        # ================================================================
                        # For node/graph classification ensembles, each individual model needs
                        # the correct out_channels (num_classes) to produce the right output
                        # shape [batch, num_classes] for voting/attention fusion.
                        #
                        # EXCEPTION: For edge_classification, models should output node EMBEDDINGS
                        # (not class logits). The EdgeLevelModelWrapper handles combining node
                        # embeddings into edge predictions, and its MLP decoder outputs num_classes.
                        #
                        # Without this distinction:
                        # - node/graph classification would infer out_channels=1 (wrong)
                        # - edge_classification would get out_channels=num_classes in GNN (wrong)
                        # ================================================================
                        is_classification = "classification" in task_type.lower()
                        is_edge_classification = task_type.lower() == "edge_classification"

                        # For node/graph classification: inject num_classes
                        # For edge_classification: DON'T inject - models output embeddings, not classes
                        if (
                            num_classes_override is not None
                            and is_classification
                            and not is_edge_classification
                        ) and "out_channels" not in model_hparams:
                            model_hparams["out_channels"] = num_classes_override
                            logger.debug(
                                f"Classification ensemble: {model_name} "
                                f"out_channels set to {num_classes_override} (num_classes)"
                            )

                        # Determine if this model needs dimension chaining
                        # - Hierarchical: level > 0 gets chained from previous level
                        # - Sequential: all models after the first get chained
                        needs_chain = False
                        if is_hierarchical and level > 0 and prev_out_channels is not None:
                            needs_chain = True
                        elif is_sequential and prev_out_channels is not None:
                            # In sequential, every model after the first needs chaining
                            needs_chain = True

                        if needs_chain and "in_channels" not in model_hparams:
                            model_hparams["in_channels"] = prev_out_channels
                            logger.debug(
                                f"Dimension chaining ({strategy}): {model_name} "
                                f"in_channels set to {prev_out_channels}"
                            )

                        # Create individual model WITHOUT task-level wrapping
                        # CRITICAL: Pass _skip_wrapper=True so individual models are NOT
                        # wrapped with GraphLevelModelWrapper or EdgeLevelModelWrapper.
                        # The ensemble (ParallelEnsemble, HierarchicalComposition) handles
                        # graph-level pooling at the composition level, not per-model.
                        # This prevents the issue where each model pools prematurely,
                        # causing node/edge index mismatches in hierarchical compositions.
                        individual_model = self.create_model(
                            name=model_name,
                            hyperparameters=model_hparams,
                            task_type=task_type,
                            sample_data=sample_data,
                            device=device,
                            _skip_wrapper=True,
                        )

                        composer.add_model(individual_model, weight=weight, level=level)

                        # Track output dimension for dimension chaining
                        # Applies to both hierarchical (between levels) and sequential (between models)
                        if needs_dimension_chaining:
                            model_out = None
                            actual_model = (
                                individual_model.model
                                if hasattr(individual_model, "model")
                                else individual_model
                            )

                            # Try common attribute names for output dimension
                            for attr in ["out_channels", "hidden_channels"]:
                                if hasattr(actual_model, attr):
                                    model_out = getattr(actual_model, attr)
                                    if model_out is not None:
                                        break

                            # Also check hyperparameters
                            if model_out is None:
                                model_out = model_hparams.get("out_channels") or model_hparams.get(
                                    "hidden_channels"
                                )

                            if model_out is not None:
                                # For sequential: every model's output feeds next model
                                # For hierarchical: each level's output feeds next level
                                prev_out_channels = model_out

            except Exception as e:
                raise ModelInstantiationError(
                    f"Failed to add models to ensemble: {e}", model_name="ensemble"
                ) from e

        # Validate composition
        try:
            validation = composer.validate_composition()
            if not validation["valid"]:
                logger.warning(f"Ensemble validation warnings: {validation.get('warnings', [])}")
                if validation.get("errors"):
                    raise ModelInstantiationError(
                        f"Ensemble validation failed: {validation['errors']}", model_name="ensemble"
                    )
        except Exception as e:
            logger.warning(f"Ensemble validation error: {e}")

        # Build ensemble
        try:
            model = composer.build()
        except Exception as e:
            raise ModelInstantiationError(f"Failed to build ensemble: {e}", model_name="ensemble") from e

        # =====================================================================
        # WRAP ENSEMBLE FOR GRAPH-LEVEL TASKS
        # =====================================================================
        # Individual models were created with _skip_wrapper=True, so they produce
        # node embeddings [num_nodes, hidden_channels]. For graph-level tasks
        # (graph_regression, graph_classification), the ENSEMBLE needs to be
        # wrapped with GraphLevelModelWrapper to:
        # 1. Apply global pooling to get [num_graphs, hidden_channels]
        # 2. Project to correct out_channels if target_selection is applied
        #
        # CRITICAL: For multi-target regression with target_selection, the
        # ensemble must output [num_graphs, selected_out_channels], not
        # [num_graphs, all_targets]. This is controlled via out_channels param.
        #
        # DYNAMIC: Handles any graph-level task automatically
        # PRODUCTION-READY: Works with target_selection and multi-target data
        # FUTURE-PROOF: Uses same wrapper pattern as single models
        # =====================================================================
        if self._is_graph_level_task(task_type):
            # Determine pooling method from config or default
            pooling_method = config.get("pooling_method", "mean")

            # Determine out_channels for output projection
            # Priority: 1. num_classes_override (for classification)
            #           2. From hyperparameters (explicit)
            #           3. Infer from sample_data.y (multi-target regression)
            wrapper_out_channels = None

            if num_classes_override is not None:
                # Classification: use num_classes
                wrapper_out_channels = num_classes_override
                logger.debug(
                    f"Graph-level ensemble: out_channels={wrapper_out_channels} from num_classes_override"
                )
            elif "out_channels" in hyperparameters:
                # Explicit out_channels in hyperparameters
                wrapper_out_channels = hyperparameters["out_channels"]
                logger.debug(
                    f"Graph-level ensemble: out_channels={wrapper_out_channels} from hyperparameters"
                )
            elif (
                sample_data is not None and hasattr(sample_data, "y") and sample_data.y is not None
            ):
                # Infer from sample_data for multi-target regression
                y = sample_data.y
                if y.dim() == 0:
                    wrapper_out_channels = 1
                elif y.dim() == 1:
                    wrapper_out_channels = y.size(0)
                else:
                    wrapper_out_channels = y.size(-1)
                logger.debug(
                    f"Graph-level ensemble: out_channels={wrapper_out_channels} inferred from sample_data.y"
                )

            model = GraphLevelModelWrapper(
                model, task_type, pooling_method, out_channels=wrapper_out_channels
            )
            logger.info(
                f"Wrapped ensemble for graph-level task: {task_type} (out_channels={wrapper_out_channels})"
            )

        # =====================================================================
        # WRAP ENSEMBLE FOR EDGE-LEVEL TASKS
        # =====================================================================
        # Individual models were created with _skip_wrapper=True, so they produce
        # node embeddings. The ENSEMBLE needs to be wrapped with EdgeLevelModelWrapper
        # to compute edge-level predictions from these node embeddings.
        #
        # For edge_classification: MLP decoder with num_classes output channels
        # For edge_regression: MLP decoder with edge_out_channels output channels
        # For link_prediction: dot_product decoder with scalar output
        #
        # CRITICAL: model_out_channels must match the ensemble's output embedding dim.
        # For edge tasks, models output hidden_channels (not num_classes), so we
        # infer this from the first model's configuration.
        # =====================================================================
        elif self._is_edge_level_task(task_type):
            decoder_method = "dot_product"
            edge_out_channels = None
            model_out_channels = None

            # Infer model_out_channels from first model in ensemble
            # Edge-level models output hidden_channels embeddings
            if "models" in config and config["models"]:
                first_model_hparams = config["models"][0].get("hyperparameters", {})
                model_out_channels = (
                    first_model_hparams.get("out_channels")
                    or first_model_hparams.get("hidden_channels")
                    or 64  # Default hidden_channels
                )
                logger.debug(
                    f"Inferred model_out_channels={model_out_channels} for edge-level ensemble"
                )

            if task_type.lower() == "edge_classification":
                edge_out_channels = num_classes_override
                if edge_out_channels is None:
                    edge_out_channels = 10  # Default
                    logger.warning(
                        f"edge_classification ensemble: num_classes not specified, defaulting to {edge_out_channels}"
                    )
                decoder_method = "hadamard_mlp"
                logger.info(
                    f"Wrapping ensemble for edge_classification with {decoder_method} decoder ({edge_out_channels} classes)"
                )

            elif task_type.lower() == "edge_regression":
                # Infer from sample_data
                if (
                    sample_data is not None
                    and hasattr(sample_data, "edge_attr")
                    and sample_data.edge_attr is not None
                ):
                    edge_out_channels = sample_data.edge_attr.shape[-1]
                if edge_out_channels is not None and edge_out_channels > 1:
                    decoder_method = "hadamard_mlp"
                logger.info(f"Wrapping ensemble for edge_regression with {decoder_method} decoder")

            else:
                # link_prediction uses dot_product decoder
                logger.info(f"Wrapping ensemble for {task_type} with {decoder_method} decoder")

            model = EdgeLevelModelWrapper(
                model,
                task_type,
                decoder_method,
                edge_out_channels,
                model_out_channels=model_out_channels,
            )

        # Move to device
        if device is not None:
            model = model.to(device)

        param_count = self._count_parameters(model)
        num_models = len(config.get("models", [])) if isinstance(config, dict) else 0
        logger.info(f"Ensemble created with {num_models} models ({param_count:,} total parameters)")

        return model

    # =========================================================================
    # HELPER METHODS (UNCHANGED)
    # =========================================================================

    def _process_hyperparameters(
        self,
        hyperparameters: dict[str, Any],
        metadata: ModelMetadata,
        sample_data: Data | None,
        task_type: str,
    ) -> dict[str, Any]:
        """
        Process hyperparameters: filter to schema, infer channels, apply defaults.

        This method performs three key operations:
        1. Filters out parameters not defined in the model's schema
        2. Infers required parameters (in_channels, out_channels) from data
        3. Applies default values from schema for missing parameters

        The filtering step is critical for robustness - it ensures only parameters
        that the model actually accepts are passed to the constructor, preventing
        errors from extraneous parameters (e.g., batch_size, epochs) that may be
        present in configuration but are not model hyperparameters.

        PHASE 4 NOTE: metadata.hyperparameters is now dynamically generated via
        PyG introspection (from pyg_introspector.py), but maintains the same dict
        format as the original static model_categories.py. The filtering logic
        remains unchanged - only the DATA SOURCE is now dynamic.

        Args:
            hyperparameters: User-provided hyperparameters (may contain extra params)
            metadata: Model metadata with hyperparameter schema (now from introspection)
            sample_data: Sample data for channel inference
            task_type: Task type for output channel inference

        Returns:
            Processed hyperparameters dictionary containing only valid model params

        Example:
            >>> # User provides params including non-model params
            >>> hparams = {"hidden_channels": 64, "num_layers": 3, "batch_size": 32}
            >>> # Factory filters batch_size, infers channels, adds defaults
            >>> processed = factory._process_hyperparameters(
            ...     hparams, metadata, sample_data, "graph_regression"
            ... )
            >>> print(processed)
            {'hidden_channels': 64, 'num_layers': 3, 'in_channels': 16,
             'out_channels': 1, 'dropout': 0.0, 'act': 'relu'}
            >>> # Note: batch_size is filtered out
        """
        # =====================================================================
        # FILTER PARAMETERS TO MODEL SCHEMA (NOW DYNAMICALLY INTROSPECTED)
        # =====================================================================
        # Only keep parameters that are defined in the model's hyperparameter schema.
        # This prevents non-model parameters (batch_size, epochs, learning_rate, etc.)
        # from being passed to the model constructor.
        #
        # Phase 4: metadata.hyperparameters now comes from dynamic introspection
        # via pyg_introspector, but maintains the same dict format.

        schema_params = set(metadata.hyperparameters.keys())
        filtered_params = {}
        filtered_out = {}
        conv_kwargs_passed = {}  # Track conv kwargs being passed through

        # =====================================================================
        # NEW: Check if model accepts **kwargs for Conv layer parameters
        # =====================================================================
        # PyG BasicGNN models (GCN, GAT, GraphSAGE, GIN) accept **kwargs which
        # are passed through to their underlying Conv layers. This allows config
        # parameters like add_self_loops=False to reach the Conv layer.
        #
        # Evidence from PyG source (basic_gnn.py):
        #     class GCN(BasicGNN):
        #         '''**kwargs (optional): Additional arguments of GCNConv.'''
        #         def init_conv(self, in_channels, out_channels, **kwargs):
        #             return GCNConv(in_channels, out_channels, **kwargs)
        #
        # DYNAMIC: Uses introspected accepts_kwargs flag from metadata
        # PRODUCTION-READY: Only passes known conv kwargs, not arbitrary params
        # FUTURE-PROOF: KNOWN_CONV_KWARGS can be extended without code changes
        # =====================================================================
        model_accepts_kwargs_flag = getattr(metadata, "accepts_kwargs", False)

        # FIX 19: Get model-specific conv kwargs to prevent passing invalid params
        # e.g., 'normalize' is valid for GCN/GraphSAGE but NOT for GAT
        model_conv_kwargs = get_model_conv_kwargs(metadata.name)

        for param, value in hyperparameters.items():
            if param in schema_params:
                # Parameter is in model's explicit schema
                filtered_params[param] = value
            elif model_accepts_kwargs_flag and param in model_conv_kwargs:
                # Parameter is a known Conv layer kwarg for THIS MODEL and model accepts **kwargs
                filtered_params[param] = value
                conv_kwargs_passed[param] = value
            else:
                # Parameter is not applicable to this model
                filtered_out[param] = value

        # Log filtered parameters for debugging/transparency
        if filtered_out:
            logger.debug(
                f"Filtered out {len(filtered_out)} non-model parameter(s) for "
                f"'{metadata.name}': {list(filtered_out.keys())}"
            )

        # Log conv kwargs being passed through
        if conv_kwargs_passed:
            logger.info(
                f"Passing {len(conv_kwargs_passed)} conv kwarg(s) through **kwargs for "
                f"'{metadata.name}': {conv_kwargs_passed}"
            )

        # =====================================================================
        # EDGE_ATTR COMPATIBILITY CHECK FOR add_self_loops
        # =====================================================================
        # PyG BasicGNN models have a class attribute `supports_edge_attr` that
        # indicates whether the model can handle multi-dimensional edge features.
        #
        # When add_self_loops=True is set on a model that does NOT support
        # edge_attr (like GCN), and the data contains multi-dimensional edge_attr,
        # the internal add_self_loops() function will crash with:
        # "Tensors must have same number of dimensions: got 1 and 2"
        #
        # This happens because:
        # - GCN uses edge_weight (1D tensor) internally, not edge_attr (2D)
        # - add_self_loops() tries to concatenate self-loop weights with edge_attr
        # - 1D and 2D tensors cannot be concatenated
        #
        # Evidence from PyG source (basic_gnn.py):
        #     class GCN(BasicGNN):
        #         supports_edge_attr: Final[bool] = False
        #     class GAT(BasicGNN):
        #         supports_edge_attr: Final[bool] = True
        #
        # CRITICAL: We ONLY modify add_self_loops if it was explicitly set to True
        # in the config. We do NOT inject add_self_loops=False for models that
        # don't have it in config because:
        # 1. Some conv layers (like SAGEConv) don't accept add_self_loops at all
        # 2. Injecting an unsupported parameter causes instantiation failure
        # 3. Models without add_self_loops in config rely on runtime edge_attr
        #    filtering in model_composer.py (which sets edge_attr=None for
        #    models with supports_edge_attr=False)
        #
        # DYNAMIC: Uses model's actual supports_edge_attr attribute at runtime
        # PRODUCTION-READY: Only modifies explicitly configured parameters
        # FUTURE-PROOF: Works with any PyG model that follows this convention
        # =====================================================================
        if filtered_params.get("add_self_loops") is True and sample_data is not None:
            # Check if data has multi-dimensional edge_attr
            has_multidim_edge_attr = (
                hasattr(sample_data, "edge_attr")
                and sample_data.edge_attr is not None
                and sample_data.edge_attr.dim() >= 2
                and sample_data.edge_attr.size(-1) > 1
            )

            if has_multidim_edge_attr:
                # Get model class to check supports_edge_attr
                model_class = self.registry.get_model(metadata.name)

                if model_class is not None:
                    # Check class attribute (PyG BasicGNN convention)
                    # Also check instance attribute for custom models
                    supports_edge_attr = getattr(model_class, "supports_edge_attr", None)

                    # Get edge_attr dimension for potential edge_dim inference
                    edge_attr_dim = sample_data.edge_attr.size(-1)

                    # =============================================================
                    # CASE 1: Model does NOT support edge_attr (GCN, GIN)
                    # =============================================================
                    # These models use edge_weight (1D) internally, not edge_attr (2D).
                    # When add_self_loops=True is explicitly configured, the internal
                    # add_self_loops() function crashes trying to concatenate
                    # 1D self-loop weights with 2D edge_attr.
                    #
                    # Action: Set add_self_loops=False to prevent crash
                    # Note: This only applies when add_self_loops=True is in config
                    # =============================================================
                    if supports_edge_attr is False:
                        filtered_params["add_self_loops"] = False
                        logger.warning(
                            f"Auto-disabled 'add_self_loops' for model '{metadata.name}': "
                            f"Model has supports_edge_attr=False but data contains "
                            f"multi-dimensional edge_attr (shape={list(sample_data.edge_attr.shape)}). "
                            f"This prevents the 'Tensors must have same number of dimensions' error."
                        )
                        # Update conv_kwargs_passed if it was recorded there
                        if "add_self_loops" in conv_kwargs_passed:
                            conv_kwargs_passed["add_self_loops"] = False

                    # =============================================================
                    # CASE 2: Model DOES support edge_attr (GAT, GATv2, PNA, etc.)
                    #         BUT edge_dim is NOT configured
                    # =============================================================
                    # These models CAN handle edge_attr, but when add_self_loops=True,
                    # they need to know the edge feature dimensionality (edge_dim) to
                    # properly create self-loop edge features using fill_value.
                    #
                    # From PyG GATConv docs:
                    #   edge_dim (int, optional): Edge feature dimensionality
                    #       (in case there are any). (default: None)
                    #   fill_value: The way to generate edge features of self-loops
                    #       (in case edge_dim != None)
                    #
                    # If edge_dim is None (default) and edge_attr is passed, the
                    # add_self_loops() utility crashes with dimension mismatch error.
                    #
                    # Note: Since outer condition requires add_self_loops=True explicitly,
                    # we only reach here when add_self_loops was explicitly configured.
                    #
                    # Action: Disable add_self_loops (safer than auto-setting edge_dim
                    #         which could have unintended effects on model architecture)
                    # =============================================================
                    elif supports_edge_attr is True:
                        # Check if edge_dim is configured in filtered_params
                        edge_dim_configured = filtered_params.get("edge_dim") is not None

                        if not edge_dim_configured:
                            filtered_params["add_self_loops"] = False
                            logger.warning(
                                f"Auto-disabled 'add_self_loops' for model '{metadata.name}': "
                                f"Model has supports_edge_attr=True but 'edge_dim' is not configured. "
                                f"Data contains multi-dimensional edge_attr (shape={list(sample_data.edge_attr.shape)}). "
                                f"When using edge_attr with add_self_loops=True, edge_dim must be set "
                                f"to specify how to generate self-loop edge features. "
                                f"To enable add_self_loops, set edge_dim={edge_attr_dim} in model hyperparameters."
                            )
                            # Update conv_kwargs_passed if it was recorded there
                            if "add_self_loops" in conv_kwargs_passed:
                                conv_kwargs_passed["add_self_loops"] = False

        processed = filtered_params

        # =====================================================================
        # INFER IN_CHANNELS (ONLY IF MODEL SCHEMA SUPPORTS IT)
        # =====================================================================
        # DYNAMIC: Only infer in_channels if the model's introspected schema
        # includes this parameter. Models like SchNet use atomic embeddings
        # and don't accept in_channels. This check ensures compatibility with
        # all PyG models regardless of their signature.
        # PRODUCTION-READY: Uses schema_params from dynamic introspection
        # FUTURE-PROOF: Works for any new PyG model with any signature
        # =====================================================================
        if (
            "in_channels" in schema_params
            and "in_channels" not in processed
            and sample_data is not None
            and hasattr(sample_data, "x")
            and sample_data.x is not None
        ):
            in_channels = sample_data.x.size(-1)
            processed["in_channels"] = in_channels
            logger.debug(f"Inferred in_channels={in_channels} from sample data")

        # =====================================================================
        # INFER OUT_CHANNELS (ONLY IF MODEL SCHEMA SUPPORTS IT)
        # =====================================================================
        # DYNAMIC: Only infer out_channels if the model's introspected schema
        # includes this parameter. Models like SchNet output scalar predictions
        # and don't accept out_channels. This check ensures compatibility with
        # all PyG models regardless of their signature.
        #
        # EXCEPTION: For edge_classification, models should output node EMBEDDINGS
        # (not class logits). The EdgeLevelModelWrapper's MLP decoder handles
        # mapping embeddings to class predictions. Do NOT infer out_channels for
        # edge_classification - let models use hidden_channels as output dim.
        #
        # PRODUCTION-READY: Uses schema_params from dynamic introspection
        # FUTURE-PROOF: Works for any new PyG model with any signature
        # =====================================================================
        is_edge_level_task = task_type.lower().startswith("edge_") or task_type.lower().startswith(
            "link_"
        )

        if "out_channels" in schema_params and "out_channels" not in processed:
            # For edge-level tasks, models output embeddings - don't infer out_channels
            # Let the model use hidden_channels as output dimension
            if is_edge_level_task:
                logger.debug(
                    f"Skipping out_channels inference for edge-level task '{task_type}': "
                    f"Model will output embeddings (hidden_channels), not class logits."
                )
            elif infer_out_channels is not None and sample_data is not None:
                out_channels = infer_out_channels(
                    data=sample_data, task_type=task_type, default=None
                )

                if out_channels is not None:
                    processed["out_channels"] = out_channels
                    logger.debug(f"Inferred out_channels={out_channels} for {task_type}")
                else:
                    logger.warning(
                        f"Could not infer out_channels for task '{task_type}'. "
                        f"Specify 'out_channels' explicitly in hyperparameters."
                    )
            else:
                # Fallback if infer_out_channels not available or no sample_data
                if "regression" in task_type.lower():
                    processed["out_channels"] = 1
                    logger.debug("Inferred out_channels=1 for regression task (fallback)")
                elif "classification" in task_type.lower() and sample_data is not None:
                    if hasattr(sample_data, "y") and sample_data.y is not None:
                        y = sample_data.y
                        if y.dim() > 1:
                            processed["out_channels"] = y.size(-1)
                        else:
                            with contextlib.suppress(Exception):
                                processed["out_channels"] = len(torch.unique(y))

        # =====================================================================
        # PNA-SPECIFIC PARAMETER HANDLING
        # =====================================================================
        # PNA (Principal Neighbourhood Aggregation) requires three mandatory
        # parameters that other BasicGNN models don't need:
        # - aggregators: List[str] - aggregation function identifiers
        # - scalers: List[str] - scaling function identifiers
        # - deg: Tensor - histogram of in-degrees (MUST be computed from data)
        #
        # Evidence from PyG documentation:
        # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.PNAConv.html
        # - aggregators: "sum", "mean", "min", "max", "var", "std"
        # - scalers: "identity", "amplification", "attenuation", "linear", "inverse_linear"
        # - deg: computed via PNAConv.get_degree_histogram() or torch_geometric.utils.degree
        #
        # DYNAMIC: Detects PNA models by name pattern, not hardcoded list
        # PRODUCTION-READY: Provides sensible defaults from original paper
        # FUTURE-PROOF: Computes deg from any available data
        # =====================================================================
        is_pna_model = metadata.name.upper() == "PNA" or "pna" in metadata.name.lower()

        if is_pna_model:
            # -----------------------------------------------------------------
            # PNA aggregators: default from original paper (NeurIPS 2020)
            # Common choices: mean, min, max, std (4 aggregators)
            # -----------------------------------------------------------------
            if "aggregators" not in processed:
                processed["aggregators"] = ["mean", "min", "max", "std"]
                logger.info(f"PNA: Using default aggregators={processed['aggregators']}")

            # -----------------------------------------------------------------
            # PNA scalers: default from original paper (NeurIPS 2020)
            # Common choices: identity, amplification, attenuation (3 scalers)
            # -----------------------------------------------------------------
            if "scalers" not in processed:
                processed["scalers"] = ["identity", "amplification", "attenuation"]
                logger.info(f"PNA: Using default scalers={processed['scalers']}")

            # -----------------------------------------------------------------
            # PNA deg: MUST be computed from training data
            # deg is a histogram of in-degrees used by scalers for normalization
            # Formula: deg[d] = count of nodes with in-degree d in training set
            # -----------------------------------------------------------------
            if "deg" not in processed:
                if sample_data is not None and hasattr(sample_data, "edge_index"):
                    try:
                        from torch_geometric.utils import degree

                        edge_index = sample_data.edge_index
                        num_nodes = (
                            sample_data.num_nodes if hasattr(sample_data, "num_nodes") else None
                        )

                        # Compute in-degree for each node (edge_index[1] is target nodes)
                        node_degrees = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.long)

                        # Create degree histogram (count of nodes with each degree)
                        # This matches PNAConv.get_degree_histogram() output format
                        max_degree = (
                            int(node_degrees.max().item()) if node_degrees.numel() > 0 else 0
                        )
                        deg_histogram = torch.bincount(node_degrees, minlength=max_degree + 1).to(
                            torch.long
                        )

                        processed["deg"] = deg_histogram
                        logger.info(
                            f"PNA: Computed deg histogram from sample_data "
                            f"(max_degree={max_degree}, histogram_len={len(deg_histogram)})"
                        )
                    except Exception as e:
                        logger.warning(
                            f"PNA: Could not compute deg from sample_data: {e}. "
                            f"Model instantiation may fail. "
                            f"Please provide 'deg' parameter explicitly."
                        )
                else:
                    logger.warning(
                        "PNA: Cannot compute 'deg' without sample_data. "
                        "PNA model requires degree histogram for normalization. "
                        "Please provide 'deg' parameter explicitly or ensure sample_data is available."
                    )

        # =====================================================================
        # 3D MOLECULAR MODEL PARAMETER HANDLING (SchNet, DimeNet, PaiNN, etc.)
        # =====================================================================
        # 3D molecular models have different requirements than standard GNNs:
        # - Forward signature: (z, pos, batch) instead of (x, edge_index, ...)
        # - Data requirements: data.z (atomic numbers) and data.pos (3D coordinates)
        # - No in_channels/out_channels: use internal atomic embeddings
        #
        # These models will FAIL at forward pass if data lacks z and pos.
        # We validate this EARLY to provide informative error messages.
        #
        # Evidence from PyG documentation:
        # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.SchNet.html
        # - forward(z: Tensor, pos: Tensor, batch: Optional[Tensor] = None)
        # - z: Atomic numbers with shape [num_atoms]
        # - pos: Coordinates with shape [num_atoms, 3]
        #
        # DYNAMIC: Detects 3D molecular models by name pattern and forward signature
        # PRODUCTION-READY: Validates data attributes before model instantiation
        # FUTURE-PROOF: Works with any new 3D model following PyG conventions
        # =====================================================================

        # Pattern-based detection of 3D molecular models
        # These models use (z, pos, batch) forward signature
        model_name_lower = metadata.name.lower()
        is_3d_molecular_model = any(
            pattern in model_name_lower
            for pattern in [
                "schnet",
                "dimenet",
                "painn",
                "egnn",
                "visnet",
                "comenet",
                "spherenet",
                "gemnet",
                "equiformer",
                "mace",
                "nequip",
            ]
        )

        # Also check forward signature via metadata (more robust for unknown models)
        if not is_3d_molecular_model and hasattr(metadata, "forward_parameters"):
            forward_params = metadata.forward_parameters
            # 3D models typically have 'z' and 'pos' as required forward parameters
            has_z_param = "z" in forward_params
            has_pos_param = "pos" in forward_params
            if has_z_param and has_pos_param:
                is_3d_molecular_model = True
                logger.debug(
                    f"Detected '{metadata.name}' as 3D molecular model via forward signature "
                    f"(has 'z' and 'pos' parameters)"
                )

        if is_3d_molecular_model:
            logger.debug(f"Processing 3D molecular model: {metadata.name}")

            # -----------------------------------------------------------------
            # VALIDATE REQUIRED DATA ATTRIBUTES (z, pos)
            # -----------------------------------------------------------------
            # 3D molecular models REQUIRE atomic numbers (z) and 3D positions (pos).
            # Without these, the model will fail at forward pass. We validate
            # early to provide clear, actionable error messages.
            # -----------------------------------------------------------------
            if sample_data is not None:
                missing_attrs = []

                # Check for atomic numbers (z)
                if not hasattr(sample_data, "z") or sample_data.z is None:
                    missing_attrs.append("'z' (atomic numbers) - required for atomic embeddings")

                # Check for 3D positions (pos)
                if not hasattr(sample_data, "pos") or sample_data.pos is None:
                    missing_attrs.append(
                        "'pos' (3D atomic coordinates) - required for distance calculations"
                    )

                if missing_attrs:
                    error_msg = (
                        f"3D molecular model '{metadata.name}' requires specialized molecular data. "
                        f"Missing required attributes:\n"
                        f"  - " + "\n  - ".join(missing_attrs) + "\n"
                        "This model uses forward(z, pos, batch) signature instead of "
                        "forward(x, edge_index, ...). Ensure your dataset contains:\n"
                        "  - data.z: Atomic numbers tensor [num_atoms] (e.g., 6 for Carbon, 8 for Oxygen)\n"
                        "  - data.pos: 3D coordinates tensor [num_atoms, 3] (in Angstroms)\n"
                        "Common molecular datasets with these attributes: QM9, MD17, ANI, GEOM."
                    )
                    raise DataCompatibilityError(error_msg)

                # Log successful validation
                logger.info(
                    f"3D molecular model '{metadata.name}': Data validation passed "
                    f"(z shape: {list(sample_data.z.shape)}, pos shape: {list(sample_data.pos.shape)})"
                )
            else:
                # No sample_data - warn but allow (validation will happen at forward pass)
                logger.warning(
                    f"3D molecular model '{metadata.name}': No sample_data provided for validation. "
                    f"Ensure data has 'z' (atomic numbers) and 'pos' (3D coordinates) attributes "
                    f"or model will fail at forward pass."
                )

            # -----------------------------------------------------------------
            # SCHNET-SPECIFIC PARAMETER HANDLING
            # -----------------------------------------------------------------
            # SchNet has specific parameters that may need defaults or validation.
            # All SchNet parameters have PyG defaults, but we provide MILIA defaults
            # that are optimized for molecular property prediction.
            #
            # Evidence from SchNet paper (NeurIPS 2017):
            # - num_interactions: 6 (default), controls depth of message passing
            # - num_gaussians: 50 (default), controls distance expansion resolution
            # - cutoff: 10.0 (PyG default), but 5.0 is common for small molecules
            # -----------------------------------------------------------------
            is_schnet = "schnet" in model_name_lower

            if is_schnet:
                # SchNet-specific defaults (already in schema via _infer_intelligent_default,
                # but we ensure consistency and provide logging)
                schnet_recommended_defaults = {
                    "hidden_channels": 128,
                    "num_filters": 128,
                    "num_interactions": 6,
                    "num_gaussians": 50,
                    "cutoff": 5.0,  # MILIA default (PyG default is 10.0)
                    "max_num_neighbors": 32,
                    "readout": "add",
                }

                # Apply recommended defaults if not already set
                applied_defaults = []
                for param, default_value in schnet_recommended_defaults.items():
                    if param in schema_params and param not in processed:
                        processed[param] = default_value
                        applied_defaults.append(f"{param}={default_value}")

                if applied_defaults:
                    logger.info(
                        f"SchNet: Applied recommended defaults: {', '.join(applied_defaults)}"
                    )

                # Log cutoff value (important for performance)
                cutoff_value = processed.get("cutoff", schnet_recommended_defaults["cutoff"])
                logger.debug(f"SchNet: Using cutoff={cutoff_value}Å for interatomic interactions")

            # -----------------------------------------------------------------
            # DIMENET-SPECIFIC PARAMETER HANDLING
            # -----------------------------------------------------------------
            # DimeNet has a HARD CONSTRAINT: num_spherical must be >= 2
            # This is enforced by PyG and will raise ValueError if violated.
            #
            # Evidence from PyG source (dimenet.py):
            # if num_spherical < 2:
            #     raise ValueError("num_spherical must be >= 2")
            # -----------------------------------------------------------------
            is_dimenet = "dimenet" in model_name_lower

            if is_dimenet:
                # Validate num_spherical constraint
                if "num_spherical" in processed:
                    num_spherical = processed["num_spherical"]
                    if num_spherical < 2:
                        logger.warning(
                            f"DimeNet: Adjusting num_spherical from {num_spherical} to 2 "
                            f"(PyG enforces num_spherical >= 2)"
                        )
                        processed["num_spherical"] = 2

                # DimeNet-specific defaults
                dimenet_recommended_defaults = {
                    "num_blocks": 6,
                    "num_bilinear": 8,
                    "num_spherical": 7,
                    "num_radial": 6,
                    "cutoff": 5.0,
                    "max_num_neighbors": 32,
                }

                # Apply recommended defaults if not already set
                applied_defaults = []
                for param, default_value in dimenet_recommended_defaults.items():
                    if param in schema_params and param not in processed:
                        processed[param] = default_value
                        applied_defaults.append(f"{param}={default_value}")

                if applied_defaults:
                    logger.info(
                        f"DimeNet: Applied recommended defaults: {', '.join(applied_defaults)}"
                    )

        # =====================================================================
        # GAE/VGAE AUTOENCODER MODEL PARAMETER HANDLING (Fix 27)
        # =====================================================================
        # Graph Autoencoders (GAE) and Variational Graph Autoencoders (VGAE)
        # require an `encoder` module to be passed at construction time.
        #
        # Unlike other PyG models that accept hyperparameters like hidden_channels,
        # num_layers, etc., GAE/VGAE expect a pre-constructed torch.nn.Module as
        # their encoder parameter.
        #
        # SOLUTION: Dynamically create an encoder module based on user config:
        # - encoder_type: Which PyG BasicGNN to use (GCN, GAT, GraphSAGE, GIN)
        # - encoder_hidden_channels: Hidden dimension for encoder layers
        # - encoder_num_layers: Number of encoder layers
        # - encoder_out_channels / out_channels: Latent embedding dimension
        #
        # For VGAE specifically, the encoder must output (mu, logstd) tensors,
        # which requires a special variational encoder architecture.
        #
        # Evidence from PyG autoencoder.py source:
        #   class GAE(torch.nn.Module):
        #       def __init__(self, encoder: Module, decoder: Optional[Module] = None):
        #
        #   class VGAE(GAE):
        #       # encoder must output (mu, logstd) for variational inference
        #
        # DYNAMIC: Detects autoencoder models by name pattern and import path
        # PRODUCTION-READY: Creates encoder using existing PyG models dynamically
        # FUTURE-PROOF: Supports any encoder_type available in PyG BasicGNN family
        # =====================================================================

        is_gae_model = model_name_lower == "gae" or "gae" in metadata.import_path.lower()
        is_vgae_model = model_name_lower == "vgae" or "vgae" in metadata.import_path.lower()
        is_autoencoder_model = is_gae_model or is_vgae_model

        # Also detect ARGA/ARGVA (adversarial variants) - these need discriminator too
        is_arga_model = model_name_lower == "arga"
        is_argva_model = model_name_lower == "argva"
        is_adversarial_autoencoder = is_arga_model or is_argva_model

        if is_autoencoder_model and not is_adversarial_autoencoder:
            logger.debug(f"Processing autoencoder model: {metadata.name}")

            # -----------------------------------------------------------------
            # ENCODER CONFIGURATION EXTRACTION
            # -----------------------------------------------------------------
            # Users can specify encoder configuration via these parameters:
            # - encoder_type: "GCN", "GAT", "GraphSAGE", "GIN" (default: "GCN")
            # - encoder_hidden_channels: Hidden dimension (default: 64)
            # - encoder_num_layers: Number of layers (default: 2)
            # - encoder_out_channels OR out_channels: Latent dimension (default: 16)
            # - encoder_dropout: Dropout rate (default: 0.0)
            # - encoder_act: Activation function (default: "relu")
            # -----------------------------------------------------------------

            encoder_type = hyperparameters.get("encoder_type", "GCN")
            encoder_hidden_channels = hyperparameters.get(
                "encoder_hidden_channels", hyperparameters.get("hidden_channels", 64)
            )
            encoder_num_layers = hyperparameters.get(
                "encoder_num_layers", hyperparameters.get("num_layers", 2)
            )
            # out_channels in autoencoder context = latent embedding dimension
            encoder_out_channels = hyperparameters.get(
                "encoder_out_channels", hyperparameters.get("out_channels", 16)
            )
            encoder_dropout = hyperparameters.get(
                "encoder_dropout", hyperparameters.get("dropout", 0.0)
            )
            encoder_act = hyperparameters.get("encoder_act", hyperparameters.get("act", "relu"))

            # -----------------------------------------------------------------
            # INFER IN_CHANNELS FROM SAMPLE DATA
            # -----------------------------------------------------------------
            encoder_in_channels = hyperparameters.get("encoder_in_channels")
            if encoder_in_channels is None:
                encoder_in_channels = hyperparameters.get("in_channels")
            if (
                encoder_in_channels is None
                and sample_data is not None
                and hasattr(sample_data, "x")
                and sample_data.x is not None
            ):
                encoder_in_channels = sample_data.x.size(-1)
                logger.debug(
                    f"GAE/VGAE: Inferred encoder_in_channels={encoder_in_channels} from sample_data.x"
                )

            if encoder_in_channels is None:
                raise HyperparameterError(
                    f"Autoencoder model '{metadata.name}' requires 'encoder_in_channels' or "
                    f"'in_channels' to be specified, or sample_data with node features (data.x). "
                    f"Please provide one of these to define the encoder input dimension."
                )

            # -----------------------------------------------------------------
            # CREATE ENCODER MODULE DYNAMICALLY
            # -----------------------------------------------------------------
            # We use PyG's BasicGNN models as the encoder backbone
            # For GAE: standard encoder outputs [num_nodes, out_channels]
            # For VGAE: variational encoder outputs (mu, logstd) tuple
            # -----------------------------------------------------------------

            try:
                encoder = self._create_autoencoder_encoder(
                    encoder_type=encoder_type,
                    in_channels=encoder_in_channels,
                    hidden_channels=encoder_hidden_channels,
                    out_channels=encoder_out_channels,
                    num_layers=encoder_num_layers,
                    dropout=encoder_dropout,
                    act=encoder_act,
                    is_variational=is_vgae_model,
                )

                logger.info(
                    f"{metadata.name}: Created {'variational ' if is_vgae_model else ''}"
                    f"encoder using {encoder_type} "
                    f"(in={encoder_in_channels}, hidden={encoder_hidden_channels}, "
                    f"out={encoder_out_channels}, layers={encoder_num_layers})"
                )

                # Set encoder in processed params
                processed["encoder"] = encoder

            except Exception as e:
                raise ModelInstantiationError(
                    f"Failed to create encoder for autoencoder model '{metadata.name}': {e}. "
                    f"Attempted encoder_type='{encoder_type}' with in_channels={encoder_in_channels}, "
                    f"hidden_channels={encoder_hidden_channels}, out_channels={encoder_out_channels}. "
                    f"Ensure encoder_type is a valid PyG BasicGNN model (GCN, GAT, GraphSAGE, GIN).",
                    model_name=metadata.name,
                ) from e

            # -----------------------------------------------------------------
            # REMOVE ENCODER CONFIG PARAMS (not needed by GAE/VGAE constructor)
            # -----------------------------------------------------------------
            # GAE/VGAE only accept 'encoder' and 'decoder', not the config params
            # we used to create the encoder
            encoder_config_params = [
                "encoder_type",
                "encoder_hidden_channels",
                "encoder_num_layers",
                "encoder_out_channels",
                "encoder_dropout",
                "encoder_act",
                "encoder_in_channels",
                "in_channels",
                "hidden_channels",
                "num_layers",
                "dropout",
                "act",
                "out_channels",
            ]
            for param in encoder_config_params:
                processed.pop(param, None)

            logger.debug(
                f"GAE/VGAE: Final processed params after encoder creation: {list(processed.keys())}"
            )

        elif is_adversarial_autoencoder:
            # =================================================================
            # ARGA/ARGVA ADVERSARIAL AUTOENCODER MODEL PARAMETER HANDLING
            # =================================================================
            # Adversarially Regularized Graph Autoencoders require BOTH an
            # encoder module AND a discriminator module at construction time.
            #
            # - ARGA: encoder (standard) + discriminator
            # - ARGVA: encoder (variational, outputs (mu, logstd)) + discriminator
            #
            # The discriminator is an MLP that takes the latent space embeddings
            # and outputs a scalar (for GAN discrimination between real prior
            # samples and encoded graph embeddings).
            #
            # Evidence from PyG autoencoder.py source:
            #   class ARGA(GAE):
            #       def __init__(self, encoder, discriminator, decoder=None):
            #           self.discriminator = discriminator
            #           super().__init__(encoder, decoder)
            #
            #   class ARGVA(ARGA):
            #       # encoder must output (mu, logstd) for variational inference
            #
            # Evidence from PyG examples/argva_node_clustering.py:
            #   class Discriminator(torch.nn.Module):
            #       def __init__(self, in_channels, hidden_channels, out_channels):
            #           self.lin1 = Linear(in_channels, hidden_channels)
            #           self.lin2 = Linear(hidden_channels, hidden_channels)
            #           self.lin3 = Linear(hidden_channels, out_channels)
            #       def forward(self, x):
            #           x = F.relu(self.lin1(x))
            #           x = F.relu(self.lin2(x))
            #           return self.lin3(x)
            #
            # DYNAMIC: Detects adversarial autoencoder models by name pattern
            # PRODUCTION-READY: Creates both encoder and discriminator dynamically
            # FUTURE-PROOF: Uses configurable architecture via parameters
            # =================================================================

            logger.debug(f"Processing adversarial autoencoder model: {metadata.name}")

            # -----------------------------------------------------------------
            # ENCODER CONFIGURATION EXTRACTION (same as GAE/VGAE)
            # -----------------------------------------------------------------
            encoder_type = hyperparameters.get("encoder_type", "GCN")
            encoder_hidden_channels = hyperparameters.get(
                "encoder_hidden_channels", hyperparameters.get("hidden_channels", 64)
            )
            encoder_num_layers = hyperparameters.get(
                "encoder_num_layers", hyperparameters.get("num_layers", 2)
            )
            # out_channels in autoencoder context = latent embedding dimension
            encoder_out_channels = hyperparameters.get(
                "encoder_out_channels", hyperparameters.get("out_channels", 32)
            )
            encoder_dropout = hyperparameters.get(
                "encoder_dropout", hyperparameters.get("dropout", 0.0)
            )
            encoder_act = hyperparameters.get("encoder_act", hyperparameters.get("act", "relu"))

            # -----------------------------------------------------------------
            # DISCRIMINATOR CONFIGURATION EXTRACTION
            # -----------------------------------------------------------------
            # The discriminator is an MLP that takes latent embeddings (z) and
            # outputs a scalar score. Architecture follows PyG examples.
            #
            # Default architecture based on PyG examples/argva_node_clustering.py:
            # - Input: latent_dim (encoder_out_channels)
            # - Hidden: discriminator_hidden_channels (2x latent_dim)
            # - Output: 1 (scalar for discrimination)
            # -----------------------------------------------------------------
            discriminator_hidden_channels = hyperparameters.get(
                "discriminator_hidden_channels",
                hyperparameters.get("hidden_channels", encoder_out_channels * 2),
            )
            discriminator_out_channels = hyperparameters.get(
                "discriminator_out_channels",
                encoder_out_channels,  # Match latent dimension for intermediate layer
            )

            # -----------------------------------------------------------------
            # INFER IN_CHANNELS FROM SAMPLE DATA
            # -----------------------------------------------------------------
            encoder_in_channels = hyperparameters.get("encoder_in_channels")
            if encoder_in_channels is None:
                encoder_in_channels = hyperparameters.get("in_channels")
            if (
                encoder_in_channels is None
                and sample_data is not None
                and hasattr(sample_data, "x")
                and sample_data.x is not None
            ):
                encoder_in_channels = sample_data.x.size(-1)
                logger.debug(
                    f"ARGA/ARGVA: Inferred encoder_in_channels={encoder_in_channels} from sample_data.x"
                )

            if encoder_in_channels is None:
                raise HyperparameterError(
                    f"Adversarial autoencoder model '{metadata.name}' requires 'encoder_in_channels' or "
                    f"'in_channels' to be specified, or sample_data with node features (data.x). "
                    f"Please provide one of these to define the encoder input dimension."
                )

            # -----------------------------------------------------------------
            # CREATE ENCODER MODULE DYNAMICALLY
            # -----------------------------------------------------------------
            # ARGA uses standard encoder, ARGVA uses variational encoder
            # Reuse existing _create_autoencoder_encoder() method
            # -----------------------------------------------------------------
            try:
                encoder = self._create_autoencoder_encoder(
                    encoder_type=encoder_type,
                    in_channels=encoder_in_channels,
                    hidden_channels=encoder_hidden_channels,
                    out_channels=encoder_out_channels,
                    num_layers=encoder_num_layers,
                    dropout=encoder_dropout,
                    act=encoder_act,
                    is_variational=is_argva_model,  # True for ARGVA, False for ARGA
                )

                logger.info(
                    f"{metadata.name}: Created {'variational ' if is_argva_model else ''}"
                    f"encoder using {encoder_type} "
                    f"(in={encoder_in_channels}, hidden={encoder_hidden_channels}, "
                    f"out={encoder_out_channels}, layers={encoder_num_layers})"
                )

                # Set encoder in processed params
                processed["encoder"] = encoder

            except Exception as e:
                raise ModelInstantiationError(
                    f"Failed to create encoder for adversarial autoencoder model '{metadata.name}': {e}. "
                    f"Attempted encoder_type='{encoder_type}' with in_channels={encoder_in_channels}, "
                    f"hidden_channels={encoder_hidden_channels}, out_channels={encoder_out_channels}. "
                    f"Ensure encoder_type is a valid PyG BasicGNN model (GCN, GAT, GraphSAGE, GIN).",
                    model_name=metadata.name,
                ) from e

            # -----------------------------------------------------------------
            # CREATE DISCRIMINATOR MODULE DYNAMICALLY
            # -----------------------------------------------------------------
            # The discriminator is an MLP that takes the latent space z and
            # outputs a scalar for GAN discrimination.
            #
            # Architecture based on PyG examples/argva_node_clustering.py:
            #   class Discriminator(torch.nn.Module):
            #       def __init__(self, in_channels, hidden_channels, out_channels):
            #           self.lin1 = Linear(in_channels, hidden_channels)
            #           self.lin2 = Linear(hidden_channels, hidden_channels)
            #           self.lin3 = Linear(hidden_channels, out_channels)
            #       def forward(self, x):
            #           x = F.relu(self.lin1(x))
            #           x = F.relu(self.lin2(x))
            #           return self.lin3(x)
            #
            # The discriminator output is then passed through sigmoid in ARGA's
            # reg_loss() and discriminator_loss() methods.
            # -----------------------------------------------------------------
            try:
                discriminator = self._create_adversarial_discriminator(
                    in_channels=encoder_out_channels,  # Takes latent embeddings
                    hidden_channels=discriminator_hidden_channels,
                    out_channels=discriminator_out_channels,
                )

                logger.info(
                    f"{metadata.name}: Created discriminator MLP "
                    f"(in={encoder_out_channels}, hidden={discriminator_hidden_channels}, "
                    f"out={discriminator_out_channels})"
                )

                # Set discriminator in processed params
                processed["discriminator"] = discriminator

            except Exception as e:
                raise ModelInstantiationError(
                    f"Failed to create discriminator for adversarial autoencoder model '{metadata.name}': {e}. "
                    f"Attempted with in_channels={encoder_out_channels}, "
                    f"hidden_channels={discriminator_hidden_channels}, out_channels={discriminator_out_channels}.",
                    model_name=metadata.name,
                ) from e

            # -----------------------------------------------------------------
            # REMOVE CONFIG PARAMS (not needed by ARGA/ARGVA constructor)
            # -----------------------------------------------------------------
            # ARGA/ARGVA only accept 'encoder', 'discriminator', and 'decoder'
            encoder_config_params = [
                "encoder_type",
                "encoder_hidden_channels",
                "encoder_num_layers",
                "encoder_out_channels",
                "encoder_dropout",
                "encoder_act",
                "encoder_in_channels",
                "in_channels",
                "hidden_channels",
                "num_layers",
                "dropout",
                "act",
                "out_channels",
                "discriminator_hidden_channels",
                "discriminator_out_channels",
            ]
            for param in encoder_config_params:
                processed.pop(param, None)

            logger.debug(
                f"ARGA/ARGVA: Final processed params after encoder/discriminator creation: {list(processed.keys())}"
            )

        # =====================================================================
        # APPLY DEFAULT VALUES FROM SCHEMA
        # =====================================================================
        for param, schema in metadata.hyperparameters.items():
            if param not in processed and "default" in schema:
                default_value = schema["default"]
                processed[param] = default_value
                logger.debug(f"Applied default {param}={default_value}")

        return processed

    @staticmethod
    def _is_graph_level_task(task_type: str | None) -> bool:
        """
        Determine if task is graph-level (requires global pooling).

        Graph-level tasks expect output shape [num_graphs, features],
        while PyG BasicGNN models output [num_nodes, features].

        Args:
            task_type: Task type string (e.g., 'graph_regression', 'node_classification')

        Returns:
            True if task_type starts with 'graph_' (case-insensitive)

        Example:
            >>> ModelFactory._is_graph_level_task('graph_regression')
            True
            >>> ModelFactory._is_graph_level_task('node_classification')
            False
            >>> ModelFactory._is_graph_level_task('GRAPH_CLASSIFICATION')
            True
        """
        if task_type is None:
            return False
        return task_type.lower().startswith("graph_")

    @staticmethod
    def _is_edge_level_task(task_type: str | None) -> bool:
        """
        Determine if task is edge-level (requires edge decoder).

        Edge-level tasks predict properties of edges (links) rather than
        nodes or entire graphs. These tasks require:
        - Edge decoder to compute scores from node embeddings
        - Edge-specific targets (edge_label, edge_value)

        Args:
            task_type: Task type string (e.g., 'link_prediction', 'edge_regression')

        Returns:
            True if task_type is an edge-level task

        Example:
            >>> ModelFactory._is_edge_level_task('link_prediction')
            True
            >>> ModelFactory._is_edge_level_task('edge_regression')
            True
            >>> ModelFactory._is_edge_level_task('graph_regression')
            False
            >>> ModelFactory._is_edge_level_task('node_classification')
            False
        """
        if task_type is None:
            return False

        task_lower = task_type.lower()

        # Explicit edge-level task types
        edge_level_tasks = ["link_prediction", "edge_regression", "edge_classification"]

        if task_lower in edge_level_tasks:
            return True

        # Future-proof: any task starting with 'link_' or 'edge_'
        return task_lower.startswith("link_") or task_lower.startswith("edge_")

    def _create_autoencoder_encoder(
        self,
        encoder_type: str,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        act: str = "relu",
        is_variational: bool = False,
    ) -> torch.nn.Module:
        """
        Dynamically create an encoder module for GAE/VGAE autoencoder models.

        This method creates an encoder using PyG's BasicGNN models (GCN, GAT,
        GraphSAGE, GIN) as the backbone. For VGAE, it creates a variational
        encoder that outputs (mu, logstd) tensors required for the reparameterization
        trick.

        DYNAMIC: Uses PyG's introspector to get any available BasicGNN model
        PRODUCTION-READY: Supports configurable architecture via parameters
        FUTURE-PROOF: Works with any PyG BasicGNN model without hardcoding

        Args:
            encoder_type: Name of PyG BasicGNN model to use as encoder backbone
                         Options: "GCN", "GAT", "GraphSAGE", "GIN", etc.
            in_channels: Input feature dimension (node features)
            hidden_channels: Hidden layer dimension
            out_channels: Output/latent embedding dimension
            num_layers: Number of message passing layers (default: 2)
            dropout: Dropout rate (default: 0.0)
            act: Activation function name (default: 'relu')
            is_variational: If True, creates variational encoder for VGAE
                           that outputs (mu, logstd) tuple

        Returns:
            torch.nn.Module: Encoder module ready for GAE/VGAE

        Raises:
            ModelInstantiationError: If encoder_type is not available

        Example:
            >>> # Standard encoder for GAE
            >>> encoder = factory._create_autoencoder_encoder(
            ...     encoder_type="GCN",
            ...     in_channels=1433,  # Cora features
            ...     hidden_channels=64,
            ...     out_channels=16,   # Latent dim
            ...     is_variational=False
            ... )
            >>> model = GAE(encoder)

            >>> # Variational encoder for VGAE
            >>> encoder = factory._create_autoencoder_encoder(
            ...     encoder_type="GCN",
            ...     in_channels=1433,
            ...     hidden_channels=64,
            ...     out_channels=16,
            ...     is_variational=True
            ... )
            >>> model = VGAE(encoder)  # encoder outputs (mu, logstd)
        """
        import importlib

        # Normalize encoder type name
        encoder_type_normalized = encoder_type.upper()
        if encoder_type_normalized == "GRAPHSAGE":
            encoder_type_normalized = "GraphSAGE"
        elif encoder_type_normalized in ("GCN", "GAT", "GIN", "PNA"):
            encoder_type_normalized = encoder_type.upper()
        else:
            # Keep original casing for other models
            encoder_type_normalized = encoder_type

        # Get encoder backbone class from PyG
        try:
            # Try torch_geometric.nn.models first (BasicGNN models)
            pyg_models = importlib.import_module("torch_geometric.nn.models")
            if hasattr(pyg_models, encoder_type_normalized):
                encoder_backbone_class = getattr(pyg_models, encoder_type_normalized)
            elif hasattr(pyg_models, encoder_type):
                encoder_backbone_class = getattr(pyg_models, encoder_type)
            else:
                # Fallback to torch_geometric.nn
                pyg_nn = importlib.import_module("torch_geometric.nn")
                if hasattr(pyg_nn, encoder_type_normalized):
                    encoder_backbone_class = getattr(pyg_nn, encoder_type_normalized)
                elif hasattr(pyg_nn, encoder_type):
                    encoder_backbone_class = getattr(pyg_nn, encoder_type)
                else:
                    raise AttributeError(f"Encoder type '{encoder_type}' not found in PyG models")
        except Exception as e:
            raise ModelInstantiationError(
                f"Could not load encoder backbone '{encoder_type}': {e}. "
                f"Supported encoder types: GCN, GAT, GraphSAGE, GIN, PNA, EdgeCNN",
                model_name=f"GAE/VGAE encoder ({encoder_type})",
            ) from e

        if is_variational:
            # =================================================================
            # VARIATIONAL ENCODER FOR VGAE
            # =================================================================
            # VGAE requires the encoder to output (mu, logstd) tensors.
            # We create a wrapper module that uses the BasicGNN backbone
            # to encode node features, then produces separate mu and logstd.
            #
            # Architecture (based on PyG autoencoder.py example):
            # - Shared GNN layers for feature extraction
            # - Separate final layers for mu and logstd
            # =================================================================

            class VariationalEncoder(torch.nn.Module):
                """
                Variational encoder for VGAE that outputs (mu, logstd).

                Uses a PyG BasicGNN as the backbone, with separate output
                heads for mean (mu) and log standard deviation (logstd).
                """

                def __init__(
                    self,
                    backbone_class: type,
                    in_channels: int,
                    hidden_channels: int,
                    out_channels: int,
                    num_layers: int,
                    dropout: float,
                    act: str,
                ):
                    super().__init__()

                    # Import GCNConv for final variational layers
                    from torch_geometric.nn import GCNConv

                    # Shared backbone (all layers except last)
                    # Use num_layers-1 for shared, final layer is split
                    if num_layers > 1:
                        self.backbone = backbone_class(
                            in_channels=in_channels,
                            hidden_channels=hidden_channels,
                            num_layers=num_layers - 1,
                            out_channels=hidden_channels,
                            dropout=dropout,
                            act=act,
                        )
                        backbone_out = hidden_channels
                    else:
                        # Single layer case - no shared backbone
                        self.backbone = None
                        backbone_out = in_channels

                    # Separate heads for mu and logstd
                    # Using GCNConv for final layers (standard practice)
                    self.conv_mu = GCNConv(backbone_out, out_channels)
                    self.conv_logstd = GCNConv(backbone_out, out_channels)

                def forward(self, x, edge_index, *args, **kwargs):
                    """
                    Forward pass returning (mu, logstd) for variational inference.

                    Args:
                        x: Node features [num_nodes, in_channels]
                        edge_index: Graph connectivity [2, num_edges]
                        *args: Additional positional args (ignored - see Fix 29)
                        **kwargs: Additional keyword args (ignored - see Fix 29)

                    Returns:
                        Tuple of (mu, logstd) tensors, each [num_nodes, out_channels]

                    Note:
                        Fix 29: This encoder intentionally does NOT pass *args or
                        **kwargs to the backbone. The reason is that GAE/VGAE are
                        typically called with edge_attr in the args (from
                        EdgeLevelModelWrapper), but GCN-based backbones interpret
                        the third positional argument as edge_weight (1D tensor).
                        Multi-dimensional edge_attr causes dimension mismatch errors.

                        Standard GAE/VGAE encoders only use (x, edge_index) as per
                        the original "Variational Graph Auto-Encoders" paper.
                    """
                    # =============================================================
                    # FIX 29: VGAE Encoder Edge Attribute Handling
                    # =============================================================
                    # ISSUE: When EdgeLevelModelWrapper calls VGAE with edge_attr,
                    # the edge_attr flows through *args to this encoder. The GCN
                    # backbone interprets the third positional arg as edge_weight
                    # (1D tensor), but edge_attr is typically multi-dimensional,
                    # causing: "RuntimeError: The size of tensor a (X) must match
                    # the size of tensor b (Y) at non-singleton dimension 0"
                    #
                    # ROOT CAUSE: GCN.forward(x, edge_index, edge_weight=None)
                    # expects edge_weight to be 1D [num_edges], but edge_attr is
                    # 2D [num_edges, edge_dim].
                    #
                    # FIX: Only pass (x, edge_index) to backbone - standard practice
                    # for GAE/VGAE which don't use edge features in encoding.
                    #
                    # DYNAMIC: Works with any backbone that accepts (x, edge_index)
                    # PRODUCTION-READY: Prevents cryptic dimension mismatch errors
                    # FUTURE-PROOF: Aligns with standard GAE/VGAE implementations
                    # =============================================================

                    # Apply shared backbone if present
                    # NOTE: Only pass x and edge_index (not *args which may contain edge_attr)
                    if self.backbone is not None:
                        x = self.backbone(x, edge_index)

                    # Compute mu and logstd from separate heads
                    mu = self.conv_mu(x, edge_index)
                    logstd = self.conv_logstd(x, edge_index)

                    return mu, logstd

            encoder = VariationalEncoder(
                backbone_class=encoder_backbone_class,
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                dropout=dropout,
                act=act,
            )

        else:
            # =================================================================
            # STANDARD ENCODER FOR GAE
            # =================================================================
            # GAE uses a simple encoder that outputs node embeddings directly.
            # We use the PyG BasicGNN directly as the encoder.
            # =================================================================

            encoder = encoder_backbone_class(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                out_channels=out_channels,
                dropout=dropout,
                act=act,
            )

        return encoder

    def _create_adversarial_discriminator(
        self, in_channels: int, hidden_channels: int, out_channels: int
    ) -> torch.nn.Module:
        """
        Dynamically create a discriminator module for ARGA/ARGVA models.

        The discriminator is an MLP that takes latent space embeddings and
        outputs a scalar for GAN discrimination. It distinguishes between
        embeddings from the encoder and random samples from a prior distribution.

        DYNAMIC: Creates discriminator with configurable architecture via parameters
        PRODUCTION-READY: Architecture follows PyG official examples
        FUTURE-PROOF: Works with any latent dimension

        Architecture based on PyG examples/argva_node_clustering.py:
            class Discriminator(torch.nn.Module):
                def __init__(self, in_channels, hidden_channels, out_channels):
                    self.lin1 = Linear(in_channels, hidden_channels)
                    self.lin2 = Linear(hidden_channels, hidden_channels)
                    self.lin3 = Linear(hidden_channels, out_channels)
                def forward(self, x):
                    x = F.relu(self.lin1(x))
                    x = F.relu(self.lin2(x))
                    return self.lin3(x)

        Note: The discriminator output is passed through sigmoid in ARGA's
        reg_loss() and discriminator_loss() methods:
            real = torch.sigmoid(self.discriminator(torch.randn_like(z)))
            fake = torch.sigmoid(self.discriminator(z.detach()))

        Args:
            in_channels: Input dimension (latent space dimension from encoder)
            hidden_channels: Hidden layer dimension for the MLP
            out_channels: Output dimension before final projection to scalar
                         (following PyG example architecture)

        Returns:
            torch.nn.Module: Discriminator module ready for ARGA/ARGVA

        Example:
            >>> discriminator = factory._create_adversarial_discriminator(
            ...     in_channels=32,      # Latent dim from encoder
            ...     hidden_channels=64,  # Hidden layer size
            ...     out_channels=32      # Intermediate output before sigmoid
            ... )
            >>> model = ARGA(encoder, discriminator)
        """
        import torch.nn.functional as F

        class Discriminator(torch.nn.Module):
            """
            Discriminator MLP for adversarial autoencoder models (ARGA/ARGVA).

            Takes latent space embeddings z and outputs a score that is passed
            through sigmoid in ARGA's loss functions to distinguish real prior
            samples from encoded graph embeddings.

            Architecture follows PyG examples/argva_node_clustering.py:
            - 3-layer MLP with ReLU activations
            - Input: latent embeddings [num_nodes, in_channels]
            - Output: scores [num_nodes, out_channels]

            The output is then processed by ARGA.reg_loss() and
            ARGA.discriminator_loss() which apply sigmoid internally.
            """

            def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
                super().__init__()

                # 3-layer MLP architecture following PyG examples
                self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
                self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
                self.lin3 = torch.nn.Linear(hidden_channels, out_channels)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """
                Forward pass through the discriminator.

                Args:
                    x: Latent embeddings [num_nodes, in_channels]
                       Can be either encoder output z or random prior samples

                Returns:
                    Discrimination scores [num_nodes, out_channels]
                    (sigmoid applied in ARGA.reg_loss() / discriminator_loss())
                """
                x = F.relu(self.lin1(x))
                x = F.relu(self.lin2(x))
                return self.lin3(x)

            def reset_parameters(self):
                """Reset all learnable parameters."""
                self.lin1.reset_parameters()
                self.lin2.reset_parameters()
                self.lin3.reset_parameters()

        discriminator = Discriminator(
            in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels
        )

        return discriminator

    @staticmethod
    def _count_parameters(model: torch.nn.Module) -> int:
        """
        Count trainable parameters in model.

        Args:
            model: PyTorch model

        Returns:
            Number of trainable parameters

        Example:
            >>> model = GCN(in_channels=16, hidden_channels=64, num_layers=3, out_channels=1)
            >>> param_count = ModelFactory._count_parameters(model)
            >>> print(f"Model has {param_count:,} parameters")
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # -------
    def get_model_info(self, name: str) -> dict[str, Any] | None:
        """
        Get comprehensive information about a model.

        Args:
            name: Model name or mode ('ensemble', 'custom')

        Returns:
            Dictionary with model information, or None if not found

        Example:
            >>> factory = ModelFactory()
            >>> info = factory.get_model_info("GCN")
            >>> print(info['description'])
            >>> print(info['supported_tasks'])
        """
        name_lower = name.lower()

        # Handle special modes (same pattern as create_model)
        if name_lower == "ensemble":
            return {
                "name": name,
                "class": "EnsembleModel",
                "description": "Ensemble of multiple models with configurable fusion strategy",
                "category": "ensemble",
                "supported_tasks": [
                    "graph_regression",
                    "graph_classification",
                    "node_regression",
                    "node_classification",
                ],
                "hyperparameters": {},
                "requires_edge_features": False,
                "requires_edge_weights": False,
                "supports_heterogeneous": False,
                "supports_directed": True,
                "is_builtin": False,
                "plugin_name": None,
                "paper_url": None,
                "paper_title": None,
                "tags": ["ensemble", "multi-model"],
                "min_pyg_version": None,
            }

        if name_lower == "custom":
            return {
                "name": name,
                "class": "CustomArchitecture",
                "description": "Custom architecture built from configuration",
                "category": "custom",
                "supported_tasks": [
                    "graph_regression",
                    "graph_classification",
                    "node_regression",
                    "node_classification",
                    "link_prediction",
                    "edge_regression",
                ],
                "hyperparameters": {},
                "requires_edge_features": False,
                "requires_edge_weights": False,
                "supports_heterogeneous": False,
                "supports_directed": True,
                "is_builtin": False,
                "plugin_name": None,
                "paper_url": None,
                "paper_title": None,
                "tags": ["custom", "configurable"],
                "min_pyg_version": None,
            }

        # Standard registry lookup for all other models
        if not self.registry.has_model(name):
            return None

        registration = self.registry._models[name]
        metadata = registration.metadata

        return {
            "name": name,
            "class": registration.model_class.__name__,
            "description": metadata.description,
            "category": metadata.category.value,
            "supported_tasks": metadata.supported_tasks,
            "hyperparameters": metadata.hyperparameters,
            "requires_edge_features": metadata.requires_edge_features,
            "requires_edge_weights": metadata.requires_edge_weights,
            "supports_heterogeneous": metadata.supports_heterogeneous,
            "supports_directed": metadata.supports_directed,
            "is_builtin": registration.is_builtin,
            "plugin_name": registration.plugin_name,
            "paper_url": metadata.paper_url,
            "paper_title": metadata.paper_title,
            "tags": metadata.tags,
            "min_pyg_version": metadata.min_pyg_version,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS (Public API)
# =============================================================================

# Global factory instance
_factory = None


def get_factory() -> ModelFactory:
    """
    Get singleton ModelFactory instance.

    Returns:
        ModelFactory singleton

    Example:
        >>> factory = get_factory()
        >>> model = factory.create_model("GCN", {...}, "graph_regression")
    """
    global _factory
    if _factory is None:
        _factory = ModelFactory()
    return _factory


def create_model(
    name: str,
    hyperparameters: dict[str, Any],
    task_type: str,
    sample_data: Data | None = None,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """
    Convenience function to create a model.

    PHASE 7 EXTENSION: Now supports custom architectures and ensembles.

    Args:
        name: Model name ("GCN", "GAT", "custom", "ensemble", etc.)
        hyperparameters: Hyperparameter dictionary
        task_type: Task type
        sample_data: Sample data for inference
        device: Target device

    Returns:
        Instantiated model

    Example:
        >>> # Standard model
        >>> from milia_pipeline.models import create_model
        >>> model = create_model(
        ...     name="GCN",
        ...     hyperparameters={"hidden_channels": 64, "num_layers": 3},
        ...     task_type="graph_regression",
        ...     sample_data=sample_data
        ... )
        >>>
        >>> # Custom architecture
        >>> model = create_model(
        ...     name="custom",
        ...     hyperparameters={"architecture_config": config},
        ...     task_type="graph_regression"
        ... )
        >>>
        >>> # Ensemble
        >>> model = create_model(
        ...     name="ensemble",
        ...     hyperparameters={"ensemble_config": config},
        ...     task_type="graph_regression"
        ... )
    """
    factory = get_factory()
    return factory.create_model(
        name=name,
        hyperparameters=hyperparameters,
        task_type=task_type,
        sample_data=sample_data,
        device=device,
    )


def get_model_info(name: str) -> dict[str, Any] | None:
    """
    Get information about a model.

    Args:
        name: Model name

    Returns:
        Model information dictionary or None

    Example:
        >>> from milia_pipeline.models import get_model_info
        >>> info = get_model_info("GCN")
        >>> print(info['description'])
    """
    factory = get_factory()
    return factory.get_model_info(name)


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

# Log module load
logger.info("model_factory module loaded (Phase 7 extended, Phase 4 dynamic introspection)")
if _BUILDERS_AVAILABLE:
    logger.info("Builders module available: custom architectures and ensembles supported")
else:
    logger.info("Builders module not available: only standard models supported")
