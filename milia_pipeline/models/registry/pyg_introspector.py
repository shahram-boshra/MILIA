"""
PyG Model Introspector

DYNAMIC, PRODUCTION-READY, FUTURE-PROOF introspection of PyG models.
Replaces all hardcoded model definitions with runtime discovery.

Pydantic V2 Migration (Phase 23):
    - Migrated ParameterInfo from @dataclass to Pydantic BaseModel (mutable)
    - Migrated DynamicModelMetadata from @dataclass to Pydantic BaseModel (mutable)
    - Uses Field(default_factory=...) for mutable default types (list, dict, set)
    - Uses ConfigDict(arbitrary_types_allowed=True) for ParameterInfo nested types
    - Uses model_dump(mode='json') for enum serialization in to_dict()
    - Added to_dict() methods wrapping model_dump() for backward compatibility
    - NON-BREAKING: Same constructor API and attribute access preserved

Author: Milia Team
Version: 2.1.0
"""

import importlib
import inspect
import logging
from enum import Enum
from typing import Any, Union, get_args, get_origin, get_type_hints

import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# =============================================================================
# BACKWARD COMPATIBLE IMPORTS WITH FALLBACK
# =============================================================================

# Import original ModelCategory enum for backward compatibility
try:
    from .model_categories import ModelCategory

    # Add GENERAL if not present (for dynamically discovered models)
    if not hasattr(ModelCategory, "GENERAL"):
        # Cannot modify existing Enum, so we'll use UTILITY as fallback
        _GENERAL_FALLBACK = ModelCategory.UTILITY
    else:
        _GENERAL_FALLBACK = ModelCategory.GENERAL
except ImportError:

    class ModelCategory(Enum):
        """Fallback ModelCategory enum if original not available."""

        BASIC_GNN = "basic_gnn"
        CONVOLUTIONAL = "convolutional"
        ATTENTION = "attention"
        POOLING = "pooling"
        AGGREGATION = "aggregation"
        ENCODER = "encoder"
        AUTOENCODER = "autoencoder"
        TRANSFORMER = "transformer"
        TEMPORAL = "temporal"
        META_LEARNING = "meta_learning"
        EXPLAINABILITY = "explainability"
        UTILITY = "utility"
        GENERAL = "general"  # For dynamically discovered models

    _GENERAL_FALLBACK = ModelCategory.GENERAL


# =============================================================================
# DYNAMIC MODEL DISCOVERY
# =============================================================================


def discover_pyg_models() -> dict[str, str]:
    """
    Dynamically discover ALL available models from PyG.

    Based on PyG documentation (pytorch-geometric.readthedocs.io):
    - Models are in torch_geometric.nn.models (GCN, GAT, GraphSAGE, etc.)
    - Convolutions are in torch_geometric.nn.conv (GCNConv, GATConv, etc.)

    Returns:
        Dict mapping model_name -> import_path

    Example:
        >>> models = discover_pyg_models()
        >>> print(models['GCN'])
        'torch_geometric.nn.models.GCN'
    """
    discovered = {}

    # ==========================================================================
    # PRIMARY: torch_geometric.nn.models (high-level model classes)
    # Evidence: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GCN.html
    # ==========================================================================
    try:
        import torch_geometric.nn.models as models_module

        for name in dir(models_module):
            if name.startswith("_"):
                continue
            obj = getattr(models_module, name)
            if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
                discovered[name] = f"torch_geometric.nn.models.{name}"
    except ImportError as e:
        logger.warning(f"Could not import torch_geometric.nn.models: {e}")

    # ==========================================================================
    # SECONDARY: Scan submodules of torch_geometric.nn.models
    # These contain specialized models (autoencoders, temporal, etc.)
    # ==========================================================================
    model_submodules = [
        "torch_geometric.nn.models.autoencoder",
        "torch_geometric.nn.models.deep_graph_infomax",
        "torch_geometric.nn.models.metapath2vec",
        "torch_geometric.nn.models.node2vec",
        "torch_geometric.nn.models.jumping_knowledge",
        "torch_geometric.nn.models.signed_gcn",
        "torch_geometric.nn.models.re_net",
        "torch_geometric.nn.models.tgn",
    ]

    for module_path in model_submodules:
        try:
            module = importlib.import_module(module_path)
            for name in dir(module):
                if name.startswith("_"):
                    continue
                obj = getattr(module, name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, nn.Module)
                    and obj is not nn.Module
                    and name not in discovered
                ):  # Avoid duplicates
                    discovered[name] = f"{module_path}.{name}"
        except ImportError as e:
            logger.debug(f"Could not import {module_path}: {e}")

    # ==========================================================================
    # ALSO DISCOVER: Models exposed directly in torch_geometric.nn
    # Some models are re-exported at the top level
    # ==========================================================================
    try:
        import torch_geometric.nn as nn_module

        for name in ["GCN", "GAT", "GraphSAGE", "GIN", "PNA", "EdgeCNN"]:
            if hasattr(nn_module, name) and name not in discovered:
                obj = getattr(nn_module, name)
                if isinstance(obj, type) and issubclass(obj, nn.Module):
                    discovered[name] = f"torch_geometric.nn.{name}"
    except ImportError:
        pass

    logger.info(f"Discovered {len(discovered)} PyG models dynamically")
    return discovered


# =============================================================================
# DYNAMIC SIGNATURE INTROSPECTION
# =============================================================================


class ParameterInfo(BaseModel):
    """
    Information about a model parameter extracted via introspection.

    Pydantic V2 Migration (Phase 23):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses Field(default=...) for optional fields with defaults
        - NON-BREAKING: Same constructor API and attribute access
    """

    name: str
    param_type: str  # 'int', 'float', 'bool', 'str', 'optional' (no hint, default=None), 'any'
    required: bool
    default: Any = None
    description: str = ""
    # For numeric types
    min_value: float | None = None
    max_value: float | None = None
    # For categorical types
    choices: list[Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Backward compatible method wrapping Pydantic V2's model_dump().
        """
        return self.model_dump()


def introspect_model_signature(model_class: type) -> dict[str, ParameterInfo]:
    """
    Dynamically introspect a model's __init__ signature.

    Args:
        model_class: The model class to introspect

    Returns:
        Dict mapping parameter_name -> ParameterInfo

    Example:
        >>> from torch_geometric.nn.models import GCN
        >>> params = introspect_model_signature(GCN)
        >>> print(params['hidden_channels'])
        ParameterInfo(name='hidden_channels', param_type='int', required=True, ...)
    """
    params = {}

    try:
        sig = inspect.signature(model_class.__init__)
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not get signature for {model_class}: {e}")
        return params

    # Try to get type hints
    try:
        hints = get_type_hints(model_class.__init__)
    except Exception:
        hints = {}

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "args", "kwargs"):
            continue

        # Determine if required
        required = param.default is inspect.Parameter.empty
        default = None if required else param.default

        # Infer type from annotation or default value
        param_type = _infer_param_type(param_name, param, hints)

        params[param_name] = ParameterInfo(
            name=param_name,
            param_type=param_type,
            required=required,
            default=default,
            description=_extract_param_description(model_class, param_name),
        )

    return params


def model_accepts_kwargs(model_class: type) -> bool:
    """
    Check if a model's __init__ accepts **kwargs.

    DYNAMIC: Uses Python's inspect module to detect VAR_KEYWORD parameters
    PRODUCTION-READY: Handles edge cases (missing __init__, no signature)
    FUTURE-PROOF: Works with ANY PyG model regardless of signature

    This is critical for determining whether a model can receive additional
    parameters beyond those explicitly defined in its signature. PyG BasicGNN
    models (GCN, GAT, GraphSAGE, GIN) accept **kwargs which are passed through
    to their underlying Conv layers (GCNConv, GATConv, etc.).

    Evidence from PyG source (basic_gnn.py):
        class GCN(BasicGNN):
            '''**kwargs (optional): Additional arguments of GCNConv.'''
            def init_conv(self, in_channels, out_channels, **kwargs):
                return GCNConv(in_channels, out_channels, **kwargs)

    Args:
        model_class: The model class to introspect

    Returns:
        True if model accepts **kwargs, False otherwise

    Example:
        >>> from torch_geometric.nn.models import GCN
        >>> model_accepts_kwargs(GCN)
        True
        >>> from torch.nn import Linear
        >>> model_accepts_kwargs(Linear)
        False
    """
    try:
        sig = inspect.signature(model_class.__init__)
    except (ValueError, TypeError) as e:
        logger.debug(f"Could not get signature for {model_class}: {e}")
        return False

    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())


# =============================================================================
# KNOWN CONV LAYER KWARGS
# =============================================================================
# These are parameters that can be passed through **kwargs to underlying
# Conv layers in PyG BasicGNN models. This list is based on the official
# PyG documentation and source code.
#
# Evidence:
# - GCNConv: add_self_loops, normalize, improved, cached, bias
#   https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html
# - GATConv: add_self_loops, negative_slope, dropout, bias, heads, concat, edge_dim, fill_value, residual
#   https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html
# - SAGEConv: normalize, bias, aggr, project, root_weight (NO add_self_loops!)
#   https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html
# - GINConv: eps, train_eps
#   https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINConv.html
#
# DYNAMIC: This set is comprehensive but not exhaustive - unknown kwargs
#          will be filtered out by the model itself if not applicable.
# PRODUCTION-READY: Based on official PyG documentation.
# FUTURE-PROOF: New conv kwargs can be added to this set without code changes.
# =============================================================================
KNOWN_CONV_KWARGS = {
    # Self-loop control (critical for edge_attr compatibility)
    "add_self_loops",
    # Normalization
    "normalize",
    "improved",  # GCN improved formulation
    # Caching
    "cached",
    # Bias
    "bias",
    # GAT-specific
    "negative_slope",
    "heads",
    "concat",
    "fill_value",  # For self-loop edge features
    "residual",  # Residual connections
    # SAGE-specific
    "aggr",
    "project",
    "root_weight",
    # GIN-specific
    "eps",
    "train_eps",
    # Edge handling
    "edge_dim",
    # MessagePassing base kwargs
    "flow",
    "node_dim",
}

# =============================================================================
# MODEL-SPECIFIC CONV KWARGS MAPPING (Fix 19)
# =============================================================================
# Each model's Conv layer accepts only specific kwargs. Passing an invalid kwarg
# (e.g., normalize=True to GAT) causes MessagePassing.__init__() to fail with:
# "got an unexpected keyword argument"
#
# This mapping specifies EXACTLY which kwargs each model's Conv layer accepts,
# based on official PyG documentation.
#
# DYNAMIC: Uses model name to filter kwargs at runtime
# PRODUCTION-READY: Based on official PyG 2.6 documentation
# FUTURE-PROOF: New models can be added by extending this mapping
# =============================================================================
MODEL_SPECIFIC_CONV_KWARGS = {
    # GCNConv: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html
    "GCN": {
        "add_self_loops",
        "normalize",
        "improved",
        "cached",
        "bias",
    },
    # GATConv: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html
    # NOTE: GATConv does NOT have 'normalize' parameter!
    "GAT": {
        "add_self_loops",
        "negative_slope",
        "heads",
        "concat",
        "edge_dim",
        "fill_value",
        "bias",
        "residual",
    },
    # SAGEConv: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html
    # NOTE: SAGEConv DOES have 'normalize' but NOT 'add_self_loops'!
    "GraphSAGE": {
        "normalize",
        "aggr",
        "project",
        "root_weight",
        "bias",
    },
    # GINConv: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINConv.html
    "GIN": {
        "eps",
        "train_eps",
    },
    # PNA: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.PNAConv.html
    # PNA requires aggregators, scalers, and deg - these are MANDATORY for PNAConv
    # They pass through BasicGNN's **kwargs to PNAConv.init_conv()
    # NOTE: PNA does NOT support add_self_loops (unlike GCN/GAT/GraphSAGE)
    "PNA": {
        "edge_dim",
        "aggregators",  # Required: List[str] - e.g., ['mean', 'min', 'max', 'std']
        "scalers",  # Required: List[str] - e.g., ['identity', 'amplification', 'attenuation']
        "deg",  # Required: Tensor - degree histogram from training data
        "towers",  # Optional: Number of towers (default: 1)
        "pre_layers",  # Optional: Transformation layers before aggregation (default: 1)
        "post_layers",  # Optional: Transformation layers after aggregation (default: 1)
        "divide_input",  # Optional: Whether to split input between towers (default: False)
        "train_norm",  # Optional: Whether normalization params are trainable (default: False)
    },
}


def get_model_conv_kwargs(model_name: str) -> set:
    """
    Get the set of valid conv kwargs for a specific model.

    DYNAMIC: Returns model-specific kwargs if known, falls back to full set
    PRODUCTION-READY: Prevents invalid kwargs from reaching Conv layers
    FUTURE-PROOF: New models automatically get full set (safe fallback)

    Args:
        model_name: Model name (e.g., 'GCN', 'GAT', 'GraphSAGE')

    Returns:
        Set of valid conv kwargs for this model
    """
    # Normalize model name for lookup (fully uppercase for case-insensitive comparison)
    normalized_name = model_name.upper()

    # Check various name formats
    for name, kwargs in MODEL_SPECIFIC_CONV_KWARGS.items():
        if name.upper() == normalized_name or normalized_name.startswith(name.upper()):
            return kwargs

    # Fall back to full set for unknown models (safe but may cause issues)
    logger.debug(
        f"No model-specific conv kwargs for '{model_name}', using full KNOWN_CONV_KWARGS set"
    )
    return KNOWN_CONV_KWARGS


def introspect_forward_signature(model_class: type) -> dict[str, ParameterInfo]:
    """
    Dynamically introspect a model's forward() method signature.

    DYNAMIC: Uses Python's inspect module to discover forward parameters at runtime
    PRODUCTION-READY: Handles edge cases (missing forward, no signature)
    FUTURE-PROOF: Works with ANY PyG model regardless of forward signature

    This is critical for supporting models with non-standard forward signatures:
    - SchNet: forward(z, pos, batch) - requires atomic numbers and positions
    - DimeNet: forward(z, pos, batch) - similar to SchNet
    - GCN: forward(x, edge_index, ...) - standard GNN signature
    - Custom models: any signature

    Args:
        model_class: The model class to introspect

    Returns:
        Dict mapping parameter_name -> ParameterInfo for forward() parameters

    Example:
        >>> from torch_geometric.nn.models import SchNet
        >>> params = introspect_forward_signature(SchNet)
        >>> print(params.keys())
        dict_keys(['z', 'pos', 'batch'])
        >>> print(params['z'].required)
        True
        >>> print(params['pos'].required)
        True
    """
    params = {}

    # Check if model has forward method
    if not hasattr(model_class, "forward"):
        logger.debug(f"{model_class.__name__} has no forward method")
        return params

    try:
        sig = inspect.signature(model_class.forward)
    except (ValueError, TypeError) as e:
        logger.debug(f"Could not get forward signature for {model_class.__name__}: {e}")
        return params

    # Try to get type hints for forward method
    try:
        hints = get_type_hints(model_class.forward)
    except Exception:
        hints = {}

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "args", "kwargs"):
            continue

        # Determine if required (no default value)
        required = param.default is inspect.Parameter.empty
        default = None if required else param.default

        # Infer type from annotation or default value
        param_type = _infer_param_type(param_name, param, hints)

        params[param_name] = ParameterInfo(
            name=param_name,
            param_type=param_type,
            required=required,
            default=default,
            description=_extract_forward_param_description(model_class, param_name),
        )

    return params


def get_required_data_attributes(model_class: type) -> set[str]:
    """
    Get the data attributes required by a model's forward method.

    DYNAMIC: Introspects forward signature to determine requirements
    PRODUCTION-READY: Maps forward params to data attributes
    FUTURE-PROOF: Works with any model

    This maps forward method parameters to PyG Data attributes:
    - z -> data.z (atomic numbers)
    - pos -> data.pos (positions)
    - x -> data.x (node features)
    - edge_index -> data.edge_index (graph connectivity)
    - edge_attr -> data.edge_attr (edge features)
    - batch -> data.batch (batch indices)

    Args:
        model_class: The model class to introspect

    Returns:
        Set of required data attribute names

    Example:
        >>> from torch_geometric.nn.models import SchNet
        >>> attrs = get_required_data_attributes(SchNet)
        >>> print(attrs)
        {'z', 'pos'}  # batch is optional, so not included
    """
    forward_params = introspect_forward_signature(model_class)

    required_attrs = set()

    # Map forward parameter names to data attribute names
    # Most are direct mappings, but some may need translation
    param_to_data_attr = {
        "z": "z",
        "pos": "pos",
        "x": "x",
        "edge_index": "edge_index",
        "edge_attr": "edge_attr",
        "edge_weight": "edge_weight",
        # batch is always available from DataLoader, not from Data object directly
    }

    for param_name, param_info in forward_params.items():
        # Only include required parameters (no default)
        if param_info.required:
            # Map to data attribute name
            data_attr = param_to_data_attr.get(param_name, param_name)
            # Skip 'batch' as it's provided by DataLoader, not Data
            if data_attr != "batch":
                required_attrs.add(data_attr)

    return required_attrs


def _extract_forward_param_description(model_class: type, param_name: str) -> str:
    """Extract parameter description from forward method docstring if available."""
    forward_method = getattr(model_class, "forward", None)
    if forward_method is None:
        return ""

    docstring = forward_method.__doc__ or ""

    # Simple extraction - look for param_name in docstring
    for line in docstring.split("\n"):
        if param_name in line:
            # Clean up and return
            return line.strip().replace(param_name, "").strip(": ").strip()

    return ""


def _infer_param_type(param_name: str, param: inspect.Parameter, hints: dict[str, Any]) -> str:
    """Infer parameter type from hints, default, or name conventions."""

    # Check type hints first
    if param_name in hints:
        hint = hints[param_name]
        return _type_hint_to_string(hint)

    # Infer from default value
    if param.default is not inspect.Parameter.empty:
        default = param.default
        if isinstance(default, bool):
            return "bool"
        elif isinstance(default, int):
            return "int"
        elif isinstance(default, float):
            return "float"
        elif isinstance(default, str):
            return "str"
        elif default is None:
            return "optional"

    # Infer from parameter name conventions
    if any(x in param_name.lower() for x in ["channels", "dim", "size", "num", "layers", "heads"]):
        return "int"
    if any(x in param_name.lower() for x in ["dropout", "ratio", "rate", "alpha", "eps"]):
        return "float"
    if any(x in param_name.lower() for x in ["use_", "is_", "has_", "enable", "normalize"]):
        return "bool"

    return "any"


def _type_hint_to_string(hint: Any) -> str:
    """Convert type hint to string representation.

    Properly unwraps Optional[X] (== Union[X, None]) to extract the inner
    type X, so that Optional[int] → "int", Optional[float] → "float", etc.

    Uses typing.get_origin / typing.get_args (Python 3.8+) for robust
    introspection instead of fragile string matching.

    PRODUCTION-READY: Handles Optional[X], Union[X, None], X | None (3.10+)
    FUTURE-PROOF: Works with any inner type, including custom types
    NON-BREAKING: Returns the same strings as before for non-Optional hints
    """
    # -------------------------------------------------------------------------
    # Step 1: Unwrap Optional[X] / Union[X, None] / X | None
    # -------------------------------------------------------------------------
    # Optional[X] is equivalent to Union[X, None]. When detected, extract
    # the non-None inner type and recurse so that Optional[int] → "int",
    # Optional[float] → "float", Optional[str] → "str", etc.
    #
    # Evidence: Python docs (typing module):
    #   get_origin(Optional[int]) returns typing.Union
    #   get_args(Optional[int]) returns (int, NoneType)
    #
    # Also handles Python 3.10+ union syntax: int | None
    #   get_origin(int | None) returns types.UnionType
    #   get_args(int | None) returns (int, NoneType)
    # -------------------------------------------------------------------------
    origin = get_origin(hint)
    if origin is Union:
        # Filter out NoneType to get the "real" type(s)
        args = [a for a in get_args(hint) if a is not type(None)]
        if len(args) == 1:
            # Simple Optional[X] — recurse to resolve the inner type
            return _type_hint_to_string(args[0])
        elif len(args) > 1:
            # Union of multiple non-None types (e.g., Union[int, str])
            # Return the first concrete type as best effort
            return _type_hint_to_string(args[0])
        else:
            # Union[None] edge case — effectively NoneType only
            return "any"

    # Python 3.10+ union syntax: int | None creates types.UnionType
    # which has a different origin than typing.Union
    try:
        import types as _types

        if isinstance(hint, _types.UnionType):
            args = [a for a in get_args(hint) if a is not type(None)]
            if len(args) == 1 or len(args) > 1:
                return _type_hint_to_string(args[0])
            else:
                return "any"
    except AttributeError:
        pass  # Python < 3.10, no UnionType

    # -------------------------------------------------------------------------
    # Step 2: Match concrete primitive types
    # -------------------------------------------------------------------------
    # Check the actual type object first (most reliable), then fall back
    # to string matching for complex or stringified annotations.
    # -------------------------------------------------------------------------
    if hint is int:
        return "int"
    if hint is float:
        return "float"
    if hint is bool:
        return "bool"
    if hint is str:
        return "str"

    # Fallback: string-based matching for complex annotations
    hint_str = str(hint).lower()
    if "int" in hint_str:
        return "int"
    elif "float" in hint_str:
        return "float"
    elif "bool" in hint_str:
        return "bool"
    elif "str" in hint_str:
        return "str"
    else:
        return "any"


def _extract_param_description(model_class: type, param_name: str) -> str:
    """Extract parameter description from docstring if available."""
    docstring = model_class.__init__.__doc__ or model_class.__doc__ or ""

    # Simple extraction - look for param_name in docstring
    for line in docstring.split("\n"):
        if param_name in line:
            # Clean up and return
            return line.strip().replace(param_name, "").strip(": ").strip()

    return ""


# =============================================================================
# DYNAMIC METADATA GENERATION
# =============================================================================

# NOTE: ModelCategory is imported at the top of file (lines 24-47)
# Do not re-import here


class DynamicModelMetadata(BaseModel):
    """
    Dynamically generated model metadata.

    BACKWARD COMPATIBLE with original ModelMetadata from model_categories.py.
    All fields match the original dataclass to ensure drop-in replacement.

    Pydantic V2 Migration (Phase 23):
        - Migrated from @dataclass to Pydantic BaseModel (mutable)
        - Uses Field(default_factory=...) for mutable default types (list, dict, set)
        - Uses ConfigDict(arbitrary_types_allowed=True) for ParameterInfo nested types
        - Uses model_dump(mode='json') in to_dict() for enum value serialization
        - NON-BREAKING: Same constructor API and attribute access
    """

    # Allow arbitrary types for ParameterInfo in Dict fields
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    category: ModelCategory  # Enum for backward compatibility
    import_path: str
    description: str = ""
    paper_url: str | None = None
    paper_title: str | None = None

    # CRITICAL: These fields match original ModelMetadata exactly
    supported_tasks: list[str] = Field(default_factory=list)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)  # Dict format, NOT ParameterInfo

    # Requirements
    requires_edge_features: bool = False
    requires_edge_weights: bool = False
    requires_edge_index: bool = True
    supports_heterogeneous: bool = False
    supports_directed: bool = True

    # Version and tags
    min_pyg_version: str = "2.0.0"
    tags: list[str] = Field(default_factory=list)

    # Additional dynamic fields (not in original, but useful)
    parameters: dict[str, ParameterInfo] = Field(
        default_factory=dict
    )  # Detailed __init__ param info

    # =========================================================================
    # NEW: Forward signature information for model-data compatibility
    # =========================================================================
    # DYNAMIC: Introspected from model.forward() at runtime
    # PRODUCTION-READY: Used for data compatibility validation
    # FUTURE-PROOF: Works with any model forward signature (SchNet, DimeNet, GCN, etc.)
    # =========================================================================
    forward_parameters: dict[str, ParameterInfo] = Field(default_factory=dict)  # forward() params
    required_data_attributes: set[str] = Field(
        default_factory=set
    )  # Required data attrs (z, pos, x, etc.)

    # =========================================================================
    # NEW: Kwargs acceptance for Conv layer parameter passthrough
    # =========================================================================
    # DYNAMIC: Introspected from model.__init__ at runtime
    # PRODUCTION-READY: Enables config parameters like add_self_loops to pass through
    # FUTURE-PROOF: Works with any PyG BasicGNN model that accepts **kwargs
    #
    # Evidence from PyG source (basic_gnn.py):
    #     class GCN(BasicGNN):
    #         '''**kwargs (optional): Additional arguments of GCNConv.'''
    # =========================================================================
    accepts_kwargs: bool = False  # Whether model accepts **kwargs for conv layer params

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Backward compatible method using Pydantic V2's model_dump(mode='json')
        for automatic enum value extraction (ModelCategory -> string).

        Returns:
            Dictionary with:
            - 'category': enum value string (e.g., 'basic_gnn', 'attention')
            - All other fields serialized appropriately
        """
        return self.model_dump(mode="json")


def generate_model_metadata(model_name: str, import_path: str) -> DynamicModelMetadata | None:
    """
    Dynamically generate complete metadata for a model.

    BACKWARD COMPATIBLE: Returns metadata matching original ModelMetadata structure.
    DYNAMIC: Introspects both __init__ and forward() signatures
    PRODUCTION-READY: Includes required_data_attributes for validation
    FUTURE-PROOF: Works with any model (SchNet, DimeNet, GCN, etc.)

    Args:
        model_name: Name of the model (e.g., 'GCN')
        import_path: Full import path (e.g., 'torch_geometric.nn.models.GCN')

    Returns:
        DynamicModelMetadata or None if model cannot be loaded
    """
    try:
        # Dynamic import
        module_path, class_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)

        # Introspect __init__ signature
        parameters = introspect_model_signature(model_class)

        # =====================================================================
        # NEW: Introspect forward() signature for model-data compatibility
        # =====================================================================
        forward_parameters = introspect_forward_signature(model_class)
        required_data_attrs = get_required_data_attributes(model_class)

        # =====================================================================
        # NEW: Check if model accepts **kwargs for Conv layer parameters
        # =====================================================================
        # This is critical for allowing parameters like add_self_loops to pass
        # through to underlying Conv layers in PyG BasicGNN models.
        # =====================================================================
        accepts_kwargs_flag = model_accepts_kwargs(model_class)
        if accepts_kwargs_flag:
            logger.debug(f"{model_name} accepts **kwargs - conv layer params can pass through")

        # Extract description from docstring
        description = (model_class.__doc__ or "").split("\n")[0].strip()

        # Infer capabilities from parameters
        requires_edge_features = any(
            "edge" in p.lower() and any(x in p.lower() for x in ["dim", "channels", "features"])
            for p in parameters
        )
        requires_edge_weights = "edge_weight" in parameters

        # Infer category from import path (returns ModelCategory enum)
        category = _infer_category_enum(import_path, model_name)

        # Convert ParameterInfo dict to hyperparameters dict format (BACKWARD COMPATIBLE)
        hyperparameters = _parameters_to_hyperparameters_dict(parameters)

        # Infer supported tasks based on model characteristics
        supported_tasks = _infer_supported_tasks(model_name, import_path, parameters)

        # Infer tags from model name and characteristics
        tags = _infer_tags(model_name, import_path, parameters)

        return DynamicModelMetadata(
            name=model_name,
            category=category,
            import_path=import_path,
            description=description,
            supported_tasks=supported_tasks,
            hyperparameters=hyperparameters,
            requires_edge_features=requires_edge_features,
            requires_edge_weights=requires_edge_weights,
            tags=tags,
            parameters=parameters,  # Keep detailed __init__ info for advanced use
            forward_parameters=forward_parameters,  # NEW: forward() params
            required_data_attributes=required_data_attrs,  # NEW: Required data attrs
            accepts_kwargs=accepts_kwargs_flag,  # NEW: Whether model accepts **kwargs
        )

    except Exception as e:
        logger.warning(f"Could not generate metadata for {model_name}: {e}")
        return None


def _infer_intelligent_default(param_name: str, param_type: str) -> Any | None:
    """
    Infer intelligent default values for required parameters based on name patterns.

    DYNAMIC: Uses pattern matching on parameter names - no hardcoded model names
    PRODUCTION-READY: Based on established defaults from original paper implementations
    FUTURE-PROOF: Pattern-based approach works for any new model with similar naming

    Evidence-based defaults from PyG documentation and original papers:
    - DimeNet (ICLR 2020): num_blocks=6, num_bilinear=8, num_spherical=7, num_radial=6
    - SchNet: num_filters=128, num_gaussians=50, num_interactions=6
    - PaiNN: num_interactions=3, num_rbf=20
    - EGNN: num_layers=4

    Args:
        param_name: Name of the parameter (e.g., 'num_blocks', 'hidden_channels')
        param_type: Type of the parameter ('int', 'float', 'bool', 'str')

    Returns:
        Intelligent default value, or None if no pattern matches

    Example:
        >>> _infer_intelligent_default('num_blocks', 'int')
        6
        >>> _infer_intelligent_default('num_spherical', 'int')
        7
        >>> _infer_intelligent_default('hidden_channels', 'int')
        128
    """
    param_lower = param_name.lower()

    # =========================================================================
    # INTEGER PARAMETERS
    # =========================================================================
    if param_type == "int":
        # ---------------------------------------------------------------------
        # Architectural block/layer counts (evidence: DimeNet, SchNet papers)
        # ---------------------------------------------------------------------
        if param_lower == "num_blocks":
            return 6  # DimeNet default from original paper
        if param_lower == "num_layers" or param_lower == "n_layers":
            return 3  # Common default for GNN depth
        if param_lower in ("num_interactions", "n_interactions", "n_interaction_blocks"):
            return 6  # SchNet default

        # ---------------------------------------------------------------------
        # Basis function counts (evidence: DimeNet, SchNet papers)
        # ---------------------------------------------------------------------
        if param_lower == "num_radial" or param_lower == "num_rbf" or param_lower == "n_rbf":
            return 6  # DimeNet default (radial basis functions)
        if param_lower == "num_spherical" or param_lower == "num_sbf" or param_lower == "n_sbf":
            return 7  # DimeNet default (spherical harmonics)
        if param_lower == "num_gaussians":
            return 50  # SchNet default (Gaussian basis)

        # ---------------------------------------------------------------------
        # DimeNet-specific parameters (evidence: DimeNet paper, DGL implementation)
        # ---------------------------------------------------------------------
        if param_lower == "num_bilinear":
            return 8  # DimeNet default (bilinear layer tensor size)
        if param_lower in ("num_before_skip", "num_residual_before_skip"):
            return 1  # DimeNet default
        if param_lower in ("num_after_skip", "num_residual_after_skip"):
            return 2  # DimeNet default
        if param_lower == "num_output_layers" or param_lower == "num_dense_output":
            return 3  # DimeNet default

        # ---------------------------------------------------------------------
        # Channel/dimension sizes (evidence: common PyG patterns)
        # ---------------------------------------------------------------------
        if "hidden_channels" in param_lower or "hidden_dim" in param_lower:
            return 128  # Common default for molecular models
        if "emb_size" in param_lower or "embed_size" in param_lower:
            return 128  # DimeNet default embedding size
        if param_lower in ("out_channels", "out_dim", "output_dim"):
            return 1  # Default for regression (single target)
        if "channels" in param_lower or "dim" in param_lower:
            return 64  # General default for hidden dimensions
        if param_lower == "num_filters":
            return 128  # SchNet default

        # ---------------------------------------------------------------------
        # Attention/head counts (evidence: GAT, Transformer patterns)
        # ---------------------------------------------------------------------
        if param_lower == "heads" or param_lower == "num_heads":
            return 4  # Common attention head count

        # ---------------------------------------------------------------------
        # Neighbor/interaction limits
        # ---------------------------------------------------------------------
        if param_lower in ("max_num_neighbors", "max_neighbors", "k"):
            return 32  # DimeNet default
        if param_lower == "envelope_exponent":
            return 5  # DimeNet default

        # ---------------------------------------------------------------------
        # DimeNet++ specific parameters
        # ---------------------------------------------------------------------
        if param_lower == "int_emb_size":
            return 64  # DimeNet++ default
        if param_lower == "basis_emb_size":
            return 8  # DimeNet++ default
        if param_lower == "out_emb_channels":
            return 256  # DimeNet++ default

    # =========================================================================
    # FLOAT PARAMETERS
    # =========================================================================
    elif param_type == "float":
        if "dropout" in param_lower:
            return 0.0  # Conservative default
        if "cutoff" in param_lower:
            return 5.0  # DimeNet/SchNet default cutoff distance (Angstroms)
        if param_lower in ("ratio", "rate"):
            return 0.5  # Common default for ratios
        if param_lower == "eps":
            return 1e-8  # Small epsilon for numerical stability

    # =========================================================================
    # BOOLEAN PARAMETERS
    # =========================================================================
    elif param_type == "bool":
        if "use_" in param_lower or "enable_" in param_lower:
            return True  # Default to enabling features
        if "normalize" in param_lower:
            return True  # Default to normalization
        if param_lower == "bias":
            return True  # Default to using bias

    # =========================================================================
    # STRING PARAMETERS
    # =========================================================================
    elif param_type == "str":
        if "act" in param_lower or "activation" in param_lower:
            return "relu"  # Common default activation
        if "aggr" in param_lower or "aggregation" in param_lower:
            return "add"  # Common default aggregation
        if param_lower == "output_initializer":
            return "zeros"  # DimeNet default

    return None


def _parameters_to_hyperparameters_dict(parameters: dict[str, ParameterInfo]) -> dict[str, Any]:
    """
    Convert ParameterInfo dict to hyperparameters dict format.

    This matches the original model_categories.py format:
    {
        "param_name": {
            "type": "integer",
            "required": True,
            "default": 64,
            "min": 1,
            "description": "..."
        }
    }

    ENHANCED: For required parameters without explicit defaults, this function
    now infers intelligent defaults based on parameter name patterns and
    domain knowledge from original paper implementations.

    This enables models like DimeNet, SchNet, PaiNN to be instantiated
    without requiring users to specify every parameter manually.

    FIX 27: Added handling for module-type parameters (encoder, decoder,
    discriminator) used by autoencoder models. These are marked with
    type='module' and special handling notes.
    """
    hyperparameters = {}

    type_mapping = {
        "int": "integer",
        "float": "float",
        "bool": "boolean",
        "str": "string",
        "optional": "any",  # No type hint available; accept any type
        "any": "any",
    }

    # Module-type parameters (typically for autoencoder models)
    # These require special handling and will be auto-created by model_factory
    module_params = {"encoder", "decoder", "discriminator"}

    for param_name, param_info in parameters.items():
        # =================================================================
        # FIX 27: Special handling for module-type parameters
        # =================================================================
        # Autoencoder models (GAE, VGAE, ARGA, ARGVA) require torch.nn.Module
        # objects as parameters (encoder, decoder, discriminator).
        # These are marked with type='module' so model_factory can handle
        # them specially (auto-create from configuration).
        # =================================================================
        if param_name in module_params:
            hp_entry = {
                "type": "module",
                "required": param_info.required,
                "description": param_info.description
                or f"{param_name.capitalize()} module (torch.nn.Module)",
                "auto_created": True,  # Flag indicating model_factory will create this
            }
            # decoder typically has default (InnerProductDecoder)
            if param_info.default is not None or not param_info.required:
                hp_entry["required"] = False
                hp_entry["default"] = None  # Will use PyG default (e.g., InnerProductDecoder)
        else:
            hp_entry = {
                "type": type_mapping.get(param_info.param_type, "any"),
                "required": param_info.required,
            }

            # Use explicit default if available
            if param_info.default is not None:
                hp_entry["default"] = param_info.default
            # For required parameters without defaults, infer intelligent defaults
            elif param_info.required:
                inferred_default = _infer_intelligent_default(param_name, param_info.param_type)
                if inferred_default is not None:
                    hp_entry["default"] = inferred_default
                    logger.debug(
                        f"Inferred default for required parameter '{param_name}': {inferred_default}"
                    )

            if param_info.description:
                hp_entry["description"] = param_info.description

            if param_info.min_value is not None:
                hp_entry["min"] = param_info.min_value

            if param_info.max_value is not None:
                hp_entry["max"] = param_info.max_value

            if param_info.choices:
                hp_entry["options"] = param_info.choices

        hyperparameters[param_name] = hp_entry

    return hyperparameters


def _infer_supported_tasks(
    model_name: str, import_path: str, parameters: dict[str, ParameterInfo]
) -> list[str]:
    """
    Infer supported tasks based on model characteristics.

    This replaces hardcoded supported_tasks lists with dynamic inference.

    Task categories:
    - Node-level: node_classification, node_regression
    - Graph-level: graph_classification, graph_regression
    - Edge-level: link_prediction, edge_regression, edge_classification
    - Embedding: node_embedding

    Task support rules:
    1. link_prediction: ANY model with node embeddings (uses dot product decoder)
    2. edge_regression: ANY model with node embeddings (uses MLP decoder via EdgeLevelModelWrapper)
    3. edge_classification: ANY model with node embeddings (uses MLP decoder via EdgeLevelModelWrapper)
    4. Autoencoder models: link_prediction, node_embedding only

    Note on edge-level tasks:
    - edge_regression and edge_classification do NOT require the model to natively process
      edge features. The EdgeLevelModelWrapper in model_factory.py uses an external MLP
      decoder that takes node embeddings from ANY GNN and produces edge-level predictions.
    - This is a standard encoder-decoder pattern in graph ML.
    - The edge-aware detection below is preserved for metadata (requires_edge_features)
      but is NOT used to restrict task support.
    """
    name_lower = model_name.lower()
    path_lower = import_path.lower()

    # Default tasks most GNN models support
    default_tasks = [
        "node_classification",
        "node_regression",
        "graph_classification",
        "graph_regression",
    ]

    # Edge-level tasks work with ANY model that produces node embeddings:
    # - link_prediction: uses dot product decoder on node pairs
    # - edge_regression: uses MLP decoder on concatenated node pair embeddings (via EdgeLevelModelWrapper)
    # - edge_classification: uses MLP decoder on concatenated node pair embeddings (via EdgeLevelModelWrapper)
    edge_level_tasks = ["link_prediction", "edge_regression", "edge_classification"]

    # Autoencoders typically support link prediction only
    if "autoencoder" in path_lower or "gae" in name_lower or "vgae" in name_lower:
        return ["link_prediction", "node_embedding"]

    # Node2Vec, MetaPath2Vec are embedding models
    if "node2vec" in name_lower or "metapath2vec" in name_lower:
        return ["node_embedding", "link_prediction"]

    # Pooling models are typically for graph-level tasks only
    if "pool" in name_lower:
        return ["graph_classification", "graph_regression"]

    # =========================================================================
    # Edge-aware model detection (PRESERVED for metadata/requires_edge_features)
    # =========================================================================
    # This detection is useful for determining if a model can NATIVELY process
    # edge features (sets requires_edge_features metadata). However, it does NOT
    # restrict task support since EdgeLevelModelWrapper handles edge-level tasks
    # for ALL models via external decoders.
    # =========================================================================

    # Check if model supports edge features natively
    # Detection: parameters with 'edge' + ('dim', 'channels', 'features') in name
    requires_edge_features = any(
        "edge" in p.lower() and any(x in p.lower() for x in ["dim", "channels", "features"])
        for p in parameters
    )

    # Known edge-aware model patterns (process edge attributes natively)
    edge_aware_patterns = [
        "nnconv",  # Neural Message Passing for Quantum Chemistry
        "cgconv",  # Crystal Graph CNN
        "ecconv",  # Edge-Conditioned Convolution
        "mpnn",  # Message Passing Neural Network
        "gatconv",  # GAT can use edge features
        "gat",  # GAT variants
        "transformerconv",  # Transformer with edge features
        "pnaconv",  # Principal Neighbourhood Aggregation
        "genconv",  # GENeralized graph convolution
        "gine",  # GIN with edge features
        "edgeconv",  # Edge Convolution
        "edgecnn",  # Edge CNN
    ]
    is_edge_aware_model = any(pattern in name_lower for pattern in edge_aware_patterns)

    # Log edge-awareness for debugging (this info is used by generate_model_metadata
    # to set requires_edge_features field, but does NOT restrict task support)
    if requires_edge_features or is_edge_aware_model:
        logger.debug(f"Model '{model_name}' detected as edge-aware (native edge feature support)")

    # =========================================================================
    # Task support assignment
    # =========================================================================
    # ALL models with out_channels support ALL edge-level tasks via EdgeLevelModelWrapper
    # The wrapper provides external MLP decoders that work with any node embeddings
    # =========================================================================

    # Models with out_channels can produce node embeddings
    if "out_channels" in parameters:
        # ALL models with node embeddings support ALL edge-level tasks
        # EdgeLevelModelWrapper handles edge_regression/edge_classification via MLP decoder
        return default_tasks + edge_level_tasks

    # For other models without out_channels, still support edge tasks
    # (most GNNs produce node embeddings even without explicit out_channels param)
    return default_tasks + edge_level_tasks


def _infer_tags(
    model_name: str, import_path: str, parameters: dict[str, ParameterInfo]
) -> list[str]:
    """Infer tags from model characteristics."""
    tags = []
    name_lower = model_name.lower()

    if "attention" in name_lower or "gat" in name_lower:
        tags.append("attention")
    if "conv" in name_lower:
        tags.append("convolutional")
    if "pool" in name_lower:
        tags.append("pooling")
    if "transformer" in name_lower:
        tags.append("transformer")
    if any(x in name_lower for x in ["schnet", "dimenet", "painn", "cgcnn"]):
        tags.append("molecular")
        tags.append("3d")
    if "heads" in parameters:
        tags.append("multi-head")

    return tags


def _infer_category_enum(import_path: str, model_name: str) -> ModelCategory:
    """
    Infer model category from import path and name.

    Returns ModelCategory enum for backward compatibility.
    """
    path_lower = import_path.lower()
    name_lower = model_name.lower()

    if "autoencoder" in path_lower or "vgae" in name_lower or "gae" in name_lower:
        return ModelCategory.AUTOENCODER
    elif "schnet" in name_lower or "dimenet" in name_lower or "painn" in name_lower:
        return ModelCategory.UTILITY  # Molecular models are in UTILITY in original
    elif "transformer" in name_lower:
        return ModelCategory.TRANSFORMER
    elif "pool" in name_lower:
        return ModelCategory.POOLING
    elif "aggr" in name_lower:
        return ModelCategory.AGGREGATION
    elif any(x in name_lower for x in ["gcn", "gat", "sage", "gin", "pna", "edgecnn"]):
        return ModelCategory.BASIC_GNN
    elif "conv" in name_lower:
        return ModelCategory.CONVOLUTIONAL
    elif "attention" in name_lower:
        return ModelCategory.ATTENTION
    else:
        return _GENERAL_FALLBACK  # Use fallback for unknown models


# =============================================================================
# DYNAMIC SEARCH SPACE GENERATION
# =============================================================================


def generate_search_space(parameters: dict[str, ParameterInfo]) -> dict[str, dict[str, Any]]:
    """
    Dynamically generate HPO search space from introspected parameters.

    Args:
        parameters: Dict of ParameterInfo from introspect_model_signature()

    Returns:
        Search space dict compatible with SearchSpaceBuilder
    """
    search_space = {"hyperparameters": {}}

    for param_name, param_info in parameters.items():
        # Skip required params that must be provided (in_channels, out_channels)
        if param_info.required and param_name in ("in_channels", "out_channels"):
            continue

        space_config = _param_to_search_space(param_name, param_info)
        if space_config:
            search_space["hyperparameters"][param_name] = space_config

    return search_space


def _param_to_search_space(param_name: str, param_info: ParameterInfo) -> dict[str, Any] | None:
    """
    Convert a parameter to a search space configuration.

    DYNAMIC: Uses pattern matching on parameter names for intelligent ranges
    PRODUCTION-READY: Respects model constraints (e.g., num_spherical >= 2 for DimeNet)
    FUTURE-PROOF: Pattern-based approach works for any new model

    Evidence-based ranges from PyG documentation and original papers:
    - DimeNet: num_spherical must be >= 2 (hard PyG validation constraint)
    - DimeNet: num_blocks typically 4-6, num_radial 6-8, num_bilinear 4-8
    - SchNet: num_filters 64-256, num_gaussians 25-100
    """
    param_lower = param_name.lower()

    if param_info.param_type == "int":
        # =================================================================
        # DimeNet-specific parameters with known constraints
        # Evidence: PyG PR #4424 - num_spherical must be >= 2
        # =================================================================
        if param_lower == "num_spherical" or param_lower == "num_sbf":
            # CRITICAL: num_spherical must be >= 2 (PyG hard constraint)
            return {"type": "int", "low": 2, "high": 10}

        if param_lower == "num_radial" or param_lower == "num_rbf":
            # Radial basis functions: typical range 4-12
            return {"type": "int", "low": 4, "high": 12}

        if param_lower == "num_blocks":
            # Building blocks: typical range 3-8
            return {"type": "int", "low": 3, "high": 8}

        if param_lower == "num_bilinear":
            # Bilinear tensor size: typical range 4-12
            return {"type": "int", "low": 4, "high": 12}

        # =================================================================
        # SchNet-specific parameters
        # =================================================================
        if param_lower == "num_gaussians":
            return {"type": "int", "low": 25, "high": 100, "step": 25}

        if param_lower == "num_filters":
            return {"type": "int", "low": 64, "high": 256, "step": 32}

        if param_lower in ("num_interactions", "n_interactions"):
            return {"type": "int", "low": 3, "high": 8}

        # =================================================================
        # DimeNet++ specific embedding parameters
        # Evidence: DimeNet++ paper defaults
        # =================================================================
        if param_lower == "int_emb_size":
            # Interaction embedding size: typical range 32-128
            return {"type": "int", "low": 32, "high": 128, "step": 32}

        if param_lower == "basis_emb_size":
            # Basis embedding size: typical range 4-16
            return {"type": "int", "low": 4, "high": 16, "step": 4}

        if param_lower == "out_emb_channels":
            # Output embedding channels: typical range 128-512
            return {"type": "int", "low": 128, "high": 512, "step": 64}

        # =================================================================
        # General channel/dimension parameters
        # =================================================================
        if "channels" in param_lower or "dim" in param_lower:
            return {"type": "int", "low": 32, "high": 256, "step": 32}

        if "layers" in param_lower:
            return {"type": "int", "low": 2, "high": 6}

        if "heads" in param_lower:
            return {"type": "int", "low": 1, "high": 8}

        # =================================================================
        # Residual/skip connection parameters
        # =================================================================
        if param_lower in ("num_before_skip", "num_after_skip"):
            return {"type": "int", "low": 1, "high": 4}

        if param_lower == "num_output_layers":
            return {"type": "int", "low": 2, "high": 5}

        # =================================================================
        # Neighbor limits
        # =================================================================
        if param_lower in ("max_num_neighbors", "max_neighbors", "k"):
            return {"type": "int", "low": 16, "high": 64, "step": 16}

        if param_lower == "envelope_exponent":
            return {"type": "int", "low": 3, "high": 7}

        # =================================================================
        # Fallback: use default if available, else conservative range
        # =================================================================
        if param_info.default is not None:
            default = param_info.default
            return {"type": "int", "low": max(1, default // 2), "high": default * 2}
        else:
            # Conservative fallback - minimum 2 to avoid edge cases
            return {"type": "int", "low": 2, "high": 10}

    elif param_info.param_type == "float":
        if "dropout" in param_lower:
            return {"type": "float", "low": 0.0, "high": 0.6}
        elif "cutoff" in param_lower:
            # Cutoff distance for molecular models (Angstroms)
            return {"type": "float", "low": 4.0, "high": 8.0}
        elif "ratio" in param_lower or "rate" in param_lower:
            return {"type": "float", "low": 0.0, "high": 1.0}
        elif param_info.default is not None:
            default = param_info.default
            if default > 0:
                return {"type": "float", "low": default / 10, "high": default * 10, "log": True}
            else:
                return {"type": "float", "low": 0.0, "high": 1.0}
        else:
            return {"type": "float", "low": 0.0, "high": 1.0}

    elif param_info.param_type == "bool":
        return {"type": "categorical", "choices": [True, False]}

    elif param_info.param_type == "str":
        # Would need to infer valid choices from docs or defaults
        if param_info.choices:
            return {"type": "categorical", "choices": param_info.choices}
        # Common string parameters
        if "aggr" in param_lower or "aggregation" in param_lower:
            return {"type": "categorical", "choices": ["mean", "max", "add"]}
        if "act" in param_lower or "activation" in param_lower:
            return {
                "type": "categorical",
                "choices": ["relu", "elu", "leaky_relu", "gelu", "swish"],
            }
        if param_lower == "output_initializer":
            return {"type": "categorical", "choices": ["zeros", "glorot_orthogonal"]}

    return None


# =============================================================================
# VALIDATION HELPERS
# =============================================================================


def validate_params_against_signature(model_class: type, params: dict[str, Any]) -> tuple:
    """
    Validate parameters against actual model signature.

    Args:
        model_class: The model class
        params: Parameters to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    signature_params = introspect_model_signature(model_class)
    valid_param_names = set(signature_params.keys())

    # Check for unknown parameters
    for param_name in params:
        if param_name not in valid_param_names:
            errors.append(f"Unknown parameter: '{param_name}'")

    # Check required parameters
    for param_name, param_info in signature_params.items():
        if param_info.required and param_name not in params:
            errors.append(f"Missing required parameter: '{param_name}'")

    return len(errors) == 0, errors


def get_valid_params_for_model(model_class: type) -> set[str]:
    """Get set of valid parameter names for a model."""
    return set(introspect_model_signature(model_class).keys())


# =============================================================================
# MAIN API - REPLACES model_categories.py FUNCTIONS
# =============================================================================


class PyGModelIntrospector:
    """
    Main API for dynamic PyG model introspection.

    Replaces static model_categories.py with runtime introspection.

    Usage:
        >>> introspector = PyGModelIntrospector()
        >>> models = introspector.get_all_model_names()
        >>> metadata = introspector.get_model_metadata("GCN")
        >>> search_space = introspector.get_search_space("GAT")
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._models: dict[str, str] = {}  # name -> import_path
        self._metadata_cache: dict[str, DynamicModelMetadata] = {}
        self._discover_models()
        self._initialized = True

    def _discover_models(self):
        """Discover all available PyG models."""
        self._models = discover_pyg_models()
        logger.info(f"PyGModelIntrospector initialized with {len(self._models)} models")

    def get_all_model_names(self) -> list[str]:
        """Get all available model names - REPLACES model_categories.get_all_model_names()"""
        return list(self._models.keys())

    def get_model_metadata(self, name: str) -> DynamicModelMetadata | None:
        """Get metadata for a model - REPLACES model_categories.get_model_metadata()"""
        if name not in self._models:
            return None

        if name not in self._metadata_cache:
            metadata = generate_model_metadata(name, self._models[name])
            if metadata:
                self._metadata_cache[name] = metadata

        return self._metadata_cache.get(name)

    def get_model_parameters(self, name: str) -> dict[str, ParameterInfo]:
        """Get valid parameters for a model - for validation/filtering."""
        metadata = self.get_model_metadata(name)
        return metadata.parameters if metadata else {}

    def get_search_space(self, name: str) -> dict[str, dict[str, Any]]:
        """Get dynamically generated search space for a model."""
        metadata = self.get_model_metadata(name)
        if not metadata:
            return {}
        return generate_search_space(metadata.parameters)

    def get_import_path(self, name: str) -> str | None:
        """Get import path for a model."""
        return self._models.get(name)

    def has_model(self, name: str) -> bool:
        """Check if model exists."""
        return name in self._models

    def refresh(self):
        """Re-discover models (e.g., after PyG update)."""
        self._metadata_cache.clear()
        self._discover_models()


# Singleton instance
_introspector: PyGModelIntrospector | None = None


def get_introspector() -> PyGModelIntrospector:
    """Get singleton introspector instance."""
    global _introspector
    if _introspector is None:
        _introspector = PyGModelIntrospector()
    return _introspector


# =============================================================================
# BACKWARD COMPATIBLE API - Drop-in replacements for model_categories.py
# =============================================================================


def get_all_model_names() -> list[str]:
    """Drop-in replacement for model_categories.get_all_model_names()"""
    return get_introspector().get_all_model_names()


def get_model_metadata(name: str) -> DynamicModelMetadata | None:
    """Drop-in replacement for model_categories.get_model_metadata()"""
    return get_introspector().get_model_metadata(name)


def get_models_by_category(category: ModelCategory) -> dict[str, DynamicModelMetadata]:
    """
    Drop-in replacement for model_categories.get_models_by_category()

    Args:
        category: ModelCategory enum value

    Returns:
        Dict of model_name -> metadata for models in that category
    """
    introspector = get_introspector()
    result = {}
    for name in introspector.get_all_model_names():
        metadata = introspector.get_model_metadata(name)
        if metadata and metadata.category == category:
            result[name] = metadata
    return result


def get_models_by_task(task_type: str) -> list[str]:
    """
    Drop-in replacement for model_categories.get_models_by_task()

    Args:
        task_type: Task type (e.g., "node_classification", "graph_regression")

    Returns:
        List of model names that support the task
    """
    introspector = get_introspector()
    result = []
    for name in introspector.get_all_model_names():
        metadata = introspector.get_model_metadata(name)
        if metadata and task_type in metadata.supported_tasks:
            result.append(name)
    return result


def get_models_by_tag(tag: str) -> list[str]:
    """
    Drop-in replacement for model_categories.get_models_by_tag()

    Args:
        tag: Tag to filter by (e.g., "attention", "temporal")

    Returns:
        List of model names with the tag
    """
    introspector = get_introspector()
    result = []
    for name in introspector.get_all_model_names():
        metadata = introspector.get_model_metadata(name)
        if metadata and tag in metadata.tags:
            result.append(name)
    return result


def get_category_statistics() -> dict[str, int]:
    """
    Drop-in replacement for model_categories.get_category_statistics()

    Returns:
        Dictionary mapping category name to model count
    """
    introspector = get_introspector()
    stats: dict[str, int] = {}

    for name in introspector.get_all_model_names():
        metadata = introspector.get_model_metadata(name)
        if metadata:
            cat_name = metadata.category.value
            stats[cat_name] = stats.get(cat_name, 0) + 1

    return stats


def search_models(query: str, search_in: list[str] = None) -> list[str]:
    """
    Drop-in replacement for model_categories.search_models()

    Args:
        query: Search query
        search_in: Fields to search in (name, description, tags)

    Returns:
        List of matching model names
    """
    if search_in is None:
        search_in = ["name", "description", "tags"]

    query_lower = query.lower()
    results = []
    introspector = get_introspector()

    for name in introspector.get_all_model_names():
        metadata = introspector.get_model_metadata(name)
        if not metadata:
            continue

        if "name" in search_in and query_lower in name.lower():
            results.append(name)
            continue
        if "description" in search_in and query_lower in metadata.description.lower():
            results.append(name)
            continue
        if "tags" in search_in and any(query_lower in tag.lower() for tag in metadata.tags):
            results.append(name)
            continue

    return results


# =============================================================================
# BACKWARD COMPATIBLE EXPORTS - Match model_categories.py module-level exports
# =============================================================================


# Lazy-loaded ALL_MODELS dict for backward compatibility
# Code that accesses model_categories.ALL_MODELS directly will still work
class _LazyAllModels(dict):
    """Lazy-loading dict that populates on first access."""

    _loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            introspector = get_introspector()
            for name in introspector.get_all_model_names():
                metadata = introspector.get_model_metadata(name)
                if metadata:
                    self[name] = metadata
            self._loaded = True

    def __getitem__(self, key):
        self._ensure_loaded()
        return super().__getitem__(key)

    def __contains__(self, key):
        self._ensure_loaded()
        return super().__contains__(key)

    def keys(self):
        self._ensure_loaded()
        return super().keys()

    def values(self):
        self._ensure_loaded()
        return super().values()

    def items(self):
        self._ensure_loaded()
        return super().items()

    def __iter__(self):
        self._ensure_loaded()
        return super().__iter__()

    def __len__(self):
        self._ensure_loaded()
        return super().__len__()


# Module-level exports for backward compatibility
ALL_MODELS = _LazyAllModels()

# Alias DynamicModelMetadata as ModelMetadata for backward compatibility
# Code that does: from model_categories import ModelMetadata will still work
ModelMetadata = DynamicModelMetadata

# Re-export ModelCategory for imports like: from pyg_introspector import ModelCategory
# (Already imported at top of file)
