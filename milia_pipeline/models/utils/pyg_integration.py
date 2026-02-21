"""
PyTorch Geometric Integration Utilities

Helper functions and utilities for working with PyTorch Geometric Data and Datasets
in the models module. Provides utilities for data validation, batch processing,
and dataset inspection.

Features:
- PyG Data object validation
- Dataset statistics and inspection
- Batch processing utilities
- Data compatibility checking
- Feature dimension inference
- Graph statistics computation

Author: Milia Team
Version: 1.0.0
"""

import logging
from typing import Any

import torch
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.loader import DataLoader

# Import exceptions with fallback
try:
    from milia_pipeline.exceptions import DataCompatibilityError, DataError, ValidationError
except ImportError:

    class DataError(Exception):
        """Exception raised for data-related errors."""

        pass

    class DataCompatibilityError(DataError):
        """Exception raised when data is incompatible."""

        pass

    class ValidationError(Exception):
        """Exception raised for validation failures."""

        pass


logger = logging.getLogger(__name__)


# =============================================================================
# DATA VALIDATION
# =============================================================================


def validate_pyg_data(data: Data, strict: bool = False) -> dict[str, Any]:
    """
    Validate PyTorch Geometric Data object.

    Checks for presence of required attributes and data consistency.

    Args:
        data: PyG Data object to validate
        strict: If True, raise exceptions on validation failures

    Returns:
        Dictionary with validation results

    Raises:
        ValidationError: If strict=True and validation fails

    Example:
        >>> from torch_geometric.data import Data
        >>> data = Data(x=torch.randn(5, 10), edge_index=torch.randint(0, 5, (2, 8)))
        >>> result = validate_pyg_data(data)
        >>> print(result['valid'])
        True
    """
    errors = []
    warnings = []
    info = {}

    # Check if it's a Data object
    if not isinstance(data, Data):
        errors.append(f"Expected Data object, got {type(data)}")
        if strict:
            raise ValidationError(f"Invalid data type: {type(data)}")
        return {"valid": False, "errors": errors, "warnings": warnings, "info": info}

    # Check node features
    if hasattr(data, "x") and data.x is not None:
        info["has_node_features"] = True
        info["num_nodes"] = data.x.size(0)
        info["num_node_features"] = data.x.size(1) if data.x.dim() > 1 else 1

        # Check for NaN or Inf
        if torch.isnan(data.x).any():
            warnings.append("Node features contain NaN values")
        if torch.isinf(data.x).any():
            warnings.append("Node features contain Inf values")
    else:
        info["has_node_features"] = False
        warnings.append("No node features (x) found")

    # Check edge_index
    if hasattr(data, "edge_index") and data.edge_index is not None:
        info["has_edge_index"] = True
        info["num_edges"] = data.edge_index.size(1)

        # Validate edge_index shape
        if data.edge_index.size(0) != 2:
            errors.append(
                f"edge_index should have shape [2, num_edges], got {data.edge_index.shape}"
            )

        # Check for valid node indices
        if hasattr(data, "x") and data.x is not None:
            max_node_idx = data.edge_index.max().item()
            if max_node_idx >= data.x.size(0):
                errors.append(
                    f"edge_index contains invalid node index {max_node_idx} "
                    f"(max valid: {data.x.size(0) - 1})"
                )

        # Check for negative indices
        if (data.edge_index < 0).any():
            errors.append("edge_index contains negative indices")
    else:
        info["has_edge_index"] = False
        errors.append("No edge_index found - required for most GNN models")

    # Check edge features
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        info["has_edge_features"] = True
        info["num_edge_features"] = data.edge_attr.size(1) if data.edge_attr.dim() > 1 else 1

        # Check dimension consistency
        if (
            hasattr(data, "edge_index")
            and data.edge_index is not None
            and data.edge_attr.size(0) != data.edge_index.size(1)
        ):
            errors.append(
                f"edge_attr size {data.edge_attr.size(0)} doesn't match "
                f"number of edges {data.edge_index.size(1)}"
            )
    else:
        info["has_edge_features"] = False

    # Check edge weights
    if hasattr(data, "edge_weight") and data.edge_weight is not None:
        info["has_edge_weights"] = True

        # Check dimension consistency
        if (
            hasattr(data, "edge_index")
            and data.edge_index is not None
            and data.edge_weight.size(0) != data.edge_index.size(1)
        ):
            errors.append(
                f"edge_weight size {data.edge_weight.size(0)} doesn't match "
                f"number of edges {data.edge_index.size(1)}"
            )
    else:
        info["has_edge_weights"] = False

    # Check labels/targets
    if hasattr(data, "y") and data.y is not None:
        info["has_labels"] = True
        info["label_shape"] = tuple(data.y.shape)

        # Determine task type
        if data.y.dim() == 0 or (data.y.dim() == 1 and data.y.size(0) == 1):
            info["task_type"] = "graph_level"
        elif data.y.dim() == 1 and hasattr(data, "x") and data.y.size(0) == data.x.size(0):
            info["task_type"] = "node_level"
        elif data.y.dim() == 2:
            info["task_type"] = "multi_target"
        else:
            info["task_type"] = "unknown"
    else:
        info["has_labels"] = False
        warnings.append("No labels (y) found")

    # Check batch attribute (for batched data)
    if hasattr(data, "batch") and data.batch is not None:
        info["is_batched"] = True
        info["batch_size"] = data.batch.max().item() + 1
    else:
        info["is_batched"] = False

    # Check for pos (3D coordinates)
    if hasattr(data, "pos") and data.pos is not None:
        info["has_3d_coords"] = True
        info["coord_dim"] = data.pos.size(1) if data.pos.dim() > 1 else 1
    else:
        info["has_3d_coords"] = False

    # Overall validation
    valid = len(errors) == 0

    if strict and not valid:
        raise ValidationError("Data validation failed:\n" + "\n".join(errors))

    return {"valid": valid, "errors": errors, "warnings": warnings, "info": info}


def check_data_compatibility(
    data: Data,
    requires_edge_index: bool = True,
    requires_edge_features: bool = False,
    requires_edge_weights: bool = False,
    requires_node_features: bool = True,
) -> tuple[bool, list[str]]:
    """
    Check if data is compatible with model requirements.

    Args:
        data: PyG Data object
        requires_edge_index: Whether model requires edge_index
        requires_edge_features: Whether model requires edge_attr
        requires_edge_weights: Whether model requires edge_weight
        requires_node_features: Whether model requires node features

    Returns:
        Tuple of (compatible, list of missing requirements)

    Example:
        >>> compatible, missing = check_data_compatibility(
        ...     data,
        ...     requires_edge_features=True
        ... )
        >>> if not compatible:
        ...     print(f"Missing: {missing}")
    """
    missing = []

    if requires_node_features and (not hasattr(data, "x") or data.x is None):
        missing.append("node features (x)")

    if requires_edge_index and (not hasattr(data, "edge_index") or data.edge_index is None):
        missing.append("edge_index")

    if requires_edge_features and (not hasattr(data, "edge_attr") or data.edge_attr is None):
        missing.append("edge features (edge_attr)")

    if requires_edge_weights and (not hasattr(data, "edge_weight") or data.edge_weight is None):
        missing.append("edge weights (edge_weight)")

    return len(missing) == 0, missing


# =============================================================================
# FEATURE DIMENSION INFERENCE
# =============================================================================


def infer_num_features(data: Data | Dataset | DataLoader) -> dict[str, int | None]:
    """
    Infer feature dimensions from data.

    Args:
        data: PyG Data, Dataset, or DataLoader

    Returns:
        Dictionary with inferred dimensions:
        - num_node_features: Number of node features
        - num_edge_features: Number of edge features (if present)
        - num_classes: Number of classes (for classification)

    Example:
        >>> dims = infer_num_features(dataset)
        >>> print(f"Node features: {dims['num_node_features']}")
        >>> print(f"Edge features: {dims['num_edge_features']}")
    """
    # Get sample data
    if isinstance(data, Data):
        sample = data
    elif isinstance(data, Dataset):
        if len(data) == 0:
            raise DataError("Cannot infer features from empty dataset")
        sample = data[0]
    elif isinstance(data, DataLoader):
        sample = next(iter(data))
    else:
        raise DataError(f"Cannot infer features from type {type(data)}")

    dims = {
        "num_node_features": None,
        "num_edge_features": None,
        "num_classes": None,
        "output_dim": None,
    }

    # Infer node features
    if hasattr(sample, "x") and sample.x is not None:
        if sample.x.dim() == 1:
            dims["num_node_features"] = 1
        else:
            dims["num_node_features"] = sample.x.size(-1)

    # Infer edge features
    if hasattr(sample, "edge_attr") and sample.edge_attr is not None:
        if sample.edge_attr.dim() == 1:
            dims["num_edge_features"] = 1
        else:
            dims["num_edge_features"] = sample.edge_attr.size(-1)

    # Infer output dimension
    if hasattr(sample, "y") and sample.y is not None:
        if sample.y.dim() == 0:
            # Single scalar output
            dims["output_dim"] = 1
        elif sample.y.dim() == 1:
            if sample.y.size(0) == 1:
                # Single output
                dims["output_dim"] = 1
            else:
                # Could be multi-class or multi-output
                # Try to determine if classification
                if sample.y.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                    # Classification - infer num_classes
                    if isinstance(data, (Dataset, DataLoader)):
                        # Scan dataset for unique labels
                        try:
                            all_labels = []
                            if isinstance(data, Dataset):
                                for i in range(min(len(data), 1000)):  # Sample up to 1000
                                    if hasattr(data[i], "y") and data[i].y is not None:
                                        all_labels.extend(
                                            data[i].y.tolist()
                                            if data[i].y.dim() > 0
                                            else [data[i].y.item()]
                                        )
                            dims["num_classes"] = len(set(all_labels)) if all_labels else None
                        except Exception:
                            dims["num_classes"] = None
                else:
                    # Regression - multi-output
                    dims["output_dim"] = sample.y.size(0)
        elif sample.y.dim() == 2:
            # Multi-target
            dims["output_dim"] = sample.y.size(-1)

    return dims


def infer_out_channels(
    data: Data | Dataset | DataLoader, task_type: str, default: int | None = None
) -> int | None:
    """
    Infer out_channels for model output layer based on task type and data.

    SINGLE SOURCE OF TRUTH for out_channels inference across the entire pipeline.
    Used by: ModelFactory._process_hyperparameters(), ModelFactory._create_custom_model()

    This function wraps infer_num_features() and applies task-specific logic to
    determine the correct output dimension for the model's final layer.

    Args:
        data: PyG Data object, Dataset, or DataLoader
        task_type: Task type string (e.g., 'graph_regression', 'node_classification')
        default: Fallback value if inference fails (None = use task-specific default)

    Returns:
        Inferred out_channels, or default/task-specific fallback if cannot infer

    Inference Rules by Task Type:
        - graph_regression: output_dim from y shape (supports multi-target)
        - graph_classification: num_classes from unique y values
        - node_regression: output_dim from y shape
        - node_classification: num_classes from unique y values
        - link_prediction: always 1 (binary edge prediction)
        - edge_regression: 1 (or from edge_value shape if available)

    Examples:
        >>> # Single-target regression
        >>> data = Data(x=torch.randn(10, 16), y=torch.tensor([1.5]))
        >>> infer_out_channels(data, 'graph_regression')
        1

        >>> # Multi-target regression (8 targets)
        >>> data = Data(x=torch.randn(10, 16), y=torch.randn(8))
        >>> infer_out_channels(data, 'graph_regression')
        8

        >>> # Classification (5 classes)
        >>> data = Data(x=torch.randn(10, 16), y=torch.tensor([0, 1, 2, 3, 4]))
        >>> infer_out_channels(data, 'graph_classification')
        5

        >>> # Link prediction (always binary)
        >>> infer_out_channels(data, 'link_prediction')
        1
    """
    if task_type is None:
        logger.debug(f"task_type is None, returning default={default}")
        return default

    task_lower = task_type.lower()

    # =========================================================================
    # LINK PREDICTION: Always binary (edge exists or not)
    # =========================================================================
    if task_lower == "link_prediction":
        logger.debug("link_prediction: out_channels=1 (binary)")
        return 1

    # =========================================================================
    # EDGE REGRESSION: Check for edge_value attribute
    # =========================================================================
    if task_lower == "edge_regression":
        # Get sample data
        if isinstance(data, Data):
            sample = data
        elif isinstance(data, Dataset):
            if len(data) == 0:
                logger.warning("Empty dataset, cannot infer out_channels for edge_regression")
                return 1 if default is None else default
            sample = data[0]
        elif isinstance(data, DataLoader):
            try:
                sample = next(iter(data))
            except StopIteration:
                logger.warning("Empty DataLoader, cannot infer out_channels for edge_regression")
                return 1 if default is None else default
        else:
            logger.warning(f"Cannot infer out_channels from type {type(data)}")
            return 1 if default is None else default

        # Check for edge_value attribute
        if hasattr(sample, "edge_value") and sample.edge_value is not None:
            ev = sample.edge_value
            if ev.dim() == 0:
                out_channels = 1
            elif ev.dim() == 1:
                out_channels = 1  # Vector of edge values, each is scalar
            else:
                out_channels = ev.size(-1)
            logger.debug(
                f"edge_regression: out_channels={out_channels} from edge_value.shape={list(ev.shape)}"
            )
            return out_channels

        # Default for edge regression
        logger.debug("edge_regression: no edge_value found, defaulting out_channels=1")
        return 1 if default is None else default

    # =========================================================================
    # USE infer_num_features() FOR OTHER TASK TYPES
    # =========================================================================
    try:
        dims = infer_num_features(data)
    except DataError as e:
        logger.warning(f"Cannot infer features from data: {e}")
        if "regression" in task_lower:
            return 1 if default is None else default
        return default

    # =========================================================================
    # CLASSIFICATION TASKS: Use num_classes or output_dim
    # =========================================================================
    if "classification" in task_lower:
        # Prefer num_classes (from scanning unique labels)
        if dims.get("num_classes") is not None:
            out_channels = dims["num_classes"]
            logger.debug(f"{task_type}: out_channels={out_channels} from num_classes")
            return out_channels

        # Fallback to output_dim (for multi-label classification with one-hot encoding)
        if dims.get("output_dim") is not None:
            out_channels = dims["output_dim"]
            logger.debug(f"{task_type}: out_channels={out_channels} from output_dim (multi-label)")
            return out_channels

        # Cannot infer for classification without data
        logger.debug(
            f"{task_type}: Cannot infer num_classes from single Data object, returning default={default}. "
            f"Pass a Dataset for automatic class inference, or specify 'out_channels' explicitly."
        )
        return default

    # =========================================================================
    # REGRESSION TASKS: Use output_dim
    # =========================================================================
    if "regression" in task_lower:
        if dims.get("output_dim") is not None:
            out_channels = dims["output_dim"]
            logger.debug(f"{task_type}: out_channels={out_channels} from output_dim")
            return out_channels

        # Default to 1 for regression if cannot infer (common case: no y attribute)
        fallback = 1 if default is None else default
        logger.debug(
            f"{task_type}: No y attribute found, returning default={fallback}. "
            f"For multi-target regression, ensure sample_data.y has correct shape."
        )
        return fallback

    # =========================================================================
    # UNKNOWN TASK TYPE
    # =========================================================================
    logger.debug(f"Unknown task type '{task_type}', returning default={default}")
    return default


# =============================================================================
# DATASET STATISTICS
# =============================================================================


def compute_dataset_statistics(dataset: Dataset, max_samples: int | None = None) -> dict[str, Any]:
    """
    Compute comprehensive statistics for a PyG dataset.

    Args:
        dataset: PyG Dataset
        max_samples: Maximum number of samples to analyze (None = all)

    Returns:
        Dictionary with dataset statistics

    Example:
        >>> stats = compute_dataset_statistics(dataset)
        >>> print(f"Graphs: {stats['num_graphs']}")
        >>> print(f"Avg nodes: {stats['avg_num_nodes']:.2f}")
    """
    if len(dataset) == 0:
        return {"num_graphs": 0, "error": "Empty dataset"}

    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)

    stats = {
        "num_graphs": len(dataset),
        "num_samples_analyzed": num_samples,
        "node_counts": [],
        "edge_counts": [],
        "has_node_features": False,
        "has_edge_features": False,
        "has_edge_weights": False,
        "has_labels": False,
        "has_pos": False,
    }

    # Collect statistics
    for i in range(num_samples):
        try:
            data = dataset[i]

            # Node count
            if hasattr(data, "x") and data.x is not None:
                stats["node_counts"].append(data.x.size(0))
                stats["has_node_features"] = True
            elif hasattr(data, "num_nodes"):
                stats["node_counts"].append(data.num_nodes)

            # Edge count
            if hasattr(data, "edge_index") and data.edge_index is not None:
                stats["edge_counts"].append(data.edge_index.size(1))

            # Check for features
            if hasattr(data, "edge_attr") and data.edge_attr is not None:
                stats["has_edge_features"] = True

            if hasattr(data, "edge_weight") and data.edge_weight is not None:
                stats["has_edge_weights"] = True

            if hasattr(data, "y") and data.y is not None:
                stats["has_labels"] = True

            if hasattr(data, "pos") and data.pos is not None:
                stats["has_pos"] = True

        except Exception as e:
            logger.warning(f"Error processing graph {i}: {e}")

    # Compute summary statistics
    if stats["node_counts"]:
        stats["avg_num_nodes"] = sum(stats["node_counts"]) / len(stats["node_counts"])
        stats["min_num_nodes"] = min(stats["node_counts"])
        stats["max_num_nodes"] = max(stats["node_counts"])

    if stats["edge_counts"]:
        stats["avg_num_edges"] = sum(stats["edge_counts"]) / len(stats["edge_counts"])
        stats["min_num_edges"] = min(stats["edge_counts"])
        stats["max_num_edges"] = max(stats["edge_counts"])
        stats["avg_degree"] = (
            (2 * stats["avg_num_edges"]) / stats["avg_num_nodes"]
            if stats["avg_num_nodes"] > 0
            else 0
        )

    # Get feature dimensions from first sample
    dims = infer_num_features(dataset)
    stats.update(dims)

    return stats


def print_dataset_summary(dataset: Dataset, name: str = "Dataset"):
    """
    Print a formatted summary of dataset statistics.

    Args:
        dataset: PyG Dataset
        name: Dataset name for display

    Example:
        >>> print_dataset_summary(train_dataset, "Training Set")
    """
    stats = compute_dataset_statistics(dataset)

    print("=" * 70)
    print(f"{name} Summary")
    print("=" * 70)
    print(f"Number of graphs: {stats['num_graphs']}")

    if "avg_num_nodes" in stats:
        print("\nGraph Structure:")
        print(f"  Avg nodes per graph: {stats['avg_num_nodes']:.2f}")
        print(f"  Min/Max nodes: {stats['min_num_nodes']}/{stats['max_num_nodes']}")

    if "avg_num_edges" in stats:
        print(f"  Avg edges per graph: {stats['avg_num_edges']:.2f}")
        print(f"  Min/Max edges: {stats['min_num_edges']}/{stats['max_num_edges']}")
        print(f"  Avg degree: {stats['avg_degree']:.2f}")

    print("\nFeatures:")
    print(f"  Node features: {stats['num_node_features']}")
    print(f"  Edge features: {stats['num_edge_features']}")
    print(f"  Has edge weights: {stats['has_edge_weights']}")
    print(f"  Has 3D coordinates: {stats['has_pos']}")

    if stats["output_dim"]:
        print("\nTarget:")
        print(f"  Output dimension: {stats['output_dim']}")
    if stats["num_classes"]:
        print(f"  Number of classes: {stats['num_classes']}")

    print("=" * 70)


# =============================================================================
# BATCH PROCESSING UTILITIES
# =============================================================================


def create_dataloader(
    dataset: Dataset, batch_size: int = 32, shuffle: bool = False, num_workers: int = 0, **kwargs
) -> DataLoader:
    """
    Create a PyG DataLoader with sensible defaults.

    Args:
        dataset: PyG Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        **kwargs: Additional DataLoader arguments

    Returns:
        PyG DataLoader

    Example:
        >>> loader = create_dataloader(dataset, batch_size=32, shuffle=True)
        >>> for batch in loader:
        ...     # Process batch
        ...     pass
    """
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs
    )


def get_batch_info(batch: Data | Batch) -> dict[str, Any]:
    """
    Get information about a batch.

    Args:
        batch: Batched PyG Data

    Returns:
        Dictionary with batch information

    Example:
        >>> for batch in dataloader:
        ...     info = get_batch_info(batch)
        ...     print(f"Batch size: {info['batch_size']}")
    """
    info = {
        "is_batched": hasattr(batch, "batch") and batch.batch is not None,
    }

    if info["is_batched"]:
        info["batch_size"] = batch.batch.max().item() + 1
        info["total_nodes"] = batch.x.size(0) if hasattr(batch, "x") else batch.num_nodes
        info["total_edges"] = batch.edge_index.size(1) if hasattr(batch, "edge_index") else 0
        info["avg_nodes_per_graph"] = info["total_nodes"] / info["batch_size"]
        info["avg_edges_per_graph"] = info["total_edges"] / info["batch_size"]
    else:
        info["batch_size"] = 1
        info["total_nodes"] = batch.x.size(0) if hasattr(batch, "x") else batch.num_nodes
        info["total_edges"] = batch.edge_index.size(1) if hasattr(batch, "edge_index") else 0

    return info


# =============================================================================
# GRAPH STATISTICS
# =============================================================================


def compute_graph_statistics(data: Data) -> dict[str, Any]:
    """
    Compute statistics for a single graph.

    Args:
        data: PyG Data object

    Returns:
        Dictionary with graph statistics

    Example:
        >>> stats = compute_graph_statistics(data)
        >>> print(f"Density: {stats['density']:.4f}")
    """
    stats = {}

    # Basic counts
    if hasattr(data, "x") and data.x is not None:
        stats["num_nodes"] = data.x.size(0)
    elif hasattr(data, "num_nodes"):
        stats["num_nodes"] = data.num_nodes
    else:
        stats["num_nodes"] = None

    if hasattr(data, "edge_index") and data.edge_index is not None:
        stats["num_edges"] = data.edge_index.size(1)
    else:
        stats["num_edges"] = 0

    # Density
    if stats["num_nodes"] and stats["num_nodes"] > 1:
        max_edges = stats["num_nodes"] * (stats["num_nodes"] - 1)
        stats["density"] = stats["num_edges"] / max_edges if max_edges > 0 else 0
    else:
        stats["density"] = 0

    # Average degree
    if stats["num_nodes"] and stats["num_nodes"] > 0:
        stats["avg_degree"] = (2 * stats["num_edges"]) / stats["num_nodes"]
    else:
        stats["avg_degree"] = 0

    # Degree distribution
    if hasattr(data, "edge_index") and data.edge_index is not None:
        from torch_geometric.utils import degree

        deg = degree(data.edge_index[0], num_nodes=stats["num_nodes"])
        stats["min_degree"] = deg.min().item()
        stats["max_degree"] = deg.max().item()
        stats["std_degree"] = deg.std().item()

    return stats


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def to_device(data: Data | Batch, device: torch.device) -> Data | Batch:
    """
    Move PyG data to device.

    Args:
        data: PyG Data or Batch
        device: Target device

    Returns:
        Data on target device

    Example:
        >>> device = torch.device('cuda')
        >>> data = to_device(data, device)
    """
    return data.to(device)


def detach_data(data: Data | Batch) -> Data | Batch:
    """
    Detach all tensors in PyG data from computation graph.

    Args:
        data: PyG Data or Batch

    Returns:
        Data with detached tensors

    Example:
        >>> data = detach_data(data)
    """
    for key in data.keys:
        if torch.is_tensor(data[key]):
            data[key] = data[key].detach()
    return data


def clone_data(data: Data) -> Data:
    """
    Create a deep copy of PyG Data object.

    Args:
        data: PyG Data object

    Returns:
        Cloned data

    Example:
        >>> data_copy = clone_data(data)
    """
    return data.clone()


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Validation
    "validate_pyg_data",
    "check_data_compatibility",
    # Feature inference
    "infer_num_features",
    "infer_out_channels",
    # Statistics
    "compute_dataset_statistics",
    "print_dataset_summary",
    "compute_graph_statistics",
    # Batch processing
    "create_dataloader",
    "get_batch_info",
    # Utilities
    "to_device",
    "detach_data",
    "clone_data",
]
