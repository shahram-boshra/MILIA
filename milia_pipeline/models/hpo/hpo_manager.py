"""
HPO Manager

Main orchestrator for hyperparameter optimization in the MILIA Pipeline.

This module coordinates hyperparameter optimization by:
1. Setting up the HPO backend (Optuna/Ray Tune)
2. Creating search space from configuration
3. Managing trial execution via objective function
4. Integrating with Trainer for model training
5. Handling cross-validation if configured

Pattern: Follows ModelFactory (model_factory.py:315-953)

**Phase 5 Migration Verification (2025-12-08)**:
- This module does NOT import from model_categories.py directly
- Uses ModelRegistry.get_metadata() which now returns DynamicModelMetadata
- _filter_search_space_for_model() works unchanged with dynamic metadata
- metadata.hyperparameters dict structure is preserved by pyg_introspector
- NO CODE CHANGES REQUIRED - Phase 3 migration handles the data source

Pydantic V2 Migration (Phase 26):
    - Removed unused `from dataclasses import asdict` import
    - All HPO config classes (HPOConfig, PrunerConfig, SamplerConfig, StudyConfig,
      SearchSpaceParamConfig) already migrated to Pydantic BaseModel in Phase 6b
    - HPOManager class is a regular Python class (not a dataclass) - no migration needed
    - NON-BREAKING: No API changes; dead code removal only

Author: Milia Team
Version: 1.2.0
"""

import logging
import time
from collections.abc import Callable
from typing import Any, Optional, Union

# Import existing modules with graceful fallbacks
try:
    from milia_pipeline.models.training.trainer import Trainer
except ImportError:
    Trainer = None

try:
    from milia_pipeline.models.training.data_splitting import DataSplitter
except ImportError:
    DataSplitter = None

try:
    from milia_pipeline.models.factory.model_factory import ModelFactory, get_factory
except ImportError:
    ModelFactory = None
    get_factory = None

# Import TargetSelectionConfig for node/edge level task support
try:
    from milia_pipeline.models.factory.target_selection_config import (
        TargetLevel,
        TargetSelectionConfig,
        TargetSource,
    )

    _TARGET_SELECTION_AVAILABLE = True
except ImportError:
    TargetSelectionConfig = None
    TargetLevel = None
    TargetSource = None
    _TARGET_SELECTION_AVAILABLE = False

try:
    from milia_pipeline.models.training.loss_functions import LossRegistry
except ImportError:
    LossRegistry = None

# Import DiscretizeTargets for classification with float targets
try:
    from milia_pipeline.transformations.custom_transforms import DiscretizeTargets

    _DISCRETIZE_AVAILABLE = True
except ImportError:
    DiscretizeTargets = None
    _DISCRETIZE_AVAILABLE = False

try:
    from milia_pipeline.models.training.optimizers import OptimizerRegistry
except ImportError:
    OptimizerRegistry = None

try:
    from milia_pipeline.models.training.schedulers import SchedulerRegistry
except ImportError:
    SchedulerRegistry = None

try:
    from milia_pipeline.models.acceleration.device_manager import DeviceManager
except ImportError:
    DeviceManager = None

# Import model registry for dynamic search space filtering
try:
    from milia_pipeline.models.registry.model_registry import ModelRegistry

    _REGISTRY_AVAILABLE = True
except ImportError:
    ModelRegistry = None
    _REGISTRY_AVAILABLE = False

# Import HPO-specific modules
from milia_pipeline.exceptions import (
    HPOConfigurationError,
    HPOError,
    StudyNotFoundError,
    TrialFailedError,
)

from .backends import HPOBackendProtocol, get_backend
from .callbacks import create_hpo_callback
from .hpo_config import (
    HPOConfig,
    SearchSpaceParamConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# TASK-SPECIFIC DATA PREPARATION (Mirrors main.py implementation)
# =============================================================================


def _prepare_data_for_task_hpo(
    train_data,
    val_data,
    task_type: str,
    discretize_config: dict[str, Any] | None = None,
    target_selection_config: Optional["TargetSelectionConfig"] = None,
) -> tuple[Any, Any, int | None]:
    """
    Prepare split data for specific task types in HPO context.

    DYNAMIC: Uses target_selection_config for automatic level/source inference
    PRODUCTION-READY: Handles all task types with clear error messages
    FUTURE-PROOF: Extensible for new task types

    This is the HPO-specific version that handles train/val only (no test).

    Args:
        train_data: Training subset from DataSplitter
        val_data: Validation subset from DataSplitter
        task_type: Task type string
        discretize_config: Optional config for DiscretizeTargets transform
            Keys: n_bins, target_column, strategy, attrs
        target_selection_config: Resolved target selection config (NEW)

    Returns:
        Tuple of (train_data, val_data, num_classes)
        - num_classes is None for regression, int for classification with discretization

    Raises:
        HPOError: If data is incompatible with task_type
    """
    task_lower = task_type.lower()
    num_classes = None  # Default: no class count override

    # =========================================================================
    # Resolve target selection config if provided
    # =========================================================================
    if target_selection_config is not None and _TARGET_SELECTION_AVAILABLE:
        # Get sample data for shape validation
        sample = (
            train_data[0] if hasattr(train_data, "__getitem__") and len(train_data) > 0 else None
        )
        target_selection_config.resolve_for_task(task_type, sample)

    # Graph-level tasks
    if task_lower == "graph_regression":
        logger.debug(f"Task '{task_type}' uses standard graph-level data (no transform needed)")
        return train_data, val_data, None

    if task_lower == "graph_classification":
        # Check if discretization is needed (float targets)
        # Pass target_level='graph' to avoid ambiguity when y.shape[0] == num_nodes
        train_data, val_data, num_classes = _prepare_classification_data_hpo(
            train_data, val_data, task_type, discretize_config, target_level="graph"
        )
        return train_data, val_data, num_classes

    # Link prediction: requires edge_label attribute
    if task_lower == "link_prediction":
        train_data, val_data = _prepare_link_prediction_data_hpo(train_data, val_data)
        return train_data, val_data, None

    # Edge regression: may need extraction from edge_attr
    if task_lower == "edge_regression":
        train_data, val_data = _prepare_edge_regression_data_hpo(
            train_data, val_data, target_selection_config
        )
        return train_data, val_data, None

    # Node-level tasks: may need extraction from x
    if task_lower == "node_regression":
        train_data, val_data = _prepare_node_level_data_hpo(
            train_data, val_data, task_type, target_selection_config
        )
        return train_data, val_data, None

    if task_lower == "node_classification":
        train_data, val_data = _prepare_node_level_data_hpo(
            train_data, val_data, task_type, target_selection_config
        )
        # Check if discretization is needed (float targets)
        # Pass target_level='node' to ensure correct handling
        train_data, val_data, num_classes = _prepare_classification_data_hpo(
            train_data, val_data, task_type, discretize_config, target_level="node"
        )
        return train_data, val_data, num_classes

    # Unknown task type: log warning and return unchanged
    logger.warning(f"Unknown task type '{task_type}'. No data preparation applied.")
    return train_data, val_data, None


def _prepare_classification_data_hpo(
    train_data,
    val_data,
    task_type: str,
    discretize_config: dict[str, Any] | None = None,
    target_level: str = "auto",
) -> tuple[Any, Any, int | None]:
    """
    Prepare data for classification task, handling float targets with discretization.

    Args:
        train_data: Training data subset
        val_data: Validation data subset
        task_type: Task type (graph_classification, node_classification, etc.)
        discretize_config: Configuration for DiscretizeTargets
        target_level: Explicit target level for DiscretizeTargets ('auto', 'graph', 'node', 'edge')

    Returns:
        Tuple of (train_data, val_data, num_classes)
        - num_classes is the number of discretization bins if applied, else None
    """
    import torch

    # Check if targets are already integer (no discretization needed)
    sample = train_data[0] if hasattr(train_data, "__getitem__") and len(train_data) > 0 else None

    if sample is None or not hasattr(sample, "y") or sample.y is None:
        logger.warning(f"{task_type}: No target 'y' found in sample data")
        return train_data, val_data, None

    y = sample.y

    # If already integer type, no discretization needed
    if y.dtype in [torch.int, torch.int32, torch.int64, torch.long]:
        # Count unique classes from training data for num_classes
        all_y = []
        for data in train_data:
            if hasattr(data, "y") and data.y is not None:
                all_y.append(data.y)
        if all_y:
            combined_y = torch.cat(
                [yi.flatten() if yi.dim() > 0 else yi.unsqueeze(0) for yi in all_y]
            )
            num_classes = int(combined_y.max().item()) + 1
            logger.info(f"{task_type}: Integer targets detected, {num_classes} classes")
        else:
            num_classes = None
        return train_data, val_data, num_classes

    # Float targets - need discretization
    if y.dtype in [torch.float, torch.float32, torch.float64]:
        logger.info(
            f"{task_type}: Float targets detected (dtype={y.dtype}), discretization required"
        )

        if not _DISCRETIZE_AVAILABLE or DiscretizeTargets is None:
            raise HPOError(
                f"{task_type}: Float targets require DiscretizeTargets transform, but it's not available",
                details="Ensure milia_pipeline.transformations.custom_transforms is importable",
            )

        # Get discretization parameters
        if discretize_config is None:
            discretize_config = {}

        n_bins = discretize_config.get("n_bins", 10)
        target_column = discretize_config.get("target_column", 0)
        strategy = discretize_config.get("strategy", "quantile")
        attrs = discretize_config.get("attrs", ["y"])

        logger.info(
            f"{task_type}: Applying DiscretizeTargets (n_bins={n_bins}, "
            f"target_column={target_column}, strategy={strategy})"
        )

        # Create and fit transform on training data
        discretize_transform = DiscretizeTargets(
            n_bins=n_bins,
            target_column=target_column,
            strategy=strategy,
            attrs=attrs,
            target_level=target_level,
        )

        # Fit on training data
        # Use try/except for iteration - handles both __iter__ and __getitem__ protocols
        # (torch.utils.data.Subset implements __getitem__ but not __iter__)
        try:
            train_list = list(train_data)
        except TypeError:
            train_list = [train_data]
        discretize_transform.fit(train_list)

        # Transform training data
        transformed_train = []
        for data in train_list:
            transformed = discretize_transform(data)
            transformed_train.append(transformed)

        # Transform validation data (using same fitted bin edges)
        try:
            val_list = list(val_data)
        except TypeError:
            val_list = [val_data]
        transformed_val = []
        for data in val_list:
            transformed = discretize_transform(data)
            transformed_val.append(transformed)

        logger.info(
            f"{task_type}: Discretized {len(transformed_train)} train and "
            f"{len(transformed_val)} val samples into {n_bins} classes"
        )

        return transformed_train, transformed_val, n_bins

    # Unknown dtype - return unchanged with warning
    logger.warning(f"{task_type}: Unexpected target dtype {y.dtype}, no transformation applied")
    return train_data, val_data, None


def _prepare_link_prediction_data_hpo(train_data, val_data):
    """Prepare data for link prediction task in HPO context."""
    from torch_geometric.transforms import RandomLinkSplit

    sample = train_data[0] if hasattr(train_data, "__getitem__") and len(train_data) > 0 else None

    if sample is not None and hasattr(sample, "edge_label") and sample.edge_label is not None:
        logger.debug("link_prediction: edge_label already exists in data")
        return train_data, val_data

    logger.info("link_prediction: Applying RandomLinkSplit transform")

    link_split_transform = RandomLinkSplit(
        num_val=0.0,
        num_test=0.0,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
    )

    train_data = _apply_transform_to_subset_hpo(train_data, link_split_transform, "train")
    val_data = _apply_transform_to_subset_hpo(val_data, link_split_transform, "val")

    return train_data, val_data


def _prepare_edge_regression_data_hpo(
    train_data,
    val_data,
    target_selection_config: Optional["TargetSelectionConfig"] = None,
) -> tuple[Any, Any]:
    """
    Prepare data for edge regression task in HPO context.

    DYNAMIC: Auto-extracts targets from configured source attribute
    PRODUCTION-READY: Validates shapes, provides clear errors
    FUTURE-PROOF: Works with any source attribute (edge_attr, edge_y, custom)

    Logic:
    1. Check if edge_y or edge_value already exists -> use as-is
    2. If not, check configured source (edge_attr by default for edge tasks)
    3. Extract targets from source and assign to edge_y

    Args:
        train_data: Training data subset
        val_data: Validation data subset
        target_selection_config: Resolved target selection config (optional)

    Returns:
        Tuple of (train_data, val_data) with edge_y properly set

    Raises:
        HPOError: If no valid target source found
    """
    sample = train_data[0] if hasattr(train_data, "__getitem__") and len(train_data) > 0 else None

    if sample is None:
        raise HPOError(
            "edge_regression: Cannot validate data - empty dataset",
            details="Provide non-empty dataset",
        )

    # =========================================================================
    # CASE 1: edge_y or edge_value already exists
    # =========================================================================
    has_edge_value = hasattr(sample, "edge_value") and sample.edge_value is not None
    has_edge_y = hasattr(sample, "edge_y") and sample.edge_y is not None

    if has_edge_value or has_edge_y:
        attr_name = "edge_value" if has_edge_value else "edge_y"
        logger.info(f"edge_regression: Using existing '{attr_name}' attribute")
        return train_data, val_data

    # =========================================================================
    # CASE 2: Extract from configured source (edge_attr by default)
    # =========================================================================
    # Determine source attribute from target_selection_config
    if (
        target_selection_config is not None
        and _TARGET_SELECTION_AVAILABLE
        and hasattr(target_selection_config, "resolved_source_attr")
        and target_selection_config.resolved_source_attr is not None
    ):
        source_attr = target_selection_config.resolved_source_attr
    else:
        # Default for edge-level: extract from edge_attr
        source_attr = "edge_attr"

    # Determine which indices to extract
    indices = None
    if target_selection_config is not None:
        if (
            hasattr(target_selection_config, "resolved_indices")
            and target_selection_config.resolved_indices is not None
        ):
            indices = target_selection_config.resolved_indices
        elif (
            hasattr(target_selection_config, "indices")
            and target_selection_config.indices is not None
        ):
            indices = target_selection_config.indices

    # Check if source exists
    if not hasattr(sample, source_attr) or getattr(sample, source_attr) is None:
        raise HPOError(
            "edge_regression task requires edge-level targets",
            details=(
                f"Neither 'edge_y' nor 'edge_value' found, and source '{source_attr}' is not available. "
                f"Configure target_selection.target_source to specify which attribute contains targets, "
                f"or add edge_value during data processing."
            ),
        )

    source_tensor = getattr(sample, source_attr)

    # Get number of edges for validation
    num_edges = (
        sample.edge_index.size(1)
        if hasattr(sample, "edge_index") and sample.edge_index is not None
        else None
    )

    if num_edges is not None and source_tensor.size(0) != num_edges:
        raise HPOError(
            f"edge_regression: Source '{source_attr}' has shape {list(source_tensor.shape)}, "
            f"expected first dim = {num_edges} (number of edges)",
            details="Source attribute must have edge-level shape [num_edges, ...]",
        )

    # =========================================================================
    # Extract targets from source and assign to edge_y
    # =========================================================================
    logger.info(
        f"edge_regression: Extracting edge-level targets from '{source_attr}' (indices={indices})"
    )

    train_data = _extract_targets_from_source(train_data, source_attr, indices, "edge_y", "train")
    val_data = _extract_targets_from_source(val_data, source_attr, indices, "edge_y", "val")

    return train_data, val_data


def _prepare_node_level_data_hpo(
    train_data,
    val_data,
    task_type: str,
    target_selection_config: Optional["TargetSelectionConfig"] = None,
) -> tuple[Any, Any]:
    """
    Prepare data for node-level tasks in HPO context.

    DYNAMIC: Auto-extracts targets from inferred source attribute
    PRODUCTION-READY: Validates shapes, provides clear errors
    FUTURE-PROOF: Works with any source attribute

    Logic:
    1. Check if y already has correct shape (num_nodes) -> use as-is
    2. If not, check inferred source (x by default for node tasks)
    3. Extract targets from source and assign to y

    Args:
        train_data: Training data subset
        val_data: Validation data subset
        task_type: Task type string
        target_selection_config: Resolved target selection config (optional)

    Returns:
        Tuple of (train_data, val_data) with y properly set

    Raises:
        HPOError: If no valid target source found
    """
    sample = train_data[0] if hasattr(train_data, "__getitem__") and len(train_data) > 0 else None

    if sample is None:
        raise HPOError(
            f"{task_type}: Cannot validate data - empty dataset",
            details="Provide non-empty dataset",
        )

    num_nodes = sample.num_nodes if hasattr(sample, "num_nodes") else None
    if num_nodes is None and hasattr(sample, "x") and sample.x is not None:
        num_nodes = sample.x.size(0)

    if num_nodes is None:
        raise HPOError(
            f"{task_type}: Cannot determine number of nodes",
            details="Data must have 'num_nodes' attribute or 'x' with shape [num_nodes, ...]",
        )

    # =========================================================================
    # CASE 1: y already has correct node-level shape
    # =========================================================================
    if hasattr(sample, "y") and sample.y is not None:
        y = sample.y
        if y.dim() >= 1 and y.size(0) == num_nodes:
            logger.debug(
                f"{task_type}: y already has node-level shape {list(y.shape)}, "
                f"num_nodes={num_nodes}. Using as-is."
            )
            return train_data, val_data

    # =========================================================================
    # CASE 2: y is graph-level or missing - extract from source
    # =========================================================================
    # Determine source attribute
    if (
        target_selection_config is not None
        and _TARGET_SELECTION_AVAILABLE
        and target_selection_config.resolved_source_attr is not None
    ):
        source_attr = target_selection_config.resolved_source_attr
    else:
        # Default for node-level: extract from x
        source_attr = "x"

    # Determine which indices to extract
    if target_selection_config is not None and target_selection_config.resolved_indices is not None:
        indices = target_selection_config.resolved_indices
    elif target_selection_config is not None and target_selection_config.indices is not None:
        indices = target_selection_config.indices
    else:
        # Default: no specific indices (will use all or last column)
        indices = None

    # Check if source exists
    if not hasattr(sample, source_attr) or getattr(sample, source_attr) is None:
        raise HPOError(
            f"{task_type} task requires node-level targets",
            details=(
                f"y.shape={list(sample.y.shape) if hasattr(sample, 'y') and sample.y is not None else 'None'} "
                f"is not node-level (expected first dim = {num_nodes}). "
                f"Source '{source_attr}' also not available. "
                f"Configure target_selection.indices to specify which columns of x to use as targets, "
                f"or ensure your data has node-level targets in y."
            ),
        )

    source_tensor = getattr(sample, source_attr)

    # Validate source has correct first dimension
    if source_tensor.size(0) != num_nodes:
        raise HPOError(
            f"{task_type}: Source '{source_attr}' has shape {list(source_tensor.shape)}, "
            f"expected first dim = {num_nodes}",
            details="Source attribute must have node-level shape [num_nodes, ...]",
        )

    # =========================================================================
    # Extract targets from source and assign to y
    # =========================================================================
    logger.info(
        f"{task_type}: Extracting node-level targets from '{source_attr}' (indices={indices})"
    )

    train_data = _extract_targets_from_source(train_data, source_attr, indices, "y", "train")
    val_data = _extract_targets_from_source(val_data, source_attr, indices, "y", "val")

    return train_data, val_data


def _extract_targets_from_source(
    data_subset,
    source_attr: str,
    indices: list[int] | None,
    target_attr: str,
    split_name: str,
) -> list:
    """
    Extract targets from source attribute and assign to target attribute.

    DYNAMIC: Works with any source/target attribute combination
    PRODUCTION-READY: Handles all tensor shapes, logs clearly
    FUTURE-PROOF: Generic extraction pattern

    Args:
        data_subset: List of PyG Data objects (or Subset)
        source_attr: Source attribute name (e.g., 'x', 'edge_attr')
        indices: Column indices to extract (None = all)
        target_attr: Target attribute name (e.g., 'y', 'edge_y')
        split_name: Name for logging ('train', 'val', 'test')

    Returns:
        Modified data_subset with targets extracted
    """
    import torch

    modified_data = []

    # Handle Subset objects by iterating
    try:
        data_iter = list(data_subset)
    except TypeError:
        # If not iterable, try indexing
        data_iter = [data_subset[i] for i in range(len(data_subset))]

    for i, data in enumerate(data_iter):
        # Get source tensor
        source = getattr(data, source_attr, None)

        if source is None:
            logger.warning(f"{split_name}[{i}]: Source '{source_attr}' is None, skipping")
            modified_data.append(data)
            continue

        # Extract specified indices
        if indices is not None and len(indices) > 0:
            if source.dim() == 1:
                # 1D tensor - select elements
                target = source[indices]
            else:
                # 2D+ tensor - select columns
                target = source[:, indices]
                # Squeeze if single column
                if target.dim() > 1 and target.size(-1) == 1:
                    target = target.squeeze(-1)
        else:
            # No indices specified - use entire source (clone to avoid modifying original)
            target = source.clone()

        # Ensure target is float for regression tasks
        if target.dtype in [torch.int, torch.int32, torch.int64, torch.long]:
            target = target.float()

        # Assign to target attribute
        setattr(data, target_attr, target)

        if i == 0:
            logger.debug(
                f"{split_name}[0]: Extracted {target_attr} from {source_attr}, "
                f"shape: {list(source.shape)} -> {list(target.shape)}"
            )

        modified_data.append(data)

    return modified_data


def _apply_transform_to_subset_hpo(subset, transform, split_name: str):
    """Apply transform to each Data object in a subset for HPO."""
    transformed = []

    for i, data in enumerate(subset):
        try:
            result = transform(data)
            if isinstance(result, tuple):
                transformed.append(result[0])
            else:
                transformed.append(result)
        except Exception as e:
            logger.warning(f"Transform failed for graph {i} in {split_name}: {e}")
            transformed.append(data)

    return transformed


def infer_task_type(
    dataset,
    metric: str | None = None,
    sample_data: Any | None = None,
) -> str:
    """
    Infer task type from dataset characteristics and metric name.

    Inference priority:
    1. Dataset metadata (if has task_type attribute)
    2. Metric name heuristics (mae, mse, rmse -> regression; accuracy, f1 -> classification)
    3. Target tensor analysis (continuous vs discrete)
    4. Default to 'graph_regression'

    Args:
        dataset: PyG dataset or DataLoader
        metric: Metric name being optimized (optional)
        sample_data: Sample data point for analysis (optional)

    Returns:
        Inferred task type string

    Examples:
        >>> task_type = infer_task_type(dataset, metric='val_mae')
        'graph_regression'

        >>> task_type = infer_task_type(dataset, metric='accuracy')
        'graph_classification'
    """
    # 1. Check dataset metadata
    if hasattr(dataset, "task_type"):
        logger.debug(f"Task type from dataset metadata: {dataset.task_type}")
        return dataset.task_type

    # 2. Check metric name heuristics
    if metric:
        metric_lower = metric.lower()

        # Regression metrics
        regression_indicators = ["mae", "mse", "rmse", "r2", "mape", "loss"]
        if any(ind in metric_lower for ind in regression_indicators):
            logger.debug(f"Inferred 'graph_regression' from metric '{metric}'")
            return "graph_regression"

        # Classification metrics
        classification_indicators = ["accuracy", "acc", "f1", "precision", "recall", "auc", "roc"]
        if any(ind in metric_lower for ind in classification_indicators):
            logger.debug(f"Inferred 'graph_classification' from metric '{metric}'")
            return "graph_classification"

    # 3. Analyze target tensor
    if sample_data is None and hasattr(dataset, "__getitem__"):
        try:
            sample_data = dataset[0]
        except (IndexError, TypeError):
            pass

    if sample_data is not None and hasattr(sample_data, "y"):
        y = sample_data.y
        if y is not None:
            import torch

            # Check if continuous (float) or discrete (int/long)
            if y.dtype in [torch.float, torch.float32, torch.float64]:
                # Check if values are actually continuous
                if y.dim() == 0 or (y.dim() == 1 and y.numel() == 1):
                    # Single scalar target - likely regression
                    logger.debug("Inferred 'graph_regression' from scalar float target")
                    return "graph_regression"
                unique_ratio = len(torch.unique(y)) / max(y.numel(), 1)
                if unique_ratio > 0.5:
                    # Many unique values - likely regression
                    logger.debug("Inferred 'graph_regression' from continuous target distribution")
                    return "graph_regression"
                else:
                    # Few unique values - likely classification
                    logger.debug(
                        "Inferred 'graph_classification' from discrete target distribution"
                    )
                    return "graph_classification"

            elif y.dtype in [torch.int, torch.int32, torch.int64, torch.long]:
                logger.debug("Inferred 'graph_classification' from integer target")
                return "graph_classification"

    # 4. Default fallback
    logger.debug("Using default task type: 'graph_regression'")
    return "graph_regression"


# =============================================================================
# REGISTRY-BASED COMPONENT CREATION HELPERS
# =============================================================================


def _get_loss_name_for_task(task_type: str) -> str:
    """
    Get the appropriate loss function name for a given task type.

    This is DYNAMIC (based on task_type string), PRODUCTION-READY (handles
    all known task types), and FUTURE-PROOF (easy to extend).

    Args:
        task_type: Task type string (e.g., 'graph_regression', 'link_prediction')

    Returns:
        Loss function name for LossRegistry (e.g., 'mse', 'bce_with_logits')
    """
    task_lower = task_type.lower()

    if task_lower == "link_prediction":
        # Link prediction is binary classification (edge exists or not)
        return "bce_with_logits"
    elif "classification" in task_lower:
        # Multi-class classification
        return "cross_entropy"
    else:
        # Default for regression tasks (graph_regression, node_regression, edge_regression)
        return "mse"


def _create_loss_from_registry(task_type: str, loss_params: dict[str, Any] | None = None):
    """
    Create loss function using LossRegistry with automatic task-based selection.

    Uses the refactored LossRegistry which filters invalid parameters automatically.

    Args:
        task_type: Task type to determine loss function
        loss_params: Optional parameters for loss function (invalid ones filtered)

    Returns:
        Instantiated loss function
    """
    import torch.nn as nn

    if LossRegistry is None:
        # Fallback if registry not available
        logger.warning("LossRegistry not available, using inline fallback")
        task_lower = task_type.lower()
        if task_lower == "link_prediction":
            return nn.BCEWithLogitsLoss()
        elif "classification" in task_lower:
            return nn.CrossEntropyLoss()
        else:
            return nn.MSELoss()

    loss_name = _get_loss_name_for_task(task_type)
    loss_fn = LossRegistry.get_loss(loss_name, loss_params)
    logger.debug(f"Created loss '{loss_name}' for task '{task_type}' via LossRegistry")
    return loss_fn


def _create_optimizer_from_registry(
    model_parameters, optimizer_params: dict[str, Any] | None = None, optimizer_name: str = "adam"
):
    """
    Create optimizer using OptimizerRegistry with automatic parameter filtering.

    Uses the refactored OptimizerRegistry which:
    - Merges registry defaults with provided params
    - Filters invalid parameters automatically

    Args:
        model_parameters: Model parameters (from model.parameters())
        optimizer_params: Optional parameters for optimizer (invalid ones filtered)
        optimizer_name: Optimizer name (default: 'adam')

    Returns:
        Instantiated optimizer
    """
    import torch.optim as optim

    if OptimizerRegistry is None:
        # Fallback if registry not available
        logger.warning("OptimizerRegistry not available, using inline fallback")
        params = optimizer_params or {}
        lr = params.get("lr", params.get("learning_rate", 0.001))
        weight_decay = params.get("weight_decay", 0.0001)
        return optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)

    optimizer = OptimizerRegistry.get_optimizer(optimizer_name, model_parameters, optimizer_params)
    logger.debug(f"Created optimizer '{optimizer_name}' via OptimizerRegistry")
    return optimizer


def _create_scheduler_from_registry(
    optimizer,
    scheduler_params: dict[str, Any] | None = None,
    scheduler_name: str = "reduce_on_plateau",
):
    """
    Create scheduler using SchedulerRegistry with automatic parameter filtering.

    Uses the refactored SchedulerRegistry which:
    - Merges registry defaults with provided params
    - Filters invalid parameters automatically

    Args:
        optimizer: PyTorch optimizer instance
        scheduler_params: Optional parameters for scheduler (invalid ones filtered)
        scheduler_name: Scheduler name (default: 'reduce_on_plateau')

    Returns:
        Instantiated scheduler or None if no params provided
    """
    if not scheduler_params:
        return None

    if SchedulerRegistry is None:
        # Fallback if registry not available
        logger.warning("SchedulerRegistry not available, using inline fallback")
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_params.get("factor", 0.5),
            patience=scheduler_params.get("patience", 10),
        )

    scheduler = SchedulerRegistry.get_scheduler(scheduler_name, optimizer, scheduler_params)
    logger.debug(f"Created scheduler '{scheduler_name}' via SchedulerRegistry")
    return scheduler


class HPOManager:
    """
    Main HPO orchestrator class.

    Coordinates hyperparameter optimization by:
    1. Setting up the HPO backend (Optuna/Ray Tune)
    2. Creating search space from configuration
    3. Managing trial execution via objective function
    4. Integrating with Trainer for model training
    5. Handling cross-validation if configured

    Pattern: Follows ModelFactory (model_factory.py:315-953)

    Attributes:
        config: HPOConfig instance
        backend: HPO backend instance
        study: Current study object (set after optimize())
        best_params: Best hyperparameters found (set after optimize())

    Usage:
        >>> # From configuration
        >>> manager = HPOManager.from_config(hpo_config)
        >>>
        >>> # Run optimization
        >>> best_params = manager.optimize(
        ...     model_name="GCN",
        ...     dataset=dataset,
        ...     base_hyperparameters={"num_layers": 3}
        ... )
        >>>
        >>> # Get results
        >>> print(f"Best params: {best_params}")
        >>> print(f"Best value: {manager.get_best_value()}")

    Examples:
        >>> # Simple optimization
        >>> from milia_pipeline.models.hpo import HPOManager, HPOConfig
        >>> config = HPOConfig(enabled=True, n_trials=50)
        >>> manager = HPOManager(config)
        >>> best_params = manager.optimize(model_name="GCN", dataset=dataset)

        >>> # With cross-validation
        >>> config = HPOConfig(enabled=True, n_trials=100, cv_folds=5)
        >>> manager = HPOManager(config)
        >>> best_params = manager.optimize(model_name="GAT", dataset=dataset)
    """

    def __init__(self, config: HPOConfig):
        """
        Initialize HPOManager.

        Args:
            config: HPOConfig instance with all HPO settings

        Raises:
            HPOConfigurationError: If configuration is invalid
        """
        if not config.enabled:
            logger.warning(
                "HPO is disabled in config. Set hpo.enabled=True to enable optimization."
            )

        self.config = config
        self.backend: HPOBackendProtocol | None = None
        self.study: Any | None = None
        self.best_params: dict[str, Any] | None = None
        self._model_factory: ModelFactory | None = None
        self._filtered_search_space: dict[str, dict[str, SearchSpaceParamConfig]] = {}

        # Initialize backend if enabled
        if config.enabled:
            self.backend = get_backend(config.backend)
            logger.info(
                f"HPOManager initialized with {config.backend} backend, n_trials={config.n_trials}"
            )

    @classmethod
    def from_config(cls, config: HPOConfig | dict[str, Any]) -> "HPOManager":
        """
        Create HPOManager from configuration.

        Pattern: Follows ModelConfig.from_yaml() (config_bridge.py:967-993)

        Args:
            config: HPOConfig instance or dict

        Returns:
            HPOManager instance

        Examples:
            >>> # From HPOConfig instance
            >>> manager = HPOManager.from_config(hpo_config)

            >>> # From dictionary
            >>> manager = HPOManager.from_config({
            ...     'enabled': True,
            ...     'n_trials': 100,
            ...     'backend': 'optuna'
            ... })
        """
        if isinstance(config, dict):
            config = HPOConfig.from_dict(config)

        return cls(config)

    @classmethod
    def from_yaml(cls, config_path: str, section: str = "models.hpo") -> "HPOManager":
        """
        Create HPOManager from YAML configuration file.

        Args:
            config_path: Path to config.yaml
            section: Config section path (default: "models.hpo")

        Returns:
            HPOManager instance

        Raises:
            FileNotFoundError: If config file not found
            HPOConfigurationError: If section not found or invalid

        Examples:
            >>> # Standard usage
            >>> manager = HPOManager.from_yaml('config.yaml')

            >>> # Custom section
            >>> manager = HPOManager.from_yaml('config.yaml', section='hpo')
        """
        import yaml

        with open(config_path) as f:
            full_config = yaml.safe_load(f)

        # Navigate to section
        hpo_config = full_config
        for key in section.split("."):
            hpo_config = hpo_config.get(key, {})

        if not hpo_config:
            raise HPOConfigurationError(
                f"HPO configuration section '{section}' not found in {config_path}",
                config_key=section,
            )

        return cls.from_config(hpo_config)

    def optimize(
        self,
        model_name: str,
        dataset,
        base_hyperparameters: dict[str, Any] | None = None,
        trainer_kwargs: dict[str, Any] | None = None,
        callbacks: list | None = None,
        config_dict: dict[str, Any] | None = None,  # For target selection
    ) -> dict[str, Any]:
        """
        Run hyperparameter optimization.

        This is the main entry point for HPO. It creates a study, defines an
        objective function, and runs optimization to find the best hyperparameters.

        Args:
            model_name: Name of model to optimize (from ModelRegistry)
            dataset: PyG dataset or DataLoader for training
            base_hyperparameters: Fixed hyperparameters not being optimized
            trainer_kwargs: Additional kwargs for Trainer initialization
            callbacks: Additional callbacks (HPO callback added automatically)

        Returns:
            Dict of best hyperparameters found

        Raises:
            HPOError: If HPO is disabled or optimization fails
            HPOConfigurationError: If configuration is invalid
            TrialFailedError: If all trials fail

        Examples:
            >>> # Basic optimization
            >>> best_params = manager.optimize(
            ...     model_name="GCN",
            ...     dataset=dataset
            ... )

            >>> # With base hyperparameters and trainer config
            >>> best_params = manager.optimize(
            ...     model_name="GAT",
            ...     dataset=dataset,
            ...     base_hyperparameters={"num_layers": 3},
            ...     trainer_kwargs={"max_epochs": 100}
            ... )
        """
        if not self.config.enabled:
            raise HPOError(
                "HPO is disabled. Set hpo.enabled=True in config.",
                details="Cannot run optimization when disabled",
            )

        if self.backend is None:
            raise HPOError(
                "Backend not initialized", details="Call from_config() or ensure enabled=True"
            )

        base_hyperparameters = base_hyperparameters or {}
        trainer_kwargs = trainer_kwargs or {}
        callbacks = callbacks or []

        logger.info(f"Starting HPO for model '{model_name}' with {self.config.n_trials} trials")

        # Get model factory
        self._model_factory = get_factory() if get_factory else None

        # Determine task_type for loss filtering (M4 fix)
        # If config specifies task_type, use it; otherwise try to infer from dataset
        filter_task_type = self.config.task_type
        if filter_task_type is None:
            # Try to infer task type from dataset for loss param filtering
            try:
                sample_data = dataset[0] if hasattr(dataset, "__getitem__") else None
                filter_task_type = infer_task_type(
                    dataset=dataset,
                    metric=self.config.study.metric,
                    sample_data=sample_data,
                )
                logger.debug(f"Inferred task_type for search space filtering: {filter_task_type}")
            except Exception as e:
                logger.debug(f"Could not infer task_type for filtering: {e}")

        # Filter search space for the target model and loss function
        # This ensures only valid hyperparameters for the model and loss are suggested
        self._filtered_search_space = self._filter_search_space_for_model(
            model_name=model_name,
            search_space=self.config.search_space,
            task_type=filter_task_type,  # M4 fix: pass task_type for loss filtering
        )

        # Create pruner
        pruner = self.backend.create_pruner(
            pruner_type=self.config.pruner.type.value,
            n_startup_trials=self.config.pruner.n_startup_trials,
            n_warmup_steps=self.config.pruner.n_warmup_steps,
            interval_steps=self.config.pruner.interval_steps,
            percentile=self.config.pruner.percentile,
        )

        # Create sampler
        sampler = self.backend.create_sampler(
            sampler_type=self.config.sampler.type.value,
            seed=self.config.sampler.seed,
            n_startup_trials=self.config.sampler.n_startup_trials,
            multivariate=self.config.sampler.multivariate,
            constant_liar=self.config.sampler.constant_liar,
        )

        # Create study
        self.study = self.backend.create_study(
            study_name=self.config.study.study_name,
            direction=self.config.study.direction.value,
            storage=self.config.study.storage,
            load_if_exists=self.config.study.load_if_exists,
            sampler=sampler,
            pruner=pruner,
        )

        # Create objective function
        objective_fn = self._create_objective(
            model_name=model_name,
            dataset=dataset,
            base_hyperparameters=base_hyperparameters,
            trainer_kwargs=trainer_kwargs,
            additional_callbacks=callbacks,
            config_dict=config_dict,  # NEW: For target selection
        )

        # Run optimization
        start_time = time.time()

        self.backend.optimize(
            study=self.study,
            objective_fn=objective_fn,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            catch=(Exception,),
        )

        elapsed = time.time() - start_time

        # Get results
        try:
            self.best_params = self.backend.get_best_params(self.study)
            best_value = self.backend.get_best_value(self.study)

            logger.info(
                f"HPO completed in {elapsed:.1f}s. "
                f"Best {self.config.study.metric}: {best_value:.6f}"
            )
            logger.info(f"Best parameters: {self.best_params}")

            return self.best_params

        except HPOError as e:
            logger.error(f"HPO failed: {e}")
            raise

    def _create_objective(
        self,
        model_name: str,
        dataset,
        base_hyperparameters: dict[str, Any],
        trainer_kwargs: dict[str, Any],
        additional_callbacks: list,
        config_dict: dict[str, Any] | None = None,  # For target selection
    ) -> Callable:
        """
        Create objective function for optimization.

        The objective function:
        1. Suggests hyperparameters from search space
        2. Merges with base hyperparameters
        3. Creates model with suggested params
        4. Trains model with pruning callback
        5. Returns metric value

        Args:
            model_name: Model name for factory
            dataset: Dataset for training
            base_hyperparameters: Fixed hyperparameters
            trainer_kwargs: Trainer configuration
            additional_callbacks: User-provided callbacks

        Returns:
            Objective function for the backend
        """
        config = self.config
        backend = self.backend
        factory = self._model_factory
        models_config = config_dict or {}  # Default to empty dict if None
        filtered_search_space = self._filtered_search_space  # Capture for closure

        def objective(trial) -> float:
            """Objective function for single trial."""
            trial_number = trial.number
            logger.info(f"Starting trial {trial_number}")

            try:
                # 1. Suggest hyperparameters from filtered search space
                # Uses filtered_search_space captured in closure from optimize()
                suggested_params = backend.suggest_params(trial, filtered_search_space)

                # 2. Flatten and merge parameters
                flat_params = _flatten_params(suggested_params)
                hyperparameters = {**base_hyperparameters, **flat_params}

                logger.debug(f"Trial {trial_number} params: {flat_params}")

                # 3. Extract special parameters (optimizer, scheduler, loss, training)
                model_params, optimizer_params, scheduler_params, loss_params, training_params = (
                    _extract_param_categories(hyperparameters)
                )

                # 4. Determine task type with validation
                sample_data = dataset[0] if hasattr(dataset, "__getitem__") else None
                inferred_task_type = infer_task_type(
                    dataset=dataset,
                    metric=config.study.metric,
                    sample_data=sample_data,
                )

                if config.task_type is not None:
                    task_type = config.task_type
                    # Validate CLI/config task type against inferred type (warn only once)
                    if task_type != inferred_task_type and not hasattr(
                        objective, "_task_type_warned"
                    ):
                        logger.warning(
                            f"Specified task_type '{task_type}' differs from data-inferred type "
                            f"'{inferred_task_type}'. Ensure your data is compatible with '{task_type}'. "
                            f"For classification tasks, targets should be integer class indices."
                        )
                        objective._task_type_warned = True
                else:
                    task_type = inferred_task_type

                # 5. Create model with metadata for intelligent forward pass
                # ============================================================
                # TARGET SELECTION: Read from config and pass to factory
                # ============================================================
                # DYNAMIC: Same config path as main.py
                # PRODUCTION-READY: Handles missing config gracefully
                # FUTURE-PROOF: Consistent with main.py implementation
                # ============================================================
                target_selection_raw = models_config.get("selection", {}).get(
                    "target_selection", {}
                )
                if _TARGET_SELECTION_AVAILABLE and TargetSelectionConfig is not None:
                    target_selection_config = TargetSelectionConfig.from_config(
                        target_selection_raw
                    )
                else:
                    target_selection_config = None

                # ============================================================
                # CLASSIFICATION FIX: Get discretize config for float targets
                # ============================================================
                discretize_config = models_config.get("transforms", {}).get(
                    "DiscretizeTargets", None
                )
                if discretize_config is None:
                    # Fallback: check under preprocessing
                    discretize_config = models_config.get("preprocessing", {}).get(
                        "discretize", None
                    )

                # ============================================================
                # CRITICAL: Prepare data BEFORE model creation to get num_classes
                # For classification with float targets, we need to:
                # 1. Split data
                # 2. Apply discretization (which determines num_classes)
                # 3. Create model with correct num_classes
                # ============================================================
                num_classes_override = None  # Will be set if discretization is applied
                train_data_prepared = None
                val_data_prepared = None

                # 5. Create HPO callback for pruning
                hpo_callback = create_hpo_callback(
                    trial=trial,
                    monitor=config.study.metric,
                    report_every=1,
                    backend=config.backend,
                )

                all_callbacks = [hpo_callback] + additional_callbacks

                # 6. Handle cross-validation if configured
                if config.cv_folds > 0:
                    metric_value = _run_cross_validation(
                        model_name=model_name,
                        dataset=dataset,
                        model_params=model_params,
                        optimizer_params=optimizer_params,
                        scheduler_params=scheduler_params,
                        loss_params=loss_params,
                        trainer_kwargs=trainer_kwargs,
                        callbacks=all_callbacks,
                        n_folds=config.cv_folds,
                        metric=config.study.metric,
                        aggregation=config.cv_metric_aggregation,
                        factory=factory,
                        task_type=task_type,
                        discretize_config=discretize_config,
                        target_selection_config=target_selection_config,  # Pass to CV
                    )
                # -------
                else:
                    # 7. Standard training (no cross-validation)
                    if Trainer is None:
                        raise HPOError(
                            "Trainer not available",
                            trial_number=trial_number,
                            details="Cannot train without Trainer class",
                        )

                    if DataSplitter is None:
                        raise HPOError(
                            "DataSplitter not available",
                            trial_number=trial_number,
                            details="Cannot split data without DataSplitter class",
                        )

                    # Import DataLoader
                    from torch_geometric.loader import DataLoader as PyGDataLoader

                    # Get batch_size from training_params or trainer_kwargs or default
                    batch_size = training_params.get(
                        "batch_size", trainer_kwargs.get("batch_size", 32)
                    )

                    # Split dataset
                    train_data, val_data, _ = DataSplitter.random_split(
                        dataset=dataset,
                        train_ratio=0.8,
                        val_ratio=0.2,
                        test_ratio=0.0,
                        random_seed=42,
                    )

                    # Apply task-specific data preparation (including discretization)
                    # This MUST happen BEFORE model creation for classification
                    train_data_prepared, val_data_prepared, num_classes_override = (
                        _prepare_data_for_task_hpo(
                            train_data,
                            val_data,
                            task_type,
                            discretize_config,
                            target_selection_config,
                        )
                    )

                    # ============================================================
                    # NOW create model with correct num_classes for classification
                    # ============================================================
                    model_info = None
                    if factory is not None:
                        model, model_info = factory.create_model_with_info(
                            name=model_name,
                            hyperparameters=model_params,
                            task_type=task_type,
                            sample_data=train_data_prepared[0]
                            if train_data_prepared
                            else dataset[0],
                            target_selection_config=target_selection_config,
                            num_classes_override=num_classes_override,  # CRITICAL: Pass num_classes
                        )

                    # Create DataLoaders with prepared data
                    train_loader = PyGDataLoader(
                        train_data_prepared, batch_size=batch_size, shuffle=True
                    )
                    val_loader = PyGDataLoader(
                        val_data_prepared, batch_size=batch_size, shuffle=False
                    )

                    # Create optimizer using registry (handles param filtering automatically)
                    optimizer = _create_optimizer_from_registry(
                        model.parameters(), optimizer_params
                    )

                    # Create scheduler using registry if params provided
                    scheduler = _create_scheduler_from_registry(optimizer, scheduler_params)

                    # Create loss function using registry (handles param filtering automatically)
                    loss_fn = _create_loss_from_registry(task_type, loss_params)

                    # Get max_epochs from trainer_kwargs or training_params
                    max_epochs = trainer_kwargs.get(
                        "max_epochs",
                        training_params.get("epochs", training_params.get("max_epochs", 100)),
                    )

                    # Create trainer with all required components
                    trainer = Trainer(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss_fn=loss_fn,
                        max_epochs=max_epochs,
                        callbacks=all_callbacks,
                        hpo_callback=hpo_callback,
                        model_info=model_info,
                    )

                    # Train model
                    results = trainer.fit()

                    # Get metric value
                    metric_value = results.get(config.study.metric, results.get("best_val_loss"))
                    # -------
                if metric_value is None:
                    raise TrialFailedError(
                        f"Metric '{config.study.metric}' not found in results",
                        trial_number=trial_number,
                        trial_params=flat_params,
                    )

                logger.info(
                    f"Trial {trial_number} completed: {config.study.metric}={metric_value:.6f}"
                )

                return metric_value

            except Exception as e:
                # Check if it's a pruning exception (should be re-raised)
                try:
                    import optuna

                    if isinstance(e, optuna.TrialPruned):
                        logger.info(f"Trial {trial_number} pruned")
                        raise
                except ImportError:
                    pass

                logger.error(f"Trial {trial_number} failed: {e}")
                raise TrialFailedError(
                    f"Trial failed: {e}",
                    trial_number=trial_number,
                    original_error=str(e),
                )

        return objective

    def _filter_search_space_for_model(
        self,
        model_name: str,
        search_space: dict[str, dict[str, SearchSpaceParamConfig]],
        task_type: str | None = None,
    ) -> dict[str, dict[str, SearchSpaceParamConfig]]:
        """
        Filter search space to only include parameters valid for the specified model
        and loss function.

        This method ensures that only hyperparameters supported by the target model
        and loss function are included in the search space, preventing invalid
        parameters from being suggested during HPO trials.

        The filtering uses:
        - Model metadata from the registry for hyperparameters
        - LossRegistry introspection for loss parameters

        For models/losses not in the registry, returns the original search space
        with a warning.

        **Phase 5 Migration Note**:
        This method uses registry.get_metadata() which now returns DynamicModelMetadata
        from pyg_introspector (via Phase 3 migration). The metadata.hyperparameters dict
        structure is identical to the original ModelMetadata, so this code works unchanged.
        The data SOURCE is now dynamic introspection, but the data FORMAT is preserved.

        **M4 Fix (2025-12-10)**:
        Now also filters 'loss' category parameters using LossRegistry.get_valid_params()
        to prevent unused parameters (e.g., 'alpha' for MSELoss) from being optimized.

        Args:
            model_name: Name of the target model (e.g., "GCN", "GAT")
            search_space: Original search space from configuration
            task_type: Optional task type for loss function filtering

        Returns:
            Filtered search space containing only valid parameters for the model
            and loss function

        Design:
            - DYNAMIC: Uses registry metadata and introspection, not hardcoded names
            - PRODUCTION-READY: Handles missing models/losses gracefully
            - FUTURE-PROOF: Works with any model/loss registered in the system
        """
        from copy import deepcopy

        # If registry not available, return original with warning
        if not _REGISTRY_AVAILABLE or ModelRegistry is None:
            logger.warning(
                f"ModelRegistry not available. Cannot filter search space for '{model_name}'. "
                f"Using unfiltered search space."
            )
            return deepcopy(search_space)

        # Get registry instance
        registry = ModelRegistry.get_instance()

        # Check if model exists in registry
        if not registry.has_model(model_name):
            logger.warning(
                f"Model '{model_name}' not found in registry. "
                f"Cannot filter search space. Using unfiltered search space."
            )
            return deepcopy(search_space)

        # Get model metadata
        metadata = registry.get_metadata(model_name)
        if metadata is None:
            logger.warning(
                f"No metadata found for model '{model_name}'. Using unfiltered search space."
            )
            return deepcopy(search_space)

        # Get valid hyperparameter names from metadata
        valid_hyperparams = set(metadata.hyperparameters.keys())

        # Get valid loss parameters (M4 fix: filter loss params too)
        valid_loss_params = set()
        loss_name = None
        if task_type is not None and LossRegistry is not None:
            try:
                loss_name = _get_loss_name_for_task(task_type)
                loss_params_dict = LossRegistry.get_valid_params(loss_name)
                valid_loss_params = set(loss_params_dict.keys())
                logger.debug(f"Loss '{loss_name}' accepts parameters: {sorted(valid_loss_params)}")
            except Exception as e:
                logger.debug(
                    f"Could not get valid params for loss (task_type={task_type}): {e}. "
                    f"Loss params will not be filtered."
                )

        # Create filtered search space
        filtered_space = {}
        removed_hyperparams = []
        removed_loss_params = []

        for category, params in search_space.items():
            filtered_space[category] = {}

            for param_name, param_config in params.items():
                # Filter 'hyperparameters' category based on model metadata
                if category == "hyperparameters":
                    if param_name in valid_hyperparams:
                        filtered_space[category][param_name] = deepcopy(param_config)
                    else:
                        removed_hyperparams.append(param_name)
                # Filter 'loss' category based on loss function introspection (M4 fix)
                elif category == "loss" and valid_loss_params:
                    if param_name in valid_loss_params:
                        filtered_space[category][param_name] = deepcopy(param_config)
                    else:
                        removed_loss_params.append(param_name)
                else:
                    # Other categories (optimizer, scheduler, training) pass through unchanged
                    filtered_space[category][param_name] = deepcopy(param_config)

        # Log filtering results for hyperparameters
        if removed_hyperparams:
            logger.info(
                f"Search space adapted for '{model_name}': "
                f"{len(removed_hyperparams)} config-suggested parameter(s) not applicable to this model architecture "
                f"(skipped: {removed_hyperparams})"
            )
            logger.debug(
                f"Removing config-suggested parameters {removed_hyperparams} that are not in "
                f"'{model_name}' model parameters: {sorted(valid_hyperparams)}"
            )

        # Log filtering results for loss parameters (M4 fix)
        if removed_loss_params:
            logger.info(
                f"Search space adapted for loss '{loss_name}': "
                f"{len(removed_loss_params)} config-suggested loss parameter(s) not applicable "
                f"(skipped: {removed_loss_params})"
            )
            logger.debug(
                f"Removing config-suggested loss parameters {removed_loss_params} that are not in "
                f"'{loss_name}' loss parameters: {sorted(valid_loss_params)}"
            )

        if not removed_hyperparams and not removed_loss_params:
            logger.debug(
                f"Search space for '{model_name}': "
                f"all config-suggested parameters are valid, no filtering needed"
            )

        return filtered_space

    def get_best_value(self) -> float:
        """
        Get best objective value from completed study.

        Returns:
            Best objective value (float)

        Raises:
            HPOError: If no study available
        """
        if self.study is None:
            raise HPOError("No study available. Run optimize() first.")
        return self.backend.get_best_value(self.study)

    def get_best_trial(self) -> dict[str, Any]:
        """
        Get information about the best trial.

        Returns:
            Dict containing trial information:
            - number: Trial number
            - params: Hyperparameters
            - value: Objective value
            - state: Trial state
            - duration: Duration in seconds

        Raises:
            HPOError: If no study available or best trial not found
        """
        if self.study is None:
            raise HPOError("No study available. Run optimize() first.")

        all_trials = self.backend.get_all_trials(self.study)
        best_value = self.get_best_value()

        for trial in all_trials:
            if trial["value"] == best_value:
                return trial

        raise HPOError("Could not find best trial in study")

    def get_all_trials(self) -> list[dict[str, Any]]:
        """
        Get information about all trials.

        Returns:
            List of trial info dicts with keys:
            - number: Trial number
            - params: Hyperparameters
            - value: Objective value (None if not completed)
            - state: Trial state (COMPLETE, PRUNED, FAIL)
            - duration: Duration in seconds

        Raises:
            HPOError: If no study available
        """
        if self.study is None:
            raise HPOError("No study available. Run optimize() first.")
        return self.backend.get_all_trials(self.study)

    def get_study_statistics(self) -> dict[str, Any]:
        """
        Get study statistics summary.

        Returns comprehensive statistics about the optimization study
        including trial counts, success rates, and timing information.

        Returns:
            Dict with statistics:
            - n_trials: Total number of trials
            - n_completed: Successfully completed trials
            - n_pruned: Pruned trials
            - n_failed: Failed trials
            - best_value: Best objective value
            - worst_value: Worst objective value
            - mean_value: Mean objective value
            - mean_duration: Mean trial duration (seconds)
            - total_duration: Total duration (seconds)
            - pruning_rate: Fraction of trials pruned

        Raises:
            HPOError: If no study available
        """
        if self.study is None:
            raise HPOError("No study available. Run optimize() first.")

        trials = self.get_all_trials()

        completed = [t for t in trials if t["state"] == "COMPLETE"]
        pruned = [t for t in trials if t["state"] == "PRUNED"]
        failed = [t for t in trials if t["state"] == "FAIL"]

        values = [t["value"] for t in completed if t["value"] is not None]
        durations = [t["duration"] for t in completed if t["duration"] is not None]

        return {
            "n_trials": len(trials),
            "n_completed": len(completed),
            "n_pruned": len(pruned),
            "n_failed": len(failed),
            "best_value": min(values) if values else None,
            "worst_value": max(values) if values else None,
            "mean_value": sum(values) / len(values) if values else None,
            "mean_duration": sum(durations) / len(durations) if durations else None,
            "total_duration": sum(durations) if durations else None,
            "pruning_rate": len(pruned) / len(trials) if trials else 0.0,
        }

    def train_final_model(
        self,
        dataset,
        model_name: str,
        base_hyperparameters: dict[str, Any] | None = None,
        training_config: dict[str, Any] | None = None,
        callbacks: list | None = None,
        config_dict: dict[str, Any] | None = None,
        final_epochs: int | None = None,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """
        Train final model with best hyperparameters found during HPO.

        This method implements the Optuna best practice of retraining with
        best parameters after hyperparameter optimization completes.

        Reference: https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/010_reuse_best_trial.html

        Args:
            dataset: PyG dataset for training
            model_name: Name of model (from ModelRegistry)
            base_hyperparameters: Fixed hyperparameters not optimized by HPO
            training_config: Training configuration dict with keys:
                - data_split: {train_ratio, val_ratio, test_ratio, random_seed}
                - batch_size: Batch size for DataLoaders
                - epochs: Default epochs if final_epochs not specified
                - final_training_epochs: Epochs for final training
                - optimizer: {name, params}
                - scheduler: {enabled, name, params}
                - loss: {name, params}
            callbacks: List of callbacks for training
            config_dict: Full models config dict (for target_selection)
            final_epochs: Override epochs for final training

        Returns:
            Tuple of (trained_model, trainer, results_dict)

        Raises:
            HPOError: If best_params not available (optimize() not called)
            HPOError: If required dependencies not available

        Examples:
            >>> # After running optimize()
            >>> best_params = manager.optimize(model_name="GCN", dataset=dataset)
            >>>
            >>> # Train final model with best params
            >>> model, trainer, results = manager.train_final_model(
            ...     dataset=dataset,
            ...     model_name="GCN",
            ...     training_config=training_config,
            ... )
            >>> print(f"Final val_loss: {results['val_loss']}")
        """
        # Validate state
        if self.best_params is None:
            raise HPOError(
                "No best parameters available. Call optimize() first.",
                details="train_final_model() requires best_params from a completed HPO study",
            )

        # Validate dependencies
        if Trainer is None:
            raise HPOError(
                "Trainer not available", details="Cannot train final model without Trainer class"
            )

        if DataSplitter is None:
            raise HPOError(
                "DataSplitter not available", details="Cannot split data without DataSplitter class"
            )

        if get_factory is None:
            raise HPOError(
                "ModelFactory not available", details="Cannot create model without ModelFactory"
            )

        # Initialize defaults
        base_hyperparameters = base_hyperparameters or {}
        training_config = training_config or {}
        callbacks = callbacks or []
        config_dict = config_dict or {}

        logger.info("=" * 60)
        logger.info("FINAL MODEL TRAINING WITH BEST HYPERPARAMETERS")
        logger.info("=" * 60)

        # Import required modules
        from torch_geometric.loader import DataLoader as PyGDataLoader

        # =================================================================
        # 1. SPLIT DATA
        # =================================================================
        data_split_config = training_config.get("data_split", {})
        train_data, val_data, test_data = DataSplitter.random_split(
            dataset=dataset,
            train_ratio=data_split_config.get("train_ratio", 0.8),
            val_ratio=data_split_config.get("val_ratio", 0.1),
            test_ratio=data_split_config.get("test_ratio", 0.1),
            random_seed=data_split_config.get("random_seed", 42),
        )
        logger.info(
            f"Data split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}"
        )

        # =================================================================
        # 2. DETERMINE TASK TYPE
        # =================================================================
        sample_data = dataset[0] if hasattr(dataset, "__getitem__") else None

        if self.config.task_type is not None:
            task_type = self.config.task_type
        else:
            task_type = infer_task_type(
                dataset=dataset,
                metric=self.config.study.metric,
                sample_data=sample_data,
            )
        logger.info(f"Task type: {task_type}")

        # =================================================================
        # 3. APPLY TASK-SPECIFIC DATA PREPARATION
        # =================================================================
        # Get discretize config from training_config or use defaults
        discretize_config = training_config.get("discretize", None)
        if discretize_config is None:
            discretize_config = config_dict.get("transforms", {}).get("DiscretizeTargets", None)
        if discretize_config is None:
            discretize_config = config_dict.get("preprocessing", {}).get("discretize", None)

        # Parse target selection config
        target_selection_raw = config_dict.get("selection", {}).get("target_selection", {})
        if _TARGET_SELECTION_AVAILABLE and TargetSelectionConfig is not None:
            target_selection_config = TargetSelectionConfig.from_config(target_selection_raw)
        else:
            target_selection_config = None

        # Use HPO version for train/val, but we also need test data
        train_data, val_data, num_classes_override = _prepare_data_for_task_hpo(
            train_data, val_data, task_type, discretize_config, target_selection_config
        )

        # Apply same preparation to test data if it exists
        if len(test_data) > 0:
            # For test data, we use the same discretize_config for consistency
            _, test_data, _ = _prepare_data_for_task_hpo(
                test_data, test_data, task_type, discretize_config, target_selection_config
            )

        # =================================================================
        # 4. CREATE DATALOADERS
        # =================================================================
        batch_size = training_config.get("batch_size", 32)
        train_loader = PyGDataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = PyGDataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = (
            PyGDataLoader(test_data, batch_size=batch_size, shuffle=False)
            if len(test_data) > 0
            else None
        )

        # =================================================================
        # 5. MERGE BEST_PARAMS WITH BASE HYPERPARAMETERS
        # =================================================================
        final_hyperparameters = base_hyperparameters.copy()

        # Extract hyperparameters.* keys for model creation
        for key, value in self.best_params.items():
            if key.startswith("hyperparameters."):
                param_name = key.replace("hyperparameters.", "")
                final_hyperparameters[param_name] = value

        logger.debug(f"Final hyperparameters: {final_hyperparameters}")

        # =================================================================
        # 6. CREATE MODEL WITH BEST HYPERPARAMETERS
        # =================================================================
        factory = get_factory()

        # Use transformed data sample for correct feature dimensions after discretization
        prepared_sample = train_data[0] if train_data else sample_data

        model, model_info = factory.create_model_with_info(
            name=model_name,
            hyperparameters=final_hyperparameters,
            task_type=task_type,
            sample_data=prepared_sample,
            target_selection_config=target_selection_config,
            num_classes_override=num_classes_override,  # CRITICAL: Pass num_classes for classification
        )
        logger.info(f"Created final model: {model.__class__.__name__} with best hyperparameters")

        # =================================================================
        # 7. EXTRACT OPTIMIZER/SCHEDULER/LOSS PARAMS FROM BEST_PARAMS
        # =================================================================
        optimizer_params = {}
        scheduler_params = {}
        loss_params = {}

        for key, value in self.best_params.items():
            if key.startswith("optimizer."):
                param_name = key.replace("optimizer.", "")
                optimizer_params[param_name] = value
            elif key.startswith("scheduler."):
                param_name = key.replace("scheduler.", "")
                scheduler_params[param_name] = value
            elif key.startswith("loss."):
                param_name = key.replace("loss.", "")
                loss_params[param_name] = value

        # Merge with training_config params (best_params take precedence)
        opt_config = training_config.get("optimizer", {}).get("params", {})
        sched_config = training_config.get("scheduler", {}).get("params", {})
        loss_config = training_config.get("loss", {}).get("params", {})

        optimizer_params = {**opt_config, **optimizer_params}
        scheduler_params = {**sched_config, **scheduler_params}
        loss_params = {**loss_config, **loss_params}

        logger.info(
            f"Applied HPO-optimized params: lr={optimizer_params.get('lr')}, "
            f"weight_decay={optimizer_params.get('weight_decay')}"
        )

        # =================================================================
        # 8. CREATE LOSS, OPTIMIZER, SCHEDULER USING REGISTRIES
        # =================================================================
        # Uses the refactored registries with automatic parameter filtering
        loss_fn = _create_loss_from_registry(task_type, loss_params)

        optimizer = _create_optimizer_from_registry(
            model.parameters(),
            optimizer_params,
            optimizer_name=training_config.get("optimizer", {}).get("name", "adam"),
        )

        # Only create scheduler if enabled in config
        scheduler = None
        if training_config.get("scheduler", {}).get("enabled", True):
            scheduler = _create_scheduler_from_registry(
                optimizer,
                scheduler_params,
                scheduler_name=training_config.get("scheduler", {}).get(
                    "name", "reduce_on_plateau"
                ),
            )

        logger.info(f"Loss: {loss_fn.__class__.__name__}")
        logger.info(f"Optimizer: {optimizer.__class__.__name__}")
        logger.info(f"Scheduler: {scheduler.__class__.__name__ if scheduler else 'None'}")

        # =================================================================
        # 9. DETERMINE FINAL EPOCHS
        # =================================================================
        if final_epochs is None:
            final_epochs = training_config.get(
                "final_training_epochs", training_config.get("epochs", 100)
            )
        logger.info(f"Final training epochs: {final_epochs}")

        # =================================================================
        # 10. CREATE TRAINER AND TRAIN
        # =================================================================
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            max_epochs=final_epochs,
            callbacks=callbacks,
            model_info=model_info,
        )

        logger.info(f"Starting final model training for {final_epochs} epochs...")
        results = trainer.fit()

        # C1 fix: trainer.fit() returns 'best_val_loss', not 'val_loss'
        logger.info(f"Final model val_loss: {results.get('best_val_loss', 'N/A')}")
        logger.info("✅ Final model training completed!")

        return model, trainer, results

    def resume_study(
        self,
        study_name: str,
        storage: str,
        additional_trials: int = 0,
    ) -> dict[str, Any]:
        """
        Resume an existing study from storage.

        Loads an existing study and optionally continues optimization
        with additional trials.

        Args:
            study_name: Name of existing study
            storage: Storage URL (e.g., "sqlite:///optuna.db")
            additional_trials: Number of additional trials to run (0 = just load)

        Returns:
            Best parameters from the study

        Raises:
            StudyNotFoundError: If study not found in storage
            HPOError: If loading or optimization fails
        """
        if self.backend is None:
            raise HPOError(
                "Backend not initialized", details="Initialize HPOManager with enabled=True"
            )

        logger.info(f"Resuming study '{study_name}' from {storage}")

        try:
            # Load existing study
            self.study = self.backend.create_study(
                study_name=study_name,
                direction=self.config.study.direction.value,
                storage=storage,
                load_if_exists=True,
            )

            existing_trials = len(self.backend.get_all_trials(self.study))
            logger.info(f"Loaded study with {existing_trials} existing trials")

            # Get current best
            self.best_params = self.backend.get_best_params(self.study)

            return self.best_params

        except Exception as e:
            raise StudyNotFoundError(
                f"Failed to resume study: {e}",
                study_name=study_name,
                storage_url=storage,
            )

    # =========================================================================
    # RESULTS SAVING (Phase 5 Refactor)
    # =========================================================================

    def save_results(
        self,
        output_dir: Union[str, "Path"],
        best_params_filename: str = "best_params.json",
        statistics_filename: str = "study_statistics.json",
        trials_filename: str = "all_trials.json",
    ) -> dict[str, "Path"]:
        """
        Save HPO results to output directory.

        DYNAMIC: Saves all available study information
        PRODUCTION-READY: Comprehensive error handling, JSON serialization
        FUTURE-PROOF: Extensible via filenames, returns saved paths

        Args:
            output_dir: Directory to save results (created if doesn't exist)
            best_params_filename: Filename for best parameters (default: 'best_params.json')
            statistics_filename: Filename for study statistics (default: 'study_statistics.json')
            trials_filename: Filename for all trials (default: 'all_trials.json')

        Returns:
            Dict with paths to saved files:
            - 'best_params_path': Path to best_params.json
            - 'statistics_path': Path to study_statistics.json
            - 'trials_path': Path to all_trials.json

        Raises:
            HPOError: If no study available or saving fails

        Example:
            >>> best_params = manager.optimize(model_name="GCN", dataset=dataset)
            >>> saved_paths = manager.save_results(output_dir='./hpo_output')
            >>> print(f"Best params saved to: {saved_paths['best_params_path']}")
        """
        import json
        from pathlib import Path

        if self.study is None:
            raise HPOError(
                "No study available to save. Run optimize() first.",
                details="save_results() requires a completed optimization study",
            )

        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        # Save best parameters
        best_params_path = output_path / best_params_filename
        try:
            with open(best_params_path, "w") as f:
                json.dump(self.best_params or {}, f, indent=2)
            saved_paths["best_params_path"] = best_params_path
            logger.info(f"Best parameters saved: {best_params_path}")
        except Exception as e:
            raise HPOError(
                f"Failed to save best parameters: {e}", details=f"Target path: {best_params_path}"
            )

        # Save study statistics
        statistics_path = output_path / statistics_filename
        try:
            stats = self.get_study_statistics()
            with open(statistics_path, "w") as f:
                json.dump(stats, f, indent=2)
            saved_paths["statistics_path"] = statistics_path
            logger.info(f"Study statistics saved: {statistics_path}")
        except Exception as e:
            raise HPOError(
                f"Failed to save study statistics: {e}", details=f"Target path: {statistics_path}"
            )

        # Save all trials
        trials_path = output_path / trials_filename
        try:
            all_trials = self.get_all_trials()
            with open(trials_path, "w") as f:
                json.dump(all_trials, f, indent=2)
            saved_paths["trials_path"] = trials_path
            logger.info(f"All trials saved: {trials_path}")
        except Exception as e:
            raise HPOError(f"Failed to save trials: {e}", details=f"Target path: {trials_path}")

        return saved_paths


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _flatten_params(params: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten nested parameter dict.

    Converts category-prefixed parameters (e.g., "hyperparameters.hidden_channels")
    to flat keys (e.g., "hidden_channels").

    Args:
        params: Dict with potentially prefixed parameter names

    Returns:
        Flattened dict with simple parameter names

    Examples:
        >>> params = {"model.hidden_channels": 128, "optimizer.lr": 0.001}
        >>> _flatten_params(params)
        {'hidden_channels': 128, 'lr': 0.001}
    """
    flat = {}
    for key, value in params.items():
        # Remove category prefix if present
        if "." in key:
            flat_key = key.split(".")[-1]
        else:
            flat_key = key
        flat[flat_key] = value
    return flat


def _extract_param_categories(
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Separate parameters into categories for different components.

    Identifies parameters that belong to optimizer, scheduler, loss function,
    or training configuration based on known parameter names.

    Args:
        params: Flat dict of all hyperparameters

    Returns:
        Tuple of (model_params, optimizer_params, scheduler_params, loss_params, training_params)

    Examples:
        >>> params = {'hidden_channels': 128, 'lr': 0.001, 'factor': 0.5, 'batch_size': 32}
        >>> model, opt, sched, loss, train = _extract_param_categories(params)
        >>> model
        {'hidden_channels': 128}
        >>> opt
        {'lr': 0.001}
        >>> sched
        {'factor': 0.5}
        >>> train
        {'batch_size': 32}
    """
    # Known parameter names for each category
    optimizer_keys = {
        "lr",
        "learning_rate",
        "weight_decay",
        "momentum",
        "betas",
        "eps",
        "amsgrad",
        "nesterov",
    }
    scheduler_keys = {
        "factor",
        "patience",
        "step_size",
        "gamma",
        "T_max",
        "eta_min",
        "cooldown",
        "min_lr",
        "T_0",
        "T_mult",
    }
    loss_keys = {
        "alpha",
        "gamma",
        "reduction",
        "weight",
        "label_smoothing",
        "pos_weight",
        "ignore_index",
    }
    training_keys = {
        "batch_size",
        "epochs",
        "max_epochs",
        "num_epochs",
        "gradient_clip_val",
        "accumulate_grad_batches",
        "log_every_n_steps",
        "shuffle",
    }

    model_params = {}
    optimizer_params = {}
    scheduler_params = {}
    loss_params = {}
    training_params = {}

    for key, value in params.items():
        if key in optimizer_keys:
            optimizer_params[key] = value
        elif key in scheduler_keys:
            scheduler_params[key] = value
        elif key in loss_keys:
            loss_params[key] = value
        elif key in training_keys:
            training_params[key] = value
        else:
            model_params[key] = value

    return model_params, optimizer_params, scheduler_params, loss_params, training_params


def _run_cross_validation(
    model_name: str,
    dataset,
    model_params: dict[str, Any],
    optimizer_params: dict[str, Any],
    scheduler_params: dict[str, Any],
    loss_params: dict[str, Any],
    trainer_kwargs: dict[str, Any],
    callbacks: list,
    n_folds: int,
    metric: str,
    aggregation: str,
    factory,
    task_type: str,
    discretize_config: dict[str, Any] | None = None,
    target_selection_config: Optional["TargetSelectionConfig"] = None,
) -> float:
    """
    Run k-fold cross-validation for a trial.

    Uses DataSplitter.k_fold_split() from data_splitting.py to create
    folds and trains a fresh model on each fold.

    Args:
        model_name: Model name for factory
        dataset: Dataset to split
        model_params: Model hyperparameters
        optimizer_params: Optimizer hyperparameters
        scheduler_params: Scheduler hyperparameters
        loss_params: Loss function hyperparameters
        trainer_kwargs: Additional trainer configuration
        callbacks: Callbacks to use (including HPO callback)
        n_folds: Number of cross-validation folds
        metric: Metric name to aggregate
        aggregation: Aggregation method ("mean", "median", "min", "max")
        factory: ModelFactory instance
        task_type: Task type string
        discretize_config: Optional config for DiscretizeTargets transform
        target_selection_config: Optional target selection config for node/edge tasks

    Returns:
        Aggregated metric value across all folds

    Raises:
        HPOError: If no valid fold metrics obtained
        ImportError: If DataSplitter or Trainer not available
    """
    from statistics import mean, median

    if DataSplitter is None:
        raise HPOError(
            "DataSplitter not available for cross-validation",
            details="Cannot run CV without data_splitting module",
        )

    if Trainer is None:
        raise HPOError(
            "Trainer not available for cross-validation",
            details="Cannot run CV without Trainer class",
        )

    # Get folds from DataSplitter
    folds = DataSplitter.k_fold_split(
        dataset=dataset,
        n_splits=n_folds,
        random_seed=42,
    )

    fold_metrics = []

    for fold_idx, (train_subset, val_subset) in enumerate(folds):
        logger.debug(f"  Cross-validation fold {fold_idx + 1}/{n_folds}")

        # Apply task-specific data preparation for this fold (including discretization)
        train_subset, val_subset, num_classes_override = _prepare_data_for_task_hpo(
            train_subset, val_subset, task_type, discretize_config, target_selection_config
        )

        # Create fresh model for each fold with metadata
        model, model_info = factory.create_model_with_info(
            name=model_name,
            hyperparameters=model_params,
            task_type=task_type,
            sample_data=train_subset[0] if hasattr(train_subset, "__getitem__") else None,
            num_classes_override=num_classes_override,  # Pass num_classes for classification
        )

        # Import DataLoader
        from torch_geometric.loader import DataLoader as PyGDataLoader

        # Get batch_size from trainer_kwargs or default
        batch_size = trainer_kwargs.get("batch_size", 32)

        # Create DataLoaders from fold subsets
        train_loader = PyGDataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = PyGDataLoader(val_subset, batch_size=batch_size, shuffle=False)
        # -------

        # Create optimizer using registry (handles param filtering automatically)
        optimizer = _create_optimizer_from_registry(model.parameters(), optimizer_params)

        # Create scheduler using registry if params provided
        scheduler = _create_scheduler_from_registry(optimizer, scheduler_params)

        # Create loss function using registry (handles param filtering automatically)
        loss_fn = _create_loss_from_registry(task_type, loss_params)

        # Get max_epochs from trainer_kwargs
        max_epochs = trainer_kwargs.get("max_epochs", 100)

        # Create trainer with all required components
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            max_epochs=max_epochs,
            callbacks=callbacks.copy(),
            model_info=model_info,
        )

        # Train model
        results = trainer.fit()
        # -------

        # Extract metric value
        fold_value = results.get(metric, results.get("best_val_loss"))
        if fold_value is not None:
            fold_metrics.append(fold_value)
            logger.debug(f"    Fold {fold_idx + 1} {metric}: {fold_value:.6f}")

    # Aggregate fold metrics
    if not fold_metrics:
        raise HPOError(
            f"No valid fold metrics obtained for '{metric}'",
            details=f"All {n_folds} folds failed to produce the metric",
        )

    if aggregation == "mean":
        result = mean(fold_metrics)
    elif aggregation == "median":
        result = median(fold_metrics)
    elif aggregation == "min":
        result = min(fold_metrics)
    elif aggregation == "max":
        result = max(fold_metrics)
    else:
        result = mean(fold_metrics)

    logger.debug(f"  CV {aggregation} {metric}: {result:.6f}")
    return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def is_hpo_enabled(config: HPOConfig | None = None) -> bool:
    """
    Check if HPO is enabled in configuration.

    Utility function to quickly check HPO status without
    creating an HPOManager instance.

    Args:
        config: HPOConfig instance to check (None returns False)

    Returns:
        True if HPO is enabled, False otherwise

    Examples:
        >>> config = HPOConfig(enabled=True)
        >>> is_hpo_enabled(config)
        True

        >>> is_hpo_enabled(None)
        False
    """
    if config is None:
        return False
    return config.enabled


def get_best_params(manager: HPOManager) -> dict[str, Any]:
    """
    Get best parameters from completed HPO.

    Convenience function to extract best parameters from
    a completed HPOManager optimization.

    Args:
        manager: HPOManager instance after optimize()

    Returns:
        Dict of best hyperparameters

    Raises:
        HPOError: If no optimization completed yet

    Examples:
        >>> manager.optimize(model_name="GCN", dataset=dataset)
        >>> best = get_best_params(manager)
        >>> print(best)
        {'hidden_channels': 128, 'lr': 0.001}
    """
    if manager.best_params is None:
        raise HPOError(
            "No optimization completed yet", details="Run optimize() before getting best parameters"
        )
    return manager.best_params


def create_hpo_manager(
    enabled: bool = True, n_trials: int = 100, backend: str = "optuna", **kwargs
) -> HPOManager:
    """
    Convenience function to create HPOManager.

    Creates an HPOManager with common default settings,
    allowing quick setup for typical use cases.

    Args:
        enabled: Enable HPO (default: True)
        n_trials: Number of trials to run (default: 100)
        backend: HPO backend to use (default: "optuna")
        **kwargs: Additional HPOConfig parameters

    Returns:
        HPOManager instance ready for optimization

    Examples:
        >>> # Quick setup with defaults
        >>> manager = create_hpo_manager()

        >>> # Custom configuration
        >>> manager = create_hpo_manager(
        ...     n_trials=200,
        ...     timeout=3600,
        ...     cv_folds=5
        ... )
    """
    config = HPOConfig(enabled=enabled, n_trials=n_trials, backend=backend, **kwargs)
    return HPOManager(config)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main class
    "HPOManager",
    # Convenience functions
    "is_hpo_enabled",
    "get_best_params",
    "create_hpo_manager",
    "infer_task_type",
    # Helper functions (for advanced use)
    "_flatten_params",
    "_extract_param_categories",
    "_run_cross_validation",
]
